import {InternalPage, LeafPage, Page, PageStore} from './page-store.js';
import {SnapshotClosedError, WriteConflictError} from './errors.js';

export interface RangeEntry<V> {
  key: string;
  value: V;
}

export interface BTreeOptions {
  /**
   * 每页最多放多少个键（叶页）。内部页最多 capacity + 1 个孩子。
   * 默认 32；测试里调小可以人为制造频繁分裂与合并。
   */
  capacity?: number;
  /**
   * 快照默认超时（毫秒）。超时后快照自动关闭，之后再读抛 SnapshotClosedError。
   * 默认不超时，靠 oldestSnapshots() 排查泄漏。
   */
  snapshotTtlMs?: number;
}

export interface SnapshotOpenOptions {
  /** 覆盖 BTree 级别的 snapshotTtlMs；显式传 0 表示这个快照不超时 */
  ttlMs?: number;
}

export interface SnapshotInfo {
  id: number;
  version: number;
  openedAt: number;
  ageMs: number;
  ttlMs: number | undefined;
  /**
   * 如果关掉这个快照，理论上可以立刻回收的页数
   * （它的可达页里、当前因回收水位线而被留住的死页数）。
   * 计算要遍历该快照的整棵树，诊断接口，别在热路径调。
   */
  pinnedPages: number;
}

interface HistoryEntry {
  version: number;
  keys: Set<string>;
}

interface SnapshotRecord {
  id: number;
  version: number;
  rootId: number;
  openedAt: number;
  ttlMs: number | undefined;
  /** 超时计时器；undefined 表示无 TTL，null 表示已触发 */
  timer: ReturnType<typeof setTimeout> | undefined | null;
}

interface PendingSplit {
  promoted: string;
  right: number;
}

const TOMBSTONE = Symbol('tombstone');
type OverlayValue<V> = V | typeof TOMBSTONE;

/** 上界二分：返回第一个 >= key 的位置 */
function lowerBound(keys: readonly string[], key: string): number {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (keys[mid] < key) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * 上界二分：返回第一个 > key 的位置。
 * 内页下降用：分隔键 s 的右侧子树含 s 自身（标准 B 树），
 * 故查找 k 时走最后一个 <= k 的分隔键右边，即 children[upperBound(keys, k)]。
 */
function upperBound(keys: readonly string[], key: string): number {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (keys[mid] <= key) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function emptyLeaf(): LeafPage {
  return {type: 'leaf', keys: [], values: []};
}

function clonePage(page: Page): Page {
  if (page.type === 'leaf') {
    return {type: 'leaf', keys: page.keys.slice(), values: page.values.slice()};
  }
  return {type: 'internal', keys: page.keys.slice(), children: page.children.slice()};
}

/** 从 rootId 开始 DFS，收集全部可达页 id */
function reachablePages(store: PageStore, rootId: number): Set<number> {
  const seen = new Set<number>();
  const stack = [rootId];
  while (stack.length) {
    const id = stack.pop()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const page = store.get(id);
    if (page.type === 'internal') stack.push(...page.children);
  }
  return seen;
}

export class Snapshot<V = unknown> {
  #tree: BTree<V>;
  #record: SnapshotRecord;
  #closed = false;

  /** @internal */
  constructor(tree: BTree<V>, record: SnapshotRecord) {
    this.#tree = tree;
    this.#record = record;
  }

  get id(): number {
    return this.#record.id;
  }

  /** 这个快照看到的是第几次提交（0 表示空树） */
  get version(): number {
    return this.#record.version;
  }

  get closed(): boolean {
    return this.#closed;
  }

  #assertOpen(): void {
    if (this.#closed) {
      throw new SnapshotClosedError(this.#record.id, this.#record.timer === null);
    }
  }

  get(key: string): V | undefined {
    this.#assertOpen();
    return readGet(this.#tree.store, this.#record.rootId, key) as V | undefined;
  }

  /**
   * 异步范围扫描，[start, end] 双闭。
   * 迭代器持有这个快照，中途无论 await 多久、发生多少次提交/分裂/合并，
   * 结果都与同快照的 get 一致。
   */
  range(start: string, end: string): AsyncIterableIterator<RangeEntry<V>> {
    this.#assertOpen();
    return traverseRange(this.#tree.store, this.#record.rootId, start, end, () =>
      this.#assertOpen(),
    );
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#tree.closeSnapshotInternal(this.#record);
  }

  /** @internal 供 BTree 在 TTL 超时时标记 */
  markClosedByTimeout(): void {
    this.#closed = true;
  }
}

export class WriteTransaction<V = unknown> {
  #tree: BTree<V>;
  #baseVersion: number;
  #rootId: number;
  #overlay = new Map<string, OverlayValue<V>>();
  #delta = 0;
  /** 本事务克隆/分裂出来的新页（尚未提交） */
  #privatePages = new Set<number>();
  /** 本事务内被新私页取代、提交时要释放的旧私页（事务内先保留不复用） */
  #discarded = new Set<number>();
  /** 本事务克隆过的、需要在提交时失效的旧页 id */
  #oldPages = new Set<number>();
  /** 分裂记录是事务私有的：递归插入中临时挂在（私有）页 id 上 */
  #pendingSplits = new Map<number, PendingSplit>();
  #finished = false;

  /** @internal */
  constructor(tree: BTree<V>, baseVersion: number, rootId: number) {
    this.#tree = tree;
    this.#baseVersion = baseVersion;
    this.#rootId = rootId;
    this.#tree.registerTxnBase(baseVersion);
  }

  get baseVersion(): number {
    return this.#baseVersion;
  }

  get finished(): boolean {
    return this.#finished;
  }

  #assertActive(): void {
    if (this.#finished) throw new Error('transaction has already committed or rolled back');
  }

  /** 事务内读到的是“工作根 + 本事务改动” */
  get(key: string): V | undefined {
    this.#assertActive();
    let value = readGet(this.#tree.store, this.#rootId, key) as V | undefined;
    const overlay = this.#overlay.get(key);
    if (overlay !== undefined) {
      value = overlay === TOMBSTONE ? undefined : (overlay as V);
    }
    return value;
  }

  insert(key: string, value: V): void {
    this.#assertActive();
    this.#overlay.set(key, value);
    this.#mutateRoot();
    this.#insertBelowRoot(key, value);
    this.#splitRootIfNeeded();
  }

  delete(key: string): void {
    this.#assertActive();
    this.#overlay.set(key, TOMBSTONE);
    this.#mutateRoot();
    if (this.#deleteBelowRoot(key)) {
      this.#delta--;
      this.#collapseRootIfNeeded();
    }
  }

  // ---------- 写时复制 ----------

  /** 确保 id 是本事务的私有可写页，返回（可能是新的）页 id */
  #writable(id: number): number {
    if (this.#privatePages.has(id)) return id;
    const copy = clonePage(this.#tree.store.get(id));
    const newId = this.#tree.store.allocatePrivate(copy);
    this.#privatePages.add(newId);
    this.#oldPages.add(id);
    return newId;
  }

  #mutateRoot(): void {
    if (!this.#privatePages.has(this.#rootId)) {
      this.#rootId = this.#writable(this.#rootId);
    }
  }

  // ---------- 插入 / 分裂 ----------

  #insertBelowRoot(key: string, value: V): void {
    const root = this.#tree.store.get(this.#rootId);
    if (root.type === 'leaf') {
      this.#leafInsert(this.#rootId, root, key, value);
      return;
    }
    const at = upperBound(root.keys, key);
    const child = this.#writable(root.children[at]);
    root.children[at] = child;
    this.#insertRec(child, key, value);
    const split = this.#takePendingSplit(child);
    if (split) {
      root.keys.splice(at, 0, split.promoted);
      root.children.splice(at + 1, 0, split.right);
      // 根内页也可能因此溢出
      if (root.keys.length > this.#tree.capacity) this.#splitInternal(this.#rootId, root);
    }
  }

  #insertRec(id: number, key: string, value: V): void {
    const page = this.#tree.store.get(id);
    if (page.type === 'leaf') {
      this.#leafInsert(id, page, key, value);
      return;
    }
    const at = upperBound(page.keys, key);
    const child = this.#writable(page.children[at]);
    page.children[at] = child;
    this.#insertRec(child, key, value);
    const split = this.#takePendingSplit(child);
    if (split) {
      page.keys.splice(at, 0, split.promoted);
      page.children.splice(at + 1, 0, split.right);
      if (page.keys.length > this.#tree.capacity) this.#splitInternal(id, page);
    }
  }

  #leafInsert(id: number, page: LeafPage, key: string, value: V): void {
    const at = lowerBound(page.keys, key);
    if (at < page.keys.length && page.keys[at] === key) {
      page.values[at] = value; // 已存在 => 更新，不计 delta
    } else {
      page.keys.splice(at, 0, key);
      page.values.splice(at, 0, value);
      this.#delta++;
    }
    if (page.keys.length > this.#tree.capacity) this.#splitLeaf(id, page);
  }

  #splitLeaf(id: number, page: LeafPage): void {
    const mid = page.keys.length >> 1;
    const promoted = page.keys[mid]; // 标准 B 树：提升键移到右叶
    const right = emptyLeaf();
    right.keys = page.keys.splice(mid);
    right.values = page.values.splice(mid);
    const rightId = this.#tree.store.allocatePrivate(right);
    this.#privatePages.add(rightId);
    this.#pendingSplits.set(id, {promoted, right: rightId});
  }

  #splitInternal(id: number, page: InternalPage): void {
    const mid = page.keys.length >> 1;
    const promoted = page.keys[mid]; // 内页：分隔键提升到父页，从本页移除
    // 左半保留 keys[0..mid-1] 与 children[0..mid]；
    // 右半拿 keys[mid+1..] 与 children[mid+1..]
    const right: InternalPage = {
      type: 'internal',
      keys: page.keys.splice(mid + 1),
      children: page.children.splice(mid + 1),
    };
    page.keys.splice(mid, 1);
    const rightId = this.#tree.store.allocatePrivate(right);
    this.#privatePages.add(rightId);
    this.#pendingSplits.set(id, {promoted, right: rightId});
  }

  #splitRootIfNeeded(): void {
    const split = this.#takePendingSplit(this.#rootId);
    if (!split) return;
    const root = this.#tree.store.get(this.#rootId);
    if (root.type === 'leaf') {
      // #splitLeaf 已原地把旧根改成左叶，提升键及之后在 split.right
      const left: LeafPage = {
        type: 'leaf',
        keys: root.keys.slice(),
        values: root.values.slice(),
      };
      const leftId = this.#tree.store.allocatePrivate(left);
      this.#privatePages.add(leftId);
      const newRoot: InternalPage = {
        type: 'internal',
        keys: [split.promoted],
        children: [leftId, split.right],
      };
      this.#rootId = this.#tree.store.allocatePrivate(newRoot);
    } else {
      // 旧根是内页。#splitInternal 的语义是：它已经原地把旧根改成了
      // 左半（keys[0..mid-1] + children[0..mid]），右半在 split.right，
      // 提升键 split.promoted。这里只需克隆“已经是左半”的旧根。
      const left: InternalPage = {
        type: 'internal',
        keys: root.keys.slice(),
        children: root.children.slice(),
      };
      const leftId = this.#tree.store.allocatePrivate(left);
      this.#privatePages.add(leftId);
      const newRoot: InternalPage = {
        type: 'internal',
        keys: [split.promoted],
        children: [leftId, split.right],
      };
      this.#rootId = this.#tree.store.allocatePrivate(newRoot);
    }
    this.#privatePages.add(this.#rootId);
  }

  // ---------- 删除 / 借位 / 合并 ----------

  #deleteBelowRoot(key: string): boolean {
    const root = this.#tree.store.get(this.#rootId);
    if (root.type === 'leaf') {
      const at = lowerBound(root.keys, key);
      if (at >= root.keys.length || root.keys[at] !== key) return false;
      root.keys.splice(at, 1);
      root.values.splice(at, 1);
      return true;
    }
    const at = upperBound(root.keys, key);
    const child = this.#writable(root.children[at]);
    root.children[at] = child;
    if (!this.#deleteRec(child, key)) return false;
    this.#fixUnderflow(root, at);
    return true;
  }

  #deleteRec(id: number, key: string): boolean {
    const page = this.#tree.store.get(id);
    if (page.type === 'leaf') {
      const at = lowerBound(page.keys, key);
      if (at >= page.keys.length || page.keys[at] !== key) return false;
      page.keys.splice(at, 1);
      page.values.splice(at, 1);
      return true;
    }
    const at = upperBound(page.keys, key);
    const child = this.#writable(page.children[at]);
    page.children[at] = child;
    if (!this.#deleteRec(child, key)) return false;
    // 本层修孩子；若合并让 page 自己也少到欠填充，返回上层后
    // 上层的 fixUnderflow 会把 page 当作“那个孩子”继续处理——天然向上传播。
    this.#fixUnderflow(page, at);
    return true;
  }

  #fixUnderflow(parent: InternalPage, index: number): void {
    const childId = parent.children[index];
    const child = this.#tree.store.get(childId);
    if (this.#count(child) >= this.#minCount(child)) return;

    // 1) 左兄弟够胖则向左借
    if (index > 0) {
      const left = this.#tree.store.get(parent.children[index - 1]);
      if (this.#count(left) > this.#minCount(left)) {
        this.#borrowFromLeft(parent, index);
        return;
      }
    }
    // 2) 右兄弟够胖则向右借
    if (index + 1 < parent.children.length) {
      const right = this.#tree.store.get(parent.children[index + 1]);
      if (this.#count(right) > this.#minCount(right)) {
        this.#borrowFromRight(parent, index);
        return;
      }
    }
    // 3) 两边都处于最瘦状态：合并。优先并左（消除槽 index-1，
    //    调用方持有的槽 index 仍指向幸存孩子）；在最左端则并右。
    if (index > 0) this.#mergeWithLeft(parent, index);
    else this.#mergeWithRight(parent);
  }

  #count(page: Page): number {
    return page.type === 'leaf' ? page.keys.length : page.keys.length + 1;
  }

  #minCount(page: Page): number {
    return page.type === 'leaf' ? this.#tree.minLeafKeys : this.#tree.minInternalKeys + 1;
  }

  #borrowFromLeft(parent: InternalPage, index: number): void {
    const leftId = this.#writable(parent.children[index - 1]);
    const childId = this.#writable(parent.children[index]);
    parent.children[index - 1] = leftId;
    parent.children[index] = childId;
    const left = this.#tree.store.get(leftId);
    const child = this.#tree.store.get(childId);
    // 同一父页下的兄弟页类型必然一致
    if (child.type === 'leaf') {
      const leftLeaf = left as LeafPage;
      const moved = leftLeaf.keys.pop()!;
      const movedVal = leftLeaf.values.pop()!;
      child.keys.unshift(moved);
      child.values.unshift(movedVal);
      // 分隔符 = 被移走的那个键（右叶新的最小键 = 左叶上界）
      parent.keys[index - 1] = moved;
    } else {
      const leftInternal = left as InternalPage;
      // 内页向左借位：左兄弟只移过来一个孩子；
      // 旧分隔符下移到孩子头部，左兄弟的末键上提为新分隔符
      child.children.unshift(leftInternal.children.pop()!);
      child.keys.unshift(parent.keys[index - 1]);
      parent.keys[index - 1] = leftInternal.keys.pop()!;
    }
  }

  #borrowFromRight(parent: InternalPage, index: number): void {
    const childId = this.#writable(parent.children[index]);
    const rightId = this.#writable(parent.children[index + 1]);
    parent.children[index] = childId;
    parent.children[index + 1] = rightId;
    const child = this.#tree.store.get(childId);
    const right = this.#tree.store.get(rightId);

    if (child.type === 'leaf') {
      const rightLeaf = right as LeafPage;
      const moved = rightLeaf.keys.shift()!;
      const movedVal = rightLeaf.values.shift()!;
      child.keys.push(moved);
      child.values.push(movedVal);
      // 分隔符 = 被移走的那个键（孩子新的最大键 = 右叶新下界）
      parent.keys[index] = moved;
    } else {
      const rightInternal = right as InternalPage;
      child.keys.push(parent.keys[index]);
      parent.keys[index] = rightInternal.keys.shift()!;
      child.children.push(rightInternal.children.shift()!);
    }
  }

  /** 把孩子 index 合并进左兄弟 index-1（孩子是更瘦的那个），消除槽 index */
  #mergeWithLeft(parent: InternalPage, index: number): void {
    const survivorId = this.#freshPrivate(parent.children[index - 1]);
    const removedId = this.#freshPrivate(parent.children[index]);
    parent.children[index - 1] = survivorId;
    const survivor = this.#tree.store.get(survivorId);
    const removed = this.#tree.store.get(removedId);
    if (survivor.type === 'leaf') {
      survivor.keys.push(...(removed as LeafPage).keys);
      survivor.values.push(...(removed as LeafPage).values);
    } else {
      survivor.keys.push(parent.keys[index - 1], ...removed.keys);
      survivor.children.push(...(removed as InternalPage).children);
    }
    this.#disposePrivate(removedId);
    parent.keys.splice(index - 1, 1);
    parent.children.splice(index, 1);
  }

  /** 仅用于 index=0：把右兄弟合并进最左孩子，消除槽 1 */
  #mergeWithRight(parent: InternalPage): void {
    const survivorId = this.#freshPrivate(parent.children[0]);
    const removedId = this.#freshPrivate(parent.children[1]);
    parent.children[0] = survivorId;
    const survivor = this.#tree.store.get(survivorId);
    const removed = this.#tree.store.get(removedId);
    if (survivor.type === 'leaf') {
      survivor.keys.push(...(removed as LeafPage).keys);
      survivor.values.push(...(removed as LeafPage).values);
    } else {
      survivor.keys.push(parent.keys[0], ...removed.keys);
      survivor.children.push(...(removed as InternalPage).children);
    }
    this.#disposePrivate(removedId);
    parent.keys.splice(0, 1);
    parent.children.splice(1, 1);
  }

  /**
   * 合并是破坏性改写：幸存者必须得到一份全新的私页，不能直接复用
   * 分裂出来的私页（它可能正挂在父页另一处）。被合并掉的原始页若也是
   * 本事务私页（分裂产物），登记到 #discarded 待提交时统一释放；
   * 克隆自旧树的原始页则交给 #oldPages 在提交时失效。
   * 事务期间不复用任何 id，避免集合里的旧引用指向新内容。
   */
  #freshPrivate(id: number): number {
    const copy = clonePage(this.#tree.store.get(id));
    const newId = this.#tree.store.allocatePrivate(copy);
    this.#privatePages.add(newId);
    if (this.#privatePages.has(id)) {
      this.#privatePages.delete(id);
      this.#discarded.add(id);
    }
    return newId;
  }

  #disposePrivate(id: number): void {
    this.#privatePages.delete(id);
    this.#discarded.add(id);
  }

  #collapseRootIfNeeded(): void {
    for (;;) {
      const root = this.#tree.store.get(this.#rootId);
      if (root.type !== 'internal' || root.keys.length > 0) return;
      this.#rootId = root.children[0];
    }
  }

  #takePendingSplit(pageId: number): PendingSplit | undefined {
    const split = this.#pendingSplits.get(pageId);
    if (split !== undefined) this.#pendingSplits.delete(pageId);
    return split;
  }

  // ---------- 提交 / 回滚 ----------

  commit(): Promise<void> {
    if (this.#finished) {
      return Promise.reject(new Error('transaction has already committed or rolled back'));
    }
    return this.#tree.serializeCommit(() => this.#commitLocked());
  }

  /** 便捷 API 的同步提交路径（单事务同步代码内不可能遇到并发冲突） */
  commitNow(): void {
    if (this.#finished) throw new Error('transaction has already committed or rolled back');
    this.#commitLockedSync();
  }

  #commitLockedSync(): void {
    try {
      // 1) 冲突检测：基版本之后别人提交过的键，与本事务写集合有交集 => 冲突
      if (this.#baseVersion < this.#tree.version) {
        const conflicting = this.#tree.conflictingKeysSince(
          this.#baseVersion,
          this.#overlay.keys(),
        );
        if (conflicting.length > 0) {
          throw new WriteConflictError(this.#baseVersion, this.#tree.version, conflicting);
        }
        // 无交集：没有任何一方的写被吞，本事务自动重放到最新根上
        this.#rebaseOnto(this.#tree.rootIdInternal);
      }

      const version = this.#tree.version + 1;

      // 2) 发布提交后仍可达的私有页；不可达的私有页（如被塌缩掉的私有根）直接丢弃
      const reachable = reachablePages(this.#tree.store, this.#rootId);
      for (const id of this.#privatePages) {
        if (reachable.has(id)) this.#tree.store.publish(id, version);
        else this.#tree.store.freePrivate(id);
      }
      // 事务内被合并/取代掉的旧私页此刻统一释放
      for (const id of this.#discarded) this.#tree.store.freePrivate(id);
      // 3) 被替换掉的旧页（旧树内容）在这个版本整体失效
      for (const id of this.#oldPages) this.#tree.store.kill(id, version);

      // 4) 新根、新版本、新计数一次性生效（先摘除自己的基版本再剪历史）
      this.#finished = true;
      this.#tree.unregisterTxnBase(this.#baseVersion);
      this.#tree.publishRoot(this.#rootId, version, this.#delta, new Set(this.#overlay.keys()));
      this.#tree.reclaimNow();
    } catch (error) {
      this.#abort();
      throw error;
    }
  }

  #commitLocked(): Promise<void> {
    this.#commitLockedSync();
    return Promise.resolve();
  }

  /** 提交路径抛错（含冲突）后的清理：效果等同回滚 */
  #abort(): void {
    if (this.#finished) return;
    for (const id of this.#privatePages) this.#tree.store.freePrivate(id);
    for (const id of this.#discarded) this.#tree.store.freePrivate(id);
    this.#tree.unregisterTxnBase(this.#baseVersion);
    this.#tree.reclaimNow();
    this.#finished = true;
  }

  rollback(): void {
    if (this.#finished) throw new Error('transaction has already committed or rolled back');
    this.#abort();
  }

  #rebaseOnto(newRootId: number): void {
    // 丢掉在旧根上做出的全部私有改动（旧页从未被 kill，继续服务老快照）
    for (const id of this.#privatePages) this.#tree.store.freePrivate(id);
    for (const id of this.#discarded) this.#tree.store.freePrivate(id);
    this.#pendingSplits.clear();
    this.#privatePages = new Set();
    this.#discarded = new Set();
    this.#oldPages = new Set();
    this.#rootId = newRootId;
    this.#delta = 0;

    // 写集里每个键只保留最后一次操作，按最终意图在新根上重放
    for (const [key, value] of this.#overlay) {
      this.#mutateRoot();
      if (value === TOMBSTONE) {
        if (this.#deleteBelowRoot(key)) {
          this.#delta--;
          this.#collapseRootIfNeeded();
        }
      } else {
        this.#insertBelowRoot(key, value as V);
        this.#splitRootIfNeeded();
      }
    }
    // 基版本前移，保证冲突检测历史里包含“别人提交、我们重放”的版本
    this.#tree.unregisterTxnBase(this.#baseVersion);
    this.#baseVersion = this.#tree.version;
    this.#tree.registerTxnBase(this.#baseVersion);
  }
}

export class BTree<V = unknown> {
  readonly capacity: number;
  readonly minLeafKeys: number;
  readonly minInternalKeys: number;
  readonly store = new PageStore();
  readonly #defaultTtlMs: number | undefined;

  #rootId: number;
  #version = 0;
  #entryCount = 0;
  #snapshots = new Set<SnapshotRecord>();
  #txnBases = new Set<number>();
  #history: HistoryEntry[] = [];
  #nextSnapshotId = 1;
  #commitChain: Promise<unknown> = Promise.resolve();

  constructor(options: BTreeOptions = {}) {
    const capacity = options.capacity ?? 32;
    if (capacity < 4) throw new Error('capacity must be >= 4');
    this.capacity = capacity;
    this.minLeafKeys = capacity >> 1;
    // 内部页孩子数下限取 floor(capacity/2)，保证两个最瘦兄弟合并后仍不溢出
    this.minInternalKeys = (capacity >> 1) - 1;
    this.#defaultTtlMs = options.snapshotTtlMs;
    this.#rootId = this.store.allocate(emptyLeaf(), 0);
  }

  /** 当前已提交版本号（第几次提交） */
  get version(): number {
    return this.#version;
  }

  /** @internal */
  get rootIdInternal(): number {
    return this.#rootId;
  }

  // ---------- 快照 ----------

  snapshot(options: SnapshotOpenOptions = {}): Snapshot<V> {
    const ttlMs =
      options.ttlMs === undefined
        ? this.#defaultTtlMs
        : options.ttlMs === 0
          ? undefined
          : options.ttlMs;
    const record: SnapshotRecord = {
      id: this.#nextSnapshotId++,
      version: this.#version,
      rootId: this.#rootId,
      openedAt: Date.now(),
      ttlMs,
      timer: undefined,
    };
    const snap = new Snapshot<V>(this, record);
    if (ttlMs !== undefined) {
      record.timer = setTimeout(() => {
        record.timer = null;
        if (!snap.closed) {
          snap.markClosedByTimeout();
          this.closeSnapshotInternal(record);
        }
      }, ttlMs);
      const timer = record.timer as {unref?: () => void};
      timer.unref?.(); // 库的超时计时器不该拖住 Node 进程退出
    }
    this.#snapshots.add(record);
    return snap;
  }

  /** @internal */
  closeSnapshotInternal(record: SnapshotRecord): void {
    if (record.timer !== undefined && record.timer !== null) clearTimeout(record.timer);
    this.#snapshots.delete(record);
    this.reclaimNow();
  }

  /**
   * 列存活最久的快照（按打开时间升序），带每个快照拖住的页数。
   * pinnedPages 要遍历该快照的整棵树，仅用于排障，别在热路径上循环调。
   */
  oldestSnapshots(limit = 10): SnapshotInfo[] {
    const records = [...this.#snapshots].sort((a, b) => a.openedAt - b.openedAt).slice(0, limit);
    const now = Date.now();
    return records.map((record) => {
      const others = new Set(this.#snapshots);
      others.delete(record);
      const floorWithout = this.#retentionFloor(others, this.#txnBases);
      let pinnedPages = 0;
      for (const id of reachablePages(this.store, record.rootId)) {
        const dead = this.store.deadAt(id);
        if (Number.isFinite(dead) && dead > floorWithout) pinnedPages++;
      }
      return {
        id: record.id,
        version: record.version,
        openedAt: record.openedAt,
        ageMs: now - record.openedAt,
        ttlMs: record.ttlMs,
        pinnedPages,
      };
    });
  }

  // ---------- 写事务 ----------

  begin(): WriteTransaction<V> {
    return new WriteTransaction<V>(this, this.#version, this.#rootId);
  }

  /**
   * 便捷写法：一次插入/更新，内部自己开事务、同步提交。
   * 需要跨多次改动做原子批次时请用 begin()。
   */
  insert(key: string, value: V): void {
    const txn = this.begin();
    txn.insert(key, value);
    txn.commitNow();
  }

  /** 便捷写法：一次删除，不存在的键不报错 */
  delete(key: string): void {
    const txn = this.begin();
    txn.delete(key);
    txn.commitNow();
  }

  // ---------- 读便捷写法 ----------

  /** 读最新已提交数据，无需开快照 */
  get(key: string): V | undefined {
    return readGet(this.store, this.#rootId, key) as V | undefined;
  }

  /**
   * 便捷范围扫描，[start, end] 双闭，返回快照数组。
   * 读取发生在调用这一刻的最新已提交版本上，天然不会看到半批次。
   * 需要异步边迭代边 await 地扫描，请显式 snapshot() 后用 snap.range()。
   */
  range(start: string, end: string): RangeEntry<V>[] {
    const out: RangeEntry<V>[] = [];
    collectRange(this.store, this.#rootId, start, end, out);
    return out;
  }

  size(): number {
    return this.#entryCount;
  }

  /** 当前活着的页数，内存监控用 */
  livePages(): number {
    return this.store.livePages();
  }

  // ---------- 供 WriteTransaction 使用的内部接口 ----------

  /** @internal */
  registerTxnBase(version: number): void {
    this.#txnBases.add(version);
  }

  /** @internal */
  unregisterTxnBase(version: number): void {
    this.#txnBases.delete(version);
  }

  /**
   * 提交串行化：写事务的 await 可以交错，但提交临界区按顺序执行。
   */
  /** @internal */
  serializeCommit<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.#commitChain.then(fn, fn);
    this.#commitChain = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  /** @internal */
  conflictingKeysSince(baseVersion: number, keys: IterableIterator<string>): string[] {
    const writeSet = new Set(keys);
    const conflicts: string[] = [];
    for (const entry of this.#history) {
      if (entry.version <= baseVersion) continue;
      for (const key of writeSet) {
        if (entry.keys.has(key)) conflicts.push(key);
      }
    }
    return [...new Set(conflicts)];
  }

  /** @internal */
  publishRoot(rootId: number, version: number, delta: number, writeSet: Set<string>): void {
    this.#rootId = rootId;
    this.#version = version;
    this.#entryCount += delta;
    this.#history.push({version, keys: writeSet});
    this.#pruneHistory();
  }

  #pruneHistory(): void {
    if (this.#txnBases.size === 0) {
      this.#history = [];
      return;
    }
    let oldestBase = Infinity;
    for (const base of this.#txnBases) oldestBase = Math.min(oldestBase, base);
    // 基版本为 b 的事务只需要 version > b 的记录
    while (this.#history.length > 0 && this.#history[0].version <= oldestBase) {
      this.#history.shift();
    }
  }

  /**
   * 回收水位线：所有存活快照版本与在途事务基版本的最小值。
   * 页 p 在版本 v 可达当且仅当 born(p) <= v < dead(p)，
   * 所以 dead <= floor 的页可以安全释放；没有任何人持旧时水位线为无穷大。
   */
  #retentionFloor(
    snapshots: Set<SnapshotRecord> = this.#snapshots,
    txnBases: Set<number> = this.#txnBases,
  ): number {
    let floor = Infinity;
    for (const record of snapshots) floor = Math.min(floor, record.version);
    for (const base of txnBases) floor = Math.min(floor, base);
    return floor;
  }

  /** @internal */
  reclaimNow(): void {
    this.store.reclaim(this.#retentionFloor());
  }

  /**
   * 结构自检：遍历整棵树检查排序、分隔键、页数填充率、条目计数。测试用。
   */
  validateStructure(): {pages: number; entries: number} {
    const validate = (id: number, low: string | null, high: string | null, isRoot: boolean): number => {
      const page = this.store.get(id);
      for (let i = 0; i < page.keys.length; i++) {
        const key = page.keys[i];
        if (i > 0 && page.keys[i - 1] > key) throw new Error(`unsorted keys at page ${id}`);
        if (low !== null && key < low) throw new Error(`key ${key} violates lower bound ${low}`);
        // 内部搜索语义：x >= keys[i] 走右孩子，故每页键区间都是 [low, high)
        if (high !== null && key >= high) {
          throw new Error(`key ${key} violates upper bound ${high} in ${page.type}`);
        }
      }
      if (page.type === 'leaf') {
        if (page.keys.length > this.capacity) throw new Error('leaf overflow');
        if (new Set(page.keys).size !== page.keys.length) throw new Error('duplicate leaf keys');
        return page.keys.length;
      }
      if (page.children.length !== page.keys.length + 1) throw new Error('internal arity mismatch');
      if (!isRoot && page.keys.length < this.minInternalKeys) {
        throw new Error(`internal underflow: ${page.keys.length} < ${this.minInternalKeys}`);
      }
      // 分隔键必须严格落在左右孩子的分界位置上
      for (let i = 0; i < page.keys.length; i++) {
        const sep = page.keys[i];
        if (sep < (low ?? '') && low !== null) throw new Error('separator below low');
        if (high !== null && sep >= high) throw new Error('separator above high');
      }
      let entries = 0;
      for (let i = 0; i < page.children.length; i++) {
        const childLow = i === 0 ? low : page.keys[i - 1];
        const childHigh = i === page.keys.length ? high : page.keys[i];
        entries += validate(page.children[i], childLow, childHigh, false);
      }
      return entries;
    };
    const entries = validate(this.#rootId, null, null, true);
    if (entries !== this.#entryCount) {
      throw new Error(`entry count ${entries} != tracked ${this.#entryCount}`);
    }
    return {pages: reachablePages(this.store, this.#rootId).size, entries};
  }
}

function readGet(store: PageStore, rootId: number, key: string): unknown {
  let id = rootId;
  for (;;) {
    const page = store.get(id);
    if (page.type === 'leaf') {
      const at = lowerBound(page.keys, key);
      if (at < page.keys.length && page.keys[at] === key) return page.values[at];
      return undefined;
    }
    id = page.children[upperBound(page.keys, key)];
  }
}

function collectRange<V>(
  store: PageStore,
  rootId: number,
  start: string,
  end: string,
  out: RangeEntry<V>[],
): void {
  const page = store.get(rootId);
  if (page.type === 'leaf') {
    let i = lowerBound(page.keys, start);
    while (i < page.keys.length && page.keys[i] <= end) {
      out.push({key: page.keys[i], value: page.values[i] as V});
      i++;
    }
    return;
  }
  const first = upperBound(page.keys, start);
  for (let i = first; i < page.children.length; i++) {
    if (i < page.keys.length && page.keys[i] > end) break;
    collectRange(store, page.children[i], start, end, out);
  }
}

/**
 * 显式栈 DFS。不依赖叶页间横向指针——分裂/合并会换页，
 * 始终靠页 id 经父页往下走，而快照把这些页全部钉住。
 */
async function* traverseRange<V>(
  store: PageStore,
  rootId: number,
  start: string,
  end: string,
  assertAlive: () => void,
): AsyncIterableIterator<RangeEntry<V>> {
  interface Frame {
    id: number;
    nextChild: number;
  }
  const root = store.get(rootId);
  const frames: Frame[] = [
    {id: rootId, nextChild: root.type === 'internal' ? upperBound(root.keys, start) : 0},
  ];

  while (frames.length > 0) {
    const frame = frames[frames.length - 1];
    const page = store.get(frame.id);

    if (page.type === 'leaf') {
      frames.pop();
      let i = lowerBound(page.keys, start);
      while (i < page.keys.length && page.keys[i] <= end) {
        assertAlive();
        yield {key: page.keys[i], value: page.values[i] as V};
        i++;
      }
      continue;
    }

    if (frame.nextChild >= page.children.length) {
      frames.pop();
      continue;
    }
    const i = frame.nextChild++;
    // child i+1 起的子树全部 >= keys[i]；它已超出 end，本帧结束
    if (i < page.keys.length && page.keys[i] > end) {
      frames.pop();
      continue;
    }
    const childId = page.children[i];
    const child = store.get(childId);
    frames.push({
      id: childId,
      nextChild: child.type === 'internal' ? upperBound(child.keys, start) : 0,
    });
  }
}
