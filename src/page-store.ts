/**
 * PageStore —— B-tree 所有页的唯一分配/回收入口。
 *
 * 每个页槽有自己的生命周期版本：
 *   born: 该内容第一次随某个版本提交（初始根为 0；写事务中新建的私有页为 -1）
 *   dead: 该内容最后一次可达的版本之后被替换的版本；Infinity 表示当前仍可达
 *
 * 快照 s 能读到页 p 当且仅当 born(p) <= s.version < dead(p)。
 * 回收时只释放 dead <= floor 的页，floor 由所有存活快照与在途写事务共同决定。
 *
 * 槽位被释放后进入 free list 复用，因此 livePages 反映真实内存占用，
 * 而不是历史上累计分配过的页数。
 */

export interface LeafPage {
  type: 'leaf';
  keys: string[];
  values: unknown[];
}

export interface InternalPage {
  type: 'internal';
  /** 分隔键，升序；children.length === keys.length + 1 */
  keys: string[];
  children: number[];
}

export type Page = LeafPage | InternalPage;

/** 写事务中已分配但尚未提交的页，born 取该占位值 */
const UNCOMMITTED = -1;

export class PageStore {
  #pages: (Page | undefined)[] = [];
  #born: number[] = [];
  #dead: number[] = [];
  #freeList: number[] = [];
  #live = 0;

  /**
   * 死页按 dead 版本分桶，版本号随提交严格递增，所以按插入顺序处理即可。
   * 只在 floor 推进时扫描；floor 不动时没有任何扫描成本。
   */
  #graveyard: Map<number, number[]> = new Map();
  #lastFloor = 0;

  /** 分配一个内容为 page 的新页槽，返回页 id */
  allocate(page: Page, born: number): number {
    this.#live++;
    const reused = this.#freeList.pop();
    if (reused === undefined) {
      const id = this.#pages.length;
      this.#pages.push(page);
      this.#born.push(born);
      this.#dead.push(Infinity);
      return id;
    }
    this.#pages[reused] = page;
    this.#born[reused] = born;
    this.#dead[reused] = Infinity;
    return reused;
  }

  /** 事务内分配私有页 */
  allocatePrivate(page: Page): number {
    return this.allocate(page, UNCOMMITTED);
  }

  isPrivate(id: number): boolean {
    return this.#born[id] === UNCOMMITTED;
  }

  get(id: number): Page {
    const p = this.#pages[id];
    if (p === undefined) {
      throw new Error(`page ${id} is not allocated`);
    }
    return p;
  }

  bornAt(id: number): number {
    return this.#born[id];
  }

  deadAt(id: number): number {
    return this.#dead[id];
  }

  /** 私有页确认随提交发布 */
  publish(id: number, version: number): void {
    this.#born[id] = version;
  }

  /** 提交后旧版本页失效；重复 kill 同一个页是安全的 */
  kill(id: number, version: number): void {
    if (this.#dead[id] !== Infinity) return;
    this.#dead[id] = version;
    let bucket = this.#graveyard.get(version);
    if (!bucket) {
      bucket = [];
      this.#graveyard.set(version, bucket);
    }
    bucket.push(id);
  }

  /** 释放从未提交过的私有页（回滚 / 提交时不可达） */
  freePrivate(id: number): void {
    if (this.#born[id] !== UNCOMMITTED) {
      throw new Error(`page ${id} was already published and cannot be freed as private`);
    }
    this.#release(id);
  }

  #release(id: number): void {
    this.#pages[id] = undefined;
    this.#born[id] = 0;
    this.#dead[id] = Infinity;
    this.#freeList.push(id);
    this.#live--;
  }

  /**
   * 回收所有 dead <= floor 的页。
   * floor 不推进时立即返回，因此十万次提交也不会反复扫历史。
   * Infinity 表示当前没有任何读者/在途事务，应清空全部死页，但不更新水位线记录。
   */
  reclaim(floor: number): void {
    if (floor !== Infinity && floor <= this.#lastFloor) return;
    if (floor !== Infinity) this.#lastFloor = floor;
    for (const [version, ids] of this.#graveyard) {
      if (version > floor) break; // Map 按插入序，版本递增
      for (const id of ids) {
        if (this.#dead[id] === version) this.#release(id);
      }
      this.#graveyard.delete(version);
    }
  }

  /** 当前活着的页数（内存监控指标） */
  livePages(): number {
    return this.#live;
  }

  /** 历史上累计使用过的槽位数（含已复用的），诊断用 */
  allocatedSlots(): number {
    return this.#pages.length;
  }
}
