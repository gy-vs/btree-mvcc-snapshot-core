/**
 * In-memory paged B+ tree with MVCC read-only snapshots.
 *
 * Storage model
 * -------------
 * The tree is built from immutable-once-published pages (leaf / internal).
 * A write transaction never edits a published page in place: it clones every
 * page on the path it touches (copy-on-write), redirects the parent's child
 * pointer to the clone, and finally publishes a new root page id together
 * with a bumped generation counter in one synchronous, atomic step. Readers
 * that already hold a snapshot keep pointing at the old root and therefore
 * keep seeing the tree exactly as it was when the snapshot was created;
 * readers that arrive after the commit see the new root.
 *
 * Reclamation
 * -----------
 * Pages replaced by copy-on-write during the commit that published
 * generation G are recorded in a retired-list keyed by G. Such a page is
 * unreachable from every root of generation >= G, so it can be freed as soon
 * as no open snapshot has a generation < G. Freeing is a delete from the
 * page table guarded against double-free, so reference counts can never go
 * negative. Rollback simply drops the transaction's private pages; published
 * pages are untouched.
 *
 * Snapshot leaks
 * --------------
 * Snapshots are explicit handles and must be closed. To keep a leaked
 * snapshot from pinning pages forever, an optional `snapshotTimeoutMs`
 * expires handles: reads on an expired snapshot throw `SnapshotExpiredError`
 * and its pin is released so reclamation can proceed. `diagnostics()`
 * always reports open snapshots (id, generation, age, expired flag) and the
 * number of retired pages waiting on them, so leaks are observable even when
 * no timeout is configured.
 */

export interface BTreeOptions {
  /** Maximum number of keys per page (leaf entries / internal separators). Default 32. */
  maxKeys?: number;
  /**
   * Milliseconds a snapshot may stay open before it is treated as leaked and
   * expired. Expiry releases its pin on old generations. Default: Infinity
   * (snapshots never expire; leaks are only visible via `diagnostics()`).
   */
  snapshotTimeoutMs?: number;
  /** Clock used for snapshot ageing. Defaults to Date.now; inject for tests. */
  now?: () => number;
}

export class WriteConflictError extends Error {
  constructor(message = 'write transaction conflict: a concurrent transaction committed first') {
    super(message);
    this.name = 'WriteConflictError';
  }
}

export class TransactionStateError extends Error {
  constructor(message = 'transaction is already finished') {
    super(message);
    this.name = 'TransactionStateError';
  }
}

export class SnapshotClosedError extends Error {
  constructor(message = 'snapshot is closed') {
    super(message);
    this.name = 'SnapshotClosedError';
  }
}

export class SnapshotExpiredError extends Error {
  constructor(message = 'snapshot exceeded its timeout and was expired') {
    super(message);
    this.name = 'SnapshotExpiredError';
  }
}

type PageId = number;

interface LeafPage<V> {
  id: PageId;
  kind: 'leaf';
  /** Generation whose commit created this page. */
  gen: number;
  /** Entry count of the whole tree. Only maintained on the root page. */
  count: number;
  keys: string[];
  values: V[];
}

interface InternalPage {
  id: PageId;
  kind: 'internal';
  gen: number;
  count: number;
  /** Separator keys; children.length === keys.length + 1. */
  keys: string[];
  children: PageId[];
}

type Page<V> = LeafPage<V> | InternalPage;

interface SnapshotHandle {
  id: number;
  generation: number;
  rootId: PageId;
  createdAt: number;
  closed: boolean;
}

export interface BTreeDiagnostics {
  tree: { generation: number; rootId: PageId; height: number; size: number };
  pages: {
    live: number;
    created: number;
    freed: number;
    discarded: number;
    /** Retired pages kept alive only because open snapshots may still reach them. */
    retiredPending: number;
  };
  snapshots: {
    open: number;
    oldestGeneration: number | null;
    list: { id: number; generation: number; ageMs: number; expired: boolean }[];
  };
  snapshotTimeoutMs: number;
}

/** First index i with keys[i] >= key. */
function lowerBound(keys: string[], key: string): number {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (keys[mid] < key) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Index of the child subtree that may contain key: count of separators <= key. */
function childIndex(keys: string[], key: string): number {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (key < keys[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

export class BTree<V> {
  readonly maxKeys: number;
  readonly minLeafKeys: number;
  readonly minInternalKeys: number;
  readonly snapshotTimeoutMs: number;
  private readonly nowFn: () => number;

  /** Page table: every page reachable from any live generation, plus retired pages awaiting reclamation. */
  private readonly pages = new Map<PageId, Page<V>>();
  private nextPageId = 1;
  private rootId: PageId;
  private generation = 0;

  /** generation -> pages that became unreachable from all roots >= that generation. */
  private readonly retired = new Map<number, PageId[]>();
  private readonly snapshots = new Map<number, SnapshotHandle>();
  private nextSnapshotId = 1;

  // Page accounting. Invariant: created === live + freed + discarded.
  private createdPages = 0;
  private freedPages = 0;
  private discardedPages = 0;

  constructor(options: BTreeOptions = {}) {
    this.maxKeys = Math.max(2, options.maxKeys ?? 32);
    this.minLeafKeys = Math.ceil(this.maxKeys / 2);
    this.minInternalKeys = Math.ceil((this.maxKeys + 1) / 2) - 1;
    this.snapshotTimeoutMs = options.snapshotTimeoutMs ?? Infinity;
    this.nowFn = options.now ?? Date.now;
    const root: LeafPage<V> = { id: this.allocPageId(), kind: 'leaf', gen: 0, count: 0, keys: [], values: [] };
    this.pages.set(root.id, root);
    this.rootId = root.id;
  }

  get currentGeneration(): number {
    return this.generation;
  }

  /** @internal */
  get currentRootId(): PageId {
    return this.rootId;
  }

  /** @internal */
  allocPageId(): PageId {
    this.createdPages++;
    return this.nextPageId++;
  }

  /** @internal */
  page(id: PageId): Page<V> {
    const p = this.pages.get(id);
    if (!p) throw new Error(`page ${id} is not live (use-after-free)`);
    return p;
  }

  // ------------------------------------------------------------------
  // Convenience auto-commit API (single-operation write transactions)
  // ------------------------------------------------------------------

  insert(key: string, value: V): void {
    const tx = this.beginWrite();
    tx.insert(key, value);
    tx.commit();
  }

  delete(key: string): boolean {
    const tx = this.beginWrite();
    const removed = tx.delete(key);
    tx.commit();
    return removed;
  }

  get(key: string): V | undefined {
    return this.readGet(this.rootId, key);
  }

  range(start: string, end: string): { key: string; value: V }[] {
    return this.readRange(this.rootId, start, end);
  }

  size(): number {
    return this.page(this.rootId).count;
  }

  // ------------------------------------------------------------------
  // Write transactions
  // ------------------------------------------------------------------

  beginWrite(): WriteTransaction<V> {
    return new WriteTransaction<V>(this);
  }

  /**
   * Atomically publish a transaction's private pages and new root.
   * Runs synchronously, so no reader can observe a half-published tree:
   * existing snapshots keep their old root id, new readers get the new one.
   * @internal
   */
  commitTransaction(
    txnRootId: PageId,
    clones: Map<PageId, Page<V>>,
    retiredIds: PageId[],
    discardedInTxn: number,
    expectedGeneration: number,
  ): number {
    if (this.generation !== expectedGeneration) {
      throw new WriteConflictError();
    }
    const newGeneration = expectedGeneration + 1;
    for (const p of clones.values()) this.pages.set(p.id, p);
    if (retiredIds.length > 0) {
      const list = this.retired.get(newGeneration);
      if (list) list.push(...retiredIds);
      else this.retired.set(newGeneration, [...retiredIds]);
    }
    this.discardedPages += discardedInTxn;
    this.rootId = txnRootId;
    this.generation = newGeneration;
    this.reclaim();
    return newGeneration;
  }

  /**
   * Account for private pages of an aborted (rolled back or conflicted)
   * transaction. They were never published, so dropping them is enough.
   * @internal
   */
  abandonTransaction(privatePageCount: number): void {
    this.discardedPages += privatePageCount;
  }

  // ------------------------------------------------------------------
  // Snapshots
  // ------------------------------------------------------------------

  /** Open a read-only snapshot pinned to the current committed generation. */
  snapshot(): Snapshot<V> {
    const handle: SnapshotHandle = {
      id: this.nextSnapshotId++,
      generation: this.generation,
      rootId: this.rootId,
      createdAt: this.nowFn(),
      closed: false,
    };
    this.snapshots.set(handle.id, handle);
    return new Snapshot<V>(this, handle);
  }

  /** @internal */
  assertSnapshotUsable(handle: SnapshotHandle): void {
    if (handle.closed) throw new SnapshotClosedError(`snapshot ${handle.id} is closed`);
    if (this.isExpired(handle)) {
      this.releaseSnapshot(handle);
      throw new SnapshotExpiredError(
        `snapshot ${handle.id} exceeded snapshotTimeoutMs=${this.snapshotTimeoutMs}`,
      );
    }
  }

  /** @internal */
  releaseSnapshot(handle: SnapshotHandle): void {
    if (handle.closed) return;
    handle.closed = true;
    this.snapshots.delete(handle.id);
    this.reclaim();
  }

  /** Expire every snapshot that exceeded its timeout. Returns how many were closed. */
  sweepExpiredSnapshots(): number {
    const before = this.snapshots.size;
    this.reclaim();
    return before - this.snapshots.size;
  }

  private isExpired(handle: SnapshotHandle): boolean {
    return this.nowFn() - handle.createdAt > this.snapshotTimeoutMs;
  }

  /**
   * Free retired pages whose generation can no longer be observed: a page
   * retired at generation G is unreachable from every root >= G, so it is
   * safe to reclaim once the oldest open snapshot has generation >= G.
   * Expired (leaked) snapshots are closed first so they cannot pin pages.
   */
  private reclaim(): void {
    for (const h of [...this.snapshots.values()]) {
      if (this.isExpired(h)) {
        h.closed = true;
        this.snapshots.delete(h.id);
      }
    }
    let oldestPinned = Infinity;
    for (const h of this.snapshots.values()) oldestPinned = Math.min(oldestPinned, h.generation);
    for (const [gen, ids] of this.retired) {
      if (gen <= oldestPinned) {
        for (const id of ids) this.freePage(id);
        this.retired.delete(gen);
      }
    }
  }

  private freePage(id: PageId): void {
    if (!this.pages.delete(id)) {
      throw new Error(`double free of page ${id}`);
    }
    this.freedPages++;
  }

  // ------------------------------------------------------------------
  // Read paths shared by the live view and snapshots
  // ------------------------------------------------------------------

  /** @internal */
  readGet(rootId: PageId, key: string): V | undefined {
    let id = rootId;
    for (;;) {
      const p = this.page(id);
      if (p.kind === 'leaf') {
        const i = lowerBound(p.keys, key);
        return p.keys[i] === key ? p.values[i] : undefined;
      }
      id = p.children[childIndex(p.keys, key)];
    }
  }

  /** @internal */
  readRange(rootId: PageId, start: string, end: string): { key: string; value: V }[] {
    const out: { key: string; value: V }[] = [];
    const walk = (id: PageId): void => {
      const p = this.page(id);
      if (p.kind === 'leaf') {
        let i = lowerBound(p.keys, start);
        while (i < p.keys.length && p.keys[i] <= end) {
          out.push({ key: p.keys[i], value: p.values[i] });
          i++;
        }
        return;
      }
      // Child i holds keys in [keys[i-1], keys[i]); prune disjoint subtrees.
      for (let i = 0; i < p.children.length; i++) {
        const lo = i === 0 ? null : p.keys[i - 1];
        const hi = i === p.children.length - 1 ? null : p.keys[i];
        if (lo !== null && lo > end) continue;
        if (hi !== null && hi <= start) continue;
        walk(p.children[i]);
      }
    };
    walk(rootId);
    return out;
  }

  // ------------------------------------------------------------------
  // Observability and invariant checking
  // ------------------------------------------------------------------

  diagnostics(): BTreeDiagnostics {
    const now = this.nowFn();
    const list = [...this.snapshots.values()].map((h) => ({
      id: h.id,
      generation: h.generation,
      ageMs: now - h.createdAt,
      expired: this.isExpired(h),
    }));
    let retiredPending = 0;
    for (const ids of this.retired.values()) retiredPending += ids.length;
    let height = 0;
    for (let p = this.page(this.rootId); p.kind === 'internal'; p = this.page(p.children[0])) height++;
    return {
      tree: { generation: this.generation, rootId: this.rootId, height, size: this.size() },
      pages: {
        live: this.pages.size,
        created: this.createdPages,
        freed: this.freedPages,
        discarded: this.discardedPages,
        retiredPending,
      },
      snapshots: {
        open: list.length,
        oldestGeneration: list.length ? Math.min(...list.map((s) => s.generation)) : null,
        list,
      },
      snapshotTimeoutMs: this.snapshotTimeoutMs,
    };
  }

  /**
   * Throw if any structural or accounting invariant is violated:
   *  - B+ tree shape (sorted keys, separator bounds, uniform leaf depth,
   *    root entry count) for the live root and every open snapshot root;
   *  - every page reachable from a live root is in the page table;
   *  - retired pages are unreachable from the current root and not yet freed;
   *  - no live page is unreachable garbage (no permanent leak);
   *  - page accounting balances: created === live + freed + discarded.
   */
  verifyIntegrity(): void {
    const reachable = new Set<PageId>();
    const walkRoots = (rootId: PageId): void => {
      const { entries } = this.checkStructure(rootId, reachable);
      const root = this.page(rootId);
      if (root.count !== entries) {
        throw new Error(`root ${rootId} count ${root.count} != actual entries ${entries}`);
      }
    };
    walkRoots(this.rootId);
    for (const h of this.snapshots.values()) walkRoots(h.rootId);

    const retiredSet = new Set<PageId>();
    const liveReachable = new Set<PageId>();
    this.collectReachable(this.rootId, liveReachable);
    for (const [gen, ids] of this.retired) {
      for (const id of ids) {
        if (!this.pages.has(id)) throw new Error(`retired page ${id} (gen ${gen}) already freed`);
        if (liveReachable.has(id)) throw new Error(`retired page ${id} reachable from current root`);
        retiredSet.add(id);
      }
    }
    for (const id of this.pages.keys()) {
      if (!reachable.has(id) && !retiredSet.has(id)) {
        throw new Error(`leaked page ${id}: live but unreachable and not pending reclamation`);
      }
    }
    if (this.freedPages < 0 || this.discardedPages < 0) {
      throw new Error('negative page counter');
    }
    if (this.createdPages !== this.pages.size + this.freedPages + this.discardedPages) {
      throw new Error(
        `page accounting mismatch: created=${this.createdPages} live=${this.pages.size} ` +
          `freed=${this.freedPages} discarded=${this.discardedPages}`,
      );
    }
  }

  private collectReachable(id: PageId, out: Set<PageId>): void {
    if (out.has(id)) return;
    out.add(id);
    const p = this.page(id);
    if (p.kind === 'internal') for (const c of p.children) this.collectReachable(c, out);
  }

  private checkStructure(
    rootId: PageId,
    reachable: Set<PageId>,
  ): { depth: number; entries: number } {
    const walk = (
      id: PageId,
      depth: number,
    ): { depth: number; min: string | null; max: string | null; entries: number } => {
      const p = this.page(id);
      if (reachable.has(id)) throw new Error(`page ${id} referenced twice`);
      reachable.add(id);
      for (let i = 1; i < p.keys.length; i++) {
        if (p.keys[i - 1] >= p.keys[i]) throw new Error(`page ${id} keys not strictly sorted`);
      }
      if (p.kind === 'leaf') {
        return {
          depth,
          min: p.keys.length ? p.keys[0] : null,
          max: p.keys.length ? p.keys[p.keys.length - 1] : null,
          entries: p.keys.length,
        };
      }
      if (p.children.length !== p.keys.length + 1) {
        throw new Error(`internal page ${id}: ${p.children.length} children for ${p.keys.length} keys`);
      }
      let entries = 0;
      let leafDepth = -1;
      let prevMax: string | null = null;
      let min: string | null = null;
      for (let i = 0; i < p.children.length; i++) {
        const c = walk(p.children[i], depth + 1);
        if (i === 0) min = c.min;
        if (leafDepth < 0) leafDepth = c.depth;
        else if (c.depth !== leafDepth) throw new Error(`uneven leaf depth below page ${id}`);
        if (i > 0) {
          const sep = p.keys[i - 1];
          if (prevMax !== null && !(prevMax < sep)) {
            throw new Error(`separator ${sep} not greater than left child max ${prevMax}`);
          }
          if (c.min !== null && !(c.min >= sep)) {
            throw new Error(`right child min ${c.min} below separator ${sep}`);
          }
        }
        prevMax = c.max;
        entries += c.entries;
      }
      return { depth: leafDepth, min, max: prevMax, entries };
    };
    const r = walk(rootId, 0);
    return { depth: r.depth, entries: r.entries };
  }
}

/**
 * A read-only, repeatable-read view of one committed generation.
 * Must be closed; see `snapshotTimeoutMs` for the leak policy.
 */
export class Snapshot<V> {
  /** @internal */
  constructor(
    private readonly tree: BTree<V>,
    private readonly handle: SnapshotHandle,
  ) {}

  get id(): number {
    return this.handle.id;
  }

  get generation(): number {
    return this.handle.generation;
  }

  get closed(): boolean {
    return this.handle.closed;
  }

  get(key: string): V | undefined {
    this.tree.assertSnapshotUsable(this.handle);
    return this.tree.readGet(this.handle.rootId, key);
  }

  range(start: string, end: string): { key: string; value: V }[] {
    this.tree.assertSnapshotUsable(this.handle);
    return this.tree.readRange(this.handle.rootId, start, end);
  }

  size(): number {
    this.tree.assertSnapshotUsable(this.handle);
    return this.tree.page(this.handle.rootId).count;
  }

  close(): void {
    this.tree.releaseSnapshot(this.handle);
  }
}

/**
 * An optimistic write transaction. Several may be open concurrently; the
 * first to commit wins, the rest fail with `WriteConflictError` on commit
 * and release their private pages. All modifications are copy-on-write, so
 * published pages are never mutated and open snapshots are unaffected.
 */
export class WriteTransaction<V> {
  private readonly tree: BTree<V>;
  private readonly baseGeneration: number;
  private txnRootId: PageId;
  /** Private pages created by COW or splits; published on commit, dropped on abort. */
  private readonly clones = new Map<PageId, Page<V>>();
  /** Published pages replaced by clones; become reclaimable at the new generation. */
  private readonly retiredIds: PageId[] = [];
  /** Private pages dropped mid-transaction (merged away). */
  private discarded = 0;
  private delta = 0;
  private readonly baseCount: number;
  private finished = false;

  /** @internal */
  constructor(tree: BTree<V>) {
    this.tree = tree;
    this.baseGeneration = tree.currentGeneration;
    this.txnRootId = tree.currentRootId;
    this.baseCount = tree.page(tree.currentRootId).count;
  }

  get startedAtGeneration(): number {
    return this.baseGeneration;
  }

  insert(key: string, value: V): void {
    this.assertActive();
    const res = this.insertRec(this.txnRootId, key, value);
    this.txnRootId = res.id;
    if (res.split) {
      const root: InternalPage = {
        id: this.tree.allocPageId(),
        kind: 'internal',
        gen: this.baseGeneration + 1,
        count: 0,
        keys: [res.split.sep],
        children: [res.id, res.split.rightId],
      };
      this.clones.set(root.id, root);
      this.txnRootId = root.id;
    }
  }

  delete(key: string): boolean {
    this.assertActive();
    if (!this.has(key)) return false;
    const res = this.deleteRec(this.txnRootId, key);
    this.txnRootId = res.id;
    // Collapse a keyless internal root onto its only child.
    for (;;) {
      const root = this.load(this.txnRootId);
      if (root.kind !== 'internal' || root.keys.length > 0) break;
      this.retirePage(root.id);
      this.txnRootId = root.children[0];
    }
    return true;
  }

  get(key: string): V | undefined {
    this.assertActive();
    let id = this.txnRootId;
    for (;;) {
      const p = this.load(id);
      if (p.kind === 'leaf') {
        const i = lowerBound(p.keys, key);
        return p.keys[i] === key ? p.values[i] : undefined;
      }
      id = p.children[childIndex(p.keys, key)];
    }
  }

  has(key: string): boolean {
    this.assertActive();
    let id = this.txnRootId;
    for (;;) {
      const p = this.load(id);
      if (p.kind === 'leaf') return p.keys[lowerBound(p.keys, key)] === key;
      id = p.children[childIndex(p.keys, key)];
    }
  }

  size(): number {
    this.assertActive();
    return this.baseCount + this.delta;
  }

  /**
   * Publish this transaction's view as the new generation. Throws
   * `WriteConflictError` (and aborts the transaction) if another
   * transaction committed first.
   */
  commit(): number {
    this.assertActive();
    this.finished = true;
    if (this.clones.size === 0) return this.tree.currentGeneration; // read-only transaction
    const root = this.cow(this.txnRootId); // ensure the root is private before stamping it
    root.count = this.baseCount + this.delta;
    this.txnRootId = root.id;
    try {
      return this.tree.commitTransaction(
        this.txnRootId,
        this.clones,
        this.retiredIds,
        this.discarded,
        this.baseGeneration,
      );
    } catch (err) {
      this.tree.abandonTransaction(this.clones.size + this.discarded);
      throw err;
    }
  }

  /** Abort the transaction and release its private pages. */
  rollback(): void {
    this.assertActive();
    this.finished = true;
    this.tree.abandonTransaction(this.clones.size + this.discarded);
  }

  // ------------------------------------------------------------------
  // Copy-on-write machinery
  // ------------------------------------------------------------------

  private assertActive(): void {
    if (this.finished) throw new TransactionStateError();
  }

  private load(id: PageId): Page<V> {
    return this.clones.get(id) ?? this.tree.page(id);
  }

  /**
   * Return a private, mutable copy of a page. The published original is
   * recorded as retired: once the new root is published, nothing reachable
   * from it references the old page, but snapshots of older generations do.
   */
  private cow(id: PageId): Page<V> {
    const existing = this.clones.get(id);
    if (existing) return existing;
    const page = this.tree.page(id);
    let clone: Page<V>;
    if (page.kind === 'leaf') {
      clone = {
        id: this.tree.allocPageId(),
        kind: 'leaf',
        gen: this.baseGeneration + 1,
        count: 0,
        keys: [...page.keys],
        values: [...page.values],
      };
    } else {
      clone = {
        id: this.tree.allocPageId(),
        kind: 'internal',
        gen: this.baseGeneration + 1,
        count: 0,
        keys: [...page.keys],
        children: [...page.children],
      };
    }
    this.clones.set(clone.id, clone);
    this.retiredIds.push(id);
    return clone;
  }

  /** Drop a page from the tree being built: private clones vanish, published pages are retired. */
  private retirePage(id: PageId): void {
    if (this.clones.delete(id)) this.discarded++;
    else this.retiredIds.push(id);
  }

  // ------------------------------------------------------------------
  // B+ tree algorithms over COW pages
  // ------------------------------------------------------------------

  private insertRec(
    id: PageId,
    key: string,
    value: V,
  ): { id: PageId; split?: { sep: string; rightId: PageId } } {
    const page = this.cow(id);
    if (page.kind === 'leaf') {
      const i = lowerBound(page.keys, key);
      if (page.keys[i] === key) {
        page.values[i] = value; // upsert
        return { id: page.id };
      }
      page.keys.splice(i, 0, key);
      page.values.splice(i, 0, value);
      this.delta++;
      if (page.keys.length <= this.tree.maxKeys) return { id: page.id };
      const mid = page.keys.length >> 1;
      const right: LeafPage<V> = {
        id: this.tree.allocPageId(),
        kind: 'leaf',
        gen: this.baseGeneration + 1,
        count: 0,
        keys: page.keys.splice(mid),
        values: page.values.splice(mid),
      };
      this.clones.set(right.id, right);
      return { id: page.id, split: { sep: right.keys[0], rightId: right.id } };
    }
    const i = childIndex(page.keys, key);
    const res = this.insertRec(page.children[i], key, value);
    page.children[i] = res.id;
    if (!res.split) return { id: page.id };
    page.keys.splice(i, 0, res.split.sep);
    page.children.splice(i + 1, 0, res.split.rightId);
    if (page.keys.length <= this.tree.maxKeys) return { id: page.id };
    const mid = page.keys.length >> 1;
    const sep = page.keys[mid];
    const right: InternalPage = {
      id: this.tree.allocPageId(),
      kind: 'internal',
      gen: this.baseGeneration + 1,
      count: 0,
      keys: page.keys.splice(mid + 1),
      children: page.children.splice(mid + 1),
    };
    page.keys.pop(); // the separator moves up, it does not stay in the left page
    this.clones.set(right.id, right);
    return { id: page.id, split: { sep, rightId: right.id } };
  }

  private deleteRec(id: PageId, key: string): { id: PageId; underflow: boolean } {
    const page = this.cow(id);
    if (page.kind === 'leaf') {
      const i = lowerBound(page.keys, key);
      if (page.keys[i] !== key) return { id: page.id, underflow: false }; // unreachable: delete() pre-checks
      page.keys.splice(i, 1);
      page.values.splice(i, 1);
      this.delta--;
      return { id: page.id, underflow: page.keys.length < this.tree.minLeafKeys };
    }
    const i = childIndex(page.keys, key);
    const res = this.deleteRec(page.children[i], key);
    page.children[i] = res.id;
    if (res.underflow) this.rebalance(page, i);
    return { id: page.id, underflow: page.keys.length < this.tree.minInternalKeys };
  }

  /**
   * Repair an underflowing child of a private internal page by borrowing
   * from a sibling or merging with it. Siblings are cloned before being
   * modified; a merged-away page is retired.
   */
  private rebalance(parent: InternalPage, i: number): void {
    const childId = parent.children[i];
    const child = this.clones.get(childId);
    if (!child) throw new Error('rebalance: child page is not private');

    if (i > 0) {
      const leftId = parent.children[i - 1];
      const left = this.load(leftId);
      const leftMin = left.kind === 'leaf' ? this.tree.minLeafKeys : this.tree.minInternalKeys;
      if (left.keys.length > leftMin) {
        const lc = this.cow(leftId);
        if (child.kind === 'leaf' && lc.kind === 'leaf') {
          child.keys.unshift(lc.keys.pop() as string);
          child.values.unshift((lc as LeafPage<V>).values.pop() as V);
          parent.keys[i - 1] = child.keys[0];
        } else if (child.kind === 'internal' && lc.kind === 'internal') {
          child.keys.unshift(parent.keys[i - 1]);
          child.children.unshift(lc.children.pop() as PageId);
          parent.keys[i - 1] = lc.keys.pop() as string;
        }
        parent.children[i - 1] = lc.id;
        return;
      }
    }

    if (i < parent.children.length - 1) {
      const rightId = parent.children[i + 1];
      const right = this.load(rightId);
      const rightMin = right.kind === 'leaf' ? this.tree.minLeafKeys : this.tree.minInternalKeys;
      if (right.keys.length > rightMin) {
        const rc = this.cow(rightId);
        if (child.kind === 'leaf' && rc.kind === 'leaf') {
          child.keys.push(rc.keys.shift() as string);
          child.values.push((rc as LeafPage<V>).values.shift() as V);
          parent.keys[i] = rc.keys[0];
        } else if (child.kind === 'internal' && rc.kind === 'internal') {
          child.keys.push(parent.keys[i]);
          child.children.push(rc.children.shift() as PageId);
          parent.keys[i] = rc.keys.shift() as string;
        }
        parent.children[i + 1] = rc.id;
        return;
      }
    }

    if (i > 0) {
      // Merge child into its left sibling; the child's clone is dropped.
      const leftId = parent.children[i - 1];
      const lc = this.cow(leftId);
      if (child.kind === 'leaf' && lc.kind === 'leaf') {
        lc.keys.push(...child.keys);
        (lc as LeafPage<V>).values.push(...child.values);
      } else if (child.kind === 'internal' && lc.kind === 'internal') {
        lc.keys.push(parent.keys[i - 1], ...child.keys);
        lc.children.push(...child.children);
      }
      parent.keys.splice(i - 1, 1);
      parent.children.splice(i, 1);
      parent.children[i - 1] = lc.id;
      this.retirePage(childId);
    } else {
      // Merge the right sibling into the child; the sibling page is retired.
      const rightId = parent.children[i + 1];
      const right = this.load(rightId);
      if (child.kind === 'leaf' && right.kind === 'leaf') {
        child.keys.push(...right.keys);
        child.values.push(...right.values);
      } else if (child.kind === 'internal' && right.kind === 'internal') {
        child.keys.push(parent.keys[i], ...right.keys);
        child.children.push(...right.children);
      }
      parent.keys.splice(i, 1);
      parent.children.splice(i + 1, 1);
      this.retirePage(rightId);
    }
  }
}
