import {
  SnapshotClosedError,
  SnapshotExpiredError,
  TransactionClosedError,
  WriteConflictError,
} from './errors.js';
import { EMPTY_ROOT, PageStore, type LeafPage, type Page } from './page-store.js';
import { lowerBound, RangeCursor, Workspace } from './workspace.js';

export {
  WriteConflictError,
  SnapshotClosedError,
  SnapshotExpiredError,
  TransactionClosedError,
};
export { PageStore } from './page-store.js';

export interface KV<V> {
  key: string;
  value: V;
}

export interface BTreeOptions {
  /** Max keys per page; rounded down to an even number. Default 32. */
  pageSize?: number;
  /**
   * Default lifetime for snapshots opened without an explicit `timeoutMs`.
   * `undefined` (default) means snapshots never expire and must be closed.
   */
  defaultSnapshotTimeoutMs?: number;
}

export interface SnapshotInfo {
  id: number;
  label: string | undefined;
  createdAt: number;
  ageMs: number;
  expiresAt: number | undefined;
  /** Total pages reachable from this snapshot's root. */
  reachablePages: number;
  /**
   * Pages reachable from this snapshot but NOT from the newest committed
   * root — i.e. history pages kept alive by old snapshots. Closing the
   * oldest snapshots reclaims these.
   */
  pinnedPages: number;
}

interface TreeState {
  root: number;
  size: number;
  version: number;
}

interface SnapshotRecord {
  id: number;
  label?: string;
  root: number;
  size: number;
  version: number;
  createdAt: number;
  expiresAt?: number;
  released: boolean;
}

/**
 * Point-in-time read view. All reads (point gets and async range scans) see
 * exactly the data committed at the moment the snapshot was opened, no matter
 * how many splits, merges or commits happen afterwards.
 *
 * Snapshots pin pages: always `close()` one when done (or give it a timeout).
 */
export class Snapshot<V> {
  private readonly tree: BTree<V>;
  private readonly rec: SnapshotRecord;

  /** @internal */
  constructor(tree: BTree<V>, rec: SnapshotRecord) {
    this.tree = tree;
    this.rec = rec;
  }

  get version(): number {
    return this.rec.version;
  }

  get closed(): boolean {
    return this.rec.released;
  }

  /** Number of entries visible in this snapshot. */
  get size(): number {
    this.tree.internalCheckSnapshot(this.rec);
    return this.rec.size;
  }

  get(key: string): V | undefined {
    this.tree.internalCheckSnapshot(this.rec);
    return this.tree.internalRead(this.rec.root, key);
  }

  /** Whether the key exists in this snapshot. */
  has(key: string): boolean {
    this.tree.internalCheckSnapshot(this.rec);
    return this.tree.internalReadWithVersion(this.rec.root, key) !== undefined;
  }

  /**
   * Inclusive `[start, end]` range scan as an async iterator. Snapshot
   * liveness is re-checked before every yielded entry, so the caller may
   * `await` anything between iterations; a snapshot closed or expired
   * mid-scan makes the next `next()` reject instead of returning data.
   */
  async *range(start: string, end: string): AsyncIterableIterator<KV<V>> {
    this.tree.internalCheckSnapshot(this.rec);
    const cursor = this.tree.internalCursor(this.rec.root, start, end);
    for (;;) {
      this.tree.internalCheckSnapshot(this.rec);
      const item = cursor.next();
      if (!item) return;
      yield { key: item.key, value: item.value };
    }
  }

  /** Collect the whole inclusive range into an array. */
  async rangeArray(start: string, end: string): Promise<KV<V>[]> {
    const out: KV<V>[] = [];
    for await (const item of this.range(start, end)) out.push(item);
    return out;
  }

  close(): void {
    this.tree.internalReleaseSnapshot(this.rec);
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return `Snapshot(version=${this.rec.version}, ${this.rec.released ? 'closed' : 'open'})`;
  }
}

/**
 * Private batch of writes. Visible to nobody until `commit()` succeeds
 * atomically; `rollback()` (or any error during commit) leaves no trace.
 */
export class WriteTransaction<V> {
  private readonly tree: BTree<V>;
  private workspace: Workspace<V>;
  private readonly baseRoot: number;
  private readonly baseVersion: number;
  /** touched key -> whether it existed in the base snapshot */
  private readonly touched = new Map<string, boolean>();
  /**
   * Ordered logical operations, used to replay the batch onto a newer base on
   * commit when another transaction committed first (optimistic rebase).
   */
  private readonly ops: Array<
    { type: 'put'; key: string; value: V } | { type: 'del'; key: string }
  > = [];
  private closed = false;

  /** @internal */
  constructor(tree: BTree<V>, baseRoot: number, baseSize: number, baseVersion: number) {
    this.tree = tree;
    this.baseRoot = baseRoot;
    this.baseVersion = baseVersion;
    this.workspace = new Workspace(tree.internalStore, baseRoot, baseSize, tree.internalPageSize);
  }

  get isClosed(): boolean {
    return this.closed;
  }

  /** Working-tree size including this transaction's uncommitted changes. */
  get size(): number {
    if (this.closed) throw new TransactionClosedError();
    return this.workspace.size;
  }

  insert(key: string, value: V): void {
    if (this.closed) throw new TransactionClosedError();
    this.touch(key);
    this.workspace.insert(key, value, -1);
    this.ops.push({ type: 'put', key, value });
  }

  get(key: string): V | undefined {
    if (this.closed) throw new TransactionClosedError();
    return this.workspace.get(key);
  }

  has(key: string): boolean {
    if (this.closed) throw new TransactionClosedError();
    return this.workspace.get(key) !== undefined;
  }

  delete(key: string): V | undefined {
    if (this.closed) throw new TransactionClosedError();
    this.touch(key);
    const removed = this.workspace.delete(key).value;
    this.ops.push({ type: 'del', key });
    return removed as V | undefined;
  }

  private touch(key: string): void {
    if (!this.touched.has(key)) {
      this.touched.set(key, this.tree.internalRead(this.baseRoot, key) !== undefined);
    }
  }

  /**
   * Commit the batch. Commits are serialized across transactions even though
   * several may be open concurrently. First-committer-wins: if another
   * transaction committed first and changed any of the same keys, this one
   * is rolled back and {@link WriteConflictError} is thrown carrying the
   * conflicting keys — the caller retries or merges explicitly; no write is
   * ever silently dropped.
   */
  commit(): Promise<void> {
    return this.tree.internalCommitTxn(this);
  }

  rollback(): void {
    if (this.closed) throw new TransactionClosedError();
    this.tree.internalAbortTxn(this);
  }

  /** @internal */
  internalDispose(committed: boolean): void {
    this.closed = true;
    if (!committed) this.tree.internalStore.freeDrafts(this.workspace.drafts);
    this.tree.internalStore.releaseRoot(this.baseRoot);
  }

  /** @internal */
  get isClosedInternal(): boolean {
    return this.closed;
  }

  /**
   * @internal Validate, (if needed) rebase onto the newest committed tree,
   * then retain and promote the draft subtree. Throws WriteConflictError
   * before changing any refcount or committed state.
   */
  internalPublish(newVersion: number): { root: number; size: number } {
    const current = this.tree.internalState;
    if (this.baseVersion !== current.version) {
      const conflicts = this.detectConflicts();
      if (conflicts.length) throw new WriteConflictError(conflicts);
      this.#rebaseOnto(current);
    }
    const root = this.workspace.root;
    this.tree.internalStore.commitDrafts(root, newVersion, this.workspace.drafts);
    return { root, size: this.workspace.size };
  }

  /**
   * Replay this batch's logical operations onto the newest committed root.
   * Called only after the conflict check proves no touched key changed since
   * the base, so the replay result is exactly the merge of the two
   * transactions — disjoint writes both survive. The old (base-built) draft
   * pages are discarded; they never reached committed state.
   */
  #rebaseOnto(current: { root: number; size: number; version: number }): void {
    const store = this.tree.internalStore;
    store.freeDrafts(this.workspace.drafts);
    const ws = new Workspace<V>(store, current.root, current.size, this.tree.internalPageSize);
    for (const op of this.ops) {
      if (op.type === 'put') ws.insert(op.key, op.value, -1);
      else ws.delete(op.key);
    }
    this.workspace = ws;
  }

  private detectConflicts(): string[] {
    const current = this.tree.internalState;
    if (this.baseVersion === current.version) return [];
    const conflicts: string[] = [];
    for (const [key, existedAtBase] of this.touched) {
      const now = this.tree.internalReadWithVersion(current.root, key);
      if (now === undefined) {
        // Another transaction deleted a key this one also wrote/deleted.
        if (existedAtBase) conflicts.push(key);
      } else if (now.version > this.baseVersion) {
        conflicts.push(key);
      }
    }
    return conflicts;
  }
}

/**
 * MVCC B-tree. Pages are fixed-capacity, copy-on-write; readers pin immutable
 * roots, writers build private draft pages and publish a new root in one
 * synchronous swap.
 */
export class BTree<V> {
  private readonly store = new PageStore<V>();
  private state: TreeState = { root: EMPTY_ROOT, size: 0, version: 0 };
  private readonly pageSize: number;
  private readonly defaultTimeoutMs: number | undefined;
  private readonly snapshots = new Set<SnapshotRecord>();
  private nextSnapshotId = 1;
  /** Serialization chain for async commit() callers. */
  private commitChain: Promise<unknown> = Promise.resolve();

  constructor(options: BTreeOptions = {}) {
    const ps = options.pageSize ?? 32;
    if (ps < 4) throw new Error('pageSize must be at least 4');
    this.pageSize = ps % 2 === 0 ? ps : ps - 1;
    this.defaultTimeoutMs = options.defaultSnapshotTimeoutMs;
  }

  // ---- convenience API (auto-commit, no explicit transaction) ----------------

  /** Insert or update one key, committed immediately. */
  insert(key: string, value: V): void {
    const txn = this.beginTransaction();
    txn.insert(key, value);
    this.commitInline(txn);
  }

  get(key: string): V | undefined {
    return this.internalRead(this.state.root, key);
  }

  has(key: string): boolean {
    return this.internalRead(this.state.root, key) !== undefined;
  }

  /** Delete one key; returns the removed value. */
  delete(key: string): V | undefined {
    const txn = this.beginTransaction();
    const removed = txn.delete(key);
    this.commitInline(txn);
    return removed;
  }

  /** Inclusive range as a plain array (synchronous convenience form). */
  range(start: string, end: string): KV<V>[] {
    const cursor = new RangeCursor<V>(this.store, this.state.root, start, end);
    const out: KV<V>[] = [];
    for (;;) {
      const item = cursor.next();
      if (!item) return out;
      out.push(item);
    }
  }

  size(): number {
    return this.state.size;
  }

  // ---- transactions & snapshots ----------------------------------------------

  beginTransaction(): WriteTransaction<V> {
    this.reapExpired();
    this.store.pinRoot(this.state.root);
    return new WriteTransaction(this, this.state.root, this.state.size, this.state.version);
  }

  /**
   * Open a snapshot — O(1) in the number of entries: it records the current
   * root and adds one root pin (interior edges are already retained).
   * @param timeoutMs optional auto-expiry; an expired snapshot releases its
   *        pages and every further read throws {@link SnapshotExpiredError}.
   * @param label optional name shown by `snapshotInfo()` for leak hunting.
   */
  snapshot(timeoutMs?: number, label?: string): Snapshot<V> {
    this.reapExpired();
    const ttl = timeoutMs ?? this.defaultTimeoutMs;
    const now = Date.now();
    const rec: SnapshotRecord = {
      id: this.nextSnapshotId++,
      label,
      root: this.state.root,
      size: this.state.size,
      version: this.state.version,
      createdAt: now,
      expiresAt: ttl === undefined ? undefined : now + ttl,
      released: false,
    };
    this.store.pinRoot(rec.root);
    this.snapshots.add(rec);
    return new Snapshot(this, rec);
  }

  // ---- monitoring ------------------------------------------------------------

  /** Pages currently allocated in the page store (committed + open drafts). */
  livePageCount(): number {
    return this.store.livePageCount();
  }

  /** Pages reachable from the newest committed root. */
  pageCount(): number {
    return this.store.reachablePageCount(this.state.root);
  }

  get currentVersion(): number {
    return this.state.version;
  }

  /**
   * Open snapshots, oldest first — use this to find leaked snapshots. Each
   * entry reports age and the old pages it keeps alive (`pinnedPages`).
   */
  snapshotInfo(): SnapshotInfo[] {
    this.reapExpired();
    const currentPages = this.store.collectReachable(this.state.root, new Set());
    return [...this.snapshots]
      .filter((s) => !s.released)
      .sort((a, b) => a.createdAt - b.createdAt)
      .map((s) => {
        const reachable = this.store.collectReachable(s.root, new Set());
        let pinned = 0;
        for (const id of reachable) if (!currentPages.has(id)) pinned++;
        return {
          id: s.id,
          label: s.label,
          createdAt: s.createdAt,
          ageMs: Date.now() - s.createdAt,
          expiresAt: s.expiresAt,
          reachablePages: reachable.size,
          pinnedPages: pinned,
        };
      });
  }

  /** Close and reclaim every expired snapshot; returns how many were reaped. */
  reapExpiredSnapshots(): number {
    return this.reapExpired();
  }

  // ---- internals (shared with Snapshot / WriteTransaction) -------------------

  /** @internal */
  get internalStore(): PageStore<V> {
    return this.store;
  }

  /** @internal */
  get internalPageSize(): number {
    return this.pageSize;
  }

  /** @internal */
  get internalState(): TreeState {
    return this.state;
  }

  /** @internal */
  internalCursor(root: number, start: string, end: string): RangeCursor<V> {
    return new RangeCursor<V>(this.store, root, start, end);
  }

  /** @internal */
  internalRead(root: number, key: string): V | undefined {
    let id = root;
    const store = this.store;
    while (id !== EMPTY_ROOT) {
      const page: Page<V> = store.get(id)!;
      if (page.kind === 'leaf') {
        const i = lowerBound(page.keys, key);
        return i < page.keys.length && page.keys[i] === key ? page.values[i] : undefined;
      }
      let i = lowerBound(page.keys, key);
      if (i < page.keys.length && page.keys[i] === key) i++;
      id = page.children[i];
    }
    return undefined;
  }

  /** @internal */
  internalReadWithVersion(
    root: number,
    key: string,
  ): { value: V; version: number } | undefined {
    let id = root;
    const store = this.store;
    while (id !== EMPTY_ROOT) {
      const page: Page<V> = store.get(id)!;
      if (page.kind === 'leaf') {
        const i = lowerBound(page.keys, key);
        if (i < page.keys.length && page.keys[i] === key) {
          return { value: page.values[i], version: (page as LeafPage<V>).versions[i] };
        }
        return undefined;
      }
      let i = lowerBound(page.keys, key);
      if (i < page.keys.length && page.keys[i] === key) i++;
      id = page.children[i];
    }
    return undefined;
  }

  /** @internal */
  internalCheckSnapshot(rec: SnapshotRecord): void {
    if (rec.released) {
      throw rec.expiresAt !== undefined && Date.now() >= rec.expiresAt
        ? new SnapshotExpiredError()
        : new SnapshotClosedError();
    }
    if (rec.expiresAt !== undefined && Date.now() >= rec.expiresAt) {
      this.internalReleaseSnapshot(rec);
      throw new SnapshotExpiredError();
    }
  }

  /** @internal */
  internalReleaseSnapshot(rec: SnapshotRecord): void {
    if (rec.released) return;
    rec.released = true;
    this.snapshots.delete(rec);
    this.store.releaseRoot(rec.root);
  }

  /** @internal */
  internalAbortTxn(txn: WriteTransaction<V>): void {
    txn.internalDispose(false);
  }

  /** Synchronous auto-commit used by the convenience API. */
  private commitInline(txn: WriteTransaction<V>): void {
    try {
      this.publish(txn);
      txn.internalDispose(true);
    } catch (err) {
      txn.internalDispose(false);
      throw err;
    }
  }

  /** @internal */
  internalCommitTxn(txn: WriteTransaction<V>): Promise<void> {
    if (txn.isClosedInternal) return Promise.reject(new TransactionClosedError());
    // Each transaction waits for earlier commit()s, then runs its own
    // (synchronous) publish; a rejection elsewhere never breaks this chain.
    const run = this.commitChain.then(() => {
      if (txn.isClosedInternal) throw new TransactionClosedError();
      this.publish(txn);
    });
    this.commitChain = run.then(
      () => undefined,
      () => undefined,
    );
    return run.then(
      () => txn.internalDispose(true),
      (err: unknown) => {
        if (!txn.isClosedInternal) txn.internalDispose(false);
        throw err;
      },
    );
  }

  private reapExpired(): number {
    const now = Date.now();
    let n = 0;
    for (const rec of this.snapshots) {
      if (!rec.released && rec.expiresAt !== undefined && now >= rec.expiresAt) {
        this.internalReleaseSnapshot(rec);
        n++;
      }
    }
    return n;
  }

  /**
   * Synchronously validate and publish one transaction. Sequence:
   *   1. conflict check (may throw — nothing has changed, rollback semantics);
   *   2. retain+promote the drafts, stamp versions and add the new tree's
   *      root pin — all while the old current pin and the transaction base
   *      pin are held, so a subtree shared by both roots never dips to zero;
   *   3. swap the root (single assignment = atomic visibility point);
   *   4. release the old current root; transaction dispose releases its base
   *      pin. Dropped pages are reclaimed then (unless a snapshot pins them).
   */
  private publish(txn: WriteTransaction<V>): void {
    this.reapExpired();
    const newVersion = this.state.version + 1;
    const oldRoot = this.state.root;
    const next = txn.internalPublish(newVersion); // throws WriteConflictError
    const newState: TreeState = {
      root: next.root,
      size: next.size,
      version: newVersion,
    };
    // commitDrafts already retained the whole new tree, including the
    // current-owner pin on its root.
    this.state = newState; // atomic visibility point
    this.store.releaseRoot(oldRoot); // drop the previous current-tree owner pin
  }
}
