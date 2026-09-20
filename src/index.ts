/**
 * In-memory paged B+-tree with MVCC read snapshots.
 *
 * Concurrency model
 * -----------------
 * - Every committed version is an immutable page tree addressed by a
 *   generation number (0 = the empty tree). A `Snapshot` pins one generation;
 *   the live root always points at the latest committed generation.
 * - A writer copy-on-writes every page it descends through. A page reachable
 *   from any committed root or open snapshot is never mutated in place.
 * - Commit atomically swaps the live-root pointer to the writer's private
 *   tree and bumps the generation (one synchronous step). Rollback drops the
 *   writer's private root, freeing every page it allocated.
 * - Every page has a reference count equal to the number of parent edges
 *   referencing it plus one per holder (live root, open snapshot, in-flight
 *   write tx). Pages are reclaimed only when the count reaches zero — i.e.
 *   after every snapshot that may reference their generation has closed.
 * - Leaked snapshots are covered by an explicit lease timeout and an
 *   optional FinalizationRegistry safety net; both emit diagnostics.
 */

export interface MvccOptions {
	/** Max entries per leaf / keys per internal node. Must be >= 2. */
	fanout?: number;
	/**
	 * Default snapshot lease in milliseconds. A snapshot older than its
	 * lease expires and releases its pages; further reads throw
	 * {@link SnapshotExpiredError}. Default Infinity (disabled).
	 */
	snapshotLeaseMs?: number;
	/**
	 * Register snapshots with a FinalizationRegistry so a snapshot object
	 * garbage-collected without close() still releases its pages.
	 * Default true.
	 */
	finalizeSnapshots?: boolean;
	/** Diagnostic callback for expired/finalized snapshots and tree close. */
	onDiagnostic?: (event: MvccDiagnostic) => void;
}

export interface MvccDiagnostic {
	kind: 'snapshot-expired' | 'snapshot-finalized' | 'tree-closed';
	snapshotId: number;
	generation: number;
	message: string;
}

/** Optimistic writer lost the race: another writer committed first. */
export class WriteConflictError extends Error {
	constructor(
		public expectedGeneration: number,
		public actualGeneration: number,
	) {
		super(
			`write conflict: expected generation ${expectedGeneration}, ` +
				`committed generation is ${actualGeneration}`,
		);
		this.name = 'WriteConflictError';
	}
}

/** A read was attempted on an expired or closed snapshot. */
export class SnapshotExpiredError extends Error {
	constructor(
		public snapshotId: number,
		public generation: number,
		reason: 'closed' | 'expired',
	) {
		super(
			`snapshot ${snapshotId} at generation ${generation} was ${reason}; ` +
				`it can no longer be read and its pages have been reclaimed`,
		);
		this.name = 'SnapshotExpiredError';
	}
}

/** Use after the tree, transaction or snapshot was closed. */
export class TreeClosedError extends Error {
	constructor(what: string) {
		super(`${what} has been closed`);
		this.name = 'TreeClosedError';
	}
}

export interface Entry<V> {
	key: string;
	value: V;
}

interface LeafPage<V> {
	id: number;
	type: 'leaf';
	entries: Entry<V>[];
	/** Right-sibling chain for range scans; EMPTY_ROOT = end of chain. */
	next: number;
	gen: number;
	refs: number;
}

interface InternalPage {
	id: number;
	type: 'internal';
	/** keys[i] separates children[i] and children[i+1]. */
	keys: string[];
	children: number[];
	gen: number;
	refs: number;
}

type Page<V> = LeafPage<V> | InternalPage;

interface PageRef<V> {
	page: Page<V>;
	alive: boolean;
}

interface SnapshotRecord {
	id: number;
	generation: number;
	rootId: number;
	expiresAt: number;
	open: boolean;
	timer: ReturnType<typeof setTimeout> | null;
	/** GC watch token; kept referenced for exactly the snapshot's lifetime. */
	token: object;
}

const EMPTY_ROOT = 0;

export class MVCCBTree<V> {
	readonly fanout: number;
	readonly snapshotLeaseMs: number;
	private readonly onDiagnostic?: (event: MvccDiagnostic) => void;

	private pages = new Map<number, PageRef<V>>();
	private nextPageId = 1;

	private generation_ = 0;
	private committedRoot: number = EMPTY_ROOT;

	private snapshots = new Map<number, SnapshotRecord>();
	private nextSnapshotId = 1;

	// Single-writer machinery. Readers are lock-free.
	private writerActive = false;
	private writerQueue: Array<() => void> = [];

	private closed_ = false;
	private finalizers: FinalizationRegistry<number> | null;

	constructor(options: MvccOptions = {}) {
		const fanout = options.fanout ?? 32;
		if (!Number.isInteger(fanout) || fanout < 2) {
			throw new Error('fanout must be an integer >= 2');
		}
		this.fanout = fanout;
		this.snapshotLeaseMs = options.snapshotLeaseMs ?? Infinity;
		this.onDiagnostic = options.onDiagnostic;
		this.finalizers =
			options.finalizeSnapshots === false ||
			typeof FinalizationRegistry === 'undefined'
				? null
				: new FinalizationRegistry((snapshotId) => {
						this.releaseSnapshot(snapshotId, 'snapshot-finalized');
					});
	}

	get generation(): number {
		return this.generation_;
	}

	get closed(): boolean {
		return this.closed_;
	}

	// ============================================================ page table

	private alloc(type: Page<V>['type'], gen: number): Page<V> {
		const id = this.nextPageId++;
		const page: Page<V> =
			type === 'leaf'
				? ({ id, type: 'leaf', entries: [], next: EMPTY_ROOT, gen, refs: 0 } as LeafPage<V>)
				: ({ id, type: 'internal', keys: [], children: [], gen, refs: 0 } as InternalPage);
		this.pages.set(id, { page, alive: true });
		return page;
	}

	private page(id: number): Page<V> {
		const ref = this.pages.get(id);
		if (!ref || !ref.alive) {
			throw new Error(`internal error: dangling page reference ${id}`);
		}
		return ref.page;
	}

	private asLeaf(id: number): LeafPage<V> {
		const p = this.page(id);
		if (p.type !== 'leaf') throw new Error('internal error: expected leaf page');
		return p;
	}

	private asInternal(id: number): InternalPage {
		const p = this.page(id);
		if (p.type !== 'internal') throw new Error('internal error: expected internal page');
		return p;
	}

	private inc(id: number, n = 1): void {
		if (id === EMPTY_ROOT) return;
		this.page(id).refs += n;
	}

	/**
	 * Drop n references to id. At zero the page is removed from the table and
	 * all edges it owns are dropped recursively. A negative count is a fatal
	 * invariant violation.
	 */
	private dec(id: number, n = 1): void {
		if (id === EMPTY_ROOT) return;
		const ref = this.pages.get(id)!;
		const page = ref.page;
		page.refs -= n;
		if (page.refs < 0) {
			throw new Error(`internal error: negative refcount on page ${page.id}`);
		}
		if (page.refs === 0) {
			ref.alive = false;
			this.pages.delete(page.id);
			if (page.type === 'internal') {
				for (const child of page.children) this.dec(child);
			} else {
				this.dec(page.next);
			}
		}
	}

	/** Replace one edge/holder: acquire the new target before releasing. */
	private retarget(fromId: number, toId: number): void {
		this.inc(toId);
		this.dec(fromId);
	}

	/** Deep-copy a page for the writer's generation; the clone owns fresh
	 *  references to all child edges. */
	private clone(page: Page<V>, gen: number): Page<V> {
		const copy = this.alloc(page.type, gen);
		if (page.type === 'leaf') {
			(copy as LeafPage<V>).entries = page.entries.slice();
			(copy as LeafPage<V>).next = page.next;
			this.inc(page.next);
		} else {
			(copy as InternalPage).keys = page.keys.slice();
			(copy as InternalPage).children = page.children.slice();
			for (const child of page.children) this.inc(child);
		}
		return copy;
	}

	// ============================================================= snapshots

	/**
	 * Open a read-only snapshot of the latest committed generation. The view
	 * never changes for the lifetime of the snapshot. Close it (or let its
	 * lease expire / GC finalize it) so old generations' pages can be freed.
	 */
	snapshot(leaseMs: number = this.snapshotLeaseMs): Snapshot<V> {
		if (this.closed_) throw new TreeClosedError('tree');
		const id = this.nextSnapshotId++;
		const rootId = this.committedRoot;
		this.inc(rootId); // snapshot holder reference
		const expiresAt = leaseMs === Infinity ? Infinity : Date.now() + leaseMs;
		const timer =
			leaseMs === Infinity
				? null
				: setTimeout(() => this.releaseSnapshot(id, 'snapshot-expired'), leaseMs);
		if (timer && typeof (timer as { unref?: () => void }).unref === 'function') {
			(timer as { unref: () => void }).unref();
		}
		const token = {};
		this.snapshots.set(id, {
			id,
			generation: this.generation_,
			rootId,
			expiresAt,
			open: true,
			timer,
			token,
		});
		this.finalizers?.register(token, id);
		return new Snapshot(this, id);
	}

	private checkSnapshot(id: number): SnapshotRecord {
		const rec = this.snapshots.get(id);
		if (!rec) throw new SnapshotExpiredError(id, -1, 'closed');
		if (!rec.open) throw new SnapshotExpiredError(id, rec.generation, 'closed');
		return rec;
	}

	/** Called from lease timer and from the FinalizationRegistry. */
	private releaseSnapshot(
		id: number,
		kind: 'snapshot-expired' | 'snapshot-finalized',
	): void {
		const rec = this.snapshots.get(id);
		if (!rec || !rec.open) return;
		rec.open = false;
		if (rec.timer) clearTimeout(rec.timer);
		this.snapshots.delete(id);
		this.dec(rec.rootId); // release snapshot holder -> may free a generation
		this.finalizers?.unregister(rec.token);
		this.onDiagnostic?.({
			kind,
			snapshotId: id,
			generation: rec.generation,
			message:
				kind === 'snapshot-expired'
					? `snapshot ${id} exceeded its lease; its generation's pages were reclaimed`
					: `snapshot ${id} was garbage collected without close(); pages reclaimed`,
		});
	}

	/** Explicit close from the Snapshot handle; idempotent. */
	private closeSnapshotHandle(id: number): void {
		const rec = this.snapshots.get(id);
		if (!rec) return;
		rec.open = false;
		if (rec.timer) clearTimeout(rec.timer);
		this.snapshots.delete(id);
		this.dec(rec.rootId);
		this.finalizers?.unregister(rec.token);
	}

	private isSnapshotOpen(id: number): boolean {
		return this.snapshots.get(id)?.open === true;
	}

	private snapshotGeneration(id: number): number {
		return this.checkSnapshot(id).generation;
	}

	/** Reap snapshots whose lease has elapsed without waiting for timers. */
	reapExpiredSnapshots(now: number = Date.now()): number {
		let reaped = 0;
		for (const rec of [...this.snapshots.values()]) {
			if (rec.expiresAt <= now) {
				this.releaseSnapshot(rec.id, 'snapshot-expired');
				reaped++;
			}
		}
		return reaped;
	}

	// =============================================================== writers

	/**
	 * Begin a write transaction.
	 * - 'serializable' (default): waits for the single-writer lock.
	 * - 'optimistic': no waiting; commit() throws WriteConflictError if the
	 *   committed generation advanced while the transaction was open.
	 */
	async beginWrite(
		mode: 'serializable' | 'optimistic' = 'serializable',
	): Promise<WriteTx<V>> {
		if (this.closed_) throw new TreeClosedError('tree');
		if (mode === 'optimistic') {
			return new WriteTx(this, this.committedRoot, this.generation_, false);
		}
		while (this.writerActive) {
			await new Promise<void>((resolve) => this.writerQueue.push(resolve));
		}
		if (this.closed_) throw new TreeClosedError('tree');
		this.writerActive = true;
		return new WriteTx(this, this.committedRoot, this.generation_, true);
	}

	private releaseWriter(): void {
		this.writerActive = false;
		this.writerQueue.shift()?.();
	}

	private holdRootForTx(rootId: number): void {
		this.inc(rootId);
	}

	private commitTx(tx: WriteTx<V>): number {
		if (tx.optimistic && tx.baseGeneration !== this.generation_) {
			throw new WriteConflictError(tx.baseGeneration, this.generation_);
		}
		// Atomic publish: single synchronous pointer swap + generation bump.
		const newRoot = tx.rootId;
		const oldRoot = this.committedRoot;
		this.committedRoot = newRoot;
		this.generation_ += 1;
		// Root holder moves from the transaction to the live-root slot.
		this.dec(oldRoot);
		tx.rootId = EMPTY_ROOT;
		tx.finished = true;
		if (tx.holdsLock) this.releaseWriter();
		return this.generation_;
	}

	private rollbackTx(tx: WriteTx<V>): void {
		if (tx.finished) return;
		// Private pages are unreachable from any committed tree/snapshot:
		// dropping the tx root frees everything this transaction allocated.
		this.dec(tx.rootId);
		tx.rootId = EMPTY_ROOT;
		tx.finished = true;
		if (tx.holdsLock) this.releaseWriter();
	}

	// ============================================================ traversal

	/** Descend from rootId to the leaf whose key range contains key. */
	private findLeaf(rootId: number, key: string): LeafPage<V> {
		let node: Page<V> = this.page(rootId);
		while (node.type === 'internal') {
			let i = 0;
			while (i < node.keys.length && key >= node.keys[i]) i++;
			node = this.page(node.children[i]);
		}
		return node;
	}

	private readGet(rootId: number, key: string): V | undefined {
		if (rootId === EMPTY_ROOT) return undefined;
		const leaf = this.findLeaf(rootId, key);
		const at = lowerBound(leaf.entries, key);
		return at < leaf.entries.length && leaf.entries[at].key === key
			? leaf.entries[at].value
			: undefined;
	}

	private readRange(rootId: number, start: string, end: string): Entry<V>[] {
		if (start > end) throw new Error('range start must be <= end');
		if (rootId === EMPTY_ROOT) return [];
		let leaf: LeafPage<V> | undefined = this.findLeaf(rootId, start);
		const out: Entry<V>[] = [];
		while (leaf) {
			for (const entry of leaf.entries) {
				if (entry.key < start) continue;
				if (entry.key > end) return out;
				out.push({ key: entry.key, value: entry.value });
			}
			leaf = leaf.next === EMPTY_ROOT ? undefined : this.asLeaf(leaf.next);
		}
		return out;
	}

	private readCount(rootId: number): number {
		if (rootId === EMPTY_ROOT) return 0;
		let node: Page<V> = this.page(rootId);
		while (node.type === 'internal') node = this.page(node.children[0]);
		let leaf = node as LeafPage<V>;
		let n = 0;
		for (;;) {
			n += leaf.entries.length;
			if (leaf.next === EMPTY_ROOT) return n;
			leaf = this.asLeaf(leaf.next);
		}
	}

	/** Lazy scan: the pinned snapshot page set stays self-consistent even if
	 *  commits happen while iteration is paused. */
	private *readScan(
		rootId: number,
		start?: string,
		end?: string,
	): Generator<Entry<V>> {
		if (rootId === EMPTY_ROOT) return;
		let leaf: LeafPage<V> | undefined;
		if (start === undefined) {
			let node: Page<V> = this.page(rootId);
			while (node.type === 'internal') node = this.page(node.children[0]);
			leaf = node as LeafPage<V>;
		} else {
			leaf = this.findLeaf(rootId, start);
		}
		while (leaf) {
			for (const entry of leaf.entries) {
				if (start !== undefined && entry.key < start) continue;
				if (end !== undefined && entry.key > end) return;
				yield { key: entry.key, value: entry.value };
			}
			leaf = leaf.next === EMPTY_ROOT ? undefined : this.asLeaf(leaf.next);
		}
	}

	// ======================================================== committed reads

	get(key: string): V | undefined;
	get(key: string, snapshot: Snapshot<V>): V | undefined;
	get(key: string, snapshot?: Snapshot<V>): V | undefined {
		if (this.closed_) throw new TreeClosedError('tree');
		const rootId = snapshot ? this.checkSnapshot(snapshot.handleId).rootId : this.committedRoot;
		return this.readGet(rootId, key);
	}

	range(start: string, end: string): Entry<V>[];
	range(start: string, end: string, snapshot: Snapshot<V>): Entry<V>[];
	range(start: string, end: string, snapshot?: Snapshot<V>): Entry<V>[] {
		if (this.closed_) throw new TreeClosedError('tree');
		const rootId = snapshot ? this.checkSnapshot(snapshot.handleId).rootId : this.committedRoot;
		return this.readRange(rootId, start, end);
	}

	scan(start?: string, end?: string, snapshot?: Snapshot<V>): Iterable<Entry<V>> {
		const rootId = snapshot ? this.checkSnapshot(snapshot.handleId).rootId : this.committedRoot;
		return this.readScan(rootId, start, end);
	}

	get size(): number {
		return this.readCount(this.committedRoot);
	}

	// ============================================================== teardown

	/** Release every snapshot and the live root. In-flight write pages free
	 *  when those transactions commit or roll back. */
	close(): void {
		if (this.closed_) return;
		this.closed_ = true;
		for (const id of [...this.snapshots.keys()]) this.closeSnapshotHandle(id);
		const root = this.committedRoot;
		this.committedRoot = EMPTY_ROOT;
		this.dec(root);
		this.writerActive = false;
		const waiters = this.writerQueue.splice(0);
		for (const w of waiters) w();
		this.onDiagnostic?.({
			kind: 'tree-closed',
			snapshotId: -1,
			generation: this.generation_,
			message: 'tree closed; all snapshots released',
		});
	}

	// ============================================================ mutations
	// All COW descent happens here. A tx starts sharing the committed root;
	// the first mutation clones the root, after which every touched page is a
	// fresh, privately-owned page (refs == edges, nobody else can reach it).

	/** First write of a tx: clone the root (or create one for an empty tree). */
	private mutableRoot(rootId: number, gen: number): number {
		if (rootId === EMPTY_ROOT) {
			const leaf = this.alloc('leaf', gen);
			return leaf.id;
		}
		const clone = this.clone(this.page(rootId), gen);
		this.retarget(rootId, clone.id);
		return clone.id;
	}

	/** COW child edge of a privately-owned internal page during descent. */
	private mutableChild(internalId: number, index: number, gen: number): number {
		const node = this.asInternal(internalId);
		const childId = node.children[index];
		const clone = this.clone(this.page(childId), gen);
		// Parent keeps exactly one edge reference: old child -> clone.
		this.retarget(childId, clone.id);
		node.children[index] = clone.id;
		return clone.id;
	}

	/**
	 * Copy-on-write descent: returns (path, leafId) where path contains
	 * [internalId, childIndex] pairs root-to-parent, all privately owned.
	 */
	private cowPath(
		rootId: number,
		key: string,
		gen: number,
	): { path: Array<[number, number]>; leafId: number } {
		const path: Array<[number, number]> = [];
		let cur = this.mutableRoot(rootId, gen);
		let node = this.page(cur);
		while (node.type === 'internal') {
			let i = 0;
			while (i < node.keys.length && key >= node.keys[i]) i++;
			const childId = this.mutableChild(cur, i, gen);
			path.push([cur, i]);
			cur = childId;
			node = this.page(cur);
		}
		return { path, leafId: cur };
	}

	private txInsert(rootId: number, key: string, value: V, gen: number): number {
		const { path, leafId } = this.cowPath(rootId, key, gen);
		const leaf = this.asLeaf(leafId);
		const at = lowerBound(leaf.entries, key);
		if (at < leaf.entries.length && leaf.entries[at].key === key) {
			leaf.entries[at] = { key, value };
			return this.rootAfterChange(rootId, path);
		}
		leaf.entries.splice(at, 0, { key, value });
		if (leaf.entries.length <= this.fanout) {
			return this.rootAfterChange(rootId, path);
		}

		// ---- leaf split ----
		const half = leaf.entries.length >> 1;
		const right = this.alloc('leaf', gen) as LeafPage<V>;
		right.entries = leaf.entries.splice(half);
		// next edge: leaf.next -> right; right.next := old leaf.next.
		const oldNext = leaf.next;
		this.inc(right.id); // leaf.next will point at right
		right.next = oldNext;
		leaf.next = right.id;
		const sepKey = right.entries[0].key;

		// Propagate the new separator upward; each overflowing internal node
		// splits as well (root split creates a brand-new one-page root).
		let pushedKey = sepKey;
		let pushedChild = right.id; // ref carried as "the edge to insert"
		for (let depth = path.length - 1; depth >= 0; depth--) {
			const [internalId, childIndex] = path[depth];
			const node = this.asInternal(internalId);
			// Insert separator pushedKey before keys[childIndex], new child
			// after children[childIndex].
			node.keys.splice(childIndex, 0, pushedKey);
			this.inc(pushedChild);
			node.children.splice(childIndex + 1, 0, pushedChild);
			if (node.keys.length < this.fanout) {
				pushedChild = EMPTY_ROOT;
				return this.rootOfPath(path);
			}

			// ---- internal split ----
			const splitAt = Math.ceil(node.keys.length / 2);
			const upKey = node.keys[splitAt];
			const rightNode = this.alloc('internal', gen) as InternalPage;
			rightNode.keys = node.keys.splice(splitAt + 1);
			node.keys.splice(splitAt, 1);
			rightNode.children = node.children.splice(splitAt + 1);
			for (const c of rightNode.children) this.inc(c);
			pushedKey = upKey;
			pushedChild = rightNode.id; // carried upward (not an edge of `node`)
		}

		// ---- root split: new root owns edges to left and right ----
		const newRoot = this.alloc('internal', gen) as InternalPage;
		const leftId = this.rootOfPath(path);
		this.inc(leftId);
		this.inc(pushedChild);
		newRoot.keys = [pushedKey];
		newRoot.children = [leftId, pushedChild];
		// Old tx root is fully replaced by the new root.
		this.dec(leftId); // remove clone-of-root's tx-root ownership
		return newRoot.id;
	}

	/** Recompute the tx root after a non-structural change (path[0][0] is the
	 *  cloned root whenever a descent happened). */
	private rootAfterChange(rootId: number, path: Array<[number, number]>): number {
		return path.length === 0 ? rootId : path[0][0];
	}

	private rootOfPath(path: Array<[number, number]>): number {
		if (path.length === 0) throw new Error('internal error: expected path');
		return path[0][0];
	}

	private txDelete(rootId: number, key: string, gen: number): [number, boolean] {
		if (rootId === EMPTY_ROOT) return [EMPTY_ROOT, false];
		const { path, leafId } = this.cowPath(rootId, key, gen);
		const leaf = this.asLeaf(leafId);
		const at = lowerBound(leaf.entries, key);
		if (at >= leaf.entries.length || leaf.entries[at].key !== key) {
			// Nothing changed; the COW clones are garbage, so roll the tx back
			// to its previous root by discarding the private copies.
			const newRoot = this.discardClones(rootId, path, leafId);
			return [newRoot, false];
		}
		leaf.entries.splice(at, 1);

		// Rebalance bottom-up.
		const minLeaf = this.fanout >> 1;
		const minInternal = Math.ceil(this.fanout / 2); // min children
		let childId = leafId;
		for (let depth = path.length - 1; depth >= 0; depth--) {
			const [parentId, index] = path[depth];
			const child = this.page(childId);
			const underfull =
				child.type === 'leaf'
					? child.entries.length < minLeaf
					: child.children.length < minInternal;
			if (!underfull) break;
			childId = this.rebalance(parentId, index, gen);
		}

		let newRoot: number;
		if (path.length === 0) {
			newRoot = leafId; // root is (still) a leaf
		} else {
			newRoot = path[0][0];
			const rootNode = this.asInternal(newRoot);
			// Collapse root while it has a single child.
			while (rootNode.children.length === 1) {
				const only = rootNode.children[0];
				this.dec(newRoot); // frees empty root; dec's child edges minus `only`
				newRoot = only;
				const next = this.page(newRoot);
				if (next.type !== 'internal') break;
			}
		}
		// Deleting the last key leaves an empty leaf as the sole root; drop it
		// so the tree becomes EMPTY_ROOT again.
		const finalNode = this.page(newRoot);
		if (finalNode.type === 'leaf' && finalNode.entries.length === 0) {
			this.dec(newRoot);
			newRoot = EMPTY_ROOT;
		}
		return [newRoot, true];
	}

	/**
	 * Fix an underfull children[index] of privately-owned parent `parentId`.
	 * Returns the (possibly new) id of the child slot to continue checking.
	 */
	private rebalance(parentId: number, index: number, gen: number): number {
		const parent = this.asInternal(parentId);
		const childId = parent.children[index];
		const child = this.page(childId);

		if (index > 0) {
			const leftId = parent.children[index - 1];
			const left = this.page(leftId);
			const leftSpare =
				left.type === 'leaf'
					? left.entries.length > (this.fanout >> 1)
					: left.children.length > Math.ceil(this.fanout / 2);
			if (leftSpare) {
				if (child.type === 'leaf') {
					return this.borrowLeafLeft(parentId, index, gen);
				}
				return this.borrowInternalLeft(parentId, index, gen);
			}
		}
		if (index < parent.children.length - 1) {
			const rightId = parent.children[index + 1];
			const right = this.page(rightId);
			const rightSpare =
				right.type === 'leaf'
					? right.entries.length > (this.fanout >> 1)
					: right.children.length > Math.ceil(this.fanout / 2);
			if (rightSpare) {
				if (child.type === 'leaf') {
					return this.borrowLeafRight(parentId, index, gen);
				}
				return this.borrowInternalRight(parentId, index, gen);
			}
		}
		// No sibling has a spare entry: merge.
		if (index > 0) return this.mergeWithLeft(parentId, index, gen);
		return this.mergeWithRight(parentId, index, gen);
	}

	private borrowLeafLeft(parentId: number, index: number, gen: number): number {
		void gen;
		const parent = this.asInternal(parentId);
		const child = this.asLeaf(parent.children[index]);
		const left = this.asLeaf(parent.children[index - 1]);
		const moved = left.entries.pop()!;
		child.entries.unshift(moved);
		parent.keys[index - 1] = child.entries[0].key;
		return child.id;
	}

	private borrowLeafRight(parentId: number, index: number, gen: number): number {
		void gen;
		const parent = this.asInternal(parentId);
		const child = this.asLeaf(parent.children[index]);
		const right = this.asLeaf(parent.children[index + 1]);
		const moved = right.entries.shift()!;
		child.entries.push(moved);
		parent.keys[index] = right.entries[0].key;
		return child.id;
	}

	private borrowInternalLeft(parentId: number, index: number, gen: number): number {
		void gen;
		const parent = this.asInternal(parentId);
		const child = this.asInternal(parent.children[index]);
		const left = this.asInternal(parent.children[index - 1]);
		const sep = parent.keys[index - 1];
		const movedChild = left.children.pop()!;
		child.children.unshift(movedChild);
		// The edge moves between privately-owned parents: balance references.
		this.inc(movedChild); // child now also references... (see dec below)
		this.dec(movedChild); // ...net zero, kept explicit for audit clarity
		const movedKey = left.keys.pop()!;
		child.keys.unshift(sep);
		parent.keys[index - 1] = movedKey;
		return child.id;
	}

	private borrowInternalRight(parentId: number, index: number, gen: number): number {
		void gen;
		const parent = this.asInternal(parentId);
		const child = this.asInternal(parent.children[index]);
		const right = this.asInternal(parent.children[index + 1]);
		const sep = parent.keys[index];
		const movedChild = right.children.shift()!;
		child.children.push(movedChild);
		this.inc(movedChild);
		this.dec(movedChild); // net-zero edge move (private pages)
		const movedKey = right.keys.shift()!;
		child.keys.push(sep);
		parent.keys[index] = movedKey;
		return child.id;
	}

	/** Merge children[index-1] + separator + children[index]. */
	private mergeWithLeft(parentId: number, index: number, gen: number): number {
		void gen;
		const parent = this.asInternal(parentId);
		const rightId = parent.children[index];
		const leftId = parent.children[index - 1];
		const left = this.page(leftId);
		const right = this.page(rightId);

		if (left.type === 'leaf' && right.type === 'leaf') {
			// Leaf merge: left absorbs right's entries; chain skips right.
			left.entries.push(...right.entries);
			const afterRight = right.next;
			this.retarget(right.next, afterRight); // left.next -> right.next
			right.next = EMPTY_ROOT;
			// Remove separator + right child from parent.
			parent.keys.splice(index - 1, 1);
			parent.children.splice(index, 1);
			// Parent lost its edge to right; free it (entries were copied).
			this.dec(rightId);
			return leftId;
		}

		const li = left as InternalPage;
		const ri = right as InternalPage;
		li.keys.push(parent.keys[index - 1]);
		// Move right's edges into left. Each child keeps one live edge total.
		for (const c of ri.children) {
			li.children.push(c);
			this.inc(c);
		}
		li.keys.push(...ri.keys);
		parent.keys.splice(index - 1, 1);
		parent.children.splice(index, 1);
		this.dec(rightId); // drops right's copy of moved edges + right page itself
		return leftId;
	}

	/** Merge children[index] + separator + children[index+1]. */
	private mergeWithRight(parentId: number, index: number, gen: number): number {
		void gen;
		const parent = this.asInternal(parentId);
		const leftId = parent.children[index];
		const rightId = parent.children[index + 1];
		const left = this.page(leftId);
		const right = this.page(rightId);

		if (left.type === 'leaf' && right.type === 'leaf') {
			left.entries.push(...right.entries);
			const afterRight = right.next;
			this.retarget(right.next, afterRight);
			right.next = EMPTY_ROOT;
			parent.keys.splice(index, 1);
			parent.children.splice(index + 1, 1);
			this.dec(rightId);
			return leftId;
		}

		const li = left as InternalPage;
		const ri = right as InternalPage;
		li.keys.push(parent.keys[index]);
		for (const c of ri.children) {
			li.children.push(c);
			this.inc(c);
		}
		li.keys.push(...ri.keys);
		parent.keys.splice(index, 1);
		parent.children.splice(index + 1, 1);
		this.dec(rightId);
		return leftId;
	}

	/**
	 * Delete missed the key: no structural change is wanted. Drop all private
	 * clones produced by the descent and return the original tx root.
	 */
	private discardClones(
		rootId: number,
		path: Array<[number, number]>,
		leafId: number,
	): number {
		// Every cloned page is unreferenced except by its private parent edge;
		// freeing the cloned root cascades through the whole clone chain.
		if (path.length === 0) {
			if (leafId !== rootId) this.dec(leafId);
			return rootId;
		}
		const clonedRoot = path[0][0];
		this.dec(clonedRoot);
		return rootId;
	}

	// =========================================================== diagnostics

	diagnostics(): {
		generation: number;
		livePages: number;
		openSnapshots: Array<{ id: number; generation: number; expiresAt: number }>;
		writerActive: boolean;
	} {
		return {
			generation: this.generation_,
			livePages: this.pages.size,
			openSnapshots: [...this.snapshots.values()].map((s) => ({
				id: s.id,
				generation: s.generation,
				expiresAt: s.expiresAt,
			})),
			writerActive: this.writerActive,
		};
	}

	/**
	 * Refcount invariant: for every live page, refs == 1 (ownership) + the
	 * number of incoming edges from live reachable pages. Also checks the
	 * B+-tree structural invariants. Throws on any violation.
	 */
	verifyAccounting(): { livePages: number; heldRoots: number; leafEntries: number } {
		const incoming = new Map<number, number>();
		const roots: number[] = [];
		if (this.committedRoot !== EMPTY_ROOT) roots.push(this.committedRoot);
		for (const rec of this.snapshots.values()) {
			if (rec.rootId !== EMPTY_ROOT) roots.push(rec.rootId);
		}

		const visited = new Set<number>();
		for (const rootId of roots) {
			const stack = [rootId];
			while (stack.length) {
				const id = stack.pop()!;
				if (id === EMPTY_ROOT || visited.has(id)) continue;
				visited.add(id);
				const page = this.page(id);
				if (page.type === 'internal') {
					if (page.children.length < 2) {
						throw new Error(`internal page ${id} has ${page.children.length} children`);
					}
					if (page.keys.length !== page.children.length - 1) {
						throw new Error(`internal page ${id}: keys/children mismatch`);
					}
					for (let i = 0; i < page.keys.length; i++) {
						if (page.keys[i] >= page.keys[i + 1] && i + 1 < page.keys.length) {
							throw new Error(`internal page ${id}: separators not strictly sorted`);
						}
					}
					for (const child of page.children) {
						incoming.set(child, (incoming.get(child) ?? 0) + 1);
						stack.push(child);
					}
				} else {
					for (let i = 1; i < page.entries.length; i++) {
						if (page.entries[i - 1].key >= page.entries[i].key) {
							throw new Error(`leaf ${id}: entries not strictly sorted`);
						}
					}
					if (page.entries.length > this.fanout) {
						throw new Error(`leaf ${id}: ${page.entries.length} > fanout ${this.fanout}`);
					}
					if (page.next !== EMPTY_ROOT) {
						incoming.set(page.next, (incoming.get(page.next) ?? 0) + 1);
						stack.push(page.next);
					}
				}
			}
		}

		let leafEntries = 0;
		for (const id of this.pages.keys()) {
			const page = this.page(id);
			const expected = (incoming.get(id) ?? 0) + 1; // +1 page-table ownership
			if (page.refs !== expected) {
				throw new Error(
					`accounting mismatch on page ${id}: refs=${page.refs} expected=${expected}`,
				);
			}
			if (page.gen > this.generation_) {
				throw new Error(`page ${id} stamped with future generation ${page.gen}`);
			}
			if (page.type === 'leaf') leafEntries += page.entries.length;
		}
		return { livePages: this.pages.size, heldRoots: roots.length, leafEntries };
	}

	// --------------------------------------------------- handle entry points

	/** @internal */
	_snapshotGeneration(id: number): number {
		return this.snapshotGeneration(id);
	}
	/** @internal */
	_isSnapshotOpen(id: number): boolean {
		return this.isSnapshotOpen(id);
	}
	/** @internal */
	_closeSnapshotHandle(id: number): void {
		this.closeSnapshotHandle(id);
	}
	/** @internal */
	_snapshotReadGet(id: number, key: string): V | undefined {
		return this.readGet(this.checkSnapshot(id).rootId, key);
	}
	/** @internal */
	_snapshotReadRange(id: number, start: string, end: string): Entry<V>[] {
		return this.readRange(this.checkSnapshot(id).rootId, start, end);
	}
	/** @internal */
	*_snapshotReadScan(
		id: number,
		start?: string,
		end?: string,
	): Generator<Entry<V>> {
		yield* this.readScan(this.checkSnapshot(id).rootId, start, end);
	}
	/** @internal */
	_holdRootForTx(rootId: number): void {
		this.holdRootForTx(rootId);
	}
	/** @internal */
	_commitTx(tx: WriteTx<V>): number {
		return this.commitTx(tx);
	}
	/** @internal */
	_rollbackTx(tx: WriteTx<V>): void {
		this.rollbackTx(tx);
	}
	/** @internal */
	_txGet(rootId: number, key: string): V | undefined {
		return this.readGet(rootId, key);
	}
	/** @internal */
	_txRange(rootId: number, start: string, end: string): Entry<V>[] {
		return this.readRange(rootId, start, end);
	}
	/** @internal */
	_txInsert(rootId: number, key: string, value: V, gen: number): number {
		return this.txInsert(rootId, key, value, gen);
	}
	/** @internal */
	_txDelete(rootId: number, key: string, gen: number): [number, boolean] {
		return this.txDelete(rootId, key, gen);
	}
}

// ============================================================ handle classes

/** Read-only view of one committed generation. */
export class Snapshot<V> {
	private tree: MVCCBTree<V> | null;
	/** @internal */ readonly handleId: number;

	/** @internal */
	constructor(tree: MVCCBTree<V>, id: number) {
		this.tree = tree;
		this.handleId = id;
	}

	get id(): number {
		return this.handleId;
	}

	get generation(): number {
		return this.#tree()._snapshotGeneration(this.handleId);
	}

	get closed(): boolean {
		return this.tree === null || !this.tree._isSnapshotOpen(this.handleId);
	}

	get(key: string): V | undefined {
		return this.#tree()._snapshotReadGet(this.handleId, key);
	}

	range(start: string, end: string): Entry<V>[] {
		return this.#tree()._snapshotReadRange(this.handleId, start, end);
	}

	scan(start?: string, end?: string): Iterable<Entry<V>> {
		return this.#tree()._snapshotReadScan(this.handleId, start, end);
	}

	close(): void {
		this.tree?._closeSnapshotHandle(this.handleId);
		this.tree = null;
	}

	[Symbol.dispose](): void {
		this.close();
	}

	#tree(): MVCCBTree<V> {
		if (!this.tree) throw new SnapshotExpiredError(this.handleId, -1, 'closed');
		return this.tree;
	}
}

/** Private copy-on-write working tree; published atomically on commit. */
export class WriteTx<V> {
	private tree: MVCCBTree<V> | null;
	/** @internal */ rootId: number;
	/** @internal */ readonly baseGeneration: number;
	/** @internal */ readonly holdsLock: boolean;
	/** @internal */ readonly optimistic: boolean;
	/** @internal */ finished = false;

	/** @internal */
	constructor(
		tree: MVCCBTree<V>,
		committedRoot: number,
		generation: number,
		holdsLock: boolean,
	) {
		this.tree = tree;
		this.rootId = committedRoot;
		this.optimistic = !holdsLock;
		this.holdsLock = holdsLock;
		this.baseGeneration = generation;
		tree._holdRootForTx(committedRoot);
	}

	get generation(): number {
		return this.baseGeneration;
	}

	get(key: string): V | undefined {
		return this.#tree()._txGet(this.rootId, key);
	}

	range(start: string, end: string): Entry<V>[] {
		return this.#tree()._txRange(this.rootId, start, end);
	}

	insert(key: string, value: V): void {
		const tree = this.#tree();
		this.rootId = tree._txInsert(this.rootId, key, value, this.baseGeneration + 1);
	}

	delete(key: string): boolean {
		const tree = this.#tree();
		const [rootId, removed] = tree._txDelete(this.rootId, key, this.baseGeneration + 1);
		this.rootId = rootId;
		return removed;
	}

	/** Atomically publish; returns the new generation. */
	commit(): number {
		const tree = this.#tree();
		const gen = tree._commitTx(this);
		this.tree = null;
		return gen;
	}

	rollback(): void {
		this.tree?._rollbackTx(this);
		this.tree = null;
	}

	[Symbol.dispose](): void {
		if (this.tree) this.rollback();
	}

	#tree(): MVCCBTree<V> {
		if (!this.tree) throw new TreeClosedError('write transaction');
		if (this.finished) throw new TreeClosedError('write transaction (already finished)');
		return this.tree;
	}
}

// ------------------------------------------------------------------ helpers

function lowerBound<V>(entries: Entry<V>[], key: string): number {
	let lo = 0;
	let hi = entries.length;
	while (lo < hi) {
		const mid = (lo + hi) >> 1;
		if (entries[mid].key < key) lo = mid + 1;
		else hi = mid;
	}
	return lo;
}
