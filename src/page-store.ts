/**
 * Page storage: the single place pages are allocated from and returned to.
 *
 * A page is either a leaf (`{ keys, values, versions }`) or an internal node
 * (`{ keys, children }`). Committed pages are immutable. Pages a write
 * transaction builds are drafts, mutated privately until commit, then
 * promoted; on abort the whole draft set is swept.
 *
 * Reclaim model — edge reference counts
 * -------------------------------------
 * Invariant for a committed page P:
 *
 *   refs(P) = number of owner pins on P (P is someone's root)
 *           + number of live committed in-tree edges pointing into P.
 *
 * Owners are the current tree, every open snapshot and every open write
 * transaction's base.
 *
 * Publishing a transaction walks only its freshly written draft region:
 *   - the new root and every draft page gain the edges that lead to them;
 *   - where a draft page points at an already-committed child, that child's
 *     count gains the one new edge and the walk stops (the shared subtree is
 *     immutable and its interior was already balanced when it was published).
 *
 * Releasing an owner removes its root pin and then walks the structure: each
 * freed page removes the edge to each child (child count -1); a child whose
 * count stays positive is shared with another live version and the walk stops
 * there immediately. Consequently reclaim visits exactly the pages that
 * became garbage — the changed root-to-leaf paths — never the whole tree, and
 * opening/closing a snapshot on a committed root is a single root +/- (its
 * interior edges are already counted).
 */

export interface LeafPage<V> {
  kind: 'leaf';
  keys: string[];
  values: V[];
  /** Commit version of each cell; -1 means "staged, not committed yet". */
  versions: number[];
  refs: number;
}

export interface InternalPage {
  kind: 'internal';
  keys: string[];
  children: number[];
  refs: number;
}

export type Page<V> = LeafPage<V> | InternalPage;

export const EMPTY_ROOT = -1;

export class PageStore<V> {
  #pages = new Map<number, Page<V>>();
  #nextId = 0;
  #freeIds: number[] = [];
  #drafts = new Set<number>();

  /** Number of currently allocated (live) pages, for memory monitoring. */
  livePageCount(): number {
    return this.#pages.size;
  }

  isDraft(id: number): boolean {
    return this.#drafts.has(id);
  }

  get(id: number): Page<V> | undefined {
    return this.#pages.get(id);
  }

  getLeaf(id: number): LeafPage<V> {
    const page = this.#pages.get(id);
    if (!page || page.kind !== 'leaf') throw new Error(`not a leaf page: ${id}`);
    return page;
  }

  getInternal(id: number): InternalPage {
    const page = this.#pages.get(id);
    if (!page || page.kind !== 'internal') throw new Error(`not an internal page: ${id}`);
    return page;
  }

  /** Allocate a fresh empty leaf as an uncommitted draft. */
  allocLeaf(): number {
    const id = this.#freeIds.pop() ?? this.#nextId++;
    this.#pages.set(id, { kind: 'leaf', keys: [], values: [], versions: [], refs: 0 });
    this.#drafts.add(id);
    return id;
  }

  /** Allocate a fresh empty internal node (with one initial child) as a draft. */
  allocInternal(childId: number): number {
    const id = this.#freeIds.pop() ?? this.#nextId++;
    this.#pages.set(id, { kind: 'internal', keys: [], children: [childId], refs: 0 });
    this.#drafts.add(id);
    return id;
  }

  /** Copy a page into a new mutable draft page. */
  clone(id: number): number {
    const src = this.#pages.get(id);
    if (!src) throw new Error(`cannot clone missing page: ${id}`);
    const copyId = this.#freeIds.pop() ?? this.#nextId++;
    const copy: Page<V> =
      src.kind === 'leaf'
        ? {
            kind: 'leaf',
            keys: src.keys.slice(),
            values: src.values.slice(),
            versions: src.versions.slice(),
            refs: 0,
          }
        : { kind: 'internal', keys: src.keys.slice(), children: src.children.slice(), refs: 0 };
    this.#pages.set(copyId, copy);
    this.#drafts.add(copyId);
    return copyId;
  }

  /**
   * Free one transaction's draft pages (rollback / failed commit). Drafts are
   * only reachable through that transaction's working root and never share
   * ids with another transaction (one can only clone committed pages), so the
   * whole set is deleted; committed children they referenced stay pinned by
   * the base owner.
   */
  freeDrafts(owner: Set<number>): number {
    let freed = 0;
    for (const id of owner) {
      if (this.#pages.delete(id)) {
        this.#freeIds.push(id);
        freed++;
      }
      this.#drafts.delete(id);
    }
    owner.clear();
    return freed;
  }

  /**
   * Retain the draft subtree reachable from `root` just before it becomes the
   * committed tree: add the new version's root pin and every edge that lives
   * inside its draft region. Edges that cross into an already-committed child
   * add one to that child and stop. Then stamp versions, clear the draft
   * flags and sweep this owner's unreachable drafts (merge losers, abandoned
   * roots). Must run while the pages are still flagged as drafts.
   */
  commitDrafts(root: number, version: number, owner: Set<number>): void {
    const reachable = new Set<number>();
    if (root !== EMPTY_ROOT) {
      const stack: number[] = [root];
      while (stack.length) {
        const id = stack.pop()!;
        if (reachable.has(id)) continue;
        reachable.add(id);
        const page = this.#pages.get(id)!;
        page.refs++; // edge from the parent draft, or the root pin
        if (page.kind === 'leaf') {
          for (let i = 0; i < page.versions.length; i++) {
            if (page.versions[i] < 0) page.versions[i] = version;
          }
        } else {
          for (const child of page.children) {
            if (this.#drafts.has(child)) {
              stack.push(child); // retained by the parent's edge
            } else {
              // One new edge into an existing committed subtree; do NOT enter
              // it — its interior counts already balance.
              this.#pages.get(child)!.refs++;
            }
          }
        }
      }
    }
    // Sweep this transaction's drafts that the new root cannot reach (merge
    // losers, abandoned draft roots). Reachable ones become committed.
    for (const id of owner) {
      if (!reachable.has(id)) {
        this.#pages.delete(id);
        this.#freeIds.push(id);
      }
      this.#drafts.delete(id);
    }
    owner.clear();
  }

  /**
   * Add a root pin for an owner sharing an already-committed tree (open
   * snapshot, transaction base). O(1): interior edges were retained when the
   * version was published.
   */
  pinRoot(root: number): void {
    if (root === EMPTY_ROOT) return;
    const page = this.#pages.get(root);
    if (!page) throw new Error(`pinRoot: missing root ${root}`);
    page.refs++;
  }

  /**
   * Release one owner. Removes the root pin and, as each page reaches zero,
   * removes its outgoing edges (child -1); traversal stops at any page still
   * pinned by another owner. Cost is the number of pages actually reclaimed.
   */
  releaseRoot(root: number): void {
    if (root === EMPTY_ROOT) return;
    const stack: number[] = [root];
    while (stack.length) {
      const id = stack.pop()!;
      const page = this.#pages.get(id);
      if (!page) continue;
      page.refs--;
      if (page.refs > 0) continue; // shared with a live version/snapshot
      if (page.kind === 'internal') {
        for (const child of page.children) {
          const childPage = this.#pages.get(child);
          if (childPage) {
            childPage.refs--;
            if (childPage.refs === 0) stack.push(child);
          }
        }
      }
      this.#pages.delete(id);
      this.#freeIds.push(id);
    }
  }

  /** Count pages reachable from a committed root. */
  reachablePageCount(root: number): number {
    if (root === EMPTY_ROOT) return 0;
    let count = 0;
    const seen = new Set<number>();
    const stack = [root];
    while (stack.length) {
      const id = stack.pop()!;
      if (seen.has(id)) continue;
      seen.add(id);
      count++;
      const page = this.#pages.get(id)!;
      if (page.kind === 'internal') stack.push(...page.children);
    }
    return count;
  }

  /** Collect ids reachable from a committed root. */
  collectReachable(root: number, out: Set<number>): Set<number> {
    if (root === EMPTY_ROOT) return out;
    const stack = [root];
    while (stack.length) {
      const id = stack.pop()!;
      if (out.has(id)) continue;
      out.add(id);
      const page = this.#pages.get(id)!;
      if (page.kind === 'internal') stack.push(...page.children);
    }
    return out;
  }
}
