import { EMPTY_ROOT, type PageStore } from './page-store.js';

/** Result of inserting into a leaf/child: either one node id or a split. */
type InsertResult =
  | { split: false; id: number; updated: boolean }
  | { split: true; leftId: number; rightId: number; sep: string; updated: boolean };

interface DeleteResult {
  id: number;
  /** The node disappeared (internal with no children left); caller drops it. */
  emptied: boolean;
  found: boolean;
  value?: unknown;
}

/**
 * A write transaction's private working tree. All structural changes
 * (copy-on-write splits, borrows, merges) happen here against the shared
 * {@link PageStore}; the base committed root is never mutated.
 *
 * Internal-node keys are boundary values: key K in node.keys[i] means every
 * key in subtree children[i] is < K and every key in children[i+1] is >= K.
 * On a leaf split the boundary starts as a *copy* of the right leaf's first
 * key (which still lives in that leaf); if the data key is later deleted the
 * boundary copy stays valid, so deleting a key that also appears on internal
 * levels is an ordinary delete inside the right subtree — no replacement
 * dance needed.
 */
export class Workspace<V> {
  /** Working root; EMPTY_ROOT while the working tree is empty. */
  root: number;
  /** Working-set size; insert/upsert/delete adjust it. */
  size: number;

  readonly store: PageStore<V>;
  /** Ids of every draft page allocated by this transaction (owned here). */
  readonly drafts = new Set<number>();
  private readonly maxKeys: number;
  /** A non-root node with fewer than minKeys keys underflows (root exempt). */
  private readonly minKeys: number;

  constructor(store: PageStore<V>, baseRoot: number, baseSize: number, pageSize: number) {
    this.store = store;
    this.root = baseRoot;
    this.size = baseSize;
    this.maxKeys = pageSize;
    // Even capacities only: then two minimum siblings + a separator fit a
    // page exactly (minKeys + 1 + minKeys === pageSize), so a merge can never
    // overflow.
    this.minKeys = pageSize / 2;
  }

  /** Insert a new key or overwrite an existing one. `version` may be -1. */
  insert(key: string, value: V, version: number): void {
    if (this.root === EMPTY_ROOT) {
      const leaf = this.#allocLeaf();
      const p = this.store.getLeaf(leaf);
      p.keys.push(key);
      p.values.push(value);
      p.versions.push(version);
      this.root = leaf;
      this.size = 1;
      return;
    }
    const res = this.#insert(this.root, key, value, version);
    if (!res.updated) this.size++;
    if (res.split) {
      const newRoot = this.#allocInternal(res.leftId);
      const rp = this.store.getInternal(newRoot);
      rp.keys.push(res.sep);
      rp.children.push(res.rightId);
      this.root = newRoot;
    } else {
      this.root = res.id;
    }
  }

  /** Delete a key. Returns the removed value, or `{ found: false }`. */
  delete(key: string): { found: boolean; value?: V } {
    if (this.root === EMPTY_ROOT) return { found: false };
    const res = this.#deleteNode(this.root, key);
    if (res.found) this.size--;
    if (res.emptied) {
      this.root = EMPTY_ROOT;
      this.size = 0;
      return { found: res.found, value: res.value as V | undefined };
    }
    this.root = res.id;
    // Root internal node with a single child: reduce tree height. (Only the
    // root may reach this state; every non-root node keeps >= minKeys keys.)
    const rootPage = this.store.get(this.root);
    if (rootPage && rootPage.kind === 'internal' && rootPage.children.length === 1) {
      this.root = rootPage.children[0];
      // The old draft root is now unreachable; swept on commit/abort.
    }
    return { found: res.found, value: res.value as V | undefined };
  }

  /** Point lookup inside the working tree (reads drafts and shared pages). */
  get(key: string): V | undefined {
    let id = this.root;
    while (id !== EMPTY_ROOT) {
      const page = this.store.get(id)!;
      if (page.kind === 'leaf') {
        const i = lowerBound(page.keys, key);
        return i < page.keys.length && page.keys[i] === key ? page.values[i] : undefined;
      }
      let i = lowerBound(page.keys, key);
      // An equal boundary value lives inside the right subtree.
      if (i < page.keys.length && page.keys[i] === key) i++;
      id = page.children[i];
    }
    return undefined;
  }

  // ---- structural operations -------------------------------------------------

  #allocLeaf(): number {
    const id = this.store.allocLeaf();
    this.drafts.add(id);
    return id;
  }

  #allocInternal(firstChild: number): number {
    const id = this.store.allocInternal(firstChild);
    this.drafts.add(id);
    return id;
  }

  #copyForWrite(id: number): number {
    if (this.store.isDraft(id)) return id;
    const copy = this.store.clone(id);
    this.drafts.add(copy);
    return copy;
  }

  #insert(id: number, key: string, value: V, version: number): InsertResult {
    const page = this.store.get(id)!;
    if (page.kind === 'leaf') {
      const at = lowerBound(page.keys, key);
      const leafId = this.#copyForWrite(id);
      const leaf = this.store.getLeaf(leafId);
      if (at < leaf.keys.length && leaf.keys[at] === key) {
        leaf.values[at] = value;
        leaf.versions[at] = version;
        return { split: false, id: leafId, updated: true };
      }
      leaf.keys.splice(at, 0, key);
      leaf.values.splice(at, 0, value);
      leaf.versions.splice(at, 0, version);
      if (leaf.keys.length <= this.maxKeys) return { split: false, id: leafId, updated: false };
      // Split: left half stays here, right half moves to a new page; the first
      // right key becomes the parent boundary.
      const cut = Math.floor(leaf.keys.length / 2);
      const sep = leaf.keys[cut];
      const rightId = this.#allocLeaf();
      const right = this.store.getLeaf(rightId);
      right.keys = leaf.keys.splice(cut);
      right.values = leaf.values.splice(cut);
      right.versions = leaf.versions.splice(cut);
      return { split: true, leftId: leafId, rightId, sep, updated: false };
    }

    let idx = lowerBound(page.keys, key);
    if (idx < page.keys.length && page.keys[idx] === key) idx++;
    const childRes = this.#insert(page.children[idx], key, value, version);
    const nodeId = this.#copyForWrite(id);
    const node = this.store.getInternal(nodeId);
    if (!childRes.split) {
      node.children[idx] = childRes.id;
      return { split: false, id: nodeId, updated: childRes.updated };
    }
    node.children[idx] = childRes.leftId;
    node.keys.splice(idx, 0, childRes.sep);
    node.children.splice(idx + 1, 0, childRes.rightId);
    if (node.keys.length <= this.maxKeys) {
      return { split: false, id: nodeId, updated: childRes.updated };
    }
    // Internal split of a page with m keys and m+1 children (m > maxKeys).
    // Layout: C0 K0 C1 K1 ... K{m-1} Cm. K[cut] is promoted and kept by
    // neither side: left gets K[0..cut-1] with C[0..cut], right gets
    // K[cut+1..m-1] with C[cut+1..m].
    const cut = Math.floor(node.keys.length / 2);
    const upKey = node.keys[cut];
    // First right-side child is C[cut+1]; remaining right children are still
    // in node.children and are spliced out and appended (never overwrite the
    // seed child — doing so silently drops C[cut+1]).
    const rightId = this.#allocInternal(node.children[cut + 1]);
    const right = this.store.getInternal(rightId);
    right.keys = node.keys.splice(cut + 1); // K[cut+1..m-1]
    const movedChildren = node.children.splice(cut + 2); // C[cut+2..m]
    right.children.push(...movedChildren);
    node.keys.splice(cut, 1); // drop promoted K[cut] from the left side
    node.children.splice(cut + 1, 1); // left keeps only C[0..cut]
    return { split: true, leftId: nodeId, rightId, sep: upKey, updated: childRes.updated };
  }

  #deleteNode(id: number, key: string): DeleteResult {
    const page = this.store.get(id)!;
    if (page.kind === 'leaf') {
      const i = lowerBound(page.keys, key);
      if (i >= page.keys.length || page.keys[i] !== key) {
        return { id, emptied: false, found: false };
      }
      const leafId = this.#copyForWrite(id);
      const leaf = this.store.getLeaf(leafId);
      const value = leaf.values[i];
      leaf.keys.splice(i, 1);
      leaf.values.splice(i, 1);
      leaf.versions.splice(i, 1);
      return { id: leafId, emptied: leaf.keys.length === 0, found: true, value };
    }

    const idx0 = lowerBound(page.keys, key);
    const exact = idx0 < page.keys.length && page.keys[idx0] === key;
    const childIdx = exact ? idx0 + 1 : idx0;
    const childRes = this.#deleteNode(page.children[childIdx], key);
    if (!childRes.found) return { id, emptied: false, found: false };

    const nodeId = this.#copyForWrite(id);
    const node = this.store.getInternal(nodeId);
    if (childRes.emptied) {
      node.children.splice(childIdx, 1);
      if (node.children.length === 0) {
        return { id: nodeId, emptied: true, found: true, value: childRes.value };
      }
      // Drop the separator to the right of the removed slot when one exists,
      // otherwise the one on the left.
      const sepIdx = Math.min(childIdx, node.keys.length - 1);
      node.keys.splice(sepIdx, 1);
      const at = Math.min(childIdx, node.children.length - 1);
      this.#rebalance(nodeId, at);
      return { id: nodeId, emptied: false, found: true, value: childRes.value };
    }

    node.children[childIdx] = childRes.id;
    this.#rebalance(nodeId, childIdx);
    return { id: nodeId, emptied: false, found: true, value: childRes.value };
  }

  /**
   * Fix an underflowed child at `idx` of draft internal node `nodeId` by
   * borrowing from a sibling or merging with one. Uses `node.keys.length` as
   * the fill metric for both node kinds (an internal node with k keys has
   * k+1 children; leaves carry k data keys).
   */
  #rebalance(nodeId: number, idx: number): void {
    const node = this.store.getInternal(nodeId);
    const childId = node.children[idx];
    const child = this.store.get(childId)!;
    if (child.keys.length >= this.minKeys) return;

    if (idx > 0) {
      const left = this.store.get(node.children[idx - 1])!;
      if (left.keys.length > this.minKeys) {
        this.#borrowLeft(nodeId, idx);
        return;
      }
      this.#merge(nodeId, idx - 1);
      return;
    }
    if (idx + 1 < node.children.length) {
      const right = this.store.get(node.children[idx + 1])!;
      if (right.keys.length > this.minKeys) {
        this.#borrowRight(nodeId, idx);
        return;
      }
      this.#merge(nodeId, 0);
    }
  }

  #borrowLeft(nodeId: number, idx: number): void {
    const node = this.store.getInternal(nodeId);
    const childId = this.#copyForWrite(node.children[idx]);
    const leftId = this.#copyForWrite(node.children[idx - 1]);
    const sep = node.keys[idx - 1];
    const child = this.store.get(childId)!;
    const left = this.store.get(leftId)!;
    if (child.kind === 'leaf' && left.kind === 'leaf') {
      child.keys.unshift(left.keys.pop()!);
      child.values.unshift(left.values.pop()!);
      child.versions.unshift(left.versions.pop()!);
      node.keys[idx - 1] = child.keys[0];
    } else if (child.kind === 'internal' && left.kind === 'internal') {
      child.keys.unshift(sep);
      child.children.unshift(left.children.pop()!);
      node.keys[idx - 1] = left.keys.pop()!;
    }
    node.children[idx - 1] = leftId;
    node.children[idx] = childId;
  }

  #borrowRight(nodeId: number, idx: number): void {
    const node = this.store.getInternal(nodeId);
    const childId = this.#copyForWrite(node.children[idx]);
    const rightId = this.#copyForWrite(node.children[idx + 1]);
    const sep = node.keys[idx];
    const child = this.store.get(childId)!;
    const right = this.store.get(rightId)!;
    if (child.kind === 'leaf' && right.kind === 'leaf') {
      child.keys.push(right.keys.shift()!);
      child.values.push(right.values.shift()!);
      child.versions.push(right.versions.shift()!);
      node.keys[idx] = right.keys[0];
    } else if (child.kind === 'internal' && right.kind === 'internal') {
      child.keys.push(sep);
      child.children.push(right.children.shift()!);
      node.keys[idx] = right.keys.shift()!;
    }
    node.children[idx] = childId;
    node.children[idx + 1] = rightId;
  }

  /**
   * Merge children[at] and children[at+1] through boundary keys[at]. The
   * caller (workspace.delete) collapses the root separately if it is left
   * with a single child; non-root nodes keep enough keys to avoid that state.
   */
  #merge(nodeId: number, at: number): void {
    const node = this.store.getInternal(nodeId);
    const sep = node.keys[at];
    const leftId = this.#copyForWrite(node.children[at]);
    const rightId = this.#copyForWrite(node.children[at + 1]);
    const left = this.store.get(leftId)!;
    const right = this.store.get(rightId)!;
    if (left.kind === 'leaf' && right.kind === 'leaf') {
      left.keys.push(...right.keys);
      left.values.push(...right.values);
      left.versions.push(...right.versions);
    } else if (left.kind === 'internal' && right.kind === 'internal') {
      left.keys.push(sep, ...right.keys);
      left.children.push(...right.children);
    }
    node.children[at] = leftId;
    node.keys.splice(at, 1);
    node.children.splice(at + 1, 1);
    // rightId is now an unreachable draft; swept from the store on commit or
    // abort (it was never wired into any other edge).
  }
}

/** First index i with keys[i] >= target (binary search). */
export function lowerBound(keys: readonly string[], target: string): number {
  let lo = 0;
  let hi = keys.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (keys[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

interface Frame {
  id: number;
  /** At an internal frame: next child slot on the right to visit. */
  index: number;
}

/**
 * Synchronous in-order cursor over one immutable committed root. It stays
 * coherent regardless of concurrent commits because every page it touches is
 * frozen; the MVCC layer wraps it in an async iterator and re-checks snapshot
 * liveness before each step (so callers may `await` freely mid-scan).
 */
export class RangeCursor<V> {
  private readonly store: PageStore<V>;
  private readonly end: string;
  private readonly stack: Frame[] = [];

  constructor(store: PageStore<V>, root: number, start: string, end: string) {
    this.store = store;
    this.end = end;
    if (root !== EMPTY_ROOT) this.#seek(root, start);
  }

  /** Follow the path to the leaf that may contain the first key >= bound. */
  #seek(id: number, bound: string): void {
    for (;;) {
      const page = this.store.get(id)!;
      if (page.kind === 'leaf') {
        this.stack.push({ id, index: lowerBound(page.keys, bound) });
        return;
      }
      let i = lowerBound(page.keys, bound);
      if (i < page.keys.length && page.keys[i] === bound) i++;
      this.stack.push({ id, index: i + 1 }); // next sibling slot to visit
      id = page.children[i];
    }
  }

  /** Push the path down to the leftmost leaf of `id`. */
  #seekLeftmost(id: number): void {
    for (;;) {
      const page = this.store.get(id)!;
      if (page.kind === 'leaf') {
        this.stack.push({ id, index: 0 });
        return;
      }
      this.stack.push({ id, index: 1 });
      id = page.children[0];
    }
  }

  next(): { key: string; value: V } | undefined {
    while (this.stack.length) {
      const frame = this.stack[this.stack.length - 1];
      const page = this.store.get(frame.id)!;
      if (page.kind === 'leaf') {
        if (frame.index >= page.keys.length) {
          this.stack.pop(); // parent frame already points past this child
          continue;
        }
        const key = page.keys[frame.index];
        if (key > this.end) return undefined;
        const value = page.values[frame.index];
        frame.index++;
        return { key, value };
      }
      if (frame.index >= page.children.length) {
        this.stack.pop();
        continue;
      }
      const childId = page.children[frame.index++];
      this.#seekLeftmost(childId);
    }
    return undefined;
  }
}
