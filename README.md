# btree-mvcc-snapshot-core

A fixed-capacity, paged B-tree with **copy-on-write MVCC, snapshot reads and
reference-counted page recycling** for TypeScript. It replaces the old
"sorted array under a global write lock" index: readers take a snapshot and
never block, and a batch of config changes becomes visible to new readers
all at once.

- Zero runtime dependencies. Ships as a library (no CLI).
- Pages are allocated from and returned to one `PageStore`; `livePageCount()`
  reports how many pages are currently allocated for memory monitoring.
- Committed pages are immutable; writers build private draft pages and
  publish a new root in a single synchronous swap, so a reader can never see
  a half-applied batch.

## Install / build / test

```bash
npm install
npm test        # vitest
npm run build   # tsc -> dist/
```

## Quick start

The five methods the gateway already uses are kept as auto-commit
convenience calls on `BTree`:

```ts
import { BTree } from 'btree-mvcc-snapshot-core';

const tree = new BTree<number>({ pageSize: 32 });

tree.insert('user:1', 1);      // insert, or update if the key exists
tree.get('user:1');            // 1
tree.range('a', 'm');          // inclusive [{ key, value }, ...]
tree.delete('user:1');         // removed value, or undefined
tree.size();                   // entry count
```

## Batched writes (transactions)

A write transaction stages any number of puts/deletes. Nothing it does is
visible to anyone else until `commit()`; `rollback()` discards everything.

```ts
const txn = tree.beginTransaction();
txn.insert('k', 1);
txn.delete('other');
await txn.commit();             // the whole batch is one visible version
// txn.rollback() instead: nothing is left behind
```

Commit is serialized across transactions (several may be open at once and
their `commit()` calls can interleave across `await`s). Concurrency control
is **optimistic, first-committer-wins**:

- If the keys your batch touched were also changed by a transaction that
  committed first, `commit()` rejects with a `WriteConflictError` carrying
  the exact `conflicts` key list. Your transaction is rolled back
  automatically — none of its writes survive, and nothing is silently
  overwritten. The caller decides: retry on a fresh base, or merge.
- If the batches touched **disjoint** keys, the later committer is
  transparently **rebased** onto the newest root and both batches survive.
  This is the common case for independent config shards and means it never
  loses a write merely because another unrelated batch landed in between.

```ts
import { WriteConflictError } from 'btree-mvcc-snapshot-core';

for (;;) {
  const txn = tree.beginTransaction();
  txn.insert('k', compute(txn.get('k')));
  try { await txn.commit(); break; }
  catch (e) {
    if (e instanceof WriteConflictError) continue; // retry
    throw e;
  }
}
```

If anything throws during commit *before* the root swap, the effect is
identical to a rollback: the committed tree keeps its old root and no draft
page is published.

## Snapshot reads

```ts
const snap = tree.snapshot();
try {
  snap.get('k');
  snap.size;
  for await (const { key, value } of snap.range('a', 'z')) {
    await doOtherWork();                 // awaiting mid-scan is fine
  }
} finally {
  snap.close();
}
```

A snapshot sees exactly the data committed when it was opened, regardless of
how many splits, merges or commits happen afterwards. A snapshot opened
later sees the newest committed version. Point lookups and a range scan in
the same snapshot always agree, even if the scan suspends on an `await`
while the live tree keeps changing: every page a snapshot reaches is frozen,
and the async iterator re-checks snapshot liveness before each step.

**Always `close()` a snapshot** (or give it a timeout). Pages that an old
snapshot could still reach are never reused; once every snapshot that needs a
page is gone, that page is returned to the store.

### Snapshot timeouts

```ts
const snap = tree.snapshot(30_000);      // 30 s lifetime
// or set a default for every snapshot:
new BTree({ defaultSnapshotTimeoutMs: 30_000 });
```

Expiry is checked lazily (on read, on opening another snapshot/transaction,
and via `reapExpiredSnapshots()`). An expired snapshot releases its pages
exactly as if `close()` had been called, and **every subsequent read —
including resuming a half-finished `range` — throws `SnapshotExpiredError`**
rather than serving data. Failing loudly is deliberate: silently returning
stale/partial data from a snapshot the service meant to release is worse.

## Finding leaked snapshots

```ts
for (const s of tree.snapshotInfo()) {
  // oldest first
  log(s.id, s.label, s.ageMs, s.reachablePages, s.pinnedPages);
}
tree.reapExpiredSnapshots();
```

Each entry gives the snapshot `label` (pass one to `snapshot(ms, label)`),
its age, total reachable pages, and `pinnedPages` — old pages kept alive
only because this snapshot is still open. Close the oldest snapshots to
reclaim them. This, together with optional timeouts, covers forgotten
snapshots without a background reaper.

## Memory / reclamation model

Each page's reference count is its number of owner pins (current tree, open
snapshots, open transaction bases) plus the live committed edges pointing
into it.

- Publishing walks only the newly written **draft region** (the changed
  root-to-leaf paths); edges that leave the region into an existing
  immutable subtree add one reference and the walk stops.
- Releasing an owner decrements its root pin and then walks only pages that
  actually reach zero, stopping immediately at any page another version still
  shares. So reclaim work is proportional to the pages a commit dropped, not
  to the total entry count.
- Freed page ids are recycled by the store.

Result: with no long-lived snapshots open, `livePageCount()` settles to the
number of pages the current tree needs and stays flat across arbitrary
numbers of commits (verified over 100,000 commits). Open snapshots retain
history; close them to release it.

## Performance

Measured on this machine (Node, `pageSize: 32`) with the benchmark scripts
used during development; absolute numbers are environment-dependent but the
**scaling** is the point:

| entries | single-key commit median | single-key commit p99 | point get | open snapshot |
|--------:|-------------------------:|----------------------:|----------:|--------------:|
| 50,000  | ~8 µs | ~85 µs | ~5 µs | ~1.4 µs |
| 500,000 | ~8 µs | ~69 µs | ~6 µs | ~0.4 µs |

- A single-key commit is **O(height)** (one root-to-leaf COW path plus
  reclamation of that path), so it does not grow with total entry count.
- Opening a snapshot is a root pointer plus one pin — **independent of data
  size**.
- A point get is a normal B-tree descent, O(log n); an async `range` visits
  each leaf once, O(log n + matched keys).

### Correctness vs. those numbers — the trade-off

Reclamation is deliberately **eager and refcount-based**, not deferred to a
compaction thread: every commit does the small, proportional retain/release
walk so `livePageCount()` reflects truth immediately. That keeps memory
bounded under the "forgotten snapshot" failure mode, at the cost of a little
extra work on the write path (still microseconds). Snapshot reads never pay
for this — they only traverse immutable pages.

The fill factor is classic B-tree: a non-root page under half full borrows
from a sibling or merges; page capacity is rounded down to even so two
minimum siblings plus a separator never overflow on merge.

## API surface

- `new BTree<V>(opts?)` — `opts.pageSize` (default 32, even),
  `opts.defaultSnapshotTimeoutMs`.
- `insert`, `get`, `has`, `delete`, `range(start, end)`, `size()`.
- `beginTransaction()` → `WriteTransaction` with `insert/get/has/delete`,
  async `commit()`, `rollback()`.
- `snapshot(timeoutMs?, label?)` → `Snapshot` with `get/has/size`,
  async `range(start, end)`, `rangeArray`, `close`, `version`, `closed`.
- `livePageCount()`, `pageCount()`, `currentVersion`,
  `snapshotInfo()`, `reapExpiredSnapshots()`.
- Errors: `WriteConflictError`, `SnapshotExpiredError` (subclass of
  `SnapshotClosedError`), `SnapshotClosedError`, `TransactionClosedError`.

`range` bounds are inclusive on both ends, as in the original array index.
