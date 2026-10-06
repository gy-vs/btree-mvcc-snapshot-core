/**
 * btree-mvcc-snapshot-core
 *
 * Fixed-capacity, paged B-tree with copy-on-write MVCC, snapshot reads and
 * refcounted page recycling. Zero runtime dependencies.
 *
 * Quick start (convenience auto-commit API):
 *
 *   const tree = new BTree<number>();
 *   tree.insert('a', 1);
 *   tree.get('a');
 *   tree.range('a', 'z');
 *   tree.delete('a');
 *   tree.size();
 *
 * Batched writes and snapshot reads:
 *
 *   const snap = tree.snapshot();
 *   try {
 *     for await (const { key, value } of snap.range('a', 'z')) { ... }
 *   } finally { snap.close(); }
 *
 *   const txn = tree.beginTransaction();
 *   txn.insert('k', 1);
 *   await txn.commit();       // whole batch visible at once
 *   // txn.rollback() discards everything
 */

export {
  BTree,
  Snapshot,
  WriteTransaction,
  PageStore,
  WriteConflictError,
  SnapshotClosedError,
  SnapshotExpiredError,
  TransactionClosedError,
} from './btree.js';

export type { BTreeOptions, SnapshotInfo, KV } from './btree.js';
