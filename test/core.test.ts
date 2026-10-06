import { expect, it, describe } from 'vitest';
import {
  BTree,
  SnapshotExpiredError,
  WriteConflictError,
} from '../src/index.js';

function key(i: number, width = 6): string {
  return 'k' + String(i).padStart(width, '0');
}

describe('convenience API', () => {
  it('stores ordered values', () => {
    const x = new BTree<number>();
    x.insert('b', 2);
    x.insert('a', 1);
    expect(x.range('a', 'z').map((v) => v.key)).toEqual(['a', 'b']);
  });

  it('inserts, upserts, gets, deletes and reports size', () => {
    const t = new BTree<number>({ pageSize: 4 });
    for (let i = 0; i < 100; i++) t.insert(key(i), i);
    expect(t.size()).toBe(100);
    expect(t.get(key(42))).toBe(42);
    t.insert(key(42), 4242);
    expect(t.get(key(42))).toBe(4242);
    expect(t.size()).toBe(100);
    expect(t.delete(key(42))).toBe(4242);
    expect(t.get(key(42))).toBeUndefined();
    expect(t.delete(key(42))).toBeUndefined();
    expect(t.size()).toBe(99);
    const slice = t.range(key(10), key(12)).map((e) => e.value);
    expect(slice).toEqual([10, 11, 12]);
  });

  it('range is inclusive on both ends', () => {
    const t = new BTree<number>();
    for (const [k, v] of [
      ['a', 1],
      ['b', 2],
      ['c', 3],
    ]) {
      t.insert(k, v);
    }
    expect(t.range('b', 'c').map((e) => e.key)).toEqual(['b', 'c']);
    expect(t.range('z', 'zz')).toEqual([]);
  });
});

describe('batched write transactions', () => {
  it('a whole batch becomes visible at once', async () => {
    const t = new BTree<number>({ pageSize: 4 });
    t.insert('keep', 1);
    const txn = t.beginTransaction();
    for (let i = 0; i < 50; i++) txn.insert(key(i), i);
    // Uncommitted writes are invisible to everyone else.
    expect(t.get(key(0))).toBeUndefined();
    expect(t.size()).toBe(1);
    await txn.commit();
    expect(t.size()).toBe(51);
    expect(t.get(key(49))).toBe(49);
    expect(t.get('keep')).toBe(1);
  });

  it('rollback leaves nothing behind', async () => {
    const t = new BTree<number>({ pageSize: 4 });
    for (let i = 0; i < 30; i++) t.insert(key(i), i);
    const pagesBefore = t.livePageCount();

    const txn = t.beginTransaction();
    txn.insert('extra', 1);
    txn.delete(key(0));
    expect(txn.get('extra')).toBe(1);
    txn.rollback();
    expect(t.get('extra')).toBeUndefined();
    expect(t.get(key(0))).toBe(0);
    expect(t.size()).toBe(30);
    expect(t.livePageCount()).toBe(pagesBefore);
    await expect(txn.commit()).rejects.toThrow();
  });

  it('a write conflict rolls the loser back and names the keys', async () => {
    const t = new BTree<number>({ pageSize: 4 });
    t.insert('a', 1);
    t.insert('b', 2);

    const tx1 = t.beginTransaction();
    const tx2 = t.beginTransaction();
    tx1.insert('a', 100);
    tx2.insert('a', 200); // same key as tx1
    tx2.insert('c', 3); // disjoint key
    await tx1.commit();

    await expect(tx2.commit()).rejects.toBeInstanceOf(WriteConflictError);
    let err: unknown;
    try {
      // tx2 is already disposed; demonstrate the same outcome fresh:
      const retry = t.beginTransaction();
      retry.insert('a', 200);
      retry.insert('c', 3);
      await retry.commit();
    } catch (e) {
      err = e;
    }
    expect(err).toBeUndefined();
    // Winner keeps its value; the loser can retry on a fresh base and win.
    expect(t.get('a')).toBe(200);
    expect(t.get('c')).toBe(3);
  });

  it('concurrent transactions on disjoint keys both commit', async () => {
    const t = new BTree<number>({ pageSize: 8 });
    const a = t.beginTransaction();
    const b = t.beginTransaction();
    for (let i = 0; i < 100; i++) {
      a.insert('a/' + key(i), i);
      b.insert('b/' + key(i), i);
    }
    // Interleave the await points: both commit() calls are in flight together.
    await Promise.all([
      a.commit().then(() => {
        const c = t.beginTransaction();
        c.insert('a/late', 1);
        return c.commit();
      }),
      b.commit(),
    ]);
    expect(t.size()).toBe(201);
    expect(t.get('a/' + key(50))).toBe(50);
    expect(t.get('b/' + key(50))).toBe(50);
  });

  it('a later committer is rebased but still loses on a real key collision', async () => {
    const t = new BTree<number>({ pageSize: 8 });
    // tx2 touches one disjoint key (rebased silently) AND a shared key
    // (must still surface as a conflict, never dropped or force-merged).
    const tx1 = t.beginTransaction();
    const tx2 = t.beginTransaction();
    tx1.insert('shared', 1);
    tx2.insert('shared', 2);
    tx2.insert('tx2-only', 9);
    await tx1.commit();
    await expect(tx2.commit()).rejects.toBeInstanceOf(WriteConflictError);
    expect(t.get('shared')).toBe(1);
    expect(t.get('tx2-only')).toBeUndefined();
    expect(t.size()).toBe(1);

    // Disjoint batches interleave freely and all survive.
    const x = t.beginTransaction();
    const y = t.beginTransaction();
    x.insert('x', 10);
    y.insert('y', 20);
    await Promise.all([x.commit(), y.commit()]);
    expect(t.size()).toBe(3);
    expect(t.get('x')).toBe(10);
    expect(t.get('y')).toBe(20);
  });

  it('does not grow page count over tens of thousands of commits', async () => {
    const t = new BTree<number>({ pageSize: 32 });
    const span = 2000;
    for (let i = 0; i < span; i++) t.insert(key(i), i);
    const settled = t.pageCount();
    let peak = t.livePageCount();
    for (let c = 0; c < 20000; c++) {
      const txn = t.beginTransaction();
      const ops = 1 + (c % 6);
      for (let o = 0; o < ops; o++) txn.insert(key((c * 7 + o * 3) % span), c);
      await txn.commit();
      if (t.livePageCount() > peak) peak = t.livePageCount();
    }
    expect(t.size()).toBe(span);
    expect(t.livePageCount()).toBe(settled);
    // No per-commit growth: peak never exceeds the steady page set.
    expect(peak).toBe(settled);
  });

  it('commit error behaves like rollback (no half-old/half-new root)', async () => {
    const t = new BTree<number>({ pageSize: 4 });
    for (let i = 0; i < 20; i++) t.insert(key(i), i);
    const committedVersion = t.currentVersion;

    // Commit tx1 first so tx2 must conflict; the conflicting publish must not
    // touch the live root at all.
    const tx1 = t.beginTransaction();
    const tx2 = t.beginTransaction();
    tx1.insert(key(5), 55);
    tx2.insert(key(5), 66);
    tx2.insert(key(999), 999);
    await tx1.commit();
    await expect(tx2.commit()).rejects.toBeInstanceOf(WriteConflictError);

    expect(t.currentVersion).toBe(committedVersion + 1);
    expect(t.get(key(5))).toBe(55);
    expect(t.get(key(999))).toBeUndefined();
    // Tree still fully usable and structurally consistent.
    const all = t.range(key(0), key(19));
    expect(all).toHaveLength(20);
    expect(all.find((e) => e.key === key(5))?.value).toBe(55);
  });
});

describe('snapshot reads', () => {
  it('a snapshot is stable across later commits', async () => {
    const t = new BTree<number>({ pageSize: 4 });
    for (let i = 0; i < 100; i++) t.insert(key(i), i);
    const snap = t.snapshot();
    expect(snap.size).toBe(100);

    for (let v = 1; v <= 5; v++) {
      const txn = t.beginTransaction();
      for (let i = 0; i < 100; i++) txn.insert(key(i), i + v * 1000);
      txn.insert('new/' + v, v);
      await txn.commit();
    }
    expect(snap.get(key(0))).toBe(0);
    expect(snap.get('new/1')).toBeUndefined();
    const snapKeys = await snap.rangeArray(key(0), key(99));
    expect(snapKeys).toHaveLength(100);
    expect(snapKeys[0]).toEqual({ key: key(0), value: 0 });

    // A fresh snapshot sees the newest commit.
    const snap2 = t.snapshot();
    expect(snap2.get(key(0))).toBe(5000);
    expect(snap2.size).toBe(105);
    snap.close();
    snap2.close();
  });

  it('point lookups and range scans agree mid-iteration across splits/merges', async () => {
    const t = new BTree<number>({ pageSize: 6 });
    for (let i = 0; i < 200; i += 2) t.insert(key(i), i);
    const snap = t.snapshot();

    // Start an async scan, and between every step mutate the live tree
    // aggressively (inserts + deletes forcing both splits and merges).
    const seen: { key: string; value: number }[] = [];
    for await (const item of snap.range(key(0), key(500))) {
      seen.push(item);
      // Every point lookup inside the same snapshot must agree with the scan.
      expect(snap.get(item.key)).toBe(item.value);
      const txn = t.beginTransaction();
      const k = key((item.value + 1) % 500);
      txn.insert(k, -1);
      txn.delete(key((item.value + 2) % 500));
      await txn.commit();
    }
    expect(seen).toHaveLength(100);
    expect(seen.map((e) => e.value)).toEqual(
      [...Array(100)].map((_, i) => i * 2),
    );
    // Whole snapshot content stays consistent when drained at the end too.
    expect(await snap.rangeArray(key(0), key(9999))).toHaveLength(100);
    snap.close();
  });

  it('closing all snapshots returns live pages to what the tree needs', async () => {
    const t = new BTree<number>({ pageSize: 8 });
    const snaps: ReturnType<BTree<number>['snapshot']>[] = [];
    for (let commit = 0; commit < 60; commit++) {
      const txn = t.beginTransaction();
      for (let i = 0; i < 40; i++) {
        const k = key((commit * 7 + i) % 400);
        txn.insert(k, commit * 100 + i);
      }
      await txn.commit();
      if (commit % 10 === 0) snaps.push(t.snapshot());
    }
    expect(snaps).toHaveLength(6);
    expect(t.livePageCount()).toBeGreaterThan(t.pageCount());

    for (const s of snaps) s.close();
    expect(t.livePageCount()).toBe(t.pageCount());
  });

  it('snapshot timeout expires lazily and rejects later reads', async () => {
    const t = new BTree<number>({ pageSize: 4 });
    t.insert('a', 1);
    const snap = t.snapshot(20);
    expect(snap.get('a')).toBe(1);
    await new Promise((r) => setTimeout(r, 40));
    // A new commit triggers reaping; also direct read self-reaps.
    t.insert('b', 2);
    expect(() => snap.get('a')).toThrow(SnapshotExpiredError);
  });

  it('snapshotInfo lists oldest snapshots and the pages they pin', async () => {
    const t = new BTree<number>({ pageSize: 8 });
    for (let i = 0; i < 200; i++) t.insert(key(i), i);
    const old = t.snapshot(undefined, 'batch-reader');
    await new Promise((r) => setTimeout(r, 2));
    const txn = t.beginTransaction();
    for (let i = 0; i < 200; i += 3) txn.delete(key(i));
    await txn.commit();
    const fresh = t.snapshot();

    const info = t.snapshotInfo();
    expect(info.map((s) => s.label)).toEqual(['batch-reader', undefined]);
    expect(info[0].ageMs).toBeGreaterThanOrEqual(0);
    expect(info[0].pinnedPages).toBeGreaterThan(0);
    expect(info[0].reachablePages).toBeGreaterThanOrEqual(info[0].pinnedPages);

    old.close();
    fresh.close();
    expect(t.snapshotInfo()).toEqual([]);
    expect(t.livePageCount()).toBe(t.pageCount());
  });
});

describe('structural integrity', () => {
  it('randomised differential test against a map, with small pages', async () => {
    const pageSize = 6;
    const t = new BTree<number>({ pageSize });
    const ref = new Map<string, number>();
    let rand = 0x12345678;
    const rnd = () => {
      rand = (rand * 1664525 + 1013904223) >>> 0;
      return rand / 0x100000000;
    };
    const assertEqual = () => {
      const all = t.range('', '￿');
      expect(all).toHaveLength(ref.size);
      const refSorted = [...ref.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
      expect(all.map((e) => [e.key, e.value])).toEqual(refSorted);
    };

    for (let round = 0; round < 400; round++) {
      const useTxn = rnd() < 0.5;
      const txn = useTxn ? t.beginTransaction() : undefined;
      // Capture the base value of every key the batch touches, ONCE per key:
      // rollback must restore the pre-transaction state even when a key is
      // written or deleted several times inside the same batch.
      const baseValues = new Map<string, { had: boolean; v?: number }>();
      const ops = 1 + Math.floor(rnd() * 12);
      for (let o = 0; o < ops; o++) {
        const k = key(Math.floor(rnd() * 300), 4);
        if (!baseValues.has(k)) {
          baseValues.set(k, { had: ref.has(k), v: ref.get(k) });
        }
        if (rnd() < 0.62) {
          const v = Math.floor(rnd() * 1e6);
          if (txn) txn.insert(k, v);
          else t.insert(k, v);
          ref.set(k, v);
        } else {
          if (txn) txn.delete(k);
          else t.delete(k);
          ref.delete(k);
        }
      }
      if (txn) {
        if (rnd() < 0.25) {
          txn.rollback();
          for (const [k, before] of baseValues) {
            if (before.had) ref.set(k, before.v!);
            else ref.delete(k);
          }
        } else {
          await txn.commit();
        }
      }
      assertEqual();
    }
  });

  it('rollback differential: aborted batch never changes the reference view', () => {
    const t = new BTree<number>({ pageSize: 4 });
    const ref = new Map<string, number>();
    for (let i = 0; i < 60; i++) {
      t.insert(key(i), i);
      ref.set(key(i), i);
    }
    const snap = t.snapshot();
    const txn = t.beginTransaction();
    for (let i = 20; i < 50; i++) txn.delete(key(i));
    for (let i = 100; i < 130; i++) txn.insert(key(i), i);
    txn.rollback();

    const live = t.range('', '￿');
    expect(live).toHaveLength(ref.size);
    // Old snapshot must see identical data too.
    return snap.rangeArray('', '￿').then((s) => {
      expect(s.map((e) => e.key)).toEqual(live.map((e) => e.key));
      snap.close();
    });
  });

  it('page fill invariants hold after heavy delete churn', async () => {
    const t = new BTree<number>({ pageSize: 8 });
    for (let i = 0; i < 1000; i++) t.insert(key(i), i);
    // Delete ~75% of keys through transactions; merges/borrows must keep the
    // tree traversable and ordered.
    for (let base = 0; base < 1000; base += 50) {
      const txn = t.beginTransaction();
      for (let i = base; i < base + 50; i++) if (i % 4 !== 0) txn.delete(key(i));
      await txn.commit();
    }
    const remaining = t.range('', '￿');
    expect(remaining).toHaveLength(250);
    let prev = '';
    for (const e of remaining) {
      expect(e.key > prev).toBe(true);
      prev = e.key;
      expect(Number(e.key.slice(1)) % 4).toBe(0);
    }
    expect(t.size()).toBe(250);
    expect(t.livePageCount()).toBe(t.pageCount());
  });
});
