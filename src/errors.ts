/**
 * Thrown when a write transaction cannot commit because an earlier committed
 * transaction modified one or more of the same keys. The transaction is rolled
 * back automatically before the error is thrown; none of its writes survive.
 */
export class WriteConflictError extends Error {
  /** Keys that changed between this transaction's base snapshot and commit. */
  readonly conflicts: readonly string[];

  constructor(conflicts: readonly string[]) {
    const preview = conflicts.slice(0, 10).join(', ');
    const more = conflicts.length > 10 ? ` (+${conflicts.length - 10} more)` : '';
    super(`write conflict on ${conflicts.length} key(s): ${preview}${more}`);
    this.name = 'WriteConflictError';
    this.conflicts = conflicts;
  }
}

/** Thrown when reading from or closing an already closed snapshot. */
export class SnapshotClosedError extends Error {
  constructor(message = 'snapshot is closed') {
    super(message);
    this.name = 'SnapshotClosedError';
  }
}

/**
 * Thrown when a snapshot with `timeoutMs` has expired. An expired snapshot is
 * released automatically (its pages become reclaimable), exactly like an
 * explicit `close()`, so every further read fails loudly instead of silently
 * serving stale data.
 */
export class SnapshotExpiredError extends SnapshotClosedError {
  constructor(message = 'snapshot has expired') {
    super(message);
    this.name = 'SnapshotExpiredError';
  }
}

/** Thrown when using a write transaction after commit or rollback. */
export class TransactionClosedError extends Error {
  constructor(message = 'transaction is closed') {
    super(message);
    this.name = 'TransactionClosedError';
  }
}
