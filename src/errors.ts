/** 快照已被 close（或超过 ttl 被自动关闭）后再次使用时抛出 */
export class SnapshotClosedError extends Error {
  readonly id: number;
  constructor(id: number, timedOut: boolean) {
    super(
      timedOut
        ? `snapshot ${id} has been auto-closed after its ttl expired`
        : `snapshot ${id} is already closed`,
    );
    this.name = 'SnapshotClosedError';
    this.id = id;
  }
}

/**
 * 提交时发现本事务基于的版本之后，别的事务已经写过同一批键。
 * 不会静默丢写：调用方需要决定重试（begin 一个新事务）还是放弃。
 */
export class WriteConflictError extends Error {
  readonly baseVersion: number;
  readonly committedVersion: number;
  /** 与已提交事务发生冲突的键 */
  readonly conflictingKeys: string[];
  constructor(baseVersion: number, committedVersion: number, conflictingKeys: string[]) {
    super(
      `write conflict: transaction based on version ${baseVersion} conflicts on keys [${conflictingKeys
        .slice(0, 10)
        .map((k) => JSON.stringify(k))
        .join(', ')}${conflictingKeys.length > 10 ? ', ...' : ''}] already committed through version ${committedVersion}`,
    );
    this.name = 'WriteConflictError';
    this.baseVersion = baseVersion;
    this.committedVersion = committedVersion;
    this.conflictingKeys = conflictingKeys;
  }
}
