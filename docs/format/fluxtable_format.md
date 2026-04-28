# .fluxtable Directory Format

This document specifies the directory layout of `.fluxtable` tables
produced by `FluxTableWriter`, the structure of its transaction log,
and the concurrency guarantees that the writer and reader uphold. For
the per-file on-disk format of individual `.flux` data files, see
`flux_format.md`.

---

## Overview

A `.fluxtable` is a directory with four artifacts:

```
my_table.fluxtable/
├── _flux_meta.json              # derived metadata cache (best-effort)
├── _flux_log/
│   ├── 00000000.json            # incremental transaction entries
│   ├── 00000001.json
│   ├── ...
│   ├── 00000009.checkpoint.json # consolidated snapshot (optional)
│   └── ...
├── _last_checkpoint.json        # pointer to the most recent checkpoint (optional)
└── data/
    ├── part-000000.flux
    ├── part-000001.flux
    └── ...
```

The transaction log is **authoritative**. `_flux_meta.json` is a
derived cache maintained on a best-effort basis; readers that do not
trust the cache reconstruct table state by replaying the log (plus any
checkpoint).

---

## Transaction Log Entry (v2)

Each `_flux_log/NNNNNNNN.json` is a JSON object with the following
schema:

| Field | Type | Description |
|---|---|---|
| `version` | `u64` | Monotonic log version, zero-indexed. Matches the filename. |
| `timestamp_ms` | `u64` | UTC commit time in milliseconds. |
| `operation` | `string` | One of `"create"`, `"append"`, `"delete"`, `"compact"`, `"metadata"`. Derived from `actions`. |
| `parent_version_crc32` | `u32` | CRC32 of the raw bytes of entry `version - 1`, or `0` for v0 / unknown. |
| `writer_id` | `string` | Opaque identifier for the writer session. Defaults to a UUID. |
| `writer_version` | `u32` | Writer protocol version (currently 2). |
| `actions` | `[Action]` | Canonical semantics; see below. |
| `data_files_added` / `data_files_removed` / `file_manifests` / `row_count_delta` | legacy | Populated alongside `actions` so pre-v2 readers can still load the table. |

### Action kinds

The canonical semantics of a commit are expressed by its `actions`
array, evaluated in order:

| `type` | Payload | Purpose |
|---|---|---|
| `protocol` | `reader_min_version`, `writer_min_version` | Advertises the minimum reader/writer protocol versions required. Emitted at v0 of a fresh table. |
| `metadata` | `table_id`, `partition_specs[]`, `current_spec_id`, `clustering_columns[]`, `properties{}` | Authoritative table metadata. Emitted whenever the caller changes metadata; later `metadata` actions completely replace prior ones. |
| `add` | `path`, `manifest` | A new data file. `manifest` mirrors the legacy `file_manifests` entry (row_count, file_size_bytes, column_stats, spec_id, partition_values). |
| `remove` | `path` | Retires a data file from the live set. |
| `txn` | `app_id`, `version` | Idempotency marker tying this commit to an external retry-safe job (e.g. Spark Structured Streaming). |
| `commit_info` | `writer_id`, `writer_version`, `operation`, `row_count_delta` | Provenance; always last. |

Unknown action kinds must be ignored by readers so forward-compatible
fields can be added without a hard break.

---

## Optimistic Concurrency Control (OCC)

`FluxTableWriter` supports concurrent writers against the same table.
Safety is provided by three layers.

### 1. Atomic log-entry claim (`put_if_absent`)

Every commit claims its version slot with a backend-native atomic
conditional write:

- **Local filesystem**: `os.link(tmp, target)` — fails with
  `FileExistsError` if the destination already exists.
- **Google Cloud Storage**: `blob.upload_from_string(bytes,
  if_generation_match=0)` via the `google-cloud-storage` SDK. Rejected
  with `PreconditionFailed` if an object already exists.
- **Amazon S3**: `client.put_object(..., IfNoneMatch='*')` via `boto3`.
  Rejected with `PreconditionFailed` / `412` if the key already exists.
- **Other fsspec backends**: `exists()` probe + write + `mv`.
  Best-effort; a one-time warning is emitted so operators can diagnose
  weak concurrency guarantees.

If the cloud SDK is unavailable the writer falls back to the
exists-then-write path and warns. For GCS / S3 production deployments,
install `google-cloud-storage` / `boto3` alongside the fsspec wrapper.

### 2. Semantic conflict detection

After discovering that entries landed while this writer was preparing,
the commit routine classifies those concurrent commits against the
actions it is about to publish:

- **`remove` vs `add`/`remove` on the same path** — hard conflict.
  Raises `ConcurrentOperationError(kind="remove", ...)` immediately; no
  retry. The caller must rebuild because the file they wanted to retire
  was touched by someone else.
- **`metadata` vs `metadata`** — if the bodies differ the commit is a
  hard conflict (`kind="metadata"`). If the caller did not explicitly
  configure metadata (no `partition_by` / `clustering_columns` argument),
  the writer instead **rebases** onto the concurrent metadata silently
  and drops its own `metadata` action — this is the append-only
  fresh-table case where two writers race to emit the initial spec.
- **`protocol` vs `protocol`** with differing versions — hard conflict
  (`kind="protocol"`).
- **Duplicate `txn(app_id, version)`** — hard conflict (`kind="txn"`).
  Separate from the pre-commit txn-dedup check that catches the common
  case before any work starts.
- **Append vs append** — always compatible. The writer just retries at
  the next free version.

### 3. Parent-CRC chain

Every entry carries `parent_version_crc32 = CRC32(bytes of entry
version - 1)`. On read, `_replay_log` verifies the chain as it walks
the log. A mismatch raises `LogForkError` by default; pass
`strict=False` to replay up to the last valid version for repair
workflows. The CRC chain guards against a theoretical write-race on
backends without native conditional-PUT (where two writers could both
successfully `put` at the same version slot during a TOCTOU window).

### Recovery

Data part files use unique names (counter-based per writer session, or
`task-prefix-NNNNNN` for Spark executors) so OCC retries never need to
rewrite bytes. If `_MAX_COMMIT_RETRIES` (default 10) is exhausted
without a successful claim, `CommitConflictError` is raised. Orphan
part files from a failed commit session are safe to delete or sweep
into a later `append` session.

---

## Idempotent Transactions

Pass `txn_app_id` and `txn_version` to `FluxTableWriter` when
committing from a retry-safe job orchestrator:

```
with FluxTableWriter(
    path,
    mode="append",
    txn_app_id="my-spark-stream",
    txn_version=epoch_id,
) as w:
    w.write_batch(...)
```

Semantics:

- Before committing, the writer scans the log for the highest
  `txn.version` belonging to `app_id`. If `prior >= txn_version` the
  commit becomes a silent no-op.
- On successful commit, a `txn` action is appended alongside the data
  changes.
- `txn_app_id` and `txn_version` must be provided together; passing one
  without the other raises `ValueError`.

Used correctly, this gives Spark Structured Streaming and similar
retry-heavy pipelines exactly-once semantics on top of FluxTable.

---

## Metadata Authoritativeness (Phase 4)

Table metadata (schema, partition specs, clustering columns,
properties) lives in two places:

- **Canonical**: `metadata` actions in the log. Reader reconstructs the
  current state by replaying all `metadata` actions (or loading a
  checkpoint's embedded `meta_snapshot`).
- **Cache**: `_flux_meta.json` at the table root. Refreshed after each
  successful commit on a best-effort basis. Readers may use the cache
  for fast paths but should fall back to the log on any discrepancy.

Writers only emit a `metadata` action when they actually change
metadata relative to the log-derived baseline. That keeps append-only
logs short and avoids spurious metadata-vs-metadata conflicts when
multiple writer sessions are pure-append.

---

## Checkpoints (Phase 1/6)

To bound reader cost on long-lived tables, `FluxTableWriter` optionally
emits a checkpoint every `checkpoint_interval` commits (default 10, set
`0` to disable). A checkpoint materialises the live set at a given
version so readers can skip past N incremental entries.

### Checkpoint file

`_flux_log/NNNNNNNN.checkpoint.json`:

```
{
  "version": 9,
  "timestamp_ms": 1730000000000,
  "is_checkpoint": true,
  "live_files": ["data/part-000000.flux", ..., "data/part-000004.flux"],
  "file_manifests": [ /* one per live file */ ],
  "meta_snapshot": { /* contents of the authoritative metadata */ }
}
```

The checkpoint file is written via `put_if_absent`; if another writer
already emitted the same version, the race loser silently no-ops.

### Pointer file

`_last_checkpoint.json` at the table root names the most recent
checkpoint:

```
{
  "version": 9,
  "path": "_flux_log/00000009.checkpoint.json"
}
```

The pointer is maintained by a read-modify-write loop that refuses to
regress the version on disk: if writer A published a pointer at v9, a
concurrent writer publishing v4 is dropped. On object stores without
CAS, this is best-effort but the worst outcome is a single stale
pointer, which readers handle gracefully.

### Reader contract

`_replay_log(fs, log_dir, table_path=path)`:

1. Reads `_last_checkpoint.json` (if present) and loads the checkpoint
   it names. `live_files` / `file_manifests` / `meta_snapshot` seed the
   initial state.
2. Iterates `_flux_log/NNNNNNNN.json` entries sorted by version,
   validating the CRC chain and applying actions strictly after
   `checkpoint.version`.
3. Raises `ProtocolVersionError` if the log declares a required
   `reader_min_version` higher than this client implements.
4. Returns `(live_files, manifests)` for the reader.

Checkpoint files end in `.checkpoint.json`; `_next_version` and
`_list_log_entries` skip them so they never contribute a version slot
and OCC claim allocation is independent of checkpoint cadence.

---

## Protocol Versioning

Every fresh-table v0 entry emits a `protocol` action that declares the
minimum reader and writer protocol versions required to operate on the
table.

- Current constants: `FLUX_TABLE_READER_VERSION = 1`,
  `FLUX_TABLE_WRITER_VERSION = 2`.
- Readers refuse tables whose highest observed `reader_min_version`
  exceeds their compiled-in value (`ProtocolVersionError`).
- Bumping these numbers signals a breaking format change; older
  clients will reject the table rather than silently misread it.

---

## Backward Compatibility

- **Pre-v2 tables** (no `actions` array): readers fall back to
  `data_files_added` / `data_files_removed` / `file_manifests`.
  Writers always emit both shapes in v2+ entries, so old readers
  continue to load new tables.
- **No `_last_checkpoint.json`**: readers replay from v0.
- **Missing `_flux_meta.json`**: readers reconstruct metadata from the
  log's `metadata` actions.
- **Legacy `parent_version_crc32 == 0`**: CRC validation skips entries
  whose declared CRC is zero (the pre-v2 default).

---

## Versioning Summary

The `.fluxtable` container is versioned by the `protocol` action in
the log. The individual `.flux` data files are versioned via the
`FLUX` magic in their footer; see `flux_format.md`. Changes that
require a bump of the reader or writer protocol version:

- New action kinds whose absence would silently mis-interpret history.
- Changes to CRC computation, manifest shape, or the on-disk checkpoint
  format.
- Removal of legacy flat fields (requires a coordinated reader update
  first).
