# Roadmap: Row-Level Mutations (MERGE / UPDATE / DELETE)

Status: design (not yet implemented).

This document maps out how FluxTable will support MERGE, UPDATE, and
DELETE operations on top of the v2 action-based log. The design
phases copy-on-write (COW) mutations first, then introduces
deletion-vector-based merge-on-read (MOR) as an optimisation when COW
write amplification becomes the bottleneck.

---

## Problem

FluxTable today is strictly append + overwrite. Real analytical
workloads need:
- **DELETE WHERE predicate** to retire wrong / stale / GDPR-subject
  rows.
- **UPDATE SET col = expr WHERE predicate** to patch values.
- **MERGE** to reconcile a source table into a target table
  (upserts, SCD-1 / SCD-2, streaming de-dup).

Without row-level mutations, users have to read the entire table,
filter or patch in memory, and write a brand-new table directory —
crushing storage costs on large tables and breaking concurrent
readers / writers.

## Current state

What we can build on:
- **Action-based log** (`docs/format/fluxtable_format.md`) with
  `add` and `remove` already wired up. A COW mutation is exactly
  `remove(old_file) + add(new_file)` in a single atomic commit.
- **Per-file min/max stats in manifests**. Enables predicate-based
  file-pruning: we only rewrite files that the predicate could touch.
- **Atomic OCC** via `put_if_absent`. A MERGE either lands completely
  or doesn't land at all.
- **Semantic conflict detection**. `remove` vs concurrent `add`/`remove`
  on the same file already raises `ConcurrentOperationError(kind="remove")`,
  so two MERGE writers targeting overlapping files serialise
  deterministically.
- **Txn tokens** for exactly-once retry.

What's missing:
- No writer surface for mutations — no `delete_where`, no
  `update_where`, no `merge`.
- No predicate engine on the writer side; we have predicate pushdown
  for *readers* (block-level min/max skipping) but not the symbolic
  representation a MERGE plan would build on.
- No deletion vectors → every row-level change forces a file rewrite,
  which is fine up to ~1 GB file but punishing for 10 GB+ cold files.
- No tombstone action → can't express "this row is logically deleted"
  without rewriting the whole file.

## Proposed changes

### 1. Predicate representation in the writer

Introduce a structured predicate AST shared between the Rust core
(`loom`) and Python surface:

```
# Python surface
Pred.col("user_id") == 42
Pred.col("region").isin(["us", "ca"])
Pred.col("ts") >= "2024-01-01"
Pred.and_(...)
Pred.or_(...)
Pred.not_(...)
```

Serialised into a compact tagged form for inclusion in the `remove`
and `add` actions (see below). Reuses the existing `Predicate` type
from `crates/loom/src/predicate.rs` with JSON round-trip support.

### 2. Writer surface

`FluxTableWriter` gains three methods:

```
with FluxTableWriter(path, mode="append") as w:
    n_deleted = w.delete_where(Pred.col("ts") < "2023-01-01")
    n_updated = w.update_where(
        Pred.col("region") == "xx",
        set={"region": "unknown"},
    )
    stats = w.merge(
        source=source_df,
        on=Pred.col("user_id") == Pred.src("user_id"),
        when_matched_update={"last_seen": Pred.src("ts")},
        when_not_matched_insert_all=True,
    )
```

Each call translates into:
1. **Plan**: enumerate data files the predicate could touch, using
   manifest min/max pruning. Files outside the predicate's range are
   untouched.
2. **Read**: scan matching files, apply the predicate, produce a new
   row set (minus deleted rows, with updated columns, with merged
   source rows).
3. **Write**: emit one or more new `.flux` files for the rewritten
   data.
4. **Commit**: a single log entry with paired `remove` (for every
   touched file) + `add` (for every new file), plus optional
   operation-specific actions.

MERGE is structurally the same — plan touched files, read them,
co-group with the source, emit rewritten output — just with richer
per-row logic.

### 3. New action kinds

`add` and `remove` already exist. We add operation-specific metadata
so tools can inspect the log without re-deriving semantics:

```
{
  "type": "delete",
  "predicate": <serialised predicate>,
  "files_removed": ["data/part-X.flux", ...],
  "files_added": ["data/part-Y.flux", ...],
  "rows_deleted": 12345,
  "rows_kept": 67890
}

{
  "type": "update",
  "predicate": <serialised predicate>,
  "set": {"col_a": <expr>, "col_b": <expr>},
  "files_removed": [...],
  "files_added": [...],
  "rows_updated": 4321
}

{
  "type": "merge",
  "source_signature": <hash>,
  "on": <serialised predicate>,
  "when_matched_update": {...} | null,
  "when_matched_delete": <predicate> | null,
  "when_not_matched_insert": {...} | null,
  "files_removed": [...],
  "files_added": [...],
  "rows_inserted": N, "rows_updated": M, "rows_deleted": K
}
```

These ride alongside `add` / `remove` actions in the same commit — the
`add`/`remove` pair is load-bearing; the `delete`/`update`/`merge`
action is descriptive and provenance-focused.

### 4. Conflict semantics

The existing conflict detector already catches the interesting cases:

- **Concurrent delete/update/merge touching overlapping files**:
  our intended `remove` overlaps their `add` or `remove` →
  `ConcurrentOperationError(kind="remove")`. User must re-plan.
- **Concurrent append + mutation**: append's `add` never intersects
  mutation's `remove` (mutation only removes pre-existing files). Both
  land.
- **Concurrent mutations on disjoint files**: no conflict.

No new conflict kinds needed; the primitive is strong enough.

### 5. Phased rollout

#### Phase A — DELETE via COW (smallest scope, unblocks GDPR)

- Implement predicate pruning (`files_that_could_match(predicate,
  manifests)`).
- Implement `delete_where` as: for each candidate file, decompress →
  filter → recompress → emit `remove + add`.
- Emit a `delete` action alongside.
- Files entirely pruned out (predicate is always-false on their
  stats) are skipped — zero write amplification.
- Files entirely matching (predicate is always-true on their stats,
  e.g. `DELETE WHERE ts < threshold` on a file whose max(ts) <
  threshold) are removed wholesale with no add — fastest path.

#### Phase B — UPDATE via COW

- Same skeleton as delete but with column-value rewrites inline.
- Expression engine for `set={"col": expr}` — initially only literals
  and column-references, expand to arithmetic / string ops over time.
- Predicate-pruned files are skipped.

#### Phase C — MERGE via COW

- Plan:
  1. Scan source to determine its join-key distribution and (if small)
     build an in-memory hash table.
  2. Scan target manifests to find files whose join-key range
     intersects the source's.
  3. For each candidate target file: stream its rows, look up each in
     the source, apply WHEN MATCHED / WHEN NOT MATCHED clauses.
  4. Append any WHEN NOT MATCHED source rows as a fresh file.
- The hard part is the plan; once the per-file streaming loop is
  written, UPDATE and MERGE share 90% of the code.

#### Phase D — Deletion vectors (MOR optimisation)

COW write amplification is acceptable for warm-tier tables but
punishing for archive-tier 10 GB+ files where a single-row delete
forces a full rewrite. Deletion vectors let us mark rows as deleted
without touching the data file:

- New action: `add_deletion_vector`:
  ```
  {
    "type": "add_deletion_vector",
    "for_path": "data/part-000000.flux",
    "dv_path": "deletion_vectors/dv-xxx.bin",
    "dv_size_bytes": 4096,
    "cardinality": 123,
    "format": "roaring_v1"
  }
  ```
- Per-file manifests gain `deletion_vector: {path, size, cardinality}`.
- Readers: before decoding a file's rows, load its deletion vector
  and skip marked indices. Roaring bitmap gives O(1) membership with
  tiny space cost (~0.1 bit/row for clustered deletes).
- Writers: DELETE becomes O(affected_rows) bytes written instead of
  O(file_size); UPDATE rewrites only the updated rows into a small
  new file and marks the old rows deleted via vector.
- Compaction: periodic `optimize()` call that merges deletion vectors
  back into their target files (writes a new, smaller file and
  removes the old file + its vector).

Phase D is a substantial change to the read path and introduces
non-trivial format surface. Worth shipping A–C first to validate
demand.

### 6. Read-path changes

- **Phase A–C (COW only)**: zero read-path changes. Replaying the log
  already honours `remove` + `add` as a replacement.
- **Phase D (deletion vectors)**: readers load the DV alongside each
  file and feed it into the decode loop as a row-index filter. The
  existing predicate-pushdown pipeline composes naturally: we AND the
  predicate-survivor set with the DV-survivor set.

### 7. Writer performance notes

- **Parallelism**: per-file mutations are independently processable;
  reuse the `FluxTableWriter` thread pool.
- **Pruning is the whole game**: for a table with 1 TB of data and a
  DELETE predicate touching 0.1% of rows, pruning brings the actual
  rewrite cost down to < 1 GB iff manifests carry accurate min/max
  stats. We should emit column stats **always** for columns likely to
  appear in predicates (partitioning columns, clustering columns,
  timestamp columns). `compute_stats=True` should be the default for
  tables that expect mutations.
- **Spark driver coordination**: MERGE on large target tables benefits
  from running the plan + rewrite per-file on executors. The eventual
  Scala DataSource V2 JAR will do this natively; Python-side MERGE
  will run on the driver for now (OK for < 10 GB targets).

### 8. Idempotency

Each mutation takes an optional `txn_app_id` / `txn_version` (via the
existing writer params). For MERGE in a streaming job, each epoch
tags its commit and re-runs become no-ops — same machinery as
`append` today.

## Open questions

- **Cascading WHEN MATCHED clauses**: Delta's MERGE lets you chain
  WHEN MATCHED AND condition THEN ... clauses. Do we ship that or
  single-clause only on day one?
- **Deletion vector format**: Roaring bitmaps are obvious for bulk
  deletes. For very sparse cases (< 10 rows out of millions) a plain
  sorted u32 array beats Roaring. Picking Roaring for v1 to keep the
  format tractable.
- **Time travel by version**: Delta lets you read `AS OF VERSION N`.
  We already have version-keyed log entries; exposing this on the
  reader is ~100 lines once we're comfortable with the mutation
  surface.
- **Vacuum**: retaining old data files forever bloats storage. We'll
  want a `vacuum(retention_hours=168)` that removes files whose last
  reference (via a `remove` action) is older than the retention
  window. Matches Delta semantics.

## Out of scope for this phase

- Schema evolution — covered in `docs/roadmap-schema-evolution.md`.
  A MERGE can drive a schema evolution (e.g. source has new column);
  that composition is fine but requires both features to land first.
- Time-travel reads (`AS OF VERSION N`). Listed as a follow-up.
- Change Data Feed (CDF) — emitting the diff of each commit as a
  separate read-only stream. Deferred; implementable later via the
  action log directly.
- Row-level CDC compaction across many small MERGE commits. Needs a
  dedicated compaction strategy.
