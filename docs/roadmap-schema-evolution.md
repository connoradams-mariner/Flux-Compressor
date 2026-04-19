# Roadmap: Schema Evolution

Status: design (not yet implemented).

This document maps out how FluxTable will handle evolving schemas —
adding columns, dropping columns, promoting types, renaming — without
rewriting existing data files. It builds on the v2 action-based log
introduced in `docs/format/fluxtable_format.md`.

---

## Problem

Real-world tables change shape over time: new columns are added to
capture new signals, old columns are deprecated or dropped, numeric
types are promoted from `int32` to `int64`, enums gain new variants.
Today FluxTable stores a single `current_spec_id` pointing at a
partition spec but has no first-class notion of a column schema. The
per-file manifests carry column stats keyed by name, and the Atlas
footer inside each `.flux` file records the columns it holds, but
nothing ties those together into a logical table schema.

Without schema evolution:
- Every reader has to union-scan every file and hope the on-disk
  layouts are compatible. A dropped-and-re-added column becomes
  ambiguous.
- Adding a column forces a table rewrite because readers can't
  distinguish "old file that never had the column" from "new file
  where the column happens to be NULL".
- Type promotion is impossible: widening `int32 → int64` on an
  existing column either breaks readers or requires a full rewrite.

## Current state

What we can build on:
- **Action-based log** (`docs/format/fluxtable_format.md`). Every
  commit carries a list of actions; adding a new `schema` action kind
  is additive and keeps pre-v2 readers working via the legacy flat
  fields.
- **Checkpoint + meta_snapshot**. Checkpoints embed the current
  metadata, so schema reconstruction on read costs the same as any
  other metadata replay.
- **Protocol versioning**. We already gate unsafe format changes
  behind `FLUX_TABLE_READER_VERSION` / `FLUX_TABLE_WRITER_VERSION`.
- **Per-file manifests** in `file_manifests[*]` → already keyed by
  `path`, already carry `spec_id`. Adding `schema_id` is a single
  optional field.
- **Metadata conflict detection**. Concurrent writers that both try to
  evolve the schema already land in `ConcurrentOperationError(kind="metadata")`.

What's missing:
- No canonical schema object in the log.
- No `schema_id` on file manifests.
- No writer-side validation that incoming batches match the declared
  schema.
- No reader-side projection layer that maps a file's schema to the
  reader's expected schema (NULL-fill dropped columns, default-fill
  added columns, promote types).

## Proposed changes

### 1. New action: `schema`

Extend the log's action vocabulary. A `schema` action is the
authoritative record of the table's logical columns at a given
version.

```
{
  "type": "schema",
  "schema_id": 3,
  "fields": [
    {
      "field_id": 1,
      "name": "user_id",
      "dtype": "uint64",
      "nullable": false,
      "default": null,
      "doc": null
    },
    {
      "field_id": 2,
      "name": "region",
      "dtype": "utf8",
      "nullable": true,
      "default": "unknown",
      "doc": "User's IP-derived region code."
    },
    {
      "field_id": 3,
      "name": "revenue_cents",
      "dtype": "int64",         // promoted from int32 at schema_id 2
      "nullable": true,
      "default": null,
      "doc": null,
      "promoted_from": {"schema_id": 2, "dtype": "int32"}
    }
  ],
  "parent_schema_id": 2,
  "change_summary": "added region; promoted revenue from int32"
}
```

Field IDs are the immutable logical identifier for a column — names
can change, types can be promoted under compatibility rules, but the
field_id is stable across the table's lifetime. This lets rename
operations work without rewriting data.

`schema_id` increases monotonically. Every `schema` action must
reference its `parent_schema_id` so readers can walk the evolution
chain when presented with an older file.

### 2. Add `schema_id` to file manifests

File manifests gain an optional `schema_id` field:

```
{
  "path": "data/part-000000.flux",
  "spec_id": 0,
  "schema_id": 3,          // NEW
  "partition_values": {...},
  ...
}
```

Writers stamp each new data file with the current schema_id.
Pre-evolution files have no `schema_id`; readers treat them as
`schema_id = 0`.

### 3. Metadata action carries `current_schema_id`

The existing `metadata` action is extended with `current_schema_id`
alongside `current_spec_id`:

```
{
  "type": "metadata",
  "table_id": "...",
  "partition_specs": [...],
  "current_spec_id": 0,
  "current_schema_id": 3,   // NEW
  "clustering_columns": [...],
  "properties": {...}
}
```

### 4. Supported evolution operations

The implementation lands in a single sweep covering the operations
that don't require data rewrites:

| Operation | Mechanism | Compatibility |
|---|---|---|
| **Add nullable column** | New `schema` action with extra field; `default: null`. Old files read as NULL for the new column. | Forward- and backward-compatible. |
| **Add column with default** | Same, but readers synthesise the default on read. | Forward- and backward-compatible. |
| **Drop column** | New `schema` action without the field. Readers ignore the physical column in older files. | Old files stay readable; downgrade of a reader to an older schema is unsafe without rewrite. |
| **Rename column** | New `schema` action with the same `field_id` but different `name`. | Fully backward-compatible because reads resolve by field_id, not name. |
| **Promote numeric type** | New `schema` action with the widened `dtype` + `promoted_from` metadata. Reader applies the widening on decode. | Permitted promotions: u/i8 → u/i16 → u/i32 → u/i64; f32 → f64; decimal precision up; utf8 length up. Narrowing is rejected. |
| **Reorder columns** | Free because we read by `field_id`. No log action needed unless the reorder is user-visible. | N/A |
| **Change nullability** | Permitted only for nullable → non-nullable iff the writer has validated no existing rows are NULL. Enforced at commit time. | Otherwise rejected with `SchemaEvolutionError`. |

Unsupported (require rewrite in Phase 2 — see mutations doc):
- Type narrowing (int64 → int32).
- Changing partition columns' type.
- Changing a column's field_id (== drop-and-add).

### 5. Writer-side validation

`FluxTableWriter` gains a `schema=` constructor parameter and an
`evolve_schema()` method.

```
with FluxTableWriter(path, schema=new_schema, mode="append") as w:
    w.evolve_schema(new_schema)  # commits a schema action
    w.write_batch(batch)         # batches are validated / projected to new schema
```

Validation at commit:
- New field_ids must not collide with existing ones.
- Type promotions must match the compatibility matrix above.
- Nullability tightening requires `allow_null_tightening=True` **and**
  a writer-supplied proof (typically the column's null_count across
  the live set is zero, which we can derive from file manifests).

### 6. Reader-side projection

Readers replay all `schema` actions → build an evolution chain keyed
by `schema_id`. When scanning a file with `schema_id = S`:

1. Load the file's physical schema from its Atlas footer.
2. Look up schema `S` in the evolution chain.
3. For each field in the caller's requested output schema:
   - If the field exists at schema `S`: decode it from the file,
     possibly widening on the fly for promoted types.
   - If the field was added after `S`: synthesise a column of the
     declared default (NULL for nullable, `default:` value otherwise).
   - If the field was dropped before `S` but exists in the output
     schema (rare, typically caller-supplied): error out with
     `FieldMissingError`.
4. Apply the rename mapping from field_id → output name.

Projection happens *during* decode, not after, so we never materialise
a wide intermediate buffer. For numeric promotions this means the
BitSlab / DeltaDelta decoders gain a target-dtype parameter.

### 7. Conflict detection

Schema evolution rides the existing metadata-vs-metadata conflict
path. Because `schema` is a metadata-class action, two writers racing
to evolve the schema serialise automatically — one wins, the other
gets `ConcurrentOperationError(kind="metadata")` and must rebase.

Append writers that don't evolve the schema are unaffected; they emit
no `schema` action and conflict only on `remove`.

### 8. Backward compatibility

- Tables that pre-date this change have no `schema` action. Readers
  treat the set of columns found in the first `.flux` file as the
  implicit `schema_id = 0`. All file manifests are assumed to carry
  that schema.
- A new writer operating on a pre-evolution table emits a `schema`
  action at its first commit that evolves metadata, capturing the
  implicit schema.
- Readers that pre-date this change ignore `schema` actions and
  resolve columns by name (current behaviour). Any table that actually
  exercises evolution (e.g., rename + add) must bump
  `writer_min_version` in its `protocol` action so older readers fail
  cleanly instead of silently mis-reading.

### 9. Phased rollout

- **Phase A — schema action + per-file schema_id** (small, safe).
  Writers start stamping files. Readers learn to replay `schema`
  actions and pick the right one per file. No evolution operations
  yet; this is just plumbing.
- **Phase B — add / drop / rename** (default-null only). Implementable
  without touching the decode pipeline; everything is NULL synthesis.
- **Phase C — type promotion**. Requires decoder changes across the
  Rust pipeline to accept a target dtype. Can land incrementally
  (start with i32 → i64, add float widening, then decimal).
- **Phase D — nullability tightening with proof**. Requires writer to
  consult manifests for null_count evidence before committing the
  evolution.

## Open questions

- **Default values for non-nullable adds**: do we allow arbitrary
  expressions (`now()`, `uuid()`) or only literals? Literals are
  simpler and what Delta supports for `ADD COLUMN ... DEFAULT`.
- **Schema fingerprinting**: should we hash the field_id-sorted fields
  list for de-dup so two identical schemas don't bloat the log? Not
  strictly needed but nice for cheap equality checks.
- **Cross-file schema stats**: column_stats in file manifests are
  keyed by name today; we'll need to key them by `field_id` (or carry
  both) so stats survive renames.
- **Partition column evolution**: Iceberg lets you evolve partition
  specs independently of the data schema. We already have that split
  (`spec_id` vs `schema_id`), but the CLI / docs don't yet expose it
  cleanly. Design TBD.

## Out of scope for this phase

- MERGE / UPDATE / DELETE — covered in `docs/roadmap-mutations.md`.
- Views, materialised views, or computed-column schemas.
- Nested type evolution (struct / list field adds/drops). Doable with
  the same field_id machinery extended recursively; deferred until we
  see demand.
- Cross-table schema registries (shared between multiple tables).
