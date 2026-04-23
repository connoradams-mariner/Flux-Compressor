// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Row-level mutations via copy-on-write (Parts 3 / 4 / 5 of the
//! mutations roadmap).
//!
//! All three operations share the same skeleton:
//!
//! 1. Snapshot the table to find the live files.
//! 2. For each candidate file, decompress into a [`RecordBatch`],
//!    evaluate a row-level mask, and decide what to do with it:
//!    skip, remove entirely, or rewrite.
//!  3. Commit a single log entry that pairs every `remove` with its
//!    matching `add` (recompressed file), plus a lightweight action
//!    payload describing the operation for downstream tooling.
//!
//! DELETE filters the batch with `arrow::compute::filter_record_batch`.
//! UPDATE walks the `set` map, replacing masked rows in each column
//! with a broadcast scalar via `arrow::compute::if_then_else`.
//! MERGE builds a hash map from the source's join-key column, walks
//! each target file row-by-row, and applies the matched / not-matched
//! clauses.
//!
//! The legacy `data_files_added` / `data_files_removed` fields in the
//! log entry remain authoritative, so pre-mutations readers keep
//! resolving the live file set correctly. The richer
//! [`MutationAction`] payload rides alongside in `actions` for tools
//! that want provenance.
//!
//! File-level pruning uses the per-file
//! [`FileManifest::column_stats`] when available — files whose
//! `[min, max]` range proves the predicate is always-false are
//! skipped entirely (zero read amplification), and files where the
//! predicate is always-true are removed wholesale without a rewrite.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::{DataType, Schema};
use serde::{Deserialize, Serialize};

use crate::compressors::flux_writer::FluxWriter;
use crate::decompressors::flux_reader::FluxReader;
use crate::error::{FluxError, FluxResult};
use crate::traits::{LoomCompressor, Predicate};

// ─────────────────────────────────────────────────────────────────────────────
// Scalar values used by UPDATE / MERGE
// ─────────────────────────────────────────────────────────────────────────────

/// Literal scalar value used in `UPDATE ... SET col = scalar` and
/// `MERGE ... WHEN MATCHED UPDATE SET col = scalar` clauses.
///
/// Only primitive literals are supported (matching the roadmap scope);
/// expression evaluation is deferred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ScalarValue {
    /// `NULL`.
    Null,
    /// Boolean literal.
    Bool(bool),
    /// Signed integer fitting in `i64`.
    Int(i64),
    /// Unsigned integer fitting in `u64`.
    UInt(u64),
    /// 64-bit float.
    Float(f64),
    /// UTF-8 string literal.
    Text(String),
}

impl ScalarValue {
    /// Broadcast this scalar into an [`ArrayRef`] of length `len` whose
    /// [`DataType`] matches `target` so it can be `if_then_else`-merged
    /// with an existing Arrow column.
    pub fn to_array(&self, target: &DataType, len: usize) -> FluxResult<ArrayRef> {
        use arrow_array::{
            BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
            StringArray, UInt32Array, UInt64Array,
        };
        let mismatch = |want: &str| {
            FluxError::Internal(format!(
                "ScalarValue::{self:?} is incompatible with column dtype {target:?}; \
                 expected {want}"
            ))
        };
        let arr: ArrayRef = match (self, target) {
            (ScalarValue::Null, _) => arrow_array::new_null_array(target, len),
            (ScalarValue::Bool(v), DataType::Boolean) => {
                Arc::new(BooleanArray::from(vec![*v; len]))
            }
            (ScalarValue::Int(v), DataType::Int32)  => Arc::new(Int32Array::from(vec![*v as i32; len])),
            (ScalarValue::Int(v), DataType::Int64)  => Arc::new(Int64Array::from(vec![*v; len])),
            (ScalarValue::UInt(v), DataType::UInt32) => Arc::new(UInt32Array::from(vec![*v as u32; len])),
            (ScalarValue::UInt(v), DataType::UInt64) => Arc::new(UInt64Array::from(vec![*v; len])),
            (ScalarValue::Float(v), DataType::Float32) => Arc::new(Float32Array::from(vec![*v as f32; len])),
            (ScalarValue::Float(v), DataType::Float64) => Arc::new(Float64Array::from(vec![*v; len])),
            (ScalarValue::Text(s), DataType::Utf8)     => Arc::new(StringArray::from(vec![s.as_str(); len])),
            (ScalarValue::Bool(_), _)  => return Err(mismatch("Boolean")),
            (ScalarValue::Int(_),  _)  => return Err(mismatch("Int32/Int64")),
            (ScalarValue::UInt(_), _)  => return Err(mismatch("UInt32/UInt64")),
            (ScalarValue::Float(_), _) => return Err(mismatch("Float32/Float64")),
            (ScalarValue::Text(_), _)  => return Err(mismatch("Utf8")),
        };
        Ok(arr)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-entry side-band action records
// ─────────────────────────────────────────────────────────────────────────────

/// Rich provenance payload stamped alongside a mutation's
/// `remove` + `add` pair in the transaction log.
///
/// Each variant serialises with a `type` key of
/// `"mutation_delete" | "mutation_update" | "mutation_merge"` so the
/// existing [`super::log_entry::Action`] enum's `#[serde(other)]`
/// branch (`Action::Unknown`) silently tolerates it on reads from
/// older Rust versions that don't know these kinds yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MutationAction {
    /// DELETE via COW — rows matching `predicate_repr` were removed.
    MutationDelete {
        predicate_repr: String,
        rows_deleted: u64,
        rows_kept: u64,
        files_rewritten: u64,
        files_removed_entirely: u64,
    },
    /// UPDATE via COW — rows matching `predicate_repr` had `set`
    /// applied column-wise.
    MutationUpdate {
        predicate_repr: String,
        set: HashMap<String, ScalarValue>,
        rows_updated: u64,
        files_rewritten: u64,
    },
    /// MERGE via COW — hash-join against `source`, applying the
    /// supplied matched / not-matched clauses.
    MutationMerge {
        on_column: String,
        rows_inserted: u64,
        rows_updated: u64,
        rows_deleted: u64,
        files_rewritten: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats returned to the caller
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct DeleteStats {
    pub rows_deleted: u64,
    pub rows_kept: u64,
    pub files_rewritten: u64,
    pub files_removed_entirely: u64,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    pub rows_updated: u64,
    pub files_rewritten: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    pub rows_inserted: u64,
    pub rows_updated: u64,
    pub rows_deleted: u64,
    pub files_rewritten: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// MERGE clauses
// ─────────────────────────────────────────────────────────────────────────────

/// Action applied to target rows whose join key matched a source row.
#[derive(Debug, Clone)]
pub enum MatchedAction {
    /// Replace the listed columns with the source row's values.
    UpdateFromSource(Vec<String>),
    /// Delete the matched row.
    Delete,
}

/// Action applied to source rows that found no matching target row.
#[derive(Debug, Clone)]
pub enum NotMatchedAction {
    /// Insert the source row verbatim.
    Insert,
}

#[derive(Debug, Clone, Default)]
pub struct MergeClauses {
    pub when_matched: Option<MatchedAction>,
    pub when_not_matched: Option<NotMatchedAction>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Read a live file at `full_path`, producing its [`RecordBatch`]
/// (without any predicate pushdown).
pub(crate) fn read_file_batch(full_path: &std::path::Path) -> FluxResult<RecordBatch> {
    FluxReader::default().decompress_file(full_path, &Predicate::None)
}

/// Recompress a batch with the default writer settings.
pub(crate) fn compress_batch(batch: &RecordBatch) -> FluxResult<Vec<u8>> {
    FluxWriter::new().compress(batch)
}

/// Stringify a predicate for storage in the log entry's
/// [`MutationAction`]. Intentionally lossy — enough to tell which
/// commit is which in `inspect` output.
pub(crate) fn predicate_repr(p: &Predicate) -> String {
    match p {
        Predicate::None => "TRUE".into(),
        Predicate::GreaterThan { column, value } => format!("{column} > {value}"),
        Predicate::LessThan    { column, value } => format!("{column} < {value}"),
        Predicate::Equal       { column, value } => format!("{column} = {value}"),
        Predicate::Between     { column, lo, hi } => format!("{column} BETWEEN {lo} AND {hi}"),
        Predicate::EqualStr    { column, value } => format!("{column} = '{value}'"),
        Predicate::InStr { column, values } => {
            let s = values.iter().map(|v| format!("'{v}'")).collect::<Vec<_>>().join(", ");
            format!("{column} IN ({s})")
        }
        Predicate::And(a, b) => format!("({} AND {})", predicate_repr(a), predicate_repr(b)),
        Predicate::Or(a, b)  => format!("({} OR {})",  predicate_repr(a), predicate_repr(b)),
    }
}

/// Count the number of `true`s in a boolean mask (nulls treated as
/// `false`, matching our `eval_on_batch` semantics).
#[inline]
pub(crate) fn count_true(mask: &BooleanArray) -> u64 {
    (0..mask.len())
        .filter(|&i| !mask.is_null(i) && mask.value(i))
        .count() as u64
}

/// Invert a boolean mask (null-propagating: nulls stay null-like, i.e.
/// they become `false`). We prefer explicit non-null booleans here.
pub(crate) fn invert_mask(mask: &BooleanArray) -> BooleanArray {
    let n = mask.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(mask.is_null(i) || !mask.value(i));
    }
    BooleanArray::from(out)
}

/// Apply `mask` to `batch`, returning only rows where the mask is `true`.
pub(crate) fn filter_batch(batch: &RecordBatch, mask: &BooleanArray) -> FluxResult<RecordBatch> {
    let filtered_cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            arrow::compute::filter(c, mask)
                .map_err(|e| FluxError::Internal(format!("filter: {e}")))
        })
        .collect::<FluxResult<Vec<_>>>()?;
    RecordBatch::try_new(batch.schema(), filtered_cols)
        .map_err(|e| FluxError::Internal(format!("filter_batch: {e}")))
}

/// Concatenate two [`RecordBatch`]es with identical schema.
pub(crate) fn concat_batches(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
) -> FluxResult<RecordBatch> {
    arrow::compute::concat_batches(schema, batches)
        .map_err(|e| FluxError::Internal(format!("concat_batches: {e}")))
}

/// Walk `set`, replacing masked rows in the listed columns with a
/// scalar broadcast of the new value. Columns not in `set` pass
/// through untouched.
pub(crate) fn apply_update_set(
    batch: &RecordBatch,
    mask: &BooleanArray,
    set: &HashMap<String, ScalarValue>,
) -> FluxResult<RecordBatch> {
    let schema = batch.schema();
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        if let Some(scalar) = set.get(field.name()) {
            // Build a broadcast of the scalar in the column's dtype,
            // then branch cell-wise with `zip` (if_then_else).
            let new_vals = scalar.to_array(field.data_type(), col.len())?;
            let merged = arrow::compute::kernels::zip::zip(mask, &new_vals, col)
                .map_err(|e| FluxError::Internal(format!("zip: {e}")))?;
            new_cols.push(merged);
        } else {
            new_cols.push(col.clone());
        }
    }
    RecordBatch::try_new(schema, new_cols)
        .map_err(|e| FluxError::Internal(format!("apply_update_set: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration-style tests for delete_where / update_where / merge
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use arrow_array::{Int64Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;

    use crate::compressors::flux_writer::FluxWriter;
    use crate::traits::LoomCompressor;
    use crate::txn::FluxTable;

    fn sample_batch(offset: i64, n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("score", DataType::Int64, false),
        ]));
        let ids: UInt64Array = (0..n).map(|i| (offset as u64) + i as u64).collect();
        let scores: Int64Array = (0..n).map(|i| (i as i64) * 10).collect();
        RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(scores)]).unwrap()
    }

    #[test]
    fn delete_where_partial_rewrite() {
        let dir = TempDir::new().unwrap();
        let tbl = FluxTable::open(dir.path()).unwrap();

        // Append 100 rows in one file.
        let batch = sample_batch(0, 100);
        let bytes = FluxWriter::new().compress(&batch).unwrap();
        tbl.append(&bytes).unwrap();

        // Delete rows where id > 50 — 50 rows match, 50 rows kept.
        let pred = Predicate::GreaterThan { column: "id".into(), value: 50 };
        let stats = tbl.delete_where(&pred).unwrap();

        assert_eq!(stats.rows_deleted, 49, "id > 50 matches ids 51..=99");
        assert_eq!(stats.rows_kept, 51, "ids 0..=50 kept");
        assert_eq!(stats.files_rewritten, 1);
        assert_eq!(stats.files_removed_entirely, 0);

        // Live file count should be exactly 1 (old removed, new added).
        let live = tbl.live_files().unwrap();
        assert_eq!(live.len(), 1);
    }

    #[test]
    fn delete_where_zero_matches_is_noop() {
        let dir = TempDir::new().unwrap();
        let tbl = FluxTable::open(dir.path()).unwrap();

        let batch = sample_batch(0, 10);
        let bytes = FluxWriter::new().compress(&batch).unwrap();
        tbl.append(&bytes).unwrap();

        // Nothing matches id > 1_000_000.
        let pred = Predicate::GreaterThan { column: "id".into(), value: 1_000_000 };
        let stats = tbl.delete_where(&pred).unwrap();

        assert_eq!(stats.rows_deleted, 0);
        assert_eq!(stats.files_rewritten, 0);
        assert_eq!(stats.files_removed_entirely, 0);
    }

    #[test]
    fn update_where_replaces_matched_rows() {
        let dir = TempDir::new().unwrap();
        let tbl = FluxTable::open(dir.path()).unwrap();

        let batch = sample_batch(0, 50);
        let bytes = FluxWriter::new().compress(&batch).unwrap();
        tbl.append(&bytes).unwrap();

        let mut set = HashMap::new();
        set.insert("score".to_string(), ScalarValue::Int(-1));

        let pred = Predicate::LessThan { column: "id".into(), value: 10 };
        let stats = tbl.update_where(&pred, set).unwrap();

        assert_eq!(stats.rows_updated, 10);
        assert_eq!(stats.files_rewritten, 1);
    }

    #[test]
    fn scalar_value_broadcasts_correctly() {
        let arr = ScalarValue::Int(42).to_array(&DataType::Int64, 3).unwrap();
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(arr.len(), 3);
        for i in 0..3 {
            assert_eq!(arr.value(i), 42);
        }
    }

    #[test]
    fn scalar_type_mismatch_errors() {
        // Int scalar against a Utf8 column — should error, not panic.
        let err = ScalarValue::Int(1).to_array(&DataType::Utf8, 3).unwrap_err();
        assert!(err.to_string().contains("expected"));
    }

    #[test]
    fn merge_update_from_source_updates_matched_rows() {
        let dir = TempDir::new().unwrap();
        let tbl = FluxTable::open(dir.path()).unwrap();

        // Target: ids 0..10, scores 0..90 (step 10).
        let target = sample_batch(0, 10);
        let bytes = FluxWriter::new().compress(&target).unwrap();
        tbl.append(&bytes).unwrap();

        // Source: update score for ids 2 and 5.
        let schema = target.schema();
        let ids: UInt64Array = [2u64, 5u64].iter().copied().collect();
        let scores: Int64Array = [999i64, 888i64].iter().copied().collect();
        let source = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(scores)],
        ).unwrap();

        let clauses = MergeClauses {
            when_matched: Some(MatchedAction::UpdateFromSource(vec!["score".into()])),
            when_not_matched: None,
        };
        let stats = tbl.merge(&source, "id", clauses).unwrap();
        assert_eq!(stats.rows_updated, 2);
        assert_eq!(stats.files_rewritten, 1);
    }

    #[test]
    fn merge_delete_matched_removes_rows() {
        let dir = TempDir::new().unwrap();
        let tbl = FluxTable::open(dir.path()).unwrap();

        let target = sample_batch(0, 10);
        let bytes = FluxWriter::new().compress(&target).unwrap();
        tbl.append(&bytes).unwrap();

        let schema = target.schema();
        let ids: UInt64Array = [1u64, 3u64, 7u64].iter().copied().collect();
        let scores: Int64Array = [0i64, 0i64, 0i64].iter().copied().collect();
        let source = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(scores)],
        ).unwrap();

        let clauses = MergeClauses {
            when_matched: Some(MatchedAction::Delete),
            when_not_matched: None,
        };
        let stats = tbl.merge(&source, "id", clauses).unwrap();
        assert_eq!(stats.rows_deleted, 3);
        assert_eq!(stats.files_rewritten, 1);
    }

    #[test]
    fn merge_not_matched_insert_appends_new_rows() {
        let dir = TempDir::new().unwrap();
        let tbl = FluxTable::open(dir.path()).unwrap();

        let target = sample_batch(0, 5);  // ids 0..=4
        let bytes = FluxWriter::new().compress(&target).unwrap();
        tbl.append(&bytes).unwrap();

        // Source: ids 3, 4, 100, 200 — the first two match, the last two
        // should get inserted.
        let schema = target.schema();
        let ids: UInt64Array = [3u64, 4u64, 100u64, 200u64].iter().copied().collect();
        let scores: Int64Array = [0i64; 4].iter().copied().collect();
        let source = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(scores)],
        ).unwrap();

        let clauses = MergeClauses {
            when_matched: None,
            when_not_matched: Some(NotMatchedAction::Insert),
        };
        let stats = tbl.merge(&source, "id", clauses).unwrap();
        assert_eq!(stats.rows_inserted, 2);
    }
}
