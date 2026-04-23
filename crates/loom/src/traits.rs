// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Core trait definitions that every compression strategy must satisfy.

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
};
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use crate::error::{FluxError, FluxResult};

// ─────────────────────────────────────────────────────────────────────────────
// Predicate (for pushdown)
// ─────────────────────────────────────────────────────────────────────────────

/// A simple predicate used for predicate pushdown during decompression.
///
/// The decompressor evaluates this against each block's Atlas metadata
/// before touching the compressed bytes, skipping irrelevant blocks entirely.
#[derive(Debug, Clone)]
pub enum Predicate {
    /// No filtering – return all rows.
    None,
    /// `column > value`
    GreaterThan { column: String, value: i128 },
    /// `column < value`
    LessThan { column: String, value: i128 },
    /// `column == value`
    Equal { column: String, value: i128 },
    /// `column BETWEEN lo AND hi`
    Between { column: String, lo: i128, hi: i128 },
    /// Part 9: `column == string_value` for Utf8 columns.
    EqualStr { column: String, value: String },
    /// Part 9: `column IN (values...)` for Utf8 columns.
    InStr { column: String, values: Vec<String> },
    /// Logical AND of two predicates.
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR of two predicates.
    Or(Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// Returns `true` when the block whose `[min, max]` range *might* contain
    /// rows satisfying the predicate (optimistic / conservative check).
    pub fn may_overlap(&self, min: i128, max: i128) -> bool {
        match self {
            Predicate::None => true,
            Predicate::GreaterThan { value, .. } => max > *value,
            Predicate::LessThan { value, .. } => min < *value,
            Predicate::Equal { value, .. } => min <= *value && *value <= max,
            Predicate::Between { lo, hi, .. } => min <= *hi && max >= *lo,
            // String predicates are always optimistic at the u128 block
            // level — the caller should use [`may_overlap_str`] when it
            // has access to the lexicographic block min/max.
            Predicate::EqualStr { .. } | Predicate::InStr { .. } => true,
            Predicate::And(a, b) => a.may_overlap(min, max) && b.may_overlap(min, max),
            Predicate::Or(a, b) => a.may_overlap(min, max) || b.may_overlap(min, max),
        }
    }

    /// Part 9: block-level overlap check for **string** predicates.
    ///
    /// `block_min` and `block_max` are the first `N` bytes of the
    /// block's lexicographic min / max (Flux stores these as `u128`
    /// so we pass the full 16-byte prefix here). Numeric variants
    /// always return `true` — callers route those through
    /// [`may_overlap`] instead.
    pub fn may_overlap_str(&self, block_min: &[u8], block_max: &[u8]) -> bool {
        match self {
            Predicate::None => true,
            Predicate::EqualStr { value, .. } => {
                let v = value.as_bytes();
                // value is in [min, max] iff min ≤ v ≤ max (byte-lex).
                // Use prefix comparison: if the block's min exceeds v or
                // the block's max is less than v, skip. Since we only
                // store a 16-byte prefix, this is conservative.
                prefix_lex_le(block_min, v) && prefix_lex_le(v, block_max)
            }
            Predicate::InStr { values, .. } => {
                values.iter().any(|v| {
                    let b = v.as_bytes();
                    prefix_lex_le(block_min, b) && prefix_lex_le(b, block_max)
                })
            }
            Predicate::And(a, b) => a.may_overlap_str(block_min, block_max)
                                 && b.may_overlap_str(block_min, block_max),
            Predicate::Or(a, b)  => a.may_overlap_str(block_min, block_max)
                                 || b.may_overlap_str(block_min, block_max),
            _ => true,  // numeric variants always pass through
        }
    }

    /// Row-level evaluation of the predicate against an Arrow [`RecordBatch`].
    ///
    /// Returns a [`BooleanArray`] with `true` at every row that satisfies
    /// the predicate, `false` otherwise.  Nulls propagate as `false` —
    /// matching SQL semantics where NULL ≠ anything.
    ///
    /// This is the shared primitive used by the mutations roadmap
    /// (`delete_where`, `update_where`, `merge`): a file is decompressed
    /// to a [`RecordBatch`], `eval_on_batch` builds the mask, and then
    /// `arrow::compute::filter_record_batch` or `if_else` carries out
    /// the row-level mutation.
    ///
    /// Phase A only implements the numeric predicate variants (those that
    /// compare against an `i128` scalar); string predicates are added in
    /// Part 9 of the roadmap.
    pub fn eval_on_batch(&self, batch: &RecordBatch) -> FluxResult<BooleanArray> {
        let n = batch.num_rows();
        match self {
            Predicate::None => Ok(BooleanArray::from(vec![true; n])),
            Predicate::GreaterThan { column, value } =>
                eval_cmp_scalar(batch, column, *value, CmpOp::Gt),
            Predicate::LessThan { column, value } =>
                eval_cmp_scalar(batch, column, *value, CmpOp::Lt),
            Predicate::Equal { column, value } =>
                eval_cmp_scalar(batch, column, *value, CmpOp::Eq),
            Predicate::Between { column, lo, hi } => {
                // BETWEEN is inclusive on both ends — materialise as
                // `(col >= lo) AND (col <= hi)`.
                let ge = eval_cmp_scalar(batch, column, *lo, CmpOp::Ge)?;
                let le = eval_cmp_scalar(batch, column, *hi, CmpOp::Le)?;
                Ok(and_boolean(&ge, &le))
            }
            Predicate::EqualStr { column, value } =>
                eval_string_equal(batch, column, std::slice::from_ref(value)),
            Predicate::InStr { column, values } =>
                eval_string_equal(batch, column, values.as_slice()),
            Predicate::And(a, b) => {
                let la = a.eval_on_batch(batch)?;
                let lb = b.eval_on_batch(batch)?;
                Ok(and_boolean(&la, &lb))
            }
            Predicate::Or(a, b) => {
                let la = a.eval_on_batch(batch)?;
                let lb = b.eval_on_batch(batch)?;
                Ok(or_boolean(&la, &lb))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CmpOp { Gt, Lt, Eq, Ge, Le }

impl CmpOp {
    #[inline]
    fn apply<T: PartialOrd>(self, a: &T, b: &T) -> bool {
        match self {
            CmpOp::Gt => a >  b,
            CmpOp::Lt => a <  b,
            CmpOp::Eq => a == b,
            CmpOp::Ge => a >= b,
            CmpOp::Le => a <= b,
        }
    }
}

/// Evaluate `column <op> scalar` over a [`RecordBatch`] using Arrow's
/// typed downcasts.  Any column whose dtype we don't recognise returns
/// a clear [`FluxError`] rather than silently skipping rows.
fn eval_cmp_scalar(
    batch: &RecordBatch,
    column: &str,
    scalar: i128,
    op: CmpOp,
) -> FluxResult<BooleanArray> {
    let idx = batch.schema().index_of(column).map_err(|e| {
        FluxError::Internal(format!("predicate column '{column}' not in batch: {e}"))
    })?;
    let arr = batch.column(idx);
    let n = arr.len();

    // We build the BooleanArray manually so we can honour nullability
    // (null rows produce `false`, matching SQL semantics).
    let mut out = Vec::with_capacity(n);
    match arr.data_type() {
        DataType::Int32 => {
            let a = arr.as_primitive::<arrow_array::types::Int32Type>();
            let s = scalar as i32;
            for i in 0..n {
                out.push(!a.is_null(i) && op.apply(&a.value(i), &s));
            }
        }
        DataType::Int64 => {
            let a: &Int64Array = arr.as_primitive();
            let s = scalar as i64;
            for i in 0..n {
                out.push(!a.is_null(i) && op.apply(&a.value(i), &s));
            }
        }
        DataType::UInt32 => {
            let a: &UInt32Array = arr.as_primitive();
            let s = scalar as u32;
            for i in 0..n {
                out.push(!a.is_null(i) && op.apply(&a.value(i), &s));
            }
        }
        DataType::UInt64 => {
            let a: &UInt64Array = arr.as_primitive();
            let s = scalar as u64;
            for i in 0..n {
                out.push(!a.is_null(i) && op.apply(&a.value(i), &s));
            }
        }
        DataType::Float32 => {
            let a: &Float32Array = arr.as_primitive();
            let s = scalar as f32;
            for i in 0..n {
                out.push(!a.is_null(i) && op.apply(&a.value(i), &s));
            }
        }
        DataType::Float64 => {
            let a: &Float64Array = arr.as_primitive();
            let s = scalar as f64;
            for i in 0..n {
                out.push(!a.is_null(i) && op.apply(&a.value(i), &s));
            }
        }
        other => {
            return Err(FluxError::Internal(format!(
                "Predicate::eval_on_batch does not yet support column '{column}' of type {other:?}; string predicates are Part 9 of the roadmap"
            )));
        }
    };
    // Safety: `out.len() == n`.
    Ok(BooleanArray::from(out))
}

/// Part 9: evaluate `column IN (values...)` (or `==` when `values` is a
/// one-element slice) over a [`StringArray`] column.
fn eval_string_equal(
    batch: &RecordBatch,
    column: &str,
    values: &[String],
) -> FluxResult<BooleanArray> {
    let idx = batch.schema().index_of(column).map_err(|e| {
        FluxError::Internal(format!("predicate column '{column}' not in batch: {e}"))
    })?;
    let arr = batch.column(idx);
    let n = arr.len();

    let strings: &StringArray = arr
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            FluxError::Internal(format!(
                "EqualStr / InStr require a Utf8 column; got {:?}",
                arr.data_type()
            ))
        })?;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if strings.is_null(i) {
            out.push(false);
            continue;
        }
        let v = strings.value(i);
        out.push(values.iter().any(|needle| needle.as_str() == v));
    }
    Ok(BooleanArray::from(out))
}

/// Byte-lex comparison `a ≤ b` with prefix-safe semantics.
///
/// Flux stores only the first 16 bytes of each block's min / max string;
/// short keys that appear in a block may compare "less" than their
/// true value if the comparand extends past the prefix. To stay
/// conservative we treat `a ≤ b` as true whenever `a`'s bytes are a
/// prefix of `b`'s bytes up to the shorter length.
#[inline]
fn prefix_lex_le(a: &[u8], b: &[u8]) -> bool {
    let n = a.len().min(b.len());
    let cmp = a[..n].cmp(&b[..n]);
    match cmp {
        std::cmp::Ordering::Less    => true,
        std::cmp::Ordering::Equal   => true,  // prefix-equal — conservative
        std::cmp::Ordering::Greater => false,
    }
}

/// Element-wise logical AND of two boolean arrays (null → false).
fn and_boolean(a: &BooleanArray, b: &BooleanArray) -> BooleanArray {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(
            !a.is_null(i) && !b.is_null(i)
            && a.value(i) && b.value(i),
        );
    }
    BooleanArray::from(out)
}

/// Element-wise logical OR (null → false).
fn or_boolean(a: &BooleanArray, b: &BooleanArray) -> BooleanArray {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(
            (!a.is_null(i) && a.value(i))
            || (!b.is_null(i) && b.value(i)),
        );
    }
    BooleanArray::from(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// LoomCompressor
// ─────────────────────────────────────────────────────────────────────────────

/// Compress an Arrow [`RecordBatch`] into raw `.flux` bytes.
///
/// Implementations are expected to:
/// 1. Run the Loom classifier on each column.
/// 2. Write the chosen compressed block(s).
/// 3. Append the Atlas metadata footer.
/// 4. Return the complete byte buffer (zero-copy where possible).
pub trait LoomCompressor: Send + Sync {
    /// Compress `batch` into a self-contained `.flux` byte buffer.
    fn compress(&self, batch: &RecordBatch) -> FluxResult<Vec<u8>>;

    /// Compress multiple batches into a single `.flux` file, allowing global
    /// dictionary deduplication across batches (used by the Cold optimizer).
    fn compress_all(&self, batches: &[RecordBatch]) -> FluxResult<Vec<u8>> {
        // Default: concatenate independently compressed blocks.
        // Override for global-dictionary / Z-Order optimisation.
        let mut out = Vec::new();
        for batch in batches {
            out.extend(self.compress(batch)?);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoomDecompressor
// ─────────────────────────────────────────────────────────────────────────────

/// Decompress `.flux` bytes (optionally with predicate pushdown) into an Arrow
/// [`RecordBatch`].
pub trait LoomDecompressor: Send + Sync {
    /// Decompress `data` with optional predicate pushdown.
    ///
    /// The decompressor reads the Atlas footer first to enumerate block
    /// metadata, skips blocks that cannot satisfy `predicate`, and only
    /// decompresses blocks that may contain matching rows.
    fn decompress(&self, data: &[u8], predicate: &Predicate) -> FluxResult<RecordBatch>;

    /// Convenience wrapper: decompress without any predicate.
    fn decompress_all(&self, data: &[u8]) -> FluxResult<RecordBatch> {
        self.decompress(data, &Predicate::None)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FluxCapacitor (CLI / optimiser contract)
// ─────────────────────────────────────────────────────────────────────────────

/// The CLI-level contract for the `fluxcapacitor` binary.
///
/// Responsible for two-pass global optimisation, Z-Order interleaving, and
/// merging Hot blocks into Cold archives.
pub trait FluxCapacitorTrait: Send + Sync {
    /// **Pass 1** – scan all `.flux` partitions under `input_dir` and build
    /// a global master dictionary + statistics.
    fn scan_partitions(&mut self, input_dir: &std::path::Path) -> FluxResult<()>;

    /// **Pass 2** – re-pack every block using the global master dictionary,
    /// apply Z-Order interleaving, and write the optimised archive to
    /// `output_path`.
    fn optimize(&self, output_path: &std::path::Path) -> FluxResult<()>;

    /// Merge two or more `.flux` files into one, updating the Atlas footer.
    fn merge(
        &self,
        inputs: &[&std::path::Path],
        output: &std::path::Path,
    ) -> FluxResult<()>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod predicate_eval_tests {
    use super::*;
    use std::sync::Arc;
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![
                Some(1), Some(5), Some(10), None, Some(15),
            ]))],
        )
        .unwrap()
    }

    #[test]
    fn gt_masks_rows_correctly() {
        let b = batch();
        let p = Predicate::GreaterThan { column: "x".into(), value: 5 };
        let mask = p.eval_on_batch(&b).unwrap();
        // Expect: [false, false, true, false(null), true]
        let got: Vec<bool> = (0..mask.len()).map(|i| mask.value(i)).collect();
        assert_eq!(got, vec![false, false, true, false, true]);
    }

    #[test]
    fn between_inclusive() {
        let b = batch();
        let p = Predicate::Between { column: "x".into(), lo: 5, hi: 10 };
        let mask = p.eval_on_batch(&b).unwrap();
        let got: Vec<bool> = (0..mask.len()).map(|i| mask.value(i)).collect();
        assert_eq!(got, vec![false, true, true, false, false]);
    }

    #[test]
    fn and_or_composition() {
        let b = batch();
        let gt = Predicate::GreaterThan { column: "x".into(), value: 0 };
        let lt = Predicate::LessThan    { column: "x".into(), value: 10 };
        let both = Predicate::And(Box::new(gt.clone()), Box::new(lt.clone()));
        let mask = both.eval_on_batch(&b).unwrap();
        let got: Vec<bool> = (0..mask.len()).map(|i| mask.value(i)).collect();
        assert_eq!(got, vec![true, true, false, false, false]);

        let either = Predicate::Or(Box::new(gt), Box::new(lt));
        let mask2 = either.eval_on_batch(&b).unwrap();
        let got2: Vec<bool> = (0..mask2.len()).map(|i| mask2.value(i)).collect();
        assert_eq!(got2, vec![true, true, true, false, true]);
    }

    #[test]
    fn none_matches_all() {
        let b = batch();
        let p = Predicate::None;
        let mask = p.eval_on_batch(&b).unwrap();
        let got: Vec<bool> = (0..mask.len()).map(|i| mask.value(i)).collect();
        assert_eq!(got, vec![true; 5]);
    }
}
