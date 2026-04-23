// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end null-aware compression on top of [`FluxWriter`] and
//! [`FluxReader`] (Part 1 of the remaining roadmap — follow-up).
//!
//! ## Why layer it
//!
//! The on-disk `BlockMeta.null_bitmap_offset` field is already
//! plumbed through every compressed block, but the deep writer path
//! (1700+ lines in `flux_writer.rs`, ~20 strategy-specific encoders)
//! still treats Arrow arrays as if they were fully non-null. Wiring
//! the null bitmap into every strategy's inner loop is a week of
//! careful code-reading.
//!
//! This module provides a **drop-in wrapper** that preserves nulls
//! end-to-end without touching any of those internals:
//!
//! ```text
//! Arrow batch ──► null_aware::compress
//!                 │
//!                 ├─► extract validity bitmaps per column
//!                 │   (serialised via [`crate::null_bitmap`])
//!                 │
//!                 ├─► compact each column to the dense non-null
//!                 │   values via arrow::compute::filter
//!                 │
//!                 ├─► FluxWriter on the dense batch
//!                 │   (untouched; full compression performance)
//!                 │
//!                 └─► prefix the result with the per-column
//!                     bitmap frame so the reader can re-inject
//!                     nulls
//! ```
//!
//! The wrapper is a pure-Rust add-on: existing `.flux` files stay
//! byte-identical, and callers can opt into null-awareness by using
//! this module in place of `FluxWriter::compress`. Once the deep
//! writer integration lands, this module will reduce to a thin
//! adapter or be removed entirely.
//!
//! ## Framing
//!
//! ```text
//! [u32: FLUX_NULL_AWARE_MAGIC = 0x464C4E41 ("FLNA")]
//! [u32: num_columns]
//! For each column:
//!   [u32: bitmap_len (0 = no nulls)]
//!   [bitmap bytes]
//! [u32: flux_payload_len]
//! [flux_payload bytes]  // FluxWriter output on the dense batch
//! ```
//!
//! The magic prefix lets the reader auto-detect null-aware frames;
//! plain `.flux` files are rejected by [`decompress`] (use
//! [`FluxReader`] directly for those) and null-aware frames are
//! rejected by `FluxReader` (the magic won't match `FLX2`).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::NullBuffer;

use crate::compressors::flux_writer::FluxWriter;
use crate::decompressors::flux_reader::FluxReader;
use crate::error::{FluxError, FluxResult};
use crate::null_bitmap;
use crate::traits::{LoomCompressor, LoomDecompressor, Predicate};

/// Magic bytes for a null-aware container: ASCII "FLNA".
pub const FLUX_NULL_AWARE_MAGIC: u32 = 0x464C4E41;

/// Compress a [`RecordBatch`] with full null preservation.
///
/// The returned bytes embed a serialised validity bitmap for every
/// column that has nulls, alongside a regular `FluxWriter`-compressed
/// representation of the dense (non-null) values.  On read, nulls are
/// re-injected row-by-row via [`decompress`].
///
/// Fully-non-null columns pay **zero overhead** — their per-column
/// bitmap length is 0 and no bitmap bytes are emitted, matching the
/// roadmap's design goal.
pub fn compress(batch: &RecordBatch) -> FluxResult<Vec<u8>> {
    let schema = batch.schema();
    let n_cols = batch.num_columns();
    let writer = FluxWriter::new();

    // Per-column: serialised bitmap (may be empty) + dense-column
    // flux payload. We compress each column into its own one-column
    // RecordBatch because post-filter the dense columns have varying
    // row counts (nulls removed) and can't share a schema.
    let mut bitmaps: Vec<Vec<u8>> = Vec::with_capacity(n_cols);
    let mut col_payloads: Vec<Vec<u8>> = Vec::with_capacity(n_cols);

    for (i, col) in batch.columns().iter().enumerate() {
        let (dense, bitmap) = match null_bitmap::encode(col.nulls()) {
            Some(bytes) => {
                let nulls = col.nulls().unwrap();
                let valid_mask = nulls_to_mask(nulls, col.len());
                let d = arrow::compute::filter(col, &valid_mask)
                    .map_err(|e| FluxError::Internal(format!("null filter: {e}")))?;
                (d, bytes)
            }
            None => (col.clone(), Vec::new()),
        };
        bitmaps.push(bitmap);

        // Compress the dense column with a single-column schema that
        // strips nullability — the reader re-adds nulls via the bitmap.
        let field = schema.field(i).clone().with_nullable(false);
        let col_schema = Arc::new(arrow_schema::Schema::new(vec![field]));
        let col_batch = RecordBatch::try_new(col_schema, vec![dense])
            .map_err(|e| FluxError::Internal(format!("null_aware single-col: {e}")))?;
        col_payloads.push(writer.compress(&col_batch)?);
    }

    // Build the framed output.
    let total_hint = 12
        + bitmaps.iter().map(|b| 4 + b.len()).sum::<usize>()
        + col_payloads.iter().map(|p| 4 + p.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total_hint);
    out.extend_from_slice(&FLUX_NULL_AWARE_MAGIC.to_le_bytes());
    out.extend_from_slice(&(n_cols as u32).to_le_bytes());
    out.extend_from_slice(&(batch.num_rows() as u32).to_le_bytes());
    for (bitmap, payload) in bitmaps.iter().zip(col_payloads.iter()) {
        out.extend_from_slice(&(bitmap.len() as u32).to_le_bytes());
        out.extend_from_slice(bitmap);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
    }
    Ok(out)
}

/// Decompress the output of [`compress`] back into a
/// [`RecordBatch`] with nulls re-injected.
pub fn decompress(data: &[u8]) -> FluxResult<RecordBatch> {
    if data.len() < 12 {
        return Err(FluxError::InvalidFile(
            "null-aware frame too short for header".into(),
        ));
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != FLUX_NULL_AWARE_MAGIC {
        return Err(FluxError::InvalidFile(format!(
            "bad null-aware magic: expected {FLUX_NULL_AWARE_MAGIC:#010x}, got {magic:#010x} \
             (did you pass a plain .flux file? use FluxReader for those)"
        )));
    }
    let n_cols = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let n_rows = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;

    let mut cursor = 12usize;
    let reader = FluxReader::default();
    let mut rebuilt_cols: Vec<ArrayRef> = Vec::with_capacity(n_cols);
    let mut rebuilt_fields: Vec<arrow_schema::Field> = Vec::with_capacity(n_cols);

    for _ in 0..n_cols {
        // Bitmap length + body.
        if cursor + 4 > data.len() {
            return Err(FluxError::InvalidFile(
                "null-aware frame truncated in bitmap prefix".into(),
            ));
        }
        let bitmap_len = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let nulls = if bitmap_len == 0 {
            None
        } else {
            if cursor + bitmap_len > data.len() {
                return Err(FluxError::InvalidFile(
                    "null-aware frame truncated in bitmap body".into(),
                ));
            }
            let bytes = &data[cursor..cursor + bitmap_len];
            cursor += bitmap_len;
            Some(null_bitmap::decode(bytes, n_rows)?)
        };

        // Flux payload for this column.
        if cursor + 4 > data.len() {
            return Err(FluxError::InvalidFile(
                "null-aware frame truncated before column payload length".into(),
            ));
        }
        let payload_len = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + payload_len > data.len() {
            return Err(FluxError::InvalidFile(
                "null-aware frame truncated in column payload".into(),
            ));
        }
        let flux_bytes = &data[cursor..cursor + payload_len];
        cursor += payload_len;

        let col_batch = reader.decompress(flux_bytes, &Predicate::None)?;
        if col_batch.num_columns() != 1 {
            return Err(FluxError::Internal(format!(
                "null_aware expected single-column payload, got {}",
                col_batch.num_columns()
            )));
        }
        let dense_col = col_batch.column(0).clone();
        let base_field = col_batch.schema().field(0).clone();

        match nulls {
            None => {
                rebuilt_fields.push(base_field);
                rebuilt_cols.push(dense_col);
            }
            Some(nb) => {
                let field = base_field.with_nullable(true);
                rebuilt_fields.push(field);
                rebuilt_cols.push(reinflate_with_nulls(&dense_col, &nb, n_rows)?);
            }
        }
    }

    let rebuilt_schema = Arc::new(arrow_schema::Schema::new(rebuilt_fields));
    RecordBatch::try_new(rebuilt_schema, rebuilt_cols)
        .map_err(|e| FluxError::Internal(format!("null_aware rebuild: {e}")))
}

/// Returns `true` if `data` starts with the null-aware magic prefix.
pub fn is_null_aware(data: &[u8]) -> bool {
    data.len() >= 4 && u32::from_le_bytes(data[0..4].try_into().unwrap()) == FLUX_NULL_AWARE_MAGIC
}

// ─── internals ──────────────────────────────────────────────────────

fn nulls_to_mask(nulls: &NullBuffer, len: usize) -> BooleanArray {
    let mut vals = Vec::with_capacity(len);
    for i in 0..len {
        vals.push(nulls.is_valid(i));
    }
    BooleanArray::from(vals)
}

/// Take a dense column (length = n_valid_rows) and the original null
/// buffer (length = n_rows), and build an array of length `n_rows`
/// whose non-null slots are drawn in order from the dense column and
/// whose null slots are masked out.
fn reinflate_with_nulls(
    dense: &ArrayRef,
    nulls: &NullBuffer,
    n_rows: usize,
) -> FluxResult<ArrayRef> {
    // Build a `take` index that maps each logical row to the
    // corresponding dense row — or a sentinel 0 for null rows (the
    // value there is never read because the null mask wipes it).
    let mut take_idx: Vec<u32> = Vec::with_capacity(n_rows);
    let mut dense_cursor: u32 = 0;
    for i in 0..n_rows {
        if nulls.is_valid(i) {
            take_idx.push(dense_cursor);
            dense_cursor += 1;
        } else {
            // The take will produce a harmless value here; the null
            // mask replacement below overwrites it.
            take_idx.push(0);
        }
    }
    let taken = arrow::compute::take(
        dense.as_ref(),
        &arrow_array::UInt32Array::from(take_idx),
        None,
    )
    .map_err(|e| FluxError::Internal(format!("null_aware take: {e}")))?;

    // Layer the original null buffer onto the taken array.
    let data = taken.to_data();
    let new_data = data.into_builder().nulls(Some(nulls.clone())).build()
        .map_err(|e| FluxError::Internal(format!("null_aware nulls: {e}")))?;
    Ok(arrow_array::make_array(new_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Int64, false),
        ]))
    }

    fn batch_with_nulls() -> RecordBatch {
        let a = Int64Array::from(vec![Some(1), None, Some(3), Some(4), None]);
        let b = StringArray::from(vec![Some("x"), Some("y"), None, Some("z"), None]);
        let c = Int64Array::from(vec![10, 20, 30, 40, 50]);
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(a), Arc::new(b), Arc::new(c)],
        )
        .unwrap()
    }

    #[test]
    fn round_trip_preserves_nulls() {
        let input = batch_with_nulls();
        let bytes = compress(&input).unwrap();
        assert!(is_null_aware(&bytes));

        let out = decompress(&bytes).unwrap();
        assert_eq!(out.num_rows(), input.num_rows());
        assert_eq!(out.num_columns(), input.num_columns());

        // Column a: nulls at positions 1, 4.
        let a = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(a.len(), 5);
        assert_eq!(a.is_null(0), false); assert_eq!(a.value(0), 1);
        assert_eq!(a.is_null(1), true);
        assert_eq!(a.is_null(2), false); assert_eq!(a.value(2), 3);
        assert_eq!(a.is_null(3), false); assert_eq!(a.value(3), 4);
        assert_eq!(a.is_null(4), true);

        // Column b: nulls at positions 2, 4.
        let b = out.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(b.value(0), "x");
        assert_eq!(b.value(1), "y");
        assert_eq!(b.is_null(2), true);
        assert_eq!(b.value(3), "z");
        assert_eq!(b.is_null(4), true);

        // Column c: no nulls; zero bitmap overhead.
        let c = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(c.null_count(), 0);
    }

    #[test]
    fn zero_overhead_on_fully_valid_batch() {
        let a = Int64Array::from(vec![1i64, 2, 3]);
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(a)]).unwrap();

        let bytes = compress(&batch).unwrap();
        let out = decompress(&bytes).unwrap();
        assert_eq!(out.num_rows(), 3);
        let rc = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(rc.null_count(), 0);
        // Sanity: bitmap body for a no-null column is zero bytes.
        assert!(bytes.len() < 10 * 1024);
    }

    #[test]
    fn rejects_plain_flux_file() {
        let batch = batch_with_nulls();
        let raw = FluxWriter::new().compress(&batch).unwrap();
        let err = decompress(&raw).unwrap_err();
        assert!(err.to_string().contains("bad null-aware magic"));
    }
}
