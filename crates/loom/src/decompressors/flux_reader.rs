// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `FluxReader` — the top-level [`LoomDecompressor`] implementation.
//!
//! ## Read Pipeline
//! 1. Locate and parse the **Atlas footer** from the tail of the byte slice.
//! 2. Apply **predicate pushdown**: skip blocks whose `[z_min, z_max]` range
//!    cannot contain any rows satisfying the predicate.
//! 3. For each surviving block: seek to `block_offset`, dispatch
//!    [`decompress_block`], collect `u128` values.
//! 4. Reconstruct an Arrow [`RecordBatch`] and return it.
//!
//! ## Zero-Copy
//! `FluxReader` never copies the input `&[u8]` — it borrows slices into the
//! caller's buffer for the entire read path.

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::{
    atlas::AtlasFooter,
    decompressors::block_reader::decompress_block,
    error::{FluxError, FluxResult},
    traits::{LoomDecompressor, Predicate},
};

// ─────────────────────────────────────────────────────────────────────────────
// FluxReader
// ─────────────────────────────────────────────────────────────────────────────

/// The primary [`LoomDecompressor`] — reads `.flux` formatted byte slices.
#[derive(Debug, Default, Clone)]
pub struct FluxReader {
    /// Column name to use when constructing the output [`RecordBatch`].
    /// Defaults to `"value"`.
    pub column_name: String,
}

impl FluxReader {
    /// Create a `FluxReader` with the given output column name.
    pub fn new(column_name: impl Into<String>) -> Self {
        Self { column_name: column_name.into() }
    }
}

impl Default for FluxReader {
    fn default() -> Self {
        Self { column_name: "value".into() }
    }
}

impl LoomDecompressor for FluxReader {
    /// Decompress `data` with optional predicate pushdown.
    ///
    /// This is the **hot read path**.  It:
    /// 1. Parses the Atlas footer (O(blocks), zero-copy).
    /// 2. Skips blocks that fail the `[z_min, z_max]` check.
    /// 3. Decompresses surviving blocks into `u128` values.
    /// 4. Returns an Arrow `RecordBatch` with a single `UInt64` column.
    fn decompress(&self, data: &[u8], predicate: &Predicate) -> FluxResult<RecordBatch> {
        // ── 1. Parse footer ──────────────────────────────────────────────────
        let footer = AtlasFooter::from_file_tail(data)?;

        if footer.blocks.is_empty() {
            return empty_batch(&self.column_name);
        }

        // ── 2. Predicate pushdown ────────────────────────────────────────────
        let candidates = footer.candidate_blocks(predicate);

        // ── 3. Decompress surviving blocks ───────────────────────────────────
        let mut all_values: Vec<u128> = Vec::new();

        for block_idx in candidates {
            let meta = &footer.blocks[block_idx];
            let start = meta.block_offset as usize;

            if start >= data.len() {
                return Err(FluxError::InvalidFile(format!(
                    "block {block_idx} offset {start} exceeds file length {}",
                    data.len()
                )));
            }

            let block_slice = &data[start..];
            let (values, _consumed) = decompress_block(block_slice)?;
            all_values.extend(values);
        }

        // ── 4. Build Arrow RecordBatch ───────────────────────────────────────
        values_to_batch(all_values, &self.column_name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn values_to_batch(values: Vec<u128>, col_name: &str) -> FluxResult<RecordBatch> {
    // Downcast u128 → u64 for Arrow interop.
    // Full u128 support requires a custom Arrow extension type; for now we
    // store the low 64 bits and the JNI bridge handles the high word separately.
    let u64_values: Vec<u64> = values.iter().map(|&v| v as u64).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new(col_name, DataType::UInt64, false),
    ]));
    let array = Arc::new(UInt64Array::from(u64_values));
    RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow)
}

fn empty_batch(col_name: &str) -> FluxResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(col_name, DataType::UInt64, false),
    ]));
    let array = Arc::new(UInt64Array::from(Vec::<u64>::new()));
    RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compressors::flux_writer::FluxWriter,
        traits::LoomCompressor,
    };
    use arrow_array::UInt64Array;
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn make_batch(values: Vec<u64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::UInt64, false),
        ]));
        let arr = Arc::new(UInt64Array::from(values));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    #[test]
    fn write_then_read_all() {
        let input: Vec<u64> = (0u64..2048).collect();
        let batch = make_batch(input.clone());

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();

        let reader = FluxReader::new("value");
        let out_batch = reader.decompress_all(&bytes).unwrap();

        let col = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let got: Vec<u64> = col.values().to_vec();
        assert_eq!(got, input);
    }

    #[test]
    fn predicate_pushdown_filters_blocks() {
        // 3 segments of 1024 each: [0..1024, 1024..2048, 2048..3072]
        let input: Vec<u64> = (0u64..3072).collect();
        let batch = make_batch(input.clone());

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();

        // Predicate: value > 2000 — should skip blocks 0 and 1.
        let reader = FluxReader::new("value");
        let pred = Predicate::GreaterThan {
            column: "value".into(),
            value: 2000,
        };
        let out_batch = reader.decompress(&bytes, &pred).unwrap();
        let col = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // All returned values must be from block 2 (≥ 2048).
        for &v in col.values() {
            assert!(v >= 2048, "unexpected value {v} from skipped block");
        }
    }

    #[test]
    fn constant_column_round_trip() {
        let input = vec![999u64; 1024];
        let batch = make_batch(input.clone());
        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();
        let col = out.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        assert!(col.values().iter().all(|&v| v == 999));
    }
}
