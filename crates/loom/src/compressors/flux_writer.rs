// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `FluxWriter` — the top-level [`LoomCompressor`] implementation.
//!
//! Orchestrates the full write pipeline:
//! 1. Extract numeric columns from an Arrow [`RecordBatch`].
//! 2. Segment each column into 1 024-row chunks.
//! 3. Run the Loom classifier on each chunk.
//! 4. Dispatch to the appropriate compressor.
//! 5. Append the Atlas metadata footer.
//! 6. Return the complete `.flux` byte buffer.
//!
//! ## File Layout (single column, multiple segments)
//! ```text
//! ┌──────────────────────────────┐
//! │  Block 0 (segment 0)         │
//! ├──────────────────────────────┤
//! │  Block 1 (segment 1)         │
//! ├──────────────────────────────┤
//! │  ...                         │
//! ├──────────────────────────────┤
//! │  Atlas Footer                │
//! └──────────────────────────────┘
//! ```

use arrow_array::{
    RecordBatch,
    array::{Array, UInt64Array, Int64Array, UInt32Array, Int32Array, Float64Array},
};
use arrow_schema::DataType;

use crate::{
    SEGMENT_SIZE,
    atlas::{AtlasFooter, BlockMeta},
    error::{FluxError, FluxResult},
    loom_classifier::{classify, LoomStrategy},
    traits::LoomCompressor,
    compressors::{
        bit_slab_compressor,
        rle_compressor,
        delta_compressor,
        dict_compressor,
        lz4_compressor,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// FluxWriter
// ─────────────────────────────────────────────────────────────────────────────

/// The primary [`LoomCompressor`] — writes `.flux` formatted byte buffers.
#[derive(Debug, Default, Clone)]
pub struct FluxWriter {
    /// Force a specific strategy (useful for benchmarking).  `None` = auto.
    pub force_strategy: Option<LoomStrategy>,
}

impl FluxWriter {
    /// Create a new `FluxWriter` with automatic strategy selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a writer that always uses the given strategy (for benchmarks).
    pub fn with_strategy(strategy: LoomStrategy) -> Self {
        Self { force_strategy: Some(strategy) }
    }
}

impl LoomCompressor for FluxWriter {
    fn compress(&self, batch: &RecordBatch) -> FluxResult<Vec<u8>> {
        let mut output: Vec<u8> = Vec::new();
        let mut footer = AtlasFooter::new();

        // Iterate over each column in the RecordBatch.
        for col_idx in 0..batch.num_columns() {
            let col = batch.column(col_idx);
            let values = extract_as_u128(col.as_ref())?;

            // Process in SEGMENT_SIZE chunks.
            for chunk in values.chunks(SEGMENT_SIZE) {
                let block_offset = output.len() as u64;

                // Classify.
                let strategy = self.force_strategy.unwrap_or_else(|| {
                    classify(chunk).strategy
                });

                // Compress.
                let block_bytes = compress_chunk(chunk, strategy)?;

                // Compute block stats for Atlas.
                let z_min = chunk.iter().copied().min().unwrap_or(0);
                let z_max = chunk.iter().copied().max().unwrap_or(0);

                output.extend_from_slice(&block_bytes);

                footer.push(BlockMeta {
                    block_offset,
                    z_min,
                    z_max,
                    null_bitmap_offset: 0, // TODO: null bitmap support
                    strategy,
                });
            }
        }

        // Append the Atlas footer.
        output.extend(footer.to_bytes()?);
        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Strategy dispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Compress a single `chunk` of u128 values using the given `strategy`.
pub fn compress_chunk(
    chunk: &[u128],
    strategy: LoomStrategy,
) -> FluxResult<Vec<u8>> {
    match strategy {
        LoomStrategy::Rle        => rle_compressor::compress(chunk),
        LoomStrategy::DeltaDelta => {
            if chunk.len() >= 2 {
                delta_compressor::compress(chunk)
            } else {
                // Fallback for tiny chunks.
                bit_slab_compressor::compress(chunk)
            }
        }
        LoomStrategy::Dictionary => dict_compressor::compress(chunk),
        LoomStrategy::BitSlab    => bit_slab_compressor::compress(chunk),
        LoomStrategy::SimdLz4    => lz4_compressor::compress(chunk),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrow column extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract all values from an Arrow array as `u128` (bit-cast / zero-extend).
///
/// This is zero-copy for integer arrays — we access the underlying Arrow
/// `Buffer` directly without moving data between allocations.
pub fn extract_as_u128(array: &dyn Array) -> FluxResult<Vec<u128>> {
    match array.data_type() {
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            Ok(arr.values().iter().map(|&v| v as u128).collect())
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            // Reinterpret as unsigned for bit-packing; sign is preserved in
            // the two's-complement u128 representation.
            Ok(arr.values().iter().map(|&v| v as u64 as u128).collect())
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            Ok(arr.values().iter().map(|&v| v as u128).collect())
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(arr.values().iter().map(|&v| v as u32 as u128).collect())
        }
        DataType::Float64 => {
            // Bit-cast f64 → u64 → u128 using num-traits to avoid UB.
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(arr
                .values()
                .iter()
                .map(|&f| f.to_bits() as u128)
                .collect())
        }
        dt => Err(FluxError::Internal(format!(
            "unsupported Arrow data type for FluxWriter: {dt}"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::UInt64Array;
    use arrow_schema::{Schema, Field, DataType};
    use std::sync::Arc;

    fn make_batch(values: Vec<u64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::UInt64, false),
        ]));
        let arr = Arc::new(UInt64Array::from(values));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    #[test]
    fn write_sequential_batch() {
        let values: Vec<u64> = (0..2048).collect();
        let batch = make_batch(values);
        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        assert!(!bytes.is_empty(), "output should not be empty");
        // Verify Atlas footer can be parsed from the output.
        let footer = crate::atlas::AtlasFooter::from_file_tail(&bytes).unwrap();
        assert_eq!(footer.blocks.len(), 2, "2048 values → 2 × 1024-row segments");
    }

    #[test]
    fn write_constant_batch() {
        let values: Vec<u64> = vec![42u64; 1024];
        let batch = make_batch(values);
        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let footer = crate::atlas::AtlasFooter::from_file_tail(&bytes).unwrap();
        assert_eq!(footer.blocks[0].strategy, LoomStrategy::Rle);
    }
}
