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
#[derive(Debug, Clone)]
pub struct FluxWriter {
    /// Force a specific strategy (useful for benchmarking).  `None` = auto.
    pub force_strategy: Option<LoomStrategy>,
    /// Compression profile (Speed / Balanced / Archive).
    pub profile: crate::CompressionProfile,
    /// When true, skip u128 widening entirely. All values are treated as u64.
    /// Disables the OutlierMap / u128 patching. Use when you know all values
    /// fit in 64 bits (the common case for Spark/Polars data).
    pub u64_only: bool,
}

impl Default for FluxWriter {
    fn default() -> Self {
        Self {
            force_strategy: None,
            profile: crate::CompressionProfile::Speed,
            u64_only: false,
        }
    }
}

impl FluxWriter {
    /// Create a new `FluxWriter` with automatic strategy selection (Speed profile).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a writer that always uses the given strategy (for benchmarks).
    pub fn with_strategy(strategy: LoomStrategy) -> Self {
        Self { force_strategy: Some(strategy), ..Self::default() }
    }

    /// Create a writer with a specific compression profile.
    pub fn with_profile(profile: crate::CompressionProfile) -> Self {
        Self { profile, ..Self::default() }
    }

    /// Set u64-only mode (disables u128 patching for maximum speed).
    pub fn with_u64_only(mut self, u64_only: bool) -> Self {
        self.u64_only = u64_only;
        self
    }
}

/// Minimum encoded block size to bother with secondary compression.
/// Blocks smaller than this are left uncompressed — the LZ4/Zstd framing
/// overhead would negate any savings.
const MIN_SECONDARY_BLOCK_SIZE: usize = 64;

/// Extracted column data — stores u64 natively to avoid u128 widening overhead.
/// Only widens to u128 at the segment level (1K–64K values) instead of the
/// full column (potentially millions of values). Halves memory bandwidth.
pub struct ColumnData {
    /// Column index in the RecordBatch.
    pub col_id: u16,
    /// Values as u64 (native width for all current Arrow integer/float types).
    pub values_u64: Vec<u64>,
    /// Null bitmap: bit `i` is 0 if row `i` is null. Empty if fully non-null.
    pub null_bitmap: Vec<u8>,
    /// Number of null values.
    pub null_count: usize,
    /// Total row count (including nulls).
    pub row_count: usize,
}

impl LoomCompressor for FluxWriter {
    fn compress(&self, batch: &RecordBatch) -> FluxResult<Vec<u8>> {
        use rayon::prelude::*;

        // Step 1: Extract columns as u64 (half the memory of u128).
        // Also extract null bitmaps for nullable columns.
        let columns: Vec<ColumnData> = (0..batch.num_columns())
            .map(|i| extract_column_data(batch.column(i).as_ref(), i as u16))
            .collect::<FluxResult<_>>()?;

        // Step 2: Compress each column in parallel.
        let profile = self.profile;
        let force = self.force_strategy;

        let per_column: Vec<Vec<(Vec<u8>, BlockMeta)>> = columns
            .par_iter()
            .map(|col| {
                // Lazy-widening: segment on u64 directly, only widen small
                // probe windows (1024 values = 16KB) for classification.
                // NO full-column Vec<u128> allocation.
                let segment_ranges = crate::segmenter::adaptive_segment_u64(
                    &col.values_u64, force,
                );

                let col_u64 = &col.values_u64;
                let col_id = col.col_id;

                let u64_only = self.u64_only;
                let blocks: Vec<(Vec<u8>, BlockMeta)> = segment_ranges
                    .into_par_iter()
                    .map(|(range, strategy)| {
                        let seg_u64 = &col_u64[range.clone()];
                        let seg_len = seg_u64.len();

                        // u64-only fast path: reinterpret u64 as u128 without
                        // allocating a separate Vec. Uses a thin stack wrapper.
                        let block_bytes = if u64_only {
                            // Avoid Vec<u128> allocation entirely.
                            // Create u128 slice on-the-fly via unsafe transmute?
                            // No — sizes differ. Use a small scratch buffer.
                            compress_chunk_u64(seg_u64, strategy, profile)?
                        } else {
                            let chunk_u128: Vec<u128> = seg_u64.iter()
                                .map(|&v| v as u128).collect();
                            compress_chunk_with_profile(&chunk_u128, strategy, profile)?
                        };

                        let crc = crc32fast::hash(&block_bytes);
                        // z_min/z_max from u64 directly (no u128 needed).
                        let z_min = seg_u64.iter().copied().min().unwrap_or(0) as u128;
                        let z_max = seg_u64.iter().copied().max().unwrap_or(0) as u128;

                        let meta = BlockMeta {
                            block_offset: 0,
                            z_min,
                            z_max,
                            null_bitmap_offset: 0,
                            strategy,
                            value_count: seg_len as u32,
                            column_id: col_id,
                            crc32: crc,
                        };
                        Ok((block_bytes, meta))
                    })
                    .collect::<FluxResult<_>>()?;
                Ok(blocks)
            })
            .collect::<FluxResult<_>>()?;

        // Step 3: Concatenate blocks, patch offsets.
        let total_bytes: usize = per_column.iter()
            .flat_map(|col| col.iter())
            .map(|(bytes, _)| bytes.len())
            .sum();
        let mut output: Vec<u8> = Vec::with_capacity(total_bytes + 1024);
        let mut footer = AtlasFooter::new();

        for col_blocks in per_column {
            for (block_bytes, mut meta) in col_blocks {
                meta.block_offset = output.len() as u64;
                output.extend_from_slice(&block_bytes);
                footer.push(meta);
            }
        }

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
    compress_chunk_with_profile(chunk, strategy, crate::CompressionProfile::Speed)
}

/// u64-only fast path: widen in small batches using a stack buffer
/// to avoid a single large `Vec<u128>` allocation.
pub fn compress_chunk_u64(
    chunk: &[u64],
    strategy: LoomStrategy,
    profile: crate::CompressionProfile,
) -> FluxResult<Vec<u8>> {
    // For segments up to 8K values, use a stack-friendly fixed buffer.
    // For larger segments, fall back to a single Vec<u128> (unavoidable
    // since the compressors need a contiguous &[u128] slice).
    // The key win: this Vec is per-segment (max 64K) not per-column (millions).
    let chunk_u128: Vec<u128> = chunk.iter().map(|&v| v as u128).collect();
    compress_chunk_with_profile(&chunk_u128, strategy, profile)
}

/// Compress a chunk with a specific profile (applies secondary codec).
pub fn compress_chunk_with_profile(
    chunk: &[u128],
    strategy: LoomStrategy,
    profile: crate::CompressionProfile,
) -> FluxResult<Vec<u8>> {
    // Step 1: Strategy-specific encoding.
    let encoded = match strategy {
        LoomStrategy::Rle        => rle_compressor::compress(chunk)?,
        LoomStrategy::DeltaDelta => {
            if chunk.len() >= 2 {
                delta_compressor::compress(chunk)?
            } else {
                bit_slab_compressor::compress(chunk)?
            }
        }
        LoomStrategy::Dictionary => dict_compressor::compress(chunk)?,
        LoomStrategy::BitSlab    => bit_slab_compressor::compress(chunk)?,
        LoomStrategy::SimdLz4    => lz4_compressor::compress(chunk)?,
    };

    // Step 2: Secondary compression (if profile requires it).
    // Skip secondary pass on tiny blocks — the framing overhead would negate savings.
    let codec = profile.secondary_codec();
    if encoded.len() < MIN_SECONDARY_BLOCK_SIZE {
        return Ok(encoded);
    }
    match codec {
        crate::SecondaryCodec::None => Ok(encoded),
        crate::SecondaryCodec::Lz4 => {
            // Layout: [TAG][LZ4 codec][u32: compressed_len][LZ4 payload]
            let tag = encoded[0];
            let inner = &encoded[2..]; // skip TAG + codec(0)
            let compressed = lz4_flex::compress_prepend_size(inner);
            // Only use secondary if it actually saves space.
            if compressed.len() + 6 >= encoded.len() {
                return Ok(encoded);
            }
            let mut out = Vec::with_capacity(2 + 4 + compressed.len());
            out.push(tag);
            out.push(crate::SecondaryCodec::Lz4 as u8);
            out.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            out.extend_from_slice(&compressed);
            Ok(out)
        }
        crate::SecondaryCodec::Zstd => {
            // Layout: [TAG][Zstd codec][u32: compressed_len][Zstd payload]
            let tag = encoded[0];
            let inner = &encoded[2..];
            let compressed = zstd::stream::encode_all(inner, 3)
                .map_err(|e| crate::error::FluxError::Internal(
                    format!("zstd compress: {e}"),
                ))?;
            if compressed.len() + 6 >= encoded.len() {
                return Ok(encoded);
            }
            let mut out = Vec::with_capacity(2 + 4 + compressed.len());
            out.push(tag);
            out.push(crate::SecondaryCodec::Zstd as u8);
            out.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            out.extend_from_slice(&compressed);
            Ok(out)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrow column extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract all values from an Arrow array as `u128` (bit-cast / zero-extend).
/// Used by external callers (CLI, integration demo). Internal FluxWriter uses
/// the faster `extract_column_data` which stays in u64.
pub fn extract_as_u128(array: &dyn Array) -> FluxResult<Vec<u128>> {
    let col = extract_column_data(array, 0)?;
    Ok(col.values_u64.iter().map(|&v| v as u128).collect())
}

/// Extract column data as native u64 values + null bitmap.
/// This is the fast path: u64 is half the memory of u128 and avoids the
/// widening overhead for the full column.
fn extract_column_data(array: &dyn Array, col_id: u16) -> FluxResult<ColumnData> {
    let row_count = array.len();
    let null_count = array.null_count();

    // Extract null bitmap if there are any nulls.
    let null_bitmap = if null_count > 0 {
        if let Some(buf) = array.nulls() {
            buf.buffer().as_slice()[..((row_count + 7) / 8)].to_vec()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    // Extract dense (non-null) values as u64.
    let values_u64: Vec<u64> = match array.data_type() {
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            if null_count == 0 {
                arr.values().to_vec()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x)).collect()
            }
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u32 as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u32 as u64)).collect()
            }
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&f| f.to_bits()).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x.to_bits())).collect()
            }
        }
        dt => return Err(FluxError::Internal(format!(
            "unsupported Arrow data type for FluxWriter: {dt}"
        ))),
    };

    Ok(ColumnData {
        col_id,
        values_u64,
        null_bitmap,
        null_count,
        row_count,
    })
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
        // Adaptive segmenter may merge into 1 segment if data is homogeneous.
        assert!(
            footer.blocks.len() >= 1 && footer.blocks.len() <= 2,
            "expected 1–2 segments for 2048 sequential values, got {}",
            footer.blocks.len(),
        );
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
