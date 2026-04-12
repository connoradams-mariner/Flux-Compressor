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
    ArrayRef, RecordBatch,
    array::{
        Array, UInt64Array, Int64Array, UInt32Array, Int32Array, Float64Array,
        UInt8Array, UInt16Array, Int8Array, Int16Array, Float32Array,
        BooleanArray, Date32Array, Date64Array,
        TimestampSecondArray, TimestampMillisecondArray,
        TimestampMicrosecondArray, TimestampNanosecondArray,
        StructArray, ListArray,
    },
};
use arrow_schema::DataType;
use arrow::compute as arrow_compute;
use std::sync::Arc;

use crate::{
    SEGMENT_SIZE,
    atlas::{AtlasFooter, BlockMeta, ColumnDescriptor},
    dtype::FluxDType,
    dtype_router::{self, RouteDecision, NativeWidth},
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
    /// Original Arrow data type tag for lossless round-trip.
    pub dtype_tag: FluxDType,
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

        let profile = self.profile;
        let force = self.force_strategy;
        let batch_schema = batch.schema();
        let mut all_blocks: Vec<Vec<(Vec<u8>, BlockMeta)>> = Vec::new();
        let mut schema_descriptors: Vec<ColumnDescriptor> = Vec::new();

        for col_idx in 0..batch.num_columns() {
            let array = batch.column(col_idx).as_ref();
            let route = dtype_router::route(array.data_type());

            let field_name = batch_schema.field(col_idx).name().clone();

            match route {
                RouteDecision::FastPath { strategy, native_width: _ } => {
                    // Extract as u64 (same as Classify path) but skip classifier.
                    let col = extract_column_data(array, col_idx as u16)?;
                    let use_strategy = force.unwrap_or(strategy);

                    // For Timestamp fast path: verify monotonicity on a probe
                    // window. If not monotone, fall back to full classification.
                    let final_strategy = if strategy == LoomStrategy::DeltaDelta
                        && force.is_none()
                        && !is_monotone_probe(&col.values_u64)
                    {
                        None
                    } else {
                        Some(use_strategy)
                    };

                    let dtype_tag = col.dtype_tag;
                    let blocks = compress_numeric_column(
                        &col, final_strategy, self.u64_only, profile,
                    )?;
                    all_blocks.push(blocks);
                    schema_descriptors.push(ColumnDescriptor {
                        name: field_name,
                        dtype_tag: dtype_tag.as_u8(),
                        children: Vec::new(),
                        column_id: col_idx as u16,
                    });
                }
                RouteDecision::Classify => {
                    let col = extract_column_data(array, col_idx as u16)?;
                    let dtype_tag = col.dtype_tag;
                    let blocks = compress_numeric_column(
                        &col, force, self.u64_only, profile,
                    )?;
                    all_blocks.push(blocks);
                    schema_descriptors.push(ColumnDescriptor {
                        name: field_name,
                        dtype_tag: dtype_tag.as_u8(),
                        children: Vec::new(),
                        column_id: col_idx as u16,
                    });
                }
                RouteDecision::StringPipeline => {
                    let col_id = col_idx as u16;
                    let dtype_tag = FluxDType::from_arrow(array.data_type())
                        .unwrap_or(FluxDType::Utf8);
                    let block_bytes = crate::compressors::string_compressor::compress_array_with_profile(array, profile)?;
                    let crc = crc32fast::hash(&block_bytes);
                    let meta = BlockMeta {
                        block_offset: 0,
                        z_min: 0,
                        z_max: u128::MAX,
                        null_bitmap_offset: 0,
                        strategy: LoomStrategy::SimdLz4,
                        value_count: array.len() as u32,
                        column_id: col_id,
                        crc32: crc,
                        u64_only: false,
                        dtype_tag,
                    };
                    all_blocks.push(vec![(block_bytes, meta)]);
                    schema_descriptors.push(ColumnDescriptor {
                        name: field_name,
                        dtype_tag: dtype_tag.as_u8(),
                        children: Vec::new(),
                        column_id: col_id,
                    });
                }
                RouteDecision::NestedPipeline => {
                    let mut next_col_id = all_blocks.iter()
                        .flat_map(|b| b.iter())
                        .map(|(_, m)| m.column_id + 1)
                        .max()
                        .unwrap_or(col_idx as u16);
                    let desc = flatten_and_compress(
                        &field_name,
                        array,
                        &mut next_col_id,
                        &mut all_blocks,
                        self.u64_only,
                        profile,
                        force,
                    )?;
                    schema_descriptors.push(desc);
                }
            }
        }

        // Concatenate blocks, patch offsets.
        let total_bytes: usize = all_blocks.iter()
            .flat_map(|col| col.iter())
            .map(|(bytes, _)| bytes.len())
            .sum();
        let mut output: Vec<u8> = Vec::with_capacity(total_bytes + 1024);
        let mut footer = AtlasFooter::new();

        for col_blocks in all_blocks {
            for (block_bytes, mut meta) in col_blocks {
                meta.block_offset = output.len() as u64;
                output.extend_from_slice(&block_bytes);
                footer.push(meta);
            }
        }

        footer.schema = schema_descriptors;
        output.extend(footer.to_bytes()?);
        Ok(output)
    }
}

/// Compress a numeric column (Classify or FastPath) into blocks.
///
/// If `force_strategy` is `Some(s)`, all segments use that strategy (skip
/// classifier). If `None`, the Loom classifier runs on each segment.
fn compress_numeric_column(
    col: &ColumnData,
    force_strategy: Option<LoomStrategy>,
    u64_only: bool,
    profile: crate::CompressionProfile,
) -> FluxResult<Vec<(Vec<u8>, BlockMeta)>> {
    use rayon::prelude::*;

    let segment_ranges = crate::segmenter::adaptive_segment_u64(
        &col.values_u64, force_strategy,
    );

    let col_u64 = &col.values_u64;
    let col_id = col.col_id;
    let dtype_tag = col.dtype_tag;

    segment_ranges
        .into_par_iter()
        .map(|(range, strategy)| {
            let seg_u64 = &col_u64[range.clone()];
            let seg_len = seg_u64.len();

            let block_bytes = if u64_only {
                compress_chunk_u64(seg_u64, strategy, profile)?
            } else {
                let chunk_u128: Vec<u128> = seg_u64.iter()
                    .map(|&v| v as u128).collect();
                compress_chunk_with_profile(&chunk_u128, strategy, profile)?
            };

            let crc = crc32fast::hash(&block_bytes);
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
                u64_only,
                dtype_tag,
            };
            Ok((block_bytes, meta))
        })
        .collect()
}

/// Quick monotonicity check on the first PROBE_SIZE values.
/// Returns `true` if the values are non-decreasing.
fn is_monotone_probe(values: &[u64]) -> bool {
    let probe = &values[..values.len().min(crate::PROBE_SIZE)];
    probe.windows(2).all(|w| w[0] <= w[1])
}

// ─────────────────────────────────────────────────────────────────────────────
// Nested type flattening (Tier 3) — parallel leaf compression
// ─────────────────────────────────────────────────────────────────────────────

/// A leaf array pending parallel compression.
struct PendingLeaf {
    col_id: u16,
    array: ArrayRef,
    dtype_tag: FluxDType,
}

/// Two-phase nested compression:
/// 1. `flatten_to_pending()` walks the tree, assigns column IDs, collects leaves
/// 2. All leaves are compressed in parallel via rayon
fn flatten_and_compress(
    name: &str,
    array: &dyn Array,
    next_col_id: &mut u16,
    all_blocks: &mut Vec<Vec<(Vec<u8>, BlockMeta)>>,
    u64_only: bool,
    profile: crate::CompressionProfile,
    force: Option<LoomStrategy>,
) -> FluxResult<ColumnDescriptor> {
    use rayon::prelude::*;

    // Phase 1: collect leaves without compressing.
    let mut pending: Vec<PendingLeaf> = Vec::new();
    let desc = flatten_to_pending(name, array, next_col_id, &mut pending);

    // Phase 2: compress all leaves in parallel.
    let compressed: Vec<(u16, Vec<(Vec<u8>, BlockMeta)>)> = pending
        .into_par_iter()
        .map(|leaf| {
            let blocks = compress_leaf_data(
                leaf.array.as_ref(), leaf.col_id, leaf.dtype_tag,
                u64_only, profile, force,
            )?;
            Ok((leaf.col_id, blocks))
        })
        .collect::<FluxResult<_>>()?;

    // Phase 3: assemble blocks in column_id order.
    let mut sorted = compressed;
    sorted.sort_by_key(|(col_id, _)| *col_id);
    for (_, blocks) in sorted {
        all_blocks.push(blocks);
    }

    Ok(desc)
}

/// Phase 1: recursively walk the tree, assign column IDs, collect leaf arrays.
/// No compression happens here — purely structural.
fn flatten_to_pending(
    name: &str,
    array: &dyn Array,
    next_col_id: &mut u16,
    pending: &mut Vec<PendingLeaf>,
) -> ColumnDescriptor {
    let dt = array.data_type();
    match dt {
        DataType::Struct(fields) => {
            let struct_arr = array.as_any().downcast_ref::<StructArray>().unwrap();
            let mut children = Vec::with_capacity(fields.len());
            for (i, field) in fields.iter().enumerate() {
                let child = struct_arr.column(i).as_ref();
                let desc = flatten_to_pending(field.name(), child, next_col_id, pending);
                children.push(desc);
            }
            ColumnDescriptor {
                name: name.to_string(),
                dtype_tag: FluxDType::StructContainer.as_u8(),
                children,
                column_id: u16::MAX,
            }
        }
        DataType::List(field) => {
            let list_arr = array.as_any().downcast_ref::<ListArray>().unwrap();
            let offsets_raw: Vec<i32> = list_arr.offsets().iter().copied().collect();
            let num_lists = offsets_raw.len() - 1;
            let lengths: Vec<i32> = offsets_raw.windows(2).map(|w| w[1] - w[0]).collect();

            let constant_len = if num_lists > 0 && lengths.iter().all(|&l| l == lengths[0]) {
                Some(lengths[0])
            } else {
                None
            };

            let lengths_desc = if let Some(clen) = constant_len {
                let clen_arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![clen]));
                collect_leaf("__const_len", clen_arr, next_col_id, pending)
            } else {
                let lengths_arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(lengths.clone()));
                collect_leaf("__lengths", lengths_arr, next_col_id, pending)
            };

            let flat_values = list_arr.values();
            let children = if flat_values.data_type().is_numeric() && num_lists > 0 {
                let values_u64: Vec<u64> = (0..flat_values.len())
                    .map(|i| extract_single_u64(flat_values.as_ref(), i))
                    .collect();

                let mut bases: Vec<u64> = Vec::with_capacity(num_lists);
                let mut deltas: Vec<u64> = Vec::with_capacity(values_u64.len());
                for i in 0..num_lists {
                    let start = offsets_raw[i] as usize;
                    let end = offsets_raw[i + 1] as usize;
                    if start < end {
                        let base = values_u64[start];
                        bases.push(base);
                        for j in start..end {
                            deltas.push(values_u64[j].wrapping_sub(base));
                        }
                    } else {
                        bases.push(0);
                    }
                }

                let bases_arr: ArrayRef = Arc::new(arrow_array::UInt64Array::from(bases));
                let bases_desc = collect_leaf("__bases", bases_arr, next_col_id, pending);
                let deltas_arr: ArrayRef = Arc::new(arrow_array::UInt64Array::from(deltas));
                let deltas_desc = collect_leaf("__deltas", deltas_arr, next_col_id, pending);
                vec![lengths_desc, bases_desc, deltas_desc]
            } else {
                let values_desc = flatten_to_pending(
                    field.name(), flat_values.as_ref(), next_col_id, pending,
                );
                vec![lengths_desc, values_desc]
            };

            ColumnDescriptor {
                name: name.to_string(),
                dtype_tag: FluxDType::ListContainer.as_u8(),
                children,
                column_id: u16::MAX,
            }
        }
        DataType::Map(_, _) => {
            let map_arr = array.as_any().downcast_ref::<arrow_array::MapArray>().unwrap();
            let offsets_raw: Vec<i32> = map_arr.offsets().iter().copied().collect();

            let lengths: Vec<i32> = offsets_raw.windows(2).map(|w| w[1] - w[0]).collect();
            let lengths_arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(lengths));
            let lengths_desc = collect_leaf("__lengths", lengths_arr, next_col_id, pending);

            let entries = map_arr.entries();
            let keys_arr = entries.column(0);
            let vals_arr = entries.column(1);
            let (sorted_keys, sorted_vals) = sort_map_entries_by_key(
                keys_arr.as_ref(), vals_arr.as_ref(), &offsets_raw,
            );

            let keys_desc = flatten_to_pending(
                "key", sorted_keys.as_ref(), next_col_id, pending,
            );
            let values_desc = flatten_to_pending(
                "value", sorted_vals.as_ref(), next_col_id, pending,
            );

            ColumnDescriptor {
                name: name.to_string(),
                dtype_tag: FluxDType::MapContainer.as_u8(),
                children: vec![lengths_desc, keys_desc, values_desc],
                column_id: u16::MAX,
            }
        }
        // Leaf type.
        _ => {
            let arr_ref = array.slice(0, array.len());
            collect_leaf(name, arr_ref, next_col_id, pending)
        }
    }
}

/// Assign a column ID and add a leaf to the pending compression queue.
fn collect_leaf(
    name: &str,
    array: ArrayRef,
    next_col_id: &mut u16,
    pending: &mut Vec<PendingLeaf>,
) -> ColumnDescriptor {
    let col_id = *next_col_id;
    *next_col_id += 1;
    let dtype_tag = FluxDType::from_arrow(array.data_type()).unwrap_or(FluxDType::UInt64);
    pending.push(PendingLeaf { col_id, array, dtype_tag });
    ColumnDescriptor {
        name: name.to_string(),
        dtype_tag: dtype_tag.as_u8(),
        children: Vec::new(),
        column_id: col_id,
    }
}

/// Compress a single leaf array (called in parallel for each leaf).
fn compress_leaf_data(
    array: &dyn Array,
    col_id: u16,
    dtype_tag: FluxDType,
    u64_only: bool,
    profile: crate::CompressionProfile,
    force: Option<LoomStrategy>,
) -> FluxResult<Vec<(Vec<u8>, BlockMeta)>> {
    let route = dtype_router::route(array.data_type());

    match route {
        RouteDecision::StringPipeline => {
            let block_bytes = crate::compressors::string_compressor::compress_array_with_profile(array, profile)?;
            let crc = crc32fast::hash(&block_bytes);
            let meta = BlockMeta {
                block_offset: 0,
                z_min: 0,
                z_max: u128::MAX,
                null_bitmap_offset: 0,
                strategy: LoomStrategy::SimdLz4,
                value_count: array.len() as u32,
                column_id: col_id,
                crc32: crc,
                u64_only: false,
                dtype_tag,
            };
            Ok(vec![(block_bytes, meta)])
        }
        _ => {
            let col = extract_column_data(array, col_id)?;
            let strat = match route {
                RouteDecision::FastPath { strategy, .. } if force.is_none() => Some(strategy),
                _ => force,
            };
            compress_numeric_column(&col, strat, u64_only, profile)
        }
    }
}

/// Extract a single value from an Arrow numeric array as u64 (bit-cast).
/// Used by delta-from-base encoding for List values.
fn extract_single_u64(array: &dyn Array, idx: usize) -> u64 {
    use arrow_schema::DataType;
    match array.data_type() {
        DataType::Int64 => array.as_any().downcast_ref::<Int64Array>().unwrap().value(idx) as u64,
        DataType::UInt64 => array.as_any().downcast_ref::<UInt64Array>().unwrap().value(idx),
        DataType::Int32 => array.as_any().downcast_ref::<Int32Array>().unwrap().value(idx) as u32 as u64,
        DataType::UInt32 => array.as_any().downcast_ref::<UInt32Array>().unwrap().value(idx) as u64,
        DataType::Float64 => array.as_any().downcast_ref::<Float64Array>().unwrap().value(idx).to_bits(),
        DataType::Float32 => array.as_any().downcast_ref::<Float32Array>().unwrap().value(idx).to_bits() as u64,
        DataType::Int16 => array.as_any().downcast_ref::<Int16Array>().unwrap().value(idx) as u16 as u64,
        DataType::UInt16 => array.as_any().downcast_ref::<UInt16Array>().unwrap().value(idx) as u64,
        DataType::Int8 => array.as_any().downcast_ref::<Int8Array>().unwrap().value(idx) as u8 as u64,
        DataType::UInt8 => array.as_any().downcast_ref::<UInt8Array>().unwrap().value(idx) as u64,
        _ => 0,
    }
}

/// Sort map entries within each row by key for maximum key-column repetition.
/// Returns (sorted_keys, sorted_values) as new ArrayRefs.
fn sort_map_entries_by_key(
    keys: &dyn Array,
    values: &dyn Array,
    offsets: &[i32],
) -> (ArrayRef, ArrayRef) {
    // Only sort string keys (the common case).
    if let Some(key_arr) = keys.as_any().downcast_ref::<arrow_array::StringArray>() {
        let n = keys.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let num_rows = offsets.len() - 1;

        // Sort indices within each row by key value.
        for row in 0..num_rows {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            if end - start > 1 {
                indices[start..end].sort_by(|&a, &b| key_arr.value(a).cmp(&key_arr.value(b)));
            }
        }

        // Reorder keys.
        let sorted_keys: Vec<&str> = indices.iter().map(|&i| key_arr.value(i)).collect();
        let new_keys: ArrayRef = Arc::new(arrow_array::StringArray::from(sorted_keys));

        // Reorder values using arrow's take kernel.
        let idx_arr = arrow_array::UInt32Array::from(
            indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        );
        let new_values = arrow_compute::take(values, &idx_arr, None)
            .unwrap_or_else(|_| values.slice(0, values.len()));

        (new_keys, new_values)
    } else {
        // Non-string keys: don't sort, return clones.
        (
            keys.slice(0, keys.len()),
            values.slice(0, values.len()),
        )
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

    let dt = array.data_type();
    let dtype_tag = FluxDType::from_arrow(dt).ok_or_else(|| {
        FluxError::Internal(format!("unsupported Arrow data type for FluxWriter: {dt}"))
    })?;

    // Extract dense (non-null) values as u64.
    let values_u64: Vec<u64> = match dt {
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
        // ── Tier 1: new fixed-width types ────────────────────────────────
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u8 as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u8 as u64)).collect()
            }
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u16 as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u16 as u64)).collect()
            }
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&f| f.to_bits() as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x.to_bits() as u64)).collect()
            }
        }
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            if null_count == 0 {
                (0..arr.len()).map(|i| if arr.value(i) { 1u64 } else { 0u64 }).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| if x { 1u64 } else { 0u64 })).collect()
            }
        }
        DataType::Date32 => {
            let arr = array.as_any().downcast_ref::<Date32Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u32 as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u32 as u64)).collect()
            }
        }
        DataType::Date64 => {
            let arr = array.as_any().downcast_ref::<Date64Array>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => {
            let arr = array.as_any().downcast_ref::<TimestampSecondArray>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
            let arr = array.as_any().downcast_ref::<TimestampMillisecondArray>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            let arr = array.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            let arr = array.as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            if null_count == 0 {
                arr.values().iter().map(|&v| v as u64).collect()
            } else {
                arr.iter().filter_map(|v| v.map(|x| x as u64)).collect()
            }
        }
        _ => return Err(FluxError::Internal(format!(
            "unsupported Arrow data type for FluxWriter: {dt}"
        ))),
    };

    Ok(ColumnData {
        col_id,
        dtype_tag,
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
