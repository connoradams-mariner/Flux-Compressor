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

use arrow_array::{
    Array as ArrowArray,
    ArrayRef, RecordBatch, UInt64Array, UInt32Array, UInt16Array, UInt8Array,
    Int64Array, Int32Array, Int16Array, Int8Array, Float64Array, Float32Array,
    BooleanArray, Date32Array, Date64Array,
    TimestampSecondArray, TimestampMillisecondArray,
    TimestampMicrosecondArray, TimestampNanosecondArray,
    StringArray, BinaryArray,
    StructArray, ListArray,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use std::sync::Arc;

use crate::{
    atlas::{AtlasFooter, ColumnDescriptor},
    dtype::FluxDType,
    decompressors::block_reader::{decompress_block, decompress_block_to_u64},
    error::{FluxError, FluxResult},
    traits::{LoomDecompressor, Predicate},
};

// ─────────────────────────────────────────────────────────────────────────────
// FluxReader
// ─────────────────────────────────────────────────────────────────────────────

/// The primary [`LoomDecompressor`] — reads `.flux` formatted byte slices.
#[derive(Debug, Clone)]
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

impl FluxReader {
    /// Decompress a `.flux` file using memory-mapped I/O.
    ///
    /// Instead of loading the entire file into RAM, the OS maps it into
    /// virtual memory and pages in only the blocks that are actually read.
    /// This is significantly faster for large files, especially with
    /// predicate pushdown (skipped blocks are never loaded).
    pub fn decompress_file(
        &self,
        path: &std::path::Path,
        predicate: &Predicate,
    ) -> FluxResult<RecordBatch> {
        let file = std::fs::File::open(path)
            .map_err(|e| FluxError::Io(e))?;
        // SAFETY: .flux files are immutable (versioned via transaction log).
        // No other process modifies them while we're reading.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| FluxError::Io(e))?;
        self.decompress(&mmap, predicate)
    }

    /// Convenience: decompress all blocks from a memory-mapped file.
    pub fn decompress_file_all(
        &self,
        path: &std::path::Path,
    ) -> FluxResult<RecordBatch> {
        self.decompress_file(path, &Predicate::None)
    }
}

impl FluxReader {
    /// Decompress a `.flux` file that has a schema tree (nested types).
    ///
    /// Optimized path: pre-decompresses ALL leaf blocks in parallel, then
    /// does a single-threaded tree reassembly pass.
    fn decompress_with_schema(
        &self,
        data: &[u8],
        footer: &AtlasFooter,
    ) -> FluxResult<RecordBatch> {
        use rayon::prelude::*;
        use std::collections::HashMap;

        // Phase 1: Collect all leaf column IDs and their block indices.
        let mut leaf_blocks: Vec<(u16, Vec<usize>)> = Vec::new();
        collect_leaf_blocks(&footer.schema, footer, &mut leaf_blocks);

        // Phase 2: Parallel decompression of all leaves.
        let decompressed: HashMap<u16, LeafData> = leaf_blocks
            .into_par_iter()
            .map(|(col_id, block_indices)| {
                let dtype_tag = footer.blocks[block_indices[0]].dtype_tag;
                let is_string = matches!(
                    dtype_tag,
                    FluxDType::Utf8 | FluxDType::LargeUtf8
                    | FluxDType::Binary | FluxDType::LargeBinary
                );

                if is_string {
                    // Collect string arrays from each block.
                    let mut arrays: Vec<ArrayRef> = Vec::new();
                    for &bi in &block_indices {
                        let start = footer.blocks[bi].block_offset as usize;
                        let arr = crate::compressors::string_compressor::decompress_to_arrow_string(&data[start..], dtype_tag)?;
                        arrays.push(arr);
                    }
                    Ok((col_id, LeafData::StringArrays(arrays)))
                } else {
                    // Numeric: decompress to u64.
                    let mut all_values: Vec<u64> = Vec::new();
                    for &bi in &block_indices {
                        let start = footer.blocks[bi].block_offset as usize;
                        let (values, _) = decompress_block_to_u64(&data[start..])?;
                        all_values.extend(values);
                    }
                    Ok((col_id, LeafData::Numeric(all_values)))
                }
            })
            .collect::<FluxResult<HashMap<u16, LeafData>>>()?;

        // Phase 3: Tree reassembly (single-threaded, just pointer chasing).
        let mut columns: Vec<ArrayRef> = Vec::new();
        let mut fields: Vec<Field> = Vec::new();

        for desc in &footer.schema {
            let (array, field) = reassemble_column_fast(footer, desc, &decompressed)?;
            columns.push(array);
            fields.push(field);
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, columns).map_err(FluxError::Arrow)
    }

    /// Decompress string/binary blocks using the zero-alloc Arrow path.
    fn decompress_string_blocks(
        &self,
        data: &[u8],
        footer: &AtlasFooter,
        candidates: &[usize],
        dtype_tag: FluxDType,
    ) -> FluxResult<RecordBatch> {
        use crate::compressors::string_compressor;

        let arrow_dt = dtype_tag.to_arrow();

        if matches!(dtype_tag, FluxDType::Utf8 | FluxDType::LargeUtf8) {
            // Fast path: build StringArray directly without per-string allocs.
            if candidates.len() == 1 {
                let start = footer.blocks[candidates[0]].block_offset as usize;
                let array = string_compressor::decompress_to_arrow_string(&data[start..], dtype_tag)?;
                let schema = Arc::new(Schema::new(vec![
                    Field::new(&self.column_name, arrow_dt, false),
                ]));
                return RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow);
            }
            // Multiple blocks: decompress each to StringArray, then concat.
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(candidates.len());
            for &block_idx in candidates {
                let start = footer.blocks[block_idx].block_offset as usize;
                arrays.push(string_compressor::decompress_to_arrow_string(&data[start..], dtype_tag)?);
            }
            let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
            let concat = arrow::compute::concat(&refs).map_err(FluxError::Arrow)?;
            let schema = Arc::new(Schema::new(vec![
                Field::new(&self.column_name, arrow_dt, false),
            ]));
            return RecordBatch::try_new(schema, vec![concat]).map_err(FluxError::Arrow);
        }

        // Binary fallback: use the Vec<Vec<u8>> path.
        let mut all_strings: Vec<Vec<u8>> = Vec::new();
        for &block_idx in candidates {
            let start = footer.blocks[block_idx].block_offset as usize;
            let (strings, _) = string_compressor::decompress(&data[start..])?;
            all_strings.extend(strings);
        }
        let refs: Vec<&[u8]> = all_strings.iter().map(|b| b.as_slice()).collect();
        let array: ArrayRef = Arc::new(BinaryArray::from(refs));
        let schema = Arc::new(Schema::new(vec![
            Field::new(&self.column_name, arrow_dt, false),
        ]));
        RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow)
    }
}

impl LoomDecompressor for FluxReader {
    /// Decompress `data` with optional predicate pushdown.
    ///
    /// Optimized hot read path:
    /// 1. Parse Atlas footer (O(blocks), zero-copy).
    /// 2. Predicate pushdown: skip non-matching blocks.
    /// 3. **Parallel block decompression** via rayon.
    /// 4. Type-aware reconstruction: use `dtype_tag` from BlockMeta to build
    ///    the correct Arrow array type (not always UInt64).
    /// 5. Build Arrow RecordBatch with correct schema.
    fn decompress(&self, data: &[u8], predicate: &Predicate) -> FluxResult<RecordBatch> {
        use rayon::prelude::*;

        let footer = AtlasFooter::from_file_tail(data)?;

        if footer.blocks.is_empty() {
            return empty_batch(&self.column_name);
        }

        let candidates = footer.candidate_blocks(predicate);

        if candidates.is_empty() {
            return empty_batch(&self.column_name);
        }

        // If footer has a schema tree (nested types), use reassemble path.
        if !footer.schema.is_empty() {
            return self.decompress_with_schema(data, &footer);
        }

        // Determine the output dtype from the first candidate block.
        let dtype_tag = footer.blocks[candidates[0]].dtype_tag;

        // ── String/Binary path: decompress via string_compressor ─────────
        if matches!(dtype_tag, FluxDType::Utf8 | FluxDType::LargeUtf8
                             | FluxDType::Binary | FluxDType::LargeBinary)
        {
            return self.decompress_string_blocks(data, &footer, &candidates, dtype_tag);
        }

        // ── Numeric path: parallel decompress to u64 ─────────────────────
        let chunks: Vec<Vec<u64>> = candidates
            .par_iter()
            .map(|&block_idx| {
                let meta = &footer.blocks[block_idx];
                let start = meta.block_offset as usize;
                if start >= data.len() {
                    return Err(FluxError::InvalidFile(format!(
                        "block {block_idx} offset {start} exceeds file length {}",
                        data.len()
                    )));
                }
                let block_slice = &data[start..];
                let (values, _) = decompress_block_to_u64(block_slice)?;
                Ok(values)
            })
            .collect::<FluxResult<_>>()?;

        // Flatten all chunks.
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut all_values: Vec<u64> = Vec::with_capacity(total);
        for chunk in chunks {
            all_values.extend(chunk);
        }

        // Reconstruct the typed Arrow array from the dtype_tag.
        let arrow_dt = dtype_tag.to_arrow();
        let array = reconstruct_array_u64(all_values, dtype_tag)?;

        let schema = Arc::new(Schema::new(vec![
            Field::new(&self.column_name, arrow_dt, false),
        ]));
        RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn empty_batch(col_name: &str) -> FluxResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(col_name, DataType::UInt64, false),
    ]));
    let array = Arc::new(UInt64Array::from(Vec::<u64>::new()));
    RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-decompressed leaf data for parallel nested decompression
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-decompressed leaf data, keyed by column_id.
enum LeafData {
    /// Numeric values as u64 (covers all types ≤ 64 bits).
    Numeric(Vec<u64>),
    /// String/binary arrays (already built as Arrow arrays).
    StringArrays(Vec<ArrayRef>),
}

/// Recursively collect all leaf (column_id, block_indices) from a schema tree.
fn collect_leaf_blocks(
    descriptors: &[ColumnDescriptor],
    footer: &AtlasFooter,
    out: &mut Vec<(u16, Vec<usize>)>,
) {
    for desc in descriptors {
        let dtype_tag = FluxDType::from_u8(desc.dtype_tag).unwrap_or(FluxDType::UInt64);
        match dtype_tag {
            FluxDType::StructContainer | FluxDType::ListContainer | FluxDType::MapContainer => {
                collect_leaf_blocks(&desc.children, footer, out);
            }
            _ => {
                let col_id = desc.column_id;
                let blocks: Vec<usize> = footer.blocks.iter()
                    .enumerate()
                    .filter(|(_, m)| m.column_id == col_id)
                    .map(|(i, _)| i)
                    .collect();
                if !blocks.is_empty() {
                    out.push((col_id, blocks));
                }
            }
        }
    }
}

/// Reassemble a column tree using pre-decompressed leaf data (no I/O).
fn reassemble_column_fast(
    footer: &AtlasFooter,
    desc: &ColumnDescriptor,
    leaves: &std::collections::HashMap<u16, LeafData>,
) -> FluxResult<(ArrayRef, Field)> {
    let dtype_tag = FluxDType::from_u8(desc.dtype_tag).unwrap_or(FluxDType::UInt64);

    match dtype_tag {
        FluxDType::StructContainer => {
            let mut child_arrays: Vec<(Arc<Field>, ArrayRef)> = Vec::new();
            for child_desc in &desc.children {
                let (arr, field) = reassemble_column_fast(footer, child_desc, leaves)?;
                child_arrays.push((Arc::new(field), arr));
            }
            let struct_arr = StructArray::from(child_arrays);
            let fields: Fields = struct_arr.fields().clone();
            let field = Field::new(&desc.name, DataType::Struct(fields), false);
            Ok((Arc::new(struct_arr), field))
        }
        FluxDType::ListContainer => {
            if desc.children.len() < 2 {
                return Err(FluxError::InvalidFile("List must have ≥2 children".into()));
            }
            let (lengths_arr, _) = reassemble_column_fast(footer, &desc.children[0], leaves)?;
            let lengths_i32 = lengths_arr.as_any().downcast_ref::<Int32Array>()
                .ok_or_else(|| FluxError::InvalidFile("List lengths not Int32".into()))?;

            let offsets = if desc.children[0].name == "__const_len" {
                let clen = lengths_i32.value(0);
                // Determine list count from bases (3 children) or values (2 children).
                let child_col_id = desc.children[1].column_id;
                let n = leaves.get(&child_col_id)
                    .map(|ld| match ld { LeafData::Numeric(v) => v.len(), LeafData::StringArrays(a) => a.iter().map(|x| x.len()).sum() })
                    .unwrap_or(0);
                let num_lists = if desc.children.len() == 3 { n } else { n / clen as usize };
                let mut offs = Vec::with_capacity(num_lists + 1);
                offs.push(0i32);
                for _ in 0..num_lists {
                    offs.push(offs.last().unwrap() + clen);
                }
                offs
            } else {
                let mut offs = Vec::with_capacity(lengths_i32.len() + 1);
                offs.push(0i32);
                for i in 0..lengths_i32.len() {
                    offs.push(offs.last().unwrap() + lengths_i32.value(i));
                }
                offs
            };

            let (values_arr, values_field) = if desc.children.len() == 3 {
                let (bases_arr, _) = reassemble_column_fast(footer, &desc.children[1], leaves)?;
                let (deltas_arr, _) = reassemble_column_fast(footer, &desc.children[2], leaves)?;
                let bases_u64 = bases_arr.as_any().downcast_ref::<UInt64Array>()
                    .ok_or_else(|| FluxError::InvalidFile("bases not UInt64".into()))?;
                let deltas_u64 = deltas_arr.as_any().downcast_ref::<UInt64Array>()
                    .ok_or_else(|| FluxError::InvalidFile("deltas not UInt64".into()))?;
                let num_lists = offsets.len() - 1;
                let total_values = *offsets.last().unwrap() as usize;
                let mut values = Vec::with_capacity(total_values);
                for i in 0..num_lists {
                    let start = offsets[i] as usize;
                    let end = offsets[i + 1] as usize;
                    let base = bases_u64.value(i);
                    for j in start..end {
                        values.push((base.wrapping_add(deltas_u64.value(j))) as i64);
                    }
                }
                let arr: ArrayRef = Arc::new(Int64Array::from(values));
                let field = Field::new("item", DataType::Int64, true);
                (arr, field)
            } else {
                reassemble_column_fast(footer, &desc.children[1], leaves)?
            };

            let offsets_buf = arrow_buffer::OffsetBuffer::new(
                arrow_buffer::ScalarBuffer::from(offsets),
            );
            let list_arr = ListArray::new(
                Arc::new(values_field), offsets_buf, values_arr, None,
            );
            let field = Field::new(&desc.name, list_arr.data_type().clone(), false);
            Ok((Arc::new(list_arr), field))
        }
        FluxDType::MapContainer => {
            if desc.children.len() != 3 {
                return Err(FluxError::InvalidFile("Map must have 3 children".into()));
            }
            let (lengths_arr, _) = reassemble_column_fast(footer, &desc.children[0], leaves)?;
            let (keys_arr, keys_field) = reassemble_column_fast(footer, &desc.children[1], leaves)?;
            let (vals_arr, vals_field) = reassemble_column_fast(footer, &desc.children[2], leaves)?;

            let lengths_i32 = lengths_arr.as_any().downcast_ref::<Int32Array>()
                .ok_or_else(|| FluxError::InvalidFile("Map lengths not Int32".into()))?;
            let mut offsets = Vec::with_capacity(lengths_i32.len() + 1);
            offsets.push(0i32);
            for i in 0..lengths_i32.len() {
                offsets.push(offsets.last().unwrap() + lengths_i32.value(i));
            }
            let offsets_buf = arrow_buffer::OffsetBuffer::new(
                arrow_buffer::ScalarBuffer::from(offsets),
            );
            let entries_fields = Fields::from(vec![
                Arc::new(keys_field), Arc::new(vals_field),
            ]);
            let entries = StructArray::try_new(
                entries_fields.clone(), vec![keys_arr, vals_arr], None,
            ).map_err(FluxError::Arrow)?;
            let entries_field = Arc::new(Field::new(
                "entries", DataType::Struct(entries_fields), false,
            ));
            let map_arr = arrow_array::MapArray::new(
                entries_field.clone(), offsets_buf, entries, None, false,
            );
            let field = Field::new(&desc.name, map_arr.data_type().clone(), false);
            Ok((Arc::new(map_arr), field))
        }
        // Leaf: look up pre-decompressed data.
        _ => {
            let col_id = desc.column_id;
            let leaf = leaves.get(&col_id).ok_or_else(|| {
                FluxError::InvalidFile(format!("missing leaf data for column_id {col_id}"))
            })?;

            match leaf {
                LeafData::Numeric(values) => {
                    let array = reconstruct_array_u64_ref(values, dtype_tag)?;
                    let field = Field::new(&desc.name, dtype_tag.to_arrow(), false);
                    Ok((array, field))
                }
                LeafData::StringArrays(arrays) => {
                    let array: ArrayRef = if arrays.len() == 1 {
                        arrays[0].clone()
                    } else {
                        let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
                        arrow::compute::concat(&refs).map_err(FluxError::Arrow)?
                    };
                    let field = Field::new(&desc.name, dtype_tag.to_arrow(), false);
                    Ok((array, field))
                }
            }
        }
    }
}

/// Reconstruct a typed Arrow array from decompressed `u128` values
/// [`FluxDType`] tag. This is the central type-dispatch for decompression —
/// the inverse of the bit-cast performed in `extract_column_data()`.
pub fn reconstruct_array(values: &[u128], dtype_tag: FluxDType) -> FluxResult<ArrayRef> {
    // Delegate to the u64 path for all types ≤ 64 bits.
    let v: Vec<u64> = values.iter().map(|&x| x as u64).collect();
    reconstruct_array_u64(v, dtype_tag)
}

/// Reconstruct a typed Arrow array directly from `u64` values.
///
/// This is the hot path: avoids the `u128 → u64` conversion for types that
/// were already decompressed to `u64` via [`decompress_block_to_u64`].
pub fn reconstruct_array_u64(values: Vec<u64>, dtype_tag: FluxDType) -> FluxResult<ArrayRef> {
    reconstruct_array_u64_ref(&values, dtype_tag)
}

/// Reconstruct a typed Arrow array from a borrowed `&[u64]` slice.
fn reconstruct_array_u64_ref(values: &[u64], dtype_tag: FluxDType) -> FluxResult<ArrayRef> {
    match dtype_tag {
        FluxDType::UInt64 => {
            Ok(Arc::new(UInt64Array::from(values.to_vec())))
        }
        FluxDType::UInt32 => {
            let v: Vec<u32> = values.iter().map(|&x| x as u32).collect();
            Ok(Arc::new(UInt32Array::from(v)))
        }
        FluxDType::UInt16 => {
            let v: Vec<u16> = values.iter().map(|&x| x as u16).collect();
            Ok(Arc::new(UInt16Array::from(v)))
        }
        FluxDType::UInt8 => {
            let v: Vec<u8> = values.iter().map(|&x| x as u8).collect();
            Ok(Arc::new(UInt8Array::from(v)))
        }
        FluxDType::Int64 => {
            let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            Ok(Arc::new(Int64Array::from(v)))
        }
        FluxDType::Int32 => {
            let v: Vec<i32> = values.iter().map(|&x| x as u32 as i32).collect();
            Ok(Arc::new(Int32Array::from(v)))
        }
        FluxDType::Int16 => {
            let v: Vec<i16> = values.iter().map(|&x| x as u16 as i16).collect();
            Ok(Arc::new(Int16Array::from(v)))
        }
        FluxDType::Int8 => {
            let v: Vec<i8> = values.iter().map(|&x| x as u8 as i8).collect();
            Ok(Arc::new(Int8Array::from(v)))
        }
        FluxDType::Float64 => {
            let v: Vec<f64> = values.iter().map(|&x| f64::from_bits(x)).collect();
            Ok(Arc::new(Float64Array::from(v)))
        }
        FluxDType::Float32 => {
            let v: Vec<f32> = values.iter().map(|&x| f32::from_bits(x as u32)).collect();
            Ok(Arc::new(Float32Array::from(v)))
        }
        FluxDType::Boolean => {
            let v: Vec<bool> = values.iter().map(|&x| x != 0).collect();
            Ok(Arc::new(BooleanArray::from(v)))
        }
        FluxDType::Date32 => {
            let v: Vec<i32> = values.iter().map(|&x| x as u32 as i32).collect();
            Ok(Arc::new(Date32Array::from(v)))
        }
        FluxDType::Date64 => {
            let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            Ok(Arc::new(Date64Array::from(v)))
        }
        FluxDType::TimestampSecond => {
            let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            Ok(Arc::new(TimestampSecondArray::from(v)))
        }
        FluxDType::TimestampMillis => {
            let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            Ok(Arc::new(TimestampMillisecondArray::from(v)))
        }
        FluxDType::TimestampMicros => {
            let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            Ok(Arc::new(TimestampMicrosecondArray::from(v)))
        }
        FluxDType::TimestampNanos => {
            let v: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            Ok(Arc::new(TimestampNanosecondArray::from(v)))
        }
        other => Err(FluxError::Internal(format!(
            "reconstruct_array: unsupported dtype_tag {:?}", other
        ))),
    }
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
    use arrow_array::{UInt64Array, Array as _};
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
        // Create data with drift to force multiple segments:
        // Block 1: constant values (RLE) → z_min=z_max=42
        // Block 2: sequential 0..1024 (DeltaDelta) → z_min=0, z_max=1023
        // Block 3: sequential 5000..6024 (DeltaDelta) → z_min=5000, z_max=6023
        let mut input: Vec<u64> = vec![42; 1024];
        input.extend(0u64..1024);
        input.extend(5000u64..6024);
        let batch = make_batch(input);

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();

        let footer = crate::atlas::AtlasFooter::from_file_tail(&bytes).unwrap();
        assert!(
            footer.blocks.len() >= 2,
            "expected multiple blocks from drift, got {}",
            footer.blocks.len(),
        );

        // Predicate: value > 4999 — should skip constant and low-range blocks.
        let reader = FluxReader::new("value");
        let pred = Predicate::GreaterThan {
            column: "value".into(),
            value: 4999,
        };
        let out_batch = reader.decompress(&bytes, &pred).unwrap();
        let col = out_batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        // Values 5000..6023 must all be present.
        let returned: std::collections::HashSet<u64> = col.values().iter().copied().collect();
        for v in 5000u64..6024 {
            assert!(returned.contains(&v), "missing value {v} from predicate result");
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

    // ── Type fidelity round-trip tests ────────────────────────────────────

    #[test]
    fn int64_round_trip_preserves_type() {
        let values: Vec<i64> = (-500i64..524).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
        ]));
        let arr = Arc::new(Int64Array::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        // Must reconstruct as Int64, not UInt64.
        assert_eq!(*out.schema().field(0).data_type(), DataType::Int64);
        let col = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let got: Vec<i64> = col.values().to_vec();
        assert_eq!(got, values);
    }

    #[test]
    fn float64_round_trip_preserves_type() {
        // Use values in a narrow range so bit patterns stay small and
        // don't trigger the delta compressor's 64-bit overflow edge case.
        let values: Vec<f64> = (0..1024).map(|i| 100.0 + (i % 50) as f64 * 0.5).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
        ]));
        let arr = Arc::new(Float64Array::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Float64);
        let col = out.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let got: Vec<f64> = col.values().to_vec();
        assert_eq!(got, values);
    }

    #[test]
    fn boolean_round_trip_preserves_type() {
        let values: Vec<bool> = (0..1024).map(|i| i % 3 == 0).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Boolean, false),
        ]));
        let arr = Arc::new(BooleanArray::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Boolean);
        let col = out.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
        let got: Vec<bool> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, values);
    }

    #[test]
    fn timestamp_micros_round_trip_preserves_type() {
        let values: Vec<i64> = (0..1024).map(|i| 1_700_000_000_000_000i64 + i * 1_000_000).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None), false),
        ]));
        let arr = Arc::new(TimestampMicrosecondArray::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(
            *out.schema().field(0).data_type(),
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
        );
        let col = out.column(0).as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
        let got: Vec<i64> = col.values().to_vec();
        assert_eq!(got, values);
    }

    #[test]
    fn date32_round_trip_preserves_type() {
        let values: Vec<i32> = (0..1024).map(|i| 18262 + (i % 3650)).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Date32, false),
        ]));
        let arr = Arc::new(Date32Array::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Date32);
        let col = out.column(0).as_any().downcast_ref::<Date32Array>().unwrap();
        let got: Vec<i32> = col.values().to_vec();
        assert_eq!(got, values);
    }

    // ── String / Tier 2 round-trip tests ──────────────────────────────────

    #[test]
    fn utf8_low_cardinality_round_trip() {
        // 5 unique strings across 1000 rows → dict path.
        let strings: Vec<String> = (0..5).map(|i| format!("category_{i:03}")).collect();
        let values: Vec<&str> = (0..1000).map(|i| strings[i % 5].as_str()).collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
        ]));
        let arr = Arc::new(StringArray::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Utf8);
        let col = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(col.len(), 1000);
        for i in 0..1000 {
            assert_eq!(col.value(i), values[i], "mismatch at row {i}");
        }
    }

    #[test]
    fn utf8_high_cardinality_round_trip() {
        // All unique strings → raw_lz4 path.
        let values: Vec<String> = (0..500).map(|i| format!("unique_{i:06}")).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
        ]));
        let arr = Arc::new(StringArray::from(refs.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Utf8);
        let col = out.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(col.len(), 500);
        for i in 0..500 {
            assert_eq!(col.value(i), refs[i], "mismatch at row {i}");
        }
    }

    // ── Nested / Tier 3 round-trip tests ─────────────────────────────────

    #[test]
    fn struct_round_trip() {
        use arrow_array::StructArray;
        use arrow_schema::Fields;

        let id_arr = UInt64Array::from(vec![1u64, 2, 3, 4]);
        let age_arr = Int32Array::from(vec![25i32, 30, 35, 40]);

        let fields = Fields::from(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("age", DataType::Int32, false),
        ]);
        let struct_arr = StructArray::from(vec![
            (Arc::new(Field::new("id", DataType::UInt64, false)), Arc::new(id_arr) as ArrayRef),
            (Arc::new(Field::new("age", DataType::Int32, false)), Arc::new(age_arr) as ArrayRef),
        ]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Struct(fields), false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_arr)]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("val");
        let out = reader.decompress_all(&bytes).unwrap();

        // The output should have the nested struct with correct child arrays.
        assert_eq!(out.num_columns(), 1);
        let out_struct = out.column(0).as_any().downcast_ref::<StructArray>().unwrap();
        let out_id = out_struct.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let out_age = out_struct.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out_id.values().to_vec(), vec![1u64, 2, 3, 4]);
        assert_eq!(out_age.values().to_vec(), vec![25i32, 30, 35, 40]);
    }

    #[test]
    fn list_round_trip() {
        use arrow_array::builder::{ListBuilder, Int64Builder};

        let mut builder = ListBuilder::new(Int64Builder::new());
        // Row 0: [10, 11]
        builder.values().append_value(10);
        builder.values().append_value(11);
        builder.append(true);
        // Row 1: [20]
        builder.values().append_value(20);
        builder.append(true);
        // Row 2: [30, 31, 32]
        builder.values().append_value(30);
        builder.values().append_value(31);
        builder.values().append_value(32);
        builder.append(true);

        let list_arr = builder.finish();
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", list_arr.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(list_arr)]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("val");
        let out = reader.decompress_all(&bytes).unwrap();

        let out_list = out.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(out_list.len(), 3);
        // Verify offsets: [0, 2, 3, 6]
        let offsets: Vec<i32> = out_list.offsets().iter().copied().collect();
        assert_eq!(offsets, vec![0, 2, 3, 6]);
    }
}
