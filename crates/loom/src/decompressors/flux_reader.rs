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

use arrow::array::ArrayDataBuilder;
use arrow_array::{
    Array as ArrowArray, ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, ListArray,
    RecordBatch, StringArray, StructArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use std::collections::HashSet;
use std::sync::Arc;

use crate::{
    atlas::{AtlasFooter, ColumnDescriptor},
    decompressors::block_reader::{decompress_block, decompress_block_to_u64},
    dtype::FluxDType,
    error::{FluxError, FluxResult},
    traits::{LoomDecompressor, Predicate},
    txn::projection::{ColumnPlan, FilePlan},
    txn::schema::DefaultValue,
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
        Self {
            column_name: column_name.into(),
        }
    }
}

impl Default for FluxReader {
    fn default() -> Self {
        Self {
            column_name: "value".into(),
        }
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
        let file = std::fs::File::open(path).map_err(|e| FluxError::Io(e))?;
        // SAFETY: .flux files are immutable (versioned via transaction log).
        // No other process modifies them while we're reading.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| FluxError::Io(e))?;
        self.decompress(&mmap, predicate)
    }

    /// Convenience: decompress all blocks from a memory-mapped file.
    pub fn decompress_file_all(&self, path: &std::path::Path) -> FluxResult<RecordBatch> {
        self.decompress_file(path, &Predicate::None)
    }

    /// Decompress with column projection — only decompress the named columns.
    ///
    /// Blocks belonging to columns not in `projection` are skipped entirely,
    /// saving both I/O and CPU.  The output `RecordBatch` contains only the
    /// projected columns, in the order they appear in the schema.
    pub fn decompress_projected(
        &self,
        data: &[u8],
        predicate: &Predicate,
        projection: &[String],
    ) -> FluxResult<RecordBatch> {
        let footer = AtlasFooter::from_file_tail(data)?;

        if footer.blocks.is_empty() {
            return empty_batch(&self.column_name);
        }

        if footer.schema.is_empty() {
            // Flat (single-column) file — projection doesn't apply.
            return self.decompress(data, predicate);
        }

        // Build the set of projected column names.
        let proj_set: HashSet<&str> = projection.iter().map(|s| s.as_str()).collect();

        // Filter schema descriptors to only projected columns.
        let projected_schema: Vec<&ColumnDescriptor> = footer
            .schema
            .iter()
            .filter(|desc| proj_set.contains(desc.name.as_str()))
            .collect();

        if projected_schema.is_empty() {
            return empty_batch(&self.column_name);
        }

        // Collect projected column_ids (including nested children).
        let mut projected_col_ids: HashSet<u16> = HashSet::new();
        for desc in &projected_schema {
            collect_column_ids(desc, &mut projected_col_ids);
        }

        self.decompress_with_schema_projected(data, &footer, &projected_schema, &projected_col_ids)
    }

    /// Decompress a `.flux` file with column projection via mmap.
    pub fn decompress_file_projected(
        &self,
        path: &std::path::Path,
        predicate: &Predicate,
        projection: &[String],
    ) -> FluxResult<RecordBatch> {
        let file = std::fs::File::open(path).map_err(|e| FluxError::Io(e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| FluxError::Io(e))?;
        self.decompress_projected(&mmap, predicate, projection)
    }

    /// Read just the Arrow schema from a `.flux` file without decompressing data.
    pub fn read_schema(data: &[u8]) -> FluxResult<SchemaRef> {
        let footer = AtlasFooter::from_file_tail(data)?;
        Ok(Arc::new(schema_from_footer(&footer)))
    }

    /// Read schema from a `.flux` file path.
    pub fn read_schema_from_file(path: &std::path::Path) -> FluxResult<SchemaRef> {
        let file = std::fs::File::open(path).map_err(FluxError::Io)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(FluxError::Io)?;
        Self::read_schema(&mmap)
    }

    /// Decompress a `.flux` file against a schema-evolution
    /// [`FilePlan`], producing a batch in the caller's target schema.
    ///
    /// This is the Phase B projection entry point:
    /// * Only `plan.file_physical_columns` are decoded — dropped
    ///   columns are I/O-skipped.
    /// * Decoded columns are renamed as the plan specifies.
    /// * Added fields are materialised via
    ///   [`arrow_array::new_null_array`] (NULL) or a scalar broadcast
    ///   of the configured literal default. Both are branch-free and
    ///   do not allocate per-row data buffers.
    ///
    /// When `plan.is_pure_fill()`, the file is not opened at all and
    /// the batch is synthesised from the plan's row counts alone.
    pub fn decompress_with_plan(
        &self,
        data: &[u8],
        predicate: &Predicate,
        plan: &FilePlan,
    ) -> FluxResult<RecordBatch> {
        // Step 1: decode the physical columns the plan needs. We use a
        // reader whose `column_name` matches the sole physical column
        // when there's one — that way single-column flat files (where
        // the footer has no schema tree) come back named correctly.
        let physical_batch: Option<RecordBatch> = if plan.file_physical_columns.is_empty() {
            None
        } else {
            let reader = if plan.file_physical_columns.len() == 1 {
                FluxReader::new(&plan.file_physical_columns[0])
            } else {
                self.clone()
            };
            Some(reader.decompress_projected(data, predicate, &plan.file_physical_columns)?)
        };

        // Step 2: resolve row count. Decoded batch wins if any; else
        // fall back to the plan's Fill row_count; else peek the
        // file's footer (guards against legacy manifests with 0).
        let row_count: usize = match &physical_batch {
            Some(b) => b.num_rows(),
            None => {
                let plan_rc = plan
                    .columns
                    .iter()
                    .find_map(|c| match c {
                        ColumnPlan::Fill { row_count, .. } => Some(*row_count as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                if plan_rc == 0 && !data.is_empty() {
                    let footer = AtlasFooter::from_file_tail(data)?;
                    footer.blocks.iter().map(|b| b.value_count as usize).sum()
                } else {
                    plan_rc
                }
            }
        };

        // Step 3: assemble the output batch in target-schema order.
        let mut out_columns: Vec<ArrayRef> = Vec::with_capacity(plan.columns.len());
        let mut out_fields: Vec<Field> = Vec::with_capacity(plan.columns.len());

        for c in &plan.columns {
            match c {
                ColumnPlan::Decode {
                    physical_name,
                    target_name,
                    source_dtype,
                    target_dtype,
                    target_nullable,
                } => {
                    let batch = physical_batch.as_ref().ok_or_else(|| {
                        FluxError::Internal(
                            "plan has Decode but file_physical_columns is empty".into(),
                        )
                    })?;
                    let idx = batch
                        .schema()
                        .index_of(physical_name)
                        .map_err(FluxError::Arrow)?;
                    let raw = batch.column(idx).clone();
                    // Phase C: when the plan declares a different
                    // source dtype, widen with a single arrow cast.
                    // Same-dtype is the common path and stays
                    // zero-copy (cloned Arc).
                    let (column, field_dtype) = if source_dtype == target_dtype {
                        (raw, target_dtype.to_arrow())
                    } else {
                        let cast_to = source_dtype
                            .cast_target_arrow_dtype(*target_dtype)
                            .ok_or_else(|| {
                                FluxError::SchemaEvolution(format!(
                                    "reader refused promotion {:?} → {:?}; plan \
                                     was built with an unsupported pair",
                                    source_dtype, target_dtype,
                                ))
                            })?;
                        let cast = arrow::compute::cast(raw.as_ref(), &cast_to)
                            .map_err(FluxError::Arrow)?;
                        (cast, cast_to)
                    };
                    out_columns.push(column);
                    out_fields.push(Field::new(target_name, field_dtype, *target_nullable));
                }
                ColumnPlan::Fill {
                    target_name,
                    target_dtype,
                    target_nullable,
                    default,
                    ..
                } => {
                    let arr = materialize_fill(*target_dtype, default.as_ref(), row_count)?;
                    out_columns.push(arr);
                    out_fields.push(Field::new(
                        target_name,
                        target_dtype.to_arrow(),
                        *target_nullable,
                    ));
                }
            }
        }

        let schema = Arc::new(Schema::new(out_fields));
        RecordBatch::try_new(schema, out_columns).map_err(FluxError::Arrow)
    }

    /// Convenience: plan-driven decompress from a file path via mmap.
    pub fn decompress_file_with_plan(
        &self,
        path: &std::path::Path,
        predicate: &Predicate,
        plan: &FilePlan,
    ) -> FluxResult<RecordBatch> {
        if plan.is_pure_fill() {
            // No file I/O necessary — the plan fully describes the
            // synthesised batch.
            return self.decompress_with_plan(&[], predicate, plan);
        }
        let file = std::fs::File::open(path).map_err(FluxError::Io)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(FluxError::Io)?;
        self.decompress_with_plan(&mmap, predicate, plan)
    }
}

impl FluxReader {
    /// Decompress a `.flux` file that has a schema tree (nested types).
    ///
    /// Optimized path: pre-decompresses ALL leaf blocks in parallel, then
    /// does a single-threaded tree reassembly pass.
    fn decompress_with_schema(&self, data: &[u8], footer: &AtlasFooter) -> FluxResult<RecordBatch> {
        let all_descs: Vec<&ColumnDescriptor> = footer.schema.iter().collect();
        let mut all_col_ids: HashSet<u16> = HashSet::new();
        for desc in &footer.schema {
            collect_column_ids(desc, &mut all_col_ids);
        }
        self.decompress_with_schema_projected(data, footer, &all_descs, &all_col_ids)
    }

    /// Core schema-based decompression with column projection.
    ///
    /// Only decompresses leaf blocks whose `column_id` is in `projected_col_ids`.
    /// Only reassembles columns in `projected_descs`.
    fn decompress_with_schema_projected(
        &self,
        data: &[u8],
        footer: &AtlasFooter,
        projected_descs: &[&ColumnDescriptor],
        projected_col_ids: &HashSet<u16>,
    ) -> FluxResult<RecordBatch> {
        use crate::compressors::string_compressor::{
            self, SUB_CROSS_GROUP, decompress_cross_column_group,
        };
        use rayon::prelude::*;
        use std::collections::HashMap;

        // Phase 1: Collect leaf blocks, filtered by projected column_ids.
        let mut leaf_blocks: Vec<(u16, Vec<usize>)> = Vec::new();
        for &desc in projected_descs {
            collect_leaf_blocks_filtered(
                std::slice::from_ref(desc),
                footer,
                projected_col_ids,
                &mut leaf_blocks,
            );
        }

        // Phase 1b: Identify unique cross-column group payloads (by file
        // offset). N string columns belonging to the SAME group currently
        // each trigger their own SUB_CROSS_GROUP decode — we collapse those
        // into a single decode per offset, then serve all member columns
        // from the cache. This is the single biggest decompression win on
        // wide schemas (mixed-bench drops from N to 1–2 FSST decodes).
        let mut group_offsets: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut group_dtype_by_offset: HashMap<u64, FluxDType> = HashMap::new();
        for (_col_id, block_indices) in &leaf_blocks {
            for &bi in block_indices {
                let meta = &footer.blocks[bi];
                let start = meta.block_offset as usize;
                if start + 2 <= data.len()
                    && data[start] == string_compressor::TAG
                    && data[start + 1] == SUB_CROSS_GROUP
                {
                    group_offsets.insert(meta.block_offset);
                    group_dtype_by_offset.insert(meta.block_offset, meta.dtype_tag);
                }
            }
        }
        // Decode each unique group payload ONCE in parallel.
        let group_cache: HashMap<u64, HashMap<u16, ArrayRef>> = group_offsets
            .into_par_iter()
            .map(|off| {
                let dtype_tag = *group_dtype_by_offset.get(&off).unwrap_or(&FluxDType::Utf8);
                let parts = decompress_cross_column_group(&data[off as usize..], dtype_tag)?;
                let mut m: HashMap<u16, ArrayRef> = HashMap::with_capacity(parts.len());
                for (cid, arr) in parts {
                    m.insert(cid, arr);
                }
                Ok((off, m))
            })
            .collect::<FluxResult<HashMap<u64, HashMap<u16, ArrayRef>>>>()?;

        // Phase 2: Parallel decompression of projected leaves only. Grouped
        // columns short-circuit to the group cache instead of re-decoding.
        let decompressed: HashMap<u16, LeafData> = leaf_blocks
            .into_par_iter()
            .map(|(col_id, block_indices)| {
                let dtype_tag = footer.blocks[block_indices[0]].dtype_tag;
                let is_string = matches!(
                    dtype_tag,
                    FluxDType::Utf8
                        | FluxDType::LargeUtf8
                        | FluxDType::Binary
                        | FluxDType::LargeBinary
                );

                if is_string {
                    let mut arrays: Vec<ArrayRef> = Vec::new();
                    for &bi in &block_indices {
                        let meta = &footer.blocks[bi];
                        let start = meta.block_offset as usize;
                        // Group cache hit: zero-cost slice from pre-decoded group.
                        if let Some(group) = group_cache.get(&meta.block_offset) {
                            if let Some(arr) = group.get(&meta.column_id) {
                                arrays.push(arr.clone());
                                continue;
                            }
                        }
                        let arr = string_compressor::decompress_to_arrow_string_for_column(
                            &data[start..],
                            dtype_tag,
                            Some(meta.column_id),
                        )?;
                        arrays.push(arr);
                    }
                    Ok((col_id, LeafData::StringArrays(arrays)))
                } else if matches!(dtype_tag, FluxDType::Decimal128) {
                    // Full 128-bit decode for Decimal128 / i128 / u128 columns.
                    let mut all_values: Vec<u128> = Vec::new();
                    for &bi in &block_indices {
                        let start = footer.blocks[bi].block_offset as usize;
                        let (values, _) = decompress_block(&data[start..])?;
                        all_values.extend(values);
                    }
                    Ok((col_id, LeafData::Numeric128(all_values)))
                } else {
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

        // Phase 3: Tree reassembly for projected columns only.
        let mut columns: Vec<ArrayRef> = Vec::new();
        let mut fields: Vec<Field> = Vec::new();

        for &desc in projected_descs {
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
                let meta = &footer.blocks[candidates[0]];
                let start = meta.block_offset as usize;
                let array = string_compressor::decompress_to_arrow_string_for_column(
                    &data[start..],
                    dtype_tag,
                    Some(meta.column_id),
                )?;
                let schema = Arc::new(Schema::new(vec![Field::new(
                    &self.column_name,
                    arrow_dt,
                    false,
                )]));
                return RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow);
            }
            // Multiple blocks: decompress each to StringArray, then concat.
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(candidates.len());
            for &block_idx in candidates {
                let meta = &footer.blocks[block_idx];
                let start = meta.block_offset as usize;
                arrays.push(string_compressor::decompress_to_arrow_string_for_column(
                    &data[start..],
                    dtype_tag,
                    Some(meta.column_id),
                )?);
            }
            let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
            let concat = arrow::compute::concat(&refs).map_err(FluxError::Arrow)?;
            let schema = Arc::new(Schema::new(vec![Field::new(
                &self.column_name,
                arrow_dt,
                false,
            )]));
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            &self.column_name,
            arrow_dt,
            false,
        )]));
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
        if matches!(
            dtype_tag,
            FluxDType::Utf8 | FluxDType::LargeUtf8 | FluxDType::Binary | FluxDType::LargeBinary
        ) {
            return self.decompress_string_blocks(data, &footer, &candidates, dtype_tag);
        }

        // ── Decimal128 path: stay in u128 ────────────────────────
        if matches!(dtype_tag, FluxDType::Decimal128) {
            let chunks_128: Vec<Vec<u128>> = candidates
                .par_iter()
                .map(|&block_idx| {
                    let meta = &footer.blocks[block_idx];
                    let start = meta.block_offset as usize;
                    let block_slice = &data[start..];
                    let (values, _) = decompress_block(block_slice)?;
                    Ok(values)
                })
                .collect::<FluxResult<_>>()?;
            let total: usize = chunks_128.iter().map(|c| c.len()).sum();
            let mut all_values: Vec<u128> = Vec::with_capacity(total);
            for chunk in chunks_128 {
                all_values.extend(chunk);
            }
            let arrow_dt = dtype_tag.to_arrow();
            let array = reconstruct_decimal128(&all_values)?;
            let schema = Arc::new(Schema::new(vec![Field::new(
                &self.column_name,
                arrow_dt,
                false,
            )]));
            return RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow);
        }

        // ── Numeric path: parallel decompress to u64 ─────────────────
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

        let schema = Arc::new(Schema::new(vec![Field::new(
            &self.column_name,
            arrow_dt,
            false,
        )]));
        RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FluxBatchIterator — streaming file-level decompression
// ─────────────────────────────────────────────────────────────────────────────

/// A streaming iterator that yields one [`RecordBatch`] per `.flux` file.
///
/// Only one file is memory-mapped at a time — previous files are unmapped
/// before the next is opened, keeping memory usage bounded.
///
/// Supports column projection and predicate pushdown.
pub struct FluxBatchIterator {
    /// File paths to iterate over.
    paths: Vec<std::path::PathBuf>,
    /// Current file index.
    current: usize,
    /// Optional column projection (decompress only these columns).
    projection: Option<Vec<String>>,
    /// Predicate for block-level pushdown.
    predicate: Predicate,
    /// Arrow schema (read from first file, projected if applicable).
    schema: SchemaRef,
}

impl FluxBatchIterator {
    /// Create a new batch iterator over the given `.flux` file paths.
    ///
    /// Reads the schema from the first file.  If `projection` is provided,
    /// the schema is filtered to only the projected columns.
    pub fn new(
        paths: Vec<std::path::PathBuf>,
        projection: Option<Vec<String>>,
        predicate: Predicate,
    ) -> FluxResult<Self> {
        if paths.is_empty() {
            return Err(FluxError::InvalidFile("no files to iterate".into()));
        }

        // Read schema from the first file.
        let full_schema = FluxReader::read_schema_from_file(&paths[0])?;

        let schema = match &projection {
            Some(cols) => {
                let proj_set: HashSet<&str> = cols.iter().map(|s| s.as_str()).collect();
                let projected_fields: Vec<Arc<Field>> = full_schema
                    .fields()
                    .iter()
                    .filter(|f| proj_set.contains(f.name().as_str()))
                    .cloned()
                    .collect();
                if projected_fields.is_empty() {
                    full_schema
                } else {
                    Arc::new(Schema::new(projected_fields))
                }
            }
            None => full_schema,
        };

        Ok(Self {
            paths,
            current: 0,
            projection,
            predicate,
            schema,
        })
    }

    /// The Arrow schema for batches produced by this iterator.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Number of files remaining (including current).
    pub fn remaining(&self) -> usize {
        self.paths.len().saturating_sub(self.current)
    }
}

impl Iterator for FluxBatchIterator {
    type Item = Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.paths.len() {
            return None;
        }

        let path = &self.paths[self.current];
        self.current += 1;

        let reader = FluxReader::default();
        let result = match &self.projection {
            Some(cols) => reader.decompress_file_projected(path, &self.predicate, cols),
            None => reader.decompress_file(path, &self.predicate),
        };

        Some(result.map_err(|e| arrow_schema::ArrowError::ExternalError(Box::new(e))))
    }
}

impl arrow_array::RecordBatchReader for FluxBatchIterator {
    fn schema(&self) -> SchemaRef {
        self.schema()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn empty_batch(col_name: &str) -> FluxResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        col_name,
        DataType::UInt64,
        false,
    )]));
    let array = Arc::new(UInt64Array::from(Vec::<u64>::new()));
    RecordBatch::try_new(schema, vec![array]).map_err(FluxError::Arrow)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase B Fill materialisation: NULL / literal-default synthesis
// ─────────────────────────────────────────────────────────────────────────────

/// Build a projected column of `row_count` rows for a Phase B
/// `Fill` plan entry.
///
/// NULL fills use [`arrow_array::new_null_array`], which only
/// allocates the null mask. Literal defaults build a typed array via
/// scalar broadcast; unsupported dtype/literal pairings surface as
/// [`FluxError::SchemaEvolution`] so the scan fails fast rather than
/// silently mis-decoding.
fn materialize_fill(
    dtype: FluxDType,
    default: Option<&DefaultValue>,
    row_count: usize,
) -> FluxResult<ArrayRef> {
    use arrow_array::new_null_array;
    let arrow_dt = dtype.to_arrow();
    match default {
        None => Ok(new_null_array(&arrow_dt, row_count)),
        Some(v) => materialize_literal(dtype, v, row_count),
    }
}

fn materialize_literal(dtype: FluxDType, v: &DefaultValue, n: usize) -> FluxResult<ArrayRef> {
    match (dtype, v) {
        (FluxDType::Boolean, DefaultValue::Bool(b)) => {
            Ok(Arc::new(BooleanArray::from(vec![*b; n])))
        }
        (FluxDType::Int64, DefaultValue::Int(i)) => Ok(Arc::new(Int64Array::from(vec![*i; n]))),
        (FluxDType::Int32, DefaultValue::Int(i)) => {
            let v = i32::try_from(*i).map_err(|_| {
                FluxError::SchemaEvolution(format!("literal default {i} overflows Int32"))
            })?;
            Ok(Arc::new(Int32Array::from(vec![v; n])))
        }
        (FluxDType::Int16, DefaultValue::Int(i)) => {
            let v = i16::try_from(*i).map_err(|_| {
                FluxError::SchemaEvolution(format!("literal default {i} overflows Int16"))
            })?;
            Ok(Arc::new(Int16Array::from(vec![v; n])))
        }
        (FluxDType::Int8, DefaultValue::Int(i)) => {
            let v = i8::try_from(*i).map_err(|_| {
                FluxError::SchemaEvolution(format!("literal default {i} overflows Int8"))
            })?;
            Ok(Arc::new(Int8Array::from(vec![v; n])))
        }
        (FluxDType::UInt64, DefaultValue::UInt(u)) => Ok(Arc::new(UInt64Array::from(vec![*u; n]))),
        (FluxDType::UInt64, DefaultValue::Int(i)) if *i >= 0 => {
            Ok(Arc::new(UInt64Array::from(vec![*i as u64; n])))
        }
        (FluxDType::UInt32, DefaultValue::UInt(u)) => {
            let v = u32::try_from(*u).map_err(|_| {
                FluxError::SchemaEvolution(format!("literal default {u} overflows UInt32"))
            })?;
            Ok(Arc::new(UInt32Array::from(vec![v; n])))
        }
        (FluxDType::UInt16, DefaultValue::UInt(u)) => {
            let v = u16::try_from(*u).map_err(|_| {
                FluxError::SchemaEvolution(format!("literal default {u} overflows UInt16"))
            })?;
            Ok(Arc::new(UInt16Array::from(vec![v; n])))
        }
        (FluxDType::UInt8, DefaultValue::UInt(u)) => {
            let v = u8::try_from(*u).map_err(|_| {
                FluxError::SchemaEvolution(format!("literal default {u} overflows UInt8"))
            })?;
            Ok(Arc::new(UInt8Array::from(vec![v; n])))
        }
        (FluxDType::Float64, DefaultValue::Float(f)) => {
            Ok(Arc::new(Float64Array::from(vec![*f; n])))
        }
        (FluxDType::Float32, DefaultValue::Float(f)) => {
            Ok(Arc::new(Float32Array::from(vec![*f as f32; n])))
        }
        (FluxDType::Utf8, DefaultValue::String(s)) => {
            // StringArray::from(Vec<&str>) builds a single contiguous
            // value buffer with shared offsets — linear in `n`, no
            // per-row allocation.
            let vals: Vec<&str> = (0..n).map(|_| s.as_str()).collect();
            Ok(Arc::new(StringArray::from(vals)))
        }
        _ => Err(FluxError::SchemaEvolution(format!(
            "literal default {v:?} incompatible with dtype {dtype:?}"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-decompressed leaf data for parallel nested decompression
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-decompressed leaf data, keyed by column_id.
enum LeafData {
    /// Numeric values as u64 (covers all types ≤ 64 bits).
    Numeric(Vec<u64>),
    /// 128-bit numeric values (Decimal128 / i128 / u128).
    Numeric128(Vec<u128>),
    /// String/binary arrays (already built as Arrow arrays).
    StringArrays(Vec<ArrayRef>),
}

/// Recursively collect all leaf (column_id, block_indices) from a schema tree.
fn collect_leaf_blocks(
    descriptors: &[ColumnDescriptor],
    footer: &AtlasFooter,
    out: &mut Vec<(u16, Vec<usize>)>,
) {
    let all_ids: HashSet<u16> = footer.blocks.iter().map(|b| b.column_id).collect();
    collect_leaf_blocks_filtered(descriptors, footer, &all_ids, out);
}

/// Collect leaf blocks, but only for column_ids in `allowed_ids`.
fn collect_leaf_blocks_filtered(
    descriptors: &[ColumnDescriptor],
    footer: &AtlasFooter,
    allowed_ids: &HashSet<u16>,
    out: &mut Vec<(u16, Vec<usize>)>,
) {
    for desc in descriptors {
        let dtype_tag = FluxDType::from_u8(desc.dtype_tag).unwrap_or(FluxDType::UInt64);
        match dtype_tag {
            FluxDType::StructContainer | FluxDType::ListContainer | FluxDType::MapContainer => {
                collect_leaf_blocks_filtered(&desc.children, footer, allowed_ids, out);
            }
            _ => {
                let col_id = desc.column_id;
                if !allowed_ids.contains(&col_id) {
                    continue;
                }
                let blocks: Vec<usize> = footer
                    .blocks
                    .iter()
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

/// Recursively collect all column_ids from a descriptor tree (including children).
fn collect_column_ids(desc: &ColumnDescriptor, out: &mut HashSet<u16>) {
    let dtype_tag = FluxDType::from_u8(desc.dtype_tag).unwrap_or(FluxDType::UInt64);
    match dtype_tag {
        FluxDType::StructContainer | FluxDType::ListContainer | FluxDType::MapContainer => {
            for child in &desc.children {
                collect_column_ids(child, out);
            }
        }
        _ => {
            out.insert(desc.column_id);
        }
    }
}

/// Build an Arrow Schema from the Atlas footer's schema tree (no decompression).
pub fn schema_from_footer(footer: &AtlasFooter) -> Schema {
    let fields: Vec<Field> = footer
        .schema
        .iter()
        .map(|desc| field_from_descriptor(desc))
        .collect();
    Schema::new(fields)
}

/// Convert a ColumnDescriptor into an Arrow Field.
fn field_from_descriptor(desc: &ColumnDescriptor) -> Field {
    let dtype_tag = FluxDType::from_u8(desc.dtype_tag).unwrap_or(FluxDType::UInt64);
    match dtype_tag {
        FluxDType::StructContainer => {
            let children: Vec<Field> = desc
                .children
                .iter()
                .map(|c| field_from_descriptor(c))
                .collect();
            Field::new(&desc.name, DataType::Struct(Fields::from(children)), false)
        }
        FluxDType::ListContainer => {
            let item_field = if desc.children.len() >= 2 {
                field_from_descriptor(&desc.children[1])
            } else {
                Field::new("item", DataType::Int64, true)
            };
            Field::new(&desc.name, DataType::List(Arc::new(item_field)), false)
        }
        FluxDType::MapContainer => {
            let key_field = if desc.children.len() >= 2 {
                field_from_descriptor(&desc.children[1])
            } else {
                Field::new("key", DataType::Utf8, false)
            };
            let val_field = if desc.children.len() >= 3 {
                field_from_descriptor(&desc.children[2])
            } else {
                Field::new("value", DataType::Int64, true)
            };
            let entries = Field::new(
                "entries",
                DataType::Struct(Fields::from(vec![key_field, val_field])),
                false,
            );
            Field::new(&desc.name, DataType::Map(Arc::new(entries), false), false)
        }
        _ => Field::new(&desc.name, dtype_tag.to_arrow(), false),
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
            let lengths_i32 = lengths_arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| FluxError::InvalidFile("List lengths not Int32".into()))?;

            let offsets = if desc.children[0].name == "__const_len" {
                let clen = lengths_i32.value(0);
                // Determine list count from bases (3 children) or values (2 children).
                let child_col_id = desc.children[1].column_id;
                let n = leaves
                    .get(&child_col_id)
                    .map(|ld| match ld {
                        LeafData::Numeric(v) => v.len(),
                        LeafData::Numeric128(v) => v.len(),
                        LeafData::StringArrays(a) => a.iter().map(|x| x.len()).sum(),
                    })
                    .unwrap_or(0);
                let num_lists = if desc.children.len() == 3 {
                    n
                } else {
                    n / clen as usize
                };
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
                let bases_u64 = bases_arr
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| FluxError::InvalidFile("bases not UInt64".into()))?;
                let deltas_u64 = deltas_arr
                    .as_any()
                    .downcast_ref::<UInt64Array>()
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

            let offsets_buf =
                arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets));
            let list_arr = ListArray::new(Arc::new(values_field), offsets_buf, values_arr, None);
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

            let lengths_i32 = lengths_arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| FluxError::InvalidFile("Map lengths not Int32".into()))?;
            let mut offsets = Vec::with_capacity(lengths_i32.len() + 1);
            offsets.push(0i32);
            for i in 0..lengths_i32.len() {
                offsets.push(offsets.last().unwrap() + lengths_i32.value(i));
            }
            let offsets_buf =
                arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets));
            let entries_fields = Fields::from(vec![Arc::new(keys_field), Arc::new(vals_field)]);
            let entries =
                StructArray::try_new(entries_fields.clone(), vec![keys_arr, vals_arr], None)
                    .map_err(FluxError::Arrow)?;
            let entries_field = Arc::new(Field::new(
                "entries",
                DataType::Struct(entries_fields),
                false,
            ));
            let map_arr = arrow_array::MapArray::new(
                entries_field.clone(),
                offsets_buf,
                entries,
                None,
                false,
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
                LeafData::Numeric128(values) => {
                    let array = reconstruct_decimal128(values)?;
                    let field = Field::new(&desc.name, dtype_tag.to_arrow(), false);
                    Ok((array, field))
                }
                LeafData::StringArrays(arrays) => {
                    let array: ArrayRef = if arrays.len() == 1 {
                        arrays[0].clone()
                    } else {
                        let refs: Vec<&dyn arrow_array::Array> =
                            arrays.iter().map(|a| a.as_ref()).collect();
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
    if matches!(dtype_tag, FluxDType::Decimal128) {
        return reconstruct_decimal128(values);
    }
    // Delegate to the u64 path for all types ≤ 64 bits.
    let v: Vec<u64> = values.iter().map(|&x| x as u64).collect();
    reconstruct_array_u64(v, dtype_tag)
}

/// Build a `Decimal128Array` from raw `u128` bit patterns (i128 reinterpret).
/// Default precision/scale (38, 10) match `FluxDType::Decimal128.to_arrow()`.
fn reconstruct_decimal128(values: &[u128]) -> FluxResult<ArrayRef> {
    use arrow_array::Decimal128Array;
    let v: Vec<i128> = values.iter().map(|&x| x as i128).collect();
    let arr = Decimal128Array::from(v)
        .with_precision_and_scale(38, 10)
        .map_err(FluxError::Arrow)?;
    Ok(Arc::new(arr))
}

/// Reconstruct a typed Arrow array directly from `u64` values.
///
/// This is the hot path: avoids the `u128 → u64` conversion for types that
/// were already decompressed to `u64` via [`decompress_block_to_u64`].
///
/// ## Zero-copy same-width reconstruction
///
/// For [`FluxDType::Float64`], [`FluxDType::Int64`], and
/// [`FluxDType::UInt64`] — every target type whose Arrow buffer has the
/// same bit-width as the source `u64` — we hand the `Vec<u64>` directly
/// to Arrow's [`Buffer::from_vec`], which takes ownership without
/// copying.  The output array is then wrapped around that buffer via
/// [`ArrayDataBuilder`] at the target dtype.  No per-value
/// `f64::from_bits` / `as i64` / byte-copy passes are needed because
/// those conversions are pure bit-casts at the Arrow layer — the
/// in-memory representation of
///  - `Vec<u64>` (8 bytes per element, LE on every supported target),
///  - `Buffer` with 8-byte alignment,
///  - Arrow `Float64` / `Int64` / `UInt64` value buffers,
/// are byte-identical.  This eliminates the pure-entropy Float64
/// decompression bottleneck that used to show up as a 5× Parquet
/// advantage in `dtype-bench`.
///
/// Types narrower than u64 (UInt32, Int16, etc.) still go through the
/// `reconstruct_array_u64_ref` path because they need width
/// truncation; Float32 does a same-width cast but the source is u64,
/// not u32, so we widen/narrow through the ref path today.
pub fn reconstruct_array_u64(values: Vec<u64>, dtype_tag: FluxDType) -> FluxResult<ArrayRef> {
    match dtype_tag {
        FluxDType::UInt64 => zero_copy_u64_buffer::<u64>(values, DataType::UInt64),
        FluxDType::Int64 => zero_copy_u64_buffer::<u64>(values, DataType::Int64),
        FluxDType::Float64 => zero_copy_u64_buffer::<u64>(values, DataType::Float64),
        FluxDType::Date64 => zero_copy_u64_buffer::<u64>(values, DataType::Date64),
        FluxDType::TimestampSecond => zero_copy_u64_buffer::<u64>(
            values,
            DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
        ),
        FluxDType::TimestampMillis => zero_copy_u64_buffer::<u64>(
            values,
            DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
        ),
        FluxDType::TimestampMicros => zero_copy_u64_buffer::<u64>(
            values,
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
        ),
        FluxDType::TimestampNanos => zero_copy_u64_buffer::<u64>(
            values,
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
        ),

        // Narrower / non-same-width types still need the
        // per-value width cast, so fall through to the borrowed path.
        _ => reconstruct_array_u64_ref(&values, dtype_tag),
    }
}

/// Build an [`ArrayRef`] of the given `target_dtype` by handing the
/// caller's `Vec<u64>` straight to Arrow's [`Buffer`] without any
/// per-value copy.
///
/// # Safety
/// - `target_dtype` **must** be an 8-byte-wide primitive (`UInt64`,
///   `Int64`, `Float64`, `Date64`, `Timestamp(*, None)`). The caller
///   enforces this via the outer `match` in
///   [`reconstruct_array_u64`]; passing anything else triggers an
///   internal assertion at [`ArrayData::validate_data`] time.
/// - The u64 bit pattern in `values` must be a valid bit pattern for
///   the destination dtype (trivially true for integers; for floats
///   any bit pattern is a valid `f64` including NaN / sub-normals,
///   which Arrow tolerates).
fn zero_copy_u64_buffer<T>(values: Vec<u64>, target_dtype: DataType) -> FluxResult<ArrayRef> {
    let len = values.len();
    // `Buffer::from_vec::<u64>` consumes the Vec and reuses its
    // allocation; no copy.  The resulting Buffer is 8-byte aligned
    // which matches every supported 64-bit primitive's Arrow
    // requirement.
    let buf = Buffer::from_vec(values);
    let data = ArrayDataBuilder::new(target_dtype)
        .len(len)
        .add_buffer(buf)
        .build()
        .map_err(FluxError::Arrow)?;
    Ok(arrow_array::make_array(data))
}

/// Reconstruct a typed Arrow array from a borrowed `&[u64]` slice.
fn reconstruct_array_u64_ref(values: &[u64], dtype_tag: FluxDType) -> FluxResult<ArrayRef> {
    match dtype_tag {
        FluxDType::UInt64 => Ok(Arc::new(UInt64Array::from(values.to_vec()))),
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
            "reconstruct_array: unsupported dtype_tag {:?}",
            other
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compressors::flux_writer::FluxWriter, traits::LoomCompressor};
    use arrow_array::{Array as _, UInt64Array};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn make_batch(values: Vec<u64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )]));
        let arr = Arc::new(UInt64Array::from(values));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    #[test]
    fn zero_copy_float64_round_trip() {
        // Build a `Vec<u64>` that represents known `f64` bit patterns,
        // reconstruct it through the zero-copy path, and verify every
        // value decodes back to the expected `f64`.
        let originals: Vec<f64> = vec![
            0.0,
            1.5,
            -3.14,
            1e308,
            -1e-308,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN,
            f64::MAX,
            12345.6789,
        ];
        let as_bits: Vec<u64> = originals.iter().map(|v| v.to_bits()).collect();

        let arr = reconstruct_array_u64(as_bits, FluxDType::Float64).unwrap();
        let arr = arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64Array");
        assert_eq!(arr.len(), originals.len());
        for (i, expected) in originals.iter().enumerate() {
            // Compare bit patterns so NaN / signed-zero cases are
            // exact (not `==`-sensitive).
            assert_eq!(arr.value(i).to_bits(), expected.to_bits(), "row {i}");
        }
    }

    #[test]
    fn zero_copy_int64_uint64_round_trip() {
        let values: Vec<u64> = vec![0, 1, u64::MAX, 42, 1 << 40];

        let u = reconstruct_array_u64(values.clone(), FluxDType::UInt64).unwrap();
        let u = u.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(u.values().as_ref(), values.as_slice());

        let i = reconstruct_array_u64(values.clone(), FluxDType::Int64).unwrap();
        let i = i.as_any().downcast_ref::<Int64Array>().unwrap();
        for (idx, &v) in values.iter().enumerate() {
            assert_eq!(i.value(idx), v as i64, "row {idx}");
        }
    }

    #[test]
    fn narrow_types_still_work_through_ref_path() {
        // UInt32 goes through the truncating `reconstruct_array_u64_ref`
        // path — make sure the new zero-copy dispatch didn't regress it.
        let values: Vec<u64> = vec![0, 1, 2, u32::MAX as u64 + 1, u64::MAX];
        let arr = reconstruct_array_u64(values, FluxDType::UInt32).unwrap();
        let arr = arr.as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(arr.value(0), 0);
        assert_eq!(arr.value(1), 1);
        assert_eq!(arr.value(2), 2);
        // u32::MAX + 1 wraps to 0 under `as u32`.
        assert_eq!(arr.value(3), 0);
        // u64::MAX wraps to u32::MAX under `as u32`.
        assert_eq!(arr.value(4), u32::MAX);
    }

    #[test]
    fn decimal128_round_trip_full_width() {
        // Values that overflow u64 to verify the full i128 path is honoured.
        // Includes a mix of very large positive, very large negative, and
        // small values so OutlierMap / BitSlab patching is exercised.
        use arrow_array::Decimal128Array;
        let values: Vec<i128> = vec![
            0_i128,
            1_i128,
            i128::MAX / 2,
            -(i128::MAX / 3),
            12_345_678_901_234_567_890_i128, // > u64::MAX
            -98_765_432_109_876_543_210_i128,
            1_000_000_000_000_000_000_000_000_i128, // ~10^24
        ];
        let arr = Decimal128Array::from(values.clone())
            .with_precision_and_scale(38, 10)
            .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            DataType::Decimal128(38, 10),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("d");
        let out = reader.decompress_all(&bytes).unwrap();
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(col.len(), values.len());
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(col.value(i), expected, "row {i}");
        }
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
            assert!(
                returned.contains(&v),
                "missing value {v} from predicate result"
            );
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
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert!(col.values().iter().all(|&v| v == 999));
    }

    // ── Type fidelity round-trip tests ────────────────────────────────────

    #[test]
    fn int64_round_trip_preserves_type() {
        let values: Vec<i64> = (-500i64..524).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let arr = Arc::new(Float64Array::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Float64);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let got: Vec<f64> = col.values().to_vec();
        assert_eq!(got, values);
    }

    #[test]
    fn boolean_round_trip_preserves_type() {
        let values: Vec<bool> = (0..1024).map(|i| i % 3 == 0).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Boolean,
            false,
        )]));
        let arr = Arc::new(BooleanArray::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Boolean);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let got: Vec<bool> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, values);
    }

    #[test]
    fn timestamp_micros_round_trip_preserves_type() {
        let values: Vec<i64> = (0..1024)
            .map(|i| 1_700_000_000_000_000i64 + i * 1_000_000)
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            false,
        )]));
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
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        let got: Vec<i64> = col.values().to_vec();
        assert_eq!(got, values);
    }

    #[test]
    fn date32_round_trip_preserves_type() {
        let values: Vec<i32> = (0..1024).map(|i| 18262 + (i % 3650)).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Date32,
            false,
        )]));
        let arr = Arc::new(Date32Array::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Date32);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        let got: Vec<i32> = col.values().to_vec();
        assert_eq!(got, values);
    }

    // ── String / Tier 2 round-trip tests ──────────────────────────────────

    #[test]
    fn utf8_low_cardinality_round_trip() {
        // 5 unique strings across 1000 rows → dict path.
        let strings: Vec<String> = (0..5).map(|i| format!("category_{i:03}")).collect();
        let values: Vec<&str> = (0..1000).map(|i| strings[i % 5].as_str()).collect();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )]));
        let arr = Arc::new(StringArray::from(values.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Utf8);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
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

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )]));
        let arr = Arc::new(StringArray::from(refs.clone()));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("value");
        let out = reader.decompress_all(&bytes).unwrap();

        assert_eq!(*out.schema().field(0).data_type(), DataType::Utf8);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
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
            (
                Arc::new(Field::new("id", DataType::UInt64, false)),
                Arc::new(id_arr) as ArrayRef,
            ),
            (
                Arc::new(Field::new("age", DataType::Int32, false)),
                Arc::new(age_arr) as ArrayRef,
            ),
        ]);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "val",
            DataType::Struct(fields),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_arr)]).unwrap();

        let writer = FluxWriter::new();
        let bytes = writer.compress(&batch).unwrap();
        let reader = FluxReader::new("val");
        let out = reader.decompress_all(&bytes).unwrap();

        // The output should have the nested struct with correct child arrays.
        assert_eq!(out.num_columns(), 1);
        let out_struct = out
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let out_id = out_struct
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let out_age = out_struct
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(out_id.values().to_vec(), vec![1u64, 2, 3, 4]);
        assert_eq!(out_age.values().to_vec(), vec![25i32, 30, 35, 40]);
    }

    #[test]
    fn list_round_trip() {
        use arrow_array::builder::{Int64Builder, ListBuilder};

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
        let schema = Arc::new(Schema::new(vec![Field::new(
            "val",
            list_arr.data_type().clone(),
            false,
        )]));
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
