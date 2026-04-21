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

/// Controls how sibling string columns share compression state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringGroupingMode {
    /// Never group — each string column is compressed independently.
    Off,
    /// Automatically group compatible string columns (default).
    Auto,
    /// Explicit grouping: each inner `Vec<String>` is one group of column
    /// names that share compression state.
    Manual(Vec<Vec<String>>),
}

impl Default for StringGroupingMode {
    fn default() -> Self {
        StringGroupingMode::Auto
    }
}

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
    /// Cross-column string grouping policy.
    pub string_grouping: StringGroupingMode,
    /// String columns that must never be cross-column grouped. Partition
    /// source columns are always isolated automatically (see `partition_spec`)
    /// but callers may add more via this list.
    pub isolated_string_columns: Vec<String>,
    /// The active partition spec for this write. Any column referenced as a
    /// `source_column` in its fields is treated as `isolated` — it's
    /// compressed standalone so partition pruning + pushdown stay correct.
    pub partition_spec: Option<crate::txn::partition::PartitionSpec>,
    /// Phase E: optional map from *batch column name* to logical
    /// `field_id`. When populated, each top-level
    /// [`ColumnDescriptor`] emitted in the Atlas footer is stamped
    /// with the matching `field_id`. Unknown names fall back to
    /// `field_id = None`, preserving pre-Phase-E behaviour.
    pub field_ids: std::collections::HashMap<String, u32>,
}

impl Default for FluxWriter {
    fn default() -> Self {
        Self {
            force_strategy: None,
            profile: crate::CompressionProfile::Speed,
            u64_only: false,
            string_grouping: StringGroupingMode::default(),
            isolated_string_columns: Vec::new(),
            partition_spec: None,
            field_ids: std::collections::HashMap::new(),
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

    /// Set the cross-column string grouping mode.
    pub fn with_string_grouping(mut self, mode: StringGroupingMode) -> Self {
        self.string_grouping = mode;
        self
    }

    /// Mark additional string columns as always-isolated (never grouped).
    pub fn with_isolated_string_columns(mut self, cols: Vec<String>) -> Self {
        self.isolated_string_columns = cols;
        self
    }

    /// Set the active partition spec. Source columns in the spec are
    /// automatically added to the isolated set.
    pub fn with_partition_spec(
        mut self,
        spec: Option<crate::txn::partition::PartitionSpec>,
    ) -> Self {
        self.partition_spec = spec;
        self
    }

    /// Phase E: attach a name→`field_id` map so every top-level
    /// [`ColumnDescriptor`] emitted into the Atlas footer carries a
    /// logical `field_id`.
    ///
    /// Typically fed from
    /// [`FluxTable::field_ids_for_current_schema`]:
    /// ```no_run
    /// # use loom::compressors::flux_writer::FluxWriter;
    /// # use loom::txn::FluxTable;
    /// # let table = FluxTable::open("t").unwrap();
    /// # let batch = unimplemented!();
    /// let writer = FluxWriter::new()
    ///     .with_field_ids(table.field_ids_for_current_schema().unwrap());
    /// let bytes = <_ as loom::traits::LoomCompressor>::compress(&writer, &batch).unwrap();
    /// ```
    pub fn with_field_ids(
        mut self,
        field_ids: std::collections::HashMap<String, u32>,
    ) -> Self {
        self.field_ids = field_ids;
        self
    }

    /// Returns the full set of isolated string column names (explicit +
    /// derived from the active partition spec).
    pub fn isolated_set(&self) -> std::collections::HashSet<String> {
        let mut out: std::collections::HashSet<String> =
            self.isolated_string_columns.iter().cloned().collect();
        if let Some(spec) = &self.partition_spec {
            for f in &spec.fields {
                out.insert(f.source_column.clone());
            }
        }
        out
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

/// Heuristics for deciding whether two string columns can share an FSST/dict.
/// All members must be the same Arrow data type, have similar average row
/// length (within 2×), similar charset signature, and the combined corpus
/// must clear a minimum where shared training amortises (~64 KB) AND stay
/// under a ceiling where independent FSST tables already amortise on each
/// column. Grouping further decides via a probe bakeoff whether the shared
/// table actually beats independent per-column compression — on datasets
/// with heterogeneous vocabularies (country codes vs URLs vs user-agents)
/// one shared symbol table is often WORSE than per-column tables.
const GROUP_MIN_COMBINED_BYTES: usize = 64 * 1024;
/// Upper bound on total grouped-corpus bytes. Above this we don't group:
/// (a) each column is already large enough to amortise its own FSST/dict,
/// and (b) concatenating N columns into a single buffer can blow memory on
/// wide tables (Databricks/Spark workers are routinely OOM-killed for
/// multi-GB intermediates). 128 MB keeps peak write-side RSS bounded.
const GROUP_MAX_COMBINED_BYTES: usize = 128 * 1024 * 1024;
/// Per-column size ceiling: columns bigger than this amortise their own
/// FSST/dict comfortably and don't need a shared table.
const GROUP_PER_COLUMN_MAX_BYTES: usize = 32 * 1024 * 1024;
/// Probe window for the grouping profitability bakeoff (rows per column).
const GROUP_PROBE_ROWS: usize = 512;
/// Shared-group output must beat the independent-columns sum by this margin
/// on the probe. Otherwise we keep the columns independent.
const GROUP_WIN_MARGIN: f64 = 0.95;
const GROUP_LENGTH_RATIO: f64 = 2.0;
const GROUP_MIN_GROUP_SIZE: usize = 2;

/// Decide which string columns from `batch` should be grouped together based on
/// the writer's [`StringGroupingMode`] and isolated set. Returns a vector of
/// groups, each group being the column indices to compress as one cross-column
/// block. Auto mode now runs a small probe bakeoff and a set of memory-safety
/// guards before committing to a group.
fn plan_string_groups(
    batch: &RecordBatch,
    mode: &StringGroupingMode,
    isolated: &std::collections::HashSet<String>,
) -> Vec<Vec<usize>> {
    if matches!(mode, StringGroupingMode::Off) {
        return Vec::new();
    }

    let schema = batch.schema();
    #[derive(Clone)]
    struct Cand {
        idx: usize,
        dtype: arrow_schema::DataType,
        bytes: usize,
        rows: usize,
        avg_len: f64,
        charset: u8,       // rough signature: 0 short-low-ASCII, 1 long-ASCII, 2 non-ASCII
    }
    let mut candidates: Vec<Cand> = Vec::new();
    for i in 0..batch.num_columns() {
        let array = batch.column(i).as_ref();
        let dt = array.data_type().clone();
        let is_string = matches!(
            &dt,
            arrow_schema::DataType::Utf8 | arrow_schema::DataType::LargeUtf8
            | arrow_schema::DataType::Binary | arrow_schema::DataType::LargeBinary
        );
        if !is_string {
            continue;
        }
        let name = schema.field(i).name();
        if isolated.contains(name) {
            continue;
        }
        let bytes = arrow_string_value_bytes(array);
        let rows = array.len();
        if rows == 0 {
            continue;
        }
        // Per-column upper bound: huge columns amortise their own FSST/dict
        // and the memory cost of grouping isn't worth it.
        if bytes > GROUP_PER_COLUMN_MAX_BYTES {
            continue;
        }
        let avg_len = (bytes as f64) / (rows as f64);
        let charset = probe_charset(array);
        candidates.push(Cand { idx: i, dtype: dt, bytes, rows, avg_len, charset });
    }

    if candidates.len() < GROUP_MIN_GROUP_SIZE {
        return Vec::new();
    }

    match mode {
        StringGroupingMode::Off => Vec::new(),
        StringGroupingMode::Manual(groups) => {
            let mut name_to_idx = std::collections::HashMap::new();
            for c in &candidates {
                name_to_idx.insert(schema.field(c.idx).name().clone(), c.idx);
            }
            groups.iter().map(|g| {
                g.iter().filter_map(|n| name_to_idx.get(n).copied()).collect()
            }).filter(|v: &Vec<usize>| v.len() >= GROUP_MIN_GROUP_SIZE).collect()
        }
        StringGroupingMode::Auto => {
            // Bucket by (dtype, log2(avg_len), charset) so truly similar
            // columns cluster. Then apply a combined-size floor + ceiling
            // and a probe bakeoff before locking in the grouping.
            let mut buckets: std::collections::HashMap<(arrow_schema::DataType, i32, u8), Vec<Cand>> =
                std::collections::HashMap::new();
            for c in &candidates {
                let bucket_key = ((c.avg_len.max(1.0)).log(GROUP_LENGTH_RATIO).floor()) as i32;
                buckets.entry((c.dtype.clone(), bucket_key, c.charset)).or_default().push(c.clone());
            }
            let mut out = Vec::new();
            for (_, members) in buckets {
                if members.len() < GROUP_MIN_GROUP_SIZE { continue; }
                let total_bytes: usize = members.iter().map(|m| m.bytes).sum();
                if total_bytes < GROUP_MIN_COMBINED_BYTES { continue; }
                if total_bytes > GROUP_MAX_COMBINED_BYTES {
                    // Memory guard: skip grouping on wide tables. Each
                    // column will still hit FSST independently.
                    continue;
                }
                let member_indices: Vec<usize> = members.iter().map(|m| m.idx).collect();
                // Probe bakeoff: compress a small sample from each member
                // both independently and grouped. Only group if shared
                // beats independent by ≥5%.
                if grouping_beats_independent(batch, &member_indices) {
                    out.push(member_indices);
                }
            }
            out
        }
    }
}

/// Probe the first few values to get a rough charset bucket:
/// 0 = short low-ASCII (codes, enums), 1 = long ASCII (URLs, logs),
/// 2 = contains non-ASCII bytes (binary / multibyte UTF-8).
fn probe_charset(array: &dyn Array) -> u8 {
    use arrow_schema::DataType;
    let mut any_non_ascii = false;
    let mut total_len = 0usize;
    let mut n = 0usize;
    let probe = 32usize.min(array.len());
    match array.data_type() {
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
            for i in 0..probe {
                let s = a.value(i).as_bytes();
                total_len += s.len();
                if s.iter().any(|&b| b >= 0x80) { any_non_ascii = true; }
                n += 1;
            }
        }
        DataType::LargeUtf8 => {
            let a = array.as_any().downcast_ref::<arrow_array::LargeStringArray>().unwrap();
            for i in 0..probe {
                let s = a.value(i).as_bytes();
                total_len += s.len();
                if s.iter().any(|&b| b >= 0x80) { any_non_ascii = true; }
                n += 1;
            }
        }
        _ => return 2,
    }
    if any_non_ascii { 2 }
    else if n > 0 && (total_len as f64) / (n as f64) >= 32.0 { 1 }
    else { 0 }
}

/// Probe-based profitability check: compress the first GROUP_PROBE_ROWS rows
/// of each group member independently, then compress the same rows as one
/// cross-column group, compare the resulting byte sizes. Returns `true` if
/// the grouped version beats the independent sum by `GROUP_WIN_MARGIN`.
fn grouping_beats_independent(batch: &RecordBatch, member_indices: &[usize]) -> bool {
    use crate::compressors::string_compressor::{
        compress_array_with_profile, compress_cross_column_group_with_profile,
    };
    // Slice each candidate column to a probe window.
    let mut probe_arrays: Vec<arrow_array::ArrayRef> = Vec::with_capacity(member_indices.len());
    for &i in member_indices {
        let col = batch.column(i);
        let n = col.len().min(GROUP_PROBE_ROWS);
        probe_arrays.push(col.slice(0, n));
    }
    let independent_sum: usize = probe_arrays.iter()
        .map(|a| compress_array_with_profile(
            a.as_ref(), crate::CompressionProfile::Speed,
        ).map(|b| b.len()).unwrap_or(usize::MAX))
        .sum();
    // For the grouped probe we reuse the synthetic column_id range 0..N.
    let cols: Vec<(u16, &dyn Array)> = probe_arrays.iter()
        .enumerate().map(|(j, a)| (j as u16, a.as_ref())).collect();
    let grouped_len = compress_cross_column_group_with_profile(
        &cols, crate::CompressionProfile::Speed,
    ).map(|b| b.len()).unwrap_or(usize::MAX);
    (grouped_len as f64) < (independent_sum as f64) * GROUP_WIN_MARGIN
}

/// Best-effort byte-size estimate for a string/binary Arrow array.
fn arrow_string_value_bytes(array: &dyn Array) -> usize {
    use arrow_schema::DataType;
    match array.data_type() {
        DataType::Utf8 => array.as_any().downcast_ref::<arrow_array::StringArray>()
            .map(|a| a.value_data().len()).unwrap_or(0),
        DataType::LargeUtf8 => array.as_any().downcast_ref::<arrow_array::LargeStringArray>()
            .map(|a| a.value_data().len()).unwrap_or(0),
        DataType::Binary => array.as_any().downcast_ref::<arrow_array::BinaryArray>()
            .map(|a| a.value_data().len()).unwrap_or(0),
        DataType::LargeBinary => array.as_any().downcast_ref::<arrow_array::LargeBinaryArray>()
            .map(|a| a.value_data().len()).unwrap_or(0),
        _ => 0,
    }
}

impl LoomCompressor for FluxWriter {
    fn compress(&self, batch: &RecordBatch) -> FluxResult<Vec<u8>> {
        use rayon::prelude::*;

        let profile = self.profile;
        let force = self.force_strategy;
        let batch_schema = batch.schema();
        let mut all_blocks: Vec<Vec<(Vec<u8>, BlockMeta)>> = Vec::new();
        let mut schema_descriptors: Vec<ColumnDescriptor> = Vec::new();

        // Plan cross-column string groups before the main loop. Columns in a
        // group are skipped in the per-column loop and emitted afterwards as
        // SUB_CROSS_GROUP blocks (one shared payload, multiple BlockMeta
        // entries pointing at the same offset).
        let isolated = self.isolated_set();
        let groups = plan_string_groups(batch, &self.string_grouping, &isolated);
        let grouped_cols: std::collections::HashSet<usize> =
            groups.iter().flat_map(|g| g.iter().copied()).collect();

        for col_idx in 0..batch.num_columns() {
            if grouped_cols.contains(&col_idx) {
                continue; // Handled by the post-loop group emission.
            }
            let array = batch.column(col_idx).as_ref();
            let route = dtype_router::route(array.data_type());

            let field_name = batch_schema.field(col_idx).name().clone();

            // Decimal128: full i128 / u128 path. The standard ColumnData
            // pipeline truncates to u64; for Decimal128 we extract i128
            // values directly and feed the existing u128 chunk pipeline so
            // the OutlierMap / BitSlab can store the full 128-bit width.
            if matches!(array.data_type(), DataType::Decimal128(_, _)) {
                let blocks = compress_decimal128_column(
                    array, col_idx as u16, force, profile,
                )?;
                all_blocks.push(blocks);
                let fid = self.field_ids.get(&field_name).copied();
                schema_descriptors.push(ColumnDescriptor {
                    name: field_name,
                    dtype_tag: FluxDType::Decimal128.as_u8(),
                    children: Vec::new(),
                    column_id: col_idx as u16,
                    field_id: fid,
                });
                continue;
            }

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
                    let fid = self.field_ids.get(&field_name).copied();
                    schema_descriptors.push(ColumnDescriptor {
                        name: field_name,
                        dtype_tag: dtype_tag.as_u8(),
                        children: Vec::new(),
                        column_id: col_idx as u16,
                        field_id: fid,
                    });
                }
                RouteDecision::Classify => {
                    let col = extract_column_data(array, col_idx as u16)?;
                    let dtype_tag = col.dtype_tag;
                    let blocks = compress_numeric_column(
                        &col, force, self.u64_only, profile,
                    )?;
                    all_blocks.push(blocks);
                    let fid = self.field_ids.get(&field_name).copied();
                    schema_descriptors.push(ColumnDescriptor {
                        name: field_name,
                        dtype_tag: dtype_tag.as_u8(),
                        children: Vec::new(),
                        column_id: col_idx as u16,
                        field_id: fid,
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
                    let fid = self.field_ids.get(&field_name).copied();
                    schema_descriptors.push(ColumnDescriptor {
                        name: field_name,
                        dtype_tag: dtype_tag.as_u8(),
                        children: Vec::new(),
                        column_id: col_id,
                        field_id: fid,
                    });
                }
                RouteDecision::NestedPipeline => {
                    let mut next_col_id = all_blocks.iter()
                        .flat_map(|b| b.iter())
                        .map(|(_, m)| m.column_id + 1)
                        .max()
                        .unwrap_or(col_idx as u16);
                    let mut desc = flatten_and_compress(
                        &field_name,
                        array,
                        &mut next_col_id,
                        &mut all_blocks,
                        self.u64_only,
                        profile,
                        force,
                    )?;
                    // Phase E: stamp field_id on the top-level nested
                    // descriptor; children remain None because they
                    // carry physical-leaf identity, not logical ids.
                    desc.field_id = self.field_ids.get(&field_name).copied();
                    schema_descriptors.push(desc);
                }
            }
        }

        // Emit cross-column string groups as ONE shared payload + N
        // BlockMeta entries (one per column, all pointing to the same offset).
        // Because compression happens once per group, the underlying FSST
        // table / zstd dictionary is shared across every member column.
        for group in &groups {
            let mut col_arrays: Vec<(u16, &dyn Array)> = Vec::with_capacity(group.len());
            for &col_idx in group {
                col_arrays.push((col_idx as u16, batch.column(col_idx).as_ref()));
            }
            let block_bytes = crate::compressors::string_compressor
                ::compress_cross_column_group_with_profile(&col_arrays, profile)?;
            let crc = crc32fast::hash(&block_bytes);

            // Emit the payload once with the FIRST column's BlockMeta and
            // a series of "shadow" entries for the remaining columns. We
            // achieve this by duplicating the payload-bytes only on the
            // first entry; the rest are zero-byte sentinels whose meta
            // points at the same offset (filled in during the offset
            // patching pass below).
            let first = group[0];
            let first_dtype = FluxDType::from_arrow(batch.column(first).data_type())
                .unwrap_or(FluxDType::Utf8);
            let mut group_blocks: Vec<(Vec<u8>, BlockMeta)> = Vec::with_capacity(group.len());
            // Real payload on the first member.
            group_blocks.push((block_bytes, BlockMeta {
                block_offset: 0,
                z_min: 0,
                z_max: u128::MAX,
                null_bitmap_offset: 0,
                strategy: LoomStrategy::SimdLz4,
                value_count: batch.column(first).len() as u32,
                column_id: first as u16,
                crc32: crc,
                u64_only: false,
                dtype_tag: first_dtype,
            }));
            // Shadow members: empty bytes; their offset will be patched to
            // the same offset as the first member during the assembly pass.
            for &col_idx in &group[1..] {
                let dtype = FluxDType::from_arrow(batch.column(col_idx).data_type())
                    .unwrap_or(FluxDType::Utf8);
                group_blocks.push((Vec::new(), BlockMeta {
                    block_offset: 0,
                    z_min: 0,
                    z_max: u128::MAX,
                    null_bitmap_offset: 0,
                    strategy: LoomStrategy::SimdLz4,
                    value_count: batch.column(col_idx).len() as u32,
                    column_id: col_idx as u16,
                    crc32: crc,
                    u64_only: false,
                    dtype_tag: dtype,
                }));
            }
            all_blocks.push(group_blocks);
            // Schema descriptors for grouped columns (schema order is
            // preserved by the per-column loop above for non-grouped, here
            // for grouped).
            for &col_idx in group {
                let field_name = batch_schema.field(col_idx).name().clone();
                let dtype = FluxDType::from_arrow(batch.column(col_idx).data_type())
                    .unwrap_or(FluxDType::Utf8);
                let fid = self.field_ids.get(&field_name).copied();
                schema_descriptors.push(ColumnDescriptor {
                    name: field_name,
                    dtype_tag: dtype.as_u8(),
                    children: Vec::new(),
                    column_id: col_idx as u16,
                    field_id: fid,
                });
            }
        }

        // Concatenate blocks, patch offsets. Shadow group entries (empty
        // bytes after the first member) inherit the offset of their group's
        // primary entry.
        let total_bytes: usize = all_blocks.iter()
            .flat_map(|col| col.iter())
            .map(|(bytes, _)| bytes.len())
            .sum();
        let mut output: Vec<u8> = Vec::with_capacity(total_bytes + 1024);
        let mut footer = AtlasFooter::new();

        for col_blocks in all_blocks {
            // Track the first-member's emitted offset so shadow entries can
            // share it.
            let mut first_offset: Option<u64> = None;
            for (block_bytes, mut meta) in col_blocks {
                if block_bytes.is_empty() {
                    // Shadow entry: reuse the most recent non-empty offset
                    // emitted in this column's batch (i.e., the group's
                    // primary payload).
                    meta.block_offset = first_offset
                        .expect("shadow group entry without primary payload");
                } else {
                    meta.block_offset = output.len() as u64;
                    if first_offset.is_none() {
                        first_offset = Some(meta.block_offset);
                    }
                    output.extend_from_slice(&block_bytes);
                }
                footer.push(meta);
            }
        }

        footer.schema = schema_descriptors;
        output.extend(footer.to_bytes()?);
        Ok(output)
    }
}

/// Compress a `Decimal128` (i128) column at full 128-bit width. Each segment
/// goes through the standard u128 Loom pipeline (BitSlab + OutlierMap or
/// DeltaDelta) so values that don't fit in 64 bits are preserved exactly.
fn compress_decimal128_column(
    array: &dyn Array,
    col_id: u16,
    force_strategy: Option<LoomStrategy>,
    profile: crate::CompressionProfile,
) -> FluxResult<Vec<(Vec<u8>, BlockMeta)>> {
    use rayon::prelude::*;
    use arrow_array::Decimal128Array;

    let arr = array.as_any().downcast_ref::<Decimal128Array>()
        .ok_or_else(|| FluxError::Internal("Decimal128 downcast failed".into()))?;
    let row_count = arr.len();
    if row_count == 0 {
        return Ok(Vec::new());
    }

    // Extract dense i128 values (treat null as 0; null bitmap handling is
    // out of scope for this minimal Decimal128 path).
    let values_u128: Vec<u128> = (0..row_count).map(|i| arr.value(i) as u128).collect();

    // Segment by chunking on PROBE_SIZE-aligned ranges, classify per segment.
    use crate::PROBE_SIZE;
    use crate::MAX_SEGMENT_SIZE;
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let seg = MAX_SEGMENT_SIZE.max(PROBE_SIZE);
    let mut pos = 0usize;
    while pos < row_count {
        let end = (pos + seg).min(row_count);
        ranges.push(pos..end);
        pos = end;
    }

    ranges
        .into_par_iter()
        .map(|range| {
            let chunk = &values_u128[range.clone()];
            let strategy = force_strategy.unwrap_or_else(|| classify(chunk).strategy);
            let block_bytes = compress_chunk_with_profile(chunk, strategy, profile)?;
            let crc = crc32fast::hash(&block_bytes);
            let z_min = chunk.iter().copied().min().unwrap_or(0);
            let z_max = chunk.iter().copied().max().unwrap_or(0);
            let meta = BlockMeta {
                block_offset: 0,
                z_min,
                z_max,
                null_bitmap_offset: 0,
                strategy,
                value_count: chunk.len() as u32,
                column_id: col_id,
                crc32: crc,
                u64_only: false,
                dtype_tag: FluxDType::Decimal128,
            };
            Ok((block_bytes, meta))
        })
        .collect()
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

            // Float fast path: try ALP first. ALP recognises decimal-shaped
            // floats (prices, lat/lon, integer-as-float) and emits a much
            // smaller block by encoding integer mantissas through the standard
            // pipeline. Falls back to the bit-cast path on truly arbitrary
            // doubles or when the outlier rate is too high.
            let alp_block = match dtype_tag {
                FluxDType::Float64 => {
                    let floats: Vec<f64> = seg_u64.iter()
                        .map(|&u| f64::from_bits(u))
                        .collect();
                    crate::compressors::alp_compressor::try_compress_f64(&floats)?
                }
                FluxDType::Float32 => {
                    let floats: Vec<f64> = seg_u64.iter()
                        .map(|&u| f32::from_bits(u as u32) as f64)
                        .collect();
                    crate::compressors::alp_compressor::try_compress_f32(&floats)?
                }
                _ => None,
            };

            let (block_bytes, used_strategy) = if let Some(b) = alp_block {
                // ALP block. Tag it with SimdLz4 in the strategy_mask as a
                // sentinel — the dispatcher routes by the block's own
                // leading TAG byte (0x09) anyway, but we still need to
                // populate something for the BlockMeta.
                (b, LoomStrategy::SimdLz4)
            } else if u64_only {
                (compress_chunk_u64(seg_u64, strategy, profile)?, strategy)
            } else {
                let chunk_u128: Vec<u128> = seg_u64.iter()
                    .map(|&v| v as u128).collect();
                (compress_chunk_with_profile(&chunk_u128, strategy, profile)?, strategy)
            };

            let crc = crc32fast::hash(&block_bytes);
            let z_min = seg_u64.iter().copied().min().unwrap_or(0) as u128;
            let z_max = seg_u64.iter().copied().max().unwrap_or(0) as u128;

            let meta = BlockMeta {
                block_offset: 0,
                z_min,
                z_max,
                null_bitmap_offset: 0,
                strategy: used_strategy,
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
                field_id: None,
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
                field_id: None,
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
                field_id: None,
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
        field_id: None,
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
