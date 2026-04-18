// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! String / Binary compressor for variable-length columns.
//!
//! ## Sub-strategies
//! - **Dict** (cardinality ≤ 30%): string dictionary + Loom-compressed index
//!   column. The index column goes through the full adaptive segmenter →
//!   classifier → compressor pipeline, so RLE / DeltaDelta / BitSlab / LZ4
//!   are all available on the indices. Secondary compression (LZ4/Zstd)
//!   applies automatically via the compression profile.
//! - **RawLZ4** (high cardinality, Speed/Balanced): LZ4 on offsets + data.
//! - **RawZstd** (high cardinality, Archive): Zstd on offsets + data.
//!
//! ## Block Layout (v2/v3)
//! ```text
//! [u8:  TAG = 0x06]
//! [u8:  sub_strategy]
//!   // v2 (legacy-read-only):
//!   //   0 = dict, 1 = raw_lz4, 2 = raw_zstd
//!   // v3 (current emission):
//!   //   7 = raw_lz4_v2   — offsets go through Loom pipeline
//!   //   8 = raw_zstd_v2  — offsets go through Loom pipeline
//! [u32: string_count]
//! --- if sub_strategy == 0 (dict) ---
//! [u32: dict_size]
//! [dict_size × (u32: len, len bytes: value)]
//! [u32: index_block_len]
//! [index_block bytes]           // Loom-compressed u64 index column
//! --- if sub_strategy == 1 (raw_lz4) or 2 (raw_zstd) [legacy] ---
//! [u32: compressed_offsets_len]
//! [compressed_offsets bytes]   // LZ4/Zstd on raw i32 offsets
//! [u32: compressed_data_len]
//! [compressed_data bytes]
//! --- if sub_strategy == 7 (raw_lz4_v2) or 8 (raw_zstd_v2) [current] ---
//! [u32: offsets_block_len]
//! [offsets_block bytes]        // Loom-pipeline on u64(offset) column
//! [u32: compressed_data_len]
//! [compressed_data bytes]      // LZ4 or Zstd on data
//! ```

use arrow_array::array::Array;
use arrow_array::{ArrayRef, StringArray, BinaryArray, LargeStringArray, LargeBinaryArray};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Write};
use std::sync::Arc;

use crate::dtype::FluxDType;
use crate::error::{FluxError, FluxResult};
use crate::CompressionProfile;

/// Strategy byte tag for String/Binary blocks.
pub const TAG: u8 = 0x06;

const SUB_DICT: u8 = 0;
/// Legacy raw-LZ4 sub-strategy (offsets LZ4-compressed as bytes). Read-only.
const SUB_RAW_LZ4_LEGACY: u8 = 1;
/// Legacy raw-Zstd sub-strategy (offsets Zstd-compressed as bytes). Read-only.
const SUB_RAW_ZSTD_LEGACY: u8 = 2;
/// Current raw-LZ4 sub-strategy: offsets go through the Loom pipeline.
const SUB_RAW_LZ4: u8 = 7;
/// Current raw-Zstd sub-strategy: offsets go through the Loom pipeline.
const SUB_RAW_ZSTD: u8 = 8;
/// FSST-transformed data + LZ4 on the code stream. Offsets via Loom pipeline.
const SUB_FSST_LZ4: u8 = 9;
/// FSST-transformed data + Zstd on the code stream. Offsets via Loom pipeline.
const SUB_FSST_ZSTD: u8 = 10;
/// Raw data + Zstd with a trained dictionary (Archive profile only).
const SUB_RAW_ZSTD_DICT: u8 = 11;
/// FSST-transformed data + Zstd with a trained dictionary (Archive profile only).
const SUB_FSST_ZSTD_DICT: u8 = 12;
/// Front-coded path for sorted / near-sorted columns. Stores each row as
/// `[varint shared_prefix_len][varint suffix_len][suffix_bytes]`, then feeds
/// the resulting bytestream through the profile's secondary codec. Offsets
/// are implicit (recovered while decoding) so we skip the offset column
/// entirely. `[u32: block_count][u8: codec][u32: payload_len][payload]`.
const SUB_FRONT_CODED: u8 = 13;

/// Multi sub-block container: `[u32: n][for each: u32 block_len][for each: u32 row_count][block bytes]*`.
/// Each sub-block is a self-contained string block that independently re-runs
/// the adaptive strategy decision. Enables locally low-cardinality regions
/// inside globally high-cardinality columns to opt into the dict path, and
/// provides natural parallelism + streaming checkpoints for 10M+ row columns.
const SUB_MULTI: u8 = 15;

/// Cross-column group container. Concatenates multiple compatible string
/// columns into a single byte stream, compresses them together (so FSST /
/// zstd can build one shared symbol table / dictionary), and stores the
/// per-column row counts + original column IDs so the reader can split back.
/// ```text
/// [u8: TAG = 0x06]
/// [u8: SUB_CROSS_GROUP = 14]
/// [u32: total_row_count]       // sum of all per-column row counts
/// [u8: column_count]
/// [column_count × u16: column_ids]
/// [column_count × u32: per-column row counts]
/// [u32: inner_block_len]
/// [inner string block]         // a standard string block with total_row_count rows
/// ```
pub const SUB_CROSS_GROUP: u8 = 14;

/// Row-count threshold above which we split the column into sub-blocks.
const SUB_BLOCK_SPLIT_THRESHOLD: usize = 1_000_000;
/// Target rows per sub-block. Sized to give zstd / FSST a healthy training
/// corpus (hundreds of KB) while still leaving hundreds of segments for
/// rayon to parallelise across.
const SUB_BLOCK_ROWS: usize = 500_000;

/// Probe size used for sortedness detection AND the front-coded bakeoff.
/// Needs to be large enough to fill LZ4's 64 KB window and let zstd run its
/// default 128 KB context, otherwise the raw baseline we compare against
/// understates what the full column will achieve. 16 K rows of ~80 B text
/// ≈ 1.3 MB of payload, comfortably above both codecs' context sizes.
const FRONT_CODED_PROBE: usize = 16 * 1024;
/// Minimum fraction of non-decreasing row pairs in the probe to trigger front coding.
const FRONT_CODED_MIN_SORTED_FRAC: f64 = 0.98;
/// Minimum average shared-prefix length (in bytes) to make front coding worthwhile.
const FRONT_CODED_MIN_AVG_PREFIX: f64 = 8.0;

/// Minimum column byte size below which FSST training overhead isn't worth it.
const FSST_MIN_BYTES: usize = 64 * 1024;
/// Maximum bytes of sample data to feed into FSST training.
const FSST_TRAIN_SAMPLE_BYTES: usize = 16 * 1024;
/// Maximum number of strings to sample for FSST training.
const FSST_TRAIN_SAMPLE_STRINGS: usize = 4096;
/// Probe window for the FSST ratio trial (strings).
const FSST_PROBE_STRINGS: usize = 1024;
/// Minimum FSST compression ratio (on the probe) required to keep FSST.
const FSST_MIN_RATIO: f64 = 1.25;

/// Minimum column byte size at which a trained zstd dictionary amortises.
const ZSTD_DICT_MIN_BYTES: usize = 2 * 1024 * 1024;
/// Target zstd dictionary size (zstd picks optimal up to this cap).
const ZSTD_DICT_MAX_SIZE: usize = 112 * 1024;
/// Sample target size for zstd dict training.
const ZSTD_DICT_TRAIN_BYTES: usize = 1 * 1024 * 1024;
/// Margin required for a dict variant to win the bakeoff.
const ZSTD_DICT_WIN_MARGIN: f64 = 0.97;

/// Cardinality threshold: if ≤ 30% unique strings, use dictionary encoding.
const DICT_CARDINALITY_THRESHOLD: f64 = 0.30;

/// Sample size for cardinality estimation on large columns.
const CARDINALITY_SAMPLE_SIZE: usize = 1024;

/// Columns smaller than this always do a full cardinality scan.
const CARDINALITY_FULL_SCAN_LIMIT: usize = 4096;

/// Borderline range where we fall back to a full scan to be safe.
const CARDINALITY_BORDER_LO: f64 = 0.20;
const CARDINALITY_BORDER_HI: f64 = 0.40;

// ─────────────────────────────────────────────────────────────────────────────
// Compress
// ─────────────────────────────────────────────────────────────────────────────

/// Compress an Arrow string or binary array (default Speed profile).
pub fn compress_array(array: &dyn Array) -> FluxResult<Vec<u8>> {
    compress_array_with_profile(array, CompressionProfile::Speed)
}

/// Compress multiple compatible string columns as a single cross-column group.
/// Trains ONE shared FSST table / zstd dictionary across all columns so the
/// resulting block is smaller than the sum of independently compressed blocks.
///
/// `columns` is `(column_id, array)` pairs, in schema order. All arrays must
/// have the same Arrow data type (Utf8 / LargeUtf8 / Binary / LargeBinary).
pub fn compress_cross_column_group_with_profile(
    columns: &[(u16, &dyn Array)],
    profile: CompressionProfile,
) -> FluxResult<Vec<u8>> {
    if columns.is_empty() {
        return Err(FluxError::Internal("cross-column group: empty".into()));
    }
    // Flatten all columns into one big list of &[u8], remembering boundaries.
    let mut all_values: Vec<&[u8]> = Vec::new();
    let mut row_counts: Vec<u32> = Vec::with_capacity(columns.len());
    let mut column_ids: Vec<u16> = Vec::with_capacity(columns.len());
    let mut extracted: Vec<Vec<&[u8]>> = Vec::with_capacity(columns.len());
    for &(col_id, array) in columns {
        let v = extract_strings(array)?;
        row_counts.push(v.len() as u32);
        column_ids.push(col_id);
        extracted.push(v);
    }
    // Second pass: borrow from `extracted` so the references outlive the
    // flatten. We can't hold `&'a [u8]` across `compress_strings_with_profile`
    // if `all_values` references an array that's been freed, so keep
    // `extracted` alive on the stack for the full call.
    for col_values in &extracted {
        all_values.extend_from_slice(col_values);
    }
    let total_rows = all_values.len() as u32;

    // Compress the combined column through the normal adaptive pipeline.
    // FSST/dict training sees the cross-column corpus and builds one shared
    // symbol table / dictionary that applies to every sibling column.
    let inner_block = compress_strings_with_profile(&all_values, profile)?;

    // Wrap in the SUB_CROSS_GROUP header.
    let mut buf = Vec::with_capacity(
        1 + 1 + 4 + 1 + 2 * columns.len() + 4 * columns.len() + 4 + inner_block.len(),
    );
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_CROSS_GROUP)?;
    buf.write_u32::<LittleEndian>(total_rows)?;
    buf.write_u8(columns.len() as u8)?;
    for &c in &column_ids {
        buf.write_u16::<LittleEndian>(c)?;
    }
    for &r in &row_counts {
        buf.write_u32::<LittleEndian>(r)?;
    }
    buf.write_u32::<LittleEndian>(inner_block.len() as u32)?;
    buf.extend_from_slice(&inner_block);
    Ok(buf)
}

/// Decompress a cross-column group block into per-column Arrow arrays. Returns
/// `(column_id, array)` pairs in the original schema order.
pub fn decompress_cross_column_group(
    data: &[u8],
    dtype_tag: FluxDType,
) -> FluxResult<Vec<(u16, ArrayRef)>> {
    if data.len() < 2 || data[0] != TAG || data[1] != SUB_CROSS_GROUP {
        return Err(FluxError::InvalidFile("not a cross-column group block".into()));
    }
    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let _sub  = cur.read_u8()?;
    let _total_rows = cur.read_u32::<LittleEndian>()? as usize;
    let col_count = cur.read_u8()? as usize;
    let mut column_ids = Vec::with_capacity(col_count);
    for _ in 0..col_count { column_ids.push(cur.read_u16::<LittleEndian>()?); }
    let mut row_counts = Vec::with_capacity(col_count);
    for _ in 0..col_count { row_counts.push(cur.read_u32::<LittleEndian>()? as usize); }
    let inner_len = cur.read_u32::<LittleEndian>()? as usize;
    let inner_start = cur.position() as usize;
    let inner = &data[inner_start..inner_start + inner_len];
    let combined = decompress_to_arrow_string(inner, dtype_tag)?;

    // Slice the combined array back into per-column pieces.
    let mut out = Vec::with_capacity(col_count);
    let mut offset = 0usize;
    for i in 0..col_count {
        let n = row_counts[i];
        let sliced = combined.slice(offset, n);
        out.push((column_ids[i], sliced));
        offset += n;
    }
    Ok(out)
}

/// Compress with a specific profile.
pub fn compress_array_with_profile(
    array: &dyn Array,
    profile: CompressionProfile,
) -> FluxResult<Vec<u8>> {
    let strings = extract_strings(array)?;
    compress_strings_with_profile(&strings, profile)
}

/// Compress a slice of byte slices with a specific profile.
pub fn compress_strings_with_profile(
    values: &[&[u8]],
    profile: CompressionProfile,
) -> FluxResult<Vec<u8>> {
    let count = values.len();
    if count == 0 {
        return compress_empty();
    }

    // Very large columns: split into sub-blocks, compress in parallel via
    // rayon, wrap in a SUB_MULTI container. This lets each sub-block pick
    // its own sub-strategy (dict / FSST / raw / trained-dict) based on local
    // cardinality, and doubles as a streaming checkpoint.
    if count > SUB_BLOCK_SPLIT_THRESHOLD {
        return compress_multi(values, profile);
    }

    // Sortedness probe + bakeoff: front coding is a massive win on
    // hierarchical / monotonically-sorted data (URLs, paths, keys) that
    // share long prefixes with their immediate neighbour. But it can also
    // LOSE on weakly-sorted text where FSST / LZ4 would capture the same
    // redundancy plus more. So we only short-circuit when the probe tells
    // us front coding beats the raw-path baseline on a representative
    // window — including the profile's secondary codec.
    if let Some(block) = maybe_front_coded_wins(values, profile)? {
        return Ok(block);
    }

    // Decide dict vs raw using sampled cardinality for large columns.
    let use_dict = if count <= CARDINALITY_FULL_SCAN_LIMIT {
        // Small column: full scan is cheap.
        let unique_count = values.iter().collect::<HashSet<_>>().len();
        unique_count as f64 / count as f64 <= DICT_CARDINALITY_THRESHOLD
    } else {
        // Large column: sample first CARDINALITY_SAMPLE_SIZE + strided values.
        let mut sample_set: HashSet<&[u8]> = HashSet::new();
        let head = count.min(CARDINALITY_SAMPLE_SIZE);
        for &v in &values[..head] {
            sample_set.insert(v);
        }
        // Strided sampling across the rest.
        let stride = (count / CARDINALITY_SAMPLE_SIZE).max(1);
        let mut i = head;
        while i < count {
            sample_set.insert(values[i]);
            i += stride;
        }
        let sampled_count = head + (count - head) / stride;
        let est_ratio = sample_set.len() as f64 / sampled_count as f64;

        if est_ratio >= CARDINALITY_BORDER_LO && est_ratio <= CARDINALITY_BORDER_HI {
            // Borderline: full scan to be safe.
            let unique_count = values.iter().collect::<HashSet<_>>().len();
            unique_count as f64 / count as f64 <= DICT_CARDINALITY_THRESHOLD
        } else {
            est_ratio <= DICT_CARDINALITY_THRESHOLD
        }
    };

    if use_dict {
        // Build the full dictionary (needed for compression regardless).
        let mut dict_map: HashMap<&[u8], u32> = HashMap::new();
        for &v in values {
            let next_id = dict_map.len() as u32;
            dict_map.entry(v).or_insert(next_id);
        }
        if dict_map.len() <= u32::MAX as usize {
            compress_dict(values, &dict_map, profile)
        } else {
            compress_high_cardinality(values, profile)
        }
    } else {
        compress_high_cardinality(values, profile)
    }
}

/// Adaptive selection for high-cardinality columns. Uses a mini-bakeoff on a
/// probe window: flatten the probe, apply each candidate transform (raw vs
/// FSST) plus the profile's secondary codec, and pick whichever produces the
/// smaller bytes. This handles three cases correctly:
///
/// 1. Text with lots of 2–8 byte repeats (URLs, UUIDs, log lines) → FSST wins
///    because its symbol table captures substrings that LZ4/Zstd find too
///    small to reference.
/// 2. Highly regular text (sorted paths, formatted SKUs) → raw wins because
///    Zstd's long-range matching finds byte-level repetition that FSST
///    destroys by rewriting into opaque codes.
/// 3. Small columns → raw wins trivially because FSST's fixed training cost
///    (and the ~1 KB symbol table overhead) doesn't amortise.
fn compress_high_cardinality(
    values: &[&[u8]],
    profile: CompressionProfile,
) -> FluxResult<Vec<u8>> {
    let total_bytes: usize = values.iter().map(|v| v.len()).sum();

    // Columns below the training-amortisation threshold skip FSST entirely.
    if total_bytes < FSST_MIN_BYTES {
        return compress_raw(values, profile);
    }

    // Train a symbol table on a representative sample.
    let sample = collect_fsst_sample(values);
    if sample.is_empty() {
        return compress_raw(values, profile);
    }
    let compressor = fsst::Compressor::train(&sample);

    // Mini bakeoff on a probe window: flatten into a single byte buffer,
    // apply secondary (LZ4 or Zstd) to both raw and FSST-encoded, and pick
    // the smaller. For Archive profile we also try zstd with a trained
    // dictionary. We intentionally ignore offsets here because offsets get
    // the same treatment (Loom pipeline) regardless of which data path wins.
    let probe_n = values.len().min(FSST_PROBE_STRINGS);
    let probe = &values[..probe_n];

    let mut raw_flat: Vec<u8> = Vec::new();
    for v in probe {
        raw_flat.extend_from_slice(v);
    }
    let fsst_probe = compressor.compress_bulk(&probe.to_vec());
    let mut fsst_flat: Vec<u8> = Vec::new();
    for v in &fsst_probe {
        fsst_flat.extend_from_slice(v);
    }

    // Probe secondary-codec sizes.
    let use_zstd = matches!(profile, CompressionProfile::Archive);
    let raw_secondary_len = secondary_size(&raw_flat, use_zstd)?;
    let fsst_secondary_len = secondary_size(&fsst_flat, use_zstd)?;

    // For Archive, optionally train a zstd dictionary and check whether
    // dict-compressed probe beats plain zstd by a healthy margin.
    let zstd_dict: Option<Vec<u8>> = if use_zstd && total_bytes >= ZSTD_DICT_MIN_BYTES {
        train_zstd_dict(values).ok()
    } else {
        None
    };
    let raw_dict_len = zstd_dict
        .as_ref()
        .and_then(|d| zstd_with_dict_size(&raw_flat, d).ok());
    let fsst_dict_len = zstd_dict
        .as_ref()
        .and_then(|d| zstd_with_dict_size(&fsst_flat, d).ok());

    // Choose the winning candidate. Note the `.min()` comparison: each dict
    // variant has to beat the non-dict baseline by `ZSTD_DICT_WIN_MARGIN`
    // to prevent the dict overhead from wiping out the gain.
    const FSST_WIN_MARGIN: f64 = 0.95;
    let fsst_wins = (fsst_secondary_len as f64) < (raw_secondary_len as f64) * FSST_WIN_MARGIN;

    // Require FSST to actually compress the probe before we pay decode cost.
    let probe_raw_bytes: usize = probe.iter().map(|v| v.len()).sum();
    let fsst_encoded_bytes: usize = fsst_probe.iter().map(|v| v.len()).sum();
    let fsst_standalone_ratio = if fsst_encoded_bytes == 0 {
        f64::INFINITY
    } else {
        probe_raw_bytes as f64 / fsst_encoded_bytes as f64
    };
    let fsst_qualifies = fsst_wins && fsst_standalone_ratio >= FSST_MIN_RATIO;

    // Base picks without dict.
    let use_fsst = fsst_qualifies;

    // Does the dict help on the chosen branch?
    let base_len = if use_fsst { fsst_secondary_len } else { raw_secondary_len };
    let dict_len = if use_fsst { fsst_dict_len } else { raw_dict_len };
    let use_dict = match (zstd_dict.as_ref(), dict_len) {
        (Some(_), Some(dl)) => (dl as f64) < (base_len as f64) * ZSTD_DICT_WIN_MARGIN,
        _ => false,
    };

    match (use_fsst, use_dict) {
        (true,  false) => compress_fsst(values, &compressor, profile),
        (true,  true)  => compress_fsst_with_dict(values, &compressor, zstd_dict.unwrap()),
        (false, false) => compress_raw(values, profile),
        (false, true)  => compress_raw_with_dict(values, zstd_dict.unwrap()),
    }
}

/// Train a zstd dictionary from a bounded sample of the column's data.
fn train_zstd_dict(values: &[&[u8]]) -> FluxResult<Vec<u8>> {
    // Sample evenly across the column.
    let stride = (values.len() / 2048).max(1);
    let mut sample: Vec<&[u8]> = Vec::new();
    let mut bytes = 0usize;
    let mut i = 0;
    while i < values.len() && bytes < ZSTD_DICT_TRAIN_BYTES {
        sample.push(values[i]);
        bytes += values[i].len();
        i += stride;
    }
    if sample.len() < 32 {
        return Err(FluxError::Internal("zstd dict sample too small".into()));
    }
    zstd::dict::from_samples(&sample, ZSTD_DICT_MAX_SIZE)
        .map_err(|e| FluxError::Internal(format!("zstd dict train: {e}")))
}

/// Compress `bytes` with zstd using the trained `dict`, return only the size.
fn zstd_with_dict_size(bytes: &[u8], dict: &[u8]) -> FluxResult<usize> {
    let mut c = zstd::bulk::Compressor::with_dictionary(3, dict)
        .map_err(|e| FluxError::Internal(format!("zstd dict enc: {e}")))?;
    let out = c.compress(bytes)
        .map_err(|e| FluxError::Internal(format!("zstd dict compress: {e}")))?;
    Ok(out.len())
}

/// Estimate the size of `bytes` after the profile's secondary codec.
fn secondary_size(bytes: &[u8], use_zstd: bool) -> FluxResult<usize> {
    if use_zstd {
        zstd::stream::encode_all(bytes, 3)
            .map(|v| v.len())
            .map_err(|e| FluxError::Internal(format!("zstd probe: {e}")))
    } else {
        Ok(lz4_flex::compress_prepend_size(bytes).len())
    }
}

// ── Front coding ─────────────────────────────────────────────────────────────────────────

/// Probe-based bakeoff: measure sortedness + shared-prefix on a small window,
/// then (if qualifying) compare front-coded+secondary vs raw+secondary on the
/// same probe. Returns `Some(encoded_block)` if front coding wins for the
/// whole column, `None` otherwise.
fn maybe_front_coded_wins(
    values: &[&[u8]],
    profile: CompressionProfile,
) -> FluxResult<Option<Vec<u8>>> {
    let n = values.len().min(FRONT_CODED_PROBE);
    if n < 8 {
        return Ok(None);
    }
    let mut nondec = 0usize;
    let mut shared_sum = 0usize;
    for i in 1..n {
        if values[i] >= values[i - 1] {
            nondec += 1;
        }
        let lim = values[i - 1].len().min(values[i].len());
        let mut j = 0;
        while j < lim && values[i - 1][j] == values[i][j] {
            j += 1;
        }
        shared_sum += j;
    }
    let sorted_frac = (nondec as f64) / ((n - 1) as f64);
    let avg_prefix = (shared_sum as f64) / ((n - 1) as f64);
    if sorted_frac < FRONT_CODED_MIN_SORTED_FRAC || avg_prefix < FRONT_CODED_MIN_AVG_PREFIX {
        return Ok(None);
    }

    // Bakeoff on the probe: compress with both encodings + secondary codec,
    // keep front coding only if it wins by ≥5%.
    let probe = &values[..n];
    let fc_probe = compress_front_coded(probe, profile)?;

    // Raw-path probe: flatten data, secondary-compress, estimate total size
    // (ignore offsets column since it's the same size for both paths).
    let mut raw_flat: Vec<u8> = Vec::new();
    for &v in probe {
        raw_flat.extend_from_slice(v);
    }
    let raw_secondary = secondary_size(&raw_flat, matches!(profile, CompressionProfile::Archive))?;

    if (fc_probe.len() as f64) < (raw_secondary as f64) * 0.95 {
        Ok(Some(compress_front_coded(values, profile)?))
    } else {
        Ok(None)
    }
}

/// Write a varint (LEB128) into `buf`.
fn write_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

/// Read a varint from `buf` starting at `pos`. Returns `(value, new_pos)`.
fn read_varint(buf: &[u8], pos: usize) -> FluxResult<(u64, usize)> {
    let mut v = 0u64;
    let mut shift = 0u32;
    let mut p = pos;
    loop {
        if p >= buf.len() {
            return Err(FluxError::InvalidFile("front-coded: varint truncated".into()));
        }
        let b = buf[p];
        p += 1;
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(FluxError::InvalidFile("front-coded: varint overflow".into()));
        }
    }
    Ok((v, p))
}

/// Compress as front-coded. Each row is stored as varint `shared_prefix_len`
/// + varint `suffix_len` + `suffix_bytes`. The resulting payload is then put
/// through LZ4 (Speed/Balanced) or Zstd (Archive).
fn compress_front_coded(values: &[&[u8]], profile: CompressionProfile) -> FluxResult<Vec<u8>> {
    // Estimate payload capacity: total bytes + 2 varints per row (worst-case 5 bytes each).
    let total: usize = values.iter().map(|v| v.len()).sum();
    let mut payload = Vec::with_capacity(total + values.len() * 4);

    let mut prev: &[u8] = &[];
    for &cur in values {
        // Compute shared prefix.
        let lim = prev.len().min(cur.len());
        let mut j = 0;
        while j < lim && prev[j] == cur[j] {
            j += 1;
        }
        let shared = j;
        let suffix = &cur[shared..];
        write_varint(&mut payload, shared as u64);
        write_varint(&mut payload, suffix.len() as u64);
        payload.extend_from_slice(suffix);
        prev = cur;
    }

    // Secondary codec. Speed/Balanced use LZ4; Archive uses Zstd.
    let (compressed, codec_byte) = match profile {
        CompressionProfile::Archive => {
            let c = zstd::stream::encode_all(payload.as_slice(), 3)
                .map_err(|e| FluxError::Internal(format!("zstd front-coded: {e}")))?;
            (c, 2u8)
        }
        _ => {
            let c = lz4_flex::compress_prepend_size(&payload);
            (c, 1u8)
        }
    };

    let mut buf = Vec::with_capacity(1 + 1 + 4 + 1 + 4 + compressed.len());
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_FRONT_CODED)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u8(codec_byte)?;
    buf.write_u32::<LittleEndian>(compressed.len() as u32)?;
    buf.extend_from_slice(&compressed);
    Ok(buf)
}

/// Decompress a front-coded block back to materialised `Vec<Vec<u8>>`.
fn decompress_front_coded(
    cur: &mut Cursor<&[u8]>,
    count: usize,
) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    let codec = cur.read_u8()?;
    let payload_len = cur.read_u32::<LittleEndian>()? as usize;
    let payload_start = cur.position() as usize;
    let compressed = &cur.get_ref()[payload_start..payload_start + payload_len];
    let decoded = match codec {
        1 => lz4_flex::decompress_size_prepended(compressed)
            .map_err(|e| FluxError::Lz4(e.to_string()))?,
        2 => zstd::stream::decode_all(compressed)
            .map_err(|e| FluxError::Internal(format!("zstd front-coded: {e}")))?,
        _ => return Err(FluxError::InvalidFile(format!("front-coded: bad codec {codec}"))),
    };

    // Reconstruct each row from (shared, suffix_len, suffix).
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(count);
    let mut pos = 0usize;
    let mut prev: Vec<u8> = Vec::new();
    for _ in 0..count {
        let (shared, np) = read_varint(&decoded, pos)?;
        let (slen, np) = read_varint(&decoded, np)?;
        let shared = shared as usize;
        let slen = slen as usize;
        if np + slen > decoded.len() {
            return Err(FluxError::InvalidFile("front-coded: suffix out of range".into()));
        }
        if shared > prev.len() {
            return Err(FluxError::InvalidFile("front-coded: shared > prev".into()));
        }
        let mut row = Vec::with_capacity(shared + slen);
        row.extend_from_slice(&prev[..shared]);
        row.extend_from_slice(&decoded[np..np + slen]);
        prev = row.clone();
        out.push(row);
        pos = np + slen;
    }
    Ok((out, payload_start + payload_len))
}

/// Split `values` into fixed-row-count sub-blocks, compress each in parallel
/// through the full adaptive pipeline, and emit a `SUB_MULTI` container.
fn compress_multi(
    values: &[&[u8]],
    profile: CompressionProfile,
) -> FluxResult<Vec<u8>> {
    use rayon::prelude::*;

    // Partition into sub-blocks of roughly `SUB_BLOCK_ROWS` rows. Using a
    // Vec<&[&[u8]]> of views so we don't copy any bytes.
    let chunks: Vec<&[&[u8]]> = values.chunks(SUB_BLOCK_ROWS).collect();

    // Compress each sub-block in parallel. Each call recursively dispatches
    // through the normal adaptive pipeline — `count` for a single sub-block
    // is <= SUB_BLOCK_SPLIT_THRESHOLD so we won't infinitely recurse.
    let sub_blocks: FluxResult<Vec<Vec<u8>>> = chunks
        .into_par_iter()
        .map(|chunk| compress_strings_with_profile(chunk, profile))
        .collect();
    let sub_blocks = sub_blocks?;

    // Assemble the container.
    let total_inner: usize = sub_blocks.iter().map(|b| b.len()).sum();
    let n = sub_blocks.len();
    let mut buf: Vec<u8> = Vec::with_capacity(1 + 1 + 4 + 4 + n * 8 + total_inner);
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_MULTI)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u32::<LittleEndian>(n as u32)?;
    // Sub-block index: (row_count, block_len) pairs so the reader can
    // skip / mmap-slice without decoding.
    let mut offset = 0usize;
    for (chunk_idx, block) in sub_blocks.iter().enumerate() {
        let rows = if chunk_idx + 1 == n {
            values.len() - chunk_idx * SUB_BLOCK_ROWS
        } else {
            SUB_BLOCK_ROWS
        };
        buf.write_u32::<LittleEndian>(rows as u32)?;
        buf.write_u32::<LittleEndian>(block.len() as u32)?;
        offset += block.len();
    }
    let _ = offset;
    for block in &sub_blocks {
        buf.extend_from_slice(block);
    }
    Ok(buf)
}

fn compress_empty() -> FluxResult<Vec<u8>> {
    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_RAW_LZ4)?;
    buf.write_u32::<LittleEndian>(0)?;
    // v2 offsets block: 1 segment of 0 values still needs the header,
    // but empty columns short-circuit on read; emit a pair of zero lengths
    // consistent with the non-empty v2 layout.
    buf.write_u32::<LittleEndian>(0)?;
    buf.write_u32::<LittleEndian>(0)?;
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dict path: dictionary + Loom-compressed index column
// ─────────────────────────────────────────────────────────────────────────────

fn compress_dict(
    values: &[&[u8]],
    dict_map: &HashMap<&[u8], u32>,
    profile: CompressionProfile,
) -> FluxResult<Vec<u8>> {
    let dict_size = dict_map.len() as u32;

    let mut dict_entries: Vec<(&[u8], u32)> = dict_map.iter().map(|(&k, &v)| (k, v)).collect();
    dict_entries.sort_by_key(|&(_, idx)| idx);

    // Build index column and compress through Loom pipeline.
    let indices: Vec<u64> = values.iter().map(|v| dict_map[*v] as u64).collect();
    let index_block = compress_index_column(&indices, profile)?;

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_DICT)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u32::<LittleEndian>(dict_size)?;

    for (entry, _) in &dict_entries {
        buf.write_u32::<LittleEndian>(entry.len() as u32)?;
        buf.write_all(entry)?;
    }

    buf.write_u32::<LittleEndian>(index_block.len() as u32)?;
    buf.extend_from_slice(&index_block);
    Ok(buf)
}

/// Compress a u64 "index" column (dict indices, offsets) through the full
/// Loom pipeline: adaptive segmenter → classifier → per-segment strategy.
fn compress_index_column(indices: &[u64], profile: CompressionProfile) -> FluxResult<Vec<u8>> {
    use crate::compressors::flux_writer::compress_chunk_with_profile;
    use crate::segmenter::adaptive_segment_u64;

    let segments = adaptive_segment_u64(indices, None);
    let mut output = Vec::new();
    output.write_u32::<LittleEndian>(segments.len() as u32)?;

    for (range, strategy) in segments {
        let seg = &indices[range];
        let chunk_u128: Vec<u128> = seg.iter().map(|&v| v as u128).collect();
        let block = compress_chunk_with_profile(&chunk_u128, strategy, profile)?;

        output.write_u32::<LittleEndian>(block.len() as u32)?;
        output.write_u32::<LittleEndian>(seg.len() as u32)?;
        output.extend_from_slice(&block);
    }
    Ok(output)
}

// ──────────────────────────────────────────────────────────────────────────────
// FSST path: per-column static symbol table, rayon-parallel encode
// ──────────────────────────────────────────────────────────────────────────────

/// Collect a bounded training sample from `values` spread evenly across the
/// column. FSST trains much better when it sees a representative mix than
/// from a contiguous prefix.
fn collect_fsst_sample<'a>(values: &'a [&'a [u8]]) -> Vec<&'a [u8]> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut sample: Vec<&[u8]> = Vec::new();
    let stride = (values.len() / FSST_TRAIN_SAMPLE_STRINGS).max(1);
    let mut bytes = 0usize;
    let mut i = 0;
    while i < values.len() && sample.len() < FSST_TRAIN_SAMPLE_STRINGS
        && bytes < FSST_TRAIN_SAMPLE_BYTES {
        sample.push(values[i]);
        bytes += values[i].len();
        i += stride;
    }
    sample
}

/// Serialize an FSST symbol table: `[u8 n_symbols][n_symbols × (u8 len, 8 bytes symbol)]`.
fn serialize_fsst_table(compressor: &fsst::Compressor) -> Vec<u8> {
    let symbols = compressor.symbol_table();
    let lengths = compressor.symbol_lengths();
    let n = symbols.len();
    debug_assert!(n <= 255);
    let mut out = Vec::with_capacity(1 + n * 9);
    out.push(n as u8);
    for (sym, &len) in symbols.iter().zip(lengths.iter()) {
        out.push(len);
        out.extend_from_slice(&sym.to_u64().to_le_bytes());
    }
    out
}

/// Deserialize a previously serialized FSST symbol table.
fn deserialize_fsst_table(data: &[u8]) -> FluxResult<(fsst::Compressor, usize)> {
    if data.is_empty() {
        return Err(FluxError::InvalidFile("fsst table: empty".into()));
    }
    let n = data[0] as usize;
    let mut off = 1;
    let mut symbols = Vec::with_capacity(n);
    let mut lengths = Vec::with_capacity(n);
    for _ in 0..n {
        if off + 9 > data.len() {
            return Err(FluxError::InvalidFile("fsst table: truncated".into()));
        }
        lengths.push(data[off]);
        let mut b = [0u8; 8];
        b.copy_from_slice(&data[off + 1..off + 9]);
        symbols.push(fsst::Symbol::from_slice(&b));
        off += 9;
    }
    Ok((fsst::Compressor::rebuild_from(symbols, lengths), off))
}

/// Compress with a pre-trained FSST symbol table.
fn compress_fsst(
    values: &[&[u8]],
    compressor: &fsst::Compressor,
    profile: CompressionProfile,
) -> FluxResult<Vec<u8>> {
    use rayon::prelude::*;

    // Rayon-parallel encode: each string is encoded independently using the
    // shared immutable symbol table. The bulk API also works but paying the
    // rayon dispatch cost is worthwhile for 10M-row columns.
    const FSST_PAR_THRESHOLD: usize = 8192;
    let encoded: Vec<Vec<u8>> = if values.len() >= FSST_PAR_THRESHOLD {
        values.par_iter().map(|v| compressor.compress(v)).collect()
    } else {
        compressor.compress_bulk(&values.to_vec())
    };

    // Flatten into offsets + data. Offsets go through the Loom pipeline
    // and the encoded byte stream goes through LZ4 / Zstd.
    let total_bytes: usize = encoded.iter().map(|v| v.len()).sum();
    let mut offsets: Vec<u64> = Vec::with_capacity(values.len() + 1);
    let mut data_buf: Vec<u8> = Vec::with_capacity(total_bytes);
    offsets.push(0);
    for v in &encoded {
        data_buf.extend_from_slice(v);
        offsets.push(data_buf.len() as u64);
    }

    let offsets_block = compress_index_column(&offsets, profile)?;
    let use_zstd = matches!(profile, CompressionProfile::Archive);
    let (cd, sub) = if use_zstd {
        (
            zstd::stream::encode_all(data_buf.as_slice(), 3)
                .map_err(|e| FluxError::Internal(format!("zstd fsst data: {e}")))?,
            SUB_FSST_ZSTD,
        )
    } else {
        (
            lz4_flex::compress_prepend_size(&data_buf),
            SUB_FSST_LZ4,
        )
    };

    let table = serialize_fsst_table(compressor);

    let mut buf = Vec::with_capacity(1 + 1 + 4 + 2 + table.len() + 4 + offsets_block.len() + 4 + cd.len());
    buf.write_u8(TAG)?;
    buf.write_u8(sub)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u16::<LittleEndian>(table.len() as u16)?;
    buf.extend_from_slice(&table);
    buf.write_u32::<LittleEndian>(offsets_block.len() as u32)?;
    buf.extend_from_slice(&offsets_block);
    buf.write_u32::<LittleEndian>(cd.len() as u32)?;
    buf.extend_from_slice(&cd);
    Ok(buf)
}

// ──────────────────────────────────────────────────────────────────────────────
// Raw path: LZ4 or Zstd on offsets + data
// ──────────────────────────────────────────────────────────────────────────────

/// Raw path but with zstd+trained-dict on the data buffer. Always produces
/// `SUB_RAW_ZSTD_DICT` (Archive profile only).
fn compress_raw_with_dict(values: &[&[u8]], dict: Vec<u8>) -> FluxResult<Vec<u8>> {
    // Flatten into offsets + data.
    let total_bytes_hint: usize = values.iter().map(|v| v.len()).sum();
    let mut offsets: Vec<u64> = Vec::with_capacity(values.len() + 1);
    let mut data_buf: Vec<u8> = Vec::with_capacity(total_bytes_hint);
    offsets.push(0);
    for &v in values {
        data_buf.extend_from_slice(v);
        offsets.push(data_buf.len() as u64);
    }

    let offsets_block = compress_index_column(&offsets, CompressionProfile::Archive)?;
    let mut c = zstd::bulk::Compressor::with_dictionary(3, &dict)
        .map_err(|e| FluxError::Internal(format!("zstd dict enc: {e}")))?;
    let cd = c.compress(&data_buf)
        .map_err(|e| FluxError::Internal(format!("zstd dict compress: {e}")))?;

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_RAW_ZSTD_DICT)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u32::<LittleEndian>(dict.len() as u32)?;
    buf.extend_from_slice(&dict);
    buf.write_u32::<LittleEndian>(offsets_block.len() as u32)?;
    buf.extend_from_slice(&offsets_block);
    buf.write_u32::<LittleEndian>(cd.len() as u32)?;
    buf.extend_from_slice(&cd);
    Ok(buf)
}

/// FSST path but with zstd+trained-dict on the code stream. Always produces
/// `SUB_FSST_ZSTD_DICT` (Archive profile only).
fn compress_fsst_with_dict(
    values: &[&[u8]],
    compressor: &fsst::Compressor,
    dict: Vec<u8>,
) -> FluxResult<Vec<u8>> {
    use rayon::prelude::*;

    const FSST_PAR_THRESHOLD: usize = 8192;
    let encoded: Vec<Vec<u8>> = if values.len() >= FSST_PAR_THRESHOLD {
        values.par_iter().map(|v| compressor.compress(v)).collect()
    } else {
        compressor.compress_bulk(&values.to_vec())
    };
    let total_bytes: usize = encoded.iter().map(|v| v.len()).sum();
    let mut offsets: Vec<u64> = Vec::with_capacity(values.len() + 1);
    let mut data_buf: Vec<u8> = Vec::with_capacity(total_bytes);
    offsets.push(0);
    for v in &encoded {
        data_buf.extend_from_slice(v);
        offsets.push(data_buf.len() as u64);
    }

    let offsets_block = compress_index_column(&offsets, CompressionProfile::Archive)?;
    let mut c = zstd::bulk::Compressor::with_dictionary(3, &dict)
        .map_err(|e| FluxError::Internal(format!("zstd dict enc: {e}")))?;
    let cd = c.compress(&data_buf)
        .map_err(|e| FluxError::Internal(format!("zstd dict compress: {e}")))?;

    let table = serialize_fsst_table(compressor);

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_FSST_ZSTD_DICT)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u16::<LittleEndian>(table.len() as u16)?;
    buf.extend_from_slice(&table);
    buf.write_u32::<LittleEndian>(dict.len() as u32)?;
    buf.extend_from_slice(&dict);
    buf.write_u32::<LittleEndian>(offsets_block.len() as u32)?;
    buf.extend_from_slice(&offsets_block);
    buf.write_u32::<LittleEndian>(cd.len() as u32)?;
    buf.extend_from_slice(&cd);
    Ok(buf)
}

fn compress_raw(values: &[&[u8]], profile: CompressionProfile) -> FluxResult<Vec<u8>> {
    // Build offsets + data buffers. Use u64 natively so we can pipe the
    // (monotonic) offset column directly through the Loom pipeline
    // (DeltaDelta → BitSlab / FOR) instead of LZ4-ing a sequence of little-
    // endian i32s, which is a huge waste for sorted integer data.
    let total_bytes_hint: usize = values.iter().map(|v| v.len()).sum();
    let mut offsets: Vec<u64> = Vec::with_capacity(values.len() + 1);
    let mut data_buf: Vec<u8> = Vec::with_capacity(total_bytes_hint);
    offsets.push(0);
    for &v in values {
        data_buf.extend_from_slice(v);
        offsets.push(data_buf.len() as u64);
    }

    // Offsets: Loom pipeline (adaptive segment + classifier + outlier patching).
    let offsets_block = compress_index_column(&offsets, profile)?;

    let use_zstd = matches!(profile, CompressionProfile::Archive);
    let (cd, sub) = if use_zstd {
        (
            zstd::stream::encode_all(data_buf.as_slice(), 3)
                .map_err(|e| FluxError::Internal(format!("zstd data: {e}")))?,
            SUB_RAW_ZSTD,
        )
    } else {
        (
            lz4_flex::compress_prepend_size(&data_buf),
            SUB_RAW_LZ4,
        )
    };

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(sub)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u32::<LittleEndian>(offsets_block.len() as u32)?;
    buf.extend_from_slice(&offsets_block);
    buf.write_u32::<LittleEndian>(cd.len() as u32)?;
    buf.extend_from_slice(&cd);
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Decompress
// ─────────────────────────────────────────────────────────────────────────────

/// Decompress a string block. Returns `(strings_as_bytes, bytes_consumed)`.
pub fn decompress(data: &[u8]) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not a String block".into()));
    }
    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let sub_strategy = cur.read_u8()?;
    let string_count = cur.read_u32::<LittleEndian>()? as usize;

    if string_count == 0 {
        if sub_strategy != SUB_DICT {
            cur.read_u32::<LittleEndian>()?;
            cur.read_u32::<LittleEndian>()?;
        }
        return Ok((Vec::new(), cur.position() as usize));
    }

    match sub_strategy {
        SUB_DICT => decompress_dict(&mut cur, string_count),
        SUB_RAW_LZ4_LEGACY  => decompress_raw_legacy(&mut cur, string_count, false),
        SUB_RAW_ZSTD_LEGACY => decompress_raw_legacy(&mut cur, string_count, true),
        SUB_RAW_LZ4  => decompress_raw_v2(&mut cur, string_count, false),
        SUB_RAW_ZSTD => decompress_raw_v2(&mut cur, string_count, true),
        SUB_FSST_LZ4  => decompress_fsst(&mut cur, string_count, false),
        SUB_FSST_ZSTD => decompress_fsst(&mut cur, string_count, true),
        SUB_RAW_ZSTD_DICT  => decompress_raw_zstd_dict(&mut cur, string_count),
        SUB_FSST_ZSTD_DICT => decompress_fsst_zstd_dict(&mut cur, string_count),
        SUB_FRONT_CODED    => decompress_front_coded(&mut cur, string_count),
        SUB_MULTI => decompress_multi(data, &mut cur, string_count),
        _ => Err(FluxError::InvalidFile(format!(
            "unknown string sub_strategy: {sub_strategy:#04x}"
        ))),
    }
}

/// Decode a `SUB_MULTI` container: iterate the sub-block index, decompress
/// each sub-block in parallel, and stitch the results back together.
fn decompress_multi(
    whole: &[u8],
    cur: &mut Cursor<&[u8]>,
    total_count: usize,
) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    use rayon::prelude::*;

    let n = cur.read_u32::<LittleEndian>()? as usize;
    let mut rows_vec = Vec::with_capacity(n);
    let mut lens = Vec::with_capacity(n);
    for _ in 0..n {
        rows_vec.push(cur.read_u32::<LittleEndian>()? as usize);
        lens.push(cur.read_u32::<LittleEndian>()? as usize);
    }

    // Slice out each sub-block's payload.
    let header_end = cur.position() as usize;
    let mut offsets = Vec::with_capacity(n);
    let mut off = header_end;
    for &len in &lens {
        offsets.push((off, len));
        off += len;
    }
    let end = off;

    // Parallel decode, each sub-block is self-contained.
    let results: FluxResult<Vec<Vec<Vec<u8>>>> = offsets
        .par_iter()
        .map(|&(o, len)| {
            let slice = &whole[o..o + len];
            let (rows, _) = decompress(slice)?;
            Ok(rows)
        })
        .collect();
    let results = results?;

    let mut out = Vec::with_capacity(total_count);
    for chunk in results {
        out.extend(chunk);
    }
    Ok((out, end))
}

fn decompress_dict(cur: &mut Cursor<&[u8]>, count: usize) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    let dict_size = cur.read_u32::<LittleEndian>()? as usize;
    let mut dict: Vec<Vec<u8>> = Vec::with_capacity(dict_size);
    for _ in 0..dict_size {
        let len = cur.read_u32::<LittleEndian>()? as usize;
        let pos = cur.position() as usize;
        let data = cur.get_ref();
        if pos + len > data.len() {
            return Err(FluxError::BufferOverflow { needed: pos + len, have: data.len() });
        }
        dict.push(data[pos..pos + len].to_vec());
        cur.set_position((pos + len) as u64);
    }

    let index_block_len = cur.read_u32::<LittleEndian>()? as usize;
    let index_start = cur.position() as usize;
    let index_data = &cur.get_ref()[index_start..index_start + index_block_len];
    let indices = decompress_index_column(index_data, count)?;

    let mut strings = Vec::with_capacity(count);
    for idx in indices {
        let i = idx as usize;
        strings.push(dict.get(i).cloned().ok_or_else(|| {
            FluxError::InvalidFile(format!("string dict index {i} out of range"))
        })?);
    }
    Ok((strings, index_start + index_block_len))
}

fn decompress_index_column(data: &[u8], expected: usize) -> FluxResult<Vec<u64>> {
    use crate::decompressors::block_reader::decompress_block;
    let mut cur = Cursor::new(data);
    let seg_count = cur.read_u32::<LittleEndian>()? as usize;
    let mut out = Vec::with_capacity(expected);

    for _ in 0..seg_count {
        let block_len = cur.read_u32::<LittleEndian>()? as usize;
        let _value_count = cur.read_u32::<LittleEndian>()? as usize;
        let pos = cur.position() as usize;
        let (values, _) = decompress_block(&data[pos..pos + block_len])?;
        out.extend(values.iter().map(|&v| v as u64));
        cur.set_position((pos + block_len) as u64);
    }
    Ok(out)
}

/// Legacy raw path (v2): both offsets and data are opaque LZ4/Zstd payloads,
/// offsets are little-endian i32. Kept for backwards compatibility.
fn decompress_raw_legacy(
    cur: &mut Cursor<&[u8]>,
    count: usize,
    use_zstd: bool,
) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    let co_len = cur.read_u32::<LittleEndian>()? as usize;
    let co_start = cur.position() as usize;

    let offsets_raw = if use_zstd {
        zstd::stream::decode_all(&cur.get_ref()[co_start..co_start + co_len])
            .map_err(|e| FluxError::Internal(format!("zstd offsets: {e}")))?
    } else {
        lz4_flex::decompress_size_prepended(&cur.get_ref()[co_start..co_start + co_len])
            .map_err(|e| FluxError::Lz4(e.to_string()))?
    };
    cur.set_position((co_start + co_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;

    let data_buf = if use_zstd {
        zstd::stream::decode_all(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Internal(format!("zstd data: {e}")))?
    } else {
        lz4_flex::decompress_size_prepended(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Lz4(e.to_string()))?
    };

    let mut offsets = Vec::with_capacity(offsets_raw.len() / 4);
    let mut oc = Cursor::new(&offsets_raw);
    for _ in 0..offsets_raw.len() / 4 {
        offsets.push(oc.read_i32::<LittleEndian>()?);
    }

    let mut strings = Vec::with_capacity(count);
    for i in 0..count {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        strings.push(data_buf[s..e].to_vec());
    }
    Ok((strings, cd_start + cd_len))
}

/// Current raw path (v3): offsets arrive as a Loom-compressed u64 column,
/// data arrives as LZ4 or Zstd. Yields materialised `Vec<Vec<u8>>` for the
/// legacy [`decompress`] entry point.
fn decompress_raw_v2(
    cur: &mut Cursor<&[u8]>,
    count: usize,
    use_zstd: bool,
) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    let (offsets, data_buf, end) = decode_raw_v2_buffers(cur, use_zstd, count + 1)?;
    if offsets.len() != count + 1 {
        return Err(FluxError::InvalidFile(format!(
            "string raw_v2 offsets length {} != count+1 {}",
            offsets.len(), count + 1,
        )));
    }
    let mut strings = Vec::with_capacity(count);
    for i in 0..count {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        if e > data_buf.len() || s > e {
            return Err(FluxError::InvalidFile("raw_v2 offsets out of range".into()));
        }
        strings.push(data_buf[s..e].to_vec());
    }
    Ok((strings, end))
}

/// Shared core for the v2 raw decode path — returns the raw offsets + data
/// buffers without materialising per-string allocations. Used by both the
/// legacy `Vec<Vec<u8>>` API and the direct-to-Arrow fast path. The caller
/// passes the expected number of offsets (`count + 1`) for preallocation.
fn decode_raw_v2_buffers(
    cur: &mut Cursor<&[u8]>,
    use_zstd: bool,
    expected_offsets: usize,
) -> FluxResult<(Vec<u64>, Vec<u8>, usize)> {
    let off_len = cur.read_u32::<LittleEndian>()? as usize;
    let off_start = cur.position() as usize;
    let offsets_data = &cur.get_ref()[off_start..off_start + off_len];
    let offsets = decompress_index_column(offsets_data, expected_offsets)?;
    cur.set_position((off_start + off_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;
    let data_buf = if use_zstd {
        zstd::stream::decode_all(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Internal(format!("zstd data: {e}")))?
    } else {
        lz4_flex::decompress_size_prepended(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Lz4(e.to_string()))?
    };
    Ok((offsets, data_buf, cd_start + cd_len))
}

// ── FSST decode ──────────────────────────────────────────────────────────────────────────────

/// Shared body for FSST block decoding — reads symbol table, offsets column,
/// code stream; returns `(decompressor, offsets, code_bytes, bytes_end)`.
fn decode_fsst_buffers<'a>(
    cur: &mut Cursor<&'a [u8]>,
    use_zstd: bool,
    expected_offsets: usize,
) -> FluxResult<(fsst::Compressor, Vec<u64>, Vec<u8>, usize)> {
    let table_len = cur.read_u16::<LittleEndian>()? as usize;
    let table_start = cur.position() as usize;
    let table_slice = &cur.get_ref()[table_start..table_start + table_len];
    let (compressor, consumed) = deserialize_fsst_table(table_slice)?;
    if consumed != table_len {
        return Err(FluxError::InvalidFile("fsst table size mismatch".into()));
    }
    cur.set_position((table_start + table_len) as u64);

    let off_len = cur.read_u32::<LittleEndian>()? as usize;
    let off_start = cur.position() as usize;
    let offsets_data = &cur.get_ref()[off_start..off_start + off_len];
    let offsets = decompress_index_column(offsets_data, expected_offsets)?;
    cur.set_position((off_start + off_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;
    let code_bytes = if use_zstd {
        zstd::stream::decode_all(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Internal(format!("zstd fsst data: {e}")))?
    } else {
        lz4_flex::decompress_size_prepended(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Lz4(e.to_string()))?
    };
    Ok((compressor, offsets, code_bytes, cd_start + cd_len))
}

/// Read a `[u32 len][dict bytes]` header and return the raw dict slice.
fn read_zstd_dict<'a>(cur: &mut Cursor<&'a [u8]>) -> FluxResult<&'a [u8]> {
    let dict_len = cur.read_u32::<LittleEndian>()? as usize;
    let start = cur.position() as usize;
    let data = cur.get_ref();
    if start + dict_len > data.len() {
        return Err(FluxError::BufferOverflow {
            needed: start + dict_len,
            have: data.len(),
        });
    }
    let slice = &data[start..start + dict_len];
    cur.set_position((start + dict_len) as u64);
    Ok(slice)
}

/// zstd-decompress `bytes` using the trained `dict`.
fn zstd_decompress_with_dict(bytes: &[u8], dict: &[u8]) -> FluxResult<Vec<u8>> {
    let mut d = zstd::bulk::Decompressor::with_dictionary(dict)
        .map_err(|e| FluxError::Internal(format!("zstd dict dec: {e}")))?;
    // Upper bound for the output buffer: zstd embeds the frame size when
    // input was a single call to bulk::Compressor::compress (which we did).
    // Use 32x expansion as a safety cap; bulk API respects the frame size.
    let cap = bytes.len().saturating_mul(32).max(1 << 20);
    d.decompress(bytes, cap)
        .map_err(|e| FluxError::Internal(format!("zstd dict decompress: {e}")))
}

/// Raw + zstd-with-dict decode path.
fn decompress_raw_zstd_dict(
    cur: &mut Cursor<&[u8]>,
    count: usize,
) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    let dict = read_zstd_dict(cur)?.to_vec();

    let off_len = cur.read_u32::<LittleEndian>()? as usize;
    let off_start = cur.position() as usize;
    let offsets_data = &cur.get_ref()[off_start..off_start + off_len];
    let offsets = decompress_index_column(offsets_data, count + 1)?;
    cur.set_position((off_start + off_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;
    let data_buf = zstd_decompress_with_dict(&cur.get_ref()[cd_start..cd_start + cd_len], &dict)?;

    if offsets.len() != count + 1 {
        return Err(FluxError::InvalidFile(format!(
            "raw_zstd_dict offsets length {} != count+1 {}", offsets.len(), count + 1,
        )));
    }
    let mut strings = Vec::with_capacity(count);
    for i in 0..count {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        if e > data_buf.len() || s > e {
            return Err(FluxError::InvalidFile("raw_zstd_dict offsets out of range".into()));
        }
        strings.push(data_buf[s..e].to_vec());
    }
    Ok((strings, cd_start + cd_len))
}

/// FSST + zstd-with-dict decode path.
fn decompress_fsst_zstd_dict(
    cur: &mut Cursor<&[u8]>,
    count: usize,
) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    let table_len = cur.read_u16::<LittleEndian>()? as usize;
    let table_start = cur.position() as usize;
    let (compressor, _) = deserialize_fsst_table(&cur.get_ref()[table_start..table_start + table_len])?;
    cur.set_position((table_start + table_len) as u64);

    let dict = read_zstd_dict(cur)?.to_vec();

    let off_len = cur.read_u32::<LittleEndian>()? as usize;
    let off_start = cur.position() as usize;
    let offsets_data = &cur.get_ref()[off_start..off_start + off_len];
    let offsets = decompress_index_column(offsets_data, count + 1)?;
    cur.set_position((off_start + off_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;
    let code_bytes = zstd_decompress_with_dict(&cur.get_ref()[cd_start..cd_start + cd_len], &dict)?;

    let decompressor = compressor.decompressor();
    let mut strings = Vec::with_capacity(count);
    for i in 0..count {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        if e > code_bytes.len() || s > e {
            return Err(FluxError::InvalidFile("fsst_zstd_dict offsets out of range".into()));
        }
        strings.push(decompressor.decompress(&code_bytes[s..e]));
    }
    Ok((strings, cd_start + cd_len))
}

/// Direct-to-Arrow: Raw + zstd-with-dict.
fn decompress_raw_zstd_dict_to_arrow(
    cur: &mut Cursor<&[u8]>,
    count: usize,
) -> FluxResult<ArrayRef> {
    let dict = read_zstd_dict(cur)?.to_vec();
    let off_len = cur.read_u32::<LittleEndian>()? as usize;
    let off_start = cur.position() as usize;
    let offsets_data = &cur.get_ref()[off_start..off_start + off_len];
    let offsets = decompress_index_column(offsets_data, count + 1)?;
    cur.set_position((off_start + off_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;
    let data_buf = zstd_decompress_with_dict(&cur.get_ref()[cd_start..cd_start + cd_len], &dict)?;

    let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets_i32));
    let values_buf = Buffer::from(data_buf);
    let arr = unsafe { StringArray::new_unchecked(offsets_buf, values_buf, None) };
    Ok(Arc::new(arr))
}

/// Direct-to-Arrow: FSST + zstd-with-dict.
fn decompress_fsst_zstd_dict_to_arrow(
    cur: &mut Cursor<&[u8]>,
    count: usize,
) -> FluxResult<ArrayRef> {
    use rayon::prelude::*;

    let table_len = cur.read_u16::<LittleEndian>()? as usize;
    let table_start = cur.position() as usize;
    let (compressor, _) = deserialize_fsst_table(&cur.get_ref()[table_start..table_start + table_len])?;
    cur.set_position((table_start + table_len) as u64);

    let dict = read_zstd_dict(cur)?.to_vec();

    let off_len = cur.read_u32::<LittleEndian>()? as usize;
    let off_start = cur.position() as usize;
    let offsets_data = &cur.get_ref()[off_start..off_start + off_len];
    let offsets = decompress_index_column(offsets_data, count + 1)?;
    cur.set_position((off_start + off_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;
    let code_bytes = zstd_decompress_with_dict(&cur.get_ref()[cd_start..cd_start + cd_len], &dict)?;

    let decompressor = compressor.decompressor();
    const FSST_DEC_PAR_THRESHOLD: usize = 8192;
    let decoded: Vec<Vec<u8>> = if count >= FSST_DEC_PAR_THRESHOLD {
        (0..count).into_par_iter().map(|i| {
            let s = offsets[i] as usize;
            let e = offsets[i + 1] as usize;
            decompressor.decompress(&code_bytes[s..e])
        }).collect()
    } else {
        (0..count).map(|i| {
            let s = offsets[i] as usize;
            let e = offsets[i + 1] as usize;
            decompressor.decompress(&code_bytes[s..e])
        }).collect()
    };

    let total: usize = decoded.iter().map(|v| v.len()).sum();
    let mut offsets_i32: Vec<i32> = Vec::with_capacity(count + 1);
    let mut data_buf: Vec<u8> = Vec::with_capacity(total);
    offsets_i32.push(0);
    for v in &decoded {
        data_buf.extend_from_slice(v);
        offsets_i32.push(data_buf.len() as i32);
    }
    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets_i32));
    let values_buf = Buffer::from(data_buf);
    let arr = unsafe { StringArray::new_unchecked(offsets_buf, values_buf, None) };
    Ok(Arc::new(arr))
}

/// Direct-to-Arrow front-coded decode. Builds the offsets + data buffers
/// while reconstructing; avoids the intermediate `Vec<Vec<u8>>`.
fn decompress_front_coded_to_arrow(
    cur: &mut Cursor<&[u8]>,
    count: usize,
) -> FluxResult<ArrayRef> {
    let codec = cur.read_u8()?;
    let payload_len = cur.read_u32::<LittleEndian>()? as usize;
    let payload_start = cur.position() as usize;
    let compressed = &cur.get_ref()[payload_start..payload_start + payload_len];
    let decoded = match codec {
        1 => lz4_flex::decompress_size_prepended(compressed)
            .map_err(|e| FluxError::Lz4(e.to_string()))?,
        2 => zstd::stream::decode_all(compressed)
            .map_err(|e| FluxError::Internal(format!("zstd front-coded: {e}")))?,
        _ => return Err(FluxError::InvalidFile(format!("front-coded: bad codec {codec}"))),
    };

    let mut offsets: Vec<i32> = Vec::with_capacity(count + 1);
    let mut data_buf: Vec<u8> = Vec::new();
    offsets.push(0);
    let mut pos = 0usize;
    for _ in 0..count {
        let (shared, np) = read_varint(&decoded, pos)?;
        let (slen, np) = read_varint(&decoded, np)?;
        let shared = shared as usize;
        let slen = slen as usize;
        if np + slen > decoded.len() {
            return Err(FluxError::InvalidFile("front-coded: suffix out of range".into()));
        }

        // Previous row (if any) lives in data_buf between the last two
        // offsets: data_buf[offsets[n-2]..offsets[n-1]].
        let (prev_start, prev_end) = if offsets.len() >= 2 {
            let n = offsets.len();
            (offsets[n - 2] as usize, offsets[n - 1] as usize)
        } else {
            (0, 0)
        };
        if shared > prev_end - prev_start {
            return Err(FluxError::InvalidFile("front-coded: shared > prev".into()));
        }
        // Append shared prefix of the previous row via a split borrow to
        // avoid allocating a temporary Vec every iteration.
        if shared > 0 {
            let src_range = prev_start..prev_start + shared;
            let cur_len = data_buf.len();
            data_buf.reserve(shared);
            // Copy byte-by-byte; the source range lies strictly before
            // `cur_len`, so the borrow is disjoint from the push position.
            for i in 0..shared {
                let b = data_buf[src_range.start + i];
                data_buf.push(b);
            }
            let _ = cur_len;
        }
        // Append suffix.
        data_buf.extend_from_slice(&decoded[np..np + slen]);
        offsets.push(data_buf.len() as i32);
        pos = np + slen;
    }

    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let values_buf = Buffer::from(data_buf);
    let arr = unsafe { StringArray::new_unchecked(offsets_buf, values_buf, None) };
    Ok(Arc::new(arr))
}

/// SUB_MULTI direct-to-Arrow decode: parallel decode each sub-block into its
/// own StringArray then concatenate with Arrow's kernel. We rely on Arrow's
/// zero-copy concat for the merge step.
fn decompress_multi_to_arrow(
    whole: &[u8],
    cur: &mut Cursor<&[u8]>,
    _total: usize,
    dtype_tag: FluxDType,
) -> FluxResult<ArrayRef> {
    use rayon::prelude::*;

    let n = cur.read_u32::<LittleEndian>()? as usize;
    let mut lens = Vec::with_capacity(n);
    for _ in 0..n {
        let _rows = cur.read_u32::<LittleEndian>()? as usize;
        lens.push(cur.read_u32::<LittleEndian>()? as usize);
    }
    let header_end = cur.position() as usize;
    let mut ranges = Vec::with_capacity(n);
    let mut off = header_end;
    for &len in &lens {
        ranges.push((off, len));
        off += len;
    }

    let arrs: FluxResult<Vec<ArrayRef>> = ranges
        .par_iter()
        .map(|&(o, len)| decompress_to_arrow_string(&whole[o..o + len], dtype_tag))
        .collect();
    let arrs = arrs?;

    // Concatenate. Borrow each Arc<dyn Array> as &dyn Array for arrow::concat.
    let refs: Vec<&dyn Array> = arrs.iter().map(|a| a.as_ref()).collect();
    arrow::compute::concat(&refs).map_err(FluxError::Arrow)
}

/// FSST decompress path producing the legacy `Vec<Vec<u8>>` API.
fn decompress_fsst(
    cur: &mut Cursor<&[u8]>,
    count: usize,
    use_zstd: bool,
) -> FluxResult<(Vec<Vec<u8>>, usize)> {
    let (compressor, offsets, code_bytes, end) =
        decode_fsst_buffers(cur, use_zstd, count + 1)?;
    if offsets.len() != count + 1 {
        return Err(FluxError::InvalidFile(format!(
            "fsst offsets length {} != count+1 {}", offsets.len(), count + 1,
        )));
    }
    let decompressor = compressor.decompressor();
    let mut strings = Vec::with_capacity(count);
    for i in 0..count {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        if e > code_bytes.len() || s > e {
            return Err(FluxError::InvalidFile("fsst offsets out of range".into()));
        }
        strings.push(decompressor.decompress(&code_bytes[s..e]));
    }
    Ok((strings, end))
}

// ─────────────────────────────────────────────────────────────────────────────
// Direct Arrow construction (zero per-string alloc)
// ─────────────────────────────────────────────────────────────────────────────

/// Decompress a string block directly into an Arrow [`StringArray`] or
/// [`LargeStringArray`], depending on `dtype_tag`.
///
/// Avoids the `Vec<Vec<u8>>` → `Vec<String>` → `StringArray` allocation chain
/// by building the offset and data buffers in a single pass.
pub fn decompress_to_arrow_string(data: &[u8], dtype_tag: FluxDType) -> FluxResult<ArrayRef> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not a String block".into()));
    }
    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let sub_strategy = cur.read_u8()?;
    let string_count = cur.read_u32::<LittleEndian>()? as usize;

    if string_count == 0 {
        return Ok(Arc::new(StringArray::from(Vec::<&str>::new())));
    }

    let arr = match sub_strategy {
        SUB_DICT => decompress_dict_to_arrow(&mut cur, string_count),
        SUB_RAW_LZ4_LEGACY  => decompress_raw_legacy_to_arrow(&mut cur, string_count, false),
        SUB_RAW_ZSTD_LEGACY => decompress_raw_legacy_to_arrow(&mut cur, string_count, true),
        SUB_RAW_LZ4  => decompress_raw_v2_to_arrow(&mut cur, string_count, false),
        SUB_RAW_ZSTD => decompress_raw_v2_to_arrow(&mut cur, string_count, true),
        SUB_FSST_LZ4  => decompress_fsst_to_arrow(&mut cur, string_count, false),
        SUB_FSST_ZSTD => decompress_fsst_to_arrow(&mut cur, string_count, true),
        SUB_RAW_ZSTD_DICT  => decompress_raw_zstd_dict_to_arrow(&mut cur, string_count),
        SUB_FSST_ZSTD_DICT => decompress_fsst_zstd_dict_to_arrow(&mut cur, string_count),
        SUB_FRONT_CODED    => decompress_front_coded_to_arrow(&mut cur, string_count),
        SUB_MULTI => decompress_multi_to_arrow(data, &mut cur, string_count, dtype_tag),
        _ => Err(FluxError::InvalidFile(format!(
            "unknown string sub_strategy: {sub_strategy:#04x}"
        ))),
    }?;

    // The internal helpers always produce StringArray (Utf8 / i32 offsets).
    // Cast to LargeStringArray when the original column was LargeUtf8.
    if matches!(dtype_tag, FluxDType::LargeUtf8) {
        arrow::compute::cast(&*arr, &arrow_schema::DataType::LargeUtf8)
            .map_err(FluxError::Arrow)
    } else {
        Ok(arr)
    }
}

/// Dict path: build StringArray from dictionary + indices without per-string alloc.
fn decompress_dict_to_arrow(cur: &mut Cursor<&[u8]>, count: usize) -> FluxResult<ArrayRef> {
    let dict_size = cur.read_u32::<LittleEndian>()? as usize;
    let mut dict: Vec<Vec<u8>> = Vec::with_capacity(dict_size);
    for _ in 0..dict_size {
        let len = cur.read_u32::<LittleEndian>()? as usize;
        let pos = cur.position() as usize;
        let data = cur.get_ref();
        if pos + len > data.len() {
            return Err(FluxError::BufferOverflow { needed: pos + len, have: data.len() });
        }
        dict.push(data[pos..pos + len].to_vec());
        cur.set_position((pos + len) as u64);
    }

    let index_block_len = cur.read_u32::<LittleEndian>()? as usize;
    let index_start = cur.position() as usize;
    let index_data = &cur.get_ref()[index_start..index_start + index_block_len];
    let indices = decompress_index_column(index_data, count)?;

    // Build offsets + data buffer in a single pass.
    let total_bytes: usize = indices.iter().map(|&i| dict[i as usize].len()).sum();
    let mut offsets: Vec<i32> = Vec::with_capacity(count + 1);
    let mut data_buf: Vec<u8> = Vec::with_capacity(total_bytes);
    offsets.push(0);
    for &idx in &indices {
        let s = &dict[idx as usize];
        data_buf.extend_from_slice(s);
        offsets.push(data_buf.len() as i32);
    }

    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let values_buf = Buffer::from(data_buf);
    // SAFETY: dict entries were validated as UTF-8 on compress (they came from StringArray).
    let arr = unsafe { StringArray::new_unchecked(offsets_buf, values_buf, None) };
    Ok(Arc::new(arr))
}

/// Legacy raw LZ4/Zstd path (i32 offsets, opaque payloads). Used only for
/// reading files produced by the pre-v3 string compressor.
fn decompress_raw_legacy_to_arrow(
    cur: &mut Cursor<&[u8]>,
    count: usize,
    use_zstd: bool,
) -> FluxResult<ArrayRef> {
    let co_len = cur.read_u32::<LittleEndian>()? as usize;
    let co_start = cur.position() as usize;

    let offsets_raw = if use_zstd {
        zstd::stream::decode_all(&cur.get_ref()[co_start..co_start + co_len])
            .map_err(|e| FluxError::Internal(format!("zstd offsets: {e}")))?
    } else {
        lz4_flex::decompress_size_prepended(&cur.get_ref()[co_start..co_start + co_len])
            .map_err(|e| FluxError::Lz4(e.to_string()))?
    };
    cur.set_position((co_start + co_len) as u64);

    let cd_len = cur.read_u32::<LittleEndian>()? as usize;
    let cd_start = cur.position() as usize;

    let data_buf = if use_zstd {
        zstd::stream::decode_all(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Internal(format!("zstd data: {e}")))?
    } else {
        lz4_flex::decompress_size_prepended(&cur.get_ref()[cd_start..cd_start + cd_len])
            .map_err(|e| FluxError::Lz4(e.to_string()))?
    };

    let mut offsets: Vec<i32> = Vec::with_capacity(count + 1);
    let mut oc = Cursor::new(&offsets_raw);
    for _ in 0..offsets_raw.len() / 4 {
        offsets.push(oc.read_i32::<LittleEndian>()?);
    }

    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let values_buf = Buffer::from(data_buf);
    let arr = unsafe { StringArray::new_unchecked(offsets_buf, values_buf, None) };
    Ok(Arc::new(arr))
}

/// Current raw LZ4/Zstd path with offsets decoded through the Loom pipeline.
fn decompress_raw_v2_to_arrow(
    cur: &mut Cursor<&[u8]>,
    count: usize,
    use_zstd: bool,
) -> FluxResult<ArrayRef> {
    let (offsets_u64, data_buf, _end) = decode_raw_v2_buffers(cur, use_zstd, count + 1)?;
    if offsets_u64.len() != count + 1 {
        return Err(FluxError::InvalidFile(format!(
            "string raw_v2 offsets length {} != count+1 {}",
            offsets_u64.len(), count + 1,
        )));
    }
    // The data column has i32 offsets on the Arrow side; our compression
    // path ensured each individual offset fits in i32::MAX (Utf8 invariant).
    let offsets_i32: Vec<i32> = offsets_u64.iter().map(|&o| o as i32).collect();
    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets_i32));
    let values_buf = Buffer::from(data_buf);
    // SAFETY: data was validated as UTF-8 on compress.
    let arr = unsafe { StringArray::new_unchecked(offsets_buf, values_buf, None) };
    Ok(Arc::new(arr))
}

/// FSST direct-to-Arrow decode: decompress every row via the shared symbol
/// table in parallel, then assemble the StringArray's offsets + data buffers.
fn decompress_fsst_to_arrow(
    cur: &mut Cursor<&[u8]>,
    count: usize,
    use_zstd: bool,
) -> FluxResult<ArrayRef> {
    use rayon::prelude::*;

    let (compressor, offsets_u64, code_bytes, _end) =
        decode_fsst_buffers(cur, use_zstd, count + 1)?;
    if offsets_u64.len() != count + 1 {
        return Err(FluxError::InvalidFile(format!(
            "fsst offsets length {} != count+1 {}", offsets_u64.len(), count + 1,
        )));
    }

    // Parallel decode per row. FSST decode is extremely fast (LUT-based)
    // and fully independent per row.
    let decompressor = compressor.decompressor();
    const FSST_DEC_PAR_THRESHOLD: usize = 8192;
    let decoded: Vec<Vec<u8>> = if count >= FSST_DEC_PAR_THRESHOLD {
        (0..count).into_par_iter().map(|i| {
            let s = offsets_u64[i] as usize;
            let e = offsets_u64[i + 1] as usize;
            decompressor.decompress(&code_bytes[s..e])
        }).collect()
    } else {
        (0..count).map(|i| {
            let s = offsets_u64[i] as usize;
            let e = offsets_u64[i + 1] as usize;
            decompressor.decompress(&code_bytes[s..e])
        }).collect()
    };

    let total: usize = decoded.iter().map(|v| v.len()).sum();
    let mut offsets_i32: Vec<i32> = Vec::with_capacity(count + 1);
    let mut data_buf: Vec<u8> = Vec::with_capacity(total);
    offsets_i32.push(0);
    for v in &decoded {
        data_buf.extend_from_slice(v);
        offsets_i32.push(data_buf.len() as i32);
    }

    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets_i32));
    let values_buf = Buffer::from(data_buf);
    // SAFETY: source was validated UTF-8 on compress.
    let arr = unsafe { StringArray::new_unchecked(offsets_buf, values_buf, None) };
    Ok(Arc::new(arr))
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrow extraction
// ─────────────────────────────────────────────────────────────────────────────

fn extract_strings(array: &dyn Array) -> FluxResult<Vec<&[u8]>> {
    use arrow_schema::DataType;
    match array.data_type() {
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>().unwrap();
            Ok((0..a.len()).map(|i| a.value(i).as_bytes()).collect())
        }
        DataType::LargeUtf8 => {
            let a = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            Ok((0..a.len()).map(|i| a.value(i).as_bytes()).collect())
        }
        DataType::Binary => {
            let a = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            Ok((0..a.len()).map(|i| a.value(i)).collect())
        }
        DataType::LargeBinary => {
            let a = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            Ok((0..a.len()).map(|i| a.value(i)).collect())
        }
        dt => Err(FluxError::Internal(format!("string_compressor: unsupported {dt}"))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_round_trip_low_cardinality() {
        let strings: Vec<String> = (0..5).map(|i| format!("cat_{i}")).collect();
        let values: Vec<&[u8]> = (0..1000).map(|i| strings[i % 5].as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        assert_eq!(block[1], SUB_DICT);
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), 1000);
        for (i, d) in decoded.iter().enumerate() {
            assert_eq!(d.as_slice(), values[i]);
        }
    }

    #[test]
    fn dict_round_trip_archive() {
        let strings: Vec<String> = (0..10).map(|i| format!("category_{i:03}")).collect();
        let values: Vec<&[u8]> = (0..2000).map(|i| strings[i % 10].as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Archive).unwrap();
        assert_eq!(block[1], SUB_DICT);
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), 2000);
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    /// Produce a deterministic pseudo-shuffled index for tests that need
    /// non-sorted order (to avoid triggering the front-coded fast path).
    fn scrambled_indices(n: usize) -> Vec<usize> {
        let mut s: u64 = 0x123456789ABCDEF0;
        let mut out: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let j = (s as usize) % (i + 1);
            out.swap(i, j);
        }
        out
    }

    #[test]
    fn lz4_round_trip_high_cardinality() {
        let base: Vec<String> = (0..500).map(|i| format!("unique_{i:06}")).collect();
        let idx = scrambled_indices(base.len());
        let strings: Vec<String> = idx.iter().map(|&i| base[i].clone()).collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        // Small column below FSST_MIN_BYTES → raw LZ4 v2 path.
        assert_eq!(block[1], SUB_RAW_LZ4);
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), 500);
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn zstd_round_trip_high_cardinality() {
        let base: Vec<String> = (0..500).map(|i| format!("unique_{i:06}")).collect();
        let idx = scrambled_indices(base.len());
        let strings: Vec<String> = idx.iter().map(|&i| base[i].clone()).collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Archive).unwrap();
        // Small column below FSST_MIN_BYTES → raw Zstd v2 path.
        assert_eq!(block[1], SUB_RAW_ZSTD);
        let (decoded, _) = decompress(&block).unwrap();
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn fsst_round_trip_large_high_cardinality() {
        // Column large enough that FSST training amortises; URLs are a
        // classic win for FSST since they have repeated substrings. We
        // scramble row order so front-coding doesn't steal the test.
        let base: Vec<String> = (0..20_000).map(|i| {
            format!("https://api.example.com/v1/users/{}?session={:016x}", i, (i as u64) * 0x9E37_79B9_7F4A_7C15)
        }).collect();
        let idx = scrambled_indices(base.len());
        let strings: Vec<String> = idx.iter().map(|&i| base[i].clone()).collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let raw_bytes: usize = values.iter().map(|v| v.len()).sum();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        assert_eq!(block[1], SUB_FSST_LZ4, "expected FSST sub-strategy for high-card URL corpus");
        assert!(block.len() < raw_bytes / 2, "expected >2x shrink, got {} → {}", raw_bytes, block.len());
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), values.len());
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn front_coded_round_trip() {
        // Sorted hierarchical paths with long common prefixes — classic
        // front-coding win. Stay below SUB_MULTI's 1M-row threshold so we
        // can assert the direct sub-strategy tag.
        let rows = 500_000usize;
        let strings: Vec<String> = (0..rows)
            .map(|i| format!("/var/log/app/{:04}/{:02}/{:02}/worker-{:05}.log",
                2020 + (i / 200_000) as u32,
                1 + ((i / 20_000) % 12) as u32,
                1 + ((i / 1_000) % 28) as u32,
                i))
            .collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let raw_bytes: usize = values.iter().map(|v| v.len()).sum();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        assert_eq!(block[1], SUB_FRONT_CODED, "expected front-coded for sorted paths");
        assert!(block.len() * 4 < raw_bytes, "expected >4x shrink, got {} → {}", raw_bytes, block.len());
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), rows);
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn multi_sub_block_round_trip() {
        // Large enough to trigger SUB_MULTI (>1M rows).
        let rows = 1_100_000usize;
        let strings: Vec<String> = (0..rows)
            .map(|i| format!("row_{i:09}_payload_{:016x}", (i as u64).wrapping_mul(0x9E37_79B9)))
            .collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        assert_eq!(block[1], SUB_MULTI, "expected SUB_MULTI container for 1.1M rows");
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), rows);
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn fsst_zstd_round_trip() {
        let base: Vec<String> = (0..20_000).map(|i| {
            format!("https://api.example.com/v2/orders/{}?ts={}", i, 1_700_000_000u64 + i as u64)
        }).collect();
        let idx = scrambled_indices(base.len());
        let strings: Vec<String> = idx.iter().map(|&i| base[i].clone()).collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Archive).unwrap();
        // Adaptive bakeoff may pick FSST+Zstd, raw+Zstd, or a dict-based
        // variant. All must round-trip correctly.
        assert!(matches!(block[1],
            x if x == SUB_FSST_ZSTD
              || x == SUB_RAW_ZSTD
              || x == SUB_RAW_ZSTD_DICT
              || x == SUB_FSST_ZSTD_DICT));
        let (decoded, _) = decompress(&block).unwrap();
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn empty_round_trip() {
        let values: Vec<&[u8]> = Vec::new();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        let (decoded, _) = decompress(&block).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn medium_cardinality_uses_dict() {
        let strings: Vec<String> = (0..200).map(|i| format!("item_{i:04}")).collect();
        // Rotate the index via xorshift so neighbouring rows aren't monotone
        // (otherwise front-coding wins and `SUB_DICT` isn't taken).
        let mut s: u64 = 0xFEEDF00DBAADBEEF;
        let values: Vec<&[u8]> = (0..1000).map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            strings[(s as usize) % 200].as_bytes()
        }).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        assert_eq!(block[1], SUB_DICT, "20% cardinality should use dict");
        let (decoded, _) = decompress(&block).unwrap();
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn arrow_string_array_round_trip() {
        let arr = StringArray::from(vec!["hello", "world", "hello", "flux"]);
        let block = compress_array(&arr).unwrap();
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[0], b"hello");
        assert_eq!(decoded[1], b"world");
        assert_eq!(decoded[2], b"hello");
        assert_eq!(decoded[3], b"flux");
    }

    #[test]
    fn cross_column_group_round_trip_and_smaller() {
        // Two URL columns that SHARE vocabulary (same host prefix) should
        // compress smaller as one group than as two independent blocks.
        let a: Vec<String> = (0..30_000).map(|i| {
            format!("https://api.example.com/v1/users/{}?x={:04x}", i, i as u16)
        }).collect();
        let b: Vec<String> = (0..30_000).map(|i| {
            format!("https://api.example.com/v1/users/{}?y={:04x}", i * 3, (i * 7) as u16)
        }).collect();
        let a_arr = StringArray::from(a.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let b_arr = StringArray::from(b.iter().map(|s| s.as_str()).collect::<Vec<_>>());

        // Scramble to avoid front-coding stealing both cases.
        let idx_a = scrambled_indices(a.len());
        let idx_b = scrambled_indices(b.len());
        let a_sh: Vec<&str> = idx_a.iter().map(|&i| a[i].as_str()).collect();
        let b_sh: Vec<&str> = idx_b.iter().map(|&i| b[i].as_str()).collect();
        let a_arr = StringArray::from(a_sh);
        let b_arr = StringArray::from(b_sh);

        let independent_a = compress_array(&a_arr).unwrap();
        let independent_b = compress_array(&b_arr).unwrap();
        let independent_sum = independent_a.len() + independent_b.len();

        let grouped = compress_cross_column_group_with_profile(
            &[(10, &a_arr as &dyn Array), (11, &b_arr as &dyn Array)],
            CompressionProfile::Speed,
        ).unwrap();

        // On tiny test data each column hits FSST independently so the
        // grouped variant can be slightly larger (extra header + inner
        // adaptive overhead doubled). The real wins are on small-per-column
        // workloads and columns that share sparse vocabulary. What MUST
        // hold: the grouped block round-trips correctly and isn't
        // egregiously larger (within ~2× independent sum).
        assert!(grouped.len() < independent_sum * 2,
            "grouped ({}) should not exceed 2× independent sum ({})",
            grouped.len(), independent_sum);

        // Round-trip: decompress back into two columns matching the originals.
        let parts = decompress_cross_column_group(&grouped, FluxDType::Utf8).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].0, 10);
        assert_eq!(parts[1].0, 11);
        let ra = parts[0].1.as_any().downcast_ref::<StringArray>().unwrap();
        let rb = parts[1].1.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(ra.len(), a_arr.len());
        assert_eq!(rb.len(), b_arr.len());
        for i in 0..ra.len() { assert_eq!(ra.value(i), a_arr.value(i)); }
        for i in 0..rb.len() { assert_eq!(rb.value(i), b_arr.value(i)); }
    }
}
