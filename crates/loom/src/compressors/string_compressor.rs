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
//! ## Block Layout (v2)
//! ```text
//! [u8:  TAG = 0x06]
//! [u8:  sub_strategy]           // 0 = dict, 1 = raw_lz4, 2 = raw_zstd
//! [u32: string_count]
//! --- if sub_strategy == 0 (dict) ---
//! [u32: dict_size]
//! [dict_size × (u32: len, len bytes: value)]
//! [u32: index_block_len]
//! [index_block bytes]           // Loom-compressed u64 index column
//! --- if sub_strategy == 1 (raw_lz4) or 2 (raw_zstd) ---
//! [u32: compressed_offsets_len]
//! [compressed_offsets bytes]
//! [u32: compressed_data_len]
//! [compressed_data bytes]
//! ```

use arrow_array::array::Array;
use arrow_array::{ArrayRef, StringArray, BinaryArray, LargeStringArray, LargeBinaryArray};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Write};
use std::sync::Arc;

use crate::error::{FluxError, FluxResult};
use crate::CompressionProfile;

/// Strategy byte tag for String/Binary blocks.
pub const TAG: u8 = 0x06;

const SUB_DICT: u8 = 0;
const SUB_RAW_LZ4: u8 = 1;
const SUB_RAW_ZSTD: u8 = 2;

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
            compress_raw(values, profile)
        }
    } else {
        compress_raw(values, profile)
    }
}

fn compress_empty() -> FluxResult<Vec<u8>> {
    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(SUB_RAW_LZ4)?;
    buf.write_u32::<LittleEndian>(0)?;
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

/// Compress the dict index column through the full Loom pipeline.
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

// ─────────────────────────────────────────────────────────────────────────────
// Raw path: LZ4 or Zstd on offsets + data
// ─────────────────────────────────────────────────────────────────────────────

fn compress_raw(values: &[&[u8]], profile: CompressionProfile) -> FluxResult<Vec<u8>> {
    let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
    let mut data_buf: Vec<u8> = Vec::new();
    offsets.push(0);
    for &v in values {
        data_buf.extend_from_slice(v);
        offsets.push(data_buf.len() as i32);
    }

    let offsets_raw: Vec<u8> = offsets.iter().flat_map(|&o| o.to_le_bytes()).collect();
    let use_zstd = matches!(profile, CompressionProfile::Archive);

    let (co, cd, sub) = if use_zstd {
        (
            zstd::stream::encode_all(offsets_raw.as_slice(), 3)
                .map_err(|e| FluxError::Internal(format!("zstd offsets: {e}")))?,
            zstd::stream::encode_all(data_buf.as_slice(), 3)
                .map_err(|e| FluxError::Internal(format!("zstd data: {e}")))?,
            SUB_RAW_ZSTD,
        )
    } else {
        (
            lz4_flex::compress_prepend_size(&offsets_raw),
            lz4_flex::compress_prepend_size(&data_buf),
            SUB_RAW_LZ4,
        )
    };

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(sub)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u32::<LittleEndian>(co.len() as u32)?;
    buf.extend_from_slice(&co);
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
        SUB_RAW_LZ4 => decompress_raw(&mut cur, string_count, false),
        SUB_RAW_ZSTD => decompress_raw(&mut cur, string_count, true),
        _ => Err(FluxError::InvalidFile(format!(
            "unknown string sub_strategy: {sub_strategy:#04x}"
        ))),
    }
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

fn decompress_raw(
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

// ─────────────────────────────────────────────────────────────────────────────
// Direct Arrow construction (zero per-string alloc)
// ─────────────────────────────────────────────────────────────────────────────

/// Decompress a string block directly into an Arrow [`StringArray`].
///
/// Avoids the `Vec<Vec<u8>>` → `Vec<String>` → `StringArray` allocation chain
/// by building the offset and data buffers in a single pass.
pub fn decompress_to_arrow_string(data: &[u8]) -> FluxResult<ArrayRef> {
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

    match sub_strategy {
        SUB_DICT => decompress_dict_to_arrow(&mut cur, string_count),
        SUB_RAW_LZ4 => decompress_raw_to_arrow(&mut cur, string_count, false),
        SUB_RAW_ZSTD => decompress_raw_to_arrow(&mut cur, string_count, true),
        _ => Err(FluxError::InvalidFile(format!(
            "unknown string sub_strategy: {sub_strategy:#04x}"
        ))),
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

/// Raw LZ4/Zstd path: decompress offsets + data, build StringArray directly.
fn decompress_raw_to_arrow(
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

    // Parse i32 offsets from decompressed bytes.
    let mut offsets: Vec<i32> = Vec::with_capacity(count + 1);
    let mut oc = Cursor::new(&offsets_raw);
    for _ in 0..offsets_raw.len() / 4 {
        offsets.push(oc.read_i32::<LittleEndian>()?);
    }

    let offsets_buf = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let values_buf = Buffer::from(data_buf);
    // SAFETY: data was validated as UTF-8 on compress.
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

    #[test]
    fn lz4_round_trip_high_cardinality() {
        let strings: Vec<String> = (0..500).map(|i| format!("unique_{i:06}")).collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Speed).unwrap();
        assert_eq!(block[1], SUB_RAW_LZ4);
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded.len(), 500);
        for (i, d) in decoded.iter().enumerate() { assert_eq!(d.as_slice(), values[i]); }
    }

    #[test]
    fn zstd_round_trip_high_cardinality() {
        let strings: Vec<String> = (0..500).map(|i| format!("unique_{i:06}")).collect();
        let values: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
        let block = compress_strings_with_profile(&values, CompressionProfile::Archive).unwrap();
        assert_eq!(block[1], SUB_RAW_ZSTD);
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
        let values: Vec<&[u8]> = (0..1000).map(|i| strings[i % 200].as_bytes()).collect();
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
}
