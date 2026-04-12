// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! BitSlab compressor: Frame-of-Reference bit-packing with Outlier Map.
//!
//! ## Block Binary Layout
//! ```text
//! [u8:  strategy = 0x04]
//! [u8:  slab_width]
//! [u16: value_count]
//! [u128 (16B): frame_of_reference  (lo-u64, hi-u64)]
//! [u32: slab_byte_len]
//! [slab_byte_len bytes: packed bit-slab]
//! [outlier_map bytes: OutlierMap::to_bytes()]
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Write};

use crate::{
    error::{FluxError, FluxResult},
    outlier_map::{encode_with_outlier_map, decode_with_outlier_map},
    loom_classifier::LoomStrategy,
};

/// Strategy byte tag for BitSlab blocks.
pub const TAG: u8 = 0x04;

/// Compress a slice of `u128` values using the BitSlab strategy.
///
/// Returns the raw block bytes (without the Atlas entry — the caller appends
/// those to the footer).
pub fn compress(values: &[u128]) -> FluxResult<Vec<u8>> {
    assert!(values.len() <= u32::MAX as usize, "block too large");

    let (slab_bytes, outlier_map, slab_width, for_val) = encode_with_outlier_map(values)?;
    let om_bytes = outlier_map.to_bytes()?;

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(0)?; // secondary_codec: None
    buf.write_u8(slab_width)?;
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    // Frame-of-reference as two u64s.
    buf.write_u64::<LittleEndian>(for_val as u64)?;
    buf.write_u64::<LittleEndian>((for_val >> 64) as u64)?;
    buf.write_u32::<LittleEndian>(slab_bytes.len() as u32)?;
    buf.extend_from_slice(&slab_bytes);
    buf.extend_from_slice(&om_bytes);

    Ok(buf)
}

/// Decompress a BitSlab block from `data`.
///
/// Returns `(values, bytes_consumed)`.
pub fn decompress(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not a BitSlab block".into()));
    }

    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let _secondary_codec = cur.read_u8()?;
    let slab_width = cur.read_u8()?;
    let value_count = cur.read_u32::<LittleEndian>()? as usize;
    let for_lo = cur.read_u64::<LittleEndian>()? as u128;
    let for_hi = cur.read_u64::<LittleEndian>()? as u128;
    let for_val = for_lo | (for_hi << 64);
    let slab_len = cur.read_u32::<LittleEndian>()? as usize;

    let header_len = cur.position() as usize;
    let slab_start = header_len;
    let slab_end = slab_start + slab_len;

    if slab_end > data.len() {
        return Err(FluxError::BufferOverflow {
            needed: slab_end,
            have: data.len(),
        });
    }

    let slab_bytes = &data[slab_start..slab_end];
    let outlier_data = &data[slab_end..];

    let values = decode_with_outlier_map(slab_bytes, outlier_data, for_val, slab_width, value_count)?;

    // Compute how many outlier bytes were consumed.
    let om_count = u32::from_le_bytes(
        outlier_data[..4].try_into().map_err(|_| FluxError::InvalidFile("om header".into()))?,
    ) as usize;
    let om_bytes_consumed = 4 + om_count * 16;
    let total_consumed = slab_end + om_bytes_consumed;

    Ok((values, total_consumed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_small() {
        let values: Vec<u128> = (0u128..256).collect();
        let block = compress(&values).unwrap();
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn round_trip_with_giant_outlier() {
        let mut values: Vec<u128> = (1000u128..1500).collect();
        values.push(u128::MAX);
        values.push(u128::MAX - 1);
        let block = compress(&values).unwrap();
        let (decoded, _) = decompress(&block).unwrap();
        assert_eq!(decoded, values);
    }
}
