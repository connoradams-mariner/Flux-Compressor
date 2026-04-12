// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Delta-Delta Encoding compressor for sorted / sequential columns.
//!
//! ## Block Layout
//! ```text
//! [u8:  TAG = 0x02]
//! [u16: value_count]
//! [u128 (16B): first_value]
//! [u128 (16B): first_delta]
//! [u8:  delta_delta_width (bits)]
//! [packed delta-delta bitstream]
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Write};
use crate::{
    bit_io::{BitWriter, BitReader, bits_needed},
    error::{FluxError, FluxResult},
};

pub const TAG: u8 = 0x02;

pub fn compress(values: &[u128]) -> FluxResult<Vec<u8>> {
    if values.len() < 2 {
        return Err(FluxError::Internal("delta encoding requires ≥2 values".into()));
    }

    // Compute first-order deltas (as i128).
    let deltas: Vec<i128> = values
        .windows(2)
        .map(|w| w[1] as i128 - w[0] as i128)
        .collect();

    let first_delta = deltas[0];

    // Second-order deltas (delta of deltas).
    let dd: Vec<i128> = deltas
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    // Find bit-width for delta-deltas.
    let max_dd = dd.iter().map(|&d| d.unsigned_abs()).max().unwrap_or(0);
    let dd_width = bits_needed(max_dd).max(1);

    // Pack delta-deltas using a sign bit.
    let pack_width = (dd_width + 1).min(64); // +1 for sign bit
    let mut writer = BitWriter::new(pack_width);
    for &d in &dd {
        let sign_bit = if d < 0 { 1u64 } else { 0u64 };
        let mag = d.unsigned_abs() as u64;
        let packed = (sign_bit << (pack_width - 1)) | (mag & ((1u64 << (pack_width - 1)) - 1));
        writer.write_value(packed)?;
    }
    let slab = writer.finish();

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(0)?; // secondary_codec: None
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    // first_value
    buf.write_u64::<LittleEndian>(values[0] as u64)?;
    buf.write_u64::<LittleEndian>((values[0] >> 64) as u64)?;
    // first_delta (i128 as two i64)
    buf.write_i64::<LittleEndian>(first_delta as i64)?;
    buf.write_i64::<LittleEndian>((first_delta >> 64) as i64)?;
    buf.write_u8(pack_width)?;
    buf.write_u32::<LittleEndian>(slab.len() as u32)?;
    buf.extend_from_slice(&slab);

    Ok(buf)
}

pub fn decompress(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not a Delta block".into()));
    }
    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let _secondary_codec = cur.read_u8()?;
    let value_count = cur.read_u32::<LittleEndian>()? as usize;
    let fv_lo = cur.read_u64::<LittleEndian>()? as u128;
    let fv_hi = cur.read_u64::<LittleEndian>()? as u128;
    let first_value = fv_lo | (fv_hi << 64);
    let fd_lo = cur.read_i64::<LittleEndian>()? as i128;
    let fd_hi = cur.read_i64::<LittleEndian>()? as i128;
    let first_delta = fd_lo | (fd_hi << 64);
    let pack_width = cur.read_u8()?;
    let slab_len = cur.read_u32::<LittleEndian>()? as usize;

    let header_end = cur.position() as usize;
    let slab = &data[header_end..header_end + slab_len];

    let dd_count = value_count.saturating_sub(2);
    let sign_mask = 1u64 << (pack_width - 1);
    let mag_mask = sign_mask - 1;

    let mut reader = BitReader::new(slab, pack_width);
    let mut dd = Vec::with_capacity(dd_count);
    for _ in 0..dd_count {
        let packed = reader.read_value().unwrap_or(0);
        let negative = (packed & sign_mask) != 0;
        let mag = (packed & mag_mask) as i128;
        dd.push(if negative { -mag } else { mag });
    }

    // Reconstruct from first_value + first_delta + delta-deltas.
    let mut values = Vec::with_capacity(value_count);
    values.push(first_value);
    if value_count > 1 {
        values.push((first_value as i128 + first_delta) as u128);
    }
    let mut cur_delta = first_delta;
    for d in dd {
        cur_delta += d;
        let last = *values.last().unwrap() as i128;
        values.push((last + cur_delta) as u128);
    }

    let total = header_end + slab_len;
    Ok((values, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_round_trip() {
        let values: Vec<u128> = (0u128..1024).collect();
        let b = compress(&values).unwrap();
        let (d, _) = decompress(&b).unwrap();
        assert_eq!(d, values);
    }

    #[test]
    fn arithmetic_stride_round_trip() {
        let values: Vec<u128> = (0u128..512).map(|i| i * 7 + 100).collect();
        let b = compress(&values).unwrap();
        let (d, _) = decompress(&b).unwrap();
        assert_eq!(d, values);
    }
}
