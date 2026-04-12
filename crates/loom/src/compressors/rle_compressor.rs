// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Run-Length Encoding compressor.
//!
//! ## Block Layout
//! ```text
//! [u8:  TAG = 0x01]
//! [u16: run_count]
//! [run_count × (u128 value, u16 length)]
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Write};
use crate::error::{FluxError, FluxResult};

pub const TAG: u8 = 0x01;

/// Compress using RLE.
pub fn compress(values: &[u128]) -> FluxResult<Vec<u8>> {
    // Build runs.
    let mut runs: Vec<(u128, u16)> = Vec::new();
    if values.is_empty() {
        let mut buf = Vec::new();
        buf.write_u8(TAG)?;
        buf.write_u8(0)?; // secondary_codec
        buf.write_u32::<LittleEndian>(0)?;
        return Ok(buf);
    }

    let mut cur_val = values[0];
    let mut cur_len: u16 = 1;

    for &v in &values[1..] {
        if v == cur_val && cur_len < u16::MAX {
            cur_len += 1;
        } else {
            runs.push((cur_val, cur_len));
            cur_val = v;
            cur_len = 1;
        }
    }
    runs.push((cur_val, cur_len));

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(0)?; // secondary_codec: None
    buf.write_u32::<LittleEndian>(runs.len() as u32)?;
    for (val, len) in &runs {
        buf.write_u64::<LittleEndian>(*val as u64)?;
        buf.write_u64::<LittleEndian>((*val >> 64) as u64)?;
        buf.write_u16::<LittleEndian>(*len)?;
    }
    Ok(buf)
}

/// Decompress an RLE block.  Returns `(values, bytes_consumed)`.
pub fn decompress(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not an RLE block".into()));
    }
    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let _secondary_codec = cur.read_u8()?;
    let run_count = cur.read_u32::<LittleEndian>()? as usize;

    let mut values = Vec::new();
    for _ in 0..run_count {
        let lo = cur.read_u64::<LittleEndian>()? as u128;
        let hi = cur.read_u64::<LittleEndian>()? as u128;
        let val = lo | (hi << 64);
        let len = cur.read_u16::<LittleEndian>()? as usize;
        values.extend(std::iter::repeat(val).take(len));
    }

    Ok((values, cur.position() as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_constant() {
        let values = vec![999u128; 1024];
        let b = compress(&values).unwrap();
        let (d, _) = decompress(&b).unwrap();
        assert_eq!(d, values);
    }

    #[test]
    fn rle_alternating() {
        let values: Vec<u128> = (0u128..512).flat_map(|i| [i, i]).collect();
        let b = compress(&values).unwrap();
        let (d, _) = decompress(&b).unwrap();
        assert_eq!(d, values);
    }
}
