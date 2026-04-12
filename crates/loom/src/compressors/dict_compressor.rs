// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Dictionary Encoding compressor for low-cardinality columns.
//!
//! ## Block Layout
//! ```text
//! [u8:  TAG = 0x03]
//! [u16: value_count]
//! [u16: dict_size]
//! [dict_size × u128 (16B each): dictionary entries]
//! [u8:  index_width]
//! [packed index bitstream]
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Write};
use crate::{
    bit_io::{BitWriter, BitReader, bits_needed},
    error::{FluxError, FluxResult},
};

pub const TAG: u8 = 0x03;

pub fn compress(values: &[u128]) -> FluxResult<Vec<u8>> {
    // Build dictionary.
    let mut dict: Vec<u128> = Vec::new();
    let mut index_map: HashMap<u128, u16> = HashMap::new();
    for &v in values {
        if !index_map.contains_key(&v) {
            index_map.insert(v, dict.len() as u16);
            dict.push(v);
        }
    }
    assert!(dict.len() <= u16::MAX as usize, "dictionary too large");

    let index_width = bits_needed((dict.len() as u128).saturating_sub(1)).max(1);
    let mut writer = BitWriter::new(index_width);
    for &v in values {
        writer.write_value(*index_map.get(&v).unwrap() as u64)?;
    }
    let slab = writer.finish();

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(0)?; // secondary_codec: None
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u16::<LittleEndian>(dict.len() as u16)?;
    for &d in &dict {
        buf.write_u64::<LittleEndian>(d as u64)?;
        buf.write_u64::<LittleEndian>((d >> 64) as u64)?;
    }
    buf.write_u8(index_width)?;
    buf.write_u32::<LittleEndian>(slab.len() as u32)?;
    buf.extend_from_slice(&slab);

    Ok(buf)
}

pub fn decompress(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not a Dict block".into()));
    }
    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let _secondary_codec = cur.read_u8()?;
    let value_count = cur.read_u32::<LittleEndian>()? as usize;
    let dict_size = cur.read_u16::<LittleEndian>()? as usize;

    let mut dict = Vec::with_capacity(dict_size);
    for _ in 0..dict_size {
        let lo = cur.read_u64::<LittleEndian>()? as u128;
        let hi = cur.read_u64::<LittleEndian>()? as u128;
        dict.push(lo | (hi << 64));
    }
    let index_width = cur.read_u8()?;
    let slab_len = cur.read_u32::<LittleEndian>()? as usize;
    let header_end = cur.position() as usize;

    let slab = &data[header_end..header_end + slab_len];
    let mut reader = BitReader::new(slab, index_width);
    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        let idx = reader.read_value().unwrap_or(0) as usize;
        values.push(*dict.get(idx).ok_or_else(|| {
            FluxError::InvalidFile(format!("dict index {idx} out of range"))
        })?);
    }

    Ok((values, header_end + slab_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_round_trip() {
        let values: Vec<u128> = (0u128..1024).map(|i| i % 5).collect();
        let b = compress(&values).unwrap();
        let (d, _) = decompress(&b).unwrap();
        assert_eq!(d, values);
    }
}
