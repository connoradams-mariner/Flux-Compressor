// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! SIMD-LZ4 fallback compressor for high-entropy / unstructured data.
//!
//! Uses `lz4_flex` (a pure-Rust, SIMD-accelerated LZ4 implementation) as the
//! compression back-end.  This is the last step in the Loom waterfall and is
//! chosen only when no structured strategy (RLE, Delta, Dict, BitSlab) applies.
//!
//! ## Block Layout
//! ```text
//! [u8:  TAG = 0x05]
//! [u16: value_count]
//! [u32: uncompressed_len]
//! [u32: compressed_len]
//! [compressed_len bytes: LZ4-compressed raw u128 values]
//! ```
//!
//! Raw values are serialised as little-endian pairs of u64 (lo, hi) before
//! compression, giving LZ4 the best opportunity to find repeated byte patterns.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Write};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use crate::error::{FluxError, FluxResult};

pub const TAG: u8 = 0x05;

/// Compress a slice of `u128` values with LZ4.
pub fn compress(values: &[u128]) -> FluxResult<Vec<u8>> {
    // Serialise to raw bytes: each u128 as (lo:u64, hi:u64) little-endian.
    let mut raw = Vec::with_capacity(values.len() * 16);
    for &v in values {
        raw.write_u64::<LittleEndian>(v as u64)?;
        raw.write_u64::<LittleEndian>((v >> 64) as u64)?;
    }

    let compressed = compress_prepend_size(&raw);

    let mut buf = Vec::new();
    buf.write_u8(TAG)?;
    buf.write_u8(0)?; // secondary_codec: None
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u32::<LittleEndian>(raw.len() as u32)?;
    buf.write_u32::<LittleEndian>(compressed.len() as u32)?;
    buf.extend_from_slice(&compressed);

    Ok(buf)
}

/// Decompress an LZ4 block.  Returns `(values, bytes_consumed)`.
pub fn decompress(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not an LZ4 block".into()));
    }

    let mut cur = Cursor::new(data);
    let _tag              = cur.read_u8()?;
    let _secondary_codec  = cur.read_u8()?;
    let value_count       = cur.read_u32::<LittleEndian>()? as usize;
    let _uncomp_len  = cur.read_u32::<LittleEndian>()? as usize;
    let comp_len     = cur.read_u32::<LittleEndian>()? as usize;
    let header_end   = cur.position() as usize;

    let compressed = &data[header_end..header_end + comp_len];
    let raw = decompress_size_prepended(compressed)
        .map_err(|e| FluxError::Lz4(e.to_string()))?;

    if raw.len() < value_count * 16 {
        return Err(FluxError::BufferOverflow {
            needed: value_count * 16,
            have: raw.len(),
        });
    }

    let mut values = Vec::with_capacity(value_count);
    let mut rc = Cursor::new(&raw);
    for _ in 0..value_count {
        let lo = rc.read_u64::<LittleEndian>()? as u128;
        let hi = rc.read_u64::<LittleEndian>()? as u128;
        values.push(lo | (hi << 64));
    }

    Ok((values, header_end + comp_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_round_trip() {
        // High-entropy-looking data (pseudo-random stride).
        let values: Vec<u128> = (0u128..512)
            .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0xdeadbeef))
            .collect();
        let b = compress(&values).unwrap();
        let (d, consumed) = decompress(&b).unwrap();
        assert_eq!(consumed, b.len());
        assert_eq!(d, values);
    }

    #[test]
    fn lz4_empty() {
        let b = compress(&[]).unwrap();
        let (d, _) = decompress(&b).unwrap();
        assert!(d.is_empty());
    }
}
