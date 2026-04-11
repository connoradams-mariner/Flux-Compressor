// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Outlier Map (Sprint 2)
//!
//! The "secret sauce" for large computed numbers.
//!
//! ## Mathematical Model
//! Given a column of values `V` with Frame-of-Reference `FoR = min(V)` and
//! slab width `W` chosen at the 99th percentile:
//!
//! ```text
//! sentinel = (1 << W) - 1               // all-ones for W bits
//! For each value x:
//!   delta = x - FoR
//!   if delta >= sentinel:
//!     write sentinel to primary slab
//!     append x (full u128) to OutlierMap
//!   else:
//!     write delta to primary slab
//! ```
//!
//! On decode the SIMD lane reads W bits; whenever it sees `sentinel` it halts,
//! fetches the next entry from the OutlierMap, and advances the patch pointer.
//!
//! ## Binary Layout
//! ```text
//! [u32: count][u128 × count]
//! ```
//! Each entry is a raw little-endian u128 (16 bytes), preserving full
//! precision for aggregation results that overflow 64 bits.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Write};
use crate::error::{FluxError, FluxResult};

// ─────────────────────────────────────────────────────────────────────────────
// OutlierMap
// ─────────────────────────────────────────────────────────────────────────────

/// Stores full-precision `u128` values for rows that overflow the primary slab.
///
/// This is the "Outlier Map" / "Patch Buffer" described in the spec.
#[derive(Debug, Default, Clone)]
pub struct OutlierMap {
    /// The raw overflow values, in insertion order (row order).
    entries: Vec<u128>,
}

impl OutlierMap {
    /// Create an empty [`OutlierMap`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a full-precision `u128` outlier value.
    #[inline]
    pub fn push(&mut self, value: u128) {
        self.entries.push(value);
    }

    /// Number of outlier entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no outlier entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all outlier values.
    pub fn iter(&self) -> impl Iterator<Item = u128> + '_ {
        self.entries.iter().copied()
    }

    /// Serialise the map to bytes:
    /// `[u32: count][u128 × count (little-endian, 16 bytes each)]`
    pub fn to_bytes(&self) -> FluxResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(4 + self.entries.len() * 16);
        buf.write_u32::<LittleEndian>(self.entries.len() as u32)?;
        for &v in &self.entries {
            // u128 as 2 × little-endian u64 (low word first).
            buf.write_u64::<LittleEndian>(v as u64)?;
            buf.write_u64::<LittleEndian>((v >> 64) as u64)?;
        }
        Ok(buf)
    }

    /// Deserialise from a byte slice.  Returns `(map, bytes_consumed)`.
    pub fn from_bytes(data: &[u8]) -> FluxResult<(Self, usize)> {
        if data.len() < 4 {
            return Err(FluxError::InvalidFile(
                "outlier map header truncated".into(),
            ));
        }
        let mut cur = Cursor::new(data);
        let count = cur.read_u32::<LittleEndian>()? as usize;
        let needed = 4 + count * 16;
        if data.len() < needed {
            return Err(FluxError::BufferOverflow {
                needed,
                have: data.len(),
            });
        }

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let lo = cur.read_u64::<LittleEndian>()? as u128;
            let hi = cur.read_u64::<LittleEndian>()? as u128;
            entries.push(lo | (hi << 64));
        }

        Ok((Self { entries }, needed))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OutlierMapReader (zero-copy, index-based)
// ─────────────────────────────────────────────────────────────────────────────

/// A zero-copy reader over a serialised [`OutlierMap`] byte slice.
///
/// The SIMD unpacker calls [`OutlierMapReader::next`] whenever it encounters
/// a sentinel value in the primary slab.
pub struct OutlierMapReader<'a> {
    data: &'a [u8],
    count: usize,
    /// Index of the next patch to return.
    patch_idx: usize,
}

impl<'a> OutlierMapReader<'a> {
    /// Create a reader over the raw bytes of a serialised [`OutlierMap`].
    ///
    /// `data` must begin at the first byte of the map (the `u32` count field).
    pub fn new(data: &'a [u8]) -> FluxResult<Self> {
        if data.len() < 4 {
            return Err(FluxError::InvalidFile(
                "outlier map reader: slice too short".into(),
            ));
        }
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        Ok(Self { data, count, patch_idx: 0 })
    }

    /// Fetch the next outlier value.  Returns `None` when all patches are
    /// exhausted.
    #[inline]
    pub fn next(&mut self) -> Option<u128> {
        if self.patch_idx >= self.count {
            return None;
        }
        let offset = 4 + self.patch_idx * 16;
        if offset + 16 > self.data.len() {
            return None;
        }
        let lo = u64::from_le_bytes(self.data[offset..offset + 8].try_into().unwrap()) as u128;
        let hi =
            u64::from_le_bytes(self.data[offset + 8..offset + 16].try_into().unwrap()) as u128;
        self.patch_idx += 1;
        Some(lo | (hi << 64))
    }

    /// Total number of patches in this map.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset the reader to the start of the patch list.
    #[inline]
    pub fn reset(&mut self) {
        self.patch_idx = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode / Decode helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a slice of `u128` values into a primary-slab bitstream + outlier map.
///
/// Returns `(slab_bytes, outlier_map, slab_width, frame_of_reference)`.
///
/// The slab stores `(value - FoR)` in `slab_width` bits.  Outliers are
/// replaced with the all-ones sentinel in the slab and their original value is
/// appended to the [`OutlierMap`].
pub fn encode_with_outlier_map(
    values: &[u128],
) -> FluxResult<(Vec<u8>, OutlierMap, u8, u128)> {
    use crate::bit_io::{BitWriter, discover_width};

    let (slab_width, for_val) = discover_width(values);
    let sentinel = if slab_width == 64 {
        u64::MAX
    } else {
        (1u64 << slab_width) - 1
    };

    let mut writer = BitWriter::new(slab_width);
    let mut outlier_map = OutlierMap::new();

    for &v in values {
        let delta = v.saturating_sub(for_val);
        if delta >= sentinel as u128 {
            // Overflow → write sentinel to slab, stash full value in outlier map.
            writer.write_sentinel()?;
            outlier_map.push(v);
        } else {
            writer.write_value(delta as u64)?;
        }
    }

    Ok((writer.finish(), outlier_map, slab_width, for_val))
}

/// Decode a primary slab + outlier map back into `u128` values.
///
/// `slab_bytes` is the packed bitstream, `outlier_data` is the serialised
/// [`OutlierMap`] bytes, `for_val` is the Frame-of-Reference, `slab_width` is
/// the bit-width per value, and `count` is the expected number of values.
pub fn decode_with_outlier_map(
    slab_bytes: &[u8],
    outlier_data: &[u8],
    for_val: u128,
    slab_width: u8,
    count: usize,
) -> FluxResult<Vec<u128>> {
    use crate::bit_io::BitReader;

    let mut reader = BitReader::new(slab_bytes, slab_width);
    let mut patches = OutlierMapReader::new(outlier_data)?;
    let mut out = Vec::with_capacity(count);

    for _ in 0..count {
        match reader.read_value() {
            None => break,
            Some(raw) if reader.is_sentinel(raw) => {
                // Fetch full-precision value from the outlier map.
                let full = patches.next().ok_or_else(|| {
                    FluxError::InvalidFile("outlier map exhausted prematurely".into())
                })?;
                out.push(full);
            }
            Some(delta) => {
                out.push(for_val.wrapping_add(delta as u128));
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_no_outliers() {
        let values: Vec<u128> = (100u128..200).collect();
        let (slab, om, w, for_val) = encode_with_outlier_map(&values).unwrap();
        let bytes = om.to_bytes().unwrap();
        let decoded =
            decode_with_outlier_map(&slab, &bytes, for_val, w, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn round_trip_with_u128_outlier() {
        let mut values: Vec<u128> = (0u128..50).collect();
        let giant = u128::MAX / 2 + 12345678901234567890;
        values.push(giant);
        values.extend(50u128..100);

        let (slab, om, w, for_val) = encode_with_outlier_map(&values).unwrap();
        assert!(!om.is_empty(), "giant value should be in outlier map");

        let om_bytes = om.to_bytes().unwrap();
        let decoded =
            decode_with_outlier_map(&slab, &om_bytes, for_val, w, values.len()).unwrap();

        assert_eq!(decoded, values);
    }

    #[test]
    fn outlier_map_serialisation() {
        let mut map = OutlierMap::new();
        map.push(u128::MAX);
        map.push(0xDEADBEEF_CAFEBABE_12345678_9ABCDEF0);
        let bytes = map.to_bytes().unwrap();
        let (decoded, consumed) = OutlierMap::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(decoded.entries, map.entries);
    }
}
