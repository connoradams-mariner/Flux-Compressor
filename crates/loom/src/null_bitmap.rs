// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Null bitmap serialization primitive (Part 1 of the remaining
//! roadmap).
//!
//! The [`BlockMeta::null_bitmap_offset`] field already exists in the
//! Atlas footer; what was missing was a way to serialise Arrow
//! validity buffers into the file tail and read them back into an
//! Arrow array on decompression.  This module provides the
//! low-level primitive plus round-trip tests — wiring it into the
//! per-block write path is a small follow-up in `flux_writer.rs`.
//!
//! ## Format
//!
//! ```text
//! [u32 length][bitmap bytes (ceil(value_count / 8))]
//! ```
//!
//! The 4-byte length prefix lets the reader skip the bitmap payload
//! in one seek and makes the format forward-compatible with
//! future extensions (compressed bitmaps, run-length encoding).  For
//! fully non-null columns — the common case — the writer leaves
//! `null_bitmap_offset = 0` and spends zero bytes on the bitmap.
//!
//! ## Semantics
//!
//! Arrow validity buffers store `1 = valid, 0 = null` in little-endian
//! bit order within each byte. We preserve that layout verbatim so no
//! re-packing is required on either side of the round-trip.
//!
//! [`BlockMeta::null_bitmap_offset`]: crate::atlas::BlockMeta::null_bitmap_offset

use arrow_buffer::{BooleanBuffer, NullBuffer};

use crate::error::{FluxError, FluxResult};

/// Serialise an Arrow [`NullBuffer`] into the inline format that
/// goes after the compressed block data when
/// `BlockMeta.null_bitmap_offset != 0`.
///
/// Returns `None` when the buffer has no nulls — callers use that
/// signal to leave `null_bitmap_offset = 0` and avoid adding any
/// bytes to the file for fully-non-null columns.
pub fn encode(nulls: Option<&NullBuffer>) -> Option<Vec<u8>> {
    let buf = nulls?;
    if buf.null_count() == 0 {
        return None;
    }
    let bytes = buf.inner().inner().as_slice();
    let len = bytes.len() as u32;
    let mut out = Vec::with_capacity(4 + bytes.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Some(out)
}

/// Read a serialised null bitmap produced by [`encode`]. `value_count`
/// is the number of logical values the column carries — used to
/// trim the final byte if the serialiser rounded up to a whole byte.
pub fn decode(data: &[u8], value_count: usize) -> FluxResult<NullBuffer> {
    if data.len() < 4 {
        return Err(FluxError::InvalidFile(
            "null bitmap too short for length prefix".into(),
        ));
    }
    let len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    let expected = (value_count + 7) / 8;
    if len != expected {
        return Err(FluxError::InvalidFile(format!(
            "null bitmap length mismatch: got {len} bytes, expected {expected} for {value_count} values"
        )));
    }
    if data.len() < 4 + len {
        return Err(FluxError::InvalidFile(
            "null bitmap body shorter than its length prefix".into(),
        ));
    }
    let body = &data[4..4 + len];
    let boolean = BooleanBuffer::new(body.to_vec().into(), 0, value_count);
    Ok(NullBuffer::new(boolean))
}

/// Convenience: build a [`NullBuffer`] from an iterator of validity
/// booleans (`true = valid, false = null`). Used by tests.
pub fn from_validity_iter<I>(iter: I, len: usize) -> NullBuffer
where
    I: IntoIterator<Item = bool>,
{
    let mut bytes = vec![0u8; (len + 7) / 8];
    for (i, valid) in iter.into_iter().enumerate().take(len) {
        if valid {
            bytes[i / 8] |= 1u8 << (i % 8);
        }
    }
    let boolean = BooleanBuffer::new(bytes.into(), 0, len);
    NullBuffer::new(boolean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_none_for_fully_valid() {
        let nulls = from_validity_iter(std::iter::repeat(true).take(100), 100);
        // 0 nulls -> encode returns None (zero-overhead fast path).
        assert!(encode(Some(&nulls)).is_none());
    }

    #[test]
    fn encode_none_when_buffer_absent() {
        assert!(encode(None).is_none());
    }

    #[test]
    fn round_trip_sparse_nulls() {
        let pattern: Vec<bool> = (0..100).map(|i| i % 3 != 0).collect();
        let nulls = from_validity_iter(pattern.iter().copied(), 100);
        let bytes = encode(Some(&nulls)).expect("has nulls");
        let decoded = decode(&bytes, 100).unwrap();
        assert_eq!(decoded.null_count(), nulls.null_count());
        for i in 0..100 {
            assert_eq!(decoded.is_valid(i), nulls.is_valid(i), "bit {i}");
        }
    }

    #[test]
    fn round_trip_all_null() {
        let nulls = from_validity_iter(std::iter::repeat(false).take(64), 64);
        let bytes = encode(Some(&nulls)).expect("has nulls");
        let decoded = decode(&bytes, 64).unwrap();
        assert_eq!(decoded.null_count(), 64);
    }

    #[test]
    fn decode_rejects_length_mismatch() {
        // Produce a bitmap for 64 values, then try to decode it as 80.
        let nulls = from_validity_iter((0..64).map(|i| i % 2 == 0), 64);
        let bytes = encode(Some(&nulls)).expect("has nulls");
        let err = decode(&bytes, 80).unwrap_err();
        assert!(err.to_string().contains("length mismatch"));
    }

    #[test]
    fn decode_rejects_truncated_payload() {
        let nulls = from_validity_iter((0..64).map(|i| i % 2 == 0), 64);
        let mut bytes = encode(Some(&nulls)).expect("has nulls");
        bytes.truncate(6); // Keep length prefix + 2 bytes of payload.
        let err = decode(&bytes, 64).unwrap_err();
        assert!(err.to_string().contains("shorter"));
    }
}
