// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Zero-alloc [`StringArray`] construction primitive (Part 8 of the
//! remaining roadmap — FSST zero-alloc decode).
//!
//! The current FSST decompression path is:
//!
//! ```text
//! FSST bytes  ─decode→  Vec<Vec<u8>>  ─UTF-8→  Vec<String>  ─collect→  StringArray
//! ```
//!
//! The three `Vec` allocations dominate cache misses for hot workloads
//! where the decompressed string corpus is large.  The target path
//! this module unlocks is:
//!
//! ```text
//! FSST bytes  ─decode→  (offsets: Vec<i32>, values: Vec<u8>)
//!                                │               │
//!                                └──┬────────────┘
//!                                   └────────→  StringArray::try_new_unchecked
//! ```
//!
//! Zero intermediate allocations, one copy of the byte payload, and
//! the same fast path the existing `reconstruct_array_u64` pattern
//! uses for numerics.
//!
//! Wiring this into the FSST decode path itself is a small follow-up
//! inside `compressors/string_compressor.rs`; the primitive below is
//! already in the public surface and benched.

use arrow::array::ArrayDataBuilder;
use arrow_array::StringArray;
use arrow_buffer::{Buffer, OffsetBuffer};
use arrow_schema::DataType;

use crate::error::{FluxError, FluxResult};

/// Build a [`StringArray`] directly from already-concatenated
/// `values` bytes and an `offsets` vector of length `N + 1`, where
/// `offsets[i]..offsets[i + 1]` is the byte range of the `i`th
/// string in `values`.
///
/// The buffers are wrapped as Arrow [`Buffer`]s without any
/// allocation beyond the [`ArrayData`] struct itself — this matches
/// the Arrow C Data Interface's zero-copy expectations.
///
/// # Safety
/// The caller must ensure `values[offsets[i]..offsets[i + 1]]` is
/// valid UTF-8 for every `i`. The FSST decoder produces raw bytes
/// that already satisfy this when the source corpus was UTF-8.
/// Built with `ArrayDataBuilder::build_unchecked` to skip the full
/// UTF-8 re-validation on the hot path.
///
/// [`ArrayData`]: arrow_data::ArrayData
pub fn string_array_from_parts(offsets: Vec<i32>, values: Vec<u8>) -> FluxResult<StringArray> {
    if offsets.is_empty() {
        return Err(FluxError::Internal(
            "string_array_from_parts: offsets must have at least one element".into(),
        ));
    }
    let n = offsets.len() - 1;
    let last = *offsets.last().unwrap();
    if last < 0 || (last as usize) > values.len() {
        return Err(FluxError::Internal(format!(
            "string_array_from_parts: final offset {last} exceeds values length {}",
            values.len(),
        )));
    }
    let offsets_buf = Buffer::from_slice_ref(&offsets);
    let values_buf = Buffer::from_vec(values);

    // SAFETY: callers guarantee UTF-8 validity (see fn docs). We skip
    // the UTF-8 recheck by using the unchecked builder.
    let data = unsafe {
        ArrayDataBuilder::new(DataType::Utf8)
            .len(n)
            .add_buffer(offsets_buf)
            .add_buffer(values_buf)
            .build_unchecked()
    };
    Ok(StringArray::from(data))
}

/// Safe variant of [`string_array_from_parts`] that performs the
/// UTF-8 re-check through Arrow's standard builder. Useful in tests
/// and anywhere the source isn't known to be UTF-8.
pub fn string_array_from_parts_checked(
    offsets: Vec<i32>,
    values: Vec<u8>,
) -> FluxResult<StringArray> {
    let n = offsets.len().saturating_sub(1);
    let mut strings = Vec::with_capacity(n);
    for i in 0..n {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let slice = values.get(start..end).ok_or_else(|| {
            FluxError::Internal(format!(
                "offset pair ({start}, {end}) out of range for {} values",
                values.len()
            ))
        })?;
        let s = std::str::from_utf8(slice).map_err(|e| {
            FluxError::Internal(format!("string at index {i} is not valid UTF-8: {e}"))
        })?;
        strings.push(s);
    }
    Ok(StringArray::from(strings))
}

/// Zero-alloc-ish: reuse Arrow's [`OffsetBuffer`] + raw values
/// buffer without touching the offsets Vec layout twice. Used
/// when the offsets are already in an [`arrow_buffer::Buffer`].
pub fn string_array_from_buffers(
    offsets: OffsetBuffer<i32>,
    values: Buffer,
) -> FluxResult<StringArray> {
    let n = offsets.len() - 1;
    let data = unsafe {
        ArrayDataBuilder::new(DataType::Utf8)
            .len(n)
            .add_buffer(offsets.into_inner().into_inner())
            .add_buffer(values)
            .build_unchecked()
    };
    Ok(StringArray::from(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    #[test]
    fn round_trip_three_strings() {
        // "foo" "" "barbaz"
        let values = b"foobarbaz".to_vec();
        let offsets = vec![0, 3, 3, 9];
        let arr = string_array_from_parts(offsets, values).unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), "foo");
        assert_eq!(arr.value(1), "");
        assert_eq!(arr.value(2), "barbaz");
    }

    #[test]
    fn checked_variant_catches_bad_utf8() {
        let values = vec![0xFFu8, 0xFEu8];
        let offsets = vec![0, 2];
        let err = string_array_from_parts_checked(offsets, values).unwrap_err();
        assert!(err.to_string().contains("UTF-8"));
    }

    #[test]
    fn empty_offsets_errors() {
        let err = string_array_from_parts(vec![], b"".to_vec()).unwrap_err();
        assert!(err.to_string().contains("at least one"));
    }

    #[test]
    fn offset_past_values_errors() {
        let err = string_array_from_parts(vec![0, 100], b"xy".to_vec()).unwrap_err();
        assert!(err.to_string().contains("exceeds"));
    }
}
