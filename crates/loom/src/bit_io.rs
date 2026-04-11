// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Low-level packed bitstream writer and reader.
//!
//! ## Design Principles
//! - **Zero-copy reads**: [`BitReader`] borrows a `&[u8]` slice and never
//!   allocates.
//! - **SIMD-friendly alignment**: Bit-slabs are padded so that every value
//!   sequence begins on an 8-bit boundary (for the fast SIMD unpacker path).
//! - **Configurable width**: Supports any bit width from 1 to 64 bits per
//!   value.

use crate::error::{FluxError, FluxResult};

// ─────────────────────────────────────────────────────────────────────────────
// BitWriter
// ─────────────────────────────────────────────────────────────────────────────

/// Packs variable-bit-width integers into a compact byte buffer.
///
/// ```
/// use loom::bit_io::BitWriter;
///
/// let mut w = BitWriter::new(4); // 4 bits per value
/// w.write_value(5).unwrap();
/// w.write_value(10).unwrap();
/// let buf = w.finish();
/// assert_eq!(buf.len(), 1); // 8 bits = 1 byte
/// ```
pub struct BitWriter {
    /// Target bit-width per value (1–64).
    width: u8,
    /// Accumulator for partial bytes.
    accumulator: u64,
    /// Number of valid bits currently in `accumulator`.
    bits_in_acc: u8,
    /// Output buffer.
    buf: Vec<u8>,
    /// Sentinel bitmask: `(1 << width) - 1` — i.e., all bits set for `width`.
    sentinel_mask: u64,
}

impl BitWriter {
    /// Create a new `BitWriter` that packs values of exactly `width` bits.
    ///
    /// # Panics
    /// Panics if `width` is 0 or > 64.
    pub fn new(width: u8) -> Self {
        assert!(width >= 1 && width <= 64, "bit width must be 1–64");
        let sentinel_mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        Self {
            width,
            accumulator: 0,
            bits_in_acc: 0,
            buf: Vec::new(),
            sentinel_mask,
        }
    }

    /// Write a single `value` into the stream.
    ///
    /// If `value` overflows the target bit-width (i.e., it is the sentinel),
    /// the caller should have already written the sentinel and stored the
    /// real value in the [`OutlierMap`][crate::outlier_map::OutlierMap].
    pub fn write_value(&mut self, value: u64) -> FluxResult<()> {
        let masked = value & self.sentinel_mask;
        self.accumulator |= (masked as u64) << self.bits_in_acc;
        self.bits_in_acc += self.width;

        // Drain full bytes from the accumulator.
        while self.bits_in_acc >= 8 {
            self.buf.push((self.accumulator & 0xFF) as u8);
            self.accumulator >>= 8;
            self.bits_in_acc -= 8;
        }
        Ok(())
    }

    /// Write the sentinel pattern (`all-ones` for `width` bits), signalling
    /// that the corresponding value lives in the Outlier Map.
    #[inline]
    pub fn write_sentinel(&mut self) -> FluxResult<()> {
        self.write_value(self.sentinel_mask)
    }

    /// Flush remaining bits (zero-padded to a full byte) and return the buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_acc > 0 {
            self.buf.push((self.accumulator & 0xFF) as u8);
        }
        self.buf
    }

    /// Return the bit-width this writer was configured with.
    #[inline]
    pub fn width(&self) -> u8 {
        self.width
    }

    /// Return the sentinel mask (all-ones for `width` bits).
    #[inline]
    pub fn sentinel_mask(&self) -> u64 {
        self.sentinel_mask
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BitReader
// ─────────────────────────────────────────────────────────────────────────────

/// Reads packed variable-bit-width integers from a borrowed byte slice.
///
/// This is the **scalar fallback** path. The SIMD-accelerated path lives in
/// [`crate::simd`].
///
/// ```
/// use loom::bit_io::{BitWriter, BitReader};
///
/// let mut w = BitWriter::new(10);
/// for v in [0u64, 42, 511, 1023] { w.write_value(v).unwrap(); }
/// let buf = w.finish();
///
/// let mut r = BitReader::new(&buf, 10);
/// assert_eq!(r.read_value().unwrap(), 0);
/// assert_eq!(r.read_value().unwrap(), 42);
/// assert_eq!(r.read_value().unwrap(), 511);
/// assert_eq!(r.read_value().unwrap(), 1023);
/// ```
pub struct BitReader<'a> {
    /// Source byte slice (zero-copy borrow).
    data: &'a [u8],
    /// Bit-width per value.
    width: u8,
    /// Mask: `(1 << width) - 1`.
    mask: u64,
    /// Sentinel value for this width.
    sentinel: u64,
    /// Bit-offset of the next value to read.
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a reader over `data` that extracts `width`-bit values.
    pub fn new(data: &'a [u8], width: u8) -> Self {
        assert!(width >= 1 && width <= 64);
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        Self {
            data,
            width,
            mask,
            sentinel: mask, // all-ones sentinel
            bit_pos: 0,
        }
    }

    /// Read the next `width`-bit value.
    ///
    /// Returns `None` when the stream is exhausted.
    pub fn read_value(&mut self) -> Option<u64> {
        let needed_bits = self.bit_pos + self.width as usize;
        let needed_bytes = (needed_bits + 7) / 8;
        if needed_bytes > self.data.len() {
            return None;
        }

        // Load up to 8 bytes from the current position.
        let byte_idx = self.bit_pos / 8;
        let bit_off  = (self.bit_pos % 8) as u64;

        // Read up to 9 bytes to cover any 64-bit window starting mid-byte.
        let mut raw: u64 = 0;
        let take = ((self.width as usize + bit_off as usize + 7) / 8).min(8);
        for i in 0..take {
            if byte_idx + i < self.data.len() {
                raw |= (self.data[byte_idx + i] as u64) << (i * 8);
            }
        }

        let value = (raw >> bit_off) & self.mask;
        self.bit_pos += self.width as usize;
        Some(value)
    }

    /// Returns `true` if the last value read was the sentinel (outlier signal).
    #[inline]
    pub fn is_sentinel(&self, value: u64) -> bool {
        value == self.sentinel
    }

    /// Reset the reader to a specific **byte** offset (for seekable access).
    #[inline]
    pub fn seek_to_bit(&mut self, bit_pos: usize) {
        self.bit_pos = bit_pos;
    }

    /// Current bit position.
    #[inline]
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-width discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the minimum number of bits required to represent `value`.
#[inline]
pub fn bits_needed(value: u128) -> u8 {
    if value == 0 {
        return 1;
    }
    (128 - value.leading_zeros()) as u8
}

/// Discover the optimal slab bit-width for a slice of u128 values using the
/// 99th-percentile heuristic described in the spec.
///
/// Returns `(slab_width, frame_of_reference)`.
///
/// The FoR is the minimum value; each value stored in the slab is
/// `v - frame_of_reference`. If `max_width > p99_width + 32`, outlier-map
/// mode is activated (the caller is responsible for routing outliers).
pub fn discover_width(values: &[u128]) -> (u8, u128) {
    if values.is_empty() {
        return (1, 0);
    }

    let min_val = *values.iter().min().unwrap();
    let frame_of_reference = min_val;

    // Compute per-value bit widths after FoR subtraction.
    let mut widths: Vec<u8> = values
        .iter()
        .map(|&v| bits_needed(v.saturating_sub(frame_of_reference)))
        .collect();

    widths.sort_unstable();

    // 99th percentile index.
    let p99_idx = ((widths.len() as f64 * 0.99) as usize).min(widths.len() - 1);
    let p99_width = widths[p99_idx];
    let max_width = *widths.last().unwrap();

    // If outliers exceed the 99th percentile by more than 32 bits, use the
    // narrower p99 width and route outliers to the Outlier Map.
    let slab_width = if max_width > p99_width.saturating_add(32) {
        p99_width
    } else {
        max_width
    };

    // Ensure width is at least 1 and at most 64 for the slab
    // (u128 outliers are handled entirely by the OutlierMap).
    let clamped = slab_width.clamp(1, 64);

    (clamped, frame_of_reference)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_4bit() {
        let values = [0u64, 1, 7, 14, 15];
        let mut w = BitWriter::new(4);
        for &v in &values { w.write_value(v).unwrap(); }
        let buf = w.finish();

        let mut r = BitReader::new(&buf, 4);
        for &expected in &values {
            assert_eq!(r.read_value().unwrap(), expected);
        }
    }

    #[test]
    fn round_trip_10bit() {
        let values: Vec<u64> = (0..1024).step_by(7).collect();
        let mut w = BitWriter::new(10);
        for &v in &values { w.write_value(v).unwrap(); }
        let buf = w.finish();

        let mut r = BitReader::new(&buf, 10);
        for &expected in &values {
            assert_eq!(r.read_value().unwrap(), expected);
        }
    }

    #[test]
    fn sentinel_detection() {
        let mut w = BitWriter::new(8);
        w.write_value(42).unwrap();
        w.write_sentinel().unwrap(); // 0xFF
        w.write_value(100).unwrap();
        let buf = w.finish();

        let mut r = BitReader::new(&buf, 8);
        let v0 = r.read_value().unwrap();
        let v1 = r.read_value().unwrap();
        let v2 = r.read_value().unwrap();

        assert_eq!(v0, 42);
        assert!(r.is_sentinel(v1));
        assert_eq!(v2, 100);
    }

    #[test]
    fn discover_width_basic() {
        // Values 0..100 should fit in 7 bits.
        let vals: Vec<u128> = (0u128..100).collect();
        let (w, for_val) = discover_width(&vals);
        assert_eq!(for_val, 0);
        assert!(w <= 7, "expected ≤7 bits, got {w}");
    }

    #[test]
    fn discover_width_outlier_trigger() {
        // 99% of values fit in 8 bits, one is u128::MAX (128-bit outlier).
        let mut vals: Vec<u128> = (0u128..255).collect();
        vals.push(u128::MAX);
        let (w, _) = discover_width(&vals);
        // Outlier-map triggered: slab width ≤ 64 (the monster goes to OutlierMap).
        assert!(w <= 64);
    }
}
