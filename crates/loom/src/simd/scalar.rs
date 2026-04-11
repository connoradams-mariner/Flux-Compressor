// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Portable scalar bit-unpacker.  Used as a fallback on all platforms and
//! as the correctness reference for SIMD implementations.

use crate::error::FluxResult;

/// Unpack `count` packed `width`-bit values from `slab` into `out`.
pub fn unpack_scalar(
    slab: &[u8],
    width: u8,
    count: usize,
    out: &mut [u64],
) -> FluxResult<()> {
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let mut bit_pos = 0usize;

    for slot in out.iter_mut().take(count) {
        let byte_idx = bit_pos / 8;
        let bit_off = (bit_pos % 8) as u64;

        // Load up to 8 bytes (safe even near end of slice).
        let mut raw = 0u64;
        let take = ((width as usize + bit_off as usize + 7) / 8).min(8);
        for i in 0..take {
            if byte_idx + i < slab.len() {
                raw |= (slab[byte_idx + i] as u64) << (i * 8);
            }
        }

        *slot = (raw >> bit_off) & mask;
        bit_pos += width as usize;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_io::BitWriter;

    #[test]
    fn scalar_round_trip() {
        for width in [1u8, 4, 7, 10, 16, 32, 64] {
            let max_val = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
            let values: Vec<u64> = (0..=max_val.min(255)).collect();

            let mut w = BitWriter::new(width);
            for &v in &values { w.write_value(v).unwrap(); }
            let buf = w.finish();

            let mut out = vec![0u64; values.len()];
            unpack_scalar(&buf, width, values.len(), &mut out).unwrap();
            assert_eq!(out, values, "failed at width={width}");
        }
    }
}
