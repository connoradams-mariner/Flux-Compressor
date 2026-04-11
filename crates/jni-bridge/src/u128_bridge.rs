// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Dual-register u128 ↔ JVM `long` pair conversion.
//!
//! ## Why Two `long`s?
//! The JVM's largest primitive integer is 64 bits (`long`).  FluxCompress
//! handles u128 aggregation results (e.g., `SUM(bigint_col)` over billions of
//! rows).  We represent these on the Java side as a `Struct(long hi, long lo)`.
//!
//! Java `long` is signed, but the bit pattern is identical to Rust `i64` /
//! `u64`.  We interpret the sign bit as a data bit — no precision is lost.
//!
//! ## Reconstruction
//! ```text
//! u128 = (hi as u64 as u128) << 64 | (lo as u64 as u128)
//! ```

/// Convert a pair of JVM `long` values (signed i64) into a Rust `u128`.
///
/// `hi` holds bits 127..64, `lo` holds bits 63..0.
#[inline]
pub fn jlong_pair_to_u128(hi: i64, lo: i64) -> u128 {
    let hi_u = hi as u64 as u128;
    let lo_u = lo as u64 as u128;
    (hi_u << 64) | lo_u
}

/// Decompose a Rust `u128` into a pair of JVM `long` values.
///
/// Returns `(hi, lo)` where `hi` is bits 127..64 and `lo` is bits 63..0.
/// Both are cast to `i64` — the bit patterns are preserved unchanged.
#[inline]
pub fn u128_to_jlong_pair(value: u128) -> (i64, i64) {
    let hi = (value >> 64) as u64 as i64;
    let lo = value as u64 as i64;
    (hi, lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero() {
        let (hi, lo) = u128_to_jlong_pair(0);
        assert_eq!(jlong_pair_to_u128(hi, lo), 0u128);
    }

    #[test]
    fn u128_max() {
        let v = u128::MAX;
        let (hi, lo) = u128_to_jlong_pair(v);
        assert_eq!(jlong_pair_to_u128(hi, lo), v);
    }

    #[test]
    fn known_value() {
        // 0x0000_0001_0000_0000_0000_0000_0000_0001
        let v: u128 = (1u128 << 64) | 1;
        let (hi, lo) = u128_to_jlong_pair(v);
        assert_eq!(hi, 1i64);
        assert_eq!(lo, 1i64);
        assert_eq!(jlong_pair_to_u128(hi, lo), v);
    }

    #[test]
    fn high_bits_set() {
        // Value where high word has the sign bit set (tests signed→unsigned cast).
        let v: u128 = (0xFFFF_FFFF_FFFF_FFFFu128 << 64) | 0xDEAD_BEEF_CAFE_BABEu128;
        let (hi, lo) = u128_to_jlong_pair(v);
        assert_eq!(jlong_pair_to_u128(hi, lo), v);
    }

    #[test]
    fn round_trip_sweep() {
        for v in [
            0u128,
            1,
            u64::MAX as u128,
            u128::MAX / 2,
            u128::MAX - 1,
            u128::MAX,
            0xDEAD_BEEF_0000_0000_CAFE_BABE_1234_5678,
        ] {
            let (hi, lo) = u128_to_jlong_pair(v);
            assert_eq!(jlong_pair_to_u128(hi, lo), v, "round-trip failed for {v:#034x}");
        }
    }
}
