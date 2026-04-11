// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # SIMD Unpacker (Sprint 3)
//!
//! Dispatches to the fastest available SIMD bit-unpacking implementation:
//! - **AVX2** (x86-64, 256-bit lanes) — `_mm256_shuffle_epi8` shuffles
//! - **NEON** (AArch64, 128-bit lanes) — `vqtbl1q_u8` table-lookups
//! - **Scalar** fallback — portable, used on all other targets
//!
//! ## Zero-Copy Contract
//! The unpacker reads from a borrowed `&[u8]` slice (the raw slab bytes) and
//! writes decoded values into a caller-supplied `&mut [u64]` output buffer.
//! No heap allocation occurs on the hot path.

pub mod scalar;

#[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
pub mod avx2;

#[cfg(all(target_arch = "aarch64", feature = "simd-neon"))]
pub mod neon;

use crate::error::FluxResult;

/// Unpack `count` values of `width` bits from `slab` into `out`.
///
/// Automatically selects the fastest available SIMD implementation at
/// compile time.  Panics if `out.len() < count`.
pub fn unpack(slab: &[u8], width: u8, count: usize, out: &mut [u64]) -> FluxResult<()> {
    assert!(out.len() >= count, "output buffer too small");

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::unpack_avx2(slab, width, count, out) };
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd-neon"))]
    {
        return unsafe { neon::unpack_neon(slab, width, count, out) };
    }

    // Scalar fallback (always compiled in).
    scalar::unpack_scalar(slab, width, count, out)
}
