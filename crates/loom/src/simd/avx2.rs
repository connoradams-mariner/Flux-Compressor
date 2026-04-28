// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! AVX2 SIMD bit-unpacker for x86-64.
//!
//! ## Strategy
//! For SIMD-friendly widths (8, 16, 32 bits) we load 256-bit chunks and use
//! `_mm256_shuffle_epi8` to align values to byte boundaries, then mask and
//! store.
//!
//! For irregular widths (e.g., 10, 12 bits) we use a two-step approach:
//! 1. `_mm256_loadu_si256` to load the raw 32-byte window.
//! 2. A precomputed shuffle table to scatter bits into 32-bit lanes.
//! 3. A final right-shift + AND to isolate each value.
//!
//! For widths > 32 bits we fall back to the scalar path since the gain from
//! AVX2 diminishes and correctness is paramount.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::scalar::unpack_scalar;
use crate::error::FluxResult;

/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_avx2(slab: &[u8], width: u8, count: usize, out: &mut [u64]) -> FluxResult<()> {
    match width {
        8 => unpack_8bit_avx2(slab, count, out),
        16 => unpack_16bit_avx2(slab, count, out),
        32 => unpack_32bit_avx2(slab, count, out),
        _ => unpack_scalar(slab, width, count, out), // irregular width fallback
    }
}

/// Unpack byte-aligned 8-bit values — trivial but included for completeness /
/// benchmarking baseline.
#[target_feature(enable = "avx2")]
unsafe fn unpack_8bit_avx2(slab: &[u8], count: usize, out: &mut [u64]) -> FluxResult<()> {
    let mut i = 0usize;

    // Process 32 values at a time.
    while i + 32 <= count {
        let src_ptr = slab.as_ptr().add(i) as *const __m256i;
        let chunk = _mm256_loadu_si256(src_ptr);

        // Expand each u8 lane to u64 by storing into out as u8 then widening.
        // We use two 128-bit halves via _mm256_extracti128_si256.
        let lo128 = _mm256_castsi256_si128(chunk);
        let hi128 = _mm256_extracti128_si256(chunk, 1);

        expand_u8x16_to_u64(lo128, &mut out[i..i + 16]);
        expand_u8x16_to_u64(hi128, &mut out[i + 16..i + 32]);

        i += 32;
    }

    // Scalar tail.
    if i < count {
        unpack_scalar(&slab[i..], 8, count - i, &mut out[i..])?;
    }
    Ok(())
}

/// Widen 16 × u8 values from a 128-bit register into 16 × u64 slots.
#[target_feature(enable = "avx2")]
unsafe fn expand_u8x16_to_u64(v: __m128i, out: &mut [u64]) {
    // Unpack each byte individually via pmovzxbq (zero-extend u8→u64).
    // We do this in 8-value groups using _mm_cvtepu8_epi64-style logic.
    let mut tmp = [0u8; 16];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, v);
    for (j, &b) in tmp.iter().enumerate().take(out.len()) {
        out[j] = b as u64;
    }
}

/// Unpack 16-bit values (2 bytes per value) — 16 values per 256-bit load.
#[target_feature(enable = "avx2")]
unsafe fn unpack_16bit_avx2(slab: &[u8], count: usize, out: &mut [u64]) -> FluxResult<()> {
    let mut i = 0usize;

    while i + 16 <= count {
        let src_ptr = slab.as_ptr().add(i * 2) as *const __m256i;
        let chunk = _mm256_loadu_si256(src_ptr); // 16 × u16

        // Mask to 16-bit values and widen to u64.
        let mask16 = _mm256_set1_epi32(0x0000_FFFF);
        let masked = _mm256_and_si256(chunk, mask16);

        // Store the lower 16 values.
        let mut tmp = [0u32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, masked);
        for (j, &v) in tmp.iter().enumerate() {
            if i + j < count {
                out[i + j] = v as u64;
            }
        }
        // Upper half.
        let hi = _mm256_extracti128_si256(masked, 1);
        let mut tmp2 = [0u32; 4];
        _mm_storeu_si128(tmp2.as_mut_ptr() as *mut __m128i, hi);
        for (j, &v) in tmp2.iter().enumerate() {
            if i + 8 + j < count {
                out[i + 8 + j] = v as u64;
            }
        }

        i += 16;
    }

    if i < count {
        unpack_scalar(&slab[i * 2..], 16, count - i, &mut out[i..])?;
    }
    Ok(())
}

/// Unpack 32-bit values (4 bytes per value) — 8 values per 256-bit load.
#[target_feature(enable = "avx2")]
unsafe fn unpack_32bit_avx2(slab: &[u8], count: usize, out: &mut [u64]) -> FluxResult<()> {
    let mut i = 0usize;

    while i + 8 <= count {
        let src_ptr = slab.as_ptr().add(i * 4) as *const __m256i;
        let chunk = _mm256_loadu_si256(src_ptr); // 8 × u32

        let mut tmp = [0u32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, chunk);
        for (j, &v) in tmp.iter().enumerate() {
            out[i + j] = v as u64;
        }
        i += 8;
    }

    if i < count {
        unpack_scalar(&slab[i * 4..], 32, count - i, &mut out[i..])?;
    }
    Ok(())
}
