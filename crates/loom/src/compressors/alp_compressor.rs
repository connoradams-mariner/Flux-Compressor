// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Adaptive Lossless floating-Point (ALP) compressor.
//!
//! Many real-world Float64 / Float32 columns aren't truly arbitrary doubles —
//! they're decimal numbers that happen to be stored as IEEE 754. Examples:
//! prices ($19.99), latitudes/longitudes (40.712800), CPU usage (0.85),
//! integer counts that got widened to f64 by CSV parsers, etc.
//!
//! For these cases ALP picks a small decimal exponent `e ∈ [0, 18]` such that
//! every value `v` satisfies `round(v · 10^e) / 10^e == v`. The integer
//! mantissas are then compressed via the normal Loom pipeline (BitSlab /
//! DeltaDelta), which crushes them because the magnitudes are orders of
//! magnitude smaller than the IEEE 754 bit pattern.
//!
//! Values that don't round-trip exactly (NaN/Inf, very large magnitudes,
//! arbitrary doubles) are stored verbatim in a sparse outlier map. If more
//! than 2% of rows are outliers, ALP gives up and the caller falls back to
//! the standard u64-bitcast path.
//!
//! ## Block Layout (TAG = 0x09)
//! ```text
//! [u8: TAG = 0x09]
//! [u8: secondary_codec = 0]   // outer secondary unused; mantissa block has its own
//! [u32: value_count]
//! [u8:  exp]                  // 0..18 for f64, 0..9 for f32
//! [u8:  dtype]                // 0 = Float64, 1 = Float32
//! [u32: outlier_count]
//! [outlier_count × (u32 row_index, u64 raw_bits)]
//! [u32: mantissa_block_len]
//! [mantissa_block bytes]      // standard Loom block (BitSlab / DeltaDelta)
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Write};

use crate::error::{FluxError, FluxResult};
use crate::compressors::{bit_slab_compressor, delta_compressor};

/// Block tag for ALP.
pub const TAG: u8 = 0x09;

/// Maximum decimal exponent we try for f64.
const ALP_MAX_EXP_F64: u8 = 18;
/// Maximum decimal exponent we try for f32 (limited by single-precision mantissa).
const ALP_MAX_EXP_F32: u8 = 9;
/// Number of values to probe when finding the best exponent.
const ALP_PROBE: usize = 256;
/// Maximum outlier fraction before ALP gives up and the caller falls back.
const ALP_MAX_OUTLIERS_FRAC: f64 = 0.02;

/// Try to ALP-encode `values` as Float64. Returns `Ok(Some(block))` on success,
/// `Ok(None)` if ALP isn't profitable for this data.
pub fn try_compress_f64(values: &[f64]) -> FluxResult<Option<Vec<u8>>> {
    try_compress_inner(values, false)
}

/// Try to ALP-encode `values` as Float32 (caller passes f64 widening of f32).
pub fn try_compress_f32(values: &[f64]) -> FluxResult<Option<Vec<u8>>> {
    try_compress_inner(values, true)
}

fn try_compress_inner(values: &[f64], is_f32: bool) -> FluxResult<Option<Vec<u8>>> {
    if values.len() < 32 {
        return Ok(None);
    }
    let max_exp = if is_f32 { ALP_MAX_EXP_F32 } else { ALP_MAX_EXP_F64 };
    let exp = match find_best_exp(values, max_exp) {
        Some(e) => e,
        None => return Ok(None),
    };

    let scale = 10f64.powi(exp as i32);
    let mut mantissas: Vec<i64> = Vec::with_capacity(values.len());
    let mut outliers: Vec<(u32, u64)> = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            outliers.push((i as u32, v.to_bits()));
            mantissas.push(0);
            continue;
        }
        let scaled = v * scale;
        if scaled.abs() >= (i64::MAX as f64) {
            outliers.push((i as u32, v.to_bits()));
            mantissas.push(0);
            continue;
        }
        let m = scaled.round() as i64;
        // Round-trip check; division by `scale` is more numerically stable
        // than multiplication by 10^-e (which compounds rounding error).
        let recovered = (m as f64) / scale;
        if recovered.to_bits() == v.to_bits() {
            mantissas.push(m);
        } else {
            outliers.push((i as u32, v.to_bits()));
            mantissas.push(0);
        }
    }

    let outlier_frac = (outliers.len() as f64) / (values.len() as f64);
    if outlier_frac > ALP_MAX_OUTLIERS_FRAC {
        return Ok(None);
    }

    // Compress mantissas via the normal Loom pipeline. We pick BitSlab vs
    // DeltaDelta based on a quick classifier probe.
    let mantissas_u128: Vec<u128> = mantissas.iter().map(|&m| m as u64 as u128).collect();
    use crate::loom_classifier::{classify, LoomStrategy};
    let cls = classify(&mantissas_u128);
    let inner = match cls.strategy {
        LoomStrategy::DeltaDelta if mantissas_u128.len() >= 2 =>
            delta_compressor::compress(&mantissas_u128)?,
        _ => bit_slab_compressor::compress(&mantissas_u128)?,
    };

    // Profitability check: ALP block must beat the trivial 8-byte-per-value
    // raw layout by at least 5%, otherwise fall back so we don't pay the
    // header cost on already-incompressible data.
    let raw_size = values.len() * 8;
    let est_alp_size = 14 + outliers.len() * 12 + inner.len();
    if est_alp_size as f64 > (raw_size as f64) * 0.95 {
        return Ok(None);
    }

    let mut buf = Vec::with_capacity(est_alp_size);
    buf.write_u8(TAG)?;
    buf.write_u8(0)?; // outer secondary codec = None
    buf.write_u32::<LittleEndian>(values.len() as u32)?;
    buf.write_u8(exp)?;
    buf.write_u8(if is_f32 { 1 } else { 0 })?;
    buf.write_u32::<LittleEndian>(outliers.len() as u32)?;
    for &(i, b) in &outliers {
        buf.write_u32::<LittleEndian>(i)?;
        buf.write_u64::<LittleEndian>(b)?;
    }
    buf.write_u32::<LittleEndian>(inner.len() as u32)?;
    buf.extend_from_slice(&inner);
    Ok(Some(buf))
}

/// Find the smallest decimal exponent `e ∈ [0, max_exp]` such that every probe
/// value round-trips exactly through `round(v · 10^e) / 10^e`. Returns `None`
/// if no such exponent exists (data isn't decimal-representable).
fn find_best_exp(values: &[f64], max_exp: u8) -> Option<u8> {
    let probe_n = values.len().min(ALP_PROBE);
    let probe = &values[..probe_n];
    'next_exp: for exp in 0u8..=max_exp {
        let scale = 10f64.powi(exp as i32);
        let mut had_finite = false;
        for &v in probe {
            if !v.is_finite() {
                continue;
            }
            had_finite = true;
            let scaled = v * scale;
            if scaled.abs() >= (i64::MAX as f64) {
                continue 'next_exp;
            }
            let m = scaled.round() as i64;
            let recovered = (m as f64) / scale;
            if recovered.to_bits() != v.to_bits() {
                continue 'next_exp;
            }
        }
        if had_finite {
            return Some(exp);
        }
    }
    None
}

/// Decompress an ALP block. Returns `(f64-bits-as-u128, bytes_consumed)`.
/// The returned values are `f64::to_bits()` cast to u128 (or for f32 columns,
/// `f32::to_bits()` zero-extended to u128) so they slot into the standard
/// numeric reader pipeline that interprets the bits at column-build time.
pub fn decompress(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    if data.is_empty() || data[0] != TAG {
        return Err(FluxError::InvalidFile("not an ALP block".into()));
    }
    let mut cur = Cursor::new(data);
    let _tag = cur.read_u8()?;
    let _codec = cur.read_u8()?;
    let count = cur.read_u32::<LittleEndian>()? as usize;
    let exp = cur.read_u8()?;
    let is_f32 = cur.read_u8()? != 0;
    let outlier_count = cur.read_u32::<LittleEndian>()? as usize;

    let mut outliers: Vec<(usize, u64)> = Vec::with_capacity(outlier_count);
    for _ in 0..outlier_count {
        let i = cur.read_u32::<LittleEndian>()? as usize;
        let b = cur.read_u64::<LittleEndian>()?;
        outliers.push((i, b));
    }

    let inner_len = cur.read_u32::<LittleEndian>()? as usize;
    let inner_start = cur.position() as usize;
    let inner = &data[inner_start..inner_start + inner_len];

    use crate::decompressors::block_reader::decompress_block;
    let (mantissas_u128, _) = decompress_block(inner)?;
    if mantissas_u128.len() != count {
        return Err(FluxError::InvalidFile(format!(
            "ALP mantissa count {} != header {}", mantissas_u128.len(), count,
        )));
    }

    let scale = 10f64.powi(exp as i32);
    let mut out: Vec<u128> = Vec::with_capacity(count);
    for m_u128 in &mantissas_u128 {
        let m = *m_u128 as u64 as i64;
        let v = (m as f64) / scale;
        if is_f32 {
            out.push((v as f32).to_bits() as u64 as u128);
        } else {
            out.push(v.to_bits() as u128);
        }
    }
    // Patch outliers with their verbatim bits.
    for (i, b) in outliers {
        if i < out.len() {
            out[i] = if is_f32 { (b as u32) as u64 as u128 } else { b as u128 };
        }
    }

    Ok((out, inner_start + inner_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alp_round_trip_two_decimal_prices() {
        // Prices like $19.99, $123.45 — the canonical ALP win case. We
        // construct each value as cents/100 so the f64 bit pattern matches
        // the ALP reconstruction formula exactly (same as parsing "X.YY"
        // from CSV).
        let values: Vec<f64> = (0..2000)
            .map(|i| ((199 + i) as f64) / 100.0)
            .collect();
        let block = try_compress_f64(&values).unwrap().expect("ALP should fire");
        assert_eq!(block[0], TAG);
        let raw_size = values.len() * 8;
        assert!(block.len() < raw_size / 2,
            "expected >2x shrink, got {} → {}", raw_size, block.len());
        let (decoded, _) = decompress(&block).unwrap();
        for (i, &orig) in values.iter().enumerate() {
            let got = f64::from_bits(decoded[i] as u64);
            assert_eq!(got.to_bits(), orig.to_bits(),
                "row {i}: {orig} vs {got}");
        }
    }

    #[test]
    fn alp_round_trip_lat_lon() {
        // 6-decimal coordinates, constructed via integer-div-by-power-of-10
        // so the f64 bits match the ALP reconstruction exactly.
        let values: Vec<f64> = (0..1500)
            .map(|i| ((40_000_000 + i) as f64) / 1_000_000.0)
            .collect();
        let block = try_compress_f64(&values).unwrap().expect("ALP should fire");
        let (decoded, _) = decompress(&block).unwrap();
        for (i, &orig) in values.iter().enumerate() {
            let got = f64::from_bits(decoded[i] as u64);
            assert_eq!(got.to_bits(), orig.to_bits(),
                "row {i}: {orig} vs {got}");
        }
    }

    #[test]
    fn alp_round_trip_integer_floats() {
        // Counts stored as Float64 — extremely common in CSV.
        let values: Vec<f64> = (0..3000).map(|i| i as f64).collect();
        let block = try_compress_f64(&values).unwrap().expect("ALP should fire on integer floats");
        let raw_size = values.len() * 8;
        assert!(block.len() < raw_size / 4,
            "expected >4x shrink, got {} → {}", raw_size, block.len());
        let (decoded, _) = decompress(&block).unwrap();
        for (i, &orig) in values.iter().enumerate() {
            assert_eq!(f64::from_bits(decoded[i] as u64).to_bits(), orig.to_bits());
        }
    }

    #[test]
    fn alp_falls_back_on_random_doubles() {
        // Truly random-bit doubles aren't decimal-representable.
        let mut s: u64 = 0xDEADBEEF;
        let values: Vec<f64> = (0..500).map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            f64::from_bits(s)
        }).collect();
        // Filter NaNs that might break round-trip.
        let values: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
        let block = try_compress_f64(&values).unwrap();
        // Random doubles → no exponent works → ALP returns None.
        assert!(block.is_none() || block.unwrap().len() < values.len() * 8);
    }
}
