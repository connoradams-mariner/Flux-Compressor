// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Loom Adaptive Classifier
//!
//! Deterministic heuristic waterfall that chooses the best compression
//! strategy for a 1 024-row segment.  No ML, no randomness — pure stats.
//!
//! ## Waterfall Order
//! 1. **RLE** — entropy ≈ 0 (all values identical / constant run)
//! 2. **Delta-Delta** — first-order differences are constant or near-Gaussian
//! 3. **Dictionary** — cardinality ratio < 5 %
//! 4. **BitSlab** — numeric range fits in ≤ 64 bits (outliers → OutlierMap)
//! 5. **SimdLz4** — fallback for high-entropy / string data

use crate::bit_io::bits_needed;

/// The compression strategy selected by the Loom classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum LoomStrategy {
    /// Run-Length Encoding — optimal for constant / near-constant columns.
    Rle         = 0x0001,
    /// Delta-Delta Encoding — optimal for sorted or monotone sequences.
    DeltaDelta  = 0x0002,
    /// Dictionary Encoding — optimal for low-cardinality string/enum columns.
    Dictionary  = 0x0003,
    /// Bit-Slab with optional Outlier Map — optimal for numeric range data.
    BitSlab     = 0x0004,
    /// SIMD-accelerated LZ4 — fallback for high-entropy data.
    SimdLz4     = 0x0005,
}

/// Bit flag in the strategy_mask u16 indicating u64-only mode.
/// When set, the block contains no u128 values and the decompressor
/// can decode directly into `Vec<u64>` without u128 intermediary.
pub const U64_ONLY_FLAG: u16 = 0x0100; // bit 8

impl LoomStrategy {
    /// Decode a strategy from its 2-byte strategy mask (low byte only).
    pub fn from_u16(v: u16) -> Option<Self> {
        match v & 0x00FF {
            0x0001 => Some(Self::Rle),
            0x0002 => Some(Self::DeltaDelta),
            0x0003 => Some(Self::Dictionary),
            0x0004 => Some(Self::BitSlab),
            0x0005 => Some(Self::SimdLz4),
            _ => None,
        }
    }

    /// Return the 2-byte strategy mask for the Atlas footer.
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    /// Encode strategy + u64_only flag into the strategy_mask.
    pub fn encode_mask(self, u64_only: bool) -> u16 {
        let base = self as u16;
        if u64_only { base | U64_ONLY_FLAG } else { base }
    }

    /// Check if the u64_only flag is set in a strategy mask.
    pub fn is_u64_only(mask: u16) -> bool {
        mask & U64_ONLY_FLAG != 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classification result
// ─────────────────────────────────────────────────────────────────────────────

/// Detailed output from the Loom classifier.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The chosen compression strategy.
    pub strategy: LoomStrategy,
    /// Bit-entropy of the sample (0.0 = constant, 1.0 = uniform).
    pub entropy: f64,
    /// Fraction of unique values (cardinality / count).
    pub cardinality_ratio: f64,
    /// 99th-percentile bit-width after Frame-of-Reference subtraction.
    pub p99_bit_width: u8,
    /// Maximum bit-width in the segment.
    pub max_bit_width: u8,
    /// Whether the outlier map should be activated.
    pub use_outlier_map: bool,
    /// Frame-of-Reference (min value), used for BitSlab.
    pub frame_of_reference: u128,
    /// Recommended slab bit-width.
    pub slab_width: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main classify() entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Loom waterfall classifier on a segment of `u128` values.
///
/// This is the primary decision function.  Call it on each 1 024-row segment
/// before compression.
///
/// # Arguments
/// * `values` — raw numeric values for this segment (up to `SEGMENT_SIZE`).
///
/// # Returns
/// A [`ClassificationResult`] describing which strategy to use and associated
/// parameters.
pub fn classify(values: &[u128]) -> ClassificationResult {
    if values.is_empty() {
        return ClassificationResult {
            strategy: LoomStrategy::Rle,
            entropy: 0.0,
            cardinality_ratio: 0.0,
            p99_bit_width: 1,
            max_bit_width: 1,
            use_outlier_map: false,
            frame_of_reference: 0,
            slab_width: 1,
        };
    }

    // ── Step 1: Entropy audit ────────────────────────────────────────
    let entropy = bit_entropy(values);

    // `bit_entropy` bins on the top 8 *occupied* bits, which drives entropy
    // near 0 for monotonic sequences whose range stays within a single
    // top-byte bucket (e.g. 65K offsets growing by ~14). RLE on those is
    // catastrophic. Guard the RLE path with an endpoint-equality sanity
    // check: for a truly constant column, first == mid == last.
    let endpoints_equal = values.len() < 3
        || (values[0] == values[values.len() / 2]
            && values[0] == values[values.len() - 1]);

    if entropy < 1e-9 && endpoints_equal {
        // All values identical → RLE.
        return ClassificationResult {
            strategy: LoomStrategy::Rle,
            entropy,
            cardinality_ratio: 1.0 / values.len() as f64,
            p99_bit_width: 1,
            max_bit_width: 1,
            use_outlier_map: false,
            frame_of_reference: values[0],
            slab_width: 1,
        };
    }

    // ── Step 2: Linearity / Delta-Delta check ────────────────────────────────
    if is_delta_stable(values) {
        let for_val = values[0];
        return ClassificationResult {
            strategy: LoomStrategy::DeltaDelta,
            entropy,
            cardinality_ratio: cardinality_ratio(values),
            p99_bit_width: 16,
            max_bit_width: 16,
            use_outlier_map: false,
            frame_of_reference: for_val,
            slab_width: 16,
        };
    }

    // ── Step 3: Cardinality threshold ────────────────────────────────────────
    let card_ratio = cardinality_ratio(values);
    if card_ratio < 0.05 {
        return ClassificationResult {
            strategy: LoomStrategy::Dictionary,
            entropy,
            cardinality_ratio: card_ratio,
            p99_bit_width: 8,
            max_bit_width: 8,
            use_outlier_map: false,
            frame_of_reference: 0,
            slab_width: 8,
        };
    }

    // ── Step 4: Bit-width discovery (BitSlab + optional OutlierMap) ──────────
    let for_val = *values.iter().min().unwrap();
    let mut bit_widths: Vec<u8> = values
        .iter()
        .map(|&v| bits_needed(v.saturating_sub(for_val)))
        .collect();
    bit_widths.sort_unstable();

    let p99_idx = ((bit_widths.len() as f64 * 0.99) as usize).min(bit_widths.len() - 1);
    let p99_width = bit_widths[p99_idx];
    let max_width = *bit_widths.last().unwrap();

    let use_outlier_map = max_width > p99_width.saturating_add(32);
    let slab_width = if use_outlier_map { p99_width } else { max_width };
    let slab_width = slab_width.clamp(1, 64);

    // If all widths fit comfortably, use BitSlab.
    if max_width <= 128 {
        return ClassificationResult {
            strategy: LoomStrategy::BitSlab,
            entropy,
            cardinality_ratio: card_ratio,
            p99_bit_width: p99_width,
            max_bit_width: max_width,
            use_outlier_map,
            frame_of_reference: for_val,
            slab_width,
        };
    }

    // ── Step 5: Fallback — SIMD-LZ4 ─────────────────────────────────────────
    ClassificationResult {
        strategy: LoomStrategy::SimdLz4,
        entropy,
        cardinality_ratio: card_ratio,
        p99_bit_width: p99_width,
        max_bit_width: max_width,
        use_outlier_map: false,
        frame_of_reference: for_val,
        slab_width: 64,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Heuristic helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute normalised bit-entropy of the value distribution.
///
/// Bins values by their most-significant occupied byte to approximate
/// entropy cheaply without a full histogram over u128 space.
fn bit_entropy(values: &[u128]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    // Determine the effective bit-width across all values so we bucket
    // on the top 8 *occupied* bits rather than bits 120-127 (which are
    // always zero for values that fit in a u64).
    let max_val = values.iter().copied().max().unwrap_or(0);
    let msb = if max_val > 0 { 128 - max_val.leading_zeros() } else { 0 };
    let shift = msb.saturating_sub(8);

    let mut counts = [0u32; 256];
    for &v in values {
        let bucket = ((v >> shift) & 0xFF) as usize;
        counts[bucket] += 1;
    }
    let n = values.len() as f64;
    let mut h = 0.0f64;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    // Normalise to [0, 1] relative to log2(256) = 8.
    h / 8.0
}

/// Returns `true` when first-order differences are constant or have very low
/// variance (suitable for Delta-Delta encoding).
///
/// Operates entirely in checked arithmetic: full-width `u128` values
/// reinterpreted as `i128` can produce differences or spans that exceed
/// the signed range (e.g. Decimal128 columns holding values near both
/// signed extremes). A column whose stride cannot even be represented
/// as a single `i128` is, by definition, *not* delta-stable, so any
/// overflow short-circuits to `false` rather than panicking.
fn is_delta_stable(values: &[u128]) -> bool {
    if values.len() < 4 {
        return false;
    }

    // Compute deltas as signed i128 differences, bailing out the moment
    // any pair's difference overflows i128. This is the cheap path and
    // matches the classifier's "no-false-positives" contract: if we
    // can't confidently model the stride, we decline the strategy.
    let mut deltas: Vec<i128> = Vec::with_capacity(values.len() - 1);
    for w in values.windows(2) {
        match (w[1] as i128).checked_sub(w[0] as i128) {
            Some(d) => deltas.push(d),
            None => return false,
        }
    }

    let first_delta = deltas[0];

    // Check if all deltas are identical (constant stride — perfectly sequential).
    if deltas.iter().all(|&d| d == first_delta) {
        return true;
    }

    // Check for narrow Gaussian: variance of deltas is small relative
    // to range. We promote to f64 for the statistics so we avoid i128
    // overflow on sum / range while still being fast.
    let n = deltas.len() as f64;
    let sum: f64 = deltas.iter().map(|&d| d as f64).sum();
    let mean = sum / n;
    let variance = deltas
        .iter()
        .map(|&d| {
            let diff = d as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;

    let std_dev = variance.sqrt();
    let min = *deltas.iter().min().unwrap();
    let max = *deltas.iter().max().unwrap();
    // `max - min` can overflow i128 when deltas span both signed
    // extremes; fall back to f64 which has enough headroom for the
    // ratio check below.
    let range_f64 = (max as f64) - (min as f64);

    range_f64 > 0.0 && std_dev / range_f64.abs() < 0.01
}

/// Fraction of unique values in the segment.
fn cardinality_ratio(values: &[u128]) -> f64 {
    use std::collections::HashSet;
    let unique: HashSet<u128> = values.iter().copied().collect();
    unique.len() as f64 / values.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_column_is_rle() {
        let values = vec![42u128; 1024];
        let r = classify(&values);
        assert_eq!(r.strategy, LoomStrategy::Rle);
    }

    #[test]
    fn sequential_is_delta_delta() {
        let values: Vec<u128> = (0u128..1024).collect();
        let r = classify(&values);
        assert_eq!(r.strategy, LoomStrategy::DeltaDelta);
    }

    #[test]
    fn low_cardinality_is_dictionary() {
        // 3 unique values across 1024 rows (ratio ≈ 0.003 < 0.05).
        let mut values = Vec::with_capacity(1024);
        for i in 0u128..1024 {
            values.push(i % 3);
        }
        let r = classify(&values);
        assert_eq!(r.strategy, LoomStrategy::Dictionary);
    }

    #[test]
    fn numeric_range_is_bit_slab() {
        // Scattered values in 0..65535 range (wrapping mod creates non-constant deltas).
        let values: Vec<u128> = (0u128..1024).map(|i| (i * 997) % 65536).collect();
        let r = classify(&values);
        assert_eq!(r.strategy, LoomStrategy::BitSlab);
    }

    #[test]
    fn giant_outlier_triggers_outlier_map() {
        let mut values: Vec<u128> = (0u128..1023).collect();
        values.push(u128::MAX); // one huge outlier
        let r = classify(&values);
        assert!(r.use_outlier_map, "expected outlier map to be triggered");
    }
}
