// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Adaptive segmenter with drift detection.
//!
//! Replaces the fixed `chunks(SEGMENT_SIZE)` approach with a probe-and-grow
//! algorithm that produces larger segments when data is homogeneous and
//! splits at drift boundaries where the optimal strategy changes.

use crate::{
    PROBE_SIZE, MAX_SEGMENT_SIZE,
    loom_classifier::{classify, LoomStrategy},
};

/// A segment produced by the adaptive segmenter: a slice of values and the
/// strategy that should be used to compress them.
pub type Segment<'a> = (&'a [u128], LoomStrategy);

/// A u64 segment: range indices + strategy. Used by the lazy-widening path
/// to avoid allocating a full `Vec<u128>` for the entire column.
pub type SegmentRange = (std::ops::Range<usize>, LoomStrategy);

/// Run the adaptive segmenter over `values`, returning a list of segments.
///
/// Each segment is a `(&[u128], LoomStrategy)` pair. The segmenter:
/// 1. Probes a `PROBE_SIZE` window to classify the data.
/// 2. Tries to grow the segment by classifying the next probe window.
/// 3. If the strategy changes (drift), emits the current segment and starts fresh.
/// 4. Caps segment size at `MAX_SEGMENT_SIZE`.
///
/// If `force_strategy` is `Some(s)`, all segments use that strategy but
/// adaptive sizing still applies (segments grow as large as possible).
pub fn adaptive_segment<'a>(
    values: &'a [u128],
    force_strategy: Option<LoomStrategy>,
) -> Vec<Segment<'a>> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut segments: Vec<Segment<'a>> = Vec::new();
    let mut pos = 0;

    while pos < values.len() {
        // 1. Probe: classify a small window.
        let probe_end = (pos + PROBE_SIZE).min(values.len());
        let probe = &values[pos..probe_end];

        let base_strategy = force_strategy.unwrap_or_else(|| {
            classify(probe).strategy
        });

        // 2. Grow: extend while the next probe classifies the same.
        //    Use geometric stride: check drift less often as the segment grows.
        //    Stride: PROBE_SIZE, 2*PROBE_SIZE, 4*PROBE_SIZE, 8*PROBE_SIZE (capped).
        //    This cuts classify() calls from ~64/segment to ~7/segment.
        let mut end = probe_end;

        if force_strategy.is_none() {
            let mut stride = PROBE_SIZE;
            while end < values.len() && (end - pos) < MAX_SEGMENT_SIZE {
                let next_end = (end + stride).min(values.len());
                // Only classify a PROBE_SIZE window at the start of the stride.
                let check_end = (end + PROBE_SIZE).min(values.len());
                let next_probe = &values[end..check_end];

                // Absorb tiny trailing fragments.
                if next_probe.len() < PROBE_SIZE / 4 {
                    end = next_end;
                    break;
                }

                let next_strategy = classify(next_probe).strategy;
                if next_strategy != base_strategy {
                    break; // Drift detected — emit current segment.
                }
                end = next_end;
                // Double the stride (geometric growth), capped at 8× PROBE_SIZE.
                stride = (stride * 2).min(PROBE_SIZE * 8);
            }
        } else {
            // Forced strategy: grow to max.
            end = (pos + MAX_SEGMENT_SIZE).min(values.len());
        }

        // 3. Emit the segment.
        segments.push((&values[pos..end], base_strategy));
        pos = end;
    }

    segments
}

/// **Lazy-widening segmenter** — works on `&[u64]` directly.
///
/// Only widens small 1024-value probe windows to `u128` for classification.
/// Returns index ranges + strategies instead of `&[u128]` slices.
/// The caller widens each segment individually (max 64K values = 1MB) instead
/// of the full column (potentially gigabytes).
pub fn adaptive_segment_u64(
    values: &[u64],
    force_strategy: Option<LoomStrategy>,
) -> Vec<SegmentRange> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut segments: Vec<SegmentRange> = Vec::new();
    let mut pos = 0;

    while pos < values.len() {
        let probe_end = (pos + PROBE_SIZE).min(values.len());

        let base_strategy = force_strategy.unwrap_or_else(|| {
            // Widen only the small probe window (1024 values = 16KB).
            let probe_u128: Vec<u128> = values[pos..probe_end]
                .iter()
                .map(|&v| v as u128)
                .collect();
            classify(&probe_u128).strategy
        });

        let mut end = probe_end;

        if force_strategy.is_none() {
            let mut stride = PROBE_SIZE;
            while end < values.len() && (end - pos) < MAX_SEGMENT_SIZE {
                let next_end = (end + stride).min(values.len());
                let check_end = (end + PROBE_SIZE).min(values.len());

                if check_end - end < PROBE_SIZE / 4 {
                    end = next_end;
                    break;
                }

                // Widen only the small check window.
                let probe_u128: Vec<u128> = values[end..check_end]
                    .iter()
                    .map(|&v| v as u128)
                    .collect();
                let next_strategy = classify(&probe_u128).strategy;
                if next_strategy != base_strategy {
                    break;
                }
                end = next_end;
                stride = (stride * 2).min(PROBE_SIZE * 8);
            }
        } else {
            end = (pos + MAX_SEGMENT_SIZE).min(values.len());
        }

        segments.push((pos..end, base_strategy));
        pos = end;
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_segment_for_uniform_data() {
        let values: Vec<u128> = vec![42; 4096];
        let segs = adaptive_segment(&values, None);
        // Uniform constant data should produce one or few large segments.
        assert!(!segs.is_empty());
        let total: usize = segs.iter().map(|(s, _)| s.len()).sum();
        assert_eq!(total, 4096);
        // All segments should be RLE.
        for (_, strategy) in &segs {
            assert_eq!(*strategy, LoomStrategy::Rle);
        }
    }

    #[test]
    fn drift_splits_segments() {
        // First half: constant (RLE), second half: sequential (DeltaDelta).
        let mut values: Vec<u128> = vec![42; 2048];
        values.extend(0u128..2048);

        let segs = adaptive_segment(&values, None);
        let total: usize = segs.iter().map(|(s, _)| s.len()).sum();
        assert_eq!(total, 4096);
        // Should have at least 2 segments with different strategies.
        assert!(segs.len() >= 2, "expected drift to split into ≥2 segments");
        assert_ne!(segs[0].1, segs.last().unwrap().1);
    }

    #[test]
    fn forced_strategy_creates_large_segments() {
        let values: Vec<u128> = (0u128..100_000).collect();
        let segs = adaptive_segment(&values, Some(LoomStrategy::BitSlab));
        let total: usize = segs.iter().map(|(s, _)| s.len()).sum();
        assert_eq!(total, 100_000);
        // With MAX_SEGMENT_SIZE = 65536, expect 2 segments.
        assert_eq!(segs.len(), 2);
        for (_, strategy) in &segs {
            assert_eq!(*strategy, LoomStrategy::BitSlab);
        }
    }
}
