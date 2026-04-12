// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # DType Router — pre-classification fast path
//!
//! Inspects the Arrow [`DataType`] *before* touching any data and returns a
//! [`RouteDecision`] that determines how the column should be compressed.
//!
//! For types whose optimal strategy is known a priori (e.g. `Boolean` → RLE,
//! `Timestamp` → DeltaDelta), the router bypasses the Loom classifier entirely,
//! avoiding the u128 widening overhead and the entropy/cardinality analysis.
//!
//! For general numeric types (`Float64`, `Int64`, etc.), the router returns
//! [`RouteDecision::Classify`] to use the standard Loom waterfall.

use arrow_schema::{DataType, TimeUnit};
use crate::loom_classifier::LoomStrategy;

/// The native width of values for a fast-path column.
///
/// Determines how data is extracted and how late the u128 widening happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeWidth {
    /// 1-bit (Boolean). No widening at all — dedicated bool compressor.
    Bit1,
    /// 8-bit unsigned (UInt8, Int8 bit-cast).
    U8,
    /// 16-bit unsigned (UInt16, Int16 bit-cast).
    U16,
    /// 32-bit unsigned (UInt32, Int32, Date32, Float32 bit-cast).
    U32,
    /// 64-bit unsigned (UInt64, Int64, Timestamp, Float64 bit-cast).
    U64,
    /// 128-bit unsigned (Decimal128).
    U128,
}

/// Routing decision made by [`route`] before any data is examined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision {
    /// Skip the Loom classifier. Use this strategy directly, stay in native
    /// width until the per-segment compression call.
    FastPath {
        /// The compression strategy to use.
        strategy: LoomStrategy,
        /// The native width of the column's values.
        native_width: NativeWidth,
    },

    /// Run the full Loom classifier waterfall (general numeric path).
    /// The data is extracted as `u64` and lazy-widened to `u128` per segment.
    Classify,

    /// Route to the string / binary pipeline (Tier 2).
    StringPipeline,

    /// Route to the nested flattening pipeline (Tier 3).
    NestedPipeline,
}

/// Inspect an Arrow [`DataType`] and decide how to compress it.
///
/// This is called once per column before any data is extracted. The decision
/// determines whether the column goes through the Loom classifier or takes
/// a type-specific fast path.
pub fn route(dt: &DataType) -> RouteDecision {
    match dt {
        // ── Fast-path: Boolean ───────────────────────────────────────────
        // Always 0/1 → RLE is optimal. Dedicated bool compressor avoids
        // Vec<u64> and Vec<u128> entirely.
        DataType::Boolean => RouteDecision::FastPath {
            strategy: LoomStrategy::Rle,
            native_width: NativeWidth::Bit1,
        },

        // ── Fast-path: 8-bit integers ────────────────────────────────────
        // Max range 256 → BitSlab with width ≤8 is always optimal.
        DataType::UInt8 | DataType::Int8 => RouteDecision::FastPath {
            strategy: LoomStrategy::BitSlab,
            native_width: NativeWidth::U8,
        },

        // ── Fast-path: 16-bit integers ───────────────────────────────────
        // Max range 65536 → BitSlab with width ≤16.
        DataType::UInt16 | DataType::Int16 => RouteDecision::FastPath {
            strategy: LoomStrategy::BitSlab,
            native_width: NativeWidth::U16,
        },

        // ── Fast-path: Timestamps ────────────────────────────────────────
        // Monotonically increasing by construction from Spark/Polars/Pandas.
        // The compressor should verify monotonicity on the first probe window
        // and fall back to Classify if not monotone.
        DataType::Timestamp(TimeUnit::Second, _)
        | DataType::Timestamp(TimeUnit::Millisecond, _)
        | DataType::Timestamp(TimeUnit::Microsecond, _)
        | DataType::Timestamp(TimeUnit::Nanosecond, _) => RouteDecision::FastPath {
            strategy: LoomStrategy::DeltaDelta,
            native_width: NativeWidth::U64,
        },

        // ── Classify: general numeric types ──────────────────────────────
        // These have unpredictable data patterns and need the full Loom
        // classifier to pick the optimal strategy.
        DataType::UInt32
        | DataType::Int32
        | DataType::UInt64
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Date64 => RouteDecision::Classify,

        // Decimal128 needs the u128 path.
        DataType::Decimal128(_, _) => RouteDecision::Classify,

        // ── String / Binary pipeline (Tier 2) ────────────────────────────
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::LargeBinary => RouteDecision::StringPipeline,

        // ── Nested containers (Tier 3) ───────────────────────────────────
        DataType::Struct(_)
        | DataType::List(_)
        | DataType::LargeList(_)
        | DataType::Map(_, _) => RouteDecision::NestedPipeline,

        // ── Unsupported ──────────────────────────────────────────────────
        // Fall back to Classify for anything else; extract_column_data()
        // will produce a proper error if it truly can't handle the type.
        _ => RouteDecision::Classify,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_routes_to_fast_path_rle() {
        assert_eq!(
            route(&DataType::Boolean),
            RouteDecision::FastPath {
                strategy: LoomStrategy::Rle,
                native_width: NativeWidth::Bit1,
            },
        );
    }

    #[test]
    fn timestamp_routes_to_fast_path_delta() {
        assert_eq!(
            route(&DataType::Timestamp(TimeUnit::Microsecond, None)),
            RouteDecision::FastPath {
                strategy: LoomStrategy::DeltaDelta,
                native_width: NativeWidth::U64,
            },
        );
    }

    #[test]
    fn uint8_routes_to_fast_path_bitslab() {
        assert_eq!(
            route(&DataType::UInt8),
            RouteDecision::FastPath {
                strategy: LoomStrategy::BitSlab,
                native_width: NativeWidth::U8,
            },
        );
    }

    #[test]
    fn float64_routes_to_classify() {
        assert_eq!(route(&DataType::Float64), RouteDecision::Classify);
    }

    #[test]
    fn utf8_routes_to_string_pipeline() {
        assert_eq!(route(&DataType::Utf8), RouteDecision::StringPipeline);
    }

    #[test]
    fn struct_routes_to_nested() {
        assert_eq!(
            route(&DataType::Struct(Default::default())),
            RouteDecision::NestedPipeline,
        );
    }
}
