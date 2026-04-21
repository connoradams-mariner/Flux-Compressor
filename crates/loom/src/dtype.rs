// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Compact data-type tag for the Atlas footer.
//!
//! Every compressed block records the original Arrow [`DataType`] as a single
//! `u8` so the reader can reconstruct correctly-typed arrays on decompression.

use arrow_schema::{DataType, TimeUnit};

/// Compact representation of an Arrow [`DataType`] stored in [`BlockMeta`].
///
/// This is a 1-byte tag written into the Atlas footer so the decompressor
/// knows how to reconstruct the original Arrow array type.
///
/// The `serde` representation uses stable lower-case spellings
/// (e.g. `"uint64"`, `"utf8"`, `"timestamp_micros"`). The spellings are
/// part of the on-disk schema-evolution log format and must not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum FluxDType {
    // ── Unsigned integers ────────────────────────────────────────────────
    #[serde(rename = "uint8")]  UInt8   = 0x00,
    #[serde(rename = "uint16")] UInt16  = 0x01,
    #[serde(rename = "uint32")] UInt32  = 0x02,
    #[serde(rename = "uint64")] UInt64  = 0x03,

    // ── Signed integers ──────────────────────────────────────────────────
    #[serde(rename = "int8")]  Int8  = 0x04,
    #[serde(rename = "int16")] Int16 = 0x05,
    #[serde(rename = "int32")] Int32 = 0x06,
    #[serde(rename = "int64")] Int64 = 0x07,

    // ── Floating point ───────────────────────────────────────────────────
    #[serde(rename = "float32")] Float32 = 0x08,
    #[serde(rename = "float64")] Float64 = 0x09,

    // ── Boolean / Date / Time ────────────────────────────────────────────
    #[serde(rename = "boolean")]          Boolean          = 0x0A,
    #[serde(rename = "date32")]           Date32           = 0x0B,
    #[serde(rename = "date64")]           Date64           = 0x0C,
    #[serde(rename = "timestamp_second")] TimestampSecond  = 0x0D,
    #[serde(rename = "timestamp_millis")] TimestampMillis  = 0x0E,
    #[serde(rename = "timestamp_micros")] TimestampMicros  = 0x0F,
    #[serde(rename = "timestamp_nanos")]  TimestampNanos   = 0x10,

    // ── Decimal ──────────────────────────────────────────────────────────
    #[serde(rename = "decimal128")] Decimal128 = 0x11,

    // ── Variable-length (Tier 2) ─────────────────────────────────────────
    #[serde(rename = "utf8")]         Utf8        = 0x20,
    #[serde(rename = "large_utf8")]   LargeUtf8   = 0x21,
    #[serde(rename = "binary")]       Binary      = 0x22,
    #[serde(rename = "large_binary")] LargeBinary = 0x23,

    // ── Internal: offset column for List/Map ─────────────────────────────
    #[serde(rename = "offsets")] Offsets = 0x30,

    // ── Nested containers (Tier 3) ───────────────────────────────────────
    #[serde(rename = "struct")] StructContainer = 0x40,
    #[serde(rename = "list")]   ListContainer   = 0x41,
    #[serde(rename = "map")]    MapContainer    = 0x42,
}

impl FluxDType {
    /// Convert from an Arrow [`DataType`] to a [`FluxDType`] tag.
    ///
    /// Returns `None` for unsupported / unmapped types.
    pub fn from_arrow(dt: &DataType) -> Option<Self> {
        match dt {
            DataType::UInt8   => Some(Self::UInt8),
            DataType::UInt16  => Some(Self::UInt16),
            DataType::UInt32  => Some(Self::UInt32),
            DataType::UInt64  => Some(Self::UInt64),
            DataType::Int8    => Some(Self::Int8),
            DataType::Int16   => Some(Self::Int16),
            DataType::Int32   => Some(Self::Int32),
            DataType::Int64   => Some(Self::Int64),
            DataType::Float32 => Some(Self::Float32),
            DataType::Float64 => Some(Self::Float64),
            DataType::Boolean => Some(Self::Boolean),
            DataType::Date32  => Some(Self::Date32),
            DataType::Date64  => Some(Self::Date64),
            DataType::Timestamp(TimeUnit::Second, _)      => Some(Self::TimestampSecond),
            DataType::Timestamp(TimeUnit::Millisecond, _)  => Some(Self::TimestampMillis),
            DataType::Timestamp(TimeUnit::Microsecond, _)  => Some(Self::TimestampMicros),
            DataType::Timestamp(TimeUnit::Nanosecond, _)   => Some(Self::TimestampNanos),
            DataType::Decimal128(_, _) => Some(Self::Decimal128),
            DataType::Utf8        => Some(Self::Utf8),
            DataType::LargeUtf8   => Some(Self::LargeUtf8),
            DataType::Binary      => Some(Self::Binary),
            DataType::LargeBinary => Some(Self::LargeBinary),
            DataType::Struct(_)   => Some(Self::StructContainer),
            DataType::List(_)     => Some(Self::ListContainer),
            DataType::Map(_, _)   => Some(Self::MapContainer),
            _ => None,
        }
    }

    /// Convert a [`FluxDType`] tag back to an Arrow [`DataType`].
    ///
    /// For types that carry parameters (Timestamp timezone, Decimal
    /// precision/scale), default values are used. The full schema should be
    /// reconstructed from the `ColumnDescriptor` tree when available.
    pub fn to_arrow(self) -> DataType {
        match self {
            Self::UInt8   => DataType::UInt8,
            Self::UInt16  => DataType::UInt16,
            Self::UInt32  => DataType::UInt32,
            Self::UInt64  => DataType::UInt64,
            Self::Int8    => DataType::Int8,
            Self::Int16   => DataType::Int16,
            Self::Int32   => DataType::Int32,
            Self::Int64   => DataType::Int64,
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
            Self::Boolean => DataType::Boolean,
            Self::Date32  => DataType::Date32,
            Self::Date64  => DataType::Date64,
            Self::TimestampSecond => DataType::Timestamp(TimeUnit::Second, None),
            Self::TimestampMillis => DataType::Timestamp(TimeUnit::Millisecond, None),
            Self::TimestampMicros => DataType::Timestamp(TimeUnit::Microsecond, None),
            Self::TimestampNanos  => DataType::Timestamp(TimeUnit::Nanosecond, None),
            Self::Decimal128 => DataType::Decimal128(38, 10),
            Self::Utf8        => DataType::Utf8,
            Self::LargeUtf8   => DataType::LargeUtf8,
            Self::Binary      => DataType::Binary,
            Self::LargeBinary => DataType::LargeBinary,
            // Containers don't map to a single leaf DataType; callers must
            // use the ColumnDescriptor tree to reconstruct nested schemas.
            Self::Offsets         => DataType::Int32,
            Self::StructContainer => DataType::Struct(Default::default()),
            Self::ListContainer   => DataType::List(std::sync::Arc::new(
                arrow_schema::Field::new("item", DataType::Int64, true),
            )),
            Self::MapContainer => DataType::Map(
                std::sync::Arc::new(arrow_schema::Field::new(
                    "entries",
                    DataType::Struct(Default::default()),
                    false,
                )),
                false,
            ),
        }
    }

    /// Decode from a raw `u8` tag byte.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00 => Some(Self::UInt8),
            0x01 => Some(Self::UInt16),
            0x02 => Some(Self::UInt32),
            0x03 => Some(Self::UInt64),
            0x04 => Some(Self::Int8),
            0x05 => Some(Self::Int16),
            0x06 => Some(Self::Int32),
            0x07 => Some(Self::Int64),
            0x08 => Some(Self::Float32),
            0x09 => Some(Self::Float64),
            0x0A => Some(Self::Boolean),
            0x0B => Some(Self::Date32),
            0x0C => Some(Self::Date64),
            0x0D => Some(Self::TimestampSecond),
            0x0E => Some(Self::TimestampMillis),
            0x0F => Some(Self::TimestampMicros),
            0x10 => Some(Self::TimestampNanos),
            0x11 => Some(Self::Decimal128),
            0x20 => Some(Self::Utf8),
            0x21 => Some(Self::LargeUtf8),
            0x22 => Some(Self::Binary),
            0x23 => Some(Self::LargeBinary),
            0x30 => Some(Self::Offsets),
            0x40 => Some(Self::StructContainer),
            0x41 => Some(Self::ListContainer),
            0x42 => Some(Self::MapContainer),
            _ => None,
        }
    }

    /// Encode to a raw `u8` tag byte.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Phase C: is `self → target` a permitted schema-evolution
    /// promotion?
    ///
    /// Allowed promotions (reader-side widening; no data rewrite):
    /// * Signed: `Int8 → Int16 → Int32 → Int64 → Decimal128`.
    /// * Unsigned: `UInt8 → UInt16 → UInt32 → UInt64 → Decimal128`.
    /// * Float: `Float32 → Float64`.
    /// * Strings: `Utf8 → LargeUtf8`, `Binary → LargeBinary`.
    /// * Identity: any dtype → itself.
    ///
    /// Everything else — narrowing, cross-family, sign changes, scale
    /// changes, `Decimal128 → anything smaller` — is rejected and
    /// lives behind a `SchemaEvolution` error at validate time.
    ///
    /// `Float128` and `Decimal256` extensions are tracked in
    /// `docs/roadmap-f128.md` and slot into this helper when the
    /// on-disk carriers land.
    pub fn can_promote_to(self, target: Self) -> bool {
        use FluxDType::*;
        if self == target {
            return true;
        }
        matches!(
            (self, target),
            // Signed integer widening.
            (Int8,  Int16) | (Int8,  Int32) | (Int8,  Int64) | (Int8,  Decimal128)
          | (Int16, Int32) | (Int16, Int64) | (Int16, Decimal128)
          | (Int32, Int64) | (Int32, Decimal128)
          | (Int64, Decimal128)
            // Unsigned integer widening.
          | (UInt8,  UInt16) | (UInt8,  UInt32) | (UInt8,  UInt64) | (UInt8,  Decimal128)
          | (UInt16, UInt32) | (UInt16, UInt64) | (UInt16, Decimal128)
          | (UInt32, UInt64) | (UInt32, Decimal128)
          | (UInt64, Decimal128)
            // Float widening.
          | (Float32, Float64)
            // String / binary length widening.
          | (Utf8, LargeUtf8)
          | (Binary, LargeBinary)
        )
    }

    /// Arrow [`DataType`] to cast to when promoting from `self` to
    /// `target`. Returns `None` when the pair is not a permitted
    /// promotion.
    ///
    /// For integer → `Decimal128` the target is pinned to
    /// `Decimal128(38, 0)` — the canonical 128-bit integer carrier
    /// with no fractional scale — which differs from the default
    /// `FluxDType::Decimal128.to_arrow()` mapping (`(38, 10)`). We
    /// want the whole integer value to land in the integer portion
    /// of the decimal; scaling is Phase C.2 territory.
    pub fn cast_target_arrow_dtype(self, target: Self) -> Option<DataType> {
        if !self.can_promote_to(target) {
            return None;
        }
        if self == target {
            return Some(self.to_arrow());
        }
        if matches!(target, FluxDType::Decimal128) {
            // Scale 0 for IntN / UIntN → Decimal128.
            return Some(DataType::Decimal128(38, 0));
        }
        Some(target.to_arrow())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_all_tags() {
        let all = [
            FluxDType::UInt8, FluxDType::UInt16, FluxDType::UInt32, FluxDType::UInt64,
            FluxDType::Int8, FluxDType::Int16, FluxDType::Int32, FluxDType::Int64,
            FluxDType::Float32, FluxDType::Float64,
            FluxDType::Boolean, FluxDType::Date32, FluxDType::Date64,
            FluxDType::TimestampSecond, FluxDType::TimestampMillis,
            FluxDType::TimestampMicros, FluxDType::TimestampNanos,
            FluxDType::Decimal128,
            FluxDType::Utf8, FluxDType::LargeUtf8, FluxDType::Binary, FluxDType::LargeBinary,
            FluxDType::Offsets,
            FluxDType::StructContainer, FluxDType::ListContainer, FluxDType::MapContainer,
        ];
        for dt in all {
            let tag = dt.as_u8();
            let decoded = FluxDType::from_u8(tag).unwrap();
            assert_eq!(decoded, dt, "round-trip failed for {dt:?} (tag={tag:#04x})");
        }
    }

    #[test]
    fn from_arrow_common_types() {
        assert_eq!(FluxDType::from_arrow(&DataType::UInt64), Some(FluxDType::UInt64));
        assert_eq!(FluxDType::from_arrow(&DataType::Float64), Some(FluxDType::Float64));
        assert_eq!(FluxDType::from_arrow(&DataType::Boolean), Some(FluxDType::Boolean));
        assert_eq!(
            FluxDType::from_arrow(&DataType::Timestamp(TimeUnit::Microsecond, None)),
            Some(FluxDType::TimestampMicros),
        );
    }

    // ── Phase C: promotion matrix ───────────────────────────────────────

    #[test]
    fn identity_promotion_allowed() {
        for t in [
            FluxDType::Int8, FluxDType::Int16, FluxDType::Int32, FluxDType::Int64,
            FluxDType::UInt8, FluxDType::UInt16, FluxDType::UInt32, FluxDType::UInt64,
            FluxDType::Float32, FluxDType::Float64,
            FluxDType::Utf8, FluxDType::LargeUtf8,
            FluxDType::Binary, FluxDType::LargeBinary,
            FluxDType::Decimal128, FluxDType::Boolean,
        ] {
            assert!(t.can_promote_to(t), "identity should be allowed for {t:?}");
        }
    }

    #[test]
    fn signed_widening_allowed_up_through_decimal128() {
        for (s, t) in [
            (FluxDType::Int8,  FluxDType::Int16),
            (FluxDType::Int8,  FluxDType::Int32),
            (FluxDType::Int8,  FluxDType::Int64),
            (FluxDType::Int8,  FluxDType::Decimal128),
            (FluxDType::Int16, FluxDType::Int32),
            (FluxDType::Int16, FluxDType::Int64),
            (FluxDType::Int16, FluxDType::Decimal128),
            (FluxDType::Int32, FluxDType::Int64),
            (FluxDType::Int32, FluxDType::Decimal128),
            (FluxDType::Int64, FluxDType::Decimal128),
        ] {
            assert!(s.can_promote_to(t), "{s:?} → {t:?} should be allowed");
        }
    }

    #[test]
    fn unsigned_widening_allowed_up_through_decimal128() {
        for (s, t) in [
            (FluxDType::UInt8,  FluxDType::UInt16),
            (FluxDType::UInt8,  FluxDType::UInt32),
            (FluxDType::UInt8,  FluxDType::UInt64),
            (FluxDType::UInt8,  FluxDType::Decimal128),
            (FluxDType::UInt16, FluxDType::UInt32),
            (FluxDType::UInt16, FluxDType::UInt64),
            (FluxDType::UInt16, FluxDType::Decimal128),
            (FluxDType::UInt32, FluxDType::UInt64),
            (FluxDType::UInt32, FluxDType::Decimal128),
            (FluxDType::UInt64, FluxDType::Decimal128),
        ] {
            assert!(s.can_promote_to(t), "{s:?} → {t:?} should be allowed");
        }
    }

    #[test]
    fn float_and_string_widening_allowed() {
        assert!(FluxDType::Float32.can_promote_to(FluxDType::Float64));
        assert!(FluxDType::Utf8.can_promote_to(FluxDType::LargeUtf8));
        assert!(FluxDType::Binary.can_promote_to(FluxDType::LargeBinary));
    }

    #[test]
    fn narrowing_and_cross_family_rejected() {
        // Narrowing.
        assert!(!FluxDType::Int64.can_promote_to(FluxDType::Int32));
        assert!(!FluxDType::UInt64.can_promote_to(FluxDType::UInt32));
        assert!(!FluxDType::Float64.can_promote_to(FluxDType::Float32));
        assert!(!FluxDType::LargeUtf8.can_promote_to(FluxDType::Utf8));
        // Sign changes.
        assert!(!FluxDType::Int32.can_promote_to(FluxDType::UInt32));
        assert!(!FluxDType::UInt32.can_promote_to(FluxDType::Int32));
        // Family changes.
        assert!(!FluxDType::Int32.can_promote_to(FluxDType::Float32));
        assert!(!FluxDType::Float64.can_promote_to(FluxDType::Int64));
        assert!(!FluxDType::Decimal128.can_promote_to(FluxDType::Int64));
        // Float → Decimal128 not allowed (lossy), Decimal128 → Float not allowed.
        assert!(!FluxDType::Float32.can_promote_to(FluxDType::Decimal128));
        assert!(!FluxDType::Decimal128.can_promote_to(FluxDType::Float64));
    }

    #[test]
    fn int_to_decimal128_casts_with_scale_zero() {
        // The 128-bit integer carrier must pin scale=0 so integer values
        // land in the integer part, not scaled by 10^10.
        let target = FluxDType::Int64
            .cast_target_arrow_dtype(FluxDType::Decimal128)
            .unwrap();
        assert_eq!(target, DataType::Decimal128(38, 0));

        let target_u = FluxDType::UInt64
            .cast_target_arrow_dtype(FluxDType::Decimal128)
            .unwrap();
        assert_eq!(target_u, DataType::Decimal128(38, 0));
    }

    #[test]
    fn cast_target_returns_none_on_rejected_pair() {
        assert!(FluxDType::Int64.cast_target_arrow_dtype(FluxDType::Int32).is_none());
        assert!(FluxDType::Float64.cast_target_arrow_dtype(FluxDType::Int64).is_none());
    }
}
