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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FluxDType {
    // ── Unsigned integers ────────────────────────────────────────────────
    UInt8   = 0x00,
    UInt16  = 0x01,
    UInt32  = 0x02,
    UInt64  = 0x03,

    // ── Signed integers ──────────────────────────────────────────────────
    Int8    = 0x04,
    Int16   = 0x05,
    Int32   = 0x06,
    Int64   = 0x07,

    // ── Floating point ───────────────────────────────────────────────────
    Float32 = 0x08,
    Float64 = 0x09,

    // ── Boolean / Date / Time ────────────────────────────────────────────
    Boolean          = 0x0A,
    Date32           = 0x0B,
    Date64           = 0x0C,
    TimestampSecond  = 0x0D,
    TimestampMillis  = 0x0E,
    TimestampMicros  = 0x0F,
    TimestampNanos   = 0x10,

    // ── Decimal ──────────────────────────────────────────────────────────
    Decimal128 = 0x11,

    // ── Variable-length (Tier 2) ─────────────────────────────────────────
    Utf8        = 0x20,
    LargeUtf8   = 0x21,
    Binary      = 0x22,
    LargeBinary = 0x23,

    // ── Internal: offset column for List/Map ─────────────────────────────
    Offsets = 0x30,

    // ── Nested containers (Tier 3) ───────────────────────────────────────
    StructContainer = 0x40,
    ListContainer   = 0x41,
    MapContainer    = 0x42,
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
}
