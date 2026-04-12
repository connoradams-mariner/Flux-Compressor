// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # loom
//!
//! The core FluxCompress compression/decompression engine.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  Arrow RecordBatch                                      │
//! │       │                                                 │
//! │  DType Router (pre-classification fast path)             │
//! │       ├─ Boolean/Timestamp/u8/u16 → skip classifier    │
//! │       ├─ String/Binary → string pipeline               │
//! │       │   └─ sampled cardinality → dict or LZ4/Zstd    │
//! │       ├─ Struct/List/Map → parallel leaf flattening     │
//! │       │   └─ lengths-not-offsets, delta-from-base,      │
//! │       │      constant-length fast path, key sorting     │
//! │       └─ General numeric → Loom Classifier              │
//! │                                                         │
//! │  Loom Classifier (RLE/Delta/Dict/BitSlab/LZ4)           │
//! │       │                                                 │
//! │  Compressors + OutlierMap + Secondary Codec              │
//! │       │                                                 │
//! │  Atlas Footer (61B BlockMeta + ColumnDescriptor tree)    │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Supported Types
//! All Arrow types: integers (u8–u128), floats, booleans, dates, timestamps,
//! strings, binary, structs, lists, and maps. Each type is tagged with a
//! [`FluxDType`] byte in the footer for lossless schema reconstruction.
//!
//! ## Performance
//! - **Compress**: parallel segments via rayon, parallel leaf compression
//!   for nested types, sampled cardinality estimation for strings
//! - **Decompress**: parallel block decompression, direct u64 reconstruction
//!   (skips u128 intermediary), direct Arrow string construction
//!   (zero per-string allocations), parallel nested leaf decompression
//!
//! ### Zero-Copy Design
//! All hot paths operate on borrowed byte slices (`&[u8]`) or Arrow
//! `Buffer`s backed by off-heap memory, avoiding JVM↔Rust heap copies.

#![warn(missing_docs)]

pub mod bit_io;
pub mod outlier_map;
pub mod loom_classifier;
pub mod compressors;
pub mod decompressors;
pub mod atlas;
pub mod simd;
pub mod traits;
pub mod error;
pub mod segmenter;
pub mod txn;
pub mod dtype;
pub mod dtype_router;

pub use traits::{LoomCompressor, LoomDecompressor, Predicate};
pub use error::FluxError;
pub use loom_classifier::{LoomStrategy, classify};
pub use dtype::FluxDType;
pub use dtype_router::{RouteDecision, NativeWidth};

/// Format v2 magic bytes written at the end of every `.flux` file.
/// "FLX2" in ASCII — v1 readers will reject v2 files cleanly.
pub const FLUX_MAGIC: u32 = 0x464C5832; // "FLX2"

/// Probe window size for the adaptive segmenter (rows).
pub const PROBE_SIZE: usize = 1024;

/// Maximum segment size the adaptive segmenter will grow to (rows).
pub const MAX_SEGMENT_SIZE: usize = 65536;

/// Legacy constant kept for backward compatibility in tests.
pub const SEGMENT_SIZE: usize = PROBE_SIZE;

/// Sentinel value (all bits set) used in the primary Bit-Slab to signal
/// that the real value is stored in the Outlier Map.
pub const SENTINEL_U128: u128 = u128::MAX;

/// Compression profile controlling the speed/ratio trade-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionProfile {
    /// No secondary compression. Fastest encode/decode.
    #[default]
    Speed,
    /// LZ4 post-pass on encoded blocks. Fast decode, ~2-3× extra compression.
    Balanced,
    /// Zstd post-pass on encoded blocks. Best ratio, higher CPU cost.
    Archive,
}

/// Secondary codec ID stored in block headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SecondaryCodec {
    /// No secondary compression.
    None = 0,
    /// LZ4 secondary compression.
    Lz4  = 1,
    /// Zstd secondary compression.
    Zstd = 2,
}

impl SecondaryCodec {
    /// Decode from a u8 tag byte.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::Zstd),
            _ => None,
        }
    }
}

impl CompressionProfile {
    /// Which secondary codec does this profile use?
    pub fn secondary_codec(self) -> SecondaryCodec {
        match self {
            Self::Speed    => SecondaryCodec::None,
            Self::Balanced => SecondaryCodec::Lz4,
            Self::Archive  => SecondaryCodec::Zstd,
        }
    }
}
