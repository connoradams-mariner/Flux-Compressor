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
//! │                     loom API                            │
//! │  LoomCompressor ──► BitSlab + OutlierMap + Atlas footer │
//! │  LoomDecompressor ◄─ SIMD Unpacker + Predicate Pushdown │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ### Zero-Copy Design
//! All hot paths operate on borrowed byte slices (`&[u8]`) or Arrow
//! `Buffer`s backed by off-heap memory, avoiding JVM↔Rust heap copies.

#![warn(missing_docs)]
#![cfg_attr(
    all(target_arch = "x86_64", feature = "simd-avx2"),
    feature(stdarch_x86_avx512)
)]

pub mod bit_io;
pub mod outlier_map;
pub mod loom_classifier;
pub mod compressors;
pub mod decompressors;
pub mod atlas;
pub mod simd;
pub mod traits;
pub mod error;

pub use traits::{LoomCompressor, LoomDecompressor, Predicate};
pub use error::FluxError;
pub use loom_classifier::{LoomStrategy, classify};

/// Magic bytes written at the end of every `.flux` file.
pub const FLUX_MAGIC: u32 = 0x464C5558; // "FLUX"

/// Number of rows per segment analysed by the Loom classifier.
pub const SEGMENT_SIZE: usize = 1024;

/// Sentinel value (all bits set) used in the primary Bit-Slab to signal
/// that the real value is stored in the Outlier Map.
pub const SENTINEL_U128: u128 = u128::MAX;
