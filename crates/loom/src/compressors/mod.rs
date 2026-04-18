// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! All [`LoomCompressor`] implementations.

pub mod bit_slab_compressor;
pub mod rle_compressor;
pub mod delta_compressor;
pub mod dict_compressor;
pub mod lz4_compressor;
pub mod string_compressor;
pub mod alp_compressor;
pub mod flux_writer;

pub use flux_writer::{FluxWriter, compress_chunk, compress_chunk_with_profile};
