// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Block-level decompression dispatcher.
//!
//! Reads the first byte (strategy TAG) of a compressed block and routes
//! to the correct decompressor.  Returns the decoded `u128` values and the
//! number of bytes consumed from the input slice.

use crate::{
    error::{FluxError, FluxResult},
    compressors::{
        bit_slab_compressor,
        rle_compressor,
        delta_compressor,
        dict_compressor,
        lz4_compressor,
    },
};

/// Decompress a single block starting at `data[0]`.
///
/// Returns `(values, bytes_consumed)`.
pub fn decompress_block(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    let tag = *data.first().ok_or_else(|| {
        FluxError::InvalidFile("empty block".into())
    })?;

    match tag {
        rle_compressor::TAG        => rle_compressor::decompress(data),
        delta_compressor::TAG      => delta_compressor::decompress(data),
        dict_compressor::TAG       => dict_compressor::decompress(data),
        bit_slab_compressor::TAG   => bit_slab_compressor::decompress(data),
        lz4_compressor::TAG        => lz4_compressor::decompress(data),
        unknown => Err(FluxError::InvalidFile(format!(
            "unknown block tag: {unknown:#04x}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressors::bit_slab_compressor;

    #[test]
    fn dispatch_bit_slab() {
        let values: Vec<u128> = (0u128..100).collect();
        let block = bit_slab_compressor::compress(&values).unwrap();
        let (decoded, _) = decompress_block(&block).unwrap();
        assert_eq!(decoded, values);
    }
}
