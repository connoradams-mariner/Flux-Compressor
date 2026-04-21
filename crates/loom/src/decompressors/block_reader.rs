// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Block-level decompression dispatcher.
//!
//! Reads the first byte (strategy TAG) of a compressed block and routes
//! to the correct decompressor.  Returns the decoded `u128` values and the
//! number of bytes consumed from the input slice.

use std::io::Read;
use crate::{
    error::{FluxError, FluxResult},
    SecondaryCodec,
    compressors::{
        bit_slab_compressor,
        rle_compressor,
        delta_compressor,
        dict_compressor,
        lz4_compressor,
        string_compressor,
        alp_compressor,
    },
};

/// Decompress a single block starting at `data[0]`.
///
/// Handles secondary decompression (LZ4/Zstd wrapper) transparently:
/// reads the `secondary_codec` byte at `data[1]` and unwraps if needed
/// before dispatching to the strategy-specific decoder.
///
/// Returns `(values, bytes_consumed)`.
pub fn decompress_block(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    if data.len() < 2 {
        return Err(FluxError::InvalidFile("block too short".into()));
    }
    let tag = data[0];
    let secondary = SecondaryCodec::from_u8(data[1]).ok_or_else(|| {
        FluxError::InvalidFile(format!("unknown secondary codec: {:#04x}", data[1]))
    })?;

    // If there is a secondary codec, read the compressed_len prefix,
    // extract only that many bytes, and decompress.
    let inner_data: Vec<u8>;
    let dispatch_data: &[u8] = match secondary {
        SecondaryCodec::None => data,
        SecondaryCodec::Lz4 => {
            // Layout: [TAG][LZ4][u32: compressed_len][LZ4 payload]
            if data.len() < 6 {
                return Err(FluxError::InvalidFile("LZ4 block too short".into()));
            }
            let comp_len = u32::from_le_bytes(data[2..6].try_into().unwrap()) as usize;
            let payload = &data[6..6 + comp_len];
            inner_data = lz4_flex::decompress_size_prepended(payload)
                .map_err(|e| FluxError::Lz4(e.to_string()))?;
            let mut rebuilt = vec![tag, 0u8];
            rebuilt.extend_from_slice(&inner_data);
            return decompress_inner(&rebuilt);
        }
        SecondaryCodec::Zstd => {
            if data.len() < 6 {
                return Err(FluxError::InvalidFile("Zstd block too short".into()));
            }
            let comp_len = u32::from_le_bytes(data[2..6].try_into().unwrap()) as usize;
            let payload = &data[6..6 + comp_len];
            inner_data = zstd::stream::decode_all(payload)
                .map_err(|e| FluxError::Internal(format!("zstd decompress: {e}")))?;
            let mut rebuilt = vec![tag, 0u8];
            rebuilt.extend_from_slice(&inner_data);
            return decompress_inner(&rebuilt);
        }
        // SecondaryCodec::Brotli (=3) is not emitted for numeric blocks —
        // the Brotli profile falls back to Zstd for numeric data and stores
        // it with the Zstd tag.  This arm exists for forward-compatibility.
        SecondaryCodec::Brotli => {
            if data.len() < 6 {
                return Err(FluxError::InvalidFile("Brotli block too short".into()));
            }
            let comp_len = u32::from_le_bytes(data[2..6].try_into().unwrap()) as usize;
            let payload = &data[6..6 + comp_len];
            let mut decompressed = Vec::new();
            brotli::Decompressor::new(payload, 4096)
                .read_to_end(&mut decompressed)
                .map_err(|e| FluxError::Internal(format!("brotli decompress: {e}")))?;
            inner_data = decompressed;
            let mut rebuilt = vec![tag, 0u8];
            rebuilt.extend_from_slice(&inner_data);
            return decompress_inner(&rebuilt);
        }
    };

    decompress_inner(dispatch_data)
}

/// Inner dispatch to the strategy-specific decompressor.
fn decompress_inner(data: &[u8]) -> FluxResult<(Vec<u128>, usize)> {
    let tag = data[0];
    match tag {
        rle_compressor::TAG        => rle_compressor::decompress(data),
        delta_compressor::TAG      => delta_compressor::decompress(data),
        dict_compressor::TAG       => dict_compressor::decompress(data),
        bit_slab_compressor::TAG   => bit_slab_compressor::decompress(data),
        lz4_compressor::TAG        => lz4_compressor::decompress(data),
        alp_compressor::TAG        => alp_compressor::decompress(data),
        // String blocks are handled separately (they return Vec<Vec<u8>>,
        // not Vec<u128>). This tag should not reach decompress_inner.
        string_compressor::TAG     => Err(FluxError::InvalidFile(
            "string block (TAG 0x06) must be decompressed via string_compressor::decompress".into(),
        )),
        unknown => Err(FluxError::InvalidFile(format!(
            "unknown block tag: {unknown:#04x}"
        ))),
    }
}

/// Decompress a block and truncate results to `u64`.
///
/// Calls [`decompress_block`] then maps `u128 as u64`, avoiding a separate
/// `Vec<u64>` collect in the caller. The truncation is safe for all types
/// ≤ 64 bits (the common case).
pub fn decompress_block_to_u64(data: &[u8]) -> FluxResult<(Vec<u64>, usize)> {
    let (values_u128, consumed) = decompress_block(data)?;
    let values_u64: Vec<u64> = values_u128.into_iter().map(|v| v as u64).collect();
    Ok((values_u64, consumed))
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
