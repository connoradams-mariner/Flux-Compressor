// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

/// All errors that can be produced by the `loom` crate.
#[derive(Debug, Error)]
pub enum FluxError {
    /// Corrupt or truncated `.flux` file.
    #[error("invalid flux file: {0}")]
    InvalidFile(String),

    /// Unsupported bit-width requested (must be 1-64).
    #[error("unsupported bit width: {0}")]
    UnsupportedWidth(u8),

    /// Buffer too small for the requested operation.
    #[error("buffer overflow: need {needed} bytes, have {have}")]
    BufferOverflow { needed: usize, have: usize },

    /// Arithmetic overflow during encoding.
    #[error("value {value} overflows frame-of-reference slab (width={width})")]
    ValueOverflow { value: u128, width: u8 },

    /// Arrow interop error.
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// LZ4 compression error.
    #[error("lz4 error: {0}")]
    Lz4(String),

    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic internal error.
    #[error("internal: {0}")]
    Internal(String),
}

pub type FluxResult<T> = Result<T, FluxError>;
