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

    /// A schema evolution transition was rejected by validation.
    ///
    /// Returned from `FluxTable::evolve_schema` when the requested new
    /// schema would violate a Phase B compatibility rule (duplicate
    /// `field_id`, dtype mismatch on a preserved id, nullable→non-
    /// nullable tightening, new non-nullable field with no default, …).
    #[error("schema evolution rejected: {0}")]
    SchemaEvolution(String),

    /// A caller-requested target schema field is not present in the
    /// file being read and has no default to fill from.
    ///
    /// Typically means the caller asked for a `field_id` that was
    /// dropped before the file's schema and the field is non-nullable
    /// with no literal default.
    #[error("field missing: {0}")]
    FieldMissing(String),
}

pub type FluxResult<T> = Result<T, FluxError>;
