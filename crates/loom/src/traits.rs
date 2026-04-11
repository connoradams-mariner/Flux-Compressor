// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Core trait definitions that every compression strategy must satisfy.

use arrow_array::RecordBatch;
use crate::error::FluxResult;

// ─────────────────────────────────────────────────────────────────────────────
// Predicate (for pushdown)
// ─────────────────────────────────────────────────────────────────────────────

/// A simple predicate used for predicate pushdown during decompression.
///
/// The decompressor evaluates this against each block's Atlas metadata
/// before touching the compressed bytes, skipping irrelevant blocks entirely.
#[derive(Debug, Clone)]
pub enum Predicate {
    /// No filtering – return all rows.
    None,
    /// `column > value`
    GreaterThan { column: String, value: i128 },
    /// `column < value`
    LessThan { column: String, value: i128 },
    /// `column == value`
    Equal { column: String, value: i128 },
    /// `column BETWEEN lo AND hi`
    Between { column: String, lo: i128, hi: i128 },
    /// Logical AND of two predicates.
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR of two predicates.
    Or(Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// Returns `true` when the block whose `[min, max]` range *might* contain
    /// rows satisfying the predicate (optimistic / conservative check).
    pub fn may_overlap(&self, min: i128, max: i128) -> bool {
        match self {
            Predicate::None => true,
            Predicate::GreaterThan { value, .. } => max > *value,
            Predicate::LessThan { value, .. } => min < *value,
            Predicate::Equal { value, .. } => min <= *value && *value <= max,
            Predicate::Between { lo, hi, .. } => min <= *hi && max >= *lo,
            Predicate::And(a, b) => a.may_overlap(min, max) && b.may_overlap(min, max),
            Predicate::Or(a, b) => a.may_overlap(min, max) || b.may_overlap(min, max),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoomCompressor
// ─────────────────────────────────────────────────────────────────────────────

/// Compress an Arrow [`RecordBatch`] into raw `.flux` bytes.
///
/// Implementations are expected to:
/// 1. Run the Loom classifier on each column.
/// 2. Write the chosen compressed block(s).
/// 3. Append the Atlas metadata footer.
/// 4. Return the complete byte buffer (zero-copy where possible).
pub trait LoomCompressor: Send + Sync {
    /// Compress `batch` into a self-contained `.flux` byte buffer.
    fn compress(&self, batch: &RecordBatch) -> FluxResult<Vec<u8>>;

    /// Compress multiple batches into a single `.flux` file, allowing global
    /// dictionary deduplication across batches (used by the Cold optimizer).
    fn compress_all(&self, batches: &[RecordBatch]) -> FluxResult<Vec<u8>> {
        // Default: concatenate independently compressed blocks.
        // Override for global-dictionary / Z-Order optimisation.
        let mut out = Vec::new();
        for batch in batches {
            out.extend(self.compress(batch)?);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoomDecompressor
// ─────────────────────────────────────────────────────────────────────────────

/// Decompress `.flux` bytes (optionally with predicate pushdown) into an Arrow
/// [`RecordBatch`].
pub trait LoomDecompressor: Send + Sync {
    /// Decompress `data` with optional predicate pushdown.
    ///
    /// The decompressor reads the Atlas footer first to enumerate block
    /// metadata, skips blocks that cannot satisfy `predicate`, and only
    /// decompresses blocks that may contain matching rows.
    fn decompress(&self, data: &[u8], predicate: &Predicate) -> FluxResult<RecordBatch>;

    /// Convenience wrapper: decompress without any predicate.
    fn decompress_all(&self, data: &[u8]) -> FluxResult<RecordBatch> {
        self.decompress(data, &Predicate::None)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FluxCapacitor (CLI / optimiser contract)
// ─────────────────────────────────────────────────────────────────────────────

/// The CLI-level contract for the `fluxcapacitor` binary.
///
/// Responsible for two-pass global optimisation, Z-Order interleaving, and
/// merging Hot blocks into Cold archives.
pub trait FluxCapacitorTrait: Send + Sync {
    /// **Pass 1** – scan all `.flux` partitions under `input_dir` and build
    /// a global master dictionary + statistics.
    fn scan_partitions(&mut self, input_dir: &std::path::Path) -> FluxResult<()>;

    /// **Pass 2** – re-pack every block using the global master dictionary,
    /// apply Z-Order interleaving, and write the optimised archive to
    /// `output_path`.
    fn optimize(&self, output_path: &std::path::Path) -> FluxResult<()>;

    /// Merge two or more `.flux` files into one, updating the Atlas footer.
    fn merge(
        &self,
        inputs: &[&std::path::Path],
        output: &std::path::Path,
    ) -> FluxResult<()>;
}
