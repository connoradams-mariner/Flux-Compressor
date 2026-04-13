// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hidden partitioning types for FluxTable.
//!
//! Partitioning is metadata-only — file paths are opaque (`data/part-NNNN.flux`),
//! and partition values are stored in [`FileManifest`] entries within the
//! transaction log.  This enables partition evolution without rewriting data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Partition transforms
// ─────────────────────────────────────────────────────────────────────────────

/// A transform applied to a source column to produce a partition value.
///
/// Matches the Iceberg partition transform model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionTransform {
    /// Use the column value as-is.
    Identity,
    /// Extract the year from a temporal column (e.g. `2024`).
    Year,
    /// Extract year + month (e.g. `2024-03`).
    Month,
    /// Extract year + month + day (e.g. `2024-03-15`).
    Day,
    /// Extract year + month + day + hour (e.g. `2024-03-15-14`).
    Hour,
    /// Hash the value into `n` buckets (0..n-1).
    Bucket(u32),
    /// Truncate the value to width `n` (integers: floor to nearest `n`;
    /// strings: first `n` characters).
    Truncate(u32),
}

/// A single field in a partition spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionField {
    /// The source column name in the table schema.
    pub source_column: String,
    /// The transform to apply to the source column.
    pub transform: PartitionTransform,
    /// Unique field ID within this spec (stable across evolution).
    pub field_id: u32,
}

/// A versioned partition specification.
///
/// Each spec has a unique `spec_id`.  When the partition scheme evolves,
/// a new spec is created with a new ID.  Files written under the old spec
/// retain their old `spec_id` reference in [`FileManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionSpec {
    /// Unique identifier for this spec version.
    pub spec_id: u32,
    /// The partition fields.
    pub fields: Vec<PartitionField>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-column statistics (for data skipping / pruning)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-column min/max statistics stored in a [`FileManifest`].
///
/// Values are stored as JSON-serializable strings so they work across
/// all column types (int, float, string, date, timestamp).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnStats {
    /// Minimum value in this column within the file (as string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<String>,
    /// Maximum value in this column within the file (as string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<String>,
    /// Number of null values in this column.
    #[serde(default)]
    pub null_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// File manifest
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata about a single `.flux` data file within a FluxTable.
///
/// Stored in the transaction log alongside (or replacing) the simple
/// file path string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileManifest {
    /// Relative path to the `.flux` file (e.g. `data/part-0042.flux`).
    pub path: String,

    /// Partition values for this file, keyed by partition field name.
    /// `None` value means the partition column is null for all rows in this file.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub partition_values: HashMap<String, Option<String>>,

    /// Which partition spec this file was written under.
    #[serde(default)]
    pub spec_id: u32,

    /// Number of rows in this file.
    #[serde(default)]
    pub row_count: u64,

    /// Size of the `.flux` file in bytes.
    #[serde(default)]
    pub file_size_bytes: u64,

    /// Per-column statistics for data skipping.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub column_stats: HashMap<String, ColumnStats>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Table metadata (_flux_meta.json)
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level table metadata stored in `_flux_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMeta {
    /// Unique table identifier.
    #[serde(default = "default_table_id")]
    pub table_id: String,

    /// All partition specs (for evolution support).
    #[serde(default)]
    pub partition_specs: Vec<PartitionSpec>,

    /// The currently active partition spec ID.
    #[serde(default)]
    pub current_spec_id: u32,

    /// Columns used for liquid clustering / Z-Order optimization.
    #[serde(default)]
    pub clustering_columns: Vec<String>,

    /// Arbitrary table-level properties.
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

fn default_table_id() -> String {
    format!("flux-{:016x}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos())
}

impl Default for TableMeta {
    fn default() -> Self {
        Self {
            table_id: default_table_id(),
            partition_specs: Vec::new(),
            current_spec_id: 0,
            clustering_columns: Vec::new(),
            properties: HashMap::new(),
        }
    }
}

impl TableMeta {
    /// Get the current partition spec, if any.
    pub fn current_spec(&self) -> Option<&PartitionSpec> {
        self.partition_specs.iter().find(|s| s.spec_id == self.current_spec_id)
    }

    /// Read from a `_flux_meta.json` file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        serde_json::from_slice(&data).map_err(|e| e.to_string())
    }

    /// Write to a `_flux_meta.json` file.
    pub fn write_to_file(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json.as_bytes()).map_err(|e| e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_spec_round_trip() {
        let spec = PartitionSpec {
            spec_id: 0,
            fields: vec![
                PartitionField {
                    source_column: "created_at".into(),
                    transform: PartitionTransform::Month,
                    field_id: 1,
                },
                PartitionField {
                    source_column: "country".into(),
                    transform: PartitionTransform::Identity,
                    field_id: 2,
                },
            ],
        };
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: PartitionSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, spec);
    }

    #[test]
    fn file_manifest_round_trip() {
        let manifest = FileManifest {
            path: "data/part-0042.flux".into(),
            partition_values: [("country".into(), Some("US".into()))].into(),
            spec_id: 0,
            row_count: 500_000,
            file_size_bytes: 25_000_000,
            column_stats: [(
                "revenue".into(),
                ColumnStats { min: Some("0.0".into()), max: Some("49999.5".into()), null_count: 0 },
            )].into(),
        };
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let parsed: FileManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn table_meta_defaults() {
        let meta = TableMeta::default();
        assert!(meta.table_id.starts_with("flux-"));
        assert!(meta.partition_specs.is_empty());
        assert!(meta.clustering_columns.is_empty());
    }

    #[test]
    fn bucket_and_truncate_serialize() {
        let t1 = PartitionTransform::Bucket(16);
        let t2 = PartitionTransform::Truncate(3);
        let j1 = serde_json::to_string(&t1).unwrap();
        let j2 = serde_json::to_string(&t2).unwrap();
        assert_eq!(serde_json::from_str::<PartitionTransform>(&j1).unwrap(), t1);
        assert_eq!(serde_json::from_str::<PartitionTransform>(&j2).unwrap(), t2);
    }
}
