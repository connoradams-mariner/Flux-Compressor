// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Transaction log entry definition.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::partition::FileManifest;

/// The type of operation recorded in a log entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    /// Initial table creation.
    Create,
    /// Append new data files.
    Append,
    /// Mark data files as deleted (tombstone).
    Delete,
    /// Merge/compact data files into fewer, optimized files.
    Compact,
    /// Schema evolution.
    SchemaChange,
}

/// A single entry in the transaction log.
///
/// Each entry is stored as a JSON file in `_flux_log/NNNNNNNN.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Monotonically increasing version number (0-based).
    pub version: u64,
    /// Unix timestamp in milliseconds when this transaction was committed.
    pub timestamp_ms: u64,
    /// The type of operation.
    pub operation: Operation,
    /// Data files added by this transaction (relative to table root).
    /// Simple path strings for backward compatibility.
    pub data_files_added: Vec<String>,
    /// Data files removed by this transaction (tombstoned).
    pub data_files_removed: Vec<String>,
    /// Rich file manifests with partition values and column stats.
    /// When present, these supersede `data_files_added` for metadata.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_manifests: Vec<FileManifest>,
    /// Net row count change (positive for appends, negative for deletes).
    pub row_count_delta: i64,
    /// Arbitrary user-supplied metadata (e.g., job ID, user name).
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl LogEntry {
    /// Serialize this entry to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a log entry from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }

    /// The filename for this entry (e.g., `00000042.json`).
    pub fn filename(&self) -> String {
        format!("{:08}.json", self.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_json() {
        let entry = LogEntry {
            version: 1,
            timestamp_ms: 1712890800000,
            operation: Operation::Append,
            data_files_added: vec!["data/part-0001.flux".into()],
            data_files_removed: vec![],
            file_manifests: vec![],
            row_count_delta: 100_000,
            metadata: [("user".into(), "spark-job-42".into())]
                .into_iter()
                .collect(),
        };
        let json = entry.to_json().unwrap();
        let parsed = LogEntry::from_json(json.as_bytes()).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.operation, Operation::Append);
        assert_eq!(parsed.data_files_added.len(), 1);
    }
}
