// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Transaction log entry definition.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::partition::{FileManifest, TableMeta};
use super::schema::TableSchema;

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
    /// Metadata-only change (partition spec, clustering, properties).
    Metadata,
}

/// A single action recorded in the v2 transaction log.
///
/// The action model is the canonical representation of a commit's
/// semantics. The legacy flat fields (`data_files_added`,
/// `data_files_removed`, `file_manifests`) continue to be populated
/// alongside actions so pre-v2 readers keep loading new tables
/// correctly.
///
/// Unknown `type` values are tolerated via [`Action::Unknown`] so this
/// Rust reader stays forward-compatible with writers that emit action
/// kinds introduced in later roadmap phases. The payload for unknown
/// kinds is intentionally dropped — the legacy fields remain the
/// authoritative source for the shapes Rust doesn't yet understand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Canonical schema record (Phase A of schema evolution).
    Schema(TableSchema),
    /// Authoritative table metadata payload.
    Metadata(TableMeta),
    /// Any action kind this Rust reader does not yet understand.
    ///
    /// Future kinds (`add`, `remove`, `protocol`, `txn`, `commit_info`,
    /// …) deserialize into this variant without error so the log stays
    /// forward-compatible.
    #[serde(other)]
    Unknown,
}

impl Action {
    /// Convenience: returns `Some(&TableSchema)` if this action is a
    /// `schema` action.
    pub fn as_schema(&self) -> Option<&TableSchema> {
        match self {
            Action::Schema(s) => Some(s),
            _ => None,
        }
    }

    /// Convenience: returns `Some(&TableMeta)` if this action is a
    /// `metadata` action.
    pub fn as_metadata(&self) -> Option<&TableMeta> {
        match self {
            Action::Metadata(m) => Some(m),
            _ => None,
        }
    }
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
    /// Canonical action list (v2+).
    ///
    /// Empty on legacy pre-v2 entries; serialization is skipped when
    /// the list is empty to keep those entries byte-identical with
    /// what legacy Rust writers used to produce.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<Action>,
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
    use crate::dtype::FluxDType;
    use crate::txn::schema::{SchemaField, TableSchema};

    fn empty_entry(version: u64, op: Operation) -> LogEntry {
        LogEntry {
            version,
            timestamp_ms: 1712890800000,
            operation: op,
            data_files_added: vec![],
            data_files_removed: vec![],
            file_manifests: vec![],
            actions: vec![],
            row_count_delta: 0,
            metadata: Default::default(),
        }
    }

    #[test]
    fn round_trip_json() {
        let mut entry = empty_entry(1, Operation::Append);
        entry.data_files_added = vec!["data/part-0001.flux".into()];
        entry.row_count_delta = 100_000;
        entry.metadata = [("user".into(), "spark-job-42".into())].into_iter().collect();

        let json = entry.to_json().unwrap();
        let parsed = LogEntry::from_json(json.as_bytes()).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.operation, Operation::Append);
        assert_eq!(parsed.data_files_added.len(), 1);
        assert!(parsed.actions.is_empty());
        // Empty actions array must not appear on the wire — keeps the
        // legacy byte-shape intact for pre-v2 readers.
        assert!(!json.contains("actions"));
    }

    #[test]
    fn schema_action_round_trip() {
        let mut entry = empty_entry(2, Operation::SchemaChange);
        entry.actions.push(Action::Schema(TableSchema::new(vec![
            SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
            SchemaField::new(2, "name", FluxDType::Utf8),
        ])));

        let json = entry.to_json().unwrap();
        assert!(json.contains("\"type\": \"schema\""));
        let parsed = LogEntry::from_json(json.as_bytes()).unwrap();
        assert_eq!(parsed.actions.len(), 1);
        let schema = parsed.actions[0].as_schema().unwrap();
        assert_eq!(schema.schema_id, 0);
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.fields[0].dtype, FluxDType::UInt64);
    }

    #[test]
    fn unknown_action_kind_is_tolerated() {
        // Simulate a log entry written by a future writer that emits a
        // novel action kind. The Rust reader must succeed and preserve
        // the legacy flat fields.
        let json = r#"{
            "version": 5,
            "timestamp_ms": 1000,
            "operation": "append",
            "data_files_added": ["data/part-0005.flux"],
            "data_files_removed": [],
            "actions": [
                {"type": "add", "path": "data/part-0005.flux"},
                {"type": "commit_info", "writer_id": "abc"}
            ],
            "row_count_delta": 42,
            "metadata": {}
        }"#;
        let parsed = LogEntry::from_json(json.as_bytes()).unwrap();
        assert_eq!(parsed.actions.len(), 2);
        assert!(matches!(parsed.actions[0], Action::Unknown));
        assert!(matches!(parsed.actions[1], Action::Unknown));
        assert_eq!(parsed.data_files_added, vec!["data/part-0005.flux"]);
    }

    #[test]
    fn legacy_entry_without_actions_field_parses() {
        // Pre-Phase-A entries omit `actions` entirely.
        let json = r#"{
            "version": 0,
            "timestamp_ms": 1000,
            "operation": "create",
            "data_files_added": ["data/part-0000.flux"],
            "data_files_removed": [],
            "row_count_delta": 0,
            "metadata": {}
        }"#;
        let parsed = LogEntry::from_json(json.as_bytes()).unwrap();
        assert!(parsed.actions.is_empty());
        assert_eq!(parsed.operation, Operation::Create);
    }
}
