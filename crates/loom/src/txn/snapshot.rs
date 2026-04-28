// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Snapshot: resolves a transaction log version to the live table
//! state — live files, their [`FileManifest`]s, the replayed
//! [`SchemaChain`], and the currently-active [`TableMeta`].
//!
//! ## Performance shape
//! Replay is a single forward pass over the log, O(N) in the number of
//! entries and bounded by the size of the live file set. It deliberately
//! avoids any per-file second pass so that scan pipelines can start
//! issuing parallel file reads the moment the snapshot is returned.
//! Schema resolution from a file's manifest is O(1) via
//! [`SchemaChain::get`].

use std::collections::{BTreeMap, BTreeSet};

use super::log_entry::{Action, LogEntry};
use super::partition::{FileManifest, TableMeta};
use super::schema::SchemaChain;

/// A resolved snapshot of the table at a specific version.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The version this snapshot represents.
    pub version: u64,
    /// Timestamp of the last transaction in this snapshot.
    pub timestamp_ms: u64,
    /// The set of live data file paths (relative to table root).
    ///
    /// Stored alongside `live_manifests` for the legacy reader path
    /// that only needs paths. Kept in sync with `live_manifests`
    /// during replay.
    pub live_files: BTreeSet<String>,
    /// Rich live manifests keyed by path. Populated from the
    /// `file_manifests` field of each log entry; files added via the
    /// legacy `data_files_added` path (no manifest) get a synthetic
    /// zero-stat manifest so callers always see a uniform map.
    pub live_manifests: BTreeMap<String, FileManifest>,
    /// Total row count across all live files.
    pub total_rows: i64,
    /// Replayed schema evolution chain, indexed by `schema_id`.
    pub schema_chain: SchemaChain,
    /// Authoritative table metadata as of this snapshot, if any
    /// `metadata` action has been observed.
    pub table_meta: Option<TableMeta>,
}

impl Snapshot {
    /// Build a snapshot by replaying log entries up to (and including)
    /// `target_version`. Later entries are ignored.
    pub fn from_log(entries: &[LogEntry], target_version: u64) -> Self {
        let mut live_files = BTreeSet::new();
        let mut live_manifests: BTreeMap<String, FileManifest> = BTreeMap::new();
        let mut total_rows: i64 = 0;
        let mut timestamp_ms = 0u64;
        let mut schema_chain = SchemaChain::new();
        let mut table_meta: Option<TableMeta> = None;

        for entry in entries {
            if entry.version > target_version {
                break;
            }

            // Legacy path additions: ensure the path is tracked even
            // when no rich manifest accompanies it. Manifests supplied
            // below will overwrite this synthetic placeholder.
            for f in &entry.data_files_added {
                live_files.insert(f.clone());
                live_manifests
                    .entry(f.clone())
                    .or_insert_with(|| synthetic_manifest(f));
            }

            // Rich manifests authoritatively describe the files they
            // reference, including partition values, stats and the
            // (Phase A) `schema_id` stamp.
            for m in &entry.file_manifests {
                live_files.insert(m.path.clone());
                live_manifests.insert(m.path.clone(), m.clone());
            }

            for f in &entry.data_files_removed {
                live_files.remove(f);
                live_manifests.remove(f);
            }

            // Canonical v2 actions. Processed after the legacy fields
            // so a metadata action in the same commit wins over any
            // implicit metadata state. Unknown action kinds are a
            // no-op — the legacy fields remain authoritative for those.
            for action in &entry.actions {
                match action {
                    Action::Schema(s) => schema_chain.insert(s.clone()),
                    Action::Metadata(m) => table_meta = Some(m.clone()),
                    Action::Unknown => {}
                }
            }

            total_rows += entry.row_count_delta;
            timestamp_ms = entry.timestamp_ms;
        }

        Snapshot {
            version: target_version,
            timestamp_ms,
            live_files,
            live_manifests,
            total_rows,
            schema_chain,
            table_meta,
        }
    }

    /// Build the latest snapshot (replay all entries).
    pub fn latest(entries: &[LogEntry]) -> Self {
        let max_version = entries.last().map(|e| e.version).unwrap_or(0);
        Self::from_log(entries, max_version)
    }

    /// Build a snapshot at the given timestamp (highest version with
    /// timestamp ≤ t).
    pub fn at_timestamp(entries: &[LogEntry], timestamp_ms: u64) -> Self {
        let target = entries
            .iter()
            .filter(|e| e.timestamp_ms <= timestamp_ms)
            .last()
            .map(|e| e.version)
            .unwrap_or(0);
        Self::from_log(entries, target)
    }

    /// Currently active `schema_id`, preferring the authoritative
    /// `metadata` action if present and falling back to the highest
    /// schema id in the chain otherwise.
    ///
    /// Returns `None` on pre-evolution tables; readers should treat
    /// that as implicit `schema_id = 0`.
    pub fn current_schema_id(&self) -> Option<u32> {
        self.table_meta
            .as_ref()
            .and_then(|m| m.current_schema_id)
            .or_else(|| self.schema_chain.max_schema_id())
    }
}

/// Build a minimal [`FileManifest`] for a path that only appears in the
/// legacy `data_files_added` list. We intentionally leave `schema_id`
/// as `None` so downstream readers treat the file as pre-evolution
/// (implicit `schema_id = 0`), matching the roadmap's backward-compat
/// rule.
fn synthetic_manifest(path: &str) -> FileManifest {
    FileManifest {
        path: path.to_string(),
        partition_values: Default::default(),
        spec_id: 0,
        schema_id: None,
        row_count: 0,
        file_size_bytes: 0,
        column_stats: Default::default(),
        column_stats_by_field_id: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::FluxDType;
    use crate::txn::log_entry::{Action, Operation};
    use crate::txn::schema::{SchemaField, TableSchema};

    fn make_entry(version: u64, timestamp_ms: u64, op: Operation) -> LogEntry {
        LogEntry {
            version,
            timestamp_ms,
            operation: op,
            data_files_added: vec![],
            data_files_removed: vec![],
            file_manifests: vec![],
            actions: vec![],
            row_count_delta: 0,
            metadata: Default::default(),
        }
    }

    fn sample_log() -> Vec<LogEntry> {
        let mut v0 = make_entry(0, 1000, Operation::Create);
        v0.data_files_added = vec!["data/part-0000.flux".into()];
        v0.row_count_delta = 10_000;

        let mut v1 = make_entry(1, 2000, Operation::Append);
        v1.data_files_added = vec!["data/part-0001.flux".into()];
        v1.row_count_delta = 5_000;

        let mut v2 = make_entry(2, 3000, Operation::Compact);
        v2.data_files_added = vec!["data/part-0002.flux".into()];
        v2.data_files_removed = vec!["data/part-0000.flux".into(), "data/part-0001.flux".into()];

        vec![v0, v1, v2]
    }

    #[test]
    fn snapshot_at_version_0() {
        let snap = Snapshot::from_log(&sample_log(), 0);
        assert_eq!(snap.live_files.len(), 1);
        assert!(snap.live_files.contains("data/part-0000.flux"));
        assert_eq!(snap.total_rows, 10_000);
        // Even legacy-path entries yield a manifest entry so scans see
        // a uniform view.
        assert_eq!(snap.live_manifests.len(), 1);
    }

    #[test]
    fn snapshot_at_version_1() {
        let snap = Snapshot::from_log(&sample_log(), 1);
        assert_eq!(snap.live_files.len(), 2);
        assert_eq!(snap.live_manifests.len(), 2);
        assert_eq!(snap.total_rows, 15_000);
    }

    #[test]
    fn snapshot_after_compact() {
        let snap = Snapshot::latest(&sample_log());
        assert_eq!(snap.version, 2);
        assert_eq!(snap.live_files.len(), 1);
        assert!(snap.live_files.contains("data/part-0002.flux"));
        assert_eq!(snap.live_manifests.len(), 1);
        assert_eq!(snap.total_rows, 15_000);
    }

    #[test]
    fn snapshot_at_timestamp() {
        let snap = Snapshot::at_timestamp(&sample_log(), 1500);
        assert_eq!(snap.version, 0);
        assert_eq!(snap.live_files.len(), 1);
    }

    #[test]
    fn schema_chain_is_replayed() {
        let schema0 = TableSchema::new(vec![SchemaField::new(1, "id", FluxDType::UInt64)]);
        let schema1 = TableSchema {
            schema_id: 1,
            parent_schema_id: Some(0),
            fields: vec![
                SchemaField::new(1, "id", FluxDType::UInt64),
                SchemaField::new(2, "name", FluxDType::Utf8),
            ],
            change_summary: Some("add name".into()),
        };

        let mut v0 = make_entry(0, 1000, Operation::SchemaChange);
        v0.actions.push(Action::Schema(schema0.clone()));
        v0.actions.push(Action::Metadata(TableMeta {
            current_schema_id: Some(0),
            ..TableMeta::default()
        }));

        let mut v1 = make_entry(1, 2000, Operation::Append);
        let manifest = FileManifest {
            path: "data/part-0000.flux".into(),
            partition_values: Default::default(),
            spec_id: 0,
            schema_id: Some(0),
            row_count: 100,
            file_size_bytes: 1024,
            column_stats: Default::default(),
            column_stats_by_field_id: Default::default(),
        };
        v1.data_files_added = vec![manifest.path.clone()];
        v1.file_manifests = vec![manifest];
        v1.row_count_delta = 100;

        let mut v2 = make_entry(2, 3000, Operation::SchemaChange);
        v2.actions.push(Action::Schema(schema1.clone()));
        v2.actions.push(Action::Metadata(TableMeta {
            current_schema_id: Some(1),
            ..TableMeta::default()
        }));

        let mut v3 = make_entry(3, 4000, Operation::Append);
        let manifest2 = FileManifest {
            path: "data/part-0001.flux".into(),
            partition_values: Default::default(),
            spec_id: 0,
            schema_id: Some(1),
            row_count: 200,
            file_size_bytes: 2048,
            column_stats: Default::default(),
            column_stats_by_field_id: Default::default(),
        };
        v3.data_files_added = vec![manifest2.path.clone()];
        v3.file_manifests = vec![manifest2];
        v3.row_count_delta = 200;

        let snap = Snapshot::latest(&[v0, v1, v2, v3]);
        assert_eq!(snap.live_files.len(), 2);
        assert_eq!(snap.schema_chain.len(), 2);
        assert_eq!(snap.current_schema_id(), Some(1));

        // O(1) lookup of a file's schema by its stamped id.
        let m = snap.live_manifests.get("data/part-0000.flux").unwrap();
        let s = snap.schema_chain.get(m.schema_id.unwrap()).unwrap();
        assert_eq!(s, &schema0);

        let m2 = snap.live_manifests.get("data/part-0001.flux").unwrap();
        let s2 = snap.schema_chain.get(m2.schema_id.unwrap()).unwrap();
        assert_eq!(s2, &schema1);
    }

    #[test]
    fn unknown_action_does_not_disturb_replay() {
        // Legacy entries with a future/unknown action must not corrupt
        // the schema chain or metadata.
        let json = r#"{
            "version": 0,
            "timestamp_ms": 1000,
            "operation": "append",
            "data_files_added": ["data/part-0000.flux"],
            "data_files_removed": [],
            "actions": [{"type": "commit_info", "writer_id": "abc"}],
            "row_count_delta": 42,
            "metadata": {}
        }"#;
        let entry = LogEntry::from_json(json.as_bytes()).unwrap();
        let snap = Snapshot::latest(&[entry]);
        assert_eq!(snap.live_files.len(), 1);
        assert!(snap.schema_chain.is_empty());
        assert!(snap.table_meta.is_none());
    }
}
