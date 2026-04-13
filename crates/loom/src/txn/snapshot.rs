// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Snapshot: resolves a transaction log version to a set of live data files.

use std::collections::BTreeSet;
use super::log_entry::{LogEntry, Operation};

/// A resolved snapshot of the table at a specific version.
///
/// Contains the set of live data files and aggregate statistics.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The version this snapshot represents.
    pub version: u64,
    /// Timestamp of the last transaction in this snapshot.
    pub timestamp_ms: u64,
    /// The set of live data file paths (relative to table root).
    pub live_files: BTreeSet<String>,
    /// Total row count across all live files.
    pub total_rows: i64,
}

impl Snapshot {
    /// Build a snapshot by replaying log entries up to (and including) `target_version`.
    pub fn from_log(entries: &[LogEntry], target_version: u64) -> Self {
        let mut live_files = BTreeSet::new();
        let mut total_rows: i64 = 0;
        let mut timestamp_ms = 0u64;

        for entry in entries {
            if entry.version > target_version {
                break;
            }
            for f in &entry.data_files_added {
                live_files.insert(f.clone());
            }
            for f in &entry.data_files_removed {
                live_files.remove(f);
            }
            total_rows += entry.row_count_delta;
            timestamp_ms = entry.timestamp_ms;
        }

        Snapshot {
            version: target_version,
            timestamp_ms,
            live_files,
            total_rows,
        }
    }

    /// Build the latest snapshot (replay all entries).
    pub fn latest(entries: &[LogEntry]) -> Self {
        let max_version = entries.last().map(|e| e.version).unwrap_or(0);
        Self::from_log(entries, max_version)
    }

    /// Build a snapshot at the given timestamp (highest version with timestamp ≤ t).
    pub fn at_timestamp(entries: &[LogEntry], timestamp_ms: u64) -> Self {
        let target = entries
            .iter()
            .filter(|e| e.timestamp_ms <= timestamp_ms)
            .last()
            .map(|e| e.version)
            .unwrap_or(0);
        Self::from_log(entries, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::txn::log_entry::Operation;

    fn sample_log() -> Vec<LogEntry> {
        vec![
            LogEntry {
                version: 0,
                timestamp_ms: 1000,
                operation: Operation::Create,
                data_files_added: vec!["data/part-0000.flux".into()],
                data_files_removed: vec![],
                file_manifests: vec![],
                row_count_delta: 10_000,
                metadata: Default::default(),
            },
            LogEntry {
                version: 1,
                timestamp_ms: 2000,
                operation: Operation::Append,
                data_files_added: vec!["data/part-0001.flux".into()],
                data_files_removed: vec![],
                file_manifests: vec![],
                row_count_delta: 5_000,
                metadata: Default::default(),
            },
            LogEntry {
                version: 2,
                timestamp_ms: 3000,
                operation: Operation::Compact,
                data_files_added: vec!["data/part-0002.flux".into()],
                data_files_removed: vec![
                    "data/part-0000.flux".into(),
                    "data/part-0001.flux".into(),
                ],
                file_manifests: vec![],
                row_count_delta: 0,
                metadata: Default::default(),
            },
        ]
    }

    #[test]
    fn snapshot_at_version_0() {
        let snap = Snapshot::from_log(&sample_log(), 0);
        assert_eq!(snap.live_files.len(), 1);
        assert!(snap.live_files.contains("data/part-0000.flux"));
        assert_eq!(snap.total_rows, 10_000);
    }

    #[test]
    fn snapshot_at_version_1() {
        let snap = Snapshot::from_log(&sample_log(), 1);
        assert_eq!(snap.live_files.len(), 2);
        assert_eq!(snap.total_rows, 15_000);
    }

    #[test]
    fn snapshot_after_compact() {
        let snap = Snapshot::latest(&sample_log());
        assert_eq!(snap.version, 2);
        assert_eq!(snap.live_files.len(), 1);
        assert!(snap.live_files.contains("data/part-0002.flux"));
        assert_eq!(snap.total_rows, 15_000);
    }

    #[test]
    fn snapshot_at_timestamp() {
        let snap = Snapshot::at_timestamp(&sample_log(), 1500);
        assert_eq!(snap.version, 0);
        assert_eq!(snap.live_files.len(), 1);
    }
}
