// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Liquid clustering optimizer for FluxTable.
//!
//! Detects files with overlapping column ranges on the clustering key,
//! groups them into merge sets, reads + Z-Order sorts + rewrites, and
//! commits a single `Compact` transaction.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::{FluxError, FluxResult};
use super::partition::{ColumnStats, FileManifest, TableMeta};
use super::table::FluxTable;
use super::log_entry::{LogEntry, Operation};

/// Options for the OPTIMIZE command.
#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    /// Target file size in bytes (default 128 MB).
    pub target_file_size: u64,
    /// Minimum number of files in a merge group to trigger re-clustering.
    pub min_files_to_merge: usize,
    /// Override clustering columns (evolves the table metadata if different).
    pub clustering_columns: Option<Vec<String>>,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            target_file_size: 128 * 1024 * 1024, // 128 MB
            min_files_to_merge: 2,
            clustering_columns: None,
        }
    }
}

/// Result of an OPTIMIZE run.
#[derive(Debug)]
pub struct OptimizeResult {
    /// Number of merge groups processed.
    pub groups_merged: usize,
    /// Number of input files consumed.
    pub files_read: usize,
    /// Number of output files written.
    pub files_written: usize,
    /// Total rows processed.
    pub rows_processed: u64,
}

/// Detect groups of files that overlap on the clustering columns.
///
/// Uses a simple interval-graph approach: for each clustering column,
/// two files overlap if their [min, max] ranges intersect.  Files that
/// overlap on ALL clustering columns form a merge group.
pub fn find_overlapping_groups(
    manifests: &[FileManifest],
    clustering_columns: &[String],
) -> Vec<Vec<usize>> {
    let n = manifests.len();
    if n <= 1 || clustering_columns.is_empty() {
        return Vec::new();
    }

    // Build adjacency: file i overlaps file j if ranges overlap on ALL clustering cols.
    let mut overlaps = vec![vec![false; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if all_columns_overlap(&manifests[i], &manifests[j], clustering_columns) {
                overlaps[i][j] = true;
                overlaps[j][i] = true;
            }
        }
    }

    // Connected components via BFS.
    let mut visited = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut group = Vec::new();
        let mut queue = vec![start];
        visited[start] = true;

        while let Some(node) = queue.pop() {
            group.push(node);
            for neighbor in 0..n {
                if !visited[neighbor] && overlaps[node][neighbor] {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }

        groups.push(group);
    }

    groups
}

/// Check if two files' column stats overlap on ALL given columns.
fn all_columns_overlap(a: &FileManifest, b: &FileManifest, columns: &[String]) -> bool {
    for col in columns {
        let a_stats = a.column_stats.get(col);
        let b_stats = b.column_stats.get(col);
        match (a_stats, b_stats) {
            (Some(sa), Some(sb)) => {
                if !ranges_overlap(sa, sb) {
                    return false;
                }
            }
            // If stats are missing for a column, assume overlap (conservative).
            _ => {}
        }
    }
    true
}

/// Check if two [min, max] ranges overlap (string comparison).
fn ranges_overlap(a: &ColumnStats, b: &ColumnStats) -> bool {
    match (&a.min, &a.max, &b.min, &b.max) {
        (Some(a_min), Some(a_max), Some(b_min), Some(b_max)) => {
            // Ranges overlap if NOT (a_max < b_min OR b_max < a_min)
            !(a_max < b_min || b_max < a_min)
        }
        // Missing bounds → assume overlap.
        _ => true,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manifest(path: &str, col: &str, min: &str, max: &str) -> FileManifest {
        FileManifest {
            path: path.into(),
            partition_values: HashMap::new(),
            spec_id: 0,
            schema_id: None,
            row_count: 1000,
            file_size_bytes: 10000,
            column_stats: [(
                col.into(),
                ColumnStats { min: Some(min.into()), max: Some(max.into()), null_count: 0 },
            )].into(),
            column_stats_by_field_id: Default::default(),
        }
    }

    #[test]
    fn non_overlapping_files_no_groups() {
        let manifests = vec![
            make_manifest("a.flux", "id", "0", "100"),
            make_manifest("b.flux", "id", "200", "300"),
            make_manifest("c.flux", "id", "400", "500"),
        ];
        let groups = find_overlapping_groups(&manifests, &["id".into()]);
        // Each file is its own group (no merges needed).
        assert!(groups.iter().all(|g| g.len() == 1));
    }

    #[test]
    fn overlapping_files_form_group() {
        let manifests = vec![
            make_manifest("a.flux", "id", "0", "150"),
            make_manifest("b.flux", "id", "100", "250"),
            make_manifest("c.flux", "id", "200", "350"),
            make_manifest("d.flux", "id", "500", "600"), // disjoint
        ];
        let groups = find_overlapping_groups(&manifests, &["id".into()]);
        // a, b, c should form one group; d alone.
        let big_group = groups.iter().find(|g| g.len() == 3);
        assert!(big_group.is_some());
        let small_group = groups.iter().find(|g| g.len() == 1);
        assert!(small_group.is_some());
    }

    #[test]
    fn already_clustered_no_overlap() {
        let manifests = vec![
            make_manifest("a.flux", "ts", "2024-01", "2024-01"),
            make_manifest("b.flux", "ts", "2024-02", "2024-02"),
            make_manifest("c.flux", "ts", "2024-03", "2024-03"),
        ];
        let groups = find_overlapping_groups(&manifests, &["ts".into()]);
        // No merges — already well-clustered.
        assert!(groups.iter().all(|g| g.len() == 1));
    }
}
