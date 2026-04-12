// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `FluxTable` — directory-based versioned table with time-travel.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{FluxError, FluxResult};
use super::log_entry::{LogEntry, Operation};
use super::snapshot::Snapshot;

/// A FluxCompress table backed by a `.fluxtable/` directory.
///
/// Provides append, read, and time-travel operations via an immutable
/// transaction log.
#[derive(Debug)]
pub struct FluxTable {
    /// Root directory of the table.
    root: PathBuf,
}

impl FluxTable {
    /// Open or create a FluxTable at the given directory path.
    pub fn open(root: impl AsRef<Path>) -> FluxResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("_flux_log"))
            .map_err(|e| FluxError::Io(e))?;
        fs::create_dir_all(root.join("data"))
            .map_err(|e| FluxError::Io(e))?;
        Ok(Self { root })
    }

    /// Path to the transaction log directory.
    pub fn log_dir(&self) -> PathBuf {
        self.root.join("_flux_log")
    }

    /// Path to the data directory.
    pub fn data_dir(&self) -> PathBuf {
        self.root.join("data")
    }

    /// Read all log entries, sorted by version.
    pub fn read_log(&self) -> FluxResult<Vec<LogEntry>> {
        let log_dir = self.log_dir();
        let mut entries = Vec::new();

        if !log_dir.exists() {
            return Ok(entries);
        }

        let mut paths: Vec<PathBuf> = fs::read_dir(&log_dir)
            .map_err(|e| FluxError::Io(e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
            .collect();
        paths.sort();

        for path in paths {
            let data = fs::read(&path).map_err(|e| FluxError::Io(e))?;
            let entry = LogEntry::from_json(&data).map_err(|e| {
                FluxError::InvalidFile(format!("bad log entry {}: {e}", path.display()))
            })?;
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Get a snapshot at the latest version.
    pub fn snapshot(&self) -> FluxResult<Snapshot> {
        let log = self.read_log()?;
        Ok(Snapshot::latest(&log))
    }

    /// Get a snapshot at a specific version.
    pub fn snapshot_at_version(&self, version: u64) -> FluxResult<Snapshot> {
        let log = self.read_log()?;
        Ok(Snapshot::from_log(&log, version))
    }

    /// Get a snapshot at a specific timestamp (milliseconds since epoch).
    pub fn snapshot_at_timestamp(&self, timestamp_ms: u64) -> FluxResult<Snapshot> {
        let log = self.read_log()?;
        Ok(Snapshot::at_timestamp(&log, timestamp_ms))
    }

    /// Determine the next version number.
    pub fn next_version(&self) -> FluxResult<u64> {
        let log = self.read_log()?;
        Ok(log.last().map(|e| e.version + 1).unwrap_or(0))
    }

    /// Current timestamp in milliseconds.
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Append compressed data to the table.
    ///
    /// Writes the data as a new `.flux` file and creates a log entry.
    /// Returns the version number of the new entry.
    pub fn append(&self, flux_data: &[u8]) -> FluxResult<u64> {
        let version = self.next_version()?;
        let filename = format!("part-{:04}.flux", version);
        let data_path = self.data_dir().join(&filename);
        let relative = format!("data/{filename}");

        // Write data file.
        fs::write(&data_path, flux_data).map_err(|e| FluxError::Io(e))?;

        // Estimate row count from footer (best-effort).
        let row_count = crate::atlas::AtlasFooter::from_file_tail(flux_data)
            .map(|f| f.blocks.iter().map(|b| b.value_count as i64).sum())
            .unwrap_or(0);

        // Write log entry.
        let entry = LogEntry {
            version,
            timestamp_ms: Self::now_ms(),
            operation: if version == 0 { Operation::Create } else { Operation::Append },
            data_files_added: vec![relative],
            data_files_removed: vec![],
            row_count_delta: row_count,
            metadata: Default::default(),
        };

        let log_path = self.log_dir().join(entry.filename());
        let json = entry.to_json().map_err(|e| {
            FluxError::Internal(format!("serialize log entry: {e}"))
        })?;
        fs::write(&log_path, json.as_bytes()).map_err(|e| FluxError::Io(e))?;

        Ok(version)
    }

    /// List the live data file paths for the latest version.
    pub fn live_files(&self) -> FluxResult<Vec<PathBuf>> {
        let snap = self.snapshot()?;
        Ok(snap.live_files.iter().map(|f| self.root.join(f)).collect())
    }

    /// Read all live data files at the latest version and concatenate their bytes.
    pub fn read_all(&self) -> FluxResult<Vec<Vec<u8>>> {
        self.live_files()?
            .iter()
            .map(|p| fs::read(p).map_err(|e| FluxError::Io(e)))
            .collect()
    }

    /// Read all live data files at a specific version.
    pub fn read_at_version(&self, version: u64) -> FluxResult<Vec<Vec<u8>>> {
        let snap = self.snapshot_at_version(version)?;
        snap.live_files
            .iter()
            .map(|f| fs::read(self.root.join(f)).map_err(|e| FluxError::Io(e)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn create_and_append() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("test.fluxtable")).unwrap();

        // Append some fake flux data (just needs to be non-empty).
        // We'll use a minimal valid flux file: empty footer.
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes()); // block_count = 0
        let footer_len: u32 = 12; // 4 + 4 + 4
        data.extend_from_slice(&footer_len.to_le_bytes());
        data.extend_from_slice(&crate::FLUX_MAGIC.to_le_bytes());

        let v0 = table.append(&data).unwrap();
        assert_eq!(v0, 0);

        let v1 = table.append(&data).unwrap();
        assert_eq!(v1, 1);

        let snap = table.snapshot().unwrap();
        assert_eq!(snap.live_files.len(), 2);

        // Time travel: version 0 should have 1 file.
        let snap0 = table.snapshot_at_version(0).unwrap();
        assert_eq!(snap0.live_files.len(), 1);
    }
}
