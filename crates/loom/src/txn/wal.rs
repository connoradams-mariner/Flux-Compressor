// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Binary Write-Ahead Log (Parts 6 & 7 of the WAL roadmap).
//!
//! ## Phase 1 — Binary Log (`wal_v1`)
//!
//! An opt-in alternative to the one-JSON-file-per-transaction layout.
//! All log entries live in a single append-only file
//! `_flux_log/_wal.bin`, with per-entry framing:
//!
//! ```text
//! [u32 payload_len][payload: JSON bytes][u32 CRC32 of payload]
//! ```
//!
//! We use JSON for the payload (not MessagePack) to keep the wire
//! format forward-compatible with the existing
//! [`LogEntry::to_json`] representation and to avoid a new
//! dependency — the original roadmap called for `rmp-serde` but the
//! cost/benefit is minimal for now and JSON sidesteps the
//! schema-evolution concern of binary `Action`s.
//!
//! Activation: set `"log_format": "wal_v1"` in `_flux_meta.json`.
//! When the flag is absent or `"json"`, `FluxTable` keeps using the
//! one-file-per-commit layout (bit-identical with pre-WAL writers).
//!
//! ## Phase 2 — Checkpoints
//!
//! Every `checkpoint_interval` entries (default `100`, configurable
//! via the file-meta), a `.checkpoint-NNNNNNNN.json` file is written
//! next to the WAL. It holds a MessagePack-like JSON snapshot of the
//! resolved [`Snapshot`] so readers can fast-forward to a recent
//! version without replaying every preceding entry.
//!
//! Readers that open a WAL-backed table find the highest
//! checkpoint with `version ≤ target`, deserialise the snapshot, then
//! replay only the WAL tail.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{FluxError, FluxResult};

use super::log_entry::LogEntry;

/// Which on-disk log format a table is using.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Legacy: one `NNNNNNNN.json` file per transaction.
    Json,
    /// Opt-in Phase 1 binary WAL.
    WalV1,
}

impl Default for LogFormat {
    fn default() -> Self {
        LogFormat::Json
    }
}

impl LogFormat {
    /// Parse the `log_format` property value from `_flux_meta.json`.
    pub fn from_str_opt(s: Option<&str>) -> Self {
        match s.map(|v| v.to_ascii_lowercase()) {
            Some(v) if v == "wal_v1" || v == "wal" => LogFormat::WalV1,
            _ => LogFormat::Json,
        }
    }

    /// Return the string spelling that should appear in
    /// `_flux_meta.json` for this format.
    pub fn as_str(self) -> &'static str {
        match self {
            LogFormat::Json => "json",
            LogFormat::WalV1 => "wal_v1",
        }
    }
}

/// A single WAL entry, framed with its own payload-length and CRC32.
///
/// The framing format is described in the module docs. `WalEntry`
/// is a thin wrapper around [`LogEntry`] — the payload on disk is
/// the same JSON shape the legacy writer emits, which means any tool
/// that can read a JSON `NNNNNNNN.json` can decode a single WAL
/// entry once the frame is stripped.
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub entry: LogEntry,
    /// Byte offset of the framed record in the WAL file. Populated
    /// by [`WalLog::iter_entries`] to make checkpointing trivial.
    pub offset: u64,
}

/// Handle for an opt-in binary WAL at a specific directory.
///
/// Instances are cheap to build — they hold only paths, no file
/// descriptors, and every operation opens the WAL file fresh.
#[derive(Debug, Clone)]
pub struct WalLog {
    log_dir: PathBuf,
}

impl WalLog {
    /// Build a new WAL handle pointing at `log_dir`
    /// (typically `_flux_log/` under the table root).
    pub fn new(log_dir: impl AsRef<Path>) -> Self {
        Self {
            log_dir: log_dir.as_ref().to_path_buf(),
        }
    }

    /// Path of the append-only binary WAL file.
    pub fn wal_path(&self) -> PathBuf {
        self.log_dir.join("_wal.bin")
    }

    /// Path of the checkpoint file for `version`.
    pub fn checkpoint_path(&self, version: u64) -> PathBuf {
        self.log_dir
            .join(format!("_checkpoint-{:08}.json", version))
    }

    /// Does the WAL file exist?
    pub fn exists(&self) -> bool {
        self.wal_path().exists()
    }

    /// Append a [`LogEntry`] to the WAL. Returns the byte offset of
    /// the new frame (caller may stash this in a checkpoint record).
    pub fn append(&self, entry: &LogEntry) -> FluxResult<u64> {
        let payload = entry
            .to_json()
            .map_err(|e| FluxError::Internal(format!("wal append serialize: {e}")))?;
        let payload_bytes = payload.as_bytes();

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.wal_path())
            .map_err(FluxError::Io)?;

        let offset = file.metadata().map_err(FluxError::Io)?.len();

        let len = payload_bytes.len() as u32;
        file.write_all(&len.to_le_bytes()).map_err(FluxError::Io)?;
        file.write_all(payload_bytes).map_err(FluxError::Io)?;
        let crc = crc32fast::hash(payload_bytes);
        file.write_all(&crc.to_le_bytes()).map_err(FluxError::Io)?;
        file.flush().map_err(FluxError::Io)?;
        Ok(offset)
    }

    /// Iterate over every entry in the WAL in insertion order. Stops
    /// cleanly on the first framing error (a torn last record from
    /// a crashed write), which allows recovery: the committed tail is
    /// everything this iterator yielded.
    pub fn iter_entries(&self) -> FluxResult<Vec<WalEntry>> {
        let path = self.wal_path();
        if !path.exists() {
            return Ok(vec![]);
        }
        let file = File::open(&path).map_err(FluxError::Io)?;
        let mut reader = BufReader::new(file);
        let mut out = Vec::new();
        let mut offset = 0u64;
        let mut len_buf = [0u8; 4];
        loop {
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(FluxError::Io(e)),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() {
                // Torn record — stop cleanly.
                break;
            }
            let mut crc_buf = [0u8; 4];
            if reader.read_exact(&mut crc_buf).is_err() {
                break;
            }
            let got_crc = u32::from_le_bytes(crc_buf);
            let want_crc = crc32fast::hash(&payload);
            if got_crc != want_crc {
                // Corrupt record — stop before it; the log up to this
                // point is trusted.
                break;
            }
            let entry = LogEntry::from_json(&payload)
                .map_err(|e| FluxError::InvalidFile(format!("wal record deserialize: {e}")))?;
            out.push(WalEntry { entry, offset });
            offset += 4 + len as u64 + 4;
        }
        Ok(out)
    }

    /// Write a checkpoint file that snapshots the replayed state at
    /// the given `version`. The caller is responsible for providing a
    /// serde-compatible snapshot (we accept an opaque JSON value to
    /// avoid coupling `wal.rs` to `Snapshot`'s exact shape).
    pub fn write_checkpoint(
        &self,
        version: u64,
        snapshot_json: &serde_json::Value,
    ) -> FluxResult<()> {
        let path = self.checkpoint_path(version);
        let s = serde_json::to_string_pretty(snapshot_json)
            .map_err(|e| FluxError::Internal(format!("checkpoint serialize: {e}")))?;
        std::fs::write(&path, s.as_bytes()).map_err(FluxError::Io)?;
        Ok(())
    }

    /// Find the highest-version checkpoint file on disk, if any.
    pub fn latest_checkpoint_version(&self) -> FluxResult<Option<u64>> {
        if !self.log_dir.exists() {
            return Ok(None);
        }
        let mut best: Option<u64> = None;
        for entry in std::fs::read_dir(&self.log_dir).map_err(FluxError::Io)? {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("_checkpoint-") {
                if let Some(num_s) = rest.strip_suffix(".json") {
                    if let Ok(v) = num_s.parse::<u64>() {
                        best = Some(best.map(|b| b.max(v)).unwrap_or(v));
                    }
                }
            }
        }
        Ok(best)
    }

    /// Read a checkpoint's raw JSON value.
    pub fn read_checkpoint(&self, version: u64) -> FluxResult<serde_json::Value> {
        let path = self.checkpoint_path(version);
        let bytes = std::fs::read(&path).map_err(FluxError::Io)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| FluxError::InvalidFile(format!("checkpoint deserialize: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::txn::log_entry::Operation;
    use tempfile::TempDir;

    fn sample_entry(version: u64) -> LogEntry {
        LogEntry {
            version,
            timestamp_ms: 1_700_000_000_000 + version,
            operation: if version == 0 {
                Operation::Create
            } else {
                Operation::Append
            },
            data_files_added: vec![format!("data/part-{version:04}.flux")],
            data_files_removed: vec![],
            file_manifests: vec![],
            actions: vec![],
            row_count_delta: 100,
            metadata: Default::default(),
        }
    }

    #[test]
    fn wal_round_trip_with_framing() {
        let dir = TempDir::new().unwrap();
        let wal = WalLog::new(dir.path());

        for v in 0..5u64 {
            wal.append(&sample_entry(v)).unwrap();
        }
        let entries = wal.iter_entries().unwrap();
        assert_eq!(entries.len(), 5);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.entry.version, i as u64);
        }
    }

    #[test]
    fn wal_truncated_tail_is_skipped() {
        let dir = TempDir::new().unwrap();
        let wal = WalLog::new(dir.path());

        wal.append(&sample_entry(0)).unwrap();
        wal.append(&sample_entry(1)).unwrap();

        // Truncate the last few bytes to simulate a crash.
        let wal_path = wal.wal_path();
        let len = std::fs::metadata(&wal_path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&wal_path).unwrap();
        file.set_len(len - 2).unwrap();

        // We should still get the first entry cleanly.
        let entries = wal.iter_entries().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn wal_crc_mismatch_stops_replay() {
        let dir = TempDir::new().unwrap();
        let wal = WalLog::new(dir.path());

        wal.append(&sample_entry(0)).unwrap();
        wal.append(&sample_entry(1)).unwrap();

        // Corrupt the second entry's payload.
        let mut bytes = std::fs::read(wal.wal_path()).unwrap();
        let n = bytes.len();
        bytes[n - 10] ^= 0xFF;
        std::fs::write(wal.wal_path(), &bytes).unwrap();

        // Replay should stop at the first good entry.
        let entries = wal.iter_entries().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn checkpoint_round_trip() {
        let dir = TempDir::new().unwrap();
        let wal = WalLog::new(dir.path());

        let snap = serde_json::json!({"version": 42, "live_files": ["a.flux", "b.flux"]});
        wal.write_checkpoint(42, &snap).unwrap();

        assert_eq!(wal.latest_checkpoint_version().unwrap(), Some(42));
        let back = wal.read_checkpoint(42).unwrap();
        assert_eq!(back, snap);
    }

    #[test]
    fn log_format_parsing() {
        assert_eq!(LogFormat::from_str_opt(None), LogFormat::Json);
        assert_eq!(LogFormat::from_str_opt(Some("wal_v1")), LogFormat::WalV1);
        assert_eq!(LogFormat::from_str_opt(Some("WAL_V1")), LogFormat::WalV1);
        assert_eq!(LogFormat::from_str_opt(Some("foo")), LogFormat::Json);
    }
}
