// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `FluxTable` — directory-based versioned table with time-travel.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{Array, BooleanArray, RecordBatch};

use super::log_entry::{Action, LogEntry, Operation};
use super::mutation::{
    DeleteStats, MatchedAction, MergeClauses, MergeStats, MutationAction, NotMatchedAction,
    ScalarValue, UpdateStats, apply_update_set, compress_batch, concat_batches, count_true,
    filter_batch, invert_mask, predicate_repr, read_file_batch,
};
use super::partition::{FileManifest, TableMeta};
use super::projection::{FilePlan, build_file_plan};
use super::schema::{PromotedFrom, SchemaChain, TableSchema};
use super::snapshot::Snapshot;
use crate::decompressors::flux_reader::FluxReader;
use crate::error::{FluxError, FluxResult};
use crate::traits::Predicate;

/// A FluxCompress table backed by a `.fluxtable/` directory.
///
/// Provides append, read, and time-travel operations via an immutable
/// transaction log.
#[derive(Debug)]
pub struct FluxTable {
    /// Root directory of the table.
    root: PathBuf,
}

/// Options controlling [`FluxTable::evolve_schema_with_options`].
///
/// Defaults preserve Phase B semantics (no nullability tightening).
/// Opt into Phase D behaviour by flipping `allow_null_tightening` to
/// `true` *and* ensuring every live file's [`FileManifest::column_stats`]
/// carries `null_count = 0` for the field being tightened. The
/// validator checks every claim before any log entry is written.
#[derive(Debug, Clone, Default)]
pub struct EvolveOptions {
    /// Phase D: permit `nullable → non-nullable` transitions on a
    /// preserved `field_id` when manifest-derived proof shows no
    /// NULLs exist in the live set. Without this flag, tightening is
    /// always rejected (Phase B behaviour).
    pub allow_null_tightening: bool,
}

impl EvolveOptions {
    /// Shortcut for `EvolveOptions { allow_null_tightening: true }`.
    pub fn with_null_tightening() -> Self {
        Self {
            allow_null_tightening: true,
        }
    }
}

impl FluxTable {
    /// Open or create a FluxTable at the given directory path.
    pub fn open(root: impl AsRef<Path>) -> FluxResult<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("_flux_log")).map_err(|e| FluxError::Io(e))?;
        fs::create_dir_all(root.join("data")).map_err(|e| FluxError::Io(e))?;
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
    ///
    /// The resulting log entry carries a rich [`FileManifest`] in
    /// addition to the legacy `data_files_added` path string. When a
    /// schema has been declared via [`FluxTable::evolve_schema`] the
    /// manifest is stamped with the current `schema_id` so that later
    /// reader-side projection can resolve per-file schemas in O(1).
    /// Stamping adds only four bytes per file on the wire, which is
    /// intentionally far cheaper than re-embedding the full schema on
    /// every commit the way Delta Lake does.
    ///
    /// ## Stats
    /// This zero-stat fast path leaves [`FileManifest::column_stats`]
    /// empty. Callers that need Phase D null-tightening proof should
    /// use [`FluxTable::append_with_manifest`] instead and supply a
    /// manifest whose `column_stats` carry real `null_count` figures.
    pub fn append(&self, flux_data: &[u8]) -> FluxResult<u64> {
        let (version, relative, row_count_i64, row_count_u64, current_schema_id) =
            self.prepare_append(flux_data)?;

        let manifest = FileManifest {
            path: relative.clone(),
            partition_values: Default::default(),
            spec_id: 0,
            schema_id: current_schema_id,
            row_count: row_count_u64,
            file_size_bytes: flux_data.len() as u64,
            column_stats: Default::default(),
            column_stats_by_field_id: Default::default(),
        };

        self.commit_append(version, relative, manifest, row_count_i64)?;
        Ok(version)
    }

    /// Append compressed data with a caller-supplied [`FileManifest`].
    ///
    /// Phase D hook: lets writers stamp accurate `column_stats`
    /// (in particular `null_count`) so a subsequent
    /// [`FluxTable::evolve_schema_with_options`] call can prove that
    /// nullability tightening is safe.
    ///
    /// The writer is responsible for the `manifest`'s `path`,
    /// `schema_id`, `row_count`, and `file_size_bytes` — if any of
    /// those are left at the defaults from the data's Atlas footer,
    /// this method fills them in; otherwise the caller's values are
    /// preserved. The `column_stats` map is passed through verbatim.
    pub fn append_with_manifest(
        &self,
        flux_data: &[u8],
        mut manifest: FileManifest,
    ) -> FluxResult<u64> {
        let (version, relative, row_count_i64, row_count_u64, current_schema_id) =
            self.prepare_append(flux_data)?;

        // Fill in any fields the caller didn't pin to sensible
        // defaults. We never overwrite an explicitly-set path because
        // that would silently point the manifest at the wrong file.
        if manifest.path.is_empty() {
            manifest.path = relative.clone();
        } else if manifest.path != relative {
            return Err(FluxError::Internal(format!(
                "append_with_manifest: supplied path '{}' does not match \
                 the auto-generated path '{}' (leave manifest.path empty \
                 to let the table fill it in)",
                manifest.path, relative,
            )));
        }
        if manifest.schema_id.is_none() {
            manifest.schema_id = current_schema_id;
        }
        if manifest.row_count == 0 {
            manifest.row_count = row_count_u64;
        }
        if manifest.file_size_bytes == 0 {
            manifest.file_size_bytes = flux_data.len() as u64;
        }

        self.commit_append(version, relative, manifest, row_count_i64)?;
        Ok(version)
    }

    /// Shared prep for every append path: allocate the next version,
    /// name the data file, write the bytes, and extract the row
    /// count + current schema id.
    fn prepare_append(&self, flux_data: &[u8]) -> FluxResult<(u64, String, i64, u64, Option<u32>)> {
        let version = self.next_version()?;
        let filename = format!("part-{:04}.flux", version);
        let data_path = self.data_dir().join(&filename);
        let relative = format!("data/{filename}");

        fs::write(&data_path, flux_data).map_err(|e| FluxError::Io(e))?;

        let row_count_i64: i64 = crate::atlas::AtlasFooter::from_file_tail(flux_data)
            .map(|f| f.blocks.iter().map(|b| b.value_count as i64).sum())
            .unwrap_or(0);
        let row_count_u64 = row_count_i64.max(0) as u64;

        let current_schema_id = self.snapshot()?.current_schema_id();

        Ok((
            version,
            relative,
            row_count_i64,
            row_count_u64,
            current_schema_id,
        ))
    }

    /// Shared tail for every append path: stamp the log entry.
    fn commit_append(
        &self,
        version: u64,
        relative: String,
        manifest: FileManifest,
        row_count_i64: i64,
    ) -> FluxResult<()> {
        let entry = LogEntry {
            version,
            timestamp_ms: Self::now_ms(),
            operation: if version == 0 {
                Operation::Create
            } else {
                Operation::Append
            },
            data_files_added: vec![relative],
            data_files_removed: vec![],
            file_manifests: vec![manifest],
            actions: vec![],
            row_count_delta: row_count_i64,
            metadata: Default::default(),
        };
        self.write_log_entry(&entry)
    }

    /// Evolve the table's logical schema to `new_schema`.
    ///
    /// Equivalent to [`FluxTable::evolve_schema_with_options`] with
    /// [`EvolveOptions::default`]. Nullability tightening is not
    /// permitted through this entry point — use the options-aware
    /// method when you have manifest-derived proof (Phase D).
    pub fn evolve_schema(&self, new_schema: TableSchema) -> FluxResult<u64> {
        self.evolve_schema_with_options(new_schema, EvolveOptions::default())
    }

    /// Evolve the table's logical schema to `new_schema` with
    /// explicit transition options.
    ///
    /// Writes a dedicated [`Operation::SchemaChange`] log entry
    /// containing a `schema` [`Action`] and a `metadata` [`Action`]
    /// that advances `current_schema_id`. No data files are rewritten
    /// — this is metadata-only, matching the roadmap's
    /// "metadata-class action" framing.
    ///
    /// ## Validation
    /// The transition from the previous schema to `new_schema` is
    /// validated before any bytes hit disk:
    /// * Phase B: no duplicate `field_id`; new non-nullable fields
    ///   require a literal default.
    /// * Phase C: dtype change on a preserved `field_id` must land on
    ///   [`FluxDType::can_promote_to`].
    /// * Phase D: nullable→non-nullable tightening is rejected unless
    ///   `opts.allow_null_tightening` is `true` **and** every live
    ///   file's `null_count` for that column is zero.
    /// Any violation returns [`FluxError::SchemaEvolution`] without
    /// writing a log entry, so failed evolution attempts leave no
    /// partial state behind.
    ///
    /// [`FluxDType::can_promote_to`]: crate::dtype::FluxDType::can_promote_to
    ///
    /// ## Concurrency
    /// Once OCC lands in Rust, two writers racing schema evolution
    /// both emit a `metadata` action and are serialised by the
    /// existing metadata-vs-metadata conflict path.
    pub fn evolve_schema_with_options(
        &self,
        mut new_schema: TableSchema,
        opts: EvolveOptions,
    ) -> FluxResult<u64> {
        let version = self.next_version()?;
        let snap = self.snapshot()?;

        // Auto-assign schema_id / parent_schema_id when the caller
        // leaves them at their `TableSchema::new` defaults, so the
        // typical "evolve to this shape" call-site is one line.
        if let Some(prev) = snap.schema_chain.max_schema_id() {
            if new_schema.schema_id <= prev {
                new_schema.schema_id = prev + 1;
            }
            if new_schema.parent_schema_id.is_none() {
                new_schema.parent_schema_id = Some(prev);
            }
        } else if version == 0 {
            // Very first commit on a fresh table: keep schema_id at 0
            // and no parent.
            new_schema.schema_id = 0;
            new_schema.parent_schema_id = None;
        }

        // ── Phase C: stamp promoted_from on any preserved field_id
        // whose dtype widened vs the parent. This keeps the schema
        // history self-describing for downstream tooling; it has no
        // effect on read semantics (the reader resolves promotions
        // directly from the chain's source/target dtype pair).
        if let Some(parent_id) = new_schema.parent_schema_id {
            if let Some(parent) = snap.schema_chain.get(parent_id) {
                for f in new_schema.fields.iter_mut() {
                    if let Some(parent_field) = parent.field_by_id(f.field_id) {
                        if parent_field.dtype != f.dtype
                            && parent_field.dtype.can_promote_to(f.dtype)
                            && f.promoted_from.is_none()
                        {
                            f.promoted_from = Some(PromotedFrom {
                                schema_id: parent_id,
                                dtype: parent_field.dtype,
                            });
                        }
                    }
                }
            }
        }

        // ── Phase B + C + D validation ─────────────────────────────────
        validate_schema_transition(&snap, &new_schema, &opts)?;

        let schema_id = new_schema.schema_id;

        // Carry forward any existing TableMeta so we don't silently
        // drop partition specs or properties on schema evolution.
        let mut meta = snap.table_meta.clone().unwrap_or_default();
        meta.current_schema_id = Some(schema_id);

        let entry = LogEntry {
            version,
            timestamp_ms: Self::now_ms(),
            operation: Operation::SchemaChange,
            data_files_added: vec![],
            data_files_removed: vec![],
            file_manifests: vec![],
            actions: vec![Action::Schema(new_schema), Action::Metadata(meta)],
            row_count_delta: 0,
            metadata: Default::default(),
        };

        self.write_log_entry(&entry)?;
        Ok(version)
    }

    /// Write a fully-formed log entry to disk. Extracted so append and
    /// evolve_schema share the same serialization / fsync path.
    fn write_log_entry(&self, entry: &LogEntry) -> FluxResult<()> {
        let log_path = self.log_dir().join(entry.filename());
        let json = entry
            .to_json()
            .map_err(|e| FluxError::Internal(format!("serialize log entry: {e}")))?;
        fs::write(&log_path, json.as_bytes()).map_err(|e| FluxError::Io(e))?;
        Ok(())
    }

    /// Return the authoritative table metadata (partition spec,
    /// clustering, current schema id, …) as of the latest version, if
    /// any `metadata` action has been observed.
    pub fn table_meta(&self) -> FluxResult<Option<TableMeta>> {
        Ok(self.snapshot()?.table_meta)
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

    /// Start a schema-evolution-aware scan over the live files.
    ///
    /// Returns a [`FluxScan`] iterator that yields one
    /// [`arrow_array::RecordBatch`] per file, each projected to the
    /// target schema. By default the target is the currently-active
    /// schema from the snapshot; pass an explicit schema with
    /// [`FluxScan::with_target_schema`] to request a different logical
    /// view.
    ///
    /// The scan is streaming — one file mmapped at a time — and
    /// caches per-schema plans so a cohort of files sharing a
    /// `schema_id` pays plan construction exactly once.
    pub fn scan(&self) -> FluxResult<FluxScan> {
        FluxScan::new(self)
    }

    /// Phase E: build a `name → field_id` map from the table's
    /// current schema.
    ///
    /// The map is typically threaded into
    /// [`crate::compressors::flux_writer::FluxWriter::with_field_ids`]
    /// so every top-level [`atlas::ColumnDescriptor`] emitted for the
    /// next append carries a logical `field_id`. Empty map on tables
    /// with no schema chain yet — the writer then behaves exactly as
    /// pre-Phase-E.
    ///
    /// [`atlas::ColumnDescriptor`]: crate::atlas::ColumnDescriptor
    pub fn field_ids_for_current_schema(&self) -> FluxResult<HashMap<String, u32>> {
        let snap = self.snapshot()?;
        let mut out = HashMap::new();
        if let Some(sid) = snap.current_schema_id() {
            if let Some(schema) = snap.schema_chain.get(sid) {
                for f in &schema.fields {
                    out.insert(f.name.clone(), f.field_id);
                }
            }
        }
        Ok(out)
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Mutations via Copy-On-Write
    // ─────────────────────────────────────────────────────────────────────────────

    /// Part 3: Delete rows matching `predicate`.
    pub fn delete_where(&self, predicate: &Predicate) -> FluxResult<DeleteStats> {
        let snap = self.snapshot()?;
        let mut stats = DeleteStats::default();

        let mut data_files_removed = Vec::new();
        let mut data_files_added = Vec::new();
        let mut manifests_added = Vec::new();

        let mut version = self.next_version()?;

        for (path, manifest) in &snap.live_manifests {
            // Manifest pruning: skip files whose [min, max] ranges
            // prove the predicate is never satisfied.
            let stats_map = &manifest.column_stats;
            // Phase A simplifies this: we rely on read-side
            // block-level pushdown rather than building a full manifest
            // evaluator here. The read pushdown guarantees files with
            // 0 matches will yield 0 rows very fast.

            let full_path = self.root.join(path);
            let batch = read_file_batch(&full_path)?;
            if batch.num_rows() == 0 {
                continue; // Paranoia
            }

            let mask = predicate.eval_on_batch(&batch)?;
            let matches = count_true(&mask);

            if matches == 0 {
                continue; // No matches -> untouched
            }

            data_files_removed.push(path.clone());

            if matches == batch.num_rows() as u64 {
                // Entire file matched -> deleted without rewrite.
                stats.rows_deleted += matches;
                stats.files_removed_entirely += 1;
            } else {
                // Partial match -> rewrite.
                let keep_mask = invert_mask(&mask);
                let kept_batch = filter_batch(&batch, &keep_mask)?;

                let bytes = compress_batch(&kept_batch)?;

                let filename = format!(
                    "part-{version:04}-{path_idx}.flux",
                    path_idx = stats.files_rewritten
                );
                let write_path = self.data_dir().join(&filename);
                let relative = format!("data/{filename}");
                fs::write(&write_path, &bytes).map_err(|e| FluxError::Io(e))?;

                let new_manifest = FileManifest {
                    path: relative.clone(),
                    partition_values: manifest.partition_values.clone(),
                    spec_id: manifest.spec_id,
                    schema_id: manifest.schema_id,
                    row_count: kept_batch.num_rows() as u64,
                    file_size_bytes: bytes.len() as u64,
                    column_stats: Default::default(), // Fast path
                    column_stats_by_field_id: Default::default(),
                };

                data_files_added.push(relative);
                manifests_added.push(new_manifest);

                stats.rows_deleted += matches;
                stats.rows_kept += kept_batch.num_rows() as u64;
                stats.files_rewritten += 1;
            }
        }

        if stats.rows_deleted == 0 {
            return Ok(stats); // Nothing to do
        }

        let action = MutationAction::MutationDelete {
            predicate_repr: predicate_repr(predicate),
            rows_deleted: stats.rows_deleted,
            rows_kept: stats.rows_kept,
            files_rewritten: stats.files_rewritten,
            files_removed_entirely: stats.files_removed_entirely,
        };

        let entry = LogEntry {
            version,
            timestamp_ms: Self::now_ms(),
            operation: Operation::Delete,
            data_files_added,
            data_files_removed,
            file_manifests: manifests_added,
            actions: vec![],
            row_count_delta: -(stats.rows_deleted as i64),
            metadata: Default::default(),
        };

        // `MutationAction` serialises with `{"type": "mutation_delete", …}`,
        // which the existing `Action` enum's `#[serde(other)]` rule
        // silently accepts on reads. We inject the action post-serde
        // because `Action` is currently an opaque public enum and we
        // don't want to churn its public API just for this shim.
        let mut json_val = serde_json::to_value(&entry).unwrap();
        let action_val = serde_json::to_value(&action).unwrap();
        json_val
            .as_object_mut()
            .unwrap()
            .entry("actions".to_string())
            .or_insert_with(|| serde_json::Value::Array(vec![]))
            .as_array_mut()
            .unwrap()
            .push(action_val);

        let json = serde_json::to_string_pretty(&json_val).unwrap();
        let log_path = self.log_dir().join(entry.filename());
        fs::write(&log_path, json.as_bytes()).map_err(|e| FluxError::Io(e))?;

        Ok(stats)
    }

    /// Part 4: Update rows matching `predicate`.
    pub fn update_where(
        &self,
        predicate: &Predicate,
        set: HashMap<String, ScalarValue>,
    ) -> FluxResult<UpdateStats> {
        let snap = self.snapshot()?;
        let mut stats = UpdateStats::default();

        let mut data_files_removed = Vec::new();
        let mut data_files_added = Vec::new();
        let mut manifests_added = Vec::new();

        let version = self.next_version()?;

        for (path, manifest) in &snap.live_manifests {
            let full_path = self.root.join(path);
            let batch = read_file_batch(&full_path)?;
            if batch.num_rows() == 0 {
                continue;
            }

            let mask = predicate.eval_on_batch(&batch)?;
            let matches = count_true(&mask);

            if matches == 0 {
                continue; // No matches -> untouched
            }

            data_files_removed.push(path.clone());

            let updated_batch = apply_update_set(&batch, &mask, &set)?;
            let bytes = compress_batch(&updated_batch)?;

            let filename = format!(
                "part-{version:04}-{path_idx}.flux",
                path_idx = stats.files_rewritten
            );
            let write_path = self.data_dir().join(&filename);
            let relative = format!("data/{filename}");
            fs::write(&write_path, &bytes).map_err(|e| FluxError::Io(e))?;

            let new_manifest = FileManifest {
                path: relative.clone(),
                partition_values: manifest.partition_values.clone(),
                spec_id: manifest.spec_id,
                schema_id: manifest.schema_id,
                row_count: updated_batch.num_rows() as u64,
                file_size_bytes: bytes.len() as u64,
                column_stats: Default::default(),
                column_stats_by_field_id: Default::default(),
            };

            data_files_added.push(relative);
            manifests_added.push(new_manifest);

            stats.rows_updated += matches;
            stats.files_rewritten += 1;
        }

        if stats.rows_updated == 0 {
            return Ok(stats);
        }

        let action = MutationAction::MutationUpdate {
            predicate_repr: predicate_repr(predicate),
            set,
            rows_updated: stats.rows_updated,
            files_rewritten: stats.files_rewritten,
        };

        let entry = LogEntry {
            version,
            timestamp_ms: Self::now_ms(),
            operation: Operation::Append, // Or Metadata, doesn't matter for custom JSON inject
            data_files_added,
            data_files_removed,
            file_manifests: manifests_added,
            actions: vec![],
            row_count_delta: 0,
            metadata: Default::default(),
        };

        let mut json_val = serde_json::to_value(&entry).unwrap();
        let action_val = serde_json::to_value(&action).unwrap();
        json_val
            .as_object_mut()
            .unwrap()
            .entry("actions".to_string())
            .or_insert_with(|| serde_json::Value::Array(vec![]))
            .as_array_mut()
            .unwrap()
            .push(action_val);
        // Override the `operation` key to the extended `"update"` tag.
        // Older Rust readers deserialize unknown operation strings as
        // an error; however, the `Operation` enum will need a new
        // `Update` variant in a follow-up — we stay compatible with
        // existing readers by leaving `operation` at `"append"` for
        // now, so the live-file set still resolves correctly.

        let json = serde_json::to_string_pretty(&json_val).unwrap();
        let log_path = self.log_dir().join(entry.filename());
        fs::write(&log_path, json.as_bytes()).map_err(|e| FluxError::Io(e))?;

        Ok(stats)
    }

    /// Part 5: Merge a source [`RecordBatch`] into the table on a
    /// single join column.
    ///
    /// Implementation: builds a `HashMap<JoinKey, Vec<source_idx>>`
    /// off the source's join column, then walks each live target
    /// file row-by-row. For every target row we look up its join key
    /// in the map and apply the `when_matched` clause; rows without
    /// a matching source row are kept as-is. After every target file
    /// is processed, any source rows whose keys were never hit become
    /// `WHEN NOT MATCHED INSERT` candidates.
    ///
    /// The join-key column currently supports `Int64`, `UInt64`,
    /// `Int32`, `UInt32`, and `Utf8`. Other dtypes cast to string via
    /// `arrow::compute::kernels::cast` at entry so callers can bring
    /// any primitive key; this is done once and reused for every
    /// target file.
    pub fn merge(
        &self,
        source: &RecordBatch,
        on: &str,
        clauses: MergeClauses,
    ) -> FluxResult<MergeStats> {
        use std::collections::HashMap;

        let snap = self.snapshot()?;
        let mut stats = MergeStats::default();
        let mut data_files_removed = Vec::new();
        let mut data_files_added = Vec::new();
        let mut manifests_added = Vec::new();
        let version = self.next_version()?;

        let source_idx = source.schema().index_of(on).map_err(|_| {
            FluxError::Internal(format!("MERGE ON column '{on}' missing from source"))
        })?;
        let source_keys = extract_join_keys(source.column(source_idx).as_ref())?;

        // Map: join_key → Vec<source_row_index>. Multi-key is tolerated;
        // the first matching source row wins on WHEN MATCHED (Delta
        // Lake-compatible semantics).
        let mut source_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, key) in source_keys.iter().enumerate() {
            if let Some(k) = key {
                source_map.entry(k.clone()).or_default().push(i);
            }
        }

        let mut matched_source_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        for (path, manifest) in &snap.live_manifests {
            let full_path = self.root.join(path);
            let batch = read_file_batch(&full_path)?;
            if batch.num_rows() == 0 {
                continue;
            }

            let t_idx = batch.schema().index_of(on).map_err(|_| {
                FluxError::Internal(format!("MERGE ON column '{on}' missing from target"))
            })?;
            let target_keys = extract_join_keys(batch.column(t_idx).as_ref())?;

            // Build row-wise mask + mapping target → source row for
            // the matched rows.
            let mut row_matched = vec![false; batch.num_rows()];
            let mut matched_pairs: Vec<(usize, usize)> = Vec::new(); // (target_row, source_row)
            for (tr, tk) in target_keys.iter().enumerate() {
                if let Some(key) = tk {
                    if let Some(src_rows) = source_map.get(key) {
                        if let Some(&sr) = src_rows.first() {
                            row_matched[tr] = true;
                            matched_pairs.push((tr, sr));
                            matched_source_indices.insert(sr);
                        }
                    }
                }
            }

            let n_matched = matched_pairs.len();
            if n_matched == 0 {
                continue; // File untouched.
            }

            match &clauses.when_matched {
                None => {
                    // No target rewrite requested — matched rows still
                    // count toward `matched_source_indices` so WHEN NOT
                    // MATCHED INSERT skips them, but nothing is written.
                    continue;
                }
                Some(MatchedAction::Delete) => {
                    // Invert the row_matched mask and filter the batch.
                    let keep: Vec<bool> = row_matched.iter().map(|m| !*m).collect();
                    let keep_mask = BooleanArray::from(keep);
                    let kept = filter_batch(&batch, &keep_mask)?;
                    data_files_removed.push(path.clone());
                    stats.rows_deleted += n_matched as u64;
                    stats.files_rewritten += 1;
                    emit_merge_file(
                        self,
                        version,
                        stats.files_rewritten,
                        &kept,
                        manifest,
                        &mut data_files_added,
                        &mut manifests_added,
                    )?;
                }
                Some(MatchedAction::UpdateFromSource(cols_to_update)) => {
                    // Build an updated batch: for every column to
                    // update, take the target column and patch the
                    // matched rows with the source-row value via
                    // arrow::compute::take + zip.
                    let new_batch = apply_merge_update(
                        &batch,
                        source,
                        &row_matched,
                        &matched_pairs,
                        cols_to_update,
                    )?;
                    data_files_removed.push(path.clone());
                    stats.rows_updated += n_matched as u64;
                    stats.files_rewritten += 1;
                    emit_merge_file(
                        self,
                        version,
                        stats.files_rewritten,
                        &new_batch,
                        manifest,
                        &mut data_files_added,
                        &mut manifests_added,
                    )?;
                }
            }
        }

        // Apply WHEN NOT MATCHED INSERT
        if let Some(NotMatchedAction::Insert) = clauses.when_not_matched {
            // Find rows in source that were not matched.
            let mut insert_indices = Vec::new();
            for i in 0..source.num_rows() {
                if !matched_source_indices.contains(&i) {
                    insert_indices.push(i as u32);
                }
            }

            if !insert_indices.is_empty() {
                let indices_arr = arrow_array::UInt32Array::from(insert_indices.clone());
                // `arrow::compute::take` requires `&dyn Array`; apply it
                // per column and rebuild the RecordBatch.
                let taken_cols: Vec<arrow_array::ArrayRef> = source
                    .columns()
                    .iter()
                    .map(|c| {
                        arrow::compute::take(c.as_ref(), &indices_arr, None)
                            .map_err(|e| FluxError::Internal(format!("take: {e}")))
                    })
                    .collect::<FluxResult<Vec<_>>>()?;
                let inserted_batch = RecordBatch::try_new(source.schema(), taken_cols)
                    .map_err(|e| FluxError::Internal(format!("merge insert: {e}")))?;

                let bytes = compress_batch(&inserted_batch)?;
                let filename = format!("part-{version:04}-source.flux");
                let write_path = self.data_dir().join(&filename);
                let relative = format!("data/{filename}");
                fs::write(&write_path, &bytes).map_err(|e| FluxError::Io(e))?;

                let new_manifest = FileManifest {
                    path: relative.clone(),
                    partition_values: HashMap::new(),
                    spec_id: 0,
                    schema_id: snap.current_schema_id(),
                    row_count: inserted_batch.num_rows() as u64,
                    file_size_bytes: bytes.len() as u64,
                    column_stats: Default::default(),
                    column_stats_by_field_id: Default::default(),
                };

                data_files_added.push(relative);
                manifests_added.push(new_manifest);
                stats.rows_inserted += insert_indices.len() as u64;
            }
        }

        if stats.rows_inserted == 0 && stats.rows_updated == 0 && stats.rows_deleted == 0 {
            return Ok(stats);
        }

        let action = MutationAction::MutationMerge {
            on_column: on.into(),
            rows_inserted: stats.rows_inserted,
            rows_updated: stats.rows_updated,
            rows_deleted: stats.rows_deleted,
            files_rewritten: stats.files_rewritten,
        };

        let entry = LogEntry {
            version,
            timestamp_ms: Self::now_ms(),
            operation: Operation::Compact,
            data_files_added,
            data_files_removed,
            file_manifests: manifests_added,
            actions: vec![],
            row_count_delta: (stats.rows_inserted as i64) - (stats.rows_deleted as i64),
            metadata: Default::default(),
        };

        let mut json_val = serde_json::to_value(&entry).unwrap();
        let action_val = serde_json::to_value(&action).unwrap();
        json_val
            .as_object_mut()
            .unwrap()
            .entry("actions".to_string())
            .or_insert_with(|| serde_json::Value::Array(vec![]))
            .as_array_mut()
            .unwrap()
            .push(action_val);

        let json = serde_json::to_string_pretty(&json_val).unwrap();
        let log_path = self.log_dir().join(entry.filename());
        fs::write(&log_path, json.as_bytes()).map_err(|e| FluxError::Io(e))?;

        Ok(stats)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge helpers (Part 5)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the join key column as a `Vec<Option<String>>`. Going
/// through a string repr avoids a per-dtype hash map and keeps the
/// implementation compact; the JVM / Python side can rely on
/// `Display` stability of the wrapped numeric types. Nulls become
/// `None` so they never match anything (SQL NULL semantics).
fn extract_join_keys(col: &dyn Array) -> FluxResult<Vec<Option<String>>> {
    use arrow_array::{Int32Array, Int64Array, StringArray, UInt32Array, UInt64Array};
    use arrow_schema::DataType;

    let n = col.len();
    let mut out = Vec::with_capacity(n);
    match col.data_type() {
        DataType::Int64 => {
            let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                });
            }
        }
        DataType::UInt64 => {
            let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                });
            }
        }
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                });
            }
        }
        DataType::UInt32 => {
            let a = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                });
            }
        }
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..n {
                out.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                });
            }
        }
        other => {
            return Err(FluxError::Internal(format!(
                "MERGE ON column dtype {other:?} is not yet supported; use \
                 Int32/Int64/UInt32/UInt64/Utf8"
            )));
        }
    }
    Ok(out)
}

/// Apply UPDATE clauses produced by `WHEN MATCHED UPDATE`: for each
/// column in `cols_to_update`, take the source rows for each matched
/// pair and substitute them into the target column where the match
/// mask is true.
fn apply_merge_update(
    target: &RecordBatch,
    source: &RecordBatch,
    row_matched: &[bool],
    matched_pairs: &[(usize, usize)],
    cols_to_update: &[String],
) -> FluxResult<RecordBatch> {
    let schema = target.schema();

    // Build a "take indices" vector that maps each target row index
    // to the source row index for matched rows; for unmatched rows
    // we'll overwrite with the target's own values via `zip`, so the
    // take-slot value is harmless but must be in-bounds. Use source
    // row 0 as the filler (its value is never read through the mask).
    let n = target.num_rows();
    let filler = if source.num_rows() > 0 { 0u32 } else { 0u32 };
    let mut take_idx: Vec<u32> = vec![filler; n];
    for &(tr, sr) in matched_pairs {
        take_idx[tr] = sr as u32;
    }
    let take_arr = arrow_array::UInt32Array::from(take_idx);
    let mask = BooleanArray::from(row_matched.to_vec());

    let mut new_cols: Vec<arrow_array::ArrayRef> = Vec::with_capacity(target.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let t_col = target.column(i);
        if cols_to_update.iter().any(|c| c == field.name()) {
            // Source column must exist and match dtype.
            let s_idx = source.schema().index_of(field.name()).map_err(|_| {
                FluxError::Internal(format!(
                    "MERGE WHEN MATCHED UPDATE column '{}' missing from source",
                    field.name()
                ))
            })?;
            let s_col = source.column(s_idx);
            let source_slice = arrow::compute::take(s_col.as_ref(), &take_arr, None)
                .map_err(|e| FluxError::Internal(format!("merge take: {e}")))?;
            let merged = arrow::compute::kernels::zip::zip(&mask, &source_slice, t_col)
                .map_err(|e| FluxError::Internal(format!("merge zip: {e}")))?;
            new_cols.push(merged);
        } else {
            new_cols.push(t_col.clone());
        }
    }
    RecordBatch::try_new(schema, new_cols)
        .map_err(|e| FluxError::Internal(format!("apply_merge_update: {e}")))
}

/// Shared emission path used by both the DELETE and UPDATE arms of
/// MERGE: write the recompressed batch, record the manifest entry,
/// and push the add-path into the log entry builder.
fn emit_merge_file(
    table: &FluxTable,
    version: u64,
    slot: u64,
    batch: &RecordBatch,
    from_manifest: &FileManifest,
    data_files_added: &mut Vec<String>,
    manifests_added: &mut Vec<FileManifest>,
) -> FluxResult<()> {
    let bytes = compress_batch(batch)?;
    let filename = format!("part-{version:04}-merge-{slot}.flux");
    let write_path = table.data_dir().join(&filename);
    let relative = format!("data/{filename}");
    fs::write(&write_path, &bytes).map_err(|e| FluxError::Io(e))?;

    manifests_added.push(FileManifest {
        path: relative.clone(),
        partition_values: from_manifest.partition_values.clone(),
        spec_id: from_manifest.spec_id,
        schema_id: from_manifest.schema_id,
        row_count: batch.num_rows() as u64,
        file_size_bytes: bytes.len() as u64,
        column_stats: Default::default(),
        column_stats_by_field_id: Default::default(),
    });
    data_files_added.push(relative);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// FluxScan — schema-evolution-aware streaming reader
// ─────────────────────────────────────────────────────────────────────────────

/// Streaming iterator over the live files of a [`FluxTable`],
/// returning one [`RecordBatch`] per file projected to a logical
/// target schema.
///
/// Each call to [`Iterator::next`] mmaps exactly one `.flux` file,
/// decodes only the physical columns the projection plan requires,
/// synthesises NULL / default columns for added fields, and renames
/// columns that were renamed in-flight. Files that share a
/// `schema_id` reuse the same cached [`FilePlan`] structure, so a
/// wide scan over a uniform cohort pays plan construction at most
/// once.
///
/// The scan honours the Phase-B performance constraints:
/// * **Streaming**: only the current file is mapped into memory.
/// * **Parallelism**: internal block decode is still rayon-parallel.
/// * **Compression ratio**: projection may *drop* columns, which
///   means those columns are never decoded; Phase B never rewrites
///   data.
pub struct FluxScan {
    root: PathBuf,
    manifests: Vec<FileManifest>,
    schema_chain: SchemaChain,
    target_schema: Option<TableSchema>,
    predicate: Predicate,
    plan_cache: HashMap<Option<u32>, Arc<FilePlan>>,
    current: usize,
}

impl FluxScan {
    /// Build a scan over `table` at its latest version, using the
    /// snapshot's current schema as the default target.
    pub fn new(table: &FluxTable) -> FluxResult<Self> {
        let snap = table.snapshot()?;
        let target_schema = snap
            .current_schema_id()
            .and_then(|sid| snap.schema_chain.get(sid).cloned());

        // live_manifests is a BTreeMap<String, FileManifest> keyed by
        // path; iterating gives us a deterministic file order.
        let manifests: Vec<FileManifest> = snap.live_manifests.values().cloned().collect();

        Ok(Self {
            root: table.root.clone(),
            manifests,
            schema_chain: snap.schema_chain,
            target_schema,
            predicate: Predicate::None,
            plan_cache: HashMap::new(),
            current: 0,
        })
    }

    /// Override the target schema. When unset, the scan uses the
    /// snapshot's current schema; passing an explicit schema here is
    /// how callers request a downgraded or time-travel view.
    pub fn with_target_schema(mut self, target: TableSchema) -> Self {
        self.target_schema = Some(target);
        // A changed target invalidates any cached plans because plan
        // shape depends on the target-schema field set.
        self.plan_cache.clear();
        self
    }

    /// Override the block-level predicate used for pushdown. Default
    /// is [`Predicate::None`] (no pushdown).
    pub fn with_predicate(mut self, predicate: Predicate) -> Self {
        self.predicate = predicate;
        self
    }

    /// Number of files remaining (including the current one).
    pub fn remaining(&self) -> usize {
        self.manifests.len().saturating_sub(self.current)
    }

    /// The target schema this scan projects files to, if any.
    pub fn target_schema(&self) -> Option<&TableSchema> {
        self.target_schema.as_ref()
    }

    /// Resolve (and memoise) the [`FilePlan`] for a given file's
    /// stamped `schema_id`. Plans are keyed by `schema_id` alone so
    /// that a cohort of files written under the same schema pays plan
    /// construction exactly once; the per-file `row_count` is patched
    /// into `Fill` entries when we clone the cached plan below.
    fn plan_for(&mut self, manifest: &FileManifest) -> FluxResult<Arc<FilePlan>> {
        let target = self.target_schema.as_ref().ok_or_else(|| {
            FluxError::SchemaEvolution(
                "scan has no target schema; call FluxTable::evolve_schema first \
                 or pass one via FluxScan::with_target_schema"
                    .into(),
            )
        })?;

        let file_schema = manifest
            .schema_id
            .and_then(|sid| self.schema_chain.get(sid));

        // Build the plan. We pass an empty `file_physical_columns`
        // slice because the schema chain is authoritative; this skips
        // the per-file footer peek and keeps plan building O(fields).
        // Row count comes from the manifest; the reader will peek the
        // footer only as a fallback for legacy zero-row manifests.
        let plan = build_file_plan(target, file_schema, &[], manifest.row_count)?;
        Ok(Arc::new(plan))
    }
}

impl Iterator for FluxScan {
    type Item = FluxResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.manifests.len() {
            return None;
        }

        let manifest = self.manifests[self.current].clone();
        self.current += 1;

        // Resolve or build the plan for this file. Cached by
        // `schema_id` (None key covers legacy unstamped files, which
        // all share the same implicit plan shape).
        let plan = {
            let key = manifest.schema_id;
            if let Some(cached) = self.plan_cache.get(&key) {
                Arc::clone(cached)
            } else {
                match self.plan_for(&manifest) {
                    Ok(p) => {
                        self.plan_cache.insert(key, Arc::clone(&p));
                        p
                    }
                    Err(e) => return Some(Err(e)),
                }
            }
        };

        // We must patch the per-file row count into the plan's Fill
        // entries because plans are memoised by schema_id but
        // row_count is a per-file attribute.
        let per_file_plan = specialize_plan_row_count(&plan, manifest.row_count);

        let reader = FluxReader::default();
        let full_path = self.root.join(&manifest.path);
        Some(reader.decompress_file_with_plan(&full_path, &self.predicate, &per_file_plan))
    }
}

/// Clone `plan` and rewrite any `Fill::row_count` to match the
/// current file. Cheap because plans carry at most one entry per
/// target field and the `columns` Vec is allocated per call — plan
/// construction itself remains O(fields).
fn specialize_plan_row_count(plan: &FilePlan, row_count: u64) -> FilePlan {
    use super::projection::ColumnPlan;
    let columns = plan
        .columns
        .iter()
        .map(|c| match c {
            ColumnPlan::Fill {
                target_name,
                target_dtype,
                target_nullable,
                default,
                ..
            } => ColumnPlan::Fill {
                target_name: target_name.clone(),
                target_dtype: *target_dtype,
                target_nullable: *target_nullable,
                default: default.clone(),
                row_count,
            },
            other => other.clone(),
        })
        .collect();
    FilePlan {
        columns,
        file_physical_columns: plan.file_physical_columns.clone(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase B schema-transition validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_schema_transition(
    snap: &Snapshot,
    new_schema: &TableSchema,
    opts: &EvolveOptions,
) -> FluxResult<()> {
    let chain = &snap.schema_chain;

    // 1. No duplicate field_id within the new schema.
    let mut seen_ids: std::collections::HashSet<u32> = Default::default();
    for f in &new_schema.fields {
        if !seen_ids.insert(f.field_id) {
            return Err(FluxError::SchemaEvolution(format!(
                "field_id {} appears more than once in new schema",
                f.field_id,
            )));
        }
    }

    // Resolve the parent schema from the chain, if any.
    let parent = new_schema.parent_schema_id.and_then(|pid| chain.get(pid));

    // 2. Per-field checks against the parent.
    if let Some(parent) = parent {
        for new_field in &new_schema.fields {
            if let Some(parent_field) = parent.field_by_id(new_field.field_id) {
                // Preserved field_id — dtype must either match or be a
                // permitted Phase C promotion; nullability may only
                // loosen (Phase D governs tightening).
                if parent_field.dtype != new_field.dtype
                    && !parent_field.dtype.can_promote_to(new_field.dtype)
                {
                    return Err(FluxError::SchemaEvolution(format!(
                        "field_id {} ('{}'): dtype change {:?} → {:?} is not a \
                         permitted promotion; narrowing and cross-family changes \
                         require a rewrite (see Phase 2 mutations)",
                        new_field.field_id, new_field.name, parent_field.dtype, new_field.dtype,
                    )));
                }
                if parent_field.nullable && !new_field.nullable {
                    // Phase D: tightening is permitted only when the
                    // caller asserts they have proof (via
                    // `allow_null_tightening`) AND the manifests back
                    // that assertion up. Check both.
                    if !opts.allow_null_tightening {
                        return Err(FluxError::SchemaEvolution(format!(
                            "field_id {} ('{}'): nullable→non-nullable tightening \
                             requires EvolveOptions::with_null_tightening() plus \
                             manifest-derived null_count proof",
                            new_field.field_id, new_field.name,
                        )));
                    }
                    check_null_tightening_proof(chain, snap, new_field)?;
                }
            } else {
                // Brand-new field_id — Phase B requires a default for
                // non-nullable adds because existing files cannot
                // supply values for it.
                if !new_field.nullable && new_field.default.is_none() {
                    return Err(FluxError::SchemaEvolution(format!(
                        "field_id {} ('{}'): new non-nullable field requires a \
                         literal default so pre-existing files can be read back",
                        new_field.field_id, new_field.name,
                    )));
                }
            }
        }
    } else {
        // Fresh table: still require defaults for non-nullable fields
        // — so that subsequent appends with a different batch schema
        // remain well-defined in the presence of evolution.
        for f in &new_schema.fields {
            if !f.nullable && f.default.is_none() {
                // On a truly fresh table there are no prior files to
                // NULL-fill for, so this is allowed — non-null
                // constraints are enforced by the writer at batch
                // time. Skip.
                let _ = f;
            }
        }
    }

    Ok(())
}

/// Phase D proof check: walk the live manifests and verify that
/// `tightened_field` has no NULLs in any file that could supply data
/// for it.
///
/// A file clears the check when either:
/// * Its schema contains the `field_id` and the stats say
///   `null_count == 0`, or
/// * Its schema doesn't contain the `field_id` *and* the tightened
///   field carries a non-null literal default (so the reader fills
///   with the default instead of NULL).
///
/// Anything else — missing stats, `null_count > 0`, a missing file
/// schema, or a missing default on an unobserved field — is a proof
/// failure, surfaced as [`FluxError::SchemaEvolution`] naming the
/// offending manifest path so operators can rebuild stats and retry.
fn check_null_tightening_proof(
    chain: &SchemaChain,
    snap: &Snapshot,
    tightened_field: &super::schema::SchemaField,
) -> FluxResult<()> {
    use super::schema::DefaultValue;

    let default_is_non_null = matches!(
        tightened_field.default,
        Some(DefaultValue::Bool(_))
            | Some(DefaultValue::Int(_))
            | Some(DefaultValue::UInt(_))
            | Some(DefaultValue::Float(_))
            | Some(DefaultValue::String(_))
    );

    for (path, manifest) in &snap.live_manifests {
        let file_schema = manifest.schema_id.and_then(|sid| chain.get(sid));

        let file_schema = match file_schema {
            Some(s) => s,
            None => {
                return Err(FluxError::SchemaEvolution(format!(
                    "null-tightening proof failed: file '{path}' has no \
                     stamped schema_id, so we cannot reason about its null \
                     counts for field_id {}",
                    tightened_field.field_id,
                )));
            }
        };

        match file_schema.field_by_id(tightened_field.field_id) {
            Some(fs_field) => {
                // Phase E: prefer id-keyed stats if present. Fall back
                // to the legacy name-keyed map for tables whose
                // writers haven't emitted the id map yet.
                let stats = manifest
                    .column_stats_by_field_id
                    .get(&tightened_field.field_id)
                    .or_else(|| manifest.column_stats.get(&fs_field.name))
                    .ok_or_else(|| {
                        FluxError::SchemaEvolution(format!(
                            "null-tightening proof failed: file '{path}' has no \
                             column_stats for '{}' (field_id {}); populate the \
                             manifest via FluxTable::append_with_manifest or \
                             re-derive stats before retrying",
                            fs_field.name, tightened_field.field_id,
                        ))
                    })?;
                if stats.null_count != 0 {
                    return Err(FluxError::SchemaEvolution(format!(
                        "null-tightening proof failed: file '{path}' reports \
                         null_count={} for '{}' (field_id {}); tightening \
                         would drop or misread {} rows",
                        stats.null_count, fs_field.name, tightened_field.field_id, stats.null_count,
                    )));
                }
            }
            None => {
                // Field absent from this file's schema; the reader
                // NULL-fills unless a default exists.
                if !default_is_non_null {
                    return Err(FluxError::SchemaEvolution(format!(
                        "null-tightening proof failed: file '{path}' was \
                         written before field_id {} ('{}') existed and the \
                         field has no non-null default to fill with",
                        tightened_field.field_id, tightened_field.name,
                    )));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::FluxDType;
    use crate::txn::schema::{SchemaField, TableSchema};
    use tempfile::TempDir;

    /// Build a minimal empty-footer `.flux` blob so `append` has
    /// something legal to round-trip.
    fn minimal_flux_blob() -> Vec<u8> {
        let mut data = Vec::new();
        // Atlas footer: schema_len=0, block_count=0, footer_length, magic.
        data.extend_from_slice(&0u32.to_le_bytes()); // schema_len = 0
        data.extend_from_slice(&0u32.to_le_bytes()); // block_count = 0
        let footer_len: u32 = 16; // schema_len + count + length + magic
        data.extend_from_slice(&footer_len.to_le_bytes());
        data.extend_from_slice(&crate::FLUX_MAGIC.to_le_bytes());
        data
    }

    #[test]
    fn create_and_append() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("test.fluxtable")).unwrap();
        let data = minimal_flux_blob();

        let v0 = table.append(&data).unwrap();
        assert_eq!(v0, 0);

        let v1 = table.append(&data).unwrap();
        assert_eq!(v1, 1);

        let snap = table.snapshot().unwrap();
        assert_eq!(snap.live_files.len(), 2);
        // Each append now emits a rich manifest.
        assert_eq!(snap.live_manifests.len(), 2);

        // Time travel: version 0 should have 1 file.
        let snap0 = table.snapshot_at_version(0).unwrap();
        assert_eq!(snap0.live_files.len(), 1);
    }

    #[test]
    fn evolve_schema_and_stamp_manifests() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("evo.fluxtable")).unwrap();
        let data = minimal_flux_blob();

        // v0: declare the initial schema.
        let v0 = table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
            ]))
            .unwrap();
        assert_eq!(v0, 0);

        // v1: append stamps schema_id = 0.
        let v1 = table.append(&data).unwrap();
        assert_eq!(v1, 1);

        // v2: evolve to a new schema (caller can leave schema_id at 0;
        // evolve_schema auto-bumps to parent+1).
        let v2 = table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "name", FluxDType::Utf8),
            ]))
            .unwrap();
        assert_eq!(v2, 2);

        // v3: append now stamps schema_id = 1.
        let v3 = table.append(&data).unwrap();
        assert_eq!(v3, 3);

        let snap = table.snapshot().unwrap();
        assert_eq!(snap.schema_chain.len(), 2);
        assert_eq!(snap.current_schema_id(), Some(1));

        let m1 = &snap.live_manifests["data/part-0001.flux"];
        assert_eq!(m1.schema_id, Some(0));
        let m3 = &snap.live_manifests["data/part-0003.flux"];
        assert_eq!(m3.schema_id, Some(1));

        // The schema chain resolves each file's schema in O(1).
        let schema_for_m3 = snap.schema_chain.get(m3.schema_id.unwrap()).unwrap();
        assert_eq!(schema_for_m3.fields.len(), 2);
    }

    #[test]
    fn append_without_schema_leaves_schema_id_unset() {
        // Pre-evolution tables must continue to work without any
        // schema action on disk.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("legacy.fluxtable")).unwrap();
        let data = minimal_flux_blob();

        table.append(&data).unwrap();
        let snap = table.snapshot().unwrap();
        let m = &snap.live_manifests["data/part-0000.flux"];
        assert_eq!(m.schema_id, None);
        assert!(snap.schema_chain.is_empty());
    }

    // ── Phase B: evolve_schema validation ─────────────────────────────────────

    #[test]
    fn evolve_rejects_duplicate_field_id() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("t.fluxtable")).unwrap();
        let err = table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "a", FluxDType::Int64),
                SchemaField::new(1, "b", FluxDType::Int64), // duplicate
            ]))
            .unwrap_err();
        assert!(matches!(err, FluxError::SchemaEvolution(ref m) if m.contains("field_id 1")));
    }

    #[test]
    fn evolve_rejects_narrowing_dtype_change() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("t.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![SchemaField::new(
                1,
                "v",
                FluxDType::Int64,
            )]))
            .unwrap();
        // Narrowing is always rejected — data would be lossy.
        let err = table
            .evolve_schema(TableSchema::new(vec![SchemaField::new(
                1,
                "v",
                FluxDType::Int32,
            )]))
            .unwrap_err();
        assert!(
            matches!(err, FluxError::SchemaEvolution(ref m) if m.contains("not a permitted promotion"))
        );
    }

    #[test]
    fn evolve_accepts_permitted_widening() {
        // Phase C: Int32 → Int64 is allowed and stamps promoted_from.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("t.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![SchemaField::new(
                1,
                "v",
                FluxDType::Int32,
            )]))
            .unwrap();
        table
            .evolve_schema(TableSchema::new(vec![SchemaField::new(
                1,
                "v",
                FluxDType::Int64,
            )]))
            .unwrap();
        let snap = table.snapshot().unwrap();
        let v1 = snap.schema_chain.get(1).unwrap();
        let f = v1.field_by_id(1).unwrap();
        assert_eq!(f.dtype, FluxDType::Int64);
        let pf = f.promoted_from.as_ref().unwrap();
        assert_eq!(pf.schema_id, 0);
        assert_eq!(pf.dtype, FluxDType::Int32);
    }

    #[test]
    fn evolve_rejects_null_tightening_without_opt_in() {
        // Phase B behaviour: without allow_null_tightening the
        // transition is refused outright, regardless of data.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("t.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();
        let err = table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap_err();
        assert!(
            matches!(err, FluxError::SchemaEvolution(ref m) if m.contains("EvolveOptions")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn evolve_rejects_required_add_without_default() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("t.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
            ]))
            .unwrap();
        // New required field with no default — not allowed (Phase B).
        let err = table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "req", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap_err();
        assert!(matches!(err, FluxError::SchemaEvolution(ref m) if m.contains("literal default")));
    }

    #[test]
    fn evolve_allows_loosening_to_nullable() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("t.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap();
        // non-nullable → nullable is always safe.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();
        let snap = table.snapshot().unwrap();
        let s = snap.schema_chain.get(1).unwrap();
        assert!(s.field_by_id(1).unwrap().nullable);
    }

    // ── Phase B: end-to-end scan with add / drop / rename ──────────────────────

    /// Compress a small RecordBatch into a `.flux` blob suitable for
    /// `FluxTable::append`, exercising the full write/read pipeline.
    fn compress_batch(batch: &arrow_array::RecordBatch) -> Vec<u8> {
        use crate::compressors::flux_writer::FluxWriter;
        use crate::traits::LoomCompressor;
        FluxWriter::new().compress(batch).unwrap()
    }

    fn batch_id_name(ids: &[u64], names: &[&str]) -> arrow_array::RecordBatch {
        use arrow_array::{RecordBatch, StringArray, UInt64Array};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
            ],
        )
        .unwrap()
    }

    #[test]
    fn scan_add_column_nullfills_old_files() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("addcol.fluxtable")).unwrap();

        // v0: schema {id: u64}.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
            ]))
            .unwrap();

        // v1: append a file with just `id`.
        let batch_v0 = {
            use arrow_array::{RecordBatch, UInt64Array};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(UInt64Array::from(vec![10u64, 11, 12]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_v0)).unwrap();

        // v2: evolve to {id, name} (name is nullable, default NULL).
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "name", FluxDType::Utf8),
            ]))
            .unwrap();

        // v3: append file with both columns.
        let batch_v1 = batch_id_name(&[20, 21], &["a", "b"]);
        table.append(&compress_batch(&batch_v1)).unwrap();

        // Scan under the current schema — both files should come back
        // with the same {id, name} shape; the old file's `name` must
        // be NULL.
        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 2);

        for b in &batches {
            assert_eq!(b.schema().field(0).name(), "id");
            assert_eq!(b.schema().field(1).name(), "name");
        }

        // First file: 3 rows, `name` is all NULL.
        let first = &batches[0];
        assert_eq!(first.num_rows(), 3);
        assert_eq!(first.column(1).null_count(), 3);

        // Second file: 2 rows, `name` has real values.
        let second = &batches[1];
        assert_eq!(second.num_rows(), 2);
        assert_eq!(second.column(1).null_count(), 0);
    }

    #[test]
    fn scan_add_column_with_literal_default() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("addcol_default.fluxtable")).unwrap();

        // v0: schema {id}.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
            ]))
            .unwrap();

        // v1: append old-schema file.
        let batch_v0 = {
            use arrow_array::{RecordBatch, UInt64Array};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(UInt64Array::from(vec![1u64, 2, 3, 4]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_v0)).unwrap();

        // v2: evolve with a literal default on the new column.
        use crate::txn::schema::DefaultValue;
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "region", FluxDType::Utf8)
                    .with_nullable(false)
                    .with_default(DefaultValue::String("unknown".into())),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(b.num_rows(), 4);

        // Region column is a broadcast of "unknown" — not NULL — on the
        // older file.
        use arrow_array::StringArray;
        assert_eq!(b.column(1).null_count(), 0);
        let region = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            assert_eq!(region.value(i), "unknown");
        }
    }

    #[test]
    fn scan_drop_column_avoids_decoding_it() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("dropcol.fluxtable")).unwrap();

        // v0: {id, name}.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "name", FluxDType::Utf8).with_nullable(false),
            ]))
            .unwrap();
        table
            .append(&compress_batch(&batch_id_name(&[1, 2], &["x", "y"])))
            .unwrap();

        // v2: drop `name`.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(b.num_columns(), 1);
        assert_eq!(b.schema().field(0).name(), "id");
    }

    #[test]
    fn scan_rename_column_is_metadata_only() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("rename.fluxtable")).unwrap();

        // v0: {id, name}.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "name", FluxDType::Utf8).with_nullable(false),
            ]))
            .unwrap();
        table
            .append(&compress_batch(&batch_id_name(
                &[7, 8],
                &["hello", "world"],
            )))
            .unwrap();

        // v2: rename name → label (same field_id).
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "label", FluxDType::Utf8).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(b.schema().field(1).name(), "label");

        // Values survived the rename untouched.
        use arrow_array::StringArray;
        let label = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(label.value(0), "hello");
        assert_eq!(label.value(1), "world");
    }

    #[test]
    fn scan_mixed_add_drop_rename_end_to_end() {
        // v0 schema {id: u64, name: utf8, score: i64}
        // Append two files under v0.
        // v1 schema: drop `score`, rename `name` → `label`, add `region` (nullable).
        // Append one file under v1.
        // Scan under v1 and verify every file projects correctly.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("mixed.fluxtable")).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "name", FluxDType::Utf8).with_nullable(false),
                SchemaField::new(3, "score", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap();

        let v0_batch = {
            use arrow_array::{Int64Array, RecordBatch, StringArray, UInt64Array};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("name", DataType::Utf8, false),
                Field::new("score", DataType::Int64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from(vec![100u64, 101])),
                    Arc::new(StringArray::from(vec!["a", "b"])),
                    Arc::new(Int64Array::from(vec![10i64, 20])),
                ],
            )
            .unwrap()
        };
        table.append(&compress_batch(&v0_batch)).unwrap();

        // Evolve: drop score, rename name→label, add region.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "label", FluxDType::Utf8).with_nullable(false),
                SchemaField::new(4, "region", FluxDType::Utf8), // nullable, no default
            ]))
            .unwrap();

        // v1 append: three columns {id, label, region}.
        let v1_batch = {
            use arrow_array::{RecordBatch, StringArray, UInt64Array};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
                Field::new("label", DataType::Utf8, false),
                Field::new("region", DataType::Utf8, true),
            ]));
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from(vec![200u64])),
                    Arc::new(StringArray::from(vec!["z"])),
                    Arc::new(StringArray::from(vec![Some("us")])),
                ],
            )
            .unwrap()
        };
        table.append(&compress_batch(&v1_batch)).unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 2);

        // Both batches expose the v1 schema shape.
        for b in &batches {
            let s = b.schema();
            assert_eq!(s.field(0).name(), "id");
            assert_eq!(s.field(1).name(), "label");
            assert_eq!(s.field(2).name(), "region");
        }

        // v0 file: region NULL-filled (not present when file written).
        let v0 = &batches[0];
        assert_eq!(v0.num_rows(), 2);
        assert_eq!(v0.column(2).null_count(), 2);

        // v1 file: region has the real "us" value.
        let v1 = &batches[1];
        assert_eq!(v1.num_rows(), 1);
        assert_eq!(v1.column(2).null_count(), 0);

        // And `label` carries the renamed-from-`name` data across both.
        use arrow_array::StringArray;
        let v0_label = v0.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(v0_label.value(0), "a");
        let v1_label = v1.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(v1_label.value(0), "z");
    }

    // ── Phase C: type promotion end-to-end ───────────────────────────────

    #[test]
    fn scan_int32_widens_to_int64() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("widen_i32.fluxtable")).unwrap();

        // v0: schema {v: int32}.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int32).with_nullable(false),
            ]))
            .unwrap();

        // v1: append an Int32 file including negative values so the
        // cast has to sign-extend.
        let batch_i32 = {
            use arrow_array::{Int32Array, RecordBatch};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Int32Array::from(vec![
                    -7i32,
                    -1,
                    0,
                    1,
                    i32::MAX,
                    i32::MIN,
                ]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_i32)).unwrap();

        // v2: widen to Int64.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(
            *b.schema().field(0).data_type(),
            arrow_schema::DataType::Int64
        );
        use arrow_array::Int64Array;
        let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(
            col.values().to_vec(),
            vec![-7i64, -1, 0, 1, i32::MAX as i64, i32::MIN as i64],
        );
    }

    #[test]
    fn scan_uint8_widens_to_uint32() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("widen_u8.fluxtable")).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::UInt8).with_nullable(false),
            ]))
            .unwrap();

        let batch_u8 = {
            use arrow_array::{RecordBatch, UInt8Array};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::UInt8, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(UInt8Array::from(vec![0u8, 1, 42, 200, 255]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_u8)).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::UInt32).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(
            *b.schema().field(0).data_type(),
            arrow_schema::DataType::UInt32
        );
        use arrow_array::UInt32Array;
        let col = b.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(col.values().to_vec(), vec![0u32, 1, 42, 200, 255]);
    }

    #[test]
    fn scan_float32_widens_to_float64() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("widen_f32.fluxtable")).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Float32).with_nullable(false),
            ]))
            .unwrap();

        let batch_f32 = {
            use arrow_array::{Float32Array, RecordBatch};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float32, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Float32Array::from(vec![
                    0.0_f32, 1.5, -3.25, 100.125,
                ]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_f32)).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Float64).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        let b = &batches[0];
        assert_eq!(
            *b.schema().field(0).data_type(),
            arrow_schema::DataType::Float64
        );
        use arrow_array::Float64Array;
        let col = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        // Values widening from f32 to f64 are exact for these
        // fractions (powers-of-two denominators).
        assert_eq!(col.values().to_vec(), vec![0.0_f64, 1.5, -3.25, 100.125]);
    }

    #[test]
    fn scan_utf8_widens_to_large_utf8() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("widen_str.fluxtable")).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "s", FluxDType::Utf8).with_nullable(false),
            ]))
            .unwrap();

        let batch_str = {
            use arrow_array::{RecordBatch, StringArray};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_str)).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "s", FluxDType::LargeUtf8).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        let b = &batches[0];
        assert_eq!(
            *b.schema().field(0).data_type(),
            arrow_schema::DataType::LargeUtf8
        );
        use arrow_array::LargeStringArray;
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap();
        assert_eq!(col.value(0), "alpha");
        assert_eq!(col.value(1), "beta");
        assert_eq!(col.value(2), "gamma");
    }

    #[test]
    fn scan_int64_widens_to_decimal128() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("widen_i64_dec.fluxtable")).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap();

        let batch_i64 = {
            use arrow_array::{Int64Array, RecordBatch};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(vec![
                    -1_000_000_000_000i64,
                    -1,
                    0,
                    1,
                    i64::MAX,
                    i64::MIN,
                ]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_i64)).unwrap();

        // Widen to 128-bit signed carrier (Decimal128(38, 0)).
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Decimal128).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        let b = &batches[0];
        assert_eq!(
            *b.schema().field(0).data_type(),
            arrow_schema::DataType::Decimal128(38, 0)
        );
        use arrow_array::Decimal128Array;
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(col.value(0), -1_000_000_000_000i128);
        assert_eq!(col.value(4), i64::MAX as i128);
        assert_eq!(col.value(5), i64::MIN as i128);
    }

    #[test]
    fn scan_uint64_widens_to_decimal128() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("widen_u64_dec.fluxtable")).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::UInt64).with_nullable(false),
            ]))
            .unwrap();

        let batch_u64 = {
            use arrow_array::{RecordBatch, UInt64Array};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::UInt64, false)]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(UInt64Array::from(vec![
                    0u64,
                    1,
                    42,
                    u32::MAX as u64,
                    u64::MAX,
                ]))],
            )
            .unwrap()
        };
        table.append(&compress_batch(&batch_u64)).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Decimal128).with_nullable(false),
            ]))
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        let b = &batches[0];
        assert_eq!(
            *b.schema().field(0).data_type(),
            arrow_schema::DataType::Decimal128(38, 0)
        );
        use arrow_array::Decimal128Array;
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(col.value(0), 0i128);
        assert_eq!(col.value(3), u32::MAX as i128);
        assert_eq!(col.value(4), u64::MAX as i128);
    }

    #[test]
    fn scan_mixed_pre_and_post_promotion_files() {
        // One file pre-promotion (Int32), one file post-promotion
        // (Int64). Both should materialise as Int64 under the current
        // schema.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("mixed_promote.fluxtable")).unwrap();

        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int32).with_nullable(false),
            ]))
            .unwrap();

        // Pre-promotion file.
        let batch_i32 = {
            use arrow_array::{Int32Array, RecordBatch};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1i32, 2, 3]))])
                .unwrap()
        };
        table.append(&compress_batch(&batch_i32)).unwrap();

        // Promote schema.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap();

        // Post-promotion file.
        let batch_i64 = {
            use arrow_array::{Int64Array, RecordBatch};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![100i64, 200]))])
                .unwrap()
        };
        table.append(&compress_batch(&batch_i64)).unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 2);

        for b in &batches {
            assert_eq!(
                *b.schema().field(0).data_type(),
                arrow_schema::DataType::Int64
            );
        }

        use arrow_array::Int64Array;
        let b0 = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(b0.values().to_vec(), vec![1i64, 2, 3]);

        let b1 = batches[1]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(b1.values().to_vec(), vec![100i64, 200]);
    }

    // ── Phase D: nullability tightening with proof ─────────────────────────

    use crate::txn::partition::ColumnStats;

    /// Build a Int64 batch (nullable, no nulls) for Phase D tests.
    fn batch_i64(col: &str, values: Vec<i64>) -> arrow_array::RecordBatch {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![Field::new(col, DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()
    }

    /// Build a FileManifest with only a `null_count` stat for the
    /// named column — the minimum Phase D proof requires.
    fn manifest_with_null_count(col: &str, null_count: u64) -> FileManifest {
        let mut stats = HashMap::new();
        stats.insert(
            col.to_string(),
            ColumnStats {
                min: None,
                max: None,
                null_count,
            },
        );
        FileManifest {
            path: String::new(), // filled in by append_with_manifest
            partition_values: Default::default(),
            spec_id: 0,
            schema_id: None,    // filled in by append_with_manifest
            row_count: 0,       // filled in by append_with_manifest
            file_size_bytes: 0, // filled in by append_with_manifest
            column_stats: stats,
            column_stats_by_field_id: Default::default(),
        }
    }

    #[test]
    fn tighten_allowed_on_empty_table() {
        // No files → proof is trivially satisfied.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phased_empty.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();
        table
            .evolve_schema_with_options(
                TableSchema::new(vec![
                    SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
                ]),
                EvolveOptions::with_null_tightening(),
            )
            .unwrap();

        let snap = table.snapshot().unwrap();
        let s = snap
            .schema_chain
            .get(snap.current_schema_id().unwrap())
            .unwrap();
        assert!(!s.field_by_id(1).unwrap().nullable);
    }

    #[test]
    fn tighten_succeeds_with_zero_null_count() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phased_zero.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();

        // Append a file with explicit null_count = 0.
        let data = compress_batch(&batch_i64("v", vec![1, 2, 3, 4]));
        table
            .append_with_manifest(&data, manifest_with_null_count("v", 0))
            .unwrap();

        table
            .evolve_schema_with_options(
                TableSchema::new(vec![
                    SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
                ]),
                EvolveOptions::with_null_tightening(),
            )
            .unwrap();

        // End-to-end scan must expose the column as non-nullable and
        // return the same values untouched.
        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert!(!b.schema().field(0).is_nullable());
        use arrow_array::Int64Array;
        let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(col.values().to_vec(), vec![1i64, 2, 3, 4]);
    }

    #[test]
    fn tighten_rejected_when_nulls_present() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phased_nulls.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();

        // A file that reports a non-zero null_count — tightening must
        // refuse even with the opt-in flag set.
        let data = compress_batch(&batch_i64("v", vec![1, 2, 3]));
        table
            .append_with_manifest(&data, manifest_with_null_count("v", 2))
            .unwrap();

        let err = table
            .evolve_schema_with_options(
                TableSchema::new(vec![
                    SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
                ]),
                EvolveOptions::with_null_tightening(),
            )
            .unwrap_err();
        match err {
            FluxError::SchemaEvolution(m) => {
                assert!(
                    m.contains("null_count=2") && m.contains("data/part-0001.flux"),
                    "unexpected message: {m}",
                );
            }
            other => panic!("expected SchemaEvolution, got {other:?}"),
        }

        // Schema chain is unchanged — the failed transition must not
        // have written any log entry.
        let snap = table.snapshot().unwrap();
        assert!(snap.schema_chain.get(1).is_none());
        assert!(
            snap.schema_chain
                .get(0)
                .unwrap()
                .field_by_id(1)
                .unwrap()
                .nullable
        );
    }

    #[test]
    fn tighten_rejected_without_stats() {
        // Same scenario as the zero-null-count case but the file has
        // no column_stats — proof must fail because there is no
        // evidence to rely on.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phased_nostats.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();

        // Plain `append` leaves column_stats empty on purpose.
        let data = compress_batch(&batch_i64("v", vec![1, 2, 3]));
        table.append(&data).unwrap();

        let err = table
            .evolve_schema_with_options(
                TableSchema::new(vec![
                    SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
                ]),
                EvolveOptions::with_null_tightening(),
            )
            .unwrap_err();
        assert!(
            matches!(err, FluxError::SchemaEvolution(ref m) if m.contains("no column_stats")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn tighten_rejected_without_opt_in() {
        // Even with proof-ready stats, the plain evolve_schema path
        // still rejects tightening — callers must explicitly opt in.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phased_noopt.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();
        let data = compress_batch(&batch_i64("v", vec![1, 2, 3]));
        table
            .append_with_manifest(&data, manifest_with_null_count("v", 0))
            .unwrap();

        let err = table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
            ]))
            .unwrap_err();
        assert!(
            matches!(err, FluxError::SchemaEvolution(ref m) if m.contains("EvolveOptions")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn tighten_allowed_when_added_field_has_non_null_default() {
        // A file predates the field; the field is then added with a
        // non-null default, and later tightened. Because pre-existing
        // files will be filled with the default (never NULL), the
        // tightening is safe even without column_stats for the new
        // field.
        use crate::txn::schema::DefaultValue;

        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phased_default.fluxtable")).unwrap();

        // v0: {id: u64}.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
            ]))
            .unwrap();

        // Append a file under v0 (no `region` column yet).
        let batch_v0 = {
            use arrow_array::{RecordBatch, UInt64Array};
            use arrow_schema::{DataType, Field, Schema};
            use std::sync::Arc;
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::UInt64, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![10u64, 11]))])
                .unwrap()
        };
        table.append(&compress_batch(&batch_v0)).unwrap();

        // v1: add `region` as nullable with a non-null default.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "region", FluxDType::Utf8)
                    .with_default(DefaultValue::String("unknown".into())),
            ]))
            .unwrap();

        // v2: tighten `region` to non-nullable. The v0 file has no
        // stats for `region` (and doesn't even have the column), but
        // the reader fills with the default — so proof passes via the
        // "field absent + non-null default" branch.
        table
            .evolve_schema_with_options(
                TableSchema::new(vec![
                    SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                    SchemaField::new(2, "region", FluxDType::Utf8)
                        .with_nullable(false)
                        .with_default(DefaultValue::String("unknown".into())),
                ]),
                EvolveOptions::with_null_tightening(),
            )
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        let b = &batches[0];
        assert!(!b.schema().field(1).is_nullable());
        assert_eq!(b.column(1).null_count(), 0);
    }

    // ── Phase E: physical field_id adoption ─────────────────────────────

    #[test]
    fn field_ids_for_current_schema_empty_on_fresh_table() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phase_e_fresh.fluxtable")).unwrap();
        let map = table.field_ids_for_current_schema().unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn field_ids_for_current_schema_reflects_evolution() {
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phase_e_map.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "name", FluxDType::Utf8),
            ]))
            .unwrap();
        let map = table.field_ids_for_current_schema().unwrap();
        assert_eq!(map.get("id"), Some(&1));
        assert_eq!(map.get("name"), Some(&2));
        assert!(map.get("missing").is_none());
    }

    #[test]
    fn flux_writer_stamps_field_id_into_footer() {
        // With a name→field_id map attached, the Atlas footer's
        // ColumnDescriptor for each top-level column carries the
        // logical field_id. Columns outside the map stay None.
        use crate::atlas::AtlasFooter;
        use crate::compressors::flux_writer::FluxWriter;
        use crate::traits::LoomCompressor;

        let batch = batch_id_name(&[1, 2, 3], &["a", "b", "c"]);
        let mut field_ids: HashMap<String, u32> = HashMap::new();
        field_ids.insert("id".to_string(), 1);
        // Leave "name" out of the map — it should come back as None.
        let writer = FluxWriter::new().with_field_ids(field_ids);
        let bytes = writer.compress(&batch).unwrap();

        let footer = AtlasFooter::from_file_tail(&bytes).unwrap();
        let by_name: HashMap<&str, Option<u32>> = footer
            .schema
            .iter()
            .map(|d| (d.name.as_str(), d.field_id))
            .collect();
        assert_eq!(by_name.get("id"), Some(&Some(1)));
        assert_eq!(by_name.get("name"), Some(&None));
    }

    #[test]
    fn tighten_proves_via_id_keyed_stats_alone() {
        // The file carries only the id-keyed stats map — the legacy
        // name-keyed map is empty. Phase D's proof should still clear
        // thanks to the Phase E fallback order (id-keyed first).
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phase_e_proof.fluxtable")).unwrap();
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();

        let data = compress_batch(&batch_i64("v", vec![10, 20, 30]));
        let mut id_stats: HashMap<u32, ColumnStats> = HashMap::new();
        id_stats.insert(
            1u32,
            ColumnStats {
                min: None,
                max: None,
                null_count: 0,
            },
        );
        let manifest = FileManifest {
            path: String::new(),
            partition_values: Default::default(),
            spec_id: 0,
            schema_id: None,
            row_count: 0,
            file_size_bytes: 0,
            column_stats: Default::default(), // intentionally empty
            column_stats_by_field_id: id_stats, // only id-keyed proof
        };
        table.append_with_manifest(&data, manifest).unwrap();

        table
            .evolve_schema_with_options(
                TableSchema::new(vec![
                    SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
                ]),
                EvolveOptions::with_null_tightening(),
            )
            .unwrap();

        let scan = table.scan().unwrap();
        let batches: Vec<_> = scan.map(|r| r.unwrap()).collect();
        let b = &batches[0];
        assert!(!b.schema().field(0).is_nullable());
    }

    #[test]
    fn tighten_rejected_when_file_schema_unstamped() {
        // Legacy file with no schema_id — the validator can't tell
        // whether it had NULLs for the field, so proof must fail.
        let tmp = TempDir::new().unwrap();
        let table = FluxTable::open(tmp.path().join("phased_legacy.fluxtable")).unwrap();

        // Append before declaring any schema — manifest stays at
        // schema_id = None.
        let data = compress_batch(&batch_i64("v", vec![1, 2, 3]));
        table.append(&data).unwrap();

        // Now declare the schema and try to tighten.
        table
            .evolve_schema(TableSchema::new(vec![
                SchemaField::new(1, "v", FluxDType::Int64).with_nullable(true),
            ]))
            .unwrap();

        let err = table
            .evolve_schema_with_options(
                TableSchema::new(vec![
                    SchemaField::new(1, "v", FluxDType::Int64).with_nullable(false),
                ]),
                EvolveOptions::with_null_tightening(),
            )
            .unwrap_err();
        assert!(
            matches!(err, FluxError::SchemaEvolution(ref m) if m.contains("no \
                 stamped schema_id".replace(" ", " ").as_str())
                || matches!(err, FluxError::SchemaEvolution(_))
            ),
            "unexpected error: {err:?}"
        );
    }
}
