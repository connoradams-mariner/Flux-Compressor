// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Schema-evolution projection planning.
//!
//! Given a caller-requested *target* [`TableSchema`] and the schema a
//! particular data file was written under, this module produces a
//! [`FilePlan`] describing how to materialise the target schema from
//! that file — which physical columns to decode, which to fill with
//! NULL or a literal default, and which output column each decoded
//! array corresponds to.
//!
//! ## Why it matters
//! The scan path resolves files against a logical schema by
//! `field_id`, not by name. That's what makes renames free, drops I/O-
//! free, and adds NULL-allocatable-only. Plan construction is pure and
//! deterministic, so a scan over N files that share the same
//! `schema_id` pays the plan cost exactly once and then dispatches by
//! `schema_id` in O(1) per file.
//!
//! ## Backward compatibility
//! When a file has no schema stamp (pre-Phase-A), the plan falls back
//! to a name match against the file's physical `ColumnDescriptor`
//! list. This preserves the legacy behaviour where column identity is
//! carried by name alone.

use crate::dtype::FluxDType;
use crate::error::{FluxError, FluxResult};

use super::schema::{DefaultValue, SchemaField, TableSchema};

// ─────────────────────────────────────────────────────────────────────────────
// ColumnPlan
// ─────────────────────────────────────────────────────────────────────────────

/// One column's projection plan.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnPlan {
    /// Decode this column from the file, optionally renaming the
    /// output and casting to a wider `target_dtype`.
    ///
    /// The `physical_name` is the name the column has in the file's
    /// physical footer (resolved via the file's [`TableSchema`] or via
    /// a name match on legacy files). `target_name` is the name the
    /// scan should expose.
    ///
    /// Phase C: when `source_dtype != target_dtype`, the reader
    /// applies an `arrow::compute::cast` after decoding to promote
    /// the column to the caller-requested dtype. The pair is
    /// guaranteed to be in [`FluxDType::can_promote_to`] at plan-build
    /// time, so the cast cannot fail for value-range reasons the
    /// promotion rules permit.
    Decode {
        /// Column name as it appears in the file's Atlas footer.
        physical_name: String,
        /// Column name to expose in the output [`RecordBatch`].
        target_name: String,
        /// Logical dtype the column has on disk in *this* file.
        source_dtype: FluxDType,
        /// Logical dtype the output should carry.
        target_dtype: FluxDType,
        /// Output nullability flag.
        target_nullable: bool,
    },
    /// Synthesise a constant column of `row_count` rows with the given
    /// literal default, or `NULL` when `default` is `None`.
    Fill {
        /// Column name to expose in the output [`RecordBatch`].
        target_name: String,
        /// Logical dtype of the synthesised column.
        target_dtype: FluxDType,
        /// Output nullability flag.
        target_nullable: bool,
        /// Literal default, or `None` for NULL-fill.
        default: Option<DefaultValue>,
        /// Number of rows in the file (and therefore in the filled
        /// column).
        row_count: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// FilePlan
// ─────────────────────────────────────────────────────────────────────────────

/// The complete per-file projection plan.
#[derive(Debug, Clone, PartialEq)]
pub struct FilePlan {
    /// Output columns in target-schema order.
    pub columns: Vec<ColumnPlan>,
    /// Physical column names the reader must decode from the file.
    /// Deduplicated and stable-ordered; scoped to the narrowest set
    /// the plan needs so dropped columns are never paid for.
    pub file_physical_columns: Vec<String>,
}

impl FilePlan {
    /// True when no physical columns need to be touched — a pure
    /// NULL/default synthesis pass. The reader uses this to skip the
    /// file open entirely, which matters for scans where a caller
    /// requests only newly-added columns.
    pub fn is_pure_fill(&self) -> bool {
        self.file_physical_columns.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`FilePlan`] that resolves `target` against a single file.
///
/// * `target` — the logical schema the caller wants back.
/// * `file_schema` — the schema the file was written under. `None`
///   indicates a pre-evolution file; the plan falls back to matching
///   `target_field.name` against `file_physical_columns`.
/// * `file_physical_columns` — column names present in the file's
///   Atlas footer. Used for the legacy path and for sanity-checking
///   the physical name we resolved via `file_schema`.
/// * `file_row_count` — from the file's manifest; used to size
///   `Fill` columns on added fields.
///
/// Returns [`FluxError::FieldMissing`] if the target schema asks for a
/// non-nullable field that the file does not have and there is no
/// literal default to fill from.
pub fn build_file_plan(
    target: &TableSchema,
    file_schema: Option<&TableSchema>,
    file_physical_columns: &[String],
    file_row_count: u64,
) -> FluxResult<FilePlan> {
    let mut columns: Vec<ColumnPlan> = Vec::with_capacity(target.fields.len());
    let mut physical: Vec<String> = Vec::with_capacity(target.fields.len());

    for target_field in &target.fields {
        let plan = plan_for_field(
            target_field,
            file_schema,
            file_physical_columns,
            file_row_count,
        )?;
        if let ColumnPlan::Decode { ref physical_name, .. } = plan {
            if !physical.iter().any(|c| c == physical_name) {
                physical.push(physical_name.clone());
            }
        }
        columns.push(plan);
    }

    Ok(FilePlan {
        columns,
        file_physical_columns: physical,
    })
}

fn plan_for_field(
    target_field: &SchemaField,
    file_schema: Option<&TableSchema>,
    file_physical_columns: &[String],
    file_row_count: u64,
) -> FluxResult<ColumnPlan> {
    // Path 1: the file's schema is known. Resolve by `field_id`, which
    // is rename-safe.
    if let Some(fs) = file_schema {
        if let Some(fs_field) = fs.field_by_id(target_field.field_id) {
            // Phase C: dtype differences are legal iff they land on
            // the permitted promotion matrix.
            if fs_field.dtype != target_field.dtype
                && !fs_field.dtype.can_promote_to(target_field.dtype)
            {
                return Err(FluxError::SchemaEvolution(format!(
                    "field_id {} ('{}'): dtype change {:?} → {:?} is not a \
                     permitted promotion",
                    target_field.field_id, target_field.name,
                    fs_field.dtype, target_field.dtype,
                )));
            }
            // Sanity-check the physical name is actually in the footer
            // (guards against stale schema chain entries).
            if !file_physical_columns.is_empty()
                && !file_physical_columns.iter().any(|c| c == &fs_field.name)
            {
                return Err(FluxError::SchemaEvolution(format!(
                    "field_id {}: schema chain names physical column '{}' but file \
                     footer has {:?}",
                    target_field.field_id, fs_field.name, file_physical_columns,
                )));
            }
            return Ok(ColumnPlan::Decode {
                physical_name: fs_field.name.clone(),
                target_name: target_field.name.clone(),
                source_dtype: fs_field.dtype,
                target_dtype: target_field.dtype,
                target_nullable: target_field.nullable,
            });
        }
        // Target field not in the file's schema ⇒ it was added after
        // this file was written. Fill from the target's default (NULL
        // for nullable, literal default otherwise).
        return fill_plan(target_field, file_row_count);
    }

    // Path 2: legacy file (no schema stamp). Match by name against the
    // physical footer. We have no way to know the file's dtype here,
    // so we assume identity; type promotion requires a stamped file
    // schema to be well-defined.
    if file_physical_columns.iter().any(|c| c == &target_field.name) {
        return Ok(ColumnPlan::Decode {
            physical_name: target_field.name.clone(),
            target_name: target_field.name.clone(),
            source_dtype: target_field.dtype,
            target_dtype: target_field.dtype,
            target_nullable: target_field.nullable,
        });
    }

    fill_plan(target_field, file_row_count)
}

fn fill_plan(target_field: &SchemaField, file_row_count: u64) -> FluxResult<ColumnPlan> {
    // Required field with no default and no matching physical column
    // → hard error, matching the roadmap's `FieldMissingError`.
    if !target_field.nullable && target_field.default.is_none() {
        return Err(FluxError::FieldMissing(format!(
            "field_id {} ('{}') is non-nullable and has no default, \
             but is absent from the file",
            target_field.field_id, target_field.name,
        )));
    }
    Ok(ColumnPlan::Fill {
        target_name: target_field.name.clone(),
        target_dtype: target_field.dtype,
        target_nullable: target_field.nullable,
        default: target_field.default.clone(),
        row_count: file_row_count,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::txn::schema::{DefaultValue, SchemaField, TableSchema};

    fn schema(id: u32, fields: Vec<SchemaField>) -> TableSchema {
        let mut s = TableSchema::new(fields);
        s.schema_id = id;
        s
    }

    #[test]
    fn decode_when_field_matches_by_id() {
        let file_schema = schema(
            0,
            vec![SchemaField::new(1, "id", FluxDType::UInt64)],
        );
        let target = schema(
            1,
            vec![SchemaField::new(1, "id", FluxDType::UInt64)],
        );
        let plan = build_file_plan(&target, Some(&file_schema), &["id".into()], 100).unwrap();
        assert_eq!(plan.columns.len(), 1);
        match &plan.columns[0] {
            ColumnPlan::Decode {
                physical_name,
                target_name,
                source_dtype,
                target_dtype,
                ..
            } => {
                assert_eq!(physical_name, "id");
                assert_eq!(target_name, "id");
                assert_eq!(*source_dtype, FluxDType::UInt64);
                assert_eq!(*target_dtype, FluxDType::UInt64);
            }
            _ => panic!("expected Decode"),
        }
        assert_eq!(plan.file_physical_columns, vec!["id".to_string()]);
    }

    #[test]
    fn rename_uses_file_physical_name_and_target_display_name() {
        let file_schema = schema(
            0,
            vec![SchemaField::new(2, "region", FluxDType::Utf8)],
        );
        let target = schema(
            1,
            vec![SchemaField::new(2, "region_code", FluxDType::Utf8)],
        );
        let plan = build_file_plan(&target, Some(&file_schema), &["region".into()], 10).unwrap();
        match &plan.columns[0] {
            ColumnPlan::Decode { physical_name, target_name, .. } => {
                assert_eq!(physical_name, "region");
                assert_eq!(target_name, "region_code");
            }
            _ => panic!("expected Decode"),
        }
    }

    #[test]
    fn added_nullable_field_fills_null() {
        let file_schema = schema(
            0,
            vec![SchemaField::new(1, "id", FluxDType::UInt64)],
        );
        let target = schema(
            1,
            vec![
                SchemaField::new(1, "id", FluxDType::UInt64),
                SchemaField::new(2, "name", FluxDType::Utf8), // nullable by default
            ],
        );
        let plan = build_file_plan(&target, Some(&file_schema), &["id".into()], 42).unwrap();
        assert_eq!(plan.columns.len(), 2);
        match &plan.columns[1] {
            ColumnPlan::Fill { target_name, default, row_count, .. } => {
                assert_eq!(target_name, "name");
                assert!(default.is_none());
                assert_eq!(*row_count, 42);
            }
            _ => panic!("expected Fill"),
        }
    }

    #[test]
    fn added_field_with_default_fills_literal() {
        let file_schema = schema(
            0,
            vec![SchemaField::new(1, "id", FluxDType::UInt64)],
        );
        let target = schema(
            1,
            vec![
                SchemaField::new(1, "id", FluxDType::UInt64),
                SchemaField::new(2, "region", FluxDType::Utf8)
                    .with_default(DefaultValue::String("unknown".into())),
            ],
        );
        let plan = build_file_plan(&target, Some(&file_schema), &["id".into()], 7).unwrap();
        match &plan.columns[1] {
            ColumnPlan::Fill { default: Some(DefaultValue::String(s)), .. } => {
                assert_eq!(s, "unknown");
            }
            other => panic!("expected Fill with String default, got {other:?}"),
        }
    }

    #[test]
    fn dropped_column_is_absent_from_physical_list() {
        let file_schema = schema(
            0,
            vec![
                SchemaField::new(1, "id", FluxDType::UInt64),
                SchemaField::new(2, "dropped", FluxDType::Utf8),
            ],
        );
        let target = schema(
            1,
            vec![SchemaField::new(1, "id", FluxDType::UInt64)],
        );
        let plan = build_file_plan(
            &target,
            Some(&file_schema),
            &["id".into(), "dropped".into()],
            5,
        )
        .unwrap();
        assert_eq!(plan.columns.len(), 1);
        assert_eq!(plan.file_physical_columns, vec!["id".to_string()]);
        assert!(!plan.file_physical_columns.contains(&"dropped".to_string()));
    }

    #[test]
    fn legacy_file_matches_by_name() {
        let target = schema(
            0,
            vec![
                SchemaField::new(1, "id", FluxDType::UInt64),
                SchemaField::new(2, "missing", FluxDType::Utf8), // absent in file
            ],
        );
        let plan = build_file_plan(&target, None, &["id".into()], 3).unwrap();
        assert!(matches!(plan.columns[0], ColumnPlan::Decode { .. }));
        assert!(matches!(plan.columns[1], ColumnPlan::Fill { .. }));
    }

    #[test]
    fn field_missing_error_on_required_no_default() {
        let file_schema = schema(
            0,
            vec![SchemaField::new(1, "id", FluxDType::UInt64)],
        );
        let target = schema(
            1,
            vec![
                SchemaField::new(1, "id", FluxDType::UInt64),
                // Required, no default, not in file schema → error.
                SchemaField::new(2, "must_have", FluxDType::Int64).with_nullable(false),
            ],
        );
        let err = build_file_plan(&target, Some(&file_schema), &["id".into()], 1).unwrap_err();
        assert!(matches!(err, FluxError::FieldMissing(_)));
    }

    #[test]
    fn permitted_promotion_carries_source_dtype() {
        // Phase C: widening Int32 → Int64 is allowed; the plan must
        // expose the source dtype so the reader knows to cast.
        let file_schema = schema(
            0,
            vec![SchemaField::new(1, "v", FluxDType::Int32)],
        );
        let target = schema(
            1,
            vec![SchemaField::new(1, "v", FluxDType::Int64)],
        );
        let plan = build_file_plan(&target, Some(&file_schema), &["v".into()], 4).unwrap();
        match &plan.columns[0] {
            ColumnPlan::Decode {
                source_dtype,
                target_dtype,
                ..
            } => {
                assert_eq!(*source_dtype, FluxDType::Int32);
                assert_eq!(*target_dtype, FluxDType::Int64);
            }
            _ => panic!("expected Decode"),
        }
    }

    #[test]
    fn rejected_dtype_change_errors() {
        // Narrowing and cross-family changes stay rejected in Phase C.
        let file_schema = schema(
            0,
            vec![SchemaField::new(1, "v", FluxDType::Int64)],
        );
        let target = schema(
            1,
            vec![SchemaField::new(1, "v", FluxDType::Int32)], // narrowing
        );
        let err = build_file_plan(&target, Some(&file_schema), &["v".into()], 1).unwrap_err();
        assert!(matches!(err, FluxError::SchemaEvolution(_)));
    }

    #[test]
    fn pure_fill_plan_is_detected() {
        let target = schema(
            0,
            vec![SchemaField::new(1, "new_col", FluxDType::Int64)],
        );
        // File schema has nothing in common with target.
        let file_schema = schema(
            0,
            vec![SchemaField::new(99, "other", FluxDType::Utf8)],
        );
        let plan = build_file_plan(&target, Some(&file_schema), &["other".into()], 12).unwrap();
        assert!(plan.is_pure_fill());
    }
}
