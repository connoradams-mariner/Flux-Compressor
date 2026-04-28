// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Schema evolution: logical schemas and the replayed evolution chain.
//!
//! This is the Phase A plumbing described in
//! `docs/roadmap-schema-evolution.md` — the types that let a FluxTable
//! carry a logical schema record in its transaction log and stamp each
//! data file with a compact `schema_id`.
//!
//! ## Design priorities
//! The on-disk representation is intentionally minimal so schema
//! evolution does not compromise the table's core performance promises:
//!
//! * **Compact**: only a single integer `schema_id` rides in each
//!   [`FileManifest`]; the richer [`TableSchema`] payload lives once per
//!   evolution inside a `schema` action in the log. This keeps per-file
//!   metadata tiny compared to formats that re-embed the full schema on
//!   every commit.
//! * **O(1) lookup**: [`SchemaChain`] stores schemas in a dense vector
//!   indexed by `schema_id`, so scan pipelines resolve a file's schema
//!   with an indexed load — suitable for rayon-parallel scans across
//!   many files.
//! * **Streaming replay**: a single forward pass over the log fills the
//!   chain without allocating wide intermediate state, and readers can
//!   start decoding files as soon as their `schema_id` is known.

use serde::{Deserialize, Serialize};

use crate::dtype::FluxDType;

// ─────────────────────────────────────────────────────────────────────────────
// Default values
// ─────────────────────────────────────────────────────────────────────────────

/// Literal default value for a [`SchemaField`].
///
/// Phase A intentionally restricts defaults to primitive literals;
/// expression defaults (e.g. `now()`, `uuid()`) are deferred to a later
/// roadmap phase. `DefaultValue::Null` is the implicit default for every
/// nullable column that does not specify one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DefaultValue {
    /// Boolean literal.
    Bool(bool),
    /// Signed integer literal (fits in `i64`).
    Int(i64),
    /// Unsigned integer literal (fits in `u64`).
    UInt(u64),
    /// Floating-point literal.
    Float(f64),
    /// String literal.
    String(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema field
// ─────────────────────────────────────────────────────────────────────────────

/// Phase C lineage marker recording that a field's dtype was
/// promoted from an earlier schema.
///
/// Purely informational — the reader resolves promotions on its own
/// via the per-file `schema_id` ↔ chain lookup. Preserving this
/// alongside the field keeps the evolution history self-describing
/// without requiring callers to diff two `TableSchema`s manually.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotedFrom {
    /// The schema id the prior dtype lived at.
    pub schema_id: u32,
    /// The dtype the field had before this promotion.
    pub dtype: FluxDType,
}

/// A single field in a [`TableSchema`].
///
/// The `field_id` is the *immutable logical identifier* for the column
/// — names can change and numeric types can be promoted under the
/// compatibility rules spelled out in the roadmap, but `field_id`
/// is stable across the table's lifetime. This is what lets rename
/// operations work without rewriting data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaField {
    /// Stable logical identifier for this column, unique within the
    /// schema.
    pub field_id: u32,
    /// Current user-facing column name.
    pub name: String,
    /// Logical dtype at the current schema version.
    pub dtype: FluxDType,
    /// Whether NULLs are permitted in this column.
    pub nullable: bool,
    /// Optional literal default applied to files that pre-date this
    /// field. `None` means `NULL` for nullable fields; required fields
    /// with no default are rejected at evolve time in later phases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<DefaultValue>,
    /// Optional human-readable documentation string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    /// Phase C: if this field's dtype was widened from an earlier
    /// schema, records the prior `(schema_id, dtype)` pair. `None`
    /// on fresh fields and on unchanged preserved fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promoted_from: Option<PromotedFrom>,
}

impl SchemaField {
    /// Construct a minimal field with the given id / name / dtype and
    /// `nullable = true`.
    pub fn new(field_id: u32, name: impl Into<String>, dtype: FluxDType) -> Self {
        Self {
            field_id,
            name: name.into(),
            dtype,
            nullable: true,
            default: None,
            doc: None,
            promoted_from: None,
        }
    }

    /// Builder: set nullability.
    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Builder: set default value.
    pub fn with_default(mut self, default: DefaultValue) -> Self {
        self.default = Some(default);
        self
    }

    /// Builder: set documentation.
    pub fn with_doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = Some(doc.into());
        self
    }

    /// Builder: stamp a Phase C `promoted_from` lineage marker.
    pub fn with_promoted_from(mut self, pf: PromotedFrom) -> Self {
        self.promoted_from = Some(pf);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Table schema
// ─────────────────────────────────────────────────────────────────────────────

/// The authoritative logical schema of a FluxTable at a given
/// `schema_id`.
///
/// Emitted as a `schema` action in the log and replayed into a
/// [`SchemaChain`] by readers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableSchema {
    /// Monotonically increasing identifier. Fresh tables start at 0.
    pub schema_id: u32,
    /// Parent schema id in the evolution chain. `None` for the first
    /// schema only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_schema_id: Option<u32>,
    /// Fields in user-visible column order. Logical identity is carried
    /// by [`SchemaField::field_id`] rather than array index.
    pub fields: Vec<SchemaField>,
    /// Optional short summary of what changed vs the parent schema
    /// (human-readable, advisory only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change_summary: Option<String>,
}

impl TableSchema {
    /// Build an initial schema (`schema_id = 0`, no parent).
    pub fn new(fields: Vec<SchemaField>) -> Self {
        Self {
            schema_id: 0,
            parent_schema_id: None,
            fields,
            change_summary: None,
        }
    }

    /// Look up a field by its stable [`field_id`]. O(n) in the number
    /// of fields, which is small in practice.
    pub fn field_by_id(&self, field_id: u32) -> Option<&SchemaField> {
        self.fields.iter().find(|f| f.field_id == field_id)
    }

    /// Look up a field by its current display name.
    pub fn field_by_name(&self, name: &str) -> Option<&SchemaField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Returns the highest `field_id` used, or `None` if the schema
    /// has no fields.
    pub fn max_field_id(&self) -> Option<u32> {
        self.fields.iter().map(|f| f.field_id).max()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema chain (replayed evolution history)
// ─────────────────────────────────────────────────────────────────────────────

/// The replayed schema history, indexed for O(1) lookup by `schema_id`.
///
/// Internally stored as a dense `Vec<Option<TableSchema>>`. Readers
/// resolve a file's schema with a single indexed load, which keeps the
/// per-block hot path branch-light. The sparse `None` slots are cheap —
/// they cost one pointer each and Phase A does not create gappy
/// histories in practice.
#[derive(Debug, Default, Clone)]
pub struct SchemaChain {
    schemas: Vec<Option<TableSchema>>,
}

impl SchemaChain {
    /// Build an empty chain.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a schema into the chain. The chain is last-wins for a
    /// given `schema_id`, which matches the "metadata actions fully
    /// replace prior ones" rule in the transaction-log format.
    pub fn insert(&mut self, schema: TableSchema) {
        let id = schema.schema_id as usize;
        if self.schemas.len() <= id {
            self.schemas.resize(id + 1, None);
        }
        self.schemas[id] = Some(schema);
    }

    /// O(1) lookup by `schema_id`.
    pub fn get(&self, schema_id: u32) -> Option<&TableSchema> {
        self.schemas
            .get(schema_id as usize)
            .and_then(|slot| slot.as_ref())
    }

    /// True when no schemas have ever been recorded.
    pub fn is_empty(&self) -> bool {
        self.schemas.iter().all(|s| s.is_none())
    }

    /// Iterate over all known schemas in ascending `schema_id` order,
    /// skipping empty slots.
    pub fn iter(&self) -> impl Iterator<Item = &TableSchema> {
        self.schemas.iter().filter_map(|s| s.as_ref())
    }

    /// The highest `schema_id` seen, if any.
    pub fn max_schema_id(&self) -> Option<u32> {
        self.schemas
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, s)| s.as_ref().map(|_| i as u32))
    }

    /// Number of distinct schemas recorded.
    pub fn len(&self) -> usize {
        self.schemas.iter().filter(|s| s.is_some()).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schema() -> TableSchema {
        TableSchema {
            schema_id: 3,
            parent_schema_id: Some(2),
            fields: vec![
                SchemaField::new(1, "user_id", FluxDType::UInt64).with_nullable(false),
                SchemaField::new(2, "region", FluxDType::Utf8)
                    .with_default(DefaultValue::String("unknown".into()))
                    .with_doc("User's IP-derived region code."),
                SchemaField::new(3, "revenue_cents", FluxDType::Int64),
            ],
            change_summary: Some("added region; promoted revenue from int32".into()),
        }
    }

    #[test]
    fn schema_json_round_trip() {
        let s = sample_schema();
        let json = serde_json::to_string(&s).unwrap();
        let parsed: TableSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn field_lookups() {
        let s = sample_schema();
        assert_eq!(s.field_by_id(2).unwrap().name, "region");
        assert_eq!(s.field_by_name("user_id").unwrap().field_id, 1);
        assert_eq!(s.max_field_id(), Some(3));
    }

    #[test]
    fn chain_insert_and_lookup() {
        let mut chain = SchemaChain::new();
        assert!(chain.is_empty());
        chain.insert(TableSchema::new(vec![SchemaField::new(
            1,
            "id",
            FluxDType::UInt64,
        )]));
        chain.insert(sample_schema());
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.max_schema_id(), Some(3));
        assert!(chain.get(0).is_some());
        assert!(chain.get(3).is_some());
        assert!(chain.get(1).is_none());
        assert!(chain.get(99).is_none());
    }

    #[test]
    fn chain_last_wins_for_same_id() {
        let mut chain = SchemaChain::new();
        chain.insert(TableSchema::new(vec![SchemaField::new(
            1,
            "id",
            FluxDType::UInt32,
        )]));
        chain.insert(TableSchema::new(vec![SchemaField::new(
            1,
            "id",
            FluxDType::UInt64,
        )]));
        assert_eq!(chain.get(0).unwrap().fields[0].dtype, FluxDType::UInt64);
    }

    #[test]
    fn default_value_json() {
        // Ensure untagged default variants serialize into their natural
        // JSON shapes (so logs stay small and human-readable).
        let s = DefaultValue::String("hello".into());
        assert_eq!(serde_json::to_string(&s).unwrap(), "\"hello\"");
        let i = DefaultValue::Int(-5);
        assert_eq!(serde_json::to_string(&i).unwrap(), "-5");
        let b = DefaultValue::Bool(true);
        assert_eq!(serde_json::to_string(&b).unwrap(), "true");
    }

    #[test]
    fn optional_fields_skipped_when_unset() {
        let field = SchemaField::new(7, "x", FluxDType::Int32);
        let json = serde_json::to_string(&field).unwrap();
        assert!(!json.contains("default"));
        assert!(!json.contains("doc"));
    }
}
