// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Round-trip tests for the `formats` module.
//!
//! For every supported file format we:
//! 1. Build a synthetic Arrow `RecordBatch`,
//! 2. Persist it via `formats::save_batches`,
//! 3. Re-read it via `formats::load_batches`,
//! 4. Verify row count, column names, and a handful of cell values match.
//!
//! In addition, a `compress_then_decompress_*` test confirms the full
//! `formats → FluxWriter → FluxReader → formats` pipeline preserves data.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::*;
use arrow_schema::{DataType, Field, Schema};
use tempfile::TempDir;

use fluxcapacitor::formats::{FileFormat, load_batches, save_batches};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn sample_batch(n: usize) -> RecordBatch {
    let id: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
    let revenue: ArrayRef = Arc::new(Float64Array::from(
        (0..n).map(|i| (i as f64) * 1.25).collect::<Vec<_>>(),
    ));
    let region: ArrayRef = Arc::new(StringArray::from(
        (0..n)
            .map(|i| match i % 4 {
                0 => "north",
                1 => "south",
                2 => "east",
                _ => "west",
            })
            .collect::<Vec<_>>(),
    ));
    let active: ArrayRef = Arc::new(BooleanArray::from(
        (0..n).map(|i| i % 3 != 0).collect::<Vec<_>>(),
    ));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("revenue", DataType::Float64, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("active", DataType::Boolean, false),
    ]));
    RecordBatch::try_new(schema, vec![id, revenue, region, active]).unwrap()
}

/// Sum the rows across multiple batches.
fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Compare two batches structurally (column count, row count, and the
/// *set* of column names). Order is intentionally ignored because some
/// readers (notably Arrow's JSON reader) emit fields in alphabetical
/// order rather than in insertion order.
fn assert_shape(orig: &RecordBatch, got: &[RecordBatch]) {
    assert!(!got.is_empty(), "no batches read back");
    assert_eq!(total_rows(got), orig.num_rows(), "row count mismatch");
    assert_eq!(
        got[0].schema().fields().len(),
        orig.schema().fields().len(),
        "column count mismatch"
    );
    let want: std::collections::BTreeSet<String> = orig
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    let have: std::collections::BTreeSet<String> = got[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();
    assert_eq!(want, have, "column name set differs");
}

/// Find the column index for `name` in `batch`'s schema.
fn col_idx(batch: &RecordBatch, name: &str) -> usize {
    batch.schema().index_of(name).unwrap()
}

/// Stitch multiple batches into a single view of one named column by
/// stringifying every value via Arrow's display formatter. Useful for
/// content equality checks across formats whose schemas may not match
/// exactly (e.g. CSV reads everything as inferred types).
fn stringify_named(batches: &[RecordBatch], name: &str) -> Vec<String> {
    use arrow::util::display::{ArrayFormatter, FormatOptions};
    let opts = FormatOptions::default();
    let mut out = Vec::new();
    for b in batches {
        let idx = col_idx(b, name);
        let arr = b.column(idx);
        let f = ArrayFormatter::try_new(arr.as_ref(), &opts).unwrap();
        for i in 0..arr.len() {
            out.push(f.value(i).to_string());
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-format round trips
// ─────────────────────────────────────────────────────────────────────────────

fn run_round_trip(extension: &str, expect_format: FileFormat) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join(format!("data.{extension}"));

    // Sanity-check the format detector.
    assert_eq!(FileFormat::from_path(&path).unwrap(), expect_format);

    let original = sample_batch(64);
    save_batches(&path, std::slice::from_ref(&original))
        .unwrap_or_else(|e| panic!("save_batches({extension}): {e}"));

    let read = load_batches(&path).unwrap_or_else(|e| panic!("load_batches({extension}): {e}"));
    assert_shape(&original, &read);

    // Compare stringified content per named column so reorderings (e.g.
    // alphabetised JSON schemas) don't cause spurious failures.
    for field in original.schema().fields() {
        let name = field.name();
        let lhs = stringify_named(std::slice::from_ref(&original), name);
        let rhs = stringify_named(&read, name);
        assert_eq!(lhs, rhs, "{extension} column {:?} mismatch", name,);
    }
}

#[test]
fn round_trip_csv() {
    run_round_trip("csv", FileFormat::Csv);
}
#[test]
fn round_trip_tsv() {
    run_round_trip("tsv", FileFormat::Tsv);
}
#[test]
fn round_trip_ndjson() {
    run_round_trip("ndjson", FileFormat::NdJson);
}
#[test]
fn round_trip_jsonl() {
    run_round_trip("jsonl", FileFormat::NdJson);
}
#[test]
fn round_trip_json() {
    run_round_trip("json", FileFormat::Json);
}
#[test]
fn round_trip_parquet() {
    run_round_trip("parquet", FileFormat::Parquet);
}
#[test]
fn round_trip_arrow_ipc() {
    run_round_trip("arrow", FileFormat::ArrowIpc);
}
#[test]
fn round_trip_feather() {
    run_round_trip("feather", FileFormat::ArrowIpc);
}
#[test]
fn round_trip_orc() {
    run_round_trip("orc", FileFormat::Orc);
}

// XLSX is special: the writer is rust_xlsxwriter, the reader is calamine.
// We test it on its own because the inferred Excel schema (Int64/Float64/
// Boolean/Utf8) matches our synthetic batch, but we relax the float column
// because Excel stores as f64 and reads back as either Int64 or Float64
// depending on whether all values were whole.
#[test]
fn round_trip_xlsx() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("data.xlsx");
    let original = sample_batch(32);

    save_batches(&path, std::slice::from_ref(&original)).unwrap();
    let read = load_batches(&path).unwrap();
    assert_shape(&original, &read);

    // Compare the integer "id" column verbatim.
    let id_orig: &Int64Array = original.column(0).as_primitive();
    let id_read: &Int64Array = read[0].column(0).as_primitive();
    assert_eq!(
        id_orig.values().to_vec(),
        id_read.values().to_vec(),
        "xlsx id column mismatch"
    );

    // Spot-check the string column.
    let region_orig: &StringArray = original.column(2).as_string();
    let region_read: &StringArray = read[0].column(2).as_string();
    for i in 0..region_orig.len() {
        assert_eq!(
            region_orig.value(i),
            region_read.value(i),
            "xlsx region row {i}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Read-only formats
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn save_to_readonly_format_errors() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope.xls");
    let batch = sample_batch(4);
    let err =
        save_batches(&path, std::slice::from_ref(&batch)).expect_err("xls writes should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("read-only"),
        "expected read-only error, got {msg}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Full pipeline: format → flux → format
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn compress_then_decompress_via_csv_pipeline() {
    use loom::compressors::flux_writer::FluxWriter;
    use loom::decompressors::flux_reader::FluxReader;
    use loom::traits::{LoomCompressor, LoomDecompressor};

    let tmp = TempDir::new().unwrap();
    let csv_in = tmp.path().join("input.csv");
    let csv_out = tmp.path().join("output.csv");
    let flux_bin = tmp.path().join("data.flux");

    // Mixed-type batch — exercises the multi-column flat-numeric path that
    // used to corrupt Float64 columns before the BitWriter/BitReader fix in
    // `loom::bit_io`.
    let original = sample_batch(128);
    save_batches(&csv_in, std::slice::from_ref(&original)).unwrap();

    // CSV → flux
    let batches = load_batches(&csv_in).unwrap();
    let flux_bytes = FluxWriter::new().compress_all(&batches).unwrap();
    std::fs::write(&flux_bin, &flux_bytes).unwrap();

    // flux → CSV
    let bytes = std::fs::read(&flux_bin).unwrap();
    let batch = FluxReader::new("value").decompress_all(&bytes).unwrap();
    save_batches(&csv_out, std::slice::from_ref(&batch)).unwrap();

    // Re-read the resulting CSV and compare every column by name so column
    // ordering doesn't matter (some readers alphabetise fields).
    let result = load_batches(&csv_out).unwrap();
    assert_eq!(total_rows(&result), original.num_rows());
    for field in original.schema().fields() {
        let name = field.name();
        assert_eq!(
            stringify_named(std::slice::from_ref(&original), name),
            stringify_named(&result, name),
            "column {:?} differs after CSV→flux→CSV round trip",
            name,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-format conversion: Parquet → CSV via the formats module alone.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parquet_to_csv_conversion() {
    let tmp = TempDir::new().unwrap();
    let parquet_path = tmp.path().join("input.parquet");
    let csv_path = tmp.path().join("output.csv");

    let original = sample_batch(50);
    save_batches(&parquet_path, std::slice::from_ref(&original)).unwrap();

    let from_parquet = load_batches(&parquet_path).unwrap();
    save_batches(&csv_path, &from_parquet).unwrap();

    let from_csv = load_batches(&csv_path).unwrap();
    assert_eq!(total_rows(&from_csv), original.num_rows());
}

/// Regression: Spark/Pandas/Polars pipelines often produce batches where
/// columns have *different* null counts. Earlier `extract_column_data`
/// dropped null rows, leaving columns at different lengths and tripping
/// `RecordBatch::try_new`'s "all columns must have the same length"
/// validation on decompress.
#[test]
fn compress_then_decompress_multicol_with_mixed_nulls() {
    use loom::compressors::flux_writer::FluxWriter;
    use loom::decompressors::flux_reader::FluxReader;
    use loom::traits::{LoomCompressor, LoomDecompressor};

    let n = 64;

    // Column A (Int64): every 5th row is null.
    let id: ArrayRef = Arc::new(Int64Array::from(
        (0..n as i64)
            .map(|i| if i % 5 == 0 { None } else { Some(i) })
            .collect::<Vec<_>>(),
    ));
    // Column B (Float64): different null pattern — every 7th row.
    let revenue: ArrayRef = Arc::new(Float64Array::from(
        (0..n)
            .map(|i| {
                if i % 7 == 0 {
                    None
                } else {
                    Some((i as f64) * 1.25)
                }
            })
            .collect::<Vec<_>>(),
    ));
    // Column C (Boolean): every 3rd row null — yet another distinct count.
    let active: ArrayRef = Arc::new(BooleanArray::from(
        (0..n)
            .map(|i| if i % 3 == 0 { None } else { Some(i % 2 == 0) })
            .collect::<Vec<_>>(),
    ));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("revenue", DataType::Float64, true),
        Field::new("active", DataType::Boolean, true),
    ]));
    let original = RecordBatch::try_new(schema, vec![id, revenue, active]).unwrap();

    let bytes = FluxWriter::new().compress(&original).unwrap();
    let out = FluxReader::new("value").decompress_all(&bytes).unwrap();

    // The actual fix: every column has the same length as the input.
    assert_eq!(out.num_rows(), n);
    for col in 0..out.num_columns() {
        assert_eq!(
            out.column(col).len(),
            n,
            "column {col} length mismatch (this was the original bug)",
        );
    }
}
