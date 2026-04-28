// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0
//
//! End-to-end file-format benchmarks.
//!
//! Each iteration builds a fresh tempdir, persists a synthetic mixed-type
//! `RecordBatch` to disk in the target format, and times either:
//!   * `save`              — Arrow → file (write throughput),
//!   * `load`              — file  → Arrow (read throughput),
//!   * `compress_pipeline` — file  → Arrow → `.flux` (full ingest path).
//!
//! Excel `.xlsx` is included for read+write timing.  Read-only formats
//! (`.xls`/`.xlsm`/`.ods`) are not benchmarked because they cannot be
//! produced from this harness without a pre-existing fixture.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};

use fluxcapacitor::formats::{load_batches, save_batches, FileFormat};
use loom::compressors::flux_writer::FluxWriter;
use loom::traits::LoomCompressor;

// ─────────────────────────────────────────────────────────────────────────────
// Data generator
// ─────────────────────────────────────────────────────────────────────────────

fn mixed_batch(rows: usize) -> RecordBatch {
    let id: ArrayRef = Arc::new(Int64Array::from((0..rows as i64).collect::<Vec<_>>()));
    let revenue: ArrayRef = Arc::new(Float64Array::from(
        (0..rows).map(|i| (i as f64) * 1.25).collect::<Vec<_>>(),
    ));
    let region: ArrayRef = Arc::new(StringArray::from(
        (0..rows)
            .map(|i| match i % 4 {
                0 => "north",
                1 => "south",
                2 => "east",
                _ => "west",
            })
            .collect::<Vec<_>>(),
    ));
    let active: ArrayRef = Arc::new(BooleanArray::from(
        (0..rows).map(|i| i % 3 != 0).collect::<Vec<_>>(),
    ));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id",      DataType::Int64,   false),
        Field::new("revenue", DataType::Float64, false),
        Field::new("region",  DataType::Utf8,    false),
        Field::new("active",  DataType::Boolean, false),
    ]));
    RecordBatch::try_new(schema, vec![id, revenue, region, active]).unwrap()
}

/// Approximate raw size in bytes of a `RecordBatch` (sum of column buffers).
fn raw_size(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(|c| c.get_array_memory_size()).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark groups
// ─────────────────────────────────────────────────────────────────────────────

const ROWS: usize = 50_000;

/// Formats included in `save` + `load` benchmarks. Read-only formats
/// (`.xls`, `.xlsm`, `.ods`) are excluded because the harness can't produce
/// them.
fn write_formats() -> Vec<(&'static str, FileFormat)> {
    vec![
        ("csv",      FileFormat::Csv),
        ("tsv",      FileFormat::Tsv),
        ("ndjson",   FileFormat::NdJson),
        ("json",     FileFormat::Json),
        ("parquet",  FileFormat::Parquet),
        ("arrow",    FileFormat::ArrowIpc),
        ("orc",      FileFormat::Orc),
        ("xlsx",     FileFormat::Xlsx),
    ]
}

fn bench_save(c: &mut Criterion) {
    let batch = mixed_batch(ROWS);
    let raw = raw_size(&batch);

    let mut group = c.benchmark_group("save");
    group.throughput(Throughput::Bytes(raw as u64));
    group.measurement_time(Duration::from_secs(6));

    for (ext, _fmt) in write_formats() {
        group.bench_with_input(BenchmarkId::from_parameter(ext), ext, |b, ext| {
            b.iter_with_setup(
                || TempDir::new().unwrap(),
                |tmp| {
                    let path = tmp.path().join(format!("data.{ext}"));
                    save_batches(&path, std::slice::from_ref(black_box(&batch))).unwrap();
                    black_box(path);
                },
            );
        });
    }
    group.finish();
}

fn bench_load(c: &mut Criterion) {
    let batch = mixed_batch(ROWS);
    let raw = raw_size(&batch);

    let mut group = c.benchmark_group("load");
    group.throughput(Throughput::Bytes(raw as u64));
    group.measurement_time(Duration::from_secs(6));

    for (ext, _fmt) in write_formats() {
        // Persist once outside the timed region.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join(format!("data.{ext}"));
        save_batches(&path, std::slice::from_ref(&batch)).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(ext), &path, |b, path| {
            b.iter(|| {
                let read = load_batches(black_box(path)).unwrap();
                black_box(read);
            });
        });

        drop(tmp);
    }
    group.finish();
}

fn bench_compress_pipeline(c: &mut Criterion) {
    let batch = mixed_batch(ROWS);
    let raw = raw_size(&batch);

    let mut group = c.benchmark_group("compress_pipeline");
    group.throughput(Throughput::Bytes(raw as u64));
    group.measurement_time(Duration::from_secs(6));

    let writer = FluxWriter::new();

    for (ext, _fmt) in write_formats() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join(format!("data.{ext}"));
        save_batches(&path, std::slice::from_ref(&batch)).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(ext), &path, |b, path| {
            b.iter(|| {
                let batches = load_batches(black_box(path)).unwrap();
                let bytes = writer.compress_all(&batches).unwrap();
                black_box(bytes);
            });
        });

        drop(tmp);
    }
    group.finish();
}

criterion_group!(benches, bench_save, bench_load, bench_compress_pipeline);
criterion_main!(benches);
