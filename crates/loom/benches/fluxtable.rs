// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for the Phase F FluxTable API.
//!
//! Covers:
//! * `FluxTable::append`          — compress + write + log entry
//! * `FluxTable::scan` (iterator) — streaming read across N files
//! * `FluxTable::evolve_schema`   — pure metadata write (JSON log entry)
//! * End-to-end compress → append — combined Rust compressor + I/O
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench fluxtable
//! ```
//!
//! Or with HTML report:
//!
//! ```text
//! cargo bench --bench fluxtable -- --output-format html
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use tempfile::TempDir;

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};

use loom::{
    compressors::flux_writer::FluxWriter,
    dtype::FluxDType,
    traits::LoomCompressor,
    txn::{
        schema::{SchemaField, TableSchema},
        EvolveOptions, FluxTable,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::UInt64, false),
        Field::new("revenue", DataType::Int64, true),
    ]))
}

fn make_batch(schema: Arc<Schema>, n: usize, offset: u64) -> RecordBatch {
    let ids: UInt64Array = (offset..offset + n as u64).collect();
    let rev: Int64Array = (0..n as i64).map(|i| (i * 37) % 99_999).collect();
    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(rev)]).unwrap()
}

fn compress(batch: &RecordBatch) -> Vec<u8> {
    FluxWriter::new().compress(batch).unwrap()
}

/// Open a table and stamp it with the initial schema.
fn open_table(dir: &TempDir, name: &str) -> FluxTable {
    let tbl = FluxTable::open(dir.path().join(name)).unwrap();
    let schema = TableSchema::new(vec![
        SchemaField::new(1, "user_id", FluxDType::UInt64).with_nullable(false),
        SchemaField::new(2, "revenue", FluxDType::Int64),
    ]);
    tbl.evolve_schema(schema).unwrap();
    tbl
}

fn schema_v2() -> TableSchema {
    TableSchema::new(vec![
        SchemaField::new(1, "user_id", FluxDType::UInt64).with_nullable(false),
        SchemaField::new(2, "revenue", FluxDType::Int64),
        SchemaField::new(3, "region", FluxDType::UInt64),
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// FluxTable::append — log entry + file write
// ─────────────────────────────────────────────────────────────────────────────

fn bench_append(c: &mut Criterion) {
    let schema = make_schema();
    let mut g = c.benchmark_group("fluxtable/append");

    for &rows in &[1_024usize, 65_536, 524_288] {
        let batch = make_batch(schema.clone(), rows, 0);
        let flux_bytes = compress(&batch);

        g.throughput(Throughput::Elements(rows as u64));
        g.bench_with_input(
            BenchmarkId::new("rows", rows),
            &flux_bytes,
            |b, bytes| {
                // One fresh table per benchmark group iteration; each
                // `b.iter` call appends to the same table (steady-state).
                let tmp = TempDir::new().unwrap();
                let tbl = open_table(&tmp, "append.fluxtable");
                // One warm-up append before timing
                tbl.append(bytes).unwrap();

                b.iter(|| tbl.append(black_box(bytes)).unwrap());
            },
        );
    }
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// FluxTable::scan — streaming read over N pre-written files
// ─────────────────────────────────────────────────────────────────────────────

fn bench_scan(c: &mut Criterion) {
    let schema = make_schema();
    let mut g = c.benchmark_group("fluxtable/scan");

    for &(rows_per_file, num_files) in &[(65_536usize, 4usize), (524_288, 2)] {
        let total = rows_per_file * num_files;
        g.throughput(Throughput::Elements(total as u64));

        g.bench_with_input(
            BenchmarkId::new("total_rows", total),
            &(rows_per_file, num_files),
            |b, &(rpf, nf)| {
                // Setup: build the table once, bench only the scan.
                let tmp = TempDir::new().unwrap();
                let tbl = open_table(&tmp, "scan.fluxtable");
                for i in 0..nf {
                    let batch = make_batch(schema.clone(), rpf, (i * rpf) as u64);
                    tbl.append(&compress(&batch)).unwrap();
                }

                b.iter(|| {
                    let mut rows = 0usize;
                    for item in tbl.scan().unwrap() {
                        rows += item.unwrap().num_rows();
                    }
                    black_box(rows)
                });
            },
        );
    }
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// FluxTable::evolve_schema — pure metadata write (JSON log entry, no data)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_evolve_schema(c: &mut Criterion) {
    let mut g = c.benchmark_group("fluxtable/evolve_schema");

    // Use iter_batched so each iteration gets a fresh table: avoids the
    // log growing unboundedly and conflating log-read overhead with the
    // evolution cost we care about.
    g.bench_function("add_column", |b| {
        b.iter_batched(
            || {
                let tmp = TempDir::new().unwrap();
                let tbl = open_table(&tmp, "evo.fluxtable");
                (tmp, tbl) // hold tmp so the dir stays alive
            },
            |(_tmp, tbl)| {
                tbl.evolve_schema(black_box(schema_v2())).unwrap()
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end: compress → append (Rust compressor + I/O combined)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_compress_and_append(c: &mut Criterion) {
    let schema = make_schema();
    let mut g = c.benchmark_group("fluxtable/compress_and_append");

    for &rows in &[65_536usize, 524_288] {
        let batch = make_batch(schema.clone(), rows, 0);
        g.throughput(Throughput::Elements(rows as u64));

        g.bench_with_input(
            BenchmarkId::new("rows", rows),
            &batch,
            |b, batch| {
                let tmp = TempDir::new().unwrap();
                let tbl = open_table(&tmp, "e2e.fluxtable");

                b.iter(|| {
                    let bytes = FluxWriter::new().compress(black_box(batch)).unwrap();
                    tbl.append(&bytes).unwrap();
                });
            },
        );
    }
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema evolution with Phase D null-tightening validation path
// (validates the proof-checking fast path doesn't regress)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_evolve_with_null_tightening_reject(c: &mut Criterion) {
    let mut g = c.benchmark_group("fluxtable/evolve_schema_reject");

    // This bench exercises the *rejected* tightening path — the validator
    // reads manifests and returns an error. We want to make sure the
    // rejection is fast (no partial state written).
    g.bench_function("reject_tightening", |b| {
        b.iter_batched(
            || {
                let tmp = TempDir::new().unwrap();
                let tbl = open_table(&tmp, "reject.fluxtable");
                (tmp, tbl)
            },
            |(_tmp, tbl)| {
                // Attempting to tighten nullability without proof must
                // return an error fast — nothing written to disk.
                let tight_schema = TableSchema::new(vec![
                    SchemaField::new(1, "user_id", FluxDType::UInt64).with_nullable(false),
                    SchemaField::new(2, "revenue", FluxDType::Int64).with_nullable(false),
                ]);
                let result = tbl.evolve_schema_with_options(
                    black_box(tight_schema),
                    EvolveOptions::default(), // allow_null_tightening = false
                );
                // Error is expected here — just black_box the result.
                black_box(result.is_err())
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_append,
    bench_scan,
    bench_evolve_schema,
    bench_compress_and_append,
    bench_evolve_with_null_tightening_reject,
);
criterion_main!(benches);
