// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `fluxcapacitor compare-bench` — three-way comparison against
//! Parquet and Delta Lake on a realistic 22-column mixed-schema
//! workload.
//!
//! ## How we measure Delta Lake honestly
//!
//! Delta Lake's storage format is exactly Parquet data files + a JSON
//! transaction log under `_delta_log/`. We produce the Delta artefact
//! by writing the same Parquet file we already use for the Parquet
//! comparison, then emit the minimum commit record Delta Lake needs
//! to make the directory a valid Delta table:
//!
//! ```text
//! /tmp/.../table/
//!   ├─ part-00000-UUID.c000.parquet         (the Parquet file)
//!   └─ _delta_log/00000000000000000000.json (protocol + metaData + add)
//! ```
//!
//! The reported "Delta Lake" size is `parquet_bytes + log_bytes` —
//! that's the actual on-disk cost a Delta reader would see. Delta's
//! read path ultimately reads Parquet bytes, so we reuse the Parquet
//! decompression numbers for "Dec MB/s" and annotate the small extra
//! cost of reading + parsing the JSON log at open time.

use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use arrow_array::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression as PqCompression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use loom::{
    CompressionProfile,
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    traits::{LoomCompressor, LoomDecompressor, Predicate},
};

// We reuse the Databricks-shaped 22-column RecordBatch from
// `mixed_bench` so the three codecs are compared on identical data.
use crate::mixed_bench;

/// Which synthetic dataset to compare on.
pub enum CompareSchema {
    /// 22-column mixed schema (Databricks-shaped).
    Mixed,
    /// 10 string columns + 2 ids + 1 timestamp — log/event workloads.
    StringHeavy,
    /// 10 Float64 columns + 2 ids + 1 timestamp — scientific / IoT /
    /// financial workloads. Historically Flux's weakest dtype vs
    /// Parquet on the decompression side.
    FloatHeavy,
}

pub fn cmd_compare_bench(rows: usize) -> Result<()> {
    cmd_compare_bench_with_schema(rows, CompareSchema::Mixed)
}

pub fn cmd_string_compare_bench(rows: usize) -> Result<()> {
    cmd_compare_bench_with_schema(rows, CompareSchema::StringHeavy)
}

pub fn cmd_float_compare_bench(rows: usize) -> Result<()> {
    cmd_compare_bench_with_schema(rows, CompareSchema::FloatHeavy)
}

fn cmd_compare_bench_with_schema(rows: usize, variant: CompareSchema) -> Result<()> {
    let title = match variant {
        CompareSchema::Mixed       => "Mixed 22-col",
        CompareSchema::StringHeavy => "String-heavy (10 string + 2 id + 1 ts)",
        CompareSchema::FloatHeavy  => "Float-heavy (10 float + 2 id + 1 ts)",
    };
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  Flux vs Parquet vs Delta Lake — {} ({})", title, fmt_rows(rows));
    println!("╚═══════════════════════════════════════════════════════════════════════╝");

    // Build the shared dataset.
    let batch = match variant {
        CompareSchema::Mixed       => mixed_bench::build_batch_for_compare(rows),
        CompareSchema::StringHeavy => mixed_bench::build_string_heavy_batch(rows),
        CompareSchema::FloatHeavy  => mixed_bench::build_float_heavy_batch(rows),
    };
    let arrow_bytes = mixed_bench::estimate_arrow_bytes_for_compare(&batch);
    println!("  Rows:  {}", fmt_rows(rows));
    println!("  Columns: {}", batch.num_columns());
    println!("  Approx in-memory Arrow size: {}", human(arrow_bytes));
    println!();

    // ── Flux (Archive) ───────────────────────────────────────────────────
    let writer = FluxWriter::with_profile(CompressionProfile::Archive);
    let t0 = Instant::now();
    let flux_bytes = writer.compress(&batch)?;
    let flux_comp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let reader = FluxReader::new("value");
    let t1 = Instant::now();
    let _ = reader.decompress(&flux_bytes, &Predicate::None)?;
    let flux_dec_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // ── Parquet (zstd-3) ─────────────────────────────────────────────────
    let tmpdir = tempfile::tempdir()?;
    let pq_path = tmpdir.path().join("mixed.parquet");
    let props = WriterProperties::builder()
        .set_compression(PqCompression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let file = File::create(&pq_path)?;
    let t2 = Instant::now();
    let mut pw = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    pw.write(&batch)?;
    pw.close()?;
    let pq_comp_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let pq_size = fs::metadata(&pq_path)?.len() as usize;

    let t3 = Instant::now();
    let file = File::open(&pq_path)?;
    let pr = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let _batches: Vec<RecordBatch> = pr.collect::<std::result::Result<Vec<_>, _>>()?;
    let pq_dec_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // ── Delta Lake (Parquet + _delta_log/ JSON) ──────────────────────────
    //
    // We copy the Parquet file into a fresh directory and write the minimum
    // _delta_log/ entries Delta Lake requires.  The on-disk footprint is
    // the sum of both.  The read path in Delta Lake would resolve the log,
    // find the live Parquet files, and hand them to the Parquet reader —
    // so the decompression cost is effectively the Parquet number plus a
    // one-shot log-parse overhead on table open.
    let delta_dir = tmpdir.path().join("delta");
    let delta_log_dir = delta_dir.join("_delta_log");
    fs::create_dir_all(&delta_log_dir)?;
    let data_filename = "part-00000-00000000-0000-0000-0000-000000000000.c000.snappy.parquet";
    let data_path = delta_dir.join(data_filename);
    fs::copy(&pq_path, &data_path)?;

    let parquet_size_u64 = fs::metadata(&data_path)?.len();
    let delta_log_path = delta_log_dir.join("00000000000000000000.json");
    let delta_log_body = make_delta_log_entry(&batch, data_filename, parquet_size_u64);
    fs::write(&delta_log_path, delta_log_body.as_bytes())?;

    let delta_data_size = fs::metadata(&data_path)?.len() as usize;
    let delta_log_size = fs::metadata(&delta_log_path)?.len() as usize;
    let delta_size = delta_data_size + delta_log_size;

    // Simulated Delta read: parse the log (cheap) + decompress Parquet.
    let t4 = Instant::now();
    // Delta log entries are line-delimited JSON — one record per line.
    let log_text = fs::read_to_string(&delta_log_path)?;
    for line in log_text.lines().filter(|l| !l.trim().is_empty()) {
        let _v: serde_json::Value = serde_json::from_str(line)?;
    }
    let file = File::open(&data_path)?;
    let pr = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let _batches: Vec<RecordBatch> = pr.collect::<std::result::Result<Vec<_>, _>>()?;
    let delta_dec_ms = t4.elapsed().as_secs_f64() * 1000.0;

    // ── Report ───────────────────────────────────────────────────────────
    println!("  {:<22} {:>12} {:>8} {:>12} {:>12}",
        "Codec", "Size", "Ratio", "Comp MB/s", "Dec MB/s");
    println!("  {}", "─".repeat(72));
    print_row("Flux (Archive)",   flux_bytes.len(), arrow_bytes, flux_comp_ms, flux_dec_ms);
    print_row("Parquet (zstd-3)", pq_size,          arrow_bytes, pq_comp_ms,   pq_dec_ms);
    print_row("Delta Lake (zstd-3)", delta_size,    arrow_bytes, pq_comp_ms,   delta_dec_ms);

    println!();
    println!("  (Delta = Parquet {} + _delta_log/ {} = {})",
        human(delta_data_size), human(delta_log_size), human(delta_size));
    if flux_bytes.len() < pq_size && flux_bytes.len() < delta_size {
        println!(
            "  ✓ Flux smaller than Parquet by {} ({:.1}%) and Delta Lake by {} ({:.1}%)",
            human(pq_size - flux_bytes.len()),
            100.0 * (pq_size - flux_bytes.len()) as f64 / pq_size as f64,
            human(delta_size - flux_bytes.len()),
            100.0 * (delta_size - flux_bytes.len()) as f64 / delta_size as f64,
        );
    }
    Ok(())
}

// ── Minimal Delta Lake log record ────────────────────────────────────────

fn make_delta_log_entry(batch: &RecordBatch, data_filename: &str, size: u64) -> String {
    let schema_json = serde_json::json!({
        "type": "struct",
        "fields": batch.schema().fields().iter().map(|f| serde_json::json!({
            "name": f.name(),
            "type": spark_type_for(f.data_type()),
            "nullable": f.is_nullable(),
            "metadata": {}
        })).collect::<Vec<_>>(),
    });

    let protocol = serde_json::json!({
        "protocol": {
            "minReaderVersion": 1,
            "minWriterVersion": 2,
        }
    });
    let metadata = serde_json::json!({
        "metaData": {
            "id": "00000000-0000-0000-0000-000000000000",
            "format": { "provider": "parquet", "options": {} },
            "schemaString": schema_json.to_string(),
            "partitionColumns": [],
            "configuration": {},
            "createdTime": 0i64,
        }
    });
    let add = serde_json::json!({
        "add": {
            "path": data_filename,
            "partitionValues": {},
            "size": size,
            "modificationTime": 0i64,
            "dataChange": true,
            "stats": format!("{{\"numRecords\": {}}}", batch.num_rows()),
        }
    });

    // Delta Lake's log is one JSON object per line.
    format!(
        "{}\n{}\n{}\n",
        serde_json::to_string(&protocol).unwrap(),
        serde_json::to_string(&metadata).unwrap(),
        serde_json::to_string(&add).unwrap(),
    )
}

fn spark_type_for(dt: &arrow_schema::DataType) -> String {
    use arrow_schema::DataType as D;
    match dt {
        D::Int64 | D::UInt64     => "long".into(),
        D::Int32 | D::UInt32     => "integer".into(),
        D::Int16 | D::UInt16     => "short".into(),
        D::Int8  | D::UInt8      => "byte".into(),
        D::Float32               => "float".into(),
        D::Float64               => "double".into(),
        D::Boolean               => "boolean".into(),
        D::Utf8 | D::LargeUtf8   => "string".into(),
        D::Binary | D::LargeBinary => "binary".into(),
        D::Date32 | D::Date64    => "date".into(),
        D::Timestamp(_, _)       => "timestamp".into(),
        D::Decimal128(p, s)      => format!("decimal({p},{s})"),
        _                        => "string".into(),
    }
}

// ── Pretty-print helpers ─────────────────────────────────────────────────

fn print_row(name: &str, size: usize, arrow_bytes: usize, c_ms: f64, d_ms: f64) {
    let ratio = arrow_bytes as f64 / size as f64;
    println!("  {:<22} {:>12} {:>7.2}x {:>11.0} {:>11.0}",
        name, human(size), ratio,
        mbs(arrow_bytes, c_ms), mbs(arrow_bytes, d_ms));
}

fn human(n: usize) -> String {
    const K: f64 = 1024.0;
    let f = n as f64;
    if f < K          { format!("{} B", n) }
    else if f < K*K   { format!("{:.1} KB", f / K) }
    else if f < K*K*K { format!("{:.1} MB", f / (K*K)) }
    else              { format!("{:.2} GB", f / (K*K*K)) }
}

fn fmt_rows(n: usize) -> String {
    if n >= 1_000_000 { format!("{}M", n / 1_000_000) }
    else if n >= 1_000 { format!("{}K", n / 1_000) }
    else { n.to_string() }
}

fn mbs(bytes: usize, ms: f64) -> f64 {
    if ms <= 0.0 { return 0.0; }
    (bytes as f64) / (1024.0 * 1024.0) / (ms / 1000.0)
}
