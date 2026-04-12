// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `fluxcapacitor bench` — native Rust file-based benchmark.
//!
//! Compares FluxCompress (Speed/Balanced/Archive + mmap) against
//! Parquet (Snappy/Zstd) and Arrow IPC (LZ4), all through Rust-native
//! codepaths with zero Python overhead.

use std::fs::{self, File};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;
use anyhow::Result;

use arrow::ipc::{reader::FileReader as IpcReader, writer::FileWriter as IpcWriter};
use arrow_array::{RecordBatch, UInt64Array, Int64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression as PqCompression;
use parquet::file::properties::WriterProperties;

use loom::{
    CompressionProfile,
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    traits::{LoomCompressor, LoomDecompressor, Predicate},
};

pub fn cmd_bench(rows: usize, pattern: &str) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  FluxCapacitor — Native Rust Benchmark (no Python overhead)     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Run single-column test.
    let batch1 = generate_single(rows, pattern);
    let raw1 = rows * 8;
    println!("\n── Single Column ({} rows, {}, {}) ──",
        fmt_rows(rows), pattern, human(raw1));
    run_comparison(&batch1, raw1, pattern, "single")?;

    // Run multi-column test (4 cols).
    if pattern == "sequential" || pattern == "random" {
        let batch4 = generate_multi(rows);
        let raw4 = rows * 8 * 4;
        println!("\n── Multi-Column ({} rows × 4 cols, {}) ──",
            fmt_rows(rows), human(raw4));
        run_comparison(&batch4, raw4, pattern, "multi")?;
    }

    println!("\n  ✓ Benchmark complete.");
    Ok(())
}

fn run_comparison(batch: &RecordBatch, raw_bytes: usize, pattern: &str, kind: &str) -> Result<()> {
    println!(
        "\n  {:<24} {:>10} {:>8} {:>10} {:>10} {:>10}",
        "Format", "Size", "Ratio", "Comp MB/s", "Dec MB/s", "mmap MB/s"
    );
    println!("  {}", "─".repeat(76));

    // ── FluxCompress profiles ────────────────────────────────────────────
    for (name, profile) in [
        ("Flux (speed)",    CompressionProfile::Speed),
        ("Flux (balanced)", CompressionProfile::Balanced),
        ("Flux (archive)",  CompressionProfile::Archive),
    ] {
        let writer = FluxWriter::with_profile(profile);

        let t0 = Instant::now();
        let flux_bytes = writer.compress(batch)?;
        let c_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let tmp = format!("/tmp/fluxbench_{kind}_{pattern}_{name}.flux");
        fs::write(&tmp, &flux_bytes)?;

        let reader = FluxReader::new("value");
        let t1 = Instant::now();
        let _ = reader.decompress_all(&flux_bytes)?;
        let d_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let _ = reader.decompress_file_all(std::path::Path::new(&tmp))?;
        let mm_ms = t2.elapsed().as_secs_f64() * 1000.0;

        print_row(name, flux_bytes.len(), raw_bytes, c_ms, d_ms, Some(mm_ms));
        let _ = fs::remove_file(&tmp);
    }

    // ── Parquet (Snappy) ────────────────────────────────────────────────
    {
        let tmp = format!("/tmp/fluxbench_{kind}_{pattern}_pq_snappy.parquet");
        let props = WriterProperties::builder()
            .set_compression(PqCompression::SNAPPY)
            .build();
        let file = File::create(&tmp)?;
        let t0 = Instant::now();
        let mut pw = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        pw.write(batch)?;
        pw.close()?;
        let c_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let size = fs::metadata(&tmp)?.len() as usize;

        let t1 = Instant::now();
        let file = File::open(&tmp)?;
        let pr = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let _batches: Vec<RecordBatch> = pr.collect::<std::result::Result<Vec<_>, _>>()?;
        let d_ms = t1.elapsed().as_secs_f64() * 1000.0;

        print_row("Parquet (snappy)", size, raw_bytes, c_ms, d_ms, None);
        let _ = fs::remove_file(&tmp);
    }

    // ── Parquet (Zstd) ──────────────────────────────────────────────────
    {
        let tmp = format!("/tmp/fluxbench_{kind}_{pattern}_pq_zstd.parquet");
        let props = WriterProperties::builder()
            .set_compression(PqCompression::ZSTD(Default::default()))
            .build();
        let file = File::create(&tmp)?;
        let t0 = Instant::now();
        let mut pw = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        pw.write(batch)?;
        pw.close()?;
        let c_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let size = fs::metadata(&tmp)?.len() as usize;

        let t1 = Instant::now();
        let file = File::open(&tmp)?;
        let pr = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        let _batches: Vec<RecordBatch> = pr.collect::<std::result::Result<Vec<_>, _>>()?;
        let d_ms = t1.elapsed().as_secs_f64() * 1000.0;

        print_row("Parquet (zstd)", size, raw_bytes, c_ms, d_ms, None);
        let _ = fs::remove_file(&tmp);
    }

    // ── Arrow IPC (uncompressed, as baseline) ───────────────────────────
    {
        let tmp = format!("/tmp/fluxbench_{kind}_{pattern}_ipc.arrow");
        let file = File::create(&tmp)?;
        let t0 = Instant::now();
        let mut iw = IpcWriter::try_new(file, batch.schema().as_ref())?;
        iw.write(batch)?;
        iw.finish()?;
        let c_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let size = fs::metadata(&tmp)?.len() as usize;

        let t1 = Instant::now();
        let file = File::open(&tmp)?;
        let ir = IpcReader::try_new(file, None)?;
        let _batches: Vec<RecordBatch> = ir.collect::<std::result::Result<Vec<_>, _>>()?;
        let d_ms = t1.elapsed().as_secs_f64() * 1000.0;

        print_row("Arrow IPC (raw)", size, raw_bytes, c_ms, d_ms, None);
        let _ = fs::remove_file(&tmp);
    }

    Ok(())
}

fn print_row(name: &str, size: usize, raw: usize, c_ms: f64, d_ms: f64, mm_ms: Option<f64>) {
    let ratio = raw as f64 / size as f64;
    let c_mbs = mbs(raw, c_ms);
    let d_mbs = mbs(raw, d_ms);
    let mm_str = mm_ms.map(|ms| format!("{:>9.0}", mbs(raw, ms)))
        .unwrap_or_else(|| "        -".to_string());
    println!(
        "  {:<24} {:>10} {:>7.1}x {:>9.0} {:>9.0} {}",
        name, human(size), ratio, c_mbs, d_mbs, mm_str,
    );
}

// ─────────────────────────────────────────────────────────────────────────────

fn generate_single(rows: usize, pattern: &str) -> RecordBatch {
    let values: Vec<u64> = match pattern {
        "sequential" => (0u64..rows as u64).collect(),
        "constant"   => vec![42u64; rows],
        "random"     => {
            let mut s: u64 = 0xDEAD_BEEF_CAFE_BABE;
            (0..rows).map(|_| { s ^= s<<13; s ^= s>>7; s ^= s<<17; s & 0xFFFF_FFFF_FFFF }).collect()
        }
        _ => (0u64..rows as u64).collect(),
    };
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::UInt64, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(values))]).unwrap()
}

fn generate_multi(rows: usize) -> RecordBatch {
    let user_id: Vec<u64> = (0..rows as u64).collect();
    let revenue: Vec<u64> = (0..rows as u64).map(|i| (i * 37) % 99_999).collect();
    let region:  Vec<u64> = (0..rows as u64).map(|i| i % 8).collect();
    let session: Vec<i64> = (0..rows as i64).map(|i| (i * 1234) % 86_400_000).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id",    DataType::UInt64, false),
        Field::new("revenue",    DataType::UInt64, false),
        Field::new("region",     DataType::UInt64, false),
        Field::new("session_ms", DataType::Int64,  false),
    ]));
    RecordBatch::try_new(schema, vec![
        Arc::new(UInt64Array::from(user_id)),
        Arc::new(UInt64Array::from(revenue)),
        Arc::new(UInt64Array::from(region)),
        Arc::new(Int64Array::from(session)),
    ]).unwrap()
}

fn human(bytes: usize) -> String {
    if bytes < 1024 { format!("{bytes} B") }
    else if bytes < 1024*1024 { format!("{:.1} KB", bytes as f64/1024.0) }
    else if bytes < 1024*1024*1024 { format!("{:.1} MB", bytes as f64/(1024.0*1024.0)) }
    else { format!("{:.2} GB", bytes as f64/(1024.0*1024.0*1024.0)) }
}

fn fmt_rows(n: usize) -> String {
    if n >= 1_000_000 { format!("{}M", n/1_000_000) }
    else if n >= 1_000 { format!("{}K", n/1_000) }
    else { format!("{n}") }
}

fn mbs(bytes: usize, ms: f64) -> f64 {
    if ms <= 0.0 { f64::INFINITY } else { (bytes as f64/(1024.0*1024.0))/(ms/1000.0) }
}
