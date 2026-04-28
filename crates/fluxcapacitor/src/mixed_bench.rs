// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `fluxcapacitor mixed-bench` — realistic 22-column mixed-schema benchmark.
//!
//! Mirrors the real-world Databricks test case the user ran: 9.95M rows with
//! Int64 / Float64 / Datetime / Boolean / Date / String columns at
//! realistic cardinalities. Reports Flux vs Parquet size + throughput so the
//! impact of auto cross-column grouping and ALP can be measured directly.

use std::fs::{self, File};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampMillisecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
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

pub fn cmd_mixed_bench(rows: usize) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Mixed 22-column bench ({} rows) — mirrors Databricks test",
        fmt_rows(rows)
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let batch = build_batch(rows);
    let arrow_bytes = estimate_arrow_bytes(&batch);
    println!("  Rows:  {}", fmt_rows(rows));
    println!(
        "  Columns: {} ({})",
        batch.num_columns(),
        summarize_schema(&batch.schema())
    );
    println!("  Approx in-memory Arrow size: {}", human(arrow_bytes));
    println!();

    // ── Flux Archive profile ─────────────────────────────────────────────
    let writer = FluxWriter::with_profile(CompressionProfile::Archive);
    let t0 = Instant::now();
    let flux_bytes = writer.compress(&batch)?;
    let flux_comp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let reader = FluxReader::new("value");
    let t1 = Instant::now();
    let _ = reader.decompress(&flux_bytes, &Predicate::None)?;
    let flux_dec_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // ── Parquet Zstd ─────────────────────────────────────────────────────
    let tmp = "/tmp/fluxbench_mixed.parquet";
    let props = WriterProperties::builder()
        .set_compression(PqCompression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let file = File::create(tmp)?;
    let t2 = Instant::now();
    let mut pw = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    pw.write(&batch)?;
    pw.close()?;
    let pq_comp_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let pq_size = fs::metadata(tmp)?.len() as usize;

    let t3 = Instant::now();
    let file = File::open(tmp)?;
    let pr = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let _batches: Vec<RecordBatch> = pr.collect::<std::result::Result<Vec<_>, _>>()?;
    let pq_dec_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let _ = fs::remove_file(tmp);

    // ── Report ───────────────────────────────────────────────────────────
    println!(
        "  {:<22} {:>12} {:>8} {:>12} {:>12}",
        "Codec", "Size", "Ratio", "Comp MB/s", "Dec MB/s"
    );
    println!("  {}", "─".repeat(70));
    print_row(
        "Flux (Archive)",
        flux_bytes.len(),
        arrow_bytes,
        flux_comp_ms,
        flux_dec_ms,
    );
    print_row(
        "Parquet (zstd-3)",
        pq_size,
        arrow_bytes,
        pq_comp_ms,
        pq_dec_ms,
    );
    println!();
    if flux_bytes.len() < pq_size {
        println!(
            "  ✓ Flux smaller by {} ({:.1}%)",
            human(pq_size - flux_bytes.len()),
            100.0 * (pq_size - flux_bytes.len()) as f64 / pq_size as f64
        );
    } else {
        println!(
            "  ✗ Parquet smaller by {} ({:.1}%)",
            human(flux_bytes.len() - pq_size),
            100.0 * (flux_bytes.len() - pq_size) as f64 / flux_bytes.len() as f64
        );
    }
    Ok(())
}

fn print_row(name: &str, size: usize, arrow_bytes: usize, c_ms: f64, d_ms: f64) {
    let ratio = arrow_bytes as f64 / size as f64;
    println!(
        "  {:<22} {:>12} {:>7.2}x {:>11.0} {:>11.0}",
        name,
        human(size),
        ratio,
        mbs(arrow_bytes, c_ms),
        mbs(arrow_bytes, d_ms)
    );
}

// ── Schema mirror of the Databricks test ─────────────────────────────────

/// Public re-export used by the `compare_bench` module so all three
/// codecs run on the exact same RecordBatch.
pub fn build_batch_for_compare(rows: usize) -> RecordBatch {
    build_batch(rows)
}

/// Public re-export of the arrow-byte estimator for the compare-bench.
pub fn estimate_arrow_bytes_for_compare(batch: &RecordBatch) -> usize {
    estimate_arrow_bytes(batch)
}

/// Float-heavy RecordBatch: 10 Float64 columns with realistic numeric
/// distributions (prices, coordinates, rates, sensor readings) plus
/// 2 identifier columns and 1 timestamp.  Mirrors scientific /
/// financial / IoT workloads.
///
/// Floats are historically Flux's weakest dtype vs Parquet on the
/// decompression side — Parquet snappy decodes Float64 at ~2.3 GB/s
/// via byte-copy whereas our BitSlab + OutlierMap pipeline needs a
/// few extra steps. This bench is how we measure improvements there.
pub fn build_float_heavy_batch(rows: usize) -> RecordBatch {
    use arrow_array::{Float64Array, Int64Array, TimestampMillisecondArray};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};

    let mut s: u64 = 0xDEAD_F107_BEEF_1234_u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let randf = |r: u64, scale: f64| (r as f64 / u64::MAX as f64) * scale;

    // Identifier columns.
    let id: Vec<i64> = (0..rows as i64).collect();
    let user_id: Vec<i64> = (0..rows).map(|_| (rnd() % 5_000_000) as i64).collect();

    // Temporal.
    let base_ts = 1_700_000_000_000_i64;
    let created_at: Vec<i64> = (0..rows as i64).map(|i| base_ts + i * 1000).collect();

    // ── 10 Float64 columns with realistic distributions ─────────────

    // price: 2 decimal places, [$0, $10000) — classic ALP target.
    let price: Vec<f64> = (0..rows)
        .map(|_| ((rnd() % 1_000_000) as f64) / 100.0)
        .collect();

    // latitude / longitude: 6 decimal places (typical GPS) — ALP target.
    let latitude: Vec<f64> = (0..rows)
        .map(|_| (25_000_000_i64 + (rnd() % 25_000_000) as i64) as f64 / 1_000_000.0)
        .collect();
    let longitude: Vec<f64> = (0..rows)
        .map(|_| (-125_000_000_i64 + (rnd() % 60_000_000) as i64) as f64 / 1_000_000.0)
        .collect();

    // cpu_usage, memory_usage: bounded [0,1] with 4 decimals — ALP target.
    let cpu_usage: Vec<f64> = (0..rows)
        .map(|_| ((rnd() % 10_000) as f64) / 10_000.0)
        .collect();
    let memory_usage: Vec<f64> = (0..rows)
        .map(|_| ((rnd() % 10_000) as f64) / 10_000.0)
        .collect();

    // temperature: Celsius with 1 decimal, [-40, 60].
    let temperature: Vec<f64> = (0..rows)
        .map(|_| -40.0 + ((rnd() % 1000) as f64) / 10.0)
        .collect();

    // latency_ms: lognormal-ish, heavy tail — bad case for ALP (full entropy).
    let latency_ms: Vec<f64> = (0..rows)
        .map(|_| {
            let r = randf(rnd(), 1.0).max(1e-9);
            (-r.ln()) * 50.0
        })
        .collect();

    // revenue: 2 decimals, long tail up to $100M.
    let revenue: Vec<f64> = (0..rows)
        .map(|_| ((rnd() % 10_000_000_000) as f64) / 100.0)
        .collect();

    // wind_speed: 1 decimal place, 0..200 km/h.
    let wind_speed: Vec<f64> = (0..rows).map(|_| ((rnd() % 2000) as f64) / 10.0).collect();

    // signal_strength: dBm-like, -120..0 with 1 decimal.
    let signal_strength: Vec<f64> = (0..rows)
        .map(|_| -120.0 + ((rnd() % 1200) as f64) / 10.0)
        .collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("price", DataType::Float64, false),
        Field::new("latitude", DataType::Float64, false),
        Field::new("longitude", DataType::Float64, false),
        Field::new("cpu_usage", DataType::Float64, false),
        Field::new("memory_usage", DataType::Float64, false),
        Field::new("temperature", DataType::Float64, false),
        Field::new("latency_ms", DataType::Float64, false),
        Field::new("revenue", DataType::Float64, false),
        Field::new("wind_speed", DataType::Float64, false),
        Field::new("signal_strength", DataType::Float64, false),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(id)),
        Arc::new(Int64Array::from(user_id)),
        Arc::new(TimestampMillisecondArray::from(created_at)),
        Arc::new(Float64Array::from(price)),
        Arc::new(Float64Array::from(latitude)),
        Arc::new(Float64Array::from(longitude)),
        Arc::new(Float64Array::from(cpu_usage)),
        Arc::new(Float64Array::from(memory_usage)),
        Arc::new(Float64Array::from(temperature)),
        Arc::new(Float64Array::from(latency_ms)),
        Arc::new(Float64Array::from(revenue)),
        Arc::new(Float64Array::from(wind_speed)),
        Arc::new(Float64Array::from(signal_strength)),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}

/// String-heavy RecordBatch: 10 string columns with realistic event-log
/// cardinalities, plus 2 identifier columns and one timestamp.  This
/// schema exercises the adaptive string pipeline (FSST, front-coding,
/// dict, cross-column groups) much more than [`build_batch`] and is a
/// better proxy for log / event / clickstream workloads.
pub fn build_string_heavy_batch(rows: usize) -> RecordBatch {
    use arrow_array::{Int64Array, StringArray, TimestampMillisecondArray};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};

    let mut s: u64 = 0xDECAF_BAD_C0DE_1234_u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    // Identifier columns.
    let id: Vec<i64> = (0..rows as i64).collect();
    let user_id: Vec<i64> = (0..rows).map(|_| (rnd() % 5_000_000) as i64).collect();

    // Temporal.
    let base_ts = 1_700_000_000_000_i64;
    let created_at: Vec<i64> = (0..rows as i64).map(|i| base_ts + i * 1000).collect();

    // ── 10 string columns with realistic cardinalities ──────────────

    // Low-card enums — target dict+loom.
    let countries = [
        "US", "CA", "UK", "FR", "DE", "IT", "ES", "JP", "AU", "BR", "IN", "CN",
    ];
    let country: Vec<&str> = (0..rows)
        .map(|_| countries[(rnd() as usize) % countries.len()])
        .collect();

    let devices = ["mobile", "desktop", "tablet", "tv", "console", "watch"];
    let device_type: Vec<&str> = (0..rows)
        .map(|_| devices[(rnd() as usize) % devices.len()])
        .collect();

    let statuses = [
        "active",
        "pending",
        "suspended",
        "deleted",
        "banned",
        "invited",
    ];
    let status: Vec<&str> = (0..rows)
        .map(|_| statuses[(rnd() as usize) % statuses.len()])
        .collect();

    let methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"];
    let http_method: Vec<&str> = (0..rows)
        .map(|_| methods[(rnd() as usize) % methods.len()])
        .collect();

    let severities = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"];
    let log_level: Vec<&str> = (0..rows)
        .map(|_| severities[(rnd() as usize) % severities.len()])
        .collect();

    // Medium cardinality — dict wins big.
    let categories: Vec<String> = (0..500).map(|i| format!("category_{i:04}")).collect();
    let category: Vec<String> = (0..rows)
        .map(|_| categories[(rnd() as usize) % categories.len()].clone())
        .collect();

    // High-card email-like — FSST target (shared suffix tails).
    let domains = [
        "@example.com",
        "@gmail.com",
        "@yahoo.com",
        "@corp.co",
        "@enterprise.net",
        "@mail.org",
    ];
    let email: Vec<String> = (0..rows)
        .map(|i| {
            format!(
                "user_{}_{}{}",
                i,
                (rnd() % 9999),
                domains[(rnd() as usize) % domains.len()]
            )
        })
        .collect();

    // URLs — FSST target (shared host + paths).
    let paths = [
        "/api/v1/users",
        "/api/v2/orders",
        "/healthz",
        "/metrics",
        "/static/img",
        "/api/v1/products",
        "/api/v2/cart",
    ];
    let request_path: Vec<String> = (0..rows)
        .map(|i| {
            format!(
                "https://api.example.com{}/{}",
                paths[(rnd() as usize) % paths.len()],
                i
            )
        })
        .collect();

    // User agents — shared Mozilla / Chrome prefixes (FSST + dict hybrid).
    let ua_prefixes = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0) AppleWebKit/605.1.15",
    ];
    let user_agent: Vec<String> = (0..rows)
        .map(|_| {
            format!(
                "{} Chrome/{}.0.0.0",
                ua_prefixes[(rnd() as usize) % ua_prefixes.len()],
                100 + (rnd() % 20)
            )
        })
        .collect();

    // Free-text log message — mostly unique suffixes, some shared prefixes.
    let log_prefixes = [
        "Request completed with status",
        "Cache hit for key",
        "Processing background job",
        "Auth check passed for user",
        "Retry scheduled due to",
    ];
    let message: Vec<String> = (0..rows)
        .map(|i| {
            format!(
                "{} {} in {}ms",
                log_prefixes[(rnd() as usize) % log_prefixes.len()],
                i,
                rnd() % 500
            )
        })
        .collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("country", DataType::Utf8, false),
        Field::new("device_type", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("http_method", DataType::Utf8, false),
        Field::new("log_level", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("email", DataType::Utf8, false),
        Field::new("request_path", DataType::Utf8, false),
        Field::new("user_agent", DataType::Utf8, false),
        Field::new("message", DataType::Utf8, false),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(id)),
        Arc::new(Int64Array::from(user_id)),
        Arc::new(TimestampMillisecondArray::from(created_at)),
        Arc::new(StringArray::from(country)),
        Arc::new(StringArray::from(device_type)),
        Arc::new(StringArray::from(status)),
        Arc::new(StringArray::from(http_method)),
        Arc::new(StringArray::from(log_level)),
        Arc::new(StringArray::from(
            category.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            email.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            request_path.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            user_agent.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            message.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}

fn build_batch(rows: usize) -> RecordBatch {
    // Pseudo-random helpers.
    let mut s: u64 = 0xCAFE_F00D_BEEF_1234;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    // Numeric columns — realistic distributions.
    let id: Vec<i64> = (0..rows as i64).collect();
    let user_id: Vec<i64> = (0..rows).map(|_| (rnd() % 5_000_000) as i64).collect();
    let session_count: Vec<i64> = (0..rows).map(|_| (rnd() % 100) as i64).collect();
    let error_code: Vec<i64> = (0..rows)
        .map(|_| {
            let v = rnd() % 100;
            if v < 95 { 0 } else { (200 + (v as i64) * 2) }
        })
        .collect();

    // Float64 — ALP targets.
    let revenue: Vec<f64> = (0..rows)
        .map(|_| {
            ((rnd() % 1_000_000) as f64) / 100.0 // 2 decimal places
        })
        .collect();
    let latitude: Vec<f64> = (0..rows)
        .map(|_| (25_000_000_i64 + (rnd() % 25_000_000) as i64) as f64 / 1_000_000.0)
        .collect();
    let longitude: Vec<f64> = (0..rows)
        .map(|_| (-125_000_000_i64 + (rnd() % 60_000_000) as i64) as f64 / 1_000_000.0)
        .collect();
    let cpu_usage: Vec<f64> = (0..rows)
        .map(|_| {
            ((rnd() % 10_000) as f64) / 10_000.0 // 4 decimal places in [0,1]
        })
        .collect();

    // Temporal.
    let base_ts = 1_700_000_000_000_i64; // 2023-11-14 in ms
    let created_at: Vec<i64> = (0..rows as i64).map(|i| base_ts + i * 1000).collect();
    let updated_at: Vec<i64> = (0..rows as i64)
        .map(|i| base_ts + i * 1000 + 60_000)
        .collect();

    // Booleans.
    let is_active: Vec<bool> = (0..rows).map(|_| rnd() % 5 > 0).collect();
    let is_verified: Vec<bool> = (0..rows).map(|_| rnd() % 3 == 0).collect();
    let has_subscription: Vec<bool> = (0..rows).map(|_| rnd() % 10 > 7).collect();

    // Date32 (days since epoch).
    let birth_date: Vec<i32> = (0..rows)
        .map(|_| (10_000 + (rnd() % 15_000)) as i32)
        .collect();

    // Strings with realistic cardinality.
    let countries = ["US", "CA", "UK", "FR", "DE", "IT", "ES", "JP", "AU", "BR"];
    let country: Vec<&str> = (0..rows)
        .map(|_| countries[(rnd() as usize) % countries.len()])
        .collect();

    let devices = ["mobile", "desktop", "tablet", "tv", "console"];
    let device_type: Vec<&str> = (0..rows)
        .map(|_| devices[(rnd() as usize) % devices.len()])
        .collect();

    let statuses = ["active", "pending", "suspended", "deleted", "banned"];
    let status: Vec<&str> = (0..rows)
        .map(|_| statuses[(rnd() as usize) % statuses.len()])
        .collect();

    let categories: Vec<String> = (0..200).map(|i| format!("category_{:04}", i)).collect();
    let category: Vec<String> = (0..rows)
        .map(|_| categories[(rnd() as usize) % categories.len()].clone())
        .collect();

    // Emails — high cardinality, share "@domain.com" suffixes.
    let domains = [
        "@example.com",
        "@gmail.com",
        "@yahoo.com",
        "@corp.co",
        "@enterprise.net",
    ];
    let email: Vec<String> = (0..rows)
        .map(|i| {
            format!(
                "user_{}_{}{}",
                i,
                (rnd() % 9999),
                domains[(rnd() as usize) % domains.len()]
            )
        })
        .collect();

    // Request paths — share host + common prefixes.
    let paths = [
        "/api/v1/users",
        "/api/v2/orders",
        "/healthz",
        "/metrics",
        "/static/img",
    ];
    let request_path: Vec<String> = (0..rows)
        .map(|i| {
            format!(
                "https://api.example.com{}/{}",
                paths[(rnd() as usize) % paths.len()],
                i
            )
        })
        .collect();

    // User agents — share "Mozilla/5.0" etc.
    let ua_prefixes = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0) AppleWebKit/605.1.15",
    ];
    let user_agent: Vec<String> = (0..rows)
        .map(|_| {
            format!(
                "{} Chrome/{}.0.0.0",
                ua_prefixes[(rnd() as usize) % ua_prefixes.len()],
                100 + (rnd() % 20)
            )
        })
        .collect();

    // Tags — short CSV-ish.
    let tag_vocab = [
        "sports", "tech", "news", "finance", "travel", "food", "music", "gaming",
    ];
    let tags: Vec<String> = (0..rows)
        .map(|_| {
            let n = 1 + (rnd() % 3) as usize;
            (0..n)
                .map(|_| tag_vocab[(rnd() as usize) % tag_vocab.len()])
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();

    // Build the RecordBatch.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),
        Field::new("session_count", DataType::Int64, false),
        Field::new("error_code", DataType::Int64, false),
        Field::new("revenue", DataType::Float64, false),
        Field::new("latitude", DataType::Float64, false),
        Field::new("longitude", DataType::Float64, false),
        Field::new("cpu_usage", DataType::Float64, false),
        Field::new(
            "created_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new(
            "updated_at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("country", DataType::Utf8, false),
        Field::new("device_type", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("email", DataType::Utf8, false),
        Field::new("request_path", DataType::Utf8, false),
        Field::new("user_agent", DataType::Utf8, false),
        Field::new("is_active", DataType::Boolean, false),
        Field::new("is_verified", DataType::Boolean, false),
        Field::new("has_subscription", DataType::Boolean, false),
        Field::new("birth_date", DataType::Date32, false),
        Field::new("tags", DataType::Utf8, false),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(id)),
        Arc::new(Int64Array::from(user_id)),
        Arc::new(Int64Array::from(session_count)),
        Arc::new(Int64Array::from(error_code)),
        Arc::new(Float64Array::from(revenue)),
        Arc::new(Float64Array::from(latitude)),
        Arc::new(Float64Array::from(longitude)),
        Arc::new(Float64Array::from(cpu_usage)),
        Arc::new(TimestampMillisecondArray::from(created_at)),
        Arc::new(TimestampMillisecondArray::from(updated_at)),
        Arc::new(StringArray::from(country)),
        Arc::new(StringArray::from(device_type)),
        Arc::new(StringArray::from(status)),
        Arc::new(StringArray::from(
            category.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            email.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            request_path.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            user_agent.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(is_active)),
        Arc::new(BooleanArray::from(is_verified)),
        Arc::new(BooleanArray::from(has_subscription)),
        Arc::new(Date32Array::from(birth_date)),
        Arc::new(StringArray::from(
            tags.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}

/// Estimate in-memory Arrow byte size (sum of column value buffers).
fn estimate_arrow_bytes(batch: &RecordBatch) -> usize {
    let mut total = 0usize;
    for col in batch.columns() {
        total += arrow_column_bytes(col.as_ref());
    }
    total
}

fn arrow_column_bytes(array: &dyn Array) -> usize {
    use arrow_schema::DataType as DT;
    let rows = array.len();
    match array.data_type() {
        DT::Int64 | DT::UInt64 | DT::Float64 | DT::Timestamp(_, _) | DT::Date64 => rows * 8,
        DT::Int32 | DT::UInt32 | DT::Float32 | DT::Date32 => rows * 4,
        DT::Int16 | DT::UInt16 => rows * 2,
        DT::Int8 | DT::UInt8 => rows,
        DT::Boolean => (rows + 7) / 8,
        DT::Utf8 => {
            if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
                a.value_data().len() + (rows + 1) * 4
            } else {
                0
            }
        }
        _ => rows * 8,
    }
}

fn summarize_schema(schema: &Schema) -> String {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in schema.fields() {
        let k = match f.data_type() {
            DataType::Int64 | DataType::UInt64 => "int64".to_string(),
            DataType::Float64 => "float64".to_string(),
            DataType::Timestamp(_, _) => "timestamp".to_string(),
            DataType::Date32 | DataType::Date64 => "date".to_string(),
            DataType::Boolean => "bool".to_string(),
            DataType::Utf8 => "string".to_string(),
            other => format!("{other:?}"),
        };
        *counts.entry(k).or_insert(0) += 1;
    }
    let mut parts: Vec<String> = counts
        .into_iter()
        .map(|(k, v)| format!("{v}×{k}"))
        .collect();
    parts.sort();
    parts.join(", ")
}

fn human(n: usize) -> String {
    const K: f64 = 1024.0;
    let f = n as f64;
    if f < K {
        format!("{} B", n)
    } else if f < K * K {
        format!("{:.1} KB", f / K)
    } else if f < K * K * K {
        format!("{:.1} MB", f / (K * K))
    } else {
        format!("{:.2} GB", f / (K * K * K))
    }
}

fn fmt_rows(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

fn mbs(bytes: usize, ms: f64) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    (bytes as f64) / (1024.0 * 1024.0) / (ms / 1000.0)
}
