// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `fluxcapacitor string-bench` — high-cardinality string compression harness.
//!
//! Generates 10M-row string columns across representative patterns
//! (URLs, UUIDs, log lines, sorted paths, mixed categorical) and reports
//! compressed size / ratio / throughput for every [`CompressionProfile`].
//!
//! Designed to be re-run after each string-compressor optimisation so we
//! can see exactly which change moved which pattern.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use loom::{
    CompressionProfile,
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    traits::{LoomCompressor, LoomDecompressor, Predicate},
};

pub fn cmd_string_bench(rows: usize) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  String Compression Bench ({} rows)                       ", fmt_rows(rows));
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let patterns: &[(&str, fn(usize) -> Vec<String>)] = &[
        ("urls_high_card",    gen_urls_high_card),
        ("uuids",             gen_uuids),
        ("log_lines",         gen_log_lines),
        ("sorted_paths",      gen_sorted_paths),
        ("mixed_categorical", gen_mixed_categorical),
        ("short_skus",        gen_short_skus),
    ];

    for (name, make) in patterns {
        println!("\n── Pattern: {} ──", name);
        let strings = make(rows);
        let raw_bytes: usize = strings.iter().map(|s| s.len()).sum();
        let batch = build_batch(&strings);
        println!("  raw data: {}  (avg {:.1} B/row)",
            human(raw_bytes),
            raw_bytes as f64 / rows as f64);
        println!(
            "  {:<18} {:>10} {:>8} {:>12} {:>12}",
            "Profile", "Size", "Ratio", "Comp MB/s", "Dec MB/s"
        );
        println!("  {}", "─".repeat(66));
        for (pname, profile) in [
            ("Speed",    CompressionProfile::Speed),
            ("Balanced", CompressionProfile::Balanced),
            ("Archive",  CompressionProfile::Archive),
        ] {
            run_one(&batch, raw_bytes, pname, profile)?;
        }
    }

    println!("\n  ✓ string-bench complete.");
    Ok(())
}

fn run_one(
    batch: &RecordBatch,
    raw_bytes: usize,
    profile_name: &str,
    profile: CompressionProfile,
) -> Result<()> {
    let writer = FluxWriter::with_profile(profile);
    let t0 = Instant::now();
    let bytes = writer.compress(batch)?;
    let c_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let reader = FluxReader::new("value");
    let t1 = Instant::now();
    let _ = reader.decompress(&bytes, &Predicate::None)?;
    let d_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let ratio = raw_bytes as f64 / bytes.len() as f64;
    let c_mbs = mbs(raw_bytes, c_ms);
    let d_mbs = mbs(raw_bytes, d_ms);
    println!(
        "  {:<18} {:>10} {:>7.1}x {:>11.0} {:>11.0}",
        profile_name,
        human(bytes.len()),
        ratio,
        c_mbs,
        d_mbs,
    );
    Ok(())
}

fn build_batch(values: &[String]) -> RecordBatch {
    let arr = StringArray::from_iter(values.iter().map(|s| Some(s.as_str())));
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
}

// ── Patterns ──────────────────────────────────────────────────────────────

fn gen_urls_high_card(rows: usize) -> Vec<String> {
    let hosts = [
        "api.example.com", "www.shop.io", "cdn.media.net", "static.assets.com",
        "app.enterprise.co", "data.warehouse.internal",
    ];
    let paths = ["/v1/users", "/v2/orders", "/api/search", "/static/img", "/healthz"];
    let mut s: u64 = 0xDEAD_BEEF_CAFE_BABE;
    (0..rows)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let h = hosts[(s as usize) % hosts.len()];
            let p = paths[(s as usize / 7) % paths.len()];
            format!("https://{h}{p}?id={}&session={:016x}&ts={}", i, s, 1700000000u64 + (i as u64))
        })
        .collect()
}

fn gen_uuids(rows: usize) -> Vec<String> {
    let mut s: u64 = 0xFEED_FACE_0BADF00D;
    (0..rows)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let a = s;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let b = s;
            format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                (a >> 32) as u32,
                ((a >> 16) & 0xFFFF) as u16,
                (a & 0xFFFF) as u16,
                ((b >> 48) & 0xFFFF) as u16,
                b & 0xFFFF_FFFF_FFFF,
            )
        })
        .collect()
}

fn gen_log_lines(rows: usize) -> Vec<String> {
    let levels = ["INFO", "WARN", "ERROR", "DEBUG"];
    let modules = ["auth", "db.pool", "http.server", "scheduler", "cache.redis"];
    let templates = [
        "request handled in {ms}ms",
        "connection pool size={n} free={f}",
        "user {u} action={a}",
        "cache hit ratio={r}",
        "shutting down worker={w}",
    ];
    let mut s: u64 = 0xCAFE_F00D_BEEF_0001;
    (0..rows)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let lvl = levels[(s as usize) % levels.len()];
            let md  = modules[(s as usize / 3) % modules.len()];
            let tpl = templates[(s as usize / 11) % templates.len()];
            format!(
                "2024-03-{:02}T{:02}:{:02}:{:02}Z {} {}: {} [trace={:016x}]",
                1 + (i as u64 / 86400) % 28,
                (i as u64 / 3600) % 24,
                (i as u64 / 60) % 60,
                (i as u64) % 60,
                lvl, md, tpl, s,
            )
        })
        .collect()
}

fn gen_sorted_paths(rows: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        out.push(format!(
            "/var/log/app/{:04}/{:02}/{:02}/worker-{:05}.log",
            2020 + (i / 2_000_000) as u32,
            1 + ((i / 200_000) % 12) as u32,
            1 + ((i / 10_000) % 28) as u32,
            i,
        ));
    }
    out
}

fn gen_mixed_categorical(rows: usize) -> Vec<String> {
    // 200 distinct values, so ~0.002% cardinality — should hit dict path
    // but still exercise the high-card fallback on other patterns.
    let cats: Vec<String> = (0..200).map(|i| format!("category_code_{:04}", i)).collect();
    let mut s: u64 = 0x0102_0304_0506_0708;
    (0..rows)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            cats[(s as usize) % cats.len()].clone()
        })
        .collect()
}

fn gen_short_skus(rows: usize) -> Vec<String> {
    // Very high cardinality, short strings, strong common-prefix structure.
    (0..rows).map(|i| format!("SKU-{:010}", i)).collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn human(n: usize) -> String {
    const K: f64 = 1024.0;
    let f = n as f64;
    if f < K              { format!("{} B", n) }
    else if f < K * K     { format!("{:.1} KB", f / K) }
    else if f < K * K * K { format!("{:.1} MB", f / (K * K)) }
    else                  { format!("{:.1} GB", f / (K * K * K)) }
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
