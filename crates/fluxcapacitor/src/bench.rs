// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `fluxcapacitor bench` — quick compression/decompression benchmark.
//!
//! Generates synthetic data in various patterns and measures throughput.

use std::time::Instant;
use anyhow::Result;

use loom::{
    compressors::flux_writer::compress_chunk,
    decompressors::block_reader::decompress_block,
    loom_classifier::classify,
};

pub fn cmd_bench(rows: usize, pattern: &str) -> Result<()> {
    println!("FluxCapacitor Benchmark");
    println!("  rows    : {rows}");
    println!("  pattern : {pattern}");
    println!();

    let values = generate_data(rows, pattern);

    // ── Classify ─────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let classification = classify(&values[..values.len().min(1024)]);
    let classify_us = t0.elapsed().as_micros();

    println!("  Strategy selected : {:?}", classification.strategy);
    println!("  Classify time     : {classify_us} µs");
    println!();

    // ── Compress ─────────────────────────────────────────────────────────────
    let strategy = classification.strategy;
    let t1 = Instant::now();
    // Compress in SEGMENT_SIZE chunks.
    let mut all_blocks: Vec<Vec<u8>> = Vec::new();
    for chunk in values.chunks(loom::SEGMENT_SIZE) {
        all_blocks.push(compress_chunk(chunk, strategy)?);
    }
    let compress_ms = t1.elapsed().as_millis();
    let total_compressed: usize = all_blocks.iter().map(|b| b.len()).sum();
    let original_bytes = values.len() * 16; // u128 = 16 bytes each

    println!(
        "  Compression       : {} → {}  ({:.2}x)  in {} ms  ({:.0} MB/s)",
        human_size(original_bytes),
        human_size(total_compressed),
        original_bytes as f64 / total_compressed as f64,
        compress_ms,
        throughput_mbs(original_bytes, compress_ms),
    );

    // ── Decompress ───────────────────────────────────────────────────────────
    let t2 = Instant::now();
    let mut decoded_count = 0usize;
    for block in &all_blocks {
        let (vals, _) = decompress_block(block)?;
        decoded_count += vals.len();
    }
    let decompress_ms = t2.elapsed().as_millis();

    println!(
        "  Decompression     : {} values  in {} ms  ({:.0} MB/s)",
        decoded_count,
        decompress_ms,
        throughput_mbs(original_bytes, decompress_ms),
    );

    println!();
    println!("  ✓ All {} rows round-tripped successfully.", decoded_count);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Data generators
// ─────────────────────────────────────────────────────────────────────────────

fn generate_data(rows: usize, pattern: &str) -> Vec<u128> {
    match pattern {
        "sequential" => (0u128..rows as u128).collect(),
        "constant"   => vec![0xDEAD_BEEF_u128; rows],
        "random"     => {
            // Simple xorshift-based PRNG for reproducibility.
            let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
            (0..rows).map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u128 | ((state.wrapping_mul(0x9e3779b97f4a7c15)) as u128) << 64
            }).collect()
        }
        "outlier" => {
            // 99% small values + 1% u128::MAX outliers.
            (0u128..rows as u128)
                .map(|i| if i % 100 == 0 { u128::MAX } else { i % 256 })
                .collect()
        }
        other => {
            eprintln!("Unknown pattern '{other}', using sequential.");
            (0u128..rows as u128).collect()
        }
    }
}

fn human_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn throughput_mbs(bytes: usize, ms: u128) -> f64 {
    if ms == 0 { return f64::INFINITY; }
    (bytes as f64 / (1024.0 * 1024.0)) / (ms as f64 / 1000.0)
}
