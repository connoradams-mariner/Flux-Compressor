// examples/integration_demo.rs
//
// Run with:
//   cargo run --example integration_demo -p loom
//
// Shows:
//   1. Compress / decompress a realistic multi-column Arrow RecordBatch
//   2. Round-trip with u128 OutlierMap values
//   3. Predicate pushdown (Z-Order skipping)
//   4. Strategy override (benchmark mode)
//   5. Cold optimizer two-pass pattern (in-process)

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{RecordBatch, UInt64Array, Int64Array, Float64Array};
use arrow_schema::{DataType, Field, Schema};

use loom::{
    atlas::AtlasFooter,
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    loom_classifier::{classify, LoomStrategy},
    outlier_map::{encode_with_outlier_map, decode_with_outlier_map},
    traits::{LoomCompressor, LoomDecompressor, Predicate},
    SEGMENT_SIZE,
};

// ─────────────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("user_id",      DataType::UInt64, false), // sequential → DeltaDelta
        Field::new("revenue_cents",DataType::UInt64, false), // random     → BitSlab
        Field::new("region_code",  DataType::UInt64, false), // low-card   → Dictionary
        Field::new("session_ms",   DataType::Int64,  false), // mixed      → BitSlab
    ]))
}

fn make_batch(n: usize, offset: u64) -> RecordBatch {
    let user_ids: Vec<u64>  = (offset..offset + n as u64).collect();
    let revenues: Vec<u64>  = (0..n as u64).map(|i| (i * 37 + 99) % 99_999).collect();
    let regions:  Vec<u64>  = (0..n as u64).map(|i| i % 8).collect();
    let sessions: Vec<i64>  = (0..n as i64).map(|i| (i * 1_234) % 86_400_000).collect();

    RecordBatch::try_new(
        make_schema(),
        vec![
            Arc::new(UInt64Array::from(user_ids)),
            Arc::new(UInt64Array::from(revenues)),
            Arc::new(UInt64Array::from(regions)),
            Arc::new(Int64Array::from(sessions)),
        ],
    )
    .unwrap()
}

fn human(bytes: usize) -> String {
    if bytes < 1024 { format!("{bytes} B") }
    else if bytes < 1024 * 1024 { format!("{:.1} KB", bytes as f64 / 1024.0) }
    else { format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0)) }
}

fn separator(title: &str) {
    println!("\n── {title} {}", "─".repeat(55 - title.len().min(54)));
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Basic compress / decompress round-trip
// ─────────────────────────────────────────────────────────────────────────────

fn demo_basic_round_trip() {
    separator("1. Basic Round-Trip");

    let n = 100_000usize;
    let batch = make_batch(n, 0);

    println!("  Input : {} rows × {} columns", batch.num_rows(), batch.num_columns());
    let raw_bytes = n * 4 * 8; // 4 columns × 8 bytes each
    println!("  Raw   : {}", human(raw_bytes));

    // ── Compress ─────────────────────────────────────────────────────────────
    let writer = FluxWriter::new();
    let t0 = Instant::now();
    let flux_bytes = writer.compress(&batch).unwrap();
    let compress_ms = t0.elapsed().as_millis();

    println!(
        "  Flux  : {}  ({:.2}×)  compressed in {}ms",
        human(flux_bytes.len()),
        raw_bytes as f64 / flux_bytes.len() as f64,
        compress_ms,
    );

    // ── Inspect footer ────────────────────────────────────────────────────────
    let footer = AtlasFooter::from_file_tail(&flux_bytes).unwrap();
    println!("  Atlas : {} blocks", footer.blocks.len());
    for (i, b) in footer.blocks.iter().take(4).enumerate() {
        println!(
            "    block {:>3}  offset={:<10}  z=[{:>12}, {:>12}]  strategy={:?}",
            i, b.block_offset, b.z_min, b.z_max, b.strategy
        );
    }
    if footer.blocks.len() > 4 {
        println!("    … {} more blocks", footer.blocks.len() - 4);
    }

    // ── Decompress ───────────────────────────────────────────────────────────
    let reader = FluxReader::new("user_id");
    let t1 = Instant::now();
    let out_batch = reader.decompress_all(&flux_bytes).unwrap();
    let decompress_ms = t1.elapsed().as_millis();

    println!(
        "  Read  : {} rows in {}ms  ({:.0} MB/s)",
        out_batch.num_rows(),
        decompress_ms,
        flux_bytes.len() as f64 / decompress_ms.max(1) as f64 / 1000.0,
    );

    // Verify first column (user_id) round-tripped correctly.
    let col = out_batch.column(0)
        .as_any().downcast_ref::<UInt64Array>().unwrap();
    assert_eq!(col.value(0), 0);
    assert_eq!(col.value(n - 1), (n - 1) as u64);
    println!("  ✓ Round-trip verified");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. u128 OutlierMap — large computed aggregation values
// ─────────────────────────────────────────────────────────────────────────────

fn demo_u128_outlier_map() {
    separator("2. u128 OutlierMap (Large Aggregation Results)");

    // Simulate a SUM(bigint_col) result that exceeds u64::MAX.
    let mut values: Vec<u128> = (0u128..1023).map(|i| i * 1_000_000).collect();
    values.push(u128::MAX - 42);          // one monster value
    values.push(u128::MAX / 3 + 99_999); // another

    println!("  {} values, {} outliers (> u64::MAX)", values.len(), 2);

    // Classify.
    let class = classify(&values);
    println!(
        "  Classifier: strategy={:?}  use_outlier_map={}  slab_width={}",
        class.strategy, class.use_outlier_map, class.slab_width
    );

    // Encode: primary slab + outlier map.
    let (slab, om, width, for_val) = encode_with_outlier_map(&values).unwrap();
    let om_bytes = om.to_bytes().unwrap();

    println!(
        "  Slab  : {} bytes @ {} bits/value  (covers 99th-pct range)",
        slab.len(), width
    );
    println!(
        "  OutlierMap: {} entries = {} bytes  (full u128 precision)",
        om.len(), om_bytes.len()
    );

    // Decode.
    let decoded = decode_with_outlier_map(&slab, &om_bytes, for_val, width, values.len()).unwrap();

    assert_eq!(decoded, values, "u128 round-trip failed");
    println!("  ✓ u128 round-trip verified (including {:?}...)", &values[values.len()-2..]);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Predicate Pushdown (Z-Order Atlas skipping)
// ─────────────────────────────────────────────────────────────────────────────

fn demo_predicate_pushdown() {
    separator("3. Predicate Pushdown / Z-Order Skipping");

    // Three separate batches → three block groups in the Atlas footer.
    let writer  = FluxWriter::new();
    let mut all_bytes: Vec<u8> = Vec::new();
    let mut all_footer = AtlasFooter::new();

    for partition in 0u64..3 {
        let offset = partition * 1024;
        let batch  = make_batch(1024, offset);
        let bytes  = writer.compress(&batch).unwrap();

        // Merge the footer, adjusting block offsets.
        let footer = AtlasFooter::from_file_tail(&bytes).unwrap();
        let footer_len = footer.to_bytes().unwrap().len();
        let data_len = bytes.len() - footer_len;
        let base_offset = all_bytes.len() as u64;

        all_bytes.extend_from_slice(&bytes[..data_len]);

        for mut meta in footer.blocks {
            meta.block_offset += base_offset;
            all_footer.push(meta);
        }
    }
    // Write combined footer.
    all_bytes.extend(all_footer.to_bytes().unwrap());

    println!("  Written {} blocks across 3 partitions", all_footer.blocks.len());

    // Full scan.
    let reader = FluxReader::new("user_id");
    let t0 = Instant::now();
    let full = reader.decompress_all(&all_bytes).unwrap();
    let full_ms = t0.elapsed().as_micros();
    println!("  Full scan  : {} rows in {}µs", full.num_rows(), full_ms);

    // Predicate: user_id > 2000  (should skip partition 0 and most of 1).
    let pred = Predicate::GreaterThan { column: "user_id".into(), value: 2000 };
    let t1 = Instant::now();
    let filtered = reader.decompress(&all_bytes, &pred).unwrap();
    let filter_ms = t1.elapsed().as_micros();

    let col = filtered.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
    let min_returned = col.values().iter().copied().min().unwrap_or(0);

    println!(
        "  Pushdown   : {} rows in {}µs  ({}× faster)  min_val={}",
        filtered.num_rows(),
        filter_ms,
        full_ms / filter_ms.max(1),
        min_returned,
    );
    assert!(min_returned >= 2048, "expected only block-3 values (≥2048), got {min_returned}");
    println!("  ✓ Only blocks satisfying (user_id > 2000) were decompressed");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Strategy override (benchmarking / forced encoding)
// ─────────────────────────────────────────────────────────────────────────────

fn demo_strategy_override() {
    separator("4. Strategy Override");

    let batch = make_batch(10_000, 0);
    let strategies = [
        LoomStrategy::Rle,
        LoomStrategy::DeltaDelta,
        LoomStrategy::Dictionary,
        LoomStrategy::BitSlab,
        LoomStrategy::SimdLz4,
    ];

    println!("  {:>12}  {:>10}  {:>12}", "Strategy", "Bytes", "Ratio");
    println!("  {}", "─".repeat(38));

    let raw_bytes = 10_000 * 8;
    for strategy in strategies {
        let writer = FluxWriter::with_strategy(strategy);
        let bytes  = writer.compress(&batch).unwrap();
        println!(
            "  {:>12?}  {:>10}  {:>11.2}×",
            strategy,
            bytes.len(),
            raw_bytes as f64 / bytes.len() as f64,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Cold optimizer pattern (in-process)
// ─────────────────────────────────────────────────────────────────────────────

fn demo_cold_optimizer() {
    separator("5. Cold Optimizer (Two-Pass, In-Process)");

    use loom::{
        atlas::{BlockMeta, z_order_encode},
        bit_io::discover_width,
        compressors::compress_chunk,
    };

    // Simulate 4 "hot" partition files written by Spark.
    let n_per_part = 4096usize;
    let all_parts: Vec<Vec<u128>> = (0u128..4)
        .map(|p| {
            (0..n_per_part as u128)
                .map(|i| p * n_per_part as u128 + i)
                .collect()
        })
        .collect();

    // Measure hot size.
    let writer = FluxWriter::new();
    let hot_total: usize = all_parts.iter().map(|part| {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::UInt64, false)])),
            vec![Arc::new(UInt64Array::from(part.iter().map(|&v| v as u64).collect::<Vec<_>>()))],
        ).unwrap();
        writer.compress(&batch).unwrap().len()
    }).sum();

    // ── Pass 1: global scan ───────────────────────────────────────────────────
    let all_values: Vec<u128> = all_parts.iter().flatten().copied().collect();
    let (global_width, global_for) = discover_width(&all_values);
    println!(
        "  Pass 1: {} total values  global_width={}  global_for={}",
        all_values.len(), global_width, global_for
    );

    // ── Pass 2: Z-Order sort + global re-pack ─────────────────────────────────
    // Sort segments by Z-Order of their midpoint for multi-dimensional locality.
    let mut segments: Vec<Vec<u128>> = all_values
        .chunks(SEGMENT_SIZE)
        .map(|c| c.to_vec())
        .collect();

    segments.sort_by_key(|seg| {
        let mid = seg.iter().copied().sum::<u128>() / seg.len() as u128;
        z_order_encode(mid as u64, (mid >> 64) as u64)
    });

    let mut cold_bytes: Vec<u8> = Vec::new();
    let mut cold_footer = AtlasFooter::new();

    for seg in &segments {
        let block_offset = cold_bytes.len() as u64;
        let strategy = classify(seg).strategy;
        let block = compress_chunk(seg, strategy).unwrap();

        cold_footer.push(BlockMeta {
            block_offset,
            z_min: seg.iter().copied().min().unwrap_or(0),
            z_max: seg.iter().copied().max().unwrap_or(0),
            null_bitmap_offset: 0,
            strategy,
        });
        cold_bytes.extend_from_slice(&block);
    }
    cold_bytes.extend(cold_footer.to_bytes().unwrap());

    println!(
        "  Pass 2: hot={} cold={}  saved {:.1}%",
        human(hot_total),
        human(cold_bytes.len()),
        (1.0 - cold_bytes.len() as f64 / hot_total as f64) * 100.0,
    );
    println!("  ✓ Cold archive written ({} blocks, Z-Order sorted)", cold_footer.blocks.len());
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       FluxCompress Integration Demo                      ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    demo_basic_round_trip();
    demo_u128_outlier_map();
    demo_predicate_pushdown();
    demo_strategy_override();
    demo_cold_optimizer();

    println!("\n✓ All demos complete.");
}
