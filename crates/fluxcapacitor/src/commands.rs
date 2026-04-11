// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Subcommand argument structs and their `run_*` functions.

use std::path::PathBuf;
use std::time::Instant;
use clap::Args;
use anyhow::{Context, Result};
use tracing::{info, warn};

use loom::{
    atlas::AtlasFooter,
    compressors::FluxWriter,
    decompressors::FluxReader,
    loom_classifier::{classify, LoomStrategy},
    traits::{LoomCompressor, LoomDecompressor, Predicate},
    SEGMENT_SIZE,
};

use crate::optimizer::ColdOptimizer;

// ─────────────────────────────────────────────────────────────────────────────
// compress
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct CompressArgs {
    /// Input file (Parquet or CSV).
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output `.flux` file.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Force a specific compression strategy instead of auto-selecting.
    #[arg(long, value_enum)]
    pub strategy: Option<StrategyArg>,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum StrategyArg {
    Rle,
    DeltaDelta,
    Dictionary,
    BitSlab,
    Lz4,
}

impl From<StrategyArg> for LoomStrategy {
    fn from(s: StrategyArg) -> Self {
        match s {
            StrategyArg::Rle        => LoomStrategy::Rle,
            StrategyArg::DeltaDelta => LoomStrategy::DeltaDelta,
            StrategyArg::Dictionary => LoomStrategy::Dictionary,
            StrategyArg::BitSlab    => LoomStrategy::BitSlab,
            StrategyArg::Lz4        => LoomStrategy::SimdLz4,
        }
    }
}

pub fn run_compress(args: CompressArgs) -> Result<()> {
    info!("Compressing {:?} → {:?}", args.input, args.output);
    let t0 = Instant::now();

    let batch = load_record_batch(&args.input)?;
    let writer = match args.strategy {
        Some(s) => FluxWriter::with_strategy(s.into()),
        None    => FluxWriter::new(),
    };

    let compressed = writer.compress(&batch)?;
    let output_size = compressed.len();
    std::fs::write(&args.output, &compressed)
        .with_context(|| format!("writing {:?}", args.output))?;

    let elapsed = t0.elapsed();
    println!(
        "✓ Compressed {} rows → {output_size} bytes in {elapsed:.2?}",
        batch.num_rows()
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// decompress
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct DecompressArgs {
    /// Input `.flux` file.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output file (Parquet or CSV).
    #[arg(short, long)]
    pub output: PathBuf,

    /// Optional predicate: only return rows where `column > value`.
    #[arg(long)]
    pub gt: Option<i128>,

    /// Column name for the predicate.
    #[arg(long, default_value = "value")]
    pub column: String,
}

pub fn run_decompress(args: DecompressArgs) -> Result<()> {
    info!("Decompressing {:?} → {:?}", args.input, args.output);
    let t0 = Instant::now();

    let data = std::fs::read(&args.input)
        .with_context(|| format!("reading {:?}", args.input))?;

    let pred = match args.gt {
        Some(threshold) => Predicate::GreaterThan {
            column: args.column.clone(),
            value: threshold,
        },
        None => Predicate::None,
    };

    let reader = FluxReader::new(&args.column);
    let batch = reader.decompress(&data, &pred)?;
    let rows = batch.num_rows();

    save_record_batch(&batch, &args.output)?;

    let elapsed = t0.elapsed();
    println!("✓ Decompressed {rows} rows in {elapsed:.2?}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// optimize (two-pass cold optimizer)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct OptimizeArgs {
    /// Directory containing `.flux` partition files.
    #[arg(short, long)]
    pub input_dir: PathBuf,

    /// Output path for the optimised `.flux` archive.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Apply Z-Order interleaving on the named column.
    #[arg(long)]
    pub z_order_column: Option<String>,
}

pub fn run_optimize(args: OptimizeArgs) -> Result<()> {
    info!("Optimizing {:?} → {:?}", args.input_dir, args.output);
    let t0 = Instant::now();

    let mut optimizer = ColdOptimizer::new();

    // Pass 1: scan all partitions.
    optimizer.scan_partitions(&args.input_dir)
        .with_context(|| "Pass 1 (partition scan) failed")?;

    info!(
        global_dict_size = optimizer.global_dict_size(),
        "Pass 1 complete"
    );

    // Pass 2: repack with global dict + Z-Order.
    optimizer.optimize(&args.output)
        .with_context(|| "Pass 2 (repack) failed")?;

    let elapsed = t0.elapsed();
    let output_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    println!(
        "✓ Optimized in {elapsed:.2?} — output {output_size} bytes"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// inspect
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// The `.flux` file to inspect.
    pub file: PathBuf,
}

pub fn run_inspect(args: InspectArgs) -> Result<()> {
    let data = std::fs::read(&args.file)
        .with_context(|| format!("reading {:?}", args.file))?;

    let footer = AtlasFooter::from_file_tail(&data)
        .with_context(|| "parsing Atlas footer")?;

    println!("FluxCompress Atlas — {:?}", args.file);
    println!("─────────────────────────────────────────────────────────────");
    println!(
        "{:>5}  {:>12}  {:>20}  {:>20}  {:>12}",
        "Block", "Offset", "Z-Min", "Z-Max", "Strategy"
    );
    println!("─────────────────────────────────────────────────────────────");

    for (i, meta) in footer.blocks.iter().enumerate() {
        println!(
            "{:>5}  {:>12}  {:>20}  {:>20}  {:?}",
            i,
            meta.block_offset,
            meta.z_min,
            meta.z_max,
            meta.strategy,
        );
    }

    println!("─────────────────────────────────────────────────────────────");
    println!("Total blocks: {}", footer.blocks.len());
    println!("File size:    {} bytes", data.len());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// bench
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Number of rows to generate for the benchmark.
    #[arg(short, long, default_value = "1048576")]
    pub rows: usize,

    /// Data pattern: sequential, random, constant, or mixed.
    #[arg(long, default_value = "sequential")]
    pub pattern: String,
}

pub fn run_bench(args: BenchArgs) -> Result<()> {
    use arrow_array::UInt64Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    println!("FluxCompress benchmark — {} rows, pattern={}", args.rows, args.pattern);
    println!("─────────────────────────────────────────────────────────────");

    let values = generate_pattern(&args.pattern, args.rows);

    // Strategy classification stats.
    let mut strategy_counts = std::collections::HashMap::new();
    for chunk in values.chunks(SEGMENT_SIZE) {
        let r = classify(chunk);
        *strategy_counts.entry(r.strategy).or_insert(0u32) += 1;
    }
    println!("Strategy distribution:");
    for (s, c) in &strategy_counts {
        println!("  {s:?}: {c} blocks");
    }

    // Build Arrow batch.
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::UInt64, false),
    ]));
    let arr = Arc::new(UInt64Array::from(
        values.iter().map(|&v| v as u64).collect::<Vec<_>>(),
    ));
    let batch = Arc::new(arrow_array::RecordBatch::try_new(schema, vec![arr])?);

    // Compress.
    let t0 = Instant::now();
    let compressed = FluxWriter::new().compress(&batch)?;
    let compress_time = t0.elapsed();

    let raw_size = args.rows * 8; // u64 = 8 bytes
    let ratio = compressed.len() as f64 / raw_size as f64;

    println!("\nCompression:");
    println!("  Raw size:        {:>10} bytes", raw_size);
    println!("  Compressed size: {:>10} bytes", compressed.len());
    println!("  Ratio:           {:>10.3}", ratio);
    println!("  Time:            {:>10.2?}", compress_time);
    println!("  Throughput:      {:>10.2} MB/s",
        raw_size as f64 / compress_time.as_secs_f64() / 1_000_000.0
    );

    // Decompress.
    let t1 = Instant::now();
    let reader = FluxReader::new("value");
    let out_batch = reader.decompress_all(&compressed)?;
    let decompress_time = t1.elapsed();

    println!("\nDecompression:");
    println!("  Rows:       {:>10}", out_batch.num_rows());
    println!("  Time:       {:>10.2?}", decompress_time);
    println!("  Throughput: {:>10.2} MB/s",
        raw_size as f64 / decompress_time.as_secs_f64() / 1_000_000.0
    );

    Ok(())
}

fn generate_pattern(pattern: &str, rows: usize) -> Vec<u128> {
    match pattern {
        "sequential" => (0u128..rows as u128).collect(),
        "constant"   => vec![42u128; rows],
        "random"     => (0u128..rows as u128)
            .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0x6c62272e07bb0142))
            .collect(),
        "mixed" => {
            let seg = rows / 4;
            let mut v = Vec::with_capacity(rows);
            v.extend((0u128..seg as u128));                       // sequential
            v.extend(vec![999u128; seg]);                          // constant
            v.extend((0u128..seg as u128).map(|i| i % 8));       // dict
            v.extend((0u128..seg as u128).map(|i|               // sparse outliers
                if i == 42 { u128::MAX } else { i * 17 }
            ));
            v
        }
        _ => {
            warn!("Unknown pattern '{}', falling back to sequential", pattern);
            (0u128..rows as u128).collect()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O helpers (Polars-based)
// ─────────────────────────────────────────────────────────────────────────────

fn load_record_batch(path: &PathBuf) -> Result<arrow_array::RecordBatch> {
    // For now we support only a single-column u64 CSV (one value per line).
    // A production implementation would use polars IPC / parquet readers
    // and convert their output to Arrow RecordBatches.
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "csv" => load_csv_batch(path),
        _     => anyhow::bail!("Unsupported input format: {ext:?} — use CSV for now"),
    }
}

fn load_csv_batch(path: &PathBuf) -> Result<arrow_array::RecordBatch> {
    use arrow_array::UInt64Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let content = std::fs::read_to_string(path)?;
    let values: Vec<u64> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<u64>())
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("parsing CSV {:?}", path))?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::UInt64, false),
    ]));
    let arr = Arc::new(UInt64Array::from(values));
    Ok(arrow_array::RecordBatch::try_new(schema, vec![arr])?)
}

fn save_record_batch(batch: &arrow_array::RecordBatch, path: &PathBuf) -> Result<()> {
    use arrow_array::UInt64Array;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "csv" => {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| anyhow::anyhow!("expected UInt64 column"))?;

            let mut out = String::new();
            for v in col.values() {
                out.push_str(&v.to_string());
                out.push('\n');
            }
            std::fs::write(path, out)?;
            Ok(())
        }
        _ => anyhow::bail!("Unsupported output format: {ext:?} — use CSV"),
    }
}
