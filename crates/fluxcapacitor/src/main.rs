// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # fluxcapacitor
//!
//! The FluxCompress command-line tool.
//!
//! ```text
//! USAGE:
//!     fluxcapacitor <COMMAND>
//!
//! COMMANDS:
//!     compress    Compress a tabular file (CSV/TSV/JSON/Parquet/Arrow/ORC/XLSX/...) into .flux
//!     decompress  Decompress a .flux file back to any supported tabular format
//!     optimize    Two-pass cold storage optimizer (global dict + Z-Order)
//!     merge       Merge multiple .flux files into one
//!     inspect     Print Atlas footer metadata for a .flux file
//!     bench       Run a quick compression benchmark
//! ```

mod bench;
mod compare_bench;
mod dtype_bench;
mod inspector;
mod mixed_bench;
mod optimizer;
mod string_bench;

use fluxcapacitor::formats;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

// ─────────────────────────────────────────────────────────────────────────────
// CLI definition
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name        = "fluxcapacitor",
    version     = env!("CARGO_PKG_VERSION"),
    author      = "FluxCompress Contributors",
    about       = "FluxCompress CLI – adaptive columnar storage optimizer",
)]
struct Cli {
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a tabular file into .flux format.
    ///
    /// Input format is auto-detected from the file extension. Supported:
    /// `.csv`, `.tsv`, `.json`, `.ndjson`/`.jsonl`, `.parquet`, `.arrow`/
    /// `.ipc`/`.feather`, `.orc`, `.xlsx`, `.xls`, `.xlsm`, `.ods`.
    Compress {
        #[arg(short, long)]
        input: std::path::PathBuf,
        #[arg(short, long)]
        output: std::path::PathBuf,
        /// auto | rle | delta | dict | bitslab | lz4
        #[arg(short, long, default_value = "auto")]
        strategy: String,
    },

    /// Decompress a .flux file to any supported tabular format.
    ///
    /// Output format is chosen by the `output` extension: `.csv`, `.tsv`,
    /// `.json`, `.ndjson`/`.jsonl`, `.parquet`, `.arrow`/`.ipc`/`.feather`,
    /// `.orc`, or `.xlsx`. (Excel `.xls`/`.xlsm`/`.ods` are read-only.)
    Decompress {
        #[arg(short, long)]
        input: std::path::PathBuf,
        #[arg(short, long)]
        output: std::path::PathBuf,
    },

    /// Two-pass global optimization: global dict + Z-Order re-packing.
    Optimize {
        #[arg(short, long)]
        input_dir: std::path::PathBuf,
        #[arg(short, long)]
        output: std::path::PathBuf,
        /// Target block size in KB (default: 256)
        #[arg(long, default_value_t = 256)]
        block_kb: usize,
    },

    /// Merge multiple .flux files into one archive.
    Merge {
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<std::path::PathBuf>,
        #[arg(short, long)]
        output: std::path::PathBuf,
    },

    /// Print the Atlas metadata footer for a .flux file.
    Inspect {
        file: std::path::PathBuf,
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Run a quick compression/decompression benchmark.
    Bench {
        #[arg(long, default_value_t = 1_000_000)]
        rows: usize,
        /// sequential | random | constant | outlier
        #[arg(long, default_value = "sequential")]
        pattern: String,
    },

    /// Multi-datatype benchmark: Flux vs Parquet vs ORC across all Arrow types.
    DtypeBench {
        #[arg(long, default_value_t = 1_000_000)]
        rows: usize,
    },

    /// High-cardinality string compression benchmark (per-pattern).
    StringBench {
        #[arg(long, default_value_t = 10_000_000)]
        rows: usize,
    },

    /// Realistic 22-column mixed-schema benchmark (mirrors Databricks test).
    MixedBench {
        #[arg(long, default_value_t = 9_950_000)]
        rows: usize,
    },

    /// Three-way compare: Flux vs Parquet vs Delta Lake on a 22-column
    /// mixed-schema dataset (Delta = Parquet + _delta_log/ JSON log).
    CompareBench {
        #[arg(long, default_value_t = 1_000_000)]
        rows: usize,
    },

    /// Three-way compare on a string-heavy schema (10 string columns +
    /// 2 ids + 1 timestamp) — models log / event / clickstream data.
    StringCompareBench {
        #[arg(long, default_value_t = 1_000_000)]
        rows: usize,
    },

    /// Three-way compare on a float-heavy schema (10 Float64 columns +
    /// 2 ids + 1 timestamp) — models scientific / IoT / financial data.
    FloatCompareBench {
        #[arg(long, default_value_t = 1_000_000)]
        rows: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match cli.command {
        Commands::Compress {
            input,
            output,
            strategy,
        } => cmd_compress(&input, &output, &strategy),
        Commands::Decompress { input, output } => cmd_decompress(&input, &output),
        Commands::Optimize {
            input_dir,
            output,
            block_kb,
        } => optimizer::cmd_optimize(&input_dir, &output, block_kb),
        Commands::Merge { inputs, output } => cmd_merge(&inputs, &output),
        Commands::Inspect { file, format } => inspector::cmd_inspect(&file, &format),
        Commands::Bench { rows, pattern } => bench::cmd_bench(rows, &pattern),
        Commands::DtypeBench { rows } => dtype_bench::cmd_dtype_bench(rows),
        Commands::StringBench { rows } => string_bench::cmd_string_bench(rows),
        Commands::MixedBench { rows } => mixed_bench::cmd_mixed_bench(rows),
        Commands::CompareBench { rows } => compare_bench::cmd_compare_bench(rows),
        Commands::StringCompareBench { rows } => compare_bench::cmd_string_compare_bench(rows),
        Commands::FloatCompareBench { rows } => compare_bench::cmd_float_compare_bench(rows),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compress
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_compress(input: &std::path::Path, output: &std::path::Path, strategy: &str) -> Result<()> {
    use loom::{
        compressors::flux_writer::FluxWriter, loom_classifier::LoomStrategy, traits::LoomCompressor,
    };
    use std::fs;

    let format = formats::FileFormat::from_path(input)?;
    tracing::info!(
        "Compressing {:?} ({}) → {:?}",
        input, format.label(), output,
    );

    let batches = formats::load_batches(input)?;
    if batches.is_empty() {
        anyhow::bail!("input {:?} produced zero RecordBatches", input);
    }

    let forced: Option<LoomStrategy> = match strategy {
        "auto" => None,
        "rle" => Some(LoomStrategy::Rle),
        "delta" => Some(LoomStrategy::DeltaDelta),
        "dict" => Some(LoomStrategy::Dictionary),
        "bitslab" => Some(LoomStrategy::BitSlab),
        "lz4" => Some(LoomStrategy::SimdLz4),
        other => anyhow::bail!("unknown strategy '{other}'"),
    };

    let writer = match forced {
        Some(s) => FluxWriter::with_strategy(s),
        None => FluxWriter::new(),
    };

    let flux_bytes = writer.compress_all(&batches)?;
    let in_size = fs::metadata(input)?.len() as usize;
    fs::write(output, &flux_bytes)?;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "Compressed {} ({} rows, {}) → {} ({:.2}× vs source bytes)",
        format.label(),
        total_rows,
        human_size(in_size),
        human_size(flux_bytes.len()),
        in_size as f64 / flux_bytes.len().max(1) as f64,
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// decompress
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_decompress(input: &std::path::Path, output: &std::path::Path) -> Result<()> {
    use loom::{decompressors::flux_reader::FluxReader, traits::LoomDecompressor};
    use std::fs;

    let flux_bytes = fs::read(input)
        .map_err(|e| anyhow::anyhow!("reading {:?}: {e}", input))?;
    let reader = FluxReader::new("value");
    let batch = reader.decompress_all(&flux_bytes)?;

    let format = formats::FileFormat::from_path(output)?;
    formats::save_batches(output, std::slice::from_ref(&batch))?;

    println!(
        "Decompressed {} rows → {:?} ({})",
        batch.num_rows(), output, format.label()
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// merge
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_merge(inputs: &[std::path::PathBuf], output: &std::path::Path) -> Result<()> {
    use loom::atlas::{AtlasFooter, BLOCK_META_SIZE};
    use std::fs;

    let mut combined_data: Vec<u8> = Vec::new();
    let mut combined_footer = AtlasFooter::new();

    for path in inputs {
        let bytes = fs::read(path)?;
        let footer = AtlasFooter::from_file_tail(&bytes)?;
        let footer_len = footer.blocks.len() * BLOCK_META_SIZE + 12;
        let data_end = bytes.len() - footer_len;
        let offset_base = combined_data.len() as u64;

        combined_data.extend_from_slice(&bytes[..data_end]);

        for mut meta in footer.blocks {
            meta.block_offset += offset_base;
            combined_footer.push(meta);
        }
    }

    combined_data.extend(combined_footer.to_bytes()?);
    fs::write(output, &combined_data)?;

    println!(
        "Merged {} files → {} ({} blocks)",
        inputs.len(),
        human_size(combined_data.len()),
        combined_footer.blocks.len(),
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn human_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
