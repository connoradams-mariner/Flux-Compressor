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
//!     compress    Compress a Parquet / CSV file into .flux format
//!     decompress  Decompress a .flux file back to Arrow IPC
//!     optimize    Two-pass cold storage optimizer (global dict + Z-Order)
//!     merge       Merge multiple .flux files into one
//!     inspect     Print Atlas footer metadata for a .flux file
//!     bench       Run a quick compression benchmark
//! ```

mod optimizer;
mod inspector;
mod bench;

use clap::{Parser, Subcommand};
use anyhow::Result;
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
    /// Compress a Parquet or CSV file into .flux format.
    Compress {
        #[arg(short, long)]
        input: std::path::PathBuf,
        #[arg(short, long)]
        output: std::path::PathBuf,
        /// auto | rle | delta | dict | bitslab | lz4
        #[arg(short, long, default_value = "auto")]
        strategy: String,
    },

    /// Decompress a .flux file to Arrow IPC.
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
        Commands::Compress { input, output, strategy } => {
            cmd_compress(&input, &output, &strategy)
        }
        Commands::Decompress { input, output } => {
            cmd_decompress(&input, &output)
        }
        Commands::Optimize { input_dir, output, block_kb } => {
            optimizer::cmd_optimize(&input_dir, &output, block_kb)
        }
        Commands::Merge { inputs, output } => {
            cmd_merge(&inputs, &output)
        }
        Commands::Inspect { file, format } => {
            inspector::cmd_inspect(&file, &format)
        }
        Commands::Bench { rows, pattern } => {
            bench::cmd_bench(rows, &pattern)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compress
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_compress(
    input: &std::path::Path,
    output: &std::path::Path,
    strategy: &str,
) -> Result<()> {
    use loom::{
        compressors::flux_writer::FluxWriter,
        loom_classifier::LoomStrategy,
        traits::LoomCompressor,
    };
    use std::fs;

    tracing::info!("Compressing {:?} → {:?}", input, output);

    let batches = load_arrow_batches(input)?;

    let forced: Option<LoomStrategy> = match strategy {
        "auto"    => None,
        "rle"     => Some(LoomStrategy::Rle),
        "delta"   => Some(LoomStrategy::DeltaDelta),
        "dict"    => Some(LoomStrategy::Dictionary),
        "bitslab" => Some(LoomStrategy::BitSlab),
        "lz4"     => Some(LoomStrategy::SimdLz4),
        other     => anyhow::bail!("unknown strategy '{other}'"),
    };

    let writer = match forced {
        Some(s) => FluxWriter::with_strategy(s),
        None    => FluxWriter::new(),
    };

    let flux_bytes = writer.compress_all(&batches)?;
    let in_size = fs::metadata(input)?.len() as usize;
    fs::write(output, &flux_bytes)?;

    println!(
        "Compressed {}  →  {}  ({:.2}x ratio)",
        human_size(in_size),
        human_size(flux_bytes.len()),
        in_size as f64 / flux_bytes.len() as f64,
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// decompress
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_decompress(input: &std::path::Path, output: &std::path::Path) -> Result<()> {
    use loom::{decompressors::flux_reader::FluxReader, traits::LoomDecompressor};
    use arrow::ipc::writer::FileWriter;
    use std::fs::{self, File};

    let flux_bytes = fs::read(input)?;
    let reader = FluxReader::new("value");
    let batch = reader.decompress_all(&flux_bytes)?;

    let file = File::create(output)?;
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref())?;
    writer.write(&batch)?;
    writer.finish()?;

    println!("Decompressed {} rows → {:?}", batch.num_rows(), output);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// merge
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_merge(inputs: &[std::path::PathBuf], output: &std::path::Path) -> Result<()> {
    use std::fs;
    use loom::atlas::{AtlasFooter, BLOCK_META_SIZE};

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

fn load_arrow_batches(path: &std::path::Path) -> Result<Vec<arrow_array::RecordBatch>> {
    use arrow::ipc::reader::FileReader;
    use std::fs::File;
    let file = File::open(path)?;
    let reader = FileReader::try_new(file, None)?;
    let mut batches = Vec::new();
    for b in reader {
        batches.push(b?);
    }
    Ok(batches)
}

pub fn human_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
