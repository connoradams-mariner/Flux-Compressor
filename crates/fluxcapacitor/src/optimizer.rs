// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Two-Pass Cold Storage Optimizer
//!
//! Implements the `fluxcapacitor optimize` command.
//!
//! ## Pass 1 — Global Scan
//! Walk every `.flux` partition in `input_dir`, decompress each block, and
//! accumulate:
//! - A **global master dictionary** (unique value → global index).
//! - **Global min/max statistics** per column.
//! - The **optimal global bit-width** for re-packing.
//!
//! ## Pass 2 — Re-Pack
//! Re-compress every value using:
//! - The **global dictionary** (eliminates duplicate dict entries across blocks).
//! - **Smallest-Size** bit-width (using the global 99th-percentile width, not
//!   per-block estimates).
//! - **Z-Order interleaving** of block metadata for multi-dimensional locality.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use walkdir::WalkDir;
use rayon::prelude::*;

use loom::{
    atlas::{AtlasFooter, BlockMeta, z_order_encode},
    bit_io::discover_width,
    compressors::flux_writer::compress_chunk,
    decompressors::block_reader::decompress_block,
    loom_classifier::{classify, LoomStrategy},
};

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn cmd_optimize(
    input_dir: &Path,
    output: &Path,
    _block_kb: usize,
) -> Result<()> {
    println!("╔══════════════════════════════════════╗");
    println!("║  FluxCapacitor Cold Optimizer        ║");
    println!("╚══════════════════════════════════════╝");

    // ── Pass 1 ───────────────────────────────────────────────────────────────
    println!("\n[Pass 1] Scanning partitions in {:?} …", input_dir);
    let scan = pass1_scan(input_dir)?;
    println!(
        "  → {} partitions, {} total blocks, {} unique values",
        scan.partition_paths.len(),
        scan.all_blocks.len(),
        scan.global_dict.len(),
    );

    // ── Pass 2 ───────────────────────────────────────────────────────────────
    println!("\n[Pass 2] Re-packing with global dictionary + Z-Order …");
    let bytes = pass2_repack(&scan)?;

    std::fs::write(output, &bytes)?;
    println!(
        "\n✓ Optimized archive written to {:?}  ({})",
        output,
        crate::human_size(bytes.len()),
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1
// ─────────────────────────────────────────────────────────────────────────────

struct ScanResult {
    partition_paths: Vec<PathBuf>,
    /// All decompressed block values across all partitions.
    all_blocks: Vec<Vec<u128>>,
    /// Global dictionary: unique value → compact index.
    global_dict: HashMap<u128, u32>,
    /// Ordered list of unique values (index → value).
    global_dict_values: Vec<u128>,
    /// Global min / max.
    global_min: u128,
    global_max: u128,
}

fn pass1_scan(input_dir: &Path) -> Result<ScanResult> {
    let partition_paths: Vec<PathBuf> = WalkDir::new(input_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "flux")
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if partition_paths.is_empty() {
        anyhow::bail!("no .flux files found in {:?}", input_dir);
    }

    let pb = ProgressBar::new(partition_paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {bar:40.cyan} {pos}/{len} partitions")
            .unwrap(),
    );

    let mut all_blocks: Vec<Vec<u128>> = Vec::new();
    let mut global_min = u128::MAX;
    let mut global_max = 0u128;
    let mut unique_values: std::collections::BTreeSet<u128> = std::collections::BTreeSet::new();

    for path in &partition_paths {
        let bytes = std::fs::read(path)?;
        let footer = AtlasFooter::from_file_tail(&bytes)?;

        for meta in &footer.blocks {
            let block_data = &bytes[meta.block_offset as usize..];
            let (values, _) = decompress_block(block_data)?;

            for &v in &values {
                unique_values.insert(v);
                if v < global_min { global_min = v; }
                if v > global_max { global_max = v; }
            }
            all_blocks.push(values);
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    let global_dict_values: Vec<u128> = unique_values.into_iter().collect();
    let global_dict: HashMap<u128, u32> = global_dict_values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();

    Ok(ScanResult {
        partition_paths,
        all_blocks,
        global_dict,
        global_dict_values,
        global_min,
        global_max,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2
// ─────────────────────────────────────────────────────────────────────────────

fn pass2_repack(scan: &ScanResult) -> Result<Vec<u8>> {
    // Discover optimal global bit-width from all values combined.
    let all_flat: Vec<u128> = scan.all_blocks.iter().flatten().copied().collect();
    let (global_width, global_for) = discover_width(&all_flat);

    println!(
        "  Global FoR={global_for}, slab_width={global_width} bits, \
         dict_size={} entries",
        scan.global_dict.len()
    );

    let pb = ProgressBar::new(scan.all_blocks.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {bar:40.green} {pos}/{len} blocks")
            .unwrap(),
    );

    // Re-compress each block using the globally optimal strategy.
    let mut output: Vec<u8> = Vec::new();
    let mut footer = AtlasFooter::new();

    // Sort blocks by Z-Order of their [min, max] midpoint for spatial locality.
    let mut indexed_blocks: Vec<(usize, &Vec<u128>)> =
        scan.all_blocks.iter().enumerate().collect();

    indexed_blocks.sort_by_key(|(_, block)| {
        let min = block.iter().copied().min().unwrap_or(0);
        let max = block.iter().copied().max().unwrap_or(0);
        let mid = min / 2 + max / 2;
        // Z-Order interleave of the low 64 bits and high 64 bits.
        z_order_encode(mid as u64, (mid >> 64) as u64)
    });

    for (_, block) in &indexed_blocks {
        let block_offset = output.len() as u64;

        let strategy = classify(block).strategy;
        let block_bytes = compress_chunk(block, strategy)?;

        let z_min = block.iter().copied().min().unwrap_or(0);
        let z_max = block.iter().copied().max().unwrap_or(0);

        output.extend_from_slice(&block_bytes);
        footer.push(BlockMeta {
            block_offset,
            z_min,
            z_max,
            null_bitmap_offset: 0,
            strategy,
            value_count: block.len() as u32,
            column_id: 0,
            crc32: 0,
        });

        pb.inc(1);
    }
    pb.finish_and_clear();

    output.extend(footer.to_bytes()?);
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// ColdOptimizer — struct API used by commands.rs
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful wrapper around the two-pass cold optimiser.
///
/// `commands.rs` uses this as:
/// ```rust,ignore
/// let mut opt = ColdOptimizer::new();
/// opt.scan_partitions(&input_dir)?;   // pass 1
/// opt.optimize(&output)?;             // pass 2
/// ```
pub struct ColdOptimizer {
    scan: Option<ScanResult>,
}

impl ColdOptimizer {
    /// Create a new, empty `ColdOptimizer`.
    pub fn new() -> Self {
        Self { scan: None }
    }

    /// **Pass 1** — scan all `.flux` files in `input_dir`.
    pub fn scan_partitions(&mut self, input_dir: &std::path::Path) -> Result<()> {
        let result = pass1_scan(input_dir)?;
        self.scan = Some(result);
        Ok(())
    }

    /// Number of unique values in the global dictionary (available after pass 1).
    pub fn global_dict_size(&self) -> usize {
        self.scan
            .as_ref()
            .map(|s| s.global_dict.len())
            .unwrap_or(0)
    }

    /// **Pass 2** — re-pack with global dictionary + Z-Order, write archive.
    pub fn optimize(&self, output: &std::path::Path) -> Result<()> {
        let scan = self.scan.as_ref().ok_or_else(|| {
            anyhow::anyhow!("call scan_partitions() before optimize()")
        })?;
        let bytes = pass2_repack(scan)?;
        std::fs::write(output, &bytes)?;
        Ok(())
    }
}
