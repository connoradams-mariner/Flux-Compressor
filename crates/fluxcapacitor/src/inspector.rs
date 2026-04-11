// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! `fluxcapacitor inspect` — print the Atlas metadata footer of a `.flux` file.

use std::path::Path;
use anyhow::Result;
use loom::atlas::AtlasFooter;

pub fn cmd_inspect(path: &Path, format: &str) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let footer = AtlasFooter::from_file_tail(&bytes)?;

    match format {
        "json" => print_json(&footer),
        _      => print_table(&footer, &bytes),
    }
    Ok(())
}

fn print_table(footer: &AtlasFooter, raw: &[u8]) {
    println!("╔══ Atlas Footer ══════════════════════════════════════════════════════╗");
    println!(
        "  File size : {}",
        crate::human_size(raw.len())
    );
    println!("  Blocks    : {}", footer.blocks.len());
    println!();
    println!(
        "  {:>6}  {:>14}  {:>20}  {:>20}  {:>12}",
        "Block", "Offset", "Z-Min", "Z-Max", "Strategy"
    );
    println!("  {}", "─".repeat(78));
    for (i, b) in footer.blocks.iter().enumerate() {
        println!(
            "  {:>6}  {:>14}  {:>20}  {:>20}  {:>12?}",
            i,
            b.block_offset,
            b.z_min,
            b.z_max,
            b.strategy,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}

fn print_json(footer: &AtlasFooter) {
    println!("{{");
    println!("  \"block_count\": {},", footer.blocks.len());
    println!("  \"blocks\": [");
    for (i, b) in footer.blocks.iter().enumerate() {
        let comma = if i + 1 < footer.blocks.len() { "," } else { "" };
        println!("    {{");
        println!("      \"index\": {i},");
        println!("      \"block_offset\": {},", b.block_offset);
        println!("      \"z_min\": \"{}\",", b.z_min);
        println!("      \"z_max\": \"{}\",", b.z_max);
        println!("      \"null_bitmap_offset\": {},", b.null_bitmap_offset);
        println!("      \"strategy\": \"{:?}\"", b.strategy);
        println!("    }}{comma}");
    }
    println!("  ]");
    println!("}}");
}
