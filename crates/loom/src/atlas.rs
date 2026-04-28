// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Atlas Metadata Footer (Sprint 4)
//!
//! The seekable trailer at the end of every `.flux` file that enables
//! **Z-Order skipping** and **predicate pushdown**.
//!
//! ## Binary Layout
//! ```text
//! ┌──────────────────────────────┐
//! │   Data Blocks (variable)     │
//! ├──────────────────────────────┤
//! │   Atlas Footer               │
//! │   ├─ [BlockMeta × N]        │
//! │   ├─ u32: block_count        │
//! │   ├─ u32: footer_length      │
//! │   └─ u32: FLUX_MAGIC (FLUX)  │
//! └──────────────────────────────┘
//! ```
//!
//! ## Per-Block Metadata (50 bytes)
//! | Field              | Size | Description                          |
//! |--------------------|------|--------------------------------------|
//! | `block_offset`     | 8 B  | Seek point for the block data        |
//! | `z_min`            | 16 B | Min 128-bit Z-Order coordinate       |
//! | `z_max`            | 16 B | Max 128-bit Z-Order coordinate       |
//! | `null_bitmap_off`  | 8 B  | Pointer to the null-mask             |
//! | `strategy_mask`    | 2 B  | Encoded [`LoomStrategy`] ID          |
//!
//! Total: 8 + 16 + 16 + 8 + 2 = **50 bytes per block**.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Write};

use crate::{
    FLUX_MAGIC,
    dtype::FluxDType,
    error::{FluxError, FluxResult},
    loom_classifier::LoomStrategy,
    traits::Predicate,
};

/// Size of one serialised [`BlockMeta`] entry in bytes (v2: 61 bytes).
pub const BLOCK_META_SIZE: usize = 61;

// ─────────────────────────────────────────────────────────────────────────────
// BlockMeta
// ─────────────────────────────────────────────────────────────────────────────

/// Per-block metadata stored in the Atlas footer (v2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMeta {
    /// Byte offset of this block's data within the `.flux` file.
    pub block_offset: u64,
    /// Minimum Z-Order coordinate (or column min value) for this block.
    pub z_min: u128,
    /// Maximum Z-Order coordinate (or column max value) for this block.
    pub z_max: u128,
    /// Byte offset of the null-bitmap for this block.
    pub null_bitmap_offset: u64,
    /// Which [`LoomStrategy`] was used to compress this block.
    pub strategy: LoomStrategy,
    /// Number of values in this block.
    pub value_count: u32,
    /// Column index within the RecordBatch (for multi-column support).
    pub column_id: u16,
    /// CRC32 checksum of the compressed block data.
    pub crc32: u32,
    /// Whether this block was compressed in u64-only mode (no u128 patching).
    pub u64_only: bool,
    /// Original Arrow data type tag for lossless reconstruction.
    pub dtype_tag: FluxDType,
}

impl BlockMeta {
    /// Serialise this entry to exactly [`BLOCK_META_SIZE`] bytes (v2: 61).
    pub fn to_bytes(&self) -> FluxResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(BLOCK_META_SIZE);
        buf.write_u64::<LittleEndian>(self.block_offset)?;
        buf.write_u64::<LittleEndian>(self.z_min as u64)?;
        buf.write_u64::<LittleEndian>((self.z_min >> 64) as u64)?;
        buf.write_u64::<LittleEndian>(self.z_max as u64)?;
        buf.write_u64::<LittleEndian>((self.z_max >> 64) as u64)?;
        buf.write_u64::<LittleEndian>(self.null_bitmap_offset)?;
        buf.write_u16::<LittleEndian>(self.strategy.encode_mask(self.u64_only))?;
        buf.write_u32::<LittleEndian>(self.value_count)?;
        buf.write_u16::<LittleEndian>(self.column_id)?;
        buf.write_u32::<LittleEndian>(self.crc32)?;
        buf.write_u8(self.dtype_tag.as_u8())?;
        debug_assert_eq!(buf.len(), BLOCK_META_SIZE);
        Ok(buf)
    }

    /// Deserialise from a [`BLOCK_META_SIZE`]-byte slice.
    pub fn from_bytes(data: &[u8]) -> FluxResult<Self> {
        if data.len() < BLOCK_META_SIZE {
            return Err(FluxError::InvalidFile(format!(
                "block meta too short: {} < {}",
                data.len(),
                BLOCK_META_SIZE
            )));
        }
        let mut cur = Cursor::new(data);
        let block_offset = cur.read_u64::<LittleEndian>()?;
        let z_min_lo = cur.read_u64::<LittleEndian>()? as u128;
        let z_min_hi = cur.read_u64::<LittleEndian>()? as u128;
        let z_max_lo = cur.read_u64::<LittleEndian>()? as u128;
        let z_max_hi = cur.read_u64::<LittleEndian>()? as u128;
        let null_bitmap_offset = cur.read_u64::<LittleEndian>()?;
        let strategy_raw = cur.read_u16::<LittleEndian>()?;

        let u64_only = LoomStrategy::is_u64_only(strategy_raw);
        let strategy = LoomStrategy::from_u16(strategy_raw).ok_or_else(|| {
            FluxError::InvalidFile(format!("unknown strategy mask: {strategy_raw:#06x}"))
        })?;
        let value_count = cur.read_u32::<LittleEndian>()?;
        let column_id = cur.read_u16::<LittleEndian>()?;
        let crc32 = cur.read_u32::<LittleEndian>()?;
        let dtype_tag_raw = cur.read_u8()?;
        let dtype_tag = FluxDType::from_u8(dtype_tag_raw).unwrap_or(FluxDType::UInt64);

        Ok(Self {
            block_offset,
            z_min: z_min_lo | (z_min_hi << 64),
            z_max: z_max_lo | (z_max_hi << 64),
            null_bitmap_offset,
            strategy,
            value_count,
            column_id,
            crc32,
            u64_only,
            dtype_tag,
        })
    }

    /// Returns `true` if this block *may* satisfy `predicate` based on its
    /// `[z_min, z_max]` hyper-rectangle.  Used for Z-Order skipping.
    pub fn may_satisfy(&self, predicate: &Predicate) -> bool {
        predicate.may_overlap(self.z_min as i128, self.z_max as i128)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AtlasFooter
// ─────────────────────────────────────────────────────────────────────────────

/// Descriptor for a column in the schema tree. Leaf columns have empty
/// `children` and a valid `column_id` that maps to `BlockMeta.column_id`.
/// Container columns (Struct/List/Map) have children but no blocks of their own.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnDescriptor {
    /// Column name (from the Arrow schema).
    pub name: String,
    /// Compact dtype tag.
    pub dtype_tag: u8,
    /// Child descriptors (empty for leaf columns).
    pub children: Vec<ColumnDescriptor>,
    /// Leaf column ID mapping to `BlockMeta.column_id`. `u16::MAX` for containers.
    pub column_id: u16,
    /// Phase E: logical `field_id` from the table's `TableSchema`.
    ///
    /// `None` on pre-Phase-E files and on columns the writer didn't
    /// have a field_id for (e.g. ad-hoc compressions that don't
    /// originate from a FluxTable schema). Serialised only when set
    /// so old `.flux` files stay byte-identical on the wire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_id: Option<u32>,
}

/// The complete Atlas footer: a list of [`BlockMeta`] entries plus a trailer.
#[derive(Debug, Default, Clone)]
pub struct AtlasFooter {
    /// Metadata for each compressed block, in file order.
    pub blocks: Vec<BlockMeta>,
    /// Optional schema tree for nested type reconstruction.
    /// Empty for flat (non-nested) schemas.
    pub schema: Vec<ColumnDescriptor>,
}

impl AtlasFooter {
    /// Create an empty footer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a block metadata entry.
    pub fn push(&mut self, meta: BlockMeta) {
        self.blocks.push(meta);
    }

    /// Serialise the footer to bytes.
    ///
    /// Layout: `[BlockMeta × N][schema_json (if non-empty)][u32: schema_len][u32: block_count][u32: footer_length][u32: MAGIC]`
    pub fn to_bytes(&self) -> FluxResult<Vec<u8>> {
        let block_count = self.blocks.len() as u32;
        let block_section_len = block_count as usize * BLOCK_META_SIZE;

        // Serialize schema tree as JSON (empty string if no nested types).
        let schema_json = if self.schema.is_empty() {
            Vec::new()
        } else {
            serde_json::to_vec(&self.schema)
                .map_err(|e| FluxError::Internal(format!("schema serialize: {e}")))?
        };
        let schema_len = schema_json.len() as u32;

        // footer_length = blocks + schema + 4(schema_len) + 4(count) + 4(length) + 4(magic)
        let footer_length = (block_section_len + schema_json.len() + 16) as u32;

        let mut buf = Vec::with_capacity(footer_length as usize);

        for meta in &self.blocks {
            buf.extend(meta.to_bytes()?);
        }
        if !schema_json.is_empty() {
            buf.extend_from_slice(&schema_json);
        }
        buf.write_u32::<LittleEndian>(schema_len)?;
        buf.write_u32::<LittleEndian>(block_count)?;
        buf.write_u32::<LittleEndian>(footer_length)?;
        buf.write_u32::<LittleEndian>(FLUX_MAGIC)?;

        Ok(buf)
    }

    /// Locate and deserialise the Atlas footer from the end of a `.flux` file.
    ///
    /// The last 4 bytes must be `FLUX_MAGIC`.
    /// The preceding 4 bytes are `footer_length`.
    pub fn from_file_tail(data: &[u8]) -> FluxResult<Self> {
        if data.len() < 16 {
            return Err(FluxError::InvalidFile(
                "file too small for flux footer".into(),
            ));
        }

        // Validate magic.
        let magic_off = data.len() - 4;
        let magic = u32::from_le_bytes(data[magic_off..].try_into().unwrap());
        if magic != FLUX_MAGIC {
            return Err(FluxError::InvalidFile(format!(
                "bad magic: expected {FLUX_MAGIC:#010x}, got {magic:#010x}"
            )));
        }

        // Read footer length.
        let len_off = data.len() - 8;
        let footer_length =
            u32::from_le_bytes(data[len_off..len_off + 4].try_into().unwrap()) as usize;

        if footer_length > data.len() {
            return Err(FluxError::InvalidFile(
                "footer_length exceeds file size".into(),
            ));
        }

        let footer_start = data.len() - footer_length;
        let footer_data = &data[footer_start..];

        // Read block count (4 bytes before footer_length).
        let count_off = footer_data.len() - 12;
        let block_count =
            u32::from_le_bytes(footer_data[count_off..count_off + 4].try_into().unwrap()) as usize;

        // Read schema length (4 bytes before block_count).
        let schema_len_off = footer_data.len() - 16;
        let schema_len = u32::from_le_bytes(
            footer_data[schema_len_off..schema_len_off + 4]
                .try_into()
                .unwrap(),
        ) as usize;

        let block_section_len = block_count * BLOCK_META_SIZE;

        let mut blocks = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let off = i * BLOCK_META_SIZE;
            blocks.push(BlockMeta::from_bytes(
                &footer_data[off..off + BLOCK_META_SIZE],
            )?);
        }

        // Parse schema JSON if present.
        let schema = if schema_len > 0 {
            let schema_start = block_section_len;
            let schema_end = schema_start + schema_len;
            serde_json::from_slice(&footer_data[schema_start..schema_end])
                .map_err(|e| FluxError::Internal(format!("schema deserialize: {e}")))?
        } else {
            Vec::new()
        };

        Ok(Self { blocks, schema })
    }

    /// Filter blocks to those that *may* satisfy `predicate`.
    ///
    /// Returns indices into `self.blocks` that pass the Z-Order check.
    pub fn candidate_blocks(&self, predicate: &Predicate) -> Vec<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, m)| m.may_satisfy(predicate))
            .map(|(i, _)| i)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Z-Order interleaving utility
// ─────────────────────────────────────────────────────────────────────────────

/// Interleave two 64-bit coordinates into a 128-bit Z-Order (Morton) code.
///
/// Used by the Cold optimizer to sort blocks for multi-dimensional locality.
pub fn z_order_encode(x: u64, y: u64) -> u128 {
    let mut result: u128 = 0;
    for i in 0..64u32 {
        let x_bit = ((x >> i) & 1) as u128;
        let y_bit = ((y >> i) & 1) as u128;
        result |= x_bit << (2 * i);
        result |= y_bit << (2 * i + 1);
    }
    result
}

/// Decode a Z-Order (Morton) code back to two 64-bit coordinates.
pub fn z_order_decode(z: u128) -> (u64, u64) {
    let mut x: u64 = 0;
    let mut y: u64 = 0;
    for i in 0..64u32 {
        x |= (((z >> (2 * i)) & 1) as u64) << i;
        y |= (((z >> (2 * i + 1)) & 1) as u64) << i;
    }
    (x, y)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(offset: u64, min: u128, max: u128) -> BlockMeta {
        BlockMeta {
            block_offset: offset,
            z_min: min,
            z_max: max,
            null_bitmap_offset: 0,
            strategy: LoomStrategy::BitSlab,
            value_count: 1024,
            column_id: 0,
            crc32: 0,
            u64_only: false,
            dtype_tag: FluxDType::UInt64,
        }
    }

    #[test]
    fn block_meta_round_trip() {
        let meta = make_meta(1024, 0, u128::MAX / 2);
        let bytes = meta.to_bytes().unwrap();
        assert_eq!(bytes.len(), BLOCK_META_SIZE);
        let decoded = BlockMeta::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, meta);
    }

    #[test]
    fn atlas_footer_round_trip() {
        let mut footer = AtlasFooter::new();
        footer.push(make_meta(0, 0, 100));
        footer.push(make_meta(4096, 101, 200));
        footer.push(make_meta(8192, 201, 300));

        // Simulate a flux file: some data bytes + the footer.
        let mut file = vec![0u8; 8192 + 512]; // dummy data
        let footer_bytes = footer.to_bytes().unwrap();
        file.extend_from_slice(&footer_bytes);

        let parsed = AtlasFooter::from_file_tail(&file).unwrap();
        assert_eq!(parsed.blocks.len(), 3);
        assert_eq!(parsed.blocks[1].block_offset, 4096);
        assert_eq!(parsed.blocks[2].z_min, 201);
    }

    #[test]
    fn predicate_pushdown_skips_blocks() {
        let mut footer = AtlasFooter::new();
        footer.push(make_meta(0, 0, 50)); // min=0, max=50
        footer.push(make_meta(100, 51, 150)); // min=51, max=150
        footer.push(make_meta(200, 151, 300)); // min=151, max=300

        // Predicate: value > 100 — should skip block 0.
        let pred = Predicate::GreaterThan {
            column: "x".into(),
            value: 100,
        };
        let candidates = footer.candidate_blocks(&pred);
        assert_eq!(candidates, vec![1, 2]);
    }

    #[test]
    fn z_order_round_trip() {
        let (x, y) = (0xDEAD_BEEF_u64, 0xCAFE_BABE_u64);
        let z = z_order_encode(x, y);
        let (dx, dy) = z_order_decode(z);
        assert_eq!(dx, x);
        assert_eq!(dy, y);
    }
}
