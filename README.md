# FluxCompress

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78%2B-orange.svg)](https://www.rust-lang.org)

A high-performance, adaptive columnar storage format designed to outperform
Parquet in the Spark/Polars ecosystem. FluxCompress specialises in
**variable-bit-width compression with outlier patching**, handling massive
`u128` integers and high-precision computed aggregations without the
fixed-width overhead of standard formats.

---

## Repository Layout

```
flux-compress/
├── Cargo.toml                        # Workspace root
└── crates/
    ├── loom/                         # Core compression engine (library)
    │   ├── src/
    │   │   ├── lib.rs
    │   │   ├── bit_io.rs             # Sprint 1 – BitWriter / BitReader
    │   │   ├── outlier_map.rs        # Sprint 2 – OutlierMap (u128 patching)
    │   │   ├── loom_classifier.rs    # Adaptive decision waterfall
    │   │   ├── atlas.rs              # Sprint 4 – Metadata footer + Z-Order
    │   │   ├── traits.rs             # LoomCompressor / LoomDecompressor traits
    │   │   ├── error.rs              # FluxError unified error type
    │   │   ├── simd/
    │   │   │   ├── mod.rs            # Sprint 3 – SIMD dispatcher
    │   │   │   ├── scalar.rs         # Portable fallback unpacker
    │   │   │   ├── avx2.rs           # x86-64 AVX2 unpacker
    │   │   │   └── neon.rs           # AArch64 NEON unpacker
    │   │   ├── compressors/
    │   │   │   ├── bit_slab_compressor.rs   # Primary numeric compressor
    │   │   │   ├── rle_compressor.rs        # Run-Length Encoding
    │   │   │   ├── delta_compressor.rs      # Delta-Delta Encoding
    │   │   │   ├── dict_compressor.rs       # Dictionary Encoding
    │   │   │   ├── lz4_compressor.rs        # SIMD-LZ4 fallback
    │   │   │   └── flux_writer.rs           # Top-level LoomCompressor
    │   │   └── decompressors/
    │   │       ├── block_reader.rs          # Per-block dispatcher
    │   │       └── flux_reader.rs           # Top-level LoomDecompressor
    │   └── benches/
    │       └── bit_slab.rs           # Criterion benchmark suite
    ├── jni-bridge/                   # Sprint 5 – JVM/Spark integration
    │   └── src/lib.rs                # JNI exports + u128 dual-register bridge
    └── fluxcapacitor/                # Sprint 5 – CLI tool
        └── src/
            ├── main.rs              # CLI entry point (clap)
            └── commands.rs          # Sub-command implementations
```

---

## Core Concepts

### The Loom Adaptive Classifier

For every 1 024-row segment the classifier runs a deterministic heuristic
waterfall — no ML, no randomness:

```
1. Entropy ≈ 0?        → RLE
2. Δ₁ constant/narrow? → Delta-Delta
3. Cardinality < 5 %?  → Dictionary
4. Numeric range?       → BitSlab  (+ OutlierMap for u128 overflow)
5. Fallback             → SIMD-LZ4
```

### The BitSlab + OutlierMap

The "secret sauce" for large aggregation results:

```
FoR  = min(block values)
W    = 99th-percentile bit-width of (value − FoR)
sentinel = (1 << W) − 1   // all-ones

For each value x:
  delta = x − FoR
  if delta ≥ sentinel:
    write sentinel to primary slab
    append x (full u128) to OutlierMap
  else:
    write delta (W bits) to primary slab
```

The SIMD decoder reads W bits; when it sees the sentinel it halts, fetches
the next entry from the `OutlierMap`, and increments the patch pointer.
This keeps the hot SIMD lane incredibly narrow while supporting infinite
precision for outliers.

### The Atlas Footer

Every `.flux` file ends with a seekable trailer enabling **Z-Order skipping**
and **predicate pushdown**:

```
[Data Blocks] [BlockMeta × N] [block_count u32] [footer_len u32] [FLUX magic u32]
```

Each `BlockMeta` (50 bytes) stores `block_offset`, `z_min`, `z_max`,
`null_bitmap_offset`, and `strategy_mask`.  The reader skips any block whose
`[z_min, z_max]` range cannot satisfy the query predicate.

### u128 Dual-Register JNI Bridge

Because the JVM has no native `u128` type, large numbers cross the boundary
as two parallel `long[]` arrays (`highWords`, `lowWords`):

```java
// Java / Spark side
long[] high = ...; // bits 127..64
long[] low  = ...; // bits  63..0
byte[] flux = FluxBridge.compressU128Column(high, low);
```

```rust
// Rust reconstruction (zero-copy from off-heap if using DirectByteBuffer)
let value: u128 = (high as u64 as u128) << 64 | low as u64 as u128;
```

---

## Getting Started

### Prerequisites

- Rust 1.78+ (`rustup update stable`)
- For AVX2: x86-64 CPU with AVX2 support
- For JNI bridge: JDK 11+

### Build

```bash
# Build everything
cargo build --release

# Build with AVX2 SIMD (default on x86-64)
cargo build --release --features simd-avx2

# Build with NEON SIMD (AArch64 / Apple Silicon)
cargo build --release --features simd-neon
```

### Run Tests

```bash
# All unit and integration tests
cargo test --workspace

# Tests with verbose output
cargo test --workspace -- --nocapture
```

### Run Benchmarks

```bash
# Full Criterion benchmark suite
cargo bench -p loom

# Open the HTML report
open target/criterion/report/index.html
```

### CLI Usage

```bash
# Compress a CSV file
fluxcapacitor compress data.csv -o output.flux

# Two-pass cold optimisation of a partition directory
fluxcapacitor optimize partitions/ -o optimised.flux

# Merge multiple .flux files
fluxcapacitor merge part1.flux part2.flux part3.flux -o merged.flux

# Inspect a .flux file (print Atlas footer)
fluxcapacitor inspect output.flux

# Benchmark decompression throughput
fluxcapacitor bench output.flux --iters 50
```

---

## Implementation Sprints

| Sprint | Deliverable | File(s) |
|--------|-------------|---------|
| 1 | `BitWriter` / `BitReader` with basic bit-packing | `bit_io.rs` |
| 2 | `OutlierMap` for `u128` precision overflow | `outlier_map.rs` |
| 3 | SIMD Unpacker (AVX2 / NEON / scalar) | `simd/` |
| 4 | Atlas Metadata Footer + Z-Order | `atlas.rs` |
| 5 | JNI Bridge (Spark) + `fluxcapacitor` CLI | `jni-bridge/`, `fluxcapacitor/` |

---

## Design Principles

**Zero-Copy** — All hot paths borrow `&[u8]` slices into caller-owned or
off-heap buffers. The JNI bridge uses `DirectByteBuffer` to read Arrow
columns without copying from the JVM heap to the Rust heap.

**FastestRead Priority** — The Loom classifier defaults to strategies that
minimise decode latency (BitSlab + SIMD), not encode time.

**Hot vs Cold Storage** — The `FluxWriter` (hot) writes independent blocks
for parallel Spark writes. `fluxcapacitor optimize` (cold) performs a
two-pass global re-pack with a master dictionary and Z-Order interleaving
for maximum density.

**Safe `u128` Handling** — All 128-bit arithmetic uses Rust's native `u128`
type. The JNI boundary splits values into `(high: i64, low: i64)` pairs and
reconstructs them on the Rust side with bitwise operations, avoiding any
undefined behaviour from bit-casting.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
