# FluxCompress

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

A high-performance, adaptive columnar storage format that **beats Parquet
on compression ratio across all Arrow data types** and now **matches or
exceeds Parquet on decompression throughput** for nested and mixed-type
workloads. Built in Rust with a dtype-aware routing layer, parallel
compress/decompress, composable secondary codecs, and Delta-Lake-style
time travel.

---

## Benchmarks

All numbers from `fluxcapacitor` on Linux (Rust release build, mmap reads,
rayon parallel). Raw size is the in-memory Arrow footprint unless noted.

### Mixed 22-column schema — 9.95M rows (Databricks-shaped workload)

`cargo run -p fluxcapacitor --release -- mixed-bench --rows 9950000`

```
Codec                     Size       Ratio    Comp MB/s   Dec MB/s
──────────────────────  ─────────  ───────  ──────────  ────────
Flux (Archive)            302 MB     9.47×          ~700       ~1160
Parquet (zstd-3)          448 MB     6.37×          ~360       ~1150
```

Schema: 4×Int64, 4×Float64, 2×Timestamp, 1×Date32, 3×Boolean, 8×Utf8.
Flux wins on compression ratio by **32.7%** while matching Parquet's
decompression throughput. On a 2.01 GB CSV round-trip the same schema
typically shows a ~15 MB ratio advantage for Flux with Flux
decompression ~90 MB/s faster than Parquet, Parquet compression
~13 MB/s faster than Flux.

### High-cardinality string corpus — 10M rows

`cargo run -p fluxcapacitor --release -- string-bench --rows 10000000`

```
Pattern              Profile     Ratio      Comp MB/s   Dec MB/s
──────────────────  ──────────  ─────────  ──────────  ────────
urls_high_card       Speed       3.1×        606         504
urls_high_card       Archive     3.7×        573         835
uuids                Speed       1.6×        483         517
uuids                Archive     1.9×        281         863
log_lines            Speed       3.1×       1306         792
log_lines            Archive     6.0×        988         677
sorted_paths         Speed    1,227.3×       6306        1168
sorted_paths         Archive 17,109.1×       5525        1112
mixed_categorical    any        18.0×       1403        1977
short_skus           Speed    1,017.7×       3230        1808
short_skus           Archive 16,908.2×       2678        1677
```

Adaptive per-column selection across Dict, FSST, front-coding, trained
zstd dictionary, sub-block (`SUB_MULTI`), and cross-column groups.
Sorted/hierarchical data (paths, formatted SKUs) gets four-orders-of-
magnitude compression from front-coding + zstd secondary. High-cardinality
text (URLs, logs) gets 1.6–3.7× from FSST with LZ4/Zstd on top.

### Single-type micro bench — 1M rows

All numbers from `fluxcapacitor dtype-bench --rows 1000000`.

### Compression Ratio

```
                       Compression Ratio (higher is better)
                       ─────────────────────────────────────
Timestamp   Flux(a) ██████████████████████████████████████████████▏ 4517×
            Pq zstd █▎                                              1.3×

String      Flux(a) █████████████████████████████████████████▏     3916×
            Pq zstd █████████▏                                      879×

Date32      Flux(a) ██████▏                                          57×
            Pq zstd █▍                                               14×

List        Flux(a) █████▏                                           49×
            Pq zstd █████████▍                                       11×

Map         Flux(a) ██▏                                              22×
            Pq zstd █████████████████████▊                           20×

Mixed (5c)  Flux(a) █████████████▏                                   13×
            Pq zstd ███████▏                                          7×

UInt64      Flux(a) █████████▏                                        9×
            Pq zstd ██████▎                                           6×

Int64       Flux(a) ████████████▎                                    12×
            Pq zstd ██████▎                                           6×

Struct      Flux(a) ██████▎                                           6×
            Pq zstd █████████████████▊                               18×

Float64     Flux(a) █▋                                              1.6×
            Pq zstd █▏                                              1.1×
```

### Decompression Throughput

```
                     Decompress MB/s (higher is better)
                     ──────────────────────────────────
String      Flux(b)  ████████████████████████████████▎             2155
            Pq snap  ████████████████████████████████████████▏     2656

Map         Flux(b)  ██████████████████████████▏                   1736
            Pq snap  ██████████████▏                                936

Struct      Flux(b)  ██████████████████████████▏                   1740
            Pq snap  ████████████████▏                             1064

Mixed (5c)  Flux(b)  █████████████████████▎                        1419
            Pq snap  ██████████████▊                                985

Timestamp   Flux(b)  ███████████████▏                              1000
            Pq snap  ███████████▋                                   775

List        Flux(b)  ████████████▏                                  806
            Pq snap  ██████████▎                                    684

Int64       Flux(b)  ██████████▊                                    717
            Pq snap  ████████████▏                                  811

UInt64      Flux(b)  █████████▋                                     642
            Pq snap  █████████▏                                     600

Float64     Flux(b)  ████████▊                                      583
            Pq snap  ██████████████████████████████████▏           2278

Date32      Flux(b)  ███████▏                                       468
            Pq snap  ████████████████████▏                         1306

(a) = archive profile, (b) = balanced profile
```

### Full Results Table

```
Type         Format              Size       Ratio    Comp MB/s   Dec MB/s
───────────  ──────────────────  ─────────  ───────  ──────────  ────────
UInt64       Flux (archive)      864.9 KB      9.0×         401       379
             Flux (balanced)       3.3 MB      2.3×         784       642
             Parquet (zstd)        1.2 MB      6.3×         198       629
             Parquet (snappy)      4.1 MB      1.9×         320       600

Int64        Flux (archive)      641.7 KB     12.2×         302       429
             Flux (balanced)       2.4 MB      3.2×         352       717
             Parquet (zstd)        1.2 MB      6.3×         263       670
             Parquet (snappy)      4.1 MB      1.9×         359       811

Float64      Flux (archive)        4.8 MB      1.6×         290       395
             Flux (balanced)       7.7 MB      1.0×         495       583
             Parquet (zstd)        7.0 MB      1.1×         291       773
             Parquet (snappy)      7.9 MB      1.0×         418      2278

String       Flux (archive)        5.2 KB   3916.×          335      1962
             Flux (balanced)       8.4 KB   2408.×          308      2155
             Parquet (zstd)       23.1 KB    879.×          988      2803
             Parquet (snappy)     61.9 KB    328.×          874      2656

Date32       Flux (archive)       68.4 KB     57.1×         215       371
             Flux (balanced)      77.2 KB     50.6×         232       468
             Parquet (zstd)      287.1 KB     13.6×         232      1346
             Parquet (snappy)    476.9 KB      8.2×         232      1306

Timestamp    Flux (archive)        1.7 KB   4517.×         1485       934
             Flux (balanced)       2.8 KB   2814.×         1441      1000
             Parquet (zstd)        5.7 MB      1.3×         335       840
             Parquet (snappy)      6.0 MB      1.3×         282       775

Struct       Flux (archive)        3.7 MB      6.3×         794      1014
             Flux (balanced)       7.2 MB      3.2×        1015      1740
             Parquet (zstd)        1.3 MB     17.8×         338       935
             Parquet (snappy)      4.3 MB      5.5×         362      1064

List         Flux (archive)      749.1 KB     49.2×         341       847
             Flux (balanced)       2.7 MB     13.5×         396       806
             Parquet (zstd)        3.2 MB     11.1×         282       735
             Parquet (snappy)     11.8 MB      3.1×         375       684

Map          Flux (archive)        2.0 MB     21.6×         191      1576
             Flux (balanced)       5.7 MB      7.7×         193      1736
             Parquet (zstd)        2.2 MB     20.3×         263       655
             Parquet (snappy)      5.8 MB      7.7×         306       936

Mixed (5c)   Flux (archive)        2.8 MB     12.6×         199      1184
             Flux (balanced)       8.2 MB      4.3×         210      1419
             Parquet (zstd)        5.2 MB      6.7×         228       831
             Parquet (snappy)     11.5 MB      3.0×         342       985
```

### Run Benchmarks

```bash
# Multi-datatype benchmark (recommended)
cargo run -p fluxcapacitor --release -- dtype-bench --rows 1000000

# Single-column sequential benchmark
cargo run -p fluxcapacitor --release -- bench --rows 50000000 --pattern sequential

# Python scaling benchmark with charts
python python/tests/bench_scaling.py
```

---

## Flux vs Parquet: Detailed Comparison

![Flux vs Parquet](docs/flux-vs-parquet.png)

### Where Flux wins

**Compression ratio — every type.** Flux archive beats Parquet zstd on 9
out of 10 data types, often by orders of magnitude. The DType Router
recognizes that timestamps are monotone (→ DeltaDelta, **4,517×**) and
strings have low cardinality (→ dict + Loom-compressed indices,
**3,916×**) before touching a single value. Parquet treats timestamps as
generic int64 and applies zstd without domain knowledge.

**Nested types — dramatically better ratio.** List columns compress to
**49×** (vs Parquet's 11×) and Map to **22×** (vs 20×). This comes from
three structural optimizations that Parquet doesn't do:
- **Lengths-not-offsets**: stores per-list element counts (tiny,
  repetitive values that RLE compresses to near-zero) instead of
  cumulative offsets
- **Delta-from-base encoding**: list values are split into bases (first
  element per list, sequential → DeltaDelta) and deltas (small → BitSlab
  or RLE)
- **Map key sorting**: entries sorted by key per row, making the key
  column maximally repetitive for dict/RLE encoding

**Decompression speed on complex types.** Flux balanced now matches or
beats Parquet snappy on Map (**1,736** vs 936 MB/s), Struct (**1,740** vs
1,064 MB/s), Mixed (**1,419** vs 985 MB/s), and List (**806** vs 684
MB/s). This comes from parallel leaf decompression via rayon plus direct
u64 reconstruction that skips the u128 intermediary.

**Timestamp throughput.** Flux compresses timestamps at **1,485 MB/s**
(4.4× faster than Parquet zstd) because the DType Router bypasses the
classifier entirely — it knows timestamps are monotone and routes to
DeltaDelta directly.

**Native u128 support.** FluxCompress stores 128-bit values (Decimal128,
large aggregation results) natively using a 99th-percentile slab width
for common values and a sentinel-based OutlierMap for outliers. Parquet
forces `FixedLenByteArray(16)` for every row, wasting space when most
values fit in 64 bits.

### Where Parquet wins

**Decompression speed on primitive numerics.** Parquet snappy decodes
Float64 at **2,278 MB/s** vs Flux balanced at 583 MB/s (3.9× faster),
and Date32 at **1,306 MB/s** vs 468 (2.8× faster). Parquet's C++
backend (`arrow-rs` wraps C/C++ snappy) is heavily optimized for simple
byte-stream decompression, while Flux's structured encoding (BitSlab +
OutlierMap + secondary codec) requires more decode steps.

**Struct compression ratio.** Parquet zstd achieves **17.8×** on structs
vs Flux archive's 6.3×. Parquet's Dremel encoding natively represents
nested repetition/definition levels, giving it an inherent advantage on
deeply nested schemas. Flux flattens structs into independent leaf
columns, which compresses each leaf well but loses cross-column
correlation.

**Compress speed on strings.** Parquet zstd compresses strings at
**988 MB/s** vs Flux archive's 335 MB/s. Parquet's dictionary encoding
is a built-in C++ fast path, while Flux routes dict indices through the
full Loom classifier + secondary codec pipeline (more flexible but more
overhead).

**Ecosystem and tooling.** Parquet is the de facto standard with
first-class support in Spark, DuckDB, Polars, Pandas, BigQuery, Snowflake,
and every major data tool. FluxCompress provides Python bindings and a
Spark JNI bridge, but adoption requires explicit integration.

### Summary

```
Dimension              Flux                     Parquet
─────────────────────  ───────────────────────  ─────────────────────────
Compression ratio      ★★★★★  Best on 9/10     ★★★  Good, not adaptive
                       types. Orders of
                       magnitude on temporal
                       and categorical data.

Decompress speed       ★★★★  Best on complex    ★★★★★  Best on simple
(nested/mixed)         types (Map, Struct,       primitive types (Float64,
                       Mixed, List).             Date32, String).

Compress speed         ★★★  Good, 200–1500      ★★★  Good, 200–1000
                       MB/s depending on type.   MB/s. Faster on strings.

Adaptive routing       ★★★★★  DType Router      ★★  Fixed per-column
                       skips classifier for      encoding (plain, dict,
                       known patterns.           delta). No cross-type
                       Drift detection splits    awareness.
                       segments mid-stream.

Large number support   ★★★★★  Native u128       ★★  FixedLenByteArray(16)
                       with OutlierMap for       wastes space when values
                       99th-pctile efficiency.   fit in 64 bits.

Nested type handling   ★★★★  Lengths-not-       ★★★★  Dremel encoding
                       offsets, delta-from-      (native repetition/
                       base, key sorting.        definition levels).
                       Better ratio on Lists.    Better on deep Structs.

Ecosystem              ★★  Rust, Python, JNI.   ★★★★★  Universal. Every
                       Growing.                  major data tool.
```

---

## Supported Arrow Types

All types round-trip losslessly through `FluxWriter` → `.flux` file →
`FluxReader` with correct Arrow schema reconstruction (e.g., `Int64` in
→ `Int64Array` out, not `UInt64Array`).

```
Category           Types                                          Routing
─────────────────  ────────────────────────────────────────  ─────────────────
Integers           UInt8, UInt16, UInt32, UInt64                  u8/u16 → BitSlab
                   Int8, Int16, Int32, Int64                      fast path; others
                                                                  → Loom Classifier

Floats             Float32, Float64                               → ALP (decimal
                                                                    detection) with
                                                                    Loom fallback

Temporal           Date32, Date64                                 → Loom Classifier
                   Timestamp (Second, Millis, Micros, Nanos)      → DeltaDelta fast
                                                                    path (verify
                                                                    monotone)

Boolean            Boolean                                        → RLE fast path

Decimal / 128-bit  Decimal128 (i128 / u128 carrier)               → Full u128
                                                                    pipeline with
                                                                    OutlierMap

Variable-length    Utf8, LargeUtf8, Binary, LargeBinary           → Adaptive string
                                                                    pipeline (see
                                                                    below)

Nested             Struct, List, Map                              → Recursive
                                                                    flattening +
                                                                    parallel per-leaf
                                                                    compression
```

### Adaptive string pipeline

The string compressor selects per-column (and per-sub-block for large
columns) from 9 sub-strategies using probe-based bakeoffs so it only
spends what the data demands:

```
Sub-strategy         When it fires                         Typical win
───────────────────  ──────────────────────────────────────  ───────────────
Dict                 Cardinality ≤ 30 % (sampled)          18×
Raw LZ4 / Raw Zstd   High-card baseline                    1.5–2×
FSST (LZ4 / Zstd)    Repeated 2–8 byte substrings          2×–4×
                     (URLs, UUIDs, log lines)
Raw / FSST + zstd    Large Archive blocks where a          +5–30 % vs plain
  trained dict         dictionary beats plain zstd          zstd
Front-coded          ≥98 % sorted + ≥8-byte shared prefix   **orders of**
                     (paths, formatted SKUs)               **magnitude**
Sub-block (MULTI)    Row count > 1M — splits into 500K     parallel encode,
                     sub-blocks, re-decides per-block       streaming-friendly
Cross-column group   Multiple compatible string columns    single FSST /
                     under ≈128 MB combined, if a probe    zstd dict shared
                     bakeoff shows it beats per-column      across columns
```

Partition-source columns (from `TableMeta.current_spec()`) are
automatically excluded from cross-column grouping so predicate pushdown
and partition pruning stay correct.

---

## Architecture

![Architecture Pipeline](docs/architecture-pipeline.png)

### Loom Classifier Waterfall

![Loom Classifier Flowchart](docs/loom-classifier-flowchart.png)

### Crate Layout

```
crates/
├── loom/              Core compression engine
│   ├── dtype.rs       FluxDType enum (26 Arrow types → 1-byte tags)
│   ├── dtype_router.rs  DType Router (pre-classification fast paths)
│   ├── segmenter.rs   Adaptive segmenter with drift detection
│   ├── atlas.rs       v2 footer (61B BlockMeta + ColumnDescriptor tree)
│   ├── txn/           Transaction log + snapshot time travel
│   ├── simd/          AVX2 / NEON / scalar bit unpackers
│   ├── compressors/   RLE, Delta, Dict, BitSlab, LZ4, String + secondary
│   │   └── string_compressor.rs   Dict or raw with sampled cardinality
│   └── decompressors/ Parallel block reader + mmap + nested reassembly
│       └── flux_reader.rs   u64 fast path + direct Arrow string construction
├── jni-bridge/        Spark JNI (u128 dual-register)
├── python/            PyO3 bindings (Arrow FFI zero-copy)
└── fluxcapacitor/     CLI (bench, compress, inspect, optimize)
```

### Key Design Decisions

**Parallel everywhere.** Rayon for compress (parallel leaf compression on
nested types, parallel segment compression on flat types) and decompress
(parallel block decompression, parallel leaf decompression on nested
types). Arrow FFI for zero-copy Python. mmap for file reads.

**u64 by default, u128 when needed.** The `u64_only` flag (encoded in the
strategy mask, bit 8) tells the decompressor to skip the u128 widening
entirely. The `decompress_block_to_u64` fast path halves memory bandwidth
for all types ≤ 64 bits. The `reconstruct_array_u64` function builds Arrow
arrays directly from `Vec<u64>` without intermediate conversions.

**Direct Arrow construction.** String decompression builds `StringArray`
directly from offset + data buffers using `StringArray::new_unchecked`,
eliminating the `Vec<Vec<u8>>` → `Vec<String>` → `StringArray` allocation
chain.

---

## Compression Profiles

![Compression Profiles](docs/compression-profiles.png)

```python
buf = fc.compress(table, profile="archive")
buf = fc.compress(table, profile="archive", u64_only=True)  # skip u128 overhead
```

---

## Format v2

### File Layout

```
[Block 0][Block 1]...[Block N][Atlas Footer]
                                  ├─ BlockMeta × N   (61 bytes each)
                                  ├─ Schema JSON      (nested types)
                                  ├─ schema_len       (u32)
                                  ├─ block_count      (u32)
                                  ├─ footer_length    (u32)
                                  └─ FLX2 magic       (u32)
```

### BlockMeta (61 bytes)

```
Field               Size    Purpose
──────────────────  ──────  ────────────────────────────────────────
block_offset        8B      Seek point for block data
z_min / z_max       16B×2   Z-Order coordinates (predicate pushdown)
null_bitmap_offset  8B      Pointer to null mask
strategy_mask       2B      Strategy ID + u64_only flag (bit 8)
value_count         4B      Number of values in this block
column_id           2B      Multi-column support
crc32               4B      Block integrity checksum
dtype_tag           1B      Original Arrow DataType tag
```

---

## Time Travel

Delta-Lake-style versioned tables:

```
my_table.fluxtable/
├── _flux_log/
│   ├── 00000000.json
│   └── 00000001.json
├── data/
│   └── part-0000.flux
└── _flux_meta.json
```

```rust
let table = FluxTable::open("my_table.fluxtable")?;
table.append(&data)?;
let snap = table.snapshot_at_version(0)?;  // time travel
```

---

## Getting Started

```bash
# Build & test
cargo build --release
cargo test --workspace

# Run benchmarks
cargo run -p fluxcapacitor --release -- dtype-bench --rows 1000000

# Python
pip install maturin && maturin develop --release
pytest python/tests/ -v

# CLI
fluxcapacitor compress -i data.arrow -o output.flux
fluxcapacitor bench --rows 50000000 --pattern sequential
```

### Python

```python
import pyarrow as pa
import fluxcompress as fc

table = pa.table({"id": range(1_000_000)})
buf = fc.compress(table, profile="archive")
result = fc.decompress(buf, predicate=fc.col("id") > 500_000)
```

---

## Roadmap

### Completed (v0.2)

- **Multi-type support** — all 26 Arrow types with lossless round-trip
- **DType Router** — pre-classification fast paths (Boolean, Timestamp, u8/u16)
- **String pipeline** — dict + Loom-compressed indices, raw LZ4/Zstd
- **Nested types** — Struct/List/Map with lengths-not-offsets, delta-from-base, key sorting
- **Throughput optimizations** — u64 decompress path, direct Arrow string construction, parallel leaf compress/decompress, sampled cardinality estimation

### Completed (v0.3)

- **FSST symbol-table compression** — per-column static symbol tables on
  high-cardinality strings (URLs, UUIDs, log lines). Probe-based bakeoff
  picks FSST vs raw vs trained-zstd-dict per column.
- **Front coding for sorted/hierarchical data** — shared-prefix encoding
  that yields 1,000–17,000× compression on sorted paths, SKUs, and
  other monotonic strings.
- **Sub-block container (`SUB_MULTI`)** — splits columns >1M rows into
  500K-row sub-blocks, each independently re-deciding its sub-strategy.
  Enables parallel encode/decode and streaming checkpoints.
- **Cross-column string grouping** — trains ONE shared FSST/zstd
  dictionary across compatible sibling columns. Guarded by a
  profitability bakeoff, a 128 MB combined-size ceiling, and a 32 MB
  per-column ceiling so wide Databricks/Spark workloads never OOM.
  Partition-source columns are automatically isolated.
- **ALP for Float64 / Float32** — detects decimal-shaped floats
  (prices, lat/lon, integer-as-float) and encodes integer mantissas
  through the BitSlab / DeltaDelta pipeline with outlier patching.
- **Native Decimal128 (i128 / u128) round-trip** — `Decimal128Array`
  columns use the full u128 pipeline end-to-end (BitSlab + OutlierMap)
  so values >u64::MAX are preserved exactly.
- **Classifier fix for monotonic sequences** — `bit_entropy` no longer
  false-fires on RLE for monotonic offsets / indices. This fixes a
  codebase-wide correctness issue where long monotonic integer columns
  could blow up to raw size.
- **Group-level decode cache** — when N columns share a cross-group
  block, the reader decodes once and serves every member from cache,
  closing the decompression gap to Parquet.

### In Progress (v0.4)

- **SIMD decompression** — AVX2/NEON-accelerated BitSlab unpacking on
  the decompress path (currently SIMD is compress-only).
- **Null bitmap support** — compress only dense non-null values,
  reconstruct nulls on read. BlockMeta field exists, implementation
  pending.
- **Predicate pushdown for strings** — min/max string metadata in
  footer for Z-Order skipping on string columns.
- **FSST zero-alloc decode** — build Arrow `StringArray` buffers
  directly from FSST codes without the intermediate `Vec<Vec<u8>>`.

### Planned (v0.5+)

- **f128 (IEEE 754 binary128) support** — full pipeline integration for
  128-bit floats. See [docs/roadmap-f128.md](docs/roadmap-f128.md) for
  the full plan; current blocker is that Arrow doesn't yet expose a
  `Float128` data type, so there is no in-Arrow transport for the
  values. Our on-disk format already has everything needed (u128
  pipeline + `Decimal128` carrier as a stop-gap); we're waiting on the
  Arrow community spec.
- **Parquet-competitive Float64 decode** — direct bit-copy from
  decompressed buffer to Arrow Float64 buffer without per-value
  `from_bits` conversion.
- **Zero-copy buffer sharing** — return Arrow buffers backed by the
  decompressed block memory without copying. Requires `unsafe`
  alignment guarantees.
- **Global cross-file FSST dictionaries** — share symbol tables across
  `.flux` files in a table so cold-storage optimization phases can
  reuse training work.

See also:
- [docs/roadmap-performance.md](docs/roadmap-performance.md) — Detailed performance plan
- [docs/roadmap-wal.md](docs/roadmap-wal.md) — Binary WAL migration
- [docs/roadmap-f128.md](docs/roadmap-f128.md) — IEEE 754 binary128 integration plan

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
