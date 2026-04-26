# Changelog

All notable changes to FluxCompress are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.6.0](https://github.com/connoradams-mariner/Flux-Compressor/compare/flux-compressor-v0.5.4...flux-compressor-v0.6.0) (2026-04-26)


### Features

* complete FluxCompress codebase (all 5 sprints) ([d224d0e](https://github.com/connoradams-mariner/Flux-Compressor/commit/d224d0e42406bd8ba69abb1d17415f4de8aae055))
* concurrent writers ([f38d453](https://github.com/connoradams-mariner/Flux-Compressor/commit/f38d453a65cc2e4717f78452a147ea241d4e0779))
* improve high cardinality string compression ([a0a719c](https://github.com/connoradams-mariner/Flux-Compressor/commit/a0a719cf1db33f2f75dfcf5aaf94011a7c58d2fb))


### Bug Fixes

* chunk compression and mmap decompress in python. feat: implement hidden partitioning and liquid clustering. ([3eb5680](https://github.com/connoradams-mariner/Flux-Compressor/commit/3eb5680eb4096e66593051eeb7b386f4834c5bc0))
* **ci:** fix cargo-workspace limitation ([776cc9e](https://github.com/connoradams-mariner/Flux-Compressor/commit/776cc9e1319288268677c701ae8ce9cafd6c08ae))
* **ci:** remove per-crate cargo-dist metadata blocks ([434b878](https://github.com/connoradams-mariner/Flux-Compressor/commit/434b8785d2da6dd3ff2ecaf68ebc3114848eb78a))
* **ci:** remove per-crate cargo-dist metadata blocks ([a91671a](https://github.com/connoradams-mariner/Flux-Compressor/commit/a91671a3aafcf2f0d3b272a9af77bd7c7c410082))
* fluxtable streaming ([3127195](https://github.com/connoradams-mariner/Flux-Compressor/commit/31271953d91a0e61fcecd66454f13ab0dc7ba672))
* high cardinality string compression. combining cols of like types for compression ([ec7a5e8](https://github.com/connoradams-mariner/Flux-Compressor/commit/ec7a5e877252fd3d2e5d23df3ee4608e167c0737))
* polars LargeUint8 type issue ([903a435](https://github.com/connoradams-mariner/Flux-Compressor/commit/903a435f59c41a3eddfdcb946cdaf3af1183e4f8))
* release-type=simple ([57886a7](https://github.com/connoradams-mariner/Flux-Compressor/commit/57886a7a402f0603d959b5bda90db59b6c11ff17))


### Documentation

* **databricks:** document serverless compute support; add compute-tier matrix ([7ebb0d7](https://github.com/connoradams-mariner/Flux-Compressor/commit/7ebb0d7a14c28519728f57c8b7747642ec7787fa))

## [Unreleased]

(No unreleased changes — see `v0.3.0` below.)

---

## [0.3.0] — 2026-04-18

This release is the **adaptive-string + real-numeric** overhaul. Headline
result on a 22-column 9.95M-row Databricks-shaped workload:
Flux Archive **302 MB (9.47×)** vs Parquet zstd-3 **448 MB (6.37×)** at
parity decompression throughput.

### Added

#### String compression (`crates/loom/src/compressors/string_compressor.rs`)
- **FSST symbol-table compressor** — new sub-strategies `SUB_FSST_LZ4`
  (0x09) and `SUB_FSST_ZSTD` (0x0A). Per-column symbol-table training
  from a sampled corpus; bulk and rayon-parallel encode paths.
- **Trained Zstd dictionary** — `SUB_RAW_ZSTD_DICT` (0x0B) and
  `SUB_FSST_ZSTD_DICT` (0x0C). Archive-profile only. Trained from a
  bounded sample and bake-off-gated.
- **Front coding** — `SUB_FRONT_CODED` (0x0D). Varint-encoded
  shared-prefix / suffix pairs, LZ4 or Zstd post-pass. Orders-of-
  magnitude wins on sorted paths / SKUs.
- **Sub-block container** — `SUB_MULTI` (0x0F) splits columns >1M rows
  into 500K-row chunks, re-decides sub-strategy per chunk, rayon-parallel
  encode and decode. Doubles as a streaming checkpoint.
- **Cross-column grouping** — `SUB_CROSS_GROUP` (0x0E) and new public
  functions `compress_cross_column_group_with_profile` +
  `decompress_cross_column_group`. Shared FSST / zstd dictionary across
  compatible sibling columns. Guarded by
  `GROUP_MAX_COMBINED_BYTES = 128 MB`, `GROUP_PER_COLUMN_MAX_BYTES = 32 MB`,
  and a probe-based profitability bakeoff. Partition-source columns
  auto-isolated via `FluxWriter::isolated_set()`.
- **Offsets through the Loom pipeline** — all current raw/FSST paths
  compress the offset column via `compress_index_column` (DeltaDelta +
  BitSlab + FOR), replacing the old LZ4-on-raw-bytes layout.
  `SUB_RAW_LZ4_LEGACY` / `SUB_RAW_ZSTD_LEGACY` are preserved read-only
  for backwards compatibility.
- `StringGroupingMode { Off, Auto, Manual(Vec<Vec<String>>) }` plus
  `FluxWriter::with_string_grouping`, `with_isolated_string_columns`,
  and `with_partition_spec` configuration knobs.
- `decompress_to_arrow_string_for_column(data, dtype_tag, Some(column_id))`
  for column-id-aware cross-group decoding.

#### Numeric compression
- **ALP for Float64 / Float32** — new `alp_compressor` module
  (TAG = 0x09). Detects decimal-shaped floats (prices, lat/lon,
  integer-as-float), encodes integer mantissas through the Loom pipeline
  with a sparse outlier map. Bit-exact round-trip check via
  `(m as f64) / scale` for numerical stability.
- **Native Decimal128 (i128 / u128) round-trip** —
  `compress_decimal128_column` in `FluxWriter` extracts i128 values
  directly from `Decimal128Array` and feeds the full u128 pipeline.
  `LeafData::Numeric128(Vec<u128>)` reader variant and
  `reconstruct_decimal128` build a proper `Decimal128Array` out.

#### Reader / streaming
- **Group-level decode cache** in `decompress_with_schema_projected`:
  when N columns share a cross-group block offset, the inner payload is
  decoded ONCE in parallel and all member columns served from cache.
  Closes the decompression throughput gap to Parquet on wide schemas.
- Mixed-schema benchmark binary `fluxcapacitor mixed-bench` mirroring
  the real-world 22-column Databricks workload.
- String compression benchmark binary `fluxcapacitor string-bench` with
  6 representative patterns at 10M rows.

#### Python pandas support (carried from Unreleased)
- `fluxcompress.pandas` module with `compress_df`, `decompress_df`,
  `compress_series`, `decompress_series`, `compress_column`, `round_trip`,
  and `compression_stats`.
- `python/tests/conftest.py` shared session-scoped fixtures.
- `python/tests/test_pandas.py` — full pandas integration test suite.

### Fixed

- **Classifier bug on monotonic sequences** — `bit_entropy` bins on the
  top 8 occupied bits, which meant long monotonic offset / index
  sequences (all in the same top-byte bucket) looked like constants
  and were mis-routed to RLE. RLE on non-constant data expanded output
  catastrophically. Added an endpoint-equality guard so RLE only fires
  when `first == mid == last`. This silently improves every use of the
  Loom pipeline on structured data.

### Changed

- String block layout bumped: raw-path blocks now emit
  `SUB_RAW_LZ4 = 0x07` / `SUB_RAW_ZSTD = 0x08` with Loom-pipeline
  offsets. Legacy 0x01 / 0x02 codes stay readable.
- `decompress_to_arrow_string` now preserves its previous single-argument
  signature but delegates to `decompress_to_arrow_string_for_column` with
  `None` to handle both grouped and non-grouped blocks uniformly.

### Known limitations

- **f128 (IEEE 754 binary128)** — not yet supported in-Arrow because
  `arrow_schema::DataType` has no `Float128` variant. The on-disk
  pipeline can already carry 128-bit bit-patterns via `Decimal128`; full
  f128 plumbing is tracked in [docs/roadmap-f128.md](docs/roadmap-f128.md).

---

## [0.1.0] — 2024-01-01

### Added

#### Core Rust Library (`crates/loom`)
- **Sprint 1** — `BitWriter` and `BitReader`: packed variable-bit-width
  bitstream encoder/decoder supporting 1–64 bits per value.
- `discover_width()`: 99th-percentile bit-width heuristic with Frame-of-Reference
  subtraction.
- **Sprint 2** — `OutlierMap`: full-precision `u128` exception buffer.
  `encode_with_outlier_map` / `decode_with_outlier_map` round-trip helpers.
  Zero-copy `OutlierMapReader` for the SIMD decode lane.
- **Loom Adaptive Classifier** — deterministic 5-step waterfall:
  RLE → DeltaDelta → Dictionary → BitSlab → SIMD-LZ4.
  Classifies 1 024-row segments in microseconds using entropy, linearity,
  cardinality, and bit-width heuristics.
- **Sprint 3** — SIMD Unpacker:
  - `simd/avx2.rs`: AVX2 256-bit unpacker for 8/16/32-bit values via
    `_mm256_shuffle_epi8`.
  - `simd/neon.rs`: AArch64 NEON unpacker via `vqtbl1q_u8` / `vmovl` chains.
  - `simd/scalar.rs`: portable fallback (correctness reference).
  - Runtime dispatch: AVX2 → NEON → scalar, selected at compile time.
- **Sprint 4** — `AtlasFooter`: seekable metadata trailer (50 bytes per block:
  `block_offset`, `z_min`, `z_max`, `null_bitmap_offset`, `strategy_mask`).
  `z_order_encode` / `z_order_decode` Morton-code utilities.
  Predicate-pushdown `candidate_blocks()` using `[z_min, z_max]` hyper-rectangles.
- **Compressors**: `BitSlab`, `RLE`, `DeltaDelta`, `Dictionary`, `SIMD-LZ4`.
  `FluxWriter`: top-level `LoomCompressor` trait implementation.
  `FluxReader`: top-level `LoomDecompressor` with predicate pushdown.
- **Sprint 5** — `fluxcapacitor` CLI:
  `compress`, `decompress`, `optimize` (two-pass cold optimizer),
  `merge`, `inspect`, `bench` subcommands.
- **Sprint 5** — JNI bridge (`crates/jni-bridge`):
  `compressLongColumn`, `compressU128Column`, `decompressToLongArray`,
  `decompressU128Column`, `compressDirectBuffer` (zero-copy off-heap).
  `u128_bridge.rs`: `jlong_pair_to_u128` / `u128_to_jlong_pair`.
  `FluxNative.java` companion class.
- Criterion benchmark suite (`crates/loom/benches/bit_slab.rs`) covering
  7 benchmark groups: BitWriter, BitReader, Classifier, Compressors,
  Decompressors, OutlierMap, and full pipeline.

#### Python Package (`crates/python` + `python/`)
- PyO3 extension module (`_fluxcompress`):
  `FluxBuffer`, `Predicate`, `Column`, `BlockInfo`, `FileInfo` classes.
  `compress()`, `decompress()`, `inspect()`, `col()`, `read_flux()`,
  `write_flux()` module-level functions.
  Arrow C Data Interface bridge: zero-copy for PyArrow and Polars.
- `fluxcompress._polars`: `compress_polars()` / `decompress_polars()`.
- `fluxcompress.spark`: `compress_dataframe()`, `decompress_dataframe()`,
  `register_udfs()`, `compress_partition_indexed()`, pandas UDF helpers.
- Type stubs (`_fluxcompress.pyi`) for full IDE and mypy/pyright support.
- `py.typed` marker (PEP 561).
- `pyproject.toml` with maturin build config, PyPI metadata, optional
  dependency groups: `polars`, `pandas`, `spark`, `dev`, `all`.
- CI/CD: GitHub Actions workflow building manylinux, macOS (x86-64 + arm64),
  and Windows wheels; OIDC trusted publishing to PyPI on tag push.

#### Examples
- `examples/integration_demo.rs`: 5-scenario Rust end-to-end demo.
- `examples/polars_example.py`: Polars compress/decompress/pushdown/optimize.
- `examples/spark_example.py`: PySpark pandas UDFs + Scala UDF reference.

---

## Version Policy

- `MAJOR` — breaking changes to the `.flux` file format or public API.
- `MINOR` — new features, new compression strategies, new integrations.
- `PATCH` — bug fixes, performance improvements, dependency updates.

[Unreleased]: https://github.com/connoradams-mariner/Flux-Compressor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/connoradams-mariner/Flux-Compressor/releases/tag/v0.1.0
