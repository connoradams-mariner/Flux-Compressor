# Changelog

All notable changes to FluxCompress are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `fluxcompress.pandas` module with `compress_df`, `decompress_df`,
  `compress_series`, `decompress_series`, `compress_column`, `round_trip`,
  and `compression_stats`.
- `python/tests/conftest.py` with shared session-scoped fixtures for all
  test modules.
- `python/tests/test_pandas.py` — full pandas integration test suite.
- **Multi-format CLI I/O** — `fluxcapacitor compress` / `decompress` now
  auto-detect the input/output format from the file extension and
  support CSV, TSV, JSON, NDJSON/JSONL, Parquet, Arrow IPC/Feather, ORC,
  and Excel (`.xlsx` read+write; `.xls`/`.xlsm`/`.ods` read-only).
  Implementation in `crates/fluxcapacitor/src/formats.rs`, with
  integration tests in `crates/fluxcapacitor/tests/formats_round_trip.rs`
  and a Criterion benchmark in
  `crates/fluxcapacitor/benches/file_formats.rs`.
- **Reference Spark V2 connector** — `java/io/fluxcompress/spark/`
  registers the `flux` short name and supports every common save mode:
  - `FluxTable` advertises `BATCH_WRITE` + `TRUNCATE` +
    `OVERWRITE_BY_FILTER` capabilities.
  - `FluxWriteBuilder` implements `SupportsTruncate` and
    `SupportsOverwriteV2` so `mode("overwrite")` and predicate-based
    `replaceWhere` work correctly.
  - `FluxTableProvider` implements `SupportsCatalogOptions`.
  - `FluxCatalog` (`TableCatalog` + `SupportsNamespaces`) lets admins
    register the connector as a Unity Catalog catalog via
    `spark.sql.catalog.flux=io.fluxcompress.spark.FluxCatalog`.
- **Connector tests** — `FluxConnectorTest.java` (JUnit 5, no Spark
  required) covers the capability matrix, truncate-flag propagation,
  catalog table lifecycle, and `FluxBatchWrite` commit semantics.
  `python/tests/spark_uc_smoke_test.py` is the cluster-side integration
  test that exercises CREATE / AppendData / SupportsTruncate /
  SupportsOverwriteV2 / DROP through the registered catalog.

### Fixed
- **Float64 / wide-slab BitSlab corruption.** `BitWriter` /
  `BitReader::read_value` / `simd::scalar::unpack_scalar` truncated
  values whenever `slab_width + bit_off > 64` because they used a `u64`
  shift window. Float64 columns (typical slab width 62–63) hit this
  every time and round-tripped to denormals. Fixed by keeping the u64
  fast path for `width + bit_off ≤ 64` and splitting the rare wide-slab
  case across an 8-byte boundary (writer) / using a `u128` 9-byte window
  (reader). Regression tests in
  `crates/loom/src/bit_io.rs::tests::round_trip_wide_widths_all_alignments`
  and `round_trip_float64_bit_pattern`.
- **Multi-column null-row alignment.** `extract_column_data` previously
  used `filter_map` to drop null rows, so columns with different null
  counts ended up with different lengths and tripped Spark's
  `Invalid argument error: all columns in a record batch must have the
  same length` on decompress. Now reads the underlying value buffer
  directly so every column produces exactly `arr.len()` u64 slots, with
  `0` placeholders for null cells. Regression test in
  `crates/fluxcapacitor/tests/formats_round_trip.rs::compress_then_decompress_multicol_with_mixed_nulls`.

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
