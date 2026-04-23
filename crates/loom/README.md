# flux-loom

[![crates.io](https://img.shields.io/crates/v/flux-loom.svg)](https://crates.io/crates/flux-loom)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)

**The core compression engine for [FluxCompress](https://github.com/connoradams-mariner/Flux-Compressor)** — a dtype-aware, probe-gated, adaptive columnar storage format that beats Parquet on compression ratio across every Arrow type and now matches or exceeds Parquet on decompression throughput for nested / mixed / float-heavy workloads.

> **Published as `flux-loom` because `loom` is already taken on crates.io.** Inside a workspace you can keep the short `loom` alias: `loom = { package = "flux-loom", version = "0.5.4" }`.

## What it does

`flux-loom` is the Rust library that reads and writes the `.flux` file format end-to-end:

- **Writer** (`compressors::flux_writer::FluxWriter`) — takes an Arrow `RecordBatch` and produces `.flux` bytes via the Loom classifier + DType router + per-column probe-gated strategy selection (BitSlab, DeltaDelta, RLE, Dict, ALP, FSST, front-coding, sub-block MULTI, cross-column FSST groups).
- **Reader** (`decompressors::flux_reader::FluxReader`) — parses the Atlas footer, applies Z-order and string-prefix predicate pushdown, decompresses blocks in parallel, and reconstructs a typed Arrow `RecordBatch` with zero-copy same-width buffer sharing on `Float64 / Int64 / UInt64 / Date64 / Timestamp(*, None)`.
- **FluxTable** (`txn::FluxTable`) — Delta-Lake-style versioned directory with JSON or binary-WAL transaction log, schema evolution (`evolve_schema`), time travel (`snapshot_at_version`), and row-level mutations (`delete_where`, `update_where`, `merge`).

## Quick start

```toml
[dependencies]
# Rename to `loom` locally so `use loom::*` stays idiomatic; `flux-loom`
# is the published name.
loom = { package = "flux-loom", version = "0.5.4" }
arrow-array = "52"
arrow-schema = "52"
```

```rust
use loom::{
    CompressionProfile,
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    traits::{LoomCompressor, LoomDecompressor},
};
use std::sync::Arc;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

let schema = Arc::new(Schema::new(vec![
    Field::new("id", DataType::Int64, false),
]));
let batch = RecordBatch::try_new(
    schema,
    vec![Arc::new(Int64Array::from((0i64..1_000_000).collect::<Vec<_>>()))],
)?;

let bytes = FluxWriter::with_profile(CompressionProfile::Archive).compress(&batch)?;
let back  = FluxReader::default().decompress_all(&bytes)?;
assert_eq!(back.num_rows(), 1_000_000);
```

## Compression profiles

| Profile     | Description                                                     |
|-------------|-----------------------------------------------------------------|
| `Speed`     | No secondary codec. Fastest encode + decode.                    |
| `Balanced`  | LZ4 secondary pass. Near-Speed decode; meaningful size savings. |
| `Archive`   | Zstd secondary pass. Best ratio; ~30–50 % slower encode.        |
| `Brotli`    | Brotli on strings + Zstd on numerics. Biggest wins on text.     |

## Benchmarks (v0.5.4)

On a realistic 13-column float-heavy schema, 10 M rows, `flux-loom` at Archive:
- **30.1 % smaller** than Parquet zstd-3 and Delta Lake.
- **3.5× faster compress** (718 vs 204 MB/s).
- **44 % faster decompress** (1,131 vs 785 MB/s).

Full comparison tables at [github.com/connoradams-mariner/Flux-Compressor#benchmarks](https://github.com/connoradams-mariner/Flux-Compressor#benchmarks).

## Crate layout

- `compressors/` — per-strategy encoders.
- `decompressors/` — block reader + top-level `FluxReader`.
- `atlas.rs` — 61-byte per-block metadata footer format.
- `traits.rs` — `LoomCompressor`, `LoomDecompressor`, `Predicate` (with row-level `eval_on_batch` + `may_overlap` block-skip).
- `txn/` — FluxTable, schema evolution, mutations (DELETE / UPDATE / MERGE via COW), binary WAL + checkpoints.
- `null_aware.rs`, `null_bitmap.rs` — end-to-end null preservation.
- `string_zero_alloc.rs` — zero-allocation `StringArray` construction primitive.

## See also

- **[fluxcompress](https://pypi.org/project/fluxcompress/)** — Python bindings (PyO3 / Arrow FFI).
- **[flux-jni-bridge](https://crates.io/crates/flux-jni-bridge)** — JVM bridge used by the Spark DataSource V2 connector.
- **[com.datamariners.fluxcompress:flux-spark-connector_2.12](https://central.sonatype.com/artifact/com.datamariners.fluxcompress/flux-spark-connector_2.12)** — Spark DSv2 connector for `df.write.format("flux")`.
- **[fluxcapacitor](https://crates.io/crates/fluxcapacitor)** — CLI: `fluxcapacitor compress | decompress | inspect | bench`.

## License

Apache 2.0.
