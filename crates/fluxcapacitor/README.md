# fluxcapacitor

[![crates.io](https://img.shields.io/crates/v/fluxcapacitor.svg)](https://crates.io/crates/fluxcapacitor)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)

**Command-line tool for [FluxCompress](https://github.com/connoradams-mariner/Flux-Compressor)** — compress, decompress, inspect, and benchmark `.flux` files from the terminal.

## Install

```bash
cargo install fluxcapacitor
```

Or build from the monorepo:

```bash
cargo build --release -p fluxcapacitor
./target/release/fluxcapacitor --help
```

## Commands

```text
fluxcapacitor compress      -i data.parquet -o data.flux
fluxcapacitor decompress    -i data.flux    -o data.arrow
fluxcapacitor inspect       data.flux                         # print Atlas footer
fluxcapacitor optimize      -i hot/         -o cold.flux      # two-pass global-dict + Z-order
fluxcapacitor merge         a.flux b.flux c.flux -o merged.flux
```

## Benchmarks

Built-in benchmark subcommands that match the numbers in the project README:

```text
fluxcapacitor bench                 --rows 50000000 --pattern sequential
fluxcapacitor dtype-bench           --rows 1000000                  # per-type Flux vs Parquet
fluxcapacitor string-bench          --rows 10000000                 # string patterns
fluxcapacitor mixed-bench           --rows 9950000                  # Databricks-shaped
fluxcapacitor compare-bench         --rows 5000000                  # Flux vs Parquet vs Delta Lake
fluxcapacitor string-compare-bench  --rows 10000000                 # string-heavy 3-way
fluxcapacitor float-compare-bench   --rows 10000000                 # float-heavy 3-way
```

## Example output — `compare-bench`

```
Codec                   Size        Ratio    Comp MB/s   Dec MB/s
─────────────────────  ──────────   ───────  ──────────  ────────
Flux (Archive)         147.4 MB     9.74×       624        1369
Parquet (zstd-3)       225.3 MB     6.37×       357        1151
Delta Lake (zstd-3)    225.3 MB     6.37×       357         834
```

## Dependencies

Wraps [`flux-loom`](https://crates.io/crates/flux-loom) (engine) + [`parquet`](https://crates.io/crates/parquet) + [`orc-rust`](https://crates.io/crates/orc-rust) + [`polars`](https://crates.io/crates/polars) for Arrow-adjacent I/O.

## See also

- **[flux-loom](https://crates.io/crates/flux-loom)** — the compression engine this CLI drives.
- **[Flux-Compressor](https://github.com/connoradams-mariner/Flux-Compressor)** — main project README with full benchmark tables.

## License

Apache 2.0.
