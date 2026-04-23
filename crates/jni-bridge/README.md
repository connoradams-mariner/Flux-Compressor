# flux-jni-bridge

[![crates.io](https://img.shields.io/crates/v/flux-jni-bridge.svg)](https://crates.io/crates/flux-jni-bridge)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)

**JNI bridge exposing [FluxCompress](https://github.com/connoradams-mariner/Flux-Compressor) to the JVM / Spark ecosystem.**

This crate is the Rust side of the `com.datamariners.fluxcompress:flux-spark-connector_2.12` DataSource V2 connector. It compiles as a `cdylib` (`libflux_jni.so` / `.dylib` / `.dll`) and exports `extern "system"` functions that the Scala / Java side resolves via `System.loadLibrary("flux_jni")`.

## What it exposes

Three layers of JNI surface, in increasing sophistication:

| Layer | Methods | Use                                                                 |
|-------|---------|---------------------------------------------------------------------|
| 1 — single-column u64       | `compress`, `decompress`                      | Zero-copy via DirectByteBuffer. Used by low-level benchmarks.       |
| 1 — single-column u128      | `compressU128`, `decompressU128`              | Two-`long[]` dual-register representation for `Decimal128` columns. |
| 2 — multi-column Arrow IPC  | `compressTable`, `decompressTable`            | Full `RecordBatch` round-trip via Arrow IPC stream bytes.           |
| 3 — FluxTable handle API    | `tableOpen`, `tableAppend`, `tableScan`, `tableEvolve`, `tableClose`, `tableCurrentVersion` | Stateful table handles (Delta-Lake-style versioned directory).      |

## Building

```bash
cargo build --release -p flux-jni-bridge
# Produces: target/release/libflux_jni.{so,dylib,dll}
```

## Using from Scala / Java

```java
package com.datamariners.fluxcompress;

public final class FluxNative {
    static { System.loadLibrary("flux_jni"); }
    public static native byte[] compressTable(byte[] ipcBatch, String profile);
    public static native byte[] decompressTable(byte[] fluxData);
    public static native long   tableOpen(String path);
    public static native long   tableAppend(long handle, byte[] fluxData);
    public static native byte[] tableScan(long handle);
    public static native long   tableEvolve(long handle, String schemaJson);
    public static native void   tableClose(long handle);
    // ...
}
```

The Spark connector's `FluxNativeLoader` auto-extracts the right `libflux_jni` for the current OS + arch from the JAR's `/native/<os>-<arch>/` resource path, so end users typically don't invoke this crate directly — they pull the Maven coordinate `com.datamariners.fluxcompress:flux-spark-connector_2.12`.

## Dependency

```toml
[dependencies]
flux-jni-bridge = "0.5.3"
```

Internally depends on [`flux-loom`](https://crates.io/crates/flux-loom) for the compression engine.

## See also

- **[flux-loom](https://crates.io/crates/flux-loom)** — the core compression engine this crate wraps.
- **[Flux-Compressor](https://github.com/connoradams-mariner/Flux-Compressor)** — the monorepo (Rust + Python + Scala/Spark).
- **[docs/databricks.md](https://github.com/connoradams-mariner/Flux-Compressor/blob/main/docs/databricks.md)** — PySpark / Databricks quickstart.

## License

Apache 2.0.
