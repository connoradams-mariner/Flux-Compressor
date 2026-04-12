# FluxCompress Documentation

**High-performance adaptive columnar compression for Python, Rust, and the JVM.**

FluxCompress is an Apache 2.0 columnar storage format that outperforms Parquet
for numeric workloads by using an adaptive *Loom* engine that selects the best
compression strategy per 1 024-row segment — no configuration required.

---

## Quick Links

| | |
|---|---|
| [Installation](installation.md) | `pip install fluxcompress` |
| [Python API Reference](api/python.md) | `compress`, `decompress`, `col`, `inspect` |
| [Polars Integration](guides/polars.md) | `compress_polars`, `decompress_polars` |
| [Pandas Integration](guides/pandas.md) | `compress_df`, `decompress_df`, `compression_stats` |
| [Apache Spark Integration](guides/spark.md) | `register_udfs`, `compress_dataframe` |
| [File Format](format/flux_format.md) | BitSlab, OutlierMap, Atlas footer |
| [CLI Reference](cli/fluxcapacitor.md) | `fluxcapacitor compress/optimize/inspect` |
| [CHANGELOG](../CHANGELOG.md) | Version history |

---

## 30-Second Tour

```python
import pyarrow as pa
import fluxcompress as fc

# Any Arrow-compatible table (PyArrow, Polars, pandas)
table = pa.table({"id": range(1_000_000), "revenue": range(1_000_000)})

# Compress — Loom automatically picks the best strategy per column segment
buf = fc.compress(table)
print(buf)  # FluxBuffer(182304 bytes)  ← ~4.4× smaller than raw u64

# Decompress
table2 = fc.decompress(buf)

# Predicate pushdown — skips blocks that can't contain matching rows
table3 = fc.decompress(buf, predicate=fc.col("id") > 800_000)

# Save / load
fc.write_flux(buf, "data.flux")
buf2 = fc.read_flux("data.flux")

# Inspect Atlas footer
info = fc.inspect(buf)
print(f"{info.num_blocks} blocks")
for block in info.blocks[:3]:
    print(f"  {block.strategy:<12} z=[{block.z_min}, {block.z_max}]")
```

---

## Why FluxCompress?

| Feature | Parquet | FluxCompress |
|---|---|---|
| Adaptive per-segment strategy | ✗ | ✅ Loom classifier |
| u128 precision (aggregations) | ✗ | ✅ OutlierMap patching |
| Predicate pushdown | ✅ | ✅ Z-Order Atlas |
| SIMD bit-unpacking | Partial | ✅ AVX2 + NEON |
| Zero-copy JVM bridge | ✗ | ✅ DirectByteBuffer |
| Python/Polars/pandas | ✅ | ✅ |
| Cold two-pass optimizer | ✗ | ✅ `fluxcapacitor optimize` |

---

## Architecture

![Architecture Pipeline](architecture-pipeline.png)

### Loom Classifier Waterfall

![Loom Classifier Flowchart](loom-classifier-flowchart.png)

### Flux vs Parquet

![Flux vs Parquet](flux-vs-parquet.png)
