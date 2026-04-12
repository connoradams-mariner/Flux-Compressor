# Polars Integration Guide

FluxCompress integrates with Polars via the **Arrow C Data Interface** — a
zero-copy memory protocol that Polars and PyArrow both implement.  No data is
copied between the two libraries when compressing or decompressing numeric
columns.

## Setup

```bash
pip install fluxcompress[polars]
```

## Compress a Polars DataFrame

```python
import polars as pl
import fluxcompress as fc

df = pl.DataFrame({
    "user_id":  list(range(1_000_000)),
    "revenue":  [i * 37 % 99_999 for i in range(1_000_000)],
    "region":   [i % 8 for i in range(1_000_000)],
})

buf = fc.compress_polars(df)
print(buf)          # FluxBuffer(... bytes)
print(len(buf))     # compressed size
```

`compress_polars()` calls `df.to_arrow()` internally — a zero-copy operation
for most numeric column types — then passes the Arrow table to the Rust engine.

## Decompress to a Polars DataFrame

```python
df2 = fc.decompress_polars(buf)
assert df["user_id"].to_list() == df2["value"].to_list()
```

## Predicate Pushdown

The Atlas footer lets FluxCompress skip blocks that can't contain rows
satisfying a predicate — no wasted decompression.

```python
# Only decompress blocks that may contain user_id > 800_000
df_filtered = fc.decompress_polars(
    buf,
    predicate=fc.col("user_id") > 800_000,
)

# Range predicate
df_range = fc.decompress_polars(
    buf,
    predicate=fc.col("revenue").between(50_000, 99_999),
)

# Compound predicate
df_complex = fc.decompress_polars(
    buf,
    predicate=(fc.col("user_id") > 500_000) & (fc.col("region") < 4),
)
```

## Strategy Override

```python
# Force DeltaDelta for a known-sequential column
buf = fc.compress_polars(df, strategy="delta")

# Force Dictionary for a known-low-cardinality column
df_enum = pl.DataFrame({"status": [i % 5 for i in range(1_000_000)]})
buf = fc.compress_polars(df_enum, strategy="dict")
```

Available strategies: `"auto"` (default), `"rle"`, `"delta"`, `"dict"`,
`"bitslab"`, `"lz4"`.

## Save and Load

```python
buf = fc.compress_polars(df)
buf.save("users.flux")

buf2 = fc.read_flux("users.flux")
df2  = fc.decompress_polars(buf2)
```

## Inspect the Atlas Footer

```python
info = fc.inspect(buf)
print(f"Blocks: {info.num_blocks}")
for block in info.blocks:
    print(f"  [{block.z_min}, {block.z_max}]  strategy={block.strategy}")
```

## LazyFrame Pattern

Polars LazyFrames are evaluated lazily — compress at the `collect()` boundary
and re-wrap as a LazyFrame afterward for further transformations:

```python
lf = (
    pl.scan_parquet("large_dataset/*.parquet")
    .filter(pl.col("revenue") > 0)
    .with_columns(pl.col("user_id").cast(pl.UInt64))
)

# Collect → compress
df = lf.collect()
buf = fc.compress_polars(df)
buf.save("filtered.flux")

# Later: decompress → LazyFrame
df_back = fc.decompress_polars(fc.read_flux("filtered.flux"))
lf_continued = df_back.lazy().group_by("region").agg(pl.col("revenue").sum())
result = lf_continued.collect()
```

## Compression Ratio Reference

Typical ratios on a 1M-row Polars DataFrame (measured on an M2 MacBook):

| Column Pattern | Strategy | Ratio | Throughput |
|---|---|---|---|
| Sequential integers | DeltaDelta | ~50× | ~2 GB/s |
| Low-cardinality enum (8 values) | Dictionary | ~3× | ~1.5 GB/s |
| Random u64 | BitSlab | ~1.5× | ~1.2 GB/s |
| Constant column | RLE | ~1000× | ~3 GB/s |
| Mixed (99% small, 1% u64::MAX) | BitSlab + OutlierMap | ~8× | ~1.1 GB/s |
