# Pandas Integration Guide

`fluxcompress.pandas` provides DataFrame and Series helpers that sit on top
of the core PyArrow bridge.  pandas is an optional dependency.

## Setup

```bash
pip install fluxcompress[pandas]
```

## Compress a DataFrame

```python
import pandas as pd
import fluxcompress.pandas as fcp

df = pd.DataFrame({
    "user_id": range(1_000_000),
    "revenue": [i * 37 % 99_999 for i in range(1_000_000)],
    "region":  [i % 8 for i in range(1_000_000)],
})

buf = fcp.compress_df(df)
print(len(buf))  # compressed size in bytes
```

Only numeric columns (`int*`, `uint*`, `float*`) are compressed by default.
Pass `columns=` to select a specific subset:

```python
buf = fcp.compress_df(df, columns=["user_id", "revenue"])
```

## Decompress to a DataFrame

```python
df2 = fcp.decompress_df(buf)
assert list(df["user_id"]) == list(df2["value"])
```

## Series Helpers

```python
s   = pd.Series(range(1_000_000), name="user_id")
buf = fcp.compress_series(s)
s2  = fcp.decompress_series(buf, name="user_id")
assert list(s) == list(s2)
```

## Predicate Pushdown

```python
import fluxcompress as fc

df_filtered = fcp.decompress_df(
    buf,
    predicate=fc.col("value") > 800_000,
)
```

## Chunked Column Compression

`compress_column` adds a new `BinaryType` column of compressed chunks — useful
for storing compressed data alongside raw data in a single DataFrame:

```python
df2 = fcp.compress_column(df, "user_id", chunk_size=65_536)
# df2["user_id_flux"] contains one bytes blob per chunk, None for padding rows
```

## Round-Trip Testing

```python
df2 = fcp.round_trip(df)   # compress → decompress
assert df.equals(df2)
```

## Compression Statistics Report

```python
stats = fcp.compression_stats(df)
print(stats.to_string(index=False))
```

Example output:
```
      column  dtype    rows  raw_bytes  flux_bytes  ratio    strategy
     user_id  int64  100000     800000        3245 246.5x  DeltaDelta
     revenue  int64  100000     800000       68421  11.7x     BitSlab
      region  int64  100000     800000        2891 276.7x  Dictionary
```

## All Numeric dtypes Supported

```python
for dtype in ["int8", "int16", "int32", "int64",
              "uint8", "uint16", "uint32", "uint64",
              "float32", "float64"]:
    df = pd.DataFrame({"x": np.arange(1024, dtype=dtype)})
    buf = fcp.compress_df(df)
    df2 = fcp.decompress_df(buf)
```

## Strategy Override

```python
buf = fcp.compress_df(df, strategy="delta")   # force DeltaDelta
buf = fcp.compress_df(df, strategy="bitslab") # force BitSlab
```
