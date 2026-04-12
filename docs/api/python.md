# Python API Reference

## Core Functions

### `compress(table, strategy="auto") → FluxBuffer`

Compress any Arrow-compatible table into a [`FluxBuffer`](#fluxbuffer).

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `table` | `pa.Table`, `pa.RecordBatch`, `pl.DataFrame`, `pd.DataFrame` | — | Input data. Any object with `__arrow_c_stream__` works. |
| `strategy` | `str` | `"auto"` | Force a strategy: `"auto"`, `"rle"`, `"delta"`, `"dict"`, `"bitslab"`, `"lz4"` |

**Returns**: `FluxBuffer`

```python
import pyarrow as pa
import fluxcompress as fc

table = pa.table({"id": range(1_000_000)})
buf = fc.compress(table)
buf = fc.compress(table, strategy="delta")  # force DeltaDelta
```

---

### `decompress(buf, predicate=None, column_name="value") → pa.Table`

Decompress a [`FluxBuffer`](#fluxbuffer) or raw `bytes` into a PyArrow Table.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `buf` | `FluxBuffer \| bytes` | — | Compressed data |
| `predicate` | `Predicate \| None` | `None` | Pushdown filter — skips blocks whose `[z_min, z_max]` cannot satisfy it |
| `column_name` | `str` | `"value"` | Output column name |

**Returns**: `pa.Table`

```python
table = fc.decompress(buf)
table = fc.decompress(buf, predicate=fc.col("id") > 500_000)
table = fc.decompress(buf, predicate=(fc.col("id") >= 100) & (fc.col("id") <= 200))
```

---

### `col(name) → Column`

Create a [`Column`](#column) expression for building predicates.

```python
fc.col("revenue") > 1_000
fc.col("age").between(18, 65)
(fc.col("x") < 10) | (fc.col("x") > 90)
```

---

### `inspect(buf) → FileInfo`

Inspect the Atlas metadata footer without decompressing any data.

```python
info = fc.inspect(buf)
print(f"{info.num_blocks} blocks, {info.size_bytes} bytes")
for block in info.blocks:
    print(f"  {block.strategy}  z=[{block.z_min}, {block.z_max}]")
```

---

### `read_flux(path) → FluxBuffer`

Load a `.flux` file from disk.

```python
buf = fc.read_flux("data.flux")
```

---

### `write_flux(buf, path)`

Write a [`FluxBuffer`](#fluxbuffer) to a `.flux` file.

```python
fc.write_flux(buf, "data.flux")
```

---

## Classes

### `FluxBuffer`

A compressed FluxCompress buffer. Returned by [`compress()`](#compresstabledestrategyauto--fluxbuffer).

```python
buf = fc.compress(table)

len(buf)                        # int: size in bytes
repr(buf)                       # "FluxBuffer(182304 bytes)"
buf.to_bytes()                  # bytes: raw compressed data
buf.decompress()                # pa.Table
buf.decompress(predicate=...)   # pa.Table with pushdown
buf.inspect()                   # FileInfo
buf.save("data.flux")           # write to disk
FluxBuffer.load("data.flux")    # load from disk
```

---

### `Predicate`

A filter predicate for pushdown during decompression. Built via [`col()`](#colname--column).

Blocks in the Atlas footer whose `[z_min, z_max]` range cannot satisfy the
predicate are skipped without reading or decompressing their data.

```python
# Comparison operators
fc.col("x") > 100
fc.col("x") < 100
fc.col("x") >= 100   # equivalent to > 99
fc.col("x") <= 100   # equivalent to < 101

# Range check
fc.col("x").between(0, 99_999)

# Logical composition
(fc.col("a") > 10) & (fc.col("b") < 50)
(fc.col("a") < 10) | (fc.col("a") > 90)
```

---

### `Column`

Intermediate expression returned by [`col()`](#colname--column).

| Method | Description |
|---|---|
| `__gt__(value)` | `col > value` |
| `__lt__(value)` | `col < value` |
| `__ge__(value)` | `col >= value` |
| `__le__(value)` | `col <= value` |
| `__eq__(value)` | `col == value` |
| `between(lo, hi)` | `lo <= col <= hi` |

---

### `FileInfo`

Returned by [`inspect()`](#inspectbuf--fileinfo).

| Attribute | Type | Description |
|---|---|---|
| `size_bytes` | `int` | Total compressed size |
| `num_blocks` | `int` | Number of Atlas blocks |
| `blocks` | `list[BlockInfo]` | Per-block metadata |

---

### `BlockInfo`

One entry in `FileInfo.blocks`.

| Attribute | Type | Description |
|---|---|---|
| `offset` | `int` | Byte offset in the buffer |
| `z_min` | `int` | Minimum Z-Order coordinate |
| `z_max` | `int` | Maximum Z-Order coordinate |
| `strategy` | `str` | Strategy name: `"Rle"`, `"DeltaDelta"`, `"Dictionary"`, `"BitSlab"`, `"SimdLz4"` |

---

## Polars Helpers

```python
import fluxcompress as fc

# compress_polars / decompress_polars
buf = fc.compress_polars(df)              # pl.DataFrame → FluxBuffer
df2 = fc.decompress_polars(buf)           # FluxBuffer → pl.DataFrame
df3 = fc.decompress_polars(buf, predicate=fc.col("id") > 500_000)
```

See [Polars Integration Guide](../guides/polars.md) for full details.

---

## Pandas Helpers

```python
import fluxcompress.pandas as fcp

buf   = fcp.compress_df(df)               # pd.DataFrame → FluxBuffer
df2   = fcp.decompress_df(buf)            # FluxBuffer → pd.DataFrame
buf   = fcp.compress_series(s)            # pd.Series → FluxBuffer
s2    = fcp.decompress_series(buf)        # FluxBuffer → pd.Series
df2   = fcp.round_trip(df)               # compress + decompress (testing)
stats = fcp.compression_stats(df)        # ratio report DataFrame
```

See [Pandas Integration Guide](../guides/pandas.md) for full details.

---

## Spark Helpers

```python
import fluxcompress.spark as fcs

fcs.register_udfs(spark)                  # SQL: flux_compress / flux_decompress
df2 = fcs.compress_dataframe(df, "col")  # adds col_flux BinaryType column
df3 = fcs.decompress_dataframe(df2, "col_flux")
```

See [Spark Integration Guide](../guides/spark.md) for full details.
