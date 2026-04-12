# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for fluxcompress.pandas integration.

Skipped automatically if pandas is not installed.
"""

import pytest

pandas = pytest.importorskip("pandas", reason="pandas not installed")
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# compress_df / decompress_df
# ─────────────────────────────────────────────────────────────────────────────

def test_compress_df_returns_flux_buffer():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": range(1024)})
    buf = fcp.compress_df(df)
    import fluxcompress as fc
    assert isinstance(buf, fc.FluxBuffer)
    assert len(buf) > 0


def test_compress_df_smaller_than_raw():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": range(10_000)})
    buf = fcp.compress_df(df)
    raw_bytes = df.memory_usage(deep=True).sum()
    assert len(buf) < raw_bytes


def test_decompress_df_round_trip():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": range(5_000)})
    buf = fcp.compress_df(df)
    df2 = fcp.decompress_df(buf)
    assert list(df["value"]) == list(df2["value"])


def test_compress_df_multi_column():
    import fluxcompress.pandas as fcp
    n = 50_000
    df = pd.DataFrame({
        "user_id":     range(n),
        "revenue":     [i * 37 % 99_999 for i in range(n)],
        "region_code": [i % 8 for i in range(n)],
    })
    buf = fcp.compress_df(df)
    assert len(buf) > 0


def test_compress_df_column_selection():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({
        "user_id": range(1000),
        "name":    ["alice"] * 1000,  # string — excluded
        "revenue": range(1000),
    })
    # Selecting only numeric columns explicitly.
    buf = fcp.compress_df(df, columns=["user_id", "revenue"])
    assert len(buf) > 0


def test_compress_df_no_numeric_raises():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"name": ["alice", "bob", "carol"]})
    with pytest.raises(ValueError, match="No numeric columns"):
        fcp.compress_df(df)


@pytest.mark.parametrize("dtype", [
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
])
def test_compress_df_all_dtypes(dtype):
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": np.arange(1024, dtype=dtype)})
    buf = fcp.compress_df(df)
    df2 = fcp.decompress_df(buf)
    assert len(df2) == 1024


# ─────────────────────────────────────────────────────────────────────────────
# compress_series / decompress_series
# ─────────────────────────────────────────────────────────────────────────────

def test_compress_series_returns_flux_buffer():
    import fluxcompress.pandas as fcp
    s = pd.Series(range(2048), name="user_id")
    buf = fcp.compress_series(s)
    import fluxcompress as fc
    assert isinstance(buf, fc.FluxBuffer)


def test_decompress_series_round_trip():
    import fluxcompress.pandas as fcp
    s = pd.Series(range(5_000), name="value")
    buf = fcp.compress_series(s)
    s2 = fcp.decompress_series(buf, name="value")
    assert list(s) == list(s2)


def test_compress_series_unnamed():
    import fluxcompress.pandas as fcp
    s = pd.Series(range(256))  # no name
    buf = fcp.compress_series(s)
    assert len(buf) > 0


# ─────────────────────────────────────────────────────────────────────────────
# compress_column (in-place chunked)
# ─────────────────────────────────────────────────────────────────────────────

def test_compress_column_adds_new_column():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"user_id": range(4096)})
    df2 = fcp.compress_column(df, "user_id", chunk_size=1024)
    assert "user_id_flux" in df2.columns
    assert "user_id" in df2.columns  # original preserved


def test_compress_column_correct_chunk_count():
    import fluxcompress.pandas as fcp
    n, chunk = 4096, 1024
    df = pd.DataFrame({"user_id": range(n)})
    df2 = fcp.compress_column(df, "user_id", chunk_size=chunk)
    # One non-None blob per chunk start.
    non_null_blobs = df2["user_id_flux"].dropna()
    assert len(non_null_blobs) == n // chunk


def test_compress_column_custom_output_name():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"x": range(1024)})
    df2 = fcp.compress_column(df, "x", output_column="x_compressed")
    assert "x_compressed" in df2.columns


def test_compress_column_blobs_are_bytes():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"val": range(2048)})
    df2 = fcp.compress_column(df, "val", chunk_size=1024)
    blobs = df2["val_flux"].dropna()
    for blob in blobs:
        assert isinstance(blob, bytes)
        assert len(blob) > 0


# ─────────────────────────────────────────────────────────────────────────────
# round_trip
# ─────────────────────────────────────────────────────────────────────────────

def test_round_trip_simple():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": range(10_000)})
    df2 = fcp.round_trip(df)
    assert list(df["value"]) == list(df2["value"])


def test_round_trip_multi_column():
    import fluxcompress.pandas as fcp
    n = 5_000
    df = pd.DataFrame({
        "a": range(n),
        "b": [i * 7 for i in range(n)],
    })
    # round_trip compresses/decompresses both columns.
    df2 = fcp.round_trip(df)
    assert len(df2) == n


# ─────────────────────────────────────────────────────────────────────────────
# compression_stats
# ─────────────────────────────────────────────────────────────────────────────

def test_compression_stats_returns_dataframe():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({
        "sequential": range(10_000),
        "constant":   [42] * 10_000,
        "random":     [i * 37 % 65536 for i in range(10_000)],
    })
    stats = fcp.compression_stats(df)
    assert isinstance(stats, pd.DataFrame)
    assert list(stats.columns) == [
        "column", "dtype", "rows", "raw_bytes", "flux_bytes", "ratio", "strategy"
    ]
    assert len(stats) == 3  # one row per column


def test_compression_stats_ratios_are_positive():
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"x": range(1024)})
    stats = fcp.compression_stats(df)
    # ratio column is a string like "4.2x"
    ratio_val = float(stats["ratio"].iloc[0].replace("x", ""))
    assert ratio_val > 0


def test_compression_stats_sequential_beats_random():
    import fluxcompress.pandas as fcp
    n = 10_000
    df = pd.DataFrame({
        "sequential": range(n),
        "random": [i * 0x9E3779B9 % (2**32) for i in range(n)],
    })
    stats = fcp.compression_stats(df).set_index("column")
    seq_ratio = float(stats.loc["sequential", "ratio"].replace("x", ""))
    rnd_ratio = float(stats.loc["random",     "ratio"].replace("x", ""))
    # Sequential data always compresses better than random.
    assert seq_ratio > rnd_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Predicate pushdown via decompress_df
# ─────────────────────────────────────────────────────────────────────────────

def test_decompress_df_predicate():
    import fluxcompress as fc
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": range(50_000)})
    buf = fcp.compress_df(df)
    df_filtered = fcp.decompress_df(buf, predicate=fc.col("value") > 40_000)
    result_set = set(df_filtered["value"].tolist())
    # All rows > 40 000 must be present.
    for v in range(40_001, 50_000):
        assert v in result_set, f"Row {v} missing from predicate result"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark (only runs with --benchmark-only)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.benchmark
def test_bench_pandas_compress(benchmark):
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": range(100_000)})
    result = benchmark(fcp.compress_df, df)
    assert len(result) > 0


@pytest.mark.benchmark
def test_bench_pandas_decompress(benchmark):
    import fluxcompress.pandas as fcp
    df = pd.DataFrame({"value": range(100_000)})
    buf = fcp.compress_df(df)
    result = benchmark(fcp.decompress_df, buf)
    assert len(result) == 100_000
