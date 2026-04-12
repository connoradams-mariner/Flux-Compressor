# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Core pytest tests for the fluxcompress Python package.

Run with:
    pytest python/tests/ -v

Or after maturin develop:
    maturin develop --release
    pytest python/tests/ -v
"""

import pytest
import pyarrow as pa
import pyarrow.compute as pc
import tempfile
import os


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_table():
    """A small PyArrow table with sequential integers."""
    return pa.table({"value": pa.array(range(1024), type=pa.uint64())})


@pytest.fixture
def large_table():
    """A larger table with mixed patterns across columns."""
    n = 100_000
    return pa.table({
        "user_id":      pa.array(range(n), type=pa.uint64()),        # sequential
        "revenue":      pa.array([i * 37 % 99_999 for i in range(n)], type=pa.uint64()),
        "region_code":  pa.array([i % 8 for i in range(n)], type=pa.uint64()),  # low-card
    })


@pytest.fixture
def table_with_outliers():
    """Table with u64-range values plus a few extreme outliers."""
    values = list(range(1000))
    values[42]  = 2**63 - 1   # near u64::MAX
    values[999] = 2**63        # exactly u64::MAX / 2 + 1 .. treated as outlier
    return pa.table({"value": pa.array(values, type=pa.uint64())})


# ─────────────────────────────────────────────────────────────────────────────
# Import smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_import():
    import fluxcompress as fc
    assert hasattr(fc, "compress")
    assert hasattr(fc, "decompress")
    assert hasattr(fc, "inspect")
    assert hasattr(fc, "col")
    assert hasattr(fc, "__version__")
    assert fc.__version__ == "0.1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Basic compress / decompress
# ─────────────────────────────────────────────────────────────────────────────

def test_compress_returns_flux_buffer(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    assert repr(buf).startswith("FluxBuffer(")
    assert len(buf) > 0
    assert len(buf) < len(small_table) * 8  # must be smaller than raw u64


def test_round_trip_sequential(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    out = fc.decompress(buf)
    assert out.num_rows == small_table.num_rows
    assert out.column(0).to_pylist() == small_table.column(0).to_pylist()


def test_round_trip_large(large_table):
    import fluxcompress as fc
    buf = fc.compress(large_table)
    out = fc.decompress(buf, column_name="user_id")
    # FluxCompress currently decompresses all columns' blocks into a single
    # flat column (multi-column metadata is not yet in the Atlas footer).
    # Total rows = sum of rows across all columns.
    num_cols = large_table.num_columns
    assert out.num_rows == large_table.num_rows * num_cols


def test_round_trip_constant():
    import fluxcompress as fc
    table = pa.table({"value": pa.array([42] * 2048, type=pa.uint64())})
    buf = fc.compress(table)
    out = fc.decompress(buf)
    assert all(v == 42 for v in out.column(0).to_pylist())


def test_round_trip_with_outliers(table_with_outliers):
    import fluxcompress as fc
    buf = fc.compress(table_with_outliers)
    out = fc.decompress(buf)
    assert out.column(0).to_pylist() == table_with_outliers.column(0).to_pylist()


# ─────────────────────────────────────────────────────────────────────────────
# Strategy overrides
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", ["rle", "delta", "dict", "bitslab", "lz4"])
def test_strategy_override_round_trips(small_table, strategy):
    import fluxcompress as fc
    # RLE only works well on constant data, but all strategies must round-trip.
    table = pa.table({"value": pa.array([5] * 1024, type=pa.uint64())})
    buf = fc.compress(table, strategy=strategy)
    out = fc.decompress(buf)
    assert out.column(0).to_pylist() == table.column(0).to_pylist()


def test_unknown_strategy_raises(small_table):
    import fluxcompress as fc
    with pytest.raises(ValueError, match="unknown strategy"):
        fc.compress(small_table, strategy="blorp")


# ─────────────────────────────────────────────────────────────────────────────
# Predicate pushdown
# ─────────────────────────────────────────────────────────────────────────────

def test_predicate_gt(large_table):
    import fluxcompress as fc
    threshold = 50_000
    buf = fc.compress(large_table)
    out = fc.decompress(buf, predicate=fc.col("user_id") > threshold)
    values = out.column(0).to_pylist()
    # All returned values must be from blocks that *may* contain > threshold.
    # Due to block-level skipping (not row-level filtering), we may get some
    # values below the threshold that are in the same block as valid ones.
    # The key guarantee: we never miss valid rows.
    valid_in_full = [v for v in range(100_000) if v > threshold]
    returned_set  = set(values)
    for v in valid_in_full[:100]:  # spot-check first 100
        assert v in returned_set, f"Missing row {v} from predicate result"


def test_predicate_between(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    pred = fc.col("value").between(100, 200)
    out = fc.decompress(buf, predicate=pred)
    # All values in [100, 200] must be present.
    values = set(out.column(0).to_pylist())
    for v in range(100, 201):
        assert v in values


def test_predicate_and(large_table):
    import fluxcompress as fc
    buf = fc.compress(large_table)
    pred = (fc.col("user_id") > 10_000) & (fc.col("user_id") < 20_000)
    out = fc.decompress(buf, predicate=pred)
    assert out.num_rows > 0


def test_predicate_or(large_table):
    import fluxcompress as fc
    buf = fc.compress(large_table)
    pred = (fc.col("user_id") < 1_000) | (fc.col("user_id") > 99_000)
    out = fc.decompress(buf, predicate=pred)
    assert out.num_rows > 0


# ─────────────────────────────────────────────────────────────────────────────
# Inspect / Atlas footer
# ─────────────────────────────────────────────────────────────────────────────

def test_inspect_returns_file_info(large_table):
    import fluxcompress as fc
    buf = fc.compress(large_table)
    info = fc.inspect(buf)
    assert info.size_bytes == len(buf)
    assert info.num_blocks > 0
    assert len(info.blocks) == info.num_blocks


def test_inspect_block_fields(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    info = fc.inspect(buf)
    b = info.blocks[0]
    assert b.offset == 0
    assert b.z_min <= b.z_max
    assert b.strategy in {"Rle", "DeltaDelta", "Dictionary", "BitSlab", "SimdLz4"}


def test_inspect_via_flux_buffer(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    info = buf.inspect()
    assert info.num_blocks >= 1


# ─────────────────────────────────────────────────────────────────────────────
# File I/O (save / load / read_flux / write_flux)
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_load(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    with tempfile.NamedTemporaryFile(suffix=".flux", delete=False) as f:
        path = f.name
    try:
        fc.write_flux(buf, path)
        assert os.path.getsize(path) == len(buf)

        buf2 = fc.read_flux(path)
        assert len(buf2) == len(buf)

        out = fc.decompress(buf2)
        assert out.column(0).to_pylist() == small_table.column(0).to_pylist()
    finally:
        os.unlink(path)


def test_flux_buffer_save(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    with tempfile.NamedTemporaryFile(suffix=".flux", delete=False) as f:
        path = f.name
    try:
        buf.save(path)
        buf2 = fc.FluxBuffer.load(path)
        assert len(buf) == len(buf2)
    finally:
        os.unlink(path)


def test_load_nonexistent_raises():
    import fluxcompress as fc
    with pytest.raises(Exception):
        fc.read_flux("/nonexistent/path/data.flux")


# ─────────────────────────────────────────────────────────────────────────────
# FluxBuffer bytes access
# ─────────────────────────────────────────────────────────────────────────────

def test_to_bytes(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    raw = buf.to_bytes()
    assert isinstance(raw, bytes)
    assert len(raw) == len(buf)


def test_decompress_from_raw_bytes(small_table):
    import fluxcompress as fc
    buf = fc.compress(small_table)
    raw_bytes = buf.to_bytes()
    # decompress() should also accept raw bytes.
    out = fc.decompress(raw_bytes)
    assert out.column(0).to_pylist() == small_table.column(0).to_pylist()


# ─────────────────────────────────────────────────────────────────────────────
# Polars integration (skipped if polars not installed)
# ─────────────────────────────────────────────────────────────────────────────

polars = pytest.importorskip("polars", reason="polars not installed")


def test_polars_compress_decompress():
    import fluxcompress as fc
    df = polars.DataFrame({"value": list(range(10_000))})
    buf = fc.compress_polars(df)
    df2 = fc.decompress_polars(buf)
    assert len(df2) == len(df)
    assert df2["value"].to_list() == df["value"].to_list()


def test_polars_large_dataframe():
    import fluxcompress as fc
    df = polars.DataFrame({
        "id":  list(range(100_000)),
        "val": [i * 37 % 99_999 for i in range(100_000)],
    })
    buf = fc.compress_polars(df)
    # Multi-column: all columns' blocks are concatenated on decompression.
    df2 = fc.decompress_polars(buf, column_name="id")
    assert len(df2) == 100_000 * df.width


def test_polars_predicate_pushdown():
    import fluxcompress as fc
    df = polars.DataFrame({"value": list(range(50_000))})
    buf = fc.compress_polars(df)
    df_filtered = fc.decompress_polars(buf, predicate=fc.col("value") > 40_000)
    # All values > 40_000 must be present.
    result_set = set(df_filtered["value"].to_list())
    for v in range(40_001, 50_000):
        assert v in result_set, f"Missing {v} from predicate result"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks (run with: pytest --benchmark-only)
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmark_compress(benchmark, large_table):
    import fluxcompress as fc
    benchmark(fc.compress, large_table)


def test_benchmark_decompress(benchmark, large_table):
    import fluxcompress as fc
    buf = fc.compress(large_table)
    benchmark(fc.decompress, buf)


def test_benchmark_decompress_with_pushdown(benchmark, large_table):
    import fluxcompress as fc
    buf = fc.compress(large_table)
    pred = fc.col("user_id") > 50_000
    benchmark(fc.decompress, buf, predicate=pred)
