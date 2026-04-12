# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
pytest conftest.py — shared fixtures for the fluxcompress test suite.

Available fixtures
------------------
table_sequential    pyarrow.Table  1 024 sequential u64 values
table_constant      pyarrow.Table  1 024 copies of the value 42
table_random        pyarrow.Table  1 024 pseudo-random u64 values
table_low_card      pyarrow.Table  1 024 values with cardinality 8
table_outliers      pyarrow.Table  1 024 values with 1 % u64-max outliers
table_large         pyarrow.Table  100 000 multi-column table
table_multi_seg     pyarrow.Table  5 × SEGMENT_SIZE rows (5 Atlas blocks)

flux_sequential     FluxBuffer     pre-compressed sequential table
flux_large          FluxBuffer     pre-compressed large table

tmp_flux_path       str            temp file path ending in .flux (auto-deleted)
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import pyarrow as pa
import pytest

# SEGMENT_SIZE = 1024 rows per Loom classification window.
SEGMENT_SIZE = 1024


# ─────────────────────────────────────────────────────────────────────────────
# Raw PyArrow table fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def table_sequential() -> pa.Table:
    """1 024 sequential u64 values — triggers DeltaDelta encoding."""
    return pa.table({"value": pa.array(range(SEGMENT_SIZE), type=pa.uint64())})


@pytest.fixture(scope="session")
def table_constant() -> pa.Table:
    """1 024 identical values — triggers RLE encoding."""
    return pa.table({"value": pa.array([42] * SEGMENT_SIZE, type=pa.uint64())})


@pytest.fixture(scope="session")
def table_random() -> pa.Table:
    """1 024 pseudo-random u64 values — triggers BitSlab encoding."""
    # Deterministic xorshift PRNG for reproducibility.
    state = 0xDEADBEEF_CAFEBABE
    values = []
    for _ in range(SEGMENT_SIZE):
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 7)  & 0xFFFFFFFFFFFFFFFF
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        values.append(state & 0x0000FFFFFFFFFFFF)  # 48-bit values
    return pa.table({"value": pa.array(values, type=pa.uint64())})


@pytest.fixture(scope="session")
def table_low_card() -> pa.Table:
    """Low-cardinality column (8 unique values) — triggers Dictionary encoding."""
    return pa.table({
        "value": pa.array([i % 8 for i in range(SEGMENT_SIZE)], type=pa.uint64())
    })


@pytest.fixture(scope="session")
def table_outliers() -> pa.Table:
    """Mostly small values with ~1 % u64-max outliers — triggers OutlierMap."""
    values = [i * 1_000 for i in range(SEGMENT_SIZE)]
    # Inject 10 outliers at regular intervals.
    for i in range(0, SEGMENT_SIZE, SEGMENT_SIZE // 10):
        values[i] = (2**63) - 1
    return pa.table({"value": pa.array(values, type=pa.uint64())})


@pytest.fixture(scope="session")
def table_large() -> pa.Table:
    """100 000-row multi-column table — realistic workload."""
    n = 100_000
    return pa.table({
        "user_id":     pa.array(range(n), type=pa.uint64()),
        "revenue":     pa.array([i * 37 % 99_999 for i in range(n)], type=pa.uint64()),
        "region_code": pa.array([i % 8 for i in range(n)], type=pa.uint64()),
        "session_ms":  pa.array(
            [(i * 1_234) % 86_400_000 for i in range(n)], type=pa.int64()
        ),
    })


@pytest.fixture(scope="session")
def table_multi_seg() -> pa.Table:
    """5 × SEGMENT_SIZE rows — produces exactly 5 Atlas blocks."""
    n = SEGMENT_SIZE * 5
    return pa.table({"value": pa.array(range(n), type=pa.uint64())})


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compressed FluxBuffer fixtures  (session scope — built once)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def flux_sequential(table_sequential) -> "FluxBuffer":
    """Pre-compressed sequential table."""
    import fluxcompress as fc
    return fc.compress(table_sequential)


@pytest.fixture(scope="session")
def flux_large(table_large) -> "FluxBuffer":
    """Pre-compressed large multi-column table."""
    import fluxcompress as fc
    return fc.compress(table_large)


# ─────────────────────────────────────────────────────────────────────────────
# File I/O fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_flux_path() -> Generator[str, None, None]:
    """
    Yield a temporary ``.flux`` file path that is automatically deleted
    after the test completes.

    Usage::

        def test_save(tmp_flux_path, table_sequential):
            import fluxcompress as fc
            buf = fc.compress(table_sequential)
            buf.save(tmp_flux_path)
            assert os.path.exists(tmp_flux_path)
    """
    with tempfile.NamedTemporaryFile(suffix=".flux", delete=False) as f:
        path = f.name
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Optional: pandas / polars fixtures (skipped if not installed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pandas_df(table_large):
    """pandas DataFrame equivalent of table_large (skipped if pandas not installed)."""
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    return table_large.to_pandas()


@pytest.fixture(scope="session")
def polars_df(table_large):
    """polars DataFrame equivalent of table_large (skipped if polars not installed)."""
    pl = pytest.importorskip("polars", reason="polars not installed")
    return pl.from_arrow(table_large)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Register custom marks used in the test suite."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
