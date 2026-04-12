# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmarks using pytest-benchmark.

Run with:
    pytest python/tests/test_benchmarks.py --benchmark-only -v

Or for a quick comparison vs raw bytes:
    pytest python/tests/test_benchmarks.py --benchmark-only --benchmark-compare
"""

import pyarrow as pa
import pytest
import fluxcompress as fc


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — pre-built tables of various shapes
# ─────────────────────────────────────────────────────────────────────────────

def _sequential(n: int):
    return pa.table({"value": pa.array(range(n), type=pa.uint64())})

def _constant(n: int):
    return pa.table({"value": pa.array([42] * n, type=pa.uint64())})

def _random(n: int):
    vals = [(i * 0x9E3779B9 + 0x6C62272E) & 0xFFFFFF for i in range(n)]
    return pa.table({"value": pa.array(vals, type=pa.uint64())})

def _low_card(n: int):
    return pa.table({"value": pa.array([i % 8 for i in range(n)], type=pa.uint64())})


SIZES = [1_024, 65_536, 1_048_576]
PATTERNS = {
    "sequential": _sequential,
    "constant":   _constant,
    "random":     _random,
    "low_card":   _low_card,
}


# ─────────────────────────────────────────────────────────────────────────────
# Compression benchmarks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_bench_compress_sequential(benchmark, n):
    table = _sequential(n)
    result = benchmark(fc.compress, table)
    assert len(result) > 0


@pytest.mark.parametrize("n", SIZES)
def test_bench_compress_random(benchmark, n):
    table = _random(n)
    result = benchmark(fc.compress, table)
    assert len(result) > 0


@pytest.mark.parametrize("n", SIZES)
def test_bench_compress_constant(benchmark, n):
    table = _constant(n)
    result = benchmark(fc.compress, table)
    assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Decompression benchmarks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", SIZES)
def test_bench_decompress_sequential(benchmark, n):
    buf = fc.compress(_sequential(n))
    result = benchmark(fc.decompress, buf)
    assert len(result) == n


@pytest.mark.parametrize("n", SIZES)
def test_bench_decompress_random(benchmark, n):
    buf = fc.compress(_random(n))
    result = benchmark(fc.decompress, buf)
    assert len(result) == n


# ─────────────────────────────────────────────────────────────────────────────
# Predicate pushdown benchmark
# ─────────────────────────────────────────────────────────────────────────────

def test_bench_predicate_pushdown(benchmark):
    n   = 1_048_576
    buf = fc.compress(_sequential(n))
    pred = fc.col("value") > n - 1024  # should skip ~99.9% of blocks

    result = benchmark(fc.decompress, buf, predicate=pred)
    assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Compression ratio report (not a timing benchmark, but useful to log)
# ─────────────────────────────────────────────────────────────────────────────

def test_compression_ratios():
    """Print a compression ratio table — informational, always passes."""
    n = 65_536
    print("\nCompression Ratio Report (n=65536)")
    print(f"{'Pattern':<15} {'Raw MB':>8} {'Flux MB':>8} {'Ratio':>7} {'Strategy'}")
    print("-" * 55)
    for name, factory in PATTERNS.items():
        table   = factory(n)
        raw_b   = n * 8
        buf     = fc.compress(table)
        info    = fc.inspect(buf)
        strats  = {b.strategy for b in info.blocks}
        ratio   = raw_b / len(buf)
        print(
            f"{name:<15} {raw_b/1e6:>8.3f} {len(buf)/1e6:>8.3f} "
            f"{ratio:>7.2f}x {', '.join(strats)}"
        )
