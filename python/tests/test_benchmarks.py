# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmarks using pytest-benchmark.

Run with:
    pytest python/tests/test_benchmarks.py --benchmark-only -v

Or for a quick comparison vs raw bytes:
    pytest python/tests/test_benchmarks.py --benchmark-only --benchmark-compare
"""

import tempfile

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

# ─────────────────────────────────────────────────────────────────────────────
# Phase F: FluxTable API benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def _make_schema_v1() -> fc.TableSchema:
    return fc.TableSchema([
        fc.SchemaField(1, "value", "uint64"),
    ])


def _make_schema_v2() -> fc.TableSchema:
    return fc.TableSchema([
        fc.SchemaField(1, "value", "uint64"),
        fc.SchemaField(2, "tag",   "int64"),
    ])


@pytest.mark.parametrize("n", [1_024, 65_536])
def test_bench_fluxtable_append(benchmark, n, tmp_path):
    """Benchmark FluxTable.append() — compress + log entry + file write."""
    table = _sequential(n)
    buf   = fc.compress(table)

    tbl = fc.FluxTable(str(tmp_path / "t.fluxtable"))
    tbl.evolve_schema(_make_schema_v1())
    # Pre-warm: one append before benchmarking so we're not in cold-start I/O.
    tbl.append(buf)

    benchmark(tbl.append, buf)


@pytest.mark.parametrize("n,num_files", [(65_536, 4), (1_024, 16)])
def test_bench_fluxtable_scan(benchmark, n, num_files, tmp_path):
    """Benchmark FluxTable.scan() — streaming read over N files."""
    table = _sequential(n)
    buf   = fc.compress(table)

    tbl = fc.FluxTable(str(tmp_path / "t.fluxtable"))
    tbl.evolve_schema(_make_schema_v1())
    for _ in range(num_files):
        tbl.append(buf)

    def _do_scan() -> int:
        return sum(b.num_rows for b in tbl.scan())

    result = benchmark(_do_scan)
    assert result == n * num_files


def test_bench_fluxtable_evolve_schema(benchmark, tmp_path):
    """
    Benchmark FluxTable.evolve_schema() — pure metadata write (JSON log entry).

    A fresh table directory is created for each iteration so log-read overhead
    doesn't accumulate across rounds.
    """
    counter = [0]

    def _do_evolve() -> None:
        tbl = fc.FluxTable(str(tmp_path / f"t{counter[0]}.fluxtable"))
        counter[0] += 1
        tbl.evolve_schema(_make_schema_v1())
        tbl.evolve_schema(_make_schema_v2())

    benchmark(_do_evolve)


@pytest.mark.parametrize("n", [1_024, 65_536])
def test_bench_compress_and_append(benchmark, n, tmp_path):
    """End-to-end: compress a batch then append it to a FluxTable."""
    table = _sequential(n)
    tbl   = fc.FluxTable(str(tmp_path / "t.fluxtable"))
    tbl.evolve_schema(_make_schema_v1())

    def _do_compress_and_append() -> None:
        tbl.append(fc.compress(table))

    benchmark(_do_compress_and_append)


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
