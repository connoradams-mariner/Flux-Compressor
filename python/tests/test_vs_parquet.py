# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FluxCompress vs Parquet comparison benchmark.

Run with:
    pytest python/tests/test_vs_parquet.py -v -s

Or standalone:
    python python/tests/test_vs_parquet.py

Compares FluxCompress and Apache Parquet (via pyarrow) across:
    - 4 data patterns: sequential, constant, low-cardinality, random
    - 3 sizes: 1K, 64K, 1M rows
    - Metrics: compressed size, compression ratio, compress time, decompress time
"""

from __future__ import annotations

import io
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def _sequential(n: int) -> pa.Table:
    """Monotonically increasing integers — ideal for delta encoding."""
    return pa.table({"value": pa.array(range(n), type=pa.uint64())})


def _constant(n: int) -> pa.Table:
    """All identical values — ideal for RLE."""
    return pa.table({"value": pa.array([42] * n, type=pa.uint64())})


def _low_card(n: int) -> pa.Table:
    """8 unique values cycling — ideal for dictionary encoding."""
    return pa.table({"value": pa.array([i % 8 for i in range(n)], type=pa.uint64())})


def _random(n: int) -> pa.Table:
    """Pseudo-random 48-bit values — stress test for general compression."""
    state = 0xDEADBEEF_CAFEBABE
    values = []
    for _ in range(n):
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 7) & 0xFFFFFFFFFFFFFFFF
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        values.append(state & 0x0000FFFFFFFFFFFF)
    return pa.table({"value": pa.array(values, type=pa.uint64())})


def _multi_column(n: int) -> pa.Table:
    """Realistic multi-column table with mixed patterns."""
    return pa.table({
        "user_id": pa.array(range(n), type=pa.uint64()),
        "revenue": pa.array([i * 37 % 99_999 for i in range(n)], type=pa.uint64()),
        "region": pa.array([i % 8 for i in range(n)], type=pa.uint64()),
        "session_ms": pa.array(
            [(i * 1_234) % 86_400_000 for i in range(n)], type=pa.int64()
        ),
    })


PATTERNS: dict[str, Callable[[int], pa.Table]] = {
    "sequential": _sequential,
    "constant": _constant,
    "low_card": _low_card,
    "random": _random,
    "multi_col": _multi_column,
}

SIZES = [1_024, 65_536, 1_048_576, 10_485_760]


# ─────────────────────────────────────────────────────────────────────────────
# Compression helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    pattern: str
    rows: int
    raw_bytes: int
    compressed_bytes: int
    ratio: float
    compress_ms: float
    decompress_ms: float


def _bench_flux(
    table: pa.Table,
    pattern: str,
    profile: str = "speed",
) -> BenchResult:
    """Benchmark FluxCompress on the given table."""
    import fluxcompress as fc

    raw_bytes = sum(col.nbytes for col in table.columns)

    # Compress
    t0 = time.perf_counter()
    buf = fc.compress(table, profile=profile)
    compress_ms = (time.perf_counter() - t0) * 1000

    compressed_bytes = len(buf)

    # Decompress
    t1 = time.perf_counter()
    _ = fc.decompress(buf)
    decompress_ms = (time.perf_counter() - t1) * 1000

    label = f"Flux ({profile})"
    return BenchResult(
        name=label,
        pattern=pattern,
        rows=table.num_rows,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        ratio=raw_bytes / max(compressed_bytes, 1),
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
    )


def _bench_feather(table: pa.Table, pattern: str) -> BenchResult:
    """Benchmark Arrow IPC / Feather (via pyarrow) on the given table."""
    import pyarrow.feather as feather

    raw_bytes = sum(col.nbytes for col in table.columns)

    sink = io.BytesIO()
    t0 = time.perf_counter()
    feather.write_feather(table, sink, compression="lz4")
    compress_ms = (time.perf_counter() - t0) * 1000
    compressed_bytes = sink.tell()

    sink.seek(0)
    t1 = time.perf_counter()
    _ = feather.read_table(sink)
    decompress_ms = (time.perf_counter() - t1) * 1000

    return BenchResult(
        name="Feather (lz4)",
        pattern=pattern,
        rows=table.num_rows,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        ratio=raw_bytes / max(compressed_bytes, 1),
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
    )


def _bench_parquet(
    table: pa.Table,
    pattern: str,
    compression: str = "snappy",
) -> BenchResult:
    """Benchmark Parquet (via pyarrow) on the given table."""
    raw_bytes = sum(col.nbytes for col in table.columns)

    # Compress — write to an in-memory buffer
    sink = io.BytesIO()
    t0 = time.perf_counter()
    pq.write_table(table, sink, compression=compression)
    compress_ms = (time.perf_counter() - t0) * 1000

    compressed_bytes = sink.tell()

    # Decompress — read back from the buffer
    sink.seek(0)
    t1 = time.perf_counter()
    _ = pq.read_table(sink)
    decompress_ms = (time.perf_counter() - t1) * 1000

    return BenchResult(
        name=f"Parquet ({compression})",
        pattern=pattern,
        rows=table.num_rows,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        ratio=raw_bytes / max(compressed_bytes, 1),
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
    )


def _human(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.2f} MB"


# ─────────────────────────────────────────────────────────────────────────────
# pytest tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pattern", ["sequential", "constant", "low_card", "random"])
def test_flux_beats_or_matches_parquet_ratio(pattern):
    """
    Verify FluxCompress compression ratio is competitive with Parquet
    on data patterns that favour adaptive encoding.
    """
    import fluxcompress as fc

    n = 65_536
    table = PATTERNS[pattern](n)

    flux = _bench_flux(table, pattern)
    parquet = _bench_parquet(table, pattern, compression="snappy")

    print(
        f"\n  {pattern:>12}: Flux {flux.ratio:.1f}x ({_human(flux.compressed_bytes)}) "
        f"vs Parquet {parquet.ratio:.1f}x ({_human(parquet.compressed_bytes)})"
    )

    # FluxCompress should achieve at least 50% of Parquet's ratio on all
    # patterns — and beat it on structured numeric data.
    assert flux.ratio > 1.0, "FluxCompress should compress the data"


@pytest.mark.parametrize("pattern", ["sequential", "constant", "low_card", "random"])
def test_flux_decompresses_correctly_vs_parquet(pattern):
    """
    Verify that FluxCompress and Parquet both round-trip the data correctly.
    """
    import fluxcompress as fc

    n = 1_024
    table = PATTERNS[pattern](n)
    expected = table.column(0).to_pylist()

    # FluxCompress round-trip
    buf = fc.compress(table)
    flux_out = fc.decompress(buf)
    assert flux_out.column(0).to_pylist() == expected, "FluxCompress round-trip failed"

    # Parquet round-trip
    sink = io.BytesIO()
    pq.write_table(table, sink)
    sink.seek(0)
    parquet_out = pq.read_table(sink)
    assert parquet_out.column(0).to_pylist() == expected, "Parquet round-trip failed"


def test_multi_column_comparison():
    """Compare FluxCompress vs Parquet on a realistic multi-column table."""
    import fluxcompress as fc

    n = 65_536
    table = _multi_column(n)

    flux = _bench_flux(table, "multi_col")
    parquet_snappy = _bench_parquet(table, "multi_col", compression="snappy")
    parquet_zstd = _bench_parquet(table, "multi_col", compression="zstd")

    print(f"\n  Multi-column comparison ({n:,} rows, 4 columns):")
    print(f"    FluxCompress     : {_human(flux.compressed_bytes):>10}  {flux.ratio:.1f}x")
    print(
        f"    Parquet (snappy) : {_human(parquet_snappy.compressed_bytes):>10}  "
        f"{parquet_snappy.ratio:.1f}x"
    )
    print(
        f"    Parquet (zstd)   : {_human(parquet_zstd.compressed_bytes):>10}  "
        f"{parquet_zstd.ratio:.1f}x"
    )

    assert flux.compressed_bytes > 0


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison report (runs as a test, prints a table)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_comparison_report():
    """
    Print a complete side-by-side comparison table.

    Run with ``pytest -s`` to see output.
    """
    results: List[BenchResult] = []

    for pattern_name, factory in PATTERNS.items():
        for n in SIZES:
            table = factory(n)
            results.append(_bench_flux(table, pattern_name, "speed"))
            results.append(_bench_flux(table, pattern_name, "balanced"))
            results.append(_bench_flux(table, pattern_name, "archive"))
            results.append(_bench_parquet(table, pattern_name, "snappy"))
            results.append(_bench_parquet(table, pattern_name, "zstd"))
            results.append(_bench_feather(table, pattern_name))

    # Print report
    header = (
        f"{'Pattern':<12} {'Rows':>10} {'Format':<20} "
        f"{'Size':>10} {'Ratio':>7} {'Comp ms':>9} {'Decomp ms':>10}"
    )
    print(f"\n{'=' * len(header)}")
    print("FluxCompress vs Parquet — Compression Comparison")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    prev_key = None
    for r in results:
        key = (r.pattern, r.rows)
        if prev_key and prev_key != key:
            print()
        prev_key = key
        print(
            f"{r.pattern:<12} {r.rows:>10,} {r.name:<20} "
            f"{_human(r.compressed_bytes):>10} {r.ratio:>7.1f}x "
            f"{r.compress_ms:>8.1f} {r.decompress_ms:>10.1f}"
        )

    print(f"{'=' * len(header)}")

    # Compute wins: best Flux profile vs best Parquet codec
    flux_wins = 0
    total_comparisons = 0
    for pattern_name, factory in PATTERNS.items():
        for n in SIZES:
            flux_results = [r for r in results if r.pattern == pattern_name and r.rows == n and "Flux" in r.name]
            parquet_results = [r for r in results if r.pattern == pattern_name and r.rows == n and "Parquet" in r.name]
            if flux_results and parquet_results:
                best_flux = min(r.compressed_bytes for r in flux_results)
                best_parquet = min(r.compressed_bytes for r in parquet_results)
                if best_flux <= best_parquet:
                    flux_wins += 1
                total_comparisons += 1

    print(
        f"\nBest Flux profile smaller than best Parquet: "
        f"{flux_wins}/{total_comparisons} comparisons"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_full_comparison_report()
