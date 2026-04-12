#!/usr/bin/env python3
# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Scaling benchmark: FluxCompress vs Parquet across row counts from 1K to 50M.

Generates 3 charts:
  1. Compression ratio vs row count
  2. Compress speed (MB/s) vs row count
  3. Decompress speed (MB/s) vs row count

Also extrapolates to TB-scale based on observed scaling curves.

Usage:
    python python/tests/bench_scaling.py
"""

from __future__ import annotations

import gc
import io
import time
from dataclasses import dataclass
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
]

# Use sequential data — the most common real-world pattern (IDs, timestamps).
def make_table(n: int) -> pa.Table:
    return pa.table({"value": pa.array(range(n), type=pa.uint64())})


@dataclass
class Result:
    name: str
    rows: int
    raw_bytes: int
    compressed_bytes: int
    ratio: float
    compress_ms: float
    decompress_ms: float
    compress_mbs: float
    decompress_mbs: float


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def bench_flux(table: pa.Table, profile: str) -> Result:
    import fluxcompress as fc
    raw = table.num_rows * 8
    gc.collect()

    t0 = time.perf_counter()
    buf = fc.compress(table, profile=profile)
    comp_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    _ = fc.decompress(buf)
    decomp_ms = (time.perf_counter() - t1) * 1000

    comp_mbs = (raw / 1e6) / (comp_ms / 1000) if comp_ms > 0 else float("inf")
    decomp_mbs = (raw / 1e6) / (decomp_ms / 1000) if decomp_ms > 0 else float("inf")

    return Result(
        name=f"Flux ({profile})",
        rows=table.num_rows,
        raw_bytes=raw,
        compressed_bytes=len(buf),
        ratio=raw / max(len(buf), 1),
        compress_ms=comp_ms,
        decompress_ms=decomp_ms,
        compress_mbs=comp_mbs,
        decompress_mbs=decomp_mbs,
    )


def bench_parquet(table: pa.Table, codec: str) -> Result:
    raw = table.num_rows * 8
    gc.collect()

    sink = io.BytesIO()
    t0 = time.perf_counter()
    pq.write_table(table, sink, compression=codec)
    comp_ms = (time.perf_counter() - t0) * 1000
    comp_bytes = sink.tell()

    sink.seek(0)
    t1 = time.perf_counter()
    _ = pq.read_table(sink)
    decomp_ms = (time.perf_counter() - t1) * 1000

    comp_mbs = (raw / 1e6) / (comp_ms / 1000) if comp_ms > 0 else float("inf")
    decomp_mbs = (raw / 1e6) / (decomp_ms / 1000) if decomp_ms > 0 else float("inf")

    return Result(
        name=f"Parquet ({codec})",
        rows=table.num_rows,
        raw_bytes=raw,
        compressed_bytes=comp_bytes,
        ratio=raw / max(comp_bytes, 1),
        compress_ms=comp_ms,
        decompress_ms=decomp_ms,
        compress_mbs=comp_mbs,
        decompress_mbs=decomp_mbs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> List[Result]:
    results = []
    for n in SIZES:
        print(f"  Benchmarking {n:>12,} rows ...", end="", flush=True)
        table = make_table(n)

        results.append(bench_flux(table, "speed"))
        results.append(bench_flux(table, "balanced"))
        results.append(bench_flux(table, "archive"))
        results.append(bench_parquet(table, "snappy"))
        results.append(bench_parquet(table, "zstd"))

        del table
        gc.collect()
        print(" done")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot charts
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Flux (speed)": "#2196F3",
    "Flux (balanced)": "#4CAF50",
    "Flux (archive)": "#FF9800",
    "Parquet (snappy)": "#9C27B0",
    "Parquet (zstd)": "#F44336",
}

MARKERS = {
    "Flux (speed)": "o",
    "Flux (balanced)": "s",
    "Flux (archive)": "D",
    "Parquet (snappy)": "^",
    "Parquet (zstd)": "v",
}


def plot_scaling(results: List[Result], output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("FluxCompress vs Parquet — Scaling Benchmark (Sequential u64 Data)",
                 fontsize=14, fontweight="bold")

    names = list(COLORS.keys())

    # ── Chart 1: Compression Ratio ────────────────────────────────────────
    ax1 = axes[0]
    for name in names:
        xs = [r.rows for r in results if r.name == name]
        ys = [r.ratio for r in results if r.name == name]
        ax1.plot(xs, ys, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Rows")
    ax1.set_ylabel("Compression Ratio (×)")
    ax1.set_title("Compression Ratio")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Chart 2: Compress Speed (MB/s) ────────────────────────────────────
    ax2 = axes[1]
    for name in names:
        xs = [r.rows for r in results if r.name == name]
        ys = [r.compress_mbs for r in results if r.name == name]
        ax2.plot(xs, ys, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Rows")
    ax2.set_ylabel("Throughput (MB/s)")
    ax2.set_title("Compress Speed")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Chart 3: Decompress Speed (MB/s) ──────────────────────────────────
    ax3 = axes[2]
    for name in names:
        xs = [r.rows for r in results if r.name == name]
        ys = [r.decompress_mbs for r in results if r.name == name]
        ax3.plot(xs, ys, marker=MARKERS[name], color=COLORS[name], label=name, linewidth=2)
    ax3.set_xscale("log")
    ax3.set_xlabel("Rows")
    ax3.set_ylabel("Throughput (MB/s)")
    ax3.set_title("Decompress Speed")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TB extrapolation
# ─────────────────────────────────────────────────────────────────────────────

def extrapolate_tb(results: List[Result]):
    tb_rows = 125_000_000_000  # ~1 TB of u64 data
    tb_bytes = tb_rows * 8

    print("\n" + "=" * 70)
    print("Extrapolation to 1 TB (125 billion rows of u64)")
    print("=" * 70)
    print(f"{'Format':<20} {'Est. Size':>12} {'Est. Comp Time':>15} {'Est. Decomp':>12}")
    print("-" * 70)

    for name in COLORS.keys():
        # Use the two largest data points to extrapolate
        pts = [(r.rows, r) for r in results if r.name == name]
        pts.sort(key=lambda x: x[0])
        if len(pts) < 2:
            continue

        # Use last point's throughput (asymptotic behavior)
        last = pts[-1][1]
        est_ratio = last.ratio
        est_size_bytes = tb_bytes / est_ratio
        est_comp_s = tb_bytes / 1e6 / last.compress_mbs if last.compress_mbs > 0 else float("inf")
        est_decomp_s = tb_bytes / 1e6 / last.decompress_mbs if last.decompress_mbs > 0 else float("inf")

        def human_bytes(b):
            if b < 1e9:
                return f"{b / 1e6:.0f} MB"
            return f"{b / 1e9:.1f} GB"

        def human_time(s):
            if s < 60:
                return f"{s:.0f}s"
            if s < 3600:
                return f"{s / 60:.1f} min"
            return f"{s / 3600:.1f} hr"

        print(
            f"{name:<20} {human_bytes(est_size_bytes):>12} "
            f"{human_time(est_comp_s):>15} {human_time(est_decomp_s):>12}"
        )
    print("=" * 70)
    print("(Extrapolated from 50M-row throughput; actual TB performance may vary)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FluxCompress vs Parquet — Scaling Benchmark")
    print("Pattern: Sequential u64 (IDs / timestamps)")
    print()

    results = run_all()

    # Print table
    print(f"\n{'Rows':>12} {'Format':<20} {'Size':>10} {'Ratio':>7} {'Comp MB/s':>10} {'Decomp MB/s':>11}")
    print("-" * 75)
    for r in results:
        def h(b):
            if b < 1024: return f"{b} B"
            if b < 1e6: return f"{b/1024:.1f} KB"
            if b < 1e9: return f"{b/1e6:.1f} MB"
            return f"{b/1e9:.2f} GB"
        print(
            f"{r.rows:>12,} {r.name:<20} {h(r.compressed_bytes):>10} "
            f"{r.ratio:>7.1f}x {r.compress_mbs:>9.0f} {r.decompress_mbs:>11.0f}"
        )

    output_path = "docs/scaling_benchmark.png"
    plot_scaling(results, output_path)
    extrapolate_tb(results)
