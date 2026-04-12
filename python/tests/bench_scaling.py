#!/usr/bin/env python3
# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Complete scaling benchmark: FluxCompress vs Parquet vs Feather.

Tests single-column and multi-column tables from 1K to 100M rows.
Generates charts and TB extrapolations.

Usage:
    python python/tests/bench_scaling.py
"""

from __future__ import annotations
import gc, io, time, sys
from dataclasses import dataclass, field
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather

# ─────────────────────────────────────────────────────────────────────────────

SINGLE_COL_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000]
MULTI_COL_SIZES  = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]

def make_single(n):
    return pa.table({"value": pa.array(range(n), type=pa.uint64())})

def make_multi(n):
    return pa.table({
        "user_id":    pa.array(range(n), type=pa.uint64()),
        "revenue":    pa.array([i * 37 % 99_999 for i in range(n)], type=pa.uint64()),
        "region":     pa.array([i % 8 for i in range(n)], type=pa.uint64()),
        "session_ms": pa.array([(i * 1234) % 86_400_000 for i in range(n)], type=pa.int64()),
    })

@dataclass
class R:
    name: str; kind: str; rows: int; raw: int; comp: int
    ratio: float; c_ms: float; d_ms: float; c_mbs: float; d_mbs: float

def _mbs(raw, ms): return (raw/1e6)/(ms/1000) if ms > 0 else float("inf")

def bench_flux(table, profile, kind):
    import fluxcompress as fc
    raw = sum(c.nbytes for c in table.columns); gc.collect()
    t0 = time.perf_counter()
    buf = fc.compress(table, profile=profile)
    c_ms = (time.perf_counter()-t0)*1000
    t1 = time.perf_counter()
    _ = fc.decompress(buf)
    d_ms = (time.perf_counter()-t1)*1000
    return R(f"Flux ({profile})", kind, table.num_rows, raw, len(buf),
             raw/max(len(buf),1), c_ms, d_ms, _mbs(raw,c_ms), _mbs(raw,d_ms))

def bench_pq(table, codec, kind):
    raw = sum(c.nbytes for c in table.columns); gc.collect()
    s = io.BytesIO()
    t0 = time.perf_counter(); pq.write_table(table, s, compression=codec)
    c_ms = (time.perf_counter()-t0)*1000; comp = s.tell()
    s.seek(0); t1 = time.perf_counter(); _ = pq.read_table(s)
    d_ms = (time.perf_counter()-t1)*1000
    return R(f"Parquet ({codec})", kind, table.num_rows, raw, comp,
             raw/max(comp,1), c_ms, d_ms, _mbs(raw,c_ms), _mbs(raw,d_ms))

def bench_feather(table, kind):
    raw = sum(c.nbytes for c in table.columns); gc.collect()
    s = io.BytesIO()
    t0 = time.perf_counter(); feather.write_feather(table, s, compression="lz4")
    c_ms = (time.perf_counter()-t0)*1000; comp = s.tell()
    s.seek(0); t1 = time.perf_counter(); _ = feather.read_table(s)
    d_ms = (time.perf_counter()-t1)*1000
    return R("Feather (lz4)", kind, table.num_rows, raw, comp,
             raw/max(comp,1), c_ms, d_ms, _mbs(raw,c_ms), _mbs(raw,d_ms))

# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    results = []
    for n in SINGLE_COL_SIZES:
        print(f"  Single-col {n:>12,} rows ...", end="", flush=True)
        t = make_single(n)
        results.append(bench_flux(t, "balanced", "single"))
        results.append(bench_flux(t, "archive", "single"))
        results.append(bench_pq(t, "snappy", "single"))
        results.append(bench_pq(t, "zstd", "single"))
        results.append(bench_feather(t, "single"))
        del t; gc.collect(); print(" done")

    for n in MULTI_COL_SIZES:
        print(f"  Multi-col  {n:>12,} rows ...", end="", flush=True)
        t = make_multi(n)
        results.append(bench_flux(t, "balanced", "multi"))
        results.append(bench_flux(t, "archive", "multi"))
        results.append(bench_pq(t, "snappy", "multi"))
        results.append(bench_pq(t, "zstd", "multi"))
        results.append(bench_feather(t, "multi"))
        del t; gc.collect(); print(" done")

    return results

# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "Flux (balanced)": ("#4CAF50","s"),
    "Flux (archive)":  ("#FF9800","D"),
    "Parquet (snappy)":("#9C27B0","^"),
    "Parquet (zstd)":  ("#F44336","v"),
    "Feather (lz4)":   ("#607D8B","x"),
}

def plot(results, kind, title_suffix, path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"FluxCompress vs Parquet — {title_suffix}", fontsize=14, fontweight="bold")
    rs = [r for r in results if r.kind == kind]

    for name,(c,m) in PALETTE.items():
        xs = [r.rows for r in rs if r.name==name]
        # Ratio
        ys = [r.ratio for r in rs if r.name==name]
        if xs: axes[0].plot(xs, ys, marker=m, color=c, label=name, linewidth=2)
        # Compress speed
        ys = [r.c_mbs for r in rs if r.name==name]
        if xs: axes[1].plot(xs, ys, marker=m, color=c, label=name, linewidth=2)
        # Decompress speed
        ys = [r.d_mbs for r in rs if r.name==name]
        if xs: axes[2].plot(xs, ys, marker=m, color=c, label=name, linewidth=2)

    for i,(t,yl) in enumerate([("Compression Ratio","Ratio (×)"),("Compress Speed","MB/s"),("Decompress Speed","MB/s")]):
        axes[i].set_xscale("log"); axes[i].set_xlabel("Rows"); axes[i].set_ylabel(yl)
        axes[i].set_title(t); axes[i].legend(fontsize=8); axes[i].grid(True, alpha=0.3)
    if kind=="single": axes[0].set_yscale("log")

    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved: {path}")

def extrapolate(results, kind, raw_per_row):
    tb = 1_000_000_000_000
    tb_rows = tb // raw_per_row
    print(f"\n{'='*75}")
    print(f"TB Extrapolation — {kind}-column ({tb_rows/1e9:.0f}B rows, {raw_per_row}B/row)")
    print(f"{'='*75}")
    print(f"{'Format':<20} {'Est. Size':>12} {'Comp Time':>12} {'Decomp Time':>12}")
    print("-"*75)

    for name in PALETTE:
        pts = sorted([(r.rows,r) for r in results if r.name==name and r.kind==kind])
        if not pts: continue
        last = pts[-1][1]
        est_size = tb / last.ratio
        est_comp = (tb/1e6)/last.c_mbs if last.c_mbs>0 else float("inf")
        est_decomp = (tb/1e6)/last.d_mbs if last.d_mbs>0 else float("inf")

        def hb(b):
            if b<1e9: return f"{b/1e6:.0f} MB"
            return f"{b/1e9:.1f} GB"
        def ht(s):
            if s<60: return f"{s:.0f}s"
            if s<3600: return f"{s/60:.1f} min"
            return f"{s/3600:.1f} hr"
        print(f"{name:<20} {hb(est_size):>12} {ht(est_comp):>12} {ht(est_decomp):>12}")
    print("="*75)

# ─────────────────────────────────────────────────────────────────────────────

def print_table(results):
    print(f"\n{'Rows':>12} {'Kind':<6} {'Format':<20} {'Size':>10} {'Ratio':>7} {'C MB/s':>8} {'D MB/s':>8}")
    print("-"*80)
    for r in results:
        def h(b):
            if b<1024: return f"{b} B"
            if b<1e6: return f"{b/1024:.1f}KB"
            if b<1e9: return f"{b/1e6:.1f}MB"
            return f"{b/1e9:.2f}GB"
        print(f"{r.rows:>12,} {r.kind:<6} {r.name:<20} {h(r.comp):>10} {r.ratio:>6.1f}x {r.c_mbs:>7.0f} {r.d_mbs:>7.0f}")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FluxCompress vs Parquet — Complete Scaling Benchmark\n")
    results = run_all()
    print_table(results)

    plot(results, "single", "Single Column (Sequential u64)", "docs/scaling_single_col.png")
    plot(results, "multi",  "Multi-Column (4 cols, mixed patterns)", "docs/scaling_multi_col.png")

    extrapolate(results, "single", 8)
    extrapolate(results, "multi", 32)
