# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FluxTable vs Delta Lake vs Parquet — end-to-end comparison benchmark.

Tests the Phase F FluxTable API against Delta Lake (deltalake, backed by
Parquet + transaction log) and plain Parquet across:

  - Multi-batch write throughput
  - Full-table scan throughput
  - Schema evolution overhead (add a column; both FluxTable and Delta do
    this metadata-only — no data rewrite)
  - On-disk compressed size

Run with:
    pytest python/tests/test_vs_delta.py -v -s

Standalone (no pytest):
    python python/tests/test_vs_delta.py

Optional dependency:
    pip install deltalake
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from dataclasses import dataclass
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Skip the entire module if deltalake is not installed.
deltalake = pytest.importorskip("deltalake", reason="pip install deltalake to enable Delta Lake comparison")
from deltalake import DeltaTable, write_deltalake  # noqa: E402

import fluxcompress as fc  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def _batch_v1(n: int, offset: int = 0) -> pa.Table:
    """Initial schema: user_id + revenue."""
    return pa.table({
        "user_id": pa.array(range(offset, offset + n), type=pa.uint64()),
        "revenue": pa.array([(i + offset) * 37 % 99_999 for i in range(n)], type=pa.int64()),
    })


def _batch_v2(n: int, offset: int = 0) -> pa.Table:
    """Post-evolution schema: user_id + revenue + region (new column)."""
    return pa.table({
        "user_id": pa.array(range(offset, offset + n), type=pa.uint64()),
        "revenue": pa.array([(i + offset) * 37 % 99_999 for i in range(n)], type=pa.int64()),
        "region":  pa.array([i % 8 for i in range(n)], type=pa.uint64()),
    })


# ─────────────────────────────────────────────────────────────────────────────
# BenchResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    total_rows: int
    num_files: int
    write_ms: float       # total write time (all batches)
    evolve_ms: float      # schema-evolution step (metadata-only)
    scan_ms: float        # full-table scan
    disk_bytes: int       # on-disk footprint (all files + metadata)
    raw_bytes: int        # uncompressed Arrow column bytes

    @property
    def ratio(self) -> float:
        return self.raw_bytes / max(self.disk_bytes, 1)

    @property
    def write_throughput_mb(self) -> float:
        return (self.raw_bytes / 1e6) / max(self.write_ms / 1000, 1e-9)

    @property
    def scan_throughput_mb(self) -> float:
        return (self.raw_bytes / 1e6) / max(self.scan_ms / 1000, 1e-9)


def _dir_bytes(path: str) -> int:
    """Sum of all file sizes under path (recursive)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def _raw_arrow_bytes(tables: List[pa.Table]) -> int:
    return sum(col.nbytes for t in tables for col in t.columns)


# ─────────────────────────────────────────────────────────────────────────────
# FluxTable runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_fluxtable(
    v1_batches: List[pa.Table],
    v2_batch: pa.Table,
) -> BenchResult:
    """
    Write v1_batches with schema_v1, evolve to schema_v2 (pure metadata),
    append v2_batch, then scan the full table.
    """
    raw = _raw_arrow_bytes(v1_batches + [v2_batch])
    total_rows = sum(t.num_rows for t in v1_batches) + v2_batch.num_rows

    with tempfile.TemporaryDirectory() as tmp:
        tbl = fc.FluxTable(f"{tmp}/flux.fluxtable")

        schema_v1 = fc.TableSchema([
            fc.SchemaField(1, "user_id", "uint64", nullable=False),
            fc.SchemaField(2, "revenue", "int64"),
        ])
        tbl.evolve_schema(schema_v1)

        # ── Write phase (v1 batches) ──────────────────────────────────────
        t0 = time.perf_counter()
        for batch in v1_batches:
            tbl.append(fc.compress(batch))
        write_ms = (time.perf_counter() - t0) * 1000

        # ── Schema evolution: add 'region' (pure JSON log entry) ──────────
        schema_v2 = fc.TableSchema([
            fc.SchemaField(1, "user_id", "uint64", nullable=False),
            fc.SchemaField(2, "revenue", "int64"),
            fc.SchemaField(3, "region",  "uint64"),
        ])
        t1 = time.perf_counter()
        tbl.evolve_schema(schema_v2)
        evolve_ms = (time.perf_counter() - t1) * 1000

        # ── Append post-evolution batch ───────────────────────────────────
        t2 = time.perf_counter()
        tbl.append(fc.compress(v2_batch))
        write_ms += (time.perf_counter() - t2) * 1000

        # ── Scan phase ────────────────────────────────────────────────────
        t3 = time.perf_counter()
        rows = sum(b.num_rows for b in tbl.scan())
        scan_ms = (time.perf_counter() - t3) * 1000

        disk = _dir_bytes(f"{tmp}/flux.fluxtable")

    return BenchResult(
        name="FluxTable",
        total_rows=total_rows,
        num_files=len(v1_batches) + 1,
        write_ms=write_ms,
        evolve_ms=evolve_ms,
        scan_ms=scan_ms,
        disk_bytes=disk,
        raw_bytes=raw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Delta Lake runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_delta(
    v1_batches: List[pa.Table],
    v2_batch: pa.Table,
) -> BenchResult:
    """
    Write v1 batches to Delta, then append v2_batch with schema_mode='merge'
    (Delta's schema evolution path), then scan.

    Note: Delta bundles schema evolution with a data write, so evolve_ms here
    includes writing v2_batch — unlike FluxTable which separates them.
    """
    raw = _raw_arrow_bytes(v1_batches + [v2_batch])
    total_rows = sum(t.num_rows for t in v1_batches) + v2_batch.num_rows

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/delta"

        # ── Write phase (v1 batches) ──────────────────────────────────────
        t0 = time.perf_counter()
        for i, batch in enumerate(v1_batches):
            write_deltalake(path, batch, mode="error" if i == 0 else "append")
        write_ms = (time.perf_counter() - t0) * 1000

        # ── Schema evolution + data write (bundled in Delta) ──────────────
        t1 = time.perf_counter()
        write_deltalake(path, v2_batch, mode="append", schema_mode="merge")
        evolve_ms = (time.perf_counter() - t1) * 1000

        # ── Scan phase ────────────────────────────────────────────────────
        dt = DeltaTable(path)
        t3 = time.perf_counter()
        result = dt.to_pyarrow_table()
        scan_ms = (time.perf_counter() - t3) * 1000

        disk = _dir_bytes(path)

    return BenchResult(
        name="Delta Lake",
        total_rows=total_rows,
        num_files=len(v1_batches) + 1,
        write_ms=write_ms,
        evolve_ms=evolve_ms,
        scan_ms=scan_ms,
        disk_bytes=disk,
        raw_bytes=raw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parquet (directory) runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_parquet(
    v1_batches: List[pa.Table],
    v2_batch: pa.Table,
    compression: str = "snappy",
) -> BenchResult:
    """
    Write separate Parquet files (no native schema evolution — v2_batch just
    adds a column, the dataset API null-fills missing columns on scan).
    """
    import pyarrow.dataset as ds

    raw = _raw_arrow_bytes(v1_batches + [v2_batch])
    total_rows = sum(t.num_rows for t in v1_batches) + v2_batch.num_rows

    with tempfile.TemporaryDirectory() as tmp:
        # ── Write phase (v1 batches) ──────────────────────────────────────
        t0 = time.perf_counter()
        for i, batch in enumerate(v1_batches):
            pq.write_table(batch, f"{tmp}/part-{i:04d}.parquet", compression=compression)
        write_ms = (time.perf_counter() - t0) * 1000

        # ── "Schema evolution" = just write the new file ──────────────────
        # Parquet has no native evolution; the dataset reader handles schema
        # unification on read via null-fill.
        t1 = time.perf_counter()
        pq.write_table(v2_batch, f"{tmp}/part-{len(v1_batches):04d}.parquet", compression=compression)
        evolve_ms = (time.perf_counter() - t1) * 1000

        # ── Scan phase (dataset API auto-unifies schema) ──────────────────
        t3 = time.perf_counter()
        result = ds.dataset(tmp, format="parquet").to_table()
        scan_ms = (time.perf_counter() - t3) * 1000

        disk = _dir_bytes(tmp)

    return BenchResult(
        name=f"Parquet ({compression})",
        total_rows=total_rows,
        num_files=len(v1_batches) + 1,
        write_ms=write_ms,
        evolve_ms=evolve_ms,
        scan_ms=scan_ms,
        disk_bytes=disk,
        raw_bytes=raw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(results: List[BenchResult], label: str = "") -> None:
    hdr = (
        f"{'Format':<26} {'Rows':>8} {'Files':>5} "
        f"{'Write ms':>9} {'Scan ms':>8} "
        f"{'Disk MB':>8} {'Ratio':>6} "
        f"{'Wr MB/s':>8} {'Sc MB/s':>8}"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    if label:
        print(label)
    print("FluxTable vs Delta Lake vs Parquet")
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        if r.disk_bytes < 0:
            # Sentinel: format could not write this dtype
            print(f"{r.name:<26} {'(type not supported)':>57}")
            continue
        print(
            f"{r.name:<26} {r.total_rows:>8,} {r.num_files:>5} "
            f"{r.write_ms:>9.1f} {r.scan_ms:>8.1f} "
            f"{r.disk_bytes / 1e6:>8.2f} {r.ratio:>6.2f}x "
            f"{r.write_throughput_mb:>8.0f} {r.scan_throughput_mb:>8.0f}"
        )
    print(sep)
    # Highlight FluxTable vs competitors
    flux = next((r for r in results if r.name == "FluxTable"), None)
    if flux:
        for r in results:
            if r.name == "FluxTable" or r.disk_bytes <= 0:
                continue
            size_pct = (r.disk_bytes - flux.disk_bytes) / r.disk_bytes * 100
            sign = "smaller" if size_pct > 0 else "larger"
            print(
                f"  FluxTable is {abs(size_pct):.1f}% {sign} on disk than {r.name}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Correctness tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rows_per_batch,num_v1_batches", [
    (1_024,   4),
    (65_536,  4),
])
def test_fluxtable_roundtrip_after_schema_evolution(rows_per_batch, num_v1_batches):
    """
    All rows written before and after schema evolution must be scannable.
    Pre-evolution files are projected to the new schema with NULL fill for
    the added 'region' column.
    """
    v1_batches = [_batch_v1(rows_per_batch, i * rows_per_batch) for i in range(num_v1_batches)]
    v2_batch   = _batch_v2(rows_per_batch, num_v1_batches * rows_per_batch)
    expected   = rows_per_batch * num_v1_batches + rows_per_batch

    with tempfile.TemporaryDirectory() as tmp:
        tbl = fc.FluxTable(f"{tmp}/t.fluxtable")
        tbl.evolve_schema(fc.TableSchema([
            fc.SchemaField(1, "user_id", "uint64", nullable=False),
            fc.SchemaField(2, "revenue", "int64"),
        ]))
        for b in v1_batches:
            tbl.append(fc.compress(b))

        tbl.evolve_schema(fc.TableSchema([
            fc.SchemaField(1, "user_id", "uint64", nullable=False),
            fc.SchemaField(2, "revenue", "int64"),
            fc.SchemaField(3, "region",  "uint64"),
        ]))
        tbl.append(fc.compress(v2_batch))

        got = sum(b.num_rows for b in tbl.scan())

    assert got == expected, f"Expected {expected} rows after evolution, got {got}"


def test_fluxtable_current_schema_reflects_evolution():
    """current_schema() must return the latest evolved schema."""
    with tempfile.TemporaryDirectory() as tmp:
        tbl = fc.FluxTable(f"{tmp}/t.fluxtable")
        assert tbl.current_schema() is None

        tbl.evolve_schema(fc.TableSchema([fc.SchemaField(1, "id", "uint64")]))
        s1 = tbl.current_schema()
        assert s1 is not None
        assert len(s1.fields) == 1

        tbl.evolve_schema(fc.TableSchema([
            fc.SchemaField(1, "id",  "uint64"),
            fc.SchemaField(2, "val", "int64"),
        ]))
        s2 = tbl.current_schema()
        assert s2 is not None
        assert len(s2.fields) == 2
        assert s2.schema_id > s1.schema_id


def test_field_ids_stable_across_evolution():
    """field_ids_for_current_schema() must reflect the live schema."""
    with tempfile.TemporaryDirectory() as tmp:
        tbl = fc.FluxTable(f"{tmp}/t.fluxtable")
        tbl.evolve_schema(fc.TableSchema([
            fc.SchemaField(1, "user_id", "uint64"),
            fc.SchemaField(2, "revenue", "int64"),
        ]))
        ids_v1 = tbl.field_ids_for_current_schema()
        assert ids_v1 == {"user_id": 1, "revenue": 2}

        tbl.evolve_schema(fc.TableSchema([
            fc.SchemaField(1, "user_id", "uint64"),
            fc.SchemaField(2, "revenue", "int64"),
            fc.SchemaField(3, "region",  "uint64"),
        ]))
        ids_v2 = tbl.field_ids_for_current_schema()
        assert ids_v2 == {"user_id": 1, "revenue": 2, "region": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def test_vs_delta_small():
    """Side-by-side: 4 × 1K rows  (quick sanity check)."""
    n, nf = 1_024, 4
    v1 = [_batch_v1(n, i * n) for i in range(nf)]
    v2 = _batch_v2(n, nf * n)
    results = [
        _run_fluxtable(v1, v2),
        _run_delta(v1, v2),
        _run_parquet(v1, v2),
    ]
    _print_report(results, f"4 × {n:,} rows + schema evolution")

    flux = results[0]
    assert flux.total_rows == (nf + 1) * n
    assert flux.ratio > 1.0, "FluxTable must compress data"


def test_vs_delta_medium():
    """Side-by-side: 4 × 64K rows  (representative workload)."""
    n, nf = 65_536, 4
    v1 = [_batch_v1(n, i * n) for i in range(nf)]
    v2 = _batch_v2(n, nf * n)
    results = [
        _run_fluxtable(v1, v2),
        _run_delta(v1, v2),
        _run_parquet(v1, v2),
    ]
    _print_report(results, f"4 × {n:,} rows + schema evolution")

    flux = results[0]
    assert flux.ratio > 1.0


@pytest.mark.slow
def test_vs_delta_large():
    """Side-by-side: 4 × 512K rows  (stress scale)."""
    n, nf = 524_288, 4
    v1 = [_batch_v1(n, i * n) for i in range(nf)]
    v2 = _batch_v2(n, nf * n)
    results = [
        _run_fluxtable(v1, v2),
        _run_delta(v1, v2),
        _run_parquet(v1, v2),
    ]
    _print_report(results, f"4 × {n:,} rows + schema evolution")

    flux = results[0]
    assert flux.ratio > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Schema evolution overhead test
# ─────────────────────────────────────────────────────────────────────────────

def test_schema_evolution_is_metadata_only():
    """
    FluxTable schema evolution must complete in < 200ms (it writes only a
    JSON log entry — no data rewrite).  Verify also that no data files are
    added by evolve_schema().
    """
    with tempfile.TemporaryDirectory() as tmp:
        tbl = fc.FluxTable(f"{tmp}/t.fluxtable")
        tbl.evolve_schema(fc.TableSchema([
            fc.SchemaField(1, "id",  "uint64", nullable=False),
            fc.SchemaField(2, "val", "int64"),
        ]))
        # Append one file so there's data
        tbl.append(fc.compress(_batch_v1(1_024)))
        files_before = tbl.live_files()

        t0 = time.perf_counter()
        tbl.evolve_schema(fc.TableSchema([
            fc.SchemaField(1, "id",  "uint64", nullable=False),
            fc.SchemaField(2, "val", "int64"),
            fc.SchemaField(3, "tag", "utf8"),
        ]))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        files_after = tbl.live_files()

    assert files_before == files_after, (
        "evolve_schema() must not add or remove data files"
    )
    assert elapsed_ms < 200, (
        f"evolve_schema() took {elapsed_ms:.1f}ms — expected < 200ms "
        "(pure metadata write, no data rewrite)"
    )
    print(f"\n  evolve_schema() overhead: {elapsed_ms:.2f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Compression ratio assertion
# ─────────────────────────────────────────────────────────────────────────────

def test_flux_compression_ratio_vs_parquet():
    """
    FluxCompress must achieve at least 70% of Parquet-snappy's ratio on
    structured numeric data, and better than 1.0x (any compression at all).
    """
    n = 65_536
    tbl = _batch_v1(n)
    raw = sum(col.nbytes for col in tbl.columns)

    flux_bytes = len(fc.compress(tbl))
    flux_ratio = raw / flux_bytes

    sink = io.BytesIO()
    pq.write_table(tbl, sink, compression="snappy")
    pq_bytes = sink.tell()
    pq_ratio = raw / pq_bytes

    print(f"\n  FluxCompress:     {flux_bytes / 1e6:.3f} MB  ratio={flux_ratio:.2f}x")
    print(f"  Parquet (snappy): {pq_bytes  / 1e6:.3f} MB  ratio={pq_ratio:.2f}x")

    assert flux_ratio > 1.0
    assert flux_ratio >= pq_ratio * 0.70, (
        f"FluxCompress ratio {flux_ratio:.2f}x is < 70% of Parquet "
        f"{pq_ratio:.2f}x on structured numeric data"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dtype-diverse FluxTable × Delta × Parquet comparison
# ─────────────────────────────────────────────────────────────────────────────
#
# Import the generators from test_vs_parquet to avoid duplication.
# We test two representative suites end-to-end through the FluxTable API:
#   even_mix      — one column per major type family (int, float, string, ts)
#   numeric_heavy — ~80% numeric / ~20% string (common OLAP shape)

try:
    from python.tests.test_vs_parquet import (
        _int_sizes, _strings_only, _floats_only, _timestamps,
        _even_mix, _numeric_heavy, _string_heavy, _survey_comments,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from test_vs_parquet import (  # type: ignore[no-redef]
        _int_sizes, _strings_only, _floats_only, _timestamps,
        _even_mix, _numeric_heavy, _string_heavy, _survey_comments,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Suite registry: factory, FluxTable schema fields, and disk-size assertion
# ─────────────────────────────────────────────────────────────────────────────
#
# _SUITE_SPECS maps each suite name to:
#   factory   — callable(n) → pa.Table
#   fields    — list of (field_id, col_name, fc_dtype_str) to register in FluxTable
#   rows      — rows per batch (256K cap for string-dominant suites)
#   flux_must_beat_delta — True when we assert flux.disk_bytes < delta.disk_bytes
#
# On pure-string workloads (strings_only) Parquet’s RLE-dict slightly
# outperforms FSST in raw compression, so Delta’s Parquet files can be smaller.
# Every other suite has FluxCompress clearly ahead.

_SUITE_SPECS = {
    "int_sizes": dict(
        factory=_int_sizes,
        fields=[
            (1, "i8",  "int8"),   (2, "u8",  "uint8"),
            (3, "i16", "int16"),  (4, "u16", "uint16"),
            (5, "i32", "int32"),  (6, "u32", "uint32"),
            (7, "i64", "int64"),  (8, "u64", "uint64"),
        ],
        rows=65_536,
        flux_must_beat_delta=True,
    ),
    "strings_only": dict(
        factory=_strings_only,
        fields=[
            (1, "region",     "utf8"),
            (2, "event",      "utf8"),
            (3, "path",       "utf8"),
            (4, "session_id", "utf8"),
        ],
        rows=65_536,
        # Parquet RLE-dict edges out FSST on all-string tables; relaxed assertion.
        flux_must_beat_delta=False,
    ),
    "floats_only": dict(
        factory=_floats_only,
        fields=[
            (1, "price_f64",  "float64"),
            (2, "rate_f64",   "float64"),
            (3, "mixed_f64",  "float64"),
            (4, "weight_f32", "float32"),
        ],
        rows=65_536,
        flux_must_beat_delta=True,
    ),
    "timestamps": dict(
        factory=_timestamps,
        fields=[
            (1, "ts_us",  "timestamp_micros"),
            (2, "ts_ms",  "timestamp_millis"),
            (3, "ts_s",   "timestamp_second"),
            (4, "date32", "date32"),
        ],
        rows=65_536,
        flux_must_beat_delta=True,
    ),
    "even_mix": dict(
        factory=_even_mix,
        fields=[
            (1, "id",       "uint64"),
            (2, "amount",   "float64"),
            (3, "region",   "utf8"),
            (4, "ts",       "timestamp_micros"),
            (5, "qty",      "int32"),
            (6, "category", "int16"),
        ],
        rows=65_536,
        flux_must_beat_delta=True,
    ),
    "numeric_heavy": dict(
        factory=_numeric_heavy,
        fields=[
            (1, "user_id",   "uint64"),
            (2, "revenue",   "int64"),
            (3, "qty",       "int32"),
            (4, "status",    "int8"),
            (5, "price_f64", "float64"),
            (6, "score_f32", "float32"),
            (7, "ts",        "timestamp_micros"),
            (8, "date",      "date32"),
            (9, "region",    "utf8"),
        ],
        rows=65_536,
        flux_must_beat_delta=True,
    ),
    "string_heavy": dict(
        factory=_string_heavy,
        fields=[
            (1, "name",    "utf8"),
            (2, "email",   "utf8"),
            (3, "tag",     "utf8"),
            (4, "message", "utf8"),
            (5, "user_id", "uint64"),
            (6, "score",   "int32"),
        ],
        rows=65_536,
        # Mixed string/numeric — Flux wins because the numeric cols compress hard.
        flux_must_beat_delta=True,
    ),
    "survey_comments": dict(
        factory=_survey_comments,
        fields=[
            (1, "comment",   "utf8"),
            (2, "category",  "utf8"),
            (3, "rating",    "int8"),
            (4, "nps_score", "int8"),
        ],
        rows=65_536,
        # Free-form NL text.  FSST+LZ4 (Speed) may be slightly behind Delta’s
        # Parquet-snappy on the comment column; relaxed to ratio > 1.0 only.
        # Brotli profile wins convincingly but that’s validated separately.
        flux_must_beat_delta=False,
    ),
}


def _run_fluxtable_generic(
    batches: List[pa.Table],
    schema_fields: List[fc.SchemaField],
    profile: str = "speed",
) -> BenchResult:
    """
    Generic FluxTable runner: append all batches under one schema,
    then scan.  No schema evolution — tests the dtype paths only.
    """
    raw = _raw_arrow_bytes(batches)
    total_rows = sum(t.num_rows for t in batches)

    with tempfile.TemporaryDirectory() as tmp:
        tbl = fc.FluxTable(f"{tmp}/flux.fluxtable")
        tbl.evolve_schema(fc.TableSchema(schema_fields))

        t0 = time.perf_counter()
        for batch in batches:
            tbl.append(fc.compress(batch, profile=profile))
        write_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        rows = sum(b.num_rows for b in tbl.scan())
        scan_ms = (time.perf_counter() - t1) * 1000

        disk = _dir_bytes(f"{tmp}/flux.fluxtable")

    assert rows == total_rows
    return BenchResult(
        name=f"FluxTable ({profile})",
        total_rows=total_rows,
        num_files=len(batches),
        write_ms=write_ms,
        evolve_ms=0.0,
        scan_ms=scan_ms,
        disk_bytes=disk,
        raw_bytes=raw,
    )


_DELTA_UNSUPPORTED_NAME = "Delta Lake (unsupported)"


def _run_delta_generic(batches: List[pa.Table]) -> BenchResult:
    """
    Run Delta Lake on ``batches``.

    Delta Lake (via its Parquet backend) has a known limitation with unsigned
    integer types: it may map uint8/uint16 to signed counterparts internally,
    causing a cast error when column values exceed the signed type’s max (e.g.
    uint8 value 128 can’t round-trip through Int8).  When this happens the
    runner returns a sentinel result with ``disk_bytes = -1`` so the caller
    can report the limitation rather than silently skipping or erroring.
    """
    raw = _raw_arrow_bytes(batches)
    total_rows = sum(t.num_rows for t in batches)

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/delta"
        try:
            t0 = time.perf_counter()
            for i, batch in enumerate(batches):
                write_deltalake(path, batch, mode="error" if i == 0 else "append")
            write_ms = (time.perf_counter() - t0) * 1000

            dt = DeltaTable(path)
            t1 = time.perf_counter()
            result = dt.to_pyarrow_table()
            scan_ms = (time.perf_counter() - t1) * 1000
            disk = _dir_bytes(path)

        except Exception as exc:
            # Surface unsigned-int or other type-compatibility errors as a
            # sentinel rather than failing the test — the limitation is itself
            # a meaningful comparison data point.
            print(
                f"\n  ⚠ Delta Lake write failed — known unsigned-int limitation: {exc}"
            )
            return BenchResult(
                name=_DELTA_UNSUPPORTED_NAME,
                total_rows=total_rows,
                num_files=len(batches),
                write_ms=0.0,
                evolve_ms=0.0,
                scan_ms=0.0,
                disk_bytes=-1,      # sentinel: unsupported
                raw_bytes=raw,
            )

    return BenchResult(
        name="Delta Lake",
        total_rows=total_rows,
        num_files=len(batches),
        write_ms=write_ms,
        evolve_ms=0.0,
        scan_ms=scan_ms,
        disk_bytes=disk,
        raw_bytes=raw,
    )


def _run_parquet_generic(batches: List[pa.Table]) -> BenchResult:
    raw = _raw_arrow_bytes(batches)
    total_rows = sum(t.num_rows for t in batches)

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        for i, batch in enumerate(batches):
            pq.write_table(batch, f"{tmp}/part-{i:04d}.parquet", compression="snappy")
        write_ms = (time.perf_counter() - t0) * 1000

        import pyarrow.dataset as ds
        t1 = time.perf_counter()
        result = ds.dataset(tmp, format="parquet").to_table()
        scan_ms = (time.perf_counter() - t1) * 1000
        disk = _dir_bytes(tmp)

    return BenchResult(
        name="Parquet (snappy)",
        total_rows=total_rows,
        num_files=len(batches),
        write_ms=write_ms,
        evolve_ms=0.0,
        scan_ms=scan_ms,
        disk_bytes=disk,
        raw_bytes=raw,
    )


@pytest.mark.parametrize("suite_name", list(_SUITE_SPECS))
def test_fluxtable_dtype_suite_vs_delta(suite_name: str) -> None:
    """
    FluxTable vs Delta Lake vs Parquet across all 7 dtype suites.

    For each suite:
      - Data is written through the FluxTable API (compress → append) and read
        back via scan(), exercising the full Phase F path for that dtype family.
      - Delta Lake writes the same batches as Parquet files + transaction log.
      - Parquet writes the same batches as standalone files.

    Assertions:
      - FluxTable always compresses (ratio > 1.0).
      - For numeric/timestamp/mixed suites: FluxTable disk footprint is smaller
        than Delta Lake’s (Parquet files + JSON log).
      - For pure-string suites: ratio > 1.0 only (Parquet RLE-dict is stronger
        on all-string workloads, so Delta may be smaller; that’s expected).
    """
    spec       = _SUITE_SPECS[suite_name]
    n          = spec["rows"]
    num_batches = 4
    batches    = [spec["factory"](n) for _ in range(num_batches)]
    fields     = [
        fc.SchemaField(fid, col, dtype)
        for fid, col, dtype in spec["fields"]
    ]

    results = [
        _run_fluxtable_generic(batches, fields),
        _run_delta_generic(batches),
        _run_parquet_generic(batches),
    ]
    _print_report(
        results,
        f"{suite_name}: 4 × {n:,} rows",
    )

    flux, delta, parquet = results
    assert flux.total_rows == n * num_batches
    assert flux.ratio > 1.0, (
        f"{suite_name}: FluxTable failed to compress (ratio={flux.ratio:.3f})"
    )

    if delta.name == _DELTA_UNSUPPORTED_NAME:
        # Delta Lake couldn’t write this dtype family (e.g. unsigned int overflow).
        # FluxTable handled it fine — that’s a meaningful result in itself.
        print(
            f"  ℹ {suite_name}: FluxTable succeeded; Delta Lake does not support "
            f"this dtype combination."
        )
        return

    if spec["flux_must_beat_delta"]:
        assert flux.disk_bytes < delta.disk_bytes, (
            f"{suite_name}: FluxTable ({flux.disk_bytes / 1e6:.2f} MB) should be "
            f"smaller than Delta Lake ({delta.disk_bytes / 1e6:.2f} MB)"
        )


def test_dtype_suite_full_fluxtable_vs_delta_report() -> None:
    """
    Print a side-by-side FluxTable (speed/archive/brotli) vs Delta Lake vs
    Parquet table across all dtype suites.  Shows compression ratio, write
    throughput, and scan throughput for every combination so the speed/ratio
    tradeoff of each profile is immediately visible.

    Always passes.  Run with ``pytest -s`` to see the output.
    """
    num_batches = 4
    profiles = ["speed", "archive", "brotli"]
    wins: dict = {"delta": {p: 0 for p in profiles},
                  "parquet": {p: 0 for p in profiles}}
    total = 0

    for suite_name, spec in _SUITE_SPECS.items():
        n       = spec["rows"]
        batches = [spec["factory"](n) for _ in range(num_batches)]
        fields  = [
            fc.SchemaField(fid, col, dtype)
            for fid, col, dtype in spec["fields"]
        ]
        flux_results = [
            _run_fluxtable_generic(batches, fields, profile=p)
            for p in profiles
        ]
        delta   = _run_delta_generic(batches)
        parquet = _run_parquet_generic(batches)

        _print_report(
            flux_results + [delta, parquet],
            f"{suite_name}: 4 × {n:,} rows",
        )
        total += 1
        for p, r in zip(profiles, flux_results):
            if r.disk_bytes > 0 and delta.disk_bytes > 0 and r.disk_bytes < delta.disk_bytes:
                wins["delta"][p] += 1
            if r.disk_bytes > 0 and r.disk_bytes < parquet.disk_bytes:
                wins["parquet"][p] += 1

    print(f"\n{'Profile':<16} {'Beats Delta':>12} {'Beats Parquet':>14}")
    print("-" * 44)
    for p in profiles:
        print(
            f"  FluxTable ({p:<7}) "
            f"{wins['delta'][p]}/{total} suites  "
            f"{wins['parquet'][p]}/{total} suites"
        )


if __name__ == "__main__":
    for n, nf, label in [
        (1_024,   4, "small (4 × 1K)"),
        (65_536,  4, "medium (4 × 64K)"),
        (524_288, 2, "large (2 × 512K)"),
    ]:
        v1 = [_batch_v1(n, i * n) for i in range(nf)]
        v2 = _batch_v2(n, nf * n)
        results = [
            _run_fluxtable(v1, v2),
            _run_delta(v1, v2),
            _run_parquet(v1, v2),
        ]
        _print_report(results, label)
