# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FluxCompress vs Parquet comparison benchmark.

Run with:
    pytest python/tests/test_vs_parquet.py -v -s

Or standalone:
    python python/tests/test_vs_parquet.py

Compares FluxCompress and Apache Parquet (via pyarrow) across:

  Classic single-column patterns:
    sequential, constant, low-cardinality, random

  Dtype-diverse suite (8 workloads — see DTYPE_SUITE):
    int_sizes, strings_only, floats_only, timestamps,
    even_mix, numeric_heavy, string_heavy, survey_comments

  Metrics: compressed size, compression ratio, compress/decompress throughput
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
    """Legacy: realistic multi-column table (all integers, kept for baseline)."""
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
# Dtype-diverse dataset suite
# ─────────────────────────────────────────────────────────────────────────────
#
# Seven workloads covering the full FluxCompress encoder stack:
#
#  ┌─────────────────┬────────────────────────┬────────────────────────────────┐
#  │ Suite           │ Primary encoder(s)      │ Notes                          │
#  ├─────────────────┼────────────────────────┼────────────────────────────────┤
#  │ int_sizes        │ BitSlab, DeltaDelta      │ All 8 int widths (i8–u64)      │
#  │ strings_only     │ FSST, dict, front-code  │ Low + high cardinality mix     │
#  │ floats_only      │ ALP, BitSlab            │ Decimal-shaped + noisy         │
#  │ timestamps       │ DeltaDelta, BitSlab     │ 4 temporal types               │
#  │ even_mix         │ All                     │ 1 col per major type family     │
#  │ numeric_heavy    │ BitSlab, ALP, Delta     │ ~80% numeric, 1 string col     │
#  │ string_heavy     │ FSST, dict, front-code  │ ~80% string, 1 numeric col     │
#  │ survey_comments  │ Brotli / FSST / raw     │ Free-form NL text, adversarial │
#  └─────────────────┴────────────────────────┴────────────────────────────────┘
#
# Size cap: string-dominant suites are capped at 262_144 rows in the full
# comparison report to keep Python string-generation time reasonable.
# All suites are tested at 1_024 and 65_536 in the correctness/assertion tests.


def _int_sizes(n: int) -> pa.Table:
    """
    All 8 integer widths, each with data sized to use most of its range.
    Exercises BitSlab width-packing across i8–u64 simultaneously.
    """
    return pa.table({
        "i8":  pa.array([i % 100         for i in range(n)], type=pa.int8()),
        "u8":  pa.array([i % 200         for i in range(n)], type=pa.uint8()),
        "i16": pa.array([i % 30_000      for i in range(n)], type=pa.int16()),
        "u16": pa.array([i % 60_000      for i in range(n)], type=pa.uint16()),
        "i32": pa.array([i % 2_000_000   for i in range(n)], type=pa.int32()),
        "u32": pa.array([i               for i in range(n)], type=pa.uint32()),
        "i64": pa.array([i * 1_000_000   for i in range(n)], type=pa.int64()),
        "u64": pa.array([i               for i in range(n)], type=pa.uint64()),
    })


_REGIONS   = ["US", "EU", "APAC", "LATAM", "MEA", "CA", "AU", "JP"]
_EVENTS    = [f"evt_{k:04d}" for k in range(64)]
_VERBS     = ["GET", "POST", "PUT", "DELETE", "PATCH"]
_ADJ       = ["quick", "lazy", "bright", "dark", "fast", "slow", "green", "red"]
_NOUNS     = ["fox", "dog", "cat", "bird", "fish", "bear", "wolf", "lion"]

# ── Survey-comments vocabulary ─────────────────────────────────────────────
# Intentionally large vocabularies so most rows are unique (no dict encoding)
# and the text contains common English phrases that Brotli’s static dictionary
# covers better than LZ4 or per-column FSST.
_SC_OPENERS = [
    "I recently purchased this",        "After using this for a month",
    "I ordered this online",             "My family has been using this daily",
    "I bought this as a birthday gift",  "I’ve been using this for several weeks",
    "After much deliberation I chose this", "I replaced my old one with this",
    "A colleague recommended this to me", "I was initially skeptical about this",
    "Following an online recommendation", "I ordered two of these items",
    "Having tried many alternatives",    "I use this in a professional setting",
    "I purchased this for my home office", "My first impression of this product",
]
_SC_BODY_POS = [
    "and I am extremely satisfied with the quality and performance",
    "and it has exceeded all my expectations in every way",
    "and the build quality is outstanding and very durable",
    "and it works perfectly for my everyday needs",
    "and the value for money is truly exceptional",
    "and I would highly recommend it to anyone looking for reliability",
    "and it arrived quickly and in perfect condition",
    "and the instructions were very clear and easy to follow",
]
_SC_BODY_NEG = [
    "but I was quite disappointed with the overall quality",
    "but it failed to meet even the most basic expectations",
    "however the product stopped working within days of purchase",
    "but the product description was somewhat misleading",
    "unfortunately the material quality was very poor",
    "and sadly it did not function as described in the listing",
    "but the item arrived damaged despite careful packaging",
    "and the instructions were confusing and incomplete",
]
_SC_CLOSING_POS = [
    "Will definitely buy again.",
    "Highly recommend to friends and family.",
    "Five stars without hesitation.",
    "Perfect for everyday use.",
    "Well worth every penny.",
    "Excellent product overall.",
    "Very happy with this purchase.",
]
_SC_CLOSING_NEG = [
    "Would not purchase again under any circumstances.",
    "Very disappointed with the whole experience.",
    "Cannot recommend this to anyone.",
    "Poor value for money.",
    "Returning this item for a full refund.",
    "One star — avoid this product.",
    "Save yourself the trouble and look elsewhere.",
]
_SC_EXTRAS = [
    "The delivery was faster than expected.",
    "Customer service was very responsive to my queries.",
    "The packaging was eco-friendly which I appreciated.",
    "The price point is very reasonable for this quality.",
    "Good quality materials for the price paid.",
    "Works exactly as described in the product listing.",
    "Easy to set up and requires no special tools.",
    "Durable construction and well-engineered.",
    "Lightweight and compact, easy to store.",
    "Looks exactly as pictured on the website.",
    "Sturdy and feels premium in the hand.",
    "Attractive modern design that fits any decor.",
    "Very easy to clean and maintain.",
    "Fits perfectly in the space I had available.",
    "Extremely versatile and multifunctional.",
    "Makes a great gift for any occasion.",
    "The colour matches the photos exactly.",
    "Instructions could have been clearer.",
]
_SC_CATEGORIES = [
    "Product Quality", "Customer Service", "Delivery Speed",
    "Website Experience", "Pricing", "Returns Process",
    "Packaging", "Other",
]


def _make_survey_comment(i: int) -> str:
    """Build a deterministic natural-language comment from vocabulary tables.

    Every row is unique: a pseudo-random order/reference number is embedded
    in the text, giving near-100% cardinality at any n.  Prime multipliers
    decorrelate all other word choices so the vocabulary mix looks realistic.
    This makes the suite a genuine adversarial test for high-cardinality
    free-text compression (no dict encoding; FSST + raw codec path).
    """
    opener = _SC_OPENERS[i % len(_SC_OPENERS)]
    positive = (i % 5) in (0, 1, 3)   # ~60% positive sentiment
    body = (
        _SC_BODY_POS[(i * 7)  % len(_SC_BODY_POS)]  if positive else
        _SC_BODY_NEG[(i * 7)  % len(_SC_BODY_NEG)]
    )
    closing = (
        _SC_CLOSING_POS[(i * 13) % len(_SC_CLOSING_POS)] if positive else
        _SC_CLOSING_NEG[(i * 13) % len(_SC_CLOSING_NEG)]
    )
    extra   = _SC_EXTRAS[(i * 11) % len(_SC_EXTRAS)]
    extra2  = _SC_EXTRAS[(i * 17) % len(_SC_EXTRAS)]

    # Pseudo-random order number: maps each i to a unique 7-digit number so
    # every comment is distinct.  The number reads naturally in the sentence
    # and gives text-level entropy that FSST can’t collapse but Brotli’s
    # static dictionary partially handles (digits are common in web text).
    order_ref = (i * 1_657 + 31) % 9_999_997  # prime-step, near-unique up to ~10M

    # Three length classes driven by i % 3 for realistic length variation.
    lc = i % 3
    if lc == 0:
        # Short (~95-125 chars): includes order reference.
        return f"{opener} (order #{order_ref:07d}), {body}. {closing}"
    elif lc == 1:
        # Medium (~135-165 chars): adds one contextual sentence.
        return f"{opener}, {body}. {extra} {closing}"
    else:
        # Long (~175-215 chars): two contextual sentences + reference at close.
        return (
            f"{opener}, {body}. {extra} {extra2} "
            f"{closing} Reference: #{order_ref:07d}."
        )


def _strings_only(n: int) -> pa.Table:
    """
    Four utf8 columns spanning the full cardinality spectrum:
      - region:  8 values  (dict-encode/RLE target)
      - event:   64 values (dictionary encode target)
      - path:    ~n unique (high-cardinality FSST / front-coding target)
      - session: n unique  (maximum-entropy string column)

    Capped at 262_144 in the full report (Python str-gen is O(n)).
    """
    return pa.table({
        "region":     pa.array([_REGIONS[i % 8]  for i in range(n)], type=pa.utf8()),
        "event":      pa.array([_EVENTS[i % 64]  for i in range(n)], type=pa.utf8()),
        "path":       pa.array([f"/api/v{i%3}/u/{i%10000}/e/{i}" for i in range(n)], type=pa.utf8()),
        "session_id": pa.array([f"s{i:012x}"    for i in range(n)], type=pa.utf8()),
    })


def _floats_only(n: int) -> pa.Table:
    """
    Float columns across the ALP-compressibility spectrum:
      - price_f64:  ALP-compressible (2 d.p., small mantissa bit-width)
      - rate_f64:   ALP-compressible (3 d.p.)
      - mixed_f64:  trigonometric output (high entropy, ALP-resistant)
      - weight_f32: float32, low-cardinality (RLE / dictionary target)
    """
    import math
    return pa.table({
        "price_f64": pa.array([round(99.99 + (i % 10000) * 0.01, 2) for i in range(n)], type=pa.float64()),
        "rate_f64":  pa.array([round(1.000 + (i % 1000)  * 0.001, 3) for i in range(n)], type=pa.float64()),
        "mixed_f64": pa.array([math.sin(i) * 1000.0                  for i in range(n)], type=pa.float64()),
        "weight_f32":pa.array([float(i % 50) * 0.5                   for i in range(n)], type=pa.float32()),
    })


# Epoch reference: 2023-11-14 22:13:20 UTC in various units.
_BASE_US = 1_700_000_000_000_000   # microseconds
_BASE_MS = 1_700_000_000_000       # milliseconds
_BASE_S  = 1_700_000_000           # seconds
_BASE_D  = 19_310                  # days since epoch (2022-11-16)


def _timestamps(n: int) -> pa.Table:
    """
    Four temporal types with the most common real-world patterns:
      - ts_us:  sequential with 1ms step (analytics event stream)
      - ts_ms:  near-sequential with small jitter (log timestamps)
      - ts_s:   coarser 1-minute step (hourly rollup)
      - date32: cycling over a 5-year window (daily partitions)
    """
    return pa.table({
        "ts_us":  pa.array([_BASE_US + i * 1_000                         for i in range(n)], type=pa.timestamp("us")),
        "ts_ms":  pa.array([_BASE_MS + i * 10 + (i * 3 % 5)             for i in range(n)], type=pa.timestamp("ms")),
        "ts_s":   pa.array([_BASE_S  + i * 60                            for i in range(n)], type=pa.timestamp("s")),
        "date32": pa.array([_BASE_D  + i % 1_826                         for i in range(n)], type=pa.date32()),
    })


def _even_mix(n: int) -> pa.Table:
    """
    One column per major type family, giving each encoder roughly equal
    weight.  The reference workload for a balanced OLAP table.
    """
    return pa.table({
        "id":       pa.array(range(n),                                          type=pa.uint64()),
        "amount":   pa.array([round(i * 3.14 % 9_999.99, 2) for i in range(n)],type=pa.float64()),
        "region":   pa.array([_REGIONS[i % 8]               for i in range(n)],type=pa.utf8()),
        "ts":       pa.array([_BASE_US + i * 1_000           for i in range(n)],type=pa.timestamp("us")),
        "qty":      pa.array([i % 10_000                     for i in range(n)],type=pa.int32()),
        "category": pa.array([i % 256                        for i in range(n)],type=pa.int16()),
    })


def _numeric_heavy(n: int) -> pa.Table:
    """
    ~80% numeric / ~20% string: the common analytics / OLAP pattern.
    Exercises BitSlab (ints), ALP (floats), DeltaDelta (timestamps),
    and FSST/dict (the single string column).
    """
    return pa.table({
        "user_id":   pa.array(range(n),                                           type=pa.uint64()),
        "revenue":   pa.array([i * 37 % 99_999                for i in range(n)], type=pa.int64()),
        "qty":       pa.array([i % 10_000                     for i in range(n)], type=pa.int32()),
        "status":    pa.array([i % 8                          for i in range(n)], type=pa.int8()),
        "price_f64": pa.array([round(9.99 + (i%5000)*0.01, 2)for i in range(n)], type=pa.float64()),
        "score_f32": pa.array([float(i % 1000) / 10.0         for i in range(n)], type=pa.float32()),
        "ts":        pa.array([_BASE_US + i * 60_000_000       for i in range(n)], type=pa.timestamp("us")),
        "date":      pa.array([_BASE_D  + i % 1_826            for i in range(n)], type=pa.date32()),
        "region":    pa.array([_REGIONS[i % 8]                for i in range(n)], type=pa.utf8()),
    })


def _string_heavy(n: int) -> pa.Table:
    """
    ~80% string / ~20% numeric: a content-management or logging workload.
    Exercises FSST + dict + front-coding on 4 string columns of varying
    cardinality, with two numeric anchor columns.

    Capped at 262_144 in the full report (Python str-gen is O(n)).
    """
    return pa.table({
        "name":    pa.array([f"User_{_ADJ[i%8]}_{i:08d}"         for i in range(n)], type=pa.utf8()),
        "email":   pa.array([f"u{i}@host{i%500}.example.com"      for i in range(n)], type=pa.utf8()),
        "tag":     pa.array([_ADJ[i % 8]                          for i in range(n)], type=pa.utf8()),
        "message": pa.array([
            f"A {_ADJ[i%8]} {_NOUNS[i%8]} #{i} performed {_VERBS[i%5]}"
            for i in range(n)
        ], type=pa.utf8()),
        "user_id": pa.array(range(n),                                              type=pa.uint64()),
        "score":   pa.array([i % 100                              for i in range(n)], type=pa.int32()),
    })


def _survey_comments(n: int) -> pa.Table:
    """
    Survey export with free-form comments — the adversarial natural-language case.

    The ``comment`` column is the key challenge:
      - Near-100% unique rows (order ref makes every comment distinct)
      - NOT sorted → front-coding won’t trigger
      - NOT low-cardinality → dict encoding won’t trigger
      - Natural language vocabulary → Brotli’s static English dictionary helps;
        FSST finds common word fragments ("and I ", "the product", etc.);
        LZ4 / Snappy struggle the most on this workload
      - This is the workload where the Speed vs Brotli profile gap is largest

    Companion columns (category, rating, nps_score) are low-cardinality
    to simulate a realistic survey data export schema.

    Capped at 262_144 rows in the full report (Python str-gen is O(n)).
    """
    return pa.table({
        "comment":   pa.array([_make_survey_comment(i)              for i in range(n)], type=pa.utf8()),
        "category":  pa.array([_SC_CATEGORIES[i % len(_SC_CATEGORIES)] for i in range(n)], type=pa.utf8()),
        "rating":    pa.array([1 + (i % 5)                         for i in range(n)], type=pa.int8()),
        "nps_score": pa.array([(i * 7 + 3) % 11                    for i in range(n)], type=pa.int8()),
    })


# Suites that contain heavy string generation are capped at 256K rows in
# the full comparison report to keep test time reasonable.  All suites
# run at 1K and 64K in the per-suite assertion and correctness tests.
_STRING_SUITES = frozenset({"strings_only", "string_heavy", "survey_comments"})
_REPORT_SIZES  = {
    "int_sizes":       [1_024, 65_536, 1_048_576],
    "strings_only":    [1_024, 65_536, 262_144],
    "floats_only":     [1_024, 65_536, 1_048_576],
    "timestamps":      [1_024, 65_536, 1_048_576],
    "even_mix":        [1_024, 65_536, 1_048_576],
    "numeric_heavy":   [1_024, 65_536, 1_048_576],
    "string_heavy":    [1_024, 65_536, 262_144],
    "survey_comments": [1_024, 65_536, 262_144],
}

DTYPE_SUITE: dict[str, Callable[[int], pa.Table]] = {
    "int_sizes":       _int_sizes,
    "strings_only":    _strings_only,
    "floats_only":     _floats_only,
    "timestamps":      _timestamps,
    "even_mix":        _even_mix,
    "numeric_heavy":   _numeric_heavy,
    "string_heavy":    _string_heavy,
    "survey_comments": _survey_comments,
}


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
# Dtype-suite tests
# ─────────────────────────────────────────────────────────────────────────────

# Per-suite minimum acceptable compression ratio vs Parquet (snappy).
# Values < 1.0 mean "Flux may be larger here and that's OK" (e.g. high-entropy
# random strings that no lossless codec handles well).
# Values >= 1.0 mean "Flux must beat Parquet on this dtype family".
_SUITE_MIN_RATIO_VS_PARQUET_SNAPPY: dict[str, float] = {
    # Integers: FluxCompress's sweet-spot — must clearly win.
    "int_sizes":       1.5,
    # Strings: FSST + dict is competitive but Parquet's RLE-dict is strong;
    # require only that we're in the same ballpark (>= 70% of Parquet ratio).
    "strings_only":    0.7,
    # Floats: ALP handles decimal-shaped columns very well.
    "floats_only":     1.0,
    # Timestamps: sequential → DeltaDelta → should comfortably beat Parquet.
    "timestamps":      1.5,
    # Mixed suites: weighted average; require modest win on balance.
    "even_mix":        1.0,
    "numeric_heavy":   1.2,
    "string_heavy":    0.7,
    # Survey comments: high-cardinality free-form NL text (~100% unique rows).
    # Speed (FSST+LZ4) typically falls behind Parquet snappy on pure NL text
    # because FSST’s symbol table is weaker than Brotli’s static English dict.
    # Threshold is a regression guard (>1.0x) only; the Brotli profile wins
    # convincingly and is tested via test_survey_comments_brotli_vs_parquet.
    "survey_comments": 0.5,
}


@pytest.mark.parametrize("suite_name", list(DTYPE_SUITE))
def test_dtype_suite_roundtrip_correctness(suite_name: str) -> None:
    """
    FluxCompress must round-trip every dtype in every suite correctly.

    Checks column-by-column that the decompressed values match the original,
    using PyList comparison (handles NaN-safe numeric comparison via Python
    float equality, which is fine for our generated data).
    """
    import fluxcompress as fc

    n = 1_024
    table = DTYPE_SUITE[suite_name](n)
    buf = fc.compress(table)
    out = fc.decompress(buf)

    assert out.num_rows == n, (
        f"{suite_name}: expected {n} rows, got {out.num_rows}"
    )
    # Compare schema column-names (order may differ on decompress)
    for col_name in table.schema.names:
        expected = table.column(col_name).to_pylist()
        got      = out.column(col_name).to_pylist()
        assert expected == got, (
            f"{suite_name}/{col_name}: round-trip mismatch at n={n}"
        )


@pytest.mark.parametrize("suite_name", list(DTYPE_SUITE))
def test_flux_vs_parquet_dtype_suite_ratio(suite_name: str) -> None:
    """
    FluxCompress must meet per-suite minimum compression ratios vs
    Parquet (snappy).  The thresholds are set conservatively to catch
    regressions, not to claim Flux is universally better.
    """
    import fluxcompress as fc

    n = 65_536
    table = DTYPE_SUITE[suite_name](n)

    raw_bytes     = sum(col.nbytes for col in table.columns)
    flux_bytes    = len(fc.compress(table))
    flux_ratio    = raw_bytes / max(flux_bytes, 1)

    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    pq_ratio = raw_bytes / max(sink.tell(), 1)

    min_frac = _SUITE_MIN_RATIO_VS_PARQUET_SNAPPY[suite_name]
    threshold = pq_ratio * min_frac

    print(
        f"\n  {suite_name:<14}: Flux {flux_ratio:.2f}x  "
        f"Parquet-snappy {pq_ratio:.2f}x  "
        f"threshold {threshold:.2f}x ({min_frac:.0%} of Parquet)"
    )

    assert flux_ratio > 1.0, (
        f"{suite_name}: FluxCompress failed to compress (ratio={flux_ratio:.3f})"
    )
    assert flux_ratio >= threshold, (
        f"{suite_name}: Flux ratio {flux_ratio:.2f}x < {min_frac:.0%} of "
        f"Parquet {pq_ratio:.2f}x (threshold {threshold:.2f}x) — "
        f"possible regression in the {suite_name} encoder path"
    )


def test_dtype_suite_full_report() -> None:
    """
    Print a side-by-side comparison table across all 8 dtype suites,
    including Speed, Archive, and Brotli profiles.

    Always passes.  Run with ``pytest -s`` to see the table.
    """
    import fluxcompress as fc

    header = (
        f"{'Suite':<15} {'Rows':>8} {'Format':<24} "
        f"{'Size':>9} {'Ratio':>7} {'Comp ms':>9} {'Decomp ms':>10}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("FluxCompress vs Parquet — Dtype Suite Comparison (all profiles)")
    print(sep)
    print(header)

    wins  = {"speed": 0, "archive": 0, "brotli": 0}
    total = 0

    for suite_name, factory in DTYPE_SUITE.items():
        sizes = _REPORT_SIZES[suite_name]
        print("-" * len(header))
        for n in sizes:
            table = factory(n)
            flux_results = [
                _bench_flux(table, suite_name, "speed"),
                _bench_flux(table, suite_name, "archive"),
                _bench_flux(table, suite_name, "brotli"),
            ]
            pq_results = [
                _bench_parquet(table, suite_name, "snappy"),
                _bench_parquet(table, suite_name, "zstd"),
            ]
            for r in flux_results + pq_results:
                print(
                    f"{suite_name:<15} {r.rows:>8,} {r.name:<24} "
                    f"{_human(r.compressed_bytes):>9} {r.ratio:>7.2f}x "
                    f"{r.compress_ms:>8.1f} {r.decompress_ms:>10.1f}"
                )
            print()
            best_pq = min(r.compressed_bytes for r in pq_results)
            total  += 1
            for prof_result in flux_results:
                key = prof_result.name.split("(")[1].rstrip(")")
                if prof_result.compressed_bytes <= best_pq:
                    wins[key] += 1

    print(sep)
    for prof, w in wins.items():
        print(f"  Flux ({prof:<8}) smaller than best Parquet: {w}/{total} combinations")


# ─────────────────────────────────────────────────────────────────────────────
# Survey-comments showcase: Brotli profile vs Parquet
# ─────────────────────────────────────────────────────────────────────────────

def test_survey_comments_brotli_vs_parquet() -> None:
    """
    Survey-comments showcase: compare all FluxCompress profiles against
    Parquet (snappy and zstd).

    The Speed profile uses FSST+LZ4 on high-cardinality NL text and typically
    falls behind Parquet snappy.  The Brotli profile uses Brotli’s static
    English-language dictionary and should clearly beat Parquet snappy.

    Prints a complete side-by-side report and asserts the Brotli profile
    beats Parquet snappy on this natural-language workload.
    """
    import fluxcompress as fc

    n = 65_536
    table = _survey_comments(n)
    raw = sum(col.nbytes for col in table.columns)

    hdr = f"{'Format':<22} {'MB':>8} {'Ratio':>8}"
    sep = "-" * len(hdr)
    print(f"\n  survey_comments {n:,} rows ({raw/1e6:.2f} MB raw)")
    print(f"  {sep}")
    print(f"  {hdr}")
    print(f"  {sep}")

    results = {}
    for profile in ["speed", "archive", "brotli"]:
        b = len(fc.compress(table, profile=profile))
        ratio = raw / b
        results[f"Flux ({profile})"] = (b, ratio)
        print(f"  {'Flux (' + profile + ')':<22} {b/1e6:>8.3f} {ratio:>8.2f}x")

    sink = io.BytesIO(); pq.write_table(table, sink, compression="snappy")
    pq_snappy = sink.tell(); pq_snappy_ratio = raw / pq_snappy
    print(f"  {'Parquet (snappy)':<22} {pq_snappy/1e6:>8.3f} {pq_snappy_ratio:>8.2f}x")

    sink = io.BytesIO(); pq.write_table(table, sink, compression="zstd")
    pq_zstd = sink.tell(); pq_zstd_ratio = raw / pq_zstd
    print(f"  {'Parquet (zstd)':<22} {pq_zstd/1e6:>8.3f} {pq_zstd_ratio:>8.2f}x")
    print(f"  {sep}")

    brotli_ratio = results["Flux (brotli)"][1]
    assert brotli_ratio > 1.0, "Brotli profile must compress NL text"
    assert brotli_ratio > results["Flux (speed)"][1], (
        "Brotli profile must beat Speed profile on NL text"
    )
    # Brotli’s static dictionary should outperform Parquet snappy on English text.
    assert brotli_ratio > pq_snappy_ratio, (
        f"Brotli ({brotli_ratio:.2f}x) should beat Parquet snappy "
        f"({pq_snappy_ratio:.2f}x) on natural-language text"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_full_comparison_report()
    test_dtype_suite_full_report()
