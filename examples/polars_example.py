"""
FluxCompress × Polars — Usage Examples
=======================================

FluxCompress integrates with Polars in two ways:

  1. Via the `fluxcapacitor` CLI  — compress/decompress Arrow IPC files that
     Polars can read and write natively.
  2. Via the `loom` Rust library + PyO3 bindings (future) — direct in-process
     compression of Polars DataFrames without writing to disk.

This file demonstrates both paths, plus the CLI approach that works today.

Dependencies
------------
    pip install polars pyarrow

Build FluxCompress
------------------
    cargo build --release -p fluxcapacitor
    # Binary lands at: target/release/fluxcapacitor
"""

import subprocess
import tempfile
import os
import time
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

# ─────────────────────────────────────────────────────────────────────────────
# Helper: locate the fluxcapacitor binary
# ─────────────────────────────────────────────────────────────────────────────

FLUX_BIN = os.environ.get(
    "FLUXCAPACITOR_BIN",
    os.path.join(os.path.dirname(__file__), "../target/release/fluxcapacitor"),
)


def flux(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run the fluxcapacitor CLI and return the result."""
    cmd = [FLUX_BIN] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Compress a Polars DataFrame → .flux
# ─────────────────────────────────────────────────────────────────────────────

def compress_dataframe(df: pl.DataFrame, flux_path: str) -> None:
    """
    Compress a Polars DataFrame to a .flux file.

    Pipeline:
      Polars DataFrame
        → Arrow RecordBatch  (zero-copy via Polars' Arrow backend)
        → Arrow IPC file     (temp file on disk)
        → fluxcapacitor compress
        → .flux file
    """
    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as tmp:
        arrow_path = tmp.name

    try:
        # Polars → Arrow IPC (uses the shared Arrow memory layout — no copy)
        arrow_table = df.to_arrow()
        with pa.OSFile(arrow_path, "wb") as sink:
            writer = ipc.new_file(sink, arrow_table.schema)
            writer.write_table(arrow_table)
            writer.close()

        # Arrow IPC → .flux  (via CLI)
        result = flux(["compress", "--input", arrow_path, "--output", flux_path])
        print(result.stdout.strip())

    finally:
        os.unlink(arrow_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Decompress .flux → Polars DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def decompress_to_dataframe(flux_path: str) -> pl.DataFrame:
    """
    Decompress a .flux file back into a Polars DataFrame.

    Pipeline:
      .flux file
        → fluxcapacitor decompress
        → Arrow IPC file
        → Polars DataFrame   (zero-copy read)
    """
    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as tmp:
        arrow_path = tmp.name

    try:
        flux(["decompress", "--input", flux_path, "--output", arrow_path])
        df = pl.read_ipc(arrow_path, memory_map=True)  # memory-mapped = zero-copy
        return df
    finally:
        os.unlink(arrow_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Predicate Pushdown — only decompress matching rows
# ─────────────────────────────────────────────────────────────────────────────

def decompress_with_predicate(
    flux_path: str,
    column: str,
    gt: int,
) -> pl.DataFrame:
    """
    Decompress only the blocks that *may* contain rows where column > gt.

    FluxCompress skips Atlas blocks whose [z_min, z_max] range cannot satisfy
    the predicate — no wasted I/O on irrelevant data.
    """
    with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as tmp:
        arrow_path = tmp.name

    try:
        flux([
            "decompress",
            "--input",  flux_path,
            "--output", arrow_path,
            "--column", column,
            "--gt",     str(gt),
        ])
        return pl.read_ipc(arrow_path, memory_map=True)
    finally:
        os.unlink(arrow_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inspect a .flux file's Atlas footer
# ─────────────────────────────────────────────────────────────────────────────

def inspect(flux_path: str, fmt: str = "table") -> str:
    result = flux(["inspect", flux_path, "--format", fmt])
    return result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# 5. Two-Pass Cold Optimization
# ─────────────────────────────────────────────────────────────────────────────

def optimize_partitions(partition_dir: str, output_path: str) -> None:
    """
    Run the two-pass global optimizer on a directory of .flux partition files.

    Pass 1: scan all partitions → build global master dictionary + stats.
    Pass 2: re-pack with global dict + Z-Order interleaving → single archive.
    """
    result = flux([
        "optimize",
        "--input-dir", partition_dir,
        "--output",    output_path,
    ])
    print(result.stdout.strip())


# ─────────────────────────────────────────────────────────────────────────────
# 6. End-to-End Demo
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("FluxCompress × Polars Demo")
    print("=" * 60)

    # ── Build a realistic DataFrame ──────────────────────────────────────────
    import random
    random.seed(42)

    n = 1_000_000
    df = pl.DataFrame({
        # Sequential IDs → DeltaDelta encoding
        "user_id":      pl.Series(range(n), dtype=pl.UInt64),
        # Low-cardinality enum → Dictionary encoding
        "region":       pl.Series([i % 8 for i in range(n)], dtype=pl.UInt64),
        # Uniform random → BitSlab encoding
        "revenue_cents":pl.Series(
            [random.randint(0, 99_999) for _ in range(n)], dtype=pl.UInt64
        ),
        # Mostly small with rare u64 giants → BitSlab + OutlierMap
        "agg_result":   pl.Series(
            [2**63 if i % 10_000 == 0 else i * 17 for i in range(n)],
            dtype=pl.UInt64,
        ),
    })

    print(f"\nDataFrame: {n:,} rows × {df.width} columns")
    print(df.head(5))

    with tempfile.TemporaryDirectory() as tmpdir:
        flux_path   = os.path.join(tmpdir, "data.flux")
        cold_path   = os.path.join(tmpdir, "data_cold.flux")

        # ── Compress ──────────────────────────────────────────────────────
        print("\n── Compressing ──────────────────────────────────────────")
        t0 = time.perf_counter()
        compress_dataframe(df, flux_path)
        compress_s = time.perf_counter() - t0

        raw_bytes = n * 4 * 8  # 4 columns × 8 bytes each
        flux_bytes = os.path.getsize(flux_path)
        print(f"  Raw size  : {raw_bytes / 1e6:.1f} MB")
        print(f"  Flux size : {flux_bytes / 1e6:.1f} MB")
        print(f"  Ratio     : {raw_bytes / flux_bytes:.2f}×")
        print(f"  Time      : {compress_s:.3f}s")

        # ── Inspect Atlas footer ──────────────────────────────────────────
        print("\n── Atlas Footer (first 5 blocks) ────────────────────────")
        lines = inspect(flux_path).splitlines()
        for line in lines[:10]:
            print(" ", line)

        # ── Decompress — all rows ─────────────────────────────────────────
        print("\n── Decompressing (all rows) ─────────────────────────────")
        t0 = time.perf_counter()
        df_back = decompress_to_dataframe(flux_path)
        decompress_s = time.perf_counter() - t0
        print(f"  Rows      : {len(df_back):,}")
        print(f"  Time      : {decompress_s:.3f}s")
        print(f"  Throughput: {flux_bytes / decompress_s / 1e6:.0f} MB/s")

        # Verify round-trip correctness on the first column.
        assert df["user_id"].to_list() == df_back["value"].to_list()[:n], \
            "Round-trip mismatch!"
        print("  ✓ Round-trip verified")

        # ── Predicate Pushdown ────────────────────────────────────────────
        print("\n── Predicate Pushdown: user_id > 900,000 ────────────────")
        t0 = time.perf_counter()
        df_filtered = decompress_with_predicate(flux_path, "value", 900_000)
        filter_s = time.perf_counter() - t0
        print(f"  Rows returned : {len(df_filtered):,}")
        print(f"  Time          : {filter_s:.3f}s  "
              f"(vs {decompress_s:.3f}s full scan)")

        # ── Cold Optimization ─────────────────────────────────────────────
        print("\n── Cold Optimization ────────────────────────────────────")
        # Write partitions (simulate 4 Spark partition files).
        part_dir = os.path.join(tmpdir, "partitions")
        os.makedirs(part_dir)
        chunk = n // 4
        for i in range(4):
            part_df = df.slice(i * chunk, chunk)
            compress_dataframe(part_df, os.path.join(part_dir, f"part-{i:04d}.flux"))

        t0 = time.perf_counter()
        optimize_partitions(part_dir, cold_path)
        cold_s = time.perf_counter() - t0
        cold_bytes = os.path.getsize(cold_path)

        print(f"  4 partitions  → 1 archive")
        print(f"  Hot size  : {flux_bytes / 1e6:.1f} MB")
        print(f"  Cold size : {cold_bytes / 1e6:.1f} MB  "
              f"({(1 - cold_bytes/flux_bytes)*100:.1f}% smaller)")
        print(f"  Time      : {cold_s:.3f}s")

    print("\n✓ Demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Polars LazyFrame integration pattern
# ─────────────────────────────────────────────────────────────────────────────

def polars_lazy_pattern():
    """
    Pattern for integrating FluxCompress with Polars LazyFrames.

    Since Polars LazyFrames are evaluated lazily, the recommended pattern is:
      1. Collect the LazyFrame to a DataFrame.
      2. Compress the DataFrame to .flux.
      3. Later: decompress .flux → DataFrame → convert back to LazyFrame.
    """
    # Build a lazy pipeline.
    lf = (
        pl.LazyFrame({"id": range(100_000), "val": range(100_000)})
        .with_columns([
            (pl.col("val") * 3).alias("val_x3"),
            (pl.col("id") % 10).alias("bucket"),
        ])
        .filter(pl.col("val") > 1000)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        flux_path = os.path.join(tmpdir, "lazy_result.flux")

        # Collect → compress.
        df = lf.collect()
        compress_dataframe(df, flux_path)

        # Later: decompress → back to LazyFrame for further transformations.
        df_back = decompress_to_dataframe(flux_path)
        lf_continued = df_back.lazy().filter(pl.col("value") > 50_000)
        result = lf_continued.collect()
        print(f"Lazy pattern result: {len(result):,} rows")


if __name__ == "__main__":
    demo()
