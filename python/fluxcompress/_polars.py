# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Polars-specific helpers for FluxCompress.

These thin wrappers convert Polars DataFrames to/from PyArrow tables so the
core Rust extension can process them.  The conversion is zero-copy for
numerical columns because Polars and PyArrow share the same Arrow memory
layout under the hood.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from fluxcompress._fluxcompress import (
    FluxBuffer, FluxBatchReader, Predicate,
    compress, decompress, decompress_file,
    merge_flux_buffers, write_flux, read_flux_schema,
)

# Default chunk size: 500K rows.  Keeps peak memory around 500MB
# per chunk for a wide table, preventing OOM on constrained drivers.
_DEFAULT_CHUNK_SIZE = 500_000


def compress_polars(
    df: "pl.DataFrame",
    strategy: str = "auto",
    profile: str = "speed",
    u64_only: bool = False,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> FluxBuffer:
    """
    Compress a Polars DataFrame into a :class:`FluxBuffer`.

    The DataFrame is converted to a PyArrow Table via the Arrow C Data
    Interface — a zero-copy operation for most numeric column types.

    For large DataFrames, the data is processed in chunks of ``chunk_size``
    rows to limit peak memory.  The resulting chunks are merged into a
    single ``FluxBuffer`` on the Rust side.

    Parameters
    ----------
    df:
        A ``polars.DataFrame``.
    strategy:
        Compression strategy override.  One of ``"auto"``, ``"rle"``,
        ``"delta"``, ``"dict"``, ``"bitslab"``, ``"lz4"``.
        ``"auto"`` (default) lets the Loom classifier decide per segment.
    profile:
        Compression profile.  One of ``"speed"``, ``"balanced"``,
        ``"archive"``.  Default ``"speed"``.
    u64_only:
        If ``True``, skip u128 widening during compression.  Halves memory
        bandwidth for types ≤ 64 bits.  Default ``False``.
    chunk_size:
        Maximum rows per chunk.  Smaller values use less memory but add
        minor overhead from merging.  Default 500,000.
        Set to ``0`` to disable chunking (compress entire DF at once).

    Returns
    -------
    FluxBuffer
        The compressed bytes with Atlas footer.

    Example
    -------
    >>> import polars as pl
    >>> import fluxcompress as fc
    >>> df = pl.DataFrame({"user_id": range(1_000_000), "revenue": range(1_000_000)})
    >>> buf = fc.compress_polars(df, profile="archive", u64_only=True)
    >>> print(buf)
    FluxBuffer(...)
    """
    try:
        import polars as pl  # noqa: F401
    except ImportError:
        raise ImportError(
            "polars is required for compress_polars(). "
            "Install it with: pip install fluxcompress[polars]"
        ) from None

    height = df.height

    # Small DF or chunking disabled: single-shot compress.
    # LargeUtf8 normalization is handled on the Rust side.
    if chunk_size <= 0 or height <= chunk_size:
        arrow_table = df.to_arrow()
        return compress(arrow_table, strategy=strategy, profile=profile, u64_only=u64_only)

    # Chunked path: compress each slice independently, then merge.
    chunk_bufs: list[FluxBuffer] = []
    for start in range(0, height, chunk_size):
        length = min(chunk_size, height - start)
        chunk_df = df.slice(start, length)
        arrow_chunk = chunk_df.to_arrow()
        buf = compress(arrow_chunk, strategy=strategy, profile=profile, u64_only=u64_only)
        chunk_bufs.append(buf)
        # Let the chunk DF and Arrow table be freed before the next iteration.
        del chunk_df, arrow_chunk

    return merge_flux_buffers(chunk_bufs)


def decompress_polars(
    buf: "FluxBuffer | bytes | str",
    predicate: Predicate | None = None,
    column_name: str = "value",
    columns: "list[str] | None" = None,
) -> "pl.DataFrame":
    """
    Decompress a :class:`FluxBuffer`, raw bytes, or ``.flux`` file path into
    a Polars DataFrame.

    When a file path is given, reads directly via memory-mapped I/O without
    loading the compressed bytes into Python memory.

    Parameters
    ----------
    buf:
        A :class:`FluxBuffer`, raw ``bytes``, or a ``str`` file path to a
        ``.flux`` file.
    predicate:
        Optional predicate for pushdown filtering.  Blocks whose
        ``[z_min, z_max]`` range cannot satisfy the predicate are skipped.
    column_name:
        Column name for the decompressed output (single-column files only).
    columns:
        Optional list of column names to decompress.  Columns not in this
        list are skipped entirely, saving memory and CPU.

    Returns
    -------
    polars.DataFrame

    Example
    -------
    >>> import fluxcompress as fc
    >>> df = fc.decompress_polars("data.flux", columns=["id", "revenue"])
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required for decompress_polars(). "
            "Install it with: pip install fluxcompress[polars]"
        ) from None

    if isinstance(buf, str):
        # File path — use mmap-based decompress (avoids loading compressed
        # bytes into Python memory).
        arrow_table = decompress_file(buf, predicate=predicate, columns=columns)
    else:
        arrow_table = decompress(buf, predicate=predicate, column_name=column_name)

    return pl.from_arrow(arrow_table)


def scan_flux(
    path: str,
    columns: "list[str] | None" = None,
    predicate: Predicate | None = None,
) -> "pl.LazyFrame":
    """
    Lazily scan one or more ``.flux`` files, returning a Polars ``LazyFrame``.

    Supports:

    - **Single file**: ``scan_flux("data.flux")``
    - **FluxTable directory**: ``scan_flux("my_table.fluxtable")`` — reads
      the transaction log, resolves live files, and streams them.
    - **Column projection**: only decompresses the requested columns.
    - **Predicate pushdown**: skips blocks that cannot match.

    Each ``.flux`` file is memory-mapped and decompressed one at a time,
    so peak memory is bounded by the largest single file.

    Parameters
    ----------
    path:
        Path to a ``.flux`` file or a ``.fluxtable`` directory.
    columns:
        Optional list of column names to load.  Unlisted columns are
        never decompressed.
    predicate:
        Optional ``Predicate`` for Z-Order block skipping.

    Returns
    -------
    polars.LazyFrame

    Example
    -------
    >>> lf = fc.scan_flux("events.fluxtable", columns=["id", "revenue"])
    >>> df = lf.filter(pl.col("revenue") > 1000).collect()
    """
    import os
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required for scan_flux(). "
            "Install it with: pip install fluxcompress[polars]"
        ) from None

    flux_paths = _resolve_flux_paths(path)
    reader = FluxBatchReader(flux_paths, columns=columns, predicate=predicate)

    # Collect batches into a list of Polars DataFrames, then concat lazily.
    # Each batch is one file — memory is freed after conversion.
    frames: list[pl.DataFrame] = []
    for py_batch in reader:
        frames.append(pl.from_arrow(py_batch))

    if not frames:
        # No data — return empty LazyFrame with correct schema.
        import pyarrow as pa
        schema = read_flux_schema(flux_paths[0])
        if columns:
            schema = pa.schema([f for f in schema if f.name in set(columns)])
        return pl.from_arrow(pa.table({f.name: pa.array([], type=f.type) for f in schema})).lazy()

    return pl.concat(frames).lazy()


def write_flux_table(
    df: "pl.DataFrame",
    path: str,
    profile: str = "speed",
    u64_only: bool = False,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    partition_by: "list[dict] | None" = None,
    clustering_columns: "list[str] | None" = None,
    mode: str = "error",
    compute_stats: bool = False,
    max_workers: "int | None" = None,
) -> str:
    """
    Write a Polars DataFrame to a ``.fluxtable`` directory.

    Thin wrapper around :class:`FluxTableWriter`. For streaming writes
    from a batch iterator (e.g. ``pl.scan_csv(...).collect_batches(N)`` or
    a PySpark DataFrame), prefer the context-manager API directly so the
    transaction log is committed once instead of per-batch.

    Parameters
    ----------
    df:
        The DataFrame to write. Also accepts PyArrow tables / pandas
        DataFrames / PySpark DataFrames — see :class:`FluxTableWriter`.
    path:
        Path for the ``.fluxtable`` directory.  Created if it doesn't exist.
        Supports ``gs://``, ``s3://``, ``az://`` URIs when fsspec is installed.
    profile:
        Compression profile (``"speed"``, ``"balanced"``, ``"archive"``).
    u64_only:
        Skip u128 widening.
    chunk_size:
        Maximum rows per part file.  Default 500,000.
    partition_by:
        Optional hidden partition spec. List of dicts with ``source_column``
        and ``transform`` keys. Duplicate specs are automatically deduped in
        ``_flux_meta.json`` (fixes a historical bug where appending to a
        partitioned table inflated the meta file).
    clustering_columns:
        Columns for liquid clustering / Z-Order optimization.
    mode:
        Table-exists behaviour (Spark-style):

        - ``"error"`` (default) — refuse if the table already has data.
        - ``"overwrite"`` — atomically replace any existing data.
        - ``"append"`` — add to existing data.
        - ``"ignore"`` — no-op if the table exists.
    compute_stats:
        Compute per-column min/max/null_count and store in the file
        manifest. Default ``False`` because on wide schemas this is often
        the single biggest cost of a write.
    max_workers:
        Threads for parallel batch compression (default = CPU count).

    Example
    -------
    >>> fc.write_flux_table(df, "events.fluxtable",
    ...     profile="archive", mode="overwrite",
    ...     partition_by=[{"source_column": "country", "transform": "identity"}],
    ...     clustering_columns=["country", "created_at"])
    """
    from fluxcompress._table_writer import FluxTableWriter

    with FluxTableWriter(
        path,
        profile=profile,
        u64_only=u64_only,
        chunk_size=chunk_size,
        partition_by=partition_by,
        clustering_columns=clustering_columns,
        compute_stats=compute_stats,
        mode=mode,
        max_workers=max_workers,
    ) as w:
        w.write_batch(df)
    return path


def optimize(
    path: str,
    clustering_columns: "list[str] | None" = None,
) -> dict:
    """
    Optimize a ``.fluxtable`` by re-clustering overlapping files.

    Reads file manifests from the transaction log, detects files with
    overlapping column ranges on the clustering key, and rewrites them
    sorted by Z-Order for minimal scan amplification.

    Parameters
    ----------
    path:
        Path to the ``.fluxtable`` directory.
    clustering_columns:
        Optional override for clustering columns.  If provided and different
        from the current metadata, the table's clustering spec is evolved.

    Returns
    -------
    dict
        Summary with keys: ``groups_merged``, ``files_read``, ``files_written``.

    Example
    -------
    >>> fc.optimize("events.fluxtable", clustering_columns=["country", "created_at"])
    """
    import os
    import json
    import time

    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required for optimize(). "
            "Install it with: pip install fluxcompress[polars]"
        ) from None

    meta_path = os.path.join(path, "_flux_meta.json")
    if not os.path.exists(meta_path):
        return {"groups_merged": 0, "files_read": 0, "files_written": 0}

    with open(meta_path) as f:
        meta = json.load(f)

    # Resolve clustering columns.
    cc = clustering_columns or meta.get("clustering_columns", [])
    if not cc:
        return {"groups_merged": 0, "files_read": 0, "files_written": 0}

    # Update metadata if clustering columns changed.
    if clustering_columns and clustering_columns != meta.get("clustering_columns"):
        meta["clustering_columns"] = clustering_columns
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Collect live file manifests from log.
    log_dir = os.path.join(path, "_flux_log")
    live_files, manifests_by_path = _resolve_live_manifests(path)

    if len(live_files) < 2:
        return {"groups_merged": 0, "files_read": 0, "files_written": 0}

    # Find overlapping groups.
    manifest_list = [manifests_by_path[f] for f in live_files if f in manifests_by_path]
    if len(manifest_list) < 2:
        return {"groups_merged": 0, "files_read": 0, "files_written": 0}

    groups = _find_overlapping_groups(manifest_list, cc)
    merge_groups = [g for g in groups if len(g) >= 2]

    if not merge_groups:
        return {"groups_merged": 0, "files_read": 0, "files_written": 0}

    data_dir = os.path.join(path, "data")
    part_num = len([f for f in os.listdir(data_dir) if f.endswith(".flux")])
    all_added: list[str] = []
    all_removed: list[str] = []
    all_new_manifests: list[dict] = []
    total_rows = 0

    for group_indices in merge_groups:
        group_manifests = [manifest_list[i] for i in group_indices]
        group_paths = [os.path.join(path, m["path"]) for m in group_manifests]

        # Read all files in the group.
        frames = []
        for fp in group_paths:
            table = decompress_file(fp)
            frames.append(pl.from_arrow(table))

        merged = pl.concat(frames)
        # Sort by clustering columns (Z-Order approximation via lexicographic sort).
        merged = merged.sort(cc)

        rows = merged.height
        total_rows += rows

        # Rewrite as new file(s).
        buf = compress_polars(merged, profile="balanced", chunk_size=0)
        filename = f"part-{part_num:04d}.flux"
        filepath = os.path.join(data_dir, filename)
        write_flux(buf, filepath)
        rel_path = f"data/{filename}"
        all_added.append(rel_path)

        # Build manifest for the new file.
        col_stats = {}
        for col_name in merged.columns:
            try:
                s = merged[col_name]
                col_stats[col_name] = {
                    "min": str(s.min()),
                    "max": str(s.max()),
                    "null_count": int(s.null_count()),
                }
            except Exception:
                pass

        all_new_manifests.append({
            "path": rel_path,
            "partition_values": {},
            "spec_id": meta.get("current_spec_id", 0),
            "row_count": rows,
            "file_size_bytes": os.path.getsize(filepath),
            "column_stats": col_stats,
        })

        for m in group_manifests:
            all_removed.append(m["path"])

        part_num += 1
        del merged, frames, buf

    # Commit compact transaction.
    version = len([f for f in os.listdir(log_dir) if f.endswith(".json")])
    entry = {
        "version": version,
        "timestamp_ms": int(time.time() * 1000),
        "operation": "compact",
        "data_files_added": all_added,
        "data_files_removed": all_removed,
        "file_manifests": all_new_manifests,
        "row_count_delta": 0,
        "metadata": {"clustering_columns": json.dumps(cc)},
    }
    log_path = os.path.join(log_dir, f"{version:08d}.json")
    with open(log_path, "w") as f:
        json.dump(entry, f, indent=2)

    return {
        "groups_merged": len(merge_groups),
        "files_read": sum(len(g) for g in merge_groups),
        "files_written": len(all_added),
    }


def _resolve_flux_paths(path: str) -> list[str]:
    """Resolve a path to a list of .flux file paths."""
    import os
    import json

    if path.endswith(".flux") and os.path.isfile(path):
        return [path]

    # FluxTable directory — read log to get live files.
    log_dir = os.path.join(path, "_flux_log")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(
            f"{path} is not a .flux file or .fluxtable directory"
        )

    log_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".json"))
    live_files: set[str] = set()
    for lf in log_files:
        with open(os.path.join(log_dir, lf)) as f:
            entry = json.load(f)
        for added in entry.get("data_files_added", []):
            live_files.add(added)
        for removed in entry.get("data_files_removed", []):
            live_files.discard(removed)

    if not live_files:
        raise FileNotFoundError(f"no live data files in {path}")

    return sorted(os.path.join(path, f) for f in live_files)


def _resolve_live_manifests(path: str) -> "tuple[list[str], dict[str, dict]]":
    """Resolve live file paths and their manifests from the transaction log."""
    import os
    import json

    log_dir = os.path.join(path, "_flux_log")
    log_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".json"))

    live_files: set[str] = set()
    manifests: dict[str, dict] = {}

    for lf in log_files:
        with open(os.path.join(log_dir, lf)) as f:
            entry = json.load(f)
        for added in entry.get("data_files_added", []):
            live_files.add(added)
        for removed in entry.get("data_files_removed", []):
            live_files.discard(removed)
            manifests.pop(removed, None)
        for fm in entry.get("file_manifests", []):
            manifests[fm["path"]] = fm

    live = sorted(live_files)
    return live, manifests


def _find_overlapping_groups(
    manifests: list[dict], clustering_columns: list[str]
) -> list[list[int]]:
    """Find groups of files that overlap on all clustering columns."""
    n = len(manifests)
    if n <= 1:
        return [[i] for i in range(n)]

    def ranges_overlap(a: dict, b: dict) -> bool:
        a_min = a.get("min")
        a_max = a.get("max")
        b_min = b.get("min")
        b_max = b.get("max")
        if a_min is None or a_max is None or b_min is None or b_max is None:
            return True  # Missing stats → assume overlap.
        return not (a_max < b_min or b_max < a_min)

    def files_overlap(i: int, j: int) -> bool:
        for col in clustering_columns:
            a_stats = manifests[i].get("column_stats", {}).get(col)
            b_stats = manifests[j].get("column_stats", {}).get(col)
            if a_stats and b_stats and not ranges_overlap(a_stats, b_stats):
                return False
        return True

    # Connected components.
    visited = [False] * n
    groups: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        group = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            group.append(node)
            for neighbor in range(n):
                if not visited[neighbor] and files_overlap(node, neighbor):
                    visited[neighbor] = True
                    queue.append(neighbor)
        groups.append(group)

    return groups
