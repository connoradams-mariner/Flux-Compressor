# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
FluxTableWriter
===============

Unified streaming writer for ``.fluxtable`` directories. Supports batches
from Polars, PySpark, PyArrow, or any other source that yields Arrow-like
data. Designed to be dramatically faster than per-batch
``write_flux_table()`` calls, especially on cloud object stores (GCS,
S3, Azure).

Key differences vs the old per-batch ``write_flux_table``:

- **Single metadata read/write per session** (not per batch).
- **In-memory ``part_num`` counter** — no ``os.listdir`` in the hot path.
- **One atomic transaction log entry** at close, not one per batch.
- **Partition spec deduplication** — reusing the same spec no longer
  duplicates it in ``_flux_meta.json``.
- **Optional column stats** (off by default), computed in a single
  vectorised Polars / PySpark plan when enabled.
- **Parallel batch compression** via a thread pool (Rust releases the
  GIL, so this scales with cores on dense batches).
- **``mode=`` argument** for ``error`` / ``overwrite`` / ``append`` /
  ``ignore`` table-exists semantics (Spark-style).
- **``fsspec`` backend** so ``gs://``, ``s3://``, ``az://``, and local
  paths all work without code changes.
"""

from __future__ import annotations

import json
import os
import posixpath
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

from fluxcompress._fluxcompress import FluxBuffer, compress, write_flux

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa
    from pyspark.sql import DataFrame as SparkDataFrame


# ─────────────────────────────────────────────────────────────────────────────
# Table-exists modes (Spark-style)
# ─────────────────────────────────────────────────────────────────────────────

MODE_ERROR = "error"              # default — refuse if table exists
MODE_ERROR_IF_EXISTS = "errorifexists"  # alias
MODE_OVERWRITE = "overwrite"      # atomic: trash old data/log, start fresh
MODE_APPEND = "append"            # add to existing table (current default)
MODE_IGNORE = "ignore"            # no-op if table exists


class TableExistsError(FileExistsError):
    """Raised when the target table already has data and ``mode='error'``."""


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem abstraction — uses fsspec if available, falls back to local `os`.
# Keeps the hot path free of an fsspec dependency for local writes, which is
# the common case in tests and dev.
# ─────────────────────────────────────────────────────────────────────────────

class _Fs:
    """Tiny filesystem abstraction covering both local and fsspec backends."""

    def __init__(self, path: str) -> None:
        self.remote = "://" in path
        if self.remote:
            try:
                import fsspec  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    f"Writing to {path!r} requires fsspec. "
                    "Install with: pip install fsspec gcsfs  (or s3fs / adlfs)"
                ) from exc
            self._fs = fsspec.filesystem(path.split("://", 1)[0])
        else:
            self._fs = None  # local: use stdlib os

    # Path helpers.
    @staticmethod
    def join(*parts: str) -> str:
        # Use posixpath for remote URIs so slashes are canonical on all OSes.
        if any("://" in p for p in parts):
            return posixpath.join(*parts)
        return os.path.join(*parts)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        if self._fs is not None:
            # fsspec auto-creates intermediate "directories" on write.
            return
        os.makedirs(path, exist_ok=exist_ok)

    def exists(self, path: str) -> bool:
        if self._fs is not None:
            return bool(self._fs.exists(path))
        return os.path.exists(path)

    def isdir(self, path: str) -> bool:
        if self._fs is not None:
            return bool(self._fs.isdir(path))
        return os.path.isdir(path)

    def read_json(self, path: str) -> Any:
        if self._fs is not None:
            with self._fs.open(path, "r") as f:
                return json.load(f)
        with open(path, "r") as f:
            return json.load(f)

    def write_json(self, path: str, value: Any) -> None:
        payload = json.dumps(value, indent=2)
        self.write_bytes(path, payload.encode("utf-8"))

    def write_bytes(self, path: str, data: bytes) -> None:
        if self._fs is not None:
            with self._fs.open(path, "wb") as f:
                f.write(data)
            return
        # Local: write to a temp sibling then rename for atomicity.
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def size(self, path: str) -> int:
        if self._fs is not None:
            info = self._fs.info(path)
            return int(info.get("size", 0))
        return os.path.getsize(path)

    def rm_recursive(self, path: str) -> None:
        if self._fs is not None:
            try:
                self._fs.rm(path, recursive=True)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                pass
            return
        if os.path.isdir(path):
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def rm_file(self, path: str) -> None:
        """Remove a single file (not a directory). Reliable on gcsfs / s3fs
        where `rm(..., recursive=True)` sometimes no-ops on standalone
        objects."""
        if self._fs is not None:
            try:
                self._fs.rm_file(path)
            except AttributeError:
                # Older fsspec: fall back to .rm with recursive=False.
                try:
                    self._fs.rm(path)
                except FileNotFoundError:
                    pass
            except FileNotFoundError:
                pass
            return
        try:
            os.unlink(path)
        except (FileNotFoundError, IsADirectoryError):
            pass

    def rename(self, src: str, dst: str) -> None:
        if self._fs is not None:
            self._fs.mv(src, dst, recursive=True)
            return
        os.rename(src, dst)

    def has_data_files(self, data_dir: str) -> bool:
        """Return True if ``data_dir`` contains any `.flux` files."""
        if self._fs is not None:
            if not self._fs.exists(data_dir):
                return False
            try:
                entries = self._fs.ls(data_dir, detail=False)
            except FileNotFoundError:
                return False
            return any(str(e).endswith(".flux") for e in entries)
        if not os.path.isdir(data_dir):
            return False
        try:
            return any(n.endswith(".flux") for n in os.listdir(data_dir))
        except FileNotFoundError:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHUNK_SIZE = 500_000


class FluxTableWriter:
    """
    Streaming writer for ``.fluxtable`` directories.

    Use as a context manager so the transaction log is committed exactly
    once on clean exit:

    >>> with FluxTableWriter("out.fluxtable",
    ...                      profile="archive",
    ...                      mode="overwrite") as w:
    ...     for batch in pl.scan_csv(fp).collect_batches(500_000):
    ...         w.write_batch(batch)

    ``write_batch`` accepts:

    - ``polars.DataFrame``
    - ``pyarrow.Table`` or ``pyarrow.RecordBatch``
    - ``pandas.DataFrame`` (lazy import)
    - ``pyspark.sql.DataFrame`` (collected to Arrow via ``toArrow()`` /
      ``toPandas()``; for large Spark DataFrames, use
      :meth:`write_spark_dataframe` which drives ``mapInArrow`` so each
      executor writes its own part files directly).

    Parameters
    ----------
    path:
        Path to the ``.fluxtable`` directory. Supports ``gs://``, ``s3://``,
        ``az://`` URIs when the corresponding fsspec backend is installed.
    profile:
        ``"speed"`` | ``"balanced"`` | ``"archive"``. Default ``"speed"``.
    u64_only:
        Skip u128 widening.
    chunk_size:
        Rows per output part file. Default 500 000. Each batch handed to
        ``write_batch`` is sub-chunked if larger.
    partition_by:
        Optional hidden partition spec. Same shape as legacy
        ``write_flux_table``.
    clustering_columns:
        Columns to use for liquid-clustering optimizations.
    compute_stats:
        When ``True``, compute per-column min/max/null_count and store in
        the file manifest. Off by default because stats are expensive on
        wide schemas and aren't required for correct reads.
    mode:
        Table-exists behaviour. One of ``"error"`` (default),
        ``"overwrite"``, ``"append"``, ``"ignore"``.
    max_workers:
        Number of threads used to compress batches in parallel. Defaults
        to the number of CPU cores. Set to 1 for fully sequential writes.
    """

    def __init__(
        self,
        path: str,
        *,
        profile: str = "speed",
        u64_only: bool = False,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        partition_by: "Sequence[Mapping[str, str]] | None" = None,
        clustering_columns: "Sequence[str] | None" = None,
        compute_stats: bool = False,
        mode: str = MODE_ERROR,
        max_workers: "int | None" = None,
    ) -> None:
        self.path = path
        self.profile = profile
        self.u64_only = u64_only
        self.chunk_size = max(int(chunk_size) if chunk_size else 0, 0)
        self.partition_by = [dict(p) for p in partition_by] if partition_by else None
        self.clustering_columns = list(clustering_columns) if clustering_columns else None
        self.compute_stats = bool(compute_stats)
        self.mode = mode.lower().strip()
        self.max_workers = max_workers or (os.cpu_count() or 4)

        # Derived.
        self._fs = _Fs(path)
        self._data_dir = _Fs.join(path, "data")
        self._log_dir = _Fs.join(path, "_flux_log")
        self._meta_path = _Fs.join(path, "_flux_meta.json")
        self._closed = False
        self._entered = False
        self._pool: "ThreadPoolExecutor | None" = None

        # Accumulated state committed at close.
        self._written_files: list[str] = []
        self._file_manifests: list[dict] = []
        self._total_rows: int = 0
        self._part_num: int = 0
        self._meta: dict = {}
        self._base_version: int = 0
        self._active_spec_id: int = 0
        # Deduping key: (source_column, transform) tuple -> spec_id
        self._spec_fingerprints: dict[tuple[tuple[str, str], ...], int] = {}

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> "FluxTableWriter":
        self._apply_mode_and_init()
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        try:
            if exc_type is None:
                self._commit()
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
            self._closed = True

    def _apply_mode_and_init(self) -> None:
        has_data = self._fs.has_data_files(self._data_dir)
        if has_data:
            if self.mode in (MODE_ERROR, MODE_ERROR_IF_EXISTS):
                raise TableExistsError(
                    f"FluxTable already exists with data at {self.path!r}. "
                    f"Use mode='overwrite', mode='append', or mode='ignore'."
                )
            if self.mode == MODE_IGNORE:
                # No-op. Set a flag so write_batch silently drops.
                self._closed = True
                return
            if self.mode == MODE_OVERWRITE:
                self._overwrite()
            elif self.mode == MODE_APPEND:
                pass  # Load existing meta below.
            else:
                raise ValueError(
                    f"Unknown mode {self.mode!r}. Use one of: "
                    f"{MODE_ERROR}, {MODE_OVERWRITE}, {MODE_APPEND}, {MODE_IGNORE}."
                )

        self._fs.makedirs(self._data_dir, exist_ok=True)
        self._fs.makedirs(self._log_dir, exist_ok=True)
        self._load_or_init_meta()

    def _overwrite(self) -> None:
        """Wipe data/log dirs AND _flux_meta.json. On object stores (GCS,
        S3, Azure) there is no true atomic directory rename, so we do a
        best-effort rename-to-trash for prefixes and a direct delete for
        the single meta file. Leftover ``.trash-*`` prefixes are harmless
        because the new table's log doesn't reference them."""
        suffix = f".trash-{uuid.uuid4().hex[:8]}"
        for sub in ("data", "_flux_log"):
            src = _Fs.join(self.path, sub)
            if not self._fs.exists(src):
                continue
            trash = _Fs.join(self.path, sub + suffix)
            moved = False
            try:
                self._fs.rename(src, trash)
                moved = True
            except Exception:
                pass
            try:
                self._fs.rm_recursive(trash if moved else src)
            except Exception:
                pass
        # IMPORTANT: remove the single meta file using rm_file, not
        # rm_recursive. gcsfs.rm(path, recursive=True) can silently no-op
        # on a standalone object, leaving stale partition_specs behind.
        meta = _Fs.join(self.path, "_flux_meta.json")
        if self._fs.exists(meta):
            self._fs.rm_file(meta)

    def _load_or_init_meta(self) -> None:
        if self._fs.exists(self._meta_path):
            self._meta = self._fs.read_json(self._meta_path)
        else:
            self._meta = {
                "table_id": f"flux-{int(time.time() * 1e9):016x}",
                "partition_specs": [],
                "current_spec_id": 0,
                "clustering_columns": [],
                "properties": {},
            }

        # Fingerprint existing specs so we can dedup new ones.
        for spec in self._meta.get("partition_specs", []):
            key = tuple(
                (f.get("source_column", ""), f.get("transform", "identity"))
                for f in spec.get("fields", [])
            )
            self._spec_fingerprints.setdefault(key, int(spec["spec_id"]))

        if self.partition_by is not None:
            key = tuple(
                (p["source_column"], p.get("transform", "identity"))
                for p in self.partition_by
            )
            if key in self._spec_fingerprints:
                self._active_spec_id = self._spec_fingerprints[key]
            else:
                new_id = max(
                    (int(s["spec_id"]) for s in self._meta["partition_specs"]),
                    default=-1,
                ) + 1
                self._meta["partition_specs"].append({
                    "spec_id": new_id,
                    "fields": [
                        {
                            "source_column": p["source_column"],
                            "transform": p.get("transform", "identity"),
                            "field_id": i,
                        }
                        for i, p in enumerate(self.partition_by)
                    ],
                })
                self._spec_fingerprints[key] = new_id
                self._active_spec_id = new_id
            self._meta["current_spec_id"] = self._active_spec_id
        else:
            self._active_spec_id = int(self._meta.get("current_spec_id", 0))

        if self.clustering_columns is not None:
            self._meta["clustering_columns"] = self.clustering_columns

        # Part counter: start after the highest-numbered existing part so we
        # never collide on append.
        self._part_num = _next_part_num(self._fs, self._data_dir)
        self._base_version = _next_version(self._fs, self._log_dir)

    # ── ingress ──────────────────────────────────────────────────────────

    def write_batch(self, batch: Any) -> None:
        """Write any Arrow-compatible batch.

        Accepts polars DataFrame, pandas DataFrame, PyArrow Table/
        RecordBatch, or any object exposing ``.to_arrow()``.
        """
        self._require_open()
        if self._closed:  # mode=ignore silently drops
            return
        table = _to_arrow_table(batch)
        if table.num_rows == 0:
            return
        self._write_arrow_table(table)

    def write_spark_dataframe(
        self,
        sdf: "SparkDataFrame",
        *,
        collect: str = "executors",
    ) -> None:
        """
        Write a PySpark DataFrame.

        ``collect='executors'`` (default) drives ``mapInArrow`` so each
        executor writes its own part files directly to the target path.
        This is the only sane choice for large Spark DataFrames because
        it avoids bringing the data back to the driver.

        ``collect='driver'`` pulls the DataFrame to the driver and writes
        sequentially. Only use for small inputs or local Spark.
        """
        self._require_open()
        if self._closed:
            return

        if collect == "driver":
            # Small path: toArrow()/toPandas() and feed the standard writer.
            try:
                table = sdf.toArrow()  # Spark 4+
            except Exception:
                # Spark 3: round-trip via pandas.
                pdf = sdf.toPandas()
                import pyarrow as pa  # noqa: F401
                table = _to_arrow_table(pdf)
            self._write_arrow_table(table)
            return

        if collect != "executors":
            raise ValueError(
                f"collect={collect!r} not understood; use 'executors' or 'driver'."
            )

        self._spark_mapinarrow_write(sdf)

    def _spark_mapinarrow_write(self, sdf: "SparkDataFrame") -> None:
        """Write from PySpark via mapInArrow so each executor writes
        directly to the target URI."""
        # Capture writer config into a dict so Spark can serialize it into
        # the UDF closure (the writer itself isn't serializable).
        cfg = {
            "data_dir": self._data_dir,
            "profile": self.profile,
            "u64_only": self.u64_only,
            "chunk_size": self.chunk_size,
            "partition_by": self.partition_by,
            "active_spec_id": self._active_spec_id,
            "compute_stats": self.compute_stats,
        }
        # Collect manifests back via a tiny accumulator schema.
        from pyspark.sql.types import (
            StringType, LongType, StructField, StructType,
        )
        out_schema = StructType([
            StructField("path", StringType(), False),
            StructField("partition_values_json", StringType(), False),
            StructField("spec_id", LongType(), False),
            StructField("row_count", LongType(), False),
            StructField("file_size_bytes", LongType(), False),
            StructField("column_stats_json", StringType(), False),
        ])

        def _udf(iterator):  # iterator of pyarrow.RecordBatch
            # Each task gets its own unique part-file prefix so there are
            # no collisions. We use a uuid prefix per task.
            from fluxcompress._table_writer import (
                _Fs, _partition_chunks, _compute_stats_arrow, _write_one_part,
            )
            task_prefix = uuid.uuid4().hex[:12]
            fs = _Fs(cfg["data_dir"])
            counter = 0
            results: list[dict] = []
            for batch in iterator:
                import pyarrow as pa  # noqa: F401
                table = pa.Table.from_batches([batch])
                for pvals, chunk_table in _partition_chunks(
                    table, cfg["partition_by"], cfg["chunk_size"],
                ):
                    fname = f"part-{task_prefix}-{counter:06d}.flux"
                    counter += 1
                    manifest = _write_one_part(
                        fs=fs,
                        data_dir=cfg["data_dir"],
                        filename=fname,
                        table=chunk_table,
                        partition_values=pvals,
                        spec_id=cfg["active_spec_id"],
                        profile=cfg["profile"],
                        u64_only=cfg["u64_only"],
                        compute_stats=cfg["compute_stats"],
                    )
                    results.append(manifest)
            # Emit manifests as a RecordBatch.
            import pyarrow as pa
            if not results:
                return
            arr_struct = {
                "path": pa.array([r["path"] for r in results]),
                "partition_values_json": pa.array(
                    [json.dumps(r["partition_values"]) for r in results]),
                "spec_id": pa.array([r["spec_id"] for r in results], type=pa.int64()),
                "row_count": pa.array([r["row_count"] for r in results], type=pa.int64()),
                "file_size_bytes": pa.array([r["file_size_bytes"] for r in results], type=pa.int64()),
                "column_stats_json": pa.array(
                    [json.dumps(r["column_stats"]) for r in results]),
            }
            yield pa.record_batch(arr_struct)

        manifests_df = sdf.mapInArrow(_udf, schema=out_schema)
        manifest_rows = manifests_df.collect()
        for r in manifest_rows:
            self._written_files.append(r.path)
            self._total_rows += int(r.row_count)
            self._file_manifests.append({
                "path": r.path,
                "partition_values": json.loads(r.partition_values_json),
                "spec_id": int(r.spec_id),
                "row_count": int(r.row_count),
                "file_size_bytes": int(r.file_size_bytes),
                "column_stats": json.loads(r.column_stats_json),
            })

    # ── core write ───────────────────────────────────────────────────────

    def _write_arrow_table(self, table: "pa.Table") -> None:
        # Break the batch into (partition_values, sub-table, filename)
        # tuples before dispatching to the thread pool.
        work: list[tuple[dict, "pa.Table", str]] = []
        for pvals, sub in _partition_chunks(
            table,
            self.partition_by,
            self.chunk_size,
        ):
            filename = f"part-{self._part_num:06d}.flux"
            self._part_num += 1
            work.append((pvals, sub, filename))

        if not work:
            return

        # Parallel compress + write.
        assert self._pool is not None
        futs = [
            self._pool.submit(
                _write_one_part,
                fs=self._fs,
                data_dir=self._data_dir,
                filename=fn,
                table=t,
                partition_values=pvals,
                spec_id=self._active_spec_id,
                profile=self.profile,
                u64_only=self.u64_only,
                compute_stats=self.compute_stats,
            )
            for (pvals, t, fn) in work
        ]
        for fut in as_completed(futs):
            m = fut.result()
            self._written_files.append(m["path"])
            self._file_manifests.append(m)
            self._total_rows += int(m["row_count"])

    # ── commit ───────────────────────────────────────────────────────────

    def _commit(self) -> None:
        """Write meta + single transaction-log entry summarizing all batches."""
        if self._closed:
            return
        # Keep file ordering stable for predictable reads.
        self._written_files.sort()
        self._file_manifests.sort(key=lambda m: m["path"])

        # Re-assert directory existence at commit time. On Databricks
        # Volumes / FUSE mounts an empty directory created by the driver's
        # __enter__ may not have persisted (object stores have no empty
        # prefixes) and can disappear between __enter__ and __exit__.
        self._fs.makedirs(self._log_dir, exist_ok=True)

        self._fs.write_json(self._meta_path, self._meta)

        entry = {
            "version": self._base_version,
            "timestamp_ms": int(time.time() * 1000),
            "operation": "create" if self._base_version == 0 else "append",
            "data_files_added": self._written_files,
            "data_files_removed": [],
            "file_manifests": self._file_manifests,
            "row_count_delta": self._total_rows,
            "metadata": {},
        }
        log_path = _Fs.join(self._log_dir, f"{self._base_version:08d}.json")
        self._fs.write_json(log_path, entry)

    def _require_open(self) -> None:
        if not self._entered:
            raise RuntimeError(
                "FluxTableWriter must be used as a context manager "
                "(`with FluxTableWriter(...) as w: w.write_batch(...)`)."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (module-level so the Spark UDF closure can import them)
# ─────────────────────────────────────────────────────────────────────────────


def _next_part_num(fs: "_Fs", data_dir: str) -> int:
    """Find the next available part-NNNN index. Runs ONCE per session."""
    if fs._fs is not None:
        if not fs._fs.exists(data_dir):
            return 0
        try:
            entries = fs._fs.ls(data_dir, detail=False)
        except FileNotFoundError:
            return 0
        names = [posixpath.basename(str(e)) for e in entries]
    else:
        if not os.path.isdir(data_dir):
            return 0
        try:
            names = os.listdir(data_dir)
        except FileNotFoundError:
            return 0
    hi = -1
    for n in names:
        if not n.startswith("part-") or not n.endswith(".flux"):
            continue
        try:
            # Accept both part-NNNN.flux and part-<taskid>-NNNNNN.flux.
            stem = n[5:-5]
            parts = stem.split("-")
            hi = max(hi, int(parts[-1]))
        except (ValueError, IndexError):
            continue
    return hi + 1


def _next_version(fs: "_Fs", log_dir: str) -> int:
    if fs._fs is not None:
        if not fs._fs.exists(log_dir):
            return 0
        try:
            entries = fs._fs.ls(log_dir, detail=False)
        except FileNotFoundError:
            return 0
        names = [posixpath.basename(str(e)) for e in entries]
    else:
        if not os.path.isdir(log_dir):
            return 0
        try:
            names = os.listdir(log_dir)
        except FileNotFoundError:
            return 0
    hi = -1
    for n in names:
        if n.endswith(".json"):
            try:
                hi = max(hi, int(n[:-5]))
            except ValueError:
                continue
    return hi + 1


def _to_arrow_table(batch: Any) -> "pa.Table":
    """Coerce a batch (polars / pandas / arrow / batch-iter) to pa.Table."""
    import pyarrow as pa

    # PyArrow fast paths.
    if isinstance(batch, pa.Table):
        return batch
    if isinstance(batch, pa.RecordBatch):
        return pa.Table.from_batches([batch])

    # Polars.
    mod = type(batch).__module__
    if mod.startswith("polars"):
        # polars.DataFrame -> pa.Table zero-copy-ish.
        return batch.to_arrow()

    # pandas.
    try:
        import pandas as pd
        if isinstance(batch, pd.DataFrame):
            return pa.Table.from_pandas(batch, preserve_index=False)
    except ImportError:
        pass

    # Generic: duck-type on to_arrow().
    if hasattr(batch, "to_arrow"):
        out = batch.to_arrow()
        if isinstance(out, pa.Table):
            return out
        if isinstance(out, pa.RecordBatch):
            return pa.Table.from_batches([out])

    raise TypeError(
        f"Unsupported batch type {type(batch)!r}. "
        "Provide a polars.DataFrame, pandas.DataFrame, pyarrow.Table / "
        "RecordBatch, or an object implementing .to_arrow()."
    )


def _partition_chunks(
    table: "pa.Table",
    partition_by: "Sequence[Mapping[str, str]] | None",
    chunk_size: int,
) -> "Iterator[tuple[dict, pa.Table]]":
    """Yield (partition_values, sub_table) pairs, sub-chunked to chunk_size."""
    import pyarrow as pa

    if partition_by:
        group_cols = [p["source_column"] for p in partition_by]
        # pyarrow.compute sort + split by distinct group key.
        import pyarrow.compute as pc
        indices = pc.sort_indices(
            table, sort_keys=[(c, "ascending") for c in group_cols],
        )
        sorted_tbl = table.take(indices)
        # Walk consecutive runs of identical group values.
        key_cols = [sorted_tbl.column(c).to_pylist() for c in group_cols]
        n = sorted_tbl.num_rows
        if n == 0:
            return
        start = 0
        cur_key = tuple(col[0] for col in key_cols)
        for i in range(1, n):
            k = tuple(col[i] for col in key_cols)
            if k != cur_key:
                yield from _subchunk(
                    sorted_tbl.slice(start, i - start),
                    {c: str(v) for c, v in zip(group_cols, cur_key)},
                    chunk_size,
                )
                start = i
                cur_key = k
        yield from _subchunk(
            sorted_tbl.slice(start, n - start),
            {c: str(v) for c, v in zip(group_cols, cur_key)},
            chunk_size,
        )
    else:
        yield from _subchunk(table, {}, chunk_size)


def _subchunk(
    table: "pa.Table",
    pvals: dict,
    chunk_size: int,
) -> "Iterator[tuple[dict, pa.Table]]":
    if chunk_size <= 0 or table.num_rows <= chunk_size:
        yield pvals, table
        return
    n = table.num_rows
    start = 0
    while start < n:
        length = min(chunk_size, n - start)
        yield pvals, table.slice(start, length)
        start += length


def _compute_stats_arrow(table: "pa.Table") -> dict:
    """Vectorised per-column min/max/null_count via pyarrow.compute.

    Avoids the slow ``polars .min()/.max()/.null_count()`` loop — a single
    pass of each compute kernel, no Python-side iteration.
    """
    import pyarrow.compute as pc
    stats: dict[str, dict] = {}
    for name in table.column_names:
        col = table.column(name)
        entry: dict = {"null_count": int(col.null_count)}
        try:
            mm = pc.min_max(col)
            entry["min"] = None if mm["min"].as_py() is None else str(mm["min"].as_py())
            entry["max"] = None if mm["max"].as_py() is None else str(mm["max"].as_py())
        except Exception:
            # Types where min_max isn't defined (e.g. structs). Skip.
            pass
        stats[name] = entry
    return stats


def _write_one_part(
    *,
    fs: "_Fs",
    data_dir: str,
    filename: str,
    table: "pa.Table",
    partition_values: dict,
    spec_id: int,
    profile: str,
    u64_only: bool,
    compute_stats: bool,
) -> dict:
    """Compress one partitioned chunk and write it to the target filesystem.

    Module-level so it can be imported inside the Spark UDF closure. We
    always re-assert the parent directory exists here (not just once on the
    driver) because:

    - Databricks Volumes / GCS FUSE mounts don't materialise empty
      directories. A driver-side ``os.makedirs(..., exist_ok=True)`` may
      succeed but still not be visible to executor processes until a real
      file lives under the prefix.
    - Spark executors run in separate filesystem views, so a `makedirs`
      on the driver is not guaranteed to propagate.

    `makedirs(..., exist_ok=True)` is idempotent and cheap, so doing it
    per-task is safe.
    """
    filepath = _Fs.join(data_dir, filename)
    buf: FluxBuffer = compress(table, profile=profile, u64_only=u64_only)
    payload = bytes(buf.to_bytes())

    if fs._fs is not None:
        # Remote (fsspec) FS: write directly; fsspec handles prefix creation.
        fs.write_bytes(filepath, payload)
        size = len(payload)
    else:
        # Local / FUSE-mounted path: ensure parent exists on THIS process's
        # filesystem view, then write. We deliberately use Python's
        # ``open()`` (not the Rust `write_flux` helper) because write_flux
        # relies on the parent already being materialised, which isn't
        # guaranteed on Databricks Volumes.
        parent = os.path.dirname(filepath)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except FileExistsError:
                pass  # Some FUSE mounts raise even with exist_ok=True
        with open(filepath, "wb") as f:
            f.write(payload)
        size = len(payload)

    rel_path = f"data/{filename}"
    col_stats = _compute_stats_arrow(table) if compute_stats else {}
    return {
        "path": rel_path,
        "partition_values": partition_values,
        "spec_id": int(spec_id),
        "row_count": int(table.num_rows),
        "file_size_bytes": int(size),
        "column_stats": col_stats,
    }
