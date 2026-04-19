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
import logging
import os
import posixpath
import tempfile
import time
import uuid
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

_log = logging.getLogger("fluxcompress.table_writer")

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

_REMOTE_BACKENDS_WITH_CONDITIONAL_PUT = {"gs", "gcs", "s3", "s3a"}

_WARNED_ABOUT_WEAK_CPUT: set[str] = set()


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
            self._protocol = path.split("://", 1)[0].lower()
            self._fs = fsspec.filesystem(self._protocol)
        else:
            self._protocol = "file"
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

    def read_bytes(self, path: str) -> bytes:
        """Read the exact bytes at ``path``.

        Used by OCC + CRC validation where we need byte-for-byte fidelity
        (re-serialising through ``json.loads`` would change whitespace).
        """
        if self._fs is not None:
            with self._fs.open(path, "rb") as f:
                return f.read()
        with open(path, "rb") as f:
            return f.read()

    # ── Atomic conditional-PUT primitive ────────────────────────────────
    #
    # ``put_if_absent(path, data)`` returns ``True`` if we successfully
    # wrote new bytes and ``False`` if the destination already existed.
    # It is the single point of serialisation for the log commit path —
    # every writer goes through this primitive to claim its version slot.
    #
    # Local filesystem: ``os.link`` gives POSIX ``O_EXCL`` semantics for
    # free.
    #
    # GCS:  ``if_generation_match=0`` header on upload.
    # S3:   ``If-None-Match: *`` header on PutObject (generally available
    #       since 2024).
    # Other remote backends: fall back to ``exists()`` probe + write +
    # mv, which has a narrow TOCTOU window. A one-time warning is emitted
    # so users building on those backends know their guarantees are
    # weaker than Delta / Iceberg equivalents.

    def put_if_absent(self, path: str, data: bytes) -> bool:
        """Atomic conditional write. Returns ``True`` on success, ``False``
        if ``path`` already exists. Never overwrites."""
        if self._fs is None:
            return self._put_if_absent_local(path, data)
        if self._protocol in ("gs", "gcs"):
            out = self._put_if_absent_gcs(path, data)
            if out is not None:
                return out
        elif self._protocol in ("s3", "s3a"):
            out = self._put_if_absent_s3(path, data)
            if out is not None:
                return out
        return self._put_if_absent_remote_fallback(path, data)

    def _put_if_absent_local(self, path: str, data: bytes) -> bool:
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                os.link(tmp, path)  # atomic O_EXCL semantics
            except FileExistsError:
                return False
            return True
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _put_if_absent_gcs(self, path: str, data: bytes) -> "bool | None":
        """Conditional write for GCS using ``if_generation_match=0``.

        Returns ``None`` if the google-cloud-storage SDK is not available,
        letting the caller fall back. The SDK is a hard requirement for
        strictly-correct concurrent commits on GCS.
        """
        try:
            from google.api_core.exceptions import PreconditionFailed  # type: ignore[import-not-found]
            from google.cloud import storage  # type: ignore[import-not-found]
        except ImportError:
            return None
        bucket_name, key = _parse_gs_uri(path)
        try:
            client = storage.Client()
        except Exception:
            # Auth or ADC unavailable — fall back.
            return None
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        try:
            blob.upload_from_string(data, if_generation_match=0)
            return True
        except PreconditionFailed:
            return False

    def _put_if_absent_s3(self, path: str, data: bytes) -> "bool | None":
        """Conditional write for S3 using ``IfNoneMatch='*'``.

        Returns ``None`` if boto3 is not available.
        """
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.exceptions import ClientError  # type: ignore[import-not-found]
        except ImportError:
            return None
        bucket_name, key = _parse_s3_uri(path)
        try:
            client = boto3.client("s3")
        except Exception:
            return None
        try:
            client.put_object(Bucket=bucket_name, Key=key, Body=data, IfNoneMatch="*")
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            # S3 reports a precondition failure in one of several forms
            # depending on region / API version. Treat all as "lost the race".
            if code in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
                "412",
            }:
                return False
            raise

    def _put_if_absent_remote_fallback(self, path: str, data: bytes) -> bool:
        """Best-effort conditional write for backends without native support.

        Performs an ``exists()`` probe, writes to a tmp sibling, then
        moves into place. This carries a TOCTOU window; emits a one-time
        warning per protocol so users can diagnose weak concurrency
        guarantees.
        """
        assert self._fs is not None
        proto = self._protocol
        if proto not in _REMOTE_BACKENDS_WITH_CONDITIONAL_PUT:
            if proto not in _WARNED_ABOUT_WEAK_CPUT:
                _WARNED_ABOUT_WEAK_CPUT.add(proto)
                warnings.warn(
                    f"FluxTableWriter: {proto!r} does not have a native "
                    "conditional-PUT path; using exists() + write + mv. "
                    "Concurrent writers may race; install the native "
                    "cloud SDK (google-cloud-storage / boto3) where "
                    "possible, or serialise writers externally.",
                    stacklevel=2,
                )
        if self._fs.exists(path):
            return False
        tmp = path + f".tmp-{uuid.uuid4().hex[:12]}"
        with self._fs.open(tmp, "wb") as f:
            f.write(data)
        try:
            self._fs.mv(tmp, path)
        except Exception:
            if self._fs.exists(path):
                try:
                    self._fs.rm_file(tmp)
                except Exception:
                    pass
                return False
            with self._fs.open(path, "wb") as f:
                f.write(data)
            try:
                self._fs.rm_file(tmp)
            except Exception:
                pass
        return True

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


# ──────────────────────────────────────────────────────────────────────────────
# Writer
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHUNK_SIZE = 500_000

# Checkpoint the log every N commits by default. Matches Delta's default.
_DEFAULT_CHECKPOINT_INTERVAL = 10
# How many times to retry a conflicting commit before giving up.
_MAX_COMMIT_RETRIES = 10
# Log-entry filename used for the materialised snapshot at a given version.
_CHECKPOINT_SUFFIX = ".checkpoint.json"
# Pointer file at the table root naming the most recent checkpoint.
_LAST_CHECKPOINT_FILE = "_last_checkpoint.json"

# Reader / writer protocol versions recorded in every new log entry.
# Readers refuse tables that declare a reader_min_version higher than this,
# and writers refuse tables that declare a writer_min_version higher than
# this. Bump these (carefully) when adding features that older code cannot
# safely read or write.
FLUX_TABLE_READER_VERSION = 1
FLUX_TABLE_WRITER_VERSION = 2  # v2: actions array + CRC validation + txn


class CommitConflictError(RuntimeError):
    """Raised when a FluxTableWriter commit fails the OCC check even after
    ``_MAX_COMMIT_RETRIES`` attempts. The caller's data files have already
    been written; the safe recovery is to remove those orphan files or to
    re-invoke the writer in ``mode='append'`` so a later commit sweeps them
    into a fresh log entry."""


class ConcurrentOperationError(CommitConflictError):
    """Raised when a concurrent writer landed a commit whose actions
    semantically conflict with this writer's intended actions.

    Unlike bare version races (which we retry), semantic conflicts are
    hard failures: the caller must rebuild the in-memory state that
    depended on the stale view and re-issue the operation.

    Attributes
    ----------
    kind:
        One of ``"remove"``, ``"metadata"``, ``"protocol"``, ``"txn"``.
    conflicting_version:
        The log version of the concurrent commit that caused the
        conflict, for debugging.
    """

    def __init__(
        self,
        kind: str,
        conflicting_version: int,
        details: str = "",
    ) -> None:
        self.kind = kind
        self.conflicting_version = conflicting_version
        self.details = details
        msg = (
            f"Concurrent {kind} conflict at v{conflicting_version}"
            + (f": {details}" if details else "")
        )
        super().__init__(msg)


class LogForkError(RuntimeError):
    """Raised when ``_replay_log`` detects a broken parent-CRC chain.

    Means two writers committed different bytes at the same version slot
    (should be impossible if the atomic claim succeeded), or the log was
    manually edited / corrupted. The table is not safe to read without
    operator intervention; pass ``strict=False`` to ``_replay_log`` to
    replay up to the last valid version for repair workflows.
    """

    def __init__(self, version: int, expected_crc: int, found_crc: int) -> None:
        self.version = version
        self.expected_crc = expected_crc
        self.found_crc = found_crc
        super().__init__(
            f"Log fork detected at v{version}: declared "
            f"parent_version_crc32=0x{found_crc:08x} but actual bytes of "
            f"v{version - 1} hash to 0x{expected_crc:08x}."
        )


class ProtocolVersionError(RuntimeError):
    """Raised when a reader or writer cannot safely operate on a table
    because the table's declared minimum protocol version is higher than
    what this client implements. Upgrade ``fluxcompress`` to proceed."""


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
    checkpoint_interval:
        Emit a consolidated ``NNNNNNNN.checkpoint.json`` every N successful
        commits. Checkpoints let readers skip replaying hundreds of
        incremental log entries. Set to ``0`` to disable. Default ``10``.
    writer_id:
        Optional string identifier for this writer session, recorded in
        each log entry for post-hoc debugging of OCC conflicts. Defaults
        to a fresh UUID.
    txn_app_id:
        Optional idempotency token identifying the writer application
        (e.g. the Spark Streaming query id). When paired with
        ``txn_version`` and a commit with the same ``(app_id, version)``
        pair has already landed, this commit becomes a silent no-op.
        Use this to make retry-heavy job orchestrators exactly-once safe.
    txn_version:
        Monotonically increasing version number per ``txn_app_id``.
        Required when ``txn_app_id`` is set.
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
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
        writer_id: "str | None" = None,
        txn_app_id: "str | None" = None,
        txn_version: "int | None" = None,
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
        self.checkpoint_interval = max(int(checkpoint_interval), 0)
        self.writer_id = writer_id or uuid.uuid4().hex
        if (txn_app_id is None) ^ (txn_version is None):
            raise ValueError(
                "txn_app_id and txn_version must be provided together"
            )
        self.txn_app_id = txn_app_id
        self.txn_version = int(txn_version) if txn_version is not None else None
        # Did the caller explicitly configure table metadata? If not, and
        # we later observe a concurrent writer that already committed
        # metadata (common on fresh-table races where table_id is auto
        # generated per session), we silently rebase onto theirs rather
        # than raising ConcurrentOperationError. Explicit changes —
        # partitioning or clustering the caller asked for — always
        # conflict deterministically.
        self._user_configured_metadata = (
            partition_by is not None or clustering_columns is not None
        )

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
        # Snapshot of the metadata already committed in the log. Used to
        # decide whether this writer session needs to emit a new metadata
        # action on commit (Phase 4).
        self._committed_meta: "dict | None" = None
        self._pending_removes: list[str] = []
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
        # Phase 4: log is authoritative. Prefer metadata derived from the
        # log over the _flux_meta.json side file, since concurrent writers
        # can race the side file last-write-wins. The side file stays as
        # a convenience cache for tools that don't want to replay.
        log_meta = _replay_metadata(
            self._fs, self._log_dir, table_path=self.path,
        )
        if log_meta is not None:
            self._meta = log_meta
            self._committed_meta = _clone_meta(log_meta)
        elif self._fs.exists(self._meta_path):
            self._meta = self._fs.read_json(self._meta_path)
            # Legacy table with no metadata action yet: treat the side
            # file as the committed baseline so we don't emit a spurious
            # metadata action on the first append.
            self._committed_meta = _clone_meta(self._meta)
        else:
            self._meta = {
                "table_id": f"flux-{int(time.time() * 1e9):016x}",
                "partition_specs": [],
                "current_spec_id": 0,
                "clustering_columns": [],
                "properties": {},
            }
            self._committed_meta = None  # Fresh table: emit on v0.

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

    # ── commit ─────────────────────────────────────────────────────────────────────

    def _commit(self) -> None:
        """Commit accumulated writes with a conflict-aware OCC retry loop.

        The commit sequence is:

        1. If ``txn_app_id`` is set, check whether the log already contains
           a matching or newer ``txn`` action. If so, the commit is a silent
           no-op (Phase 5 — idempotent transactions).
        2. Build the list of actions this commit will carry (Phase 3).
        3. Enter the OCC retry loop. At each attempt:
             a. Find the next free log version.
             b. If any entries have landed since the writer session started,
                scan them for **semantic** conflicts against our actions
                (overlapping removes, concurrent metadata / protocol / txn
                changes). On conflict, raise ``ConcurrentOperationError``
                rather than retrying — the caller needs to rebuild.
             c. Compute ``parent_version_crc32`` from the bytes on disk.
             d. Call ``_Fs.put_if_absent`` which uses backend-native
                conditional-PUT primitives (``os.link`` / GCS
                ``if_generation_match=0`` / S3 ``IfNoneMatch='*'``). On
                conflict, retry at the next version.
        4. After a successful claim, refresh the ``_flux_meta.json`` cache
           and optionally emit a checkpoint.

        Data files (``data/part-*.flux``) are already uploaded by the time
        we get here; they carry unique names so retries never have to
        rewrite them.
        """
        if self._closed:
            return

        # Idempotent transaction dedup (Phase 5).
        if self.txn_app_id is not None and self.txn_version is not None:
            prior = _find_committed_txn_version(
                self._fs, self._log_dir, self.txn_app_id,
            )
            if prior is not None and prior >= self.txn_version:
                _log.info(
                    "FluxTableWriter: skipping commit — txn (%s, %d) already "
                    "committed at or after this version (last seen %d).",
                    self.txn_app_id, self.txn_version, prior,
                )
                self._closed = True
                return

        # Nothing to do: no files, no metadata change, no txn marker.
        needs_metadata_action = self._committed_meta is None or (
            _clone_meta(self._meta) != self._committed_meta
        )
        if (
            not self._written_files
            and not self._pending_removes
            and not needs_metadata_action
            and self.txn_app_id is None
        ):
            return

        self._written_files.sort()
        self._file_manifests.sort(key=lambda m: m["path"])
        self._pending_removes.sort()

        # Re-assert directory existence: Volumes / FUSE can drop empty dirs.
        self._fs.makedirs(self._log_dir, exist_ok=True)

        our_actions = self._build_actions(
            include_metadata=needs_metadata_action,
        )

        base_seen = self._base_version  # what we saw at load time
        last_err: "Exception | None" = None
        for attempt in range(_MAX_COMMIT_RETRIES):
            target_version = _next_version(self._fs, self._log_dir)

            # Phase 3: semantic conflict detection. Any entry committed at
            # version in (base_seen, target_version) is a concurrent
            # writer we hadn't seen at load time; check that their actions
            # don't conflict with ours before attempting a claim.
            if target_version > base_seen:
                conflict = _detect_conflicts(
                    self._fs, self._log_dir,
                    from_version_inclusive=base_seen,
                    to_version_exclusive=target_version,
                    our_actions=our_actions,
                    our_txn_app_id=self.txn_app_id,
                    our_txn_version=self.txn_version,
                )
                if conflict is not None:
                    # Rebase path: if the conflict is purely metadata
                    # and the caller didn't explicitly ask for a
                    # metadata change, adopt whatever the concurrent
                    # writer committed and drop our metadata action.
                    # Pure-append sessions then continue without
                    # spurious failures on fresh-table races.
                    if (
                        conflict.kind == "metadata"
                        and not self._user_configured_metadata
                        and needs_metadata_action
                    ):
                        fresh_meta = _replay_metadata(
                            self._fs, self._log_dir, table_path=self.path,
                        )
                        if fresh_meta is not None:
                            self._meta = fresh_meta
                            self._committed_meta = _clone_meta(fresh_meta)
                            self._active_spec_id = int(
                                fresh_meta.get("current_spec_id", 0)
                            )
                            needs_metadata_action = False
                            our_actions = self._build_actions(
                                include_metadata=False,
                            )
                            base_seen = target_version
                            continue
                    raise conflict

            parent_crc = _crc32_of_committed_entry(
                self._fs, self._log_dir, target_version - 1,
            ) if target_version > 0 else 0

            entry = self._build_entry(
                target_version=target_version,
                parent_crc=parent_crc,
                actions=our_actions,
                include_metadata=needs_metadata_action,
            )
            entry_bytes = json.dumps(entry, indent=2).encode("utf-8")
            log_path = _Fs.join(self._log_dir, f"{target_version:08d}.json")
            try:
                if self._fs.put_if_absent(log_path, entry_bytes):
                    self._base_version = target_version
                    if needs_metadata_action:
                        self._committed_meta = _clone_meta(self._meta)
                    break
            except Exception as exc:
                last_err = exc  # fall through to retry
            # Race lost or transient error — bump our "seen" watermark so the
            # next retry's conflict check only considers newer commits.
            base_seen = target_version + 1
        else:
            raise CommitConflictError(
                f"FluxTable commit failed after {_MAX_COMMIT_RETRIES} OCC retries "
                f"at {self.path!r}. "
                f"Last error: {last_err!r}. "
                f"Orphan data files may remain in {self._data_dir!r} — remove or "
                f"reuse via mode='append' in a subsequent writer session."
            ) from last_err

        # Refresh the cache. A failure here is safe because the log is the
        # authoritative source of metadata (Phase 4) — readers that don't
        # trust the cache will reconstruct from the log.
        try:
            self._fs.write_json(self._meta_path, self._meta)
        except Exception:  # pragma: no cover — best-effort cache update
            _log.warning(
                "FluxTableWriter: failed to refresh _flux_meta.json cache; "
                "log remains canonical.", exc_info=True,
            )

        # Optional checkpoint.
        if (
            self.checkpoint_interval > 0
            and self._base_version > 0
            and (self._base_version + 1) % self.checkpoint_interval == 0
        ):
            try:
                self._write_checkpoint()
            except Exception:
                # Checkpoints are an optimization; never fail the commit
                # because one couldn't be written.
                _log.warning(
                    "FluxTableWriter: checkpoint write failed at v%d; "
                    "reader will fall back to full-log replay.",
                    self._base_version, exc_info=True,
                )

    # ── commit helpers ────────────────────────────────────────────────────

    def _build_actions(self, *, include_metadata: bool) -> list[dict]:
        """Assemble the ``actions`` array for the current commit.

        Action kinds (see ``docs/format/fluxtable_format.md``):

        * ``protocol`` — emitted only at v0 of a fresh table.
        * ``metadata`` — emitted whenever the caller changed schema /
          partitioning / clustering / properties vs the committed baseline.
        * ``add`` — one per newly-created data file.
        * ``remove`` — one per retired data file.
        * ``txn`` — idempotency marker, if the caller provided one.
        * ``commit_info`` — provenance; always last.
        """
        actions: list[dict] = []
        if self._committed_meta is None:
            actions.append({
                "type": "protocol",
                "reader_min_version": FLUX_TABLE_READER_VERSION,
                "writer_min_version": FLUX_TABLE_WRITER_VERSION,
            })
        if include_metadata:
            actions.append({
                "type": "metadata",
                "table_id": self._meta.get("table_id"),
                "partition_specs": list(self._meta.get("partition_specs", [])),
                "current_spec_id": int(self._meta.get("current_spec_id", 0)),
                "clustering_columns": list(
                    self._meta.get("clustering_columns", [])
                ),
                "properties": dict(self._meta.get("properties", {})),
            })
        # Manifests are indexed by path so we can attach them to ``add``
        # actions without duplicating the flat list.
        manifests_by_path = {m["path"]: m for m in self._file_manifests}
        for p in self._written_files:
            actions.append({
                "type": "add",
                "path": p,
                "manifest": manifests_by_path.get(p, {}),
            })
        for p in self._pending_removes:
            actions.append({"type": "remove", "path": p})
        if self.txn_app_id is not None and self.txn_version is not None:
            actions.append({
                "type": "txn",
                "app_id": self.txn_app_id,
                "version": int(self.txn_version),
            })
        actions.append({
            "type": "commit_info",
            "writer_id": self.writer_id,
            "writer_version": FLUX_TABLE_WRITER_VERSION,
            "operation": _infer_operation(
                has_adds=bool(self._written_files),
                has_removes=bool(self._pending_removes),
                is_first=self._committed_meta is None,
            ),
            "row_count_delta": int(self._total_rows),
        })
        return actions

    def _build_entry(
        self,
        *,
        target_version: int,
        parent_crc: int,
        actions: list[dict],
        include_metadata: bool,
    ) -> dict:
        """Render a log entry dict carrying both the canonical ``actions``
        array (v2+) and the legacy flat fields (v1) for backward
        compatibility with older readers."""
        op = _infer_operation(
            has_adds=bool(self._written_files),
            has_removes=bool(self._pending_removes),
            is_first=target_version == 0,
        )
        return {
            "version": target_version,
            "timestamp_ms": int(time.time() * 1000),
            "operation": op,
            "parent_version_crc32": parent_crc,
            "writer_id": self.writer_id,
            "writer_version": FLUX_TABLE_WRITER_VERSION,
            "actions": actions,
            # Legacy flat fields — readers that pre-date the actions array
            # continue to see the same data.
            "data_files_added": list(self._written_files),
            "data_files_removed": list(self._pending_removes),
            "file_manifests": list(self._file_manifests),
            "row_count_delta": int(self._total_rows),
            "metadata": {},
        }

    def _write_checkpoint(self) -> None:
        """Write a materialised snapshot of the live set at ``self._base_version``
        plus a hardened ``_last_checkpoint.json`` pointer.

        Phase 6: the checkpoint file itself is written via the atomic
        conditional-PUT primitive, and the pointer file is only updated
        if the version we're publishing is strictly newer than whatever
        pointer is currently on disk. That closes the old
        last-write-wins race where two concurrent writers racing to emit
        a checkpoint could leave the pointer naming an older version than
        the most recent checkpoint file.
        """
        live, manifests = _replay_log(
            self._fs, self._log_dir,
            up_to_version=self._base_version,
            table_path=self.path,
            strict=False,  # checkpoints should survive a mildly-broken tail
        )
        checkpoint = {
            "version": self._base_version,
            "timestamp_ms": int(time.time() * 1000),
            "is_checkpoint": True,
            "live_files": sorted(live),
            "file_manifests": [
                manifests[f] for f in sorted(live) if f in manifests
            ],
            "meta_snapshot": self._meta,
        }
        cp_name = f"{self._base_version:08d}{_CHECKPOINT_SUFFIX}"
        cp_path = _Fs.join(self._log_dir, cp_name)
        cp_bytes = json.dumps(checkpoint, indent=2).encode("utf-8")
        # If the checkpoint file already exists (another writer already
        # emitted it), we don't rewrite it — the content is deterministic
        # modulo timestamp, and the existing one is safe to keep.
        self._fs.put_if_absent(cp_path, cp_bytes)

        pointer_path = _Fs.join(self.path, _LAST_CHECKPOINT_FILE)
        _update_checkpoint_pointer(
            self._fs, pointer_path, self._base_version, cp_name,
        )

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
        # Skip checkpoints: they end in .checkpoint.json and are not the
        # primary version entries. int(n[:-5]) already rejects them via
        # ValueError, but be explicit for clarity.
        if not n.endswith(".json") or n.endswith(_CHECKPOINT_SUFFIX):
            continue
        try:
            hi = max(hi, int(n[:-5]))
        except ValueError:
            continue
    return hi + 1


def _list_log_entries(fs: "_Fs", log_dir: str) -> list[tuple[int, str]]:
    """Return sorted ``(version, basename)`` for every non-checkpoint log entry."""
    if fs._fs is not None:
        if not fs._fs.exists(log_dir):
            return []
        try:
            names = [posixpath.basename(str(e))
                     for e in fs._fs.ls(log_dir, detail=False)]
        except FileNotFoundError:
            return []
    else:
        if not os.path.isdir(log_dir):
            return []
        try:
            names = os.listdir(log_dir)
        except FileNotFoundError:
            return []
    out: list[tuple[int, str]] = []
    for n in names:
        if not n.endswith(".json") or n.endswith(_CHECKPOINT_SUFFIX):
            continue
        try:
            out.append((int(n[:-5]), n))
        except ValueError:
            continue
    out.sort()
    return out


def _read_json(fs: "_Fs", path: str) -> "dict | None":
    """Read a JSON file, returning ``None`` if missing or unparseable."""
    try:
        return fs.read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    except Exception:
        return None


def _replay_log(
    fs: "_Fs",
    log_dir: str,
    *,
    up_to_version: "int | None" = None,
    table_path: "str | None" = None,
    strict: bool = True,
    validate_protocol: bool = True,
) -> "tuple[set[str], dict[str, dict]]":
    """Materialise ``(live_files, manifests)`` by replaying the log.

    If ``table_path`` is provided and a ``_last_checkpoint.json`` pointer
    exists at that location, we load the checkpoint as the initial state
    and replay only the log entries strictly after the checkpoint version.
    Otherwise we replay every entry from version 0.

    If ``up_to_version`` is set, entries with version > ``up_to_version``
    are ignored (used when emitting a new checkpoint).

    Phase 2: each entry's ``parent_version_crc32`` is validated against
    the CRC32 of the raw bytes of the prior committed log entry. On
    mismatch we raise :class:`LogForkError` by default; pass
    ``strict=False`` to tolerate a broken tail (replay stops at the last
    valid version and a warning is emitted).

    Phase 3: if an entry carries an ``actions`` array we replay the
    actions; otherwise we fall back to the legacy flat fields. Both paths
    produce identical ``(live_files, manifests)`` results.

    Phase 4: ``validate_protocol`` makes a best-effort check of the
    highest declared ``writer_min_version`` / ``reader_min_version`` and
    raises :class:`ProtocolVersionError` if this client can't safely read
    the table.
    """
    live: set[str] = set()
    manifests: dict[str, dict] = {}
    start_version = 0

    if table_path is not None:
        pointer_path = _Fs.join(table_path, _LAST_CHECKPOINT_FILE)
        if fs.exists(pointer_path):
            ptr = _read_json(fs, pointer_path)
            if ptr and isinstance(ptr, dict) and "path" in ptr:
                cp_full = _Fs.join(table_path, ptr["path"])
                cp = _read_json(fs, cp_full)
                if cp and cp.get("is_checkpoint"):
                    live.update(cp.get("live_files", []))
                    for fm in cp.get("file_manifests", []):
                        manifests[fm["path"]] = fm
                    start_version = int(cp.get("version", 0)) + 1

    entries = _list_log_entries(fs, log_dir)

    # CRC validation piggy-backs on a single sequential pass: we keep the
    # raw bytes of the prior entry around so we can compute its CRC in
    # O(n) total instead of re-reading every ancestor per entry.
    prev_bytes: "bytes | None" = None
    prev_version: int = -1

    reader_max_required = 0
    writer_max_required = 0

    for version, name in entries:
        if up_to_version is not None and version > up_to_version:
            break
        path = _Fs.join(log_dir, name)
        try:
            entry_bytes = fs.read_bytes(path)
        except FileNotFoundError:
            continue
        try:
            entry = json.loads(entry_bytes)
        except json.JSONDecodeError:
            if strict:
                raise
            warnings.warn(
                f"FluxTable: v{version} log entry is unparseable JSON; "
                "stopping replay at last valid version.",
                stacklevel=2,
            )
            break

        # CRC chain validation. We can only check when we have the
        # parent's raw bytes on hand: if the parent entry sits before
        # ``start_version`` (because we loaded a checkpoint) we skip the
        # first check; that's safe because the checkpoint itself is
        # authoritative for the state up to its version.
        if (
            version > 0
            and prev_bytes is not None
            and prev_version == version - 1
        ):
            expected = zlib.crc32(prev_bytes) & 0xFFFFFFFF
            declared = int(entry.get("parent_version_crc32", 0))
            if declared != 0 and declared != expected:
                err = LogForkError(version, expected, declared)
                if strict:
                    raise err
                warnings.warn(
                    f"FluxTable: {err}; replaying up to last valid version.",
                    stacklevel=2,
                )
                break

        # Track protocol requirements across the entire log.
        proto = _entry_protocol(entry)
        if proto is not None:
            reader_max_required = max(
                reader_max_required, int(proto.get("reader_min_version", 0))
            )
            writer_max_required = max(
                writer_max_required, int(proto.get("writer_min_version", 0))
            )

        if version >= start_version:
            _apply_entry(entry, live, manifests)

        prev_bytes = entry_bytes
        prev_version = version

    if validate_protocol:
        if reader_max_required > FLUX_TABLE_READER_VERSION:
            raise ProtocolVersionError(
                f"Table requires reader_min_version={reader_max_required}, "
                f"but this fluxcompress client implements "
                f"{FLUX_TABLE_READER_VERSION}. Upgrade fluxcompress to read."
            )

    return live, manifests


def _apply_entry(
    entry: dict, live: set[str], manifests: dict[str, dict],
) -> None:
    """Apply an entry's ``add`` / ``remove`` / ``metadata`` effects on the
    in-memory live set. Honours the canonical ``actions`` array when
    present; falls back to the flat legacy fields for pre-v2 entries."""
    actions = entry.get("actions")
    if isinstance(actions, list) and actions:
        for a in actions:
            kind = a.get("type")
            if kind == "add":
                p = a.get("path")
                if p is None:
                    continue
                live.add(p)
                m = a.get("manifest")
                if isinstance(m, dict) and m:
                    manifests[p] = m
            elif kind == "remove":
                p = a.get("path")
                if p is None:
                    continue
                live.discard(p)
                manifests.pop(p, None)
        return
    # Legacy flat-fields entry.
    for added in entry.get("data_files_added", []):
        live.add(added)
    for removed in entry.get("data_files_removed", []):
        live.discard(removed)
        manifests.pop(removed, None)
    for fm in entry.get("file_manifests", []):
        manifests[fm["path"]] = fm


# ─────────────────────────────────────────────────────────────────────────────
# Action / entry accessors
# ─────────────────────────────────────────────────────────────────────────────


def _entry_actions(entry: dict) -> list[dict]:
    """Return the actions array for an entry, synthesising one from the
    legacy flat fields if the entry pre-dates the v2 format.

    The synthesised actions carry enough signal for conflict detection
    (``add`` / ``remove`` paths) but lack manifest payloads and
    ``commit_info`` — callers that need those should go through
    ``entry['actions']`` directly and bail out on legacy entries.
    """
    acts = entry.get("actions")
    if isinstance(acts, list):
        return acts
    synth: list[dict] = []
    for p in entry.get("data_files_added", []) or []:
        synth.append({"type": "add", "path": p})
    for p in entry.get("data_files_removed", []) or []:
        synth.append({"type": "remove", "path": p})
    return synth


def _entry_protocol(entry: dict) -> "dict | None":
    for a in _entry_actions(entry):
        if a.get("type") == "protocol":
            return a
    return None


def _entry_metadata(entry: dict) -> "dict | None":
    for a in _entry_actions(entry):
        if a.get("type") == "metadata":
            return a
    return None


def _entry_txn(entry: dict) -> "dict | None":
    for a in _entry_actions(entry):
        if a.get("type") == "txn":
            return a
    return None


def _entry_adds(entry: dict) -> set[str]:
    out: set[str] = set()
    for a in _entry_actions(entry):
        if a.get("type") == "add":
            p = a.get("path")
            if isinstance(p, str):
                out.add(p)
    return out


def _entry_removes(entry: dict) -> set[str]:
    out: set[str] = set()
    for a in _entry_actions(entry):
        if a.get("type") == "remove":
            p = a.get("path")
            if isinstance(p, str):
                out.add(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Semantic conflict detection (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────


def _detect_conflicts(
    fs: "_Fs",
    log_dir: str,
    *,
    from_version_inclusive: int,
    to_version_exclusive: int,
    our_actions: list[dict],
    our_txn_app_id: "str | None" = None,
    our_txn_version: "int | None" = None,
) -> "ConcurrentOperationError | None":
    """Scan committed entries in ``[from, to)`` for semantic conflicts.

    Classifies conflicts the way Delta does:

    * Our ``remove`` overlapping another writer's ``add`` or ``remove``
      on the same path → conflict (the file we wanted to retire was
      already touched).
    * Our ``metadata`` action racing another writer's ``metadata`` →
      conflict (metadata changes serialise).
    * Our ``protocol`` action racing another writer's ``protocol`` →
      conflict.
    * Duplicate txn ``(app_id, version)`` pair → conflict (the caller
      would have been caught by the pre-commit check if the commit
      landed before we started, but this guards the race window).

    Returns ``None`` if the concurrent commits are compatible with ours
    (append-vs-append is the common case and always succeeds).
    """
    our_removes = {
        a["path"] for a in our_actions
        if a.get("type") == "remove" and isinstance(a.get("path"), str)
    }
    our_metadata = next(
        (a for a in our_actions if a.get("type") == "metadata"), None,
    )
    our_protocol = next(
        (a for a in our_actions if a.get("type") == "protocol"), None,
    )

    for version, name in _list_log_entries(fs, log_dir):
        if version < from_version_inclusive:
            continue
        if version >= to_version_exclusive:
            break
        entry = _read_json(fs, _Fs.join(log_dir, name))
        if entry is None:
            continue

        their_adds = _entry_adds(entry)
        their_removes = _entry_removes(entry)

        overlap = our_removes & (their_adds | their_removes)
        if overlap:
            return ConcurrentOperationError(
                "remove", version,
                f"files {sorted(overlap)} already modified by concurrent writer",
            )

        if our_metadata is not None:
            their_metadata = _entry_metadata(entry)
            # Two writers publishing byte-for-byte identical metadata
            # (common on a fresh table where both sessions emit the
            # default spec) is a benign race — accept it silently.
            if (
                their_metadata is not None
                and not _metadata_bodies_equal(our_metadata, their_metadata)
            ):
                return ConcurrentOperationError(
                    "metadata", version,
                    "another writer already committed a different metadata change",
                )

        if our_protocol is not None:
            their_protocol = _entry_protocol(entry)
            if (
                their_protocol is not None
                and not _protocol_bodies_equal(our_protocol, their_protocol)
            ):
                return ConcurrentOperationError(
                    "protocol", version,
                    "another writer already committed a different protocol",
                )

        if (
            our_txn_app_id is not None
            and our_txn_version is not None
        ):
            their_txn = _entry_txn(entry)
            if (
                their_txn is not None
                and their_txn.get("app_id") == our_txn_app_id
                and int(their_txn.get("version", -1)) >= int(our_txn_version)
            ):
                return ConcurrentOperationError(
                    "txn", version,
                    f"txn ({our_txn_app_id}, {our_txn_version}) already "
                    f"committed while we were preparing",
                )
    return None


def _metadata_bodies_equal(a: dict, b: dict) -> bool:
    """Compare two metadata actions ignoring the ``type`` key and
    non-semantic wrappers.

    A concurrent writer that emits byte-for-byte identical metadata is
    not actually racing us — accepting their commit silently keeps
    pure-append workflows from spuriously failing on fresh tables.
    """
    def _body(m: dict) -> dict:
        return {
            "table_id": m.get("table_id"),
            "partition_specs": m.get("partition_specs", []),
            "current_spec_id": int(m.get("current_spec_id", 0)),
            "clustering_columns": list(m.get("clustering_columns", [])),
            "properties": dict(m.get("properties", {})),
        }
    return _body(a) == _body(b)


def _protocol_bodies_equal(a: dict, b: dict) -> bool:
    return (
        int(a.get("reader_min_version", 0))
        == int(b.get("reader_min_version", 0))
        and int(a.get("writer_min_version", 0))
        == int(b.get("writer_min_version", 0))
    )


def _find_committed_txn_version(
    fs: "_Fs", log_dir: str, app_id: str,
) -> "int | None":
    """Return the highest ``txn.version`` already committed for ``app_id``,
    or ``None`` if no such ``txn`` action has landed."""
    best: "int | None" = None
    for _, name in _list_log_entries(fs, log_dir):
        entry = _read_json(fs, _Fs.join(log_dir, name))
        if entry is None:
            continue
        t = _entry_txn(entry)
        if t is None or t.get("app_id") != app_id:
            continue
        v = int(t.get("version", -1))
        if best is None or v > best:
            best = v
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Metadata replay (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────


def _replay_metadata(
    fs: "_Fs", log_dir: str, *, table_path: "str | None" = None,
) -> "dict | None":
    """Reconstruct the current table metadata from the log.

    Returns ``None`` if no ``metadata`` action has been committed yet,
    letting callers fall back to the legacy ``_flux_meta.json`` side
    file. When a checkpoint is present we prefer its embedded
    ``meta_snapshot`` as the seed, then overlay any post-checkpoint
    ``metadata`` actions.
    """
    current: "dict | None" = None
    start_version = 0

    if table_path is not None:
        pointer_path = _Fs.join(table_path, _LAST_CHECKPOINT_FILE)
        if fs.exists(pointer_path):
            ptr = _read_json(fs, pointer_path)
            if ptr and isinstance(ptr, dict) and "path" in ptr:
                cp = _read_json(fs, _Fs.join(table_path, ptr["path"]))
                if cp and cp.get("is_checkpoint"):
                    snap = cp.get("meta_snapshot")
                    if isinstance(snap, dict):
                        current = _clone_meta(snap)
                    start_version = int(cp.get("version", 0)) + 1

    for version, name in _list_log_entries(fs, log_dir):
        if version < start_version:
            continue
        entry = _read_json(fs, _Fs.join(log_dir, name))
        if entry is None:
            continue
        m = _entry_metadata(entry)
        if m is None:
            continue
        # Rebuild the meta dict from the metadata action.
        current = {
            "table_id": m.get("table_id"),
            "partition_specs": list(m.get("partition_specs", [])),
            "current_spec_id": int(m.get("current_spec_id", 0)),
            "clustering_columns": list(m.get("clustering_columns", [])),
            "properties": dict(m.get("properties", {})),
        }
    return current


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────


def _crc32_of_committed_entry(
    fs: "_Fs", log_dir: str, version: int,
) -> int:
    """Return ``CRC32(bytes)`` of the log entry at ``version``, or ``0``
    if the entry is missing (e.g. the caller is committing v0, or the
    log has been truncated)."""
    if version < 0:
        return 0
    path = _Fs.join(log_dir, f"{version:08d}.json")
    if not fs.exists(path):
        return 0
    try:
        data = fs.read_bytes(path)
    except Exception:
        return 0
    return zlib.crc32(data) & 0xFFFFFFFF


def _clone_meta(meta: dict) -> dict:
    """Deep-copy a metadata dict for equality comparison. JSON round-trip
    is the simplest way to drop references and normalise ints."""
    return json.loads(json.dumps(meta))


def _infer_operation(
    *, has_adds: bool, has_removes: bool, is_first: bool,
) -> str:
    if is_first:
        return "create"
    if has_removes and has_adds:
        return "compact"
    if has_removes:
        return "delete"
    if has_adds:
        return "append"
    return "metadata"


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/key/with/slashes`` into ``(bucket, key)``."""
    if not (uri.startswith("gs://") or uri.startswith("gcs://")):
        raise ValueError(f"Not a GCS URI: {uri!r}")
    stripped = uri.split("://", 1)[1]
    bucket, _, key = stripped.partition("/")
    if not bucket or not key:
        raise ValueError(f"GCS URI missing bucket or key: {uri!r}")
    return bucket, key


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/with/slashes`` into ``(bucket, key)``."""
    if not (uri.startswith("s3://") or uri.startswith("s3a://")):
        raise ValueError(f"Not an S3 URI: {uri!r}")
    stripped = uri.split("://", 1)[1]
    bucket, _, key = stripped.partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI missing bucket or key: {uri!r}")
    return bucket, key


def _update_checkpoint_pointer(
    fs: "_Fs",
    pointer_path: str,
    new_version: int,
    cp_name: str,
    *,
    max_attempts: int = 5,
) -> None:
    """Advance the ``_last_checkpoint.json`` pointer to ``new_version``
    only if the current pointer names an older version.

    The pointer is intentionally allowed to be stale (readers fall back
    to full replay or to an older checkpoint), but we still try to avoid
    regressions: if writer A publishes a pointer at v9 and writer B
    concurrently tries to publish a pointer at v4, B's update is
    dropped. This uses a read-modify-write loop; on object stores
    without CAS this is best-effort but the worst outcome is a single
    stale pointer, which readers handle gracefully.
    """
    payload = json.dumps({
        "version": new_version,
        "path": f"_flux_log/{cp_name}",
    }, indent=2).encode("utf-8")

    for _ in range(max_attempts):
        current = _read_json(fs, pointer_path)
        if (
            current is not None
            and isinstance(current, dict)
            and int(current.get("version", -1)) >= new_version
        ):
            return  # A concurrent writer already published a newer pointer.
        try:
            fs.write_bytes(pointer_path, payload)
            return
        except Exception:
            continue


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
