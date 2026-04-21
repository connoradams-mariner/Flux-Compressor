# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Type stubs for the ``fluxcompress._fluxcompress`` native extension module.

These stubs allow IDEs (PyCharm, VS Code + Pylance) and type checkers
(mypy, pyright) to understand the Rust-compiled API without executing it.
"""

from __future__ import annotations
from typing import Optional

import pyarrow as pa

__version__: str

# ─────────────────────────────────────────────────────────────────────────────
# Predicate
# ─────────────────────────────────────────────────────────────────────────────

class Predicate:
    """
    A filter predicate for predicate pushdown during decompression.

    Built via :func:`col` and comparison operators::

        pred = col("user_id") > 1_000_000
        pred = (col("revenue") >= 0) & (col("revenue") <= 99_999)
    """

    def __and__(self, other: Predicate) -> Predicate: ...
    def __or__(self, other: Predicate) -> Predicate: ...
    def __repr__(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Column
# ─────────────────────────────────────────────────────────────────────────────

class Column:
    """
    Intermediate expression returned by :func:`col`.

    Supports comparison operators that return :class:`Predicate` objects::

        col("age") > 30
        col("revenue").between(0, 99_999)
    """

    def __gt__(self, value: int) -> Predicate: ...
    def __lt__(self, value: int) -> Predicate: ...
    def __eq__(self, value: int) -> Predicate: ...  # type: ignore[override]
    def __ge__(self, value: int) -> Predicate: ...
    def __le__(self, value: int) -> Predicate: ...
    def between(self, lo: int, hi: int) -> Predicate: ...
    def __repr__(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# BlockInfo
# ─────────────────────────────────────────────────────────────────────────────

class BlockInfo:
    """Metadata for a single compressed block from the Atlas footer."""

    offset: int
    """Byte offset of this block within the .flux buffer."""

    z_min: int
    """Minimum Z-Order coordinate (or column min value)."""

    z_max: int
    """Maximum Z-Order coordinate (or column max value)."""

    strategy: str
    """Compression strategy name (e.g., ``'BitSlab'``, ``'Rle'``)."""

    def __repr__(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# FileInfo
# ─────────────────────────────────────────────────────────────────────────────

class FileInfo:
    """Full inspection result returned by :func:`inspect`."""

    size_bytes: int
    """Total size of the .flux buffer in bytes."""

    num_blocks: int
    """Number of compressed blocks."""

    blocks: list[BlockInfo]
    """Per-block metadata list."""

    def __repr__(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# FluxBuffer
# ─────────────────────────────────────────────────────────────────────────────

class FluxBuffer:
    """
    A compressed FluxCompress buffer.

    Returned by :func:`compress`. Supports::

        buf = fc.compress(table)
        len(buf)             # size in bytes
        buf.to_bytes()       # raw bytes
        buf.decompress()     # → pyarrow.Table
        buf.inspect()        # → FileInfo
        buf.save("x.flux")
        FluxBuffer.load("x.flux")
    """

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

    def to_bytes(self) -> bytes:
        """Return the raw compressed bytes."""
        ...

    def decompress(
        self,
        predicate: Optional[Predicate] = None,
        column_name: str = "value",
    ) -> pa.Table:
        """Decompress to a PyArrow Table."""
        ...

    def inspect(self) -> FileInfo:
        """Inspect the Atlas metadata footer."""
        ...

    def save(self, path: str) -> None:
        """Write the compressed buffer to a ``.flux`` file."""
        ...

    @staticmethod
    def load(path: str) -> FluxBuffer:
        """Load a ``.flux`` file from disk."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Module-level functions
# ─────────────────────────────────────────────────────────────────────────────

def compress(
    table: pa.Table,
    strategy: str = "auto",
) -> FluxBuffer:
    """
    Compress a PyArrow Table (or any ``__arrow_c_stream__`` object) into a
    :class:`FluxBuffer`.

    Args:
        table:    A ``pyarrow.Table``, ``pyarrow.RecordBatch``, or a Polars
                  ``DataFrame`` (any object with ``__arrow_c_stream__``).
        strategy: One of ``"auto"``, ``"rle"``, ``"delta"``, ``"dict"``,
                  ``"bitslab"``, ``"lz4"``.  Default ``"auto"``.

    Returns:
        :class:`FluxBuffer`
    """
    ...


def decompress(
    buf: FluxBuffer | bytes,
    predicate: Optional[Predicate] = None,
    column_name: str = "value",
) -> pa.Table:
    """
    Decompress a :class:`FluxBuffer` or raw bytes into a PyArrow Table.

    Args:
        buf:         A :class:`FluxBuffer` or ``bytes`` object.
        predicate:   Optional pushdown predicate.  Blocks that cannot
                     satisfy the predicate are skipped entirely.
        column_name: Column name for the output table.

    Returns:
        ``pyarrow.Table``
    """
    ...


def inspect(buf: FluxBuffer | bytes) -> FileInfo:
    """
    Inspect the Atlas footer of a :class:`FluxBuffer`.

    Returns:
        :class:`FileInfo` with block-level metadata.
    """
    ...


def col(name: str) -> Column:
    """
    Return a :class:`Column` expression for building predicates.

    Example::

        pred = col("user_id") > 1_000_000
        pred = (col("revenue") >= 0) & (col("revenue") <= 99_999)
    """
    ...


def read_flux(path: str) -> FluxBuffer:
    """Load a ``.flux`` file from disk."""
    ...


def write_flux(buf: FluxBuffer, path: str) -> None:
    """Write a :class:`FluxBuffer` to a ``.flux`` file."""
    ...


# ─────────────────────────────────────────────────────────────────────────────
# Stateful writer with per-column Zstd dictionary cache
# ─────────────────────────────────────────────────────────────────────────────

class FluxWriter:
    """
    A stateful compressor with a per-column Zstd dictionary cache.

    On the **first** :meth:`compress` call per string column, a compact Zstd
    dictionary (8 KB) is trained from the column data.  Subsequent calls
    reuse the cached dictionary at Zstd level 5, achieving near-Archive
    compression ratio at 3–4× the write throughput.

    Ideal for repeated :meth:`FluxTable.append` calls on a stable table schema::

        writer = fc.FluxWriter(profile="archive")
        for batch in incoming_stream:
            tbl.append(writer.compress(batch))  # 2nd+ batches use cached dicts

    For one-shot compression, use the module-level :func:`compress` function
    which creates a fresh (uncached) writer each time.
    """

    def __new__(
        cls,
        profile: str = "archive",
        u64_only: bool = False,
    ) -> FluxWriter: ...

    def compress(self, table: pa.Table) -> FluxBuffer:
        """
        Compress a PyArrow Table or RecordBatch, using cached column
        dictionaries on the second and subsequent calls.
        """
        ...

    @property
    def dict_count(self) -> int:
        """Number of column dictionaries currently held in the cache."""
        ...

    def clear_cache(self) -> None:
        """Evict all cached dictionaries, forcing re-training on the next call."""
        ...

    def __repr__(self) -> str: ...


# ─────────────────────────────────────────────────────────────────────────────
# Phase F — Schema-evolution surface
# ─────────────────────────────────────────────────────────────────────────────

class SchemaField:
    """
    A single field in a :class:`TableSchema`.

    The ``field_id`` is the *immutable logical identifier* for the column —
    names can change and numeric types can be promoted, but ``field_id`` is
    stable for the table's lifetime.

    Example::

        f = fc.SchemaField(1, "user_id", "uint64", nullable=False)
        f = fc.SchemaField(2, "rev", "int64").with_int_default(0).with_doc("Cents")
    """

    field_id: int
    """Stable logical identifier for this column."""

    name: str
    """Current user-facing column name."""

    dtype: str
    """Logical dtype string (e.g. ``"uint64"``, ``"utf8"``)."""

    nullable: bool
    """Whether NULLs are permitted."""

    default_value: object
    """Literal default value, or ``None``."""

    doc: Optional[str]
    """Optional documentation string."""

    def __new__(
        cls,
        field_id: int,
        name: str,
        dtype: str,
        nullable: bool = True,
    ) -> SchemaField: ...

    def with_nullable(self, nullable: bool) -> SchemaField: ...
    def with_int_default(self, value: int) -> SchemaField: ...
    def with_uint_default(self, value: int) -> SchemaField: ...
    def with_float_default(self, value: float) -> SchemaField: ...
    def with_bool_default(self, value: bool) -> SchemaField: ...
    def with_str_default(self, value: str) -> SchemaField: ...
    def with_doc(self, doc: str) -> SchemaField: ...
    def __repr__(self) -> str: ...


class TableSchema:
    """
    The logical schema of a :class:`FluxTable` at a given schema version.

    Example::

        schema = fc.TableSchema([
            fc.SchemaField(1, "id",   "uint64", nullable=False),
            fc.SchemaField(2, "name", "utf8"),
        ])
    """

    schema_id: int
    """Monotonically increasing schema version identifier."""

    parent_schema_id: Optional[int]
    """Parent schema id (``None`` for the first schema)."""

    fields: list[SchemaField]
    """Fields in user-visible column order."""

    change_summary: Optional[str]
    """Human-readable change summary (advisory only)."""

    def __new__(cls, fields: list[SchemaField]) -> TableSchema: ...

    def field_by_id(self, field_id: int) -> Optional[SchemaField]:
        """Look up a field by stable field_id. Returns ``None`` if not found."""
        ...

    def field_by_name(self, name: str) -> Optional[SchemaField]:
        """Look up a field by display name. Returns ``None`` if not found."""
        ...

    def __repr__(self) -> str: ...


class EvolveOptions:
    """
    Options controlling schema-evolution behaviour.

    Example::

        # Default — no nullability tightening allowed
        opts = fc.EvolveOptions()

        # Opt into Phase D null tightening (requires manifest-derived proof)
        opts = fc.EvolveOptions(allow_null_tightening=True)
        opts = fc.EvolveOptions.with_null_tightening()  # shortcut
    """

    allow_null_tightening: bool
    """Whether nullable→non-nullable tightening is permitted."""

    def __new__(
        cls,
        allow_null_tightening: bool = False,
    ) -> EvolveOptions: ...

    @staticmethod
    def with_null_tightening() -> EvolveOptions:
        """Shortcut for ``EvolveOptions(allow_null_tightening=True)``."""
        ...

    def __repr__(self) -> str: ...


class FluxScan:
    """
    Streaming iterator over the live files of a :class:`FluxTable`.

    Returned by :meth:`FluxTable.scan`. Yields one ``pyarrow.Table`` per
    live file, projected to the current logical schema.

    Example::

        for batch in tbl.scan():
            print(batch.num_rows)
    """

    remaining: int
    """Number of live files not yet yielded."""

    def __iter__(self) -> FluxScan: ...
    def __next__(self) -> pa.Table: ...


class FluxTable:
    """
    A versioned columnar table backed by a ``.fluxtable/`` directory.

    Provides schema-evolution-aware append, scan, and time-travel operations
    via an immutable transaction log.

    Example::

        import fluxcompress as fc
        import pyarrow as pa

        tbl = fc.FluxTable("my_table.fluxtable")

        schema = fc.TableSchema([
            fc.SchemaField(1, "id",  "uint64", nullable=False),
            fc.SchemaField(2, "val", "int64"),
        ])
        tbl.evolve_schema(schema)

        buf = fc.compress(pa.table({"id": [1, 2], "val": [10, 20]}))
        tbl.append(buf)

        for batch in tbl.scan():
            print(batch)
    """

    def __new__(cls, path: str) -> FluxTable: ...

    @staticmethod
    def open(path: str) -> FluxTable:
        """Open (or create) a FluxTable — alias for the constructor."""
        ...

    def append(self, buf: FluxBuffer) -> int:
        """
        Append a :class:`FluxBuffer` to the table.

        Returns:
            ``int`` — the version number of the new log entry.
        """
        ...

    def evolve_schema(self, schema: TableSchema) -> int:
        """
        Evolve the table's logical schema (no data rewrite).

        Returns:
            ``int`` — the version number of the schema-change log entry.
        """
        ...

    def evolve_schema_with_options(
        self,
        schema: TableSchema,
        opts: EvolveOptions,
    ) -> int:
        """
        Evolve the schema with explicit :class:`EvolveOptions`.

        Returns:
            ``int`` — the version number of the schema-change log entry.
        """
        ...

    def scan(self) -> FluxScan:
        """
        Start a streaming scan over all live files projected to the current schema.

        Returns:
            :class:`FluxScan` iterator yielding one ``pyarrow.Table`` per file.
        """
        ...

    def live_files(self) -> list[str]:
        """Return absolute paths of all live data files at the latest version."""
        ...

    def current_version(self) -> Optional[int]:
        """Return the latest log version, or ``None`` if the log is empty."""
        ...

    def field_ids_for_current_schema(self) -> dict[str, int]:
        """
        Return a ``dict`` mapping column names to their stable ``field_id``
        values. Empty dict when no schema has been declared yet.
        """
        ...

    def current_schema(self) -> Optional[TableSchema]:
        """Return the current :class:`TableSchema`, or ``None`` if unset."""
        ...

    def __repr__(self) -> str: ...
