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
