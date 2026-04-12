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

from fluxcompress._fluxcompress import FluxBuffer, Predicate, compress, decompress


def compress_polars(df: "pl.DataFrame", strategy: str = "auto") -> FluxBuffer:
    """
    Compress a Polars DataFrame into a :class:`FluxBuffer`.

    The DataFrame is converted to a PyArrow Table via the Arrow C Data
    Interface — a zero-copy operation for most numeric column types.

    Parameters
    ----------
    df:
        A ``polars.DataFrame``.
    strategy:
        Compression strategy override.  One of ``"auto"``, ``"rle"``,
        ``"delta"``, ``"dict"``, ``"bitslab"``, ``"lz4"``.
        ``"auto"`` (default) lets the Loom classifier decide per segment.

    Returns
    -------
    FluxBuffer
        The compressed bytes with Atlas footer.

    Example
    -------
    >>> import polars as pl
    >>> import fluxcompress as fc
    >>> df = pl.DataFrame({"user_id": range(1_000_000), "revenue": range(1_000_000)})
    >>> buf = fc.compress_polars(df)
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

    # to_arrow() uses the Arrow C Data Interface — zero-copy for numeric types.
    arrow_table = df.to_arrow()
    return compress(arrow_table, strategy=strategy)


def decompress_polars(
    buf: FluxBuffer | bytes,
    predicate: Predicate | None = None,
    column_name: str = "value",
) -> "pl.DataFrame":
    """
    Decompress a :class:`FluxBuffer` into a Polars DataFrame.

    Parameters
    ----------
    buf:
        A :class:`FluxBuffer` or raw ``bytes`` from a ``.flux`` file.
    predicate:
        Optional predicate for pushdown filtering.  Blocks whose
        ``[z_min, z_max]`` range cannot satisfy the predicate are skipped.
    column_name:
        Column name for the decompressed output.

    Returns
    -------
    polars.DataFrame

    Example
    -------
    >>> import fluxcompress as fc
    >>> buf = fc.read_flux("data.flux")
    >>> df = fc.decompress_polars(buf, predicate=fc.col("user_id") > 500_000)
    """
    try:
        import polars as pl
    except ImportError:
        raise ImportError(
            "polars is required for decompress_polars(). "
            "Install it with: pip install fluxcompress[polars]"
        ) from None

    arrow_table = decompress(buf, predicate=predicate, column_name=column_name)
    # from_arrow() is also zero-copy for numeric types.
    return pl.from_arrow(arrow_table)
