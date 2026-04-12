# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
fluxcompress.pandas
===================

Pandas DataFrame helpers for FluxCompress.

The integration uses PyArrow as the interchange layer — pandas ships with
PyArrow support via ``pandas.api.types`` and ``pd.ArrowDtype`` (pandas 1.5+).
For older pandas, values are copied through NumPy.

Usage
-----
>>> import pandas as pd
>>> import fluxcompress.pandas as fcp
>>>
>>> df = pd.DataFrame({"user_id": range(1_000_000), "revenue": range(1_000_000)})
>>> buf = fcp.compress_df(df)
>>> df2 = fcp.decompress_df(buf)
>>> assert df.equals(df2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import pyarrow as pa

from fluxcompress._fluxcompress import (
    FluxBuffer,
    Predicate,
    compress,
    decompress,
)

if TYPE_CHECKING:
    import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# compress_df
# ─────────────────────────────────────────────────────────────────────────────

def compress_df(
    df: "pd.DataFrame",
    columns: Sequence[str] | None = None,
    strategy: str = "auto",
) -> FluxBuffer:
    """
    Compress a pandas DataFrame into a :class:`~fluxcompress.FluxBuffer`.

    Only numeric columns are compressed by default.  Pass ``columns`` to
    select a specific subset.

    Parameters
    ----------
    df:
        A ``pandas.DataFrame``.
    columns:
        Column names to include.  ``None`` (default) selects all numeric
        columns (int, float, unsigned int).
    strategy:
        Compression strategy override: ``"auto"``, ``"rle"``, ``"delta"``,
        ``"dict"``, ``"bitslab"``, ``"lz4"``.

    Returns
    -------
    FluxBuffer
        Compressed bytes with Atlas footer.

    Raises
    ------
    ImportError
        If pandas is not installed.
    ValueError
        If no numeric columns are found and ``columns`` is ``None``.

    Example
    -------
    >>> import pandas as pd
    >>> import fluxcompress.pandas as fcp
    >>> df = pd.DataFrame({"id": range(100_000), "val": range(100_000)})
    >>> buf = fcp.compress_df(df)
    >>> print(buf)
    FluxBuffer(...)
    """
    pd = _require_pandas()

    if columns is not None:
        df = df[list(columns)]
    else:
        # Auto-select numeric columns.
        df = df.select_dtypes(include=["integer", "floating", "unsigned integer"])
        if df.empty:
            raise ValueError(
                "No numeric columns found in DataFrame.  "
                "Pass columns= to select specific columns explicitly."
            )

    # Convert to PyArrow via the most efficient path available.
    arrow_table = _df_to_arrow(df)
    return compress(arrow_table, strategy=strategy)


# ─────────────────────────────────────────────────────────────────────────────
# decompress_df
# ─────────────────────────────────────────────────────────────────────────────

def decompress_df(
    buf: FluxBuffer | bytes,
    predicate: Predicate | None = None,
    column_name: str = "value",
) -> "pd.DataFrame":
    """
    Decompress a :class:`~fluxcompress.FluxBuffer` into a pandas DataFrame.

    Parameters
    ----------
    buf:
        A :class:`~fluxcompress.FluxBuffer` or raw ``bytes`` from a
        ``.flux`` file.
    predicate:
        Optional pushdown predicate.  Blocks whose ``[z_min, z_max]`` range
        cannot satisfy the predicate are skipped without decompression.
    column_name:
        Column name for the decompressed output column.

    Returns
    -------
    pandas.DataFrame

    Example
    -------
    >>> import fluxcompress as fc
    >>> import fluxcompress.pandas as fcp
    >>> buf = fc.read_flux("data.flux")
    >>> df = fcp.decompress_df(buf, predicate=fc.col("value") > 50_000)
    """
    _require_pandas()

    arrow_table = decompress(buf, predicate=predicate, column_name=column_name)
    return _arrow_to_df(arrow_table)


# ─────────────────────────────────────────────────────────────────────────────
# compress_series  /  decompress_series
# ─────────────────────────────────────────────────────────────────────────────

def compress_series(
    series: "pd.Series",
    strategy: str = "auto",
) -> FluxBuffer:
    """
    Compress a single pandas Series into a :class:`~fluxcompress.FluxBuffer`.

    Parameters
    ----------
    series:
        A ``pandas.Series`` of a numeric dtype.
    strategy:
        Compression strategy override.

    Returns
    -------
    FluxBuffer

    Example
    -------
    >>> import pandas as pd
    >>> import fluxcompress.pandas as fcp
    >>> s = pd.Series(range(1_000_000), name="user_id")
    >>> buf = fcp.compress_series(s)
    """
    _require_pandas()

    name = series.name or "value"
    arr = pa.array(series.to_numpy(), type=_pandas_dtype_to_arrow(series.dtype))
    table = pa.table({str(name): arr})
    return compress(table, strategy=strategy)


def decompress_series(
    buf: FluxBuffer | bytes,
    predicate: Predicate | None = None,
    name: str = "value",
) -> "pd.Series":
    """
    Decompress a :class:`~fluxcompress.FluxBuffer` into a pandas Series.

    Parameters
    ----------
    buf:
        A :class:`~fluxcompress.FluxBuffer` or raw ``bytes``.
    predicate:
        Optional pushdown predicate.
    name:
        Name for the resulting Series.

    Returns
    -------
    pandas.Series
    """
    _require_pandas()

    arrow_table = decompress(buf, predicate=predicate, column_name=name)
    col = arrow_table.column(0)
    return col.to_pandas().rename(name)


# ─────────────────────────────────────────────────────────────────────────────
# compress_column  (in-place: add a _flux column to a DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def compress_column(
    df: "pd.DataFrame",
    column: str,
    output_column: str | None = None,
    strategy: str = "auto",
    chunk_size: int = 65_536,
) -> "pd.DataFrame":
    """
    Add a compressed ``.flux`` representation of ``column`` as a new bytes
    column in the DataFrame.

    The column is split into ``chunk_size``-row chunks and each chunk is
    compressed independently (the Hot Stream Model), mirroring how Spark
    partitions work.  The compressed blobs are stored as Python ``bytes``
    objects in the new column — one blob per chunk, ``None`` for padding rows.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Name of the numeric column to compress.
    output_column:
        Name for the new compressed column.  Defaults to ``f"{column}_flux"``.
    strategy:
        Compression strategy.
    chunk_size:
        Rows per compressed chunk (default 65 536 ≈ 64 KB of u64).

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with an additional object column of ``bytes``.

    Example
    -------
    >>> df = compress_column(df, "user_id")
    >>> df["user_id_flux"]  # Series of bytes objects (one per chunk)
    """
    import pandas as pd
    _require_pandas()

    out_col = output_column or f"{column}_flux"
    series = df[column]
    n = len(series)
    blobs: list[bytes | None] = [None] * n

    for start in range(0, n, chunk_size):
        chunk = series.iloc[start : start + chunk_size]
        buf = compress_series(chunk, strategy=strategy)
        # Store the blob at the first row of the chunk; rest stay None.
        blobs[start] = bytes(buf.to_bytes())

    result = df.copy()
    result[out_col] = blobs
    return result


# ─────────────────────────────────────────────────────────────────────────────
# round_trip  (convenience for testing)
# ─────────────────────────────────────────────────────────────────────────────

def round_trip(
    df: "pd.DataFrame",
    columns: Sequence[str] | None = None,
    strategy: str = "auto",
) -> "pd.DataFrame":
    """
    Compress and immediately decompress a DataFrame — useful for testing
    and validating round-trip correctness.

    Parameters
    ----------
    df:
        Input DataFrame (numeric columns only).
    columns:
        Columns to include.  ``None`` = all numeric columns.
    strategy:
        Compression strategy.

    Returns
    -------
    pandas.DataFrame
        Decompressed copy.  Column names are preserved.

    Example
    -------
    >>> df2 = fcp.round_trip(df)
    >>> assert df.equals(df2)
    """
    buf = compress_df(df, columns=columns, strategy=strategy)
    return decompress_df(buf)


# ─────────────────────────────────────────────────────────────────────────────
# compression_stats  (ratio report)
# ─────────────────────────────────────────────────────────────────────────────

def compression_stats(
    df: "pd.DataFrame",
    columns: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """
    Return a DataFrame summarising compression statistics for each column.

    Columns in the output:

    - ``column``     — column name
    - ``dtype``      — pandas dtype
    - ``rows``       — number of rows
    - ``raw_bytes``  — uncompressed size (dtype width × rows)
    - ``flux_bytes`` — compressed size
    - ``ratio``      — raw_bytes / flux_bytes
    - ``strategy``   — Loom strategy chosen for the first segment

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to analyse.  ``None`` = all numeric columns.

    Returns
    -------
    pandas.DataFrame

    Example
    -------
    >>> print(fcp.compression_stats(df))
        column  dtype    rows  raw_bytes  flux_bytes  ratio strategy
    0  user_id  int64  100000     800000       12345  64.8x  DeltaDelta
    1  revenue  int64  100000     800000       45678  17.5x    BitSlab
    """
    import pandas as pd
    import fluxcompress as fc
    _require_pandas()

    if columns is None:
        cols = df.select_dtypes(include=["integer", "floating"]).columns.tolist()
    else:
        cols = list(columns)

    records = []
    for col_name in cols:
        series = df[col_name]
        buf = compress_series(series)
        info = fc.inspect(buf)
        strategy = info.blocks[0].strategy if info.blocks else "unknown"

        itemsize = series.dtype.itemsize if hasattr(series.dtype, "itemsize") else 8
        raw_bytes = len(series) * itemsize
        flux_bytes = len(buf)

        records.append({
            "column":     col_name,
            "dtype":      str(series.dtype),
            "rows":       len(series),
            "raw_bytes":  raw_bytes,
            "flux_bytes": flux_bytes,
            "ratio":      f"{raw_bytes / flux_bytes:.1f}x",
            "strategy":   strategy,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_pandas():
    try:
        import pandas  # noqa: F401
        return pandas
    except ImportError:
        raise ImportError(
            "pandas is required for fluxcompress.pandas helpers.  "
            "Install it with: pip install fluxcompress[pandas]"
        ) from None


def _df_to_arrow(df: "pd.DataFrame") -> pa.Table:
    """Convert a pandas DataFrame to a PyArrow Table via the most efficient path."""
    try:
        # Fastest path: pyarrow native conversion (zero-copy for most dtypes).
        return pa.Table.from_pandas(df, preserve_index=False)
    except Exception:
        # Fallback: column-by-column via numpy.
        arrays = []
        fields = []
        for col in df.columns:
            arr = pa.array(df[col].to_numpy())
            arrays.append(arr)
            fields.append(pa.field(str(col), arr.type))
        return pa.table(dict(zip(df.columns, arrays)))


def _arrow_to_df(table: pa.Table) -> "pd.DataFrame":
    """Convert a PyArrow Table to a pandas DataFrame."""
    return table.to_pandas()


def _pandas_dtype_to_arrow(dtype) -> pa.DataType:
    """Map a pandas dtype to the best Arrow type."""
    import numpy as np
    dtype = np.dtype(dtype)
    mapping = {
        np.int8:    pa.int8(),
        np.int16:   pa.int16(),
        np.int32:   pa.int32(),
        np.int64:   pa.int64(),
        np.uint8:   pa.uint8(),
        np.uint16:  pa.uint16(),
        np.uint32:  pa.uint32(),
        np.uint64:  pa.uint64(),
        np.float32: pa.float32(),
        np.float64: pa.float64(),
    }
    return mapping.get(dtype.type, pa.int64())


# ─────────────────────────────────────────────────────────────────────────────
# __all__
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "compress_df",
    "decompress_df",
    "compress_series",
    "decompress_series",
    "compress_column",
    "round_trip",
    "compression_stats",
]
