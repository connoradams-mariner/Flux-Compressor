# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
fluxcompress
============

High-performance adaptive columnar compression for Python.

Works natively with PyArrow, Polars, and Apache Spark DataFrames.

Quick start
-----------
>>> import pyarrow as pa
>>> import fluxcompress as fc

>>> # Compress any Arrow-compatible table
>>> table = pa.table({"id": pa.array(range(1_000_000)), "val": pa.array(range(1_000_000))})
>>> buf = fc.compress(table)
>>> print(buf)
FluxBuffer(312451 bytes)

>>> # Decompress back to a PyArrow table
>>> table2 = fc.decompress(buf)

>>> # Predicate pushdown — only decompress relevant blocks
>>> table3 = fc.decompress(buf, predicate=fc.col("id") > 500_000)

>>> # Save / load .flux files
>>> fc.write_flux(buf, "data.flux")
>>> buf2 = fc.read_flux("data.flux")

>>> # Inspect Atlas footer metadata
>>> info = fc.inspect(buf)
>>> print(info.num_blocks, "blocks")

Polars
------
>>> import polars as pl
>>> df = pl.DataFrame({"a": range(1_000_000)})
>>> buf = fc.compress(df)       # Polars DataFrame accepted directly
>>> df2 = fc.decompress_polars(buf)

Apache Spark
------------
See ``fluxcompress.spark`` for Spark UDF helpers.
"""

from __future__ import annotations

# Import the compiled Rust extension.
# On a development install (maturin develop) this is built in-place.
# On a wheel install this is the pre-compiled .so / .pyd.
from fluxcompress._fluxcompress import (  # noqa: F401
    # Classes
    Predicate,
    Column,
    BlockInfo,
    FileInfo,
    FluxBuffer,
    FluxBatchReader,
    # Core functions
    compress,
    decompress,
    decompress_file,
    inspect,
    col,
    read_flux,
    read_flux_schema,
    write_flux,
    merge_flux_buffers,
    # Version
    __version__,
)

from fluxcompress._polars import (  # noqa: F401
    compress_polars,
    decompress_polars,
    scan_flux,
    write_flux_table,
    optimize,
)

# pandas helpers are imported lazily in fluxcompress.pandas to avoid a hard
# dependency.  Re-export the submodule reference so users can do:
#   import fluxcompress.pandas as fcp
# without a separate install step failing at import time.
from fluxcompress import pandas as pandas  # noqa: F401 (re-export submodule)

__all__ = [
    # Classes
    "Predicate",
    "Column",
    "BlockInfo",
    "FileInfo",
    "FluxBuffer",
    # Core functions
    "compress",
    "decompress",
    "inspect",
    "col",
    "read_flux",
    "write_flux",
    "merge_flux_buffers",
    # Polars helpers
    "compress_polars",
    "decompress_polars",
    "scan_flux",
    "write_flux_table",
    "optimize",
    # Batch reader
    "FluxBatchReader",
    # Submodules
    "pandas",
    # Version
    "__version__",
]
