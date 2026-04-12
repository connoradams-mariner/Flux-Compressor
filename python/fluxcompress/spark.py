# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
fluxcompress.spark
==================

PySpark helper functions and UDFs for integrating FluxCompress into
Spark pipelines.

Usage
-----
>>> from pyspark.sql import SparkSession
>>> import fluxcompress.spark as fcs
>>>
>>> spark = SparkSession.builder.getOrCreate()
>>> fcs.register_udfs(spark)
>>>
>>> df = spark.range(10_000_000)
>>> df_compressed = df.withColumn("id_flux", fcs.flux_compress_col("id"))
>>> df_compressed.write.parquet("s3://bucket/compressed/")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import pyarrow as pa
import pyarrow.ipc as pa_ipc

from fluxcompress._fluxcompress import (
    FluxBuffer,
    compress,
    decompress,
    Predicate,
)

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession
    from pyspark.sql.types import StructType


# ─────────────────────────────────────────────────────────────────────────────
# Pandas UDFs (Arrow-based, low serialisation overhead)
# ─────────────────────────────────────────────────────────────────────────────

def _make_compress_udf():
    """Build the compress pandas UDF (lazy import to avoid hard Spark dep)."""
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BinaryType
    import pandas as pd

    @pandas_udf(BinaryType())
    def _compress_series(series: pd.Series) -> pd.Series:
        """
        Compress each partition's column into a single .flux blob.

        Because this is a scalar UDF, Spark calls it once per batch
        (Arrow RecordBatch from the executor).  We receive a pandas Series
        of values, convert to an Arrow array, compress, and return a
        single bytes value.
        """
        arr = pa.array(series.values)
        table = pa.table({"value": arr})
        buf: FluxBuffer = compress(table)
        return pd.Series([buf.to_bytes()])

    return _compress_series


def _make_decompress_udf():
    """Build the decompress pandas UDF."""
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BinaryType
    import pandas as pd

    @pandas_udf("array<long>")
    def _decompress_series(series: pd.Series) -> pd.Series:
        results = []
        for flux_bytes in series:
            table = decompress(bytes(flux_bytes))
            col = table.column(0).to_pylist()
            results.append(col)
        return pd.Series(results)

    return _decompress_series


# ─────────────────────────────────────────────────────────────────────────────
# Partition-level compression (foreachPartition / mapPartitions)
# ─────────────────────────────────────────────────────────────────────────────

def compress_partition(
    rows: Iterator,
    column_index: int = 0,
) -> Iterator[tuple[int, bytes]]:
    """
    Map function for ``rdd.mapPartitionsWithIndex``.

    Collects all values in the partition, compresses them as a single
    FluxCompress block, and yields ``(partition_id, flux_bytes)``.

    Example
    -------
    >>> from functools import partial
    >>> compressed_rdd = (
    ...     df.rdd
    ...     .mapPartitionsWithIndex(partial(compress_partition, column_index=0))
    ... )
    """
    from itertools import chain

    rows_list = list(rows)
    if not rows_list:
        return

    # Build an Arrow table from the partition rows.
    values = [row[column_index] for row in rows_list]
    arr = pa.array(values, type=pa.int64())
    table = pa.table({"value": arr})

    buf: FluxBuffer = compress(table)
    flux_bytes: bytes = bytes(buf.to_bytes())

    # We don't have partition_id in mapPartitions; caller uses
    # mapPartitionsWithIndex for that.
    yield flux_bytes


def compress_partition_indexed(
    partition_id: int,
    rows: Iterator,
    column_index: int = 0,
) -> Iterator[tuple[int, bytes]]:
    """
    Map function for ``rdd.mapPartitionsWithIndex``.

    Yields ``(partition_id, flux_bytes)`` tuples.
    """
    rows_list = list(rows)
    if not rows_list:
        return

    values = [row[column_index] for row in rows_list]
    arr = pa.array(values, type=pa.int64())
    table = pa.table({"value": arr})

    buf: FluxBuffer = compress(table)
    yield (partition_id, bytes(buf.to_bytes()))


# ─────────────────────────────────────────────────────────────────────────────
# High-level DataFrame API
# ─────────────────────────────────────────────────────────────────────────────

def compress_dataframe(
    df: "DataFrame",
    column: str,
    output_column: str | None = None,
    strategy: str = "auto",
) -> "DataFrame":
    """
    Compress a Spark DataFrame column into a BinaryType ``.flux`` blob column.

    Each Spark partition is compressed independently (the Hot Stream Model)
    so that parallel executors can write without global coordination.

    Parameters
    ----------
    df:
        Input Spark DataFrame.
    column:
        Name of the numeric column to compress.
    output_column:
        Name for the new compressed column.  Defaults to ``f"{column}_flux"``.
    strategy:
        Compression strategy override (``"auto"``, ``"rle"``, etc.).

    Returns
    -------
    DataFrame
        Original DataFrame with an additional ``BinaryType`` column.

    Example
    -------
    >>> df_out = fcs.compress_dataframe(df, column="user_id")
    >>> df_out.printSchema()
    root
     |-- user_id: long (nullable = false)
     |-- user_id_flux: binary (nullable = true)
    """
    from pyspark.sql import functions as F
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import BinaryType
    import pandas as pd

    out_col = output_column or f"{column}_flux"

    # Build a strategy-aware UDF closure.
    _strategy = strategy  # capture for closure

    @pandas_udf(BinaryType())
    def _compress(series: pd.Series) -> pd.Series:
        arr = pa.array(series.values)
        table = pa.table({"value": arr})
        buf = compress(table, strategy=_strategy)
        return pd.Series([bytes(buf.to_bytes())] * len(series))

    return df.withColumn(out_col, _compress(F.col(column)))


def decompress_dataframe(
    df: "DataFrame",
    flux_column: str,
    output_schema: "StructType | None" = None,
) -> "DataFrame":
    """
    Decompress a BinaryType ``.flux`` column back into its original values.

    Parameters
    ----------
    df:
        DataFrame containing a ``BinaryType`` column of ``.flux`` blobs.
    flux_column:
        Name of the column containing compressed ``.flux`` bytes.
    output_schema:
        Optional Spark schema for the decompressed values.
        Defaults to ``ArrayType(LongType())``.

    Returns
    -------
    DataFrame
        DataFrame with a new column ``{flux_column}_decoded`` of type
        ``ArrayType(LongType())``.

    Example
    -------
    >>> df_decoded = fcs.decompress_dataframe(df_compressed, "user_id_flux")
    """
    from pyspark.sql import functions as F
    from pyspark.sql.functions import pandas_udf
    import pandas as pd

    out_col = f"{flux_column}_decoded"

    @pandas_udf("array<long>")
    def _decompress(series: pd.Series) -> pd.Series:
        results = []
        for flux_bytes in series:
            if flux_bytes is None:
                results.append(None)
                continue
            table = decompress(bytes(flux_bytes))
            results.append(table.column(0).to_pylist())
        return pd.Series(results)

    return df.withColumn(out_col, _decompress(F.col(flux_column)))


# ─────────────────────────────────────────────────────────────────────────────
# UDF registration (SQL interface)
# ─────────────────────────────────────────────────────────────────────────────

def register_udfs(spark: "SparkSession") -> None:
    """
    Register FluxCompress UDFs so they can be used in Spark SQL.

    After calling this, you can use::

        SELECT flux_compress(user_id) AS user_id_flux
        FROM my_table

    Registered UDFs
    ---------------
    ``flux_compress(col: LongType) → BinaryType``
        Compress a long column to .flux bytes.
    ``flux_decompress(col: BinaryType) → ArrayType(LongType)``
        Decompress .flux bytes to an array of longs.

    Parameters
    ----------
    spark:
        The active ``SparkSession``.

    Example
    -------
    >>> fcs.register_udfs(spark)
    >>> spark.sql("SELECT flux_compress(id) FROM range(1000000)").show()
    """
    from pyspark.sql.types import BinaryType
    from pyspark.sql.functions import pandas_udf
    import pandas as pd

    @pandas_udf(BinaryType())
    def flux_compress(series: pd.Series) -> pd.Series:
        arr = pa.array(series.values)
        table = pa.table({"value": arr})
        buf = compress(table)
        return pd.Series([bytes(buf.to_bytes())] * len(series))

    @pandas_udf("array<long>")
    def flux_decompress(series: pd.Series) -> pd.Series:
        results = []
        for flux_bytes in series:
            if flux_bytes is None:
                results.append(None)
                continue
            table = decompress(bytes(flux_bytes))
            results.append(table.column(0).to_pylist())
        return pd.Series(results)

    spark.udf.register("flux_compress",   flux_compress)
    spark.udf.register("flux_decompress", flux_decompress)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience column references (mirrors fc.col() for Spark)
# ─────────────────────────────────────────────────────────────────────────────

def flux_compress_col(column_name: str) -> "Column":
    """
    Return a Spark ``Column`` expression that compresses the named column.

    Example
    -------
    >>> df.withColumn("id_flux", fcs.flux_compress_col("id"))
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import BinaryType
    from pyspark.sql.functions import pandas_udf
    import pandas as pd

    @pandas_udf(BinaryType())
    def _udf(series: pd.Series) -> pd.Series:
        arr = pa.array(series.values)
        table = pa.table({"value": arr})
        buf = compress(table)
        return pd.Series([bytes(buf.to_bytes())] * len(series))

    return _udf(F.col(column_name))
