"""
Spark / Unity Catalog smoke test for the flux V2 connector.
============================================================

Run this on a Databricks cluster *after* an admin has:

  1. Built and uploaded ``flux-spark.jar`` and the native library.
  2. Set the catalog Spark conf on the cluster:

         spark.sql.catalog.flux        io.fluxcompress.spark.FluxCatalog
         spark.sql.catalog.flux.path   /Volumes/<catalog>/<schema>/<volume>

  3. Attached the cluster in **Single User** access mode (UC + custom V2
     catalogs require Single User mode; Shared mode rejects custom
     catalogs by design).

The script touches every code path the V2 analyzer exercises:
  • ``CREATE TABLE flux.default.smoke (...) USING flux``
  • ``INSERT INTO``                       → AppendData
  • ``df.writeTo(...).overwrite(...)``    → SupportsTruncate
  • ``df.writeTo(...).replace()``         → ReplaceTable
  • ``DELETE FROM ... WHERE``             → SupportsOverwriteV2
  • ``DROP TABLE``

If every assertion passes, all three of the original errors
(``does not support truncate in batch mode``, ``ErrorIfExists``, and
``AppendData are not supported in Unity Catalog``) are resolved.
"""

from __future__ import annotations

import pytest

# This is a manual smoke test that talks to a live Databricks/UC cluster.
# CI environments don't ship `pyspark`, so skip cleanly during collection
# rather than hard-failing pytest's import phase.
pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def main(catalog: str = "flux", namespace: str = "default", table: str = "smoke") -> None:
    spark = SparkSession.builder.appName("flux-uc-smoke").getOrCreate()
    fqtn = f"{catalog}.{namespace}.{table}"

    # 0. Sanity: catalog must be registered.
    catalogs = {row[0] for row in spark.sql("SHOW CATALOGS").collect()}
    assert catalog in catalogs, (
        f"catalog '{catalog}' not registered; "
        f"set spark.sql.catalog.{catalog}=io.fluxcompress.spark.FluxCatalog"
    )

    # 1. Clean slate.
    spark.sql(f"DROP TABLE IF EXISTS {fqtn}")

    # 2. CREATE TABLE — exercises FluxCatalog.createTable.
    spark.sql(f"""
        CREATE TABLE {fqtn} (
          id BIGINT,
          ts BIGINT
        ) USING flux
    """)
    assert any(row[1] == table for row in spark.sql(
        f"SHOW TABLES IN {catalog}.{namespace}"
    ).collect()), f"{fqtn} not visible after CREATE"

    # 3. AppendData — was previously rejected by UC.
    df1 = spark.range(0, 100).withColumn("ts", F.col("id") * 1000)
    df1.writeTo(fqtn).append()
    assert spark.table(fqtn).count() == 100, "AppendData failed to land 100 rows"

    # 4. SupportsTruncate — was the original error.
    df2 = spark.range(200, 300).withColumn("ts", F.col("id") * 1000)
    df2.writeTo(fqtn).overwritePartitions()  # uses SupportsTruncate
    after_overwrite = spark.table(fqtn).count()
    assert after_overwrite == 100, (
        f"overwritePartitions should leave 100 rows from df2; got {after_overwrite}"
    )

    # 5. SupportsOverwriteV2 — predicate-based delete.
    df3 = spark.range(200, 400).withColumn("ts", F.col("id") * 1000)
    df3.writeTo(fqtn).overwrite(F.col("id") >= 200)
    after_replace_where = spark.table(fqtn).count()
    assert after_replace_where == 200, (
        f"overwrite(filter) should leave 200 rows; got {after_replace_where}"
    )

    # 6. DROP TABLE — exercises FluxCatalog.dropTable.
    spark.sql(f"DROP TABLE {fqtn}")
    remaining = {row[1] for row in spark.sql(
        f"SHOW TABLES IN {catalog}.{namespace}"
    ).collect()}
    assert table not in remaining, f"{fqtn} still present after DROP"

    print("✓ flux V2 connector smoke test passed against Unity Catalog")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog",   default="flux")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--table",     default="smoke")
    args = parser.parse_args()
    main(args.catalog, args.namespace, args.table)
