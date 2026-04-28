# Flux Spark V2 connector

A reference Spark V2 datasource that registers the `flux` short name and
delegates compression to the Rust core via the existing
`io.fluxcompress.FluxNative` JNI bridge.

## Capability matrix

This connector deliberately advertises every capability the V2 analyzer
needs to accept the common save modes:

| User call                              | V2 plan node            | Required capability / interface                             |
|----------------------------------------|-------------------------|-------------------------------------------------------------|
| `df.write.format("flux").save(p)`      | `CreateTableAsSelect`   | `BATCH_WRITE` on table                                      |
| `df.write.mode("append").save(p)`      | `AppendData`            | `BATCH_WRITE` on table                                      |
| `df.write.mode("overwrite").save(p)`   | `OverwriteByExpression` | `TRUNCATE` (table) **and** `SupportsTruncate` (builder)     |
| `option("replaceWhere", ...)`          | `OverwriteByExpression` | `OVERWRITE_BY_FILTER` (table) **and** `SupportsOverwriteV2` |
| `DELETE FROM t WHERE ...`              | `DeleteFromTable`       | `OVERWRITE_BY_FILTER` (table)                               |

Both **Option A** (`SupportsTruncate` + `SupportsOverwriteV2` on
`FluxWriteBuilder`) and **Option B** (`TRUNCATE` capability on
`FluxTable`) are present — Spark requires *both* in different code paths.

## Building the jar

The connector is plain Java; pair it with whatever build tool you already
use for the JNI bridge artifact. A minimal Maven layout:

```xml
<dependencies>
  <dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-sql_2.12</artifactId>
    <version>3.5.0</version>
    <scope>provided</scope>
  </dependency>
  <dependency>
    <groupId>org.apache.hadoop</groupId>
    <artifactId>hadoop-common</artifactId>
    <version>3.3.6</version>
    <scope>provided</scope>
  </dependency>
</dependencies>
```

Build:
```bash
javac -d build \
  -cp "$SPARK_HOME/jars/*" \
  $(find java -name '*.java')
jar cf flux-spark.jar -C build . -C java META-INF
```

Then submit alongside the native library:
```bash
spark-submit \
  --conf spark.driver.extraJavaOptions="-Djava.library.path=target/release" \
  --conf spark.executor.extraJavaOptions="-Djava.library.path=target/release" \
  --jars flux-spark.jar \
  your_job.py
```

## Unity Catalog remediation

Databricks Unity Catalog refuses arbitrary V2 datasources with the error
`The command(s): AppendData are not supported in Unity Catalog` because
UC's path-based write check only allow-lists Delta/Parquet/JSON/CSV/Text/
ORC/Avro. The supported way to plug a non-Delta source into UC is to
register it as a **catalog**, not as a path-based format.

The connector ships everything required for that:

* [`FluxTableProvider`](FluxTableProvider.java) implements
  `SupportsCatalogOptions`, so DataFrame ops hit a catalog rather than
  a raw path when one is configured.
* [`FluxCatalog`](FluxCatalog.java) implements `TableCatalog` +
  `SupportsNamespaces`, so SQL DDL like
  `CREATE TABLE flux.db.t (...) USING flux` works.

### Admin recipe (Option 3 — register flux as a UC catalog)

1. **Build the jar** as above (`flux-spark.jar`) and upload along with
   the native library (`libflux_jni.so`).

2. **Create a UC Volume** to back the catalog — this gives you a
   UC-governed location that the catalog can read/write without UC
   intercepting the writes:
   ```sql
   CREATE CATALOG IF NOT EXISTS flux_data;
   CREATE SCHEMA  IF NOT EXISTS flux_data.runtime;
   CREATE VOLUME  IF NOT EXISTS flux_data.runtime.tables;
   ```

3. **Configure a Single-User cluster** (Shared access mode rejects
   custom catalogs by design). On the cluster's *Spark Config*:
   ```
   spark.driver.extraJavaOptions    -Djava.library.path=/path/to/native
   spark.executor.extraJavaOptions  -Djava.library.path=/path/to/native

   spark.sql.catalog.flux           io.fluxcompress.spark.FluxCatalog
   spark.sql.catalog.flux.path      /Volumes/flux_data/runtime/tables
   ```
   On the *Libraries* tab install `flux-spark.jar`.

4. **Use it from any notebook attached to that cluster**:
   ```python
   spark.sql("USE CATALOG flux")
   spark.sql("""
     CREATE TABLE IF NOT EXISTS default.orders (
       order_id BIGINT,
       ts       BIGINT
     ) USING flux
   """)

   df.writeTo("flux.default.orders").append()         # AppendData OK
   df.writeTo("flux.default.orders").overwrite(F.col("ts") > 0)
   df.writeTo("flux.default.orders").overwritePartitions()  # SupportsTruncate
   ```

5. **Verify everything end-to-end** with the included smoke test:
   ```python
   %run /Workspace/.../python/tests/spark_uc_smoke_test.py
   ```
   This script exercises CREATE / AppendData / SupportsTruncate /
   SupportsOverwriteV2 / DROP and asserts each succeeds.

### Why this works where path-based writes don't

UC's V2 plan rule rejects custom data sources only on the
*path-based* write code path (`DataFrameWriter.format(...).save(path)`).
The *catalog-based* path (`writeTo("catalog.db.table")` and SQL DDL)
goes through `TableCatalog.createTable` / `loadTable`, which UC permits
for any registered catalog. By implementing `TableCatalog` we route the
entire write through the supported channel.

### Fallbacks (if you can't switch to Single-User)

* **Hybrid pattern** — compress to a `BinaryType` column with a UDF and
  store inside a Delta table. UC accepts Delta on every cluster type;
  see `examples/spark_example.py`.
* **Non-UC path** — write to `dbfs:/tmp/...` or a non-UC volume. The
  same connector's path-based code path (`format("flux").save(...)`)
  works there because the new `SupportsTruncate` and `TRUNCATE`
  capability are now in place.

## File layout

```
java/
├── io/fluxcompress/
│   ├── FluxNative.java                ← existing JNI primitives
│   └── spark/
│       ├── FluxTableProvider.java     ← `format("flux")` + SupportsCatalogOptions
│       ├── FluxCatalog.java           ← TableCatalog for UC catalog registration
│       ├── FluxTable.java             ← V2 Table  (TRUNCATE capability)
│       ├── FluxWriteBuilder.java      ← V2 WriteBuilder (SupportsTruncate)
│       ├── FluxWriteMode.java
│       ├── FluxWrite.java
│       ├── FluxBatchWrite.java        ← truncate / overwrite executed here
│       ├── FluxDataWriter*.java
│       ├── FluxWriterCommitMessage.java
│       └── test/
│           └── FluxConnectorTest.java ← JUnit 5 tests (no Spark required)
└── META-INF/services/
    └── org.apache.spark.sql.sources.DataSourceRegister

python/tests/
└── spark_uc_smoke_test.py             ← cluster-side end-to-end smoke test
```

## Tests

Two layers, mirroring the deployment story:

* **Unit tests — no Spark needed.**
  [`FluxConnectorTest.java`](test/FluxConnectorTest.java) covers the
  capability matrix, truncate-flag propagation, the catalog table
  lifecycle, and `FluxBatchWrite`'s commit-time truncate semantics. Run
  with JUnit 5 against any Spark 3.5+ classpath.

* **Integration smoke test — runs on a UC cluster.**
  [`python/tests/spark_uc_smoke_test.py`](../../../python/tests/spark_uc_smoke_test.py)
  exercises CREATE / AppendData / SupportsTruncate / SupportsOverwriteV2
  / DROP through the registered catalog. Use it as the
  acceptance gate after rolling out the Spark conf changes.
