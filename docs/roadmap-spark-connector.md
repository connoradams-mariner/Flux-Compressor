# Roadmap: Spark DataSource V2 Connector (Phase H)

Status: Phase H initial cut implemented in `spark-connector/`.

## What shipped

- New `spark-connector/` sbt project (Scala 2.12, Spark 3.5 `provided`).
- SPI registration under
  `META-INF/services/org.apache.spark.sql.sources.DataSourceRegister`
  resolves `format("flux")` to `io.fluxcompress.spark.FluxDataSource`.
- `FluxSchemaConverter` — Spark `StructType` ↔ `TableSchema` JSON, using
  the existing `FluxDType` serde spellings (`"uint64"`, `"utf8"`,
  `"timestamp_micros"`, …).
- `FluxDataSource` + `FluxSparkTable` — `TableProvider`,
  `DataSourceRegister`, `SupportsRead`, `SupportsWrite`.
- Read path: `FluxScan` declares columnar reads; `FluxPartitionReader`
  calls `FluxNative.tableScan`, feeds the Arrow IPC stream through a
  shared allocator, yields one `ColumnarBatch` per IPC record-batch
  message.
- Write path: `FluxDataWriter` buffers `InternalRow`s into a
  `VectorSchemaRoot` via Spark's own `ArrowWriter`, flushes
  `batchSize`-row IPC batches through
  `FluxNative.compressTable` → `FluxNative.tableAppend`.
- Schema-evolution push-down: `option("evolve", "true")` calls
  `FluxNative.tableEvolve` once on the driver before any tasks start.
- `SparkArrowBridge` lives under `org.apache.spark.sql.fluxcompress`
  so it can call Spark's `private[sql]` `ArrowUtils` / `ArrowWriter`.
- ScalaTest suite that skips end-to-end DataFrame round-trip when
  `flux_jni` is not on `java.library.path`, but always runs the
  schema-converter unit tests.

## Open follow-ups

- **Per-file input partitioning.** Today `tableScan` returns the full
  dataset in a single Arrow IPC stream, so Phase H emits one
  `InputPartition`. A future `FluxNative.listLiveFiles(handle)` will
  let us emit one partition per `.flux` file and parallelise the scan
  across executors.
- **Nested types in `TableSchema` JSON.** The Arrow IPC round-trip
  already handles Struct/List/Map on both sides, but the
  evolution-push-down path emits only primitive `FluxDType` names
  because the on-disk schema doesn't yet spell containers inline.
- **True multi-task abort.** Per-task appends commit to the log as
  soon as `FluxDataWriter.commit()` runs, so `BatchWrite.abort()`
  cannot roll them back. A future two-phase protocol (stage per-task
  appends, commit atomically on the driver) will close this gap.
- **Filter pushdown.** `FluxScanBuilder` does not yet implement
  `SupportsPushDownFilters`; translating Spark's `Filter` expressions
  into `Predicate` variants for block-level skipping (and eventually
  row-level `eval_on_batch`) is the natural next step.
- **Spark 3.4 / 3.3 matrix.** The connector compiles against Spark
  3.5's DSv2 API surface. Back-porting to 3.4 requires accommodating
  the pre-`largeVarTypes` `ArrowUtils.toArrowSchema` signature.

## How to run

```bash
# 1. Build the Rust JNI cdylib.
cargo build --release -p flux-jni-bridge

# 2. Build + test the Scala connector.
cd spark-connector
sbt test                     # schema-converter tests + e2e (if cdylib present)
sbt package                  # produces the shaded JAR

# 3. Submit to Spark.
spark-submit \
  --jars spark-connector/target/scala-2.12/flux-spark-connector_2.12-0.1.0.jar \
  --conf "spark.driver.extraLibraryPath=$PWD/target/release" \
  --conf "spark.executor.extraLibraryPath=$PWD/target/release" \
  your_app.py
```

## Design notes

- **Why Scala 2.12?** Spark 3.5's published artefacts are
  cross-built for Scala 2.12 and 2.13; we target 2.12 by default
  because it's still the most common in enterprise Spark deployments.
- **Why split the bridge across two packages?** Spark's Arrow
  integration (`ArrowUtils`, `ArrowWriter`) is `private[sql]`, so
  `SparkArrowBridge` has to live under `org.apache.spark.sql.*`.
  Everything user-visible lives under `io.fluxcompress.spark.*` and
  only calls the bridge across the module boundary — same trick that
  Delta Lake and Iceberg use.
- **Why hand-rolled JSON in `FluxSchemaConverter`?** Spark already
  ships a jackson version on the classpath and conflicting transitive
  versions are a frequent cause of DataSource V2 classloader pain.
  The `TableSchema` JSON we emit is well-formed and tiny, so a
  purpose-built encoder keeps the dependency footprint minimal.
