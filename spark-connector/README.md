# FluxCompress — Spark DataSource V2 connector

A Scala/sbt connector that lets Spark 3.5 read and write `.fluxtable`
datasets through the standard DataSource V2 `format("flux")` entry
point:

```scala
df.write.format("flux").option("evolve", "true").save("/data/events.fluxtable")
val out = spark.read.format("flux").load("/data/events.fluxtable")
```

## Build

```bash
# 1. Build the Rust JNI cdylib.
cargo build --release -p flux-jni-bridge

# 2. Build + test the Scala connector.
cd spark-connector
sbt test     # e2e tests auto-skip if the cdylib is missing
sbt package  # produces a shaded JAR under target/scala-2.12/
```

## Usage from `spark-submit`

```bash
spark-submit \
  --jars spark-connector/target/scala-2.12/flux-spark-connector_2.12-0.1.0.jar \
  --conf "spark.driver.extraLibraryPath=$PWD/target/release" \
  --conf "spark.executor.extraLibraryPath=$PWD/target/release" \
  your_app.py
```

## Options

| Key         | Default  | Description                                           |
|-------------|----------|-------------------------------------------------------|
| `path`      | —        | Required.  Directory that backs the `.fluxtable/`.    |
| `profile`   | `speed`  | `speed` \| `balanced` \| `archive` \| `brotli`.       |
| `batchSize` | `10000`  | Rows per IPC flush on write.                          |
| `evolve`    | `false`  | Push the Spark schema into the Flux table before writing. |

## Scope

Phase H delivers a correct DataSource V2 end-to-end path on top of the
existing JNI surface:

- Single input partition per read (streaming per-file partitioning is
  deferred — it needs a `listLiveFiles` JNI call).
- Per-task independent `tableAppend`; abort cannot roll back already-
  committed per-task appends.
- Primitive types only: `byte/short/int/long`, `float/double`,
  `boolean`, `date`, `timestamp`, `decimal`, `string`, `binary`.
  Complex types round-trip through Arrow IPC but cannot be declared
  in `TableSchema` JSON yet.
