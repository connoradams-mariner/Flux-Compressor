# Using FluxCompress on Databricks

FluxCompress ships a Spark DataSource V2 connector that registers the
`flux` short name, so on a Databricks cluster you can write:

```python
(sdf.write
    .format("flux")
    .option("evolve", "true")
    .save("/Volumes/catalog/schema/events.fluxtable"))

df = spark.read.format("flux").load("/Volumes/catalog/schema/events.fluxtable")
```

Once the v0.5.0 release pipeline completes (see
[`.github/workflows/release-maven.yml`](../.github/workflows/release-maven.yml))
the artefact lives at:

```
com.datamariners.fluxcompress:flux-spark-connector_2.12:0.5.0
```

The published JAR **bundles the `flux_jni` native library for Linux
x86_64, Linux aarch64, macOS x64, macOS arm64, and Windows x64**.  The
first JNI call on the driver (and each executor) extracts the right
variant from the JAR into a temp file and `System.load`s it — no DBFS
upload, no `--conf spark.driver.extraLibraryPath`, no manual
configuration.  See
[`FluxNativeLoader.scala`](../spark-connector/src/main/scala/com/datamariners/fluxcompress/FluxNativeLoader.scala).

## Install on a Databricks cluster

### Option A — Cluster-scoped Maven library (recommended)

1. Open your cluster → **Libraries** → **Install new** → **Maven**.
2. Paste the coordinate:
   ```
   com.datamariners.fluxcompress:flux-spark-connector_2.12:0.5.0
   ```
   (use `_2.13` if you're on a Databricks runtime that ships Scala 2.13,
   such as DBR 14.x with Photon on Scala 2.13).
3. Click **Install**.  The library attaches automatically on the next
   cluster restart; Databricks propagates it to every executor.

### Option B — Notebook-scoped via `%pip` + Maven sidecar

If you'd rather install per-notebook (for experimentation):

```python
%pip install fluxcompress>=0.5.0
dbutils.library.restartPython()
```

Then attach the Maven coordinate via an init script, or run
`spark.sql("ADD JAR ...")` pointing at a DBFS-cached copy of the
connector JAR.

### Option C — Shared cluster init script (fleet deployment)

For large-scale deployments where individual users shouldn't manage
libraries, drop this into a cluster init script:

```bash
#!/usr/bin/env bash
set -euo pipefail
JAR=/databricks/jars/flux-spark-connector_2.12-0.5.0.jar
if [ ! -f "$JAR" ]; then
  curl -L -o "$JAR" \
    https://repo1.maven.org/maven2/com/datamariners/fluxcompress/flux-spark-connector_2.12/0.5.0/flux-spark-connector_2.12-0.5.0.jar
fi
```

Databricks will add `$JAR` to the driver + executor classpath for
every job on the cluster.

## Usage

### Write

```python
# Prepare a Spark DataFrame.
sdf = (spark.read.format("delta")
    .load("/Volumes/catalog/raw/events")
    .withColumnRenamed("ts", "event_ts"))

# Save as a FluxTable.
(sdf.write
    .format("flux")
    .option("evolve", "true")        # stamp Spark schema into the table
    .option("profile", "archive")    # speed | balanced | archive | brotli
    .option("batchSize", "50000")    # rows per IPC flush (default 10000)
    .mode("append")
    .save("/Volumes/catalog/compressed/events.fluxtable"))
```

### Read

```python
df = spark.read.format("flux").load("/Volumes/catalog/compressed/events.fluxtable")
df.show()
df.filter("region = 'us' AND revenue > 100").count()
```

### Python-only (no cluster library)

The `fluxcompress` Python wheel also ships on PyPI and works without
the Spark connector for single-node Arrow / Polars workflows:

```python
%pip install fluxcompress[polars]>=0.5.0
dbutils.library.restartPython()
```

```python
import pyarrow as pa
import fluxcompress as fc

table = pa.table({"id": range(1_000_000)})
buf = fc.compress(table, profile="archive")
```

## Configuration

| Option        | Default   | Notes                                          |
|---------------|-----------|------------------------------------------------|
| `path`        | —         | Required; directory backing the `.fluxtable/`. |
| `profile`     | `speed`   | Compression profile forwarded to `compressTable`. |
| `batchSize`   | `10000`   | Rows per IPC-to-Flux flush on write.           |
| `evolve`      | `false`   | If `true`, push the Spark schema via `tableEvolve`. |

System properties (set via `--conf spark.driver.extraJavaOptions`
and `spark.executor.extraJavaOptions`):

| Property                | Effect                                                   |
|-------------------------|----------------------------------------------------------|
| `flux.native.path`      | Absolute path to a user-supplied cdylib (overrides bundled). |
| `flux.native.loadlib=true` | Use `System.loadLibrary("flux_jni")` via `java.library.path` instead of extracting from the JAR. |

## Troubleshooting

**`UnsatisfiedLinkError: FluxCompress does not ship a native library for os=…`**

The cdylib for your runtime architecture isn't bundled in the JAR.
This typically only happens on exotic platforms; fall back to a
local cargo build:

```bash
cargo build --release -p flux-jni-bridge
```

then set `spark.driver.extraJavaOptions` and
`spark.executor.extraJavaOptions` to include:

```
-Dflux.native.path=/dbfs/FileStore/flux/libflux_jni.so
```

**`java.lang.ClassNotFoundException: com.datamariners.fluxcompress.spark.FluxDataSource`**

The connector JAR isn't on the classpath.  Revisit option A above;
make sure the library shows green on the Cluster → Libraries page.

**Arrow version conflicts**

Databricks runtimes bundle a specific Arrow version; the connector
targets Arrow 15.0.0 which matches DBR 14.x / 15.x.  Older runtimes
(DBR 12.x and below) may need a shaded build — file an issue if you
hit this.
