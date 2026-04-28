# Using FluxCompress on Databricks

FluxCompress ships a Spark DataSource V2 connector that registers the
`flux` short name, so on a Databricks classic cluster you can write:

```python
(sdf.write
    .format("flux")
    .option("evolve", "true")
    .save("/Volumes/catalog/schema/events.fluxtable"))

df = spark.read.format("flux").load("/Volumes/catalog/schema/events.fluxtable")
```

The Maven artefact lives at:

```
com.datamariners.fluxcompress:flux-spark-connector_2.12:0.5.4
```

## Compute tier support matrix

| Compute tier                                       | `format("flux")` DSv2 | Python `fluxcompress` | Notes |
|----------------------------------------------------|:---------------------:|:---------------------:|-------|
| **Classic — Single User**                          | ✅ supported          | ✅ supported          | Recommended. Install Maven JAR + `pip install fluxcompress[spark]`. |
| **Classic — Shared / No Isolation Shared**         | ✅ supported          | ✅ supported          | Same install steps; Databricks propagates the JAR to every executor. |
| **Databricks Serverless Compute**                  | ❌ not today           | ✅ supported          | Driver-side Python-only path only — see "Serverless limitations" below. |
| **SQL Warehouse (Classic / Pro)**                  | ❌ not supported       | n/a                   | SQL warehouses don't accept user-installed libraries. |
| **SQL Warehouse (Serverless)**                     | ❌ not supported       | n/a                   | Same as above. |

> **Short version:** run on **Classic compute** to get the full
> `sdf.write.format("flux").save(path)` integration. On serverless
> you can still use the `fluxcompress` Python package for driver-side
> compress/decompress, but the distributed DataSource V2 write path
> is blocked by platform restrictions (details below).

## Serverless limitations

Databricks Serverless Compute (as of DBR 15.x / 2026) imposes three
restrictions that each independently block
`sdf.write.format("flux")`:

1. **No Maven library installs.** The Libraries UI on serverless is
   PyPI-only. There's no Maven input, no JAR upload, no init scripts,
   and no `spark.sql("ADD JAR ...")`. Without the JAR,
   `spark.read.format("flux")` raises `ClassNotFoundException` before
   any code of ours runs.
2. **`System.load()` / JNI is blocked.** Even if the JAR were on the
   classpath, the serverless Java Security Manager refuses
   `System.load("/tmp/libflux_jni.so")`. Our connector's
   `FluxNativeLoader` extracts the bundled native library at first
   use; that's exactly the API serverless forbids.
3. **DSv2 registration is allow-listed.** Serverless filters the
   `DataSourceRegister` SPI to Databricks-approved connectors.
   Third-party short names like `flux` don't survive the filter.

### Python-only fallback on serverless

PyPI installs *are* allowed on serverless, so the pure-Python path
works — you just have to skip the Spark DSv2 layer and drive
compression from the driver:

```python
%pip install fluxcompress
dbutils.library.restartPython()

import fluxcompress as fc

# `.toPandas()` collects the Spark DataFrame to the driver.
# Works for anything that fits in driver memory.
buf = fc.compress_polars(sdf.toPandas(), profile="archive")
fc.write_flux(buf, "/Volumes/catalog/schema/events.flux")

# Read side (still driver-local):
buf   = fc.read_flux("/Volumes/catalog/schema/events.flux")
table = fc.decompress(buf)
```

Caveats for the driver-side path:

- `sdf.toPandas()` materialises every partition on the driver. For
  anything above a few GB of in-memory Arrow, use Classic compute.
- You lose the per-task `.fluxtable/` transaction log — output is a
  single `.flux` file rather than a Delta-Lake-style directory.
- Predicate pushdown on read still works (`fc.col("id") > 42`), but
  scans happen driver-side, not on the Spark executors.

### Roadmap for serverless support

We're tracking three paths to enable the full DSv2 integration on
serverless:

1. **Pure-JVM writer** — re-implement the `.flux` writer in
   Scala/Java without JNI. Signed Maven JARs that don't touch native
   code are expected to unlock on serverless in late 2026.
2. **Databricks Verified Partner / Custom JARs preview** — a limited
   allow-list program exists for ISVs shipping JARs + native libs.
   We're evaluating it.
3. **Spark DSv2-via-PyPI** — Databricks has signalled an upcoming
   feature that lets a pure-PyPI package register a DataSource V2 by
   name, routing through a Python subprocess. When it GAs, the
   existing `fluxcompress` wheel can hook into it without a JVM
   connector.

If this blocks you today, open an issue on the
[FluxCompress repo](https://github.com/connoradams-mariner/Flux-Compressor/issues)
with your workload shape — that helps us prioritise which path to
chase first.

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
   com.datamariners.fluxcompress:flux-spark-connector_2.12:0.5.4
   ```
   (use `_2.13` if you're on a Databricks runtime that ships Scala 2.13,
   such as DBR 14.x with Photon on Scala 2.13).
3. Click **Install**.  The library attaches automatically on the next
   cluster restart; Databricks propagates it to every executor.

### Option B — Notebook-scoped via `%pip` + Maven sidecar

If you'd rather install per-notebook (for experimentation):

```python
%pip install fluxcompress>=0.5.4
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
JAR=/databricks/jars/flux-spark-connector_2.12-0.5.4.jar
if [ ! -f "$JAR" ]; then
  curl -L -o "$JAR" \
    https://repo1.maven.org/maven2/com/datamariners/fluxcompress/flux-spark-connector_2.12/0.5.4/flux-spark-connector_2.12-0.5.4.jar
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
%pip install 'fluxcompress[polars]>=0.5.4'
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
