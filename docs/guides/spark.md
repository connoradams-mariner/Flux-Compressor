# Apache Spark Integration Guide

`fluxcompress.spark` provides PySpark UDFs and DataFrame helpers for
integrating FluxCompress into Spark pipelines.

## Setup

```bash
pip install fluxcompress[spark]
```

Build the JNI bridge (for the zero-copy Java path):
```bash
cargo build --release -p flux-jni-bridge
# → target/release/libflux_jni.so  (Linux)
# → target/release/libflux_jni.dylib  (macOS)
```

## Quick Start

```python
from pyspark.sql import SparkSession
import fluxcompress.spark as fcs

spark = (
    SparkSession.builder
    .appName("FluxCompress Demo")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

df = spark.range(10_000_000).toDF("user_id")

# Add a compressed column (one .flux blob per partition)
df2 = fcs.compress_dataframe(df, column="user_id")
df2.printSchema()
# root
#  |-- user_id: long (nullable = false)
#  |-- user_id_flux: binary (nullable = true)

# Decompress back
df3 = fcs.decompress_dataframe(df2, flux_column="user_id_flux")
```

## Register SQL UDFs

```python
fcs.register_udfs(spark)

# Now usable in Spark SQL:
spark.sql("""
    SELECT flux_compress(user_id) AS user_id_flux
    FROM range(1000000)
""").show()
```

Registered UDFs:

| Name | Input | Output | Description |
|---|---|---|---|
| `flux_compress` | `LongType` | `BinaryType` | Compress a long column |
| `flux_decompress` | `BinaryType` | `ArrayType(LongType)` | Decompress to array |

## Partition-Level Compression

For batch workloads, compress each partition independently (the Hot Stream
Model) then run `fluxcapacitor optimize` for Cold storage:

```python
from functools import partial

compressed_rdd = (
    df.rdd
    .mapPartitionsWithIndex(
        partial(fcs.compress_partition_indexed, column_index=0)
    )
)
# Each element: (partition_id, flux_bytes)
```

## Write Compressed Blobs to Object Storage

```python
def write_to_s3(rows, bucket, prefix):
    import boto3
    rows = list(rows)
    if not rows:
        return
    partition_id = rows[0]["partition_id"]
    blob = b"".join(r["user_id_flux"] for r in rows if r["user_id_flux"])
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=f"{prefix}/part-{partition_id:04d}.flux",
        Body=blob,
    )

df2.foreachPartition(lambda rows: write_to_s3(rows, "my-bucket", "flux/"))
```

## Cold Optimization After Batch Write

After all partitions are written, run the two-pass cold optimizer:

```bash
# Via CLI
fluxcapacitor optimize \
  --input-dir s3://my-bucket/flux/ \
  --output    s3://my-bucket/flux-cold/merged.flux

# Or from Python
import subprocess
subprocess.run([
    "fluxcapacitor", "optimize",
    "--input-dir", "/mnt/flux-partitions/",
    "--output",    "/mnt/flux-cold/merged.flux",
], check=True)
```

Pass 1 builds a global master dictionary and discovers the optimal global
bit-width.  Pass 2 re-packs all blocks with Z-Order interleaving, typically
saving 10–15% over the sum of independently compressed partition files.

## Zero-Copy JNI Path (Scala/Java)

For production Spark jobs written in Scala, use the JNI bridge directly to
avoid the Python overhead:

```scala
import io.fluxcompress.FluxNative

val fluxCompress = udf { values: Seq[Long] =>
  val n   = values.length
  val buf = FluxNative.allocateU64Buffer(n)
  buf.asLongBuffer().put(values.toArray)
  buf.rewind()
  FluxNative.compress(buf, n)  // zero-copy: reads from off-heap Arrow buffer
}

df.withColumn("id_flux", fluxCompress(collect_list($"user_id")))
```

See [`FluxNative.java`](../../java/io/fluxcompress/FluxNative.java) for the
complete Java API including the u128 dual-register bridge for
`DECIMAL(38,0)` / large `SUM` aggregation results.
