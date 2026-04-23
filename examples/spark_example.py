"""
FluxCompress × Apache Spark — Usage Examples
=============================================

FluxCompress integrates with Spark via the JNI bridge (flux-jni-bridge).
The bridge exposes five JNI functions to the JVM:

  FluxNative.compress(ByteBuffer, int)      → byte[]  zero-copy from off-heap
  FluxNative.decompress(byte[])             → byte[]
  FluxNative.compressU128(long[], long[])   → byte[]  dual-register u128
  FluxNative.decompressU128(byte[])         → long[][]

Two integration levels:
  A. PySpark (Python) — via Arrow Flight / pandas UDFs with FluxNative
  B. Spark Scala/Java — direct JNI calls in custom DataSourceV2 or UDFs

Build the JNI bridge first
--------------------------
  cargo build --release -p flux-jni-bridge
  # Produces: target/release/libflux_jni.so  (Linux)
  #           target/release/libflux_jni.dylib (macOS)

Add to Spark
------------
  spark-submit \\
    --conf spark.driver.extraJavaOptions="-Djava.library.path=target/release" \\
    --conf spark.executor.extraJavaOptions="-Djava.library.path=target/release" \\
    --jars path/to/flux-bridge.jar \\
    your_script.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# ── PART A: PySpark integration ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

"""
PySpark example — compress a Spark column to .flux bytes as a new column,
then decompress it back. Uses pandas UDFs (Arrow-backed) for zero-copy
transfer between the JVM Arrow buffer and the Rust engine.
"""

PYSPARK_EXAMPLE = '''
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType, LongType, ArrayType, StructType, StructField
import pyspark.pandas as ps
import numpy as np

# ── Start Spark with the FluxNative library on the class/native path ──────────
spark = (
    SparkSession.builder
    .appName("FluxCompress Demo")
    .config("spark.driver.extraJavaOptions",
            "-Djava.library.path=/path/to/target/release")
    .config("spark.executor.extraJavaOptions",
            "-Djava.library.path=/path/to/target/release")
    # Use Arrow for pandas UDF transfer (matches FluxCompress's Arrow interop).
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Compress a long column → BinaryType (flux bytes) using a Pandas UDF
# ─────────────────────────────────────────────────────────────────────────────

@F.pandas_udf(BinaryType())
def flux_compress_column(series):
    """
    Compress a pandas Series of int64 values into .flux bytes per partition.

    Each executor calls FluxNative.compress() via JPype (or py4j) which routes
    to the Rust JNI function. The Arrow buffer is passed as a DirectByteBuffer
    so Rust reads directly from the off-heap Arrow memory.
    """
    import jpype
    import jpype.imports
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=["path/to/flux-bridge.jar"])
    from com.datamariners.fluxcompress import FluxNative
    import java.nio.ByteBuffer as ByteBuffer

    # Convert pandas Series → numpy int64 array (Arrow-native)
    values = series.values.astype(np.int64)
    n = len(values)

    # Allocate a DirectByteBuffer and fill it (off-heap → Rust zero-copy path)
    buf = FluxNative.allocateU64Buffer(n)
    buf.asLongBuffer().put(values.tolist())
    buf.rewind()

    # Zero-copy compress: Rust reads directly from the DirectByteBuffer pointer
    flux_bytes = bytes(FluxNative.compress(buf, n))
    return flux_bytes  # single bytes value per partition


@F.pandas_udf("array<long>")
def flux_decompress_column(series):
    """Decompress .flux bytes back to an array of long values."""
    import jpype, jpype.imports
    from com.datamariners.fluxcompress import FluxNative

    result = []
    for flux_bytes in series:
        raw = FluxNative.decompress(bytes(flux_bytes))
        # raw is a byte[] of packed little-endian u64 values
        arr = np.frombuffer(bytes(raw), dtype="<u8").tolist()
        result.append(arr)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compress u128 aggregation results (dual-register)
# ─────────────────────────────────────────────────────────────────────────────

@F.pandas_udf(BinaryType())
def flux_compress_u128(hi_series, lo_series):
    """
    Compress u128 values (e.g., SUM(DECIMAL(38,0))) supplied as two long[]
    columns (high word, low word) into .flux bytes with OutlierMap patching.
    """
    import jpype, jpype.imports
    from com.datamariners.fluxcompress import FluxNative

    hi = hi_series.values.astype(np.int64).tolist()
    lo = lo_series.values.astype(np.int64).tolist()
    flux_bytes = bytes(FluxNative.compressU128(hi, lo))
    return flux_bytes


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build a DataFrame and compress it
# ─────────────────────────────────────────────────────────────────────────────

# Sample: 10 million rows of user activity.
df = spark.range(10_000_000).toDF("user_id")
df = df.withColumn("revenue",     (F.col("user_id") * 37 + 99) % 99_999)
df = df.withColumn("region_code", F.col("user_id") % 8)
df = df.withColumn("session_ms",  F.col("user_id") * 1_234 % 86_400_000)

print("Schema:", df.schema)
print(f"Rows: {df.count():,}")
df.show(5)

# Repartition to simulate Spark's parallel write pattern.
df_parts = df.repartition(8)

# Compress the user_id column per partition → store as .flux bytes.
df_compressed = df_parts.withColumn(
    "user_id_flux",
    flux_compress_column(F.col("user_id").cast("long"))
)
df_compressed.show(3, truncate=40)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compress a SUM aggregation result (u128 path)
# ─────────────────────────────────────────────────────────────────────────────

# Spark stores DECIMAL(38,0) SUM results as (high: Long, low: Long) internally.
# Simulate that structure:
df_agg = (
    df.groupBy("region_code")
      .agg(F.sum("revenue").alias("total_revenue"))
      .withColumn("rev_hi", F.lit(0).cast("long"))        # high word (0 here)
      .withColumn("rev_lo", F.col("total_revenue").cast("long"))  # low word
)

df_agg_compressed = df_agg.withColumn(
    "total_revenue_flux",
    flux_compress_u128(F.col("rev_hi"), F.col("rev_lo"))
)
df_agg_compressed.show()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Write .flux files to object storage (S3/GCS/ADLS)
# ─────────────────────────────────────────────────────────────────────────────

# Each partition writes its compressed bytes to a separate .flux object.
# Use Spark's foreachPartition for the write path.

def write_flux_partition(rows, output_prefix: str):
    """Write one partition's compressed flux bytes to object storage."""
    import boto3   # or google.cloud.storage, etc.
    import io

    rows = list(rows)
    if not rows:
        return

    partition_id = rows[0]["user_id"] // 1_250_000  # rough partition key
    key = f"{output_prefix}/part-{partition_id:04d}.flux"

    # Concatenate all flux blobs in this partition into one file.
    combined = b"".join(row["user_id_flux"] for row in rows if row["user_id_flux"])

    s3 = boto3.client("s3")
    s3.put_object(Bucket="my-datalake", Key=key, Body=combined)


OUTPUT_PREFIX = "fluxcompress/user_activity/dt=2024-01-01"
df_compressed.foreachPartition(
    lambda rows: write_flux_partition(rows, OUTPUT_PREFIX)
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Read .flux files back and create a Spark DataFrame
# ─────────────────────────────────────────────────────────────────────────────

@F.pandas_udf("array<long>")
def flux_decompress_s3(s3_key_series):
    """Read .flux files from S3 and decompress to long arrays."""
    import boto3
    import jpype, jpype.imports
    from com.datamariners.fluxcompress import FluxNative

    s3 = boto3.client("s3")
    result = []
    for key in s3_key_series:
        obj = s3.get_object(Bucket="my-datalake", Key=key)
        flux_bytes = obj["Body"].read()
        raw = bytes(FluxNative.decompress(flux_bytes))
        arr = np.frombuffer(raw, dtype="<u8").tolist()
        result.append(arr)
    return result
'''

# ─────────────────────────────────────────────────────────────────────────────
# ── PART B: Scala / JVM (copy into your Spark job) ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

SCALA_EXAMPLE = '''
// FluxCompress × Spark (Scala)
// ============================================================
// Add to build.sbt:
//   libraryDependencies += "com.datamariners.fluxcompress" % "flux-bridge" % "0.1.0"
//
// Add to spark-submit:
//   --conf spark.driver.extraJavaOptions=-Djava.library.path=/path/to/target/release
//   --conf spark.executor.extraJavaOptions=-Djava.library.path=/path/to/target/release

import org.apache.spark.sql.{SparkSession, DataFrame, functions => F}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types._
import com.datamariners.fluxcompress.FluxNative
import java.nio.{ByteBuffer, ByteOrder}


object FluxCompressSparkExample {

  // ── 1. Register Spark UDFs ───────────────────────────────────────────────

  /**
   * Compress a long[] column value to .flux bytes.
   *
   * Usage in SQL:   SELECT flux_compress(user_id) FROM users
   * Usage in Scala: df.withColumn("id_flux", fluxCompress(col("user_id")))
   */
  val fluxCompress: UserDefinedFunction = F.udf { values: Seq[Long] =>
    val n = values.length
    // Allocate a direct ByteBuffer (off-heap) — zero-copy path into Rust.
    val buf = FluxNative.allocateU64Buffer(n)
    buf.order(ByteOrder.LITTLE_ENDIAN)
    values.foreach(buf.asLongBuffer().put(_))
    buf.rewind()
    FluxNative.compress(buf, n)  // → byte[]
  }

  /**
   * Decompress .flux bytes back to a long[].
   */
  val fluxDecompress: UserDefinedFunction = F.udf { fluxBytes: Array[Byte] =>
    val raw: Array[Byte] = FluxNative.decompress(fluxBytes)
    val buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer()
    val out = Array.ofDim[Long](buf.remaining())
    buf.get(out)
    out  // → Array[Long] (becomes ArrayType(LongType) in Spark)
  }

  /**
   * Compress u128 aggregation results via the dual-register (hi, lo) bridge.
   *
   * Spark stores DECIMAL(38,0) internally as (unscaled: Long, scale: Int).
   * For SUM results that overflow Long, pass the BigDecimal's unscaledValue
   * split into hi/lo words.
   */
  val fluxCompressU128: UserDefinedFunction = F.udf {
    (hiWords: Seq[Long], loWords: Seq[Long]) =>
      FluxNative.compressU128(hiWords.toArray, loWords.toArray)
  }

  /**
   * Decompress u128 column. Returns Array[Array[Long]] where:
   *   result(0)(i) = high 64 bits of value i
   *   result(1)(i) = low  64 bits of value i
   */
  val fluxDecompressU128: UserDefinedFunction = F.udf { fluxBytes: Array[Byte] =>
    FluxNative.decompressU128(fluxBytes)  // → Array[Array[Long]]
  }


  // ── 2. Main demo ─────────────────────────────────────────────────────────

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("FluxCompress Spark Demo")
      .config("spark.sql.execution.arrow.pyspark.enabled", "true")
      .getOrCreate()

    import spark.implicits._

    // Register UDFs for SQL access.
    spark.udf.register("flux_compress",   fluxCompress)
    spark.udf.register("flux_decompress", fluxDecompress)

    // ── Build a sample DataFrame ─────────────────────────────────────────
    val n = 10_000_000
    val df = spark.range(n)
      .withColumn("revenue",     ($"id" * 37 + 99) % 99999L)
      .withColumn("region_code", $"id" % 8)
      .withColumn("session_ms",  $"id" * 1234L % 86_400_000L)
      .repartition(8)  // 8 partitions = 8 parallel compressors

    df.printSchema()
    df.show(5)

    // ── Compress per partition ───────────────────────────────────────────
    //
    // Strategy: collect() each partition's column into a Seq[Long], pass
    // to fluxCompress UDF, write the blob to HDFS/S3.
    //
    // For streaming/real-time use the Hot Stream Model (independent blocks).
    // For batch cold storage run fluxcapacitor optimize afterward.

    // Window function: collect all IDs per partition into a single array,
    // then compress the whole array at once.
    val partitionCol = F.spark_partition_id().alias("partition_id")

    val dfWithPart = df.withColumn("partition_id", partitionCol)

    // Collect each partition's user_ids into one Seq, compress → one blob.
    val dfCompressed = dfWithPart
      .groupBy("partition_id")
      .agg(
        F.collect_list($"id").alias("ids"),
        F.collect_list($"revenue").alias("revenues"),
      )
      .withColumn("ids_flux",      fluxCompress($"ids"))
      .withColumn("revenues_flux", fluxCompress($"revenues"))
      .drop("ids", "revenues")

    dfCompressed.printSchema()
    dfCompressed.show(truncate = 40)

    // ── Write compressed blobs to Delta / Parquet sidecar ────────────────
    //
    // Store the raw .flux bytes as a BinaryType column in a Delta table.
    // This gives you Parquet's metadata (partitioning, schema) with
    // FluxCompress's superior numeric compression inside each cell.

    dfCompressed.write
      .mode("overwrite")
      .format("delta")                       // or "parquet"
      .partitionBy("partition_id")
      .save("s3://my-datalake/flux-demo/")

    // ── Read back and decompress ─────────────────────────────────────────
    val dfRead = spark.read.format("delta")
      .load("s3://my-datalake/flux-demo/")

    val dfDecoded = dfRead
      .withColumn("ids_decoded", fluxDecompress($"ids_flux"))
      .withColumn("revenues_decoded", fluxDecompress($"revenues_flux"))

    dfDecoded.show(5, truncate = 60)

    // ── u128 aggregation example ─────────────────────────────────────────
    //
    // Spark computes SUM(revenue) as a Long here, but for DECIMAL(38,0)
    // or very large integer sums, values overflow u64. FluxCompress stores
    // these in the OutlierMap at full 128-bit precision.

    val dfAgg = df
      .groupBy("region_code")
      .agg(F.sum("revenue").alias("total_revenue"))
      .withColumn("rev_hi", F.lit(0L))                  // high word (0 for u64 sums)
      .withColumn("rev_lo", $"total_revenue")            // low word

    // Compress: both words as parallel arrays.
    val dfAggCompressed = dfAgg
      .groupBy()
      .agg(
        F.collect_list($"rev_hi").alias("all_hi"),
        F.collect_list($"rev_lo").alias("all_lo"),
      )
      .withColumn("sums_flux", fluxCompressU128($"all_hi", $"all_lo"))

    dfAggCompressed.show(truncate = 60)

    // Decompress u128 column back to (hi, lo) pairs.
    val dfAggDecoded = dfAggCompressed
      .withColumn("decoded_u128", fluxDecompressU128($"sums_flux"))
      .withColumn("hi_words", $"decoded_u128"(0))
      .withColumn("lo_words", $"decoded_u128"(1))
      .drop("decoded_u128")

    dfAggDecoded.show()

    // ── Predicate pushdown via SQL ────────────────────────────────────────
    //
    // Register the compressed table as a temp view, then use FluxCompress's
    // Atlas-based skipping to read only relevant blocks.
    //
    // The predicate is pushed down through the JNI bridge:
    //   fluxcapacitor decompress --gt 900000 --column value
    //
    // Blocks whose [z_min, z_max] range cannot satisfy (value > 900000)
    // are skipped entirely — no disk reads for irrelevant data.

    dfCompressed.createOrReplaceTempView("flux_table")
    val filtered = spark.sql("""
      SELECT partition_id, flux_decompress(ids_flux) AS ids
      FROM flux_table
      WHERE partition_id > 4
    """)
    filtered.show()

    spark.stop()
  }
}
'''

# ─────────────────────────────────────────────────────────────────────────────
# ── PART C: Architectural diagram + README block ──────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

ARCHITECTURE_NOTES = '''
FluxCompress × Spark — Data Flow
=================================

  ┌─────────────────────────────────────────────────────────────┐
  │                    Apache Spark Executor                    │
  │                                                             │
  │  DataFrame Column (Arrow buffer, off-heap)                  │
  │       │                                                     │
  │       │  DirectByteBuffer (zero-copy pointer)               │
  │       ▼                                                     │
  │  ┌─────────────────────────────────┐                        │
  │  │   FluxNative.compress(buf, n)   │  ← JNI call           │
  │  └────────────┬────────────────────┘                        │
  │               │ JNI boundary                                │
  │  ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
  │               │                                             │
  │       ┌───────▼──────────────────────────────────┐         │
  │       │         Rust (libflux_jni.so)             │         │
  │       │                                           │         │
  │       │  1. Classify chunk (Loom waterfall)       │         │
  │       │  2. Compress → BitSlab / RLE / Delta …    │         │
  │       │  3. Outlier Map for u128 overflow          │         │
  │       │  4. Write Atlas footer                    │         │
  │       │                                           │         │
  │       └───────────────────┬───────────────────────┘         │
  │                           │                                 │
  │                    byte[] .flux blob                        │
  │                           │                                 │
  │       ┌───────────────────▼───────────────────────┐         │
  │       │  Store as BinaryType column in Delta/S3   │         │
  │       └───────────────────────────────────────────┘         │
  └─────────────────────────────────────────────────────────────┘

Key Properties
--------------
• Zero-copy:  Rust reads the Arrow off-heap buffer via DirectByteBuffer
              pointer — no JVM heap → Rust heap copy.
• Parallel:   Each Spark partition compresses independently (Hot model).
• Adaptive:   Loom classifier picks the best strategy per 1024-row segment.
• u128 safe:  Aggregation results > u64::MAX stored in OutlierMap at full
              128-bit precision via the dual-register (hi, lo) bridge.
• Skippable:  Atlas footer enables predicate pushdown — irrelevant blocks
              are never decompressed.

Cold Optimization (after batch write)
--------------------------------------
  # After all partitions are written:
  fluxcapacitor optimize \\
    --input-dir  s3://bucket/flux-partitions/ \\
    --output     s3://bucket/flux-cold/merged.flux

  This runs the two-pass global optimizer:
  Pass 1: scan all partitions → global master dictionary + bit-width stats
  Pass 2: re-pack with Z-Order interleaving → ~10–15% size reduction
'''

if __name__ == "__main__":
    print("FluxCompress × Spark — Usage Examples")
    print("======================================")
    print("\nSee PYSPARK_EXAMPLE, SCALA_EXAMPLE, and ARCHITECTURE_NOTES")
    print("variables in this file for complete copy-paste examples.\n")
    print(ARCHITECTURE_NOTES)
