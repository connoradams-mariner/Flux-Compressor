// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress.spark

import com.datamariners.fluxcompress.FluxNative

import org.apache.arrow.memory.BufferAllocator

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.fluxcompress.SparkArrowBridge
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Read side of the Flux DataSource V2 connector.
 *
 * The read pipeline does the minimum needed to surface a
 * `spark.read.format("flux").load(path)` DataFrame:
 *
 *  1. `FluxScanBuilder` just stores the path / schema / options.
 *  2. `FluxScan` declares the table is batched-columnar: one
 *     partition per logical snapshot (we treat the whole table as one
 *     partition in Phase H; per-file partitioning is a future step
 *     that needs a new `listLiveFiles` JNI entry).
 *  3. `FluxPartitionReader` opens the table, calls `tableScan` to get
 *     all live rows as an Arrow IPC stream, and converts each IPC
 *     record batch to a Spark [[ColumnarBatch]].
 */
class FluxScanBuilder(
    path: String,
    requestedSchema: StructType,
    options: Map[String, String],
) extends ScanBuilder {
  override def build(): Scan = new FluxScan(path, requestedSchema, options)
}

/**
 * Top-level scan description.  Advertises columnar reads so Spark can
 * consume the Arrow vectors directly without a per-row conversion.
 */
class FluxScan(
    path: String,
    requestedSchema: StructType,
    options: Map[String, String],
) extends Scan with Batch with SupportsReportStatistics {

  // Spark calls readSchema() before any IO — inferring on demand is
  // cheap (it only reads the IPC prefix) but may hit the filesystem.
  // Cache the result for the lifetime of this Scan.
  private lazy val cachedSchema: StructType =
    if (requestedSchema != null && requestedSchema.nonEmpty) requestedSchema
    else FluxBridge.inferSchemaAt(path)

  override def readSchema(): StructType = cachedSchema

  override def description(): String = s"FluxScan($path)"

  override def toBatch: Batch = this

  override def planInputPartitions(): Array[InputPartition] =
    Array(FluxInputPartition(path))

  override def createReaderFactory(): PartitionReaderFactory =
    new FluxPartitionReaderFactory(cachedSchema, options)

  // Statistics are not tracked yet — return a conservative "unknown"
  // estimate so Spark's planner doesn't make bad assumptions.
  override def estimateStatistics(): Statistics = new Statistics {
    override def sizeInBytes(): java.util.OptionalLong = java.util.OptionalLong.empty()
    override def numRows(): java.util.OptionalLong     = java.util.OptionalLong.empty()
  }
}

/**
 * A single input partition — carries just the table path today.
 *
 * Per-file partitioning is a future extension: once the JNI layer can
 * list live files, we'll emit one [[FluxInputPartition]] per file so
 * Spark can parallelise the scan across executors.
 */
case class FluxInputPartition(path: String) extends InputPartition

/**
 * Per-task factory that constructs the [[FluxPartitionReader]]s.
 *
 * The factory is serialised from the driver to each executor and must
 * be stable under Java serialisation — hence only simple values.
 */
class FluxPartitionReaderFactory(
    schema: StructType,
    options: Map[String, String],
) extends PartitionReaderFactory {

  // We always produce ColumnarBatches; Spark wraps them for row-based
  // consumers automatically, but executing in columnar mode skips the
  // per-row conversion entirely.
  override def supportColumnarReads(partition: InputPartition): Boolean = true

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] = {
    throw new UnsupportedOperationException(
      "FluxScan always returns columnar batches; Spark should use " +
      "createColumnarReader instead of createReader."
    )
  }

  override def createColumnarReader(
      partition: InputPartition,
  ): PartitionReader[ColumnarBatch] = {
    val p = partition.asInstanceOf[FluxInputPartition]
    new FluxPartitionReader(p.path)
  }
}

/**
 * The actual partition reader.  One instance per task.
 *
 * Implementation:
 *  1. Open a fresh FluxTable handle (executors may be on different
 *     JVMs from the driver — no handle sharing).
 *  2. Call `tableScan` once to fetch the full Arrow IPC stream.
 *  3. Parse that stream into a sequence of [[ColumnarBatch]]es, held
 *     in-memory.  Spark walks them via repeated `next()` / `get()`.
 *  4. On `close()`, close any outstanding batches and the allocator.
 */
class FluxPartitionReader(path: String)
    extends PartitionReader[ColumnarBatch] {

  private val allocator: BufferAllocator = SparkArrowBridge.newRootAllocator(s"flux-read-$path")

  // Eagerly load everything; a future version should stream file-by-file
  // once the JNI surface supports per-file reads.
  private val batches: Iterator[ColumnarBatch] = {
    val handle = FluxNative.tableOpen(path)
    val ipc    = try FluxNative.tableScan(handle)
                 finally FluxNative.tableClose(handle)
    SparkArrowBridge.ipcToBatches(ipc, allocator).iterator
  }

  private var current: ColumnarBatch = _

  override def next(): Boolean = {
    closeCurrent()
    if (batches.hasNext) {
      current = batches.next()
      true
    } else {
      false
    }
  }

  override def get(): ColumnarBatch = current

  override def close(): Unit = {
    closeCurrent()
    // Drain any unread batches so their vectors release before the
    // allocator closes — otherwise Arrow will complain loudly.
    batches.foreach(_.close())
    try allocator.close() catch { case _: Throwable => () }
  }

  private def closeCurrent(): Unit = {
    if (current != null) {
      try current.close() catch { case _: Throwable => () }
      current = null
    }
  }
}
