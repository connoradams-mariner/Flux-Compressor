// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress.spark

import java.io.ByteArrayOutputStream

import com.datamariners.fluxcompress.FluxNative

import org.apache.arrow.memory.{BufferAllocator, RootAllocator}
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.ArrowStreamWriter

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write._
import org.apache.spark.sql.execution.arrow.ArrowWriter
import org.apache.spark.sql.fluxcompress.SparkArrowBridge
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.ArrowUtils

/**
 * Write side of the Flux DataSource V2 connector.
 *
 * Pipeline:
 *  1. [[FluxWriteBuilder]] captures path + schema + options.
 *  2. [[FluxWrite]] materialises as [[FluxBatchWrite]] on
 *     `toBatch`, which optionally pushes the Spark schema into the
 *     FluxTable when the user sets `evolve = true`.
 *  3. Each task runs one [[FluxDataWriter]] that buffers incoming
 *     [[InternalRow]]s into an Arrow [[VectorSchemaRoot]] via
 *     [[ArrowWriter]], flushes `batchSize` rows at a time through
 *     `FluxNative.compressTable` and `FluxNative.tableAppend`, and
 *     reports the version range it produced in its commit message.
 */
class FluxWriteBuilder(
    path: String,
    schema: StructType,
    options: Map[String, String],
) extends WriteBuilder {
  override def build(): Write = new FluxWrite(path, schema, options)
}

class FluxWrite(
    path: String,
    schema: StructType,
    options: Map[String, String],
) extends Write {
  override def toBatch: BatchWrite = new FluxBatchWrite(path, schema, options)
}

class FluxBatchWrite(
    path: String,
    schema: StructType,
    options: Map[String, String],
) extends BatchWrite {

  override def createBatchWriterFactory(info: PhysicalWriteInfo): DataWriterFactory = {
    if (FluxBridge.evolveRequested(options)) {
      // Stamp the schema once on the driver before any tasks start —
      // otherwise racing tasks would each try to evolve independently.
      FluxBridge.evolveTo(path, schema)
    }
    new FluxDataWriterFactory(
      path     = path,
      schema   = schema,
      profile  = FluxBridge.profile(options),
      batchSize = FluxBridge.batchSize(options),
    )
  }

  override def commit(messages: Array[WriterCommitMessage]): Unit = {
    // Nothing to do centrally — each task committed its own appends
    // directly via `tableAppend`.  The message payload is kept for
    // observability only.
    val (totalRows, totalBytes) = messages.collect {
      case m: FluxWriterCommitMessage => (m.rowCount, m.compressedBytes)
    }.fold((0L, 0L))((a, b) => (a._1 + b._1, a._2 + b._2))
    // Leave a breadcrumb on the driver log — Spark prints any
    // Exception thrown here, so intentionally don't throw.
    System.err.println(
      s"[flux] commit: path=$path rows=$totalRows compressed_bytes=$totalBytes"
    )
  }

  override def abort(messages: Array[WriterCommitMessage]): Unit = {
    // Per-task appends were already committed when their `commit()`
    // ran — true atomic multi-task aborts need a two-phase protocol
    // that the current single-file-per-append log format can't offer.
    // Document the limitation rather than pretend to roll back.
    System.err.println(
      s"[flux] abort: path=$path — per-task appends may have already " +
      s"been committed to the transaction log and cannot be rolled " +
      s"back automatically."
    )
  }
}

/**
 * Serialisable factory shipped to each executor.
 */
class FluxDataWriterFactory(
    path: String,
    schema: StructType,
    profile: String,
    batchSize: Int,
) extends DataWriterFactory {
  override def createWriter(
      partitionId: Int,
      taskId: Long,
  ): DataWriter[InternalRow] =
    new FluxDataWriter(path, schema, profile, batchSize)
}

/** Lightweight commit message carrying just what the driver logs. */
case class FluxWriterCommitMessage(
    rowCount: Long,
    compressedBytes: Long,
    lastVersion: Long,
) extends WriterCommitMessage

/**
 * The per-task writer.
 *
 * Buffers `batchSize` rows in an Arrow [[VectorSchemaRoot]], flushes
 * to the Flux native layer, then resets the root for the next batch.
 * Handles rollover cleanly on `commit()` and releases all resources
 * on `close()` even if the task aborted.
 */
class FluxDataWriter(
    path: String,
    schema: StructType,
    profile: String,
    batchSize: Int,
) extends DataWriter[InternalRow] {

  // One allocator per task, shared across all flushes.
  private val allocator: BufferAllocator =
    new RootAllocator(Long.MaxValue)

  // Arrow schema derived once — timezone UTC to match the Rust bridge.
  private val arrowSchema =
    ArrowUtils.toArrowSchema(schema, "UTC", errorOnDuplicatedFieldNames = false, largeVarTypes = false)

  private var root: VectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator)
  private var arrowWriter: ArrowWriter = ArrowWriter.create(root)

  // Opened lazily so aborted tasks that never receive a row don't
  // create an empty `.fluxtable/` directory.
  private var handle: Long = -1L
  private var rowsInBatch: Int = 0
  private var totalRows: Long = 0L
  private var totalBytes: Long = 0L
  private var lastVersion: Long = -1L

  override def write(record: InternalRow): Unit = {
    if (handle < 0) handle = FluxNative.tableOpen(path)
    arrowWriter.write(record)
    rowsInBatch += 1
    totalRows   += 1
    if (rowsInBatch >= batchSize) flushBatch()
  }

  /**
   * Serialise the current [[VectorSchemaRoot]] as an IPC stream,
   * compress + append it to the Flux table, then reset the root
   * in-place for the next batch.
   */
  private def flushBatch(): Unit = {
    if (rowsInBatch == 0) return
    arrowWriter.finish()

    val bos    = new ByteArrayOutputStream()
    val writer = new ArrowStreamWriter(root, null, bos)
    writer.start()
    writer.writeBatch()
    writer.end()
    val ipcBytes = bos.toByteArray

    val fluxBytes = FluxNative.compressTable(ipcBytes, profile)
    lastVersion = FluxNative.tableAppend(handle, fluxBytes)
    totalBytes += fluxBytes.length

    // Reset for the next batch.  `root.close()` + recreate is simpler
    // and safer than trying to reuse the vectors in-place; Arrow's
    // `VectorSchemaRoot.allocateNew` is brittle across releases.
    arrowWriter.reset()
    rowsInBatch = 0
  }

  override def commit(): WriterCommitMessage = {
    flushBatch()
    FluxWriterCommitMessage(totalRows, totalBytes, lastVersion)
  }

  override def abort(): Unit = {
    // Drop any buffered rows; already-appended batches are durable.
    rowsInBatch = 0
  }

  override def close(): Unit = {
    try if (root != null) { root.close(); root = null } catch { case _: Throwable => () }
    try if (allocator != null) allocator.close() catch { case _: Throwable => () }
    if (handle >= 0) {
      try FluxNative.tableClose(handle) catch { case _: Throwable => () }
      handle = -1
    }
  }
}
