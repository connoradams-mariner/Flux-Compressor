// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package org.apache.spark.sql.fluxcompress

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import org.apache.arrow.memory.{BufferAllocator, RootAllocator}
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.arrow.ArrowWriter
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.sql.vectorized.{ArrowColumnVector, ColumnVector, ColumnarBatch}

/**
 * Bridge helpers between Spark's columnar internals and the Arrow IPC
 * stream format that the FluxCompress JNI layer speaks.
 *
 * This file lives under `org.apache.spark.sql.*` on purpose — Spark's
 * `ArrowUtils` and `ArrowWriter` are `private[sql]` in Spark 3.5 so
 * only code in that package namespace can call them.  Everything the
 * connector exposes to users lives under `com.datamariners.fluxcompress.spark.*`;
 * this module is the thin Spark-private shim it calls into.
 */
private[fluxcompress] object SparkArrowBridge {

  /** Timezone string used for Arrow timestamp vectors.  UTC keeps the
   *  on-disk representation timezone-stable and matches the JNI
   *  bridge's `FluxDType::TimestampMicros` mapping. */
  private val TimeZone = "UTC"

  /**
   * Allocate a root Arrow allocator.  Callers are responsible for
   * closing it; typical usage is `try { ... } finally { allocator.close() }`.
   */
  def newRootAllocator(tag: String): BufferAllocator =
    new RootAllocator(Long.MaxValue)

  /**
   * Serialise Spark [[InternalRow]]s into an Arrow IPC *stream* payload
   * using Spark's own [[ArrowWriter]].  The returned bytes are ready to
   * be handed to `FluxNative.compressTable`.
   *
   * @param rows    the rows to serialise (consumed)
   * @param schema  Spark schema describing the rows
   * @return        IPC stream bytes containing a single record batch
   */
  def rowsToIpc(
      rows: Iterator[InternalRow],
      schema: StructType,
      allocator: BufferAllocator,
  ): Array[Byte] = {
    val arrowSchema = ArrowUtils.toArrowSchema(schema, TimeZone, errorOnDuplicatedFieldNames = false, largeVarTypes = false)
    val root        = VectorSchemaRoot.create(arrowSchema, allocator)
    val arrowWriter = ArrowWriter.create(root)
    try {
      while (rows.hasNext) {
        arrowWriter.write(rows.next())
      }
      arrowWriter.finish()

      val out     = new ByteArrayOutputStream()
      val writer  = new ArrowStreamWriter(root, null, out)
      writer.start()
      writer.writeBatch()
      writer.end()
      out.toByteArray
    } finally {
      // `root.close()` recursively closes its vectors.
      root.close()
    }
  }

  /**
   * Deserialise an Arrow IPC *stream* payload into a sequence of
   * Spark [[ColumnarBatch]]es.  The batches own their Arrow vectors
   * via [[ArrowColumnVector]] and must be `close()`d by the caller.
   *
   * @param ipc       IPC stream bytes (output of Flux `decompressTable`
   *                  or the multi-batch `tableScan`)
   * @param allocator Arrow allocator backing the returned vectors —
   *                  callers must keep it alive at least as long as
   *                  the returned batches
   * @return          one [[ColumnarBatch]] per IPC record-batch message
   */
  def ipcToBatches(
      ipc: Array[Byte],
      allocator: BufferAllocator,
  ): Seq[ColumnarBatch] = {
    if (ipc == null || ipc.isEmpty) return Seq.empty
    val in     = new ByteArrayInputStream(ipc)
    val reader = new ArrowStreamReader(in, allocator)
    try {
      val out = scala.collection.mutable.ArrayBuffer.empty[ColumnarBatch]
      val root = reader.getVectorSchemaRoot
      while (reader.loadNextBatch()) {
        val rowCount = root.getRowCount
        val vectors: Array[ColumnVector] = (0 until root.getSchema.getFields.size).map { i =>
          new ArrowColumnVector(root.getVector(i)): ColumnVector
        }.toArray
        val batch = new ColumnarBatch(vectors)
        batch.setNumRows(rowCount)
        out += batch
        // We intentionally do NOT close `root` between batches — each
        // ColumnarBatch captures live Arrow vectors.  Callers close the
        // batches, which closes the vectors, which the allocator owns.
      }
      out.toSeq
    } finally {
      // The reader's root is kept alive by the ColumnarBatches above.
      // We still close the reader so the stream's own buffers are freed.
      reader.close(false)
    }
  }

  /**
   * Peek at the Arrow schema embedded in an IPC stream without
   * materialising its record batches.  Used by the read path to infer
   * the Spark `StructType` for schema-less opens.
   */
  def ipcSchema(ipc: Array[Byte], allocator: BufferAllocator): StructType = {
    if (ipc == null || ipc.isEmpty) return new StructType()
    val in     = new ByteArrayInputStream(ipc)
    val reader = new ArrowStreamReader(in, allocator)
    try {
      // Triggers schema read without loading any batches.
      val arrowSchema = reader.getVectorSchemaRoot.getSchema
      ArrowUtils.fromArrowSchema(arrowSchema)
    } finally {
      reader.close(false)
    }
  }
}
