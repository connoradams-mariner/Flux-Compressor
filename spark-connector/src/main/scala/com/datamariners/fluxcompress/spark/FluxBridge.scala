// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress.spark

import com.datamariners.fluxcompress.FluxNative

import org.apache.spark.sql.fluxcompress.SparkArrowBridge
import org.apache.spark.sql.types.StructType

/**
 * Scala-flavoured facade around [[com.datamariners.fluxcompress.FluxNative]].
 *
 * The raw JNI class is a static `native` surface with unchecked
 * exceptions and manual handle lifecycles.  This facade wraps the
 * handle in a try/finally idiom and centralises option parsing so the
 * read / write classes stay small.
 */
private[spark] object FluxBridge {

  // ── Option parsing ──────────────────────────────────────────────────

  val ProfileOption   = "profile"
  val BatchSizeOption = "batchsize"
  val EvolveOption    = "evolve"

  val DefaultProfile   = "speed"
  val DefaultBatchSize = 10000

  /** Read the `profile` option, lowercased, with `"speed"` fallback. */
  def profile(opts: Map[String, String]): String =
    lookup(opts, ProfileOption).map(_.toLowerCase).getOrElse(DefaultProfile)

  def batchSize(opts: Map[String, String]): Int =
    lookup(opts, BatchSizeOption).map(_.toInt).getOrElse(DefaultBatchSize)

  def evolveRequested(opts: Map[String, String]): Boolean =
    lookup(opts, EvolveOption).exists(_.toLowerCase == "true")

  private def lookup(opts: Map[String, String], key: String): Option[String] =
    opts.iterator.collectFirst { case (k, v) if k.equalsIgnoreCase(key) => v }

  // ── Handle lifecycle ───────────────────────────────────────────────

  /**
   * Run `body` with a freshly-opened FluxTable handle, guaranteeing
   * `tableClose` runs even if the body throws.
   */
  def withHandle[A](path: String)(body: Long => A): A = {
    val h = FluxNative.tableOpen(path)
    try body(h) finally {
      try FluxNative.tableClose(h) catch {
        case _: Throwable => // best-effort — handle may already be closed
      }
    }
  }

  // ── Schema inference ───────────────────────────────────────────────

  /**
   * Infer the Spark schema of the FluxTable at `path`.
   *
   * Strategy:
   *  1. Open the table.  If it has no committed data yet, return an
   *     empty [[StructType]] — Spark will substitute the DataFrame's
   *     own schema on the first write.
   *  2. Otherwise, call `tableScan` and read the embedded Arrow schema
   *     from the IPC prefix without materialising the full dataset.
   */
  def inferSchemaAt(path: String): StructType = withHandle(path) { h =>
    val ipc = try FluxNative.tableScan(h) catch {
      case t: Throwable =>
        // Newly-created tables or read errors before any data exists
        // should surface as "empty schema", not an exception — Spark
        // invokes inferSchema during write planning too.
        return new StructType()
    }
    if (ipc == null || ipc.isEmpty) {
      new StructType()
    } else {
      val allocator = SparkArrowBridge.newRootAllocator("flux-infer")
      try SparkArrowBridge.ipcSchema(ipc, allocator)
      finally allocator.close()
    }
  }

  /**
   * Push a Spark schema into a FluxTable via `tableEvolve`.  No-op
   * when the schemas are already equivalent; caller-provided.
   */
  def evolveTo(path: String, schema: StructType): Long = withHandle(path) { h =>
    val json = FluxSchemaConverter.toTableSchemaJson(schema)
    FluxNative.tableEvolve(h, json)
  }
}
