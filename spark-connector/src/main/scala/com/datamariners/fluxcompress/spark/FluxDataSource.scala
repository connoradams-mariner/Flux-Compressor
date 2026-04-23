// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress.spark

import java.util

import scala.collection.JavaConverters._

import org.apache.spark.sql.connector.catalog.{SupportsRead, SupportsWrite, Table, TableCapability, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, WriteBuilder}
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * Spark DataSource V2 entry point for FluxCompress.
 *
 * Registering the short name `"flux"` via
 * `META-INF/services/org.apache.spark.sql.sources.DataSourceRegister`
 * lets users say:
 * {{{
 *   val df = spark.read.format("flux").load("/path/to/table")
 *   df.write.format("flux").save("/path/to/table")
 * }}}
 *
 * Options (all case-insensitive, but the names below are canonical):
 *  - `path` — mandatory; directory that backs the `.fluxtable/` layout.
 *  - `profile` — `speed` | `balanced` | `archive` | `brotli` (default `speed`).
 *  - `batchSize` — rows per IPC flush on write (default 10000).
 *  - `evolve` — `true` to push the Spark schema to the Flux table via
 *    `FluxNative.tableEvolve` before any rows are written.
 */
class FluxDataSource extends TableProvider with DataSourceRegister {

  override def shortName(): String = "flux"

  override def supportsExternalMetadata(): Boolean = true

  override def inferSchema(options: CaseInsensitiveStringMap): StructType = {
    // Empty tables (newly-created paths) must yield an empty schema
    // rather than throw — Spark always calls `inferSchema` when the
    // user has not passed an explicit schema, including on first write.
    val path = requirePath(options)
    FluxBridge.inferSchemaAt(path)
  }

  override def inferPartitioning(options: CaseInsensitiveStringMap): Array[Transform] =
    Array.empty

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: util.Map[String, String],
  ): Table = {
    val opts = new CaseInsensitiveStringMap(properties)
    val path = requirePath(opts)
    new FluxSparkTable(path, schema, opts)
  }

  private def requirePath(opts: CaseInsensitiveStringMap): String = {
    val p = Option(opts.get("path")).getOrElse(
      throw new IllegalArgumentException(
        "FluxCompress DataSource requires a 'path' option, e.g. " +
        "`.load(\"/tmp/my.fluxtable\")` or `.option(\"path\", ...)`"
      )
    )
    p
  }
}

/**
 * Concrete `Table` handle holding a path + schema + user options.
 *
 * Declares the standard read / write capabilities.  Schema is taken
 * verbatim from whatever Spark hands us — `inferSchema` on an empty
 * path returns an empty `StructType`, which Spark replaces with the
 * DataFrame's own schema on the first write.
 */
class FluxSparkTable(
    val path: String,
    private val tableSchema: StructType,
    options: CaseInsensitiveStringMap,
) extends Table with SupportsRead with SupportsWrite {

  override def name(): String = s"flux:$path"

  override def schema(): StructType = tableSchema

  override def capabilities(): util.Set[TableCapability] = {
    val caps = new util.HashSet[TableCapability]()
    caps.add(TableCapability.BATCH_READ)
    caps.add(TableCapability.BATCH_WRITE)
    caps.add(TableCapability.ACCEPT_ANY_SCHEMA)
    caps
  }

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new FluxScanBuilder(path, tableSchema, mergeOptions(options))

  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder =
    new FluxWriteBuilder(path, info.schema(), mergeOptions(info.options()))

  /** Merge Table-level options with those supplied to the scan/write
   *  builder.  Builder-provided options take precedence. */
  private def mergeOptions(extra: CaseInsensitiveStringMap): Map[String, String] = {
    val base = options.asScala.toMap
    base ++ extra.asScala.toMap
  }
}
