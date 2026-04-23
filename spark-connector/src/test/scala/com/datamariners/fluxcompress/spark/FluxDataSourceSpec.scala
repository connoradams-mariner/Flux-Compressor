// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress.spark

import java.nio.file.{Files, Paths}

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

/**
 * Tests for the FluxCompress Spark DataSource V2 connector.
 *
 * The suite is split into two tiers:
 *
 *  - **Schema-converter tests** — pure Scala, no native library
 *    needed.  Always run.
 *
 *  - **End-to-end DataFrame round-trip** — exercises the full Spark
 *    pipeline by writing and reading a small DataFrame.  Requires
 *    the `flux_jni` cdylib on `java.library.path`; skipped cleanly
 *    when absent so CI jobs that haven't built the Rust side still
 *    pass.
 */
class FluxDataSourceSpec extends AnyFunSuite with BeforeAndAfterAll {

  // ── Detection of the native library ────────────────────────────────

  private lazy val nativeAvailable: Boolean = {
    // `System.loadLibrary` does not expose a "can this resolve?" check;
    // the only reliable signal is attempting to load and catching the
    // `UnsatisfiedLinkError`.  We do it once at suite startup so each
    // test doesn't re-pay the cost.
    val paths = Option(System.getProperty("java.library.path"))
      .getOrElse("")
      .split(java.io.File.pathSeparatorChar)
    val candidates = Seq("libflux_jni.so", "libflux_jni.dylib", "flux_jni.dll")
    paths.exists { dir =>
      candidates.exists { name => Files.exists(Paths.get(dir, name)) }
    }
  }

  // ── Lazy Spark session for the e2e tests ──────────────────────────

  private lazy val spark: SparkSession = SparkSession.builder()
    .appName("FluxDataSourceSpec")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    // Force single-partition writes for determinism — multi-task
    // writes append independently and the order of log versions is
    // task-arrival order, which is unstable in unit tests.
    .config("spark.default.parallelism", "1")
    .getOrCreate()

  override def afterAll(): Unit = {
    if (nativeAvailable) spark.stop()
    super.afterAll()
  }

  // ── Schema converter tests ─────────────────────────────────────────

  test("FluxSchemaConverter: emits canonical TableSchema JSON for primitives") {
    val schema = StructType(Array(
      StructField("id",    LongType,       nullable = false),
      StructField("score", DoubleType,     nullable = false),
      StructField("label", StringType,     nullable = true),
      StructField("when",  TimestampType,  nullable = true),
    ))
    val json = FluxSchemaConverter.toTableSchemaJson(schema)
    assert(json.contains(""""name":"id""""))
    assert(json.contains(""""dtype":"int64""""))
    assert(json.contains(""""dtype":"float64""""))
    assert(json.contains(""""dtype":"utf8""""))
    assert(json.contains(""""dtype":"timestamp_micros""""))
    assert(json.contains(""""nullable":false"""))
    assert(json.contains(""""nullable":true"""))
  }

  test("FluxSchemaConverter: round-trips through fromTableSchemaJson") {
    val schema = StructType(Array(
      StructField("id",    LongType,    nullable = false),
      StructField("label", StringType,  nullable = true),
    ))
    val round = FluxSchemaConverter.fromTableSchemaJson(
      FluxSchemaConverter.toTableSchemaJson(schema))
    assert(round.fields.length === schema.fields.length)
    assert(round.fields(0).name === "id")
    assert(round.fields(0).dataType === LongType)
    assert(round.fields(1).dataType === StringType)
  }

  test("FluxSchemaConverter: rejects unsupported types") {
    intercept[IllegalArgumentException] {
      FluxSchemaConverter.toTableSchemaJson(
        StructType(Array(
          StructField("arr", ArrayType(IntegerType))
        ))
      )
    }
  }

  // ── End-to-end round-trip (requires native library) ────────────────

  test("DataFrame round-trip through format(\"flux\")") {
    assume(nativeAvailable,
      "flux_jni not on java.library.path — skipping e2e test. " +
      "Run `cargo build --release -p flux-jni-bridge` first.")

    import spark.implicits._

    val tmpDir = Files.createTempDirectory("flux-spark-spec-").toFile
    tmpDir.deleteOnExit()
    val path = tmpDir.getAbsolutePath

    // Write a simple DataFrame, then read it back.  Coalesce to a
    // single partition so the test remains order-deterministic.
    val df = Seq(
      (1L, 1.5, "alpha"),
      (2L, 2.5, "beta"),
      (3L, 3.5, "gamma"),
    ).toDF("id", "score", "label").coalesce(1)

    df.write.format("flux").option("evolve", "true").save(path)

    val out = spark.read.format("flux").load(path)
    assert(out.count() === 3L)
    val rows = out.collect().map(r => (r.getLong(0), r.getDouble(1), r.getString(2))).toSet
    assert(rows === Set((1L, 1.5, "alpha"), (2L, 2.5, "beta"), (3L, 3.5, "gamma")))
  }
}
