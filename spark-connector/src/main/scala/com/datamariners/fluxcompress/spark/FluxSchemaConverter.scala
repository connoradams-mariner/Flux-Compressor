// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress.spark

import org.apache.spark.sql.types._

/**
 * Bridge between Spark's [[org.apache.spark.sql.types.StructType]] and
 * the FluxCompress logical [[TableSchema]] JSON shape that the JNI
 * bridge accepts on `FluxNative.tableEvolve`.
 *
 * The JSON layout must match the Rust `serde` representation declared
 * in `crates/loom/src/txn/schema.rs`:
 *
 * {{{
 * {
 *   "schema_id": 0,
 *   "fields": [
 *     { "field_id": 1, "name": "id",    "dtype": "uint64", "nullable": false },
 *     { "field_id": 2, "name": "label", "dtype": "utf8",   "nullable": true  }
 *   ]
 * }
 * }}}
 *
 * ...where the `dtype` strings are the `FluxDType` serde renames
 * (`"uint8"`, `"int64"`, `"utf8"`, `"timestamp_micros"`, ...).
 *
 * Only types that round-trip safely through Arrow IPC and through the
 * Rust `FluxDType::from_arrow` map are emitted.  Unsupported types
 * surface as [[IllegalArgumentException]] so users see a clear error
 * at write time rather than a cryptic JSON parse failure in the JNI
 * bridge.
 */
private[spark] object FluxSchemaConverter {

  // ── Spark DataType ──→ FluxDType serde name ─────────────────────────

  private def sparkToFluxDtype(dt: DataType): String = dt match {
    // Integers.  Spark has no unsigned types; we default all signed
    // widths to the matching FluxDType and leave unsigned types for
    // callers that explicitly stamp them via `FluxSchemaConverter.overrideDtype`.
    case ByteType     => "int8"
    case ShortType    => "int16"
    case IntegerType  => "int32"
    case LongType     => "int64"

    // Floats.
    case FloatType    => "float32"
    case DoubleType   => "float64"

    // Boolean / date / timestamp.
    case BooleanType  => "boolean"
    case DateType     => "date32"
    case TimestampType | TimestampNTZType => "timestamp_micros"

    // Decimal: always fall back to the compact 128-bit variant.
    // FluxDType::Decimal128 round-trips with precision 38 / scale 10
    // by default; callers that need a different scale should pre-cast
    // their DataFrame columns.
    case _: DecimalType => "decimal128"

    // Variable-length.  Spark `StringType` ↔ Flux `utf8`.  Binary → `binary`.
    case StringType   => "utf8"
    case BinaryType   => "binary"

    case other =>
      throw new IllegalArgumentException(
        s"Spark DataType $other is not supported by the FluxCompress " +
        s"DataSource V2 connector yet.  Cast the column to one of: " +
        s"byte/short/int/long, float/double, boolean, date, timestamp, " +
        s"decimal, string, binary."
      )
  }

  // ── Flux dtype → Spark DataType (used for read-path schema inference) ─

  private def fluxToSparkDtype(name: String): DataType = name match {
    case "uint8"  | "int8"   => ByteType
    case "uint16" | "int16"  => ShortType
    case "uint32" | "int32"  => IntegerType
    case "uint64" | "int64"  => LongType
    case "float32"           => FloatType
    case "float64"           => DoubleType
    case "boolean"           => BooleanType
    case "date32" | "date64" => DateType
    case "timestamp_second" | "timestamp_millis" |
         "timestamp_micros" | "timestamp_nanos"   => TimestampType
    case "decimal128"        => DecimalType(38, 10)
    case "utf8" | "large_utf8"     => StringType
    case "binary" | "large_binary" => BinaryType
    case other =>
      throw new IllegalArgumentException(
        s"FluxDType '$other' cannot be represented as a Spark DataType."
      )
  }

  /**
   * Convert a Spark [[StructType]] into the FluxCompress `TableSchema`
   * JSON that `FluxNative.tableEvolve` accepts.
   *
   * Field ids are assigned 1-based in column order.  Spark's
   * `StructField.nullable` maps directly to `SchemaField.nullable`.
   *
   * @param schema    the logical schema the caller wants on the table
   * @param schemaId  `schema_id` to stamp into the JSON (usually taken
   *                  from an earlier `current_schema_id + 1`; 0 for
   *                  fresh tables)
   * @return          serialised JSON string ready for `tableEvolve`
   */
  def toTableSchemaJson(schema: StructType, schemaId: Int = 0): String = {
    val sb = new StringBuilder
    sb.append("{")
    sb.append(s""""schema_id":$schemaId,""")
    sb.append(""""fields":[""")
    var first = true
    var fieldId = 1
    for (f <- schema.fields) {
      if (!first) sb.append(",")
      first = false
      val dtype = sparkToFluxDtype(f.dataType)
      sb.append("{")
      sb.append(s""""field_id":$fieldId,""")
      sb.append(s""""name":${jsonString(f.name)},""")
      sb.append(s""""dtype":"$dtype",""")
      sb.append(s""""nullable":${f.nullable}""")
      sb.append("}")
      fieldId += 1
    }
    sb.append("]}")
    sb.toString
  }

  /**
   * Inverse of [[toTableSchemaJson]]: parse the minimal subset of the
   * TableSchema JSON that we emit and return the equivalent Spark
   * [[StructType]].  Kept deliberately small — we only parse the
   * `fields[]` array, not the optional evolution metadata.
   *
   * Callers that round-trip through Arrow IPC can skip this entirely
   * and let `ArrowUtils.fromArrowSchema` do the job; this helper exists
   * for the read path when the only available metadata is a Rust-side
   * schema JSON (e.g. when no data files have been written yet).
   */
  def fromTableSchemaJson(json: String): StructType = {
    val fieldsStart = json.indexOf("\"fields\"")
    require(fieldsStart >= 0, s"TableSchema JSON missing 'fields' key: $json")
    val bracketStart = json.indexOf('[', fieldsStart)
    val bracketEnd   = findMatchingBracket(json, bracketStart)
    require(bracketEnd > bracketStart,
      s"TableSchema JSON has unterminated 'fields' array: $json")
    val arrayBody = json.substring(bracketStart + 1, bracketEnd)
    val objects = splitTopLevelObjects(arrayBody)
    val fields = objects.map { obj =>
      val name     = extractString(obj, "name")
      val dtype    = extractString(obj, "dtype")
      val nullable = extractBool(obj, "nullable")
      StructField(name, fluxToSparkDtype(dtype), nullable)
    }
    StructType(fields)
  }

  // ── Tiny purpose-built JSON helpers ─────────────────────────────────
  //
  // We intentionally avoid pulling in jackson / circe / json4s here
  // because Spark already ships a jackson version on the classpath and
  // conflicting transitive versions are a frequent cause of DataSource
  // V2 classloader pain.  The emitted JSON is always well-formed and
  // small, so a hand-rolled parser keeps the dependency footprint
  // minimal.

  private def jsonString(s: String): String = {
    val sb = new StringBuilder("\"")
    s.foreach {
      case '"'  => sb.append("\\\"")
      case '\\' => sb.append("\\\\")
      case '\n' => sb.append("\\n")
      case '\r' => sb.append("\\r")
      case '\t' => sb.append("\\t")
      case c if c < 0x20 => sb.append("\\u%04x".format(c.toInt))
      case c    => sb.append(c)
    }
    sb.append('"')
    sb.toString
  }

  private def findMatchingBracket(s: String, openIdx: Int): Int = {
    var depth = 0
    var i = openIdx
    var inString = false
    var escape = false
    while (i < s.length) {
      val c = s.charAt(i)
      if (inString) {
        if (escape) escape = false
        else if (c == '\\') escape = true
        else if (c == '"')  inString = false
      } else c match {
        case '"' => inString = true
        case '[' | '{' => depth += 1
        case ']' | '}' =>
          depth -= 1
          if (depth == 0) return i
        case _ =>
      }
      i += 1
    }
    -1
  }

  private def splitTopLevelObjects(body: String): Seq[String] = {
    val out = scala.collection.mutable.ArrayBuffer.empty[String]
    var i = 0
    while (i < body.length) {
      // Skip whitespace and commas between objects.
      while (i < body.length && (body.charAt(i).isWhitespace || body.charAt(i) == ',')) i += 1
      if (i < body.length && body.charAt(i) == '{') {
        val end = findMatchingBracket(body, i)
        require(end > i, s"unterminated object in JSON array: $body")
        out += body.substring(i, end + 1)
        i = end + 1
      } else {
        i += 1
      }
    }
    out.toSeq
  }

  private def extractString(obj: String, key: String): String = {
    val needle = s""""$key""""
    val k = obj.indexOf(needle)
    require(k >= 0, s"missing key '$key' in object: $obj")
    // Find the value's opening quote after the colon.
    var i = k + needle.length
    while (i < obj.length && obj.charAt(i) != ':') i += 1
    i += 1 // past ':'
    while (i < obj.length && obj.charAt(i).isWhitespace) i += 1
    require(i < obj.length && obj.charAt(i) == '"',
      s"value for '$key' is not a string in: $obj")
    val sb = new StringBuilder
    i += 1
    var escape = false
    while (i < obj.length) {
      val c = obj.charAt(i)
      if (escape) {
        c match {
          case '"'  => sb.append('"')
          case '\\' => sb.append('\\')
          case 'n'  => sb.append('\n')
          case 'r'  => sb.append('\r')
          case 't'  => sb.append('\t')
          case other => sb.append(other)
        }
        escape = false
      } else c match {
        case '\\' => escape = true
        case '"'  => return sb.toString
        case other => sb.append(other)
      }
      i += 1
    }
    throw new IllegalArgumentException(s"unterminated string for '$key' in: $obj")
  }

  private def extractBool(obj: String, key: String): Boolean = {
    val needle = s""""$key""""
    val k = obj.indexOf(needle)
    require(k >= 0, s"missing key '$key' in object: $obj")
    var i = k + needle.length
    while (i < obj.length && obj.charAt(i) != ':') i += 1
    i += 1
    while (i < obj.length && obj.charAt(i).isWhitespace) i += 1
    if (obj.startsWith("true", i)) true
    else if (obj.startsWith("false", i)) false
    else throw new IllegalArgumentException(
      s"value for '$key' is not a boolean in: $obj"
    )
  }
}
