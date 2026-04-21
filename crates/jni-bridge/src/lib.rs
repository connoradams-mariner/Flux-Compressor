// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # JNI Bridge (Phase G)
//!
//! Exposes the full FluxCompress surface to the Spark/JVM ecosystem via JNI.
//!
//! ## Layers
//!
//! ### Layer 1 — Single-column numeric (Sprint 5 / Phase F legacy)
//! Direct compression of a `DirectByteBuffer` of u64 values or two parallel
//! `long[]` arrays for u128 columns.  Zero-copy: the JNI bridge maps the
//! off-heap buffer into Rust without any heap copy.
//!
//! ### Layer 2 — Multi-column Arrow IPC batch (Phase G)
//! Exchange full multi-column [`RecordBatch`]es as Arrow IPC *stream* bytes.
//! The Arrow IPC stream format is natively supported by `org.apache.arrow` on
//! the Java side, so Spark can hand off partition data with a single
//! `ArrowStreamWriter` call.
//!
//! Compression profile is selected by a `String` argument:
//! `"speed"` (default), `"balanced"`, `"archive"`, or `"brotli"`.
//!
//! ### Layer 3 — FluxTable handle API (Phase G)
//! Stateful FluxTable operations exposed via opaque `long` handles that
//! index into a global thread-safe registry (`table_registry`).  Handles are
//! created by `tableOpen`, kept alive across multiple JNI calls, and
//! destroyed by `tableClose`.
//!
//! Schema evolution is driven from the JVM side by passing a JSON-serialised
//! `TableSchema` to `tableEvolve`.
//!
//! ## u128 "Dual-Register" Strategy
//! The JVM has no native `u128` type.  We represent each 128-bit value as a
//! `long[2]` array `{hi, lo}`.  See `u128_bridge` for the bit-level
//! reconstruction.
//!
//! ## Java API
//! ```java
//! package io.fluxcompress;
//!
//! public class FluxNative {
//!     static { System.loadLibrary("flux_jni"); }
//!
//!     // ── Layer 1: single-column numeric ───────────────────────────────────
//!
//!     /** Compress a DirectByteBuffer of u64 values → .flux bytes. */
//!     public static native byte[] compress(ByteBuffer data, int valueCount);
//!
//!     /** Decompress .flux bytes → flat little-endian u64 bytes. */
//!     public static native byte[] decompress(byte[] fluxData);
//!
//!     /** Compress a u128 column (two parallel long[] hi/lo arrays) → .flux bytes. */
//!     public static native byte[] compressU128(long[] hi, long[] lo);
//!
//!     /** Decompress a .flux u128 block → long[2][] {hi[], lo[]}. */
//!     public static native long[][] decompressU128(byte[] fluxData);
//!
//!     // ── Layer 2: multi-column Arrow IPC batch ────────────────────────────
//!
//!     /**
//!      * Compress an Arrow IPC stream payload into .flux bytes.
//!      *
//!      * @param ipcBatch  Arrow IPC stream bytes (ArrowStreamWriter output).
//!      * @param profile   "speed" | "balanced" | "archive" | "brotli".
//!      * @return          Compressed .flux bytes.
//!      */
//!     public static native byte[] compressTable(byte[] ipcBatch, String profile);
//!
//!     /**
//!      * Decompress .flux bytes back to an Arrow IPC stream payload.
//!      *
//!      * @param fluxData  Compressed .flux bytes.
//!      * @return          Arrow IPC stream bytes (read with ArrowStreamReader).
//!      */
//!     public static native byte[] decompressTable(byte[] fluxData);
//!
//!     // ── Layer 3: FluxTable handle API ────────────────────────────────────
//!
//!     /**
//!      * Open (or create) a FluxTable at the given filesystem path.
//!      *
//!      * @param path  Directory path for the .fluxtable/ directory.
//!      * @return      Opaque handle (> 0). Caller must call tableClose().
//!      */
//!     public static native long tableOpen(String path);
//!
//!     /**
//!      * Append pre-compressed .flux bytes to an open table.
//!      *
//!      * Typically the caller first compresses a RecordBatch with
//!      * compressTable(), then appends the resulting bytes here.
//!      *
//!      * @param handle   Handle returned by tableOpen().
//!      * @param fluxData Compressed .flux bytes.
//!      * @return         Transaction log version number of the new append.
//!      */
//!     public static native long tableAppend(long handle, byte[] fluxData);
//!
//!     /**
//!      * Scan all live files in the table and return the result as a
//!      * concatenated Arrow IPC stream.
//!      *
//!      * The returned stream may span multiple record-batch messages when
//!      * the table has more than one data file.  Read with
//!      * ArrowStreamReader in a loop.
//!      *
//!      * @param handle  Handle returned by tableOpen().
//!      * @return        Arrow IPC stream bytes, or an empty array for empty tables.
//!      */
//!     public static native byte[] tableScan(long handle);
//!
//!     /**
//!      * Evolve the table's logical schema.
//!      *
//!      * The schema is supplied as a JSON-serialised {@code TableSchema}
//!      * (see the Rust crate docs for the exact format).
//!      *
//!      * @param handle     Handle returned by tableOpen().
//!      * @param schemaJson JSON representation of the new TableSchema.
//!      * @return           Transaction log version of the schema-change entry.
//!      */
//!     public static native long tableEvolve(long handle, String schemaJson);
//!
//!     /**
//!      * Close and release a table handle.
//!      *
//!      * After this call the handle is invalid; further use will throw.
//!      *
//!      * @param handle  Handle returned by tableOpen().
//!      */
//!     public static native void tableClose(long handle);
//!
//!     /**
//!      * Return the current (latest) version of an open table.
//!      *
//!      * @param handle  Handle returned by tableOpen().
//!      * @return        Latest committed version number, or -1 if the table
//!      *               has no committed transactions yet.
//!      */
//!     public static native long tableCurrentVersion(long handle);
//! }
//! ```

#![allow(non_snake_case)]

use jni::JNIEnv;
use jni::objects::{
    JByteArray, JByteBuffer, JClass, JLongArray, JObject, JObjectArray, JString,
};
use jni::sys::{jbyteArray, jlong, jlongArray, jobjectArray, jint};

use loom::{
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    traits::{LoomCompressor, LoomDecompressor, Predicate},
    txn::FluxTable,
    CompressionProfile,
};

mod u128_bridge;
pub use u128_bridge::{jlong_pair_to_u128, u128_to_jlong_pair};

mod ipc_bridge;
mod table_registry;

// ─────────────────────────────────────────────────────────────────────────────
// compress — u64 column (DirectByteBuffer)
// ─────────────────────────────────────────────────────────────────────────────

/// JNI: `byte[] FluxNative.compress(ByteBuffer data, int valueCount)`
///
/// `data` must be a `DirectByteBuffer` holding `valueCount` little-endian u64
/// values (8 bytes each).  Returns the compressed `.flux` bytes.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_compress(
    mut env: JNIEnv,
    _class: JClass,
    data: JByteBuffer,
    value_count: jint,
) -> jbyteArray {
    let result = compress_direct_buffer(&mut env, &data, value_count as usize);
    match result {
        Ok(bytes) => {
            match env.byte_array_from_slice(&bytes) {
                Ok(arr) => arr.into_raw(),
                Err(e) => {
                    let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
                    std::ptr::null_mut()
                }
            }
        }
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn compress_direct_buffer(
    env: &mut JNIEnv,
    buffer: &JByteBuffer,
    value_count: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Zero-copy: get raw pointer into the JVM's off-heap DirectByteBuffer.
    let ptr = env.get_direct_buffer_address(buffer)?;
    let capacity = env.get_direct_buffer_capacity(buffer)? as usize;

    let needed = value_count * 8;
    if capacity < needed {
        return Err(format!(
            "DirectByteBuffer too small: need {needed} bytes, have {capacity}"
        ).into());
    }

    // SAFETY: `ptr` is valid for `capacity` bytes as guaranteed by the JVM.
    // We borrow it as a `&[u8]` for the duration of this call.
    let raw: &[u8] = unsafe { std::slice::from_raw_parts(ptr, needed) };

    // Interpret as u64 values and build a u128 column.
    let values: Vec<u128> = raw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as u128)
        .collect();

    // Compress.
    use loom::compressors::flux_writer::compress_chunk;
    use loom::loom_classifier::classify;
    let strategy = classify(&values).strategy;
    let block = compress_chunk(&values, strategy)?;
    Ok(block)
}

// ─────────────────────────────────────────────────────────────────────────────
// compressU128 — u128 column as dual long[] arrays (hi, lo)
// ─────────────────────────────────────────────────────────────────────────────

/// JNI: `byte[] FluxNative.compressU128(long[] hi, long[] lo)`
///
/// Accepts two parallel `long[]` arrays representing the high and low 64 bits
/// of each u128 value.  Reconstructs full `u128` in Rust and compresses.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_compressU128(
    mut env: JNIEnv,
    _class: JClass,
    hi_arr: JLongArray,
    lo_arr: JLongArray,
) -> jbyteArray {
    let result = compress_u128_arrays(&mut env, &hi_arr, &lo_arr);
    match result {
        Ok(bytes) => env
            .byte_array_from_slice(&bytes)
            .map(|a| a.into_raw())
            .unwrap_or_else(|_| std::ptr::null_mut()),
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn compress_u128_arrays(
    env: &mut JNIEnv,
    hi_arr: &JLongArray,
    lo_arr: &JLongArray,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let hi_len = env.get_array_length(hi_arr)? as usize;
    let lo_len = env.get_array_length(lo_arr)? as usize;
    if hi_len != lo_len {
        return Err("hi and lo arrays must have equal length".into());
    }

    let mut hi_buf = vec![0i64; hi_len];
    let mut lo_buf = vec![0i64; lo_len];

    env.get_long_array_region(hi_arr, 0, &mut hi_buf)?;
    env.get_long_array_region(lo_arr, 0, &mut lo_buf)?;

    // Reconstruct u128 values from dual-register pairs.
    let values: Vec<u128> = hi_buf
        .iter()
        .zip(lo_buf.iter())
        .map(|(&h, &l)| jlong_pair_to_u128(h, l))
        .collect();

    use loom::compressors::flux_writer::compress_chunk;
    use loom::loom_classifier::classify;
    let strategy = classify(&values).strategy;
    let block = compress_chunk(&values, strategy)?;
    Ok(block)
}

// ─────────────────────────────────────────────────────────────────────────────
// decompress — returns byte[] of u64 values
// ─────────────────────────────────────────────────────────────────────────────

/// JNI: `byte[] FluxNative.decompress(byte[] fluxData)`
///
/// Decompresses a flux block and returns the values as a flat byte array of
/// little-endian u64 values (8 bytes each).
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_decompress(
    mut env: JNIEnv,
    _class: JClass,
    flux_data: JByteArray,
) -> jbyteArray {
    let result = decompress_to_u64_bytes(&mut env, &flux_data);
    match result {
        Ok(bytes) => env
            .byte_array_from_slice(&bytes)
            .map(|a| a.into_raw())
            .unwrap_or_else(|_| std::ptr::null_mut()),
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn decompress_to_u64_bytes(
    env: &mut JNIEnv,
    flux_data: &JByteArray,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let bytes = env.convert_byte_array(flux_data)?;
    let data: Vec<u8> = bytes.iter().map(|&b| b as u8).collect();

    use loom::decompressors::block_reader::decompress_block;
    let (values, _) = decompress_block(&data)?;

    // Encode as flat little-endian u64 bytes (low 64 bits of each u128).
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&(v as u64).to_le_bytes());
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// decompressU128 — returns long[][] (hi, lo pairs)
// ─────────────────────────────────────────────────────────────────────────────

/// JNI: `long[][] FluxNative.decompressU128(byte[] fluxData)`
///
/// Decompresses a flux block containing u128 values and returns them as a
/// `long[2][]` array where `result[0]` is the high words and `result[1]` is
/// the low words.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_decompressU128(
    mut env: JNIEnv,
    _class: JClass,
    flux_data: JByteArray,
) -> jobjectArray {
    let result = decompress_u128_pairs(&mut env, &flux_data);
    match result {
        Ok(arr) => arr,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn decompress_u128_pairs(
    env: &mut JNIEnv,
    flux_data: &JByteArray,
) -> Result<jobjectArray, Box<dyn std::error::Error>> {
    let bytes = env.convert_byte_array(flux_data)?;
    let data: Vec<u8> = bytes.iter().map(|&b| b as u8).collect();

    use loom::decompressors::block_reader::decompress_block;
    let (values, _) = decompress_block(&data)?;

    let count = values.len();

    // Split into (hi, lo) long arrays.
    let mut hi_vec = Vec::with_capacity(count);
    let mut lo_vec = Vec::with_capacity(count);
    for v in &values {
        let (hi, lo) = u128_to_jlong_pair(*v);
        hi_vec.push(hi);
        lo_vec.push(lo);
    }

    // Build long[2][] result.
    let long_class = env.find_class("[J")?; // [J = long[]
    let outer = env.new_object_array(2, long_class, JObject::null())?;

    let hi_arr = env.new_long_array(count as i32)?;
    env.set_long_array_region(&hi_arr, 0, &hi_vec)?;
    env.set_object_array_element(&outer, 0, &hi_arr)?;

    let lo_arr = env.new_long_array(count as i32)?;
    env.set_long_array_region(&lo_arr, 0, &lo_vec)?;
    env.set_object_array_element(&outer, 1, &lo_arr)?;

    Ok(outer.into_raw())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase G — Layer 2: multi-column Arrow IPC batch
// ─────────────────────────────────────────────────────────────────────────────

/// JNI: `byte[] FluxNative.compressTable(byte[] ipcBatch, String profile)`
///
/// Accepts an Arrow IPC *stream* payload (output of `ArrowStreamWriter` on
/// the Java side), decompresses it into a [`RecordBatch`], and returns the
/// FluxCompress-compressed `.flux` bytes.
///
/// `profile` controls the speed/ratio trade-off:
/// - `"speed"` (default) — no secondary codec
/// - `"balanced"` — LZ4 post-pass
/// - `"archive"` — Zstd post-pass (best ratio, higher CPU)
/// - `"brotli"` — Brotli for text columns, Zstd for numeric
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_compressTable(
    mut env: JNIEnv,
    _class: JClass,
    ipc_batch: JByteArray,
    profile: JString,
) -> jbyteArray {
    let result = compress_table_inner(&mut env, &ipc_batch, &profile);
    match result {
        Ok(bytes) => env
            .byte_array_from_slice(&bytes)
            .map(|a| a.into_raw())
            .unwrap_or_else(|_| std::ptr::null_mut()),
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn compress_table_inner(
    env: &mut JNIEnv,
    ipc_batch: &JByteArray,
    profile_str: &JString,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let ipc_bytes = env.convert_byte_array(ipc_batch)?;
    let profile_java: String = env.get_string(profile_str)?.into();
    let profile = parse_profile(&profile_java)?;

    let batch = ipc_bridge::batch_from_ipc(&ipc_bytes)?;
    let writer = FluxWriter::with_profile(profile);
    Ok(writer.compress(&batch)?)
}

/// JNI: `byte[] FluxNative.decompressTable(byte[] fluxData)`
///
/// Decompresses a `.flux` byte payload and returns the result as an
/// Arrow IPC *stream* that the Java side reads with `ArrowStreamReader`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_decompressTable(
    mut env: JNIEnv,
    _class: JClass,
    flux_data: JByteArray,
) -> jbyteArray {
    let result = decompress_table_inner(&mut env, &flux_data);
    match result {
        Ok(bytes) => env
            .byte_array_from_slice(&bytes)
            .map(|a| a.into_raw())
            .unwrap_or_else(|_| std::ptr::null_mut()),
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn decompress_table_inner(
    env: &mut JNIEnv,
    flux_data: &JByteArray,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let data = env.convert_byte_array(flux_data)?;
    let reader = FluxReader::default();
    let batch = reader.decompress_all(&data)?;
    Ok(ipc_bridge::batch_to_ipc(&batch)?)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase G — Layer 3: FluxTable handle API
// ─────────────────────────────────────────────────────────────────────────────

/// JNI: `long FluxNative.tableOpen(String path)`
///
/// Opens (or creates) a FluxTable at `path` and returns an opaque handle.
/// The handle is valid until `tableClose` is called.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_tableOpen(
    mut env: JNIEnv,
    _class: JClass,
    path: JString,
) -> jlong {
    let result = table_open_inner(&mut env, &path);
    match result {
        Ok(h) => h,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            0
        }
    }
}

fn table_open_inner(
    env: &mut JNIEnv,
    path: &JString,
) -> Result<jlong, Box<dyn std::error::Error>> {
    let path_str: String = env.get_string(path)?.into();
    let table = FluxTable::open(&path_str)?;
    Ok(table_registry::insert(table))
}

/// JNI: `long FluxNative.tableAppend(long handle, byte[] fluxData)`
///
/// Appends pre-compressed `.flux` bytes to the open table and returns the
/// new transaction log version number.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_tableAppend(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    flux_data: JByteArray,
) -> jlong {
    let result = table_append_inner(&mut env, handle, &flux_data);
    match result {
        Ok(v) => v as jlong,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            -1
        }
    }
}

fn table_append_inner(
    env: &mut JNIEnv,
    handle: jlong,
    flux_data: &JByteArray,
) -> Result<u64, Box<dyn std::error::Error>> {
    let data = env.convert_byte_array(flux_data)?;
    table_registry::with(handle, |table: &FluxTable| table.append(&data))
        .ok_or_else(|| invalid_handle_err(handle))?
        .map_err(Into::into)
}

/// JNI: `byte[] FluxNative.tableScan(long handle)`
///
/// Scans all live files in the open table and returns the result as a single
/// Arrow IPC *stream* payload.  If the table has no live files, returns an
/// empty byte array.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_tableScan(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jbyteArray {
    let result = table_scan_inner(&mut env, handle);
    match result {
        Ok(bytes) => {
            if bytes.is_empty() {
                // Return an empty byte[].
                env.new_byte_array(0)
                    .map(|a| a.into_raw())
                    .unwrap_or_else(|_| std::ptr::null_mut())
            } else {
                env.byte_array_from_slice(&bytes)
                    .map(|a| a.into_raw())
                    .unwrap_or_else(|_| std::ptr::null_mut())
            }
        }
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            std::ptr::null_mut()
        }
    }
}

fn table_scan_inner(
    _env: &mut JNIEnv,
    handle: jlong,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let result: Option<Result<Vec<_>, Box<dyn std::error::Error>>> =
        table_registry::with(handle, |table| scan_all_files(table));

    let batches = result.ok_or_else(|| invalid_handle_err(handle))??;
    ipc_bridge::batches_to_ipc(&batches)
}

/// Collect all live batches from a FluxTable.
///
/// Two paths:
/// * **Schema-aware**: when the table has an evolved schema (`current_schema_id`
///   is `Some`), use [`FluxScan`] so column projections, null-fills, and renames
///   are applied correctly.
/// * **Raw**: when no schema has been declared yet, fall back to direct
///   per-file decompression via [`FluxReader`].  Callers that need consistent
///   schema projection should call `tableEvolve` before `tableScan`.
fn scan_all_files(
    table: &FluxTable,
) -> Result<Vec<arrow_array::RecordBatch>, Box<dyn std::error::Error>> {
    let snap = table.snapshot()?;

    if snap.current_schema_id().is_some() {
        // Schema-aware path: FluxScan handles projection, null-fill, rename.
        let scan = table.scan()?;
        let mut batches = Vec::new();
        for item in scan {
            batches.push(item?);
        }
        return Ok(batches);
    }

    // No schema yet — raw decompression, one batch per file.
    let reader = FluxReader::default();
    let mut batches = Vec::new();
    for file_path in table.live_files()? {
        let data = std::fs::read(&file_path)?;
        let batch = reader.decompress_all(&data)?;
        batches.push(batch);
    }
    Ok(batches)
}

/// JNI: `long FluxNative.tableEvolve(long handle, String schemaJson)`
///
/// Evolves the table's logical schema.  `schemaJson` is the JSON
/// serialisation of a `TableSchema` (see the Rust crate docs for the
/// exact format).  Returns the transaction log version of the
/// schema-change entry.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_tableEvolve(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    schema_json: JString,
) -> jlong {
    let result = table_evolve_inner(&mut env, handle, &schema_json);
    match result {
        Ok(v) => v as jlong,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            -1
        }
    }
}

fn table_evolve_inner(
    env: &mut JNIEnv,
    handle: jlong,
    schema_json: &JString,
) -> Result<u64, Box<dyn std::error::Error>> {
    let json_str: String = env.get_string(schema_json)?.into();
    let schema: loom::txn::schema::TableSchema = serde_json::from_str(&json_str)?;
    table_registry::with(handle, |table: &FluxTable| table.evolve_schema(schema))
        .ok_or_else(|| invalid_handle_err(handle))?
        .map_err(Into::into)
}

/// JNI: `void FluxNative.tableClose(long handle)`
///
/// Releases the handle and drops the underlying `FluxTable`.  The handle is
/// invalid after this call; passing it to any other JNI function will throw
/// `IllegalArgumentException`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_tableClose(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if table_registry::remove(handle).is_none() {
        // Already closed or never existed — throw but let execution continue.
        let _ = env.throw_new(
            "java/lang/IllegalArgumentException",
            format!("tableClose: unknown handle {handle}"),
        );
    }
}

/// JNI: `long FluxNative.tableCurrentVersion(long handle)`
///
/// Returns the latest committed transaction log version number, or `-1` when
/// the table has no entries yet.
#[unsafe(no_mangle)]
pub extern "system" fn Java_io_fluxcompress_FluxNative_tableCurrentVersion(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jlong {
    let result: Option<Result<jlong, loom::error::FluxError>> =
        table_registry::with(handle, |table: &FluxTable| {
            table.next_version()
                .map(|v| if v == 0 { -1 } else { (v - 1) as jlong })
        });
    match result {
        Some(Ok(v)) => v,
        Some(Err(e)) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            -1
        }
        None => {
            let _ = env.throw_new(
                "java/lang/IllegalArgumentException",
                format!("tableCurrentVersion: unknown handle {handle}"),
            );
            -1
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a profile name string into a [`CompressionProfile`].
fn parse_profile(s: &str) -> Result<CompressionProfile, Box<dyn std::error::Error>> {
    match s.to_ascii_lowercase().as_str() {
        "speed"    | ""       => Ok(CompressionProfile::Speed),
        "balanced"             => Ok(CompressionProfile::Balanced),
        "archive"              => Ok(CompressionProfile::Archive),
        "brotli"               => Ok(CompressionProfile::Brotli),
        other => Err(format!(
            "unknown compression profile '{other}'; \
             expected one of: speed, balanced, archive, brotli"
        ).into()),
    }
}

/// Build a boxed error for an invalid / stale table handle.
#[inline]
fn invalid_handle_err(handle: jlong) -> Box<dyn std::error::Error> {
    format!("invalid or closed FluxTable handle: {handle}").into()
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase G non-JNI unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod phase_g_tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use arrow_array::RecordBatch;
    use tempfile::TempDir;

    use loom::{
        compressors::flux_writer::FluxWriter,
        decompressors::flux_reader::FluxReader,
        traits::{LoomCompressor, LoomDecompressor},
        txn::{schema::{SchemaField, TableSchema}, FluxTable},
        dtype::FluxDType,
        CompressionProfile,
    };

    use crate::ipc_bridge::{batch_from_ipc, batch_to_ipc, batches_to_ipc};
    use crate::table_registry;

    fn make_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id",    DataType::UInt64, false),
            Field::new("score", DataType::Int32,  false),
            Field::new("label", DataType::Utf8,   true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
                Arc::new(Int32Array::from_iter_values((0..rows as i32).map(|x| x * 10))),
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|i| format!("row_{i}")),
                )),
            ],
        )
        .unwrap()
    }

    // ── Layer 2 tests ────────────────────────────────────────────────────────

    #[test]
    fn compress_decompress_table_round_trip() {
        let batch   = make_batch(100);
        let ipc_in  = batch_to_ipc(&batch).unwrap();

        // Simulate compressTable JNI call (sans JNI machinery).
        let flux = {
            let decoded_batch = batch_from_ipc(&ipc_in).unwrap();
            FluxWriter::with_profile(CompressionProfile::Speed)
                .compress(&decoded_batch)
                .unwrap()
        };

        // Simulate decompressTable JNI call.
        let ipc_out = {
            let reader  = FluxReader::default();
            let decoded = reader.decompress_all(&flux).unwrap();
            batch_to_ipc(&decoded).unwrap()
        };

        let result = batch_from_ipc(&ipc_out).unwrap();
        assert_eq!(result.num_rows(),    batch.num_rows());
        assert_eq!(result.num_columns(), batch.num_columns());

        let ids = result.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(ids.value(0), 0);
        assert_eq!(ids.value(99), 99);
    }

    #[test]
    fn compress_table_brotli_profile() {
        let batch  = make_batch(50);
        let ipc_in = batch_to_ipc(&batch).unwrap();
        let decoded_batch = batch_from_ipc(&ipc_in).unwrap();
        let flux = FluxWriter::with_profile(CompressionProfile::Brotli)
            .compress(&decoded_batch)
            .unwrap();
        let result = FluxReader::default().decompress_all(&flux).unwrap();
        assert_eq!(result.num_rows(), 50);
    }

    // ── Layer 3 tests ────────────────────────────────────────────────────────

    #[test]
    fn table_open_append_scan_close_round_trip() {
        let dir   = TempDir::new().unwrap();
        let table = FluxTable::open(dir.path().join("test.fluxtable")).unwrap();
        let handle = table_registry::insert(table);

        // Compress a batch and append it.
        let batch = make_batch(64);
        let flux = FluxWriter::new().compress(&batch).unwrap();

        let version = table_registry::with(handle, |t| t.append(&flux))
            .unwrap()
            .unwrap();
        assert_eq!(version, 0);

        // Scan should return 64 rows (via the scan_all_files fallback path
        // since no schema has been declared on this table).
        let batches = table_registry::with(handle, |t| {
            crate::scan_all_files(t)
        })
        .unwrap()
        .unwrap();
        let ipc = batches_to_ipc(&batches).unwrap();
        let result = batch_from_ipc(&ipc).unwrap();
        assert_eq!(result.num_rows(), 64);

        // tableClose: removes handle.
        let removed = table_registry::remove(handle);
        assert!(removed.is_some());
        assert!(!table_registry::contains(handle));
    }

    #[test]
    fn table_evolve_then_scan_projects_correctly() {
        let dir    = TempDir::new().unwrap();
        let table  = FluxTable::open(dir.path().join("evo.fluxtable")).unwrap();
        let handle = table_registry::insert(table);

        // v0 schema: {id: u64}.
        let schema_v0 = TableSchema::new(vec![
            SchemaField::new(1, "id", FluxDType::UInt64).with_nullable(false),
        ]);
        let schema_json = serde_json::to_string(&schema_v0).unwrap();
        let v0: loom::txn::schema::TableSchema = serde_json::from_str(&schema_json).unwrap();
        let _ = table_registry::with(handle, |t| t.evolve_schema(v0)).unwrap().unwrap();

        // Append a batch.
        let batch_v0 = {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::UInt64, false),
            ]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(UInt64Array::from(vec![10u64, 20, 30]))],
            ).unwrap()
        };
        let flux_v0 = FluxWriter::new().compress(&batch_v0).unwrap();
        table_registry::with(handle, |t| t.append(&flux_v0)).unwrap().unwrap();

        // Evolve to {id, label}.
        let schema_v1 = TableSchema::new(vec![
            SchemaField::new(1, "id",    FluxDType::UInt64).with_nullable(false),
            SchemaField::new(2, "label", FluxDType::Utf8),
        ]);
        let _ = table_registry::with(handle, |t| t.evolve_schema(schema_v1)).unwrap().unwrap();

        // Scan — old file should be null-filled for `label`.
        let batches: Vec<RecordBatch> = table_registry::with(handle, |t| {
            t.scan().unwrap().map(|r| r.unwrap()).collect::<Vec<_>>()
        }).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 2);
        assert_eq!(batches[0].column(1).null_count(), 3); // null-filled

        let _ = table_registry::remove(handle);
    }

    #[test]
    fn invalid_handle_produces_none() {
        let bad: i64 = 0xDEAD_BEEF;
        assert!(table_registry::with(bad, |_| ()).is_none());
        assert!(table_registry::remove(bad).is_none());
    }
}
