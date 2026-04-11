// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # JNI Bridge (Sprint 5)
//!
//! Exposes FluxCompress to the Spark/JVM ecosystem via JNI.
//!
//! ## u128 "Dual-Register" Strategy
//! The JVM has no native `u128` type.  We represent each 128-bit value as a
//! `long[2]` array `{hi, lo}` on the Java side.  The JNI bridge reconstructs
//! the full `u128` in Rust before passing it to the Outlier Map.
//!
//! ## Zero-Copy Off-Heap Handshake
//! Large data transfers use `java.nio.ByteBuffer.allocateDirect()` on the Java
//! side.  The JNI function receives a `jobject` (the `DirectByteBuffer`),
//! obtains a raw pointer via `GetDirectBufferAddress`, and wraps it in a Rust
//! `&[u8]` slice — **without copying a single byte between JVM heap and Rust
//! heap**.
//!
//! ## Java API (generated side)
//! ```java
//! package io.fluxcompress;
//!
//! public class FluxNative {
//!     static { System.loadLibrary("flux_jni"); }
//!
//!     // Compress a direct ByteBuffer of u64 values → compressed flux bytes.
//!     public static native byte[] compress(ByteBuffer data, int valueCount);
//!
//!     // Decompress flux bytes → direct ByteBuffer of u64 values.
//!     public static native ByteBuffer decompress(byte[] fluxData);
//!
//!     // Compress a u128 column supplied as two parallel long[] arrays.
//!     public static native byte[] compressU128(long[] hi, long[] lo);
//!
//!     // Decompress a flux block that contains u128 values → long[2][] (hi, lo pairs).
//!     public static native long[][] decompressU128(byte[] fluxData);
//! }
//! ```

#![allow(non_snake_case)]

use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JLongArray, JObject, JObjectArray};
use jni::sys::{jbyteArray, jlongArray, jobjectArray, jint};

use loom::{
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    traits::{LoomCompressor, LoomDecompressor, Predicate},
};

mod u128_bridge;
pub use u128_bridge::{jlong_pair_to_u128, u128_to_jlong_pair};

// ─────────────────────────────────────────────────────────────────────────────
// compress — u64 column (DirectByteBuffer)
// ─────────────────────────────────────────────────────────────────────────────

/// JNI: `byte[] FluxNative.compress(ByteBuffer data, int valueCount)`
///
/// `data` must be a `DirectByteBuffer` holding `valueCount` little-endian u64
/// values (8 bytes each).  Returns the compressed `.flux` bytes.
#[no_mangle]
pub extern "system" fn Java_io_fluxcompress_FluxNative_compress(
    mut env: JNIEnv,
    _class: JClass,
    data: JObject,
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
    buffer: &JObject,
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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
