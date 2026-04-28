// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * JNI bridge to the FluxCompress Rust core.
 *
 * <h2>Loading the native library</h2>
 * <pre>
 *   # Build the cdylib first:
 *   cargo build --release -p flux-jni-bridge
 *
 *   # Then add to your JVM flags:
 *   -Djava.library.path=target/release
 * </pre>
 *
 * <h2>u128 Dual-Register Convention</h2>
 * The JVM has no native {@code u128}.  FluxCompress represents 128-bit values
 * as a pair of {@code long} values: {@code hi} (bits 127..64) and {@code lo}
 * (bits 63..0).  Both are treated as unsigned — the sign bit is a data bit.
 *
 * <pre>
 *   u128 = (Long.toUnsignedString(hi) << 64) | Long.toUnsignedString(lo)
 * </pre>
 */
public final class FluxNative {

    static {
        // Route library loading through the Scala-side bootstrapper so
        // the bundled cdylib inside the connector JAR is extracted and
        // registered without requiring the user to configure
        // `java.library.path`. Falls back to the legacy
        // `System.loadLibrary` behaviour on any failure so pre-0.5
        // installs that still carry `-Djava.library.path=…` keep
        // working.
        try {
            com.datamariners.fluxcompress.FluxNativeLoader$.MODULE$.ensureLoaded();
        } catch (Throwable t) {
            // Final fallback: classic `java.library.path`.
            System.loadLibrary("flux_jni");
        }
    }

    // ── Prevent instantiation ───────────────────────────────────────────────
    private FluxNative() {}

    // ── Core API ─────────────────────────────────────────────────────────────

    /**
     * Compress {@code valueCount} little-endian {@code u64} values from a
     * <em>direct</em> {@link ByteBuffer} into FluxCompress bytes.
     *
     * <p>The buffer must have at least {@code valueCount * 8} bytes remaining.
     * This call is <strong>zero-copy</strong>: Rust reads directly from the
     * JVM's off-heap memory without allocating an intermediate buffer.
     *
     * @param  data       a {@code DirectByteBuffer} containing packed u64 values
     * @param  valueCount number of u64 values to compress
     * @return            compressed {@code .flux} block bytes
     */
    public static native byte[] compress(ByteBuffer data, int valueCount);

    /**
     * Decompress a {@code .flux} block and return the values as a flat byte
     * array of little-endian {@code u64} values (8 bytes each).
     *
     * @param  fluxData compressed block bytes
     * @return          flat {@code byte[]} of {@code count * 8} bytes
     */
    public static native byte[] decompress(byte[] fluxData);

    /**
     * Compress a column of {@code u128} values supplied as two parallel
     * {@code long[]} arrays.
     *
     * <pre>
     *   hi[i] = bits 127..64 of value i
     *   lo[i] = bits  63..0  of value i
     * </pre>
     *
     * @param  hi high 64 bits of each u128 value
     * @param  lo low  64 bits of each u128 value
     * @return   compressed {@code .flux} block bytes
     */
    public static native byte[] compressU128(long[] hi, long[] lo);

    /**
     * Decompress a {@code .flux} block that contains u128 values.
     *
     * <p>Returns {@code long[2][n]} where:
     * <ul>
     *   <li>{@code result[0][i]} = high 64 bits of value {@code i}</li>
     *   <li>{@code result[1][i]} = low  64 bits of value {@code i}</li>
     * </ul>
     *
     * @param  fluxData compressed block bytes
     * @return          {@code long[2][n]} dual-register u128 representation
     */
    public static native long[][] decompressU128(byte[] fluxData);

    // ── Layer 2: multi-column Arrow IPC batch ──────────────────────────────────

    /**
     * Compress an Arrow IPC stream payload (output of
     * {@code ArrowStreamWriter}) into {@code .flux} bytes.
     *
     * @param  ipcBatch Arrow IPC stream bytes containing a single record batch.
     * @param  profile  {@code "speed"} | {@code "balanced"} | {@code "archive"} | {@code "brotli"}.
     * @return          Compressed {@code .flux} bytes.
     */
    public static native byte[] compressTable(byte[] ipcBatch, String profile);

    /**
     * Decompress {@code .flux} bytes back to an Arrow IPC stream payload
     * that the Java side reads with {@code ArrowStreamReader}.
     *
     * @param  fluxData Compressed {@code .flux} bytes.
     * @return          Arrow IPC stream bytes.
     */
    public static native byte[] decompressTable(byte[] fluxData);

    // ── Layer 3: FluxTable handle API ───────────────────────────────────────────

    /**
     * Open (or create) a {@code FluxTable} at the given filesystem path.
     *
     * @param  path Directory path for the {@code .fluxtable/} directory.
     * @return      Opaque handle (&gt; 0). Caller must call {@link #tableClose(long)}.
     */
    public static native long tableOpen(String path);

    /**
     * Append pre-compressed {@code .flux} bytes to an open table.
     *
     * @param  handle   Handle returned by {@link #tableOpen(String)}.
     * @param  fluxData Compressed {@code .flux} bytes (typically from {@link #compressTable}).
     * @return          Transaction log version of the new append.
     */
    public static native long tableAppend(long handle, byte[] fluxData);

    /**
     * Scan all live files in the open table and return the result as a
     * concatenated Arrow IPC stream.
     *
     * @param  handle Handle returned by {@link #tableOpen(String)}.
     * @return        Arrow IPC stream bytes, or empty array for empty tables.
     */
    public static native byte[] tableScan(long handle);

    /**
     * Evolve the table's logical schema.
     *
     * @param  handle     Handle returned by {@link #tableOpen(String)}.
     * @param  schemaJson JSON serialisation of a {@code TableSchema}.
     * @return            Transaction log version of the schema-change entry.
     */
    public static native long tableEvolve(long handle, String schemaJson);

    /**
     * Close and release a table handle. After this call the handle is
     * invalid; further use will throw.
     *
     * @param handle Handle returned by {@link #tableOpen(String)}.
     */
    public static native void tableClose(long handle);

    /**
     * Return the current (latest) committed version of an open table,
     * or {@code -1} if the table has no committed transactions yet.
     *
     * @param  handle Handle returned by {@link #tableOpen(String)}.
     * @return        Latest committed version number, or {@code -1}.
     */
    public static native long tableCurrentVersion(long handle);

    // ── Convenience helpers ─────────────────────────────────────────────────

    /**
     * Allocate a direct {@link ByteBuffer} suitable for zero-copy transfer
     * of {@code count} u64 values to Rust.
     */
    public static ByteBuffer allocateU64Buffer(int count) {
        return ByteBuffer
            .allocateDirect(count * Long.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Reconstruct a {@link java.math.BigInteger} from a dual-register u128
     * (hi, lo) pair.  Useful for display / arithmetic on the Java side.
     */
    public static java.math.BigInteger toUnsignedBigInteger(long hi, long lo) {
        java.math.BigInteger hiBI = java.math.BigInteger.valueOf(hi >>> 1)
            .shiftLeft(1)
            .add(java.math.BigInteger.valueOf(hi & 1));
        java.math.BigInteger loBI = java.math.BigInteger.valueOf(lo >>> 1)
            .shiftLeft(1)
            .add(java.math.BigInteger.valueOf(lo & 1));
        return hiBI.shiftLeft(64).add(loBI);
    }
}
