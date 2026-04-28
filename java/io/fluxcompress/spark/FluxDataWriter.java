// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.fluxcompress.FluxNative;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

/**
 * Per-partition writer that produces a single {@code *.flux.tmp} file.
 *
 * <p>For the v0 reference connector this writer accepts a single numeric
 * column at a time (the {@link FluxNative} JNI primitives operate on one
 * {@code u64} stream per call). Multi-column tables can be written by
 * repeating the call per column or by adopting the recommended hybrid
 * pattern of storing flux blobs inside a Delta table.
 */
public final class FluxDataWriter implements DataWriter<InternalRow> {

    private final String tablePath;
    private final StructType schema;
    private final int partitionId;
    private final long taskId;
    private final List<Long> values = new ArrayList<>(64 * 1024);
    private final List<String> tempFiles = new ArrayList<>();

    public FluxDataWriter(
        String tablePath, StructType schema,
        Map<String, String> options,
        int partitionId, long taskId
    ) {
        this.tablePath = tablePath;
        this.schema = schema;
        this.partitionId = partitionId;
        this.taskId = taskId;

        if (schema.fields().length != 1) {
            throw new IllegalArgumentException(
                "FluxDataWriter v0 supports exactly one numeric column; got "
                + schema.fields().length + " columns. "
                + "For multi-column tables use the hybrid pattern: store flux "
                + "blobs as a BinaryType column inside a Delta table."
            );
        }
    }

    @Override
    public void write(InternalRow row) {
        DataType dt = schema.fields()[0].dataType();
        if (row.isNullAt(0)) {
            values.add(0L);             // placeholder; null-bitmap support is on the v0.4 roadmap
        } else if (DataTypes.LongType.equals(dt) || DataTypes.IntegerType.equals(dt)) {
            values.add(row.getLong(0));
        } else if (DataTypes.DoubleType.equals(dt)) {
            values.add(Double.doubleToRawLongBits(row.getDouble(0)));
        } else if (DataTypes.FloatType.equals(dt)) {
            values.add((long) Float.floatToRawIntBits(row.getFloat(0)) & 0xFFFFFFFFL);
        } else {
            throw new UnsupportedOperationException(
                "FluxDataWriter v0 only supports Long/Integer/Double/Float; got " + dt
            );
        }
    }

    @Override
    public WriterCommitMessage commit() throws IOException {
        if (values.isEmpty()) {
            return new FluxWriterCommitMessage(tempFiles, 0);
        }

        // Pack the column into a direct ByteBuffer for the zero-copy JNI path.
        int n = values.size();
        ByteBuffer buf = FluxNative.allocateU64Buffer(n);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        for (Long v : values) {
            buf.putLong(v);
        }
        buf.rewind();

        byte[] flux = FluxNative.compress(buf, n);

        Path dir = Paths.get(tablePath);
        Files.createDirectories(dir);
        String name = String.format("part-%05d-%d.flux.tmp", partitionId, taskId);
        Path tmp = dir.resolve(name);
        Files.write(tmp, flux);

        tempFiles.add(tmp.toString());
        return new FluxWriterCommitMessage(tempFiles, n);
    }

    @Override
    public void abort() {
        for (String tmp : tempFiles) {
            try { Files.deleteIfExists(Paths.get(tmp)); } catch (IOException ignore) {}
        }
        values.clear();
    }

    @Override
    public void close() {
        values.clear();
    }
}
