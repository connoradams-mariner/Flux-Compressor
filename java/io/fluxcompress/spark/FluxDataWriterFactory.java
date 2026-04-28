// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.types.StructType;

/**
 * Per-partition data-writer factory.
 *
 * <p>Spark serialises one instance to every executor; each executor calls
 * {@link #createWriter(int, long)} to obtain a {@link FluxDataWriter} that
 * writes a single partition's rows to a {@code *.flux.tmp} file under the
 * table path.
 */
public final class FluxDataWriterFactory implements DataWriterFactory, Serializable {

    private static final long serialVersionUID = 1L;

    private final String tablePath;
    private final StructType schema;
    private final Map<String, String> options;

    public FluxDataWriterFactory(String tablePath, StructType schema, Map<String, String> options) {
        this.tablePath = tablePath;
        this.schema    = schema;
        this.options   = options == null ? new HashMap<>() : new HashMap<>(options);
    }

    @Override
    public DataWriter<InternalRow> createWriter(int partitionId, long taskId) {
        return new FluxDataWriter(tablePath, schema, options, partitionId, taskId);
    }
}
