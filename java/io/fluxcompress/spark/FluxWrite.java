// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.util.Map;

import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.write.BatchWrite;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.Write;

/**
 * V2 {@link Write} for the {@code flux} datasource.
 *
 * <p>Holds the write mode chosen by {@link FluxWriteBuilder} and produces
 * a {@link FluxBatchWrite} that knows whether to truncate, replace by
 * filter, or simply append at commit time.
 */
public final class FluxWrite implements Write {

    private final LogicalWriteInfo info;
    private final Map<String, String> options;
    private final FluxWriteMode mode;
    private final Predicate[] overwriteFilters;

    public FluxWrite(
        LogicalWriteInfo info,
        Map<String, String> options,
        FluxWriteMode mode,
        Predicate[] overwriteFilters
    ) {
        this.info = info;
        this.options = options;
        this.mode = mode;
        this.overwriteFilters = overwriteFilters;
    }

    @Override
    public BatchWrite toBatch() {
        return new FluxBatchWrite(info, options, mode, overwriteFilters);
    }

    @Override
    public String description() {
        return "FluxWrite[mode=" + mode + ", filters=" + overwriteFilters.length + "]";
    }
}
