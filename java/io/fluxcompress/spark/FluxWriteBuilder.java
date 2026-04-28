// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.util.Map;

import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.SupportsOverwriteV2;
import org.apache.spark.sql.connector.write.SupportsTruncate;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriteBuilder;

/**
 * V2 {@link WriteBuilder} for the {@code flux} datasource.
 *
 * <h2>Option A — write-side interfaces</h2>
 * Implements both:
 * <ul>
 *   <li>{@link SupportsTruncate}      — the V2 analyzer calls
 *       {@link #truncate()} when the user passes
 *       {@code .mode("overwrite")} without a filter expression. Without
 *       this interface Spark refuses overwrite mode and surfaces
 *       {@code UNSUPPORTED_FEATURE.TABLE_OPERATION}.</li>
 *   <li>{@link SupportsOverwriteV2}   — handles
 *       {@code .mode("overwrite").option("replaceWhere", "...")} and
 *       {@code DELETE FROM ... WHERE ...} via a filter expression.</li>
 * </ul>
 *
 * <p>The chosen mode is captured here and forwarded to {@link FluxWrite}
 * via the constructor so the per-partition writer can decide whether to
 * delete or augment existing data files at commit time.
 */
public final class FluxWriteBuilder implements WriteBuilder, SupportsTruncate, SupportsOverwriteV2 {

    private final LogicalWriteInfo info;
    private final Map<String, String> options;

    /** Tracks which write semantics the analyzer has requested. */
    private FluxWriteMode mode = FluxWriteMode.APPEND;

    /** Optional filter for {@link SupportsOverwriteV2}. */
    private Predicate[] overwriteFilters = new Predicate[0];

    public FluxWriteBuilder(LogicalWriteInfo info, Map<String, String> options) {
        this.info = info;
        this.options = options;
    }

    /**
     * Called by Spark when the user invokes {@code mode("overwrite")}
     * without a {@code replaceWhere} expression. Equivalent to
     * "delete every existing data file, then append the new data".
     */
    @Override
    public WriteBuilder truncate() {                  // ◄ Option A: SupportsTruncate
        this.mode = FluxWriteMode.TRUNCATE;
        return this;
    }

    /**
     * Called by Spark for predicate-based overwrites
     * ({@code DELETE FROM t WHERE ...} or
     * {@code .option("replaceWhere", "...").mode("overwrite")}).
     */
    @Override
    public WriteBuilder overwrite(Predicate[] filters) {  // ◄ Option A: SupportsOverwriteV2
        this.mode = FluxWriteMode.OVERWRITE_BY_FILTER;
        this.overwriteFilters = filters;
        return this;
    }

    @Override
    public Write build() {
        return new FluxWrite(info, options, mode, overwriteFilters);
    }
}
