// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.spark.sql.connector.catalog.SupportsWrite;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.StructType;

/**
 * V2 Table implementation for the {@code flux} datasource.
 *
 * <h2>Option B — Capability declarations</h2>
 * The capability set is what makes the V2 analyzer accept each save mode:
 * <ul>
 *   <li>{@link TableCapability#BATCH_WRITE}        — required for any
 *       batch-mode write (covers default and {@code mode("append")}).</li>
 *   <li>{@link TableCapability#TRUNCATE}           — required for
 *       {@code mode("overwrite")}; pairs with {@link
 *       org.apache.spark.sql.connector.write.SupportsTruncate} on the
 *       {@link FluxWriteBuilder}.</li>
 *   <li>{@link TableCapability#OVERWRITE_BY_FILTER} — enables
 *       {@code DELETE FROM ... WHERE} and predicate-based
 *       {@link org.apache.spark.sql.connector.write.SupportsOverwriteV2}
 *       semantics.</li>
 *   <li>{@link TableCapability#ACCEPT_ANY_SCHEMA}  — let callers ship a
 *       DataFrame with any schema; the writer materialises it lossless.</li>
 * </ul>
 *
 * <p>Without {@code TRUNCATE} present here, Spark refuses
 * {@code mode("overwrite")} with the
 * {@code UNSUPPORTED_FEATURE.TABLE_OPERATION ... does not support truncate
 * in batch mode} error.
 */
public final class FluxTable implements SupportsWrite {

    private final StructType schema;
    private final Transform[] partitioning;
    private final Map<String, String> properties;

    public FluxTable(StructType schema, Transform[] partitioning, Map<String, String> properties) {
        this.schema = schema;
        this.partitioning = partitioning != null ? partitioning : new Transform[0];
        this.properties = properties != null ? new HashMap<>(properties) : new HashMap<>();
    }

    @Override
    public String name() {
        String path = properties.getOrDefault("path", "<no-path>");
        return "flux:" + path;
    }

    @Override
    public StructType schema() {
        return schema;
    }

    @Override
    public Transform[] partitioning() {
        return partitioning;
    }

    @Override
    public Map<String, String> properties() {
        return properties;
    }

    /**
     * Capability matrix that the V2 write planner consults before
     * dispatching {@code AppendData}, {@code OverwriteByExpression},
     * etc. Every mode the user might invoke from {@code .mode(...)} needs
     * its corresponding capability listed here.
     */
    @Override
    public Set<TableCapability> capabilities() {
        Set<TableCapability> caps = EnumSet.noneOf(TableCapability.class);
        caps.add(TableCapability.BATCH_WRITE);            // append / default save
        caps.add(TableCapability.TRUNCATE);               // mode("overwrite")  ◄ Option B
        caps.add(TableCapability.OVERWRITE_BY_FILTER);    // DELETE / SupportsOverwriteV2
        caps.add(TableCapability.ACCEPT_ANY_SCHEMA);      // tolerate user-supplied schemas
        return caps;
    }

    @Override
    public WriteBuilder newWriteBuilder(LogicalWriteInfo info) {
        return new FluxWriteBuilder(info, properties);
    }

    /**
     * Convenience: returns the lower-cased set of capability names so the
     * connector's logs / `DESCRIBE EXTENDED` output is greppable.
     */
    public Set<String> capabilityNames() {
        Set<String> out = new HashSet<>();
        for (TableCapability c : capabilities()) out.add(c.name());
        return out;
    }
}
