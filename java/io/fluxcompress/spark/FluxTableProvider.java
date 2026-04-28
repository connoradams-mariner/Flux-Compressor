// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.util.Map;

import org.apache.spark.sql.connector.catalog.SupportsCatalogOptions;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Spark V2 entry point for the {@code "flux"} datasource short name.
 *
 * <p>Registered via the {@link DataSourceRegister} ServiceLoader contract
 * (see {@code META-INF/services/org.apache.spark.sql.sources.DataSourceRegister}).
 *
 * <p>Hands the schema and write options off to {@link FluxTable}, which
 * advertises {@link org.apache.spark.sql.connector.catalog.TableCapability#TRUNCATE}
 * so {@code .mode("overwrite")} is accepted by the V2 analyzer.
 *
 * <p>Implements {@link SupportsCatalogOptions} so the provider can be
 * registered with Unity Catalog via
 * {@code spark.sql.catalog.flux=io.fluxcompress.spark.FluxCatalog}. When
 * UC is active, write paths flow through {@link FluxCatalog} (a
 * {@link org.apache.spark.sql.connector.catalog.TableCatalog}) instead
 * of being treated as raw filesystem paths — which is what UC requires
 * to permit a non-Delta data source.
 */
public final class FluxTableProvider
    implements TableProvider, DataSourceRegister, SupportsCatalogOptions {

    /** The short name used in {@code spark.read/write.format("flux")}. */
    @Override
    public String shortName() {
        return "flux";
    }

    /**
     * Schema is required up-front for V2 datasources. Callers either supply
     * a schema explicitly or rely on Spark to infer it from the DataFrame
     * being written.
     */
    @Override
    public StructType inferSchema(CaseInsensitiveStringMap options) {
        // For writes, Spark passes the DataFrame schema via getTable(). For
        // reads, callers must currently supply schema(...) on the DataFrameReader
        // until the read path is fully implemented.
        return new StructType();
    }

    @Override
    public Table getTable(StructType schema, Transform[] partitioning, Map<String, String> properties) {
        return new FluxTable(schema, partitioning, properties);
    }

    /**
     * Tell Spark that we already know the schema (passed to {@link #getTable})
     * and it doesn't need to call {@link #inferSchema} again.
     */
    @Override
    public boolean supportsExternalMetadata() {
        return true;
    }

    // ── SupportsCatalogOptions ───────────────────────────────────────

    /**
     * Identify the table from {@code .option("path", ...)} (or the implicit
     * path supplied to {@code .save("...")}). Catalog routing uses this to
     * decide which {@link FluxTable} a given DataFrame call refers to.
     */
    @Override
    public Identifier extractIdentifier(CaseInsensitiveStringMap options) {
        String table = options.getOrDefault("table",
                       options.getOrDefault("path", "default"));
        String namespace = options.getOrDefault("namespace", "default");
        return Identifier.of(new String[] { namespace }, table);
    }

    /**
     * Catalog name to look up via {@code spark.sql.catalog.<name>=...}.
     * Defaults to {@code flux} so admins can simply set
     * {@code spark.sql.catalog.flux=io.fluxcompress.spark.FluxCatalog}.
     */
    @Override
    public String extractCatalog(CaseInsensitiveStringMap options) {
        return options.getOrDefault("catalog", "flux");
    }
}
