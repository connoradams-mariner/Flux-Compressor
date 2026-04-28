// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark.test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.fluxcompress.spark.FluxBatchWrite;
import io.fluxcompress.spark.FluxCatalog;
import io.fluxcompress.spark.FluxTable;
import io.fluxcompress.spark.FluxTableProvider;
import io.fluxcompress.spark.FluxWriteBuilder;
import io.fluxcompress.spark.FluxWriteMode;
import io.fluxcompress.spark.FluxWriterCommitMessage;

import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.analysis.TableAlreadyExistsException;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pure-Java JUnit 5 tests for the V2 connector wiring.
 *
 * <p>These tests deliberately do not spin up a SparkSession. They cover
 * the parts that are pure data-class invariants (capability matrix,
 * truncate flag propagation, catalog directory plumbing). The
 * integration smoke test lives in
 * {@code python/tests/spark_uc_smoke_test.py} and runs on the cluster
 * with a real session.
 *
 * <h2>Run</h2>
 * <pre>
 *   javac -d build -cp "$SPARK_HOME/jars/*:junit-jupiter-api.jar" \
 *     $(find java -name '*.java')
 *   java -jar junit-platform-console-standalone.jar \
 *     --class-path build --scan-class-path
 * </pre>
 */
public class FluxConnectorTest {

    // ── Option B: TableCapability matrix ──────────────────────────────────

    @Test
    void table_advertises_truncate_capability() {
        FluxTable t = newTable();
        Set<TableCapability> caps = t.capabilities();
        assertTrue(caps.contains(TableCapability.BATCH_WRITE),
                   "BATCH_WRITE missing — append/default save would fail");
        assertTrue(caps.contains(TableCapability.TRUNCATE),
                   "TRUNCATE missing — mode(\"overwrite\") would fail");
        assertTrue(caps.contains(TableCapability.OVERWRITE_BY_FILTER),
                   "OVERWRITE_BY_FILTER missing — replaceWhere / DELETE would fail");
    }

    // ── Option A: SupportsTruncate threading ──────────────────────────────

    @Test
    void writebuilder_truncate_flag_reaches_write() {
        FluxWriteBuilder builder = newWriteBuilder();
        WriteBuilder afterTruncate = builder.truncate();
        assertSame(builder, afterTruncate, "truncate() must return the same builder for chaining");
        Write write = builder.build();
        assertNotNull(write);
        // The mode is captured inside the builder; if Spark forgot to call
        // truncate(), we'd default to APPEND and overwrite would silently
        // turn into append. The build() chain has produced a Write instance,
        // confirming the wiring runs end-to-end.
        assertTrue(write.description().contains("TRUNCATE"),
                   "Write description should reflect TRUNCATE mode, got: " + write.description());
    }

    @Test
    void writebuilder_default_mode_is_append() {
        FluxWriteBuilder builder = newWriteBuilder();
        Write write = builder.build();
        assertTrue(write.description().contains("APPEND"),
                   "Default mode should be APPEND, got: " + write.description());
    }

    // ── SupportsCatalogOptions ─────────────────────────────────────────────

    @Test
    void provider_extracts_catalog_and_identifier() {
        FluxTableProvider provider = new FluxTableProvider();
        Map<String, String> opts = new HashMap<>();
        opts.put("path", "/Volumes/main/default/myvol/orders");
        opts.put("namespace", "sales");
        opts.put("table", "orders");
        CaseInsensitiveStringMap m = new CaseInsensitiveStringMap(opts);

        assertEquals("flux", provider.extractCatalog(m));
        Identifier id = provider.extractIdentifier(m);
        assertArrayEquals(new String[] { "sales" }, id.namespace());
        assertEquals("orders", id.name());
    }

    // ── FluxCatalog table lifecycle ────────────────────────────────────────

    @Test
    void catalog_creates_lists_drops_table(@TempDir Path tempDir) throws Exception {
        FluxCatalog catalog = newCatalog(tempDir);

        // 1. Empty namespace lists no tables.
        assertEquals(0, catalog.listTables(new String[] { "default" }).length);

        // 2. createTable persists a directory + schema sidecar.
        Identifier ident = Identifier.of(new String[] { "default" }, "events");
        StructType schema = new StructType()
            .add("id", DataTypes.LongType, false)
            .add("ts", DataTypes.LongType, false);
        Table created = catalog.createTable(ident, schema, new Transform[0], Map.of());
        assertNotNull(created);
        assertTrue(Files.isDirectory(tempDir.resolve("default").resolve("events")),
                   "createTable should produce a directory");
        assertTrue(Files.exists(tempDir.resolve("default").resolve("events").resolve("_schema.json")),
                   "createTable should write the schema sidecar");

        // 3. listTables now reports it.
        Identifier[] tables = catalog.listTables(new String[] { "default" });
        assertEquals(1, tables.length);
        assertEquals("events", tables[0].name());

        // 4. loadTable reconstructs the schema.
        Table loaded = catalog.loadTable(ident);
        assertEquals(schema, loaded.schema());

        // 5. Creating again throws TableAlreadyExistsException.
        assertThrows(TableAlreadyExistsException.class, () ->
            catalog.createTable(ident, schema, new Transform[0], Map.of())
        );

        // 6. dropTable removes the directory.
        assertTrue(catalog.dropTable(ident));
        assertFalse(Files.exists(tempDir.resolve("default").resolve("events")));
        assertThrows(NoSuchTableException.class, () -> catalog.loadTable(ident));
    }

    @Test
    void catalog_rejects_init_without_path() {
        FluxCatalog catalog = new FluxCatalog();
        IllegalArgumentException ex = assertThrows(
            IllegalArgumentException.class,
            () -> catalog.initialize("flux", new CaseInsensitiveStringMap(Map.of()))
        );
        assertTrue(ex.getMessage().contains("path must be set"));
    }

    // ── FluxBatchWrite truncate semantics ─────────────────────────────────

    @Test
    void truncate_deletes_existing_flux_files(@TempDir Path tempDir) throws Exception {
        // Seed a couple of pre-existing data files.
        Files.writeString(tempDir.resolve("part-00000-existing.flux"), "stale");
        Files.writeString(tempDir.resolve("part-00001-existing.flux"), "stale");
        // And a non-flux file we should never delete.
        Files.writeString(tempDir.resolve("README.md"),                  "keep me");

        // Stage one new tmp file as if a task had just produced it.
        Path tmp = tempDir.resolve("part-00002.flux.tmp");
        Files.writeString(tmp, "fresh");

        FluxBatchWrite write = new FluxBatchWrite(
            /* info */ null,
            Map.of("path", tempDir.toString()),
            FluxWriteMode.TRUNCATE,
            new org.apache.spark.sql.connector.expressions.filter.Predicate[0]
        );
        FluxWriterCommitMessage msg = new FluxWriterCommitMessage(List.of(tmp.toString()), 1);
        write.commit(new org.apache.spark.sql.connector.write.WriterCommitMessage[] { msg });

        // Old .flux files are gone, README is still there, tmp is promoted.
        assertFalse(Files.exists(tempDir.resolve("part-00000-existing.flux")));
        assertFalse(Files.exists(tempDir.resolve("part-00001-existing.flux")));
        assertTrue(Files.exists(tempDir.resolve("README.md")));
        assertTrue(Files.exists(tempDir.resolve("part-00002.flux")));
        assertFalse(Files.exists(tmp));
    }

    @Test
    void append_keeps_existing_flux_files(@TempDir Path tempDir) throws Exception {
        Files.writeString(tempDir.resolve("part-00000-existing.flux"), "stale");
        Path tmp = tempDir.resolve("part-00001.flux.tmp");
        Files.writeString(tmp, "fresh");

        FluxBatchWrite write = new FluxBatchWrite(
            null,
            Map.of("path", tempDir.toString()),
            FluxWriteMode.APPEND,
            new org.apache.spark.sql.connector.expressions.filter.Predicate[0]
        );
        FluxWriterCommitMessage msg = new FluxWriterCommitMessage(List.of(tmp.toString()), 1);
        write.commit(new org.apache.spark.sql.connector.write.WriterCommitMessage[] { msg });

        // Old file survives; new file is promoted.
        assertTrue(Files.exists(tempDir.resolve("part-00000-existing.flux")));
        assertTrue(Files.exists(tempDir.resolve("part-00001.flux")));
        assertFalse(Files.exists(tmp));
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static FluxTable newTable() {
        return new FluxTable(new StructType().add("v", DataTypes.LongType), new Transform[0], Map.of());
    }

    private static FluxWriteBuilder newWriteBuilder() {
        Map<String, String> opts = new HashMap<>();
        opts.put("path", "/tmp/flux-builder-test");
        return new FluxWriteBuilder(/* info */ null, opts);
    }

    private static FluxCatalog newCatalog(Path tempDir) {
        FluxCatalog catalog = new FluxCatalog();
        catalog.initialize("flux", new CaseInsensitiveStringMap(Map.of("path", tempDir.toString())));
        return catalog;
    }
}
