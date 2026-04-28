// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.spark.sql.catalyst.analysis.NoSuchNamespaceException;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.analysis.TableAlreadyExistsException;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.SupportsNamespaces;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCatalog;
import org.apache.spark.sql.connector.catalog.TableChange;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * V2 {@link TableCatalog} that backs SQL like
 * {@code CREATE TABLE flux.db.t (...) USING flux}.
 *
 * <p>Configure with:
 * <pre>
 *   spark.sql.catalog.flux        = io.fluxcompress.spark.FluxCatalog
 *   spark.sql.catalog.flux.path   = /Volumes/&lt;catalog&gt;/&lt;schema&gt;/&lt;volume&gt;
 * </pre>
 *
 * <p>The catalog uses the configured {@code path} (typically a UC Volume)
 * as a directory root; each {@link Identifier} maps to a sub-directory
 * containing one or more {@code part-*.flux} files. This mirrors how the
 * built-in file-based catalogs work, but with flux as the on-disk format.
 *
 * <h2>Why this matters for Unity Catalog</h2>
 * UC's {@code AppendData not supported} error fires when a custom V2
 * datasource is invoked through path-based {@code DataFrameWriter.save}
 * because UC only allow-lists Delta/Parquet/etc. for path writes.
 * Routing through a registered {@link TableCatalog} is the supported way
 * to get a non-Delta source into UC: workspace admins enable it once,
 * then users call {@code spark.table("flux.db.t")} and
 * {@code df.writeTo("flux.db.t").overwrite(...)} \u2014 both go through this
 * catalog instead of UC's path allow-list.
 */
public final class FluxCatalog implements TableCatalog, SupportsNamespaces {

    private String name;
    private CaseInsensitiveStringMap options;
    private Path rootPath;

    @Override
    public void initialize(String name, CaseInsensitiveStringMap options) {
        this.name = name;
        this.options = options;
        String path = options.get("path");
        if (path == null) {
            throw new IllegalArgumentException(
                "spark.sql.catalog." + name + ".path must be set "
                + "(typically to a UC Volume like /Volumes/<catalog>/<schema>/<volume>)"
            );
        }
        this.rootPath = Paths.get(path);
    }

    @Override
    public String name() {
        return name;
    }

    // ── Table operations ────────────────────────────────────────────────────

    @Override
    public Identifier[] listTables(String[] namespace) {
        Path nsDir = namespacePath(namespace);
        if (!Files.isDirectory(nsDir)) return new Identifier[0];
        try (Stream<Path> children = Files.list(nsDir)) {
            return children
                .filter(Files::isDirectory)
                .map(p -> Identifier.of(namespace, p.getFileName().toString()))
                .toArray(Identifier[]::new);
        } catch (IOException e) {
            throw new RuntimeException("FluxCatalog.listTables failed under " + nsDir, e);
        }
    }

    @Override
    public Table loadTable(Identifier ident) throws NoSuchTableException {
        Path tableDir = tablePath(ident);
        if (!Files.exists(tableDir)) {
            throw new NoSuchTableException(ident);
        }
        StructType schema = readSchemaSidecar(tableDir);
        Map<String, String> props = new HashMap<>();
        props.put("path", tableDir.toString());
        return new FluxTable(schema, new Transform[0], props);
    }

    @Override
    public Table createTable(
        Identifier ident,
        StructType schema,
        Transform[] partitions,
        Map<String, String> properties
    ) throws TableAlreadyExistsException, NoSuchNamespaceException {
        Path tableDir = tablePath(ident);
        if (Files.exists(tableDir)) {
            throw new TableAlreadyExistsException(ident);
        }
        try {
            Files.createDirectories(tableDir);
        } catch (IOException e) {
            throw new RuntimeException("FluxCatalog.createTable failed for " + ident, e);
        }
        writeSchemaSidecar(tableDir, schema);
        Map<String, String> props = new HashMap<>(properties != null ? properties : Map.of());
        props.put("path", tableDir.toString());
        return new FluxTable(schema, partitions, props);
    }

    @Override
    public Table alterTable(Identifier ident, TableChange... changes) throws NoSuchTableException {
        // Schema evolution lives on the v0.4 roadmap (see crates/loom/src/txn/).
        // For now, alter is a no-op that just returns the existing Table so
        // that DDL like SET TBLPROPERTIES doesn't fail outright.
        return loadTable(ident);
    }

    @Override
    public boolean dropTable(Identifier ident) {
        Path tableDir = tablePath(ident);
        if (!Files.exists(tableDir)) return false;
        try (Stream<Path> walk = Files.walk(tableDir)) {
            walk.sorted(Comparator.reverseOrder())
                .forEach(p -> { try { Files.deleteIfExists(p); } catch (IOException ignore) {} });
            return true;
        } catch (IOException e) {
            throw new RuntimeException("FluxCatalog.dropTable failed for " + ident, e);
        }
    }

    @Override
    public void renameTable(Identifier oldIdent, Identifier newIdent)
        throws NoSuchTableException, TableAlreadyExistsException {
        Path src = tablePath(oldIdent);
        Path dst = tablePath(newIdent);
        if (!Files.exists(src)) throw new NoSuchTableException(oldIdent);
        if (Files.exists(dst))  throw new TableAlreadyExistsException(newIdent);
        try {
            Files.createDirectories(dst.getParent());
            Files.move(src, dst);
        } catch (IOException e) {
            throw new RuntimeException("FluxCatalog.renameTable failed for " + oldIdent, e);
        }
    }

    // ── SupportsNamespaces ──────────────────────────────────────────────────

    @Override
    public String[][] listNamespaces() {
        try (Stream<Path> children = Files.list(rootPath)) {
            return children.filter(Files::isDirectory)
                .map(p -> new String[] { p.getFileName().toString() })
                .toArray(String[][]::new);
        } catch (IOException e) {
            return new String[0][];
        }
    }

    @Override
    public String[][] listNamespaces(String[] namespace) {
        return listNamespaces();
    }

    @Override
    public Map<String, String> loadNamespaceMetadata(String[] namespace) {
        Map<String, String> out = new HashMap<>();
        out.put("location", namespacePath(namespace).toString());
        return out;
    }

    @Override
    public void createNamespace(String[] namespace, Map<String, String> metadata) {
        try { Files.createDirectories(namespacePath(namespace)); }
        catch (IOException e) { throw new RuntimeException(e); }
    }

    @Override
    public void alterNamespace(String[] namespace, org.apache.spark.sql.connector.catalog.NamespaceChange... changes) {
        // No-op: metadata is implicit in directory layout.
    }

    @Override
    public boolean dropNamespace(String[] namespace, boolean cascade) {
        Path nsDir = namespacePath(namespace);
        if (!Files.exists(nsDir)) return false;
        try (Stream<Path> walk = Files.walk(nsDir)) {
            walk.sorted(Comparator.reverseOrder())
                .forEach(p -> { try { Files.deleteIfExists(p); } catch (IOException ignore) {} });
            return true;
        } catch (IOException e) {
            throw new RuntimeException("FluxCatalog.dropNamespace failed", e);
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private Path namespacePath(String[] namespace) {
        Path p = rootPath;
        for (String segment : namespace) p = p.resolve(segment);
        return p;
    }

    private Path tablePath(Identifier ident) {
        return namespacePath(ident.namespace()).resolve(ident.name());
    }

    /** Schema is persisted as a tiny JSON sidecar so loadTable can rebuild it. */
    private static void writeSchemaSidecar(Path tableDir, StructType schema) {
        try {
            Files.writeString(tableDir.resolve("_schema.json"), schema.json());
        } catch (IOException e) {
            throw new RuntimeException("FluxCatalog: writing schema sidecar failed", e);
        }
    }

    private static StructType readSchemaSidecar(Path tableDir) {
        Path file = tableDir.resolve("_schema.json");
        if (!Files.exists(file)) return new StructType();
        try {
            return (StructType) StructType.fromJson(Files.readString(file));
        } catch (IOException e) {
            throw new RuntimeException("FluxCatalog: reading schema sidecar failed", e);
        }
    }
}
