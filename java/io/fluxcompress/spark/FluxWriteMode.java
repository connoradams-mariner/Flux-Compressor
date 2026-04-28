// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

/**
 * Which V2 write semantics the analyzer requested.
 *
 * <p>Captured by {@link FluxWriteBuilder} and consumed by
 * {@link FluxBatchWrite#commit} so the right files are deleted before
 * the new commit becomes visible.
 */
public enum FluxWriteMode {
    /** Default save / {@code mode("append")} — keep existing files. */
    APPEND,
    /** {@code mode("overwrite")} without a filter — delete everything first. */
    TRUNCATE,
    /** {@code .option("replaceWhere", "...")} — delete files matching filter. */
    OVERWRITE_BY_FILTER,
}
