// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package io.fluxcompress.spark;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

import org.apache.spark.sql.connector.write.WriterCommitMessage;

/**
 * Per-task commit message — carries the list of temporary {@code *.flux.tmp}
 * files the task wrote. The driver promotes these to their final
 * {@code *.flux} names in {@link FluxBatchWrite#commit(WriterCommitMessage[])}.
 */
public final class FluxWriterCommitMessage implements WriterCommitMessage, Serializable {

    private static final long serialVersionUID = 1L;

    private final List<String> tempFiles;
    private final long rowsWritten;

    public FluxWriterCommitMessage(List<String> tempFiles, long rowsWritten) {
        this.tempFiles  = tempFiles == null ? Collections.emptyList() : tempFiles;
        this.rowsWritten = rowsWritten;
    }

    public List<String> tempFiles() { return tempFiles; }
    public long rowsWritten()        { return rowsWritten; }
}
