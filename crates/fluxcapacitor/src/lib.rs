// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # fluxcapacitor (lib)
//!
//! Library surface for the `fluxcapacitor` CLI. Only items used by the
//! integration tests, benchmarks, and the binary live here.
//!
//! The most important sub-module is [`formats`], which provides a
//! file-format-agnostic way to load and save Arrow `RecordBatch`es from
//! CSV, TSV, JSON, NDJSON, Parquet, Arrow IPC, ORC, and Excel files.

pub mod formats;
