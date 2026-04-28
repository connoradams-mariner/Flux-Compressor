// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # Transaction Log for Time Travel
//!
//! Provides Delta-Lake-style versioned access to `.flux` datasets.
//!
//! ## Directory Layout
//! ```text
//! my_table.fluxtable/
//! ├── _flux_log/
//! │   ├── 00000000.json   # version 0: create
//! │   ├── 00000001.json   # version 1: append
//! │   └── ...
//! ├── data/
//! │   ├── part-0000.flux
//! │   └── ...
//! └── _flux_meta.json
//! ```

mod log_entry;
pub mod mutation;
pub mod optimizer;
pub mod partition;
pub mod projection;
pub mod schema;
mod snapshot;
mod table;
pub mod wal;

pub use log_entry::{Action, LogEntry, Operation};
pub use mutation::{
    DeleteStats, MatchedAction, MergeClauses, MergeStats, MutationAction, NotMatchedAction,
    ScalarValue, UpdateStats,
};
pub use partition::{
    ColumnStats, FileManifest, PartitionField, PartitionSpec, PartitionTransform, TableMeta,
};
pub use projection::{ColumnPlan, FilePlan, build_file_plan};
pub use schema::{DefaultValue, PromotedFrom, SchemaChain, SchemaField, TableSchema};
pub use snapshot::Snapshot;
pub use table::{EvolveOptions, FluxScan, FluxTable};
pub use wal::{LogFormat, WalEntry, WalLog};
