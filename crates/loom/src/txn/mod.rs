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
mod snapshot;
mod table;

pub use log_entry::{LogEntry, Operation};
pub use snapshot::Snapshot;
pub use table::FluxTable;
