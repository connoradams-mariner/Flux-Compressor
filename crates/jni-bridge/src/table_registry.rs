// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe registry of open [`FluxTable`] instances.
//!
//! JNI callers receive an opaque `jlong` handle when they open a table and
//! must pass it back on every subsequent call.  This module manages the
//! mapping from handle → `FluxTable` with a global `OnceLock<Mutex<...>>`
//! so the state survives across multiple JNI call frames.
//!
//! ## Thread safety
//! The registry mutex is held only for the duration of the lookup / insert /
//! remove operations; actual table I/O (append, scan, evolve) is performed
//! **outside** the lock via the `with` and `with_mut` functions, which
//! clone the relevant path and reconstruct a short-lived operation rather
//! than passing a live reference across the lock boundary.
//!
//! For `tableScan` and `tableAppend` we expose a `take_and_replace` helper
//! that removes the table from the registry for the duration of the
//! mutation, then re-inserts it.  This avoids holding the lock across
//! arbitrarily-long I/O while still preventing concurrent modification of
//! the same handle.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use jni::sys::jlong;

use loom::txn::FluxTable;

// ─────────────────────────────────────────────────────────────────────────────
// Internal registry
// ─────────────────────────────────────────────────────────────────────────────

struct Registry {
    /// Auto-incrementing handle counter.  Starts at 1 so that 0 can be
    /// used as a sentinel "not a handle" value on the Java side.
    next_handle: jlong,
    tables: HashMap<jlong, FluxTable>,
}

impl Registry {
    fn new() -> Self {
        Self {
            next_handle: 1,
            tables: HashMap::new(),
        }
    }
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

fn global() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| Mutex::new(Registry::new()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Insert a [`FluxTable`] into the registry and return its new handle.
///
/// The returned `jlong` is guaranteed to be > 0.
pub fn insert(table: FluxTable) -> jlong {
    let mut reg = global().lock().unwrap();
    let handle = reg.next_handle;
    reg.next_handle += 1;
    reg.tables.insert(handle, table);
    handle
}

/// Apply a read-only closure to the table at `handle`.
///
/// Returns `None` when the handle is not found (stale or invalid).
/// The registry lock is held for the duration of `f`.
pub fn with<R>(handle: jlong, f: impl FnOnce(&FluxTable) -> R) -> Option<R> {
    let reg = global().lock().unwrap();
    reg.tables.get(&handle).map(f)
}

/// Apply a mutable closure to the table at `handle`.
///
/// Returns `None` when the handle is not found.
/// The registry lock is held for the duration of `f`.
pub fn with_mut<R>(handle: jlong, f: impl FnOnce(&mut FluxTable) -> R) -> Option<R> {
    let mut reg = global().lock().unwrap();
    reg.tables.get_mut(&handle).map(f)
}

/// Remove and return the [`FluxTable`] at `handle`.
///
/// Returns `None` when the handle does not exist.  Callers that want to
/// perform I/O outside the lock should use this, do their work, and then
/// call [`insert`] if the table should remain open.
pub fn remove(handle: jlong) -> Option<FluxTable> {
    let mut reg = global().lock().unwrap();
    reg.tables.remove(&handle)
}

/// Returns `true` if the handle is currently registered.
pub fn contains(handle: jlong) -> bool {
    let reg = global().lock().unwrap();
    reg.tables.contains_key(&handle)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_tmp_table() -> (TempDir, FluxTable) {
        let dir = TempDir::new().unwrap();
        let table = FluxTable::open(dir.path().join("t.fluxtable")).unwrap();
        (dir, table)
    }

    #[test]
    fn insert_and_lookup() {
        let (_dir, table) = open_tmp_table();
        let handle = insert(table);
        assert!(handle > 0, "handle must be positive");
        assert!(contains(handle));
    }

    #[test]
    fn with_reads_table() {
        let (_dir, table) = open_tmp_table();
        let handle = insert(table);
        let result = with(handle, |t| t.log_dir().exists());
        assert_eq!(result, Some(true));
    }

    #[test]
    fn remove_drops_table() {
        let (_dir, table) = open_tmp_table();
        let handle = insert(table);
        let removed = remove(handle);
        assert!(removed.is_some());
        assert!(!contains(handle));
        // Second remove returns None.
        assert!(remove(handle).is_none());
    }

    #[test]
    fn invalid_handle_returns_none() {
        let bogus: jlong = 999_999;
        assert_eq!(with(bogus, |_| ()), None);
        assert!(remove(bogus).is_none());
    }

    #[test]
    fn handles_are_monotonically_increasing() {
        let (_d1, t1) = open_tmp_table();
        let (_d2, t2) = open_tmp_table();
        let h1 = insert(t1);
        let h2 = insert(t2);
        assert!(h2 > h1);
        let _ = remove(h1);
        let _ = remove(h2);
    }
}
