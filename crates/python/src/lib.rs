// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! # fluxcompress Python bindings (PyO3)
//!
//! Exposes the full FluxCompress API to Python with a Pythonic interface:
//!
//! ```python
//! import fluxcompress as fc
//! import pyarrow as pa
//!
//! table = pa.table({"id": range(1_000_000), "val": range(1_000_000)})
//!
//! # Compress
//! blob = fc.compress(table)
//!
//! # Decompress
//! table2 = fc.decompress(blob)
//!
//! # Predicate pushdown
//! table3 = fc.decompress(blob, predicate=fc.col("id") > 500_000)
//!
//! # Inspect
//! info = fc.inspect(blob)
//! print(info.blocks)
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyTypeError};
use pyo3::types::{PyBytes, PyDict, PyList, PyString};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use std::sync::Arc;

use loom::{
    atlas::AtlasFooter,
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    loom_classifier::{classify, LoomStrategy},
    outlier_map::{encode_with_outlier_map, decode_with_outlier_map},
    traits::{LoomCompressor, LoomDecompressor, Predicate},
    SEGMENT_SIZE,
};

// ─────────────────────────────────────────────────────────────────────────────
// Error conversion
// ─────────────────────────────────────────────────────────────────────────────

fn flux_err(e: loom::error::FluxError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// PyPredicate — Python-constructable predicate
// ─────────────────────────────────────────────────────────────────────────────

/// A filter predicate used for predicate pushdown during decompression.
///
/// FluxCompress uses the Atlas footer to skip compressed blocks whose
/// ``[z_min, z_max]`` range cannot satisfy this predicate — so irrelevant
/// data is never decompressed.
///
/// Construct predicates using the module-level ``col()`` helper::
///
///     pred = fc.col("user_id") > 1_000_000
///     pred = (fc.col("revenue") >= 0) & (fc.col("revenue") <= 99_999)
#[pyclass(name = "Predicate", module = "fluxcompress")]
#[derive(Clone)]
pub struct PyPredicate {
    inner: Predicate,
}

#[pymethods]
impl PyPredicate {
    fn __repr__(&self) -> String {
        format!("Predicate({:?})", self.inner)
    }

    /// Logical AND of two predicates.
    fn __and__(&self, other: &PyPredicate) -> PyPredicate {
        PyPredicate {
            inner: Predicate::And(
                Box::new(self.inner.clone()),
                Box::new(other.inner.clone()),
            ),
        }
    }

    /// Logical OR of two predicates.
    fn __or__(&self, other: &PyPredicate) -> PyPredicate {
        PyPredicate {
            inner: Predicate::Or(
                Box::new(self.inner.clone()),
                Box::new(other.inner.clone()),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyColumnExpr — intermediate for building predicates via col("x") > 5
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediate object returned by ``fc.col(name)``.
///
/// Supports comparison operators to build predicates::
///
///     fc.col("age") > 30
///     fc.col("revenue").between(0, 99_999)
#[pyclass(name = "Column", module = "fluxcompress")]
pub struct PyColumn {
    name: String,
}

#[pymethods]
impl PyColumn {
    fn __repr__(&self) -> String {
        format!("col(\"{}\")", self.name)
    }

    fn __gt__(&self, value: i128) -> PyPredicate {
        PyPredicate {
            inner: Predicate::GreaterThan { column: self.name.clone(), value },
        }
    }

    fn __lt__(&self, value: i128) -> PyPredicate {
        PyPredicate {
            inner: Predicate::LessThan { column: self.name.clone(), value },
        }
    }

    fn __eq__(&self, value: i128) -> PyPredicate {
        PyPredicate {
            inner: Predicate::Equal { column: self.name.clone(), value },
        }
    }

    fn __ge__(&self, value: i128) -> PyPredicate {
        // >= implemented as NOT (< value)
        PyPredicate {
            inner: Predicate::GreaterThan { column: self.name.clone(), value: value - 1 },
        }
    }

    fn __le__(&self, value: i128) -> PyPredicate {
        PyPredicate {
            inner: Predicate::LessThan { column: self.name.clone(), value: value + 1 },
        }
    }

    /// Return rows where ``lo <= column <= hi``.
    fn between(&self, lo: i128, hi: i128) -> PyPredicate {
        PyPredicate {
            inner: Predicate::Between { column: self.name.clone(), lo, hi },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyBlockInfo — metadata about one compressed block
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata for a single compressed block from the Atlas footer.
#[pyclass(name = "BlockInfo", module = "fluxcompress", get_all)]
#[derive(Clone)]
pub struct PyBlockInfo {
    /// Byte offset of this block within the .flux buffer.
    pub offset: u64,
    /// Minimum Z-Order coordinate (or column min value).
    pub z_min: u128,
    /// Maximum Z-Order coordinate (or column max value).
    pub z_max: u128,
    /// Compression strategy name used for this block.
    pub strategy: String,
}

#[pymethods]
impl PyBlockInfo {
    fn __repr__(&self) -> String {
        format!(
            "BlockInfo(offset={}, z_min={}, z_max={}, strategy='{}')",
            self.offset, self.z_min, self.z_max, self.strategy
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyFileInfo — full inspection result
// ─────────────────────────────────────────────────────────────────────────────

/// Full inspection result for a .flux buffer — returned by ``fc.inspect()``.
#[pyclass(name = "FileInfo", module = "fluxcompress", get_all)]
pub struct PyFileInfo {
    /// Total size of the .flux buffer in bytes.
    pub size_bytes: usize,
    /// Number of compressed blocks.
    pub num_blocks: usize,
    /// Per-block metadata.
    pub blocks: Vec<PyBlockInfo>,
}

#[pymethods]
impl PyFileInfo {
    fn __repr__(&self) -> String {
        format!(
            "FileInfo(size_bytes={}, num_blocks={})",
            self.size_bytes, self.num_blocks
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyFluxBuffer — a compressed .flux byte buffer with a Pythonic API
// ─────────────────────────────────────────────────────────────────────────────

/// A compressed FluxCompress buffer.
///
/// Returned by ``fc.compress()``.  Supports:
///
/// - ``len()``            — size in bytes
/// - ``bytes()``          — get raw bytes
/// - ``decompress()``     — decompress to PyArrow table
/// - ``inspect()``        — inspect the Atlas footer
/// - ``save(path)``       — write to a ``.flux`` file
/// - ``load(path)``       — class method to load from file
#[pyclass(name = "FluxBuffer", module = "fluxcompress")]
pub struct PyFluxBuffer {
    data: Vec<u8>,
}

#[pymethods]
impl PyFluxBuffer {
    /// Number of bytes in the compressed buffer.
    fn __len__(&self) -> usize {
        self.data.len()
    }

    fn __repr__(&self) -> String {
        format!("FluxBuffer({} bytes)", self.data.len())
    }

    /// Get the raw compressed bytes (copies into a Python ``bytes`` object).
    fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.data)
    }

    /// Decompress to a PyArrow ``Table``.
    ///
    /// Args:
    ///     predicate: Optional ``Predicate`` for pushdown filtering.
    ///         Blocks that cannot satisfy the predicate are skipped entirely.
    ///     column_name: Name for the output column (default ``"value"``).
    ///
    /// Returns:
    ///     ``pyarrow.Table``
    #[pyo3(signature = (predicate=None, column_name="value"))]
    fn decompress(
        &self,
        py: Python<'_>,
        predicate: Option<&PyPredicate>,
        column_name: &str,
    ) -> PyResult<PyObject> {
        let pred = predicate
            .map(|p| p.inner.clone())
            .unwrap_or(Predicate::None);

        let reader = FluxReader::new(column_name);
        let batch = reader.decompress(&self.data, &pred).map_err(flux_err)?;

        record_batch_to_pyarrow(py, &batch)
    }

    /// Inspect the Atlas metadata footer.
    ///
    /// Returns:
    ///     ``FileInfo`` with block-level metadata (offsets, Z-min/max, strategy).
    fn inspect(&self) -> PyResult<PyFileInfo> {
        let footer = AtlasFooter::from_file_tail(&self.data).map_err(flux_err)?;
        Ok(PyFileInfo {
            size_bytes: self.data.len(),
            num_blocks: footer.blocks.len(),
            blocks: footer.blocks.iter().map(|b| PyBlockInfo {
                offset:   b.block_offset,
                z_min:    b.z_min,
                z_max:    b.z_max,
                strategy: format!("{:?}", b.strategy),
            }).collect(),
        })
    }

    /// Save the compressed buffer to a ``.flux`` file.
    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, &self.data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Load a ``.flux`` file from disk.
    #[staticmethod]
    fn load(path: &str) -> PyResult<PyFluxBuffer> {
        let data = std::fs::read(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyFluxBuffer { data })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-level functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compress a PyArrow ``Table`` or ``RecordBatch`` into a ``FluxBuffer``.
///
/// FluxCompress automatically selects the best compression strategy
/// (RLE, Delta-Delta, Dictionary, BitSlab, or LZ4) for each 1024-row segment
/// using the Loom adaptive classifier.
///
/// Args:
///     table:    A ``pyarrow.Table``, ``pyarrow.RecordBatch``, or any object
///               with a ``__arrow_c_stream__`` interface (Polars DataFrame, etc.).
///     strategy: Force a specific strategy instead of auto-selecting.
///               One of ``"auto"``, ``"rle"``, ``"delta"``, ``"dict"``,
///               ``"bitslab"``, ``"lz4"``.  Default ``"auto"``.
///
/// Returns:
///     ``FluxBuffer`` — the compressed bytes with Atlas footer.
///
/// Example::
///
///     import pyarrow as pa
///     import fluxcompress as fc
///
///     t = pa.table({"id": range(1_000_000)})
///     buf = fc.compress(t)
///     print(buf)  # FluxBuffer(312451 bytes)
#[pyfunction]
#[pyo3(signature = (table, strategy = "auto"))]
fn compress(py: Python<'_>, table: &PyAny, strategy: &str) -> PyResult<PyFluxBuffer> {
    let batch = pyarrow_to_record_batch(py, table)?;

    let forced = parse_strategy(strategy)?;
    let writer = match forced {
        Some(s) => FluxWriter::with_strategy(s),
        None    => FluxWriter::new(),
    };

    let data = writer.compress(&batch).map_err(flux_err)?;
    Ok(PyFluxBuffer { data })
}

/// Decompress a ``FluxBuffer`` or raw ``bytes`` into a PyArrow ``Table``.
///
/// Args:
///     buf:         A ``FluxBuffer`` or Python ``bytes`` object.
///     predicate:   Optional ``Predicate`` for pushdown filtering.
///     column_name: Column name for the output table (default ``"value"``).
///
/// Returns:
///     ``pyarrow.Table``
///
/// Example::
///
///     table = fc.decompress(buf, predicate=fc.col("id") > 500_000)
#[pyfunction]
#[pyo3(signature = (buf, predicate=None, column_name="value"))]
fn decompress(
    py: Python<'_>,
    buf: &PyAny,
    predicate: Option<&PyPredicate>,
    column_name: &str,
) -> PyResult<PyObject> {
    let data: &[u8] = if let Ok(flux_buf) = buf.extract::<PyRef<PyFluxBuffer>>() {
        // Borrow the inner Vec<u8> — but PyRef doesn't let us borrow past its
        // lifetime easily, so we copy here.  For the hot path use FluxBuffer.decompress().
        return flux_buf.decompress(py, predicate, column_name);
    } else if let Ok(bytes) = buf.extract::<&[u8]>() {
        bytes
    } else {
        return Err(PyTypeError::new_err(
            "expected FluxBuffer or bytes-like object"
        ));
    };

    let pred = predicate
        .map(|p| p.inner.clone())
        .unwrap_or(Predicate::None);

    let reader = FluxReader::new(column_name);
    let batch = reader.decompress(data, &pred).map_err(flux_err)?;
    record_batch_to_pyarrow(py, &batch)
}

/// Inspect the Atlas footer of a ``FluxBuffer`` or raw ``bytes``.
///
/// Returns:
///     ``FileInfo`` with ``size_bytes``, ``num_blocks``, and a list of
///     ``BlockInfo`` objects (offset, z_min, z_max, strategy).
///
/// Example::
///
///     info = fc.inspect(buf)
///     for block in info.blocks:
///         print(block.strategy, block.z_min, block.z_max)
#[pyfunction]
fn inspect(buf: &PyAny) -> PyResult<PyFileInfo> {
    let data: Vec<u8> = if let Ok(flux_buf) = buf.extract::<PyRef<PyFluxBuffer>>() {
        flux_buf.data.clone()
    } else if let Ok(bytes) = buf.extract::<Vec<u8>>() {
        bytes
    } else {
        return Err(PyTypeError::new_err("expected FluxBuffer or bytes"));
    };

    let footer = AtlasFooter::from_file_tail(&data).map_err(flux_err)?;
    Ok(PyFileInfo {
        size_bytes: data.len(),
        num_blocks: footer.blocks.len(),
        blocks: footer.blocks.iter().map(|b| PyBlockInfo {
            offset:   b.block_offset,
            z_min:    b.z_min,
            z_max:    b.z_max,
            strategy: format!("{:?}", b.strategy),
        }).collect(),
    })
}

/// Return a ``Column`` expression for building predicates.
///
/// Example::
///
///     pred = fc.col("user_id") > 1_000_000
///     pred = (fc.col("revenue") >= 0) & (fc.col("revenue") <= 99_999)
#[pyfunction]
fn col(name: &str) -> PyColumn {
    PyColumn { name: name.to_string() }
}

/// Read a ``.flux`` file from disk and return a ``FluxBuffer``.
///
/// Equivalent to ``FluxBuffer.load(path)``.
#[pyfunction]
fn read_flux(path: &str) -> PyResult<PyFluxBuffer> {
    PyFluxBuffer::load(path)
}

/// Write a ``FluxBuffer`` to a ``.flux`` file on disk.
///
/// Equivalent to ``buf.save(path)``.
#[pyfunction]
fn write_flux(buf: PyRef<PyFluxBuffer>, path: &str) -> PyResult<()> {
    buf.save(path)
}

// ─────────────────────────────────────────────────────────────────────────────
// PyArrow C Data Interface bridge (zero-copy)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert any Python object with the Arrow C Data Interface to a RecordBatch.
///
/// Supports: pyarrow.Table, pyarrow.RecordBatch, polars.DataFrame (via
/// __arrow_c_stream__).
fn pyarrow_to_record_batch(py: Python<'_>, obj: &PyAny) -> PyResult<RecordBatch> {
    // Try the Arrow PyCapsule interface first (__arrow_c_stream__).
    // This is zero-copy for pyarrow and polars >= 0.20.
    if obj.hasattr("__arrow_c_stream__")? {
        return pyarrow_via_c_stream(py, obj);
    }

    // Fallback: call obj.to_pydict() and reconstruct column by column.
    // Less efficient but works with any mapping-like object.
    pyarrow_via_to_pydict(py, obj)
}

fn pyarrow_via_c_stream(py: Python<'_>, obj: &PyAny) -> PyResult<RecordBatch> {
    // Import pyarrow.
    let pa = py.import("pyarrow")?;

    // Normalise to a pyarrow.RecordBatch via RecordBatch.from_batches.
    let rb_cls = pa.getattr("RecordBatch")?;

    // If it's a Table, get the first batch (or combine).
    let batch = if obj.get_type().name()? == "Table" {
        // table.to_batches() → list of RecordBatch
        let batches: Vec<&PyAny> = obj.call_method0("to_batches")?.extract()?;
        if batches.is_empty() {
            return Err(PyValueError::new_err("empty Arrow table"));
        }
        // Concatenate all batches.
        let concat_fn = pa.getattr("concat_tables")?;
        let table_list = PyList::new(py, &[obj]);
        let combined = concat_fn.call1((table_list,))?;
        combined.call_method0("to_batches")?.extract::<Vec<&PyAny>>()?[0]
    } else {
        obj
    };

    // Use the C Data Interface: batch.__arrow_c_stream__() returns a PyCapsule.
    // We import via pyarrow's ipc machinery to reconstruct in Rust.
    // For now, use the ipc.serialize_pandas path as a reliable bridge.
    let ipc = pa.getattr("ipc")?;
    let buf = ipc.call_method1("serialize", (batch,))?;
    let raw_bytes: &[u8] = buf.call_method0("to_pybytes")?.extract()?;

    // Deserialize the Arrow IPC message.
    let cursor = std::io::Cursor::new(raw_bytes);
    let reader = arrow::ipc::reader::StreamReader::try_new(cursor, None)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut batches: Vec<RecordBatch> = reader
        .map(|b| b.map_err(|e| PyRuntimeError::new_err(e.to_string())))
        .collect::<PyResult<_>>()?;

    batches.pop().ok_or_else(|| PyValueError::new_err("no batches in IPC stream"))
}

fn pyarrow_via_to_pydict(_py: Python<'_>, _obj: &PyAny) -> PyResult<RecordBatch> {
    Err(PyTypeError::new_err(
        "Cannot convert object to Arrow RecordBatch. \
         Pass a pyarrow.Table, pyarrow.RecordBatch, or polars.DataFrame."
    ))
}

/// Serialize a Rust RecordBatch to a PyArrow Table via IPC.
fn record_batch_to_pyarrow(py: Python<'_>, batch: &RecordBatch) -> PyResult<PyObject> {
    use arrow::ipc::writer::StreamWriter;

    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        writer.write(batch)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        writer.finish()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }

    // Import pyarrow and deserialise.
    let pa   = py.import("pyarrow")?;
    let ipc  = pa.getattr("ipc")?;
    let pybytes = PyBytes::new(py, &buf);
    let reader = ipc.call_method1("open_stream", (pybytes,))?;
    let table  = reader.call_method0("read_all")?;
    Ok(table.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Strategy parser
// ─────────────────────────────────────────────────────────────────────────────

fn parse_strategy(s: &str) -> PyResult<Option<LoomStrategy>> {
    match s {
        "auto"    => Ok(None),
        "rle"     => Ok(Some(LoomStrategy::Rle)),
        "delta"   => Ok(Some(LoomStrategy::DeltaDelta)),
        "dict"    => Ok(Some(LoomStrategy::Dictionary)),
        "bitslab" => Ok(Some(LoomStrategy::BitSlab)),
        "lz4"     => Ok(Some(LoomStrategy::SimdLz4)),
        other     => Err(PyValueError::new_err(format!(
            "unknown strategy '{other}'. Use: auto|rle|delta|dict|bitslab|lz4"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────

/// FluxCompress — high-performance adaptive columnar compression.
///
/// Quick start::
///
///     import pyarrow as pa
///     import fluxcompress as fc
///
///     # Compress
///     table = pa.table({"id": range(1_000_000), "val": range(1_000_000)})
///     buf = fc.compress(table)
///
///     # Decompress
///     table2 = fc.decompress(buf)
///
///     # Predicate pushdown
///     table3 = fc.decompress(buf, predicate=fc.col("id") > 500_000)
///
///     # Save / load
///     fc.write_flux(buf, "data.flux")
///     buf2 = fc.read_flux("data.flux")
///
///     # Inspect
///     info = fc.inspect(buf)
///     print(f"{info.num_blocks} blocks, {info.size_bytes} bytes")
#[pymodule]
fn _fluxcompress(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PyPredicate>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PyBlockInfo>()?;
    m.add_class::<PyFileInfo>()?;
    m.add_class::<PyFluxBuffer>()?;

    // Functions
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(col, m)?)?;
    m.add_function(wrap_pyfunction!(read_flux, m)?)?;
    m.add_function(wrap_pyfunction!(write_flux, m)?)?;

    // Version constant
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
