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

use arrow_array::{ArrayRef, RecordBatch};
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
/// Construct predicates using the module-level ``col()`` helper:
///
/// ```python
/// pred = fc.col("user_id") > 1_000_000
/// pred = (fc.col("revenue") >= 0) & (fc.col("revenue") <= 99_999)
/// ```
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
/// Supports comparison operators to build predicates:
///
/// ```python
/// fc.col("age") > 30
/// fc.col("revenue").between(0, 99_999)
/// ```
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
// PyFluxBatchReader — streaming batch reader for Python
// ─────────────────────────────────────────────────────────────────────────────

use loom::decompressors::flux_reader::FluxBatchIterator;

/// Streaming batch reader over one or more ``.flux`` files.
///
/// Yields one ``pyarrow.RecordBatch`` per file.  Only one file is
/// memory-mapped at a time, keeping peak memory bounded.
///
/// Example:
///
/// ```python
/// reader = fc.FluxBatchReader(["part-0.flux", "part-1.flux"], columns=["id"])
/// for batch in reader:
///     print(batch.num_rows)
/// ```
#[pyclass(name = "FluxBatchReader", module = "fluxcompress")]
pub struct PyFluxBatchReader {
    inner: std::cell::RefCell<FluxBatchIterator>,
    py_schema: PyObject,
}

#[pymethods]
impl PyFluxBatchReader {
    /// Create a batch reader over the given ``.flux`` file paths.
    ///
    /// Args:
    ///     paths:     List of ``.flux`` file paths.
    ///     columns:   Optional column projection (only these columns are decompressed).
    ///     predicate: Optional ``Predicate`` for block-level pushdown.
    #[new]
    #[pyo3(signature = (paths, columns=None, predicate=None))]
    fn new(
        py: Python<'_>,
        paths: Vec<String>,
        columns: Option<Vec<String>>,
        predicate: Option<&PyPredicate>,
    ) -> PyResult<Self> {
        use arrow::pyarrow::ToPyArrow;

        let path_bufs: Vec<std::path::PathBuf> = paths.iter()
            .map(|s| std::path::PathBuf::from(s))
            .collect();
        let pred = predicate
            .map(|p| p.inner.clone())
            .unwrap_or(Predicate::None);

        let iter = FluxBatchIterator::new(path_bufs, columns, pred)
            .map_err(flux_err)?;

        let py_schema = iter.schema().to_pyarrow(py)
            .map_err(|e| PyRuntimeError::new_err(format!("Arrow FFI: {e}")))?;

        Ok(Self {
            inner: std::cell::RefCell::new(iter),
            py_schema,
        })
    }

    /// The Arrow schema of batches produced by this reader.
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyObject {
        self.py_schema.clone_ref(py)
    }

    /// Number of files remaining.
    #[getter]
    fn remaining(&self) -> usize {
        self.inner.borrow().remaining()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let batch_opt = self.inner.borrow_mut().next();
        match batch_opt {
            None => Ok(None),
            Some(Ok(batch)) => {
                let py_batch = record_batch_to_pyarrow(py, &batch)?;
                Ok(Some(py_batch))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to a ``pyarrow.RecordBatchReader``.
    ///
    /// The returned reader lazily pulls batches from this iterator.
    fn to_reader(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pa = py.import("pyarrow")?;
        let reader_cls = pa.getattr("RecordBatchReader")?;
        let reader = reader_cls.call_method1(
            "from_batches",
            (&self.py_schema, self.as_batch_list(py)?),
        )?;
        Ok(reader.into())
    }

    /// Collect all remaining batches as a list of pyarrow.RecordBatch.
    fn as_batch_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut batches = Vec::new();
        loop {
            let batch_opt = self.inner.borrow_mut().next();
            match batch_opt {
                None => break,
                Some(Ok(batch)) => batches.push(record_batch_to_pyarrow(py, &batch)?),
                Some(Err(e)) => return Err(PyRuntimeError::new_err(e.to_string())),
            }
        }
        let list = pyo3::types::PyList::new(py, &batches);
        Ok(list.into())
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
/// Example:
///
/// ```python
/// import pyarrow as pa
/// import fluxcompress as fc
///
/// t = pa.table({"id": range(1_000_000)})
/// buf = fc.compress(t)
/// print(buf)  # FluxBuffer(312451 bytes)
/// ```
#[pyfunction]
#[pyo3(signature = (table, strategy = "auto", profile = "speed", u64_only = false))]
fn compress(
    py: Python<'_>,
    table: &PyAny,
    strategy: &str,
    profile: &str,
    u64_only: bool,
) -> PyResult<PyFluxBuffer> {
    let batch = pyarrow_to_record_batch(py, table)?;

    let forced = parse_strategy(strategy)?;
    let prof = parse_profile(profile)?;
    let mut writer = match forced {
        Some(s) => FluxWriter::with_strategy(s),
        None    => FluxWriter::new(),
    };
    writer.profile = prof;
    writer.u64_only = u64_only;

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
/// Example:
///
/// ```python
/// table = fc.decompress(buf, predicate=fc.col("id") > 500_000)
/// ```
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
/// Example:
///
/// ```python
/// info = fc.inspect(buf)
/// for block in info.blocks:
///     print(block.strategy, block.z_min, block.z_max)
/// ```
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
/// Example:
///
/// ```python
/// pred = fc.col("user_id") > 1_000_000
/// pred = (fc.col("revenue") >= 0) & (fc.col("revenue") <= 99_999)
/// ```
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

/// Decompress a ``.flux`` file directly from disk using memory-mapped I/O.
///
/// This is more memory-efficient than ``read_flux()`` + ``decompress()``
/// because the compressed bytes are never loaded into Python memory — the OS
/// maps the file into virtual memory and only pages in the blocks that are
/// actually decompressed.
///
/// Args:
///     path:       Path to a ``.flux`` file.
///     predicate:  Optional ``Predicate`` for pushdown filtering.
///     columns:    Optional list of column names for projection.
///                 Only these columns will be decompressed.
///
/// Returns:
///     ``pyarrow.Table``
#[pyfunction]
#[pyo3(signature = (path, predicate=None, columns=None))]
fn decompress_file(
    py: Python<'_>,
    path: &str,
    predicate: Option<&PyPredicate>,
    columns: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let pred = predicate
        .map(|p| p.inner.clone())
        .unwrap_or(Predicate::None);

    let reader = FluxReader::default();
    let file_path = std::path::Path::new(path);

    let batch = match columns {
        Some(cols) => reader.decompress_file_projected(file_path, &pred, &cols).map_err(flux_err)?,
        None => reader.decompress_file(file_path, &pred).map_err(flux_err)?,
    };

    record_batch_to_pyarrow(py, &batch)
}

/// Read the Arrow schema from a ``.flux`` file without decompressing any data.
///
/// Args:
///     path: Path to a ``.flux`` file.
///
/// Returns:
///     ``pyarrow.Schema``
#[pyfunction]
fn read_flux_schema(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    use arrow::pyarrow::ToPyArrow;
    let schema = FluxReader::read_schema_from_file(std::path::Path::new(path))
        .map_err(flux_err)?;
    schema.to_pyarrow(py)
        .map_err(|e| PyRuntimeError::new_err(format!("Arrow FFI: {e}")))
}

/// Write a ``FluxBuffer`` to a ``.flux`` file on disk.
///
/// Equivalent to ``buf.save(path)``.
#[pyfunction]
fn write_flux(buf: PyRef<PyFluxBuffer>, path: &str) -> PyResult<()> {
    buf.save(path)
}

/// Merge multiple ``FluxBuffer`` objects into a single buffer.
///
/// This is used by chunked compression: each chunk is compressed
/// independently, then the block bytes and footer metadata are
/// stitched together into one valid ``.flux`` buffer.
///
/// Args:
///     buffers: A list of ``FluxBuffer`` objects to merge.
///
/// Returns:
///     A single ``FluxBuffer`` containing all blocks from all inputs.
#[pyfunction]
fn merge_flux_buffers(buffers: Vec<PyRef<PyFluxBuffer>>) -> PyResult<PyFluxBuffer> {
    if buffers.is_empty() {
        return Err(PyValueError::new_err("no buffers to merge"));
    }
    if buffers.len() == 1 {
        return Ok(PyFluxBuffer { data: buffers[0].data.clone() });
    }

    // Parse each footer, collect block bytes and metadata.
    let mut all_block_bytes: Vec<u8> = Vec::new();
    let mut merged_footer = AtlasFooter::new();
    let mut schema_set = false;

    for buf_ref in &buffers {
        let footer = AtlasFooter::from_file_tail(&buf_ref.data).map_err(flux_err)?;

        // Use the schema from the first buffer (all chunks share the same schema).
        if !schema_set && !footer.schema.is_empty() {
            merged_footer.schema = footer.schema.clone();
            schema_set = true;
        }

        for meta in &footer.blocks {
            let block_start = meta.block_offset as usize;
            // Determine block end: next block's offset, or start of footer.
            let block_end = footer.blocks.iter()
                .map(|m| m.block_offset as usize)
                .filter(|&o| o > block_start)
                .min()
                .unwrap_or_else(|| {
                    // Last block: ends where footer begins.
                    // footer_length is the last 12 bytes: schema_len + block_count + footer_length + magic
                    let data_len = buf_ref.data.len();
                    let footer_bytes = footer.blocks.len() * loom::atlas::BLOCK_META_SIZE
                        + 4 + 4 + 4; // block_count + footer_length + magic
                    let schema_json_len: usize = if footer.schema.is_empty() { 0 } else {
                        // schema_json + schema_len(u32)
                        serde_json::to_vec(&footer.schema).map(|v| v.len() + 4).unwrap_or(4)
                    };
                    data_len - footer_bytes - schema_json_len
                });

            let mut adjusted_meta = meta.clone();
            adjusted_meta.block_offset = all_block_bytes.len() as u64;
            all_block_bytes.extend_from_slice(&buf_ref.data[block_start..block_end]);
            merged_footer.push(adjusted_meta);
        }
    }

    let footer_bytes = merged_footer.to_bytes().map_err(flux_err)?;
    let mut output = Vec::with_capacity(all_block_bytes.len() + footer_bytes.len());
    output.extend_from_slice(&all_block_bytes);
    output.extend_from_slice(&footer_bytes);

    Ok(PyFluxBuffer { data: output })
}

// ─────────────────────────────────────────────────────────────────────────────
// PyArrow C Data Interface bridge (zero-copy)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert any Python object with the Arrow C Data Interface to a RecordBatch.
///
/// Supports: pyarrow.Table, pyarrow.RecordBatch, polars.DataFrame (via
/// __arrow_c_stream__).
///
/// Handles `LargeUtf8`/`LargeBinary` normalization on the Rust side so the
/// Python caller does not need to copy the table for type casting.
fn pyarrow_to_record_batch(py: Python<'_>, obj: &PyAny) -> PyResult<RecordBatch> {
    use arrow::pyarrow::FromPyArrow;

    let pa = py.import("pyarrow")?;
    let pa_table_cls = pa.getattr("Table")?;

    // If obj is a Table, combine into a single RecordBatch first.
    let batch_obj: &PyAny = if obj.is_instance(pa_table_cls)? {
        // table.combine_chunks().to_batches()[0] gives one contiguous batch.
        let combined = obj.call_method0("combine_chunks")?;
        let batches: Vec<&PyAny> = combined.call_method0("to_batches")?.extract()?;
        if batches.is_empty() {
            return Err(PyValueError::new_err("empty Arrow table"));
        }
        batches[0]
    } else {
        obj
    };

    // Arrow C Data Interface: zero-copy FFI from PyArrow → Rust Arrow.
    let batch = RecordBatch::from_pyarrow_bound(&batch_obj.as_borrowed())
        .map_err(|e| PyRuntimeError::new_err(format!("Arrow FFI: {e}")))?;

    // Normalize LargeUtf8 → Utf8 and LargeBinary → Binary in-place.
    // Polars to_arrow() can produce inconsistent schema/column type combos;
    // fixing it here avoids a full Python-side table.cast() copy.
    normalize_large_string_columns(batch)
        .map_err(|e| PyRuntimeError::new_err(format!("schema normalization: {e}")))
}

/// Cast any `LargeUtf8` / `LargeBinary` columns down to `Utf8` / `Binary`.
/// Columns that don't need casting are returned as-is (zero-copy).
fn normalize_large_string_columns(batch: RecordBatch) -> Result<RecordBatch, arrow::error::ArrowError> {
    use arrow_schema::DataType;

    let schema = batch.schema();
    let needs_cast = schema.fields().iter().any(|f| {
        matches!(f.data_type(), DataType::LargeUtf8 | DataType::LargeBinary)
    });
    if !needs_cast {
        return Ok(batch);
    }

    let mut new_fields = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        match field.data_type() {
            DataType::LargeUtf8 => {
                let cast_col = arrow::compute::cast(col, &DataType::Utf8)?;
                new_fields.push(Arc::new(field.as_ref().clone().with_data_type(DataType::Utf8)));
                new_columns.push(cast_col);
            }
            DataType::LargeBinary => {
                let cast_col = arrow::compute::cast(col, &DataType::Binary)?;
                new_fields.push(Arc::new(field.as_ref().clone().with_data_type(DataType::Binary)));
                new_columns.push(cast_col);
            }
            _ => {
                new_fields.push(Arc::new(schema.field(i).clone()));
                new_columns.push(Arc::clone(col));
            }
        }
    }

    let new_schema = Arc::new(Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_columns)
}

/// Convert a Rust RecordBatch to a PyArrow Table via Arrow C Data Interface (zero-copy).
fn record_batch_to_pyarrow(py: Python<'_>, batch: &RecordBatch) -> PyResult<PyObject> {
    use arrow::pyarrow::ToPyArrow;

    // Zero-copy FFI: Rust Arrow → PyArrow RecordBatch.
    let py_batch = batch.to_pyarrow(py)
        .map_err(|e| PyRuntimeError::new_err(format!("Arrow FFI: {e}")))?;

    // Wrap in a Table for consistency with the rest of the API.
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let table = table_cls.call_method1("from_batches", (vec![py_batch],))?;
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

fn parse_profile(s: &str) -> PyResult<loom::CompressionProfile> {
    match s {
        "speed"    => Ok(loom::CompressionProfile::Speed),
        "balanced" => Ok(loom::CompressionProfile::Balanced),
        "archive"  => Ok(loom::CompressionProfile::Archive),
        other      => Err(PyValueError::new_err(format!(
            "unknown profile '{other}'. Use: speed|balanced|archive"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────

/// FluxCompress — high-performance adaptive columnar compression.
///
/// Quick start:
///
/// ```python
/// import pyarrow as pa
/// import fluxcompress as fc
///
/// # Compress
/// table = pa.table({"id": range(1_000_000), "val": range(1_000_000)})
/// buf = fc.compress(table)
///
/// # Decompress
/// table2 = fc.decompress(buf)
///
/// # Predicate pushdown
/// table3 = fc.decompress(buf, predicate=fc.col("id") > 500_000)
///
/// # Save / load
/// fc.write_flux(buf, "data.flux")
/// buf2 = fc.read_flux("data.flux")
///
/// # Inspect
/// info = fc.inspect(buf)
/// print(f"{info.num_blocks} blocks, {info.size_bytes} bytes")
/// ```
#[pymodule]
fn _fluxcompress(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Classes
    m.add_class::<PyPredicate>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PyBlockInfo>()?;
    m.add_class::<PyFileInfo>()?;
    m.add_class::<PyFluxBuffer>()?;
    m.add_class::<PyFluxBatchReader>()?;

    // Functions
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(col, m)?)?;
    m.add_function(wrap_pyfunction!(read_flux, m)?)?;
    m.add_function(wrap_pyfunction!(write_flux, m)?)?;
    m.add_function(wrap_pyfunction!(merge_flux_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_file, m)?)?;
    m.add_function(wrap_pyfunction!(read_flux_schema, m)?)?;

    // Version constant
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
