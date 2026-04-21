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
// PyFluxWriter — stateful writer with per-column Zstd dictionary cache
// ─────────────────────────────────────────────────────────────────────────────

/// A stateful compressor that caches a trained Zstd dictionary for each
/// string column.  On the first :meth:`compress` call per column the
/// dictionary is trained from the column’s data; subsequent calls reuse it
/// at Zstd level 3, achieving near-Archive ratio at ~6× the write throughput.
///
/// Ideal for repeated :meth:`FluxTable.append` calls on the same table schema:
///
/// .. code-block:: python
///
///     writer = fc.FluxWriter(profile="archive")
///     for batch in stream:
///         tbl.append(writer.compress(batch))   # 2nd+ batches use cached dicts
#[pyclass(name = "FluxWriter", module = "fluxcompress")]
pub struct PyFluxWriter {
    inner: loom::compressors::flux_writer::FluxWriter,
}

#[pymethods]
impl PyFluxWriter {
    /// Create a new stateful :class:`FluxWriter`.
    ///
    /// Args:
    ///     profile: Compression profile — ``"speed"``, ``"balanced"``,
    ///              ``"archive"``, or ``"brotli"`` (default ``"archive"``).
    ///     u64_only: Skip the 128-bit int widening path (default ``False``).
    #[new]
    #[pyo3(signature = (profile = "archive", u64_only = false))]
    fn new(profile: &str, u64_only: bool) -> PyResult<Self> {
        let p = parse_profile(profile)?;
        let inner = loom::compressors::flux_writer::FluxWriter {
            profile: p,
            u64_only,
            ..Default::default()
        };
        Ok(Self { inner })
    }

    /// Compress a PyArrow Table or RecordBatch into a :class:`FluxBuffer`.
    ///
    /// The per-column Zstd dictionary is trained on the **first call** and
    /// reused on all subsequent calls — subsequent batches are compressed at
    /// Zstd level 3 with the trained dict, recovering most of the write-speed
    /// gap vs the uncached path while maintaining similar ratio.
    fn compress(&self, py: Python<'_>, table: &PyAny) -> PyResult<PyFluxBuffer> {
        use loom::traits::LoomCompressor;
        let batch = pyarrow_to_record_batch(py, table)?;
        let data = self.inner.compress(&batch).map_err(flux_err)?;
        Ok(PyFluxBuffer { data })
    }

    /// Number of column dictionaries currently in the cache.
    #[getter]
    fn dict_count(&self) -> usize {
        self.inner.dict_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Clear the dictionary cache, forcing re-training on the next batch.
    fn clear_cache(&self) {
        self.inner.dict_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    fn __repr__(&self) -> String {
        let n = self.inner.dict_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        format!(
            "FluxWriter(profile={:?}, cached_dicts={})",
            self.inner.profile, n
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase F — Schema-evolution Python surface (thin PyO3 skeleton)
// ─────────────────────────────────────────────────────────────────────────────

use loom::dtype::FluxDType;
use loom::txn::{
    EvolveOptions, FluxScan, FluxTable,
    schema::{DefaultValue, SchemaField, TableSchema},
};

// ── dtype string helpers ──────────────────────────────────────────────────────

fn parse_dtype(s: &str) -> PyResult<FluxDType> {
    serde_json::from_str::<FluxDType>(&format!("\"{}\"" , s)).map_err(|_| {
        PyValueError::new_err(format!(
            "unknown dtype '{}'. Valid values: uint8, uint16, uint32, uint64, \
             int8, int16, int32, int64, float32, float64, boolean, date32, \
             date64, timestamp_second, timestamp_millis, timestamp_micros, \
             timestamp_nanos, decimal128, utf8, large_utf8, binary, large_binary",
            s
        ))
    })
}

fn dtype_to_str(d: FluxDType) -> &'static str {
    match d {
        FluxDType::UInt8            => "uint8",
        FluxDType::UInt16           => "uint16",
        FluxDType::UInt32           => "uint32",
        FluxDType::UInt64           => "uint64",
        FluxDType::Int8             => "int8",
        FluxDType::Int16            => "int16",
        FluxDType::Int32            => "int32",
        FluxDType::Int64            => "int64",
        FluxDType::Float32          => "float32",
        FluxDType::Float64          => "float64",
        FluxDType::Boolean          => "boolean",
        FluxDType::Date32           => "date32",
        FluxDType::Date64           => "date64",
        FluxDType::TimestampSecond  => "timestamp_second",
        FluxDType::TimestampMillis  => "timestamp_millis",
        FluxDType::TimestampMicros  => "timestamp_micros",
        FluxDType::TimestampNanos   => "timestamp_nanos",
        FluxDType::Decimal128       => "decimal128",
        FluxDType::Utf8             => "utf8",
        FluxDType::LargeUtf8        => "large_utf8",
        FluxDType::Binary           => "binary",
        FluxDType::LargeBinary      => "large_binary",
        FluxDType::Offsets          => "offsets",
        FluxDType::StructContainer  => "struct",
        FluxDType::ListContainer    => "list",
        FluxDType::MapContainer     => "map",
    }
}

// ── PySchemaField ─────────────────────────────────────────────────────────────

/// A single field in a :class:`TableSchema`.
///
/// The ``field_id`` is the *immutable logical identifier* for the column —
/// names can change and numeric types can be promoted under the schema-evolution
/// compatibility rules, but ``field_id`` is stable for the table's lifetime.
///
/// Example::
///
///     field = fc.SchemaField(1, "user_id", "uint64", nullable=False)
///     field = fc.SchemaField(2, "revenue", "int64").with_int_default(0).with_doc("Cents")
#[pyclass(name = "SchemaField", module = "fluxcompress")]
#[derive(Clone)]
pub struct PySchemaField {
    pub inner: SchemaField,
}

#[pymethods]
impl PySchemaField {
    /// Create a new SchemaField.
    ///
    /// Args:
    ///     field_id:  Stable integer identifier, unique within the schema.
    ///     name:      User-facing column name.
    ///     dtype:     Column dtype as a string (e.g. ``"uint64"``, ``"utf8"``).
    ///     nullable:  Whether NULLs are permitted (default ``True``).
    #[new]
    #[pyo3(signature = (field_id, name, dtype, nullable = true))]
    fn new(field_id: u32, name: &str, dtype: &str, nullable: bool) -> PyResult<Self> {
        let d = parse_dtype(dtype)?;
        let mut field = SchemaField::new(field_id, name, d);
        field.nullable = nullable;
        Ok(Self { inner: field })
    }

    /// Stable logical identifier for this column.
    #[getter]
    fn field_id(&self) -> u32 { self.inner.field_id }

    /// Current user-facing column name.
    #[getter]
    fn name(&self) -> &str { &self.inner.name }

    /// Logical dtype at the current schema version (e.g. ``"uint64"``).
    #[getter]
    fn dtype(&self) -> &'static str { dtype_to_str(self.inner.dtype) }

    /// Whether NULLs are permitted in this column.
    #[getter]
    fn nullable(&self) -> bool { self.inner.nullable }

    /// Optional literal default value, or ``None``.
    #[getter]
    fn default_value(&self, py: Python<'_>) -> PyObject {
        match &self.inner.default {
            None => py.None(),
            Some(DefaultValue::Bool(b))   => b.into_py(py),
            Some(DefaultValue::Int(i))    => i.into_py(py),
            Some(DefaultValue::UInt(u))   => u.into_py(py),
            Some(DefaultValue::Float(f))  => f.into_py(py),
            Some(DefaultValue::String(s)) => s.clone().into_py(py),
        }
    }

    /// Optional documentation string, or ``None``.
    #[getter]
    fn doc(&self, py: Python<'_>) -> PyObject {
        match &self.inner.doc {
            None    => py.None(),
            Some(s) => s.clone().into_py(py),
        }
    }

    /// Return a copy of this field with nullability set.
    fn with_nullable(&self, nullable: bool) -> Self {
        Self { inner: self.inner.clone().with_nullable(nullable) }
    }

    /// Return a copy of this field with a literal integer default.
    fn with_int_default(&self, value: i64) -> Self {
        Self { inner: self.inner.clone().with_default(DefaultValue::Int(value)) }
    }

    /// Return a copy of this field with a literal unsigned-integer default.
    fn with_uint_default(&self, value: u64) -> Self {
        Self { inner: self.inner.clone().with_default(DefaultValue::UInt(value)) }
    }

    /// Return a copy of this field with a literal float default.
    fn with_float_default(&self, value: f64) -> Self {
        Self { inner: self.inner.clone().with_default(DefaultValue::Float(value)) }
    }

    /// Return a copy of this field with a literal boolean default.
    fn with_bool_default(&self, value: bool) -> Self {
        Self { inner: self.inner.clone().with_default(DefaultValue::Bool(value)) }
    }

    /// Return a copy of this field with a literal string default.
    fn with_str_default(&self, value: &str) -> Self {
        Self { inner: self.inner.clone().with_default(DefaultValue::String(value.to_string())) }
    }

    /// Return a copy of this field with a documentation string.
    fn with_doc(&self, doc: &str) -> Self {
        Self { inner: self.inner.clone().with_doc(doc) }
    }

    fn __repr__(&self) -> String {
        format!(
            "SchemaField(field_id={}, name={:?}, dtype={:?}, nullable={})",
            self.inner.field_id,
            self.inner.name,
            dtype_to_str(self.inner.dtype),
            self.inner.nullable,
        )
    }
}

// ── PyTableSchema ─────────────────────────────────────────────────────────────

/// The logical schema of a :class:`FluxTable` at a given schema version.
///
/// Example::
///
///     schema = fc.TableSchema([
///         fc.SchemaField(1, "id",   "uint64", nullable=False),
///         fc.SchemaField(2, "name", "utf8"),
///     ])
#[pyclass(name = "TableSchema", module = "fluxcompress")]
#[derive(Clone)]
pub struct PyTableSchema {
    pub inner: TableSchema,
}

#[pymethods]
impl PyTableSchema {
    /// Create a TableSchema from a list of :class:`SchemaField` objects.
    #[new]
    fn new(fields: Vec<PyRef<PySchemaField>>) -> Self {
        let inner_fields: Vec<SchemaField> = fields.iter()
            .map(|f| f.inner.clone())
            .collect();
        Self { inner: TableSchema::new(inner_fields) }
    }

    /// Monotonically increasing schema version identifier.
    #[getter]
    fn schema_id(&self) -> u32 { self.inner.schema_id }

    /// Parent schema id in the evolution chain (``None`` for the first schema).
    #[getter]
    fn parent_schema_id(&self, py: Python<'_>) -> PyObject {
        match self.inner.parent_schema_id {
            None    => py.None(),
            Some(p) => p.into_py(py),
        }
    }

    /// Fields in user-visible column order.
    #[getter]
    fn fields(&self) -> Vec<PySchemaField> {
        self.inner.fields.iter()
            .map(|f| PySchemaField { inner: f.clone() })
            .collect()
    }

    /// Optional human-readable change summary (``None`` if unset).
    #[getter]
    fn change_summary(&self, py: Python<'_>) -> PyObject {
        match &self.inner.change_summary {
            None    => py.None(),
            Some(s) => s.clone().into_py(py),
        }
    }

    /// Look up a field by its stable ``field_id``. Returns ``None`` if not found.
    fn field_by_id(&self, field_id: u32, py: Python<'_>) -> PyObject {
        match self.inner.field_by_id(field_id) {
            None    => py.None(),
            Some(f) => PySchemaField { inner: f.clone() }.into_py(py),
        }
    }

    /// Look up a field by its current display name. Returns ``None`` if not found.
    fn field_by_name(&self, name: &str, py: Python<'_>) -> PyObject {
        match self.inner.field_by_name(name) {
            None    => py.None(),
            Some(f) => PySchemaField { inner: f.clone() }.into_py(py),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TableSchema(schema_id={}, fields={})",
            self.inner.schema_id,
            self.inner.fields.len(),
        )
    }
}

// ── PyEvolveOptions ───────────────────────────────────────────────────────────

/// Options controlling schema-evolution behaviour.
///
/// Example::
///
///     # Default: no nullability tightening allowed
///     opts = fc.EvolveOptions()
///
///     # Opt into Phase D null tightening (requires manifest-derived proof)
///     opts = fc.EvolveOptions(allow_null_tightening=True)
///     # Equivalent shortcut:
///     opts = fc.EvolveOptions.with_null_tightening()
#[pyclass(name = "EvolveOptions", module = "fluxcompress")]
#[derive(Clone)]
pub struct PyEvolveOptions {
    pub inner: EvolveOptions,
}

#[pymethods]
impl PyEvolveOptions {
    /// Create EvolveOptions.
    ///
    /// Args:
    ///     allow_null_tightening: If ``True``, permit nullable→non-nullable
    ///         transitions when manifest-derived proof shows no NULLs exist.
    #[new]
    #[pyo3(signature = (allow_null_tightening = false))]
    fn new(allow_null_tightening: bool) -> Self {
        Self {
            inner: EvolveOptions { allow_null_tightening },
        }
    }

    /// Shortcut that sets ``allow_null_tightening=True``.
    #[staticmethod]
    fn with_null_tightening() -> Self {
        Self { inner: EvolveOptions::with_null_tightening() }
    }

    /// Whether nullable→non-nullable tightening is permitted.
    #[getter]
    fn allow_null_tightening(&self) -> bool { self.inner.allow_null_tightening }

    fn __repr__(&self) -> String {
        format!("EvolveOptions(allow_null_tightening={})", self.inner.allow_null_tightening)
    }
}

// ── PyFluxScan ────────────────────────────────────────────────────────────────

/// Streaming iterator over the live files of a :class:`FluxTable`.
///
/// Returned by :meth:`FluxTable.scan`. Yields one ``pyarrow.Table`` per
/// live file, projected to the current logical schema.
///
/// Example::
///
///     for batch in table.scan():
///         print(batch.num_rows)
#[pyclass(name = "FluxScan", module = "fluxcompress")]
pub struct PyFluxScan {
    inner: std::cell::RefCell<Option<FluxScan>>,
}

#[pymethods]
impl PyFluxScan {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let mut guard = self.inner.borrow_mut();
        let scan = match guard.as_mut() {
            None => return Ok(None),
            Some(s) => s,
        };
        match scan.next() {
            None => {
                // Exhausted — drop the scan to free resources.
                *guard = None;
                Ok(None)
            }
            Some(Ok(batch)) => {
                let py_table = record_batch_to_pyarrow(py, &batch)?;
                Ok(Some(py_table))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Number of live files remaining (including the current position).
    #[getter]
    fn remaining(&self) -> usize {
        self.inner.borrow()
            .as_ref()
            .map(|s| s.remaining())
            .unwrap_or(0)
    }
}

// ── PyFluxTable ───────────────────────────────────────────────────────────────

/// A versioned columnar table backed by a ``.fluxtable/`` directory.
///
/// Provides schema-evolution-aware append, scan, and time-travel operations
/// via an immutable transaction log.
///
/// Example::
///
///     import fluxcompress as fc
///     import pyarrow as pa
///
///     # Open (or create) a table
///     tbl = fc.FluxTable("my_table.fluxtable")
///
///     # Declare an initial schema
///     schema = fc.TableSchema([
///         fc.SchemaField(1, "id",  "uint64", nullable=False),
///         fc.SchemaField(2, "val", "int64"),
///     ])
///     tbl.evolve_schema(schema)
///
///     # Append data
///     buf = fc.compress(pa.table({"id": [1, 2], "val": [10, 20]}))
///     tbl.append(buf)
///
///     # Scan all files projected to the current schema
///     for batch in tbl.scan():
///         print(batch)
#[pyclass(name = "FluxTable", module = "fluxcompress")]
pub struct PyFluxTable {
    inner: FluxTable,
}

#[pymethods]
impl PyFluxTable {
    /// Open (or create) a FluxTable at ``path``.
    ///
    /// The directory is created if it does not exist yet.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        FluxTable::open(path)
            .map(|t| Self { inner: t })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Open (or create) a FluxTable — static alias for the constructor.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        PyFluxTable::new(path)
    }

    /// Append a :class:`FluxBuffer` to the table.
    ///
    /// Writes the compressed data as a new ``.flux`` file and creates a log entry.
    ///
    /// Returns:
    ///     ``int`` — the version number of the new log entry.
    fn append(&self, buf: PyRef<PyFluxBuffer>) -> PyResult<u64> {
        self.inner.append(&buf.data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Evolve the table's logical schema (no data rewrite).
    ///
    /// Validates the transition and writes a ``SchemaChange`` log entry.
    /// Nullability tightening is not permitted through this entry point
    /// (use :meth:`evolve_schema_with_options` with manifest proof).
    ///
    /// Args:
    ///     schema: The new :class:`TableSchema`.
    ///
    /// Returns:
    ///     ``int`` — the version number of the schema-change log entry.
    fn evolve_schema(&self, schema: &PyTableSchema) -> PyResult<u64> {
        self.inner
            .evolve_schema(schema.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Evolve the schema with explicit :class:`EvolveOptions`.
    ///
    /// Args:
    ///     schema: The new :class:`TableSchema`.
    ///     opts:   :class:`EvolveOptions` controlling allowed transitions.
    ///
    /// Returns:
    ///     ``int`` — the version number of the schema-change log entry.
    fn evolve_schema_with_options(
        &self,
        schema: &PyTableSchema,
        opts: &PyEvolveOptions,
    ) -> PyResult<u64> {
        self.inner
            .evolve_schema_with_options(schema.inner.clone(), opts.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Start a schema-evolution-aware streaming scan over all live files.
    ///
    /// Returns a :class:`FluxScan` iterator that yields one
    /// ``pyarrow.Table`` per live file, projected to the current schema.
    ///
    /// Raises ``RuntimeError`` if no schema has been declared yet.
    fn scan(&self) -> PyResult<PyFluxScan> {
        let scan = self.inner.scan()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyFluxScan {
            inner: std::cell::RefCell::new(Some(scan)),
        })
    }

    /// List absolute paths of all live data files at the latest version.
    fn live_files(&self) -> PyResult<Vec<String>> {
        self.inner
            .live_files()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            .map(|paths| {
                paths.iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect()
            })
    }

    /// Return the version number of the latest log entry, or ``None`` if
    /// the log is empty.
    fn current_version(&self, py: Python<'_>) -> PyResult<PyObject> {
        let log = self.inner.read_log()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(match log.last() {
            None    => py.None(),
            Some(e) => e.version.into_py(py),
        })
    }

    /// Return a ``dict`` mapping column names to their stable ``field_id``
    /// values for the current schema. Returns an empty dict when no schema
    /// has been declared yet.
    fn field_ids_for_current_schema(
        &self,
    ) -> PyResult<std::collections::HashMap<String, u32>> {
        self.inner
            .field_ids_for_current_schema()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the current :class:`TableSchema`, or ``None`` if no schema
    /// has been declared yet.
    fn current_schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        let snap = self.inner.snapshot()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let schema = snap.current_schema_id()
            .and_then(|sid| snap.schema_chain.get(sid).cloned());
        Ok(match schema {
            None    => py.None(),
            Some(s) => PyTableSchema { inner: s }.into_py(py),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "FluxTable({:?})",
            self.inner.log_dir()
                .parent()
                .unwrap_or(std::path::Path::new("?"))
                .display()
        )
    }
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
        "brotli"   => Ok(loom::CompressionProfile::Brotli),
        other      => Err(PyValueError::new_err(format!(
            "unknown profile '{other}'. Use: speed|balanced|archive|brotli"
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
    // Core classes
    m.add_class::<PyPredicate>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PyBlockInfo>()?;
    m.add_class::<PyFileInfo>()?;
    m.add_class::<PyFluxBuffer>()?;
    m.add_class::<PyFluxBatchReader>()?;

    // Stateful writer with dict cache
    m.add_class::<PyFluxWriter>()?;

    // Phase F: schema-evolution classes
    m.add_class::<PySchemaField>()?;
    m.add_class::<PyTableSchema>()?;
    m.add_class::<PyEvolveOptions>()?;
    m.add_class::<PyFluxScan>()?;
    m.add_class::<PyFluxTable>()?;

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
