// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Format-agnostic Arrow `RecordBatch` I/O.
//!
//! Supports the following file formats, dispatched by extension:
//!
//! | Extension              | Read | Write | Backend            |
//! |------------------------|------|-------|--------------------|
//! | `.csv`                 |  ✓   |  ✓    | `arrow::csv`       |
//! | `.tsv`, `.tab`         |  ✓   |  ✓    | `arrow::csv` (tab) |
//! | `.json`                |  ✓   |  ✓    | `arrow::json`      |
//! | `.ndjson`, `.jsonl`    |  ✓   |  ✓    | `arrow::json`      |
//! | `.parquet`, `.pq`      |  ✓   |  ✓    | `parquet::arrow`   |
//! | `.arrow`, `.ipc`       |  ✓   |  ✓    | `arrow::ipc`       |
//! | `.feather`             |  ✓   |  ✓    | `arrow::ipc`       |
//! | `.orc`                 |  ✓   |  ✓    | `orc-rust`         |
//! | `.xlsx`                |  ✓   |  ✓    | calamine + xlsxwr. |
//! | `.xls`, `.xlsm`, `.ods`|  ✓   |  ✗    | calamine           |
//!
//! Type inference for CSV / JSON / Excel is done from a sample of the
//! first 100 rows.

use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use arrow_array::builder::*;
use arrow_array::*;
use arrow_schema::{DataType, Field, Schema, SchemaRef};

// ─────────────────────────────────────────────────────────────────────────────
// Format detection
// ─────────────────────────────────────────────────────────────────────────────

/// All file formats `fluxcapacitor` knows how to read and/or write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Comma-separated values (RFC 4180-ish). Schema inferred.
    Csv,
    /// Tab-separated values (`\t`). Schema inferred.
    Tsv,
    /// Newline-separated JSON objects (NDJSON / JSONL).
    NdJson,
    /// JSON array of objects: `[{...}, {...}]`. We treat as NDJSON-equivalent
    /// after a quick rewrite. (Arrow's JSON reader operates on NDJSON.)
    Json,
    /// Apache Parquet.
    Parquet,
    /// Arrow IPC file format (a.k.a. Feather v2).
    ArrowIpc,
    /// Apache ORC.
    Orc,
    /// Microsoft Excel 2007+ (.xlsx). Read + write.
    Xlsx,
    /// Microsoft Excel 97-2003 (.xls). Read-only.
    Xls,
    /// Microsoft Excel macro-enabled (.xlsm). Read-only.
    Xlsm,
    /// OpenDocument Spreadsheet (.ods). Read-only.
    Ods,
}

impl FileFormat {
    /// Detect a format from a file path's extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("file {:?} has no extension", path))?
            .to_ascii_lowercase();
        Self::from_extension(&ext).with_context(|| {
            format!(
                "unsupported file format for {:?} (extension {:?})",
                path, ext
            )
        })
    }

    /// Detect a format from a bare extension (without the leading dot).
    pub fn from_extension(ext: &str) -> Result<Self> {
        Ok(match ext {
            "csv" => Self::Csv,
            "tsv" | "tab" => Self::Tsv,
            "ndjson" | "jsonl" => Self::NdJson,
            "json" => Self::Json,
            "parquet" | "pq" => Self::Parquet,
            "arrow" | "ipc" | "feather" => Self::ArrowIpc,
            "orc" => Self::Orc,
            "xlsx" => Self::Xlsx,
            "xls" => Self::Xls,
            "xlsm" => Self::Xlsm,
            "ods" => Self::Ods,
            _ => bail!("unknown extension '{ext}'"),
        })
    }

    /// Whether `save_batches` accepts this format.
    pub fn supports_write(self) -> bool {
        !matches!(self, Self::Xls | Self::Xlsm | Self::Ods)
    }

    /// Human-readable label used in error messages and benchmark output.
    pub fn label(self) -> &'static str {
        match self {
            Self::Csv => "CSV",
            Self::Tsv => "TSV",
            Self::NdJson => "NDJSON",
            Self::Json => "JSON",
            Self::Parquet => "Parquet",
            Self::ArrowIpc => "Arrow IPC",
            Self::Orc => "ORC",
            Self::Xlsx => "Excel (xlsx)",
            Self::Xls => "Excel (xls)",
            Self::Xlsm => "Excel (xlsm)",
            Self::Ods => "OpenDocument (ods)",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level read / write entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Load all `RecordBatch`es from a file. Format is detected from the
/// extension of `path`.
pub fn load_batches(path: &Path) -> Result<Vec<RecordBatch>> {
    let format = FileFormat::from_path(path)?;
    match format {
        FileFormat::Csv => load_csv(path, b','),
        FileFormat::Tsv => load_csv(path, b'\t'),
        FileFormat::NdJson => load_ndjson(path),
        FileFormat::Json => load_json_array(path),
        FileFormat::Parquet => load_parquet(path),
        FileFormat::ArrowIpc => load_arrow_ipc(path),
        FileFormat::Orc => load_orc(path),
        FileFormat::Xlsx | FileFormat::Xls | FileFormat::Xlsm | FileFormat::Ods => load_excel(path),
    }
}

/// Persist `batches` to disk in the format implied by `path`'s extension.
pub fn save_batches(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    if batches.is_empty() {
        bail!("save_batches: no batches to write");
    }
    let format = FileFormat::from_path(path)?;
    if !format.supports_write() {
        bail!(
            "format {} is read-only; choose a writable format like csv/parquet/xlsx",
            format.label()
        );
    }
    match format {
        FileFormat::Csv => save_csv(path, batches, b','),
        FileFormat::Tsv => save_csv(path, batches, b'\t'),
        FileFormat::NdJson => save_ndjson(path, batches),
        FileFormat::Json => save_json_array(path, batches),
        FileFormat::Parquet => save_parquet(path, batches),
        FileFormat::ArrowIpc => save_arrow_ipc(path, batches),
        FileFormat::Orc => save_orc(path, batches),
        FileFormat::Xlsx => save_xlsx(path, batches),
        _ => unreachable!("read-only formats handled above"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV / TSV
// ─────────────────────────────────────────────────────────────────────────────

fn load_csv(path: &Path, delimiter: u8) -> Result<Vec<RecordBatch>> {
    use arrow::csv::ReaderBuilder;
    use arrow::csv::reader::Format;

    let file = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let mut reader = BufReader::new(file);

    let format = Format::default()
        .with_header(true)
        .with_delimiter(delimiter);
    let (schema, _records_read) = format
        .infer_schema(&mut reader, Some(100))
        .with_context(|| format!("inferring schema for {:?}", path))?;
    reader.seek(SeekFrom::Start(0))?;

    let csv_reader = ReaderBuilder::new(Arc::new(schema))
        .with_format(format)
        .build(reader)?;

    let mut batches = Vec::new();
    for b in csv_reader {
        batches.push(b?);
    }
    Ok(batches)
}

fn save_csv(path: &Path, batches: &[RecordBatch], delimiter: u8) -> Result<()> {
    use arrow::csv::WriterBuilder;
    let file = File::create(path).with_context(|| format!("creating {:?}", path))?;
    let mut writer = WriterBuilder::new()
        .with_header(true)
        .with_delimiter(delimiter)
        .build(BufWriter::new(file));
    for batch in batches {
        writer.write(batch)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON / NDJSON
// ─────────────────────────────────────────────────────────────────────────────

fn load_ndjson(path: &Path) -> Result<Vec<RecordBatch>> {
    use arrow::json::ReaderBuilder;
    use arrow::json::reader::infer_json_schema_from_seekable;

    let file = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let mut reader = BufReader::new(file);

    let (schema, _) = infer_json_schema_from_seekable(&mut reader, Some(100))
        .with_context(|| format!("inferring schema for {:?}", path))?;
    reader.seek(SeekFrom::Start(0))?;

    let json_reader = ReaderBuilder::new(Arc::new(schema)).build(reader)?;
    let mut batches = Vec::new();
    for b in json_reader {
        batches.push(b?);
    }
    Ok(batches)
}

/// Read a JSON document containing a top-level array of records, e.g.
/// `[{"a": 1}, {"a": 2}]`. Internally this is reshaped into NDJSON.
fn load_json_array(path: &Path) -> Result<Vec<RecordBatch>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading {:?}", path))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing JSON {:?}", path))?;

    let array = match parsed {
        serde_json::Value::Array(items) => items,
        // Already NDJSON-shaped or a bare object — fall back to NDJSON path
        // by writing the raw string to a temp file and recursing.
        _ => return load_ndjson(path),
    };

    // Re-emit as NDJSON into a temp file. Cheap because the data was already in
    // memory once.
    let tmp = tempfile::NamedTempFile::new()?;
    {
        let mut w = BufWriter::new(tmp.as_file());
        for item in &array {
            serde_json::to_writer(&mut w, item)?;
            writeln!(&mut w)?;
        }
        w.flush()?;
    }
    load_ndjson(tmp.path())
}

fn save_ndjson(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    use arrow::json::LineDelimitedWriter;
    let file = File::create(path).with_context(|| format!("creating {:?}", path))?;
    let mut writer = LineDelimitedWriter::new(BufWriter::new(file));
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(())
}

fn save_json_array(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    use arrow::json::ArrayWriter;
    let file = File::create(path).with_context(|| format!("creating {:?}", path))?;
    let mut writer = ArrayWriter::new(BufWriter::new(file));
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Parquet
// ─────────────────────────────────────────────────────────────────────────────

fn load_parquet(path: &Path) -> Result<Vec<RecordBatch>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut batches = Vec::new();
    for b in reader {
        batches.push(b?);
    }
    Ok(batches)
}

fn save_parquet(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    let schema = batches[0].schema();
    let file = File::create(path).with_context(|| format!("creating {:?}", path))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrow IPC
// ─────────────────────────────────────────────────────────────────────────────

fn load_arrow_ipc(path: &Path) -> Result<Vec<RecordBatch>> {
    use arrow::ipc::reader::FileReader;
    let file = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let reader = FileReader::try_new(file, None)?;
    let mut batches = Vec::new();
    for b in reader {
        batches.push(b?);
    }
    Ok(batches)
}

fn save_arrow_ipc(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    use arrow::ipc::writer::FileWriter;
    let schema = batches[0].schema();
    let file = File::create(path).with_context(|| format!("creating {:?}", path))?;
    let mut writer = FileWriter::try_new(file, schema.as_ref())?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ORC
// ─────────────────────────────────────────────────────────────────────────────

fn load_orc(path: &Path) -> Result<Vec<RecordBatch>> {
    let file = File::open(path).with_context(|| format!("opening {:?}", path))?;
    let reader = orc_rust::ArrowReaderBuilder::try_new(file)?.build();
    let mut batches = Vec::new();
    for b in reader {
        batches.push(b?);
    }
    Ok(batches)
}

fn save_orc(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    let schema = batches[0].schema();
    let file = File::create(path).with_context(|| format!("creating {:?}", path))?;
    let mut writer = orc_rust::ArrowWriterBuilder::new(file, schema).try_build()?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Excel (xlsx, xls, xlsm, ods)
// ─────────────────────────────────────────────────────────────────────────────

fn load_excel(path: &Path) -> Result<Vec<RecordBatch>> {
    use calamine::{Data, Range, Reader, open_workbook_auto};

    let mut workbook =
        open_workbook_auto(path).with_context(|| format!("opening Excel workbook {:?}", path))?;
    let sheet_names = workbook.sheet_names().to_vec();
    let first = sheet_names
        .first()
        .ok_or_else(|| anyhow!("workbook {:?} contains no sheets", path))?;
    let range: Range<Data> = workbook
        .worksheet_range(first)
        .with_context(|| format!("reading worksheet '{}'", first))?;

    let rows: Vec<&[Data]> = range.rows().collect();
    if rows.is_empty() {
        bail!("worksheet '{}' is empty", first);
    }

    // First row → header.
    let headers: Vec<String> = rows[0]
        .iter()
        .enumerate()
        .map(|(i, c)| match c {
            Data::String(s) if !s.is_empty() => s.clone(),
            _ => format!("col_{i}"),
        })
        .collect();
    let n_cols = headers.len();
    let body: &[&[Data]] = &rows[1..];

    // Per-column type inference.
    let mut col_types = vec![ColumnKind::Empty; n_cols];
    for row in body.iter().take(100) {
        for (i, cell) in row.iter().enumerate().take(n_cols) {
            col_types[i] = col_types[i].promote(ColumnKind::from_cell(cell));
        }
    }

    let mut fields = Vec::with_capacity(n_cols);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(n_cols);

    for (i, kind) in col_types.iter().enumerate() {
        let dtype = kind.arrow_type();
        let array = build_excel_column(*kind, body, i, n_cols);
        fields.push(Field::new(&headers[i], dtype, true));
        columns.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns)?;
    Ok(vec![batch])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColumnKind {
    Empty,
    Bool,
    Int,
    Float,
    String,
}

impl ColumnKind {
    fn from_cell(c: &calamine::Data) -> Self {
        use calamine::Data;
        match c {
            Data::Empty | Data::Error(_) => Self::Empty,
            Data::Bool(_) => Self::Bool,
            Data::Int(_) => Self::Int,
            Data::Float(f) if f.fract() == 0.0 && f.is_finite() => Self::Int,
            Data::Float(_) => Self::Float,
            Data::DateTime(_) => Self::Float,
            Data::String(_) | Data::DateTimeIso(_) | Data::DurationIso(_) => Self::String,
        }
    }

    /// Type promotion lattice: Empty < Bool < Int < Float < String.
    fn promote(self, other: Self) -> Self {
        use ColumnKind::*;
        match (self, other) {
            (a, Empty) | (Empty, a) => a,
            (String, _) | (_, String) => String,
            (Float, _) | (_, Float) => Float,
            (Int, Bool) | (Bool, Int) => Int,
            (a, b) if a == b => a,
            _ => String,
        }
    }

    fn arrow_type(self) -> DataType {
        match self {
            Self::Empty | Self::String => DataType::Utf8,
            Self::Bool => DataType::Boolean,
            Self::Int => DataType::Int64,
            Self::Float => DataType::Float64,
        }
    }
}

fn build_excel_column(
    kind: ColumnKind,
    body: &[&[calamine::Data]],
    col: usize,
    n_cols: usize,
) -> ArrayRef {
    use calamine::Data;

    match kind {
        ColumnKind::Bool => {
            let mut b = BooleanBuilder::with_capacity(body.len());
            for row in body {
                let cell = row.get(col).unwrap_or(&Data::Empty);
                match cell {
                    Data::Bool(v) => b.append_value(*v),
                    Data::Empty | Data::Error(_) => b.append_null(),
                    Data::String(s) => b.append_value(s.eq_ignore_ascii_case("true")),
                    Data::Int(v) => b.append_value(*v != 0),
                    Data::Float(v) => b.append_value(*v != 0.0),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnKind::Int => {
            let mut b = Int64Builder::with_capacity(body.len());
            for row in body {
                let cell = row.get(col).unwrap_or(&Data::Empty);
                match cell {
                    Data::Int(v) => b.append_value(*v),
                    Data::Float(f) if f.is_finite() => b.append_value(*f as i64),
                    Data::Bool(v) => b.append_value(if *v { 1 } else { 0 }),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnKind::Float => {
            let mut b = Float64Builder::with_capacity(body.len());
            for row in body {
                let cell = row.get(col).unwrap_or(&Data::Empty);
                match cell {
                    Data::Float(v) => b.append_value(*v),
                    Data::Int(v) => b.append_value(*v as f64),
                    Data::DateTime(d) => b.append_value(d.as_f64()),
                    Data::Bool(v) => b.append_value(if *v { 1.0 } else { 0.0 }),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        ColumnKind::Empty | ColumnKind::String => {
            let mut b = StringBuilder::with_capacity(body.len(), body.len() * 8);
            for row in body {
                let cell = row.get(col).unwrap_or(&Data::Empty);
                match cell {
                    Data::Empty | Data::Error(_) => b.append_null(),
                    Data::String(s) => b.append_value(s),
                    Data::Int(v) => b.append_value(v.to_string()),
                    Data::Float(v) => b.append_value(v.to_string()),
                    Data::Bool(v) => b.append_value(v.to_string()),
                    Data::DateTime(v) => b.append_value(v.as_f64().to_string()),
                    Data::DateTimeIso(s) => b.append_value(s),
                    Data::DurationIso(s) => b.append_value(s),
                }
            }
            // Silence the unused-arg lint when n_cols is irrelevant.
            let _ = n_cols;
            Arc::new(b.finish())
        }
    }
}

fn save_xlsx(path: &Path, batches: &[RecordBatch]) -> Result<()> {
    use rust_xlsxwriter::Workbook;

    let mut wb = Workbook::new();
    let ws = wb.add_worksheet();

    let schema: SchemaRef = batches[0].schema();

    // Write header row.
    for (col_idx, field) in schema.fields().iter().enumerate() {
        ws.write_string(0, col_idx as u16, field.name())
            .map_err(|e| anyhow!("xlsx header write: {e}"))?;
    }

    let mut row_offset: u32 = 1;
    for batch in batches {
        let n_rows = batch.num_rows() as u32;
        for (col_idx, column) in batch.columns().iter().enumerate() {
            let col_idx_u16 = col_idx as u16;
            write_xlsx_column(ws, column, row_offset, col_idx_u16)?;
        }
        row_offset += n_rows;
    }

    wb.save(path).map_err(|e| anyhow!("xlsx save: {e}"))?;
    Ok(())
}

fn write_xlsx_column(
    ws: &mut rust_xlsxwriter::Worksheet,
    column: &ArrayRef,
    row_offset: u32,
    col_idx: u16,
) -> Result<()> {
    use arrow_array::cast::AsArray;

    macro_rules! write_numeric {
        ($arr:expr, $iter:expr) => {{
            let arr = $arr;
            for (i, v) in $iter.enumerate() {
                let row = row_offset + i as u32;
                if arr.is_null(i) {
                    ws.write_blank(row, col_idx, &Default::default())
                        .map_err(|e| anyhow!("xlsx blank write: {e}"))?;
                } else {
                    ws.write_number(row, col_idx, v as f64)
                        .map_err(|e| anyhow!("xlsx number write: {e}"))?;
                }
            }
        }};
    }

    match column.data_type() {
        DataType::Boolean => {
            let arr = column.as_boolean();
            for i in 0..arr.len() {
                let row = row_offset + i as u32;
                if arr.is_null(i) {
                    ws.write_blank(row, col_idx, &Default::default())
                        .map_err(|e| anyhow!("xlsx blank write: {e}"))?;
                } else {
                    ws.write_boolean(row, col_idx, arr.value(i))
                        .map_err(|e| anyhow!("xlsx bool write: {e}"))?;
                }
            }
        }
        DataType::Int8 => {
            let arr: &Int8Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::Int16 => {
            let arr: &Int16Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::Int32 => {
            let arr: &Int32Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::Int64 => {
            let arr: &Int64Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::UInt8 => {
            let arr: &UInt8Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::UInt16 => {
            let arr: &UInt16Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::UInt32 => {
            let arr: &UInt32Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::UInt64 => {
            let arr: &UInt64Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::Float32 => {
            let arr: &Float32Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::Float64 => {
            let arr: &Float64Array = column.as_primitive();
            write_numeric!(arr, arr.iter().map(|x| x.unwrap_or_default()));
        }
        DataType::Utf8 => {
            let arr: &StringArray = column.as_string();
            for i in 0..arr.len() {
                let row = row_offset + i as u32;
                if arr.is_null(i) {
                    ws.write_blank(row, col_idx, &Default::default())
                        .map_err(|e| anyhow!("xlsx blank write: {e}"))?;
                } else {
                    ws.write_string(row, col_idx, arr.value(i))
                        .map_err(|e| anyhow!("xlsx string write: {e}"))?;
                }
            }
        }
        DataType::LargeUtf8 => {
            let arr: &LargeStringArray = column.as_string();
            for i in 0..arr.len() {
                let row = row_offset + i as u32;
                if arr.is_null(i) {
                    ws.write_blank(row, col_idx, &Default::default())
                        .map_err(|e| anyhow!("xlsx blank write: {e}"))?;
                } else {
                    ws.write_string(row, col_idx, arr.value(i))
                        .map_err(|e| anyhow!("xlsx string write: {e}"))?;
                }
            }
        }
        other => {
            // Fall-back: stringify via Arrow's display formatter.
            use arrow::util::display::{ArrayFormatter, FormatOptions};
            let opts = FormatOptions::default();
            let formatter = ArrayFormatter::try_new(column.as_ref(), &opts)
                .with_context(|| format!("xlsx fallback formatter for {other:?}"))?;
            for i in 0..column.len() {
                let row = row_offset + i as u32;
                if column.is_null(i) {
                    ws.write_blank(row, col_idx, &Default::default())
                        .map_err(|e| anyhow!("xlsx blank write: {e}"))?;
                } else {
                    let s = formatter.value(i).to_string();
                    ws.write_string(row, col_idx, &s)
                        .map_err(|e| anyhow!("xlsx fallback write: {e}"))?;
                }
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_format_from_extension() {
        assert_eq!(FileFormat::from_extension("csv").unwrap(), FileFormat::Csv);
        assert_eq!(FileFormat::from_extension("tsv").unwrap(), FileFormat::Tsv);
        assert_eq!(
            FileFormat::from_extension("ndjson").unwrap(),
            FileFormat::NdJson
        );
        assert_eq!(
            FileFormat::from_extension("json").unwrap(),
            FileFormat::Json
        );
        assert_eq!(
            FileFormat::from_extension("parquet").unwrap(),
            FileFormat::Parquet
        );
        assert_eq!(
            FileFormat::from_extension("arrow").unwrap(),
            FileFormat::ArrowIpc
        );
        assert_eq!(FileFormat::from_extension("orc").unwrap(), FileFormat::Orc);
        assert_eq!(
            FileFormat::from_extension("xlsx").unwrap(),
            FileFormat::Xlsx
        );
        assert_eq!(FileFormat::from_extension("ods").unwrap(), FileFormat::Ods);
        assert!(FileFormat::from_extension("docx").is_err());
    }

    #[test]
    fn write_capability_matrix() {
        for f in [
            FileFormat::Csv,
            FileFormat::Tsv,
            FileFormat::Json,
            FileFormat::NdJson,
            FileFormat::Parquet,
            FileFormat::ArrowIpc,
            FileFormat::Orc,
            FileFormat::Xlsx,
        ] {
            assert!(f.supports_write(), "{:?} should be writable", f);
        }
        for f in [FileFormat::Xls, FileFormat::Xlsm, FileFormat::Ods] {
            assert!(!f.supports_write(), "{:?} should be read-only", f);
        }
    }
}
