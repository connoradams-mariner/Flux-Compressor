// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Arrow IPC stream encode/decode helpers.
//!
//! The JNI boundary speaks raw bytes, so every multi-column exchange goes
//! through the Arrow IPC **stream** format (not the file format — no
//! seeking required).  The helpers here keep that serialisation detail in
//! one place rather than scattering it across every JNI entry point.
//!
//! ## Wire format
//! Arrow IPC stream: schema message → one or more record-batch messages →
//! end-of-stream marker.  The Java side uses the same format via the
//! `org.apache.arrow.vector.ipc.ArrowStreamReader` /
//! `ArrowStreamWriter` classes in `arrow-vector`.

use std::io::Cursor;
use std::sync::Arc;

use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_array::RecordBatch;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ─────────────────────────────────────────────────────────────────────────────
// IPC → RecordBatch
// ─────────────────────────────────────────────────────────────────────────────

/// Deserialise an Arrow IPC *stream* byte buffer into a single
/// [`RecordBatch`].
///
/// If the stream contains more than one record-batch message, all batches
/// are concatenated with [`arrow::compute::concat_batches`] before being
/// returned, which is the standard semantics Spark uses when writing a
/// partition as a single IPC payload.
///
/// Returns `Err` on malformed IPC, schema mismatch between batches, or an
/// empty stream with no schema.
pub fn batch_from_ipc(bytes: &[u8]) -> Result<RecordBatch> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None)?;
    let schema = reader.schema();

    let batches: std::result::Result<Vec<RecordBatch>, _> = reader.collect();
    let batches = batches?;

    match batches.len() {
        0 => Ok(RecordBatch::new_empty(schema)),
        1 => Ok(batches.into_iter().next().unwrap()),
        _ => {
            // Multiple batches — concatenate into one.
            let refs: Vec<&RecordBatch> = batches.iter().collect();
            Ok(arrow::compute::concat_batches(&schema, refs)?)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RecordBatch(es) → IPC
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise a single [`RecordBatch`] to Arrow IPC stream bytes.
pub fn batch_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut buf, batch.schema().as_ref())?;
        w.write(batch)?;
        w.finish()?;
    }
    Ok(buf)
}

/// Serialise a slice of [`RecordBatch`]es to Arrow IPC stream bytes.
///
/// All batches must share the same schema.  Returns an empty `Vec` when
/// `batches` is empty (the Java side treats an empty IPC stream as zero
/// rows).
pub fn batches_to_ipc(batches: &[RecordBatch]) -> Result<Vec<u8>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }
    let schema = batches[0].schema();
    let mut buf = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut buf, schema.as_ref())?;
        for batch in batches {
            w.write(batch)?;
        }
        w.finish()?;
    }
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (no JNI required)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};

    fn three_col_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id",    DataType::UInt64, false),
            Field::new("score", DataType::Int32,  false),
            Field::new("label", DataType::Utf8,   true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1u64, 2, 3])),
                Arc::new(Int32Array::from(vec![10i32, 20, 30])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn round_trip_single_batch() {
        let batch   = three_col_batch();
        let encoded = batch_to_ipc(&batch).unwrap();
        let decoded = batch_from_ipc(&encoded).unwrap();

        assert_eq!(decoded.num_rows(),    batch.num_rows());
        assert_eq!(decoded.num_columns(), batch.num_columns());

        // Check every column name round-trips.
        for (i, field) in batch.schema().fields().iter().enumerate() {
            assert_eq!(decoded.schema().field(i).name(), field.name());
        }

        // Spot-check the UInt64 column.
        let ids = decoded.column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(ids.values().to_vec(), vec![1u64, 2, 3]);
    }

    #[test]
    fn batches_to_ipc_then_back() {
        let b1 = three_col_batch();
        let b2 = three_col_batch();
        let encoded = batches_to_ipc(&[b1, b2]).unwrap();
        let decoded = batch_from_ipc(&encoded).unwrap();

        // Two batches of 3 rows each → concatenated to 6.
        assert_eq!(decoded.num_rows(), 6);
        assert_eq!(decoded.num_columns(), 3);
    }

    #[test]
    fn empty_batches_returns_empty_vec() {
        let bytes = batches_to_ipc(&[]).unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn single_empty_batch_round_trips() {
        let batch   = three_col_batch();
        let empty   = RecordBatch::new_empty(batch.schema());
        let encoded = batch_to_ipc(&empty).unwrap();
        let decoded = batch_from_ipc(&encoded).unwrap();
        assert_eq!(decoded.num_rows(), 0);
        assert_eq!(decoded.num_columns(), 3);
    }
}
