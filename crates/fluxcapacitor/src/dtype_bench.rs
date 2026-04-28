// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Multi-datatype benchmark: Flux vs Parquet vs ORC.
//!
//! Tests compression across all major Arrow data types including complex
//! types (Struct, List, Map) that FluxCompress does not yet support.
//! This gives an honest comparison showing where FluxCompress wins and
//! where it needs work.

use anyhow::Result;
use std::fs::{self, File};
use std::sync::Arc;
use std::time::Instant;

use arrow::ipc::{reader::FileReader as IpcReader, writer::FileWriter as IpcWriter};
use arrow_array::builder::*;
use arrow_array::*;
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::ArrowWriter as PqWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression as PqCompression;
use parquet::file::properties::WriterProperties;

use loom::{
    CompressionProfile,
    compressors::flux_writer::FluxWriter,
    decompressors::flux_reader::FluxReader,
    traits::{LoomCompressor, LoomDecompressor, Predicate},
};

// ─────────────────────────────────────────────────────────────────────────────
// Data generators
// ─────────────────────────────────────────────────────────────────────────────

fn gen_uint64(n: usize) -> RecordBatch {
    let arr = UInt64Array::from((0..n as u64).collect::<Vec<_>>());
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "val",
            DataType::UInt64,
            false,
        )])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_int64(n: usize) -> RecordBatch {
    let arr = Int64Array::from((0..n as i64).map(|i| i - n as i64 / 2).collect::<Vec<_>>());
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("val", DataType::Int64, false)])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_float64(n: usize) -> RecordBatch {
    let arr = Float64Array::from((0..n).map(|i| i as f64 * 1.234567).collect::<Vec<_>>());
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "val",
            DataType::Float64,
            false,
        )])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_string(n: usize) -> RecordBatch {
    // Low cardinality strings (100 unique values).
    let strings: Vec<String> = (0..100).map(|i| format!("category_{i:03}")).collect();
    let arr = StringArray::from(
        (0..n)
            .map(|i| strings[i % 100].as_str())
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("val", DataType::Utf8, false)])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_date32(n: usize) -> RecordBatch {
    // Days since epoch (2020-01-01 = 18262).
    let arr = Date32Array::from(
        (0..n as i32)
            .map(|i| 18262 + (i % 3650))
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "val",
            DataType::Date32,
            false,
        )])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_timestamp(n: usize) -> RecordBatch {
    // Microsecond timestamps.
    let arr = TimestampMicrosecondArray::from(
        (0..n as i64)
            .map(|i| 1_700_000_000_000_000 + i * 1_000_000)
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "val",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            false,
        )])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_struct(n: usize) -> RecordBatch {
    let id_arr = UInt64Array::from((0..n as u64).collect::<Vec<_>>());
    let name_arr = StringArray::from(
        (0..n)
            .map(|i| format!("user_{}", i % 1000))
            .collect::<Vec<_>>(),
    );
    let age_arr = Int32Array::from((0..n as i32).map(|i| 20 + (i % 60)).collect::<Vec<_>>());

    let fields = Fields::from(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);
    let struct_arr = StructArray::from(vec![
        (
            Arc::new(Field::new("id", DataType::UInt64, false)),
            Arc::new(id_arr) as ArrayRef,
        ),
        (
            Arc::new(Field::new("name", DataType::Utf8, false)),
            Arc::new(name_arr) as ArrayRef,
        ),
        (
            Arc::new(Field::new("age", DataType::Int32, false)),
            Arc::new(age_arr) as ArrayRef,
        ),
    ]);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "val",
            DataType::Struct(fields),
            false,
        )])),
        vec![Arc::new(struct_arr)],
    )
    .unwrap()
}

fn gen_list(n: usize) -> RecordBatch {
    let mut builder = ListBuilder::new(Int64Builder::new());
    for i in 0..n {
        let len = 1 + (i % 5); // lists of 1-5 elements
        for j in 0..len {
            builder.values().append_value(i as i64 * 10 + j as i64);
        }
        builder.append(true);
    }
    let arr = builder.finish();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "val",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            false,
        )])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_map(n: usize) -> RecordBatch {
    let keys_builder = StringBuilder::new();
    let values_builder = Int64Builder::new();
    let mut builder = MapBuilder::new(None, keys_builder, values_builder);
    for i in 0..n {
        let entries = 1 + (i % 3);
        for j in 0..entries {
            builder.keys().append_value(format!("key_{j}"));
            builder.values().append_value(i as i64 + j as i64);
        }
        builder.append(true).unwrap();
    }
    let arr = builder.finish();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "val",
            arr.data_type().clone(),
            false,
        )])),
        vec![Arc::new(arr)],
    )
    .unwrap()
}

fn gen_mixed(n: usize) -> RecordBatch {
    // Realistic multi-type table.
    let user_id = UInt64Array::from((0..n as u64).collect::<Vec<_>>());
    let revenue = Float64Array::from(
        (0..n)
            .map(|i| (i as f64) * 37.5 % 99999.0)
            .collect::<Vec<_>>(),
    );
    let region = StringArray::from(
        (0..n)
            .map(|i| match i % 4 {
                0 => "North",
                1 => "South",
                2 => "East",
                _ => "West",
            })
            .collect::<Vec<_>>(),
    );
    let ts = TimestampMicrosecondArray::from(
        (0..n as i64)
            .map(|i| 1_700_000_000_000_000 + i * 1_000)
            .collect::<Vec<_>>(),
    );
    let active = BooleanArray::from((0..n).map(|i| i % 7 != 0).collect::<Vec<_>>());

    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::UInt64, false),
            Field::new("revenue", DataType::Float64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new(
                "timestamp",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("active", DataType::Boolean, false),
        ])),
        vec![
            Arc::new(user_id),
            Arc::new(revenue),
            Arc::new(region),
            Arc::new(ts),
            Arc::new(active),
        ],
    )
    .unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark runners
// ─────────────────────────────────────────────────────────────────────────────

struct BenchResult {
    format: String,
    size: usize,
    ratio: f64,
    comp_mbs: f64,
    decomp_mbs: f64,
    supported: bool,
}

fn raw_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|c| c.get_array_memory_size())
        .sum()
}

fn bench_flux(batch: &RecordBatch, raw: usize, profile: CompressionProfile) -> BenchResult {
    let name = match profile {
        CompressionProfile::Speed => "Flux (speed)",
        CompressionProfile::Balanced => "Flux (balanced)",
        CompressionProfile::Archive => "Flux (archive)",
        CompressionProfile::Brotli => "Flux (brotli)",
    };

    // FluxCompress now supports all Arrow types (numeric, string, nested).
    let all_supported = true;

    if !all_supported {
        return BenchResult {
            format: name.into(),
            size: 0,
            ratio: 0.0,
            comp_mbs: 0.0,
            decomp_mbs: 0.0,
            supported: false,
        };
    }

    let writer = FluxWriter::with_profile(profile);
    let t0 = Instant::now();
    let bytes = writer.compress(batch).unwrap();
    let c_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tmp = "/tmp/fluxbench_dtype.flux";
    fs::write(tmp, &bytes).unwrap();

    let reader = FluxReader::new("value");
    let t1 = Instant::now();
    let _ = reader
        .decompress_file_all(std::path::Path::new(tmp))
        .unwrap();
    let d_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let _ = fs::remove_file(tmp);

    BenchResult {
        format: name.into(),
        size: bytes.len(),
        ratio: raw as f64 / bytes.len() as f64,
        comp_mbs: mbs(raw, c_ms),
        decomp_mbs: mbs(raw, d_ms),
        supported: true,
    }
}

fn bench_parquet(batch: &RecordBatch, raw: usize, codec: &str) -> BenchResult {
    let tmp = format!("/tmp/fluxbench_dtype_{codec}.parquet");
    let compression = match codec {
        "snappy" => PqCompression::SNAPPY,
        "zstd" => PqCompression::ZSTD(Default::default()),
        _ => PqCompression::SNAPPY,
    };
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();

    let file = File::create(&tmp).unwrap();
    let t0 = Instant::now();
    let mut pw = PqWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    pw.write(batch).unwrap();
    pw.close().unwrap();
    let c_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let size = fs::metadata(&tmp).unwrap().len() as usize;

    let t1 = Instant::now();
    let file = File::open(&tmp).unwrap();
    let pr = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let _: Vec<RecordBatch> = pr.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    let d_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let _ = fs::remove_file(&tmp);

    BenchResult {
        format: format!("Parquet ({codec})"),
        size,
        ratio: raw as f64 / size as f64,
        comp_mbs: mbs(raw, c_ms),
        decomp_mbs: mbs(raw, d_ms),
        supported: true,
    }
}

fn bench_orc(batch: &RecordBatch, raw: usize) -> BenchResult {
    let tmp = "/tmp/fluxbench_dtype.orc";

    // ORC doesn't support all Arrow types (e.g., unsigned integers).
    // Catch panics gracefully.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let file = File::create(tmp).unwrap();
        let mut ow = orc_rust::ArrowWriterBuilder::new(file, batch.schema())
            .try_build()
            .unwrap();
        ow.write(batch).unwrap();
        ow.close().unwrap();
    }));

    if result.is_err() {
        let _ = fs::remove_file(tmp);
        return BenchResult {
            format: "ORC (zlib)".into(),
            size: 0,
            ratio: 0.0,
            comp_mbs: 0.0,
            decomp_mbs: 0.0,
            supported: false,
        };
    }

    // Re-time for accuracy.
    let _ = fs::remove_file(tmp);
    let file = File::create(tmp).unwrap();
    let t0 = Instant::now();
    let mut ow = orc_rust::ArrowWriterBuilder::new(file, batch.schema())
        .try_build()
        .unwrap();
    ow.write(batch).unwrap();
    ow.close().unwrap();
    let c_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let size = fs::metadata(tmp).unwrap().len() as usize;

    let read_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let file = File::open(tmp).unwrap();
        let reader = orc_rust::ArrowReaderBuilder::try_new(file).unwrap().build();
        let _: Vec<RecordBatch> = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    }));

    let d_ms = if read_result.is_ok() {
        let file = File::open(tmp).unwrap();
        let t1 = Instant::now();
        let reader = orc_rust::ArrowReaderBuilder::try_new(file).unwrap().build();
        let _: Vec<RecordBatch> = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
        t1.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };

    let _ = fs::remove_file(tmp);

    BenchResult {
        format: "ORC (zlib)".into(),
        size,
        ratio: raw as f64 / size as f64,
        comp_mbs: mbs(raw, c_ms),
        decomp_mbs: if d_ms > 0.0 { mbs(raw, d_ms) } else { 0.0 },
        supported: true,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn cmd_dtype_bench(rows: usize) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  FluxCapacitor — Multi-DataType Benchmark                       ║");
    println!("║  Flux vs Parquet vs ORC across all Arrow types                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("\n  Rows: {rows}\n");

    let tests: Vec<(&str, RecordBatch)> = vec![
        ("UInt64", gen_uint64(rows)),
        ("Int64", gen_int64(rows)),
        ("Float64", gen_float64(rows)),
        ("String", gen_string(rows)),
        ("Date32", gen_date32(rows)),
        ("Timestamp", gen_timestamp(rows)),
        ("Struct", gen_struct(rows)),
        ("List", gen_list(rows)),
        ("Map", gen_map(rows)),
        ("Mixed (5c)", gen_mixed(rows)),
    ];

    for (dtype, batch) in &tests {
        let raw = raw_bytes(batch);
        println!(
            "── {dtype} ({}, {} cols) ──",
            human(raw),
            batch.num_columns()
        );
        println!(
            "  {:<24} {:>10} {:>8} {:>10} {:>10}",
            "Format", "Size", "Ratio", "Comp MB/s", "Dec MB/s"
        );
        println!("  {}", "─".repeat(66));

        let results = vec![
            bench_flux(batch, raw, CompressionProfile::Archive),
            bench_flux(batch, raw, CompressionProfile::Balanced),
            bench_parquet(batch, raw, "zstd"),
            bench_parquet(batch, raw, "snappy"),
        ];

        for r in &results {
            if r.supported {
                println!(
                    "  {:<24} {:>10} {:>7.1}x {:>9.0} {:>9.0}",
                    r.format,
                    human(r.size),
                    r.ratio,
                    r.comp_mbs,
                    r.decomp_mbs,
                );
            } else {
                println!(
                    "  {:<24} {:>10} {:>8} {:>10} {:>10}",
                    r.format, "—", "N/A", "—", "—",
                );
            }
        }
        println!();
    }

    println!("  Legend: N/A = FluxCompress does not yet support this data type.");
    println!("         Struct/List/Map support is on the roadmap (v0.3).\n");
    println!("  ✓ Benchmark complete.");
    Ok(())
}

fn human(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn mbs(bytes: usize, ms: f64) -> f64 {
    if ms <= 0.0 {
        f64::INFINITY
    } else {
        (bytes as f64 / (1024.0 * 1024.0)) / (ms / 1000.0)
    }
}
