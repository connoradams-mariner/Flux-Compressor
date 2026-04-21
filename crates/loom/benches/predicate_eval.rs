// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

//! Criterion bench for `Predicate::eval_on_batch` — the row-level
//! filter primitive shared by the DELETE / UPDATE / MERGE roadmap.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench predicate_eval
//! ```

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use loom::traits::Predicate;

fn make_int_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    let values: Int64Array = (0..n as i64).collect();
    RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap()
}

fn bench_eval_on_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("predicate/eval_on_batch");

    for &rows in &[64_000usize, 524_288, 2_097_152] {
        let batch = make_int_batch(rows);
        g.throughput(Throughput::Elements(rows as u64));

        // 1. Simple `>` — fastest case, single-column single-op.
        g.bench_with_input(
            BenchmarkId::new("gt", rows),
            &batch,
            |b, batch| {
                let p = Predicate::GreaterThan { column: "x".into(), value: (rows as i128) / 2 };
                b.iter(|| black_box(&p).eval_on_batch(black_box(batch)).unwrap())
            },
        );

        // 2. `BETWEEN` — exercises AND under the hood.
        g.bench_with_input(
            BenchmarkId::new("between", rows),
            &batch,
            |b, batch| {
                let lo = (rows as i128) / 4;
                let hi = (rows as i128) * 3 / 4;
                let p = Predicate::Between { column: "x".into(), lo, hi };
                b.iter(|| black_box(&p).eval_on_batch(black_box(batch)).unwrap())
            },
        );

        // 3. Nested AND / OR — worst case for the recursive walker.
        g.bench_with_input(
            BenchmarkId::new("and_or_deep", rows),
            &batch,
            |b, batch| {
                let gt = Predicate::GreaterThan { column: "x".into(), value: 100 };
                let lt = Predicate::LessThan    { column: "x".into(), value: 10_000_000 };
                let eq = Predicate::Equal       { column: "x".into(), value: 42 };
                let p = Predicate::Or(
                    Box::new(Predicate::And(Box::new(gt), Box::new(lt))),
                    Box::new(eq),
                );
                b.iter(|| black_box(&p).eval_on_batch(black_box(batch)).unwrap())
            },
        );
    }

    g.finish();
}

criterion_group!(benches, bench_eval_on_batch);
criterion_main!(benches);
