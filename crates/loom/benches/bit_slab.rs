// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use loom::{
    SEGMENT_SIZE,
    bit_io::{BitReader, BitWriter},
    compressors::{bit_slab_compressor, delta_compressor, lz4_compressor, rle_compressor},
    decompressors::block_reader::decompress_block,
    loom_classifier::classify,
    simd,
};

fn sequential_u128(n: usize) -> Vec<u128> {
    (0u128..n as u128).collect()
}
fn constant_u128(n: usize) -> Vec<u128> {
    vec![42u128; n]
}
fn outlier_u128(n: usize) -> Vec<u128> {
    (0u128..n as u128)
        .map(|i| if i % 100 == 0 { u128::MAX } else { i % 256 })
        .collect()
}

fn bench_bitwriter(c: &mut Criterion) {
    let mut g = c.benchmark_group("bitwriter");
    for &w in &[8u8, 10, 16, 32] {
        let vals: Vec<u64> = (0u64..SEGMENT_SIZE as u64).collect();
        g.throughput(Throughput::Elements(vals.len() as u64));
        g.bench_with_input(BenchmarkId::new("write", w), &w, |b, &w| {
            b.iter(|| {
                let mut wr = BitWriter::new(w);
                for &v in black_box(&vals) {
                    wr.write_value(v).unwrap();
                }
                black_box(wr.finish())
            });
        });
    }
    g.finish();
}

fn bench_bitreader(c: &mut Criterion) {
    let mut g = c.benchmark_group("bitreader");
    for &w in &[8u8, 10, 16, 32] {
        let vals: Vec<u64> = (0u64..SEGMENT_SIZE as u64).collect();
        let mut wr = BitWriter::new(w);
        for &v in &vals {
            wr.write_value(v).unwrap();
        }
        let buf = wr.finish();
        g.throughput(Throughput::Elements(vals.len() as u64));
        g.bench_with_input(BenchmarkId::new("read", w), &w, |b, &w| {
            b.iter(|| {
                let mut r = BitReader::new(black_box(&buf), w);
                let mut s = 0u64;
                while let Some(v) = r.read_value() {
                    s = s.wrapping_add(v);
                }
                black_box(s)
            });
        });
    }
    g.finish();
}

fn bench_simd_unpack(c: &mut Criterion) {
    let mut g = c.benchmark_group("simd_unpack");
    for &w in &[8u8, 16, 32] {
        let vals: Vec<u64> = (0u64..SEGMENT_SIZE as u64).collect();
        let mut wr = BitWriter::new(w);
        for &v in &vals {
            wr.write_value(v).unwrap();
        }
        let buf = wr.finish();
        let mut out = vec![0u64; SEGMENT_SIZE];
        g.throughput(Throughput::Bytes(buf.len() as u64));
        g.bench_with_input(BenchmarkId::new("unpack", w), &w, |b, &w| {
            b.iter(|| {
                // Returning `black_box(&out)` from a `FnMut` closure that
                // captures `out` by reference is rejected by rustc 1.80+
                // ("captured variable cannot escape FnMut closure body").
                // The workaround: consume the side-effecting call in a
                // trivial `black_box(())` so no captured reference leaves
                // the closure — the compiler still can't elide the call
                // because `simd::unpack` writes through `&mut out`.
                simd::unpack(black_box(&buf), w, SEGMENT_SIZE, black_box(&mut out)).unwrap();
                black_box(());
            });
        });
    }
    g.finish();
}

fn bench_compress(c: &mut Criterion) {
    let mut g = c.benchmark_group("compress");
    let n = SEGMENT_SIZE;
    let seq = sequential_u128(n);
    let con = constant_u128(n);
    let out = outlier_u128(n);
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("bit_slab/seq", |b| {
        b.iter(|| black_box(bit_slab_compressor::compress(black_box(&seq)).unwrap()))
    });
    g.bench_function("bit_slab/out", |b| {
        b.iter(|| black_box(bit_slab_compressor::compress(black_box(&out)).unwrap()))
    });
    g.bench_function("rle/const", |b| {
        b.iter(|| black_box(rle_compressor::compress(black_box(&con)).unwrap()))
    });
    g.bench_function("delta/seq", |b| {
        b.iter(|| black_box(delta_compressor::compress(black_box(&seq)).unwrap()))
    });
    g.bench_function("lz4/seq", |b| {
        b.iter(|| black_box(lz4_compressor::compress(black_box(&seq)).unwrap()))
    });
    g.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let mut g = c.benchmark_group("decompress");
    let n = SEGMENT_SIZE;
    let seq_b = bit_slab_compressor::compress(&sequential_u128(n)).unwrap();
    let con_b = rle_compressor::compress(&constant_u128(n)).unwrap();
    let out_b = bit_slab_compressor::compress(&outlier_u128(n)).unwrap();
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("bit_slab/seq", |b| {
        b.iter(|| black_box(decompress_block(black_box(&seq_b)).unwrap()))
    });
    g.bench_function("rle/const", |b| {
        b.iter(|| black_box(decompress_block(black_box(&con_b)).unwrap()))
    });
    g.bench_function("bit_slab/out", |b| {
        b.iter(|| black_box(decompress_block(black_box(&out_b)).unwrap()))
    });
    g.finish();
}

fn bench_classify(c: &mut Criterion) {
    let mut g = c.benchmark_group("classify");
    let seq = sequential_u128(SEGMENT_SIZE);
    let con = constant_u128(SEGMENT_SIZE);
    g.bench_function("sequential", |b| {
        b.iter(|| black_box(classify(black_box(&seq))))
    });
    g.bench_function("constant", |b| {
        b.iter(|| black_box(classify(black_box(&con))))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_bitwriter,
    bench_bitreader,
    bench_simd_unpack,
    bench_compress,
    bench_decompress,
    bench_classify
);
criterion_main!(benches);
