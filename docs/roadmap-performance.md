# Roadmap: Performance Optimizations

## Current State (v2)
FluxCompress beats Parquet on compression ratio (20/20 test configs) and
matches or beats Parquet on compress speed at 1M rows. At 10M+ rows,
Parquet's C++ backend has advantages in memory efficiency and decompress
throughput.

## Completed Optimizations (v0.1)

### Arrow FFI Zero-Copy Bridge
Replaced IPC serialization (Python → bytes → Rust) with Arrow C Data
Interface (`from_pyarrow_bound` / `to_pyarrow`). Eliminated 2 full data
copies per compress/decompress call.

**Impact:** 3–7× compress speedup at 1M rows.

### Rayon Parallel Compression
Columns compressed in parallel via `rayon::par_iter`. Segments within each
column also parallelized via `into_par_iter`.

**Impact:** ~2–4× on multi-column tables.

### Geometric Probe Stride
Adaptive segmenter uses doubling stride (1024 → 2048 → 4096 → 8192) when
checking for drift, reducing classify() calls from ~64/segment to ~7/segment.

**Impact:** ~15% compress speedup on single-column data.

### Skip-Tiny-Blocks Secondary Compression
Blocks smaller than 64 bytes skip LZ4/Zstd post-pass. Also skips secondary
compression when the compressed output is larger than the input (negative
gain detection).

**Impact:** Better ratio on small blocks (e.g., constant data).

### Pre-Allocated Output Buffers
`Vec::with_capacity` for output buffer based on estimated total size.
`extract_as_u128` uses pre-allocated Vec with `extend` instead of `collect`.

**Impact:** ~5–10% less allocation overhead.

## Planned Optimizations

### Phase 1: Lazy Per-Segment Widening (v0.2)
**Status:** Done — `adaptive_segment_u64()` works on `&[u64]` directly,
only widening 1024-value probe windows for classification. Each segment
is widened to u128 individually (max 64K values = 1MB).
**Impact:** +39% compress throughput at 10M rows (Speed profile).

The current code does:
```
let values_u128: Vec<u128> = col.values_u64.iter().map(...).collect(); // full column!
let segments = adaptive_segment(&values_u128, force);
```

The fix: make the segmenter work on `&[u64]` directly and only widen the
small probe windows (1024 values) for classification. Each segment is widened
to u128 only when passed to the compressor (64K values max, not millions).

Changes:
- `adaptive_segment_u64(&[u64])` that classifies u64 probes directly
- Classifier accepts `&[u64]` with cheap inline u128 widening for bit-width checks
- Compressor receives `&[u64]` segment + widens only at encode time
- Eliminates the O(N) `Vec<u128>` allocation entirely for u64-fit columns

### Phase 1b: Generic Classifier (v0.2)
**Status:** Not started
**Impact:** Medium (avoids redundant u128 widening in classify hot path)

Make the Loom classifier generic over value width:
- `classify<T: Into<u128> + Copy>(values: &[T])` 
- Entropy, cardinality, and delta checks operate on native width
- Only `bits_needed()` widens to u128 (single value, not slice)

### Phase 2: Fused Encode + Compress (v0.2)
**Status:** Not started
**Impact:** Medium (estimated 15–20% compress speedup)

Instead of: encode block → allocate Vec → secondary compress Vec,
stream the encoder output directly into a Zstd/LZ4 streaming compressor.
Eliminates the intermediate buffer allocation.

Changes:
- Implement `Write` trait wrapper around LZ4/Zstd stream encoder
- Strategy compressors write to a generic `impl Write` instead of `Vec<u8>`

### Phase 3: Streaming Zstd Dictionary (v0.3)
**Status:** Research
**Impact:** Medium-High (30–40% compress speedup on Archive profile)

Train a Zstd dictionary on the first few blocks, then compress subsequent
blocks with the dictionary context. Much faster than cold-start per-block
Zstd.

**Risk:** Ties blocks together — may break independent block decompression.
Needs a "dictionary scope" concept in the Atlas footer.

### Phase 4: Memory-Mapped Decompression (v0.3)
**Status:** Not started
**Impact:** High on decompress speed at TB scale

Use `mmap` for reading `.flux` files instead of `fs::read`. The OS handles
page faulting and caching, reducing memory pressure on large files.

Changes:
- `FluxReader::decompress_mmap(&Path)` method
- `memmap2` crate for cross-platform mmap

### Phase 5: Null Bitmap Support (v0.2)
**Status:** Not started (BlockMeta field exists, implementation pending)
**Impact:** Correctness + ratio improvement on nullable Spark data

Extract Arrow null bitmap, store inline in block, compress only dense
non-null values, reconstruct nulls on read. Skipped when column is fully
non-null (common case, zero overhead).

## Benchmarking
Run the scaling benchmark to measure any changes:

```bash
python python/tests/bench_scaling.py
```

This generates `docs/scaling_benchmark.png` with ratio, compress speed,
and decompress speed charts from 1K to 50M rows.
