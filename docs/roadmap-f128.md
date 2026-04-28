# f128 (IEEE 754 binary128) Support — Roadmap

Status: **Planned for v0.5+**.
On-disk format is ready; blocked on Arrow ecosystem. A `Decimal128`
carrier works as a bit-pattern stop-gap today.

---

## Why wait?

FluxCompress's job is to round-trip Arrow columns. The ingest boundary
is `&dyn arrow_array::Array` — if Arrow doesn't have a 128-bit float
type, there's no Arrow array for callers to hand us, and the decode side
has nothing typed to return into. Adding a Rust-internal f128 pipeline
without Arrow integration would force callers to juggle their own
binary128 containers and would not interoperate with PyArrow / Polars /
Spark.

### What's missing upstream

| Component                      | Status                                           |
|--------------------------------|--------------------------------------------------|
| `arrow_schema::DataType::Float128` | Does not exist in arrow-rs 52 or on main.    |
| Arrow IPC spec `Float(HALF/SINGLE/DOUBLE)` | No `QUAD` variant in the Flatbuffers schema. |
| PyArrow / C++ Arrow            | No `float128` logical type; follows IPC spec.    |
| Rust stable `f128`             | Nightly-only (`feature(f128)`, rustc ≥1.77).     |
| LLVM soft-float `__trunctfhf2` etc. | Shipped as `compiler-builtins` on most targets, good enough for our needs. |

The upstream path is an Arrow specification proposal (like `Float16`
landed via flatbuffer schema evolution in 2022). Until that lands, any
binary128 "Arrow array" is a custom extension type stored in an
`ExtensionArray` (which is still a `FixedSizeBinary(16)` at rest).

---

## What we can do now (stop-gap)

### Option A: Decimal128-as-carrier

Callers that already have `u128` bit patterns (e.g. from `f128::to_bits()`
on nightly, or from a C++ `__float128`) can store them in a
`Decimal128Array` with precision/scale `(38, 0)`:

```rust
let bits: Vec<u128> = quad_values.iter().map(|q| q.to_bits()).collect();
let i128s: Vec<i128> = bits.iter().map(|&u| u as i128).collect();
let arr = Decimal128Array::from(i128s)
    .with_precision_and_scale(38, 0)?;
```

This works today, losslessly, through the full Flux pipeline
(compress → .flux → decompress → `Decimal128Array` → reinterpret). It
re-uses the BitSlab + OutlierMap code we already ship. The caller owns
the reinterpret step.

### Option B: `FixedSizeBinary(16)` carrier

Semantically cleaner but has no native compression path in Flux v0.3.
Callers would get raw-LZ4/Zstd on the 16-byte layout. Probably worse
ratio than Option A.

Option A is the recommended stop-gap.

---

## Full integration plan (v0.5+)

### Assumptions to revisit before starting

1. Either Arrow has landed `DataType::Float128` **or** we accept that our
   public API exposes a custom `FluxDType::Float128` + `ExtensionArray`
   registered with Arrow's extension-type registry (see
   `arrow_schema::extension_type`).
2. Stable Rust `f128` is available (MSRV bump) **or** we use the
   `compiler-builtins` soft-float operations gated behind a
   `f128-soft` cargo feature.

### Scope

1. **`FluxDType` tag** — add `Float128 = 0x12` next to `Decimal128`.
   Serialises identically (u128 bit pattern).
2. **Router** — `dtype_router::route(Float128)` returns
   `RouteDecision::Classify`, matching `Decimal128`.
3. **Writer** — clone `compress_decimal128_column` into
   `compress_float128_column` but:
   - Extract values as u128 bit patterns via
     `FixedSizeBinaryArray::value(i)` (16 bytes → `u128::from_le_bytes`).
   - Feed the existing u128 pipeline (BitSlab + OutlierMap + DeltaDelta).
   - On top of that, add a **float-aware pre-transform** (ALP-style but
     wider): detect decimal-shaped quads (`value == (m as f128) / 10^e`
     for small e) and emit i128 mantissas. This is the same idea as ALP
     for f64 but with a wider scale range and native 128-bit mantissas.
4. **Reader** —
   - New `LeafData::Numeric128` already exists; reuse it.
   - New `reconstruct_float128` builds the Arrow representation chosen
     in step 1 (either the real `Float128Array` if arrow-rs adds it, or
     a `FixedSizeBinaryArray` + extension metadata).
5. **Python / JNI bindings** — pyo3 / JNI sides mirror Arrow: if Arrow
   doesn't have `Float128`, the bindings expose `decimal128` with a
   pyarrow extension-type adapter.
6. **Round-trip test** — write-read with values that exercise
   - subnormals, infinities, NaN payloads,
   - `-0.0` vs `+0.0`,
   - values requiring the full 112-bit mantissa,
   - decimal-shaped values that hit the ALP path,
   - `f128::MAX` / `f128::MIN` to stress the outlier map.

### Compression ratio targets

Based on the f64 ALP numbers, reasonable expectations for v0.5 on f128:

| Data shape                              | Expected Flux ratio | Notes                       |
|-----------------------------------------|---------------------|-----------------------------|
| Decimal-shaped (prices in quad precision) | 4–8× (via ALP-128)  | Wider mantissa range.       |
| Physics / scientific simulation output   | 1.3–2× (via BitSlab + zstd) | Depends on entropy. |
| Random bit patterns                      | ~1.0× (fallback to raw LZ4) | OutlierMap patching only. |

### Non-goals

- **No native arithmetic.** Flux stores bit patterns; it doesn't do
  `f128` math. The `compiler-builtins` soft-float routines are needed
  only inside the ALP pre-transform.
- **Write v0.4 files that readers expect**. The new TAG (`Float128`)
  is additive — pre-0.5 readers that don't know the tag will error
  cleanly, which is the correct behaviour for a new type.

### Implementation order

1. Upstream watch — track
   [`apache/arrow-rs` issue #5050 (Float16/Float128)](https://github.com/apache/arrow-rs/issues/5050).
   When it progresses, start here.
2. Land `FluxDType::Float128` + block-meta plumbing (no codec path yet).
3. Land `compress_float128_column` + `reconstruct_float128` with the
   plain u128 pipeline. Validate round-trip.
4. Land ALP-128 pre-transform as a separate commit.
5. Python + JNI bindings update.
6. Bench on:
   - Financial quote data (lots of decimal-shaped quads),
   - Simulated scientific trajectories (dense arbitrary quads),
   - Mixed Decimal128/Float128/Float64 schemas to exercise the per-column
     adaptive routing.

---

## Decision log

- **2026-04**: Shipped Decimal128 round-trip (i128/u128 bit patterns).
  Explicitly deferred Float128 until Arrow adds `DataType::Float128`
  or we commit to an extension-type API. The writer's u128 pipeline
  and the reader's `Numeric128` leaf variant are both general over
  bit patterns, so the remaining work is purely routing + Arrow-side
  adapter code.
