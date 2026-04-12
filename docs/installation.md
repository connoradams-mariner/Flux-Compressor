# Installation

## From PyPI (recommended)

```bash
# Core package (PyArrow only)
pip install fluxcompress

# With Polars support
pip install fluxcompress[polars]

# With pandas support
pip install fluxcompress[pandas]

# With Apache Spark support
pip install fluxcompress[spark]

# Everything (dev tools included)
pip install fluxcompress[all]
```

## From Source (development)

Requires: **Rust 1.78+** and **maturin 1.5+**.

```bash
# 1. Clone
git clone https://github.com/connoradams-mariner/Flux-Compressor.git
cd Flux-Compressor

# 2. Install build tool
pip install maturin

# 3. Build + install in editable/development mode
maturin develop --release

# 4. Verify
python -c "import fluxcompress; print(fluxcompress.__version__)"
```

## Building Wheels

```bash
# Linux (manylinux — works on all major distros)
maturin build --release --features simd-avx2

# macOS Apple Silicon
maturin build --release --target aarch64-apple-darwin --features simd-neon

# macOS Intel
maturin build --release --target x86_64-apple-darwin --features simd-avx2

# Windows
maturin build --release --features simd-avx2
```

Wheels are written to `target/wheels/`.

## System Requirements

| Platform | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.11+ |
| PyArrow | 14.0 | 16.0+ |
| Polars | 0.40 | 0.40+ |
| Rust | 1.78 | stable |
| CPU (x86-64) | Any | AVX2 (2013+) |
| CPU (ARM) | Any | NEON (all modern ARM) |

## SIMD Flags

FluxCompress auto-detects AVX2/NEON at runtime. To build with the fastest
SIMD paths compiled in:

```bash
# Maximum performance (native CPU)
RUSTFLAGS="-C target-cpu=native" maturin build --release --features simd-avx2
```
