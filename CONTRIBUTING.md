# Contributing to FluxCompress

Thank you for contributing! This guide covers the complete workflow for
developing, testing, and releasing the FluxCompress Python package.

---

## Repository Layout (Python-relevant paths)

```
flux-compress/
├── pyproject.toml                  ← maturin config + PyPI metadata
├── crates/python/
│   ├── Cargo.toml                  ← PyO3 crate config
│   └── src/lib.rs                  ← Rust → Python bindings (PyO3)
└── python/
    ├── fluxcompress/
    │   ├── __init__.py             ← Public API re-exports
    │   ├── _fluxcompress.pyi       ← Type stubs (IDEs / mypy)
    │   ├── _polars.py              ← Polars DataFrame helpers
    │   └── spark.py                ← Apache Spark UDF helpers
    └── tests/
        ├── test_core.py            ← Core round-trip + predicate tests
        └── test_benchmarks.py      ← pytest-benchmark performance tests
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | stable (≥ 1.78) | Compile the Rust core |
| Python | ≥ 3.8 | Run tests and examples |
| [maturin](https://github.com/PyO3/maturin) | ≥ 1.5 | Build Python wheels from Rust |
| pyarrow | ≥ 14.0 | Arrow interop |

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Install Python dev dependencies
pip install pyarrow polars pytest pytest-benchmark
```

---

## Local Development Build

The fastest way to iterate: `maturin develop` compiles the Rust extension
in-place inside your active virtual environment.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install the package in editable / development mode
maturin develop

# Now import it directly
python -c "import fluxcompress as fc; print(fc.__version__)"
```

For a faster iteration loop (debug symbols, no optimisation):

```bash
maturin develop --profile dev
```

For release-level performance (slower build, same speed as a published wheel):

```bash
maturin develop --release
```

With AVX2 SIMD (x86-64 only):

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --release --features simd-avx2
```

---

## Running Tests

```bash
# All Python tests
pytest python/tests/ -v

# Core correctness tests only
pytest python/tests/test_core.py -v

# Benchmarks (with timing output)
pytest python/tests/test_benchmarks.py --benchmark-only -v

# Rust unit tests
cargo test --workspace

# Rust + Python together
cargo test --workspace && pytest python/tests/ -v
```

---

## Building Wheels

### Single platform (your machine)

```bash
maturin build --release
# Wheel lands in: dist/fluxcompress-0.1.0-cp38-abi3-<platform>.whl
```

### All platforms (via GitHub Actions)

Push a tag to trigger the CI release pipeline:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The CI workflow (`.github/workflows/CI.yml`) automatically:
1. Builds wheels for Linux x86-64, Linux aarch64, macOS arm64, macOS x86-64, Windows x86-64
2. Runs the Python test suite on each wheel
3. Publishes to PyPI using OIDC trusted publishing (no API token needed)

### Manual cross-platform build

```bash
# Linux (manylinux — works on all glibc >= 2.17 systems)
maturin build --release --target x86_64-unknown-linux-gnu --manylinux auto

# macOS (Apple Silicon)
maturin build --release --target aarch64-apple-darwin

# Windows
maturin build --release --target x86_64-pc-windows-msvc
```

---

## PyPI Publishing

FluxCompress uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no long-lived API tokens required.

### One-time setup (first release only)

1. Go to [https://pypi.org/manage/project/fluxcompress/settings/publishing/](https://pypi.org/manage/project/fluxcompress/settings/publishing/)
2. Add a new trusted publisher:
   - **Owner**: `connoradams-mariner`
   - **Repository**: `Flux-Compressor`
   - **Workflow name**: `CI.yml`
   - **Environment name**: `pypi`
3. On GitHub, create an environment named `pypi` at:
   `Settings → Environments → New environment`

### Releasing a new version

```bash
# 1. Bump the version in Cargo.toml (workspace.package.version)
#    and pyproject.toml (project.version) — must match.
vim Cargo.toml           # version = "0.2.0"
vim pyproject.toml       # version = "0.2.0"

# 2. Commit and tag
git add Cargo.toml pyproject.toml
git commit -m "chore: release v0.2.0"
git tag v0.2.0
git push origin main --tags

# CI takes it from here → builds all wheels → publishes to PyPI.
```

---

## Adding a New PyO3 Function

1. Add the Rust function to `crates/python/src/lib.rs`:

```rust
/// My new function — docstring shows in Python.
#[pyfunction]
fn my_function(value: i64) -> i64 {
    value * 2
}
```

2. Register it in `_fluxcompress()`:

```rust
m.add_function(wrap_pyfunction!(my_function, m)?)?;
```

3. Add the type stub to `python/fluxcompress/_fluxcompress.pyi`:

```python
def my_function(value: int) -> int:
    """Double a value."""
    ...
```

4. Re-export it from `python/fluxcompress/__init__.py`:

```python
from fluxcompress._fluxcompress import (
    ...
    my_function,
)
```

5. Write a test in `python/tests/test_core.py`:

```python
def test_my_function():
    assert fc.my_function(21) == 42
```

6. Rebuild and test:

```bash
maturin develop && pytest python/tests/test_core.py::test_my_function -v
```

---

## Code Style

- **Rust**: `cargo fmt` (enforced by CI), `cargo clippy -- -D warnings`
- **Python**: `ruff check` + `ruff format` (configured in `pyproject.toml`)

```bash
# Format everything
cargo fmt --all
ruff format python/

# Lint
cargo clippy --workspace --all-targets
ruff check python/
```

---

## Architecture Notes

### Why maturin?

maturin handles the entire wheel-building pipeline: it compiles the Rust
crate, packages the `.so` / `.pyd` extension into the correct wheel directory
structure, and sets the `abi3` stable ABI tag so one wheel works on Python
3.8 through 3.12+.

### Why ABI3 (`abi3-py38`)?

The `abi3` (stable ABI) build uses a restricted subset of the CPython C API
that is guaranteed to be compatible across all CPython versions ≥ 3.8. This
means **one Linux wheel works on Python 3.8, 3.9, 3.10, 3.11, and 3.12**
without separate builds — the filename ends in `cp38-abi3` to indicate this.

### Zero-Copy Arrow Transfer

Polars DataFrames and PyArrow Tables expose the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
via `__arrow_c_stream__`. The Rust bridge uses this to receive Arrow data
without copying it from Python-managed memory to Rust-managed memory.
