# fluxcompress-python

[![PyPI](https://img.shields.io/pypi/v/fluxcompress.svg)](https://pypi.org/project/fluxcompress/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)

**PyO3 bindings for [FluxCompress](https://github.com/connoradams-mariner/Flux-Compressor).** This Rust crate is the backing extension for the `fluxcompress` wheel on PyPI — end users should `pip install fluxcompress`, not consume this crate directly.

## For Python users

```bash
pip install fluxcompress               # core + PyArrow
pip install 'fluxcompress[polars]'     # + Polars helpers
pip install 'fluxcompress[spark]'      # + PySpark helpers
```

```python
import pyarrow as pa
import fluxcompress as fc

table = pa.table({"id": range(1_000_000)})
buf   = fc.compress(table, profile="archive")
back  = fc.decompress(buf, predicate=fc.col("id") > 500_000)
```

Wheels are published for Linux x86_64 / aarch64, macOS x64 / arm64, and Windows x64, with `abi3-py38` so one wheel covers Python 3.8 → 3.13+.

## For Rust developers

This crate is built by `maturin`; it is not a conventional Rust library and is therefore **not independently publishable on crates.io**. If you want to link Rust code against the FluxCompress engine, depend on [`flux-loom`](https://crates.io/crates/flux-loom) instead:

```toml
[dependencies]
loom = { package = "flux-loom", version = "0.6.3" } # x-release-please-version
```

## Building locally

From the monorepo root:

```bash
pip install maturin
maturin develop --release -m crates/python/Cargo.toml
pytest python/tests/ -v
```

## See also

- **[fluxcompress on PyPI](https://pypi.org/project/fluxcompress/)** — the actual user-facing package.
- **[flux-loom](https://crates.io/crates/flux-loom)** — core Rust engine.
- **[Flux-Compressor](https://github.com/connoradams-mariner/Flux-Compressor)** — main repo.

## License

Apache 2.0.
