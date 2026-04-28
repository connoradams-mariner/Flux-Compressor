# Release process

FluxCompress uses **[release-please](https://github.com/googleapis/release-please)**
to automate semantic versioning and release cuts. Contributors only
need to know two things:

1. Branch through `staging` into `main` (never push to `main` directly
   except for documentation-only fixes).
2. Write **Conventional Commit** messages — that's how release-please
   decides whether the next release is patch / minor / major.

## Conventional commits — the rules

Your commit message / PR title must start with one of these prefixes:

| Prefix      | Effect                                             | Bumps version? |
|-------------|----------------------------------------------------|----------------|
| `feat:`     | A new user-visible feature.                        | **minor** (0.5.4 → 0.6.0) |
| `fix:`      | A bug fix.                                         | **patch** (0.5.4 → 0.5.5) |
| `perf:`     | Performance improvement with no API change.        | **patch** |
| `revert:`   | Revert an earlier commit.                          | **patch** |
| `feat!:`, `fix!:`, … | Any prefix with `!` **or** a `BREAKING CHANGE:` footer in the body. | **major** (0.5.4 → 1.0.0) |
| `docs:`     | Documentation-only changes.                        | no bump, but lands in CHANGELOG |
| `chore:`    | Maintenance that users don't need to know about.   | no bump, not shown |
| `ci:`       | CI / release-pipeline changes.                     | no bump, not shown |
| `refactor:` | Code refactor with no behaviour change.            | no bump, not shown |
| `test:`     | Adding/updating tests only.                        | no bump, not shown |
| `style:`    | Whitespace / formatting only.                      | no bump, not shown |
| `build:`    | Build-system changes.                              | no bump, not shown |

### Examples

```text
feat: add null-aware compression wrapper
fix(spark): route ArrowUtils through SparkArrowBridge
feat!: drop the pre-Atlas-v1 file format

docs: document serverless limitations on Databricks
chore: bump workspace deps
ci: switch to macos-14 for x86_64 builds
```

### Scopes are optional but useful

`feat(python):`, `fix(spark):`, `perf(loom):` — these show up in
the generated CHANGELOG grouped by scope, which makes release notes
easier to skim.

### Breaking changes

Either put a `!` after the type (`feat!:`, `refactor!:`) **or**
include a `BREAKING CHANGE:` footer:

```text
refactor: rename FluxTable::scan() to FluxTable::iter_rows()

BREAKING CHANGE: `FluxTable::scan()` is now `FluxTable::iter_rows()`.
Callers need to update their imports.
```

Either form triggers a major version bump.

## Day-to-day workflow

```text
                   ┌───────────────────────────────────────────────────┐
                   │ You work on `staging` (or a feature branch)       │
                   │ Every commit message is Conventional.             │
                   └────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
                   ┌───────────────────────────────────────────────────┐
                   │ Open a PR: `staging` → `main`                    │
                   │ CI runs. Review. Merge.                           │
                   └────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
                   ┌───────────────────────────────────────────────────┐
                   │ release-please sees the new commit on `main`,     │
                   │ updates the long-lived Release PR titled          │
                   │ `chore(main): release 0.5.5`.                     │
                   │ It bumps Cargo.toml / pyproject.toml /            │
                   │ build.sbt / READMEs and updates CHANGELOG.md.     │
                   └────────────────────────┬──────────────────────────┘
                                            │
                       …repeat for each PR you merge into main…
                                            │
                                            ▼
                   ┌───────────────────────────────────────────────────┐
                   │ When you're ready to ship, review the Release PR  │
                   │ and merge it.                                     │
                   │ release-please creates a `vX.Y.Z` git tag +       │
                   │ GitHub Release.                                   │
                   └────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
                   ┌───────────────────────────────────────────────────┐
                   │ `release-crates.yml`, `release-pypi.yml`,         │
                   │ `release-maven.yml` all trigger on the tag.       │
                   │ crates.io + PyPI + Maven Central publish.         │
                   └───────────────────────────────────────────────────┘
```

## Documentation-only changes

If the change is strictly documentation (markdown, roadmap docs,
docstrings), you may push directly to `main` as long as branch
protection allows it. Use the `docs:` prefix:

```bash
git commit -m "docs(databricks): clarify serverless JNI blocker"
git push origin main
```

release-please will include this in the next CHANGELOG without
forcing a version bump — you can keep shipping prose fixes without
needing a release.

## Where the bump actually happens

release-please updates every file listed under `extra-files` in
[`release-please-config.json`](../release-please-config.json):

- `Cargo.toml` — `[workspace.package] version` + `[workspace.dependencies] loom.version`
- `pyproject.toml` — `[project] version`
- `spark-connector/build.sbt` — the `ThisBuild / version := ...` line (annotated `// x-release-please-version`)
- Per-crate `README.md` files — version snippets inside dependency
  examples, annotated with `# x-release-please-version` or
  `<!-- x-release-please-version -->`

If you add a new file that holds a version literal, annotate it
and add it to `release-please-config.json`'s `extra-files` list.

## Manual bumping (fallback)

If release-please is down or you need an out-of-band release, the
`scripts/bump-version.sh` helper still works:

```bash
./scripts/bump-version.sh 0.5.5 --tag
```

That's the escape hatch; day-to-day, let release-please drive.
