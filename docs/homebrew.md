# Homebrew distribution

`fluxcapacitor` ships as a prebuilt binary via a Homebrew tap, driven by
[`cargo-dist`](https://github.com/axodotdev/cargo-dist). This document
covers the one-time setup, the per-release flow, and how end users
install.

## End-user install

Once the tap is live and a tagged release has run:

```bash
brew install connoradams-mariner/tap/fluxcapacitor
```

This works on macOS (Apple Silicon + Intel) and on Linux via Linuxbrew.
The formula is auto-updated by CI on every `vX.Y.Z` tag, so
`brew upgrade fluxcapacitor` always tracks the latest release.

The same `vX.Y.Z` tags also produce a curl|sh installer:

```bash
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/connoradams-mariner/Flux-Compressor/releases/latest/download/fluxcapacitor-installer.sh \
  | sh
```

Both paths drop a `fluxcapacitor` binary on `$PATH` and verify the
SHA-256 of the downloaded archive against the manifest published with
the release.

## How the pipeline fits together

```
release-please PR merge on main
        │
        ▼
 v0.5.5 git tag pushed
        │
        ├──► release-crates.yml      (publishes flux-loom, flux-jni-bridge, fluxcapacitor on crates.io)
        ├──► release-pypi.yml        (builds + uploads abi3 wheels for PyPI)
        ├──► release-maven.yml       (builds Spark connector JAR for Maven Central)
        └──► release-binaries.yml    (this pipeline)
                │
                ├─ build matrix: aarch64-apple-darwin, x86_64-apple-darwin,
                │                x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu
                ├─ generates  fluxcapacitor-installer.sh
                ├─ generates  Formula/fluxcapacitor.rb
                ├─ uploads    artifacts to the GitHub Release
                └─ pushes     the formula to connoradams-mariner/homebrew-tap@main
```

All four `release-*.yml` workflows trigger on the same `v*.*.*` tag and
run in parallel; they touch disjoint artifacts so there is no
coordination required between them.

## One-time setup

### 1. Create the tap repository

The tap is a separate GitHub repo with a specific layout:

```
homebrew-tap/
└── Formula/
    └── fluxcapacitor.rb       (auto-generated, do not hand-edit)
```

Steps:

1. Sign in as `connoradams-mariner` (or whichever org/user owns the
   tap — must match the `tap = "..."` value in
   `[workspace.metadata.dist]`).
2. Create a new public repo named **exactly** `homebrew-tap`. Homebrew
   discovers taps by the `homebrew-` prefix; the user-facing
   `brew tap connoradams-mariner/tap` strips it automatically.
3. Initialize with a one-line README so the default branch (`main`)
   exists. cargo-dist will create `Formula/` on its first push.
4. No protected-branch rules — the publish job force-pushes to `main`
   on every release, so requiring PR review will block releases.

### 2. Mint the publishing token

The `release-binaries.yml` workflow needs to push to the tap repo.
Use a **fine-grained personal access token** scoped to that single
repo:

1. https://github.com/settings/personal-access-tokens/new
2. Resource owner: `connoradams-mariner` (the org/user that owns the
   tap).
3. Repository access: **Only select repositories** → pick
   `homebrew-tap`.
4. Permissions → Repository permissions:
   - Contents: **Read and write**
   - Metadata: Read (auto-selected)
5. Expiration: 90 days is a sensible default; rotate on calendar.
6. Copy the generated `github_pat_…` value.

Then on the **`Flux-Compressor`** repo (not the tap):

1. Settings → Secrets and variables → Actions → New repository secret.
2. Name: `HOMEBREW_TAP_TOKEN`.
3. Value: paste the PAT.

The workflow references this secret in the `publish-homebrew-formula`
job; nothing else uses it.

### 3. (Optional) Test before the next release

You don't need to wait for a real release-please cut to validate the
pipeline:

```bash
# Push a throwaway pre-release tag
git tag v0.0.0-dist-test
git push origin v0.0.0-dist-test

# Watch it run
gh run watch --repo connoradams-mariner/Flux-Compressor

# Clean up after — delete the tag, the GitHub Release, and the
# Formula/fluxcapacitor.rb commit on the tap repo.
git push origin :refs/tags/v0.0.0-dist-test
gh release delete v0.0.0-dist-test --repo connoradams-mariner/Flux-Compressor --yes
gh api -X DELETE repos/connoradams-mariner/homebrew-tap/git/refs/heads/main \
  || git -C ../homebrew-tap reset --hard HEAD~1 && git -C ../homebrew-tap push --force
```

`publish-prereleases = false` in the workspace metadata means a real
prerelease tag (e.g. `v0.6.0-rc1`) will be skipped by the publish job;
`v0.0.0-dist-test` doesn't have a prerelease suffix so it goes through
the full pipeline.

## Updating cargo-dist

cargo-dist publishes new versions every few weeks. To upgrade:

```bash
# Install the version you want
cargo install cargo-dist --version 0.28.0

# Bump the pin in [workspace.metadata.dist] cargo-dist-version
# Then regenerate the workflow:
cargo dist init --yes
```

`cargo dist init` rewrites `release-binaries.yml` from scratch using
the metadata in `Cargo.toml`. Diff carefully — the regen wipes any
hand-edits, so anything bespoke must live in `[workspace.metadata.dist]`
or in a follow-up workflow file.

## Troubleshooting

**`brew install` reports `SHA256 mismatch`.** A release was re-tagged
(e.g. you force-pushed a tag after a failed run) and the cached
artifact on the GitHub Release no longer matches the formula. Delete
the GitHub Release, re-run the workflow, and the formula will be
regenerated.

**Publish job fails with `403` on `git push`.** The
`HOMEBREW_TAP_TOKEN` PAT expired or was scoped to the wrong repo.
Re-mint per step 2 above.

**Linux aarch64 build OOMs in CI.** The `aarch64-unknown-linux-gnu`
job cross-compiles from `ubuntu-latest` (x86_64) and the link step on
`fluxcapacitor` (which transitively depends on `polars`, `parquet`,
`arrow`) can spike to ~6 GB. Switch the `os:` for that matrix entry
to `ubuntu-22.04-arm` once GitHub's ARM-native fleet is stable, or
add `[profile.release] codegen-units = 1` followed by
`cargo-dist`-supported lto tuning.

**Formula publishes but `brew install` errors with
`Error: undefined method 'on_arm'`.** The user's Homebrew is older
than 4.0. cargo-dist 0.27 emits modern formula syntax; bump local
brew with `brew update`.

## When to graduate to homebrew-core

Once `fluxcapacitor` has clear adoption signals (≥75 stars,
external contributors, or downstream packaging requests), submit a
formula PR to https://github.com/Homebrew/homebrew-core. Requirements:

- Stable, versioned releases ✓ (we tag `vX.Y.Z`).
- Permissive license ✓ (Apache-2.0).
- Builds from source on the homebrew-core CI without prebuilt binaries
  — they bottle internally. The current `cargo-dist` pipeline does
  *not* satisfy this; we'd need a hand-written formula that runs
  `cargo install --root #{prefix}` against the workspace.
- A separate `homebrew-core` formula does not replace the tap; users
  who already typed `brew install connoradams-mariner/tap/fluxcapacitor`
  keep working off the tap.
