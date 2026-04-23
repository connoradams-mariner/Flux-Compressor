#!/usr/bin/env bash
# Bump the FluxCompress release version across every file that still
# pins a literal version string.
#
# Usage:
#     ./scripts/bump-version.sh 0.5.5
#     ./scripts/bump-version.sh 0.5.5 --tag      # also cut + push the tag
#
# What it touches:
#   * Cargo.toml                                  (workspace.package.version)
#   * Cargo.toml                                  ([workspace.dependencies] loom)
#   * pyproject.toml                              ([project].version)
#   * spark-connector/build.sbt                   (ThisBuild / version)
#   * crates/{loom,jni-bridge,fluxcapacitor,python}/README.md
#
# The per-crate Cargo.toml files **do not** need bumping — they
# inherit `loom.workspace = true` and `version.workspace = true`.
#
# What it does NOT do (by design, to keep it simple):
#   * Edit CHANGELOG.md  — write it yourself before running this.
#   * Commit or push      — the default run just edits files.
#                           Pass `--tag` to also `git add`, commit,
#                           tag, and push.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <new-version> [--tag]" >&2
  exit 1
fi

NEW="$1"
TAG="${2:-}"

# Simple SemVer shape check.
if ! [[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]; then
  echo "error: '$NEW' is not a SemVer string (e.g. 1.2.3 or 1.2.3-rc1)" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
REPO=$(pwd)
echo "Bumping FluxCompress to $NEW in $REPO"

# ── 1. Cargo workspace version -------------------------------------------
sed -i -E \
  "0,/^version\s*=\s*\"[^\"]+\"/{s|^version\s*=\s*\"[^\"]+\"|version     = \"$NEW\"|}" \
  Cargo.toml

# ── 2. [workspace.dependencies] loom version -----------------------------
sed -i -E \
  "s|^loom\s*=\s*\{ version = \"[^\"]+\", path = \"crates/loom\", package = \"flux-loom\" \}|loom        = { version = \"$NEW\", path = \"crates/loom\", package = \"flux-loom\" }|" \
  Cargo.toml

# ── 3. pyproject.toml ---------------------------------------------------
sed -i -E \
  "0,/^version\s*=\s*\"[^\"]+\"/{s|^version\s*=\s*\"[^\"]+\"|version     = \"$NEW\"|}" \
  pyproject.toml

# ── 4. spark-connector/build.sbt ---------------------------------------
sed -i -E \
  "s|ThisBuild / version\s*:=\s*\"[^\"]+\"|ThisBuild / version      := \"$NEW\"|" \
  spark-connector/build.sbt

# ── 5. Per-crate READMEs (just the version snippets) -------------------
# Hunt-and-replace every `version = "X.Y.Z"` that appears inside a
# dependency table in our own READMEs. This is a best-effort sweep;
# it won't touch prose like "v0.5.4 shipped the zero-copy path".
for f in crates/loom/README.md \
         crates/jni-bridge/README.md \
         crates/fluxcapacitor/README.md \
         crates/python/README.md; do
  if [ -f "$f" ]; then
    sed -i -E "s|version = \"[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?\"|version = \"$NEW\"|g" "$f"
    # Also catch the `flux-X = "0.5.Y"` short-form.
    sed -i -E \
      "s|^(flux-[a-z-]+)\s*=\s*\"[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?\"|\\1 = \"$NEW\"|g" \
      "$f"
    # And the `Benchmarks (vX.Y.Z)` heading used in crates/loom/README.md.
    sed -i -E "s|Benchmarks \\(v[0-9]+\.[0-9]+\.[0-9]+\\)|Benchmarks (v$NEW)|g" "$f"
  fi
done

echo "Bumped. Diff:"
git --no-pager diff --stat

# ── 6. Optional: commit + tag + push -----------------------------------
if [ "$TAG" = "--tag" ]; then
  git add -A
  git commit -m "v$NEW"
  git tag -a "v$NEW" -m "FluxCompress v$NEW"
  git push origin main
  git push origin "v$NEW"
  echo "Pushed tag v$NEW. Three release workflows (crates / pypi / maven) + ci should now be running."
else
  echo
  echo "Next steps:"
  echo "  git diff                      # review the changes"
  echo "  git add -A && git commit -m \"v$NEW\""
  echo "  git tag -a v$NEW -m \"FluxCompress v$NEW\""
  echo "  git push origin main && git push origin v$NEW"
fi
