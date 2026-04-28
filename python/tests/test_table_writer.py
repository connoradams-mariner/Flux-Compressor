# Copyright 2024 FluxCompress Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ``FluxTableWriter`` checkpointing + OCC (optimistic
concurrency control).

These tests bypass the Arrow/Rust compression pipeline so they stay fast
and deterministic; they drive the commit path directly by seeding the
writer's internal accumulator state (``_written_files`` /
``_file_manifests``) before calling ``_commit()``.

Covered scenarios:

* Log-entry fields: ``parent_version_crc32`` + ``writer_id`` present.
* ``_next_version`` and ``_list_log_entries`` skip ``.checkpoint.json``.
* Checkpoint file + ``_last_checkpoint.json`` pointer emitted at the
  configured interval.
* ``_replay_log`` uses the checkpoint and replays only post-checkpoint
  entries (reader-side correctness).
* Backwards compat: tables with no checkpoint still replay from v0.
* OCC: pre-claimed log entries force a retry; the writer lands at the
  next free version.
* OCC exhaustion raises ``CommitConflictError`` after ``_MAX_COMMIT_RETRIES``.
"""

from __future__ import annotations

import json
import os
import warnings
import zlib
from pathlib import Path

import pytest

from fluxcompress._table_writer import (  # type: ignore[import-not-found]
    _CHECKPOINT_SUFFIX,
    _LAST_CHECKPOINT_FILE,
    _MAX_COMMIT_RETRIES,
    FLUX_TABLE_READER_VERSION,
    FLUX_TABLE_WRITER_VERSION,
    CommitConflictError,
    ConcurrentOperationError,
    FluxTableWriter,
    LogForkError,
    ProtocolVersionError,
    _clone_meta,
    _crc32_of_committed_entry,
    _detect_conflicts,
    _entry_actions,
    _entry_metadata,
    _entry_txn,
    _find_committed_txn_version,
    _Fs,
    _list_log_entries,
    _next_version,
    _parse_gs_uri,
    _parse_s3_uri,
    _replay_log,
    _replay_metadata,
    _update_checkpoint_pointer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────


def _seeded_writer(
    path: str,
    *,
    checkpoint_interval: int = 10,
    writer_id: str = "test-writer",
) -> FluxTableWriter:
    """Return a writer in its ``__enter__``-ed state, so internal methods
    may be exercised without running the Arrow/Rust pipeline. Caller is
    responsible for closing via ``__exit__``."""
    w = FluxTableWriter(
        path,
        mode="append",
        checkpoint_interval=checkpoint_interval,
        writer_id=writer_id,
    )
    # Don't call __enter__ (which spins a ThreadPool we don't need); replay
    # just the init bits the commit path depends on.
    w._apply_mode_and_init()
    w._entered = True
    return w


def _fake_commit(
    writer: FluxTableWriter,
    file_basename: str,
    *,
    row_count: int = 1,
) -> None:
    """Populate the writer's accumulator as if a single part file had been
    produced, then drive ``_commit``."""
    rel = f"data/{file_basename}"
    writer._written_files = [rel]
    writer._file_manifests = [
        {
            "path": rel,
            "partition_values": {},
            "spec_id": writer._active_spec_id,
            "row_count": row_count,
            "file_size_bytes": 128,
            "column_stats": {},
        }
    ]
    writer._total_rows = row_count
    writer._commit()
    # Clear so the next commit doesn't double-append.
    writer._written_files = []
    writer._file_manifests = []
    writer._total_rows = 0


def _write_log_entry(log_dir: str, version: int, payload: dict) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{version:08d}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Log scanning helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_next_version_skips_checkpoint_files(tmp_path: Path) -> None:
    log_dir = tmp_path / "_flux_log"
    log_dir.mkdir()
    (log_dir / "00000000.json").write_text("{}")
    (log_dir / "00000001.json").write_text("{}")
    (log_dir / f"00000001{_CHECKPOINT_SUFFIX}").write_text("{}")

    fs = _Fs(str(tmp_path))
    assert _next_version(fs, str(log_dir)) == 2
    entries = _list_log_entries(fs, str(log_dir))
    assert entries == [(0, "00000000.json"), (1, "00000001.json")]


# ─────────────────────────────────────────────────────────────────────────────
# Commit writes parent_version_crc32 + writer_id
# ─────────────────────────────────────────────────────────────────────────────


def test_commit_writes_crc_and_writer_id(tmp_path: Path) -> None:
    table_path = str(tmp_path / "t.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0, writer_id="w-1")
    try:
        _fake_commit(w, "part-000000.flux")
        _fake_commit(w, "part-000001.flux")
    finally:
        w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    with open(log_dir / "00000000.json") as f:
        e0 = json.load(f)
    with open(log_dir / "00000001.json") as f:
        e1 = json.load(f)

    assert e0["writer_id"] == "w-1"
    assert e0["parent_version_crc32"] == 0  # no parent
    assert e0["operation"] == "create"

    assert e1["writer_id"] == "w-1"
    assert e1["operation"] == "append"
    expected_crc = zlib.crc32((log_dir / "00000000.json").read_bytes()) & 0xFFFFFFFF
    assert e1["parent_version_crc32"] == expected_crc


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint emission at interval boundary
# ─────────────────────────────────────────────────────────────────────────────


def test_checkpoint_fires_at_interval(tmp_path: Path) -> None:
    """With ``checkpoint_interval=5``, the first checkpoint lands at the
    commit whose ``base_version`` satisfies ``(base_version + 1) % 5 == 0``
    — i.e. version 4 (the 5th commit, 0-indexed)."""
    table_path = str(tmp_path / "cp.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=5)
    try:
        for i in range(5):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    cp = log_dir / f"00000004{_CHECKPOINT_SUFFIX}"
    assert cp.exists(), "checkpoint should be emitted at base_version=4"

    pointer = Path(table_path) / _LAST_CHECKPOINT_FILE
    assert pointer.exists()
    with open(pointer) as f:
        ptr = json.load(f)
    assert ptr["version"] == 4
    assert ptr["path"] == f"_flux_log/00000004{_CHECKPOINT_SUFFIX}"

    with open(cp) as f:
        snap = json.load(f)
    assert snap["is_checkpoint"] is True
    assert snap["version"] == 4
    assert sorted(snap["live_files"]) == [f"data/part-{i:06d}.flux" for i in range(5)]
    # Manifests replayed into the checkpoint.
    assert len(snap["file_manifests"]) == 5
    # Meta snapshot embedded.
    assert "meta_snapshot" in snap
    assert snap["meta_snapshot"].get("current_spec_id") == 0


def test_checkpoint_disabled_when_interval_zero(tmp_path: Path) -> None:
    table_path = str(tmp_path / "nocp.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        for i in range(6):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    assert not any(p.name.endswith(_CHECKPOINT_SUFFIX) for p in log_dir.iterdir())
    assert not (Path(table_path) / _LAST_CHECKPOINT_FILE).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Reader-side replay with and without checkpoint
# ─────────────────────────────────────────────────────────────────────────────


def test_replay_without_checkpoint(tmp_path: Path) -> None:
    table_path = str(tmp_path / "nocp.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        for i in range(3):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    fs = _Fs(table_path)
    live, manifests = _replay_log(
        fs, os.path.join(table_path, "_flux_log"), table_path=table_path
    )
    assert sorted(live) == [f"data/part-{i:06d}.flux" for i in range(3)]
    assert len(manifests) == 3


def test_replay_uses_checkpoint_and_post_entries(tmp_path: Path) -> None:
    """After a checkpoint lands, the reader should skip replaying log
    entries ≤ checkpoint.version and only apply entries strictly after it.
    We verify this by deleting the pre-checkpoint log entries on disk and
    confirming the reader still returns the correct live set."""
    table_path = str(tmp_path / "cp.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=5)
    try:
        # First 5 commits → checkpoint lands at v4.
        for i in range(5):
            _fake_commit(w, f"part-{i:06d}.flux")
        # Two more commits after the checkpoint.
        for i in range(5, 7):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    # Delete entries 0..4 to prove the reader no longer needs them.
    for i in range(5):
        (log_dir / f"{i:08d}.json").unlink()

    fs = _Fs(table_path)
    live, manifests = _replay_log(fs, str(log_dir), table_path=table_path)
    assert sorted(live) == [f"data/part-{i:06d}.flux" for i in range(7)]
    assert len(manifests) == 7


def test_replay_backward_compat_missing_pointer(tmp_path: Path) -> None:
    """A table produced by an older writer has no ``_last_checkpoint.json``
    pointer even though checkpoint-shaped files might one day exist in
    ``_flux_log``. The reader must fall back to a full replay."""
    table_path = str(tmp_path / "legacy.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        for i in range(4):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    assert not (Path(table_path) / _LAST_CHECKPOINT_FILE).exists()

    fs = _Fs(table_path)
    live, manifests = _replay_log(
        fs, os.path.join(table_path, "_flux_log"), table_path=table_path
    )
    assert sorted(live) == [f"data/part-{i:06d}.flux" for i in range(4)]
    assert len(manifests) == 4


# ─────────────────────────────────────────────────────────────────────────────
# OCC conflict resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_occ_retries_past_pre_existing_entry(tmp_path: Path) -> None:
    """A second writer that observes a stale ``next_version`` must detect
    the conflict on ``os.link`` and retry against the bumped version."""
    table_path = str(tmp_path / "occ.fluxtable")

    # Writer A commits version 0 normally.
    w_a = _seeded_writer(table_path, checkpoint_interval=0, writer_id="A")
    try:
        _fake_commit(w_a, "part-000000.flux")
    finally:
        w_a._closed = True

    # Simulate writer B that has a cached view of next_version=1 but
    # another committer slipped in entry 1 first.
    log_dir = Path(table_path) / "_flux_log"
    _write_log_entry(
        str(log_dir),
        version=1,
        payload={
            "version": 1,
            "timestamp_ms": 0,
            "operation": "append",
            "parent_version_crc32": 0,
            "writer_id": "other",
            "data_files_added": ["data/part-000001.flux"],
            "data_files_removed": [],
            "file_manifests": [
                {
                    "path": "data/part-000001.flux",
                    "partition_values": {},
                    "spec_id": 0,
                    "row_count": 1,
                    "file_size_bytes": 128,
                    "column_stats": {},
                }
            ],
            "row_count_delta": 1,
            "metadata": {},
        },
    )

    # Writer B commits — must land at version 2 after a single retry.
    w_b = _seeded_writer(table_path, checkpoint_interval=0, writer_id="B")
    try:
        _fake_commit(w_b, "part-B-000000.flux")
    finally:
        w_b._closed = True

    assert (log_dir / "00000002.json").exists()
    with open(log_dir / "00000002.json") as f:
        e2 = json.load(f)
    assert e2["writer_id"] == "B"
    assert e2["data_files_added"] == ["data/part-B-000000.flux"]
    # CRC chain must reference the *observed* parent (entry 1, written by
    # writer "other"), not writer B's original view of the log.
    expected_parent_crc = (
        zlib.crc32((log_dir / "00000001.json").read_bytes()) & 0xFFFFFFFF
    )
    assert e2["parent_version_crc32"] == expected_parent_crc


def test_occ_exhaustion_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the atomic claim keeps reporting a conflict the writer must
    eventually surface ``CommitConflictError`` rather than silently
    dropping the commit. We simulate sustained contention by stubbing
    ``_Fs.put_if_absent`` to always report the target path is already
    claimed; this drives the retry loop to exhaustion regardless of what
    ``_next_version`` returns."""
    table_path = str(tmp_path / "occ-x.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0, writer_id="stuck")

    call_count = {"n": 0}

    def _always_conflict(self, target_path: str, data: bytes) -> bool:
        call_count["n"] += 1
        return False

    monkeypatch.setattr(_Fs, "put_if_absent", _always_conflict)

    try:
        with pytest.raises(CommitConflictError):
            _fake_commit(w, "part-stuck.flux")
    finally:
        w._closed = True

    assert call_count["n"] == _MAX_COMMIT_RETRIES


# ─────────────────────────────────────────────────────────────────────────────
# Meta is updated only after a successful commit
# ─────────────────────────────────────────────────────────────────────────────


def test_meta_written_after_log(tmp_path: Path) -> None:
    table_path = str(tmp_path / "meta.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        _fake_commit(w, "part-000000.flux")
    finally:
        w._closed = True

    meta_path = Path(table_path) / "_flux_meta.json"
    log0 = Path(table_path) / "_flux_log" / "00000000.json"
    assert meta_path.exists()
    assert log0.exists()
    # The meta file's mtime should be >= the log entry's mtime since we
    # write the log first, then the meta.
    assert meta_path.stat().st_mtime_ns >= log0.stat().st_mtime_ns


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: conditional PUT primitive
# ─────────────────────────────────────────────────────────────────────────────


def test_put_if_absent_local_succeeds_then_conflicts(tmp_path: Path) -> None:
    fs = _Fs(str(tmp_path))
    target = str(tmp_path / "t.json")
    assert fs.put_if_absent(target, b"first") is True
    assert fs.put_if_absent(target, b"second") is False
    with open(target, "rb") as f:
        assert f.read() == b"first"


def test_put_if_absent_local_no_tmp_leftovers(tmp_path: Path) -> None:
    """After a successful or failed put_if_absent the tmp sibling must be
    cleaned up — otherwise a long-running writer session leaks entries in
    the log directory that confuse _next_version scans."""
    fs = _Fs(str(tmp_path))
    target = str(tmp_path / "x.json")
    fs.put_if_absent(target, b"a")
    fs.put_if_absent(target, b"b")  # second call is a conflict
    stray = [p.name for p in tmp_path.iterdir() if ".json.tmp" in p.name]
    assert stray == []


def test_parse_uri_helpers() -> None:
    assert _parse_gs_uri("gs://my-bucket/path/to/obj") == ("my-bucket", "path/to/obj")
    assert _parse_gs_uri("gcs://b/k") == ("b", "k")
    assert _parse_s3_uri("s3://bucket/key") == ("bucket", "key")
    assert _parse_s3_uri("s3a://x/y/z") == ("x", "y/z")
    with pytest.raises(ValueError):
        _parse_gs_uri("http://bad")
    with pytest.raises(ValueError):
        _parse_s3_uri("s3://bucket-only")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: CRC chain validation
# ─────────────────────────────────────────────────────────────────────────────


def test_replay_detects_log_fork(tmp_path: Path) -> None:
    """A forged entry whose parent_version_crc32 claims the wrong parent
    hash must trigger LogForkError under strict replay."""
    table_path = str(tmp_path / "fork.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        for i in range(2):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    # Forge entry at version 2 that claims a bogus parent CRC.
    _write_log_entry(
        str(log_dir),
        version=2,
        payload={
            "version": 2,
            "parent_version_crc32": 0xDEADBEEF,
            "writer_id": "attacker",
            "data_files_added": ["data/part-forged.flux"],
            "data_files_removed": [],
            "file_manifests": [],
            "row_count_delta": 0,
        },
    )

    fs = _Fs(table_path)
    with pytest.raises(LogForkError) as exc:
        _replay_log(fs, str(log_dir), table_path=table_path, strict=True)
    assert exc.value.version == 2
    assert exc.value.found_crc == 0xDEADBEEF


def test_replay_salvage_mode_tolerates_fork(tmp_path: Path) -> None:
    """Salvage mode should stop replay at the last valid version rather
    than raising, so operators can recover readable state."""
    table_path = str(tmp_path / "salvage.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        for i in range(2):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    _write_log_entry(
        str(log_dir),
        version=2,
        payload={
            "version": 2,
            "parent_version_crc32": 0xDEADBEEF,
            "data_files_added": ["data/part-forged.flux"],
        },
    )

    fs = _Fs(table_path)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        live, _ = _replay_log(
            fs, str(log_dir), table_path=table_path, strict=False,
        )
    # The forged v2 add is NOT applied because replay stopped at v1.
    assert "data/part-forged.flux" not in live
    assert live == {"data/part-000000.flux", "data/part-000001.flux"}


def test_crc_chain_intact_across_sequential_commits(tmp_path: Path) -> None:
    """Back-to-back commits must leave a fully valid CRC chain so strict
    replay succeeds end-to-end."""
    table_path = str(tmp_path / "chain.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        for i in range(5):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    fs = _Fs(table_path)
    live, _ = _replay_log(fs, str(Path(table_path) / "_flux_log"),
                          table_path=table_path, strict=True)
    assert sorted(live) == [f"data/part-{i:06d}.flux" for i in range(5)]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: actions array + semantic conflict detection
# ─────────────────────────────────────────────────────────────────────────────


def test_entry_carries_actions_array(tmp_path: Path) -> None:
    table_path = str(tmp_path / "acts.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        _fake_commit(w, "part-000000.flux")
    finally:
        w._closed = True

    log0 = Path(table_path) / "_flux_log" / "00000000.json"
    entry = json.loads(log0.read_text())
    assert isinstance(entry.get("actions"), list)
    kinds = [a["type"] for a in entry["actions"]]
    # v0: protocol + metadata + add + commit_info (no txn, no remove).
    assert kinds[0] == "protocol"
    assert "metadata" in kinds
    assert "add" in kinds
    assert kinds[-1] == "commit_info"
    # And the legacy flat fields must still be populated for backward compat.
    assert entry["data_files_added"] == ["data/part-000000.flux"]


def test_append_vs_append_no_conflict(tmp_path: Path) -> None:
    """Two concurrent pure-append writers must both succeed: their
    commits are semantically compatible."""
    table_path = str(tmp_path / "appends.fluxtable")

    w_a = _seeded_writer(table_path, checkpoint_interval=0, writer_id="A")
    w_b = _seeded_writer(table_path, checkpoint_interval=0, writer_id="B")
    try:
        _fake_commit(w_a, "part-A.flux")
        # Before B commits, A's entry is already at v0. B should land at v1.
        _fake_commit(w_b, "part-B.flux")
    finally:
        w_a._closed = True
        w_b._closed = True

    log_dir = Path(table_path) / "_flux_log"
    assert (log_dir / "00000000.json").exists()
    assert (log_dir / "00000001.json").exists()


def test_remove_vs_remove_conflict(tmp_path: Path) -> None:
    """A writer attempting to remove a file that a concurrent commit
    also touched must surface ConcurrentOperationError, not retry."""
    table_path = str(tmp_path / "rmrm.fluxtable")
    # v0: create file-X.
    w0 = _seeded_writer(table_path, checkpoint_interval=0, writer_id="creator")
    try:
        _fake_commit(w0, "file-X.flux")
    finally:
        w0._closed = True

    # Writer B loads the table, then we sneak a concurrent remove-of-X in
    # at v1 before B commits. B tries to remove the same file → conflict.
    w_b = FluxTableWriter(
        table_path, mode="append", checkpoint_interval=0, writer_id="B",
    )
    w_b._apply_mode_and_init()
    w_b._entered = True
    try:
        # Simulate a concurrent writer removing file-X at v1.
        log_dir = Path(table_path) / "_flux_log"
        _write_log_entry(
            str(log_dir),
            version=1,
            payload={
                "version": 1,
                "parent_version_crc32": _crc32_of_committed_entry(
                    w_b._fs, w_b._log_dir, 0,
                ),
                "writer_id": "concurrent",
                "actions": [
                    {"type": "remove", "path": "data/file-X.flux"},
                ],
                "data_files_added": [],
                "data_files_removed": ["data/file-X.flux"],
                "file_manifests": [],
            },
        )
        # B's commit tries to remove the same file.
        w_b._pending_removes = ["data/file-X.flux"]
        with pytest.raises(ConcurrentOperationError) as exc:
            w_b._commit()
        assert exc.value.kind == "remove"
        assert exc.value.conflicting_version == 1
    finally:
        w_b._closed = True


def test_metadata_vs_metadata_conflict(tmp_path: Path) -> None:
    table_path = str(tmp_path / "mdmd.fluxtable")
    # v0 lays down the initial metadata.
    w0 = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        _fake_commit(w0, "part-000000.flux")
    finally:
        w0._closed = True

    # Writer B prepares a new partition spec (triggering a metadata
    # action). A concurrent writer sneaks a metadata-change commit in
    # before B commits → conflict.
    w_b = FluxTableWriter(
        table_path, mode="append", checkpoint_interval=0, writer_id="B",
        partition_by=[{"source_column": "region"}],
    )
    w_b._apply_mode_and_init()
    w_b._entered = True
    try:
        log_dir = Path(table_path) / "_flux_log"
        _write_log_entry(
            str(log_dir),
            version=1,
            payload={
                "version": 1,
                "parent_version_crc32": _crc32_of_committed_entry(
                    w_b._fs, w_b._log_dir, 0,
                ),
                "writer_id": "concurrent",
                "actions": [
                    {
                        "type": "metadata",
                        "table_id": "other",
                        "partition_specs": [],
                        "current_spec_id": 0,
                        "clustering_columns": ["c1"],
                        "properties": {},
                    },
                ],
                "data_files_added": [],
                "data_files_removed": [],
                "file_manifests": [],
            },
        )
        with pytest.raises(ConcurrentOperationError) as exc:
            w_b._commit()
        assert exc.value.kind == "metadata"
    finally:
        w_b._closed = True


def test_detect_conflicts_empty_range_returns_none(tmp_path: Path) -> None:
    """Sanity: empty (from, to) range cannot conflict."""
    table_path = str(tmp_path / "empty.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        _fake_commit(w, "part-000000.flux")
    finally:
        w._closed = True
    fs = _Fs(table_path)
    assert _detect_conflicts(
        fs, str(Path(table_path) / "_flux_log"),
        from_version_inclusive=5,
        to_version_exclusive=5,
        our_actions=[{"type": "remove", "path": "anything"}],
    ) is None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: metadata authoritativeness
# ─────────────────────────────────────────────────────────────────────────────


def test_metadata_reconstructs_from_log(tmp_path: Path) -> None:
    """Replay the log to rebuild the authoritative metadata snapshot,
    even if _flux_meta.json is missing / stale."""
    table_path = str(tmp_path / "mdlog.fluxtable")
    w = FluxTableWriter(
        table_path, mode="error", checkpoint_interval=0,
        partition_by=[{"source_column": "region"}],
        clustering_columns=["user_id"],
    )
    w._apply_mode_and_init()
    w._entered = True
    try:
        _fake_commit(w, "part-000000.flux")
    finally:
        w._closed = True

    # Delete the side file; the log should still be authoritative.
    meta_file = Path(table_path) / "_flux_meta.json"
    meta_file.unlink()

    fs = _Fs(table_path)
    rebuilt = _replay_metadata(
        fs, str(Path(table_path) / "_flux_log"), table_path=table_path,
    )
    assert rebuilt is not None
    assert rebuilt["clustering_columns"] == ["user_id"]
    assert rebuilt["partition_specs"][0]["fields"][0]["source_column"] == "region"


def test_metadata_action_emitted_on_change(tmp_path: Path) -> None:
    table_path = str(tmp_path / "mdchange.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        _fake_commit(w, "part-000000.flux")
    finally:
        w._closed = True
    # The first commit MUST have emitted a metadata action because the
    # committed baseline was None on a fresh table.
    log0 = Path(table_path) / "_flux_log" / "00000000.json"
    entry = json.loads(log0.read_text())
    assert _entry_metadata(entry) is not None


def test_no_metadata_action_on_unchanged_append(tmp_path: Path) -> None:
    """A subsequent append that doesn't mutate metadata should skip the
    metadata action to keep the log light."""
    table_path = str(tmp_path / "mdskip.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        _fake_commit(w, "part-0.flux")
    finally:
        w._closed = True

    # Fresh session: meta already committed, no changes in this writer.
    w2 = _seeded_writer(table_path, checkpoint_interval=0)
    try:
        _fake_commit(w2, "part-1.flux")
    finally:
        w2._closed = True

    log1 = Path(table_path) / "_flux_log" / "00000001.json"
    entry = json.loads(log1.read_text())
    assert _entry_metadata(entry) is None, (
        "append-only commit should not emit metadata action"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: idempotent transaction tokens
# ─────────────────────────────────────────────────────────────────────────────


def test_txn_params_must_be_provided_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FluxTableWriter(str(tmp_path / "x.fluxtable"), txn_app_id="a")
    with pytest.raises(ValueError):
        FluxTableWriter(str(tmp_path / "y.fluxtable"), txn_version=1)


def test_txn_action_recorded_in_log(tmp_path: Path) -> None:
    table_path = str(tmp_path / "txn.fluxtable")
    w = FluxTableWriter(
        table_path, mode="error", checkpoint_interval=0,
        txn_app_id="stream-xyz", txn_version=42,
    )
    w._apply_mode_and_init()
    w._entered = True
    try:
        _fake_commit(w, "part-000000.flux")
    finally:
        w._closed = True

    log0 = Path(table_path) / "_flux_log" / "00000000.json"
    entry = json.loads(log0.read_text())
    txn = _entry_txn(entry)
    assert txn == {"type": "txn", "app_id": "stream-xyz", "version": 42}


def test_txn_duplicate_is_noop(tmp_path: Path) -> None:
    """A writer with a (app_id, version) pair already committed must
    silently skip its commit — critical for exactly-once streaming."""
    table_path = str(tmp_path / "txndup.fluxtable")
    w = FluxTableWriter(
        table_path, mode="error", checkpoint_interval=0,
        txn_app_id="app", txn_version=1,
    )
    w._apply_mode_and_init()
    w._entered = True
    try:
        _fake_commit(w, "first.flux")
    finally:
        w._closed = True

    # Replay the same txn id / version at or below the committed mark.
    w2 = FluxTableWriter(
        table_path, mode="append", checkpoint_interval=0,
        txn_app_id="app", txn_version=1,
    )
    w2._apply_mode_and_init()
    w2._entered = True
    try:
        _fake_commit(w2, "duplicate.flux")
    finally:
        w2._closed = True

    # Only v0 should exist — the duplicate commit skipped itself.
    log_dir = Path(table_path) / "_flux_log"
    existing = sorted(p.name for p in log_dir.iterdir() if p.name.endswith(".json"))
    assert existing == ["00000000.json"]


def test_txn_higher_version_commits(tmp_path: Path) -> None:
    table_path = str(tmp_path / "txnhi.fluxtable")
    w = FluxTableWriter(
        table_path, mode="error", checkpoint_interval=0,
        txn_app_id="app", txn_version=1,
    )
    w._apply_mode_and_init()
    w._entered = True
    try:
        _fake_commit(w, "first.flux")
    finally:
        w._closed = True

    w2 = FluxTableWriter(
        table_path, mode="append", checkpoint_interval=0,
        txn_app_id="app", txn_version=2,
    )
    w2._apply_mode_and_init()
    w2._entered = True
    try:
        _fake_commit(w2, "second.flux")
    finally:
        w2._closed = True

    log_dir = Path(table_path) / "_flux_log"
    assert (log_dir / "00000001.json").exists()
    fs = _Fs(table_path)
    assert _find_committed_txn_version(fs, str(log_dir), "app") == 2


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: atomic checkpoint pointer
# ─────────────────────────────────────────────────────────────────────────────


def test_checkpoint_pointer_never_regresses(tmp_path: Path) -> None:
    pointer = str(tmp_path / _LAST_CHECKPOINT_FILE)
    fs = _Fs(str(tmp_path))

    _update_checkpoint_pointer(fs, pointer, 9, "00000009.checkpoint.json")
    with open(pointer) as f:
        assert json.load(f)["version"] == 9

    # Attempting to publish an older pointer must be a no-op.
    _update_checkpoint_pointer(fs, pointer, 4, "00000004.checkpoint.json")
    with open(pointer) as f:
        assert json.load(f)["version"] == 9  # still 9

    # A strictly-newer pointer advances.
    _update_checkpoint_pointer(fs, pointer, 19, "00000019.checkpoint.json")
    with open(pointer) as f:
        assert json.load(f)["version"] == 19


def test_checkpoint_file_written_once_via_put_if_absent(tmp_path: Path) -> None:
    """If two writers race on the same checkpoint version, put_if_absent
    only lets one actually write; the other is a silent no-op."""
    table_path = str(tmp_path / "cprace.fluxtable")
    w = _seeded_writer(table_path, checkpoint_interval=5)
    try:
        for i in range(5):
            _fake_commit(w, f"part-{i:06d}.flux")
    finally:
        w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    cp = log_dir / f"00000004{_CHECKPOINT_SUFFIX}"
    assert cp.exists()
    before = cp.read_bytes()

    # A second writer session emits another checkpoint at the same version.
    w2 = _seeded_writer(table_path, checkpoint_interval=5)
    try:
        _fake_commit(w2, "extra.flux")  # bumps to v5, no new checkpoint
    finally:
        w2._closed = True

    # Drive _write_checkpoint directly for v4 again to exercise
    # put_if_absent on an existing checkpoint path.
    w3 = _seeded_writer(table_path, checkpoint_interval=5)
    try:
        w3._base_version = 4
        w3._write_checkpoint()
    finally:
        w3._closed = True

    # Content preserved — put_if_absent refused to overwrite.
    after = cp.read_bytes()
    assert before == after


# ─────────────────────────────────────────────────────────────────────────────
# Actions-based round-trip replay (reader honours actions array)
# ─────────────────────────────────────────────────────────────────────────────


def test_reader_uses_actions_array(tmp_path: Path) -> None:
    """Simulate a v2-format entry that ONLY has an actions array (no
    flat fields populated) and confirm _replay_log applies its adds."""
    table_path = str(tmp_path / "onlyactions.fluxtable")
    log_dir = Path(table_path) / "_flux_log"
    log_dir.mkdir(parents=True)
    _write_log_entry(
        str(log_dir),
        version=0,
        payload={
            "version": 0,
            "parent_version_crc32": 0,
            "actions": [
                {"type": "protocol",
                 "reader_min_version": 1, "writer_min_version": 2},
                {"type": "add", "path": "data/only-a.flux",
                 "manifest": {"path": "data/only-a.flux", "row_count": 1,
                              "file_size_bytes": 1, "spec_id": 0,
                              "column_stats": {}, "partition_values": {}}},
            ],
        },
    )

    fs = _Fs(table_path)
    live, manifests = _replay_log(
        fs, str(log_dir), table_path=table_path, strict=True,
    )
    assert live == {"data/only-a.flux"}
    assert manifests["data/only-a.flux"]["row_count"] == 1


def test_protocol_version_guard(tmp_path: Path) -> None:
    """A log that declares a higher reader_min_version than this client
    implements should trigger ProtocolVersionError."""
    table_path = str(tmp_path / "proto.fluxtable")
    log_dir = Path(table_path) / "_flux_log"
    log_dir.mkdir(parents=True)
    _write_log_entry(
        str(log_dir),
        version=0,
        payload={
            "version": 0,
            "parent_version_crc32": 0,
            "actions": [
                {"type": "protocol",
                 "reader_min_version": FLUX_TABLE_READER_VERSION + 99,
                 "writer_min_version": FLUX_TABLE_WRITER_VERSION},
            ],
        },
    )
    fs = _Fs(table_path)
    with pytest.raises(ProtocolVersionError):
        _replay_log(fs, str(log_dir), table_path=table_path)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: multi-process-style concurrent append stress
# ─────────────────────────────────────────────────────────────────────────────


def test_concurrent_writers_no_lost_commits(tmp_path: Path) -> None:
    """Simulate many alternating writer sessions (as two independent
    processes would) and verify every commit lands at a unique version
    with an intact CRC chain."""
    table_path = str(tmp_path / "stress.fluxtable")
    num_sessions = 20
    for i in range(num_sessions):
        w = FluxTableWriter(
            table_path,
            mode="error" if i == 0 else "append",
            checkpoint_interval=7,
            writer_id=f"w-{i}",
        )
        w._apply_mode_and_init()
        w._entered = True
        try:
            _fake_commit(w, f"part-{i:06d}.flux")
        finally:
            w._closed = True

    log_dir = Path(table_path) / "_flux_log"
    versions = sorted(
        int(p.stem) for p in log_dir.iterdir()
        if p.name.endswith(".json") and not p.name.endswith(_CHECKPOINT_SUFFIX)
    )
    assert versions == list(range(num_sessions))

    fs = _Fs(table_path)
    live, _ = _replay_log(
        fs, str(log_dir), table_path=table_path, strict=True,
    )
    assert live == {f"data/part-{i:06d}.flux" for i in range(num_sessions)}
