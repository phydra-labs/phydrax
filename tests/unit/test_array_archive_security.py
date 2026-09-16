#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

import phydrax._array_archive as archive_module
from phydrax._array_archive import (
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    read_array_archive,
    write_array_archive,
)


def _npy_payload(value: object, /, *, allow_pickle: bool = False) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=allow_pickle)
    return buffer.getvalue()


def _write_raw_archive(
    path: Path,
    arrays: dict[str, tuple[np.ndarray, bytes]],
    /,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    inventory: dict[str, dict[str, object]] = {}
    members: dict[str, bytes] = {}
    for index, (name, (value, payload)) in enumerate(sorted(arrays.items())):
        member = f"arrays/{index:06d}.npy"
        members[member] = payload
        inventory[name] = {
            "member": member,
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {**(metadata or {}), "arrays": inventory}
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for member, payload in members.items():
            archive.writestr(member, payload)


def _fail_if_numpy_loads(*args: object, **kwargs: object) -> None:
    raise AssertionError("np.load was reached before archive admission completed")


def test_container_member_manifest_and_aggregate_limits_precede_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = write_array_archive(
        tmp_path / "bounded.zip",
        manifest={"kind": "limit-fixture"},
        arrays={"values": np.arange(32, dtype=np.float64)},
    )
    monkeypatch.setattr(archive_module.np, "load", _fail_if_numpy_loads)

    policies = (
        ArrayArchiveLimits(max_container_bytes=path.stat().st_size - 1),
        ArrayArchiveLimits(max_aggregate_bytes=1),
        ArrayArchiveLimits(max_member_bytes=64),
        ArrayArchiveLimits(max_manifest_bytes=16),
        ArrayArchiveLimits(max_members=1),
    )
    for policy in policies:
        with pytest.raises(ArrayArchiveCorruptionError):
            read_array_archive(path, limits=policy)


def test_array_dtype_rank_shape_and_element_limits_precede_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    high_rank = write_array_archive(
        tmp_path / "high-rank.zip",
        manifest={"kind": "rank-fixture"},
        arrays={"values": np.zeros((1,) * 9, dtype=np.float32)},
        limits=None,
    )
    values = write_array_archive(
        tmp_path / "elements.zip",
        manifest={"kind": "element-fixture"},
        arrays={
            "first": np.ones((2,), dtype=np.float32),
            "second": np.ones((2,), dtype=np.float32),
        },
    )
    object_array = np.asarray([{"secret": "must-not-unpickle"}], dtype=object)
    object_archive = tmp_path / "object.zip"
    _write_raw_archive(
        object_archive,
        {"object": (object_array, _npy_payload(object_array, allow_pickle=True))},
    )
    monkeypatch.setattr(archive_module.np, "load", _fail_if_numpy_loads)

    with pytest.raises(ArrayArchiveCorruptionError, match="rank limit"):
        read_array_archive(high_rank)
    with pytest.raises(ArrayArchiveCorruptionError, match="axis-length limit"):
        read_array_archive(values, limits=ArrayArchiveLimits(max_axis_length=1))
    with pytest.raises(ArrayArchiveCorruptionError, match="element limit"):
        read_array_archive(values, limits=ArrayArchiveLimits(max_array_elements=1))
    with pytest.raises(ArrayArchiveCorruptionError, match="aggregate element"):
        read_array_archive(values, limits=ArrayArchiveLimits(max_total_array_elements=3))
    with pytest.raises(ArrayArchiveCorruptionError, match="dtype"):
        read_array_archive(object_archive)


def test_manifest_nesting_limit_precedes_array_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = write_array_archive(
        tmp_path / "nested.zip",
        manifest={"telemetry": {"nested": {"deeper": {"value": 1}}}},
        arrays={"value": np.asarray(1.0)},
    )
    monkeypatch.setattr(archive_module.np, "load", _fail_if_numpy_loads)

    with pytest.raises(ArrayArchiveCorruptionError, match="nesting limit"):
        read_array_archive(path, limits=ArrayArchiveLimits(max_manifest_nesting=3))


def test_explicit_trusted_policy_preserves_legacy_high_rank_archive(tmp_path: Path):
    expected = np.zeros((1,) * 9, dtype=np.float32)
    structured = np.array(
        [(1.0, 2)], dtype=np.dtype([("value", "<f8"), ("index", "<i4")])
    )
    path = write_array_archive(
        tmp_path / "trusted.zip",
        manifest={"kind": "trusted-local"},
        arrays={"value": expected, "structured": structured},
        limits=None,
    )

    manifest, arrays = read_array_archive(path, limits=None)

    assert manifest["kind"] == "trusted-local"
    np.testing.assert_array_equal(arrays["value"], expected)
    np.testing.assert_array_equal(arrays["structured"], structured)
    assert not arrays["value"].flags.writeable


def test_array_archive_rejects_symlink_and_fifo_sources_without_following(
    tmp_path: Path,
):
    archive = write_array_archive(
        tmp_path / "source.zip",
        manifest={"kind": "path-safety"},
        arrays={"value": np.asarray(1.0)},
    )
    linked = tmp_path / "linked.zip"
    linked.symlink_to(archive)
    fifo = tmp_path / "blocking.zip"
    os.mkfifo(fifo)

    with pytest.raises(ArrayArchiveCorruptionError):
        read_array_archive(linked)
    with pytest.raises(ArrayArchiveCorruptionError, match="regular file"):
        read_array_archive(fifo)


def test_array_archive_rejects_symlinked_parent_component(tmp_path: Path):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    archive = write_array_archive(
        trusted / "source.zip",
        manifest={"kind": "parent-path-safety"},
        arrays={"value": np.asarray(1.0)},
    )
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(trusted, target_is_directory=True)

    with pytest.raises(ArrayArchiveCorruptionError):
        read_array_archive(linked_parent / archive.name)


def test_writer_enforces_the_same_limits_as_reader(tmp_path: Path):
    destination = tmp_path / "writer-bounded.zip"
    with pytest.raises(ValueError, match="rank limit"):
        write_array_archive(
            destination,
            manifest={"kind": "writer-limit"},
            arrays={"value": np.zeros((1,) * 9, dtype=np.float32)},
        )
    assert not destination.exists()


def test_expected_inventory_rejects_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = write_array_archive(
        tmp_path / "inventory.zip",
        manifest={"kind": "inventory"},
        arrays={"value": np.ones((4,), dtype=np.float64)},
    )
    monkeypatch.setattr(archive_module.np, "load", _fail_if_numpy_loads)
    with pytest.raises(ArrayArchiveCorruptionError, match="template"):
        read_array_archive(
            path,
            expected_inventory={"value": ((5,), np.dtype(np.float64))},
        )


def test_reader_returns_defensive_c_contiguous_read_only_arrays(tmp_path: Path):
    source = np.asfortranarray(np.arange(12, dtype=np.float64).reshape((3, 4)))
    path = write_array_archive(
        tmp_path / "defensive.zip",
        manifest={"kind": "defensive"},
        arrays={"value": source},
    )
    _, arrays = read_array_archive(
        path,
        expected_inventory={"value": (source.shape, source.dtype)},
    )
    restored = arrays["value"]
    assert restored.flags.c_contiguous
    assert not restored.flags.writeable
    source[:] = -1.0
    np.testing.assert_array_equal(restored, np.arange(12).reshape((3, 4)))
