#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax._array_archive as archive_module
from phydrax._array_archive import (
    array_collection_digest,
    array_payload_byte_count,
    array_payload_digest,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    read_array_archive,
    write_array_archive,
)


def test_array_archive_reads_are_bounded_before_payload_allocation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bounded.phx"
    write_array_archive(
        path,
        manifest={"kind": "bounded-test"},
        arrays={"state": np.arange(32, dtype=np.float64)},
    )

    with pytest.raises(ArrayArchiveCorruptionError):
        read_array_archive(path, limits=ArrayArchiveLimits(max_member_bytes=32))
    with pytest.raises(ArrayArchiveCorruptionError):
        read_array_archive(path, limits=ArrayArchiveLimits(max_aggregate_bytes=32))
    with pytest.raises(ArrayArchiveCorruptionError):
        read_array_archive(path, limits=ArrayArchiveLimits(max_members=1))


def test_array_archive_rejects_compression_and_path_traversal_before_reading(
    tmp_path: Path,
) -> None:
    compressed = tmp_path / "compressed.phx"
    with zipfile.ZipFile(
        compressed, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("manifest.json", b'{"arrays":{}}')
    with pytest.raises(ArrayArchiveCorruptionError, match="canonical"):
        read_array_archive(compressed)

    traversal = tmp_path / "traversal.phx"
    with zipfile.ZipFile(traversal, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", b'{"arrays":{}}')
        archive.writestr("arrays/../../payload.npy", b"payload")
    with pytest.raises(ArrayArchiveCorruptionError):
        read_array_archive(traversal)


def test_array_archive_never_loads_pickled_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / "object.phx"
    buffer = io.BytesIO()
    value = np.asarray([{"not": "portable"}], dtype=object)
    np.save(buffer, value, allow_pickle=True)
    payload = buffer.getvalue()
    manifest = {
        "kind": "malicious-object-array",
        "arrays": {
            "state": {
                "member": "arrays/000000.npy",
                "shape": [1],
                "dtype": value.dtype.str,
                "order": "C",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode(),
        )
        archive.writestr("arrays/000000.npy", payload)

    with pytest.raises(ArrayArchiveCorruptionError, match="dtype is not admitted"):
        read_array_archive(path)
    with pytest.raises(TypeError, match="object dtype"):
        write_array_archive(
            tmp_path / "writer-rejection.phx",
            manifest={"kind": "object-array"},
            arrays={"state": value},
        )


def test_streamed_archive_matches_prechange_golden_bytes(tmp_path: Path) -> None:
    arrays = {f"a{i:04d}": np.arange(131072, dtype=np.float64) for i in range(4)}
    path = tmp_path / "golden.zip"
    write_array_archive(path, manifest={"kind": "memory-benchmark"}, arrays=arrays)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "6cd1f83ac9cef3cc5f3a0588e4ae5fc3478de7656a28dc01dcfea3245d6416da"
    )
    assert array_collection_digest(arrays) == (
        "db9d1ddef35d46b00f6920f7ce1d6a6ab890493b00a7f9de5e87c648fc67205d"
    )


@pytest.mark.parametrize("layout", ["scalar", "strided", "fortran", "empty", "endian"])
def test_streamed_array_digest_matches_independent_numpy_encoding(layout: str) -> None:
    match layout:
        case "scalar":
            value = np.asarray(3.0, dtype=np.float64)
        case "strided":
            value = np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2]
        case "fortran":
            value = np.asfortranarray(np.arange(30, dtype=np.float64).reshape(5, 6))
        case "empty":
            value = np.empty((0, 2), dtype=np.float64)
        case _:
            value = np.arange(5, dtype=">f8")
    buffer = io.BytesIO()
    np.save(
        buffer, np.ascontiguousarray(value) if value.ndim else value, allow_pickle=False
    )
    payload = buffer.getvalue()
    assert array_payload_digest(value) == hashlib.sha256(payload).hexdigest()
    assert array_payload_byte_count(value) == len(payload)
    reference = hashlib.sha256()
    for part in (b"state", payload):
        reference.update(len(part).to_bytes(8, "big"))
        reference.update(part)
    assert array_collection_digest({"state": value}) == reference.hexdigest()


def test_archive_source_change_cannot_replace_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "existing.zip"
    original = np.arange(8, dtype=np.float64)
    write_array_archive(path, manifest={"kind": "original"}, arrays={"a": original})
    before = path.read_bytes()
    publish = archive_module.publish_file

    def mutate_then_publish(*args: Any, **kwargs: Any) -> object:
        original[0] += 1
        return publish(*args, **kwargs)

    monkeypatch.setattr(archive_module, "publish_file", mutate_then_publish)
    with pytest.raises(ValueError, match="changed while writing"):
        write_array_archive(
            path, manifest={"kind": "replacement"}, arrays={"a": original}
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "limits",
    [
        ArrayArchiveLimits(max_member_bytes=160),
        ArrayArchiveLimits(max_aggregate_bytes=100),
        ArrayArchiveLimits(max_members=1),
    ],
    ids=["member", "aggregate", "members"],
)
def test_write_limit_refusal_preserves_destination(
    tmp_path: Path, limits: ArrayArchiveLimits
) -> None:
    path = tmp_path / "existing.zip"
    path.write_bytes(b"untouched")
    with pytest.raises(ValueError):
        write_array_archive(
            path,
            manifest={},
            arrays={"state": np.arange(8, dtype=np.float64)},
            limits=limits,
        )
    assert path.read_bytes() == b"untouched"


def test_archive_reads_are_readonly_and_independent(tmp_path: Path) -> None:
    path = tmp_path / "state.zip"
    original = np.arange(8, dtype=np.float64)
    write_array_archive(path, manifest={}, arrays={"state": original})
    _, first = read_array_archive(path)
    _, second = read_array_archive(path)
    original[:] = -1
    np.testing.assert_array_equal(first["state"], np.arange(8, dtype=np.float64))
    assert not first["state"].flags.writeable
    assert not np.shares_memory(first["state"], second["state"])
    with pytest.raises(ValueError):
        first["state"][0] = 1
