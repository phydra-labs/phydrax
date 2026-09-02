#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from phydrax._array_archive import (
    ArrayArchiveCorruptionError,
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

    with pytest.raises(ArrayArchiveCorruptionError, match="byte bound"):
        read_array_archive(path, max_array_bytes=32)
    with pytest.raises(ArrayArchiveCorruptionError, match="total byte bound"):
        read_array_archive(path, max_total_bytes=32)
    with pytest.raises(ArrayArchiveCorruptionError, match="too many members"):
        read_array_archive(path, max_members=1)


def test_array_archive_rejects_compression_and_path_traversal_before_reading(
    tmp_path: Path,
) -> None:
    compressed = tmp_path / "compressed.phx"
    with zipfile.ZipFile(
        compressed, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("manifest.json", b'{"arrays":{}}')
    with pytest.raises(ArrayArchiveCorruptionError, match="canonical bounded stored"):
        read_array_archive(compressed)

    traversal = tmp_path / "traversal.phx"
    with zipfile.ZipFile(traversal, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", b'{"arrays":{}}')
        archive.writestr("arrays/../../payload.npy", b"payload")
    with pytest.raises(ArrayArchiveCorruptionError, match="canonical bounded stored"):
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

    with pytest.raises(ArrayArchiveCorruptionError, match="invalid"):
        read_array_archive(path)
    with pytest.raises(TypeError, match="object dtype"):
        write_array_archive(
            tmp_path / "writer-rejection.phx",
            manifest={"kind": "object-array"},
            arrays={"state": value},
        )
