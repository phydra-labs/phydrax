#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class ArrayArchiveError(RuntimeError):
    """Base class for portable array-archive failures."""


class ArrayArchiveCorruptionError(ArrayArchiveError):
    """Raised when an array archive is incomplete, corrupt, or noncanonical."""


def write_array_archive(
    path: str | os.PathLike[str],
    /,
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> Path:
    """Atomically write finite JSON metadata and pickle-free NumPy arrays."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for index, name in enumerate(sorted(arrays)):
        if not isinstance(name, str) or not name:
            raise TypeError("Archive array names must be non-empty strings.")
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"Archive array {name!r} cannot have object dtype.")
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payload = buffer.getvalue()
        member = f"arrays/{index:06d}.npy"
        payloads[member] = payload
        inventory[name] = {
            "member": member,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    complete_manifest = dict(manifest)
    if "arrays" in complete_manifest:
        raise ValueError("Archive manifest reserves the field 'arrays'.")
    complete_manifest["arrays"] = inventory
    try:
        manifest_payload = json.dumps(
            complete_manifest,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("Archive manifest must contain finite JSON values.") from error

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_STORED,
            strict_timestamps=False,
        ) as archive:
            archive.writestr("manifest.json", manifest_payload)
            for member in sorted(payloads):
                archive.writestr(member, payloads[member])
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_array_archive(
    path: str | os.PathLike[str],
    /,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read and checksum-validate one canonical pickle-free array archive."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            members = archive.infolist()
            member_names = [member.filename for member in members]
            if len(set(member_names)) != len(member_names):
                raise ArrayArchiveCorruptionError(
                    "Archive contains duplicate members."
                )
            if any(
                member.compress_type != zipfile.ZIP_STORED
                or member.file_size != member.compress_size
                for member in members
            ):
                raise ArrayArchiveCorruptionError(
                    "Archive members must use canonical stored encoding."
                )
            if archive.testzip() is not None:
                raise ArrayArchiveCorruptionError("Archive CRC validation failed.")
            names = set(member_names)
            if "manifest.json" not in names:
                raise ArrayArchiveCorruptionError("Archive manifest is missing.")
            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArrayArchiveCorruptionError(
                    "Archive manifest is invalid JSON."
                ) from error
            if not isinstance(manifest, dict):
                raise ArrayArchiveCorruptionError("Archive manifest must be an object.")
            inventory = manifest.get("arrays")
            if not isinstance(inventory, dict):
                raise ArrayArchiveCorruptionError(
                    "Archive array inventory is missing."
                )
            expected_members = {"manifest.json"}
            values: dict[str, np.ndarray] = {}
            for logical_name, record in inventory.items():
                if not isinstance(logical_name, str) or not isinstance(record, dict):
                    raise ArrayArchiveCorruptionError(
                        "Archive array inventory is invalid."
                    )
                member = record.get("member")
                if not isinstance(member, str) or member not in names:
                    raise ArrayArchiveCorruptionError(
                        f"Archive member for array {logical_name!r} is missing."
                    )
                expected_members.add(member)
                payload = archive.read(member)
                if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} checksum failed."
                    )
                try:
                    value = np.load(io.BytesIO(payload), allow_pickle=False)
                except (OSError, ValueError) as error:
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} is invalid."
                    ) from error
                if (
                    list(value.shape) != record.get("shape")
                    or value.dtype.str != record.get("dtype")
                ):
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} metadata is inconsistent."
                    )
                value.setflags(write=False)
                values[logical_name] = value
            if names != expected_members:
                raise ArrayArchiveCorruptionError(
                    "Archive contains unexpected members."
                )
            return manifest, values
    except ArrayArchiveError:
        raise
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile, OSError) as error:
        raise ArrayArchiveCorruptionError(
            f"Cannot read array archive {source}."
        ) from error


__all__ = [
    "ArrayArchiveCorruptionError",
    "ArrayArchiveError",
    "read_array_archive",
    "write_array_archive",
]
