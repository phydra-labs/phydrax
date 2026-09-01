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

import jax
import jax.numpy as jnp
import numpy as np


class ArrayArchiveError(RuntimeError):
    """Base class for portable array-archive failures."""


class ArrayArchiveCorruptionError(ArrayArchiveError):
    """Raised when an array archive is incomplete, corrupt, or noncanonical."""


def array_payload_digest(value: Any, /) -> str:
    """Return the checksum used by the canonical archive for one array."""

    return hashlib.sha256(_array_payload(value)).hexdigest()


def array_payload_byte_count(value: Any, /) -> int:
    """Return the encoded byte count used by the canonical archive."""

    return len(_array_payload(value))


def array_collection_digest(arrays: Mapping[str, Any], /) -> str:
    """Content-address an ordered logical collection of canonical array payloads."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise TypeError("Archive array names must be non-empty strings.")
        payload = _array_payload(arrays[name])
        for part in (name.encode("utf-8"), payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _array_payload(value: Any, /) -> bytes:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Archive arrays cannot have object dtype.")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_stored_member(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
    /,
) -> None:
    information = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_STORED
    information.create_system = 3
    information.external_attr = 0o600 << 16
    archive.writestr(information, payload)


def pack_array_tree(
    prefix: str,
    tree: Any,
    arrays: dict[str, object],
    /,
) -> dict[str, object]:
    """Pack one array-only PyTree into a deterministic archive namespace."""

    identifier = str(prefix)
    if not identifier:
        raise ValueError("Array-tree archive prefix must be nonempty.")
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    paths: list[str] = []
    names: list[str] = []
    for index, (path, leaf) in enumerate(path_leaves):
        value = np.asarray(leaf)
        if value.dtype.hasobject:
            raise TypeError("Archived PyTrees cannot contain object arrays.")
        name = f"{identifier}/{index:06d}"
        paths.append(jax.tree_util.keystr(path) or "<root>")
        names.append(name)
        arrays[name] = value
    return {"paths": paths, "arrays": names, "num_leaves": len(names)}


def unpack_array_tree(
    specification: Mapping[str, Any],
    arrays: Mapping[str, Any],
    template: Any,
    /,
) -> Any:
    """Restore one array-only PyTree against an exact runtime template."""
    if not isinstance(specification, Mapping):
        raise ValueError("Archived PyTree specification must be a mapping.")

    template_path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    expected_paths = [
        jax.tree_util.keystr(path) or "<root>" for path, _ in template_path_leaves
    ]
    names = specification.get("arrays")
    if (
        specification.get("paths") != expected_paths
        or specification.get("num_leaves") != len(expected_paths)
        or not isinstance(names, list)
        or len(names) != len(expected_paths)
    ):
        raise ValueError("Archived PyTree does not match the runtime template.")
    leaves = []
    for name, (_, template_leaf) in zip(names, template_path_leaves, strict=True):
        if not isinstance(name, str) or name not in arrays:
            raise ValueError("Archived PyTree array is missing.")
        value = jnp.asarray(arrays[name])
        expected = jnp.asarray(template_leaf)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError("Archived PyTree array shape or dtype changed.")
        leaves.append(value)
    return jax.tree_util.tree_unflatten(treedef, leaves)


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
        payload = _array_payload(array)
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
            _write_stored_member(archive, "manifest.json", manifest_payload)
            for member in sorted(payloads):
                _write_stored_member(archive, member, payloads[member])
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
                raise ArrayArchiveCorruptionError("Archive contains duplicate members.")
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
                raise ArrayArchiveCorruptionError("Archive array inventory is missing.")
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
                if list(value.shape) != record.get(
                    "shape"
                ) or value.dtype.str != record.get("dtype"):
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} metadata is inconsistent."
                    )
                value.setflags(write=False)
                values[logical_name] = value
            if names != expected_members:
                raise ArrayArchiveCorruptionError("Archive contains unexpected members.")
            return manifest, values
    except ArrayArchiveError:
        raise
    except (FileNotFoundError, PermissionError, zipfile.BadZipFile, OSError) as error:
        raise ArrayArchiveCorruptionError(
            f"Cannot read array archive {source}."
        ) from error


__all__ = [
    "array_collection_digest",
    "array_payload_byte_count",
    "array_payload_digest",
    "ArrayArchiveCorruptionError",
    "ArrayArchiveError",
    "read_array_archive",
    "pack_array_tree",
    "unpack_array_tree",
    "write_array_archive",
]
