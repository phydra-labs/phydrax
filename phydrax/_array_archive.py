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


_DEFAULT_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_ARRAY_BYTES = 4 * 1024 * 1024 * 1024
_DEFAULT_MAX_TOTAL_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_MAX_MEMBERS = 100_001


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
    *,
    max_manifest_bytes: int = _DEFAULT_MAX_MANIFEST_BYTES,
    max_array_bytes: int = _DEFAULT_MAX_ARRAY_BYTES,
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
    max_members: int = _DEFAULT_MAX_MEMBERS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read a checksum-validated archive without performing unbounded reads."""
    source = Path(path)
    manifest_limit = _positive_archive_limit(max_manifest_bytes, "max_manifest_bytes")
    array_limit = _positive_archive_limit(max_array_bytes, "max_array_bytes")
    total_limit = _positive_archive_limit(max_total_bytes, "max_total_bytes")
    member_limit = _positive_archive_limit(max_members, "max_members")
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            members = archive.infolist()
            if len(members) > member_limit:
                raise ArrayArchiveCorruptionError(
                    "Archive contains too many members for the configured bound."
                )
            member_names = [member.filename for member in members]
            if len(set(member_names)) != len(member_names):
                raise ArrayArchiveCorruptionError("Archive contains duplicate members.")
            if any(
                member.is_dir()
                or member.compress_type != zipfile.ZIP_STORED
                or member.flag_bits & 0x1
                or member.file_size != member.compress_size
                or not _canonical_member_name(member.filename)
                for member in members
            ):
                raise ArrayArchiveCorruptionError(
                    "Archive members must be canonical bounded stored files."
                )
            if sum(member.file_size for member in members) > total_limit:
                raise ArrayArchiveCorruptionError(
                    "Archive exceeds the configured total byte bound."
                )
            names = set(member_names)
            if "manifest.json" not in names:
                raise ArrayArchiveCorruptionError("Archive manifest is missing.")
            manifest_payload = _read_zip_member(archive, "manifest.json", manifest_limit)
            try:
                manifest = json.loads(
                    manifest_payload,
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
                ValueError,
            ) as error:
                raise ArrayArchiveCorruptionError(
                    "Archive manifest is invalid finite JSON."
                ) from error
            if not isinstance(manifest, dict):
                raise ArrayArchiveCorruptionError("Archive manifest must be an object.")
            inventory = manifest.get("arrays")
            if not isinstance(inventory, dict):
                raise ArrayArchiveCorruptionError("Archive array inventory is missing.")
            expected_members = {"manifest.json"}
            values: dict[str, np.ndarray] = {}
            for logical_name, record in inventory.items():
                if (
                    not isinstance(logical_name, str)
                    or not logical_name
                    or not isinstance(record, dict)
                ):
                    raise ArrayArchiveCorruptionError(
                        "Archive array inventory is invalid."
                    )
                member = record.get("member")
                if (
                    not isinstance(member, str)
                    or member not in names
                    or member in expected_members
                ):
                    raise ArrayArchiveCorruptionError(
                        f"Archive member for array {logical_name!r} is missing or reused."
                    )
                expected_members.add(member)
                payload = _read_zip_member(archive, member, array_limit)
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
                    value.dtype.hasobject
                    or list(value.shape) != record.get("shape")
                    or value.dtype.str != record.get("dtype")
                ):
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


def _read_zip_member(
    archive: zipfile.ZipFile,
    name: str,
    maximum_bytes: int,
    /,
) -> bytes:
    information = archive.getinfo(name)
    if information.file_size > maximum_bytes:
        raise ArrayArchiveCorruptionError(
            f"Archive member {name!r} exceeds its configured byte bound."
        )
    with archive.open(information, mode="r") as stream:
        payload = stream.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes or len(payload) != information.file_size:
        raise ArrayArchiveCorruptionError(
            f"Archive member {name!r} changed size while reading."
        )
    return payload


def _canonical_member_name(name: str, /) -> bool:
    if name == "manifest.json":
        return True
    parts = name.split("/")
    return (
        len(parts) == 2
        and parts[0] == "arrays"
        and len(parts[1]) == 10
        and parts[1].endswith(".npy")
        and parts[1][:6].isdigit()
    )


def _unique_json_object(
    pairs: list[tuple[str, object]],
    /,
) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"Duplicate JSON member {name!r} is forbidden.")
        value[name] = item
    return value


def _reject_json_constant(value: str, /) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")


def _positive_archive_limit(value: int, name: str, /) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive.")
    return normalized


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
