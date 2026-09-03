#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import struct
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

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


@dataclass(frozen=True, slots=True)
class ArrayArchiveLimits:
    """Pre-allocation admission limits for one untrusted array archive."""

    max_container_bytes: int = 1_073_741_824
    max_aggregate_bytes: int = 1_073_741_824
    max_member_bytes: int = 268_435_456
    max_manifest_bytes: int = 1_048_576
    max_members: int = 257
    max_central_directory_bytes: int = 1_048_576
    max_npy_header_bytes: int = 65_536
    max_npy_header_nesting: int = 16
    max_array_rank: int = 8
    max_axis_length: int = 67_108_864
    max_array_elements: int = 67_108_864
    max_total_array_elements: int = 268_435_456
    max_dtype_itemsize: int = 16
    max_manifest_nesting: int = 16
    allow_structured_dtypes: bool = False
    allowed_dtype_kinds: frozenset[str] = frozenset({"b", "i", "u", "f", "c"})

    def __post_init__(self) -> None:
        integer_limits = (
            self.max_container_bytes,
            self.max_aggregate_bytes,
            self.max_member_bytes,
            self.max_manifest_bytes,
            self.max_members,
            self.max_central_directory_bytes,
            self.max_npy_header_bytes,
            self.max_npy_header_nesting,
            self.max_array_rank,
            self.max_axis_length,
            self.max_array_elements,
            self.max_total_array_elements,
            self.max_dtype_itemsize,
            self.max_manifest_nesting,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_limits
        ):
            raise ValueError("Array archive limits must be positive integers.")
        if not isinstance(self.allow_structured_dtypes, bool):
            raise ValueError("allow_structured_dtypes must be boolean.")
        if (
            not isinstance(self.allowed_dtype_kinds, frozenset)
            or not self.allowed_dtype_kinds
            or any(
                not isinstance(kind, str) or len(kind) != 1
                for kind in self.allowed_dtype_kinds
            )
        ):
            raise ValueError(
                "allowed_dtype_kinds must be a non-empty frozenset of dtype kinds."
            )


DEFAULT_ARRAY_ARCHIVE_LIMITS = ArrayArchiveLimits()


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


_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
_ZIP_END_SIGNATURE = b"PK\x05\x06"
_NPY_MAGIC = b"\x93NUMPY"
_HASH_CHUNK_BYTES = 1_048_576


def _validate_zip_container(
    stream: BinaryIO,
    limits: ArrayArchiveLimits,
    /,
) -> None:
    file_size = os.fstat(stream.fileno()).st_size
    if file_size > limits.max_container_bytes:
        raise ArrayArchiveCorruptionError("Archive exceeds the container byte limit.")
    if file_size < _END_OF_CENTRAL_DIRECTORY.size:
        raise ArrayArchiveCorruptionError("Archive ZIP directory is missing.")
    tail_size = min(file_size, 65_557)
    stream.seek(file_size - tail_size)
    tail = stream.read(tail_size)
    offset = tail.rfind(_ZIP_END_SIGNATURE)
    if offset < 0 or len(tail) - offset < _END_OF_CENTRAL_DIRECTORY.size:
        raise ArrayArchiveCorruptionError("Archive ZIP directory is missing.")
    (
        signature,
        disk_number,
        directory_disk,
        entries_on_disk,
        entry_count,
        directory_size,
        directory_offset,
        comment_size,
    ) = _END_OF_CENTRAL_DIRECTORY.unpack_from(tail, offset)
    if (
        signature != _ZIP_END_SIGNATURE
        or disk_number != 0
        or directory_disk != 0
        or entries_on_disk != entry_count
        or comment_size != len(tail) - offset - _END_OF_CENTRAL_DIRECTORY.size
    ):
        raise ArrayArchiveCorruptionError("Archive ZIP directory is noncanonical.")
    if entry_count == 0xFFFF or directory_size == 0xFFFFFFFF:
        raise ArrayArchiveCorruptionError("ZIP64 array archives are not accepted.")
    if entry_count > limits.max_members:
        raise ArrayArchiveCorruptionError("Archive exceeds the member count limit.")
    if directory_size > limits.max_central_directory_bytes:
        raise ArrayArchiveCorruptionError(
            "Archive exceeds the central-directory byte limit."
        )
    if directory_offset + directory_size > file_size - _END_OF_CENTRAL_DIRECTORY.size:
        raise ArrayArchiveCorruptionError("Archive ZIP directory is invalid.")
    stream.seek(0)


@contextmanager
def _preflight_zip_container(
    source: Path,
    limits: ArrayArchiveLimits,
    /,
) -> Iterator[BinaryIO]:
    with source.open("rb") as stream:
        _validate_zip_container(stream, limits)
        yield stream


def _preflight_members(
    archive: zipfile.ZipFile,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[list[zipfile.ZipInfo], zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > limits.max_members:
        raise ArrayArchiveCorruptionError("Archive exceeds the member count limit.")
    member_names = [member.filename for member in members]
    if len(set(member_names)) != len(member_names):
        raise ArrayArchiveCorruptionError("Archive contains duplicate members.")
    if any(
        member.compress_type != zipfile.ZIP_STORED
        or member.file_size != member.compress_size
        or member.flag_bits & 0x1
        or member.is_dir()
        for member in members
    ):
        raise ArrayArchiveCorruptionError(
            "Archive members must use canonical stored, unencrypted encoding."
        )
    if sum(member.file_size for member in members) > limits.max_aggregate_bytes:
        raise ArrayArchiveCorruptionError(
            "Archive exceeds the aggregate member byte limit."
        )
    manifests = [member for member in members if member.filename == "manifest.json"]
    if len(manifests) != 1:
        raise ArrayArchiveCorruptionError("Archive manifest is missing or duplicated.")
    manifest_info = manifests[0]
    if any(
        member.file_size > limits.max_member_bytes
        for member in members
        if member is not manifest_info
    ):
        raise ArrayArchiveCorruptionError("Archive member exceeds the member byte limit.")
    if manifest_info.file_size > limits.max_manifest_bytes:
        raise ArrayArchiveCorruptionError(
            "Archive manifest exceeds the manifest byte limit."
        )
    return members, manifest_info


def _validate_json_nesting(payload: str, maximum: int, /) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in payload:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum:
                raise ArrayArchiveCorruptionError(
                    "Archive manifest exceeds the nesting limit."
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ArrayArchiveCorruptionError(
                    "Archive manifest JSON nesting is invalid."
                )


def _validate_npy_header_nesting(payload: str, maximum: int, /) -> None:
    depth = 0
    quote = ""
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    openers: list[str] = []
    for character in payload:
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
        elif character in "\"'":
            quote = character
        elif character in "([{":
            openers.append(character)
            depth += 1
            if depth > maximum:
                raise ArrayArchiveCorruptionError(
                    "Archive NPY header exceeds the nesting limit."
                )
        elif character in ")]}":
            if not openers or openers.pop() != pairs[character]:
                raise ArrayArchiveCorruptionError(
                    "Archive NPY header nesting is invalid."
                )
            depth -= 1


def _validate_json_values(value: Any, /) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            if any(not isinstance(key, str) or not key for key in current):
                raise ArrayArchiveCorruptionError(
                    "Archive manifest keys must be non-empty strings."
                )
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, float) and not math.isfinite(current):
            raise ArrayArchiveCorruptionError("Archive manifest numbers must be finite.")
        elif current is not None and not isinstance(current, (bool, int, float, str)):
            raise ArrayArchiveCorruptionError(
                "Archive manifest contains a non-JSON value."
            )


def _read_exact(stream: BinaryIO, size: int, /) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ArrayArchiveCorruptionError("Archive NPY header is truncated.")
    return payload


def _read_npy_metadata(
    stream: BinaryIO,
    member_size: int,
    limits: ArrayArchiveLimits,
    /,
) -> tuple[np.dtype[Any], tuple[int, ...], int]:
    if _read_exact(stream, len(_NPY_MAGIC)) != _NPY_MAGIC:
        raise ArrayArchiveCorruptionError("Archive array has invalid NPY magic.")
    major, minor = _read_exact(stream, 2)
    version = (major, minor)
    if version == (1, 0):
        header_size = struct.unpack("<H", _read_exact(stream, 2))[0]
        encoding = "latin1"
    elif version in ((2, 0), (3, 0)):
        header_size = struct.unpack("<I", _read_exact(stream, 4))[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise ArrayArchiveCorruptionError("Archive array NPY version is unsupported.")
    if header_size > limits.max_npy_header_bytes:
        raise ArrayArchiveCorruptionError(
            "Archive array exceeds the NPY header byte limit."
        )
    try:
        header_text = _read_exact(stream, header_size).decode(encoding)
        _validate_npy_header_nesting(header_text, limits.max_npy_header_nesting)
        header = ast.literal_eval(header_text)
    except (RecursionError, SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "Archive array NPY header is invalid."
        ) from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise ArrayArchiveCorruptionError("Archive array NPY header is noncanonical.")
    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or any(
            isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
            for extent in shape
        )
        or not isinstance(header["fortran_order"], bool)
    ):
        raise ArrayArchiveCorruptionError("Archive array shape metadata is invalid.")
    if len(shape) > limits.max_array_rank:
        raise ArrayArchiveCorruptionError("Archive array exceeds the rank limit.")
    elements = 1
    for extent in shape:
        if extent > limits.max_axis_length:
            raise ArrayArchiveCorruptionError(
                "Archive array shape exceeds the axis-length limit."
            )
        if extent and elements > limits.max_array_elements // extent:
            raise ArrayArchiveCorruptionError("Archive array exceeds the element limit.")
        elements *= extent
    try:
        dtype = np.dtype(header["descr"])
    except (TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "Archive array dtype metadata is invalid."
        ) from error
    if (
        dtype.hasobject
        or (
            (dtype.fields is not None or dtype.subdtype is not None)
            and not limits.allow_structured_dtypes
        )
        or dtype.metadata is not None
        or dtype.kind not in limits.allowed_dtype_kinds
        or dtype.itemsize > limits.max_dtype_itemsize
    ):
        raise ArrayArchiveCorruptionError(
            "Archive array dtype is not admitted by policy."
        )
    expected_size = stream.tell() + elements * dtype.itemsize
    if expected_size != member_size:
        raise ArrayArchiveCorruptionError(
            "Archive array byte size is inconsistent with its NPY metadata."
        )
    return dtype, shape, elements


def _member_sha256(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    /,
) -> str:
    digest = hashlib.sha256()
    with archive.open(member, mode="r") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def read_array_archive(
    path: str | os.PathLike[str],
    /,
    *,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one archive after bounded preflight and checksum validation.

    The default limits admit untrusted archives conservatively. Pass ``None``
    explicitly only for a trusted local archive that must retain legacy,
    effectively unbounded size limits.
    """
    source = Path(path)
    policy = limits
    if policy is not None and not isinstance(policy, ArrayArchiveLimits):
        raise TypeError("limits must be ArrayArchiveLimits or None.")
    if policy is None:
        policy = ArrayArchiveLimits(
            max_container_bytes=2**63 - 1,
            max_aggregate_bytes=2**63 - 1,
            max_member_bytes=2**63 - 1,
            max_manifest_bytes=2**63 - 1,
            max_members=2**31 - 1,
            max_central_directory_bytes=2**63 - 1,
            max_npy_header_bytes=2**31 - 1,
            max_npy_header_nesting=2**31 - 1,
            max_array_rank=2**31 - 1,
            max_axis_length=2**63 - 1,
            max_array_elements=2**63 - 1,
            max_total_array_elements=2**63 - 1,
            max_dtype_itemsize=2**31 - 1,
            max_manifest_nesting=2**31 - 1,
            allowed_dtype_kinds=frozenset("?biufcmMUSV"),
            allow_structured_dtypes=True,
        )
    try:
        with (
            _preflight_zip_container(source, policy) as container,
            zipfile.ZipFile(container, mode="r") as archive,
        ):
            members, manifest_info = _preflight_members(archive, policy)
            try:
                manifest_text = archive.read(manifest_info).decode("utf-8")
            except UnicodeDecodeError as error:
                raise ArrayArchiveCorruptionError(
                    "Archive manifest is invalid UTF-8."
                ) from error
            _validate_json_nesting(manifest_text, policy.max_manifest_nesting)
            try:
                manifest = json.loads(manifest_text)
            except (json.JSONDecodeError, RecursionError) as error:
                raise ArrayArchiveCorruptionError(
                    "Archive manifest is invalid finite JSON."
                ) from error
            if not isinstance(manifest, dict):
                raise ArrayArchiveCorruptionError("Archive manifest must be an object.")
            _validate_json_values(manifest)
            inventory = manifest.get("arrays")
            if not isinstance(inventory, dict):
                raise ArrayArchiveCorruptionError("Archive array inventory is missing.")
            member_by_name = {member.filename: member for member in members}
            expected_members = {"manifest.json"}
            admitted: dict[
                str, tuple[zipfile.ZipInfo, np.dtype[Any], tuple[int, ...]]
            ] = {}
            total_elements = 0
            for logical_name, record in inventory.items():
                if (
                    not isinstance(logical_name, str)
                    or not logical_name
                    or not isinstance(record, dict)
                    or set(record) != {"member", "shape", "dtype", "sha256"}
                ):
                    raise ArrayArchiveCorruptionError(
                        "Archive array inventory is invalid."
                    )
                member_name = record["member"]
                if (
                    not isinstance(member_name, str)
                    or member_name == "manifest.json"
                    or member_name not in member_by_name
                    or member_name in expected_members
                ):
                    raise ArrayArchiveCorruptionError(
                        f"Archive member for array {logical_name!r} is invalid."
                    )
                checksum = record["sha256"]
                if (
                    not isinstance(checksum, str)
                    or len(checksum) != 64
                    or any(character not in "0123456789abcdef" for character in checksum)
                ):
                    raise ArrayArchiveCorruptionError(
                        f"Archive checksum for array {logical_name!r} is invalid."
                    )
                member = member_by_name[member_name]
                with archive.open(member, mode="r") as stream:
                    dtype, shape, elements = _read_npy_metadata(
                        stream, member.file_size, policy
                    )
                record_shape = record["shape"]
                record_dtype = record["dtype"]
                if (
                    not isinstance(record_shape, list)
                    or any(type(extent) is not int for extent in record_shape)
                    or not isinstance(record_dtype, str)
                    or record_shape != list(shape)
                    or record_dtype != dtype.str
                ):
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} metadata is inconsistent."
                    )
                if total_elements > policy.max_total_array_elements - elements:
                    raise ArrayArchiveCorruptionError(
                        "Archive arrays exceed the aggregate element limit."
                    )
                total_elements += elements
                expected_members.add(member_name)
                admitted[logical_name] = (member, dtype, shape)
            if set(member_by_name) != expected_members:
                raise ArrayArchiveCorruptionError("Archive contains unexpected members.")

            values: dict[str, np.ndarray] = {}
            for logical_name, record in inventory.items():
                member, dtype, shape = admitted[logical_name]
                if _member_sha256(archive, member) != record["sha256"]:
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} checksum failed."
                    )
                try:
                    with archive.open(member, mode="r") as stream:
                        value = np.load(
                            stream,
                            allow_pickle=False,
                            max_header_size=policy.max_npy_header_bytes,
                        )
                except (EOFError, OSError, ValueError) as error:
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} is invalid."
                    ) from error
                if value.shape != shape or value.dtype != dtype:
                    raise ArrayArchiveCorruptionError(
                        f"Archive array {logical_name!r} metadata changed while loading."
                    )
                value.setflags(write=False)
                values[logical_name] = value
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
    "ArrayArchiveLimits",
    "DEFAULT_ARRAY_ARCHIVE_LIMITS",
    "read_array_archive",
    "pack_array_tree",
    "unpack_array_tree",
    "write_array_archive",
]
