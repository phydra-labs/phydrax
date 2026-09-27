#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Binary, checksummed, memory-mappable array exchange directories.

A directory holds one little-endian C-order NPY (format 1.0) file per array,
``<name>.npy``, and ``manifest.json``: canonical JSON
``{"arrays": [{"dtype", "name", "sha256", "shape"}, ...], "parts": [...]}``
with arrays and parts sorted by name. ``sha256`` digests the raw payload after
the NPY header. A part is a nested exchange directory without parts (one per
rank of a collective worker). No other entries are permitted. The native
reader/writer is ``native/providers/common/phydrax_exchange.hpp``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._fingerprint import canonical_json


EXCHANGE_DTYPES: Mapping[str, np.dtype] = {
    "<f8": np.dtype("<f8"),
    "<i8": np.dtype("<i8"),
    "<i4": np.dtype("<i4"),
    "<u8": np.dtype("<u8"),
    "|u1": np.dtype("|u1"),
    "|i1": np.dtype("|i1"),
}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class ExchangeArrayRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ExchangeManifest:
    """Verified manifest identity of one exchange directory (parts included)."""

    arrays: tuple[ExchangeArrayRecord, ...]
    parts: tuple[str, ...]
    manifest_sha256: str
    total_bytes: int


@dataclass(frozen=True, slots=True)
class ExchangeContents:
    """Read-only verified arrays of one directory and its rank parts."""

    arrays: Mapping[str, np.ndarray]
    parts: Mapping[str, Mapping[str, np.ndarray]]
    manifest: ExchangeManifest


def _name(value: Any, role: str, /) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"Invalid exchange {role} name {value!r}.")
    return value


def _payload_digest(array: np.ndarray, /) -> str:
    # Digest the contiguous payload bytes in place (empty arrays included).
    payload = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
    return hashlib.sha256(memoryview(payload)).hexdigest()


def write_exchange(
    directory: str | os.PathLike[str],
    arrays: Mapping[str, np.ndarray],
    /,
    *,
    maximum_bytes: int,
) -> ExchangeManifest:
    """Write one exchange directory (without parts) into an existing empty path."""
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer.")
    if not isinstance(arrays, Mapping):
        raise TypeError("Exchange arrays must be a mapping of name to array.")
    root = Path(directory)
    if any(root.iterdir()):
        raise ValueError("Exchange output directory must be empty.")
    records = []
    total = 0
    for name in sorted(arrays):
        _name(name, "array")
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Exchange array {name!r} must be a NumPy array.")
        dtype = value.dtype.newbyteorder("<") if value.dtype.itemsize > 1 else value.dtype
        if dtype.str not in EXCHANGE_DTYPES or value.dtype.kind not in "fiu":
            raise TypeError(
                f"Exchange array {name!r} has unsupported dtype {value.dtype}."
            )
        payload = np.ascontiguousarray(value, dtype=EXCHANGE_DTYPES[dtype.str])
        total += payload.nbytes
        if total > maximum_bytes:
            raise ValueError("Exchange arrays exceed maximum_bytes.")
        with (root / f"{name}.npy").open("xb") as stream:
            np.lib.format.write_array(stream, payload, version=(1, 0))
        records.append(
            ExchangeArrayRecord(
                name, payload.dtype.str, tuple(payload.shape), _payload_digest(payload)
            )
        )
    manifest = canonical_json(
        {
            "arrays": [
                {
                    "dtype": record.dtype,
                    "name": record.name,
                    "sha256": record.sha256,
                    "shape": list(record.shape),
                }
                for record in records
            ],
            "parts": [],
        }
    ).encode("ascii")
    total += len(manifest)
    if total > maximum_bytes:
        raise ValueError("Exchange arrays exceed maximum_bytes.")
    # The manifest is written last: a partial directory is never well-formed.
    with (root / _MANIFEST).open("xb") as stream:
        stream.write(manifest)
    return ExchangeManifest(
        tuple(records), (), hashlib.sha256(manifest).hexdigest(), total
    )


def _decode_manifest(data: bytes, /) -> dict:
    def pairs(items: Any) -> Any:
        record = {}
        for key, value in items:
            if key in record:
                raise ValueError(f"Exchange manifest repeats field {key!r}.")
            record[key] = value
        return record

    value = json.loads(data.decode("ascii"), object_pairs_hook=pairs)
    if not isinstance(value, dict) or set(value) != {"arrays", "parts"}:
        raise ValueError("Exchange manifest fields are invalid.")
    if not isinstance(value["arrays"], list) or not isinstance(value["parts"], list):
        raise ValueError("Exchange manifest arrays and parts must be lists.")
    return value


def _record(value: Any, /) -> ExchangeArrayRecord:
    if not isinstance(value, dict) or set(value) != {"dtype", "name", "sha256", "shape"}:
        raise ValueError("Exchange array record fields are invalid.")
    name = _name(value["name"], "array")
    if value["dtype"] not in EXCHANGE_DTYPES:
        raise ValueError(f"Exchange array {name!r} has an unsupported dtype.")
    shape = value["shape"]
    if not isinstance(shape, list) or any(
        type(extent) is not int or extent < 0 for extent in shape
    ):
        raise ValueError(f"Exchange array {name!r} has an invalid shape.")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Exchange array {name!r} has an invalid checksum.")
    return ExchangeArrayRecord(name, value["dtype"], tuple(shape), digest)


def _load_array(path: Path, record: ExchangeArrayRecord, /) -> np.ndarray:
    dtype = EXCHANGE_DTYPES[record.dtype]
    payload = math.prod(record.shape) * dtype.itemsize
    with path.open("rb") as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError(f"Exchange array {record.name!r} is not NPY format 1.0.")
        shape, fortran, header_dtype = np.lib.format.read_array_header_1_0(stream)
        offset = stream.tell()
    if fortran or header_dtype != dtype or tuple(shape) != record.shape:
        raise ValueError(f"NPY header of {record.name!r} contradicts the manifest.")
    if path.stat().st_size != offset + payload:
        raise ValueError(f"Exchange array {record.name!r} has an inconsistent size.")
    array = (
        np.empty(record.shape, dtype=dtype)
        if payload == 0
        else np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=record.shape)
    )
    if _payload_digest(np.ascontiguousarray(array)) != record.sha256:
        raise ValueError(f"Exchange checksum mismatch for {record.name!r}.")
    array.flags.writeable = False
    return array


def _read_directory(
    root: Path, maximum_bytes: int, consumed: int, allow_parts: bool, /
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict, int]:
    manifest_path = root / _MANIFEST
    size = manifest_path.stat().st_size
    if consumed + size > maximum_bytes:
        raise ValueError("Exchange manifest exceeds maximum_bytes.")
    data = manifest_path.read_bytes()
    consumed += len(data)
    value = _decode_manifest(data)
    records = tuple(_record(item) for item in value["arrays"])
    names = [record.name for record in records]
    parts = [_name(item, "part") for item in value["parts"]]
    if names != sorted(set(names)) or parts != sorted(set(parts)):
        raise ValueError("Exchange arrays and parts must be unique and sorted.")
    if parts and not allow_parts:
        raise ValueError("Exchange parts cannot be nested.")
    expected = {_MANIFEST, *(f"{name}.npy" for name in names), *parts}
    if len(expected) != 1 + len(names) + len(parts):
        raise ValueError("Exchange array and part names collide.")
    entries = {entry.name for entry in os.scandir(root)}
    if entries != expected:
        raise ValueError("Exchange directory contains undeclared or missing entries.")
    arrays = {}
    for record in records:
        path = root / f"{record.name}.npy"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Exchange array {record.name!r} is not a regular file.")
        consumed += math.prod(record.shape) * EXCHANGE_DTYPES[record.dtype].itemsize
        if consumed > maximum_bytes:
            raise ValueError("Exchange arrays exceed maximum_bytes.")
        arrays[record.name] = _load_array(path, record)
    nested = {}
    for part in parts:
        path = root / part
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"Exchange part {part!r} is not a directory.")
        part_arrays, _, _, consumed = _read_directory(
            path, maximum_bytes, consumed, False
        )
        nested[part] = part_arrays
    identity = {
        "digest": hashlib.sha256(data).hexdigest(),
        "records": records,
        "parts": tuple(parts),
    }
    return arrays, nested, identity, consumed


def read_exchange(
    directory: str | os.PathLike[str], /, *, maximum_bytes: int
) -> ExchangeContents:
    """Verify and memory-map every array of one exchange directory and its parts."""
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer.")
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Exchange path must be a directory.")
    arrays, parts, identity, total = _read_directory(root, maximum_bytes, 0, True)
    return ExchangeContents(
        arrays,
        parts,
        ExchangeManifest(
            identity["records"], identity["parts"], identity["digest"], total
        ),
    )


__all__ = [
    "EXCHANGE_DTYPES",
    "ExchangeArrayRecord",
    "ExchangeContents",
    "ExchangeManifest",
    "read_exchange",
    "write_exchange",
]
