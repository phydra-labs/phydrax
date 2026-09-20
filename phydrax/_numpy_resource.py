#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pre-allocation-safe NPY and NPZ decoding for external resources."""

from __future__ import annotations

import ast
import io
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._external_resource import (
    bounded_resource_from_bytes,
    BoundedResource,
    ResourceLimits,
    ResourceReadError,
)
from ._resource_archive import admit_zip_resource, ArchiveLimits, read_zip_members


_NPY_MAGIC = b"\x93NUMPY"


@dataclass(frozen=True, slots=True)
class NumpyFormatLimits:
    """Finite metadata and allocation bounds for external NumPy resources."""

    max_container_bytes: int = 1_073_741_824
    max_aggregate_bytes: int = 1_073_741_824
    max_array_bytes: int = 268_435_456
    max_arrays: int = 257
    max_header_bytes: int = 65_536
    max_header_nesting: int = 16
    max_rank: int = 8
    max_axis_length: int = 67_108_864
    max_array_elements: int = 67_108_864
    max_total_elements: int = 268_435_456
    max_dtype_itemsize: int = 16
    allowed_dtype_kinds: frozenset[str] = frozenset({"b", "i", "u", "f", "c"})
    allow_structured_dtypes: bool = False

    def __post_init__(self) -> None:
        values = (
            self.max_container_bytes,
            self.max_aggregate_bytes,
            self.max_array_bytes,
            self.max_arrays,
            self.max_header_bytes,
            self.max_header_nesting,
            self.max_rank,
            self.max_axis_length,
            self.max_array_elements,
            self.max_total_elements,
            self.max_dtype_itemsize,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("NumPy format limits must be positive integers.")
        if self.max_array_bytes > self.max_aggregate_bytes:
            raise ValueError("NumPy array bytes cannot exceed aggregate bytes.")
        if not self.allowed_dtype_kinds or any(
            not isinstance(kind, str) or len(kind) != 1
            for kind in self.allowed_dtype_kinds
        ):
            raise ValueError("NumPy allowed dtype kinds are invalid.")


DEFAULT_NUMPY_FORMAT_LIMITS = NumpyFormatLimits()


@dataclass(frozen=True, slots=True)
class DecodedNumpyArray:
    """One defensively copied, read-only external NPY array."""

    value: np.ndarray
    resource: BoundedResource


@dataclass(frozen=True, slots=True)
class DecodedNumpyArchive:
    """Exact read-only arrays decoded from one admitted external NPZ resource."""

    arrays: dict[str, np.ndarray]
    resource: BoundedResource


def decode_npy_resource(
    resource: BoundedResource,
    /,
    *,
    limits: NumpyFormatLimits = DEFAULT_NUMPY_FORMAT_LIMITS,
) -> DecodedNumpyArray:
    """Decode NPY only after shape, dtype, element, and byte admission."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if resource.manifest.size_bytes > limits.max_array_bytes:
        raise ResourceReadError("limit", "NPY resource exceeds its array byte limit.")
    dtype, shape, _ = _read_npy_metadata(resource.data, limits)
    try:
        value = np.load(
            io.BytesIO(resource.data),
            allow_pickle=False,
            max_header_size=limits.max_header_bytes,
        )
    except (EOFError, OSError, ValueError) as error:
        raise ResourceReadError("malformed", "NPY resource is invalid.") from error
    if value.dtype != dtype or value.shape != shape:
        raise ResourceReadError("inconsistent", "NPY metadata changed during decoding.")
    defensive = np.array(value, copy=True, order="C", subok=False)
    defensive.setflags(write=False)
    return DecodedNumpyArray(defensive, resource)


def decode_npz_resource(
    resource: BoundedResource,
    /,
    *,
    limits: NumpyFormatLimits = DEFAULT_NUMPY_FORMAT_LIMITS,
    expected_names: tuple[str, ...] | None = None,
) -> DecodedNumpyArchive:
    """Decode an exact external NPZ inventory without permitting object arrays."""

    archive = admit_zip_resource(
        resource,
        limits=ArchiveLimits(
            max_container_bytes=limits.max_container_bytes,
            max_members=limits.max_arrays,
            max_member_bytes=limits.max_array_bytes,
            max_total_uncompressed_bytes=limits.max_aggregate_bytes,
            max_name_bytes=1024,
            max_depth=1,
            max_compression_ratio=1000,
        ),
    )
    member_names = tuple(member.relative_path for member in archive.members)
    if any(not name.endswith(".npy") or "/" in name for name in member_names):
        raise ResourceReadError("malformed", "NPZ contains a noncanonical member name.")
    logical_names = tuple(name[:-4] for name in member_names)
    if any(not name for name in logical_names) or len(logical_names) != len(
        set(logical_names)
    ):
        raise ResourceReadError("malformed", "NPZ logical array names are invalid.")
    if expected_names is not None:
        if isinstance(expected_names, str) or len(expected_names) != len(
            set(expected_names)
        ):
            raise ValueError("expected_names must be a unique tuple or None.")
        if set(logical_names) != set(expected_names):
            raise ResourceReadError("policy", "NPZ inventory does not match expectation.")
    payloads = read_zip_members(archive, member_names)
    arrays: dict[str, np.ndarray] = {}
    total_elements = 0
    for member_name, logical_name in zip(member_names, logical_names, strict=True):
        payload = payloads[member_name]
        source_limits = resource.manifest.limits
        member_resource = bounded_resource_from_bytes(
            payload,
            limits=ResourceLimits(
                max(source_limits.max_bytes, limits.max_array_bytes),
                source_limits.max_depth,
                source_limits.max_nodes,
                source_limits.max_attributes,
                source_limits.max_losses,
            ),
            source_path=f"{resource.manifest.source_path or '<memory>'}!/{member_name}",
        )
        decoded = decode_npy_resource(member_resource, limits=limits)
        elements = int(decoded.value.size)
        if total_elements > limits.max_total_elements - elements:
            raise ResourceReadError(
                "limit", "NPZ arrays exceed their aggregate element limit."
            )
        total_elements += elements
        arrays[logical_name] = decoded.value
    return DecodedNumpyArchive(arrays, resource)


def _read_npy_metadata(
    payload: bytes,
    limits: NumpyFormatLimits,
    /,
) -> tuple[np.dtype[Any], tuple[int, ...], int]:
    stream = io.BytesIO(payload)
    if _read_exact(stream, len(_NPY_MAGIC)) != _NPY_MAGIC:
        raise ResourceReadError("malformed", "NPY resource has invalid magic.")
    major, minor = _read_exact(stream, 2)
    version = (major, minor)
    if version == (1, 0):
        header_size = struct.unpack("<H", _read_exact(stream, 2))[0]
        encoding = "latin1"
    elif version in ((2, 0), (3, 0)):
        header_size = struct.unpack("<I", _read_exact(stream, 4))[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise ResourceReadError("malformed", "NPY version is unsupported.")
    if header_size > limits.max_header_bytes:
        raise ResourceReadError("limit", "NPY header exceeds its byte limit.")
    try:
        header_text = _read_exact(stream, header_size).decode(encoding)
        _validate_header_nesting(header_text, limits.max_header_nesting)
        header = ast.literal_eval(header_text)
    except (RecursionError, SyntaxError, UnicodeDecodeError, ValueError) as error:
        raise ResourceReadError("malformed", "NPY header is invalid.") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise ResourceReadError("malformed", "NPY header is noncanonical.")
    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or any(type(extent) is not int or extent < 0 for extent in shape)
        or not isinstance(header["fortran_order"], bool)
    ):
        raise ResourceReadError("malformed", "NPY shape metadata is invalid.")
    if len(shape) > limits.max_rank:
        raise ResourceReadError("limit", "NPY array exceeds its rank limit.")
    elements = 1
    for extent in shape:
        if extent > limits.max_axis_length:
            raise ResourceReadError("limit", "NPY shape exceeds its axis-length limit.")
        if extent and elements > limits.max_array_elements // extent:
            raise ResourceReadError("limit", "NPY array exceeds its element limit.")
        elements *= extent
    try:
        dtype = np.dtype(header["descr"])
    except (TypeError, ValueError) as error:
        raise ResourceReadError("malformed", "NPY dtype metadata is invalid.") from error
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
        raise ResourceReadError("policy", "NPY dtype is not admitted.")
    expected_size = stream.tell() + elements * dtype.itemsize
    if expected_size != len(payload):
        raise ResourceReadError(
            "malformed", "NPY byte size is inconsistent with metadata."
        )
    return dtype, shape, elements


def _read_exact(stream: io.BytesIO, size: int, /) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ResourceReadError("malformed", "NPY header is truncated.")
    return payload


def _validate_header_nesting(payload: str, maximum: int, /) -> None:
    quote = ""
    escaped = False
    openers: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
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
            if len(openers) > maximum:
                raise ResourceReadError("limit", "NPY header exceeds its nesting limit.")
        elif character in ")]}" and (not openers or openers.pop() != pairs[character]):
            raise ResourceReadError("malformed", "NPY header nesting is invalid.")
    if openers or quote:
        raise ResourceReadError("malformed", "NPY header nesting is invalid.")


__all__ = [
    "DEFAULT_NUMPY_FORMAT_LIMITS",
    "DecodedNumpyArchive",
    "DecodedNumpyArray",
    "NumpyFormatLimits",
    "decode_npy_resource",
    "decode_npz_resource",
]
