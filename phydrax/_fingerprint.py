#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

import jax
import numpy as np
from jaxtyping import PyTree


def canonical_json(value: Any, /) -> str:
    """Serialize a JSON-compatible value with one deterministic representation."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_fingerprint(value: Any, /) -> str:
    """Return SHA-256 of canonical JSON, content-addressing numeric array leaves."""
    return hashlib.sha256(
        canonical_json(_canonical_payload(value)).encode("utf-8")
    ).hexdigest()


def array_tree_signature(tree: PyTree[Any], /) -> list[dict[str, Any]]:
    """Return stable array-path, shape, and dtype records for a PyTree."""
    return [
        {
            "path": path,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
        for path, array in _array_records(tree)
    ]


def array_tree_fingerprint(tree: PyTree[Any], /) -> dict[str, Any]:
    """Return a content-sensitive, JSON-compatible fingerprint for an array tree."""
    records = _array_records(tree)
    digest = hashlib.sha256()
    for path, value in records:
        contiguous = np.ascontiguousarray(value)
        metadata = canonical_json(
            {
                "path": path,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            }
        ).encode("ascii")
        payload = contiguous.tobytes(order="C")
        for chunk in (metadata, payload):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)
    return {
        "signature": [
            {
                "path": path,
                "shape": list(value.shape),
                "dtype": value.dtype.str,
            }
            for path, value in records
        ],
        "sha256": digest.hexdigest(),
    }


def _canonical_payload(value: Any, /) -> Any:
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        return {"__array__": array_tree_fingerprint(np.asarray(value))}
    if isinstance(value, Mapping):
        return {key: _canonical_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_payload(item) for item in value]
    return value


def canonical_mapping(value: Mapping[str, Any], /) -> dict[str, Any]:
    """Return an independent JSON-normalized mapping or reject unsupported values."""
    normalized = json.loads(canonical_json(dict(value)))
    if not isinstance(normalized, dict):
        raise TypeError("Canonical mapping input must serialize to a JSON object.")
    return normalized


def _array_records(tree: PyTree[Any], /) -> list[tuple[str, np.ndarray]]:
    records: list[tuple[str, np.ndarray]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        try:
            value = np.asarray(leaf)
        except (TypeError, ValueError):
            continue
        if value.dtype.hasobject:
            if isinstance(leaf, np.ndarray):
                raise TypeError(
                    f"Array leaf {jax.tree_util.keystr(path) or '<root>'} has object dtype and cannot be fingerprinted."
                )
            continue
        records.append((jax.tree_util.keystr(path) or "<root>", value))
    return records


@dataclass(frozen=True, slots=True)
class LogicalArrayChunk:
    """One bounded canonical C-order byte interval, independent of placement."""

    offset: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class LogicalArrayContent:
    """Globally logical array specification and an explicitly ordered byte stream.

    A distributed producer reconciles addressable shards into C-order intervals;
    this boundary never converts a global JAX array or retains the complete array.
    Chunk boundaries do not participate in scientific identity.
    """

    shape: tuple[int, ...]
    dtype: str
    chunks: Iterable[LogicalArrayChunk]


def logical_array_collection_digest(
    arrays: Mapping[str, LogicalArrayContent],
    /,
    *,
    maximum_chunk_bytes: int = 1 << 20,
) -> str:
    """Hash canonical logical content with the lifecycle archive's exact recipe.

    Coverage is exact, disjoint and ordered: gaps, overlaps, duplicated intervals,
    truncated streams and trailing bytes are rejected. The caller explicitly owns
    interprocess byte routing; a set of rank-local hashes is not a substitute.
    Serial and repartitioned streams produce the same ``array_collection_digest``.
    """

    limit = operator.index(maximum_chunk_bytes)
    if isinstance(maximum_chunk_bytes, bool) or limit <= 0:
        raise ValueError("maximum_chunk_bytes must be a positive integer.")
    if any(not isinstance(name, str) or not name for name in arrays):
        raise TypeError("Logical array names must be non-empty strings.")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        content = arrays[name]
        if not isinstance(content, LogicalArrayContent):
            raise TypeError("Logical arrays require LogicalArrayContent.")
        shape = tuple(operator.index(size) for size in content.shape)
        if len(shape) > 32 or any(
            isinstance(size, bool) or size < 0 for size in content.shape
        ):
            raise ValueError(
                "Logical array shapes require at most 32 non-negative extents."
            )
        dtype = np.dtype(content.dtype)
        if dtype.kind not in "biufc" or dtype.hasobject:
            raise TypeError("Logical array content must have a primitive numeric dtype.")
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False,
                "shape": shape,
            },
        )
        prefix = header.getvalue()
        byte_count = prod(shape) * dtype.itemsize
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update((len(prefix) + byte_count).to_bytes(8, "big"))
        digest.update(prefix)
        cursor = 0
        for chunk in content.chunks:
            if not isinstance(chunk, LogicalArrayChunk):
                raise TypeError("Logical content streams require LogicalArrayChunk.")
            if (
                isinstance(chunk.offset, bool)
                or operator.index(chunk.offset) != cursor
                or not isinstance(chunk.payload, bytes)
                or not 0 < len(chunk.payload) <= limit
                or cursor + len(chunk.payload) > byte_count
            ):
                raise ValueError(
                    "Logical array chunks do not provide bounded exact coverage."
                )
            digest.update(chunk.payload)
            cursor += len(chunk.payload)
        if cursor != byte_count:
            raise ValueError("Logical array chunks leave incomplete content coverage.")
    return digest.hexdigest()


def logical_array_value_collection_digest(
    arrays: Mapping[str, jax.Array | np.ndarray],
    /,
    *,
    maximum_chunk_bytes: int = 1 << 20,
    logical_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> str:
    """Bounded canonical hashing of host source data and logical JAX shards.

    NumPy sources stream C-order buffers without a device roundtrip or whole
    contiguous copy. JAX requires contiguous leading-axis shards; replicas are
    read once. Global byte traffic is explicit, not a parallel scaling claim.
    Tiled execution layouts require an explicit canonical redistribution.
    """

    from jax.experimental.multihost_utils import broadcast_one_to_all

    limit = operator.index(maximum_chunk_bytes)
    if isinstance(maximum_chunk_bytes, bool) or limit <= 0:
        raise ValueError("maximum_chunk_bytes must be a positive integer.")
    shapes = {} if logical_shapes is None else dict(logical_shapes)
    if set(shapes) - set(arrays):
        raise ValueError("Logical shape overrides refer to undeclared arrays.")

    def content(name: str, value: jax.Array | np.ndarray) -> LogicalArrayContent:
        if not isinstance(value, (jax.Array, np.ndarray)):
            raise TypeError(
                "Logical numerical hashing accepts canonical JAX or NumPy arrays."
            )
        if value.dtype.itemsize > limit:
            raise ValueError("The chunk capacity must hold at least one array element.")
        shape = shapes.get(name, value.shape)
        if (
            len(shape) != value.ndim
            or any(isinstance(size, bool) or operator.index(size) < 0 for size in shape)
            or (
                value.ndim > 0
                and (shape[1:] != value.shape[1:] or shape[0] > value.shape[0])
            )
        ):
            raise ValueError(
                "Logical shapes may trim only canonical leading-axis capacity padding."
            )
        if isinstance(value, np.ndarray):
            host = value[: shape[0]] if shape else value

            def host_chunks() -> Iterable[LogicalArrayChunk]:
                offset = 0
                iterator = np.nditer(
                    host,
                    flags=("external_loop", "buffered", "zerosize_ok"),
                    op_flags=("readonly",),
                    order="C",
                    buffersize=limit // host.dtype.itemsize,
                )
                for chunk in iterator:
                    encoded = np.ascontiguousarray(chunk).tobytes(order="C")
                    yield LogicalArrayChunk(offset, encoded)
                    offset += len(encoded)

            return LogicalArrayContent(shape, value.dtype.str, host_chunks())
        intervals: dict[tuple[int, int], jax.Device] = {}
        for device, index in value.sharding.devices_indices_map(value.shape).items():
            if not value.shape:
                start, stop = 0, 1
            else:
                if (
                    len(index) != value.ndim
                    or any(
                        not isinstance(entry, slice)
                        or entry.indices(size) != (0, size, 1)
                        for entry, size in zip(index[1:], value.shape[1:], strict=True)
                    )
                    or not isinstance(index[0], slice)
                ):
                    raise ValueError(
                        "Logical hashing requires contiguous leading-axis shards."
                    )
                start, stop, step = index[0].indices(value.shape[0])
                if step != 1:
                    raise ValueError("Logical hashing does not admit strided shards.")
            key = (start, stop)
            previous = intervals.get(key)
            if previous is None or (device.process_index, device.id) < (
                previous.process_index,
                previous.id,
            ):
                intervals[key] = device
        cursor = 0
        for start, stop in sorted(intervals):
            if start != cursor or stop < start:
                raise ValueError(
                    "Logical JAX shard intervals overlap or leave coverage gaps."
                )
            cursor = stop
        if cursor != (value.shape[0] if value.shape else 1):
            raise ValueError("Logical JAX shards leave incomplete global coverage.")
        local = {shard.device: shard.data for shard in value.addressable_shards}
        trailing = prod(value.shape[1:]) if value.shape else 1
        elements_per_chunk = limit // value.dtype.itemsize

        def chunks() -> Iterable[LogicalArrayChunk]:
            offset = 0
            for interval, device in sorted(intervals.items()):
                stop = min(interval[1], shape[0]) if shape else interval[1]
                count = max(0, stop - interval[0]) * trailing
                source = device.process_index == jax.process_index()
                for first in range(0, count, elements_per_chunk):
                    last = min(first + elements_per_chunk, count)
                    if source:
                        shard = local.get(device)
                        if shard is None:
                            raise ValueError(
                                "Canonical shard owner has no addressable payload."
                            )
                        payload = np.asarray(shard.reshape(-1)[first:last])
                    else:
                        payload = np.empty((last - first,), dtype=value.dtype)
                    if jax.process_count() > 1 and not value.is_fully_addressable:
                        payload = broadcast_one_to_all(payload, is_source=source)
                    encoded = np.ascontiguousarray(payload).tobytes(order="C")
                    yield LogicalArrayChunk(offset, encoded)
                    offset += len(encoded)

        return LogicalArrayContent(shape, value.dtype.str, chunks())

    return logical_array_collection_digest(
        {name: content(name, value) for name, value in arrays.items()},
        maximum_chunk_bytes=limit,
    )


__all__ = [
    "array_tree_fingerprint",
    "array_tree_signature",
    "canonical_fingerprint",
    "canonical_json",
    "canonical_mapping",
    "LogicalArrayChunk",
    "LogicalArrayContent",
    "logical_array_collection_digest",
    "logical_array_value_collection_digest",
]
