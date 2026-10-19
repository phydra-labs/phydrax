#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
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
    """Return SHA-256 of deterministic JSON with content-addressed array leaves."""
    return hashlib.sha256(
        canonical_json(_canonical_payload(value)).encode("utf-8")
    ).hexdigest()


def iter_array_bytes(value: np.ndarray, /) -> Iterator[memoryview]:
    """Yield bounded C-order byte views; consume each before advancing."""
    if value.dtype.hasobject:
        raise TypeError(
            f"iter_array_bytes.value.dtype must exclude object references; got {value.dtype!r}."
        )
    if value.nbytes == 0:
        return
    if value.flags.c_contiguous:
        payload = memoryview(value.reshape(-1).view(np.uint8))
        for start in range(0, len(payload), 65536):
            yield payload[start : start + 65536]
        return
    with np.nditer(
        value,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=max(1, 65536 // max(1, value.dtype.itemsize)),
    ) as chunks:
        for chunk in chunks:
            yield memoryview(np.ascontiguousarray(chunk).view(np.uint8))


def array_bytes_digest(value: np.ndarray, /) -> str:
    """Return SHA-256 of the C-order bytes without whole-array staging."""
    digest = hashlib.sha256()
    for chunk in iter_array_bytes(value):
        digest.update(chunk)
    return digest.hexdigest()


def canonical_array_hex_fingerprint(value: np.ndarray, /) -> str:
    """Fingerprint the JSON string of C-order hexadecimal bytes."""
    digest = hashlib.sha256(b'"')
    for chunk in iter_array_bytes(value):
        digest.update(chunk.hex().encode("ascii"))
    digest.update(b'"')
    return digest.hexdigest()


def _iter_array_list_json(value: np.ndarray, /) -> Iterator[str]:
    if value.ndim == 0:
        yield canonical_json(_canonical_payload(value.tolist()))
        return
    yield "["
    if value.ndim == 1:
        # Keep Python scalar lists and their JSON text bounded, including for
        # strided inputs. tolist owns NumPy's legacy scalar conversion semantics.
        for start in range(0, value.size, 1024):
            if start:
                yield ","
            yield canonical_json(
                _canonical_payload(value[start : start + 1024].tolist())
            )[1:-1]
    else:
        for index in range(value.shape[0]):
            if index:
                yield ","
            yield from _iter_array_list_json(value[index])
    yield "]"


def _iter_array_list_payload_json(value: Any, /) -> Iterator[str]:
    if isinstance(value, np.ndarray):
        yield from _iter_array_list_json(value)
    elif isinstance(value, Mapping):
        yield "{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield ","
            yield canonical_json(key)
            yield ":"
            yield from _iter_array_list_payload_json(value[key])
        yield "}"
    elif isinstance(value, (tuple, list)):
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from _iter_array_list_payload_json(item)
        yield "]"
    else:
        yield canonical_json(_canonical_payload(value))


def canonical_array_list_fingerprint(value: Any, /) -> str:
    """Fingerprint deterministic JSON with arrays represented by nested lists."""
    digest = hashlib.sha256()
    for chunk in _iter_array_list_payload_json(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


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
        # Legacy ascontiguousarray promoted scalar leaves to a length-one axis.
        shape = value.shape if value.ndim else (1,)
        metadata = canonical_json(
            {
                "path": path,
                "dtype": value.dtype.str,
                "shape": list(shape),
            }
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.nbytes.to_bytes(8, "big"))
        for chunk in iter_array_bytes(value):
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


type _ArrayEqual = Callable[[jax.Array | np.ndarray, jax.Array | np.ndarray], bool]


def tree_content_equal(
    first: PyTree[Any],
    second: PyTree[Any],
    /,
    *,
    array_equal: _ArrayEqual | None = None,
) -> bool:
    """Whether two PyTrees have equal structure, static metadata and leaf content.

    Numerical leaves and numerical static metadata (arrays, NumPy scalars,
    Python floats and complexes) are equal when type, shape, dtype and C-order
    bytes agree, so identical NaN payloads compare equal and any changed bit
    does not. Dataclass fields are compared recursively, including fields excluded
    from generated equality. Other leaves and static values must have the same
    type and compare equal.

    ``array_equal`` may authenticate array bytes through an admitted distributed
    bank. Array type, shape and dtype equality, and every nonarray or static
    comparison, remain this function's responsibility; the default is host-local.

    Shared subtrees are compared once per distinct pair of objects, so aliased
    subgraphs cost their distinct size rather than their expanded size.
    """
    # Proven pairs are retained for the whole comparison so their identities
    # cannot be reused by temporary children created during flattening.
    proven: dict[tuple[int, int], tuple[object, object]] = {}
    return _tree_equal(first, second, array_equal, proven)


def _tree_equal(
    first: object,
    second: object,
    array_equal: _ArrayEqual | None,
    proven: dict[tuple[int, int], tuple[object, object]],
    /,
) -> bool:
    if first is second or (id(first), id(second)) in proven:
        return True
    first_children, first_node = jax.tree_util.tree_flatten(
        first, is_leaf=lambda node: node is not first
    )
    second_children, second_node = jax.tree_util.tree_flatten(
        second, is_leaf=lambda node: node is not second
    )
    first_leaf = jax.tree_util.treedef_is_leaf(first_node)
    if first_leaf or jax.tree_util.treedef_is_leaf(second_node):
        equal = (
            first_leaf
            and jax.tree_util.treedef_is_leaf(second_node)
            and _static_equal(first, second, array_equal)
        )
    else:
        equal = (
            _node_data_equal(first_node, second_node, array_equal)
            and len(first_children) == len(second_children)
            and all(
                _tree_equal(left, right, array_equal, proven)
                for left, right in zip(first_children, second_children, strict=True)
            )
        )
    if equal:
        proven[(id(first), id(second))] = (first, second)
    return equal


def _node_data_equal(
    first: jax.tree_util.PyTreeDef,
    second: jax.tree_util.PyTreeDef,
    array_equal: _ArrayEqual | None,
    /,
) -> bool:
    # PyTreeDef equality compares static metadata with ``==``: a NaN static
    # value makes bitwise-identical trees unequal and signed zeros compare
    # equal, so node metadata is compared bitwise instead.
    first_data, second_data = first.node_data(), second.node_data()
    if first_data is None or second_data is None:
        return first_data is None and second_data is None
    return first_data[0] is second_data[0] and _static_equal(
        first_data[1], second_data[1], array_equal
    )


def _static_equal(
    first: Any, second: Any, /, array_equal: _ArrayEqual | None = None
) -> bool:
    if type(first) is not type(second):
        return False
    if array_equal is not None and isinstance(first, (jax.Array, np.ndarray)):
        return (
            first.shape == second.shape
            and first.dtype == second.dtype
            and array_equal(first, second)
        )
    if isinstance(first, (jax.Array, np.ndarray, np.generic, float, complex)):
        # Shapes are compared before `ascontiguousarray`, which promotes a 0-d
        # value to one element and would equate a scalar with a length-one axis.
        left_value, right_value = np.asarray(first), np.asarray(second)
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            return False
        left = np.ascontiguousarray(left_value).reshape(-1).view(np.uint8)
        right = np.ascontiguousarray(right_value).reshape(-1).view(np.uint8)
        return np.array_equal(left, right)
    if is_dataclass(first) and not isinstance(first, type):
        return all(
            _static_equal(
                getattr(first, field.name), getattr(second, field.name), array_equal
            )
            for field in fields(first)
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _static_equal(left, right, array_equal)
            for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, Mapping):
        if first.keys() != second.keys():
            return False
        # Python equality merges `1`, `1.0` and `True` keys; each key must match
        # its counterpart's type and bits.
        second_keys = {key: key for key in second}
        return all(
            _static_equal(key, second_keys[key], array_equal)
            and _static_equal(first[key], second[key], array_equal)
            for key in first
        )
    return bool(first == second)


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
        raise TypeError(
            f"canonical_mapping.value must serialize to a JSON object; got {type(normalized).__qualname__}."
        )
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
                    f"array_tree_fingerprint.leaf.dtype must exclude objects; got path={jax.tree_util.keystr(path) or '<root>'}, dtype={value.dtype}."
                )
            continue
        records.append((jax.tree_util.keystr(path) or "<root>", value))
    return records


@dataclass(frozen=True, slots=True)
class LogicalArrayChunk:
    """One bounded C-order byte interval, independent of placement."""

    offset: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class LogicalArrayContent:
    """Globally logical array specification and an explicitly ordered byte stream.

    A distributed producer reconciles addressable shards into C-order intervals;
    this boundary never converts a global JAX array or retains the whole array.
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
    """Hash logical content with the lifecycle archive's byte recipe.

    Coverage is disjoint and ordered: gaps, overlaps, duplicated intervals,
    truncated streams and trailing bytes are rejected. The caller explicitly owns
    interprocess byte routing; a set of rank-local hashes is not a substitute.
    Serial and repartitioned streams produce the same ``array_collection_digest``.
    """

    limit = operator.index(maximum_chunk_bytes)
    if isinstance(maximum_chunk_bytes, bool) or limit <= 0:
        raise ValueError(
            f"logical_array_collection_digest.maximum_chunk_bytes must be a positive integer; got {maximum_chunk_bytes!r}."
        )
    if any(not isinstance(name, str) or not name for name in arrays):
        raise TypeError(
            f"logical_array_collection_digest.arrays.keys must be nonempty strings; got {tuple(arrays)!r}."
        )
    digest = hashlib.sha256()
    for name in sorted(arrays):
        content = arrays[name]
        if not isinstance(content, LogicalArrayContent):
            raise TypeError(
                f"logical_array_collection_digest.content must be LogicalArrayContent; got {type(content).__qualname__}."
            )
        shape = tuple(operator.index(size) for size in content.shape)
        if len(shape) > 32 or any(
            isinstance(size, bool) or size < 0 for size in content.shape
        ):
            raise ValueError(
                f"logical_array_collection_digest.content.shape must have at most 32 nonnegative extents; got {content.shape!r}."
            )
        dtype = np.dtype(content.dtype)
        if dtype.kind not in "biufc" or dtype.hasobject:
            raise TypeError(
                f"logical_array_collection_digest.content.dtype must be primitive numeric; got {dtype}."
            )
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
                raise TypeError(
                    f"logical_array_collection_digest.chunk must be LogicalArrayChunk; got {type(chunk).__qualname__}."
                )
            if (
                isinstance(chunk.offset, bool)
                or operator.index(chunk.offset) != cursor
                or not isinstance(chunk.payload, bytes)
                or not 0 < len(chunk.payload) <= limit
                or cursor + len(chunk.payload) > byte_count
            ):
                raise ValueError(
                    f"logical_array_collection_digest.chunk must provide bounded contiguous coverage; got offset={chunk.offset!r}, payload_type={type(chunk.payload).__qualname__}, cursor={cursor}, byte_count={byte_count}, limit={limit}."
                )
            digest.update(chunk.payload)
            cursor += len(chunk.payload)
        if cursor != byte_count:
            raise ValueError(
                f"logical_array_collection_digest.chunks must cover all content bytes; got covered={cursor}, expected={byte_count}."
            )
    return digest.hexdigest()


def _logical_array_shard_intervals(
    value: jax.Array, /
) -> dict[tuple[int, int], jax.Device]:
    """Select one replica per contiguous shard and require global coverage."""
    intervals: dict[tuple[int, int], jax.Device] = {}
    for device, index in value.sharding.devices_indices_map(value.shape).items():
        if not value.shape:
            start, stop = 0, 1
        else:
            if (
                len(index) != value.ndim
                or any(
                    not isinstance(entry, slice) or entry.indices(size) != (0, size, 1)
                    for entry, size in zip(index[1:], value.shape[1:], strict=True)
                )
                or not isinstance(index[0], slice)
            ):
                raise ValueError(
                    f"logical_array_value_collection_digest.shard.index must select a contiguous leading-axis shard; got {index!r}, shape={value.shape}."
                )
            start, stop, step = index[0].indices(value.shape[0])
            if step != 1:
                raise ValueError(
                    f"logical_array_value_collection_digest.shard.step must be one; got {step}."
                )
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
                f"logical_array_value_collection_digest.shard.intervals must provide disjoint contiguous coverage; got start={start}, stop={stop}, cursor={cursor}."
            )
        cursor = stop
    if cursor != (value.shape[0] if value.shape else 1):
        raise ValueError(
            f"logical_array_value_collection_digest.shard.intervals must cover the global leading axis; got covered={cursor}, shape={value.shape}."
        )
    return intervals


def _logical_shard_array_chunks(
    value: jax.Array,
    shape: tuple[int, ...],
    intervals: dict[tuple[int, int], jax.Device],
    local: dict[jax.Device, jax.Array],
    trailing: int,
    elements_per_chunk: int,
    broadcast: Callable[..., np.ndarray],
    /,
) -> Iterable[LogicalArrayChunk]:
    """Stream the selected replicas in global C order with bounded transfers."""
    offset = 0
    for interval, device in sorted(intervals.items()):
        stop = min(interval[1], shape[0]) if shape else interval[1]
        count = max(0, stop - interval[0]) * trailing
        source = device.process_index == jax.process_index()
        shard = local.get(device) if source else None
        if source and shard is None:
            raise ValueError(
                f"logical_array_value_collection_digest.shard.payload must be addressable on the selected process; got device={device}, process={jax.process_index()}."
            )
        # CPU shard buffers already live in host memory: a zero-copy view
        # replaces per-chunk device slicing, which compiles one program per
        # distinct chunk bound. Accelerator shards keep bounded device slices.
        host_view = (
            np.asarray(shard).reshape(-1)
            if shard is not None and device.platform == "cpu"
            else None
        )
        for first in range(0, count, elements_per_chunk):
            last = min(first + elements_per_chunk, count)
            if host_view is not None:
                payload = host_view[first:last]
            elif shard is not None:
                payload = np.asarray(shard.reshape(-1)[first:last])
            else:
                payload = np.empty((last - first,), dtype=value.dtype)
            if jax.process_count() > 1 and not value.is_fully_addressable:
                payload = broadcast(payload, is_source=source)
            encoded = np.ascontiguousarray(payload).tobytes(order="C")
            yield LogicalArrayChunk(offset, encoded)
            offset += len(encoded)


def logical_array_value_collection_digest(
    arrays: Mapping[str, jax.Array | np.ndarray],
    /,
    *,
    maximum_chunk_bytes: int = 1 << 20,
    logical_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> str:
    """Hash host source data and logical JAX shards with bounded byte chunks.

    NumPy sources stream C-order buffers without a device roundtrip or whole
    contiguous copy. JAX requires contiguous leading-axis shards; replicas are
    read once. Global byte traffic is explicit, not a parallel scaling claim.
    Tiled execution layouts require an explicit C-order redistribution.
    """

    from jax.experimental.multihost_utils import broadcast_one_to_all

    limit = operator.index(maximum_chunk_bytes)
    if isinstance(maximum_chunk_bytes, bool) or limit <= 0:
        raise ValueError(
            f"logical_array_value_collection_digest.maximum_chunk_bytes must be a positive integer; got {maximum_chunk_bytes!r}."
        )
    shapes = {} if logical_shapes is None else dict(logical_shapes)
    if set(shapes) - set(arrays):
        raise ValueError(
            f"logical_array_value_collection_digest.logical_shapes must refer to declared arrays; got undeclared={set(shapes) - set(arrays)!r}."
        )

    def content(name: str, value: jax.Array | np.ndarray) -> LogicalArrayContent:
        if not isinstance(value, (jax.Array, np.ndarray)):
            raise TypeError(
                f"logical_array_value_collection_digest.value must be a JAX or NumPy array; got {type(value).__qualname__}."
            )
        if value.dtype.itemsize > limit:
            raise ValueError(
                f"logical_array_value_collection_digest.maximum_chunk_bytes must hold one array element; got limit={limit}, itemsize={value.dtype.itemsize}."
            )
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
                f"logical_array_value_collection_digest.logical_shapes must trim only leading-axis capacity padding; got logical={shape}, stored={value.shape}."
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
        intervals = _logical_array_shard_intervals(value)
        local = {shard.device: shard.data for shard in value.addressable_shards}
        trailing = prod(value.shape[1:]) if value.shape else 1
        elements_per_chunk = limit // value.dtype.itemsize
        return LogicalArrayContent(
            shape,
            value.dtype.str,
            _logical_shard_array_chunks(
                value,
                shape,
                intervals,
                local,
                trailing,
                elements_per_chunk,
                broadcast_one_to_all,
            ),
        )

    return logical_array_collection_digest(
        {name: content(name, value) for name, value in arrays.items()},
        maximum_chunk_bytes=limit,
    )


__all__ = [
    "array_tree_fingerprint",
    "array_bytes_digest",
    "array_tree_signature",
    "canonical_fingerprint",
    "canonical_json",
    "canonical_array_hex_fingerprint",
    "canonical_array_list_fingerprint",
    "canonical_mapping",
    "LogicalArrayChunk",
    "LogicalArrayContent",
    "logical_array_collection_digest",
    "logical_array_value_collection_digest",
    "iter_array_bytes",
]
