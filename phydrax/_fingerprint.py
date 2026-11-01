#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from operator import itemgetter
from typing import Any

import jax
import numpy as np
from jaxtyping import PyTree


_ENCODER = json.JSONEncoder(
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
)
_JSON_SCALAR_TYPES = frozenset((str, int, float, bool, type(None)))
_SEQUENCE_CHUNK = 4096
_DIGEST_BUFFER_CHARACTERS = 1 << 16
_ITEM_KEY = itemgetter(0)


def canonical_json(value: Any, /) -> str:
    """Serialize a JSON-compatible value with one deterministic representation."""
    return _ENCODER.encode(value)


class LazyCanonicalSequence:
    """A JSON array whose items are produced while it is written.

    ``canonical_fingerprint`` encodes it exactly as the tuple of the same items,
    without holding that tuple: large identity payloads stay bounded.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Callable[[], Iterable[Any]], /) -> None:
        self._items = items

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items())


def canonical_fingerprint(value: Any, /) -> str:
    """Return SHA-256 of canonical JSON, content-addressing numeric array leaves.

    The canonical text is streamed into the digest in bounded pieces; nested
    containers are never copied.
    """
    digest = hashlib.sha256()
    parts: list[str] = []
    buffered = 0

    def write(text: str, /) -> None:
        nonlocal buffered
        parts.append(text)
        buffered += len(text)
        if buffered >= _DIGEST_BUFFER_CHARACTERS:
            digest.update("".join(parts).encode("ascii"))
            parts.clear()
            buffered = 0

    _write_canonical(value, write)
    digest.update("".join(parts).encode("ascii"))
    return digest.hexdigest()


def iter_array_bytes(value: np.ndarray, /) -> Iterator[memoryview]:
    """Yield bounded C-order byte views; consume each before advancing."""
    if value.dtype.hasobject:
        raise TypeError("Object arrays cannot be hashed as canonical bytes.")
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
    """Fingerprint the canonical JSON string of C-order hexadecimal bytes."""
    digest = hashlib.sha256(b'"')
    for chunk in iter_array_bytes(value):
        digest.update(chunk.hex().encode("ascii"))
    digest.update(b'"')
    return digest.hexdigest()


def _iter_array_list_json(value: np.ndarray, /) -> Iterator[str]:
    if value.ndim == 0:
        yield _canonical_text(value.tolist())
        return
    yield "["
    if value.ndim == 1:
        # Keep Python scalar lists and their JSON text bounded, including for
        # strided inputs. tolist owns NumPy's legacy scalar conversion semantics.
        for start in range(0, value.size, 1024):
            if start:
                yield ","
            yield _canonical_text(value[start : start + 1024].tolist())[1:-1]
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
        yield _canonical_text(value)


def canonical_array_list_fingerprint(value: Any, /) -> str:
    """Fingerprint canonical JSON with arrays represented by legacy nested lists."""
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
    digest = hashlib.sha256()
    signature = []
    for path, value in _array_records(tree):
        signature.append(
            {"path": path, "shape": list(value.shape), "dtype": value.dtype.str}
        )
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
        "signature": signature,
        "sha256": digest.hexdigest(),
    }


def _canonical_text(value: Any, /) -> str:
    parts: list[str] = []
    _write_canonical(value, parts.append)
    return "".join(parts)


def _write_canonical(value: Any, write: Callable[[str], object], /) -> None:
    """Write canonical JSON with numeric array leaves replaced by their fingerprints.

    Tuples and lists encode as JSON arrays and mappings as key-sorted objects,
    exactly as ``canonical_json`` would encode the normalized copy, without
    building that copy.
    """
    kind = type(value)
    if kind in _JSON_SCALAR_TYPES:
        write(_ENCODER.encode(value))
    elif kind is tuple or kind is list:
        _write_sequence(value, write)
    elif kind is dict:
        _write_mapping(value, write)
    elif isinstance(value, (jax.Array, np.ndarray, np.generic)):
        write(_ENCODER.encode({"__array__": array_tree_fingerprint(np.asarray(value))}))
    elif isinstance(value, Mapping):
        _write_mapping(value, write)
    elif isinstance(value, (tuple, list)):
        _write_sequence(value, write)
    elif kind is LazyCanonicalSequence:
        write("[")
        for index, item in enumerate(value):
            if index:
                write(",")
            _write_canonical(item, write)
        write("]")
    else:
        write(_ENCODER.encode(value))


def _plain_json_size(value: tuple[Any, ...] | list[Any], budget: int, /) -> int:
    """Remaining ``budget`` after the items of a nest of tuples/lists of JSON
    scalars, or -1 when the nest holds anything else or exceeds the budget."""
    for item in value:
        budget -= 1
        if budget < 0:
            return -1
        kind = type(item)
        if kind in _JSON_SCALAR_TYPES:
            continue
        if kind is not tuple and kind is not list:
            return -1
        budget = _plain_json_size(item, budget)
        if budget < 0:
            return -1
    return budget


def _write_sequence(
    value: tuple[Any, ...] | list[Any], write: Callable[[str], object], /
) -> None:
    if _JSON_SCALAR_TYPES.issuperset(map(type, value)):
        if len(value) <= _SEQUENCE_CHUNK:
            write(_ENCODER.encode(value))
            return
        write("[")
        for start in range(0, len(value), _SEQUENCE_CHUNK):
            if start:
                write(",")
            write(_ENCODER.encode(value[start : start + _SEQUENCE_CHUNK])[1:-1])
        write("]")
        return
    if _plain_json_size(value, _SEQUENCE_CHUNK) >= 0:
        # A bounded nest of tuples/lists of JSON scalars encodes as JSON
        # arrays exactly as the item-by-item walk below writes it.
        write(_ENCODER.encode(value))
        return
    write("[")
    for index, item in enumerate(value):
        if index:
            write(",")
        _write_canonical(item, write)
    write("]")


def _write_mapping(value: Mapping[Any, Any], write: Callable[[str], object], /) -> None:
    write("{")
    # The JSON encoder sorts (key, value) items; mapping keys are unique, so the
    # order is decided by the keys alone.
    for index, (key, item) in enumerate(sorted(value.items(), key=_ITEM_KEY)):
        if index:
            write(",")
        write(_json_key(key))
        write(":")
        _write_canonical(item, write)
    write("}")


def _json_key(key: Any, /) -> str:
    # Same coercion as the JSON encoder: strings verbatim; floats, integers,
    # booleans and None through their JSON scalar text.
    if isinstance(key, str):
        return _ENCODER.encode(key)
    if key is None or isinstance(key, (int, float)):
        return _ENCODER.encode(_ENCODER.encode(key))
    raise TypeError(
        f"keys must be str, int, float, bool or None, not {type(key).__name__}"
    )


def canonical_mapping(value: Mapping[str, Any], /) -> dict[str, Any]:
    """Return an independent JSON-normalized mapping or reject unsupported values."""
    normalized = json.loads(canonical_json(dict(value)))
    if not isinstance(normalized, dict):
        raise TypeError("Canonical mapping input must serialize to a JSON object.")
    return normalized


def _array_records(tree: PyTree[Any], /) -> Iterator[tuple[str, np.ndarray]]:
    """Traverse keyed leaves without retaining the complete flattened path bank."""
    def walk(node: Any, path: tuple[Any, ...], /) -> Iterator[tuple[str, np.ndarray]]:
        if jax.tree_util.is_tree_node(type(node)):
            children, _ = jax.tree_util.flatten_one_level_with_keys(node)
            for key, child in children:
                yield from walk(child, (*path, key))
            return
        try:
            value = np.asarray(node)
        except (TypeError, ValueError):
            return
        if value.dtype.hasobject:
            if isinstance(node, np.ndarray):
                raise TypeError(
                    f"Array leaf {jax.tree_util.keystr(path) or '<root>'} has object dtype and cannot be fingerprinted."
                )
            return
        yield jax.tree_util.keystr(path) or "<root>", value

    yield from walk(tree, ())


__all__ = [
    "LazyCanonicalSequence",
    "array_tree_fingerprint",
    "array_bytes_digest",
    "array_tree_signature",
    "canonical_fingerprint",
    "canonical_json",
    "canonical_array_hex_fingerprint",
    "canonical_array_list_fingerprint",
    "canonical_mapping",
    "iter_array_bytes",
]
