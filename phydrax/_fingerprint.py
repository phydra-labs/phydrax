#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
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


__all__ = [
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
