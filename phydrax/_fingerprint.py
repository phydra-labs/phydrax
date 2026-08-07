#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
                    f"Array leaf {jax.tree_util.keystr(path) or '<root>'} "
                    "has object dtype and cannot be fingerprinted."
                )
            continue
        records.append((jax.tree_util.keystr(path) or "<root>", value))
    return records


__all__ = [
    "array_tree_fingerprint",
    "array_tree_signature",
    "canonical_fingerprint",
    "canonical_json",
    "canonical_mapping",
]
