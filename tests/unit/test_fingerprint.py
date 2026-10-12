from __future__ import annotations

import hashlib
import json
from typing import Any

import jax
import numpy as np
import pytest

from benchmarks._memory import host_allocation_peak
from phydrax._fingerprint import (
    array_bytes_digest,
    array_tree_fingerprint,
    canonical_array_hex_fingerprint,
    iter_array_bytes,
)


def _legacy(tree: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    signature = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        value = np.asarray(leaf)
        if value.dtype.hasobject:
            continue
        name = jax.tree_util.keystr(path) or "<root>"
        contiguous = np.ascontiguousarray(value)
        metadata = json.dumps(
            {
                "path": name,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        payload = contiguous.tobytes(order="C")
        for chunk in (metadata, payload):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)
        signature.append(
            {"path": name, "shape": list(value.shape), "dtype": value.dtype.str}
        )
    return {"signature": signature, "sha256": digest.hexdigest()}


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.int8,
        np.int64,
        np.complex64,
        np.complex128,
        np.bool_,
        np.dtype(">f8"),
        np.dtype([("x", "<i4"), ("y", ">f8")]),
    ],
)
@pytest.mark.parametrize(
    "layout", ["scalar", "contiguous", "strided", "reversed", "empty"]
)
def test_array_fingerprint_matches_legacy(dtype: np.dtype, layout: str) -> None:
    value = np.zeros((12, 10), dtype=dtype)
    if value.dtype.fields is None:
        value[...] = np.arange(120).reshape(12, 10) % 7
    else:
        value["x"] = np.arange(120).reshape(12, 10)
        value["y"] = 0.25
    match layout:
        case "scalar":
            value = np.asarray(value[0, 0])
        case "strided":
            value = value[::2, ::3].T
        case "reversed":
            value = value[::-1, ::-1]
        case "empty":
            value = value[:0]
    assert array_tree_fingerprint({"array": value}) == _legacy({"array": value})


def test_array_fingerprint_preserves_nan_bits() -> None:
    values = np.asarray(
        [0x7FF8000000000001, 0x7FF8000000000002, 0x8000000000000000], dtype=np.uint64
    ).view(np.float64)
    assert array_tree_fingerprint(values) == _legacy(values)
    assert array_tree_fingerprint(values[:1]) != array_tree_fingerprint(values[1:2])


@pytest.mark.parametrize("strided", [False, True])
def test_array_fingerprint_has_bounded_scratch(strided: bool) -> None:
    small = np.arange(131072, dtype=np.float64)
    large = np.arange(524288, dtype=np.float64)
    if strided:
        small = small.reshape(-1, 2).T
        large = large.reshape(-1, 2).T
    array_tree_fingerprint(small)
    _, small_peak = host_allocation_peak(lambda: array_tree_fingerprint(small))
    _, large_peak = host_allocation_peak(lambda: array_tree_fingerprint(large))
    assert large_peak <= max(16384, small_peak * 1.5)


def test_array_fingerprint_rejects_object_arrays() -> None:
    with pytest.raises(TypeError, match="object dtype"):
        array_tree_fingerprint(np.asarray(["value"], dtype=np.object_))


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]", "U3", "S3", "V0"])
def test_non_numeric_array_fingerprint_matches_legacy(dtype: str) -> None:
    value = np.zeros((3, 4), dtype=np.dtype(dtype))
    assert array_tree_fingerprint(value.T) == _legacy(value.T)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.bool_, ">f8"])
@pytest.mark.parametrize("empty", [False, True])
def test_bounded_array_byte_and_hex_hashes(dtype: np.dtype, empty: bool) -> None:
    value = np.arange(24).reshape(4, 6).astype(dtype)[::-1, ::2]
    if empty:
        value = value[:0]
    payload = value.tobytes(order="C")
    assert array_bytes_digest(value) == hashlib.sha256(payload).hexdigest()
    canonical_hex = json.dumps(payload.hex(), separators=(",", ":")).encode("ascii")
    assert (
        canonical_array_hex_fingerprint(value)
        == hashlib.sha256(canonical_hex).hexdigest()
    )
    assert b"".join(bytes(chunk) for chunk in iter_array_bytes(value)) == payload
    assert all(len(chunk) <= 65536 for chunk in iter_array_bytes(value))
