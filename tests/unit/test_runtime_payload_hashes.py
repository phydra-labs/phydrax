from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks._memory import host_allocation_peak
from phydrax._external_runtime import _array_record
from phydrax._fingerprint import canonical_array_list_fingerprint
from phydrax.atomistic.interchange._core import canonical_source_digest
from phydrax.interchange import _dafoam_worker


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _representative(dtype: str, layout: str) -> np.ndarray:
    value = np.arange(120).reshape(12, 10).astype(np.dtype(dtype))
    match layout:
        case "scalar":
            return np.asarray(value[0, 0])
        case "strided":
            return value[::2, ::3].T
        case "reversed":
            return value[::-1, ::-1]
        case "empty":
            return value[:0]
        case "empty_inner":
            return value[:, :0]
        case _:
            return value


@pytest.mark.parametrize("dtype", ["f4", "f8", "i8", "c16", "?", ">f8"])
@pytest.mark.parametrize(
    "layout", ["scalar", "contiguous", "strided", "reversed", "empty", "empty_inner"]
)
def test_runtime_array_record_has_independent_legacy_pin(dtype: str, layout: str) -> None:
    value = _representative(dtype, layout)
    expected = [
        list(value.shape),
        value.dtype.str,
        hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest(),
    ]
    assert _array_record(value) == expected
    if value.size:
        changed = value.copy()
        changed.flat[0] = not changed.flat[0]
        assert _array_record(changed) != expected


@pytest.mark.parametrize("dtype", ["f4", "f8", "i8", "?", ">f8", "U3", "M8[ns]"])
@pytest.mark.parametrize(
    "layout", ["scalar", "contiguous", "strided", "reversed", "empty", "empty_inner"]
)
def test_array_list_fingerprint_and_source_digest_have_independent_pins(
    dtype: str, layout: str
) -> None:
    value = _representative(dtype, layout)
    legacy_array = {
        "shape": list(value.shape),
        "dtype": value.dtype.name,
        "data": value.tolist(),
    }
    expected = _json_digest(
        {
            "kind": "atomistic-interchange-source",
            "content": {"nested": [legacy_array], "name": "molecule\u00e9"},
        }
    )
    assert (
        canonical_source_digest({"name": "molecule\u00e9", "nested": [value]}) == expected
    )
    assert canonical_array_list_fingerprint({"data": value}) == _json_digest(
        {"data": value.tolist()}
    )


def test_array_list_fingerprint_crosses_chunk_boundaries_and_empty_axes() -> None:
    value = np.linspace(-2, 3, 4097, dtype=np.float64)[::-1]
    assert canonical_array_list_fingerprint(value) == _json_digest(value.tolist())
    empty = np.empty((2, 0, 3), dtype=np.float64)
    assert canonical_array_list_fingerprint(empty) == _json_digest([[], []])
    source = {"array": value, "scalar": np.float64(-0.0), "tuple": (True, "\u03bb")}
    expected = _json_digest(
        {
            "kind": "atomistic-interchange-source",
            "content": {
                "array": {"shape": [4097], "dtype": "float64", "data": value.tolist()},
                "scalar": -0.0,
                "tuple": [True, "\u03bb"],
            },
        }
    )
    assert canonical_source_digest(source) == expected
    source["array"] = value.copy()
    source["array"][0] += 1
    assert canonical_source_digest(source) != expected


def test_source_digest_preserves_jax_array_and_rejection_contracts() -> None:
    value = jnp.asarray([1.0, -0.0], dtype=jnp.float32)
    host = np.asarray(value)
    expected = _json_digest(
        {
            "kind": "atomistic-interchange-source",
            "content": {
                "array": {"shape": [2], "dtype": "float32", "data": host.tolist()}
            },
        }
    )
    assert canonical_source_digest({"array": value}) == expected
    with pytest.raises(TypeError, match="complex"):
        canonical_source_digest({"array": np.asarray([1 + 2j], dtype=np.complex128)})
    for nonfinite in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="Out of range float"):
            canonical_source_digest({"array": np.asarray([nonfinite], dtype=np.float64)})


@pytest.mark.parametrize("dtype", ["f4", "f8", "i8", "?", ">f8"])
@pytest.mark.parametrize(
    "layout", ["scalar", "contiguous", "strided", "reversed", "empty"]
)
def test_dafoam_array_hash_preserves_float64_pin(dtype: str, layout: str) -> None:
    value = _representative(dtype, layout)
    expected = hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float64).tobytes()
    ).hexdigest()
    assert _dafoam_worker._array_hash(value) == expected
    if value.size:
        changed = value.copy()
        changed.flat[0] = not changed.flat[0]
        assert _dafoam_worker._array_hash(changed) != expected


def test_dafoam_hash_worker_imports_without_phydrax() -> None:
    script = (
        "import builtins, runpy, sys; "
        "original = builtins.__import__; "
        "builtins.__import__ = lambda name, *args, **kwargs: "
        "(_ for _ in ()).throw(ImportError(name)) if name.startswith('phydrax') "
        "else original(name, *args, **kwargs); "
        "worker = runpy.run_path(sys.argv[1]); "
        "import numpy as np; "
        "print(worker['_array_hash'](np.asarray([1., 2.], dtype=np.float64)))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, _dafoam_worker.__file__],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = hashlib.sha256(
        np.asarray([1.0, 2.0], dtype=np.float64).tobytes()
    ).hexdigest()
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("operation", [_array_record, canonical_source_digest])
def test_runtime_payload_hash_scratch_is_bounded(operation: Any, strided: bool) -> None:
    small = np.arange(32768, dtype=np.float64)
    large = np.arange(131072, dtype=np.float64)
    if strided:
        small = small.reshape(-1, 2).T
        large = large.reshape(-1, 2).T
    small_input = {"array": small} if operation is canonical_source_digest else small
    large_input = {"array": large} if operation is canonical_source_digest else large
    operation(small_input)
    _, small_peak = host_allocation_peak(lambda: operation(small_input))
    _, large_peak = host_allocation_peak(lambda: operation(large_input))
    assert large_peak <= max(131072, small_peak * 1.5)
