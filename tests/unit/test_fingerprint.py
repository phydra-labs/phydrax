from __future__ import annotations

import collections
import enum
import hashlib
import json
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from benchmarks._memory import host_allocation_peak
from phydrax import _fingerprint as fingerprint_module
from phydrax._fingerprint import (
    array_bytes_digest,
    array_tree_fingerprint,
    canonical_array_hex_fingerprint,
    canonical_fingerprint,
    iter_array_bytes,
)
from phydrax.discretization import CellMesh
from phydrax.lifecycle._composition_rebind import Composition, CompositionEntry
from phydrax.meshing import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._model_codec import (
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
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


def _legacy_payload(value: Any) -> Any:
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        return {"__array__": array_tree_fingerprint(np.asarray(value))}
    if isinstance(value, Mapping):
        return {key: _legacy_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_legacy_payload(item) for item in value]
    return value


def _legacy_canonical_fingerprint(value: Any) -> str:
    """The copy-then-encode definition the streamed fingerprint must reproduce."""
    text = json.dumps(
        _legacy_payload(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _outcome(function: Callable[[Any], str], value: Any) -> str | type[Exception]:
    try:
        return function(value)
    except (TypeError, ValueError) as error:
        return type(error)


class _Label(str, enum.Enum):
    NAME = "na\u00efve"


class _Rank(enum.IntEnum):
    THREE = 3


class _Real(float):
    pass


class _Items(list[Any]):
    pass


_Pair = collections.namedtuple("_Pair", "first second")


_CANONICAL_CASES: tuple[Any, ...] = (
    1,
    1.5,
    "snow \u2603",
    None,
    True,
    (),
    [],
    {},
    (1, (2, 3.0), [None, False]),
    {"b": (1, 2), "a": {"z": np.arange(3), "y": np.float64(2.5), "x": np.int64(4)}},
    types.MappingProxyType({"k": (1, 2)}),
    {1: "a", 2: "b", 10: "c"},
    {1.5: 1, 2.0: 2},
    {True: 1, False: 2},
    {None: (None,)},
    {_Label.NAME: 1},
    {_Rank.THREE: (_Rank.THREE, _Label.NAME)},
    _Pair(1, (2, 3)),
    _Items([1, (2,)]),
    _Real(0.1),
    [_Real(0.1), 2],
    tuple(range(10_000)),
    tuple((index, (index, index + 1), str(index)) for index in range(9_000)),
    [0.1] * 9_001 + [np.float32(1.0)],
    jnp.ones(3),
    np.str_("q"),
    np.bool_(True),
    [np.str_("q"), "r"],
    ((1, 2), [(3, ("4",))]),
    [1e300, -0.0, 5e-324, 2**70, -(2**70)],
    '\x00\x1f"\\',
    {"nan": float("nan")},
    {(1, 2): 3},
    {"object": object()},
    {1: 1, "a": 2},
    {True: 1, None: 2},
)


@pytest.mark.parametrize("value", _CANONICAL_CASES)
def test_canonical_fingerprint_matches_copy_then_encode(value: Any) -> None:
    assert _outcome(canonical_fingerprint, value) == _outcome(
        _legacy_canonical_fingerprint, value
    )


def test_canonical_fingerprint_streams_nested_payloads_without_copies() -> None:
    def payload(cells: int) -> dict[str, Any]:
        exponents = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        return {
            "kind": "per-cell-charts",
            "cells": tuple(range(cells)),
            "charts": tuple(
                tuple(
                    (
                        tuple(
                            (exponent, f"{cell + axis}/{term + 2}")
                            for term, exponent in enumerate(exponents)
                        ),
                        (((0, 0, 0), "1"),),
                    )
                    for axis in range(3)
                )
                for cell in range(cells)
            ),
        }

    small, large = payload(2_000), payload(32_000)
    assert canonical_fingerprint(large) == _legacy_canonical_fingerprint(large)
    _, small_peak = host_allocation_peak(lambda: canonical_fingerprint(small))
    _, large_peak = host_allocation_peak(lambda: canonical_fingerprint(large))
    _, legacy_peak = host_allocation_peak(lambda: _legacy_canonical_fingerprint(large))
    assert large_peak <= max(1 << 20, 2 * small_peak)
    assert 20 * large_peak < legacy_peak


def _record_canonical_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[Any]:
    recorded: list[Any] = []
    depth = [0]
    write_canonical = fingerprint_module._write_canonical

    def recording(value: Any, write: Callable[[str], object], /) -> None:
        if not depth[0]:
            recorded.append(value)
        depth[0] += 1
        try:
            write_canonical(value, write)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(fingerprint_module, "_write_canonical", recording)
    return recorded


def test_canonical_fingerprints_of_meshes_certificates_witnesses_and_archives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded = _record_canonical_payloads(monkeypatch)
    points = np.stack(
        np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)), axis=-1
    ).reshape((-1, 2))
    cells = []
    for row in range(3):
        for column in range(3):
            first = 4 * row + column
            cells.extend(((first, first + 1, first + 5), (first, first + 5, first + 4)))
    mesh = CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    adapted = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(
                np.asarray((0, 5, 11), dtype=np.int64), np.asarray((), dtype=np.int64)
            ),
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION),
        )
    )
    target = adapted.target.mesh
    composition = Composition(
        tuple(
            CompositionEntry(
                value,
                entry_id=f"mesh/{index}",
                role="topology",
                owner_id="phydrax.discretization",
                structure_id=value.topology_id,
                revision_id=value.mesh_id,
                semantics_id="fingerprint-corpus-mesh",
            )
            for index, value in enumerate((mesh, target))
        ),
        boundary_id="fingerprint-corpus",
    )
    archive = write_model_artifact(tmp_path / "corpus.phx", composition, limits=None)
    restored = read_model_artifact(
        archive,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert restored.entries[1].value.mesh_id == target.mesh_id
    corpus = tuple(recorded)
    monkeypatch.undo()
    assert len(corpus) > 100
    for value in corpus:
        assert canonical_fingerprint(value) == _legacy_canonical_fingerprint(value)
