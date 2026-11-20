# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Embedded native CAD fixture against the Python geometry bundle."""

from __future__ import annotations

import ctypes
import os
import re
from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    _CAD_FACE_FAILURE_DTYPE,
    cad_mesh,
    CadModel,
    CadSurfaceResult,
    MeshcoreStatus,
)
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.meshing.providers._native_cad_bundle import cad_geometry_bundle
from tests.unit.meshing.test_native_cad_bundle import _quadric_model
from tests.unit.meshing.test_native_surface_verifiers import (
    _ChartFailure,
    _surface_verdict,
)


@pytest.fixture(scope="module")
def domain() -> MeshingDomain:
    return MeshingDomain.from_brep(_quadric_model())


def _assert_surface_verifiers(model: CadModel, result: CadSurfaceResult) -> None:
    chart = result.chart
    mesh_rows = chart.chart_to_mesh >= 0
    np.testing.assert_array_equal(
        chart.triangles[mesh_rows], result.triangles[chart.chart_to_mesh[mesh_rows]]
    )
    np.testing.assert_array_equal(
        np.sort(chart.chart_to_mesh[mesh_rows]),
        np.arange(len(result.triangles), dtype=np.int64),
    )
    assert np.count_nonzero(~mesh_rows) == len(result.sliver_uv)
    assert _surface_verdict(result.points, result.triangles.astype(np.int64))
    library = ctypes.CDLL(os.environ["PHYDRAX_MESHCORE_LIBRARY"])
    function = library.phx_mc_verify_chart_partition
    function.restype = ctypes.c_int32
    function.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
    ]
    evidence = (_ChartFailure * (3 * len(chart.triangles) + len(chart.chord_use) + 10))()
    count = ctypes.c_int64()
    status = function(
        model.surface_nodes.size,
        len(chart.triangles),
        chart.patches.ctypes.data,
        chart.uv.ctypes.data,
        len(chart.use_patch),
        chart.use_patch.ctypes.data,
        chart.use_range.ctypes.data,
        len(model.loop_surfaces),
        len(chart.chord_use),
        chart.chord_use.ctypes.data,
        chart.chord_loop.ctypes.data,
        chart.chord_uv.ctypes.data,
        chart.chord_range.ctypes.data,
        len(evidence),
        evidence,
        ctypes.byref(count),
    )
    assert status == 0
    assert count.value == 0


def test_face_failure_dtype_matches_native_layout() -> None:
    # These offsets and size are also static_asserted against the C struct.
    offsets = {
        "face": 0,
        "loop": 4,
        "use": 8,
        "axis": 12,
        "reason": 16,
        "turns": 24,
        "parameter": 32,
        "triangle": 40,
        "bad_size": 44,
        "bad_angle": 48,
        "refusal_kind": 52,
        "curve": 56,
        "curve_segment": 60,
        "candidate_uv": 64,
        "triangle_uv": 80,
        "segment_parameters": 128,
    }
    assert _CAD_FACE_FAILURE_DTYPE.itemsize == 144
    fields = _CAD_FACE_FAILURE_DTYPE.fields
    if fields is None:
        pytest.fail("face failure dtype has no fields")
    assert {name: field[1] for name, field in fields.items()} == offsets


def _fixture_arrays() -> dict[str, np.ndarray]:
    path = (
        Path(__file__).resolve().parents[3]
        / "native/meshcore/tests/cad_quadric_model.inc"
    )
    text = path.read_text(encoding="utf-8")
    pattern = (
        r"std::array<(int8_t|int32_t|int64_t|double), (\d+)> quadric_(\w+)\{\{(.*?)\}\};"
    )
    dtypes = {
        "int8_t": np.int8,
        "int32_t": np.int32,
        "int64_t": np.int64,
        "double": np.float64,
    }
    arrays: dict[str, np.ndarray] = {}
    for kind, count, name, body in re.findall(pattern, text, re.DOTALL):
        tokens = [value.strip() for value in body.split(",") if value.strip()]
        if kind == "double":
            values = [
                (-np.inf if value.startswith("-") else np.inf)
                if "infinity()" in value
                else float.fromhex(value)
                for value in tokens
            ]
        else:
            values = [int(value) for value in tokens]
        array = np.asarray(values, dtype=dtypes[kind])
        if array.size != int(count):
            raise ValueError(f"embedded {name} has the wrong literal count")
        arrays[name] = array
    return arrays


def test_quadric_embedded_bundle_matches_domain(domain: MeshingDomain) -> None:
    embedded = _fixture_arrays()
    model = cad_geometry_bundle(domain).model
    assert set(embedded) == set(model._fields)
    shapes = {
        "corner_points": (2, 3),
        "surface_periods": (3, 2),
        "surface_parameter_domains": (3, 2, 2),
        "curve_corners": (3, 2),
        "curve_ranges": (3, 2),
        "use_ranges": (6, 2),
    }
    for name, actual in zip(model._fields, model, strict=True):
        reference = embedded[name]
        assert actual.dtype == reference.dtype, name
        assert actual.shape == shapes.get(name, reference.shape), name
        current = np.asarray(actual).reshape(-1)
        if np.issubdtype(current.dtype, np.integer):
            np.testing.assert_array_equal(current, reference, err_msg=name)
            continue
        # Host Python/JAX transcendental math can differ by up to four ulps
        # across the supported macOS, Windows and Linux machines.
        a = current.view(np.uint64)
        b = reference.view(np.uint64)
        sign = np.uint64(1) << np.uint64(63)
        aa = np.where((a & sign) != 0, ~a, a | sign)
        bb = np.where((b & sign) != 0, ~b, b | sign)
        gap = np.where(aa >= bb, aa - bb, bb - aa)
        gap = np.where(current == reference, np.uint64(0), gap)
        differing = np.flatnonzero(gap)
        if differing.size:
            index = differing[0]
            print(f"{name}[{index}] differs by {gap[index]} ulps")
        failures = np.flatnonzero(gap > np.uint64(4))
        if failures.size:
            index = failures[0]
            pytest.fail(f"{name}[{index}] differs by {gap[index]} ulps (limit 4)")


def _closest_triangle_distance(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    answer = np.full((points.shape[0],), np.inf, dtype=np.float64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    normal2 = np.sum(normal * normal, axis=1)
    for i, point in enumerate(points):
        ap = point - a
        signed = np.sum(ap * normal, axis=1)
        projection = point - (signed / normal2)[:, None] * normal
        projected = projection - a
        d00 = np.sum(ab * ab, axis=1)
        d01 = np.sum(ab * ac, axis=1)
        d11 = np.sum(ac * ac, axis=1)
        d20 = np.sum(projected * ab, axis=1)
        d21 = np.sum(projected * ac, axis=1)
        determinant = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / determinant
        w = (d00 * d21 - d01 * d20) / determinant
        distances = np.where(
            (v >= 0) & (w >= 0) & (v + w <= 1), signed * signed / normal2, np.inf
        )
        for start, end in ((a, b), (b, c), (c, a)):
            direction = end - start
            length2 = np.sum(direction * direction, axis=1)
            fraction = np.clip(
                np.sum((point - start) * direction, axis=1) / length2, 0, 1
            )
            delta = point - (start + fraction[:, None] * direction)
            distances = np.minimum(distances, np.sum(delta * delta, axis=1))
        answer[i] = np.sqrt(np.min(distances))
    return answer


def test_quadric_surface_candidate(domain: MeshingDomain) -> None:
    model = cad_geometry_bundle(domain).model
    result = cad_mesh(
        model,
        np.full((model.surface_nodes.size,), 0.12, dtype=np.float64),
        np.full((model.surface_nodes.size,), 0.4, dtype=np.float64),
        np.full((model.curve_nodes.size,), np.inf, dtype=np.float64),
    )
    assert result.status is MeshcoreStatus.OK
    _assert_surface_verifiers(model, result)
    triangles = result.points[result.triangles]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    for first, second in ((b - a, c - a), (c - b, a - b), (a - c, b - c)):
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        assert np.min(np.arccos(np.clip(cosine, -1, 1))) >= np.deg2rad(25)
    centers = (a + b + c) / 3
    vertical = np.abs(np.hypot(centers[:, 0], centers[:, 1]) - 1)
    transverse = np.abs(np.hypot(centers[:, 1], centers[:, 2]) - 0.6)
    mesh_to_source = np.max(np.where(result.triangle_patches < 2, vertical, transverse))
    u = np.linspace(0, 2 * np.pi, 161, endpoint=False, dtype=np.float64)
    keep = np.abs(np.sin(u)) < 0.6
    u = u[keep]
    z_extent = np.sqrt(0.36 - np.sin(u) ** 2)
    fractions = np.linspace(-1, 1, 9, dtype=np.float64)
    vertical_points = np.stack(
        np.broadcast_arrays(
            np.cos(u)[:, None], np.sin(u)[:, None], z_extent[:, None] * fractions
        ),
        axis=-1,
    ).reshape((-1, 3))
    v = np.linspace(0, 2 * np.pi, 161, endpoint=False, dtype=np.float64)
    y, z = 0.6 * np.cos(v), 0.6 * np.sin(v)
    x_extent = np.sqrt(1 - y * y)
    transverse_points = np.stack(
        np.broadcast_arrays(x_extent[:, None] * fractions, y[:, None], z[:, None]),
        axis=-1,
    ).reshape((-1, 3))
    source_to_mesh = np.max(
        _closest_triangle_distance(
            np.concatenate((vertical_points, transverse_points)), triangles
        )
    )
    assert mesh_to_source < 0.2
    assert source_to_mesh < 0.2


def test_sphere_pole_sector_evidence_and_shared_samples() -> None:
    sphere = phx.geometry.brep.brep_sphere(
        1.0, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    model = cad_geometry_bundle(MeshingDomain.from_brep(sphere)).model
    result = cad_mesh(
        model,
        np.full((model.surface_nodes.size,), 0.2, dtype=np.float64),
        np.full((model.surface_nodes.size,), 1e6, dtype=np.float64),
        np.full((model.curve_nodes.size,), np.inf, dtype=np.float64),
    )
    assert result.status is MeshcoreStatus.OK
    _assert_surface_verifiers(model, result)
    assert result.pole_sectors.shape == (2, 4)
    np.testing.assert_array_equal(
        result.pole_sectors[:, 2], np.array([8, 8], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        result.pole_sectors[:, 3], np.array([0, 0], dtype=np.int64)
    )
    for curve in range(model.curve_nodes.size):
        used = result.curve_edges[result.curve_edge_curves == curve]
        assert len(used) == (
            result.curve_offsets[curve + 1] - result.curve_offsets[curve] - 1
        )
        expected = result.curve_parameters[
            result.curve_offsets[curve] : result.curve_offsets[curve + 1]
        ]
        vertex_rows = np.unique(used)
        interior = vertex_rows[result.vertex_dimensions[vertex_rows] == 1]
        assert np.all(result.vertex_entities[interior] == curve)
        native = model.curve_ranges[curve, 0] + expected[1:-1] * (
            model.curve_ranges[curve, 1] - model.curve_ranges[curve, 0]
        )
        np.testing.assert_array_equal(
            np.sort(result.vertex_parameters[interior, 0]), np.sort(native)
        )


def test_quadric_refinement_failures_reach_python(domain: MeshingDomain) -> None:
    model = cad_geometry_bundle(domain).model
    result = cad_mesh(
        model,
        np.full((model.surface_nodes.size,), 0.08, dtype=np.float64),
        np.full((model.surface_nodes.size,), 0.4, dtype=np.float64),
        np.full((model.curve_nodes.size,), np.inf, dtype=np.float64),
    )
    assert result.status is MeshcoreStatus.UNRESOLVED
    failures = result.face_failures
    assert failures.size == np.sum(result.face_stats["unresolved_triangles"])
    assert failures.size > 0
    assert np.all(failures["reason"] == 4)
    assert np.all((failures["bad_size"] != 0) | (failures["bad_angle"] != 0))
    assert np.all(failures["triangle"] >= 0)
    assert np.all(np.isfinite(failures["triangle_uv"]))
    assert np.any(failures["curve"] >= 0)
    blocked = failures[failures["curve"] >= 0]
    assert np.all(blocked["refusal_kind"] == 1)
    assert np.all(np.isfinite(blocked["candidate_uv"]))
    for failure in failures:
        if failure["curve"] < 0:
            continue
        curve = failure["curve"]
        segment = failure["curve_segment"]
        assert 0 <= curve < model.curve_nodes.size
        assert segment >= 0
        start = result.curve_offsets[curve] + segment
        assert start + 1 < result.curve_offsets[curve + 1]
        np.testing.assert_array_equal(
            np.sort(failure["segment_parameters"]),
            result.curve_parameters[start : start + 2],
        )
