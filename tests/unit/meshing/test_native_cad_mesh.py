# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Embedded native CAD fixture against the Python geometry bundle."""

from __future__ import annotations

import ctypes
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    _CAD_FACE_FAILURE_DTYPE,
    cad_mesh,
    cad_mesh_solid,
    CadFidelityChords,
    CadFidelityMesh,
    CadModel,
    CadSurfaceResult,
    load_meshcore,
    MeshcoreStatus,
    verify_cad_fidelity,
)
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.meshing.providers._native_cad_bundle import (
    cad_geometry_bundle,
    coefficient_bounds,
)
from tests.unit.meshing.test_native_cad_bundle import _quadric_model
from tests.unit.meshing.test_native_surface_verifiers import (
    _ChartFailure,
    _surface_verdict,
)


@pytest.fixture(scope="module")
def domain() -> MeshingDomain:
    return MeshingDomain.from_brep(_quadric_model())


def _assert_surface_verifiers(
    domain: MeshingDomain, model: CadModel, result: CadSurfaceResult
) -> None:
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
    function = load_meshcore()["phx_mc_verify_chart_partition"]
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
        ctypes.addressof(evidence),
        ctypes.addressof(count),
    )
    assert status == 0
    assert count.value == 0
    _assert_cad_fidelity(domain, model, result)


def _assert_cad_fidelity(
    domain: MeshingDomain, model: CadModel, result: CadSurfaceResult
) -> None:
    """Run V3 on the exported face stage and compare with dense source samples.

    The reference is the Python source surface at sampled chart points against
    the linear interpolant of the physical corners, which is the correspondence
    distance V3 bounds. Its maximum fixes both tolerances: away from poles the
    adaptive bounds certify at 3x it, while the first-order pole sliver cap
    (sup|S_r| times the radial extent, never subdivided) measured 6.3x-8.1x it
    on these models, so the whole mesh is certified at 10x it.
    """
    chart = result.chart
    steps = 12
    weights = np.asarray(
        [
            (1 - (i + j) / steps, i / steps, j / steps)
            for i in range(steps + 1)
            for j in range(steps + 1 - i)
        ],
        dtype=np.float64,
    )
    measured = np.empty((len(chart.triangles),), dtype=np.float64)
    for patch in np.unique(chart.patches):
        rows = np.flatnonzero(chart.patches == patch)
        surface = domain.patches[int(patch)].surface
        uv = chart.uv[rows]
        corners = result.points[chart.triangles[rows]]
        # Pins the chart patch numbering and UV convention to the source.
        np.testing.assert_allclose(
            np.asarray(
                surface.evaluate(jnp.asarray(uv.reshape((-1, 2)), dtype=jnp.float64)),
                dtype=np.float64,
            ),
            corners.reshape((-1, 3)),
            rtol=0,
            atol=1e-12,
        )
        samples = np.einsum("kv,rvd->rkd", weights, uv)
        exact = np.asarray(
            surface.evaluate(jnp.asarray(samples.reshape((-1, 2)), dtype=jnp.float64)),
            dtype=np.float64,
        ).reshape((len(rows), len(weights), 3))
        linear = np.einsum("kv,rvd->rkd", weights, corners)
        measured[rows] = np.max(np.linalg.norm(exact - linear, axis=2), axis=1)
    lower, upper = coefficient_bounds(cad_geometry_bundle(domain), domain)
    mesh = CadFidelityMesh(
        result.points,
        chart.triangles,
        chart.patches,
        chart.uv,
        result.triangles.astype(np.int64),
        chart.chart_to_mesh,
        chart.edge_chord,
    )
    chords = CadFidelityChords(
        chart.use_patch,
        chart.use_range,
        len(model.loop_surfaces),
        chart.chord_use,
        chart.chord_loop,
        chart.chord_uv,
        chart.chord_range,
    )
    slivers = chart.chart_to_mesh < 0
    pole_free = verify_cad_fidelity(
        model, lower, upper, mesh, chords, 3 * float(np.max(measured)), 8
    )
    assert pole_free.findings.size == 0
    assert np.all(pole_free.item_status[~slivers] == MeshcoreStatus.OK)
    tolerance = 10 * float(np.max(measured))
    verdict = verify_cad_fidelity(model, lower, upper, mesh, chords, tolerance, 8)
    assert verdict.status is MeshcoreStatus.OK
    assert verdict.findings.size == 0
    assert np.all(verdict.item_status == MeshcoreStatus.OK)
    assert np.all(verdict.reasons == 0)
    assert verdict.maximum_upper <= tolerance
    assert np.all(verdict.upper >= measured)
    # An interior vertex pushed off the surface by twice the tolerance along
    # its area-weighted normal refuses exactly the chart rows through it.
    vertex = int(np.flatnonzero(result.vertex_dimensions == 2)[0])
    incident = result.points[result.triangles[np.any(result.triangles == vertex, axis=1)]]
    normal = np.sum(
        np.cross(incident[:, 1] - incident[:, 0], incident[:, 2] - incident[:, 0]), axis=0
    )
    points = result.points.copy()
    points[vertex] += 2 * tolerance * normal / np.linalg.norm(normal)
    moved = verify_cad_fidelity(
        model, lower, upper, mesh._replace(points=points), chords, tolerance, 8
    )
    touched = np.any(chart.triangles == vertex, axis=1)
    assert np.any(touched)
    assert moved.status is MeshcoreStatus.UNRESOLVED
    assert np.all(moved.item_status[touched] == MeshcoreStatus.UNRESOLVED)
    assert np.all(moved.item_status[~touched] == MeshcoreStatus.OK)
    np.testing.assert_array_equal(moved.upper[~touched], verdict.upper[~touched])


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
    _assert_surface_verifiers(domain, model, result)
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
    sphere_domain = MeshingDomain.from_brep(sphere)
    model = cad_geometry_bundle(sphere_domain).model
    result = cad_mesh(
        model,
        np.full((model.surface_nodes.size,), 0.2, dtype=np.float64),
        np.full((model.surface_nodes.size,), 1e6, dtype=np.float64),
        np.full((model.curve_nodes.size,), np.inf, dtype=np.float64),
    )
    assert result.status is MeshcoreStatus.OK
    _assert_surface_verifiers(sphere_domain, model, result)
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


def _washer() -> phx.geometry.brep.BRepModel:
    brep = phx.geometry.brep
    return brep.brep_revolution(
        brep.PlanarProfile(
            brep.ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            brep.ProfileLoop.polygon([(0.5, 0.0), (1.0, 0.0), (1.0, 0.3), (0.5, 0.3)]),
        ),
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )


@pytest.mark.parametrize(
    ("name", "size"),
    (
        ("cone_apex", 0.15),
        # At 0.2 a chart triangle's frontal point and circumcenter both lie
        # within half a metric unit of a vertex; the longest-edge midpoint
        # fallback resolves it.
        ("cone_apex", 0.2),
        ("capped_cylinder", 0.2),
        ("washer", 0.1),
    ),
)
def test_revolved_surfaces_pass_native_verifiers(name: str, size: float) -> None:
    si = phx.SpatialCoordinateContract.si()
    match name:
        case "cone_apex":
            source = phx.geometry.brep.brep_cone(1.0, 0.0, 1.0, coordinate_contract=si)
        case "capped_cylinder":
            source = phx.geometry.brep.brep_cylinder(1.0, 1.0, coordinate_contract=si)
        case _:
            source = _washer()
    domain = MeshingDomain.from_brep(source)
    model = cad_geometry_bundle(domain).model
    result = cad_mesh(
        model,
        np.full((model.surface_nodes.size,), size, dtype=np.float64),
        np.full((model.surface_nodes.size,), 1e6, dtype=np.float64),
        np.full((model.curve_nodes.size,), np.inf, dtype=np.float64),
    )
    assert result.status is MeshcoreStatus.OK
    _assert_surface_verifiers(domain, model, result)


def test_quadric_refinement_failures_reach_python(domain: MeshingDomain) -> None:
    model = cad_geometry_bundle(domain).model
    # A 1e-5 tolerance on curve 0 alone samples it far finer than the curves
    # meeting it at its corners; the fixed short/long chord corners leave
    # refusal records on the faces through it.
    curve_tolerances = np.full((model.curve_nodes.size,), np.inf, dtype=np.float64)
    curve_tolerances[0] = 1e-5
    result = cad_mesh(
        model,
        np.full((model.surface_nodes.size,), 0.12, dtype=np.float64),
        np.full((model.surface_nodes.size,), 0.4, dtype=np.float64),
        curve_tolerances,
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


@pytest.mark.parametrize("order", [1, 2])
def test_quadric_solid(domain: MeshingDomain, order: int) -> None:
    model = cad_geometry_bundle(domain).model
    arguments = (
        model,
        np.full((model.surface_nodes.size,), 0.3, dtype=np.float64),
        np.full((model.surface_nodes.size,), 0.4, dtype=np.float64),
        np.full((model.curve_nodes.size,), np.inf, dtype=np.float64),
    )
    result = cad_mesh_solid(*arguments, geometry_order=order)
    assert result.status is MeshcoreStatus.OK
    surface, volume = result.surface, result.volume
    stats = volume.stats[0]
    assert stats["recovery_status"] == 0 and stats["boundary_splits"] == 0
    assert stats["unmet_cells"] == 0
    assert all(stats[f"{s}_ns"] > 0 for s in ("curve", "face", "volume"))
    count = surface.points.shape[0]
    np.testing.assert_array_equal(volume.points[:count], surface.points)
    cells = volume.tetrahedra
    corners = volume.points[cells]
    determinant = np.linalg.det(corners[:, 1:] - corners[:, :1])
    assert np.all(determinant > 0.0)
    assert np.all(volume.tetrahedron_regions == 0)
    # Each surface triangle is the face its boundary row names.
    cell, local = volume.boundary_faces[:, 0], volume.boundary_faces[:, 1]
    faces = np.stack([cells[cell, (local + k) % 4] for k in (1, 2, 3)], axis=1)
    np.testing.assert_array_equal(
        np.sort(faces, axis=1), np.sort(surface.triangles, axis=1)
    )
    if order == 1:
        assert volume.edges.shape[0] == 0 and volume.cell_edges.shape[0] == 0
        assert stats["quadratic_ns"] == 0
    else:
        assert np.all(volume.cell_decisions == 0)
        curved = volume.edge_kinds != 0
        nodes = volume.nodes[curved]
        # On the z cylinder (r 1), the x cylinder (r 0.6) or a planar cap.
        residual = np.min(
            np.stack(
                [
                    np.abs(np.hypot(nodes[:, 0], nodes[:, 1]) - 1.0),
                    np.abs(np.hypot(nodes[:, 1], nodes[:, 2]) - 0.6),
                    np.abs(np.abs(nodes[:, 2]) - 1.2),
                    np.abs(np.abs(nodes[:, 0]) - 1.5),
                ]
            ),
            axis=0,
        )
        assert curved.any() and np.max(residual) < 1e-12
    again = cad_mesh_solid(*arguments, geometry_order=order)
    np.testing.assert_array_equal(again.volume.points, volume.points)
    np.testing.assert_array_equal(again.volume.tetrahedra, cells)
    np.testing.assert_array_equal(again.volume.nodes, volume.nodes)
