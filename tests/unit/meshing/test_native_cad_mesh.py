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
from phydrax.meshing._contracts import VolumeMeshingSpec
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


def _cad_volume_plan(
    model: phx.geometry.brep.BRepModel,
    size: float,
    order: int,
    /,
    *,
    radius_edge: float = 2.0,
    dihedral: float = 15.5,
) -> phx.meshing.NativeMeshingPlan:
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing._scope import MeshingScope
    from phydrax.meshing._sizing import UniformSizeControl
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_options import NativeMeshingOptions

    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3, 3, CellFamilyPolicy(required=("tetrahedron",)), geometry_order=order
        ),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, size),),
        maximum_deviation=0.05,
    )
    return phx.meshing.NativeMeshingProvider(
        NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=radius_edge, minimum_dihedral_degrees=dihedral
            ),
        )
    ).plan(model, specification)


def _sphere_model() -> phx.geometry.brep.BRepModel:
    return phx.geometry.brep.brep_sphere(
        0.5,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        source_id="native-cad-route-sphere",
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    ("name", "size", "radius"),
    [("quadric", 0.3, None), ("sphere", 0.2, 0.5)],
)
def test_unproved_cad_solid_takes_the_native_route(
    name: str, size: float, radius: float | None, order: int
) -> None:
    from phydrax.meshing import CellMeshingCandidate

    model = _quadric_model() if name == "quadric" else _sphere_model()
    candidate = _cad_volume_plan(model, size, order).execute(prove=False)
    assert isinstance(candidate, CellMeshingCandidate)
    assert candidate.route == "cad_tetrahedral"
    mesh = candidate.mesh
    assert mesh.topological_dimension == 3 and len(mesh.blocks) == 1
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners = points[cells]
    assert np.all(np.linalg.det(corners[:, 1:] - corners[:, :1]) > 0.0)
    # The boundary is closed: every boundary face occurs once.
    faces = np.sort(
        np.concatenate(
            [
                cells[:, [1, 2, 3]],
                cells[:, [0, 2, 3]],
                cells[:, [0, 1, 3]],
                cells[:, [0, 1, 2]],
            ]
        ),
        axis=1,
    )
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    assert set(counts.tolist()) == {1, 2}
    boundary = np.unique(unique[counts == 1])
    if radius is not None:
        np.testing.assert_allclose(
            np.linalg.norm(points[boundary], axis=1), radius, atol=1e-12
        )
    # Volume of the native mesh is the solid's within its deviation.
    volume = float(np.sum(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0)
    expected = 4.0 / 3.0 * np.pi * radius**3 if radius is not None else None
    if expected is not None:
        assert 0.9 * expected < volume < expected
    geometry = candidate.geometry
    assert geometry.elements[0].local_dof_count == (4 if order == 1 else 10)
    report = candidate._construction.compliance
    assert "maximum_deviation" not in dict(report.achieved)
    # Size compliance judges the boundary circumradius the chart kernel
    # bounds, R <= h / sqrt(2): the native mesh at h passes at zero tolerance,
    # and the same mesh judged at h / 2 (a mesh ignoring the size, built for
    # twice the request) fails.
    from phydrax.meshing._sizing import (
        CAD_CHART_SURFACE_SIZE,
        SizeCompliancePolicy,
        triangle_circumradius_measures,
    )

    assert not [issue for issue in report.issues if issue.startswith("size:")]
    policy = SizeCompliancePolicy()
    measures = triangle_circumradius_measures(points, unique[counts == 1])
    for statistic in ("p50", "p95"):
        assert not CAD_CHART_SURFACE_SIZE.exceeds(measures, statistic, size, policy)
        assert CAD_CHART_SURFACE_SIZE.exceeds(measures, statistic, 0.5 * size, policy)


def _right_triangles(legs: list[tuple[float, float]], /) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            point
            for row, (x, y) in enumerate(legs)
            for point in ((0.0, 0.0, row), (x, 0.0, row), (0.0, y, row))
        ],
        dtype=np.float64,
    )
    return points, np.arange(points.shape[0], dtype=np.int64).reshape((-1, 3))


def test_size_bound_ties_are_decided_exactly() -> None:
    # A unit box face meshed at h = 0.5 has right triangles with legs h:
    # R = h / sqrt(2) exactly, which the float statistic rounds above the
    # float bound. Exactly R^2 = h^2 / 2 = t^2 passes; a leg one ulp longer
    # fails, one ulp shorter passes, at either statistic.
    from phydrax.meshing._sizing import (
        CAD_CHART_SURFACE_SIZE,
        edge_length_measures,
        S3_BOUNDARY_EDGE_SIZE,
        SizeCompliancePolicy,
        triangle_circumradius_measures,
    )

    exact = SizeCompliancePolicy()
    up, down = np.nextafter(0.5, 1.0), np.nextafter(0.5, 0.0)
    for statistic in ("p50", "p95"):
        for legs, exceeds in (
            ([(0.5, 0.5)] * 4, False),
            ([(0.5, up)] * 4, True),
            ([(0.5, down)] * 4, False),
            # One row above t: the p50 interpolates between t and above it.
            ([(0.5, 0.5), (0.5, up)], True),
            ([(0.5, down), (0.5, 0.5)], False),
            # A degenerate row (R infinite) is above t.
            ([(0.5, 0.5), (0.5, 0.0)], True),
        ):
            points, triangles = _right_triangles(legs)
            measures = triangle_circumradius_measures(points, triangles)
            assert (
                CAD_CHART_SURFACE_SIZE.exceeds(measures, statistic, 0.5, exact) is exceeds
            ), (legs, statistic)
    # With an absolute tolerance d, t = h / sqrt(2) + d is irrational: the tie
    # row stays below it, and a p50 between the tie row and R = sqrt(5) / 4
    # (mean 0.4564) is decided exactly against t = 0.4786 and t = 0.4036.
    points, triangles = _right_triangles([(0.5, 0.5)] * 3)
    measures = triangle_circumradius_measures(points, triangles)
    tolerant = SizeCompliancePolicy(absolute_tolerance=0.125)
    assert not CAD_CHART_SURFACE_SIZE.exceeds(measures, "p50", 0.5, tolerant)
    points, triangles = _right_triangles([(0.5, 0.5), (0.5, 1.0)])
    measures = triangle_circumradius_measures(points, triangles)
    assert not CAD_CHART_SURFACE_SIZE.exceeds(measures, "p50", 0.5, tolerant)
    tight = SizeCompliancePolicy(absolute_tolerance=0.05)
    assert CAD_CHART_SURFACE_SIZE.exceeds(measures, "p50", 0.5, tight)
    # Edge rows compare exact squared lengths with (h (1 + rel) + abs)^2.
    points = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, up, 0.0]])
    tie = edge_length_measures(points, np.asarray([[0, 1]]))
    above = edge_length_measures(points, np.asarray([[0, 2]]))
    assert not S3_BOUNDARY_EDGE_SIZE.exceeds(tie, "p95", 0.5, exact)
    assert S3_BOUNDARY_EDGE_SIZE.exceeds(above, "p95", 0.5, exact)


def test_size_bound_decision_matches_the_float_statistic_away_from_ties() -> None:
    from phydrax.meshing._sizing import (
        CAD_CHART_SURFACE_SIZE,
        size_statistic,
        SizeCompliancePolicy,
        triangle_circumradii,
        triangle_circumradius_measures,
    )

    generator = np.random.default_rng(7)
    points = generator.random((300, 3))
    triangles = generator.integers(0, 300, (100, 3))
    triangles = triangles[
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    ]
    policy = SizeCompliancePolicy(relative_tolerance=0.01)
    radii = triangle_circumradii(points, triangles)
    measures = triangle_circumradius_measures(points, triangles)
    for statistic in ("p50", "p95"):
        value = size_statistic(radii, statistic)
        bound = np.sqrt(0.5) * 1.01
        for size in (0.5 * value / bound, 2.0 * value / bound, value / bound * 1.001):
            assert CAD_CHART_SURFACE_SIZE.exceeds(measures, statistic, size, policy) is (
                bool(value > bound * size)
            )


def test_triangle_circumradii_are_exact_and_refuse_degenerate_rows() -> None:
    from phydrax.meshing._sizing import triangle_circumradii

    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    radii = triangle_circumradii(points, np.asarray([[0, 1, 2], [0, 1, 3], [0, 1, 4]]))
    np.testing.assert_allclose(radii[:2], [1.0 / np.sqrt(3.0), np.sqrt(2.0) / 2.0])
    assert np.isinf(radii[2])


def test_proved_cad_solid_keeps_the_python_route() -> None:
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
    from phydrax.meshing._result import CellMeshingResult

    result = _cad_volume_plan(
        _sphere_model(), 0.3, 1, radius_edge=8.0, dihedral=0.0
    ).execute()
    assert isinstance(result, CellMeshingResult)
    assert result.certification is not None and result.certification.passed
    witness = result.certification.inputs.fidelity_witness
    # The Python construction's CAD ancestry, absent from the native route.
    assert isinstance(witness, SourceBoundaryChartWitness)
    assert witness.cad_plc is not None
    assert result.mesh.blocks[0].name != "native-cad-tetrahedra"


def test_proved_quadratic_cad_solid_is_refused() -> None:
    from phydrax.meshing._contracts import MeshingFailure

    plan = _cad_volume_plan(_sphere_model(), 0.3, 2)
    with pytest.raises(MeshingFailure, match="geometry order 2 is constructed unproved"):
        plan.execute()


@pytest.mark.parametrize("order", [1, 2])
def test_native_cad_candidate_prove_is_refused(order: int) -> None:
    # Until the native verifiers are accepted, a native candidate has no
    # witness: prove() refuses it rather than certify it or rebuild the
    # Python mesh.
    from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
    from phydrax.meshing.providers._native_cad_mesh import (
        NATIVE_CAD_SOLID_UNCERTIFIABLE,
    )

    candidate = _cad_volume_plan(_sphere_model(), 0.3, order).execute(prove=False)
    with pytest.raises(MeshingFailure) as failure:
        candidate.prove()
    assert failure.value.category is MeshingFailureCategory.UNSUPPORTED_COMBINATION
    assert str(failure.value) == NATIVE_CAD_SOLID_UNCERTIFIABLE
    assert "V3 CAD fidelity" in NATIVE_CAD_SOLID_UNCERTIFIABLE


def test_invalid_trim_cad_solid_is_refused_at_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A p-curve without a chart enclosure is invalid source for every proof
    # level: planning refuses it before any construction.
    from phydrax.geometry._meshing_domain import SourcePreparation
    from phydrax.meshing._contracts import MeshingFailure

    def no_enclosure(*_: object) -> np.ndarray:
        raise ValueError("p-curve enclosure diverges")

    monkeypatch.setattr(SourcePreparation, "curve_box", no_enclosure)
    with pytest.raises(MeshingFailure, match="requires a continuous p-curve enclosure"):
        _cad_volume_plan(_sphere_model(), 0.3, 1)


def test_incomplete_trim_cover_is_refused_at_proof_not_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Trim cover coverage is proof preparation: the plan and the unproved
    # native candidate do not consume it; the proved execution refuses.
    from types import SimpleNamespace

    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.meshing import CellMeshingCandidate
    from phydrax.meshing._contracts import MeshingFailure

    monkeypatch.setattr(
        MeshingDomainBoundarySource,
        "boundary_chart_cover",
        lambda *_, **__: SimpleNamespace(complete=False),
    )
    plan = _cad_volume_plan(_sphere_model(), 0.3, 1, radius_edge=8.0, dihedral=0.0)
    assert isinstance(plan.execute(prove=False), CellMeshingCandidate)
    with pytest.raises(
        MeshingFailure,
        match="does not provide complete selected-patch coverage",
    ):
        plan.execute()


def test_cad_volume_proof_preparation_runs_once_on_first_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing.providers._native_surface as native_surface

    calls: list[native_surface.NativeSurfaceRequest] = []
    prepare = native_surface.prepare_native_surface_proof

    def counted(
        request: native_surface.NativeSurfaceRequest,
    ) -> native_surface._PreparedNativeSurface:
        calls.append(request)
        return prepare(request)

    monkeypatch.setattr(native_surface, "prepare_native_surface_proof", counted)
    plan = _cad_volume_plan(_sphere_model(), 0.3, 1, radius_edge=8.0, dihedral=0.0)
    plan.execute(prove=False)
    assert not calls
    first = plan.execute()
    second = plan.execute()
    assert len(calls) == 1
    assert first.certification is not None and first.certification.passed
    assert second.mesh.mesh_id == first.mesh.mesh_id


def test_unproved_cad_solid_with_region_control_honours_it() -> None:
    # The native candidate has no organization records, so a region control
    # keeps the existing construction, whose candidate carries the control.
    from phydrax.meshing import CellMeshingCandidate
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing._controls import RegionControl, RegionSeed
    from phydrax.meshing._organization import RegionRole
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._sizing import UniformSizeControl
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_options import NativeMeshingOptions

    model = _sphere_model()
    domain = MeshingDomain.from_brep(model)
    boundary = MeshingScope.whole(model, 2)
    region_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        domain.scope_entity_set_id(3),
        np.asarray(domain.scope_entity_ids(3)[:1], dtype=np.int64),
    )
    region_name = domain.regions[0].name
    control = RegionControl(
        region_scope, region_name, "controlled-solid", RegionRole.SOLID
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.3),),
        region_controls=(control,),
        region_seeds=(
            RegionSeed(
                np.zeros(3, dtype=np.float64),
                region_name,
                "seed-material",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.05,
    )
    plan = phx.meshing.NativeMeshingProvider(
        NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
            ),
        )
    ).plan(model, specification)
    candidate = plan.execute(prove=False)
    assert isinstance(candidate, CellMeshingCandidate)
    assert candidate.mesh.blocks[0].name != "native-cad-tetrahedra"
    (evidence,) = candidate._construction.organization.region_boundary_evidence
    assert evidence.control_id == control.control_id
    assert evidence.material_id == "controlled-solid"


def _field_specification(
    field: str, model: phx.geometry.brep.BRepModel
) -> VolumeMeshingSpec:
    """A coarse sphere specification that sets only ``field``."""
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
    )
    from phydrax.meshing._controls import (
        FeatureKind,
        HoleSeed,
        PeriodicConstraint,
        ProtectedFeature,
        RegionSeed,
    )
    from phydrax.meshing._organization import RegionRole
    from phydrax.meshing._scope import MeshingScope
    from phydrax.meshing._sizing import (
        CurvatureSizeControl,
        SizeCompliancePolicy,
        SizeControl,
        UniformSizeControl,
    )

    boundary = MeshingScope.whole(model, 2)
    solid = MeshingScope.whole(model, 3)
    tetrahedra = CellFamilyPolicy(required=("tetrahedron",))
    size = UniformSizeControl(boundary, 0.3)

    def build(
        *,
        families: CellFamilyPolicy = tetrahedra,
        fill: VolumeFillStrategy = VolumeFillStrategy.SIMPLEX,
        size_controls: tuple[SizeControl, ...] = (size,),
        protected_features: tuple[ProtectedFeature, ...] = (),
        region_seeds: tuple[RegionSeed, ...] = (),
        hole_seeds: tuple[HoleSeed, ...] = (),
        periodic_constraints: tuple[PeriodicConstraint, ...] = (),
        size_compliance: SizeCompliancePolicy | None = None,
        maximum_deviation: float | None = 0.05,
    ) -> VolumeMeshingSpec:
        return VolumeMeshingSpec(
            CellMeshingTarget(3, 3, families),
            boundary,
            fill,
            size_controls=size_controls,
            protected_features=protected_features,
            region_seeds=region_seeds,
            hole_seeds=hole_seeds,
            periodic_constraints=periodic_constraints,
            size_compliance=size_compliance,
            maximum_deviation=maximum_deviation,
        )

    match field:
        case "protected_features":
            return build(
                protected_features=(
                    ProtectedFeature(
                        boundary, FeatureKind.SURFACE, maximum_deviation=0.05
                    ),
                )
            )
        case "region_seeds":
            region = MeshingDomain.from_brep(model).regions[0].name
            return build(
                region_seeds=(
                    RegionSeed(np.zeros(3), region, "seeded", RegionRole.SOLID),
                )
            )
        case "hole_seeds":
            return build(hole_seeds=(HoleSeed(np.zeros(3), solid),))
        case "size_maximum":
            return build(
                size_controls=(UniformSizeControl(boundary, 0.2, maximum_size=0.2),)
            )
        case "size_compliance":
            return build(
                size_controls=(UniformSizeControl(boundary, 0.2),),
                size_compliance=SizeCompliancePolicy(target_statistics=("p95",)),
            )
        case "size_curvature":
            return build(size_controls=(size, CurvatureSizeControl(boundary, 0.5)))
        case "size_growth":
            return build(
                size_controls=(
                    UniformSizeControl(boundary, 0.3, maximum_growth_rate=1.2),
                )
            )
        case "size_volume_scope":
            return build(size_controls=(UniformSizeControl(solid, 0.3),))
        case "periodic_constraints":
            return build(
                periodic_constraints=(PeriodicConstraint(boundary, boundary, np.eye(4)),)
            )
        case "maximum_deviation":
            return build(maximum_deviation=None)
        case "cell_families":
            return build(families=CellFamilyPolicy(required=("hexahedron",)))
        case "fill_strategy":
            return build(fill=VolumeFillStrategy.POLYHEDRAL)
    raise AssertionError(field)


_FIELD_OUTCOMES = (
    ("protected_features", "existing"),
    ("region_seeds", "existing"),
    ("hole_seeds", "PLC source preparation failed (REGION_LEAK"),
    ("size_maximum", "native"),
    ("size_compliance", "native"),
    ("size_curvature", "supports uniform size controls on source faces only"),
    ("size_growth", "patch-size growth controls without a source geodesic"),
    ("size_volume_scope", "size controls must bind source face occurrences"),
    ("periodic_constraints", "does not support periodic constraints"),
    ("maximum_deviation", "requires an explicit maximum_deviation"),
    ("cell_families", "a tetrahedral volume target"),
    ("fill_strategy", "the simplex volume-fill strategy"),
)


@pytest.mark.parametrize(
    ("field", "outcome"),
    _FIELD_OUTCOMES,
    ids=[field for field, _ in _FIELD_OUTCOMES],
)
def test_unproved_cad_solid_honours_or_refuses_every_field(
    field: str, outcome: str
) -> None:
    # No specification field is silently dropped by the native route: it is
    # honoured there (reported size statistics), the plan keeps the existing
    # construction, or planning/execution refuses it with a stated reason.
    from phydrax.meshing import CellMeshingCandidate
    from phydrax.meshing._contracts import MeshingFailure
    from phydrax.meshing._sizing import CAD_CHART_SURFACE_SIZE, UniformSizeControl
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_options import NativeMeshingOptions

    model = _sphere_model()

    def execute() -> CellMeshingCandidate:
        specification = _field_specification(field, model)
        candidate = (
            phx.meshing.NativeMeshingProvider(
                NativeMeshingOptions(
                    "cad_tetrahedral",
                    volume_schedule=NativeVolumeSchedule(
                        radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
                    ),
                )
            )
            .plan(model, specification)
            .execute(prove=False)
        )
        assert isinstance(candidate, CellMeshingCandidate)
        return candidate

    if outcome not in {"native", "existing"}:
        with pytest.raises(MeshingFailure, match=re.escape(outcome)):
            execute()
        return
    candidate = execute()
    native = candidate.mesh.blocks[0].name == "native-cad-tetrahedra"
    assert native == (outcome == "native")
    specification = _field_specification(field, model)
    report = candidate._construction.compliance
    if field == "region_seeds":
        organization = candidate._construction.organization
        assert organization is not None
        assert [zone.material_id for zone in organization.zones] == ["seeded"]
    if field == "protected_features":
        (feature,) = specification.protected_features
        key = f"protected:{feature.feature_id}:maximum_deviation"
        assert dict(report.requested)[key] == 0.05
        assert key in dict(report.achieved)
    if not native:
        return
    (control,) = specification.size_controls
    assert isinstance(control, UniformSizeControl)
    policy = specification.size_compliance
    requested, achieved = dict(report.requested), dict(report.achieved)
    # Requested bound h / sqrt(2) on the boundary circumradius, and the bound
    # its row is an issue above.
    bound = CAD_CHART_SURFACE_SIZE.bound_factor * control.target_size
    bounds = {
        f"{name}_circumradius": (
            bound,
            bound * (1.0 + policy.relative_tolerance) + policy.absolute_tolerance,
        )
        for name in policy.target_statistics
    }
    assert np.isclose(bound, control.target_size / np.sqrt(2.0), rtol=1e-15)
    if control.maximum_size is not None:
        bounds["maximum"] = (control.maximum_size, control.maximum_size)
    rows = {key for key in achieved if key.startswith("size:")}
    assert rows == {f"size:{control.control_id}:{name}" for name in bounds}
    for name, (target, allowed) in bounds.items():
        key = f"size:{control.control_id}:{name}"
        assert requested[key] == target
        assert (key in report.issues) == (achieved[key] > allowed)


@pytest.mark.parametrize(
    ("shape", "deviation"),
    [("sphere", 0.0), ("box", 1e-12)],
    ids=["curved-zero", "planar-tiny"],
)
def test_unattainable_protected_surface_deviation_is_refused_promptly(
    shape: str, deviation: float
) -> None:
    # A curved feature never meets a zero (or near-zero) chord deviation: its
    # curve sampling stops at the surface schedule's sample budget with the
    # unmet deviation and the feature that set it. The same tiny bound on a
    # planar feature is met (straight chords certify at rounding level), so
    # the refusal is the budget, not the requested value.
    from phydrax.meshing import CellMeshingCandidate
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        MeshingFailure,
        MeshingFailureCategory,
        VolumeFillStrategy,
    )
    from phydrax.meshing._controls import FeatureKind, ProtectedFeature
    from phydrax.meshing._scope import MeshingScope
    from phydrax.meshing._sizing import UniformSizeControl
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_options import NativeMeshingOptions

    model = (
        _sphere_model()
        if shape == "sphere"
        else phx.geometry.brep.brep_box(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
    )
    boundary = MeshingScope.whole(model, 2)
    feature = ProtectedFeature(boundary, FeatureKind.SURFACE, maximum_deviation=deviation)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.5),),
        protected_features=(feature,),
        maximum_deviation=0.05,
    )
    plan = phx.meshing.NativeMeshingProvider(
        NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
            ),
        )
    ).plan(model, specification)
    if shape == "box":
        assert isinstance(plan.execute(prove=False), CellMeshingCandidate)
        return
    with pytest.raises(
        MeshingFailure, match="did not meet its chord deviation"
    ) as raised:
        plan.execute(prove=False)
    failure = raised.value
    assert failure.category is MeshingFailureCategory.COMPLIANCE_FAILED
    evidence = failure.evidence
    assert dict(evidence.requested_quantities)["maximum_deviation"] == deviation
    assert dict(evidence.achieved_quantities)["unmet_chord_deviation"] > deviation
    assert evidence.logical_findings == (f"protected_feature:{feature.feature_id}",)


def test_curve_sample_budget_of_a_surface_schedule_bounds_curve_sampling() -> None:
    # A non-default sample budget bounds the same unmet curved chord limit
    # (a hard protected surface at 1e-12) in fewer samples; the budget enters
    # the schedule identity only when it differs from the default.
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        MeshingFailure,
        MeshingFailureCategory,
        SurfaceMeshingSpec,
    )
    from phydrax.meshing._controls import FeatureKind, ProtectedFeature
    from phydrax.meshing._scope import MeshingScope
    from phydrax.meshing._sizing import UniformSizeControl
    from phydrax.meshing.providers._native_options import NativeMeshingOptions
    from phydrax.meshing.providers._native_schedules import (
        DEFAULT_MAXIMUM_CURVE_SAMPLES,
        NativeSurfaceSchedule,
    )

    default = NativeSurfaceSchedule()
    assert (
        NativeSurfaceSchedule(
            maximum_curve_samples=DEFAULT_MAXIMUM_CURVE_SAMPLES
        ).schedule_id
        == default.schedule_id
    )
    schedule = NativeSurfaceSchedule(maximum_curve_samples=256)
    assert schedule.schedule_id != default.schedule_id
    model = _sphere_model()
    scope = MeshingScope.whole(model, 2)
    feature = ProtectedFeature(scope, FeatureKind.SURFACE, maximum_deviation=1e-12)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.5),),
        protected_features=(feature,),
        maximum_deviation=0.05,
    )
    plan = phx.meshing.NativeMeshingProvider(
        NativeMeshingOptions("parametric_surface", surface_schedule=schedule)
    ).plan(model, specification)
    with pytest.raises(MeshingFailure, match="within 256 samples") as raised:
        plan.execute(prove=False)
    failure = raised.value
    assert failure.category is MeshingFailureCategory.COMPLIANCE_FAILED
    evidence = failure.evidence
    requested = dict(evidence.requested_quantities)
    assert requested == {"maximum_curve_samples": 256.0, "maximum_deviation": 1e-12}
    assert dict(evidence.achieved_quantities)["unmet_chord_deviation"] > 1e-12
    assert evidence.logical_findings == (f"protected_feature:{feature.feature_id}",)
