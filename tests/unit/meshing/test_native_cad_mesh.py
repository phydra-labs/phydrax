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
    CadProximityControl,
    CadSolidResult,
    CadSurfaceResult,
    load_meshcore,
    MeshcoreStatus,
    verify_cad_fidelity,
)
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.meshing import CellMeshingCandidate
from phydrax.meshing._contracts import VolumeMeshingSpec
from phydrax.meshing._organization import (
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import ProximitySizeControl
from phydrax.meshing.providers._native import _PreparedDirectBRepCadVolume
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
        # V3 associates sphere pole-region rows in the pole chart
        # (x, y) = rho (cos u, sin u): their reference is sampled there.
        poles = np.asarray(model.use_kinds)[
            chart.chord_use[chart.use_patch[chart.chord_use] == patch]
        ]
        if np.any(poles == 1):
            half = np.float64(np.pi / 2)
            for sigma in (1.0, -1.0):
                local = np.flatnonzero(
                    (uv[:, :, 1] == sigma * half).any(axis=1)
                    | (sigma * uv[:, :, 1] >= float.fromhex("0x1.3f004dc9db074p-1")).any(
                        axis=1
                    )
                )
                if local.size == 0 or not np.any(uv[:, :, 1] == sigma * half):
                    continue
                rho = half - sigma * uv[local, :, 1]
                pole = uv[local, :, 1] == sigma * half
                x = np.where(pole, 0.0, rho * np.cos(uv[local, :, 0])) @ weights.T
                y = np.where(pole, 0.0, rho * np.sin(uv[local, :, 0])) @ weights.T
                chart_points = np.stack(
                    (np.arctan2(y, x), sigma * (half - np.hypot(x, y))), axis=-1
                )
                pole_exact = np.asarray(
                    surface.evaluate(
                        jnp.asarray(chart_points.reshape((-1, 2)), dtype=jnp.float64)
                    ),
                    dtype=np.float64,
                ).reshape((local.size, len(weights), 3))
                measured[rows[local]] = np.max(
                    np.linalg.norm(pole_exact - linear[local], axis=2), axis=1
                )
            # A planar disk center (pole chord at v = pole_v): rows touching
            # it take the disk-center chart (x, y) = |v - pole_v| (cos u,
            # sin u), where S is affine.
            pole_chords = (chart.use_patch[chart.chord_use] == patch) & (
                np.asarray(model.use_kinds)[chart.chord_use] == 1
            )
            for pole_v in np.unique(chart.chord_uv[pole_chords][:, :, 1]):
                if abs(pole_v) == half:
                    continue
                local = np.flatnonzero((uv[:, :, 1] == pole_v).any(axis=1))
                if local.size == 0:
                    continue
                sigma = 1.0 if np.all(uv[local, :, 1] >= pole_v) else -1.0
                probe = np.asarray(
                    surface.evaluate(
                        jnp.asarray(
                            [
                                [0.0, pole_v],
                                [0.0, pole_v + sigma * 0.1],
                                [1.0, pole_v + sigma * 0.1],
                                [2.0, pole_v + sigma * 0.2],
                            ],
                            dtype=jnp.float64,
                        )
                    ),
                    dtype=np.float64,
                )
                normal = np.cross(probe[1] - probe[0], probe[2] - probe[0])
                if abs(float(np.dot(normal, probe[3] - probe[0]))) > 1e-12 * float(
                    np.linalg.norm(normal)
                ):
                    continue
                rho = sigma * (uv[local, :, 1] - pole_v)
                pole = uv[local, :, 1] == pole_v
                x = np.where(pole, 0.0, rho * np.cos(uv[local, :, 0])) @ weights.T
                y = np.where(pole, 0.0, rho * np.sin(uv[local, :, 0])) @ weights.T
                chart_points = np.stack(
                    (np.arctan2(y, x), pole_v + sigma * np.hypot(x, y)), axis=-1
                )
                flat_exact = np.asarray(
                    surface.evaluate(
                        jnp.asarray(chart_points.reshape((-1, 2)), dtype=jnp.float64)
                    ),
                    dtype=np.float64,
                ).reshape((local.size, len(weights), 3))
                measured[rows[local]] = np.max(
                    np.linalg.norm(flat_exact - linear[local], axis=2), axis=1
                )
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


def _unit_cap(latitude_degrees: float, above: bool) -> phx.geometry.brep.BRepModel:
    """The unit ball's part above (or below) a plane at the given latitude."""
    brep = phx.geometry.brep
    z = float(np.sin(np.radians(latitude_degrees)))
    r = float(np.sqrt(1.0 - z * z))
    if above:
        vertices = ((0.0, z), (r, z), (0.0, 1.0))
        segments = (brep.ProfileLine(), brep.ProfileArc((0.0, 0.0)), brep.ProfileLine())
    else:
        vertices = ((0.0, -1.0), (r, z), (0.0, z))
        segments = (brep.ProfileArc((0.0, 0.0)), brep.ProfileLine(), brep.ProfileLine())
    return brep.brep_revolution(
        brep.PlanarProfile(
            brep.ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            brep.ProfileLoop(vertices, segments),
        ),
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )


_COARSE_POLE_SHAPES = {
    "sphere": lambda: phx.geometry.brep.brep_sphere(
        1.0, coordinate_contract=phx.SpatialCoordinateContract.si()
    ),
    # Hemispheres, and caps cut near the first and second ring latitudes and
    # near the pole (a pole zone reaching or nearly reaching the trim).
    **{
        f"cap_{side}_{degrees}": (
            lambda degrees=degrees, above=(side == "above"): _unit_cap(degrees, above)
        )
        for side, latitudes in (
            ("above", (0, 20, 45, 55, 70, 85)),
            ("below", (0, 20, 40)),
        )
        for degrees in latitudes
    },
    # Short cones: the apex zone grows to the base circle.
    **{
        f"cone_{height}": (
            lambda height=height: phx.geometry.brep.brep_cone(
                1.0, 0.0, height, coordinate_contract=phx.SpatialCoordinateContract.si()
            )
        )
        for height in (0.1, 0.3, 1.0)
    },
}


@pytest.mark.parametrize("name", sorted(_COARSE_POLE_SHAPES))
def test_coarse_pole_zones_stop_short_of_other_fixed_vertices(name: str) -> None:
    # A pole zone's seeded rings used to end within ulps of the other pole's
    # last ring (the sphere at h/r 0.6 and 0.8, theta 0.6-0.9: two equator
    # rings 2 ulps apart). The zero-area strip between them sent the
    # fixed-boundary volume stage into minutes of improvement (now refused
    # before recovery). Zones stop half a spacing short of the midline to
    # another pole; ring points near a trim are omitted by the boundary
    # proximity rule. Over h/r 0.4-0.8 and theta 0.5-0.9: no edge of the
    # surface route (the solid route grades a rim to its wedge,
    # test_rim_wedges_above_the_dihedral_target_resolve) is shorter than
    # h / 10, and the volume stage does bounded work (the sphere resolves,
    # with at most the bounded improvement below; the thin caps and cones
    # may leave coarse-quality targets unmet, promptly).
    model = cad_geometry_bundle(
        MeshingDomain.from_brep(_COARSE_POLE_SHAPES[name]())
    ).model
    faces, curves = model.surface_nodes.size, model.curve_nodes.size
    for size in (0.4, 0.6, 0.8):
        for theta in (0.5, 0.6, 0.7, 0.8, 0.9):
            tolerance = size / 6.0
            arguments = (
                model,
                np.full((faces,), size, dtype=np.float64),
                np.full((faces,), tolerance, dtype=np.float64),
                np.full((curves,), tolerance, dtype=np.float64),
            )
            surface_only = cad_mesh(*arguments, tolerance_fraction=theta)
            corners = surface_only.points[surface_only.triangles]
            edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
            assert edges.min() > 0.1 * size, (size, theta)
            solid = cad_mesh_solid(*arguments, tolerance_fraction=theta)
            surface = solid.surface
            stats = solid.volume.stats
            assert solid.volume.tetrahedra.shape[0] > 0, (size, theta)
            # Bounded: at most 4 insertions per surface triangle (the hang
            # made 270 per triangle).
            work = int(stats["refine_inserted"][0]) + int(stats["improve_inserted"][0])
            assert work <= 4 * surface.triangles.shape[0], (size, theta, work)
            if name == "sphere":
                assert solid.status is MeshcoreStatus.OK, (size, theta)


@pytest.mark.parametrize("radius", (0.5, 1.0, 1.7, 3.0))
def test_solid_spheres_snap_graded_pole_rings_to_the_seam(radius: float) -> None:
    # Solid route pole rings start at the graded base size at the pole, the
    # chord of the seam's first segment, while that vertex sits at the arc
    # offset (longer by O(kappa^2 L^3)). Comparing the arc against the chord
    # spacing refused the snap, the first ring sat 2e-4 short of the seam
    # vertex (unit sphere, h 0.2), the ring points beside the seam were
    # omitted and 8 pole triangles failed the stored size test: the surface
    # stage was UNRESOLVED (42 of 60 runs of this grid). Every run resolves
    # with no face failure (fails if the snap window is the graded spacing).
    model = cad_geometry_bundle(
        MeshingDomain.from_brep(
            phx.geometry.brep.brep_sphere(
                radius, coordinate_contract=phx.SpatialCoordinateContract.si()
            )
        )
    ).model
    curves = model.curve_nodes.size
    for size in (0.1, 0.2, 0.3, 0.5):
        tolerance = size * size / (4.0 * radius)  # kappa h^2 / 4
        for theta in (0.7, 0.8, 0.9):
            solid = cad_mesh_solid(
                model,
                np.full((1,), size, dtype=np.float64),
                np.full((1,), tolerance, dtype=np.float64),
                np.full((curves,), tolerance, dtype=np.float64),
                tolerance_fraction=theta,
            )
            assert solid.surface.face_failures.shape[0] == 0, (size, theta)
            assert solid.status is MeshcoreStatus.OK, (size, theta)


@pytest.mark.parametrize("latitude", (45, 50, 55, 60, 65, 70, 75, 80, 85))
def test_coarse_caps_below_a_polar_trim_resolve(latitude: int) -> None:
    # The unit ball below a latitude cut at 45-85 deg: the dome's trim circle
    # lies where the latitude chart is as anisotropic as beside a pole. Before
    # trim-line zones (rings seeded from the trim as from a pole) and the
    # side-integrated size confirmation, 55 of these 135 runs returned the
    # face stage UNRESOLVED (refusals beside the trim; with the confirmation
    # alone most runs at 75-85 deg still did, with the zones alone one at 45
    # deg). Each run must resolve, keep every edge over h / 10, keep every
    # triangle's circumradius within h / sqrt(2) and stay within theta times
    # the requested tolerance of the sphere: the distance of a flat
    # triangle's point x to the unit sphere is |1 - |x||, sampled on a
    # barycentric grid. Fault caught: triangles from the pole zone's last
    # ring to the trim zone's, all of fixed points and never refined (cap
    # below 85 deg, h/r 0.6, theta 0.9: circumradius 1.026 h / sqrt(2),
    # distance 0.99 tol).
    domain = MeshingDomain.from_brep(_unit_cap(float(latitude), above=False))
    model = cad_geometry_bundle(domain).model
    faces, curves = model.surface_nodes.size, model.curve_nodes.size
    steps = 8
    weights = np.asarray(
        [
            (1 - (i + j) / steps, i / steps, j / steps)
            for i in range(steps + 1)
            for j in range(steps + 1 - i)
        ],
        dtype=np.float64,
    )
    for size in (0.4, 0.6, 0.8):
        for theta in (0.5, 0.6, 0.7, 0.8, 0.9):
            tolerance = size / 6.0
            result = cad_mesh(
                model,
                np.full((faces,), size, dtype=np.float64),
                np.full((faces,), tolerance, dtype=np.float64),
                np.full((curves,), tolerance, dtype=np.float64),
                tolerance_fraction=theta,
            )
            assert result.status is MeshcoreStatus.OK, (size, theta)
            corners = result.points[result.triangles]
            edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
            assert edges.min() > 0.1 * size, (size, theta)
            sides = edges.prod(axis=1)
            area2 = np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
            circumradius = sides / (2.0 * area2)
            assert circumradius.max() <= size / np.sqrt(2.0), (
                size,
                theta,
                circumradius.max(),
            )
            radius = np.linalg.norm(corners, axis=2)
            dome = np.all(np.abs(radius - 1.0) < 1e-12, axis=1)
            assert np.any(dome) and not np.all(dome)
            samples = np.einsum("kv,rvd->rkd", weights, corners[dome])
            distance = np.abs(1.0 - np.linalg.norm(samples, axis=2))
            assert distance.max() <= theta * tolerance, (size, theta, distance.max())


@pytest.mark.parametrize("latitude", (50, 65, 85))
def test_polar_trim_band_rows_meet_the_chart_linear_bound(latitude: int) -> None:
    # The unit ball below a latitude cut: its north pole is trimmed away, so
    # V3 certifies the rows beside the trim in the (longitude, latitude)
    # chart, by the chart-linear deviation |S - I_T|. A row with corners x_i
    # deviates there at most (A du^2 + 2 B du dv + C dv^2) / 8 (A, B, C the
    # bounds of |S_uu| = cos v, |S_uv| = |sin v|, |S_vv| <= |cos v| + |sin v|
    # over its latitudes, du, dv its chart extents; cad_faces.hpp). The trim
    # circle's chords and a seam vertex at the design band (curve stage), the
    # band ring snapped to it and aligned to the trim's vertices (face stage)
    # are sized by that bound, so every row with a corner on the trim meets
    # it within theta tol. Faults caught: chords sized in 3D alone (the trim
    # at 85 deg in three chords, du = 2 pi / 3: rows over the bound, V3
    # refused them though the flat triangles lay within theta tol), and a
    # band ring chain left open at the seam (the seam-end row reaching the
    # seam's next vertex: up to 2.3 times the bound).
    trim = np.radians(float(latitude))
    model = cad_geometry_bundle(
        MeshingDomain.from_brep(_unit_cap(float(latitude), above=False))
    ).model
    faces, curves = model.surface_nodes.size, model.curve_nodes.size
    for size in (0.4, 0.6, 0.8):
        for theta in (0.5, 0.7, 0.9):
            tolerance = size / 6.0
            result = cad_mesh(
                model,
                np.full((faces,), size, dtype=np.float64),
                np.full((faces,), tolerance, dtype=np.float64),
                np.full((curves,), tolerance, dtype=np.float64),
                tolerance_fraction=theta,
            )
            assert result.status is MeshcoreStatus.OK, (size, theta)
            uv = np.asarray(result.chart.uv)
            corners = result.points[np.asarray(result.chart.triangles)]
            dome = np.all(np.abs(np.linalg.norm(corners, axis=2) - 1.0) < 1e-12, axis=1)
            on_trim = np.abs(uv[:, :, 1] - trim) < 1e-12
            rows = dome & on_trim.any(axis=1)
            assert rows.any(), (size, theta)
            latitudes = uv[rows, :, 1]
            du = np.ptp(uv[rows, :, 0], axis=1)
            dv = np.ptp(latitudes, axis=1)
            low, high = latitudes.min(axis=1), latitudes.max(axis=1)
            a = np.maximum(np.abs(np.cos(low)), np.abs(np.cos(high)))
            b = np.maximum(np.abs(np.sin(low)), np.abs(np.sin(high)))
            bound = (a * du * du + 2.0 * b * du * dv + (a + b) * dv * dv) / 8.0
            assert bound.max() <= theta * tolerance, (size, theta, bound.max())


@pytest.mark.parametrize(
    ("latitude", "size", "theta"),
    (
        (55.02410624307131, 0.7399833208919971, 0.8262382502818686),
        (84.90768372797963, 0.7379609933402859, 0.848268379781792),
    ),
)
def test_polar_trim_zone_rows_meet_the_chart_linear_bound(
    latitude: float, size: float, theta: float
) -> None:
    # A polar trim's zone has a ring at each seam vertex (the seam sized by
    # the chart-linear bound (A du^2 + 2 B du dv + C dv^2) / 8 of the test
    # above), each ring's stops the previous ring's cut or joined, pieces
    # within the bound over the gap: every zone row meets it. The zone runs
    # to the first seam vertex at or below latitude 45 deg (where |S_u|^2 =
    # |S_v|^2 / 2), so every row with a corner above that is a zone row and
    # meets the bound within theta tol. Faults caught (random coarse caps
    # that resolved before the band sizing): rings placed by the spacing past
    # the band ring, none after it beside a 31 deg seam segment (the cap below
    # 55 deg: oversize refusal), and a band ring short of the seam's design
    # band vertex, its chain open (the cap below 84.9 deg).
    model = cad_geometry_bundle(
        MeshingDomain.from_brep(_unit_cap(latitude, above=False))
    ).model
    faces, curves = model.surface_nodes.size, model.curve_nodes.size
    tolerance = size / 6.0
    result = cad_mesh(
        model,
        np.full((faces,), size, dtype=np.float64),
        np.full((faces,), tolerance, dtype=np.float64),
        np.full((curves,), tolerance, dtype=np.float64),
        tolerance_fraction=theta,
    )
    assert result.status is MeshcoreStatus.OK
    uv = np.asarray(result.chart.uv)
    corners = result.points[np.asarray(result.chart.triangles)]
    dome = np.all(np.abs(np.linalg.norm(corners, axis=2) - 1.0) < 1e-12, axis=1)
    rows = dome & (uv[:, :, 1].max(axis=1) > np.pi / 4.0)
    assert rows.any()
    latitudes = uv[rows, :, 1]
    du = np.ptp(uv[rows, :, 0], axis=1)
    dv = np.ptp(latitudes, axis=1)
    low, high = latitudes.min(axis=1), latitudes.max(axis=1)
    a = np.maximum(np.abs(np.cos(low)), np.abs(np.cos(high)))
    b = np.maximum(np.abs(np.sin(low)), np.abs(np.sin(high)))
    bound = (a * du * du + 2.0 * b * du * dv + (a + b) * dv * dv) / 8.0
    assert bound.max() <= theta * tolerance, bound.max() / (theta * tolerance)


@pytest.mark.parametrize(
    ("latitude", "size", "theta"),
    (
        (55.77753523254207, 0.5324148676759605, 0.8955251222934396),
        (60.80933008398488, 0.7473363451869702, 0.7660372624808137),
    ),
)
def test_pole_region_boundary_rows_certify(
    latitude: float, size: float, theta: float
) -> None:
    # V3 certifies a pole's rows in the pole chart when a corner lies at
    # |latitude| >= kCapLatitude (35.7 deg), the rest of the face in the
    # (longitude, latitude) chart; an edge between the two sets is charged its
    # seam lens on both rows. A face whose mesh has such a region-boundary
    # edge over tol (pole_boundary_overruns) is meshed again with the kernel
    # splitting those over theta tol (pole_boundary_split) and the moves after
    # it kept within tol, so both rows of every such pair certify. Fault
    # caught (random coarse caps below a polar trim, which resolved): without
    # the splits V3 refused 2 and 6 rows beside the south pole's region.
    model_domain = MeshingDomain.from_brep(_unit_cap(latitude, above=False))
    bundle = cad_geometry_bundle(model_domain)
    model = bundle.model
    faces, curves = model.surface_nodes.size, model.curve_nodes.size
    tolerance = size / 6.0
    result = cad_mesh(
        model,
        np.full((faces,), size, dtype=np.float64),
        np.full((faces,), tolerance, dtype=np.float64),
        np.full((curves,), tolerance, dtype=np.float64),
        tolerance_fraction=theta,
    )
    assert result.status is MeshcoreStatus.OK
    chart = result.chart
    lower, upper = coefficient_bounds(bundle, model_domain)
    verdict = verify_cad_fidelity(
        model,
        lower,
        upper,
        CadFidelityMesh(
            result.points,
            chart.triangles,
            chart.patches,
            chart.uv,
            result.triangles.astype(np.int64),
            chart.chart_to_mesh,
            chart.edge_chord,
        ),
        CadFidelityChords(
            chart.use_patch,
            chart.use_range,
            len(model.loop_surfaces),
            chart.chord_use,
            chart.chord_loop,
            chart.chord_uv,
            chart.chord_range,
        ),
        float(tolerance),
        8,
    )
    uv = np.asarray(chart.uv)
    corners = result.points[np.asarray(chart.triangles)]
    dome = np.all(np.abs(np.linalg.norm(corners, axis=2) - 1.0) < 1e-12, axis=1)
    cap = np.radians(35.7)
    region = dome & (uv[:, :, 1] <= -cap).any(axis=1)
    boundary = dome & ~region & (uv[:, :, 1] < 0.0).any(axis=1)
    assert region.any() and boundary.any()
    assert np.all(np.asarray(verdict.item_status)[region | boundary] == MeshcoreStatus.OK)
    assert verdict.status == MeshcoreStatus.OK


def _cell_quality(points: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: the smallest dihedral angle (degrees) and the radius-edge ratio.

    Closed forms: the dihedral at edge (i, j) is the angle between the other
    two corners' offsets projected across the edge; the circumcenter solves
    2 (p_k - p_0) . c = |p_k|^2 - |p_0|^2.
    """
    corners = points[cells]
    smallest = np.full((cells.shape[0],), np.inf, dtype=np.float64)
    for i, j, k, m in (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ):
        axis = corners[:, j] - corners[:, i]
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        a = corners[:, k] - corners[:, i]
        b = corners[:, m] - corners[:, i]
        a -= np.sum(a * axis, axis=1)[:, None] * axis
        b -= np.sum(b * axis, axis=1)[:, None] * axis
        cosine = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        smallest = np.minimum(smallest, np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    matrix = 2.0 * (corners[:, 1:] - corners[:, :1])
    rhs = np.sum(corners[:, 1:] ** 2, axis=2) - np.sum(corners[:, :1] ** 2, axis=2)
    center = np.linalg.solve(matrix, rhs[:, :, None])[:, :, 0]
    radius = np.linalg.norm(center - corners[:, 0], axis=1)
    edges = np.stack(
        [
            np.linalg.norm(corners[:, p] - corners[:, q], axis=1)
            for p in range(4)
            for q in range(p + 1, 4)
        ],
        axis=1,
    )
    return smallest, radius / edges.min(axis=1)


def _rim_cone(height: float) -> phx.geometry.brep.BRepModel:
    return phx.geometry.brep.brep_cone(
        1.0, 0.0, height, coordinate_contract=phx.SpatialCoordinateContract.si()
    )


def _rim_solid(model: CadModel, size: float, theta: float) -> CadSolidResult:
    faces, curves = model.surface_nodes.size, model.curve_nodes.size
    tolerance = size / 6.0
    return cad_mesh_solid(
        model,
        np.full((faces,), size, dtype=np.float64),
        np.full((faces,), tolerance, dtype=np.float64),
        np.full((curves,), tolerance, dtype=np.float64),
        tolerance_fraction=theta,
    )


# True rim wedges: a cap above latitude phi meets its disk at 90 - phi deg, a
# unit-radius cone of height g its base at atan(g).
_RIM_WEDGE_SHAPES = {
    **{
        f"cap_above_{d}": (lambda d=d: _unit_cap(float(d), above=True))
        for d in (45, 55, 60, 65)
    },
    **{f"cone_{g}": (lambda g=g: _rim_cone(g)) for g in (0.5, 1.0)},
}


@pytest.mark.parametrize("name", sorted(_RIM_WEDGE_SHAPES))
def test_rim_wedges_above_the_dihedral_target_resolve(name: str) -> None:
    # A rim edge's cells lie in the meshed wedge between its two boundary
    # triangles, whose chords tilt off the curved face by ~ kappa s / 2. With
    # the rim spacing at the request size, caps above 60-65 deg (true wedge
    # 30-25 deg) meshed their rim near the 15.5 deg target and ended
    # UNRESOLVED. The rim size (cad_curves.cpp) keeps half the margin over
    # the target: every run resolves, and every cell meets the targets by
    # the closed forms above (fails if the rim rule is off).
    model = cad_geometry_bundle(MeshingDomain.from_brep(_RIM_WEDGE_SHAPES[name]())).model
    for size in (0.4, 0.6, 0.8):
        for theta in (0.5, 0.7, 0.9):
            solid = _rim_solid(model, size, theta)
            assert solid.status is MeshcoreStatus.OK, (size, theta)
            dihedral, ratio = _cell_quality(solid.volume.points, solid.volume.tetrahedra)
            assert dihedral.min() >= 15.5 - 1e-9, (size, theta, dihedral.min())
            assert ratio.max() <= 2.0 + 1e-9, (size, theta, ratio.max())


def _rim_cells(
    points: np.ndarray, cells: np.ndarray, rim_z: float, rim_radius: float
) -> np.ndarray:
    """Per cell: whether one of its edges has both ends on the rim circle."""
    on_rim = (np.abs(points[:, 2] - rim_z) < 1e-12) & (
        np.abs(np.hypot(points[:, 0], points[:, 1]) - rim_radius) < 1e-12
    )
    return on_rim[cells].sum(axis=1) >= 2


def test_unmet_cells_split_into_forced_and_unforced_per_cell() -> None:
    # Both kinds in one solid: the cone of height 0.25 (true rim wedge
    # atan(0.25) = 14.0 deg, below the 15.5 deg target) at h 0.2. A rim cell
    # whose only failure is its dihedral at the rim is forced (no mesh of
    # that boundary meets the target there); a cell also failing
    # radius-edge, which the wedge never forces (r/e = sqrt(1/2 +
    # tan^2(w/2)/4) <= 2 is reachable), or failing off the rim is unforced.
    # The decision is per cell: the run is UNRESOLVED because some unmet
    # cell is unforced, and the closed forms bound both counts: forced cells
    # number at most the rim cells below the dihedral target that meet the
    # radius-edge bound, and the two counts cover every cell the closed forms
    # find below a target (fails if a cell failing radius-edge or off the rim
    # counts as forced, if no cell is forced, or if the counts miss cells).
    model = cad_geometry_bundle(MeshingDomain.from_brep(_rim_cone(0.25))).model
    solid = _rim_solid(model, 0.2, 0.7)
    stats = solid.volume.stats
    forced = int(stats["unmet_forced_cells"][0])
    unforced = int(stats["unmet_unforced_cells"][0])
    assert solid.status is MeshcoreStatus.UNRESOLVED
    assert forced > 0 and unforced > 0
    # One set: the split partitions unmet_cells exactly.
    assert forced + unforced == int(stats["unmet_cells"][0])
    points, cells = solid.volume.points, solid.volume.tetrahedra
    dihedral, ratio = _cell_quality(points, cells)
    rim = _rim_cells(points, cells, 0.0, 1.0)
    assert forced <= int((rim & (dihedral < 15.5) & (ratio <= 2.0)).sum())
    assert forced + unforced >= int(((dihedral < 15.5) | (ratio > 2.0)).sum())
    # The input finding: the rim curve and its exact true wedge atan(0.25)
    # (cone side against base disk), the only curve below the target (fails
    # if the wedge is taken from the mesh instead of the faces, or a reflex
    # or seam curve is reported).
    wedges = solid.volume.curve_rim_wedges
    below = np.flatnonzero(wedges < 15.5)
    assert below.size == 1
    assert abs(float(wedges[below[0]]) - np.degrees(np.arctan(0.25))) < 1e-9


def test_soft_shape_schedule_and_forced_cells_agree() -> None:
    # The rim cone of height 0.25 at h 0.2 has forced and unforced unmet
    # cells. A hard schedule makes the unforced cells an issue (with the rim
    # finding); a soft one publishes the unmet shape rows and only unforced
    # hard cells (validity, determinant, nonrepresentable shape) stay issues:
    # a forced cell is never a hard cell (fails if the native hard count
    # includes forced cells or the soft route reports shape-only cells).
    from phydrax.meshing.providers._native_cad_mesh import construct_cad_solid

    reports = {}
    for soft in (False, True):
        plan = _cad_volume_plan(
            _rim_cone(0.25), 0.2, 1, radius_edge=2.0, dihedral=15.5, soft_shape=soft
        )
        prepared = plan.prepared
        assert isinstance(prepared, _PreparedDirectBRepCadVolume)
        stats = construct_cad_solid(prepared).volume.stats[0]
        forced = int(stats["unmet_forced_cells"])
        unforced = int(stats["unmet_unforced_cells"])
        hard = int(stats["unmet_hard_cells"])
        assert forced > 0 and unforced > 0
        assert hard <= unforced
        candidate = plan.execute(prove=False)
        report = candidate._construction.compliance
        achieved = dict(report.achieved)
        assert achieved["unmet_forced_quality_cells"] == forced
        assert achieved["unmet_unforced_quality_cells"] == unforced
        reports[soft] = (report, hard, int(stats["unmet_shape_rows"]))
    report, _, _ = reports[False]
    unforced = int(dict(report.achieved)["unmet_unforced_quality_cells"])
    assert f"native_unmet_quality_cells:{unforced}" in report.issues
    assert any(
        issue.startswith("input_rim_wedge_below_dihedral_target:")
        for issue in report.issues
    )
    report, hard, shape_rows = reports[True]
    assert dict(report.achieved)["soft_shape_target_unmet_count"] == shape_rows > 0
    quality = [issue for issue in report.issues if issue.startswith("native_unmet_")]
    assert quality == ([f"native_unmet_quality_cells:{hard}"] if hard else [])
    assert not any(issue.startswith("input_rim_wedge") for issue in report.issues)


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


def test_quadric_refused_triangle_beside_a_curve_resolves_by_a_flip(
    domain: MeshingDomain,
) -> None:
    # Quadric at h 0.0998, theta 0.8, tolerance kappa h^2 / 4 per face (kappa
    # the faces' curvature 1 / radius: 1 on the vertical cylinder's faces 0
    # and 1, 5/3 on the transverse one), curves at their faces' smallest. The
    # chart kernel left one triangle on each mirror face, on curve segment
    # 10 / 49, at metric circumradius 0.78 > 1/sqrt(2): a neighbor's vertex
    # inserted under its own (smaller) metric sat 0.43 from the circumcenter
    # in the triangle's centroid metric, every candidate was within 0.5 of a
    # vertex and the long edge's midpoint encroached the segment, so the face
    # stage returned UNRESOLVED with two encroachment refusals. Flipping the
    # long unconstrained edge leaves both triangles under the bound. Fault
    # caught: a refused triangle left in place instead of flipped.
    model = cad_geometry_bundle(domain).model
    size = 0.0998
    curvature = np.asarray([1.0, 1.0, 5.0 / 3.0], dtype=np.float64)
    face_tolerances = curvature * size * size / 4.0
    curve_tolerances = np.full((model.curve_nodes.size,), np.inf, dtype=np.float64)
    offsets = np.asarray(model.loop_use_offsets)
    for loop, face in enumerate(np.asarray(model.loop_surfaces)):
        for use in range(offsets[loop], offsets[loop + 1]):
            if model.use_kinds[use] == 0:
                curve = model.use_entities[use]
                curve_tolerances[curve] = min(
                    curve_tolerances[curve], face_tolerances[face]
                )
    result = cad_mesh(
        model,
        np.full((model.surface_nodes.size,), size, dtype=np.float64),
        face_tolerances,
        curve_tolerances,
        tolerance_fraction=0.8,
    )
    assert result.status is MeshcoreStatus.OK
    assert result.face_failures.size == 0
    assert result.triangles.shape[0] > 0


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
    model: phx.geometry.brep.BRepModel | MeshingDomain,
    size: float,
    order: int,
    /,
    *,
    radius_edge: float = 2.0,
    dihedral: float = 15.5,
    soft_shape: bool = False,
) -> phx.meshing.NativeMeshingPlan:
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
        VolumeMeshingSpec,
    )
    from phydrax.meshing._scope import MeshingScope
    from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
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
                radius_edge_bound=radius_edge,
                minimum_dihedral_degrees=dihedral,
                shape_strength=SizeControlStrength.SOFT
                if soft_shape
                else SizeControlStrength.HARD,
            ),
        )
    ).plan(model, specification)


def test_shape_strength_decides_whether_unmet_shape_cells_are_issues() -> None:
    # A dihedral aim the box cannot reach leaves unmet shape cells: a hard
    # schedule makes them an issue, a soft one publishes their count.
    box = phx.geometry.brep.brep_box(
        np.zeros(3),
        np.ones(3),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    for soft in (False, True):
        candidate = _cad_volume_plan(
            box, 0.5, 1, radius_edge=1.0, dihedral=60.0, soft_shape=soft
        ).execute(prove=False)
        report = candidate._construction.compliance
        achieved = dict(report.achieved)
        assert achieved["unmet_quality_cells"] > 0
        unmet = [issue for issue in report.issues if issue.startswith("native_unmet")]
        if soft:
            assert not unmet
            assert achieved["soft_shape_target_unmet_count"] > 0
        else:
            assert unmet
            assert "soft_shape_target_unmet_count" not in achieved


def test_soft_shape_keeps_a_determinant_failure_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The schedule's determinant floor is fixed; raising it in the native call
    # makes validity rows, which a soft shape schedule must still gate.
    import phydrax.meshing.providers._native_cad_mesh as native_cad_mesh

    construct = native_cad_mesh.cad_mesh_solid

    def strict(
        model: CadModel,
        face_sizes: object,
        face_tolerances: object,
        curve_tolerances: object,
        /,
        *,
        radius_edge_bound: float,
        min_dihedral_degrees: float,
        min_relative_determinant: float,
        geometry_order: int,
        proximity_controls: tuple[CadProximityControl, ...],
    ) -> CadSolidResult:
        assert min_relative_determinant < 0.9
        return construct(
            model,
            face_sizes,
            face_tolerances,
            curve_tolerances,
            radius_edge_bound=radius_edge_bound,
            min_dihedral_degrees=min_dihedral_degrees,
            min_relative_determinant=0.9,
            geometry_order=geometry_order,
            proximity_controls=proximity_controls,
        )

    monkeypatch.setattr(native_cad_mesh, "cad_mesh_solid", strict)
    box = phx.geometry.brep.brep_box(
        np.zeros(3),
        np.ones(3),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    candidate = _cad_volume_plan(
        box, 0.5, 1, radius_edge=8.0, dihedral=0.0, soft_shape=True
    ).execute(prove=False)
    report = candidate._construction.compliance
    assert any(issue.startswith("native_unmet_quality_cells:") for issue in report.issues)


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


def test_stored_size_allowance_passes_rounded_ties_and_refuses_real_excess() -> None:
    # Fault caught: an allowance missing (a leg one ulp past the tie fails)
    # or too wide (a leg 1e-9 past the tie passes).
    from phydrax.meshing._sizing import (
        CAD_CHART_SURFACE_SIZE,
        SizeCompliancePolicy,
        stored_circumradius_measures,
    )

    exact = SizeCompliancePolicy()
    for size in (0.5, 0.15, 0.1):
        up = np.nextafter(np.nextafter(size, 1.0), 1.0)
        for legs, exceeds in (
            ([(size, size)] * 4, False),
            ([(up, up)] * 4, False),
            ([(size, size * (1.0 + 1e-9))] * 4, True),
        ):
            points, triangles = _right_triangles(legs)
            measures = stored_circumradius_measures(points, triangles)
            assert (
                CAD_CHART_SURFACE_SIZE.exceeds(measures, "p95", size, exact) is exceeds
            ), (legs, size)


def test_stored_size_allowance_bounds_vertex_perturbations() -> None:
    # Independent reference: R^2 of triangles whose vertices are moved by at
    # most eps = sqrt(3) 4u M (the allowance's perturbation radius), computed
    # in exact rationals, stays within (1 + tau) of the stored R^2 both ways.
    # Fault caught: a sensitivity term dropped from tau (an obtuse triangle's
    # area term dominates).
    from fractions import Fraction

    from phydrax.meshing._sizing import stored_size_allowance

    def squared(corners: list[tuple[Fraction, ...]]) -> Fraction:
        p, q, r = corners
        e0 = [b - a for a, b in zip(q, r, strict=True)]
        e1 = [b - a for a, b in zip(r, p, strict=True)]
        e2 = [b - a for a, b in zip(p, q, strict=True)]
        normal = (
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        )
        product = Fraction(1)
        for side in (e0, e1, e2):
            product *= sum(value * value for value in side)
        return product / (4 * sum(value * value for value in normal))

    generator = np.random.default_rng(3)
    unit = float(np.finfo(np.float64).eps) / 2.0
    for corners in (
        np.asarray([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0]]),
        np.asarray([[0.3, 0.1, 0.2], [0.45, 0.1, 0.2], [0.375, 0.13, 0.25]]),
        np.asarray([[1.0, 2.0, 3.0], [1.5, 2.0, 3.0], [1.25, 2.02, 3.0]]),
    ):
        tau = Fraction(float(stored_size_allowance(corners, np.asarray([[0, 1, 2]]))[0]))
        assert tau > 0
        eps = np.sqrt(3.0) * 4.0 * unit * float(np.max(np.abs(corners)))
        stored = squared([tuple(Fraction(float(x)) for x in row) for row in corners])
        for _ in range(50):
            directions = generator.normal(size=(3, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            moved = [
                tuple(
                    Fraction(float(x)) + Fraction(float(eps)) * Fraction(float(d))
                    for x, d in zip(row, direction, strict=True)
                )
                for row, direction in zip(
                    corners.tolist(), directions.tolist(), strict=True
                )
            ]
            ratio = squared(moved) / stored
            assert 1 / (1 + tau) <= ratio <= 1 + tau


@pytest.mark.parametrize(
    ("side", "size", "count"), [(1.0, 0.5, 48), (0.3, 0.1, 144), (0.3, 0.15, 48)]
)
def test_native_box_faces_meet_the_stored_size_bound_without_splitting_ties(
    side: float, size: float, count: int
) -> None:
    # Every box face is commensurate with h, so its right triangles sit at
    # R = h / sqrt(2) up to coordinate rounding (0.49999999999999994 for a
    # face center at h 0.5; 0.15 is not representable): the stored test
    # keeps them (count unchanged from the metric-only kernel) and every
    # boundary triangle meets R^2 <= (h^2 / 2)(1 + tau) exactly. Fault
    # caught: an exact test without the allowance splitting ties (more
    # triangles), or an allowance the route's check does not honor.
    from fractions import Fraction

    from phydrax.meshing._sizing import stored_circumradius_measures

    model = cad_geometry_bundle(
        MeshingDomain.from_brep(
            phx.geometry.brep.brep_box(
                (0.0, 0.0, 0.0),
                (side, side, side),
                coordinate_contract=phx.SpatialCoordinateContract.si(),
            )
        )
    ).model
    solid = cad_mesh_solid(
        model,
        np.full(6, size),
        np.full(6, 0.05),
        np.full(model.curve_nodes.size, 0.05),
        radius_edge_bound=8.0,
        min_dihedral_degrees=0.0,
    )
    assert solid.status is MeshcoreStatus.OK
    points = np.asarray(solid.volume.points, dtype=np.float64)
    triangles = np.asarray(solid.surface.triangles, dtype=np.int64)
    assert triangles.shape[0] == count
    measures = stored_circumradius_measures(points, triangles)
    bound = Fraction(size) ** 2 / 2
    for row in range(triangles.shape[0]):
        value = measures.exact(row)
        assert value is not None and value <= bound, row


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


_CERTIFIED_FIXTURES = {
    "quadric": (_quadric_model, 0.2293),
    "sphere": (_sphere_model, 0.2),
    "cylinder": (
        lambda: phx.geometry.brep.brep_cylinder(
            0.5, 1.0, coordinate_contract=phx.SpatialCoordinateContract.si()
        ),
        0.3,
    ),
    "box": (
        lambda: phx.geometry.brep.brep_box(
            np.zeros(3),
            np.asarray((1.0, 0.5, 0.25)),
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        ),
        0.3,
    ),
}

# Their native trim sheets are unproved, as on the Python route.
_UNPROVED_SHELL_FIXTURES = {
    "cone": (
        lambda: phx.geometry.brep.brep_cone(
            0.5, 0.25, 1.0, coordinate_contract=phx.SpatialCoordinateContract.si()
        ),
        0.3,
    ),
    "torus": (
        lambda: phx.geometry.brep.brep_torus(
            1.0, 0.25, coordinate_contract=phx.SpatialCoordinateContract.si()
        ),
        0.3,
    ),
}


@pytest.mark.parametrize("name", sorted(_CERTIFIED_FIXTURES))
def test_proved_cad_solid_is_the_certified_native_candidate(name: str) -> None:
    from phydrax.meshing._result import CellMeshingResult

    factory, size = _CERTIFIED_FIXTURES[name]
    plan = _cad_volume_plan(factory(), size, 1, radius_edge=8.0, dihedral=0.0)
    result = plan.execute()
    assert isinstance(result, CellMeshingResult)
    assert result.mesh.blocks[0].name == "native-cad-tetrahedra"
    certification = result.certification
    assert certification is not None and certification.passed
    native = certification.native_cad
    assert native is not None and native.status == "certified"
    # V3's certified bound reaches the report against the request tolerance.
    assert native.tolerance == 0.05
    assert native.maximum_deviation is not None
    assert 0.0 <= native.maximum_deviation <= 0.05
    # The source shell and the occurrence records are proved, not premises.
    assert native.source_shell is not None
    records = certification.inputs.native_cad_records
    assert records is not None and native.records_id == records.records_id
    # One construction, then certification: the same candidate and report.
    again = plan.execute(prove=False).prove()
    assert again.mesh.mesh_id == result.mesh.mesh_id
    assert again.certification is not None
    assert again.certification.report_id == certification.report_id


@pytest.mark.parametrize("name", sorted(_UNPROVED_SHELL_FIXTURES))
def test_unproved_source_shell_refuses_the_cad_solid(name: str) -> None:
    from phydrax.meshing._contracts import MeshingFailure

    factory, size = _UNPROVED_SHELL_FIXTURES[name]
    plan = _cad_volume_plan(factory(), size, 1, radius_edge=8.0, dihedral=0.0)
    with pytest.raises(
        MeshingFailure,
        match="Rooted CAD source shell embedding or material orientation remains "
        r"unresolved \(refused at: trim sheet of source patch",
    ):
        plan.execute()


def test_source_shell_is_proved_once_per_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._native_cad_shell as native_shell

    calls: list[MeshingDomain] = []
    prove = native_shell.prove_source_shell

    def counted(domain: MeshingDomain, /) -> native_shell.NativeSourceShell:
        calls.append(domain)
        return prove(domain)

    monkeypatch.setattr(native_shell, "prove_source_shell", counted)
    domain = MeshingDomain.from_brep(_sphere_model())
    first = _cad_volume_plan(domain, 0.3, 1, radius_edge=8.0, dihedral=0.0).execute()
    # Another mesh of the same domain reuses the model part of the proof.
    second = _cad_volume_plan(domain, 0.2, 1, radius_edge=8.0, dihedral=0.0).execute()
    assert calls == [domain]
    shells = []
    for result in (first, second):
        assert result.certification is not None and result.certification.passed
        native = result.certification.native_cad
        assert native is not None and native.source_shell is not None
        shells.append(native.source_shell)
    # The chart part is per mesh.
    assert first.mesh.mesh_id != second.mesh.mesh_id
    assert shells[0] != shells[1]


def test_reversed_source_shell_is_refused() -> None:
    from phydrax.meshing._native_cad_shell import prove_source_shell

    box = _CERTIFIED_FIXTURES["box"][0]()
    assert prove_source_shell(MeshingDomain.from_brep(box)).refusal is None
    # The same faces with every orientation flipped bound an inward shell.
    reversed_box = phx.geometry.brep.BRepModel(
        geometry=box.geometry,
        patches=box.patches,
        parameter_bounds=np.asarray(box.parameter_bounds),
        orientation=-np.asarray(box.orientation),
        physical_tags=box.physical_tags,
        coordinate_contract=box.coordinate_contract,
        report=box.report,
        trim_domains=box.trim_domains,
    )
    shell = prove_source_shell(MeshingDomain.from_brep(reversed_box))
    assert shell.refusal == "shell volume sign"


@pytest.mark.parametrize("request_field", ["protected features", "normal deviation"])
def test_native_cad_certificate_refuses_requests_without_native_owner(
    request_field: str,
) -> None:
    # Neither request reaches the native route today (protected features keep
    # the existing construction; volume specifications have no normal bound);
    # given one, the native certificate refuses it rather than dropping it.
    from phydrax.meshing._controls import FeatureKind, ProtectedFeature
    from phydrax.meshing._native_cad_certificate import NativeCadCertificate
    from phydrax.meshing._scope import MeshingScope

    candidate = _cad_volume_plan(
        _sphere_model(), 0.3, 1, radius_edge=8.0, dihedral=0.0
    ).execute(prove=False)
    request = candidate._construction.certification
    witness = request.native_cad_surface
    domain = request.domain
    records = request.native_cad_records
    assert witness is not None and records is not None
    assert isinstance(domain, MeshingDomain)
    features = (
        (ProtectedFeature(MeshingScope.whole(domain, 2), FeatureKind.SURFACE),)
        if request_field == "protected features"
        else ()
    )
    normal = 0.5 if request_field == "normal deviation" else None
    certificate = NativeCadCertificate(
        candidate.mesh,
        candidate.geometry,
        domain,
        witness,
        records,
        0.05,
        features,
        normal,
    )
    assert certificate.status == "unresolved"
    assert certificate.source_shell is None
    (finding,) = certificate.findings
    assert finding.check == "cad_native_unsupported_request"
    assert (
        "protected features" if features else "maximum_normal_deviation"
    ) in finding.detail


def test_proved_quadratic_cad_solid_is_refused() -> None:
    from phydrax.meshing._contracts import MeshingFailure

    plan = _cad_volume_plan(_sphere_model(), 0.3, 2)
    with pytest.raises(MeshingFailure, match="geometry order 2 CAD solid"):
        plan.execute()
    with pytest.raises(MeshingFailure, match="geometry order 2 CAD solid"):
        plan.execute(prove=False).prove()


@pytest.mark.parametrize("prove_later", [False, True])
def test_flipped_boundary_triangle_is_refused_by_the_chart_partition(
    monkeypatch: pytest.MonkeyPatch, prove_later: bool
) -> None:
    # One boundary triangle reversed in its face chart (as a construction
    # defect would emit it): V4 refuses its orientation and prove() reports it.
    import phydrax.meshing.providers._native_cad_mesh as route
    from phydrax.meshing._contracts import MeshingFailure

    construct = route.construct_cad_solid

    def flipped(prepared: _PreparedDirectBRepCadVolume) -> CadSolidResult:
        solid = construct(prepared)
        chart = solid.surface.chart
        row = int(np.flatnonzero(chart.chart_to_mesh >= 0)[0])
        triangles, uv = chart.triangles.copy(), chart.uv.copy()
        triangles[row, [1, 2]] = triangles[row, [2, 1]]
        uv[row, [1, 2]] = uv[row, [2, 1]]
        surface = solid.surface._replace(chart=chart._replace(triangles=triangles, uv=uv))
        return solid._replace(surface=surface)

    monkeypatch.setattr(route, "construct_cad_solid", flipped)
    plan = _cad_volume_plan(_sphere_model(), 0.3, 1, radius_edge=8.0, dihedral=0.0)
    with pytest.raises(MeshingFailure) as failure:
        plan.execute(prove=False).prove() if prove_later else plan.execute()
    assert "cad_chart_partition: violated" in str(failure.value)
    assert "ORIENTATION" in str(failure.value)


def test_native_cad_certificate_survives_the_archive(tmp_path: Path) -> None:
    import subprocess
    import sys

    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        write_model_artifact,
    )

    result = _cad_volume_plan(
        _sphere_model(), 0.3, 1, radius_edge=8.0, dihedral=0.0
    ).execute()
    path = tmp_path / "native-cad-result.phx"
    write_model_artifact(path, result, limits=MODEL_ARCHIVE_LIMITS)
    content_id = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    assert result.certification is not None
    native = result.certification.native_cad
    assert native is not None
    records = result.certification.inputs.native_cad_records
    assert records is not None and native.records_id == records.records_id
    associations = ",".join(item.association_id for item in result.associations)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value
from phydrax.meshing._result import CellMeshingResult
r = read_model_artifact(sys.argv[1], expected_type=CellMeshingResult,
    restore=restore_model_archive_value, expected_content_id=sys.argv[2], limits=MODEL_ARCHIVE_LIMITS)
assert r.certification.report_id == sys.argv[3]
assert r.certification.native_cad.certificate_id == sys.argv[4]
assert r.certification.inputs.native_cad_records.records_id == sys.argv[5]
assert r.certification.native_cad.records_id == sys.argv[5]
assert ",".join(item.association_id for item in r.associations) == sys.argv[6]
""",
            str(path),
            content_id,
            result.certification.report_id,
            native.certificate_id,
            records.records_id,
            associations,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_restored_domain_proves_a_cad_solid_in_process(tmp_path: Path) -> None:
    # An archive-restored domain holds equal, unshared copies of the source
    # objects; in the same process (bundle cache warm) it proves the solid.
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from phydrax.meshing._result import CellMeshingResult

    result = _cad_volume_plan(
        _sphere_model(), 0.3, 1, radius_edge=8.0, dihedral=0.0
    ).execute()
    path = tmp_path / "native-cad-domain.phx"
    write_model_artifact(path, result, limits=MODEL_ARCHIVE_LIMITS)
    restored = read_model_artifact(
        path,
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        expected_content_id=model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert restored.certification is not None and result.certification is not None
    assert restored.certification.report_id == result.certification.report_id
    domain = restored.certification.inputs.domain
    assert isinstance(domain, MeshingDomain)
    again = _cad_volume_plan(domain, 0.3, 1, radius_edge=8.0, dihedral=0.0).execute()
    assert again.certification is not None and again.certification.passed


def _region_scope(domain: MeshingDomain, /) -> MeshingScope:
    from phydrax.meshing._scope import MeshingEntityKind

    return MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        domain.scope_entity_set_id(3),
        np.asarray(domain.scope_entity_ids(3)[:1], dtype=np.int64),
    )


def _organized_plan(
    model: phx.geometry.brep.BRepModel,
    size: float,
    *,
    region_seed: tuple[float, float, float] | None,
    hole_seed: tuple[float, float, float] | None = None,
    controls: bool = True,
) -> phx.meshing.NativeMeshingPlan:
    """A coarse CAD solid with a patch control, a region control and seeds."""
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
    )
    from phydrax.meshing._controls import (
        HoleSeed,
        PatchControl,
        RegionControl,
        RegionSeed,
    )
    from phydrax.meshing._organization import RegionRole
    from phydrax.meshing._scope import MeshingEntityKind
    from phydrax.meshing._sizing import UniformSizeControl
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_options import NativeMeshingOptions

    domain = MeshingDomain.from_brep(model)
    boundary = MeshingScope.whole(model, 2)
    faces = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        np.asarray(domain.scope_entity_ids(2)[:2], dtype=np.int64),
    )
    region = domain.regions[0].name
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, size),),
        region_controls=(
            (RegionControl(_region_scope(domain), "renamed", "steel", RegionRole.SOLID),)
            if controls
            else ()
        ),
        patch_controls=(
            (PatchControl("inlet", faces, ("renamed",)),) if controls else ()
        ),
        region_seeds=(
            ()
            if region_seed is None
            else (
                RegionSeed(np.asarray(region_seed), region, "seeded", RegionRole.FLUID),
            )
        ),
        hole_seeds=(
            ()
            if hole_seed is None
            else (HoleSeed(np.asarray(hole_seed), MeshingScope.whole(model, 3)),)
        ),
        maximum_deviation=0.05,
    )
    return phx.meshing.NativeMeshingProvider(
        NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
            ),
        )
    ).plan(model, specification)


type _ViewRows = tuple[tuple[object, ...], ...]


def _organization_view(
    patches: tuple[MeshPatch, ...],
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    evidence: tuple[RegionBoundaryEvidence, ...],
) -> tuple[_ViewRows, _ViewRows, _ViewRows, _ViewRows]:
    """Mesh-independent organization semantics (names, adjacency, evidence)."""
    zone_names = {zone.zone_id: zone.name for zone in zones}
    patch_names = {patch.patch_id: patch.name for patch in patches}
    return (
        tuple(
            (
                patch.name,
                patch.scope.entity_ids.size > 0,
                patch.connected,
                tuple(zone_names[zone] for zone in patch.adjacent_zone_ids),
                patch.source_adjacent_region_ids,
                patch.source_adjacent_region_occurrence_paths,
            )
            for patch in patches
        ),
        tuple(
            (zone.name, zone.role, zone.material_id, zone.region_role) for zone in zones
        ),
        tuple(
            sorted(
                (
                    label.name,
                    label.scope.entity_dimension,
                    label.scope.entity_ids.size > 0,
                )
                for label in labels
            )
        ),
        tuple(
            (
                item.source_region_id,
                item.source_scope.scope_id,
                item.control_id,
                item.material_id,
                item.role,
                tuple(
                    sorted((patch_names[patch], side) for patch, side in item.patch_sides)
                ),
                item.source_occurrence_path,
            )
            for item in evidence
        ),
    )


_ORGANIZED_SOLIDS = (
    ("box", 0.3, (0.5, 0.5, 0.5)),
    ("quadric", 0.6, (0.0, 0.0, 0.0)),
)


def _organized_model(name: str, /) -> phx.geometry.brep.BRepModel:
    if name == "quadric":
        return _quadric_model()
    return phx.geometry.brep.brep_box(
        np.zeros(3),
        np.ones(3),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        source_id="native-cad-route-organized-box",
    )


@pytest.mark.parametrize(
    ("name", "size", "seed"), _ORGANIZED_SOLIDS, ids=[row[0] for row in _ORGANIZED_SOLIDS]
)
@pytest.mark.parametrize("prove", [False, True], ids=["candidate", "proved"])
def test_cad_solid_organization_is_native_and_matches_existing_route(
    name: str,
    size: float,
    seed: tuple[float, float, float],
    prove: bool,
    tmp_path: Path,
) -> None:
    # Patch and region controls and region seeds take the native route, whose
    # patches, zones, labels and region evidence carry the existing route's
    # semantics; the proved result certifies and survives the archive.
    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from phydrax.meshing._native_construction import NativeMeshConstruction
    from phydrax.meshing._organization import RegionRole
    from phydrax.meshing._result import CellMeshingResult

    plan = _organized_plan(_organized_model(name), size, region_seed=seed)
    prepared = plan.prepared
    assert isinstance(prepared, _PreparedDirectBRepCadVolume)
    existing = prepared.execute()
    assert isinstance(existing, NativeMeshConstruction)
    assert existing.mesh.blocks[0].name != "native-cad-tetrahedra"
    expected = _organization_view(
        existing.organization.patches,
        existing.organization.zones,
        existing.organization.labels,
        existing.organization.region_boundary_evidence,
    )
    patches, zones, _, evidence = expected
    assert [row[0] for row in patches][-1] == "inlet"
    assert [(row[2], row[3]) for row in zones] == [("steel", RegionRole.SOLID)]
    assert evidence[0][3:5] == ("steel", RegionRole.SOLID)
    if not prove:
        candidate = plan.execute(prove=False)
        assert isinstance(candidate, CellMeshingCandidate)
        assert candidate.mesh.blocks[0].name == "native-cad-tetrahedra"
        organization = candidate._construction.organization
        assert (
            _organization_view(
                organization.patches,
                organization.zones,
                organization.labels,
                organization.region_boundary_evidence,
            )
            == expected
        )
        return
    result = plan.execute()
    assert isinstance(result, CellMeshingResult)
    assert result.mesh.blocks[0].name == "native-cad-tetrahedra"
    assert result.certification is not None and result.certification.passed
    view = _organization_view(
        result.patches, result.zones, result.labels, result.region_boundary_evidence
    )
    assert view == expected
    path = tmp_path / "native-cad-organized.phx"
    write_model_artifact(path, result, limits=MODEL_ARCHIVE_LIMITS)
    restored = read_model_artifact(
        path,
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        expected_content_id=model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert restored.certification is not None
    assert restored.certification.report_id == result.certification.report_id
    assert [patch.patch_id for patch in restored.patches] == [
        patch.patch_id for patch in result.patches
    ]
    assert [zone.zone_id for zone in restored.zones] == [
        zone.zone_id for zone in result.zones
    ]
    assert [item.evidence_id for item in restored.region_boundary_evidence] == [
        item.evidence_id for item in result.region_boundary_evidence
    ]
    assert (
        _organization_view(
            restored.patches,
            restored.zones,
            restored.labels,
            restored.region_boundary_evidence,
        )
        == view
    )


_SEED_OUTCOMES = (
    ("hole_inside", None, (0.5, 0.5, 0.5), "Native CAD hole seed"),
    ("region_outside", (2.0, 2.0, 2.0), None, "lies outside the solid"),
    ("region_on_surface", (0.5, 0.5, 0.0), None, "lies on the certified surface"),
    ("hole_outside", None, (2.0, 2.0, 2.0), None),
)


@pytest.mark.parametrize(
    ("case", "region_seed", "hole_seed", "refusal"),
    _SEED_OUTCOMES,
    ids=[row[0] for row in _SEED_OUTCOMES],
)
def test_cad_solid_seeds_classify_by_the_certified_surface(
    case: str,
    region_seed: tuple[float, float, float] | None,
    hole_seed: tuple[float, float, float] | None,
    refusal: str | None,
) -> None:
    # A region seed must lie inside its region and a hole seed outside the
    # solid (exact point-in-shell test); a seed on a surface triangle refuses.
    from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory

    plan = _organized_plan(
        _organized_model("box"),
        0.3,
        region_seed=region_seed,
        hole_seed=hole_seed,
        controls=False,
    )
    if refusal is None:
        candidate = plan.execute(prove=False)
        assert isinstance(candidate, CellMeshingCandidate)
        assert candidate.mesh.blocks[0].name == "native-cad-tetrahedra"
        return
    with pytest.raises(MeshingFailure, match=re.escape(refusal)) as error:
        plan.execute(prove=False)
    assert error.value.category is MeshingFailureCategory.REGION_RESOLUTION_FAILED


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
    ("region_seeds", "native"),
    ("hole_seeds", "Native CAD hole seed"),
    ("size_maximum", "native"),
    ("size_compliance", "native"),
    ("size_curvature", "supports uniform and proximity size controls on source faces"),
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
def test_unproved_cad_solid_honors_or_refuses_every_field(
    field: str, outcome: str
) -> None:
    # No specification field is silently dropped by the native route: it is
    # honored there (reported size statistics), the plan keeps the existing
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


def _face_scope(model: phx.geometry.brep.BRepModel, row: int) -> MeshingScope:
    """The face scope of one model face row (cad_geometry_bundle's surface row)."""
    from phydrax.meshing._scope import MeshingEntityKind
    from phydrax.meshing.providers._native_cad_volume import _domain_scope_rows

    domain = MeshingDomain.from_brep(model)
    identifiers = np.asarray(domain.scope_entity_ids(2), dtype=np.int64)
    rows = _domain_scope_rows(domain, MeshingScope.whole(model, 2), 2)
    (index,) = np.flatnonzero(rows == row)
    return MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        identifiers[index : index + 1],
    )


def _thin_solid_candidate(
    model: phx.geometry.brep.BRepModel,
    size: float,
    proximity: tuple[ProximitySizeControl, ...] = (),
) -> CellMeshingCandidate:
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        VolumeFillStrategy,
    )
    from phydrax.meshing._sizing import UniformSizeControl
    from phydrax.meshing._volume_generation import NativeVolumeSchedule
    from phydrax.meshing.providers._native_options import NativeMeshingOptions

    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, size), *proximity),
        maximum_deviation=0.01,
    )
    candidate = (
        phx.meshing.NativeMeshingProvider(
            NativeMeshingOptions(
                "cad_tetrahedral",
                volume_schedule=NativeVolumeSchedule(
                    radius_edge_bound=2.0, minimum_dihedral_degrees=15.5
                ),
            )
        )
        .plan(model, specification)
        .execute(prove=False)
    )
    assert isinstance(candidate, CellMeshingCandidate)
    assert candidate.mesh.blocks[0].name == "native-cad-tetrahedra"
    return candidate


def _thin_disk() -> phx.geometry.brep.BRepModel:
    # A unit disk 0.06 thick: face rows 0 (z = 0), 1 (side), 2 (z = 0.06).
    return phx.geometry.brep.brep_cylinder(
        1.0, 0.06, coordinate_contract=phx.SpatialCoordinateContract.si()
    )


@pytest.mark.parametrize("size", [0.4, 0.8])
def test_thin_solid_without_proximity_control_reports_the_thin_region(
    size: float,
) -> None:
    # The request's size is kept: a disk 0.06 thick at h >= 0.4 is thinner
    # than the volume targets allow at h (t >= h / (sqrt(2) sqrt(15)) =
    # 0.073 h / 0.4; _thin_region_findings), so the run is UNRESOLVED and the
    # finding names the two disk faces, the certified thickness (two planes:
    # exactly 0.06) and the thickness h needs (fails if the route refines
    # implicitly or the finding is missing).
    candidate = _thin_solid_candidate(_thin_disk(), size)
    issues = candidate._construction.compliance.issues
    (finding,) = (x for x in issues if x.startswith("input_thin_region"))
    fields = dict(part.split("=") for part in finding.split(":")[1:5])
    assert fields["faces"] == "0,2"
    assert float(fields["thickness_at_most"]) == pytest.approx(0.06, abs=1e-12)
    needed = size / np.sqrt(2.0) / np.sqrt(15.0)
    assert float(fields["thickness_needed"]) == pytest.approx(needed, rel=1e-12)
    assert any(x.startswith("native_unmet_quality_cells:") for x in issues)


@pytest.mark.parametrize("size", [0.4, 0.8])
def test_thin_solid_with_proximity_control_resolves(size: float) -> None:
    # A ProximitySizeControl between the disk faces (one element per gap)
    # sizes them at the local gap 0.06: every cell meets the targets, and
    # the control's rows state its request and the achieved circumradius
    # ratio R / (gap / sqrt(2)) <= 1 on every scoped triangle (fails if the
    # native route ignores the control or reports no rows).
    model = _thin_disk()
    control = ProximitySizeControl(_face_scope(model, 0), _face_scope(model, 2), 1)
    candidate = _thin_solid_candidate(model, size, (control,))
    report = candidate._construction.compliance
    assert report.issues == ()
    requested, achieved = dict(report.requested), dict(report.achieved)
    key = f"proximity:{control.control_id}"
    assert requested[f"{key}:elements_per_gap"] == 1.0
    assert achieved[f"{key}:gap_estimate"] == pytest.approx(0.06, abs=1e-12)
    assert achieved[f"{key}:maximum_circumradius_ratio"] <= 1.0
    assert achieved[f"{key}:minimum_elements_per_gap"] >= 1.0
    assert achieved["unmet_quality_cells"] == 0.0
    points = np.asarray(candidate.mesh.coordinates, dtype=np.float64)
    cells = np.asarray(candidate.mesh.blocks[0].vertices, dtype=np.int64)
    dihedral, ratio = _cell_quality(points, cells)
    assert dihedral.min() >= 15.5 - 1e-9
    assert ratio.max() <= 2.0 + 1e-9


def test_touching_proximity_scopes_need_a_minimum_size() -> None:
    # The cap's disk and sphere meet at the rim, so their gap reaches zero:
    # without minimum_size the control is refused (as Gmsh refuses touching
    # scopes), naming the remedy.
    from phydrax.meshing._contracts import MeshingFailure

    model = _unit_cap(70.0, above=True)
    control = ProximitySizeControl(_face_scope(model, 0), _face_scope(model, 1), 1)
    with pytest.raises(MeshingFailure, match="declare minimum_size"):
        _thin_solid_candidate(model, 0.4, (control,))
