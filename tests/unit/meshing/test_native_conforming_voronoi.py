"""Conforming Voronoi meshing: the native kernel and the plc_conforming_voronoi route."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.spatial import Voronoi

import phydrax._meshcore as meshcore
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import PolyhedralConnectivity
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    MeshingScope,
    SizeControlStrength,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
)
from phydrax.meshing.providers._native_sources import NativePlcSource


_QUADS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def _cube_vertices() -> np.ndarray:
    return np.asarray(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=np.float64,
    )


def _cube_triangles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(
        [(a, b, c) for a, b, c, d in _QUADS] + [(a, c, d) for a, b, c, d in _QUADS],
        dtype=np.int32,
    )
    facets = np.asarray(list(range(6)) * 2, dtype=np.int32)
    regions = np.tile(np.asarray([[-1, 0]], dtype=np.int32), (12, 1))
    return triangles, regions, facets


def _kernel_cube(size: float, *, prove: bool) -> meshcore.ConformingVoronoi:
    triangles, regions, facets = _cube_triangles()
    return meshcore.conforming_voronoi(
        _cube_vertices(),
        triangles,
        regions,
        facets,
        size=size,
        feature_angle_degrees=20.0,
        prove=prove,
    )


def _qhull_volumes(seeds: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Cell volumes of the Voronoi diagram of all seeds (Qhull) by pyramids."""
    diagram = Voronoi(seeds)
    volumes = np.zeros((seeds.shape[0],), dtype=np.float64)
    for (first, second), ridge in zip(
        diagram.ridge_points, diagram.ridge_vertices, strict=True
    ):
        if -1 in ridge or not (interior[first] or interior[second]):
            continue
        polygon = diagram.vertices[ridge]
        area = 0.5 * np.linalg.norm(
            np.sum(np.cross(polygon[1:-1] - polygon[0], polygon[2:] - polygon[0]), axis=0)
        )
        height = 0.5 * np.linalg.norm(seeds[first] - seeds[second])
        volumes[first] += area * height / 3.0
        volumes[second] += area * height / 3.0
    return volumes


def test_kernel_cells_are_the_voronoi_cells_of_all_seeds() -> None:
    cells = _kernel_cube(0.3, prove=True)

    assert cells.status is meshcore.MeshcoreStatus.OK
    stats = cells.stats
    assert stats[15] == 1.0 and stats[16] == 0.0 and stats[17] == 0.0
    interior = cells.seed_regions >= 0
    assert np.all(interior[cells.cell_seeds])
    reference = _qhull_volumes(cells.seeds, interior)[cells.cell_seeds]
    np.testing.assert_allclose(cells.cell_volumes, reference, rtol=0.0, atol=1e-13)
    # Mirrored seeds put the boundary exactly on the cube.
    assert abs(float(np.sum(cells.cell_volumes)) - 1.0) <= 1e-13
    exterior = np.flatnonzero(cells.face_cells[:, 1] < 0)
    entries = np.concatenate(
        [
            cells.face_vertices[cells.face_offsets[face] : cells.face_offsets[face + 1]]
            for face in exterior
        ]
    )
    on_cube = np.any(
        (cells.vertices[entries] == 0.0) | (cells.vertices[entries] == 1.0), axis=1
    )
    assert np.all(on_cube)
    assert np.all(cells.face_facets[exterior] >= 0)


def test_kernel_cells_are_convex_and_deterministic() -> None:
    first = _kernel_cube(0.25, prove=False)
    second = _kernel_cube(0.25, prove=False)

    for name in ("vertices", "face_vertices", "face_cells", "seeds", "exact_values"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    seeds = first.seeds
    vertices = first.vertices
    for face, (owner, neighbor) in enumerate(first.face_cells.tolist()):
        if neighbor < 0:
            continue
        # Every vertex of the owner lies on its side of the bisector.
        a, b = seeds[first.cell_seeds[owner]], seeds[first.cell_seeds[neighbor]]
        rows = np.flatnonzero(first.face_cells[:, 0] == owner)
        loops = np.concatenate(
            [
                first.face_vertices[first.face_offsets[f] : first.face_offsets[f + 1]]
                for f in rows
            ]
        )
        heights = (vertices[loops] - 0.5 * (a + b)) @ (b - a) / np.linalg.norm(b - a)
        assert np.max(heights) <= 1e-12, face


def _rotation() -> np.ndarray:
    rotation = np.eye(3)
    for axis, angle in enumerate((0.3, 0.5, 0.7)):
        i, j = (k for k in range(3) if k != axis)
        turn = np.eye(3)
        turn[i, i] = turn[j, j] = np.cos(angle)
        turn[i, j], turn[j, i] = -np.sin(angle), np.sin(angle)
        rotation = rotation @ turn
    return rotation


def _shear() -> np.ndarray:
    shear = np.eye(3)
    shear[0, 2] = 0.3
    return shear


@pytest.mark.parametrize("transform", [_rotation, _shear], ids=["rotated", "sheared"])
def test_kernel_resolves_oblique_facets_within_the_certified_bounds(
    transform: Callable[[], np.ndarray],
) -> None:
    # Oblique facets (every facet of a rotated cube, two of a sheared one):
    # snapped mirror reflections leave snap-level vertex clusters that rigid
    # (mirror-preserving) seed moves keep; lone moves resolve them, so the
    # boundary is certified within its fidelity bound instead of exact
    # (measured: distance ~4e-3 h, volume error ~5e-4 against 1e-15 on the
    # axis-aligned cube).
    triangles, regions, facets = _cube_triangles()
    for size in (0.25, 0.1):
        cells = meshcore.conforming_voronoi(
            _cube_vertices() @ transform().T,
            triangles,
            regions,
            facets,
            size=size,
            feature_angle_degrees=20.0,
            prove=True,
        )
        stats = cells.stats
        assert cells.status is meshcore.MeshcoreStatus.OK
        assert stats[15] == 1.0 and stats[16] == 0.0 and stats[17] == 0.0
        assert stats[14] > 0.0 and stats[24] == 0.0
        assert 0.0 < stats[19] <= min(stats[20], 1e-2)
        assert abs(stats[18]) <= min(stats[25], 1e-3)


def _plc_cube() -> NativePlcSource:
    vertices = _cube_vertices()
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(np.asarray(quad, dtype=np.int64) for quad in _QUADS),
        np.arange(6, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        ("solid",),
        vertex_ids=tuple(f"v{row}" for row in range(8)),
        vertex_occurrence_paths=((),) * 8,
        facet_ids=tuple(f"f{row}" for row in range(6)),
        facet_occurrence_paths=((),) * 6,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    return NativePlcSource(complex_, "voronoi-cube", "r1")


def _specification(
    source: NativePlcSource, size: float, *, maximum_deviation: float | None = None
) -> VolumeMeshingSpec:
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            UniformSizeControl(
                MeshingScope.whole(source, 3), size, strength=SizeControlStrength.SOFT
            ),
        ),
        maximum_deviation=maximum_deviation,
    )


def test_route_publishes_closed_polyhedra_filling_the_cube() -> None:
    source = _plc_cube()
    plan = NativeMeshingProvider(NativeMeshingOptions("plc_conforming_voronoi")).plan(
        source,
        _specification(source, 0.3),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    assert plan.route == "plc_conforming_voronoi"
    candidate = plan.execute(prove=False)

    connectivity = candidate.mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    assert {block.cell_kind for block in candidate.mesh.blocks} == {"polyhedron"}
    points = np.asarray(candidate.mesh.coordinates, dtype=np.float64)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    # Every boundary face lies on a cube face.
    for face in np.flatnonzero(neighbor < 0):
        loop = points[values[offsets[face] : offsets[face + 1]]]
        assert np.any(np.all(loop == loop[0], axis=0) & np.isin(loop[0], (0.0, 1.0)))
    again = plan.execute(prove=False)
    np.testing.assert_array_equal(np.asarray(again.mesh.coordinates), points)


def test_route_proves_a_small_cube() -> None:
    source = _plc_cube()
    result = (
        NativeMeshingProvider(NativeMeshingOptions("plc_conforming_voronoi"))
        .plan(
            source,
            _specification(source, 0.45),
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        .execute()
    )

    certification = result.certification
    assert certification is not None and certification.passed
    # Axis-aligned facets: the boundary is exact and coverage is certified.
    assert certification.inputs.schedule.route == "volume_plc"
    assert certification.coverage is not None
    assert dict(result.compliance.achieved)["voronoi:exact_boundary"] == 1.0
    names = {attribute.name for attribute in result.attributes}
    assert {"voronoi_seed_index", "voronoi_seed"} <= names


def _plc_sphere(levels: int) -> NativePlcSource:
    """A unit icosphere: every facet off the axis planes."""
    t = (1.0 + 5.0**0.5) / 2.0
    points = [
        np.asarray(p, dtype=np.float64) / np.linalg.norm(p)
        for p in (
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        )
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    for _ in range(levels):
        middles: dict[tuple[int, int], int] = {}

        def middle(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in middles:
                point = points[a] + points[b]
                points.append(point / np.linalg.norm(point))
                middles[key] = len(points) - 1
            return middles[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    count = len(faces)
    complex_ = PiecewiseLinearComplex(
        np.asarray(points, dtype=np.float64),
        tuple(np.asarray(face, dtype=np.int64) for face in faces),
        np.arange(count, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (count, 1)),
        ("solid",),
        vertex_ids=tuple(f"v{row}" for row in range(len(points))),
        vertex_occurrence_paths=((),) * len(points),
        facet_ids=tuple(f"f{row}" for row in range(count)),
        facet_occurrence_paths=((),) * count,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    return NativePlcSource(complex_, "voronoi-sphere", "r1")


def _sphere_plan(maximum_deviation: float | None) -> NativeMeshingPlan:
    source = _plc_sphere(2)
    specification = _specification(source, 0.5, maximum_deviation=maximum_deviation)
    return NativeMeshingProvider(NativeMeshingOptions("plc_conforming_voronoi")).plan(
        source, specification, coordinate_contract=SpatialCoordinateContract.si()
    )


def test_route_certifies_a_curved_plc_under_its_fidelity_contract() -> None:
    result = _sphere_plan(0.05).execute()

    certification = result.certification
    assert certification is not None and certification.passed
    # The bisector boundary approximates the sphere: a certified two-sided
    # distance bound replaces exact coverage.
    assert certification.inputs.schedule.route == "volume_plc_fidelity"
    assert certification.coverage is None
    fidelity = certification.fidelity
    assert fidelity is not None and fidelity.status == "certified"
    assert 0.0 < fidelity.mesh_to_source_lower <= fidelity.mesh_to_source_upper <= 0.05
    assert 0.0 < fidelity.source_to_mesh_upper <= 0.05
    achieved = dict(result.compliance.achieved)
    assert achieved["voronoi:exact_boundary"] == 0.0
    assert achieved["voronoi:volume_error_bound"] >= abs(achieved["voronoi:volume_error"])


def test_route_refuses_a_tolerance_below_the_certified_bound() -> None:
    with pytest.raises(MeshingFailure, match="source_fidelity_tolerance"):
        _sphere_plan(1e-4).execute()


def test_polyhedral_fill_of_a_plc_still_resolves_to_restricted_power() -> None:
    source = _plc_cube()
    plan = NativeMeshingProvider().plan(
        source,
        _specification(source, 0.5),
        coordinate_contract=SpatialCoordinateContract.si(),
    )

    assert plan.route == "plc_restricted_power"


def test_route_requires_a_size_control() -> None:
    source = _plc_cube()
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
    )
    with pytest.raises(MeshingFailure, match="uniform size control"):
        NativeMeshingProvider(NativeMeshingOptions("plc_conforming_voronoi")).plan(
            source, specification, coordinate_contract=SpatialCoordinateContract.si()
        )
