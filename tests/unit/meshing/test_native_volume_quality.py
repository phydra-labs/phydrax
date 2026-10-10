from collections import Counter
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    delaunay_3d,
    exact_orient3d,
    meshcore_available,
    MeshcoreStatus,
    TET_MESH_UNMET_REASONS,
    TetMesh3D,
    TetMeshArrays,
    TetMeshBoundaryPolicy,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)

# Box domains on [0, 1]^3 with dyadic coordinates, so exact midpoints and
# in-plane circumcenters of the axis-aligned constraints are representable.
# Oracles are independent of the mesher: coordinates on the side planes,
# exact orientation, analytic volumes/areas and numpy circumballs.
_INTERFACE = 6


def _box_points(per_side: int, interior: int, seed: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, per_side)
    lattice = np.asarray(list(product(axis, axis, axis)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    extra = rng.integers(1, 1023, size=(interior, 3)).astype(np.float64) / 1024.0
    return np.concatenate((lattice, extra))


def _side(corners: np.ndarray, split: float | None) -> int:
    for axis in range(3):
        values = corners[:, axis]
        if np.all(values == values[0]):
            if values[0] in (0.0, 1.0):
                return 2 * axis + int(values[0])
            if axis == 0 and values[0] == split:
                return _INTERFACE
    return -1


def _box_domain(points: np.ndarray, split: float | None = None) -> tuple[np.ndarray, ...]:
    cells, _ = delaunay_3d(points)
    centroid = points[cells].mean(axis=1)[:, 0]
    regions = (
        np.zeros(len(cells), dtype=np.int32)
        if split is None
        else (centroid > split).astype(np.int32)
    )
    owners: dict[tuple[int, ...], list[int]] = {}
    for cell, row in enumerate(cells):
        for opposite in range(4):
            key = tuple(sorted(np.delete(row, opposite).tolist()))
            owners.setdefault(key, []).append(cell)
    faces, sources = [], []
    for key, touching in owners.items():
        if len(touching) == 1 or regions[touching[0]] != regions[touching[1]]:
            faces.append(key)
            sources.append(_side(points[list(key)], split))
    edges: dict[tuple[int, int], list[int]] = {}
    for key, source in zip(faces, sources, strict=True):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            edges.setdefault((key[a], key[b]), []).append(source)
    segments = [
        edge
        for edge, incident in edges.items()
        if len(incident) != 2 or incident[0] != incident[1]
    ]
    return (
        points,
        cells,
        regions,
        np.asarray(faces, dtype=np.int32),
        np.asarray(sources, dtype=np.int32),
        np.asarray(segments, dtype=np.int32).reshape(-1, 2),
        np.arange(len(segments), dtype=np.int32),
    )


def _mesh(
    domain: tuple[np.ndarray, ...],
    policy: TetMeshBoundaryPolicy = "conforming",
    max_vertices: int | None = None,
) -> TetMesh3D:
    points, cells, regions, faces, sources, segments, labels = domain
    return TetMesh3D(
        points,
        cells,
        regions,
        faces,
        sources,
        segments,
        labels,
        boundary_policy=policy,
        max_vertices=max_vertices,
    )


def _volumes(arrays: TetMeshArrays) -> np.ndarray:
    corners = arrays.points[arrays.tetrahedra]
    return np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0


def _circumballs(arrays: TetMeshArrays) -> tuple[np.ndarray, np.ndarray]:
    corners = arrays.points[arrays.tetrahedra]
    edges = corners[:, 1:] - corners[:, :1]
    offsets = np.linalg.solve(edges, 0.5 * np.sum(edges * edges, axis=2)[..., None])[
        ..., 0
    ]
    return corners[:, 0] + offsets, np.linalg.norm(offsets, axis=1)


def _radius_edge(arrays: TetMeshArrays) -> np.ndarray:
    corners = arrays.points[arrays.tetrahedra]
    pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    lengths = np.linalg.norm(corners[:, pairs[:, 0]] - corners[:, pairs[:, 1]], axis=2)
    return _circumballs(arrays)[1] / lengths.min(axis=1)


def _areas(arrays: TetMeshArrays) -> dict[int, float]:
    corners = arrays.points[arrays.faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = 0.5 * np.linalg.norm(normals, axis=1)
    return {
        int(source): float(area[arrays.face_sources == source].sum())
        for source in np.unique(arrays.face_sources)
    }


def _assert_exact_domain(arrays: TetMeshArrays, split: float | None = None) -> None:
    corners = arrays.points[arrays.tetrahedra]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )
    sides = [_side(arrays.points[face], split) for face in arrays.faces]
    np.testing.assert_array_equal(sides, arrays.face_sources)
    assert _volumes(arrays).sum() == pytest.approx(1.0, abs=1e-12)


def _minimum_dihedral_degrees(arrays: TetMeshArrays) -> float:
    used = np.unique(arrays.tetrahedra)
    renumber = np.full(arrays.points.shape[0], -1, dtype=np.int32)
    renumber[used] = np.arange(used.size, dtype=np.int32)
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        arrays.points[used], renumber[arrays.tetrahedra]
    )
    angles = np.asarray(phx.meshing.evaluate_cell_quality(mesh).minimum_angle)
    return float(np.degrees(angles.min()))


def test_refined_cube_point_set_meets_the_radius_edge_bound() -> None:
    domain = _box_domain(_box_points(3, 40, 7))
    mesh = _mesh(domain)
    assert _radius_edge(mesh.arrays()).max() > 2.0

    run = mesh.refine(radius_edge_bound=2.0)

    arrays = mesh.arrays()
    assert run.status == MeshcoreStatus.OK
    assert _radius_edge(arrays).max() <= 2.0 + 1e-9
    _assert_exact_domain(arrays)
    assert _areas(arrays) == pytest.approx({side: 1.0 for side in range(6)}, abs=1e-12)
    assert mesh.unmet().tetrahedra.shape == (0, 4)


def test_refinement_is_deterministic() -> None:
    domain = _box_domain(_box_points(3, 30, 11))
    first, second = _mesh(domain), _mesh(domain)
    first.refine(radius_edge_bound=1.8)
    second.refine(radius_edge_bound=1.8)
    left, right = first.arrays(), second.arrays()
    for name in ("points", "tetrahedra", "faces", "segments", "vertex_dimension"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_two_region_box_keeps_its_interface_and_region_volumes() -> None:
    # A lattice keeps every Delaunay cell inside one lattice cube, so the
    # plane x = 0.5 is a union of faces of the initial tetrahedralization.
    domain = _box_domain(_box_points(5, 0, 3), split=0.5)
    mesh = _mesh(domain)

    run = mesh.refine(radius_edge_bound=2.0, sizes=np.full(domain[0].shape[0], 0.3))

    arrays = mesh.arrays()
    assert run.status == MeshcoreStatus.OK
    _assert_exact_domain(arrays, split=0.5)
    volumes = _volumes(arrays)
    for region in (0, 1):
        assert volumes[arrays.tetrahedron_regions == region].sum() == pytest.approx(
            0.5, abs=1e-12
        )
    x = arrays.points[arrays.tetrahedra][..., 0]
    assert np.all(x[arrays.tetrahedron_regions == 0] <= 0.5)
    assert np.all(x[arrays.tetrahedron_regions == 1] >= 0.5)
    assert _areas(arrays)[_INTERFACE] == pytest.approx(1.0, abs=1e-12)


def test_analytic_size_field_is_sampled_in_batches_of_new_vertices() -> None:
    domain = _box_domain(_box_points(3, 0, 1))
    mesh = _mesh(domain)
    batches: list[int] = []

    def size(points: np.ndarray) -> np.ndarray:
        batches.append(points.shape[0])
        return 0.12 + 0.25 * points[:, 0]

    run = mesh.refine(sizes=size, max_rounds=12)

    arrays = mesh.arrays()
    assert run.status == MeshcoreStatus.OK
    assert batches[0] == domain[0].shape[0]
    assert sum(batches) == arrays.points.shape[0]
    np.testing.assert_allclose(arrays.vertex_sizes, size(arrays.points), rtol=0, atol=0)
    radius = _circumballs(arrays)[1]
    target = arrays.vertex_sizes[arrays.tetrahedra].mean(axis=1)
    assert np.all(radius <= target * (1.0 + 1e-12))
    _assert_exact_domain(arrays)


def test_improvement_raises_the_smallest_dihedral_angle_to_the_target() -> None:
    domain = _box_domain(_box_points(3, 60, 13))
    mesh = _mesh(domain)
    mesh.refine(radius_edge_bound=2.0)
    refined = mesh.arrays()
    assert _minimum_dihedral_degrees(refined) < 15.0

    run = mesh.improve(min_dihedral_degrees=15.0, max_passes=20)

    improved = mesh.arrays()
    quality = mesh.quality(sliver_degrees=15.0)
    assert run.status == MeshcoreStatus.OK
    assert _minimum_dihedral_degrees(improved) >= 15.0
    assert quality.minimum_dihedral == pytest.approx(
        _minimum_dihedral_degrees(improved), abs=1e-9
    )
    assert quality.slivers == 0
    assert quality.dihedral_histogram.sum() == improved.tetrahedra.shape[0]
    np.testing.assert_array_equal(improved.segments, refined.segments)
    _assert_exact_domain(improved)
    assert _areas(improved) == pytest.approx(_areas(refined), abs=1e-12)


def test_fixed_boundary_is_never_split_and_blocked_cells_are_reported() -> None:
    domain = _box_domain(_box_points(3, 30, 5))
    mesh = _mesh(domain, "fixed")
    before = mesh.arrays()

    run = mesh.refine(radius_edge_bound=1.2)

    after = mesh.arrays()
    unmet = mesh.unmet()
    assert run.status == MeshcoreStatus.REFINEMENT_LIMIT
    np.testing.assert_array_equal(after.faces, before.faces)
    assert np.all(after.vertex_dimension[before.points.shape[0] :] == 3)
    reasons = Counter(TET_MESH_UNMET_REASONS[reason] for reason in unmet.reasons)
    assert reasons["fixed_boundary"] > 0
    np.testing.assert_allclose(
        unmet.values, _radius_edge(after)[_rows_of(after, unmet.tetrahedra)], rtol=1e-9
    )
    _assert_exact_domain(after)


def _rows_of(arrays: TetMeshArrays, cells: np.ndarray) -> np.ndarray:
    index = {
        tuple(row): position for position, row in enumerate(arrays.tetrahedra.tolist())
    }
    return np.asarray([index[tuple(row)] for row in cells.tolist()], dtype=np.int64)


def test_budgets_stop_refinement_with_a_valid_mesh() -> None:
    domain = _box_domain(_box_points(3, 40, 7))
    mesh = _mesh(domain)

    limited = mesh.refine(max_insertions=3)
    assert limited.status == MeshcoreStatus.REFINEMENT_LIMIT
    assert mesh.arrays().points.shape[0] == domain[0].shape[0] + 3
    assert set(TET_MESH_UNMET_REASONS[r] for r in mesh.unmet().reasons) == {"budget"}

    starved = mesh.refine(work_limit=10)
    assert starved.status == MeshcoreStatus.CAPACITY_EXCEEDED
    _assert_exact_domain(mesh.arrays())

    assert mesh.refine().status == MeshcoreStatus.OK


def test_vertex_capacity_stops_refinement() -> None:
    domain = _box_domain(_box_points(3, 40, 7))
    mesh = _mesh(domain, max_vertices=domain[0].shape[0] + 4)

    run = mesh.refine()

    assert run.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert mesh.arrays().points.shape[0] == domain[0].shape[0] + 4
    _assert_exact_domain(mesh.arrays())


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param("open_boundary", id="open-boundary"),
        pytest.param("missing_segments", id="missing-segments"),
        pytest.param("inverted_cell", id="inverted-cell"),
    ],
)
def test_invalid_domains_are_refused(damage: str) -> None:
    points, cells, regions, faces, sources, segments, labels = _box_domain(
        _box_points(3, 5, 2)
    )
    match damage:
        case "open_boundary":
            faces, sources = faces[1:], sources[1:]
        case "missing_segments":
            segments, labels = segments[:0], labels[:0]
        case "inverted_cell":
            cells = cells.copy()
            cells[0, [0, 1]] = cells[0, [1, 0]]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        TetMesh3D(
            points,
            cells,
            regions,
            faces,
            sources,
            segments,
            labels,
            boundary_policy="fixed",
        )


def test_unknown_boundary_policy_is_refused() -> None:
    with pytest.raises(ValueError, match="boundary policy"):
        _mesh(_box_domain(_box_points(3, 5, 2)), "sliding")  # ty: ignore[invalid-argument-type]


@pytest.fixture
def bipyramid_mesh() -> TetMesh3D:
    points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.25, 0.25, 1), (0.25, 0.25, -1)),
        dtype=np.float64,
    )
    hull = np.asarray(((0, 1, 3), (1, 2, 3), (0, 3, 2), (0, 4, 1), (1, 4, 2), (0, 2, 4)))
    edges = np.asarray(
        ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4))
    )
    return TetMesh3D(
        points,
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
        np.zeros(2, dtype=np.int32),
        hull,
        np.arange(6, dtype=np.int32),
        edges,
        np.arange(9, dtype=np.int32),
        boundary_policy="fixed",
    )


def test_face_flip_and_edge_removal_are_inverse_operations(
    bipyramid_mesh: TetMesh3D,
) -> None:
    mesh = bipyramid_mesh
    original = mesh.arrays().tetrahedra

    assert mesh.flip_face((0, 1, 2))
    flipped = mesh.arrays()
    assert flipped.tetrahedra.shape == (3, 4)
    rows = flipped.tetrahedra
    assert np.all((rows == 3).any(axis=1) & (rows == 4).any(axis=1))
    assert _volumes(flipped).sum() == pytest.approx(1.0 / 3.0, abs=1e-15)

    assert mesh.remove_edge(3, 4)
    np.testing.assert_array_equal(mesh.arrays().tetrahedra, original)
    assert not mesh.flip_face((0, 1, 3))
    assert not mesh.remove_edge(0, 1)


def test_directional_collapse_reverses_interior_split_without_boundary_drift(
    bipyramid_mesh: TetMesh3D,
) -> None:
    mesh = bipyramid_mesh
    assert mesh.flip_face((0, 1, 2))
    before = mesh.arrays()
    inserted = mesh.split_edge(3, 4, np.asarray((0.25, 0.25, 0.0), dtype=np.float64))
    assert inserted == 5
    assert mesh.collapse_edge(inserted, 3)
    after = mesh.arrays()
    np.testing.assert_array_equal(after.tetrahedra, before.tetrahedra)
    np.testing.assert_array_equal(after.faces, before.faces)
    np.testing.assert_array_equal(after.segments, before.segments)
    assert after.vertex_dimension[inserted] == -1
    assert not mesh.collapse_edge(0, 1)


def test_fixed_source_edge_split_refuses_atomically(bipyramid_mesh: TetMesh3D) -> None:
    mesh = bipyramid_mesh
    before = mesh.arrays()
    assert mesh.split_edge(0, 1, np.asarray((0.5, 0.0, 0.0), dtype=np.float64)) is None
    after = mesh.arrays()
    np.testing.assert_array_equal(after.points, before.points)
    np.testing.assert_array_equal(after.tetrahedra, before.tetrahedra)
    np.testing.assert_array_equal(after.faces, before.faces)
    np.testing.assert_array_equal(after.segments, before.segments)


def test_scheduled_curve_relocation_preserves_exact_original_source() -> None:
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    segments = np.asarray(
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32
    )
    mesh = TetMesh3D(
        points,
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        segments,
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
        max_vertices=32,
        max_tetrahedra=128,
    )
    mesh.protect_vertices(np.arange(4, dtype=np.int32))
    inserted = mesh.split_edge(
        0,
        1,
        np.asarray((0.5, 0.0, 0.0)),
        work_limit=1_000_000,
    )
    assert inserted == 4
    before = mesh.arrays()
    run = mesh.improve(
        min_dihedral_degrees=70.0,
        minimum_relative_determinant=0.0,
        max_passes=1,
        work_limit=1_000_000,
    )
    after = mesh.arrays()
    assert run.status == MeshcoreStatus.REFINEMENT_LIMIT
    assert run.counters[2] > 0
    assert after.vertex_dimension[inserted] == 1
    assert 0.0 < after.points[inserted, 0] < 1.0
    assert after.points[inserted, 0] != 0.5
    np.testing.assert_array_equal(after.points[inserted, 1:], (0.0, 0.0))
    np.testing.assert_array_equal(after.points[:4], points)
    assert _minimum_dihedral_degrees(after) > _minimum_dihedral_degrees(before)
    np.testing.assert_array_equal(after.segments, before.segments)
    np.testing.assert_array_equal(after.segment_sources, before.segment_sources)
    np.testing.assert_array_equal(after.tetrahedron_regions, before.tetrahedron_regions)
    corners = after.points[after.tetrahedra]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )
    assert _volumes(after).sum() == pytest.approx(1.0 / 6.0, abs=1e-15)
    assert _areas(after) == pytest.approx(_areas(before), abs=1e-15)
    for face, source in zip(after.faces, after.face_sources, strict=True):
        original = points[faces[source]]
        candidate = after.points[face]
        assert np.all(
            exact_orient3d(
                np.broadcast_to(original[0], candidate.shape),
                np.broadcast_to(original[1], candidate.shape),
                np.broadcast_to(original[2], candidate.shape),
                candidate,
            )
            == 0
        )
    evidence = mesh.source_evidence()
    assert evidence.achieved_bound == 0.0
    assert evidence.witness_strata[inserted] == 1
    assert evidence.witness_entities[inserted] == 0
    assert evidence.witness_deviations[inserted] == 0.0
    parameter = Fraction(float(evidence.witness_parameters[inserted, 0]))
    for axis in range(3):
        expected = (1 - parameter) * Fraction(float(points[0, axis]))
        expected += parameter * Fraction(float(points[1, axis]))
        assert Fraction(float(after.points[inserted, axis])) == expected
    for edge, source, ancestor in zip(
        after.segments,
        after.segment_sources,
        evidence.segment_ancestors,
        strict=True,
    ):
        assert ancestor == source
        a, b = points[segments[source]]
        for vertex in after.points[edge]:
            assert np.array_equal(np.cross(b - a, vertex - a), np.zeros(3))
    assert mesh.unmet().tetrahedra.shape[0] > 0


def test_rotated_decimal_edge_split_retains_exact_source_planes() -> None:
    points = np.asarray(
        ((-1.6, 1.4, 0.9), (0.6, 0.5, -0.3), (0.2, -1.0, 1.0), (1.0, 2.0, 2.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    if exact_orient3d(points[0], points[1], points[2], points[3]) < 0:
        cells[0, [0, 1]] = cells[0, [1, 0]]
    faces = np.asarray(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)), dtype=np.int32)
    edges = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32)
    mesh = TetMesh3D(
        points,
        cells,
        np.zeros(1, dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        edges,
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
    )
    construction = mesh.edge_split_point(0, 1)
    assert construction is not None
    position, parameter = construction.position, construction.parameter
    for axis in range(3):
        exact = (1 - Fraction(parameter)) * Fraction(float(points[0, axis])) + Fraction(
            parameter
        ) * Fraction(float(points[1, axis]))
        assert Fraction(float(position[axis])) == exact
    inserted = mesh.split_edge(0, 1, position, source_fraction=parameter)
    assert inserted == 4
    after = mesh.arrays()
    np.testing.assert_array_equal(after.points[:4], points)
    for face, source in zip(after.faces, after.face_sources, strict=True):
        original = points[faces[source]]
        candidate = after.points[face]
        assert np.all(
            exact_orient3d(
                np.broadcast_to(original[0], candidate.shape),
                np.broadcast_to(original[1], candidate.shape),
                np.broadcast_to(original[2], candidate.shape),
                candidate,
            )
            == 0
        )
    assert _volumes(after).sum() == pytest.approx(
        np.linalg.det(points[cells[0, 1:]] - points[cells[0, 0]]) / 6.0, abs=1e-14
    )


def test_protected_interior_point_rejects_motion_and_collapse(
    bipyramid_mesh: TetMesh3D,
) -> None:
    mesh = bipyramid_mesh
    assert mesh.flip_face((0, 1, 2))
    inserted = mesh.split_edge(3, 4, np.asarray((0.25, 0.25, 0.0), dtype=np.float64))
    assert inserted == 5
    mesh.protect_vertices((inserted,))
    before = mesh.arrays()
    assert not mesh.relocate(inserted, np.asarray((0.25, 0.25, 0.125), dtype=np.float64))
    assert not mesh.collapse_edge(inserted, 3)
    after = mesh.arrays()
    np.testing.assert_array_equal(after.points, before.points)
    np.testing.assert_array_equal(after.tetrahedra, before.tetrahedra)
    assert after.vertex_dimension[inserted] == 0
