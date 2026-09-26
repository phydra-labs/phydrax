#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellMesh
from phydrax.meshing import (
    BisectionCompatibility,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MeshingEntityKind,
    MeshingScope,
    prepare_mesh_adaptation,
)


def _signed_measures(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    corners = np.asarray(points)[cells]
    edges = corners[:, 1:] - corners[:, :1]
    if cells.shape[1] == 3:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    return np.linalg.det(edges) / 6.0


def _triangle_grid(columns: int, rows: int, height: float = 1.0) -> CellMesh:
    xs = np.linspace(0.0, 1.0, columns + 1)
    ys = np.linspace(0.0, height, rows + 1)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape((-1, 2))
    cells = []
    for j, i in itertools.product(range(rows), range(columns)):
        a = j * (columns + 1) + i
        b, c, d = a + 1, a + columns + 2, a + columns + 1
        cells.extend(((a, b, c), (a, c, d)))
    return CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))


def _kuhn_grid(size: int) -> CellMesh:
    axis = np.linspace(0.0, 1.0, size + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape((-1, 3))
    cells = []
    for origin in itertools.product(range(size), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = list(origin)
            path = [(corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2]]
            for direction in order:
                corner[direction] += 1
                path.append((corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2])
            cells.append(path)
    cells = np.asarray(cells, dtype=np.int32)
    negative = _signed_measures(points, cells) < 0.0
    cells[negative] = cells[negative][:, [1, 0, 2, 3]]
    return CellMesh.from_tetrahedra(points, cells)


def _certified(mesh: CellMesh):
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _adapt(source, refine=(), coarsen=(), /, *, hierarchy=None, **options):
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION, **options)
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _cells(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks))


def _cell_ids(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks))


def _edge_keys(mesh: CellMesh) -> np.ndarray:
    edges = np.asarray(mesh.connectivity.edges)
    return np.sort(np.asarray(mesh.vertex_global_ids)[edges], axis=1)


def _minimum_angles(mesh: CellMesh) -> np.ndarray:
    corners = np.asarray(mesh.coordinates)[_cells(mesh)]
    angles = []
    for vertex in range(3):
        first = corners[:, (vertex + 1) % 3] - corners[:, vertex]
        second = corners[:, (vertex + 2) % 3] - corners[:, vertex]
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.min(np.stack(angles, axis=1), axis=1)


def _mean_ratios(mesh: CellMesh) -> np.ndarray:
    points = np.asarray(mesh.coordinates)
    cells = _cells(mesh)
    pairs = np.asarray(tuple(itertools.combinations(range(4), 2)))
    corners = points[cells]
    squared = np.sum((corners[:, pairs[:, 0]] - corners[:, pairs[:, 1]]) ** 2, axis=2)
    volumes = np.abs(_signed_measures(points, cells))
    return 12.0 * (3.0 * volumes) ** (2.0 / 3.0) / np.sum(squared, axis=1)


def _assert_conforming(mesh: CellMesh, measure: float) -> None:
    """Facets bound one or two cells, no vertex hangs on an edge, measure is kept."""

    cells = _cells(mesh)
    points = np.asarray(mesh.coordinates)
    width = cells.shape[1]
    facets = np.sort(
        np.stack([np.delete(cells, i, axis=1) for i in range(width)], axis=1), axis=2
    ).reshape((-1, width - 1))
    _, counts = np.unique(facets, axis=0, return_counts=True)
    assert set(np.unique(counts).tolist()) <= {1, 2}
    pairs = np.asarray(tuple(itertools.combinations(range(width), 2)))
    edges = np.unique(np.sort(cells[:, pairs].reshape((-1, 2)), axis=1), axis=0)
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    for start in range(0, midpoints.shape[0], 512):
        chunk = midpoints[start : start + 512, None, :] - points[None, :, :]
        assert np.min(np.linalg.norm(chunk, axis=2)) > 1.0e-9
    measures = _signed_measures(points, cells)
    assert np.all(measures > 0.0)
    assert np.isclose(np.sum(measures), measure, rtol=0.0, atol=1.0e-12)


def _assert_boundary_on_unit_box(mesh: CellMesh) -> None:
    cells = _cells(mesh)
    width = cells.shape[1]
    facets = np.sort(
        np.stack([np.delete(cells, i, axis=1) for i in range(width)], axis=1), axis=2
    ).reshape((-1, width - 1))
    keys, counts = np.unique(facets, axis=0, return_counts=True)
    corners = np.asarray(mesh.coordinates)[keys[counts == 1]]
    on_face = np.all(np.isclose(corners, 0.0), axis=1) | np.all(
        np.isclose(corners, 1.0), axis=1
    )
    assert np.all(np.any(on_face, axis=1))


def _corner_cells(mesh: CellMesh) -> np.ndarray:
    corner = np.argmin(np.linalg.norm(np.asarray(mesh.coordinates), axis=1))
    return np.sort(_cell_ids(mesh)[np.any(_cells(mesh) == corner, axis=1)])


@pytest.mark.parametrize(
    ("mesh", "cycles", "stride"),
    [(_triangle_grid(4, 4), 6, 7), (_kuhn_grid(2), 4, 11)],
    ids=["triangles", "tetrahedra"],
)
def test_adversarial_local_refinement_stays_conforming(mesh, cycles, stride):
    result = _adapt(_certified(mesh), (0, 5, 17, 30))
    for _ in range(cycles):
        target = result.target.mesh
        marks = np.union1d(_corner_cells(target), np.sort(_cell_ids(target))[::stride])
        result = _adapt(result.target, marks, hierarchy=result.hierarchy)
        _assert_conforming(result.target.mesh, 1.0)
        _assert_boundary_on_unit_box(result.target.mesh)
    assert result.status is MeshAdaptationStatus.COMPLETE
    assert result.evidence.maximum_generation >= cycles


def _uniform_class_bound(source, quality, rounds: int) -> float:
    """Minimum quality over the first uniform bisection generations."""

    bound = float(np.min(quality(source.mesh)))
    current, hierarchy = source, None
    for _ in range(rounds):
        result = _adapt(current, np.sort(_cell_ids(current.mesh)), hierarchy=hierarchy)
        current, hierarchy = result.target, result.hierarchy
        bound = min(bound, float(np.min(quality(current.mesh))))
    return bound


def test_newest_vertex_bisection_keeps_the_initial_angle_bound():
    source = _certified(_triangle_grid(2, 2, height=0.6))
    bound = _uniform_class_bound(source, _minimum_angles, 4)
    result = _adapt(source, _corner_cells(source.mesh))
    for _ in range(20):
        assert np.min(_minimum_angles(result.target.mesh)) >= bound - 1.0e-12
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
    assert np.min(_minimum_angles(result.target.mesh)) >= bound - 1.0e-12
    assert result.evidence.maximum_generation >= 20


def test_tetrahedral_bisection_keeps_bounded_similarity_classes():
    source = _certified(_kuhn_grid(1))
    bound = _uniform_class_bound(source, _mean_ratios, 3)
    result = _adapt(source, _corner_cells(source.mesh))
    for _ in range(9):
        assert np.min(_mean_ratios(result.target.mesh)) >= 0.999 * bound
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
    assert np.min(_mean_ratios(result.target.mesh)) >= 0.999 * bound
    assert result.evidence.maximum_generation >= 9


@pytest.mark.parametrize(
    "mesh", [_triangle_grid(3, 3), _kuhn_grid(2)], ids=["triangles", "tetrahedra"]
)
def test_refine_then_coarsen_restores_the_source_exactly(mesh):
    source = _certified(mesh)
    refined = _adapt(source, np.sort(_cell_ids(mesh))[::3])
    refined = _adapt(
        refined.target, _corner_cells(refined.target.mesh), hierarchy=refined.hierarchy
    )
    assert refined.target.mesh.topology_id != source.mesh.topology_id
    restored = _adapt(
        refined.target,
        (),
        np.sort(_cell_ids(refined.target.mesh)),
        hierarchy=refined.hierarchy,
    )
    target = restored.target.mesh
    # Marked cells that were never refined have no family to merge into.
    assert np.all(np.isin(restored.evidence.rejected_coarsening_ids, _cell_ids(mesh)))
    assert restored.evidence.coarsened_vertices == (
        refined.target.mesh.coordinates.shape[0] - mesh.coordinates.shape[0]
    )
    assert target.topology_id == source.mesh.topology_id
    np.testing.assert_array_equal(target.vertex_global_ids, mesh.vertex_global_ids)
    np.testing.assert_array_equal(_cell_ids(target), _cell_ids(mesh))
    np.testing.assert_array_equal(target.coordinates, mesh.coordinates)


@pytest.mark.parametrize(
    "mesh", [_triangle_grid(3, 3), _kuhn_grid(2)], ids=["triangles", "tetrahedra"]
)
def test_linear_fields_transfer_exactly(mesh):
    result = _adapt(_certified(mesh), np.sort(_cell_ids(mesh))[::2])
    result = _adapt(
        result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
    )
    slope = np.linspace(0.5, 1.5, mesh.ambient_dimension)
    before = np.asarray(result.source.mesh.coordinates) @ slope + 0.25
    expected = np.asarray(result.target.mesh.coordinates) @ slope + 0.25
    np.testing.assert_allclose(
        np.asarray(result.transfer.apply(before)), expected, rtol=0.0, atol=1.0e-13
    )


def test_marks_whose_closure_splits_a_protected_edge_are_rejected():
    source = _certified(_triangle_grid(4, 4))
    mesh = source.mesh
    edges = mesh.entity_set(1)
    keys = _edge_keys(mesh)
    points = np.asarray(mesh.coordinates)
    offsets = points[keys[:, 1]] - points[keys[:, 0]]
    diagonal = np.flatnonzero(np.all(np.isclose(offsets, 0.25), axis=1))[4]
    guarded = keys[diagonal]
    identifier = np.asarray(edges.entity_ids)[diagonal]
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        edges.entity_set_id,
        np.asarray([identifier]),
    )
    cell_vertices = np.asarray(mesh.vertex_global_ids)[_cells(mesh)]
    through = np.sort(
        _cell_ids(mesh)[np.sum(np.isin(cell_vertices, guarded), axis=1) == 2]
    )
    elsewhere = np.setdiff1d(_cell_ids(mesh), through)[[0, -1]]
    result = _adapt(source, np.union1d(through, elsewhere), protected_scopes=(scope,))
    assert result.status is MeshAdaptationStatus.PARTIAL
    np.testing.assert_array_equal(result.evidence.rejected_refinement_ids, through)
    target = result.target.mesh
    row = np.flatnonzero(np.all(_edge_keys(target) == guarded, axis=1))
    assert row.size == 1
    assert np.asarray(target.entity_set(1).entity_ids)[row[0]] == identifier
    assert not np.any(np.isin(_cell_ids(target), elsewhere))


def test_protected_vertices_are_never_coarsened_away():
    source = _certified(_triangle_grid(2, 2))
    refined = _adapt(source, np.sort(_cell_ids(source.mesh)))
    refined = _adapt(
        refined.target,
        np.sort(_cell_ids(refined.target.mesh)),
        hierarchy=refined.hierarchy,
    )
    mesh = refined.target.mesh
    identifiers = np.asarray(mesh.vertex_global_ids)
    created = np.setdiff1d(identifiers, np.asarray(source.mesh.vertex_global_ids))
    kept = created[:1]
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        mesh.entity_set(0).entity_set_id,
        kept,
    )
    result = _adapt(
        refined.target,
        (),
        np.sort(_cell_ids(mesh)),
        hierarchy=refined.hierarchy,
        protected_scopes=(scope,),
    )
    remaining = np.asarray(result.target.mesh.vertex_global_ids)
    surviving = np.setdiff1d(remaining, np.asarray(source.mesh.vertex_global_ids))
    np.testing.assert_array_equal(surviving, kept)
    assert result.evidence.coarsened_vertices == created.size - 1
    _assert_conforming(result.target.mesh, 1.0)


def test_incompatible_labelling_is_rejected_unless_uniform_refinement_is_requested():
    points = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)])
    mesh = CellMesh.from_triangles(points, np.asarray([(0, 1, 2), (0, 3, 1)]))
    source = _certified(mesh)
    with pytest.raises(ValueError, match="UNIFORM_REFINEMENT"):
        _adapt(source, (0,))
    result = _adapt(source, (0,), compatibility=BisectionCompatibility.UNIFORM_REFINEMENT)
    assert result.evidence.uniform_refinement_applied
    assert result.evidence.initially_compatible is False
    _assert_conforming(result.target.mesh, 3.5)
    follow = _adapt(
        result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
    )
    _assert_conforming(follow.target.mesh, 3.5)


def test_bisection_is_deterministic():
    first = _adapt(_certified(_kuhn_grid(2)), (0, 7, 13))
    second = _adapt(_certified(_kuhn_grid(2)), (0, 7, 13))
    assert first.result_id == second.result_id
    assert first.hierarchy.hierarchy_id == second.hierarchy.hierarchy_id
