#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
from fractions import Fraction
from typing import Any, NamedTuple

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
from phydrax.meshing._bisection import execute_bisection
from phydrax.meshing._topology_edit import (
    assemble_topology_edit,
    reference_cell_maps_from_edit,
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


def _certified(mesh: CellMesh) -> Any:
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _adapt(
    source: Any,
    refine: Any = (),
    coarsen: Any = (),
    /,
    *,
    hierarchy: Any = None,
    **options: Any,
) -> Any:
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION, **options)
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )

def test_coarsening_defers_early_chart_replay_error_for_unchanged_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref

    from phydrax.meshing import _adaptation

    source = _certified(_triangle_grid(1, 1))
    refined = _adapt(source, _cell_ids(source.mesh))

    class ProofFrame:
        pass

    retained: list[weakref.ReferenceType[ProofFrame]] = []

    def fail_replay(source: Any) -> None:
        proof = ProofFrame()
        retained.append(weakref.ref(proof))
        try:
            raise RuntimeError("ancestor replay refused")
        except RuntimeError as cause:
            raise ValueError("chart replay refused") from cause

    original_bisection = _adaptation.execute_bisection

    def checked_bisection(*args: Any, **kwargs: Any) -> Any:
        assert retained and all(reference() is None for reference in retained)
        return original_bisection(*args, **kwargs)

    monkeypatch.setattr(_adaptation, "execute_bisection", checked_bisection)
    monkeypatch.setattr(_adaptation, "_validated_source_chart_parent", fail_replay)
    unchanged = _adapt(
        refined.target, (), _cell_ids(refined.target.mesh)[:1],
        hierarchy=refined.hierarchy,
    )
    assert unchanged.status in (MeshAdaptationStatus.UNCHANGED, MeshAdaptationStatus.PARTIAL)
    with pytest.raises(ValueError, match="chart replay refused"):
        _adapt(
            refined.target, (), _cell_ids(refined.target.mesh),
            hierarchy=refined.hierarchy,
        )



def _cells(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks))


def _cell_ids(mesh: CellMesh) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks))


def _edge_keys(mesh: CellMesh) -> np.ndarray:
    # ty: ignore[unresolved-attribute]
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


def test_bisection_scenario_1() -> None:
    for mesh, cycles, stride in [(_triangle_grid(4, 4), 6, 7), (_kuhn_grid(2), 4, 11)]:
        result = _adapt(_certified(mesh), (0, 5, 17, 30))
        for _ in range(cycles):
            target = result.target.mesh
            marks = np.union1d(
                _corner_cells(target), np.sort(_cell_ids(target))[::stride]
            )
            result = _adapt(result.target, marks, hierarchy=result.hierarchy)
            _assert_conforming(result.target.mesh, 1.0)
            _assert_boundary_on_unit_box(result.target.mesh)
        assert result.status is MeshAdaptationStatus.COMPLETE
        assert result.evidence.maximum_generation >= cycles
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


def _uniform_class_bound(source: Any, quality: Any, rounds: int) -> float:
    """Minimum quality over the first uniform bisection generations."""

    bound = float(np.min(quality(source.mesh)))
    current, hierarchy = source, None
    for _ in range(rounds):
        result = _adapt(current, np.sort(_cell_ids(current.mesh)), hierarchy=hierarchy)
        current, hierarchy = result.target, result.hierarchy
        bound = min(bound, float(np.min(quality(current.mesh))))
    return bound


def test_bisection_scenario_2() -> None:
    for mesh in [_triangle_grid(3, 3), _kuhn_grid(2)]:
        source = _certified(mesh)
        refined = _adapt(source, np.sort(_cell_ids(mesh))[::3])
        refined = _adapt(
            refined.target,
            _corner_cells(refined.target.mesh),
            hierarchy=refined.hierarchy,
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
    for mesh in [_triangle_grid(3, 3), _kuhn_grid(2)]:
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


def test_bisection_scenario_3() -> None:
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
    first = _adapt(_certified(_kuhn_grid(2)), (0, 7, 13))
    second = _adapt(_certified(_kuhn_grid(2)), (0, 7, 13))
    assert first.result_id == second.result_id
    assert first.hierarchy.hierarchy_id == second.hierarchy.hierarchy_id


def test_uniform_barycentric_subdivision_coarsens_back_to_the_source() -> None:
    points = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)])
    source = _certified(
        CellMesh.from_triangles(points, np.asarray([(0, 1, 2), (0, 3, 1)]))
    )
    uniform = BisectionCompatibility.UNIFORM_REFINEMENT
    refined = _adapt(source, (0,), compatibility=uniform)
    hierarchy = refined.hierarchy
    np.testing.assert_array_equal(
        np.asarray(hierarchy.subdivision_parent_ids), np.sort(_cell_ids(source.mesh))
    )
    assert np.asarray(hierarchy.subdivision_child_ids).shape == (2, 6)
    # Children of one source cell alone cannot undo its subdivision: the shared
    # edge midpoint still bounds the other cell's children.
    one_side = np.asarray(hierarchy.subdivision_child_ids)[1]
    partial = _adapt(
        refined.target,
        (),
        np.intersect1d(_cell_ids(refined.target.mesh), one_side),
        hierarchy=hierarchy,
        compatibility=uniform,
    )
    assert np.asarray(partial.hierarchy.subdivision_parent_ids).size == 2
    coarsened = _adapt(
        refined.target,
        (),
        np.sort(_cell_ids(refined.target.mesh)),
        hierarchy=hierarchy,
        compatibility=uniform,
    )
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    np.testing.assert_array_equal(
        np.asarray(coarsened.target.mesh.coordinates), np.asarray(source.mesh.coordinates)
    )
    restored = coarsened.hierarchy
    assert np.asarray(restored.subdivision_parent_ids).size == 0
    assert np.asarray(restored.record_parent_ids).size == 0
    np.testing.assert_array_equal(np.asarray(restored.tags), np.zeros((2,)))
    with pytest.raises(ValueError, match="UNIFORM_REFINEMENT"):
        _adapt(coarsened.target, (0,), hierarchy=restored)
    again = _adapt(coarsened.target, (0,), hierarchy=restored, compatibility=uniform)
    assert again.evidence.uniform_refinement_applied
    assert np.asarray(again.hierarchy.subdivision_parent_ids).size == 2
    _assert_conforming(again.target.mesh, 3.5)
    assert not np.intersect1d(
        _cell_ids(again.target.mesh), _cell_ids(refined.target.mesh)
    ).size


def test_uniform_barycentric_refinement_keeps_exact_reference_ancestry() -> None:
    points = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)])
    mesh = CellMesh.from_triangles(points, np.asarray([(0, 1, 2), (0, 3, 1)]))
    outcome = execute_bisection(
        mesh,
        np.asarray((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        hierarchy=None,
        compatibility=BisectionCompatibility.UNIFORM_REFINEMENT,
        protected_edges=np.empty((0, 2), dtype=np.int64),
        protected_vertices=np.empty((0,), dtype=np.int64),
        cell_classes=np.zeros((mesh.entity_set(2).count,), dtype=np.int64),
        facet_classes=np.zeros((mesh.entity_set(1).count,), dtype=np.int64),
    )
    assert outcome.evidence.uniform_refinement_applied
    exact = outcome.edit.stencil_exact_weights
    assert exact is not None
    valid = np.asarray(outcome.edit.stencil_valid, dtype=np.bool_)
    assert all(
        sum((value for value, used in zip(row, mask, strict=True) if used), Fraction(0))
        == 1
        for row, mask in zip(exact, valid.tolist(), strict=True)
    )
    assert (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)) in exact
    target, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="uniform-barycentric-refinement"
    )
    maps = reference_cell_maps_from_edit(mesh, target, outcome.edit)
    assert sum(np.asarray(group.target_cell_ids).size for group in maps) == sum(
        block.cell_count for block in target.blocks
    )


type _Point = tuple[Fraction, ...]
type _Edge = frozenset[_Point]


class _MarkedTetrahedron(NamedTuple):
    """One tetrahedron of the direct Arnold-Mukherjee-Pouly marked bisection."""

    vertices: frozenset[_Point]
    refinement_edge: _Edge
    face_marks: dict[frozenset[_Point], _Edge]
    flagged: bool


def _midpoint(first: _Point, second: _Point) -> _Point:
    return tuple((a + b) / 2 for a, b in zip(first, second, strict=True))


def _marked_tetrahedra(points: np.ndarray, cells: np.ndarray) -> list[_MarkedTetrahedron]:
    """Initial marking: every face and tetrahedron marks its longest edge.

    Ties are broken by the ascending vertex-ID pair, so neighbors share face marks.
    """

    exact = [tuple(Fraction(int(value)) for value in point) for point in points]
    order = {point: index for index, point in enumerate(exact)}

    def rank(edge: tuple[_Point, _Point]) -> tuple[Fraction, int, int]:
        low, high = sorted(edge, key=order.__getitem__)
        length = sum(((a - b) ** 2 for a, b in zip(low, high, strict=True)), Fraction(0))
        return -length, order[low], order[high]

    def longest(simplex: tuple[_Point, ...]) -> _Edge:
        return frozenset(min(itertools.combinations(simplex, 2), key=rank))

    tetrahedra = []
    for cell in cells:
        corners = tuple(exact[int(vertex)] for vertex in cell)
        marks = {
            frozenset(face): longest(face) for face in itertools.combinations(corners, 3)
        }
        tetrahedra.append(
            _MarkedTetrahedron(frozenset(corners), longest(corners), marks, False)
        )
    return tetrahedra


def _marked_kind(tetrahedron: _MarkedTetrahedron) -> str:
    first, second = tuple(tetrahedron.refinement_edge)
    others = tetrahedron.vertices - tetrahedron.refinement_edge
    first_mark = tetrahedron.face_marks[others | {first}]
    second_mark = tetrahedron.face_marks[others | {second}]
    if first_mark == second_mark:
        return "opposite"
    if first_mark & second_mark:
        planar = first in first_mark and second in second_mark
        return "planar" if planar else "adjacent"
    return "mixed"


def _marked_children(tetrahedron: _MarkedTetrahedron) -> list[_MarkedTetrahedron]:
    """Bisect the refinement edge; children mark inherited and new faces by AMP."""

    first, second = tuple(tetrahedron.refinement_edge)
    others = tetrahedron.vertices - tetrahedron.refinement_edge
    middle = _midpoint(first, second)
    planar = _marked_kind(tetrahedron) == "planar"
    refinement = {
        keep: tetrahedron.face_marks[others | {keep}] for keep in (first, second)
    }
    children = []
    for keep in (first, second):
        marks = {others | {keep}: refinement[keep]}
        for other in others:
            marks[frozenset((keep, middle, other))] = frozenset((keep, other))
        if planar and tetrahedron.flagged:
            (common,) = refinement[first] & refinement[second]
            marks[others | {middle}] = frozenset((middle, common))
        else:
            marks[others | {middle}] = others
        children.append(
            _MarkedTetrahedron(
                others | {keep, middle},
                refinement[keep],
                marks,
                planar and not tetrahedron.flagged,
            )
        )
    return children


def _marked_refinement(
    leaves: list[_MarkedTetrahedron], marked: list[int], midpoints: set[_Point]
) -> list[_MarkedTetrahedron]:
    """Bisect the marked leaves, then every leaf with a hanging edge midpoint."""

    while marked:
        midpoints |= {_midpoint(*tuple(leaves[row].refinement_edge)) for row in marked}
        chosen = set(marked)
        leaves = [leaf for row, leaf in enumerate(leaves) if row not in chosen] + [
            child for row in marked for child in _marked_children(leaves[row])
        ]
        marked = [
            row
            for row, leaf in enumerate(leaves)
            if any(
                _midpoint(first, second) in midpoints
                for first, second in itertools.combinations(leaf.vertices, 2)
            )
        ]
    return leaves


def _exact_cells(mesh: CellMesh) -> set[frozenset[_Point]]:
    points = np.asarray(mesh.coordinates)
    return {
        frozenset(
            tuple(Fraction(float(value)) for value in points[vertex]) for vertex in cell
        )
        for cell in _cells(mesh)
    }


def _oriented_tetrahedra(points: np.ndarray, cells: np.ndarray) -> CellMesh:
    cells = np.asarray(cells, dtype=np.int32).copy()
    negative = _signed_measures(points, cells) < 0.0
    cells[negative] = cells[negative][:, [1, 0, 2, 3]]
    return CellMesh.from_tetrahedra(np.asarray(points, dtype=np.float64), cells)


def _uniform_bisection(mesh: CellMesh, refine: np.ndarray, hierarchy: Any) -> Any:
    return execute_bisection(
        mesh,
        np.asarray(refine, dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        hierarchy=hierarchy,
        compatibility=BisectionCompatibility.REJECT,
        protected_edges=np.empty((0, 2), dtype=np.int64),
        protected_vertices=np.empty((0,), dtype=np.int64),
        cell_classes=np.zeros((mesh.entity_set(3).count,), dtype=np.int64),
        facet_classes=np.zeros((mesh.entity_set(2).count,), dtype=np.int64),
    )


def test_marked_tetrahedron_bisection_equals_the_direct_marked_tetrahedron_algorithm() -> (
    None
):
    # Integer lattice with integer jitter: every bisection vertex is an exact
    # dyadic point, so target cells compare exactly with the direct algorithm.
    lattice = np.rint(np.asarray(_kuhn_grid(2).coordinates) * 32.0).astype(np.int64)
    points = lattice + np.random.default_rng(4).integers(-6, 7, size=lattice.shape)
    mesh = _oriented_tetrahedra(points.astype(np.float64), _cells(_kuhn_grid(2)))
    leaves = _marked_tetrahedra(np.asarray(mesh.coordinates), _cells(mesh))
    kinds = {_marked_kind(leaf) for leaf in leaves}
    assert kinds == {"planar", "mixed", "adjacent", "opposite"}
    corner = tuple(Fraction(int(value)) for value in points[0])
    midpoints: set[_Point] = set()
    hierarchy = None
    for generation in range(9):
        if generation == 0:
            marked = list(range(len(leaves)))
            refine = np.sort(_cell_ids(mesh))
        else:
            marked = [row for row, leaf in enumerate(leaves) if corner in leaf.vertices]
            at_corner = np.all(np.asarray(mesh.coordinates) == points[0], axis=1)
            refine = np.sort(_cell_ids(mesh)[np.any(at_corner[_cells(mesh)], axis=1)])
        leaves = _marked_refinement(leaves, marked, midpoints)
        outcome = _uniform_bisection(mesh, refine, hierarchy)
        if generation == 0:
            assert outcome.evidence.initially_compatible is False
            assert outcome.evidence.incompatible_facets > 0
            assert not outcome.evidence.uniform_refinement_applied
        mesh, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version=f"marked-generation-{generation}"
        )
        hierarchy = outcome.hierarchy
        assert _exact_cells(mesh) == {leaf.vertices for leaf in leaves}


def _jittered_kuhn_grid(size: int, amplitude: float, seed: int) -> CellMesh:
    """Kuhn grid of the unit cube with every vertex moved within its boundary stratum."""

    grid = _kuhn_grid(size)
    points = np.asarray(grid.coordinates)
    free = (points > 0.0) & (points < 1.0)
    shift = np.random.default_rng(seed).uniform(-amplitude, amplitude, points.shape)
    return _oriented_tetrahedra(points + np.where(free, shift / size, 0.0), _cells(grid))


def test_marked_tetrahedra_refine_and_coarsen_a_mesh_violating_the_matching_condition() -> (
    None
):
    source = _certified(_jittered_kuhn_grid(2, 0.4, 0))
    result = _adapt(source, (0, 5, 17, 30))
    evidence = result.evidence
    assert evidence.initially_compatible is False
    assert evidence.incompatible_facets > 0
    assert not evidence.uniform_refinement_applied
    assert evidence.bisections >= 4
    assert _adapt(source, (0, 5, 17, 30)).result_id == result.result_id
    for _ in range(3):
        _assert_conforming(result.target.mesh, 1.0)
        _assert_boundary_on_unit_box(result.target.mesh)
        result = _adapt(
            result.target, _corner_cells(result.target.mesh), hierarchy=result.hierarchy
        )
    _assert_conforming(result.target.mesh, 1.0)
    assert result.status is MeshAdaptationStatus.COMPLETE
    restored = _adapt(
        result.target,
        (),
        np.sort(_cell_ids(result.target.mesh)),
        hierarchy=result.hierarchy,
    )
    assert restored.target.mesh.topology_id == source.mesh.topology_id
    np.testing.assert_array_equal(
        restored.target.mesh.coordinates, source.mesh.coordinates
    )


def _similarity_classes(mesh: CellMesh) -> set[tuple[float, ...]]:
    corners = np.asarray(mesh.coordinates)[_cells(mesh)]
    pairs = np.asarray(tuple(itertools.combinations(range(4), 2)))
    squared = np.sort(
        np.sum((corners[:, pairs[:, 0]] - corners[:, pairs[:, 1]]) ** 2, axis=2), axis=1
    )
    return {tuple(np.round(row / row[-1], 9).tolist()) for row in squared}


def test_marked_tetrahedron_descendants_have_finitely_many_similarity_classes() -> None:
    # One adjacent and one opposite marked tetrahedron sharing an incompatible face.
    points = np.asarray(
        [(27, 5, 2), (27, 0, 17), (2, 9, 15), (13, 12, 0), (0, 3, 0)], dtype=np.float64
    )
    mesh = _oriented_tetrahedra(points, np.asarray([(0, 1, 2, 3), (0, 1, 2, 4)]))
    hierarchy, generations = None, []
    for generation in range(9):
        outcome = _uniform_bisection(mesh, np.sort(_cell_ids(mesh)), hierarchy)
        if generation == 0:
            assert outcome.evidence.initially_compatible is False
            assert {4, 5} <= set(
                np.asarray(outcome.hierarchy.record_parent_tags).tolist()
            )
        mesh, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version=f"uniform-generation-{generation}"
        )
        hierarchy = outcome.hierarchy
        generations.append(_similarity_classes(mesh))
    assert len(_cells(mesh)) > 2 * 2**12
    assert set().union(*generations[6:]) <= set().union(*generations[:6])
