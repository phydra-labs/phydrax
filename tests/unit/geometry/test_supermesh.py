#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    PredicateMode,
    prepare_common_refinement,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]


def _grid(cells: int, /) -> Any:
    axis = np.linspace(0.0, 1.0, cells + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel()))
    index = np.arange((cells + 1) ** 2).reshape(cells + 1, cells + 1)
    quads = np.stack(
        (index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1
    ).reshape((-1, 4))
    return points, quads


def _quad_mesh(cells: int, /) -> CellMesh:
    points, quads = _grid(cells)
    return CellMesh(points, (CellBlock("quads", "quadrilateral", quads),))


def _perturbed_triangles(cells: int, seed: int, /) -> CellMesh:
    points, quads = _grid(cells)
    rng = np.random.default_rng(seed)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    points[interior] += rng.uniform(-0.2, 0.2, (int(np.sum(interior)), 2)) / cells
    return CellMesh.from_triangles(
        points, np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    )


def _unit_square_moments(dimension: int, /) -> Any:
    first = np.full((dimension,), 0.5)
    second = np.full((dimension, dimension), 0.25) + np.eye(dimension) / 12.0
    return first, second


def _cube_lattice(cells: int, /) -> Any:
    axis = np.linspace(0.0, 1.0, cells + 1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    index = np.arange((cells + 1) ** 3).reshape((cells + 1,) * 3)
    corners = np.stack(
        [
            index[i : cells + i, j : cells + j, k : cells + k].ravel()
            for k in (0, 1)
            for j in (0, 1)
            for i in (0, 1)
        ],
        axis=1,
    )
    return points, corners


def _kuhn_tetrahedra(cells: int, /) -> CellMesh:
    points, corners = _cube_lattice(cells)
    paths = ((1, 3, 7), (1, 5, 7), (2, 3, 7), (2, 6, 7), (4, 5, 7), (4, 6, 7))
    tetrahedra = np.concatenate([corners[:, [0, a, b, c]] for a, b, c in paths])
    edges = points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]]
    negative = np.sum(edges[:, 0] * np.cross(edges[:, 1], edges[:, 2]), axis=1) < 0.0
    tetrahedra[negative] = tetrahedra[negative][:, [0, 2, 1, 3]]
    return CellMesh.from_tetrahedra(points, tetrahedra)


def _hexahedra(cells: int, /) -> CellMesh:
    points, corners = _cube_lattice(cells)
    return CellMesh(
        points,
        (CellBlock("hexahedra", "hexahedron", corners[:, [0, 1, 3, 2, 4, 5, 7, 6]]),),
    )


def _assert_complete_moments(refinement: Any, dimension: int, /) -> None:
    first, second = _unit_square_moments(dimension)
    assert refinement.status is CommonRefinementStatus.SUCCESS
    assert refinement.succeeded
    assert np.sum(np.asarray(refinement.volumes)) == pytest.approx(1.0, rel=1e-13)
    np.testing.assert_allclose(
        np.sum(np.asarray(refinement.first_moments), axis=0), first, rtol=1e-13
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(refinement.second_moments), axis=0), second, rtol=1e-12
    )
    evidence = refinement.evidence
    assert evidence.maximum_relative_source_defect < 1e-12
    assert evidence.maximum_relative_target_defect < 1e-12
    assert evidence.source_gap_count == evidence.source_double_count == 0


def test_supermesh_scenario_1() -> None:
    mesh = _perturbed_triangles(6, 3)
    refinement = prepare_common_refinement(mesh, mesh)

    assert refinement.succeeded
    cells = np.arange(mesh.blocks[0].cell_count)
    np.testing.assert_array_equal(refinement.target_offsets, np.arange(cells.size + 1))
    np.testing.assert_array_equal(refinement.source_cells, cells)
    np.testing.assert_array_equal(refinement.target_cells, cells)
    np.testing.assert_allclose(refinement.volumes, refinement.source_measures, rtol=1e-14)
    np.testing.assert_allclose(
        refinement.first_moments / refinement.volumes[:, None],
        np.mean(np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)], 1),
        rtol=1e-13,
    )
    np.testing.assert_array_equal(
        refinement.source_cell_global_ids, mesh.blocks[0].global_ids
    )
    # Neighbors sharing a face only touch: exact classification drops them.
    assert refinement.evidence.candidate_pair_count > refinement.entry_count
    assert refinement.source_mesh_id == refinement.target_mesh_id == mesh.mesh_id
    points, quads = _grid(5)
    # Moving vertex (0.4, 0.4) inward turns its lower-left quadrilateral into a
    # nonconvex dart that needs a certified cone from a vertex.
    points[14] = (0.28, 0.28)
    hexagon = np.asarray((0, 6, 7, 8, 2, 1))
    loops = [hexagon]
    loops += [quad for quad in quads[2:10]]
    loops += [quads[10][[0, 1, 2]], quads[10][[0, 2, 3]]]
    loops += [quad for quad in quads[11:]]
    source = CellMesh.from_polygons(points, loops)
    assert {block.cell_kind for block in source.blocks} == {
        "triangle",
        "quadrilateral",
        "polygon",
    }
    target = _perturbed_triangles(7, 5)
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    refinement = prepare_common_refinement(source, target, policy=policy)

    _assert_complete_moments(refinement, 2)
    simplices = np.asarray(refinement.simplices)
    edges = simplices[:, 1:] - simplices[:, :1]
    areas = 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    offsets = np.asarray(refinement.simplex_offsets)
    np.testing.assert_allclose(
        np.add.reduceat(areas, offsets[:-1]), refinement.volumes, atol=1e-16
    )
    np.testing.assert_allclose(
        np.bincount(
            refinement.source_cells,
            weights=refinement.volumes,
            minlength=refinement.source_cell_count,
        ),
        refinement.source_measures,
        rtol=1e-12,
    )
    rows = np.asarray(refinement.target_cells)
    assert np.all(np.diff(rows) >= 0)
    ordered = np.lexsort((np.asarray(refinement.source_cells), rows))
    np.testing.assert_array_equal(ordered, np.arange(rows.size))
    # Reflex vertex (1, 1): the cone from vertex 0 folds, the cone from vertex 1
    # is certified.
    dart = CellMesh.from_polygons(
        np.asarray(((0.0, 0.0), (2.0, 1.0), (0.0, 2.0), (1.0, 1.0))), [np.arange(4)]
    )
    square = CellMesh.from_polygons(
        np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))), [np.arange(4)]
    )
    refinement = prepare_common_refinement(
        dart,
        square,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
    )

    assert refinement.succeeded
    assert float(refinement.source_measures[0]) == pytest.approx(1.0, rel=1e-15)
    assert float(refinement.volumes[0]) == pytest.approx(1.0, rel=1e-15)
    np.testing.assert_allclose(refinement.first_moments[0], (1.0, 1.0), rtol=1e-14)


def test_supermesh_scenario_2() -> None:
    tetrahedra = _kuhn_tetrahedra(3)
    hexahedra = _hexahedra(2)
    points, corners = _cube_lattice(2)
    hexahedron_faces = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4))
    hexahedron_faces += ((1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6))
    polyhedra = CellMesh.from_polyhedra(
        points, [[cell[list(face)] for face in hexahedron_faces] for cell in corners]
    )
    prism_points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)),
        dtype=np.float64,
    )
    prism_points = np.concatenate((prism_points, ((1, 1, 1),)))
    prisms = CellMesh(
        prism_points,
        # ty: ignore[invalid-argument-type]
        (CellBlock("prisms", "prism", ((0, 1, 2, 3, 4, 5), (1, 6, 2, 4, 7, 5))),),
    )
    policy = CommonRefinementPolicy(second_moments=True)
    for source, target in (
        (tetrahedra, hexahedra),
        (hexahedra, polyhedra),
        (prisms, tetrahedra),
    ):
        _assert_complete_moments(
            prepare_common_refinement(source, target, policy=policy), 3
        )
    pyramid_points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 0.5)), dtype=np.float64
    )
    pyramid = CellMesh(
        pyramid_points,
        # ty: ignore[invalid-argument-type]
        (CellBlock("pyramids", "pyramid", ((0, 1, 2, 3, 4),)),),
    )
    refinement = prepare_common_refinement(
        pyramid,
        hexahedra,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
    )
    assert refinement.succeeded
    assert float(np.sum(refinement.volumes)) == pytest.approx(1.0 / 6.0, rel=1e-14)
    source = _quad_mesh(4)
    points, quads = _grid(2)
    inner = CellMesh(0.5 * points + 0.25, (CellBlock("inner", "quadrilateral", quads),))
    complete = prepare_common_refinement(source, inner)
    covered_target = prepare_common_refinement(
        source,
        inner,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.TARGET),
    )
    covered_source = prepare_common_refinement(
        source,
        inner,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
    )

    assert complete.status is CommonRefinementStatus.COVERAGE_GAP
    assert complete.evidence.source_gap_count == 12
    assert complete.evidence.target_gap_count == 0
    assert covered_target.succeeded
    assert covered_source.status is CommonRefinementStatus.COVERAGE_GAP
    np.testing.assert_allclose(np.sum(complete.volumes), 0.25, rtol=1e-14)
    # Two cells covering [0.5, 1] x [0, 1] twice.
    overlapping = CellMesh.from_polygons(
        np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.5, 1.0),
            )
        ),
        [np.arange(4), np.arange(4, 8)],
    )
    refinement = prepare_common_refinement(overlapping, _quad_mesh(2))

    assert refinement.status is CommonRefinementStatus.DOUBLE_COVERAGE
    assert refinement.evidence.target_double_count == 2


def test_supermesh_scenario_3() -> None:
    # Nearly collinear corner: the filter cannot resolve it, exact predicates can.
    sliver = np.asarray(((0.5, 0.5), (12.0, 12.0), (24.0, 24.0 + 2.0**-48)))
    # ty: ignore[invalid-argument-type]
    mesh = CellMesh.from_triangles(sliver, ((0, 1, 2),))
    filtered = prepare_common_refinement(
        mesh, mesh, policy=CommonRefinementPolicy(predicate_mode=PredicateMode.FILTERED)
    )
    exact = prepare_common_refinement(mesh, mesh)
    tiny = CellMesh.from_triangles(
        1.0e-40 * np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        # ty: ignore[invalid-argument-type]
        ((0, 1, 2),),
    )
    outside = prepare_common_refinement(tiny, tiny)
    bowtie = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))),
        # ty: ignore[invalid-argument-type]
        (CellBlock("bowtie", "quadrilateral", ((0, 1, 2, 3),)),),
    )
    invalid = prepare_common_refinement(bowtie, _quad_mesh(1))

    assert filtered.status is CommonRefinementStatus.PREDICATE_UNCERTAIN
    # One unresolved cell in each of the two (identical) meshes.
    assert filtered.evidence.uncertain_predicate_count == 2
    assert filtered.entry_count == 0
    assert exact.succeeded
    assert outside.status is CommonRefinementStatus.PREDICATE_UNCERTAIN
    assert invalid.status is CommonRefinementStatus.INVALID_GEOMETRY
    assert invalid.evidence.invalid_cell_count == 1
    source = _quad_mesh(6)
    target = _perturbed_triangles(5, 1)
    baseline = prepare_common_refinement(source, target)
    for policy in (
        CommonRefinementPolicy(maximum_candidate_pairs=10),
        CommonRefinementPolicy(maximum_accepted_pairs=10),
        CommonRefinementPolicy(maximum_memory_bytes=4096),
    ):
        refused = prepare_common_refinement(source, target, policy=policy)
        assert refused.status is CommonRefinementStatus.RESOURCE_LIMIT
        assert refused.entry_count == 0
        np.testing.assert_array_equal(refused.target_offsets, 0)
    assert baseline.succeeded
    assert baseline.evidence.retained_bytes > 0
    source = _quad_mesh(3)
    target = _perturbed_triangles(4, 2)
    first = prepare_common_refinement(source, target)
    second = prepare_common_refinement(source, target)

    assert first.refinement_id == second.refinement_id
    np.testing.assert_array_equal(first.volumes, second.volumes)
    with pytest.raises(TypeError, match="CellMesh"):
        # ty: ignore[invalid-argument-type]
        prepare_common_refinement(source, object())
    with pytest.raises(ValueError, match="dimension"):
        prepare_common_refinement(source, _kuhn_tetrahedra(1))
    with pytest.raises(ValueError, match="FILTERED or EXACT"):
        CommonRefinementPolicy(predicate_mode=PredicateMode.FILTERED_DEVICE)
