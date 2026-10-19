#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


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
from tests._support.common_refinement import (
    assert_complete_moments,
    CELL_CENTROID_RTOL,
    CELL_MEASURE_RTOL,
    cube_lattice,
    hexahedra as hexahedra_mesh,
    kuhn_tetrahedra,
    PARTITION_MEASURE_ATOL,
    perturbed_triangles,
    quad_mesh,
    SINGLE_CELL_RTOL,
    SOURCE_COVERAGE_RTOL,
    square_grid,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]


def test_identical_triangle_refinement_retains_cell_order_and_moments() -> None:
    mesh = perturbed_triangles(6, 3)
    refinement = prepare_common_refinement(mesh, mesh)

    assert refinement.succeeded
    cells = np.arange(mesh.blocks[0].cell_count, dtype=np.int64)
    np.testing.assert_array_equal(
        refinement.target_offsets, np.arange(cells.size + 1, dtype=np.int64)
    )
    np.testing.assert_array_equal(refinement.source_cells, cells)
    np.testing.assert_array_equal(refinement.target_cells, cells)
    np.testing.assert_allclose(
        refinement.volumes, refinement.source_measures, rtol=CELL_MEASURE_RTOL
    )
    np.testing.assert_allclose(
        refinement.first_moments / refinement.volumes[:, None],
        np.mean(np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)], 1),
        rtol=CELL_CENTROID_RTOL,
    )
    np.testing.assert_array_equal(
        refinement.source_cell_global_ids, mesh.blocks[0].global_ids
    )
    # Neighbors sharing a face only touch: exact classification drops them.
    assert refinement.evidence.candidate_pair_count > refinement.entry_count
    assert refinement.source_mesh_id == refinement.target_mesh_id == mesh.mesh_id


def test_mixed_polygon_refinement_retains_ordered_overlap_partitions() -> None:
    points, quads = square_grid(5)
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
    target = perturbed_triangles(7, 5)
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    refinement = prepare_common_refinement(source, target, policy=policy)

    assert_complete_moments(refinement, 2)
    simplices = np.asarray(refinement.simplices)
    edges = simplices[:, 1:] - simplices[:, :1]
    areas = 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    offsets = np.asarray(refinement.simplex_offsets)
    np.testing.assert_allclose(
        np.add.reduceat(areas, offsets[:-1]),
        refinement.volumes,
        atol=PARTITION_MEASURE_ATOL,
    )
    np.testing.assert_allclose(
        np.bincount(
            refinement.source_cells,
            weights=refinement.volumes,
            minlength=refinement.source_cell_count,
        ),
        refinement.source_measures,
        rtol=SOURCE_COVERAGE_RTOL,
    )
    rows = np.asarray(refinement.target_cells)
    assert np.all(np.diff(rows) >= 0)
    ordered = np.lexsort((np.asarray(refinement.source_cells), rows))
    np.testing.assert_array_equal(ordered, np.arange(rows.size, dtype=np.int64))


def test_reflex_polygon_uses_a_certified_cone_apex() -> None:
    # Reflex vertex (1, 1): the cone from vertex 0 folds, the cone from vertex 1
    # is certified.
    dart = CellMesh.from_polygons(
        np.asarray(((0.0, 0.0), (2.0, 1.0), (0.0, 2.0), (1.0, 1.0))),
        [np.arange(4, dtype=np.int64)],
    )
    square = CellMesh.from_polygons(
        np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))),
        [np.arange(4, dtype=np.int64)],
    )
    refinement = prepare_common_refinement(
        dart,
        square,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
    )

    assert refinement.succeeded
    assert float(refinement.source_measures[0]) == pytest.approx(
        1.0, rel=SINGLE_CELL_RTOL
    )
    assert float(refinement.volumes[0]) == pytest.approx(1.0, rel=SINGLE_CELL_RTOL)
    np.testing.assert_allclose(
        refinement.first_moments[0], (1.0, 1.0), rtol=CELL_MEASURE_RTOL
    )


@pytest.mark.parametrize(
    "pair",
    ("tetrahedra_hexahedra", "hexahedra_polyhedra", "prisms_tetrahedra"),
)
def test_volume_family_refinement_integrates_unit_cube_moments(pair: str) -> None:
    tetrahedra = kuhn_tetrahedra(3)
    hexahedra = hexahedra_mesh(2)
    points, corners = cube_lattice(2)
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
    pairs = {
        "tetrahedra_hexahedra": (tetrahedra, hexahedra),
        "hexahedra_polyhedra": (hexahedra, polyhedra),
        "prisms_tetrahedra": (prisms, tetrahedra),
    }
    source, target = pairs[pair]
    assert_complete_moments(prepare_common_refinement(source, target, policy=policy), 3)


def test_pyramid_refinement_retains_its_volume() -> None:
    hexahedra = hexahedra_mesh(2)
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
    assert float(np.sum(refinement.volumes)) == pytest.approx(
        1.0 / 6.0, rel=CELL_MEASURE_RTOL
    )


def test_coverage_policy_distinguishes_source_and_target_gaps() -> None:
    source = quad_mesh(4)
    points, quads = square_grid(2)
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
    np.testing.assert_allclose(np.sum(complete.volumes), 0.25, rtol=CELL_MEASURE_RTOL)


def test_overlapping_source_cells_report_double_target_coverage() -> None:
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
        [np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64)],
    )
    refinement = prepare_common_refinement(overlapping, quad_mesh(2))

    assert refinement.status is CommonRefinementStatus.DOUBLE_COVERAGE
    assert refinement.evidence.target_double_count == 2


def test_filtered_sliver_refuses_while_exact_refinement_succeeds() -> None:
    # Nearly collinear corner: the filter cannot resolve it, exact predicates can.
    sliver = np.asarray(((0.5, 0.5), (12.0, 12.0), (24.0, 24.0 + 2.0**-48)))
    # ty: ignore[invalid-argument-type]
    mesh = CellMesh.from_triangles(sliver, ((0, 1, 2),))
    filtered = prepare_common_refinement(
        mesh, mesh, policy=CommonRefinementPolicy(predicate_mode=PredicateMode.FILTERED)
    )
    exact = prepare_common_refinement(mesh, mesh)
    assert filtered.status is CommonRefinementStatus.PREDICATE_UNCERTAIN
    # One unresolved cell in each of the two (identical) meshes.
    assert filtered.evidence.uncertain_predicate_count == 2
    assert filtered.entry_count == 0
    assert exact.succeeded


def test_outside_exact_domain_reports_uncertain_refinement() -> None:
    tiny = CellMesh.from_triangles(
        1.0e-40 * np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        # ty: ignore[invalid-argument-type]
        ((0, 1, 2),),
    )
    outside = prepare_common_refinement(tiny, tiny)
    assert outside.status is CommonRefinementStatus.PREDICATE_UNCERTAIN


def test_bowtie_cell_reports_invalid_geometry() -> None:
    bowtie = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))),
        # ty: ignore[invalid-argument-type]
        (CellBlock("bowtie", "quadrilateral", ((0, 1, 2, 3),)),),
    )
    invalid = prepare_common_refinement(bowtie, quad_mesh(1))

    assert invalid.status is CommonRefinementStatus.INVALID_GEOMETRY
    assert invalid.evidence.invalid_cell_count == 1


@pytest.mark.parametrize(
    "policy",
    (
        pytest.param(CommonRefinementPolicy(maximum_candidate_pairs=10), id="candidates"),
        pytest.param(CommonRefinementPolicy(maximum_accepted_pairs=10), id="accepted"),
        pytest.param(CommonRefinementPolicy(maximum_memory_bytes=4096), id="memory"),
    ),
)
def test_resource_refusal_retains_empty_publication(
    policy: CommonRefinementPolicy,
) -> None:
    source = quad_mesh(6)
    target = perturbed_triangles(5, 1)
    baseline = prepare_common_refinement(source, target)
    refused = prepare_common_refinement(source, target, policy=policy)
    assert refused.status is CommonRefinementStatus.RESOURCE_LIMIT
    assert refused.entry_count == 0
    np.testing.assert_array_equal(refused.target_offsets, 0)
    assert baseline.succeeded
    assert baseline.evidence.retained_bytes > 0


def test_repeated_refinement_retains_content_identity() -> None:
    source = quad_mesh(3)
    target = perturbed_triangles(4, 2)
    first = prepare_common_refinement(source, target)
    second = prepare_common_refinement(source, target)

    assert first.refinement_id == second.refinement_id
    np.testing.assert_array_equal(first.volumes, second.volumes)


def test_refinement_refuses_wrong_mesh_kind() -> None:
    source = quad_mesh(3)
    with pytest.raises(TypeError, match="CellMesh"):
        # ty: ignore[invalid-argument-type]
        prepare_common_refinement(source, object())


def test_refinement_refuses_incompatible_dimensions() -> None:
    source = quad_mesh(3)
    with pytest.raises(ValueError, match="dimension"):
        prepare_common_refinement(source, kuhn_tetrahedra(1))


def test_policy_refuses_device_predicate_mode() -> None:
    with pytest.raises(ValueError, match="FILTERED or EXACT"):
        CommonRefinementPolicy(predicate_mode=PredicateMode.FILTERED_DEVICE)
