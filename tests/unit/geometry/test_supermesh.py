#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from phydrax._meshcore import (
    meshcore_available,
    MeshcoreStatus,
    tetrahedron_intersection_moments,
    tetrahedron_intersection_simplices,
)
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    PredicateMode,
    prepare_common_refinement,
)
from phydrax.geometry._supermesh import _simplex_second_moments
from phydrax.geometry._tetra_intersections import (
    _exact_intersection_pieces,
    _exact_pieces_volume,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]


def _thin_overlap_tetrahedra() -> tuple[np.ndarray, np.ndarray]:
    source = (
        "0x1.998159b62a6b7p-1",
        "0x0p+0",
        "0x1.4013b5588f94dp-1",
        "0x1.91cca930f0bd5p-1",
        "0x0p+0",
        "0x1.36c43925439f2p-1",
        "0x1.9bb4e3ff3e99dp-1",
        "0x1.be61ebc283a9fp-7",
        "0x1.3a496f160e86bp-1",
        "0x1.99067e5d6242ap-1",
        "0x0p+0",
        "0x1.38a14cfbcaa8ap-1",
    )
    target = (
        "0x1.8fe65498785eap-1",
        "0x1.6666666666666p-8",
        "0x1.34fbb62c3b693p-1",
        "0x1.956993c7297ffp-1",
        "0x0p+0",
        "0x1.37b2c3108723ep-1",
        "0x1.9a5db12e506e4p-1",
        "0x1.be61ebc283a9fp-8",
        "0x1.39755e08ec97bp-1",
        "0x1.96c0c69817ab9p-1",
        "0x1.be61ebc283a9fp-8",
        "0x1.3886d41da912fp-1",
    )
    return (
        np.asarray([float.fromhex(x) for x in source], dtype=np.float64).reshape(4, 3),
        np.asarray([float.fromhex(x) for x in target], dtype=np.float64).reshape(4, 3),
    )


def _exact_carrier_second(simplex: np.ndarray, /) -> np.ndarray:
    points = [[Fraction(float(value)) for value in point] for point in simplex]
    u, v, w = (
        [points[i][axis] - points[0][axis] for axis in range(3)] for i in range(1, 4)
    )
    determinant = (
        u[0] * (v[1] * w[2] - v[2] * w[1])
        - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
    )
    return np.asarray(
        [
            [
                float(
                    determinant
                    * (
                        sum(point[j] * point[k] for point in points)
                        + sum(point[j] for point in points)
                        * sum(point[k] for point in points)
                    )
                    / 120
                )
                for k in range(3)
            ]
            for j in range(3)
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    "scale,shift",
    [(1.0, (0.0, 0.0, 0.0)), (8.0, (2.0, -4.0, 6.0)), (0.25, (0.125, -0.25, 0.5))],
    ids=("unit", "scaled-translated-large", "scaled-translated-small"),
)
@pytest.mark.parametrize("reverse", [False, True], ids=("native-order", "reversed-order"))
def test_signed_thin_carrier_second_moments(
    scale: float, shift: tuple[float, float, float], reverse: bool
) -> None:
    source, target = _thin_overlap_tetrahedra()
    simplices, counts, _, _, status = tetrahedron_intersection_simplices(
        source[None], target[None]
    )
    assert status[0] == 0 and counts[0] > 0
    carriers = simplices[0, : counts[0]] * scale + np.asarray(shift, dtype=np.float64)
    if reverse:
        carriers = carriers[:, [0, 2, 1, 3]]
    expected = np.asarray([_exact_carrier_second(piece) for piece in carriers])
    np.testing.assert_allclose(
        _simplex_second_moments(carriers), expected, rtol=2e-14, atol=0.0
    )


def test_well_conditioned_carrier_second_moments() -> None:
    simplex = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    carriers = np.stack((simplex, simplex[[0, 2, 1, 3]]))
    expected = np.asarray([_exact_carrier_second(piece) for piece in carriers])
    np.testing.assert_allclose(
        _simplex_second_moments(carriers), expected, rtol=2e-15, atol=0.0
    )


def test_thin_common_refinement_carrier_second_moments() -> None:
    source, target = _thin_overlap_tetrahedra()
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    refinement = prepare_common_refinement(
        CellMesh.from_tetrahedra(source, cells),
        CellMesh.from_tetrahedra(target, cells),
        policy=CommonRefinementPolicy(
            coverage=CommonRefinementCoverage.PARTIAL,
            second_moments=True,
            overlap_simplices=True,
        ),
    )
    assert refinement.succeeded and refinement.entry_count == 1
    carriers = np.asarray(refinement.simplices)
    expected = np.sum([_exact_carrier_second(piece) for piece in carriers], axis=0)
    np.testing.assert_allclose(
        np.asarray(refinement.second_moments)[0], expected, rtol=2e-14, atol=0.0
    )


def test_positive_overlap_with_collapsed_carriers_is_not_partial_success() -> None:
    source = np.asarray(
        [
            float.fromhex(value)
            for value in (
                "0x1.c5ad1c6273193p+0",
                "0x1p+0",
                "0x1.a07e9952a3352p+0",
                "0x1.ccc0acdb1535cp+0",
                "0x1p+0",
                "0x1.a009daac47ca6p+0",
                "0x1.c8e65498785eap+0",
                "0x1p+0",
                "0x1.9b621c92a1cf9p+0",
                "0x1.cdda71ff9f4cep+0",
                "0x1.037cc3d785075p+0",
                "0x1.9d24b78b07436p+0",
            )
        ],
        dtype=np.float64,
    ).reshape(4, 3)
    target = np.asarray(
        [
            float.fromhex(value)
            for value in (
                "0x1.c7f32a4c3c2f5p+0",
                "0x1.0166666666666p+0",
                "0x1.9a7ddb161db4ap+0",
                "0x1.cab4c9e394c00p+0",
                "0x1p+0",
                "0x1.9bd961884391fp+0",
                "0x1.cd2ed89728372p+0",
                "0x1.01be61ebc283bp+0",
                "0x1.9cbaaf04764bep+0",
                "0x1.cb60634c0bd5cp+0",
                "0x1.01be61ebc283bp+0",
                "0x1.9c436a0ed4898p+0",
            )
        ],
        dtype=np.float64,
    ).reshape(4, 3)
    exact = _exact_pieces_volume(_exact_intersection_pieces(source, target))
    assert exact > 0
    assert float(exact) == pytest.approx(2.0102052517994655e-48, rel=1e-15)
    _, counts, _, _, status = tetrahedron_intersection_simplices(
        source[None], target[None]
    )
    assert counts[0] == 0 and status[0] == MeshcoreStatus.UNRESOLVED
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    refinement = prepare_common_refinement(
        CellMesh.from_tetrahedra(source, cells),
        CellMesh.from_tetrahedra(target, cells),
        policy=CommonRefinementPolicy(
            coverage=CommonRefinementCoverage.PARTIAL, overlap_simplices=True
        ),
    )
    assert not refinement.succeeded
    assert refinement.status is CommonRefinementStatus.INTERSECTION_FAILURE
    assert refinement.evidence.intersection_failure_count == 1


def test_positive_symbolic_overlap_below_binary64_is_unresolved() -> None:
    small, large = 2.0**-120, 2.0**120
    source = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    target = np.asarray(
        (
            (small, -large, large),
            (-large, large, 0),
            (0, small, -small),
            (-large, -large, -large),
        ),
        dtype=np.float64,
    )
    exact = _exact_pieces_volume(_exact_intersection_pieces(source, target))
    assert 0 < exact < Fraction(float(np.nextafter(0.0, 1.0))) / 2
    volume, _, status = tetrahedron_intersection_moments(source[None], target[None])
    assert volume[0] == 0.0 and status[0] == MeshcoreStatus.UNRESOLVED
    _, counts, volume, _, status = tetrahedron_intersection_simplices(
        source[None], target[None]
    )
    assert counts[0] == 0 and volume[0] == 0.0 and status[0] == MeshcoreStatus.UNRESOLVED


def test_partial_refinement_accepts_true_zero_contact() -> None:
    source = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    target = np.asarray(((0, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    refinement = prepare_common_refinement(
        CellMesh.from_tetrahedra(source, cells),
        CellMesh.from_tetrahedra(target, cells),
        policy=CommonRefinementPolicy(
            coverage=CommonRefinementCoverage.PARTIAL, overlap_simplices=True
        ),
    )
    assert refinement.succeeded and refinement.entry_count == 0
    assert refinement.evidence.intersection_failure_count == 0


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
    mesh = CellMesh.from_triangles(sliver, ((0, 1, 2),))
    filtered = prepare_common_refinement(
        mesh, mesh, policy=CommonRefinementPolicy(predicate_mode=PredicateMode.FILTERED)
    )
    exact = prepare_common_refinement(mesh, mesh)
    tiny = CellMesh.from_triangles(
        1.0e-40 * np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        ((0, 1, 2),),
    )
    outside = prepare_common_refinement(tiny, tiny)
    bowtie = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))),
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


def test_polyhedral_decomposition_measures_are_finite_volume_cell_volumes() -> None:
    from phydrax.discretization.finite_volume import (
        prepare_polyhedral_finite_volume_geometry,
    )

    points, corners = _cube_lattice(2)
    rng = np.random.default_rng(11)
    # Lifting interior and top vertices warps faces of every cell.
    points[:, 2] += rng.uniform(-0.05, 0.05, points.shape[0]) * points[:, 2]
    faces = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4))
    faces += ((1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6))
    mesh = CellMesh.from_polyhedra(
        points, [[cell[list(face)] for face in faces] for cell in corners]
    )
    volumes = np.asarray(prepare_polyhedral_finite_volume_geometry(mesh).cell_volumes)
    refinement = prepare_common_refinement(mesh, mesh)

    assert refinement.succeeded
    np.testing.assert_allclose(refinement.source_measures, volumes, rtol=1e-14)
    np.testing.assert_allclose(
        np.bincount(
            refinement.target_cells,
            weights=refinement.volumes,
            minlength=refinement.target_cell_count,
        ),
        volumes,
        rtol=1e-13,
    )
    # Planar caps of an L-shaped prism that do not fan from their first vertex
    # have no finite-volume geometry: the cell stays uncertified, with no measure.
    outline = ((2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0), (0.0, 0.0))
    prism_points = np.asarray([(x, y, z) for z in (0.0, 1.0) for x, y in outline])
    count = len(outline)
    loops = [tuple(reversed(range(count))), tuple(range(count, 2 * count))]
    loops += [
        (i, (i + 1) % count, count + (i + 1) % count, count + i) for i in range(count)
    ]
    prism = CellMesh.from_polyhedra(prism_points, [loops])
    invalid = prepare_common_refinement(prism, prism)

    assert invalid.status is CommonRefinementStatus.INVALID_GEOMETRY
    assert invalid.evidence.invalid_cell_count == 2
    np.testing.assert_array_equal(invalid.source_measures, (0.0,))


def test_warped_prism_remaps_with_a_certified_interior_cone() -> None:
    from phydrax.discretization.finite_volume import (
        prepare_unstructured_conservative_remap,
        UnstructuredFiniteVolumePlan,
    )

    # Cell 2 of concave-box-core-lifecycle: its FV face triangles admit an
    # interior cone, but no mesh vertex sees every oriented boundary triangle.
    points = np.asarray(
        (
            (0.0, 0.0, 0.3333333333333333),
            (0.0, 0.3333333333333333, 0.3333333333333333),
            (0.0, 0.3333333333333333, 0.6666666666666666),
            (0.010000000000000002, 0.010000000000000005, 0.3347518376925911),
            (0.009999999999999998, 0.33564663370289066, 0.33564663370289066),
            (0.010000000000000004, 0.3352757174673883, 0.6646241600507783),
        ),
        dtype=np.float64,
    )
    faces = ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
    mesh = CellMesh.from_polyhedra(points, [faces])
    volume = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
    remap = prepare_unstructured_conservative_remap(
        volume, volume, provenance="certified-interior-prism"
    )

    assert remap.status is CommonRefinementStatus.SUCCESS
    assert remap.plan is not None
    np.testing.assert_allclose(
        remap.refinement.source_measures, volume.cell_volumes, rtol=1e-13, atol=0.0
    )
    assert float(volume.cell_volumes[0]) == pytest.approx(
        0.0005455764894995036, rel=1e-13, abs=0.0
    )
    np.testing.assert_allclose(
        remap.plan.apply(np.asarray([1.25], dtype=np.float64)),
        np.asarray([1.25], dtype=np.float64),
        rtol=1e-13,
        atol=0.0,
    )
