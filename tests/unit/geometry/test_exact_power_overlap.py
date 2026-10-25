#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Check rational intersection integrals and typed overlap resource refusal."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    CoordinateSourceBank,
    ExactScalar,
    Polynomial,
)
from phydrax.geometry._affine_mapped_contact import _Intersection, Plane, Point
from phydrax.geometry._embedding_cells import Cell
from phydrax.geometry._exact_polyhedral_geometry import (
    convex_cell_overlap,
    convex_cells_disjoint,
    ExactConvexCell,
    ExactOverlapRefusal,
    ExactOverlapStatus,
    ExactTetrahedronOverlap,
    star_determinant,
    tetrahedra_disjoint,
    Tetrahedron,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
    triangulate_loop,
)
from tests._support.coordinate_references import coordinate_reference_budget


@pytest.fixture(scope="module")
def unit_tetrahedron() -> Tetrahedron:
    return (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def test_overlap_retains_rational_integrals_and_oriented_simplices(
    unit_tetrahedron: Tetrahedron,
) -> None:
    with coordinate_reference_budget() as budget:
        starting_work = budget.work_units
        starting_root_work = budget.ticket.work_units
        overlap = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            second_moments=True,
            retain_simplices=True,
            budget=budget,
        )
        assert overlap.operation_count == budget.work_units - starting_work
        assert overlap.operation_count == budget.ticket.work_units - starting_root_work
    assert isinstance(overlap, ExactTetrahedronOverlap)
    assert overlap.status is ExactOverlapStatus.SUCCESS
    # Uniform simplex barycentrics have E[x_i]=1/4, E[x_i^2]=1/10 and
    # E[x_i*x_j]=1/20 for i != j; the unit tetrahedron has volume 1/6.
    assert overlap.volume == Fraction(1, 6)
    assert overlap.first_moment == (Fraction(1, 24),) * 3
    assert overlap.second_moment == tuple(
        tuple(Fraction(1, 60 if row == column else 120) for column in range(3))
        for row in range(3)
    )
    assert overlap.simplices
    assert overlap.operation_count > 0


def test_overlap_work_refusal_has_typed_status_without_partial_integrals(
    unit_tetrahedron: Tetrahedron,
) -> None:
    with coordinate_reference_budget() as budget:
        starting_work = budget.work_units
        refusal = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            maximum_work=0,
            budget=budget,
        )
    assert isinstance(refusal, ExactOverlapRefusal)
    assert refusal.status is ExactOverlapStatus.RESOURCE_LIMIT
    assert refusal.resource == "coefficient_work"
    assert refusal.operation_count >= 0
    assert refusal.operation_count == 0
    assert refusal.operation_count == budget.work_units - starting_work


@pytest.mark.parametrize("offset", (1, 2), ids=("shared-boundary", "separated"))
def test_disjoint_bounds_and_overlap_agree_on_zero_volume(
    unit_tetrahedron: Tetrahedron,
    offset: int,
) -> None:
    translated = tuple(
        (point[0] + offset, point[1], point[2]) for point in unit_tetrahedron
    )
    second: Tetrahedron = (translated[0], translated[1], translated[2], translated[3])
    assert tetrahedra_disjoint(unit_tetrahedron, second)
    with coordinate_reference_budget() as budget:
        overlap = tetrahedron_overlap(unit_tetrahedron, second, budget=budget)
    assert isinstance(overlap, ExactTetrahedronOverlap)
    assert overlap.status is ExactOverlapStatus.SUCCESS
    assert overlap.volume == 0
    assert overlap.first_moment == (Fraction(0),) * 3


@pytest.mark.parametrize("second_moments", (False, True))
def test_earned_tetrahedron_planes_preserve_integrals_and_reduce_actual_work(
    unit_tetrahedron: Tetrahedron, second_moments: bool
) -> None:
    with coordinate_reference_budget() as budget:
        preparation = TetrahedronPlanePreparation(budget)
        first = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            second_moments=second_moments,
            retain_simplices=True,
            budget=budget,
            preparation=preparation,
        )
        retained = budget.retained_basis_bytes
        second = tetrahedron_overlap(
            unit_tetrahedron,
            unit_tetrahedron,
            second_moments=second_moments,
            retain_simplices=True,
            budget=budget,
            preparation=preparation,
        )
        assert budget.retained_basis_bytes == retained
    assert isinstance(first, ExactTetrahedronOverlap)
    assert isinstance(second, ExactTetrahedronOverlap)
    assert first.volume == second.volume == Fraction(1, 6)
    assert first.first_moment == second.first_moment
    assert first.second_moment == second.second_moment
    assert first.simplices == second.simplices
    assert 0 < second.operation_count < first.operation_count
    assert retained > 0


def test_earned_tetrahedron_planes_bind_full_coordinates_and_request(
    unit_tetrahedron: Tetrahedron,
) -> None:
    translated = tuple((point[0] + 2, point[1], point[2]) for point in unit_tetrahedron)
    second: Tetrahedron = (translated[0], translated[1], translated[2], translated[3])
    with coordinate_reference_budget() as budget:
        preparation = TetrahedronPlanePreparation(budget)
        first = tetrahedron_overlap(
            unit_tetrahedron, unit_tetrahedron, budget=budget, preparation=preparation
        )
        distinct = tetrahedron_overlap(
            unit_tetrahedron, second, budget=budget, preparation=preparation
        )
        assert isinstance(first, ExactTetrahedronOverlap)
        assert isinstance(distinct, ExactTetrahedronOverlap)
        assert distinct.volume == 0
        assert distinct.first_moment == (Fraction(0),) * 3
    with coordinate_reference_budget() as fresh:
        with pytest.raises(ValueError, match="current request budget"):
            tetrahedron_overlap(
                unit_tetrahedron,
                unit_tetrahedron,
                budget=fresh,
                preparation=preparation,
            )


def test_earned_star_determinant_reuses_only_full_source_piece(
    unit_tetrahedron: Tetrahedron,
) -> None:
    with coordinate_reference_budget() as budget:
        before = budget.work_units
        first = star_determinant(unit_tetrahedron, budget=budget)
        computed = budget.work_units - before
        work = budget.work_units
        retained = budget.retained_basis_bytes
        assert star_determinant(unit_tetrahedron, budget=budget) == first == 1
        # A reuse charges only hashing the twelve coordinates of its full key.
        assert budget.work_units - work == 12 < computed
        assert budget.retained_basis_bytes == retained
        reversed_tet = (
            unit_tetrahedron[0],
            unit_tetrahedron[1],
            unit_tetrahedron[3],
            unit_tetrahedron[2],
        )
        work = budget.work_units
        assert star_determinant(reversed_tet, budget=budget) == -1
        assert budget.work_units - work > 12


@pytest.mark.parametrize("shift", (Fraction(1, 4), Fraction(2, 3)))
def test_translated_simplex_overlap_matches_closed_form(
    unit_tetrahedron: Tetrahedron, shift: Fraction
) -> None:
    # The standard simplex meets its x-translate in a simplex of side 1 - shift
    # with corner (shift, 0, 0): volume (1-shift)^3/6, centroid corner + side/4.
    moved = tuple((point[0] + shift, point[1], point[2]) for point in unit_tetrahedron)
    second: Tetrahedron = (moved[0], moved[1], moved[2], moved[3])
    side = 1 - shift
    with coordinate_reference_budget() as budget:
        overlap = tetrahedron_overlap(unit_tetrahedron, second, budget=budget)
    assert isinstance(overlap, ExactTetrahedronOverlap)
    volume = side**3 / 6
    assert overlap.volume == volume
    assert overlap.first_moment == (
        volume * (shift + side / 4),
        volume * side / 4,
        volume * side / 4,
    )


def _box(lower: tuple[Fraction, ...], upper: tuple[Fraction, ...]) -> ExactConvexCell:
    corners = tuple(
        (
            upper[0] if bits & 1 else lower[0],
            upper[1] if bits & 2 else lower[1],
            upper[2] if bits & 4 else lower[2],
        )
        for bits in range(8)
    )
    edges = tuple(
        (first, first | axis)
        for first in range(8)
        for axis in (1, 2, 4)
        if not first & axis
    )
    planes = tuple(
        plane
        for axis in range(3)
        for plane in (
            tuple(Fraction(-1 if index == axis else 0) for index in range(3))
            + (lower[axis],),
            tuple(Fraction(1 if index == axis else 0) for index in range(3))
            + (-upper[axis],),
        )
    )
    return ExactConvexCell(corners, edges, planes, lower, upper)


def test_convex_cell_overlap_matches_closed_form_box_integrals() -> None:
    zero, one = Fraction(0), Fraction(1)
    first = _box((zero,) * 3, (one,) * 3)
    low = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 4))
    second = _box(low, (Fraction(3, 2), Fraction(4, 3), Fraction(5, 4)))
    assert not convex_cells_disjoint(first, second)
    with coordinate_reference_budget() as budget:
        overlap = convex_cell_overlap(first, second, second_moments=True, budget=budget)
    assert isinstance(overlap, ExactTetrahedronOverlap)
    lengths = tuple(1 - value for value in low)
    volume = lengths[0] * lengths[1] * lengths[2]
    centers = tuple((1 + value) / 2 for value in low)
    squares = tuple((1 - value**3) / (3 * (1 - value)) for value in low)
    assert overlap.volume == volume
    assert overlap.first_moment == tuple(volume * center for center in centers)
    assert overlap.second_moment == tuple(
        tuple(
            volume * (squares[row] if row == column else centers[row] * centers[column])
            for column in range(3)
        )
        for row in range(3)
    )


@pytest.mark.parametrize("offset", (1, 2), ids=("shared-face", "separated"))
def test_convex_cell_overlap_refuses_volume_for_touching_or_separate_boxes(
    offset: int,
) -> None:
    zero, one = Fraction(0), Fraction(1)
    first = _box((zero,) * 3, (one,) * 3)
    second = _box((Fraction(offset), zero, zero), (Fraction(offset + 1), one, one))
    assert convex_cells_disjoint(first, second)
    with coordinate_reference_budget() as budget:
        overlap = convex_cell_overlap(first, second, budget=budget)
    assert isinstance(overlap, ExactTetrahedronOverlap)
    assert overlap.volume == 0
    assert overlap.first_moment == (Fraction(0),) * 3


def _shoelace(points: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    return sum(
        (
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        ),
        Fraction(0),
    ) / 2


@pytest.mark.parametrize(
    "loop",
    (
        ((0, 0), (4, 0), (5, 2), (2, 4), (-1, 2)),
        ((0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)),
    ),
    ids=("convex-pentagon", "nonconvex-l"),
)
def test_loop_triangulation_preserves_signed_area_with_positive_ears(
    loop: tuple[tuple[int, int], ...],
) -> None:
    points = np.asarray(
        tuple((Fraction(x), Fraction(y), Fraction(1)) for x, y in loop), dtype=object
    )
    row = tuple(range(len(loop)))
    with coordinate_reference_budget() as budget:
        triangles = triangulate_loop(points, row, budget=budget)
    assert len(triangles) == len(loop) - 2
    areas = tuple(
        _shoelace(tuple(tuple(points[index][:2]) for index in triangle))
        for triangle in triangles
    )
    assert all(area > 0 for area in areas)
    assert sum(areas) == _shoelace(tuple(tuple(point[:2]) for point in points))


def test_loop_triangulation_refuses_self_intersecting_bowtie() -> None:
    points = np.asarray(
        tuple(
            (Fraction(x), Fraction(y), Fraction(0))
            for x, y in ((0, 0), (2, 2), (2, 0), (0, 2))
        ),
        dtype=object,
    )
    with (
        coordinate_reference_budget() as budget,
        pytest.raises(ValueError, match="simple"),
    ):
        triangulate_loop(points, (0, 1, 2, 3), budget=budget)


def _brute_force_vertices(planes: tuple[Plane, ...], dimension: int) -> set[Point]:
    """Independent oracle: every feasible solution of an independent plane tuple."""
    from itertools import combinations as choose

    def det(rows: list[list[ExactScalar]]) -> ExactScalar:
        if len(rows) == 2:
            return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        return sum(
            (-1) ** column
            * rows[0][column]
            * det([row[:column] + row[column + 1 :] for row in rows[1:]])
            for column in range(3)
        )

    found = set()
    for selected in choose(planes, dimension):
        matrix = [list(normal) for normal, _ in selected]
        base = det(matrix)
        if not base:
            continue
        point = tuple(
            det(
                [
                    [(-offset if axis == column else value) for axis, value in enumerate(row)]
                    for row, (_, offset) in zip(matrix, selected, strict=True)
                ]
            )
            / base
            for column in range(dimension)
        )
        if all(
            sum((a * x for a, x in zip(normal, point, strict=True)), offset) >= 0
            for normal, offset in planes
        ):
            found.add(point)
    return found


def _simplex(*rows: tuple[int | Fraction, ...]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(value) for value in row) for row in rows)


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (
            _simplex((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            _simplex((Fraction(1, 4), Fraction(1, 5), 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)),
        ),
        (
            _simplex((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            _simplex((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)),
        ),
        (
            _simplex((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            _simplex((1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1)),
        ),
        (
            _simplex((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
            _simplex((Fraction(1, 2), Fraction(1, 2), 0), (3, 0, 0), (0, 3, 0), (1, 1, -1)),
        ),
        (
            _simplex((0, 0), (1, 0), (0, 1)),
            _simplex((Fraction(1, 3), Fraction(1, 3)), (2, 0), (0, 2)),
        ),
        (
            _simplex((0, 0), (1, 0), (0, 1)),
            _simplex((1, 0), (0, 1), (1, 1)),
        ),
    ),
    ids=("overlap", "shared-face", "shared-vertex", "coplanar-face-overlap", "triangles", "shared-edge"),
)
def test_simplex_crossing_vertices_match_independent_plane_enumeration(
    first: tuple[tuple[Fraction, ...], ...], second: tuple[tuple[Fraction, ...], ...]
) -> None:
    from phydrax.geometry._affine_mapped_contact import (
        _simplex_crossing_vertices,
        _simplex_planes,
    )

    dimension = len(first[0])
    planes = (*_simplex_planes(first, dimension), *_simplex_planes(second, dimension))
    with coordinate_reference_budget() as budget:
        intersection = _simplex_crossing_vertices(
            first, second, planes, dimension, 10_000, 0, budget=budget
        )
    assert not intersection.exhausted
    assert intersection.points == _brute_force_vertices(planes, dimension)


@pytest.mark.parametrize("dimension", (2, 3))
def test_simplex_crossing_vertices_match_enumeration_on_degenerate_lattice_sweep(
    dimension: int,
) -> None:
    """Small-lattice corners force shared, collinear and coplanar active planes."""
    from random import Random

    from phydrax.geometry._affine_mapped_contact import (
        _simplex_crossing_vertices,
        _simplex_planes,
    )

    generator = Random(20261020 + dimension)

    def full_rank(rows: tuple[tuple[Fraction, ...], ...]) -> bool:
        e = [[row[axis] - rows[0][axis] for axis in range(dimension)] for row in rows[1:]]
        if dimension == 2:
            return e[0][0] * e[1][1] - e[0][1] * e[1][0] != 0
        return (
            e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
            - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
            + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
        ) != 0

    def corners() -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(Fraction(generator.randint(-2, 2), 2) for _ in range(dimension))
            for _ in range(dimension + 1)
        )

    compared = 0
    for _ in range(160):
        first = corners()
        # Reusing corners of the first simplex creates shared faces and edges.
        second = tuple(
            first[generator.randrange(dimension + 1)] if generator.random() < 0.4 else row
            for row in corners()
        )
        if not (full_rank(first) and full_rank(second)):
            continue
        planes = (*_simplex_planes(first, dimension), *_simplex_planes(second, dimension))
        with coordinate_reference_budget() as budget:
            intersection = _simplex_crossing_vertices(
                first, second, planes, dimension, 100_000, 0, budget=budget
            )
        assert not intersection.exhausted
        assert intersection.points == _brute_force_vertices(planes, dimension)
        compared += 1
    assert compared >= 30


def test_convex_hull_proof_admits_coplanar_face_subdivision() -> None:
    from phydrax.discretization import CellMesh
    from phydrax.geometry._exact_polyhedral_geometry import (
        exact_convex_cells,
        star_tetrahedra,
    )

    box = _box((Fraction(0),) * 3, (Fraction(1),) * 3)
    points = np.asarray(box.vertices, dtype=object)
    faces = (
        (0, 2, 3, 1), (4, 5, 7, 6), (0, 4, 6, 2),
        (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3),
    )
    triangles = [
        np.asarray(triangle, dtype=np.int64)
        for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))
    ]
    mesh = CellMesh.from_polyhedra(np.asarray(points, dtype=float), [triangles])
    with coordinate_reference_budget() as budget:
        assert len(star_tetrahedra(mesh, points, budget=budget)[0]) == 12
        convex = exact_convex_cells(mesh, points, budget=budget)[0]
        assert convex is not None
        assert len(convex.planes) == 6
        overlap = convex_cell_overlap(
            convex, box, second_moments=True, retain_simplices=True, budget=budget
        )
    assert isinstance(overlap, ExactTetrahedronOverlap)
    assert overlap.volume == 1
    assert overlap.first_moment == (Fraction(1, 2),) * 3
    assert overlap.second_moment == tuple(
        tuple(Fraction(1, 3 if i == j else 4) for j in range(3)) for i in range(3)
    )
    assert overlap.simplices


@pytest.mark.parametrize("height", (Fraction(0), Fraction(-1)))
def test_supporting_plane_refuses_flat_or_two_sided_vertices(height: Fraction) -> None:
    from phydrax.geometry._exact_polyhedral_geometry import _supporting_plane

    points = np.asarray([
        tuple(Fraction(value) for value in row) for row in (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, height), (0, 0, -height),
        )
    ], dtype=object)
    with coordinate_reference_budget() as budget:
        assert _supporting_plane(points, (0, 1, 2), tuple(range(6)), budget=budget) is None


def _affine_contact_cell(
    points: tuple[tuple[Fraction, ...], ...], identifiers: tuple[int, ...]
) -> Cell:
    from phydrax.geometry._embedding_cells import Cell

    dimension = len(points[0])
    origin = (0,) * dimension
    axes = tuple(
        tuple(int(axis == column) for column in range(dimension))
        for axis in range(dimension)
    )
    coordinates: tuple[Polynomial, ...] = tuple(
        {
            origin: points[0][coordinate],
            **{
                index: points[axis + 1][coordinate] - points[0][coordinate]
                for axis, index in enumerate(axes)
            },
        }
        for coordinate in range(dimension)
    )
    reference = tuple(
        tuple(Fraction(value) for value in row) for row in (origin, *axes)
    )
    return Cell(
        coordinates, "simplex", "triangle" if dimension == 2 else "tetrahedron",
        dimension, identifiers, reference,
    )


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize(
    "case", ("separated", "shared", "oblique-shared", "authored", "oblique-authored", "volume")
)
def test_affine_contact_prepares_trace_authority_only_when_consumed(
    dimension: int, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phydrax.geometry._affine_mapped_contact as contact_owner

    points = tuple(
        tuple(Fraction(int(row == column + 1)) for column in range(dimension))
        for row in range(dimension + 1)
    )
    if case == "separated":
        other = tuple(tuple(value + 3 for value in row) for row in points)
    elif case in ("shared", "oblique-shared", "authored", "oblique-authored"):
        other = (*points[:-1], (*points[-1][:-1], Fraction(-1)))
    else:
        other = points
    if case.startswith("oblique-"):
        points, other = (
            tuple(tuple(value + sum(row) for value in row) for row in bank)
            for bank in (points, other)
        )
    first = _affine_contact_cell(points, tuple(range(dimension + 1)))
    identifiers = (
        (*range(dimension), 2 * dimension + 1)
        if case in ("shared", "oblique-shared") else tuple(range(dimension + 1, 2 * dimension + 2))
    )
    second = _affine_contact_cell(other, identifiers)
    calls, intersections = [], []
    original = contact_owner._intersection_vertices

    def counted(
        first_points: tuple[Point, ...],
        second_points: tuple[Point, ...],
        planes: tuple[Plane, ...],
        dimension: int,
        maximum_work: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> _Intersection:
        intersections.append(None)
        return original(
            first_points, second_points, planes, dimension, maximum_work, budget=budget
        )

    def traces() -> tuple[tuple[int, ...], ...]:
        calls.append(None)
        return (tuple(range(dimension)),)

    monkeypatch.setattr(contact_owner, "_intersection_vertices", counted)
    with coordinate_reference_budget() as budget:
        result = contact_owner.affine_simplex_contact(
            first, second, 10000, trace_preparation=traces, budget=budget
        )
    assert result is not None
    assert result.status == ("overlap" if case == "volume" else "separated")
    assert len(intersections) == int(case not in ("separated", "shared", "authored"))
    assert len(calls) == int(case in ("authored", "oblique-authored", "volume"))


def test_lazy_affine_contact_refuses_invalid_authored_simplex_trace() -> None:
    from phydrax.geometry._affine_mapped_contact import affine_simplex_contact

    points = _simplex((0, 0), (1, 0), (0, 1))
    first = _affine_contact_cell(points, (0, 1, 2))
    second = _affine_contact_cell(points, (3, 4, 5))
    with coordinate_reference_budget() as budget:
        with pytest.raises(ValueError, match="proper first-simplex traces"):
            affine_simplex_contact(
                first, second, 10000,
                trace_preparation=lambda: ((0, 1, 2),),
                budget=budget,
            )


def test_affine_corner_cache_binds_full_source_values_and_reference_points() -> None:
    from dataclasses import replace

    from phydrax.geometry._affine_mapped_contact import _cell_images

    points = _simplex(
        (Fraction(1, 3), 0), (Fraction(4, 3), 0), (Fraction(1, 3), 1)
    )
    cell = _affine_contact_cell(points, (0, 1, 2))
    with coordinate_reference_budget() as budget:
        first = _cell_images(cell, budget=budget)
        work, retained = budget.work_units, budget.retained_basis_bytes
        renamed = replace(cell, vertices=(3, 4, 5))
        assert _cell_images(renamed, budget=budget) is first
        assert budget.work_units == work
        assert budget.retained_basis_bytes == retained
        delta = Fraction(1, 2**100)
        coordinate = cell.coordinates[0]
        assert isinstance(coordinate, dict)
        coordinate[(0, 0)] += delta
        changed = _cell_images(cell, budget=budget)
        assert changed == tuple((point[0] + delta, point[1]) for point in first)
        np.testing.assert_array_equal(
            np.asarray(changed, dtype=float), np.asarray(first, dtype=float)
        )
        assert budget.work_units > work
        other_reference = replace(
            cell,
            reference_vertices=_simplex((0, 0), (1, 0), (1, 1)),
        )
        sampled = _cell_images(other_reference, budget=budget)
        assert sampled[-1] == (Fraction(4, 3) + delta, Fraction(1))
    assert retained > 0


@pytest.mark.parametrize(
    ("support", "traces", "covered"),
    (
        ((1,), ((1, 2, 3),), True),
        ((2, 1), ((1, 2, 3),), True),
        ((1, 2, 3), ((1, 2), (2, 3)), False),
        ((4,), ((1, 2, 3),), False),
    ),
)
def test_authored_trace_subsumption_requires_one_convex_trace(
    support: tuple[int, ...], traces: tuple[tuple[int, ...], ...], covered: bool
) -> None:
    from phydrax.geometry._periodic_embedding import _trace_already_permitted

    assert _trace_already_permitted(support, dict.fromkeys(traces)) is covered


def test_affine_corner_images_follow_nested_polynomial_preparation_lifetimes() -> None:
    from phydrax.geometry._affine_mapped_contact import _cell_images

    cells = tuple(
        _affine_contact_cell(
            _simplex((offset, 0), (offset + 1, 0), (offset, 1)), (0, 1, 2)
        )
        for offset in range(3)
    )
    with coordinate_reference_budget() as budget:
        root = _cell_images(cells[0], budget=budget)
        retained, temporary = budget.retained_basis_bytes, budget.temporary_bytes_upper
        with budget.polynomial_preparation_scope():
            first = _cell_images(cells[1], budget=budget)
            assert _cell_images(cells[0], budget=budget) is root
            outer_bytes = budget.temporary_bytes_upper
            assert len(budget.affine_cell_corner_images_cache) == 2
            with budget.polynomial_preparation_scope():
                _cell_images(cells[2], budget=budget)
                assert _cell_images(cells[1], budget=budget) is first
                assert len(budget.affine_cell_corner_images_cache) == 3
            assert len(budget.affine_cell_corner_images_cache) == 2
            assert budget.temporary_bytes_upper == outer_bytes
        assert len(budget.affine_cell_corner_images_cache) == 1
        assert budget.retained_basis_bytes == retained
        assert budget.temporary_bytes_upper == temporary


@pytest.mark.parametrize("turns", (1, 2, 3))
def test_inverse_candidate_bank_covers_exact_inverse_contacts(turns: int) -> None:
    from phydrax.geometry._affine_mapped_contact import _simplex_planes
    from phydrax.geometry._embedding_state import boxes, candidate_pairs
    from phydrax.geometry._periodic_embedding import _transpose_image_candidates

    cells = (
        _simplex((0, 0), (1, 0), (0, 1)),
        _simplex((1, 0), (2, 0), (1, 1)),
        _simplex((Fraction(-1, 2), 0), (Fraction(1, 2), 0), (0, 1)),
    )

    def action(point: tuple[Fraction, ...], inverse: bool = False) -> tuple[Fraction, Fraction]:
        x, y = point
        for _ in range(turns):
            x, y = (
                (y - Fraction(1, 4), Fraction(1, 2) - x)
                if inverse else (-y + Fraction(1, 2), x + Fraction(1, 4))
            )
        return x, y

    points = np.asarray(tuple(point for cell in cells for point in cell), dtype=object)
    moved = np.asarray(tuple(action(point) for point in points), dtype=object)
    indices = np.arange(len(points)).reshape((-1, 3))
    first, second, exceeded = candidate_pairs(
        *boxes(points, indices), 1000, other=boxes(moved, indices)
    )
    assert not exceeded
    rows = tuple(
        tuple(sorted(int(value) for value in first[second == index]))
        for index in range(len(cells))
    )
    with coordinate_reference_budget() as budget:
        inverse_rows = _transpose_image_candidates(rows, budget=budget)
        assert _transpose_image_candidates(inverse_rows, budget=budget) == rows
    candidates = {
        (first, second) for second, row in enumerate(inverse_rows) for first in row
    }
    for a, first_cell in enumerate(cells):
        for b, second_cell in enumerate(cells):
            inverse_cell = tuple(action(point, inverse=True) for point in second_cell)
            planes = (*_simplex_planes(first_cell, 2), *_simplex_planes(inverse_cell, 2))
            if _brute_force_vertices(planes, 2):
                assert (a, b) in candidates


@pytest.mark.parametrize(
    "rotation",
    (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((Fraction(3, 5), Fraction(-4, 5), 0), (Fraction(4, 5), Fraction(3, 5), 0), (0, 0, 1)),
    ),
)
def test_earned_plane_transport_matches_independent_vertex_reconstruction(
    rotation: tuple[tuple[int | Fraction, ...], ...],
) -> None:
    from phydrax.geometry._periodic_embedding import (
        _transport_source_plane,
        source_polygon_plane,
    )

    points = _simplex(
        (0, 0, 0), (1, 0, Fraction(-1, 3)),
        (1, 1, -1), (0, 1, Fraction(-2, 3)),
    )
    action = tuple(
        tuple(Fraction(value) for value in row) + (shift,)
        for row, shift in zip(
            rotation, (Fraction(-2, 3), Fraction(5, 7), Fraction(1, 2)), strict=True
        )
    ) + ((Fraction(0), Fraction(0), Fraction(0), Fraction(1)),)
    with coordinate_reference_budget() as budget:
        source = source_polygon_plane(points, budget=budget)
        assert source is not None
        before = budget.work_units
        transported = _transport_source_plane(source[0], action, budget=budget)
        prepared_work = budget.work_units - before
        before = budget.work_units
        budget.reserve(len(points) * 3 * 4)
        transformed = tuple(
            tuple(
                sum((a * b for a, b in zip(row[:-1], point, strict=True)), row[-1])
                for row in action[:-1]
            )
            for point in points
        )
        expected = source_polygon_plane(transformed, budget=budget)
        assert expected is not None and transported == expected[0]
        assert prepared_work < budget.work_units - before


@pytest.mark.parametrize("reverse", (False, True))
def test_convex_source_containment_matches_full_concave_boundary_algorithm(reverse: bool) -> None:
    import phydrax.geometry._periodic_embedding as owner

    square = _simplex((0, 0), (1, 0), (2, 0), (2, 2), (0, 2))
    concave = _simplex((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))
    subjects = (
        _simplex((Fraction(7, 4), Fraction(3, 4)), (Fraction(3, 4), Fraction(7, 4))),
        _simplex((Fraction(1, 4), Fraction(1, 4)), (Fraction(7, 4), Fraction(3, 4))),
        _simplex((0, 0), (2, 0), (1, 1)),
        _simplex((3, 0), (3, 1)),
    )

    def reference(
        source: CoordinateSourceBank, subject: CoordinateSourceBank, budget: CoordinateEnclosureBudget
    ) -> bool:
        if not all(owner._closed_polygon_point(source, point, budget=budget) for point in subject):
            return False
        for a, b in zip(subject, (*subject[1:], subject[0]), strict=True):
            if a == b:
                continue
            ordered = owner._source_edge_parameters(source, a, b, budget=budget)
            for low, high in zip(ordered, ordered[1:]):
                middle = (low + high) / 2
                point = tuple(x + middle * (y - x) for x, y in zip(a, b, strict=True))
                if not owner._closed_polygon_point(source, point, budget=budget):
                    return False
        return True

    with coordinate_reference_budget() as budget:
        for source, convex in ((square, True), (concave, False)):
            source = source[::-1] if reverse else source
            for subject in subjects:
                assert owner.closed_source_polygon_contains(
                    source, subject, budget=budget
                ) is reference(source, subject, budget)
            assert budget.simple_source_polygon_convexity_cache[source] is convex
        assert owner.closed_source_polygon_contains(square, subjects[0], budget=budget)
        assert not owner.closed_source_polygon_contains(concave, subjects[0], budget=budget)
