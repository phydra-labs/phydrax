#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import overload, TYPE_CHECKING


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from itertools import combinations

import numpy as np

import phydrax.discretization._coordinate_budget as coordinate_budget_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResource,
)
from phydrax.geometry._planar_primitives import (
    convex_hull,
    plane_key,
    project,
    signed_measure,
    turn,
)

from .._geometry_predicates import bigint_bytes, exact_bits, exact_charge, exact_reserve
from .._strict import StrictModule
from ..discretization._cell_complex import PolyhedralConnectivity
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ..typing import checked


type Point = tuple[Fraction, ...]
type Tetrahedron = tuple[Point, Point, Point, Point]


def determinant3(
    first: Point,
    second: Point,
    third: Point,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:

    budget.reserve(4 * 3**3)
    result = prepare_exact_small_linear_actions(
        (first, second, third), ((), (), ()), budget=budget
    )
    budget.ticket.consume(result.operation_count)
    return result.determinant


def exact_vertices(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    geometry.resolve(mesh, exact_source_prepared=True, budget=budget)
    return np.asarray(geometry.source_coordinates(budget=budget), dtype=object)


def _between(point: Point, a: Point, b: Point, /) -> bool:
    return all(min(x, y) <= p <= max(x, y) for p, x, y in zip(point, a, b, strict=True))


def _segment_hit(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    values = (
        turn(a, b, c, budget=budget),
        turn(a, b, d, budget=budget),
        turn(c, d, a, budget=budget),
        turn(c, d, b, budget=budget),
    )
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    return any(
        value == 0 and _between(point, first, second)
        for value, point, first, second in (
            (values[0], c, a, b),
            (values[1], d, a, b),
            (values[2], a, c, d),
            (values[3], b, c, d),
        )
    )


def triangulation_charge(
    points: np.ndarray,
    row: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int]:
    """Bound work and storage of ``triangulate_loop`` for one rational loop.

    Work counts rational operations per fixed branch of ``triangulate_loop``
    for ``m`` vertices: at most two plane keys (29 operations each) per
    candidate corner, ``3 m`` hashed coordinates, ``8 m`` planarity
    operations, 84 per nonadjacent edge pair test (four turns, two sign
    products, the closed-box fallback), ``4 m + 3`` for the signed measure and
    ``r (9 + 27 r)`` per ear pass over ``r`` remaining vertices; the input
    bit scan charges itself. A Fraction sum or product has at most
    ``bits(a) + bits(b) + 1`` bits before reduction; composing it through the
    plane key, planarity residual (the deepest chain), turns and shoelace sum
    of inputs of at most ``b`` bits bounds every value and every pre-reduction
    temporary by ``max(102 b + 67, m (4 b + 3) + 1)`` bits. Storage keeps two
    such integers and a Fraction header for every operation result, plus seven
    transient integers of one operation.
    """
    count = len(row)
    bits = exact_bits(points[list(row)], budget=budget)
    return triangulation_bounds(count, bits)


def triangulation_bounds(count: int, bits: int, /) -> tuple[int, int]:
    """Return loop-triangulation work/storage bounds from validated metadata."""
    work = (
        58 * max(count - 2, 0)
        + 3 * count
        + 8 * count
        + 84 * (count * max(count - 3, 0) // 2)
        + 4 * count
        + 3
        + sum(remaining * (9 + 27 * remaining) for remaining in range(4, count + 1))
    )
    integer = bigint_bytes(max(102 * bits + 67, count * (4 * bits + 3) + 1))
    return work, work * (2 * integer + 64) + 7 * integer


def triangulate_loop(
    points: np.ndarray,
    row: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate after rational planarity and simple-loop proofs."""
    vertices = tuple(tuple(value for value in points[index]) for index in row)
    key = next(
        (
            plane_key((vertices[0], vertices[index], vertices[index + 1]), budget=budget)
            for index in range(1, len(vertices) - 1)
            if plane_key(
                (vertices[0], vertices[index], vertices[index + 1]), budget=budget
            )
            is not None
        ),
        None,
    )
    if key is None or len(set(vertices)) != len(vertices):
        raise ValueError(
            "polyhedral.facet.vertices must be distinct with plane rank two; got repeated vertices or deficient rank"
        )
    plane, axes, _, _ = key
    if any(
        sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) != 0
        for point in vertices
    ):
        raise ValueError(
            "polyhedral.facet.vertices must lie in one plane; got nonzero plane residual"
        )
    projected = project(vertices, axes, budget=budget)
    count = len(row)
    for first in range(count):
        for second in range(first + 1, count):
            if second == first + 1 or (first == 0 and second == count - 1):
                continue
            if _segment_hit(
                projected[first],
                projected[(first + 1) % count],
                projected[second],
                projected[(second + 1) % count],
                budget=budget,
            ):
                raise ValueError(
                    "polyhedral.facet.loop must be simple; got intersecting nonadjacent edges"
                )
    measure = signed_measure(projected, budget=budget)
    if measure == 0:
        raise ValueError(
            "polyhedral.facet.loop must have nonzero signed area; got zero area"
        )
    sign = 1 if measure > 0 else -1
    remaining, triangles = list(range(count)), []
    while len(remaining) > 3:
        for position, middle in enumerate(remaining):
            previous, following = (
                remaining[position - 1],
                remaining[(position + 1) % len(remaining)],
            )
            a, b, c = projected[previous], projected[middle], projected[following]
            if sign * turn(a, b, c, budget=budget) <= 0:
                continue
            if any(
                all(
                    sign * turn(x, y, projected[index], budget=budget) >= 0
                    for x, y in ((a, b), (b, c), (c, a))
                )
                for index in remaining
                if index not in (previous, middle, following)
            ):
                continue
            triangles.append((row[previous], row[middle], row[following]))
            remaining.pop(position)
            break
        else:
            raise ValueError(
                "polyhedral.facet.loop must admit oriented ears; got an untriangulable remainder"
            )
    triangles.append(tuple(row[index] for index in remaining))
    return tuple(triangles)


@dataclass(slots=True)
class StarTetrahedronPreparation:
    """Reuse signed-volume proofs for source-bound star pieces in one request."""

    budget: CoordinateEnclosureBudget
    determinants: dict[Tetrahedron, Fraction] = field(default_factory=dict)

    def determinant(self, tet: Tetrahedron, /) -> Fraction:
        cached = self.determinants.get(tet)
        if cached is not None:
            return cached
        columns = tuple(
            tuple(value - base for value, base in zip(point, tet[0], strict=True))
            for point in tet[1:]
        )
        determinant = determinant3(*columns, budget=self.budget)
        self.budget.retain_basis((tet, determinant))
        self.determinants[tet] = determinant
        return determinant


def star_tetrahedra(
    mesh: CellMesh,
    points: np.ndarray,
    /,
    *,
    require_positive: bool = True,
    budget: CoordinateEnclosureBudget,
    preparation: StarTetrahedronPreparation | None = None,
) -> tuple[tuple[Tetrahedron, ...], ...]:
    if preparation is not None and preparation.budget is not budget:
        raise ValueError(
            "star_tetrahedra.preparation must belong to the current request budget; got another budget"
        )
    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            pass
        case _:
            raise TypeError(
                "polyhedral.mesh.connectivity must contain packed faces; got another connectivity family"
            )
    offsets = np.asarray(connectivity.cell_face_offsets)
    face_offsets, faces = (
        np.asarray(connectivity.face_vertex_offsets),
        np.asarray(connectivity.face_vertex_values),
    )
    cell_faces, signs = (
        np.asarray(connectivity.cell_face_values),
        np.asarray(connectivity.cell_face_sign_values),
    )
    result = []
    for cell in range(offsets.size - 1):
        occurrences = tuple(range(offsets[cell], offsets[cell + 1]))
        indices = tuple(
            sorted(
                set(
                    index
                    for occurrence in occurrences
                    for index in faces[
                        face_offsets[cell_faces[occurrence]] : face_offsets[
                            cell_faces[occurrence] + 1
                        ]
                    ].tolist()
                )
            )
        )
        center = tuple(
            sum((points[index, axis] for index in indices), Fraction(0)) / len(indices)
            for axis in range(3)
        )
        pieces = []
        for occurrence in occurrences:
            face = cell_faces[occurrence]
            row = tuple(faces[face_offsets[face] : face_offsets[face + 1]].tolist())
            if signs[occurrence] < 0:
                row = row[::-1]
            for triangle in triangulate_loop(points, row, budget=budget):
                corners = tuple(tuple(points[index]) for index in triangle)
                tetrahedron = (center, corners[0], corners[1], corners[2])
                if (
                    require_positive
                    and (
                        determinant3(
                            *(
                                tuple(
                                    value - base
                                    for value, base in zip(point, center, strict=True)
                                )
                                for point in corners
                            ),
                            budget=budget,
                        )
                        if preparation is None
                        else preparation.determinant(tetrahedron)
                    )
                    <= 0
                ):
                    raise ValueError(
                        "polyhedral.cell.star must contain positive-volume tetrahedra; got nonpositive determinant"
                    )
                pieces.append(tetrahedron)
        result.append(tuple(pieces))
    return tuple(result)


def tetrahedron_planes(
    tet: Tetrahedron,
    visit: Callable[[int], None],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Point, ...]:
    """Return outward face planes and charge their scalar side tests.

    ``plane_key`` charges plane construction. The opposite-vertex test uses
    seven rational operations, and an orientation flip uses four.
    """

    planes = []
    for opposite in range(4):
        points = tuple(tet[index] for index in range(4) if index != opposite)
        key = plane_key(points, budget=budget)
        if key is None:
            raise ValueError("tetrahedron.face must have rank two; got deficient rank")
        plane = key[0]
        visit(7)
        side = sum((plane[axis] * tet[opposite][axis] for axis in range(3)), plane[3])
        if side == 0:
            raise ValueError(
                "tetrahedron.vertices must have positive volume rank; got zero"
            )
        if side > 0:
            visit(4)
            planes.append(tuple(-value for value in plane))
        else:
            planes.append(plane)
    return tuple(planes)


@dataclass(slots=True)
class TetrahedronPlanePreparation:
    """Retain earned face planes under rational coordinates in one request."""

    budget: CoordinateEnclosureBudget
    planes: dict[Tetrahedron, tuple[Point, ...]] = field(default_factory=dict)

    def prepare(
        self, tet: Tetrahedron, visit: Callable[[int], None], /
    ) -> tuple[Point, ...]:
        cached = self.planes.get(tet)
        if cached is not None:
            return cached
        planes = tetrahedron_planes(tet, visit, budget=self.budget)
        self.budget.retain_basis((tet, planes))
        self.planes[tet] = planes
        return planes


class ExactOverlapStatus(StrEnum):
    SUCCESS = "success"
    RESOURCE_LIMIT = "resource_limit"


class ExactTetrahedronOverlap(StrictModule):
    volume: Fraction
    first_moment: Point
    second_moment: tuple[Point, ...] | None
    simplices: tuple[Tetrahedron, ...] | None
    operation_count: int
    status: ExactOverlapStatus = ExactOverlapStatus.SUCCESS


class ExactOverlapRefusal(StrictModule):
    """Return the coefficient-resource refusal without an overlap payload."""

    resource: CoordinateEnclosureResource
    operation_count: int
    status: ExactOverlapStatus = ExactOverlapStatus.RESOURCE_LIMIT


@dataclass(slots=True)
class _OverlapWork:
    budget: CoordinateEnclosureBudget
    maximum_work: int | None
    starting_work: int
    work: int = 0

    def actual_work(self) -> int:
        return self.work + self.budget.work_units - self.starting_work

    def admit(self, count: int) -> None:
        completed = self.actual_work()
        if self.maximum_work is not None and completed + count > self.maximum_work:
            # Source-owned coefficient operations may have consumed the local
            # allowance before this visit; retain that work as admission evidence.
            raise coordinate_budget_owner.CoordinateEnclosureResourceError(
                "coefficient_work",
                self.maximum_work,
                completed + count,
                completed,
                admission=completed > self.maximum_work or count == 0,
            )

        if count > self.budget.ticket.remaining_work_units:
            self.budget.ticket.refuse(count)

    def visit(self, count: int) -> None:
        self.admit(count)
        self.budget.ticket.consume(count)
        self.work += count

    def solve(
        self, matrix: tuple[Point, ...], right: tuple[Point, ...]
    ) -> tuple[tuple[Point, ...] | None, Fraction]:
        self.admit(4 * 3**3)
        result = prepare_exact_small_linear_actions(matrix, right, budget=self.budget)
        self.budget.ticket.consume(result.operation_count)
        self.work += result.operation_count
        return result.actions, result.determinant


class _OverlapPolytope(StrictModule):
    planes: tuple[Point, ...]
    vertices: tuple[Point, ...]
    center: Point


def tetrahedra_disjoint(first: Tetrahedron, second: Tetrahedron, /) -> bool:
    """Return whether axis bounds exclude a positive-volume intersection."""
    return any(
        max(
            min(point[axis] for point in first),
            min(point[axis] for point in second),
        )
        >= min(
            max(point[axis] for point in first),
            max(point[axis] for point in second),
        )
        for axis in range(3)
    )


def _overlap_planes_separate(
    first: Tetrahedron,
    second: Tetrahedron,
    first_planes: tuple[Point, ...],
    second_planes: tuple[Point, ...],
    work: _OverlapWork,
    /,
) -> bool:
    for own_planes, other in ((first_planes, second), (second_planes, first)):
        for plane in own_planes:
            sides = []
            for point in other:
                work.visit(7)
                sides.append(
                    sum((plane[axis] * point[axis] for axis in range(3)), plane[3])
                )
            if min(sides) >= 0:
                return True
    return False


def _overlap_polytope(
    planes: tuple[Point, ...], work: _OverlapWork, /
) -> _OverlapPolytope | None:
    vertices: set[Point] = set()
    work.admit(1)
    for selected in combinations(planes, 3):
        matrix = tuple(tuple(plane[:3]) for plane in selected)
        right = tuple((-plane[3],) for plane in selected)
        actions, _ = work.solve(matrix, right)
        if actions is None:
            continue
        point = tuple(row[0] for row in actions)
        inside = True
        for plane in planes:
            work.visit(7)
            if sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) > 0:
                inside = False
                break
        if inside:
            vertices.add(point)
    if len(vertices) < 4:
        return None
    work.visit(3 * (len(vertices) + 1))
    center = tuple(
        sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
        for axis in range(3)
    )
    return _OverlapPolytope(planes, tuple(vertices), center)


@checked
def tetrahedron_overlap(
    first: Tetrahedron,
    second: Tetrahedron,
    /,
    *,
    second_moments: bool = False,
    retain_simplices: bool = False,
    maximum_work: int | None = None,
    budget: CoordinateEnclosureBudget,
    preparation: TetrahedronPlanePreparation | None = None,
) -> ExactTetrahedronOverlap | ExactOverlapRefusal:
    """Return intersection integrals or a typed coefficient-resource refusal."""
    if preparation is not None and preparation.budget is not budget:
        raise ValueError(
            "tetrahedron_overlap.preparation must belong to the current request budget; got another budget"
        )
    work = _OverlapWork(budget, maximum_work, budget.work_units)
    try:
        with budget.measurement(), budget.temporary_scope():
            first_planes = (
                tetrahedron_planes(first, work.visit, budget=budget)
                if preparation is None
                else preparation.prepare(first, work.visit)
            )
            second_planes = (
                tetrahedron_planes(second, work.visit, budget=budget)
                if preparation is None
                else preparation.prepare(second, work.visit)
            )
            if _overlap_planes_separate(first, second, first_planes, second_planes, work):
                polytope = None
            else:
                planes = tuple(sorted(set((*first_planes, *second_planes))))
                polytope = _overlap_polytope(planes, work)
            if polytope is None:
                return ExactTetrahedronOverlap(
                    Fraction(0),
                    (Fraction(0),) * 3,
                    ((Fraction(0),) * 3,) * 3 if second_moments else None,
                    () if retain_simplices else None,
                    work.actual_work(),
                )
            return _overlap_integrals(polytope, work, second_moments, retain_simplices)
    except coordinate_budget_owner.CoordinateEnclosureResourceError as error:
        return ExactOverlapRefusal(error.resource, work.actual_work())


def _accumulate_overlap_moments(
    corners: Tetrahedron,
    measure: Fraction,
    moment: list[Fraction],
    second_moment: list[list[Fraction]] | None,
    work: _OverlapWork,
    /,
) -> None:
    """Accumulate barycentric moments in axis order, including their work charges."""
    for axis in range(3):
        work.visit(6)
        moment[axis] += measure * sum((point[axis] for point in corners), Fraction(0)) / 4
    if second_moment is not None:
        work.visit(3 * 4)
        sums = tuple(
            sum((point[axis] for point in corners), Fraction(0)) for axis in range(3)
        )
        for first_axis in range(3):
            for second_axis in range(3):
                work.visit(11)
                second_moment[first_axis][second_axis] += (
                    measure
                    * (
                        sums[first_axis] * sums[second_axis]
                        + sum(
                            (point[first_axis] * point[second_axis] for point in corners),
                            Fraction(0),
                        )
                    )
                    / 20
                )


def _overlap_integrals(
    polytope: _OverlapPolytope,
    work: _OverlapWork,
    second_moments: bool,
    retain_simplices: bool,
    /,
) -> ExactTetrahedronOverlap:
    """Integrate the polytope's oriented tetrahedral fan.

    Simplex first and second moments use barycentric means 1/4, 1/10 and
    1/20. Keep the face, corner, axis and charge order coupled to these formulas.
    """
    planes, vertices, center = polytope.planes, polytope.vertices, polytope.center
    budget = work.budget
    volume, moment = Fraction(0), [Fraction(0)] * 3
    second_moment = [[Fraction(0)] * 3 for _ in range(3)] if second_moments else None
    simplices = [] if retain_simplices else None
    for plane in planes:
        work.visit(7 * len(vertices))
        face = tuple(
            point
            for point in vertices
            if sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) == 0
        )
        if len(face) < 3:
            continue
        axes = ((1, 2), (2, 0), (0, 1))[next(axis for axis in range(3) if plane[axis])]
        projected = project(face, axes, budget=budget)
        hull = convex_hull(projected, budget=budget)
        lookup = dict(zip(projected, face, strict=True))
        ordered = tuple(lookup[point] for point in hull)
        for index in range(1, len(ordered) - 1):
            corners = (center, ordered[0], ordered[index], ordered[index + 1])
            columns = tuple(
                tuple(value - base for value, base in zip(point, center, strict=True))
                for point in corners[1:]
            )
            _, signed = work.solve(columns, ((Fraction(0),),) * 3)
            work.visit(12)
            measure = abs(signed) / 6
            if measure == 0:
                continue
            if simplices is not None:
                simplices.append(
                    corners
                    if signed > 0
                    else (corners[0], corners[1], corners[3], corners[2])
                )
            volume += measure
            _accumulate_overlap_moments(corners, measure, moment, second_moment, work)
    return ExactTetrahedronOverlap(
        volume,
        tuple(moment),
        None if second_moment is None else tuple(tuple(row) for row in second_moment),
        None if simplices is None else tuple(simplices),
        work.actual_work(),
    )


def _exact_value(value: Fraction | float, /) -> Fraction:
    match value:
        case Fraction():
            return value
        case _:
            return Fraction(float(value))


@overload
def coordinate_integer_profile(
    points: np.ndarray,
    /,
    maximum_bits: None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int]: ...


@overload
def coordinate_integer_profile(
    points: np.ndarray,
    /,
    maximum_bits: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int] | None: ...


@overload
def coordinate_integer_profile(
    points: np.ndarray,
    /,
    maximum_bits: int | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int] | None: ...


def coordinate_integer_profile(
    points: np.ndarray,
    /,
    maximum_bits: int | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int] | None:
    """Return the coordinate denominator ``D`` and a bit bound on ``points * D``.

    ``bits(n * (D // d)) <= bits(D) + bits(n) - bits(d) + 1``. With
    ``maximum_bits`` the profile is ``None`` as soon as that bound must exceed
    it: each step ``D * (d // gcd(D, d))`` is refused before it is formed when
    even its smallest possible size is too large, so the running multiple never
    exceeds ``maximum_bits + 1`` bits. Both passes are charged before they run
    (one visit per coordinate each) with the multiple's transient bound.
    """
    with exact_charge(points.size, budget=budget):
        denominator_bits, excess = 0, 0
        for value in points.flat:
            exact = _exact_value(value)
            denominator_bits += exact.denominator.bit_length()
            excess = max(
                excess, exact.numerator.bit_length() - exact.denominator.bit_length() + 1
            )
    multiple_bits = (
        denominator_bits
        if maximum_bits is None
        else min(denominator_bits, maximum_bits + 1)
    )
    with exact_charge(points.size, 3 * bigint_bytes(multiple_bits), budget=budget):
        denominator = 1
        for value in points.flat:
            factor = _exact_value(value).denominator
            factor //= math.gcd(denominator, factor)
            if (
                maximum_bits is not None
                and denominator.bit_length() + factor.bit_length() - 1 + excess
                > maximum_bits
            ):
                return None
            denominator *= factor
            if (
                maximum_bits is not None
                and denominator.bit_length() + excess > maximum_bits
            ):
                return None
    return denominator, denominator.bit_length() + excess


def coordinate_integers(
    points: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, Fraction]:
    """Integers ``N`` and the scale ``1 / D`` with ``points == N / D`` exactly.

    ``D`` is the least common denominator. The scaled bank is charged to the
    active coordinate ledger before it is formed and stays charged while the
    caller's enclosing scope retains it.
    """
    profile = coordinate_integer_profile(points, budget=budget)
    denominator, bits = profile
    exact_reserve(points.size, points.size * bigint_bytes(bits), budget=budget)
    return np.asarray(
        tuple(
            tuple(
                _exact_value(value).numerator
                * (denominator // _exact_value(value).denominator)
                for value in row
            )
            for row in points
        ),
        dtype=object,
    ), Fraction(1, denominator)


def _triangle_contains(
    point: Point,
    triangle: tuple[Point, ...],
    axes: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    polygon = project(triangle, axes, budget=budget)
    projected = project((point,), axes, budget=budget)[0]
    sign = 1 if signed_measure(polygon, budget=budget) > 0 else -1
    return all(
        sign * turn(a, b, projected, budget=budget) >= 0
        for a, b in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
    )


def triangle_contact(
    points: np.ndarray,
    first: tuple[int, ...],
    second: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Detect rational triangle contact outside the shared incidence simplex."""

    shared = tuple(sorted(set(first) & set(second)))
    if len(shared) == 3:
        return False
    a, b = (
        tuple(tuple(points[index]) for index in first),
        tuple(tuple(points[index]) for index in second),
    )
    ka, kb = plane_key(a, budget=budget), plane_key(b, budget=budget)
    if ka is None or kb is None:
        raise ValueError(
            "triangle_contact.triangles must have rank two; got degenerate triangles"
        )
    coplanar = all(
        sum((ka[0][axis] * point[axis] for axis in range(3)), ka[0][3]) == 0
        for point in b
    )
    if (
        coplanar
        and planar_primitives_owner.intersection(
            project(a, ka[1], budget=budget),
            project(b, ka[1], budget=budget),
            budget=budget,
        )
        > 0
    ):
        return True
    hits: set[Point] = set()
    for own, other, key in ((a, b, kb), (b, a, ka)):
        _triangle_edge_hits(own, other, key, coplanar, hits, budget=budget)
    return _contact_outside_shared(points, shared, hits)


def _triangle_edge_hits(
    own: tuple[Point, ...],
    other: tuple[Point, ...],
    key: tuple[Point, tuple[int, ...], int, int],
    coplanar: bool,
    hits: set[Point],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    plane, axes, _, _ = key
    for p, q in zip(own, (*own[1:], own[0]), strict=True):
        low = sum((plane[axis] * p[axis] for axis in range(3)), plane[3])
        high = sum((plane[axis] * q[axis] for axis in range(3)), plane[3])
        if low == 0 and _triangle_contains(p, other, axes, budget=budget):
            hits.add(p)
        if low * high < 0:
            ratio = low / (low - high)
            point = tuple(x + ratio * (y - x) for x, y in zip(p, q, strict=True))
            if _triangle_contains(point, other, axes, budget=budget):
                hits.add(point)
        if coplanar:
            for r, s in zip(other, (*other[1:], other[0]), strict=True):
                pp, qq, rr, ss = project((p, q, r, s), axes, budget=budget)
                first_side, second_side = (
                    turn(rr, ss, pp, budget=budget),
                    turn(rr, ss, qq, budget=budget),
                )
                if (
                    first_side * second_side < 0
                    and turn(pp, qq, rr, budget=budget) * turn(pp, qq, ss, budget=budget)
                    <= 0
                ):
                    ratio = first_side / (first_side - second_side)
                    hits.add(
                        tuple(x + ratio * (y - x) for x, y in zip(p, q, strict=True))
                    )


def _contact_outside_shared(
    points: np.ndarray, shared: tuple[int, ...], hits: set[Point], /
) -> bool:
    for hit in hits:
        if not shared:
            return True
        if len(shared) == 1 and hit != tuple(points[shared[0]]):
            return True
        if len(shared) == 2:
            p, q = tuple(points[shared[0]]), tuple(points[shared[1]])
            direction, delta = (
                tuple(y - x for x, y in zip(p, q, strict=True)),
                tuple(y - x for x, y in zip(p, hit, strict=True)),
            )
            if not _between(hit, p, q) or any(
                direction[i] * delta[j] != direction[j] * delta[i]
                for i, j in combinations(range(3), 2)
            ):
                return True
    return False


__all__ = [
    "exact_vertices",
    "triangulate_loop",
    "star_tetrahedra",
    "tetrahedron_overlap",
    "determinant3",
    "coordinate_integers",
]
