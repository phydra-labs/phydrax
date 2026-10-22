#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import overload, TYPE_CHECKING


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    CoordinateLiveStorage,
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
from .._host_storage import declared_integer_storage_bytes
from ..discretization._cell_complex import PolyhedralConnectivity
from ..linalg._small_batched import exact_cofactor_determinant_3
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
    """Return the exact cofactor determinant, charging its fourteen operations once.

    The expanded numerator or common denominator of the six three-factor terms
    involves at most all nine input numerators and denominators, so eighteen
    operand occurrences bound every intermediate rational.
    """
    planar_primitives_owner.reserve_fraction_work(
        (first, second, third), 14, 1, 18, budget=budget
    )
    return exact_cofactor_determinant_3(first, second, third)


def exact_vertices(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Return the exact source vertex bank retained once per request geometry."""
    from ..discretization._coordinate_enclosure import prepared_coordinate_source_bank

    geometry.resolve(mesh, exact_source_prepared=True, budget=budget)
    return np.asarray(prepared_coordinate_source_bank(geometry, budget=budget), dtype=object)


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
    budget.reserve(2)
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    if 0 in values:
        # Closed-box fallback: six comparisons for each collinear endpoint.
        budget.reserve(24)
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
    """Triangulate after rational planarity and simple-loop proofs.

    A strictly convex loop with turning number one is simple, so its proof
    needs only the cyclic turns and direction crossings; its triangles are the
    ears that clipping selects for that loop. Other loops use the general
    pairwise simplicity proof and ear clipping. Every rational operation is
    charged here as performed; ``triangulation_charge`` only bounds them.

    Every decision reads only the exact coordinates in loop order, never row
    ids, so a completed proof is reused within the request under that full
    ordered key; each lookup charges hashing its ``3 m`` coordinates.
    """
    vertices = tuple(tuple(value for value in points[index]) for index in row)
    budget.reserve(3 * len(vertices))
    cache = budget.loop_triangulation_cache
    local = cache.get(vertices)
    if local is None:
        local = _triangulate_positions(vertices, budget=budget)
        budget.retain_basis((vertices, local))
        cache[vertices] = local
    return tuple(tuple(row[index] for index in triangle) for triangle in local)


def _triangulate_positions(
    vertices: tuple[Point, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, int, int], ...]:
    key = None
    for index in range(1, len(vertices) - 1):
        key = plane_key((vertices[0], vertices[index], vertices[index + 1]), budget=budget)
        if key is not None:
            break
    budget.reserve(3 * len(vertices))
    if key is None or len(set(vertices)) != len(vertices):
        raise ValueError(
            "polyhedral.facet.vertices must be distinct with plane rank two; got repeated vertices or deficient rank"
        )
    plane, axes, _, _ = key
    budget.reserve(8 * len(vertices))
    if any(
        sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) != 0
        for point in vertices
    ):
        raise ValueError(
            "polyhedral.facet.vertices must lie in one plane; got nonzero plane residual"
        )
    projected = project(vertices, axes, budget=budget)
    count = len(vertices)
    convex = _strictly_convex_fan(projected, budget=budget)
    if convex is not None:
        return convex
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
            triangles.append((previous, middle, following))
            remaining.pop(position)
            break
        else:
            raise ValueError(
                "polyhedral.facet.loop must admit oriented ears; got an untriangulable remainder"
            )
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return tuple(triangles)


def _strictly_convex_fan(
    projected: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, int, int], ...] | None:
    """Return ear-clipping triangles of a proven strictly convex simple loop.

    Every cyclic turn has one strict sign, so edge directions rotate
    monotonically by less than a half turn per vertex. Counting the edges whose
    direction crosses the positive first axis in that rotation sense gives the
    exact turning number; a value of one proves a simple convex loop. For such
    a loop each clipping pass accepts the first remaining corner, so the ears
    fan from the last vertex.
    """
    count = len(projected)
    turns = tuple(
        turn(
            projected[index - 1],
            projected[index],
            projected[(index + 1) % count],
            budget=budget,
        )
        for index in range(count)
    )
    if all(value > 0 for value in turns):
        sign = 1
    elif all(value < 0 for value in turns):
        sign = -1
    else:
        return None
    budget.reserve(2 * count)
    rises = tuple(
        projected[(index + 1) % count][1] - projected[index][1] for index in range(count)
    )
    crossings = sum(
        1
        for index in range(count)
        if sign * rises[index] < 0 <= sign * rises[(index + 1) % count]
    )
    if crossings != 1:
        return None
    last = count - 1
    return (
        *((last, index, index + 1) for index in range(count - 3)),
        (count - 3, count - 2, last),
    )


def star_determinant(tet: Tetrahedron, /, *, budget: CoordinateEnclosureBudget) -> Fraction:
    """Return the signed edge cofactor determinant of one star tetrahedron.

    The value depends only on the four exact corners in order, so a completed
    determinant is reused within the request under that full key; each lookup
    charges hashing its twelve coordinates.
    """
    budget.reserve(12)
    cache = budget.star_determinant_cache
    determinant = cache.get(tet)
    if determinant is None:
        budget.reserve(9)
        determinant = determinant3(
            *(
                tuple(value - base for value, base in zip(point, tet[0], strict=True))
                for point in tet[1:]
            ),
            budget=budget,
        )
        budget.retain_basis((tet, determinant))
        cache[tet] = determinant
    return determinant


def star_tetrahedra(
    mesh: CellMesh,
    points: np.ndarray,
    /,
    *,
    require_positive: bool = True,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Tetrahedron, ...], ...]:
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
    # Both cells of an interior face use one triangulation of its stored loop;
    # the reversed occurrence reverses each triangle, a triangulation of the
    # reversed loop with the same orientation.
    face_triangles: dict[int, tuple[tuple[int, int, int], ...]] = {}
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
        budget.reserve(3 * len(indices) + 3)
        center = tuple(
            sum((points[index, axis] for index in indices), Fraction(0)) / len(indices)
            for axis in range(3)
        )
        pieces = []
        for occurrence in occurrences:
            face = int(cell_faces[occurrence])
            triangles = face_triangles.get(face)
            if triangles is None:
                row = tuple(faces[face_offsets[face] : face_offsets[face + 1]].tolist())
                triangles = triangulate_loop(points, row, budget=budget)
                face_triangles[face] = triangles
            if signs[occurrence] < 0:
                triangles = tuple((a, c, b) for a, b, c in triangles)
            for triangle in triangles:
                corners = tuple(tuple(points[index]) for index in triangle)
                tetrahedron = (center, corners[0], corners[1], corners[2])
                if require_positive and star_determinant(tetrahedron, budget=budget) <= 0:
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
    """Stream existing scalar prices through the canonical request counter.

    Source-owned plane/projection/hull primitives and direct overlap primitives
    now share one measured work delta. Root ticket consumption is unchanged;
    evidence no longer omits the former direct-ticket scalar charges. Geometry,
    arithmetic order, and source/target mesh identities are unaffected.
    """

    budget: CoordinateEnclosureBudget
    maximum_work: int | None
    starting_work: int

    def actual_work(self) -> int:
        return self.budget.work_units - self.starting_work

    def admit(self, count: int) -> None:
        completed = self.actual_work()
        if self.maximum_work is not None and completed + count > self.maximum_work:
            # Every scalar and source-owned primitive uses the same request
            # counter; this dry admission does not count an unexecuted step.
            raise coordinate_budget_owner.CoordinateEnclosureResourceError(
                "coefficient_work",
                self.maximum_work,
                completed + count,
                completed,
                admission=completed > self.maximum_work or count == 0,
            )

        self.budget.admit_work_bound(count)

    def visit(self, count: int) -> None:
        self.admit(count)
        self.budget.reserve(count)


class _OverlapPolytope(StrictModule):
    planes: tuple[Point, ...]
    vertices: tuple[Point, ...]
    center: Point


@dataclass(frozen=True, slots=True)
class _OverlapStorageBounds:
    """Operand-derived scalar widths, not a whole-intersection reservation."""

    coordinate_bits: int
    plane_bits: int

    @property
    def side_bits(self) -> int:
        return 4 * self.plane_bits + 3 * self.coordinate_bits + 4

    @property
    def crossing_bits(self) -> int:
        return 3 * self.side_bits + 3 * self.coordinate_bits + 4


def _overlap_rational_storage(count: int, bits: int, /) -> int:
    return count * (320 + 2 * declared_integer_storage_bytes(bits))


@dataclass(slots=True)
class _OverlapStorage:
    """Disjoint live banks for sides, direct arithmetic, and returned data.

    The result owner is also the caller's pair owner. Inserting a crossing
    therefore never transfers or duplicates its Fraction credit. Its prefix
    owns earlier pair results and two scalar banks cover the returned integral
    and replacement pair accumulators while their old values still coexist.
    """

    bounds: _OverlapStorageBounds
    output: CoordinateLiveStorage
    scratch: CoordinateLiveStorage
    sides: CoordinateLiveStorage
    prefix: int
    vertices: int = 0

    def side_rows(self, count: int, rows: int, /) -> None:
        self.sides.set_bound(
            4096 + 256 * rows
            + _overlap_rational_storage(count, self.bounds.side_bits)
        )
        self.scratch.set_bound(
            4096 + _overlap_rational_storage(16, self.bounds.side_bits)
        )

    def crossings(self) -> None:
        # Parameter, three coordinate expressions, side checks, and both
        # generations of their replacement temporaries coexist here.
        self.scratch.set_bound(
            4096 + _overlap_rational_storage(64, 8 * self.bounds.crossing_bits + 8)
        )

    def add_vertex(self, vertices: set[Point], point: Point, /) -> None:
        if point not in vertices:
            count = len(vertices) + 1
            self.output.set_bound(
                self.prefix + 4096 + 256 * count
                + _overlap_rational_storage(3 * count, self.bounds.crossing_bits)
            )
            vertices.add(point)
            self.vertices = count

    def integrals(self, planes: int, second_moments: bool, retain_simplices: bool, /) -> None:
        count = self.vertices
        # Crossing helper frames and side rows have ended before this stage.
        self.scratch.set_bound(0)
        crossing_bits = self.bounds.crossing_bits
        center_bits = count * (crossing_bits + 1) + count.bit_length() + 2
        determinant_bits = 18 * (2 * center_bits + 1) + 18
        moment_bits = (
            determinant_bits + (16 if second_moments else 4) * center_bits + 64
        )
        triangles = planes * max(count - 2, 0)
        # All fan coordinates share Q, the product of the discovered vertex
        # denominators times V. Integral denominators divide 120*Q**5.
        common_bits = 3 * count * crossing_bits + count.bit_length() + 1
        sum_bits = 5 * (common_bits + crossing_bits + 8) + triangles.bit_length() + 8
        scalars = 13 if second_moments else 4
        self.output.set_bound(
            self.prefix + 4096 + 256 * count
            + (256 * triangles if retain_simplices else 0)
            + _overlap_rational_storage(3 * count, crossing_bits)
            + _overlap_rational_storage(3, center_bits)
            + _overlap_rational_storage(2 * scalars, sum_bits)
        )
        # Face containers reference the result bank; only the current face
        # and current simplex's direct rational circuit are live scratch.
        self.scratch.set_bound(
            4096 + 1024 * (count + planes)
            + _overlap_rational_storage(64, max(moment_bits, 8 * crossing_bits + 8))
        )


@contextmanager
def _overlap_storage(
    first: Tetrahedron | ExactConvexCell,
    second: Tetrahedron | ExactConvexCell,
    budget: CoordinateEnclosureBudget,
    bounds: _OverlapStorageBounds | None,
    output: CoordinateLiveStorage | None,
    scratch: CoordinateLiveStorage | None,
    /,
) -> Iterator[_OverlapStorage]:
    with (
        budget.live_storage() as local_output,
        budget.live_storage() as local_scratch,
        budget.live_storage() as sides,
    ):
        output = local_output if output is None else output
        scratch = local_scratch if scratch is None else scratch
        prefix = output.bound
        output.set_bound(prefix + 4096)
        if bounds is None:
            # Direct overlap callers need the same preadmitted operand scan.
            with budget.temporary_scope():
                budget.reserve(0, 4096 + 128 * (
                    len(first.vertices) + len(second.vertices)
                    + len(first.planes) + len(second.planes)
                    if isinstance(first, ExactConvexCell)
                    and isinstance(second, ExactConvexCell) else 16
                ))
                if isinstance(first, ExactConvexCell) and isinstance(second, ExactConvexCell):
                    bits = exact_bits(
                        np.asarray((*first.vertices, *second.vertices), dtype=object),
                        budget=budget,
                    )
                    plane_bits = exact_bits(
                        np.asarray((*first.planes, *second.planes), dtype=object),
                        budget=budget,
                    )
                else:
                    assert not isinstance(first, ExactConvexCell)
                    assert not isinstance(second, ExactConvexCell)
                    bits = exact_bits(np.asarray((*first, *second), dtype=object), budget=budget)
                    plane_bits = 51 * (bits + 1)
                bounds = _OverlapStorageBounds(bits, plane_bits)
        try:
            yield _OverlapStorage(bounds, output, scratch, sides, prefix)
        finally:
            scratch.set_bound(0)



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


def _plane_side(plane: Point, point: Point, work: _OverlapWork, /) -> Fraction:
    work.visit(7)
    return sum((plane[axis] * point[axis] for axis in range(3)), plane[3])


def _tetrahedron_crossings(
    own: Tetrahedron,
    other_planes: tuple[Point, ...],
    sides: tuple[tuple[Fraction, ...], ...],
    vertices: set[Point],
    work: _OverlapWork,
    allocation: _OverlapStorage,
    /,
) -> None:
    for corner, point in enumerate(own):
        if all(row[corner] <= 0 for row in sides):
            allocation.add_vertex(vertices, point)
    for start, end in combinations(range(4), 2):
        for crossed, row in enumerate(sides):
            low, high = row[start], row[end]
            if not (low < 0 < high or high < 0 < low):
                continue
            work.visit(11)
            parameter = low / (low - high)
            point = tuple(
                own[start][axis] + parameter * (own[end][axis] - own[start][axis])
                for axis in range(3)
            )
            if all(
                index == crossed or _plane_side(plane, point, work) <= 0
                for index, plane in enumerate(other_planes)
            ):
                allocation.add_vertex(vertices, point)


def _overlap_polytope(
    first: Tetrahedron,
    second: Tetrahedron,
    first_planes: tuple[Point, ...],
    second_planes: tuple[Point, ...],
    work: _OverlapWork,
    allocation: _OverlapStorage,
    second_moments: bool,
    retain_simplices: bool,
    /,
) -> _OverlapPolytope | None:
    """Enumerate contained corners and exact edge/plane crossings in order."""
    planes = tuple(sorted(set((*first_planes, *second_planes))))
    vertices: set[Point] = set()
    work.admit(1)
    for own, other_planes in ((first, second_planes), (second, first_planes)):
        allocation.side_rows(len(own) * len(other_planes), len(other_planes))
        sides = tuple(
            tuple(_plane_side(plane, point, work) for point in own)
            for plane in other_planes
        )
        allocation.crossings()
        _tetrahedron_crossings(own, other_planes, sides, vertices, work, allocation)
        del sides
        allocation.sides.set_bound(0)
    if len(vertices) < 4:
        return None
    allocation.integrals(len(planes), second_moments, retain_simplices)
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
    _storage_bounds: _OverlapStorageBounds | None = None,
    _output_storage: CoordinateLiveStorage | None = None,
    _workspace: CoordinateLiveStorage | None = None,
) -> ExactTetrahedronOverlap | ExactOverlapRefusal:
    """Return intersection integrals or a typed coefficient-resource refusal."""
    if preparation is not None and preparation.budget is not budget:
        raise ValueError(
            "tetrahedron_overlap.preparation must belong to the current request budget; got another budget"
        )
    work = _OverlapWork(budget, maximum_work, budget.work_units)
    try:
        with (
            budget.measurement(),
            budget.bound_stage(
                budget.maximum_work_units if maximum_work is None else maximum_work,
                budget.maximum_memory_bytes,
            ),
            budget.temporary_scope(),
        ):
            work.admit(1)
            with _overlap_storage(
                first, second, budget, _storage_bounds, _output_storage, _workspace
            ) as allocation:
                # Plane primitives retain their existing preadmissions until
                # this helper ends; cached planes also keep their canonical
                # request-owned basis credits.
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
                allocation.side_rows(4, 1)
                if _overlap_planes_separate(first, second, first_planes, second_planes, work):
                    polytope = None
                else:
                    polytope = _overlap_polytope(
                        first, second, first_planes, second_planes, work, allocation,
                        second_moments, retain_simplices,
                    )
                if polytope is None:
                    return _empty_overlap(
                        second_moments, retain_simplices, work.actual_work()
                    )
                return _overlap_integrals(polytope, work, second_moments, retain_simplices)
    except coordinate_budget_owner.CoordinateEnclosureResourceError as error:
        return ExactOverlapRefusal(error.resource, work.actual_work())


def tetrahedron_contact_vertices(
    first: Tetrahedron, second: Tetrahedron, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Point, ...]:
    """Retain the exact closed intersection, including faces, edges and points.

    Unlike volume integration this deliberately does not discard rank-deficient
    intersections. Affine trace equality is decided on this vertex bank.
    """
    work = _OverlapWork(budget, None, budget.work_units)
    with budget.temporary_scope(), _overlap_storage(
        first, second, budget, None, None, None
    ) as allocation:
        first_planes = tetrahedron_planes(first, work.visit, budget=budget)
        second_planes = tetrahedron_planes(second, work.visit, budget=budget)
        vertices: set[Point] = set()
        for own, planes in ((first, second_planes), (second, first_planes)):
            allocation.side_rows(4 * len(planes), len(planes))
            sides = tuple(
                tuple(_plane_side(plane, point, work) for point in own)
                for plane in planes
            )
            allocation.crossings()
            _tetrahedron_crossings(own, planes, sides, vertices, work, allocation)
        result = tuple(sorted(vertices))
        budget.retain_basis(result)
        return result


def _halfspace_piece(
    vertices: tuple[Point, ...], planes: tuple[Point, ...], plane: Point,
    work: _OverlapWork, /, *, retain_simplices: bool = True
) -> tuple[tuple[Point, ...], tuple[Tetrahedron, ...]]:
    """Split one convex bank using the overlap owner's arithmetic and storage."""
    budget = work.budget
    budget.reserve(6 * len(vertices), 512 + 256 * len(vertices))
    lower = tuple(min(point[axis] for point in vertices) for axis in range(3))
    upper = tuple(max(point[axis] for point in vertices) for axis in range(3))
    cell = ExactConvexCell(vertices, (), (*planes, plane), lower, upper)
    with _overlap_storage(cell, cell, budget, None, None, None) as allocation:
        allocation.side_rows(len(vertices), 1)
        sides = tuple(_plane_side(plane, point, work) for point in vertices)
        if all(value <= 0 for value in sides):
            selected = vertices
        elif all(value >= 0 for value in sides):
            return (), ()
        else:
            allocation.crossings()
            hits: set[Point] = set()
            for point, side in zip(vertices, sides, strict=True):
                if side <= 0:
                    allocation.add_vertex(hits, point)
            # Chords are a harmless edge superset of a convex hull. Extra
            # coplanar/interior hits are removed by the canonical face hull.
            for start, end in combinations(range(len(vertices)), 2):
                low, high = sides[start], sides[end]
                if not (low < 0 < high or high < 0 < low):
                    continue
                work.visit(11)
                parameter = low / (low - high)
                point = tuple(
                    vertices[start][axis]
                    + parameter * (vertices[end][axis] - vertices[start][axis])
                    for axis in range(3)
                )
                allocation.add_vertex(hits, point)
            selected = tuple(sorted(hits))
        if len(selected) < 4:
            return (), ()
        if not retain_simplices:
            budget.retain_basis(selected)
            return selected, ()
        allocation.vertices = len(selected)
        all_planes = tuple(sorted(set((*planes, plane))))
        allocation.integrals(len(all_planes), False, True)
        work.visit(3 * (len(selected) + 1))
        center = tuple(
            sum((point[axis] for point in selected), Fraction(0)) / len(selected)
            for axis in range(3)
        )
        result = _overlap_integrals(
            _OverlapPolytope(all_planes, selected, center), work, False, True
        )
        if result.volume == 0:
            return (), ()
        assert result.simplices is not None
        budget.retain_basis((selected, result.simplices))
        return selected, result.simplices


class ExactPolyhedralRefusal(ValueError):
    """Preserve the canonical typed overlap refusal without invented counters."""

    def __init__(self, refusal: ExactOverlapRefusal, /) -> None:
        self.overlap_refusal = refusal
        super().__init__(f"exact_polyhedral clipping refused: {refusal.resource}")


def tetrahedron_difference(
    first: Tetrahedron, second: Tetrahedron, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Tetrahedron, ...]:
    """Partition ``first \\ second`` into exact positive tetrahedra.

    Successive outward planes emit mutually disjoint convex pieces; the
    remainder satisfies all preceding inward halfspaces. Boundaries have zero
    content. All clipping, hulls, moments and storage use this exact owner.
    """
    with budget.live_storage() as output:
        overlap = tetrahedron_overlap(
            first, second, budget=budget, _output_storage=output
        )
        if isinstance(overlap, ExactOverlapRefusal):
            raise ExactPolyhedralRefusal(overlap)
        if overlap.volume == 0:
            return (first,)
        if overlap.volume == abs(star_determinant(first, budget=budget)) / 6:
            return ()
    work = _OverlapWork(budget, None, budget.work_units)
    with budget.temporary_scope():
        planes = tetrahedron_planes(first, work.visit, budget=budget)
        cuts = tetrahedron_planes(second, work.visit, budget=budget)
        vertices = tuple(first)
        result: list[Tetrahedron] = []
        for plane in cuts:
            if not vertices:
                break
            work.visit(4)
            outside = tuple(-value for value in plane)
            _, emitted = _halfspace_piece(vertices, planes, outside, work)
            budget.reserve(len(emitted), 64 + 8 * (len(result) + len(emitted)))
            result.extend(emitted)
            vertices, _ = _halfspace_piece(
                vertices, planes, plane, work, retain_simplices=False
            )
            planes = (*planes, plane)
        retained = tuple(result)
        budget.retain_basis(retained)
        return retained


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


def _integrate_overlap_face(
    plane: Point,
    vertices: tuple[Point, ...],
    center: Point,
    volume: Fraction,
    moment: list[Fraction],
    second_moment: list[list[Fraction]] | None,
    simplices: list[Tetrahedron] | None,
    work: _OverlapWork,
    /,
) -> Fraction:
    """Keep projection/hull temporaries within one completed face frame."""
    budget = work.budget
    work.visit(7 * len(vertices))
    face = tuple(
        point
        for point in vertices
        if sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) == 0
    )
    if len(face) < 3:
        return volume
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
        work.visit(12 + 14)
        signed = exact_cofactor_determinant_3(*columns)
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
    return volume


def _overlap_integrals(
    polytope: _OverlapPolytope,
    work: _OverlapWork,
    second_moments: bool,
    retain_simplices: bool,
    /,
) -> ExactTetrahedronOverlap:
    """Integrate the oriented fan with unchanged face/corner/axis charge order."""
    planes, vertices, center = polytope.planes, polytope.vertices, polytope.center
    volume, moment = Fraction(0), [Fraction(0)] * 3
    second_moment = [[Fraction(0)] * 3 for _ in range(3)] if second_moments else None
    simplices = [] if retain_simplices else None
    for plane in planes:
        with work.budget.temporary_scope():
            volume = _integrate_overlap_face(
                plane, vertices, center, volume, moment, second_moment, simplices, work
            )
    return ExactTetrahedronOverlap(
        volume,
        tuple(moment),
        None if second_moment is None else tuple(tuple(row) for row in second_moment),
        None if simplices is None else tuple(simplices),
        work.actual_work(),
    )


@dataclass(frozen=True, slots=True)
class ExactConvexCell:
    """Exact halfspace, vertex and edge description of one verified convex cell.

    ``planes`` are outward rational face planes whose closed nonpositive sides
    contain every cell vertex. Edges are the cell's packed face-loop edges, a
    superset of its geometric edges.
    """

    vertices: tuple[Point, ...]
    edges: tuple[tuple[int, int], ...]
    planes: tuple[Point, ...]
    lower: Point
    upper: Point


def _cell_face_rows(
    connectivity: PolyhedralConnectivity, cell: int, /
) -> tuple[tuple[int, ...], ...]:
    offsets = np.asarray(connectivity.cell_face_offsets)
    face_offsets = np.asarray(connectivity.face_vertex_offsets)
    faces = np.asarray(connectivity.face_vertex_values)
    cell_faces = np.asarray(connectivity.cell_face_values)
    return tuple(
        tuple(
            faces[
                face_offsets[cell_faces[occurrence]] : face_offsets[
                    cell_faces[occurrence] + 1
                ]
            ].tolist()
        )
        for occurrence in range(offsets[cell], offsets[cell + 1])
    )


def _supporting_plane(
    points: np.ndarray,
    row: tuple[int, ...],
    cell_vertices: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Point | None:
    """Return the outward face plane if it supports every cell vertex."""
    loop = tuple(tuple(points[index]) for index in row)
    key = None
    for index in range(1, len(loop) - 1):
        key = plane_key((loop[0], loop[index], loop[index + 1]), budget=budget)
        if key is not None:
            break
    if key is None:
        return None
    plane = key[0]
    on_face = set(row)
    budget.reserve(7 * len(cell_vertices))
    sides = tuple(
        (
            vertex,
            sum(
                (plane[axis] * points[vertex][axis] for axis in range(3)),
                plane[3],
            ),
        )
        for vertex in cell_vertices
    )
    if any(side != 0 for vertex, side in sides if vertex in on_face):
        return None
    off_face = tuple(side for vertex, side in sides if vertex not in on_face)
    if off_face and all(side <= 0 for side in off_face) and any(side < 0 for side in off_face):
        return plane
    if off_face and all(side >= 0 for side in off_face) and any(side > 0 for side in off_face):
        return tuple(-value for value in plane)
    return None


def exact_convex_cells(
    mesh: CellMesh,
    points: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    selected: frozenset[int] | None = None,
) -> tuple[ExactConvexCell | None, ...]:
    """Return exact convex descriptions of cells, or ``None`` where unproven.

    A planar face whose plane keeps every other cell vertex in one closed
    halfspace, with at least one strictly inside, is a supporting face of the
    vertex hull. When every face supports the hull and the caller has certified
    a closed cell boundary with positive star tetrahedra about the vertex mean,
    that boundary equals the hull boundary, so the cell is the intersection of
    the outward halfspaces. Coplanar sibling faces share one deduplicated plane. Cells
    outside ``selected`` are not examined and report ``None``.
    """
    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            pass
        case _:
            raise TypeError(
                "polyhedral.mesh.connectivity must contain packed faces; got another connectivity family"
            )
    vertex_offsets = np.asarray(connectivity.cell_vertex_offsets)
    vertex_values = np.asarray(connectivity.cell_vertex_values)
    result: list[ExactConvexCell | None] = []
    for cell in range(connectivity.cell_count):
        if selected is not None and cell not in selected:
            result.append(None)
            continue
        cell_vertices = tuple(
            vertex_values[vertex_offsets[cell] : vertex_offsets[cell + 1]].tolist()
        )
        rows = _cell_face_rows(connectivity, cell)
        planes: set[Point] = set()
        for row in rows:
            plane = _supporting_plane(points, row, cell_vertices, budget=budget)
            if plane is None:
                break
            planes.add(plane)
        else:
            local = {vertex: index for index, vertex in enumerate(cell_vertices)}
            edges = {
                tuple(sorted((local[row[index]], local[row[index - 1]])))
                for row in rows
                for index in range(len(row))
            }
            vertices = tuple(tuple(points[vertex]) for vertex in cell_vertices)
            budget.reserve(6 * len(vertices))
            result.append(
                ExactConvexCell(
                    vertices,
                    tuple(sorted(edges)),
                    tuple(sorted(planes)),
                    tuple(min(point[axis] for point in vertices) for axis in range(3)),
                    tuple(max(point[axis] for point in vertices) for axis in range(3)),
                )
            )
            continue
        result.append(None)
    return tuple(result)


def convex_cells_disjoint(first: ExactConvexCell, second: ExactConvexCell, /) -> bool:
    """Return whether closed exact boxes exclude a positive-volume intersection."""
    return any(
        max(first.lower[axis], second.lower[axis])
        >= min(first.upper[axis], second.upper[axis])
        for axis in range(3)
    )


def _convex_side_rows(
    planes: tuple[Point, ...],
    vertices: tuple[Point, ...],
    work: _OverlapWork,
    /,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Return plane-by-vertex sides, or ``None`` at the first separating plane."""
    rows = []
    for plane in planes:
        row = tuple(_plane_side(plane, point, work) for point in vertices)
        if min(row) >= 0:
            return None
        rows.append(row)
    return tuple(rows)


def _convex_crossings(
    own: ExactConvexCell,
    other: ExactConvexCell,
    sides: tuple[tuple[Fraction, ...], ...],
    vertices: set[Point],
    work: _OverlapWork,
    allocation: _OverlapStorage,
    /,
) -> None:
    for corner, point in enumerate(own.vertices):
        if all(row[corner] <= 0 for row in sides):
            allocation.add_vertex(vertices, point)
    for start, end in own.edges:
        for crossed, row in enumerate(sides):
            low, high = row[start], row[end]
            if not (low < 0 < high or high < 0 < low):
                continue
            work.visit(11)
            parameter = low / (low - high)
            first, last = own.vertices[start], own.vertices[end]
            point = tuple(
                first[axis] + parameter * (last[axis] - first[axis]) for axis in range(3)
            )
            if all(
                index == crossed or _plane_side(plane, point, work) <= 0
                for index, plane in enumerate(other.planes)
            ):
                allocation.add_vertex(vertices, point)


def _empty_overlap(
    second_moments: bool, retain_simplices: bool, operation_count: int, /
) -> ExactTetrahedronOverlap:
    return ExactTetrahedronOverlap(
        Fraction(0),
        (Fraction(0),) * 3,
        ((Fraction(0),) * 3,) * 3 if second_moments else None,
        () if retain_simplices else None,
        operation_count,
    )


@checked
def convex_cell_overlap(
    first: ExactConvexCell,
    second: ExactConvexCell,
    /,
    *,
    second_moments: bool = False,
    retain_simplices: bool = False,
    maximum_work: int | None = None,
    budget: CoordinateEnclosureBudget,
    _storage_bounds: _OverlapStorageBounds | None = None,
    _output_storage: CoordinateLiveStorage | None = None,
    _workspace: CoordinateLiveStorage | None = None,
) -> ExactTetrahedronOverlap | ExactOverlapRefusal:
    """Return exact integrals of two verified convex cells' intersection.

    A face plane holding the other cell on its closed outer side proves a
    zero-volume intersection. Otherwise every intersection vertex lies on
    three independent face planes, so it is a contained corner or a face-loop
    edge point on a face plane of the other cell; the oriented fan about the
    vertex mean then integrates the intersection exactly.
    """
    work = _OverlapWork(budget, maximum_work, budget.work_units)
    try:
        with (
            budget.measurement(),
            budget.bound_stage(
                budget.maximum_work_units if maximum_work is None else maximum_work,
                budget.maximum_memory_bytes,
            ),
            budget.temporary_scope(),
        ):
            work.admit(1)
            with _overlap_storage(
                first, second, budget, _storage_bounds, _output_storage, _workspace
            ) as allocation:
                allocation.side_rows(
                    len(second.planes) * len(first.vertices)
                    + len(first.planes) * len(second.vertices),
                    len(first.planes) + len(second.planes),
                )
                first_sides = _convex_side_rows(second.planes, first.vertices, work)
                second_sides = (
                    None
                    if first_sides is None
                    else _convex_side_rows(first.planes, second.vertices, work)
                )
                if first_sides is None or second_sides is None:
                    return _empty_overlap(
                        second_moments, retain_simplices, work.actual_work()
                    )
                vertices: set[Point] = set()
                allocation.crossings()
                _convex_crossings(first, second, first_sides, vertices, work, allocation)
                _convex_crossings(second, first, second_sides, vertices, work, allocation)
                del first_sides, second_sides
                allocation.sides.set_bound(0)
                if len(vertices) < 4:
                    return _empty_overlap(
                        second_moments, retain_simplices, work.actual_work()
                    )
                allocation.integrals(
                    len(first.planes) + len(second.planes),
                    second_moments, retain_simplices,
                )
                work.visit(3 * (len(vertices) + 1))
                center = tuple(
                    sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
                    for axis in range(3)
                )
                planes = tuple(sorted(set((*first.planes, *second.planes))))
                return _overlap_integrals(
                    _OverlapPolytope(planes, tuple(vertices), center),
                    work,
                    second_moments,
                    retain_simplices,
                )
    except coordinate_budget_owner.CoordinateEnclosureResourceError as error:
        return ExactOverlapRefusal(error.resource, work.actual_work())


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
    exceeds ``maximum_bits + 1`` bits. The first pass visits every coordinate;
    the second forms the multiple once per distinct denominator, in source order.
    """
    return _coordinate_integer_profile(points, maximum_bits, budget=budget)[0]


@dataclass(frozen=True, slots=True)
class _CoordinateIntegerScalars:
    """One paid input visit's exact scalar values and occurrence routing."""

    ratios: tuple[tuple[int, int], ...]
    routes: tuple[int, ...]


def _coordinate_integer_profile(
    points: np.ndarray,
    maximum_bits: int | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    collect_scalars: bool = False,
) -> tuple[tuple[int, int] | None, int, _CoordinateIntegerScalars | None]:
    """Profile actual values, factoring only repeated numerical operations."""
    # Dictionary keys borrow the original plain integer numerator/denominator
    # owners. This bounds their entries, ratio tuples, route IDs and tuple
    # packing before the first new metadata owner is allocated. The enclosing
    # coordinate temporary scope owns any returned scalar recipe.
    budget.reserve(0, 1024 + (256 if collect_scalars else 96) * points.size)
    denominators: dict[int, None] = {}
    scalar_indices: dict[tuple[int, int], int] = {}
    routes: list[int] = []
    with exact_charge(points.size, budget=budget):
        denominator_bits, excess, rational_bits = 0, 0, 0
        for value in points.flat:
            exact = _exact_value(value)
            numerator_length = exact.numerator.bit_length()
            denominator_length = exact.denominator.bit_length()
            if exact.denominator not in denominators:
                denominators[exact.denominator] = None
                denominator_bits += denominator_length
            if collect_scalars:
                ratio = (exact.numerator, exact.denominator)
                index = scalar_indices.get(ratio)
                if index is None:
                    index = len(scalar_indices)
                    scalar_indices[ratio] = index
                routes.append(index)
            excess = max(excess, numerator_length - denominator_length + 1)
            rational_bits = max(rational_bits, numerator_length, denominator_length)
    scalars = (
        _CoordinateIntegerScalars(tuple(scalar_indices), tuple(routes))
        if collect_scalars else None
    )
    multiple_bits = (
        denominator_bits
        if maximum_bits is None
        else min(denominator_bits, maximum_bits + 1)
    )
    with exact_charge(len(denominators), 3 * bigint_bytes(multiple_bits), budget=budget):
        denominator = 1
        for source_denominator in denominators:
            factor = source_denominator // math.gcd(denominator, source_denominator)
            if (
                maximum_bits is not None
                and denominator.bit_length() + factor.bit_length() - 1 + excess
                > maximum_bits
            ):
                return None, rational_bits, scalars
            denominator *= factor
            if (
                maximum_bits is not None
                and denominator.bit_length() + excess > maximum_bits
            ):
                return None, rational_bits, scalars
    return (denominator, denominator.bit_length() + excess), rational_bits, scalars


class CoordinateIntegerPreparation:
    """One caller-lived normalization of an immutable rational coordinate bank.

    The enclosing coordinate temporary scope owns its storage. Reuse requires
    the same ledger and exact current Fraction values, not an array identity or
    an unbound denominator override. Nothing is retained across requests.
    """

    def __init__(
        self,
        points: np.ndarray,
        maximum_bits: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        if points.ndim != 2:
            raise TypeError(
                "CoordinateIntegerPreparation.points must be a rank-two Fraction bank."
            )
        budget.reserve(0, 512 + 16 * points.size + 64 * points.shape[0])
        if any(type(value) is not Fraction for value in points.flat):
            raise TypeError(
                "CoordinateIntegerPreparation.points must be a rank-two Fraction bank."
            )
        self._owner = budget
        self._shape = points.shape
        self._source = tuple(tuple(row) for row in points)
        self._profile, self._rational_bits, self._scalars = _coordinate_integer_profile(
            points, maximum_bits, budget=budget, collect_scalars=True,
        )
        self._integers: tuple[np.ndarray, Fraction] | None = None

    @property
    def profile(self) -> tuple[int, int] | None:
        return self._profile

    @property
    def integer_bits(self) -> int:
        if self._profile is None:
            raise ValueError("CoordinateIntegerPreparation must have a completed bit profile.")
        return self._profile[1]

    def rational_bits(
        self, points: np.ndarray, /, *, budget: CoordinateEnclosureBudget
    ) -> int:
        """Largest numerator or denominator bit length of the bound source bank."""
        self._require_source(points, budget=budget)
        return self._rational_bits

    def _require_source(
        self, points: np.ndarray, /, *, budget: CoordinateEnclosureBudget
    ) -> None:
        if budget is not self._owner or points.shape != self._shape:
            raise ValueError(
                "CoordinateIntegerPreparation must bind the current source values and ledger."
            )
        with budget.temporary_scope():
            budget.reserve(0, 512 + 16 * points.size + 64 * points.shape[0])
            if (
                any(type(value) is not Fraction for value in points.flat)
                or tuple(tuple(row) for row in points) != self._source
                or self._scalars is None
                or any(
                    (value.numerator, value.denominator) != self._scalars.ratios[index]
                    for value, index in zip(points.flat, self._scalars.routes, strict=True)
                )
            ):
                raise ValueError(
                    "CoordinateIntegerPreparation must bind the current source values and ledger."
                )

    def coordinates(
        self, points: np.ndarray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[np.ndarray, Fraction]:
        self._require_source(points, budget=budget)
        if self._profile is None:
            raise ValueError("CoordinateIntegerPreparation must have a completed bit profile.")
        if self._integers is None:
            assert self._scalars is not None
            self._integers = _scale_coordinate_integers(
                points, self._profile, self._scalars, budget=budget,
            )
            self._integers[0].flags.writeable = False
        return self._integers


def coordinate_integers(
    points: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparation: CoordinateIntegerPreparation | None = None,
) -> tuple[np.ndarray, Fraction]:
    """Integers ``N`` and the scale ``1 / D`` with ``points == N / D`` exactly.

    ``D`` is the least common denominator. The scaled bank is charged to the
    active coordinate ledger before it is formed and stays charged while the
    caller's enclosing scope retains it.
    """
    if preparation is not None:
        return preparation.coordinates(points, budget=budget)
    profile, _, scalars = _coordinate_integer_profile(
        points, None, budget=budget, collect_scalars=True,
    )
    assert profile is not None and scalars is not None
    return _scale_coordinate_integers(points, profile, scalars, budget=budget)


def _scale_coordinate_integers(
    points: np.ndarray,
    profile: tuple[int, int],
    scalars: _CoordinateIntegerScalars,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, Fraction]:
    """Scale each distinct actual scalar once, then pack its occurrence routes."""
    denominator, bits = profile
    # The full output bank remains admitted: numeric equality saves actual
    # division/multiplication, not the retained per-occurrence pointer storage.
    exact_reserve(len(scalars.ratios), points.size * bigint_bytes(bits), budget=budget)
    scaled = tuple(
        numerator * (denominator // divisor)
        for numerator, divisor in scalars.ratios
    )
    return np.asarray(
        tuple(scaled[index] for index in scalars.routes), dtype=object,
    ).reshape(points.shape), Fraction(1, denominator)


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


def _triangle_bounds(
    points: tuple[Point, ...], /, *, budget: CoordinateEnclosureBudget
) -> tuple[Point, Point]:
    budget.reserve(0, 512)
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


@dataclass(slots=True)
class TriangleContactPreparation:
    """Retain exact geometric subproofs, never a vertex-incidence verdict."""

    budget: CoordinateEnclosureBudget
    planes: dict[tuple[Point, ...], tuple[Point, tuple[int, ...], int, int] | None] = field(default_factory=dict)
    projections: dict[tuple[tuple[Point, ...], tuple[int, ...]], tuple[Point, ...]] = field(default_factory=dict)
    sides: dict[tuple[Point, Point], Fraction] = field(default_factory=dict)
    bounds: dict[tuple[Point, ...], tuple[Point, Point]] = field(default_factory=dict)

    def box(self, points: tuple[Point, ...], /) -> tuple[Point, Point]:
        if points not in self.bounds:
            value = _triangle_bounds(points, budget=self.budget)
            self.budget.retain_basis((points, value))
            self.bounds[points] = value
        return self.bounds[points]

    def plane(
        self, points: tuple[Point, ...], /
    ) -> tuple[Point, tuple[int, ...], int, int] | None:
        if points not in self.planes:
            value = plane_key(points, budget=self.budget)
            self.budget.retain_basis((points, value))
            self.planes[points] = value
        return self.planes[points]

    def projected(
        self, points: tuple[Point, ...], axes: tuple[int, ...], /
    ) -> tuple[Point, ...]:
        key = points, axes
        if key not in self.projections:
            value = project(points, axes, budget=self.budget)
            self.budget.retain_basis((key, value))
            self.projections[key] = value
        return self.projections[key]

    def side(self, plane: Point, point: Point, /) -> Fraction:
        key = plane, point
        if key not in self.sides:
            planar_primitives_owner.reserve_fraction_work(
                key, 2 * len(point) + 1, 1, 8, budget=self.budget
            )
            value = sum((a * b for a, b in zip(plane[:-1], point, strict=True)), plane[-1])
            self.budget.retain_basis((key, (value,)))
            self.sides[key] = value
        return self.sides[key]

    def turn_sign(self, first: Point, second: Point, third: Point, /) -> int:
        # The canonical plane owner removes edge length and orientation.
        # Collinear subedges therefore reuse one plane-point side proof.
        key = self.plane((first, second))
        if key is None:
            return 0
        value = self.side(key[0], third)
        return -key[3] * ((value > 0) - (value < 0))


def triangle_contact(
    points: np.ndarray,
    first: tuple[int, ...],
    second: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparation: TriangleContactPreparation | None = None,
) -> bool:
    """Detect rational triangle contact outside the shared incidence simplex."""

    shared = tuple(sorted(set(first) & set(second)))
    if len(shared) == 3:
        return False
    a, b = (
        tuple(tuple(points[index]) for index in first),
        tuple(tuple(points[index]) for index in second),
    )
    if preparation is not None and preparation.budget is not budget:
        raise ValueError("triangle_contact.preparation must belong to the active request budget")
    ka, kb = (
        (plane_key(a, budget=budget), plane_key(b, budget=budget))
        if preparation is None
        else (preparation.plane(a), preparation.plane(b))
    )
    if ka is None or kb is None:
        raise ValueError(
            "triangle_contact.triangles must have rank two; got degenerate triangles"
        )
    (lower_a, upper_a), (lower_b, upper_b) = (
        (_triangle_bounds(a, budget=budget), _triangle_bounds(b, budget=budget))
        if preparation is None else (preparation.box(a), preparation.box(b))
    )
    lower = tuple(max(x, y) for x, y in zip(lower_a, lower_b, strict=True))
    upper = tuple(min(x, y) for x, y in zip(upper_a, upper_b, strict=True))
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        return False
    # A point/axis-segment bounding intersection is already the entire
    # possible contact set. Its two endpoints suffice to prove containment in
    # the current shared simplex, using entity IDs rather than coordinate welds.
    if sum(low != high for low, high in zip(lower, upper, strict=True)) <= 1:
        if not _contact_outside_shared(points, shared, {lower, upper}):
            return False
    # A triangle in one open halfspace, apart from shared vertices on the
    # other triangle's plane, can meet that plane only in the shared simplex.
    # This is an exact convexity proof, including noncoplanar shared edges;
    # it avoids constructing intersections that are already known incidence.
    for vertices, ids, key in ((b, second, ka), (a, first, kb)):
        unshared = tuple(
            point for vertex, point in zip(ids, vertices, strict=True)
            if vertex not in shared
        )
        if preparation is None:
            planar_primitives_owner.reserve_fraction_work(
                (*unshared, key[0]), 7 * len(unshared), len(unshared), 8, budget=budget
            )
            sides = tuple(
                sum((key[0][axis] * point[axis] for axis in range(3)), key[0][3])
                for point in unshared
            )
        else:
            sides = tuple(preparation.side(key[0], point) for point in unshared)
        if all(side > 0 for side in sides) or all(side < 0 for side in sides):
            return False
    coplanar = all(
        sum((ka[0][axis] * point[axis] for axis in range(3)), ka[0][3]) == 0
        for point in b
    )
    if coplanar:
        projected_a, projected_b = (
            project(vertices, ka[1], budget=budget)
            if preparation is None else preparation.projected(vertices, ka[1])
            for vertices in (a, b)
        )
        # A supporting edge reduces contact to an exact one-dimensional
        # interval. Its intersection must be empty or contained in the edge's
        # shared entity simplex; coincident coordinates alone give no authority.
        for own, other, own_ids in (
            (projected_a, projected_b, first),
            (projected_b, projected_a, second),
        ):
            orientation = (
                turn(*own, budget=budget)
                if preparation is None else preparation.turn_sign(*own)
            )
            for p, q in zip(own, (*own[1:], own[0]), strict=True):
                sides = tuple(
                    turn(p, q, point, budget=budget)
                    if preparation is None else preparation.turn_sign(p, q, point)
                    for point in other
                )
                if any(side * orientation > 0 for side in sides):
                    continue
                zero_points = tuple(
                    point for point, side in zip(other, sides, strict=True) if side == 0
                )
                if not zero_points:
                    return False
                axis = next(axis for axis in range(2) if p[axis] != q[axis])
                low = max(min(p[axis], q[axis]), min(point[axis] for point in zero_points))
                high = min(max(p[axis], q[axis]), max(point[axis] for point in zero_points))
                if low > high:
                    return False
                shared_edge = tuple(
                    point[axis] for vertex, point in zip(own_ids, own, strict=True)
                    if vertex in shared and point in (p, q)
                )
                if shared_edge and min(shared_edge) <= low <= high <= max(shared_edge):
                    return False
        if planar_primitives_owner.intersection(
            projected_a, projected_b, budget=budget
        ) > 0:
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
    "tetrahedron_contact_vertices",
    "tetrahedron_difference",
    "ExactPolyhedralRefusal",
    "determinant3",
    "coordinate_integers",
    "ExactConvexCell",
    "exact_convex_cells",
    "convex_cells_disjoint",
    "convex_cell_overlap",
]
