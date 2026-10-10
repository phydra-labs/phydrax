#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np

from .._geometry_predicates import bigint_bytes, exact_bits, exact_charge, exact_reserve
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ._planar_coverage import convex_hull, plane_key, project, signed_measure, turn


type Point = tuple[Fraction, ...]
type Tetrahedron = tuple[Point, Point, Point, Point]


def determinant3(first: Point, second: Point, third: Point, /) -> Fraction:
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(4 * 3**3)
    from .._meshcore import current_native_execution_budget

    execution = current_native_execution_budget()
    if execution is not None:
        execution.admit_work_bound(4 * 3**3)
    result = prepare_exact_small_linear_actions((first, second, third), ((), (), ()))
    if execution is not None:
        execution.charge(work=result.operation_count)
    return result.determinant


def exact_vertices(mesh: CellMesh, geometry: CellGeometrySpec, /) -> np.ndarray:
    geometry._resolve(mesh, exact_source_prepared=True)
    return np.asarray(geometry.source_coordinates(), dtype=object)


def _between(point: Point, a: Point, b: Point, /) -> bool:
    return all(min(x, y) <= p <= max(x, y) for p, x, y in zip(point, a, b, strict=True))


def _segment_hit(a: Point, b: Point, c: Point, d: Point, /) -> bool:
    values = (turn(a, b, c), turn(a, b, d), turn(c, d, a), turn(c, d, b))
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


def triangulation_charge(points: np.ndarray, row: tuple[int, ...], /) -> tuple[int, int]:
    """Sound work and storage bound of ``triangulate_loop`` for one exact loop.

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
    bits = exact_bits(points[list(row)])
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
    points: np.ndarray, row: tuple[int, ...], /
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate only after exact planarity and simple-loop proofs."""
    vertices = tuple(tuple(value for value in points[index]) for index in row)
    key = next(
        (
            plane_key((vertices[0], vertices[index], vertices[index + 1]))
            for index in range(1, len(vertices) - 1)
            if plane_key((vertices[0], vertices[index], vertices[index + 1])) is not None
        ),
        None,
    )
    if key is None or len(set(vertices)) != len(vertices):
        raise ValueError(
            "Exact polyhedral facet has repeated vertices or deficient plane rank."
        )
    plane, axes, _, _ = key
    if any(
        sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) != 0
        for point in vertices
    ):
        raise ValueError("Exact polyhedral facet is not planar in its source geometry.")
    projected = project(vertices, axes)
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
            ):
                raise ValueError("Exact polyhedral facet loop self-intersects.")
    measure = signed_measure(projected)
    if measure == 0:
        raise ValueError("Exact polyhedral facet has zero oriented area.")
    sign = 1 if measure > 0 else -1
    remaining, triangles = list(range(count)), []
    while len(remaining) > 3:
        for position, middle in enumerate(remaining):
            previous, following = (
                remaining[position - 1],
                remaining[(position + 1) % len(remaining)],
            )
            a, b, c = projected[previous], projected[middle], projected[following]
            if sign * turn(a, b, c) <= 0:
                continue
            if any(
                all(
                    sign * turn(x, y, projected[index]) >= 0
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
            raise ValueError("Exact polyhedral facet has no lawful ear decomposition.")
    triangles.append(tuple(row[index] for index in remaining))
    return tuple(triangles)


def star_tetrahedra(
    mesh: CellMesh, points: np.ndarray, /, *, require_positive: bool = True
) -> tuple[tuple[Tetrahedron, ...], ...]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Exact polyhedral geometry requires packed face connectivity.")
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
            for triangle in triangulate_loop(points, row):
                corners = tuple(tuple(points[index]) for index in triangle)
                tetrahedron = (center, corners[0], corners[1], corners[2])
                columns = tuple(
                    tuple(value - base for value, base in zip(point, center, strict=True))
                    for point in corners
                )
                if require_positive and determinant3(*columns) <= 0:
                    raise ValueError(
                        "Exact polyhedral source lacks a positive star decomposition."
                    )
                pieces.append(tetrahedron)
        result.append(tuple(pieces))
    return tuple(result)


def tetrahedron_planes(tet: Tetrahedron, /) -> tuple[Point, ...]:
    planes = []
    for opposite in range(4):
        points = tuple(tet[index] for index in range(4) if index != opposite)
        key = plane_key(points)
        if key is None:
            raise ValueError("Exact tetrahedron has deficient face rank.")
        plane = key[0]
        side = sum((plane[axis] * tet[opposite][axis] for axis in range(3)), plane[3])
        if side == 0:
            raise ValueError("Exact tetrahedron has deficient volume rank.")
        planes.append(tuple(-value for value in plane) if side > 0 else plane)
    return tuple(planes)


@dataclass(frozen=True, slots=True)
class ExactTetrahedronOverlap:
    volume: Fraction
    first_moment: Point
    second_moment: tuple[Point, ...] | None
    simplices: tuple[Tetrahedron, ...] | None
    operation_count: int


def tetrahedron_overlap(
    first: Tetrahedron,
    second: Tetrahedron,
    /,
    *,
    second_moments: bool = False,
    retain_simplices: bool = False,
    maximum_work: int | None = None,
) -> ExactTetrahedronOverlap:
    """Exact convex intersection volume and first moment, using canonical solves."""
    from .._meshcore import current_native_execution_budget
    from ..discretization._coordinate_enclosure import CoordinateEnclosureResourceError

    execution = current_native_execution_budget()
    work = 0

    def admit(count: int) -> None:
        if maximum_work is not None and work + count > maximum_work:
            raise CoordinateEnclosureResourceError(
                "coefficient_work", maximum_work, work + count, work
            )
        if execution is not None:
            execution.admit_work_bound(count)

    def visit(count: int) -> None:
        nonlocal work
        admit(count)
        if execution is not None:
            execution.charge(work=count)
        work += count

    visit(8 * 40)
    first_planes, second_planes = tetrahedron_planes(first), tetrahedron_planes(second)
    for own_planes, other in ((first_planes, second), (second_planes, first)):
        for plane in own_planes:
            sides = []
            for point in other:
                visit(7)
                sides.append(
                    sum((plane[axis] * point[axis] for axis in range(3)), plane[3])
                )
            if min(sides) >= 0:
                return ExactTetrahedronOverlap(
                    Fraction(0),
                    (Fraction(0),) * 3,
                    ((Fraction(0),) * 3,) * 3 if second_moments else None,
                    () if retain_simplices else None,
                    work,
                )
    planes = tuple(sorted(set((*first_planes, *second_planes))))
    vertices: set[Point] = set()
    admit(1)
    for selected in combinations(planes, 3):
        matrix = tuple(tuple(plane[:3]) for plane in selected)
        right = tuple((-plane[3],) for plane in selected)
        admit(4 * 3**3)
        result = prepare_exact_small_linear_actions(matrix, right)
        if execution is not None:
            execution.charge(work=result.operation_count)
        work += result.operation_count
        if result.actions is None:
            continue
        point = tuple(row[0] for row in result.actions)
        inside = True
        for plane in planes:
            visit(7)
            if sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) > 0:
                inside = False
                break
        if inside:
            vertices.add(point)
    if len(vertices) < 4:
        return ExactTetrahedronOverlap(
            Fraction(0),
            (Fraction(0),) * 3,
            ((Fraction(0),) * 3,) * 3 if second_moments else None,
            () if retain_simplices else None,
            work,
        )
    visit(3 * (len(vertices) + 1))
    center = tuple(
        sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
        for axis in range(3)
    )
    volume, moment = Fraction(0), [Fraction(0)] * 3
    second_moment = [[Fraction(0)] * 3 for _ in range(3)] if second_moments else None
    simplices = [] if retain_simplices else None
    for plane in planes:
        visit(7 * len(vertices))
        face = tuple(
            point
            for point in vertices
            if sum((plane[axis] * point[axis] for axis in range(3)), plane[3]) == 0
        )
        if len(face) < 3:
            continue
        axes = ((1, 2), (2, 0), (0, 1))[next(axis for axis in range(3) if plane[axis])]
        projected = project(face, axes)
        hull = convex_hull(projected)
        lookup = dict(zip(projected, face, strict=True))
        ordered = tuple(lookup[point] for point in hull)
        for index in range(1, len(ordered) - 1):
            corners = (center, ordered[0], ordered[index], ordered[index + 1])
            columns = tuple(
                tuple(value - base for value, base in zip(point, center, strict=True))
                for point in corners[1:]
            )
            admit(4 * 3**3)
            determinant = prepare_exact_small_linear_actions(
                columns, ((Fraction(0),),) * 3
            )
            signed = determinant.determinant
            if execution is not None:
                execution.charge(work=determinant.operation_count)
            work += determinant.operation_count
            visit(12)
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
            for axis in range(3):
                visit(6)
                moment[axis] += (
                    measure * sum((point[axis] for point in corners), Fraction(0)) / 4
                )
            if second_moment is not None:
                visit(3 * 4)
                sums = tuple(
                    sum((point[axis] for point in corners), Fraction(0))
                    for axis in range(3)
                )
                for first_axis in range(3):
                    for second_axis in range(3):
                        visit(11)
                        second_moment[first_axis][second_axis] += (
                            measure
                            * (
                                sums[first_axis] * sums[second_axis]
                                + sum(
                                    (
                                        point[first_axis] * point[second_axis]
                                        for point in corners
                                    ),
                                    Fraction(0),
                                )
                            )
                            / 20
                        )
    return ExactTetrahedronOverlap(
        volume,
        tuple(moment),
        None if second_moment is None else tuple(tuple(row) for row in second_moment),
        None if simplices is None else tuple(simplices),
        work,
    )


def _exact_value(value: Fraction | float, /) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(float(value))


def coordinate_integer_profile(
    points: np.ndarray, /, maximum_bits: int | None = None
) -> tuple[int, int] | None:
    """Least common denominator ``D`` of exact coordinates and a bound on the bits of ``points * D``.

    ``bits(n * (D // d)) <= bits(D) + bits(n) - bits(d) + 1``. With
    ``maximum_bits`` the profile is ``None`` as soon as that bound must exceed
    it: each step ``D * (d // gcd(D, d))`` is refused before it is formed when
    even its smallest possible size is too large, so the running multiple never
    exceeds ``maximum_bits + 1`` bits. Both passes are charged before they run
    (one visit per coordinate each) with the multiple's transient bound.
    """
    with exact_charge(points.size):
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
    with exact_charge(points.size, 3 * bigint_bytes(multiple_bits)):
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


def coordinate_integers(points: np.ndarray, /) -> tuple[np.ndarray, Fraction]:
    """Integers ``N`` and the scale ``1 / D`` with ``points == N / D`` exactly.

    ``D`` is the least common denominator. The scaled bank is charged to the
    active coordinate ledger before it is formed and stays charged while the
    caller's enclosing scope retains it.
    """
    profile = coordinate_integer_profile(points)
    if profile is None:
        raise RuntimeError("An unbounded coordinate profile is never refused.")
    denominator, bits = profile
    exact_reserve(points.size, points.size * bigint_bytes(bits))
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
    point: Point, triangle: tuple[Point, ...], axes: tuple[int, ...], /
) -> bool:
    polygon = project(triangle, axes)
    projected = project((point,), axes)[0]
    sign = 1 if signed_measure(polygon) > 0 else -1
    return all(
        sign * turn(a, b, projected) >= 0
        for a, b in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
    )


def triangle_contact(
    points: np.ndarray, first: tuple[int, ...], second: tuple[int, ...], /
) -> bool:
    """Exact triangle contact outside the combinatorially shared simplex."""
    from ._planar_coverage import intersection

    shared = tuple(sorted(set(first) & set(second)))
    if len(shared) == 3:
        return False
    a, b = (
        tuple(tuple(points[index]) for index in first),
        tuple(tuple(points[index]) for index in second),
    )
    ka, kb = plane_key(a), plane_key(b)
    if ka is None or kb is None:
        raise ValueError(
            "Exact triangle contact requires nondegenerate source triangles."
        )
    hits: set[Point] = set()
    coplanar = all(
        sum((ka[0][axis] * point[axis] for axis in range(3)), ka[0][3]) == 0
        for point in b
    )
    if coplanar and intersection(project(a, ka[1]), project(b, ka[1])) > 0:
        return True
    for own, other, key in ((a, b, kb), (b, a, ka)):
        plane, axes, _, _ = key
        for p, q in zip(own, (*own[1:], own[0]), strict=True):
            low = sum((plane[axis] * p[axis] for axis in range(3)), plane[3])
            high = sum((plane[axis] * q[axis] for axis in range(3)), plane[3])
            if low == 0 and _triangle_contains(p, other, axes):
                hits.add(p)
            if low * high < 0:
                ratio = low / (low - high)
                point = tuple(x + ratio * (y - x) for x, y in zip(p, q, strict=True))
                if _triangle_contains(point, other, axes):
                    hits.add(point)
            if coplanar:
                for r, s in zip(other, (*other[1:], other[0]), strict=True):
                    pp, qq, rr, ss = project((p, q, r, s), axes)
                    first_side, second_side = turn(rr, ss, pp), turn(rr, ss, qq)
                    if (
                        first_side * second_side < 0
                        and turn(pp, qq, rr) * turn(pp, qq, ss) <= 0
                    ):
                        ratio = first_side / (first_side - second_side)
                        hits.add(
                            tuple(x + ratio * (y - x) for x, y in zip(p, q, strict=True))
                        )
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
