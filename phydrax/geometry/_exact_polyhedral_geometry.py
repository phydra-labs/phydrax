#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact contact predicates for affine simplices and boundary triangles."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement


type Point2 = tuple[ExactScalar, ExactScalar]
type Point3 = tuple[ExactScalar, ExactScalar, ExactScalar]
type Point = Point2 | Point3
type _Interval = tuple[NDArray[np.float64], NDArray[np.float64]]


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return (value > 0) - (value < 0)


def _cross2(first: Point2, second: Point2, /) -> ExactScalar:
    return first[0] * second[1] - first[1] * second[0]


def _sub2(first: Point2, second: Point2, /) -> Point2:
    return first[0] - second[0], first[1] - second[1]


def _add_scaled2(origin: Point2, direction: Point2, scale: ExactScalar, /) -> Point2:
    return origin[0] + scale * direction[0], origin[1] + scale * direction[1]


def _dot3(first: Point3, second: Point3, /) -> ExactScalar:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)), Fraction(0)
    )


def _cross3(first: Point3, second: Point3, /) -> Point3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _sub3(first: Point3, second: Point3, /) -> Point3:
    return first[0] - second[0], first[1] - second[1], first[2] - second[2]


def _add_scaled3(origin: Point3, direction: Point3, scale: ExactScalar, /) -> Point3:
    return (
        origin[0] + scale * direction[0],
        origin[1] + scale * direction[1],
        origin[2] + scale * direction[2],
    )


def orient2(first: Point2, second: Point2, third: Point2, /) -> ExactScalar:
    return _cross2(_sub2(second, first), _sub2(third, first))


def projected_convex_hull(
    indices: Sequence[int],
    vertices: Sequence[Sequence[ExactScalar]],
    normal: Sequence[ExactScalar],
    offset: ExactScalar,
    /,
) -> tuple[int, ...]:
    """Return the exact cyclic boundary, retaining collinear edge vertices."""
    rows = tuple(indices)
    if (
        len(rows) < 3
        or len(set(rows)) != len(rows)
        or any(row < 0 or row >= len(vertices) for row in rows)
        or len(normal) != 3
        or any(len(vertices[row]) != 3 for row in rows)
    ):
        raise ValueError("Projected exact hull inputs are malformed.")
    normal3: Point3 = (normal[0], normal[1], normal[2])
    coordinates: dict[int, Point3] = {
        row: (vertices[row][0], vertices[row][1], vertices[row][2]) for row in rows
    }
    if not any(normal3):
        raise ValueError("Projected exact hull has a zero normal.")
    if any(_dot3(normal3, coordinates[row]) != offset for row in rows):
        raise ValueError("Projected exact hull points are not exactly coplanar.")
    dropped_axis = max(range(3), key=lambda axis: abs(normal3[axis]))
    axes = tuple(axis for axis in range(3) if axis != dropped_axis)
    projected = {
        row: (coordinates[row][axes[0]], coordinates[row][axes[1]]) for row in rows
    }
    if len(set(projected.values())) != len(rows):
        raise ValueError("Projected exact hull contains duplicate points.")
    ordered = sorted(
        rows,
        key=lambda row: (projected[row][0], projected[row][1], row),
    )
    lower: list[int] = []
    for row in ordered:
        while (
            len(lower) >= 2
            and orient2(projected[lower[-2]], projected[lower[-1]], projected[row]) <= 0
        ):
            lower.pop()
        lower.append(row)
    upper: list[int] = []
    for row in reversed(ordered):
        while (
            len(upper) >= 2
            and orient2(projected[upper[-2]], projected[upper[-1]], projected[row]) <= 0
        ):
            upper.pop()
        upper.append(row)
    strict_hull = tuple(lower[:-1] + upper[:-1])
    if len(strict_hull) < 3:
        raise ValueError("Projected exact hull has fewer than three extreme vertices.")

    boundary: list[int] = []
    for first, second in zip(
        strict_hull, (*strict_hull[1:], strict_hull[0]), strict=True
    ):
        first_point = coordinates[first]
        second_point = coordinates[second]
        axis = next(axis for axis in axes if first_point[axis] != second_point[axis])
        denominator = second_point[axis] - first_point[axis]
        edge_points = [
            row
            for row in rows
            if orient2(projected[first], projected[second], projected[row]) == 0
            and 0 <= (coordinates[row][axis] - first_point[axis]) / denominator <= 1
        ]
        edge_points.sort(
            key=lambda row: (coordinates[row][axis] - first_point[axis]) / denominator
        )
        if not edge_points or edge_points[0] != first or edge_points[-1] != second:
            raise ValueError("Projected exact hull has an inconsistent boundary edge.")
        boundary.extend(edge_points[:-1])
    if len(boundary) != len(rows) or set(boundary) != set(rows):
        raise ValueError("Projected exact hull contains a non-boundary input vertex.")
    return tuple(boundary)


def determinant3(
    first: Point3, second: Point3, third: Point3, fourth: Point3, /
) -> ExactScalar:
    return _dot3(_cross3(_sub3(second, first), _sub3(third, first)), _sub3(fourth, first))


def _between(value: ExactScalar, first: ExactScalar, second: ExactScalar, /) -> bool:
    return min(first, second) <= value <= max(first, second)


def point_on_segment2(point: Point2, first: Point2, second: Point2, /) -> bool:
    return orient2(first, second, point) == 0 and all(
        _between(value, a, b) for value, a, b in zip(point, first, second, strict=True)
    )


def _segment_contact_points2(
    first: Point2, second: Point2, third: Point2, fourth: Point2, /
) -> tuple[Point2, ...]:
    r, s = _sub2(second, first), _sub2(fourth, third)
    denominator = _cross2(r, s)
    delta = _sub2(third, first)
    if denominator:
        t = _cross2(delta, s) / denominator
        u = _cross2(delta, r) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (_add_scaled2(first, r, t),)
        return ()
    if _cross2(delta, r):
        return ()
    return tuple(
        point
        for point in (first, second, third, fourth)
        if point_on_segment2(point, first, second)
        and point_on_segment2(point, third, fourth)
    )


def point_in_triangle2(point: Point2, triangle: tuple[Point2, Point2, Point2], /) -> bool:
    first, second, third = triangle
    signs = tuple(
        _sign(orient2(a, b, point))
        for a, b in ((first, second), (second, third), (third, first))
    )
    return not (any(value > 0 for value in signs) and any(value < 0 for value in signs))


def triangle_contact_points2(
    first: tuple[Point2, Point2, Point2],
    second: tuple[Point2, Point2, Point2],
    /,
) -> tuple[Point2, ...]:
    """Return exact vertices of the closed intersection of two planar triangles."""
    points: set[Point2] = set()
    points.update(point for point in first if point_in_triangle2(point, second))
    points.update(point for point in second if point_in_triangle2(point, first))
    for left, right in ((0, 1), (1, 2), (2, 0)):
        for other_left, other_right in ((0, 1), (1, 2), (2, 0)):
            points.update(
                _segment_contact_points2(
                    first[left], first[right], second[other_left], second[other_right]
                )
            )
    return tuple(sorted(points, key=lambda point: tuple(map(float, point))))


def _project(point: Point3, dropped_axis: int, /) -> Point2:
    axes = tuple(axis for axis in range(3) if axis != dropped_axis)
    return point[axes[0]], point[axes[1]]


def _plane_normal(triangle: tuple[Point3, Point3, Point3], /) -> Point3:
    return _cross3(_sub3(triangle[1], triangle[0]), _sub3(triangle[2], triangle[0]))


def _drop_axis(normal: Point3, /) -> int:
    return max(range(3), key=lambda axis: abs(normal[axis]))


def point_in_triangle3(point: Point3, triangle: tuple[Point3, Point3, Point3], /) -> bool:
    normal = _plane_normal(triangle)
    if not _dot3(normal, _sub3(point, triangle[0])):
        drop = _drop_axis(normal)
        return point_in_triangle2(
            _project(point, drop),
            (
                _project(triangle[0], drop),
                _project(triangle[1], drop),
                _project(triangle[2], drop),
            ),
        )
    return False


def _segment_plane_points(
    first: Point3,
    second: Point3,
    triangle: tuple[Point3, Point3, Point3],
    /,
) -> tuple[Point3, ...]:
    normal = _plane_normal(triangle)
    side_first = _dot3(normal, _sub3(first, triangle[0]))
    side_second = _dot3(normal, _sub3(second, triangle[0]))
    if side_first == 0 and side_second == 0:
        drop = _drop_axis(normal)
        projected_triangle = tuple(_project(row, drop) for row in triangle)
        projected_first, projected_second = _project(first, drop), _project(second, drop)
        result = [
            point for point in (first, second) if point_in_triangle3(point, triangle)
        ]
        for edge_first, edge_second in ((0, 1), (1, 2), (2, 0)):
            result.extend(
                _lift_projected(point, triangle[0], normal, drop)
                for point in _segment_contact_points2(
                    projected_first,
                    projected_second,
                    projected_triangle[edge_first],
                    projected_triangle[edge_second],
                )
            )
        return tuple(dict.fromkeys(result))
    if side_first == 0:
        return (first,) if point_in_triangle3(first, triangle) else ()
    if side_second == 0:
        return (second,) if point_in_triangle3(second, triangle) else ()
    if _sign(side_first) == _sign(side_second):
        return ()
    parameter = side_first / (side_first - side_second)
    point = _add_scaled3(first, _sub3(second, first), parameter)
    return (point,) if point_in_triangle3(point, triangle) else ()


def _lift_projected(
    point: Point2, triangle_origin: Point3, normal: Point3, dropped_axis: int, /
) -> Point3:
    axes = tuple(axis for axis in range(3) if axis != dropped_axis)
    result = [Fraction(0), Fraction(0), Fraction(0)]
    result[axes[0]], result[axes[1]] = point
    offset = _dot3(normal, triangle_origin)
    result[dropped_axis] = (
        offset - normal[axes[0]] * point[0] - normal[axes[1]] * point[1]
    ) / normal[dropped_axis]
    return result[0], result[1], result[2]


def triangle_contact_points3(
    first: tuple[Point3, Point3, Point3],
    second: tuple[Point3, Point3, Point3],
    /,
) -> tuple[Point3, ...]:
    """Return exact vertices of the closed intersection of two affine triangles."""
    normal_first, normal_second = _plane_normal(first), _plane_normal(second)
    if not any(normal_first) or not any(normal_second):
        raise ValueError("Triangle contact requires nondegenerate exact triangle images.")
    if all(not _dot3(normal_first, _sub3(point, first[0])) for point in second):
        drop = _drop_axis(normal_first)
        points2 = triangle_contact_points2(
            (
                _project(first[0], drop),
                _project(first[1], drop),
                _project(first[2], drop),
            ),
            (
                _project(second[0], drop),
                _project(second[1], drop),
                _project(second[2], drop),
            ),
        )
        return tuple(
            _lift_projected(point, first[0], normal_first, drop) for point in points2
        )

    points: set[Point3] = set()
    points.update(point for point in first if point_in_triangle3(point, second))
    points.update(point for point in second if point_in_triangle3(point, first))
    edges = ((0, 1), (1, 2), (2, 0))
    for left, right in edges:
        points.update(_segment_plane_points(first[left], first[right], second))
        points.update(_segment_plane_points(second[left], second[right], first))
    return tuple(sorted(points, key=lambda point: tuple(map(float, point))))


def point_on_segment3(point: Point3, first: Point3, second: Point3, /) -> bool:
    direction = _sub3(second, first)
    displacement = _sub3(point, first)
    if any(_cross3(direction, displacement)):
        return False
    axis = next((index for index, value in enumerate(direction) if value), None)
    if axis is None:
        return point == first
    return _between(point[axis], first[axis], second[axis])


def point_in_tetrahedron(point: Point3, tetrahedron: tuple[Point3, ...], /) -> bool:
    if len(tetrahedron) != 4:
        raise ValueError("A tetrahedron requires four exact vertices.")
    first, second, third, fourth = tetrahedron
    orientation = determinant3(first, second, third, fourth)
    if not orientation:
        raise ValueError("Tetrahedron containment requires a nondegenerate exact cell.")
    numerators = (
        determinant3(point, second, third, fourth),
        determinant3(first, point, third, fourth),
        determinant3(first, second, point, fourth),
        determinant3(first, second, third, point),
    )
    orientation_sign = _sign(orientation)
    return all(_sign(value) in (0, orientation_sign) for value in numerators)


def _support_contact_points(
    first: tuple[Point3, ...], second: tuple[Point3, ...], /
) -> tuple[Point3, ...]:
    """Intersect exact coplanar supporting features, each of dimension at most two."""
    match len(first), len(second):
        case 1, 1:
            return first if first[0] == second[0] else ()
        case 1, 2:
            return first if point_on_segment3(first[0], second[0], second[1]) else ()
        case 1, 3:
            triangle = (second[0], second[1], second[2])
            return first if point_in_triangle3(first[0], triangle) else ()
        case (2, 1) | (3, 1) | (3, 2):
            return _support_contact_points(second, first)
        case 2, 2:
            return segment_contact_points3(first[0], first[1], second[0], second[1])
        case 2, 3:
            return _segment_plane_points(
                first[0], first[1], (second[0], second[1], second[2])
            )
        case 3, 3:
            return triangle_contact_points3(
                (first[0], first[1], first[2]),
                (second[0], second[1], second[2]),
            )
        case _:
            raise ValueError(
                "A nonzero tetrahedron support must have one to three vertices."
            )


def _tetrahedron_contact_axes(
    first: tuple[Point3, ...], second: tuple[Point3, ...], /
) -> Iterator[Point3]:
    for cell in (first, second):
        for a, b, c in combinations(cell, 3):
            yield _plane_normal((a, b, c))
    second_edges = tuple(_sub3(b, a) for a, b in combinations(second, 2))
    for a, b in combinations(first, 2):
        first_edge = _sub3(b, a)
        for second_edge in second_edges:
            yield _cross3(first_edge, second_edge)


def _tetrahedron_support_contacts(
    first: tuple[Point3, ...], second: tuple[Point3, ...], /
) -> tuple[Point3, ...] | None:
    """Prove empty/lower-dimensional contact; None retains volumetric clipping."""
    for axis in _tetrahedron_contact_axes(first, second):
        if not any(axis):
            continue
        first_projection = tuple(_dot3(axis, point) for point in first)
        second_projection = tuple(_dot3(axis, point) for point in second)
        first_lower, first_upper = min(first_projection), max(first_projection)
        second_lower, second_upper = min(second_projection), max(second_projection)
        if first_upper < second_lower or second_upper < first_lower:
            return ()
        if first_upper == second_lower:
            contact_plane = first_upper
        elif second_upper == first_lower:
            contact_plane = second_upper
        else:
            continue
        # Every intersection point lies on this common supporting plane, so
        # intersecting its complete support features loses no contact.
        first_support = tuple(
            point
            for point, projection in zip(first, first_projection, strict=True)
            if projection == contact_plane
        )
        second_support = tuple(
            point
            for point, projection in zip(second, second_projection, strict=True)
            if projection == contact_plane
        )
        contacts = _support_contact_points(first_support, second_support)
        return tuple(sorted(set(contacts), key=lambda point: tuple(map(float, point))))
    return None


def tetrahedron_contact_points(
    first: tuple[Point3, ...], second: tuple[Point3, ...], /
) -> tuple[Point3, ...]:
    """Return exact vertices of the closed intersection of two affine tetrahedra."""
    if len(first) != 4 or len(second) != 4:
        raise ValueError("Tetrahedron contact requires two four-vertex cells.")
    if not determinant3(*first) or not determinant3(*second):
        raise ValueError("Tetrahedron contact requires nondegenerate exact cells.")
    support_contacts = _tetrahedron_support_contacts(first, second)
    if support_contacts is not None:
        return support_contacts
    points: set[Point3] = set()
    points.update(point for point in first if point_in_tetrahedron(point, second))
    points.update(point for point in second if point_in_tetrahedron(point, first))
    routes = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    for first_route in routes:
        first_face = (first[first_route[0]], first[first_route[1]], first[first_route[2]])
        for second_route in routes:
            second_face = (
                second[second_route[0]],
                second[second_route[1]],
                second[second_route[2]],
            )
            points.update(triangle_contact_points3(first_face, second_face))
    return tuple(sorted(points, key=lambda point: tuple(map(float, point))))


def segment_contact_points3(
    first: Point3, second: Point3, third: Point3, fourth: Point3, /
) -> tuple[Point3, ...]:
    """Return exact contact points for two closed spatial segments."""
    left = _sub3(second, first)
    right = _sub3(fourth, third)
    normal = _cross3(left, right)
    delta = _sub3(third, first)
    if not any(normal):
        if any(_cross3(delta, left)):
            return ()
        return tuple(
            point
            for point in (first, second, third, fourth)
            if point_on_segment3(point, first, second)
            and point_on_segment3(point, third, fourth)
        )
    if _dot3(delta, normal):
        return ()
    axis = next(index for index, value in enumerate(normal) if value)
    projected = _segment_contact_points2(
        _project(first, axis),
        _project(second, axis),
        _project(third, axis),
        _project(fourth, axis),
    )
    points = tuple(_lift_projected(point, first, normal, axis) for point in projected)
    return tuple(
        point
        for point in points
        if point_on_segment3(point, first, second)
        and point_on_segment3(point, third, fourth)
    )


def _rational_point_enclosures(
    points: Sequence[Point3], /
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Binary64 outward enclosures of exact spatial points.

    `float` rounds integers and fractions correctly, so one unit in the last place
    on each side encloses each coordinate, and exactly representable values keep
    a point interval.  Points with a radical or non-finite coordinate stay
    unbounded.
    """
    rounded = np.full((len(points), 3), np.nan, dtype=np.float64)
    exact = np.zeros((len(points), 3), dtype=np.bool_)
    for row, point in enumerate(points):
        if not all(isinstance(value, (int, Fraction)) for value in point):
            continue
        try:
            approximation = tuple(float(value) for value in point)
        except OverflowError:
            continue
        rounded[row] = approximation
        exact[row] = tuple(
            value == estimate
            for value, estimate in zip(point, approximation, strict=True)
        )
    finite = np.all(np.isfinite(rounded), axis=1, keepdims=True)
    lower = np.where(exact, rounded, np.nextafter(rounded, -np.inf))
    upper = np.where(exact, rounded, np.nextafter(rounded, np.inf))
    return np.where(finite, lower, -np.inf), np.where(finite, upper, np.inf)


def _interval_take(value: _Interval, index: Any, /) -> _Interval:
    return value[0][index], value[1][index]


# Each binary64 operation below is rounded to nearest, so stepping its result
# one unit in the last place outward encloses the exact real operation.
def _interval_sub(first: _Interval, second: _Interval, /) -> _Interval:
    return (
        np.nextafter(first[0] - second[1], -np.inf),
        np.nextafter(first[1] - second[0], np.inf),
    )


def _interval_add(first: _Interval, second: _Interval, /) -> _Interval:
    return (
        np.nextafter(first[0] + second[0], -np.inf),
        np.nextafter(first[1] + second[1], np.inf),
    )


def _interval_mul(first: _Interval, second: _Interval, /) -> _Interval:
    products = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    return (
        np.nextafter(np.minimum.reduce(products), -np.inf),
        np.nextafter(np.maximum.reduce(products), np.inf),
    )


def _interval_cross(first: _Interval, second: _Interval, /) -> _Interval:
    components = tuple(
        _interval_sub(
            _interval_mul(
                _interval_take(first, (..., (axis + 1) % 3)),
                _interval_take(second, (..., (axis + 2) % 3)),
            ),
            _interval_mul(
                _interval_take(first, (..., (axis + 2) % 3)),
                _interval_take(second, (..., (axis + 1) % 3)),
            ),
        )
        for axis in range(3)
    )
    return (
        np.stack([component[0] for component in components], axis=-1),
        np.stack([component[1] for component in components], axis=-1),
    )


def _interval_dot(first: _Interval, second: _Interval, /) -> _Interval:
    total = _interval_mul(
        _interval_take(first, (..., 0)), _interval_take(second, (..., 0))
    )
    for axis in (1, 2):
        total = _interval_add(
            total,
            _interval_mul(
                _interval_take(first, (..., axis)), _interval_take(second, (..., axis))
            ),
        )
    return total


def _certified_sign(value: _Interval, /) -> NDArray[np.int8]:
    """+1 or -1 where the enclosure excludes zero, 0 where the sign is undecided."""
    return np.where(value[0] > 0, 1, np.where(value[1] < 0, -1, 0)).astype(np.int8)


# Face `m` of a tetrahedron omits corner `m`; its plane passes through the
# remaining corners in this order.
_FACE_CORNERS = np.asarray(((1, 2, 3), (2, 3, 0), (3, 0, 1), (0, 1, 2)), dtype=np.intp)
_EDGE_CORNERS = np.asarray(tuple(combinations(range(4), 2)), dtype=np.intp)
# The two corners outside edge `(i, j)`, for `i != j`.
_EDGE_COMPLEMENT = np.asarray(
    [
        [
            [corner for corner in range(4) if corner not in (first, second)][:2]
            for second in range(4)
        ]
        for first in range(4)
    ],
    dtype=np.intp,
)


@dataclass(frozen=True, slots=True)
class TetrahedronEnclosures:
    """Binary64 corner enclosures and certified face planes of exact tetrahedra.

    `point_ids` name exact corner points: equal identities denote equal exact
    points and distinct identities distinct points.  `apex_signs[t, m]` is the
    certified sign of corner `m` relative to the plane of face `m` (zero when
    undecided); `nondegenerate[t]` holds when one of them is certified, so the
    exact orientation determinant of tetrahedron `t` is nonzero.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    point_ids: NDArray[np.int64]
    normals: _Interval
    apex_signs: NDArray[np.int8]
    nondegenerate: NDArray[np.bool_]
    centers: NDArray[np.float64]
    face_directions: NDArray[np.float64]
    face_offsets: NDArray[np.float64]

    @classmethod
    def prepare(
        cls,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        point_ids: NDArray[np.int64],
        /,
    ) -> TetrahedronEnclosures:
        """Prepare `(T, 4, 3)` corner enclosures with `(T, 4)` exact point identities.

        `centers`, `face_directions` (unit normals pointing away from each apex)
        and `face_offsets` are midpoint estimates that only rank candidate planes.
        """
        corners = (lower, upper)
        origin = _interval_take(corners, (slice(None), _FACE_CORNERS[:, 0]))
        normals = _interval_cross(
            _interval_sub(
                _interval_take(corners, (slice(None), _FACE_CORNERS[:, 1])), origin
            ),
            _interval_sub(
                _interval_take(corners, (slice(None), _FACE_CORNERS[:, 2])), origin
            ),
        )
        apex_signs = _certified_sign(
            _interval_dot(normals, _interval_sub(corners, origin))
        )
        finite = np.all(np.isfinite(lower), axis=(1, 2)) & np.all(
            np.isfinite(upper), axis=(1, 2)
        )
        apex_signs = np.where(finite[:, None], apex_signs, 0).astype(np.int8)
        centers = (lower + upper) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            directions = (normals[0] + normals[1]) / 2
            directions = np.where(
                (apex_signs != 0)[:, :, None],
                -apex_signs[:, :, None]
                * directions
                / np.linalg.norm(directions, axis=2, keepdims=True),
                np.nan,
            )
        return cls(
            lower,
            upper,
            point_ids,
            normals,
            apex_signs,
            np.any(apex_signs != 0, axis=1),
            centers,
            directions,
            np.einsum("tmk,tmk->tm", directions, centers[:, _FACE_CORNERS[:, 0]]),
        )

    def common_hull_contacts(
        self, first: NDArray[np.intp], second: NDArray[np.intp], /
    ) -> NDArray[np.bool_]:
        """Certify pairs whose closed intersection lies in the hull of common corners.

        Common corners are those with equal point identities.  A pair is certified
        only when both tetrahedra are certified nondegenerate and one of these
        outward-rounded tests holds:

        - their boxes, or an axis across a pair of their edges, strictly separate
          them (no common corner; the intersection is empty);
        - a face plane of one has every corner of the other strictly opposite its
          apex or equal to a corner of that face, so the plane meets the other
          tetrahedron only in the hull of those equal corners;
        - for one common corner `a`, the plane through `a` and edges `ab`, `ad`
          of the two tetrahedra has their remaining corners strictly on opposite
          sides, so the intersection lies in `ab ∩ ad = {a}`.
        """
        certified = np.zeros(first.shape, dtype=np.bool_)
        admitted = self.nondegenerate[first] & self.nondegenerate[second]
        equal = self.point_ids[first][:, :, None] == self.point_ids[second][:, None, :]
        common = np.any(equal, axis=(1, 2))
        disjoint_boxes = np.any(
            (self.upper[first].max(axis=1) < self.lower[second].min(axis=1))
            | (self.upper[second].max(axis=1) < self.lower[first].min(axis=1)),
            axis=1,
        )
        certified[admitted & ~common & disjoint_boxes] = True
        rows = np.flatnonzero(admitted & ~certified)
        if rows.size:
            side, face = self._widest_face(first[rows], second[rows], equal[rows])
            certified[rows] = self._face_plane_supports(
                np.where(side == 0, first[rows], second[rows]),
                np.where(side == 0, second[rows], first[rows]),
                face,
            )
        rows = np.flatnonzero(admitted & ~certified & (np.sum(equal, axis=(1, 2)) == 1))
        if rows.size:
            certified[rows] = self._shared_corner_planes_separate(
                first[rows],
                second[rows],
                np.argmax(np.any(equal[rows], axis=2), axis=1),
                np.argmax(np.any(equal[rows], axis=1), axis=1),
            )
        rows = np.flatnonzero(admitted & ~certified & ~common)
        if rows.size:
            certified[rows] = self._edge_axis_separates(first[rows], second[rows])
        return certified

    def _widest_face(
        self, first: NDArray[np.intp], second: NDArray[np.intp], equal: NDArray, /
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Midpoint estimate of the face plane with the widest clearance.

        Only selects which plane the outward-rounded test examines; it certifies
        nothing.
        """
        scores = []
        for plane, other, on_plane in (
            (first, second, equal),
            (second, first, np.swapaxes(equal, 1, 2)),
        ):
            clearance = (
                np.einsum(
                    "rmk,rvk->rmv", self.face_directions[plane], self.centers[other]
                )
                - self.face_offsets[plane][:, :, None]
            )
            clearance[np.any(on_plane[:, _FACE_CORNERS, :], axis=2)] = np.inf
            scores.append(clearance.min(axis=2))
        score = np.concatenate(scores, axis=1)
        choice = np.argmax(np.where(np.isnan(score), -np.inf, score), axis=1)
        return choice // 4, choice % 4

    def _face_plane_supports(
        self, plane: NDArray[np.intp], other: NDArray[np.intp], face: NDArray[np.intp], /
    ) -> NDArray[np.bool_]:
        corners = (self.lower, self.upper)
        apex = self.apex_signs[plane, face]
        origin = _interval_take(corners, (plane, _FACE_CORNERS[face, 0]))
        normal = _interval_take(self.normals, (plane, face))
        face_ids = self.point_ids[plane[:, None], _FACE_CORNERS[face]]
        signs = _certified_sign(
            _interval_dot(
                (normal[0][:, None], normal[1][:, None]),
                _interval_sub(
                    _interval_take(corners, other),
                    (origin[0][:, None], origin[1][:, None]),
                ),
            )
        )
        on_face = np.any(
            self.point_ids[other][:, :, None] == face_ids[:, None, :], axis=2
        )
        return (apex != 0) & np.all(on_face | (signs == -apex[:, None]), axis=1)

    def _shared_corner_planes_separate(
        self,
        first: NDArray[np.intp],
        second: NDArray[np.intp],
        first_corner: NDArray[np.intp],
        second_corner: NDArray[np.intp],
        /,
    ) -> NDArray[np.bool_]:
        corners = (self.lower, self.upper)
        apex = _interval_take(corners, (first, first_corner))
        separated = np.zeros(first.shape, dtype=np.bool_)
        for first_shift, second_shift in product(range(1, 4), repeat=2):
            first_edge = (first_corner + first_shift) % 4
            second_edge = (second_corner + second_shift) % 4
            normal = _interval_cross(
                _interval_sub(_interval_take(corners, (first, first_edge)), apex),
                _interval_sub(_interval_take(corners, (second, second_edge)), apex),
            )
            sides = []
            for cell, rest in (
                (first, _EDGE_COMPLEMENT[first_corner, first_edge]),
                (second, _EDGE_COMPLEMENT[second_corner, second_edge]),
            ):
                sides.append(
                    _certified_sign(
                        _interval_dot(
                            (normal[0][:, None], normal[1][:, None]),
                            _interval_sub(
                                _interval_take(corners, (cell[:, None], rest)),
                                (apex[0][:, None], apex[1][:, None]),
                            ),
                        )
                    )
                )
            first_sides, second_sides = sides
            separated |= (
                (first_sides[:, 0] != 0)
                & (first_sides[:, 1] == first_sides[:, 0])
                & (second_sides[:, 0] == -first_sides[:, 0])
                & (second_sides[:, 1] == -first_sides[:, 0])
            )
        return separated

    def _edge_axis_separates(
        self, first: NDArray[np.intp], second: NDArray[np.intp], /
    ) -> NDArray[np.bool_]:
        """Test the edge-pair axis with the widest midpoint gap, outward-rounded."""
        first_centers, second_centers = self.centers[first], self.centers[second]
        first_edges = (
            first_centers[:, _EDGE_CORNERS[:, 1]] - first_centers[:, _EDGE_CORNERS[:, 0]]
        )
        second_edges = (
            second_centers[:, _EDGE_CORNERS[:, 1]]
            - second_centers[:, _EDGE_CORNERS[:, 0]]
        )
        axes = np.cross(first_edges[:, :, None], second_edges[:, None, :]).reshape(
            (first.size, -1, 3)
        )
        first_projection = np.einsum("rak,rvk->rav", axes, first_centers)
        second_projection = np.einsum("rak,rvk->rav", axes, second_centers)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.maximum(
                second_projection.min(axis=2) - first_projection.max(axis=2),
                first_projection.min(axis=2) - second_projection.max(axis=2),
            ) / np.linalg.norm(axes, axis=2)
        left, right = np.divmod(
            np.argmax(np.where(np.isnan(gap), -np.inf, gap), axis=1), len(_EDGE_CORNERS)
        )
        corners = (self.lower, self.upper)
        first_corners = _interval_take(corners, first)
        second_corners = _interval_take(corners, second)
        rows = np.arange(first.size)
        axis = _interval_cross(
            _interval_sub(
                _interval_take(first_corners, (rows, _EDGE_CORNERS[left, 1])),
                _interval_take(first_corners, (rows, _EDGE_CORNERS[left, 0])),
            ),
            _interval_sub(
                _interval_take(second_corners, (rows, _EDGE_CORNERS[right, 1])),
                _interval_take(second_corners, (rows, _EDGE_CORNERS[right, 0])),
            ),
        )
        axis = (axis[0][:, None], axis[1][:, None])
        reference = (first_corners[0][:, :1], first_corners[1][:, :1])
        first_projection_bounds = _interval_dot(
            axis, _interval_sub(first_corners, reference)
        )
        second_projection_bounds = _interval_dot(
            axis, _interval_sub(second_corners, reference)
        )
        return (
            first_projection_bounds[1].max(axis=1)
            < second_projection_bounds[0].min(axis=1)
        ) | (
            second_projection_bounds[1].max(axis=1)
            < first_projection_bounds[0].min(axis=1)
        )


def _point_key(point: Point3, /) -> Hashable:
    """Exact identity key; rational coordinates key by their reduced integer pairs."""
    rationals = tuple(value for value in point if isinstance(value, (int, Fraction)))
    if len(rationals) == len(point):
        return tuple((value.numerator, value.denominator) for value in rationals)
    return point


def outward_tetrahedron_enclosures(
    tetrahedra: Sequence[tuple[Point3, ...]], /
) -> TetrahedronEnclosures:
    """Prepare binary64 enclosures with exact point identities for exact tetrahedra.

    Only rational points receive finite enclosures, and their keys are canonical,
    so on every admitted tetrahedron equal identities are exactly equal points.
    """
    identities: dict[Hashable, int] = {}
    points: list[Point3] = []
    rows: list[int] = []
    for tetrahedron in tetrahedra:
        for point in tetrahedron:
            key = _point_key(point)
            identifier = identities.get(key)
            if identifier is None:
                identifier = identities[key] = len(points)
                points.append(point)
            rows.append(identifier)
    point_ids = np.asarray(rows, dtype=np.int64).reshape((len(tetrahedra), 4))
    lower, upper = _rational_point_enclosures(points)
    return TetrahedronEnclosures.prepare(lower[point_ids], upper[point_ids], point_ids)


def contact_is_declared[Key: Hashable](
    points: tuple[Point, ...],
    first_ids: tuple[Key, ...],
    second_ids: tuple[Key, ...],
    id_to_point: Mapping[Key, Point],
    /,
) -> bool:
    """Whether every exact intersection vertex lies on the declared shared stratum."""
    if not points:
        return True
    second_set = set(second_ids)
    shared = tuple(identifier for identifier in first_ids if identifier in second_set)
    if len(shared) == 1:
        return all(point == id_to_point[shared[0]] for point in points)
    if len(shared) == 2:
        first, second = id_to_point[shared[0]], id_to_point[shared[1]]
        if len(first) == 2 and len(second) == 2:
            return all(
                len(point) == 2 and point_on_segment2(point, first, second)
                for point in points
            )
        if len(first) == 3 and len(second) == 3:
            return all(
                len(point) == 3 and point_on_segment3(point, first, second)
                for point in points
            )
        return False
    if len(shared) == 3:
        first, second, third = (id_to_point[identifier] for identifier in shared)
        if len(first) == 3 and len(second) == 3 and len(third) == 3:
            return all(
                len(point) == 3 and point_in_triangle3(point, (first, second, third))
                for point in points
            )
    return False


__all__ = [
    "Point",
    "Point2",
    "Point3",
    "contact_is_declared",
    "determinant3",
    "orient2",
    "point_in_tetrahedron",
    "point_in_triangle2",
    "point_in_triangle3",
    "point_on_segment2",
    "point_on_segment3",
    "projected_convex_hull",
    "segment_contact_points3",
    "tetrahedron_contact_points",
    "triangle_contact_points2",
    "triangle_contact_points3",
]
