#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact planar cell-chain triangulation and boundary checks."""

from __future__ import annotations

from fractions import Fraction

from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ._exact_polyhedral_geometry import (
    orient2,
    Point2,
    Point3,
    point_in_triangle2,
    segment_contact_points3,
)


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return (value > 0) - (value < 0)


def _projection(points: tuple[tuple[ExactScalar, ...], ...], /) -> tuple[Point2, ...]:
    if not points:
        raise ValueError("A planar polygon must contain vertices.")
    ambient = len(points[0])
    if ambient == 2:
        if any(len(point) != 2 for point in points):
            raise ValueError("Planar polygon points have inconsistent ambient dimension.")
        return tuple((point[0], point[1]) for point in points)
    if ambient != 3 or any(len(point) != 3 for point in points):
        raise ValueError("Planar polygon points must be two- or three-dimensional.")
    origin = points[0]
    normal: Point3 | None = None
    for first in range(1, len(points)):
        for second in range(first + 1, len(points)):
            a = tuple(points[first][axis] - origin[axis] for axis in range(3))
            b = tuple(points[second][axis] - origin[axis] for axis in range(3))
            candidate = (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
            if any(candidate):
                normal = candidate
                break
        if normal is not None:
            break
    if normal is None:
        raise ValueError("Planar polygon is exactly degenerate.")
    if any(
        sum(
            (normal[axis] * (point[axis] - origin[axis]) for axis in range(3)),
            Fraction(0),
        )
        for point in points
    ):
        raise ValueError("Polygon vertices are not exactly coplanar.")
    dropped = max(range(3), key=lambda axis: abs(normal[axis]))
    axes = tuple(axis for axis in range(3) if axis != dropped)
    return tuple((point[axes[0]], point[axes[1]]) for point in points)


def _edge_contacts(
    first: tuple[ExactScalar, ...],
    second: tuple[ExactScalar, ...],
    third: tuple[ExactScalar, ...],
    fourth: tuple[ExactScalar, ...],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    if len(first) == 2:
        points = tuple(
            (point[0], point[1], Fraction(0)) for point in (first, second, third, fourth)
        )
        return segment_contact_points3(*points)
    points3 = tuple((x, y, z) for x, y, z in (first, second, third, fourth))
    return segment_contact_points3(*points3)


def triangulate_planar_polygon(
    points: tuple[tuple[ExactScalar, ...], ...], /
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate a simple exact polygon without changing its represented boundary."""
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("A polygon must have at least three distinct exact vertices.")
    projected = _projection(points)
    area2 = sum(
        (
            projected[index][0] * projected[(index + 1) % len(points)][1]
            - projected[index][1] * projected[(index + 1) % len(points)][0]
            for index in range(len(points))
        ),
        Fraction(0),
    )
    orientation = _sign(area2)
    if orientation == 0:
        raise ValueError("A planar polygon has zero exact area.")
    count = len(points)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first == second or first_next == second or second_next == first:
                continue
            if _edge_contacts(
                points[first], points[first_next], points[second], points[second_next]
            ):
                raise ValueError("Polygon boundary self-intersects or self-touches.")
    remaining = list(range(count))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            if (
                _sign(
                    orient2(projected[previous], projected[current], projected[following])
                )
                != orientation
            ):
                continue
            triangle = (projected[previous], projected[current], projected[following])
            if any(
                point_in_triangle2(projected[index], triangle)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[position]
            ear_found = True
            break
        if not ear_found:
            raise ValueError(
                "No exact simple-polygon ear is available for triangulation."
            )
    if _sign(orient2(*(projected[index] for index in remaining))) == 0:
        raise ValueError("Polygon triangulation produced a degenerate final triangle.")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return tuple(triangles)


def point_in_convex_planar_hull(
    point: tuple[ExactScalar, ...],
    vertices: tuple[tuple[ExactScalar, ...], ...],
    /,
) -> bool:
    """Decide exact membership in the planar convex hull of unordered vertices."""
    projected = _projection((*vertices, point))
    test = projected[-1]
    points = sorted(set(projected[:-1]))
    if not points:
        return False
    if len(points) == 1:
        return test == points[0]
    if len(points) == 2:
        first, second = points
        return orient2(first, second, test) == 0 and all(
            min(first[axis], second[axis]) <= test[axis] <= max(first[axis], second[axis])
            for axis in range(2)
        )

    lower: list[Point2] = []
    for candidate in points:
        while len(lower) >= 2 and _sign(orient2(lower[-2], lower[-1], candidate)) <= 0:
            lower.pop()
        lower.append(candidate)
    upper: list[Point2] = []
    for candidate in reversed(points):
        while len(upper) >= 2 and _sign(orient2(upper[-2], upper[-1], candidate)) <= 0:
            upper.pop()
        upper.append(candidate)
    hull = (*lower[:-1], *upper[:-1])
    signs = tuple(
        _sign(orient2(hull[index], hull[(index + 1) % len(hull)], test))
        for index in range(len(hull))
    )
    return not (any(value > 0 for value in signs) and any(value < 0 for value in signs))


__all__ = ["point_in_convex_planar_hull", "triangulate_planar_polygon"]
