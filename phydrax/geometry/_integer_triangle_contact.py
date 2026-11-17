#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact triangle contacts using homogeneous integer coordinates."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from ._exact_polyhedral_geometry import Point3


type IntPoint = tuple[int, int, int]
type IntTriangle = tuple[IntPoint, IntPoint, IntPoint]
type HomogeneousPoint = tuple[IntPoint, int]
type IntegerTriangle = tuple[IntTriangle, int]


def integer_triangle(points: tuple[Point3, Point3, Point3], /) -> IntegerTriangle | None:
    """Clear all coordinate denominators once for a rational triangle."""
    scalars = tuple(value for point in points for value in point)
    if any(not isinstance(value, Fraction) for value in scalars):
        return None
    denominator = lcm(
        *(value.denominator for value in scalars if isinstance(value, Fraction))
    )
    rows = tuple(
        tuple(
            value.numerator * (denominator // value.denominator)
            for value in point
            if isinstance(value, Fraction)
        )
        for point in points
    )
    return (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    ), denominator


def _sub(a: IntPoint, b: IntPoint, /) -> IntPoint:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _cross(a: IntPoint, b: IntPoint, /) -> IntPoint:
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def _dot(a: IntPoint, b: IntPoint, /) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _relative(point: HomogeneousPoint, origin: IntPoint, /) -> IntPoint:
    p, d = point
    return p[0] - d * origin[0], p[1] - d * origin[1], p[2] - d * origin[2]


def _inside(point: HomogeneousPoint, triangle: IntTriangle, normal: IntPoint, /) -> bool:
    if _dot(normal, _relative(point, triangle[0])):
        return False
    return all(
        _dot(
            normal,
            _cross(
                _sub(triangle[(i + 1) % 3], triangle[i]), _relative(point, triangle[i])
            ),
        )
        >= 0
        for i in range(3)
    )


def _interpolate(
    a: IntPoint, b: IntPoint, numerator: int, denominator: int, /
) -> HomogeneousPoint:
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    return (
        (
            a[0] * denominator + (b[0] - a[0]) * numerator,
            a[1] * denominator + (b[1] - a[1]) * numerator,
            a[2] * denominator + (b[2] - a[2]) * numerator,
        ),
        denominator,
    )


def _cross2(a: IntPoint, b: IntPoint, axes: tuple[int, int], /) -> int:
    return a[axes[0]] * b[axes[1]] - a[axes[1]] * b[axes[0]]


def _between_zero_and(value: int, denominator: int, /) -> bool:
    return 0 <= value <= denominator if denominator > 0 else denominator <= value <= 0


def _segment_intersections(
    a: IntPoint, b: IntPoint, c: IntPoint, d: IntPoint, axes: tuple[int, int], /
) -> tuple[HomogeneousPoint, ...]:
    r, s, delta = _sub(b, a), _sub(d, c), _sub(c, a)
    denominator = _cross2(r, s, axes)
    if denominator:
        t, u = _cross2(delta, s, axes), _cross2(delta, r, axes)
        if _between_zero_and(t, denominator) and _between_zero_and(u, denominator):
            return (_interpolate(a, b, t, denominator),)
        return ()
    if _cross2(delta, r, axes):
        return ()
    return tuple(
        (point, 1)
        for point in (a, b, c, d)
        if all(min(x, y) <= z <= max(x, y) for z, x, y in zip(point, a, b, strict=True))
        and all(min(x, y) <= z <= max(x, y) for z, x, y in zip(point, c, d, strict=True))
    )


def _edge_contacts(
    a: IntPoint, b: IntPoint, triangle: IntTriangle, normal: IntPoint, /
) -> tuple[HomogeneousPoint, ...]:
    side_a = _dot(normal, _sub(a, triangle[0]))
    side_b = _dot(normal, _sub(b, triangle[0]))
    if side_a == side_b == 0:
        drop = max(range(3), key=lambda axis: abs(normal[axis]))
        selected = tuple(axis for axis in range(3) if axis != drop)
        axes = (selected[0], selected[1])
        result = [(p, 1) for p in (a, b) if _inside((p, 1), triangle, normal)]
        for i in range(3):
            result.extend(
                _segment_intersections(a, b, triangle[i], triangle[(i + 1) % 3], axes)
            )
        return tuple(result)
    if side_a and side_b and (side_a > 0) == (side_b > 0):
        return ()
    point = _interpolate(a, b, side_a, side_a - side_b)
    return (point,) if _inside(point, triangle, normal) else ()


def _scale_triangle(points: IntTriangle, factor: int, /) -> IntTriangle:
    def scale_point(point: IntPoint, /) -> IntPoint:
        return point[0] * factor, point[1] * factor, point[2] * factor

    return scale_point(points[0]), scale_point(points[1]), scale_point(points[2])


def integer_triangle_contacts(
    first: IntegerTriangle, second: IntegerTriangle, /
) -> tuple[int, tuple[HomogeneousPoint, ...], int]:
    """Return degeneracy, homogeneous intersection vertices, and coordinate scale."""
    first_points, first_scale = first
    second_points, second_scale = second
    scale = lcm(first_scale, second_scale)
    a = _scale_triangle(first_points, scale // first_scale)
    b = _scale_triangle(second_points, scale // second_scale)
    if any(
        max(p[axis] for p in a) < min(p[axis] for p in b)
        or max(p[axis] for p in b) < min(p[axis] for p in a)
        for axis in range(3)
    ):
        return 0, (), scale
    normal_a = _cross(_sub(a[1], a[0]), _sub(a[2], a[0]))
    normal_b = _cross(_sub(b[1], b[0]), _sub(b[2], b[0]))
    if not any(normal_a) or not any(normal_b):
        return 1, (), scale
    contacts: list[HomogeneousPoint] = []
    for i in range(3):
        contacts.extend(_edge_contacts(a[i], a[(i + 1) % 3], b, normal_b))
        contacts.extend(_edge_contacts(b[i], b[(i + 1) % 3], a, normal_a))
    return 0, tuple(contacts), scale


def contact_at_point(point: HomogeneousPoint, scale: int, target: Point3, /) -> bool:
    numerator, denominator = point
    return all(
        isinstance(value, Fraction)
        and number * value.denominator == value.numerator * denominator * scale
        for number, value in zip(numerator, target, strict=True)
    )


def contact_on_segment(
    point: HomogeneousPoint, scale: int, first: Point3, second: Point3, /
) -> bool:
    values = (*first, *second)
    if any(not isinstance(value, Fraction) for value in values):
        return False
    denominator = lcm(
        *(value.denominator for value in values if isinstance(value, Fraction))
    )
    a_values = tuple(
        value.numerator * (denominator // value.denominator)
        for value in first
        if isinstance(value, Fraction)
    )
    b_values = tuple(
        value.numerator * (denominator // value.denominator)
        for value in second
        if isinstance(value, Fraction)
    )
    a: IntPoint = (a_values[0] * scale, a_values[1] * scale, a_values[2] * scale)
    b: IntPoint = (b_values[0] * scale, b_values[1] * scale, b_values[2] * scale)
    p, d = point
    relative: IntPoint = (
        p[0] * denominator - a[0] * d,
        p[1] * denominator - a[1] * d,
        p[2] * denominator - a[2] * d,
    )
    direction = _sub(b, a)
    return not any(_cross(direction, relative)) and all(
        min(0, value * d) <= delta <= max(0, value * d)
        for delta, value in zip(relative, direction, strict=True)
    )
