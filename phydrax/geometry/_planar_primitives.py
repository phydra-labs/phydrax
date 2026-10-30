#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact rational primitives for planar source and candidate partitions."""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from typing import TypeAlias


Point: TypeAlias = tuple[Fraction, Fraction]
Polygon: TypeAlias = tuple[Point, ...]


def cross(first: Point, second: Point, third: Point, /) -> Fraction:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def signed_area2(polygon: Polygon, /) -> Fraction:
    if len(polygon) < 3:
        return Fraction(0)
    return sum(
        (
            first[0] * second[1] - first[1] * second[0]
            for first, second in zip(polygon, polygon[1:] + polygon[:1], strict=True)
        ),
        Fraction(0),
    )


def _clean(polygon: Polygon, /) -> Polygon:
    if not polygon:
        return ()
    points: list[Point] = []
    for point in polygon:
        if not points or points[-1] != point:
            points.append(point)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    changed = True
    while changed and len(points) >= 3:
        changed = False
        keep = []
        for index, point in enumerate(points):
            if cross(points[index - 1], point, points[(index + 1) % len(points)]) == 0:
                changed = True
            else:
                keep.append(point)
        points = keep
    return tuple(points)


def oriented_polygon(points: Polygon, /) -> Polygon:
    result = _clean(points)
    if signed_area2(result) < 0:
        result = tuple(reversed(result))
    return result


def intersection(first: Polygon, second: Polygon, /) -> Polygon:
    """Intersect two convex rational polygons without rounding predicates."""
    subject = oriented_polygon(first)
    clipper = oriented_polygon(second)
    if len(subject) < 3 or len(clipper) < 3:
        return ()
    for edge_start, edge_end in zip(clipper, clipper[1:] + clipper[:1], strict=True):
        if not subject:
            break
        output: list[Point] = []
        previous = subject[-1]
        previous_side = cross(edge_start, edge_end, previous)
        for current in subject:
            current_side = cross(edge_start, edge_end, current)
            previous_inside = previous_side >= 0
            current_inside = current_side >= 0
            if previous_inside != current_inside:
                denominator = previous_side - current_side
                if denominator:
                    parameter = previous_side / denominator
                    output.append(
                        (
                            previous[0] + parameter * (current[0] - previous[0]),
                            previous[1] + parameter * (current[1] - previous[1]),
                        )
                    )
            if current_inside:
                output.append(current)
            previous = current
            previous_side = current_side
        subject = _clean(tuple(output))
    return subject


def area2(polygon: Polygon, /) -> Fraction:
    return abs(signed_area2(polygon))


def _primitive_integer_row(values: tuple[Fraction, ...], /) -> tuple[int, ...]:
    denominator = reduce(lcm, (value.denominator for value in values), 1)
    integers = tuple(int(value * denominator) for value in values)
    divisor = reduce(gcd, (abs(value) for value in integers if value), 0)
    if divisor == 0:
        raise ValueError("A plane requires three non-collinear points.")
    normalized = tuple(value // divisor for value in integers)
    first = next(value for value in normalized if value)
    return tuple(-value for value in normalized) if first < 0 else normalized


def plane_key(points: tuple[tuple[Fraction, ...], ...], /) -> tuple[int, int, int, int]:
    """Canonical exact plane equation through three 3-D points."""
    if len(points) < 3 or any(len(point) != 3 for point in points):
        raise ValueError("plane_key requires at least three points in 3-D.")
    origin = points[0]
    first = tuple(points[1][axis] - origin[axis] for axis in range(3))
    second = tuple(points[2][axis] - origin[axis] for axis in range(3))
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    if not any(normal):
        raise ValueError("plane_key requires non-collinear points.")
    offset = -sum((normal[axis] * origin[axis] for axis in range(3)), Fraction(0))
    key = _primitive_integer_row((*normal, offset))
    return (key[0], key[1], key[2], key[3])


def plane_orientation(points: tuple[tuple[Fraction, ...], ...], /) -> int:
    """Orientation of an ordered triangle relative to its canonical plane."""
    if len(points) < 3:
        raise ValueError("plane_orientation requires an ordered triangle.")
    raw = tuple(
        points[axis + 1][axis2] - points[0][axis2]
        for axis in range(2)
        for axis2 in range(3)
    )
    first = raw[:3]
    second = raw[3:]
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    canonical = plane_key(points)
    return 1 if sum(normal[i] * canonical[i] for i in range(3)) > 0 else -1


def projection_axes(plane: tuple[int, int, int, int], /) -> tuple[int, int]:
    normal = plane[:3]
    drop = max(range(3), key=lambda axis: abs(normal[axis]))
    return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[drop]


def project_polygon(
    points: tuple[tuple[Fraction, ...], ...], axes: tuple[int, int], /
) -> Polygon:
    return tuple((point[axes[0]], point[axes[1]]) for point in points)


def line_key(points: tuple[Point, Point], /) -> tuple[int, int, int]:
    first, second = points
    direction = (second[0] - first[0], second[1] - first[1])
    if direction == (0, 0):
        raise ValueError("A source segment must have distinct endpoints.")
    offset = direction[0] * first[1] - direction[1] * first[0]
    key = _primitive_integer_row((direction[0], direction[1], offset))
    return (key[0], key[1], key[2])


def segment_parameter(point: Point, axis: int, /) -> Fraction:
    return point[axis]


def segment_overlap(
    first: tuple[Point, Point], second: tuple[Point, Point], /
) -> Fraction:
    """Return exact one-dimensional overlap length on a shared supporting line."""
    if line_key(first) != line_key(second):
        return Fraction(0)
    delta = (first[1][0] - first[0][0], first[1][1] - first[0][1])
    axis = 0 if delta[0] else 1
    first_interval = sorted((first[0][axis], first[1][axis]))
    second_interval = sorted((second[0][axis], second[1][axis]))
    return max(
        Fraction(0),
        min(first_interval[1], second_interval[1])
        - max(first_interval[0], second_interval[0]),
    )


__all__ = [
    "Point",
    "Polygon",
    "area2",
    "cross",
    "intersection",
    "line_key",
    "oriented_polygon",
    "plane_key",
    "plane_orientation",
    "project_polygon",
    "projection_axes",
    "segment_overlap",
    "signed_area2",
]
