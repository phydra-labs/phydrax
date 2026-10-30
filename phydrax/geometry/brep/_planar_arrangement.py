#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact rational predicates for polygonal CAD source loops."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction

from dataclasses import dataclass
from functools import cmp_to_key
from itertools import pairwise

import numpy as np
from .._rational import rational_cross_2d


type Point = tuple[Fraction, Fraction]
type Segment = tuple[Point, Point]
type Loops = tuple[tuple[Point, ...], ...]


def _sub(first: Point, second: Point, /) -> Point:
    return first[0] - second[0], first[1] - second[1]


def _along(first: Point, second: Point, parameter: Fraction, /) -> Point:
    return (
        first[0] + parameter * (second[0] - first[0]),
        first[1] + parameter * (second[1] - first[1]),
    )


def on_segment(point: Point, segment: Segment, /) -> bool:
    first, second = segment
    return (
        rational_cross_2d(_sub(point, first), _sub(second, first)) == 0
        and min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
        and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
    )


def intersections(first: Segment, second: Segment, /) -> tuple[Point, ...]:
    start, end = first
    other, finish = second
    direction = _sub(end, start)
    transverse = _sub(finish, other)
    determinant = rational_cross_2d(direction, transverse)
    displacement = _sub(other, start)
    if determinant:
        along = rational_cross_2d(displacement, transverse) / determinant
        across = rational_cross_2d(displacement, direction) / determinant
        return (
            (_along(start, end, along),) if 0 <= along <= 1 and 0 <= across <= 1 else ()
        )
    if rational_cross_2d(displacement, direction):
        return ()
    return tuple(
        sorted(
            {
                point
                for point in (*first, *second)
                if on_segment(point, first) and on_segment(point, second)
            }
        )
    )


def inside_loop(point: Point, loop: tuple[Point, ...], /) -> bool:
    """Return strict polygon interior; boundary membership is tested separately."""
    inside = False
    for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
        if on_segment(point, (first, second)):
            return False
        if (first[1] > point[1]) != (second[1] > point[1]):
            crossing = first[0] + (point[1] - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if crossing > point[0]:
                inside = not inside
    return inside


def area(loop: tuple[Point, ...], /) -> Fraction:
    return (
        sum(
            (
                rational_cross_2d(first, second)
                for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
            ),
            Fraction(0),
        )
        / 2
    )


def _input_loop(value: object, /) -> tuple[Point, ...]:
    if isinstance(value, np.ndarray):
        points = np.asarray(value, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("Planar source loops must be finite arrays of shape (n, 2).")
        return tuple(
            (Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1])))
            for point in points
        )
    try:
        if not isinstance(value, Iterable):
            raise TypeError
        point_values: list[tuple[object, ...]] = []
        for point in value:
            if not isinstance(point, Iterable):
                raise TypeError
            point_values.append(tuple(point))
        points = tuple(point_values)
    except TypeError as error:
        raise TypeError(
            "Planar source loops must be coordinate arrays or exact points."
        ) from error
    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError("Planar source loops require at least three 2D points.")
    exact_points: list[Point] = []
    for point in points:
        first, second = point
        if not isinstance(first, Fraction) or not isinstance(second, Fraction):
            raise TypeError("Exact planar source loops must use rational coordinates.")
        exact_points.append((first, second))
    return tuple(exact_points)


def rational_loops(loops: tuple[object, ...], /) -> Loops:
    """Normalize simple polygon loops to exact, positively oriented rational banks."""
    if not loops:
        raise ValueError("Planar regions require an outer polygon loop.")
    result = []
    for loop_index, source_loop in enumerate(loops):
        loop = _input_loop(source_loop)
        if len(set(loop)) != len(loop):
            raise ValueError("Planar source loops cannot repeat a vertex.")
        signed = area(loop)
        if signed == 0:
            raise ValueError("Planar source loops must enclose nonzero area.")
        loop = loop if (signed > 0) == (loop_index == 0) else tuple(reversed(loop))
        segments = tuple(zip(loop, (*loop[1:], loop[0]), strict=True))
        for first_index, first in enumerate(segments):
            for second_index in range(first_index + 1, len(segments)):
                second = segments[second_index]
                contacts = intersections(first, second)
                if not contacts:
                    continue
                adjacent = second_index == first_index + 1 or (
                    first_index == 0 and second_index == len(segments) - 1
                )
                expected = (
                    loop[second_index] if second_index == first_index + 1 else loop[0]
                )
                if not adjacent or any(point != expected for point in contacts):
                    raise ValueError(
                        "Planar source loops must be simple and non-self-intersecting."
                    )
        result.append(loop)
    normalized = tuple(result)
    segments = tuple(
        tuple(zip(loop, (*loop[1:], loop[0]), strict=True)) for loop in normalized
    )
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            if any(intersections(a, b) for a in first for b in second):
                raise ValueError("Planar region loops must have disjoint boundaries.")
    for hole_index, hole in enumerate(normalized[1:]):
        if not inside_loop(hole[0], normalized[0]):
            raise ValueError("Every planar hole must lie strictly inside its outer loop.")
        if any(
            inside_loop(hole[0], other) or inside_loop(other[0], hole)
            for other in normalized[hole_index + 2 :]
        ):
            raise ValueError("Planar holes must be mutually disjoint and non-nested.")
    return normalized


def _inside(point: Point, loops: Loops, /) -> bool:
    return inside_loop(point, loops[0]) and not any(
        inside_loop(point, hole) for hole in loops[1:]
    )


def _angle_compare(first: Point, second: Point, /) -> int:
    first_half = first[1] < 0 or (first[1] == 0 and first[0] < 0)
    second_half = second[1] < 0 or (second[1] == 0 and second[0] < 0)
    if first_half != second_half:
        return 1 if first_half else -1
    determinant = rational_cross_2d(first, second)
    return -1 if determinant > 0 else 1 if determinant < 0 else 0


def _left_probe(segment: Segment, all_segments: tuple[Segment, ...], /) -> Point:
    start, end = segment
    middle = _along(start, end, Fraction(1, 2))
    direction = _sub(end, start)
    normal = (-direction[1], direction[0])
    nearest = Fraction(1)
    for first, second in all_segments:
        other_direction = _sub(second, first)
        determinant = rational_cross_2d(normal, other_direction)
        if not determinant:
            continue
        displacement = _sub(first, middle)
        distance = rational_cross_2d(displacement, other_direction) / determinant
        along = rational_cross_2d(displacement, normal) / determinant
        if distance > 0 and 0 <= along <= 1:
            nearest = min(nearest, distance)
    return middle[0] + nearest * normal[0] / 2, middle[1] + nearest * normal[1] / 2


@dataclass(frozen=True, slots=True)
class Cell:
    """One exact arrangement cell with an oriented outer loop and optional holes."""

    loops: Loops
    members: tuple[int, ...]

    @property
    def area(self) -> Fraction:
        return sum((area(loop) for loop in self.loops), Fraction(0))


def arrange(regions: tuple[tuple[object, ...], ...], /) -> tuple[Cell, ...]:
    """Build exhaustive n-ary arrangement cells for rational straight-loop regions."""
    loops = tuple(rational_loops(region) for region in regions)
    if not loops:
        raise ValueError("A planar arrangement requires at least one source region.")
    segments = tuple(
        (first, second)
        for region in loops
        for loop in region
        for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
    )
    splits = [set(segment) for segment in segments]
    for index, first in enumerate(segments):
        for other_index in range(index + 1, len(segments)):
            common = intersections(first, segments[other_index])
            splits[index].update(common)
            splits[other_index].update(common)
    atoms = set()
    for points in splits:
        for first, second in pairwise(sorted(points)):
            if first != second:
                atoms.add((first, second))
    adjacency: dict[Point, list[Point]] = {}
    for first, second in sorted(atoms):
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    for point, neighbors in adjacency.items():

        def compare(first: Point, second: Point, /) -> int:
            return _angle_compare(_sub(first, point), _sub(second, point))

        neighbors.sort(key=cmp_to_key(compare))
    unused = {
        (first, second) for first, neighbors in adjacency.items() for second in neighbors
    }
    cycles: list[tuple[tuple[Point, ...], tuple[int, ...]]] = []
    all_segments = tuple(sorted(atoms))
    while unused:
        first_edge = min(unused)
        edge = first_edge
        cycle: list[Point] = []
        while True:
            if edge not in unused:
                raise ValueError(
                    "Planar arrangement halfedges must form disjoint closed cycles."
                )
            unused.remove(edge)
            start, end = edge
            cycle.append(start)
            neighbors = adjacency[end]
            edge = (end, neighbors[(neighbors.index(start) - 1) % len(neighbors)])
            if edge == first_edge:
                break
        loop = tuple(cycle)
        signed_area = area(loop)
        if not signed_area:
            raise ValueError("Planar arrangement contains a zero-area or dangling cycle.")
        probe = _left_probe(first_edge, all_segments)
        members = tuple(
            index for index, region in enumerate(loops) if _inside(probe, region)
        )
        cycles.append((loop, members))
    outers = [(loop, members) for loop, members in cycles if area(loop) > 0]
    holes: list[list[tuple[Point, ...]]] = [[] for _ in outers]
    for loop, members in cycles:
        if area(loop) >= 0 or not members:
            continue
        candidates = [
            index
            for index, (outer, owner) in enumerate(outers)
            if owner == members and inside_loop(loop[0], outer)
        ]
        if not candidates:
            raise ValueError(
                "Planar arrangement hole has no containing outer cell with matching membership."
            )
        owner = min(candidates, key=lambda index: area(outers[index][0]))
        holes[owner].append(loop)
    return tuple(
        Cell((outer, *tuple(sorted(holes[index]))), members)
        for index, (outer, members) in enumerate(outers)
        if members
    )


__all__ = [
    "Cell",
    "Loops",
    "Point",
    "Segment",
    "area",
    "arrange",
    "inside_loop",
    "intersections",
    "on_segment",
    "rational_loops",
]
