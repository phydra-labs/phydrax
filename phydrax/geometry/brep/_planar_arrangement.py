#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host exact-rational straight-edge arrangements for native planar B-Reps.

Incidence, ordering, intersection and membership are decided over the exact
binary input coordinates. Floating-point coordinates are emitted only at the
native carrier construction boundary; no query tessellation decides topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from itertools import pairwise

import numpy as np

from ..._geometry_predicates import (
    polygon_simplicity_2d,
    PolygonSimplicityStatus,
    PredicateMode,
)
from .._planar_embedding import PlanarEmbedding
from ._model import BRepModel
from ._patches import LineCurve, PlanePatch


type Point = tuple[Fraction, Fraction]
type Segment = tuple[Point, Point]
type Loops = tuple[tuple[Point, ...], ...]


def _point(value: np.ndarray, /) -> Point:
    return Fraction(float(value[0])), Fraction(float(value[1]))


def _sub(first: Point, second: Point, /) -> Point:
    return first[0] - second[0], first[1] - second[1]


def _cross(first: Point, second: Point, /) -> Fraction:
    return first[0] * second[1] - first[1] * second[0]


def _along(first: Point, second: Point, parameter: Fraction, /) -> Point:
    return (
        first[0] + parameter * (second[0] - first[0]),
        first[1] + parameter * (second[1] - first[1]),
    )


def _on_segment(point: Point, segment: Segment, /) -> bool:
    first, second = segment
    return (
        _cross(_sub(point, first), _sub(second, first)) == 0
        and min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
        and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
    )


def _intersections(first: Segment, second: Segment, /) -> tuple[Point, ...]:
    start, end = first
    other, finish = second
    direction = _sub(end, start)
    transverse = _sub(finish, other)
    determinant = _cross(direction, transverse)
    displacement = _sub(other, start)
    if determinant:
        along = _cross(displacement, transverse) / determinant
        across = _cross(displacement, direction) / determinant
        return (
            (_along(start, end, along),) if 0 <= along <= 1 and 0 <= across <= 1 else ()
        )
    if _cross(displacement, direction):
        return ()
    return tuple(
        sorted(
            {
                p
                for p in (*first, *second)
                if _on_segment(p, first) and _on_segment(p, second)
            }
        )
    )


def _inside_loop(point: Point, loop: tuple[Point, ...], /) -> bool:
    inside = False
    for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
        if _on_segment(point, (first, second)):
            return False
        if (first[1] > point[1]) != (second[1] > point[1]):
            crossing = first[0] + (point[1] - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if crossing > point[0]:
                inside = not inside
    return inside


def _inside(point: Point, loops: Loops, /) -> bool:
    return _inside_loop(point, loops[0]) and not any(
        _inside_loop(point, hole) for hole in loops[1:]
    )


def _area(loop: tuple[Point, ...], /) -> Fraction:
    return (
        sum(
            (_cross(a, b) for a, b in zip(loop, (*loop[1:], loop[0]), strict=True)),
            Fraction(),
        )
        / 2
    )


def _rational_loops(loops: tuple[np.ndarray, ...], /) -> Loops:
    if not loops:
        raise ValueError("A planar region requires an outer boundary.")
    result = []
    for index, points in enumerate(loops):
        points = np.asarray(points, dtype=np.float64)
        simplicity = polygon_simplicity_2d(points, mode=PredicateMode.EXACT)
        if int(np.asarray(simplicity.status)) != PolygonSimplicityStatus.SIMPLE:
            raise ValueError(
                "Native planar boundaries must be simple nondegenerate polygons."
            )
        loop = tuple(_point(point) for point in points)
        area = _area(loop)
        if not area:
            raise ValueError("Native planar boundaries must enclose nonzero area.")
        result.append(loop if (area > 0) == (index == 0) else tuple(reversed(loop)))
    rational = tuple(result)
    segments = tuple(
        tuple(zip(loop, (*loop[1:], loop[0]), strict=True)) for loop in rational
    )
    for index, first in enumerate(segments):
        for second in segments[index + 1 :]:
            if any(_intersections(a, b) for a in first for b in second):
                raise ValueError(
                    "Planar outer and hole boundaries cannot touch or intersect."
                )
    for index, hole in enumerate(rational[1:]):
        if not _inside_loop(hole[0], rational[0]):
            raise ValueError(
                "Every planar hole must lie strictly inside its outer boundary."
            )
        if any(
            _inside_loop(hole[0], other) or _inside_loop(other[0], hole)
            for other in rational[index + 2 :]
        ):
            raise ValueError("Planar holes cannot overlap or contain one another.")
    return rational


def _require_coplanar_face(
    model: BRepModel, face: int, embedding: PlanarEmbedding, /
) -> None:
    patch = model.patches[face]
    if not isinstance(patch, PlanePatch):
        raise ValueError("Native planar partition requires exact planar source faces.")
    scale = max(1.0, float(np.max(np.abs(np.asarray(patch.origin)))))
    tolerance = 512.0 * np.finfo(np.float64).eps * scale
    normal = np.asarray(embedding.normal, dtype=np.float64)
    oriented = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
    norm = float(np.linalg.norm(oriented))
    if norm == 0.0:
        raise ValueError("A native planar source face has a singular chart.")
    oriented *= float(np.asarray(model.orientation)[face]) / norm
    if np.linalg.norm(oriented - normal) > 512.0 * np.finfo(np.float64).eps:
        raise ValueError(
            "Every planar source face must have the embedding's positive orientation."
        )
    if abs(float(embedding.plane_residual(np.asarray(patch.origin)))) > tolerance or any(
        abs(float(normal @ np.asarray(axis))) > tolerance
        for axis in (patch.first_axis, patch.second_axis)
    ):
        raise ValueError("A planar source face is not coplanar with its embedding.")


def _curve_segment(
    model: BRepModel, edge: int, embedding: PlanarEmbedding, /
) -> np.ndarray:
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "Native planar queries require exact edge curves and oriented loops."
        )
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0 or not isinstance(geometry.curves[curve_index], LineCurve):
        raise ValueError(
            "Native straight-edge planar arrangement does not support curved trims; exact curve intersection and subdivision are required."
        )
    world = np.asarray(geometry.vertex_points, dtype=np.float64)[
        list(geometry.edge_vertices[edge])
    ]
    tolerance = 512.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(world))))
    if np.max(np.abs(embedding.plane_residual(world)), initial=0.0) > tolerance:
        raise ValueError("A planar patch edge lies outside the declared embedding.")
    planar = embedding.to_planar(world)
    if np.linalg.norm(planar[1] - planar[0]) <= tolerance:
        raise ValueError("A planar patch edge has zero length.")
    return planar


def _face_loops(
    model: BRepModel, face: int, embedding: PlanarEmbedding, /
) -> tuple[np.ndarray, ...]:
    _require_coplanar_face(model, face, embedding)
    geometry = model.geometry
    if geometry is None:
        raise ValueError("Native planar partition requires exact oriented face loops.")
    loops = []
    for loop in geometry.face_loops[face]:
        points = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            segment = _curve_segment(model, edge, embedding)
            points.append(segment[0 if geometry.coedge_senses[coedge] > 0 else 1])
        loops.append(np.asarray(points, dtype=np.float64))
    rational = _rational_loops(tuple(loops))
    return tuple(np.asarray(loop, dtype=np.float64) for loop in rational)


def _angle_compare(first: Point, second: Point, /) -> int:
    first_half = first[1] < 0 or (first[1] == 0 and first[0] < 0)
    second_half = second[1] < 0 or (second[1] == 0 and second[0] < 0)
    if first_half != second_half:
        return 1 if first_half else -1
    determinant = _cross(first, second)
    return -1 if determinant > 0 else 1 if determinant < 0 else 0


def _left_probe(segment: Segment, all_segments: tuple[Segment, ...], /) -> Point:
    start, end = segment
    middle = _along(start, end, Fraction(1, 2))
    direction = _sub(end, start)
    normal = (-direction[1], direction[0])
    nearest = Fraction(1)
    for first, second in all_segments:
        other_direction = _sub(second, first)
        determinant = _cross(normal, other_direction)
        if not determinant:
            continue
        displacement = _sub(first, middle)
        distance = _cross(displacement, other_direction) / determinant
        along = _cross(displacement, normal) / determinant
        if distance > 0 and 0 <= along <= 1:
            nearest = min(nearest, distance)
    return middle[0] + nearest * normal[0] / 2, middle[1] + nearest * normal[1] / 2


@dataclass(frozen=True, slots=True)
class _Cell:
    loops: Loops
    members: tuple[int, ...]

    @property
    def area(self) -> Fraction:
        return sum((_area(loop) for loop in self.loops), Fraction())


def _arrange(regions: tuple[tuple[np.ndarray, ...], ...], /) -> tuple[_Cell, ...]:
    loops = tuple(_rational_loops(region) for region in regions)
    segments = tuple(
        (a, b)
        for region in loops
        for loop in region
        for a, b in zip(loop, (*loop[1:], loop[0]), strict=True)
    )
    splits = [set(segment) for segment in segments]
    for index, first in enumerate(segments):
        for other_index in range(index + 1, len(segments)):
            intersections = _intersections(first, segments[other_index])
            splits[index].update(intersections)
            splits[other_index].update(intersections)
    atoms = set()
    for points in splits:
        for first, second in pairwise(sorted(points)):
            atoms.add((first, second))
    adjacency: dict[Point, list[Point]] = {}
    for first, second in sorted(atoms):
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    for point, neighbors in adjacency.items():

        def compare(first: Point, second: Point, /) -> int:
            return _angle_compare(_sub(first, point), _sub(second, point))

        neighbors.sort(key=cmp_to_key(compare))
    unused = {(a, b) for a, neighbors in adjacency.items() for b in neighbors}
    cycles = []
    while unused:
        first = min(unused)
        edge = first
        cycle = []
        while True:
            if edge not in unused:
                raise ValueError(
                    "Native planar arrangement has an invalid half-edge cycle."
                )
            unused.remove(edge)
            start, end = edge
            cycle.append(start)
            neighbors = adjacency[end]
            edge = (end, neighbors[(neighbors.index(start) - 1) % len(neighbors)])
            if edge == first:
                break
        loop = tuple(cycle)
        if not _area(loop):
            raise ValueError(
                "Native planar arrangement contains a zero-area or dangling boundary."
            )
        probe = _left_probe(first, tuple(sorted(atoms)))
        members = tuple(
            index for index, region in enumerate(loops) if _inside(probe, region)
        )
        cycles.append((loop, members))
    outers = [(loop, members) for loop, members in cycles if _area(loop) > 0]
    holes: list[list[tuple[Point, ...]]] = [[] for _ in outers]
    for loop, members in cycles:
        if _area(loop) >= 0 or not members:
            continue
        candidates = [
            index
            for index, (outer, owner) in enumerate(outers)
            if owner == members and _inside_loop(loop[0], outer)
        ]
        if not candidates:
            raise ValueError(
                "Native planar arrangement could not attach a hole boundary."
            )
        owner = min(candidates, key=lambda index: _area(outers[index][0]))
        holes[owner].append(loop)
    return tuple(
        _Cell((outer, *sorted(holes[index])), members)
        for index, (outer, members) in enumerate(outers)
        if members
    )


def _common_area(
    first: tuple[np.ndarray, ...], second: tuple[np.ndarray, ...], /
) -> float:
    return float(
        sum(
            (cell.area for cell in _arrange((first, second)) if cell.members == (0, 1)),
            Fraction(),
        )
    )
