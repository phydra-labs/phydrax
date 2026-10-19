#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Arrange straight edges over rational binary input coordinates on the host.

Incidence, ordering, intersection and membership are decided over the binary input coordinates. Floating-point coordinates are emitted only at the
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
from ...typing import checked
from .._planar_embedding import PlanarEmbedding
from .._rational import rational_cross_2d, rational_vector
from ._model import BRepModel
from ._patches import LineCurve, PlanePatch


type Point = tuple[Fraction, Fraction]
type Segment = tuple[Point, Point]
type Loops = tuple[tuple[Point, ...], ...]


def point(value: np.ndarray, /) -> Point:
    return rational_vector((float(value[0]), float(value[1])))


def sub(first: Point, second: Point, /) -> Point:
    return first[0] - second[0], first[1] - second[1]


def _along(first: Point, second: Point, parameter: Fraction, /) -> Point:
    return (
        first[0] + parameter * (second[0] - first[0]),
        first[1] + parameter * (second[1] - first[1]),
    )


def on_segment(point: Point, segment: Segment, /) -> bool:
    first, second = segment
    return (
        rational_cross_2d(sub(point, first), sub(second, first)) == 0
        and min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
        and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
    )


def intersections(first: Segment, second: Segment, /) -> tuple[Point, ...]:
    start, end = first
    other, finish = second
    direction = sub(end, start)
    transverse = sub(finish, other)
    determinant = rational_cross_2d(direction, transverse)
    displacement = sub(other, start)
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
                p
                for p in (*first, *second)
                if on_segment(p, first) and on_segment(p, second)
            }
        )
    )


def inside_loop(point: Point, loop: tuple[Point, ...], /) -> bool:
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


def _inside(point: Point, loops: Loops, /) -> bool:
    return inside_loop(point, loops[0]) and not any(
        inside_loop(point, hole) for hole in loops[1:]
    )


def area(loop: tuple[Point, ...], /) -> Fraction:
    return (
        sum(
            (
                rational_cross_2d(a, b)
                for a, b in zip(loop, (*loop[1:], loop[0]), strict=True)
            ),
            Fraction(),
        )
        / 2
    )


def rational_loops(loops: tuple[np.ndarray, ...], /) -> Loops:
    if not loops:
        raise ValueError(
            "planar_arrangement.loops must contain an outer boundary; got empty loops"
        )
    result = []
    for index, points in enumerate(loops):
        points = np.asarray(points, dtype=np.float64)
        simplicity = polygon_simplicity_2d(points, mode=PredicateMode.EXACT)
        if int(np.asarray(simplicity.status)) != PolygonSimplicityStatus.SIMPLE:
            raise ValueError(
                "planar_arrangement.loops must be simple nondegenerate polygons; got a nonsimple polygon"
            )
        loop = tuple(point(point_bank) for point_bank in points)
        area_bank = area(loop)
        if not area_bank:
            raise ValueError(
                "planar_arrangement.loops must enclose nonzero area; got zero area"
            )
        result.append(loop if (area_bank > 0) == (index == 0) else tuple(reversed(loop)))
    rational = tuple(result)
    segments = tuple(
        tuple(zip(loop, (*loop[1:], loop[0]), strict=True)) for loop in rational
    )
    for index, first in enumerate(segments):
        for second in segments[index + 1 :]:
            if any(intersections(a, b) for a in first for b in second):
                raise ValueError(
                    "planar_arrangement.loops must have disjoint boundaries; got touching or intersecting loops"
                )
    for index, hole in enumerate(rational[1:]):
        if not inside_loop(hole[0], rational[0]):
            raise ValueError(
                "planar_arrangement.holes must lie strictly inside the outer boundary; got an outside hole"
            )
        if any(
            inside_loop(hole[0], other) or inside_loop(other[0], hole)
            for other in rational[index + 2 :]
        ):
            raise ValueError(
                "planar_arrangement.holes must be mutually disjoint; got overlapping or nested holes"
            )
    return rational


@checked
def require_coplanar_face(
    model: BRepModel, face: int, embedding: PlanarEmbedding, /
) -> None:
    patch = model.patches[face]
    if not isinstance(patch, PlanePatch):
        raise ValueError(
            "planar_arrangement.model.patches must be PlanePatch values; got a nonplanar carrier"
        )
    scale = max(1.0, float(np.max(np.abs(np.asarray(patch.origin)))))
    tolerance = 512.0 * np.finfo(np.float64).eps * scale
    normal = np.asarray(embedding.normal, dtype=np.float64)
    oriented = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
    norm = float(np.linalg.norm(oriented))
    if norm == 0.0:
        raise ValueError(
            "planar_arrangement.patch.axes must span a plane; got a singular chart"
        )
    oriented *= float(np.asarray(model.orientation)[face]) / norm
    if np.linalg.norm(oriented - normal) > 512.0 * np.finfo(np.float64).eps:
        raise ValueError(
            "planar_arrangement.patch.orientation must agree with the embedding normal; got opposite orientation"
        )
    if abs(float(embedding.plane_residual(np.asarray(patch.origin)))) > tolerance or any(
        abs(float(normal @ np.asarray(axis))) > tolerance
        for axis in (patch.first_axis, patch.second_axis)
    ):
        raise ValueError(
            "planar_arrangement.patch must lie in the embedding plane; got noncoplanar chart axes or origin"
        )


@checked
def curve_segment(
    model: BRepModel, edge: int, embedding: PlanarEmbedding, /
) -> np.ndarray:
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "planar_arrangement.model.geometry must contain edge curves and oriented loops; got None"
        )
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0 or not isinstance(geometry.curves[curve_index], LineCurve):
        raise ValueError(
            "planar_arrangement.geometry.curves must use straight LineCurve trims; got a curved or implicit trim"
        )
    world = np.asarray(geometry.vertex_points, dtype=np.float64)[
        list(geometry.edge_vertices[edge])
    ]
    tolerance = 512.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(world))))
    if np.max(np.abs(embedding.plane_residual(world)), initial=0.0) > tolerance:
        raise ValueError(
            "planar_arrangement.edge must lie in the embedding plane; got a plane residual above tolerance"
        )
    planar = embedding.to_planar(world)
    if np.linalg.norm(planar[1] - planar[0]) <= tolerance:
        raise ValueError(
            "planar_arrangement.edge must have length above tolerance; got zero length"
        )
    return planar


@checked
def face_loops(
    model: BRepModel, face: int, embedding: PlanarEmbedding, /
) -> tuple[np.ndarray, ...]:
    require_coplanar_face(model, face, embedding)
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "planar_arrangement.model.geometry must contain oriented face loops; got None"
        )
    loops = []
    for loop in geometry.face_loops[face]:
        points = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            segment = curve_segment(model, edge, embedding)
            points.append(segment[0 if geometry.coedge_senses[coedge] > 0 else 1])
        loops.append(np.asarray(points, dtype=np.float64))
    rational = rational_loops(tuple(loops))
    return tuple(np.asarray(loop, dtype=np.float64) for loop in rational)


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
    direction = sub(end, start)
    normal = (-direction[1], direction[0])
    nearest = Fraction(1)
    for first, second in all_segments:
        other_direction = sub(second, first)
        determinant = rational_cross_2d(normal, other_direction)
        if not determinant:
            continue
        displacement = sub(first, middle)
        distance = rational_cross_2d(displacement, other_direction) / determinant
        along = rational_cross_2d(displacement, normal) / determinant
        if distance > 0 and 0 <= along <= 1:
            nearest = min(nearest, distance)
    return middle[0] + nearest * normal[0] / 2, middle[1] + nearest * normal[1] / 2


@dataclass(frozen=True, slots=True)
class Cell:
    loops: Loops
    members: tuple[int, ...]

    @property
    def area(self) -> Fraction:
        return sum((area(loop) for loop in self.loops), Fraction())


def arrange(regions: tuple[tuple[np.ndarray, ...], ...], /) -> tuple[Cell, ...]:
    loops = tuple(rational_loops(region) for region in regions)
    segments = tuple(
        (a, b)
        for region in loops
        for loop in region
        for a, b in zip(loop, (*loop[1:], loop[0]), strict=True)
    )
    splits = [set(segment) for segment in segments]
    for index, first in enumerate(segments):
        for other_index in range(index + 1, len(segments)):
            intersections_bank = intersections(first, segments[other_index])
            splits[index].update(intersections_bank)
            splits[other_index].update(intersections_bank)
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
            return _angle_compare(sub(first, point), sub(second, point))

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
                    "planar_arrangement.halfedges must form disjoint closed cycles; got a reused halfedge"
                )
            unused.remove(edge)
            start, end = edge
            cycle.append(start)
            neighbors = adjacency[end]
            edge = (end, neighbors[(neighbors.index(start) - 1) % len(neighbors)])
            if edge == first:
                break
        loop = tuple(cycle)
        if not area(loop):
            raise ValueError(
                "planar_arrangement.cycles must enclose nonzero area; got a zero-area or dangling boundary"
            )
        probe = _left_probe(first, tuple(sorted(atoms)))
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
                "planar_arrangement.holes must have a containing outer cycle with matching members; got no candidate"
            )
        owner = min(candidates, key=lambda index: area(outers[index][0]))
        holes[owner].append(loop)
    return tuple(
        Cell((outer, *sorted(holes[index])), members)
        for index, (outer, members) in enumerate(outers)
        if members
    )


def common_area(
    first: tuple[np.ndarray, ...], second: tuple[np.ndarray, ...], /
) -> float:
    return float(
        sum(
            (cell.area for cell in arrange((first, second)) if cell.members == (0, 1)),
            Fraction(),
        )
    )
