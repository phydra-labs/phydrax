#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact priority ownership of the original cells of planar triangle meshes.

All predicates, constructed vertices, and areas are rational: a binary64 input
coordinate denotes its exact binary value.  Each subtraction splits a convex
piece by the supporting half-planes of a convex obstacle.  The discarded and
retained pieces therefore partition the original cell up to their boundaries;
no sampling or geometric tolerance is involved.  Returned pieces retain the
identity of the input triangle, not a replacement finite-element mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._planar_primitives import (
    Point,
    Polygon,
    area2,
    box_overlap_pairs,
    cross,
    interiors_separated,
    intersection,
    oriented_polygon,
    polygon_boxes,
    signed_area2,
)


@dataclass(frozen=True, slots=True)
class OwnedPolygon:
    """A positive-area convex portion of an original source triangle.

    Vertices are counterclockwise and exact, and edges are directed accordingly.
    Edges can be internal subdivision edges or partial interface segments; their
    exact endpoints permit cancellation and collinear splitting by a consumer.
    """

    part_name: str
    part_index: int
    cell_index: int
    vertices: Polygon
    area: Fraction

    @property
    def edges(self) -> tuple[tuple[Point, Point], ...]:
        return tuple(zip(self.vertices, self.vertices[1:] + self.vertices[:1], strict=True))


@dataclass(frozen=True, slots=True)
class CellArea:
    """Exact source and owned area evidence for one original triangle."""

    part_name: str
    part_index: int
    cell_index: int
    source_area: Fraction
    cavity_area: Fraction
    owned_area: Fraction

    @property
    def excluded_area(self) -> Fraction:
        """Area removed by the union of cavities and higher-priority parts."""
        return self.source_area - self.owned_area


@dataclass(frozen=True, slots=True)
class OversetPartition:
    """Exact owned polygons and per-original-cell area accounting."""

    pieces: tuple[OwnedPolygon, ...]
    total_area: Fraction
    cells: tuple[CellArea, ...]


def _array(value: object, dtype: np.dtype, columns: int, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise ValueError(f"{label} must be a {dtype} numpy array")
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{label} must have shape (N, {columns})")
    return value


def _polygon(points: np.ndarray) -> Polygon:
    return tuple((Fraction.from_float(float(x)), Fraction.from_float(float(y))) for x, y in points)


def _cavity(points: np.ndarray) -> Polygon:
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("cavity must have at least three finite vertices")
    raw = _polygon(points)
    # Accept an explicitly closed polygon, but reject repeated vertices elsewhere.
    if raw[0] == raw[-1]:
        raw = raw[:-1]
    if len(set(raw)) != len(raw):
        raise ValueError("cavity vertices must be distinct")
    orientation = signed_area2(raw)
    if not orientation:
        raise ValueError("cavity must have positive area")
    # Test the original ordering before removing collinear vertices, so that
    # backtracking edges and self-intersecting orderings cannot be hidden.
    if any(
        cross(start, end, point) * orientation < 0
        for start, end in zip(raw, raw[1:] + raw[:1], strict=True)
        for point in raw
    ):
        raise ValueError("cavity must be a convex polygon in cyclic order")
    polygon = oriented_polygon(raw)
    if len(polygon) < 3 or signed_area2(polygon) <= 0:
        raise ValueError("cavity must have positive area")
    return polygon


def _split(polygon: Polygon, start: Point, end: Point) -> tuple[Polygon, Polygon]:
    """Split a convex polygon into the closed left and right half-planes."""
    left: list[Point] = []
    right: list[Point] = []
    previous = polygon[-1]
    previous_side = cross(start, end, previous)
    for current in polygon:
        current_side = cross(start, end, current)
        if (previous_side < 0 < current_side) or (current_side < 0 < previous_side):
            parameter = previous_side / (previous_side - current_side)
            point = (
                previous[0] + parameter * (current[0] - previous[0]),
                previous[1] + parameter * (current[1] - previous[1]),
            )
            left.append(point)
            right.append(point)
        if current_side >= 0:
            left.append(current)
        if current_side <= 0:
            right.append(current)
        previous, previous_side = current, current_side
    return oriented_polygon(tuple(left)), oriented_polygon(tuple(right))


def _subtract(polygon: Polygon, obstacle: Polygon) -> tuple[Polygon, ...]:
    """Disjoint convex decomposition of polygon minus a convex obstacle."""
    if not area2(intersection(polygon, obstacle)):
        return (polygon,)
    remainder = polygon
    result: list[Polygon] = []
    for start, end in zip(obstacle, obstacle[1:] + obstacle[:1], strict=True):
        inside, outside = _split(remainder, start, end)
        if area2(outside):
            result.append(outside)
        if not area2(inside):
            break
        remainder = inside
    return tuple(result)


def partition_overset_triangles(
    parts: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    cavities: tuple[np.ndarray, ...],
) -> OversetPartition:
    """Partition input triangles by part priority, excluding every cavity.

    Each part is ``(name, float64 coordinates[N, 2], int64 triangles[M, 3])``;
    each cavity is a cyclic, convex float64 polygon.  Triangles within each
    part must have disjoint interiors; triangles of different parts may overlap.
    Earlier parts own all overlap.  Finite binary64 coordinates are interpreted
    exactly, including when intersections are not representable in binary64.

    Every source triangle has a corresponding entry in ``cells``, including
    triangles completely excluded by cavities or higher priority parts.  The
    exact source and owned areas establish the area accounting per cell, and
    ``total_area`` is the exact sum of all owned polygon areas.  Closed polygon
    boundaries may coincide, but positive-area interiors do not overlap.
    """
    triangles: list[Polygon] = []
    identities: list[tuple[str, int, int]] = []
    for part_index, (name, points_value, cells_value) in enumerate(parts):
        if not isinstance(name, str):
            raise ValueError("part names must be strings")
        points = _array(points_value, np.dtype(np.float64), 2, "coordinates")
        cells = _array(cells_value, np.dtype(np.int64), 3, "triangles")
        if not np.all(np.isfinite(points)):
            raise ValueError("coordinates must be finite")
        if np.any(cells < 0) or np.any(cells >= len(points)):
            raise ValueError("triangle vertex index out of range")
        exact_points = _polygon(points)
        for cell_index, cell in enumerate(cells):
            triangle = oriented_polygon(tuple(exact_points[int(index)] for index in cell))
            if len(triangle) != 3 or not area2(triangle):
                raise ValueError("triangles must have positive area")
            triangles.append(triangle)
            identities.append((name, part_index, cell_index))

    walls = tuple(_cavity(_array(wall, np.dtype(np.float64), 2, "cavity")) for wall in cavities)
    triangle_boxes = polygon_boxes(triangles)
    candidates: list[list[Polygon]] = [[] for _ in triangles]
    for first, second in box_overlap_pairs(
        len(triangles), len(triangles), triangle_boxes, triangle_boxes, self_pairs=True
    ):
        first, second = int(first), int(second)
        first_triangle, second_triangle = triangles[first], triangles[second]
        if interiors_separated(first_triangle, second_triangle):
            continue
        if identities[first][1] == identities[second][1]:
            raise ValueError("triangles within a part must have disjoint interiors")
        candidates[second].append(first_triangle)

    wall_candidates: list[list[Polygon]] = [[] for _ in triangles]
    for cell, wall in box_overlap_pairs(
        len(triangles), len(walls), triangle_boxes, polygon_boxes(walls), self_pairs=False
    ):
        wall_candidates[int(cell)].append(walls[int(wall)])

    pieces: list[OwnedPolygon] = []
    cell_areas: list[CellArea] = []
    total_area = Fraction(0)
    for triangle, (name, part_index, cell_index), higher, protected in zip(
        triangles, identities, candidates, wall_candidates, strict=True
    ):
        owned = (triangle,)
        for obstacle in protected:
            owned = tuple(piece for polygon in owned for piece in _subtract(polygon, obstacle))
            if not owned:
                break
        source_area = area2(triangle) / 2
        cavity_area = source_area - sum((area2(polygon) / 2 for polygon in owned), Fraction(0))
        for obstacle in higher:
            owned = tuple(piece for polygon in owned for piece in _subtract(polygon, obstacle))
            if not owned:
                break
        owned_area = Fraction(0)
        for polygon in owned:
            area = area2(polygon) / 2
            pieces.append(OwnedPolygon(name, part_index, cell_index, polygon, area))
            owned_area += area
        cell_areas.append(CellArea(name, part_index, cell_index, source_area, cavity_area, owned_area))
        total_area += owned_area
    return OversetPartition(tuple(pieces), total_area, tuple(cell_areas))


__all__ = ["CellArea", "OwnedPolygon", "OversetPartition", "partition_overset_triangles"]
