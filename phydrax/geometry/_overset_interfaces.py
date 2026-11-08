#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact atomic edges of an owned planar overset partition.

The canonical direction is lexicographic in the endpoints.  Normals are
unnormalized: multiplying ``normal`` by an owner's orientation gives its exact
outward normal.  Its squared magnitude is ``length_squared``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._overset_partition import OversetPartition
from ._planar_primitives import (
    area2,
    box_overlap_pairs,
    cross,
    intersection,
    Point,
    polygon_boxes,
    signed_area2,
)


@dataclass(frozen=True, slots=True)
class EdgeOwner:
    part_name: str
    part_index: int
    cell_index: int
    orientation: int
    piece_index: int


@dataclass(frozen=True, slots=True)
class AtomicEdge:
    start: Point
    end: Point
    owners: tuple[EdgeOwner, ...]
    kind: str
    normal: Point
    length_squared: Fraction


def _candidates(polygons: tuple[tuple[Point, ...], ...]) -> tuple[()] | np.ndarray:
    if not polygons:
        return ()
    boxes = polygon_boxes(polygons)
    if boxes is None:
        raise ValueError("cannot index partition geometry with finite bounding boxes")
    return box_overlap_pairs(len(polygons), len(polygons), boxes, boxes, self_pairs=True)


def _vector_cross(a: Point, b: Point) -> Fraction:
    return a[0] * b[1] - a[1] * b[0]


def _subtract(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _on_segment(point: Point, a: Point, b: Point) -> bool:
    return (
        cross(a, b, point) == 0
        and min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= point[1] <= max(a[1], b[1])
    )


def partition_overset_interfaces(partition: OversetPartition) -> tuple[AtomicEdge, ...]:
    """Split all polygon boundaries at exact crossings and collinear endpoints.

    The input must be a non-overlapping collection of convex CCW polygons.
    Positive-area overlaps, invalid polygons, and inconsistent adjacency fail
    closed. Spatial bounding-box indexing precedes exact pair predicates.
    """
    pieces = partition.pieces
    polygons = tuple(piece.vertices for piece in pieces)
    for piece in pieces:
        vertices = piece.vertices
        if (
            len(vertices) < 3
            or len(set(vertices)) != len(vertices)
            or any(not isinstance(c, Fraction) for p in vertices for c in p)
            or signed_area2(vertices) <= 0
            or signed_area2(vertices) != 2 * piece.area
            or any(cross(a, b, p) < 0 for a, b in piece.edges for p in vertices)
        ):
            raise ValueError("partition contains an invalid convex CCW polygon")
    names_by_index: dict[int, str] = {}
    indices_by_name: dict[str, int] = {}
    for piece in pieces:
        if (
            piece.part_index in names_by_index
            and names_by_index[piece.part_index] != piece.part_name
            or piece.part_name in indices_by_name
            and indices_by_name[piece.part_name] != piece.part_index
        ):
            raise ValueError("inconsistent partition part identity")
        names_by_index[piece.part_index] = piece.part_name
        indices_by_name[piece.part_name] = piece.part_index
    for i, j in _candidates(polygons):
        if area2(intersection(polygons[int(i)], polygons[int(j)])):
            raise ValueError("partition polygons overlap in their interiors")

    segments = tuple(
        (a, b, index) for index, piece in enumerate(pieces) for a, b in piece.edges
    )
    cuts = [{a, b} for a, b, _ in segments]
    segment_boxes = tuple((a, b) for a, b, _ in segments)
    for i, j in _candidates(segment_boxes):
        i, j = int(i), int(j)
        a, b, _ = segments[i]
        c, d, _ = segments[j]
        r, s = _subtract(b, a), _subtract(d, c)
        denominator = _vector_cross(r, s)
        if denominator:
            t = _vector_cross(_subtract(c, a), s) / denominator
            u = _vector_cross(_subtract(c, a), r) / denominator
            if 0 <= t <= 1 and 0 <= u <= 1:
                point = (a[0] + t * r[0], a[1] + t * r[1])
                cuts[i].add(point)
                cuts[j].add(point)
        elif cross(a, b, c) == 0:
            for point in (c, d):
                if _on_segment(point, a, b):
                    cuts[i].add(point)
            for point in (a, b):
                if _on_segment(point, c, d):
                    cuts[j].add(point)

    adjacency: dict[tuple[Point, Point], list[EdgeOwner]] = {}
    for (a, b, piece_index), points in zip(segments, cuts, strict=True):
        piece = pieces[piece_index]
        ordered = sorted(points)
        orientation = 1 if a < b else -1
        for start, end in zip(ordered, ordered[1:]):
            owner = EdgeOwner(
                piece.part_name,
                piece.part_index,
                piece.cell_index,
                orientation,
                piece_index,
            )
            adjacency.setdefault((start, end), []).append(owner)

    result = []
    boundary_degree: dict[Point, int] = {}
    for (start, end), owner_list in sorted(adjacency.items()):
        owners = tuple(sorted(owner_list, key=lambda owner: owner.piece_index))
        if (
            len(owners) > 2
            or len({o.piece_index for o in owners}) != len(owners)
            or (len(owners) == 2 and owners[0].orientation == owners[1].orientation)
        ):
            raise ValueError("nonmanifold or inconsistently oriented partition edge")
        if len(owners) == 1:
            kind = "physical_boundary"
            for point in (start, end):
                boundary_degree[point] = boundary_degree.get(point, 0) + 1
        elif owners[0].part_index == owners[1].part_index:
            if owners[0].part_name != owners[1].part_name:
                raise ValueError("inconsistent partition part identity")
            kind = "same_part"
        else:
            if owners[0].part_name == owners[1].part_name:
                raise ValueError("inconsistent partition part identity")
            kind = "inter_part"
        dx, dy = end[0] - start[0], end[1] - start[1]
        result.append(AtomicEdge(start, end, owners, kind, (dy, -dx), dx * dx + dy * dy))
    if any(degree != 2 for degree in boundary_degree.values()):
        raise ValueError("nonmanifold physical boundary vertex")
    return tuple(result)
