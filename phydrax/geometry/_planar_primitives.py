#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact rational primitives for planar source and candidate partitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from typing import TypeAlias

import numpy as np

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh


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


def interiors_separated(first: Polygon, second: Polygon, /) -> bool:
    """Whether an edge line of either triangle exactly separates their interiors.

    Two convex polygons have disjoint interiors exactly when a supporting line
    of one of their edges leaves the other polygon in its closed outer half
    plane, so this decides zero overlap area without clipping.
    """
    for inner, outer in ((first, second), (second, first)):
        orientation = signed_area2(inner)
        if orientation == 0:
            return True
        for index in range(3):
            start, end = inner[index], inner[(index + 1) % 3]
            if all(cross(start, end, point) * orientation <= 0 for point in outer):
                return True
    return False


type PolygonBoxes = tuple[np.ndarray, np.ndarray]
type _Interval = tuple[np.ndarray, np.ndarray]

# Banks with at most this many candidate pairs test every box pair at once on
# the host; larger banks use the packed BVH. Both report every pair of closed
# boxes that meet, so the exact findings do not depend on the choice.
_DENSE_BOX_PAIRS = 1 << 16


def _float_down(value: Fraction, /) -> float:
    result = float(value)
    return result if Fraction(result) <= value else float(np.nextafter(result, -np.inf))


def _float_up(value: Fraction, /) -> float:
    result = float(value)
    return result if Fraction(result) >= value else float(np.nextafter(result, np.inf))


def polygon_boxes(polygons: Sequence[Polygon], /) -> PolygonBoxes | None:
    """Outward-rounded binary64 boxes of exact planar polygons (None if unbounded)."""
    if not polygons:
        return None
    lower = np.empty((len(polygons), 2), dtype=np.float64)
    upper = np.empty_like(lower)
    try:
        for row, polygon in enumerate(polygons):
            for axis in range(2):
                lower[row, axis] = _float_down(min(point[axis] for point in polygon))
                upper[row, axis] = _float_up(max(point[axis] for point in polygon))
    except OverflowError:
        return None
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    return lower, upper


def box_overlap_pairs(
    first_count: int,
    second_count: int,
    first_boxes: PolygonBoxes | None,
    second_boxes: PolygonBoxes | None,
    /,
    *,
    self_pairs: bool,
) -> np.ndarray:
    """Pairs (first row, second row) whose closed boxes meet, ``(P, 2)`` in row order.

    Disjoint closed boxes have zero intersection area, so these pairs carry
    every positive-area overlap; self pairs keep first row < second row.
    Without boxes every pair is a candidate.
    """
    if self_pairs and first_count != second_count:
        raise ValueError("Self-overlap banks must be the same polygon bank.")
    if first_count == 0 or second_count == 0:
        return np.empty((0, 2), dtype=np.int64)
    if first_boxes is None or second_boxes is None:
        if self_pairs:
            rows, columns = np.triu_indices(first_count, k=1)
        else:
            rows, columns = np.indices((first_count, second_count)).reshape((2, -1))
        return np.column_stack((rows, columns)).astype(np.int64, copy=False)
    first_lower, first_upper = first_boxes
    second_lower, second_upper = second_boxes
    if first_count * second_count <= _DENSE_BOX_PAIRS:
        meets = np.all(
            (first_lower[:, None, :] <= second_upper[None, :, :])
            & (second_lower[None, :, :] <= first_upper[:, None, :]),
            axis=2,
        )
        if self_pairs:
            meets = np.triu(meets, k=1)
        return np.argwhere(meets).astype(np.int64, copy=False)
    first_bvh = prepare_bvh(first_lower, first_upper, dtype=np.float64)
    second_bvh = (
        first_bvh
        if self_pairs
        else prepare_bvh(second_lower, second_upper, dtype=np.float64)
    )
    blocks: list[np.ndarray] = []
    for first_ids, second_ids in bvh_overlap_pair_blocks(
        first_bvh, second_bvh, include_touching=True
    ):
        if self_pairs:
            keep = first_ids < second_ids
            first_ids, second_ids = first_ids[keep], second_ids[keep]
        if first_ids.size:
            blocks.append(
                np.column_stack((first_ids, second_ids)).astype(np.int64, copy=False)
            )
    if not blocks:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.concatenate(blocks, axis=0)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


@dataclass(frozen=True, slots=True)
class _TriangleBank:
    """Outward binary64 corner enclosures ``(rows, 3, 2)`` and exact corner identities."""

    lower: np.ndarray
    upper: np.ndarray
    corners: np.ndarray
    valid: np.ndarray


def _triangle_bank(
    polygons: Sequence[Polygon], identities: dict[Point, int], /
) -> _TriangleBank:
    count = len(polygons)
    lower = np.zeros((count, 3, 2), dtype=np.float64)
    upper = np.zeros((count, 3, 2), dtype=np.float64)
    corners = np.full((count, 3), -1, dtype=np.int64)
    valid = np.zeros(count, dtype=np.bool_)
    for row, polygon in enumerate(polygons):
        if len(polygon) != 3:
            continue
        try:
            for corner, point in enumerate(polygon):
                corners[row, corner] = identities.setdefault(point, len(identities))
                for axis in range(2):
                    lower[row, corner, axis] = _float_down(point[axis])
                    upper[row, corner, axis] = _float_up(point[axis])
        except OverflowError:
            continue
        valid[row] = True
    valid &= np.all(np.isfinite(lower) & np.isfinite(upper), axis=(1, 2))
    return _TriangleBank(lower, upper, corners, valid)


def _interval_subtract(first: _Interval, second: _Interval, /) -> _Interval:
    return (
        np.nextafter(first[0] - second[1], -np.inf),
        np.nextafter(first[1] - second[0], np.inf),
    )


def _interval_multiply(first: _Interval, second: _Interval, /) -> _Interval:
    products = np.stack(
        (
            first[0] * second[0],
            first[0] * second[1],
            first[1] * second[0],
            first[1] * second[1],
        )
    )
    # NaN products (0 * inf) propagate and leave the sign undecided.
    return (
        np.nextafter(np.min(products, axis=0), -np.inf),
        np.nextafter(np.max(products, axis=0), np.inf),
    )


def _interval_cross(
    first: tuple[_Interval, _Interval],
    second: tuple[_Interval, _Interval],
    third: tuple[_Interval, _Interval],
    /,
) -> _Interval:
    """Enclosure of `cross` from per-pair coordinate intervals ``(x, y)``."""
    edge_x = _interval_subtract(second[0], first[0])
    edge_y = _interval_subtract(second[1], first[1])
    point_x = _interval_subtract(third[0], first[0])
    point_y = _interval_subtract(third[1], first[1])
    return _interval_subtract(
        _interval_multiply(edge_x, point_y), _interval_multiply(edge_y, point_x)
    )


def _bank_corner(
    bank: _TriangleBank, rows: np.ndarray, corner: int, /
) -> tuple[_Interval, _Interval]:
    return (
        (bank.lower[rows, corner, 0], bank.upper[rows, corner, 0]),
        (bank.lower[rows, corner, 1], bank.upper[rows, corner, 1]),
    )


def _directed_separation(
    inner: _TriangleBank,
    inner_rows: np.ndarray,
    outer: _TriangleBank,
    outer_rows: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs an edge line of ``inner`` provably separates / provably none does."""
    inner_corners = [_bank_corner(inner, inner_rows, corner) for corner in range(3)]
    outer_corners = [_bank_corner(outer, outer_rows, corner) for corner in range(3)]
    orientation = _interval_cross(*inner_corners)
    positive = orientation[0] > 0.0
    signed = positive | (orientation[1] < 0.0)
    separates = np.zeros(inner_rows.size, dtype=np.bool_)
    every_edge_meets = np.ones(inner_rows.size, dtype=np.bool_)
    inner_ids = inner.corners[inner_rows]
    outer_ids = outer.corners[outer_rows]
    for index in range(3):
        following = (index + 1) % 3
        closed_outside = np.ones(inner_rows.size, dtype=np.bool_)
        some_inside = np.zeros(inner_rows.size, dtype=np.bool_)
        for point in range(3):
            value = _interval_cross(
                inner_corners[index], inner_corners[following], outer_corners[point]
            )
            # An exactly shared corner lies on the edge line: its term is 0.
            on_line = (outer_ids[:, point] == inner_ids[:, index]) | (
                outer_ids[:, point] == inner_ids[:, following]
            )
            nonpositive = np.where(positive, value[1] <= 0.0, value[0] >= 0.0)
            strictly_positive = np.where(positive, value[0] > 0.0, value[1] < 0.0)
            closed_outside &= nonpositive | on_line
            some_inside |= strictly_positive & ~on_line
        separates |= closed_outside
        every_edge_meets &= some_inside
    return signed & separates, signed & every_edge_meets


def triangle_pair_separation(
    first: Sequence[Polygon],
    second: Sequence[Polygon],
    pairs: np.ndarray,
    /,
) -> np.ndarray:
    """Binary64 screen of `interiors_separated` over ``(first row, second row)`` pairs.

    Returns int8 per pair: 1 when an edge line provably separates the
    interiors, 0 when provably no edge line of either triangle does, and -1
    when the outward-rounded interval evaluation of the exact edge predicate
    leaves it undecided (the exact test decides those). A corner shared exactly
    by both triangles contributes an exactly zero edge term. Non-triangles and
    nonrepresentable corners are undecided.
    """
    rows = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    decisions = np.full(rows.shape[0], -1, dtype=np.int8)
    if rows.shape[0] == 0:
        return decisions
    identities: dict[Point, int] = {}
    first_bank = _triangle_bank(first, identities)
    second_bank = first_bank if second is first else _triangle_bank(second, identities)
    valid = first_bank.valid[rows[:, 0]] & second_bank.valid[rows[:, 1]]
    selected = np.flatnonzero(valid)
    if selected.size == 0:
        return decisions
    first_rows, second_rows = rows[selected, 0], rows[selected, 1]
    forward, forward_meets = _directed_separation(
        first_bank, first_rows, second_bank, second_rows
    )
    backward, backward_meets = _directed_separation(
        second_bank, second_rows, first_bank, first_rows
    )
    separated = forward | backward
    decisions[selected[separated]] = 1
    decisions[selected[forward_meets & backward_meets & ~separated]] = 0
    return decisions


__all__ = [
    "Point",
    "Polygon",
    "PolygonBoxes",
    "area2",
    "box_overlap_pairs",
    "cross",
    "interiors_separated",
    "intersection",
    "line_key",
    "oriented_polygon",
    "plane_key",
    "plane_orientation",
    "polygon_boxes",
    "project_polygon",
    "projection_axes",
    "segment_overlap",
    "signed_area2",
    "triangle_pair_separation",
]
