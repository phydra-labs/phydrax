#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact constrained projection and parity of authored affine CAD faces.

Binary-rational coefficient algebra cancels large source translations before
any distance arithmetic. Placed operation trees are evaluated algebraically;
no approximately orthogonal pose is replaced by its transpose as an inverse.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..._bvh import PackedBVH
from ...linalg._small_batched import (
    ExactSmallLinearActions,
    prepare_exact_small_linear_actions,
)
from ._intersection_curve import _affine_curve_coefficients
from ._model import BRepModel
from ._patches import AbstractCurve, AbstractSurfacePatch, PlanePatch
from ._placed import PlacedSurface
from ._root_bindings import BRepPlacedVertex
from ._sphere_membership import _square_root_interval


type Vector = tuple[Fraction, ...]


def _vector(values: np.ndarray, /) -> Vector:
    return tuple(Fraction(float(value)) for value in values)


def _add(first: Vector, second: Vector, /) -> Vector:
    return tuple(a + b for a, b in zip(first, second, strict=True))


def _subtract(first: Vector, second: Vector, /) -> Vector:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _scale(vector: Vector, factor: Fraction, /) -> Vector:
    return tuple(value * factor for value in vector)


def _dot(first: Vector, second: Vector, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _plane(patch: AbstractSurfacePatch, /) -> tuple[Vector, Vector, Vector] | None:
    if isinstance(patch, PlanePatch):
        return (
            _vector(np.asarray(patch.origin)),
            _vector(np.asarray(patch.first_axis)),
            _vector(np.asarray(patch.second_axis)),
        )
    if not isinstance(patch, PlacedSurface):
        return None
    source = _plane(patch.definition)
    if source is None:
        return None
    rows = tuple(_vector(row) for row in np.asarray(patch.rotation))
    mapped = tuple(tuple(_dot(row, value) for row in rows) for value in source)
    return _add(mapped[0], _vector(np.asarray(patch.translation))), mapped[1], mapped[2]


@dataclass(frozen=True, slots=True)
class AffineSegment:
    first: Vector
    last: Vector
    edge: int
    first_parameter: Fraction
    last_parameter: Fraction


@dataclass(frozen=True, slots=True)
class AffineFace:
    origin: Vector
    first_axis: Vector
    second_axis: Vector
    segments: tuple[AffineSegment, ...]
    projection_action: ExactSmallLinearActions

    def evaluate(self, uv: Vector, /) -> Vector:
        return _add(
            self.origin,
            _add(_scale(self.first_axis, uv[0]), _scale(self.second_axis, uv[1])),
        )

    def projection(self, point: Vector, /) -> Vector:
        delta = _subtract(point, self.origin)
        actions = self.projection_action.actions
        if actions is None or not self.projection_action.successful:
            raise RuntimeError("An affine face lost its native exact rank-two action.")
        return tuple(_dot(row, delta) for row in actions)

    def trim(self, uv: Vector, /) -> tuple[bool, bool]:
        winding = 0
        boundary = False
        x, y = uv
        for segment in self.segments:
            a, b = segment.first, segment.last
            cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            boundary |= (
                cross == 0
                and min(a[0], b[0]) <= x <= max(a[0], b[0])
                and min(a[1], b[1]) <= y <= max(a[1], b[1])
            )
            winding += int(a[1] <= y < b[1] and cross > 0)
            winding -= int(b[1] <= y < a[1] and cross < 0)
        return winding != 0 or boundary, boundary


@dataclass(frozen=True, slots=True)
class QueryBroadphase:
    lower: np.ndarray
    upper: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaves: np.ndarray
    items: np.ndarray

    @classmethod
    def prepare(cls, bvh: PackedBVH, /) -> QueryBroadphase:
        return cls(
            *(
                np.asarray(value)
                for value in (
                    bvh.bbox_min,
                    bvh.bbox_max,
                    bvh.left,
                    bvh.right,
                    bvh.leaf_id,
                    bvh.leaf_items,
                )
            )
        )

    def nearest(
        self,
        point: Vector,
        ceiling: list[Fraction | None],
        work: list[int],
        /,
    ) -> Iterator[int]:
        queue: list[tuple[Fraction, int]] = [(Fraction(0), 0)]
        while queue:
            bound, node = heapq.heappop(queue)
            if ceiling[0] is not None and bound > ceiling[0]:
                continue
            work[0] += 1
            leaf = self.leaves[node]
            if leaf >= 0:
                for item in self.items[leaf]:
                    if item >= 0:
                        yield int(item)
                continue
            for child in (self.left[node], self.right[node]):
                work[0] += 1
                gap = tuple(
                    max(
                        Fraction(0),
                        Fraction(float(self.lower[child, axis])) - point[axis],
                        point[axis] - Fraction(float(self.upper[child, axis])),
                    )
                    for axis in range(3)
                )
                heapq.heappush(queue, (_dot(gap, gap), int(child)))

    def ray(self, point: Vector, direction: Vector, work: list[int], /) -> Iterator[int]:
        queue = [0]
        while queue:
            node = queue.pop()
            work[0] += 1
            first, last = Fraction(0), None
            for axis in range(3):
                a = (Fraction(float(self.lower[node, axis])) - point[axis]) / direction[
                    axis
                ]
                b = (Fraction(float(self.upper[node, axis])) - point[axis]) / direction[
                    axis
                ]
                first = max(first, min(a, b))
                last = max(a, b) if last is None else min(last, max(a, b))
            if last is not None and last < first:
                continue
            leaf = self.leaves[node]
            if leaf >= 0:
                for item in self.items[leaf]:
                    if item >= 0:
                        yield int(item)
            else:
                queue.extend((int(self.left[node]), int(self.right[node])))


def prepare_affine_faces(
    model: BRepModel,
    faces: tuple[int, ...],
    /,
    *,
    pose: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[AffineFace, ...] | None:
    """Admit only literal parameter endpoints and exact affine trim carriers."""
    geometry = model.geometry
    if geometry is None:
        return None
    ranges = np.asarray(geometry.edge_ranges)
    deviations = np.asarray(model.coedge_deviation_bounds)
    prepared = []
    for index in faces:
        plane = _plane(model.patches[index])
        if plane is None:
            return None
        if pose is not None:
            rows = tuple(_vector(row) for row in pose[0])
            mapped = tuple(tuple(_dot(row, value) for row in rows) for value in plane)
            plane = (_add(mapped[0], _vector(pose[1])), mapped[1], mapped[2])
        gram = tuple(tuple(_dot(a, b) for b in plane[1:]) for a in plane[1:])
        action = prepare_exact_small_linear_actions(gram, plane[1:])
        if not action.successful:
            return None
        segments = []
        for loop in geometry.face_loops[index]:
            loop_start = len(segments)
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if deviations[coedge] != 0.0:
                    return None
                if any(root is not None for root in geometry.edge_endpoint_roots[edge]):
                    return None
                for vertex in geometry.edge_vertices[edge]:
                    root = geometry.vertex_roots[vertex]
                    if root is not None and (
                        not isinstance(root.primary, BRepPlacedVertex)
                        or root.primary.source_root is not None
                    ):
                        return None
                pcurve = geometry.pcurves[coedge]
                if not isinstance(pcurve, AbstractCurve):
                    return None
                coefficients = _affine_curve_coefficients(
                    pcurve, np.zeros((2,), dtype=np.float64)
                )
                if coefficients is None:
                    return None
                first, last = _vector(ranges[edge])
                if geometry.coedge_senses[coedge] < 0:
                    first, last = last, first
                a, b = (
                    _add(coefficients[0], _scale(coefficients[1], parameter))
                    for parameter in (first, last)
                )
                segments.append(AffineSegment(a, b, edge, first, last))
            loop_segments = segments[loop_start:]
            if any(
                a.last != b.first
                for a, b in zip(
                    loop_segments, (*loop_segments[1:], loop_segments[0]), strict=True
                )
            ):
                return None
        if not segments:
            return None
        prepared.append(AffineFace(*plane, tuple(segments), action))
    return tuple(prepared)


@dataclass(frozen=True, slots=True)
class AffineClosest:
    face: int
    point: Vector
    parameter: Vector
    squared_distance: Fraction
    edge: int
    edge_parameter: Fraction | None
    ambiguous: bool
    operations: int

    @property
    def distance_bounds(self) -> tuple[float, float]:
        return _square_root_interval(self.squared_distance)


def closest_affine(
    point: np.ndarray,
    faces: tuple[AffineFace, ...],
    boxes: np.ndarray,
    tolerance: float,
    broadphase: QueryBroadphase,
    indices: tuple[int, ...],
    /,
) -> AffineClosest:
    """Cover the full stationary set; prune only by conservative carrier boxes."""
    query = _vector(point)
    best: AffineClosest | None = None
    second: tuple[Fraction, Vector] | None = None
    operations = 0
    work = [0]
    ceiling: list[Fraction | None] = [None]
    selected = {source: local for local, source in enumerate(indices)}
    for source in broadphase.nearest(query, ceiling, work):
        if source not in selected:
            continue
        index = selected[source]
        face = faces[index]
        operations += 1
        gap = tuple(
            max(
                Fraction(0),
                Fraction(float(boxes[index, 0, axis])) - query[axis],
                query[axis] - Fraction(float(boxes[index, 1, axis])),
            )
            for axis in range(3)
        )
        if (
            best is not None
            and _dot(gap, gap)
            > (Fraction(best.distance_bounds[1]) + Fraction(tolerance)) ** 2
        ):
            continue
        uv = face.projection(query)
        operations += 1
        inside, _ = face.trim(uv)
        operations += len(face.segments)
        candidates: list[tuple[Vector, int, Fraction | None]] = []
        if inside:
            candidates.append((uv, -1, None))
        for segment in face.segments:
            operations += 1
            start, stop = face.evaluate(segment.first), face.evaluate(segment.last)
            direction = _subtract(stop, start)
            length = _dot(direction, direction)
            fraction = (
                min(
                    Fraction(1),
                    max(Fraction(0), _dot(_subtract(query, start), direction) / length),
                )
                if length
                else Fraction(0)
            )
            candidates.append(
                (
                    _add(
                        segment.first,
                        _scale(_subtract(segment.last, segment.first), fraction),
                    ),
                    segment.edge,
                    segment.first_parameter
                    + fraction * (segment.last_parameter - segment.first_parameter),
                )
            )
        for parameters, edge, edge_parameter in candidates:
            value = face.evaluate(parameters)
            delta = _subtract(value, query)
            squared = _dot(delta, delta)
            if best is None or squared < best.squared_distance:
                if best is not None and value != best.point:
                    second = best.squared_distance, best.point
                best = AffineClosest(
                    index,
                    value,
                    parameters,
                    squared,
                    edge,
                    edge_parameter,
                    False,
                    operations,
                )
            elif value != best.point:
                if second is None or squared < second[0]:
                    second = squared, value
            elif index == best.face and edge >= 0 and best.edge < 0:
                best = AffineClosest(
                    best.face,
                    best.point,
                    best.parameter,
                    squared,
                    edge,
                    edge_parameter,
                    False,
                    operations,
                )
        if best is not None:
            ceiling[0] = (Fraction(best.distance_bounds[1]) + Fraction(tolerance)) ** 2
    if best is None:
        raise ValueError("An affine boundary must contain a nonempty trim.")
    ambiguous = (
        second is not None
        and _square_root_interval(second[0])[0] <= best.distance_bounds[1] + tolerance
    )
    return AffineClosest(
        best.face,
        best.point,
        best.parameter,
        best.squared_distance,
        best.edge,
        best.edge_parameter,
        ambiguous,
        operations + work[0],
    )


def contains_affine(
    point: np.ndarray,
    faces: tuple[AffineFace, ...],
    broadphase: QueryBroadphase,
    indices: tuple[int, ...],
    /,
) -> tuple[bool, bool, int]:
    """Complete exact ray parity; edge/tangent coincidences remain unresolved."""
    query = _vector(point)
    operations = 0
    work = [0]
    selected = {source: local for local, source in enumerate(indices)}
    for values in ((1, 317, 613), (487, 1000, 293), (211, 719, 1000)):
        direction = tuple(Fraction(value) for value in values)
        crossings: set[Fraction] = set()
        unresolved = False
        for source in broadphase.ray(query, direction, work):
            if source not in selected:
                continue
            face = faces[selected[source]]
            operations += 1
            a, b = face.first_axis, face.second_axis
            normal = (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
            denominator = _dot(normal, direction)
            numerator = _dot(normal, _subtract(face.origin, query))
            if denominator == 0:
                unresolved |= numerator == 0
                continue
            parameter = numerator / denominator
            if parameter < 0:
                continue
            uv = face.projection(_add(query, _scale(direction, parameter)))
            operations += 1
            operations += len(face.segments)
            inside, boundary = face.trim(uv)
            if inside:
                unresolved |= boundary or parameter == 0 or parameter in crossings
                crossings.add(parameter)
        if not unresolved:
            return bool(len(crossings) % 2), True, operations + work[0]
    return False, False, operations + work[0]
