#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Project onto authored affine CAD faces and classify polygon-face parity.

Binary-rational coefficient algebra cancels source translations before
distance arithmetic. Placed operation trees retain their coefficients;
an approximately orthogonal map is not inverted by its transpose.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..._bvh import PackedBVH
from ..._strict import StrictModule
from ..._validation import nonnegative_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg._small_batched import (
    ExactSmallLinearActions,
    prepare_exact_small_linear_actions,
)
from ...typing import checked
from .._rational import rational_cross, rational_dot, rational_vector
from ._coordinate_request import coordinate_source_request
from ._exact import square_root_interval
from ._intersection_curve import affine_curve_coefficients
from ._model import BRepGeometry, BRepModel
from ._patches import AbstractCurve, AbstractSurfacePatch, ExtrusionSurface, PlanePatch
from ._placed import PlacedSurface
from ._root_bindings import BRepPlacedVertex


type Vector = tuple[Fraction, ...]


def _add(first: Vector, second: Vector, /) -> Vector:
    return tuple(a + b for a, b in zip(first, second, strict=True))


def _subtract(first: Vector, second: Vector, /) -> Vector:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _scale(vector: Vector, factor: Fraction, /) -> Vector:
    return tuple(value * factor for value in vector)


def _plane(patch: AbstractSurfacePatch, /) -> tuple[Vector, Vector, Vector] | None:
    match patch:
        case PlanePatch():
            return (
                rational_vector(np.asarray(patch.origin)),
                rational_vector(np.asarray(patch.first_axis)),
                rational_vector(np.asarray(patch.second_axis)),
            )
        case ExtrusionSurface():
            coefficients = affine_curve_coefficients(
                patch.curve, np.zeros((3,), dtype=np.float64)
            )
            if coefficients is None:
                return None
            return (
                coefficients[0],
                coefficients[1],
                rational_vector(np.asarray(patch.direction)),
            )
        case PlacedSurface():
            source = _plane(patch.definition)
            if source is None:
                return None
            rows = tuple(rational_vector(row) for row in np.asarray(patch.rotation))
            mapped = tuple(
                tuple(rational_dot(row, value) for row in rows) for value in source
            )
            return (
                _add(mapped[0], rational_vector(np.asarray(patch.translation))),
                mapped[1],
                mapped[2],
            )
        case _:
            return None


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
    projection_coefficients: tuple[Vector, ...]

    def evaluate(self, uv: Vector, /) -> Vector:
        return _add(
            self.origin,
            _add(_scale(self.first_axis, uv[0]), _scale(self.second_axis, uv[1])),
        )

    def projection(self, point: Vector, /) -> Vector:
        delta = _subtract(point, self.origin)
        return tuple(rational_dot(row, delta) for row in self.projection_coefficients)

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
                heapq.heappush(queue, (rational_dot(gap, gap), int(child)))

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


class _AffinePlane(StrictModule):
    coefficients: tuple[Vector, Vector, Vector]
    action: ExactSmallLinearActions
    projection_coefficients: tuple[Vector, ...]
    source_coefficients: tuple[Vector, Vector, Vector]


class _AffineTrim(StrictModule):
    segments: tuple[AffineSegment, ...]


def _prepare_plane(
    patch: AbstractSurfacePatch,
    pose: tuple[np.ndarray, np.ndarray] | None,
    budget: CoordinateEnclosureBudget,
    /,
) -> _AffinePlane | None:
    plane = _plane(patch)
    if plane is None:
        return None
    source_coefficients = plane
    if pose is not None:
        rows = tuple(rational_vector(row) for row in pose[0])
        mapped = tuple(tuple(rational_dot(row, value) for row in rows) for value in plane)
        plane = (_add(mapped[0], rational_vector(pose[1])), mapped[1], mapped[2])
    gram = tuple(tuple(rational_dot(a, b) for b in plane[1:]) for a in plane[1:])
    action = prepare_exact_small_linear_actions(gram, plane[1:], budget=budget)
    coefficients = action.actions
    if not action.successful or coefficients is None:
        return None
    return _AffinePlane(plane, action, coefficients, source_coefficients)


def _affine_segment_agrees(
    geometry: BRepGeometry,
    edge: int,
    parameters: tuple[Vector, Vector],
    plane: tuple[Vector, Vector, Vector],
    /,
) -> bool:
    carrier_index = geometry.edge_curves[edge]
    if carrier_index < 0:
        return False
    match geometry.curves[carrier_index]:
        case AbstractCurve() as curve:
            source = affine_curve_coefficients(curve, np.zeros((3,), dtype=np.float64))
        case _:
            return False
    if source is None:
        return False
    origin, first_axis, second_axis = plane
    constant, derivative = parameters
    expected = (
        _add(
            origin,
            _add(_scale(first_axis, constant[0]), _scale(second_axis, constant[1])),
        ),
        _add(_scale(first_axis, derivative[0]), _scale(second_axis, derivative[1])),
    )
    return source == expected


def _prepare_segment(
    geometry: BRepGeometry,
    coedge: int,
    ranges: np.ndarray,
    deviations: np.ndarray,
    plane: tuple[Vector, Vector, Vector],
    /,
) -> AffineSegment | None:
    edge = geometry.coedge_edges[coedge]
    if any(root is not None for root in geometry.edge_endpoint_roots[edge]):
        return None
    for vertex in geometry.edge_vertices[edge]:
        root = geometry.vertex_roots[vertex]
        if root is not None:
            match root.primary:
                case BRepPlacedVertex(source_root=None):
                    pass
                case _:
                    return None
    match geometry.pcurves[coedge]:
        case AbstractCurve() as pcurve:
            coefficients = affine_curve_coefficients(
                pcurve, np.zeros((2,), dtype=np.float64)
            )
        case _:
            return None
    if coefficients is None:
        return None
    # A positive numerical enclosure is not evidence of a nonzero gap.
    # Admit it only when the authored coefficient equations prove zero gap.
    if deviations[coedge] != 0.0 and not _affine_segment_agrees(
        geometry, edge, coefficients, plane
    ):
        return None
    first, last = rational_vector(ranges[edge])
    if geometry.coedge_senses[coedge] < 0:
        first, last = last, first
    a, b = (
        _add(coefficients[0], _scale(coefficients[1], parameter))
        for parameter in (first, last)
    )
    return AffineSegment(a, b, edge, first, last)


def _prepare_trim(
    geometry: BRepGeometry,
    index: int,
    ranges: np.ndarray,
    deviations: np.ndarray,
    plane: tuple[Vector, Vector, Vector],
    /,
) -> _AffineTrim | None:
    segments: list[AffineSegment] = []
    for loop in geometry.face_loops[index]:
        loop_start = len(segments)
        for coedge in loop:
            segment = _prepare_segment(geometry, coedge, ranges, deviations, plane)
            if segment is None:
                return None
            segments.append(segment)
        loop_segments = segments[loop_start:]
        if any(
            a.last != b.first
            for a, b in zip(
                loop_segments, (*loop_segments[1:], loop_segments[0]), strict=True
            )
        ):
            return None
    return _AffineTrim(tuple(segments)) if segments else None


@checked
def prepare_affine_faces(
    model: BRepModel,
    faces: tuple[int, ...],
    /,
    *,
    pose: tuple[np.ndarray, np.ndarray] | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[AffineFace, ...] | None:
    """Prepare affine face actions and closed polygon trims with literal endpoints."""
    if budget is None:
        with coordinate_source_request((model, faces)) as budget:
            return prepare_affine_faces(model, faces, pose=pose, budget=budget)
    geometry = model.geometry
    if geometry is None:
        return None
    ranges = np.asarray(geometry.edge_ranges)
    deviations = np.asarray(model.coedge_deviation_bounds)
    prepared: list[AffineFace] = []
    for index in faces:
        plane = _prepare_plane(model.patches[index], pose, budget)
        if plane is None:
            return None
        trim = _prepare_trim(
            geometry, index, ranges, deviations, plane.source_coefficients
        )
        if trim is None:
            return None
        prepared.append(
            AffineFace(
                *plane.coefficients,
                trim.segments,
                plane.action,
                plane.projection_coefficients,
            )
        )
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
        return square_root_interval(self.squared_distance)


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
    query = rational_vector(point)
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
            and rational_dot(gap, gap)
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
            length = rational_dot(direction, direction)
            fraction = (
                min(
                    Fraction(1),
                    max(
                        Fraction(0),
                        rational_dot(_subtract(query, start), direction) / length,
                    ),
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
            squared = rational_dot(delta, delta)
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
        raise ValueError(
            "affine_boundary.trim must be nonempty; got no closest candidate"
        )
    ambiguous = (
        second is not None
        and square_root_interval(second[0])[0] <= best.distance_bounds[1] + tolerance
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


class AffineParity(StrictModule):
    """Certified polygon-face parity and bounded-work refusal evidence."""

    inside: bool
    complete: bool
    crossings: int
    operations: int
    unresolved_faces: tuple[int, ...]
    resource_exhausted: bool


@dataclass
class _ParityWork:
    maximum: int | None
    operations: int = 0

    def charge(self, amount: int = 1) -> bool:
        if self.maximum is not None and self.operations + amount > self.maximum:
            return False
        self.operations += amount
        return True


class _ParityPreparation(StrictModule):
    normals: tuple[Vector, ...]
    constraints: tuple[Vector, ...]
    unresolved_faces: tuple[int, ...]
    exhausted: bool


def _prepare_parity(
    query: Vector,
    faces: tuple[AffineFace, ...],
    indices: tuple[int, ...],
    work: _ParityWork,
    /,
) -> _ParityPreparation:
    normals: list[Vector] = []
    constraints: list[Vector] = []
    for source, face in zip(indices, faces, strict=True):
        if not work.charge():
            return _ParityPreparation(tuple(normals), tuple(constraints), (source,), True)
        normal = rational_cross(face.first_axis, face.second_axis)
        normals.append(normal)
        constraints.append(normal)
        if rational_dot(normal, _subtract(face.origin, query)) == 0:
            if not work.charge(1 + len(face.segments)):
                return _ParityPreparation(
                    tuple(normals), tuple(constraints), (source,), True
                )
            inside, _ = face.trim(face.projection(query))
            if inside:
                return _ParityPreparation(
                    tuple(normals), tuple(constraints), (source,), False
                )
        for segment in face.segments:
            if not work.charge():
                return _ParityPreparation(
                    tuple(normals), tuple(constraints), (source,), True
                )
            first = _subtract(face.evaluate(segment.first), query)
            last = _subtract(face.evaluate(segment.last), query)
            edge_plane = rational_cross(first, last)
            # A collinear query and edge already lie in their face plane; a
            # transverse ray can meet that line only at the boundary query.
            if any(edge_plane):
                constraints.append(edge_plane)
    return _ParityPreparation(tuple(normals), tuple(constraints), (), False)


def _parity_direction(
    constraints: tuple[Vector, ...], work: _ParityWork, /
) -> Vector | None:
    # Each nonzero n·(1,k,k²) has at most two roots. This finite candidate
    # count admits a transverse ray that misses every edge and vertex.
    for k in range(1, 2 * len(constraints) + 2):
        direction = (Fraction(1), Fraction(k), Fraction(k * k))
        admitted = True
        for normal in constraints:
            if not work.charge():
                return None
            if rational_dot(normal, direction) == 0:
                admitted = False
                break
        if admitted:
            return direction
    raise RuntimeError(
        "affine_parity.direction must avoid finitely many degree-two roots; got no admitted candidate"
    )


def certify_affine_parity(
    point: np.ndarray,
    faces: tuple[AffineFace, ...],
    indices: tuple[int, ...],
    /,
    *,
    maximum_operations: int | None = None,
) -> AffineParity:
    """Classify a nonboundary query by half-open polygon-face parity.

    Integer directions avoid every face tangent and query-edge plane. Polygon
    trim winding uses half-open edge intervals; crossing distances are counted
    once across shared faces. Boundary queries and exhausted work remain
    unresolved. The direction search has a finite degree-two exclusion bound.
    """
    if maximum_operations is not None:
        maximum_operations = nonnegative_integer(
            maximum_operations, "affine_parity.maximum_operations"
        )
    query = rational_vector(point)
    work = _ParityWork(maximum_operations)
    preparation = _prepare_parity(query, faces, indices, work)
    if preparation.unresolved_faces:
        return AffineParity(
            False,
            False,
            0,
            work.operations,
            preparation.unresolved_faces,
            preparation.exhausted,
        )
    direction = _parity_direction(preparation.constraints, work)
    if direction is None:
        return AffineParity(False, False, 0, work.operations, indices, True)
    crossings: set[Fraction] = set()
    for source, face, normal in zip(indices, faces, preparation.normals, strict=True):
        if not work.charge():
            return AffineParity(
                False, False, len(crossings), work.operations, (source,), True
            )
        numerator = rational_dot(normal, _subtract(face.origin, query))
        parameter = numerator / rational_dot(normal, direction)
        if parameter <= 0:
            continue
        if not work.charge(1 + len(face.segments)):
            return AffineParity(
                False, False, len(crossings), work.operations, (source,), True
            )
        uv = face.projection(_add(query, _scale(direction, parameter)))
        inside, boundary = face.trim(uv)
        if boundary:
            raise RuntimeError(
                "affine_parity.direction must miss trim boundaries; got an edge crossing"
            )
        if inside:
            crossings.add(parameter)
    return AffineParity(
        bool(len(crossings) % 2), True, len(crossings), work.operations, (), False
    )
