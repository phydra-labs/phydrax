"""Deterministic host-side intersections of two-dimensional convex polygons.

The implementation deliberately does not use JAX.  Geometry preparation is a
host operation and therefore can reject an uncertain predicate before an
artifact is consumed by a compiled finite-volume program.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


_DEFAULT_RELATIVE_TOLERANCE = 64.0 * np.finfo(np.float64).eps


class IntersectionStatus(str, Enum):
    """Terminal status of a convex-polygon intersection."""

    SUCCESS = "success"
    ZERO_MEASURE = "zero_measure"
    EMPTY = "empty"
    INVALID_INPUT = "invalid_input"
    NONFINITE_INPUT = "nonfinite_input"
    NONCONVEX_INPUT = "nonconvex_input"
    SELF_INTERSECTING = "self_intersecting"
    UNCERTAIN_PREDICATE = "uncertain_predicate"

    @property
    def successful(self) -> bool:
        """Whether the result contains a certified positive-area polygon."""

        return self is IntersectionStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class PredicateEvidence:
    """Evidence accumulated while evaluating orientation predicates.

    A predicate with an exactly zero value is certain contact.  A nonzero
    value at or below ``tolerance`` is instead marked uncertain: it is never
    silently promoted to either side of a half-plane.
    """

    minimum_abs_predicate: float
    predicate_scale: float
    tolerance: float
    evaluated: int
    exact_zero: int
    uncertain_count: int

    @property
    def uncertain(self) -> bool:
        return self.uncertain_count != 0

    @property
    def predicate_uncertain(self) -> bool:
        return self.uncertain

    @property
    def minimum_margin(self) -> float:
        return self.minimum_abs_predicate


@dataclass(frozen=True, slots=True)
class IntersectionResult:
    """Conservative geometric artifact for one source/target polygon pair."""

    status: IntersectionStatus
    vertices: np.ndarray
    area: float
    centroid: np.ndarray
    source_pair_id: str
    target_pair_id: str
    pair_id: str
    predicate_evidence: PredicateEvidence

    def __post_init__(self) -> None:
        # Arrays are host artifacts, not mutable scratch buffers.  Marking
        # them read-only prevents accidental mutation after fingerprinting.
        vertices = np.asarray(self.vertices, dtype=np.float64)
        centroid = np.asarray(self.centroid, dtype=np.float64)
        vertices.setflags(write=False)
        centroid.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "centroid", centroid)

    @property
    def successful(self) -> bool:
        return self.status is IntersectionStatus.SUCCESS

    @property
    def positive_measure(self) -> bool:
        return self.successful

    @property
    def zero_measure(self) -> bool:
        return self.status is IntersectionStatus.ZERO_MEASURE

    @property
    def canonical_vertices(self) -> np.ndarray:
        return self.vertices

    @property
    def compensated_area(self) -> float:
        return self.area

    @property
    def area_compensated(self) -> float:
        return self.area

    @property
    def predicate_uncertain(self) -> bool:
        return self.predicate_evidence.uncertain

    @property
    def evidence(self) -> PredicateEvidence:
        return self.predicate_evidence

    @property
    def uncertainty_evidence(self) -> PredicateEvidence:
        return self.predicate_evidence


@dataclass(slots=True)
class _PredicateTracker:
    tolerance: float
    scale: float
    evaluated: int = 0
    exact_zero: int = 0
    uncertain_count: int = 0
    minimum_abs: np.longdouble = np.longdouble(np.inf)

    def observe(self, value: Any, scale: Any = None) -> int:
        """Record a predicate and return its exact sign, or zero for contact."""

        value_ = np.longdouble(value)
        magnitude = np.abs(value_)
        self.evaluated += 1
        if magnitude < self.minimum_abs:
            self.minimum_abs = magnitude
        if value_ == 0:
            self.exact_zero += 1
            return 0
        predicate_scale = np.longdouble(self.scale if scale is None else scale)
        bound = np.longdouble(self.tolerance) * predicate_scale * predicate_scale
        if magnitude <= bound:
            self.uncertain_count += 1
        return 1 if value_ > 0 else -1

    def evidence(self) -> PredicateEvidence:
        minimum = 0.0 if not np.isfinite(self.minimum_abs) else float(self.minimum_abs)
        return PredicateEvidence(
            minimum_abs_predicate=minimum,
            predicate_scale=float(self.scale),
            tolerance=float(self.tolerance),
            evaluated=self.evaluated,
            exact_zero=self.exact_zero,
            uncertain_count=self.uncertain_count,
        )


def _as_points(points: Any) -> tuple[np.ndarray | None, IntersectionStatus | None]:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return None, IntersectionStatus.INVALID_INPUT
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 3:
        return None, IntersectionStatus.INVALID_INPUT
    if not np.all(np.isfinite(array)):
        return None, IntersectionStatus.NONFINITE_INPUT
    # A repeated closing point is conventional in host polygon formats.  It
    # is not a geometric vertex and would otherwise look like a degenerate
    # edge to the convexity checks.
    if np.array_equal(array[0], array[-1]):
        array = array[:-1]
    if array.shape[0] < 3:
        return None, IntersectionStatus.INVALID_INPUT
    keep = [0]
    for index in range(1, array.shape[0]):
        if not np.array_equal(array[index], array[keep[-1]]):
            keep.append(index)
    array = array[np.asarray(keep, dtype=np.intp)]
    if array.shape[0] < 3:
        return None, IntersectionStatus.INVALID_INPUT
    return array, None


def _extent(points_a: np.ndarray, points_b: np.ndarray | None = None) -> float:
    values = (
        points_a if points_b is None else np.concatenate((points_a, points_b), axis=0)
    )
    extent = np.max(values, axis=0) - np.min(values, axis=0)
    scale = float(np.max(extent))
    return max(scale, np.finfo(np.float64).tiny)


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.longdouble:
    ax = np.longdouble(a[0])
    ay = np.longdouble(a[1])
    bx = np.longdouble(b[0])
    by = np.longdouble(b[1])
    cx = np.longdouble(c[0])
    cy = np.longdouble(c[1])
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _signed_area2(points: np.ndarray) -> np.longdouble:
    """Return twice the signed area using a translation-invariant sum."""

    origin = points[0]
    total = np.longdouble(0.0)
    compensation = np.longdouble(0.0)
    for index in range(1, points.shape[0] - 1):
        first = points[index] - origin
        second = points[index + 1] - origin
        term = np.longdouble(first[0]) * np.longdouble(second[1])
        term -= np.longdouble(first[1]) * np.longdouble(second[0])
        corrected = term - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
    return total


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    return bool(
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tracker: _PredicateTracker,
) -> bool:
    scale = tracker.scale
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    signs = [
        tracker.observe(ab_c, scale),
        tracker.observe(ab_d, scale),
        tracker.observe(cd_a, scale),
        tracker.observe(cd_b, scale),
    ]
    if 0 in signs:
        if signs[0] == 0 and _on_segment(a, b, c):
            return True
        if signs[1] == 0 and _on_segment(a, b, d):
            return True
        if signs[2] == 0 and _on_segment(c, d, a):
            return True
        if signs[3] == 0 and _on_segment(c, d, b):
            return True
    return (signs[0] * signs[1] < 0) and (signs[2] * signs[3] < 0)


def _prepare_polygon(
    points: np.ndarray,
    tracker: _PredicateTracker,
) -> tuple[np.ndarray | None, IntersectionStatus | None]:
    count = points.shape[0]
    # Check non-adjacent edges before area/convexity.  This distinguishes a
    # bow-tie (whose signed area may be exactly zero) from a flat polygon.
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first == second or first_next == second or second_next == first:
                continue
            if _segments_intersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
                tracker,
            ):
                return None, IntersectionStatus.SELF_INTERSECTING

    area2 = _signed_area2(points)
    area_sign = tracker.observe(area2, tracker.scale)
    if area_sign == 0:
        return None, IntersectionStatus.INVALID_INPUT
    if tracker.uncertain_count:
        return None, IntersectionStatus.UNCERTAIN_PREDICATE
    if area_sign < 0:
        points = points[::-1].copy()

    # Exact collinear vertices are harmless and are removed.  A nonzero but
    # uncertain turn is not removed: doing so would inflate a tolerance and can
    # turn a thin positive-area cell into contact.
    changed = True
    while changed and points.shape[0] >= 3:
        changed = False
        count = points.shape[0]
        remove: list[int] = []
        for index in range(count):
            turn = _cross(
                points[(index - 1) % count], points[index], points[(index + 1) % count]
            )
            sign = tracker.observe(turn, tracker.scale)
            if sign < 0:
                return None, IntersectionStatus.NONCONVEX_INPUT
            if sign == 0:
                remove.append(index)
            elif tracker.uncertain_count:
                return None, IntersectionStatus.UNCERTAIN_PREDICATE
        if remove:
            if len(remove) >= points.shape[0] - 2:
                return None, IntersectionStatus.INVALID_INPUT
            points = np.delete(points, remove, axis=0)
            changed = True

    if points.shape[0] < 3:
        return None, IntersectionStatus.INVALID_INPUT
    return points, None


def _clip_by_edge(
    polygon: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tracker: _PredicateTracker,
) -> np.ndarray:
    if polygon.shape[0] == 0:
        return polygon
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_distance = _cross(start, end, previous)
    previous_sign = tracker.observe(previous_distance, tracker.scale)
    previous_inside = previous_sign >= 0
    for current in polygon:
        current_distance = _cross(start, end, current)
        current_sign = tracker.observe(current_distance, tracker.scale)
        current_inside = current_sign >= 0
        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            if denominator == 0:
                tracker.uncertain_count += 1
            else:
                fraction = previous_distance / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_distance = current_distance
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=np.longdouble)
    return np.asarray(output, dtype=np.longdouble)


def _clean_intersection_vertices(
    vertices: np.ndarray,
    tracker: _PredicateTracker,
) -> tuple[np.ndarray, IntersectionStatus | None]:
    if vertices.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64), IntersectionStatus.EMPTY
    unique: list[np.ndarray] = []
    for vertex in vertices:
        if not unique or not np.array_equal(vertex, unique[-1]):
            unique.append(vertex)
    if len(unique) > 1 and np.array_equal(unique[0], unique[-1]):
        unique.pop()
    if not unique:
        return np.empty((0, 2), dtype=np.float64), IntersectionStatus.EMPTY
    values = np.asarray(unique, dtype=np.longdouble)

    # Remove only exact collinear points.  Near-collinear positive turns are
    # evidence of an unresolved predicate, not permission to simplify.
    changed = True
    while changed and values.shape[0] >= 3:
        changed = False
        remove: list[int] = []
        for index in range(values.shape[0]):
            turn = _cross(
                values[(index - 1) % values.shape[0]],
                values[index],
                values[(index + 1) % values.shape[0]],
            )
            sign = tracker.observe(turn, tracker.scale)
            if sign == 0:
                remove.append(index)
            elif tracker.uncertain_count:
                return np.asarray(
                    values, dtype=np.float64
                ), IntersectionStatus.UNCERTAIN_PREDICATE
        if remove:
            if len(remove) >= values.shape[0] - 2:
                break
            values = np.delete(values, remove, axis=0)
            changed = True

    if values.shape[0] >= 3:
        area2 = _signed_area2(values)
        area_sign = tracker.observe(area2, tracker.scale)
        if tracker.uncertain_count:
            return np.asarray(
                values, dtype=np.float64
            ), IntersectionStatus.UNCERTAIN_PREDICATE
        if area_sign < 0:
            values = values[::-1]
        if area_sign == 0:
            return _contact_vertices(values), IntersectionStatus.ZERO_MEASURE
    elif values.shape[0] == 1:
        return np.asarray(values, dtype=np.float64), IntersectionStatus.ZERO_MEASURE
    elif values.shape[0] == 2:
        if np.array_equal(values[0], values[1]):
            return np.asarray(
                values[:1], dtype=np.float64
            ), IntersectionStatus.ZERO_MEASURE
        values = _sort_segment(values)
        return np.asarray(values, dtype=np.float64), IntersectionStatus.ZERO_MEASURE

    # Lexicographic rotation is invariant under cyclic input permutations.
    order = np.lexsort((np.asarray(values[:, 1]), np.asarray(values[:, 0])))
    first = int(order[0])
    values = np.concatenate((values[first:], values[:first]), axis=0)
    return np.asarray(values, dtype=np.float64), IntersectionStatus.SUCCESS


def _sort_segment(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((values[:, 1], values[:, 0]))
    return values[order]


def _contact_vertices(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    unique = np.unique(np.asarray(values, dtype=np.float64), axis=0)
    if unique.shape[0] <= 2:
        return _sort_segment(unique)
    return _sort_segment(np.asarray((unique[0], unique[-1]), dtype=np.float64))


def _compensated_centroid(vertices: np.ndarray, area2: np.longdouble) -> Any:
    if area2 == 0:
        if vertices.shape[0] == 0:
            return np.full(2, np.nan, dtype=np.float64)
        return np.asarray(np.mean(vertices, axis=0, dtype=np.float64))
    origin = np.asarray(vertices[0], dtype=np.longdouble)
    x_sum = np.longdouble(0.0)
    y_sum = np.longdouble(0.0)
    x_comp = np.longdouble(0.0)
    y_comp = np.longdouble(0.0)
    for index in range(vertices.shape[0]):
        other = (index + 1) % vertices.shape[0]
        first = np.asarray(vertices[index], dtype=np.longdouble) - origin
        second = np.asarray(vertices[other], dtype=np.longdouble) - origin
        cross = first[0] * second[1] - first[1] * second[0]
        x_term = (first[0] + second[0]) * cross
        y_term = (first[1] + second[1]) * cross
        x_corrected = x_term - x_comp
        x_updated = x_sum + x_corrected
        x_comp = (x_updated - x_sum) - x_corrected
        x_sum = x_updated
        y_corrected = y_term - y_comp
        y_updated = y_sum + y_corrected
        y_comp = (y_updated - y_sum) - y_corrected
        y_sum = y_updated
    denominator = 3.0 * area2
    centroid_offset = np.asarray(
        (x_sum / denominator, y_sum / denominator), dtype=np.longdouble
    )
    return np.asarray(origin + centroid_offset, dtype=np.float64)


def _stable_polygon_id(points: Any) -> str:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        payload = repr(points).encode("utf-8")
    else:
        if array.ndim == 2 and array.shape[1:] == (2,):
            if array.shape[0] > 1 and np.array_equal(array[0], array[-1]):
                array = array[:-1]
            # A sorted fallback is deterministic even for rejected input.  A
            # valid polygon gets a stronger cyclic canonical ID below.
            if array.shape[0] >= 3 and np.all(np.isfinite(array)):
                area2 = _signed_area2(array)
                if area2 < 0:
                    array = array[::-1]
                index = np.lexsort((array[:, 1], array[:, 0]))[0]
                array = np.concatenate((array[index:], array[:index]), axis=0)
            else:
                array = array[np.lexsort((array[:, 1], array[:, 0]))]
            payload = np.ascontiguousarray(array, dtype="<f8").tobytes()
        else:
            payload = repr(points).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _id_text(identifier: Any, fallback: str) -> str:
    return fallback if identifier is None else str(identifier)


def _empty_result(
    status: IntersectionStatus,
    source_pair_id: str,
    target_pair_id: str,
    tracker: _PredicateTracker,
) -> IntersectionResult:
    pair_id = hashlib.sha256(
        f"{source_pair_id}\0{target_pair_id}".encode("utf-8")
    ).hexdigest()[:24]
    return IntersectionResult(
        status=status,
        vertices=np.empty((0, 2), dtype=np.float64),
        area=0.0,
        centroid=np.full(2, np.nan, dtype=np.float64),
        source_pair_id=source_pair_id,
        target_pair_id=target_pair_id,
        pair_id=pair_id,
        predicate_evidence=tracker.evidence(),
    )


def intersect_convex_polygons(
    source: Any,
    target: Any,
    source_id: Any = None,
    target_id: Any = None,
    *,
    source_pair_id: Any = None,
    target_pair_id: Any = None,
    tolerance: float | None = None,
) -> IntersectionResult:
    """Intersect two host-side convex polygons conservatively.

    ``source`` and ``target`` may be triangles, quadrilaterals, or any finite
    convex polygon represented by an ``(N, 2)`` array.  The returned vertices
    are canonical CCW vertices.  A zero-area point or edge contact is returned
    explicitly with :attr:`IntersectionStatus.ZERO_MEASURE`; uncertain
    nonzero predicates fail closed with :attr:`IntersectionStatus.UNCERTAIN_PREDICATE`.
    """

    if tolerance is None:
        relative_tolerance = _DEFAULT_RELATIVE_TOLERANCE
    else:
        try:
            relative_tolerance = float(tolerance)
        except (TypeError, ValueError):
            relative_tolerance = np.nan
        if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
            relative_tolerance = np.nan

    source_array, source_error = _as_points(source)
    target_array, target_error = _as_points(target)
    source_fallback = _stable_polygon_id(source)
    target_fallback = _stable_polygon_id(target)
    source_pair = _id_text(
        source_pair_id if source_pair_id is not None else source_id, source_fallback
    )
    target_pair = _id_text(
        target_pair_id if target_pair_id is not None else target_id, target_fallback
    )
    extent_arrays = [array for array in (source_array, target_array) if array is not None]
    scale = (
        _extent(extent_arrays[0], extent_arrays[1]) if len(extent_arrays) == 2 else 1.0
    )
    tracker = _PredicateTracker(
        tolerance=float(relative_tolerance)
        if np.isfinite(relative_tolerance)
        else np.inf,
        scale=scale,
    )
    if not np.isfinite(relative_tolerance):
        return _empty_result(
            IntersectionStatus.INVALID_INPUT, source_pair, target_pair, tracker
        )
    if source_error is not None:
        return _empty_result(source_error, source_pair, target_pair, tracker)
    if target_error is not None:
        return _empty_result(target_error, source_pair, target_pair, tracker)

    if source_array is None:
        return _empty_result(
            IntersectionStatus.INVALID_INPUT, source_pair, target_pair, tracker
        )
    source_prepared, source_status = _prepare_polygon(source_array, tracker)
    if source_status is not None:
        return _empty_result(source_status, source_pair, target_pair, tracker)
    if source_prepared is None:
        return _empty_result(
            IntersectionStatus.INVALID_INPUT, source_pair, target_pair, tracker
        )
    if target_array is None:
        return _empty_result(
            IntersectionStatus.INVALID_INPUT, source_pair, target_pair, tracker
        )
    target_prepared, target_status = _prepare_polygon(target_array, tracker)
    if target_status is not None:
        return _empty_result(target_status, source_pair, target_pair, tracker)

    if target_prepared is None:
        return _empty_result(
            IntersectionStatus.INVALID_INPUT, source_pair, target_pair, tracker
        )
    clipped = np.asarray(source_prepared, dtype=np.longdouble)
    target_long = np.asarray(target_prepared, dtype=np.longdouble)
    for index in range(target_long.shape[0]):
        clipped = _clip_by_edge(
            clipped,
            target_long[index],
            target_long[(index + 1) % target_long.shape[0]],
            tracker,
        )
        if clipped.shape[0] == 0:
            break
    vertices, status = _clean_intersection_vertices(clipped, tracker)
    if status is None:
        status = IntersectionStatus.EMPTY
    area2 = _signed_area2(vertices) if vertices.shape[0] >= 3 else np.longdouble(0.0)
    area = abs(float(area2) * 0.5)
    if status is IntersectionStatus.SUCCESS:
        centroid = np.asarray(_compensated_centroid(vertices, area2), dtype=np.float64)
    elif vertices.shape[0] == 0:
        centroid = np.full(2, np.nan, dtype=np.float64)
    else:
        centroid = np.asarray(np.mean(vertices, axis=0), dtype=np.float64)
    if tracker.uncertain_count and status in (
        IntersectionStatus.SUCCESS,
        IntersectionStatus.ZERO_MEASURE,
        IntersectionStatus.EMPTY,
    ):
        status = IntersectionStatus.UNCERTAIN_PREDICATE
    pair_id = hashlib.sha256(f"{source_pair}\0{target_pair}".encode("utf-8")).hexdigest()[
        :24
    ]
    return IntersectionResult(
        status=status,
        vertices=vertices,
        area=area,
        centroid=centroid,
        source_pair_id=source_pair,
        target_pair_id=target_pair,
        pair_id=pair_id,
        predicate_evidence=tracker.evidence(),
    )


__all__ = [
    "IntersectionStatus",
    "PredicateEvidence",
    "IntersectionResult",
    "intersect_convex_polygons",
]
