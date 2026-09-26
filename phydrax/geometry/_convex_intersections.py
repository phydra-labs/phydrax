"""Deterministic host-side intersections of two-dimensional convex polygons.

The implementation deliberately does not use JAX.  Geometry preparation is a
host operation and therefore can reject an uncertain predicate before an
artifact is consumed by a compiled finite-volume program.  Every orientation
decision is certified by the geometric predicates of the precision policy
(exact with meshcore; otherwise filtered, with unresolved signs reported as
``UNCERTAIN_PREDICATE``).  Areas are correctly rounded values of the exact
shoelace sum over the constructed vertex coordinates.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .._geometry_precision import GeometryPrecisionPolicy
from ._predicates import orient2d, PredicateMode, resolve_host_predicate_mode


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

    ``mode`` is the effective predicate route.  An exactly zero orientation is
    certain contact; an unresolved filtered sign is counted as uncertain and is
    never promoted to either side of a half-plane.
    """

    mode: PredicateMode
    evaluated: int
    exact_zero: int
    uncertain_count: int

    @property
    def uncertain(self) -> bool:
        return self.uncertain_count != 0

    @property
    def predicate_uncertain(self) -> bool:
        return self.uncertain


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
    mode: PredicateMode
    evaluated: int = 0
    exact_zero: int = 0
    uncertain_count: int = 0

    def orient(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Batched orientation signs; unresolved entries carry ``UNCERTAIN`` (2)."""

        result = orient2d(a, b, c, mode=self.mode)
        signs = np.asarray(result.signs)
        self.evaluated += signs.size
        self.exact_zero += int(np.count_nonzero(signs == 0))
        self.uncertain_count += int(np.count_nonzero(~np.asarray(result.certain)))
        return signs

    def evidence(self) -> PredicateEvidence:
        return PredicateEvidence(
            mode=self.mode,
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


def _exact_products(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Error-free transformation: ``first * second`` as (rounded, error) pairs.

    Dekker's split is exact for binary64 inputs without overflow or underflow.
    """

    splitter = 134217729.0  # 2**27 + 1
    scaled_first = splitter * first
    first_high = scaled_first - (scaled_first - first)
    first_low = first - first_high
    scaled_second = splitter * second
    second_high = scaled_second - (scaled_second - second)
    second_low = second - second_high
    product = first * second
    error = (
        (first_high * second_high - product)
        + first_high * second_low
        + first_low * second_high
    ) + first_low * second_low
    return np.concatenate((product, error))


def _signed_area2(points: np.ndarray) -> float:
    """Twice the signed area, correctly rounded from the exact shoelace sum."""

    following = np.roll(points, -1, axis=0)
    terms = np.concatenate(
        (
            _exact_products(points[:, 0], following[:, 1]),
            -_exact_products(points[:, 1], following[:, 0]),
        )
    )
    return math.fsum(terms.tolist())


def _turns(points: np.ndarray, tracker: _PredicateTracker) -> np.ndarray:
    return tracker.orient(np.roll(points, 1, axis=0), points, np.roll(points, -1, axis=0))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    return bool(
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _self_intersection(
    points: np.ndarray,
    tracker: _PredicateTracker,
) -> bool | None:
    """Whether non-adjacent edges touch; ``None`` when a sign is unresolved."""

    count = points.shape[0]
    pairs = [
        (first, second)
        for first in range(count)
        for second in range(first + 2, count)
        if not (first == 0 and second == count - 1)
    ]
    if not pairs:
        return False
    first_index = np.asarray([pair[0] for pair in pairs], dtype=np.intp)
    second_index = np.asarray([pair[1] for pair in pairs], dtype=np.intp)
    a = points[first_index]
    b = points[(first_index + 1) % count]
    c = points[second_index]
    d = points[(second_index + 1) % count]
    signs = np.stack(
        (
            tracker.orient(a, b, c),
            tracker.orient(a, b, d),
            tracker.orient(c, d, a),
            tracker.orient(c, d, b),
        ),
        axis=1,
    )
    if np.any(signs == 2):
        return None
    for row, (ab_c, ab_d, cd_a, cd_b) in enumerate(signs.tolist()):
        if ab_c * ab_d < 0 and cd_a * cd_b < 0:
            return True
        touching = (
            (ab_c == 0 and _on_segment(a[row], b[row], c[row]))
            or (ab_d == 0 and _on_segment(a[row], b[row], d[row]))
            or (cd_a == 0 and _on_segment(c[row], d[row], a[row]))
            or (cd_b == 0 and _on_segment(c[row], d[row], b[row]))
        )
        if touching:
            return True
    return False


def _prepare_polygon(
    points: np.ndarray,
    tracker: _PredicateTracker,
) -> tuple[np.ndarray | None, IntersectionStatus | None]:
    # Check non-adjacent edges before convexity.  This distinguishes a bow-tie
    # (whose turns may all share a sign) from a convex polygon.
    crossing = _self_intersection(points, tracker)
    if crossing is None:
        return None, IntersectionStatus.UNCERTAIN_PREDICATE
    if crossing:
        return None, IntersectionStatus.SELF_INTERSECTING

    # A simple polygon whose nonzero turns share one sign is convex with that
    # orientation; exactly collinear vertices are harmless and are removed.
    turns = _turns(points, tracker)
    if np.any(turns == 2):
        return None, IntersectionStatus.UNCERTAIN_PREDICATE
    if not np.any(turns != 0):
        return None, IntersectionStatus.INVALID_INPUT
    if np.any(turns > 0) and np.any(turns < 0):
        return None, IntersectionStatus.NONCONVEX_INPUT
    if np.any(turns < 0):
        points = points[::-1].copy()

    while points.shape[0] >= 3:
        turns = _turns(points, tracker)
        if np.any(turns == 2):
            return None, IntersectionStatus.UNCERTAIN_PREDICATE
        if np.any(turns < 0):
            return None, IntersectionStatus.NONCONVEX_INPUT
        remove = np.flatnonzero(turns == 0)
        if remove.size == 0:
            break
        if remove.size >= points.shape[0] - 2:
            return None, IntersectionStatus.INVALID_INPUT
        points = np.delete(points, remove, axis=0)

    if points.shape[0] < 3:
        return None, IntersectionStatus.INVALID_INPUT
    return points, None


def _clip_by_edge(
    polygon: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tracker: _PredicateTracker,
) -> tuple[np.ndarray, bool]:
    """Clip by the left half-plane of ``start -> end``; flag unresolved decisions."""

    if polygon.shape[0] == 0:
        return polygon, False
    signs = tracker.orient(
        np.broadcast_to(start, polygon.shape),
        np.broadcast_to(end, polygon.shape),
        polygon,
    )
    if np.any(signs == 2):
        return polygon, True
    edge = end - start
    # Signed distances only construct crossing points; the side of every vertex
    # is the certified sign.
    distances = edge[0] * (polygon[:, 1] - start[1]) - edge[1] * (
        polygon[:, 0] - start[0]
    )
    output: list[np.ndarray] = []
    previous = polygon.shape[0] - 1
    for current in range(polygon.shape[0]):
        previous_inside = signs[previous] >= 0
        current_inside = signs[current] >= 0
        if current_inside != previous_inside and signs[previous] * signs[current] < 0:
            denominator = distances[previous] - distances[current]
            if denominator == 0.0:
                tracker.uncertain_count += 1
                return polygon, True
            # Clamp: the certified signs place the crossing on the segment.
            fraction = min(max(distances[previous] / denominator, 0.0), 1.0)
            output.append(
                polygon[previous] + fraction * (polygon[current] - polygon[previous])
            )
        if current_inside:
            output.append(polygon[current])
        previous = current
    if not output:
        return np.empty((0, 2), dtype=np.float64), False
    return np.asarray(output, dtype=np.float64), False


def _remove_collinear(
    values: np.ndarray,
    tracker: _PredicateTracker,
) -> tuple[np.ndarray, IntersectionStatus | None]:
    """Drop exactly collinear constructed vertices of a clipped convex polygon.

    A constructed vertex whose turn contradicts the convex orientation is an
    unresolved construction, not permission to simplify.
    """

    while values.shape[0] >= 3:
        turns = _turns(values, tracker)
        if np.any(turns == 2):
            return values, IntersectionStatus.UNCERTAIN_PREDICATE
        if not np.any(turns != 0):
            return _contact_vertices(values), IntersectionStatus.ZERO_MEASURE
        if np.any(turns > 0) and np.any(turns < 0):
            tracker.uncertain_count += int(np.count_nonzero(turns < 0))
            return values, IntersectionStatus.UNCERTAIN_PREDICATE
        if np.any(turns < 0):
            values = values[::-1]
            continue
        remove = np.flatnonzero(turns == 0)
        if remove.size == 0:
            break
        values = np.delete(values, remove, axis=0)
    return values, None


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
    values = np.asarray(unique, dtype=np.float64)

    if values.shape[0] >= 3:
        values, status = _remove_collinear(values, tracker)
        if status is not None:
            return values, status
    if values.shape[0] == 1:
        return values, IntersectionStatus.ZERO_MEASURE
    if values.shape[0] == 2:
        if np.array_equal(values[0], values[1]):
            return values[:1], IntersectionStatus.ZERO_MEASURE
        return _sort_segment(values), IntersectionStatus.ZERO_MEASURE

    # Lexicographic rotation is invariant under cyclic input permutations.
    order = np.lexsort((values[:, 1], values[:, 0]))
    first = int(order[0])
    values = np.concatenate((values[first:], values[:first]), axis=0)
    return values, IntersectionStatus.SUCCESS


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


def _compensated_centroid(vertices: np.ndarray, area2: float) -> np.ndarray:
    if area2 == 0.0:
        if vertices.shape[0] == 0:
            return np.full(2, np.nan, dtype=np.float64)
        return np.asarray(np.mean(vertices, axis=0, dtype=np.float64))
    origin = vertices[0]
    first = vertices - origin
    second = np.roll(vertices, -1, axis=0) - origin
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    x_sum = math.fsum(((first[:, 0] + second[:, 0]) * cross).tolist())
    y_sum = math.fsum(((first[:, 1] + second[:, 1]) * cross).tolist())
    denominator = 3.0 * area2
    return origin + np.asarray(
        (x_sum / denominator, y_sum / denominator), dtype=np.float64
    )


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
    precision: GeometryPrecisionPolicy | None = None,
) -> IntersectionResult:
    """Intersect two host-side convex polygons conservatively.

    ``source`` and ``target`` may be triangles, quadrilaterals, or any finite
    convex polygon represented by an ``(N, 2)`` array.  The returned vertices
    are canonical CCW vertices.  A zero-area point or edge contact is returned
    explicitly with :attr:`IntersectionStatus.ZERO_MEASURE`.  Orientation
    decisions use ``precision.predicate_mode``; with ``EXACT`` and meshcore
    installed every decision is exact, otherwise unresolved filtered signs fail
    closed with :attr:`IntersectionStatus.UNCERTAIN_PREDICATE`.
    """

    policy = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(policy, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    tracker = _PredicateTracker(resolve_host_predicate_mode(policy.predicate_mode))
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
    if source_error is not None:
        return _empty_result(source_error, source_pair, target_pair, tracker)
    if target_error is not None:
        return _empty_result(target_error, source_pair, target_pair, tracker)
    if source_array is None or target_array is None:
        return _empty_result(
            IntersectionStatus.INVALID_INPUT, source_pair, target_pair, tracker
        )
    source_prepared, source_status = _prepare_polygon(source_array, tracker)
    if source_status is not None or source_prepared is None:
        return _empty_result(
            source_status or IntersectionStatus.INVALID_INPUT,
            source_pair,
            target_pair,
            tracker,
        )
    target_prepared, target_status = _prepare_polygon(target_array, tracker)
    if target_status is not None or target_prepared is None:
        return _empty_result(
            target_status or IntersectionStatus.INVALID_INPUT,
            source_pair,
            target_pair,
            tracker,
        )
    clipped = source_prepared
    unresolved = False
    for index in range(target_prepared.shape[0]):
        clipped, unresolved = _clip_by_edge(
            clipped,
            target_prepared[index],
            target_prepared[(index + 1) % target_prepared.shape[0]],
            tracker,
        )
        if unresolved or clipped.shape[0] == 0:
            break
    if unresolved:
        return _empty_result(
            IntersectionStatus.UNCERTAIN_PREDICATE, source_pair, target_pair, tracker
        )
    vertices, status = _clean_intersection_vertices(clipped, tracker)
    if status is None:
        status = IntersectionStatus.EMPTY
    area2 = _signed_area2(vertices) if vertices.shape[0] >= 3 else 0.0
    area = abs(area2 * 0.5)
    if status is IntersectionStatus.SUCCESS:
        centroid = _compensated_centroid(vertices, area2)
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
