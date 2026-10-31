#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import weakref
from abc import abstractmethod
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Protocol

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..linalg import orthonormal_frame


class _CurveTrimSegment(Protocol):
    def evaluate(self, parameters: Array, /) -> Array: ...

    def bounding_box(self, first: float, last: float, /) -> np.ndarray: ...

    def identity_payload(self, /) -> dict[str, object]: ...


class AbstractBoundaryMap(StrictModule):
    """Abstract batched map from uniform reference cells to a boundary."""

    @property
    @abstractmethod
    def num_charts(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def reference_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def ambient_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        raise NotImplementedError


class TrimClassification(StrictModule):
    """Three-way host classification; unresolved is never treated as outside."""

    inside: Array
    boundary: Array
    resolved: Array

    def __init__(
        self, inside: ArrayLike, boundary: ArrayLike, resolved: ArrayLike
    ) -> None:
        inside_ = jnp.asarray(inside, dtype=jnp.bool_)
        boundary_ = jnp.asarray(boundary, dtype=jnp.bool_)
        resolved_ = jnp.asarray(resolved, dtype=jnp.bool_)
        if inside_.shape != boundary_.shape or inside_.shape != resolved_.shape:
            raise ValueError("Trim classification arrays must have matching shapes.")
        self.inside, self.boundary, self.resolved = inside_, boundary_, resolved_


def _polygon_point_state(
    point: tuple[Fraction, Fraction],
    vertices: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> tuple[bool, bool]:
    inside = False
    px, py = point
    for start, end in zip(vertices, (*vertices[1:], vertices[0]), strict=True):
        x0, y0 = start
        x1, y1 = end
        cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        if (
            cross == 0
            and min(x0, x1) <= px <= max(x0, x1)
            and min(y0, y1) <= py <= max(y0, y1)
        ):
            return False, True
        if (y0 > py) != (y1 > py):
            crossing = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if px < crossing:
                inside = not inside
    return inside, False


class _SegmentEnclosures:
    """Lazily refined, memoized enclosure tree of one immutable trim segment.

    Subdivision is deterministic midpoint bisection of the unit interval, so
    every box query descends the same nodes; each node box and endpoint value
    is computed once per segment and shared by all queries.
    """

    __slots__ = ("boxes", "points")

    def __init__(self) -> None:
        self.boxes: dict[tuple[float, float], np.ndarray | None] = {}
        self.points: dict[float, np.ndarray | None] = {}


# Keyed by segment identity; records hold only host arrays, never the segment,
# and each entry is dropped when its segment is collected.
_SEGMENT_ENCLOSURES: dict[int, _SegmentEnclosures] = {}


def _segment_enclosures(segment: _CurveTrimSegment, /) -> _SegmentEnclosures:
    key = id(segment)
    record = _SEGMENT_ENCLOSURES.get(key)
    if record is None:
        record = _SegmentEnclosures()
        _SEGMENT_ENCLOSURES[key] = record
        weakref.finalize(segment, _SEGMENT_ENCLOSURES.pop, key, None)
    return record


def _turn_start_parameter(segment: _CurveTrimSegment, parameter: float, /) -> float:
    """Map the end of a self-closing whole native turn to its start parameter.

    Such a segment returns to its start point exactly; evaluating its end through
    the rounded binary64 turn would open the loop by the rounding of ``2 pi``.
    """
    from .brep._intersection import closes_native_turn
    from .brep._intersection_curve import CurveTrimSegment

    if not isinstance(segment, CurveTrimSegment):
        return parameter
    turn_end = 0.0 if segment.reversed else 1.0
    if parameter != turn_end or not closes_native_turn(
        segment.last_root, segment.first_root
    ):
        return parameter
    return 1.0 - turn_end


def _segment_point(segment: _CurveTrimSegment, parameter: float, /) -> np.ndarray | None:
    points = _segment_enclosures(segment).points
    if parameter not in points:
        try:
            value = np.asarray(
                segment.evaluate(
                    jnp.asarray(
                        _turn_start_parameter(segment, parameter), dtype=jnp.float64
                    )
                ),
                dtype=np.float64,
            ).reshape(-1)
        except (AttributeError, NotImplementedError, TypeError, ValueError, RuntimeError):
            points[parameter] = None
        else:
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                points[parameter] = None
            else:
                value.setflags(write=False)
                points[parameter] = value
    return points[parameter]


def _curve_segment_box(
    segment: _CurveTrimSegment, first: float, last: float, /
) -> np.ndarray | None:
    boxes = _segment_enclosures(segment).boxes
    if (first, last) in boxes:
        return boxes[first, last]
    result = _compute_segment_box(segment, first, last)
    if result is not None:
        result.setflags(write=False)
    boxes[first, last] = result
    return result


def _compute_segment_box(
    segment: _CurveTrimSegment, first: float, last: float, /
) -> np.ndarray | None:
    try:
        box = np.asarray(segment.bounding_box(first, last), dtype=np.float64)
    except (AttributeError, NotImplementedError, TypeError, ValueError, RuntimeError):
        return None
    start, end = _segment_point(segment, first), _segment_point(segment, last)
    if (
        start is None
        or end is None
        or box.shape != (2, 2)
        or not np.all(np.isfinite(box))
        or np.any(box[0] > box[1])
    ):
        return None
    return np.stack(
        (
            np.minimum(box[0], np.minimum(start, end)),
            np.maximum(box[1], np.maximum(start, end)),
        )
    )


def _boxes_disjoint(first: np.ndarray, second: np.ndarray, /) -> bool:
    return bool(np.any(first[1] < second[0]) or np.any(second[1] < first[0]))


def _box_misses_interior(first: np.ndarray, second: np.ndarray, /) -> bool:
    """Whether closed box ``first`` misses the open interior of ``second``."""
    return bool(np.any(first[1] <= second[0]) or np.any(second[1] <= first[0]))


type _ChordPiece = tuple[float, float, np.ndarray]


def _segment_chord_pieces(
    segment: _CurveTrimSegment, query_box: np.ndarray, /
) -> list[_ChordPiece] | None:
    """Order parameter pieces whose source boxes all avoid the query box."""
    pending: list[tuple[float, float, np.ndarray | None]] = [(0.0, 1.0, None)]
    accepted: list[_ChordPiece] = []
    while pending:
        first, last, parent_box = pending.pop()
        box = _curve_segment_box(segment, first, last)
        if box is None:
            return None
        if _boxes_disjoint(box, query_box):
            accepted.append((first, last, box))
            continue
        if parent_box is not None and np.array_equal(box, parent_box):
            return None
        middle = first + 0.5 * (last - first)
        if middle <= first or middle >= last:
            return None
        left_box = _curve_segment_box(segment, first, middle)
        right_box = _curve_segment_box(segment, middle, last)
        if left_box is None or right_box is None:
            return None
        if (
            not _boxes_disjoint(left_box, query_box) and np.array_equal(left_box, box)
        ) or (
            not _boxes_disjoint(right_box, query_box) and np.array_equal(right_box, box)
        ):
            return None
        pending.append((middle, last, box))
        pending.append((first, middle, box))
    return accepted


def _split_terminal_piece(
    segment: _CurveTrimSegment,
    pieces: list[_ChordPiece],
    query_box: np.ndarray,
    index: int,
    /,
) -> bool:
    first, last, _ = pieces[index]
    middle = first + 0.5 * (last - first)
    if not first < middle < last:
        return False
    left = _curve_segment_box(segment, first, middle)
    right = _curve_segment_box(segment, middle, last)
    if (
        left is None
        or right is None
        or not _boxes_disjoint(left, query_box)
        or not _boxes_disjoint(right, query_box)
    ):
        return False
    pieces[index : index + 1] = [(first, middle, left), (middle, last, right)]
    return True


def _junction_avoids_query(
    incoming: np.ndarray, outgoing: np.ndarray, query_box: np.ndarray, /
) -> bool | None:
    """Decide whether a junction gap between two numeric chord ends is admissible.

    Adjacent loop segments share one exact source root, contained in both end
    boxes. The gap chord and both true end arcs then lie in the hull of those
    boxes, so a hull avoiding the query preserves the winding parity. Disjoint
    end boxes cannot share that root and refuse the loop.
    """
    if _boxes_disjoint(incoming, outgoing):
        return None
    hull = np.stack(
        (np.minimum(incoming[0], outgoing[0]), np.maximum(incoming[1], outgoing[1]))
    )
    return _boxes_disjoint(hull, query_box)


def _curve_loop_chords(
    loop: tuple[_CurveTrimSegment, ...], query_box: np.ndarray, /
) -> tuple[tuple[Fraction, Fraction], ...] | None:
    pieces: list[list[_ChordPiece]] = []
    for segment in loop:
        accepted = _segment_chord_pieces(segment, query_box)
        if not accepted:
            return None
        pieces.append(accepted)
    count = len(loop)
    for index in range(count):
        previous = (index - 1) % count
        while True:
            incoming = pieces[previous][-1]
            outgoing = pieces[index][0]
            end = _segment_point(loop[previous], incoming[1])
            start = _segment_point(loop[index], outgoing[0])
            if end is None or start is None:
                return None
            if np.array_equal(end, start):
                break
            decision = _junction_avoids_query(incoming[2], outgoing[2], query_box)
            if decision is None:
                return None
            if decision:
                break
            split_incoming = _split_terminal_piece(
                loop[previous], pieces[previous], query_box, -1
            )
            split_outgoing = _split_terminal_piece(
                loop[index], pieces[index], query_box, 0
            )
            if not split_incoming and not split_outgoing:
                return None
    points: list[np.ndarray] = []
    for segment, accepted in zip(loop, pieces, strict=True):
        for first, last, _ in accepted:
            for parameter in (first, last):
                point = _segment_point(segment, parameter)
                if point is None:
                    return None
                if not points or not np.array_equal(points[-1], point):
                    points.append(point)
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points.pop()
    # The chord polygon is homotopic to the source loop in the query complement,
    # so even a degenerate one or two vertex polygon carries the correct parity.
    return tuple(
        (Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1])))
        for point in points
    )


def _curve_loops_point_state(
    loops: tuple[tuple[_CurveTrimSegment, ...], ...], point: np.ndarray, /
) -> tuple[bool, bool, bool]:
    query_box = np.stack((point, point))
    states: list[bool] = []
    for loop in loops:
        polygon = _curve_loop_chords(loop, query_box)
        if polygon is None:
            return False, False, False
        inside, boundary = _polygon_point_state(
            (
                Fraction.from_float(float(point[0])),
                Fraction.from_float(float(point[1])),
            ),
            polygon,
        )
        if boundary:
            return False, False, False
        states.append(inside)
    return bool(states and states[0] and not any(states[1:])), False, True


def _curve_loops_avoid_box(
    loops: tuple[tuple[_CurveTrimSegment, ...], ...],
    box: np.ndarray,
    disjoint: Callable[[np.ndarray, np.ndarray], bool] = _boxes_disjoint,
    /,
) -> bool:
    for loop in loops:
        for segment in loop:
            pending: list[tuple[float, float, np.ndarray | None]] = [(0.0, 1.0, None)]
            while pending:
                first, last, parent_box = pending.pop()
                image = _curve_segment_box(segment, first, last)
                if image is None:
                    return False
                if disjoint(image, box):
                    continue
                if parent_box is not None and np.array_equal(image, parent_box):
                    return False
                middle = first + 0.5 * (last - first)
                if middle <= first or middle >= last:
                    return False
                left = _curve_segment_box(segment, first, middle)
                right = _curve_segment_box(segment, middle, last)
                if left is None or right is None:
                    return False
                if (not disjoint(left, box) and np.array_equal(left, image)) or (
                    not disjoint(right, box) and np.array_equal(right, image)
                ):
                    return False
                pending.append((middle, last, image))
                pending.append((first, middle, image))
    return True


class TrimDomain(StrictModule):
    """One source trim domain: an oriented polygon or exact source curve loops."""

    outer: Array | None
    holes: tuple[Array, ...]
    curve_loops: tuple[tuple[_CurveTrimSegment, ...], ...]
    numeric_only: bool = eqx.field(static=True)

    def __init__(
        self,
        outer: npt.ArrayLike | None = None,
        holes: Sequence[npt.ArrayLike] = (),
        *,
        curve_loops: Sequence[Sequence[_CurveTrimSegment]] = (),
        _numeric_only: bool = False,
    ) -> None:
        if not isinstance(_numeric_only, bool):
            raise TypeError("_numeric_only must be bool.")
        loops = tuple(tuple(loop) for loop in curve_loops)
        if loops:
            from .brep._differentiable import _RealizedTrimSegment
            from .brep._intersection_curve import CurveTrimSegment

            if outer is not None or holes:
                raise ValueError(
                    "A TrimDomain must use either polygon loops or exact curve loops, not both."
                )
            if any(
                not loop
                or any(
                    not isinstance(segment, (CurveTrimSegment, _RealizedTrimSegment))
                    for segment in loop
                )
                for loop in loops
            ):
                raise TypeError(
                    "TrimDomain.curve_loops must contain nonempty trim-segment loops."
                )
            self.outer = None
            self.holes = ()
            self.curve_loops = loops
            self.numeric_only = _numeric_only or any(
                isinstance(segment, _RealizedTrimSegment)
                for loop in loops
                for segment in loop
            )
            return
        if outer is None:
            raise ValueError(
                "A TrimDomain requires a polygon outer loop or exact curve loops."
            )
        if _numeric_only:
            outer_ = jnp.asarray(outer, dtype=jnp.float64)
            holes_ = tuple(jnp.asarray(hole, dtype=jnp.float64) for hole in holes)
            if outer_.ndim != 2 or outer_.shape[1] != 2 or outer_.shape[0] < 3:
                raise ValueError("TrimDomain.outer must have shape (num_points >= 3, 2).")
            if any(
                hole.ndim != 2 or hole.shape[1] != 2 or hole.shape[0] < 3
                for hole in holes_
            ):
                raise ValueError("Every trim hole must have shape (num_points >= 3, 2).")
            outer_ = eqx.error_if(
                outer_,
                jnp.any(~jnp.isfinite(outer_)),
                "TrimDomain polygon coordinates must be finite.",
            )
            holes_ = tuple(
                eqx.error_if(
                    hole,
                    jnp.any(~jnp.isfinite(hole)),
                    "TrimDomain polygon coordinates must be finite.",
                )
                for hole in holes_
            )
            self.outer = outer_
            self.holes = holes_
            self.curve_loops = ()
            self.numeric_only = True
            return
        outer_host = np.asarray(outer, dtype=np.float64)
        hole_hosts = tuple(np.asarray(hole, dtype=np.float64) for hole in holes)
        if outer_host.ndim != 2 or outer_host.shape[1] != 2 or outer_host.shape[0] < 3:
            raise ValueError("TrimDomain.outer must have shape (num_points >= 3, 2).")
        if any(
            hole.ndim != 2 or hole.shape[1] != 2 or hole.shape[0] < 3
            for hole in hole_hosts
        ):
            raise ValueError("Every trim hole must have shape (num_points >= 3, 2).")
        if not np.all(np.isfinite(outer_host)) or any(
            not np.all(np.isfinite(hole)) for hole in hole_hosts
        ):
            raise ValueError("TrimDomain polygon coordinates must be finite.")
        self.outer = jnp.asarray(outer_host, dtype=jnp.float64)
        self.holes = tuple(jnp.asarray(hole, dtype=jnp.float64) for hole in hole_hosts)
        self.curve_loops = ()
        self.numeric_only = False

    @classmethod
    def from_curve_loops(
        cls, curve_loops: Sequence[Sequence[_CurveTrimSegment]], /
    ) -> TrimDomain:
        return cls(curve_loops=curve_loops)

    def identity_payload(self) -> dict[str, object]:
        if self.numeric_only:
            raise NotImplementedError(
                "A numeric trim realization has no exact source identity."
            )
        if self.curve_loops:
            from .brep._intersection_curve import CurveTrimSegment

            if any(
                not isinstance(segment, CurveTrimSegment)
                for loop in self.curve_loops
                for segment in loop
            ):
                raise NotImplementedError(
                    "A numeric trim realization has no exact source identity."
                )
            return {
                "kind": "source-curve-trim-domain",
                "curve_loops": tuple(
                    tuple(segment.identity_payload() for segment in loop)
                    for loop in self.curve_loops
                ),
            }
        return {"kind": "polygon-trim-domain", "outer": self.outer, "holes": self.holes}

    @staticmethod
    def _inside_loop(points: Array, loop: Array) -> Array:
        start = loop
        end = jnp.roll(loop, -1, axis=0)
        x = points[..., 0, None]
        y = points[..., 1, None]
        crossing = (start[:, 1] > y) != (end[:, 1] > y)
        intersection = (end[:, 0] - start[:, 0]) * (y - start[:, 1]) / jnp.where(
            end[:, 1] != start[:, 1],
            end[:, 1] - start[:, 1],
            1.0,
        ) + start[:, 0]
        return jnp.sum(crossing & (x < intersection), axis=-1) % 2 == 1

    def contains(self, reference: Array, /) -> Array:
        if self.curve_loops or self.outer is None:
            raise NotImplementedError(
                "Curve-backed TrimDomain membership requires certified host classification; no chord mask is authoritative."
            )
        reference_ = jnp.asarray(reference, dtype=jnp.float64)
        inside = self._inside_loop(reference_, self.outer)
        for hole in self.holes:
            inside &= ~self._inside_loop(reference_, hole)
        return inside

    def classify(self, points: npt.ArrayLike, /) -> TrimClassification:
        points_ = np.asarray(points, dtype=np.float64)
        if points_.ndim < 1 or points_.shape[-1] != 2 or not np.all(np.isfinite(points_)):
            raise ValueError(
                "TrimDomain.classify points must be finite with trailing dimension two."
            )
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 2))
        inside = np.zeros((flat.shape[0],), dtype=np.bool_)
        boundary = np.zeros_like(inside)
        resolved = np.zeros_like(inside)
        if self.curve_loops:
            for index, point in enumerate(flat):
                inside[index], boundary[index], resolved[index] = (
                    _curve_loops_point_state(self.curve_loops, point)
                )
        else:
            assert self.outer is not None
            polygon_loops = (self.outer, *self.holes)
            rational_loops = tuple(
                tuple(
                    (
                        Fraction.from_float(float(row[0])),
                        Fraction.from_float(float(row[1])),
                    )
                    for row in np.asarray(loop, dtype=np.float64)
                )
                for loop in polygon_loops
            )
            for index, point in enumerate(flat):
                query = (
                    Fraction.from_float(float(point[0])),
                    Fraction.from_float(float(point[1])),
                )
                states = tuple(
                    _polygon_point_state(query, loop) for loop in rational_loops
                )
                boundary[index] = any(on_boundary for _, on_boundary in states)
                if boundary[index]:
                    resolved[index] = True
                    continue
                inside[index] = bool(
                    states[0][0] and not any(state[0] for state in states[1:])
                )
                resolved[index] = True
        return TrimClassification(
            inside.reshape(leading), boundary.reshape(leading), resolved.reshape(leading)
        )

    def classify_boxes(
        self, boxes: npt.ArrayLike, /, *, interior: bool = False
    ) -> TrimClassification:
        """Classify closed boxes, or with ``interior`` their open interiors.

        An open interior that no trim curve meets lies on one side of the trim,
        so trim curves running along its edges do not leave it undecided.
        """
        boxes_ = np.asarray(boxes, dtype=np.float64)
        if (
            boxes_.ndim < 3
            or boxes_.shape[-2:] != (2, 2)
            or not np.all(np.isfinite(boxes_))
            or np.any(boxes_[..., 0, :] > boxes_[..., 1, :])
        ):
            raise ValueError(
                "TrimDomain.classify_boxes requires finite ordered (..., 2, 2) boxes."
            )
        disjoint = _box_misses_interior if interior else _boxes_disjoint
        leading = boxes_.shape[:-2]
        flat = boxes_.reshape((-1, 2, 2))
        inside = np.zeros((flat.shape[0],), dtype=np.bool_)
        boundary = np.zeros_like(inside)
        resolved = np.zeros_like(inside)
        for index, box in enumerate(flat):
            if self.curve_loops:
                clear = _curve_loops_avoid_box(self.curve_loops, box, disjoint)
            else:
                assert self.outer is not None
                clear = True
                for loop in (self.outer, *self.holes):
                    vertices = np.asarray(loop, dtype=np.float64)
                    for start, end in zip(
                        vertices, np.roll(vertices, -1, axis=0), strict=True
                    ):
                        segment_box = np.stack(
                            (np.minimum(start, end), np.maximum(start, end))
                        )
                        if not disjoint(segment_box, box):
                            clear = False
                            break
                    if not clear:
                        break
            if not clear:
                continue
            center = box[0] + 0.5 * (box[1] - box[0])
            decision = self.classify(center[None, :])
            if not bool(np.asarray(decision.resolved[0])):
                continue
            inside[index] = bool(np.asarray(decision.inside[0]))
            boundary[index] = bool(np.asarray(decision.boundary[0]))
            resolved[index] = True
        return TrimClassification(
            inside.reshape(leading), boundary.reshape(leading), resolved.reshape(leading)
        )


class BoundaryFrame(StrictModule):
    """Physical chart frame evaluated at one or more reference points."""

    origin: Array
    tangents: Array
    normal: Array
    jacobian: Array
    rank: Array
    condition_estimate: Array
    regularity_margin: Array
    finite: Array
    regular: Array
    jacobian_consistent: Array

    def __init__(
        self,
        *,
        origin: Array,
        tangents: Array,
        normal: Array,
        jacobian: Array,
        rank: Array,
        condition_estimate: Array,
        regularity_margin: Array,
        finite: Array,
        regular: Array,
        jacobian_consistent: Array,
    ) -> None:
        self.origin = jnp.asarray(origin, dtype=jnp.float64)
        self.tangents = jnp.asarray(tangents, dtype=jnp.float64)
        self.normal = jnp.asarray(normal, dtype=jnp.float64)
        self.jacobian = jnp.asarray(jacobian, dtype=jnp.float64)
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.condition_estimate = jnp.asarray(condition_estimate, dtype=jnp.float64)
        self.regularity_margin = jnp.asarray(regularity_margin, dtype=jnp.float64)
        self.finite = jnp.asarray(finite, dtype=jnp.bool_)
        self.regular = jnp.asarray(regular, dtype=jnp.bool_)
        self.jacobian_consistent = jnp.asarray(
            jacobian_consistent,
            dtype=jnp.bool_,
        )


class BoundaryAtlas(StrictModule):
    """Representation-independent collection of oriented boundary charts."""

    mapping: AbstractBoundaryMap
    source_entity_ids: Array
    source_id: str = eqx.field(static=True)
    physical_tags: tuple[str, ...] = eqx.field(static=True)
    orientation: Array
    seam_owner: Array
    trim_domains: tuple[TrimDomain | None, ...]

    def __init__(
        self,
        mapping: AbstractBoundaryMap,
        *,
        source_entity_ids: Array,
        source_id: str,
        physical_tags: Sequence[str] | None = None,
        orientation: Array | None = None,
        seam_owner: Array | None = None,
        trim_domains: Sequence[TrimDomain | None] | None = None,
    ) -> None:
        entity_ids = jnp.asarray(source_entity_ids, dtype=jnp.int32).reshape((-1,))
        if entity_ids.shape != (mapping.num_charts,):
            raise ValueError("source_entity_ids must contain one ID per boundary chart.")
        tags = (
            tuple("boundary" for _ in range(mapping.num_charts))
            if physical_tags is None
            else tuple(physical_tags)
        )
        if len(tags) != mapping.num_charts or any(not tag for tag in tags):
            raise ValueError("physical_tags must contain one non-empty tag per chart.")
        orientation_ = (
            jnp.ones((mapping.num_charts,), dtype=jnp.float64)
            if orientation is None
            else jnp.asarray(orientation, dtype=jnp.float64).reshape((-1,))
        )
        if orientation_.shape != (mapping.num_charts,):
            raise ValueError("orientation must contain one sign per chart.")
        if isinstance(orientation_, jax.core.Tracer):
            orientation_ = eqx.error_if(
                orientation_,
                jnp.any((orientation_ != 1.0) & (orientation_ != -1.0)),
                "orientation entries must be +1 or -1.",
            )
        else:
            orientation_host = np.asarray(orientation_)
            if np.any((orientation_host != 1.0) & (orientation_host != -1.0)):
                raise ValueError("orientation entries must be +1 or -1.")
        seam_owner_ = (
            jnp.ones((mapping.num_charts,), dtype=jnp.bool_)
            if seam_owner is None
            else jnp.asarray(seam_owner, dtype=jnp.bool_).reshape((-1,))
        )
        if seam_owner_.shape != (mapping.num_charts,):
            raise ValueError("seam_owner must contain one flag per chart.")
        trims = (
            tuple(None for _ in range(mapping.num_charts))
            if trim_domains is None
            else tuple(trim_domains)
        )
        if len(trims) != mapping.num_charts or any(
            trim is not None and not isinstance(trim, TrimDomain) for trim in trims
        ):
            raise ValueError(
                "trim_domains must contain one TrimDomain or None per chart."
            )
        if not source_id:
            raise ValueError("BoundaryAtlas.source_id must be non-empty.")
        self.mapping = mapping
        self.source_entity_ids = entity_ids
        self.source_id = source_id
        self.physical_tags = tags
        self.orientation = orientation_
        self.seam_owner = seam_owner_
        self.trim_domains = trims

    @property
    def num_charts(self) -> int:
        return self.mapping.num_charts

    @property
    def reference_dimension(self) -> int:
        return self.mapping.reference_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.mapping.ambient_dimension

    def _validate_inputs(
        self, chart_indices: ArrayLike, reference: ArrayLike
    ) -> tuple[Array, Array]:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=jnp.float64)
        if reference_.shape[:-1] != indices.shape:
            raise ValueError("chart_indices must match reference leading dimensions.")
        if reference_.shape[-1] != self.reference_dimension:
            raise ValueError(
                f"reference must have trailing dimension {self.reference_dimension}."
            )
        return indices, reference_

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        return self.mapping.map(indices, reference_)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        return self.mapping.jacobian(indices, reference_)

    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        """Return whether reference points lie inside their charts' trim domains."""
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        if self.reference_dimension != 2 or all(
            trim is None for trim in self.trim_domains
        ):
            return jnp.ones(indices.shape, dtype=jnp.bool_)
        flat_indices = indices.reshape((-1,))
        flat_reference = reference_.reshape((-1, self.reference_dimension))
        branches = tuple(
            (
                (lambda coordinate: jnp.asarray(True))
                if trim is None
                else (lambda coordinate, trim=trim: trim.contains(coordinate))
            )
            for trim in self.trim_domains
        )
        values = jax.vmap(
            lambda index, coordinate: jax.lax.switch(index, branches, coordinate)
        )(flat_indices, flat_reference)
        return values.reshape(indices.shape)

    def differential(self, chart_indices: Array, reference: Array, /) -> Array:
        """Return chart derivatives with shape ``(..., ambient_dim, reference_dimension)``."""
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        leading = indices.shape
        flat_indices = indices.reshape((-1,))
        flat_reference = reference_.reshape((-1, self.reference_dimension))
        differential = jax.vmap(
            lambda index, coordinate: jax.jacfwd(
                lambda value: self.mapping.map(index, value)
            )(coordinate)
        )(flat_indices, flat_reference)
        return differential.reshape(
            (*leading, self.ambient_dimension, self.reference_dimension)
        )

    def frame(self, chart_indices: Array, reference: Array, /) -> BoundaryFrame:
        indices, reference_ = self._validate_inputs(chart_indices, reference)
        if self.reference_dimension + 1 != self.ambient_dimension:
            raise NotImplementedError(
                "BoundaryFrame represents codimension-one charts only."
            )
        origin = self.mapping.map(indices, reference_)
        differential = self.differential(indices, reference_)
        if self.reference_dimension == 1 and self.ambient_dimension == 2:
            tangent = differential[..., :, 0]
            tangent_norm = jnp.linalg.norm(tangent, axis=-1)
            tangent_unit = tangent / jnp.maximum(
                tangent_norm[..., None],
                jnp.finfo(tangent.dtype).tiny,
            )
            tangents = tangent_unit[..., None, :]
            normal = jnp.stack((tangent_unit[..., 1], -tangent_unit[..., 0]), axis=-1)
            singular_values = tangent_norm[..., None]
            derived_jacobian = tangent_norm
        elif self.reference_dimension == 2 and self.ambient_dimension == 3:
            first = differential[..., :, 0]
            second = differential[..., :, 1]
            first_norm = jnp.linalg.norm(first, axis=-1)
            first_unit = first / jnp.maximum(
                first_norm[..., None],
                jnp.finfo(first.dtype).tiny,
            )
            second_orthogonal = (
                second - jnp.sum(second * first_unit, axis=-1, keepdims=True) * first_unit
            )
            second_norm = jnp.linalg.norm(second_orthogonal, axis=-1)
            second_unit = second_orthogonal / jnp.maximum(
                second_norm[..., None],
                jnp.finfo(second.dtype).tiny,
            )
            tangents = jnp.stack((first_unit, second_unit), axis=-2)
            normal = jnp.cross(first_unit, second_unit)
            singular_values = jnp.stack((first_norm, second_norm), axis=-1)
            derived_jacobian = first_norm * second_norm
        else:
            orthogonal = orthonormal_frame(differential)
            tangents = jnp.swapaxes(orthogonal.tangents, -1, -2)
            normal = orthogonal.normal_basis[..., :, 0]
            singular_values = orthogonal.singular_values
            derived_jacobian = jnp.prod(singular_values, axis=-1)
        normal = normal * self.orientation[indices][..., None]
        jacobian = self.mapping.jacobian(indices, reference_)
        evidence_jacobian = jax.lax.stop_gradient(jacobian)
        evidence_derived = jax.lax.stop_gradient(derived_jacobian)
        evidence_singular = jax.lax.stop_gradient(singular_values)
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(evidence_jacobian), jnp.abs(evidence_derived)),
            1.0,
        )
        tolerance = 256.0 * jnp.finfo(jacobian.dtype).eps * scale
        minimum_singular = jnp.min(evidence_singular, axis=-1)
        maximum_singular = jnp.max(evidence_singular, axis=-1)
        rank = jnp.sum(
            evidence_singular > tolerance[..., None],
            axis=-1,
            dtype=jnp.int32,
        )
        condition_estimate = maximum_singular / jnp.maximum(
            minimum_singular,
            jnp.finfo(maximum_singular.dtype).tiny,
        )
        regularity_margin = minimum_singular - tolerance
        jacobian_consistent = jnp.abs(evidence_jacobian - evidence_derived) <= tolerance
        finite = (
            jnp.all(
                jnp.isfinite(jax.lax.stop_gradient(differential)),
                axis=(-2, -1),
            )
            & jnp.all(
                jnp.isfinite(jax.lax.stop_gradient(tangents)),
                axis=(-2, -1),
            )
            & jnp.all(
                jnp.isfinite(jax.lax.stop_gradient(normal)),
                axis=-1,
            )
            & jnp.isfinite(evidence_jacobian)
        )
        regular = (
            (rank == self.reference_dimension)
            & finite
            & (evidence_jacobian > 0.0)
            & jacobian_consistent
        )
        return BoundaryFrame(
            origin=origin,
            tangents=tangents,
            normal=normal,
            jacobian=jacobian,
            rank=rank,
            condition_estimate=condition_estimate,
            regularity_margin=regularity_margin,
            finite=finite,
            regular=regular,
            jacobian_consistent=jacobian_consistent,
        )

    def select(
        self,
        *,
        entity_ids: Sequence[int] | None = None,
        tags: Sequence[str] | None = None,
    ) -> BoundaryAtlas:
        """Select charts by source entity ID and/or physical tag."""
        mask = np.ones((self.num_charts,), dtype=np.bool_)
        if entity_ids is not None:
            mask = mask & np.asarray(
                np.isin(
                    np.asarray(self.source_entity_ids),
                    np.asarray(tuple(entity_ids), dtype=np.int32),
                ),
                dtype=np.bool_,
            )
        if tags is not None:
            selected_tags = frozenset(tags)
            mask = mask & np.asarray(
                [tag in selected_tags for tag in self.physical_tags], dtype=np.bool_
            )
        chart_indices = np.flatnonzero(mask).astype(np.int32)
        if chart_indices.size == 0:
            raise ValueError("BoundaryAtlas selection contains no charts.")
        return BoundaryAtlas(
            _SelectedBoundaryMap(self.mapping, jnp.asarray(chart_indices)),
            source_entity_ids=self.source_entity_ids[chart_indices],
            source_id=self.source_id,
            physical_tags=tuple(self.physical_tags[index] for index in chart_indices),
            orientation=self.orientation[chart_indices],
            seam_owner=self.seam_owner[chart_indices],
            trim_domains=tuple(self.trim_domains[index] for index in chart_indices),
        )

    def translated(self, offset: Array, /) -> BoundaryAtlas:
        return BoundaryAtlas(
            _TranslatedBoundaryMap(self.mapping, offset),
            source_entity_ids=self.source_entity_ids,
            source_id=self.source_id,
            physical_tags=self.physical_tags,
            orientation=self.orientation,
            seam_owner=self.seam_owner,
            trim_domains=self.trim_domains,
        )


class _SelectedBoundaryMap(AbstractBoundaryMap):
    base: AbstractBoundaryMap
    chart_indices: Array

    def __init__(self, base: AbstractBoundaryMap, chart_indices: Array) -> None:
        self.base = base
        self.chart_indices = jnp.asarray(chart_indices, dtype=jnp.int32).reshape((-1,))

    @property
    def num_charts(self) -> int:
        return self.chart_indices.shape[0]

    @property
    def reference_dimension(self) -> int:
        return self.base.reference_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(self.chart_indices[chart_indices], reference)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(self.chart_indices[chart_indices], reference)


class _TranslatedBoundaryMap(AbstractBoundaryMap):
    base: AbstractBoundaryMap
    offset: Array

    def __init__(self, base: AbstractBoundaryMap, offset: Array) -> None:
        offset_ = jnp.asarray(offset, dtype=jnp.float64).reshape((-1,))
        if offset_.shape != (base.ambient_dimension,):
            raise ValueError(
                f"Translation offset must have shape ({base.ambient_dimension},)."
            )
        self.base = base
        self.offset = offset_

    @property
    def num_charts(self) -> int:
        return self.base.num_charts

    @property
    def reference_dimension(self) -> int:
        return self.base.reference_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.base.ambient_dimension

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.map(chart_indices, reference) + self.offset

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        return self.base.jacobian(chart_indices, reference)


class _CircleBoundaryMap(AbstractBoundaryMap):
    center: Array
    radius: Array

    def __init__(self, center: Array, radius: Array) -> None:
        self.center = jnp.asarray(center, dtype=jnp.float64).reshape((2,))
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())

    @property
    def num_charts(self) -> int:
        return 4

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 2

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        angle = 0.5 * jnp.pi * (chart_indices.astype(reference.dtype) + reference[..., 0])
        direction = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
        return self.center + self.radius * direction

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return jnp.broadcast_to(0.5 * jnp.pi * self.radius, reference.shape[:-1])


class _SphereBoundaryMap(AbstractBoundaryMap):
    center: Array
    radius: Array

    def __init__(self, center: Array, radius: Array) -> None:
        self.center = jnp.asarray(center, dtype=jnp.float64).reshape((3,))
        self.radius = jnp.asarray(radius, dtype=jnp.float64).reshape(())

    @property
    def num_charts(self) -> int:
        return 2

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        chart = chart_indices.astype(reference.dtype)
        first = reference[..., 0]
        second = reference[..., 1]
        square_first = jnp.where(chart == 0.0, first, 1.0 - second)
        square_second = jnp.where(chart == 0.0, second, first + second)
        azimuth = 2.0 * jnp.pi * square_first
        vertical = 1.0 - 2.0 * square_second
        radial = jnp.sqrt(jnp.maximum(1.0 - vertical * vertical, 0.0))
        direction = jnp.stack(
            (
                radial * jnp.cos(azimuth),
                radial * jnp.sin(azimuth),
                vertical,
            ),
            axis=-1,
        )
        return self.center + self.radius * direction

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del chart_indices
        return jnp.broadcast_to(
            4.0 * jnp.pi * self.radius**2,
            reference.shape[:-1],
        )


class _BoxBoundaryMap(AbstractBoundaryMap):
    origins: Array
    first_axes: Array
    second_axes: Array
    jacobians: Array

    def __init__(self, center: Array, size: Array) -> None:
        center_ = jnp.asarray(center, dtype=jnp.float64).reshape((3,))
        size_ = jnp.asarray(size, dtype=jnp.float64).reshape((3,))
        half = 0.5 * size_
        hx, hy, hz = half
        dx, dy, dz = size_
        self.origins = center_ + jnp.asarray(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [-hx, -hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, -hz],
                [-hx, -hy, hz],
            ]
        )
        self.first_axes = jnp.asarray(
            [
                [0.0, dy, 0.0],
                [0.0, dy, 0.0],
                [dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
                [dx, 0.0, 0.0],
            ]
        )
        self.second_axes = jnp.asarray(
            [
                [0.0, 0.0, dz],
                [0.0, 0.0, dz],
                [0.0, 0.0, dz],
                [0.0, 0.0, dz],
                [0.0, dy, 0.0],
                [0.0, dy, 0.0],
            ]
        )
        self.jacobians = jnp.asarray(
            [dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy]
        )

    @property
    def num_charts(self) -> int:
        return 6

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        origins = self.origins[chart_indices]
        first_axes = self.first_axes[chart_indices]
        second_axes = self.second_axes[chart_indices]
        return (
            origins + reference[..., :1] * first_axes + reference[..., 1:2] * second_axes
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        del reference
        return self.jacobians[chart_indices]


def circle_boundary_atlas(
    center: Array,
    radius: Array,
    /,
    *,
    source_id: str,
) -> BoundaryAtlas:
    return BoundaryAtlas(
        _CircleBoundaryMap(center, radius),
        source_entity_ids=jnp.zeros((4,), dtype=jnp.int32),
        source_id=source_id,
    )


def sphere_boundary_atlas(
    center: Array,
    radius: Array,
    /,
    *,
    source_id: str,
) -> BoundaryAtlas:
    triangle = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    return BoundaryAtlas(
        _SphereBoundaryMap(center, radius),
        source_entity_ids=jnp.asarray([0, 0], dtype=jnp.int32),
        orientation=-jnp.ones((2,), dtype=jnp.float64),
        source_id=source_id,
        trim_domains=(TrimDomain(triangle), TrimDomain(triangle)),
    )


def box_boundary_atlas(
    center: Array,
    size: Array,
    /,
    *,
    source_id: str,
) -> BoundaryAtlas:
    return BoundaryAtlas(
        _BoxBoundaryMap(center, size),
        orientation=jnp.asarray([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0]),
        physical_tags=("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
        source_entity_ids=jnp.arange(6, dtype=jnp.int32),
        source_id=source_id,
    )


BoundaryMap = AbstractBoundaryMap


__all__ = [
    "AbstractBoundaryMap",
    "TrimDomain",
    "BoundaryAtlas",
    "BoundaryFrame",
    "BoundaryMap",
]
