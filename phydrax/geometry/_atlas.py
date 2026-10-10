#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import orient2d, PredicateMode, resolve_host_predicate_mode
from .._strict import StrictModule
from ..linalg import orthonormal_frame
from ..typing import Dim, HostBool, HostFloat64, HostInt32, parse, Scope


def overlapping_box_pairs(
    lower: np.ndarray, upper: np.ndarray, maximum_pairs: int, /
) -> tuple[np.ndarray | None, int]:
    """Every ``(i, j)``, ``i < j``, whose closed 2D boxes overlap, and the work.

    Candidates come from a uniform bucket grid. Rounded box-to-bucket maps are
    monotone, so closed overlapping boxes share the bucket of a common point;
    long collinear chains (chart seams) meet only their neighbours instead of
    the whole active set of a sweep. Returns ``None`` when bucket incidences
    or candidate pairs exceed ``maximum_pairs``.
    """
    count = lower.shape[0]
    if count < 2:
        return np.empty((0, 2), dtype=np.int64), 0
    origin = np.min(lower, axis=0)
    extent = np.max(upper, axis=0) - origin
    size = np.where(extent > 0.0, extent / np.ceil(np.sqrt(count)), 1.0)
    first_cell = np.floor((lower - origin) / size).astype(np.int64)
    last_cell = np.floor((upper - origin) / size).astype(np.int64)
    spans = last_cell - first_cell + 1
    incidences = spans[:, 0] * spans[:, 1]
    if int(np.sum(incidences)) > maximum_pairs:
        return None, maximum_pairs
    rows = np.repeat(np.arange(count, dtype=np.int64), incidences)
    offsets = np.arange(rows.size, dtype=np.int64) - np.repeat(
        np.cumsum(incidences) - incidences, incidences
    )
    columns = spans[rows, 1]
    keys = (first_cell[rows, 0] + offsets // columns) * (
        int(np.max(last_cell[:, 1])) + 1
    ) + (first_cell[rows, 1] + offsets % columns)
    order = np.lexsort((rows, keys))
    keys, rows = keys[order], rows[order]
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    sizes = np.diff(np.concatenate((starts, [keys.size])))
    work = int(np.sum(sizes * (sizes - 1) // 2))
    if work > maximum_pairs:
        return None, maximum_pairs
    blocks = [
        np.stack(np.triu_indices(int(length), 1), axis=1) + int(start)
        for start, length in zip(starts, sizes, strict=True)
        if length > 1
    ]
    if not blocks:
        return np.empty((0, 2), dtype=np.int64), work
    pairs = np.unique(rows[np.concatenate(blocks)], axis=0)
    left, right = pairs[:, 0], pairs[:, 1]
    overlap = np.all(
        (upper[right] >= lower[left]) & (lower[right] <= upper[left]), axis=1
    )
    return pairs[overlap], work


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


class AbstractTrimCurve(StrictModule):
    """Oriented exact curve in a two-dimensional chart.

    Trim consumers use two capabilities: pointwise evaluation over
    ``parameter_interval`` and a conservative axis-aligned enclosure of any
    sub-arc. The enclosure contains the arc and therefore its chord, which makes
    chord winding exact for every point outside the enclosure.
    """

    @property
    @abstractmethod
    def parameter_interval(self) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> Array:
        """Chart coordinates ``(..., 2)`` at curve parameters ``(...)``."""
        raise NotImplementedError

    @abstractmethod
    def enclosure(self, first: float, last: float, /) -> np.ndarray:
        """Conservative ``(2, 2)`` box ``[lower, upper]`` of the arc ``[first, last]``."""
        raise NotImplementedError

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conservative oriented jets; unsupported capability is explicit."""
        raise NotImplementedError(
            "This trim curve does not provide interval derivative bounds."
        )

    def shares_endpoint(self, other: AbstractTrimCurve, /) -> bool:
        """Whether this end and the next start are one exact source parameter."""
        return False

    @property
    def ambient_dimension(self) -> int:
        return 2

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return self.parameter_interval

    @property
    def period(self) -> float | None:
        return None

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        lower, upper = self.parameter_interval
        if not (
            np.isfinite(first) and np.isfinite(last) and lower <= first < last <= upper
        ):
            raise ValueError("A trim curve range must lie inside its parameter interval.")
        return float(first), float(last)

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        self.validate_range(first, last)
        return self.enclosure(first, last)


class _LoopVertexDim(Dim):
    """Vertices of a polygonal trim loop or of a certified chord cover."""


class _ArcDim(Dim):
    """Certified arcs of a curve trim loop."""


class _TrimPointDim(Dim):
    """Classified chart points."""


class _TrimLoopDim(Dim):
    """Loops of one trim domain (outer first, then holes)."""


class TrimTopologyEvidence(StrictModule):
    """Source/chord isotopy proof, bound to the exact loop and chord polygon."""

    certified: bool = eqx.field(static=True)
    unresolved_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    pairs_checked: int = eqx.field(static=True)
    budget_exhausted: bool = eqx.field(static=True)
    loop_id: str = eqx.field(static=True)
    chord_id: str = eqx.field(static=True)


class PolygonTrimLoop(StrictModule):
    """Explicit affine trim loop: a closed polygon of chart points."""

    __strict_contract__ = True

    vertices: HostFloat64[_LoopVertexDim, Literal[2]]

    def __init__(self, vertices: npt.ArrayLike) -> None:
        host = np.asarray(vertices, dtype=np.float64)
        if host.ndim != 2 or host.shape[1] != 2 or host.shape[0] < 3:
            raise ValueError("A polygon trim loop must have shape (num_points >= 3, 2).")
        if not np.all(np.isfinite(host)):
            raise ValueError("Polygon trim loop vertices must be finite.")
        host = np.array(host, dtype=np.float64)
        host.setflags(write=False)
        self.vertices = host

    @property
    def chords(self) -> np.ndarray:
        return self.vertices

    @property
    def identity(self) -> dict[str, object]:
        return {"kind": "polygon", "vertices": array_tree_fingerprint(self.vertices)}


class CurveTrimLoop(StrictModule):
    """Oriented trim curves with a certified chord cover and bounded joins.

    Each curve is split into arcs until every arc enclosure has a diagonal no
    larger than ``tolerance``. The chord polygon of the arc endpoints classifies
    chart points exactly outside the union of arc enclosures (the approximation
    band); `TrimDomain.classify` resolves points inside the band by refining the
    exact curves. By default joins must be exact up to directed roundoff. A
    positive ``relative_closure_tolerance`` explicitly retains an already
    validated source-domain coincidence bound, authorizing declared topology
    without claiming exact geometric coincidence.
    """

    __strict_contract__ = True

    curves: tuple[AbstractTrimCurve, ...]
    chords: HostFloat64[_ArcDim, Literal[2]]
    source_chords: HostFloat64[_ArcDim, Literal[2]]
    arc_curves: HostInt32[_ArcDim]
    arc_first: HostFloat64[_ArcDim]
    arc_last: HostFloat64[_ArcDim]
    arc_lower: HostFloat64[_ArcDim, Literal[2]]
    arc_upper: HostFloat64[_ArcDim, Literal[2]]
    junction_lower: HostFloat64[_LoopVertexDim, Literal[2]]
    junction_upper: HostFloat64[_LoopVertexDim, Literal[2]]
    tolerance: float = eqx.field(static=True)
    relative_closure_tolerance: float = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)
    closure_gap: float = eqx.field(static=True)
    chord_endpoint_error: float = eqx.field(static=True)

    def __init__(
        self,
        curves: Sequence[AbstractTrimCurve],
        *,
        tolerance: float,
        maximum_arcs: int = 65536,
        arc_parameters: Sequence[npt.ArrayLike] | None = None,
        chord_vertices: npt.ArrayLike | None = None,
        relative_closure_tolerance: float = 0.0,
    ) -> None:
        curves_ = tuple(curves)
        if not curves_ or any(
            not isinstance(curve, AbstractTrimCurve) for curve in curves_
        ):
            raise TypeError("A curve trim loop requires AbstractTrimCurve values.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Curve trim loop tolerance must be finite and positive.")
        if maximum_arcs < len(curves_):
            raise ValueError("maximum_arcs cannot be smaller than the curve count.")
        starts = []
        ends = []
        for curve in curves_:
            first, last = curve.parameter_interval
            points = np.asarray(
                curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64)),
                dtype=np.float64,
            )
            starts.append(points[0])
            ends.append(points[1])
        gaps = [
            float(np.max(np.abs(ends[index] - starts[(index + 1) % len(curves_)])))
            for index in range(len(curves_))
        ]
        closure_gap = max(gaps)
        maximum_endpoint = float(np.max(np.abs(np.asarray((*starts, *ends)))))
        roundoff_tolerance = float(
            128.0 * np.finfo(np.float64).eps * (1.0 + maximum_endpoint)
        )
        relative_tolerance = float(relative_closure_tolerance)
        if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
            raise ValueError(
                "relative_closure_tolerance must be finite and non-negative."
            )
        closure_tolerance = max(
            roundoff_tolerance,
            relative_tolerance * max(1.0, maximum_endpoint),
        )
        exact_joins = [
            curve.shares_endpoint(curves_[(index + 1) % len(curves_)])
            for index, curve in enumerate(curves_)
        ]
        if any(
            gap > closure_tolerance and not exact
            for gap, exact in zip(gaps, exact_joins, strict=True)
        ):
            raise ValueError(f"Curve trim loop is not closed (gap {closure_gap:.3e}).")
        arcs: list[tuple[int, float, float, np.ndarray, np.ndarray]] = []
        if arc_parameters is not None and len(arc_parameters) != len(curves_):
            raise ValueError(
                "arc_parameters must provide one ordered partition per curve."
            )
        for index, curve in enumerate(curves_):
            first, last = curve.parameter_interval
            if arc_parameters is None:
                arcs.extend(
                    _certified_arcs(
                        curve, index, first, last, tolerance_, maximum_arcs - len(arcs)
                    )
                )
                continue
            partition = np.asarray(arc_parameters[index], dtype=np.float64)
            if (
                partition.ndim != 1
                or partition.size < 2
                or not np.all(np.isfinite(partition))
                or partition[0] != first
                or partition[-1] != last
                or np.any(np.diff(partition) <= 0)
            ):
                raise ValueError(
                    "Prescribed arc partitions must cover each full curve interval in order."
                )
            if len(arcs) + partition.size - 1 > maximum_arcs:
                raise ValueError("Prescribed trim cover exceeds its arc capacity.")
            points = np.asarray(curve.evaluate(jnp.asarray(partition)))
            for lower, upper, start in zip(
                partition[:-1], partition[1:], points[:-1], strict=True
            ):
                box, _ = _arc_record(curve, float(lower), float(upper), start)
                if np.linalg.norm(box[1] - box[0]) > tolerance_:
                    raise ValueError(
                        "A prescribed arc enclosure exceeds the requested cover tolerance."
                    )
                arcs.append((index, float(lower), float(upper), start, box))
        if chord_vertices is not None and arc_parameters is None:
            raise ValueError(
                "Prescribed chords require prescribed source arc partitions."
            )
        self.curves = curves_
        self.source_chords = np.asarray([arc[3] for arc in arcs], dtype=np.float64)
        self.chords = np.asarray(
            [arc[3] for arc in arcs] if chord_vertices is None else chord_vertices,
            dtype=np.float64,
        )
        if self.chords.shape != (len(arcs), 2) or not np.all(np.isfinite(self.chords)):
            raise ValueError(
                "Prescribed chord vertices must be finite and aligned with the arc partition."
            )
        endpoint_error = 0.0
        for vertex, arc in zip(self.chords, arcs, strict=True):
            curve = curves_[arc[0]]
            endpoint_box = np.asarray(curve.enclosure(arc[1], arc[1]))
            endpoint_error = max(
                endpoint_error,
                float(np.max(np.nextafter(np.abs(endpoint_box - vertex), np.inf))),
            )
        self.chord_endpoint_error = endpoint_error
        self.arc_curves = np.asarray([arc[0] for arc in arcs], dtype=np.int32)
        self.arc_first = np.asarray([arc[1] for arc in arcs], dtype=np.float64)
        self.arc_last = np.asarray([arc[2] for arc in arcs], dtype=np.float64)
        boxes = np.asarray([arc[4] for arc in arcs], dtype=np.float64)
        following_chord = np.roll(self.chords, -1, axis=0)
        self.arc_lower = np.minimum(boxes[:, 0], np.minimum(self.chords, following_chord))
        self.arc_upper = np.maximum(boxes[:, 1], np.maximum(self.chords, following_chord))
        following = np.roll(np.asarray(starts), -1, axis=0)
        junction_boxes = []
        for index, curve in enumerate(curves_):
            next_curve = curves_[(index + 1) % len(curves_)]
            end = curve.parameter_interval[1]
            start = next_curve.parameter_interval[0]
            first_box = np.asarray(curve.enclosure(end, end))
            second_box = np.asarray(next_curve.enclosure(start, start))
            junction_boxes.append(
                np.stack(
                    (
                        np.minimum(first_box[0], second_box[0]),
                        np.maximum(first_box[1], second_box[1]),
                    )
                )
            )
        junction_boxes_ = np.asarray(junction_boxes)
        self.junction_lower = np.nextafter(
            np.minimum(junction_boxes_[:, 0], np.minimum(ends, following)), -np.inf
        )
        self.junction_upper = np.nextafter(
            np.maximum(junction_boxes_[:, 1], np.maximum(ends, following)), np.inf
        )
        self.tolerance = tolerance_
        self.relative_closure_tolerance = relative_tolerance
        self.closure_tolerance = closure_tolerance
        self.closure_gap = closure_gap

    @property
    def identity(self) -> dict[str, object]:
        return {
            "kind": "curves",
            "curve_types": [type(curve).__name__ for curve in self.curves],
            "curves": array_tree_fingerprint(self.curves),
            "tolerance": self.tolerance,
            "relative_closure_tolerance": self.relative_closure_tolerance,
            "closure_tolerance": self.closure_tolerance,
            "closure_gap": self.closure_gap,
            "arc_partition": array_tree_fingerprint(
                (self.arc_curves, self.arc_first, self.arc_last)
            ),
            "chords": array_tree_fingerprint(self.chords),
        }

    def certify_topology(
        self, /, *, maximum_pairs: int = 200_000
    ) -> TrimTopologyEvidence:
        """Prove source/chord isotopy for this bounded chord cover.

        Source arcs and adjacent pairs must be monotone along a separating
        projection, and nonadjacent homotopy boxes must be disjoint. Ambiguous
        covers require a tighter caller-selected cover, not a sampled topology
        decision. Source joins are either exact or remain inside the explicit
        directed closure bound retained by this loop.
        """
        if (
            isinstance(maximum_pairs, bool)
            or not isinstance(maximum_pairs, int)
            or maximum_pairs < 1
        ):
            raise ValueError("maximum_pairs must be a positive integer.")
        unresolved: set[tuple[int, int]] = set()
        work = 0
        exhausted = False
        count = self.chords.shape[0]

        def positive_projection(
            bounds: tuple[np.ndarray, np.ndarray], direction: np.ndarray
        ) -> bool:
            if not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(direction)):
                return False
            terms = np.minimum(bounds[0] * direction, bounds[1] * direction)
            margin = 16 * np.finfo(np.float64).eps * np.sum(np.abs(terms))
            return bool(np.sum(terms) - margin > 0)

        derivative_bounds = []
        for arc in range(count):
            curve = self.curves[int(self.arc_curves[arc])]
            try:
                bounds = curve.derivative_bounds(
                    float(self.arc_first[arc]), float(self.arc_last[arc])
                )
            except (NotImplementedError, ValueError):
                bounds = (np.full(2, -np.inf), np.full(2, np.inf))
            derivative_bounds.append(bounds)
            direction = self.chords[(arc + 1) % count] - self.chords[arc]
            if not positive_projection(bounds, direction):
                unresolved.add((arc, arc))
        for arc in range(count):
            following = (arc + 1) % count
            first_curve = int(self.arc_curves[arc])
            second_curve = int(self.arc_curves[following])
            if first_curve != second_curve or following == 0:
                if not self.curves[first_curve].shares_endpoint(
                    self.curves[second_curve]
                ):
                    first_parameter = self.curves[first_curve].parameter_interval[1]
                    second_parameter = self.curves[second_curve].parameter_interval[0]
                    endpoints = np.asarray(
                        (
                            self.curves[first_curve].evaluate(
                                jnp.asarray(first_parameter, dtype=jnp.float64)
                            ),
                            self.curves[second_curve].evaluate(
                                jnp.asarray(second_parameter, dtype=jnp.float64)
                            ),
                        )
                    )
                    if (
                        float(np.max(np.abs(endpoints[0] - endpoints[1])))
                        > self.closure_tolerance
                    ):
                        unresolved.add((arc, following))
            direction = self.chords[(arc + 2) % count] - self.chords[arc]
            chord_a = self.chords[following] - self.chords[arc]
            chord_b = self.chords[(arc + 2) % count] - self.chords[following]
            if not (
                positive_projection(derivative_bounds[arc], direction)
                and positive_projection(derivative_bounds[following], direction)
                and positive_projection((chord_a, chord_a), direction)
                and positive_projection((chord_b, chord_b), direction)
            ):
                unresolved.add((arc, following))

        # Bucketed boxes rather than a quadratic candidate-pair matrix.
        pairs, work = overlapping_box_pairs(self.arc_lower, self.arc_upper, maximum_pairs)
        if pairs is None:
            exhausted = True
        else:
            for arc, other in pairs.tolist():
                if (arc - other) % count not in (1, count - 1):
                    unresolved.add((arc, other))
        return TrimTopologyEvidence(
            not unresolved and not exhausted,
            tuple(sorted(unresolved)),
            min(work, maximum_pairs),
            exhausted,
            canonical_fingerprint(self.identity),
            canonical_fingerprint(array_tree_fingerprint(self.chords)),
        )


@eqx.filter_jit
def _trim_points(curve: AbstractTrimCurve, parameters: Array, /) -> Array:
    """One stable lowering; original curve numerical leaves remain dynamic."""
    return curve.evaluate(parameters)


def _arc_record(
    curve: AbstractTrimCurve, first: float, last: float, start: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Enclosure of one arc, widened to contain its evaluated endpoints."""
    box = np.asarray(curve.enclosure(first, last), dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)):
        raise ValueError("Trim curve enclosures must be finite (2, 2) boxes.")
    end = np.asarray(_trim_points(curve, jnp.asarray(last, dtype=jnp.float64)))
    lower = np.minimum(box[0], np.minimum(start, end))
    upper = np.maximum(box[1], np.maximum(start, end))
    return np.stack((lower, upper)), end


def _certified_arcs(
    curve: AbstractTrimCurve,
    index: int,
    first: float,
    last: float,
    tolerance: float,
    capacity: int,
    /,
) -> list[tuple[int, float, float, np.ndarray, np.ndarray]]:
    start = np.asarray(_trim_points(curve, jnp.asarray(first, dtype=jnp.float64)))
    pending = [(first, last, start)]
    accepted: list[tuple[int, float, float, np.ndarray, np.ndarray]] = []
    while pending:
        lower, upper, point = pending.pop()
        box, _ = _arc_record(curve, lower, upper, point)
        if np.linalg.norm(box[1] - box[0]) <= tolerance:
            accepted.append((index, lower, upper, point, box))
            if len(accepted) > capacity:
                raise ValueError("Curve trim loop exceeds its arc capacity.")
            continue
        middle = 0.5 * (lower + upper)
        if not lower < middle < upper:
            raise ValueError("Curve trim enclosure cannot reach the requested tolerance.")
        middle_point = np.asarray(_trim_points(curve, jnp.asarray(middle, jnp.float64)))
        pending.append((middle, upper, middle_point))
        pending.append((lower, middle, point))
    return accepted


type TrimLoop = PolygonTrimLoop | CurveTrimLoop


def _as_loop(value: TrimLoop | npt.ArrayLike, /) -> TrimLoop:
    if isinstance(value, (PolygonTrimLoop, CurveTrimLoop)):
        return value
    return PolygonTrimLoop(value)


class TrimBoxClassification(StrictModule):
    """Whole-box trim decisions; overlap/limits are explicit unresolved rows."""

    __strict_contract__ = True

    inside: HostBool[_TrimPointDim]
    resolved: HostBool[_TrimPointDim]
    refinements: HostInt32[_TrimPointDim]

    def __init__(
        self, inside: npt.ArrayLike, resolved: npt.ArrayLike, refinements: npt.ArrayLike
    ) -> None:
        scope = Scope()
        self.inside = parse(
            np.asarray(inside, dtype=np.bool_),
            HostBool[_TrimPointDim],
            "inside",
            scope=scope,
        )
        self.resolved = parse(
            np.asarray(resolved, dtype=np.bool_),
            HostBool[_TrimPointDim],
            "resolved",
            scope=scope,
        )
        self.refinements = parse(
            np.asarray(refinements, dtype=np.int32),
            HostInt32[_TrimPointDim],
            "refinements",
            scope=scope,
        )


class TrimClassification(StrictModule):
    """Exact point-in-trim decisions with per-point evidence.

    ``inside`` is meaningful where ``resolved`` holds. ``boundary`` marks points
    exactly on a polygon edge or within the refined enclosure of a curve arc at
    the refinement budget; such points are unresolved unless they lie exactly on
    an affine edge. ``winding`` holds the exact chord winding number per loop.
    """

    __strict_contract__ = True

    inside: HostBool[_TrimPointDim]
    boundary: HostBool[_TrimPointDim]
    resolved: HostBool[_TrimPointDim]
    winding: HostInt32[_TrimPointDim, _TrimLoopDim]
    refinements: HostInt32[_TrimPointDim]

    def __init__(
        self,
        *,
        inside: npt.ArrayLike,
        boundary: npt.ArrayLike,
        resolved: npt.ArrayLike,
        winding: npt.ArrayLike,
        refinements: npt.ArrayLike,
    ) -> None:
        scope = Scope()
        self.inside = parse(
            np.asarray(inside, dtype=np.bool_),
            HostBool[_TrimPointDim],
            "inside",
            scope=scope,
        )
        self.boundary = parse(
            np.asarray(boundary, dtype=np.bool_),
            HostBool[_TrimPointDim],
            "boundary",
            scope=scope,
        )
        self.resolved = parse(
            np.asarray(resolved, dtype=np.bool_),
            HostBool[_TrimPointDim],
            "resolved",
            scope=scope,
        )
        self.winding = parse(
            np.asarray(winding, dtype=np.int32),
            HostInt32[_TrimPointDim, _TrimLoopDim],
            "winding",
            scope=scope,
        )
        self.refinements = parse(
            np.asarray(refinements, dtype=np.int32),
            HostInt32[_TrimPointDim],
            "refinements",
            scope=scope,
        )


def _exact_winding(
    point: np.ndarray, chords: np.ndarray, mode: PredicateMode, /
) -> tuple[int, bool]:
    """Exact winding number of a closed chord polygon and on-edge detection."""
    start = chords
    end = np.roll(chords, -1, axis=0)
    signs = np.asarray(
        orient2d(start, end, np.broadcast_to(point, start.shape), mode=mode).signs,
        dtype=np.int8,
    )
    upward = (start[:, 1] <= point[1]) & (end[:, 1] > point[1])
    downward = (end[:, 1] <= point[1]) & (start[:, 1] > point[1])
    winding = int(np.sum(upward & (signs > 0)) - np.sum(downward & (signs < 0)))
    within = ((np.minimum(start, end) <= point) & (point <= np.maximum(start, end))).all(
        axis=1
    )
    return winding, bool(np.any((signs == 0) & within))


def _refined_chords(
    loop: CurveTrimLoop, point: np.ndarray, maximum_depth: int, /
) -> tuple[np.ndarray, bool, int]:
    """Chord polygon whose arc enclosures exclude ``point`` where resolvable."""
    chords: list[np.ndarray] = []
    # Numerical endpoint agreement is not exact endpoint equality. Never make
    # an exact membership decision inside the junction uncertainty boxes.
    unresolved = bool(
        np.any(
            np.all(
                (point >= loop.junction_lower) & (point <= loop.junction_upper), axis=1
            )
        )
    )
    refinements = 0
    for arc in range(loop.chords.shape[0]):
        lower, upper = loop.arc_lower[arc], loop.arc_upper[arc]
        if np.any(point < lower) or np.any(point > upper):
            chords.append(loop.source_chords[arc])
            continue
        curve = loop.curves[int(loop.arc_curves[arc])]
        stack = [(float(loop.arc_first[arc]), float(loop.arc_last[arc]), 0)]
        start = loop.source_chords[arc]
        pieces: list[np.ndarray] = []
        while stack:
            first, last, depth = stack.pop()
            box, _ = _arc_record(curve, first, last, start)
            outside = np.any(point < box[0]) or np.any(point > box[1])
            middle = 0.5 * (first + last)
            if outside or depth >= maximum_depth or not first < middle < last:
                unresolved |= not outside
                pieces.append(start)
                start = np.asarray(_trim_points(curve, jnp.asarray(last, jnp.float64)))
                continue
            refinements += 1
            stack.append((middle, last, depth + 1))
            stack.append((first, middle, depth + 1))
        chords.extend(pieces)
    return np.asarray(chords, dtype=np.float64), unresolved, refinements


class TrimDomain(StrictModule):
    """Oriented trim domain in a two-dimensional chart.

    Loops are explicit affine polygons (`PolygonTrimLoop`; array inputs are
    converted once) or chains of exact curves (`CurveTrimLoop`). A point is
    inside when it has nonzero winding about the outer loop and zero winding
    about every hole.
    """

    outer: TrimLoop
    holes: tuple[TrimLoop, ...]
    trim_id: str = eqx.field(static=True)

    def __init__(
        self,
        outer: TrimLoop | npt.ArrayLike,
        holes: Sequence[TrimLoop | npt.ArrayLike] = (),
    ) -> None:
        outer_ = _as_loop(outer)
        holes_ = tuple(_as_loop(hole) for hole in holes)
        self.outer = outer_
        self.holes = holes_
        self.trim_id = canonical_fingerprint(
            {
                "kind": "trim-domain",
                "outer": outer_.identity,
                "holes": [hole.identity for hole in holes_],
            }
        )

    @property
    def loops(self) -> tuple[TrimLoop, ...]:
        return (self.outer, *self.holes)

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
        """Traceable chord-cover membership.

        Exact for polygon loops; for curve loops it agrees with the exact curves
        outside each loop's certified approximation band (`classify` resolves
        points inside the band).
        """
        reference_ = jnp.asarray(reference, dtype=jnp.float64)
        inside = self._inside_loop(reference_, jnp.asarray(self.outer.chords))
        for hole in self.holes:
            inside &= ~self._inside_loop(reference_, jnp.asarray(hole.chords))
        return inside

    def in_band(self, reference: Array, /) -> Array:
        """Whether points lie in any curve loop's approximation band."""
        reference_ = jnp.asarray(reference, dtype=jnp.float64)
        band = jnp.zeros(reference_.shape[:-1], dtype=jnp.bool_)
        for loop in self.loops:
            if isinstance(loop, CurveTrimLoop):
                lower = jnp.concatenate(
                    (jnp.asarray(loop.arc_lower), jnp.asarray(loop.junction_lower))
                )
                upper = jnp.concatenate(
                    (jnp.asarray(loop.arc_upper), jnp.asarray(loop.junction_upper))
                )
                point = reference_[..., None, :]
                band |= jnp.any(
                    jnp.all((point >= lower) & (point <= upper), axis=-1), axis=-1
                )
        return band

    def classify_boxes(
        self,
        reference: npt.ArrayLike,
        /,
        *,
        maximum_depth: int = 48,
        maximum_refinements: int = 100_000,
    ) -> TrimBoxClassification:
        """Classify a complete closed rectangle by excluding its source boundary.

        A winding decision at the center is used only AFTER source arc/junction
        enclosures prove no boundary meets any point of that rectangle.
        """
        boxes = np.asarray(reference, dtype=np.float64)
        if (
            boxes.ndim != 3
            or boxes.shape[1:] != (2, 2)
            or not np.all(np.isfinite(boxes))
            or np.any(boxes[:, 0] > boxes[:, 1])
        ):
            raise ValueError(
                "Trim box queries require finite (num_boxes,2,2) lower/upper rectangles."
            )
        if maximum_depth < 0 or maximum_refinements < 0:
            raise ValueError("Trim refinement budgets must be nonnegative.")
        inside = np.zeros(boxes.shape[0], dtype=np.bool_)
        resolved = np.ones(boxes.shape[0], dtype=np.bool_)
        work = np.zeros(boxes.shape[0], dtype=np.int32)
        used = 0

        def disjoint(a: np.ndarray, b: np.ndarray) -> bool:
            return bool(np.any(a[1] < b[0]) or np.any(b[1] < a[0]))

        for row in range(boxes.shape[0]):
            query = np.asarray(boxes[row], dtype=np.float64)
            windings = []
            for loop in self.loops:
                if isinstance(loop, PolygonTrimLoop):
                    vertices = loop.vertices
                    corners = np.asarray(
                        (
                            query[0],
                            (query[1, 0], query[0, 1]),
                            query[1],
                            (query[0, 0], query[1, 1]),
                        )
                    )
                    for start, end in zip(
                        vertices, np.roll(vertices, -1, axis=0), strict=True
                    ):
                        edge_box = np.stack(
                            (np.minimum(start, end), np.maximum(start, end))
                        )
                        if disjoint(query, edge_box):
                            continue
                        signs = np.asarray(
                            orient2d(
                                np.broadcast_to(start, corners.shape),
                                np.broadcast_to(end, corners.shape),
                                corners,
                                mode=PredicateMode.EXACT,
                            ).signs
                        )
                        if not (np.all(signs > 0) or np.all(signs < 0)):
                            resolved[row] = False
                            break
                    chords = vertices
                else:
                    chords_list = []
                    for lower, upper in zip(
                        loop.junction_lower, loop.junction_upper, strict=True
                    ):
                        if not disjoint(query, np.stack((lower, upper))):
                            resolved[row] = False
                            break
                    for arc in range(loop.source_chords.shape[0]):
                        curve = loop.curves[int(loop.arc_curves[arc])]
                        pending = [
                            (float(loop.arc_first[arc]), float(loop.arc_last[arc]), 0)
                        ]
                        while pending:
                            first, last, depth = pending.pop()
                            box = np.asarray(curve.enclosure(first, last))
                            if disjoint(query, box):
                                chords_list.append(
                                    np.asarray(curve.evaluate(jnp.asarray(first)))
                                )
                                continue
                            middle = 0.5 * (first + last)
                            if (
                                depth >= maximum_depth
                                or used >= maximum_refinements
                                or not first < middle < last
                            ):
                                resolved[row] = False
                                break
                            used += 1
                            work[row] += 1
                            pending.append((middle, last, depth + 1))
                            pending.append((first, middle, depth + 1))
                        if not resolved[row]:
                            break
                    chords = np.asarray(chords_list)
                if not resolved[row]:
                    break
                winding, _ = _exact_winding(
                    0.5 * (query[0] + query[1]), chords, PredicateMode.EXACT
                )
                windings.append(winding)
            if resolved[row]:
                inside[row] = windings[0] != 0 and all(
                    value == 0 for value in windings[1:]
                )
        return TrimBoxClassification(inside, resolved, work)

    def classify(
        self,
        reference: npt.ArrayLike,
        /,
        *,
        mode: PredicateMode = PredicateMode.EXACT,
        maximum_depth: int = 48,
    ) -> TrimClassification:
        """Exact host classification with bounded refinement of curve loops."""
        mode_ = resolve_host_predicate_mode(mode)
        points = np.asarray(reference, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("reference must have shape (num_points, 2).")
        if not np.all(np.isfinite(points)):
            raise ValueError("Trim classification reference points must be finite.")
        if maximum_depth < 0:
            raise ValueError("maximum_depth must be nonnegative.")
        loops = self.loops
        winding = np.zeros((points.shape[0], len(loops)), dtype=np.int32)
        boundary = np.zeros((points.shape[0],), dtype=np.bool_)
        resolved = np.ones((points.shape[0],), dtype=np.bool_)
        refinements = np.zeros((points.shape[0],), dtype=np.int32)
        for row in range(points.shape[0]):
            point = np.asarray(points[row], dtype=np.float64)
            for column, loop in enumerate(loops):
                if isinstance(loop, CurveTrimLoop):
                    chords, unresolved, count = _refined_chords(
                        loop, point, maximum_depth
                    )
                    refinements[row] += count
                    if unresolved:
                        boundary[row] = True
                        resolved[row] = False
                else:
                    chords = loop.vertices
                value, on_edge = _exact_winding(point, chords, mode_)
                winding[row, column] = value
                if on_edge and isinstance(loop, PolygonTrimLoop):
                    boundary[row] = True
        inside = (winding[:, 0] != 0) & np.all(winding[:, 1:] == 0, axis=1)
        return TrimClassification(
            inside=inside & ~boundary,
            boundary=boundary,
            resolved=resolved,
            winding=winding,
            refinements=refinements,
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

    def classify_reference(
        self,
        chart_indices: npt.ArrayLike,
        reference: npt.ArrayLike,
        /,
        *,
        mode: PredicateMode = PredicateMode.EXACT,
        maximum_depth: int = 48,
    ) -> TrimClassification:
        """Exact host trim classification of flat ``(num_points,)`` chart points.

        Untrimmed charts classify every point as inside and resolved.
        """
        indices = np.asarray(chart_indices, dtype=np.int32).reshape((-1,))
        points = np.asarray(reference, dtype=np.float64)
        if self.reference_dimension != 2:
            raise ValueError("Trim classification requires two-dimensional charts.")
        if points.shape != (indices.shape[0], 2):
            raise ValueError("reference must have shape (num_points, 2).")
        if np.any((indices < 0) | (indices >= self.num_charts)):
            raise ValueError("chart_indices lie outside the atlas.")
        loop_count = max(
            (1 + len(trim.holes) for trim in self.trim_domains if trim is not None),
            default=1,
        )
        count = indices.shape[0]
        inside = np.ones((count,), dtype=np.bool_)
        boundary = np.zeros((count,), dtype=np.bool_)
        resolved = np.ones((count,), dtype=np.bool_)
        winding = np.zeros((count, loop_count), dtype=np.int32)
        refinements = np.zeros((count,), dtype=np.int32)
        for chart in np.unique(indices):
            trim = self.trim_domains[int(chart)]
            if trim is None:
                continue
            rows = np.flatnonzero(indices == chart)
            result = trim.classify(points[rows], mode=mode, maximum_depth=maximum_depth)
            inside[rows] = result.inside
            boundary[rows] = result.boundary
            resolved[rows] = result.resolved
            winding[rows, : result.winding.shape[1]] = result.winding
            refinements[rows] = result.refinements
        return TrimClassification(
            inside=inside,
            boundary=boundary,
            resolved=resolved,
            winding=winding,
            refinements=refinements,
        )

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
    "AbstractTrimCurve",
    "CurveTrimLoop",
    "TrimBoxClassification",
    "TrimTopologyEvidence",
    "PolygonTrimLoop",
    "TrimClassification",
    "TrimDomain",
    "TrimLoop",
    "BoundaryAtlas",
    "BoundaryFrame",
    "BoundaryMap",
]
