#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

import phydrax

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    orient2d_host,
    PredicateMode,
    resolve_host_predicate_mode,
)
from .._strict import StrictModule
from .._validation import (
    canonical_identifier,
    nonnegative_finite_float,
    positive_finite_float,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..typing import checked, Dim, HostBool, HostFloat64, HostInt32, parse, Scope
from ._planar_primitives import rational_points, reserve_fraction_work
from .brep._carrier_pieces import (
    AffineTrimChordEvidence,
    carrier_payload,
    child_coefficient_enclosures,
    CoefficientEnclosureSource,
)
from .brep._coordinate_request import coordinate_source_request
from .brep._request_preparations import TrimRequestPreparations


if TYPE_CHECKING:
    from .brep._patches import BernsteinRestrictionBank, MeridianOverlaps


@contextmanager
@checked
def _trim_coordinate_request(
    inputs: tuple[object, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
    preparations: TrimRequestPreparations | None,
) -> Iterator[tuple[CoordinateEnclosureBudget, TrimRequestPreparations]]:
    request = TrimRequestPreparations() if preparations is None else preparations
    with coordinate_source_request(inputs, budget=budget) as owner:
        yield owner, request


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


class AbstractTrimCurve(StrictModule, CoefficientEnclosureSource):
    """Represent an oriented source curve in a two-dimensional chart.

    Trim consumers use two capabilities: pointwise evaluation over
    ``parameter_interval`` and a conservative axis-aligned enclosure of any
    sub-arc. The enclosure contains the arc and therefore its chord, which makes
    chord winding agree with source winding for every point outside the enclosure.
    """

    @property
    @abstractmethod
    def parameter_interval(self) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        """Chart coordinates ``(..., 2)`` at curve parameters ``(...)``.

        Eager request consumers supply their resource owner and preparations.
        With no supplied context, the curve retains its standalone evaluation
        behavior; traced point kernels do not capture a host resource owner.
        """
        raise NotImplementedError

    @abstractmethod
    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        """Conservative ``(2, 2)`` box ``[lower, upper]`` of the arc ``[first, last]``."""
        raise NotImplementedError

    def enclosures(
        self,
        intervals: npt.ArrayLike,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        """Conservative ``(N, 2, 2)`` boxes of independent ``(first, last)`` arcs.

        Row ``k`` is ``enclosure(*intervals[k])``; carriers may share one
        evaluation pass between rows without changing any row's enclosure.
        """
        rows = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
        if rows.shape[0] == 0:
            return np.empty((0, 2, 2), dtype=np.float64)
        return np.stack(
            [
                np.asarray(
                    self.enclosure(
                        first, last, budget=budget, preparations=preparations
                    ),
                    dtype=np.float64,
                )
                for first, last in rows.tolist()
            ]
        )

    def arc_partition_binding(self) -> tuple[tuple[object, ...], bool] | None:
        """Source binding shared by both orientations of one ``[0, 1]`` trim.

        ``(sources, reversed)`` states that every trim with identical sources
        traverses one point set, local ``s`` of one orientation being ``1 - s``
        of the other. Carriers without such a proof return None.
        """
        return None

    @checked
    def affine_chord_evidence(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> AffineTrimChordEvidence | None:
        """Return source-bound affine endpoint evidence when nominally proved.

        The base capability leaves affinity unestablished. Source owners
        provide rational coefficients and certified endpoint-root bounds;
        point samples or a straight-looking enclosure establish no such proof.
        """
        return None

    def host_points(
        self,
        parameters: ArrayLike,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        """Evaluate concrete parameters through one compiled program, on the host.

        Carriers whose own evaluation already runs a shape-stable compiled
        program override this to avoid a per-carrier outer compilation, and
        may reuse numerical inputs prepared in an explicit request
        (``preparations`` with its ``budget``). This generic program retains
        nothing in the request.
        """
        return np.asarray(jax.device_get(_trim_points(self, parameters)))

    def derivative_bounds(
        self,
        first: float,
        last: float,
        /,
        *,
        order: int = 1,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conservative oriented jets; unsupported capability is explicit."""
        if budget is None or preparations is None:
            with _trim_coordinate_request(
                (
                    self,
                    first,
                    last,
                    order,
                ),
                budget=budget,
                preparations=preparations,
            ) as (budget, preparations):
                return self.derivative_bounds(
                    first,
                    last,
                    order=order,
                    budget=budget,
                    preparations=preparations,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
        raise NotImplementedError(
            f"AbstractTrimCurve.derivative_bounds must be implemented for this carrier; got {type(self).__qualname__}"
        )

    @checked
    def shares_endpoint(
        self,
        other: AbstractTrimCurve,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        """Return whether this end and the next start share one source parameter."""
        return False

    @checked
    def is_c1_on(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> bool:
        """Return whether first-derivative continuity is certified on the range.

        The base capability leaves continuity unestablished. Source carriers
        override it when their parameterization supplies continuity evidence.
        """
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

    def native_period_symbol(self) -> str | None:
        """Return a certified analytic period symbol when the carrier supplies one."""
        return None

    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return construction intervals retained by descendant coefficient sources."""
        return child_coefficient_enclosures(self)

    def validate_range(self, first: float, last: float, /) -> tuple[float, float]:
        lower, upper = self.parameter_interval
        if not (
            np.isfinite(first) and np.isfinite(last) and lower <= first < last <= upper
        ):
            raise ValueError(
                f"AbstractTrimCurve.range must lie inside {self.parameter_interval!r}; got ({first!r}, {last!r})"
            )
        return float(first), float(last)

    @checked
    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        if budget is None or preparations is None:
            with _trim_coordinate_request(
                (
                    self,
                    first,
                    last,
                ),
                budget=budget,
                preparations=preparations,
            ) as (budget, preparations):
                return self.bounding_box(
                    first, last, budget=budget, preparations=preparations
                )
        self.validate_range(first, last)
        return self.enclosure(first, last, budget=budget, preparations=preparations)


class _LoopVertexDim(Dim):
    """Vertices of a polygonal trim loop or of a certified chord cover."""


class _ArcDim(Dim):
    """Certified arcs of a curve trim loop."""


class _TrimPointDim(Dim):
    """Classified chart points."""


class _TrimLoopDim(Dim):
    """Loops of one trim domain (outer first, then holes)."""


class TrimTopologyEvidence(StrictModule):
    """Record source/chord isotopy evidence bound to the loop and chord polygon."""

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
            raise ValueError(
                f"PolygonTrimLoop.vertices must have shape (num_points >= 3, 2); got {host.shape}"
            )
        if not np.all(np.isfinite(host)):
            raise ValueError(
                "PolygonTrimLoop.vertices must be finite; got nonfinite coordinates"
            )
        host = np.array(host, dtype=np.float64)
        host.setflags(write=False)
        self.vertices = host

    @property
    def chords(self) -> np.ndarray:
        return self.vertices

    @property
    def identity(self) -> dict[str, object]:
        return {"kind": "polygon", "vertices": array_tree_fingerprint(self.vertices)}


class _LoopClosure(StrictModule):
    """Evaluated curve joins and their retained closure bound."""

    starts: tuple[np.ndarray, ...]
    ends: tuple[np.ndarray, ...]
    gap: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


def _prepare_loop_closure(
    curves: tuple[AbstractTrimCurve, ...],
    relative_closure_tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> _LoopClosure:
    starts = []
    ends = []
    for curve in curves:
        first, last = curve.parameter_interval
        points = np.asarray(
            jax.device_get(
                curve.evaluate(
                    jnp.asarray((first, last), dtype=jnp.float64),
                    budget=budget,
                    preparations=preparations,
                )
            ),
            dtype=np.float64,
        )
        starts.append(points[0])
        ends.append(points[1])
    gaps = [
        float(np.max(np.abs(ends[index] - starts[(index + 1) % len(curves)])))
        for index in range(len(curves))
    ]
    closure_gap = max(gaps)
    maximum_endpoint = float(np.max(np.abs(np.asarray((*starts, *ends)))))
    roundoff_tolerance = float(
        128.0 * np.finfo(np.float64).eps * (1.0 + maximum_endpoint)
    )
    relative_tolerance = nonnegative_finite_float(
        relative_closure_tolerance, "CurveTrimLoop.relative_closure_tolerance"
    )
    closure_tolerance = max(
        roundoff_tolerance, relative_tolerance * max(1.0, maximum_endpoint)
    )
    exact_joins = [
        curve.shares_endpoint(
            curves[(index + 1) % len(curves)], budget=budget, preparations=preparations
        )
        for index, curve in enumerate(curves)
    ]
    if any(
        gap > closure_tolerance and not exact
        for gap, exact in zip(gaps, exact_joins, strict=True)
    ):
        raise ValueError(
            f"CurveTrimLoop.curves must form a closed loop; got gap {closure_gap:.3e}"
        )
    return _LoopClosure(
        tuple(starts), tuple(ends), closure_gap, relative_tolerance, closure_tolerance
    )


def _prepare_arc_partition(
    curves: tuple[AbstractTrimCurve, ...],
    tolerance: float,
    maximum_arcs: int,
    arc_parameters: Sequence[npt.ArrayLike] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> _ArcBatch:
    arcs: list[tuple[int, float, float, np.ndarray, np.ndarray]] = []
    if arc_parameters is not None and len(arc_parameters) != len(curves):
        raise ValueError(
            f"CurveTrimLoop.arc_parameters must provide one ordered partition per curve; got {len(arc_parameters)} partitions for {len(curves)} curves"
        )
    for index, curve in enumerate(curves):
        first, last = curve.parameter_interval
        if arc_parameters is None:
            arcs.extend(
                _certified_arcs(
                    curve,
                    index,
                    first,
                    last,
                    tolerance,
                    maximum_arcs - len(arcs),
                    budget=budget,
                    preparations=preparations,
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
                f"CurveTrimLoop.arc_parameters must partition each full curve interval in order; got curve {index} partition {partition!r}"
            )
        if len(arcs) + partition.size - 1 > maximum_arcs:
            raise ValueError(
                f"CurveTrimLoop.arc_parameters must fit the arc capacity; got {len(arcs) + partition.size - 1} > {maximum_arcs}"
            )
        points = np.asarray(
            jax.device_get(
                curve.evaluate(
                    jnp.asarray(partition), budget=budget, preparations=preparations
                )
            )
        )
        for lower, upper, start in zip(
            partition[:-1], partition[1:], points[:-1], strict=True
        ):
            box, end = _arc_enclosure(
                curve,
                float(lower),
                float(upper),
                start,
                budget=budget,
                preparations=preparations,
            )
            if np.linalg.norm(
                box[1] - box[0]
            ) > tolerance and not _affine_chord_within_tolerance(
                curve,
                float(lower),
                float(upper),
                start,
                end,
                tolerance,
                budget=budget,
                preparations=preparations,
            ):
                raise ValueError(
                    f"CurveTrimLoop.arc_parameters must meet the cover tolerance; got curve {index} arc [{lower}, {upper}]"
                )
            arcs.append((index, float(lower), float(upper), start, box))
    return _arc_record(arcs)


class _LoopGeometry(StrictModule):
    """Chord endpoint error and the arc/junction uncertainty boxes."""

    endpoint_error: float = eqx.field(static=True)
    arc_lower: np.ndarray
    arc_upper: np.ndarray
    junction_lower: np.ndarray
    junction_upper: np.ndarray


def _prepare_loop_geometry(
    curves: tuple[AbstractTrimCurve, ...],
    batch: _ArcBatch,
    chords: np.ndarray,
    closure: _LoopClosure,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> _LoopGeometry:
    endpoint_error = 0.0
    curve_indices = np.asarray(batch.curves, dtype=np.int64)
    endpoint_boxes = np.empty((chords.shape[0], 2, 2), dtype=np.float64)
    for index, curve in enumerate(curves):
        rows = np.flatnonzero(curve_indices == index)
        if rows.size:
            endpoint_boxes[rows] = _point_enclosures(
                curve,
                batch.first[rows].tolist(),
                budget=budget,
                preparations=preparations,
            )
    for vertex, endpoint_box in zip(chords, endpoint_boxes, strict=True):
        endpoint_error = max(
            endpoint_error,
            float(np.max(np.nextafter(np.abs(endpoint_box - vertex), np.inf))),
        )
    following_chord = np.roll(chords, -1, axis=0)
    arc_lower = np.minimum(batch.boxes[:, 0], np.minimum(chords, following_chord))
    arc_upper = np.maximum(batch.boxes[:, 1], np.maximum(chords, following_chord))
    following = np.roll(np.asarray(closure.starts), -1, axis=0)
    junction_boxes = []
    for index, curve in enumerate(curves):
        next_curve = curves[(index + 1) % len(curves)]
        end = curve.parameter_interval[1]
        start = next_curve.parameter_interval[0]
        first_box = _point_enclosures(
            curve, [end], budget=budget, preparations=preparations
        )[0]
        second_box = _point_enclosures(
            next_curve, [start], budget=budget, preparations=preparations
        )[0]
        junction_boxes.append(
            np.stack(
                (
                    np.minimum(first_box[0], second_box[0]),
                    np.maximum(first_box[1], second_box[1]),
                )
            )
        )
    junction_boxes_ = np.asarray(junction_boxes)
    return _LoopGeometry(
        endpoint_error,
        arc_lower,
        arc_upper,
        np.nextafter(
            np.minimum(junction_boxes_[:, 0], np.minimum(closure.ends, following)),
            -np.inf,
        ),
        np.nextafter(
            np.maximum(junction_boxes_[:, 1], np.maximum(closure.ends, following)), np.inf
        ),
    )


class CurveTrimLoop(StrictModule):
    """Oriented trim curves with a certified chord cover and bounded joins.

    Each curve is split until its arc enclosure diagonal is at most ``tolerance``,
    or a source-bound affine certificate bounds chord displacement by that
    tolerance, including stored endpoint rounding and certified root uncertainty.
    The chord polygon classifies chart points outside the union of arc enclosures
    using exact predicates; `TrimDomain.classify` resolves points inside that band by refining
    source curves. By default joins retain only directed-roundoff uncertainty.
    A positive ``relative_closure_tolerance`` retains a validated source-domain
    coincidence bound; this bound remains separate from source endpoint identity.
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

    @checked
    def __init__(
        self,
        curves: Sequence[AbstractTrimCurve],
        *,
        tolerance: float,
        maximum_arcs: int = 65536,
        arc_parameters: Sequence[npt.ArrayLike] | None = None,
        chord_vertices: npt.ArrayLike | None = None,
        relative_closure_tolerance: float = 0.0,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> None:
        with _trim_coordinate_request(
            (
                curves,
                tolerance,
                maximum_arcs,
                arc_parameters,
                chord_vertices,
                relative_closure_tolerance,
            ),
            budget=budget,
            preparations=preparations,
        ) as (budget, preparations):
            curves_ = tuple(curves)
            if not curves_ or any(
                not isinstance(curve, AbstractTrimCurve) for curve in curves_
            ):
                raise TypeError(
                    f"CurveTrimLoop.curves must contain AbstractTrimCurve values; got {curves_!r}"
                )
            tolerance_ = positive_finite_float(tolerance, "CurveTrimLoop.tolerance")
            if maximum_arcs < len(curves_):
                raise ValueError(
                    f"CurveTrimLoop.maximum_arcs must be at least the curve count; got {maximum_arcs} < {len(curves_)}"
                )
            closure = _prepare_loop_closure(
                curves_,
                relative_closure_tolerance,
                budget=budget,
                preparations=preparations,
            )
            batch = _prepare_arc_partition(
                curves_,
                tolerance_,
                maximum_arcs,
                arc_parameters,
                budget=budget,
                preparations=preparations,
            )
            if chord_vertices is not None and arc_parameters is None:
                raise ValueError(
                    "CurveTrimLoop.chord_vertices must have prescribed source arc partitions; got arc_parameters=None"
                )
            chords = np.asarray(
                batch.chords if chord_vertices is None else chord_vertices,
                dtype=np.float64,
            )
            if chords.shape != (batch.curves.size, 2) or not np.all(np.isfinite(chords)):
                raise ValueError(
                    f"CurveTrimLoop.chord_vertices must be finite and aligned with the arc partition; got shape {chords.shape}"
                )
            geometry = _prepare_loop_geometry(
                curves_, batch, chords, closure, budget=budget, preparations=preparations
            )
            self.curves = curves_
            self.source_chords = batch.chords
            self.chords = chords
            self.chord_endpoint_error = geometry.endpoint_error
            self.arc_curves = batch.curves
            self.arc_first = batch.first
            self.arc_last = batch.last
            self.arc_lower = geometry.arc_lower
            self.arc_upper = geometry.arc_upper
            self.junction_lower = geometry.junction_lower
            self.junction_upper = geometry.junction_upper
            self.tolerance = tolerance_
            self.relative_closure_tolerance = closure.relative_tolerance
            self.closure_tolerance = closure.tolerance
            self.closure_gap = closure.gap

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

    @checked
    def certify_topology(
        self,
        /,
        *,
        maximum_pairs: int = 200_000,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> TrimTopologyEvidence:
        """Prove source/chord isotopy for this bounded chord cover.

        Source arcs and adjacent pairs must be monotone along a separating
        projection, and nonadjacent homotopy boxes must be disjoint. Ambiguous
        covers require a tighter caller-selected cover, not a sampled topology
        decision. Source joins share endpoint identity or remain inside the
        directed closure bound retained by this loop.
        """
        if budget is None or preparations is None:
            with _trim_coordinate_request(
                (
                    self,
                    maximum_pairs,
                ),
                budget=budget,
                preparations=preparations,
            ) as (budget, preparations):
                return self.certify_topology(
                    maximum_pairs=maximum_pairs, budget=budget, preparations=preparations
                )
        if (
            isinstance(maximum_pairs, bool)
            or not isinstance(maximum_pairs, int)
            or maximum_pairs < 1
        ):
            raise ValueError(
                f"CurveTrimLoop.maximum_pairs must be a positive integer; got {maximum_pairs!r}"
            )
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
                with budget.measurement():
                    bounds = curve.derivative_bounds(
                        float(self.arc_first[arc]),
                        float(self.arc_last[arc]),
                        budget=budget,
                        preparations=preparations,
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
                    self.curves[second_curve], budget=budget, preparations=preparations
                ):
                    first_parameter = self.curves[first_curve].parameter_interval[1]
                    second_parameter = self.curves[second_curve].parameter_interval[0]
                    endpoints = np.asarray(
                        jax.device_get(
                            (
                                self.curves[first_curve].evaluate(
                                    jnp.asarray(first_parameter, dtype=jnp.float64),
                                    budget=budget,
                                    preparations=preparations,
                                ),
                                self.curves[second_curve].evaluate(
                                    jnp.asarray(second_parameter, dtype=jnp.float64),
                                    budget=budget,
                                    preparations=preparations,
                                ),
                            )
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
def _trim_points(curve: AbstractTrimCurve, parameters: ArrayLike, /) -> Array:
    """Evaluate the curve with dynamic numerical leaves in one stable device kernel."""
    return curve.evaluate(jnp.asarray(parameters, dtype=jnp.float64))


class _ArcBatch(StrictModule):
    """Align host arrays of one certified trim partition."""

    __strict_contract__ = True

    curves: HostInt32[_ArcDim]
    first: HostFloat64[_ArcDim]
    last: HostFloat64[_ArcDim]
    chords: HostFloat64[_ArcDim, Literal[2]]
    boxes: HostFloat64[_ArcDim, Literal[2], Literal[2]]


def _arc_record(
    arcs: list[tuple[int, float, float, np.ndarray, np.ndarray]], /
) -> _ArcBatch:
    """Pack a bounded arc partition into aligned host arrays."""
    return _ArcBatch(
        curves=np.asarray([arc[0] for arc in arcs], dtype=np.int32),
        first=np.asarray([arc[1] for arc in arcs], dtype=np.float64),
        last=np.asarray([arc[2] for arc in arcs], dtype=np.float64),
        chords=np.asarray([arc[3] for arc in arcs], dtype=np.float64),
        boxes=np.asarray([arc[4] for arc in arcs], dtype=np.float64),
    )


def _arc_enclosure(
    curve: AbstractTrimCurve,
    first: float,
    last: float,
    start: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclosure of one arc, widened to contain its evaluated endpoints."""
    return _widened_arc_box(
        curve,
        curve.enclosure(first, last, budget=budget, preparations=preparations),
        last,
        start,
        preparations=preparations,
        budget=budget,
    )


def _widened_arc_box(
    curve: AbstractTrimCurve,
    source_box: npt.ArrayLike,
    last: float,
    start: np.ndarray,
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Admit one source arc box and widen it by the arc's evaluated endpoints."""
    box = np.asarray(source_box, dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)):
        raise ValueError(
            f"AbstractTrimCurve.enclosure must return finite (2, 2) boxes; got shape {box.shape}"
        )
    end = type(curve).host_points(
        curve, np.asarray(last, dtype=np.float64), preparations=preparations, budget=budget
    )
    lower = np.minimum(box[0], np.minimum(start, end))
    upper = np.maximum(box[1], np.maximum(start, end))
    return np.stack((lower, upper)), end


def _affine_endpoint_error_squared(
    lower: np.ndarray,
    upper: np.ndarray,
    point: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Admit and evaluate the two-coordinate endpoint displacement expression."""
    low_values, high_values, realized_values = (
        tuple(
            rational_coefficient(value, role="trim_curve.endpoint_coordinate")
            for value in rational_points(values[None, :], budget=budget)[0]
        )
        for values in (lower, upper, point)
    )
    # Four differences, four absolute values, two extremum comparisons,
    # two squares and one sum. Expanding the norm uses eight input occurrences.
    reserve_fraction_work(
        (low_values, high_values, realized_values), 13, 13, 8, budget=budget
    )
    displacement = tuple(
        max(abs(low - realized), abs(high - realized))
        for low, high, realized in zip(
            low_values, high_values, realized_values, strict=True
        )
    )
    return displacement[0] * displacement[0] + displacement[1] * displacement[1]


def _affine_chord_within_tolerance(
    curve: AbstractTrimCurve,
    first: float,
    last: float,
    start: np.ndarray,
    end: np.ndarray,
    tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
) -> bool:
    """Use convexity and rational squared endpoint distances to bound a chord.

    Every admitted source endpoint lies in its recorded box. For an affine
    carrier, interpolation of its two endpoint displacements is a convex
    combination, so its Euclidean norm cannot exceed the larger endpoint norm.
    Fraction arithmetic retains the stored chord's rounding displacement and
    every recorded root uncertainty; the squared tolerance needs no square root.
    """
    evidence = curve.affine_chord_evidence(
        first, last, budget=budget, preparations=preparations
    )
    if evidence is None:
        return False
    if evidence.source_id != canonical_fingerprint(carrier_payload(curve)):
        raise ValueError(
            "CurveTrimLoop.affine_chord_evidence.source_id must bind the queried carrier; got a mismatched source fingerprint"
        )
    tolerance_value = rational_points(
        np.asarray(((tolerance,),), dtype=np.float64), budget=budget
    )[0][0]
    reserve_fraction_work(((tolerance_value,),), 1, 1, 2, budget=budget)
    tolerance_squared = tolerance_value * tolerance_value
    maximum_squared = Fraction(0)
    for lower, upper, point in (
        (evidence.start_lower, evidence.start_upper, start),
        (evidence.end_lower, evidence.end_upper, end),
    ):
        squared = _affine_endpoint_error_squared(lower, upper, point, budget=budget)
        reserve_fraction_work(((maximum_squared, squared),), 1, 2, 2, budget=budget)
        maximum_squared = max(maximum_squared, squared)
    reserve_fraction_work(((maximum_squared, tolerance_squared),), 1, 2, 2, budget=budget)
    return maximum_squared <= tolerance_squared


type _SourceArc = tuple[float, float, np.ndarray, np.ndarray, np.ndarray]


@dataclass(slots=True)
class TrimSourceArcs:
    """Request facts of one oriented-trim source binding, in forward orientation.

    ``sources`` is the binding from ``AbstractTrimCurve.arc_partition_binding``.
    Point boxes are keyed by forward local parameter on ``[0, 1]``; partitions
    map a cover tolerance to accepted ``(lower, upper, start, end, box)`` arcs.
    A reversed trim reads both through the exact involution ``s -> 1 - s``.
    """

    sources: tuple[object, ...]
    points: dict[float, np.ndarray] = field(default_factory=dict)
    partitions: dict[float, tuple[_SourceArc, ...]] = field(default_factory=dict)


def _source_arcs(
    curve: AbstractTrimCurve, preparations: TrimRequestPreparations | None, /
) -> tuple[TrimSourceArcs, bool] | None:
    binding = None if preparations is None else curve.arc_partition_binding()
    if binding is None or preparations is None:
        return None
    sources, reversed_ = binding
    key = tuple(value if isinstance(value, float) else id(value) for value in sources)
    record = preparations.trim_source_arcs.get(key)
    if record is None or any(
        retained is not value and not (isinstance(value, float) and retained == value)
        for retained, value in zip(record.sources, sources, strict=True)
    ):
        record = TrimSourceArcs(sources)
        preparations.trim_source_arcs[key] = record
    return record, reversed_


def _mirror(parameter: float, /) -> float | None:
    """``1 - parameter`` when binary64 represents it exactly, else None."""
    image = 1.0 - parameter
    return image if Fraction(image) == 1 - Fraction(parameter) else None


def _mirrored_arcs(arcs: Sequence[_SourceArc], /) -> tuple[_SourceArc, ...] | None:
    mirrored = []
    for lower, upper, start, end, box in reversed(arcs):
        first, last = _mirror(upper), _mirror(lower)
        if first is None or last is None:
            return None
        mirrored.append((first, last, end, start, box))
    return tuple(mirrored)


def _point_enclosures(
    curve: AbstractTrimCurve,
    parameters: Sequence[float],
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
) -> np.ndarray:
    """Point boxes ``enclosure(s, s)``, shared by both orientations of one source."""
    source = _source_arcs(curve, preparations)
    values = [float(parameter) for parameter in parameters]
    keys = [
        None if source is None else (_mirror(value) if source[1] else value)
        for value in values
    ]
    boxes: dict[int, np.ndarray] = {}
    for position, key in enumerate(keys):
        if source is not None and key is not None and key in source[0].points:
            boxes[position] = source[0].points[key]
    missing = [position for position in range(len(values)) if position not in boxes]
    if missing:
        computed = curve.enclosures(
            np.asarray([(values[position], values[position]) for position in missing]),
            budget=budget,
            preparations=preparations,
        )
        for position, box in zip(missing, computed, strict=True):
            boxes[position] = np.asarray(box, dtype=np.float64)
            key = keys[position]
            if source is not None and key is not None:
                source[0].points[key] = boxes[position]
    return np.stack([boxes[position] for position in range(len(values))])


def _certified_arcs(
    curve: AbstractTrimCurve,
    index: int,
    first: float,
    last: float,
    tolerance: float,
    capacity: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> list[tuple[int, float, float, np.ndarray, np.ndarray]]:
    """Bisect until every arc box meets the tolerance or an affine chord proof.

    Each subdivision level queries its pending arcs together; every arc keeps
    the enclosure, acceptance and split rules of one-at-a-time bisection. A
    source binding shares the accepted cover between both orientations.
    """
    source = _source_arcs(curve, preparations) if (first, last) == (0.0, 1.0) else None
    if source is not None:
        cover = source[0].partitions.get(tolerance)
        arcs = (
            None
            if cover is None
            else (_mirrored_arcs(cover) if source[1] else cover)
        )
        if arcs is not None:
            if len(arcs) > capacity:
                raise ValueError(
                    f"CurveTrimLoop.maximum_arcs must admit the certified cover; got {len(arcs)} > {capacity}"
                )
            return [
                (index, lower, upper, start, box) for lower, upper, start, _, box in arcs
            ]

    from .brep._intersection import _padded_call

    def points(parameters: Sequence[float]) -> np.ndarray:
        (values,) = _padded_call(
            lambda rows: curve.host_points(
                rows, preparations=preparations, budget=budget
            ),
            np.asarray(parameters, dtype=np.float64),
            64,
        )
        return values

    endpoints = points((first, last))
    level: list[tuple[float, float, np.ndarray, np.ndarray]] = [
        (first, last, endpoints[0], endpoints[1])
    ]
    accepted: list[_SourceArc] = []
    while level:
        budget.ticket.checkpoint()
        raw = curve.enclosures(
            np.asarray([(lower, upper) for lower, upper, _, _ in level]),
            budget=budget,
            preparations=preparations,
        )
        split: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
        for (lower, upper, start, end), enclosure in zip(level, raw, strict=True):
            if enclosure.shape != (2, 2) or not np.all(np.isfinite(enclosure)):
                raise ValueError(
                    f"AbstractTrimCurve.enclosure must return finite (2, 2) boxes; got shape {enclosure.shape}"
                )
            box = np.stack(
                (
                    np.minimum(enclosure[0], np.minimum(start, end)),
                    np.maximum(enclosure[1], np.maximum(start, end)),
                )
            )
            if np.linalg.norm(
                box[1] - box[0]
            ) <= tolerance or _affine_chord_within_tolerance(
                curve,
                lower,
                upper,
                start,
                end,
                tolerance,
                budget=budget,
                preparations=preparations,
            ):
                accepted.append((lower, upper, start, end, box))
                continue
            middle = 0.5 * (lower + upper)
            if not lower < middle < upper:
                raise ValueError(
                    f"CurveTrimLoop.tolerance must be reachable by representable subdivisions; got unsplittable interval ({lower!r}, {upper!r})"
                )
            split.append((lower, middle, upper, start, end))
        # Every pending arc contributes at least one accepted arc.
        required = len(accepted) + 2 * len(split)
        if required > capacity:
            raise ValueError(
                f"CurveTrimLoop.maximum_arcs must admit the certified cover; got {required} > {capacity}"
            )
        following: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        if split:
            middle_points = points([arc[1] for arc in split])
            for (lower, middle, upper, start, end), point in zip(
                split, middle_points, strict=True
            ):
                following.extend(
                    ((lower, middle, start, point), (middle, upper, point, end))
                )
        level = following
    accepted.sort(key=lambda arc: arc[0])
    if source is not None:
        forward = _mirrored_arcs(accepted) if source[1] else tuple(accepted)
        if forward is not None:
            source[0].partitions[tolerance] = forward
    return [(index, lower, upper, start, box) for lower, upper, start, _, box in accepted]


class DynamicCurveTrimLoop(StrictModule):
    """Device-resident source-arc cover for a fixed-incidence design epoch.

    Chords give winding only outside conservative arc and junction boxes.
    Overlapping covers require device query evidence, not host refinement.
    """

    chords: Array
    arc_lower: Array
    arc_upper: Array
    junction_lower: Array
    junction_upper: Array
    source_id: str = eqx.field(static=True)

    @property
    def identity(self) -> dict[str, object]:
        return {"kind": "dynamic-curve-loop", "source_id": self.source_id}


type TrimLoop = PolygonTrimLoop | CurveTrimLoop | DynamicCurveTrimLoop


def _as_loop(value: TrimLoop | npt.ArrayLike, /) -> TrimLoop:
    match value:
        case PolygonTrimLoop() | CurveTrimLoop() | DynamicCurveTrimLoop():
            return value
        case _:
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
    """Record source-winding point-in-trim decisions with per-point evidence.

    ``inside`` is meaningful where ``resolved`` holds. ``boundary`` marks points
    exactly on a polygon edge or within the refined enclosure of a curve arc at
    the refinement budget; such points are unresolved unless they lie exactly on
    an affine edge. ``winding`` holds the signed chord winding number per loop.
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
    """Compute closed-polygon winding with exact predicates and on-edge detection."""
    start = chords
    end = np.roll(chords, -1, axis=0)
    # Host classification: take the certified host signs directly instead of
    # staging them on the device and reading them back for every query.
    signs = np.asarray(
        orient2d_host(start, end, np.broadcast_to(point, start.shape), mode=mode).signs,
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
    loop: CurveTrimLoop,
    point: np.ndarray,
    maximum_depth: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> tuple[np.ndarray, bool, int]:
    """Chord polygon whose arc enclosures exclude ``point`` where resolvable.

    Raw source boxes of the dyadic arc tree are the request's earned facts in
    ``TrimArcRefinements`` (shared with ``classify_boxes``); each query still
    widens them by its own evaluated endpoints and decides independently.
    """
    chords: list[np.ndarray] = []
    # Junction boxes retain endpoint uncertainty; membership inside them remains
    # unresolved independently of numerical endpoint agreement.
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
        memo = _request_arc_refinements(
            loop, arc, budget=budget, preparations=preparations
        )
        # Heap positions: one is the whole arc; 2p and 2p + 1 its halves.
        stack = [(float(loop.arc_first[arc]), float(loop.arc_last[arc]), 0, 1)]
        start = loop.source_chords[arc]
        pieces: list[np.ndarray] = []
        while stack:
            first, last, depth, position = stack.pop()
            box, end = _widened_arc_box(
                curve,
                _arc_node_box(
                    memo, curve, position, first, last, preparations=preparations
                ),
                last,
                start,
                preparations=preparations,
                budget=memo.budget,
            )
            outside = np.any(point < box[0]) or np.any(point > box[1])
            middle = 0.5 * (first + last)
            if outside or depth >= maximum_depth or not first < middle < last:
                unresolved |= not outside
                pieces.append(start)
                start = end
                continue
            refinements += 1
            stack.append((middle, last, depth + 1, 2 * position + 1))
            stack.append((first, middle, depth + 1, 2 * position))
        chords.extend(pieces)
    return np.asarray(chords, dtype=np.float64), unresolved, refinements


@dataclass(slots=True)
class TrimArcRefinements:
    """Retain earned dyadic source facts for one arc and request resource owner.

    Heap positions identify the lazy subdivision tree: one is the
    full arc, and twice a position and twice plus one identify its children.
    Only visited source intervals are retained; policy limits remain with the
    consuming query.
    """

    loop: CurveTrimLoop
    budget: CoordinateEnclosureBudget
    boxes: dict[int, np.ndarray] = field(default_factory=dict)
    chords: dict[int, np.ndarray] = field(default_factory=dict)
    refined: set[int] = field(default_factory=set)


class _BoxLoopExclusion(StrictModule):
    """A loop chord cover and the work consumed to exclude a query rectangle."""

    chords: np.ndarray
    resolved: bool = eqx.field(static=True)
    refinements: int = eqx.field(static=True)


def _disjoint_boxes(
    query: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> bool:
    # Trim charts are two-dimensional; keep existing bounds separate without a tiny allocation.
    return bool(
        query[1, 0] < lower[0]
        or query[1, 1] < lower[1]
        or upper[0] < query[0, 0]
        or upper[1] < query[0, 1]
    )


def _exclude_polygon_boundary(
    loop: PolygonTrimLoop, query: np.ndarray, /
) -> _BoxLoopExclusion:
    corners = np.asarray(
        (
            query[0],
            (query[1, 0], query[0, 1]),
            query[1],
            (query[0, 0], query[1, 1]),
        )
    )
    for start, end in zip(loop.vertices, np.roll(loop.vertices, -1, axis=0), strict=True):
        if _disjoint_boxes(query, np.minimum(start, end), np.maximum(start, end)):
            continue
        signs = np.asarray(
            orient2d_host(
                np.broadcast_to(start, corners.shape),
                np.broadcast_to(end, corners.shape),
                corners,
                mode=PredicateMode.EXACT,
            ).signs
        )
        if not (np.all(signs > 0) or np.all(signs < 0)):
            return _BoxLoopExclusion(loop.vertices, False, 0)
    return _BoxLoopExclusion(loop.vertices, True, 0)


# Container metadata of one request arc tree: the slotted record (80 B), its
# two empty mappings and set (344 B), the (loop, arc) key (120 B) and one
# amortized request slot including table growth (96 B).
_ARC_REFINEMENTS_BYTES = 640
# One position-keyed slot of an arc tree mapping, amortized with table growth.
_ARC_NODE_SLOT_BYTES = 96


def _request_arc_refinements(
    loop: CurveTrimLoop,
    arc: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> TrimArcRefinements:
    key = (id(loop), arc)
    memo = preparations.trim_arc_refinements.get(key)
    if memo is None or memo.loop is not loop or memo.budget is not budget:
        budget.retain_operand_metadata(_ARC_REFINEMENTS_BYTES)
        memo = TrimArcRefinements(loop, budget)
        preparations.trim_arc_refinements[key] = memo
    return memo


def _arc_node_box(
    memo: TrimArcRefinements,
    curve: AbstractTrimCurve,
    position: int,
    first: float,
    last: float,
    /,
    *,
    preparations: TrimRequestPreparations,
) -> np.ndarray:
    """One raw dyadic arc box; only finite ``(2, 2)`` boxes become request facts.

    A retained box is a private read-only copy, so neither a provider nor a
    consumer can change an earned fact after admission. Nonfinite provider
    boxes are returned to the consuming query for its own refusal and never
    retained.
    """
    box = memo.boxes.get(position)
    if box is None:
        # Keep override dispatch without building an Eqx BoundMethod over the source DAG.
        box = np.array(
            type(curve).enclosure(
                curve, first, last, budget=memo.budget, preparations=preparations
            ),
            dtype=np.float64,
        )
        if box.shape == (2, 2) and np.all(np.isfinite(box)):
            box.setflags(write=False)
            memo.budget.retain_operand_metadata(_ARC_NODE_SLOT_BYTES)
            memo.budget.retain_basis((position, box))
            memo.boxes[position] = box
    return box


def _arc_node_chord(
    memo: TrimArcRefinements,
    curve: AbstractTrimCurve,
    position: int,
    first: float,
    /,
    *,
    preparations: TrimRequestPreparations,
) -> np.ndarray:
    point = memo.chords.get(position)
    if point is None:
        point = type(curve).host_points(
            curve, np.asarray(first, dtype=np.float64),
            budget=memo.budget,
            preparations=preparations,
        )
        memo.budget.retain_basis((position, point))
        memo.chords[position] = point
    return point


def _refine_arc_for_box(
    memo: TrimArcRefinements,
    arc: int,
    query: np.ndarray,
    maximum_depth: int,
    remaining_refinements: int,
    /,
    *,
    preparations: TrimRequestPreparations,
) -> _BoxLoopExclusion:
    """Consume a lazy source partition and charge only newly earned splits."""
    loop, budget = memo.loop, memo.budget
    curve = loop.curves[int(loop.arc_curves[arc])]
    pending = [(float(loop.arc_first[arc]), float(loop.arc_last[arc]), 0, 1)]
    chords = []
    used = 0
    while pending:
        first, last, depth, position = pending.pop()
        box = _arc_node_box(memo, curve, position, first, last, preparations=preparations)
        if _disjoint_boxes(query, box[0], box[1]):
            point = _arc_node_chord(
                memo, curve, position, first, preparations=preparations
            )
            chords.append(point)
            continue
        middle = 0.5 * (first + last)
        if depth >= maximum_depth or not first < middle < last:
            return _BoxLoopExclusion(np.asarray(chords), False, used)
        if position not in memo.refined:
            if used >= remaining_refinements:
                return _BoxLoopExclusion(np.asarray(chords), False, used)
            budget.reserve(1)
            used += 1
            memo.refined.add(position)
        pending.append((middle, last, depth + 1, 2 * position + 1))
        pending.append((first, middle, depth + 1, 2 * position))
    return _BoxLoopExclusion(np.asarray(chords), True, used)


def _exclude_curve_boundary(
    loop: CurveTrimLoop,
    query: np.ndarray,
    maximum_depth: int,
    remaining_refinements: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> _BoxLoopExclusion:
    chords = []
    resolved = True
    used = 0
    junction_lower, junction_upper = loop.junction_lower, loop.junction_upper
    for junction in range(junction_lower.shape[0]):
        if not _disjoint_boxes(query, junction_lower[junction, :], junction_upper[junction, :]):
            return _BoxLoopExclusion(np.empty((0, 2), dtype=np.float64), False, 0)
    for arc in range(loop.source_chords.shape[0]):
        if _disjoint_boxes(query, loop.arc_lower[arc], loop.arc_upper[arc]):
            chords.append(loop.source_chords[arc])
            continue
        memo = _request_arc_refinements(
            loop, arc, budget=budget, preparations=preparations
        )
        exclusion = _refine_arc_for_box(
            memo,
            arc,
            query,
            maximum_depth,
            remaining_refinements - used,
            preparations=preparations,
        )
        chords.extend(exclusion.chords)
        used += exclusion.refinements
        if not exclusion.resolved:
            resolved = False
            break
    return _BoxLoopExclusion(np.asarray(chords), resolved, used)


class TrimDomain(StrictModule):
    """Oriented trim domain in a two-dimensional chart.

    Loops are explicit affine polygons (`PolygonTrimLoop`; array inputs are
    converted once), source-curve chains (`CurveTrimLoop`), or device-resident
    source-arc covers (`DynamicCurveTrimLoop`). A point is inside when it has
    nonzero winding about the outer loop and zero winding about every hole.
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

        Polygon winding is evaluated directly; curve-loop winding agrees with
        source winding outside each certified approximation band. ``classify``
        resolves points inside the band.
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
            if isinstance(loop, (CurveTrimLoop, DynamicCurveTrimLoop)):
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

    @checked
    def classify_boxes(
        self,
        reference: npt.ArrayLike,
        /,
        *,
        maximum_depth: int = 48,
        maximum_refinements: int = 100_000,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> TrimBoxClassification:
        """Classify a closed rectangle by excluding its source boundary.

        Center winding is admitted only after arc/junction enclosures exclude
        the boundary from every point of the rectangle.
        Disjoint certified arc covers reuse their stored source chords; only
        overlapping covers request source refinement.
        Dyadic source boxes and chords are retained within the supplied request;
        ``refinements`` counts newly earned splits, charged once before evaluation.
        Junction overlap fixes the box as unresolved before arc refinement, so
        its unused refinement allowance remains available to later boxes.
        """
        if budget is None or preparations is None:
            with _trim_coordinate_request(
                (
                    self,
                    reference,
                    maximum_depth,
                    maximum_refinements,
                ),
                budget=budget,
                preparations=preparations,
            ) as (budget, preparations):
                return self.classify_boxes(
                    reference,
                    maximum_depth=maximum_depth,
                    maximum_refinements=maximum_refinements,
                    budget=budget,
                    preparations=preparations,
                )
        boxes = np.asarray(reference, dtype=np.float64)
        if (
            boxes.ndim != 3
            or boxes.shape[1:] != (2, 2)
            or not np.all(np.isfinite(boxes))
            or np.any(boxes[:, 0] > boxes[:, 1])
        ):
            raise ValueError(
                f"TrimDomain.reference must contain finite ordered (num_boxes, 2, 2) rectangles; got shape {boxes.shape}"
            )
        if maximum_depth < 0 or maximum_refinements < 0:
            raise ValueError(
                f"TrimDomain.refinement_limits must be nonnegative; got depth={maximum_depth}, refinements={maximum_refinements}"
            )
        inside = np.zeros(boxes.shape[0], dtype=np.bool_)
        resolved = np.ones(boxes.shape[0], dtype=np.bool_)
        work = np.zeros(boxes.shape[0], dtype=np.int32)
        used = 0

        # The refinement allowance is shared across rows in their input order.
        for row in range(boxes.shape[0]):
            query = np.asarray(boxes[row], dtype=np.float64)
            windings = []
            for loop in self.loops:
                match loop:
                    case PolygonTrimLoop():
                        exclusion = _exclude_polygon_boundary(loop, query)
                    case CurveTrimLoop():
                        exclusion = _exclude_curve_boundary(
                            loop,
                            query,
                            maximum_depth,
                            maximum_refinements - used,
                            budget=budget,
                            preparations=preparations,
                        )
                    case DynamicCurveTrimLoop():
                        raise ValueError(
                            "TrimDomain.loops must use device execution for dynamic covers; got a host refinement request."
                        )
                    case _:
                        raise TypeError(
                            f"TrimDomain.loops must contain supported trim loops; got {type(loop).__qualname__}"
                        )
                used += exclusion.refinements
                work[row] += exclusion.refinements
                resolved[row] = exclusion.resolved
                chords = exclusion.chords
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

    @checked
    def classify(
        self,
        reference: npt.ArrayLike,
        /,
        *,
        mode: PredicateMode = PredicateMode.EXACT,
        maximum_depth: int = 48,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> TrimClassification:
        """Classify host trim queries with bounded source-curve refinement."""
        if budget is None or preparations is None:
            with _trim_coordinate_request(
                (self, reference, maximum_depth), budget=budget, preparations=preparations
            ) as (budget, preparations):
                return self.classify(
                    reference,
                    mode=mode,
                    maximum_depth=maximum_depth,
                    budget=budget,
                    preparations=preparations,
                )
        mode_ = resolve_host_predicate_mode(mode)
        points = np.asarray(reference, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"TrimDomain.reference must have shape (num_points, 2); got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("TrimDomain.reference must be finite; got nonfinite points")
        if maximum_depth < 0:
            raise ValueError(
                f"TrimDomain.maximum_depth must be nonnegative; got {maximum_depth}"
            )
        loops = self.loops
        winding = np.zeros((points.shape[0], len(loops)), dtype=np.int32)
        boundary = np.zeros((points.shape[0],), dtype=np.bool_)
        resolved = np.ones((points.shape[0],), dtype=np.bool_)
        refinements = np.zeros((points.shape[0],), dtype=np.int32)
        for row in range(points.shape[0]):
            point = np.asarray(points[row], dtype=np.float64)
            for column, loop in enumerate(loops):
                if isinstance(loop, DynamicCurveTrimLoop):
                    raise ValueError(
                        "TrimDomain.loops must use device execution for dynamic covers; got a host refinement request."
                    )
                if isinstance(loop, CurveTrimLoop):
                    chords, unresolved, count = _refined_chords(
                        loop,
                        point,
                        maximum_depth,
                        budget=budget,
                        preparations=preparations,
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

    @checked
    def __init__(
        self,
        mapping: AbstractBoundaryMap,
        *,
        source_entity_ids: ArrayLike,
        source_id: str,
        physical_tags: Sequence[str] | None = None,
        orientation: ArrayLike | None = None,
        seam_owner: ArrayLike | None = None,
        trim_domains: Sequence[TrimDomain | None] | None = None,
    ) -> None:
        entity_ids = np.asarray(source_entity_ids, dtype=np.int32).reshape((-1,))
        if entity_ids.shape != (mapping.num_charts,):
            raise ValueError(
                f"BoundaryAtlas.source_entity_ids must contain one ID per chart; got {entity_ids.shape} for {mapping.num_charts} charts"
            )
        tags = (
            tuple("boundary" for _ in range(mapping.num_charts))
            if physical_tags is None
            else tuple(physical_tags)
        )
        if len(tags) != mapping.num_charts or any(not tag for tag in tags):
            raise ValueError(
                f"BoundaryAtlas.physical_tags must contain one nonempty tag per chart; got {tags!r}"
            )
        orientation_ = (
            np.ones((mapping.num_charts,), dtype=np.float64)
            if orientation is None
            else np.asarray(orientation, dtype=np.float64).reshape((-1,))
        )
        if orientation_.shape != (mapping.num_charts,):
            raise ValueError(
                f"BoundaryAtlas.orientation must contain one sign per chart; got {orientation_.shape}"
            )
        if np.any((orientation_ != 1.0) & (orientation_ != -1.0)):
            raise ValueError(
                f"BoundaryAtlas.orientation must contain only +1 or -1; got {orientation_!r}"
            )
        seam_owner_ = (
            np.ones((mapping.num_charts,), dtype=np.bool_)
            if seam_owner is None
            else np.asarray(seam_owner, dtype=np.bool_).reshape((-1,))
        )
        if seam_owner_.shape != (mapping.num_charts,):
            raise ValueError(
                f"BoundaryAtlas.seam_owner must contain one flag per chart; got {seam_owner_.shape}"
            )
        trims = (
            tuple(None for _ in range(mapping.num_charts))
            if trim_domains is None
            else tuple(trim_domains)
        )
        if len(trims) != mapping.num_charts or any(
            trim is not None and not isinstance(trim, TrimDomain) for trim in trims
        ):
            raise ValueError(
                f"BoundaryAtlas.trim_domains must contain one TrimDomain or None per chart; got {trims!r}"
            )
        source_id_ = canonical_identifier(source_id, "BoundaryAtlas.source_id")
        self.mapping = mapping
        self.source_entity_ids = jnp.asarray(entity_ids, dtype=jnp.int32)
        self.source_id = source_id_
        self.physical_tags = tags
        self.orientation = jnp.asarray(orientation_, dtype=jnp.float64)
        self.seam_owner = jnp.asarray(seam_owner_, dtype=jnp.bool_)
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
            raise ValueError(
                f"BoundaryAtlas.chart_indices must match reference leading dimensions; got {indices.shape} for {reference_.shape}"
            )
        if reference_.shape[-1] != self.reference_dimension:
            raise ValueError(
                f"BoundaryAtlas.reference must have trailing dimension {self.reference_dimension}; got {reference_.shape}"
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

    @checked
    def classify_reference(
        self,
        chart_indices: npt.ArrayLike,
        reference: npt.ArrayLike,
        /,
        *,
        mode: PredicateMode = PredicateMode.EXACT,
        maximum_depth: int = 48,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> TrimClassification:
        """Classify flat ``(num_points,)`` chart points against source trim boundaries.

        Untrimmed charts classify every point as inside and resolved.
        """
        indices = np.asarray(chart_indices, dtype=np.int32).reshape((-1,))
        points = np.asarray(reference, dtype=np.float64)
        if self.reference_dimension != 2:
            raise ValueError(
                f"BoundaryAtlas.reference_dimension must be 2 for trim classification; got {self.reference_dimension}"
            )
        if points.shape != (indices.shape[0], 2):
            raise ValueError(
                f"BoundaryAtlas.reference must have shape ({indices.shape[0]}, 2); got {points.shape}"
            )
        if np.any((indices < 0) | (indices >= self.num_charts)):
            raise ValueError(
                f"BoundaryAtlas.chart_indices must identify charts in [0, {self.num_charts}); got {indices!r}"
            )
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
            result = trim.classify(
                points[rows],
                mode=mode,
                maximum_depth=maximum_depth,
                budget=budget,
                preparations=preparations,
            )
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
                f"BoundaryAtlas.mapping must have codimension one for frame evaluation; got dimensions {self.reference_dimension}, {self.ambient_dimension}"
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
            # Higher-dimensional frames resolve the linalg facade on request.
            orthogonal = phydrax.linalg.orthonormal_frame(differential)
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
            raise ValueError(
                f"BoundaryAtlas.selection must contain at least one chart; got entity_ids={entity_ids!r}, tags={tags!r}"
            )
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
                f"BoundaryAtlas.offset must have shape ({base.ambient_dimension},); got {offset_.shape}"
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

    def __init__(self, center: ArrayLike, radius: ArrayLike) -> None:
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

    def __init__(self, center: ArrayLike, radius: ArrayLike) -> None:
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

    def __init__(self, center: ArrayLike, size: ArrayLike) -> None:
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
    center: ArrayLike, radius: ArrayLike, /, *, source_id: str
) -> BoundaryAtlas:
    return BoundaryAtlas(
        _CircleBoundaryMap(center, radius),
        source_entity_ids=np.zeros((4,), dtype=np.int32),
        source_id=source_id,
    )


def sphere_boundary_atlas(
    center: ArrayLike, radius: ArrayLike, /, *, source_id: str
) -> BoundaryAtlas:
    triangle = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    return BoundaryAtlas(
        _SphereBoundaryMap(center, radius),
        source_entity_ids=np.asarray([0, 0], dtype=np.int32),
        orientation=-np.ones((2,), dtype=np.float64),
        source_id=source_id,
        trim_domains=(TrimDomain(triangle), TrimDomain(triangle)),
    )


def box_boundary_atlas(
    center: ArrayLike, size: ArrayLike, /, *, source_id: str
) -> BoundaryAtlas:
    return BoundaryAtlas(
        _BoxBoundaryMap(center, size),
        orientation=np.asarray([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0], dtype=np.float64),
        physical_tags=("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
        source_entity_ids=np.arange(6, dtype=np.int32),
        source_id=source_id,
    )


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
    "box_boundary_atlas",
    "circle_boundary_atlas",
    "sphere_boundary_atlas",
]
