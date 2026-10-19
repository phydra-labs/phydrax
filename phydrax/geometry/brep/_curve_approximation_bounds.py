#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Enclose continuous, coupled external approximations.

The source is the certified implicit graph. Bernstein coefficient
intervals enclose the fitted splines.  Each closed chart/knot cell uses a
midpoint *enclosure*, first difference jets and a second difference remainder.
Native surface correspondence is proved separately for both fitted UV curves.
Arithmetic uses the outward-rounded interval interpreter (IEEE basic
operations and the documented four-ulp transcendental assumption).

``maximum_bytes`` bounds proof-owned numeric storage, retained cell records and
interval-interpreter numeric workspace; it is not a Python/JAX process RSS limit.
Source definitions and the runtime's existing compilation caches are borrowed,
not duplicated or changed.  There are no numerical source-root evaluations here.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import ShapedArray
from jax.extend.core import Jaxpr

from ..._strict import StrictModule
from ..._validation import nonnegative_integer, positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._interval_enclosure import interval_subtract, PreparedIntervalFunction
from ._carrier_pieces import CurvePiece
from ._coordinate_request import coordinate_source_request
from ._correspondence import bernstein_rational_piece, RationalCurveSurfaceJets
from ._intersection_curve import (
    curve_pieces_for_interval,
    intersection_derivative_bounds,
    IntersectionCurve,
    IntersectionJetPreparation,
    surface_pieces_for_box,
)
from ._patches import (
    AbstractCurve,
    BernsteinCurvePiece,
    BSplineCurve,
    BSplineSurfacePatch,
    rational_piece_jet_bounds,
)
from ._request_preparations import TrimRequestPreparations
from ._source_correspondence import norm_upper


@dataclass(frozen=True, slots=True)
class CoupledApproximationEvidence:
    """Whole-domain upper bounds, not sampled maximum errors.

    ``interval_evidence`` has columns (first, last, depth, distance, parameter,
    first correspondence, second correspondence, accepted).  It retains EVERY
    processed cell, including parents that required subdivision. ``cells`` counts
    processed cells, not just accepted leaves. ``jet_evaluations`` counts interval
    value/Jacobian/Hessian calls, including source enclosure contraction calls.
    """

    distance_bound: float
    parameter_bound: float
    first_correspondence_bound: float
    second_correspondence_bound: float
    cells: int
    maximum_depth_used: int
    interval_evidence: np.ndarray
    jet_evaluations: int
    bytes_used: int
    peak_bytes: int


class ApproximationResourceError(ValueError):
    """A continuous proof exhausted an explicit resource, not a failed sample."""

    def __init__(
        self,
        resource: str,
        cause: str,
        *,
        cells: int,
        depth: int,
        bytes_used: int,
        jet_evaluations: int,
        peak_bytes: int,
    ) -> None:
        self.resource = resource
        self.cause = cause
        self.cells = cells
        self.maximum_depth_used = depth
        self.bytes_used = bytes_used
        self.jet_evaluations = jet_evaluations
        self.peak_bytes = peak_bytes
        super().__init__(
            f"CoupledApproximation.{resource} must cover the continuous proof; got {cause}, cells={cells}, depth={depth}, bytes={bytes_used}"
        )


class ApproximationDeviationError(ValueError):
    """Report a source-enclosed point whose deviation exceeds its tolerance."""

    def __init__(
        self,
        kind: str,
        interval: tuple[float, float],
        lower_bound: float,
        tolerance: float,
        budget: _Budget,
    ) -> None:
        self.kind, self.interval = kind, interval
        self.lower_bound, self.tolerance = lower_bound, tolerance
        self.cells, self.jet_evaluations = budget.cells, budget.jets
        self.bytes_used, self.peak_bytes = budget.bytes, budget.peak
        self.maximum_depth_used = budget.depth
        super().__init__(
            f"CoupledApproximation.deviation must be at most {tolerance}; got {kind} lower bound {lower_bound} at interval {interval}"
        )


def _norm_lower(interval: tuple[np.ndarray, np.ndarray]) -> float:
    lower, upper = interval
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        return 0.0
    components = np.where(
        (lower <= 0.0) & (upper >= 0.0), 0.0, np.minimum(np.abs(lower), np.abs(upper))
    )
    squared = 0.0
    for component in components:
        term = max(0.0, float(np.nextafter(component * component, -np.inf)))
        squared = max(0.0, float(np.nextafter(squared + term, -np.inf)))
    return max(0.0, float(np.nextafter(np.sqrt(squared), -np.inf)))


class _Budget:
    def __init__(self, cells: int, depth: int, memory: int) -> None:
        self.maximum_cells, self.maximum_depth, self.maximum_bytes = cells, depth, memory
        self.cells = self.depth = self.bytes = self.peak = self.jets = 0

    def fail(self, resource: str, cause: str) -> None:
        raise ApproximationResourceError(
            resource,
            cause,
            cells=self.cells,
            depth=self.depth,
            bytes_used=self.bytes,
            jet_evaluations=self.jets,
            peak_bytes=self.peak,
        )

    def reserve(self, size: int, cause: str) -> None:
        if size > self.maximum_bytes - self.bytes:
            self.fail("maximum_bytes", cause)
        self.bytes += size
        self.peak = max(self.peak, self.bytes)

    def release(self, size: int) -> None:
        self.bytes -= size

    def cell(self, depth: int) -> None:
        if self.cells >= self.maximum_cells:
            self.fail("maximum_cells", "another closed interval requires processing")
        self.cells += 1
        self.depth = max(self.depth, depth)


def _graph_storage(prepared: PreparedIntervalFunction) -> tuple[int, int]:
    """Numeric constants and conservative interpreter live-buffer storage.

    The interpreter retains equation outputs in its environment. Reserving two
    bounds for EVERY variable (including nested jaxprs) plus twenty largest
    buffers also covers arithmetic stacks, gather-free structural temporaries,
    padding, and output copies. This is intentionally an upper, not an estimate
    based on sampled evaluations.
    """
    constants: dict[int, Array | np.ndarray | np.generic] = {}
    total = largest = 8

    def retain(value: object) -> None:
        match value:
            case Array() | np.ndarray() | np.generic():
                constants[id(value)] = value
            case _:
                pass

    def visit(graph: Jaxpr) -> None:
        nonlocal total, largest
        for value in graph.consts:
            retain(value)
        variables = list(graph.constvars) + list(graph.invars)
        for equation in graph.eqns:
            variables.extend(equation.outvars)
            for value in equation.params.values():
                match value:
                    case Jaxpr():
                        visit(value)
                    case tuple() | list():
                        for child in value:
                            match child:
                                case Jaxpr():
                                    visit(child)
                                case _:
                                    pass
                    case _:
                        pass
        for variable in variables:
            match variable.aval:
                case ShapedArray() as aval:
                    size = math.prod(aval.shape) * np.dtype(aval.dtype).itemsize
                    total += 2 * size
                    largest = max(largest, size)
                case _:
                    pass

    visit(prepared.closed_jaxpr)
    for triplet in prepared.constant_bounds:
        for value in triplet:
            retain(value)
    retained = sum(value.nbytes for value in constants.values())
    return retained, total + 20 * largest


class _IntervalMap:
    def __init__(
        self,
        prepared: PreparedIntervalFunction,
        budget: _Budget,
        *,
        borrowed: bool = False,
    ) -> None:
        self.prepared, self.budget = prepared, budget
        retained, self.workspace = _graph_storage(prepared)
        if not borrowed:
            budget.reserve(retained, "retaining interval coefficient buffers")

    def evaluate(
        self, box: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.budget.reserve(self.workspace, "evaluating source or spline interval jets")
        try:
            self.budget.jets += 1
            lower, upper = self.prepared.evaluate(box[0][None], box[1][None])
            return lower[0], upper[0]
        finally:
            self.budget.release(self.workspace)


def _map(
    function: Callable[[Array], Array],
    dimension: int,
    coefficients: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    budget: _Budget,
) -> _IntervalMap:
    # Build the prepared interpreter without its eager validation
    # allocation; validation is performed through the bounded evaluator below.
    # Tracing metadata is owned by JAX, not numeric interval proof workspace.
    budget.reserve(16 * dimension, "tracing one fixed-size interval input")
    try:
        probe = jnp.zeros((1, dimension), dtype=jnp.float64)
        closed = jax.make_jaxpr(jax.vmap(function))(probe)
        if len(closed.jaxpr.outvars) != 1:
            raise ValueError(
                "CoupledApproximation.program must return one array; got multiple outputs"
            )
        output_aval = closed.jaxpr.outvars[0].aval
        if not isinstance(output_aval, ShapedArray):
            raise ValueError(
                "CoupledApproximation.program must return a shaped array; got an unsupported output abstract value"
            )
        shape = tuple(output_aval.shape[1:])
        prepared = PreparedIntervalFunction(
            closed, dimension, shape, 1, tuple(coefficients)
        )
    finally:
        budget.release(16 * dimension)
    result = _IntervalMap(prepared, budget)
    result.evaluate((np.zeros(dimension), np.ones(dimension)))
    return result


class _BatchMap:
    """Adapt a counted one-box interpreter to the interval batch protocol."""

    def __init__(self, interval_map: _IntervalMap) -> None:
        self.interval_map = interval_map

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if lower.shape[0] != 1:
            raise ValueError(
                f"CoupledApproximation.boxes must contain one interval box; got {lower.shape[0]} boxes"
            )
        result = self.interval_map.evaluate((lower[0], upper[0]))
        return result[0][None], result[1][None]


class _SourceJets:
    """Account for numerical work in the implicit source graph."""

    def __init__(self, source: IntersectionCurve, budget: _Budget) -> None:
        self.source = source

        def observe_point_jet() -> None:
            budget.jets += 1

        self.preparation = IntersectionJetPreparation(
            source,
            prepare=lambda function, dimension, coefficients: _BatchMap(
                _map(function, dimension, coefficients, budget)
            ),
            wrap=lambda prepared: _BatchMap(
                _IntervalMap(prepared, budget, borrowed=True)
            ),
            observe_point_jet=observe_point_jet,
        )

    def enclosure(
        self,
        chart: int,
        first: float,
        last: float,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    chart,
                    first,
                    last,
                )
            ) as budget:
                return self.enclosure(
                    chart, first, last, preparations=preparations, budget=budget
                )
        return self.source.parameter_enclosures(
            first,
            last,
            minimum_chart=chart,
            maximum_chart=chart,
            _preparation=self.preparation,
            preparations=preparations,
            budget=budget,
        )[0]

    def value(self, chart: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        system = self.preparation.pairs[0][int(self.preparation.pairs[1][chart])]
        prepared = self.preparation.program(
            chart, "spatial-value", system.first.evaluate, 2
        )
        result = prepared.evaluate(box[None, 0, :2], box[None, 1, :2])
        return result[0][0], result[1][0]

    def jet(
        self,
        chart: int,
        first: float,
        last: float,
        order: int,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    chart,
                    first,
                    last,
                    order,
                )
            ) as budget:
                return self.jet(
                    chart, first, last, order, preparations=preparations, budget=budget
                )
        return intersection_derivative_bounds(
            self.source,
            first,
            last,
            order,
            "all",
            minimum_chart=chart,
            maximum_chart=chart,
            _preparation=self.preparation,
            preparations=preparations,
            budget=budget,
        )


class _SplineJets:
    def __init__(self, evaluator: BernsteinCurvePiece, budget: _Budget) -> None:
        self.piece = bernstein_rational_piece(evaluator)
        self.budget = budget

    def jet(self, first: float, last: float, order: int) -> tuple[np.ndarray, np.ndarray]:
        self.budget.jets += 1
        return rational_piece_jet_bounds(self.piece, first, last, order=order)


def _taylor_bound(
    value: tuple[np.ndarray, np.ndarray],
    first: tuple[np.ndarray, np.ndarray],
    center_first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    radius: float,
) -> np.ndarray:
    magnitude = np.maximum(np.abs(value[0]), np.abs(value[1]))
    derivative = np.maximum(np.abs(first[0]), np.abs(first[1]))
    center_derivative = np.maximum(np.abs(center_first[0]), np.abs(center_first[1]))
    curvature = np.maximum(np.abs(second[0]), np.abs(second[1]))
    linear = np.nextafter(magnitude + np.nextafter(derivative * radius, np.inf), np.inf)
    radius_squared = np.nextafter(radius * radius, np.inf)
    quadratic_radius = np.nextafter(0.5 * radius_squared, np.inf)
    quadratic = np.nextafter(
        magnitude
        + np.nextafter(
            np.nextafter(center_derivative * radius, np.inf)
            + np.nextafter(curvature * quadratic_radius, np.inf),
            np.inf,
        ),
        np.inf,
    )
    # A failed second jet must not destroy a finite first-order proof.
    quadratic = np.where(np.isnan(quadratic), np.inf, quadratic)
    return np.minimum(linear, quadratic)


def _reserve_bernstein_source(
    item: BSplineCurve | BSplineSurfacePatch, budget: _Budget
) -> None:
    """Preflight knot refinement and its simultaneously live coefficient arrays."""
    if isinstance(item, BSplineCurve):
        knots, degrees = (item.knots,), (item.degree,)
    else:
        knots, degrees = (item.u_knots, item.v_knots), (item.u_degree, item.v_degree)
    refined = 1
    knot_storage = 0
    for vector, degree in zip(knots, degrees, strict=True):
        host = np.asarray(vector)
        spans = sum(float(a) < float(b) for a, b in zip(host[:-1], host[1:], strict=True))
        refined *= max(1, spans * (degree + 1))
        knot_storage += 8 * int(vector.size) * (degree + 1)
    refined = max(math.prod(item.control_points.shape[:-1]), refined)
    budget.reserve(
        12 * refined * (int(item.control_points.shape[-1]) + 1) * 8 + knot_storage,
        "constructing source-enclosed Bernstein pieces",
    )


type _Interval = tuple[np.ndarray, np.ndarray]


def _subtract_array_intervals(first: _Interval, second: _Interval, /) -> _Interval:
    """Subtract tensor-valued source jets and retain ndarray result storage."""
    lower, upper = interval_subtract(first, second)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


class _ApproximationWorkset(StrictModule):
    segments: tuple[tuple[CurvePiece[AbstractCurve], ...], ...]
    source_jets: _SourceJets
    pending: list[tuple[float, float, int]]
    pending_bytes: int
    record_bytes: int


class _ApproximationCell(StrictModule):
    first: float
    last: float
    center: float
    radius: float
    selected: tuple[CurvePiece[AbstractCurve], ...]
    splines: tuple[_SplineJets, ...]
    values: tuple[_Interval, ...]
    distance: float
    parameter: float


def _prepare_approximation_workset(
    source: IntersectionCurve,
    fitted: tuple[BSplineCurve, ...],
    budget: _Budget,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _ApproximationWorkset:
    # Fixed cell arithmetic and source contraction arrays are at most 7D.
    # Interpreter live buffers are reserved separately by _IntervalMap.
    budget.reserve(65536, "allocating the fixed coupled cell workspace")
    if source.num_charts > budget.maximum_cells:
        budget.fail(
            "maximum_cells", "source chart splitting already exceeds the cell budget"
        )
    split_capacity = source.num_charts + 1 + sum(item.knots.size for item in fitted)
    budget.reserve(
        128 * split_capacity, "retaining full-domain chart/knot split coordinates"
    )
    for item in fitted:
        _reserve_bernstein_source(item, budget)
    for region in (source.first, source.second):
        for leaf in jax.tree_util.tree_leaves(
            region.patch,
            is_leaf=lambda value: isinstance(value, (BSplineCurve, BSplineSurfacePatch)),
        ):
            match leaf:
                case BSplineCurve() | BSplineSurfacePatch():
                    _reserve_bernstein_source(leaf, budget)
                case Array() | np.ndarray() | np.generic():
                    budget.reserve(
                        12 * leaf.nbytes,
                        "retaining native source evaluator coefficient storage",
                    )
                case _:
                    pass
    domain = source.parameter_domain
    segments = tuple(
        curve_pieces_for_interval(item, *domain, coordinate_budget=coordinate_budget)
        for item in fitted
    )
    source_jets = _SourceJets(source, budget)
    breaks = {
        domain[0],
        domain[1],
        *(float(index) for index in range(source.num_charts + 1)),
    }
    for pieces in segments:
        breaks.update(piece.lower for piece in pieces)
        breaks.update(piece.upper for piece in pieces)
    ordered = sorted(breaks)
    # Include tuple/scalar storage, list pointers and list-growth allowance.
    pending_bytes = (
        sys.getsizeof((0.0, 0.0, 0)) + 2 * sys.getsizeof(0.0) + sys.getsizeof(0) + 16
    )
    record_bytes = sys.getsizeof((0.0,) * 8) + 8 * sys.getsizeof(0.0) + 16
    initial_cells = len(ordered) - 1
    if initial_cells > budget.maximum_cells:
        budget.fail(
            "maximum_cells", "chart/knot splitting already exceeds the cell budget"
        )
    budget.reserve(initial_cells * pending_bytes, "retaining chart/knot split cells")
    pending = [
        (ordered[index], ordered[index + 1], 0)
        for index in range(initial_cells - 1, -1, -1)
    ]
    return _ApproximationWorkset(
        segments, source_jets, pending, pending_bytes, record_bytes
    )


def _combine_jets(jets: list[_Interval]) -> _Interval:
    return np.concatenate([jet[0] for jet in jets]), np.concatenate(
        [jet[1] for jet in jets]
    )


def _enclose_approximation_cell(
    workset: _ApproximationWorkset,
    first: float,
    last: float,
    spline_jets: dict[tuple[int, int], _SplineJets],
    budget: _Budget,
    distance_tolerance: float,
    parameter_tolerance: float,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _ApproximationCell:
    center = first + 0.5 * (last - first)
    radius = float(np.nextafter(max(center - first, last - center), np.inf))
    source_prepared = workset.source_jets
    chart = min(int(math.floor(center)), source_prepared.source.num_charts - 1)
    center_box = source_prepared.enclosure(
        chart, center, center, preparations=preparations, budget=coordinate_budget
    )
    physical_center = source_prepared.value(chart, center_box)
    source_value = (
        np.concatenate((physical_center[0], center_box[0])),
        np.concatenate((physical_center[1], center_box[1])),
    )
    source_first = source_prepared.jet(
        chart, first, last, 1, preparations=preparations, budget=coordinate_budget
    )
    source_second = source_prepared.jet(
        chart, first, last, 2, preparations=preparations, budget=coordinate_budget
    )
    source_center_first = source_prepared.jet(
        chart, center, center, 1, preparations=preparations, budget=coordinate_budget
    )
    selected = tuple(
        next(piece for piece in pieces if piece.lower <= center <= piece.upper)
        for pieces in workset.segments
    )
    prepared_splines = []
    for side, piece in enumerate(selected):
        key = (side, piece.index)
        if key not in spline_jets:
            match piece.evaluator:
                case BernsteinCurvePiece() as evaluator:
                    spline_jets[key] = _SplineJets(evaluator, budget)
                case _:
                    raise ValueError(
                        "CoupledApproximation.spline_pieces must retain rational Bernstein controls; got an unsupported fitted evaluator"
                    )
        prepared_splines.append(spline_jets[key])
    values, firsts, center_firsts, seconds = [], [], [], []
    for prepared in prepared_splines:
        values.append(prepared.jet(center, center, 0))
        firsts.append(prepared.jet(first, last, 1))
        center_firsts.append(prepared.jet(center, center, 1))
        seconds.append(prepared.jet(first, last, 2))
    residual = _subtract_array_intervals(source_value, _combine_jets(values))
    for kind, columns, tolerance in (
        ("distance", slice(0, 3), distance_tolerance),
        ("first parameter", slice(3, 5), parameter_tolerance),
        ("second parameter", slice(5, 7), parameter_tolerance),
    ):
        lower_bound = _norm_lower((residual[0][columns], residual[1][columns]))
        if lower_bound > tolerance:
            raise ApproximationDeviationError(
                kind, (first, last), lower_bound, tolerance, budget
            )
    derivative = _subtract_array_intervals(source_first, _combine_jets(firsts))
    center_derivative = _subtract_array_intervals(
        source_center_first, _combine_jets(center_firsts)
    )
    second_derivative = _subtract_array_intervals(source_second, _combine_jets(seconds))
    errors = _taylor_bound(
        residual, derivative, center_derivative, second_derivative, radius
    )
    return _ApproximationCell(
        first,
        last,
        center,
        radius,
        selected,
        tuple(prepared_splines),
        tuple(values),
        norm_upper(errors[:3]),
        max(norm_upper(errors[3:5]), norm_upper(errors[5:7])),
    )


def _enclose_surface_lift(
    source: IntersectionCurve,
    side: int,
    cell: _ApproximationCell,
    correspondence: dict[tuple[int, int, int, int], RationalCurveSurfaceJets],
    budget: _Budget,
    distance_tolerance: float,
    preparations: TrimRequestPreparations,
    coordinate_budget: CoordinateEnclosureBudget,
) -> float:
    region = (source.first, source.second)[side]
    uv_piece = cell.selected[side + 1]
    uv_box = uv_piece.evaluator.bounding_box(
        cell.first, cell.last, preparations=preparations, budget=coordinate_budget
    )
    # Polynomial continuations cover every intersected surface piece, including
    # knot crossings whose preimage has no representable scalar split.
    patches = surface_pieces_for_box(region.patch, uv_box, budget=coordinate_budget)
    if not patches:
        return np.inf
    bound = 0.0
    for patch in patches:
        key = (side, cell.selected[0].index, uv_piece.index, patch.index)
        if key not in correspondence:

            def observe_curve_jet() -> None:
                budget.jets += 1

            correspondence[key] = RationalCurveSurfaceJets(
                cell.splines[0].piece,
                cell.splines[side + 1].piece,
                patch.evaluator,
                point=np.zeros(3),
                prepare=lambda function, dimension, coefficients: _BatchMap(
                    _map(function, dimension, coefficients, budget)
                ),
                observe_curve_jet=observe_curve_jet,
            )
        maps = correspondence[key]
        value = maps.jet(
            cell.center,
            cell.center,
            0,
            preparations=preparations,
            budget=coordinate_budget,
        )
        center_uv = cell.values[side + 1]
        if np.all(center_uv[0] >= patch.lower) and np.all(center_uv[1] <= patch.upper):
            lower_bound = _norm_lower(value)
            if lower_bound > distance_tolerance:
                raise ApproximationDeviationError(
                    kind=f"{('first', 'second')[side]} correspondence",
                    interval=(cell.first, cell.last),
                    lower_bound=lower_bound,
                    tolerance=distance_tolerance,
                    budget=budget,
                )
        d1 = maps.jet(
            cell.first,
            cell.last,
            1,
            preparations=preparations,
            budget=coordinate_budget,
        )
        cd1 = maps.jet(
            cell.center,
            cell.center,
            1,
            preparations=preparations,
            budget=coordinate_budget,
        )
        d2 = maps.jet(
            cell.first,
            cell.last,
            2,
            preparations=preparations,
            budget=coordinate_budget,
        )
        bound = max(bound, norm_upper(_taylor_bound(value, d1, cd1, d2, cell.radius)))
    return bound


def _freeze_approximation_evidence(
    records: list[tuple[float, ...]], maxima: np.ndarray, budget: _Budget
) -> CoupledApproximationEvidence:
    budget.reserve(2 * len(records) * 8 * 8, "freezing all continuous interval evidence")
    # Immutable bytes backing prevents callers from re-enabling writeability.
    evidence = np.frombuffer(
        np.asarray(records, dtype=np.float64).tobytes(), dtype=np.float64
    ).reshape((-1, 8))
    return CoupledApproximationEvidence(
        float(maxima[0]),
        float(maxima[1]),
        float(maxima[2]),
        float(maxima[3]),
        budget.cells,
        budget.depth,
        evidence,
        budget.jets,
        budget.bytes,
        budget.peak,
    )


@checked
def certify_coupled_approximation(
    source: IntersectionCurve,
    curve: BSplineCurve,
    first_pcurve: BSplineCurve,
    second_pcurve: BSplineCurve,
    *,
    distance_tolerance: float,
    parameter_tolerance: float,
    maximum_cells: int,
    maximum_depth: int,
    maximum_bytes: int,
    preparations: TrimRequestPreparations | None = None,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> CoupledApproximationEvidence:
    """Certify all three fitted curves and both native surface lifts continuously.

    The parameter bound is the maximum Euclidean UV error of the two p-curves;
    all three physical bounds must meet the supplied physical distance tolerance.
    Parameter domains must be identical: no reparameterization, period wrapping,
    source identity change or tolerance relaxation is performed.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if coordinate_budget is None:
        with coordinate_source_request(
            (
                source,
                curve,
                first_pcurve,
                second_pcurve,
            )
        ) as coordinate_budget:
            return certify_coupled_approximation(
                source,
                curve,
                first_pcurve,
                second_pcurve,
                distance_tolerance=distance_tolerance,
                parameter_tolerance=parameter_tolerance,
                maximum_cells=maximum_cells,
                maximum_depth=maximum_depth,
                maximum_bytes=maximum_bytes,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
    if not source.fully_certified:
        raise ValueError(
            "CoupledApproximation.source must be a fully certified IntersectionCurve; got an uncertified source"
        )
    fitted = (curve, first_pcurve, second_pcurve)
    domain = source.parameter_domain
    if any(item.parameter_domain != domain for item in fitted):
        raise ValueError(
            "CoupledApproximation.fitted_domains must retain the source parameter domain; got mismatched fitted domains"
        )
    if tuple(item.ambient_dimension for item in fitted) != (3, 2, 2):
        raise ValueError(
            "CoupledApproximation.fitted_dimensions must be (3, 2, 2); got incompatible carrier dimensions"
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (distance_tolerance, parameter_tolerance)
    ):
        raise ValueError(
            f"CoupledApproximation.tolerances must be finite and positive; got distance={distance_tolerance}, parameter={parameter_tolerance}"
        )
    budget = _Budget(
        positive_integer(maximum_cells, "maximum_cells"),
        nonnegative_integer(maximum_depth, "maximum_depth"),
        positive_integer(maximum_bytes, "maximum_bytes"),
    )
    workset = _prepare_approximation_workset(source, fitted, budget, coordinate_budget)
    pending = workset.pending
    spline_jets: dict[tuple[int, int], _SplineJets] = {}
    correspondence: dict[tuple[int, int, int, int], RationalCurveSurfaceJets] = {}
    records: list[tuple[float, ...]] = []
    maxima = np.zeros(4)
    while pending:
        first, last, depth = pending.pop()
        budget.release(workset.pending_bytes)
        budget.cell(depth)
        cell = _enclose_approximation_cell(
            workset,
            first,
            last,
            spline_jets,
            budget,
            distance_tolerance,
            parameter_tolerance,
            preparations,
            coordinate_budget,
        )
        lift_bounds = tuple(
            _enclose_surface_lift(
                source,
                side,
                cell,
                correspondence,
                budget,
                distance_tolerance,
                preparations,
                coordinate_budget,
            )
            for side in (0, 1)
        )
        bounds = (cell.distance, cell.parameter, *lift_bounds)
        accepted = (
            all(math.isfinite(value) for value in bounds)
            and cell.distance <= distance_tolerance
            and cell.parameter <= parameter_tolerance
            and max(lift_bounds) <= distance_tolerance
        )
        budget.reserve(
            workset.record_bytes, "retaining another continuous interval evidence record"
        )
        records.append((first, last, float(depth), *bounds, float(accepted)))
        if accepted:
            maxima = np.maximum(maxima, bounds)
            continue
        if depth >= maximum_depth:
            budget.fail(
                "maximum_depth",
                f"unproved interval [{first}, {last}] has continuous bounds {bounds}",
            )
        if not first < cell.center < last:
            raise ValueError(
                f"CoupledApproximation.bounds must meet their tolerances; got bounds {bounds} at machine-resolved interval [{first}, {last}]"
            )
        if budget.cells + len(pending) + 2 > maximum_cells:
            budget.fail(
                "maximum_cells", f"subdivision required by continuous bounds {bounds}"
            )
        budget.reserve(
            2 * workset.pending_bytes, "subdividing an unproved continuous interval"
        )
        pending.extend(((cell.center, last, depth + 1), (first, cell.center, depth + 1)))
    return _freeze_approximation_evidence(records, maxima, budget)


__all__ = [
    "ApproximationDeviationError",
    "ApproximationResourceError",
    "CoupledApproximationEvidence",
    "certify_coupled_approximation",
]
