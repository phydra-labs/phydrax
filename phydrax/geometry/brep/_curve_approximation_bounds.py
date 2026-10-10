#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous, coupled external approximation bounds, never sample receipts.

The source is the original certified implicit graph.  Bernstein coefficient
intervals enclose the fitted splines.  Each closed chart/knot cell uses a
midpoint *enclosure*, first difference jets and a second difference remainder.
Native surface correspondence is proved separately for both fitted UV curves.
Arithmetic uses the canonical outward-rounded interval interpreter (IEEE basic
operations and its explicitly conditional four-ulp transcendental premise).

``maximum_bytes`` bounds proof-owned numeric storage, retained cell records and
interval-interpreter numeric workspace; it is not a Python/JAX process RSS limit.
Source definitions and the runtime's existing compilation caches are borrowed,
not duplicated or changed.  There are no numerical source-root evaluations here.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray

from .._interval_enclosure import interval_subtract, PreparedIntervalFunction
from ._correspondence import (
    _bernstein_rational_piece,
    _norm_upper,
    _RationalCurveSurfaceJets,
)
from ._intersection_curve import (
    _intersection_derivative_bounds,
    _IntersectionJetPreparation,
    BernsteinCurvePiece,
    curve_pieces_for_interval,
    IntersectionCurve,
    surface_pieces_for_box,
)
from ._patches import _rational_piece_jet_bounds, BSplineCurve, BSplineSurfacePatch


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
            f"Coupled approximation {resource} exhausted: {cause}; cells={cells}, depth={depth}, bytes={bytes_used}."
        )


class ApproximationDeviationError(ValueError):
    """A source-enclosed point witnesses genuine geometric out-of-tolerance."""

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
            f"Certified {kind} deviation at interval {interval}: lower bound {lower_bound} > tolerance {tolerance}."
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
    constants: dict[int, Any] = {}
    total = largest = 8

    def visit(closed: Any) -> None:
        nonlocal total, largest
        graph = getattr(closed, "jaxpr", closed)
        for value in getattr(closed, "consts", ()):
            constants[id(value)] = value
        variables = list(graph.constvars) + list(graph.invars)
        for equation in graph.eqns:
            variables.extend(equation.outvars)
            for value in equation.params.values():
                if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                    visit(value)
                elif isinstance(value, (tuple, list)):
                    for child in value:
                        if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                            visit(child)
        for variable in variables:
            aval = getattr(variable, "aval", None)
            if aval is not None and hasattr(aval, "shape"):
                size = math.prod(aval.shape) * np.dtype(aval.dtype).itemsize
                total += 2 * size
                largest = max(largest, size)

    visit(prepared.closed_jaxpr)
    for triplet in prepared.constant_bounds:
        for value in triplet:
            constants[id(value)] = value
    retained = sum(int(getattr(value, "nbytes", 0)) for value in constants.values())
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
    function: Any, dimension: int, coefficients: Any, budget: _Budget
) -> _IntervalMap:
    # Build the canonical prepared interpreter without its eager validation
    # allocation; validation is performed through the bounded evaluator below.
    # Tracing metadata is owned by JAX, not numeric interval proof workspace.
    budget.reserve(16 * dimension, "tracing one fixed-size interval input")
    try:
        probe = jnp.zeros((1, dimension), dtype=jnp.float64)
        closed = jax.make_jaxpr(jax.vmap(function))(probe)
        if len(closed.jaxpr.outvars) != 1:
            raise ValueError("An approximation interval program must return one array.")
        output_aval = closed.jaxpr.outvars[0].aval
        if not isinstance(output_aval, ShapedArray):
            raise ValueError(
                "An approximation interval program must return a shaped array."
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
    """Adapt a counted one-box interpreter to the canonical batch protocol."""

    def __init__(self, interval_map: _IntervalMap) -> None:
        self.interval_map = interval_map

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if lower.shape[0] != 1:
            raise ValueError("Coupled graph preparation owns exactly one interval box.")
        result = self.interval_map.evaluate((lower[0], upper[0]))
        return result[0][None], result[1][None]


class _SourceJets:
    """Resource adapter to the sole canonical implicit graph implementation."""

    def __init__(self, source: IntersectionCurve, budget: _Budget) -> None:
        self.source = source

        def observe_point_jet() -> None:
            budget.jets += 1

        self.preparation = _IntersectionJetPreparation(
            source,
            prepare=lambda function, dimension, coefficients: _BatchMap(
                _map(function, dimension, coefficients, budget)
            ),
            wrap=lambda prepared: _BatchMap(
                _IntervalMap(prepared, budget, borrowed=True)
            ),
            observe_point_jet=observe_point_jet,
        )

    def enclosure(self, chart: int, first: float, last: float) -> np.ndarray:
        return self.source.parameter_enclosures(
            first,
            last,
            minimum_chart=chart,
            maximum_chart=chart,
            _preparation=self.preparation,
        )[0]

    def value(self, chart: int, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        system = self.preparation.pairs[0][int(self.preparation.pairs[1][chart])]
        prepared = self.preparation.program(
            chart, "spatial-value", system.first.evaluate, 2
        )
        result = prepared.evaluate(box[None, 0, :2], box[None, 1, :2])
        return result[0][0], result[1][0]

    def jet(
        self, chart: int, first: float, last: float, order: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return _intersection_derivative_bounds(
            self.source,
            first,
            last,
            order,
            "all",
            minimum_chart=chart,
            maximum_chart=chart,
            _preparation=self.preparation,
        )


class _SplineJets:
    def __init__(self, evaluator: BernsteinCurvePiece, budget: _Budget) -> None:
        self.piece = _bernstein_rational_piece(evaluator)
        self.budget = budget

    def jet(self, first: float, last: float, order: int) -> tuple[np.ndarray, np.ndarray]:
        self.budget.jets += 1
        return _rational_piece_jet_bounds(self.piece, first, last, order=order)


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


def _integer(value: int, name: str, minimum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


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
) -> CoupledApproximationEvidence:
    """Certify all three fitted curves and both native surface lifts continuously.

    The parameter bound is the maximum Euclidean UV error of the two p-curves;
    all three physical bounds must meet the supplied physical distance tolerance.
    Parameter domains must be identical: no reparameterization, period wrapping,
    source identity change or tolerance relaxation is performed.
    """
    if not isinstance(source, IntersectionCurve) or not source.fully_certified:
        raise ValueError(
            "A continuous approximation requires the original fully certified IntersectionCurve."
        )
    fitted = (curve, first_pcurve, second_pcurve)
    if any(not isinstance(item, BSplineCurve) for item in fitted):
        raise ValueError("All fitted carriers must be BSplineCurve instances.")
    domain = source.parameter_domain
    if any(item.parameter_domain != domain for item in fitted):
        raise ValueError(
            "All fitted carriers must retain the full original source parameter domain."
        )
    if tuple(item.ambient_dimension for item in fitted) != (3, 2, 2):
        raise ValueError("Fitted carriers must have dimensions (3, 2, 2).")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (distance_tolerance, parameter_tolerance)
    ):
        raise ValueError("Physical and parameter tolerances must be finite and positive.")
    budget = _Budget(
        _integer(maximum_cells, "maximum_cells", 1),
        _integer(maximum_depth, "maximum_depth", 0),
        _integer(maximum_bytes, "maximum_bytes", 1),
    )
    # Fixed cell arithmetic, source contraction arrays and output jets. All are
    # at most 7D; the interval interpreter's larger buffers are reserved separately.
    budget.reserve(65536, "allocating the fixed coupled cell workspace")
    # Preflight Bernstein refinement BEFORE constructing it. Knot insertion can
    # refine at most (degree+1) controls per nonempty span. Three coefficient
    # arrays and both simultaneous refinement passes are covered by factor 12.
    if source.num_charts > maximum_cells:
        budget.fail(
            "maximum_cells", "source chart splitting already exceeds the cell budget"
        )
    split_capacity = source.num_charts + 1 + sum(int(item.knots.size) for item in fitted)
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
            if isinstance(leaf, (BSplineCurve, BSplineSurfacePatch)):
                _reserve_bernstein_source(leaf, budget)
            elif hasattr(leaf, "nbytes"):
                budget.reserve(
                    12 * int(leaf.nbytes),
                    "retaining native source evaluator coefficient storage",
                )
    segments = tuple(curve_pieces_for_interval(item, *domain) for item in fitted)
    source_prepared = _SourceJets(source, budget)
    spline_jets: dict[tuple[int, int], _SplineJets] = {}
    correspondence: dict[tuple[int, int, int, int], _RationalCurveSurfaceJets] = {}
    breaks = {
        domain[0],
        domain[1],
        *(float(index) for index in range(source.num_charts + 1)),
    }
    for pieces in segments:
        breaks.update(piece.lower for piece in pieces)
        breaks.update(piece.upper for piece in pieces)
    ordered = sorted(breaks)
    # Python record sizes include floats, tuple storage, list pointer and a
    # conservative list-growth allowance. Final immutable numeric copies are
    # reserved separately before allocation.
    pending_bytes = (
        sys.getsizeof((0.0, 0.0, 0)) + 2 * sys.getsizeof(0.0) + sys.getsizeof(0) + 16
    )
    record_bytes = sys.getsizeof((0.0,) * 8) + 8 * sys.getsizeof(0.0) + 16
    initial_cells = len(ordered) - 1
    if initial_cells > maximum_cells:
        budget.fail(
            "maximum_cells", "chart/knot splitting already exceeds the cell budget"
        )
    budget.reserve(initial_cells * pending_bytes, "retaining chart/knot split cells")
    # Traverse the represented domain in increasing order, including children.
    pending = [
        (ordered[index], ordered[index + 1], 0)
        for index in range(initial_cells - 1, -1, -1)
    ]
    records: list[tuple[float, ...]] = []
    maxima = np.zeros(4)
    while pending:
        first, last, depth = pending.pop()
        budget.release(pending_bytes)
        budget.cell(depth)
        center = first + 0.5 * (last - first)
        radius = float(np.nextafter(max(center - first, last - center), np.inf))
        chart = min(int(math.floor(center)), source.num_charts - 1)
        center_box = source_prepared.enclosure(chart, center, center)
        physical_center = source_prepared.value(chart, center_box)
        source_value = (
            np.concatenate((physical_center[0], center_box[0])),
            np.concatenate((physical_center[1], center_box[1])),
        )
        source_first = source_prepared.jet(chart, first, last, 1)
        source_second = source_prepared.jet(chart, first, last, 2)
        source_center_first = source_prepared.jet(chart, center, center, 1)
        selected = tuple(
            next(piece for piece in pieces if piece.lower <= center <= piece.upper)
            for pieces in segments
        )
        prepared_splines = []
        for side, piece in enumerate(selected):
            key = (side, piece.index)
            if key not in spline_jets:
                if not isinstance(piece.evaluator, BernsteinCurvePiece):
                    raise ValueError(
                        "Fitted spline pieces must retain canonical rational Bernstein controls."
                    )
                spline_jets[key] = _SplineJets(piece.evaluator, budget)
            prepared_splines.append(spline_jets[key])
        values, firsts, center_firsts, seconds = [], [], [], []
        for prepared in prepared_splines:
            values.append(prepared.jet(center, center, 0))
            firsts.append(prepared.jet(first, last, 1))
            center_firsts.append(prepared.jet(center, center, 1))
            seconds.append(prepared.jet(first, last, 2))

        def combine(
            jets: list[tuple[np.ndarray, np.ndarray]],
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.concatenate([jet[0] for jet in jets]), np.concatenate(
                [jet[1] for jet in jets]
            )

        residual = interval_subtract(source_value, combine(values))
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
        derivative = interval_subtract(source_first, combine(firsts))
        center_derivative = interval_subtract(source_center_first, combine(center_firsts))
        assert source_second is not None
        second_derivative = interval_subtract(source_second, combine(seconds))
        errors = _taylor_bound(
            residual, derivative, center_derivative, second_derivative, radius
        )
        distance = _norm_upper(errors[:3])
        parameter = max(_norm_upper(errors[3:5]), _norm_upper(errors[5:7]))
        lift_bounds = []
        for side, region in enumerate((source.first, source.second)):
            uv_piece = selected[side + 1]
            uv_box = uv_piece.evaluator.bounding_box(first, last)
            # Each intersected native piece's polynomial continuation encloses
            # its residual on the scalar subset actually owned by that piece.
            # Taking all pieces also covers a surface-knot crossing whose exact
            # preimage is not a representable scalar split.
            patches = surface_pieces_for_box(region.patch, uv_box)
            if not patches:
                lift_bounds.append(np.inf)
                continue
            bound = 0.0
            for patch in patches:
                key = (side, selected[0].index, uv_piece.index, patch.index)
                if key not in correspondence:

                    def observe_curve_jet() -> None:
                        budget.jets += 1

                    correspondence[key] = _RationalCurveSurfaceJets(
                        prepared_splines[0].piece,
                        prepared_splines[side + 1].piece,
                        patch.evaluator,
                        point=np.zeros(3),
                        prepare=lambda function, dimension, coefficients: _BatchMap(
                            _map(function, dimension, coefficients, budget)
                        ),
                        observe_curve_jet=observe_curve_jet,
                    )
                maps = correspondence[key]
                value = maps.jet(center, center, 0)
                center_uv = values[side + 1]
                if np.all(center_uv[0] >= patch.lower) and np.all(
                    center_uv[1] <= patch.upper
                ):
                    lower_bound = _norm_lower(value)
                    if lower_bound > distance_tolerance:
                        raise ApproximationDeviationError(
                            f"{('first', 'second')[side]} correspondence",
                            (first, last),
                            lower_bound,
                            distance_tolerance,
                            budget,
                        )
                d1 = maps.jet(first, last, 1)
                cd1 = maps.jet(center, center, 1)
                d2 = maps.jet(first, last, 2)
                bound = max(bound, _norm_upper(_taylor_bound(value, d1, cd1, d2, radius)))
            lift_bounds.append(bound)
        bounds = (distance, parameter, *lift_bounds)
        accepted = (
            all(math.isfinite(value) for value in bounds)
            and distance <= distance_tolerance
            and parameter <= parameter_tolerance
            and max(lift_bounds) <= distance_tolerance
        )
        budget.reserve(
            record_bytes, "retaining another continuous interval evidence record"
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
        if not first < center < last:
            raise ValueError(
                f"Continuous coupled bounds {bounds} remain out of tolerance at machine-resolved interval [{first}, {last}]."
            )
        if budget.cells + len(pending) + 2 > maximum_cells:
            budget.fail(
                "maximum_cells", f"subdivision required by continuous bounds {bounds}"
            )
        budget.reserve(2 * pending_bytes, "subdividing an unproved continuous interval")
        pending.extend(((center, last, depth + 1), (first, center, depth + 1)))
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


__all__ = [
    "ApproximationDeviationError",
    "ApproximationResourceError",
    "CoupledApproximationEvidence",
    "certify_coupled_approximation",
]
