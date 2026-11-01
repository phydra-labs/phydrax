#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from math import tau
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._interpolation._bspline_extraction import (
    _exact_scalar_interval,
    _tensor_bernstein_derivative,
    _tensor_bernstein_linear_combination,
    _tensor_bernstein_product,
    RationalBSplineSurfaceSpan,
)
from ..._strict import StrictModule
from ...discretization._coordinate_enclosure import ExactScalar
from .._interval_enclosure import (
    HostArray,
    Interval,
    interval_add,
    interval_divide,
    interval_multiply,
    interval_sqrt,
    interval_subtract,
    prepare_interval_function,
)
from .._rational import fraction_down, fraction_up
from ._intersection import (
    ParametricIntersectionPolicy,
    SurfaceIntersectionResult,
    UnresolvedIntersectionReason,
    UnresolvedParameterRegion,
)
from ._intersection_curve import IntersectionCurve, SurfacePairSystem, SurfaceRegion
from ._intersection_kernel import krawczyk, KrawczykEvaluator, KrawczykResult
from ._patches import (
    AbstractSurfacePatch,
    BSplineSurfacePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
)
from ._tspline import TSplineSurfacePatch


class _SplineSource:
    """Exact rational Bernstein span enclosures of a NURBS or T-spline source.

    Spans on the parameter-domain boundary are continued by their own rational
    Bézier polynomial, so root boxes may straddle the boundary; enclosures of
    a continuation with a nonpositive denominator are unbounded.
    """

    def __init__(self, patch: TSplineSurfacePatch | BSplineSurfacePatch, /) -> None:
        spans = patch.prepare_spans().spans
        if not spans:
            raise ValueError(
                "A spline source must extract exact rational Bernstein spans."
            )
        domain = tuple(
            (
                min(span.parameter_bounds[axis][0] for span in spans),
                max(span.parameter_bounds[axis][1] for span in spans),
            )
            for axis in range(2)
        )
        self.patch = patch
        self.spans = spans
        self.nets = {id(span): _SpanQuotientNets(span) for span in spans}
        self.reach = {
            id(span): tuple(
                (
                    None
                    if span.parameter_bounds[axis][0] == domain[axis][0]
                    else span.parameter_bounds[axis][0],
                    None
                    if span.parameter_bounds[axis][1] == domain[axis][1]
                    else span.parameter_bounds[axis][1],
                )
                for axis in range(2)
            )
            for span in spans
        }

    def breakpoints(self, axis: int, /) -> tuple[float, ...]:
        """Exact span boundaries along one parameter axis, as binary64 values."""
        return tuple(
            sorted(
                {
                    float(bound)
                    for span in self.spans
                    for bound in span.parameter_bounds[axis]
                }
            )
        )

    def _clipped_spans(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[tuple[RationalBSplineSurfaceSpan, _ExactBox], ...]:
        box = tuple(
            (
                Fraction.from_float(float(lower[axis])),
                Fraction.from_float(float(upper[axis])),
            )
            for axis in range(2)
        )
        clipped: list[tuple[RationalBSplineSurfaceSpan, _ExactBox]] = []
        for span in self.spans:
            bounds = tuple(
                (
                    box[axis][0] if first is None else max(box[axis][0], first),
                    box[axis][1] if last is None else min(box[axis][1], last),
                )
                for axis, (first, last) in enumerate(self.reach[id(span)])
            )
            if all(first <= last for first, last in bounds):
                clipped.append((span, (bounds[0], bounds[1])))
        if not clipped:
            raise ValueError(
                "A spline interval box does not overlap its exact Bézier partition."
            )
        return tuple(clipped)

    def values(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Outward hull of the projected Bernstein controls restricted to a box.

        With positive weights a rational Bézier patch is a convex combination of
        its projected control points, so the restricted control hull encloses
        the patch with second-order excess in the box width.
        """
        value_lower: np.ndarray | None = None
        value_upper: np.ndarray | None = None
        for span, bounds in self._clipped_spans(lower, upper):
            values = self.nets[id(span)].values(bounds)
            if value_lower is None or value_upper is None:
                value_lower, value_upper = values
            else:
                value_lower = np.minimum(value_lower, values[0])
                value_upper = np.maximum(value_upper, values[1])
        if value_lower is None or value_upper is None:
            raise RuntimeError("Clipped spline spans must be nonempty.")
        return value_lower, value_upper

    def jacobians(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """Outward global-parameter Jacobian bounds over a parameter box."""
        jacobian_lower: np.ndarray | None = None
        jacobian_upper: np.ndarray | None = None
        for span, bounds in self._clipped_spans(lower, upper):
            jac_lo, jac_hi = self.nets[id(span)].jacobians(bounds)
            if jacobian_lower is None or jacobian_upper is None:
                jacobian_lower, jacobian_upper = jac_lo, jac_hi
            else:
                jacobian_lower = np.minimum(jacobian_lower, jac_lo)
                jacobian_upper = np.maximum(jacobian_upper, jac_hi)
        if jacobian_lower is None or jacobian_upper is None:
            raise RuntimeError("Clipped spline spans must be nonempty.")
        return jacobian_lower, jacobian_upper

    def bounds(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Outward value and global-parameter Jacobian bounds over a parameter box."""
        return (*self.values(lower, upper), *self.jacobians(lower, upper))


type _ExactBox = tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]]
type _Net = tuple[tuple[ExactScalar, ...], ...]


class _SpanQuotientNets:
    """Bernstein numerator/denominator nets of one rational span and its gradient.

    A rational Bernstein quotient ``N / D`` with positive denominator
    coefficients is a convex combination of the coefficient ratios
    ``N_ij / D_ij``. Both nets are restricted to a box by blossoming, so the
    restricted ratio hull is an exact enclosure with second-order excess
    whenever the restricted denominator coefficients stay positive. Position
    uses ``N / w``; the global-parameter derivatives use
    ``(N_u w - N w_u) / w^2`` with the numerator degree-raised to the ``w^2``
    net.
    """

    def __init__(self, span: RationalBSplineSurfaceSpan, /) -> None:
        homogeneous = span.homogeneous_bernstein_coefficients
        nets = tuple(
            tuple(tuple(point[component] for point in row) for row in homogeneous)
            for component in range(len(homogeneous[0][0]))
        )
        numerators, weight = nets[:-1], nets[-1]
        bounds = span.parameter_bounds
        weight_u = _tensor_bernstein_derivative(weight, 0, bounds)
        weight_v = _tensor_bernstein_derivative(weight, 1, bounds)
        # Multiplying by the unit polynomial raises a derivative numerator to the
        # w^2 degree; a constant direction already matches it.
        one = Fraction(1)
        raise_u: _Net = ((one,), (one,)) if span.degrees[0] else ((one,),)
        raise_v: _Net = ((one, one),) if span.degrees[1] else ((one,),)
        self.parameter_bounds = bounds
        self.numerators = numerators
        self.weight = weight
        self.weight_squared = _tensor_bernstein_product(weight, weight)
        self.du = tuple(
            _tensor_bernstein_product(
                _tensor_bernstein_linear_combination(
                    (
                        (
                            Fraction(1),
                            _tensor_bernstein_product(
                                _tensor_bernstein_derivative(numerator, 0, bounds),
                                weight,
                            ),
                        ),
                        (Fraction(-1), _tensor_bernstein_product(numerator, weight_u)),
                    )
                ),
                raise_u,
            )
            for numerator in numerators
        )
        self.dv = tuple(
            _tensor_bernstein_product(
                _tensor_bernstein_linear_combination(
                    (
                        (
                            Fraction(1),
                            _tensor_bernstein_product(
                                _tensor_bernstein_derivative(numerator, 1, bounds),
                                weight,
                            ),
                        ),
                        (Fraction(-1), _tensor_bernstein_product(numerator, weight_v)),
                    )
                ),
                raise_v,
            )
            for numerator in numerators
        )

    def _relative(self, bounds: _ExactBox, /) -> tuple[ExactScalar, ...]:
        (span_u0, span_u1), (span_v0, span_v1) = self.parameter_bounds
        (u0, u1), (v0, v1) = bounds
        return (
            (u0 - span_u0) / (span_u1 - span_u0),
            (u1 - span_u0) / (span_u1 - span_u0),
            (v0 - span_v0) / (span_v1 - span_v0),
            (v1 - span_v0) / (span_v1 - span_v0),
        )

    def values(self, bounds: _ExactBox, /) -> tuple[np.ndarray, np.ndarray]:
        relative = self._relative(bounds)
        return _quotient_hull(
            tuple(_restrict_net(numerator, relative) for numerator in self.numerators),
            _restrict_net(self.weight, relative),
        )

    def jacobians(self, bounds: _ExactBox, /) -> tuple[np.ndarray, np.ndarray]:
        relative = self._relative(bounds)
        denominator = _restrict_net(self.weight_squared, relative)
        du = _quotient_hull(
            tuple(_restrict_net(net, relative) for net in self.du), denominator
        )
        dv = _quotient_hull(
            tuple(_restrict_net(net, relative) for net in self.dv), denominator
        )
        return np.stack((du[0], dv[0]), axis=1), np.stack((du[1], dv[1]), axis=1)


def _de_casteljau_split(
    coefficients: Sequence[ExactScalar], parameter: ExactScalar, /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]]:
    """Bernstein coefficients on ``[0, t]`` and ``[t, 1]`` for any real ``t``."""
    row = tuple(coefficients)
    left = [row[0]]
    right = [row[-1]]
    while len(row) > 1:
        row = tuple(
            (1 - parameter) * first + parameter * second
            for first, second in zip(row[:-1], row[1:], strict=True)
        )
        left.append(row[0])
        right.append(row[-1])
    return tuple(left), tuple(reversed(right))


def _restrict_coefficients(
    coefficients: Sequence[ExactScalar], lower: ExactScalar, upper: ExactScalar, /
) -> tuple[ExactScalar, ...]:
    """Exact Bernstein coefficients of the polynomial on ``[lower, upper]``.

    De Casteljau splitting is a polynomial identity, so it also continues the
    polynomial past ``[0, 1]``; the result is degenerate (constant) for a
    point interval.
    """
    if lower != 1:
        _, right = _de_casteljau_split(coefficients, lower)
        return _de_casteljau_split(right, (upper - lower) / (1 - lower))[0]
    left, _ = _de_casteljau_split(coefficients, upper)
    return _de_casteljau_split(left, lower / upper)[1]


def _restrict_net(net: _Net, relative: Sequence[ExactScalar], /) -> _Net:
    lower_u, upper_u, lower_v, upper_v = relative
    columns = tuple(
        _restrict_coefficients(tuple(row[index] for row in net), lower_u, upper_u)
        for index in range(len(net[0]))
    )
    return tuple(
        _restrict_coefficients(tuple(column[row] for column in columns), lower_v, upper_v)
        for row in range(len(columns[0]))
    )


def _quotient_hull(
    numerators: Sequence[_Net], denominator: _Net, /
) -> tuple[np.ndarray, np.ndarray]:
    count = len(numerators)
    if min(value for row in denominator for value in row) <= 0:
        return np.full((count,), -np.inf), np.full((count,), np.inf)
    lower: list[float] = []
    upper: list[float] = []
    for numerator in numerators:
        ratios = tuple(
            value / weight
            for numerator_row, weight_row in zip(numerator, denominator, strict=True)
            for value, weight in zip(numerator_row, weight_row, strict=True)
        )
        lower.append(fraction_down(_exact_scalar_interval(min(ratios))[0]))
        upper.append(fraction_up(_exact_scalar_interval(max(ratios))[1]))
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


class _AnalyticSource:
    def __init__(self, patch: AbstractSurfacePatch, /) -> None:
        self.patch = patch
        self._values = prepare_interval_function(patch.evaluate, 2)
        self._jacobians = prepare_interval_function(jax.jacfwd(patch.evaluate), 2)

    def values(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        value_lower, value_upper = self._values.evaluate(lower[None], upper[None])
        return value_lower[0], value_upper[0]

    def jacobians(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        jacobian_lower, jacobian_upper = self._jacobians.evaluate(
            lower[None], upper[None]
        )
        return jacobian_lower[0], jacobian_upper[0]


class _PairValue:
    def __init__(self, prepared: _TSplinePairPrepared, /) -> None:
        self.prepared = prepared

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
        result_lower = np.empty((lo.shape[0], 3), dtype=np.float64)
        result_upper = np.empty_like(result_lower)
        for row in range(lo.shape[0]):
            first = self.prepared.first_source.values(lo[row, :2], hi[row, :2])
            second = self.prepared.second_source.values(lo[row, 2:], hi[row, 2:])
            result_lower[row], result_upper[row] = interval_subtract(first, second)
        return result_lower, result_upper


class _PairJacobian:
    def __init__(self, prepared: _TSplinePairPrepared, /) -> None:
        self.prepared = prepared

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
        result_lower = np.empty((lo.shape[0], 3, 4), dtype=np.float64)
        result_upper = np.empty_like(result_lower)
        for row in range(lo.shape[0]):
            first = self.prepared.first_source.jacobians(lo[row, :2], hi[row, :2])
            second = self.prepared.second_source.jacobians(lo[row, 2:], hi[row, 2:])
            result_lower[row, :, :2], result_upper[row, :, :2] = first
            result_lower[row, :, 2:] = -second[1]
            result_upper[row, :, 2:] = -second[0]
        return result_lower, result_upper


class _TSplinePairPrepared(StrictModule):
    system: SurfacePairSystem
    source_regions: tuple[SurfaceRegion, SurfaceRegion] | None
    first_source: _SplineSource | _AnalyticSource
    second_source: _SplineSource | _AnalyticSource
    value: _PairValue
    jacobian: _PairJacobian

    def __init__(
        self,
        system: SurfacePairSystem,
        source_regions: tuple[SurfaceRegion, SurfaceRegion] | None,
        /,
    ) -> None:
        if source_regions is not None and (
            source_regions[0].patch is not system.first
            or source_regions[1].patch is not system.second
        ):
            raise ValueError(
                "T-spline interval preparation must bind the exact source regions."
            )
        self.system = system
        self.source_regions = source_regions
        self.first_source = _surface_source(system.first)
        self.second_source = _surface_source(system.second)
        self.value = _PairValue(self)
        self.jacobian = _PairJacobian(self)

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        lower, upper = self.jacobian.evaluate(values, values)
        return 0.5 * lower + 0.5 * upper


def _prepare_tspline_pair(
    system: SurfacePairSystem,
    source_regions: tuple[SurfaceRegion, SurfaceRegion] | None,
    /,
) -> KrawczykEvaluator:
    """Prepare exact local Bézier and native analytic bounds for a T-spline pair."""
    if not isinstance(system.first, TSplineSurfacePatch) and not isinstance(
        system.second, TSplineSurfacePatch
    ):
        raise TypeError("T-spline pair preparation requires a T-spline source patch.")
    return _TSplinePairPrepared(system, source_regions)


def _surface_source(patch: AbstractSurfacePatch, /) -> _SplineSource | _AnalyticSource:
    if isinstance(patch, (TSplineSurfacePatch, BSplineSurfacePatch)):
        return _SplineSource(patch)
    return _AnalyticSource(patch)


class _NodePrepared:
    def __init__(self, pair: KrawczykEvaluator, axis: int, value: float, /) -> None:
        self.pair = pair
        self.axis = axis
        self.value = value

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        pair_lower, pair_upper = self.pair.value.evaluate(lower, upper)
        fixed_lower, fixed_upper = interval_subtract(
            (lower[:, self.axis], upper[:, self.axis]),
            (np.asarray(self.value), np.asarray(self.value)),
        )
        return (
            np.concatenate((pair_lower, np.asarray(fixed_lower)[:, None]), axis=1),
            np.concatenate((pair_upper, np.asarray(fixed_upper)[:, None]), axis=1),
        )

    def jacobian_bounds(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        jacobian_lower, jacobian_upper = self.pair.jacobian.evaluate(lower, upper)
        row = np.zeros((lower.shape[0], 1, 4), dtype=np.float64)
        row[:, 0, self.axis] = 1.0
        return (
            np.concatenate((jacobian_lower, row), axis=1),
            np.concatenate((jacobian_upper, row), axis=1),
        )


class _NodeEvaluator:
    def __init__(self, node: _NodePrepared, /) -> None:
        self.node = node

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.node.evaluate(lower, upper)


class _NodeJacobian:
    def __init__(self, node: _NodePrepared, /) -> None:
        self.node = node

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.node.jacobian_bounds(lower, upper)


def _node_preparation(
    pair: KrawczykEvaluator, axis: int, value: float, /
) -> KrawczykEvaluator:
    node = _NodePrepared(pair, axis, value)

    class Prepared:
        value = _NodeEvaluator(node)
        jacobian = _NodeJacobian(node)

        def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
            lower, upper = node.jacobian_bounds(points, points)
            return 0.5 * lower + 0.5 * upper

    return Prepared()


def _unresolved(
    first: SurfaceRegion,
    second: SurfaceRegion,
    reason: UnresolvedIntersectionReason,
    /,
) -> SurfaceIntersectionResult:
    lower = np.concatenate(
        (np.asarray(first.parameter_box[0]), np.asarray(second.parameter_box[0]))
    )
    upper = np.concatenate(
        (np.asarray(first.parameter_box[1]), np.asarray(second.parameter_box[1]))
    )
    return SurfaceIntersectionResult(
        unresolved=(UnresolvedParameterRegion(lower, upper, reason),)
    )


def intersect_tspline_surface_regions(
    first: SurfaceRegion,
    second: SurfaceRegion,
    /,
    *,
    policy: ParametricIntersectionPolicy,
) -> SurfaceIntersectionResult | None:
    """Intersect a T-spline graph with a plane or a regular analytic quadric."""
    if not isinstance(first.patch, TSplineSurfacePatch) and not isinstance(
        second.patch, TSplineSurfacePatch
    ):
        return None
    if isinstance(first.patch, TSplineSurfacePatch):
        spline_region, other_region, spline_first = first, second, True
    else:
        spline_region, other_region, spline_first = second, first, False
    if isinstance(other_region.patch, (TSplineSurfacePatch, BSplineSurfacePatch)):
        return _intersect_spline_pair(first, second, policy)
    if not isinstance(other_region.patch, (PlanePatch, CylinderPatch, SpherePatch)):
        return _unresolved(first, second, "continuation")
    return _intersect_level_graph(
        first, second, spline_region, other_region, spline_first, policy
    )


def _intersect_level_graph(
    first: SurfaceRegion,
    second: SurfaceRegion,
    spline_region: SurfaceRegion,
    other_region: SurfaceRegion,
    spline_first: bool,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult:
    patch = spline_region.patch
    if not isinstance(patch, TSplineSurfacePatch):
        raise TypeError("A T-spline level graph requires a TSplineSurfacePatch.")
    try:
        pair = SurfacePairSystem(first.patch, second.patch)
        prepared = _prepare_tspline_pair(pair, (first, second))
        if not isinstance(prepared, _TSplinePairPrepared):
            raise TypeError("T-spline preparation must retain its extracted source.")
        spline_source = prepared.first_source if spline_first else prepared.second_source
        if not isinstance(spline_source, _SplineSource):
            raise TypeError(
                "A T-spline intersection graph requires extracted Bézier source pieces."
            )
        if not isinstance(other_region.patch, (PlanePatch, CylinderPatch, SpherePatch)):
            raise TypeError("A T-spline level graph requires an analytic quadric.")
        level = _LevelSet(patch, spline_region, other_region.patch, spline_source)
        graph = level.graph_orientation()
        if graph is None:
            reason = "singular" if level.derivative_singular() else "continuation"
            if level.no_intersection():
                return SurfaceIntersectionResult()
            return _unresolved(first, second, reason)
        axis = 0 if spline_first else 2
        t0, t1 = spline_region.lower[0], spline_region.upper[0]
        breakpoints = sorted(
            {
                t0,
                t1,
                *(bound for bound in spline_source.breakpoints(0) if t0 < bound < t1),
            }
        )
        pending = [
            (breakpoints[index], breakpoints[index + 1])
            for index in range(len(breakpoints) - 1)
        ]
        segments: list[
            tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        unresolved: list[UnresolvedParameterRegion] = []
        region_lower = np.concatenate(
            (np.asarray(first.parameter_box[0]), np.asarray(second.parameter_box[0]))
        )
        region_upper = np.concatenate(
            (np.asarray(first.parameter_box[1]), np.asarray(second.parameter_box[1]))
        )
        while pending:
            lower_t, upper_t = pending.pop()
            start = level.point_at(lower_t, spline_first, other_region)
            end = level.point_at(upper_t, spline_first, other_region)
            bounds = level.graph_box(lower_t, upper_t, spline_first, other_region)
            if bounds is not None:
                inflation = _inflated_krawczyk(prepared, bounds[0], bounds[1], axis)
                if inflation.certified:
                    segments.append(
                        (lower_t, upper_t, start, end, inflation.lower, inflation.upper)
                    )
                    continue
            middle = lower_t + 0.5 * (upper_t - lower_t)
            if not lower_t < middle < upper_t:
                slab_lower, slab_upper = region_lower.copy(), region_upper.copy()
                slab_lower[axis], slab_upper[axis] = lower_t, upper_t
                unresolved.append(
                    UnresolvedParameterRegion(slab_lower, slab_upper, "resolution")
                )
                continue
            pending.append((middle, upper_t))
            pending.append((lower_t, middle))
        if unresolved or not segments:
            return SurfaceIntersectionResult(
                unresolved=tuple(unresolved)
                or (
                    UnresolvedParameterRegion(
                        np.concatenate(
                            (
                                np.asarray(first.parameter_box[0]),
                                np.asarray(second.parameter_box[0]),
                            )
                        ),
                        np.concatenate(
                            (
                                np.asarray(first.parameter_box[1]),
                                np.asarray(second.parameter_box[1]),
                            )
                        ),
                        "continuation",
                    ),
                )
            )
        segments.sort(key=lambda item: item[0])
        nodes = [segments[0][0]] + [item[1] for item in segments]
        node_points = [
            level.point_at(value, spline_first, other_region) for value in nodes
        ]
        node_bounds = tuple(
            _certified_node_box(prepared, point, axis, value)
            for point, value in zip(node_points, nodes, strict=True)
        )
        if any(item is None for item in node_bounds):
            return _unresolved(first, second, "resolution")
        node_lower = np.stack([item[0] for item in node_bounds if item is not None])
        node_upper = np.stack([item[1] for item in node_bounds if item is not None])
        curve = IntersectionCurve(
            first,
            second,
            chart_axes=np.full((len(segments),), axis, dtype=np.int32),
            chart_start=np.stack([item[2] for item in segments]),
            chart_end=np.stack([item[3] for item in segments]),
            chart_lower=np.stack([item[4] for item in segments]),
            chart_upper=np.stack([item[5] for item in segments]),
            node_axes=np.full((len(nodes),), axis, dtype=np.int32),
            node_values=np.asarray(nodes, dtype=np.float64),
            node_lower=node_lower,
            node_upper=node_upper,
            node_period_shifts=np.zeros((len(nodes), 2, 4), dtype=np.int32),
            closed=False,
            start_kind="boundary",
            end_kind="boundary",
        )
        return SurfaceIntersectionResult(curves=(curve,))
    except (ValueError, FloatingPointError, OverflowError):
        return _unresolved(first, second, "resolution")


def _bisect_box(
    lower: np.ndarray, upper: np.ndarray, extent: np.ndarray, /
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    axis = int(np.argmax((upper - lower) / extent))
    middle = lower[axis] + 0.5 * (upper[axis] - lower[axis])
    if not lower[axis] < middle < upper[axis]:
        return ()
    left_upper, right_lower = upper.copy(), lower.copy()
    left_upper[axis] = right_lower[axis] = middle
    return (lower, left_upper), (right_lower, upper)


class _LevelSet:
    def __init__(
        self,
        spline: TSplineSurfacePatch,
        spline_region: SurfaceRegion,
        other: PlanePatch | CylinderPatch | SpherePatch,
        source: _SplineSource,
        /,
    ) -> None:
        self.spline = spline
        self.region = spline_region
        self.other = other
        self.source = source
        self._evaluate = eqx.filter_jit(spline.evaluate)
        self.normal: Interval | None = None
        if isinstance(other, PlanePatch):
            self.normal = _plane_normal_interval(other)
            if self.normal is None:
                raise ValueError("Plane patch axes must span a plane.")

    def graph_orientation(self, /) -> int | None:
        """Certify that every source u-line crosses the level set exactly once.

        Boxes are bisected until each excludes the level or has a strict level
        derivative along v; all such signs must agree, and the two v-boundary
        edges must carry opposite strict level signs. A box that can no longer
        be split leaves the orientation unproved.
        """
        lower = np.asarray(self.region.parameter_box[0], dtype=np.float64)
        upper = np.asarray(self.region.parameter_box[1], dtype=np.float64)
        extent = upper - lower
        orientation: int | None = None
        pending = [(lower, upper)]
        while pending:
            box_lower, box_upper = pending.pop()
            value, derivative = self._value_and_derivative(box_lower, box_upper, 1)
            if value[0] > 0.0 or value[1] < 0.0:
                continue
            sign = 1 if derivative[0] > 0.0 else -1 if derivative[1] < 0.0 else 0
            if sign:
                if orientation is not None and sign != orientation:
                    return None
                orientation = sign
                continue
            children = _bisect_box(box_lower, box_upper, extent)
            if not children:
                return None
            pending.extend(children)
        if orientation is None:
            return None
        for edge, sign in ((lower[1], -orientation), (upper[1], orientation)):
            if not self._edge_has_sign(lower[0], upper[0], float(edge), sign):
                return None
        return orientation

    def _value_and_derivative(
        self, lower: np.ndarray, upper: np.ndarray, axis: int, /
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        value_lower, value_upper, jacobian_lower, jacobian_upper = self.source.bounds(
            lower, upper
        )
        value = self._implicit_value((value_lower, value_upper))
        derivative = self._implicit_derivative(
            (value_lower, value_upper),
            (jacobian_lower[:, axis], jacobian_upper[:, axis]),
        )
        return (
            (float(value[0]), float(value[1])),
            (float(derivative[0]), float(derivative[1])),
        )

    def _edge_has_sign(
        self, first: float, last: float, edge: float, sign: int, /
    ) -> bool:
        pending = [(first, last)]
        while pending:
            lower_u, upper_u = pending.pop()
            value = self.value_bounds((lower_u, edge), (upper_u, edge))
            if (sign > 0 and value[0] > 0.0) or (sign < 0 and value[1] < 0.0):
                continue
            middle = lower_u + 0.5 * (upper_u - lower_u)
            if (
                (sign > 0 and value[1] <= 0.0)
                or (sign < 0 and value[0] >= 0.0)
                or not lower_u < middle < upper_u
            ):
                return False
            pending.extend(((lower_u, middle), (middle, upper_u)))
        return True

    def no_intersection(self, /) -> bool:
        bounds = np.asarray(self.region.parameter_box, dtype=np.float64)
        value = self.value_bounds(
            (float(bounds[0, 0]), float(bounds[0, 1])),
            (float(bounds[1, 0]), float(bounds[1, 1])),
        )
        return value[0] > 0.0 or value[1] < 0.0

    def derivative_singular(self, /) -> bool:
        bounds = np.asarray(self.region.parameter_box, dtype=np.float64)
        derivative = self._derivative_bounds(bounds[0], bounds[1], 1)
        return derivative[0] <= 0.0 <= derivative[1]

    def value_bounds(
        self, lower: tuple[float, float], upper: tuple[float, float], /
    ) -> tuple[float, float]:
        point = self._implicit_value(
            self.source.values(np.asarray(lower), np.asarray(upper))
        )
        return float(point[0]), float(point[1])

    def _derivative_bounds(
        self, lower: np.ndarray, upper: np.ndarray, axis: int, /
    ) -> tuple[float, float]:
        value_lower, value_upper, jacobian_lower, jacobian_upper = self.source.bounds(
            lower, upper
        )
        value = self._implicit_derivative(
            (value_lower, value_upper),
            (jacobian_lower[:, axis], jacobian_upper[:, axis]),
        )
        return float(value[0]), float(value[1])

    def _implicit_value(self, source: Interval, /) -> Interval:
        return _implicit_value_interval(source, self.other, self.normal)

    def _implicit_derivative(
        self,
        source: Interval,
        derivative: Interval,
        /,
    ) -> Interval:
        return _implicit_derivative_interval(source, derivative, self.other, self.normal)

    def _point_value(self, parameters: np.ndarray, /) -> float:
        point = np.asarray(
            self._evaluate(jnp.asarray(parameters, dtype=jnp.float64)),
            dtype=np.float64,
        )
        bounds = self._implicit_value((point, point))
        return 0.5 * (float(bounds[0]) + float(bounds[1]))

    def point_at(
        self, value: float, spline_first: bool, other_region: SurfaceRegion, /
    ) -> np.ndarray:
        region = np.asarray(self.region.parameter_box, dtype=np.float64)
        lower, upper = float(region[0, 1]), float(region[1, 1])
        first_value, last_value = (
            self._point_value(np.asarray((value, lower))),
            self._point_value(np.asarray((value, upper))),
        )
        increasing = last_value > first_value
        if first_value == 0.0:
            root = lower
        elif last_value == 0.0:
            root = upper
        elif first_value * last_value > 0.0:
            raise ValueError(
                "The T-spline level does not cross its certified graph interval."
            )
        else:
            left, right = lower, upper
            while True:
                middle = left + 0.5 * (right - left)
                if not left < middle < right:
                    break
                middle_value = self._point_value(np.asarray((value, middle)))
                if (middle_value < 0.0) == increasing:
                    left = middle
                else:
                    right = middle
                if right - left <= 8.0 * np.finfo(np.float64).eps * max(1.0, abs(middle)):
                    break
            root = left + 0.5 * (right - left)
        parameters = np.asarray((value, root), dtype=np.float64)
        point = np.asarray(self._evaluate(jnp.asarray(parameters)), dtype=np.float64)
        if isinstance(other_region.patch, PlanePatch):
            plane = other_region.patch
            axes = np.column_stack(
                (np.asarray(plane.first_axis), np.asarray(plane.second_axis))
            )
            coordinates = np.linalg.lstsq(
                axes, point - np.asarray(plane.origin), rcond=None
            )[0]
        elif isinstance(other_region.patch, CylinderPatch):
            cylinder = other_region.patch
            delta = point - np.asarray(cylinder.origin)
            first_axis = np.asarray(cylinder.first_axis)
            second_axis = np.asarray(cylinder.second_axis)
            angle = float(np.arctan2(delta @ second_axis, delta @ first_axis))
            angle = _periodic_value(
                angle, other_region.lower[0], other_region.upper[0], tau
            )
            coordinates = np.asarray(
                (angle, delta @ np.asarray(cylinder.axis)), dtype=np.float64
            )
        elif isinstance(other_region.patch, SpherePatch):
            sphere = other_region.patch
            delta = point - np.asarray(sphere.center)
            radius = float(np.asarray(sphere.radius))
            first_axis, second_axis, axis = (
                np.asarray(value)
                for value in (sphere.first_axis, sphere.second_axis, sphere.axis)
            )
            longitude = float(np.arctan2(delta @ second_axis, delta @ first_axis))
            latitude = float(np.arcsin(np.clip((delta @ axis) / radius, -1.0, 1.0)))
            longitude = _periodic_value(
                longitude, other_region.lower[0], other_region.upper[0], tau
            )
            coordinates = np.asarray((longitude, latitude), dtype=np.float64)
        else:
            raise TypeError("T-spline level graph source must be a supported quadric.")
        if np.any(coordinates < np.asarray(other_region.parameter_box[0])) or np.any(
            coordinates > np.asarray(other_region.parameter_box[1])
        ):
            raise ValueError(
                "The quadric parameter point lies outside its source region."
            )
        return (
            np.concatenate((parameters, coordinates))
            if spline_first
            else np.concatenate((coordinates, parameters))
        )

    def _graph_tangent_bounds(
        self,
        spline_first: bool,
        other_region: SurfaceRegion,
        spline_lower: np.ndarray,
        spline_upper: np.ndarray,
        /,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        value_lower, value_upper, jacobian_lower, jacobian_upper = self.source.bounds(
            spline_lower, spline_upper
        )
        f_u = self._implicit_derivative(
            (value_lower, value_upper),
            (jacobian_lower[:, 0], jacobian_upper[:, 0]),
        )
        f_v = self._implicit_derivative(
            (value_lower, value_upper),
            (jacobian_lower[:, 1], jacobian_upper[:, 1]),
        )
        if float(f_v[0]) <= 0.0 <= float(f_v[1]):
            return None
        dv_du = interval_divide((-np.asarray(f_u[1]), -np.asarray(f_u[0])), f_v)
        physical_tangent = interval_add(
            (jacobian_lower[:, 0], jacobian_upper[:, 0]),
            interval_multiply((jacobian_lower[:, 1], jacobian_upper[:, 1]), dv_du),
        )
        other = other_region.patch
        if isinstance(other, PlanePatch):
            transform = _plane_parameter_map(other)
            if transform is None:
                return None
            other_tangent = _linear_map_interval(transform, physical_tangent)
        elif isinstance(other, CylinderPatch):
            origin = np.asarray(other.origin, dtype=np.float64)
            first_axis = np.asarray(other.first_axis, dtype=np.float64)
            second_axis = np.asarray(other.second_axis, dtype=np.float64)
            axis = np.asarray(other.axis, dtype=np.float64)
            radial = interval_subtract(
                (value_lower, value_upper), _constant_interval(origin)
            )
            radial_first = _interval_dot(radial, _constant_interval(first_axis))
            radial_second = _interval_dot(radial, _constant_interval(second_axis))
            tangent_first = _interval_dot(
                physical_tangent, _constant_interval(first_axis)
            )
            tangent_second = _interval_dot(
                physical_tangent, _constant_interval(second_axis)
            )
            numerator = interval_subtract(
                interval_multiply(radial_first, tangent_second),
                interval_multiply(radial_second, tangent_first),
            )
            radius = np.asarray(other.radius, dtype=np.float64)
            angle_tangent = interval_divide(numerator, _radius_squared_interval(radius))
            height_tangent = _interval_dot(physical_tangent, _constant_interval(axis))
            other_tangent = (
                np.stack((angle_tangent[0], height_tangent[0])),
                np.stack((angle_tangent[1], height_tangent[1])),
            )
        elif isinstance(other, SpherePatch):
            center = np.asarray(other.center, dtype=np.float64)
            first_axis = np.asarray(other.first_axis, dtype=np.float64)
            second_axis = np.asarray(other.second_axis, dtype=np.float64)
            axis = np.asarray(other.axis, dtype=np.float64)
            radial = interval_subtract(
                (value_lower, value_upper), _constant_interval(center)
            )
            radial_first = _interval_dot(radial, _constant_interval(first_axis))
            radial_second = _interval_dot(radial, _constant_interval(second_axis))
            radial_axis = _interval_dot(radial, _constant_interval(axis))
            tangent_first = _interval_dot(
                physical_tangent, _constant_interval(first_axis)
            )
            tangent_second = _interval_dot(
                physical_tangent, _constant_interval(second_axis)
            )
            tangent_axis = _interval_dot(physical_tangent, _constant_interval(axis))
            longitude_numerator = interval_subtract(
                interval_multiply(radial_first, tangent_second),
                interval_multiply(radial_second, tangent_first),
            )
            horizontal_squared = _norm_squared_interval(
                (
                    np.asarray((radial_first[0], radial_second[0])),
                    np.asarray((radial_first[1], radial_second[1])),
                )
            )
            if float(horizontal_squared[0]) <= 0.0:
                return None
            longitude_tangent = interval_divide(longitude_numerator, horizontal_squared)
            radius = np.asarray(other.radius, dtype=np.float64)
            axial_squared = interval_multiply(radial_axis, radial_axis)
            latitude_denominator = interval_sqrt(
                interval_subtract(_radius_squared_interval(radius), axial_squared)
            )
            if float(latitude_denominator[0]) <= 0.0:
                return None
            latitude_tangent = interval_divide(tangent_axis, latitude_denominator)
            other_tangent = (
                np.stack((longitude_tangent[0], latitude_tangent[0])),
                np.stack((longitude_tangent[1], latitude_tangent[1])),
            )
        else:
            return None
        spline_tangent = (
            np.asarray((1.0, dv_du[0]), dtype=np.float64),
            np.asarray((1.0, dv_du[1]), dtype=np.float64),
        )
        if not spline_first:
            spline_tangent, other_tangent = other_tangent, spline_tangent
        return (
            np.concatenate((spline_tangent[0], other_tangent[0])),
            np.concatenate((spline_tangent[1], other_tangent[1])),
        )

    def graph_box(
        self,
        lower_t: float,
        upper_t: float,
        spline_first: bool,
        other_region: SurfaceRegion,
        /,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        midpoint_t = lower_t + 0.5 * (upper_t - lower_t)
        midpoint = self.point_at(midpoint_t, spline_first, other_region)
        region = np.asarray(self.region.parameter_box, dtype=np.float64)
        v_index = 1 if spline_first else 3
        span = upper_t - lower_t
        margin = 64.0 * np.finfo(np.float64).eps * (1.0 + np.abs(midpoint))
        # The tangent enclosure over the whole v-range of the cell bounds the
        # graph's v-excursion; a second pass over that narrower strip tightens
        # the box. Krawczyk certifies the final box independently.
        v_lower, v_upper = float(region[0, 1]), float(region[1, 1])
        widths: np.ndarray | None = None
        for _ in range(2):
            tangent = self._graph_tangent_bounds(
                spline_first,
                other_region,
                np.asarray((lower_t, v_lower)),
                np.asarray((upper_t, v_upper)),
            )
            if tangent is None:
                break
            slope = np.maximum(np.abs(tangent[0]), np.abs(tangent[1]))
            widths = slope * (0.5 * span) + margin
            v_lower = max(float(region[0, 1]), float(midpoint[v_index] - widths[v_index]))
            v_upper = min(float(region[1, 1]), float(midpoint[v_index] + widths[v_index]))
            if not v_lower < v_upper:
                break
        if widths is None:
            return None
        region_lower = (
            np.concatenate(
                (
                    np.asarray(self.region.parameter_box[0]),
                    np.asarray(other_region.parameter_box[0]),
                )
            )
            if spline_first
            else np.concatenate(
                (
                    np.asarray(other_region.parameter_box[0]),
                    np.asarray(self.region.parameter_box[0]),
                )
            )
        )
        region_upper = (
            np.concatenate(
                (
                    np.asarray(self.region.parameter_box[1]),
                    np.asarray(other_region.parameter_box[1]),
                )
            )
            if spline_first
            else np.concatenate(
                (
                    np.asarray(other_region.parameter_box[1]),
                    np.asarray(self.region.parameter_box[1]),
                )
            )
        )
        axis = 0 if spline_first else 2
        lower, upper = midpoint - widths, midpoint + widths
        lower = np.maximum(lower, region_lower)
        upper = np.minimum(upper, region_upper)
        lower[axis], upper[axis] = lower_t, upper_t
        if np.any(lower >= upper):
            return None
        return lower, upper


class _Inflation(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray
    witness: KrawczykResult

    @property
    def certified(self) -> bool:
        return bool(self.witness.certified[0])

    @property
    def excluded(self) -> bool:
        return bool(self.witness.excluded[0])


def _inflated_krawczyk(
    prepared: KrawczykEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    axis: int | None,
    /,
) -> _Inflation:
    """Certify or exclude roots of a box by Krawczyk with epsilon-inflation.

    The (parametric) image ``K(X)`` encloses every root in ``X``, over the
    segment's parameter interval when ``axis`` is given, so while it is
    neither interior nor disjoint the next candidate is the hull of ``X`` and
    ``K(X)`` inflated about its center. Each candidate contains the input box,
    so a certificate (a unique root or graph) or an exclusion of a candidate
    also decides the input box. Candidates grow strictly, so the loop ends by
    representability or at a non-contracting or unbounded image.
    """
    free = np.asarray(
        [index for index in range(lower.size) if index != axis], dtype=np.int64
    )
    while True:
        witness = krawczyk(prepared, lower[None], upper[None], parameter_axis=axis)
        inflation = _Inflation(lower, upper, witness)
        if inflation.certified or inflation.excluded:
            return inflation
        image_lower = np.asarray(witness.lower[0], dtype=np.float64)
        image_upper = np.asarray(witness.upper[0], dtype=np.float64)
        if not (
            float(witness.contraction[0]) < 1.0
            and np.all(np.isfinite(image_lower))
            and np.all(np.isfinite(image_upper))
        ):
            return inflation
        center = 0.5 * image_lower + 0.5 * image_upper
        radius = np.maximum(center - image_lower, image_upper - center)
        radius = 2.0 * radius + 64.0 * np.finfo(np.float64).eps * (1.0 + np.abs(center))
        candidate_lower, candidate_upper = lower.copy(), upper.copy()
        candidate_lower[free] = np.minimum(lower[free], center - radius)
        candidate_upper[free] = np.maximum(upper[free], center + radius)
        if np.array_equal(candidate_lower, lower) and np.array_equal(
            candidate_upper, upper
        ):
            return inflation
        lower, upper = candidate_lower, candidate_upper


def _interval_sum(lower: HostArray, upper: HostArray, /) -> Interval:
    lower_values = np.asarray(lower)
    upper_values = np.asarray(upper)
    total_lower = np.asarray(0.0, dtype=np.float64)
    total_upper = np.asarray(0.0, dtype=np.float64)
    for axis in range(lower_values.shape[0]):
        total_lower, total_upper = interval_add(
            (total_lower, total_upper),
            (lower_values[axis], upper_values[axis]),
        )
    return total_lower, total_upper


def _interval_dot(first: Interval, second: Interval, /) -> Interval:
    products = interval_multiply(first, second)
    return _interval_sum(products[0], products[1])


def _linear_map_interval(
    matrix: Interval,
    values: Interval,
    /,
) -> Interval:
    matrix_lower = np.asarray(matrix[0])
    matrix_upper = np.asarray(matrix[1])
    lower = np.empty((matrix_lower.shape[0],), dtype=np.float64)
    upper = np.empty_like(lower)
    for row in range(matrix_lower.shape[0]):
        bounds = _interval_dot(values, (matrix_lower[row], matrix_upper[row]))
        lower[row], upper[row] = bounds
    return lower, upper


def _plane_axis_fractions(
    plane: PlanePatch, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    first = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(plane.first_axis, dtype=np.float64)
    )
    second = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(plane.second_axis, dtype=np.float64)
    )
    return first, second


def _plane_normal_interval(plane: PlanePatch, /) -> Interval | None:
    first, second = _plane_axis_fractions(plane)
    exact = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    if not any(exact):
        return None
    lower = np.asarray([fraction_down(value) for value in exact], dtype=np.float64)
    upper = np.asarray([fraction_up(value) for value in exact], dtype=np.float64)
    return lower, upper


def _plane_parameter_map(plane: PlanePatch, /) -> Interval | None:
    first, second = _plane_axis_fractions(plane)
    g00 = sum((value * value for value in first), Fraction(0))
    g01 = sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
    g11 = sum((value * value for value in second), Fraction(0))
    determinant = g00 * g11 - g01 * g01
    if determinant == 0:
        return None
    exact = (
        tuple(
            (g11 * first[axis] - g01 * second[axis]) / determinant for axis in range(3)
        ),
        tuple(
            (g00 * second[axis] - g01 * first[axis]) / determinant for axis in range(3)
        ),
    )
    lower = np.asarray(
        [[fraction_down(value) for value in row] for row in exact],
        dtype=np.float64,
    )
    upper = np.asarray(
        [[fraction_up(value) for value in row] for row in exact],
        dtype=np.float64,
    )
    return lower, upper


def _constant_interval(values: np.ndarray, /) -> Interval:
    return values, values


def _radius_squared_interval(radius: np.ndarray, /) -> Interval:
    return interval_multiply((radius, radius), (radius, radius))


def _norm_squared_interval(values: Interval, /) -> Interval:
    products = interval_multiply(values, values)
    return _interval_sum(products[0], products[1])


def _implicit_value_interval(
    values: Interval,
    other: PlanePatch | CylinderPatch | SpherePatch,
    normal: Interval | None,
    /,
) -> Interval:
    if isinstance(other, PlanePatch):
        delta = interval_subtract(
            values, _constant_interval(np.asarray(other.origin, dtype=np.float64))
        )
        if normal is None:
            raise RuntimeError("Plane level set has no prepared normal.")
        return _interval_dot(delta, normal)
    if isinstance(other, SpherePatch):
        delta = interval_subtract(
            values, _constant_interval(np.asarray(other.center, dtype=np.float64))
        )
        squared = _norm_squared_interval(delta)
        radius = np.asarray(other.radius, dtype=np.float64)
        return interval_subtract(squared, _radius_squared_interval(radius))
    origin = np.asarray(other.origin, dtype=np.float64)
    axis = np.asarray(other.axis, dtype=np.float64)
    delta = interval_subtract(values, _constant_interval(origin))
    axial = _interval_dot(delta, _constant_interval(axis))
    radial = interval_subtract(delta, interval_multiply(axial, _constant_interval(axis)))
    squared = _norm_squared_interval(radial)
    radius = np.asarray(other.radius, dtype=np.float64)
    return interval_subtract(squared, _radius_squared_interval(radius))


def _implicit_derivative_interval(
    values: Interval,
    derivatives: Interval,
    other: PlanePatch | CylinderPatch | SpherePatch,
    normal: Interval | None,
    /,
) -> Interval:
    if isinstance(other, PlanePatch):
        if normal is None:
            raise RuntimeError("Plane level set has no prepared normal.")
        return _interval_dot(derivatives, normal)
    if isinstance(other, SpherePatch):
        delta = interval_subtract(
            values, _constant_interval(np.asarray(other.center, dtype=np.float64))
        )
        product = _interval_dot(delta, derivatives)
    else:
        axis = np.asarray(other.axis, dtype=np.float64)
        delta = interval_subtract(
            values, _constant_interval(np.asarray(other.origin, dtype=np.float64))
        )
        radial = interval_subtract(
            delta,
            interval_multiply(
                _interval_dot(delta, _constant_interval(axis)),
                _constant_interval(axis),
            ),
        )
        radial_derivative = interval_subtract(
            derivatives,
            interval_multiply(
                _interval_dot(derivatives, _constant_interval(axis)),
                _constant_interval(axis),
            ),
        )
        product = _interval_dot(radial, radial_derivative)
    return interval_multiply((np.asarray(2.0), np.asarray(2.0)), product)


def _periodic_value(value: float, lower: float, upper: float, period: float, /) -> float:
    candidates = tuple(
        value + turn * period
        for turn in range(-3, 4)
        if lower <= value + turn * period <= upper
    )
    if len(candidates) != 1:
        return value
    return candidates[0]


def _certified_node_box(
    prepared: KrawczykEvaluator,
    point: np.ndarray,
    axis: int,
    value: float,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Isolate the unique coupled root on one fixed source coordinate."""
    radius = 64.0 * np.finfo(np.float64).eps * (1.0 + np.abs(point))
    lower, upper = point - radius, point + radius
    lower[axis] = min(value, float(lower[axis]))
    upper[axis] = max(value, float(upper[axis]))
    inflation = _inflated_krawczyk(
        _node_preparation(prepared, axis, value), lower, upper, None
    )
    return (inflation.lower, inflation.upper) if inflation.certified else None


def _free_axis_indices(axis: int, /) -> tuple[int, int, int]:
    match axis:
        case 0:
            return 1, 2, 3
        case 1:
            return 0, 2, 3
        case 2:
            return 0, 1, 3
        case 3:
            return 0, 1, 2
        case _:
            raise ValueError("A coupled surface axis must lie in [0, 4).")


def _intersect_spline_pair(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult:
    """Trace a spline-pair intersection as certified graphs over a T-spline axis.

    Before tracing, the route proves that (1) the coupled Jacobian in the three
    free coordinates is regular on every region box that may hold a root, so
    the solution set is a 1-manifold transverse to every slice of the graph
    axis, with no turning points and hence no closed loops, and (2) no root
    lies on a free face of the region. Every component is then a graph over the
    whole graph-axis interval, so the certified roots of the first slice
    enumerate all components; each is traced by inflated parametric Krawczyk
    segments chained through certified node roots.
    """
    pair = SurfacePairSystem(first.patch, second.patch)
    prepared = _TSplinePairPrepared(pair, (first, second))
    lower_domain = np.concatenate(
        (np.asarray(first.parameter_box[0]), np.asarray(second.parameter_box[0]))
    )
    upper_domain = np.concatenate(
        (np.asarray(first.parameter_box[1]), np.asarray(second.parameter_box[1]))
    )
    if isinstance(first.patch, TSplineSurfacePatch):
        source, axis = prepared.first_source, 0
    elif isinstance(second.patch, TSplineSurfacePatch):
        source, axis = prepared.second_source, 2
    else:
        raise TypeError("A spline-pair route requires a T-spline operand.")
    if not isinstance(source, _SplineSource):
        raise TypeError("A T-spline operand must retain exact Bernstein spans.")
    singular = _transversality_failure(prepared, lower_domain, upper_domain, axis)
    if singular is not None:
        return SurfaceIntersectionResult(unresolved=(singular,))
    for face_axis in _free_axis_indices(axis):
        for side in (lower_domain[face_axis], upper_domain[face_axis]):
            face_lower, face_upper = lower_domain.copy(), upper_domain.copy()
            face_lower[face_axis] = face_upper[face_axis] = side
            face_roots, face_unknown = _isolate_roots(
                prepared, face_lower, face_upper, face_axis
            )
            if face_roots or face_unknown is not None:
                return _unresolved(first, second, "continuation")
    t0, t1 = float(lower_domain[axis]), float(upper_domain[axis])
    slice_lower, slice_upper = lower_domain.copy(), upper_domain.copy()
    slice_upper[axis] = t0
    roots, unknown = _isolate_roots(prepared, slice_lower, slice_upper, axis)
    if unknown is not None:
        return SurfaceIntersectionResult(unresolved=(unknown,))
    breakpoints = sorted(
        {t1, *(bound for bound in source.breakpoints(0) if t0 < bound < t1)}
    )
    residual = eqx.filter_jit(pair.residual)
    branches: list[IntersectionCurve] = []
    for root in roots:
        curve = _trace_spline_pair_branch(
            prepared,
            residual,
            first,
            second,
            root,
            axis,
            breakpoints,
            lower_domain,
            upper_domain,
            policy,
        )
        if curve is None:
            return _unresolved(first, second, "resolution")
        branches.append(curve)
    return SurfaceIntersectionResult(curves=tuple(branches))


def _transversality_failure(
    prepared: KrawczykEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    axis: int,
    /,
) -> UnresolvedParameterRegion | None:
    """First box where the free Jacobian is not proved regular near a root.

    A box is decided when its value enclosure or Krawczyk image excludes a
    root, or when the preconditioned free Jacobian enclosure contracts
    (``|I - C J(X)| < 1``), which proves every Jacobian in the box regular.
    Undecided boxes are bisected until they can no longer be split.
    """
    extent = upper - lower
    pending = [(lower, upper)]
    while pending:
        box_lower, box_upper = pending.pop()
        value_lower, value_upper = prepared.value.evaluate(
            box_lower[None], box_upper[None]
        )
        if _all_exclude_zero(value_lower[0], value_upper[0]):
            continue
        witness = krawczyk(
            prepared, box_lower[None], box_upper[None], parameter_axis=axis
        )
        if bool(witness.excluded[0]) or float(witness.contraction[0]) < 1.0:
            continue
        children = _bisect_box(box_lower, box_upper, extent)
        if not children:
            return UnresolvedParameterRegion(box_lower, box_upper, "singular")
        pending.extend(children)
    return None


def _isolate_roots(
    prepared: KrawczykEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed_axis: int,
    /,
) -> tuple[tuple[np.ndarray, ...], UnresolvedParameterRegion | None]:
    """Certified roots of the coupled system on one fixed-coordinate slice.

    Boxes are excluded by value enclosures or Krawczyk images, decided by an
    inflated Krawczyk certificate (one root in a box containing them), or
    bisected until they can no longer be split, which leaves the slice
    unresolved. Distinct certificates are merged when one root enclosure lies
    in the other certificate's box.
    """
    extent = upper - lower
    extent[fixed_axis] = np.inf
    pending = [(lower, upper)]
    certificates: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    free = np.asarray(_free_axis_indices(fixed_axis))
    while pending:
        box_lower, box_upper = pending.pop()
        value_lower, value_upper = prepared.value.evaluate(
            box_lower[None], box_upper[None]
        )
        if _all_exclude_zero(value_lower[0], value_upper[0]) or any(
            np.all(box_lower >= known[0]) and np.all(box_upper <= known[1])
            for known in certificates
        ):
            continue
        inflation = _inflated_krawczyk(prepared, box_lower, box_upper, fixed_axis)
        if inflation.excluded:
            continue
        if inflation.certified:
            root_lower = inflation.lower.copy()
            root_upper = inflation.upper.copy()
            root_lower[free] = inflation.witness.lower[0]
            root_upper[free] = inflation.witness.upper[0]
            duplicate = False
            for (
                known_lower,
                known_upper,
                known_root_lower,
                known_root_upper,
            ) in certificates:
                if (
                    np.all(root_lower >= known_lower)
                    and np.all(root_upper <= known_upper)
                ) or (
                    np.all(known_root_lower >= inflation.lower)
                    and np.all(known_root_upper <= inflation.upper)
                ):
                    duplicate = True
                    break
                if np.all(root_lower <= known_root_upper) and np.all(
                    known_root_lower <= root_upper
                ):
                    return (), UnresolvedParameterRegion(
                        np.minimum(root_lower, known_root_lower),
                        np.maximum(root_upper, known_root_upper),
                        "resolution",
                    )
            if not duplicate:
                certificates.append(
                    (inflation.lower, inflation.upper, root_lower, root_upper)
                )
            continue
        children = _bisect_box(box_lower, box_upper, extent)
        if not children:
            return (), UnresolvedParameterRegion(box_lower, box_upper, "resolution")
        pending.extend(children)
    roots = tuple(
        0.5 * root_lower + 0.5 * root_upper
        for _, _, root_lower, root_upper in certificates
    )
    return tuple(sorted(roots, key=lambda point: tuple(point.tolist()))), None


def _trace_spline_pair_branch(
    prepared: KrawczykEvaluator,
    residual: Callable[[Array], Array],
    first: SurfaceRegion,
    second: SurfaceRegion,
    start: np.ndarray,
    axis: int,
    breakpoints: Sequence[float],
    lower_domain: np.ndarray,
    upper_domain: np.ndarray,
    policy: ParametricIntersectionPolicy,
    /,
) -> IntersectionCurve | None:
    """Chain certified graph segments of one transverse branch across the slab.

    Segments end at span breakpoints and are halved until an inflated
    parametric Krawczyk box containing the Newton roots at both ends and the
    midpoint certifies; halving ends by representability.
    """
    segments: list[
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    lower_t = float(start[axis])
    for target in breakpoints:
        while lower_t < target:
            upper_t = target
            while True:
                middle_t = lower_t + 0.5 * (upper_t - lower_t)
                if not lower_t < middle_t < upper_t:
                    return None
                end = _solve_slice_root(
                    residual, axis, upper_t, start, lower_domain, upper_domain, policy
                )
                middle = (
                    None
                    if end is None
                    else _solve_slice_root(
                        residual,
                        axis,
                        middle_t,
                        0.5 * start + 0.5 * end,
                        lower_domain,
                        upper_domain,
                        policy,
                    )
                )
                if end is not None and middle is not None:
                    points = np.stack((start, middle, end))
                    margin = (
                        64.0
                        * np.finfo(np.float64).eps
                        * (1.0 + np.max(np.abs(points), axis=0))
                    )
                    box_lower = np.min(points, axis=0) - margin
                    box_upper = np.max(points, axis=0) + margin
                    box_lower[axis], box_upper[axis] = lower_t, upper_t
                    inflation = _inflated_krawczyk(prepared, box_lower, box_upper, axis)
                    if inflation.certified:
                        segments.append(
                            (
                                lower_t,
                                upper_t,
                                start,
                                end,
                                inflation.lower,
                                inflation.upper,
                            )
                        )
                        break
                upper_t = middle_t
            lower_t, start = upper_t, segments[-1][3]
    node_values = [segments[0][0]] + [segment[1] for segment in segments]
    node_points = [segments[0][2]] + [segment[3] for segment in segments]
    node_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    free = np.asarray(_free_axis_indices(axis))
    for index, (point, value) in enumerate(zip(node_points, node_values, strict=True)):
        bounds = _certified_node_box(prepared, point, axis, value)
        if bounds is None:
            return None
        # The node's unique slice root must lie in both adjacent certified
        # graph boxes, which chains the segments into one branch.
        for segment in segments[max(index - 1, 0) : index + 1]:
            if np.any(bounds[0][free] < segment[4][free]) or np.any(
                bounds[1][free] > segment[5][free]
            ):
                return None
        node_bounds.append(bounds)
    return IntersectionCurve(
        first,
        second,
        chart_axes=np.full((len(segments),), axis, dtype=np.int32),
        chart_start=np.stack([segment[2] for segment in segments]),
        chart_end=np.stack([segment[3] for segment in segments]),
        chart_lower=np.stack([segment[4] for segment in segments]),
        chart_upper=np.stack([segment[5] for segment in segments]),
        node_axes=np.full((len(node_values),), axis, dtype=np.int32),
        node_values=np.asarray(node_values, dtype=np.float64),
        node_lower=np.stack([bounds[0] for bounds in node_bounds]),
        node_upper=np.stack([bounds[1] for bounds in node_bounds]),
        node_period_shifts=np.zeros((len(node_values), 2, 4), dtype=np.int32),
        closed=False,
        start_kind="boundary",
        end_kind="boundary",
    )


def _all_exclude_zero(lower: np.ndarray, upper: np.ndarray, /) -> bool:
    return bool(np.any((lower > 0.0) | (upper < 0.0)))


def _solve_slice_root(
    residual: Callable[[Array], Array],
    axis: int,
    value: float,
    initial: np.ndarray,
    lower_domain: np.ndarray,
    upper_domain: np.ndarray,
    policy: ParametricIntersectionPolicy,
    /,
) -> np.ndarray | None:
    from scipy.optimize import root

    free = list(_free_axis_indices(axis))
    template = initial.copy()
    template[axis] = value

    def free_residual(parameters: np.ndarray) -> np.ndarray:
        point = template.copy()
        point[free] = parameters
        return np.asarray(
            residual(jnp.asarray(point, dtype=jnp.float64)), dtype=np.float64
        )

    solution = root(free_residual, template[free])
    if not solution.success or not np.all(np.isfinite(solution.x)):
        return None
    point = template.copy()
    point[free] = solution.x
    residual_norm = float(np.linalg.norm(free_residual(solution.x)))
    scale = max(1.0, float(np.linalg.norm(point)))
    if (
        residual_norm > policy.relative_resolution * scale
        or np.any(point < lower_domain)
        or np.any(point > upper_domain)
    ):
        return None
    return point


__all__ = ["intersect_tspline_surface_regions"]
