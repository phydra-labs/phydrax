#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, nextafter, tau

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._interpolation import rational_bspline_surface_spans
from ..._strict import StrictModule
from ...discretization._coordinate_enclosure import ExactScalar
from ...discretization._exact_algebraic_geometry import coefficient_interval
from .._interval_enclosure import (
    HostArray,
    Interval,
    interval_add,
    interval_divide,
    interval_multiply,
    interval_sqrt,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import fraction_down, fraction_up
from ._intersection import (
    ParametricIntersectionPolicy,
    SurfaceIntersectionResult,
    UnresolvedIntersectionReason,
    UnresolvedParameterRegion,
)
from ._intersection_curve import IntersectionCurve, SurfacePairSystem, SurfaceRegion
from ._intersection_kernel import krawczyk, KrawczykEvaluator
from ._patches import (
    AbstractSurfacePatch,
    BSplineSurfacePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
)
from ._tspline import TSplineSurfacePatch


@dataclass(frozen=True, slots=True)
class _SplinePiece:
    bounds: tuple[Fraction, Fraction, Fraction, Fraction]
    values: PreparedIntervalFunction
    jacobians: PreparedIntervalFunction


class _SplineSource:
    def __init__(self, patch: TSplineSurfacePatch | BSplineSurfacePatch, /) -> None:
        self.patch = patch
        self.pieces = _spline_pieces(patch)

    def bounds(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        value_lower: np.ndarray | None = None
        value_upper: np.ndarray | None = None
        jacobian_lower: np.ndarray | None = None
        jacobian_upper: np.ndarray | None = None
        for piece in self.pieces:
            a0, a1, b0, b1 = piece.bounds
            piece_lower = np.asarray(
                (nextafter(float(a0), -np.inf), nextafter(float(b0), -np.inf))
            )
            piece_upper = np.asarray(
                (nextafter(float(a1), np.inf), nextafter(float(b1), np.inf))
            )
            lo = np.maximum(lower, piece_lower)
            hi = np.minimum(upper, piece_upper)
            if np.any(lo > hi):
                continue
            local_lower, local_upper = _piece_local_bounds(piece.bounds, lo, hi)
            value_lo, value_hi = piece.values.evaluate(
                local_lower[None], local_upper[None]
            )
            jac_lo, jac_hi = piece.jacobians.evaluate(
                local_lower[None], local_upper[None]
            )
            value_lower = (
                value_lo[0]
                if value_lower is None
                else np.minimum(value_lower, value_lo[0])
            )
            value_upper = (
                value_hi[0]
                if value_upper is None
                else np.maximum(value_upper, value_hi[0])
            )
            jacobian_lower = (
                jac_lo[0]
                if jacobian_lower is None
                else np.minimum(jacobian_lower, jac_lo[0])
            )
            jacobian_upper = (
                jac_hi[0]
                if jacobian_upper is None
                else np.maximum(jacobian_upper, jac_hi[0])
            )
        if (
            value_lower is None
            or value_upper is None
            or jacobian_lower is None
            or jacobian_upper is None
        ):
            raise ValueError(
                "A spline interval box does not overlap its exact Bézier partition."
            )
        return value_lower, value_upper, jacobian_lower, jacobian_upper


class _AnalyticSource:
    def __init__(self, patch: AbstractSurfacePatch, /) -> None:
        self.patch = patch
        self.values = prepare_interval_function(patch.evaluate, 2)
        self.jacobians = prepare_interval_function(jax.jacfwd(patch.evaluate), 2)

    def bounds(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        value_lower, value_upper = self.values.evaluate(lower[None], upper[None])
        jacobian_lower, jacobian_upper = self.jacobians.evaluate(lower[None], upper[None])
        return value_lower[0], value_upper[0], jacobian_lower[0], jacobian_upper[0]


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
            first = self.prepared.first_source.bounds(lo[row, :2], hi[row, :2])
            second = self.prepared.second_source.bounds(lo[row, 2:], hi[row, 2:])
            lower_value, upper_value = interval_subtract(
                (first[0], first[1]), (second[0], second[1])
            )
            result_lower[row], result_upper[row] = lower_value, upper_value
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
            first = self.prepared.first_source.bounds(lo[row, :2], hi[row, :2])
            second = self.prepared.second_source.bounds(lo[row, 2:], hi[row, 2:])
            result_lower[row, :, :2], result_upper[row, :, :2] = first[2], first[3]
            result_lower[row, :, 2:] = -second[3]
            result_upper[row, :, 2:] = -second[2]
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


type _BezierCoefficientBank = (
    Sequence[Sequence[ExactScalar]] | Sequence[Sequence[Sequence[ExactScalar]]]
)


def _spline_pieces(
    patch: TSplineSurfacePatch | BSplineSurfacePatch, /
) -> tuple[_SplinePiece, ...]:
    raw: list[
        tuple[
            tuple[Fraction, Fraction, Fraction, Fraction],
            _BezierCoefficientBank,
            int,
            int,
        ]
    ] = []
    if isinstance(patch, TSplineSurfacePatch):
        patch.validate_exact_transfer_chain()
        for extraction in patch.realization.extractions:
            rows = patch.realization.exact_homogeneous_piece(
                extraction.piece_id, patch.exact_target_homogeneous
            )
            net = tuple(tuple(row) for row in rows)
            raw.append((extraction.parameter_bounds, net, 3, 3))
    else:
        spans = rational_bspline_surface_spans(
            np.asarray(patch.control_points, dtype=np.float64).tolist(),
            np.asarray(patch.weights, dtype=np.float64).tolist(),
            np.asarray(patch.u_knots, dtype=np.float64).tolist(),
            np.asarray(patch.v_knots, dtype=np.float64).tolist(),
            patch.u_degree,
            patch.v_degree,
            exact_control_points=patch.exact_control_points,
            exact_weights=patch.exact_weights,
            exact_u_knots=patch.exact_u_knots,
            exact_v_knots=patch.exact_v_knots,
        )
        for span in spans:
            u0, u1 = span.parameter_bounds[0]
            v0, v1 = span.parameter_bounds[1]
            if (
                not isinstance(u0, Fraction)
                or not isinstance(u1, Fraction)
                or not isinstance(v0, Fraction)
                or not isinstance(v1, Fraction)
            ):
                raise ValueError("Exact spline span bounds must be rational.")
            raw.append(
                (
                    (u0, u1, v0, v1),
                    tuple(tuple(row) for row in span.homogeneous_bernstein_coefficients),
                    span.degrees[0],
                    span.degrees[1],
                )
            )
    return tuple(_prepare_bezier_piece(*piece) for piece in raw)


def _prepare_bezier_piece(
    bounds: tuple[Fraction, Fraction, Fraction, Fraction],
    coefficients: _BezierCoefficientBank,
    degree_u: int,
    degree_v: int,
    /,
) -> _SplinePiece:
    net = np.asarray(coefficients, dtype=object).reshape((degree_u + 1, degree_v + 1, 4))
    nominal_flat = np.empty((net.size,), dtype=np.float64)
    lower_flat = np.empty_like(nominal_flat)
    upper_flat = np.empty_like(nominal_flat)
    for index, coefficient in enumerate(net.reshape((-1,))):
        lower_exact, upper_exact = coefficient_interval(coefficient, bits=160)
        nominal_flat[index] = float(coefficient)
        lower_flat[index] = fraction_down(lower_exact)
        upper_flat[index] = fraction_up(upper_exact)
    nominal = nominal_flat.reshape(net.shape)
    lower = lower_flat.reshape(net.shape)
    upper = upper_flat.reshape(net.shape)
    controls = tuple(
        jnp.asarray(nominal[:, :, axis], dtype=jnp.float64) for axis in range(4)
    )

    def evaluate(parameters: Array) -> Array:
        first_basis = _bernstein_basis(parameters[0], degree_u)
        second_basis = _bernstein_basis(parameters[1], degree_v)
        basis = first_basis[:, None] * second_basis[None, :]
        homogeneous = tuple(jnp.sum(basis * control) for control in controls)
        return jnp.stack(tuple(homogeneous[axis] / homogeneous[3] for axis in range(3)))

    constant_bounds = tuple(
        (
            controls[axis],
            jnp.asarray(lower[:, :, axis]),
            jnp.asarray(upper[:, :, axis]),
        )
        for axis in range(4)
    )
    value = prepare_interval_function(evaluate, 2, constant_bounds=constant_bounds)
    jacobian = prepare_interval_function(
        jax.jacfwd(evaluate), 2, constant_bounds=constant_bounds
    )
    return _SplinePiece(bounds, value, jacobian)


def _bernstein_basis(value: Array, degree: int, /) -> Array:
    return jnp.stack(
        tuple(
            comb(degree, index) * value**index * (1.0 - value) ** (degree - index)
            for index in range(degree + 1)
        )
    )


def _piece_local_bounds(
    bounds: tuple[Fraction, Fraction, Fraction, Fraction],
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    result_lower = np.empty((2,), dtype=np.float64)
    result_upper = np.empty((2,), dtype=np.float64)
    for axis, (source_lower, source_upper) in enumerate(
        ((bounds[0], bounds[1]), (bounds[2], bounds[3]))
    ):
        width = source_upper - source_lower
        local_lower = (Fraction.from_float(float(lower[axis])) - source_lower) / width
        local_upper = (Fraction.from_float(float(upper[axis])) - source_lower) / width
        result_lower[axis] = max(0.0, nextafter(float(local_lower), -np.inf))
        result_upper[axis] = min(1.0, nextafter(float(local_upper), np.inf))
    return result_lower, result_upper


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
                *(
                    float(bound)
                    for piece in _spline_pieces(patch)
                    for bound in piece.bounds[:2]
                    if t0 < float(bound) < t1
                ),
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
        resolution = policy.relative_resolution * max(1.0, t1 - t0)
        while pending:
            lower_t, upper_t = pending.pop()
            start = level.point_at(lower_t, spline_first, other_region)
            end = level.point_at(upper_t, spline_first, other_region)
            bounds = level.graph_box(lower_t, upper_t, spline_first, other_region, policy)
            if bounds is not None:
                witness = krawczyk(
                    prepared, bounds[0][None], bounds[1][None], parameter_axis=axis
                )
                if bool(witness.certified[0]):
                    segments.append((lower_t, upper_t, start, end, bounds[0], bounds[1]))
                    continue
            middle = lower_t + 0.5 * (upper_t - lower_t)
            if upper_t - lower_t <= resolution or not lower_t < middle < upper_t:
                region_lower = np.concatenate(
                    (
                        np.asarray(first.parameter_box[0]),
                        np.asarray(second.parameter_box[0]),
                    )
                )
                region_upper = np.concatenate(
                    (
                        np.asarray(first.parameter_box[1]),
                        np.asarray(second.parameter_box[1]),
                    )
                )
                unresolved.append(
                    UnresolvedParameterRegion(region_lower, region_upper, "resolution")
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
            _certified_node_box(prepared, point, axis, value, first, second, policy)
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
        self.normal: Interval | None = None
        if isinstance(other, PlanePatch):
            self.normal = _plane_normal_interval(other)
            if self.normal is None:
                raise ValueError("Plane patch axes must span a plane.")

    def graph_orientation(self, /) -> int | None:
        lower = np.asarray(self.region.parameter_box[0], dtype=np.float64)
        upper = np.asarray(self.region.parameter_box[1], dtype=np.float64)
        derivative = self._derivative_bounds(lower, upper, 1)
        if derivative[0] > 0.0:
            orientation = 1
        elif derivative[1] < 0.0:
            orientation = -1
        else:
            return None
        first_boundary = self.value_bounds((lower[0], lower[1]), (upper[0], lower[1]))
        second_boundary = self.value_bounds((lower[0], upper[1]), (upper[0], upper[1]))
        if (
            orientation * first_boundary[1] >= 0.0
            or orientation * second_boundary[0] <= 0.0
        ):
            return None
        return orientation

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
        lo, hi, _, _ = self.source.bounds(np.asarray(lower), np.asarray(upper))
        point = self._implicit_value((lo, hi))
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
            self.spline.evaluate(jnp.asarray(parameters, dtype=jnp.float64)),
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
        point = np.asarray(
            self.spline.evaluate(jnp.asarray(parameters)), dtype=np.float64
        )
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
        self, spline_first: bool, other_region: SurfaceRegion, /
    ) -> tuple[np.ndarray, np.ndarray] | None:
        bounds = np.asarray(self.region.parameter_box, dtype=np.float64)
        value_lower, value_upper, jacobian_lower, jacobian_upper = self.source.bounds(
            bounds[0], bounds[1]
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
        policy: ParametricIntersectionPolicy,
        /,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        midpoint_t = lower_t + 0.5 * (upper_t - lower_t)
        midpoint = self.point_at(midpoint_t, spline_first, other_region)
        tangent = self._graph_tangent_bounds(spline_first, other_region)
        if tangent is None:
            return None
        slope = np.maximum(np.abs(tangent[0]), np.abs(tangent[1]))
        span = upper_t - lower_t
        margin = 64.0 * np.finfo(np.float64).eps * (1.0 + np.abs(midpoint))
        widths = slope * (0.5 * span) + margin
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
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    lower_domain = np.concatenate(
        (np.asarray(first.parameter_box[0]), np.asarray(second.parameter_box[0]))
    )
    upper_domain = np.concatenate(
        (np.asarray(first.parameter_box[1]), np.asarray(second.parameter_box[1]))
    )
    scale = max(1.0, float(np.max(np.abs(point))))
    radius = 64.0 * np.finfo(np.float64).eps * scale
    maximum = policy.relative_probe * scale
    while radius <= maximum:
        lower = np.maximum(lower_domain, point - radius)
        upper = np.minimum(upper_domain, point + radius)
        lower[axis] = min(value, lower[axis])
        upper[axis] = max(value, upper[axis])
        if np.all(lower < upper):
            witness = krawczyk(
                _node_preparation(prepared, axis, value), lower[None], upper[None]
            )
            if bool(witness.certified[0]):
                return lower, upper
        next_radius = radius * 2.0
        if next_radius <= radius:
            break
        radius = next_radius
    return None


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


def _free_coordinates(parameters: np.ndarray, axis: int, /) -> np.ndarray:
    first, second, third = _free_axis_indices(axis)
    return np.asarray(
        (parameters[first], parameters[second], parameters[third]),
        dtype=np.float64,
    )


def _intersect_spline_pair(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult:
    pair = SurfacePairSystem(first.patch, second.patch)
    prepared = _prepare_tspline_pair(pair, (first, second))
    lower_domain = np.concatenate(
        (np.asarray(first.parameter_box[0]), np.asarray(second.parameter_box[0]))
    )
    upper_domain = np.concatenate(
        (np.asarray(first.parameter_box[1]), np.asarray(second.parameter_box[1]))
    )
    if isinstance(first.patch, TSplineSurfacePatch):
        patch, axis = first.patch, 0
    elif isinstance(second.patch, TSplineSurfacePatch):
        patch, axis = second.patch, 2
    else:
        raise TypeError("A spline-pair route requires a T-spline operand.")
    breaks = sorted(
        {
            float(lower_domain[axis]),
            float(upper_domain[axis]),
            *(
                float(bound)
                for piece in _spline_pieces(patch)
                for bound in piece.bounds[:2]
                if lower_domain[axis] < float(bound) < upper_domain[axis]
            ),
        }
    )
    tolerance = policy.relative_resolution * np.maximum(1.0, upper_domain - lower_domain)
    roots_by_break: list[tuple[np.ndarray, ...]] = []
    unresolved: list[UnresolvedParameterRegion] = []
    for value in breaks:
        roots, unknown = _isolate_slice_roots(
            prepared,
            lower_domain,
            upper_domain,
            axis,
            value,
            tolerance,
        )
        roots_by_break.append(roots)
        unresolved.extend(unknown)
    if unresolved:
        return SurfaceIntersectionResult(unresolved=tuple(unresolved))
    if not any(roots_by_break):
        value_lower, value_upper = prepared.value.evaluate(
            lower_domain[None], upper_domain[None]
        )
        if _all_exclude_zero(value_lower[0], value_upper[0]):
            return SurfaceIntersectionResult()
        return _unresolved(first, second, "continuation")
    root_count = len(roots_by_break[0])
    if root_count == 0 or any(len(roots) != root_count for roots in roots_by_break[1:]):
        return _unresolved(first, second, "continuation")
    branch_points = [[root] for root in roots_by_break[0]]
    for slice_index in range(1, len(breaks)):
        remaining = list(roots_by_break[slice_index])
        for points in branch_points:
            if not remaining:
                return _unresolved(first, second, "continuation")
            previous = points[-1]
            matched_index = min(
                range(len(remaining)),
                key=lambda index: float(
                    np.linalg.norm(
                        _free_coordinates(remaining[index], axis)
                        - _free_coordinates(previous, axis)
                    )
                ),
            )
            points.append(remaining.pop(matched_index))
        if remaining:
            return _unresolved(first, second, "continuation")
    branches: list[IntersectionCurve] = []
    for points in branch_points:
        pending = [
            (breaks[index], breaks[index + 1], points[index], points[index + 1])
            for index in range(len(breaks) - 1)
        ]
        segments: list[
            tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        while pending:
            lower_t, upper_t, start_point, end_point = pending.pop()
            middle_t = lower_t + 0.5 * (upper_t - lower_t)
            middle_point = _solve_slice_root(
                pair,
                axis,
                middle_t,
                0.5 * (start_point + end_point),
                lower_domain,
                upper_domain,
                policy,
            )
            bounds = (
                None
                if middle_point is None
                else _graph_root_box(
                    axis,
                    lower_t,
                    upper_t,
                    start_point,
                    middle_point,
                    end_point,
                    lower_domain,
                    upper_domain,
                    policy,
                )
            )
            if bounds is not None:
                witness = krawczyk(
                    prepared,
                    bounds[0][None],
                    bounds[1][None],
                    parameter_axis=axis,
                )
                if bool(witness.certified[0]):
                    segments.append(
                        (
                            lower_t,
                            upper_t,
                            start_point,
                            end_point,
                            bounds[0],
                            bounds[1],
                        )
                    )
                    continue
            if (
                middle_point is None
                or upper_t - lower_t <= tolerance[axis]
                or not lower_t < middle_t < upper_t
            ):
                return _unresolved(first, second, "resolution")
            pending.append((middle_t, upper_t, middle_point, end_point))
            pending.append((lower_t, middle_t, start_point, middle_point))
        segments.sort(key=lambda item: item[0])
        node_values = [segments[0][0]] + [segment[1] for segment in segments]
        node_points = [segments[0][2]] + [segment[3] for segment in segments]
        node_bounds = tuple(
            _certified_node_box(prepared, point, axis, value, first, second, policy)
            for point, value in zip(node_points, node_values, strict=True)
        )
        if any(bounds is None for bounds in node_bounds):
            return _unresolved(first, second, "resolution")
        certified_nodes = tuple(bounds for bounds in node_bounds if bounds is not None)
        branches.append(
            IntersectionCurve(
                first,
                second,
                chart_axes=np.full((len(segments),), axis, dtype=np.int32),
                chart_start=np.stack([segment[2] for segment in segments]),
                chart_end=np.stack([segment[3] for segment in segments]),
                chart_lower=np.stack([segment[4] for segment in segments]),
                chart_upper=np.stack([segment[5] for segment in segments]),
                node_axes=np.full((len(node_values),), axis, dtype=np.int32),
                node_values=np.asarray(node_values, dtype=np.float64),
                node_lower=np.stack([bounds[0] for bounds in certified_nodes]),
                node_upper=np.stack([bounds[1] for bounds in certified_nodes]),
                node_period_shifts=np.zeros((len(node_values), 2, 4), dtype=np.int32),
                closed=False,
                start_kind="boundary",
                end_kind="boundary",
            )
        )
    return SurfaceIntersectionResult(curves=tuple(branches))


def _isolate_slice_roots(
    prepared: KrawczykEvaluator,
    lower_domain: np.ndarray,
    upper_domain: np.ndarray,
    axis: int,
    value: float,
    tolerance: np.ndarray,
    /,
) -> tuple[tuple[np.ndarray, ...], tuple[UnresolvedParameterRegion, ...]]:
    lower = lower_domain.copy()
    upper = upper_domain.copy()
    lower[axis] = upper[axis] = value
    pending = [(lower, upper)]
    roots: list[np.ndarray] = []
    unresolved: list[UnresolvedParameterRegion] = []
    while pending:
        lower, upper = pending.pop()
        value_lower, value_upper = prepared.value.evaluate(lower[None], upper[None])
        if _all_exclude_zero(value_lower[0], value_upper[0]):
            continue
        witness = krawczyk(prepared, lower[None], upper[None], parameter_axis=axis)
        if bool(witness.certified[0]):
            point = lower.copy()
            free_axes = _free_axis_indices(axis)
            free_root = np.asarray(
                0.5 * witness.lower[0] + 0.5 * witness.upper[0],
                dtype=np.float64,
            )
            for free_index, parameter_axis in enumerate(free_axes):
                point[parameter_axis] = float(free_root[free_index])
            if not any(np.all(np.abs(point - old) <= tolerance) for old in roots):
                roots.append(point)
            continue
        if bool(witness.excluded[0]):
            continue
        widths = upper - lower
        widths[axis] = 0.0
        if np.all((widths <= tolerance) | (widths == 0.0)):
            unresolved.append(UnresolvedParameterRegion(lower, upper, "resolution"))
            continue
        normalized = np.where(
            widths > 0.0,
            widths / np.maximum(upper_domain - lower_domain, np.finfo(np.float64).tiny),
            -1.0,
        )
        split_axis = int(np.argmax(normalized))
        middle = lower[split_axis] + 0.5 * widths[split_axis]
        if not lower[split_axis] < middle < upper[split_axis]:
            unresolved.append(UnresolvedParameterRegion(lower, upper, "resolution"))
            continue
        left_upper = upper.copy()
        left_upper[split_axis] = middle
        right_lower = lower.copy()
        right_lower[split_axis] = middle
        pending.append((right_lower, upper.copy()))
        pending.append((lower.copy(), left_upper))
    roots.sort(key=lambda point: tuple(float(value) for value in point))
    return tuple(roots), tuple(unresolved)


def _all_exclude_zero(lower: np.ndarray, upper: np.ndarray, /) -> bool:
    return bool(np.any((lower > 0.0) | (upper < 0.0)))


def _solve_slice_root(
    pair: SurfacePairSystem,
    axis: int,
    value: float,
    initial: np.ndarray,
    lower_domain: np.ndarray,
    upper_domain: np.ndarray,
    policy: ParametricIntersectionPolicy,
    /,
) -> np.ndarray | None:
    from scipy.optimize import root

    free = _free_axis_indices(axis)
    template = initial.copy()
    template[axis] = value
    initial_free = np.asarray(
        (template[free[0]], template[free[1]], template[free[2]]),
        dtype=np.float64,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        point = template.copy()
        for index, parameter_axis in enumerate(free):
            point[parameter_axis] = float(parameters[index])
        return np.asarray(
            pair.residual(jnp.asarray(point, dtype=jnp.float64)), dtype=np.float64
        )

    solution = root(residual, initial_free)
    if not solution.success or not np.all(np.isfinite(solution.x)):
        return None
    point = template.copy()
    for index, parameter_axis in enumerate(free):
        point[parameter_axis] = float(solution.x[index])
    residual_norm = float(np.linalg.norm(residual(solution.x)))
    scale = max(1.0, float(np.linalg.norm(point)))
    if (
        residual_norm > policy.relative_resolution * scale
        or np.any(point < lower_domain)
        or np.any(point > upper_domain)
    ):
        return None
    return point


def _graph_root_box(
    axis: int,
    lower_t: float,
    upper_t: float,
    first: np.ndarray,
    middle: np.ndarray,
    last: np.ndarray,
    lower_domain: np.ndarray,
    upper_domain: np.ndarray,
    policy: ParametricIntersectionPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.stack((first, middle, last))
    center = middle
    span = upper_t - lower_t
    margin = np.maximum(
        128.0 * np.finfo(np.float64).eps * (1.0 + np.abs(center)),
        policy.relative_probe * span * (1.0 + np.abs(center)),
    )
    lower = np.maximum(lower_domain, np.min(points, axis=0) - margin)
    upper = np.minimum(upper_domain, np.max(points, axis=0) + margin)
    lower[axis], upper[axis] = lower_t, upper_t
    if np.any(lower >= upper):
        return None
    return lower, upper


__all__ = ["intersect_tspline_surface_regions"]
