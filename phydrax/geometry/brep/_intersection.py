#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native interval-certified curve and surface intersection source roots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, cast, Literal, Protocol, Sequence, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    coefficient_payload,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ...typing import checked, parse
from .._interval_enclosure import (
    prepare_interval_function,
)
from .._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
)
from ._intersection_curve import (
    _DisjointOrthogonalCylinderFamily,
    _OrthogonalCylinderFamily,
    _SingularOrthogonalCylinderFamily,
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionEndpointKind,
    IntersectionPCurve,
    IntersectionSide,
    interval_context,
    orthogonal_cylinder_family,
    SurfacePairSystem,
    SurfaceRegion,
)
from ._intersection_kernel import IntervalEvaluator, krawczyk, KrawczykEvaluator
from ._patches import (
    _combine_conic_parameter_roots,
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    CylinderPatch,
    EllipseCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from ._trim_helpers import period_offset, period_scalar_bounds, PeriodOffset


ParametricIntersectionKind: TypeAlias = Literal[
    "crossing", "tangent_point", "tangent_curve", "singular_point"
]
ParametricIntersectionCertificate: TypeAlias = Literal[
    "krawczyk", "tangency-krawczyk", "analytic-contact"
]
UnresolvedIntersectionReason: TypeAlias = Literal[
    "resolution", "singular", "continuation"
]


class _SourceEdgeLift(Protocol):
    """Structural source correspondence accepted by joint-root image proofs."""

    curve: AbstractCurve
    surface: AbstractSurfacePatch
    pcurve: AbstractCurve
    first: float
    last: float


class ParametricIntersectionPolicy(StrictModule):
    """Physical coincidence/tangency and finite mathematical resolution controls."""

    relative_resolution: float = eqx.field(static=True)
    relative_probe: float = eqx.field(static=True)
    relative_coincidence: float = eqx.field(static=True)
    relative_tangency: float = eqx.field(static=True)
    relative_junction: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_resolution: float = 1.0e-10,
        relative_probe: float = 1.0e-2,
        relative_coincidence: float = 1.0e-11,
        relative_tangency: float = 1.0e-10,
        relative_junction: float = 1.0e-3,
    ) -> None:
        values = (
            relative_resolution,
            relative_probe,
            relative_coincidence,
            relative_tangency,
            relative_junction,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Intersection tolerances and resolution must be finite and positive."
            )
        if relative_resolution >= relative_probe:
            raise ValueError("relative_resolution must be smaller than relative_probe.")
        self.relative_resolution = float(relative_resolution)
        self.relative_probe = float(relative_probe)
        self.relative_coincidence = float(relative_coincidence)
        self.relative_tangency = float(relative_tangency)
        self.relative_junction = float(relative_junction)


class ParametricIntersectionPoint(StrictModule):
    """A root equation plus isolating source-parameter box and rounded realization."""

    kind: ParametricIntersectionKind = eqx.field(static=True)
    certificate: ParametricIntersectionCertificate = eqx.field(static=True)
    parameters: Array
    parameter_lower: Array
    parameter_upper: Array
    point: Array
    point_exact: tuple[Fraction, ...] | None = eqx.field(static=True)
    gap_bound: float = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
    branch_count: int = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: ParametricIntersectionKind,
        certificate: ParametricIntersectionCertificate,
        parameters: ArrayLike,
        parameter_lower: ArrayLike,
        parameter_upper: ArrayLike,
        point: ArrayLike,
        point_exact: tuple[Fraction, ...] | None = None,
        gap_bound: float,
        condition_estimate: float,
        branch_count: int = -1,
        root_id: str | None = None,
    ) -> None:
        params = np.asarray(parameters, dtype=np.float64)
        lower = np.asarray(parameter_lower, dtype=np.float64)
        upper = np.asarray(parameter_upper, dtype=np.float64)
        xyz = np.asarray(point, dtype=np.float64)
        if params.ndim != 1 or lower.shape != params.shape or upper.shape != params.shape:
            raise ValueError(
                "Intersection point parameters and isolating bounds must be aligned vectors."
            )
        if (
            np.any(~np.isfinite(params))
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError("Intersection point root boxes must be finite and ordered.")
        if xyz.ndim != 1 or not np.all(np.isfinite(xyz)):
            raise ValueError("Intersection point realization must be a finite vector.")
        exact_point = None if point_exact is None else tuple(point_exact)
        if exact_point is not None:
            if len(exact_point) != xyz.shape[0] or any(
                not isinstance(value, Fraction) for value in exact_point
            ):
                raise TypeError(
                    "point_exact must contain one Fraction per spatial coordinate."
                )
            exact_rounding = np.asarray(
                tuple(float(value) for value in exact_point), dtype=np.float64
            )
            if not np.array_equal(exact_rounding, xyz):
                raise ValueError(
                    "Rounded point must realize its exact Fraction source root."
                )
        if not math.isfinite(gap_bound) or gap_bound < 0 or branch_count < -1:
            raise ValueError("Intersection gap and branch count are invalid.")
        self.kind, self.certificate = kind, certificate
        self.parameters = jnp.asarray(params)
        self.parameter_lower = jnp.asarray(lower)
        self.parameter_upper = jnp.asarray(upper)
        self.point = jnp.asarray(xyz)
        self.point_exact = exact_point
        self.gap_bound = float(gap_bound)
        self.condition_estimate = float(condition_estimate)
        self.branch_count = int(branch_count)
        self.root_id = root_id or canonical_fingerprint(
            {
                "kind": "parametric-intersection-root",
                "parameters": tuple(float(value).hex() for value in params),
                "lower": tuple(float(value).hex() for value in lower),
                "upper": tuple(float(value).hex() for value in upper),
                "point_exact": None
                if exact_point is None
                else tuple((value.numerator, value.denominator) for value in exact_point),
            }
        )


class CoincidentParameterRegion(StrictModule):
    """Certified parameter cells in which the source operands coincide."""

    cell_lower: Array
    cell_upper: Array
    partial: Array
    distance_bound: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        cell_lower: ArrayLike,
        cell_upper: ArrayLike,
        partial: ArrayLike,
        distance_bound: float,
    ) -> None:
        lower = np.asarray(cell_lower, dtype=np.float64)
        upper = np.asarray(cell_upper, dtype=np.float64)
        partial_ = np.asarray(partial, dtype=np.bool_)
        if (
            lower.ndim != 2
            or upper.shape != lower.shape
            or partial_.shape != (lower.shape[0],)
        ):
            raise ValueError("Coincident parameter cells must have aligned arrays.")
        if (
            np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError("Coincident parameter cells must be finite and ordered.")
        if not math.isfinite(distance_bound) or distance_bound < 0.0:
            raise ValueError("Coincident distance bound must be finite and nonnegative.")
        self.cell_lower = jnp.asarray(lower)
        self.cell_upper = jnp.asarray(upper)
        self.partial = jnp.asarray(partial_)
        self.distance_bound = float(distance_bound)


class UnresolvedParameterRegion(StrictModule):
    lower: Array
    upper: Array
    reason: UnresolvedIntersectionReason = eqx.field(static=True)

    def __init__(
        self, lower: ArrayLike, upper: ArrayLike, reason: UnresolvedIntersectionReason
    ) -> None:
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
        if (
            lo.ndim != 1
            or hi.shape != lo.shape
            or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi))
            or np.any(lo > hi)
        ):
            raise ValueError("Unresolved parameter boxes must be finite and ordered.")
        reason_ = parse(reason, UnresolvedIntersectionReason, "reason")
        self.lower, self.upper, self.reason = (
            jnp.asarray(lo),
            jnp.asarray(hi),
            reason_,
        )


class CurveIntersectionResult(StrictModule):
    points: tuple[ParametricIntersectionPoint, ...]
    coincident: tuple[CoincidentParameterRegion, ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]

    def __init__(
        self,
        points: Sequence[ParametricIntersectionPoint] = (),
        coincident: Sequence[CoincidentParameterRegion] = (),
        unresolved: Sequence[UnresolvedParameterRegion] = (),
    ) -> None:
        self.points, self.coincident, self.unresolved = (
            tuple(points),
            tuple(coincident),
            tuple(unresolved),
        )

    @property
    def complete(self) -> bool:
        return not self.unresolved and not any(
            np.any(region.partial) for region in self.coincident
        )


class SurfaceIntersectionResult(StrictModule):
    curves: tuple[IntersectionCurve, ...]
    points: tuple[ParametricIntersectionPoint, ...]
    coincident: tuple[CoincidentParameterRegion, ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]

    def __init__(
        self,
        curves: Sequence[IntersectionCurve] = (),
        points: Sequence[ParametricIntersectionPoint] = (),
        coincident: Sequence[CoincidentParameterRegion] = (),
        unresolved: Sequence[UnresolvedParameterRegion] = (),
    ) -> None:
        self.curves, self.points = tuple(curves), tuple(points)
        self.coincident, self.unresolved = tuple(coincident), tuple(unresolved)

    @property
    def complete(self) -> bool:
        return (
            not self.unresolved
            and not any(np.any(region.partial) for region in self.coincident)
            and all(
                curve.fully_certified
                and curve.start_kind != "uncertified"
                and curve.end_kind != "uncertified"
                for curve in self.curves
            )
        )


class CurveRange(StrictModule):
    curve: AbstractCurve
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)

    @checked
    def __init__(
        self, curve: AbstractCurve, first: float | None = None, last: float | None = None
    ) -> None:
        if not isinstance(curve, AbstractCurve):
            raise TypeError("CurveRange.curve must be an AbstractCurve.")
        domain = curve.parameter_domain
        period = curve.period
        lower = (
            (domain[0] if domain is not None else 0.0) if first is None else float(first)
        )
        upper = (
            (
                domain[1]
                if domain is not None
                else (float(period) if period is not None else 1.0)
            )
            if last is None
            else float(last)
        )
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError("CurveRange endpoints must be finite and nondecreasing.")
        if domain is not None and (lower < domain[0] or upper > domain[1]):
            raise ValueError("CurveRange must lie inside its source parameter domain.")
        self.curve, self.first, self.last = curve, lower, upper


def _curve_chart_affine(curve: AbstractCurve, /) -> tuple[Fraction, Fraction, int] | None:
    if isinstance(curve, IntersectionPCurve):
        if curve.reversed:
            return (
                Fraction(-1),
                Fraction.from_float(float(curve.first + curve.last)),
                curve.curve.num_charts,
            )
        return Fraction(1), Fraction(0), curve.curve.num_charts
    if isinstance(curve, CurveTrimSegment):
        source = _curve_chart_affine(curve.curve)
        if source is None:
            return None
        inner_scale, inner_offset, count = source
        if curve.reversed:
            scale = Fraction.from_float(float(curve.first - curve.last))
            offset = Fraction.from_float(float(curve.last))
        else:
            scale = Fraction.from_float(float(curve.last - curve.first))
            offset = Fraction.from_float(float(curve.first))
        return inner_scale * scale, inner_scale * offset + inner_offset, count
    return None


def _curve_chart_intervals(
    curve: AbstractCurve, first: float, last: float, /
) -> tuple[tuple[float, float, int | None], ...] | None:
    affine = _curve_chart_affine(curve)
    if affine is None:
        return ((first, last, None),)
    scale, offset, count = affine
    if scale == 0:
        return None
    parameter_first = Fraction.from_float(float(first))
    parameter_last = Fraction.from_float(float(last))
    branch_first = scale * parameter_first + offset
    branch_last = scale * parameter_last + offset
    branch_lower, branch_upper = sorted((branch_first, branch_last))
    first_boundary = branch_lower.numerator // branch_lower.denominator + 1
    stop_boundary = math.ceil(branch_upper)
    cuts = [float(first), float(last)]
    for boundary in range(first_boundary, stop_boundary):
        parameter = (Fraction(boundary) - offset) / scale
        if parameter_first < parameter < parameter_last:
            numeric = float(parameter)
            if Fraction.from_float(numeric) != parameter:
                return None
            cuts.append(numeric)
    cuts.sort()
    segments: list[tuple[float, float, int | None]] = []
    for lower, upper in zip(cuts[:-1], cuts[1:], strict=True):
        middle = (Fraction.from_float(lower) + Fraction.from_float(upper)) / 2
        source_middle = scale * middle + offset
        chart = source_middle.numerator // source_middle.denominator
        if chart == count:
            chart -= 1
        source_ends = (
            scale * Fraction.from_float(lower) + offset,
            scale * Fraction.from_float(upper) + offset,
        )
        if (
            chart < 0
            or chart >= count
            or min(source_ends) < chart
            or max(source_ends) > chart + 1
        ):
            return None
        segments.append((lower, upper, int(chart)))
    return tuple(segments)


def _curve_value_on_chart(
    curve: AbstractCurve, parameter: ArrayLike, chart: int | None, /
) -> Array:
    parameter_ = jnp.asarray(parameter, dtype=jnp.float64)
    if chart is None:
        return curve.evaluate(parameter_)
    if isinstance(curve, IntersectionPCurve):
        return curve._evaluate_on_chart(parameter_, chart)
    if isinstance(curve, CurveTrimSegment):
        return _curve_value_on_chart(
            curve.curve, curve.source_parameters(parameter_), chart
        )
    raise ValueError("A fixed chart index requires a chart-bearing source curve.")


def _curve_chart_evaluator(
    curve: AbstractCurve, chart: int | None, /
) -> Callable[[Array], Array]:
    if isinstance(curve, CurveTrimSegment):
        source = _curve_chart_evaluator(curve.curve, chart)
        return lambda parameter: source(
            jnp.asarray(curve.source_parameters(parameter), dtype=jnp.float64)
        )
    if chart is None:
        return curve.evaluate
    if isinstance(curve, IntersectionPCurve):
        return curve._chart_evaluator(chart)
    raise ValueError("A fixed chart index requires a chart-bearing source curve.")


def _curve_chart_periods(
    curve: AbstractCurve, /
) -> tuple[tuple[float | None, float | None], tuple[str | None, str | None]] | None:
    if isinstance(curve, IntersectionPCurve):
        patch = (
            curve.curve.first.patch if curve.side == "first" else curve.curve.second.patch
        )
        return patch.periods, patch.native_period_symbols()
    if isinstance(curve, CurveTrimSegment):
        return _curve_chart_periods(curve.curve)
    return None


def _periodic_boxes_overlap(
    first_lower: float,
    first_upper: float,
    second_lower: float,
    second_upper: float,
    /,
) -> bool:
    first_lower_exact = Fraction.from_float(float(first_lower))
    first_upper_exact = Fraction.from_float(float(first_upper))
    second_lower_exact = Fraction.from_float(float(second_lower))
    second_upper_exact = Fraction.from_float(float(second_upper))
    period_lower, period_upper = period_scalar_bounds(
        PeriodOffset(Fraction(0), Fraction(1))
    )
    if (
        first_upper_exact - first_lower_exact >= period_upper
        or second_upper_exact - second_lower_exact >= period_upper
    ):
        return True
    difference_lower = second_lower_exact - first_upper_exact
    difference_upper = second_upper_exact - first_lower_exact
    shifts: list[int] = []
    if difference_upper >= 0:
        lower_shift = max(0, math.ceil(difference_lower / period_upper))
        upper_shift = math.floor(difference_upper / period_lower)
        shifts.extend(range(lower_shift, upper_shift + 1))
    if difference_lower <= 0:
        lower_shift = math.ceil(difference_lower / period_lower)
        upper_shift = min(-1, math.floor(difference_upper / period_upper))
        shifts.extend(range(lower_shift, upper_shift + 1))
    for shift in shifts:
        shift_lower, shift_upper = period_scalar_bounds(
            PeriodOffset(Fraction(0), Fraction(shift))
        )
        if (
            first_lower_exact + shift_lower <= second_upper_exact
            and second_lower_exact <= first_upper_exact + shift_upper
        ):
            return True
    return False


def _trim_chart_boxes_disjoint(
    first: AbstractCurve,
    first_lower: float,
    first_upper: float,
    second: AbstractCurve,
    second_lower: float,
    second_upper: float,
    source_patch: AbstractSurfacePatch | None,
    /,
) -> bool:
    try:
        first_box = np.asarray(
            first.bounding_box(first_lower, first_upper), dtype=np.float64
        )
        second_box = np.asarray(
            second.bounding_box(second_lower, second_upper), dtype=np.float64
        )
    except NotImplementedError:
        return False
    if (
        first_box.shape != (2, 2)
        or second_box.shape != (2, 2)
        or not np.all(np.isfinite(first_box))
        or not np.all(np.isfinite(second_box))
    ):
        return False
    if source_patch is not None:
        periods = source_patch.periods
        symbols = source_patch.native_period_symbols()
    else:
        first_periods = _curve_chart_periods(first)
        second_periods = _curve_chart_periods(second)
        if first_periods is None and second_periods is None:
            return False
        periods = tuple(
            (
                first_periods[0][axis]
                if first_periods is not None
                else second_periods[0][axis]
                if second_periods is not None
                else None
            )
            for axis in range(2)
        )
        symbols = tuple(
            "two_pi"
            if (
                (first_periods is not None and first_periods[1][axis] == "two_pi")
                or (second_periods is not None and second_periods[1][axis] == "two_pi")
            )
            else None
            for axis in range(2)
        )
    for axis in range(2):
        if symbols[axis] == "two_pi" and periods[axis] is not None:
            if not _periodic_boxes_overlap(
                first_box[0, axis],
                first_box[1, axis],
                second_box[0, axis],
                second_box[1, axis],
            ):
                return True
        elif (
            first_box[1, axis] < second_box[0, axis]
            or second_box[1, axis] < first_box[0, axis]
        ):
            return True
    return False


class _CurveSurfaceSourceBounds(StrictModule):
    """Source enclosures, distinct from curve and surface execution shadows."""

    curve: AbstractCurve
    surface: AbstractSurfacePatch
    derivative: bool = eqx.field(static=True)

    def __init__(
        self, curve: AbstractCurve, surface: AbstractSurfacePatch, derivative: bool, /
    ) -> None:
        if not isinstance(curve, AbstractCurve) or not isinstance(
            surface, AbstractSurfacePatch
        ):
            raise TypeError("Curve/surface bounds require native source carriers.")
        if not isinstance(derivative, bool):
            raise TypeError("The derivative role must be a bool.")
        if curve.ambient_dimension != 3:
            raise ValueError("Curve/surface bounds require a spatial curve.")
        self.curve, self.surface, self.derivative = curve, surface, derivative

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        if lo.ndim != 2 or lo.shape[1] != 3 or hi.shape != lo.shape:
            raise ValueError("Curve/surface boxes require aligned (count, 3) arrays.")
        shape = (lo.shape[0], 3, 3) if self.derivative else (lo.shape[0], 3)
        result_lower = np.empty(shape, dtype=np.float64)
        result_upper = np.empty(shape, dtype=np.float64)
        from .._interval_enclosure import interval_subtract

        for row in range(lo.shape[0]):
            first, last = float(lo[row, 0]), float(hi[row, 0])
            box = np.stack((lo[row, 1:], hi[row, 1:]))
            if self.derivative:
                curve_lower, curve_upper = self.curve.derivative_bounds(first, last)
                surface_lower, surface_upper = self.surface.derivative_bounds(box)
                if (
                    np.shape(curve_lower) != (3,)
                    or np.shape(curve_upper) != (3,)
                    or np.shape(surface_lower) != (3, 2)
                    or np.shape(surface_upper) != (3, 2)
                ):
                    raise ValueError(
                        "Source derivative enclosures have incompatible shapes."
                    )
                result_lower[row, :, 0], result_upper[row, :, 0] = (
                    curve_lower,
                    curve_upper,
                )
                result_lower[row, :, 1:] = -surface_upper
                result_upper[row, :, 1:] = -surface_lower
            else:
                curve_box = np.asarray(
                    self.curve.bounding_box(first, last), dtype=np.float64
                )
                surface_box = np.asarray(self.surface.bounding_box(box), dtype=np.float64)
                if curve_box.shape != (2, 3) or surface_box.shape != (2, 3):
                    raise ValueError("Source point enclosures have incompatible shapes.")
                result_lower[row], result_upper[row] = interval_subtract(
                    (curve_box[0], curve_box[1]), (surface_box[0], surface_box[1])
                )
        if (
            not np.all(np.isfinite(result_lower))
            or not np.all(np.isfinite(result_upper))
            or np.any(result_lower > result_upper)
        ):
            raise ValueError("Source enclosures must be finite and ordered.")
        return result_lower, result_upper


class _LinearTrimSourceBounds(StrictModule):
    """Enclose affine chart equations without treating root-angle shadows as exact."""

    first: AbstractCurve
    second: AbstractCurve
    derivative: bool = eqx.field(static=True)
    jacobian_lower: np.ndarray
    jacobian_upper: np.ndarray

    def __init__(
        self, first: AbstractCurve, second: AbstractCurve, derivative: bool, /
    ) -> None:
        from ._patches import _linear_pcurve_velocity

        first_velocity = _linear_pcurve_velocity(first)
        second_velocity = _linear_pcurve_velocity(second)
        if first_velocity is None or second_velocity is None:
            raise ValueError("Exact affine trim bounds require linear chart actions.")
        lower = np.empty((2, 2), dtype=np.float64)
        upper = np.empty((2, 2), dtype=np.float64)
        for axis in range(2):
            for column, value in enumerate(
                (first_velocity[axis], -second_velocity[axis])
            ):
                first_bound, last_bound = coefficient_interval(value, bits=160)
                lower[axis, column] = fraction_down(first_bound)
                upper[axis, column] = fraction_up(last_bound)
        self.first, self.second, self.derivative = first, second, derivative
        self.jacobian_lower, self.jacobian_upper = lower, upper

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._interval_enclosure import interval_subtract

        lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        if lo.ndim != 2 or lo.shape[1] != 2 or hi.shape != lo.shape:
            raise ValueError("Affine trim boxes require aligned (count, 2) arrays.")
        if self.derivative:
            shape = (lo.shape[0], 2, 2)
            return (
                np.broadcast_to(self.jacobian_lower, shape),
                np.broadcast_to(self.jacobian_upper, shape),
            )
        result_lower = np.empty((lo.shape[0], 2), dtype=np.float64)
        result_upper = np.empty_like(result_lower)
        for row in range(lo.shape[0]):
            first = np.asarray(
                self.first.bounding_box(float(lo[row, 0]), float(hi[row, 0])),
                dtype=np.float64,
            )
            second = np.asarray(
                self.second.bounding_box(float(lo[row, 1]), float(hi[row, 1])),
                dtype=np.float64,
            )
            if first.shape != (2, 2) or second.shape != (2, 2):
                raise ValueError("Affine trim source bounds must have shape (2, 2).")
            result_lower[row], result_upper[row] = interval_subtract(
                (first[0], first[1]), (second[0], second[1])
            )
        if (
            not np.all(np.isfinite(result_lower))
            or not np.all(np.isfinite(result_upper))
            or np.any(result_lower > result_upper)
        ):
            raise ValueError("Affine trim source enclosures must be finite and ordered.")
        return result_lower, result_upper


def _curve_surface_residual(
    curve: AbstractCurve, surface: AbstractSurfacePatch, values: Array, /
) -> Array:
    return curve.evaluate(values[0]) - surface.evaluate(values[1:])


class _RootEquation(StrictModule):
    function: Callable[[Array], Array]
    dimension: int = eqx.field(static=True)
    value: IntervalEvaluator
    jacobian: IntervalEvaluator

    def __init__(
        self,
        function: Callable[[Array], Array],
        dimension: int,
        *,
        source_value: IntervalEvaluator | None = None,
        source_jacobian: IntervalEvaluator | None = None,
    ) -> None:
        self.function = function
        self.dimension = dimension
        self.value = (
            prepare_interval_function(function, dimension)
            if source_value is None
            else source_value
        )
        self.jacobian = (
            prepare_interval_function(jax.jacfwd(function), dimension)
            if source_jacobian is None
            else source_jacobian
        )

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        values = jnp.asarray(points, dtype=jnp.float64)
        return np.asarray(jax.vmap(jax.jacfwd(self.function))(values), dtype=np.float64)


def _root_witness(
    system: _RootEquation, lower: ArrayLike, upper: ArrayLike, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != (system.dimension,) or hi.shape != lo.shape or np.any(lo >= hi):
        raise ValueError(
            "A root isolation box must have positive finite width in every parameter."
        )
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("A root isolation box must be finite.")
    result = krawczyk(system, lo[None], hi[None])
    if not bool(result.certified[0]):
        raise ValueError("The source root box does not certify a unique regular root.")
    contracted_lower = np.maximum(lo, result.lower[0])
    contracted_upper = np.minimum(hi, result.upper[0])
    point = 0.5 * contracted_lower + 0.5 * contracted_upper
    return (
        point,
        contracted_lower,
        contracted_upper,
        result.preconditioner[0],
        float(result.contraction[0]),
    )


class TrimIntersectionRoot(StrictModule):
    first: AbstractCurve
    second: AbstractCurve
    parameter_lower: Array
    parameter_upper: Array
    parameters: Array
    enclosure_lower: Array
    enclosure_upper: Array
    preconditioner: Array
    endpoint_witness: ConicRootLineEndpointWitness | None
    contraction: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        first: AbstractCurve,
        second: AbstractCurve,
        /,
        *,
        parameter_lower: ArrayLike,
        parameter_upper: ArrayLike,
        endpoint_witness: ConicRootLineEndpointWitness | None = None,
    ) -> None:
        if (
            first.ambient_dimension != second.ambient_dimension
            or first.ambient_dimension != 2
        ):
            raise ValueError("TrimIntersectionRoot requires two planar source curves.")
        lower_input = np.asarray(parameter_lower, dtype=np.float64)
        upper_input = np.asarray(parameter_upper, dtype=np.float64)
        first_segments = _curve_chart_intervals(
            first, float(lower_input[0]), float(upper_input[0])
        )
        second_segments = _curve_chart_intervals(
            second, float(lower_input[1]), float(upper_input[1])
        )
        if (
            first_segments is None
            or second_segments is None
            or len(first_segments) != 1
            or len(second_segments) != 1
        ):
            raise ValueError(
                "TrimIntersectionRoot bounds must remain inside one certified source chart."
            )
        first_chart, second_chart = first_segments[0][2], second_segments[0][2]
        if endpoint_witness is not None:
            if not isinstance(endpoint_witness, ConicRootLineEndpointWitness):
                raise TypeError(
                    "TrimIntersectionRoot.endpoint_witness must be a "
                    "ConicRootLineEndpointWitness or None."
                )
            if not endpoint_witness.binds_curves(first, second):
                raise ValueError(
                    "A conic endpoint witness must bind the exact trim operands."
                )
            witness_box = endpoint_witness.parameter_box()
            if np.any(lower_input > witness_box[0]) or np.any(
                upper_input < witness_box[1]
            ):
                raise ValueError(
                    "TrimIntersectionRoot bounds must enclose the analytic endpoint image."
                )

        def residual(values: Array) -> Array:
            return _curve_value_on_chart(
                first, values[0], first_chart
            ) - _curve_value_on_chart(second, values[1], second_chart)

        from ._patches import _linear_pcurve_velocity

        affine_sources = (
            _linear_pcurve_velocity(first) is not None
            and _linear_pcurve_velocity(second) is not None
        )
        system = _RootEquation(
            residual,
            2,
            source_value=(
                _LinearTrimSourceBounds(first, second, False) if affine_sources else None
            ),
            source_jacobian=(
                _LinearTrimSourceBounds(first, second, True) if affine_sources else None
            ),
        )
        params, contracted_lower, contracted_upper, preconditioner, contraction = (
            _root_witness(system, lower_input, upper_input)
        )
        self.first, self.second = first, second
        self.parameter_lower, self.parameter_upper = (
            jnp.asarray(lower_input),
            jnp.asarray(upper_input),
        )
        self.parameters, self.preconditioner = (
            jnp.asarray(params),
            jnp.asarray(preconditioner),
        )
        self.enclosure_lower, self.enclosure_upper = (
            jnp.asarray(contracted_lower),
            jnp.asarray(contracted_upper),
        )
        self.endpoint_witness = endpoint_witness
        self.contraction = contraction
        self.root_id = canonical_fingerprint(
            {
                "kind": "trim-intersection-root",
                "first": _curve_payload(first),
                "second": _curve_payload(second),
                "lower": _float_hex(lower_input),
                "upper": _float_hex(upper_input),
                "endpoint_witness": (
                    None
                    if endpoint_witness is None
                    else endpoint_witness.identity_payload()
                ),
            }
        )

    def parameter_enclosure(self) -> np.ndarray:
        """Return the Krawczyk-contracted enclosure of the unique root.

        ``parameter_lower``/``parameter_upper`` remain the authored isolation
        box that proves uniqueness and names the root; the contracted box is a
        deterministic function of that definition.
        """
        return np.stack(
            (np.asarray(self.enclosure_lower), np.asarray(self.enclosure_upper))
        )


class CurveSurfaceIntersectionRoot(StrictModule):
    curve: AbstractCurve | IntersectionCurve
    surface: SurfaceRegion
    parameter_lower: Array
    parameter_upper: Array
    parameters: Array
    preconditioner: Array
    endpoint_witness: ConicRootLineEndpointWitness | None
    contraction: float = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: AbstractCurve | IntersectionCurve,
        surface: SurfaceRegion,
        /,
        *,
        parameter_lower: ArrayLike,
        parameter_upper: ArrayLike,
        endpoint_witness: ConicRootLineEndpointWitness | None = None,
    ) -> None:
        if curve.ambient_dimension != 3:
            raise ValueError(
                "CurveSurfaceIntersectionRoot.curve must be three-dimensional."
            )

        lower_input = np.asarray(parameter_lower, dtype=np.float64)
        upper_input = np.asarray(parameter_upper, dtype=np.float64)
        if endpoint_witness is None:

            def residual(values: Array) -> Array:
                return curve.evaluate(values[0]) - surface.patch.evaluate(values[1:])

            system = _RootEquation(residual, 3)
            params, _, _, preconditioner, contraction = _root_witness(
                system, lower_input, upper_input
            )
        else:
            if not isinstance(curve, AbstractCurve):
                raise TypeError(
                    "A constructive endpoint requires a native spatial curve."
                )
            endpoint_witness.require_current()
            point_box = endpoint_witness.curve_surface_parameter_box(curve, surface)
            if (
                lower_input.shape != (3,)
                or upper_input.shape != (3,)
                or not np.all(np.isfinite(lower_input))
                or not np.all(np.isfinite(upper_input))
                or np.any(lower_input >= upper_input)
                or np.any(point_box[0] < lower_input)
                or np.any(point_box[1] > upper_input)
            ):
                raise ValueError(
                    "A spatial endpoint root needs a finite isolation box containing its source witness."
                )
            system = _RootEquation(
                eqx.Partial(_curve_surface_residual, curve, surface.patch),
                3,
                source_value=_CurveSurfaceSourceBounds(curve, surface.patch, False),
                source_jacobian=_CurveSurfaceSourceBounds(curve, surface.patch, True),
            )
            result = krawczyk(system, lower_input[None], upper_input[None])
            contraction = float(result.contraction[0])
            # Existence comes from the source construction, including a root on
            # a chart boundary. A strict contraction proves uniqueness there.
            if (
                bool(result.excluded[0])
                or not np.isfinite(contraction)
                or not contraction < 1.0
                or not np.all(np.isfinite(result.preconditioner[0]))
            ):
                raise ValueError(
                    "The constructive endpoint lacks a source-faithful uniqueness proof."
                )
            params = 0.5 * point_box[0] + 0.5 * point_box[1]
            preconditioner = result.preconditioner[0]
        self.curve, self.surface = curve, surface
        self.parameter_lower, self.parameter_upper = (
            jnp.asarray(lower_input),
            jnp.asarray(upper_input),
        )
        self.parameters, self.preconditioner = (
            jnp.asarray(params),
            jnp.asarray(preconditioner),
        )
        self.contraction = contraction
        self.endpoint_witness = endpoint_witness
        self.root_id = canonical_fingerprint(
            {
                "kind": "curve-surface-root",
                "curve": _curve_payload(curve),
                "surface": surface.identity_payload(),
                "lower": _float_hex(lower_input),
                "upper": _float_hex(upper_input),
                "endpoint_witness": (
                    None
                    if endpoint_witness is None
                    else endpoint_witness.identity_payload()
                ),
            }
        )

    def parameter_enclosure(self) -> np.ndarray:
        if self.endpoint_witness is not None:
            return self.endpoint_witness.curve_surface_parameter_box(
                self.curve, self.surface
            )
        return np.stack(
            (np.asarray(self.parameter_lower), np.asarray(self.parameter_upper))
        )


class IntersectionCurvePointRoot(StrictModule):
    curve: IntersectionCurve
    parameter: float = eqx.field(static=True)
    parameters: Array
    parameter_lower: Array
    parameter_upper: Array
    root_id: str = eqx.field(static=True)
    point_id: str = eqx.field(static=True)

    @checked
    def __init__(self, curve: IntersectionCurve, parameter: float, /) -> None:
        if (
            not curve.fully_certified
            or not math.isfinite(parameter)
            or not 0.0 <= parameter <= curve.num_charts
        ):
            raise ValueError(
                "IntersectionCurvePointRoot requires a certified branch parameter in range."
            )
        self.curve, self.parameter = curve, float(parameter)
        self.parameters = self.parameter_lower = self.parameter_upper = jnp.asarray(
            [self.parameter]
        )
        phase = 0.0 if curve.closed and parameter == curve.num_charts else self.parameter
        self.root_id = canonical_fingerprint(
            {
                "kind": "intersection-curve-point-root",
                "branch": curve.branch_id,
                "parameter": self.parameter.hex(),
            }
        )
        self.point_id = canonical_fingerprint(
            {
                "kind": "intersection-curve-point",
                "branch": curve.branch_id,
                "phase": phase.hex(),
            }
        )

    def point_enclosure(self) -> np.ndarray:
        return np.asarray(self.curve.evaluate_coupled(jnp.asarray(self.parameter)).point)


class TripleSurfaceIntersectionRoot(StrictModule):
    first: SurfaceRegion
    second: SurfaceRegion
    third: SurfaceRegion
    parameter_lower: Array
    parameter_upper: Array
    parameters: Array
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        first: SurfaceRegion,
        second: SurfaceRegion,
        third: SurfaceRegion,
        /,
        *,
        parameter_lower: ArrayLike,
        parameter_upper: ArrayLike,
    ) -> None:
        def residual(values: Array) -> Array:
            a = first.patch.evaluate(values[:2])
            return jnp.concatenate(
                (
                    a - second.patch.evaluate(values[2:4]),
                    a - third.patch.evaluate(values[4:6]),
                )
            )

        system = _RootEquation(residual, 6)

        lower_input = np.asarray(parameter_lower, dtype=np.float64)
        upper_input = np.asarray(parameter_upper, dtype=np.float64)
        params, _, _, _, _ = _root_witness(system, lower_input, upper_input)
        self.first, self.second, self.third = first, second, third
        self.parameter_lower, self.parameter_upper = (
            jnp.asarray(lower_input),
            jnp.asarray(upper_input),
        )
        self.parameters = jnp.asarray(params)
        self.root_id = canonical_fingerprint(
            {
                "kind": "triple-surface-root",
                "sources": (
                    first.identity_payload(),
                    second.identity_payload(),
                    third.identity_payload(),
                ),
                "lower": _float_hex(lower_input),
                "upper": _float_hex(upper_input),
            }
        )

    def certifies_root(
        self,
        root: TrimIntersectionRoot
        | CurveSurfaceIntersectionRoot
        | TripleSurfaceIntersectionRoot,
        /,
        *,
        source_edge_lifts: Sequence[_SourceEdgeLift] = (),
    ) -> bool:
        """Prove that another isolated source event is this same joint root."""
        if root is self:
            return True
        if not isinstance(
            root,
            (
                TrimIntersectionRoot,
                CurveSurfaceIntersectionRoot,
                TripleSurfaceIntersectionRoot,
            ),
        ):
            return False
        image = _joint_root_parameter_image(
            root,
            (self.first, self.second, self.third),
            source_edge_lifts=source_edge_lifts,
        )
        if image is None:
            return False
        lower = np.asarray(self.parameter_lower, dtype=np.float64)
        upper = np.asarray(self.parameter_upper, dtype=np.float64)
        return bool(np.all(image[0] >= lower) and np.all(image[1] <= upper))


def _same_parameter_source(first: object, second: object, /) -> bool:
    if type(first) is not type(second):
        return False
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    if first_tree != second_tree or len(first_leaves) != len(second_leaves):
        return False
    try:
        return all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(first_leaves, second_leaves, strict=True)
        )
    except (TypeError, ValueError):
        return False


def _float_fraction_interval(lower: Fraction, upper: Fraction, /) -> tuple[float, float]:
    low = float(lower)
    high = float(upper)
    if Fraction.from_float(low) > lower:
        low = math.nextafter(low, -math.inf)
    if Fraction.from_float(high) < upper:
        high = math.nextafter(high, math.inf)
    return low, high


def _affine_parameter_interval(
    lower: float,
    upper: float,
    scale: Fraction,
    offset: Fraction,
    /,
) -> tuple[float, float]:
    first = scale * Fraction.from_float(lower) + offset
    last = scale * Fraction.from_float(upper) + offset
    return _float_fraction_interval(min(first, last), max(first, last))


def _curve_parameter_box(
    curve: AbstractCurve,
    first: float,
    last: float,
    /,
) -> np.ndarray | None:
    if not (math.isfinite(first) and math.isfinite(last) and first <= last):
        return None
    method = getattr(curve, "bounding_box", None)
    if callable(method):
        try:
            box = np.asarray(method(first, last), dtype=np.float64)
        except (NotImplementedError, TypeError, ValueError, RuntimeError):
            return None
    else:
        try:
            interval = prepare_interval_function(curve.evaluate, 1)
            lower, upper = interval.evaluate(
                np.asarray([[first]], dtype=np.float64),
                np.asarray([[last]], dtype=np.float64),
            )
            box = np.stack((np.asarray(lower)[0], np.asarray(upper)[0]))
        except (NotImplementedError, TypeError, ValueError, RuntimeError):
            return None
    if (
        box.ndim != 2
        or box.shape[0] != 2
        or not np.all(np.isfinite(box))
        or np.any(box[0] > box[1])
    ):
        return None
    return box


def _surface_slot(
    patch: AbstractSurfacePatch,
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    /,
) -> int | None:
    matches = tuple(
        index
        for index, region in enumerate(regions)
        if _same_parameter_source(patch, region.patch)
    )
    return matches[0] if len(matches) == 1 else None


def _add_surface_image(
    images: list[np.ndarray | None],
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    /,
) -> bool:
    slot = _surface_slot(patch, regions)
    if slot is None or box.shape != (2, 2):
        return False
    existing = images[slot]
    if existing is None:
        images[slot] = box
        return True
    lower = np.maximum(existing[0], box[0])
    upper = np.minimum(existing[1], box[1])
    if np.any(lower > upper):
        return False
    images[slot] = np.stack((lower, upper))
    return True


def _branch_surface_images(
    curve: IntersectionCurve,
    first: float,
    last: float,
    images: list[np.ndarray | None],
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    /,
) -> bool:
    if not curve.fully_certified:
        return False
    for side, region in (("first", curve.first), ("second", curve.second)):
        pcurve = curve.p_curve(side)
        box = _curve_parameter_box(pcurve, first, last)
        if box is None or not _add_surface_image(images, region.patch, box, regions):
            return False
    return True


def _curve_surface_root_images(
    root: CurveSurfaceIntersectionRoot,
    images: list[np.ndarray | None],
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    source_edge_lifts: Sequence[_SourceEdgeLift],
    /,
) -> bool:
    enclosure = root.parameter_enclosure()
    lower, upper = enclosure[0], enclosure[1]
    if lower.shape != (3,) or upper.shape != (3,):
        return False
    if isinstance(root.curve, IntersectionCurve):
        if not _branch_surface_images(
            root.curve, float(lower[0]), float(upper[0]), images, regions
        ):
            return False
    else:
        edge_first, edge_last = float(lower[0]), float(upper[0])
        lifted = False
        for lift in source_edge_lifts:
            if (
                not _same_parameter_source(root.curve, lift.curve)
                or edge_first < lift.first
                or edge_last > lift.last
            ):
                continue
            box = _curve_parameter_box(lift.pcurve, edge_first, edge_last)
            if box is None or not _add_surface_image(images, lift.surface, box, regions):
                return False
            lifted = True
        if not lifted:
            from ._correspondence import native_edge_pcurve

            for region in regions:
                pcurve = native_edge_pcurve(root.curve, region.patch)
                if pcurve is None:
                    continue
                box = _curve_parameter_box(pcurve, edge_first, edge_last)
                if box is None or not _add_surface_image(
                    images, region.patch, box, regions
                ):
                    return False
    surface_box = np.stack((lower[1:], upper[1:]))
    if not _add_surface_image(images, root.surface.patch, surface_box, regions):
        return False
    return True


def _trim_operand_surface_images(
    curve: AbstractCurve,
    first: float,
    last: float,
    images: list[np.ndarray | None],
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    source_edge_lifts: Sequence[_SourceEdgeLift],
    /,
) -> bool:
    if isinstance(curve, CurveTrimSegment):
        scale = Fraction.from_float(curve.last) - Fraction.from_float(curve.first)
        offset = Fraction.from_float(curve.first)
        if curve.reversed:
            scale = -scale
            offset = Fraction.from_float(curve.last)
        first_, last_ = _affine_parameter_interval(first, last, scale, offset)
        return _trim_operand_surface_images(
            curve.curve, first_, last_, images, regions, source_edge_lifts
        )
    if isinstance(curve, IntersectionPCurve):
        if curve.reversed:
            scale = Fraction(-1)
            offset = Fraction.from_float(curve.first) + Fraction.from_float(curve.last)
            first, last = _affine_parameter_interval(first, last, scale, offset)
        if first < curve.first or last > curve.last:
            return False
        if not _branch_surface_images(curve.curve, first, last, images, regions):
            return False
        return True
    matching_lifts = tuple(
        lift
        for lift in source_edge_lifts
        if _same_parameter_source(curve, lift.pcurve)
        and lift.first <= first <= last <= lift.last
    )
    if not matching_lifts:
        return False
    found = False
    for source_lift in matching_lifts:
        for lift in source_edge_lifts:
            if (
                not _same_parameter_source(source_lift.curve, lift.curve)
                or first < lift.first
                or last > lift.last
                or _surface_slot(lift.surface, regions) is None
            ):
                continue
            box = _curve_parameter_box(lift.pcurve, first, last)
            if box is None or not _add_surface_image(images, lift.surface, box, regions):
                return False
            found = True
    return found


def _joint_root_parameter_image(
    root: TrimIntersectionRoot
    | CurveSurfaceIntersectionRoot
    | TripleSurfaceIntersectionRoot,
    regions: tuple[SurfaceRegion, SurfaceRegion, SurfaceRegion],
    /,
    *,
    source_edge_lifts: Sequence[_SourceEdgeLift] = (),
) -> np.ndarray | None:
    images: list[np.ndarray | None] = [None, None, None]
    match root:
        case TripleSurfaceIntersectionRoot():
            lower = np.asarray(root.parameter_lower, dtype=np.float64)
            upper = np.asarray(root.parameter_upper, dtype=np.float64)
            if lower.shape != (6,) or upper.shape != (6,):
                return None
            for index, region in enumerate((root.first, root.second, root.third)):
                if not _add_surface_image(
                    images,
                    region.patch,
                    np.stack(
                        (
                            lower[2 * index : 2 * index + 2],
                            upper[2 * index : 2 * index + 2],
                        )
                    ),
                    regions,
                ):
                    return None
        case CurveSurfaceIntersectionRoot():
            if not _curve_surface_root_images(root, images, regions, source_edge_lifts):
                return None
        case TrimIntersectionRoot():
            enclosure = root.parameter_enclosure()
            lower, upper = enclosure[0], enclosure[1]
            if lower.shape != (2,) or upper.shape != (2,):
                return None
            if not _trim_operand_surface_images(
                root.first,
                float(lower[0]),
                float(upper[0]),
                images,
                regions,
                source_edge_lifts,
            ):
                return None
            second_lower, second_upper = (
                (float(root.endpoint_witness.parameter_exact),) * 2
                if root.endpoint_witness is not None
                else (float(lower[1]), float(upper[1]))
            )
            if not _trim_operand_surface_images(
                root.second,
                second_lower,
                second_upper,
                images,
                regions,
                source_edge_lifts,
            ):
                return None
        case _:
            return None
    if any(image is None for image in images):
        return None
    concrete = tuple(image for image in images if image is not None)
    return np.stack(
        (
            np.concatenate(tuple(image[0] for image in concrete)),
            np.concatenate(tuple(image[1] for image in concrete)),
        )
    )


def _float_hex(values: ArrayLike, /) -> tuple[str, ...]:
    return tuple(
        float(value).hex() for value in np.asarray(values, dtype=np.float64).reshape(-1)
    )


def _curve_payload(
    curve: AbstractCurve | AbstractSurfacePatch | IntersectionCurve, /
) -> dict[str, object]:
    identity = getattr(curve, "identity_payload", None)
    return (
        identity()
        if callable(identity)
        else {
            "type": f"{type(curve).__module__}.{type(curve).__qualname__}",
            "arrays": array_tree_fingerprint(curve),
        }
    )


def _retained_conic_endpoint_lift(
    lifts: tuple[_SourceEdgeLift, ...],
    curve: AbstractCurve,
    patch: AbstractSurfacePatch,
    source_pcurve: AbstractCurve,
    source_interval: tuple[Fraction, Fraction],
    /,
) -> _SourceEdgeLift:
    """Select a source-identical lift covering the original witness interval."""
    from ._boolean_core import AffinePCurve, PeriodicPCurve
    from ._intersection_curve import CurveTrimSegment

    for lift in lifts:
        if not (
            _same_parameter_source(lift.curve, curve)
            and _same_parameter_source(lift.surface, patch)
        ):
            continue
        carrier = lift.pcurve
        lower, upper = Fraction.from_float(lift.first), Fraction.from_float(lift.last)
        while True:
            if isinstance(carrier, CurveTrimSegment):
                first, last = (
                    Fraction.from_float(carrier.first),
                    Fraction.from_float(carrier.last),
                )
                lower, upper = sorted(
                    first + (1 - value if carrier.reversed else value) * (last - first)
                    for value in (lower, upper)
                )
                carrier = carrier.curve
            elif isinstance(carrier, AffinePCurve):
                carrier = carrier.curve
            elif isinstance(carrier, PeriodicPCurve):
                carrier = carrier.source_curve
            else:
                break
        if (
            _same_parameter_source(carrier, source_pcurve)
            and lower <= source_interval[0]
            and source_interval[1] <= upper
        ):
            return lift
    raise ValueError(
        "The endpoint has no retained source-identical lift covering its authored interval."
    )


class ConicRootLineEndpointWitness(StrictModule):
    """Source-bound proof for one endpoint of a rooted conic chart line."""

    source_curve: AbstractCurve
    source_pcurve: AbstractCurve
    source_region: SurfaceRegion
    parameter_exact: Fraction = eqx.field(static=True)
    source_interval: tuple[Fraction, Fraction] = eqx.field(static=True)
    trim_curve: AbstractCurve
    trim_interval: tuple[Fraction, Fraction] = eqx.field(static=True)
    line_parameter_lower: float = eqx.field(static=True)
    line_parameter_upper: float = eqx.field(static=True)
    source_uv_enclosure: Array
    trim_uv_enclosure: Array
    uv_enclosure: Array
    source_point_enclosure: Array
    surface_point_enclosure: Array
    point_enclosure: Array
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_curve: AbstractCurve,
        source_pcurve: AbstractCurve,
        source_region: SurfaceRegion,
        parameter_exact: Fraction,
        source_interval: tuple[Fraction, Fraction],
        trim_curve: AbstractCurve,
        trim_interval: tuple[Fraction, Fraction],
        /,
    ) -> None:
        from .._rational import fraction_down, fraction_up
        from ._correspondence import exact_source_lift
        from ._patches import (
            _validate_source_shadows,
            ConicRootLinePCurve,
            ExactLineCurve,
            LineCurve,
        )

        if source_curve.ambient_dimension != 3:
            raise ValueError("A conic endpoint source curve must be 3D.")
        if not isinstance(source_pcurve, ConicRootLinePCurve):
            raise TypeError("A conic endpoint witness requires a ConicRootLinePCurve.")
        if parameter_exact not in (Fraction(0), Fraction(1)):
            raise ValueError(
                "A conic endpoint witness parameter must be the exact endpoint 0 or 1."
            )

        def exact_interval(
            interval: tuple[Fraction, Fraction], name: str, /
        ) -> tuple[Fraction, Fraction]:
            if (
                not isinstance(interval, tuple)
                or len(interval) != 2
                or any(not isinstance(value, Fraction) for value in interval)
            ):
                raise TypeError(f"{name} must be an exact Fraction pair.")
            if interval[0] >= interval[1]:
                raise ValueError(f"{name} must be strictly increasing.")
            return interval

        source_interval_ = exact_interval(source_interval, "source_interval")
        trim_interval_ = exact_interval(trim_interval, "trim_interval")
        if parameter_exact not in source_interval_:
            raise ValueError(
                "The conic endpoint parameter must lie on the retained source interval."
            )

        def exact_domain(curve: AbstractCurve, name: str, /) -> tuple[Fraction, Fraction]:
            domain = curve.parameter_domain
            if domain is None:
                raise ValueError(f"{name} must declare its source parameter domain.")
            result = (
                Fraction.from_float(float(domain[0])),
                Fraction.from_float(float(domain[1])),
            )
            if result != source_interval_:
                raise ValueError(
                    f"{name} must retain the exact source interval identity."
                )
            return result

        exact_domain(source_curve, "source_curve")
        exact_domain(source_pcurve, "source_pcurve")
        if not isinstance(trim_curve, (LineCurve, ExactLineCurve)):
            raise TypeError(
                "A conic endpoint witness requires an exact affine trim line."
            )
        if trim_curve.ambient_dimension != 2:
            raise ValueError("The witnessed trim line must be planar.")
        trim_domain = trim_curve.parameter_domain
        if trim_domain is not None and (
            float(trim_interval_[0]) < trim_domain[0]
            or float(trim_interval_[1]) > trim_domain[1]
        ):
            raise ValueError(
                "The witnessed trim interval must lie in its source line domain."
            )
        _validate_source_shadows(source_curve, set())
        _validate_source_shadows(source_pcurve, set())
        _validate_source_shadows(source_region.patch, set())
        if not exact_source_lift(
            source_curve,
            source_pcurve,
            source_region,
            float(source_interval_[0]),
            float(source_interval_[1]),
        ):
            raise ValueError(
                "A conic endpoint witness requires an exact source-interval lift."
            )

        if isinstance(trim_curve, LineCurve):
            line_origin = trim_curve.origin_exact
            line_direction = trim_curve.direction_exact
        else:
            line_origin = trim_curve.first_exact
            line_direction = tuple(
                last - first
                for first, last in zip(
                    trim_curve.first_exact, trim_curve.last_exact, strict=True
                )
            )
        if (
            len(line_origin) != 2
            or len(line_direction) != 2
            or line_direction[0] == 0
            or line_direction[1] != 0
        ):
            raise ValueError(
                "The affine trim line must vary only in the conic angular coordinate."
            )
        height = (
            source_pcurve.height_origin_exact
            + source_pcurve.height_direction_exact * parameter_exact
        )
        if line_origin[1] != height:
            raise ValueError(
                "The trim line height must equal the exact rooted-line endpoint height."
            )

        shift_lower, shift_upper = coefficient_interval(
            source_pcurve.phase_shift_exact, bits=160
        )
        angular_lower = (
            Fraction.from_float(float(source_pcurve.root.parameter_lower)) + shift_lower
        )
        angular_upper = (
            Fraction.from_float(float(source_pcurve.root.parameter_upper)) + shift_upper
        )
        origin_lower, origin_upper = coefficient_interval(line_origin[0], bits=160)
        direction_lower, direction_upper = coefficient_interval(
            line_direction[0], bits=160
        )
        if direction_lower <= 0 <= direction_upper:
            raise ValueError("The trim line angular direction must be nonzero.")
        numerator = (
            angular_lower - origin_upper,
            angular_upper - origin_lower,
        )
        quotients = tuple(
            value / direction
            for value in numerator
            for direction in (direction_lower, direction_upper)
        )
        line_lower_exact, line_upper_exact = min(quotients), max(quotients)
        trim_first, trim_last = trim_interval_
        if line_lower_exact < trim_first or line_upper_exact > trim_last:
            raise ValueError(
                "The exact conic angular endpoint image lies outside the trim-line interval."
            )
        line_lower = fraction_down(line_lower_exact)
        line_upper = fraction_up(line_upper_exact)

        source_uv = np.asarray(
            source_pcurve.bounding_box(float(parameter_exact), float(parameter_exact)),
            dtype=np.float64,
        )
        trim_uv = np.asarray(
            trim_curve.bounding_box(float(line_lower), float(line_upper)),
            dtype=np.float64,
        )
        if (
            source_uv.shape != (2, 2)
            or trim_uv.shape != (2, 2)
            or not np.all(np.isfinite(source_uv))
            or not np.all(np.isfinite(trim_uv))
            or np.any(source_uv[0] > source_uv[1])
            or np.any(trim_uv[0] > trim_uv[1])
        ):
            raise ValueError(
                "The conic endpoint witness requires finite outward UV enclosures."
            )
        uv_lower = np.maximum(source_uv[0], trim_uv[0])
        uv_upper = np.minimum(source_uv[1], trim_uv[1])
        if np.any(uv_lower > uv_upper):
            raise ValueError(
                "The exact conic endpoint and affine trim line have disjoint UV enclosures."
            )
        region_box = np.asarray(source_region.parameter_box, dtype=np.float64)
        for axis in range(2):
            if uv_lower[axis] < uv_upper[axis]:
                continue
            value = uv_lower[axis]
            if value <= region_box[0, axis]:
                lower_axis = value
                upper_axis = np.nextafter(value, np.inf)
            elif value >= region_box[1, axis]:
                lower_axis = np.nextafter(value, -np.inf)
                upper_axis = value
            else:
                lower_axis = np.nextafter(value, -np.inf)
                upper_axis = np.nextafter(value, np.inf)
            uv_lower[axis] = max(lower_axis, region_box[0, axis])
            uv_upper[axis] = min(upper_axis, region_box[1, axis])
            if not uv_lower[axis] < uv_upper[axis]:
                raise ValueError(
                    "The conic endpoint has no positive-width source UV enclosure."
                )
        uv_box = np.stack((uv_lower, uv_upper))
        if np.any(uv_box[0] < region_box[0]) or np.any(uv_box[1] > region_box[1]):
            raise ValueError(
                "The conic endpoint UV enclosure lies outside its source region."
            )
        surface_point = np.asarray(
            source_region.patch.bounding_box(uv_box), dtype=np.float64
        )
        source_point = np.asarray(
            source_curve.bounding_box(float(parameter_exact), float(parameter_exact)),
            dtype=np.float64,
        )
        if (
            surface_point.shape != (2, 3)
            or source_point.shape != (2, 3)
            or not np.all(np.isfinite(surface_point))
            or not np.all(np.isfinite(source_point))
            or np.any(surface_point[0] > surface_point[1])
            or np.any(source_point[0] > source_point[1])
        ):
            raise ValueError(
                "The conic endpoint witness requires finite outward point enclosures."
            )
        point_lower = np.maximum(surface_point[0], source_point[0])
        point_upper = np.minimum(surface_point[1], source_point[1])
        if np.any(point_lower > point_upper):
            raise ValueError(
                "The source curve and source-region endpoint enclosures are disjoint."
            )

        self.source_curve = source_curve
        self.source_pcurve = source_pcurve
        self.source_region = source_region
        self.parameter_exact = parameter_exact
        self.source_interval = source_interval_
        self.trim_curve = trim_curve
        self.trim_interval = trim_interval_
        self.line_parameter_lower, self.line_parameter_upper = (
            float(line_lower),
            float(line_upper),
        )
        self.source_uv_enclosure = jnp.asarray(source_uv)
        self.trim_uv_enclosure = jnp.asarray(trim_uv)
        self.uv_enclosure = jnp.asarray(uv_box)
        self.source_point_enclosure = jnp.asarray(source_point)
        self.surface_point_enclosure = jnp.asarray(surface_point)
        self.point_enclosure = jnp.asarray(np.stack((point_lower, point_upper)))
        self.root_id = canonical_fingerprint(self.identity_payload())

    def require_current(self, /) -> None:
        rebuilt = ConicRootLineEndpointWitness(
            self.source_curve,
            self.source_pcurve,
            self.source_region,
            self.parameter_exact,
            self.source_interval,
            self.trim_curve,
            self.trim_interval,
        )
        if (
            self.root_id != rebuilt.root_id
            or self.identity_payload() != rebuilt.identity_payload()
        ):
            raise ValueError(
                "The conic endpoint witness no longer matches its source proof."
            )

    def curve_surface_parameter_box(
        self,
        curve: AbstractCurve | IntersectionCurve,
        surface: SurfaceRegion,
        /,
    ) -> np.ndarray:
        """Prove the cap curve meets the selected ruled rail at this source endpoint."""
        from ._correspondence import exact_source_lift
        from ._patches import RuledSurface

        if not isinstance(curve, AbstractCurve) or not isinstance(
            surface.patch, RuledSurface
        ):
            raise TypeError(
                "A constructive spatial endpoint requires a native curve and ruled surface."
            )
        rails = (surface.patch.first, surface.patch.second)
        sides = tuple(
            side
            for side, rail in enumerate(rails)
            if _same_parameter_source(rail, self.source_curve)
        )
        if len(sides) != 1:
            raise ValueError(
                "The spatial endpoint must identify one exact ruled source rail."
            )
        if not exact_source_lift(
            curve,
            self.trim_curve,
            self.source_region,
            float(self.trim_interval[0]),
            float(self.trim_interval[1]),
        ):
            raise ValueError(
                "The spatial curve does not retain the witnessed angular source lift."
            )
        uv = np.asarray((float(self.parameter_exact), float(sides[0])), dtype=np.float64)
        region = np.asarray(surface.parameter_box, dtype=np.float64)
        if np.any(uv < region[0]) or np.any(uv > region[1]):
            raise ValueError(
                "The constructive endpoint lies outside the ruled source region."
            )
        return np.asarray(
            (
                (self.line_parameter_lower, uv[0], uv[1]),
                (self.line_parameter_upper, uv[0], uv[1]),
            ),
            dtype=np.float64,
        )

    def parameter_box(self, /) -> np.ndarray:
        return np.asarray(
            (
                (self.line_parameter_lower, float(self.parameter_exact)),
                (self.line_parameter_upper, float(self.parameter_exact)),
            ),
            dtype=np.float64,
        )

    def binds_curves(self, first: AbstractCurve, second: AbstractCurve, /) -> bool:
        return _curve_payload(first) == _curve_payload(
            self.trim_curve
        ) and _curve_payload(second) == _curve_payload(self.source_pcurve)

    def identity_payload(self, /) -> dict[str, object]:
        return {
            "kind": "conic-root-line-endpoint-witness",
            "source_curve": _curve_payload(self.source_curve),
            "source_pcurve": _curve_payload(self.source_pcurve),
            "source_region": self.source_region.identity_payload(),
            "parameter_exact": (
                self.parameter_exact.numerator,
                self.parameter_exact.denominator,
            ),
            "source_interval": tuple(
                (value.numerator, value.denominator) for value in self.source_interval
            ),
            "trim_curve": _curve_payload(self.trim_curve),
            "trim_interval": tuple(
                (value.numerator, value.denominator) for value in self.trim_interval
            ),
            "line_parameter_enclosure": (
                float(self.line_parameter_lower).hex(),
                float(self.line_parameter_upper).hex(),
            ),
            "source_uv_enclosure": _float_hex(self.source_uv_enclosure),
            "trim_uv_enclosure": _float_hex(self.trim_uv_enclosure),
            "uv_enclosure": _float_hex(self.uv_enclosure),
            "source_point_enclosure": _float_hex(self.source_point_enclosure),
            "surface_point_enclosure": _float_hex(self.surface_point_enclosure),
            "point_enclosure": _float_hex(self.point_enclosure),
        }

    def _require_rail_face_binding(
        self,
        curve: AbstractCurve,
        pcurve: AbstractCurve,
        region: SurfaceRegion,
        parameter: Fraction,
        endpoint: TrimRootEndpoint,
        lifts: tuple[_SourceEdgeLift, ...],
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transport the exact rail endpoint through a proved second surface lift."""
        from ._correspondence import exact_source_lift
        from ._patches import _validate_source_shadows

        self.require_current()
        if (
            not isinstance(endpoint, TrimRootEndpoint)
            or not isinstance(endpoint.root, TrimIntersectionRoot)
            or endpoint.root.endpoint_witness is None
            or endpoint.root.endpoint_witness.identity_payload()
            != self.identity_payload()
            or not self.binds_curves(endpoint.root.first, endpoint.root.second)
            or endpoint.operand != "second"
            or endpoint.source_pcurve is None
            or endpoint.source_surface is None
            or not _same_parameter_source(endpoint.source_pcurve, pcurve)
            or not _same_parameter_source(endpoint.source_surface.patch, region.patch)
            or not _same_parameter_source(curve, self.source_curve)
        ):
            raise ValueError(
                "The cross-chart endpoint must retain the same exact spatial rail."
            )
        first, last = endpoint.parameter_enclosure()
        if (
            parameter != self.parameter_exact
            or Fraction.from_float(first) != self.parameter_exact
            or Fraction.from_float(last) != self.parameter_exact
        ):
            raise ValueError(
                "The cross-chart rail endpoint must retain its exact source parameter."
            )
        _retained_conic_endpoint_lift(
            lifts,
            curve,
            self.source_region.patch,
            self.source_pcurve,
            self.source_interval,
        )
        _validate_source_shadows(pcurve, set())
        _validate_source_shadows(region.patch, set())
        if not exact_source_lift(
            curve,
            pcurve,
            region,
            float(self.source_interval[0]),
            float(self.source_interval[1]),
        ):
            raise ValueError(
                "The current face lacks an exact full-range lift of the witnessed rail."
            )
        uv = np.asarray(pcurve.bounding_box(first, last), dtype=np.float64)
        domain = np.asarray(region.parameter_box, dtype=np.float64)
        if (
            uv.shape != (2, 2)
            or not np.all(np.isfinite(uv))
            or np.any(uv[0] > uv[1])
            or np.any(uv[0] < domain[0])
            or np.any(uv[1] > domain[1])
        ):
            raise ValueError(
                "The cross-chart endpoint lies outside the current face region."
            )
        boxes = (
            np.asarray(curve.bounding_box(first, last), dtype=np.float64),
            np.asarray(region.patch.bounding_box(uv), dtype=np.float64),
            np.asarray(self.point_enclosure, dtype=np.float64),
        )
        if any(
            box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] > box[1])
            for box in boxes
        ):
            raise ValueError(
                "The rail endpoint needs finite ordered source point enclosures."
            )
        lower = np.maximum.reduce(tuple(box[0] for box in boxes))
        upper = np.minimum.reduce(tuple(box[1] for box in boxes))
        if np.any(lower > upper):
            raise ValueError(
                "The exact rail endpoint has inconsistent source enclosures."
            )
        return uv, np.stack((lower, upper))

    def require_source_binding(
        self,
        actual_curve: AbstractCurve,
        actual_pcurve: AbstractCurve,
        actual_region: SurfaceRegion,
        actual_endpoint_parameter: Fraction,
        endpoint_binding: TrimRootEndpoint,
        /,
        *,
        source_lifts: tuple[_SourceEdgeLift, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate one endpoint against its authored and current source carriers.

        The retained source lift replays the witness interval; the model still
        checks current-segment incidence and face gaps independently.
        """
        from .._rational import fraction_down, fraction_up
        from ._boolean_core import AffinePCurve, PeriodicPCurve
        from ._intersection_curve import CurveTrimSegment
        from ._patches import (
            _validate_source_shadows,
            ConicRootLinePCurve,
        )
        from ._root_bindings import BRepCurveSurfaceLift

        if (
            not isinstance(actual_curve, AbstractCurve)
            or not isinstance(actual_pcurve, AbstractCurve)
            or not isinstance(actual_region, SurfaceRegion)
            or not isinstance(source_lifts, tuple)
            or not source_lifts
            or any(not isinstance(lift, BRepCurveSurfaceLift) for lift in source_lifts)
        ):
            raise TypeError(
                "A current source binding needs an edge, p-curve, region, "
                "and retained BRepCurveSurfaceLift tuple."
            )
        if not isinstance(actual_endpoint_parameter, Fraction):
            raise TypeError("The current endpoint parameter must be an exact Fraction.")
        if actual_curve.ambient_dimension != 3 or actual_pcurve.ambient_dimension != 2:
            raise ValueError(
                "A conic endpoint attachment requires a 3D edge and 2D p-curve."
            )
        if not _same_parameter_source(actual_region.patch, self.source_region.patch):
            return self._require_rail_face_binding(
                actual_curve,
                actual_pcurve,
                actual_region,
                actual_endpoint_parameter,
                endpoint_binding,
                source_lifts,
            )
        if (
            not isinstance(endpoint_binding, TrimRootEndpoint)
            or not isinstance(endpoint_binding.root, TrimIntersectionRoot)
            or endpoint_binding.root.endpoint_witness is None
            or endpoint_binding.root.endpoint_witness.root_id != self.root_id
            or not endpoint_binding.root.endpoint_witness.binds_curves(
                endpoint_binding.root.first, endpoint_binding.root.second
            )
            or endpoint_binding.source_pcurve is None
            or endpoint_binding.source_surface is None
            or not _same_parameter_source(
                endpoint_binding.source_surface.patch, self.source_region.patch
            )
        ):
            raise ValueError(
                "The current endpoint must retain this witness on its source face."
            )
        if not _same_parameter_source(actual_region.patch, self.source_region.patch):
            raise ValueError(
                "The current face does not retain the witnessed source surface."
            )
        if endpoint_binding.operand == "second":
            source_carrier = self.source_pcurve
            source_interval = self.source_interval
        else:
            source_carrier = self.trim_curve
            source_interval = self.trim_interval
        source_lift = _retained_conic_endpoint_lift(
            source_lifts,
            actual_curve,
            actual_region.patch,
            source_carrier,
            source_interval,
        )
        _validate_source_shadows(self.source_curve, set())
        _validate_source_shadows(self.source_pcurve, set())
        _validate_source_shadows(self.trim_curve, set())
        _validate_source_shadows(endpoint_binding.source_pcurve, set())
        _validate_source_shadows(source_lift.pcurve, set())
        _validate_source_shadows(actual_curve, set())
        _validate_source_shadows(actual_pcurve, set())
        _validate_source_shadows(actual_region.patch, set())
        if self.root_id != canonical_fingerprint(self.identity_payload()):
            raise ValueError(
                "The conic endpoint witness identity does not match its exact evidence."
            )
        endpoint_lower, endpoint_upper = endpoint_binding.parameter_enclosure()
        endpoint_witness = endpoint_binding.root.endpoint_witness
        if (
            endpoint_witness.identity_payload() != self.identity_payload()
            or endpoint_witness.root_id
            != canonical_fingerprint(endpoint_witness.identity_payload())
        ):
            raise ValueError(
                "The endpoint root no longer carries this exact source witness."
            )
        endpoint_lower_exact = Fraction.from_float(float(endpoint_lower))
        endpoint_upper_exact = Fraction.from_float(float(endpoint_upper))
        if not (
            endpoint_lower_exact <= actual_endpoint_parameter <= endpoint_upper_exact
        ):
            raise ValueError(
                "The current numeric endpoint lies outside its retained source-root enclosure."
            )

        parameter = actual_endpoint_parameter
        carrier = actual_pcurve
        chart_wrappers: list[AffinePCurve | PeriodicPCurve] = []
        scalar_maps: list[tuple[Fraction, Fraction, bool]] = []
        while True:
            if isinstance(carrier, CurveTrimSegment):
                if not Fraction(0) <= parameter <= Fraction(1):
                    raise ValueError(
                        "A current trimmed p-curve endpoint lies outside its unit interval."
                    )
                first = Fraction.from_float(float(carrier.first))
                last = Fraction.from_float(float(carrier.last))
                scalar_maps.append((first, last, carrier.reversed))
                local = Fraction(1) - parameter if carrier.reversed else parameter
                parameter = first + local * (last - first)
                carrier = carrier.curve
            elif isinstance(carrier, AffinePCurve):
                chart_wrappers.append(carrier)
                carrier = carrier.curve
            elif isinstance(carrier, PeriodicPCurve):
                chart_wrappers.append(carrier)
                carrier = carrier.source_curve
            else:
                break

        source_carrier = endpoint_binding.source_pcurve
        while isinstance(
            source_carrier, (CurveTrimSegment, AffinePCurve, PeriodicPCurve)
        ):
            source_carrier = (
                source_carrier.curve
                if isinstance(source_carrier, (CurveTrimSegment, AffinePCurve))
                else source_carrier.source_curve
            )
        if not _same_parameter_source(carrier, source_carrier):
            raise ValueError(
                "The current endpoint carrier is not an exact descendant of its source p-curve."
            )
        source_chart = endpoint_binding.source_pcurve
        while isinstance(source_chart, CurveTrimSegment):
            source_chart = source_chart.curve
        actual_chart = actual_pcurve
        while isinstance(actual_chart, CurveTrimSegment):
            actual_chart = actual_chart.curve
        from ._boolean_core import pcurve_affine_correspondence

        try:
            chart_transport = pcurve_affine_correspondence(source_chart, actual_chart)
        except (ArithmeticError, TypeError, ValueError, NotImplementedError):
            chart_transport = None
        if chart_transport is None:
            raise ValueError(
                "The current p-curve has no exact affine/periodic source chart transport."
            )
        lift_chart = source_lift.pcurve
        while isinstance(lift_chart, CurveTrimSegment):
            lift_chart = lift_chart.curve
        try:
            lift_transport = pcurve_affine_correspondence(lift_chart, actual_chart)
        except (ArithmeticError, TypeError, ValueError, NotImplementedError):
            lift_transport = None
        if lift_transport is None:
            raise ValueError(
                "The whole-range source lift has no exact current p-curve chart transport."
            )

        def base_parameter(value: Fraction, /) -> Fraction:
            for first, last, reversed_ in scalar_maps:
                local = Fraction(1) - value if reversed_ else value
                value = first + local * (last - first)
            return value

        base_lower, base_upper = sorted(
            (
                base_parameter(endpoint_lower_exact),
                base_parameter(endpoint_upper_exact),
            )
        )
        lift_first = Fraction.from_float(float(source_lift.first))
        lift_last = Fraction.from_float(float(source_lift.last))
        lift_scalar_maps: list[tuple[Fraction, Fraction, bool]] = []
        lift_carrier = source_lift.pcurve
        while True:
            if isinstance(lift_carrier, CurveTrimSegment):
                first = Fraction.from_float(float(lift_carrier.first))
                last = Fraction.from_float(float(lift_carrier.last))
                lift_scalar_maps.append((first, last, lift_carrier.reversed))
                lift_carrier = lift_carrier.curve
            elif isinstance(lift_carrier, AffinePCurve):
                lift_carrier = lift_carrier.curve
            elif isinstance(lift_carrier, PeriodicPCurve):
                lift_carrier = lift_carrier.source_curve
            else:
                break

        def lift_parameter(value: Fraction, /) -> Fraction:
            for first, last, reversed_ in reversed(lift_scalar_maps):
                local = (value - first) / (last - first)
                value = Fraction(1) - local if reversed_ else local
            return value

        lift_lower, lift_upper = sorted(
            (lift_parameter(base_lower), lift_parameter(base_upper))
        )
        if lift_lower < lift_first or lift_upper > lift_last:
            raise ValueError(
                "The retained source lift does not cover the witnessed root enclosure."
            )
        if endpoint_binding.operand == "second":
            if (
                not isinstance(carrier, ConicRootLinePCurve)
                or not _same_parameter_source(carrier, self.source_pcurve)
                or parameter != self.parameter_exact
                or base_lower != self.parameter_exact
                or base_upper != self.parameter_exact
                or not _same_parameter_source(actual_curve, self.source_curve)
            ):
                raise ValueError(
                    "The current source p-curve does not retain the exact rooted endpoint."
                )
            source_uv = np.asarray(self.source_uv_enclosure, dtype=np.float64)
        elif endpoint_binding.operand == "first":
            if (
                not _same_parameter_source(carrier, self.trim_curve)
                or base_lower > Fraction.from_float(self.line_parameter_lower)
                or base_upper < Fraction.from_float(self.line_parameter_upper)
            ):
                raise ValueError(
                    "The current affine trim line does not enclose the witnessed source endpoint."
                )
            source_uv = np.asarray(self.trim_uv_enclosure, dtype=np.float64)
        else:
            raise ValueError("A conic source endpoint must select a trim operand.")
        source_parameter_interval = (
            self.source_interval
            if endpoint_binding.operand == "second"
            else self.trim_interval
        )
        if not source_parameter_interval[0] <= parameter <= source_parameter_interval[1]:
            raise ValueError(
                "The current endpoint parameter lies outside the witnessed source interval."
            )

        bounds = tuple(
            (
                Fraction.from_float(float(source_uv[0, axis])),
                Fraction.from_float(float(source_uv[1, axis])),
            )
            for axis in range(2)
        )
        for wrapper in reversed(chart_wrappers):
            if isinstance(wrapper, AffinePCurve):
                mapped: list[tuple[Fraction, Fraction]] = []
                for row in range(2):
                    lower, upper = wrapper.offset[row], wrapper.offset[row]
                    for column in range(2):
                        coefficient = wrapper.matrix[row][column]
                        first, last = bounds[column]
                        products = (coefficient * first, coefficient * last)
                        lower += min(products)
                        upper += max(products)
                    mapped.append((lower, upper))
                bounds = (mapped[0], mapped[1])
            else:
                if not _same_parameter_source(wrapper.patch, actual_region.patch):
                    raise ValueError(
                        "A native-period p-curve gauge must retain its declared source surface."
                    )
                shifted = list(bounds)
                for axis, turns in enumerate(wrapper.period_shifts):
                    if not turns:
                        continue
                    period = period_scalar_bounds(
                        PeriodOffset(Fraction(0), Fraction(turns))
                    )
                    shifted[axis] = (
                        shifted[axis][0] + Fraction.from_float(float(period[0])),
                        shifted[axis][1] + Fraction.from_float(float(period[1])),
                    )
                bounds = (shifted[0], shifted[1])
        mapped_uv = np.asarray(
            (
                tuple(fraction_down(lower) for lower, _ in bounds),
                tuple(fraction_up(upper) for _, upper in bounds),
            ),
            dtype=np.float64,
        )
        actual_uv = np.asarray(
            actual_pcurve.bounding_box(endpoint_lower, endpoint_upper),
            dtype=np.float64,
        )
        if (
            mapped_uv.shape != (2, 2)
            or actual_uv.shape != (2, 2)
            or not np.all(np.isfinite(mapped_uv))
            or not np.all(np.isfinite(actual_uv))
        ):
            raise ValueError(
                "The current endpoint has no finite outward p-curve enclosure."
            )
        uv_lower = np.maximum(mapped_uv[0], actual_uv[0])
        uv_upper = np.minimum(mapped_uv[1], actual_uv[1])
        if np.any(uv_lower > uv_upper):
            raise ValueError(
                "The exact source endpoint and current p-curve enclosures are disjoint."
            )
        region_box = np.asarray(actual_region.parameter_box, dtype=np.float64)
        for axis in range(2):
            if uv_lower[axis] < uv_upper[axis]:
                continue
            value = uv_lower[axis]
            if value <= region_box[0, axis]:
                lower_axis, upper_axis = value, np.nextafter(value, np.inf)
            elif value >= region_box[1, axis]:
                lower_axis, upper_axis = np.nextafter(value, -np.inf), value
            else:
                lower_axis = np.nextafter(value, -np.inf)
                upper_axis = np.nextafter(value, np.inf)
            uv_lower[axis] = max(lower_axis, region_box[0, axis])
            uv_upper[axis] = min(upper_axis, region_box[1, axis])
            if not uv_lower[axis] < uv_upper[axis]:
                raise ValueError(
                    "The current source region has no positive-width endpoint enclosure."
                )
        uv_box = np.stack((uv_lower, uv_upper))
        if np.any(uv_box[0] < region_box[0]) or np.any(uv_box[1] > region_box[1]):
            raise ValueError(
                "The witnessed endpoint enclosure lies outside the current face region."
            )
        surface_point = np.asarray(
            actual_region.patch.bounding_box(uv_box), dtype=np.float64
        )
        curve_point = np.asarray(
            actual_curve.bounding_box(endpoint_lower, endpoint_upper),
            dtype=np.float64,
        )
        source_point = np.asarray(self.point_enclosure, dtype=np.float64)
        if (
            surface_point.shape != (2, 3)
            or curve_point.shape != (2, 3)
            or source_point.shape != (2, 3)
            or not np.all(np.isfinite(surface_point))
            or not np.all(np.isfinite(curve_point))
            or not np.all(np.isfinite(source_point))
            or np.any(surface_point[0] > surface_point[1])
            or np.any(curve_point[0] > curve_point[1])
            or np.any(source_point[0] > source_point[1])
        ):
            raise ValueError(
                "The current endpoint does not have finite outward point enclosures."
            )
        point_lower = np.maximum(
            np.maximum(surface_point[0], curve_point[0]), source_point[0]
        )
        point_upper = np.minimum(
            np.minimum(surface_point[1], curve_point[1]), source_point[1]
        )
        if np.any(point_lower > point_upper):
            raise ValueError(
                "The current edge/face endpoint enclosures disagree with the witnessed source point."
            )
        return uv_box, np.stack((point_lower, point_upper))


class TrimRootEndpoint(StrictModule):
    root: TrimIntersectionRoot | CurveSurfaceIntersectionRoot | IntersectionCurvePointRoot
    operand: IntersectionSide = eqx.field(static=True)
    affine_parameter_offset: float = eqx.field(static=True)
    affine_parameter_scale: float = eqx.field(static=True)
    affine_transforms: tuple[tuple[float, float], ...] = eqx.field(static=True)
    source_pcurve: AbstractCurve | IntersectionPCurve | None
    source_surface: SurfaceRegion | None

    @checked
    def __init__(
        self,
        root: TrimIntersectionRoot
        | CurveSurfaceIntersectionRoot
        | IntersectionCurvePointRoot,
        operand: IntersectionSide,
        /,
        *,
        affine_parameter_offset: float = 0.0,
        affine_parameter_scale: float = 1.0,
        affine_transforms: tuple[tuple[float, float], ...] = (),
        source_pcurve: AbstractCurve | IntersectionPCurve | None = None,
        source_surface: SurfaceRegion | None = None,
    ) -> None:
        operand_ = parse(operand, IntersectionSide, "operand")
        if (
            not math.isfinite(affine_parameter_offset)
            or not math.isfinite(affine_parameter_scale)
            or affine_parameter_scale == 0.0
        ):
            raise ValueError(
                "Trim root endpoint operand and affine transport must be valid."
            )
        if (source_pcurve is None) != (source_surface is None):
            raise ValueError(
                "A p-curve endpoint lift must specify both p-curve and supporting surface."
            )
        self.root, self.operand = root, operand_
        self.affine_parameter_offset, self.affine_parameter_scale = (
            float(affine_parameter_offset),
            float(affine_parameter_scale),
        )
        self.affine_transforms = tuple(
            (float(scale), float(offset)) for scale, offset in affine_transforms
        )
        self.source_pcurve, self.source_surface = source_pcurve, source_surface

    def parameter_enclosure(self) -> tuple[float, float]:
        if (
            isinstance(self.root, TrimIntersectionRoot)
            and self.root.endpoint_witness is not None
            and self.operand == "second"
            and _same_parameter_source(
                self.root.second, self.root.endpoint_witness.source_pcurve
            )
        ):
            lower = upper = float(self.root.endpoint_witness.parameter_exact)
        elif isinstance(self.root, IntersectionCurvePointRoot):
            lower = upper = self.root.parameter
        elif isinstance(self.root, TrimIntersectionRoot):
            column = 0 if self.operand == "first" else 1
            enclosure = self.root.parameter_enclosure()
            lower = float(enclosure[0, column])
            upper = float(enclosure[1, column])
        elif isinstance(self.root, CurveSurfaceIntersectionRoot):
            if self.operand != "first":
                raise ValueError(
                    "A curve-surface root has no scalar surface-operand parameter."
                )
            enclosure = self.root.parameter_enclosure()
            lower = float(enclosure[0, 0])
            upper = float(enclosure[1, 0])
        else:
            raise TypeError(
                "TrimRootEndpoint.root must define a scalar source parameter."
            )
        lower, upper = _affine_parameter_interval(
            lower,
            upper,
            Fraction.from_float(self.affine_parameter_scale),
            Fraction.from_float(self.affine_parameter_offset),
        )
        for scale, offset in self.affine_transforms:
            lower, upper = _affine_parameter_interval(
                lower,
                upper,
                Fraction.from_float(scale),
                Fraction.from_float(offset),
            )
        return lower, upper


class BranchRootEndpoint(StrictModule):
    root: CurveSurfaceIntersectionRoot | TripleSurfaceIntersectionRoot
    curve: IntersectionCurve
    chart: int = eqx.field(static=True)
    source_pcurve: AbstractCurve | None
    source_side: IntersectionSide | None = eqx.field(static=True)
    source_first: float | None = eqx.field(static=True)
    source_last: float | None = eqx.field(static=True)
    periodic_shifts: tuple[int, int, int, int] = eqx.field(static=True)
    affine_transforms: tuple[tuple[float, float], ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: CurveSurfaceIntersectionRoot | TripleSurfaceIntersectionRoot,
        curve: IntersectionCurve,
        chart: int,
        /,
        *,
        source_pcurve: AbstractCurve | None = None,
        source_side: IntersectionSide | None = None,
        source_first: float | None = None,
        source_last: float | None = None,
        periodic_shifts: tuple[int, int, int, int] = (0, 0, 0, 0),
        affine_transforms: tuple[tuple[float, float], ...] = (),
    ) -> None:
        source_side_ = (
            None
            if source_side is None
            else parse(source_side, IntersectionSide, "source_side")
        )
        if (
            not curve.fully_certified
            or type(chart) is not int
            or not 0 <= chart < curve.num_charts
        ):
            raise ValueError("Branch root endpoint must select a certified branch chart.")
        if len(periodic_shifts) != 4 or any(
            type(value) is not int for value in periodic_shifts
        ):
            raise ValueError(
                "Branch root periodic gauge must contain four integer shifts."
            )
        self.root, self.curve, self.chart = root, curve, chart
        self.source_pcurve, self.source_side = source_pcurve, source_side_
        self.source_first, self.source_last = source_first, source_last
        self.periodic_shifts = tuple(periodic_shifts)
        self.affine_transforms = tuple(
            (float(scale), float(offset)) for scale, offset in affine_transforms
        )

    def _chart_parameter_enclosure(self) -> tuple[float, float]:
        image = self.source_image()
        chart_lower = np.asarray(self.curve.chart_lower[self.chart], dtype=np.float64)
        chart_upper = np.asarray(self.curve.chart_upper[self.chart], dtype=np.float64)
        if np.all(image[0] >= chart_lower) and np.all(image[1] <= chart_upper):
            if (
                isinstance(self.root, CurveSurfaceIntersectionRoot)
                and isinstance(self.root.curve, IntersectionCurve)
                and self.root.curve.branch_id == self.curve.branch_id
            ):
                lower = float(self.root.parameter_lower[0])
                upper = float(self.root.parameter_upper[0])
            else:
                axis = int(self.curve.chart_axes[self.chart])
                start = Fraction.from_float(
                    float(self.curve.chart_start[self.chart, axis])
                )
                end = Fraction.from_float(float(self.curve.chart_end[self.chart, axis]))
                advance = end - start
                if advance == 0:
                    raise ValueError(
                        "A branch source chart must have a nonzero parameter advance."
                    )
                q_lower = Fraction.from_float(float(image[0, axis]))
                q_upper = Fraction.from_float(float(image[1, axis]))
                values = (
                    Fraction(self.chart) + (q_lower - start) / advance,
                    Fraction(self.chart) + (q_upper - start) / advance,
                )
                lower, upper = _float_fraction_interval(min(values), max(values))
            if self.chart <= lower <= upper <= self.chart + 1:
                return lower, upper

        # A node isolation box may extend beyond either adjacent graph chart.
        # Its independent full-box uniqueness proof still identifies that node.
        for node in (self.chart, self.chart + 1):
            point_root = IntersectionCurvePointRoot(self.curve, float(node))
            if self.certifies_point_root(point_root):
                return float(node), float(node)
        raise ValueError(
            "A branch source endpoint image lies outside its selected chart."
        )

    def parameter_enclosure(self) -> tuple[float, float]:
        lower, upper = self._chart_parameter_enclosure()
        for scale, offset in self.affine_transforms:
            lower, upper = _affine_parameter_interval(
                lower,
                upper,
                Fraction.from_float(scale),
                Fraction.from_float(offset),
            )
        return lower, upper

    def _source_pcurve_carrier(self) -> tuple[AbstractCurve | None, np.ndarray]:
        """Split an integer native-period gauge off the source-edge p-curve.

        The root box is enclosed on the ungauged carrier, where an iso-parameter
        coordinate stays exact; the turns join the endpoint's sheet gauge.
        """
        from ._boolean_core import PeriodicPCurve

        turns = np.zeros((4,), dtype=np.int64)
        pcurve = self.source_pcurve
        if self.source_side is None:
            return pcurve, turns
        offset = 0 if self.source_side == "first" else 2
        while isinstance(pcurve, PeriodicPCurve):
            turns[offset : offset + 2] += np.asarray(pcurve.period_shifts, dtype=np.int64)
            pcurve = pcurve.source_curve
        return pcurve, turns

    def source_gauges(self) -> np.ndarray:
        """Return only explicit integer shifts on natively periodic chart axes."""
        shifts = np.asarray(self.periodic_shifts, dtype=np.int64)
        if isinstance(self.root, CurveSurfaceIntersectionRoot) and not isinstance(
            self.root.curve, IntersectionCurve
        ):
            shifts = shifts + self._source_pcurve_carrier()[1]
        symbols = (
            *self.curve.first.patch.native_period_symbols(),
            *self.curve.second.patch.native_period_symbols(),
        )
        if any(shift and symbols[axis] != "two_pi" for axis, shift in enumerate(shifts)):
            raise ValueError(
                "A branch source gauge requires a declared native two_pi axis."
            )
        return shifts.copy()

    def source_image(self, /, *, include_periodic_shifts: bool = True) -> np.ndarray:
        """Enclose the branch's two source charts from the endpoint root definition."""
        images: list[np.ndarray | None] = [None, None]
        if isinstance(self.root, TripleSurfaceIntersectionRoot):
            lower = np.asarray(self.root.parameter_lower, dtype=np.float64)
            upper = np.asarray(self.root.parameter_upper, dtype=np.float64)
            if lower.shape != (6,) or upper.shape != (6,):
                raise ValueError(
                    "A triple source endpoint needs a six-parameter isolation box."
                )
            source_regions = (self.root.first, self.root.second, self.root.third)
            for region_index, region in enumerate(source_regions):
                for target_index, target in enumerate(
                    (self.curve.first, self.curve.second)
                ):
                    if not _same_parameter_source(region.patch, target.patch):
                        continue
                    if images[target_index] is not None:
                        raise ValueError(
                            "A branch source endpoint has ambiguous patch identity."
                        )
                    images[target_index] = np.stack(
                        (
                            lower[2 * region_index : 2 * region_index + 2],
                            upper[2 * region_index : 2 * region_index + 2],
                        )
                    )
        elif isinstance(self.root, CurveSurfaceIntersectionRoot):
            enclosure = self.root.parameter_enclosure()
            lower, upper = enclosure[0], enclosure[1]
            if lower.shape != (3,) or upper.shape != (3,):
                raise ValueError(
                    "A curve-surface endpoint needs a three-parameter isolation box."
                )
            if isinstance(self.root.curve, IntersectionCurve):
                if not self.root.curve.fully_certified:
                    raise ValueError(
                        "A branch endpoint source curve must be fully certified."
                    )
                for target_index, target in enumerate(
                    (self.curve.first, self.curve.second)
                ):
                    side = next(
                        (
                            candidate
                            for candidate, region in (
                                ("first", self.root.curve.first),
                                ("second", self.root.curve.second),
                            )
                            if _same_parameter_source(region.patch, target.patch)
                        ),
                        None,
                    )
                    if side is None:
                        continue
                    box = _curve_parameter_box(
                        self.root.curve.p_curve(side), float(lower[0]), float(upper[0])
                    )
                    if box is None:
                        raise ValueError("A branch endpoint p-curve image is unresolved.")
                    images[target_index] = box
            else:
                if (
                    self.source_pcurve is None
                    or self.source_side is None
                    or self.source_first is None
                    or self.source_last is None
                ):
                    raise ValueError(
                        "A source-edge branch endpoint requires its exact coedge p-curve lift."
                    )
                source_index = 0 if self.source_side == "first" else 1
                opposite_index = 1 - source_index
                source_region = (self.curve.first, self.curve.second)[source_index]
                opposite_region = (self.curve.first, self.curve.second)[opposite_index]
                first, last = sorted((self.source_first, self.source_last))
                if float(lower[0]) < first or float(upper[0]) > last:
                    raise ValueError(
                        "A source-edge root must lie inside its lifted coedge range."
                    )
                from ._correspondence import exact_source_lift

                if not exact_source_lift(
                    self.root.curve,
                    self.source_pcurve,
                    source_region,
                    first,
                    last,
                ):
                    raise ValueError(
                        "A source-edge branch endpoint p-curve lift is not exact."
                    )
                if not _same_parameter_source(
                    self.root.surface.patch, opposite_region.patch
                ):
                    raise ValueError(
                        "A source-edge branch endpoint must bind the opposite branch face."
                    )
                carrier, _ = self._source_pcurve_carrier()
                if carrier is None:
                    raise ValueError(
                        "A source-edge branch endpoint requires its exact coedge p-curve lift."
                    )
                source_box = _curve_parameter_box(
                    carrier, float(lower[0]), float(upper[0])
                )
                if source_box is None:
                    raise ValueError(
                        "A source-edge branch endpoint p-curve image is unresolved."
                    )
                images[source_index] = source_box
                images[opposite_index] = np.stack((lower[1:], upper[1:]))
        else:
            raise TypeError("BranchRootEndpoint.root must be a source intersection root.")
        if any(image is None for image in images):
            raise ValueError("A branch source endpoint must bind both source charts.")
        concrete = tuple(image for image in images if image is not None)
        image = np.concatenate(concrete, axis=1)
        if include_periodic_shifts:
            from .._interval_enclosure import interval_add

            for axis, shift in enumerate(self.source_gauges()):
                if not shift:
                    continue
                period_lower, period_upper = period_scalar_bounds(
                    PeriodOffset(Fraction(0), Fraction(int(shift)))
                )
                lower, upper = interval_add(
                    (image[0, axis], image[1, axis]),
                    (period_lower, period_upper),
                )
                image[0, axis], image[1, axis] = lower, upper
        return image

    def certifies_point_root(self, point_root: IntersectionCurvePointRoot, /) -> bool:
        """Prove a source root is the isolated branch node using its fixed chart axis."""
        if (
            not isinstance(point_root, IntersectionCurvePointRoot)
            or point_root.curve.branch_id != self.curve.branch_id
            or not point_root.parameter.is_integer()
        ):
            return False
        point_node = int(point_root.parameter)
        image = self.source_image(include_periodic_shifts=False)
        for node in (self.chart, self.chart + 1):
            same_node = point_node == node or (
                self.curve.closed and {point_node, node} == {0, self.curve.num_charts}
            )
            if not same_node:
                continue
            shift_axis = 1 if node == self.chart else 0
            node_shifts = np.asarray(
                self.curve.node_period_shifts[node, shift_axis], dtype=np.int64
            )
            difference = self.source_gauges() - node_shifts
            frame = node
            if self.curve.closed and node == self.curve.num_charts:
                # The closing node is the start root seen through the closure
                # deck; compare in the start frame, whose fixed value is exact.
                closure = self.curve.closure_period_shifts()
                if np.any(closure):
                    difference = difference - closure
                    frame = 0
            transported = image
            if np.any(difference):
                lower, upper = self.curve.shift_parameter_bounds(difference)
                from .._interval_enclosure import interval_add

                transported = np.stack(interval_add((image[0], image[1]), (lower, upper)))
            fixed_axis = int(self.curve.node_axes[frame])
            fixed_value = float(self.curve.node_values[frame])
            node_lower = np.asarray(self.curve.node_lower[frame], dtype=np.float64)
            node_upper = np.asarray(self.curve.node_upper[frame], dtype=np.float64)
            if (
                transported[0, fixed_axis] == fixed_value
                and transported[1, fixed_axis] == fixed_value
            ):
                if np.all(transported[0] >= node_lower) and np.all(
                    transported[1] <= node_upper
                ):
                    return True
                if self._isolation_contains_node(transported, node_lower, node_upper):
                    return True
                continue
            # A node proved to lie exactly on another source coordinate shares
            # the source line's root when it sits inside that root's isolation.
            if any(
                axis != fixed_axis
                and transported[0, axis] == value
                and transported[1, axis] == value
                for axis, value in self.curve.node_exact_coordinates(frame)
            ) and self._isolation_contains_node(transported, node_lower, node_upper):
                return True
        return False

    def _isolation_contains_node(
        self, image: np.ndarray, node_lower: np.ndarray, node_upper: np.ndarray, /
    ) -> bool:
        """Identify a node lying strictly inside a source-line root isolation box.

        For a source edge with an affine p-curve, the strict image interior is
        covered by the edge itself. A node there lies on the edge and on the
        opposite surface within the Krawczyk box, so it is that unique root.
        """
        from ._boolean_core import PeriodicPCurve
        from ._patches import LineCurve

        if (
            not isinstance(self.root, CurveSurfaceIntersectionRoot)
            or isinstance(self.root.curve, IntersectionCurve)
            or self.source_side is None
        ):
            return False
        pcurve = self.source_pcurve
        while isinstance(pcurve, PeriodicPCurve):
            pcurve = pcurve.source_curve
        if not isinstance(pcurve, LineCurve):
            return False
        source_axes = (0, 1) if self.source_side == "first" else (2, 3)
        # Only an iso-parameter line fills the interior of its own image box.
        if all(image[0, axis] < image[1, axis] for axis in source_axes):
            return False
        for axis in range(4):
            degenerate = image[0, axis] == image[1, axis]
            if degenerate and axis in source_axes:
                # A constant affine coordinate is exact along the whole edge.
                if not node_lower[axis] <= image[0, axis] <= node_upper[axis]:
                    return False
                continue
            if not (
                image[0, axis] < node_lower[axis] and node_upper[axis] < image[1, axis]
            ):
                return False
        return True


class NativePeriodEndpoint(StrictModule):
    curve: AbstractCurve | IntersectionCurve
    patch: AbstractSurfacePatch | None
    axis: int | None = eqx.field(static=True)
    rational: Fraction = eqx.field(static=True)
    turns: Fraction = eqx.field(static=True)

    @checked
    def __init__(
        self,
        curve: AbstractCurve | IntersectionCurve,
        patch: AbstractSurfacePatch | None = None,
        axis: int | None = None,
        /,
        *,
        rational: Fraction | int = 0,
        turns: Fraction | int = 1,
    ) -> None:
        if (
            not isinstance(rational, (Fraction, int))
            or isinstance(rational, bool)
            or not isinstance(turns, (Fraction, int))
            or isinstance(turns, bool)
        ):
            raise TypeError(
                "Native period coefficients must be exact rationals or integers."
            )
        if patch is None:
            if axis is not None:
                raise ValueError("A period axis requires its supporting patch.")
            period_symbol = curve.native_period_symbol()
            raw_period = curve.period
            domain = curve.parameter_domain
        else:
            if axis is None:
                raise ValueError("A surface period endpoint needs axis zero or one.")
            if type(axis) is not int or axis not in (0, 1):
                raise ValueError("A surface period endpoint needs axis zero or one.")
            period_symbol = patch.native_period_symbols()[axis]
            raw_period = patch.periods[axis]
            parameter_domain = patch.parameter_domain
            domain = (
                (parameter_domain[0][axis], parameter_domain[1][axis])
                if parameter_domain is not None
                else None
            )
        if period_symbol != "two_pi":
            raise ValueError(
                "NativePeriodEndpoint requires a source-declared mathematical 2pi period."
            )
        if raw_period is None:
            raise ValueError(
                "NativePeriodEndpoint source period must be finite and positive."
            )
        period = float(raw_period)
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError(
                "NativePeriodEndpoint source period must be finite and positive."
            )
        if domain is None:
            domain = (0.0, period)
        exact_rational, exact_turns = Fraction(rational), Fraction(turns)
        low, high = period_scalar_bounds(period_offset(exact_rational, exact_turns))
        # A declared one-turn seam may start at any exact phase when the
        # authored source domain spans one complete native period.
        exact_domain_span = Fraction.from_float(float(domain[1])) - Fraction.from_float(
            float(domain[0])
        )
        period_lower, period_upper = period_scalar_bounds(
            period_offset(Fraction(0), Fraction(1))
        )
        full_period_end = abs(exact_turns) == 1 and Fraction.from_float(
            float(period_lower)
        ) <= exact_domain_span <= Fraction.from_float(float(period_upper))
        if not full_period_end and (
            float(low) < float(domain[0]) or float(high) > float(domain[1])
        ):
            raise ValueError(
                "NativePeriodEndpoint exact parameter must lie in its source period domain."
            )
        self.curve, self.patch, self.axis = curve, patch, axis
        self.rational, self.turns = Fraction(rational), Fraction(turns)

    @property
    def exact_parameter(self) -> Fraction | PeriodOffset:
        return period_offset(self.rational, self.turns)

    @property
    def parameter(self) -> float:
        return float(self.rational) + float(self.turns) * math.tau

    def parameter_enclosure(self) -> tuple[float, float]:
        lower, upper = period_scalar_bounds(self.exact_parameter)
        return float(lower), float(upper)


def closes_native_turn(end: object, start: object, /) -> bool:
    """Prove a self-closing use returns to its start after one native turn.

    Both endpoints bind one curve-owned period (no surface chart axis) with the
    same rational phase and turn counts one apart; the carrier's ``2 pi``
    periodicity then maps the two parameters to the same point exactly.
    """
    from ._carrier_pieces import carrier_payload

    return (
        isinstance(end, NativePeriodEndpoint)
        and isinstance(start, NativePeriodEndpoint)
        and end.patch is None
        and start.patch is None
        and end.rational == start.rational
        and abs(end.turns - start.turns) == 1
        and carrier_payload(end.curve) == carrier_payload(start.curve)
    )


class ConicParameterEndpoint(StrictModule):
    """Exact conic-root endpoint on one unchanged edge or coedge carrier."""

    root: ConicParameterRoot
    carrier: AbstractCurve
    parameter_exact: Fraction | None = eqx.field(static=True)
    point_exact: tuple[ExactScalar, ExactScalar, ExactScalar] = eqx.field(static=True)
    root_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: ConicParameterRoot,
        carrier: AbstractCurve,
        parameter_exact: Fraction | None,
        point_exact: tuple[ExactScalar, ExactScalar, ExactScalar],
        /,
    ) -> None:
        if not isinstance(root, ConicParameterRoot) or root.conic.ambient_dimension != 3:
            raise TypeError(
                "ConicParameterEndpoint.root must define a 3D conic direction."
            )
        if not isinstance(carrier, AbstractCurve) or carrier.ambient_dimension not in (
            2,
            3,
        ):
            raise TypeError(
                "ConicParameterEndpoint.carrier must be a 2D p-curve or 3D edge carrier."
            )
        if parameter_exact is not None and not isinstance(parameter_exact, Fraction):
            raise TypeError(
                "ConicParameterEndpoint.parameter_exact must be a Fraction or None."
            )
        if len(point_exact) != 3 or any(
            not isinstance(value, (Fraction, RadicalElement)) for value in point_exact
        ):
            raise TypeError(
                "ConicParameterEndpoint.point_exact must contain three exact algebraic coordinates."
            )
        self.root, self.carrier = root, carrier
        self.parameter_exact, self.point_exact = parameter_exact, tuple(point_exact)
        self.root_id = canonical_fingerprint(self.identity_payload())

    def parameter_enclosure(self, /) -> tuple[float, float]:
        if self.parameter_exact is None:
            return self.root.parameter_enclosure()
        return _float_fraction_interval(self.parameter_exact, self.parameter_exact)

    def identity_payload(self, /) -> dict[str, object]:

        return {
            "kind": "conic-parameter-endpoint",
            "root": self.root.identity_payload(),
            "carrier": _curve_payload(self.carrier),
            "parameter_exact": None
            if self.parameter_exact is None
            else (self.parameter_exact.numerator, self.parameter_exact.denominator),
            "point_exact": tuple(
                coefficient_payload(value) for value in self.point_exact
            ),
        }


RootEndpoint: TypeAlias = (
    TrimRootEndpoint | BranchRootEndpoint | NativePeriodEndpoint | ConicParameterEndpoint
)


def _root_definition_payload(root: object, /) -> dict[str, object]:
    if isinstance(root, TrimIntersectionRoot):
        return {
            "kind": "trim-intersection-root",
            "first": _curve_payload(root.first),
            "second": _curve_payload(root.second),
            "parameter_lower": _float_hex(root.parameter_lower),
            "parameter_upper": _float_hex(root.parameter_upper),
            "endpoint_witness": (
                None
                if root.endpoint_witness is None
                else root.endpoint_witness.identity_payload()
            ),
        }
    if isinstance(root, CurveSurfaceIntersectionRoot):
        return {
            "kind": "curve-surface-root",
            "curve": _curve_payload(root.curve),
            "surface": root.surface.identity_payload(),
            "parameter_lower": _float_hex(root.parameter_lower),
            "parameter_upper": _float_hex(root.parameter_upper),
            "endpoint_witness": (
                None
                if root.endpoint_witness is None
                else root.endpoint_witness.identity_payload()
            ),
        }
    if isinstance(root, IntersectionCurvePointRoot):
        return {
            "kind": "intersection-curve-point-root",
            "branch_id": root.curve.branch_id,
            "parameter": float(root.parameter).hex(),
        }
    if isinstance(root, TripleSurfaceIntersectionRoot):
        return {
            "kind": "triple-surface-root",
            "surfaces": (
                root.first.identity_payload(),
                root.second.identity_payload(),
                root.third.identity_payload(),
            ),
            "parameter_lower": _float_hex(root.parameter_lower),
            "parameter_upper": _float_hex(root.parameter_upper),
        }
    raise TypeError(f"Unsupported source root record: {type(root)!r}.")


def endpoint_root_payload(root: RootEndpoint, /) -> dict[str, object]:
    if isinstance(root, ConicParameterEndpoint):
        return root.identity_payload()
    if isinstance(root, NativePeriodEndpoint):
        return {
            "kind": "native-period-endpoint",
            "curve": _curve_payload(root.curve),
            "patch": None if root.patch is None else _curve_payload(root.patch),
            "axis": root.axis,
            "rational": (root.rational.numerator, root.rational.denominator),
            "turns": (root.turns.numerator, root.turns.denominator),
        }
    if isinstance(root, TrimRootEndpoint):
        return {
            "kind": "trim-root-endpoint",
            "root": _root_definition_payload(root.root),
            "operand": root.operand,
            "affine_parameter_offset": float(root.affine_parameter_offset).hex(),
            "affine_parameter_scale": float(root.affine_parameter_scale).hex(),
            "affine_transforms": tuple(
                (float(a).hex(), float(b).hex()) for a, b in root.affine_transforms
            ),
            "source_pcurve": None
            if root.source_pcurve is None
            else _curve_payload(root.source_pcurve),
            "source_surface": None
            if root.source_surface is None
            else root.source_surface.identity_payload(),
        }
    return {
        "kind": "branch-root-endpoint",
        "root": _root_definition_payload(root.root),
        "branch_id": root.curve.branch_id,
        "chart": root.chart,
        "source_pcurve": None
        if root.source_pcurve is None
        else _curve_payload(root.source_pcurve),
        "source_side": root.source_side,
        "source_first": None
        if root.source_first is None
        else float(root.source_first).hex(),
        "source_last": None
        if root.source_last is None
        else float(root.source_last).hex(),
        "periodic_shifts": root.periodic_shifts,
        "affine_transforms": tuple(
            (float(a).hex(), float(b).hex()) for a, b in root.affine_transforms
        ),
    }


class _IntersectionClassification(StrictModule):
    pass


def _unresolved_surface(
    first: SurfaceRegion,
    second: SurfaceRegion,
    reason: UnresolvedIntersectionReason = "continuation",
) -> SurfaceIntersectionResult:
    bounds = np.concatenate(
        (np.asarray(first.parameter_box), np.asarray(second.parameter_box)), axis=1
    )
    return SurfaceIntersectionResult(
        unresolved=(UnresolvedParameterRegion(bounds[0], bounds[1], reason),)
    )


def _coincident_planes(
    first: SurfaceRegion, second: SurfaceRegion, /
) -> CoincidentParameterRegion | Literal[False] | None:
    a, b = first.patch, second.patch
    assert isinstance(a, PlanePatch) and isinstance(b, PlanePatch)
    a1, a2 = (
        rational_vector(np.asarray(a.first_axis)),
        rational_vector(np.asarray(a.second_axis)),
    )
    b1, b2 = (
        rational_vector(np.asarray(b.first_axis)),
        rational_vector(np.asarray(b.second_axis)),
    )
    normal_a = (
        a1[1] * a2[2] - a1[2] * a2[1],
        a1[2] * a2[0] - a1[0] * a2[2],
        a1[0] * a2[1] - a1[1] * a2[0],
    )
    normal_b = (
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    )
    if any(
        normal_a[i] * normal_b[j] != normal_a[j] * normal_b[i]
        for i in range(3)
        for j in range(3)
    ):
        return None
    origin_delta = rational_vector(np.asarray(a.origin) - np.asarray(b.origin))
    if rational_dot(normal_b, origin_delta) != 0:
        return None
    g00, g01, g11 = rational_dot(b1, b1), rational_dot(b1, b2), rational_dot(b2, b2)
    determinant = g00 * g11 - g01 * g01
    if determinant == 0:
        return None
    matrix = []
    for axis in (a1, a2):
        rhs0, rhs1 = rational_dot(b1, axis), rational_dot(b2, axis)
        matrix.append(
            (
                (g11 * rhs0 - g01 * rhs1) / determinant,
                (g00 * rhs1 - g01 * rhs0) / determinant,
            )
        )
    matrix = [(matrix[0][0], matrix[1][0]), (matrix[0][1], matrix[1][1])]
    offset_rhs0 = rational_dot(b1, origin_delta)
    offset_rhs1 = rational_dot(b2, origin_delta)
    offset = (
        (g11 * offset_rhs0 - g01 * offset_rhs1) / determinant,
        (g00 * offset_rhs1 - g01 * offset_rhs0) / determinant,
    )
    lower = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(first.parameter_box[0])
    )
    upper = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(first.parameter_box[1])
    )
    second_lower = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(second.parameter_box[0])
    )
    second_upper = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(second.parameter_box[1])
    )
    polygon = [
        (lower[0], lower[1]),
        (upper[0], lower[1]),
        (upper[0], upper[1]),
        (lower[0], upper[1]),
    ]

    for axis in range(2):
        row = matrix[axis]

        # Keep the affine source image inside the second trim rectangle.
        def signed_value(point: tuple[Fraction, Fraction]) -> Fraction:
            return row[0] * point[0] + row[1] * point[1] + offset[axis]

        for bound, above in ((second_lower[axis], True), (second_upper[axis], False)):
            if not polygon:
                break
            result: list[tuple[Fraction, Fraction]] = []
            previous = polygon[-1]
            previous_raw = signed_value(previous) - bound
            previous_value = previous_raw if above else -previous_raw
            for current in polygon:
                current_raw = signed_value(current) - bound
                current_value = current_raw if above else -current_raw
                previous_inside, current_inside = previous_value >= 0, current_value >= 0
                if previous_inside != current_inside:
                    fraction = previous_value / (previous_value - current_value)
                    result.append(
                        (
                            previous[0] + fraction * (current[0] - previous[0]),
                            previous[1] + fraction * (current[1] - previous[1]),
                        )
                    )
                if current_inside:
                    result.append(current)
                previous, previous_value = current, current_value
            polygon = result
    if not polygon:
        return False
    first_box = tuple(
        (min(point[axis] for point in polygon), max(point[axis] for point in polygon))
        for axis in range(2)
    )
    second_points = [
        tuple(
            matrix[axis][0] * point[0] + matrix[axis][1] * point[1] + offset[axis]
            for axis in range(2)
        )
        for point in polygon
    ]
    second_box = tuple(
        (
            min(point[axis] for point in second_points),
            max(point[axis] for point in second_points),
        )
        for axis in range(2)
    )
    corners = {
        (first_box[0][0], first_box[1][0]),
        (first_box[0][1], first_box[1][0]),
        (first_box[0][1], first_box[1][1]),
        (first_box[0][0], first_box[1][1]),
    }
    full_rectangle = len(polygon) == 4 and set(polygon) == corners
    cell_lower = np.asarray(
        [
            tuple(float(value[0]) for value in first_box)
            + tuple(float(value[0]) for value in second_box)
        ]
    )
    cell_upper = np.asarray(
        [
            tuple(float(value[1]) for value in first_box)
            + tuple(float(value[1]) for value in second_box)
        ]
    )
    return CoincidentParameterRegion(
        cell_lower=cell_lower,
        cell_upper=cell_upper,
        partial=np.asarray([not full_rectangle]),
        distance_bound=0.0,
    )


def _sphere_frame_valid(sphere: SpherePatch, /) -> bool:
    axes = tuple(
        rational_vector(np.asarray(axis))
        for axis in (sphere.first_axis, sphere.second_axis, sphere.axis)
    )
    return all(rational_dot(axes[i], axes[i]) == 1 for i in range(3)) and all(
        rational_dot(axes[i], axes[j]) == 0 for i in range(3) for j in range(i + 1, 3)
    )


def _sphere_parameters(
    sphere: SpherePatch, point: tuple[Fraction, ...], /
) -> tuple[float, float]:
    center = rational_vector(np.asarray(sphere.center))
    radius = Fraction.from_float(float(sphere.radius))
    direction = tuple(float((point[i] - center[i]) / radius) for i in range(3))
    first = np.asarray(sphere.first_axis, dtype=np.float64)
    second = np.asarray(sphere.second_axis, dtype=np.float64)
    axis = np.asarray(sphere.axis, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    return math.atan2(float(vector @ second), float(vector @ first)), math.asin(
        float(np.clip(vector @ axis, -1.0, 1.0))
    )


def _sphere_region_parameters(
    sphere: SpherePatch, point: tuple[Fraction, ...], region: SurfaceRegion, /
) -> tuple[float, float] | None:
    values = list(_sphere_parameters(sphere, point))
    bounds = np.asarray(region.parameter_box, dtype=np.float64)
    for axis in range(2):
        lower, upper = float(bounds[0, axis]), float(bounds[1, axis])
        symbol = sphere.native_period_symbols()[axis]
        period = sphere.periods[axis]
        if symbol == "two_pi" and period is not None and upper - lower == float(period):
            while values[axis] < lower:
                values[axis] += float(period)
            while values[axis] > upper:
                values[axis] -= float(period)
        if values[axis] < lower or values[axis] > upper:
            return None
    return values[0], values[1]


def _exact_contact_point(
    first: SpherePatch, second: SpherePatch, /, *, external: bool
) -> tuple[Fraction, Fraction, Fraction] | None:
    center_a, center_b = (
        rational_vector(np.asarray(first.center)),
        rational_vector(np.asarray(second.center)),
    )
    radius_a, radius_b = (
        Fraction.from_float(float(first.radius)),
        Fraction.from_float(float(second.radius)),
    )
    delta = tuple(center_b[i] - center_a[i] for i in range(3))
    distance_squared = rational_dot(delta, delta)
    if external:
        denominator = radius_a + radius_b
        if denominator <= 0 or distance_squared != denominator * denominator:
            return None
        factor = radius_a / denominator
    else:
        denominator = radius_a - radius_b
        if denominator == 0 or distance_squared != denominator * denominator:
            return None
        factor = radius_a / denominator
    return (
        center_a[0] + factor * delta[0],
        center_a[1] + factor * delta[1],
        center_a[2] + factor * delta[2],
    )


def _sphere_sphere_contact(
    first: SurfaceRegion, second: SurfaceRegion, policy: ParametricIntersectionPolicy
) -> ParametricIntersectionPoint | None:
    a, b = first.patch, second.patch
    if not isinstance(a, SpherePatch) or not isinstance(b, SpherePatch):
        return None
    if not (_sphere_frame_valid(a) and _sphere_frame_valid(b)):
        return None
    center_a, center_b = (
        rational_vector(np.asarray(a.center)),
        rational_vector(np.asarray(b.center)),
    )
    radius_a, radius_b = (
        Fraction.from_float(float(a.radius)),
        Fraction.from_float(float(b.radius)),
    )
    delta = tuple(center_b[i] - center_a[i] for i in range(3))
    distance_squared = rational_dot(delta, delta)
    external = (radius_a + radius_b) > 0 and distance_squared == (
        radius_a + radius_b
    ) ** 2
    internal = radius_a != radius_b and distance_squared == (radius_a - radius_b) ** 2
    if not (external or internal):
        return None
    exact = _exact_contact_point(a, b, external=external)
    if exact is None:
        return None
    point = np.asarray(tuple(float(value) for value in exact), dtype=np.float64)
    uv_a = _sphere_region_parameters(a, exact, first)
    uv_b = _sphere_region_parameters(b, exact, second)
    if uv_a is None or uv_b is None:
        return None
    parameters = np.asarray((*uv_a, *uv_b), dtype=np.float64)
    lower = np.asarray(
        (*np.asarray(first.parameter_box[0]), *np.asarray(second.parameter_box[0])),
        dtype=np.float64,
    )
    upper = np.asarray(
        (*np.asarray(first.parameter_box[1]), *np.asarray(second.parameter_box[1])),
        dtype=np.float64,
    )
    root_id = canonical_fingerprint(
        {
            "kind": "sphere-sphere-exact-tangency",
            "first": first.identity_payload(),
            "second": second.identity_payload(),
            "point": tuple((value.numerator, value.denominator) for value in exact),
        }
    )
    return ParametricIntersectionPoint(
        kind="tangent_point",
        certificate="analytic-contact",
        parameters=parameters,
        parameter_lower=lower,
        parameter_upper=upper,
        point=point,
        point_exact=exact,
        gap_bound=0.0,
        condition_estimate=float("inf"),
        branch_count=0,
        root_id=root_id,
    )


def _plane_sphere_contact(
    first: SurfaceRegion, second: SurfaceRegion, policy: ParametricIntersectionPolicy
) -> ParametricIntersectionPoint | None:
    if isinstance(first.patch, PlanePatch) and isinstance(second.patch, SpherePatch):
        plane, sphere, plane_region, sphere_region = (
            first.patch,
            second.patch,
            first,
            second,
        )
        plane_first = True
    elif isinstance(first.patch, SpherePatch) and isinstance(second.patch, PlanePatch):
        plane, sphere, plane_region, sphere_region = (
            second.patch,
            first.patch,
            second,
            first,
        )
        plane_first = False
    else:
        return None
    if not _sphere_frame_valid(sphere):
        return None
    u, v = (
        rational_vector(np.asarray(plane.first_axis)),
        rational_vector(np.asarray(plane.second_axis)),
    )
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    norm_squared = rational_dot(normal, normal)
    if norm_squared == 0:
        return None
    center = rational_vector(np.asarray(sphere.center))
    origin = rational_vector(np.asarray(plane.origin))
    radius = Fraction.from_float(float(sphere.radius))
    signed = rational_dot(tuple(center[i] - origin[i] for i in range(3)), normal)
    if signed * signed != radius * radius * norm_squared:
        return None
    exact = tuple(center[i] - signed * normal[i] / norm_squared for i in range(3))
    delta = tuple(exact[i] - origin[i] for i in range(3))
    g00, g01, g11 = rational_dot(u, u), rational_dot(u, v), rational_dot(v, v)
    determinant = g00 * g11 - g01 * g01
    if determinant == 0:
        return None
    rhs0, rhs1 = rational_dot(u, delta), rational_dot(v, delta)
    uv = (
        (g11 * rhs0 - g01 * rhs1) / determinant,
        (g00 * rhs1 - g01 * rhs0) / determinant,
    )
    plane_lower = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(plane_region.parameter_box[0])
    )
    plane_upper = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(plane_region.parameter_box[1])
    )
    if any(uv[i] < plane_lower[i] or uv[i] > plane_upper[i] for i in range(2)):
        return None
    point = np.asarray(tuple(float(value) for value in exact), dtype=np.float64)
    sphere_uv = _sphere_region_parameters(sphere, exact, sphere_region)
    if sphere_uv is None:
        return None
    params = np.asarray(
        (*uv, *sphere_uv) if plane_first else (*sphere_uv, *uv), dtype=np.float64
    )
    first_lower, second_lower = (
        np.asarray(first.parameter_box[0]),
        np.asarray(second.parameter_box[0]),
    )
    first_upper, second_upper = (
        np.asarray(first.parameter_box[1]),
        np.asarray(second.parameter_box[1]),
    )
    lower = np.asarray((*first_lower, *second_lower), dtype=np.float64)
    upper = np.asarray((*first_upper, *second_upper), dtype=np.float64)
    root_id = canonical_fingerprint(
        {
            "kind": "plane-sphere-exact-tangency",
            "first": first.identity_payload(),
            "second": second.identity_payload(),
            "point": tuple((value.numerator, value.denominator) for value in exact),
        }
    )
    return ParametricIntersectionPoint(
        kind="tangent_point",
        certificate="analytic-contact",
        parameters=params,
        parameter_lower=lower,
        parameter_upper=upper,
        point=point,
        point_exact=exact,
        gap_bound=0.0,
        condition_estimate=float("inf"),
        branch_count=0,
        root_id=root_id,
    )


__all__ = [
    "BranchRootEndpoint",
    "CoincidentParameterRegion",
    "ConicRootLineEndpointWitness",
    "ConicParameterEndpoint",
    "CurveIntersectionResult",
    "CurveRange",
    "CurveSurfaceIntersectionRoot",
    "IntersectionCurvePointRoot",
    "NativePeriodEndpoint",
    "ParametricIntersectionKind",
    "ParametricIntersectionPoint",
    "ParametricIntersectionPolicy",
    "RootEndpoint",
    "SurfaceIntersectionResult",
    "TripleSurfaceIntersectionRoot",
    "TrimIntersectionRoot",
    "TrimRootEndpoint",
    "UnresolvedParameterRegion",
    "endpoint_root_payload",
    "intersect_surface_regions",
]


def _contains_zero(lower: np.ndarray, upper: np.ndarray, /) -> bool:
    return bool(np.all((lower <= 0.0) & (upper >= 0.0)))


@dataclass(frozen=True, slots=True)
class _RootIsolation:
    roots: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]
    unresolved: tuple[UnresolvedParameterRegion, ...]


def _isolate_square_system(
    system: _RootEquation,
    lower: ArrayLike,
    upper: ArrayLike,
    domain_lower: ArrayLike,
    domain_upper: ArrayLike,
    policy: ParametricIntersectionPolicy,
    /,
) -> _RootIsolation:
    """Exhaustively exclude or uniquely isolate every square-system root."""
    initial_lower = np.asarray(lower, dtype=np.float64)
    initial_upper = np.asarray(upper, dtype=np.float64)
    domain_width = np.asarray(domain_upper, dtype=np.float64) - np.asarray(
        domain_lower, dtype=np.float64
    )
    pending = [(initial_lower, initial_upper)]
    roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    unresolved: list[UnresolvedParameterRegion] = []
    while pending:
        box_lower, box_upper = pending.pop()
        value_lower, value_upper = system.value.evaluate(box_lower[None], box_upper[None])
        if not _contains_zero(value_lower[0], value_upper[0]):
            continue
        witness = krawczyk(system, box_lower[None], box_upper[None])
        if bool(witness.excluded[0]):
            continue
        if bool(witness.certified[0]):
            isolated_lower = np.maximum(box_lower, witness.lower[0])
            isolated_upper = np.minimum(box_upper, witness.upper[0])
            point = 0.5 * isolated_lower + 0.5 * isolated_upper
            if not any(
                np.all(point >= old_lower) and np.all(point <= old_upper)
                for _, old_lower, old_upper, _ in roots
            ):
                roots.append(
                    (
                        point,
                        isolated_lower,
                        isolated_upper,
                        witness.preconditioner[0],
                    )
                )
            continue
        widths = box_upper - box_lower
        resolution = policy.relative_resolution * np.maximum(1.0, np.abs(domain_width))
        midpoint = box_lower + 0.5 * widths
        representable = (midpoint > box_lower) & (midpoint < box_upper)
        if np.any(widths <= resolution) or not np.all(representable):
            unresolved.append(
                UnresolvedParameterRegion(box_lower, box_upper, "resolution")
            )
            continue
        axis = int(
            np.argmax(widths / np.maximum(np.abs(domain_width), np.finfo(float).tiny))
        )
        left_upper = box_upper.copy()
        left_upper[axis] = midpoint[axis]
        right_lower = box_lower.copy()
        right_lower[axis] = midpoint[axis]
        pending.append((right_lower, box_upper.copy()))
        pending.append((box_lower.copy(), left_upper))
    roots.sort(key=lambda item: tuple(float(value) for value in item[0]))
    return _RootIsolation(tuple(roots), tuple(unresolved))


def intersect_trim_curves(
    first: CurveRange | AbstractCurve,
    second: CurveRange | AbstractCurve,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
    source_patch: AbstractSurfacePatch | None = None,
) -> CurveIntersectionResult:
    """Isolate all regular intersections of two planar source trim curves."""
    a = first if isinstance(first, CurveRange) else CurveRange(first)
    b = second if isinstance(second, CurveRange) else CurveRange(second)
    if a.curve.ambient_dimension != 2 or b.curve.ambient_dimension != 2:
        return CurveIntersectionResult(
            unresolved=(
                UnresolvedParameterRegion(
                    np.asarray((a.first, b.first)),
                    np.asarray((a.last, b.last)),
                    "continuation",
                ),
            )
        )
    if source_patch is not None and not isinstance(source_patch, AbstractSurfacePatch):
        raise TypeError("source_patch must be an AbstractSurfacePatch or None.")
    selected = ParametricIntersectionPolicy() if policy is None else policy
    first_segments = _curve_chart_intervals(a.curve, a.first, a.last)
    second_segments = _curve_chart_intervals(b.curve, b.first, b.last)
    if first_segments is None or second_segments is None:
        return CurveIntersectionResult(
            unresolved=(
                UnresolvedParameterRegion(
                    np.asarray((a.first, b.first)),
                    np.asarray((a.last, b.last)),
                    "continuation",
                ),
            )
        )
    domain_lower = np.asarray((a.first, b.first), dtype=np.float64)
    domain_upper = np.asarray((a.last, b.last), dtype=np.float64)
    root_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    unresolved: list[UnresolvedParameterRegion] = []
    for first_lower, first_upper, first_chart in first_segments:
        for second_lower, second_upper, second_chart in second_segments:
            if _trim_chart_boxes_disjoint(
                a.curve,
                first_lower,
                first_upper,
                b.curve,
                second_lower,
                second_upper,
                source_patch,
            ):
                continue

            first_value = _curve_chart_evaluator(a.curve, first_chart)
            second_value = _curve_chart_evaluator(b.curve, second_chart)

            def residual(parameters: Array) -> Array:
                return first_value(parameters[0]) - second_value(parameters[1])

            system = _RootEquation(residual, 2)
            segment_lower = np.asarray((first_lower, second_lower), dtype=np.float64)
            segment_upper = np.asarray((first_upper, second_upper), dtype=np.float64)
            isolated = _isolate_square_system(
                system,
                segment_lower,
                segment_upper,
                domain_lower,
                domain_upper,
                selected,
            )
            root_records.extend(isolated.roots)
            unresolved.extend(isolated.unresolved)

    points: list[ParametricIntersectionPoint] = []
    seen_roots: set[str] = set()
    for parameters, lower, upper, _ in root_records:
        try:
            root = TrimIntersectionRoot(
                a.curve,
                b.curve,
                parameter_lower=lower,
                parameter_upper=upper,
            )
        except ValueError:
            unresolved.append(UnresolvedParameterRegion(lower, upper, "continuation"))
            continue
        if root.root_id in seen_roots:
            continue
        seen_roots.add(root.root_id)
        point = np.asarray(a.curve.evaluate(jnp.asarray(parameters[0])))
        gap = float(
            np.linalg.norm(
                point - np.asarray(b.curve.evaluate(jnp.asarray(parameters[1])))
            )
        )
        points.append(
            ParametricIntersectionPoint(
                kind="crossing",
                certificate="krawczyk",
                parameters=parameters,
                parameter_lower=lower,
                parameter_upper=upper,
                point=point,
                gap_bound=gap,
                condition_estimate=1.0,
                root_id=root.root_id,
            )
        )
    return CurveIntersectionResult(points, unresolved=tuple(unresolved))


def _line_cylinder_region_intersections(
    source: CurveRange,
    surface: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> CurveIntersectionResult | None:
    """Isolate a line/revolution-cylinder pair from its exact quadratic."""
    from .._rational import fraction_down, fraction_up
    from ._patches import CylinderPatch, ExactLineCurve, LineCurve, RevolutionSurface

    if not isinstance(surface.patch, RevolutionSurface) or not isinstance(
        source.curve, (LineCurve, ExactLineCurve)
    ):
        return None
    view = _revolution_cylinder_view(surface)
    if view is None:
        return None
    cylinder_region, profile_slope = view
    cylinder_patch = cylinder_region.patch
    if not isinstance(cylinder_patch, CylinderPatch):
        return None
    cylinder = cylinder_patch
    if (
        surface.patch.periods[0] is None
        or surface.patch.native_period_symbols()[0] != "two_pi"
        or Fraction.from_float(float(surface.parameter_box[1, 0]))
        - Fraction.from_float(float(surface.parameter_box[0, 0]))
        != Fraction.from_float(float(surface.patch.periods[0]))
    ):
        return None
    if isinstance(source.curve, LineCurve):
        line_origin = source.curve.origin_exact
        line_direction = source.curve.direction_exact
    else:
        line_origin = source.curve.first_exact
        line_direction = tuple(
            last - first
            for first, last in zip(
                source.curve.first_exact,
                source.curve.last_exact,
                strict=True,
            )
        )
    vectors = (
        line_origin,
        line_direction,
        cylinder.origin_exact,
        cylinder.axis_exact,
        cylinder.first_axis_exact,
        cylinder.second_axis_exact,
    )
    if any(
        not all(isinstance(value, Fraction) for value in vector) for vector in vectors
    ) or not isinstance(cylinder.radius_exact, Fraction):
        return None
    line_origin = cast(tuple[Fraction, ...], line_origin)
    line_direction = cast(tuple[Fraction, ...], line_direction)
    cylinder_origin = cast(tuple[Fraction, ...], cylinder.origin_exact)
    axis = cast(tuple[Fraction, ...], cylinder.axis_exact)
    first_axis = cast(tuple[Fraction, ...], cylinder.first_axis_exact)
    second_axis = cast(tuple[Fraction, ...], cylinder.second_axis_exact)
    radius = cylinder.radius_exact
    displacement = tuple(
        point - origin for point, origin in zip(line_origin, cylinder_origin, strict=True)
    )
    axial_offset = rational_dot(displacement, axis)
    axial_rate = rational_dot(line_direction, axis)
    radial_offset = tuple(
        value - axial_offset * direction
        for value, direction in zip(displacement, axis, strict=True)
    )
    radial_direction = tuple(
        value - axial_rate * direction
        for value, direction in zip(line_direction, axis, strict=True)
    )
    quadratic = rational_dot(radial_direction, radial_direction)
    linear = 2 * rational_dot(radial_offset, radial_direction)
    constant = rational_dot(radial_offset, radial_offset) - radius * radius
    domain_lower = np.concatenate(
        (
            np.asarray((source.first,), dtype=np.float64),
            np.asarray(surface.parameter_box[0], dtype=np.float64),
        )
    )
    domain_upper = np.concatenate(
        (
            np.asarray((source.last,), dtype=np.float64),
            np.asarray(surface.parameter_box[1], dtype=np.float64),
        )
    )
    if quadratic == 0:
        if constant == 0:
            return CurveIntersectionResult(
                unresolved=(
                    UnresolvedParameterRegion(domain_lower, domain_upper, "singular"),
                )
            )
        return CurveIntersectionResult()
    discriminant = linear * linear - 4 * quadratic * constant
    if discriminant < 0:
        return CurveIntersectionResult()
    if discriminant == 0:
        return CurveIntersectionResult(
            unresolved=(
                UnresolvedParameterRegion(domain_lower, domain_upper, "singular"),
            )
        )
    radical = RadicalElement.sqrt(discriminant)
    source_roots = (
        (-linear - radical) / (2 * quadratic),
        (-linear + radical) / (2 * quadratic),
    )
    source_lower = Fraction.from_float(float(source.first))
    source_upper = Fraction.from_float(float(source.last))
    profile_lower = Fraction.from_float(float(surface.parameter_box[0, 1]))
    profile_upper = Fraction.from_float(float(surface.parameter_box[1, 1]))
    context = interval_context(100)

    def scalar(value: Fraction) -> Any:
        return context.mpf(value.numerator) / context.mpf(value.denominator)

    def interval(value: ExactScalar) -> Any:
        if isinstance(value, Fraction):
            return scalar(value)
        if isinstance(value, RadicalElement):
            lower, upper = value.interval(bits=40)
            return context.mpf((scalar(lower), scalar(upper)))
        raise TypeError("Line-cylinder roots must retain rational radical coordinates.")

    def outward(value: ExactScalar, parameter_scale: float, /) -> tuple[float, float]:
        padding = Fraction.from_float(
            policy.relative_resolution * max(1.0, parameter_scale)
        )
        if isinstance(value, Fraction):
            lower_fraction, upper_fraction = value - padding, value + padding
        elif isinstance(value, RadicalElement):
            lower_fraction, upper_fraction = value.interval(bits=40)
            lower_fraction -= padding
            upper_fraction += padding
        else:
            raise TypeError(
                "Line-cylinder roots must retain rational radical coordinates."
            )
        return fraction_down(lower_fraction), fraction_up(upper_fraction)

    def exact_dot(
        first: tuple[ExactScalar, ...], second: tuple[Fraction, ...], /
    ) -> ExactScalar:
        return sum(
            (
                value * coefficient
                for value, coefficient in zip(first, second, strict=True)
            ),
            Fraction(0),
        )

    points: list[ParametricIntersectionPoint] = []
    for source_parameter in source_roots:
        if source_parameter < source_lower or source_parameter > source_upper:
            continue
        point = tuple(
            origin + source_parameter * direction
            for origin, direction in zip(line_origin, line_direction, strict=True)
        )
        relative = tuple(
            value - origin for value, origin in zip(point, cylinder_origin, strict=True)
        )
        axial = exact_dot(relative, axis)
        radial = tuple(
            value - axial * direction
            for value, direction in zip(relative, axis, strict=True)
        )
        profile_parameter = axial / profile_slope
        if profile_parameter < profile_lower or profile_parameter > profile_upper:
            continue
        cosine = exact_dot(radial, first_axis)
        sine = exact_dot(radial, second_axis)
        angle = context.atan2(interval(sine), interval(cosine))
        angle_lower = math.nextafter(float(angle.a), -math.inf)
        angle_upper = math.nextafter(float(angle.b), math.inf)
        period = float(surface.patch.periods[0])
        turns = math.ceil(
            (float(surface.parameter_box[0, 0]) - 0.5 * angle_lower - 0.5 * angle_upper)
            / period
        )
        angle_scale = max(1.0, period)
        angle_padding = policy.relative_resolution * angle_scale
        angle_lower = math.nextafter(
            angle_lower + turns * period - angle_padding,
            -math.inf,
        )
        angle_upper = math.nextafter(
            angle_upper + turns * period + angle_padding,
            math.inf,
        )
        line_bounds, profile_bounds = (
            outward(source_parameter, source.last - source.first),
            outward(
                profile_parameter,
                float(surface.parameter_box[1, 1] - surface.parameter_box[0, 1]),
            ),
        )
        lower = np.asarray(
            (line_bounds[0], angle_lower, profile_bounds[0]), dtype=np.float64
        )
        upper = np.asarray(
            (line_bounds[1], angle_upper, profile_bounds[1]), dtype=np.float64
        )
        try:
            root = CurveSurfaceIntersectionRoot(
                source.curve,
                surface,
                parameter_lower=lower,
                parameter_upper=upper,
            )
        except ValueError as error:
            if "root box" not in str(error):
                raise
            return CurveIntersectionResult(
                points=points,
                unresolved=(UnresolvedParameterRegion(lower, upper, "continuation"),),
            )
        parameters = np.asarray(root.parameters, dtype=np.float64)
        source_point = np.asarray(
            source.curve.evaluate(jnp.asarray(parameters[0])), dtype=np.float64
        )
        surface_point = np.asarray(
            surface.patch.evaluate(jnp.asarray(parameters[1:])), dtype=np.float64
        )
        points.append(
            ParametricIntersectionPoint(
                kind="crossing",
                certificate="krawczyk",
                parameters=parameters,
                parameter_lower=np.asarray(root.parameter_lower, dtype=np.float64),
                parameter_upper=np.asarray(root.parameter_upper, dtype=np.float64),
                point=source_point,
                gap_bound=float(np.linalg.norm(source_point - surface_point)),
                condition_estimate=1.0,
                root_id=root.root_id,
            )
        )
    return CurveIntersectionResult(points=points)


def intersect_curve_ranges(
    first: CurveRange | AbstractCurve,
    second: CurveRange | AbstractCurve,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
) -> CurveIntersectionResult:
    """Intersect equal-dimensional source curves over their authored ranges."""
    return intersect_trim_curves(first, second, policy=policy)


def intersect_curve_region(
    curve_range: CurveRange | AbstractCurve,
    surface: SurfaceRegion,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
) -> CurveIntersectionResult:
    """Isolate regular intersections between a 3D source curve and a surface."""
    source = (
        curve_range if isinstance(curve_range, CurveRange) else CurveRange(curve_range)
    )
    if source.curve.ambient_dimension != 3:
        raise ValueError("intersect_curve_region requires a three-dimensional curve.")
    selected = ParametricIntersectionPolicy() if policy is None else policy
    line_cylinder = _line_cylinder_region_intersections(source, surface, selected)
    if line_cylinder is not None:
        return line_cylinder

    def residual(parameters: Array) -> Array:
        return source.curve.evaluate(parameters[0]) - surface.patch.evaluate(
            parameters[1:]
        )

    domain_lower = np.asarray((source.first, *np.asarray(surface.parameter_box[0])))
    domain_upper = np.asarray((source.last, *np.asarray(surface.parameter_box[1])))
    system = _RootEquation(residual, 3)
    roots = _isolate_square_system(
        system,
        domain_lower,
        domain_upper,
        domain_lower,
        domain_upper,
        selected,
    )
    points: list[ParametricIntersectionPoint] = []
    for parameters, lower, upper, _ in roots.roots:
        root = CurveSurfaceIntersectionRoot(
            source.curve,
            surface,
            parameter_lower=lower,
            parameter_upper=upper,
        )
        spatial = np.asarray(source.curve.evaluate(jnp.asarray(parameters[0])))
        target = np.asarray(surface.patch.evaluate(jnp.asarray(parameters[1:])))
        points.append(
            ParametricIntersectionPoint(
                kind="crossing",
                certificate="krawczyk",
                parameters=parameters,
                parameter_lower=lower,
                parameter_upper=upper,
                point=spatial,
                gap_bound=float(np.linalg.norm(spatial - target)),
                condition_estimate=1.0,
                root_id=root.root_id,
            )
        )
    return CurveIntersectionResult(points, unresolved=roots.unresolved)


__all__ = [
    *__all__,
    "intersect_curve_ranges",
    "intersect_curve_region",
    "intersect_trim_curves",
]


def _exact_unit_frame_values(
    first: tuple[ExactScalar, ...],
    second: tuple[ExactScalar, ...],
    axis: tuple[ExactScalar, ...],
    /,
) -> (
    tuple[
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
    ]
    | None
):
    if any(
        len(vector) != 3
        or any(not isinstance(value, (Fraction, RadicalElement)) for value in vector)
        for vector in (first, second, axis)
    ):
        return None

    def exact_dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    if (
        exact_dot(first, second) != 0
        or exact_dot(first, first) != 1
        or exact_dot(second, second) != 1
        or exact_dot(axis, axis) != 1
        or exact_dot(first, axis) != 0
        or exact_dot(second, axis) != 0
    ):
        return None
    return (
        (first[0], first[1], first[2]),
        (second[0], second[1], second[2]),
        (axis[0], axis[1], axis[2]),
    )


def _exact_unit_frame(
    first: ArrayLike, second: ArrayLike, axis: ArrayLike, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    first_exact = rational_vector(np.asarray(first, dtype=np.float64))
    second_exact = rational_vector(np.asarray(second, dtype=np.float64))
    axis_exact = rational_vector(np.asarray(axis, dtype=np.float64))
    if _exact_unit_frame_values(first_exact, second_exact, axis_exact) is None:
        return None
    return first_exact, second_exact, axis_exact


def _cylinder_chart_point(
    first: CylinderPatch,
    second: CylinderPatch,
    n: np.ndarray,
    delta_n: float,
    delta_a: float,
    delta_b: float,
    phi: float,
    orientation: float,
    radius_first: float,
    radius_second: float,
    sign: int,
    parameter: float,
    first_angle_reference: float,
) -> np.ndarray:
    first_axis = np.asarray(first.axis, dtype=np.float64)
    second_axis = np.asarray(second.axis, dtype=np.float64)
    radial_b = second_axis
    theta = phi + orientation * parameter
    y = delta_n + radius_second * math.cos(theta)
    z = delta_a + radius_second * math.sin(theta)
    x = sign * math.sqrt(radius_first * radius_first - y * y)
    point = (
        np.asarray(first.origin, dtype=np.float64) + x * radial_b + y * n + z * first_axis
    )
    radial_first = point - np.asarray(first.origin, dtype=np.float64) - z * first_axis
    angle_first = math.atan2(
        float(radial_first @ np.asarray(first.second_axis)),
        float(radial_first @ np.asarray(first.first_axis)),
    )
    angle_first += math.tau * round((first_angle_reference - angle_first) / math.tau)
    height_second = float(
        (point - np.asarray(second.origin, dtype=np.float64)) @ second_axis
    )
    return np.asarray((angle_first, z, parameter, height_second), dtype=np.float64)


def _surface_branch_node_box(
    pair: SurfacePairSystem,
    parameters: np.ndarray,
    axis: int,
    value: float,
    policy: ParametricIntersectionPolicy,
    /,
    *,
    pair_prepared: KrawczykEvaluator | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    from ._intersection_curve import (
        _exact_ruled_plane_node_prepared,
        _NodeSystem,
        _prepare_pair,
        _PreparedNode,
    )

    system = _NodeSystem(pair, axis, value)
    prepared = _exact_ruled_plane_node_prepared(pair.first, pair.second, axis, value)
    if prepared is None:
        pair_prepared = _prepare_pair(pair) if pair_prepared is None else pair_prepared
        prepared = _PreparedNode(system, pair_prepared)
    scale = max(1.0, float(np.max(np.abs(parameters))))
    radius = 64.0 * np.finfo(np.float64).eps * scale
    maximum = policy.relative_probe * scale
    while radius <= maximum:
        lower, upper = parameters - radius, parameters + radius
        lower[axis], upper[axis] = value - radius, value + radius
        result = krawczyk(prepared, lower[None], upper[None])
        if bool(result.certified[0]):
            return lower, upper
        next_radius = radius * 2.0
        if next_radius <= radius:
            break
        radius = next_radius
    return None


def _periodic_endpoint_deck_shift(
    endpoint: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    patches: tuple[AbstractSurfacePatch, AbstractSurfacePatch],
    /,
) -> np.ndarray | None:
    endpoint_ = np.asarray(endpoint, dtype=np.float64)
    lower_ = np.asarray(lower, dtype=np.float64)
    upper_ = np.asarray(upper, dtype=np.float64)
    if (
        endpoint_.shape != (4,)
        or lower_.shape != (4,)
        or upper_.shape != (4,)
        or not np.all(np.isfinite(endpoint_))
        or not np.all(np.isfinite(lower_))
        or not np.all(np.isfinite(upper_))
        or np.any(lower_ >= upper_)
    ):
        return None
    shifts = np.zeros((4,), dtype=np.int32)
    for axis, value in enumerate(endpoint_):
        exact_value = Fraction.from_float(float(value))
        exact_lower = Fraction.from_float(float(lower_[axis]))
        exact_upper = Fraction.from_float(float(upper_[axis]))
        if exact_lower <= exact_value <= exact_upper:
            continue
        region_index, local_axis = divmod(axis, 2)
        patch = patches[region_index]
        period = patch.periods[local_axis]
        if period is None or patch.native_period_symbols()[local_axis] != "two_pi":
            return None
        period_value = float(np.asarray(period))
        center = 0.5 * (float(lower_[axis]) + float(upper_[axis]))
        estimate = int(round((float(value) - center) / period_value))
        matches: list[int] = []
        for turn in range(estimate - 2, estimate + 3):
            shift_lower, shift_upper = period_scalar_bounds(
                PeriodOffset(Fraction(0), Fraction(turn))
            )
            exact_shift_lower = Fraction.from_float(float(np.asarray(shift_lower)))
            exact_shift_upper = Fraction.from_float(float(np.asarray(shift_upper)))
            if (
                exact_lower + exact_shift_lower
                <= exact_value
                <= exact_upper + exact_shift_upper
            ):
                matches.append(turn)
        if (
            len(matches) != 1
            or matches[0] < np.iinfo(np.int32).min
            or matches[0] > np.iinfo(np.int32).max
        ):
            return None
        shifts[axis] = matches[0]
    return shifts


def _orthogonal_cylinder_branch(
    first: SurfaceRegion,
    second: SurfaceRegion,
    sign: int,
    policy: ParametricIntersectionPolicy,
    /,
    *,
    source_regions: tuple[SurfaceRegion, SurfaceRegion] | None = None,
    source_slopes: tuple[Fraction, Fraction] | None = None,
    pair_preparation: Callable[
        [SurfacePairSystem, tuple[SurfaceRegion, SurfaceRegion] | None],
        KrawczykEvaluator,
    ]
    | None = None,
) -> tuple[IntersectionCurve | None, tuple[UnresolvedParameterRegion, ...]]:
    from ._intersection_curve import _prepare_pair

    a, b = first.patch, second.patch
    if not isinstance(a, CylinderPatch) or not isinstance(b, CylinderPatch):
        return None, ()
    frame_a = _exact_unit_frame(a.first_axis, a.second_axis, a.axis)
    frame_b = _exact_unit_frame(b.first_axis, b.second_axis, b.axis)
    if frame_a is None or frame_b is None:
        return None, ()
    aa, ae, axis_a = frame_a
    ba, be, axis_b = frame_b
    if rational_dot(axis_a, axis_b) != 0:
        return None, ()
    cross = (
        axis_a[1] * axis_b[2] - axis_a[2] * axis_b[1],
        axis_a[2] * axis_b[0] - axis_a[0] * axis_b[2],
        axis_a[0] * axis_b[1] - axis_a[1] * axis_b[0],
    )
    if rational_dot(cross, cross) != 1:
        return None, ()
    n = np.asarray(tuple(float(value) for value in cross), dtype=np.float64)
    first_axis = np.asarray(a.axis, dtype=np.float64)
    second_axis = np.asarray(b.axis, dtype=np.float64)
    radial_b = np.asarray(b.first_axis, dtype=np.float64)
    radial_b_second = np.asarray(b.second_axis, dtype=np.float64)
    phase = math.atan2(float(radial_b @ first_axis), float(radial_b @ n))
    orientation = (
        1.0
        if float(np.dot(np.cross(radial_b, radial_b_second), second_axis)) > 0.0
        else -1.0
    )
    radius_first, radius_second = float(a.radius), float(b.radius)
    if not (radius_first > 0.0 and radius_second > 0.0):
        return None, ()
    origin_delta = np.asarray(b.origin, dtype=np.float64) - np.asarray(
        a.origin, dtype=np.float64
    )
    exact_delta_n = rational_dot(rational_vector(origin_delta), rational_vector(n))
    exact_delta_a = rational_dot(
        rational_vector(origin_delta), rational_vector(first_axis)
    )
    exact_delta_b = rational_dot(
        rational_vector(origin_delta), rational_vector(second_axis)
    )
    if abs(exact_delta_n) + Fraction.from_float(radius_second) >= Fraction.from_float(
        radius_first
    ):
        return None, ()
    delta_n, delta_a, delta_b = (
        float(exact_delta_n),
        float(exact_delta_a),
        float(exact_delta_b),
    )
    first_box, second_box = (
        np.asarray(first.parameter_box),
        np.asarray(second.parameter_box),
    )
    if (
        a.native_period_symbols()[0] != "two_pi"
        or b.native_period_symbols()[0] != "two_pi"
    ):
        return None, ()
    period_a_value, period_b_value = a.periods[0], b.periods[0]
    if period_a_value is None or period_b_value is None:
        return None, ()
    period_a, period_b = float(period_a_value), float(period_b_value)
    first_period = Fraction.from_float(float(first_box[1, 0])) - Fraction.from_float(
        float(first_box[0, 0])
    )
    second_period = Fraction.from_float(float(second_box[1, 0])) - Fraction.from_float(
        float(second_box[0, 0])
    )
    if first_period != Fraction.from_float(
        period_a
    ) or second_period != Fraction.from_float(period_b):
        return None, ()
    u_lower, u_upper = float(second_box[0, 0]), float(second_box[1, 0])
    z_lower, z_upper = delta_a - radius_second, delta_a + radius_second
    x_min = math.sqrt(radius_first * radius_first - (abs(delta_n) + radius_second) ** 2)
    v2_lower, v2_upper = sorted((sign * x_min - delta_b, sign * radius_first - delta_b))
    if z_lower < float(first_box[0, 1]) or z_upper > float(first_box[1, 1]):
        return None, ()
    if v2_lower < float(second_box[0, 1]) or v2_upper > float(second_box[1, 1]):
        return None, ()
    if (source_regions is None) != (source_slopes is None):
        return None, ()
    if source_slopes is not None and any(slope == 0 for slope in source_slopes):
        return None, ()
    pair = SurfacePairSystem(a, b)
    prepared = (
        _prepare_pair(pair) if pair_preparation is None else pair_preparation(pair, None)
    )
    source_pair = None
    source_prepared = None
    source_lower = source_upper = None
    if source_regions is not None and source_slopes is not None:
        source_pair = SurfacePairSystem(source_regions[0].patch, source_regions[1].patch)
        source_prepared = (
            _prepare_pair(source_pair, source_regions=source_regions)
            if pair_preparation is None
            else pair_preparation(source_pair, source_regions)
        )
        first_source_box = np.asarray(source_regions[0].parameter_box, dtype=np.float64)
        second_source_box = np.asarray(source_regions[1].parameter_box, dtype=np.float64)
        source_lower = np.concatenate((first_source_box[0], second_source_box[0]))
        source_upper = np.concatenate((first_source_box[1], second_source_box[1]))

    def source_point(point: np.ndarray, /) -> np.ndarray:
        result = point.copy()
        if source_slopes is not None:
            for axis, slope in ((1, source_slopes[0]), (3, source_slopes[1])):
                result[axis] = float(Fraction.from_float(float(point[axis])) / slope)
        return result

    def source_box(
        lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        result_lower, result_upper = lower.copy(), upper.copy()
        if source_slopes is not None:
            from .._rational import fraction_down, fraction_up

            for axis, slope in ((1, source_slopes[0]), (3, source_slopes[1])):
                first_value = Fraction.from_float(float(lower[axis])) / slope
                last_value = Fraction.from_float(float(upper[axis])) / slope
                result_lower[axis] = fraction_down(min(first_value, last_value))
                result_upper[axis] = fraction_up(max(first_value, last_value))
        return result_lower, result_upper

    def source_bounds_admit(lower: np.ndarray, upper: np.ndarray, /) -> bool:
        if source_regions is None or source_lower is None or source_upper is None:
            return False
        for axis in range(4):
            if source_lower[axis] <= lower[axis] and upper[axis] <= source_upper[axis]:
                continue
            region_index, local_axis = divmod(axis, 2)
            patch = source_regions[region_index].patch
            period = patch.periods[local_axis]
            symbols = patch.native_period_symbols()
            if period is None or symbols[local_axis] != "two_pi":
                return False
            region_box = np.asarray(
                source_regions[region_index].parameter_box, dtype=np.float64
            )
            exact_span = Fraction.from_float(float(region_box[1, local_axis])) - (
                Fraction.from_float(float(region_box[0, local_axis]))
            )
            exact_period = Fraction.from_float(float(period))
            exact_width = Fraction.from_float(float(upper[axis])) - Fraction.from_float(
                float(lower[axis])
            )
            if exact_span != exact_period or exact_width > exact_period:
                return False
        return True

    def certify_source_chart(
        lower: np.ndarray,
        upper: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        /,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if source_prepared is None or source_lower is None or source_upper is None:
            return source_point(start), source_point(end), lower.copy(), upper.copy()
        chart_lower, chart_upper = source_box(lower, upper)
        chart_start, chart_end = source_point(start), source_point(end)
        if not source_bounds_admit(chart_lower, chart_upper):
            return None
        witness = krawczyk(
            source_prepared,
            chart_lower[None, :],
            chart_upper[None, :],
            parameter_axis=2,
        )
        if not (
            bool(witness.certified[0])
            and np.all(np.isfinite(witness.lower[0]))
            and np.all(np.isfinite(witness.upper[0]))
        ):
            return None
        if (
            np.any(chart_start < chart_lower)
            or np.any(chart_start > chart_upper)
            or np.any(chart_end < chart_lower)
            or np.any(chart_end > chart_upper)
        ):
            return None
        return chart_start, chart_end, chart_lower, chart_upper

    start_angle = float(np.asarray(first_box[0, 0]))
    starts: list[float] = [u_lower, u_upper]
    theta_start, theta_end = phase + orientation * u_lower, phase + orientation * u_upper
    theta_lo, theta_hi = sorted((theta_start, theta_end))
    first_multiple = math.floor(theta_lo / (math.pi / 2.0)) - 1
    last_multiple = math.ceil(theta_hi / (math.pi / 2.0)) + 1
    for multiple in range(first_multiple, last_multiple + 1):
        theta = multiple * (math.pi / 2.0)
        u = orientation * (theta - phase)
        if u_lower < u < u_upper:
            starts.append(u)
    if abs(delta_n) < radius_second:
        root_angle = math.acos(-delta_n / radius_second)
        for theta_base in (root_angle, -root_angle):
            low_k = math.floor((theta_lo - theta_base) / math.tau) - 1
            high_k = math.ceil((theta_hi - theta_base) / math.tau) + 1
            for winding in range(low_k, high_k + 1):
                theta = theta_base + winding * math.tau
                u = orientation * (theta - phase)
                if u_lower < u < u_upper:
                    starts.append(u)
    breakpoints = sorted(set(starts))
    pending = [
        (breakpoints[index], breakpoints[index + 1])
        for index in range(len(breakpoints) - 1)
    ]
    segments: list[
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    unresolved: list[UnresolvedParameterRegion] = []
    while pending:
        lo, hi = pending.pop()
        middle = lo + 0.5 * (hi - lo)
        reference = _cylinder_chart_point(
            a,
            b,
            n,
            delta_n,
            delta_a,
            delta_b,
            phase,
            orientation,
            radius_first,
            radius_second,
            sign,
            middle,
            start_angle,
        )[0]
        values = [lo, middle, hi]
        theta_l, theta_h = phase + orientation * lo, phase + orientation * hi
        theta_min, theta_max = sorted((theta_l, theta_h))
        for multiple in range(
            math.floor(theta_min / (math.pi / 2.0)) - 1,
            math.ceil(theta_max / (math.pi / 2.0)) + 2,
        ):
            u = orientation * (multiple * (math.pi / 2.0) - phase)
            if lo < u < hi:
                values.append(u)
        if abs(delta_n) < radius_second:
            angle = math.acos(-delta_n / radius_second)
            for base in (angle, -angle):
                for winding in range(
                    math.floor((theta_min - base) / math.tau) - 1,
                    math.ceil((theta_max - base) / math.tau) + 2,
                ):
                    u = orientation * (base + winding * math.tau - phase)
                    if lo < u < hi:
                        values.append(u)
        q_values = np.stack(
            [
                _cylinder_chart_point(
                    a,
                    b,
                    n,
                    delta_n,
                    delta_a,
                    delta_b,
                    phase,
                    orientation,
                    radius_first,
                    radius_second,
                    sign,
                    u,
                    reference,
                )
                for u in values
            ]
        )
        chart_lo, chart_hi = np.min(q_values, axis=0), np.max(q_values, axis=0)
        chart_lo[2], chart_hi[2] = lo, hi
        free = np.asarray((0, 1, 3))
        pad = np.maximum(
            0.2 * (chart_hi[free] - chart_lo[free]),
            128
            * np.finfo(float).eps
            * (1 + np.maximum(np.abs(chart_lo[free]), np.abs(chart_hi[free]))),
        )
        chart_lo[free] -= pad
        chart_hi[free] += pad
        result = krawczyk(prepared, chart_lo[None], chart_hi[None], parameter_axis=2)
        if bool(result.certified[0]):
            q_lo = _cylinder_chart_point(
                a,
                b,
                n,
                delta_n,
                delta_a,
                delta_b,
                phase,
                orientation,
                radius_first,
                radius_second,
                sign,
                lo,
                reference,
            )
            q_hi = _cylinder_chart_point(
                a,
                b,
                n,
                delta_n,
                delta_a,
                delta_b,
                phase,
                orientation,
                radius_first,
                radius_second,
                sign,
                hi,
                reference,
            )
            source_chart = certify_source_chart(chart_lo, chart_hi, q_lo, q_hi)
            if source_chart is not None:
                chart_start, chart_end, chart_lower, chart_upper = source_chart
                segments.append(
                    (lo, hi, chart_start, chart_end, chart_lower, chart_upper)
                )
                continue
        physical_resolution = policy.relative_resolution * max(1.0, abs(hi - lo))
        if hi - lo <= physical_resolution or not lo < middle < hi:
            unresolved_lower, unresolved_upper = (
                source_box(chart_lo, chart_hi)
                if source_regions is not None
                else (chart_lo, chart_hi)
            )
            unresolved.append(
                UnresolvedParameterRegion(
                    unresolved_lower, unresolved_upper, "resolution"
                )
            )
            continue
        pending.append((middle, hi))
        pending.append((lo, middle))
    if unresolved:
        return None, tuple(unresolved)
    segments.sort(key=lambda value: value[0])
    node_values = [segments[0][0]] + [segment[1] for segment in segments]
    node_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    pair_for_curve = pair if source_pair is None else source_pair
    curve_first = first if source_regions is None else source_regions[0]
    curve_second = second if source_regions is None else source_regions[1]
    for index, value in enumerate(node_values):
        reference = segments[0][2] if index == 0 else segments[index - 1][3]
        root_prepared = prepared if source_prepared is None else source_prepared
        root_box = _surface_branch_node_box(
            pair_for_curve,
            reference,
            2,
            value,
            policy,
            pair_prepared=root_prepared,
        )
        if root_box is None:
            unresolved_lower, unresolved_upper = (
                (source_lower, source_upper)
                if source_lower is not None and source_upper is not None
                else (reference - 1.0e-6, reference + 1.0e-6)
            )
            return None, (
                UnresolvedParameterRegion(
                    unresolved_lower, unresolved_upper, "resolution"
                ),
            )
        node_boxes.append(root_box)
    charts = list(segments)
    node_shifts = np.zeros((len(node_values), 2, 4), dtype=np.int32)
    source_patches = (curve_first.patch, curve_second.patch)
    for index, (root_lower, root_upper) in enumerate(node_boxes):
        endpoints = (
            (0, segments[index - 1][3]) if index > 0 else None,
            (1, segments[index][2]) if index < len(segments) else None,
        )
        for endpoint_data in endpoints:
            if endpoint_data is None:
                continue
            direction, endpoint = endpoint_data
            shifts = _periodic_endpoint_deck_shift(
                endpoint, root_lower, root_upper, source_patches
            )
            if shifts is None:
                unresolved_lower, unresolved_upper = (
                    (source_lower, source_upper)
                    if source_lower is not None and source_upper is not None
                    else (
                        np.concatenate((first_box[0], second_box[0])),
                        np.concatenate((first_box[1], second_box[1])),
                    )
                )
                return None, (
                    UnresolvedParameterRegion(
                        unresolved_lower, unresolved_upper, "continuation"
                    ),
                )
            node_shifts[index, direction] = shifts
    try:
        curve = IntersectionCurve(
            curve_first,
            curve_second,
            chart_axes=np.full((len(charts),), 2, dtype=np.int32),
            chart_start=np.stack([chart[2] for chart in charts]),
            chart_end=np.stack([chart[3] for chart in charts]),
            chart_lower=np.stack([chart[4] for chart in charts]),
            chart_upper=np.stack([chart[5] for chart in charts]),
            node_axes=np.full((len(node_values),), 2, dtype=np.int32),
            node_values=np.asarray(node_values),
            node_lower=np.stack([box[0] for box in node_boxes]),
            node_upper=np.stack([box[1] for box in node_boxes]),
            node_period_shifts=node_shifts,
            closed=True,
            start_kind="closed",
            end_kind="closed",
        )
    except ValueError:
        unresolved_lower, unresolved_upper = (
            (source_lower, source_upper)
            if source_lower is not None and source_upper is not None
            else (
                np.concatenate((first_box[0], second_box[0])),
                np.concatenate((first_box[1], second_box[1])),
            )
        )
        return None, (
            UnresolvedParameterRegion(unresolved_lower, unresolved_upper, "continuation"),
        )
    return curve, ()


def _orthogonal_equal_cylinder_singular(
    first: SurfaceRegion, second: SurfaceRegion, policy: ParametricIntersectionPolicy, /
) -> SurfaceIntersectionResult | None:
    a, b = first.patch, second.patch
    if not isinstance(a, CylinderPatch) or not isinstance(b, CylinderPatch):
        return None
    frame_a, frame_b = (
        _exact_unit_frame(a.first_axis, a.second_axis, a.axis),
        _exact_unit_frame(b.first_axis, b.second_axis, b.axis),
    )
    if frame_a is None or frame_b is None:
        return None
    _, _, axis_a = frame_a
    _, _, axis_b = frame_b
    if rational_dot(axis_a, axis_b) != 0:
        return None
    normal = (
        axis_a[1] * axis_b[2] - axis_a[2] * axis_b[1],
        axis_a[2] * axis_b[0] - axis_a[0] * axis_b[2],
        axis_a[0] * axis_b[1] - axis_a[1] * axis_b[0],
    )
    if rational_dot(normal, normal) != 1:
        return None
    radius_a, radius_b = (
        Fraction.from_float(float(a.radius)),
        Fraction.from_float(float(b.radius)),
    )
    if radius_a <= 0 or radius_a != radius_b:
        return None
    origin_delta = tuple(
        rational_vector(np.asarray(b.origin))[i]
        - rational_vector(np.asarray(a.origin))[i]
        for i in range(3)
    )
    delta_n, delta_a, delta_b = (
        rational_dot(origin_delta, normal),
        rational_dot(origin_delta, axis_a),
        rational_dot(origin_delta, axis_b),
    )
    if delta_n != 0:
        return None
    n = np.asarray(tuple(float(value) for value in normal), dtype=np.float64)
    origin = rational_vector(np.asarray(a.origin))
    axis = rational_vector(np.asarray(a.axis))
    frame_a1, frame_a2 = np.asarray(a.first_axis), np.asarray(a.second_axis)
    frame_b1, frame_b2 = np.asarray(b.first_axis), np.asarray(b.second_axis)
    events: list[ParametricIntersectionPoint] = []
    unresolved: list[UnresolvedParameterRegion] = []
    for sign in (-1, 1):
        exact = tuple(
            origin[i] + sign * radius_a * normal[i] + delta_a * axis[i] for i in range(3)
        )
        point = np.asarray(tuple(float(value) for value in exact), dtype=np.float64)
        radial = sign * float(radius_a) * n
        u1 = math.atan2(float(radial @ frame_a2), float(radial @ frame_a1))
        u2 = math.atan2(float(radial @ frame_b2), float(radial @ frame_b1))
        uv1 = _region_periodic_angle(u1, first, 0)
        uv2 = _region_periodic_angle(u2, second, 0)
        v1, v2 = float(delta_a), float(-delta_b)
        if uv1 is None or uv2 is None:
            continue
        if not (
            float(np.asarray(first.parameter_box[0, 1]))
            <= v1
            <= float(np.asarray(first.parameter_box[1, 1]))
            and float(np.asarray(second.parameter_box[0, 1]))
            <= v2
            <= float(np.asarray(second.parameter_box[1, 1]))
        ):
            continue
        parameters = np.asarray((uv1, v1, uv2, v2), dtype=np.float64)
        lower = np.asarray(
            (*np.asarray(first.parameter_box[0]), *np.asarray(second.parameter_box[0])),
            dtype=np.float64,
        )
        upper = np.asarray(
            (*np.asarray(first.parameter_box[1]), *np.asarray(second.parameter_box[1])),
            dtype=np.float64,
        )
        root_id = canonical_fingerprint(
            {
                "kind": "orthogonal-cylinder-singular-root",
                "first": first.identity_payload(),
                "second": second.identity_payload(),
                "point": tuple((value.numerator, value.denominator) for value in exact),
            }
        )
        events.append(
            ParametricIntersectionPoint(
                kind="singular_point",
                certificate="analytic-contact",
                parameters=parameters,
                parameter_lower=lower,
                parameter_upper=upper,
                point=point,
                point_exact=exact,
                gap_bound=0.0,
                condition_estimate=float("inf"),
                branch_count=4,
                root_id=root_id,
            )
        )
        radius = policy.relative_junction * np.maximum(upper - lower, 1.0)
        unresolved.append(
            UnresolvedParameterRegion(
                np.maximum(lower, parameters - radius),
                np.minimum(upper, parameters + radius),
                "singular",
            )
        )
    return (
        None
        if not events
        else SurfaceIntersectionResult(points=events, unresolved=unresolved)
    )


def _region_periodic_angle(
    value: float, region: SurfaceRegion, axis: int, /
) -> float | None:
    patch = region.patch
    symbol = patch.native_period_symbols()[axis]
    period = patch.periods[axis]
    lower = float(np.asarray(region.parameter_box[0, axis]))
    upper = float(np.asarray(region.parameter_box[1, axis]))
    if symbol == "two_pi" and period is not None and upper - lower == float(period):
        while value < lower:
            value += float(period)
        while value > upper:
            value -= float(period)
    return value if lower <= value <= upper else None


def _swap_intersection_curve_order(
    curve: IntersectionCurve, first: SurfaceRegion, second: SurfaceRegion, /
) -> IntersectionCurve:
    columns = np.asarray((2, 3, 0, 1), dtype=np.int64)
    axis_map = (2, 3, 0, 1)
    return IntersectionCurve(
        first,
        second,
        chart_axes=np.asarray(
            [axis_map[int(axis)] for axis in np.asarray(curve.chart_axes)], dtype=np.int32
        ),
        chart_start=np.asarray(curve.chart_start)[:, columns],
        chart_end=np.asarray(curve.chart_end)[:, columns],
        chart_lower=np.asarray(curve.chart_lower)[:, columns],
        chart_upper=np.asarray(curve.chart_upper)[:, columns],
        node_axes=np.asarray(
            [axis_map[int(axis)] for axis in np.asarray(curve.node_axes)], dtype=np.int32
        ),
        node_values=curve.node_values,
        node_lower=np.asarray(curve.node_lower)[:, columns],
        node_upper=np.asarray(curve.node_upper)[:, columns],
        node_period_shifts=np.asarray(curve.node_period_shifts)[:, :, columns],
        closed=curve.closed,
        start_kind=curve.start_kind,
        end_kind=curve.end_kind,
    )


def _orthogonal_cylinder_intersection(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
    *,
    source_regions: tuple[SurfaceRegion, SurfaceRegion] | None = None,
    source_slopes: tuple[Fraction, Fraction] | None = None,
) -> SurfaceIntersectionResult | None:
    if (source_regions is None) != (source_slopes is None):
        return None
    if not isinstance(first.patch, CylinderPatch) or not isinstance(
        second.patch, CylinderPatch
    ):
        return None
    singular = _orthogonal_equal_cylinder_singular(first, second, policy)
    if singular is not None:
        if source_regions is None:
            return singular
        return _unresolved_surface(source_regions[0], source_regions[1], "continuation")
    from ._intersection_curve import _prepare_pair

    pair_preparations: dict[
        tuple[int, int, int, int],
        tuple[
            SurfacePairSystem,
            tuple[SurfaceRegion, SurfaceRegion] | None,
            KrawczykEvaluator,
        ],
    ] = {}

    def prepare_pair(
        system: SurfacePairSystem,
        regions: tuple[SurfaceRegion, SurfaceRegion] | None,
        /,
    ) -> KrawczykEvaluator:
        first_region, second_region = (
            (0, 0) if regions is None else (id(regions[0]), id(regions[1]))
        )
        key = (
            id(system.first),
            id(system.second),
            first_region,
            second_region,
        )
        cached = pair_preparations.get(key)
        if cached is not None:
            cached_system, cached_regions, cached_prepared = cached
            if (
                cached_system.first is system.first
                and cached_system.second is system.second
                and (
                    (cached_regions is None and regions is None)
                    or (
                        cached_regions is not None
                        and regions is not None
                        and cached_regions[0] is regions[0]
                        and cached_regions[1] is regions[1]
                    )
                )
            ):
                return cached_prepared
        prepared = (
            _prepare_pair(system)
            if regions is None
            else _prepare_pair(system, source_regions=regions)
        )
        pair_preparations[key] = (system, regions, prepared)
        return prepared

    direct_curves = []
    direct_unresolved = []
    for sign in (-1, 1):
        curve, pending = _orthogonal_cylinder_branch(
            first,
            second,
            sign,
            policy,
            source_regions=source_regions,
            source_slopes=source_slopes,
            pair_preparation=prepare_pair,
        )
        if curve is not None:
            direct_curves.append(curve)
        direct_unresolved.extend(pending)
    if len(direct_curves) == 2 and not direct_unresolved:
        return SurfaceIntersectionResult(curves=direct_curves)
    reverse_curves = []
    reverse_unresolved = []
    reverse_regions = (
        None if source_regions is None else (source_regions[1], source_regions[0])
    )
    reverse_slopes = (
        None if source_slopes is None else (source_slopes[1], source_slopes[0])
    )
    for sign in (-1, 1):
        curve, pending = _orthogonal_cylinder_branch(
            second,
            first,
            sign,
            policy,
            source_regions=reverse_regions,
            source_slopes=reverse_slopes,
            pair_preparation=prepare_pair,
        )
        if curve is not None:
            reverse_curves.append(curve)
        reverse_unresolved.extend(pending)
    if len(reverse_curves) == 2 and not reverse_unresolved:
        source_first, source_second = (
            (first, second) if source_regions is None else source_regions
        )
        return SurfaceIntersectionResult(
            curves=tuple(
                _swap_intersection_curve_order(curve, source_first, source_second)
                for curve in reverse_curves
            )
        )
    unresolved_first, unresolved_second = (
        (first, second) if source_regions is None else source_regions
    )
    return _unresolved_surface(unresolved_first, unresolved_second, "continuation")


def _plane_uv_affine_exact(
    patch: PlanePatch,
    origin: tuple[ExactScalar, ...],
    direction: tuple[ExactScalar, ...],
    /,
) -> tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]] | None:
    first = rational_vector(np.asarray(patch.first_axis))
    second = rational_vector(np.asarray(patch.second_axis))
    if len(origin) != 3 or len(direction) != 3:
        return None

    def exact_dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    g00, g01, g11 = (
        rational_dot(first, first),
        rational_dot(first, second),
        rational_dot(second, second),
    )
    determinant = g00 * g11 - g01 * g01
    if determinant == 0:
        return None
    patch_origin = rational_vector(np.asarray(patch.origin))
    delta = tuple(origin[index] - patch_origin[index] for index in range(3))
    values = []
    for vector in (delta, direction):
        rhs0, rhs1 = exact_dot(first, vector), exact_dot(second, vector)
        values.append(
            (
                (g11 * rhs0 - g01 * rhs1) / determinant,
                (g00 * rhs1 - g01 * rhs0) / determinant,
            )
        )
    return (values[0][0], values[0][1]), (values[1][0], values[1][1])


def _plane_uv_affine(
    patch: PlanePatch, origin: tuple[Fraction, ...], direction: tuple[Fraction, ...], /
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] | None:
    values = _plane_uv_affine_exact(patch, origin, direction)
    if values is None or any(
        not isinstance(value, Fraction) for pair in values for value in pair
    ):
        return None
    return cast(tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], values)


def _plane_plane_intersection(
    first: SurfaceRegion, second: SurfaceRegion, policy: ParametricIntersectionPolicy, /
) -> SurfaceIntersectionResult | None:
    from ...linalg._exact import solve_exact
    from ._intersection_curve import _NodeSystem, _PreparedNode, _PreparedPair

    a, b = first.patch, second.patch
    if not isinstance(a, PlanePatch) or not isinstance(b, PlanePatch):
        return None
    a1, a2 = (
        rational_vector(np.asarray(a.first_axis)),
        rational_vector(np.asarray(a.second_axis)),
    )
    b1, b2 = (
        rational_vector(np.asarray(b.first_axis)),
        rational_vector(np.asarray(b.second_axis)),
    )
    normal_a = (
        a1[1] * a2[2] - a1[2] * a2[1],
        a1[2] * a2[0] - a1[0] * a2[2],
        a1[0] * a2[1] - a1[1] * a2[0],
    )
    normal_b = (
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    )
    direction = (
        normal_a[1] * normal_b[2] - normal_a[2] * normal_b[1],
        normal_a[2] * normal_b[0] - normal_a[0] * normal_b[2],
        normal_a[0] * normal_b[1] - normal_a[1] * normal_b[0],
    )
    if all(value == 0 for value in direction):
        return SurfaceIntersectionResult()
    origin_a, origin_b = (
        rational_vector(np.asarray(a.origin)),
        rational_vector(np.asarray(b.origin)),
    )
    rhs = (
        rational_dot(normal_a, origin_a),
        rational_dot(normal_b, origin_b),
        Fraction(0),
    )
    try:
        point0 = tuple(
            row[0]
            for row in solve_exact(
                (normal_a, normal_b, direction), tuple((value,) for value in rhs)
            )
        )
    except ValueError:
        return None
    map_a = _plane_uv_affine(a, point0, direction)
    map_b = _plane_uv_affine(b, point0, direction)
    if map_a is None or map_b is None:
        return None
    map_a_source: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = map_a
    map_b_source: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = map_b
    first_bounds = np.asarray(first.parameter_box)
    second_bounds = np.asarray(second.parameter_box)
    interval: tuple[Fraction | None, Fraction | None] = (None, None)
    for mapping, box in ((map_a, first_bounds), (map_b, second_bounds)):
        offset, slope = mapping
        for axis in range(2):
            low, high = (
                Fraction.from_float(float(box[0, axis])),
                Fraction.from_float(float(box[1, axis])),
            )
            value, rate = offset[axis], slope[axis]
            if rate == 0:
                if not low <= value <= high:
                    return SurfaceIntersectionResult()
                continue
            ta, tb = (low - value) / rate, (high - value) / rate
            local_low, local_high = min(ta, tb), max(ta, tb)
            interval = (
                local_low if interval[0] is None else max(interval[0], local_low),
                local_high if interval[1] is None else min(interval[1], local_high),
            )
            if (
                interval[0] is not None
                and interval[1] is not None
                and interval[0] > interval[1]
            ):
                return SurfaceIntersectionResult()
    t_lower, t_upper = interval
    if t_lower is None or t_upper is None:
        return _unresolved_surface(first, second, "continuation")

    def coupled(t: Fraction, /) -> tuple[Fraction, ...]:
        uv_a = tuple(
            map_a_source[0][axis] + t * map_a_source[1][axis] for axis in range(2)
        )
        uv_b = tuple(
            map_b_source[0][axis] + t * map_b_source[1][axis] for axis in range(2)
        )
        return (*uv_a, *uv_b)

    def world(t: Fraction, /) -> tuple[Fraction, ...]:
        return tuple(point0[i] + t * direction[i] for i in range(3))

    if t_lower == t_upper:
        params = coupled(t_lower)
        point_exact = world(t_lower)
        parameters = np.asarray(tuple(float(value) for value in params))
        point = np.asarray(tuple(float(value) for value in point_exact))
        root_id = canonical_fingerprint(
            {
                "kind": "plane-plane-boundary-root",
                "first": first.identity_payload(),
                "second": second.identity_payload(),
                "parameters": tuple((v.numerator, v.denominator) for v in params),
            }
        )
        event = ParametricIntersectionPoint(
            kind="crossing",
            certificate="analytic-contact",
            parameters=parameters,
            parameter_lower=parameters,
            parameter_upper=parameters,
            point=point,
            point_exact=point_exact,
            gap_bound=0.0,
            condition_estimate=1.0,
            root_id=root_id,
        )
        return SurfaceIntersectionResult(points=(event,))
    domain_width = np.concatenate(
        (first_bounds[1] - first_bounds[0], second_bounds[1] - second_bounds[0])
    )
    slopes = tuple(float(value) for value in (*map_a[1], *map_b[1]))
    axis = int(
        np.argmax(
            np.abs(np.asarray(slopes)) / np.maximum(domain_width, np.finfo(float).tiny)
        )
    )
    if slopes[axis] == 0.0:
        return _unresolved_surface(first, second, "continuation")
    pair = SurfacePairSystem(a, b)
    prepared = _PreparedPair(pair)
    pending = [(t_lower, t_upper)]
    segments = []
    unresolved = []
    while pending:
        low_t, high_t = pending.pop()
        ql, qh = coupled(low_t), coupled(high_t)
        qlf, qhf = (
            np.asarray(tuple(float(value) for value in ql)),
            np.asarray(tuple(float(value) for value in qh)),
        )
        chart_low, chart_high = np.minimum(qlf, qhf), np.maximum(qlf, qhf)
        free = np.asarray([index for index in range(4) if index != axis], dtype=np.int64)
        pad = (
            128
            * np.finfo(float).eps
            * (1 + np.maximum(np.abs(chart_low[free]), np.abs(chart_high[free])))
        )
        chart_low[free] -= pad
        chart_high[free] += pad
        chart_low[axis], chart_high[axis] = (
            min(qlf[axis], qhf[axis]),
            max(qlf[axis], qhf[axis]),
        )
        witness = krawczyk(
            prepared, chart_low[None], chart_high[None], parameter_axis=axis
        )
        if bool(witness.certified[0]):
            segments.append((low_t, high_t, qlf, qhf, chart_low, chart_high))
            continue
        middle = low_t + (high_t - low_t) / 2
        threshold = Fraction.from_float(
            policy.relative_resolution * max(1.0, float(high_t - low_t))
        )
        if high_t - low_t <= threshold or not low_t < middle < high_t:
            unresolved.append(
                UnresolvedParameterRegion(chart_low, chart_high, "resolution")
            )
            continue
        pending.append((middle, high_t))
        pending.append((low_t, middle))
    if unresolved:
        return SurfaceIntersectionResult(unresolved=unresolved)
    segments.sort(key=lambda value: value[0])
    node_ts = [segments[0][0]] + [segment[1] for segment in segments]
    node_values = np.asarray([float(coupled(value)[axis]) for value in node_ts])
    node_boxes = []
    for value, q_exact in zip(node_ts, (coupled(t) for t in node_ts), strict=True):
        q = np.asarray(tuple(float(component) for component in q_exact))
        system = _NodeSystem(pair, axis, float(q[axis]))
        isolated = _PreparedNode(system)
        radius = 128 * np.finfo(float).eps * (1 + float(np.max(np.abs(q))))
        upper_radius = policy.relative_probe * (1 + float(np.max(np.abs(q))))
        root_box = None
        while radius <= upper_radius:
            low, high = q - radius, q + radius
            proof = krawczyk(isolated, low[None], high[None])
            if bool(proof.certified[0]):
                root_box = (low, high)
                break
            radius *= 2.0
        if root_box is None:
            return _unresolved_surface(first, second, "resolution")
        node_boxes.append(root_box)
    charts = segments
    curve = IntersectionCurve(
        first,
        second,
        chart_axes=np.full((len(charts),), axis, dtype=np.int32),
        chart_start=np.stack([item[2] for item in charts]),
        chart_end=np.stack([item[3] for item in charts]),
        chart_lower=np.stack([item[4] for item in charts]),
        chart_upper=np.stack([item[5] for item in charts]),
        node_axes=np.full((len(node_ts),), axis, dtype=np.int32),
        node_values=node_values,
        node_lower=np.stack([item[0] for item in node_boxes]),
        node_upper=np.stack([item[1] for item in node_boxes]),
        node_period_shifts=np.zeros((len(node_ts), 2, 4), dtype=np.int32),
        closed=False,
        start_kind="boundary",
        end_kind="boundary",
    )
    events = []
    for t, box in ((node_ts[0], node_boxes[0]), (node_ts[-1], node_boxes[-1])):
        params = coupled(t)
        point_exact = world(t)
        parameters = np.asarray(tuple(float(value) for value in params))
        point = np.asarray(tuple(float(value) for value in point_exact))
        root_id = canonical_fingerprint(
            {
                "kind": "plane-plane-boundary-root",
                "first": first.identity_payload(),
                "second": second.identity_payload(),
                "parameters": tuple((v.numerator, v.denominator) for v in params),
            }
        )
        events.append(
            ParametricIntersectionPoint(
                kind="crossing",
                certificate="analytic-contact",
                parameters=parameters,
                parameter_lower=box[0],
                parameter_upper=box[1],
                point=point,
                point_exact=point_exact,
                gap_bound=0.0,
                condition_estimate=1.0,
                root_id=root_id,
            )
        )
    return SurfaceIntersectionResult(curves=(curve,), points=events)


def _angular_surface_branch(
    first: SurfaceRegion,
    second: SurfaceRegion,
    chart_axis: int,
    theta_lower: float,
    theta_upper: float,
    policy: ParametricIntersectionPolicy,
    point_at: Callable[[float], np.ndarray],
    box_for: Callable[[float, float], tuple[np.ndarray, np.ndarray]],
    /,
    *,
    closed: bool = True,
    start_kind: IntersectionEndpointKind = "closed",
    end_kind: IntersectionEndpointKind = "closed",
    end_nodes: tuple[tuple[int, float], tuple[int, float]] | None = None,
    seam: tuple[float, float] | None = None,
) -> IntersectionCurve | None:
    """Build an angular source branch with certified chart and node roots.

    ``end_nodes`` fixes the first and last node on another source coordinate
    (the trim edge a boundary branch ends on) instead of the angle. Chart boxes
    of such a branch may then reach past the trimmed region near its ends: the
    branch itself stays inside, and each box only isolates the branch graph.

    ``seam`` is the angle window ``(lower, upper)`` of a branch that runs from
    ``theta_lower`` up to the window's upper end and on from its lower end to
    ``theta_upper``. The crossing is one node fixing the angle at the exact
    lower window value; the charts before it see it through one exact deck.
    """
    from ._intersection_curve import _PreparedPair

    pair = SurfacePairSystem(first.patch, second.patch)
    prepared = _PreparedPair(pair)
    region_lower = np.concatenate(
        (np.asarray(first.parameter_box[0]), np.asarray(second.parameter_box[0]))
    )
    region_upper = np.concatenate(
        (np.asarray(first.parameter_box[1]), np.asarray(second.parameter_box[1]))
    )
    free = np.asarray([axis for axis in range(4) if axis != chart_axis], dtype=np.int64)
    pieces = (
        ((theta_lower, theta_upper),)
        if seam is None
        else ((theta_lower, seam[1]), (seam[0], theta_upper))
    )
    resolution = policy.relative_resolution * max(
        1.0, sum(upper - lower for lower, upper in pieces)
    )
    trimmed = end_nodes is not None

    def subdivide(
        piece_lower: float, piece_upper: float, /
    ) -> list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None:
        breakpoints = [
            piece_lower + (piece_upper - piece_lower) * index / 4.0 for index in range(5)
        ]
        pending = [(breakpoints[index], breakpoints[index + 1]) for index in range(4)]
        found = []
        while pending:
            lower_angle, upper_angle = pending.pop()
            lower, upper = box_for(lower_angle, upper_angle)
            if (
                lower[chart_axis] < region_lower[chart_axis]
                or upper[chart_axis] > region_upper[chart_axis]
            ):
                return None
            if not trimmed and (
                np.any(lower[free] < region_lower[free])
                or np.any(upper[free] > region_upper[free])
            ):
                return None
            desired = np.maximum(
                0.25 * (upper[free] - lower[free]),
                128.0
                * np.finfo(np.float64).eps
                * (1.0 + np.maximum(np.abs(lower[free]), np.abs(upper[free]))),
            )
            room = np.minimum(
                lower[free] - region_lower[free],
                region_upper[free] - upper[free],
            )
            padding = desired if trimmed else np.minimum(desired, 0.5 * room)
            lower[free] -= padding
            upper[free] += padding
            witness = krawczyk(
                prepared,
                lower[None],
                upper[None],
                parameter_axis=chart_axis,
            )
            if bool(witness.certified[0]):
                found.append(
                    (
                        lower_angle,
                        upper_angle,
                        point_at(lower_angle),
                        point_at(upper_angle),
                        lower,
                        upper,
                    )
                )
                continue
            middle = lower_angle + 0.5 * (upper_angle - lower_angle)
            if (
                upper_angle - lower_angle <= resolution
                or not lower_angle < middle < upper_angle
            ):
                return None
            pending.append((middle, upper_angle))
            pending.append((lower_angle, middle))
        found.sort(key=lambda segment: segment[0])
        return found

    segments: list[
        tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for piece_lower, piece_upper in pieces:
        piece_segments = subdivide(piece_lower, piece_upper)
        if piece_segments is None:
            return None
        segments.extend(piece_segments)
    # One node per chart boundary; the seam crossing's two chart ends share it.
    node_angles = [segments[0][0]] + [segment[1] for segment in segments]
    if seam is not None:
        seam_node = next(
            index for index, angle in enumerate(node_angles) if angle == seam[1]
        )
        node_angles[seam_node] = seam[0]
    node_boxes = []
    for index, angle in enumerate(node_angles):
        node_axis, node_value = chart_axis, angle
        if end_nodes is not None and index in (0, len(node_angles) - 1):
            node_axis, node_value = end_nodes[0 if index == 0 else 1]
        isolated = _surface_branch_node_box(
            pair, point_at(angle), node_axis, node_value, policy
        )
        if isolated is None:
            return None
        node_boxes.append(isolated)
    node_shifts = np.zeros((len(node_angles), 2, 4), dtype=np.int32)
    if seam is not None:
        patches = first.patch, second.patch
        for index, (lower, upper) in enumerate(node_boxes):
            for direction, point in (
                (0, segments[index - 1][3] if index > 0 else None),
                (1, segments[index][2] if index < len(segments) else None),
            ):
                if point is None:
                    continue
                shifts = _periodic_endpoint_deck_shift(point, lower, upper, patches)
                if shifts is None:
                    return None
                node_shifts[index, direction] = shifts
    node_axes = np.full((len(node_angles),), chart_axis, dtype=np.int32)
    node_values = np.asarray(node_angles, dtype=np.float64)
    if end_nodes is not None:
        (node_axes[0], node_values[0]), (node_axes[-1], node_values[-1]) = end_nodes
    try:
        return IntersectionCurve(
            first,
            second,
            chart_axes=np.full((len(segments),), chart_axis, dtype=np.int32),
            chart_start=np.stack([segment[2] for segment in segments]),
            chart_end=np.stack([segment[3] for segment in segments]),
            chart_lower=np.stack([segment[4] for segment in segments]),
            chart_upper=np.stack([segment[5] for segment in segments]),
            node_axes=node_axes,
            node_values=node_values,
            node_lower=np.stack([bounds[0] for bounds in node_boxes]),
            node_upper=np.stack([bounds[1] for bounds in node_boxes]),
            node_period_shifts=node_shifts,
            closed=closed,
            start_kind=start_kind,
            end_kind=end_kind,
        )
    except ValueError:
        return None


def _parallel_plane_sphere_circle(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    """Certify a complete regular circle when a plane is normal to a sphere axis."""
    if isinstance(first.patch, PlanePatch) and isinstance(second.patch, SpherePatch):
        plane, sphere, plane_region, sphere_region = (
            first.patch,
            second.patch,
            first,
            second,
        )
        plane_first = True
    elif isinstance(first.patch, SpherePatch) and isinstance(second.patch, PlanePatch):
        plane, sphere, plane_region, sphere_region = (
            second.patch,
            first.patch,
            second,
            first,
        )
        plane_first = False
    else:
        return None
    if not _sphere_frame_valid(sphere):
        return None

    from .._interval_enclosure import (
        interval_add,
        interval_cos,
        interval_multiply,
        interval_sin,
    )
    from .._rational import fraction_down, fraction_up
    from ._exact import square_root_interval

    plane_first_axis = rational_vector(np.asarray(plane.first_axis))
    plane_second_axis = rational_vector(np.asarray(plane.second_axis))
    plane_normal = (
        plane_first_axis[1] * plane_second_axis[2]
        - plane_first_axis[2] * plane_second_axis[1],
        plane_first_axis[2] * plane_second_axis[0]
        - plane_first_axis[0] * plane_second_axis[2],
        plane_first_axis[0] * plane_second_axis[1]
        - plane_first_axis[1] * plane_second_axis[0],
    )
    sphere_axis = rational_vector(np.asarray(sphere.axis))
    if all(value == 0 for value in plane_normal) or any(
        plane_normal[i] * sphere_axis[j] != plane_normal[j] * sphere_axis[i]
        for i in range(3)
        for j in range(3)
    ):
        return None
    sphere_center = rational_vector(np.asarray(sphere.center))
    plane_origin = rational_vector(np.asarray(plane.origin))
    height = rational_dot(
        tuple(plane_origin[i] - sphere_center[i] for i in range(3)), sphere_axis
    )
    radius = Fraction.from_float(float(sphere.radius))
    if radius <= 0:
        return None
    squared_radius = radius * radius - height * height
    if squared_radius < 0:
        return SurfaceIntersectionResult()
    if squared_radius == 0:
        contact = _plane_sphere_contact(first, second, policy)
        return (
            SurfaceIntersectionResult(points=(contact,))
            if contact is not None
            else SurfaceIntersectionResult()
        )

    theta_lower, theta_upper = (
        float(value) for value in np.asarray(sphere_region.parameter_box)[:, 0]
    )
    period = sphere.periods[0]
    if (
        sphere.native_period_symbols()[0] != "two_pi"
        or period is None
        or Fraction.from_float(theta_upper) - Fraction.from_float(theta_lower)
        != Fraction.from_float(float(period))
    ):
        return _unresolved_surface(first, second, "continuation")

    circle_center = tuple(sphere_center[i] + height * sphere_axis[i] for i in range(3))
    first_map = _plane_uv_affine(
        plane, circle_center, rational_vector(np.asarray(sphere.first_axis))
    )
    second_map = _plane_uv_affine(
        plane, circle_center, rational_vector(np.asarray(sphere.second_axis))
    )
    if first_map is None or second_map is None:
        return None
    uv_center, uv_cosine = first_map
    _, uv_sine = second_map
    radius_lower, radius_upper = square_root_interval(squared_radius)
    radius_interval = np.asarray(radius_lower), np.asarray(radius_upper)
    plane_box = np.asarray(plane_region.parameter_box, dtype=np.float64)
    for axis in range(2):
        amplitude_squared = squared_radius * (
            uv_cosine[axis] * uv_cosine[axis] + uv_sine[axis] * uv_sine[axis]
        )
        amplitude_lower, amplitude_upper = square_root_interval(amplitude_squared)
        center_bounds = (
            np.asarray(fraction_down(uv_center[axis])),
            np.asarray(fraction_up(uv_center[axis])),
        )
        lower_bounds = interval_add(
            center_bounds,
            (np.asarray(-amplitude_upper), np.asarray(-amplitude_lower)),
        )
        upper_bounds = interval_add(
            center_bounds,
            (np.asarray(amplitude_lower), np.asarray(amplitude_upper)),
        )
        lower_value, upper_value = float(lower_bounds[0]), float(upper_bounds[1])
        if upper_value < plane_box[0, axis] or lower_value > plane_box[1, axis]:
            return SurfaceIntersectionResult()
        if lower_value < plane_box[0, axis] or upper_value > plane_box[1, axis]:
            return _unresolved_surface(first, second, "continuation")

    latitude = math.asin(float(height / radius))
    latitude_domain = np.asarray(sphere_region.parameter_box, dtype=np.float64)[:, 1]
    chart_axis = 2 if plane_first else 0
    radius_midpoint = 0.5 * radius_interval[0] + 0.5 * radius_interval[1]

    def coupled(angle: float, /) -> np.ndarray:
        cosine, sine = math.cos(angle), math.sin(angle)
        plane_uv = tuple(
            float(uv_center[index])
            + radius_midpoint
            * (float(uv_cosine[index]) * cosine + float(uv_sine[index]) * sine)
            for index in range(2)
        )
        sphere_uv = (angle, latitude)
        return np.asarray(
            (*plane_uv, *sphere_uv) if plane_first else (*sphere_uv, *plane_uv),
            dtype=np.float64,
        )

    def chart_box(
        lower_angle: float,
        upper_angle: float,
        lower_latitude: float,
        upper_latitude: float,
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        cosine = interval_cos((np.asarray(lower_angle), np.asarray(upper_angle)))
        sine = interval_sin((np.asarray(lower_angle), np.asarray(upper_angle)))
        plane_lower = np.empty((2,), dtype=np.float64)
        plane_upper = np.empty((2,), dtype=np.float64)
        for index in range(2):
            center_bounds = (
                np.asarray(fraction_down(uv_center[index])),
                np.asarray(fraction_up(uv_center[index])),
            )
            cosine_term = interval_multiply(
                (
                    np.asarray(fraction_down(uv_cosine[index])),
                    np.asarray(fraction_up(uv_cosine[index])),
                ),
                interval_multiply(radius_interval, cosine),
            )
            sine_term = interval_multiply(
                (
                    np.asarray(fraction_down(uv_sine[index])),
                    np.asarray(fraction_up(uv_sine[index])),
                ),
                interval_multiply(radius_interval, sine),
            )
            bounds = interval_add(interval_add(center_bounds, cosine_term), sine_term)
            plane_lower[index], plane_upper[index] = float(bounds[0]), float(bounds[1])
        if plane_first:
            lower = np.asarray(
                (*plane_lower, lower_angle, lower_latitude), dtype=np.float64
            )
            upper = np.asarray(
                (*plane_upper, upper_angle, upper_latitude), dtype=np.float64
            )
        else:
            lower = np.asarray(
                (lower_angle, lower_latitude, *plane_lower), dtype=np.float64
            )
            upper = np.asarray(
                (upper_angle, upper_latitude, *plane_upper), dtype=np.float64
            )
        return lower, upper

    def build_curve(
        lower_latitude: float, upper_latitude: float, /
    ) -> IntersectionCurve | None:
        if not lower_latitude <= latitude <= upper_latitude:
            return None
        return _angular_surface_branch(
            first,
            second,
            chart_axis,
            theta_lower,
            theta_upper,
            policy,
            coupled,
            lambda lower, upper: chart_box(lower, upper, lower_latitude, upper_latitude),
        )

    initial_pad = max(
        policy.relative_resolution * max(1.0, abs(latitude)),
        64.0 * np.finfo(np.float64).eps * max(1.0, abs(latitude)),
    )
    maximum_pad = max(
        abs(latitude - float(latitude_domain[0])),
        abs(float(latitude_domain[1]) - latitude),
    )
    pad = initial_pad
    while True:
        lower_latitude = max(float(latitude_domain[0]), latitude - pad)
        upper_latitude = min(float(latitude_domain[1]), latitude + pad)
        curve = build_curve(lower_latitude, upper_latitude)
        if curve is not None:
            return SurfaceIntersectionResult(curves=(curve,))
        if pad >= maximum_pad:
            return _unresolved_surface(first, second, "resolution")
        next_pad = min(2.0 * pad, maximum_pad)
        if next_pad <= pad:
            return _unresolved_surface(first, second, "resolution")
        pad = next_pad


@dataclass(frozen=True, slots=True, eq=False)
class _PlaneCylinderAngularData:
    first: SurfaceRegion
    second: SurfaceRegion
    plane: PlanePatch
    cylinder: CylinderPatch
    plane_region: SurfaceRegion
    cylinder_region: SurfaceRegion
    plane_first: bool
    cylinder_origin: tuple[ExactScalar, ExactScalar, ExactScalar]
    cylinder_axis: tuple[ExactScalar, ExactScalar, ExactScalar]
    radial_first: tuple[ExactScalar, ExactScalar, ExactScalar]
    radial_second: tuple[ExactScalar, ExactScalar, ExactScalar]
    height_center: ExactScalar
    height_cosine: ExactScalar
    height_sine: ExactScalar
    uv_center: tuple[ExactScalar, ExactScalar]
    uv_cosine: tuple[ExactScalar, ExactScalar]
    uv_sine: tuple[ExactScalar, ExactScalar]
    theta_lower: float
    theta_upper: float
    closed: bool


def _prepare_plane_cylinder_angular_data(
    first: SurfaceRegion, second: SurfaceRegion, /
) -> _PlaneCylinderAngularData | None:
    if isinstance(first.patch, PlanePatch) and isinstance(second.patch, CylinderPatch):
        plane, cylinder, plane_region, cylinder_region = (
            first.patch,
            second.patch,
            first,
            second,
        )
        plane_first = True
    elif isinstance(first.patch, CylinderPatch) and isinstance(second.patch, PlanePatch):
        plane, cylinder, plane_region, cylinder_region = (
            second.patch,
            first.patch,
            second,
            first,
        )
        plane_first = False
    else:
        return None

    frame = _exact_unit_frame_values(
        cylinder.first_axis_exact, cylinder.second_axis_exact, cylinder.axis_exact
    )
    if frame is None:
        return None
    radial_first, radial_second, cylinder_axis = frame
    plane_first_axis = rational_vector(np.asarray(plane.first_axis))
    plane_second_axis = rational_vector(np.asarray(plane.second_axis))
    plane_normal = (
        plane_first_axis[1] * plane_second_axis[2]
        - plane_first_axis[2] * plane_second_axis[1],
        plane_first_axis[2] * plane_second_axis[0]
        - plane_first_axis[0] * plane_second_axis[2],
        plane_first_axis[0] * plane_second_axis[1]
        - plane_first_axis[1] * plane_second_axis[0],
    )

    def exact_dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    denominator = exact_dot(plane_normal, cylinder_axis)
    radius = cylinder.radius_exact
    if denominator == 0 or radius <= 0:
        return None
    cylinder_origin = (
        cylinder.origin_exact[0],
        cylinder.origin_exact[1],
        cylinder.origin_exact[2],
    )
    plane_origin = rational_vector(np.asarray(plane.origin))
    height_center = (
        exact_dot(
            plane_normal,
            tuple(plane_origin[index] - cylinder_origin[index] for index in range(3)),
        )
        / denominator
    )
    height_cosine = -radius * exact_dot(plane_normal, radial_first) / denominator
    height_sine = -radius * exact_dot(plane_normal, radial_second) / denominator
    world_center = tuple(
        cylinder_origin[index] + height_center * cylinder_axis[index]
        for index in range(3)
    )
    world_cosine = tuple(
        radius * radial_first[index] + height_cosine * cylinder_axis[index]
        for index in range(3)
    )
    world_sine = tuple(
        radius * radial_second[index] + height_sine * cylinder_axis[index]
        for index in range(3)
    )
    center_map = _plane_uv_affine_exact(plane, world_center, world_cosine)
    sine_map = _plane_uv_affine_exact(plane, world_center, world_sine)
    if center_map is None or sine_map is None:
        return None
    theta_lower, theta_upper = (
        float(value) for value in np.asarray(cylinder_region.parameter_box)[:, 0]
    )
    period = cylinder.periods[0]
    if cylinder.native_period_symbols()[0] != "two_pi" or period is None:
        return None
    angle_span = Fraction.from_float(theta_upper) - Fraction.from_float(theta_lower)
    period_span = Fraction.from_float(float(period))
    if angle_span <= 0 or angle_span > period_span:
        return None
    return _PlaneCylinderAngularData(
        first,
        second,
        plane,
        cylinder,
        plane_region,
        cylinder_region,
        plane_first,
        cylinder_origin,
        cylinder_axis,
        radial_first,
        radial_second,
        height_center,
        height_cosine,
        height_sine,
        center_map[0],
        center_map[1],
        sine_map[1],
        theta_lower,
        theta_upper,
        angle_span == period_span,
    )


@dataclass(frozen=True, slots=True, eq=False)
class PlaneCylinderConicSection:
    """Exact conic and coupled p-curves for a trimmed plane/cylinder section."""

    edge_curve: EllipseCurve
    plane_pcurve: PlaneCylinderSectionPCurve
    cylinder_pcurve: PlaneCylinderSectionPCurve
    parameter_range: tuple[float, float]
    endpoint_roots: tuple[ConicParameterRoot, ConicParameterRoot]


def exact_plane_cylinder_conic_section(
    first: SurfaceRegion,
    second: SurfaceRegion,
    cylinder_angle_roots: tuple[ConicParameterRoot, ConicParameterRoot],
    /,
) -> PlaneCylinderConicSection | None:
    """Build exact ellipse carriers from exact cylinder-angle endpoint roots.

    The supplied roots must name the two endpoints of the cylinder region's
    native angular interval.  Their directions and turns, rather than rounded
    endpoint coordinates, define the resulting conic trim.
    """

    data = _prepare_plane_cylinder_angular_data(first, second)
    if (
        data is None
        or data.closed
        or not isinstance(cylinder_angle_roots, tuple)
        or len(cylinder_angle_roots) != 2
        or any(not isinstance(root, ConicParameterRoot) for root in cylinder_angle_roots)
    ):
        return None
    expected_circle = (
        (Fraction(0), Fraction(0), Fraction(0)),
        data.radial_first,
        data.radial_second,
        Fraction(1),
    )
    for root, parameter in zip(
        cylinder_angle_roots, (data.theta_lower, data.theta_upper), strict=True
    ):
        source = root.conic
        if (
            not isinstance(source, CircleCurve)
            or (
                source.center_exact,
                source.first_axis_exact,
                source.second_axis_exact,
                source.radius_exact,
            )
            != expected_circle
            or source.native_period_symbol() != "two_pi"
            or root.parameter != parameter
        ):
            return None

    height_squared = (
        data.height_cosine * data.height_cosine + data.height_sine * data.height_sine
    )
    radius_squared = data.cylinder.radius_exact * data.cylinder.radius_exact
    try:
        height_squared_rational = rational_coefficient(
            height_squared, role="plane-cylinder height harmonic norm"
        )
        radius_squared_rational = rational_coefficient(
            radius_squared, role="plane-cylinder radius squared"
        )
    except (TypeError, ValueError):
        return None
    if height_squared_rational < 0 or radius_squared_rational <= 0:
        return None
    phase_cosine: ExactScalar
    phase_sine: ExactScalar
    phase_root: ConicParameterRoot | None
    if height_squared_rational == 0:
        phase_cosine, phase_sine = Fraction(1), Fraction(0)
        height_amplitude: ExactScalar = Fraction(0)
        phase_root = None
    else:
        height_amplitude = RadicalElement.sqrt(height_squared_rational)
        phase_cosine = data.height_cosine / height_amplitude
        phase_sine = data.height_sine / height_amplitude
        unit_circle = CircleCurve(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            1.0,
            center_exact=(Fraction(0), Fraction(0)),
            first_axis_exact=(Fraction(1), Fraction(0)),
            second_axis_exact=(Fraction(0), Fraction(1)),
            radius_exact=Fraction(1),
        )
        try:
            phase_root = ConicParameterRoot(
                unit_circle, phase_cosine, phase_sine, turns=0
            )
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None

    def exact_dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    world_center = tuple(
        data.cylinder_origin[axis] + data.height_center * data.cylinder_axis[axis]
        for axis in range(3)
    )
    world_cosine = tuple(
        data.cylinder.radius_exact * data.radial_first[axis]
        + data.height_cosine * data.cylinder_axis[axis]
        for axis in range(3)
    )
    world_sine = tuple(
        data.cylinder.radius_exact * data.radial_second[axis]
        + data.height_sine * data.cylinder_axis[axis]
        for axis in range(3)
    )
    principal_cosine = tuple(
        world_cosine[axis] * phase_cosine + world_sine[axis] * phase_sine
        for axis in range(3)
    )
    principal_sine = tuple(
        -world_cosine[axis] * phase_sine + world_sine[axis] * phase_cosine
        for axis in range(3)
    )
    first_radius_squared = exact_dot(principal_cosine, principal_cosine)
    second_radius_squared = exact_dot(principal_sine, principal_sine)
    if (
        exact_dot(principal_cosine, principal_sine) != 0
        or first_radius_squared != radius_squared + height_squared
        or second_radius_squared != radius_squared
    ):
        return None
    try:
        first_radius_squared = rational_coefficient(
            first_radius_squared, role="plane-cylinder ellipse first radius squared"
        )
        second_radius_squared = rational_coefficient(
            second_radius_squared, role="plane-cylinder ellipse second radius squared"
        )
        first_radius = RadicalElement.sqrt(first_radius_squared)
        second_radius = RadicalElement.sqrt(second_radius_squared)
        first_axis = tuple(value / first_radius for value in principal_cosine)
        second_axis = tuple(value / second_radius for value in principal_sine)
        edge = EllipseCurve(
            tuple(float(value) for value in world_center),
            tuple(float(value) for value in first_axis),
            tuple(float(value) for value in second_axis),
            float(first_radius),
            float(second_radius),
            center_exact=world_center,
            first_axis_exact=first_axis,
            second_axis_exact=second_axis,
            first_radius_exact=first_radius,
            second_radius_exact=second_radius,
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None

    first_uv_cosine = (
        data.uv_cosine[0] * phase_cosine + data.uv_sine[0] * phase_sine,
        data.uv_cosine[1] * phase_cosine + data.uv_sine[1] * phase_sine,
    )
    first_uv_sine = (
        -data.uv_cosine[0] * phase_sine + data.uv_sine[0] * phase_cosine,
        -data.uv_cosine[1] * phase_sine + data.uv_sine[1] * phase_cosine,
    )
    try:
        endpoint_roots: list[ConicParameterRoot] = []
        for cylinder_root in cylinder_angle_roots:
            if phase_root is None:
                endpoint_roots.append(
                    ConicParameterRoot(
                        edge,
                        cylinder_root.cosine_exact,
                        cylinder_root.sine_exact,
                        turns=cylinder_root.turns,
                    )
                )
                continue
            combined = _combine_conic_parameter_roots(
                cylinder_root, phase_root, subtract=True
            )
            if combined is None:
                return None
            endpoint_roots.append(
                ConicParameterRoot(
                    edge,
                    combined.cosine_exact,
                    combined.sine_exact,
                    turns=combined.turns,
                )
            )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    start_root, end_root = endpoint_roots
    if (
        start_root.cosine_exact == end_root.cosine_exact
        and start_root.sine_exact == end_root.sine_exact
    ):
        return None
    parameter_range = (start_root.parameter, end_root.parameter)
    if not parameter_range[0] < parameter_range[1]:
        return None

    try:
        plane_pcurve = PlaneCylinderSectionPCurve(
            data.uv_center,
            (Fraction(0), Fraction(0)),
            first_uv_cosine,
            first_uv_sine,
            parameter_range=parameter_range,
        )
        cylinder_pcurve = PlaneCylinderSectionPCurve(
            (Fraction(0), data.height_center),
            (Fraction(1), Fraction(0)),
            (Fraction(0), height_amplitude),
            (Fraction(0), Fraction(0)),
            parameter_range=parameter_range,
            angle_offset_root=phase_root,
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None

    height_range = _plane_cylinder_harmonic_interval(
        data.height_center,
        data.height_cosine,
        data.height_sine,
        data.theta_lower,
        data.theta_upper,
    )
    cylinder_bounds = np.asarray(data.cylinder_region.parameter_box, dtype=np.float64)
    if height_range[0] < cylinder_bounds[0, 1] or height_range[1] > cylinder_bounds[1, 1]:
        return None
    plane_bounds = np.asarray(data.plane_region.parameter_box, dtype=np.float64)
    for axis in range(2):
        uv_range = _plane_cylinder_harmonic_interval(
            data.uv_center[axis],
            data.uv_cosine[axis],
            data.uv_sine[axis],
            data.theta_lower,
            data.theta_upper,
        )
        if uv_range[0] < plane_bounds[0, axis] or uv_range[1] > plane_bounds[1, axis]:
            return None
    return PlaneCylinderConicSection(
        edge,
        plane_pcurve,
        cylinder_pcurve,
        parameter_range,
        (start_root, end_root),
    )


def _plane_cylinder_harmonic_interval(
    base: ExactScalar,
    cosine_coefficient: ExactScalar,
    sine_coefficient: ExactScalar,
    angle_lower: float,
    angle_upper: float,
    /,
) -> tuple[float, float]:
    from ...discretization._exact_algebraic_geometry import coefficient_interval
    from .._interval_enclosure import (
        interval_add,
        interval_cos,
        interval_multiply,
        interval_sin,
    )
    from .._rational import fraction_down, fraction_up

    def coefficient(value: ExactScalar, /) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = coefficient_interval(value, bits=128)
        return np.asarray(fraction_down(lower)), np.asarray(fraction_up(upper))

    cosine = interval_cos((np.asarray(angle_lower), np.asarray(angle_upper)))
    sine = interval_sin((np.asarray(angle_lower), np.asarray(angle_upper)))
    cosine_term = interval_multiply(coefficient(cosine_coefficient), cosine)
    sine_term = interval_multiply(coefficient(sine_coefficient), sine)
    bounds = interval_add(interval_add(coefficient(base), cosine_term), sine_term)
    return float(bounds[0]), float(bounds[1])


def _plane_cylinder_angular_branch(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    data = _prepare_plane_cylinder_angular_data(first, second)
    if data is None:
        return None
    cylinder_bounds = np.asarray(data.cylinder_region.parameter_box, dtype=np.float64)
    height_range = _plane_cylinder_harmonic_interval(
        data.height_center,
        data.height_cosine,
        data.height_sine,
        data.theta_lower,
        data.theta_upper,
    )
    if height_range[1] < cylinder_bounds[0, 1] or height_range[0] > cylinder_bounds[1, 1]:
        return SurfaceIntersectionResult()
    if height_range[0] < cylinder_bounds[0, 1] or height_range[1] > cylinder_bounds[1, 1]:
        return _unresolved_surface(first, second, "continuation")
    plane_bounds = np.asarray(data.plane_region.parameter_box, dtype=np.float64)
    for axis in range(2):
        image_range = _plane_cylinder_harmonic_interval(
            data.uv_center[axis],
            data.uv_cosine[axis],
            data.uv_sine[axis],
            data.theta_lower,
            data.theta_upper,
        )
        if (
            image_range[1] < plane_bounds[0, axis]
            or image_range[0] > plane_bounds[1, axis]
        ):
            return SurfaceIntersectionResult()
        if (
            image_range[0] < plane_bounds[0, axis]
            or image_range[1] > plane_bounds[1, axis]
        ):
            return _unresolved_surface(first, second, "continuation")

    branch_data: _PlaneCylinderAngularData = data
    chart_axis = 2 if data.plane_first else 0

    def point_at(angle: float, /) -> np.ndarray:
        cosine, sine = math.cos(angle), math.sin(angle)
        plane_uv = tuple(
            float(branch_data.uv_center[index])
            + float(branch_data.uv_cosine[index]) * cosine
            + float(branch_data.uv_sine[index]) * sine
            for index in range(2)
        )
        height = (
            float(branch_data.height_center)
            + float(branch_data.height_cosine) * cosine
            + float(branch_data.height_sine) * sine
        )
        cylinder_uv = (angle, height)
        return np.asarray(
            (*plane_uv, *cylinder_uv)
            if branch_data.plane_first
            else (*cylinder_uv, *plane_uv),
            dtype=np.float64,
        )

    def box_for(
        lower_angle: float, upper_angle: float, /
    ) -> tuple[np.ndarray, np.ndarray]:
        plane_uv_bounds = tuple(
            _plane_cylinder_harmonic_interval(
                branch_data.uv_center[index],
                branch_data.uv_cosine[index],
                branch_data.uv_sine[index],
                lower_angle,
                upper_angle,
            )
            for index in range(2)
        )
        height_bounds = _plane_cylinder_harmonic_interval(
            branch_data.height_center,
            branch_data.height_cosine,
            branch_data.height_sine,
            lower_angle,
            upper_angle,
        )
        if branch_data.plane_first:
            lower = np.asarray(
                (
                    plane_uv_bounds[0][0],
                    plane_uv_bounds[1][0],
                    lower_angle,
                    height_bounds[0],
                ),
                dtype=np.float64,
            )
            upper = np.asarray(
                (
                    plane_uv_bounds[0][1],
                    plane_uv_bounds[1][1],
                    upper_angle,
                    height_bounds[1],
                ),
                dtype=np.float64,
            )
        else:
            lower = np.asarray(
                (
                    lower_angle,
                    height_bounds[0],
                    plane_uv_bounds[0][0],
                    plane_uv_bounds[1][0],
                ),
                dtype=np.float64,
            )
            upper = np.asarray(
                (
                    upper_angle,
                    height_bounds[1],
                    plane_uv_bounds[0][1],
                    plane_uv_bounds[1][1],
                ),
                dtype=np.float64,
            )
        return lower, upper

    endpoint_kind = "closed" if branch_data.closed else "boundary"
    curve = _angular_surface_branch(
        first,
        second,
        chart_axis,
        branch_data.theta_lower,
        branch_data.theta_upper,
        policy,
        point_at,
        box_for,
        closed=branch_data.closed,
        start_kind=endpoint_kind,
        end_kind=endpoint_kind,
    )
    return (
        SurfaceIntersectionResult(curves=(curve,))
        if curve is not None
        else _unresolved_surface(first, second, "resolution")
    )


def _horizontal_torus_plane_circles(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    """Certify the two regular circles cut from an aligned ring torus."""
    if isinstance(first.patch, TorusPatch) and isinstance(second.patch, PlanePatch):
        torus, plane, torus_region, plane_region = (
            first.patch,
            second.patch,
            first,
            second,
        )
        torus_first = True
    elif isinstance(first.patch, PlanePatch) and isinstance(second.patch, TorusPatch):
        torus, plane, torus_region, plane_region = (
            second.patch,
            first.patch,
            second,
            first,
        )
        torus_first = False
    else:
        return None
    frame = _exact_unit_frame(torus.first_axis, torus.second_axis, torus.axis)
    if frame is None:
        return None

    from .._interval_enclosure import (
        interval_add,
        interval_cos,
        interval_multiply,
        interval_sin,
    )
    from .._rational import fraction_down, fraction_up
    from ._exact import square_root_interval

    first_axis, second_axis, torus_axis = frame
    plane_first_axis = rational_vector(np.asarray(plane.first_axis))
    plane_second_axis = rational_vector(np.asarray(plane.second_axis))
    plane_normal = (
        plane_first_axis[1] * plane_second_axis[2]
        - plane_first_axis[2] * plane_second_axis[1],
        plane_first_axis[2] * plane_second_axis[0]
        - plane_first_axis[0] * plane_second_axis[2],
        plane_first_axis[0] * plane_second_axis[1]
        - plane_first_axis[1] * plane_second_axis[0],
    )
    center = rational_vector(np.asarray(torus.center))
    plane_origin = rational_vector(np.asarray(plane.origin))
    if any(
        plane_normal[i] * torus_axis[j] != plane_normal[j] * torus_axis[i]
        for i in range(3)
        for j in range(3)
    ):
        return None
    if all(value == 0 for value in plane_normal):
        return None
    height = rational_dot(
        tuple(plane_origin[i] - center[i] for i in range(3)), torus_axis
    )
    major_radius = Fraction.from_float(float(torus.major_radius))
    minor_radius = Fraction.from_float(float(torus.minor_radius))
    if major_radius <= 0 or minor_radius <= 0:
        return None
    radial_squared = minor_radius * minor_radius - height * height
    if radial_squared < 0:
        return SurfaceIntersectionResult()
    if radial_squared == 0:
        return _unresolved_surface(first, second, "singular")

    theta_lower, theta_upper = (
        float(value) for value in np.asarray(torus_region.parameter_box)[:, 0]
    )
    period = torus.periods[0]
    tube_period = torus.periods[1]
    tube_lower, tube_upper = (
        float(value) for value in np.asarray(torus_region.parameter_box)[:, 1]
    )
    if (
        torus.native_period_symbols()[0] != "two_pi"
        or period is None
        or Fraction.from_float(theta_upper) - Fraction.from_float(theta_lower)
        != Fraction.from_float(float(period))
        or torus.native_period_symbols()[1] != "two_pi"
        or tube_period is None
        or Fraction.from_float(tube_upper) - Fraction.from_float(tube_lower)
        != Fraction.from_float(float(tube_period))
    ):
        return _unresolved_surface(first, second, "continuation")

    radial_interval = square_root_interval(radial_squared)
    radius_down, radius_up = fraction_down(major_radius), fraction_up(major_radius)
    tube_angle = math.asin(float(height / minor_radius))
    tube_domain = np.asarray((tube_lower, tube_upper), dtype=np.float64)
    center_circle = tuple(center[i] + height * torus_axis[i] for i in range(3))
    cosine_map = _plane_uv_affine(plane, center_circle, first_axis)
    sine_map = _plane_uv_affine(plane, center_circle, second_axis)
    if cosine_map is None or sine_map is None:
        return None
    uv_center, uv_cosine = cosine_map
    _, uv_sine = sine_map
    plane_bounds = np.asarray(plane_region.parameter_box, dtype=np.float64)
    curves: list[IntersectionCurve] = []

    for outer in (True, False):
        radius_interval = (
            interval_add(
                (np.asarray(radius_down), np.asarray(radius_up)),
                (np.asarray(radial_interval[0]), np.asarray(radial_interval[1])),
            )
            if outer
            else interval_add(
                (np.asarray(radius_down), np.asarray(radius_up)),
                (np.asarray(-radial_interval[1]), np.asarray(-radial_interval[0])),
            )
        )
        if float(radius_interval[0]) <= 0.0:
            return _unresolved_surface(first, second, "singular")
        disjoint = False
        partial = False
        for axis in range(2):
            norm_interval = square_root_interval(
                uv_cosine[axis] * uv_cosine[axis] + uv_sine[axis] * uv_sine[axis]
            )
            amplitude = interval_multiply(
                radius_interval,
                (np.asarray(norm_interval[0]), np.asarray(norm_interval[1])),
            )
            center_bounds = (
                np.asarray(fraction_down(uv_center[axis])),
                np.asarray(fraction_up(uv_center[axis])),
            )
            image_lower = interval_add(
                center_bounds,
                (np.negative(amplitude[1]), np.negative(amplitude[0])),
            )
            image_upper = interval_add(center_bounds, amplitude)
            low, high = float(image_lower[0]), float(image_upper[1])
            if high < plane_bounds[0, axis] or low > plane_bounds[1, axis]:
                disjoint = True
                break
            if low < plane_bounds[0, axis] or high > plane_bounds[1, axis]:
                partial = True
        if disjoint:
            continue
        if partial:
            return _unresolved_surface(first, second, "continuation")

        branch_angle = tube_angle if outer else math.pi - tube_angle
        tube_period_value = float(tube_period)
        turns = math.ceil((tube_domain[0] - branch_angle) / tube_period_value)
        branch_angle += turns * tube_period_value
        if branch_angle > tube_domain[1]:
            branch_angle -= tube_period_value
        if branch_angle < tube_domain[0] or branch_angle > tube_domain[1]:
            return _unresolved_surface(first, second, "continuation")
        radius_midpoint = 0.5 * float(radius_interval[0]) + 0.5 * float(
            radius_interval[1]
        )
        chart_axis = 0 if torus_first else 2

        def point_at(angle: float, /) -> np.ndarray:
            cosine, sine = math.cos(angle), math.sin(angle)
            plane_uv = tuple(
                float(uv_center[index])
                + radius_midpoint
                * (float(uv_cosine[index]) * cosine + float(uv_sine[index]) * sine)
                for index in range(2)
            )
            torus_uv = (angle, branch_angle)
            return np.asarray(
                (*torus_uv, *plane_uv) if torus_first else (*plane_uv, *torus_uv),
                dtype=np.float64,
            )

        def box_for(
            lower_angle: float, upper_angle: float, /
        ) -> tuple[np.ndarray, np.ndarray]:
            cosine = interval_cos((np.asarray(lower_angle), np.asarray(upper_angle)))
            sine = interval_sin((np.asarray(lower_angle), np.asarray(upper_angle)))
            uv_lower = np.empty((2,), dtype=np.float64)
            uv_upper = np.empty((2,), dtype=np.float64)
            for axis in range(2):
                center_bounds = (
                    np.asarray(fraction_down(uv_center[axis])),
                    np.asarray(fraction_up(uv_center[axis])),
                )
                cosine_coefficient = interval_multiply(
                    radius_interval,
                    (
                        np.asarray(fraction_down(uv_cosine[axis])),
                        np.asarray(fraction_up(uv_cosine[axis])),
                    ),
                )
                sine_coefficient = interval_multiply(
                    radius_interval,
                    (
                        np.asarray(fraction_down(uv_sine[axis])),
                        np.asarray(fraction_up(uv_sine[axis])),
                    ),
                )
                cosine_term = interval_multiply(cosine_coefficient, cosine)
                sine_term = interval_multiply(sine_coefficient, sine)
                bounds = interval_add(interval_add(center_bounds, cosine_term), sine_term)
                uv_lower[axis], uv_upper[axis] = float(bounds[0]), float(bounds[1])
            if torus_first:
                lower = np.asarray(
                    (lower_angle, branch_angle, *uv_lower), dtype=np.float64
                )
                upper = np.asarray(
                    (upper_angle, branch_angle, *uv_upper), dtype=np.float64
                )
            else:
                lower = np.asarray(
                    (*uv_lower, lower_angle, branch_angle), dtype=np.float64
                )
                upper = np.asarray(
                    (*uv_upper, upper_angle, branch_angle), dtype=np.float64
                )
            return lower, upper

        pad = max(
            policy.relative_resolution * max(1.0, abs(branch_angle)),
            64.0 * np.finfo(np.float64).eps * max(1.0, abs(branch_angle)),
        )
        maximum_pad = max(
            abs(branch_angle - float(tube_domain[0])),
            abs(float(tube_domain[1]) - branch_angle),
        )
        curve = None
        while True:
            # The fixed tube-angle enclosure is included in each chart box.
            lower_tube = max(float(tube_domain[0]), branch_angle - pad)
            upper_tube = min(float(tube_domain[1]), branch_angle + pad)
            if lower_tube <= branch_angle <= upper_tube:

                def branch_box(
                    lower_angle: float, upper_angle: float, /
                ) -> tuple[np.ndarray, np.ndarray]:
                    lower, upper = box_for(lower_angle, upper_angle)
                    tube_index = 1 if torus_first else 3
                    lower[tube_index] = lower_tube
                    upper[tube_index] = upper_tube
                    return lower, upper

                curve = _angular_surface_branch(
                    first,
                    second,
                    chart_axis,
                    theta_lower,
                    theta_upper,
                    policy,
                    point_at,
                    branch_box,
                )
            if curve is not None:
                curves.append(curve)
                break
            if pad >= maximum_pad:
                return _unresolved_surface(first, second, "resolution")
            next_pad = min(2.0 * pad, maximum_pad)
            if next_pad <= pad:
                return _unresolved_surface(first, second, "resolution")
            pad = next_pad
    return SurfaceIntersectionResult(curves=curves)


def _revolution_plane_conic_circles(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    """Intersect a circle-profile revolution with a transverse plane exactly."""
    from ._patches import CircleCurve, RevolutionSurface

    if isinstance(first.patch, RevolutionSurface) and isinstance(
        second.patch, PlanePatch
    ):
        revolution, plane, revolution_region, plane_region = (
            first.patch,
            second.patch,
            first,
            second,
        )
        plane_is_first = False
    elif isinstance(first.patch, PlanePatch) and isinstance(
        second.patch, RevolutionSurface
    ):
        revolution, plane, revolution_region, plane_region = (
            second.patch,
            first.patch,
            second,
            first,
        )
        plane_is_first = True
    else:
        return None
    profile = revolution.curve
    if not isinstance(profile, CircleCurve) or profile.ambient_dimension != 3:
        return None

    from .._rational import rational_cross, rational_vector

    def exact_dot(
        first_vector: tuple[ExactScalar, ...], second_vector: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum(
            (
                left * right
                for left, right in zip(first_vector, second_vector, strict=True)
            ),
            Fraction(0),
        )

    def fraction_vector(
        values: tuple[ExactScalar, ...], /
    ) -> tuple[Fraction, ...] | None:
        result: list[Fraction] = []
        for value in values:
            if not isinstance(value, Fraction):
                return None
            result.append(value)
        return tuple(result)

    profile_center = fraction_vector(profile.center_exact)
    profile_first = fraction_vector(profile.first_axis_exact)
    profile_second = fraction_vector(profile.second_axis_exact)
    if profile_center is None or profile_first is None or profile_second is None:
        return None
    if not isinstance(profile.radius_exact, Fraction) or profile.radius_exact <= 0:
        return None
    radius = profile.radius_exact

    axis_origin = rational_vector(np.asarray(revolution.axis_origin))
    axis = rational_vector(np.asarray(revolution.axis_direction))
    plane_origin = rational_vector(np.asarray(plane.origin))
    plane_first_axis = rational_vector(np.asarray(plane.first_axis))
    plane_second_axis = rational_vector(np.asarray(plane.second_axis))
    plane_normal = rational_cross(plane_first_axis, plane_second_axis)
    profile_normal = rational_cross(profile_first, profile_second)
    if (
        rational_dot(axis, axis) != 1
        or rational_dot(profile_first, profile_first) != 1
        or rational_dot(profile_second, profile_second) != 1
        or rational_dot(profile_first, profile_second) != 0
        or rational_dot(axis, profile_normal) != 0
        or rational_dot(
            tuple(profile_center[index] - axis_origin[index] for index in range(3)),
            profile_normal,
        )
        != 0
        or all(value == 0 for value in plane_normal)
        or any(
            plane_normal[i] * axis[j] != plane_normal[j] * axis[i]
            for i in range(3)
            for j in range(3)
        )
    ):
        return None
    radial_axis = rational_cross(axis, profile_normal)
    if rational_dot(radial_axis, radial_axis) != 1:
        return None

    height = rational_dot(
        tuple(plane_origin[index] - axis_origin[index] for index in range(3)), axis
    )
    profile_height = rational_dot(
        tuple(profile_center[index] - axis_origin[index] for index in range(3)), axis
    )
    offset = height - profile_height
    first_height = rational_dot(profile_first, axis)
    second_height = rational_dot(profile_second, axis)

    frame_norm = first_height * first_height + second_height * second_height
    if frame_norm == 0:
        return (
            SurfaceIntersectionResult()
            if offset != 0
            else _unresolved_surface(first, second, "singular")
        )
    ratio = offset / radius
    discriminant = Fraction(1) - ratio * ratio / frame_norm
    if discriminant < 0:
        return SurfaceIntersectionResult()
    if discriminant == 0:
        return _unresolved_surface(first, second, "singular")

    angle_lower, angle_upper = (
        float(value) for value in np.asarray(revolution_region.parameter_box)[:, 0]
    )
    angle_period = revolution.periods[0]
    if (
        revolution.native_period_symbols()[0] != "two_pi"
        or angle_period is None
        or Fraction.from_float(angle_upper) - Fraction.from_float(angle_lower)
        != Fraction.from_float(float(angle_period))
    ):
        return _unresolved_surface(first, second, "continuation")
    profile_period = profile.period
    if profile.native_period_symbol() != "two_pi" or profile_period is None:
        return _unresolved_surface(first, second, "continuation")
    profile_box = np.asarray(revolution_region.parameter_box, dtype=np.float64)
    profile_lower, profile_upper = float(profile_box[0, 1]), float(profile_box[1, 1])
    if Fraction.from_float(profile_upper) - Fraction.from_float(
        profile_lower
    ) > Fraction.from_float(float(profile_period)):
        return _unresolved_surface(first, second, "continuation")

    root_square = RadicalElement.sqrt(discriminant)
    frame_square = RadicalElement.sqrt(frame_norm)
    center_cosine = ratio * first_height / frame_norm
    center_sine = ratio * second_height / frame_norm
    curves: list[IntersectionCurve] = []
    unresolved_root = False

    for sign in (-1, 1):
        cosine_exact = center_cosine + sign * root_square * (
            -second_height / frame_square
        )
        sine_exact = center_sine + sign * root_square * (first_height / frame_square)
        try:
            source_root = ConicParameterRoot(profile, cosine_exact, sine_exact)
        except ValueError:
            return _unresolved_surface(first, second, "resolution")
        center_parameter = 0.5 * (profile_lower + profile_upper)
        nearest_turn = round(
            (center_parameter - source_root.parameter) / float(profile_period)
        )
        root_candidates: list[ConicParameterRoot] = []
        partial_root = False
        for turns in (nearest_turn - 1, nearest_turn, nearest_turn + 1):
            candidate = ConicParameterRoot(profile, cosine_exact, sine_exact, turns=turns)
            candidate_lower, candidate_upper = candidate.parameter_enclosure()
            if candidate_upper < profile_lower or candidate_lower > profile_upper:
                continue
            if candidate_lower >= profile_lower and candidate_upper <= profile_upper:
                root_candidates.append(candidate)
            else:
                partial_root = True
        if not root_candidates:
            if partial_root:
                unresolved_root = True
            continue
        profile_root = min(
            root_candidates,
            key=lambda candidate: abs(candidate.parameter - center_parameter),
        )
        profile_root_lower, profile_root_upper = profile_root.parameter_enclosure()
        profile_point = profile_root.exact_point()
        relative_point = tuple(
            profile_point[index] - axis_origin[index] for index in range(3)
        )
        radial_vector = tuple(
            relative_point[index] - height * axis[index] for index in range(3)
        )
        radial_radius = exact_dot(radial_vector, radial_axis)
        if radial_radius == 0:
            return _unresolved_surface(first, second, "singular")
        if any(
            radial_vector[index] != radial_radius * radial_axis[index]
            for index in range(3)
        ):
            return None
        branch = _revolution_plane_isoline(
            first,
            second,
            policy,
            plane_is_first=plane_is_first,
            plane_region=plane_region,
            axis_origin=axis_origin,
            axis=axis,
            height=height,
            radial_vector=radial_vector,
            parameter=profile_root.parameter,
            parameter_enclosure=(profile_root_lower, profile_root_upper),
            profile_range=(profile_lower, profile_upper),
            angle_range=(angle_lower, angle_upper),
        )
        if branch is None:
            return None
        if isinstance(branch, SurfaceIntersectionResult):
            if branch.unresolved:
                return branch
            continue
        curves.append(branch)
    if unresolved_root:
        return _unresolved_surface(first, second, "continuation")
    return SurfaceIntersectionResult(curves=curves)


def _exact_dot(
    first: Sequence[ExactScalar], second: Sequence[ExactScalar], /
) -> ExactScalar:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)), Fraction(0)
    )


def _exact_harmonic_range(
    base: ExactScalar,
    cosine_coefficient: ExactScalar,
    sine_coefficient: ExactScalar,
    /,
) -> tuple[float, float] | None:
    """Enclose ``base + a cos t + b sin t`` over a full turn."""
    from .._interval_enclosure import interval_add
    from ._exact import square_root_interval

    squared_amplitude = (
        cosine_coefficient * cosine_coefficient + sine_coefficient * sine_coefficient
    )
    amplitude_bounds = coefficient_interval(squared_amplitude, bits=128)
    if amplitude_bounds[1] < 0:
        return None
    amplitude_lower = square_root_interval(max(Fraction(0), amplitude_bounds[0]))[0]
    amplitude_upper = square_root_interval(max(Fraction(0), amplitude_bounds[1]))[1]
    center_lower, center_upper = coefficient_interval(base, bits=128)
    center_bounds = (
        np.asarray(fraction_down(center_lower)),
        np.asarray(fraction_up(center_upper)),
    )
    lower_bounds = interval_add(
        center_bounds, (np.asarray(-amplitude_upper), np.asarray(-amplitude_lower))
    )
    upper_bounds = interval_add(
        center_bounds, (np.asarray(amplitude_lower), np.asarray(amplitude_upper))
    )
    return float(lower_bounds[0]), float(upper_bounds[1])


@dataclass(frozen=True, slots=True)
class _IsolineArc:
    """One arc of a plane circle inside a plane chart rectangle.

    Each end is a crossing of one rectangle edge: the chart coordinate
    ``*_axis`` equals the exact edge value ``*_value`` there, at an angle
    enclosed by ``[*_lower, *_upper]``.
    """

    start_lower: float
    start_upper: float
    start_axis: int
    start_value: float
    end_lower: float
    end_upper: float
    end_axis: int
    end_value: float
    across_seam: bool = False


def _isoline_rectangle_arcs(
    center: Sequence[ExactScalar],
    cosine: Sequence[ExactScalar],
    sine: Sequence[ExactScalar],
    bounds: np.ndarray,
    window_lower: float,
    /,
) -> tuple[_IsolineArc, ...] | Literal["full"] | None:
    """Split the circle ``center + cosine cos t + sine sin t`` by a chart rectangle.

    Edge crossings solve ``c + a cos t + b sin t = e`` in closed form and are
    enclosed at 80 digits; between consecutive crossings the circle stays on
    one side of every edge line, so one interior sample decides each arc.
    Returns ``"full"`` for a circle inside the rectangle, the inside arcs (an
    arc that passes the angle window's seam is marked), or ``None`` for a
    tangency or a corner crossing.
    """
    context = interval_context(80)

    def interval(value: ExactScalar, /) -> Any:
        lower, upper = coefficient_interval(value, bits=256)
        first = context.mpf(lower.numerator) / context.mpf(lower.denominator)
        last = context.mpf(upper.numerator) / context.mpf(upper.denominator)
        return context.mpf([first.a, last.b])

    centers = [interval(value) for value in center]
    cosines = [interval(value) for value in cosine]
    sines = [interval(value) for value in sine]
    two_pi = 2 * context.pi
    window = interval(Fraction.from_float(float(window_lower)))

    def side(angle: Any, /) -> int | None:
        # +1 strictly inside the rectangle, -1 strictly outside, None undecided.
        inside = True
        for axis in range(2):
            value = (
                centers[axis]
                + cosines[axis] * context.cos(angle)
                + sines[axis] * context.sin(angle)
            )
            lower = interval(Fraction.from_float(float(bounds[0, axis])))
            upper = interval(Fraction.from_float(float(bounds[1, axis])))
            if value.b < lower.a or value.a > upper.b:
                return -1
            if not (value.a > lower.b and value.b < upper.a):
                inside = False
        return 1 if inside else None

    events: list[tuple[Any, int, float]] = []
    for axis in range(2):
        amplitude = context.sqrt(cosines[axis] ** 2 + sines[axis] ** 2)
        if not amplitude.a > 0:
            return None
        phase = context.atan2(sines[axis], cosines[axis])
        for edge in range(2):
            value = float(bounds[edge, axis])
            ratio = (interval(Fraction.from_float(value)) - centers[axis]) / amplitude
            if ratio.b < -1 or ratio.a > 1:
                continue
            if not (-1 < ratio.a and ratio.b < 1):
                return None
            offset = context.atan2(context.sqrt(1 - ratio**2), ratio)
            for angle in (phase - offset, phase + offset):
                middle = 0.5 * float(angle.a) + 0.5 * float(angle.b)
                angle = angle - math.floor((middle - window_lower) / math.tau) * two_pi
                if not (window.b < angle.a and angle.b < (window + two_pi).a):
                    return None
                events.append((angle, axis, value))
    if not events:
        state = side(window + context.pi)
        if state is None:
            return None
        return "full" if state > 0 else ()
    events.sort(key=lambda event: float(event[0].a))
    if any(
        not events[index][0].b < events[index + 1][0].a
        for index in range(len(events) - 1)
    ):
        return None
    arcs = []
    for index, (start, start_axis, start_value) in enumerate(events):
        across_seam = index == len(events) - 1
        end, end_axis, end_value = events[(index + 1) % len(events)]
        state = side(0.5 * (start + end + (two_pi if across_seam else 0)))
        if state is None:
            return None
        if state < 0:
            continue
        arcs.append(
            _IsolineArc(
                float(np.nextafter(float(start.a), -np.inf)),
                float(np.nextafter(float(start.b), np.inf)),
                start_axis,
                start_value,
                float(np.nextafter(float(end.a), -np.inf)),
                float(np.nextafter(float(end.b), np.inf)),
                end_axis,
                end_value,
                across_seam,
            )
        )
    return tuple(arcs)


def _plane_chart_coordinates(
    plane: PlanePatch, vector: Sequence[ExactScalar], /, *, point: bool
) -> tuple[ExactScalar, ExactScalar] | None:
    first_axis = rational_vector(np.asarray(plane.first_axis))
    second_axis = rational_vector(np.asarray(plane.second_axis))
    origin = rational_vector(np.asarray(plane.origin))
    g00 = rational_dot(first_axis, first_axis)
    g01 = rational_dot(first_axis, second_axis)
    g11 = rational_dot(second_axis, second_axis)
    determinant = g00 * g11 - g01 * g01
    if determinant == 0:
        return None
    delta = (
        tuple(vector[index] - origin[index] for index in range(3)) if point else vector
    )
    rhs0, rhs1 = _exact_dot(first_axis, delta), _exact_dot(second_axis, delta)
    return (
        (g11 * rhs0 - g01 * rhs1) / determinant,
        (g00 * rhs1 - g01 * rhs0) / determinant,
    )


def _revolution_plane_isoline(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
    *,
    plane_is_first: bool,
    plane_region: SurfaceRegion,
    axis_origin: Sequence[Fraction],
    axis: Sequence[Fraction],
    height: Fraction,
    radial_vector: Sequence[ExactScalar],
    parameter: float,
    parameter_enclosure: tuple[float, float],
    profile_range: tuple[float, float],
    angle_range: tuple[float, float],
) -> IntersectionCurve | SurfaceIntersectionResult | None:
    """Certify one transverse-plane circle at a fixed revolution profile parameter.

    Returns the angular branch, an empty result when the circle misses the
    plane chart, an unresolved result, or None for unsupported coefficients.
    """
    plane = plane_region.patch
    if not isinstance(plane, PlanePatch):
        return None
    profile_lower, profile_upper = profile_range
    angle_lower, angle_upper = angle_range
    parameter_lower, parameter_upper = parameter_enclosure
    plane_bounds = np.asarray(plane_region.parameter_box, dtype=np.float64)
    tangent_axis = (
        axis[1] * radial_vector[2] - axis[2] * radial_vector[1],
        axis[2] * radial_vector[0] - axis[0] * radial_vector[2],
        axis[0] * radial_vector[1] - axis[1] * radial_vector[0],
    )
    circle_center = tuple(axis_origin[index] + height * axis[index] for index in range(3))
    uv_center = _plane_chart_coordinates(plane, circle_center, point=True)
    uv_cosine = _plane_chart_coordinates(plane, radial_vector, point=False)
    uv_sine = _plane_chart_coordinates(plane, tangent_axis, point=False)
    if uv_center is None or uv_cosine is None or uv_sine is None:
        return None
    uv_center_source: tuple[ExactScalar, ExactScalar] = uv_center
    uv_cosine_source: tuple[ExactScalar, ExactScalar] = uv_cosine
    uv_sine_source: tuple[ExactScalar, ExactScalar] = uv_sine

    for parameter_axis in range(2):
        image_range = _exact_harmonic_range(
            uv_center_source[parameter_axis],
            uv_cosine_source[parameter_axis],
            uv_sine_source[parameter_axis],
        )
        if image_range is None:
            return None
        if (
            image_range[1] < plane_bounds[0, parameter_axis]
            or image_range[0] > plane_bounds[1, parameter_axis]
        ):
            return SurfaceIntersectionResult()
    arcs = _isoline_rectangle_arcs(
        uv_center_source, uv_cosine_source, uv_sine_source, plane_bounds, angle_lower
    )
    if arcs is None:
        return _unresolved_surface(first, second, "continuation")
    if not arcs:
        return SurfaceIntersectionResult()

    chart_axis = 2 if plane_is_first else 0
    profile_chart_lower = parameter_lower
    profile_chart_upper = parameter_upper
    # A trimmed arc's chart boxes enclose the arc's own plane image; the whole
    # plane chart would make every enclosure of the arc cover the face.
    trimmed = False

    def point_at(angle: float, /) -> np.ndarray:
        cosine, sine = math.cos(angle), math.sin(angle)
        plane_uv = tuple(
            float(uv_center_source[index])
            + float(uv_cosine_source[index]) * cosine
            + float(uv_sine_source[index]) * sine
            for index in range(2)
        )
        revolution_uv = (angle, parameter)
        return np.asarray(
            (*plane_uv, *revolution_uv)
            if plane_is_first
            else (*revolution_uv, *plane_uv),
            dtype=np.float64,
        )

    def box_for(
        lower_angle: float, upper_angle: float, /
    ) -> tuple[np.ndarray, np.ndarray]:
        plane_lower = np.asarray(plane_region.parameter_box[0], dtype=np.float64)
        plane_upper = np.asarray(plane_region.parameter_box[1], dtype=np.float64)
        if trimmed:
            harmonic = [
                _plane_cylinder_harmonic_interval(
                    uv_center_source[index],
                    uv_cosine_source[index],
                    uv_sine_source[index],
                    lower_angle,
                    upper_angle,
                )
                for index in range(2)
            ]
            plane_lower = np.asarray([value[0] for value in harmonic])
            plane_upper = np.asarray([value[1] for value in harmonic])
        if plane_is_first:
            lower = np.asarray(
                (plane_lower[0], plane_lower[1], lower_angle, profile_chart_lower),
                dtype=np.float64,
            )
            upper = np.asarray(
                (plane_upper[0], plane_upper[1], upper_angle, profile_chart_upper),
                dtype=np.float64,
            )
        else:
            lower = np.asarray(
                (lower_angle, profile_chart_lower, plane_lower[0], plane_lower[1]),
                dtype=np.float64,
            )
            upper = np.asarray(
                (upper_angle, profile_chart_upper, plane_upper[0], plane_upper[1]),
                dtype=np.float64,
            )
        return lower, upper

    initial_pad = max(
        policy.relative_probe * max(1.0, abs(parameter)),
        128.0 * np.finfo(np.float64).eps * max(1.0, abs(parameter)),
    )
    maximum_pad = max(
        abs(parameter_lower - profile_lower),
        abs(profile_upper - parameter_upper),
    )
    plane_offset = 0 if plane_is_first else 2

    def inner_angle(angle: float, step: float, /) -> float | None:
        # The float that represents a trim crossing lies on the arc's side, so
        # the chart endpoint stays inside the plane chart it is authored in.
        for _ in range(8):
            angle = float(np.nextafter(angle, step))
            plane_uv = point_at(angle)[plane_offset : plane_offset + 2]
            if np.all(plane_uv >= plane_bounds[0]) and np.all(
                plane_uv <= plane_bounds[1]
            ):
                return angle
        return None

    def certified_branch(
        arc: _IsolineArc | None, /
    ) -> IntersectionCurve | SurfaceIntersectionResult:
        nonlocal profile_chart_lower, profile_chart_upper, trimmed
        trimmed = arc is not None
        end_nodes: tuple[tuple[int, float], tuple[int, float]] | None = None
        if arc is None:
            theta_lower, theta_upper = angle_lower, angle_upper
        else:
            start = inner_angle(arc.start_upper, np.inf)
            end = inner_angle(arc.end_lower, -np.inf)
            if (
                start is None
                or end is None
                or not (arc.across_seam or start < end)
                or (arc.across_seam and not (end < angle_upper and angle_lower < start))
            ):
                return _unresolved_surface(first, second, "continuation")
            theta_lower, theta_upper = start, end
            end_nodes = (
                (plane_offset + arc.start_axis, arc.start_value),
                (plane_offset + arc.end_axis, arc.end_value),
            )
        pad = initial_pad
        while True:
            profile_chart_lower = max(profile_lower, parameter_lower - pad)
            profile_chart_upper = min(profile_upper, parameter_upper + pad)
            curve = _angular_surface_branch(
                first,
                second,
                chart_axis,
                theta_lower,
                theta_upper,
                policy,
                point_at,
                box_for,
                closed=arc is None,
                start_kind="closed" if arc is None else "boundary",
                end_kind="closed" if arc is None else "boundary",
                end_nodes=end_nodes,
                seam=(
                    (angle_lower, angle_upper)
                    if arc is not None and arc.across_seam
                    else None
                ),
            )
            if curve is not None:
                return curve
            if pad >= maximum_pad:
                return _unresolved_surface(first, second, "resolution")
            next_pad = min(2.0 * pad, maximum_pad)
            if next_pad <= pad:
                return _unresolved_surface(first, second, "resolution")
            pad = next_pad

    if arcs == "full":
        return certified_branch(None)
    curves = []
    for arc in arcs:
        branch = certified_branch(arc)
        if isinstance(branch, SurfaceIntersectionResult):
            return branch
        curves.append(branch)
    return SurfaceIntersectionResult(curves=tuple(curves))


def _revolution_plane_line_circle(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    """Intersect a straight-profile revolution with a plane normal to its axis.

    The plane meets the profile line at one exact rational parameter, whose
    revolution isoline is the complete intersection circle.
    """
    from ._patches import ExactLineCurve, LineCurve, RevolutionSurface

    if isinstance(first.patch, RevolutionSurface) and isinstance(
        second.patch, PlanePatch
    ):
        revolution, plane = first.patch, second.patch
        revolution_region, plane_region, plane_is_first = first, second, False
    elif isinstance(first.patch, PlanePatch) and isinstance(
        second.patch, RevolutionSurface
    ):
        revolution, plane = second.patch, first.patch
        revolution_region, plane_region, plane_is_first = second, first, True
    else:
        return None
    profile = revolution.curve
    if isinstance(profile, LineCurve):
        origin_values, direction_values = profile.origin_exact, profile.direction_exact
    elif isinstance(profile, ExactLineCurve):
        origin_values = profile.first_exact
        direction_values = tuple(
            last - start
            for start, last in zip(profile.first_exact, profile.last_exact, strict=True)
        )
    else:
        return None
    if len(origin_values) != 3 or any(
        not isinstance(value, Fraction) for value in (*origin_values, *direction_values)
    ):
        return None
    origin = cast(tuple[Fraction, Fraction, Fraction], tuple(origin_values))
    direction = cast(tuple[Fraction, Fraction, Fraction], tuple(direction_values))
    axis_origin = rational_vector(np.asarray(revolution.axis_origin))
    axis = rational_vector(np.asarray(revolution.axis_direction))
    plane_origin = rational_vector(np.asarray(plane.origin))
    plane_normal = rational_cross(
        rational_vector(np.asarray(plane.first_axis)),
        rational_vector(np.asarray(plane.second_axis)),
    )
    if (
        rational_dot(axis, axis) != 1
        or all(value == 0 for value in plane_normal)
        or any(rational_cross(plane_normal, axis))
    ):
        return None
    height = rational_dot(
        tuple(plane_origin[index] - axis_origin[index] for index in range(3)), axis
    )
    rate = rational_dot(direction, axis)
    offset = height - rational_dot(
        tuple(origin[index] - axis_origin[index] for index in range(3)), axis
    )
    if rate == 0:
        return (
            SurfaceIntersectionResult()
            if offset != 0
            else _unresolved_surface(first, second, "singular")
        )
    box = np.asarray(revolution_region.parameter_box, dtype=np.float64)
    angle_lower, angle_upper = float(box[0, 0]), float(box[1, 0])
    period = revolution.periods[0]
    if (
        revolution.native_period_symbols()[0] != "two_pi"
        or period is None
        or Fraction.from_float(angle_upper) - Fraction.from_float(angle_lower)
        != Fraction.from_float(float(period))
    ):
        return _unresolved_surface(first, second, "continuation")
    profile_lower, profile_upper = float(box[0, 1]), float(box[1, 1])
    parameter = offset / rate
    if parameter < Fraction.from_float(profile_lower) or parameter > Fraction.from_float(
        profile_upper
    ):
        return SurfaceIntersectionResult()
    if parameter in (
        Fraction.from_float(profile_lower),
        Fraction.from_float(profile_upper),
    ):
        # A boundary isoline is a source edge, not an interior branch.
        return _unresolved_surface(first, second, "continuation")
    radial_vector = tuple(
        origin[index]
        + parameter * direction[index]
        - axis_origin[index]
        - height * axis[index]
        for index in range(3)
    )
    if all(value == 0 for value in radial_vector):
        return _unresolved_surface(first, second, "singular")
    branch = _revolution_plane_isoline(
        first,
        second,
        policy,
        plane_is_first=plane_is_first,
        plane_region=plane_region,
        axis_origin=axis_origin,
        axis=axis,
        height=height,
        radial_vector=radial_vector,
        parameter=float(parameter),
        parameter_enclosure=(fraction_down(parameter), fraction_up(parameter)),
        profile_range=(profile_lower, profile_upper),
        angle_range=(angle_lower, angle_upper),
    )
    if branch is None or isinstance(branch, SurfaceIntersectionResult):
        return branch
    return SurfaceIntersectionResult(curves=(branch,))


def _revolution_plane_generator_lines(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    """Intersect a straight cylinder revolution with a plane parallel to its axis.

    The plane meets the cylinder in at most two generator lines at the angles
    solving ``a cos t + b sin t = h`` exactly (rational ``a``, ``b``, ``h``).
    Each line is affine in the profile parameter on the plane chart, so its
    trim against the plane rectangle and the profile range is decided by
    certified interval comparisons. A trimmed end fixes the exact coordinate
    of the edge it ends on: a plane chart edge value or a profile range end.
    """
    from ._patches import ExactLineCurve, LineCurve, RevolutionSurface

    if isinstance(first.patch, RevolutionSurface) and isinstance(
        second.patch, PlanePatch
    ):
        revolution, plane = first.patch, second.patch
        revolution_region, plane_region, plane_is_first = first, second, False
    elif isinstance(first.patch, PlanePatch) and isinstance(
        second.patch, RevolutionSurface
    ):
        revolution, plane = second.patch, first.patch
        revolution_region, plane_region, plane_is_first = second, first, True
    else:
        return None
    profile = revolution.curve
    if isinstance(profile, LineCurve):
        origin_values, direction_values = profile.origin_exact, profile.direction_exact
    elif isinstance(profile, ExactLineCurve):
        origin_values = profile.first_exact
        direction_values = tuple(
            last - start
            for start, last in zip(profile.first_exact, profile.last_exact, strict=True)
        )
    else:
        return None
    if len(origin_values) != 3 or any(
        not isinstance(value, Fraction) for value in (*origin_values, *direction_values)
    ):
        return None
    origin = cast(tuple[Fraction, Fraction, Fraction], tuple(origin_values))
    direction = cast(tuple[Fraction, Fraction, Fraction], tuple(direction_values))
    axis_origin = rational_vector(np.asarray(revolution.axis_origin))
    axis = rational_vector(np.asarray(revolution.axis_direction))
    plane_origin = rational_vector(np.asarray(plane.origin))
    plane_normal = rational_cross(
        rational_vector(np.asarray(plane.first_axis)),
        rational_vector(np.asarray(plane.second_axis)),
    )
    if (
        rational_dot(axis, axis) != 1
        or all(value == 0 for value in plane_normal)
        or rational_dot(plane_normal, axis) != 0
        or any(rational_cross(direction, axis))
        or all(value == 0 for value in direction)
    ):
        return None
    relative = tuple(origin[index] - axis_origin[index] for index in range(3))
    axial = rational_dot(relative, axis)
    radial = tuple(relative[index] - axial * axis[index] for index in range(3))
    if all(value == 0 for value in radial):
        return None
    tangent = rational_cross(axis, radial)
    box = np.asarray(revolution_region.parameter_box, dtype=np.float64)
    angle_lower, angle_upper = float(box[0, 0]), float(box[1, 0])
    profile_lower, profile_upper = float(box[0, 1]), float(box[1, 1])
    period = revolution.periods[0]
    if (
        revolution.native_period_symbols()[0] != "two_pi"
        or period is None
        or Fraction.from_float(angle_upper) - Fraction.from_float(angle_lower)
        != Fraction.from_float(float(period))
    ):
        return _unresolved_surface(first, second, "continuation")
    # The plane through ``plane_origin`` holds the axis direction, so only the
    # radial point ``cos t * radial + sin t * tangent`` decides incidence.
    cosine_coefficient = rational_dot(plane_normal, radial)
    sine_coefficient = rational_dot(plane_normal, tangent)
    offset = rational_dot(
        plane_normal,
        tuple(
            plane_origin[index] - axis_origin[index] - axial * axis[index]
            for index in range(3)
        ),
    )
    squared_amplitude = cosine_coefficient**2 + sine_coefficient**2
    if offset**2 > squared_amplitude:
        return SurfaceIntersectionResult()
    if offset**2 == squared_amplitude:
        return _unresolved_surface(first, second, "singular")
    base_point = tuple(axis_origin[index] + axial * axis[index] for index in range(3))
    uv_base = _plane_chart_coordinates(plane, base_point, point=True)
    uv_radial = _plane_chart_coordinates(plane, radial, point=False)
    uv_tangent = _plane_chart_coordinates(plane, tangent, point=False)
    uv_rate = _plane_chart_coordinates(plane, direction, point=False)
    if uv_base is None or uv_radial is None or uv_tangent is None or uv_rate is None:
        return None
    plane_bounds = np.asarray(plane_region.parameter_box, dtype=np.float64)
    context = interval_context(80)

    def interval(value: ExactScalar, /) -> Any:
        lower, upper = coefficient_interval(value, bits=256)
        start = context.mpf(lower.numerator) / context.mpf(lower.denominator)
        stop = context.mpf(upper.numerator) / context.mpf(upper.denominator)
        return context.mpf([start.a, stop.b])

    def exact(value: float, /) -> Any:
        return interval(Fraction.from_float(value))

    amplitude = context.sqrt(interval(squared_amplitude))
    phase = context.atan2(interval(sine_coefficient), interval(cosine_coefficient))
    ratio = interval(offset) / amplitude
    half_width = context.atan2(context.sqrt(1 - ratio**2), ratio)
    window = exact(angle_lower)
    two_pi = 2 * context.pi
    revolution_offset = 2 if plane_is_first else 0
    plane_offset = 0 if plane_is_first else 2
    chart_axis = revolution_offset + 1
    curves = []
    for angle in (phase - half_width, phase + half_width):
        middle = 0.5 * float(angle.a) + 0.5 * float(angle.b)
        angle = angle - math.floor((middle - angle_lower) / math.tau) * two_pi
        if not (window.b < angle.a and angle.b < (window + two_pi).a):
            # A generator on the revolution seam is a source edge contact.
            return _unresolved_surface(first, second, "continuation")
        cosine, sine = context.cos(angle), context.sin(angle)
        line_base = [
            interval(uv_base[index])
            + interval(uv_radial[index]) * cosine
            + interval(uv_tangent[index]) * sine
            for index in range(2)
        ]
        # Each trim constraint is a lower or upper bound on the profile
        # parameter, tagged with the coordinate and exact value it fixes.
        lowers: list[tuple[Any, int, float]] = [
            (exact(profile_lower), chart_axis, profile_lower)
        ]
        uppers: list[tuple[Any, int, float]] = [
            (exact(profile_upper), chart_axis, profile_upper)
        ]
        outside = False
        for index in range(2):
            rate = uv_rate[index]
            for edge in range(2):
                value = float(plane_bounds[edge, index])
                if rate == 0:
                    gap = line_base[index] - exact(value)
                    if gap.a <= 0 <= gap.b:
                        return _unresolved_surface(first, second, "continuation")
                    if (gap.a > 0) == (edge == 1):
                        outside = True
                    continue
                bound = (exact(value) - line_base[index]) / interval(rate)
                is_lower = (edge == 0) == (rate > 0)
                (lowers if is_lower else uppers).append(
                    (bound, plane_offset + index, value)
                )
        if outside:
            continue

        def binding(
            candidates: list[tuple[Any, int, float]], largest: bool, /
        ) -> tuple[Any, int, float] | None:
            ordered = sorted(
                candidates, key=lambda item: float(item[0].a), reverse=largest
            )
            first_bound, second_bound = ordered[0], ordered[1]
            separated = (
                first_bound[0].a > second_bound[0].b
                if largest
                else first_bound[0].b < second_bound[0].a
            )
            return first_bound if separated else None

        start = binding(lowers, True)
        stop = binding(uppers, False)
        if start is None or stop is None:
            return _unresolved_surface(first, second, "continuation")
        if stop[0].b < start[0].a:
            continue
        if not start[0].b < stop[0].a:
            return _unresolved_surface(first, second, "continuation")
        angle_value = 0.5 * float(angle.a) + 0.5 * float(angle.b)
        # Chart boxes only need to contain the line and certify one root per
        # profile slice. The angle margin absorbs the Krawczyk coupling of the
        # constant angle to the chart's wide plane coordinates and the interval
        # spread of the source surfaces around the transcendental angle.
        margin = 2.0**-24 * max(1.0, abs(angle_value))
        angle_bounds = (
            float(np.nextafter(float(angle.a), -np.inf)) - margin,
            float(np.nextafter(float(angle.b), np.inf)) + margin,
        )
        base_bounds = [
            (
                float(np.nextafter(float(value.a), -np.inf)),
                float(np.nextafter(float(value.b), np.inf)),
            )
            for value in line_base
        ]
        rates = [float(uv_rate[index]) for index in range(2)]
        line_point: tuple[float, ...] = tuple(
            0.5 * (low + high) for low, high in base_bounds
        )

        def point_at(parameter: float, /) -> np.ndarray:
            plane_uv = tuple(
                line_point[index] + rates[index] * parameter for index in range(2)
            )
            revolution_uv = (angle_value, parameter)
            return np.asarray(
                (*plane_uv, *revolution_uv)
                if plane_is_first
                else (*revolution_uv, *plane_uv),
                dtype=np.float64,
            )

        def box_for(
            lower_parameter: float, upper_parameter: float, /
        ) -> tuple[np.ndarray, np.ndarray]:
            plane_lower, plane_upper = [], []
            for index in range(2):
                ends = [
                    base_bounds[index][side] + rates[index] * parameter
                    for side in range(2)
                    for parameter in (lower_parameter, upper_parameter)
                ]
                # Plane coordinates move with the angle, and those that advance
                # with the profile parameter get the chart's own advance again
                # as slack: the Krawczyk image of a parameter chart overshoots
                # the exact image range by about that much.
                plane_margin = 2.0**-20 * max(1.0, abs(min(ends)), abs(max(ends))) + abs(
                    rates[index]
                ) * (upper_parameter - lower_parameter)
                plane_lower.append(float(np.nextafter(min(ends) - plane_margin, -np.inf)))
                plane_upper.append(float(np.nextafter(max(ends) + plane_margin, np.inf)))
            revolution_lower = (angle_bounds[0], lower_parameter)
            revolution_upper = (angle_bounds[1], upper_parameter)
            if plane_is_first:
                return (
                    np.asarray((*plane_lower, *revolution_lower), dtype=np.float64),
                    np.asarray((*plane_upper, *revolution_upper), dtype=np.float64),
                )
            return (
                np.asarray((*revolution_lower, *plane_lower), dtype=np.float64),
                np.asarray((*revolution_upper, *plane_upper), dtype=np.float64),
            )

        def inner(bound: tuple[Any, int, float], step: float, /) -> float | None:
            # An exact profile end is its own representative; a plane edge
            # crossing is represented by the nearest float on the line's side.
            if bound[1] == chart_axis:
                return bound[2]
            parameter = float(bound[0].b if step > 0 else bound[0].a)
            for _ in range(8):
                parameter = float(np.nextafter(parameter, step))
                plane_uv = point_at(parameter)[plane_offset : plane_offset + 2]
                if (
                    np.all(plane_uv >= plane_bounds[0])
                    and np.all(plane_uv <= plane_bounds[1])
                    and profile_lower <= parameter <= profile_upper
                ):
                    return parameter
            return None

        start_parameter = inner(start, np.inf)
        stop_parameter = inner(stop, -np.inf)
        if (
            start_parameter is None
            or stop_parameter is None
            or not start_parameter < stop_parameter
        ):
            return _unresolved_surface(first, second, "continuation")
        curve = _angular_surface_branch(
            first,
            second,
            chart_axis,
            start_parameter,
            stop_parameter,
            policy,
            point_at,
            box_for,
            closed=False,
            start_kind="boundary",
            end_kind="boundary",
            end_nodes=((start[1], start[2]), (stop[1], stop[2])),
        )
        if curve is None:
            return _unresolved_surface(first, second, "resolution")
        curves.append(curve)
    return SurfaceIntersectionResult(curves=tuple(curves))


def _exact_cylinder_axis(
    region: SurfaceRegion, /
) -> (
    tuple[
        tuple[Fraction, Fraction, Fraction], tuple[Fraction, Fraction, Fraction], Fraction
    ]
    | None
):
    """Return the exact axis point, axis direction and radius of a circular cylinder."""
    patch = region.patch
    if not isinstance(patch, CylinderPatch):
        view = _revolution_cylinder_view(region)
        if view is None:
            return None
        patch = view[0].patch
        if not isinstance(patch, CylinderPatch):
            return None
    vectors = (
        patch.origin_exact,
        patch.axis_exact,
        patch.first_axis_exact,
        patch.second_axis_exact,
    )
    if any(
        vector is None
        or len(vector) != 3
        or not all(isinstance(value, Fraction) for value in vector)
        for vector in vectors
    ) or not isinstance(patch.radius_exact, Fraction):
        return None
    origin, axis, first_axis, second_axis = (
        cast(tuple[Fraction, Fraction, Fraction], vector) for vector in vectors
    )
    radius = patch.radius_exact
    # The section normal to the axis is a circle of the stated radius only for
    # an orthonormal radial frame orthogonal to the axis.
    if (
        radius <= 0
        or rational_dot(first_axis, first_axis) != 1
        or rational_dot(second_axis, second_axis) != 1
        or rational_dot(first_axis, second_axis) != 0
        or rational_dot(first_axis, axis) != 0
        or rational_dot(second_axis, axis) != 0
        or rational_dot(axis, axis) == 0
    ):
        return None
    return origin, axis, radius


def _parallel_cylinder_separation(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    """Prove two parallel circular cylinders disjoint from their exact axes.

    Parallel cylinders with axis distance ``d`` and radii ``r`` and ``s`` are
    disjoint exactly when ``d > r + s`` (separated) or ``d < |r - s|``
    (nested). Both comparisons are made on exact squared rationals, so the
    empty result is a proof for every parameter box of either cylinder.
    """
    del policy
    first_data = _exact_cylinder_axis(first)
    if first_data is None:
        return None
    second_data = _exact_cylinder_axis(second)
    if second_data is None:
        return None
    first_origin, axis, first_radius = first_data
    second_origin, second_axis, second_radius = second_data
    if any(rational_cross(axis, second_axis)):
        return None
    displacement = tuple(second_origin[index] - first_origin[index] for index in range(3))
    axial = rational_dot(displacement, axis) / rational_dot(axis, axis)
    radial = tuple(displacement[index] - axial * axis[index] for index in range(3))
    distance_squared = rational_dot(radial, radial)
    outer = (first_radius + second_radius) ** 2
    inner = (first_radius - second_radius) ** 2
    if distance_squared > outer or distance_squared < inner:
        return SurfaceIntersectionResult()
    return None


def _revolution_cylinder_view(
    region: SurfaceRegion, /
) -> tuple[SurfaceRegion, Fraction] | None:
    from .._rational import fraction_down, fraction_up
    from ._patches import ExactLineCurve, LineCurve, RevolutionSurface

    patch = region.patch
    if not isinstance(patch, RevolutionSurface):
        return None
    if isinstance(patch.curve, LineCurve):
        profile_origin = patch.curve.origin_exact
        profile_direction = patch.curve.direction_exact
    elif isinstance(patch.curve, ExactLineCurve):
        profile_origin = patch.curve.first_exact
        profile_direction = tuple(
            last - first
            for first, last in zip(
                patch.curve.first_exact, patch.curve.last_exact, strict=True
            )
        )
    else:
        return None
    if (
        len(profile_origin) != 3
        or len(profile_direction) != 3
        or any(
            not isinstance(value, Fraction)
            for value in (*profile_origin, *profile_direction)
        )
    ):
        return None
    profile_origin = cast(tuple[Fraction, ...], profile_origin)
    profile_direction = cast(tuple[Fraction, ...], profile_direction)
    axis_origin_values = rational_vector(np.asarray(patch.axis_origin))
    axis_values = rational_vector(np.asarray(patch.axis_direction))
    if any(
        not isinstance(value, Fraction) for value in (*axis_origin_values, *axis_values)
    ):
        return None
    axis_origin = axis_origin_values
    axis = axis_values

    def dot(first: tuple[Fraction, ...], second: tuple[Fraction, ...], /) -> Fraction:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    if dot(axis, axis) != 1:
        return None
    relative = tuple(profile_origin[index] - axis_origin[index] for index in range(3))
    axial_origin = dot(relative, axis)
    radial = tuple(relative[index] - axial_origin * axis[index] for index in range(3))
    axial_direction = dot(profile_direction, axis)
    radial_direction = tuple(
        profile_direction[index] - axial_direction * axis[index] for index in range(3)
    )
    if any(radial_direction) or axial_direction == 0:
        return None
    radius_squared = dot(radial, radial)
    if radius_squared <= 0:
        return None
    numerator_root = math.isqrt(radius_squared.numerator)
    denominator_root = math.isqrt(radius_squared.denominator)
    if (
        numerator_root * numerator_root != radius_squared.numerator
        or denominator_root * denominator_root != radius_squared.denominator
    ):
        return None
    radius = Fraction(numerator_root, denominator_root)
    first_axis = tuple(value / radius for value in radial)
    second_axis = (
        axis[1] * first_axis[2] - axis[2] * first_axis[1],
        axis[2] * first_axis[0] - axis[0] * first_axis[2],
        axis[0] * first_axis[1] - axis[1] * first_axis[0],
    )
    cylinder_origin = tuple(
        axis_origin[index] + axial_origin * axis[index] for index in range(3)
    )
    cylinder = CylinderPatch(
        np.asarray(tuple(float(value) for value in cylinder_origin)),
        np.asarray(tuple(float(value) for value in first_axis)),
        np.asarray(tuple(float(value) for value in second_axis)),
        np.asarray(tuple(float(value) for value in axis)),
        float(radius),
        origin_exact=cylinder_origin,
        first_axis_exact=first_axis,
        second_axis_exact=second_axis,
        axis_exact=axis,
        radius_exact=radius,
    )
    source_box = np.asarray(region.parameter_box, dtype=np.float64)
    source_profile_lower = Fraction.from_float(float(source_box[0, 1]))
    source_profile_upper = Fraction.from_float(float(source_box[1, 1]))
    heights = (
        axial_direction * source_profile_lower,
        axial_direction * source_profile_upper,
    )
    lower_height, upper_height = min(heights), max(heights)
    lower_numeric = fraction_down(lower_height)
    upper_numeric = fraction_up(upper_height)
    if (
        not np.isfinite(lower_numeric)
        or not np.isfinite(upper_numeric)
        or not (lower_numeric < upper_numeric)
    ):
        return None
    cylinder_region = SurfaceRegion(
        cylinder,
        np.asarray(
            (
                (source_box[0, 0], lower_numeric),
                (source_box[1, 0], upper_numeric),
            ),
            dtype=np.float64,
        ),
    )
    return cylinder_region, axial_direction


def _orthogonal_cylinder_family_intersection(
    family: _OrthogonalCylinderFamily, /
) -> SurfaceIntersectionResult:
    """Construct the exact two-loop analytic branch atlas for a cylinder family."""
    from ._intersection_curve import (
        IntersectionCurve,
        orthogonal_cylinder_closing_root_bounds,
        orthogonal_cylinder_graph_bounds,
        orthogonal_cylinder_graph_point,
        orthogonal_cylinder_seam_crossings,
        OrthogonalCylinderSeamCrossing,
    )

    regions = family.source_regions
    small_region = regions[family.small_index]
    small_box = np.asarray(small_region.parameter_box, dtype=np.float64)
    theta_lower, theta_upper = float(small_box[0, 0]), float(small_box[1, 0])
    phase_n = math.atan2(
        float(family.small_normal_coefficients[1]),
        float(family.small_normal_coefficients[0]),
    )
    phase_a = math.atan2(
        float(family.small_large_axis_coefficients[1]),
        float(family.small_large_axis_coefficients[0]),
    )
    breakpoints = {theta_lower, theta_upper}

    def add_periodic(base: float, period: float, /) -> None:
        first_turn = math.floor((theta_lower - base) / period) - 1
        last_turn = math.ceil((theta_upper - base) / period) + 1
        for turn in range(first_turn, last_turn + 1):
            value = base + turn * period
            if theta_lower < value < theta_upper:
                breakpoints.add(value)

    add_periodic(phase_n, math.pi)
    add_periodic(phase_a, math.pi)
    if abs(family.delta_n) < family.small_radius:
        root_angle = math.acos(float(-family.delta_n / family.small_radius))
        for phase in (phase_n + root_angle, phase_n - root_angle):
            add_periodic(phase, math.tau)
    ordered_breakpoints = tuple(sorted(breakpoints))
    if len(ordered_breakpoints) < 2:
        return _unresolved_surface(regions[0], regions[1], "continuation")

    large_angle_axis = 2 * family.large_index
    theta_axis = 2 * family.small_index
    source_patches = regions[0].patch, regions[1].patch
    large_box = np.asarray(regions[family.large_index].parameter_box, dtype=np.float64)
    large_lower, large_upper = float(large_box[0, 0]), float(large_box[1, 0])

    periods = np.asarray(
        tuple(
            0.0 if value is None else float(value)
            for value in (*source_patches[0].periods, *source_patches[1].periods)
        ),
        dtype=np.float64,
    )

    def sheet_reference(bounds: tuple[np.ndarray, np.ndarray], /) -> float:
        # The large angle is monotone on every chart: the breakpoints isolate
        # each extremum of the normal offset, and the large angle is monotone
        # in that offset on one branch. The chart midpoint therefore selects
        # the one authored large-source sheet that holds the whole chart.
        middle = 0.5 * float(bounds[0][large_angle_axis]) + 0.5 * float(
            bounds[1][large_angle_axis]
        )
        return middle - math.floor((middle - large_lower) / math.tau) * math.tau

    def meets_sheet(lower: np.ndarray, upper: np.ndarray, turn: int, /) -> bool:
        # A node seen through an exact deck turn must touch the authored large
        # sheet; otherwise its chart crosses a seam strictly inside.
        shift_lower, shift_upper = (
            period_scalar_bounds(PeriodOffset(Fraction(0), Fraction(turn)))
            if turn
            else (np.asarray(0.0), np.asarray(0.0))
        )
        return Fraction.from_float(float(lower[large_angle_axis])) + Fraction.from_float(
            float(shift_lower)
        ) <= Fraction.from_float(large_upper) and Fraction.from_float(
            float(upper[large_angle_axis])
        ) + Fraction.from_float(float(shift_upper)) >= Fraction.from_float(large_lower)

    curves = []
    for branch_sign in (-1, 1):
        crossings = orthogonal_cylinder_seam_crossings(family, branch_sign)
        if crossings is None:
            return _unresolved_surface(regions[0], regions[1], "continuation")
        start_on_seam = any(item.start for item in crossings)
        seam_crossings = tuple(item for item in crossings if not item.start)
        # A seam crossing becomes a node fixing the large angle at the exact
        # seam value; a float breakpoint that only represents the same closed
        # form is replaced by it.
        thetas = tuple(
            value
            for value in ordered_breakpoints
            if value in (theta_lower, theta_upper)
            or not any(
                item.theta_lower - 8.0 * math.ulp(value)
                <= value
                <= item.theta_upper + 8.0 * math.ulp(value)
                for item in seam_crossings
            )
        )
        layout: list[tuple[float, OrthogonalCylinderSeamCrossing | None]] = sorted(
            (
                *((value, None) for value in thetas),
                *(
                    (0.5 * item.theta_lower + 0.5 * item.theta_upper, item)
                    for item in seam_crossings
                ),
            ),
            key=lambda entry: entry[0],
        )
        chart_spans = tuple(
            (layout[index][0], layout[index + 1][0]) for index in range(len(layout) - 1)
        )
        chart_references = []
        for first, last in chart_spans:
            if not first < last:
                return _unresolved_surface(regions[0], regions[1], "continuation")
            bounds = orthogonal_cylinder_graph_bounds(
                family, first, last, branch_sign, 0.0
            )
            if bounds is None:
                return _unresolved_surface(regions[0], regions[1], "continuation")
            chart_references.append(sheet_reference(bounds))
        # Seam nodes take the sheet holding the exact seam value. Every other
        # node takes the sheet of the chart leaving it, and the closing node
        # that of the chart entering it. Seam crossings then appear as exact
        # per-node deck turns, and the closure turn of every source chart
        # follows from the start and closing nodes.
        node_references = [
            large_lower
            if crossing is not None or (node == 0 and start_on_seam)
            else chart_references[min(node, len(chart_spans) - 1)]
            for node, (_, crossing) in enumerate(layout)
        ]
        node_axes = np.asarray(
            [
                theta_axis if crossing is None else large_angle_axis
                for _, crossing in layout
            ],
            dtype=np.int32,
        )
        node_values = np.asarray(
            [theta if crossing is None else large_lower for theta, crossing in layout],
            dtype=np.float64,
        )
        node_lower: list[np.ndarray] = []
        node_upper: list[np.ndarray] = []
        node_points: list[np.ndarray] = []
        for node, (theta, crossing) in enumerate(layout):
            reference = node_references[node]
            point = orthogonal_cylinder_graph_point(family, theta, branch_sign, reference)
            if crossing is None:
                bounds = orthogonal_cylinder_graph_bounds(
                    family, theta, theta, branch_sign, reference
                )
            else:
                theta_range = (
                    float(
                        np.nextafter(np.nextafter(crossing.theta_lower, -np.inf), -np.inf)
                    ),
                    float(
                        np.nextafter(np.nextafter(crossing.theta_upper, np.inf), np.inf)
                    ),
                )
                bounds = orthogonal_cylinder_graph_bounds(
                    family, *theta_range, branch_sign, reference
                )
            if bounds is None or point is None:
                return _unresolved_surface(regions[0], regions[1], "continuation")
            lower, upper = bounds[0].copy(), bounds[1].copy()
            if node == len(layout) - 1:
                # The closing node is the start root's exact one-turn deck image,
                # not only the root on the float slice that represents it.
                closing = orthogonal_cylinder_closing_root_bounds(
                    family, float(node_values[0]), branch_sign, reference
                )
                if closing is None:
                    return _unresolved_surface(regions[0], regions[1], "continuation")
                lower = np.minimum(lower, closing[0])
                upper = np.maximum(upper, closing[1])
            lower = np.minimum(np.nextafter(lower, -np.inf), point)
            upper = np.maximum(np.nextafter(upper, np.inf), point)
            if crossing is not None:
                # The small angle is the free variable of a seam node's box.
                lower[theta_axis], upper[theta_axis] = theta_range
            node_lower.append(lower)
            node_upper.append(upper)
            node_points.append(point)

        charts = []
        node_shifts = np.zeros((len(node_values), 2, 4), dtype=np.int32)
        for chart, (first, last) in enumerate(chart_spans):
            reference = chart_references[chart]
            bounds = orthogonal_cylinder_graph_bounds(
                family, first, last, branch_sign, reference
            )
            if bounds is None:
                return _unresolved_surface(regions[0], regions[1], "continuation")
            endpoints = []
            for node, direction in ((chart, 1), (chart + 1, 0)):
                turn = round((reference - node_references[node]) / math.tau)
                deck = np.zeros((4,), dtype=np.int32)
                deck[large_angle_axis] = turn
                point = node_points[node] + deck * periods
                shifts = _periodic_endpoint_deck_shift(
                    point, node_lower[node], node_upper[node], source_patches
                )
                if (
                    shifts is None
                    or np.any(shifts != deck)
                    or not meets_sheet(node_lower[node], node_upper[node], turn)
                ):
                    return _unresolved_surface(regions[0], regions[1], "continuation")
                node_shifts[node, direction] = shifts
                endpoints.append(point)
            start, end = endpoints
            lower = np.nextafter(np.minimum(bounds[0], np.minimum(start, end)), -np.inf)
            upper = np.nextafter(np.maximum(bounds[1], np.maximum(start, end)), np.inf)
            if np.any(lower >= upper):
                return _unresolved_surface(regions[0], regions[1], "continuation")
            charts.append((start, end, lower, upper))
        try:
            curve = IntersectionCurve(
                regions[0],
                regions[1],
                chart_axes=np.full((len(charts),), theta_axis, dtype=np.int32),
                chart_start=np.stack([item[0] for item in charts]),
                chart_end=np.stack([item[1] for item in charts]),
                chart_lower=np.stack([item[2] for item in charts]),
                chart_upper=np.stack([item[3] for item in charts]),
                node_axes=node_axes,
                node_values=node_values,
                node_lower=np.stack(node_lower),
                node_upper=np.stack(node_upper),
                node_period_shifts=node_shifts,
                closed=True,
                start_kind="closed",
                end_kind="closed",
            )
        except ValueError:
            return _unresolved_surface(regions[0], regions[1], "continuation")
        curves.append(curve)
    return SurfaceIntersectionResult(curves=curves)


def _revolution_cylinder_intersection(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    try:
        family = orthogonal_cylinder_family(first, second)
    except _DisjointOrthogonalCylinderFamily:
        return SurfaceIntersectionResult()
    except _SingularOrthogonalCylinderFamily:
        return _unresolved_surface(first, second, "singular")
    if family is not None:
        return _orthogonal_cylinder_family_intersection(family)
    first_view = _revolution_cylinder_view(first)
    second_view = _revolution_cylinder_view(second)
    if first_view is None or second_view is None:
        return None
    return _orthogonal_cylinder_intersection(
        first_view[0],
        second_view[0],
        policy,
        source_regions=(first, second),
        source_slopes=(first_view[1], second_view[1]),
    )


def _coaxial_revolution_line_intersection(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    from ._intersection_curve import _PreparedPair
    from ._patches import ExactLineCurve, LineCurve, RevolutionSurface

    if not isinstance(first.patch, RevolutionSurface) or not isinstance(
        second.patch, RevolutionSurface
    ):
        return None
    first_surface, second_surface = first.patch, second.patch
    if (
        first_surface.native_period_symbols()[0] != "two_pi"
        or second_surface.native_period_symbols()[0] != "two_pi"
        or first_surface.periods[0] is None
        or second_surface.periods[0] != first_surface.periods[0]
    ):
        return None
    first_box = np.asarray(first.parameter_box, dtype=np.float64)
    second_box = np.asarray(second.parameter_box, dtype=np.float64)
    period = Fraction.from_float(float(first_surface.periods[0]))
    if (
        Fraction.from_float(float(first_box[1, 0]))
        - Fraction.from_float(float(first_box[0, 0]))
        != period
        or Fraction.from_float(float(second_box[1, 0]))
        - Fraction.from_float(float(second_box[0, 0]))
        != period
        or Fraction.from_float(float(first_box[0, 0]))
        != Fraction.from_float(float(second_box[0, 0]))
    ):
        return None

    def profile_line(
        curve: AbstractCurve, /
    ) -> (
        tuple[
            tuple[Fraction, Fraction, Fraction],
            tuple[Fraction, Fraction, Fraction],
        ]
        | None
    ):
        if isinstance(curve, LineCurve):
            origin, direction = curve.origin_exact, curve.direction_exact
        elif isinstance(curve, ExactLineCurve):
            origin = curve.first_exact
            direction = tuple(
                last - first
                for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
            )
        else:
            return None
        if (
            len(origin) != 3
            or len(direction) != 3
            or any(not isinstance(value, Fraction) for value in (*origin, *direction))
        ):
            return None
        return (
            cast(tuple[Fraction, Fraction, Fraction], tuple(origin)),
            cast(tuple[Fraction, Fraction, Fraction], tuple(direction)),
        )

    line_a, line_b = profile_line(first_surface.curve), profile_line(second_surface.curve)
    if line_a is None or line_b is None:
        return None
    origin_a, direction_a = line_a
    origin_b, direction_b = line_b
    axis_origin_values = rational_vector(np.asarray(first_surface.axis_origin))
    axis_values = rational_vector(np.asarray(first_surface.axis_direction))
    second_axis_origin = rational_vector(np.asarray(second_surface.axis_origin))
    second_axis = rational_vector(np.asarray(second_surface.axis_direction))
    if any(
        not isinstance(value, Fraction)
        for value in (
            *axis_origin_values,
            *axis_values,
            *second_axis_origin,
            *second_axis,
        )
    ):
        return None
    axis_origin = cast(tuple[Fraction, Fraction, Fraction], axis_origin_values)
    axis = cast(tuple[Fraction, Fraction, Fraction], axis_values)
    second_axis_origin = cast(tuple[Fraction, Fraction, Fraction], second_axis_origin)
    second_axis = cast(tuple[Fraction, Fraction, Fraction], second_axis)
    if (
        axis_origin != second_axis_origin
        or axis != second_axis
        or sum((value * value for value in axis), Fraction(0)) != 1
    ):
        return None

    def dot(
        first_vector: tuple[Fraction, ...], second_vector: tuple[Fraction, ...], /
    ) -> Fraction:
        return sum(
            (
                left * right
                for left, right in zip(first_vector, second_vector, strict=True)
            ),
            Fraction(0),
        )

    def cross(
        first_vector: tuple[Fraction, Fraction, Fraction],
        second_vector: tuple[Fraction, Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction, Fraction]:
        return (
            first_vector[1] * second_vector[2] - first_vector[2] * second_vector[1],
            first_vector[2] * second_vector[0] - first_vector[0] * second_vector[2],
            first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0],
        )

    def meridian_normal(
        origin: tuple[Fraction, Fraction, Fraction],
        direction: tuple[Fraction, Fraction, Fraction],
        /,
    ) -> tuple[Fraction, Fraction, Fraction] | None:
        relative = cast(
            tuple[Fraction, Fraction, Fraction],
            tuple(value - base for value, base in zip(origin, axis_origin, strict=True)),
        )
        axial = dot(relative, axis)
        radial = cast(
            tuple[Fraction, Fraction, Fraction],
            tuple(relative[i] - axial * axis[i] for i in range(3)),
        )
        if not any(radial):
            axial_direction = dot(direction, axis)
            radial = cast(
                tuple[Fraction, Fraction, Fraction],
                tuple(direction[i] - axial_direction * axis[i] for i in range(3)),
            )
        normal = cross(axis, radial)
        if not any(normal):
            return None
        if dot(normal, relative) != 0 or dot(normal, direction) != 0:
            return None
        return normal

    normal_a = meridian_normal(origin_a, direction_a)
    normal_b = meridian_normal(origin_b, direction_b)
    if normal_a is None or normal_b is None or any(cross(normal_a, normal_b)):
        return None

    def reflected_line(
        origin: tuple[Fraction, ...],
        direction: tuple[Fraction, ...],
        /,
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
        relative = tuple(
            value - base for value, base in zip(origin, axis_origin, strict=True)
        )
        axial = dot(relative, axis)
        reflected_origin = tuple(
            base + 2 * axial * axis[i] - relative[i] for i, base in enumerate(axis_origin)
        )
        axial_direction = dot(direction, axis)
        reflected_direction = tuple(
            2 * axial_direction * axis[i] - direction[i] for i in range(3)
        )
        return reflected_origin, reflected_direction

    def line_intersection(
        first_origin: tuple[Fraction, ...],
        first_direction: tuple[Fraction, ...],
        second_origin: tuple[Fraction, ...],
        second_direction: tuple[Fraction, ...],
        /,
    ) -> tuple[Fraction, Fraction, tuple[Fraction, ...]] | None:
        delta = tuple(second_origin[axis] - first_origin[axis] for axis in range(3))
        for first_axis in range(3):
            for second_axis in range(first_axis + 1, 3):
                determinant = (
                    first_direction[first_axis] * -second_direction[second_axis]
                    - first_direction[second_axis] * -second_direction[first_axis]
                )
                if determinant == 0:
                    continue
                first_parameter = (
                    delta[first_axis] * -second_direction[second_axis]
                    - delta[second_axis] * -second_direction[first_axis]
                ) / determinant
                second_parameter = (
                    first_direction[first_axis] * delta[second_axis]
                    - first_direction[second_axis] * delta[first_axis]
                ) / determinant
                first_point = tuple(
                    first_origin[axis] + first_parameter * first_direction[axis]
                    for axis in range(3)
                )
                second_point = tuple(
                    second_origin[axis] + second_parameter * second_direction[axis]
                    for axis in range(3)
                )
                if first_point == second_point:
                    return first_parameter, second_parameter, first_point
                return None
        return None

    profile_a = tuple(
        Fraction.from_float(float(first_box[index, 1])) for index in range(2)
    )
    profile_b = tuple(
        Fraction.from_float(float(second_box[index, 1])) for index in range(2)
    )
    branches: list[tuple[Fraction, Fraction, float]] = []
    signatures: set[tuple[Fraction, Fraction]] = set()
    for sign in (1, -1):
        origin_b_, direction_b_ = (
            (origin_b, direction_b) if sign > 0 else reflected_line(origin_b, direction_b)
        )
        isolated = line_intersection(origin_a, direction_a, origin_b_, direction_b_)
        if isolated is None:
            continue
        parameter_a, parameter_b, point = isolated
        if not (
            profile_a[0] <= parameter_a <= profile_a[1]
            and profile_b[0] <= parameter_b <= profile_b[1]
        ):
            continue
        relative = tuple(point[index] - axis_origin[index] for index in range(3))
        height = dot(relative, axis)
        radial = tuple(relative[index] - height * axis[index] for index in range(3))
        radius_squared = dot(radial, radial)
        if radius_squared == 0:
            return _unresolved_surface(first, second, "singular")
        signature = radius_squared, height
        if signature in signatures:
            return _unresolved_surface(first, second, "singular")
        signatures.add(signature)
        branches.append((parameter_a, parameter_b, 0.0 if sign > 0 else math.pi))
    if not branches:
        return SurfaceIntersectionResult()

    pair = SurfacePairSystem(first.patch, second.patch)

    prepared = _PreparedPair(pair)
    theta_lower = float(first_box[0, 0])
    theta_upper = float(first_box[1, 0])
    curves: list[IntersectionCurve] = []
    for parameter_a, parameter_b, phase in branches:
        radius_a = max(
            128.0 * np.finfo(np.float64).eps * (1.0 + abs(float(parameter_a))),
            min(
                policy.relative_probe,
                0.25 * float(profile_a[1] - profile_a[0]),
            ),
        )
        radius_b = max(
            128.0 * np.finfo(np.float64).eps * (1.0 + abs(float(parameter_b))),
            min(
                policy.relative_probe,
                0.25 * float(profile_b[1] - profile_b[0]),
            ),
        )
        angle_margin = policy.relative_resolution * max(1.0, abs(theta_upper), abs(phase))
        initial = [
            (
                theta_lower + (theta_upper - theta_lower) * index / 4.0,
                theta_lower + (theta_upper - theta_lower) * (index + 1) / 4.0,
            )
            for index in range(4)
        ]
        pending = list(initial)
        chart_data: list[
            tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        while pending:
            lower_angle, upper_angle = pending.pop()
            start = np.asarray(
                (
                    lower_angle,
                    float(parameter_a),
                    lower_angle + phase,
                    float(parameter_b),
                ),
                dtype=np.float64,
            )
            end = np.asarray(
                (
                    upper_angle,
                    float(parameter_a),
                    upper_angle + phase,
                    float(parameter_b),
                ),
                dtype=np.float64,
            )
            lower = np.asarray(
                (
                    lower_angle,
                    float(parameter_a) - radius_a,
                    lower_angle + phase - angle_margin,
                    float(parameter_b) - radius_b,
                ),
                dtype=np.float64,
            )
            upper = np.asarray(
                (
                    upper_angle,
                    float(parameter_a) + radius_a,
                    upper_angle + phase + angle_margin,
                    float(parameter_b) + radius_b,
                ),
                dtype=np.float64,
            )
            witness = krawczyk(
                prepared,
                lower[None],
                upper[None],
                parameter_axis=0,
            )
            if bool(witness.certified[0]):
                chart_data.append((lower_angle, upper_angle, start, end, lower, upper))
                continue
            middle = lower_angle + 0.5 * (upper_angle - lower_angle)
            if not lower_angle < middle < upper_angle:
                return _unresolved_surface(first, second, "resolution")
            pending.append((lower_angle, middle))
            pending.append((middle, upper_angle))
        chart_data.sort(key=lambda item: item[0])
        nodes = [theta_lower] + [segment[1] for segment in chart_data[:-1]]
        nodes.append(theta_lower)
        node_lower: list[np.ndarray] = []
        node_upper: list[np.ndarray] = []
        for node_index, node in enumerate(nodes):
            reference_angle = theta_lower if node_index == len(nodes) - 1 else node
            reference = np.asarray(
                (
                    reference_angle,
                    float(parameter_a),
                    reference_angle + phase,
                    float(parameter_b),
                ),
                dtype=np.float64,
            )
            box = _surface_branch_node_box(pair, reference, 0, reference_angle, policy)
            if box is None:
                return _unresolved_surface(first, second, "resolution")
            node_lower.append(box[0])
            node_upper.append(box[1])
        node_shifts = np.zeros((len(nodes), 2, 4), dtype=np.int32)
        node_shifts[-1, 0, 0] = 1
        node_shifts[-1, 0, 2] = 1
        try:
            curves.append(
                IntersectionCurve(
                    first,
                    second,
                    chart_axes=np.zeros((len(chart_data),), dtype=np.int32),
                    chart_start=np.stack([item[2] for item in chart_data]),
                    chart_end=np.stack([item[3] for item in chart_data]),
                    chart_lower=np.stack([item[4] for item in chart_data]),
                    chart_upper=np.stack([item[5] for item in chart_data]),
                    node_axes=np.zeros((len(nodes),), dtype=np.int32),
                    node_values=np.asarray(nodes, dtype=np.float64),
                    node_lower=np.stack(node_lower),
                    node_upper=np.stack(node_upper),
                    node_period_shifts=node_shifts,
                    closed=True,
                    start_kind="closed",
                    end_kind="closed",
                )
            )
        except (ValueError, np.linalg.LinAlgError):
            return _unresolved_surface(first, second, "continuation")
    return SurfaceIntersectionResult(curves=curves)


def _ruled_plane_cap_section(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    from ...discretization._exact_algebraic_geometry import rational_coefficient
    from ._patches import (
        ConicRootLinePCurve,
        PlanePatch,
        RuledSurface,
        SurfaceMappedCurve,
    )

    if isinstance(first.patch, RuledSurface) and isinstance(second.patch, PlanePatch):
        ruled_region, plane_region = first, second
        ruled_first = True
    elif isinstance(first.patch, PlanePatch) and isinstance(second.patch, RuledSurface):
        ruled_region, plane_region = second, first
        ruled_first = False
    else:
        return None
    ruled_patch, plane_patch = ruled_region.patch, plane_region.patch
    if not isinstance(ruled_patch, RuledSurface) or not isinstance(
        plane_patch, PlanePatch
    ):
        return None
    ruled, plane = ruled_patch, plane_patch
    first_rail, second_rail = ruled.first, ruled.second
    if not isinstance(first_rail, SurfaceMappedCurve) or not isinstance(
        second_rail, SurfaceMappedCurve
    ):
        return None
    rail_first, rail_second = first_rail, second_rail
    normal = rational_cross(
        rational_vector(np.asarray(plane.first_axis)),
        rational_vector(np.asarray(plane.second_axis)),
    )
    plane_origin = rational_vector(np.asarray(plane.origin))

    def dot(left: Sequence[ExactScalar], right: Sequence[ExactScalar], /) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    source_data = []
    for rail in (rail_first, rail_second):
        source, patch = rail.pcurve, rail.surface
        if not isinstance(patch, CylinderPatch) or not isinstance(
            source, ConicRootLinePCurve
        ):
            return None
        if (
            dot(normal, patch.first_axis_exact) != 0
            or dot(normal, patch.second_axis_exact) != 0
        ):
            return None
        speed = tuple(source.height_direction_exact * value for value in patch.axis_exact)
        origin = tuple(
            patch.origin_exact[axis]
            + source.height_origin_exact * patch.axis_exact[axis]
            - plane_origin[axis]
            for axis in range(3)
        )
        source_data.append((dot(normal, origin), dot(normal, speed)))
    if source_data[0][1] == 0 or source_data[0][1] != source_data[1][1]:
        return None
    try:
        source_parameter = rational_coefficient(
            -source_data[0][0] / source_data[0][1],
            role="ruled cap plane source parameter",
        )
        second_parameter = rational_coefficient(
            -source_data[1][0] / source_data[1][1],
            role="ruled cap plane source parameter",
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if source_parameter != second_parameter:
        return None
    ruled_box = np.asarray(ruled_region.parameter_box, dtype=np.float64)
    source_lower = Fraction.from_float(float(ruled_box[0, 0]))
    source_upper = Fraction.from_float(float(ruled_box[1, 0]))
    if not source_lower <= source_parameter <= source_upper:
        return SurfaceIntersectionResult()
    first_v, last_v = float(ruled_box[0, 1]), float(ruled_box[1, 1])
    if not first_v < last_v:
        return None

    axes = np.column_stack(
        (
            np.asarray(plane.first_axis, dtype=np.float64),
            np.asarray(plane.second_axis, dtype=np.float64),
        )
    )
    gram = axes.T @ axes
    try:
        inverse_gram = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        return None
    origin = np.asarray(plane.origin, dtype=np.float64)
    fixed = float(source_parameter)
    pair = SurfacePairSystem(first.patch, second.patch)
    from ._intersection_curve import (
        _exact_ruled_plane_pair_prepared,
        _PreparedPair,
    )

    prepared = _exact_ruled_plane_pair_prepared(first.patch, second.patch)
    if prepared is None:
        prepared = _PreparedPair(pair)
    chart_axis = 1 if ruled_first else 3

    def chart_parameters(v: float, /) -> np.ndarray:
        point = np.asarray(
            ruled.evaluate(jnp.asarray((fixed, v), dtype=jnp.float64)),
            dtype=np.float64,
        )
        uv = inverse_gram @ axes.T @ (point - origin)
        return (
            np.asarray((fixed, v, *uv), dtype=np.float64)
            if ruled_first
            else np.asarray((*uv, fixed, v), dtype=np.float64)
        )

    pending = [(first_v, last_v)]
    charts: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    fixed_radius = max(
        policy.relative_resolution * max(1.0, abs(fixed)),
        128.0 * np.finfo(np.float64).eps * max(1.0, abs(fixed)),
    )
    fixed_box_lower = fixed - fixed_radius
    fixed_box_upper = fixed + fixed_radius
    while pending:
        start_v, end_v = pending.pop()
        start = chart_parameters(start_v)
        end = chart_parameters(end_v)
        uv_start = start[2:] if ruled_first else start[:2]
        uv_end = end[2:] if ruled_first else end[:2]
        scale = max(1.0, float(np.max(np.abs(uv_start))), float(np.max(np.abs(uv_end))))
        padding = max(
            policy.relative_resolution * scale,
            128.0 * np.finfo(np.float64).eps * scale,
        )
        uv_lower = np.minimum(uv_start, uv_end) - padding
        uv_upper = np.maximum(uv_start, uv_end) + padding
        if ruled_first:
            lower = np.asarray((fixed_box_lower, start_v, *uv_lower), dtype=np.float64)
            upper = np.asarray((fixed_box_upper, end_v, *uv_upper), dtype=np.float64)
        else:
            lower = np.asarray((*uv_lower, fixed_box_lower, start_v), dtype=np.float64)
            upper = np.asarray((*uv_upper, fixed_box_upper, end_v), dtype=np.float64)
        witness = krawczyk(
            prepared,
            lower[None],
            upper[None],
            parameter_axis=chart_axis,
        )
        if bool(witness.certified[0]):
            charts.append((start_v, end_v, start, end, lower, upper))
            continue
        middle = start_v + 0.5 * (end_v - start_v)
        if not start_v < middle < end_v:
            return _unresolved_surface(first, second, "resolution")
        pending.append((middle, end_v))
        pending.append((start_v, middle))
    charts.sort(key=lambda chart: chart[0])
    nodes = [first_v] + [chart[1] for chart in charts]
    node_lower: list[np.ndarray] = []
    node_upper: list[np.ndarray] = []
    for value in nodes:
        reference = chart_parameters(value)
        box = _surface_branch_node_box(pair, reference, chart_axis, value, policy)
        if box is None:
            return _unresolved_surface(first, second, "resolution")
        node_lower.append(box[0])
        node_upper.append(box[1])
    try:
        curve = IntersectionCurve(
            first,
            second,
            chart_axes=np.full((len(charts),), chart_axis, dtype=np.int32),
            chart_start=np.stack([chart[2] for chart in charts]),
            chart_end=np.stack([chart[3] for chart in charts]),
            chart_lower=np.stack([chart[4] for chart in charts]),
            chart_upper=np.stack([chart[5] for chart in charts]),
            node_axes=np.full((len(nodes),), chart_axis, dtype=np.int32),
            node_values=np.asarray(nodes, dtype=np.float64),
            node_lower=np.stack(node_lower),
            node_upper=np.stack(node_upper),
            node_period_shifts=np.zeros((len(nodes), 2, 4), dtype=np.int32),
            closed=False,
            start_kind="boundary",
            end_kind="boundary",
        )
    except (ValueError, np.linalg.LinAlgError):
        return _unresolved_surface(first, second, "continuation")
    return SurfaceIntersectionResult(curves=(curve,))


def _placed_surface_result(
    result: SurfaceIntersectionResult,
    first: object,
    second: object,
    /,
) -> SurfaceIntersectionResult:
    from ._exact import gram_deviation_bound, square_root_interval
    from ._placed import PlacedSurface

    assert isinstance(first, PlacedSurface) and isinstance(second, PlacedSurface)
    rotation = np.asarray(first.rotation, dtype=np.float64)
    translation = np.asarray(first.translation, dtype=np.float64)
    rotation_exact = tuple(rational_vector(row) for row in rotation)
    translation_exact = rational_vector(translation)
    gram_error = Fraction.from_float(gram_deviation_bound(rotation))
    deviation_scale = square_root_interval(Fraction(1) + gram_error)[1]
    curves = []
    for curve in result.curves:
        placed_first = SurfaceRegion(
            PlacedSurface(curve.first.patch, rotation, translation),
            curve.first.parameter_box,
        )
        placed_second = SurfaceRegion(
            PlacedSurface(curve.second.patch, rotation, translation),
            curve.second.parameter_box,
        )
        curves.append(
            IntersectionCurve(
                placed_first,
                placed_second,
                chart_axes=curve.chart_axes,
                chart_start=curve.chart_start,
                chart_end=curve.chart_end,
                chart_lower=curve.chart_lower,
                chart_upper=curve.chart_upper,
                node_axes=curve.node_axes,
                node_values=curve.node_values,
                node_lower=curve.node_lower,
                node_upper=curve.node_upper,
                node_period_shifts=curve.node_period_shifts,
                closed=curve.closed,
                start_kind=curve.start_kind,
                end_kind=curve.end_kind,
            )
        )
    points = []
    for event in result.points:
        source_point = np.asarray(event.point, dtype=np.float64)
        point = source_point @ rotation.T + translation
        exact = (
            None
            if event.point_exact is None
            else tuple(
                translation_exact[i]
                + sum(
                    (rotation_exact[i][j] * event.point_exact[j] for j in range(3)),
                    Fraction(0),
                )
                for i in range(3)
            )
        )
        points.append(
            ParametricIntersectionPoint(
                kind=event.kind,
                certificate=event.certificate,
                parameters=event.parameters,
                parameter_lower=event.parameter_lower,
                parameter_upper=event.parameter_upper,
                point=point,
                point_exact=exact,
                gap_bound=deviation_scale * event.gap_bound,
                condition_estimate=event.condition_estimate,
                branch_count=event.branch_count,
                root_id=canonical_fingerprint(
                    {
                        "kind": "placed-intersection-root",
                        "source_root": event.root_id,
                        "rotation": array_tree_fingerprint(first.rotation),
                        "translation": array_tree_fingerprint(first.translation),
                    }
                ),
            )
        )
    return SurfaceIntersectionResult(
        curves=curves,
        points=points,
        coincident=result.coincident,
        unresolved=result.unresolved,
    )


def intersect_surface_regions(
    first: SurfaceRegion,
    second: SurfaceRegion,
    /,
    *,
    policy: ParametricIntersectionPolicy | None = None,
) -> SurfaceIntersectionResult:
    if not isinstance(first, SurfaceRegion) or not isinstance(second, SurfaceRegion):
        raise TypeError("intersect_surface_regions requires two SurfaceRegion values.")
    selected = ParametricIntersectionPolicy() if policy is None else policy
    from ._placed import PlacedSurface, same_source_pose

    if isinstance(first.patch, PlacedSurface) and isinstance(second.patch, PlacedSurface):
        if not same_source_pose(first.patch, second.patch):
            return _unresolved_surface(first, second, "continuation")
        source_result = intersect_surface_regions(
            SurfaceRegion(first.patch.definition, first.parameter_box),
            SurfaceRegion(second.patch.definition, second.parameter_box),
            policy=selected,
        )
        return _placed_surface_result(source_result, first.patch, second.patch)
    if isinstance(first.patch, PlacedSurface) or isinstance(second.patch, PlacedSurface):
        return _unresolved_surface(first, second, "continuation")
    if isinstance(first.patch, PlanePatch) and isinstance(second.patch, PlanePatch):
        coincident = _coincident_planes(first, second)
        if coincident is False:
            return SurfaceIntersectionResult()
        if coincident is not None:
            return SurfaceIntersectionResult(coincident=(coincident,))
    if isinstance(first.patch, PlanePatch) and isinstance(second.patch, PlanePatch):
        planar = _plane_plane_intersection(first, second, selected)
        if planar is not None:
            return planar
    revolution_plane = _revolution_plane_conic_circles(first, second, selected)
    if revolution_plane is not None:
        return revolution_plane
    revolution_line_plane = _revolution_plane_line_circle(first, second, selected)
    if revolution_line_plane is not None:
        return revolution_line_plane
    generator_lines = _revolution_plane_generator_lines(first, second, selected)
    if generator_lines is not None:
        return generator_lines
    parallel_cylinders = _parallel_cylinder_separation(first, second, selected)
    if parallel_cylinders is not None:
        return parallel_cylinders
    revolution_cylinders = _revolution_cylinder_intersection(first, second, selected)
    if revolution_cylinders is not None:
        return revolution_cylinders
    revolutions = _coaxial_revolution_line_intersection(first, second, selected)
    if revolutions is not None:
        return revolutions
    tangent = _sphere_sphere_contact(first, second, selected)
    if tangent is not None:
        return SurfaceIntersectionResult(points=(tangent,))
    plane_contact = _plane_sphere_contact(first, second, selected)
    if plane_contact is not None:
        return SurfaceIntersectionResult(points=(plane_contact,))
    plane_circle = _parallel_plane_sphere_circle(first, second, selected)
    if plane_circle is not None:
        return plane_circle
    ruled_cap = _ruled_plane_cap_section(first, second, selected)
    if ruled_cap is not None:
        return ruled_cap
    plane_cylinder = _plane_cylinder_angular_branch(first, second, selected)
    if plane_cylinder is not None:
        return plane_cylinder
    torus_planes = _horizontal_torus_plane_circles(first, second, selected)
    if torus_planes is not None:
        return torus_planes
    cylinders = _orthogonal_cylinder_intersection(first, second, selected)
    if cylinders is not None:
        return cylinders
    from ._tspline import TSplineSurfacePatch

    if isinstance(first.patch, TSplineSurfacePatch) or isinstance(
        second.patch, TSplineSurfacePatch
    ):
        from ._tspline_intersection import intersect_tspline_surface_regions

        tspline = intersect_tspline_surface_regions(
            first,
            second,
            policy=selected,
        )
        if tspline is not None:
            return tspline
    try:
        first_box = np.asarray(
            first.patch.bounding_box(first.parameter_box), dtype=np.float64
        )
        second_box = np.asarray(
            second.patch.bounding_box(second.parameter_box), dtype=np.float64
        )
    except NotImplementedError:
        pass
    else:
        if (
            first_box.shape == (2, 3)
            and second_box.shape == (2, 3)
            and np.all(np.isfinite(first_box))
            and np.all(np.isfinite(second_box))
            and np.any((first_box[1] < second_box[0]) | (second_box[1] < first_box[0]))
        ):
            return SurfaceIntersectionResult()
    return _unresolved_surface(first, second)


__all__ = [
    *__all__,
    "PlaneCylinderConicSection",
    "exact_plane_cylinder_conic_section",
    "intersect_surface_regions",
]
