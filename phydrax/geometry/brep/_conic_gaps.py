#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified gaps for binary64 native conics and their extrusion boundaries.

Trigonometric enclosures use the interval substrate's conditional libm
contract. All algebra after that enclosure is rational: translation cancellation
never loses the small authored endpoint discrepancy. No vertex is moved.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._interval_enclosure import interval_cos, interval_sin
from .._rational import fraction_down, fraction_up, rational_vector
from ._carrier_pieces import carrier_payload
from ._exact import square_root_interval
from ._intersection import NativePeriodEndpoint
from ._intersection_curve import affine_curve_coefficients, exact_curve_parameter_point
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
)


type _Vector = tuple[Fraction, ...]
type _Harmonics = tuple[_Vector, _Vector, _Vector]
type _Interval = tuple[Fraction, Fraction]


def _conic_coefficients(curve: CircleCurve | EllipseCurve, /) -> _Harmonics:
    """Rational constant, cosine and sine coefficients of the authored carrier."""
    if isinstance(curve, CircleCurve):
        first_radius = second_radius = Fraction(float(curve.radius))
    else:
        first_radius = Fraction(float(curve.first_radius))
        second_radius = Fraction(float(curve.second_radius))
    return (
        rational_vector(np.asarray(curve.center)),
        tuple(
            first_radius * value
            for value in rational_vector(np.asarray(curve.first_axis))
        ),
        tuple(
            second_radius * value
            for value in rational_vector(np.asarray(curve.second_axis))
        ),
    )


def _trig_bounds(first: float, last: float, /) -> tuple[_Interval, _Interval]:
    """Retain the exact zero-angle identity; otherwise use outward interval trig."""
    if first == last == 0.0:
        return (Fraction(1), Fraction(1)), (Fraction(0), Fraction(0))
    angle = (
        np.asarray(min(first, last), dtype=np.float64),
        np.asarray(max(first, last), dtype=np.float64),
    )
    cosine, sine = interval_cos(angle), interval_sin(angle)
    return (
        (Fraction(float(cosine[0])), Fraction(float(cosine[1]))),
        (Fraction(float(sine[0])), Fraction(float(sine[1]))),
    )


def _scaled_interval(coefficient: Fraction, bounds: _Interval, /) -> _Interval:
    first, last = coefficient * bounds[0], coefficient * bounds[1]
    return min(first, last), max(first, last)


def _harmonic_bounds(
    coefficients: _Harmonics,
    cosine: _Interval,
    sine: _Interval,
    /,
) -> tuple[_Interval, ...]:
    result: list[_Interval] = []
    for constant, first, second in zip(*coefficients, strict=True):
        a, b = _scaled_interval(first, cosine), _scaled_interval(second, sine)
        result.append((constant + a[0] + b[0], constant + a[1] + b[1]))
    return tuple(result)


def _squared_upper(bounds: tuple[_Interval, ...], /) -> Fraction:
    return sum((max(first * first, last * last) for first, last in bounds), Fraction(0))


def _symbolic_endpoint_squared(
    curve: AbstractCurve,
    root: NativePeriodEndpoint,
    parameter: float,
    vertex: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Bind a native periodic atom and compare its rational source image."""
    if not isinstance(root, NativePeriodEndpoint):
        raise ValueError(
            "endpoint_gap_bounds.endpoint_roots must use native periodic bindings; got unsupported root"
        )
    if carrier_payload(root.carrier) != carrier_payload(curve):
        raise ValueError(
            "endpoint_gap_bounds.endpoint_roots must bind the unchanged source carrier; got mismatched carrier"
        )
    lower, upper = root.parameter_enclosure(budget=budget)
    if not lower <= parameter <= upper:
        raise ValueError(
            "endpoint_gap_bounds.parameters must represent their native source atoms; got inconsistent endpoint"
        )
    point = exact_curve_parameter_point(
        curve, root.exact_parameter, coordinate_budget=budget
    )
    if point is None:
        raise ValueError(
            "endpoint_gap_bounds.endpoint_roots must have proved rational source images; got unsupported native atom"
        )
    squared = Fraction(0)
    for value, target in zip(point, rational_vector(vertex), strict=True):
        if not isinstance(value, Fraction):
            raise ValueError(
                "endpoint_gap_bounds.endpoint_roots must have rational source coordinates; got unsupported native atom"
            )
        squared += (value - target) ** 2
    return squared


def _endpoint_squared(
    curve: CircleCurve | EllipseCurve | LineCurve | BSplineCurve,
    parameter: float,
    vertex: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    root: NativePeriodEndpoint | None = None,
) -> Fraction:
    if root is not None:
        return _symbolic_endpoint_squared(curve, root, parameter, vertex, budget=budget)
    point = rational_vector(vertex)
    if isinstance(curve, BSplineCurve):
        exact = exact_curve_parameter_point(
            curve, Fraction(parameter), coordinate_budget=budget
        )
        if exact is None:
            raise ValueError(
                "endpoint_gap_bounds.spline must have a proved rational source endpoint; got unsupported parameter"
            )
        squared = Fraction(0)
        for value, target in zip(exact, point, strict=True):
            if not isinstance(value, Fraction):
                raise ValueError(
                    "endpoint_gap_bounds.spline must have rational endpoint coordinates; got unsupported source expression"
                )
            squared += (value - target) ** 2
        return squared
    if isinstance(curve, LineCurve):
        origin = rational_vector(np.asarray(curve.origin))
        direction = rational_vector(np.asarray(curve.direction))
        return sum(
            (
                (a + Fraction(parameter) * b - c) ** 2
                for a, b, c in zip(origin, direction, point, strict=True)
            ),
            Fraction(0),
        )
    center, cosine, sine = _conic_coefficients(curve)
    residual = (tuple(a - b for a, b in zip(center, point, strict=True)), cosine, sine)
    return _squared_upper(_harmonic_bounds(residual, *_trig_bounds(parameter, parameter)))


def endpoint_gap_bounds(
    curve: AbstractCurve,
    parameters: tuple[float, float],
    vertices: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
    endpoint_roots: tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None] = (
        None,
        None,
    ),
) -> tuple[float, float]:
    """Outward endpoint-distance upper bounds against two unchanged source vertices.

    Literal native circle, ellipse and affine line coefficients, and independently
    proved rational B-spline source endpoints, are admitted. The reservation covers
    trig intervals and rational squared distances; unsupported carriers refuse.
    """
    if type(curve) not in (CircleCurve, EllipseCurve, LineCurve, BSplineCurve):
        raise ValueError(
            "endpoint_gap_bounds.curve must have native endpoint coefficients; got unsupported carrier"
        )
    if not isinstance(curve, (CircleCurve, EllipseCurve, LineCurve, BSplineCurve)):
        raise ValueError(
            "endpoint_gap_bounds.curve must have native endpoint coefficients; got unsupported carrier"
        )
    if len(parameters) != 2 or not all(np.isfinite(value) for value in parameters):
        raise ValueError(
            "endpoint_gap_bounds.parameters must contain two finite binary64 values; got invalid endpoints"
        )
    if vertices.shape != (2, curve.ambient_dimension) or not np.all(
        np.isfinite(vertices)
    ):
        raise ValueError(
            "endpoint_gap_bounds.vertices must contain two finite source points of the carrier dimension; got invalid vertices"
        )
    if len(endpoint_roots) != 2:
        raise ValueError(
            "endpoint_gap_bounds.endpoint_roots must contain two source endpoint bindings; got invalid rows"
        )
    with budget.temporary_scope():
        budget.reserve(
            512 * curve.ambient_dimension, 8192 + 4096 * curve.ambient_dimension
        )
        result = tuple(
            square_root_interval(
                _endpoint_squared(curve, parameter, vertex, budget=budget, root=root)
            )[1]
            for parameter, vertex, root in zip(
                parameters, vertices, endpoint_roots, strict=True
            )
        )
    if not all(np.isfinite(value) for value in result):
        raise ValueError(
            "endpoint_gap_bounds.distances must have finite certified bounds; got unbounded distance"
        )
    return result[0], result[1]


def _plane_coefficients(patch: PlanePatch, coefficients: _Harmonics, /) -> _Harmonics:
    origin = rational_vector(np.asarray(patch.origin))
    axes = tuple(
        rational_vector(np.asarray(axis))
        for axis in (patch.first_axis, patch.second_axis)
    )
    images = tuple(
        tuple(
            (origin[dimension] if component == 0 else Fraction(0))
            + sum(
                (axes[axis][dimension] * vector[axis] for axis in range(2)), Fraction(0)
            )
            for dimension in range(3)
        )
        for component, vector in enumerate(coefficients)
    )
    return images[0], images[1], images[2]


def _extrusion_coefficients(
    patch: AbstractSurfacePatch,
    /,
) -> tuple[_Harmonics, _Vector] | None:
    if isinstance(patch, CylinderPatch):
        radius = Fraction(float(patch.radius))
        return (
            (
                rational_vector(np.asarray(patch.origin)),
                tuple(
                    radius * value
                    for value in rational_vector(np.asarray(patch.first_axis))
                ),
                tuple(
                    radius * value
                    for value in rational_vector(np.asarray(patch.second_axis))
                ),
            ),
            rational_vector(np.asarray(patch.axis)),
        )
    if isinstance(patch, ExtrusionSurface) and type(patch.curve) in (
        CircleCurve,
        EllipseCurve,
    ):
        if isinstance(patch.curve, (CircleCurve, EllipseCurve)):
            return _conic_coefficients(patch.curve), rational_vector(
                np.asarray(patch.direction)
            )
    return None


def _harmonic_difference(first: _Harmonics, second: _Harmonics, /) -> _Harmonics:
    differences = tuple(
        tuple(a - b for a, b in zip(actual, expected, strict=True))
        for actual, expected in zip(first, second, strict=True)
    )
    return differences[0], differences[1], differences[2]


def _extrusion_gap_squared(
    curve: CircleCurve | EllipseCurve | LineCurve,
    pcurve: AbstractCurve,
    patch: AbstractSurfacePatch,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    source = _extrusion_coefficients(patch)
    uv = affine_curve_coefficients(pcurve, np.zeros((2,), dtype=np.float64))
    if source is None or uv is None or len(uv[0]) != 2:
        return None
    coefficients, direction = source
    if isinstance(curve, LineCurve):
        # A seam has one constant angular source parameter. Its discrepancy is
        # affine in height; convexity certifies the whole seam at its endpoints.
        if uv[1][0] != 0:
            return None
        base = _harmonic_bounds(
            coefficients,
            *_trig_bounds(fraction_down(uv[0][0]), fraction_up(uv[0][0])),
        )
        origin = rational_vector(np.asarray(curve.origin))
        slope = rational_vector(np.asarray(curve.direction))
        return max(
            _squared_upper(
                tuple(
                    (
                        bounds[0]
                        + (uv[0][1] + Fraction(t) * uv[1][1]) * axis
                        - a
                        - Fraction(t) * b,
                        bounds[1]
                        + (uv[0][1] + Fraction(t) * uv[1][1]) * axis
                        - a
                        - Fraction(t) * b,
                    )
                    for bounds, axis, a, b in zip(
                        base, direction, origin, slope, strict=True
                    )
                )
            )
            for t in (first, last)
        )
    if uv[0][0] != 0 or uv[1] != (Fraction(1), Fraction(0)):
        return None
    center, cosine, sine = coefficients
    lifted = (
        tuple(
            value + uv[0][1] * axis for value, axis in zip(center, direction, strict=True)
        ),
        cosine,
        sine,
    )
    difference = _harmonic_difference(_conic_coefficients(curve), lifted)
    if not any(any(row) for row in difference):
        return Fraction(0)
    return _squared_upper(_harmonic_bounds(difference, *_trig_bounds(first, last)))


def _same_spline_basis(first: BSplineCurve, second: BSplineCurve, /) -> bool:
    """Authenticate a common positive rational partition of unity."""
    return (
        first.degree == second.degree
        and first.control_points.shape[0] == second.control_points.shape[0]
        and np.array_equal(np.asarray(first.knots), np.asarray(second.knots))
        and np.array_equal(np.asarray(first.weights), np.asarray(second.weights))
        and bool(np.all(np.asarray(first.weights) > 0.0))
    )


def _affine_squared(
    constant: _Vector, slope: _Vector, first: float, last: float, /
) -> Fraction:
    """The squared norm of an affine discrepancy attains its maximum at an end."""
    return max(
        sum(
            ((a + Fraction(t) * b) ** 2 for a, b in zip(constant, slope, strict=True)),
            Fraction(0),
        )
        for t in (first, last)
    )


def _spline_plane_gap_squared(
    curve: BSplineCurve, pcurve: BSplineCurve, patch: PlanePatch, /
) -> Fraction | None:
    if pcurve.ambient_dimension != 2 or not _same_spline_basis(curve, pcurve):
        return None
    origin = rational_vector(np.asarray(patch.origin))
    axes = tuple(
        rational_vector(np.asarray(axis))
        for axis in (patch.first_axis, patch.second_axis)
    )
    # Positive common rational basis functions form a convex combination of
    # these rational control discrepancies over every supported source interval.
    return max(
        sum(
            (
                (
                    origin[dimension]
                    + sum(
                        (
                            axes[axis][dimension] * Fraction(float(uv[axis]))
                            for axis in range(2)
                        ),
                        Fraction(0),
                    )
                    - Fraction(float(xyz[dimension]))
                )
                ** 2
                for dimension in range(3)
            ),
            Fraction(0),
        )
        for xyz, uv in zip(
            np.asarray(curve.control_points),
            np.asarray(pcurve.control_points),
            strict=True,
        )
    )


def _spline_extrusion_control_gap_squared(
    curve: BSplineCurve,
    basis: BSplineCurve,
    direction: _Vector,
    uv: tuple[_Vector, _Vector],
    /,
) -> Fraction | None:
    """Bound a translated same-basis spline by its control discrepancy hull."""
    if (
        uv[0][0] != 0
        or uv[1] != (Fraction(1), Fraction(0))
        or not _same_spline_basis(curve, basis)
    ):
        return None
    return max(
        sum(
            (
                (a + uv[0][1] * d - b) ** 2
                for a, d, b in zip(
                    rational_vector(base),
                    direction,
                    rational_vector(target),
                    strict=True,
                )
            ),
            Fraction(0),
        )
        for base, target in zip(
            np.asarray(basis.control_points),
            np.asarray(curve.control_points),
            strict=True,
        )
    )


def _spline_extrusion_seam_gap_squared(
    basis: BSplineCurve,
    direction: _Vector,
    actual: tuple[_Vector, _Vector] | None,
    uv: tuple[_Vector, _Vector],
    first: float,
    last: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Bound an affine seam against a proved rational spline endpoint."""
    if uv[1][0] != 0 or actual is None:
        return None
    exact = exact_curve_parameter_point(basis, uv[0][0], coordinate_budget=budget)
    if exact is None:
        return None
    constant_values: list[Fraction] = []
    for value, axis, target in zip(exact, direction, actual[0], strict=True):
        if not isinstance(value, Fraction):
            return None
        constant_values.append(value + uv[0][1] * axis - target)
    slope = tuple(
        uv[1][1] * axis - target
        for axis, target in zip(direction, actual[1], strict=True)
    )
    return _affine_squared(tuple(constant_values), slope, first, last)


def _rational_extrusion_gap_squared(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: ExtrusionSurface,
    first: float,
    last: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    uv = affine_curve_coefficients(pcurve, np.zeros((2,), dtype=np.float64))
    if uv is None or len(uv[0]) != 2:
        return None
    direction = rational_vector(np.asarray(patch.direction))
    actual = affine_curve_coefficients(
        curve, np.zeros((curve.ambient_dimension,), dtype=np.float64)
    )
    basis = affine_curve_coefficients(patch.curve, np.zeros((3,), dtype=np.float64))
    if basis is not None and actual is not None:
        constant = tuple(
            a + uv[0][0] * b + uv[0][1] * d - x
            for a, b, d, x in zip(*basis, direction, actual[0], strict=True)
        )
        slope = tuple(
            uv[1][0] * b + uv[1][1] * d - x
            for b, d, x in zip(basis[1], direction, actual[1], strict=True)
        )
        return _affine_squared(constant, slope, first, last)
    if not isinstance(patch.curve, BSplineCurve):
        return None
    if isinstance(curve, BSplineCurve):
        control_gap = _spline_extrusion_control_gap_squared(
            curve, patch.curve, direction, uv
        )
        if control_gap is not None:
            return control_gap
    return _spline_extrusion_seam_gap_squared(
        patch.curve, direction, actual, uv, first, last, budget=budget
    )


def _native_plane_gap_squared(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: PlanePatch,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    """Dispatch planar lifts to common spline or conic coefficient proofs."""
    if isinstance(curve, BSplineCurve) and isinstance(pcurve, BSplineCurve):
        return _spline_plane_gap_squared(curve, pcurve, patch)
    if not isinstance(curve, (CircleCurve, EllipseCurve)) or type(pcurve) not in (
        CircleCurve,
        EllipseCurve,
    ):
        return None
    if (
        not isinstance(pcurve, (CircleCurve, EllipseCurve))
        or pcurve.ambient_dimension != 2
    ):
        return None
    difference = _harmonic_difference(
        _conic_coefficients(curve),
        _plane_coefficients(patch, _conic_coefficients(pcurve)),
    )
    if not any(any(row) for row in difference):
        return Fraction(0)
    return _squared_upper(_harmonic_bounds(difference, *_trig_bounds(first, last)))


def native_face_gap_squared(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: AbstractSurfacePatch,
    first: float,
    last: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Recompute native conic and same-basis rational extrusion boundary gaps.

    ``None`` means the source correspondence is unsupported, never zero.
    Coefficient cancellation proves zero without evaluating trigonometry.
    """
    if type(curve) not in (CircleCurve, EllipseCurve, LineCurve, BSplineCurve):
        return None
    if not isinstance(curve, (CircleCurve, EllipseCurve, LineCurve, BSplineCurve)):
        return None
    controls = curve.control_points.shape[0] if isinstance(curve, BSplineCurve) else 0
    if isinstance(patch, ExtrusionSurface) and isinstance(patch.curve, BSplineCurve):
        controls += patch.curve.control_points.shape[0]
    with budget.temporary_scope():
        budget.reserve(2048 + 256 * controls, 32768 + 4096 * controls)
        if isinstance(patch, PlanePatch):
            return _native_plane_gap_squared(curve, pcurve, patch, first, last)
        if isinstance(patch, ExtrusionSurface):
            rational_gap = _rational_extrusion_gap_squared(
                curve, pcurve, patch, first, last, budget=budget
            )
            if rational_gap is not None:
                return rational_gap
        if isinstance(curve, BSplineCurve):
            return None
        return _extrusion_gap_squared(curve, pcurve, patch, first, last)
