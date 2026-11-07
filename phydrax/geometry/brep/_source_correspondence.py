#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact analytic source correspondence and authored-pose bounds."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from .._rational import fraction_up, rational_cross, rational_vector
from ._exact import fraction_norm_upper, gram_deviation_bound, square_root_interval
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    LineCurve,
    OffsetCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
)
from ._placed import PlacedCurve, source_transform_bounds


def norm_upper(components: np.ndarray, /) -> float:
    values = np.asarray(components, dtype=np.float64)
    if np.any(np.isnan(values)):
        return float("inf")
    if np.any(np.isinf(values)):
        return float("inf")
    return fraction_norm_upper(rational_vector(values.reshape(-1)))


def _exact_norm_upper(values: tuple[ExactScalar, ...], /) -> float:
    magnitudes = []
    for value in values:
        lower, upper = coefficient_interval(value, bits=128)
        magnitudes.append(max(abs(lower), abs(upper)))
    return fraction_norm_upper(tuple(magnitudes))


def _offset_affine_coefficients(
    curve: OffsetCurve, point: tuple[ExactScalar, ...], /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]] | None:
    layers: list[OffsetCurve] = []
    base: AbstractCurve = curve
    while isinstance(base, OffsetCurve):
        base.validate_source_bank()
        layers.append(base)
        base = base.base
    source = _affine_curve_coefficients(base, point)
    if source is None:
        return None
    origin, tangent = source
    for layer in reversed(layers):
        normal = (
            (tangent[1], -tangent[0])
            if layer.direction is None
            else rational_cross(tangent, rational_vector(np.asarray(layer.direction)))
        )
        squared = sum((value * value for value in normal), Fraction(0))
        if squared == 0:
            return None
        try:
            rational_squared = rational_coefficient(
                squared, role="offset normal squared length"
            )
        except ValueError:
            return None
        length = RadicalElement.sqrt(rational_squared)
        origin = tuple(
            value + layer.distance_exact * component / length
            for value, component in zip(origin, normal, strict=True)
        )
    return origin, tangent


def _affine_curve_coefficients(
    curve: AbstractCurve, point: tuple[ExactScalar, ...], /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]] | None:
    from ._intersection_curve import CurveTrimSegment

    if isinstance(curve, PlacedCurve):
        if len(point) != 3:
            return None
        source = _affine_curve_coefficients(
            curve.definition, (Fraction(0), Fraction(0), Fraction(0))
        )
        if source is None:
            return None
        origin, direction = source
        matrix = curve.affine.matrix
        placed_origin = curve.affine.apply(origin)
        placed_direction = tuple(
            sum(
                (matrix[row][column] * direction[column] for column in range(3)),
                Fraction(0),
            )
            for row in range(3)
        )
        return tuple(
            placed_origin[axis] - point[axis] for axis in range(3)
        ), placed_direction
    if isinstance(curve, OffsetCurve):
        return _offset_affine_coefficients(curve, point)
    if isinstance(curve, CurveTrimSegment):
        inner = _affine_curve_coefficients(curve.curve, point)
        if inner is None:
            return None
        source_first = Fraction.from_float(curve.last if curve.reversed else curve.first)
        source_scale = (
            Fraction.from_float(curve.first) - Fraction.from_float(curve.last)
            if curve.reversed
            else Fraction.from_float(curve.last) - Fraction.from_float(curve.first)
        )
        constant, slope = inner
        return tuple(
            constant[i] + source_first * slope[i] for i in range(len(point))
        ), tuple(source_scale * slope[i] for i in range(len(point)))
    if isinstance(curve, LineCurve):
        origin = curve.origin_exact
        direction = curve.direction_exact
        if len(origin) != len(point):
            return None
        return tuple(origin[i] - point[i] for i in range(len(point))), direction
    if isinstance(curve, ExactLineCurve):
        if len(curve.first_exact) != len(point):
            return None
        return tuple(curve.first_exact[i] - point[i] for i in range(len(point))), tuple(
            curve.last_exact[i] - curve.first_exact[i] for i in range(len(point))
        )
    if isinstance(curve, PlaneCylinderSectionPCurve):
        phase = curve.angle_offset_root
        if (
            len(point) != 2
            or any(value != 0 for value in (*curve.cosine_exact, *curve.sine_exact))
            or (
                phase is not None
                and (phase.cosine_exact != 1 or phase.sine_exact != 0 or phase.turns != 0)
            )
        ):
            return None
        return tuple(
            curve.offset_exact[axis] - point[axis] for axis in range(2)
        ), curve.linear_exact
    if isinstance(curve, BSplineCurve):
        # A clamped degree-one span with equal weights is the exact segment
        # P0 + (t - a) / (b - a) * (P1 - P0) on its knot interval [a, b].
        if (
            curve.degree != 1
            or len(curve.exact_control_points) != 2
            or len(curve.exact_knots) != 4
            or curve.exact_weights[0] != curve.exact_weights[1]
        ):
            return None
        first_point, last_point = curve.exact_control_points
        lower, _, _, upper = curve.exact_knots
        if (
            len(first_point) != len(point)
            or curve.exact_knots[1] != lower
            or curve.exact_knots[2] != upper
            or not upper > lower
        ):
            return None
        span = upper - lower
        direction = tuple(
            (last_point[i] - first_point[i]) / span for i in range(len(point))
        )
        return tuple(
            first_point[i] - lower * direction[i] - point[i] for i in range(len(point))
        ), direction
    if isinstance(curve, ConicRootLinePCurve):
        # Only the principal root at angle zero has a rational chart coordinate;
        # every other root angle is transcendental in the conic parameter.
        root = curve.root
        if (
            len(point) != 2
            or root.cosine_exact != 1
            or root.sine_exact != 0
            or root.turns != 0
        ):
            return None
        return (
            curve.phase_shift_exact - point[0],
            curve.height_origin_exact - point[1],
        ), (Fraction(0), curve.height_direction_exact)
    return None


def _harmonic_coefficients(
    curve: CircleCurve | EllipseCurve, /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...], tuple[ExactScalar, ...]]:
    center = curve.center_exact
    first = curve.first_axis_exact
    second = curve.second_axis_exact
    if isinstance(curve, CircleCurve):
        first_radius = second_radius = curve.radius_exact
    else:
        first_radius = curve.first_radius_exact
        second_radius = curve.second_radius_exact
    return (
        center,
        tuple(first_radius * value for value in first),
        tuple(second_radius * value for value in second),
    )


def _plane_harmonics(
    plane: PlanePatch,
    coefficients: tuple[
        tuple[ExactScalar, ...],
        tuple[ExactScalar, ...],
        tuple[ExactScalar, ...],
    ],
) -> tuple[
    tuple[ExactScalar, ExactScalar, ExactScalar],
    tuple[ExactScalar, ExactScalar, ExactScalar],
    tuple[ExactScalar, ExactScalar, ExactScalar],
]:
    origin = rational_vector(np.asarray(plane.origin))
    axes = (
        rational_vector(np.asarray(plane.first_axis)),
        rational_vector(np.asarray(plane.second_axis)),
    )
    center, cosine, sine = coefficients

    def project(component: tuple[ExactScalar, ...], dimension: int, /) -> ExactScalar:
        return sum(
            (axes[axis][dimension] * component[axis] for axis in range(2)),
            Fraction(0),
        )

    return (
        (
            origin[0] + project(center, 0),
            origin[1] + project(center, 1),
            origin[2] + project(center, 2),
        ),
        (
            project(cosine, 0),
            project(cosine, 1),
            project(cosine, 2),
        ),
        (
            project(sine, 0),
            project(sine, 1),
            project(sine, 2),
        ),
    )


def _cylinder_harmonics(
    cylinder: CylinderPatch,
    pcurve: AbstractCurve,
    /,
) -> (
    tuple[
        tuple[ExactScalar, ...],
        tuple[ExactScalar, ...],
        tuple[ExactScalar, ...],
    ]
    | None
):
    zero = (Fraction(0), Fraction(0))
    coefficients = _affine_curve_coefficients(pcurve, zero)
    if coefficients is None:
        return None
    offset, slope = coefficients
    if offset[0] != 0 or slope != (Fraction(1), Fraction(0)):
        return None
    center = tuple(
        origin + offset[1] * axis
        for origin, axis in zip(cylinder.origin_exact, cylinder.axis_exact, strict=True)
    )
    return (
        center,
        tuple(cylinder.radius_exact * value for value in cylinder.first_axis_exact),
        tuple(cylinder.radius_exact * value for value in cylinder.second_axis_exact),
    )


def circle_correspondence(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: AbstractSurfacePatch,
    /,
) -> float | None:
    """Continuous deviation from exact common-frequency conic coefficients."""
    if not isinstance(curve, (CircleCurve, EllipseCurve)):
        return None
    actual = _harmonic_coefficients(curve)
    if isinstance(surface, PlanePatch):
        pcurve_coefficients = (
            _harmonic_coefficients(pcurve)
            if isinstance(pcurve, (CircleCurve, EllipseCurve))
            else None
        )
        expected = (
            None
            if pcurve_coefficients is None
            else _plane_harmonics(surface, pcurve_coefficients)
        )
    elif isinstance(surface, CylinderPatch):
        expected = _cylinder_harmonics(surface, pcurve)
    else:
        return None
    if expected is None:
        return None
    bounds = tuple(
        _exact_norm_upper(tuple(a - b for a, b in zip(left, right, strict=True)))
        for left, right in zip(actual, expected, strict=True)
    )
    total = sum((Fraction.from_float(bound) for bound in bounds), Fraction(0))
    return fraction_up(total)


def plane_correspondence(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: PlanePatch,
    first: float,
    last: float,
    point: np.ndarray,
    /,
) -> float | None:
    """Enclose whole-interval source error by exact affine or rational spline data."""
    point_exact = rational_vector(np.asarray(point, dtype=np.float64))
    source = None if curve is None else _affine_curve_coefficients(curve, point_exact)
    parameters = _affine_curve_coefficients(pcurve, (Fraction(0), Fraction(0)))
    origin = rational_vector(np.asarray(surface.origin))
    axes = (
        rational_vector(np.asarray(surface.first_axis)),
        rational_vector(np.asarray(surface.second_axis)),
    )
    if source is not None and parameters is not None:
        constant_source, slope_source = source
        constant_parameter, slope_parameter = parameters
        constant = tuple(
            origin[d]
            + sum((axes[k][d] * constant_parameter[k] for k in range(2)), Fraction(0))
            - constant_source[d]
            - point_exact[d]
            for d in range(3)
        )
        slope = tuple(
            sum((axes[k][d] * slope_parameter[k] for k in range(2)), Fraction(0))
            - slope_source[d]
            for d in range(3)
        )
        return max(
            _exact_norm_upper(
                tuple(constant[d] + Fraction(parameter) * slope[d] for d in range(3))
            )
            for parameter in (
                Fraction.from_float(float(first)),
                Fraction.from_float(float(last)),
            )
        )
    if isinstance(curve, BSplineCurve) and isinstance(pcurve, BSplineCurve):
        if (
            curve.degree != pcurve.degree
            or not np.array_equal(np.asarray(curve.knots), np.asarray(pcurve.knots))
            or not np.array_equal(np.asarray(curve.weights), np.asarray(pcurve.weights))
        ):
            return None
        residuals = []
        for xyz, uv in zip(
            np.asarray(curve.control_points),
            np.asarray(pcurve.control_points),
            strict=True,
        ):
            world = tuple(
                origin[d]
                + sum(
                    (axes[k][d] * Fraction.from_float(float(uv[k])) for k in range(2)),
                    Fraction(0),
                )
                - Fraction.from_float(float(xyz[d]))
                for d in range(3)
            )
            residuals.append(fraction_norm_upper(world))
        return max(residuals)
    return None


def placed_deviation_bound(rotation: np.ndarray, deviation: float, /) -> float:
    """Enclose the operator image of source deviation under an authored pose."""
    matrix = np.asarray(rotation, dtype=np.float64)
    value = float(deviation)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("placed_deviation_bound requires a finite 3x3 rotation matrix.")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "placed_deviation_bound requires a finite nonnegative deviation."
        )
    gram_error = Fraction.from_float(gram_deviation_bound(matrix))
    scale_upper = square_root_interval(Fraction(1) + gram_error)[1]
    return fraction_up(Fraction.from_float(scale_upper) * Fraction.from_float(value))


__all__ = [
    "circle_correspondence",
    "fraction_norm_upper",
    "norm_upper",
    "placed_deviation_bound",
    "plane_correspondence",
    "source_transform_bounds",
]
