#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact carrier intersections for explicit external-source repair.

These routines construct candidates from carrier equations only.  The caller
must use the authored incidence and parameter domains to select a unique
candidate; numerical vertex positions are not evidence of incidence.
"""

from __future__ import annotations

from fractions import Fraction
from math import isfinite, isqrt
from typing import cast

from ...discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint
from ._intersection_curve import IntersectionCurve
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    LineCurve,
)
from ._root_bindings import BRepConicVertex, BRepVertexRoot


type ExactPoint = tuple[ExactScalar, ExactScalar, ExactScalar]


def _dot(a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...], /) -> ExactScalar:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _sub(
    a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...], /) -> ExactPoint:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rational(values: tuple[ExactScalar, ...], /) -> bool:
    return all(isinstance(value, Fraction) for value in values)


def _sqrt_rational(value: Fraction, /) -> ExactScalar:
    """Avoid radical factorization when an exact rational square is present."""
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if (
        numerator * numerator == value.numerator
        and denominator * denominator == value.denominator
    ):
        return Fraction(numerator, denominator)
    return RadicalElement.sqrt(value)


def _circle_data(
    circle: CircleCurve, /
) -> tuple[
    tuple[ExactScalar, ...],
    tuple[ExactScalar, ...],
    tuple[ExactScalar, ...],
    ExactScalar,
]:
    if not isinstance(circle, CircleCurve) or circle.ambient_dimension != 3:
        raise ValueError("Source repair requires a three-dimensional circle.")
    center, first, second, radius = (
        circle.center_exact,
        circle.first_axis_exact,
        circle.second_axis_exact,
        circle.radius_exact,
    )
    if not (
        all(
            isinstance(value, (Fraction, RadicalElement))
            for value in (*center, *first, *second)
        )
        and isinstance(radius, (Fraction, RadicalElement))
        and radius > 0
    ):
        raise ValueError("Source repair intersection requires exact circle coefficients.")
    if _dot(first, first) != 1 or _dot(second, second) != 1 or _dot(first, second) != 0:
        raise ValueError("Source repair requires an exact orthonormal circle frame.")
    return center, first, second, radius


def _point_on_circle(point: ExactPoint, circle: CircleCurve, /) -> bool:
    center, first, second, radius = _circle_data(circle)
    delta = _sub(point, center)
    return (
        _dot(delta, _cross(first, second)) == 0 and _dot(delta, delta) == radius * radius
    )


def circle_circle_points(
    first: CircleCurve, second: CircleCurve, /
) -> tuple[ExactPoint, ...]:
    """Return every isolated supported exact intersection of two 3D circles.

    Coincident circles are refused because they do not select a vertex.
    """
    c, u, v, r = _circle_data(first)
    d, p, q, s = _circle_data(second)
    normal = _cross(p, q)
    delta = _sub(c, d)
    a, b, h = r * _dot(u, normal), r * _dot(v, normal), -_dot(delta, normal)
    if a == 0 and b == 0:
        if h != 0:
            return ()
        # Coplanar: subtract the two circle equations to get a line.
        a, b = 2 * r * _dot(delta, u), 2 * r * _dot(delta, v)
        h = s * s - r * r - _dot(delta, delta)
    norm = a * a + b * b
    if norm == 0:
        if h == 0:
            raise ValueError("Coincident circle carriers do not isolate an endpoint.")
        return ()
    disc = norm - h * h
    if disc < 0:
        return ()
    try:
        disc = rational_coefficient(disc, role="circle intersection discriminant")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Circle intersection needs a supported rational discriminant."
        ) from error
    root = _sqrt_rational(disc)
    signs = (0,) if disc == 0 else (-1, 1)
    result = []
    for sign in signs:
        cosine = (a * h - sign * b * root) / norm
        sine = (b * h + sign * a * root) / norm
        point = cast(
            ExactPoint, tuple(c[i] + r * (u[i] * cosine + v[i] * sine) for i in range(3))
        )
        if _point_on_circle(point, second):
            result.append(point)
    return tuple(result)


def circle_line_points(circle: CircleCurve, line: LineCurve, /) -> tuple[ExactPoint, ...]:
    """Return every isolated supported exact intersection of a circle and line."""
    center, first, second, radius = _circle_data(circle)
    if not isinstance(line, LineCurve) or line.ambient_dimension != 3:
        raise ValueError("Source repair requires a three-dimensional line.")
    origin, direction = line.origin_exact, line.direction_exact
    if not (_rational(origin) and _rational(direction)):
        raise ValueError(
            "Source repair intersection requires rational line coefficients."
        )
    delta = _sub(origin, center)
    normal = _cross(first, second)
    plane_offset, plane_direction = _dot(delta, normal), _dot(direction, normal)
    if plane_direction:
        parameters = (-plane_offset / plane_direction,)
    elif plane_offset:
        return ()
    else:
        a, b, c = (
            _dot(direction, direction),
            2 * _dot(delta, direction),
            _dot(delta, delta) - radius * radius,
        )
        if a == 0:
            raise ValueError("A zero-direction line does not isolate an endpoint.")
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return ()
        try:
            discriminant = rational_coefficient(
                discriminant, role="circle-line discriminant"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Circle-line intersection needs a supported rational discriminant."
            ) from error
        square = _sqrt_rational(discriminant)
        parameters = (
            (-b / (2 * a),)
            if discriminant == 0
            else ((-b - square) / (2 * a), (-b + square) / (2 * a))
        )
    points = []
    for parameter in parameters:
        point = cast(
            ExactPoint, tuple(origin[i] + parameter * direction[i] for i in range(3))
        )
        if _point_on_circle(point, circle):
            points.append(point)
    return tuple(points)


def conic_binding(
    circle: CircleCurve, point: ExactPoint, angle_hint: float, /
) -> tuple[BRepVertexRoot, ConicParameterEndpoint]:
    """Bind an exact point to this physical circle; hint chooses only a turn."""
    if not isfinite(angle_hint) or not _point_on_circle(point, circle):
        raise ValueError("The endpoint is not an exact point on the circle.")
    root = ConicParameterRoot.from_exact_point(circle, point, angle_hint)
    if root is None or root.exact_point() != point:
        raise ValueError("The exact endpoint has no supported conic root.")
    return BRepVertexRoot(BRepConicVertex(root)), ConicParameterEndpoint(
        root, circle, None, point
    )


def line_parameter(line: LineCurve, point: ExactPoint, /) -> Fraction:
    """Prove incidence and return a representable exact line parameter."""
    if not isinstance(line, LineCurve) or line.ambient_dimension != 3:
        raise ValueError("Source repair requires a three-dimensional line.")
    origin, direction = line.origin_exact, line.direction_exact
    if not (_rational(origin) and _rational(direction)):
        raise ValueError("Source repair requires rational line coefficients.")
    nonzero = next((i for i, value in enumerate(direction) if value != 0), None)
    if nonzero is None:
        raise ValueError("A zero-direction line does not isolate an endpoint.")
    parameter = (point[nonzero] - origin[nonzero]) / direction[nonzero]
    try:
        parameter = rational_coefficient(parameter, role="line endpoint parameter")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The endpoint has no supported exact rational line parameter."
        ) from error
    if any(origin[i] + parameter * direction[i] != point[i] for i in range(3)):
        raise ValueError("The endpoint is not on the exact line carrier.")
    return parameter


def endpoint_on_carrier(
    root: ConicParameterRoot,
    carrier: AbstractCurve,
    parameter: Fraction | None,
    point: ExactPoint,
    /,
) -> ConicParameterEndpoint:
    """Attach an already-proved physical endpoint to its edge or p-curve."""
    if root.exact_point() != point:
        raise ValueError("The conic root does not define the endpoint point.")
    if isinstance(carrier, LineCurve) and carrier.ambient_dimension == 3:
        if parameter != line_parameter(carrier, point):
            raise ValueError(
                "The line endpoint parameter does not match its exact point."
            )
    elif isinstance(carrier, CircleCurve) and carrier.ambient_dimension == 3:
        if not _point_on_circle(point, carrier):
            raise ValueError("The endpoint is not on the exact circle carrier.")
    return ConicParameterEndpoint(root, carrier, parameter, point)


def native_turn_endpoints(
    carrier: AbstractCurve | IntersectionCurve,
    /,
    *,
    patch: AbstractSurfacePatch | None = None,
    axis: int | None = None,
    rational: Fraction | int = Fraction(0),
    turns: Fraction | int = Fraction(0),
) -> tuple[NativePeriodEndpoint, NativePeriodEndpoint]:
    """Bind a full native turn to its actual edge or chart carrier."""
    return (
        NativePeriodEndpoint(carrier, patch, axis, rational=rational, turns=turns),
        NativePeriodEndpoint(carrier, patch, axis, rational=rational, turns=turns + 1),
    )
