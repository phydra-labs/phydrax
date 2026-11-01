#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified source face gaps of edge curves on rational spline faces.

The gap ``|S(p(t)) - C(t)|`` between an edge curve ``C`` and the lift of its
p-curve ``p`` through a rational B-spline or T-spline face ``S`` is enclosed by
the exact Bernstein coefficients of the composed difference.  On every
elementary parameter interval (between source knots and the exact chart
crossings of surface knot lines) the difference is one rational polynomial
``N(s) / W(s)`` with ``W > 0``; its Bernstein hull bounds ``|N| / W``, and the
interval is bisected exactly until the certified upper bound is within a factor
two of an attained gap.  An exact iso-parameter identity makes every numerator
coefficient vanish, so the proved gap is exactly zero.

Circles and ellipses are not polynomial in their angle.  On intervals of width
at most one half they are replaced by exact Taylor polynomials whose truncation
and binary64 trigonometric enclosures are carried as an explicit per-axis error.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Sequence
from fractions import Fraction
from math import comb, factorial

import numpy as np

from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
)
from .._interval_enclosure import interval_cos, interval_sin
from .._rational import fraction_down, fraction_up, rational_vector
from ._intersection_curve import SurfaceRegion
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    EllipseCurve,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
)
from ._trim_helpers import PeriodOffset


type _Poly = tuple[Fraction, ...]
type _Homogeneous = tuple[_Poly, ...]
type _Span = tuple[Fraction, Fraction, tuple[tuple[Fraction, ...], ...]]

_TRIG_WIDTH = Fraction(1, 2)
_TAYLOR_REMAINDER = Fraction(1, 2**80)
_ZERO = (Fraction(0),)
_ONE = (Fraction(1),)


def _trim(poly: Sequence[Fraction], /) -> _Poly:
    values = list(poly)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values) if values else _ZERO


def _add(first: _Poly, second: _Poly, /) -> _Poly:
    size = max(len(first), len(second))
    return _trim(
        [
            (first[index] if index < len(first) else Fraction(0))
            + (second[index] if index < len(second) else Fraction(0))
            for index in range(size)
        ]
    )


def _scale(poly: _Poly, factor: Fraction, /) -> _Poly:
    return _trim([value * factor for value in poly])


def _multiply(first: _Poly, second: _Poly, /) -> _Poly:
    result = [Fraction(0)] * (len(first) + len(second) - 1)
    for first_power, first_value in enumerate(first):
        if first_value == 0:
            continue
        for second_power, second_value in enumerate(second):
            result[first_power + second_power] += first_value * second_value
    return _trim(result)


def _compose_affine(poly: _Poly, offset: Fraction, slope: Fraction, /) -> _Poly:
    """Return ``poly(offset + slope * s)`` by Horner composition."""
    result: _Poly = _ZERO
    for value in reversed(poly):
        result = _add(_multiply(result, (offset, slope)), (value,))
    return result


def _bernstein_to_power(coefficients: Sequence[Fraction], /) -> _Poly:
    degree = len(coefficients) - 1
    power = [Fraction(0)] * (degree + 1)
    for index, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        for exponent in range(index, degree + 1):
            power[exponent] += (
                coefficient
                * comb(degree, index)
                * comb(degree - index, exponent - index)
                * (-1) ** (exponent - index)
            )
    return _trim(power)


def _power_to_bernstein(poly: _Poly, degree: int, /) -> tuple[Fraction, ...]:
    if len(poly) - 1 > degree:
        raise ValueError("Bernstein degree must cover the polynomial degree.")
    return tuple(
        sum(
            (
                poly[index] * Fraction(comb(target, index), comb(degree, index))
                for index in range(min(target, len(poly) - 1) + 1)
            ),
            Fraction(0),
        )
        for target in range(degree + 1)
    )


def _bisect(
    coefficients: tuple[Fraction, ...], /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    left, right = [coefficients[0]], [coefficients[-1]]
    row = list(coefficients)
    while len(row) > 1:
        row = [(row[index] + row[index + 1]) / 2 for index in range(len(row) - 1)]
        left.append(row[0])
        right.append(row[-1])
    return tuple(left), tuple(reversed(right))


def _fractions(
    values: Sequence[ExactScalar | PeriodOffset], /
) -> tuple[Fraction, ...] | None:
    """The values as exact rationals, or ``None`` when any one is irrational."""
    exact = tuple(value for value in values if isinstance(value, Fraction))
    return exact if len(exact) == len(values) else None


class _SurfacePiece:
    """One exact surface span as homogeneous power polynomials in local coordinates."""

    __slots__ = ("bounds", "degrees", "power")

    def __init__(
        self,
        bounds: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
        degrees: tuple[int, int],
        power: tuple[tuple[tuple[Fraction, ...], ...], ...],
    ) -> None:
        self.bounds = bounds
        self.degrees = degrees
        # power[i][j][k]: coefficient of u^i v^j of homogeneous component k.
        self.power = power

    def contains(self, point: tuple[Fraction, Fraction], /) -> bool:
        return all(
            self.bounds[axis][0] <= point[axis] <= self.bounds[axis][1]
            for axis in range(2)
        )

    def compose(self, chart: _Homogeneous, /) -> _Homogeneous:
        """Compose with a homogeneous chart ``(U, V, W)`` lying inside this span."""
        u_degree, v_degree = self.degrees
        (u_lower, u_upper), (v_lower, v_upper) = self.bounds
        weight = chart[2]
        local_u = _scale(
            _add(chart[0], _scale(weight, -u_lower)), 1 / (u_upper - u_lower)
        )
        local_v = _scale(
            _add(chart[1], _scale(weight, -v_lower)), 1 / (v_upper - v_lower)
        )
        total = u_degree + v_degree
        u_powers = [_ONE]
        v_powers = [_ONE]
        w_powers = [_ONE]
        for _ in range(total):
            u_powers.append(_multiply(u_powers[-1], local_u))
            v_powers.append(_multiply(v_powers[-1], local_v))
            w_powers.append(_multiply(w_powers[-1], weight))
        components = len(self.power[0][0])
        result: list[_Poly] = [_ZERO] * components
        for u_index in range(u_degree + 1):
            for v_index in range(v_degree + 1):
                coefficient = self.power[u_index][v_index]
                if not any(coefficient):
                    continue
                basis = _multiply(
                    _multiply(u_powers[u_index], v_powers[v_index]),
                    w_powers[total - u_index - v_index],
                )
                for component in range(components):
                    if coefficient[component]:
                        result[component] = _add(
                            result[component], _scale(basis, coefficient[component])
                        )
        return tuple(result)


class _PlanePiece:
    """An affine plane ``origin + u * first_axis + v * second_axis``, defined everywhere."""

    __slots__ = ("axes", "origin")

    def __init__(self, patch: PlanePatch) -> None:
        self.origin = rational_vector(np.asarray(patch.origin))
        self.axes = (
            rational_vector(np.asarray(patch.first_axis)),
            rational_vector(np.asarray(patch.second_axis)),
        )

    def contains(self, point: tuple[Fraction, Fraction], /) -> bool:
        del point
        return True

    def compose(self, chart: _Homogeneous, /) -> _Homogeneous:
        first_chart, second_chart, weight = chart
        return (
            *(
                _add(
                    _scale(weight, self.origin[axis]),
                    _add(
                        _scale(first_chart, self.axes[0][axis]),
                        _scale(second_chart, self.axes[1][axis]),
                    ),
                )
                for axis in range(3)
            ),
            weight,
        )


type _Piece = _SurfacePiece | _PlanePiece


def _surface_pieces(surface: AbstractSurfacePatch, /) -> tuple[_Piece, ...] | None:
    from .._meshing_domain import _prepare_surface_spans

    if isinstance(surface, PlanePatch):
        return (_PlanePiece(surface),)

    prepared = _prepare_surface_spans(surface)
    if prepared is None or not prepared.spans:
        return None
    pieces: list[_SurfacePiece] = []
    for span in prepared.spans:
        u_bounds = _fractions(span.parameter_bounds[0])
        v_bounds = _fractions(span.parameter_bounds[1])
        if (
            u_bounds is None
            or v_bounds is None
            or u_bounds[0] >= u_bounds[1]
            or v_bounds[0] >= v_bounds[1]
        ):
            return None
        u_degree, v_degree = span.degrees
        net = span.homogeneous_bernstein_coefficients
        rows: list[list[tuple[Fraction, ...]]] = []
        for row in net:
            converted: list[tuple[Fraction, ...]] = []
            for coefficient in row:
                exact = _fractions(coefficient)
                if exact is None or len(exact) != 4 or exact[-1] <= 0:
                    return None
                converted.append(exact)
            rows.append(converted)
        if len(rows) != u_degree + 1 or any(len(row) != v_degree + 1 for row in rows):
            return None
        # Bernstein -> power along v for every u row, then along u per v power.
        along_v = [
            [
                _padded(
                    _bernstein_to_power([row[j][k] for j in range(v_degree + 1)]),
                    v_degree,
                )
                for k in range(4)
            ]
            for row in rows
        ]
        along_u = [
            [
                _padded(
                    _bernstein_to_power([along_v[i][k][j] for i in range(u_degree + 1)]),
                    u_degree,
                )
                for k in range(4)
            ]
            for j in range(v_degree + 1)
        ]
        power = tuple(
            tuple(tuple(along_u[j][k][i] for k in range(4)) for j in range(v_degree + 1))
            for i in range(u_degree + 1)
        )
        pieces.append(
            _SurfacePiece(
                ((u_bounds[0], u_bounds[1]), (v_bounds[0], v_bounds[1])),
                (u_degree, v_degree),
                power,
            )
        )
    return tuple(pieces)


def _padded(poly: _Poly, degree: int, /) -> _Poly:
    return poly + (Fraction(0),) * (degree + 1 - len(poly))


def _exact_curve_spans(curve: BSplineCurve, /) -> tuple[_Span, ...] | None:
    from ._correspondence import _exact_bspline_curve_spans

    spans = _exact_bspline_curve_spans(curve)
    if spans is None:
        return None
    result: list[_Span] = []
    for lower, upper, coefficients in spans:
        rows = tuple(_fractions(row) for row in coefficients)
        if any(row is None or row[-1] <= 0 for row in rows):
            return None
        result.append((lower, upper, tuple(row for row in rows if row is not None)))
    return tuple(result)


def _span_on_interval(span: _Span, first: Fraction, last: Fraction, /) -> _Homogeneous:
    """Homogeneous power polynomials of one exact span over ``[first, last]``."""
    lower, upper, coefficients = span
    width = upper - lower
    offset, slope = (first - lower) / width, (last - first) / width
    return tuple(
        _compose_affine(
            _bernstein_to_power([row[component] for row in coefficients]),
            offset,
            slope,
        )
        for component in range(len(coefficients[0]))
    )


def _containing_span(
    spans: tuple[_Span, ...], first: Fraction, last: Fraction, /
) -> _Span | None:
    middle = (first + last) / 2
    for span in spans:
        if span[0] <= first and last <= span[1] and span[0] <= middle <= span[1]:
            return span
    return None


class _Chart:
    """An exact affine or rational B-spline p-curve."""

    __slots__ = ("line", "spans")

    def __init__(
        self,
        line: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] | None,
        spans: tuple[_Span, ...] | None,
    ) -> None:
        self.line, self.spans = line, spans

    def point(self, parameter: Fraction, /) -> tuple[Fraction, Fraction]:
        if self.line is None:
            raise ValueError("Only affine charts evaluate pointwise here.")
        origin, direction = self.line
        return (
            origin[0] + parameter * direction[0],
            origin[1] + parameter * direction[1],
        )

    def homogeneous(self, first: Fraction, last: Fraction, /) -> _Homogeneous | None:
        if self.line is not None:
            origin, direction = self.line
            return (
                _trim(
                    (
                        origin[0] + first * direction[0],
                        (last - first) * direction[0],
                    )
                ),
                _trim(
                    (
                        origin[1] + first * direction[1],
                        (last - first) * direction[1],
                    )
                ),
                _ONE,
            )
        if self.spans is None:
            raise ValueError("A chart is either affine or spanned.")
        span = _containing_span(self.spans, first, last)
        return None if span is None else _span_on_interval(span, first, last)


def _chart(pcurve: AbstractCurve, first: float, last: float, /) -> _Chart | None:
    from ._boolean_core import _pcurve_execution_shadows_match, line_chart_coefficients
    from ._correspondence import _bspline_chart_line

    if pcurve.ambient_dimension != 2 or not _pcurve_execution_shadows_match(pcurve):
        return None
    if isinstance(pcurve, (LineCurve, ExactLineCurve)):
        coefficients = line_chart_coefficients(pcurve)
        if coefficients is None:
            return None
        origin, direction = _fractions(coefficients[0]), _fractions(coefficients[1])
        if origin is None or direction is None:
            return None
        return _Chart(((origin[0], origin[1]), (direction[0], direction[1])), None)
    if not isinstance(pcurve, BSplineCurve):
        return None
    line = _bspline_chart_line(pcurve, first, last) if first < last else None
    if line is not None:
        return _Chart(line, None)
    spans = _exact_curve_spans(pcurve)
    return None if spans is None else _Chart(None, spans)


def _trig_point(parameter: Fraction, /) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Exact midpoints and radii of certified cosine and sine enclosures."""
    bounds = (np.asarray(fraction_down(parameter)), np.asarray(fraction_up(parameter)))
    cosine = interval_cos(bounds)
    sine = interval_sin(bounds)
    cos_lower, cos_upper = Fraction(float(cosine[0])), Fraction(float(cosine[1]))
    sin_lower, sin_upper = Fraction(float(sine[0])), Fraction(float(sine[1]))
    return (
        (cos_lower + cos_upper) / 2,
        (cos_upper - cos_lower) / 2,
        (sin_lower + sin_upper) / 2,
        (sin_upper - sin_lower) / 2,
    )


def _exact_midpoint(value: ExactScalar, /) -> tuple[Fraction, Fraction]:
    lower, upper = coefficient_interval(value)
    return (lower + upper) / 2, (upper - lower) / 2


class _Source:
    """The edge curve as homogeneous polynomials plus a certified per-axis error."""

    __slots__ = ("breakpoints", "conic", "line", "spans")

    def __init__(self, curve: AbstractCurve) -> None:
        self.spans: tuple[_Span, ...] | None = None
        self.line: tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None = None
        self.conic: tuple[tuple[tuple[Fraction, Fraction], ...], ...] | None = None
        self.breakpoints: tuple[Fraction, ...] = ()
        if isinstance(curve, BSplineCurve):
            self.spans = _exact_curve_spans(curve)
            if self.spans is not None:
                self.breakpoints = tuple(
                    sorted({value for span in self.spans for value in span[:2]})
                )
        elif isinstance(curve, (LineCurve, ExactLineCurve)):
            from ._conic_gaps import _affine_coefficients

            affine = _affine_coefficients(curve)
            if affine is not None:
                origin, direction = _fractions(affine[0]), _fractions(affine[1])
                if origin is not None and direction is not None:
                    self.line = (origin, direction)
        elif isinstance(curve, (CircleCurve, EllipseCurve)):
            from ._conic_gaps import _conic_coefficients

            self.conic = tuple(
                tuple(_exact_midpoint(value) for value in vector)
                for vector in _conic_coefficients(curve)
            )

    @property
    def supported(self) -> bool:
        return self.spans is not None or self.line is not None or self.conic is not None

    @property
    def trigonometric(self) -> bool:
        return self.conic is not None

    def homogeneous(
        self, first: Fraction, last: Fraction, /
    ) -> tuple[_Homogeneous, tuple[Fraction, ...]] | None:
        if self.spans is not None:
            span = _containing_span(self.spans, first, last)
            if span is None:
                return None
            return _span_on_interval(span, first, last), (Fraction(0),) * 3
        if self.line is not None:
            origin, direction = self.line
            return (
                *(
                    _trim(
                        (
                            origin[axis] + first * direction[axis],
                            (last - first) * direction[axis],
                        )
                    )
                    for axis in range(3)
                ),
                _ONE,
            ), (Fraction(0),) * 3
        return self._taylor(first, last)

    def _taylor(
        self, first: Fraction, last: Fraction, /
    ) -> tuple[_Homogeneous, tuple[Fraction, ...]]:
        if self.conic is None:
            raise ValueError("Only conic sources have a Taylor form.")
        center, cosine_axis, sine_axis = self.conic
        middle, half = (first + last) / 2, (last - first) / 2
        cos_mid, cos_radius, sin_mid, sin_radius = _trig_point(middle)
        order = 1
        while half ** (order + 1) / factorial(order + 1) > _TAYLOR_REMAINDER:
            order += 1
        remainder = half ** (order + 1) / factorial(order + 1)
        offset = (-half, 2 * half)
        offset_powers = [_ONE]
        for _ in range(order):
            offset_powers.append(_multiply(offset_powers[-1], offset))
        cosine: _Poly = _ZERO
        sine: _Poly = _ZERO
        for exponent in range(order + 1):
            term = _scale(
                offset_powers[exponent],
                Fraction((-1) ** (exponent // 2), factorial(exponent)),
            )
            if exponent % 2:
                sine = _add(sine, term)
            else:
                cosine = _add(cosine, term)
        trig_error = cos_radius + sin_radius + (abs(cos_mid) + abs(sin_mid)) * remainder
        components: list[_Poly] = []
        errors: list[Fraction] = []
        for axis in range(3):
            constant, constant_radius = center[axis]
            first_value, first_radius = cosine_axis[axis]
            second_value, second_radius = sine_axis[axis]
            components.append(
                _add(
                    (constant,),
                    _add(
                        _scale(cosine, first_value * cos_mid + second_value * sin_mid),
                        _scale(sine, second_value * cos_mid - first_value * sin_mid),
                    ),
                )
            )
            errors.append(
                constant_radius
                + first_radius
                + second_radius
                + (abs(first_value) + abs(second_value)) * trig_error
            )
        return (*components, _ONE), tuple(errors)


class _Interval:
    __slots__ = ("denominator", "errors", "first", "last", "numerators", "upper")

    def __init__(
        self,
        first: Fraction,
        last: Fraction,
        numerators: tuple[tuple[Fraction, ...], ...],
        denominator: tuple[Fraction, ...],
        errors: tuple[Fraction, ...],
    ) -> None:
        self.first, self.last = first, last
        self.numerators, self.denominator, self.errors = numerators, denominator, errors
        lowest = min(denominator)
        self.upper: Fraction | None = (
            None
            if lowest <= 0
            else sum(
                (
                    (max(abs(value) for value in numerator) / lowest + error) ** 2
                    for numerator, error in zip(numerators, errors, strict=True)
                ),
                Fraction(0),
            )
        )

    @property
    def exact_zero(self) -> bool:
        return not any(self.errors) and not any(
            value for numerator in self.numerators for value in numerator
        )

    def attained_lower(self) -> Fraction:
        """A certified lower bound of the squared gap at the interval endpoints."""
        best = Fraction(0)
        for index in (0, -1):
            squared = Fraction(0)
            for numerator, error in zip(self.numerators, self.errors, strict=True):
                magnitude = abs(numerator[index] / self.denominator[index]) - error
                if magnitude > 0:
                    squared += magnitude * magnitude
            best = max(best, squared)
        return best

    def bisect(self) -> tuple[_Interval, _Interval]:
        middle = (self.first + self.last) / 2
        split = tuple(_bisect(numerator) for numerator in self.numerators)
        left_denominator, right_denominator = _bisect(self.denominator)
        return (
            _Interval(
                self.first,
                middle,
                tuple(pair[0] for pair in split),
                left_denominator,
                self.errors,
            ),
            _Interval(
                middle,
                self.last,
                tuple(pair[1] for pair in split),
                right_denominator,
                self.errors,
            ),
        )


def _difference_interval(
    first: Fraction,
    last: Fraction,
    source: tuple[_Homogeneous, tuple[Fraction, ...]],
    lifted: _Homogeneous,
    /,
) -> _Interval | None:
    (*source_components, source_weight), errors = source
    *lifted_components, lifted_weight = lifted
    numerators = tuple(
        _add(
            _multiply(source_components[axis], lifted_weight),
            _scale(_multiply(lifted_components[axis], source_weight), Fraction(-1)),
        )
        for axis in range(3)
    )
    denominator = _multiply(source_weight, lifted_weight)
    degree = max(len(poly) - 1 for poly in (*numerators, denominator))
    denominator_bernstein = _power_to_bernstein(denominator, degree)
    if denominator_bernstein[0] <= 0 or denominator_bernstein[-1] <= 0:
        return None
    return _Interval(
        first,
        last,
        tuple(_power_to_bernstein(numerator, degree) for numerator in numerators),
        denominator_bernstein,
        errors,
    )


def _affine_breakpoints(
    chart: _Chart,
    pieces: tuple[_Piece, ...],
    first: Fraction,
    last: Fraction,
    /,
) -> set[Fraction]:
    if chart.line is None:
        raise ValueError("Only affine charts cross knot lines at rational parameters.")
    origin, direction = chart.line
    breakpoints: set[Fraction] = set()
    for axis in range(2):
        if direction[axis] == 0:
            continue
        for piece in pieces:
            if not isinstance(piece, _SurfacePiece):
                continue
            for knot in piece.bounds[axis]:
                parameter = (knot - origin[axis]) / direction[axis]
                if first < parameter < last:
                    breakpoints.add(parameter)
    return breakpoints


def rational_spline_face_gap_squared(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    source_region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    """Certified squared gap of ``curve`` against its p-curve lift on a spline face.

    ``None`` signals an unsupported correspondence, never a zero gap.
    """
    if not (np.isfinite(first) and np.isfinite(last) and first <= last):
        return None
    for domain in (curve.parameter_domain, pcurve.parameter_domain):
        if domain is not None and (first < domain[0] or last > domain[1]):
            return None
    if curve.ambient_dimension != 3:
        return None
    source = _Source(curve)
    if not source.supported:
        return None
    chart = _chart(pcurve, first, last)
    if chart is None:
        return None
    pieces = _surface_pieces(source_region.patch)
    if pieces is None:
        return None
    region_box = np.asarray(source_region.parameter_box, dtype=np.float64)
    if region_box.shape != (2, 2) or not np.all(np.isfinite(region_box)):
        return None
    region = tuple(
        (
            Fraction.from_float(float(region_box[0, axis])),
            Fraction.from_float(float(region_box[1, axis])),
        )
        for axis in range(2)
    )
    first_exact, last_exact = Fraction.from_float(first), Fraction.from_float(last)
    breakpoints = {first_exact, last_exact}
    breakpoints.update(
        value for value in source.breakpoints if first_exact < value < last_exact
    )
    if chart.line is not None:
        for endpoint in (first_exact, last_exact):
            point = chart.point(endpoint)
            if not all(
                region[axis][0] <= point[axis] <= region[axis][1] for axis in range(2)
            ):
                return None
        breakpoints.update(_affine_breakpoints(chart, pieces, first_exact, last_exact))
    else:
        if chart.spans is None:
            raise ValueError("A chart is either affine or spanned.")
        if len(pieces) != 1:
            # A curved chart crosses surface knot lines at algebraic parameters.
            return None
        for span in chart.spans:
            if span[1] <= first_exact or span[0] >= last_exact:
                continue
            breakpoints.update(
                value for value in span[:2] if first_exact < value < last_exact
            )
            for row in span[2]:
                point = (row[0] / row[2], row[1] / row[2])
                if not all(
                    region[axis][0] <= point[axis] <= region[axis][1] for axis in range(2)
                ) or not pieces[0].contains(point):
                    return None
    ordered = sorted(breakpoints)
    if len(ordered) == 1:
        # A zero-length range is one constant interval at its single parameter.
        ordered = [first_exact, last_exact]
    elif source.trigonometric:
        refined: list[Fraction] = [ordered[0]]
        for lower, upper in itertools.pairwise(ordered):
            count = -(-(upper - lower) // _TRIG_WIDTH)
            refined.extend(
                lower + (upper - lower) * index / count for index in range(1, count + 1)
            )
        ordered = refined
    intervals: list[_Interval] = []
    for lower, upper in itertools.pairwise(ordered):
        source_polynomials = source.homogeneous(lower, upper)
        chart_polynomials = chart.homogeneous(lower, upper)
        if source_polynomials is None or chart_polynomials is None:
            return None
        if chart.line is not None:
            middle = chart.point((lower + upper) / 2)
            endpoints = (chart.point(lower), chart.point(upper))
            owners = tuple(
                piece
                for piece in pieces
                if piece.contains(middle)
                and all(piece.contains(point) for point in endpoints)
            )
        else:
            owners = pieces
        if not owners:
            return None
        for owner in owners:
            interval = _difference_interval(
                lower, upper, source_polynomials, owner.compose(chart_polynomials)
            )
            if interval is None:
                return None
            intervals.append(interval)
    return _refine(intervals)


def _refine(intervals: list[_Interval], /) -> Fraction:
    """Bisect the worst interval until its bound is within four times an attained gap.

    ``floor`` is the irreducible trigonometric resolution; without it a curve
    that matches its lift up to that resolution could never be decided.
    """
    floor = sum(
        (
            (3 * max(interval.errors[axis] for interval in intervals)) ** 2
            for axis in range(3)
        ),
        Fraction(0),
    )
    attained = Fraction(0)
    counter = itertools.count()
    # Unbounded intervals (denominator hull not yet positive) order first.
    heap: list[tuple[int, Fraction, int, _Interval]] = []

    def push(interval: _Interval, /) -> None:
        nonlocal attained
        attained = max(attained, interval.attained_lower())
        if interval.exact_zero:
            return
        if interval.upper is None:
            heapq.heappush(heap, (0, Fraction(0), next(counter), interval))
        else:
            heapq.heappush(heap, (1, -interval.upper, next(counter), interval))

    for interval in intervals:
        push(interval)
    while heap:
        bounded, negative_upper, _, worst = heap[0]
        if bounded and -negative_upper <= 4 * (attained + floor):
            return -negative_upper
        heapq.heappop(heap)
        for child in worst.bisect():
            push(child)
    return Fraction(0)


__all__ = ["rational_spline_face_gap_squared"]
