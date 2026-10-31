#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, nextafter
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from .._fingerprint import canonical_fingerprint


if TYPE_CHECKING:
    from ..discretization._coordinate_enclosure import ExactScalar


def _insert_knot(
    values: np.ndarray, knots: np.ndarray, degree: int, knot: float | ExactScalar, /
) -> tuple[np.ndarray, np.ndarray]:
    """Insert one knot along the leading coefficient axis (Boehm)."""
    multiplicity = int(np.count_nonzero(knots == knot))
    span = bisect_right(knots, knot) - 1
    inserted = np.empty((values.shape[0] + 1, *values.shape[1:]), dtype=values.dtype)
    inserted[: span - degree + 1] = values[: span - degree + 1]
    inserted[span - multiplicity + 1 :] = values[span - multiplicity :]
    for index in range(span - degree + 1, span - multiplicity + 1):
        alpha = (knot - knots[index]) / (knots[index + degree] - knots[index])
        inserted[index] = alpha * values[index] + (1 - alpha) * values[index - 1]
    inserted_knots = np.empty((knots.shape[0] + 1,), dtype=knots.dtype)
    inserted_knots[: span + 1] = knots[: span + 1]
    inserted_knots[span + 1] = knot
    inserted_knots[span + 2 :] = knots[span + 1 :]
    return inserted, inserted_knots


def _insert_knot_once_exact(
    knots: Sequence[ExactScalar],
    coefficients: Sequence[Sequence[ExactScalar]],
    degree: int,
    knot: ExactScalar,
    /,
) -> tuple[tuple[ExactScalar, ...], tuple[tuple[ExactScalar, ...], ...]]:
    """Apply exact rational/algebraic Boehm insertion to a coefficient bank."""
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree < 0:
        raise ValueError("degree must be a nonnegative integer.")
    degree_ = int(degree)
    knot_vector = tuple(knots)
    rows = tuple(tuple(row) for row in coefficients)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Exact knot insertion requires a nonempty rectangular bank.")
    if len(knot_vector) != len(rows) + degree_ + 1:
        raise ValueError("Exact coefficient and knot-vector extents are inconsistent.")
    if any(right < left for left, right in zip(knot_vector[:-1], knot_vector[1:])):
        raise ValueError("Exact knot insertion requires a nondecreasing knot vector.")
    if not knot_vector[degree_] < knot < knot_vector[-degree_ - 1]:
        raise ValueError("Exact knot insertion requires a strict interior knot.")
    if knot_vector.count(knot) > degree_:
        raise ValueError("A knot cannot be inserted beyond degree + 1 multiplicity.")
    inserted, inserted_knots = _insert_knot(
        np.asarray(rows, dtype=object),
        np.asarray(knot_vector, dtype=object),
        degree_,
        knot,
    )
    return tuple(inserted_knots), tuple(tuple(row) for row in inserted)


def _insert_knot_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    knots: np.ndarray,
    degree: int,
    knot: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boehm insertion that outwardly bounds every ratio and coefficient."""
    multiplicity = int(np.count_nonzero(knots == knot))
    span = int(np.searchsorted(knots, knot, side="right") - 1)
    shape = (lower.shape[0] + 1, *lower.shape[1:])
    inserted_lower, inserted_upper = np.empty(shape), np.empty(shape)
    inserted_lower[: span - degree + 1] = lower[: span - degree + 1]
    inserted_upper[: span - degree + 1] = upper[: span - degree + 1]
    inserted_lower[span - multiplicity + 1 :] = lower[span - multiplicity :]
    inserted_upper[span - multiplicity + 1 :] = upper[span - multiplicity :]
    for index in range(span - degree + 1, span - multiplicity + 1):
        numerator = knot - knots[index]
        denominator = knots[index + degree] - knots[index]
        denominator_lower = np.nextafter(denominator, -np.inf)
        if denominator_lower <= 0.0:
            raise ValueError("Boehm insertion requires a positive knot interval.")
        alpha_lower = np.nextafter(
            np.nextafter(numerator, -np.inf) / np.nextafter(denominator, np.inf),
            -np.inf,
        )
        alpha_upper = np.nextafter(
            np.nextafter(numerator, np.inf) / denominator_lower, np.inf
        )
        alpha_lower, alpha_upper = max(0.0, alpha_lower), min(1.0, alpha_upper)
        other_lower = max(0.0, np.nextafter(1.0 - alpha_upper, -np.inf))
        other_upper = min(1.0, np.nextafter(1.0 - alpha_lower, np.inf))
        first = np.stack(
            (
                alpha_lower * lower[index],
                alpha_upper * lower[index],
                alpha_lower * upper[index],
                alpha_upper * upper[index],
            )
        )
        second = np.stack(
            (
                other_lower * lower[index - 1],
                other_upper * lower[index - 1],
                other_lower * upper[index - 1],
                other_upper * upper[index - 1],
            )
        )
        inserted_lower[index] = np.nextafter(
            np.nextafter(np.min(first, axis=0), -np.inf)
            + np.nextafter(np.min(second, axis=0), -np.inf),
            -np.inf,
        )
        inserted_upper[index] = np.nextafter(
            np.nextafter(np.max(first, axis=0), np.inf)
            + np.nextafter(np.max(second, axis=0), np.inf),
            np.inf,
        )
    return inserted_lower, inserted_upper, np.insert(knots, span + 1, knot)


def bezier_refinement(
    controls: np.ndarray,
    knots: np.ndarray,
    degree: int,
    axis: int,
    /,
    *,
    control_upper: np.ndarray | None = None,
    exact: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract Bezier coefficient nets by knot insertion.

    Without ``control_upper``, returns ``(pieces, span_bounds)``. With an upper
    coefficient bank, ``controls`` and ``control_upper`` bound each source
    coefficient and the result is ``(lower_pieces, upper_pieces, span_bounds)``.
    Exact mode retains Fraction and RadicalElement values without rounding.
    """
    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError("B-spline degree must be an integer.")
    if isinstance(axis, bool) or not isinstance(axis, Integral):
        raise TypeError("B-spline refinement axis must be an integer.")
    degree_, axis_ = int(degree), int(axis)
    source = np.asarray(controls, dtype=object if exact else np.float64)
    if source.ndim == 0 or not -source.ndim <= axis_ < source.ndim:
        raise ValueError("Bezier refinement axis must select a coefficient axis.")
    axis_ %= source.ndim
    if degree_ < 1:
        raise ValueError("Bezier refinement requires a positive degree.")
    if exact and control_upper is not None:
        raise ValueError("Exact Bezier extraction cannot use rounded coefficient bounds.")
    if exact:
        values_source = np.asarray(
            [_exact_scalar(value, "control coefficient") for value in source.reshape(-1)],
            dtype=object,
        ).reshape(source.shape)
        values = np.moveaxis(values_source, axis_, 0)
        raw_knots = np.asarray(knots, dtype=object).reshape(-1)
        knots_ = np.asarray(
            [_exact_scalar(value, "knot") for value in raw_knots], dtype=object
        )
    else:
        values = np.moveaxis(source, axis_, 0)
        knots_ = np.asarray(knots, dtype=np.float64).reshape(-1)
    if knots_.size != values.shape[0] + degree_ + 1:
        raise ValueError("Knot count must equal control count + degree + 1.")
    try:
        finite_knots = np.asarray(knots_, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("B-spline knots must have finite numeric realizations.") from exc
    if not np.all(np.isfinite(finite_knots)) or np.any(np.diff(knots_) < 0):
        raise ValueError("B-spline knots must be finite and nondecreasing.")
    upper_values = None
    if control_upper is not None:
        lower_source = np.asarray(controls, dtype=np.float64)
        upper_source = np.asarray(control_upper, dtype=np.float64)
        if (
            upper_source.shape != lower_source.shape
            or np.any(lower_source > upper_source)
            or not np.all(np.isfinite(lower_source))
            or not np.all(np.isfinite(upper_source))
        ):
            raise ValueError("Coefficient bounds must be finite, aligned, and ordered.")
        values = np.moveaxis(lower_source, axis_, 0)
        upper_values = np.moveaxis(upper_source, axis_, 0)
    lower, upper = knots_[degree_], knots_[-degree_ - 1]
    if not lower < upper:
        raise ValueError("B-spline knots must define a positive active interval.")
    for endpoint in (lower, upper):
        while int(np.count_nonzero(knots_ == endpoint)) < degree_ + 1:
            if upper_values is None:
                values, knots_ = _insert_knot(values, knots_, degree_, endpoint)
            else:
                values, upper_values, knots_ = _insert_knot_bounds(
                    values, upper_values, knots_, degree_, float(endpoint)
                )
    begin = int(np.searchsorted(knots_, lower, side="left"))
    end = int(np.searchsorted(knots_, upper, side="right"))
    active_count = end - begin - degree_ - 1
    values = values[begin : begin + active_count]
    if upper_values is not None:
        upper_values = upper_values[begin : begin + active_count]
    knots_ = knots_[begin:end]
    for internal in np.unique(knots_[(knots_ > lower) & (knots_ < upper)]):
        while int(np.count_nonzero(knots_ == internal)) < degree_:
            if upper_values is None:
                values, knots_ = _insert_knot(values, knots_, degree_, internal)
            else:
                values, upper_values, knots_ = _insert_knot_bounds(
                    values, upper_values, knots_, degree_, float(internal)
                )
    breaks = np.unique(knots_)
    span_count = breaks.size - 1
    discontinuities = tuple(
        knot
        for knot in breaks[1:-1]
        if int(np.count_nonzero(knots_ == knot)) == degree_ + 1
    )
    expected_count = span_count * degree_ + 1 + len(discontinuities)
    if values.shape[0] != expected_count:
        raise ValueError("Knot multiplicities do not define complete Bezier spans.")
    starts = tuple(
        index * degree_
        + sum(discontinuity <= breaks[index] for discontinuity in discontinuities)
        for index in range(span_count)
    )
    pieces = np.stack([values[start : start + degree_ + 1] for start in starts])
    bounds = np.stack((breaks[:-1], breaks[1:]), axis=1)
    pieces = np.moveaxis(pieces, 1, axis_ + 1)
    if upper_values is None:
        return pieces, bounds
    upper_pieces = np.stack(
        [upper_values[start : start + degree_ + 1] for start in starts]
    )
    return pieces, np.moveaxis(upper_pieces, 1, axis_ + 1), bounds


def exact_bezier_refinement(
    controls: np.ndarray, knots: np.ndarray, degree: int, axis: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Bezier pieces with exact rational/algebraic Boehm insertion."""
    result = bezier_refinement(controls, knots, degree, axis, exact=True)
    if len(result) != 2:
        raise RuntimeError("Exact Bezier extraction returned an unexpected bound bank.")
    return result


def _exact_fraction(value: object, name: str, /) -> Fraction:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a rational scalar, not bool.")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, Integral):
        result = Fraction(int(value))
    elif isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        result = Fraction.from_float(number)
    else:
        try:
            scalar = np.asarray(value)
            if scalar.ndim != 0 or scalar.dtype.kind == "b":
                raise TypeError
            if scalar.dtype.kind in "iu":
                result = Fraction(int(scalar))
            elif scalar.dtype.kind == "f":
                result = Fraction.from_float(float(scalar))
            else:
                raise TypeError
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            raise TypeError(f"{name} must be a finite rational scalar.") from exc
    try:
        finite = np.isfinite(float(result))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must have a finite numeric realization.")
    return result


def _exact_scalar(value: object, name: str, /) -> ExactScalar:
    if isinstance(value, (Fraction, Integral, float, np.floating)):
        return _exact_fraction(value, name)
    from ..discretization._exact_algebraic_geometry import RadicalElement

    if not isinstance(value, RadicalElement):
        return _exact_fraction(value, name)
    result = RadicalElement.from_stored_fields(value)
    if not result.terms:
        return Fraction(0)
    if len(result.terms) == 1 and result.terms[0][0] == 1:
        return result.terms[0][1]
    try:
        finite = np.isfinite(float(result))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must have a finite numeric realization.")
    return result


def _exact_scalar_interval(
    value: ExactScalar, bits: int = 128, /
) -> tuple[Fraction, Fraction]:
    if isinstance(value, Fraction):
        return value, value
    from ..discretization._exact_algebraic_geometry import RadicalElement

    if not isinstance(value, RadicalElement):
        raise TypeError("Exact span arithmetic received an unsupported scalar type.")
    return value.interval(bits=bits)


def _positive_rational_lower_bound(value: ExactScalar, /) -> Fraction:
    if value <= 0:
        raise ValueError("A positive exact scalar is required for a lower bound.")
    if isinstance(value, Fraction):
        return value
    bits = 64
    while True:
        lower, _ = _exact_scalar_interval(value, bits)
        if lower > 0:
            return lower
        bits *= 2


def _exact_scalar_payload(value: ExactScalar, /) -> object:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    from ..discretization._exact_algebraic_geometry import RadicalElement

    if not isinstance(value, RadicalElement):
        raise TypeError("Exact span identity received an unsupported scalar type.")
    return {
        "radical_terms": [
            [radicand, coefficient.numerator, coefficient.denominator]
            for radicand, coefficient in value.terms
        ]
    }


def _univariate_span_extraction(
    knots: tuple[ExactScalar, ...], degree: int, span: int, /
) -> tuple[tuple[ExactScalar, ...], ...]:
    if degree == 0:
        return ((Fraction(1),),)
    local_knots = np.asarray(knots[span - degree : span + degree + 2], dtype=object)
    identity = np.asarray(
        [
            [Fraction(int(row == column)) for column in range(degree + 1)]
            for row in range(degree + 1)
        ],
        dtype=object,
    )
    pieces, bounds = exact_bezier_refinement(identity, local_knots, degree, 0)
    if pieces.shape != (1, degree + 1, degree + 1) or tuple(bounds[0]) != (
        knots[span],
        knots[span + 1],
    ):
        raise ValueError("Knot insertion did not isolate the requested B-spline span.")
    return tuple(
        tuple(
            _exact_scalar(pieces[0, bernstein, control], "span extraction coefficient")
            for bernstein in range(degree + 1)
        )
        for control in range(degree + 1)
    )


def tensor_bspline_span_extraction(
    u_knots: Sequence[ExactScalar | int | float],
    v_knots: Sequence[ExactScalar | int | float],
    u_degree: int,
    v_degree: int,
    u_span: int,
    v_span: int,
    control_point_ids: Sequence[int],
    /,
) -> tuple[tuple[int, ...], tuple[tuple[ExactScalar, ...], ...]]:
    """Return exact local extraction and persistent IDs for one tensor span.

    Span indices are expanded-knot interval indices. The supplied global IDs
    are u-major/v-fast, matching a flattened ``(n_u, n_v)`` control net. The
    returned active IDs use the same order. Extraction rows follow those local
    IDs, and columns use v-fast tensor Bernstein ordering. Only the local
    support and dense local operator are materialized.
    """
    degrees, spans = (u_degree, v_degree), (u_span, v_span)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in degrees
    ):
        raise TypeError("Tensor B-spline degrees must be integers.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in spans):
        raise TypeError("Tensor B-spline spans must be integer knot-vector indices.")
    degrees_ = tuple(int(value) for value in degrees)
    spans_ = tuple(int(value) for value in spans)
    if any(value < 0 for value in degrees_):
        raise ValueError("Tensor B-spline degrees must be nonnegative.")
    knot_vectors = tuple(
        tuple(_exact_scalar(value, "knot") for value in axis_knots)
        for axis_knots in (u_knots, v_knots)
    )
    control_counts = tuple(
        len(knots) - degree - 1
        for knots, degree in zip(knot_vectors, degrees_, strict=True)
    )
    for axis, (knots, degree, span, count) in enumerate(
        zip(knot_vectors, degrees_, spans_, control_counts, strict=True)
    ):
        if any(right < left for left, right in zip(knots[:-1], knots[1:], strict=True)):
            raise ValueError("Tensor B-spline knots must be nondecreasing.")
        if count < degree + 1:
            raise ValueError("Each tensor B-spline axis requires degree + 1 controls.")
        if span < degree or span >= count:
            raise ValueError(f"Axis {axis} span lies outside its active knot indices.")
        if not knots[span] < knots[span + 1]:
            raise ValueError(f"Axis {axis} span must have strictly positive width.")
    raw_ids = tuple(control_point_ids)
    if len(raw_ids) != control_counts[0] * control_counts[1]:
        raise ValueError("Persistent IDs must cover the full tensor control bank.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in raw_ids
    ):
        raise TypeError("Persistent control-point IDs must be integers.")
    identifiers = tuple(int(value) for value in raw_ids)
    if any(value < 0 for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Persistent control-point IDs must be unique and nonnegative.")

    u_support = tuple(range(spans_[0] - degrees_[0], spans_[0] + 1))
    v_support = tuple(range(spans_[1] - degrees_[1], spans_[1] + 1))
    anchor_ids = tuple(
        identifiers[u_row * control_counts[1] + v_row]
        for u_row in u_support
        for v_row in v_support
    )
    u_operator = _univariate_span_extraction(knot_vectors[0], degrees_[0], spans_[0])
    v_operator = _univariate_span_extraction(knot_vectors[1], degrees_[1], spans_[1])
    operator = tuple(
        tuple(
            u_operator[u_row][u_bernstein] * v_operator[v_row][v_bernstein]
            for u_bernstein in range(degrees_[0] + 1)
            for v_bernstein in range(degrees_[1] + 1)
        )
        for u_row in range(degrees_[0] + 1)
        for v_row in range(degrees_[1] + 1)
    )
    return anchor_ids, operator


def _bernstein_product(
    left: Sequence[ExactScalar], right: Sequence[ExactScalar], /
) -> tuple[ExactScalar, ...]:
    left_degree, right_degree = len(left) - 1, len(right) - 1
    result: list[ExactScalar] = [
        Fraction(0) for _ in range(left_degree + right_degree + 1)
    ]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += (
                Fraction(
                    comb(left_degree, left_index) * comb(right_degree, right_index),
                    comb(left_degree + right_degree, left_index + right_index),
                )
                * left_value
                * right_value
            )
    return tuple(result)


def _bernstein_linear_combination(
    terms: Sequence[tuple[ExactScalar, Sequence[ExactScalar]]], /
) -> tuple[ExactScalar, ...]:
    if not terms:
        raise ValueError("A Bernstein combination requires at least one term.")
    width = len(terms[0][1])
    if width == 0 or any(len(coefficients) != width for _, coefficients in terms):
        raise ValueError("Bernstein combination terms must have a common degree.")
    return tuple(
        sum((scale * coefficients[index] for scale, coefficients in terms), Fraction(0))
        for index in range(width)
    )


def _bernstein_derivative(
    coefficients: Sequence[ExactScalar],
    parameter_bounds: tuple[ExactScalar, ExactScalar],
    /,
) -> tuple[ExactScalar, ...]:
    degree = len(coefficients) - 1
    if degree == 0:
        return (Fraction(0),)
    width = parameter_bounds[1] - parameter_bounds[0]
    if width <= 0:
        raise ValueError("Bernstein derivatives require a positive span width.")
    return tuple(
        Fraction(degree) * (coefficients[index + 1] - coefficients[index]) / width
        for index in range(degree)
    )


def _bernstein_split(
    coefficients: Sequence[ExactScalar], parameter: ExactScalar, /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]]:
    if not 0 <= parameter <= 1:
        raise ValueError("A Bernstein split parameter must lie in [0, 1].")
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


def _bernstein_restrict(
    coefficients: Sequence[ExactScalar],
    lower: ExactScalar,
    upper: ExactScalar,
    /,
) -> tuple[ExactScalar, ...]:
    if not 0 <= lower <= upper <= 1:
        raise ValueError(
            "Bernstein restriction bounds must satisfy 0 <= lower <= upper <= 1."
        )
    if lower == upper:
        left, _ = _bernstein_split(coefficients, lower)
        return (left[-1],)
    if lower == 0:
        right = tuple(coefficients)
    else:
        _, right = _bernstein_split(coefficients, lower)
    if upper == 1:
        return right
    relative_upper = (upper - lower) / (1 - lower)
    restricted, _ = _bernstein_split(right, relative_upper)
    return restricted


def _bernstein_bounds(
    coefficients: Sequence[ExactScalar],
    lower: ExactScalar,
    upper: ExactScalar,
    /,
    *,
    positive_lower: bool = False,
) -> tuple[Fraction, Fraction]:
    restricted = _bernstein_restrict(coefficients, lower, upper)
    exact_lower, exact_upper = min(restricted), max(restricted)
    lower_bound = (
        _positive_rational_lower_bound(exact_lower)
        if positive_lower
        else _exact_scalar_interval(exact_lower)[0]
    )
    upper_bound = _exact_scalar_interval(exact_upper)[1]
    return lower_bound, upper_bound


def _positive_quotient_bounds(
    numerator: tuple[Fraction, Fraction],
    denominator: tuple[Fraction, Fraction],
    /,
) -> tuple[Fraction, Fraction]:
    lower_denominator, upper_denominator = denominator
    if lower_denominator <= 0 or upper_denominator < lower_denominator:
        raise ValueError("Rational Bernstein enclosures require a positive denominator.")
    candidates = tuple(
        numerator_value / denominator_value
        for numerator_value in numerator
        for denominator_value in denominator
    )
    return min(candidates), max(candidates)


def _float_bound(value: Fraction, lower: bool, /) -> float:
    maximum = Fraction.from_float(float(np.finfo(np.float64).max))
    if value > maximum:
        return float(np.finfo(np.float64).max) if lower else float("inf")
    if value < -maximum:
        return float("-inf") if lower else -float(np.finfo(np.float64).max)
    rounded = float(value)
    represented = Fraction.from_float(rounded)
    if lower and represented > value:
        return nextafter(rounded, float("-inf"))
    if not lower and represented < value:
        return nextafter(rounded, float("inf"))
    return rounded


@dataclass(frozen=True, slots=True)
class RationalBSplineCurveFloatEnclosure:
    """Outward binary64 view of exact rational curve-span bounds."""

    source_id: str
    span_index: int
    span_parameter_bounds: tuple[Fraction, Fraction]
    parameter_bounds: tuple[Fraction, Fraction]
    continuity_at_lower: int | None
    continuity_at_upper: int | None
    position_lower: tuple[float, ...]
    position_upper: tuple[float, ...]
    first_derivative_lower: tuple[float, ...]
    first_derivative_upper: tuple[float, ...]
    second_derivative_lower: tuple[float, ...]
    second_derivative_upper: tuple[float, ...]
    denominator_lower: float
    denominator_upper: float


@dataclass(frozen=True, slots=True)
class RationalBSplineCurveSpanEnclosure:
    """Rational outward bounds for one side of one exact rational spline span."""

    source_id: str
    span_index: int
    span_parameter_bounds: tuple[ExactScalar, ExactScalar]
    parameter_bounds: tuple[ExactScalar, ExactScalar]
    continuity_at_lower: int | None
    continuity_at_upper: int | None
    value_bounds: tuple[tuple[Fraction, Fraction], ...]
    first_derivative_bounds: tuple[tuple[Fraction, Fraction], ...]
    second_derivative_bounds: tuple[tuple[Fraction, Fraction], ...]
    denominator_bounds: tuple[Fraction, Fraction]

    @property
    def positive_denominator(self) -> bool:
        return self.denominator_bounds[0] > 0

    def outward_float_bounds(self, /) -> RationalBSplineCurveFloatEnclosure:
        span_bounds = (
            _exact_scalar_interval(self.span_parameter_bounds[0])[0],
            _exact_scalar_interval(self.span_parameter_bounds[1])[1],
        )
        parameter_bounds = (
            _exact_scalar_interval(self.parameter_bounds[0])[0],
            _exact_scalar_interval(self.parameter_bounds[1])[1],
        )
        return RationalBSplineCurveFloatEnclosure(
            self.source_id,
            self.span_index,
            span_bounds,
            parameter_bounds,
            self.continuity_at_lower,
            self.continuity_at_upper,
            tuple(_float_bound(lower, True) for lower, _ in self.value_bounds),
            tuple(_float_bound(upper, False) for _, upper in self.value_bounds),
            tuple(_float_bound(lower, True) for lower, _ in self.first_derivative_bounds),
            tuple(
                _float_bound(upper, False) for _, upper in self.first_derivative_bounds
            ),
            tuple(
                _float_bound(lower, True) for lower, _ in self.second_derivative_bounds
            ),
            tuple(
                _float_bound(upper, False) for _, upper in self.second_derivative_bounds
            ),
            _float_bound(self.denominator_bounds[0], True),
            _float_bound(self.denominator_bounds[1], False),
        )


@dataclass(frozen=True, slots=True)
class RationalBSplineCurveSpan:
    """Exact local homogeneous Bernstein representation of one knot span."""

    source_id: str
    span_index: int
    degree: int
    coefficient_rows: tuple[int, ...]
    parameter_bounds: tuple[ExactScalar, ExactScalar]
    source_parameter_bounds: tuple[ExactScalar, ExactScalar]
    continuity_at_lower: int | None
    continuity_at_upper: int | None
    homogeneous_bernstein_coefficients: tuple[tuple[ExactScalar, ...], ...]
    positive_denominator_lower_bound: Fraction
    _first_derivative_numerators: tuple[tuple[ExactScalar, ...], ...] = field(repr=False)
    _second_derivative_numerators: tuple[tuple[ExactScalar, ...], ...] = field(repr=False)
    _denominator_squared: tuple[ExactScalar, ...] = field(repr=False)
    _denominator_cubed: tuple[ExactScalar, ...] = field(repr=False)

    def enclose(
        self,
        parameter_bounds: Sequence[ExactScalar | int | float],
        /,
    ) -> RationalBSplineCurveSpanEnclosure:
        if len(parameter_bounds) != 2:
            raise ValueError("A spline span enclosure requires two parameter endpoints.")
        lower, upper = tuple(
            _exact_scalar(value, "parameter endpoint") for value in parameter_bounds
        )
        span_lower, span_upper = self.parameter_bounds
        if not span_lower <= lower <= upper <= span_upper:
            raise ValueError(
                "Enclosure bounds must lie inside one exact nonempty knot span."
            )
        span_width = span_upper - span_lower
        relative_lower = (lower - span_lower) / span_width
        relative_upper = (upper - span_lower) / span_width
        homogeneous = self.homogeneous_bernstein_coefficients
        denominator = tuple(row[-1] for row in homogeneous)
        denominator_bounds = _bernstein_bounds(
            denominator, relative_lower, relative_upper, positive_lower=True
        )
        value_bounds = tuple(
            _positive_quotient_bounds(
                _bernstein_bounds(
                    tuple(row[component] for row in homogeneous),
                    relative_lower,
                    relative_upper,
                ),
                denominator_bounds,
            )
            for component in range(len(homogeneous[0]) - 1)
        )
        denominator_squared_bounds = _bernstein_bounds(
            self._denominator_squared,
            relative_lower,
            relative_upper,
            positive_lower=True,
        )
        denominator_cubed_bounds = _bernstein_bounds(
            self._denominator_cubed,
            relative_lower,
            relative_upper,
            positive_lower=True,
        )
        first_derivative_bounds = tuple(
            _positive_quotient_bounds(
                _bernstein_bounds(values, relative_lower, relative_upper),
                denominator_squared_bounds,
            )
            for values in self._first_derivative_numerators
        )
        second_derivative_bounds = tuple(
            _positive_quotient_bounds(
                _bernstein_bounds(values, relative_lower, relative_upper),
                denominator_cubed_bounds,
            )
            for values in self._second_derivative_numerators
        )
        return RationalBSplineCurveSpanEnclosure(
            self.source_id,
            self.span_index,
            self.parameter_bounds,
            (lower, upper),
            self.continuity_at_lower,
            self.continuity_at_upper,
            value_bounds,
            first_derivative_bounds,
            second_derivative_bounds,
            denominator_bounds,
        )


def rational_bspline_curve_spans(
    control_points: Sequence[Sequence[ExactScalar | int | float]],
    weights: Sequence[ExactScalar | int | float],
    knots: Sequence[ExactScalar | int | float],
    degree: int,
    /,
    *,
    exact_control_points: Sequence[Sequence[ExactScalar | int | float]] | None = None,
    exact_weights: Sequence[ExactScalar | int | float] | None = None,
    exact_knots: Sequence[ExactScalar | int | float] | None = None,
) -> tuple[RationalBSplineCurveSpan, ...]:
    """Build source-bound exact rational curve-span enclosures.

    Float leaves are exact dyadics only when no exact source bank is supplied.
    Explicit rational/algebraic banks are checked against their one binary64
    realization before exact span construction. Each active knot interval
    remains a separate one-sided span, including at repeated knots.
    """
    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError("B-spline degree must be an integer.")
    degree_ = int(degree)
    if degree_ < 0:
        raise ValueError("B-spline degree must be nonnegative.")
    raw_points = tuple(tuple(row) for row in control_points)
    point_source = (
        raw_points
        if exact_control_points is None
        else tuple(tuple(row) for row in exact_control_points)
    )
    points = tuple(
        tuple(_exact_scalar(value, "control point") for value in row)
        for row in point_source
    )
    raw_weights = tuple(weights)
    weight_source = raw_weights if exact_weights is None else tuple(exact_weights)
    weights_ = tuple(_exact_scalar(value, "weight") for value in weight_source)
    raw_knots = tuple(knots)
    knot_source = raw_knots if exact_knots is None else tuple(exact_knots)
    knots_ = tuple(_exact_scalar(value, "knot") for value in knot_source)
    if (
        not points
        or not points[0]
        or len(points) != len(raw_points)
        or any(len(row) != len(raw_points[index]) for index, row in enumerate(points))
        or any(len(row) != len(points[0]) for row in points)
        or len(weights_) != len(raw_weights)
        or len(knots_) != len(raw_knots)
    ):
        raise ValueError(
            "Rational curve exact source banks must align with numeric leaves."
        )
    if exact_control_points is not None and any(
        float(numeric) != float(exact)
        for numeric_row, exact_row in zip(raw_points, points, strict=True)
        for numeric, exact in zip(numeric_row, exact_row, strict=True)
    ):
        raise ValueError("Exact B-spline controls must match their binary64 realization.")
    if exact_weights is not None and any(
        float(numeric) != float(exact)
        for numeric, exact in zip(raw_weights, weights_, strict=True)
    ):
        raise ValueError("Exact B-spline weights must match their binary64 realization.")
    if exact_knots is not None and any(
        float(numeric) != float(exact)
        for numeric, exact in zip(raw_knots, knots_, strict=True)
    ):
        raise ValueError("Exact B-spline knots must match their binary64 realization.")
    if len(weights_) != len(points) or len(knots_) != len(points) + degree_ + 1:
        raise ValueError("Rational curve controls, weights, and knots must align.")
    if any(weight <= 0 for weight in weights_):
        raise ValueError("Rational curve weights must be strictly positive.")
    if any(right < left for left, right in zip(knots_[:-1], knots_[1:], strict=True)):
        raise ValueError("Rational curve knots must be nondecreasing.")
    knot_multiplicities = Counter(knots_)
    if any(count > degree_ + 1 for count in knot_multiplicities.values()):
        raise ValueError("Rational curve knot multiplicity exceeds degree + 1.")
    active_lower, active_upper = knots_[degree_], knots_[len(points)]
    if not active_lower < active_upper:
        raise ValueError("Rational curve knots must define a positive active interval.")
    source_id = canonical_fingerprint(
        {
            "kind": "exact-rational-bspline-curve-source",
            "degree": degree_,
            "knots": [_exact_scalar_payload(value) for value in knots_],
            "control_points": [
                [_exact_scalar_payload(value) for value in row] for row in points
            ],
            "weights": [_exact_scalar_payload(value) for value in weights_],
        }
    )
    homogeneous = tuple(
        tuple(weight * value for value in point) + (weight,)
        for point, weight in zip(points, weights_, strict=True)
    )
    spans: list[RationalBSplineCurveSpan] = []
    for span_index in range(degree_, len(points)):
        span_lower, span_upper = knots_[span_index], knots_[span_index + 1]
        if span_lower >= span_upper:
            continue
        coefficient_rows = tuple(range(span_index - degree_, span_index + 1))
        extraction = _univariate_span_extraction(knots_, degree_, span_index)
        local_homogeneous = tuple(homogeneous[row] for row in coefficient_rows)
        bernstein = tuple(
            tuple(
                sum(
                    (
                        extraction[row][control] * local_homogeneous[row][component]
                        for row in range(degree_ + 1)
                    ),
                    Fraction(0),
                )
                for component in range(len(points[0]) + 1)
            )
            for control in range(degree_ + 1)
        )
        denominator = tuple(row[-1] for row in bernstein)
        exact_denominator_lower = min(denominator)
        if exact_denominator_lower <= 0:
            raise ValueError(
                "Exact knot extraction did not certify a positive span denominator."
            )
        denominator_lower = _positive_rational_lower_bound(exact_denominator_lower)
        span_bounds = (span_lower, span_upper)
        denominator_first = _bernstein_derivative(denominator, span_bounds)
        denominator_second = _bernstein_derivative(denominator_first, span_bounds)
        denominator_squared = _bernstein_product(denominator, denominator)
        denominator_cubed = _bernstein_product(denominator_squared, denominator)
        first_numerators: list[tuple[ExactScalar, ...]] = []
        second_numerators: list[tuple[ExactScalar, ...]] = []
        for component in range(len(points[0])):
            numerator = tuple(row[component] for row in bernstein)
            numerator_first = _bernstein_derivative(numerator, span_bounds)
            numerator_second = _bernstein_derivative(numerator_first, span_bounds)
            first_numerators.append(
                _bernstein_linear_combination(
                    (
                        (Fraction(1), _bernstein_product(numerator_first, denominator)),
                        (Fraction(-1), _bernstein_product(numerator, denominator_first)),
                    )
                )
            )
            if degree_ == 0:
                second_numerator = (Fraction(0),)
            elif degree_ == 1:
                second_numerator = _bernstein_linear_combination(
                    (
                        (
                            Fraction(-2),
                            _bernstein_product(
                                _bernstein_product(numerator_first, denominator),
                                denominator_first,
                            ),
                        ),
                        (
                            Fraction(2),
                            _bernstein_product(
                                numerator,
                                _bernstein_product(denominator_first, denominator_first),
                            ),
                        ),
                    )
                )
            else:
                second_numerator = _bernstein_linear_combination(
                    (
                        (
                            Fraction(1),
                            _bernstein_product(numerator_second, denominator_squared),
                        ),
                        (
                            Fraction(-1),
                            _bernstein_product(
                                _bernstein_product(numerator, denominator_second),
                                denominator,
                            ),
                        ),
                        (
                            Fraction(-2),
                            _bernstein_product(
                                _bernstein_product(numerator_first, denominator),
                                denominator_first,
                            ),
                        ),
                        (
                            Fraction(2),
                            _bernstein_product(
                                numerator,
                                _bernstein_product(denominator_first, denominator_first),
                            ),
                        ),
                    )
                )
            second_numerators.append(second_numerator)
        continuity_at_lower = (
            None
            if span_lower == active_lower
            else degree_ - knot_multiplicities[span_lower]
        )
        continuity_at_upper = (
            None
            if span_upper == active_upper
            else degree_ - knot_multiplicities[span_upper]
        )
        spans.append(
            RationalBSplineCurveSpan(
                source_id,
                span_index,
                degree_,
                coefficient_rows,
                span_bounds,
                (active_lower, active_upper),
                continuity_at_lower,
                continuity_at_upper,
                bernstein,
                denominator_lower,
                tuple(first_numerators),
                tuple(second_numerators),
                denominator_squared,
                denominator_cubed,
            )
        )
    if not spans:
        raise ValueError("Rational curve source contains no nonempty active knot spans.")
    return tuple(spans)


@dataclass(frozen=True, slots=True)
class RationalBSplineCurveContinuity:
    """Exact one-sided endpoint jets and continuity at one knot or closed seam."""

    source_id: str
    span_index_before: int
    span_index_after: int
    endpoint_parameters: tuple[ExactScalar, ExactScalar]
    guaranteed_continuity_order: int | None
    observed_continuity_order: int
    left_endpoint: RationalBSplineCurveSpanEnclosure
    right_endpoint: RationalBSplineCurveSpanEnclosure
    closed_seam: bool


@dataclass(frozen=True, slots=True)
class RationalBSplineCurveContinuityReport:
    """Source-bound C0/C1/C2 decisions at every knot and optional closed seam."""

    source_id: str
    junctions: tuple[RationalBSplineCurveContinuity, ...]
    closed_seam: RationalBSplineCurveContinuity | None


def _observed_continuity_order(
    left: RationalBSplineCurveSpanEnclosure,
    right: RationalBSplineCurveSpanEnclosure,
    /,
) -> int:
    order = -1
    for derivative_bounds_left, derivative_bounds_right in zip(
        (
            left.value_bounds,
            left.first_derivative_bounds,
            left.second_derivative_bounds,
        ),
        (
            right.value_bounds,
            right.first_derivative_bounds,
            right.second_derivative_bounds,
        ),
        strict=True,
    ):
        if derivative_bounds_left != derivative_bounds_right:
            break
        order += 1
    return order


def rational_bspline_curve_continuity(
    spans: Sequence[RationalBSplineCurveSpan],
    /,
    *,
    closed: bool = False,
) -> RationalBSplineCurveContinuityReport:
    """Prove exact one-sided C0/C1/C2 continuity at knots and an optional seam."""
    spans_ = tuple(spans)
    if not spans_ or any(
        not isinstance(span, RationalBSplineCurveSpan) for span in spans_
    ):
        raise ValueError(
            "Continuity evidence requires a nonempty rational spline span chain."
        )
    source_id = spans_[0].source_id
    if any(span.source_id != source_id for span in spans_):
        raise ValueError("Continuity spans must share one authoritative source identity.")
    if type(closed) is not bool:
        raise TypeError("closed must be a boolean seam request.")
    active_bounds = spans_[0].source_parameter_bounds
    if (
        any(span.source_parameter_bounds != active_bounds for span in spans_)
        or spans_[0].parameter_bounds[0] != active_bounds[0]
        or spans_[-1].parameter_bounds[1] != active_bounds[1]
    ):
        raise ValueError(
            "Continuity evidence must cover the full source parameter domain."
        )
    junctions: list[RationalBSplineCurveContinuity] = []
    for before, after in zip(spans_[:-1], spans_[1:], strict=True):
        knot = before.parameter_bounds[1]
        if (
            knot != after.parameter_bounds[0]
            or before.span_index >= after.span_index
            or before.continuity_at_upper != after.continuity_at_lower
        ):
            raise ValueError(
                "Rational spline spans must form an ordered, gap-free knot chain."
            )
        left_endpoint = before.enclose((knot, knot))
        right_endpoint = after.enclose((knot, knot))
        junctions.append(
            RationalBSplineCurveContinuity(
                source_id,
                before.span_index,
                after.span_index,
                (knot, knot),
                before.continuity_at_upper,
                _observed_continuity_order(left_endpoint, right_endpoint),
                left_endpoint,
                right_endpoint,
                False,
            )
        )
    seam = None
    if closed:
        before, after = spans_[-1], spans_[0]
        before_parameter = before.parameter_bounds[1]
        after_parameter = after.parameter_bounds[0]
        left_endpoint = before.enclose((before_parameter, before_parameter))
        right_endpoint = after.enclose((after_parameter, after_parameter))
        seam = RationalBSplineCurveContinuity(
            source_id,
            before.span_index,
            after.span_index,
            (before_parameter, after_parameter),
            None,
            _observed_continuity_order(left_endpoint, right_endpoint),
            left_endpoint,
            right_endpoint,
            True,
        )
    return RationalBSplineCurveContinuityReport(source_id, tuple(junctions), seam)


def _tensor_bernstein_shape(
    coefficients: Sequence[Sequence[ExactScalar]], /
) -> tuple[int, int]:
    if not coefficients or not coefficients[0]:
        raise ValueError("Tensor Bernstein coefficients must be a nonempty net.")
    width = len(coefficients[0])
    if any(len(row) != width for row in coefficients):
        raise ValueError("Tensor Bernstein coefficient nets must be rectangular.")
    return len(coefficients) - 1, width - 1


def _tensor_bernstein_product(
    first: Sequence[Sequence[ExactScalar]],
    second: Sequence[Sequence[ExactScalar]],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    first_u, first_v = _tensor_bernstein_shape(first)
    second_u, second_v = _tensor_bernstein_shape(second)
    result: list[list[ExactScalar]] = [
        [Fraction(0) for _ in range(first_v + second_v + 1)]
        for _ in range(first_u + second_u + 1)
    ]
    for first_u_index in range(first_u + 1):
        for first_v_index in range(first_v + 1):
            for second_u_index in range(second_u + 1):
                for second_v_index in range(second_v + 1):
                    factor = Fraction(
                        comb(first_u, first_u_index) * comb(second_u, second_u_index),
                        comb(first_u + second_u, first_u_index + second_u_index),
                    ) * Fraction(
                        comb(first_v, first_v_index) * comb(second_v, second_v_index),
                        comb(first_v + second_v, first_v_index + second_v_index),
                    )
                    result[first_u_index + second_u_index][
                        first_v_index + second_v_index
                    ] += (
                        factor
                        * first[first_u_index][first_v_index]
                        * second[second_u_index][second_v_index]
                    )
    return tuple(tuple(row) for row in result)


def _tensor_bernstein_linear_combination(
    terms: Sequence[tuple[ExactScalar, Sequence[Sequence[ExactScalar]]]],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    if not terms:
        raise ValueError("A tensor Bernstein combination requires at least one term.")
    shape = _tensor_bernstein_shape(terms[0][1])
    if any(_tensor_bernstein_shape(net) != shape for _, net in terms):
        raise ValueError("Tensor Bernstein combination nets must have one degree.")
    return tuple(
        tuple(
            sum(
                (scale * net[u_index][v_index] for scale, net in terms),
                Fraction(0),
            )
            for v_index in range(shape[1] + 1)
        )
        for u_index in range(shape[0] + 1)
    )


def _tensor_bernstein_derivative(
    coefficients: Sequence[Sequence[ExactScalar]],
    axis: int,
    parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    degree_u, degree_v = _tensor_bernstein_shape(coefficients)
    if axis == 0:
        degree = degree_u
        if degree == 0:
            return tuple(
                tuple(Fraction(0) for _ in range(degree_v + 1)) for _ in range(1)
            )
        width = parameter_bounds[0][1] - parameter_bounds[0][0]
        if width <= 0:
            raise ValueError("Tensor Bernstein derivatives require positive span widths.")
        return tuple(
            tuple(
                Fraction(degree)
                * (coefficients[u_index + 1][v_index] - coefficients[u_index][v_index])
                / width
                for v_index in range(degree_v + 1)
            )
            for u_index in range(degree)
        )
    if axis == 1:
        degree = degree_v
        if degree == 0:
            return tuple(
                tuple(Fraction(0) for _ in range(1)) for _ in range(degree_u + 1)
            )
        width = parameter_bounds[1][1] - parameter_bounds[1][0]
        if width <= 0:
            raise ValueError("Tensor Bernstein derivatives require positive span widths.")
        return tuple(
            tuple(
                Fraction(degree)
                * (coefficients[u_index][v_index + 1] - coefficients[u_index][v_index])
                / width
                for v_index in range(degree)
            )
            for u_index in range(degree_u + 1)
        )
    raise ValueError("Tensor Bernstein differentiation axis must be zero or one.")


# One de Casteljau step ``(1 - t) * a + t * b`` rounds ``1 - t``, two products
# and one sum; four units of roundoff per magnitude bound all of them, and the
# smallest subnormal covers gradual underflow.
_INTERVAL_ROUNDING = 4.0 * float(np.finfo(np.float64).eps)
_INTERVAL_UNDERFLOW = float(np.finfo(np.float64).smallest_subnormal)


def _binary64_down(value: Fraction, /) -> float:
    result = float(value)
    return result if Fraction(result) <= value else nextafter(result, -np.inf)


def _binary64_up(value: Fraction, /) -> float:
    result = float(value)
    return result if Fraction(result) >= value else nextafter(result, np.inf)


def _interval_net(
    coefficients: Sequence[Sequence[ExactScalar]], /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward binary64 enclosure of an exact tensor Bernstein net."""
    _tensor_bernstein_shape(coefficients)
    intervals = [[_exact_scalar_interval(value) for value in row] for row in coefficients]
    lower = np.asarray(
        [[_binary64_down(low) for low, _ in row] for row in intervals], dtype=np.float64
    )
    upper = np.asarray(
        [[_binary64_up(high) for _, high in row] for row in intervals], dtype=np.float64
    )
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("Bernstein interval nets require finite binary64 enclosures.")
    return lower, upper


def _interval_casteljau_step(
    lower: np.ndarray,
    upper: np.ndarray,
    parameter: tuple[float, float],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose ``(1 - t) x_i + t x_{i+1}`` along axis 0 for every ``t`` in range.

    The combination is increasing in both coefficients and affine in ``t``, so
    its extremes over the boxes occur at coefficient and parameter endpoints;
    each endpoint value is then widened by its rounding error bound.
    """
    candidates_lower = []
    candidates_upper = []
    for t in parameter:
        complement = 1.0 - t
        for values, out, direction in (
            (lower, candidates_lower, -np.inf),
            (upper, candidates_upper, np.inf),
        ):
            combined = complement * values[:-1] + t * values[1:]
            error = (
                _INTERVAL_ROUNDING
                * (complement * np.abs(values[:-1]) + t * np.abs(values[1:]))
                + _INTERVAL_UNDERFLOW
            )
            widened = combined - error if direction < 0 else combined + error
            out.append(np.nextafter(widened, direction))
    return (
        np.minimum(candidates_lower[0], candidates_lower[1]),
        np.maximum(candidates_upper[0], candidates_upper[1]),
    )


def _interval_parameter(value: ExactScalar, /) -> tuple[float, float]:
    low, high = _exact_scalar_interval(value)
    return max(0.0, _binary64_down(low)), min(1.0, _binary64_up(high))


def _interval_ratio(
    numerator: ExactScalar, denominator: ExactScalar, /
) -> tuple[float, float]:
    """Enclose ``numerator / denominator`` in ``[0, 1]`` for ``0 < numerator <= denominator``."""
    if isinstance(numerator, Fraction) and isinstance(denominator, Fraction):
        return _interval_parameter(numerator / denominator)
    numerator_low, numerator_high = _exact_scalar_interval(numerator)
    denominator_low, denominator_high = _exact_scalar_interval(denominator)
    if numerator_low < 0 or denominator_low <= 0:
        return 0.0, 1.0
    return (
        max(0.0, _binary64_down(numerator_low / denominator_high)),
        min(1.0, _binary64_up(numerator_high / denominator_low)),
    )


def _interval_restrict_axis0(
    lower: np.ndarray,
    upper: np.ndarray,
    first: ExactScalar,
    last: ExactScalar,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a Bernstein interval net to ``[first, last]`` of ``[0, 1]`` on axis 0."""
    if not 0 <= first <= last <= 1:
        raise ValueError(
            "Bernstein restriction bounds must satisfy 0 <= lower <= upper <= 1."
        )
    if last < 1:
        parameter = _interval_parameter(last)
        left_lower, left_upper = [lower[0]], [upper[0]]
        row_lower, row_upper = lower, upper
        while row_lower.shape[0] > 1:
            row_lower, row_upper = _interval_casteljau_step(
                row_lower, row_upper, parameter
            )
            left_lower.append(row_lower[0])
            left_upper.append(row_upper[0])
        lower, upper = np.stack(left_lower), np.stack(left_upper)
    if first > 0:
        parameter = _interval_ratio(first, last)
        right_lower, right_upper = [lower[-1]], [upper[-1]]
        row_lower, row_upper = lower, upper
        while row_lower.shape[0] > 1:
            row_lower, row_upper = _interval_casteljau_step(
                row_lower, row_upper, parameter
            )
            right_lower.append(row_lower[-1])
            right_upper.append(row_upper[-1])
        lower, upper = np.stack(right_lower[::-1]), np.stack(right_upper[::-1])
    return lower, upper


def _interval_tensor_restrict(
    net: tuple[np.ndarray, np.ndarray],
    lower_u: ExactScalar,
    upper_u: ExactScalar,
    lower_v: ExactScalar,
    upper_v: ExactScalar,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Outward enclosure of the exact tensor net restricted to a relative sub-box.

    Every restricted coefficient is a convex combination of the source
    coefficients; each binary64 step is rounded outward, so the returned
    interval net contains the exact restricted net coefficientwise.
    """
    lower, upper = _interval_restrict_axis0(net[0], net[1], lower_u, upper_u)
    lower, upper = _interval_restrict_axis0(lower.T, upper.T, lower_v, upper_v)
    return lower.T, upper.T


def _interval_tensor_bounds(
    net: tuple[np.ndarray, np.ndarray],
    relative_bounds: tuple[ExactScalar, ExactScalar, ExactScalar, ExactScalar],
    /,
) -> tuple[Fraction, Fraction]:
    """Rational bounds of a tensor Bernstein polynomial on a relative sub-box."""
    lower, upper = _interval_tensor_restrict(net, *relative_bounds)
    minimum, maximum = float(np.min(lower)), float(np.max(upper))
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        raise ValueError("Bernstein interval restriction overflowed binary64.")
    return Fraction(minimum), Fraction(maximum)


@dataclass(frozen=True, slots=True)
class RationalBSplineSurfaceSpanEnclosure:
    """Exact rational position and Jacobian bounds on one tensor knot sub-box."""

    source_id: str
    span_indices: tuple[int, int]
    span_parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    value_bounds: tuple[tuple[Fraction, Fraction], ...]
    du_bounds: tuple[tuple[Fraction, Fraction], ...]
    dv_bounds: tuple[tuple[Fraction, Fraction], ...]
    denominator_bounds: tuple[Fraction, Fraction]

    @property
    def positive_denominator(self) -> bool:
        return self.denominator_bounds[0] > 0


@dataclass(frozen=True, slots=True)
class RationalBSplineSurfaceDerivativeEnclosure:
    """Exact first-derivative bounds for one rational Bernstein surface sub-box."""

    source_id: str
    span_indices: tuple[int, int]
    span_parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    du_bounds: tuple[tuple[Fraction, Fraction], ...]
    dv_bounds: tuple[tuple[Fraction, Fraction], ...]
    denominator_bounds: tuple[Fraction, Fraction]

    @property
    def positive_denominator(self) -> bool:
        return self.denominator_bounds[0] > 0


@dataclass(frozen=True, slots=True)
class RationalBSplineSurfaceHessianEnclosure:
    """Exact first- and second-derivative bounds on one rational surface sub-box."""

    source_id: str
    span_indices: tuple[int, int]
    span_parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    du_bounds: tuple[tuple[Fraction, Fraction], ...]
    dv_bounds: tuple[tuple[Fraction, Fraction], ...]
    duu_bounds: tuple[tuple[Fraction, Fraction], ...]
    duv_bounds: tuple[tuple[Fraction, Fraction], ...]
    dvv_bounds: tuple[tuple[Fraction, Fraction], ...]
    denominator_bounds: tuple[Fraction, Fraction]

    @property
    def positive_denominator(self) -> bool:
        return self.denominator_bounds[0] > 0


@dataclass(frozen=True, slots=True)
class _RationalSurfaceDerivativeData:
    parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    relative_bounds: tuple[ExactScalar, ExactScalar, ExactScalar, ExactScalar]
    du_bounds: tuple[tuple[Fraction, Fraction], ...]
    dv_bounds: tuple[tuple[Fraction, Fraction], ...]
    duu_bounds: tuple[tuple[Fraction, Fraction], ...] | None
    duv_bounds: tuple[tuple[Fraction, Fraction], ...] | None
    dvv_bounds: tuple[tuple[Fraction, Fraction], ...] | None
    denominator_bounds: tuple[Fraction, Fraction]

    @property
    def positive_denominator(self) -> bool:
        return self.denominator_bounds[0] > 0


@dataclass(frozen=True, slots=True)
class RationalBSplineSurfaceSpan:
    """Exact local homogeneous Bernstein representation of one surface span."""

    source_id: str
    span_indices: tuple[int, int]
    degrees: tuple[int, int]
    coefficient_rows: tuple[int, ...]
    parameter_bounds: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    homogeneous_bernstein_coefficients: tuple[tuple[tuple[ExactScalar, ...], ...], ...]
    positive_denominator_lower_bound: Fraction
    _du_numerators: tuple[tuple[tuple[ExactScalar, ...], ...], ...] = field(repr=False)
    _dv_numerators: tuple[tuple[tuple[ExactScalar, ...], ...], ...] = field(repr=False)
    _duu_numerators: tuple[tuple[tuple[ExactScalar, ...], ...], ...] | None = field(
        repr=False
    )
    _duv_numerators: tuple[tuple[tuple[ExactScalar, ...], ...], ...] | None = field(
        repr=False
    )
    _dvv_numerators: tuple[tuple[tuple[ExactScalar, ...], ...], ...] | None = field(
        repr=False
    )
    # Outward binary64 enclosures of the exact nets, prepared once per span.
    _interval_nets: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def has_hessian(self) -> bool:
        return (
            self._duu_numerators is not None
            and self._duv_numerators is not None
            and self._dvv_numerators is not None
        )

    def _exact_net(self, name: str, component: int, /) -> Sequence[Sequence[ExactScalar]]:
        homogeneous = self.homogeneous_bernstein_coefficients
        if name == "denominator":
            return tuple(tuple(point[-1] for point in row) for row in homogeneous)
        if name == "value":
            return tuple(tuple(point[component] for point in row) for row in homogeneous)
        nets = {
            "du": self._du_numerators,
            "dv": self._dv_numerators,
            "duu": self._duu_numerators,
            "duv": self._duv_numerators,
            "dvv": self._dvv_numerators,
        }[name]
        if nets is None:
            raise RuntimeError("Rational Bernstein Hessian numerators are absent.")
        return nets[component]

    def _bounds(
        self,
        name: str,
        component: int,
        relative_bounds: tuple[ExactScalar, ExactScalar, ExactScalar, ExactScalar],
        /,
    ) -> tuple[Fraction, Fraction]:
        """Outward-rounded bounds of one prepared net on a relative sub-box."""
        key = (name, component)
        net = self._interval_nets.get(key)
        if net is None:
            net = _interval_net(self._exact_net(name, component))
            self._interval_nets[key] = net
        return _interval_tensor_bounds(net, relative_bounds)

    def _derivative_data(
        self,
        parameter_bounds: Sequence[Sequence[ExactScalar | int | float]],
        /,
        *,
        include_hessian: bool,
    ) -> _RationalSurfaceDerivativeData:
        if len(parameter_bounds) != 2 or any(
            len(axis_bounds) != 2 for axis_bounds in parameter_bounds
        ):
            raise ValueError("A surface span enclosure requires two parameter intervals.")
        u_bounds, v_bounds = parameter_bounds
        lower_u = _exact_scalar(u_bounds[0], "parameter endpoint")
        upper_u = _exact_scalar(u_bounds[1], "parameter endpoint")
        lower_v = _exact_scalar(v_bounds[0], "parameter endpoint")
        upper_v = _exact_scalar(v_bounds[1], "parameter endpoint")
        bounds = ((lower_u, upper_u), (lower_v, upper_v))
        (span_lower_u, span_upper_u), (span_lower_v, span_upper_v) = self.parameter_bounds
        if not (
            span_lower_u <= lower_u <= upper_u <= span_upper_u
            and span_lower_v <= lower_v <= upper_v <= span_upper_v
        ):
            raise ValueError(
                "Enclosure intervals must lie inside one exact nonempty surface span."
            )
        relative_lower_u = (lower_u - span_lower_u) / (span_upper_u - span_lower_u)
        relative_upper_u = (upper_u - span_lower_u) / (span_upper_u - span_lower_u)
        relative_lower_v = (lower_v - span_lower_v) / (span_upper_v - span_lower_v)
        relative_upper_v = (upper_v - span_lower_v) / (span_upper_v - span_lower_v)
        relative_bounds = (
            relative_lower_u,
            relative_upper_u,
            relative_lower_v,
            relative_upper_v,
        )
        denominator_interval = self._bounds("denominator", 0, relative_bounds)
        # Every restricted weight is a convex combination of the span weights.
        denominator_bounds = (
            max(denominator_interval[0], self.positive_denominator_lower_bound),
            denominator_interval[1],
        )
        denominator_squared_bounds = (
            denominator_bounds[0] ** 2,
            denominator_bounds[1] ** 2,
        )
        du_bounds = tuple(
            _positive_quotient_bounds(
                self._bounds("du", component, relative_bounds),
                denominator_squared_bounds,
            )
            for component in range(len(self._du_numerators))
        )
        dv_bounds = tuple(
            _positive_quotient_bounds(
                self._bounds("dv", component, relative_bounds),
                denominator_squared_bounds,
            )
            for component in range(len(self._dv_numerators))
        )
        duu_bounds = None
        duv_bounds = None
        dvv_bounds = None
        if include_hessian:
            if not self.has_hessian:
                raise ValueError(
                    "This rational Bernstein span does not retain Hessian numerators."
                )
            denominator_cubed_bounds = (
                denominator_bounds[0] ** 3,
                denominator_bounds[1] ** 3,
            )
            duu_bounds, duv_bounds, dvv_bounds = (
                tuple(
                    _positive_quotient_bounds(
                        self._bounds(name, component, relative_bounds),
                        denominator_cubed_bounds,
                    )
                    for component in range(len(self._du_numerators))
                )
                for name in ("duu", "duv", "dvv")
            )
        return _RationalSurfaceDerivativeData(
            bounds,
            relative_bounds,
            du_bounds,
            dv_bounds,
            duu_bounds,
            duv_bounds,
            dvv_bounds,
            denominator_bounds,
        )

    def enclose_derivatives(
        self,
        parameter_bounds: Sequence[Sequence[ExactScalar | int | float]],
        /,
    ) -> RationalBSplineSurfaceDerivativeEnclosure:
        data = self._derivative_data(parameter_bounds, include_hessian=False)
        return RationalBSplineSurfaceDerivativeEnclosure(
            self.source_id,
            self.span_indices,
            self.parameter_bounds,
            data.parameter_bounds,
            data.du_bounds,
            data.dv_bounds,
            data.denominator_bounds,
        )

    def enclose_hessian(
        self,
        parameter_bounds: Sequence[Sequence[ExactScalar | int | float]],
        /,
    ) -> RationalBSplineSurfaceHessianEnclosure:
        data = self._derivative_data(parameter_bounds, include_hessian=True)
        if data.duu_bounds is None or data.duv_bounds is None or data.dvv_bounds is None:
            raise RuntimeError(
                "Exact Hessian bounds require prepared Hessian numerators."
            )
        return RationalBSplineSurfaceHessianEnclosure(
            self.source_id,
            self.span_indices,
            self.parameter_bounds,
            data.parameter_bounds,
            data.du_bounds,
            data.dv_bounds,
            data.duu_bounds,
            data.duv_bounds,
            data.dvv_bounds,
            data.denominator_bounds,
        )

    def enclose(
        self,
        parameter_bounds: Sequence[Sequence[ExactScalar | int | float]],
        /,
    ) -> RationalBSplineSurfaceSpanEnclosure:
        data = self._derivative_data(parameter_bounds, include_hessian=False)
        component_count = len(self.homogeneous_bernstein_coefficients[0][0]) - 1
        value_bounds = tuple(
            _positive_quotient_bounds(
                self._bounds("value", component, data.relative_bounds),
                data.denominator_bounds,
            )
            for component in range(component_count)
        )
        return RationalBSplineSurfaceSpanEnclosure(
            self.source_id,
            self.span_indices,
            self.parameter_bounds,
            data.parameter_bounds,
            value_bounds,
            data.du_bounds,
            data.dv_bounds,
            data.denominator_bounds,
        )


def _rational_bernstein_surface_span(
    source_id: str,
    span_indices: tuple[int, int],
    degrees: tuple[int, int],
    coefficient_rows: Sequence[int],
    parameter_bounds: Sequence[Sequence[ExactScalar | int | float]],
    homogeneous_bernstein_coefficients: Sequence[
        Sequence[Sequence[ExactScalar | int | float]]
    ],
    /,
    *,
    include_hessian: bool = False,
) -> RationalBSplineSurfaceSpan:
    if not isinstance(include_hessian, bool):
        raise TypeError("include_hessian must be bool.")
    if not source_id or len(span_indices) != 2 or len(degrees) != 2:
        raise ValueError("A rational Bernstein surface span requires exact identities.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in (*span_indices, *degrees)
    ):
        raise ValueError(
            "Rational Bernstein surface span indices and degrees are invalid."
        )
    degree_u, degree_v = int(degrees[0]), int(degrees[1])
    bounds_values = tuple(
        tuple(_exact_scalar(value, "surface span parameter bound") for value in axis)
        for axis in parameter_bounds
    )
    if len(bounds_values) != 2 or any(len(axis) != 2 for axis in bounds_values):
        raise ValueError(
            "A rational Bernstein surface span requires two parameter intervals."
        )
    parameter_bounds_ = (
        (bounds_values[0][0], bounds_values[0][1]),
        (bounds_values[1][0], bounds_values[1][1]),
    )
    if any(lower >= upper for lower, upper in parameter_bounds_):
        raise ValueError(
            "Rational Bernstein surface spans require positive parameter extents."
        )
    rows = tuple(int(value) for value in coefficient_rows)
    if (
        not rows
        or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in coefficient_rows
        )
        or len(set(rows)) != len(rows)
    ):
        raise ValueError(
            "Rational Bernstein coefficient rows must be unique nonnegative IDs."
        )
    homogeneous = tuple(
        tuple(
            tuple(
                _exact_scalar(value, "homogeneous Bernstein coefficient")
                for value in point
            )
            for point in row
        )
        for row in homogeneous_bernstein_coefficients
    )
    if (
        len(homogeneous) != degree_u + 1
        or any(len(row) != degree_v + 1 for row in homogeneous)
        or not homogeneous[0]
        or not homogeneous[0][0]
        or any(
            len(point) != len(homogeneous[0][0]) for row in homogeneous for point in row
        )
        or len(homogeneous[0][0]) < 2
    ):
        raise ValueError(
            "Homogeneous Bernstein surface controls must form a nonempty tensor grid."
        )
    denominator = tuple(tuple(point[-1] for point in row) for row in homogeneous)
    denominator_minimum = min(value for row in denominator for value in row)
    if denominator_minimum <= 0:
        raise ValueError(
            "Exact rational Bernstein surface extraction did not certify a positive denominator."
        )
    denominator_lower = _positive_rational_lower_bound(denominator_minimum)
    denominator_u = _tensor_bernstein_derivative(denominator, 0, parameter_bounds_)
    denominator_v = _tensor_bernstein_derivative(denominator, 1, parameter_bounds_)
    du_numerators: list[tuple[tuple[ExactScalar, ...], ...]] = []
    dv_numerators: list[tuple[tuple[ExactScalar, ...], ...]] = []
    duu_numerators: list[tuple[tuple[ExactScalar, ...], ...]] = []
    duv_numerators: list[tuple[tuple[ExactScalar, ...], ...]] = []
    dvv_numerators: list[tuple[tuple[ExactScalar, ...], ...]] = []
    for component in range(len(homogeneous[0][0]) - 1):
        numerator = tuple(tuple(point[component] for point in row) for row in homogeneous)
        numerator_u = _tensor_bernstein_derivative(numerator, 0, parameter_bounds_)
        numerator_v = _tensor_bernstein_derivative(numerator, 1, parameter_bounds_)
        derivative_u = _tensor_bernstein_linear_combination(
            (
                (
                    Fraction(1),
                    _tensor_bernstein_product(numerator_u, denominator),
                ),
                (
                    Fraction(-1),
                    _tensor_bernstein_product(numerator, denominator_u),
                ),
            )
        )
        derivative_v = _tensor_bernstein_linear_combination(
            (
                (
                    Fraction(1),
                    _tensor_bernstein_product(numerator_v, denominator),
                ),
                (
                    Fraction(-1),
                    _tensor_bernstein_product(numerator, denominator_v),
                ),
            )
        )
        du_numerators.append(derivative_u)
        dv_numerators.append(derivative_v)
        if include_hessian:
            derivative_uu = _tensor_bernstein_derivative(
                derivative_u, 0, parameter_bounds_
            )
            derivative_uv = _tensor_bernstein_derivative(
                derivative_u, 1, parameter_bounds_
            )
            derivative_vv = _tensor_bernstein_derivative(
                derivative_v, 1, parameter_bounds_
            )
            duu_numerators.append(
                _tensor_bernstein_linear_combination(
                    (
                        (
                            Fraction(1),
                            _tensor_bernstein_product(derivative_uu, denominator),
                        ),
                        (
                            Fraction(-2),
                            _tensor_bernstein_product(derivative_u, denominator_u),
                        ),
                    )
                )
            )
            duv_numerators.append(
                _tensor_bernstein_linear_combination(
                    (
                        (
                            Fraction(1),
                            _tensor_bernstein_product(derivative_uv, denominator),
                        ),
                        (
                            Fraction(-2),
                            _tensor_bernstein_product(derivative_u, denominator_v),
                        ),
                    )
                )
            )
            dvv_numerators.append(
                _tensor_bernstein_linear_combination(
                    (
                        (
                            Fraction(1),
                            _tensor_bernstein_product(derivative_vv, denominator),
                        ),
                        (
                            Fraction(-2),
                            _tensor_bernstein_product(derivative_v, denominator_v),
                        ),
                    )
                )
            )
    return RationalBSplineSurfaceSpan(
        str(source_id),
        (int(span_indices[0]), int(span_indices[1])),
        (degree_u, degree_v),
        rows,
        parameter_bounds_,
        homogeneous,
        denominator_lower,
        tuple(du_numerators),
        tuple(dv_numerators),
        tuple(duu_numerators) if include_hessian else None,
        tuple(duv_numerators) if include_hessian else None,
        tuple(dvv_numerators) if include_hessian else None,
    )


def _matches_binary64_shadow(numeric: object, exact: ExactScalar, /) -> bool:
    if isinstance(numeric, (bool, np.bool_)):
        return False
    if not isinstance(numeric, (int, float, np.integer, np.floating)):
        return False
    try:
        numeric_value = float(numeric)
        exact_value = float(exact)
    except (OverflowError, TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric_value) and numeric_value == exact_value)


def rational_bspline_surface_spans(
    control_points: Sequence[Sequence[Sequence[ExactScalar | int | float]]],
    weights: Sequence[Sequence[ExactScalar | int | float]],
    u_knots: Sequence[ExactScalar | int | float],
    v_knots: Sequence[ExactScalar | int | float],
    u_degree: int,
    v_degree: int,
    /,
    *,
    exact_control_points: Sequence[Sequence[Sequence[ExactScalar | int | float]]]
    | None = None,
    exact_weights: Sequence[Sequence[ExactScalar | int | float]] | None = None,
    exact_u_knots: Sequence[ExactScalar | int | float] | None = None,
    exact_v_knots: Sequence[ExactScalar | int | float] | None = None,
    include_hessian: bool = False,
) -> tuple[RationalBSplineSurfaceSpan, ...]:
    """Build exact rational surface spans from authored tensor control banks.

    Optional exact source banks must have the supplied binary64 realization.
    Every span retains homogeneous Bernstein coefficients and exact position and
    Jacobian enclosures. When requested, exact rational Hessian numerators are
    also retained for sub-box bounds.
    """
    if not isinstance(include_hessian, bool):
        raise TypeError("include_hessian must be bool.")
    degrees = (u_degree, v_degree)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in degrees
    ):
        raise TypeError("Tensor B-spline surface degrees must be integers.")
    degree_u_, degree_v_ = int(u_degree), int(v_degree)
    if degree_u_ < 0 or degree_v_ < 0:
        raise ValueError("Tensor B-spline surface degrees must be nonnegative.")
    raw_points = tuple(tuple(tuple(point) for point in row) for row in control_points)
    point_source = (
        raw_points
        if exact_control_points is None
        else tuple(tuple(tuple(point) for point in row) for row in exact_control_points)
    )
    points = tuple(
        tuple(
            tuple(_exact_scalar(value, "control point") for value in point)
            for point in row
        )
        for row in point_source
    )
    raw_weights = tuple(tuple(row) for row in weights)
    weight_source = (
        raw_weights
        if exact_weights is None
        else tuple(tuple(row) for row in exact_weights)
    )
    weights_ = tuple(
        tuple(_exact_scalar(value, "weight") for value in row) for row in weight_source
    )
    raw_u_knots = tuple(u_knots)
    raw_v_knots = tuple(v_knots)
    u_knot_source = raw_u_knots if exact_u_knots is None else tuple(exact_u_knots)
    v_knot_source = raw_v_knots if exact_v_knots is None else tuple(exact_v_knots)
    u_knots_ = tuple(_exact_scalar(value, "u knot") for value in u_knot_source)
    v_knots_ = tuple(_exact_scalar(value, "v knot") for value in v_knot_source)
    if not raw_points or not raw_points[0] or not raw_points[0][0]:
        raise ValueError("Rational surface control banks must be nonempty.")
    control_count_u, control_count_v = len(raw_points), len(raw_points[0])
    ambient_dimension = len(raw_points[0][0])
    if (
        ambient_dimension == 0
        or any(
            len(row) != control_count_v
            or any(len(point) != ambient_dimension for point in row)
            for row in raw_points
        )
        or len(points) != control_count_u
        or any(
            len(row) != control_count_v
            or any(len(point) != ambient_dimension for point in row)
            for row in points
        )
        or len(raw_weights) != control_count_u
        or any(len(row) != control_count_v for row in raw_weights)
        or len(weights_) != control_count_u
        or any(len(row) != control_count_v for row in weights_)
        or len(raw_u_knots) != len(u_knots_)
        or len(raw_v_knots) != len(v_knots_)
    ):
        raise ValueError(
            "Rational surface exact source banks must align with numeric leaves."
        )
    if exact_control_points is not None and any(
        not _matches_binary64_shadow(numeric, exact)
        for numeric_row, exact_row in zip(raw_points, points, strict=True)
        for numeric_point, exact_point in zip(numeric_row, exact_row, strict=True)
        for numeric, exact in zip(numeric_point, exact_point, strict=True)
    ):
        raise ValueError("Exact B-spline controls must match their binary64 realization.")
    if exact_weights is not None and any(
        not _matches_binary64_shadow(numeric, exact)
        for numeric_row, exact_row in zip(raw_weights, weights_, strict=True)
        for numeric, exact in zip(numeric_row, exact_row, strict=True)
    ):
        raise ValueError("Exact B-spline weights must match their binary64 realization.")
    if exact_u_knots is not None and any(
        not _matches_binary64_shadow(numeric, exact)
        for numeric, exact in zip(raw_u_knots, u_knots_, strict=True)
    ):
        raise ValueError("Exact B-spline u knots must match their binary64 realization.")
    if exact_v_knots is not None and any(
        not _matches_binary64_shadow(numeric, exact)
        for numeric, exact in zip(raw_v_knots, v_knots_, strict=True)
    ):
        raise ValueError("Exact B-spline v knots must match their binary64 realization.")
    if (
        control_count_u < degree_u_ + 1
        or control_count_v < degree_v_ + 1
        or len(u_knots_) != control_count_u + degree_u_ + 1
        or len(v_knots_) != control_count_v + degree_v_ + 1
    ):
        raise ValueError("Rational surface control, weight, and knot banks must align.")
    if any(float(weight) <= 0.0 for row in raw_weights for weight in row):
        raise ValueError("Rational surface numeric weights must be strictly positive.")
    if any(weight <= 0 for row in weights_ for weight in row):
        raise ValueError("Rational surface weights must be strictly positive.")
    if any(right < left for left, right in zip(u_knots_[:-1], u_knots_[1:], strict=True)):
        raise ValueError("Rational surface u knots must be nondecreasing.")
    if any(right < left for left, right in zip(v_knots_[:-1], v_knots_[1:], strict=True)):
        raise ValueError("Rational surface v knots must be nondecreasing.")
    if any(count > degree_u_ + 1 for count in Counter(u_knots_).values()):
        raise ValueError("Rational surface u knot multiplicity exceeds degree + 1.")
    if any(count > degree_v_ + 1 for count in Counter(v_knots_).values()):
        raise ValueError("Rational surface v knot multiplicity exceeds degree + 1.")
    active_u = (u_knots_[degree_u_], u_knots_[control_count_u])
    active_v = (v_knots_[degree_v_], v_knots_[control_count_v])
    if not active_u[0] < active_u[1] or not active_v[0] < active_v[1]:
        raise ValueError("Rational surface knots must define positive active intervals.")
    source_id = canonical_fingerprint(
        {
            "kind": "exact-rational-bspline-surface-source",
            "degrees": (degree_u_, degree_v_),
            "u_knots": [_exact_scalar_payload(value) for value in u_knots_],
            "v_knots": [_exact_scalar_payload(value) for value in v_knots_],
            "control_points": [
                [[_exact_scalar_payload(value) for value in point] for point in row]
                for row in points
            ],
            "weights": [
                [_exact_scalar_payload(value) for value in row] for row in weights_
            ],
        }
    )
    homogeneous = tuple(
        tuple(
            tuple(weight * value for value in point) + (weight,)
            for point, weight in zip(point_row, weight_row, strict=True)
        )
        for point_row, weight_row in zip(points, weights_, strict=True)
    )
    flat_homogeneous = tuple(value for row in homogeneous for value in row)
    control_ids = tuple(range(control_count_u * control_count_v))
    spans: list[RationalBSplineSurfaceSpan] = []
    for u_span in range(degree_u_, control_count_u):
        lower_u, upper_u = u_knots_[u_span], u_knots_[u_span + 1]
        if lower_u >= upper_u:
            continue
        for v_span in range(degree_v_, control_count_v):
            lower_v, upper_v = v_knots_[v_span], v_knots_[v_span + 1]
            if lower_v >= upper_v:
                continue
            coefficient_rows, extraction = tensor_bspline_span_extraction(
                u_knots_,
                v_knots_,
                degree_u_,
                degree_v_,
                u_span,
                v_span,
                control_ids,
            )
            local_homogeneous = tuple(flat_homogeneous[row] for row in coefficient_rows)
            bernstein = tuple(
                tuple(
                    tuple(
                        sum(
                            (
                                extraction[local_row][u_index * (degree_v_ + 1) + v_index]
                                * local_homogeneous[local_row][component]
                                for local_row in range(len(coefficient_rows))
                            ),
                            Fraction(0),
                        )
                        for component in range(ambient_dimension + 1)
                    )
                    for v_index in range(degree_v_ + 1)
                )
                for u_index in range(degree_u_ + 1)
            )
            span_bounds = ((lower_u, upper_u), (lower_v, upper_v))
            spans.append(
                _rational_bernstein_surface_span(
                    source_id,
                    (u_span, v_span),
                    (degree_u_, degree_v_),
                    coefficient_rows,
                    span_bounds,
                    bernstein,
                    include_hessian=include_hessian,
                )
            )
    if not spans:
        raise ValueError("Rational surface source contains no nonempty knot spans.")
    return tuple(spans)
