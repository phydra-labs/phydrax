#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact host polynomial expressions and outward Bernstein enclosures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import cache
from itertools import product
from math import comb, prod
from types import MappingProxyType
from typing import cast, Literal, NamedTuple, overload, Protocol, TYPE_CHECKING

import numpy as np

from .._fingerprint import array_tree_fingerprint
from ._exact_algebraic_geometry import (
    coefficient_endpoint,
    coefficient_rounding_error,
    RadicalElement,
)


type ExactScalar = Fraction | RadicalElement
type Polynomial = dict[tuple[int, ...], ExactScalar]
type Expression = Polynomial | RationalPolynomial
type ExpressionInput = Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial
type CoordinateSourceBank = tuple[tuple[ExactScalar, ...], ...]
type CoordinateCoefficients = np.ndarray | CoordinateSourceBank
type BernsteinDomain = Literal["simplex", "prism", "box"]
type AffineMap = tuple[tuple[ExactScalar, ...], tuple[tuple[ExactScalar, ...], ...]]

if TYPE_CHECKING:
    from ._cell_geometry import ScalarCoordinateElement


class _CellGeometrySourceOwner(Protocol):
    def coordinate_source_basis(
        self,
        element: ScalarCoordinateElement,
        /,
        *,
        cell_row: int | None = None,
    ) -> tuple[Expression, ...]: ...

    def coordinate_effective_basis(
        self,
        element: ScalarCoordinateElement,
        /,
        *,
        cell_row: int | None = None,
    ) -> tuple[Expression, ...]: ...

    def coordinate_execution_basis(
        self,
        element: ScalarCoordinateElement,
        /,
        *,
        cell_row: int | None = None,
    ) -> tuple[Expression, ...]: ...

    def coordinate_reference_chain(
        self,
        element: ScalarCoordinateElement,
        /,
        *,
        ancestor: ScalarCoordinateElement | None = None,
    ) -> tuple[ScalarCoordinateElement, ...]: ...

    def coordinate_reference_arguments(
        self,
        element: ScalarCoordinateElement,
        /,
        *,
        cell_row: int | None = None,
    ) -> tuple[Expression, ...] | None: ...


@cache
def _cell_geometry_dispatch() -> _CellGeometrySourceOwner:
    from . import _cell_geometry

    return cast(_CellGeometrySourceOwner, _cell_geometry)


def _scalar(
    value: ExactScalar | int | float | np.integer | np.floating, /
) -> ExactScalar:
    if isinstance(value, RadicalElement):
        return value
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("coordinate coefficient must be finite.")
        return Fraction(number)
    raise TypeError(
        f"coordinate coefficient has unsupported type {type(value).__name__}."
    )


def _canonical_polynomial(
    value: Mapping[tuple[int, ...], ExactScalar | int | float], /
) -> Polynomial:
    result: Polynomial = {}
    dimension: int | None = None
    for raw_index, raw_coefficient in value.items():
        index = tuple(raw_index)
        if any(
            isinstance(power, bool) or not isinstance(power, int) or power < 0
            for power in index
        ):
            raise ValueError("polynomial exponents must be nonnegative integers.")
        if dimension is None:
            dimension = len(index)
        elif len(index) != dimension:
            raise ValueError("polynomial monomials must have one common dimension.")
        coefficient = _scalar(raw_coefficient)
        if coefficient:
            result[index] = result.get(index, Fraction(0)) + coefficient
    return {index: coefficient for index, coefficient in result.items() if coefficient}


class RationalPolynomial:
    """Exact rational quotient of two multivariate polynomial expressions."""

    __slots__ = ("numerator", "denominator")

    numerator: Mapping[tuple[int, ...], ExactScalar]
    denominator: Mapping[tuple[int, ...], ExactScalar]

    def __init__(
        self,
        numerator: Mapping[tuple[int, ...], ExactScalar | int | float],
        denominator: Mapping[tuple[int, ...], ExactScalar | int | float],
        /,
    ) -> None:
        numerator_ = _canonical_polynomial(numerator)
        denominator_ = _canonical_polynomial(denominator)
        if not denominator_:
            raise ValueError("RationalPolynomial.denominator must be nonzero.")
        object.__setattr__(self, "numerator", MappingProxyType(numerator_))
        object.__setattr__(self, "denominator", MappingProxyType(denominator_))

    def __bool__(self) -> bool:
        return bool(self.numerator)

    def __eq__(self, other: object, /) -> bool:
        raise TypeError("RationalPolynomial equality must use expression_equal().")

    __hash__ = None


def affine_map(value: tuple[Expression, ...], dimension: int, /) -> AffineMap | None:
    """Extract an affine map; rational denominators must be nonzero on the unit box."""
    offsets: list[ExactScalar] = []
    rows: list[tuple[ExactScalar, ...]] = []
    for expression in value:
        polynomial: Polynomial
        if isinstance(expression, RationalPolynomial):
            denominator = expression.denominator.get((0,) * dimension)
            if len(expression.denominator) == 1 and denominator:
                polynomial = {
                    exponent: coefficient / denominator
                    for exponent, coefficient in expression.numerator.items()
                }
            else:
                controls = bernstein_coefficients(
                    expression.denominator, "box", dimension
                )
                if not (
                    all(coefficient > 0 for coefficient in controls)
                    or all(coefficient < 0 for coefficient in controls)
                ):
                    return None
                origin = tuple(Fraction(0) for _ in range(dimension))
                polynomial = constant(evaluate_expression(expression, origin), dimension)
                for axis, variable in enumerate(axes(dimension)):
                    slope = evaluate_expression(derivative(expression, axis), origin)
                    polynomial = add(polynomial, scale(variable, slope))
                if multiply(polynomial, expression.denominator) != expression.numerator:
                    return None
        else:
            polynomial = expression
        if any(
            len(exponent) != dimension or sum(exponent) > 1 for exponent in polynomial
        ):
            return None
        offsets.append(polynomial.get((0,) * dimension, Fraction(0)))
        rows.append(
            tuple(
                polynomial.get(
                    tuple(int(axis == column) for axis in range(dimension)),
                    Fraction(0),
                )
                for column in range(dimension)
            )
        )
    return tuple(offsets), tuple(rows)


def constant(value: ExactScalar | int | float, dimension: int, /) -> Polynomial:
    if dimension < 0:
        raise ValueError("constant.dimension must be nonnegative.")
    coefficient = _scalar(value)
    return {(0,) * dimension: coefficient} if coefficient else {}


def axes(dimension: int, /) -> tuple[Polynomial, ...]:
    if dimension < 1:
        raise ValueError("axes.dimension must be positive.")
    return tuple(
        {tuple(int(axis == index) for axis in range(dimension)): Fraction(1)}
        for index in range(dimension)
    )


@overload
def add(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    /,
) -> Polynomial: ...


@overload
def add(first: ExpressionInput, second: ExpressionInput, /) -> Expression: ...


def add(first: ExpressionInput, second: ExpressionInput, /) -> Expression:
    if isinstance(first, RationalPolynomial) or isinstance(second, RationalPolynomial):
        dimension = max(_expression_dimension(first), _expression_dimension(second))
        first_num, first_den = (
            (first.numerator, first.denominator)
            if isinstance(first, RationalPolynomial)
            else (first, constant(1, dimension))
        )
        second_num, second_den = (
            (second.numerator, second.denominator)
            if isinstance(second, RationalPolynomial)
            else (second, constant(1, dimension))
        )
        if first_den == second_den:
            # Keeping the common denominator preserves its pole set as well as
            # the value, without multiplying repeated denominator factors.
            return RationalPolynomial(add(first_num, second_num), first_den)
        return RationalPolynomial(
            add(multiply(first_num, second_den), multiply(second_num, first_den)),
            multiply(first_den, second_den),
        )
    result = dict(first)
    for index, coefficient in second.items():
        updated = result.get(index, Fraction(0)) + coefficient
        if updated:
            result[index] = updated
        else:
            result.pop(index, None)
    return result


@overload
def subtract(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    /,
) -> Polynomial: ...


@overload
def subtract(first: ExpressionInput, second: ExpressionInput, /) -> Expression: ...


def subtract(first: ExpressionInput, second: ExpressionInput, /) -> Expression:
    return add(first, scale(second, Fraction(-1)))


@overload
def scale(
    value: Mapping[tuple[int, ...], ExactScalar],
    coefficient: ExactScalar | int | float,
    /,
) -> Polynomial: ...


@overload
def scale(
    value: ExpressionInput, coefficient: ExactScalar | int | float, /
) -> Expression: ...


def scale(
    value: ExpressionInput, coefficient: ExactScalar | int | float, /
) -> Expression:
    scalar = _scalar(coefficient)
    if not scalar:
        if isinstance(value, RationalPolynomial):
            return RationalPolynomial({}, value.denominator)
        return {}
    if isinstance(value, RationalPolynomial):
        return RationalPolynomial(
            {index: scalar * term for index, term in value.numerator.items()},
            value.denominator,
        )
    result: Polynomial = {}
    for index, term in value.items():
        product_ = scalar * term
        if product_:
            result[index] = product_
    return result


@overload
def multiply(
    first: Mapping[tuple[int, ...], ExactScalar],
    second: Mapping[tuple[int, ...], ExactScalar],
    /,
) -> Polynomial: ...


@overload
def multiply(first: ExpressionInput, second: ExpressionInput, /) -> Expression: ...


def multiply(first: ExpressionInput, second: ExpressionInput, /) -> Expression:
    if isinstance(first, RationalPolynomial) or isinstance(second, RationalPolynomial):
        dimension = max(_expression_dimension(first), _expression_dimension(second))
        first_num, first_den = (
            (first.numerator, first.denominator)
            if isinstance(first, RationalPolynomial)
            else (first, constant(1, dimension))
        )
        second_num, second_den = (
            (second.numerator, second.denominator)
            if isinstance(second, RationalPolynomial)
            else (second, constant(1, dimension))
        )
        return RationalPolynomial(
            multiply(first_num, second_num), multiply(first_den, second_den)
        )
    if not first or not second:
        return {}
    first_dimension = _polynomial_dimension(first)
    if _polynomial_dimension(second) != first_dimension:
        raise ValueError("polynomial multiplication requires equal parameter dimensions.")
    result: Polynomial = {}
    for left_index, left_value in first.items():
        for right_index, right_value in second.items():
            index = tuple(a + b for a, b in zip(left_index, right_index, strict=True))
            result[index] = result.get(index, Fraction(0)) + left_value * right_value
    return {index: coefficient for index, coefficient in result.items() if coefficient}


def power(
    value: ExpressionInput,
    exponent: int,
    /,
    *,
    dimension: int | None = None,
) -> Expression:
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise TypeError("polynomial exponent must be an integer.")
    inferred = _expression_dimension(value)
    dimension_ = inferred if dimension is None else dimension
    if dimension_ < 0 or (value and inferred != dimension_):
        raise ValueError("power.dimension must match the polynomial variable count.")
    if exponent < 0:
        if isinstance(value, RationalPolynomial):
            positive = power(value, -exponent, dimension=dimension_)
            if not positive:
                raise ZeroDivisionError(
                    "zero rational polynomial cannot have a negative exponent."
                )
            if not isinstance(positive, RationalPolynomial):
                return RationalPolynomial({(0,) * dimension_: Fraction(1)}, positive)
            return RationalPolynomial(positive.denominator, positive.numerator)
        if not value:
            raise ZeroDivisionError("zero polynomial cannot have a negative exponent.")
        positive = power(value, -exponent, dimension=dimension_)
        if isinstance(positive, RationalPolynomial):
            raise RuntimeError("A nonnegative polynomial power must remain polynomial.")
        return RationalPolynomial({(0,) * dimension_: Fraction(1)}, positive)
    result: Expression = constant(1, dimension_)
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply(result, factor)
        remaining >>= 1
        if remaining:
            factor = multiply(factor, factor)
    return result


@overload
def derivative(
    value: Mapping[tuple[int, ...], ExactScalar], axis: int, /
) -> Polynomial: ...


@overload
def derivative(value: ExpressionInput, axis: int, /) -> Expression: ...


def derivative(value: ExpressionInput, axis: int, /) -> Expression:
    if isinstance(axis, bool) or not isinstance(axis, int) or axis < 0:
        raise ValueError("derivative.axis must be a nonnegative integer.")
    if isinstance(value, RationalPolynomial):
        numerator = subtract(
            multiply(derivative(dict(value.numerator), axis), dict(value.denominator)),
            multiply(dict(value.numerator), derivative(dict(value.denominator), axis)),
        )
        denominator = power(dict(value.denominator), 2)
        if isinstance(numerator, RationalPolynomial) or isinstance(
            denominator, RationalPolynomial
        ):
            raise RuntimeError(
                "rational derivative composition did not reduce to polynomial support."
            )
        return RationalPolynomial(numerator, denominator)
    result: Polynomial = {}
    for index, coefficient in value.items():
        if axis >= len(index):
            raise ValueError("derivative.axis exceeds the polynomial dimension.")
        exponent = index[axis]
        if exponent:
            selected = list(index)
            selected[axis] -= 1
            result[tuple(selected)] = coefficient * exponent
    return result


@overload
def sum_polynomials(
    values: Sequence[Mapping[tuple[int, ...], ExactScalar]], dimension: int, /
) -> Polynomial: ...


@overload
def sum_polynomials(
    values: Sequence[ExpressionInput], dimension: int, /
) -> Expression: ...


def sum_polynomials(values: Sequence[ExpressionInput], dimension: int, /) -> Expression:
    result: Expression = constant(0, dimension)
    for value in values:
        result = add(result, value)
    return result


@overload
def linear_combinations(
    basis: Sequence[Mapping[tuple[int, ...], ExactScalar]],
    coefficients: Sequence[Sequence[ExactScalar]],
    /,
) -> tuple[Polynomial, ...]: ...


@overload
def linear_combinations(
    basis: Sequence[ExpressionInput],
    coefficients: Sequence[Sequence[ExactScalar]],
    /,
) -> tuple[Expression, ...]: ...


def linear_combinations(
    basis: Sequence[ExpressionInput],
    coefficients: Sequence[Sequence[ExactScalar]],
    /,
) -> tuple[Expression, ...]:
    if any(len(row) != len(basis) for row in coefficients):
        raise ValueError("linear-combination rows must align with the basis width.")
    dimension = _expression_dimension(basis[0]) if basis else 0
    return tuple(
        sum_polynomials(
            tuple(
                scale(function, coefficient)
                for function, coefficient in zip(basis, row, strict=True)
            ),
            dimension,
        )
        for row in coefficients
    )


@overload
def compose(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    arguments: tuple[Mapping[tuple[int, ...], ExactScalar], ...],
    /,
    *,
    dimension: int | None = None,
) -> Polynomial: ...


@overload
def compose(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    arguments: tuple[ExpressionInput, ...],
    /,
    *,
    dimension: int | None = None,
) -> Expression: ...


def compose(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    arguments: tuple[ExpressionInput, ...],
    /,
    *,
    dimension: int | None = None,
) -> Expression:
    expression = _canonical_polynomial(polynomial)
    known_dimensions = tuple(
        _expression_dimension(argument)
        for argument in arguments
        if isinstance(argument, RationalPolynomial) or bool(argument)
    )
    if known_dimensions and any(
        value != known_dimensions[0] for value in known_dimensions
    ):
        raise ValueError("composition arguments must share one target dimension.")
    dimension_ = (
        known_dimensions[0]
        if dimension is None and known_dimensions
        else 0
        if dimension is None
        else dimension
    )
    if dimension_ < 0:
        raise ValueError("compose.dimension must be nonnegative.")
    result: Expression = constant(0, dimension_)
    # Each argument power is formed once per call, not once per monomial.
    powers: dict[tuple[int, int], Expression] = {}
    for exponent, coefficient in expression.items():
        if len(exponent) != len(arguments):
            raise ValueError("composition arguments must match the polynomial dimension.")
        term: Expression = constant(coefficient, dimension_)
        for axis, degree in enumerate(exponent):
            if degree:
                factor = powers.get((axis, degree))
                if factor is None:
                    factor = power(arguments[axis], degree, dimension=dimension_)
                    powers[axis, degree] = factor
                term = multiply(term, factor)
        result = add(result, term)
    return result


def expression_equal(first: Expression, second: Expression, /) -> bool:
    if isinstance(first, RationalPolynomial) and isinstance(second, RationalPolynomial):
        return multiply(first.numerator, second.denominator) == multiply(
            second.numerator, first.denominator
        )
    if isinstance(first, RationalPolynomial):
        return multiply(second, first.denominator) == dict(first.numerator)
    if isinstance(second, RationalPolynomial):
        return multiply(first, second.denominator) == second.numerator
    return dict(first) == dict(second)


def expression_sequence_equal(
    first: Sequence[Expression], second: Sequence[Expression], /
) -> bool:
    return len(first) == len(second) and all(
        expression_equal(left, right) for left, right in zip(first, second, strict=True)
    )


def _expression_dimension(value: ExpressionInput, /) -> int:
    if isinstance(value, RationalPolynomial):
        return _polynomial_dimension(value.denominator)
    return _polynomial_dimension(value)


def _polynomial_dimension(value: Mapping[tuple[int, ...], ExactScalar], /) -> int:
    if not value:
        return 0
    dimension = len(next(iter(value)))
    if any(len(index) != dimension for index in value):
        raise ValueError("polynomial monomials must have one common dimension.")
    return dimension


def _falling(value: int, order: int, /) -> int:
    if order > value:
        return 0
    return prod(range(value - order + 1, value + 1))


def _compositions(total: int, parts: int, /) -> tuple[tuple[int, ...], ...]:
    if parts < 1 or total < 0:
        raise ValueError("simplex Bernstein indices require positive parts and degree.")
    if parts == 1:
        return ((total,),)
    return tuple(
        (first, *rest)
        for first in range(total, -1, -1)
        for rest in _compositions(total - first, parts - 1)
    )


def _simplex_weight(
    beta: tuple[int, ...], alpha: tuple[int, ...], degree: int, /
) -> Fraction:
    total = sum(alpha)
    if total > degree:
        return Fraction(0)
    numerator = prod(_falling(b, a) for b, a in zip(beta, alpha, strict=True))
    return Fraction(numerator, _falling(degree, total))


def _box_weight(
    beta: tuple[int, ...], alpha: tuple[int, ...], degrees: tuple[int, ...], /
) -> Fraction:
    weight = Fraction(1)
    for control, exponent, degree in zip(beta, alpha, degrees, strict=True):
        if exponent > degree or exponent > control:
            return Fraction(0)
        weight *= Fraction(comb(control, exponent), comb(degree, exponent))
    return weight


type _WeightColumn = tuple[tuple[Fraction, ...], tuple[tuple[int, Fraction], ...]]


def _weight_column(weights: tuple[Fraction, ...], /) -> _WeightColumn:
    return weights, tuple((row, weight) for row, weight in enumerate(weights) if weight)


@cache
def _simplex_weight_column(
    alpha: tuple[int, ...], degree: int, parts: int, /
) -> _WeightColumn:
    """Exact simplex Bernstein weights of one monomial, formed once per degree."""
    return _weight_column(
        tuple(
            _simplex_weight(beta, alpha, degree) for beta in _compositions(degree, parts)
        )
    )


@cache
def _box_weight_column(
    alpha: tuple[int, ...], degrees: tuple[int, ...], /
) -> _WeightColumn:
    """Exact tensor Bernstein weights of one monomial, formed once per degree."""
    return _weight_column(
        tuple(
            _box_weight(beta, alpha, degrees)
            for beta in product(*(range(degree + 1) for degree in degrees))
        )
    )


def _accumulate_controls(
    controls: list[ExactScalar], coefficient: ExactScalar, column: _WeightColumn, /
) -> None:
    weights, nonzero = column
    if isinstance(coefficient, Fraction):
        # A rational zero product leaves a rational control unchanged.
        for row, weight in nonzero:
            controls[row] += coefficient * weight
        return
    for row, weight in enumerate(weights):
        controls[row] += coefficient * weight


def bernstein_degree(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[int, ...]:
    """Return the least Bernstein degree of a polynomial on its domain.

    The degree is ``(total,)`` on a simplex, one degree per axis on a box, and
    ``(triangle, line)`` on a prism.
    """
    match domain:
        case "simplex":
            return (max((sum(index) for index in polynomial), default=0),)
        case "box":
            return tuple(
                max((index[axis] for index in polynomial), default=0)
                for axis in range(dimension)
            )
        case "prism":
            if dimension != 3:
                raise ValueError(
                    "prism Bernstein controls require three reference dimensions."
                )
            return (
                max((sum(index[:2]) for index in polynomial), default=0),
                max((index[2] for index in polynomial), default=0),
            )
        case _:
            raise ValueError(f"Unsupported Bernstein domain {domain!r}.")


def bernstein_coefficients(
    polynomial: Mapping[tuple[int, ...], ExactScalar],
    domain: BernsteinDomain,
    dimension: int,
    /,
    *,
    degree: tuple[int, ...] | None = None,
) -> tuple[ExactScalar, ...]:
    """Return exact Bernstein controls on a simplex, prism, or unit box.

    ``degree`` (shaped as `bernstein_degree`) elevates the controls; each
    elevated control is a convex combination of the least-degree controls.
    """
    value = _canonical_polynomial(polynomial)
    if dimension < 1 or any(len(index) != dimension for index in value):
        raise ValueError("Bernstein dimension must match polynomial monomial dimensions.")
    least = bernstein_degree(value, domain, dimension)
    if degree is None:
        degree = least
    elif len(degree) != len(least) or any(
        int(chosen) < int(minimum) for chosen, minimum in zip(degree, least, strict=True)
    ):
        raise ValueError("Bernstein elevation degree is below the polynomial degree.")
    controls: list[ExactScalar]
    match domain:
        case "simplex":
            (total,) = degree
            betas = _compositions(total, dimension + 1)
            controls = [Fraction(0) for _ in betas]
            for index, coefficient in value.items():
                _accumulate_controls(
                    controls,
                    coefficient,
                    _simplex_weight_column((0, *index), total, dimension + 1),
                )
            return tuple(controls)
        case "box":
            controls = [Fraction(0) for _ in range(prod(axis + 1 for axis in degree))]
            for index, coefficient in value.items():
                _accumulate_controls(
                    controls, coefficient, _box_weight_column(index, degree)
                )
            return tuple(controls)
        case "prism":
            triangle_degree, line_degree = degree
            triangle_indices = _compositions(triangle_degree, 3)
            controls = [
                Fraction(0) for _ in range(len(triangle_indices) * (line_degree + 1))
            ]
            for index, coefficient in value.items():
                alpha = (0, index[0], index[1])
                for triangle_row, beta in enumerate(triangle_indices):
                    triangle_weight = _simplex_weight(beta, alpha, triangle_degree)
                    for line_control in range(line_degree + 1):
                        line_weight = _box_weight(
                            (line_control,), (index[2],), (line_degree,)
                        )
                        row = triangle_row * (line_degree + 1) + line_control
                        controls[row] += coefficient * triangle_weight * line_weight
            return tuple(controls)
    raise ValueError(f"Unsupported Bernstein domain {domain!r}.")


def polynomial_bounds(
    polynomial: Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial,
    domain: BernsteinDomain,
    dimension: int,
    /,
) -> tuple[float, float]:
    """Return an outward binary64 enclosure from exact Bernstein controls."""
    if isinstance(polynomial, RationalPolynomial):
        numerator = polynomial_bounds(polynomial.numerator, domain, dimension)
        denominator = polynomial_bounds(polynomial.denominator, domain, dimension)
        lower_denominator, upper_denominator = denominator
        if lower_denominator <= 0.0 <= upper_denominator:
            return -math.inf, math.inf
        values = tuple(
            numerator_value / denominator_value
            for numerator_value in numerator
            for denominator_value in denominator
        )
        if any(math.isnan(value) for value in values):
            return -math.inf, math.inf
        return (
            float(np.nextafter(min(values), -np.inf)),
            float(np.nextafter(max(values), np.inf)),
        )
    controls = bernstein_coefficients(polynomial, domain, dimension)
    if not controls:
        return 0.0, 0.0
    return (
        coefficient_endpoint(min(controls), upper=False, saturate=True),
        coefficient_endpoint(max(controls), upper=True, saturate=True),
    )


def affine_arguments(origin: np.ndarray, matrix: np.ndarray, /) -> tuple[Polynomial, ...]:
    """Represent a finite binary64 affine chart exactly over Fraction coefficients."""
    offset = np.asarray(origin, dtype=np.float64)
    linear = np.asarray(matrix, dtype=np.float64)
    if (
        offset.ndim != 1
        or linear.ndim != 2
        or linear.shape[0] != offset.shape[0]
        or not np.all(np.isfinite(offset))
        or not np.all(np.isfinite(linear))
    ):
        raise ValueError(
            "affine chart inputs must be finite with compatible matrix shapes."
        )
    variables = axes(linear.shape[1])
    return tuple(
        add(
            constant(Fraction(float(offset[row])), linear.shape[1]),
            sum_polynomials(
                tuple(
                    scale(variables[column], Fraction(float(linear[row, column])))
                    for column in range(linear.shape[1])
                ),
                linear.shape[1],
            ),
        )
        for row in range(linear.shape[0])
    )


def evaluate_polynomial(
    polynomial: Mapping[tuple[int, ...], ExactScalar], point: tuple[ExactScalar, ...], /
) -> ExactScalar:
    total: ExactScalar = Fraction(0)
    for exponent, coefficient in polynomial.items():
        if len(exponent) != len(point):
            raise ValueError(
                "evaluation point dimension must match polynomial variables."
            )
        monomial: ExactScalar = coefficient
        for coordinate, power_ in zip(point, exponent, strict=True):
            if power_:
                monomial *= coordinate**power_
        total += monomial
    return total


def evaluate_expression(
    expression: ExpressionInput, point: tuple[ExactScalar, ...], /
) -> ExactScalar:
    if isinstance(expression, RationalPolynomial):
        numerator = evaluate_polynomial(expression.numerator, point)
        denominator = evaluate_polynomial(expression.denominator, point)
        if not denominator:
            raise ZeroDivisionError(
                "rational coordinate denominator vanishes at the point."
            )
        return numerator / denominator
    return evaluate_polynomial(expression, point)


def rounded_point(point: tuple[ExactScalar, ...], /) -> np.ndarray:
    """Return one nearest binary64 realization of an exact source point."""
    result = np.asarray([float(value) for value in point], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("rounded_point requires a finite binary64 realization.")
    return result


def coordinate_point_rounding_error(
    point: tuple[ExactScalar, ...], representative: np.ndarray, /
) -> Fraction:
    """Return an exact Euclidean error bound for one rounded point."""
    rounded = np.asarray(representative, dtype=np.float64)
    if rounded.shape != (len(point),) or not np.all(np.isfinite(rounded)):
        raise ValueError(
            "coordinate point representative must be a finite aligned vector."
        )
    errors = tuple(
        coefficient_rounding_error(value, float(actual))
        for value, actual in zip(point, rounded, strict=True)
    )
    squared = sum((error * error for error in errors), Fraction(0))
    from ..linalg._hermitian_spectral import fraction_sqrt_interval

    return fraction_sqrt_interval(squared)[1]


def compose_expression(
    expression: ExpressionInput, arguments: tuple[ExpressionInput, ...], /
) -> Expression:
    """Compose exact charts without treating rational operands as polynomial maps."""
    if isinstance(expression, RationalPolynomial):
        return multiply(
            compose(expression.numerator, arguments),
            power(compose(expression.denominator, arguments), -1),
        )
    return compose(expression, arguments)

class PyramidApexNormalization(NamedTuple):
    """Exact pyramid pullback and its collapsed-apex trace conditions."""

    expression: Expression
    apex_trace: Expression | None
    apex_value: ExactScalar | None


def pyramid_reference_to_box_arguments() -> tuple[Expression, Expression, Expression]:
    """Map the unit cube onto the canonical pyramid reference cell."""
    u, v, w = axes(3)
    one = constant(1, 3)
    width = add(one, scale(w, -1))
    half_height = scale(w, Fraction(1, 2))
    return (
        add(multiply(width, u), half_height),
        add(multiply(width, v), half_height),
        w,
    )


def _divide_by_one_minus_w(
    polynomial: Mapping[tuple[int, ...], ExactScalar], /
) -> Polynomial | None:
    """Exactly divide cube polynomials by the collapsed-pyramid factor."""
    if not polynomial:
        return {}
    coefficients_by_transverse: dict[tuple[int, ...], list[ExactScalar]] = {}
    for exponent, coefficient in polynomial.items():
        if len(exponent) != 3:
            raise ValueError("Pyramid pullback polynomials must use cube coordinates.")
        transverse = exponent[:2]
        coefficients = coefficients_by_transverse.setdefault(transverse, [])
        while len(coefficients) <= exponent[2]:
            coefficients.append(Fraction(0))
        coefficients[exponent[2]] = coefficients[exponent[2]] + coefficient

    quotient: Polynomial = {}
    for transverse, coefficients in coefficients_by_transverse.items():
        total: ExactScalar = Fraction(0)
        for coefficient in coefficients:
            total = total + coefficient
        if total != 0:
            return None
        partial: ExactScalar = Fraction(0)
        for exponent, coefficient in enumerate(coefficients[:-1]):
            partial = partial + coefficient
            if partial:
                quotient[(*transverse, exponent)] = partial
    return _canonical_polynomial(quotient)


def _cancel_shared_pyramid_apex_factors(
    expression: RationalPolynomial, /
) -> RationalPolynomial:
    numerator = dict(expression.numerator)
    denominator = dict(expression.denominator)
    while True:
        numerator_quotient = _divide_by_one_minus_w(numerator)
        if numerator_quotient is None:
            break
        denominator_quotient = _divide_by_one_minus_w(denominator)
        if denominator_quotient is None:
            break
        numerator = numerator_quotient
        denominator = denominator_quotient
    return RationalPolynomial(numerator, denominator)


def _pyramid_apex_trace(
    polynomial: Mapping[tuple[int, ...], ExactScalar], /
) -> Polynomial:
    trace: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        if len(exponent) != 3:
            raise ValueError("Pyramid pullback polynomials must use cube coordinates.")
        trace_exponent = exponent[:2]
        trace[trace_exponent] = trace.get(trace_exponent, Fraction(0)) + coefficient
    return _canonical_polynomial(trace)


def _pyramid_apex_trace_and_value(
    expression: Expression, /
) -> tuple[Expression | None, ExactScalar | None]:
    if isinstance(expression, RationalPolynomial):
        numerator = expression.numerator
        denominator = expression.denominator
    else:
        numerator = expression
        denominator: Polynomial = {(0, 0, 0): Fraction(1)}
    numerator_trace = _pyramid_apex_trace(numerator)
    denominator_trace = _pyramid_apex_trace(denominator)
    if not denominator_trace:
        return None, None
    if (
        len(denominator_trace) == 1
        and denominator_trace.get((0, 0)) == Fraction(1)
    ):
        trace_expression: Expression = numerator_trace
    else:
        trace_expression = RationalPolynomial(numerator_trace, denominator_trace)
    pivot = min(denominator_trace)
    numerator_coefficient = numerator_trace.get(pivot, Fraction(0))
    denominator_coefficient = denominator_trace[pivot]
    apex_value = (
        numerator_coefficient / denominator_coefficient
        if isinstance(denominator_coefficient, Fraction)
        else numerator_coefficient * denominator_coefficient.reciprocal()
    )
    for exponent in numerator_trace.keys() | denominator_trace.keys():
        if numerator_trace.get(exponent, Fraction(0)) != apex_value * denominator_trace.get(
            exponent, Fraction(0)
        ):
            return trace_expression, None
    return trace_expression, apex_value


def normalize_pyramid_reference_expression(
    expression: ExpressionInput, /
) -> PyramidApexNormalization:
    """Pull a pyramid-coordinate map to the cube and cancel common apex factors.

    Only exact common powers of one minus w are canceled. The returned
    denominator remains subject to a separate nonzero-domain proof, and the
    source expression is never rewritten.
    """
    pulled = compose_expression(expression, pyramid_reference_to_box_arguments())
    if isinstance(pulled, RationalPolynomial):
        normalized_rational = _cancel_shared_pyramid_apex_factors(pulled)
        unit = (0, 0, 0)
        normalized: Expression = (
            dict(normalized_rational.numerator)
            if len(normalized_rational.denominator) == 1
            and normalized_rational.denominator.get(unit) == Fraction(1)
            else normalized_rational
        )
    else:
        normalized = pulled
    apex_trace, apex_value = _pyramid_apex_trace_and_value(normalized)
    return PyramidApexNormalization(normalized, apex_trace, apex_value)


def normalize_pyramid_reference_derivative(
    expression: ExpressionInput, physical_axis: int, /
) -> PyramidApexNormalization:
    """Differentiate in pyramid coordinates before exact cube pullback."""
    return normalize_pyramid_reference_expression(derivative(expression, physical_axis))


def coordinate_reference_chain(
    element: ScalarCoordinateElement,
    /,
    *,
    ancestor: ScalarCoordinateElement | None = None,
    cell_row: int | None = None,
) -> tuple[ScalarCoordinateElement, tuple[Expression, ...]]:
    """Return the retained root element and exact local-to-root chart map."""
    owner = _cell_geometry_dispatch()
    chain = owner.coordinate_reference_chain(element, ancestor=ancestor)
    if not chain:
        raise ValueError("Coordinate source ancestry must retain a root element.")
    arguments: tuple[Expression, ...] | None = None
    for step in chain[1:]:
        local = owner.coordinate_reference_arguments(step, cell_row=cell_row)
        if local is None:
            continue
        arguments = (
            local
            if arguments is None
            else tuple(compose_expression(value, local) for value in arguments)
        )
    if arguments is None:
        from ._reference_cell import reference_cell_topology

        arguments = axes(reference_cell_topology(element.cell_kind).dimension)
    return chain[0], arguments


def reference_composition_arguments(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Expression, ...]:
    return coordinate_reference_chain(element, cell_row=cell_row)[1]


def restriction_chart_arguments(
    element: ScalarCoordinateElement,
    /,
    *,
    ancestor: ScalarCoordinateElement | None = None,
    cell_row: int | None = None,
) -> tuple[Expression, ...]:
    return coordinate_reference_chain(element, ancestor=ancestor, cell_row=cell_row)[1]


def coordinate_source_basis(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Expression, ...]:
    return _cell_geometry_dispatch().coordinate_effective_basis(
        element, cell_row=cell_row
    )


def coordinate_execution_basis(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Expression, ...]:
    """Compose the actual binary64 element-coefficient shadows."""
    return _cell_geometry_dispatch().coordinate_execution_basis(
        element, cell_row=cell_row
    )


def coordinate_partition_unity_reference_chain(
    element: ScalarCoordinateElement,
    /,
    *,
    ancestor: ScalarCoordinateElement | None = None,
    cell_row: int | None = None,
) -> tuple[ScalarCoordinateElement, tuple[Expression, ...]]:
    root, arguments = coordinate_reference_chain(
        element, ancestor=ancestor, cell_row=cell_row
    )
    basis = _cell_geometry_dispatch().coordinate_source_basis(root)
    dimension = len(arguments)
    if not expression_equal(sum_polynomials(basis, dimension), constant(1, dimension)):
        raise ValueError(
            "Coordinate reference chain does not preserve partition of unity."
        )
    return root, arguments


def _source_bank(value: CoordinateCoefficients, /) -> CoordinateSourceBank:
    if isinstance(value, np.ndarray):
        if (
            value.ndim != 2
            or value.dtype.kind not in "fiu"
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                "Coordinate coefficient arrays must be finite real matrices."
            )
        return tuple(tuple(_scalar(component) for component in row) for row in value)
    rows = tuple(tuple(_scalar(component) for component in row) for row in value)
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Coordinate source rows must share one ambient dimension.")
    return rows


def coordinate_map_expressions(
    element: ScalarCoordinateElement,
    local: CoordinateCoefficients,
    /,
    *,
    cell_row: int | None = None,
) -> tuple[Expression, ...]:
    """Compose an exact element basis with its represented coordinate bank."""
    basis = coordinate_source_basis(element, cell_row=cell_row)
    bank = _source_bank(local)
    if len(basis) != len(bank):
        raise ValueError("Exact coordinate basis and local source bank widths differ.")
    if not bank:
        raise ValueError("Coordinate map requires a nonempty coefficient bank.")
    components = tuple(
        tuple(bank[dof][axis] for dof in range(len(bank))) for axis in range(len(bank[0]))
    )
    return linear_combinations(basis, components)


def coordinate_execution_map_expressions(
    element: ScalarCoordinateElement,
    local: CoordinateCoefficients,
    /,
    *,
    cell_row: int | None = None,
) -> tuple[Expression, ...]:
    """Compose the execution basis with its binary64 coordinate coefficients."""
    basis = coordinate_execution_basis(element, cell_row=cell_row)
    bank = _source_bank(local)
    if len(basis) != len(bank):
        raise ValueError("Execution coordinate basis and local coefficient bank widths differ.")
    if not bank:
        raise ValueError("Coordinate map requires a nonempty coefficient bank.")
    components = tuple(
        tuple(bank[dof][axis] for dof in range(len(bank)))
        for axis in range(len(bank[0]))
    )
    return linear_combinations(basis, components)


def coordinate_corner_images(
    element: ScalarCoordinateElement,
    local: CoordinateCoefficients,
    /,
    *,
    cell_row: int | None = None,
) -> tuple[tuple[ExactScalar, ...], ...]:
    """Evaluate the exact coordinate map at every reference-cell vertex."""
    from ._reference_cell import reference_cell_topology

    expressions = coordinate_map_expressions(element, local, cell_row=cell_row)
    topology = reference_cell_topology(element.cell_kind)
    vertices = tuple(
        tuple(_scalar(component) for component in point)
        for point in np.asarray(topology.vertices, dtype=np.float64)
    )
    return tuple(
        tuple(evaluate_expression(expression, point) for expression in expressions)
        for point in vertices
    )


def coordinate_map_images(
    element: ScalarCoordinateElement,
    local: CoordinateCoefficients,
    reference_points: Sequence[Sequence[ExactScalar | float]],
    /,
    *,
    cell_row: int | None = None,
) -> tuple[tuple[ExactScalar, ...], ...]:
    expressions = coordinate_map_expressions(element, local, cell_row=cell_row)
    points = tuple(tuple(_scalar(value) for value in row) for row in reference_points)
    return tuple(
        tuple(evaluate_expression(expression, point) for expression in expressions)
        for point in points
    )


def coordinate_map_bounds(
    element: ScalarCoordinateElement,
    local: CoordinateCoefficients,
    domain: BernsteinDomain,
    /,
    *,
    cell_row: int | None = None,
) -> tuple[tuple[float, float], ...]:
    """Enclose every physical coordinate over the complete reference cell."""
    expressions = coordinate_map_expressions(element, local, cell_row=cell_row)
    dimension = element.topological_dimension
    return tuple(
        polynomial_bounds(expression, domain, dimension) for expression in expressions
    )


def coordinate_source_signature(element: ScalarCoordinateElement, /) -> dict[str, object]:
    """Retain exact source element identity and full ancestry structure."""
    chain = _cell_geometry_dispatch().coordinate_reference_chain(element)
    return {
        "element_id": element.element_id,
        "chain_ids": tuple(item.element_id for item in chain),
        "element_arrays": array_tree_fingerprint(element),
    }


__all__ = [
    "AffineMap",
    "BernsteinDomain",
    "CoordinateCoefficients",
    "CoordinateSourceBank",
    "ExactScalar",
    "Expression",
    "ExpressionInput",
    "Polynomial",
    "RationalPolynomial",
    "PyramidApexNormalization",
    "add",
    "affine_arguments",
    "affine_map",
    "axes",
    "bernstein_coefficients",
    "bernstein_degree",
    "compose",
    "compose_expression",
    "constant",
    "coordinate_corner_images",
    "coordinate_map_bounds",
    "coordinate_execution_basis",
    "coordinate_execution_map_expressions",
    "coordinate_map_expressions",
    "coordinate_partition_unity_reference_chain",
    "coordinate_point_rounding_error",
    "coordinate_reference_chain",
    "coordinate_source_basis",
    "coordinate_source_signature",
    "derivative",
    "evaluate_expression",
    "evaluate_polynomial",
    "expression_equal",
    "expression_sequence_equal",
    "linear_combinations",
    "multiply",
    "polynomial_bounds",
    "power",
    "normalize_pyramid_reference_derivative",
    "normalize_pyramid_reference_expression",
    "pyramid_reference_to_box_arguments",
    "reference_composition_arguments",
    "restriction_chart_arguments",
    "rounded_point",
    "scale",
    "subtract",
    "sum_polynomials",
]
