from __future__ import annotations

from fractions import Fraction
from random import Random

import pytest

from phydrax.discretization._coordinate_enclosure import (
    compose_expression,
    Expression,
    Polynomial,
    RationalPolynomial,
)
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry._stratum_subdivision import _compose_root_expression


def _reference(
    outer: dict[tuple[int, ...], Fraction],
    arguments: tuple[dict[tuple[int, ...], Fraction], ...],
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for exponent, coefficient in outer.items():
        terms = {(0, 0, 0): coefficient}
        for argument, degree in zip(arguments, exponent, strict=True):
            for _ in range(degree):
                product: dict[tuple[int, ...], Fraction] = {}
                for left, value in terms.items():
                    for right, factor in argument.items():
                        index = tuple(a + b for a, b in zip(left, right, strict=True))
                        product[index] = product.get(index, Fraction(0)) + value * factor
                terms = product
        for index, value in terms.items():
            result[index] = result.get(index, Fraction(0)) + value
    return {index: value for index, value in result.items() if value}


def test_quadratic_rational_composition_matches_fraction_reference() -> None:
    random = Random(51721)
    indices: list[tuple[int, ...]] = [
        (x, y, z) for x in range(3) for y in range(3 - x) for z in range(3 - x - y)
    ]
    for _ in range(250):
        polynomials = [
            {
                index: Fraction(random.randrange(-8, 9), random.randrange(1, 12))
                for index in indices
                if random.randrange(3) == 0
            }
            or {(0, 0, 0): Fraction(0)}
            for _ in range(4)
        ]
        outer, *inner = polynomials
        arguments = tuple(inner)
        outer_expression: Polynomial = {index: value for index, value in outer.items()}
        argument_expressions: tuple[Expression, ...] = tuple(
            {index: value for index, value in argument.items()} for argument in arguments
        )
        actual = _compose_root_expression(outer_expression, argument_expressions)
        expected = compose_expression(outer_expression, argument_expressions)
        assert actual == _reference(outer, arguments)
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise TypeError("Polynomial composition returned a rational expression.")
        assert list(actual.items()) == list(expected.items())


def test_nonquadratic_and_algebraic_expressions_use_general_composition() -> None:
    identity: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    expressions: tuple[Expression, ...] = (
        {(3, 0, 0): Fraction(2, 3)},
        {(1, 0, 0): RadicalElement.sqrt(2)},
        RationalPolynomial({(1, 0, 0): Fraction(1)}, {(0, 0, 0): Fraction(2)}),
    )
    for expression in expressions:
        actual = _compose_root_expression(expression, identity)
        expected = compose_expression(expression, identity)
        if isinstance(actual, RationalPolynomial) and isinstance(
            expected, RationalPolynomial
        ):
            assert actual.numerator == expected.numerator
            assert actual.denominator == expected.denominator
        else:
            assert actual == expected


def test_malformed_composition_preserves_error() -> None:
    expression: Expression = {(-1, 0, 0): Fraction(1)}
    arguments: tuple[Expression, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    with pytest.raises(
        ValueError, match="polynomial exponents must be nonnegative integers"
    ):
        _compose_root_expression(expression, arguments)
