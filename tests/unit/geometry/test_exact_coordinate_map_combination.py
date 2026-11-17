"""Exact coordinate-basis combination and mapped coverage regressions."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from random import Random

import pytest

from phydrax.discretization import (
    _coordinate_enclosure as e,
    lagrange_element,
    lagrange_reference_node_data,
)
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry import _mapped_coverage as coverage


def _generic_combinations(
    basis: Sequence[e.ExpressionInput],
    rows: Sequence[Sequence[e.ExactScalar]],
    dimension: int,
) -> tuple[e.Expression, ...]:
    """Combine with generic exact expression operations, not the prepared kernel."""
    results = []
    for row in rows:
        result = e.constant(0, dimension)
        for function, coefficient in zip(basis, row, strict=True):
            result = e.add(result, e.scale(function, coefficient))
        results.append(result)
    return tuple(results)


@pytest.mark.parametrize(
    "cell_kind", ["triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism"]
)
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_random_exact_coordinate_maps_match_generic_basis(
    cell_kind: str, degree: int
) -> None:
    element = lagrange_element(cell_kind, degree)
    basis = e.coordinate_source_basis(element)
    dimension = element.topological_dimension
    rng = Random(82431 + degree * 101 + dimension * 11 + len(basis))
    prepared = e.PreparedLinearCombinationBasis(basis)
    for sample in range(5):
        bank = tuple(
            tuple(
                Fraction(rng.randint(-20, 20), rng.choice((1, 2, 3, 5, 7, 16)))
                for _ in range(3)
            )
            for _ in basis
        )
        if sample == 0:
            bank = tuple((Fraction(0), Fraction(1, 3), row[2]) for row in bank)
        rows = tuple(tuple(row[axis] for row in bank) for axis in range(3))
        expected = _generic_combinations(basis, rows, dimension)
        actual = prepared.combine(rows)
        assert actual == expected
        assert e.coordinate_map_expressions(element, bank) == expected
        assert tuple(prepared.combine((row,))[0] for row in rows) == expected
        assert prepared.combine(rows[::-1]) == expected[::-1]
        assert prepared.combine((rows[0], rows[1], rows[0])) == (
            expected[0],
            expected[1],
            expected[0],
        )


def test_radical_coefficients_and_rational_basis_use_exact_combinations() -> None:
    root = RadicalElement.sqrt(2)
    polynomial_basis: tuple[e.Polynomial, ...] = (
        {(0, 0): Fraction(1, 3), (1, 0): Fraction(2, 5)},
        {(0, 0): Fraction(-1, 3), (0, 1): Fraction(7, 2)},
    )
    denominator: e.Polynomial = {(0, 0): Fraction(1), (1, 0): Fraction(1, 3)}
    rational_basis = tuple(
        e.RationalPolynomial(value, denominator) for value in polynomial_basis
    )
    distinct_denominators = (
        rational_basis[0],
        e.RationalPolynomial(
            polynomial_basis[1], {(0, 0): Fraction(1), (0, 1): Fraction(1, 5)}
        ),
    )
    radical_basis: tuple[e.Expression, ...] = (
        polynomial_basis[0],
        {(1, 0): root, (0, 1): Fraction(2, 3)},
    )
    rng = Random(977)
    rows = tuple(
        (
            Fraction(rng.randint(-5, 5), 3) + root * rng.randint(-3, 3),
            Fraction(rng.randint(-5, 5), 7) + root * rng.randint(-3, 3),
        )
        for _ in range(8)
    ) + ((Fraction(0), Fraction(0)),)
    for basis in (polynomial_basis, rational_basis, distinct_denominators, radical_basis):
        prepared = e.PreparedLinearCombinationBasis(basis)
        expected = _generic_combinations(basis, rows, 2)
        actual = prepared.combine(rows)
        if basis is polynomial_basis or basis is radical_basis:
            assert actual == expected
        assert all(
            e.expression_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
        for row, result in zip(rows, actual, strict=True):
            assert e.expression_equal(prepared.combine((row,))[0], result)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_coordinate_map_exact_coverage_for_polynomial_triangle(degree: int) -> None:
    element = lagrange_element("triangle", degree)
    nodes = lagrange_reference_node_data(element)
    coefficient = Fraction(2, 3)
    bank = []
    for node in nodes:
        assert node.exact_point is not None
        u, v = node.exact_point
        bank.append((u + coefficient * u**degree, v))
    coordinates = e.coordinate_map_expressions(element, tuple(bank))
    assert coordinates == _generic_combinations(
        e.coordinate_source_basis(element),
        tuple(tuple(row[axis] for row in bank) for axis in range(2)),
        2,
    )
    # det(D map) = 1 + a*d*u**(d-1) on the unit triangle.
    assert coverage.integrate_jacobian(coordinates, "triangle") == Fraction(
        1, 2
    ) + coefficient / (degree + 1)
    assert coverage.integrate_jacobian(coordinates[::-1], "triangle") == -Fraction(
        1, 2
    ) - coefficient / (degree + 1)
