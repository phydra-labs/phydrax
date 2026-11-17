from fractions import Fraction
from itertools import permutations, product
from math import factorial, prod
from random import Random

import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._coordinate_enclosure import Polynomial, RationalPolynomial
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry import _mapped_coverage as coverage


def _reference_integral(coordinates: tuple[Polynomial, ...], domain: str) -> Fraction:
    dimension = len(coordinates)
    total = Fraction(0)
    for permutation in permutations(range(dimension)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(dimension)
            for j in range(i + 1, dimension)
        )
        terms: dict[tuple[int, ...], Fraction] = {
            (0,) * dimension: Fraction((-1) ** inversions)
        }
        for row, axis in enumerate(permutation):
            next_terms: dict[tuple[int, ...], Fraction] = {}
            for exponent, coefficient in coordinates[row].items():
                if exponent[axis] == 0:
                    continue
                if not isinstance(coefficient, Fraction):
                    raise TypeError(
                        "The independent reference requires rational coefficients."
                    )
                derivative_exponent = tuple(
                    power - (index == axis) for index, power in enumerate(exponent)
                )
                for previous, previous_coefficient in terms.items():
                    combined = tuple(
                        a + b for a, b in zip(previous, derivative_exponent, strict=True)
                    )
                    next_terms[combined] = next_terms.get(combined, Fraction(0)) + (
                        previous_coefficient * coefficient * exponent[axis]
                    )
            terms = next_terms
        for exponent, coefficient in terms.items():
            if domain == "simplex":
                moment = Fraction(
                    prod(factorial(power) for power in exponent),
                    factorial(dimension + sum(exponent)),
                )
            elif domain == "prism":
                a, b, c = exponent
                moment = Fraction(
                    factorial(a) * factorial(b), factorial(a + b + 2) * (c + 1)
                )
            else:
                moment = Fraction(1, prod(power + 1 for power in exponent))
            total += coefficient * moment
    return total


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable")
@pytest.mark.parametrize(
    "cell_kind,dimension,domain",
    [
        ("triangle", 2, "simplex"),
        ("quadrilateral", 2, "box"),
        ("tetrahedron", 3, "simplex"),
        ("hexahedron", 3, "box"),
        ("prism", 3, "prism"),
    ],
)
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_random_polynomial_signed_coverage(
    cell_kind: str, dimension: int, domain: str, degree: int
) -> None:
    rng = Random(431 + dimension * 10 + degree)
    exponents = [
        powers
        for powers in product(range(degree + 1), repeat=dimension)
        if sum(powers) <= degree
    ]
    maps: list[tuple[Polynomial, ...]] = []
    for _ in range(8):
        coordinates: tuple[Polynomial, ...] = tuple(
            {
                powers: Fraction(rng.randint(-3, 3), rng.randint(1, 5))
                for powers in exponents
            }
            for _ in range(dimension)
        )
        maps.extend(
            (
                coordinates,
                (coordinates[1], coordinates[0], *coordinates[2:]),
                (coordinates[0], coordinates[0], *coordinates[2:]),
            )
        )
    expected = tuple(_reference_integral(coordinates, domain) for coordinates in maps)
    assert tuple(coverage.integrate_jacobians(maps, cell_kind)) == expected
    assert all(
        expected[index + 1] == -expected[index] and expected[index + 2] == 0
        for index in range(0, len(expected), 3)
    )


def test_exact_integrals_preserve_algebraic_and_rational_maps() -> None:
    x: Polynomial = {(1, 0): Fraction(1)}
    y: Polynomial = {(0, 1): Fraction(1)}
    assert coverage.integrate_jacobian((x, y), "triangle") == Fraction(1, 2)
    assert coverage.integrate_jacobian(
        ({(1, 0): RadicalElement.sqrt(2)}, y), "quadrilateral"
    ) == RadicalElement.sqrt(2)
    constant_denominator = RationalPolynomial({(1, 0): 2}, {(0, 0): 2})
    assert coverage.integrate_jacobian((constant_denominator, y), "triangle") == Fraction(
        1, 2
    )
    rational = RationalPolynomial({(1, 0): 1}, {(0, 0): 1, (1, 0): 1})
    with pytest.raises(ValueError, match="polynomial or constant denominator"):
        coverage.integrate_jacobian((rational, y), "quadrilateral")
    identity3: tuple[Polynomial, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    assert coverage.integrate_jacobian(identity3, "pyramid") == Fraction(1, 3)
    huge: tuple[Polynomial, ...] = ({(1, 0): Fraction(1 << 4200)}, y)
    assert coverage.integrate_jacobian(huge, "triangle") == Fraction(1 << 4199)


def test_batch_yields_prior_measures_before_later_preparation_error() -> None:
    inverted: tuple[Polynomial, ...] = ({(1, 0): Fraction(-1)}, {(0, 1): Fraction(1)})
    invalid: tuple[Polynomial, ...] = ({(1,): Fraction(1)},)
    measures = coverage.integrate_jacobians((inverted, invalid), "triangle")
    assert next(measures) == Fraction(-1, 2)
    with pytest.raises(ValueError, match="2-D or 3-D"):
        next(measures)


def test_mixed_large_coefficients_and_zero_have_exact_measures() -> None:
    y: Polynomial = {(0, 1): Fraction(1)}
    huge: tuple[Polynomial, ...] = ({(1, 0): Fraction(1 << 4200)}, y)
    ordinary: tuple[Polynomial, ...] = ({(1, 0): Fraction(1, 7)}, y)
    zero: tuple[Polynomial, ...] = (y, y)
    assert tuple(coverage.integrate_jacobians((huge, ordinary, zero), "triangle")) == (
        Fraction(1 << 4199),
        Fraction(1, 14),
        Fraction(0),
    )
