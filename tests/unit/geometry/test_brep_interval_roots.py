"""Exact root-count checks for the intersection projection isolation route."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.algebraic._exact_univariate import RealRootIsolator
from phydrax.geometry.brep._interval_roots import (
    _coefficient_bounds,
    _program,
    IntervalRootIsolator,
)


F = Fraction


def _check_roots(
    polynomial: tuple[Fraction, ...],
    actual: tuple[tuple[Fraction, Fraction, int], ...],
) -> None:
    reference = RealRootIsolator(polynomial).isolate(tolerance=F(1, 2**35))
    assert len(actual) == len(reference)
    for (lo, hi, multiplicity), (exact_lo, exact_hi, exact_multiplicity) in zip(
        actual, reference, strict=True
    ):
        assert lo <= exact_hi and exact_lo <= hi
        assert multiplicity == exact_multiplicity

        def evaluate(x: Fraction) -> Fraction:
            result = F(0)
            for coefficient in reversed(polynomial):
                result = result * x + coefficient
            return result

        if lo == hi:
            assert evaluate(lo) == 0
        elif multiplicity == 1:
            assert evaluate(lo) * evaluate(hi) < 0
    assert all(left[1] < right[0] for left, right in zip(actual, actual[1:]))


@pytest.mark.parametrize(
    "polynomial",
    [
        (F(-2), F(0), F(1)),
        (F(1), F(-2), F(1)),
        (F(0), F(-1), F(1)),
        (F(2**2000), F(1)),
        (F(2**40 + 1), F(-(2**41 + 1)), F(2**40)),
    ],
)
def test_certified_roots_cover_exact_roots(polynomial: tuple[Fraction, ...]) -> None:
    isolator = IntervalRootIsolator(polynomial)
    for width in (F(1, 2**8), F(1, 2**20), F(1, 2**30)):
        roots = isolator.isolate(tolerance=width)
        _check_roots(polynomial, roots)
        assert all(hi - lo <= width for lo, hi, _ in roots)


def test_random_polynomials_match_exact_root_counts() -> None:
    generator = np.random.default_rng(9281)
    for _ in range(30):
        coefficients = generator.integers(-7, 8, size=5, dtype=np.int64)
        if coefficients[-1] == 0:
            coefficients[-1] = 1
        polynomial = tuple(F(int(value)) for value in coefficients)
        roots = IntervalRootIsolator(polynomial).isolate(tolerance=F(1, 2**20))
        _check_roots(polynomial, roots)


def test_double_root_retains_exact_fallback_result() -> None:
    polynomial = (F(1), F(-2), F(1))
    width = F(1, 2**20)
    assert IntervalRootIsolator(polynomial).isolate(tolerance=width) == (
        RealRootIsolator(polynomial).isolate(tolerance=width)
    )


def test_wrong_polynomial_enclosure_is_detected() -> None:
    polynomial = (F(-2), F(0), F(1))
    wrong_input = (F(-3), F(0), F(1))
    wrong_roots = IntervalRootIsolator(wrong_input).isolate(tolerance=F(1, 2**20))
    with pytest.raises(AssertionError):
        _check_roots(polynomial, wrong_roots)


def test_inward_coefficient_enclosure_is_detected() -> None:
    coefficients = (2**53 + 1, 1)
    bounds = _coefficient_bounds(coefficients)
    assert bounds is not None
    lower, upper = bounds
    exact = F(coefficients[0], 2**54)

    def check_enclosure() -> None:
        result_lower, result_upper = _program().evaluate(lower[None, :], upper[None, :])
        assert F.from_float(float(result_lower[0, 0])) <= exact
        assert exact <= F.from_float(float(result_upper[0, 0]))

    check_enclosure()
    wrong = np.nextafter(lower[1] - 1.0e-12, -np.inf)
    lower[1] = upper[1] = wrong
    with pytest.raises(AssertionError):
        check_enclosure()
