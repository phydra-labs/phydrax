"""Directed trim-period bounds preserve scalar-array storage and enclose source values."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry.brep._trim_helpers import period_scalar_bounds, PeriodOffset


def _atan_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    # The alternating atan series has decreasing positive magnitudes here.
    # An even 32-term sum is below atan(value); its next term bounds the remainder.
    lower = sum(
        (
            (-1) ** index * value ** (2 * index + 1) / (2 * index + 1)
            for index in range(32)
        ),
        Fraction(),
    )
    return lower, lower + value**65 / 65


_FIVE_LOWER, _FIVE_UPPER = _atan_bounds(Fraction(1, 5))
_OTHER_LOWER, _OTHER_UPPER = _atan_bounds(Fraction(1, 239))
# Machin's identity: pi = 16 atan(1/5) - 4 atan(1/239).
_PI_LOWER = 16 * _FIVE_LOWER - 4 * _OTHER_UPPER
_PI_UPPER = 16 * _FIVE_UPPER - 4 * _OTHER_LOWER
_RATIONAL = Fraction(1, 7)


@pytest.mark.parametrize(
    ("value", "reference_lower", "reference_upper"),
    [
        (_RATIONAL, _RATIONAL, _RATIONAL),
        (
            PeriodOffset(_RATIONAL, Fraction(1, 3)),
            _RATIONAL + Fraction(2, 3) * _PI_LOWER,
            _RATIONAL + Fraction(2, 3) * _PI_UPPER,
        ),
        (
            PeriodOffset(_RATIONAL, Fraction(-7, 5)),
            _RATIONAL - Fraction(14, 5) * _PI_UPPER,
            _RATIONAL - Fraction(14, 5) * _PI_LOWER,
        ),
        (PeriodOffset(_RATIONAL, Fraction()), _RATIONAL, _RATIONAL),
    ],
    ids=["rational", "positive-turns", "negative-turns", "zero-turns"],
)
def test_period_bounds_retain_float64_scalar_arrays_and_contain_source_expression(
    value: Fraction | PeriodOffset,
    reference_lower: Fraction,
    reference_upper: Fraction,
) -> None:
    lower, upper = period_scalar_bounds(value)
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert lower.shape == upper.shape == ()
    assert lower.dtype == upper.dtype == np.dtype(np.float64)
    assert Fraction(float(lower)) <= reference_lower
    assert Fraction(float(upper)) >= reference_upper
