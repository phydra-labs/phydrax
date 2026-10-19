# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rational orientation actions of the small linear algebra substrate."""

from fractions import Fraction

import pytest

from phydrax.linalg._small_batched import exact_cofactor_determinant_2


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    (
        (
            (Fraction(1, 3), Fraction(1, 7)),
            (Fraction(1, 5), Fraction(1, 2)),
            Fraction(29, 210),
        ),
        (
            (Fraction(1, 5), Fraction(1, 2)),
            (Fraction(1, 3), Fraction(1, 7)),
            Fraction(-29, 210),
        ),
        ((Fraction(1, 3), Fraction(1, 7)), (Fraction(2, 3), Fraction(2, 7)), Fraction(0)),
    ),
    ids=("positive-rational-area", "reversed-rational-area", "collapsed-rational-area"),
)
def test_rational_orientation_retains_sign_and_fraction(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    expected: Fraction,
) -> None:
    # The positive case is 1/6 - 1/35 = 29/210, evaluated without binary rounding.
    assert exact_cofactor_determinant_2(first, second) == expected
