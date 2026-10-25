# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Rational orientation actions of the small linear algebra substrate."""

from fractions import Fraction

import pytest

from phydrax.linalg._small_batched import (
    exact_cofactor_action_2,
    exact_cofactor_determinant_2,
)


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


@pytest.mark.parametrize("reversed_rows", (False, True), ids=("positive", "negative"))
def test_exact_two_row_action_preserves_rational_solution(reversed_rows: bool) -> None:
    first = Fraction(1, 3), Fraction(1, 7)
    second = Fraction(1, 5), Fraction(1, 2)
    # Independent substitution of x=2/7, y=-3/5 in the two rows.
    right = Fraction(1, 105), Fraction(-17, 70)
    if reversed_rows:
        first, second, right = second, first, right[::-1]
    assert exact_cofactor_action_2(first, second, right) == (
        Fraction(2, 7),
        Fraction(-3, 5),
    )


@pytest.mark.parametrize(
    "right",
    ((Fraction(1), Fraction(2)), (Fraction(1), Fraction(3))),
    ids=("dependent-consistent", "dependent-inconsistent"),
)
def test_exact_two_row_action_refuses_nonunique_systems(
    right: tuple[Fraction, Fraction],
) -> None:
    assert (
        exact_cofactor_action_2(
            (Fraction(1, 3), Fraction(1, 7)),
            (Fraction(2, 3), Fraction(2, 7)),
            right,
        )
        is None
    )
