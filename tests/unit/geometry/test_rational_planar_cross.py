"""Rational planar orientation identities used by B-Rep arrangements."""

from fractions import Fraction

import pytest

from phydrax.geometry._rational import rational_cross_2d


type RationalPoint = tuple[Fraction, Fraction]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        pytest.param(
            (Fraction(1, 3), Fraction(2, 5)),
            (Fraction(7, 11), Fraction(-3, 13)),
            Fraction(-237, 715),
            id="fractional-orientation",
        ),
        pytest.param(
            (Fraction(7, 11), Fraction(-3, 13)),
            (Fraction(1, 3), Fraction(2, 5)),
            Fraction(237, 715),
            id="reversed-orientation",
        ),
        pytest.param(
            (Fraction(1, 3), Fraction(2, 5)),
            (Fraction(2, 3), Fraction(4, 5)),
            Fraction(0),
            id="collinear",
        ),
    ],
)
def test_rational_planar_cross_has_the_signed_reference_value(
    first: RationalPoint, second: RationalPoint, expected: Fraction
) -> None:
    assert rational_cross_2d(first, second) == expected


def test_rational_planar_cross_retains_cancellation_below_binary64_product_resolution() -> (
    None
):
    # All coordinates are representable in binary64. The determinant of
    # [[n + 1, n], [n, n - 1]] is -1 for every integer n, even though rounded
    # binary64 products at this n cannot retain that unit difference.
    n = 2**52
    first = Fraction(n + 1), Fraction(n)
    second = Fraction(n), Fraction(n - 1)
    assert rational_cross_2d(first, second) == -1
