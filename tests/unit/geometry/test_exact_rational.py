from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry._rational import (
    exact_float,
    fraction_endpoint,
    rational_cross,
    rational_vector,
    translation_ranges,
)


def test_binary64_lifts_and_directed_rational_endpoints_are_exact() -> None:
    value = 0.1
    assert rational_vector((value,)) == (Fraction.from_float(value),)

    rational = Fraction(1, 3)
    lower = fraction_endpoint(rational, upper=False)
    upper = fraction_endpoint(rational, upper=True)
    assert Fraction(lower) <= rational <= Fraction(upper)
    with pytest.raises(ValueError, match="exact"):
        exact_float(rational)


def test_rational_cross_and_translation_ranges_keep_exact_dimensions() -> None:
    assert rational_cross(
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
    ) == (Fraction(0), Fraction(0), Fraction(1))
    boxes = [
        (
            np.asarray((0.0, 0.0), dtype=np.float64),
            np.asarray((1.0, 1.0), dtype=np.float64),
        )
    ]
    assert translation_ranges(
        boxes,
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
    ) == (range(-1, 2), range(-1, 2))


def test_rational_endpoint_saturation_is_outward() -> None:
    maximum = Fraction(float(np.finfo(np.float64).max))
    assert fraction_endpoint(maximum + 1, upper=False, saturate=True) == float(
        np.finfo(np.float64).max
    )
    assert fraction_endpoint(maximum + 1, upper=True, saturate=True) == np.inf
