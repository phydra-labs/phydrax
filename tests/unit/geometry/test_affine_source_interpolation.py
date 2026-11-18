"""Exact affine interpolation residuals and the general polynomial route."""

import math
from fractions import Fraction

import pytest

from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry._source_interpolation import (
    _exact_surface_interpolation_bound,
    _exact_surface_map_is_rational_affine,
)


@pytest.mark.parametrize(
    "chart",
    [
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
        (
            (Fraction(1, 3), Fraction(-2, 7)),
            (Fraction(8, 9), Fraction(1, 5)),
            (Fraction(-7, 11), Fraction(5, 13)),
        ),
        ((Fraction(1, 3), Fraction(1, 5)),) * 3,
    ],
)
def test_rational_affine_residual_is_positive_zero(
    chart: tuple[tuple[Fraction, Fraction], ...],
) -> None:
    expressions: tuple[Polynomial, ...] = (
        {(0, 0): Fraction(1, 3), (1, 0): Fraction(7, 11)},
        {(0, 1): Fraction(-2, 9)},
        {(0, 0): Fraction(4, 7), (2, 0): Fraction(0)},
    )
    assert _exact_surface_map_is_rational_affine(expressions)
    bound = _exact_surface_interpolation_bound(expressions, chart)
    assert bound == 0.0
    assert math.copysign(1.0, bound) == 1.0


def test_non_affine_and_radical_maps_keep_general_route() -> None:
    chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    quadratic: tuple[Polynomial, ...] = ({(2, 0): Fraction(1)}, {}, {})
    radical: tuple[Polynomial, ...] = ({(1, 0): RadicalElement(1)}, {}, {})
    assert not _exact_surface_map_is_rational_affine(quadratic)
    assert _exact_surface_interpolation_bound(quadratic, chart) == 0.25
    assert not _exact_surface_map_is_rational_affine(radical)
    assert _exact_surface_interpolation_bound(radical, chart) == 0.0
