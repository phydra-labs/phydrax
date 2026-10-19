#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host exact determinants and outward square-root enclosures."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...linalg._small_batched import exact_cofactor_determinant_3
from .._rational import fraction_down, fraction_up, rational_vector


def exact_determinant(matrix: np.ndarray, /) -> Fraction:
    """Return the determinant of the binary64 coefficients of a 3-by-3 matrix."""
    first, second, third = (rational_vector(row) for row in matrix)
    return rational_determinant(first, second, third)


def rational_determinant(
    first: tuple[Fraction, ...],
    second: tuple[Fraction, ...],
    third: tuple[Fraction, ...],
    /,
) -> Fraction:
    """Return the determinant of three rational rows."""
    if len(first) != 3 or len(second) != 3 or len(third) != 3:
        raise ValueError(
            "rational_determinant.rows must have three components; got "
            f"{(len(first), len(second), len(third))}."
        )
    return exact_cofactor_determinant_3(
        (first[0], first[1], first[2]),
        (second[0], second[1], second[2]),
        (third[0], third[1], third[2]),
    )


def square_root_interval(value: Fraction, /) -> tuple[float, float]:
    """Return outward binary64 bounds for the square root of a nonnegative rational."""
    if value == 0:
        return 0.0, 0.0
    lo, hi = float(np.sqrt(fraction_down(value))), float(np.sqrt(fraction_up(value)))
    if Fraction(lo) ** 2 > value:
        lo = float(np.nextafter(lo, -np.inf))
    if Fraction(hi) ** 2 < value:
        hi = float(np.nextafter(hi, np.inf))
    return lo, hi


def gram_deviation_bound(rotation: np.ndarray, /) -> float:
    """Enclose the spectral Gram defect by a rational absolute-row-sum bound."""
    axes = tuple(rational_vector(axis) for axis in rotation.T)
    defect = tuple(
        tuple(
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
            - (i == j)
            for j, second in enumerate(axes)
        )
        for i, first in enumerate(axes)
    )
    return fraction_up(
        max(sum((abs(value) for value in row), Fraction(0)) for row in defect)
    )
