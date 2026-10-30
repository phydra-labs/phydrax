#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact represented-coordinate CAD predicates and directed algebraic bounds."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from jax.typing import ArrayLike

from .._rational import fraction_down, fraction_up, rational_vector


def rational_determinant(
    first: tuple[Fraction, ...],
    second: tuple[Fraction, ...],
    third: tuple[Fraction, ...],
    /,
) -> Fraction:
    if len(first) != 3 or len(second) != 3 or len(third) != 3:
        raise ValueError("rational_determinant requires three 3D rows.")
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def exact_determinant(matrix: np.ndarray, /) -> Fraction:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError("exact_determinant requires a finite binary64 3x3 matrix.")
    first, second, third = (rational_vector(row) for row in values)
    return rational_determinant(first, second, third)


def square_root_interval(value: Fraction, /) -> tuple[float, float]:
    """Outward binary64 enclosure of the exact square root of a rational."""
    if not isinstance(value, Fraction):
        raise TypeError("square_root_interval requires an exact Fraction.")
    if value < 0:
        raise ValueError("square_root_interval requires a nonnegative rational.")
    if value == 0:
        return 0.0, 0.0
    lower = float(np.sqrt(fraction_down(value)))
    upper = float(np.sqrt(fraction_up(value)))
    if Fraction.from_float(lower) ** 2 > value:
        lower = float(np.nextafter(lower, -np.inf))
    if Fraction.from_float(upper) ** 2 < value:
        upper = float(np.nextafter(upper, np.inf))
    return lower, upper


def gram_deviation_bound(rotation: np.ndarray, /) -> float:
    values = np.asarray(rotation, dtype=np.float64)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError("gram_deviation_bound requires a finite 3x3 matrix.")
    columns = tuple(rational_vector(axis) for axis in values.T)
    defect = tuple(
        tuple(
            sum(
                (left * right for left, right in zip(first, second, strict=True)),
                Fraction(0),
            )
            - Fraction(int(i == j))
            for j, second in enumerate(columns)
        )
        for i, first in enumerate(columns)
    )
    return fraction_up(
        max(sum((abs(value) for value in row), Fraction(0)) for row in defect)
    )


def exact_frame_orientation(
    first_axis: ArrayLike, second_axis: ArrayLike, axis: ArrayLike, /
) -> int | None:
    """Return the handedness of an exactly orthonormal represented 3D frame."""
    vectors = tuple(
        np.asarray(value, dtype=np.float64) for value in (first_axis, second_axis, axis)
    )
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in vectors):
        raise ValueError("Frame axes must be finite three-vectors.")
    frame = np.column_stack(vectors)
    if gram_deviation_bound(frame) != 0.0:
        return None
    return 1 if exact_determinant(frame) > 0 else -1


__all__ = [
    "exact_determinant",
    "exact_frame_orientation",
    "fraction_norm_upper",
    "gram_deviation_bound",
    "radical_sign",
    "rational_determinant",
    "square_root_interval",
]


def radical_sign(a: Fraction, b: Fraction, radicand: Fraction, /) -> int:
    """Return the exact sign of a plus b times square-root of radicand."""
    if any(not isinstance(value, Fraction) for value in (a, b, radicand)):
        raise TypeError("radical_sign coefficients must be Fractions.")
    if radicand < 0:
        raise ValueError("radical_sign requires a nonnegative radicand.")
    if b == 0 or radicand == 0:
        return (a > 0) - (a < 0)
    if a == 0:
        return (b > 0) - (b < 0)
    if (a > 0) == (b > 0):
        return 1 if a > 0 else -1
    comparison = a * a - b * b * radicand
    if comparison == 0:
        return 0
    if a > 0:
        return 1 if comparison > 0 else -1
    return 1 if comparison < 0 else -1


def fraction_norm_upper(components: tuple[Fraction, ...], /) -> float:
    """Outward upper bound for the Euclidean norm of exact rational components."""
    if not isinstance(components, tuple) or any(
        not isinstance(value, Fraction) for value in components
    ):
        raise TypeError("fraction_norm_upper requires a tuple of Fractions.")
    squared = sum((value * value for value in components), Fraction(0))
    return square_root_interval(squared)[1]
