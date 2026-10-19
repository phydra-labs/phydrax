#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Value-preserving binary64 lifts and directed rational execution views."""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import overload

import numpy as np

from ..linalg._exact import ExactField, solve_exact


type RationalVector = tuple[Fraction, ...]


@overload
def rational_vector(values: tuple[float, float], /) -> tuple[Fraction, Fraction]: ...


@overload
def rational_vector(
    values: tuple[float, float, float], /
) -> tuple[Fraction, Fraction, Fraction]: ...


@overload
def rational_vector(values: Iterable[float], /) -> RationalVector: ...


def rational_vector(values: Iterable[float], /) -> RationalVector:
    """Lift finite binary64 components without changing their represented values."""
    return tuple(Fraction(float(value)) for value in values)


def rational_dot(first: RationalVector, second: RationalVector, /) -> Fraction:
    """Return the Fraction dot product, requiring equal component counts."""
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def rational_cross[T: ExactField](
    first: tuple[T, ...], second: tuple[T, ...], /
) -> tuple[T, T, T]:
    """Return the three-dimensional cross product using coefficient-field arithmetic."""
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def rational_cross_2d[T: ExactField](first: tuple[T, T], second: tuple[T, T], /) -> T:
    """Return the signed planar cross product over coefficient-field coordinates."""
    return first[0] * second[1] - first[1] * second[0]


def fraction_down(value: Fraction, /) -> float:
    """Round a rational toward negative infinity in binary64."""
    result = float(value)
    return result if Fraction(result) <= value else float(np.nextafter(result, -np.inf))


def fraction_up(value: Fraction, /) -> float:
    """Round a rational toward positive infinity in binary64."""
    result = float(value)
    return result if Fraction(result) >= value else float(np.nextafter(result, np.inf))


def fraction_endpoint(
    value: Fraction, /, *, upper: bool, saturate: bool = False
) -> float:
    """Return a directed endpoint, optionally enclosing binary64 overflow."""
    if saturate:
        maximum = float(np.finfo(np.float64).max)
        if value > Fraction(maximum):
            return math.inf if upper else maximum
        if value < -Fraction(maximum):
            return -maximum if upper else -math.inf
    return fraction_up(value) if upper else fraction_down(value)


def exact_float(
    value: Fraction,
    /,
    *,
    error: type[Exception] = ValueError,
    message: str = "rational coefficient must have an exact finite binary64 representation",
) -> float:
    """Return an exactly represented binary64 value or raise the caller's error."""
    result = float(value)
    if not math.isfinite(result) or Fraction(result) != value:
        raise error(message)
    return result


def upper_float(value: Fraction, /) -> float:
    """Return a finite binary64 upper bound for a nonnegative rational."""
    if value < 0:
        raise ValueError(f"upper_float.value must be nonnegative; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"upper_float.value must convert to finite binary64; got {value!r} converting to {result!r}"
        )
    return float(np.nextafter(result, np.inf)) if Fraction(result) < value else result


def dyadic_integers(values: np.ndarray, /) -> tuple[np.ndarray, int]:
    """Integers ``n`` and one exponent ``e`` with ``values == n * 2**e`` exactly."""

    array = np.asarray(values, dtype=np.float64)
    mantissa, exponent = np.frexp(array)
    significand = np.ldexp(mantissa, 53).astype(np.int64)
    shift = exponent.astype(np.int64) - 53
    nonzero = significand != 0
    base = int(np.min(shift[nonzero])) if np.any(nonzero) else 0
    shift = np.where(nonzero, shift - base, 0)
    integers = np.asarray(
        [
            int(value) << int(amount)
            for value, amount in zip(
                significand.reshape(-1).tolist(), shift.reshape(-1).tolist(), strict=True
            )
        ],
        dtype=object,
    ).reshape(array.shape)
    return integers, base


def translation_ranges(
    boxes: list[tuple[np.ndarray, np.ndarray]], vectors: tuple[tuple[Fraction, ...], ...]
) -> tuple[range, ...]:
    if not vectors:
        return ()
    # Independent translation generators give a left inverse on their span.
    # Applying it to any difference T*n returns n, even for a partial lattice.
    gram = [
        [sum((a * b for a, b in zip(x, y, strict=True)), Fraction(0)) for y in vectors]
        for x in vectors
    ]
    inverse = solve_exact(
        gram,
        [[Fraction(i == j) for j in range(len(vectors))] for i in range(len(vectors))],
    )
    dual = tuple(
        tuple(
            sum((inverse[i][k] * vectors[k][j] for k in range(len(vectors))), Fraction(0))
            for j in range(len(vectors[0]))
        )
        for i in range(len(vectors))
    )
    ranges = []
    for row in dual:
        minima, maxima = [], []
        for lower, upper in boxes:
            minima.append(
                sum(
                    (
                        weight * Fraction(float(lo if weight >= 0 else hi))
                        for weight, lo, hi in zip(row, lower, upper, strict=True)
                    ),
                    Fraction(0),
                )
            )
            maxima.append(
                sum(
                    (
                        weight * Fraction(float(hi if weight >= 0 else lo))
                        for weight, lo, hi in zip(row, lower, upper, strict=True)
                    ),
                    Fraction(0),
                )
            )
        diameter = max(maxima) - min(minima)
        limit = diameter.numerator // diameter.denominator
        ranges.append(range(-limit, limit + 1))
    return tuple(ranges)
