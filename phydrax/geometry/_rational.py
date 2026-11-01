#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
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
    """Lift finite binary64 components without changing represented values."""
    represented = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in represented):
        raise ValueError("rational_vector.values must be finite binary64 components.")
    return tuple(Fraction(value) for value in represented)


def rational_dot(first: RationalVector, second: RationalVector, /) -> Fraction:
    """Return the Fraction dot product, requiring equal component counts."""
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def rational_cross[T: ExactField](
    first: tuple[T, ...], second: tuple[T, ...], /
) -> tuple[T, T, T]:
    """Return the three-dimensional cross product over the coefficient field."""
    if len(first) != 3 or len(second) != 3:
        raise ValueError("rational_cross requires two three-dimensional vectors.")
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def rational_cross_2d[T: ExactField](first: tuple[T, T], second: tuple[T, T], /) -> T:
    """Return the signed planar cross product over coefficient-field values."""
    return first[0] * second[1] - first[1] * second[0]


_BINARY64_MAXIMUM = float(np.finfo(np.float64).max)
_RATIONAL_BINARY64_MAXIMUM = Fraction(_BINARY64_MAXIMUM)


def _float_overflow_endpoint(value: Fraction, /, *, upper: bool) -> float:
    if value > 0:
        return math.inf if upper else _BINARY64_MAXIMUM
    return -_BINARY64_MAXIMUM if upper else -math.inf


def _rounding_sign(result: float, value: Fraction, /) -> int:
    """Sign of ``result - value``; rationals use one integer cross-multiplication."""
    if type(value) is not Fraction and type(value) is not int:
        represented = Fraction(result)
        return (represented > value) - (represented < value)
    numerator, denominator = result.as_integer_ratio()
    left = numerator * value.denominator
    right = value.numerator * denominator
    return (left > right) - (left < right)


def fraction_down(value: Fraction, /) -> float:
    """Round a rational toward negative infinity in binary64."""
    try:
        result = float(value)
    except OverflowError:
        return _float_overflow_endpoint(value, upper=False)
    return (
        result
        if _rounding_sign(result, value) <= 0
        else math.nextafter(result, -math.inf)
    )


def fraction_up(value: Fraction, /) -> float:
    """Round a rational toward positive infinity in binary64."""
    try:
        result = float(value)
    except OverflowError:
        return _float_overflow_endpoint(value, upper=True)
    return (
        result if _rounding_sign(result, value) >= 0 else math.nextafter(result, math.inf)
    )


def fraction_endpoint(
    value: Fraction, /, *, upper: bool, saturate: bool = False
) -> float:
    """Return a directed binary64 endpoint, optionally saturating overflow."""
    if saturate:
        if value > _RATIONAL_BINARY64_MAXIMUM:
            return math.inf if upper else _BINARY64_MAXIMUM
        if value < -_RATIONAL_BINARY64_MAXIMUM:
            return -_BINARY64_MAXIMUM if upper else -math.inf
    return fraction_up(value) if upper else fraction_down(value)


def exact_float(
    value: Fraction,
    /,
    *,
    error: type[Exception] = ValueError,
    message: str = "rational coefficient must have an exact finite binary64 representation",
) -> float:
    """Return an exactly represented binary64 value or raise the caller's error."""
    try:
        result = float(value)
    except OverflowError as failure:
        raise error(message) from failure
    if not math.isfinite(result) or _rounding_sign(result, value):
        raise error(message)
    return result


def upper_float(value: Fraction, /) -> float:
    """Return a finite binary64 upper bound for a nonnegative rational."""
    if value < 0:
        raise ValueError(f"upper_float.value must be nonnegative; got {value!r}")
    try:
        result = float(value)
    except OverflowError as failure:
        raise ValueError(
            f"upper_float.value must convert to finite binary64; got {value!r}"
        ) from failure
    if not math.isfinite(result):
        raise ValueError(
            f"upper_float.value must convert to finite binary64; got {value!r}"
        )
    return (
        math.nextafter(result, math.inf) if _rounding_sign(result, value) < 0 else result
    )


def dyadic_integers(values: np.ndarray, /) -> tuple[np.ndarray, int]:
    """Return integers ``n`` and one exponent ``e`` with ``values == n*2**e``."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("dyadic_integers.values must be finite binary64 values.")
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
    boxes: list[tuple[np.ndarray, np.ndarray]],
    vectors: tuple[tuple[Fraction, ...], ...],
) -> tuple[range, ...]:
    """Bound integer translations intersecting any supplied axis-aligned box."""
    if not vectors:
        return ()
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors) or any(
        lower.shape != (dimension,)
        or upper.shape != (dimension,)
        or np.any(lower > upper)
        for lower, upper in boxes
    ):
        raise ValueError("translation_ranges requires aligned source vectors and boxes.")

    if not boxes:
        return tuple(range(0) for _ in vectors)
    gram = [
        [sum((a * b for a, b in zip(x, y, strict=True)), Fraction(0)) for y in vectors]
        for x in vectors
    ]
    inverse = solve_exact(
        gram,
        [[Fraction(i == j) for j in range(len(vectors))] for i in range(len(vectors))],
        singular_message="translation_ranges.vectors must be linearly independent.",
    )
    dual = tuple(
        tuple(
            sum((inverse[i][k] * vectors[k][j] for k in range(len(vectors))), Fraction(0))
            for j in range(dimension)
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


__all__ = [
    "RationalVector",
    "dyadic_integers",
    "exact_float",
    "fraction_down",
    "fraction_endpoint",
    "fraction_up",
    "rational_cross",
    "rational_cross_2d",
    "rational_dot",
    "rational_vector",
    "translation_ranges",
    "upper_float",
]
