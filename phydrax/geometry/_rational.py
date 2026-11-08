#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import overload

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

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


def binary64_difference_magnitude(
    first: ArrayLike, second: ArrayLike, /
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Directed binary64 bounds of ``|first - second|``, elementwise.

    Under IEEE 754 round-to-nearest with gradual underflow a difference rounds to
    zero only when it is exactly zero; otherwise the exact difference lies
    between the neighbors of its rounding. ``abs`` is exact, so stepping the
    rounded magnitude once down and once up encloses the exact magnitude.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.abs(
            np.subtract(
                np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
            )
        )
    exact = magnitude == 0.0
    return (
        np.where(exact, 0.0, np.nextafter(magnitude, -np.inf)),
        np.where(exact, 0.0, np.nextafter(magnitude, np.inf)),
    )


def _directed_norm(components: ArrayLike, upward: bool, /) -> NDArray[np.float64]:
    values = np.abs(np.asarray(components, dtype=np.float64))
    if values.ndim == 0:
        raise ValueError("binary64 norm bounds require a component axis.")
    direction = np.inf if upward else -np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        squares = np.where(
            values == 0.0,
            0.0,
            np.maximum(np.nextafter(values * values, direction), 0.0),
        )
        total = np.zeros(values.shape[:-1], dtype=np.float64)
        for axis in range(values.shape[-1]):
            term = squares[..., axis]
            rounded = total + term
            total = np.where(
                (total == 0.0) | (term == 0.0),
                rounded,
                np.nextafter(rounded, direction),
            )
        return np.where(total == 0.0, 0.0, np.nextafter(np.sqrt(total), direction))


def binary64_norm_upper(components: ArrayLike, /) -> NDArray[np.float64]:
    """Upper bound of the Euclidean norm along the last axis.

    Every square, partial sum, and the square root is one correctly rounded IEEE
    754 operation followed by one step toward ``+inf`` unless an operand is an
    exact zero, so the bound holds on every IEEE platform without relying on a
    platform ``hypot``. Squares beyond the binary64 range give ``inf``.
    """
    return _directed_norm(components, True)


def binary64_norm_lower(components: ArrayLike, /) -> NDArray[np.float64]:
    """Lower bound of the Euclidean norm along the last axis.

    The operations of ``binary64_norm_upper`` each step toward ``-inf`` instead;
    squares that underflow contribute zero.
    """
    return _directed_norm(components, False)


def binary64_dyadic_entries(
    values: np.ndarray, /
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return per-entry odd significands ``s`` and exponents ``e`` with ``value == s*2**e``.

    Zero (including ``-0.0``) is ``(0, 0)``; every other significand is odd, so the
    pair is the canonical integer-dyadic form of each finite binary64 entry.
    """
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("binary64_dyadic_entries.values must be finite binary64 values.")
    mantissa, exponent = np.frexp(array)
    significand = np.ldexp(mantissa, 53).astype(np.int64)
    nonzero = significand != 0
    # The lowest set bit is a power of two below 2**53, so binary64 holds it exactly.
    lowest = np.where(nonzero, significand & -significand, 1)
    trailing = np.frexp(lowest.astype(np.float64))[1].astype(np.int64) - 1
    significands = np.where(nonzero, significand >> trailing, 0)
    exponents = np.where(nonzero, exponent.astype(np.int64) - 53 + trailing, 0)
    return significands, exponents


def dyadic_entry(value: Fraction, /) -> tuple[int, int] | None:
    """Canonical ``(odd significand, exponent)`` of a dyadic rational, else ``None``."""
    numerator, denominator = value.numerator, value.denominator
    if denominator & (denominator - 1):
        return None
    if numerator == 0:
        return 0, 0
    if denominator > 1:
        return numerator, 1 - denominator.bit_length()
    trailing = (numerator & -numerator).bit_length() - 1
    return numerator >> trailing, trailing


def dyadic_integers(values: np.ndarray, /) -> tuple[np.ndarray, int]:
    """Return integers ``n`` and one exponent ``e`` with ``values == n*2**e``.

    ``e`` is the least exponent of the 53-bit binary64 significands of the
    nonzero entries (``0`` when every entry is zero).
    """
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("dyadic_integers.values must be finite binary64 values.")
    significands, exponents = binary64_dyadic_entries(array)
    nonzero = significands != 0
    widths = np.frexp(np.abs(significands).astype(np.float64))[1].astype(np.int64)
    base = int(np.min((exponents + widths - 53)[nonzero])) if np.any(nonzero) else 0
    integers = np.asarray(
        [
            int(value) << int(amount - base) if value else 0
            for value, amount in zip(
                significands.reshape(-1).tolist(),
                exponents.reshape(-1).tolist(),
                strict=True,
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
    "binary64_difference_magnitude",
    "binary64_dyadic_entries",
    "binary64_norm_lower",
    "binary64_norm_upper",
    "dyadic_entry",
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
