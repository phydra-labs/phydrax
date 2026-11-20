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
_RATIONAL_BINARY64_OVERFLOW_THRESHOLD = _RATIONAL_BINARY64_MAXIMUM + Fraction(2**970)


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


def binary64_dot(first: ArrayLike, second: ArrayLike, /) -> NDArray[np.float64]:
    """Dot broadcast batches of last-axis vectors in binary64 round-to-nearest.

    Multiply each pair separately and add the products from index zero upward,
    rounding every multiplication and addition separately (no fused multiply-add).
    An empty contraction is positive zero; otherwise the first product initializes
    the sum, preserving its signed zero. NaN, infinity, overflow, and underflow
    follow IEEE elementwise multiplication and addition.
    Scalar paths with contraction length two or three do not report underflow;
    overflow and invalid operations are reported as by the array route.
    """
    if (
        type(first) is np.ndarray
        and type(second) is np.ndarray
        and first.dtype == np.float64
        and second.dtype == np.float64
        and first.ndim == second.ndim == 1
        and first.shape == second.shape
        and first.size in (2, 3)
    ):
        a = first.tolist()
        b = a if first is second else second.tolist()
        scalar = a[0] * b[0]
        scalar = scalar + a[1] * b[1]
        if len(a) == 3:
            scalar = scalar + a[2] * b[2]
        if math.isfinite(scalar):
            return np.asarray(scalar, dtype=np.float64)
    if np.iscomplexobj(first) or np.iscomplexobj(second):
        raise TypeError("Complex operands are not supported.")
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim == 0 or b.ndim == 0 or a.shape[-1] != b.shape[-1]:
        raise ValueError("Vector product dimensions must agree.")
    shape = (
        a.shape[:-1]
        if a.shape[:-1] == b.shape[:-1]
        else np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    )
    if a.shape[-1] == 0:
        return np.zeros(shape, dtype=np.float64)
    result = np.multiply(a[..., 0], b[..., 0])
    for index in range(1, a.shape[-1]):
        product = np.multiply(a[..., index], b[..., index])
        result = np.add(result, product)
    return np.asarray(result, dtype=np.float64)


def binary64_norm(values: ArrayLike, /) -> NDArray[np.float64]:
    """Return the correctly rounded square root of the ordered sum of squares.

    Squares and additions follow :func:`binary64_dot`. No rescaling is performed:
    squares may overflow or underflow. Empty vectors and zero squares yield
    positive zero; NaNs and infinities propagate through the specified operations.
    Scalar paths with contraction length two or three do not report underflow;
    overflow and invalid operations are reported as by the array route.
    """
    if (
        type(values) is np.ndarray
        and values.dtype == np.float64
        and values.ndim == 1
        and values.size in (2, 3)
    ):
        return np.asarray(np.sqrt(binary64_dot(values, values)), dtype=np.float64)
    if np.iscomplexobj(values):
        raise TypeError("Complex operands are not supported.")
    a = np.asarray(values, dtype=np.float64)
    return np.asarray(np.sqrt(binary64_dot(a, a)), dtype=np.float64)


def binary64_matmul(first: ArrayLike, second: ArrayLike, /) -> NDArray[np.float64]:
    """Multiply vectors or broadcast matrix batches with NumPy matmul shapes.

    Each binary64 product is rounded separately, then added left-to-right with
    a separate rounding at each addition; fused multiply-add is not used. Empty
    contractions give positive zero. Otherwise the first product initializes the
    sum, so signed zeros, NaNs, infinities, overflow and underflow follow the
    specified IEEE multiplication and addition sequence.
    Scalar paths with contraction length two or three do not report underflow;
    overflow and invalid operations are reported as by the array route.
    """
    if (
        type(first) is np.ndarray
        and type(second) is np.ndarray
        and first.dtype == np.float64
        and second.dtype == np.float64
        and first.ndim == 2
        and second.ndim == 1
        and first.shape[0] in (2, 3)
        and first.shape[1] == second.shape[0] == 3
    ):
        rows = first.tolist()
        vector = second.tolist()
        values = [
            (row[0] * vector[0] + row[1] * vector[1]) + row[2] * vector[2] for row in rows
        ]
        if all(math.isfinite(value) for value in values):
            return np.asarray(values, dtype=np.float64)
    if (
        type(first) is np.ndarray
        and type(second) is np.ndarray
        and first.dtype == np.float64
        and second.dtype == np.float64
        and second.ndim == 2
        and second.shape in ((3, 2), (3, 3))
        and (
            (first.ndim == 1 and first.shape == (3,))
            or (first.ndim == 2 and first.shape in ((2, 3), (3, 3)))
        )
    ):
        left = first.tolist()
        right = second.tolist()
        if first.ndim == 1:
            row = left
            values = [
                (row[0] * right[0][column] + row[1] * right[1][column])
                + row[2] * right[2][column]
                for column in range(second.shape[1])
            ]
            if all(math.isfinite(value) for value in values):
                return np.asarray(values, dtype=np.float64)
        else:
            matrix = [
                [
                    (row[0] * right[0][column] + row[1] * right[1][column])
                    + row[2] * right[2][column]
                    for column in range(second.shape[1])
                ]
                for row in left
            ]
            if all(math.isfinite(value) for row in matrix for value in row):
                return np.asarray(matrix, dtype=np.float64)
    if (
        type(first) is np.ndarray
        and type(second) is np.ndarray
        and first.dtype == np.float64
        and second.dtype == np.float64
    ):
        a, b = first, second
    else:
        if np.iscomplexobj(first) or np.iscomplexobj(second):
            raise TypeError("Complex operands are not supported.")
        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("Matrix product operands must have at least one dimension.")
    left_vector = a.ndim == 1
    right_vector = b.ndim == 1
    if left_vector:
        a = a[np.newaxis, :]
    if right_vector:
        b = b[:, np.newaxis]
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("Matrix product dimensions must agree.")
    batch_shape = (
        a.shape[:-2]
        if a.shape[:-2] == b.shape[:-2]
        else np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    )
    shape = batch_shape + (a.shape[-2], b.shape[-1])
    if a.shape[-1] == 0:
        result = np.zeros(shape, dtype=np.float64)
    elif math.prod(batch_shape) >= _COMPONENT_BATCH:
        result = _component_matmul(a, b, batch_shape)
    else:
        result = np.multiply(a[..., :, 0, None], b[..., None, 0, :])
        for index in range(1, a.shape[-1]):
            product = np.multiply(a[..., :, index, None], b[..., None, index, :])
            result = np.add(result, product)
    if left_vector:
        result = np.squeeze(result, axis=-2)
    if right_vector:
        result = np.squeeze(result, axis=-1)
    return np.asarray(result, dtype=np.float64)


# Batch size from which matrix entries are evaluated as contiguous batch arrays.
_COMPONENT_BATCH = 512


def _component_matmul(
    a: np.ndarray, b: np.ndarray, batch_shape: tuple[int, ...], /
) -> NDArray[np.float64]:
    """Evaluate :func:`binary64_matmul` entry by entry over contiguous batches.

    The products and left-to-right additions are those of the broadcast route;
    only the memory layout differs, so every entry is bitwise identical. Small
    matrix axes innermost would make every NumPy loop a few elements long.
    """
    rows, inner, columns = a.shape[-2], a.shape[-1], b.shape[-1]
    rank = len(batch_shape)
    left = np.ascontiguousarray(
        np.broadcast_to(
            np.moveaxis(
                a.reshape((1,) * (rank + 2 - a.ndim) + a.shape), (-2, -1), (0, 1)
            ),
            (rows, inner, *batch_shape),
        )
    )
    right = np.ascontiguousarray(
        np.broadcast_to(
            np.moveaxis(
                b.reshape((1,) * (rank + 2 - b.ndim) + b.shape), (-2, -1), (0, 1)
            ),
            (inner, columns, *batch_shape),
        )
    )
    result = np.empty((rows, columns, *batch_shape), dtype=np.float64)
    product = np.empty(batch_shape, dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            entry = result[row, column]
            np.multiply(left[row, 0], right[0, column], out=entry)
            for index in range(1, inner):
                np.multiply(left[row, index], right[index, column], out=product)
                np.add(entry, product, out=entry)
    return np.moveaxis(result, (0, 1), (-2, -1))


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


def _upward(value: NDArray[np.float64], /) -> NDArray[np.float64]:
    """One step toward ``+inf`` after a rounded operation, unless exactly zero."""
    return np.where(value == 0.0, 0.0, np.nextafter(value, np.inf))


def binary64_first_order_interpolation_upper(
    spread: ArrayLike, vertex_lower: ArrayLike, vertex_upper: ArrayLike, /
) -> float:
    """Upper bound of ``|f - I f|`` on a simplex from the width of a Jacobian enclosure.

    The ``n + 1`` parameter vertices of an ``n``-simplex lie in the boxes
    ``[vertex_lower[j], vertex_upper[j]]``, ``I f`` is the affine interpolant
    of ``f`` at them, and ``spread[i, k]`` bounds the width of an enclosure of
    ``d f_i / d x_k`` over the simplex. ``f`` must be continuous and piecewise
    smooth there: an interval program of its derivative, evaluated over the
    simplex, then encloses the derivative along every segment except at
    finitely many points, so the bound needs no second derivative and holds
    across kinks.

    From a vertex ``v_j``, ``p = v_j + sum_k l_k (v_k - v_j)`` gives
    ``f_i(p) - I f_i(p) = sum_k l_k (G - G_k) . (v_k - v_j)``, where ``G`` and
    ``G_k`` are averages of ``grad f_i`` along segments and so lie in its
    enclosure. Each term is at most ``spread[i] . |v_k - v_j|``, and with the
    largest barycentric weight ``l_j >= 1 / (n + 1)`` every component is at
    most ``n / (n + 1)`` times its largest edge term. Every rounded operation
    steps toward ``+inf``; a nonfinite spread gives ``inf``.
    """
    spread_ = np.asarray(spread, dtype=np.float64)
    lower_ = np.asarray(vertex_lower, dtype=np.float64)
    upper_ = np.asarray(vertex_upper, dtype=np.float64)
    if (
        lower_.ndim != 2
        or lower_.shape[0] != lower_.shape[1] + 1
        or upper_.shape != lower_.shape
        or spread_.ndim != 2
        or spread_.shape[1] != lower_.shape[1]
    ):
        raise ValueError(
            "first-order interpolation bounds need (n + 1, n) vertex boxes and an (m, n) spread."
        )
    if (
        not np.all(np.isfinite(lower_))
        or not np.all(np.isfinite(upper_))
        or np.any(lower_ > upper_)
    ):
        raise ValueError(
            "first-order interpolation vertex boxes must be finite and ordered."
        )
    if not np.all(spread_ >= 0.0):
        return math.inf
    dimension = lower_.shape[1]
    terms = np.zeros((spread_.shape[0],), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for first in range(dimension + 1):
            for second in range(first + 1, dimension + 1):
                edge = np.maximum(
                    binary64_difference_magnitude(upper_[second], lower_[first])[1],
                    binary64_difference_magnitude(lower_[second], upper_[first])[1],
                )
                term = np.zeros_like(terms)
                for axis in range(dimension):
                    term = _upward(term + _upward(spread_[:, axis] * edge[axis]))
                terms = np.maximum(terms, term)
        ratio = dimension / (dimension + 1)
        if Fraction(ratio) != Fraction(dimension, dimension + 1):
            ratio = math.nextafter(ratio, math.inf)
        bound = _upward(binary64_norm_upper(terms) * ratio)
    return float(bound) if np.isfinite(bound) else math.inf


def binary64_spectral_upper(matrix: ArrayLike, /) -> float:
    """Certified spectral norm bound for a finite binary64 matrix.

    The Frobenius norm and ``sqrt(||A||_1 * ||A||_inf)`` are both upper
    bounds. Every nontrivial addition, multiplication and square root in the
    latter is directed upward.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("spectral norm bound requires a finite matrix.")
    frobenius = float(binary64_norm_upper(values.reshape(-1)))
    rows, columns = values.shape
    row_norm = 0.0
    column_norm = 0.0
    for row in range(rows):
        total = 0.0
        for column in range(columns):
            term = abs(float(values[row, column]))
            total = (
                math.nextafter(total + term, math.inf) if total and term else total + term
            )
        row_norm = max(row_norm, total)
    for column in range(columns):
        total = 0.0
        for row in range(rows):
            term = abs(float(values[row, column]))
            total = (
                math.nextafter(total + term, math.inf) if total and term else total + term
            )
        column_norm = max(column_norm, total)
    if row_norm == 0.0 or column_norm == 0.0:
        return 0.0
    product = math.nextafter(row_norm * column_norm, math.inf)
    induced = math.nextafter(math.sqrt(product), math.inf)
    return min(frobenius, induced)


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


def binary64_sum(values: ArrayLike, /, *, axis: int) -> NDArray[np.float64]:
    """Sum one axis from low to high index with separate binary64 additions.

    The first element initializes the sum; an empty axis gives positive zero.
    NaNs, infinities, overflow and underflow follow the elementwise additions.
    """
    if np.iscomplexobj(values):
        raise TypeError("Complex operands are not supported.")
    array = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    if array.shape[-1] == 0:
        return np.zeros(array.shape[:-1], dtype=np.float64)
    result = np.array(array[..., 0], dtype=np.float64, copy=True)
    for index in range(1, array.shape[-1]):
        np.add(result, array[..., index], out=result)
    return result


def binary64_least_squares(matrix: ArrayLike, right: ArrayLike, /) -> NDArray[np.float64]:
    """Solve a small binary64 least-squares problem over the exact rationals.

    Inputs are lifted exactly. The normal equations are solved over the
    rationals and each component is rounded once to binary64, ties to even.
    A solution whose correctly rounded component is nonfinite raises ValueError;
    values above the largest finite binary64 are accepted when they round to it.
    When the matrix is rank deficient, the unique minimum-Euclidean-norm
    least-squares solution is selected exactly before rounding.
    """
    from ..linalg._exact import factor_exact_sparse

    if np.iscomplexobj(matrix) or np.iscomplexobj(right):
        raise TypeError("Complex operands are not supported.")
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.shape[0] or a.shape[1] == 0:
        raise ValueError("Expected a matrix and a matching nonempty right-hand side.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Least-squares inputs must be finite.")
    rows = [rational_vector(row) for row in a]
    rhs = rational_vector(b)
    width = a.shape[1]
    columns = tuple(tuple(row[index] for row in rows) for index in range(width))
    gram = [[rational_dot(first, second) for second in columns] for first in columns]
    target = [[rational_dot(column, rhs)] for column in columns]
    factor = factor_exact_sparse(
        [{index: value for index, value in enumerate(row) if value} for row in gram],
        width,
        ordered=True,
    )
    nullspace = factor.nullspace_basis()
    if nullspace:
        augmented = [
            row + [vector[index] for vector in nullspace]
            for index, row in enumerate(gram)
        ]
        augmented.extend(
            [list(vector) + [Fraction(0)] * len(nullspace) for vector in nullspace]
        )
        target.extend([[Fraction(0)] for _ in nullspace])
        solution = solve_exact(augmented, target)
    else:
        solution = solve_exact(gram, target)
    if any(
        abs(solution[index][0]) >= _RATIONAL_BINARY64_OVERFLOW_THRESHOLD
        for index in range(width)
    ):
        raise ValueError("Least-squares solution is not finite in binary64.")
    return np.asarray(
        [float(solution[index][0]) for index in range(width)], dtype=np.float64
    )


__all__ = [
    "RationalVector",
    "binary64_difference_magnitude",
    "binary64_dyadic_entries",
    "binary64_dot",
    "binary64_least_squares",
    "binary64_matmul",
    "binary64_norm",
    "binary64_norm_lower",
    "binary64_norm_upper",
    "binary64_spectral_upper",
    "binary64_sum",
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

# These constants and evaluation order mirror meshcore's angle.cpp.  In
# particular, keep each multiply and add separate rather than using a ufunc
# polynomial evaluator that could reassociate or fuse operations.
_ATAN_COEFFICIENTS = tuple(
    float.fromhex(value)
    for value in (
        "0x1.5555555555555p-2",
        "-0x1.999999999999ap-3",
        "0x1.2492492492492p-3",
        "-0x1.c71c71c71c71cp-4",
        "0x1.745d1745d1746p-4",
        "-0x1.3b13b13b13b14p-4",
        "0x1.1111111111111p-4",
        "-0x1.e1e1e1e1e1e1ep-5",
        "0x1.af286bca1af28p-5",
        "-0x1.8618618618618p-5",
        "0x1.642c8590b2164p-5",
        "-0x1.47ae147ae147bp-5",
        "0x1.2f684bda12f68p-5",
        "-0x1.1a7b9611a7b96p-5",
    )
)
_ANGLE_PI = float.fromhex("0x1.921fb54442d18p+1")
_ANGLE_HALF_PI = float.fromhex("0x1.921fb54442d18p+0")
_ANGLE_PI_LOW = float.fromhex("0x1.1a62633145c07p-53")
_ANGLE_HALF_PI_LOW = float.fromhex("0x1.1a62633145c07p-54")


def _binary64_arctangent_unit(u: NDArray[np.float64]) -> NDArray[np.float64]:
    high = u >= 0.75
    middle = (u >= 0.25) & ~high
    v = np.where(
        high, (u - 1.0) / (1.0 + u), np.where(middle, (u - 0.5) / (1.0 + 0.5 * u), u)
    )
    base = np.where(
        high,
        float.fromhex("0x1.921fb54442d18p-1"),
        np.where(middle, float.fromhex("0x1.dac670561bb4fp-2"), 0.0),
    )
    low = np.where(
        high,
        float.fromhex("0x1.1a62633145c07p-55"),
        np.where(middle, float.fromhex("0x1.a2b7f222f65e2p-56"), 0.0),
    )
    q = v * v
    total = np.full_like(q, _ATAN_COEFFICIENTS[-1])
    for coefficient in reversed(_ATAN_COEFFICIENTS[:-1]):
        total = total * q + coefficient
    return base + (v - (v * (q * total) - low))


def _binary64_arctangent(t: NDArray[np.float64]) -> NDArray[np.float64]:
    unit = t <= 1.0
    argument = np.where(unit, t, 1.0 / t)
    value = _binary64_arctangent_unit(argument)
    return np.where(unit, value, _ANGLE_HALF_PI - (value - _ANGLE_HALF_PI_LOW))


def binary64_angle_between(across: ArrayLike, along: ArrayLike, /) -> NDArray[np.float64]:
    """Evaluate meshcore's unsigned angle kernel elementwise in binary64."""
    a, b = np.broadcast_arrays(
        np.asarray(across, dtype=np.float64), np.asarray(along, dtype=np.float64)
    )
    with np.errstate(all="ignore"):
        angle = _binary64_arctangent(a / np.where(b < 0.0, -b, b))
        result = np.where(
            b > 0.0,
            angle,
            np.where(
                b < 0.0,
                _ANGLE_PI - (angle - _ANGLE_PI_LOW),
                np.where(a > 0.0, _ANGLE_HALF_PI, 0.0),
            ),
        )
        return np.asarray(
            np.where((a >= 0.0) & np.isfinite(a) & np.isfinite(b), result, np.nan),
            dtype=np.float64,
        )


def binary64_arctangent2(
    ordinate: ArrayLike, abscissa: ArrayLike, /
) -> NDArray[np.float64]:
    """Evaluate meshcore's signed angle kernel, including IEEE signed zeros."""
    y, x = np.broadcast_arrays(
        np.asarray(ordinate, dtype=np.float64), np.asarray(abscissa, dtype=np.float64)
    )
    angle = binary64_angle_between(np.abs(y), x)
    angle = np.where(
        (y == 0.0) & (x == 0.0), np.where(np.signbit(x), _ANGLE_PI, 0.0), angle
    )
    return np.asarray(
        np.where(np.isfinite(y) & np.isfinite(x), np.copysign(angle, y), np.nan),
        dtype=np.float64,
    )


def binary64_cube_root(values: ArrayLike, /) -> NDArray[np.float64]:
    """Evaluate plc3d's eight-step binary64 Newton cube root elementwise.

    Nonpositive and nonfinite inputs are returned unchanged, including NaN bits.
    """
    value = np.asarray(values, dtype=np.float64)
    valid = (value > 0.0) & np.isfinite(value)
    _, exponent = np.frexp(np.where(valid, value, 1.0))
    # C++ integer division truncates toward zero, unlike Python floor division.
    exponent = np.where(exponent < 0, -((-exponent) // 3), exponent // 3)
    root = np.ldexp(np.ones_like(value), exponent)
    with np.errstate(all="ignore"):
        for _ in range(8):
            root = (2.0 * root + value / (root * root)) / 3.0
    return np.asarray(np.where(valid, root, value), dtype=np.float64)


__all__ += ["binary64_angle_between", "binary64_arctangent2", "binary64_cube_root"]
