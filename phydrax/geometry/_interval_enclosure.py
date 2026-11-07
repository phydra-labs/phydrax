#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Outward-rounded interval evaluation for pure real JAX maps."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import reduce
from typing import Any, cast, Literal, TYPE_CHECKING

import jax
import numpy as np
from jax import Array
from jax._src.numpy.hijax import SearchSorted
from jax.core import eval_jaxpr, ShapedArray
from jax.extend import core as jax_core
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._exact_kernels import (
    ElementaryFunction,
    float64_interval_elementary,
    float64_interval_product,
    float64_interval_sum,
)
from .._interpolation._bspline import bspline_jet_rule_alignment
from .._interpolation._bspline_extraction import (
    _bernstein_derivative,
    _binary64_down,
    _binary64_up,
    _univariate_span_extraction,
)
from .._jax_program import custom_jvp_rule_owner
from .._meshcore import meshcore_available
from ..typing import parse


if TYPE_CHECKING:
    from ._interval_native import NativeIntervalProgram


type HostArray = np.ndarray | np.generic
type Interval = tuple[HostArray, HostArray]
type IntervalExecution = Literal["host", "native"]


class IntervalExtensionError(ValueError):
    """A traced map contains an operation without a sound real interval extension."""


def _as_interval(value: object, /) -> Interval:
    array = np.asarray(value, dtype=np.float64)
    return array, array


def _round_out(lower: HostArray, upper: HostArray, /) -> Interval:
    low, high = np.asarray(lower), np.asarray(upper)
    if low.dtype.kind not in "fc" or high.dtype.kind not in "fc":
        return low, high
    invalid = np.isnan(low) | np.isnan(high)
    low = np.where(invalid, -np.inf, low)
    high = np.where(invalid, np.inf, high)
    return np.nextafter(low, -np.inf), np.nextafter(high, np.inf)


# Veltkamp splitter 2**27 + 1 and the magnitude window of an error-free product:
# the split must not overflow, and the product error must not underflow.
_VELTKAMP_SPLITTER = 134217729.0
_SPLIT_MAGNITUDE_LIMIT = 2.0**996
_PRODUCT_MAGNITUDE_FLOOR = 2.0**-969


def _sum_is_exact(first: HostArray, second: HostArray, total: HostArray, /) -> NDArray:
    """Whether binary64 ``total = first + second`` has no rounding error (TwoSum)."""
    if np.asarray(total).dtype.kind not in "fc":
        return np.ones(np.shape(total), dtype=np.bool_)
    with np.errstate(invalid="ignore", over="ignore"):
        partner = np.subtract(total, first)
        error = np.add(
            np.subtract(first, np.subtract(total, partner)), np.subtract(second, partner)
        )
    return np.isfinite(total) & (error == 0.0)


def _product_is_exact(
    first: HostArray, second: HostArray, product: HostArray, /
) -> NDArray:
    """Whether binary64 ``product = first * second`` has no rounding error (TwoProduct).

    The Veltkamp/Dekker error is exact only without split overflow and without
    product underflow; outside that window the product is treated as inexact,
    except an exact zero factor.
    """
    first_, second_, product_ = (
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        np.asarray(product, dtype=np.float64),
    )
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        first_scaled = _VELTKAMP_SPLITTER * first_
        first_high = first_scaled - (first_scaled - first_)
        first_low = first_ - first_high
        second_scaled = _VELTKAMP_SPLITTER * second_
        second_high = second_scaled - (second_scaled - second_)
        second_low = second_ - second_high
        error = (
            (first_high * second_high - product_)
            + first_high * second_low
            + first_low * second_high
        ) + first_low * second_low
    magnitude = np.abs(product_)
    window = (
        np.isfinite(product_)
        & (np.abs(first_) < _SPLIT_MAGNITUDE_LIMIT)
        & (np.abs(second_) < _SPLIT_MAGNITUDE_LIMIT)
        & (magnitude >= _PRODUCT_MAGNITUDE_FLOOR)
    )
    zero_factor = ((first_ == 0.0) & np.isfinite(second_)) | (
        (second_ == 0.0) & np.isfinite(first_)
    )
    return zero_factor | (window & (error == 0.0))


def _round_out_inexact(
    lower: HostArray, upper: HostArray, lower_exact: HostArray, upper_exact: HostArray, /
) -> Interval:
    """Round inexact endpoints outward; either NaN makes the whole line."""
    widened = _round_out(lower, upper)
    valid = ~(np.isnan(lower) | np.isnan(upper))
    return (
        np.where(valid & lower_exact, lower, widened[0]),
        np.where(valid & upper_exact, upper, widened[1]),
    )


def _ieee_minimum(first: HostArray, second: HostArray | float, /) -> NDArray:
    """IEEE 754-2019 ``minimum``: -0 orders below +0 and the first NaN operand wins.

    NumPy leaves the operand of a signed-zero tie unspecified (x86-64 returns
    the second, aarch64 orders -0 first); exact comparisons and selects order
    endpoints exactly as the native kernels do on every platform.
    """
    take_first = np.isnan(first) | (
        ~np.isnan(second)
        & (np.less(first, second) | (np.equal(first, second) & np.signbit(first)))
    )
    return np.asarray(np.where(take_first, first, second))


def _ieee_maximum(first: HostArray, second: HostArray | float, /) -> NDArray:
    """IEEE 754-2019 ``maximum``: +0 orders above -0 and the first NaN operand wins."""
    take_first = np.isnan(first) | (
        ~np.isnan(second)
        & (np.greater(first, second) | (np.equal(first, second) & ~np.signbit(first)))
    )
    return np.asarray(np.where(take_first, first, second))


type _IeeeExtremum = Callable[[HostArray, HostArray | float], NDArray]


def _ieee_accumulate(extremum: _IeeeExtremum, value: HostArray, axis: int, /) -> NDArray:
    """Running IEEE extremum along ``axis`` in ``ceil(log2(extent))`` select passes.

    Each pass folds the adjacent earlier window into the later one, so every
    prefix reduces in index order and its first NaN wins, as a native fold does.
    """
    running = np.moveaxis(np.asarray(value), axis, -1)
    shift = 1
    while shift < running.shape[-1]:
        running = np.concatenate(
            (
                running[..., :shift],
                extremum(running[..., :-shift], running[..., shift:]),
            ),
            axis=-1,
        )
        shift *= 2
    return np.moveaxis(running, -1, axis)


def _hull(values: Sequence[Interval], /) -> Interval:
    if not values:
        raise ValueError("interval hull requires at least one value.")
    lower = reduce(_ieee_minimum, (np.asarray(value[0]) for value in values))
    upper = reduce(_ieee_maximum, (np.asarray(value[1]) for value in values))
    return lower, upper


def _sign_clamped(
    value: Interval, nonnegative: HostArray, nonpositive: HostArray, /
) -> Interval:
    # Outward rounding widens an exact zero bound across the origin. A result
    # whose sign is fixed by its operand signs keeps that sign, so a sum of
    # squares remains an admissible base for even roots and real powers.
    return (
        np.where(nonnegative, _ieee_maximum(value[0], 0.0), value[0]),
        np.where(nonpositive, _ieee_minimum(value[1], 0.0), value[1]),
    )


def _native_float64(first: Interval, second: Interval, /) -> bool:
    """Whether the native endpoint kernels apply: float64 endpoints and meshcore.

    The kernels repeat the host operations below per element, including the
    IEEE 754-2019 endpoint ordering of ``_ieee_minimum``/``_ieee_maximum``, so
    their enclosures are bit-identical on every platform; they remove the
    per-call NumPy overhead that dominates the interpreter's small batches.
    """
    return (
        all(
            getattr(value, "dtype", None) == np.float64
            for value in (first[0], first[1], second[0], second[1])
        )
        and meshcore_available()
    )


def interval_add(first: Interval, second: Interval, /) -> Interval:
    shape = np.broadcast_shapes(np.shape(first[0]), np.shape(second[0]))
    if (
        np.all(np.asarray(second[0]) == 0)
        and np.all(np.asarray(second[1]) == 0)
        and not any(np.any(np.isnan(value)) for value in first)
    ):
        return np.broadcast_to(first[0], shape), np.broadcast_to(first[1], shape)
    if (
        np.all(np.asarray(first[0]) == 0)
        and np.all(np.asarray(first[1]) == 0)
        and not any(np.any(np.isnan(value)) for value in second)
    ):
        return np.broadcast_to(second[0], shape), np.broadcast_to(second[1], shape)
    if _native_float64(first, second):
        return float64_interval_sum(*first, *second, subtract=False)
    return _host_interval_add(first, second)


def _host_interval_add(first: Interval, second: Interval, /) -> Interval:
    with np.errstate(invalid="ignore", over="ignore"):
        lower = np.add(first[0], second[0])
        upper = np.add(first[1], second[1])
        result = _round_out_inexact(
            lower,
            upper,
            _sum_is_exact(first[0], second[0], lower),
            _sum_is_exact(first[1], second[1], upper),
        )
    valid = ~(np.isnan(lower) | np.isnan(upper))
    return _sign_clamped(
        result,
        valid & (np.asarray(first[0]) >= 0.0) & (np.asarray(second[0]) >= 0.0),
        valid & (np.asarray(first[1]) <= 0.0) & (np.asarray(second[1]) <= 0.0),
    )


def interval_subtract(first: Interval, second: Interval, /) -> Interval:
    if _native_float64(first, second):
        return float64_interval_sum(*first, *second, subtract=True)
    return _host_interval_subtract(first, second)


def _host_interval_subtract(first: Interval, second: Interval, /) -> Interval:
    with np.errstate(invalid="ignore", over="ignore"):
        lower = np.subtract(first[0], second[1])
        upper = np.subtract(first[1], second[0])
        return _round_out_inexact(
            lower,
            upper,
            _sum_is_exact(first[0], np.negative(second[1]), lower),
            _sum_is_exact(first[1], np.negative(second[0]), upper),
        )


def _zero_times_unbounded(left: HostArray, right: HostArray, /) -> NDArray:
    left_, right_ = np.asarray(left), np.asarray(right)
    return ((left_ == 0.0) & np.isinf(right_)) | (np.isinf(left_) & (right_ == 0.0))


def interval_multiply(first: Interval, second: Interval, /) -> Interval:
    # Interval endpoints bound real numbers; an infinite endpoint marks an
    # unbounded side, never a member. An exact zero endpoint times it therefore
    # contributes 0, as in IEEE 1788 set-based multiplication.
    if _native_float64(first, second):
        return float64_interval_product(*first, *second)
    return _host_interval_multiply(first, second)


def _host_interval_multiply(first: Interval, second: Interval, /) -> Interval:
    pairs = (
        (first[0], second[0]),
        (first[0], second[1]),
        (first[1], second[0]),
        (first[1], second[1]),
    )
    zero_unbounded = tuple(_zero_times_unbounded(left, right) for left, right in pairs)
    with np.errstate(invalid="ignore", over="ignore"):
        products = tuple(
            np.where(zero, 0.0, np.multiply(left, right))
            for (left, right), zero in zip(pairs, zero_unbounded, strict=True)
        )
    invalid = np.logical_or.reduce(tuple(np.isnan(value) for value in products))
    exact = tuple(
        zero | _product_is_exact(left, right, product)
        for (left, right), product, zero in zip(
            pairs, products, zero_unbounded, strict=True
        )
    )
    # Bound every candidate before reducing: an exact zero tied with an
    # underflowed nonzero product must not suppress its outward rounding.
    bounds = tuple(
        _round_out_inexact(product, product, flag, flag)
        for product, flag in zip(products, exact, strict=True)
    )
    result = (
        np.where(invalid, -np.inf, reduce(_ieee_minimum, (pair[0] for pair in bounds))),
        np.where(invalid, np.inf, reduce(_ieee_maximum, (pair[1] for pair in bounds))),
    )
    first_nonnegative = np.asarray(first[0]) >= 0.0
    first_nonpositive = np.asarray(first[1]) <= 0.0
    second_nonnegative = np.asarray(second[0]) >= 0.0
    second_nonpositive = np.asarray(second[1]) <= 0.0
    return _sign_clamped(
        result,
        ~invalid
        & (
            (first_nonnegative & second_nonnegative)
            | (first_nonpositive & second_nonpositive)
        ),
        ~invalid
        & (
            (first_nonnegative & second_nonpositive)
            | (first_nonpositive & second_nonnegative)
        ),
    )


def _quotient_is_exact(
    numerator: HostArray | float, denominator: HostArray, quotient: HostArray, /
) -> NDArray:
    """Whether binary64 ``quotient = numerator / denominator`` is exact."""
    numerator_ = np.asarray(numerator, dtype=np.float64)
    denominator_ = np.asarray(denominator, dtype=np.float64)
    quotient_ = np.asarray(quotient, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        product = np.multiply(quotient_, denominator_)
    return (
        np.isfinite(quotient_)
        & (denominator_ != 0.0)
        & (product == numerator_)
        & _product_is_exact(quotient_, denominator_, product)
    )


def interval_divide(first: Interval, second: Interval, /) -> Interval:
    denominator_lower = np.asarray(second[0])
    denominator_upper = np.asarray(second[1])
    singular = (denominator_lower <= 0.0) & (denominator_upper >= 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        reciprocal_lower = np.divide(1.0, denominator_upper)
        reciprocal_upper = np.divide(1.0, denominator_lower)
        reciprocal = _round_out_inexact(
            reciprocal_lower,
            reciprocal_upper,
            _quotient_is_exact(1.0, denominator_upper, reciprocal_lower),
            _quotient_is_exact(1.0, denominator_lower, reciprocal_upper),
        )
        result = interval_multiply(first, reciprocal)
        # A point quotient that binary64 division represents exactly is a point.
        quotient = np.divide(first[0], second[0])
        point = (
            (np.asarray(first[0]) == np.asarray(first[1]))
            & (denominator_lower == denominator_upper)
            & _quotient_is_exact(first[0], second[0], quotient)
        )
        result = (
            np.where(point, quotient, result[0]),
            np.where(point, quotient, result[1]),
        )
    return (
        np.where(singular, -np.inf, result[0]),
        np.where(singular, np.inf, result[1]),
    )


def interval_square(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    with np.errstate(invalid="ignore", over="ignore"):
        first, second = np.square(lower), np.square(upper)
    crosses = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(crosses, 0.0, _ieee_minimum(first, second))
    high = _ieee_maximum(first, second)
    rounded = _round_out(low, high)
    return _ieee_maximum(rounded[0], 0.0), rounded[1]


def interval_sqrt(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    with np.errstate(invalid="ignore", over="ignore"):
        low = np.sqrt(_ieee_maximum(lower, 0.0))
        high = np.sqrt(upper)
    low = np.where(lower < 0.0, 0.0, low)
    high = np.where(upper < 0.0, np.nan, high)
    return _round_out(low, high)


def _elementary(function: ElementaryFunction, value: Interval, /) -> Interval:
    """Rigorous enclosure from the native libm-free elementary kernel.

    No host rule is sound on every platform (libm results carry no error
    bound), so without meshcore the primitive has no sound interval extension.
    """
    if not meshcore_available():
        raise IntervalExtensionError(
            f"the rigorous {function} enclosure requires the native meshcore library."
        )
    return float64_interval_elementary(function, value[0], value[1])


def interval_exp(value: Interval, /) -> Interval:
    return _elementary("exp", value)


def interval_log(value: Interval, /) -> Interval:
    return _elementary("log", value)


def interval_sin(value: Interval, /) -> Interval:
    return _elementary("sin", value)


def interval_cos(value: Interval, /) -> Interval:
    return _elementary("cos", value)


# Outward binary64 enclosure of the real constant 2 pi.
_TWO_PI: Interval = (
    np.nextafter(np.float64(2.0 * math.pi), -np.inf),
    np.nextafter(np.float64(2.0 * math.pi), np.inf),
)


def interval_periodic_harmonic(
    coordinate: Interval, frequency: HostArray, order: int, /
) -> Interval:
    """Enclose ``d^k/du^k (sin(2 pi n u), cos(2 pi n u))`` on a new last axis.

    The real functions are periodic, so the enclosure is taken over the
    unreduced coordinate interval with an outward enclosure of ``2 pi``; the
    point program's exact reduction changes rounding only, never the value.
    """
    frequencies = np.asarray(frequency, dtype=np.float64)
    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or order < 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies < 0.0)
        or np.any(frequencies != np.floor(frequencies))
    ):
        raise IntervalExtensionError(
            "periodic harmonics require exact nonnegative integer frequencies and order."
        )
    exact = (frequencies, frequencies)
    phase = interval_multiply(exact, coordinate)
    if order % 4:
        quarter = np.float64(0.25 * (order % 4))
        phase = interval_add(phase, (quarter, quarter))
    angle = interval_multiply(_TWO_PI, phase)
    sine, cosine = interval_sin(angle), interval_cos(angle)
    if order:
        rate = interval_multiply(_TWO_PI, exact)
        scale = rate
        for _ in range(order - 1):
            scale = interval_multiply(scale, rate)
        sine, cosine = interval_multiply(scale, sine), interval_multiply(scale, cosine)
    sine_lower, sine_upper, cosine_lower, cosine_upper = np.broadcast_arrays(
        sine[0], sine[1], cosine[0], cosine[1]
    )
    return (
        np.stack((sine_lower, cosine_lower), axis=-1),
        np.stack((sine_upper, cosine_upper), axis=-1),
    )


def interval_tan(value: Interval, /) -> Interval:
    return _elementary("tan", value)


def interval_absolute(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    crosses = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(crosses, 0.0, _ieee_minimum(np.abs(lower), np.abs(upper)))
    high = _ieee_maximum(np.abs(lower), np.abs(upper))
    return _round_out(low, high)


def interval_maximum(first: Interval, second: Interval, /) -> Interval:
    return _ieee_maximum(first[0], second[0]), _ieee_maximum(first[1], second[1])


def interval_minimum(first: Interval, second: Interval, /) -> Interval:
    return _ieee_minimum(first[0], second[0]), _ieee_minimum(first[1], second[1])


def _reduce_sum(value: Interval, axes: tuple[int, ...], /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    for axis in sorted(axes, reverse=True):
        moved_lower = np.moveaxis(lower, axis, -1)
        moved_upper = np.moveaxis(upper, axis, -1)
        if moved_lower.shape[-1] == 0:
            lower = np.zeros(moved_lower.shape[:-1], dtype=np.float64)
            upper = np.zeros(moved_upper.shape[:-1], dtype=np.float64)
            continue
        low = moved_lower[..., 0]
        high = moved_upper[..., 0]
        for index in range(1, moved_lower.shape[-1]):
            low, high = interval_add(
                (low, high), (moved_lower[..., index], moved_upper[..., index])
            )
        lower, upper = np.asarray(low), np.asarray(high)
    return lower, upper


def _reduce_extremum(
    value: Interval, axes: tuple[int, ...], extremum: _IeeeExtremum, /
) -> Interval:
    """Reduce both endpoints over ``axes`` in the native lane order (C order of ``axes``)."""

    def endpoint(bound: HostArray, /) -> NDArray:
        array = np.asarray(bound)
        kept = array.ndim - len(axes)
        moved = np.moveaxis(array, axes, tuple(range(kept, array.ndim)))
        lanes = moved.reshape((*moved.shape[:kept], math.prod(moved.shape[kept:])))
        if lanes.shape[-1] == 0:
            raise IntervalExtensionError("interval extremum of an empty axis.")
        return _ieee_accumulate(extremum, lanes, -1)[..., -1]

    return endpoint(value[0]), endpoint(value[1])


def _dot_general(
    left: Interval,
    right: Interval,
    dimension_numbers: tuple[
        tuple[Sequence[int], Sequence[int]], tuple[Sequence[int], Sequence[int]]
    ],
    /,
) -> Interval:
    (left_contract, right_contract), (left_batch, right_batch) = dimension_numbers
    left_contract_ = tuple(int(axis) for axis in left_contract)
    right_contract_ = tuple(int(axis) for axis in right_contract)
    left_batch_ = tuple(int(axis) for axis in left_batch)
    right_batch_ = tuple(int(axis) for axis in right_batch)
    left_rank, right_rank = np.ndim(left[0]), np.ndim(right[0])
    if len(left_contract_) != len(right_contract_) or len(left_batch_) != len(
        right_batch_
    ):
        raise IntervalExtensionError(
            "dot_general has incompatible contraction dimensions."
        )
    left_noncontract = tuple(
        axis for axis in range(left_rank) if axis not in (*left_contract_, *left_batch_)
    )
    right_noncontract = tuple(
        axis
        for axis in range(right_rank)
        if axis not in (*right_contract_, *right_batch_)
    )
    left_order = (*left_batch_, *left_noncontract, *left_contract_)
    right_order = (*right_batch_, *right_contract_, *right_noncontract)
    left_shape = np.asarray(left[0]).shape
    right_shape = np.asarray(right[0]).shape
    batch_shape = tuple(left_shape[axis] for axis in left_batch_)
    contract_shape = tuple(left_shape[axis] for axis in left_contract_)
    if any(
        left_shape[a] != right_shape[b]
        for a, b in zip(left_batch_, right_batch_, strict=True)
    ):
        raise IntervalExtensionError(
            "dot_general batch dimensions have different extents."
        )
    if any(
        left_shape[a] != right_shape[b]
        for a, b in zip(left_contract_, right_contract_, strict=True)
    ):
        raise IntervalExtensionError(
            "dot_general contraction dimensions have different extents."
        )
    left_free_shape = tuple(left_shape[axis] for axis in left_noncontract)
    right_free_shape = tuple(right_shape[axis] for axis in right_noncontract)
    batch_count = math.prod(batch_shape) if batch_shape else 1
    left_count = math.prod(left_free_shape) if left_free_shape else 1
    contract_count = math.prod(contract_shape) if contract_shape else 1
    right_count = math.prod(right_free_shape) if right_free_shape else 1

    def bank(
        value: Interval, order: tuple[int, ...], shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.transpose(np.asarray(value[0]), order).reshape(shape)
        upper = np.transpose(np.asarray(value[1]), order).reshape(shape)
        return lower, upper

    left_lower, left_upper = bank(
        left,
        left_order,
        (batch_count, left_count, contract_count),
    )
    right_lower, right_upper = bank(
        right,
        right_order,
        (batch_count, contract_count, right_count),
    )
    result_lower = np.zeros((batch_count, left_count, right_count), dtype=np.float64)
    result_upper = np.zeros_like(result_lower)
    for index in range(contract_count):
        term = interval_multiply(
            (left_lower[:, :, index, None], left_upper[:, :, index, None]),
            (right_lower[:, index, None, :], right_upper[:, index, None, :]),
        )
        result_lower, result_upper = interval_add((result_lower, result_upper), term)
    output_shape = (*batch_shape, *left_free_shape, *right_free_shape)
    return result_lower.reshape(output_shape), result_upper.reshape(output_shape)


def _comparison(name: str, first: Interval, second: Interval, /) -> Interval:
    left_lower, left_upper = np.asarray(first[0]), np.asarray(first[1])
    right_lower, right_upper = np.asarray(second[0]), np.asarray(second[1])
    if name == "lt":
        definitely_true, definitely_false = (
            left_upper < right_lower,
            left_lower >= right_upper,
        )
    elif name == "le":
        definitely_true, definitely_false = (
            left_upper <= right_lower,
            left_lower > right_upper,
        )
    elif name == "gt":
        definitely_true, definitely_false = (
            left_lower > right_upper,
            left_upper <= right_lower,
        )
    elif name == "ge":
        definitely_true, definitely_false = (
            left_lower >= right_upper,
            left_upper < right_lower,
        )
    elif name == "eq":
        definitely_true = (
            (left_lower == left_upper)
            & (right_lower == right_upper)
            & (left_lower == right_lower)
        )
        definitely_false = (left_upper < right_lower) | (right_upper < left_lower)
    elif name == "ne":
        definitely_false = (
            (left_lower == left_upper)
            & (right_lower == right_upper)
            & (left_lower == right_lower)
        )
        definitely_true = (left_upper < right_lower) | (right_upper < left_lower)
    else:
        raise IntervalExtensionError(f"unsupported comparison primitive {name!r}.")
    return definitely_true, np.logical_not(definitely_false)


def _boolean_interval(value: Interval, /) -> tuple[HostArray, HostArray]:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    if (
        lower.dtype.kind != "b"
        or upper.dtype.kind != "b"
        or lower.shape != upper.shape
        or np.any(lower & ~upper)
    ):
        raise IntervalExtensionError("Boolean interval endpoints must be ordered.")
    return lower, upper


def _boolean_binary(name: str, first: Interval, second: Interval, /) -> Interval:
    first_lower, first_upper = _boolean_interval(first)
    second_lower, second_upper = _boolean_interval(second)
    if name == "and":
        return (
            np.logical_and(first_lower, second_lower),
            np.logical_and(first_upper, second_upper),
        )
    if name == "or":
        return (
            np.logical_or(first_lower, second_lower),
            np.logical_or(first_upper, second_upper),
        )
    raise IntervalExtensionError(f"unsupported boolean primitive {name!r}.")


def _boolean_not(value: Interval, /) -> Interval:
    lower, upper = _boolean_interval(value)
    return np.logical_not(upper), np.logical_not(lower)


def _boolean_any(value: Interval, axes: tuple[int, ...], /) -> Interval:
    lower, upper = _boolean_interval(value)
    return np.any(lower, axis=axes), np.any(upper, axis=axes)


def _interval_is_finite(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    if (
        lower.dtype.kind not in "biuf"
        or upper.dtype.kind not in "biuf"
        or lower.shape != upper.shape
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise IntervalExtensionError(
            "is_finite requires ordered, non-NaN real interval endpoints."
        )
    definitely_finite = np.isfinite(lower) & np.isfinite(upper)
    definitely_nonfinite = (lower == upper) & np.isinf(lower)
    return definitely_finite, np.logical_not(definitely_nonfinite)


def _select_n(index: Interval, cases: tuple[Interval, ...], /) -> Interval:
    lower_index, upper_index = np.asarray(index[0]), np.asarray(index[1])
    if (
        np.any(lower_index < 0)
        or np.any(upper_index >= len(cases))
        or np.any(lower_index > upper_index)
    ):
        raise IntervalExtensionError("Selection interval is outside the branch table.")
    shape = np.broadcast_shapes(
        lower_index.shape,
        upper_index.shape,
        *(np.shape(bound) for case in cases for bound in case),
    )
    dtype = np.result_type(*(np.asarray(bound).dtype for case in cases for bound in case))
    lower = np.zeros(shape, dtype=dtype)
    upper = np.zeros(shape, dtype=dtype)
    covered = np.zeros(shape, dtype=np.bool_)
    for number, case in enumerate(cases):
        selected = (lower_index <= number) & (number <= upper_index)
        next_lower = np.where(covered, _ieee_minimum(lower, case[0]), case[0])
        next_upper = np.where(covered, _ieee_maximum(upper, case[1]), case[1])
        lower = np.where(selected, next_lower, lower)
        upper = np.where(selected, next_upper, upper)
        covered |= selected
    if not np.all(covered):
        raise IntervalExtensionError("Selection interval has no integer branch.")
    return lower, upper


def _broadcast_in_dim(
    value: Interval, shape: tuple[int, ...], dimensions: tuple[int, ...], /
) -> Interval:
    def expand(array: HostArray) -> np.ndarray:
        source = np.asarray(array)
        if len(dimensions) != source.ndim:
            raise IntervalExtensionError("broadcast_in_dim mapping has the wrong rank.")
        target = [1] * len(shape)
        for axis, dimension in enumerate(dimensions):
            target[dimension] = source.shape[axis]
        return np.broadcast_to(source.reshape(tuple(target)), shape)

    return expand(value[0]), expand(value[1])


def _pad_interval(
    value: Interval,
    padding_value: Interval,
    configuration: Sequence[tuple[int, int, int]],
    /,
) -> Interval:
    """Transport bounds through cropping and interior padding without rounding."""
    shape = np.asarray(value[0]).shape
    if len(configuration) != len(shape):
        raise IntervalExtensionError("Padding configuration must match operand rank.")
    output_shape = tuple(
        extent + low + high + interior * max(extent - 1, 0)
        for extent, (low, high, interior) in zip(shape, configuration, strict=True)
    )
    if any(extent < 0 for extent in output_shape) or any(
        interior < 0 for _, _, interior in configuration
    ):
        raise IntervalExtensionError("Padding produces an invalid output shape.")
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for extent, output_extent, (low, _, interior) in zip(
        shape, output_shape, configuration, strict=True
    ):
        step = interior + 1
        first = max(0, -(low // step))
        stop = max(first, min(extent, -((low - output_extent) // step)))
        source_slices.append(slice(first, stop))
        target_slices.append(slice(low + first * step, low + stop * step, step))
    lower = np.full(output_shape, padding_value[0], dtype=np.asarray(value[0]).dtype)
    upper = np.full(output_shape, padding_value[1], dtype=np.asarray(value[1]).dtype)
    lower[tuple(target_slices)] = np.asarray(value[0])[tuple(source_slices)]
    upper[tuple(target_slices)] = np.asarray(value[1])[tuple(source_slices)]
    return lower, upper


def _convert_element_type(value: Interval, dtype: Any, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    target = np.dtype(dtype)
    if (
        lower.shape != upper.shape
        or lower.dtype.kind not in "biuf"
        or upper.dtype.kind not in "biuf"
    ):
        raise IntervalExtensionError("Element conversion requires real interval values.")
    if target.kind == "b":
        if lower.dtype.kind == "b" and upper.dtype.kind == "b":
            return lower, upper
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise IntervalExtensionError("Boolean conversion cannot enclose NaN values.")
        definitely_true = (lower > 0) | (upper < 0)
        definitely_false = (lower == 0) & (upper == 0)
        return definitely_true, np.logical_not(definitely_false)
    if target.kind in "iu":
        if lower.dtype.kind == "f" or upper.dtype.kind == "f":
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise IntervalExtensionError(
                    "Integer conversion requires finite interval endpoints."
                )
            lower, upper = np.trunc(lower), np.trunc(upper)
        bits = target.itemsize * 8
        lower_limit = -(1 << (bits - 1)) if target.kind == "i" else 0
        upper_limit = (1 << (bits - 1)) - 1 if target.kind == "i" else (1 << bits) - 1
        if (
            np.any(lower < lower_limit)
            or np.any(upper > upper_limit)
            or np.any(lower > upper)
        ):
            raise IntervalExtensionError(
                "Integer conversion interval is outside the target type."
            )
        return np.asarray(lower, dtype=target), np.asarray(upper, dtype=target)
    if target.kind == "f":
        return np.asarray(lower, dtype=target), np.asarray(upper, dtype=target)
    raise IntervalExtensionError(f"unsupported conversion target dtype {target!r}.")


def _integer_index_interval(value: Interval, vector_size: int, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    if (
        lower.shape != upper.shape
        or lower.ndim < 1
        or lower.shape[-1] != vector_size
        or np.result_type(lower.dtype, upper.dtype).kind not in "iuf"
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise IntervalExtensionError(
            "Index intervals must be ordered finite integer-vector bounds."
        )
    if lower.dtype.kind == "f" or upper.dtype.kind == "f":
        integer_lower = np.ceil(lower)
        integer_upper = np.floor(upper)
    else:
        integer_lower, integer_upper = lower, upper
    if np.any(integer_lower > integer_upper):
        raise IntervalExtensionError("Index interval contains no integer value.")
    return integer_lower, integer_upper


def _argmin_interval(value: Interval, params: Mapping[str, Any], /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    axes = tuple(int(axis) for axis in params["axes"])
    if (
        lower.shape != upper.shape
        or lower.dtype.kind not in "iuf"
        or upper.dtype.kind not in "iuf"
        or len(axes) != 1
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise IntervalExtensionError(
            "argmin requires ordered non-NaN intervals and one reduction axis."
        )
    axis = axes[0]
    if axis < 0:
        axis += lower.ndim
    if axis < 0 or axis >= lower.ndim:
        raise IntervalExtensionError("argmin reduction axis is outside its input.")
    moved_lower = np.moveaxis(lower, axis, -1)
    moved_upper = np.moveaxis(upper, axis, -1)
    extent = moved_lower.shape[-1]
    if extent == 0:
        raise IntervalExtensionError("argmin cannot reduce an empty interval axis.")
    output_shape = moved_lower.shape[:-1]
    possible_lower = np.full(output_shape, extent, dtype=np.int64)
    possible_upper = np.full(output_shape, -1, dtype=np.int64)
    for index in range(extent):
        candidate = moved_lower[..., index]
        can_win = np.ones(output_shape, dtype=np.bool_)
        if index:
            can_win &= np.all(moved_upper[..., :index] > candidate[..., None], axis=-1)
        if index + 1 < extent:
            can_win &= np.all(
                moved_upper[..., index + 1 :] >= candidate[..., None], axis=-1
            )
        possible_lower = np.where(
            can_win, np.minimum(possible_lower, index), possible_lower
        )
        possible_upper = np.where(
            can_win, np.maximum(possible_upper, index), possible_upper
        )
    if np.any(possible_upper < 0):
        raise IntervalExtensionError("argmin interval has no possible minimum index.")
    index_dtype = np.dtype(params["index_dtype"])
    if index_dtype.kind not in "iu":
        raise IntervalExtensionError("argmin index dtype must be integral.")
    return (
        np.asarray(possible_lower, dtype=index_dtype),
        np.asarray(possible_upper, dtype=index_dtype),
    )


def _gather_interval(
    operand: Interval, start_indices: Interval, params: Mapping[str, Any], /
) -> Interval:
    operand_lower, operand_upper = np.asarray(operand[0]), np.asarray(operand[1])
    exact_lower, exact_upper = np.asarray(start_indices[0]), np.asarray(start_indices[1])
    if (
        operand_lower.shape == operand_upper.shape
        and exact_lower.shape == exact_upper.shape
        and exact_lower.dtype.kind in "iuf"
        and np.array_equal(exact_lower, exact_upper)
        and np.all(np.isfinite(exact_lower))
        and np.all(exact_lower == np.floor(exact_lower))
    ):
        # Exactly known indices select the same entries from both endpoint
        # arrays: the primitive itself, under every dimension layout and mode,
        # is an exact selection.
        indices = exact_lower.astype(np.int64)
        return (
            np.asarray(jax.lax.gather_p.bind(operand_lower, indices, **params)),
            np.asarray(jax.lax.gather_p.bind(operand_upper, indices, **params)),
        )
    index_lower, index_upper = _integer_index_interval(start_indices, 1)
    dimensions = params["dimension_numbers"]
    start_index_map = tuple(dimensions.start_index_map)
    mode = params["mode"]
    # A batched scalar dynamic index (``x[i]`` under vmap) keeps its size-one
    # slice axis instead of collapsing it and clamps its start index.
    collapsed = tuple(dimensions.collapsed_slice_dims) == start_index_map
    if (
        operand_lower.shape != operand_upper.shape
        or len(start_index_map) != 1
        or not (collapsed or not dimensions.collapsed_slice_dims)
        or dimensions.operand_batching_dims
        or dimensions.start_indices_batching_dims
        or mode
        not in (
            jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
            jax.lax.GatherScatterMode.CLIP,
        )
        or params["fill_value"] is not None
    ):
        raise IntervalExtensionError(
            "gather requires one unbatched, in-bounds single-axis slice."
        )
    axis = start_index_map[0]
    if axis < 0 or axis >= operand_lower.ndim:
        raise IntervalExtensionError("gather index axis is outside its operand.")
    index_shape = index_lower.shape[:-1]
    remaining_axes = tuple(
        dimension
        for dimension in range(operand_lower.ndim)
        if dimension != axis or not collapsed
    )
    output_shape = (
        *index_shape,
        *(
            1 if dimension == axis else operand_lower.shape[dimension]
            for dimension in remaining_axes
        ),
    )
    expected_offsets = tuple(range(len(index_shape), len(output_shape)))
    expected_slice_sizes = tuple(
        1 if dimension == axis else extent
        for dimension, extent in enumerate(operand_lower.shape)
    )
    if (
        tuple(dimensions.offset_dims) != expected_offsets
        or tuple(params["slice_sizes"]) != expected_slice_sizes
    ):
        raise IntervalExtensionError(
            "gather requires full slices and one collapsed index dimension."
        )
    integer_lower, integer_upper = (
        np.asarray(index_lower)[..., 0],
        np.asarray(index_upper)[..., 0],
    )
    extent = operand_lower.shape[axis]
    if mode == jax.lax.GatherScatterMode.CLIP:
        # Clamping is monotone: clamped endpoints enclose every clamped index.
        integer_lower = np.clip(integer_lower, 0, extent - 1)
        integer_upper = np.clip(integer_upper, 0, extent - 1)
    if np.any(integer_lower < 0) or np.any(integer_upper >= extent):
        raise IntervalExtensionError("gather index interval is outside the operand.")
    integer_lower = integer_lower.astype(np.intp)
    integer_upper = integer_upper.astype(np.intp)
    if not collapsed:
        operand_lower = np.expand_dims(operand_lower, axis + 1)
        operand_upper = np.expand_dims(operand_upper, axis + 1)
    if np.array_equal(integer_lower, integer_upper):
        return (
            np.take(operand_lower, integer_lower, axis=axis),
            np.take(operand_upper, integer_upper, axis=axis),
        )

    result_lower = np.zeros(output_shape, dtype=operand_lower.dtype)
    result_upper = np.zeros(output_shape, dtype=operand_upper.dtype)
    covered = np.zeros(index_shape, dtype=np.bool_)
    index_rank = len(index_shape)
    result_rank = len(output_shape)
    mask_shape = (*index_shape, *((1,) * (result_rank - index_rank)))
    selection_shape = (1,) * index_rank
    for number in range(int(np.min(integer_lower)), int(np.max(integer_upper)) + 1):
        active = (integer_lower <= number) & (number <= integer_upper)
        if not np.any(active):
            continue
        selected_lower = np.asarray(np.take(operand_lower, number, axis=axis))
        selected_upper = np.asarray(np.take(operand_upper, number, axis=axis))
        selected_lower = np.broadcast_to(
            selected_lower.reshape((*selection_shape, *selected_lower.shape)),
            output_shape,
        )
        selected_upper = np.broadcast_to(
            selected_upper.reshape((*selection_shape, *selected_upper.shape)),
            output_shape,
        )
        active_mask = active.reshape(mask_shape)
        covered_mask = covered.reshape(mask_shape)
        next_lower = np.where(
            covered_mask, _ieee_minimum(result_lower, selected_lower), selected_lower
        )
        next_upper = np.where(
            covered_mask, _ieee_maximum(result_upper, selected_upper), selected_upper
        )
        result_lower = np.where(active_mask, next_lower, result_lower)
        result_upper = np.where(active_mask, next_upper, result_upper)
        covered |= active
    if not np.all(covered):
        raise IntervalExtensionError("gather index interval selects no operand slice.")
    return result_lower, result_upper


def _scatter_add_interval(
    operand: Interval,
    scatter_indices: Interval,
    updates: Interval,
    params: Mapping[str, Any],
    /,
) -> Interval:
    operand_lower, operand_upper = np.asarray(operand[0]), np.asarray(operand[1])
    updates_lower, updates_upper = np.asarray(updates[0]), np.asarray(updates[1])
    index_lower, index_upper = _integer_index_interval(scatter_indices, 1)
    dimensions = params["dimension_numbers"]
    indexed_axes = tuple(dimensions.scatter_dims_to_operand_dims)
    operand_batch_axes = tuple(dimensions.operand_batching_dims)
    index_batch_axes = tuple(dimensions.scatter_indices_batching_dims)
    if (
        operand_lower.shape != operand_upper.shape
        or updates_lower.shape != updates_upper.shape
        or len(indexed_axes) != 1
        or tuple(dimensions.inserted_window_dims) != indexed_axes
        or len(operand_batch_axes) != len(index_batch_axes)
        or len(operand_batch_axes) > 1
    ):
        raise IntervalExtensionError(
            "scatter-add requires one indexed operand dimension and at most one batch axis."
        )
    batched = bool(operand_batch_axes)
    if batched and (
        operand_batch_axes != (0,)
        or index_batch_axes != (0,)
        or operand_lower.ndim < 2
        or index_lower.ndim < 3
        or operand_lower.shape[0] != index_lower.shape[0]
        or operand_lower.shape[0] != updates_lower.shape[0]
    ):
        raise IntervalExtensionError(
            "scatter-add supports only aligned leading batch dimensions."
        )
    axis = indexed_axes[0]
    if axis < 0 or axis >= operand_lower.ndim or (batched and axis == 0):
        raise IntervalExtensionError("scatter-add index axis is outside its operand.")
    index_shape = index_lower.shape[1:-1] if batched else index_lower.shape[:-1]
    remaining_axes = tuple(
        dimension
        for dimension in range(operand_lower.ndim)
        if dimension != axis and (not batched or dimension != 0)
    )
    expected_update_shape = (
        *((operand_lower.shape[0],) if batched else ()),
        *index_shape,
        *(operand_lower.shape[dimension] for dimension in remaining_axes),
    )
    expected_update_dims = tuple(
        range(int(batched) + len(index_shape), len(expected_update_shape))
    )
    if (
        tuple(dimensions.update_window_dims) != expected_update_dims
        or updates_lower.shape != expected_update_shape
        or operand_lower.dtype != operand_upper.dtype
        or updates_lower.dtype != operand_lower.dtype
        or updates_upper.dtype != operand_lower.dtype
        or (
            updates_lower.dtype.kind == "f"
            and (np.any(np.isnan(updates_lower)) or np.any(np.isnan(updates_upper)))
        )
        or np.any(updates_lower > updates_upper)
    ):
        raise IntervalExtensionError(
            "scatter-add requires ordered updates and full, unbatched update windows."
        )
    update_jaxpr = params["update_jaxpr"]
    update_consts = params["update_consts"]
    if isinstance(update_jaxpr, jax_core.ClosedJaxpr):
        if update_jaxpr.consts:
            raise IntervalExtensionError("scatter-add update has captured constants.")
        update_jaxpr = update_jaxpr.jaxpr
    if not isinstance(update_jaxpr, jax_core.Jaxpr):
        raise IntervalExtensionError("scatter-add update is not a supported jaxpr.")
    if (
        update_consts
        or update_jaxpr.constvars
        or len(update_jaxpr.invars) != 2
        or len(update_jaxpr.outvars) != 1
        or len(update_jaxpr.eqns) != 1
        or update_jaxpr.eqns[0].primitive.name not in ("add", "add_any")
        or len(update_jaxpr.eqns[0].invars) != 2
    ):
        raise IntervalExtensionError("scatter-add supports only pure addition updates.")
    mode = params["mode"]
    promise_in_bounds = mode == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS
    if not promise_in_bounds and mode != jax.lax.GatherScatterMode.FILL_OR_DROP:
        raise IntervalExtensionError("scatter-add has an unsupported out-of-bounds mode.")
    integer_lower, integer_upper = (
        np.asarray(index_lower)[..., 0],
        np.asarray(index_upper)[..., 0],
    )
    extent = operand_lower.shape[axis]
    if promise_in_bounds and (
        np.any(integer_lower < 0) or np.any(integer_upper >= extent)
    ):
        raise IntervalExtensionError(
            "scatter-add index interval is outside the promised operand bounds."
        )
    result_lower = np.array(operand_lower, copy=True)
    result_upper = np.array(operand_upper, copy=True)
    batch_size = operand_lower.shape[0] if batched else 0
    batch_positions: Sequence[int | None] = range(batch_size) if batched else (None,)
    for batch_index in batch_positions:
        if batch_index is None:
            lane_lower, lane_upper = integer_lower, integer_upper
            lane_updates_lower, lane_updates_upper = updates_lower, updates_upper
            lane_result_lower, lane_result_upper = result_lower, result_upper
            lane_axis = axis
        else:
            lane_lower, lane_upper = (
                integer_lower[batch_index],
                integer_upper[batch_index],
            )
            lane_updates_lower = updates_lower[batch_index]
            lane_updates_upper = updates_upper[batch_index]
            lane_result_lower, lane_result_upper = (
                result_lower[batch_index],
                result_upper[batch_index],
            )
            lane_axis = axis - 1
        for update_position in np.ndindex(index_shape):
            first_index = int(lane_lower[update_position])
            last_index = int(lane_upper[update_position])
            first_valid = max(0, first_index)
            last_valid = min(extent - 1, last_index)
            if first_valid > last_valid:
                continue
            update_lower = np.asarray(lane_updates_lower[update_position])
            update_upper = np.asarray(lane_updates_upper[update_position])
            guaranteed = first_index == last_index
            for destination in range(first_valid, last_valid + 1):
                if guaranteed:
                    contribution = update_lower, update_upper
                else:
                    contribution = (
                        _ieee_minimum(update_lower, 0.0),
                        _ieee_maximum(update_upper, 0.0),
                    )
                selection: list[int | slice] = [slice(None)] * lane_result_lower.ndim
                selection[lane_axis] = destination
                key = tuple(selection)
                destination_interval = (
                    lane_result_lower[key],
                    lane_result_upper[key],
                )
                updated = interval_add(destination_interval, contribution)
                lane_result_lower[key], lane_result_upper[key] = updated
    return result_lower, result_upper


def _searchsorted_interval(
    primitive: SearchSorted, sorted_values: Interval, query: Interval, /
) -> Interval:
    """Enclose the insertion index of every query in its sorted-array interval.

    The index counts sorted entries below the query (``<`` for ``left``, ``<=``
    for ``right``), so it is monotone nondecreasing in the query and
    nonincreasing in every entry. Running maxima of the entry upper bounds and
    reversed running minima of the lower bounds stay sorted and still bound
    every admissible entry, so the concrete search on those sorted bounds
    encloses the index of every real value in the boxes.
    """
    entry_lower = np.asarray(sorted_values[0], dtype=np.float64)
    entry_upper = np.asarray(sorted_values[1], dtype=np.float64)
    query_lower = np.asarray(query[0], dtype=np.float64)
    query_upper = np.asarray(query[1], dtype=np.float64)
    if (
        np.any(np.isnan(entry_lower))
        or np.any(np.isnan(entry_upper))
        or np.any(np.isnan(query_lower))
        or np.any(np.isnan(query_upper))
    ):
        raise IntervalExtensionError("searchsorted requires non-NaN interval bounds.")
    axis = primitive.dimension
    sorted_upper = _ieee_accumulate(_ieee_maximum, entry_upper, axis)
    sorted_lower = np.flip(
        _ieee_accumulate(_ieee_minimum, np.flip(entry_lower, axis=axis), axis), axis=axis
    )
    lower = np.asarray(primitive.expand(sorted_upper, query_lower))
    upper = np.asarray(primitive.expand(sorted_lower, query_upper))
    return lower, upper


type _JetTable = tuple[Interval, ...]


def _bspline_span_jet_table(
    knots: NDArray[np.float64],
    degree: int,
    span: int,
    order_count: int,
    /,
) -> _JetTable:
    """Outward Bernstein coefficients of each local basis jet on one span.

    Entry ``r`` holds, per local basis function, the exact Bernstein net of its
    ``r``-th parameter derivative on ``[knots[span], knots[span + 1]]`` rounded
    outward to binary64, with shape ``(degree + 1, max(degree - r, 0) + 1)``.
    """
    exact_knots = tuple(Fraction(float(value)) for value in knots)
    bounds = (exact_knots[span], exact_knots[span + 1])
    # Rational knots keep every extracted coefficient rational.
    rows = [
        [cast(Fraction, value) for value in row]
        for row in _univariate_span_extraction(exact_knots, degree, span)
    ]
    table: list[Interval] = []
    for order in range(order_count):
        if order:
            rows = [
                [cast(Fraction, value) for value in _bernstein_derivative(row, bounds)]
                for row in rows
            ]
        table.append(
            (
                np.asarray([[_binary64_down(value) for value in row] for row in rows]),
                np.asarray([[_binary64_up(value) for value in row] for row in rows]),
            )
        )
    return tuple(table)


def _bernstein_range(net: Interval, first: Interval, last: Interval, /) -> Interval:
    """Enclose a Bernstein polynomial on ``[first, last]`` of its unit domain.

    ``net`` has shape ``(local, degree + 1)``; ``first``/``last`` enclose the
    local sub-interval endpoints of every box, shape ``(boxes,)``. The
    restricted net's coefficients are blossom values with ``j`` arguments at
    ``last`` and the rest at ``first``; their hull encloses the polynomial on
    the sub-interval (convex hull property), evaluated with outward interval
    de Casteljau steps.
    """
    degree = np.shape(net[0])[-1] - 1
    boxes = np.shape(first[0])[0]
    lower = np.broadcast_to(np.asarray(net[0]), (boxes, *np.shape(net[0])))
    upper = np.broadcast_to(np.asarray(net[1]), lower.shape)
    one = (np.ones(()), np.ones(()))
    result_lower: NDArray[np.float64] | None = None
    result_upper: NDArray[np.float64] | None = None
    for index in range(degree + 1):
        row_lower, row_upper = lower, upper
        for step in range(degree):
            point = last if step < index else first
            parameter = (
                np.asarray(point[0])[:, None, None],
                np.asarray(point[1])[:, None, None],
            )
            combined = interval_add(
                interval_multiply(
                    interval_subtract(one, parameter),
                    (row_lower[..., :-1], row_upper[..., :-1]),
                ),
                interval_multiply(parameter, (row_lower[..., 1:], row_upper[..., 1:])),
            )
            row_lower, row_upper = np.asarray(combined[0]), np.asarray(combined[1])
        value_lower = row_lower[..., 0]
        value_upper = row_upper[..., 0]
        result_lower = (
            value_lower
            if result_lower is None
            else _ieee_minimum(result_lower, value_lower)
        )
        result_upper = (
            value_upper
            if result_upper is None
            else _ieee_maximum(result_upper, value_upper)
        )
    if result_lower is None or result_upper is None:
        raise ValueError("a Bernstein range needs at least one coefficient.")
    return result_lower, result_upper


def _bspline_jet_interval(
    aligned: bool,
    knots: Interval,
    parameters: Interval,
    spans: Interval,
    output_shape: tuple[int, ...],
    tables: dict[tuple[bytes, int, int, int], _JetTable],
    /,
) -> Interval | None:
    """Enclose B-spline basis jets per knot span in Bernstein form.

    Each span index the span interval admits contributes its polynomial
    pieces' Bernstein enclosure on the box. Searched (``aligned``) spans
    contain their parameters, so the box is first clipped to the span and
    spans the box misses contribute nothing; a box crossing knots is thus
    split at them. Piece spans are evaluated on the whole box. Returns
    ``None`` when the equation is outside this rule's preconditions (moving or
    unsorted knots, misaligned shapes, zero-length pieces); callers then
    interpret the equation's own program.
    """
    knot_values = np.asarray(knots[0])
    if (
        knot_values.ndim != 1
        or knot_values.dtype.kind != "f"
        or not np.array_equal(knot_values, np.asarray(knots[1]))
        or not np.all(np.isfinite(knot_values))
        or np.any(np.diff(knot_values) < 0.0)
        or len(output_shape) < 2
    ):
        return None
    order_count, local = output_shape[-2:]
    degree = local - 1
    count = knot_values.size - degree - 1
    parameter_lower = np.asarray(parameters[0], dtype=np.float64)
    parameter_upper = np.asarray(parameters[1], dtype=np.float64)
    span_lower, span_upper = np.asarray(spans[0]), np.asarray(spans[1])
    shape = parameter_lower.shape
    # Spans the batch shares (an explicit piece under ``vmap``) stay unbatched
    # operands; the equation applies them to every parameter row.
    if span_lower.shape == span_upper.shape and span_lower.shape != shape:
        try:
            broadcast = np.broadcast_shapes(span_lower.shape, shape) == shape
        except ValueError:
            broadcast = False
        if broadcast:
            span_lower = np.broadcast_to(span_lower, shape)
            span_upper = np.broadcast_to(span_upper, shape)
    if (
        count <= degree
        or output_shape != (*shape, order_count, local)
        or parameter_upper.shape != shape
        or span_lower.shape != shape
        or span_upper.shape != shape
        # Integer operands travel as exact binary64 interval endpoints.
        or span_lower.dtype.kind not in "iuf"
        or span_upper.dtype.kind not in "iuf"
        or not np.all(np.isfinite(span_lower))
        or not np.all(np.isfinite(span_upper))
        or np.any(span_lower != np.floor(span_lower))
        or np.any(span_upper != np.floor(span_upper))
        or not np.all(np.isfinite(parameter_lower))
        or not np.all(np.isfinite(parameter_upper))
        or np.any(span_lower < degree)
        or np.any(span_upper >= count)
        or np.any(span_lower > span_upper)
    ):
        return None
    boxes = int(np.prod(shape, dtype=np.int64))
    lower = parameter_lower.reshape((boxes,))
    upper = parameter_upper.reshape((boxes,))
    first_span = span_lower.reshape((boxes,)).astype(np.int64)
    last_span = span_upper.reshape((boxes,)).astype(np.int64)
    result_lower = np.full((boxes, order_count, local), np.inf)
    result_upper = np.full((boxes, order_count, local), -np.inf)
    key = knot_values.tobytes()
    for span in range(
        int(first_span.min(initial=degree)), int(last_span.max(initial=degree)) + 1
    ):
        active = (first_span <= span) & (span <= last_span)
        left, right = float(knot_values[span]), float(knot_values[span + 1])
        if aligned:
            first = _ieee_maximum(lower, left)
            last = _ieee_minimum(upper, right)
            active &= first <= last
        else:
            first, last = lower, upper
        if not np.any(active):
            continue
        if left == right:
            # Knot search selects a zero-length span only by clipping into a
            # degenerate last span, where de Boor's quotients are not the
            # span's polynomial; a piece span of zero length has no piece.
            if not aligned or span == count - 1:
                return None
            continue
        table_key = (key, degree, span, order_count)
        table = tables.get(table_key)
        if table is None:
            table = _bspline_span_jet_table(knot_values, degree, span, order_count)
            tables[table_key] = table
        origin = (np.asarray(left), np.asarray(left))
        width = interval_subtract((np.asarray(right), np.asarray(right)), origin)
        start = interval_divide(
            interval_subtract((first[active], first[active]), origin), width
        )
        stop = interval_divide(
            interval_subtract((last[active], last[active]), origin), width
        )
        if aligned:
            # The clipped endpoints lie in the span: their local values lie in [0, 1].
            start = (
                _ieee_minimum(_ieee_maximum(start[0], 0.0), 1.0),
                _ieee_minimum(_ieee_maximum(start[1], 0.0), 1.0),
            )
            stop = (
                _ieee_minimum(_ieee_maximum(stop[0], 0.0), 1.0),
                _ieee_minimum(_ieee_maximum(stop[1], 0.0), 1.0),
            )
        for order, net in enumerate(table):
            jet_lower, jet_upper = _bernstein_range(net, start, stop)
            result_lower[active, order] = _ieee_minimum(
                result_lower[active, order], jet_lower
            )
            result_upper[active, order] = _ieee_maximum(
                result_upper[active, order], jet_upper
            )
    if np.any(result_lower > result_upper):
        return None
    return (
        result_lower.reshape(output_shape),
        result_upper.reshape(output_shape),
    )


def _bspline_jet_equation(
    equation: jax_core.JaxprEqn,
    arguments: tuple[Interval, ...],
    tables: dict[tuple[bytes, int, int, int], _JetTable],
    /,
) -> Interval | None:
    """Span-wise jet enclosure of a staged B-spline basis-jet equation, if it is one."""
    owner = custom_jvp_rule_owner(equation.params.get("jvp_jaxpr_fun"))
    aligned = None if owner is None else bspline_jet_rule_alignment(owner.f)
    operands = arguments[int(equation.params.get("num_consts", 0)) :]
    if aligned is None or len(operands) != 3 or len(equation.outvars) != 1:
        return None
    aval = equation.outvars[0].aval
    if not isinstance(aval, ShapedArray):
        return None
    knots, parameters, spans = operands
    return _bspline_jet_interval(
        aligned, knots, parameters, spans, tuple(aval.shape), tables
    )


type CustomIntervalExtension = Callable[[tuple[Interval, ...]], Interval]

# Declared interval extensions of custom-JVP functions, keyed by their declared
# JVP rule. Each owner registers the sound enclosure of the real function its
# equation denotes; the interpreter then never walks the staged primal.
_CUSTOM_EXTENSIONS: dict[object, CustomIntervalExtension] = {}


def register_custom_jvp_interval_extension(
    rule: object, extension: CustomIntervalExtension, /
) -> None:
    """Declare the interval extension of a custom-JVP function by its JVP rule.

    ``extension`` receives the operand intervals of one staged equation and
    returns an enclosure of its single output over them.
    """
    if rule in _CUSTOM_EXTENSIONS:
        raise ValueError("A custom-JVP rule already declares an interval extension.")
    _CUSTOM_EXTENSIONS[rule] = extension


def _custom_extension_equation(
    equation: jax_core.JaxprEqn, arguments: tuple[Interval, ...], /
) -> Interval | None:
    """Declared enclosure of a staged custom-JVP equation, if its rule has one."""
    owner = custom_jvp_rule_owner(equation.params.get("jvp_jaxpr_fun"))
    extension = None if owner is None else _CUSTOM_EXTENSIONS.get(owner.f)
    if extension is None or len(equation.outvars) != 1:
        return None
    result = extension(arguments[int(equation.params.get("num_consts", 0)) :])
    aval = equation.outvars[0].aval
    if (
        not isinstance(aval, ShapedArray)
        or np.shape(result[0]) != aval.shape
        or np.shape(result[1]) != aval.shape
    ):
        raise IntervalExtensionError(
            "A declared custom-JVP interval extension returned a misaligned enclosure."
        )
    return result


def _primitive(
    name: str, values: tuple[Interval, ...], params: Mapping[str, Any], /
) -> tuple[Interval, ...]:
    # JAX owns the heterogeneous parameter schemas at this IR interchange boundary.
    match name:
        case "add" | "add_any":
            return (interval_add(values[0], values[1]),)
        case "sub":
            return (interval_subtract(values[0], values[1]),)
        case "mul":
            return (interval_multiply(values[0], values[1]),)
        case "div":
            return (interval_divide(values[0], values[1]),)
        case "neg":
            return (_round_out(-np.asarray(values[0][1]), -np.asarray(values[0][0])),)
        case "abs":
            return (interval_absolute(values[0]),)
        case "square":
            return (interval_square(values[0]),)
        case "sqrt":
            return (interval_sqrt(values[0]),)
        case "rsqrt":
            return (
                interval_divide(
                    (np.ones_like(values[0][0]), np.ones_like(values[0][1])),
                    interval_sqrt(values[0]),
                ),
            )
        case "exp":
            return (interval_exp(values[0]),)
        case "log":
            return (interval_log(values[0]),)
        case "sin":
            return (interval_sin(values[0]),)
        case "cos":
            return (interval_cos(values[0]),)
        case "tan":
            return (interval_tan(values[0]),)
        case "atan":
            return (_elementary("atan", values[0]),)
        case "tanh":
            return (_elementary("tanh", values[0]),)
        case "integer_pow":
            exponent = int(params["y"])
            if exponent == 0:
                one = np.ones_like(values[0][0], dtype=np.float64)
                return ((one, one),)
            if exponent < 0:
                one = np.ones_like(values[0][0], dtype=np.float64)
                return (
                    interval_divide(
                        (one, one), _primitive("integer_pow", values, {"y": -exponent})[0]
                    ),
                )
            if exponent % 2 == 0:
                half = _primitive("integer_pow", values, {"y": exponent // 2})[0]
                return (interval_square(half),)
            previous = _primitive("integer_pow", values, {"y": exponent - 1})[0]
            return (interval_multiply(previous, values[0]),)
        case "pow":
            exponent = values[1]
            selected = np.asarray(exponent[0], dtype=np.float64).reshape((-1,))
            # An integer exponent that arrives as one value broadcast per box
            # (e.g. ``blockiness / 2 - 1`` of a traced parameter) is an integer
            # power, so ``0 ** 0`` stays 1 instead of ``exp(0 * log 0)``.
            if (
                selected.size
                and np.array_equal(exponent[0], exponent[1])
                and np.all(selected == selected[0])
                and float(selected[0]).is_integer()
            ):
                shape = np.broadcast_shapes(np.shape(values[0][0]), np.shape(exponent[0]))
                base = (
                    np.broadcast_to(values[0][0], shape),
                    np.broadcast_to(values[0][1], shape),
                )
                return _primitive("integer_pow", (base,), {"y": int(selected[0])})
            if np.any(np.asarray(values[0][0]) < 0.0):
                raise IntervalExtensionError(
                    "noninteger power requires a nonnegative base interval."
                )
            # log extends 0 to -inf, so exp(y log x) encloses 0**y for every
            # exponent sign; a nonnegative base has a nonnegative real power.
            power = interval_exp(interval_multiply(exponent, interval_log(values[0])))
            return ((_ieee_maximum(power[0], 0.0), power[1]),)
        case "maximum" | "max":
            return (interval_maximum(values[0], values[1]),)
        case "minimum" | "min":
            return (interval_minimum(values[0], values[1]),)
        case "lt" | "le" | "gt" | "ge" | "eq" | "ne":
            return (_comparison(name, values[0], values[1]),)
        case "and" | "or":
            return (_boolean_binary(name, values[0], values[1]),)
        case "not":
            return (_boolean_not(values[0]),)
        case "is_finite":
            return (_interval_is_finite(values[0]),)
        case "reduce_or":
            return (_boolean_any(values[0], tuple(params["axes"])),)
        case "unvmap_any":
            axes = tuple(range(np.ndim(values[0][0])))
            return (_boolean_any(values[0], axes),)
        case "select_n":
            return (
                _select_n(
                    values[0],
                    values[1:],
                ),
            )
        case "select":
            predicate = np.asarray(values[0][0], dtype=np.bool_)
            uncertain = np.asarray(values[0][0]) != np.asarray(values[0][1])
            shape = np.broadcast_shapes(
                predicate.shape, np.shape(values[1][0]), np.shape(values[2][0])
            )
            while predicate.ndim < len(shape):
                predicate = predicate[..., None]
                uncertain = uncertain[..., None]
            predicate = np.broadcast_to(predicate, shape)
            uncertain = np.broadcast_to(uncertain, shape)
            selected = (
                np.where(predicate, values[1][0], values[2][0]),
                np.where(predicate, values[1][1], values[2][1]),
            )
            hull = _hull((values[1], values[2]))
            return (
                (
                    np.where(uncertain, hull[0], selected[0]),
                    np.where(uncertain, hull[1], selected[1]),
                ),
            )
        case "broadcast_in_dim":
            return (
                _broadcast_in_dim(
                    values[0],
                    tuple(params["shape"]),
                    tuple(params["broadcast_dimensions"]),
                ),
            )
        case "reshape":
            shape = tuple(params["new_sizes"])
            dimensions = params.get("dimensions")
            lower, upper = np.asarray(values[0][0]), np.asarray(values[0][1])
            if dimensions is not None:
                lower, upper = (
                    np.transpose(lower, dimensions),
                    np.transpose(upper, dimensions),
                )
            return ((np.reshape(lower, shape), np.reshape(upper, shape)),)
        case "transpose":
            axes = tuple(params["permutation"])
            return (
                (
                    np.transpose(np.asarray(values[0][0]), axes),
                    np.transpose(np.asarray(values[0][1]), axes),
                ),
            )
        case "squeeze":
            axes = tuple(params["dimensions"])
            return (
                (
                    np.squeeze(np.asarray(values[0][0]), axis=axes),
                    np.squeeze(np.asarray(values[0][1]), axis=axes),
                ),
            )
        case "concatenate":
            axis = int(params["dimension"])
            return (
                (
                    np.concatenate(tuple(value[0] for value in values), axis=axis),
                    np.concatenate(tuple(value[1] for value in values), axis=axis),
                ),
            )
        case "stack":
            axis = params["axis"]
            return (
                (
                    np.stack(tuple(value[0] for value in values), axis=axis),
                    np.stack(tuple(value[1] for value in values), axis=axis),
                ),
            )
        case "unstack":
            lower, upper = np.asarray(values[0][0]), np.asarray(values[0][1])
            axis = int(params["axis"])
            if lower.shape != upper.shape or lower.ndim == 0:
                raise IntervalExtensionError(
                    "unstack requires aligned non-scalar intervals."
                )
            if axis < 0:
                axis += lower.ndim
            if axis < 0 or axis >= lower.ndim:
                raise IntervalExtensionError("unstack axis is outside the operand rank.")
            return tuple(
                (
                    np.asarray(np.take(lower, index, axis=axis)),
                    np.asarray(np.take(upper, index, axis=axis)),
                )
                for index in range(lower.shape[axis])
            )
        case "split":
            axis = params["axis"]
            sizes = params["sizes"]
            cuts = np.cumsum(sizes, dtype=np.int64)[:-1]
            lower = np.split(np.asarray(values[0][0]), cuts, axis=axis)
            upper = np.split(np.asarray(values[0][1]), cuts, axis=axis)
            return tuple(zip(lower, upper, strict=True))
        case "pad":
            return (_pad_interval(values[0], values[1], params["padding_config"]),)
        case "slice":
            starts, stops = params["start_indices"], params["limit_indices"]
            strides = params["strides"]
            if strides is None:
                slices = tuple(
                    slice(start, stop) for start, stop in zip(starts, stops, strict=True)
                )
            else:
                slices = tuple(
                    slice(start, stop, stride)
                    for start, stop, stride in zip(starts, stops, strides, strict=True)
                )
            return ((np.asarray(values[0][0])[slices], np.asarray(values[0][1])[slices]),)
        case "gather":
            return (_gather_interval(values[0], values[1], params),)
        case "scatter-add":
            return (_scatter_add_interval(values[0], values[1], values[2], params),)
        case "reduce_sum":
            return (_reduce_sum(values[0], tuple(params["axes"])),)
        case "reduce_max":
            return (_reduce_extremum(values[0], tuple(params["axes"]), _ieee_maximum),)
        case "reduce_min":
            return (_reduce_extremum(values[0], tuple(params["axes"]), _ieee_minimum),)
        case "argmin":
            return (_argmin_interval(values[0], params),)
        case "dot_general":
            return (_dot_general(values[0], values[1], params["dimension_numbers"]),)
        case "convert_element_type":
            return (_convert_element_type(values[0], params["new_dtype"]),)
        case "copy" | "stop_gradient" | "device_put":
            return (values[0],)
        case "iota":
            shape = tuple(params["shape"])
            dimension = int(params["dimension"])
            value = np.arange(shape[dimension], dtype=np.dtype(params["dtype"]))
            reshape = tuple(
                shape[axis] if axis == dimension else 1 for axis in range(len(shape))
            )
            result = np.broadcast_to(value.reshape(reshape), shape)
            return ((result, result),)
        case "call_hi_primitive" if isinstance(params["_prim"], SearchSorted):
            return (_searchsorted_interval(params["_prim"], values[0], values[1]),)
        case "call_hi_primitive":
            raise IntervalExtensionError(
                f"no sound interval extension for JAX high-level primitive {type(params['_prim']).__name__!r}."
            )
        case "pure_callback":
            # Opaque host callbacks (traced runtime checks such as Equinox
            # error_if) are reached only when the check is not certainly
            # passing on the box: the interval route refuses rather than
            # certify a value whose own validity check may fail.
            raise IntervalExtensionError(
                "a traced runtime check is not certainly satisfied on this interval box."
            )
        case _:
            raise IntervalExtensionError(
                f"no sound interval extension for JAX primitive {name!r}."
            )


def _read_atom(
    atom: jax_core.Var | jax_core.Literal, environment: dict[jax_core.Var, Interval], /
) -> Interval:
    if isinstance(atom, jax_core.Var):
        return environment[atom]
    return _as_interval(atom.val)


# Pure value-preserving restructurings: equal primitive, parameters and operand
# identity imply the same real value, so their outputs share one identity.
_VIEW_PRIMITIVES = frozenset(
    (
        "slice",
        "squeeze",
        "reshape",
        "expand_dims",
        "broadcast_in_dim",
        "transpose",
        "convert_element_type",
        "copy",
    )
)

type _StaticParameter = (
    int | bool | str | np.dtype[Any] | None | tuple[_StaticParameter, ...]
)


def _static_parameter(value: object, /) -> bool:
    if value is None or isinstance(value, (int, bool, str, np.dtype)):
        return True
    return isinstance(value, tuple) and all(_static_parameter(item) for item in value)


@dataclass(slots=True)
class IntervalOperandIdentity:
    """Real-value identity of jaxpr variables within one jaxpr walk.

    Interval arithmetic loses the dependency between two operands that denote
    the same real value. JAX traces ``x[i] * x[i]`` as two separate slices, so
    identity follows equal view equations back to one representative before a
    product is recognized as an even power.
    """

    representatives: dict[jax_core.Var, jax_core.Var] = field(default_factory=dict)
    views: dict[
        tuple[str, tuple[jax_core.Var, ...], tuple[tuple[str, _StaticParameter], ...]],
        jax_core.Var,
    ] = field(default_factory=dict)

    def _representative(self, variable: jax_core.Var, /) -> jax_core.Var:
        return self.representatives.get(variable, variable)

    def record(self, equation: jax_core.JaxprEqn, /) -> None:
        """Record the identity of a single-output view equation."""
        name = equation.primitive.name
        if (
            name not in _VIEW_PRIMITIVES
            or len(equation.outvars) != 1
            or isinstance(equation.outvars[0], jax_core.DropVar)
            or not all(_static_parameter(value) for value in equation.params.values())
        ):
            return
        operands: list[jax_core.Var] = []
        for atom in equation.invars:
            if not isinstance(atom, jax_core.Var):
                return
            operands.append(self._representative(atom))
        key = (name, tuple(operands), tuple(sorted(equation.params.items())))
        output = equation.outvars[0]
        self.representatives[output] = self.views.setdefault(key, output)

    def is_square(self, equation: jax_core.JaxprEqn, /) -> bool:
        """Return whether a real product multiplies one value by itself."""
        if equation.primitive.name != "mul" or len(equation.invars) != 2:
            return False
        first, second = equation.invars
        return (
            isinstance(first, jax_core.Var)
            and isinstance(second, jax_core.Var)
            and self._representative(first) is self._representative(second)
            and isinstance(first.aval, ShapedArray)
            and first.aval.dtype.kind == "f"
        )


class _Unkeyed:
    """Marks an equation parameter without a canonical hashable spelling."""


_UNKEYED = _Unkeyed()


def _parameter_key(value: object, /) -> object:
    """Hashable spelling of a static equation parameter, else ``_UNKEYED``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.dtype):
        return ("dtype", value.str)
    if isinstance(value, np.generic):
        return ("scalar", value.dtype.str, value.item())
    if isinstance(value, Enum):
        return ("enum", type(value).__qualname__, value.name)
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value, key=str):
            item = _parameter_key(value[key])
            if item is _UNKEYED:
                return _UNKEYED
            items.append((str(key), item))
        return ("mapping", tuple(items))
    if isinstance(value, (tuple, list)):
        items = []
        for entry in value:
            item = _parameter_key(entry)
            if item is _UNKEYED:
                return _UNKEYED
            items.append(item)
        return (type(value).__qualname__, tuple(items))
    return _UNKEYED


def _nested_jaxprs(value: object, /) -> tuple[jax_core.Jaxpr, ...]:
    if isinstance(value, jax_core.ClosedJaxpr):
        return (value.jaxpr,)
    if isinstance(value, jax_core.Jaxpr):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(nested for entry in value for nested in _nested_jaxprs(entry))
    return ()


# A variable whose batch axis cannot be named attributes its events to every box.
_UNATTRIBUTED_AXIS = -1


def _branch_batch_axes(
    first: jax_core.Jaxpr, second: jax_core.Jaxpr, sizes: tuple[int, int], /
) -> dict[jax_core.Var, int | None]:
    """Name the vmapped box axis of every variable from two traced batch sizes.

    The two traces differ only in their batch extent, so the one axis whose
    extent changes is the box axis; an unchanged shape is independent of the
    boxes. Any other difference attributes that variable to every box.
    """
    axes: dict[jax_core.Var, int | None] = {}

    def assign(variable: object, other: object, /) -> None:
        if not isinstance(variable, jax_core.Var) or not isinstance(other, jax_core.Var):
            return
        aval, other_aval = variable.aval, other.aval
        axis: int | None = _UNATTRIBUTED_AXIS
        if (
            isinstance(aval, ShapedArray)
            and isinstance(other_aval, ShapedArray)
            and aval.ndim == other_aval.ndim
        ):
            changed = [
                index
                for index, (extent, other_extent) in enumerate(
                    zip(aval.shape, other_aval.shape, strict=True)
                )
                if extent != other_extent
            ]
            if not changed:
                axis = None
            elif (
                len(changed) == 1
                and aval.shape[changed[0]] == sizes[0]
                and other_aval.shape[changed[0]] == sizes[1]
            ):
                axis = changed[0]
        if axes.setdefault(variable, axis) != axis:
            axes[variable] = _UNATTRIBUTED_AXIS

    def walk(left: jax_core.Jaxpr, right: jax_core.Jaxpr, /) -> None:
        if (
            len(left.eqns) != len(right.eqns)
            or len(left.invars) != len(right.invars)
            or len(left.constvars) != len(right.constvars)
        ):
            raise IntervalExtensionError(
                "Branch evidence requires one jaxpr structure at every batch size."
            )
        for variable, other in zip(left.constvars, right.constvars, strict=True):
            assign(variable, other)
        for variable, other in zip(left.invars, right.invars, strict=True):
            assign(variable, other)
        for equation, other_equation in zip(left.eqns, right.eqns, strict=True):
            if equation.primitive is not other_equation.primitive or len(
                equation.outvars
            ) != len(other_equation.outvars):
                raise IntervalExtensionError(
                    "Branch evidence requires one jaxpr structure at every batch size."
                )
            for variable, other in zip(
                equation.outvars, other_equation.outvars, strict=True
            ):
                assign(variable, other)
            for key, value in equation.params.items():
                nested = _nested_jaxprs(value)
                other_nested = _nested_jaxprs(other_equation.params.get(key))
                if len(nested) != len(other_nested):
                    raise IntervalExtensionError(
                        "Branch evidence requires one jaxpr structure at every batch size."
                    )
                for inner, other_inner in zip(nested, other_nested, strict=True):
                    walk(inner, other_inner)

    walk(first, second)
    return axes


def _endpoints(value: Interval, /) -> tuple[NDArray[Any], NDArray[Any]]:
    return np.asarray(value[0]), np.asarray(value[1])


def _meets_zero(value: Interval, /) -> NDArray[np.bool_]:
    """Elements whose enclosure may contain zero (NaN counts as undecided)."""
    lower, upper = _endpoints(value)
    return ~((lower > 0.0) | (upper < 0.0))


def _not_positive(value: Interval, /) -> NDArray[np.bool_]:
    return ~(_endpoints(value)[0] > 0.0)


def _undecided(value: Interval, /) -> NDArray[np.bool_]:
    lower, upper = _endpoints(value)
    return np.asarray(lower != upper)


# Primitives that select among their operands' values; a program using any
# of them tracks value identities so selections stay decided (`_ValueIdentity`).
_SELECTION_PRIMITIVES = frozenset(
    ("min", "max", "minimum", "maximum", "reduce_min", "reduce_max")
)


def _selects_values(jaxpr: jax_core.Jaxpr, /) -> bool:
    return any(
        equation.primitive.name in _SELECTION_PRIMITIVES
        or any(
            _selects_values(inner)
            for value in equation.params.values()
            for inner in _nested_jaxprs(value)
        )
        for equation in jaxpr.eqns
    )


@dataclass(frozen=True, slots=True)
class _Extremum:
    """Which operand values equal a min/max result on every point of the box.

    ``operands`` are the identities of the selected values (one reduced
    operand, or the two operands of an elementwise min/max); ``winners[i]``
    marks, in the shape of the comparison against the result, the elements
    of operand ``i`` that equal the result at every point, and ``decided``
    where the winner is known.
    """

    operands: tuple[int, ...]
    winners: tuple[NDArray[np.bool_], ...]
    decided: NDArray[np.bool_]
    # Reduced operand axes; empty for an elementwise min/max.
    axes: tuple[int, ...]


def _extremum_winners(
    minimum: bool,
    lower: NDArray[Any],
    upper: NDArray[Any],
    axes: tuple[int, ...],
    groups: tuple[int, ...] | None,
    /,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Elements equal to a reduction's extremum on the whole box, and where it is decided.

    The extremum is decided where one value strictly beats every other over
    the box. ``groups`` names equal values along a single reduced axis; a
    winning group's members all equal the extremum. Returns operand-shaped
    winners and keepdims-shaped decisions.
    """
    representative_lower, representative_upper = lower, upper
    order: tuple[int, ...] = ()
    if groups is not None:
        (axis,) = axes
        distinct = tuple(dict.fromkeys(groups))
        first = tuple(groups.index(group) for group in distinct)
        order = tuple(distinct.index(group) for group in groups)
        representative_lower = np.take(lower, first, axis=axis)
        representative_upper = np.take(upper, first, axis=axis)
    if minimum:
        bound = np.min(representative_upper, axis=axes, keepdims=True)
        candidates = representative_lower <= bound
    else:
        bound = np.max(representative_lower, axis=axes, keepdims=True)
        candidates = representative_upper >= bound
    decided = np.sum(candidates, axis=axes, keepdims=True) == 1
    winners = candidates & decided
    if groups is not None:
        winners = np.take(winners, order, axis=axes[0])
    return np.asarray(winners, dtype=np.bool_), np.asarray(decided, dtype=np.bool_)


class _ValueIdentity:
    """Real-value identities and decided selections across one host evaluation.

    Equations with the same primitive, static parameters and operand
    identities denote the same real value (every supported primitive is
    pure); nested calls take their operands' identities. A min/max whose
    winner is decided over the box equals that winner at every point, so a
    comparison of an operand against the result is decided: JAX derivative
    rules of min/max compare exactly so, and their tangents then stay finite.
    """

    def __init__(self) -> None:
        self._identities: dict[object, int] = {}
        self._representatives: dict[jax_core.Var, int] = {}
        self._stacks: dict[jax_core.Var, tuple[int, tuple[int, ...]]] = {}
        self._extrema: dict[jax_core.Var, _Extremum] = {}
        self._reductions: dict[jax_core.Var, _Extremum] = {}
        self._attainments: dict[jax_core.Var, _Extremum] = {}

    def _intern(self, key: object, /) -> int:
        return self._identities.setdefault(key, len(self._identities))

    def _fresh(self) -> int:
        return self._intern(("fresh", len(self._identities)))

    def atom(self, atom: object, /) -> int:
        if isinstance(atom, jax_core.Literal):
            value = np.asarray(atom.val)
            return self._intern(
                ("literal", value.dtype.str, value.shape, value.tobytes())
            )
        if not isinstance(atom, jax_core.Var):
            return self._fresh()
        representative = self._representatives.get(atom)
        if representative is None:
            representative = self._representatives[atom] = self._fresh()
        return representative

    def bind_inputs(self, variables: Sequence[jax_core.Var], /) -> None:
        for index, variable in enumerate(variables):
            self._representatives[variable] = self._intern(("input", index))

    def bind_constants(
        self, variables: Sequence[jax_core.Var], values: Sequence[Interval], /
    ) -> None:
        """Exact constants are equal by value; an enclosed constant only to itself."""
        for variable, value in zip(variables, values, strict=True):
            lower, upper = _endpoints(value)
            self._representatives[variable] = (
                self._intern(("constant", lower.dtype.str, lower.shape, lower.tobytes()))
                if lower.shape == upper.shape and np.array_equal(lower, upper)
                else self._fresh()
            )

    def bind_fresh(self, variables: Sequence[jax_core.Var], /) -> None:
        for variable in variables:
            self._representatives[variable] = self._fresh()

    def descend(self, inner: Sequence[jax_core.Var], outer: Sequence[object], /) -> None:
        """Give a nested jaxpr's inputs the identities of the call operands."""
        for variable, atom in zip(inner, outer, strict=True):
            self._representatives[variable] = self.atom(atom)

    def extremum(
        self, equation: jax_core.JaxprEqn, arguments: tuple[Interval, ...], /
    ) -> _Extremum | None:
        """Winner evidence of a min/max equation over its operand intervals."""
        name = equation.primitive.name
        if name in ("min", "max", "minimum", "maximum"):
            first, second = (self.atom(atom) for atom in equation.invars)
            first_lower, first_upper = _endpoints(arguments[0])
            second_lower, second_upper = _endpoints(arguments[1])
            if first == second:
                same = np.ones(
                    np.broadcast_shapes(first_lower.shape, second_lower.shape),
                    dtype=np.bool_,
                )
                return _Extremum((first, second), (same, same), same, ())
            if name in ("min", "minimum"):
                first_wins = first_upper < second_lower
                second_wins = second_upper < first_lower
            else:
                first_wins = first_lower > second_upper
                second_wins = second_lower > first_upper
            return _Extremum(
                (first, second),
                (np.asarray(first_wins), np.asarray(second_wins)),
                np.asarray(first_wins | second_wins),
                (),
            )
        if name not in ("reduce_min", "reduce_max", "argmin", "argmax"):
            return None
        axes = tuple(int(axis) for axis in equation.params["axes"])
        source = equation.invars[0]
        stacked = self._stacks.get(source) if isinstance(source, jax_core.Var) else None
        groups = stacked[1] if stacked is not None and axes == (stacked[0],) else None
        lower, upper = _endpoints(arguments[0])
        winners, decided = _extremum_winners(
            name in ("reduce_min", "argmin"), lower, upper, axes, groups
        )
        return _Extremum((self.atom(source),), (winners,), decided, axes)

    def identify(
        self, equation: jax_core.JaxprEqn, nested_outputs: Sequence[object] | None, /
    ) -> tuple[int, ...]:
        """Assign the outputs' identities from the operands'; return the operands'."""
        operands = tuple(self.atom(atom) for atom in equation.invars)
        if nested_outputs is not None:
            identities = tuple(self.atom(atom) for atom in nested_outputs)
        else:
            parameters = _parameter_key(dict(equation.params))
            identities = tuple(
                self._fresh()
                if parameters is _UNKEYED
                else self._intern((equation.primitive.name, operands, parameters, index))
                for index in range(len(equation.outvars))
            )
        for variable, identity in zip(equation.outvars, identities, strict=True):
            self._representatives[variable] = identity
        return operands

    def record(
        self,
        equation: jax_core.JaxprEqn,
        arguments: tuple[Interval, ...],
        outputs: tuple[Interval, ...],
        nested_outputs: Sequence[object] | None,
        /,
    ) -> tuple[tuple[int, ...], _Extremum | None]:
        """Assign the outputs' identities; return the operands' and any winners."""
        name = equation.primitive.name
        operands = self.identify(equation, nested_outputs)
        extremum = self.extremum(equation, arguments)
        if not equation.outvars:
            return operands, extremum
        result = equation.outvars[0]
        if name == "stack" and len(equation.outvars) == 1:
            rank = np.ndim(outputs[0][0])
            self._stacks[result] = (int(equation.params["axis"]) % rank, operands)
        elif extremum is not None and name in ("min", "max", "minimum", "maximum"):
            self._extrema[result] = extremum
        elif extremum is not None and name in ("reduce_min", "reduce_max"):
            self._reductions[result] = extremum
        elif (
            name == "convert_element_type"
            and isinstance(equation.invars[0], jax_core.Var)
            and equation.invars[0] in self._attainments
        ):
            self._attainments[result] = self._attainments[equation.invars[0]]
        elif name in ("reshape", "broadcast_in_dim") and isinstance(
            equation.invars[0], jax_core.Var
        ):
            reduced = self._reductions.get(equation.invars[0])
            if reduced is not None:
                # A view restoring the reduced axes with extent one (or the
                # operand's extent) compares with the operand element by element.
                operand_shape = reduced.winners[0].shape
                keepdims = reduced.decided.shape
                kept = tuple(
                    axis for axis in range(len(operand_shape)) if axis not in reduced.axes
                )
                shape = tuple(np.shape(outputs[0][0]))
                if (
                    name == "reshape"
                    and equation.params.get("dimensions") is None
                    and shape == keepdims
                ) or (
                    name == "broadcast_in_dim"
                    and tuple(
                        int(axis) for axis in equation.params["broadcast_dimensions"]
                    )
                    == kept
                    and len(shape) == len(operand_shape)
                    and all(
                        extent in (keepdims[axis], operand_shape[axis])
                        for axis, extent in enumerate(shape)
                    )
                ):
                    self._extrema[result] = reduced
        return operands, extremum

    def decide_comparison(
        self,
        equation: jax_core.JaxprEqn,
        arguments: tuple[Interval, ...],
        result: Interval,
        /,
    ) -> Interval:
        """Decide ``eq``/``ne`` of a min/max operand against its decided result."""
        name = equation.primitive.name
        if name not in ("eq", "ne") or len(equation.invars) != 2:
            return result
        lower, upper = _endpoints(result)
        for operand, selected in (equation.invars, tuple(reversed(equation.invars))):
            extremum = (
                self._extrema.get(selected)
                if isinstance(selected, jax_core.Var)
                else None
            )
            if extremum is None:
                continue
            identity = self.atom(operand)
            if identity not in extremum.operands:
                continue
            equal = extremum.winners[extremum.operands.index(identity)]
            if equal.shape != lower.shape:
                continue
            try:
                decided = np.broadcast_to(extremum.decided, lower.shape)
            except ValueError:
                continue
            if name == "eq" and extremum.axes and equation.outvars:
                self._attainments[equation.outvars[0]] = extremum
            truth = equal if name == "eq" else ~equal
            lower = np.where(decided, truth, lower)
            upper = np.where(decided, truth, upper)
        return lower, upper

    def bound_attainment(
        self, equation: jax_core.JaxprEqn, result: Interval, /
    ) -> Interval:
        """Count at least one attaining element in a sum of extremum indicators.

        A reduction attains its extremum at every point, so the indicators of
        ``operand == extremum`` summed over the reduced axes are at least one;
        JAX divides its min/max tangent by exactly this count.
        """
        source = equation.invars[0] if equation.invars else None
        extremum = (
            self._attainments.get(source) if isinstance(source, jax_core.Var) else None
        )
        if (
            equation.primitive.name != "reduce_sum"
            or extremum is None
            or tuple(int(axis) for axis in equation.params["axes"]) != extremum.axes
        ):
            return result
        lower, upper = _endpoints(result)
        return np.maximum(lower, 1.0), upper


class _BranchLedger:
    """Per-box decisions of the non-smooth operations of one host evaluation.

    A box is *smooth* when the traced map is one fixed composition of
    C-infinity primitives on a neighbourhood of the box: every min/max,
    absolute value and extremum reduction has a strict winner or a sign over
    the box, every selection index, comparison and other integer or Boolean
    result is decided, and square roots, reciprocals, logarithms and
    fractional powers see arguments bounded away from their singular points.
    A box is *continuous* when at least no discontinuous operation (an
    undecided selection, index or comparison, a pole or a nonfinite
    enclosure) is undecided; a kink of min/max/abs or a root at zero keeps
    continuity. Custom-JVP calls with a declared extension carry no declared
    regularity and count as discontinuous.

    Values that do not depend on the boxes (no vmapped axis) are constants of
    the point map and never make it non-smooth. Through `_ValueIdentity` a
    min/max or reduction over equal operands and a selection between equal
    cases is decided whatever their intervals.
    """

    def __init__(self, axes: Mapping[jax_core.Var, int | None], capacity: int, /) -> None:
        self.axes = axes
        self.capacity = capacity
        self.smooth = np.ones((capacity,), dtype=np.bool_)
        self.continuous = np.ones((capacity,), dtype=np.bool_)

    def _boxes(self, variable: object, mask: ArrayLike, /) -> NDArray[np.bool_] | None:
        flags = np.asarray(mask, dtype=np.bool_)
        if not np.any(flags):
            return None
        axis = (
            self.axes.get(variable, _UNATTRIBUTED_AXIS)
            if isinstance(variable, jax_core.Var)
            else _UNATTRIBUTED_AXIS
        )
        if axis is None:
            return None
        aval = variable.aval if isinstance(variable, jax_core.Var) else None
        if (
            axis == _UNATTRIBUTED_AXIS
            or not isinstance(aval, ShapedArray)
            or axis >= aval.ndim
            or aval.shape[axis] != self.capacity
        ):
            return np.ones((self.capacity,), dtype=np.bool_)
        try:
            spread = np.broadcast_to(flags, aval.shape)
        except ValueError:
            return np.ones((self.capacity,), dtype=np.bool_)
        return np.any(np.moveaxis(spread, axis, 0).reshape((self.capacity, -1)), axis=1)

    def kink(self, variable: object, mask: ArrayLike, /) -> None:
        """A continuous but possibly non-smooth operation is undecided."""
        boxes = self._boxes(variable, mask)
        if boxes is not None:
            self.smooth &= ~boxes

    def jump(self, variable: object, mask: ArrayLike, /) -> None:
        """A possibly discontinuous operation is undecided."""
        boxes = self._boxes(variable, mask)
        if boxes is not None:
            self.smooth &= ~boxes
            self.continuous &= ~boxes

    def inspect(
        self,
        equation: jax_core.JaxprEqn,
        arguments: tuple[Interval, ...],
        outputs: tuple[Interval, ...],
        operands: tuple[int, ...],
        extremum: _Extremum | None,
        /,
    ) -> None:
        name = equation.primitive.name
        result = equation.outvars[0] if equation.outvars else None
        match name:
            case "max" | "maximum" | "min" | "minimum" if extremum is not None:
                self.kink(result, ~extremum.decided)
            case "abs":
                self.kink(result, _meets_zero(arguments[0]))
            case "sqrt":
                self.kink(result, _not_positive(arguments[0]))
                self.jump(result, ~(_endpoints(arguments[0])[0] >= 0.0))
            case "rsqrt" | "log":
                self.jump(result, _not_positive(arguments[0]))
            case "div":
                self.jump(result, _meets_zero(arguments[1]))
            case "integer_pow":
                if int(equation.params["y"]) < 0:
                    self.jump(result, _meets_zero(arguments[0]))
            case "pow":
                exponent_lower, exponent_upper = _endpoints(arguments[1])
                integral = (
                    np.array_equal(exponent_lower, exponent_upper)
                    and exponent_lower.size > 0
                    and np.all(np.isfinite(exponent_lower))
                    and np.all(exponent_lower == np.round(exponent_lower))
                )
                if not integral:
                    self.kink(result, _not_positive(arguments[0]))
                    self.jump(result, ~(_endpoints(arguments[0])[0] >= 0.0))
                elif np.any(exponent_lower < 0.0):
                    self.jump(result, _meets_zero(arguments[0]))
            case "reduce_max" | "reduce_min" if extremum is not None:
                self.kink(result, ~np.squeeze(extremum.decided, axis=extremum.axes))
            case "argmax" | "argmin" if extremum is not None:
                self.jump(result, ~np.squeeze(extremum.decided, axis=extremum.axes))
            case "select_n" | "select":
                if len(set(operands[1:])) > 1:
                    self.jump(equation.invars[0], _undecided(arguments[0]))
            case "gather" | "scatter-add" | "dynamic_slice" | "dynamic_update_slice":
                for atom, value in zip(equation.invars[1:], arguments[1:], strict=True):
                    if np.asarray(value[0]).dtype.kind in "biu":
                        self.jump(atom, _undecided(value))
            case "stop_gradient":
                self.kink(result, _undecided(arguments[0]))
            case _:
                pass
        for variable, value in zip(equation.outvars, outputs, strict=True):
            lower, upper = _endpoints(value)
            if lower.dtype.kind in "biu":
                # A selection, index or comparison that the box leaves open.
                self.jump(variable, lower != upper)
            elif lower.dtype.kind == "f":
                self.jump(variable, ~(np.isfinite(lower) & np.isfinite(upper)))


def _evaluate_jaxpr(
    jaxpr: jax_core.Jaxpr,
    constants: tuple[Interval, ...],
    inputs: tuple[Interval, ...],
    tables: dict[tuple[bytes, int, int, int], _JetTable],
    /,
    values: _ValueIdentity | None = None,
    ledger: _BranchLedger | None = None,
) -> tuple[Interval, ...]:
    """Evaluate one jaxpr level; ``values`` decides selections, ``ledger`` needs it."""
    environment: dict[jax_core.Var, Interval] = {}
    identity = IntervalOperandIdentity()
    for variable, value in zip(jaxpr.constvars, constants, strict=True):
        environment[variable] = value
    for variable, value in zip(jaxpr.invars, inputs, strict=True):
        environment[variable] = value
    if values is not None:
        values.bind_constants(jaxpr.constvars, constants)
    for equation in jaxpr.eqns:
        nested_outputs: Sequence[object] | None = None
        identity.record(equation)
        arguments = tuple(_read_atom(atom, environment) for atom in equation.invars)
        name = equation.primitive.name
        jets = None
        if name == "custom_jvp_call":
            jets = _bspline_jet_equation(equation, arguments, tables)
            if jets is None:
                jets = _custom_extension_equation(equation, arguments)
        if jets is not None:
            outputs = (jets,)
            if ledger is not None:
                # A declared extension states an enclosure, not a regularity.
                ledger.jump(equation.outvars[0], np.ones((), dtype=np.bool_))
        elif name in ("pjit", "jit", "custom_jvp_call"):
            nested = equation.params.get(
                "call_jaxpr" if name == "custom_jvp_call" else "jaxpr"
            )
            if isinstance(nested, jax_core.ClosedJaxpr):
                inner = nested.jaxpr
                inner_constants = tuple(_as_interval(value) for value in nested.consts)
            elif isinstance(nested, jax_core.Jaxpr):
                inner, inner_constants = nested, ()
            else:
                raise IntervalExtensionError(
                    f"unsupported nested jaxpr for primitive {name!r}."
                )
            if values is not None:
                values.descend(inner.invars, equation.invars)
                nested_outputs = inner.outvars
            outputs = _evaluate_jaxpr(
                inner, inner_constants, arguments, tables, values, ledger
            )
        elif name == "cond":
            branches = tuple(equation.params["branches"])
            predicate = arguments[0]
            branch_index = np.asarray(predicate[0])
            if np.array_equal(predicate[0], predicate[1]) and branch_index.ndim == 0:
                selected = int(branch_index)
                if selected < 0 or selected >= len(branches):
                    raise IntervalExtensionError(
                        "conditional branch index is outside its branch table."
                    )
                branch = branches[selected]
                if values is not None:
                    values.descend(branch.jaxpr.invars, equation.invars[1:])
                    nested_outputs = branch.jaxpr.outvars
                outputs = _evaluate_jaxpr(
                    branch.jaxpr,
                    tuple(_as_interval(value) for value in branch.consts),
                    arguments[1:],
                    tables,
                    values,
                    ledger,
                )
            else:
                if ledger is not None:
                    ledger.jump(equation.invars[0], _undecided(predicate))
                if values is not None:
                    for branch in branches:
                        values.descend(branch.jaxpr.invars, equation.invars[1:])
                branch_outputs = tuple(
                    _evaluate_jaxpr(
                        branch.jaxpr,
                        tuple(_as_interval(value) for value in branch.consts),
                        arguments[1:],
                        tables,
                        values,
                        ledger,
                    )
                    for branch in branches
                )
                outputs = tuple(
                    _hull(tuple(row[index] for row in branch_outputs))
                    for index in range(len(branch_outputs[0]))
                )
        elif name == "scan":
            outputs = _evaluate_scan(equation, arguments, tables, values, ledger)
        elif identity.is_square(equation):
            outputs = (interval_square(arguments[0]),)
        else:
            outputs = _primitive(name, arguments, equation.params)
            if values is not None and name in ("eq", "ne"):
                outputs = (values.decide_comparison(equation, arguments, outputs[0]),)
            elif values is not None and name == "reduce_sum":
                outputs = (values.bound_attainment(equation, outputs[0]),)
        if len(outputs) != len(equation.outvars):
            raise IntervalExtensionError(
                f"primitive {name!r} returned the wrong interval arity."
            )
        for variable, value in zip(equation.outvars, outputs, strict=True):
            if not isinstance(variable, jax_core.DropVar):
                environment[variable] = value
        if values is not None:
            operands, extremum = values.record(
                equation, arguments, outputs, nested_outputs
            )
            if ledger is not None:
                ledger.inspect(equation, arguments, outputs, operands, extremum)
    return tuple(_read_atom(atom, environment) for atom in jaxpr.outvars)


def _evaluate_scan(
    equation: jax_core.JaxprEqn,
    arguments: tuple[Interval, ...],
    tables: dict[tuple[bytes, int, int, int], _JetTable],
    /,
    values: _ValueIdentity | None = None,
    ledger: _BranchLedger | None = None,
) -> tuple[Interval, ...]:
    """Unroll a fixed-length scan: each step is a sound enclosure of the last."""
    body = equation.params["jaxpr"]
    if "num_consts" in equation.params:
        constant_count = int(equation.params["num_consts"])
        carry_count = int(equation.params["num_carry"])
    else:
        # Flat-tree scan parameters group inputs as (consts, carry, xs).
        constant_group, carry_group, _ = equation.params["ft_in"].unpack()
        constant_count, carry_count = len(list(constant_group)), len(list(carry_group))
    length = int(equation.params["length"])
    if not isinstance(body, jax_core.ClosedJaxpr):
        raise IntervalExtensionError("unsupported nested jaxpr for primitive 'scan'.")
    body_constants = tuple(_as_interval(value) for value in body.consts)
    constants = arguments[:constant_count]
    carry = arguments[constant_count : constant_count + carry_count]
    scanned = arguments[constant_count + carry_count :]
    steps = range(length - 1, -1, -1) if equation.params["reverse"] else range(length)
    stacked: list[tuple[Interval, ...]] = []
    for step in steps:
        sliced = tuple(
            (np.asarray(lower)[step], np.asarray(upper)[step]) for lower, upper in scanned
        )
        if values is not None:
            values.bind_fresh(body.jaxpr.invars)
        result = _evaluate_jaxpr(
            body.jaxpr,
            body_constants,
            (*constants, *carry, *sliced),
            tables,
            values,
            ledger,
        )
        carry = result[:carry_count]
        stacked.append(result[carry_count:])
    if equation.params["reverse"]:
        stacked.reverse()
    outputs = list(carry)
    for index, variable in enumerate(equation.outvars[carry_count:]):
        if stacked:
            outputs.append(
                (
                    np.stack([np.asarray(row[index][0]) for row in stacked]),
                    np.stack([np.asarray(row[index][1]) for row in stacked]),
                )
            )
        else:
            aval = variable.aval
            if not isinstance(aval, ShapedArray):
                raise IntervalExtensionError("scan outputs must have array shapes.")
            empty = np.empty(aval.shape, dtype=np.float64)
            outputs.append((empty, empty))
    return tuple(outputs)


@dataclass(frozen=True, slots=True)
class PreparedIntervalFunction:
    """One immutable jaxpr interval extension with a scheduling-only batch size."""

    closed_jaxpr: jax_core.ClosedJaxpr
    input_dimension: int
    output_shape: tuple[int, ...]
    batch_capacity: int
    constant_intervals: tuple[Interval, ...]
    execution: IntervalExecution = "host"
    # The traced point map; branch evidence retraces it at a second batch size.
    function: Callable[[Array], Array] | None = field(
        default=None, repr=False, compare=False
    )
    native_program: NativeIntervalProgram | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # Exact span Bernstein tables of this program's B-spline jets, derived on
    # first use from its captured knots and kept for this program's lifetime.
    jet_tables: dict[tuple[bytes, int, int, int], _JetTable] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # Vmapped box axis of every traced variable, derived on first branch query.
    branch_axes: dict[jax_core.Var, int | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # The native program with branch-evidence events, lowered on first query.
    native_branch_program: NativeIntervalProgram | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # Whether a min/max selects among values: evaluation then tracks value
    # identities so a decided selection is decided in every comparison.
    selects_values: bool = field(init=False, repr=False, compare=False)
    execution_reason: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "selects_values", _selects_values(self.closed_jaxpr.jaxpr)
        )
        selected_execution = parse(self.execution, IntervalExecution, "execution")
        if selected_execution == "host":
            object.__setattr__(
                self, "execution_reason", "host interval reference was requested."
            )
            return
        object.__setattr__(self, "native_program", self._lower_native(None))
        object.__setattr__(
            self,
            "execution_reason",
            "meshcore advertised every opcode in the native interval program.",
        )

    def _lower_native(
        self, branch_axes: Mapping[jax_core.Var, int | None] | None, /
    ) -> NativeIntervalProgram:
        """Lower to an advertised native program, with ledger events given box axes."""
        from ._interval_native import lower_interval_jaxpr

        program = lower_interval_jaxpr(
            self.closed_jaxpr,
            self.constant_intervals,
            self.batch_capacity,
            self.input_dimension,
            branch_axes,
        )
        from .._meshcore import load_meshcore, MeshcoreUnavailableError

        try:
            supported_opcodes = load_meshcore().interval_program_capabilities()
        except MeshcoreUnavailableError as error:
            raise IntervalExtensionError(
                f"native interval execution requires the meshcore capability query: {error}"
            ) from error
        program.require_supported_opcodes(supported_opcodes)
        return program

    def _boxes(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        if (
            lower_.ndim != 2
            or lower_.shape[1] != self.input_dimension
            or upper_.shape != lower_.shape
        ):
            raise ValueError(
                f"interval boxes must have aligned shape (count, {self.input_dimension}); got {lower_.shape} and {upper_.shape}."
            )
        if (
            np.any(np.isnan(lower_))
            or np.any(np.isnan(upper_))
            or np.any(lower_ > upper_)
        ):
            raise ValueError("interval box endpoints must be ordered and non-NaN.")
        return lower_, upper_

    def _padded(
        self, values: NDArray[np.float64], start: int, stop: int, /
    ) -> NDArray[np.float64]:
        chunk = values[start:stop]
        pad = self.batch_capacity - (stop - start)
        if pad:
            chunk = np.concatenate((chunk, np.repeat(chunk[-1:], pad, axis=0)))
        return chunk

    def evaluate(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower_, upper_ = self._boxes(lower, upper)
        count = lower_.shape[0]
        low_result = np.empty((count, *self.output_shape), dtype=np.float64)
        high_result = np.empty_like(low_result)
        for start in range(0, count, self.batch_capacity):
            stop = min(start + self.batch_capacity, count)
            active = stop - start
            chunk_lower = self._padded(lower_, start, stop)
            chunk_upper = self._padded(upper_, start, stop)
            if self.native_program is None:
                values: _ValueIdentity | None = None
                if self.selects_values:
                    values = _ValueIdentity()
                    values.bind_inputs(self.closed_jaxpr.jaxpr.invars)
                (output,) = _evaluate_jaxpr(
                    self.closed_jaxpr.jaxpr,
                    self.constant_intervals,
                    ((chunk_lower, chunk_upper),),
                    self.jet_tables,
                    values,
                )
                chunk_low, chunk_high = output
            else:
                chunk_low, chunk_high = self.native_program.evaluate(
                    chunk_lower, chunk_upper
                )
            low_result[start:stop] = np.asarray(chunk_low)[:active]
            high_result[start:stop] = np.asarray(chunk_high)[:active]
        return low_result, high_result

    def evaluate_branches(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
    ]:
        """Enclosures with per-box branch decisions from the same evaluation.

        Returns ``(lower, upper, smooth, continuous)``. ``smooth[i]`` proves
        the traced map C-infinity on a neighbourhood of box ``i`` (see
        `_BranchLedger`), so derivative programs of the same map enclose its
        true derivatives there; it assumes custom-JVP rules are derivatives
        of their primal maps where those are smooth. ``continuous[i]`` proves
        the map continuous on box ``i``. The native route records the same
        ledger events in its program (``phx_mc_interval_program_branches``).
        """
        lower_, upper_ = self._boxes(lower, upper)
        axes = self._branch_batch_axes()
        count = lower_.shape[0]
        low_result = np.empty((count, *self.output_shape), dtype=np.float64)
        high_result = np.empty_like(low_result)
        smooth = np.empty((count,), dtype=np.bool_)
        continuous = np.empty((count,), dtype=np.bool_)
        jaxpr = self.closed_jaxpr.jaxpr
        native = None
        if self.native_program is not None:
            native = self.native_branch_program
            if native is None:
                native = self._lower_native(axes)
                object.__setattr__(self, "native_branch_program", native)
        for start in range(0, count, self.batch_capacity):
            stop = min(start + self.batch_capacity, count)
            active = stop - start
            chunk_lower = self._padded(lower_, start, stop)
            chunk_upper = self._padded(upper_, start, stop)
            if native is not None:
                output_low, output_high, chunk_smooth, chunk_continuous = (
                    native.evaluate_branches(chunk_lower, chunk_upper)
                )
            else:
                values = _ValueIdentity()
                values.bind_inputs(jaxpr.invars)
                ledger = _BranchLedger(axes, self.batch_capacity)
                ((output_low, output_high),) = _evaluate_jaxpr(
                    jaxpr,
                    self.constant_intervals,
                    ((chunk_lower, chunk_upper),),
                    self.jet_tables,
                    values,
                    ledger,
                )
                chunk_smooth, chunk_continuous = ledger.smooth, ledger.continuous
            low_result[start:stop] = np.asarray(output_low)[:active]
            high_result[start:stop] = np.asarray(output_high)[:active]
            smooth[start:stop] = chunk_smooth[:active]
            continuous[start:stop] = chunk_continuous[:active]
        return low_result, high_result, smooth, continuous

    def _branch_batch_axes(self) -> dict[jax_core.Var, int | None]:
        if not self.branch_axes:
            if self.function is None:
                raise IntervalExtensionError(
                    "Branch evidence requires the traced point map of the program."
                )
            probe = jax.ShapeDtypeStruct(
                (self.batch_capacity + 1, self.input_dimension), np.float64
            )
            other = jax.make_jaxpr(jax.vmap(self.function))(probe)
            self.branch_axes.update(
                _branch_batch_axes(
                    self.closed_jaxpr.jaxpr,
                    other.jaxpr,
                    (self.batch_capacity, self.batch_capacity + 1),
                )
            )
        return self.branch_axes

    def evaluate_host_points(self, points: ArrayLike, /) -> NDArray[np.float64]:
        points_ = np.asarray(points, dtype=np.float64)
        if (
            points_.ndim != 2
            or points_.shape[1] != self.input_dimension
            or not np.all(np.isfinite(points_))
        ):
            raise ValueError(
                f"interval points must be finite with shape (count, {self.input_dimension}); got {points_.shape}."
            )
        count = points_.shape[0]
        result = np.empty((count, *self.output_shape), dtype=np.float64)
        for start in range(0, count, self.batch_capacity):
            stop = min(start + self.batch_capacity, count)
            active = stop - start
            chunk = points_[start:stop]
            if active < self.batch_capacity:
                chunk = np.concatenate(
                    (chunk, np.repeat(chunk[-1:], self.batch_capacity - active, axis=0))
                )
            outputs = eval_jaxpr(
                self.closed_jaxpr.jaxpr,
                self.closed_jaxpr.consts,
                jax.device_put(chunk),
            )
            result[start:stop] = np.asarray(outputs[0])[:active]
        return result


def prepare_interval_function(
    function: Callable[[Array], Array],
    input_dimension: int,
    /,
    *,
    batch_capacity: int = 64,
    constant_bounds: Sequence[tuple[ArrayLike, ArrayLike, ArrayLike]] = (),
    execution: IntervalExecution = "host",
) -> PreparedIntervalFunction:
    """Trace a pure point map and prepare outward interval evaluation.

    ``host`` is the independent reference interpreter. ``native`` requires a
    complete lowering and reports unsupported primitives explicitly.
    """
    selected_execution = parse(execution, IntervalExecution, "execution")
    if input_dimension < 1 or batch_capacity < 1:
        raise ValueError("interval input_dimension and batch_capacity must be positive.")
    probe = jax.ShapeDtypeStruct((batch_capacity, input_dimension), np.float64)
    closed = jax.make_jaxpr(jax.vmap(function))(probe)
    if len(closed.jaxpr.outvars) != 1 or len(closed.jaxpr.invars) != 1:
        raise ValueError(
            "interval functions must have one array input and one array output."
        )
    output_aval = closed.jaxpr.outvars[0].aval
    if not isinstance(output_aval, ShapedArray) or output_aval.dtype.kind == "c":
        raise ValueError("interval functions must return one real shaped array.")
    if constant_bounds and len(constant_bounds) != len(closed.consts):
        raise ValueError("constant_bounds must align with every captured jaxpr constant.")
    intervals: list[Interval] = []
    if constant_bounds:
        for nominal, lower, upper in constant_bounds:
            nominal_ = np.asarray(nominal, dtype=np.float64)
            lower_ = np.asarray(lower, dtype=np.float64)
            upper_ = np.asarray(upper, dtype=np.float64)
            if (
                lower_.shape != nominal_.shape
                or upper_.shape != nominal_.shape
                or not np.all(np.isfinite(nominal_))
                or not np.all(np.isfinite(lower_))
                or not np.all(np.isfinite(upper_))
                or np.any(lower_ > nominal_)
                or np.any(nominal_ > upper_)
            ):
                raise ValueError(
                    "constant bounds must align and enclose finite nominal values."
                )
            intervals.append(_round_out(lower_, upper_))
    else:
        for constant_value in closed.consts:
            exact = np.asarray(constant_value, dtype=np.float64)
            if not np.all(np.isfinite(exact)):
                raise ValueError(
                    "captured interval constants must be finite real arrays."
                )
            intervals.append((exact, exact))
    return PreparedIntervalFunction(
        closed_jaxpr=closed,
        input_dimension=input_dimension,
        output_shape=tuple(output_aval.shape[1:]),
        batch_capacity=batch_capacity,
        constant_intervals=tuple(intervals),
        execution=selected_execution,
        function=function,
    )


__all__ = [
    "HostArray",
    "CustomIntervalExtension",
    "Interval",
    "IntervalExecution",
    "IntervalExtensionError",
    "PreparedIntervalFunction",
    "interval_add",
    "interval_cos",
    "interval_divide",
    "interval_exp",
    "interval_log",
    "interval_maximum",
    "interval_minimum",
    "interval_multiply",
    "interval_periodic_harmonic",
    "interval_sin",
    "interval_sqrt",
    "interval_square",
    "interval_subtract",
    "interval_tan",
    "prepare_interval_function",
    "register_custom_jvp_interval_extension",
]
