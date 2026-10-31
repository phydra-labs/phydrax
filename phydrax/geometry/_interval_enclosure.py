#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Outward-rounded interval evaluation for pure real JAX maps."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import jax
import numpy as np
from jax import Array
from jax.core import eval_jaxpr, ShapedArray
from jax.extend import core as jax_core
from jax.typing import ArrayLike
from numpy.typing import NDArray

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


def _hull(values: Sequence[Interval], /) -> Interval:
    if not values:
        raise ValueError("interval hull requires at least one value.")
    lower = np.minimum.reduce(tuple(np.asarray(value[0]) for value in values))
    upper = np.maximum.reduce(tuple(np.asarray(value[1]) for value in values))
    return lower, upper


def _sign_clamped(
    value: Interval, nonnegative: HostArray, nonpositive: HostArray, /
) -> Interval:
    # Outward rounding widens an exact zero bound across the origin. A result
    # whose sign is fixed by its operand signs keeps that sign, so a sum of
    # squares remains an admissible base for even roots and real powers.
    return (
        np.where(nonnegative, np.maximum(value[0], 0.0), value[0]),
        np.where(nonpositive, np.minimum(value[1], 0.0), value[1]),
    )


def interval_add(first: Interval, second: Interval, /) -> Interval:
    shape = np.broadcast_shapes(np.shape(first[0]), np.shape(second[0]))
    if np.all(np.asarray(second[0]) == 0) and np.all(np.asarray(second[1]) == 0):
        return np.broadcast_to(first[0], shape), np.broadcast_to(first[1], shape)
    if np.all(np.asarray(first[0]) == 0) and np.all(np.asarray(first[1]) == 0):
        return np.broadcast_to(second[0], shape), np.broadcast_to(second[1], shape)
    with np.errstate(invalid="ignore", over="ignore"):
        result = _round_out(np.add(first[0], second[0]), np.add(first[1], second[1]))
    return _sign_clamped(
        result,
        (np.asarray(first[0]) >= 0.0) & (np.asarray(second[0]) >= 0.0),
        (np.asarray(first[1]) <= 0.0) & (np.asarray(second[1]) <= 0.0),
    )


def interval_subtract(first: Interval, second: Interval, /) -> Interval:
    with np.errstate(invalid="ignore", over="ignore"):
        return _round_out(
            np.subtract(first[0], second[1]), np.subtract(first[1], second[0])
        )


def interval_multiply(first: Interval, second: Interval, /) -> Interval:
    with np.errstate(invalid="ignore", over="ignore"):
        products = (
            np.multiply(first[0], second[0]),
            np.multiply(first[0], second[1]),
            np.multiply(first[1], second[0]),
            np.multiply(first[1], second[1]),
        )
        lower = np.minimum.reduce(products)
        upper = np.maximum.reduce(products)
    invalid = np.logical_or.reduce(tuple(np.isnan(value) for value in products))
    result = _round_out(
        np.where(invalid, -np.inf, lower), np.where(invalid, np.inf, upper)
    )
    first_nonnegative = np.asarray(first[0]) >= 0.0
    first_nonpositive = np.asarray(first[1]) <= 0.0
    second_nonnegative = np.asarray(second[0]) >= 0.0
    second_nonpositive = np.asarray(second[1]) <= 0.0
    return _sign_clamped(
        result,
        (first_nonnegative & second_nonnegative)
        | (first_nonpositive & second_nonpositive),
        (first_nonnegative & second_nonpositive)
        | (first_nonpositive & second_nonnegative),
    )


def interval_divide(first: Interval, second: Interval, /) -> Interval:
    denominator_lower = np.asarray(second[0])
    denominator_upper = np.asarray(second[1])
    singular = (denominator_lower <= 0.0) & (denominator_upper >= 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        reciprocal_lower = np.divide(1.0, denominator_upper)
        reciprocal_upper = np.divide(1.0, denominator_lower)
        reciprocal = _round_out(reciprocal_lower, reciprocal_upper)
        result = interval_multiply(first, reciprocal)
    return (
        np.where(singular, -np.inf, result[0]),
        np.where(singular, np.inf, result[1]),
    )


def interval_square(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    with np.errstate(invalid="ignore", over="ignore"):
        first, second = np.square(lower), np.square(upper)
    crosses = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(crosses, 0.0, np.minimum(first, second))
    high = np.maximum(first, second)
    rounded = _round_out(low, high)
    return np.maximum(rounded[0], 0.0), rounded[1]


def interval_sqrt(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    with np.errstate(invalid="ignore", over="ignore"):
        low = np.sqrt(np.maximum(lower, 0.0))
        high = np.sqrt(upper)
    low = np.where(lower < 0.0, 0.0, low)
    high = np.where(upper < 0.0, np.nan, high)
    return _round_out(low, high)


def interval_exp(value: Interval, /) -> Interval:
    with np.errstate(over="ignore", invalid="ignore"):
        return _round_out(np.exp(value[0]), np.exp(value[1]))


def interval_log(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        low = np.log(lower)
        high = np.log(upper)
    low = np.where(lower <= 0.0, -np.inf, low)
    high = np.where(upper <= 0.0, np.inf, high)
    return _round_out(low, high)


def _trigonometric_interval(value: Interval, kind: str, /) -> Interval:
    lower, upper = (
        np.asarray(value[0], dtype=np.float64),
        np.asarray(value[1], dtype=np.float64),
    )
    width = upper - lower
    with np.errstate(invalid="ignore"):
        representable_step = np.maximum(
            np.spacing(np.abs(lower)), np.spacing(np.abs(upper))
        )
    wide = (
        ~np.isfinite(lower)
        | ~np.isfinite(upper)
        | (width >= 2.0 * math.pi)
        | (representable_step >= 0.5 * math.pi)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        if kind == "sin":
            low = np.minimum(np.sin(lower), np.sin(upper))
            high = np.maximum(np.sin(lower), np.sin(upper))
            maximum = np.ceil((lower - 0.5 * math.pi) / (2.0 * math.pi))
            minimum = np.ceil((lower + 0.5 * math.pi) / (2.0 * math.pi))
            high = np.where(0.5 * math.pi + maximum * (2.0 * math.pi) <= upper, 1.0, high)
            low = np.where(-0.5 * math.pi + minimum * (2.0 * math.pi) <= upper, -1.0, low)
        elif kind == "cos":
            low = np.minimum(np.cos(lower), np.cos(upper))
            high = np.maximum(np.cos(lower), np.cos(upper))
            maximum = np.ceil(lower / (2.0 * math.pi))
            minimum = np.ceil((lower - math.pi) / (2.0 * math.pi))
            high = np.where(maximum * (2.0 * math.pi) <= upper, 1.0, high)
            low = np.where(math.pi + minimum * (2.0 * math.pi) <= upper, -1.0, low)
        else:
            raise ValueError(f"unknown trigonometric interval function {kind!r}.")
    low = np.where(wide, -1.0, low)
    high = np.where(wide, 1.0, high)
    return _round_out(low, high)


def interval_sin(value: Interval, /) -> Interval:
    return _trigonometric_interval(value, "sin")


def interval_cos(value: Interval, /) -> Interval:
    return _trigonometric_interval(value, "cos")


def interval_tan(value: Interval, /) -> Interval:
    lower, upper = (
        np.asarray(value[0], dtype=np.float64),
        np.asarray(value[1], dtype=np.float64),
    )
    with np.errstate(invalid="ignore"):
        resolution = np.maximum(np.spacing(np.abs(lower)), np.spacing(np.abs(upper)))
    first_pole = np.ceil((lower - 0.5 * math.pi) / math.pi)
    pole = 0.5 * math.pi + first_pole * math.pi
    crosses = (
        ~np.isfinite(lower)
        | ~np.isfinite(upper)
        | (resolution >= 0.5 * math.pi)
        | (pole <= upper)
    )
    with np.errstate(invalid="ignore", over="ignore"):
        low, high = np.tan(lower), np.tan(upper)
    return _round_out(np.where(crosses, -np.inf, low), np.where(crosses, np.inf, high))


def interval_absolute(value: Interval, /) -> Interval:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    crosses = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(crosses, 0.0, np.minimum(np.abs(lower), np.abs(upper)))
    high = np.maximum(np.abs(lower), np.abs(upper))
    return _round_out(low, high)


def interval_maximum(first: Interval, second: Interval, /) -> Interval:
    return np.maximum(first[0], second[0]), np.maximum(first[1], second[1])


def interval_minimum(first: Interval, second: Interval, /) -> Interval:
    return np.minimum(first[0], second[0]), np.minimum(first[1], second[1])


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
        next_lower = np.where(covered, np.minimum(lower, case[0]), case[0])
        next_upper = np.where(covered, np.maximum(upper, case[1]), case[1])
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
    index_lower, index_upper = _integer_index_interval(start_indices, 1)
    dimensions = params["dimension_numbers"]
    start_index_map = tuple(dimensions.start_index_map)
    if (
        operand_lower.shape != operand_upper.shape
        or len(start_index_map) != 1
        or tuple(dimensions.collapsed_slice_dims) != start_index_map
        or dimensions.operand_batching_dims
        or dimensions.start_indices_batching_dims
        or params["mode"] != jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS
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
        dimension for dimension in range(operand_lower.ndim) if dimension != axis
    )
    output_shape = (
        *index_shape,
        *(operand_lower.shape[dimension] for dimension in remaining_axes),
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
    if np.any(integer_lower < 0) or np.any(integer_upper >= extent):
        raise IntervalExtensionError("gather index interval is outside the operand.")
    integer_lower = integer_lower.astype(np.intp)
    integer_upper = integer_upper.astype(np.intp)
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
            covered_mask, np.minimum(result_lower, selected_lower), selected_lower
        )
        next_upper = np.where(
            covered_mask, np.maximum(result_upper, selected_upper), selected_upper
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
                        np.minimum(update_lower, 0.0),
                        np.maximum(update_upper, 0.0),
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
            return (_round_out(np.arctan(values[0][0]), np.arctan(values[0][1])),)
        case "tanh":
            return (_round_out(np.tanh(values[0][0]), np.tanh(values[0][1])),)
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
            if np.array_equal(exponent[0], exponent[1]) and np.all(
                np.equal(exponent[0], np.floor(exponent[0]))
            ):
                selected = np.asarray(exponent[0])
                if selected.ndim == 0:
                    return _primitive("integer_pow", (values[0],), {"y": int(selected)})
            if np.any(np.asarray(values[0][0]) < 0.0):
                raise IntervalExtensionError(
                    "noninteger power requires a nonnegative base interval."
                )
            # log extends 0 to -inf, so exp(y log x) encloses 0**y for every
            # exponent sign; a nonnegative base has a nonnegative real power.
            power = interval_exp(interval_multiply(exponent, interval_log(values[0])))
            return ((np.maximum(power[0], 0.0), power[1]),)
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
        case "reduce_max" | "reduce_min":
            axes = tuple(params["axes"])
            if name == "reduce_max":
                return (
                    (np.max(values[0][0], axis=axes), np.max(values[0][1], axis=axes)),
                )
            return ((np.min(values[0][0], axis=axes), np.min(values[0][1], axis=axes)),)
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


def _evaluate_jaxpr(
    jaxpr: jax_core.Jaxpr,
    constants: tuple[Interval, ...],
    inputs: tuple[Interval, ...],
    /,
) -> tuple[Interval, ...]:
    environment: dict[jax_core.Var, Interval] = {}
    identity = IntervalOperandIdentity()
    for variable, value in zip(jaxpr.constvars, constants, strict=True):
        environment[variable] = value
    for variable, value in zip(jaxpr.invars, inputs, strict=True):
        environment[variable] = value
    for equation in jaxpr.eqns:
        identity.record(equation)
        arguments = tuple(_read_atom(atom, environment) for atom in equation.invars)
        name = equation.primitive.name
        if name in ("pjit", "jit", "custom_jvp_call"):
            nested = equation.params.get(
                "call_jaxpr" if name == "custom_jvp_call" else "jaxpr"
            )
            if isinstance(nested, jax_core.ClosedJaxpr):
                outputs = _evaluate_jaxpr(
                    nested.jaxpr,
                    tuple(_as_interval(value) for value in nested.consts),
                    arguments,
                )
            elif isinstance(nested, jax_core.Jaxpr):
                outputs = _evaluate_jaxpr(nested, (), arguments)
            else:
                raise IntervalExtensionError(
                    f"unsupported nested jaxpr for primitive {name!r}."
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
                outputs = _evaluate_jaxpr(
                    branch.jaxpr,
                    tuple(_as_interval(value) for value in branch.consts),
                    arguments[1:],
                )
            else:
                branch_outputs = tuple(
                    _evaluate_jaxpr(
                        branch.jaxpr,
                        tuple(_as_interval(value) for value in branch.consts),
                        arguments[1:],
                    )
                    for branch in branches
                )
                outputs = tuple(
                    _hull(tuple(row[index] for row in branch_outputs))
                    for index in range(len(branch_outputs[0]))
                )
        elif name == "scan":
            outputs = _evaluate_scan(equation, arguments)
        elif identity.is_square(equation):
            outputs = (interval_square(arguments[0]),)
        else:
            outputs = _primitive(name, arguments, equation.params)
        if len(outputs) != len(equation.outvars):
            raise IntervalExtensionError(
                f"primitive {name!r} returned the wrong interval arity."
            )
        for variable, value in zip(equation.outvars, outputs, strict=True):
            if not isinstance(variable, jax_core.DropVar):
                environment[variable] = value
    return tuple(_read_atom(atom, environment) for atom in jaxpr.outvars)


def _evaluate_scan(
    equation: jax_core.JaxprEqn, arguments: tuple[Interval, ...], /
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
        result = _evaluate_jaxpr(
            body.jaxpr, body_constants, (*constants, *carry, *sliced)
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
    native_program: NativeIntervalProgram | None = field(
        default=None, init=False, repr=False, compare=False
    )
    execution_reason: str = field(init=False)

    def __post_init__(self) -> None:
        selected_execution = parse(self.execution, IntervalExecution, "execution")
        if selected_execution == "host":
            object.__setattr__(
                self, "execution_reason", "host interval reference was requested."
            )
            return
        from ._interval_native import lower_interval_jaxpr

        program = lower_interval_jaxpr(
            self.closed_jaxpr,
            self.constant_intervals,
            self.batch_capacity,
            self.input_dimension,
        )
        from .._meshcore import load_meshcore, MeshcoreUnavailableError

        try:
            supported_opcodes = load_meshcore().interval_program_capabilities()
        except MeshcoreUnavailableError as error:
            raise IntervalExtensionError(
                f"native interval execution requires the meshcore capability query: {error}"
            ) from error
        program.require_supported_opcodes(supported_opcodes)
        object.__setattr__(self, "native_program", program)
        object.__setattr__(
            self,
            "execution_reason",
            "meshcore advertised every opcode in the native interval program.",
        )

    def evaluate(
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
        count = lower_.shape[0]
        low_result = np.empty((count, *self.output_shape), dtype=np.float64)
        high_result = np.empty_like(low_result)
        for start in range(0, count, self.batch_capacity):
            stop = min(start + self.batch_capacity, count)
            active = stop - start
            pad = self.batch_capacity - active
            chunk_lower = lower_[start:stop]
            chunk_upper = upper_[start:stop]
            if pad:
                chunk_lower = np.concatenate(
                    (chunk_lower, np.repeat(chunk_lower[-1:], pad, axis=0))
                )
                chunk_upper = np.concatenate(
                    (chunk_upper, np.repeat(chunk_upper[-1:], pad, axis=0))
                )
            if self.native_program is None:
                (output,) = _evaluate_jaxpr(
                    self.closed_jaxpr.jaxpr,
                    self.constant_intervals,
                    ((chunk_lower, chunk_upper),),
                )
                chunk_low, chunk_high = output
            else:
                chunk_low, chunk_high = self.native_program.evaluate(
                    chunk_lower, chunk_upper
                )
            low_result[start:stop] = np.asarray(chunk_low)[:active]
            high_result[start:stop] = np.asarray(chunk_high)[:active]
        return low_result, high_result

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
    )


__all__ = [
    "HostArray",
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
    "interval_sin",
    "interval_sqrt",
    "interval_square",
    "interval_subtract",
    "interval_tan",
    "prepare_interval_function",
]
