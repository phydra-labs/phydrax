#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Outward-rounded interval enclosures of pure JAX maps by jaxpr interpretation.

A pure ``jax.numpy`` map ``R^n -> R^m`` is traced once over a fixed-capacity batch
of points and its jaxpr is re-interpreted over interval arrays. Structural
primitives act on both interval bounds; arithmetic and elementary functions use
natural interval extensions with directed outward rounding. Arithmetic bounds
assume IEEE round-to-nearest operations; transcendental bounds additionally
assume the platform elementary functions are accurate within four ulps.
This libm assumption is conditional on the platform implementation. Bounded
subdivision and Krawczyk certification consume these conditional enclosures.

Fixed-length scans retain their recurrence order, finite conditional branches
are enclosed by their hull, and uncertain gather indices retain the operand
hull. Primitives without a sound extension and unbounded control flow are
refused. ``PreparedIntervalFunction.evaluate_host_points`` interprets the same
prepared program with NumPy nominal arithmetic; ``device_interval_evaluation_error``
retains JAX numerical leaves and admits known input intervals for rounding error.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.core import ShapedArray
from jax.extend import core as jax_core

from ..typing import checked, Float, VariadicDim


type HostTensor = npt.NDArray[np.number | np.bool_]
type HostArray = HostTensor | np.number | np.bool_
type Interval = tuple[HostArray, HostArray]
type DotDimensions = tuple[
    tuple[Sequence[int], Sequence[int]],
    tuple[Sequence[int], Sequence[int]],
]


class _IntervalPointDims(VariadicDim):
    """Shape of one numerical input to a device interval evaluation."""


def _array_aval(variable: jax_core.Var | jax_core.Literal, /) -> ShapedArray:
    match variable.aval:
        case ShapedArray() as aval:
            return aval
        case aval:
            raise IntervalExtensionError(
                f"interval.output must have a shaped array abstract value; got {type(aval).__qualname__}"
            )


class IntervalExtensionError(ValueError):
    """A traced map contains an operation without a sound interval extension."""


_TWO_PI = 2.0 * math.pi
# One nextafter step covers correctly rounded basic operations. Elementary
# functions assume libm accuracy within four ulps; this assumption is conditional
# on the platform implementation.
_TRANSCENDENTAL_STEPS = 4


def _floating(value: HostArray, /) -> bool:
    return np.issubdtype(value.dtype, np.floating)


def _round_out(lower: HostArray, upper: HostArray, steps: int = 1, /) -> Interval:
    if not _floating(lower):
        return lower, upper
    for _ in range(steps):
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
    return lower, upper


def _clean(lower: HostArray, upper: HostArray, /) -> Interval:
    """Replace NaN bounds, which arise from ``0 * inf``, by the unbounded hull."""
    if not _floating(lower):
        return lower, upper
    return np.where(np.isnan(lower), -np.inf, lower), np.where(
        np.isnan(upper), np.inf, upper
    )


def _replace_proved_zero(value: HostArray, zero: HostArray, /) -> HostArray:
    """Retain the NumPy tensor/scalar carrier while applying a proved zero identity."""
    replaced = np.where(zero, np.zeros_like(value), value)
    match value:
        case np.ndarray():
            return replaced
        case _:
            return replaced[()]


def _degenerate(value: Interval, /) -> bool:
    return bool(np.array_equal(value[0], value[1]))


def interval_add(first: Interval, second: Interval, /) -> Interval:
    return _round_out(np.add(first[0], second[0]), np.add(first[1], second[1]))


def interval_subtract(first: Interval, second: Interval, /) -> Interval:
    lower, upper = _round_out(
        np.subtract(first[0], second[1]), np.subtract(first[1], second[0])
    )
    zero = (
        (first[0] == first[1])
        & (second[0] == second[1])
        & (first[0] == second[0])
        & np.isfinite(first[0])
    )
    return _replace_proved_zero(lower, zero), _replace_proved_zero(upper, zero)


def interval_multiply(first: Interval, second: Interval, /) -> Interval:
    products = np.stack(
        np.broadcast_arrays(
            np.multiply(first[0], second[0]),
            np.multiply(first[0], second[1]),
            np.multiply(first[1], second[0]),
            np.multiply(first[1], second[1]),
        )
    )
    products = np.where(np.isnan(products), 0.0, products)
    return _round_out(np.min(products, axis=0), np.max(products, axis=0))


def interval_dot(first: Interval, second: Interval, /) -> Interval:
    """Enclose a vector dot product in left-to-right product accumulation order."""
    products = interval_multiply(first, second)
    lower = upper = np.asarray(0.0, dtype=np.float64)
    for low, high in zip(np.asarray(products[0]), np.asarray(products[1]), strict=True):
        lower, upper = interval_add((lower, upper), (low, high))
    return lower, upper


def _reciprocal(value: Interval, /) -> Interval:
    lower, upper = value
    straddles = (lower <= 0.0) & (upper >= 0.0)
    with np.errstate(divide="ignore"):
        low = np.where(straddles, -np.inf, np.divide(1.0, upper))
        high = np.where(straddles, np.inf, np.divide(1.0, lower))
    return _round_out(low, high)


def interval_divide(first: Interval, second: Interval, /) -> Interval:
    lower, upper = interval_multiply(first, _reciprocal(second))
    undefined = (second[0] <= 0) & (second[1] >= 0)
    return np.where(undefined, -np.inf, lower), np.where(undefined, np.inf, upper)


def _integer_power(value: Interval, exponent: int, /) -> Interval:
    lower, upper = value
    if exponent == 0:
        return np.ones_like(lower), np.ones_like(upper)
    if exponent < 0:
        return _reciprocal(_integer_power(value, -exponent))
    low_power = np.power(lower, exponent)
    high_power = np.power(upper, exponent)
    if exponent % 2 == 1:
        return _round_out(low_power, high_power, 2)
    straddles = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(straddles, 0.0, np.minimum(low_power, high_power))
    high = np.maximum(low_power, high_power)
    return _round_out(low, high, 2)


def _contains_phase(lower: HostArray, upper: HostArray, phase: float, /) -> HostArray:
    """Whether ``phase + 2 k pi`` may lie in ``[lower, upper]`` for an integer ``k``.

    The test is conservative against the floating representation of ``pi``.
    """
    slack = 8.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(lower))
    index = np.ceil((lower - slack - phase) / _TWO_PI)
    return phase + index * _TWO_PI <= upper + slack


def interval_cos(value: Interval, /) -> Interval:
    lower, upper = value
    wide = (
        ~(np.subtract(upper, lower) < _TWO_PI) | ~np.isfinite(lower) | ~np.isfinite(upper)
    )
    endpoints = np.stack((np.cos(lower), np.cos(upper)))
    low = np.min(endpoints, axis=0)
    high = np.max(endpoints, axis=0)
    high = np.where(_contains_phase(lower, upper, 0.0) | wide, 1.0, high)
    low = np.where(_contains_phase(lower, upper, math.pi) | wide, -1.0, low)
    low, high = _round_out(low, high, _TRANSCENDENTAL_STEPS)
    return np.maximum(low, -1.0), np.minimum(high, 1.0)


def interval_sin(value: Interval, /) -> Interval:
    lower, upper = value
    wide = (
        ~(np.subtract(upper, lower) < _TWO_PI) | ~np.isfinite(lower) | ~np.isfinite(upper)
    )
    endpoints = np.stack((np.sin(lower), np.sin(upper)))
    low = np.min(endpoints, axis=0)
    high = np.max(endpoints, axis=0)
    high = np.where(_contains_phase(lower, upper, 0.5 * math.pi) | wide, 1.0, high)
    low = np.where(_contains_phase(lower, upper, -0.5 * math.pi) | wide, -1.0, low)
    low, high = _round_out(low, high, _TRANSCENDENTAL_STEPS)
    return np.maximum(low, -1.0), np.minimum(high, 1.0)


def interval_tan(value: Interval, /) -> Interval:
    lower, upper = value
    same_branch = np.floor(np.add(lower, 0.5 * math.pi) / math.pi) == np.floor(
        np.add(upper, 0.5 * math.pi) / math.pi
    )
    interior = (
        same_branch
        & (np.abs(np.cos(lower)) > 1.0e-300)
        & (np.abs(np.cos(upper)) > 1.0e-300)
    )
    low = np.where(interior, np.tan(lower), -np.inf)
    high = np.where(interior, np.tan(upper), np.inf)
    return _round_out(low, high, _TRANSCENDENTAL_STEPS)


def monotone(
    function: Callable[[HostArray], HostArray], increasing: bool, steps: int, /
) -> Callable[[Interval], Interval]:
    def apply(value: Interval) -> Interval:
        with np.errstate(invalid="ignore", divide="ignore"):
            first = function(value[0])
            second = function(value[1])
        low, high = (first, second) if increasing else (second, first)
        return _clean(*_round_out(low, high, steps))

    return apply


def _interval_sqrt(value: Interval, /) -> Interval:
    if np.any(value[1] < 0.0):
        raise IntervalExtensionError(
            "interval.sqrt.upper must admit a nonnegative real value; got strictly negative bounds"
        )
    return monotone(np.sqrt, True, 1)((np.maximum(value[0], 0.0), value[1]))


def _interval_abs(value: Interval, /) -> Interval:
    lower, upper = value
    straddles = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(straddles, 0.0, np.minimum(np.abs(lower), np.abs(upper)))
    return low, np.maximum(np.abs(lower), np.abs(upper))


def interval_midpoint_radius(value: Interval, /) -> Interval:
    """Return the midpoint and outward radius in the source addition order."""
    lower, upper = value
    middle = 0.5 * np.add(lower, upper)
    radius = np.nextafter(np.maximum(middle - lower, upper - middle), np.inf)
    return middle, radius


def interval_dot_general(
    first: Interval, second: Interval, dimension_numbers: DotDimensions, /
) -> Interval:
    """Midpoint-radius enclosure of a general tensor contraction."""
    first_mid, first_rad = interval_midpoint_radius(first)
    second_mid, second_rad = interval_midpoint_radius(second)

    def contract(left: HostArray, right: HostArray) -> HostArray:
        (left_contract, right_contract), (left_batch, right_batch) = dimension_numbers
        left_labels = list(range(left.ndim))
        right_labels = list(range(left.ndim, left.ndim + right.ndim))
        for a, b in zip(left_contract, right_contract, strict=True):
            right_labels[b] = left_labels[a]
        for a, b in zip(left_batch, right_batch, strict=True):
            right_labels[b] = left_labels[a]
        output_labels = (
            [left_labels[axis] for axis in left_batch]
            + [
                left_labels[axis]
                for axis in range(left.ndim)
                if axis not in left_contract and axis not in left_batch
            ]
            + [
                right_labels[axis]
                for axis in range(right.ndim)
                if axis not in right_contract and axis not in right_batch
            ]
        )
        return np.einsum(left, left_labels, right, right_labels, output_labels)

    middle = contract(first_mid, second_mid)
    magnitude = contract(np.abs(first_mid), np.abs(second_mid))
    radius = np.add(
        np.add(
            contract(np.abs(first_mid), second_rad),
            contract(first_rad, np.abs(second_mid)),
        ),
        contract(first_rad, second_rad),
    )
    (contracting, _), _ = dimension_numbers
    length = 1 + math.prod(first_mid.shape[axis] for axis in contracting)
    unit = np.finfo(middle.dtype.str).eps
    # Standard a-priori bound for a floating dot product of the given length,
    # plus the rounding of the radius accumulation itself.
    gamma = 2.0 * length * unit / (1.0 - 2.0 * length * unit)
    # JAX backends may flush subnormal products; include the normal-threshold
    # allowance rather than assuming gradual underflow in a tensor contraction.
    tiny = np.finfo(middle.dtype.str).tiny
    radius = np.nextafter(
        radius + gamma * (magnitude + radius) + 4 * length * tiny, np.inf
    )
    return _clean(*_round_out(middle - radius, middle + radius))


def _reduce_sum(value: Interval, axes: Sequence[int], /) -> Interval:
    axes_ = tuple(int(axis) for axis in axes)
    lower = np.sum(value[0], axis=axes_)
    upper = np.sum(value[1], axis=axes_)
    if not _floating(lower):
        return lower, upper
    length = max(1, math.prod(value[0].shape[axis] for axis in axes_))
    unit = np.finfo(lower.dtype.str).eps
    gamma = length * unit / (1.0 - length * unit)
    scale = np.sum(np.maximum(np.abs(value[0]), np.abs(value[1])), axis=axes_)
    return _round_out(lower - gamma * scale, upper + gamma * scale)


def _host_total_order_key(value: HostArray, /) -> HostArray:
    """Map IEEE floating encodings to their StableHLO total-order integer keys."""
    if not np.issubdtype(value.dtype, np.floating):
        return value
    match value.dtype.itemsize:
        case 2:
            unsigned = np.dtype(np.uint16)
        case 4:
            unsigned = np.dtype(np.uint32)
        case 8:
            unsigned = np.dtype(np.uint64)
        case size:
            raise IntervalExtensionError(
                f"interval.total_order.dtype must use a supported IEEE width; got {size} bytes"
            )
    array = np.asarray(value)
    native = array.astype(array.dtype.newbyteorder("="), copy=False)
    bits = native.view(unsigned)
    sign = np.asarray(1 << (8 * unsigned.itemsize - 1), dtype=unsigned)
    key = np.where(
        np.bitwise_and(bits, sign) != 0, np.invert(bits), np.bitwise_or(bits, sign)
    )
    return key if isinstance(value, np.ndarray) else key[()]


def _device_total_order_key(value: Array, /) -> Array:
    if not jnp.issubdtype(value.dtype, jnp.floating):
        return value
    match value.dtype.itemsize:
        case 2:
            unsigned = jnp.uint16
        case 4:
            unsigned = jnp.uint32
        case 8:
            unsigned = jnp.uint64
        case size:
            raise IntervalExtensionError(
                f"interval.total_order.dtype must use a supported IEEE width; got {size} bytes"
            )
    bits = jax.lax.bitcast_convert_type(value, unsigned)
    sign = jnp.asarray(1 << (8 * value.dtype.itemsize - 1), dtype=unsigned)
    return jnp.where((bits & sign) != 0, ~bits, bits | sign)


def _compare(name: str, first: Interval, second: Interval, /) -> Interval:
    """Three-valued comparison: lower bound is 'certainly', upper is 'possibly'."""
    match name:
        case "lt":
            return first[1] < second[0], first[0] < second[1]
        case "le":
            return first[1] <= second[0], first[0] <= second[1]
        case "gt":
            return first[0] > second[1], first[1] > second[0]
        case "ge":
            return first[0] >= second[1], first[1] >= second[0]
        case "eq":
            certain = (
                (first[0] == first[1])
                & (second[0] == second[1])
                & (first[0] == second[0])
            )
            possible = (first[0] <= second[1]) & (second[0] <= first[1])
            return certain, possible
        case "ne":
            equal_certain, equal_possible = _compare("eq", first, second)
            return np.logical_not(equal_possible), np.logical_not(equal_certain)
        case _:
            raise IntervalExtensionError(
                f"interval.comparison must be supported; got {name!r}"
            )


def _select(predicate: Interval, cases: Sequence[Interval], /) -> Interval:
    low_index, high_index = predicate
    low_index = np.asarray(low_index).astype(np.int64)
    high_index = np.asarray(high_index).astype(np.int64)
    lowers = np.stack(np.broadcast_arrays(*(case[0] for case in cases)))
    uppers = np.stack(np.broadcast_arrays(*(case[1] for case in cases)))
    count = len(cases)
    choices = np.arange(count).reshape((count,) + (1,) * (lowers.ndim - 1))
    active = (choices >= low_index) & (choices <= high_index)
    low = np.min(np.where(active, lowers, np.inf), axis=0)
    high = np.max(np.where(active, uppers, -np.inf), axis=0)
    if lowers.dtype == np.bool_:
        return low.astype(np.bool_), high.astype(np.bool_)
    return low.astype(lowers.dtype), high.astype(uppers.dtype)


_STRUCTURAL = frozenset(
    {
        "broadcast_in_dim",
        "reshape",
        "squeeze",
        "expand_dims",
        "slice",
        "transpose",
        "rev",
        "copy",
        "copy_p",
        "concatenate",
        "split",
        "stack",
        "real",
    }
)

_UNARY: dict[str, Callable[[Interval], Interval]] = {
    "sin": interval_sin,
    "cos": interval_cos,
    "tan": interval_tan,
    "sqrt": _interval_sqrt,
    "abs": _interval_abs,
    "exp": monotone(np.exp, True, _TRANSCENDENTAL_STEPS),
    "log": monotone(np.log, True, _TRANSCENDENTAL_STEPS),
    "atan": monotone(np.arctan, True, _TRANSCENDENTAL_STEPS),
    "tanh": monotone(np.tanh, True, _TRANSCENDENTAL_STEPS),
    "floor": monotone(np.floor, True, 0),
    "ceil": monotone(np.ceil, True, 0),
    "sign": monotone(np.sign, True, 0),
    "rsqrt": monotone(lambda value: 1.0 / np.sqrt(value), False, 2),
    "sinh": monotone(np.sinh, True, _TRANSCENDENTAL_STEPS),
    "expm1": monotone(np.expm1, True, _TRANSCENDENTAL_STEPS),
    "log1p": monotone(np.log1p, True, _TRANSCENDENTAL_STEPS),
    "asin": monotone(np.arcsin, True, _TRANSCENDENTAL_STEPS),
    "acos": monotone(np.arccos, False, _TRANSCENDENTAL_STEPS),
    "asinh": monotone(np.arcsinh, True, _TRANSCENDENTAL_STEPS),
    "acosh": monotone(np.arccosh, True, _TRANSCENDENTAL_STEPS),
    "atanh": monotone(np.arctanh, True, _TRANSCENDENTAL_STEPS),
}


def _host_gather(
    equation: jax_core.JaxprEqn, arrays: Sequence[HostTensor], /
) -> HostArray:
    """Execute fixed-index gather on NumPy, including collapsed and batch axes."""
    operand, indices = arrays
    dimensions = equation.params["dimension_numbers"]
    sizes = equation.params["slice_sizes"]
    output_shape = _array_aval(equation.outvars[0]).shape
    output = np.empty(output_shape, dtype=operand.dtype)
    batch_axes = tuple(
        axis for axis in range(output.ndim) if axis not in dimensions.offset_dims
    )
    window_axes = tuple(
        axis
        for axis in range(operand.ndim)
        if axis not in dimensions.collapsed_slice_dims
        and axis not in dimensions.operand_batching_dims
    )
    for coordinate in np.ndindex(output_shape):
        batch = tuple(coordinate[axis] for axis in batch_axes)
        starts = [0] * operand.ndim
        for axis, value in zip(dimensions.start_index_map, indices[batch], strict=True):
            starts[axis] = int(value)
        valid = all(
            0 <= start <= extent - size
            for start, extent, size in zip(starts, operand.shape, sizes, strict=True)
        )
        for axis in dimensions.start_index_map:
            starts[axis] = min(max(starts[axis], 0), operand.shape[axis] - sizes[axis])
        for operand_axis, indices_axis in zip(
            dimensions.operand_batching_dims,
            dimensions.start_indices_batching_dims,
            strict=True,
        ):
            starts[operand_axis] = batch[indices_axis]
        for operand_axis, output_axis in zip(
            window_axes, dimensions.offset_dims, strict=True
        ):
            starts[operand_axis] += coordinate[output_axis]
        if (
            not valid
            and equation.params["mode"] is jax.lax.GatherScatterMode.FILL_OR_DROP
        ):
            output[coordinate] = equation.params["fill_value"]
        else:
            output[coordinate] = operand[tuple(starts)]
    return output


def _scatter_addresses(
    equation: jax_core.JaxprEqn,
    operand_shape: tuple[int, ...],
    indices: HostTensor,
    update_shape: tuple[int, ...],
    /,
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Route scatter update windows to operand addresses in source order."""
    dimensions = equation.params["dimension_numbers"]
    batch_axes = tuple(
        axis
        for axis in range(len(update_shape))
        if axis not in dimensions.update_window_dims
    )
    window_axes = tuple(
        axis
        for axis in range(len(operand_shape))
        if axis not in dimensions.inserted_window_dims
        and axis not in dimensions.operand_batching_dims
    )
    window_sizes = [1] * len(operand_shape)
    for operand_axis, update_axis in zip(
        window_axes, dimensions.update_window_dims, strict=True
    ):
        window_sizes[operand_axis] = update_shape[update_axis]
    for coordinate in np.ndindex(update_shape):
        batch = tuple(coordinate[axis] for axis in batch_axes)
        starts = [0] * len(operand_shape)
        for axis, value in zip(
            dimensions.scatter_dims_to_operand_dims, indices[batch], strict=True
        ):
            starts[axis] = int(value)
        for operand_axis, indices_axis in zip(
            dimensions.operand_batching_dims,
            dimensions.scatter_indices_batching_dims,
            strict=True,
        ):
            starts[operand_axis] = batch[indices_axis]
        if equation.params["mode"] is jax.lax.GatherScatterMode.CLIP:
            for axis in dimensions.scatter_dims_to_operand_dims:
                starts[axis] = min(
                    max(starts[axis], 0), operand_shape[axis] - window_sizes[axis]
                )
        for operand_axis, update_axis in zip(
            window_axes, dimensions.update_window_dims, strict=True
        ):
            starts[operand_axis] += coordinate[update_axis]
        if all(
            0 <= index < size for index, size in zip(starts, operand_shape, strict=True)
        ):
            yield tuple(starts), coordinate


def _host_scatter(
    equation: jax_core.JaxprEqn, arrays: Sequence[HostTensor], /
) -> HostTensor:
    """Execute overwrite or unique fixed-index scatter arithmetic in source order."""
    operand, indices, updates = arrays
    name = equation.primitive.name
    if name == "scatter" and equation.params["update_jaxpr"] is not None:
        raise IntervalExtensionError(
            "interval.scatter must use overwrite updates; got a reduction program"
        )
    if name != "scatter" and not equation.params["unique_indices"]:
        raise IntervalExtensionError(
            "interval.scatter.indices must be unique for ordered arithmetic; got possible collisions"
        )
    output = np.array(operand, copy=True)
    for target, source in _scatter_addresses(
        equation, operand.shape, indices, updates.shape
    ):
        match name:
            case "scatter":
                output[target] = updates[source]
            case "scatter-mul":
                output[target] = np.multiply(output[target], updates[source])
            case "scatter-add":
                output[target] = np.add(output[target], updates[source])
            case "scatter-max":
                output[target] = np.maximum(output[target], updates[source])
            case "scatter-min":
                output[target] = np.minimum(output[target], updates[source])
            case _:
                raise IntervalExtensionError(
                    f"interval.scatter must have a supported update operator; got {name!r}"
                )
    return output


def _host_scatter_bounds(
    equation: jax_core.JaxprEqn, values: Sequence[Interval], /
) -> list[Interval]:
    """Apply interval arithmetic only at proved fixed unique scatter addresses."""
    if not _degenerate(values[1]) or not equation.params["unique_indices"]:
        raise IntervalExtensionError(
            "interval.scatter.indices must be fixed and unique for arithmetic bounds; got uncertain routing"
        )
    lower, upper = (np.array(value, copy=True) for value in values[0])
    indices = np.asarray(values[1][0])
    update_lower, update_upper = (np.asarray(value) for value in values[2])
    for target, source in _scatter_addresses(
        equation, lower.shape, indices, update_lower.shape
    ):
        operand = lower[target], upper[target]
        update = update_lower[source], update_upper[source]
        match equation.primitive.name:
            case "scatter-mul":
                low, high = interval_multiply(operand, update)
            case "scatter-add":
                low, high = interval_add(operand, update)
            case "scatter-max":
                low, high = (
                    np.maximum(operand[0], update[0]),
                    np.maximum(operand[1], update[1]),
                )
            case "scatter-min":
                low, high = (
                    np.minimum(operand[0], update[0]),
                    np.minimum(operand[1], update[1]),
                )
            case name:
                raise IntervalExtensionError(
                    f"interval.scatter must have a supported update operator; got {name!r}"
                )
        lower[target], upper[target] = low, high
    return [(lower, upper)]


def _host_pad(equation: jax_core.JaxprEqn, arrays: Sequence[HostTensor], /) -> HostArray:
    """Pad and crop a host array with the primitive's interior-padding convention."""
    operand, value = arrays
    configuration = equation.params["padding_config"]
    interior_shape = tuple(
        size + max(size - 1, 0) * interior
        for size, (_, _, interior) in zip(operand.shape, configuration, strict=True)
    )
    expanded = np.full(interior_shape, value, dtype=operand.dtype)
    expanded[
        tuple(slice(None, None, interior + 1) for _, _, interior in configuration)
    ] = operand
    cropped = expanded[
        tuple(
            slice(max(-low, 0), size - max(-high, 0))
            for size, (low, high, _) in zip(interior_shape, configuration, strict=True)
        )
    ]
    return np.pad(
        cropped,
        tuple((max(low, 0), max(high, 0)) for low, high, _ in configuration),
        constant_values=value,
    )


def _host_bind(
    equation: jax_core.JaxprEqn, arrays: Sequence[HostArray], /
) -> list[HostArray]:
    """Execute structural and constant-index primitives without device dispatch."""
    arrays = [np.asarray(array) for array in arrays]
    parameters = equation.params
    match equation.primitive.name:
        case "broadcast_in_dim":
            shape = [1] * len(parameters["shape"])
            for size, axis in zip(
                arrays[0].shape, parameters["broadcast_dimensions"], strict=True
            ):
                shape[axis] = size
            return [np.broadcast_to(arrays[0].reshape(shape), parameters["shape"])]
        case "reshape":
            array = arrays[0]
            if parameters.get("dimensions") is not None:
                array = np.transpose(array, parameters["dimensions"])
            return [np.reshape(array, parameters["new_sizes"])]
        case "transpose":
            return [np.transpose(arrays[0], parameters["permutation"])]
        case "squeeze":
            return [np.squeeze(arrays[0], axis=parameters["dimensions"])]
        case "expand_dims":
            return [np.expand_dims(arrays[0], axis=parameters["dimensions"])]
        case "slice":
            strides = parameters.get("strides") or (1,) * len(parameters["start_indices"])
            slices = tuple(
                slice(first, last, stride)
                for first, last, stride in zip(
                    parameters["start_indices"],
                    parameters["limit_indices"],
                    strides,
                    strict=True,
                )
            )
            return [arrays[0][slices]]
        case "dynamic_slice":
            sizes = parameters["slice_sizes"]
            starts = tuple(
                min(max(int(start), 0), extent - size)
                for start, extent, size in zip(
                    arrays[1:], arrays[0].shape, sizes, strict=True
                )
            )
            return [
                arrays[0][
                    tuple(
                        slice(start, start + size)
                        for start, size in zip(starts, sizes, strict=True)
                    )
                ]
            ]
        case "concatenate":
            return [np.concatenate(arrays, axis=parameters["dimension"])]
        case "stack":
            return [np.stack(arrays, axis=parameters["axis"])]
        case "split":
            indices = np.cumsum(parameters["sizes"], dtype=np.int64)[:-1]
            return list(np.split(arrays[0], indices, axis=parameters["axis"]))
        case "rev":
            return [np.flip(arrays[0], axis=parameters["dimensions"])]
        case "copy" | "copy_p":
            return [arrays[0]]
        case "real":
            return [np.real(arrays[0])]
        case "gather":
            return [_host_gather(equation, arrays)]
        case "scatter" | "scatter-mul" | "scatter-add" | "scatter-min" | "scatter-max":
            return [_host_scatter(equation, arrays)]
        case "pad":
            return [_host_pad(equation, arrays)]
        case "iota":
            shape = parameters["shape"]
            axis = parameters["dimension"]
            reshape = tuple(
                shape[axis] if index == axis else 1 for index in range(len(shape))
            )
            return [
                np.broadcast_to(
                    np.arange(shape[axis], dtype=parameters["dtype"]).reshape(reshape),
                    shape,
                )
            ]
        case "cumsum":
            array = arrays[0]
            axis = parameters["axis"]
            if parameters["reverse"]:
                return [
                    np.flip(
                        np.cumsum(np.flip(array, axis), axis=axis, dtype=array.dtype),
                        axis,
                    )
                ]
            return [np.cumsum(array, axis=axis, dtype=array.dtype)]
        case "argmax" | "argmin" as name:
            operation = np.argmax if name == "argmax" else np.argmin
            return [
                np.asarray(
                    operation(arrays[0], axis=parameters["axes"][0]),
                    dtype=parameters["index_dtype"],
                )
            ]
        case "sort":
            axis = parameters["dimension"]
            keys = arrays[: parameters["num_keys"]]
            order = np.lexsort(tuple(reversed(keys)), axis=axis)
            return [np.take_along_axis(array, order, axis) for array in arrays]
        case _:
            raise IntervalExtensionError(
                f"interval.primitive must support host structural evaluation; got {equation.primitive.name!r}"
            )


def _structural(
    equation: jax_core.JaxprEqn, values: Sequence[Interval], /
) -> list[Interval]:
    lowers = _host_bind(equation, [value[0] for value in values])
    uppers = _host_bind(equation, [value[1] for value in values])
    return list(zip(lowers, uppers, strict=True))


def _degenerate_bind(
    equation: jax_core.JaxprEqn, values: Sequence[Interval], /
) -> list[Interval]:
    """Evaluate constant-index operations on host arrays."""
    return [(item, item) for item in _host_bind(equation, [value[0] for value in values])]


def _host_arithmetic(
    equation: jax_core.JaxprEqn, values: Sequence[HostArray], /
) -> HostArray:
    """Execute a nominal arithmetic primitive with its NumPy formula."""
    parameters = equation.params
    match equation.primitive.name:
        case "add" | "add_any":
            return np.asarray(np.add(values[0], values[1]))
        case "sub":
            return np.asarray(np.subtract(values[0], values[1]))
        case "mul":
            return np.asarray(np.multiply(values[0], values[1]))
        case "div":
            if np.issubdtype(values[0].dtype, np.integer):
                quotient = np.floor_divide(values[0], values[1])
                correction = ((values[0] < 0) != (values[1] < 0)) & (
                    np.remainder(values[0], values[1]) != 0
                )
                return np.asarray(quotient + correction.astype(quotient.dtype))
            return np.asarray(np.divide(values[0], values[1]))
        case "neg":
            return np.asarray(np.negative(values[0]))
        case "integer_pow":
            return np.asarray(np.power(values[0], parameters["y"]))
        case "square":
            return np.asarray(np.power(values[0], 2))
        case "pow":
            return np.asarray(np.power(values[0], values[1]))
        case "dot_general":
            (left_contract, right_contract), (left_batch, right_batch) = parameters[
                "dimension_numbers"
            ]
            left_labels = list(range(values[0].ndim))
            right_labels = list(range(values[0].ndim, values[0].ndim + values[1].ndim))
            for left, right in zip(
                (*left_contract, *left_batch),
                (*right_contract, *right_batch),
                strict=True,
            ):
                right_labels[right] = left_labels[left]
            output_labels = (
                [left_labels[axis] for axis in left_batch]
                + [
                    left_labels[axis]
                    for axis in range(values[0].ndim)
                    if axis not in left_contract and axis not in left_batch
                ]
                + [
                    right_labels[axis]
                    for axis in range(values[1].ndim)
                    if axis not in right_contract and axis not in right_batch
                ]
            )
            # Host interpretation retains the primitive's axis order and dense output.
            return np.einsum(
                values[0], left_labels, values[1], right_labels, output_labels
            )
        case (
            "reduce_sum"
            | "reduce_max"
            | "reduce_min"
            | "reduce_prod"
            | "reduce_or"
            | "reduce_and" as name
        ):
            if name == "reduce_sum":
                return np.asarray(
                    np.sum(
                        values[0], axis=tuple(parameters["axes"]), dtype=values[0].dtype
                    )
                )
            if name == "reduce_prod":
                return np.asarray(
                    np.prod(
                        values[0], axis=tuple(parameters["axes"]), dtype=values[0].dtype
                    )
                )
            operation = {
                "reduce_max": np.max,
                "reduce_min": np.min,
                "reduce_or": np.any,
                "reduce_and": np.all,
            }[name]
            return np.asarray(operation(values[0], axis=tuple(parameters["axes"])))
        case "convert_element_type":
            return values[0].astype(parameters["new_dtype"])
        case "bitcast_convert_type":
            return values[0].view(parameters["new_dtype"])
        case "select_n":
            cases = np.stack(np.broadcast_arrays(*values[1:]))
            indices = np.broadcast_to(values[0], cases.shape[1:]).astype(np.intp)
            return np.take_along_axis(cases, indices[None], axis=0)[0]
        case "clamp":
            return np.maximum(values[0], np.minimum(values[1], values[2]))
        case "unvmap_any" | "unvmap_all" | "unvmap_max" as name:
            operation = {
                "unvmap_any": np.any,
                "unvmap_all": np.all,
                "unvmap_max": np.max,
            }[name]
            return np.asarray(operation(values[0]))
        case "lt_to" | "le_to" | "eq_to" as name:
            operation = {"lt_to": np.less, "le_to": np.less_equal, "eq_to": np.equal}[
                name
            ]
            return np.asarray(
                operation(
                    _host_total_order_key(values[0]), _host_total_order_key(values[1])
                )
            )
        case "stop_gradient":
            return values[0]
        case name:
            unary = {
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                "asin": np.arcsin,
                "acos": np.arccos,
                "atan": np.arctan,
                "asinh": np.arcsinh,
                "acosh": np.arccosh,
                "atanh": np.arctanh,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "exp": np.exp,
                "expm1": np.expm1,
                "log": np.log,
                "log1p": np.log1p,
                "floor": np.floor,
                "ceil": np.ceil,
                "sign": np.sign,
                "not": np.bitwise_not,
                "is_finite": np.isfinite,
            }.get(name)
            if unary is not None:
                return np.asarray(unary(values[0]))
            binary = {
                "max": np.maximum,
                "min": np.minimum,
                "lt": np.less,
                "le": np.less_equal,
                "gt": np.greater,
                "ge": np.greater_equal,
                "eq": np.equal,
                "ne": np.not_equal,
                "and": np.bitwise_and,
                "or": np.bitwise_or,
                "xor": np.bitwise_xor,
                "atan2": np.arctan2,
                "rem": np.fmod,
                "shift_left": np.left_shift,
                "shift_right_arithmetic": np.right_shift,
                "shift_right_logical": np.right_shift,
            }.get(name)
            if binary is not None:
                return np.asarray(binary(values[0], values[1]))
            if name == "rsqrt":
                return np.asarray(1.0 / np.sqrt(values[0]))
            raise IntervalExtensionError(
                f"interval.primitive must support nominal host evaluation; got {name!r}"
            )


_CALL_PRIMITIVES = frozenset(
    {
        "pjit",
        "closed_call",
        "core_call",
        "custom_jvp_call",
        "remat",
        "checkpoint",
        "custom_vjp_call",
        "custom_vjp_call_jaxpr",
        "jit",
    }
)
_HOST_INDEX_PRIMITIVES = frozenset(
    {
        "iota",
        "gather",
        "dynamic_slice",
        "pad",
        "cumsum",
        "argmax",
        "argmin",
        "sort",
        "scatter",
        "scatter-mul",
        "scatter-add",
        "scatter-min",
        "scatter-max",
    }
)
_INTEGER_ARITHMETIC = frozenset(
    {
        "add",
        "add_any",
        "sub",
        "mul",
        "div",
        "neg",
        "integer_pow",
        "square",
        "max",
        "min",
        "and",
        "or",
        "not",
        "rem",
        "select_n",
        "clamp",
        "convert_element_type",
        "reduce_sum",
        "reduce_max",
        "reduce_min",
    }
)


def _evaluate_host_jaxpr(
    closed: jax_core.ClosedJaxpr, arguments: Sequence[HostArray], /
) -> list[HostArray]:
    environment: dict[jax_core.Var, HostArray] = {}

    def read_atom(atom: jax_core.Var | jax_core.Literal) -> HostArray:
        match atom:
            case jax_core.Literal():
                return np.asarray(atom.val)
            case _:
                return environment[atom]

    for variable, constant in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        environment[variable] = np.asarray(constant)
    for variable, argument in zip(closed.jaxpr.invars, arguments, strict=True):
        environment[variable] = argument
    for equation in closed.jaxpr.eqns:
        values = [read_atom(atom) for atom in equation.invars]
        name = equation.primitive.name
        if name in _CALL_PRIMITIVES:
            outputs = _evaluate_host_jaxpr(_as_closed(_nested_jaxpr(equation)), values)
        elif name == "scan":
            outputs = _host_scan(equation, values)
        elif name == "cond":
            branches = equation.params["branches"]
            index = min(max(int(values[0]), 0), len(branches) - 1)
            branch = branches[index]
            outputs = _evaluate_host_jaxpr(_as_closed(branch), values[1:])
        elif name in _STRUCTURAL or name in _HOST_INDEX_PRIMITIVES:
            outputs = _host_bind(equation, values)
        else:
            outputs = [_host_arithmetic(equation, values)]
        for variable, output in zip(equation.outvars, outputs, strict=True):
            environment[variable] = output
    return [read_atom(atom) for atom in closed.jaxpr.outvars]


def _host_scan(
    equation: jax_core.JaxprEqn, values: Sequence[HostArray], /
) -> list[HostArray]:
    """Interpret a fixed-length scan with the source recurrence and output order."""
    parameters = equation.params
    constants_, carry_, sequences_ = parameters["ft_in"].update(values).unpack()
    constants = list(constants_)
    carry = list(carry_)
    sequences = [np.asarray(array) for array in sequences_]
    num_carry = len(carry)
    _, output_variables = parameters["ft_out"].update(equation.outvars).unpack()
    length = parameters["length"]
    outputs = [
        np.empty(_array_aval(variable).shape, dtype=_array_aval(variable).dtype)
        for variable in output_variables
    ]
    indices = range(length - 1, -1, -1) if parameters["reverse"] else range(length)
    for index in indices:
        result = _evaluate_host_jaxpr(
            _as_closed(parameters["jaxpr"]),
            [*constants, *carry, *(array[index] for array in sequences)],
        )
        carry = result[:num_carry]
        for array, value in zip(outputs, result[num_carry:], strict=True):
            array[index] = value
    return [*carry, *outputs]


type DeviceInterval = tuple[Array, Array]


def _device_round_out(lower: Array, upper: Array, steps: int = 1, /) -> DeviceInterval:
    if not jnp.issubdtype(lower.dtype, jnp.floating):
        return lower, upper
    # Each primitive may flush a subnormal result. The normal threshold bounds
    # that displacement independently of gradual-underflow support.
    tiny = jnp.finfo(lower.dtype).tiny
    lower = lower - tiny
    upper = upper + tiny
    for _ in range(steps):
        lower = jnp.nextafter(lower, -jnp.inf)
        upper = jnp.nextafter(upper, jnp.inf)
    invalid = jnp.isnan(lower) | jnp.isnan(upper)
    return jnp.where(invalid, -jnp.inf, lower), jnp.where(invalid, jnp.inf, upper)


def _device_multiply(first: DeviceInterval, second: DeviceInterval, /) -> DeviceInterval:
    products = jnp.stack(
        (
            first[0] * second[0],
            first[0] * second[1],
            first[1] * second[0],
            first[1] * second[1],
        )
    )
    return _device_round_out(jnp.min(products, axis=0), jnp.max(products, axis=0))


def _device_dot_general(
    equation: jax_core.JaxprEqn, first: DeviceInterval, second: DeviceInterval, /
) -> DeviceInterval:
    """Enclose the full dot output using the same midpoint-radius bound as host."""
    first_mid = 0.5 * first[0] + 0.5 * first[1]
    second_mid = 0.5 * second[0] + 0.5 * second[1]
    first_rad = jnp.maximum(first_mid - first[0], first[1] - first_mid)
    second_rad = jnp.maximum(second_mid - second[0], second[1] - second_mid)
    dimensions = equation.params["dimension_numbers"]

    def contract(left: Array, right: Array) -> Array:
        # The interpreter owns this primitive's dense output and axis semantics.
        return jax.lax.dot_general(left, right, dimensions)

    middle = contract(first_mid, second_mid)
    magnitude = contract(jnp.abs(first_mid), jnp.abs(second_mid))
    radius = (
        contract(jnp.abs(first_mid), second_rad)
        + contract(first_rad, jnp.abs(second_mid))
        + contract(first_rad, second_rad)
    )
    (contracting, _), _ = dimensions
    length = 1 + math.prod(first_mid.shape[axis] for axis in contracting)
    unit = jnp.finfo(middle.dtype).eps
    gamma = 2.0 * length * unit / (1.0 - 2.0 * length * unit)
    radius = jnp.nextafter(
        radius + gamma * (magnitude + radius) + 4 * length * jnp.finfo(middle.dtype).tiny,
        jnp.inf,
    )
    return _device_round_out(middle - radius, middle + radius)


def _device_trigonometric(name: str, value: DeviceInterval, /) -> DeviceInterval:
    lower, upper = value
    slack = (
        16
        * jnp.finfo(lower.dtype).eps
        * (1 + jnp.maximum(jnp.abs(lower), jnp.abs(upper)))
    )

    def contains_phase(phase: float) -> Array:
        index = jnp.ceil((lower - phase - slack) / _TWO_PI)
        return phase + index * _TWO_PI <= upper + slack

    if name == "tan":
        width = math.pi
        pole = jnp.ceil((lower - 0.5 * width - slack) / width)
        crosses = 0.5 * width + pole * width <= upper + slack
        low, high = _device_round_out(
            jnp.tan(lower), jnp.tan(upper), _TRANSCENDENTAL_STEPS
        )
        return jnp.where(crosses, -jnp.inf, low), jnp.where(crosses, jnp.inf, high)
    operation = jnp.sin if name == "sin" else jnp.cos
    low = jnp.minimum(operation(lower), operation(upper))
    high = jnp.maximum(operation(lower), operation(upper))
    full = upper - lower >= _TWO_PI
    maximum_phase = 0.5 * math.pi if name == "sin" else 0.0
    minimum_phase = 1.5 * math.pi if name == "sin" else math.pi
    low = jnp.where(full | contains_phase(minimum_phase), -1.0, low)
    high = jnp.where(full | contains_phase(maximum_phase), 1.0, high)
    low, high = _device_round_out(low, high, _TRANSCENDENTAL_STEPS)
    return jnp.maximum(low, -1.0), jnp.minimum(high, 1.0)


def _device_compare(
    name: str, first: DeviceInterval, second: DeviceInterval, /
) -> DeviceInterval:
    match name:
        case "lt":
            return first[1] < second[0], first[0] < second[1]
        case "le":
            return first[1] <= second[0], first[0] <= second[1]
        case "gt":
            return second[1] < first[0], second[0] < first[1]
        case "ge":
            return second[1] <= first[0], second[0] <= first[1]
        case "eq":
            return (
                (first[0] == first[1])
                & (second[0] == second[1])
                & (first[0] == second[0]),
                (first[0] <= second[1]) & (second[0] <= first[1]),
            )
        case "ne":
            low, high = _device_compare("eq", first, second)
            return ~high, ~low
        case _:
            raise IntervalExtensionError(
                f"interval.comparison must be supported; got {name!r}"
            )


def _device_select(
    predicate: DeviceInterval,
    cases: Sequence[DeviceInterval],
    /,
    *,
    case_indices: Sequence[int] | None = None,
) -> DeviceInterval:
    lowers = jnp.stack(jnp.broadcast_arrays(*(value[0] for value in cases)))
    uppers = jnp.stack(jnp.broadcast_arrays(*(value[1] for value in cases)))
    low_index = jnp.broadcast_to(
        jnp.asarray(predicate[0], dtype=jnp.int32), lowers.shape[1:]
    )
    high_index = jnp.broadcast_to(
        jnp.asarray(predicate[1], dtype=jnp.int32), lowers.shape[1:]
    )
    shape = (len(cases),) + (1,) * (lowers.ndim - 1)
    indices = (
        jnp.arange(len(cases), dtype=jnp.int32)
        if case_indices is None
        else jnp.asarray(tuple(case_indices), dtype=jnp.int32)
    ).reshape(shape)
    valid = (indices >= low_index[None, ...]) & (indices <= high_index[None, ...])
    if jnp.issubdtype(lowers.dtype, jnp.bool_):
        minimum = jnp.asarray(False, dtype=jnp.bool_)
        maximum = jnp.asarray(True, dtype=jnp.bool_)
    elif jnp.issubdtype(lowers.dtype, jnp.integer):
        minimum = jnp.asarray(jnp.iinfo(lowers.dtype).min, dtype=lowers.dtype)
        maximum = jnp.asarray(jnp.iinfo(lowers.dtype).max, dtype=lowers.dtype)
    else:
        minimum = jnp.asarray(-jnp.inf, dtype=lowers.dtype)
        maximum = jnp.asarray(jnp.inf, dtype=lowers.dtype)
    return (
        jnp.min(jnp.where(valid, lowers, maximum), axis=0),
        jnp.max(jnp.where(valid, uppers, minimum), axis=0),
    )


def _device_unary(name: str, value: DeviceInterval, /) -> DeviceInterval:
    lower, upper = value
    match name:
        case "sin" | "cos" | "tan":
            return _device_trigonometric(name, value)
        case "abs":
            low = jnp.where(
                (lower <= 0) & (upper >= 0),
                0,
                jnp.minimum(jnp.abs(lower), jnp.abs(upper)),
            )
            return low, jnp.maximum(jnp.abs(lower), jnp.abs(upper))
        case "cosh":
            low = jnp.where(
                (lower <= 0) & (upper >= 0),
                1,
                jnp.minimum(jnp.cosh(lower), jnp.cosh(upper)),
            )
            return _device_round_out(
                low, jnp.maximum(jnp.cosh(lower), jnp.cosh(upper)), _TRANSCENDENTAL_STEPS
            )
        case "sqrt":
            return _device_round_out(
                jnp.sqrt(jnp.maximum(lower, 0)), jnp.sqrt(jnp.maximum(upper, 0))
            )
        case "rsqrt":
            return _device_round_out(1 / jnp.sqrt(upper), 1 / jnp.sqrt(lower), 2)
        case "acos":
            return _device_round_out(
                jnp.arccos(upper), jnp.arccos(lower), _TRANSCENDENTAL_STEPS
            )
        case "is_finite":
            lower, upper = value
            low_finite = jnp.isfinite(lower)
            high_finite = jnp.isfinite(upper)
            return low_finite & high_finite, low_finite | high_finite | (lower < upper)
        case "unvmap_any" | "unvmap_all" | "unvmap_max" as name:
            operation = {
                "unvmap_any": jnp.any,
                "unvmap_all": jnp.all,
                "unvmap_max": jnp.max,
            }[name]
            return operation(lower), operation(upper)
        case "stop_gradient":
            return value
        case name:
            operation = {
                "exp": jnp.exp,
                "expm1": jnp.expm1,
                "log": jnp.log,
                "log1p": jnp.log1p,
                "sinh": jnp.sinh,
                "tanh": jnp.tanh,
                "asin": jnp.arcsin,
                "atan": jnp.arctan,
                "asinh": jnp.arcsinh,
                "acosh": jnp.arccosh,
                "atanh": jnp.arctanh,
                "floor": jnp.floor,
                "ceil": jnp.ceil,
                "sign": jnp.sign,
            }.get(name)
            if operation is None:
                raise IntervalExtensionError(
                    f"interval.primitive must support device interval evaluation; got {name!r}"
                )
            if name in {"floor", "ceil", "sign"}:
                return operation(lower), operation(upper)
            return _device_round_out(
                operation(lower), operation(upper), _TRANSCENDENTAL_STEPS
            )


def _device_index_extremum(
    equation: jax_core.JaxprEqn, value: DeviceInterval, /
) -> DeviceInterval:
    """Enclose winning indices, including the primitive's first-index tie rule."""
    axes = tuple(equation.params["axes"])
    if len(axes) != 1:
        raise IntervalExtensionError(
            f"interval.index_extremum.axes must select one axis; got {axes!r}"
        )
    axis = axes[0]
    lower, upper = value
    size = lower.shape[axis]
    index_shape = [1] * lower.ndim
    index_shape[axis] = size
    indices = jnp.arange(size, dtype=equation.params["index_dtype"]).reshape(index_shape)
    match equation.primitive.name:
        case "argmax":
            prefix = jax.lax.associative_scan(jnp.maximum, lower, axis=axis)
            candidates = (upper >= jnp.max(lower, axis=axis, keepdims=True)) & (
                (indices == 0) | (upper > jnp.roll(prefix, 1, axis=axis))
            )
        case "argmin":
            prefix = jax.lax.associative_scan(jnp.minimum, upper, axis=axis)
            candidates = (lower <= jnp.min(upper, axis=axis, keepdims=True)) & (
                (indices == 0) | (lower < jnp.roll(prefix, 1, axis=axis))
            )
        case name:
            raise IntervalExtensionError(
                f"interval.index_extremum must select argmax or argmin; got {name!r}"
            )
    # A NaN interval has no ordered numeric comparison proof. Every index is
    # then admissible; singleton inputs still execute the native tie rule.
    nan = jnp.isnan(lower) | jnp.isnan(upper)
    candidates = candidates | jnp.any(nan, axis=axis, keepdims=True)
    first = jnp.min(jnp.where(candidates, indices, size), axis=axis)
    last = jnp.max(jnp.where(candidates, indices, 0), axis=axis)
    singleton = jnp.all(
        (lower == upper) | (jnp.isnan(lower) & jnp.isnan(upper)),
        axis=axis,
    )
    nominal = equation.primitive.bind(lower, **equation.params)
    return jnp.where(singleton, nominal, first), jnp.where(singleton, nominal, last)


def _host_index_extremum(equation: jax_core.JaxprEqn, value: Interval, /) -> Interval:
    """Enclose all extremum winners without guessing a routing index."""
    axes = tuple(equation.params["axes"])
    if len(axes) != 1:
        raise IntervalExtensionError(
            f"interval.index_extremum.axes must select one axis; got {axes!r}"
        )
    axis = axes[0]
    lower, upper = (np.asarray(endpoint) for endpoint in value)
    size = lower.shape[axis]
    index_shape = [1] * lower.ndim
    index_shape[axis] = size
    indices = np.arange(size, dtype=equation.params["index_dtype"]).reshape(index_shape)
    match equation.primitive.name:
        case "argmax":
            prefix = np.maximum.accumulate(lower, axis=axis)
            candidates = (upper >= np.max(lower, axis=axis, keepdims=True)) & (
                (indices == 0) | (upper > np.roll(prefix, 1, axis=axis))
            )
        case "argmin":
            prefix = np.minimum.accumulate(upper, axis=axis)
            candidates = (lower <= np.min(upper, axis=axis, keepdims=True)) & (
                (indices == 0) | (lower < np.roll(prefix, 1, axis=axis))
            )
        case name:
            raise IntervalExtensionError(
                f"interval.index_extremum must select argmax or argmin; got {name!r}"
            )
    candidates |= np.any(np.isnan(lower) | np.isnan(upper), axis=axis, keepdims=True)
    return (
        np.min(np.where(candidates, indices, size), axis=axis),
        np.max(np.where(candidates, indices, 0), axis=axis),
    )


def _device_arithmetic(
    equation: jax_core.JaxprEqn, values: Sequence[DeviceInterval], /
) -> DeviceInterval:
    """Dispatch each primitive to its directed interval formula.

    This case table keeps primitive and reduction order visible. Dot products,
    comparisons, transcendental extrema, and static powers have separate bound
    kernels; the remaining cases are straight-line arithmetic formulas.
    """
    name = equation.primitive.name
    if (
        name == "convert_element_type"
        and values[0][0].dtype == equation.params["new_dtype"]
    ):
        return values[0]
    if name == "convert_element_type" and jnp.issubdtype(values[0][0].dtype, jnp.bool_):
        dtype = equation.params["new_dtype"]
        return values[0][0].astype(dtype), values[0][1].astype(dtype)
    if name in _INTEGER_ARITHMETIC and np.issubdtype(
        _array_aval(equation.outvars[0]).dtype, np.integer
    ):
        return _device_integer_arithmetic(equation, values)
    match name:
        case "add" | "add_any":
            return _device_round_out(
                values[0][0] + values[1][0], values[0][1] + values[1][1]
            )
        case "sub":
            first, second = values
            low, high = _device_round_out(first[0] - second[1], first[1] - second[0])
            # Equal finite singleton operands subtract to zero without rounding
            # error; retaining that identity keeps repeated knots decidable.
            zero = (
                (first[0] == first[1])
                & (second[0] == second[1])
                & (first[0] == second[0])
                & jnp.isfinite(first[0])
            )
            if equation.invars[0] is equation.invars[1]:
                zero = jnp.isfinite(first[0]) & jnp.isfinite(first[1])
            return jnp.where(zero, jnp.zeros_like(low), low), jnp.where(
                zero, jnp.zeros_like(high), high
            )
        case "mul":
            return _device_multiply(values[0], values[1])
        case "div":
            lower, upper = values[1]
            reciprocal = _device_round_out(1 / upper, 1 / lower)
            low, high = _device_multiply(values[0], reciprocal)
            crosses = (lower <= 0) & (upper >= 0)
            return jnp.where(crosses, -jnp.inf, low), jnp.where(crosses, jnp.inf, high)
        case "neg":
            return -values[0][1], -values[0][0]
        case "integer_pow" | "square":
            exponent = equation.params["y"] if name == "integer_pow" else 2
            if exponent == 0:
                ones = jnp.ones_like(values[0][0])
                return ones, ones
            if exponent < 0:
                positive = _device_integer_power(values[0], -exponent)
                lower, upper = positive
                low, high = _device_round_out(1 / upper, 1 / lower, 2)
                crosses = (lower <= 0) & (upper >= 0)
                return jnp.where(crosses, -jnp.inf, low), jnp.where(
                    crosses, jnp.inf, high
                )
            return _device_integer_power(values[0], exponent)
        case "dot_general":
            return _device_dot_general(equation, values[0], values[1])
        case "reduce_sum":
            axes = tuple(equation.params["axes"])
            lower, upper = values[0]
            low, high = jnp.sum(lower, axis=axes), jnp.sum(upper, axis=axes)
            if not jnp.issubdtype(low.dtype, jnp.floating):
                return low, high
            length = max(1, math.prod(lower.shape[axis] for axis in axes))
            unit = jnp.finfo(low.dtype).eps
            gamma = length * unit / (1 - length * unit)
            scale = jnp.sum(jnp.maximum(jnp.abs(lower), jnp.abs(upper)), axis=axes)
            allowance = gamma * scale + length * jnp.finfo(low.dtype).tiny
            return _device_round_out(low - allowance, high + allowance)
        case "reduce_max" | "reduce_min" | "reduce_or" | "reduce_and":
            operation = {
                "reduce_max": jnp.max,
                "reduce_min": jnp.min,
                "reduce_or": jnp.any,
                "reduce_and": jnp.all,
            }[name]
            axes = tuple(equation.params["axes"])
            return operation(values[0][0], axis=axes), operation(values[0][1], axis=axes)
        case "max" | "min":
            operation = jnp.maximum if name == "max" else jnp.minimum
            return operation(values[0][0], values[1][0]), operation(
                values[0][1], values[1][1]
            )
        case "lt_to" | "le_to" | "eq_to" as total_name:
            first = (
                _device_total_order_key(values[0][0]),
                _device_total_order_key(values[0][1]),
            )
            second = (
                _device_total_order_key(values[1][0]),
                _device_total_order_key(values[1][1]),
            )
            return _device_compare(total_name.removesuffix("_to"), first, second)
        case "lt" | "le" | "gt" | "ge" | "eq" | "ne":
            return _device_compare(name, values[0], values[1])
        case "and" | "or":
            operation = jnp.bitwise_and if name == "and" else jnp.bitwise_or
            return operation(values[0][0], values[1][0]), operation(
                values[0][1], values[1][1]
            )
        case "not":
            return ~values[0][1], ~values[0][0]
        case "convert_element_type":
            dtype = equation.params["new_dtype"]
            lower, upper = values[0][0].astype(dtype), values[0][1].astype(dtype)
            return (
                _device_round_out(lower, upper)
                if jnp.issubdtype(dtype, jnp.floating)
                else (lower, upper)
            )
        case "clamp":
            minimum, operand, maximum = values
            return jnp.maximum(
                minimum[0], jnp.minimum(operand[0], maximum[0])
            ), jnp.maximum(minimum[1], jnp.minimum(operand[1], maximum[1]))
        case "select_n":
            return _device_select(values[0], values[1:])
        case "rem":
            if jnp.issubdtype(values[0][0].dtype, jnp.integer):
                return _device_integer_arithmetic(equation, values)
            raise IntervalExtensionError(
                f"interval.rem must operate on integer index intervals; got {values[0][0].dtype}"
            )
        case _:
            return _device_unary(name, values[0])


def _device_integer_arithmetic(
    equation: jax_core.JaxprEqn, values: Sequence[DeviceInterval], /
) -> DeviceInterval:
    """Preserve singleton integer primitive results; otherwise enclose the dtype."""
    certain = jnp.all(jnp.stack([jnp.all(lower == upper) for lower, upper in values]))
    result = equation.primitive.bind(*(value[0] for value in values), **equation.params)
    limits = jnp.iinfo(result.dtype)
    return jnp.where(certain, result, limits.min), jnp.where(certain, result, limits.max)


def _device_integer_power(value: DeviceInterval, exponent: int, /) -> DeviceInterval:
    """Enclose JAX's static repeated-squaring lowering, including every multiply."""
    while exponent % 2 == 0:
        value = _device_multiply(value, value)
        exponent //= 2
    result = value
    exponent //= 2
    while exponent:
        value = _device_multiply(value, value)
        if exponent & 1:
            result = _device_multiply(result, value)
        exponent //= 2
    return result


def _device_scatter_bounds(
    equation: jax_core.JaxprEqn, values: Sequence[DeviceInterval], /
) -> list[DeviceInterval]:
    """Enclose fixed unique scatter arithmetic, leaving untouched entries unchanged."""
    if not equation.params["unique_indices"]:
        raise IntervalExtensionError(
            "interval.scatter.indices must be unique for arithmetic bounds; got possible collisions"
        )
    index = values[1][0]
    certain = jnp.all(index == values[1][1])
    candidates = [
        equation.primitive.bind(operand, index, update, **equation.params)
        for operand in values[0]
        for update in values[2]
    ]
    lower = jnp.min(jnp.stack(candidates), axis=0)
    upper = jnp.max(jnp.stack(candidates), axis=0)
    updated = jax.lax.scatter(
        jnp.zeros_like(values[0][0], dtype=jnp.bool_),
        index,
        jnp.ones_like(values[2][0], dtype=jnp.bool_),
        equation.params["dimension_numbers"],
        indices_are_sorted=equation.params["indices_are_sorted"],
        unique_indices=True,
        mode=equation.params["mode"],
    )
    if equation.primitive.name in {"scatter-add", "scatter-mul"}:
        low, high = _device_round_out(lower, upper)
        lower = jnp.where(updated, low, lower)
        upper = jnp.where(updated, high, upper)
    return eqx.error_if(
        [(lower, upper)],
        ~certain,
        "interval.scatter.indices must be fixed for arithmetic bounds; got uncertain routing",
    )


def _device_structural(
    equation: jax_core.JaxprEqn, values: Sequence[DeviceInterval], /
) -> list[DeviceInterval]:
    def bind(arrays: Sequence[Array]) -> list[Array]:
        if (
            equation.primitive.name == "scatter"
            and equation.params["update_jaxpr"] is not None
        ):
            raise IntervalExtensionError(
                "interval.scatter must use overwrite updates; got a reduction update program"
            )
        result = equation.primitive.bind(*arrays, **equation.params)
        return list(result) if equation.primitive.multiple_results else [result]

    lower = bind([value[0] for value in values])
    upper = bind([value[1] for value in values])
    if equation.primitive.name in {"gather", "dynamic_slice"}:
        certain = jnp.all(jnp.stack([jnp.all(low == high) for low, high in values[1:]]))
        # An uncertain index selects a subset of the operand, so its global hull
        # encloses every possible selection without inferring one nominal index.
        lower = [jnp.where(certain, item, jnp.min(values[0][0])) for item in lower]
        upper = [jnp.where(certain, item, jnp.max(values[0][1])) for item in upper]
    if equation.primitive.name == "scatter":
        certain = jnp.all(values[1][0] == values[1][1])
        minimum = jnp.minimum(jnp.min(values[0][0]), jnp.min(values[2][0]))
        maximum = jnp.maximum(jnp.max(values[0][1]), jnp.max(values[2][1]))
        lower = [jnp.where(certain, item, minimum) for item in lower]
        upper = [jnp.where(certain, item, maximum) for item in upper]
    return list(zip(lower, upper, strict=True))


def _device_scan(
    equation: jax_core.JaxprEqn, values: Sequence[DeviceInterval], /
) -> list[DeviceInterval]:
    parameters = equation.params
    constants_, carry_, sequences_ = parameters["ft_in"].update(values).unpack()
    constants = tuple(constants_)
    carry = tuple(carry_)
    sequences = tuple(sequences_)
    num_carry = len(carry)

    def body(
        state: tuple[DeviceInterval, ...], items: tuple[DeviceInterval, ...]
    ) -> tuple[tuple[DeviceInterval, ...], tuple[DeviceInterval, ...]]:
        results = _evaluate_device_jaxpr(
            _as_closed(parameters["jaxpr"]), [*constants, *state, *items]
        )
        return tuple(results[:num_carry]), tuple(results[num_carry:])

    final, outputs = jax.lax.scan(
        body,
        carry,
        sequences,
        length=parameters["length"],
        reverse=parameters["reverse"],
    )
    return [*final, *outputs]


def _contains_opaque_callback(closed: jax_core.ClosedJaxpr, /) -> bool:
    """Identify callback-bearing programs before conditional branch admission."""

    def contains_parameter(value: object) -> bool:
        match value:
            case jax_core.ClosedJaxpr():
                return _contains_opaque_callback(value)
            case tuple():
                return any(contains_parameter(item) for item in value)
            case _:
                return False

    return any(
        equation.primitive.name in {"pure_callback", "io_callback", "debug_callback"}
        or any(contains_parameter(value) for value in equation.params.values())
        for equation in closed.jaxpr.eqns
    )


def _device_conditional(
    equation: jax_core.JaxprEqn, values: Sequence[DeviceInterval], /
) -> list[DeviceInterval]:
    """Admit numeric branches only under a proof that opaque branches cannot run."""
    programs = tuple(_as_closed(branch) for branch in equation.params["branches"])
    lower_index = jnp.clip(values[0][0], 0, len(programs) - 1)
    upper_index = jnp.clip(values[0][1], 0, len(programs) - 1)
    predicate = lower_index, upper_index
    forbidden = jnp.asarray(False, dtype=jnp.bool_)
    admitted: list[int] = []
    results: list[list[DeviceInterval]] = []
    for index, program in enumerate(programs):
        if _contains_opaque_callback(program):
            forbidden |= (lower_index <= index) & (index <= upper_index)
        else:
            admitted.append(index)
            results.append(_evaluate_device_jaxpr(program, values[1:]))
    if not admitted:
        raise IntervalExtensionError(
            "interval.cond must admit a numeric branch; got only opaque callback branches"
        )
    outputs = [
        _device_select(predicate, cases, case_indices=admitted)
        for cases in zip(*results, strict=True)
    ]
    return eqx.error_if(
        outputs,
        forbidden,
        "interval.cond must exclude opaque callback branches throughout the input interval; got a possible callback branch",
    )


def _evaluate_device_jaxpr(
    closed: jax_core.ClosedJaxpr, arguments: Sequence[DeviceInterval], /
) -> list[DeviceInterval]:
    environment: dict[jax_core.Var, DeviceInterval] = {}

    def read_atom(atom: jax_core.Var | jax_core.Literal) -> DeviceInterval:
        match atom:
            case jax_core.Literal():
                value = jnp.asarray(atom.val)
                return value, value
            case _:
                return environment[atom]

    for variable, constant in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        value = jax.lax.stop_gradient(jnp.asarray(constant))
        environment[variable] = value, value
    for variable, argument in zip(closed.jaxpr.invars, arguments, strict=True):
        environment[variable] = argument
    for equation in closed.jaxpr.eqns:
        values = [read_atom(atom) for atom in equation.invars]
        name = equation.primitive.name
        if name in _CALL_PRIMITIVES:
            outputs = _evaluate_device_jaxpr(_as_closed(_nested_jaxpr(equation)), values)
        elif name == "scan":
            outputs = _device_scan(equation, values)
        elif name == "cond":
            outputs = _device_conditional(equation, values)
        elif name in {"scatter-mul", "scatter-add", "scatter-min", "scatter-max"}:
            outputs = _device_scatter_bounds(equation, values)
        elif name in {"argmax", "argmin"}:
            outputs = [_device_index_extremum(equation, values[0])]
        elif name in _STRUCTURAL or name in {
            "iota",
            "gather",
            "dynamic_slice",
            "pad",
            "scatter",
        }:
            outputs = _device_structural(equation, values)
        else:
            outputs = [_device_arithmetic(equation, values)]
        for variable, output in zip(equation.outvars, outputs, strict=True):
            environment[variable] = output
    return [read_atom(atom) for atom in closed.jaxpr.outvars]


@checked
def device_interval_evaluation_error(
    function: Callable[[Array], Array],
    point: Float[_IntervalPointDims],
    /,
    *,
    input_bounds: tuple[Float[_IntervalPointDims], Float[_IntervalPointDims]]
    | None = None,
) -> Array:
    """Bound coordinatewise evaluation and input-rounding error on the device.

    Dynamic callback leaves remain jaxpr constants, not cached static metadata.
    Bounds assume IEEE basic arithmetic and the four-ulp elementary-function
    accuracy assumption used by the host interpreter. The width encloses the difference
    between real evaluation and any rounded evaluation admitted by the same
    primitive bounds. Unsupported primitives raise ``IntervalExtensionError``
    during tracing.

    ``input_bounds`` encloses known source coordinates and must contain ``point``.
    Omitted bounds retain a singleton. No input uncertainty is inferred from
    coordinate shape or from the callback's carrier family.
    """
    point_ = jax.lax.stop_gradient(point)
    if input_bounds is None:
        argument = point_, point_
    else:
        input_lower, input_upper = input_bounds
        if input_lower.dtype != point.dtype or input_upper.dtype != point.dtype:
            raise ValueError(
                f"interval.input_bounds must share the point shape and dtype; got {input_lower.shape}/{input_lower.dtype}, {input_upper.shape}/{input_upper.dtype} for {point.shape}/{point.dtype}"
            )
        point_ = eqx.error_if(
            point_,
            jnp.any(
                jnp.isnan(input_lower)
                | jnp.isnan(input_upper)
                | (input_lower > point_)
                | (point_ > input_upper)
            ),
            "interval.input_bounds must contain the nominal point in ordered non-NaN bounds; got invalid bounds",
        )
        argument = (
            jax.lax.stop_gradient(input_lower),
            jax.lax.stop_gradient(input_upper),
        )
    closed = jax.make_jaxpr(function)(point_)
    if len(closed.jaxpr.outvars) != 1:
        raise ValueError(
            f"interval.function must return one array; got {len(closed.jaxpr.outvars)} outputs"
        )
    ((lower, upper),) = _evaluate_device_jaxpr(closed, [argument])
    if not jnp.issubdtype(lower.dtype, jnp.floating):
        lower, upper = _device_round_out(
            lower.astype(point.dtype), upper.astype(point.dtype)
        )
    error = jnp.nextafter(upper - lower, jnp.inf)
    return jax.lax.stop_gradient(jnp.where(jnp.isnan(error), jnp.inf, error))


def _nested_jaxpr(
    equation: jax_core.JaxprEqn, /
) -> jax_core.ClosedJaxpr | jax_core.Jaxpr:
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in equation.params:
            return equation.params[key]
    raise IntervalExtensionError(
        f"interval.primitive must expose a nested jaxpr; got {equation.primitive.name!r}"
    )


def _as_closed(jaxpr: jax_core.ClosedJaxpr | jax_core.Jaxpr, /) -> jax_core.ClosedJaxpr:
    match jaxpr:
        case jax_core.ClosedJaxpr():
            return jaxpr
        case jax_core.Jaxpr():
            return jax_core.ClosedJaxpr(jaxpr, ())
        case _:
            raise TypeError(
                f"interval.jaxpr must be a Jaxpr or ClosedJaxpr; got {type(jaxpr).__qualname__}"
            )


# Keep primitive selection explicit so dtype guards, directed formulas, and fail-closed errors share one admission order.
def _arithmetic(equation: jax_core.JaxprEqn, values: Sequence[Interval], /) -> Interval:
    name = equation.primitive.name
    if name == "convert_element_type" and values[0][0].dtype == np.dtype(
        equation.params["new_dtype"]
    ):
        return values[0]
    if name == "convert_element_type" and np.issubdtype(values[0][0].dtype, np.bool_):
        dtype = equation.params["new_dtype"]
        return values[0][0].astype(dtype), values[0][1].astype(dtype)
    if name in _INTEGER_ARITHMETIC and np.issubdtype(
        _array_aval(equation.outvars[0]).dtype, np.integer
    ):
        return _host_integer_arithmetic(equation, values)
    match name:
        case "add" | "add_any":
            return interval_add(values[0], values[1])
        case "sub":
            if equation.invars[0] is equation.invars[1]:
                finite = np.isfinite(values[0][0]) & np.isfinite(values[0][1])
                bounds = interval_subtract(values[0], values[1])
                return _replace_proved_zero(bounds[0], finite), _replace_proved_zero(
                    bounds[1], finite
                )
            return interval_subtract(values[0], values[1])
        case "mul":
            return interval_multiply(values[0], values[1])
        case "div":
            return interval_divide(values[0], values[1])
        case "neg":
            return np.negative(values[0][1]), np.negative(values[0][0])
        case "integer_pow":
            return _integer_power(values[0], int(equation.params["y"]))
        case "square":
            return _integer_power(values[0], 2)
        case "max":
            return np.maximum(values[0][0], values[1][0]), np.maximum(
                values[0][1], values[1][1]
            )
        case "min":
            return np.minimum(values[0][0], values[1][0]), np.minimum(
                values[0][1], values[1][1]
            )
        case "dot_general":
            return interval_dot_general(
                values[0], values[1], equation.params["dimension_numbers"]
            )
        case "reduce_sum":
            return _reduce_sum(values[0], equation.params["axes"])
        case "reduce_max":
            axes = tuple(int(axis) for axis in equation.params["axes"])
            return np.max(values[0][0], axis=axes), np.max(values[0][1], axis=axes)
        case "reduce_min":
            axes = tuple(int(axis) for axis in equation.params["axes"])
            return np.min(values[0][0], axis=axes), np.min(values[0][1], axis=axes)
        case "convert_element_type":
            dtype = np.dtype(equation.params["new_dtype"])
            lower, upper = values[0]
            if np.issubdtype(dtype, np.floating):
                return _round_out(lower.astype(dtype), upper.astype(dtype))
            if not _degenerate(values[0]) and np.issubdtype(lower.dtype, np.floating):
                raise IntervalExtensionError(
                    f"interval.convert_element_type must preserve nonconstant real intervals; got output dtype {dtype}"
                )
            return lower.astype(dtype), upper.astype(dtype)
        case "lt_to" | "le_to" | "eq_to" as total_name:
            first = (
                _host_total_order_key(values[0][0]),
                _host_total_order_key(values[0][1]),
            )
            second = (
                _host_total_order_key(values[1][0]),
                _host_total_order_key(values[1][1]),
            )
            return _compare(total_name.removesuffix("_to"), first, second)
        case "lt" | "le" | "gt" | "ge" | "eq" | "ne":
            return _compare(name, values[0], values[1])
        case "and":
            return np.bitwise_and(values[0][0], values[1][0]), np.bitwise_and(
                values[0][1], values[1][1]
            )
        case "or":
            return np.bitwise_or(values[0][0], values[1][0]), np.bitwise_or(
                values[0][1], values[1][1]
            )
        case "not":
            return np.invert(values[0][1]), np.invert(values[0][0])
        case "clamp":
            minimum, operand, maximum = values
            return (
                np.maximum(minimum[0], np.minimum(operand[0], maximum[0])),
                np.maximum(minimum[1], np.minimum(operand[1], maximum[1])),
            )
        case "select_n":
            return _select(values[0], values[1:])
        case "rem":
            if np.issubdtype(values[0][0].dtype, np.integer):
                return _host_integer_arithmetic(equation, values)
            raise IntervalExtensionError(
                f"interval.rem must operate on integer index intervals; got {values[0][0].dtype}"
            )
        case "cosh":
            lower, upper = values[0]
            low = np.where(
                (lower <= 0) & (upper >= 0), 1, np.minimum(np.cosh(lower), np.cosh(upper))
            )
            return _round_out(
                low, np.maximum(np.cosh(lower), np.cosh(upper)), _TRANSCENDENTAL_STEPS
            )
        case "is_finite":
            lower, upper = values[0]
            low_finite = np.isfinite(lower)
            high_finite = np.isfinite(upper)
            return (
                np.bitwise_and(low_finite, high_finite),
                np.bitwise_or(
                    np.bitwise_or(low_finite, high_finite), np.less(lower, upper)
                ),
            )
        case "unvmap_any" | "unvmap_all" | "unvmap_max" as name:
            operation = {
                "unvmap_any": np.any,
                "unvmap_all": np.all,
                "unvmap_max": np.max,
            }[name]
            return np.asarray(operation(values[0][0])), np.asarray(
                operation(values[0][1])
            )
        case "stop_gradient":
            return values[0]
        case "reduce_or" | "reduce_and" as name:
            operation = np.any if name == "reduce_or" else np.all
            axes = tuple(equation.params["axes"])
            return np.asarray(operation(values[0][0], axis=axes)), np.asarray(
                operation(values[0][1], axis=axes)
            )
        case _:
            unary = _UNARY.get(name)
            if unary is None:
                raise IntervalExtensionError(
                    f"interval.primitive must have a supported extension; got {name!r}"
                )
            return unary(values[0])


def _host_integer_arithmetic(
    equation: jax_core.JaxprEqn, values: Sequence[Interval], /
) -> Interval:
    result = np.asarray(
        _host_arithmetic(equation, [value[0] for value in values]),
        dtype=_array_aval(equation.outvars[0]).dtype,
    )
    certain = all(_degenerate(value) for value in values)
    limits = np.iinfo(result.dtype)
    return np.where(certain, result, limits.min), np.where(certain, result, limits.max)


def _admit_interval_program(
    closed: jax_core.ClosedJaxpr, /, *, guarded_callbacks: bool = False
) -> None:
    """Admit supported opcodes without inventing a numerical source parameter box.

    Callback-bearing conditional branches retain query-time admission: only
    bounds excluding those branches can use the interval program.
    """
    arithmetic = {
        "add",
        "add_any",
        "sub",
        "mul",
        "div",
        "neg",
        "integer_pow",
        "square",
        "max",
        "min",
        "dot_general",
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_or",
        "reduce_and",
        "convert_element_type",
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "lt_to",
        "le_to",
        "eq_to",
        "and",
        "or",
        "not",
        "clamp",
        "select_n",
        "rem",
        "cosh",
        "is_finite",
        "unvmap_any",
        "unvmap_all",
        "unvmap_max",
        "stop_gradient",
    }
    for equation in closed.jaxpr.eqns:
        name = equation.primitive.name
        if name in _CALL_PRIMITIVES:
            _admit_interval_program(
                _as_closed(_nested_jaxpr(equation)), guarded_callbacks=guarded_callbacks
            )
        elif name == "scan":
            _admit_interval_program(
                _as_closed(equation.params["jaxpr"]), guarded_callbacks=guarded_callbacks
            )
        elif name == "cond":
            for branch in equation.params["branches"]:
                _admit_interval_program(_as_closed(branch), guarded_callbacks=True)
        elif (
            name in {"pure_callback", "io_callback", "debug_callback"}
            and guarded_callbacks
        ):
            continue
        elif (
            name in _STRUCTURAL
            or name in _HOST_INDEX_PRIMITIVES
            or name in arithmetic
            or name in _UNARY
        ):
            continue
        else:
            raise IntervalExtensionError(
                f"interval.primitive must have a supported extension; got {name!r}"
            )


def _prepare_host_jaxpr(closed: jax_core.ClosedJaxpr, /) -> jax_core.ClosedJaxpr:
    """Transfer captured constants once while retaining the program's equations."""

    def prepare_parameter(value: object) -> object:
        match value:
            case jax_core.ClosedJaxpr():
                return _prepare_host_jaxpr(value)
            case tuple() if type(value) is tuple:
                return tuple(prepare_parameter(item) for item in value)
            case _:
                return value

    constants = []
    for value in jax.device_get(closed.consts):
        constant = np.array(value, copy=True)
        constant.setflags(write=False)
        constants.append(constant)
    equations = [
        equation.replace(
            params={
                name: prepare_parameter(value) for name, value in equation.params.items()
            }
        )
        for equation in closed.jaxpr.eqns
    ]
    return closed.replace(eqns=equations, consts=constants)


def _host_interval_scan(
    equation: jax_core.JaxprEqn,
    values: Sequence[Interval],
    constant_bounds: Sequence[tuple[HostArray, HostArray, HostArray]],
    /,
) -> list[Interval]:
    """Interpret a fixed-length interval recurrence in the source scan order."""
    parameters = equation.params
    constants_, carry_, sequences_ = parameters["ft_in"].update(values).unpack()
    constants = list(constants_)
    carry = list(carry_)
    sequences = [(np.asarray(low), np.asarray(high)) for low, high in sequences_]
    num_carry = len(carry)
    _, output_variables = parameters["ft_out"].update(equation.outvars).unpack()
    outputs = [
        (
            np.empty(_array_aval(variable).shape, dtype=_array_aval(variable).dtype),
            np.empty(_array_aval(variable).shape, dtype=_array_aval(variable).dtype),
        )
        for variable in output_variables
    ]
    length = parameters["length"]
    indices = range(length - 1, -1, -1) if parameters["reverse"] else range(length)
    for index in indices:
        result = _evaluate_jaxpr(
            _as_closed(parameters["jaxpr"]),
            [*constants, *carry, *((low[index], high[index]) for low, high in sequences)],
            constant_bounds,
        )
        carry = result[:num_carry]
        for (lower, upper), (low, high) in zip(outputs, result[num_carry:], strict=True):
            lower[index] = low
            upper[index] = high
    return [*carry, *outputs]


def _host_interval_conditional(
    equation: jax_core.JaxprEqn,
    values: Sequence[Interval],
    constant_bounds: Sequence[tuple[HostArray, HostArray, HostArray]],
    /,
) -> list[Interval]:
    """Evaluate exactly the reachable branches, then union bounds in branch order."""
    count = len(equation.params["branches"])
    first, last = (min(max(int(value), 0), count - 1) for value in values[0])
    if first == last:
        return _evaluate_jaxpr(
            _as_closed(equation.params["branches"][first]),
            values[1:],
            constant_bounds,
        )
    branches = [
        _evaluate_jaxpr(_as_closed(branch), values[1:], constant_bounds)
        for branch in equation.params["branches"][first : last + 1]
    ]
    return [
        (
            np.minimum.reduce([value[0] for value in cases]),
            np.maximum.reduce([value[1] for value in cases]),
        )
        for cases in zip(*branches, strict=True)
    ]


def coefficient_interval(
    value: HostArray,
    constant_bounds: Sequence[tuple[HostArray, HostArray, HostArray]],
    /,
) -> Interval:
    """Apply the interpreter's retained source-construction coefficient bounds."""
    for nominal, lower, upper in constant_bounds:
        if value.shape == nominal.shape and np.array_equal(value, nominal):
            return lower, upper
    return value, value


def _evaluate_jaxpr(
    closed: jax_core.ClosedJaxpr,
    arguments: Sequence[Interval],
    /,
    constant_bounds: Sequence[tuple[HostArray, HostArray, HostArray]] = (),
) -> list[Interval]:
    environment: dict[jax_core.Var, Interval] = {}

    def read(atom: jax_core.Var | jax_core.Literal) -> Interval:
        match atom:
            case jax_core.Literal():
                value = np.asarray(atom.val)
                return value, value
            case _:
                return environment[atom]

    for variable, constant in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        value = np.asarray(constant)
        environment[variable] = coefficient_interval(value, constant_bounds)
    for variable, argument in zip(closed.jaxpr.invars, arguments, strict=True):
        environment[variable] = argument
    for equation in closed.jaxpr.eqns:
        values = [read(atom) for atom in equation.invars]
        name = equation.primitive.name
        if name in _STRUCTURAL:
            outputs = _structural(equation, values)
        elif name in _CALL_PRIMITIVES:
            outputs = _evaluate_jaxpr(
                _as_closed(_nested_jaxpr(equation)), values, constant_bounds
            )
        elif name == "scan":
            outputs = _host_interval_scan(equation, values, constant_bounds)
        elif name == "cond":
            outputs = _host_interval_conditional(equation, values, constant_bounds)
        elif all(_degenerate(value) for value in values) and name in {
            "iota",
            "gather",
            "dynamic_slice",
            "pad",
            "cumsum",
            "argmax",
            "argmin",
            "sort",
        }:
            outputs = _degenerate_bind(equation, values)
        elif name in {"argmax", "argmin"}:
            outputs = [_host_index_extremum(equation, values[0])]
        elif name in {"gather", "dynamic_slice"} and all(
            _degenerate(value) for value in values[1:]
        ):
            outputs = _structural(equation, values)
        elif name in {"gather", "dynamic_slice"}:
            shape = _array_aval(equation.outvars[0]).shape
            outputs = [
                (
                    np.full(shape, np.min(values[0][0]), dtype=values[0][0].dtype),
                    np.full(shape, np.max(values[0][1]), dtype=values[0][1].dtype),
                )
            ]
        elif name in {"scatter-mul", "scatter-add", "scatter-min", "scatter-max"}:
            outputs = _host_scatter_bounds(equation, values)
        elif name == "scatter" and _degenerate(values[1]):
            outputs = _structural(equation, values)
        elif name == "scatter":
            if equation.params["update_jaxpr"] is not None:
                raise IntervalExtensionError(
                    "interval.scatter must use overwrite updates; got a reduction update program"
                )
            minimum = np.minimum(np.min(values[0][0]), np.min(values[2][0]))
            maximum = np.maximum(np.max(values[0][1]), np.max(values[2][1]))
            shape = _array_aval(equation.outvars[0]).shape
            outputs = [
                (
                    np.full(shape, minimum, dtype=values[0][0].dtype),
                    np.full(shape, maximum, dtype=values[0][1].dtype),
                )
            ]
        elif name == "pad" and _degenerate(values[1]):
            outputs = _structural(equation, values)
        elif name == "iota":
            outputs = _degenerate_bind(equation, values)
        else:
            outputs = [_arithmetic(equation, values)]
        for variable, output in zip(equation.outvars, outputs, strict=True):
            environment[variable] = output
    return [read(atom) for atom in closed.jaxpr.outvars]


@dataclass(frozen=True, slots=True)
class PreparedIntervalFunction:
    """Interval extension of one traced map at a fixed batch capacity."""

    closed_jaxpr: jax_core.ClosedJaxpr
    input_dimension: int
    output_shape: tuple[int, ...]
    batch_capacity: int
    constant_bounds: tuple[tuple[HostArray, HostArray, HostArray], ...] = ()

    def evaluate(
        self, lower: npt.ArrayLike, upper: npt.ArrayLike, /
    ) -> tuple[HostTensor, HostTensor]:
        """Enclose the map's range over boxes ``[lower, upper]`` of shape ``(B, n)``."""
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        if (
            lower_.ndim != 2
            or lower_.shape[1] != self.input_dimension
            or upper_.shape != lower_.shape
        ):
            raise ValueError(
                f"interval.boxes must have shape (num_boxes, {self.input_dimension}); got {lower_.shape}, {upper_.shape}"
            )
        if np.any(lower_ > upper_):
            raise ValueError(
                "interval.boxes must have lower <= upper; got reversed endpoints"
            )
        if np.any(np.isnan(lower_)) or np.any(np.isnan(upper_)):
            raise ValueError("interval.boxes must have non-NaN endpoints; got NaN")
        count = lower_.shape[0]
        output_lower = np.empty((count, *self.output_shape), dtype=np.float64)
        output_upper = np.empty_like(output_lower)
        for start in range(0, count, self.batch_capacity):
            stop = min(start + self.batch_capacity, count)
            padding = self.batch_capacity - (stop - start)
            chunk_lower = np.concatenate(
                (lower_[start:stop], np.repeat(lower_[start : start + 1], padding, 0))
            )
            chunk_upper = np.concatenate(
                (upper_[start:stop], np.repeat(upper_[start : start + 1], padding, 0))
            )
            (result,) = _evaluate_jaxpr(
                self.closed_jaxpr, [(chunk_lower, chunk_upper)], self.constant_bounds
            )
            output_lower[start:stop] = np.asarray(result[0])[: stop - start]
            output_upper[start:stop] = np.asarray(result[1])[: stop - start]
        return output_lower, output_upper

    def evaluate_host_points(self, points: npt.ArrayLike, /) -> HostTensor:
        """Evaluate nominal points with NumPy using the prepared jaxpr and batch order."""
        points_ = np.asarray(points, dtype=np.float64)
        if points_.ndim != 2 or points_.shape[1] != self.input_dimension:
            raise ValueError(
                f"interval.points must have shape (num_points, {self.input_dimension}); got {points_.shape}"
            )
        count = points_.shape[0]
        result = np.empty((count, *self.output_shape), dtype=np.float64)
        for start in range(0, count, self.batch_capacity):
            stop = min(start + self.batch_capacity, count)
            padding = self.batch_capacity - (stop - start)
            chunk = np.concatenate(
                (
                    points_[start:stop],
                    np.repeat(points_[start : start + 1], padding, axis=0),
                )
            )
            # Preserve JAX special-value semantics in inactive conditional arms;
            # NaN/Inf remain output values for the consumer's finite check.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                (values,) = _evaluate_host_jaxpr(self.closed_jaxpr, [chunk])
            result[start:stop] = np.asarray(values)[: stop - start]
        return result


@checked
def prepare_interval_function(
    function: Callable[[Array], Array],
    input_dimension: int,
    /,
    *,
    batch_capacity: int = 64,
    constant_bounds: Sequence[tuple[HostArray, HostArray, HostArray]] = (),
) -> PreparedIntervalFunction:
    """Trace ``function`` (one point ``(n,)`` to an array) for interval evaluation."""
    if input_dimension < 1 or batch_capacity < 1:
        raise ValueError(
            f"interval.capacities must have positive input_dimension and batch_capacity; got {input_dimension}, {batch_capacity}"
        )
    coefficient_bounds = []
    for nominal, lower, upper in constant_bounds:
        nominal_, lower_, upper_ = (
            np.asarray(value, dtype=np.float64) for value in (nominal, lower, upper)
        )
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
                f"interval.constant_bounds must contain finite nominal coefficients in aligned finite intervals; got shapes {nominal_.shape}, {lower_.shape}, {upper_.shape}"
            )
        coefficient_bounds.append((nominal_, lower_, upper_))
    probe = jax.ShapeDtypeStruct((batch_capacity, input_dimension), np.float64)
    closed = _prepare_host_jaxpr(jax.make_jaxpr(jax.vmap(function))(probe))
    if len(closed.jaxpr.outvars) != 1:
        raise ValueError(
            f"interval.function must return one array; got {len(closed.jaxpr.outvars)} outputs"
        )
    output_aval = closed.jaxpr.outvars[0].aval
    if not isinstance(output_aval, ShapedArray):
        raise ValueError(
            f"interval.function must return one shaped array; got {type(output_aval).__qualname__}"
        )
    output_shape = tuple(output_aval.shape[1:])
    prepared = PreparedIntervalFunction(
        closed, input_dimension, output_shape, batch_capacity, tuple(coefficient_bounds)
    )
    _admit_interval_program(closed)
    return prepared


__all__ = [
    "IntervalExtensionError",
    "PreparedIntervalFunction",
    "device_interval_evaluation_error",
    "interval_add",
    "interval_divide",
    "interval_dot_general",
    "interval_dot",
    "interval_multiply",
    "interval_subtract",
    "interval_midpoint_radius",
    "prepare_interval_function",
]
