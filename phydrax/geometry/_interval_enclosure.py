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
That libm premise is explicit, not a platform-independent proof. Bounded
subdivision and Krawczyk certification consume these conditional enclosures.

Primitives without a sound interval extension, data-dependent control flow and
gathers driven by interval-valued indices are refused explicitly rather than
approximated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.core import ShapedArray
from jax.extend import core as jax_core


type HostArray = npt.NDArray[Any]
type Interval = tuple[HostArray, HostArray]


class IntervalExtensionError(ValueError):
    """A traced map contains an operation without a sound interval extension."""


_TWO_PI = 2.0 * math.pi
# One nextafter step covers correctly rounded basic operations. Elementary
# functions use an explicit four-ulp libm premise; NumPy does not itself provide
# a portable correctly-rounded transcendental-function guarantee.
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


def _degenerate(value: Interval, /) -> bool:
    return bool(np.array_equal(value[0], value[1]))


def interval_add(first: Interval, second: Interval, /) -> Interval:
    return _round_out(first[0] + second[0], first[1] + second[1])


def interval_subtract(first: Interval, second: Interval, /) -> Interval:
    return _round_out(first[0] - second[1], first[1] - second[0])


def interval_multiply(first: Interval, second: Interval, /) -> Interval:
    products = np.stack(
        np.broadcast_arrays(
            first[0] * second[0],
            first[0] * second[1],
            first[1] * second[0],
            first[1] * second[1],
        )
    )
    products = np.where(np.isnan(products), 0.0, products)
    return _round_out(np.min(products, axis=0), np.max(products, axis=0))


def _reciprocal(value: Interval, /) -> Interval:
    lower, upper = value
    straddles = (lower <= 0.0) & (upper >= 0.0)
    with np.errstate(divide="ignore"):
        low = np.where(straddles, -np.inf, 1.0 / upper)
        high = np.where(straddles, np.inf, 1.0 / lower)
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
    low_power = lower**exponent
    high_power = upper**exponent
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


def _interval_cos(value: Interval, /) -> Interval:
    lower, upper = value
    wide = ~(upper - lower < _TWO_PI) | ~np.isfinite(lower) | ~np.isfinite(upper)
    endpoints = np.stack((np.cos(lower), np.cos(upper)))
    low = np.min(endpoints, axis=0)
    high = np.max(endpoints, axis=0)
    high = np.where(_contains_phase(lower, upper, 0.0) | wide, 1.0, high)
    low = np.where(_contains_phase(lower, upper, math.pi) | wide, -1.0, low)
    low, high = _round_out(low, high, _TRANSCENDENTAL_STEPS)
    return np.maximum(low, -1.0), np.minimum(high, 1.0)


def _interval_sin(value: Interval, /) -> Interval:
    lower, upper = value
    wide = ~(upper - lower < _TWO_PI) | ~np.isfinite(lower) | ~np.isfinite(upper)
    endpoints = np.stack((np.sin(lower), np.sin(upper)))
    low = np.min(endpoints, axis=0)
    high = np.max(endpoints, axis=0)
    high = np.where(_contains_phase(lower, upper, 0.5 * math.pi) | wide, 1.0, high)
    low = np.where(_contains_phase(lower, upper, -0.5 * math.pi) | wide, -1.0, low)
    low, high = _round_out(low, high, _TRANSCENDENTAL_STEPS)
    return np.maximum(low, -1.0), np.minimum(high, 1.0)


def _interval_tan(value: Interval, /) -> Interval:
    lower, upper = value
    same_branch = np.floor((lower + 0.5 * math.pi) / math.pi) == np.floor(
        (upper + 0.5 * math.pi) / math.pi
    )
    interior = (
        same_branch
        & (np.abs(np.cos(lower)) > 1.0e-300)
        & (np.abs(np.cos(upper)) > 1.0e-300)
    )
    low = np.where(interior, np.tan(lower), -np.inf)
    high = np.where(interior, np.tan(upper), np.inf)
    return _round_out(low, high, _TRANSCENDENTAL_STEPS)


def _monotone(
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
        raise IntervalExtensionError("sqrt encloses a strictly negative interval.")
    return _monotone(np.sqrt, True, 1)((np.maximum(value[0], 0.0), value[1]))


def _interval_abs(value: Interval, /) -> Interval:
    lower, upper = value
    straddles = (lower <= 0.0) & (upper >= 0.0)
    low = np.where(straddles, 0.0, np.minimum(np.abs(lower), np.abs(upper)))
    return low, np.maximum(np.abs(lower), np.abs(upper))


def _midpoint_radius(value: Interval, /) -> Interval:
    lower, upper = value
    middle = 0.5 * (lower + upper)
    radius = np.nextafter(np.maximum(middle - lower, upper - middle), np.inf)
    return middle, radius


def interval_dot_general(
    first: Interval, second: Interval, dimension_numbers: Any, /
) -> Interval:
    """Midpoint-radius enclosure of a general tensor contraction."""
    first_mid, first_rad = _midpoint_radius(first)
    second_mid, second_rad = _midpoint_radius(second)

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
    radius = (
        contract(np.abs(first_mid), second_rad)
        + contract(first_rad, np.abs(second_mid))
        + contract(first_rad, second_rad)
    )
    (contracting, _), _ = dimension_numbers
    length = 1 + math.prod(first_mid.shape[axis] for axis in contracting)
    unit = np.finfo(middle.dtype).eps
    # Standard a-priori bound for a floating dot product of the given length,
    # plus the rounding of the radius accumulation itself.
    gamma = 2.0 * length * unit / (1.0 - 2.0 * length * unit)
    # JAX backends may flush subnormal products; include the normal-threshold
    # allowance rather than assuming gradual underflow in a tensor contraction.
    tiny = np.finfo(middle.dtype).tiny
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
    unit = np.finfo(lower.dtype).eps
    gamma = length * unit / (1.0 - length * unit)
    scale = np.sum(np.maximum(np.abs(value[0]), np.abs(value[1])), axis=axes_)
    return _round_out(lower - gamma * scale, upper + gamma * scale)


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
            return ~equal_possible, ~equal_certain
        case _:
            raise IntervalExtensionError(f"Unknown comparison primitive {name!r}.")


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
    "sin": _interval_sin,
    "cos": _interval_cos,
    "tan": _interval_tan,
    "sqrt": _interval_sqrt,
    "abs": _interval_abs,
    "exp": _monotone(np.exp, True, _TRANSCENDENTAL_STEPS),
    "log": _monotone(np.log, True, _TRANSCENDENTAL_STEPS),
    "atan": _monotone(np.arctan, True, _TRANSCENDENTAL_STEPS),
    "tanh": _monotone(np.tanh, True, _TRANSCENDENTAL_STEPS),
    "floor": _monotone(np.floor, True, 0),
    "ceil": _monotone(np.ceil, True, 0),
    "sign": _monotone(np.sign, True, 0),
    "rsqrt": _monotone(lambda value: 1.0 / np.sqrt(value), False, 2),
}


def _structural(equation: Any, values: Sequence[Interval], /) -> list[Interval]:
    def bind(arrays: Sequence[HostArray]) -> list[HostArray]:
        name, parameters = equation.primitive.name, equation.params
        if name == "broadcast_in_dim":
            shape = [1] * len(parameters["shape"])
            for size, axis in zip(
                arrays[0].shape, parameters["broadcast_dimensions"], strict=True
            ):
                shape[axis] = size
            return [np.broadcast_to(arrays[0].reshape(shape), parameters["shape"])]
        if name == "reshape":
            array = arrays[0]
            dimensions = parameters.get("dimensions")
            if dimensions is not None:
                array = np.transpose(array, dimensions)
            return [np.reshape(array, parameters["new_sizes"])]
        if name == "transpose":
            return [np.transpose(arrays[0], parameters["permutation"])]
        if name == "squeeze":
            return [np.squeeze(arrays[0], axis=parameters["dimensions"])]
        if name == "slice":
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
        if name == "concatenate":
            return [np.concatenate(arrays, axis=parameters["dimension"])]
        if name == "rev":
            return [np.flip(arrays[0], axis=parameters["dimensions"])]
        if name in {"copy", "copy_p"}:
            return [arrays[0]]
        if name == "real":
            return [np.real(arrays[0])]
        output = equation.primitive.bind(
            *(jnp.asarray(array) for array in arrays), **equation.params
        )
        if equation.primitive.multiple_results:
            return [np.asarray(item) for item in output]
        return [np.asarray(output)]

    lowers = bind([value[0] for value in values])
    uppers = bind([value[1] for value in values])
    return list(zip(lowers, uppers, strict=True))


def _degenerate_bind(equation: Any, values: Sequence[Interval], /) -> list[Interval]:
    """Evaluate an operation whose inputs are all exact constants."""
    output = equation.primitive.bind(
        *(jnp.asarray(value[0]) for value in values), **equation.params
    )
    outputs = output if equation.primitive.multiple_results else [output]
    return [(np.asarray(item), np.asarray(item)) for item in outputs]


def _nested_jaxpr(equation: Any, /) -> Any:
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in equation.params:
            return equation.params[key]
    raise IntervalExtensionError(
        f"Primitive {equation.primitive.name!r} has no nested jaxpr to interpret."
    )


def _as_closed(jaxpr: Any, /) -> Any:
    if isinstance(jaxpr, jax_core.ClosedJaxpr):
        return jaxpr
    return jax_core.ClosedJaxpr(jaxpr, ())


def _arithmetic(equation: Any, values: Sequence[Interval], /) -> Interval:
    name = equation.primitive.name
    match name:
        case "add" | "add_any":
            return interval_add(values[0], values[1])
        case "sub":
            return interval_subtract(values[0], values[1])
        case "mul":
            return interval_multiply(values[0], values[1])
        case "div":
            return interval_divide(values[0], values[1])
        case "neg":
            return -values[0][1], -values[0][0]
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
            if not _degenerate(values[0]):
                raise IntervalExtensionError(
                    "Nonconstant real-to-integer conversions are unsupported."
                )
            return lower.astype(dtype), upper.astype(dtype)
        case "lt" | "le" | "gt" | "ge" | "eq" | "ne":
            return _compare(name, values[0], values[1])
        case "and":
            return values[0][0] & values[1][0], values[0][1] & values[1][1]
        case "or":
            return values[0][0] | values[1][0], values[0][1] | values[1][1]
        case "not":
            return ~values[0][1], ~values[0][0]
        case "clamp":
            minimum, operand, maximum = values
            return (
                np.maximum(minimum[0], np.minimum(operand[0], maximum[0])),
                np.maximum(minimum[1], np.minimum(operand[1], maximum[1])),
            )
        case "select_n":
            return _select(values[0], values[1:])
        case _:
            unary = _UNARY.get(name)
            if unary is None:
                raise IntervalExtensionError(
                    f"No interval extension is available for primitive {name!r}."
                )
            return unary(values[0])


def _evaluate_jaxpr(
    closed: Any,
    arguments: Sequence[Interval],
    /,
    constant_bounds: Sequence[tuple[HostArray, HostArray, HostArray]] = (),
) -> list[Interval]:
    environment: dict[Any, Interval] = {}

    def read(atom: Any) -> Interval:
        if isinstance(atom, jax_core.Literal):
            value = np.asarray(atom.val)
            return value, value
        return environment[atom]

    for variable, constant in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        value = np.asarray(constant)
        enclosure = (value, value)
        for nominal, lower, upper in constant_bounds:
            if value.shape == nominal.shape and np.array_equal(value, nominal):
                enclosure = (lower, upper)
                break
        environment[variable] = enclosure
    for variable, argument in zip(closed.jaxpr.invars, arguments, strict=True):
        environment[variable] = argument
    for equation in closed.jaxpr.eqns:
        values = [read(atom) for atom in equation.invars]
        name = equation.primitive.name
        if name in _STRUCTURAL:
            outputs = _structural(equation, values)
        elif name in {
            "pjit",
            "closed_call",
            "core_call",
            "custom_jvp_call",
            "remat",
            "checkpoint",
            "custom_vjp_call",
            "custom_vjp_call_jaxpr",
            "jit",
        }:
            outputs = _evaluate_jaxpr(
                _as_closed(_nested_jaxpr(equation)), values, constant_bounds
            )
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
        elif name in {"gather", "dynamic_slice"} and all(
            _degenerate(value) for value in values[1:]
        ):
            outputs = _structural(equation, values)
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

    closed_jaxpr: Any
    input_dimension: int
    output_shape: tuple[int, ...]
    batch_capacity: int
    constant_bounds: tuple[tuple[HostArray, HostArray, HostArray], ...] = ()

    def evaluate(self, lower: HostArray, upper: HostArray, /) -> Interval:
        """Enclose the map's range over boxes ``[lower, upper]`` of shape ``(B, n)``."""
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        if (
            lower_.ndim != 2
            or lower_.shape[1] != self.input_dimension
            or upper_.shape != lower_.shape
        ):
            raise ValueError(
                f"Interval boxes must have shape (num_boxes, {self.input_dimension})."
            )
        if np.any(lower_ > upper_):
            raise ValueError("Interval boxes require lower <= upper.")
        if np.any(np.isnan(lower_)) or np.any(np.isnan(upper_)):
            raise ValueError("Interval box endpoints must not be NaN.")
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
            output_lower[start:stop] = result[0][: stop - start]
            output_upper[start:stop] = result[1][: stop - start]
        return output_lower, output_upper


def prepare_interval_function(
    function: Callable[[Array], Array],
    input_dimension: int,
    /,
    *,
    batch_capacity: int = 64,
    constant_bounds: Sequence[tuple[HostArray, HostArray, HostArray]] = (),
) -> PreparedIntervalFunction:
    """Trace ``function`` (one point ``(n,)`` to an array) for interval evaluation."""
    if not callable(function):
        raise TypeError("function must be callable.")
    if input_dimension < 1 or batch_capacity < 1:
        raise ValueError("input_dimension and batch_capacity must be positive.")
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
                "Constant coefficient enclosures must contain their nominal arrays."
            )
        coefficient_bounds.append((nominal_, lower_, upper_))
    probe = jnp.zeros((batch_capacity, input_dimension), dtype=jnp.float64)
    closed = jax.make_jaxpr(jax.vmap(function))(probe)
    if len(closed.jaxpr.outvars) != 1:
        raise ValueError("Interval functions must return exactly one array.")
    output_aval = closed.jaxpr.outvars[0].aval
    if not isinstance(output_aval, ShapedArray):
        raise ValueError("Interval functions must return exactly one shaped array.")
    output_shape = tuple(output_aval.shape[1:])
    prepared = PreparedIntervalFunction(
        closed, input_dimension, output_shape, batch_capacity, tuple(coefficient_bounds)
    )
    # Validate once that every traced primitive has an interval extension.
    prepared.evaluate(np.zeros((1, input_dimension)), np.ones((1, input_dimension)))
    return prepared


__all__ = [
    "IntervalExtensionError",
    "PreparedIntervalFunction",
    "interval_add",
    "interval_divide",
    "interval_dot_general",
    "interval_multiply",
    "interval_subtract",
    "prepare_interval_function",
]
