#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Box enclosures of implicit boundary fields and their gradients.

Three bound sources exist, in decreasing strength:

- ``interval``: the source's own field program is evaluated in outward-rounded
  interval arithmetic with forward-mode interval tangents. Values and gradients
  are rigorous enclosures of the real-arithmetic program on every box. Selects,
  ``max``/``min`` and ``abs`` whose branch is undecided take the componentwise
  hull of the branch gradients, which contains the Clarke generalized gradient
  of a continuous piecewise-smooth field. Programs with primitives outside the
  admitted rule set are refused at preparation rather than sampled.
- ``lipschitz``: the field certificate's declared Lipschitz bound ``L`` and
  evaluation error ``e`` give the value enclosure ``f(c) +/- (L |h| + e)`` and
  the gradient enclosure ``[-L, L]`` per component. Values are rigorous; the
  gradient enclosure never certifies regularity.
- ``sampled``: corner and center samples of the field and its gradient. No
  enclosure claim is made.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import final, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.extend import core as jax_core
from jax.typing import DTypeLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import charge_native_geometry_queries
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._atlas import BoundaryAtlas
from .._certificate import FieldCertificate
from .._contracts import CompiledGeometry, GeometryKernel
from ..design._schema import DesignState
from ._policy import ImplicitDiscoveryEnclosure
from ._projection import _field_and_gradient


_EPSILON = float(np.finfo(np.float64).eps)
# Correctly rounded IEEE operations are widened by one ulp; library
# transcendental functions carry a documented few-ulp error and are widened by
# `_TRANSCENDENTAL_ULPS` before the final outward step.
_TRANSCENDENTAL_ULPS = 4.0
_BOX_CHUNK = 4096
# Directional-derivative seeds: the axes first, then the face and body
# diagonals. Each direction's derivative is enclosed independently, so an
# undecided max/min keeps the scalar hull of its branch derivatives instead of a
# componentwise gradient hull that would contain zero at every sharp crease.
DISCOVERY_DIRECTIONS = np.asarray(
    (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
    ),
    dtype=np.float64,
)
_SMALL_BOX_CHUNK = 256
_AXES = 3
_STRUCTURAL = frozenset(
    {
        "broadcast_in_dim",
        "concatenate",
        "copy",
        "copy_p",
        "dynamic_slice",
        "dynamic_update_slice",
        "expand_dims",
        "gather",
        "pad",
        "reshape",
        "rev",
        "slice",
        "split",
        "squeeze",
        "stack",
        "transpose",
    }
)
_CALLS = frozenset(
    {"jit", "pjit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call"}
)
_UNARY = frozenset(
    {
        "abs",
        "atan",
        "ceil",
        "cos",
        "exp",
        "expm1",
        "floor",
        "integer_pow",
        "log",
        "log1p",
        "logistic",
        "neg",
        "round",
        "rsqrt",
        "sign",
        "sin",
        "sqrt",
        "square",
        "tanh",
    }
)
_BINARY = frozenset({"add", "add_any", "sub", "mul", "div", "max", "min"})
_COMPARISON = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})
_LOGIC = frozenset({"and", "or", "not"})
_OTHER = frozenset(
    {
        "checkpoint",
        "remat",
        "clamp",
        "cond",
        "convert_element_type",
        "cumsum",
        "dot_general",
        "iota",
        "reduce_and",
        "reduce_max",
        "reduce_min",
        "reduce_or",
        "reduce_precision",
        "reduce_sum",
        "select_n",
        "stop_gradient",
    }
)
_SUPPORTED = _STRUCTURAL | _CALLS | _UNARY | _BINARY | _COMPARISON | _LOGIC | _OTHER
ENCLOSURE_ROUNDING_MODEL = (
    "outward-nextafter; one ulp for IEEE arithmetic, "
    f"{_TRANSCENDENTAL_ULPS:g} ulps for transcendental functions"
)


@dataclass(frozen=True, slots=True)
class _Interval:
    """Float interval with forward interval tangents shaped ``(inputs,) + shape``."""

    lower: Array
    upper: Array
    tangent_lower: Array
    tangent_upper: Array


@dataclass(frozen=True, slots=True)
class _Exact:
    """Integer or boolean data known exactly."""

    value: Array


@dataclass(frozen=True, slots=True)
class _Truth:
    """Three-valued boolean: each entry may be true, false, or both."""

    possible_true: Array
    possible_false: Array


type _Value = _Interval | _Exact | _Truth


def _down(value: Array) -> Array:
    return jnp.where(jnp.isnan(value), -jnp.inf, jnp.nextafter(value, -jnp.inf))


def _up(value: Array) -> Array:
    return jnp.where(jnp.isnan(value), jnp.inf, jnp.nextafter(value, jnp.inf))


def _widened(lower: Array, upper: Array, ulps: float = 0.0) -> tuple[Array, Array]:
    if ulps:
        lower = lower - ulps * jnp.finfo(lower.dtype).eps * jnp.abs(lower)
        upper = upper + ulps * jnp.finfo(upper.dtype).eps * jnp.abs(upper)
    return _down(lower), _up(upper)


def _product(
    first_lower: Array, first_upper: Array, second_lower: Array, second_upper: Array
) -> tuple[Array, Array]:
    products = jnp.stack(
        (
            first_lower * second_lower,
            first_lower * second_upper,
            first_upper * second_lower,
            first_upper * second_upper,
        )
    )
    # Infinite endpoints denote unbounded, never attained, values, so the
    # extended-interval convention 0 * inf = 0 is sound.
    products = jnp.where(jnp.isnan(products), 0.0, products)
    return _widened(jnp.min(products, axis=0), jnp.max(products, axis=0))


def _reciprocal(lower: Array, upper: Array) -> tuple[Array, Array]:
    excludes_zero = (lower > 0.0) | (upper < 0.0)
    return (
        jnp.where(excludes_zero, _down(1.0 / upper), -jnp.inf),
        jnp.where(excludes_zero, _up(1.0 / lower), jnp.inf),
    )


def _magnitude(lower: Array, upper: Array) -> Array:
    return jnp.maximum(jnp.abs(lower), jnp.abs(upper))


def _mignitude(lower: Array, upper: Array) -> Array:
    return jnp.where(
        (lower <= 0.0) & (upper >= 0.0), 0.0, jnp.minimum(jnp.abs(lower), jnp.abs(upper))
    )


def _ranked(value: _Interval, rank: int) -> _Interval:
    """Insert leading value axes so a scalar operand broadcasts against ``rank``."""

    missing = rank - value.lower.ndim
    if missing <= 0:
        return value
    shape = (1,) * missing + value.lower.shape
    tangent_shape = (value.tangent_lower.shape[0], *shape)
    return _Interval(
        value.lower.reshape(shape),
        value.upper.reshape(shape),
        value.tangent_lower.reshape(tangent_shape),
        value.tangent_upper.reshape(tangent_shape),
    )


def _aligned(*values: _Interval) -> list[_Interval]:
    rank = max(value.lower.ndim for value in values)
    return [_ranked(value, rank) for value in values]


def _hull(first: _Interval, second: _Interval) -> _Interval:
    return _Interval(
        jnp.minimum(first.lower, second.lower),
        jnp.maximum(first.upper, second.upper),
        jnp.minimum(first.tangent_lower, second.tangent_lower),
        jnp.maximum(first.tangent_upper, second.tangent_upper),
    )


def _where(mask: Array, when_true: _Interval, when_false: _Interval) -> _Interval:
    tangent_mask = mask[None]
    return _Interval(
        jnp.where(mask, when_true.lower, when_false.lower),
        jnp.where(mask, when_true.upper, when_false.upper),
        jnp.where(tangent_mask, when_true.tangent_lower, when_false.tangent_lower),
        jnp.where(tangent_mask, when_true.tangent_upper, when_false.tangent_upper),
    )


def _select(truth: _Truth, when_false: _Interval, when_true: _Interval) -> _Interval:
    when_false, when_true = _aligned(when_false, when_true)
    chosen = _where(truth.possible_true, when_true, when_false)
    both = truth.possible_true & truth.possible_false
    return _where(both, _hull(when_false, when_true), chosen)


def _scale_tangent(value: _Interval, lower: Array, upper: Array) -> tuple[Array, Array]:
    """Tangent of ``g(value)`` for a derivative enclosure ``[lower, upper]`` of ``g'``."""

    return _product(value.tangent_lower, value.tangent_upper, lower, upper)


def _constant(value: Array, inputs: int) -> _Interval:
    array = jnp.asarray(value)
    zeros = jnp.zeros((inputs, *array.shape), dtype=array.dtype)
    return _Interval(array, array, zeros, zeros)


def _cos_range(lower: Array, upper: Array) -> tuple[Array, Array]:
    period = 2.0 * math.pi
    wide = upper - lower >= period
    first_even = jnp.ceil(lower / period) * period
    first_odd = jnp.ceil((lower - math.pi) / period) * period + math.pi
    contains_even = first_even <= upper
    contains_odd = first_odd <= upper
    ends_lower = jnp.minimum(jnp.cos(lower), jnp.cos(upper))
    ends_upper = jnp.maximum(jnp.cos(lower), jnp.cos(upper))
    value_lower = jnp.where(wide | contains_odd, -1.0, ends_lower)
    value_upper = jnp.where(wide | contains_even, 1.0, ends_upper)
    value_lower, value_upper = _widened(value_lower, value_upper, _TRANSCENDENTAL_ULPS)
    return jnp.maximum(value_lower, -1.0), jnp.minimum(value_upper, 1.0)


def _monotone_unary(
    value: _Interval,
    function: Callable[[Array], Array],
    derivative: tuple[Array, Array],
    *,
    increasing: bool,
    ulps: float,
) -> _Interval:
    first = function(value.lower)
    second = function(value.upper)
    lower, upper = (first, second) if increasing else (second, first)
    lower, upper = _widened(lower, upper, ulps)
    tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
    return _Interval(lower, upper, tangent_lower, tangent_upper)


def _unary(name: str, value: _Interval, params: dict[str, object]) -> _Interval:
    lower, upper = value.lower, value.upper
    zeros = jnp.zeros_like(value.tangent_lower)
    match name:
        case "neg":
            return _Interval(-upper, -lower, -value.tangent_upper, -value.tangent_lower)
        case "abs":
            positive = lower >= 0.0
            negative = upper <= 0.0
            magnitude = _magnitude(value.tangent_lower, value.tangent_upper)
            return _Interval(
                _mignitude(lower, upper),
                _magnitude(lower, upper),
                jnp.where(
                    positive,
                    value.tangent_lower,
                    jnp.where(negative, -value.tangent_upper, -magnitude),
                ),
                jnp.where(
                    positive,
                    value.tangent_upper,
                    jnp.where(negative, -value.tangent_lower, magnitude),
                ),
            )
        case "sign" | "floor" | "ceil" | "round":
            function = {
                "sign": jnp.sign,
                "floor": jnp.floor,
                "ceil": jnp.ceil,
                "round": jnp.round,
            }[name]
            return _Interval(function(lower), function(upper), zeros, zeros)
        case "exp" | "expm1":
            function = jnp.exp if name == "exp" else jnp.expm1
            derivative = _widened(jnp.exp(lower), jnp.exp(upper), _TRANSCENDENTAL_ULPS)
            return _monotone_unary(
                value,
                function,
                derivative,
                increasing=True,
                ulps=_TRANSCENDENTAL_ULPS,
            )
        case "log" | "log1p":
            if name == "log":
                argument_lower, argument_upper = lower, upper
            else:
                argument_lower, argument_upper = _widened(lower + 1.0, upper + 1.0)
            defined = argument_upper > 0.0
            positive = argument_lower > 0.0
            result_lower, result_upper = _widened(
                jnp.log(jnp.where(positive, argument_lower, 1.0)),
                jnp.log(jnp.where(defined, argument_upper, 1.0)),
                _TRANSCENDENTAL_ULPS,
            )
            derivative = _reciprocal(argument_lower, argument_upper)
            tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
            return _Interval(
                jnp.where(positive, result_lower, -jnp.inf),
                jnp.where(defined, result_upper, jnp.inf),
                tangent_lower,
                tangent_upper,
            )
        case "sqrt":
            defined = upper >= 0.0
            root_lower, root_upper = _widened(
                jnp.sqrt(jnp.maximum(lower, 0.0)), jnp.sqrt(jnp.maximum(upper, 0.0))
            )
            root_lower = jnp.maximum(root_lower, 0.0)
            derivative = _reciprocal(2.0 * root_lower, 2.0 * root_upper)
            derivative = (
                jnp.where(root_lower > 0.0, derivative[0], 0.0),
                jnp.where(root_lower > 0.0, derivative[1], jnp.inf),
            )
            tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
            return _Interval(
                jnp.where(defined, root_lower, -jnp.inf),
                jnp.where(defined, root_upper, jnp.inf),
                tangent_lower,
                tangent_upper,
            )
        case "rsqrt":
            positive = lower > 0.0
            root = _unary("sqrt", value, params)
            reciprocal = _reciprocal(root.lower, root.upper)
            # d rsqrt(x) = -rsqrt(x)^3 / 2 dx.
            cube_lower, cube_upper = _product(
                *reciprocal, *_product(*reciprocal, *reciprocal)
            )
            derivative = (-0.5 * cube_upper, -0.5 * cube_lower)
            tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
            return _Interval(
                jnp.where(positive, reciprocal[0], -jnp.inf),
                jnp.where(positive, reciprocal[1], jnp.inf),
                tangent_lower,
                tangent_upper,
            )
        case "tanh":
            result = _widened(jnp.tanh(lower), jnp.tanh(upper), _TRANSCENDENTAL_ULPS)
            square_lower = _mignitude(*result) ** 2
            square_upper = _magnitude(*result) ** 2
            derivative = _widened(1.0 - square_upper, 1.0 - square_lower)
            tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
            return _Interval(*result, tangent_lower, tangent_upper)
        case "logistic":
            result = _widened(
                jax.nn.sigmoid(lower), jax.nn.sigmoid(upper), _TRANSCENDENTAL_ULPS
            )
            ends = jnp.stack(
                (result[0] * (1.0 - result[0]), result[1] * (1.0 - result[1]))
            )
            peak = (result[0] <= 0.5) & (result[1] >= 0.5)
            derivative = _widened(
                jnp.min(ends, axis=0), jnp.where(peak, 0.25, jnp.max(ends, axis=0))
            )
            tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
            return _Interval(*result, tangent_lower, tangent_upper)
        case "atan":
            derivative = _widened(
                1.0 / (1.0 + _magnitude(lower, upper) ** 2),
                1.0 / (1.0 + _mignitude(lower, upper) ** 2),
            )
            return _monotone_unary(
                value,
                jnp.arctan,
                derivative,
                increasing=True,
                ulps=_TRANSCENDENTAL_ULPS,
            )
        case "cos":
            result = _cos_range(lower, upper)
            sine = _cos_range(lower - 0.5 * math.pi, upper - 0.5 * math.pi)
            tangent_lower, tangent_upper = _scale_tangent(value, -sine[1], -sine[0])
            return _Interval(*result, tangent_lower, tangent_upper)
        case "sin":
            result = _cos_range(lower - 0.5 * math.pi, upper - 0.5 * math.pi)
            cosine = _cos_range(lower, upper)
            tangent_lower, tangent_upper = _scale_tangent(value, *cosine)
            return _Interval(*result, tangent_lower, tangent_upper)
        case "square":
            return _power(value, 2)
        case "integer_pow":
            exponent = params["y"]
            if not isinstance(exponent, int):
                raise TypeError("integer_pow requires an integer exponent.")
            return _power(value, exponent)
        case _:
            raise ValueError(f"Interval enclosure has no unary rule for {name!r}.")


def _power(value: _Interval, exponent: int) -> _Interval:
    lower, upper = value.lower, value.upper
    if exponent == 0:
        ones = jnp.ones_like(lower)
        return _Interval(
            ones,
            ones,
            jnp.zeros_like(value.tangent_lower),
            jnp.zeros_like(value.tangent_upper),
        )
    if exponent < 0:
        positive = _power(value, -exponent)
        result = _reciprocal(positive.lower, positive.upper)
        # d x^-n = -n x^-(n+1) dx.
        denominator = _power(value, -exponent + 1)
        scale = _reciprocal(denominator.lower, denominator.upper)
        derivative = _product(
            *scale, jnp.asarray(float(exponent)), jnp.asarray(float(exponent))
        )
        tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
        return _Interval(*result, tangent_lower, tangent_upper)
    if exponent % 2 == 0:
        result = _widened(
            _mignitude(lower, upper) ** exponent, _magnitude(lower, upper) ** exponent
        )
        result = (jnp.maximum(result[0], 0.0), result[1])
    else:
        result = _widened(lower**exponent, upper**exponent)
    if exponent == 1:
        derivative = (jnp.ones_like(lower), jnp.ones_like(lower))
    else:
        base = _power(
            _Interval(lower, upper, value.tangent_lower, value.tangent_upper),
            exponent - 1,
        )
        derivative = _product(
            base.lower,
            base.upper,
            jnp.asarray(float(exponent)),
            jnp.asarray(float(exponent)),
        )
    tangent_lower, tangent_upper = _scale_tangent(value, *derivative)
    return _Interval(*result, tangent_lower, tangent_upper)


def _binary(
    name: str, first: _Interval, second: _Interval, same_operand: bool
) -> _Interval:
    first, second = _aligned(first, second)
    match name:
        case "add" | "add_any":
            return _Interval(
                *_widened(first.lower + second.lower, first.upper + second.upper),
                *_widened(
                    first.tangent_lower + second.tangent_lower,
                    first.tangent_upper + second.tangent_upper,
                ),
            )
        case "sub":
            return _Interval(
                *_widened(first.lower - second.upper, first.upper - second.lower),
                *_widened(
                    first.tangent_lower - second.tangent_upper,
                    first.tangent_upper - second.tangent_lower,
                ),
            )
        case "mul":
            if same_operand:
                return _power(first, 2)
            value = _product(first.lower, first.upper, second.lower, second.upper)
            left = _product(
                first.tangent_lower, first.tangent_upper, second.lower, second.upper
            )
            right = _product(
                first.lower, first.upper, second.tangent_lower, second.tangent_upper
            )
            return _Interval(*value, *_widened(left[0] + right[0], left[1] + right[1]))
        case "div":
            reciprocal = _reciprocal(second.lower, second.upper)
            value = _product(first.lower, first.upper, *reciprocal)
            # d(a / b) = (da - (a / b) db) / b.
            quotient_tangent = _product(
                value[0], value[1], second.tangent_lower, second.tangent_upper
            )
            numerator = _widened(
                first.tangent_lower - quotient_tangent[1],
                first.tangent_upper - quotient_tangent[0],
            )
            tangent = _product(*numerator, *reciprocal)
            return _Interval(*value, *tangent)
        case "max" | "min":
            larger = name == "max"
            function = jnp.maximum if larger else jnp.minimum
            if larger:
                first_wins = first.lower > second.upper
                second_wins = second.lower > first.upper
            else:
                first_wins = first.upper < second.lower
                second_wins = second.upper < first.lower
            tangents = _where(
                first_wins, first, _where(second_wins, second, _hull(first, second))
            )
            return _Interval(
                function(first.lower, second.lower),
                function(first.upper, second.upper),
                tangents.tangent_lower,
                tangents.tangent_upper,
            )
        case _:
            raise ValueError(f"Interval enclosure has no binary rule for {name!r}.")


def _compare(name: str, first: _Interval, second: _Interval) -> _Truth:
    match name:
        case "lt":
            return _Truth(first.lower < second.upper, first.upper >= second.lower)
        case "le":
            return _Truth(first.lower <= second.upper, first.upper > second.lower)
        case "gt":
            return _Truth(first.upper > second.lower, first.lower <= second.upper)
        case "ge":
            return _Truth(first.upper >= second.lower, first.lower < second.upper)
        case "eq" | "ne":
            overlap = (first.lower <= second.upper) & (second.lower <= first.upper)
            equal = (
                (first.lower == first.upper)
                & (second.lower == second.upper)
                & (first.lower == second.lower)
            )
            if name == "eq":
                return _Truth(overlap, ~equal)
            return _Truth(~equal, overlap)
        case _:
            raise ValueError(f"Interval enclosure has no comparison rule for {name!r}.")


def _truth(value: _Value) -> _Truth:
    match value:
        case _Truth():
            return value
        case _Exact():
            boolean = value.value.astype(jnp.bool_)
            return _Truth(boolean, ~boolean)
        case _Interval():
            raise ValueError("A float interval cannot act as a boolean.")


def _interval(value: _Value, inputs: int) -> _Interval:
    match value:
        case _Interval():
            return value
        case _Exact():
            return _constant(value.value.astype(jnp.float64), inputs)
        case _Truth():
            zeros = jnp.zeros((inputs, *value.possible_true.shape), dtype=jnp.float64)
            return _Interval(
                jnp.where(value.possible_false, 0.0, 1.0),
                jnp.where(value.possible_true, 1.0, 0.0),
                zeros,
                zeros,
            )


def _dot_general(
    first: _Interval, second: _Interval, params: dict[str, object]
) -> _Interval:
    def dot(left: Array, right: Array) -> Array:
        return jax.lax.dot_general(
            left,
            right,
            params["dimension_numbers"],  # ty: ignore[invalid-argument-type]
            preferred_element_type=jnp.float64,
        )

    def interval_dot(
        left_lower: Array, left_upper: Array, right_lower: Array, right_upper: Array
    ) -> tuple[Array, Array]:
        # Midpoint-radius product with a Rump-style summation error bound.
        left_mid = 0.5 * (left_lower + left_upper)
        right_mid = 0.5 * (right_lower + right_upper)
        left_radius = _up(jnp.maximum(left_upper - left_mid, left_mid - left_lower))
        right_radius = _up(jnp.maximum(right_upper - right_mid, right_mid - right_lower))
        center = dot(left_mid, right_mid)
        radius = (
            dot(jnp.abs(left_mid), right_radius)
            + dot(left_radius, jnp.abs(right_mid))
            + dot(left_radius, right_radius)
        )
        contraction = max(
            1,
            math.prod(
                left_lower.shape[axis]
                for axis in params["dimension_numbers"][0][0]  # ty: ignore[not-subscriptable]
            ),
        )
        rounding = (
            (contraction + 2)
            * _EPSILON
            * (dot(jnp.abs(left_mid) + left_radius, jnp.abs(right_mid) + right_radius))
        )
        return _widened(center - radius - rounding, center + radius + rounding)

    value = interval_dot(first.lower, first.upper, second.lower, second.upper)
    left = jax.vmap(interval_dot, in_axes=(0, 0, None, None))(
        first.tangent_lower, first.tangent_upper, second.lower, second.upper
    )
    right = jax.vmap(interval_dot, in_axes=(None, None, 0, 0))(
        first.lower, first.upper, second.tangent_lower, second.tangent_upper
    )
    return _Interval(*value, *_widened(left[0] + right[0], left[1] + right[1]))


def _reduce(name: str, value: _Interval, params: dict[str, object]) -> _Interval:
    axes = tuple(params["axes"])  # ty: ignore[invalid-argument-type]
    tangent_axes = tuple(axis + 1 for axis in axes)
    match name:
        case "reduce_sum":
            count = max(1, math.prod(value.lower.shape[axis] for axis in axes))
            magnitude = jnp.sum(_magnitude(value.lower, value.upper), axis=axes)
            slack = count * _EPSILON * magnitude
            tangent_slack = (
                count
                * _EPSILON
                * jnp.sum(
                    _magnitude(value.tangent_lower, value.tangent_upper),
                    axis=tangent_axes,
                )
            )
            return _Interval(
                *_widened(
                    jnp.sum(value.lower, axis=axes) - slack,
                    jnp.sum(value.upper, axis=axes) + slack,
                ),
                *_widened(
                    jnp.sum(value.tangent_lower, axis=tangent_axes) - tangent_slack,
                    jnp.sum(value.tangent_upper, axis=tangent_axes) + tangent_slack,
                ),
            )
        case "reduce_max" | "reduce_min":
            larger = name == "reduce_max"
            if larger:
                lower = jnp.max(value.lower, axis=axes)
                upper = jnp.max(value.upper, axis=axes)
                candidate = value.upper >= jnp.expand_dims(lower, axes)
            else:
                lower = jnp.min(value.lower, axis=axes)
                upper = jnp.min(value.upper, axis=axes)
                candidate = value.lower <= jnp.expand_dims(upper, axes)
            mask = candidate[None]
            return _Interval(
                lower,
                upper,
                jnp.min(jnp.where(mask, value.tangent_lower, jnp.inf), axis=tangent_axes),
                jnp.max(
                    jnp.where(mask, value.tangent_upper, -jnp.inf), axis=tangent_axes
                ),
            )
        case _:
            raise ValueError(f"Interval enclosure has no reduction rule for {name!r}.")


def _structural(
    equation: jax_core.JaxprEqn, inputs: Sequence[_Value], dimensions: int
) -> list[_Value]:
    """Data-movement primitives: monotone in every operand and linear in tangents."""

    primitive = equation.primitive
    params = equation.params

    def bind(*arguments: Array) -> list[Array]:
        result = primitive.bind(*arguments, **params)
        return list(result) if primitive.multiple_results else [result]

    if not any(isinstance(value, _Interval) for value in inputs):
        if all(isinstance(value, _Exact) for value in inputs):
            return [_Exact(item) for item in bind(*(_exact(value) for value in inputs))]
        truths = [_truth(value) for value in inputs]
        possible_true = bind(*(truth.possible_true for truth in truths))
        possible_false = bind(*(truth.possible_false for truth in truths))
        return [
            _Truth(first, second)
            for first, second in zip(possible_true, possible_false, strict=True)
        ]
    operands = [
        _interval(value, dimensions)
        if isinstance(value, _Interval)
        or jnp.issubdtype(_exact(value).dtype, jnp.floating)
        else value
        for value in inputs
    ]

    def endpoint(selected: Callable[[_Interval], Array]) -> list[Array]:
        return bind(
            *(
                selected(value) if isinstance(value, _Interval) else _exact(value)
                for value in operands
            )
        )

    def tangent(selected: Callable[[_Interval], Array]) -> list[Array]:
        axes = tuple(0 if isinstance(value, _Interval) else None for value in operands)
        return jax.vmap(bind, in_axes=axes)(
            *(
                selected(value) if isinstance(value, _Interval) else _exact(value)
                for value in operands
            )
        )

    return [
        _Interval(*items)
        for items in zip(
            endpoint(lambda value: value.lower),
            endpoint(lambda value: value.upper),
            tangent(lambda value: value.tangent_lower),
            tangent(lambda value: value.tangent_upper),
            strict=True,
        )
    ]


def _exact(value: _Value) -> Array:
    if not isinstance(value, _Exact):
        raise ValueError("Interval enclosure expected exact integer or boolean data.")
    return value.value


def _closed(jaxpr: object) -> tuple[jax_core.Jaxpr, Sequence[object]]:
    if isinstance(jaxpr, jax_core.ClosedJaxpr):
        return jaxpr.jaxpr, jaxpr.consts
    if isinstance(jaxpr, jax_core.Jaxpr):
        return jaxpr, ()
    raise TypeError("Interval enclosure expected a nested jaxpr.")


def _call_jaxpr(equation: jax_core.JaxprEqn) -> object:
    params = equation.params
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in params:
            return params[key]
    raise ValueError(f"{equation.primitive.name} carries no nested jaxpr.")


def _lift(value: object, dimensions: int) -> _Value:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.floating):
        return _constant(array, dimensions)
    return _Exact(array)


def _cond(
    equation: jax_core.JaxprEqn, inputs: Sequence[_Value], dimensions: int
) -> list[_Value]:
    branches = equation.params["branches"]
    index, *operands = inputs
    if isinstance(index, _Exact):
        # A vmapped index is a tracer; select every branch and combine by index.
        results = [
            _evaluate(*_closed(branch), operands, dimensions) for branch in branches
        ]
        selected: list[_Value] = list(results[0])
        for branch_index, candidate in enumerate(results[1:], start=1):
            chosen = index.value == branch_index
            truth = _Truth(chosen, ~chosen)
            selected = [
                _select(truth, _interval(old, dimensions), _interval(new, dimensions))
                for old, new in zip(selected, candidate, strict=True)
            ]
        return selected
    if isinstance(index, _Truth) and len(branches) == 2:
        false_branch = _evaluate(*_closed(branches[0]), operands, dimensions)
        true_branch = _evaluate(*_closed(branches[1]), operands, dimensions)
        return [
            _select(index, _interval(first, dimensions), _interval(second, dimensions))
            for first, second in zip(false_branch, true_branch, strict=True)
        ]
    raise ValueError("Interval enclosure requires a boolean or exact cond index.")


def _convert(value: _Value, params: dict[str, object], dimensions: int) -> _Value:
    dtype = params["new_dtype"]
    if not isinstance(dtype, np.dtype):
        raise TypeError("convert_element_type requires a NumPy dtype parameter.")
    floating = jnp.issubdtype(dtype, jnp.floating)
    match value:
        case _Interval():
            if not floating:
                raise ValueError(
                    "Interval enclosure cannot convert an uncertain float to integers."
                )
            if dtype == jnp.float64:
                return value
            return _Interval(
                *_widened(
                    value.lower.astype(dtype).astype(jnp.float64),
                    value.upper.astype(dtype).astype(jnp.float64),
                    1.0,
                ),
                value.tangent_lower,
                value.tangent_upper,
            )
        case _Truth():
            return _interval(value, dimensions) if floating else value
        case _Exact():
            converted = value.value.astype(dtype)
            return _constant(converted, dimensions) if floating else _Exact(converted)


def _select_n(inputs: Sequence[_Value], dimensions: int) -> _Value:
    predicate, *cases = inputs
    if all(isinstance(case, _Exact) for case in (predicate, *cases)):
        return _Exact(
            jax.lax.select_n(_exact(predicate), *(_exact(case) for case in cases))
        )
    intervals = [_interval(case, dimensions) for case in cases]
    if isinstance(predicate, _Truth) or (
        isinstance(predicate, _Exact) and predicate.value.dtype == jnp.bool_
    ):
        if len(intervals) != 2:
            raise ValueError("A boolean select requires two cases.")
        return _select(_truth(predicate), intervals[0], intervals[1])
    if not isinstance(predicate, _Exact):
        raise ValueError("Interval enclosure cannot select by an uncertain index.")
    selected = intervals[0]
    for index, candidate in enumerate(intervals[1:], start=1):
        chosen = predicate.value == index
        selected = _select(_Truth(chosen, ~chosen), selected, candidate)
    return selected


def _clamp(inputs: Sequence[_Value], dimensions: int) -> _Interval:
    minimum, operand, maximum = _aligned(
        *(_interval(value, dimensions) for value in inputs)
    )
    interior = (operand.lower > minimum.upper) & (operand.upper < maximum.lower)
    below = operand.upper < minimum.lower
    above = operand.lower > maximum.upper
    hull = _hull(_hull(minimum, operand), maximum)
    tangents = _where(
        interior,
        operand,
        _where(below, minimum, _where(above, maximum, hull)),
    )
    return _Interval(
        jnp.clip(operand.lower, minimum.lower, maximum.lower),
        jnp.clip(operand.upper, minimum.upper, maximum.upper),
        tangents.tangent_lower,
        tangents.tangent_upper,
    )


def _logic(name: str, inputs: Sequence[_Value]) -> _Value:
    if all(isinstance(value, _Exact) for value in inputs):
        function = {"and": jnp.logical_and, "or": jnp.logical_or, "not": jnp.logical_not}[
            name
        ]
        return _Exact(function(*(_exact(value) for value in inputs)))
    truths = [_truth(value) for value in inputs]
    match name:
        case "not":
            return _Truth(truths[0].possible_false, truths[0].possible_true)
        case "and":
            return _Truth(
                truths[0].possible_true & truths[1].possible_true,
                truths[0].possible_false | truths[1].possible_false,
            )
        case "or":
            return _Truth(
                truths[0].possible_true | truths[1].possible_true,
                truths[0].possible_false & truths[1].possible_false,
            )
        case _:
            raise ValueError(f"Interval enclosure has no logic rule for {name!r}.")


def _evaluate_equation(
    equation: jax_core.JaxprEqn, inputs: list[_Value], dimensions: int
) -> list[_Value]:
    name = equation.primitive.name
    params = dict(equation.params)
    if name in _CALLS or name in ("checkpoint", "remat"):
        return _evaluate(*_closed(_call_jaxpr(equation)), inputs, dimensions)
    if all(isinstance(value, _Exact) for value in inputs) and name != "cond":
        result = equation.primitive.bind(*(_exact(value) for value in inputs), **params)
        results = result if equation.primitive.multiple_results else [result]
        return [_lift(item, dimensions) for item in results]
    if name in _STRUCTURAL:
        return _structural(equation, inputs, dimensions)
    if name in _UNARY:
        return [_unary(name, _interval(inputs[0], dimensions), params)]
    if name in _BINARY:
        first, second = (_interval(value, dimensions) for value in inputs)
        same = (
            isinstance(equation.invars[0], jax_core.Var)
            and equation.invars[0] is equation.invars[1]
        )
        return [_binary(name, first, second, same)]
    if name in _COMPARISON:
        first, second = (_interval(value, dimensions) for value in inputs)
        return [_compare(name, first, second)]
    if name in _LOGIC:
        return [_logic(name, inputs)]
    match name:
        case "convert_element_type":
            return [_convert(inputs[0], params, dimensions)]
        case "select_n":
            return [_select_n(inputs, dimensions)]
        case "clamp":
            return [_clamp(inputs, dimensions)]
        case "dot_general":
            first, second = (_interval(value, dimensions) for value in inputs)
            return [_dot_general(first, second, params)]
        case "reduce_sum" | "reduce_max" | "reduce_min":
            return [_reduce(name, _interval(inputs[0], dimensions), params)]
        case "reduce_and" | "reduce_or":
            truth = _truth(inputs[0])
            axes = tuple(params["axes"])
            if name == "reduce_and":
                return [
                    _Truth(
                        jnp.all(truth.possible_true, axis=axes),
                        jnp.any(truth.possible_false, axis=axes),
                    )
                ]
            return [
                _Truth(
                    jnp.any(truth.possible_true, axis=axes),
                    jnp.all(truth.possible_false, axis=axes),
                )
            ]
        case "cumsum":
            value = _interval(inputs[0], dimensions)
            axis = params["axis"]
            reverse = params["reverse"]
            if not isinstance(axis, int) or not isinstance(reverse, bool):
                raise TypeError("cumsum requires integer axis and boolean reverse.")
            count = value.lower.shape[axis]
            slack = (
                count
                * _EPSILON
                * jax.lax.cumsum(
                    _magnitude(value.lower, value.upper), axis=axis, reverse=reverse
                )
            )
            tangent_slack = (
                count
                * _EPSILON
                * jax.lax.cumsum(
                    _magnitude(value.tangent_lower, value.tangent_upper),
                    axis=axis + 1,
                    reverse=reverse,
                )
            )
            return [
                _Interval(
                    *_widened(
                        jax.lax.cumsum(value.lower, axis=axis, reverse=reverse) - slack,
                        jax.lax.cumsum(value.upper, axis=axis, reverse=reverse) + slack,
                    ),
                    *_widened(
                        jax.lax.cumsum(
                            value.tangent_lower, axis=axis + 1, reverse=reverse
                        )
                        - tangent_slack,
                        jax.lax.cumsum(
                            value.tangent_upper, axis=axis + 1, reverse=reverse
                        )
                        + tangent_slack,
                    ),
                )
            ]
        case "reduce_precision":
            value = _interval(inputs[0], dimensions)
            rounded = equation.primitive.bind(value.lower, **params)
            rounded_upper = equation.primitive.bind(value.upper, **params)
            scale = 2.0 ** (52 - int(params["mantissa_bits"]))
            return [
                _Interval(
                    *_widened(rounded, rounded_upper, scale),
                    value.tangent_lower,
                    value.tangent_upper,
                )
            ]
        case "stop_gradient":
            value = _interval(inputs[0], dimensions)
            zeros = jnp.zeros_like(value.tangent_lower)
            return [_Interval(value.lower, value.upper, zeros, zeros)]
        case "cond":
            return _cond(equation, inputs, dimensions)
        case _:
            raise ValueError(f"Interval enclosure has no rule for primitive {name!r}.")


def _evaluate(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[object],
    arguments: Sequence[_Value],
    dimensions: int,
) -> list[_Value]:
    environment: dict[jax_core.Var, _Value] = {}

    def read(atom: object) -> _Value:
        if isinstance(atom, jax_core.Literal):
            return _lift(atom.val, dimensions)
        if not isinstance(atom, jax_core.Var):
            raise TypeError("Interval enclosure met an unknown jaxpr atom.")
        return environment[atom]

    for variable, value in zip(jaxpr.constvars, consts, strict=True):
        environment[variable] = _lift(value, dimensions)
    for variable, value in zip(jaxpr.invars, arguments, strict=True):
        environment[variable] = value
    for equation in jaxpr.eqns:
        outputs = _evaluate_equation(
            equation, [read(atom) for atom in equation.invars], dimensions
        )
        for variable, value in zip(equation.outvars, outputs, strict=True):
            if isinstance(variable, jax_core.DropVar):
                continue
            environment[variable] = value
    return [read(atom) for atom in jaxpr.outvars]


def _unsupported_primitives(jaxpr: jax_core.Jaxpr) -> set[str]:
    missing: set[str] = set()
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if name not in _SUPPORTED:
            missing.add(name)
        for nested in jax_core.jaxprs_in_params(equation.params):
            missing |= _unsupported_primitives(nested)
    return missing


@dataclass(frozen=True, slots=True)
class _BoxBounds:
    """Host value, gradient, and directional-derivative bounds of boxes.

    ``directional_lower/upper[:, k]`` bound the derivative along
    ``DISCOVERY_DIRECTIONS[k]``; the first three columns are the gradient.
    """

    value_lower: np.ndarray
    value_upper: np.ndarray
    directional_lower: np.ndarray
    directional_upper: np.ndarray

    @property
    def gradient_lower(self) -> np.ndarray:
        return self.directional_lower[:, :_AXES]

    @property
    def gradient_upper(self) -> np.ndarray:
        return self.directional_upper[:, :_AXES]


def _point_field(kernel: GeometryKernel, state: DesignState) -> Callable[[Array], Array]:
    def field(point: Array) -> Array:
        return kernel.boundary_field(state, point[None, :])[0]

    return field


def _sampled_boxes(
    kernel: GeometryKernel,
    state: DesignState,
    lower: np.ndarray,
    upper: np.ndarray,
) -> _BoxBounds:
    """Corner and center value/gradient sample ranges of every box."""

    dimension = lower.shape[1]
    charge_native_geometry_queries(lower.shape[0] * ((1 << dimension) + 1))
    corners = np.asarray(
        [
            [(mask >> axis) & 1 for axis in range(dimension)]
            for mask in range(1 << dimension)
        ],
        dtype=np.float64,
    )
    samples = np.concatenate(
        (
            lower[:, None, :] + corners[None] * (upper - lower)[:, None, :],
            (0.5 * (lower + upper))[:, None, :],
        ),
        axis=1,
    )
    directions = DISCOVERY_DIRECTIONS.shape[0]
    if not samples.shape[0]:
        empty = np.zeros((0,), dtype=np.float64)
        directional = np.zeros((0, directions), dtype=np.float64)
        return _BoxBounds(empty, empty, directional, directional)
    values, gradients = _field_and_gradient(
        kernel, state, jnp.asarray(samples.reshape((-1, dimension)))
    )
    values_ = np.asarray(values, dtype=np.float64).reshape(samples.shape[:2])
    gradients_ = np.asarray(gradients, dtype=np.float64).reshape(samples.shape)
    directional = gradients_ @ DISCOVERY_DIRECTIONS.T
    return _BoxBounds(
        np.min(values_, axis=1),
        np.max(values_, axis=1),
        np.min(directional, axis=1),
        np.max(directional, axis=1),
    )


class _AbstractFieldBounds(StrictModule):
    """Box bounds of one compiled field state; host batches in, host bounds out."""

    kernel: eqx.AbstractVar[GeometryKernel]
    state: eqx.AbstractVar[DesignState]
    dimension: eqx.AbstractVar[int]

    @property
    @abstractmethod
    def enclosure(self) -> ImplicitDiscoveryEnclosure: ...

    @property
    @abstractmethod
    def value_rigorous(self) -> bool:
        """Whether value bounds enclose the field on every box."""

    @property
    @abstractmethod
    def gradient_rigorous(self) -> bool:
        """Whether gradient bounds enclose every generalized gradient on every box."""

    @property
    def continuity_rigorous(self) -> bool:
        """Whether the actual source expression establishes spatial continuity."""
        return False

    @abstractmethod
    def boxes(self, lower: np.ndarray, upper: np.ndarray, /) -> _BoxBounds:
        """Value bounds and the gradient bounds that drive refinement."""

    @abstractmethod
    def point_values(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds of the field at points (equal when sampled)."""

    def field_values(self, points: np.ndarray, /) -> np.ndarray:
        """Ordinary floating-point field values at points."""
        if not points.shape[0]:
            return np.zeros((0,), dtype=np.float64)
        charge_native_geometry_queries(points.shape[0])
        return np.asarray(
            self.kernel.boundary_field(self.state, jnp.asarray(points)),
            dtype=np.float64,
        )


@runtime_checkable
class _ShapedIntervalValue(Protocol):
    """Actual compiler abstract value required by the interval executor."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> DTypeLike: ...


@dataclass(frozen=True, slots=True, eq=False)
class _IntervalProgram:
    """Derived execution only; no source module retains this live program."""

    value_dtype: str
    evaluate: Callable[[Array, Array], tuple[Array, Array, Array, Array]]
    continuity_rigorous: bool


type _IntervalProgramKey = tuple[
    jax.tree_util.PyTreeDef,
    str,
    tuple[int, ...],
    FieldCertificate | None,
    int,
    tuple[int, ...],
    str,
    str,
]
_INTERVAL_PROGRAMS: OrderedDict[_IntervalProgramKey, _IntervalProgram] = OrderedDict()


def _point_continuity(
    jaxpr: jax_core.Jaxpr, arguments: Sequence[tuple[bool, bool]], /
) -> list[tuple[bool, bool]]:
    """Track point dependence and continuity through the actual expression."""
    environment: dict[jax_core.Var, tuple[bool, bool]] = {
        variable: (False, True) for variable in jaxpr.constvars
    }

    def read(atom: object) -> tuple[bool, bool]:
        if isinstance(atom, jax_core.Literal):
            return False, True
        if not isinstance(atom, jax_core.Var):
            raise TypeError("Continuity analysis met an unknown source atom.")
        return environment[atom]

    for variable, argument in zip(jaxpr.invars, arguments, strict=True):
        environment[variable] = argument
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        inputs = [read(atom) for atom in equation.invars]
        dependent = any(value[0] for value in inputs)
        continuous = all(value[1] for value in inputs)
        if name in _CALLS:
            nested, _ = _closed(_call_jaxpr(equation))
            outputs = _point_continuity(nested, inputs)
        else:
            if dependent and name in (
                _COMPARISON
                | _LOGIC
                | {"ceil", "floor", "round", "sign", "cond", "select_n"}
            ):
                continuous = False
            if (
                dependent
                and name == "convert_element_type"
                and not jnp.issubdtype(equation.params["new_dtype"], jnp.floating)
            ):
                continuous = False
            for nested in jax_core.jaxprs_in_params(equation.params):
                continuous &= all(
                    value[1]
                    for value in _point_continuity(
                        nested, [(dependent, continuous)] * len(nested.invars)
                    )
                )
            outputs = [(dependent, continuous)] * len(equation.outvars)
        for variable, output in zip(equation.outvars, outputs, strict=True):
            if not isinstance(variable, jax_core.DropVar):
                environment[variable] = output
    return [read(atom) for atom in jaxpr.outvars]


def _derived_interval_program(
    source: tuple[GeometryKernel, DesignState] | BoundaryAtlas,
    operator: Callable[[Array], Array],
    operator_tokens: tuple[int, ...],
    input_dimension: int,
    output_shape: tuple[int, ...],
    directions: np.ndarray,
    /,
    *,
    point_dtype: str,
    certificate: FieldCertificate | None = None,
    value_dtype: str | None = None,
) -> _IntervalProgram:
    """Prepare a bounded runtime program from complete actual source semantics."""
    if point_dtype != np.dtype(np.float64).name:
        raise ValueError(
            "Interval source queries require the declared binary64 point precision."
        )
    point = jnp.zeros((input_dimension,), dtype=jnp.float64)
    if np.dtype(point.dtype).name != point_dtype:
        raise ValueError(
            "The runtime cannot realize the declared interval point precision."
        )
    seeds = np.asarray(directions, dtype=np.float64)
    if (
        seeds.ndim != 2
        or seeds.shape[1] != input_dimension
        or not np.all(np.isfinite(seeds))
    ):
        raise ValueError(
            "Interval derivative directions must bind the actual input dimension."
        )
    key: _IntervalProgramKey = (
        jax.tree_util.tree_structure(source),
        canonical_fingerprint(array_tree_fingerprint((source, seeds))),
        operator_tokens,
        certificate,
        input_dimension,
        output_shape,
        point_dtype,
        ENCLOSURE_ROUNDING_MODEL,
    )
    cached = _INTERVAL_PROGRAMS.get(key)
    if cached is not None:
        if value_dtype is not None and cached.value_dtype != value_dtype:
            raise ValueError("The source's declared interval value precision is stale.")
        _INTERVAL_PROGRAMS.move_to_end(key)
        return cached
    closed = jax.make_jaxpr(operator)(point)
    if len(closed.jaxpr.invars) != 1 or len(closed.jaxpr.outvars) != 1:
        raise ValueError("An interval source program requires one input and one output.")
    input_aval = closed.jaxpr.invars[0].aval
    output_aval = closed.jaxpr.outvars[0].aval
    if (
        not isinstance(input_aval, _ShapedIntervalValue)
        or input_aval.shape != (input_dimension,)
        or np.dtype(input_aval.dtype).name != point_dtype
        or not isinstance(output_aval, _ShapedIntervalValue)
        or output_aval.shape != output_shape
        or not jnp.issubdtype(output_aval.dtype, jnp.floating)
    ):
        raise ValueError(
            "The actual interval operator has an incompatible shape or scalar kind."
        )
    actual_dtype = np.dtype(output_aval.dtype).name
    if value_dtype is not None and actual_dtype != value_dtype:
        raise ValueError("The source's declared interval value precision is stale.")
    missing = _unsupported_primitives(closed.jaxpr)
    if missing:
        raise ValueError(
            f"Interval source enclosure has no rule for primitives {sorted(missing)}."
        )
    consts = tuple(jnp.asarray(value) for value in closed.consts)
    tangents = jnp.asarray(seeds, dtype=jnp.float64)

    def enclose(lower: Array, upper: Array) -> tuple[Array, Array, Array, Array]:
        argument = _Interval(lower, upper, tangents, tangents)
        (output,) = _evaluate(closed.jaxpr, consts, [argument], seeds.shape[0])
        result = _interval(output, seeds.shape[0])
        return result.lower, result.upper, result.tangent_lower, result.tangent_upper

    continuous = all(
        value[1] for value in _point_continuity(closed.jaxpr, [(True, True)])
    )
    program = _IntervalProgram(actual_dtype, jax.jit(jax.vmap(enclose)), continuous)
    _INTERVAL_PROGRAMS[key] = program
    if len(_INTERVAL_PROGRAMS) > 8:
        _INTERVAL_PROGRAMS.popitem(last=False)
    return program


@final
class _IntervalFieldBounds(_AbstractFieldBounds, NonTrainableState):
    """Outward-rounded interval evaluation of the field program with tangents."""

    kernel: GeometryKernel
    state: DesignState
    dimension: int = eqx.field(static=True)
    point_dtype: str = eqx.field(static=True)
    value_dtype: str = eqx.field(static=True)
    certificate: FieldCertificate = eqx.field(static=True)
    rounding_model: str = eqx.field(static=True)

    def __init__(self, geometry: CompiledGeometry, /) -> None:
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("geometry must be CompiledGeometry.")
        dimension = geometry.ambient_dimension
        if dimension != DISCOVERY_DIRECTIONS.shape[1]:
            raise ValueError("Interval field enclosures are three-dimensional.")
        DesignState(geometry.state.schema, geometry.state.values)
        if not bool(np.asarray(geometry.validity().accepted)):
            raise ValueError(
                "Interval source preparation requires a valid geometry state."
            )
        point_dtype = np.dtype(np.float64).name
        certificate = geometry.kernel.field_certificate
        program = _derived_interval_program(
            (geometry.kernel, geometry.state),
            _point_field(geometry.kernel, geometry.state),
            (id(type(geometry.kernel).boundary_field),),
            dimension,
            (),
            DISCOVERY_DIRECTIONS,
            point_dtype=point_dtype,
            certificate=certificate,
        )
        self.kernel = geometry.kernel
        self.state = geometry.state
        self.dimension = dimension
        self.point_dtype = point_dtype
        self.value_dtype = program.value_dtype
        self.certificate = certificate
        self.rounding_model = ENCLOSURE_ROUNDING_MODEL

    @property
    def enclosure(self) -> ImplicitDiscoveryEnclosure:
        return "interval"

    @property
    def value_rigorous(self) -> bool:
        return True

    @property
    def gradient_rigorous(self) -> bool:
        return self.continuity_rigorous

    @property
    def continuity_rigorous(self) -> bool:
        return self._validated_program().continuity_rigorous

    def _validated_program(self) -> _IntervalProgram:
        """Reject inconsistent restored source facts before any numerical output."""
        if not isinstance(self.kernel, GeometryKernel) or not isinstance(
            self.state, DesignState
        ):
            raise TypeError(
                "An interval source requires its actual kernel and DesignState."
            )
        DesignState(self.state.schema, self.state.values)
        geometry = CompiledGeometry(self.kernel, self.state)
        if (
            self.dimension != geometry.ambient_dimension
            or self.dimension != DISCOVERY_DIRECTIONS.shape[1]
            or self.point_dtype != np.dtype(np.float64).name
            or self.certificate != self.kernel.field_certificate
            or self.rounding_model != ENCLOSURE_ROUNDING_MODEL
            or not bool(np.asarray(geometry.validity().accepted))
        ):
            raise ValueError(
                "The restored interval source's state, certificate or precision is invalid."
            )
        return _derived_interval_program(
            (self.kernel, self.state),
            _point_field(self.kernel, self.state),
            (id(type(self.kernel).boundary_field),),
            self.dimension,
            (),
            DISCOVERY_DIRECTIONS,
            point_dtype=self.point_dtype,
            certificate=self.certificate,
            value_dtype=self.value_dtype,
        )

    def validate_source_integrity(self) -> None:
        self._validated_program()

    def field_values(self, points: np.ndarray, /) -> np.ndarray:
        self.validate_source_integrity()
        return super().field_values(points)

    def _evaluate_boxes(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        program = self._validated_program()
        if (
            lower.ndim != 2
            or lower.shape[1] != self.dimension
            or upper.shape != lower.shape
        ):
            raise ValueError(
                "Interval source boxes must bind the actual input dimension."
            )
        dimension = DISCOVERY_DIRECTIONS.shape[0]
        count = lower.shape[0]
        charge_native_geometry_queries(count)
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for start in range(0, count, _BOX_CHUNK):
            stop = min(start + _BOX_CHUNK, count)
            size = stop - start
            # Two padded bucket extents bound the compiled program count.
            bucket = _SMALL_BOX_CHUNK if size <= _SMALL_BOX_CHUNK else _BOX_CHUNK
            padding = ((0, bucket - size), (0, 0))
            result = program.evaluate(
                jnp.asarray(np.pad(lower[start:stop], padding, mode="edge")),
                jnp.asarray(np.pad(upper[start:stop], padding, mode="edge")),
            )
            value_lower, value_upper, tangent_lower, tangent_upper = (
                np.asarray(item, dtype=np.float64)[:size] for item in result
            )
            parts.append(
                (
                    value_lower.reshape((size,)),
                    value_upper.reshape((size,)),
                    tangent_lower.reshape((size, dimension)),
                    tangent_upper.reshape((size, dimension)),
                )
            )
        if not parts:
            empty = np.zeros((0,), dtype=np.float64)
            gradient = np.zeros((0, dimension), dtype=np.float64)
            return empty, empty, gradient, gradient
        return (
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]),
            np.concatenate([part[3] for part in parts]),
        )

    def boxes(self, lower: np.ndarray, upper: np.ndarray, /) -> _BoxBounds:
        count = lower.shape[0]
        center = 0.5 * (lower + upper)
        value_lower, value_upper, gradient_lower, gradient_upper = self._evaluate_boxes(
            np.concatenate((lower, center), axis=0),
            np.concatenate((upper, center), axis=0),
        )
        # Mean-value form: f(B) lies inside f(c) + G(B) (B - c).
        half = np.maximum(upper - center, center - lower)
        magnitude = np.maximum(
            np.abs(gradient_lower[:count, :_AXES]), np.abs(gradient_upper[:count, :_AXES])
        )
        with np.errstate(invalid="ignore"):
            spread = np.sum(magnitude * half, axis=1) * (1.0 + 4.0 * _EPSILON)
            mean_lower = np.nextafter(value_lower[count:] - spread, -np.inf)
            mean_upper = np.nextafter(value_upper[count:] + spread, np.inf)
        mean_lower = np.where(np.isnan(mean_lower), -np.inf, mean_lower)
        mean_upper = np.where(np.isnan(mean_upper), np.inf, mean_upper)
        return _BoxBounds(
            np.maximum(value_lower[:count], mean_lower),
            np.minimum(value_upper[:count], mean_upper),
            gradient_lower[:count],
            gradient_upper[:count],
        )

    def point_values(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        value_lower, value_upper, _, _ = self._evaluate_boxes(points, points)
        return value_lower, value_upper


@final
class _LipschitzFieldBounds(_AbstractFieldBounds, NonTrainableState):
    """Certificate Lipschitz value enclosures; sampled gradients steer refinement."""

    kernel: GeometryKernel
    state: DesignState
    dimension: int = eqx.field(static=True)
    lipschitz: float = eqx.field(static=True)
    evaluation_error: float = eqx.field(static=True)

    def __init__(self, geometry: CompiledGeometry, /) -> None:
        certificate = geometry.field_certificate
        if (
            not certificate.bounds_established
            or certificate.validity_region != "all_space"
        ):
            raise ValueError(
                "The lipschitz enclosure requires owner-established global Lipschitz "
                "and evaluation-error bounds; select the 'interval' or 'sampled' "
                "enclosure."
            )
        self.kernel = geometry.kernel
        self.state = geometry.state
        self.dimension = geometry.ambient_dimension
        self.lipschitz = float(certificate.lipschitz_upper_bound or 0.0)
        self.evaluation_error = float(certificate.evaluation_error or 0.0)

    @property
    def enclosure(self) -> ImplicitDiscoveryEnclosure:
        return "lipschitz"

    @property
    def value_rigorous(self) -> bool:
        return True

    @property
    def gradient_rigorous(self) -> bool:
        return False

    def boxes(self, lower: np.ndarray, upper: np.ndarray, /) -> _BoxBounds:
        center = 0.5 * (lower + upper)
        radius = 0.5 * np.linalg.norm(upper - lower, axis=1)
        values = self.field_values(center)
        # The center is rounded by at most one ulp per axis; the relative slack
        # covers that displacement and the rounding of the enclosure itself.
        spread = (self.lipschitz * radius + self.evaluation_error) * (
            1.0 + 64.0 * _EPSILON
        )
        sampled = _sampled_boxes(self.kernel, self.state, lower, upper)
        return _BoxBounds(
            np.nextafter(values - spread, -np.inf),
            np.nextafter(values + spread, np.inf),
            sampled.directional_lower,
            sampled.directional_upper,
        )

    def point_values(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        values = self.field_values(points)
        return values - self.evaluation_error, values + self.evaluation_error


@final
class _SampledFieldBounds(_AbstractFieldBounds, NonTrainableState):
    """Corner and center sample ranges; heuristic only, never an enclosure."""

    kernel: GeometryKernel
    state: DesignState
    dimension: int = eqx.field(static=True)

    def __init__(self, geometry: CompiledGeometry, /) -> None:
        self.kernel = geometry.kernel
        self.state = geometry.state
        self.dimension = geometry.ambient_dimension

    @property
    def enclosure(self) -> ImplicitDiscoveryEnclosure:
        return "sampled"

    @property
    def value_rigorous(self) -> bool:
        return False

    @property
    def gradient_rigorous(self) -> bool:
        return False

    def boxes(self, lower: np.ndarray, upper: np.ndarray, /) -> _BoxBounds:
        return _sampled_boxes(self.kernel, self.state, lower, upper)

    def point_values(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        values = self.field_values(points)
        return values, values


def _field_bounds(
    geometry: CompiledGeometry, enclosure: ImplicitDiscoveryEnclosure, /
) -> _AbstractFieldBounds:
    match enclosure:
        case "interval":
            return _IntervalFieldBounds(geometry)
        case "lipschitz":
            return _LipschitzFieldBounds(geometry)
        case "sampled":
            return _SampledFieldBounds(geometry)
        case _:
            raise ValueError(f"Unknown implicit discovery enclosure {enclosure!r}.")


__all__ = ["DISCOVERY_DIRECTIONS", "ENCLOSURE_ROUNDING_MODEL"]
