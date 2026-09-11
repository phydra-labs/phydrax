#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private, order-leading spherical-Bessel sequence kernels."""

from __future__ import annotations

import math
from functools import partial
from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from ._dtype import promote_real


_SequenceKind = Literal["j", "y", "h1", "i", "k"]
_SERIES_BOUNDARY = 0.75
_MILLER_EXTRA_ORDERS = 96
_SERIES_TERMS = 24


def _validated_maximum_order(maximum_order: int, /) -> int:
    if isinstance(maximum_order, bool) or not isinstance(maximum_order, Integral):
        raise TypeError("Maximum spherical-Bessel order must be a static integer.")
    maximum = int(maximum_order)
    if maximum < 0:
        raise ValueError("Maximum spherical-Bessel order must be nonnegative.")
    return maximum


def _prepare_argument(argument: ArrayLike, /) -> Array:
    value = jnp.asarray(argument)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        dtype = jnp.complex128 if value.dtype == jnp.complex128 else jnp.complex64
        return jnp.asarray(value, dtype=dtype)
    return promote_real("spherical Bessel sequence", value)[0]


def _complex_argument(argument: ArrayLike, /) -> Array:
    value = _prepare_argument(argument)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        return value
    dtype = jnp.complex128 if value.dtype == jnp.float64 else jnp.complex64
    return value.astype(dtype)


def _regular_series(
    maximum_order: int,
    argument: Array,
    /,
    *,
    modified: bool,
) -> tuple[Array, Array]:
    """Return the regular j/i sequence and its argument derivative by power series."""
    sign = 1.0 if modified else -1.0
    leading = jnp.ones_like(argument)
    leading_derivative = jnp.zeros_like(argument)
    values = []
    derivatives = []

    for order in range(maximum_order + 1):
        term = leading
        term_derivative = leading_derivative
        total = term
        total_derivative = term_derivative
        for index in range(1, _SERIES_TERMS):
            denominator = 2.0 * index * (2.0 * order + 2.0 * index + 1.0)
            factor = sign * argument * argument / denominator
            factor_derivative = 2.0 * sign * argument / denominator
            previous_term = term
            term = previous_term * factor
            term_derivative = term_derivative * factor + previous_term * factor_derivative
            total = total + term
            total_derivative = total_derivative + term_derivative
        values.append(total)
        derivatives.append(total_derivative)

        denominator = 2.0 * order + 3.0
        previous_leading = leading
        leading = previous_leading * argument / denominator
        leading_derivative = (
            leading_derivative * argument + previous_leading
        ) / denominator

    return jnp.stack(values, axis=0), jnp.stack(derivatives, axis=0)


def _ordinary_j_bases(argument: Array, /) -> tuple[Array, Array]:
    base = jnp.sin(argument) / argument
    following = (base - jnp.cos(argument)) / argument
    return base, following


def _modified_i_bases(argument: Array, /, *, scaled: bool) -> tuple[Array, Array]:
    exponential = jnp.exp(-2.0 * argument)
    base = -jnp.expm1(-2.0 * argument) / (2.0 * argument)
    following = (argument - 1.0 + (argument + 1.0) * exponential) / (
        2.0 * argument * argument
    )
    if scaled:
        return base, following
    scale = jnp.exp(argument)
    return scale * base, scale * following


def _upward_regular_sequence(
    maximum_order: int,
    argument: Array,
    base: Array,
    following: Array,
    /,
    *,
    modified: bool,
) -> Array:
    values = [base]
    if maximum_order == 0:
        return jnp.stack(values, axis=0)
    values.append(following)
    previous, current = base, following
    for order in range(1, maximum_order):
        coefficient = (2.0 * order + 1.0) / argument
        next_value = (
            previous - coefficient * current
            if modified
            else coefficient * current - previous
        )
        values.append(next_value)
        previous, current = current, next_value
    return jnp.stack(values, axis=0)


def _upward_k_sequence(
    maximum_order: int,
    argument: Array,
    base: Array,
    following: Array,
    /,
) -> Array:
    values = [base]
    if maximum_order == 0:
        return jnp.stack(values, axis=0)
    values.append(following)
    previous, current = base, following
    for order in range(1, maximum_order):
        next_value = previous + (2.0 * order + 1.0) * current / argument
        values.append(next_value)
        previous, current = current, next_value
    return jnp.stack(values, axis=0)


def _miller_regular_sequence(
    maximum_order: int,
    argument: Array,
    base: Array,
    following: Array,
    /,
    *,
    modified: bool,
) -> Array:
    """Miller recurrence normalized by the analytic orders zero and one."""
    start = maximum_order + _MILLER_EXTRA_ORDERS
    values = jnp.zeros((maximum_order + 1,) + argument.shape, dtype=argument.dtype)
    logarithmic_scales = jnp.zeros(
        (maximum_order + 1,) + argument.shape, dtype=argument.real.dtype
    )
    cumulative_scale = jnp.zeros_like(argument.real)
    recurrence_limit = jnp.sqrt(jnp.asarray(jnp.finfo(argument.real.dtype).max))

    def body(index: int, state: tuple[Array, Array, Array, Array, Array]):
        later, current, captured, captured_scales, cumulative = state
        order = start - index
        coefficient = (2.0 * order + 1.0) / argument
        previous = (
            coefficient * current + later if modified else coefficient * current - later
        )
        magnitude = jnp.maximum(jnp.abs(previous), jnp.abs(current))
        scale = jnp.where(
            magnitude > recurrence_limit, magnitude, jnp.ones_like(magnitude)
        )
        previous = previous / scale
        current = current / scale
        cumulative = cumulative + jnp.log(scale)

        capture_index = order - 1
        active = capture_index <= maximum_order
        destination = jnp.minimum(capture_index, maximum_order)
        captured = captured.at[destination].set(
            jnp.where(active, previous, captured[destination])
        )
        captured_scales = captured_scales.at[destination].set(
            jnp.where(active, cumulative, captured_scales[destination])
        )
        return current, previous, captured, captured_scales, cumulative

    initial = (
        jnp.zeros_like(argument),
        jnp.ones_like(argument),
        values,
        logarithmic_scales,
        cumulative_scale,
    )
    relative_one, relative_zero, values, logarithmic_scales, cumulative_scale = (
        lax.fori_loop(0, start, body, initial)
    )
    relative_values = values * jnp.exp(
        logarithmic_scales - cumulative_scale[jnp.newaxis, ...]
    )

    denominator = (
        jnp.conj(relative_zero) * relative_zero + jnp.conj(relative_one) * relative_one
    ).real
    normalization = (
        jnp.conj(relative_zero) * base + jnp.conj(relative_one) * following
    ) / denominator
    return relative_values * normalization[jnp.newaxis, ...]


def _regular_sequence(
    maximum_order: int,
    argument: Array,
    /,
    *,
    modified: bool,
    scaled: bool,
) -> Array:
    magnitude = jnp.abs(argument)
    near_zero = magnitude < _SERIES_BOUNDARY
    safe_argument = jnp.where(near_zero, jnp.ones_like(argument), argument)
    if modified:
        base, following = _modified_i_bases(safe_argument, scaled=scaled)
    else:
        base, following = _ordinary_j_bases(safe_argument)

    upward = _upward_regular_sequence(
        maximum_order,
        safe_argument,
        base,
        following,
        modified=modified,
    )
    if maximum_order >= 2:
        use_miller = (~near_zero) & (magnitude < maximum_order + 1.0)
        miller_argument = jnp.where(use_miller, argument, jnp.ones_like(argument))
        if modified:
            miller_base, miller_following = _modified_i_bases(
                miller_argument, scaled=scaled
            )
        else:
            miller_base, miller_following = _ordinary_j_bases(miller_argument)
        miller = _miller_regular_sequence(
            maximum_order,
            miller_argument,
            miller_base,
            miller_following,
            modified=modified,
        )
        recurrent = jnp.where(use_miller[jnp.newaxis, ...], miller, upward)
    else:
        recurrent = upward

    series_argument = jnp.where(near_zero, argument, jnp.zeros_like(argument))
    series, _ = _regular_series(maximum_order, series_argument, modified=modified)
    if modified and scaled:
        series = jnp.exp(-series_argument)[jnp.newaxis, ...] * series
    return jnp.where(near_zero[jnp.newaxis, ...], series, recurrent)


def _raw_spherical_j_sequence(maximum_order: int, argument: ArrayLike, /) -> Array:
    """Return ``j_0`` through ``j_maximum_order`` with order as the leading axis."""
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    return _regular_sequence(maximum, value, modified=False, scaled=False)


def _raw_spherical_y_sequence(maximum_order: int, argument: ArrayLike, /) -> Array:
    """Return ``y_0`` through ``y_maximum_order`` with order as the leading axis."""
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    zero = jnp.abs(value) == 0.0
    safe = jnp.where(zero, jnp.ones_like(value), value)
    base = -jnp.cos(safe) / safe
    following = (base - jnp.sin(safe)) / safe
    sequence = _upward_regular_sequence(maximum, safe, base, following, modified=False)
    singular = jnp.full_like(sequence, -jnp.inf)
    if jnp.issubdtype(sequence.dtype, jnp.complexfloating):
        singular = lax.complex(
            jnp.full_like(sequence.real, jnp.nan),
            jnp.full_like(sequence.real, -jnp.inf),
        )
    return jnp.where(zero[jnp.newaxis, ...], singular, sequence)


def _raw_spherical_hankel1_sequence(
    maximum_order: int,
    argument: ArrayLike,
    /,
    *,
    scaled: bool = False,
) -> Array:
    """Return outgoing spherical Hankel values, optionally scaled by ``exp(-i z)``."""
    maximum = _validated_maximum_order(maximum_order)
    value = _complex_argument(argument)
    zero = jnp.abs(value) == 0.0
    safe = jnp.where(zero, jnp.ones_like(value), value)
    base = -1j / safe
    following = -(1.0 + 1j / safe) / safe
    sequence = _upward_regular_sequence(maximum, safe, base, following, modified=False)
    if not scaled:
        sequence = jnp.exp(1j * safe)[jnp.newaxis, ...] * sequence

    regular_zero = _raw_spherical_j_sequence(maximum, value)
    singular_zero = lax.complex(
        regular_zero.real,
        jnp.full_like(regular_zero.real, -jnp.inf),
    )
    return jnp.where(zero[jnp.newaxis, ...], singular_zero, sequence)


def _raw_spherical_i_sequence(
    maximum_order: int,
    argument: ArrayLike,
    /,
    *,
    scaled: bool = False,
) -> Array:
    """Return modified spherical ``i`` values, optionally scaled by ``exp(-z)``."""
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    return _regular_sequence(maximum, value, modified=True, scaled=scaled)


def _raw_spherical_k_sequence(
    maximum_order: int,
    argument: ArrayLike,
    /,
    *,
    scaled: bool = False,
) -> Array:
    """Return modified spherical ``k`` values, optionally scaled by ``exp(z)``."""
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    zero = jnp.abs(value) == 0.0
    safe = jnp.where(zero, jnp.ones_like(value), value)
    base = math.pi / (2.0 * safe)
    following = base * (1.0 + 1.0 / safe)
    sequence = _upward_k_sequence(maximum, safe, base, following)
    if not scaled:
        sequence = jnp.exp(-safe)[jnp.newaxis, ...] * sequence
    singular = jnp.full_like(sequence, jnp.inf)
    if jnp.issubdtype(sequence.dtype, jnp.complexfloating):
        singular = lax.complex(
            jnp.full_like(sequence.real, jnp.inf),
            jnp.full_like(sequence.real, jnp.nan),
        )
    return jnp.where(zero[jnp.newaxis, ...], singular, sequence)


def _sequence_for_kind(
    maximum_order: int,
    argument: Array,
    kind: _SequenceKind,
    /,
    *,
    scaled: bool,
) -> Array:
    if kind == "j":
        return _raw_spherical_j_sequence(maximum_order, argument)
    if kind == "y":
        return _raw_spherical_y_sequence(maximum_order, argument)
    if kind == "h1":
        return _raw_spherical_hankel1_sequence(maximum_order, argument, scaled=scaled)
    if kind == "i":
        return _raw_spherical_i_sequence(maximum_order, argument, scaled=scaled)
    return _raw_spherical_k_sequence(maximum_order, argument, scaled=scaled)


def _spherical_sequence_derivative(
    values: Array,
    argument: ArrayLike,
    /,
    *,
    kind: _SequenceKind,
    scaled: bool = False,
) -> Array:
    """Differentiate an order-leading spherical sequence using neighbor identities."""
    if kind not in ("j", "y", "h1", "i", "k"):
        raise ValueError("Unknown spherical-Bessel sequence kind.")
    if scaled and kind not in ("h1", "i", "k"):
        raise ValueError("Only outgoing or modified spherical sequences are scaled.")
    sequence = jnp.asarray(values)
    if sequence.ndim < 1 or sequence.shape[0] < 1:
        raise ValueError("Spherical-Bessel values require a nonempty leading order axis.")

    value = _prepare_argument(argument)
    dtype = jnp.result_type(sequence, value)
    sequence = jnp.asarray(sequence, dtype=dtype)
    value = jnp.broadcast_to(jnp.asarray(value, dtype=dtype), sequence.shape[1:])
    zero = jnp.abs(value) == 0.0
    safe = jnp.where(zero, jnp.ones_like(value), value)
    order = jnp.arange(sequence.shape[0], dtype=value.real.dtype).reshape(
        (sequence.shape[0],) + (1,) * value.ndim
    )
    previous = jnp.concatenate((jnp.zeros_like(sequence[:1]), sequence[:-1]), axis=0)

    if kind == "k":
        derivative = -previous - (order + 1.0) * sequence / safe
    else:
        derivative = previous - (order + 1.0) * sequence / safe

    if sequence.shape[0] >= 2:
        if kind == "i":
            base_derivative = sequence[1]
        else:
            base_derivative = -sequence[1]
    else:
        pair = _sequence_for_kind(1, value, kind, scaled=scaled)
        base_derivative = pair[1] if kind == "i" else -pair[1]
    if scaled:
        if kind == "h1":
            derivative = derivative - 1j * sequence
            base_derivative = base_derivative - 1j * sequence[0]
        elif kind == "i":
            derivative = derivative - sequence
            base_derivative = base_derivative - sequence[0]
        else:
            derivative = derivative + sequence
            base_derivative = base_derivative + sequence[0]
    derivative = derivative.at[0].set(base_derivative)

    if kind in ("j", "i"):
        series_argument = jnp.where(
            jnp.abs(value) < _SERIES_BOUNDARY, value, jnp.zeros_like(value)
        )
        series, series_derivative = _regular_series(
            sequence.shape[0] - 1,
            series_argument,
            modified=kind == "i",
        )
        if scaled:
            factor = jnp.exp(-series_argument)[jnp.newaxis, ...]
            series_derivative = factor * (series_derivative - series)
        derivative = jnp.where(
            (jnp.abs(value) < _SERIES_BOUNDARY)[jnp.newaxis, ...],
            series_derivative,
            derivative,
        )
    elif kind == "y":
        derivative = jnp.where(
            zero[jnp.newaxis, ...], jnp.full_like(derivative, jnp.inf), derivative
        )
    elif kind == "k":
        derivative = jnp.where(
            zero[jnp.newaxis, ...], jnp.full_like(derivative, -jnp.inf), derivative
        )
    else:
        singular = lax.complex(
            jnp.zeros_like(derivative.real),
            jnp.full_like(derivative.real, jnp.inf),
        )
        derivative = jnp.where(zero[jnp.newaxis, ...], singular, derivative)
    return derivative


@partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3))
def _spherical_sequence_array(
    maximum_order: int,
    argument: Array,
    kind: _SequenceKind,
    scaled: bool,
    /,
) -> Array:
    return _sequence_for_kind(maximum_order, argument, kind, scaled=scaled)


@_spherical_sequence_array.defjvp
def _spherical_sequence_array_jvp(
    maximum_order: int,
    kind: _SequenceKind,
    scaled: bool,
    primals,
    tangents,
):
    (argument,) = primals
    (argument_tangent,) = tangents
    values = _sequence_for_kind(maximum_order, argument, kind, scaled=scaled)
    derivative = _spherical_sequence_derivative(
        values,
        argument,
        kind=kind,
        scaled=scaled,
    )
    return values, derivative * argument_tangent[jnp.newaxis, ...]


def _spherical_j_sequence(maximum_order: int, argument: ArrayLike, /) -> Array:
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    return _spherical_sequence_array(maximum, value, "j", False)


def _spherical_y_sequence(maximum_order: int, argument: ArrayLike, /) -> Array:
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    return _spherical_sequence_array(maximum, value, "y", False)


def _spherical_hankel1_sequence(
    maximum_order: int,
    argument: ArrayLike,
    /,
    *,
    scaled: bool = False,
) -> Array:
    maximum = _validated_maximum_order(maximum_order)
    value = _complex_argument(argument)
    return _spherical_sequence_array(maximum, value, "h1", scaled)


def _spherical_i_sequence(
    maximum_order: int,
    argument: ArrayLike,
    /,
    *,
    scaled: bool = False,
) -> Array:
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    return _spherical_sequence_array(maximum, value, "i", scaled)


def _spherical_k_sequence(
    maximum_order: int,
    argument: ArrayLike,
    /,
    *,
    scaled: bool = False,
) -> Array:
    maximum = _validated_maximum_order(maximum_order)
    value = _prepare_argument(argument)
    return _spherical_sequence_array(maximum, value, "k", scaled)
