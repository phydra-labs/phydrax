#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# The large-order asymptotic methods are adapted from SciPy XSF, and the jv/yv
# kernels from its bundled Cephes sources. Both are licensed under BSD-3-Clause.
# See NOTICE, LICENSES/SCIPY-XSF-BSD-3-CLAUSE.txt, and
# LICENSES/CEPHES-BSD-3-CLAUSE.txt.
#

"""Real-order cylindrical Bessel and Hankel functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import Array
from jax.custom_derivatives import SymbolicZero
from jax.typing import ArrayLike

from ._airy import airy
from ._dtype import (
    _exact_zero,
    _positive_log,
    _positive_subnormal,
    _signbit,
    complex_from_parts,
    promote_real,
)
from ._gamma import log_gamma_one_plus_minus_difference
from ._modified_bessel import _DEBYE_POLYNOMIALS


def _gauss_legendre(count: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    nodes = [0.0] * count
    weights = [0.0] * count
    half = (count + 1) // 2
    for index in range(half):
        root = math.cos(math.pi * (index + 0.75) / (count + 0.5))
        derivative = 0.0
        for _ in range(12):
            previous = 1.0
            current = root
            for degree in range(2, count + 1):
                following = (
                    (2.0 * degree - 1.0) * root * current - (degree - 1.0) * previous
                ) / degree
                previous, current = current, following
            derivative = count * (root * current - previous) / (root * root - 1.0)
            root -= current / derivative
        weight = 2.0 / ((1.0 - root * root) * derivative * derivative)
        lower = 0.5 * (1.0 - root)
        upper = 0.5 * (1.0 + root)
        scaled_weight = 0.5 * weight
        nodes[index] = lower
        nodes[count - index - 1] = upper
        weights[index] = scaled_weight
        weights[count - index - 1] = scaled_weight
    return tuple(nodes), tuple(weights)


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = _gauss_legendre(192)


def _lambda_mu(count: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lambdas = [1.0]
    mus = [1.0]
    for index in range(1, count):
        value = lambdas[-1] * ((6.0 * index - 1.0) * (6.0 * index - 5.0) / (48.0 * index))
        lambdas.append(value)
        mus.append(-(6.0 * index + 1.0) * value / (6.0 * index - 1.0))
    return tuple(lambdas), tuple(mus)


_LAMBDAS, _MUS = _lambda_mu(9)


def _power_series_j(v: Array, x: Array) -> Array:
    quarter_x_squared = -0.25 * x * x

    def body(k: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        term, total = state
        index = jnp.asarray(k, dtype=x.dtype)
        term = term * quarter_x_squared / (index * (v + index))
        return term, total + term

    _, series = jax.lax.fori_loop(1, 61, body, (jnp.ones_like(x), jnp.ones_like(x)))
    log_prefactor = v * (_positive_log(x) - math.log(2.0)) - jsp.gammaln(v + 1.0)
    log_prefactor = jnp.where(
        _exact_zero(v), jnp.zeros_like(log_prefactor), log_prefactor
    )
    return jnp.exp(log_prefactor) * series


def _hankel_asymptotic(v: Array, x: Array) -> tuple[Array, Array]:
    mu = 4.0 * v * v
    denominator = 8.0 * x
    initial_q = (mu - 1.0) / denominator

    def body(
        pair: int,
        state: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        term, p, q, sign, best_error, best_p, best_q = state
        pair_value = jnp.asarray(pair, dtype=x.dtype)
        first_odd = 4.0 * pair_value + 3.0
        first_index = 2.0 * pair_value + 2.0
        next_sign = -sign
        term = term * (mu - first_odd * first_odd) / (first_index * denominator)
        next_p = p + next_sign * term
        second_odd = first_odd + 2.0
        second_index = first_index + 1.0
        term = term * (mu - second_odd * second_odd) / (second_index * denominator)
        next_q = q + next_sign * term
        error = jnp.abs(term / next_p)
        improved = error < best_error
        return (
            term,
            next_p,
            next_q,
            next_sign,
            jnp.where(improved, error, best_error),
            jnp.where(improved, next_p, best_p),
            jnp.where(improved, next_q, best_q),
        )

    _, _, _, _, _, p, q = jax.lax.fori_loop(
        0,
        20,
        body,
        (
            initial_q,
            jnp.ones_like(x),
            initial_q,
            jnp.ones_like(x),
            jnp.full_like(x, jnp.inf),
            jnp.ones_like(x),
            initial_q,
        ),
    )
    order_phase = (0.5 * v + 0.25) * math.pi
    cosine_x = jnp.cos(x)
    sine_x = jnp.sin(x)
    cosine_phase = jnp.cos(order_phase)
    sine_phase = jnp.sin(order_phase)
    cosine = cosine_x * cosine_phase + sine_x * sine_phase
    sine = sine_x * cosine_phase - cosine_x * sine_phase
    scale = jnp.sqrt(2.0 / (math.pi * x))
    return scale * (p * cosine - q * sine), scale * (p * sine + q * cosine)


def _integral_pair_values(v: Array, x: Array) -> tuple[Array, Array, Array, Array]:
    nodes = jnp.asarray(_QUADRATURE_NODES, dtype=x.dtype)
    weights = jnp.asarray(_QUADRATURE_WEIGHTS, dtype=x.dtype)
    cosine_order = jnp.cos(math.pi * v)
    sine_order = jnp.sin(math.pi * v)
    log_x = _positive_log(x)
    log_inverse_x = jnp.maximum(math.log(2.0) - log_x, 0.0)
    cutoff = 80.0 + (v + 2.0) * (log_inverse_x + 5.0)
    upper_limit = jnp.log(2.0 * cutoff) - log_x

    def body(
        index: int,
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        (
            first_cosine,
            first_sine,
            second_j,
            second_y,
            next_first_cosine,
            next_first_sine,
            next_second_j,
            next_second_y,
        ) = state
        node = nodes[index]
        weight = weights[index]
        angle = math.pi * node
        sine_argument = x * jnp.sin(angle)
        phase = sine_argument - v * angle
        next_phase = phase - angle
        first_cosine = first_cosine + weight * jnp.cos(phase)
        first_sine = first_sine + weight * jnp.sin(phase)
        next_first_cosine = next_first_cosine + weight * jnp.cos(next_phase)
        next_first_sine = next_first_sine + weight * jnp.sin(next_phase)

        integration_variable = upper_limit * node
        jacobian = upper_limit
        common = -x * jnp.sinh(integration_variable)
        decaying = jnp.exp(common - v * integration_variable)
        growing = jnp.exp(common + v * integration_variable)
        next_decaying = jnp.exp(common - (v + 1.0) * integration_variable)
        next_growing = jnp.exp(common + (v + 1.0) * integration_variable)
        weighted_jacobian = weight * jacobian
        second_j = second_j + weighted_jacobian * decaying
        second_y = second_y + weighted_jacobian * (growing + cosine_order * decaying)
        next_second_j = next_second_j + weighted_jacobian * next_decaying
        next_second_y = next_second_y + weighted_jacobian * (
            next_growing - cosine_order * next_decaying
        )
        return (
            first_cosine,
            first_sine,
            second_j,
            second_y,
            next_first_cosine,
            next_first_sine,
            next_second_j,
            next_second_y,
        )

    zero = jnp.zeros_like(x)
    (
        first_cosine,
        first_sine,
        second_j,
        second_y,
        next_first_cosine,
        next_first_sine,
        next_second_j,
        next_second_y,
    ) = jax.lax.fori_loop(0, len(_QUADRATURE_NODES), body, (zero,) * 8)
    return (
        first_cosine - sine_order * second_j / math.pi,
        first_sine - second_y / math.pi,
        next_first_cosine + sine_order * next_second_j / math.pi,
        next_first_sine - next_second_y / math.pi,
    )


def _debye_value(index: int, inverse_zz: Array, root_abs_zz: Array, zz: Array) -> Array:
    coefficients = _DEBYE_POLYNOMIALS[index - 1]
    reduced = tuple(coefficients[degree] for degree in range(index, len(coefficients), 2))
    value = jnp.zeros_like(zz)
    for coefficient in reversed(reduced):
        value = value * inverse_zz + coefficient
    if index % 2:
        denominator = root_abs_zz * jnp.power(zz, (index - 1) // 2)
    else:
        denominator = jnp.power(zz, index // 2)
    return value / denominator


def _transition(v: Array, x: Array) -> tuple[Array, Array]:
    cube_root = jnp.cbrt(v)
    coordinate = (x - v) / cube_root
    airy_argument = -jnp.cbrt(2.0) * coordinate
    ai, aip, bi, bip = airy(airy_argument)
    square = coordinate * coordinate
    cube = square * coordinate
    f = (
        jnp.ones_like(v),
        -coordinate / 5.0,
        square * (-0.09 * cube + 3.0 / 35.0),
        (957.0 / 7000.0) * cube * cube - (173.0 / 3150.0) * cube - 1.0 / 225.0,
        coordinate
        * (
            27.0 * cube * cube * cube / 20000.0
            - 47146.0 * cube * cube / 294000.0
            + 1135.0 * cube / 26650.0
            + 13.0 / 4756.0
        ),
    )
    g = (
        0.3 * square,
        -17.0 * cube / 70.0 + 1.0 / 70.0,
        coordinate * (-0.009 * cube * cube + 611.0 * cube / 3150.0 - 37.0 / 3150.0),
        square
        * (549.0 * cube * cube / 28000.0 - 15824.0 * cube / 99000.0 + 79.0 / 12375.0),
    )
    inverse_order_power = jnp.power(v, -2.0 / 3.0)
    power = jnp.ones_like(v)
    p = jnp.zeros_like(v)
    q = jnp.zeros_like(v)
    for index in range(5):
        p = p + f[index] * power
        if index < 4:
            q = q + g[index] * power
        power = power * inverse_order_power
    first_scale = jnp.cbrt(2.0) / cube_root
    second_scale = jnp.cbrt(4.0) / v
    return (
        first_scale * ai * p + second_scale * aip * q,
        -(first_scale * bi * p + second_scale * bip * q),
    )


def _uniform(v: Array, x: Array) -> tuple[Array, Array]:
    ratio = x / v
    zz = 1.0 - ratio * ratio
    root_abs_zz = jnp.sqrt(jnp.abs(zz))
    below = zz > 0.0
    safe_ratio = jnp.where(ratio > 0.0, ratio, jnp.ones_like(ratio))
    positive_t = 1.5 * (jnp.log((1.0 + root_abs_zz) / safe_ratio) - root_abs_zz)
    negative_t = 1.5 * (root_abs_zz - jnp.arccos(1.0 / safe_ratio))
    t = jnp.where(below, positive_t, negative_t)
    zeta = jnp.where(below, jnp.cbrt(t * t), -jnp.cbrt(t * t))
    inverse_t = jnp.abs(1.0 / t)
    inverse_zz = 1.0 / zz
    u = (jnp.ones_like(v),) + tuple(
        _debye_value(index, inverse_zz, root_abs_zz, zz) for index in range(1, 8)
    )
    flag = jnp.where(below, jnp.ones_like(v), -jnp.ones_like(v))
    inverse_order_squared = 1.0 / (v * v)
    order_power = jnp.ones_like(v)
    p = jnp.zeros_like(v)
    q = jnp.zeros_like(v)
    for outer in range(4):
        even_index = 2 * outer
        odd_index = even_index + 1
        zeta_power = jnp.ones_like(v)
        a = jnp.zeros_like(v)
        b = jnp.zeros_like(v)
        for inner in range(even_index + 1):
            sign_a = flag if inner % 4 > 1 else 1.0
            a = a + sign_a * _MUS[inner] * zeta_power * u[even_index - inner]
            sign_b = flag if (odd_index - inner + 1) % 4 > 1 else 1.0
            b = b + sign_b * _LAMBDAS[inner] * zeta_power * u[odd_index - inner]
            zeta_power = zeta_power * inverse_t
        b = b + _LAMBDAS[odd_index] * zeta_power
        p = p + a * order_power
        q = q - b * order_power / jnp.cbrt(t)
        order_power = order_power * inverse_order_squared

    airy_argument = jnp.power(v, 2.0 / 3.0) * zeta
    ai, aip, bi, bip = airy(airy_argument)
    prefactor = jnp.sqrt(jnp.sqrt(4.0 * zeta / zz))
    first_scale = prefactor / jnp.cbrt(v)
    second_scale = prefactor / jnp.power(v, 5.0 / 3.0)
    return (
        first_scale * ai * p + second_scale * aip * q,
        -(first_scale * bi * p + second_scale * bip * q),
    )


def _small_x_base_y(order: Array, x: Array) -> tuple[Array, Array]:
    log_half_x = _positive_log(x) - math.log(2.0)
    log_j = order * log_half_x - jsp.gammaln(order + 1.0)
    log_negative_j = -order * log_half_x - jsp.gammaln(1.0 - order)
    gamma_difference = log_gamma_one_plus_minus_difference(order)
    delta = -2.0 * order * log_half_x + gamma_difference
    cosine = jnp.cos(math.pi * order)
    sine = jnp.sin(math.pi * order)
    cosine_minus_one = -2.0 * jnp.sin(0.5 * math.pi * order) ** 2
    cancellation_safe = jnp.exp(log_j) * (cosine_minus_one - jnp.expm1(delta))
    direct = cosine * jnp.exp(log_j) - jnp.exp(log_negative_j)
    numerator = jnp.where(jnp.abs(delta) < 0.5, cancellation_safe, direct)
    zero_order = _exact_zero(order)
    safe_sine = jnp.where(zero_order, jnp.ones_like(sine), sine)
    y_value = numerator / safe_sine
    y_zero = (2.0 / math.pi) * (log_half_x + 0.577215664901532860606512090082402431)
    y_value = jnp.where(zero_order, y_zero, y_value)
    next_log_magnitude = (
        jsp.gammaln(order + 1.0) - (order + 1.0) * log_half_x - math.log(math.pi)
    )
    return y_value, -jnp.exp(next_log_magnitude)


def _base_pair_values(order: Array, x: Array) -> tuple[Array, Array, Array, Array]:
    next_order = order + 1.0
    power_region = x < 3.6 * jnp.sqrt(order + 1.0)
    next_power_region = x < 3.6 * jnp.sqrt(next_order + 1.0)
    hankel_region = x > 24.0
    series_x = jnp.where(power_region, x, jnp.ones_like(x))
    next_series_x = jnp.where(next_power_region, x, jnp.ones_like(x))
    series = _power_series_j(order, series_x)
    next_series = _power_series_j(next_order, next_series_x)
    small_x = x < 1e-20
    integral_region = (~hankel_region) & (~small_x)
    integral_x = jnp.where(integral_region, x, jnp.ones_like(x))
    integral_j, integral_y, next_integral_j, next_integral_y = _integral_pair_values(
        order, integral_x
    )
    asymptotic_x = jnp.where(hankel_region, x, jnp.full_like(x, 50.0))
    asymptotic = _hankel_asymptotic(order, asymptotic_x)
    next_asymptotic = _hankel_asymptotic(next_order, asymptotic_x)
    j_value = jnp.where(power_region, series, integral_j)
    next_j_value = jnp.where(next_power_region, next_series, next_integral_j)
    j_value = jnp.where(hankel_region, asymptotic[0], j_value)
    y_value = jnp.where(hankel_region, asymptotic[1], integral_y)
    next_j_value = jnp.where(hankel_region, next_asymptotic[0], next_j_value)
    small_y, small_next_y = _small_x_base_y(order, x)
    y_value = jnp.where(small_x, small_y, y_value)
    next_y_value = jnp.where(hankel_region, next_asymptotic[1], next_integral_y)
    next_y_value = jnp.where(small_x, small_next_y, next_y_value)
    return j_value, y_value, next_j_value, next_y_value


def _forward_pair(
    fractional_order: Array,
    integer_order: Array,
    x: Array,
    first_j: Array,
    second_j: Array,
    first_y: Array,
    second_y: Array,
) -> tuple[Array, Array]:
    initial = (first_j, second_j, first_y, second_y)

    def body(
        index: int, state: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        previous_j, current_j, previous_y, current_y = state
        index_value = jnp.asarray(index, dtype=x.dtype)
        order = fractional_order + index_value + 1.0
        active = index < integer_order - 1
        recurrence_x = jnp.where(active, x, jnp.ones_like(x))
        coefficient = 2.0 * order / recurrence_x
        following_j = coefficient * current_j - previous_j
        following_y = coefficient * current_y - previous_y
        return (
            jnp.where(active, current_j, previous_j),
            jnp.where(active, following_j, current_j),
            jnp.where(active, current_y, previous_y),
            jnp.where(active, following_y, current_y),
        )

    def run(maximum: int) -> tuple[Array, Array]:
        _, result_j, _, result_y = jax.lax.fori_loop(0, maximum, body, initial)
        return (
            jnp.where(integer_order == 0, first_j, result_j),
            jnp.where(integer_order == 0, first_y, result_y),
        )

    largest_order = jnp.max(integer_order)
    return jax.lax.cond(
        largest_order <= 8,
        lambda: run(8),
        lambda: jax.lax.cond(
            largest_order <= 32,
            lambda: run(32),
            lambda: jax.lax.cond(
                largest_order <= 128,
                lambda: run(128),
                lambda: run(499),
            ),
        ),
    )


def _backward_j(
    v: Array,
    integer_order: Array,
    x: Array,
    base_j: Array,
    base_j_next: Array,
) -> Array:
    extra_steps = 64

    def body(index: int, state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        following, current, target = state
        index_value = jnp.asarray(index, dtype=x.dtype)
        order = v + extra_steps - index_value
        previous = 2.0 * order * current / x - following
        capture = index == extra_steps - 1
        target = jnp.where(capture, previous, target)
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(previous), jnp.abs(current)), jnp.abs(target)
        )
        scale_limit = jnp.sqrt(jnp.finfo(x.dtype).max)
        scale = jnp.where(scale > scale_limit, scale, jnp.ones_like(scale))
        next_state = (current / scale, previous / scale, target / scale)
        active = index < integer_order + extra_steps
        return tuple(
            jnp.where(active, new, old)
            for new, old in zip(next_state, state, strict=True)
        )

    initial = (jnp.zeros_like(x), jnp.ones_like(x), jnp.zeros_like(x))

    def run(maximum: int) -> tuple[Array, Array, Array]:
        return jax.lax.fori_loop(0, maximum, body, initial)

    largest_order = jnp.max(integer_order)
    relative_next, relative_base, relative_target = jax.lax.cond(
        largest_order <= 8,
        lambda: run(72),
        lambda: jax.lax.cond(
            largest_order <= 32,
            lambda: run(96),
            lambda: jax.lax.cond(
                largest_order <= 128,
                lambda: run(192),
                lambda: run(563),
            ),
        ),
    )
    denominator = relative_base * relative_base + relative_next * relative_next
    normalization = (relative_base * base_j + relative_next * base_j_next) / denominator
    return relative_target * normalization


def _jy_primal(v: Array, x: Array) -> tuple[Array, Array]:
    negative_v = _signbit(v) & (~_exact_zero(v))
    negative_x = _signbit(x) & (~_exact_zero(x))
    invalid = negative_v | negative_x | jnp.isnan(v) | jnp.isnan(x) | jnp.isinf(v)
    zero = _exact_zero(x)
    infinity = jnp.isposinf(x)
    safe_v = jnp.where(invalid, jnp.ones_like(v), v)
    safe_x = jnp.where(invalid | zero | infinity, jnp.ones_like(x), x)

    integer_order = jnp.floor(safe_v).astype(jnp.int32)
    fractional_order = safe_v - integer_order
    base_j, base_y, base_j_next, base_y_next = _base_pair_values(fractional_order, safe_x)
    forward_j, forward_y = _forward_pair(
        fractional_order,
        integer_order,
        safe_x,
        base_j,
        base_j_next,
        base_y,
        base_y_next,
    )
    power_region = safe_x < 3.6 * jnp.sqrt(safe_v + 1.0)
    backward_region = (~power_region) & (safe_x < safe_v)
    backward_x = jnp.where(backward_region, safe_x, jnp.ones_like(safe_x))
    backward_j = _backward_j(
        safe_v,
        integer_order,
        backward_x,
        base_j,
        base_j_next,
    )

    target_series_x = jnp.where(power_region, safe_x, jnp.ones_like(safe_x))
    target_series = _power_series_j(safe_v, target_series_x)
    recurrent_j = jnp.where(safe_x >= safe_v, forward_j, backward_j)
    moderate_j = jnp.where(power_region, target_series, recurrent_j)

    large_order = safe_v >= 500.0
    hankel_region = large_order & (safe_x > 0.3 * safe_v * safe_v)
    transition_region = jnp.abs(safe_x - safe_v) / jnp.cbrt(safe_v) <= 0.7
    uniform_v = jnp.where(large_order, safe_v, jnp.full_like(safe_v, 500.0))
    uniform_x = jnp.where(
        large_order & (~transition_region) & (~hankel_region),
        safe_x,
        0.5 * uniform_v,
    )
    uniform = _uniform(uniform_v, uniform_x)
    transition_x = jnp.where(large_order & transition_region, safe_x, uniform_v)
    transition = _transition(uniform_v, transition_x)
    large_j = jnp.where(transition_region, transition[0], uniform[0])
    large_y = jnp.where(transition_region, transition[1], uniform[1])

    asymptotic_x = jnp.where(hankel_region, safe_x, jnp.full_like(safe_x, 100.0))
    asymptotic = _hankel_asymptotic(safe_v, asymptotic_x)

    j_value = jnp.where(large_order, large_j, moderate_j)
    y_value = jnp.where(large_order, large_y, forward_y)
    j_value = jnp.where(large_order & power_region, target_series, j_value)
    j_value = jnp.where(hankel_region, asymptotic[0], j_value)
    y_value = jnp.where(hankel_region, asymptotic[1], y_value)

    j_value = jnp.where(zero, jnp.where(v == 0.0, 1.0, 0.0), j_value)
    y_value = jnp.where(zero, jnp.full_like(y_value, -jnp.inf), y_value)
    j_value = jnp.where(infinity & jnp.isfinite(v), 0.0, j_value)
    y_value = jnp.where(infinity & jnp.isfinite(v), 0.0, y_value)
    nan = jnp.full_like(j_value, jnp.nan)
    return jnp.where(invalid, nan, j_value), jnp.where(invalid, nan, y_value)


@jax.custom_jvp
def _jv_array(v: Array, x: Array) -> Array:
    return _jy_primal(v, x)[0]


@jax.custom_jvp
def _yv_array(v: Array, x: Array) -> Array:
    return _jy_primal(v, x)[1]


def _require_constant_order(v_tangent: Array | SymbolicZero) -> None:
    if not isinstance(v_tangent, SymbolicZero):
        raise TypeError(
            "cylindrical Bessel functions are not differentiable with respect "
            "to the order; differentiate with respect to x instead"
        )


def _j_zero_derivative(v: Array, x: Array) -> Array:
    x2 = x * x
    zero_order = -0.5 * x + x * x2 / 16.0
    first_order = 0.5 - 3.0 * x2 / 16.0 + 5.0 * x2 * x2 / 384.0
    second_order = 0.25 * x - x * x2 / 24.0

    exact_zero_order = _exact_zero(v)
    special_order = exact_zero_order | (v == 1.0) | (v == 2.0) | jnp.isinf(v)
    safe_v = jnp.where(special_order, jnp.full_like(v, 3.0), v)
    coefficient = jnp.exp(-safe_v * math.log(2.0) - jsp.gammaln(safe_v + 1.0))
    general_x = jnp.where(_exact_zero(x) & special_order, jnp.ones_like(x), x)
    log_x = _positive_log(general_x)
    derivative = coefficient * (
        safe_v * jnp.exp((safe_v - 1.0) * log_x)
        - (safe_v + 2.0) * jnp.exp((safe_v + 1.0) * log_x) / (4.0 * (safe_v + 1.0))
        + (safe_v + 4.0)
        * jnp.exp((safe_v + 3.0) * log_x)
        / (32.0 * (safe_v + 1.0) * (safe_v + 2.0))
    )
    derivative = jnp.where(exact_zero_order, zero_order, derivative)
    derivative = jnp.where(v == 1.0, first_order, derivative)
    derivative = jnp.where(v == 2.0, second_order, derivative)
    return jnp.where(jnp.isposinf(v), jnp.zeros_like(derivative), derivative)


def _y_zero_derivative(v: Array, x: Array) -> Array:
    exact_zero_order = _exact_zero(v)
    positive_finite_order = (~_signbit(v)) & (~exact_zero_order) & jnp.isfinite(v)
    safe_v = jnp.where(positive_finite_order, v, jnp.ones_like(v))
    general_x = jnp.where(positive_finite_order, x, jnp.ones_like(x))
    coefficient = jnp.exp(
        jsp.gammaln(safe_v) + safe_v * math.log(2.0) - math.log(math.pi)
    )
    derivative = (
        safe_v * coefficient * jnp.exp((-safe_v - 1.0) * _positive_log(general_x))
    )
    zero_x = jnp.where(exact_zero_order, x, jnp.ones_like(x))
    zero_derivative = (2.0 / math.pi) * jnp.exp(-_positive_log(zero_x))
    derivative = jnp.where(exact_zero_order, zero_derivative, derivative)
    return jnp.where(jnp.isposinf(v), jnp.full_like(derivative, jnp.inf), derivative)


def _jv_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    v, x = primals
    v_tangent, x_tangent = tangents
    _require_constant_order(v_tangent)
    value = _jv_array(v, x)
    small_x = _exact_zero(x) | _positive_subnormal(x)
    recurrence_x = jnp.where(small_x, jnp.ones_like(x), x)
    zero_order = _exact_zero(v)
    ratio_x = jnp.where(zero_order, jnp.ones_like(recurrence_x), recurrence_x)
    order_term = jnp.where(zero_order, jnp.zeros_like(value), v * value / ratio_x)
    derivative = order_term - _jv_array(v + 1.0, recurrence_x)
    derivative = jnp.where(small_x, _j_zero_derivative(v, x), derivative)
    derivative = jnp.where(jnp.isposinf(x), 0.0, derivative)
    if isinstance(x_tangent, SymbolicZero):
        return value, jnp.zeros_like(value)
    return value, derivative * x_tangent


def _yv_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    v, x = primals
    v_tangent, x_tangent = tangents
    _require_constant_order(v_tangent)
    value = _yv_array(v, x)
    zero = _exact_zero(x)
    subnormal = _positive_subnormal(x)
    small_x = zero | subnormal
    regular_positive = (~_signbit(x)) & (~small_x) & jnp.isfinite(x)
    safe_x = jnp.where(regular_positive, x, jnp.ones_like(x))
    safe_value = _yv_array(v, safe_x)
    zero_order = _exact_zero(v)
    ratio_x = jnp.where(zero_order, jnp.ones_like(safe_x), safe_x)
    order_term = jnp.where(
        zero_order, jnp.zeros_like(safe_value), v * safe_value / ratio_x
    )
    derivative = order_term - _yv_array(v + 1.0, safe_x)
    derivative = jnp.where(small_x, _y_zero_derivative(v, x), derivative)
    derivative = jnp.where(jnp.isposinf(x), 0.0, derivative)
    if isinstance(x_tangent, SymbolicZero):
        return value, jnp.zeros_like(value)
    return value, derivative * x_tangent


_jv_array.defjvp(_jv_jvp, symbolic_zeros=True)
_yv_array.defjvp(_yv_jvp, symbolic_zeros=True)


def _prepare(name: str, v: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
    v, x = promote_real(name, v, x)
    v, x = jnp.broadcast_arrays(v, x)
    return v, x


def jv(v: ArrayLike, x: ArrayLike) -> Array:
    """Cylindrical Bessel function of the first kind ``J_v(x)``."""
    return _jv_array(*_prepare("jv", v, x))


def yv(v: ArrayLike, x: ArrayLike) -> Array:
    """Cylindrical Bessel function of the second kind ``Y_v(x)``."""
    return _yv_array(*_prepare("yv", v, x))


def hankel1(v: ArrayLike, x: ArrayLike) -> Array:
    """Outgoing cylindrical wave ``J_v(x) + i Y_v(x)``."""
    v, x = _prepare("hankel1", v, x)
    j_value = _jv_array(v, x)
    y_value = _yv_array(v, x)
    return complex_from_parts("hankel1", j_value, y_value)


def hankel2(v: ArrayLike, x: ArrayLike) -> Array:
    """Incoming cylindrical wave ``J_v(x) - i Y_v(x)``."""
    v, x = _prepare("hankel2", v, x)
    j_value = _jv_array(v, x)
    y_value = _yv_array(v, x)
    return complex_from_parts("hankel2", j_value, -y_value)


__all__ = ["hankel1", "hankel2", "jv", "yv"]
