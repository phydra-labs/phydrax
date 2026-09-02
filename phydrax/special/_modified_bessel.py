#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# The numerical regime structure is adapted from Numerax 1.4.0, licensed under
# MIT. See NOTICE and LICENSES/NUMERAX-MIT.txt.
#

"""Real-order modified Bessel functions on the nonnegative real axis."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import Array
from jax.custom_derivatives import SymbolicZero
from jax.typing import ArrayLike

from ._dtype import (
    _exact_zero,
    _positive_log,
    _positive_subnormal,
    _signbit,
    promote_real,
)
from ._gamma import log_gamma_one_plus_minus_difference


_EULER_GAMMA = 0.577215664901532860606512090082402431


def _add_polynomial_term(target: list[float], degree: int, coefficient: float) -> None:
    while len(target) <= degree:
        target.append(0.0)
    target[degree] += coefficient


def _debye_polynomials(count: int) -> tuple[tuple[float, ...], ...]:
    """Generate the Debye ``U_k`` polynomials from their defining recurrence."""
    polynomials: list[list[float]] = [[1.0]]
    for _ in range(count):
        current = polynomials[-1]
        next_polynomial: list[float] = []
        for degree, coefficient in enumerate(current[1:], start=1):
            derivative = degree * coefficient
            _add_polynomial_term(next_polynomial, degree + 1, 0.5 * derivative)
            _add_polynomial_term(next_polynomial, degree + 3, -0.5 * derivative)
        for degree, coefficient in enumerate(current):
            _add_polynomial_term(
                next_polynomial, degree + 1, coefficient / (8.0 * (degree + 1))
            )
            _add_polynomial_term(
                next_polynomial,
                degree + 3,
                -5.0 * coefficient / (8.0 * (degree + 3)),
            )
        polynomials.append(next_polynomial)
    return tuple(tuple(polynomial) for polynomial in polynomials[1:])


_DEBYE_POLYNOMIALS = _debye_polynomials(8)


def _polynomial(x: Array, coefficients: tuple[float, ...]) -> Array:
    value = jnp.zeros_like(x)
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def _olver_logs(v: Array, x: Array) -> tuple[Array, Array]:
    ratio = x / v
    root = jnp.hypot(jnp.ones_like(ratio), ratio)
    t = 1.0 / root
    large_ratio = ratio >= 1.0
    safe_large_ratio = jnp.where(large_ratio, ratio, jnp.ones_like(ratio))
    large_shared_exponent = (1.0 / root) / (1.0 + safe_large_ratio / root) - jnp.arcsinh(
        1.0 / safe_large_ratio
    )
    safe_small_ratio = jnp.where(large_ratio, jnp.ones_like(ratio), ratio)
    small_root = jnp.hypot(jnp.ones_like(safe_small_ratio), safe_small_ratio)
    small_shared_exponent = (
        1.0 / (small_root + safe_small_ratio)
        + jnp.log(safe_small_ratio)
        - jnp.log1p(small_root)
    )
    shared_exponent = jnp.where(large_ratio, large_shared_exponent, small_shared_exponent)
    inverse_order = 1.0 / v
    power = inverse_order
    ive_sum = jnp.ones_like(v)
    kve_sum = jnp.ones_like(v)
    for index, coefficients in enumerate(_DEBYE_POLYNOMIALS, start=1):
        term = _polynomial(t, coefficients) * power
        ive_sum = ive_sum + term
        kve_sum = kve_sum + (-term if index % 2 else term)
        power = power * inverse_order
    common = -0.5 * jnp.log(root)
    log_ive = (
        -0.5 * jnp.log(2.0 * math.pi * v)
        + common
        + v * shared_exponent
        + jnp.log(ive_sum)
    )
    log_kve = (
        0.5 * jnp.log(math.pi / (2.0 * v))
        + common
        - v * shared_exponent
        + jnp.log(kve_sum)
    )
    return log_ive, log_kve


def _ive_series_log(v: Array, x: Array) -> Array:
    log_half_x = _positive_log(x) - math.log(2.0)
    initial_term = v * log_half_x - jsp.gammaln(v + 1.0)
    initial_term = jnp.where(_exact_zero(v), jnp.zeros_like(initial_term), initial_term)

    def body(k: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        log_term, log_total = state
        k_value = jnp.asarray(k, dtype=x.dtype)
        log_term = log_term + 2.0 * log_half_x - jnp.log(k_value) - jnp.log(v + k_value)
        return log_term, jnp.logaddexp(log_total, log_term)

    _, log_total = jax.lax.fori_loop(1, 513, body, (initial_term, initial_term))
    return log_total - x


def _kve_large_x_log(v: Array, x: Array) -> Array:
    mu = 4.0 * v * v

    def body(k: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        term, total = state
        k_value = jnp.asarray(k, dtype=x.dtype)
        odd = 2.0 * k_value - 1.0
        term = term * (mu - odd * odd) / (8.0 * k_value * x)
        return term, total + term

    _, correction = jax.lax.fori_loop(1, 16, body, (jnp.ones_like(x), jnp.ones_like(x)))
    return 0.5 * (math.log(math.pi / 2.0) - jnp.log(x)) + jnp.log(correction)


def _ive_large_x_log(v: Array, x: Array) -> Array:
    mu = 4.0 * v * v

    def body(k: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        term, total = state
        k_value = jnp.asarray(k, dtype=x.dtype)
        odd = 2.0 * k_value - 1.0
        term = -term * (mu - odd * odd) / (8.0 * k_value * x)
        return term, total + term

    _, correction = jax.lax.fori_loop(1, 16, body, (jnp.ones_like(x), jnp.ones_like(x)))
    return -0.5 * (math.log(2.0 * math.pi) + jnp.log(x)) + jnp.log(correction)


def _ive_log(v: Array, x: Array) -> Array:
    float32_uniform = (v.dtype == jnp.float32) & (v >= 10.0) & (x > 50.0)
    large_order = (v >= 30.0) | float32_uniform
    large_argument = x > jnp.maximum(50.0, 0.5 * v * v)
    safe_series_v = jnp.where(large_order, jnp.ones_like(v), v)
    safe_series_x = jnp.where(
        large_order | large_argument | _exact_zero(x), jnp.ones_like(x), x
    )
    series = _ive_series_log(safe_series_v, safe_series_x)
    asymptotic_x = jnp.where(large_argument, x, jnp.full_like(x, 100.0))
    asymptotic = _ive_large_x_log(safe_series_v, asymptotic_x)
    small_order = jnp.where(large_argument, asymptotic, series)
    olver_v = jnp.where(large_order, v, jnp.full_like(v, 30.0))
    positive_x = (~_signbit(x)) & (~_exact_zero(x))
    olver_x = jnp.where(large_order & positive_x, x, jnp.ones_like(x))
    olver, _ = _olver_logs(olver_v, olver_x)
    return jnp.where(large_order, olver, small_order)


def _temme_gamma_coefficients(v: Array) -> tuple[Array, Array, Array, Array]:
    log_minus = -jsp.gammaln(1.0 - v)
    log_plus = -jsp.gammaln(1.0 + v)
    mean = 0.5 * (log_minus + log_plus)
    difference = 0.5 * log_gamma_one_plus_minus_difference(v)
    exponential = jnp.exp(mean)
    coefficient_one = exponential * jnp.where(
        v == 0.0, -_EULER_GAMMA, jnp.sinh(difference) / v
    )
    coefficient_two = exponential * jnp.cosh(difference)
    reciprocal_gamma_plus = jnp.exp(log_plus)
    reciprocal_gamma_minus = jnp.exp(log_minus)
    return (
        coefficient_one,
        coefficient_two,
        reciprocal_gamma_plus,
        reciprocal_gamma_minus,
    )


def _temme_small_k_logs(v: Array, x: Array) -> tuple[Array, Array]:
    coefficient_one, coefficient_two, reciprocal_gamma_plus, reciprocal_gamma_minus = (
        _temme_gamma_coefficients(v)
    )
    log_half_x = _positive_log(x) - math.log(2.0)
    mu = -v * log_half_x
    sinc = jnp.where(v == 0.0, jnp.ones_like(v), jnp.sinc(v))
    sinhc = jnp.where(mu == 0.0, jnp.ones_like(mu), jnp.sinh(mu) / mu)
    f = (coefficient_one * jnp.cosh(mu) - coefficient_two * log_half_x * sinhc) / sinc
    p = 0.5 * jnp.exp(mu) / reciprocal_gamma_plus
    q = 0.5 * jnp.exp(-mu) / reciprocal_gamma_minus

    def body(
        k: int, state: tuple[Array, Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        current_f, current_p, current_q, coefficient, k_sum, kp1_sum = state
        index = jnp.asarray(k, dtype=x.dtype)
        current_f = (index * current_f + current_p + current_q) / (index * index - v * v)
        current_p = current_p / (index - v)
        current_q = current_q / (index + v)
        h = current_p - index * current_f
        coefficient = coefficient * x * x / (4.0 * index)
        return (
            current_f,
            current_p,
            current_q,
            coefficient,
            k_sum + coefficient * current_f,
            kp1_sum + coefficient * h,
        )

    _, _, _, _, k_sum, kp1_sum = jax.lax.fori_loop(
        1,
        61,
        body,
        (f, p, q, jnp.ones_like(x), f, p),
    )
    log_kp1 = jnp.log(2.0 * kp1_sum) - _positive_log(x)
    return jnp.log(k_sum) + x, log_kp1 + x


def _continued_fraction_k_logs(v: Array, x: Array) -> tuple[Array, Array]:
    initial_numerator = v * v - 0.25
    initial_denominator = 2.0 * (x + 1.0)
    initial_ratio = 1.0 / initial_denominator
    initial_sequence = -initial_numerator
    tolerance = jnp.finfo(x.dtype).eps

    def body(index: int, state: tuple[Array, ...]) -> tuple[Array, ...]:
        (
            numerator,
            denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            previous_k,
            current_k,
            coefficient,
            q,
            hypergeometric_sum,
            active,
        ) = state
        index_value = jnp.asarray(index, dtype=x.dtype)
        next_numerator = numerator - 2.0 * (index_value - 1.0)
        next_coefficient = -coefficient * next_numerator / index_value
        next_k = (previous_k - denominator * current_k) / next_numerator
        next_q = q + next_coefficient * next_k
        next_denominator = denominator + 2.0
        next_denominator_ratio = 1.0 / (
            next_denominator + next_numerator * denominator_ratio
        )
        next_difference = convergent_difference * (
            next_denominator * next_denominator_ratio - 1.0
        )
        next_ratio = hypergeometric_ratio + next_difference
        next_sum = hypergeometric_sum + next_q * next_difference
        converged = jnp.abs(next_q * next_difference) < jnp.abs(next_sum) * tolerance
        choose = active
        return (
            jnp.where(choose, next_numerator, numerator),
            jnp.where(choose, next_denominator, denominator),
            jnp.where(choose, next_denominator_ratio, denominator_ratio),
            jnp.where(choose, next_difference, convergent_difference),
            jnp.where(choose, next_ratio, hypergeometric_ratio),
            jnp.where(choose, current_k, previous_k),
            jnp.where(choose, next_k, current_k),
            jnp.where(choose, next_coefficient, coefficient),
            jnp.where(choose, next_q, q),
            jnp.where(choose, next_sum, hypergeometric_sum),
            active & ~converged,
        )

    (
        _,
        _,
        _,
        _,
        hypergeometric_ratio,
        _,
        _,
        _,
        _,
        hypergeometric_sum,
        _,
    ) = jax.lax.fori_loop(
        2,
        101,
        body,
        (
            initial_numerator,
            initial_denominator,
            initial_ratio,
            initial_ratio,
            initial_ratio,
            jnp.zeros_like(x),
            jnp.ones_like(x),
            initial_sequence,
            initial_sequence,
            1.0 - initial_numerator * initial_ratio,
            jnp.ones_like(x, dtype=jnp.bool_),
        ),
    )
    log_k = 0.5 * jnp.log(math.pi / (2.0 * x)) - jnp.log(hypergeometric_sum)
    log_kp1 = (
        log_k
        + jnp.log1p(2.0 * (v + x + initial_numerator * hypergeometric_ratio))
        - jnp.log(2.0 * x)
    )
    return log_k, log_kp1


def _small_order_kve_log(v: Array, x: Array) -> Array:
    nearest = jnp.floor(v + 0.5)
    reduced_order = v - nearest
    half_integer = jnp.abs(reduced_order) == 0.5
    evaluation_order = jnp.where(half_integer, jnp.zeros_like(v), reduced_order)
    small_x = x <= 2.0
    temme_x = jnp.where(small_x, x, jnp.ones_like(x))
    fraction_x = jnp.where(small_x, jnp.full_like(x, 4.0), x)
    temme = _temme_small_k_logs(evaluation_order, temme_x)
    fraction = _continued_fraction_k_logs(evaluation_order, fraction_x)
    current = jnp.where(small_x, temme[0], fraction[0])
    following = jnp.where(small_x, temme[1], fraction[1])
    half_integer_log = 0.5 * (math.log(math.pi / 2.0) - _positive_log(x))
    current = jnp.where(half_integer, half_integer_log, current)
    following = jnp.where(half_integer, half_integer_log, following)

    def body(index: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        current_log, following_log = state
        index_value = jnp.asarray(index, dtype=x.dtype)
        active = index_value <= nearest
        log_coefficient = jnp.log(2.0 * (reduced_order + index_value)) - _positive_log(x)
        next_log = jnp.logaddexp(current_log, following_log + log_coefficient)
        return (
            jnp.where(active, following_log, current_log),
            jnp.where(active, next_log, following_log),
        )

    current, _ = jax.lax.fori_loop(1, 31, body, (current, following))
    return current


def _kve_log(v: Array, x: Array) -> Array:
    large_order = v >= 30.0
    large_argument = x > jnp.maximum(50.0, 0.5 * v * v)
    safe_v = jnp.where(large_order, jnp.full_like(v, 0.25), v)
    safe_x = jnp.where(_exact_zero(x) | large_order | large_argument, jnp.ones_like(x), x)
    small_order = _small_order_kve_log(safe_v, safe_x)
    asymptotic_x = jnp.where(large_argument, x, jnp.full_like(x, 100.0))
    asymptotic = _kve_large_x_log(safe_v, asymptotic_x)
    small_order = jnp.where(large_argument, asymptotic, small_order)
    olver_v = jnp.where(large_order, v, jnp.full_like(v, 30.0))
    positive_x = (~_signbit(x)) & (~_exact_zero(x))
    olver_x = jnp.where(large_order & positive_x, x, jnp.ones_like(x))
    _, olver = _olver_logs(olver_v, olver_x)
    return jnp.where(large_order, olver, small_order)


def _invalid(v: Array, x: Array) -> Array:
    negative_v = _signbit(v) & (~_exact_zero(v))
    negative_x = _signbit(x) & (~_exact_zero(x))
    return negative_v | negative_x | jnp.isnan(v) | jnp.isnan(x)


def _ive_primal(v: Array, x: Array) -> Array:
    invalid = _invalid(v, x)
    zero = _exact_zero(x)
    safe_v = jnp.where(invalid | jnp.isinf(v), jnp.ones_like(v), v)
    safe_x = jnp.where(invalid | zero | jnp.isinf(x), jnp.ones_like(x), x)
    value = jnp.exp(_ive_log(safe_v, safe_x))
    value = jnp.where(zero, jnp.where(v == 0.0, 1.0, 0.0), value)
    value = jnp.where(jnp.isposinf(x) & jnp.isfinite(v), 0.0, value)
    value = jnp.where(jnp.isposinf(v) & jnp.isfinite(x), 0.0, value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _iv_primal(v: Array, x: Array) -> Array:
    invalid = _invalid(v, x)
    zero = _exact_zero(x)
    safe_v = jnp.where(invalid | jnp.isinf(v), jnp.ones_like(v), v)
    safe_x = jnp.where(invalid | zero | jnp.isinf(x), jnp.ones_like(x), x)
    value = jnp.exp(_ive_log(safe_v, safe_x) + safe_x)
    value = jnp.where(zero, jnp.where(v == 0.0, 1.0, 0.0), value)
    value = jnp.where(jnp.isposinf(x) & jnp.isfinite(v), jnp.inf, value)
    value = jnp.where(jnp.isposinf(v) & jnp.isfinite(x), 0.0, value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _kve_primal(v: Array, x: Array) -> Array:
    invalid = _invalid(v, x)
    zero = _exact_zero(x)
    safe_v = jnp.where(invalid | jnp.isinf(v), jnp.ones_like(v), v)
    safe_x = jnp.where(invalid | zero | jnp.isinf(x), jnp.ones_like(x), x)
    value = jnp.exp(_kve_log(safe_v, safe_x))
    value = jnp.where(zero, jnp.full_like(value, jnp.inf), value)
    value = jnp.where(jnp.isposinf(x) & jnp.isfinite(v), 0.0, value)
    value = jnp.where(jnp.isposinf(v) & jnp.isfinite(x), jnp.inf, value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _kv_primal(v: Array, x: Array) -> Array:
    invalid = _invalid(v, x)
    zero = _exact_zero(x)
    safe_v = jnp.where(invalid | jnp.isinf(v), jnp.ones_like(v), v)
    safe_x = jnp.where(invalid | zero | jnp.isinf(x), jnp.ones_like(x), x)
    value = jnp.exp(_kve_log(safe_v, safe_x) - safe_x)
    value = jnp.where(zero, jnp.full_like(value, jnp.inf), value)
    value = jnp.where(jnp.isposinf(x) & jnp.isfinite(v), 0.0, value)
    value = jnp.where(jnp.isposinf(v) & jnp.isfinite(x), jnp.inf, value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@jax.custom_jvp
def _ive_array(v: Array, x: Array) -> Array:
    return _ive_primal(v, x)


@jax.custom_jvp
def _iv_array(v: Array, x: Array) -> Array:
    return _iv_primal(v, x)


@jax.custom_jvp
def _kve_array(v: Array, x: Array) -> Array:
    return _kve_primal(v, x)


@jax.custom_jvp
def _kv_array(v: Array, x: Array) -> Array:
    return _kv_primal(v, x)


def _apply_tangent(derivative: Array, tangent: Array | SymbolicZero) -> Array:
    if isinstance(tangent, SymbolicZero):
        return jnp.zeros_like(derivative)
    return derivative * tangent


def _i_zero_derivative(v: Array, x: Array, *, scaled: bool) -> Array:
    x2 = x * x
    i0 = 1.0 + 0.25 * x2 + x2 * x2 / 64.0
    i0_derivative = 0.5 * x + x * x2 / 16.0
    zero_order = jnp.exp(-x) * (i0_derivative - i0) if scaled else i0_derivative

    i1 = 0.5 * x + x * x2 / 16.0 + x * x2 * x2 / 384.0
    i1_derivative = 0.5 + 3.0 * x2 / 16.0 + 5.0 * x2 * x2 / 384.0
    first_order = jnp.exp(-x) * (i1_derivative - i1) if scaled else i1_derivative

    i2 = x2 / 8.0 + x2 * x2 / 96.0
    i2_derivative = 0.25 * x + x * x2 / 24.0
    second_order = jnp.exp(-x) * (i2_derivative - i2) if scaled else i2_derivative

    exact_zero_order = _exact_zero(v)
    special_order = exact_zero_order | (v == 1.0) | (v == 2.0) | jnp.isinf(v)
    safe_v = jnp.where(special_order, jnp.full_like(v, 3.0), v)
    coefficient = jnp.exp(-safe_v * math.log(2.0) - jsp.gammaln(safe_v + 1.0))
    general_x = jnp.where(_exact_zero(x) & special_order, jnp.ones_like(x), x)
    log_x = _positive_log(general_x)
    value = coefficient * jnp.exp(safe_v * log_x) * (1.0 + x2 / (4.0 * (safe_v + 1.0)))
    derivative = coefficient * (
        safe_v * jnp.exp((safe_v - 1.0) * log_x)
        + (safe_v + 2.0) * jnp.exp((safe_v + 1.0) * log_x) / (4.0 * (safe_v + 1.0))
    )
    if scaled:
        derivative = jnp.exp(-x) * (derivative - value)
    derivative = jnp.where(exact_zero_order, zero_order, derivative)
    derivative = jnp.where(v == 1.0, first_order, derivative)
    derivative = jnp.where(v == 2.0, second_order, derivative)
    return jnp.where(jnp.isposinf(v), jnp.zeros_like(derivative), derivative)


def _k_zero_derivative(v: Array, x: Array) -> Array:
    exact_zero_order = _exact_zero(v)
    positive_finite_order = (~_signbit(v)) & (~exact_zero_order) & jnp.isfinite(v)
    safe_v = jnp.where(positive_finite_order, v, jnp.ones_like(v))
    general_x = jnp.where(positive_finite_order, x, jnp.ones_like(x))
    coefficient = jnp.exp((safe_v - 1.0) * math.log(2.0) + jsp.gammaln(safe_v))
    derivative = (
        -safe_v * coefficient * jnp.exp((-safe_v - 1.0) * _positive_log(general_x))
    )
    zero_x = jnp.where(exact_zero_order, x, jnp.ones_like(x))
    zero_derivative = -jnp.exp(-_positive_log(zero_x))
    derivative = jnp.where(exact_zero_order, zero_derivative, derivative)
    return jnp.where(jnp.isposinf(v), jnp.full_like(derivative, -jnp.inf), derivative)


def _scaled_log_derivative(
    evaluator: Callable[[Array, Array], Array], v: Array, x: Array
) -> Array:
    nonnegative_v = (~_signbit(v)) | _exact_zero(v)
    positive_x = (~_signbit(x)) & (~_exact_zero(x))
    valid = nonnegative_v & jnp.isfinite(v) & positive_x & jnp.isfinite(x)
    safe_v = jnp.where(valid, v, jnp.ones_like(v))
    safe_x = jnp.where(valid, x, jnp.ones_like(x))
    _, derivative = jax.jvp(
        lambda argument: evaluator(safe_v, argument),
        (safe_x,),
        (jnp.ones_like(safe_x),),
    )
    return jnp.where(valid, derivative, jnp.zeros_like(derivative))


def _ive_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    v, x = primals
    v_tangent, x_tangent = tangents
    value = _ive_array(v, x)
    small_x = _exact_zero(x) | _positive_subnormal(x)
    derivative_x = jnp.where(small_x, jnp.ones_like(x), x)
    derivative = value * _scaled_log_derivative(_ive_log, v, derivative_x)
    derivative = jnp.where(small_x, _i_zero_derivative(v, x, scaled=True), derivative)
    derivative = jnp.where(jnp.isposinf(x), jnp.zeros_like(x), derivative)
    tangent = _apply_tangent(derivative, x_tangent)
    if not isinstance(v_tangent, SymbolicZero):
        from ._continuation import ive_order_derivative

        tangent = (
            tangent + jnp.real(ive_order_derivative(v, x)).astype(value.dtype) * v_tangent
        )
    return value, tangent


def _iv_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    v, x = primals
    v_tangent, x_tangent = tangents
    value = _iv_array(v, x)
    small_x = _exact_zero(x) | _positive_subnormal(x)
    recurrence_x = jnp.where(small_x, jnp.ones_like(x), x)
    zero_order = _exact_zero(v)
    ratio_x = jnp.where(zero_order, jnp.ones_like(recurrence_x), recurrence_x)
    order_term = jnp.where(zero_order, jnp.zeros_like(value), v * value / ratio_x)
    derivative = _iv_array(v + 1.0, recurrence_x) + order_term
    derivative = jnp.where(small_x, _i_zero_derivative(v, x, scaled=False), derivative)
    derivative = jnp.where(jnp.isposinf(x), jnp.full_like(x, jnp.inf), derivative)
    tangent = _apply_tangent(derivative, x_tangent)
    if not isinstance(v_tangent, SymbolicZero):
        from ._continuation import iv_order_derivative

        tangent = (
            tangent + jnp.real(iv_order_derivative(v, x)).astype(value.dtype) * v_tangent
        )
    return value, tangent


def _kve_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    v, x = primals
    v_tangent, x_tangent = tangents
    value = _kve_array(v, x)
    zero = _exact_zero(x)
    subnormal = _positive_subnormal(x)
    small_x = zero | subnormal
    regular_positive = (~_signbit(x)) & (~small_x) & jnp.isfinite(x)
    safe_x = jnp.where(regular_positive, x, jnp.ones_like(x))
    safe_value = _kve_array(v, safe_x)
    derivative = safe_value * _scaled_log_derivative(_kve_log, v, safe_x)
    derivative = jnp.where(small_x, _k_zero_derivative(v, x), derivative)
    derivative = jnp.where(jnp.isposinf(x), jnp.zeros_like(x), derivative)
    tangent = _apply_tangent(derivative, x_tangent)
    if not isinstance(v_tangent, SymbolicZero):
        from ._continuation import kve_order_derivative

        tangent = (
            tangent + jnp.real(kve_order_derivative(v, x)).astype(value.dtype) * v_tangent
        )
    return value, tangent


def _kv_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    v, x = primals
    v_tangent, x_tangent = tangents
    value = _kv_array(v, x)
    zero = _exact_zero(x)
    subnormal = _positive_subnormal(x)
    small_x = zero | subnormal
    regular_positive = (~_signbit(x)) & (~small_x) & jnp.isfinite(x)
    safe_x = jnp.where(regular_positive, x, jnp.ones_like(x))
    safe_value = _kv_array(v, safe_x)
    zero_order = _exact_zero(v)
    ratio_x = jnp.where(zero_order, jnp.ones_like(safe_x), safe_x)
    order_term = jnp.where(
        zero_order, jnp.zeros_like(safe_value), v * safe_value / ratio_x
    )
    derivative = order_term - _kv_array(v + 1.0, safe_x)
    derivative = jnp.where(small_x, _k_zero_derivative(v, x), derivative)
    derivative = jnp.where(jnp.isposinf(x), jnp.zeros_like(x), derivative)
    tangent = _apply_tangent(derivative, x_tangent)
    if not isinstance(v_tangent, SymbolicZero):
        from ._continuation import kv_order_derivative

        tangent = (
            tangent + jnp.real(kv_order_derivative(v, x)).astype(value.dtype) * v_tangent
        )
    return value, tangent


_ive_array.defjvp(_ive_jvp, symbolic_zeros=True)
_iv_array.defjvp(_iv_jvp, symbolic_zeros=True)
_kve_array.defjvp(_kve_jvp, symbolic_zeros=True)
_kv_array.defjvp(_kv_jvp, symbolic_zeros=True)


def _prepare(name: str, v: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
    v, x = promote_real(name, v, x)
    v, x = jnp.broadcast_arrays(v, x)
    return v, x


def ive(v: ArrayLike, x: ArrayLike) -> Array:
    """Exponentially scaled principal modified Bessel function ``I_v(x)``."""
    if jnp.issubdtype(jnp.result_type(v, x), jnp.complexfloating):
        from ._continuation import complex_iv, promote_principal

        (_, z) = promote_principal(v, x)
        return jnp.exp(-jnp.abs(jnp.real(z))) * complex_iv(v, z)
    return _ive_array(*_prepare("ive", v, x))


def iv(v: ArrayLike, x: ArrayLike) -> Array:
    """Principal modified Bessel function of the first kind ``I_v(x)``."""
    if jnp.issubdtype(jnp.result_type(v, x), jnp.complexfloating):
        from ._continuation import complex_iv

        return complex_iv(v, x)
    return _iv_array(*_prepare("iv", v, x))


def kve(v: ArrayLike, x: ArrayLike) -> Array:
    """Exponentially scaled principal modified Bessel function ``K_v(x)``."""
    if jnp.issubdtype(jnp.result_type(v, x), jnp.complexfloating):
        from ._continuation import complex_kv, promote_principal

        (_, z) = promote_principal(v, x)
        return jnp.exp(z) * complex_kv(v, z)
    return _kve_array(*_prepare("kve", v, x))


def kv(v: ArrayLike, x: ArrayLike) -> Array:
    """Principal modified Bessel function of the second kind ``K_v(x)``."""
    if jnp.issubdtype(jnp.result_type(v, x), jnp.complexfloating):
        from ._continuation import complex_kv

        return complex_kv(v, x)
    return _kv_array(*_prepare("kv", v, x))


__all__ = ["iv", "ive", "kv", "kve"]
