#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete and incomplete real Legendre elliptic integrals."""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._carlson import elliprd, elliprf, elliprj
from ._dtype import _exact_zero, _positive_log, _signbit, promote_real


def _complete_k_primal(m: Array) -> Array:
    invalid = m > 1.0
    pole = m == 1.0
    safe_m = jnp.where(invalid | pole, jnp.zeros_like(m), m)
    value = elliprf(jnp.zeros_like(safe_m), 1.0 - safe_m, jnp.ones_like(safe_m))
    value = jnp.where(pole, jnp.full_like(value, jnp.inf), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _complete_e_primal(m: Array) -> Array:
    invalid = m > 1.0
    endpoint = m == 1.0
    negative = m < 0.0
    transformed = jnp.where(negative, m / (m - 1.0), m)
    transformed_endpoint = negative & (transformed == 1.0)
    safe_m = jnp.where(
        invalid | endpoint | transformed_endpoint, jnp.zeros_like(m), transformed
    )
    rf = elliprf(jnp.zeros_like(safe_m), 1.0 - safe_m, jnp.ones_like(safe_m))
    rd = elliprd(jnp.zeros_like(safe_m), 1.0 - safe_m, jnp.ones_like(safe_m))
    value = rf - safe_m * rd / 3.0
    value = jnp.where(transformed_endpoint, jnp.ones_like(value), value)
    value = jnp.where(negative, jnp.sqrt(1.0 - m) * value, value)
    value = jnp.where(endpoint, jnp.ones_like(value), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@jax.custom_jvp
def _ellipk_array(m: Array) -> Array:
    return _complete_k_primal(m)


@_ellipk_array.defjvp
def _ellipk_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (m,) = primals
    (m_dot,) = tangents
    value = _ellipk_array(m)
    near_zero = jnp.abs(m) < (1e-3 if m.dtype == jnp.float32 else 1e-7)
    safe_m = jnp.where(near_zero | (m == 1.0) | (m > 1.0), jnp.full_like(m, 0.5), m)
    e = _ellipe_array(jnp.where(m > 1.0, jnp.zeros_like(m), m))
    regular = e / (2.0 * safe_m) / (1.0 - safe_m) - value / (2.0 * safe_m)
    series = (math.pi / 2.0) * (0.25 + 18.0 * m / 64.0 + 75.0 * m * m / 256.0)
    derivative = jnp.where(near_zero, series, regular)
    derivative = jnp.where(m == 1.0, jnp.full_like(derivative, jnp.inf), derivative)
    derivative = jnp.where(m > 1.0, jnp.full_like(derivative, jnp.nan), derivative)
    return value, derivative * m_dot


@jax.custom_jvp
def _ellipe_array(m: Array) -> Array:
    return _complete_e_primal(m)


@_ellipe_array.defjvp
def _ellipe_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (m,) = primals
    (m_dot,) = tangents
    value = _ellipe_array(m)
    near_zero = jnp.abs(m) < (1e-3 if m.dtype == jnp.float32 else 1e-7)
    safe_m = jnp.where(near_zero | (m > 1.0), jnp.ones_like(m), m)
    k = _ellipk_array(jnp.where(m > 1.0, jnp.zeros_like(m), m))
    regular = (value - k) / (2.0 * safe_m)
    series = (math.pi / 2.0) * (-0.25 - 6.0 * m / 64.0 - 15.0 * m * m / 256.0)
    derivative = jnp.where(near_zero, series, regular)
    derivative = jnp.where(m == 1.0, jnp.full_like(derivative, -jnp.inf), derivative)
    derivative = jnp.where(m > 1.0, jnp.full_like(derivative, jnp.nan), derivative)
    return value, derivative * m_dot


def ellipk(m: ArrayLike) -> Array:
    """Complete elliptic integral of the first kind with parameter ``m``."""
    (promoted_m,) = promote_real("ellipk", m)
    return _ellipk_array(promoted_m)


def ellipe(m: ArrayLike) -> Array:
    """Complete elliptic integral of the second kind with parameter ``m``."""
    (promoted_m,) = promote_real("ellipe", m)
    return _ellipe_array(promoted_m)


def _km1_series(p: Array) -> Array:
    log_term = jnp.log(4.0) - 0.5 * _positive_log(p)
    p2 = p * p
    return (
        log_term
        + 0.25 * p * (log_term - 1.0)
        + 9.0 * p2 * (log_term - 7.0 / 6.0) / 64.0
        + 25.0 * p2 * p * (log_term - 37.0 / 30.0) / 256.0
    )


def _km1_series_derivative(p: Array) -> Array:
    positive_log = _positive_log(p)
    log_term = jnp.log(4.0) - 0.5 * positive_log
    reciprocal = jnp.exp(-positive_log)
    p2 = p * p
    return (
        -0.5 * reciprocal
        + 0.25 * (log_term - 1.5)
        + 9.0 * p * (2.0 * log_term - 17.0 / 6.0) / 64.0
        + 25.0 * p2 * (3.0 * log_term - 116.0 / 30.0) / 256.0
    )


@jax.custom_jvp
def _ellipkm1_array(p: Array) -> Array:
    pole = _exact_zero(p)
    invalid = _signbit(p) & ~pole
    positive = ~_signbit(p) & ~pole
    threshold = 1e-4 if p.dtype == jnp.float32 else 1e-8
    use_series = positive & (p < threshold)
    safe_p = jnp.where(positive, p, jnp.ones_like(p))
    direct_p = jnp.where(use_series | invalid | pole, jnp.ones_like(p), p)
    direct = elliprf(jnp.zeros_like(p), direct_p, jnp.ones_like(p))
    value = jnp.where(use_series, _km1_series(safe_p), direct)
    value = jnp.where(pole, jnp.full_like(value, jnp.inf), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@_ellipkm1_array.defjvp
def _ellipkm1_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (p,) = primals
    (p_dot,) = tangents
    value = _ellipkm1_array(p)
    threshold = 1e-4 if p.dtype == jnp.float32 else 1e-8
    pole = _exact_zero(p)
    invalid = _signbit(p) & ~pole
    positive = ~_signbit(p) & ~pole
    use_series = positive & (p < threshold)
    near_one = jnp.abs(p - 1.0) < (1e-3 if p.dtype == jnp.float32 else 1e-7)
    series_p = jnp.where(positive, p, jnp.ones_like(p))
    formula_p = jnp.where(invalid | pole | use_series, jnp.full_like(p, 0.5), p)
    m = 1.0 - formula_p
    safe_m = jnp.where(near_one, jnp.ones_like(m), m)
    e = _ellipe_array(m)
    regular = -e / (2.0 * safe_m) / formula_p + value / (2.0 * safe_m)
    derivative = jnp.where(use_series, _km1_series_derivative(series_p), regular)
    m2 = m * m
    near_one_derivative = (
        -math.pi / 8.0
        - 9.0 * math.pi * m / 64.0
        - 75.0 * math.pi * m2 / 512.0
        - 1225.0 * math.pi * m2 * m / 8192.0
    )
    derivative = jnp.where(near_one, near_one_derivative, derivative)
    derivative = jnp.where(pole, jnp.full_like(derivative, -jnp.inf), derivative)
    derivative = jnp.where(invalid, jnp.full_like(derivative, jnp.nan), derivative)
    return value, derivative * p_dot


def ellipkm1(p: ArrayLike) -> Array:
    """Evaluate ``K(1 - p)`` accurately near the logarithmic singularity."""
    (promoted_p,) = promote_real("ellipkm1", p)
    return _ellipkm1_array(promoted_p)


def _use_negative_pi_transform(n: Array, m: Array) -> Array:
    return (n < -1.0) & (n < m)


def _negative_pi_elementary_primal(n: Array, phi: Array, m: Array) -> Array:
    transformed_n = m / n
    characteristic_scale = jnp.sqrt(1.0 - transformed_n)
    n_scale = jnp.sqrt(1.0 - n)
    sine = jnp.sin(phi)
    cosine = jnp.cos(phi)
    angle = jnp.arctan2(
        n_scale * characteristic_scale * sine,
        cosine * jnp.sqrt(1.0 - m * sine * sine),
    )
    return angle / n_scale / characteristic_scale


@jax.custom_jvp
def _negative_pi_elementary(n: Array, phi: Array, m: Array) -> Array:
    return _negative_pi_elementary_primal(n, phi, m)


@partial(_negative_pi_elementary.defjvp, symbolic_zeros=True)
def _negative_pi_elementary_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, Array],
) -> tuple[Array, Array]:
    n, phi, m = primals
    n_dot, phi_dot, m_dot = tangents
    value = _negative_pi_elementary(n, phi, m)
    zero = jnp.zeros_like(value)
    one = jnp.ones_like(value)
    _, derivative_n = jax.jvp(
        _negative_pi_elementary_primal,
        (n, phi, m),
        (one, zero, zero),
    )
    _, derivative_phi = jax.jvp(
        _negative_pi_elementary_primal,
        (n, phi, m),
        (zero, one, zero),
    )
    _, derivative_m = jax.jvp(
        _negative_pi_elementary_primal,
        (n, phi, m),
        (zero, zero, one),
    )
    tangent = zero
    if not isinstance(n_dot, jax.custom_derivatives.SymbolicZero):
        tangent = tangent + derivative_n * n_dot
    if not isinstance(phi_dot, jax.custom_derivatives.SymbolicZero):
        tangent = tangent + derivative_phi * phi_dot
    if not isinstance(m_dot, jax.custom_derivatives.SymbolicZero):
        tangent = tangent + derivative_m * m_dot
    return value, tangent


def _complete_pi_negative(n: Array, m: Array) -> Array:
    transformed_n = m / n
    elementary = 0.5 * math.pi * jnp.sqrt((-n) / (1.0 - n)) / jnp.sqrt(m - n)
    correction = (
        transformed_n
        * elliprj(
            jnp.zeros_like(m),
            1.0 - m,
            jnp.ones_like(m),
            1.0 - transformed_n,
        )
        / 3.0
    )
    return elementary - correction


def _complete_pi_primal(n: Array, m: Array) -> Array:
    invalid = (n >= 1.0) | (m > 1.0)
    safe_n = jnp.where(invalid, jnp.full_like(n, 0.25), n)
    safe_m = jnp.where(invalid, jnp.full_like(m, 0.5), m)
    transformed = _use_negative_pi_transform(safe_n, safe_m)

    direct_n = jnp.where(transformed, jnp.full_like(safe_n, 0.25), safe_n)
    direct_m = jnp.where(transformed, jnp.full_like(safe_m, 0.5), safe_m)
    direct = elliprf(jnp.zeros_like(m), 1.0 - direct_m, jnp.ones_like(m))
    direct = (
        direct
        + direct_n
        * elliprj(
            jnp.zeros_like(m),
            1.0 - direct_m,
            jnp.ones_like(m),
            1.0 - direct_n,
        )
        / 3.0
    )

    negative_n = jnp.where(transformed, safe_n, jnp.full_like(safe_n, -2.0))
    negative_m = jnp.where(transformed, safe_m, jnp.zeros_like(safe_m))
    value = jnp.where(transformed, _complete_pi_negative(negative_n, negative_m), direct)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@jax.custom_jvp
def _complete_pi_array(n: Array, m: Array) -> Array:
    return _complete_pi_primal(n, m)


@_complete_pi_array.defjvp
def _complete_pi_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    n, m = primals
    n_dot, m_dot = tangents
    value = _complete_pi_array(n, m)
    invalid = (n >= 1.0) | (m >= 1.0)
    transformed = _use_negative_pi_transform(n, m) & ~invalid
    threshold = 1e-3 if m.dtype == jnp.float32 else 1e-7
    near_zero_n = jnp.abs(n) < threshold
    near_line = jnp.abs(n - m) < threshold

    degenerate = invalid | near_zero_n | near_line | transformed
    formula_n = jnp.where(degenerate, jnp.full_like(n, 0.25), n)
    formula_m = jnp.where(degenerate, jnp.full_like(m, 0.5), m)
    formula_value = _complete_pi_array(formula_n, formula_m)
    formula_k = _ellipk_array(formula_m)
    formula_e = _ellipe_array(formula_m)
    derivative_n = (
        formula_n * formula_e
        + (formula_m - formula_n) * formula_k
        + (formula_n * formula_n - formula_m) * formula_value
    ) / (2.0 * (formula_m - formula_n) * (formula_n - 1.0) * formula_n)
    derivative_m = (formula_value - formula_e / (1.0 - formula_m)) / (
        2.0 * (formula_n - formula_m)
    )

    evaluation_m = jnp.where(invalid, jnp.full_like(m, 0.5), m)
    k = _ellipk_array(evaluation_m)
    e = _ellipe_array(evaluation_m)
    near_zero_m = jnp.abs(evaluation_m) < threshold
    safe_m = jnp.where(near_zero_m, jnp.full_like(evaluation_m, 0.5), evaluation_m)
    e_m = (e - k) / (2.0 * safe_m)
    k_m = e / (2.0 * safe_m * (1.0 - evaluation_m)) - k / (2.0 * safe_m)
    a1 = (k - e) / safe_m
    a1_m = (k_m - e_m) / safe_m - (k - e) / (safe_m * safe_m)
    a2 = ((2.0 + evaluation_m) * k - 2.0 * (1.0 + evaluation_m) * e) / (
        3.0 * safe_m * safe_m
    )
    a2_numerator = (2.0 + evaluation_m) * k - 2.0 * (1.0 + evaluation_m) * e
    a2_numerator_m = (
        k + (2.0 + evaluation_m) * k_m - 2.0 * e - 2.0 * (1.0 + evaluation_m) * e_m
    )
    a2_m = a2_numerator_m / (3.0 * safe_m * safe_m) - 2.0 * a2_numerator / (
        3.0 * safe_m * safe_m * safe_m
    )

    pi = math.pi
    m2 = evaluation_m * evaluation_m
    e_m_series = -pi / 8.0 - 3.0 * pi * evaluation_m / 64.0 - 15.0 * pi * m2 / 512.0
    k_m_series = pi / 8.0 + 9.0 * pi * evaluation_m / 64.0 + 75.0 * pi * m2 / 512.0
    a1_series = pi / 4.0 + 3.0 * pi * evaluation_m / 32.0 + 15.0 * pi * m2 / 256.0
    a1_m_series = 3.0 * pi / 32.0 + 15.0 * pi * evaluation_m / 128.0
    a2_series = (
        3.0 * pi / 16.0 + 5.0 * pi * evaluation_m / 64.0 + 105.0 * pi * m2 / 2048.0
    )
    a2_m_series = 5.0 * pi / 64.0 + 105.0 * pi * evaluation_m / 1024.0
    e_m = jnp.where(near_zero_m, e_m_series, e_m)
    k_m = jnp.where(near_zero_m, k_m_series, k_m)
    a1 = jnp.where(near_zero_m, a1_series, a1)
    a1_m = jnp.where(near_zero_m, a1_m_series, a1_m)
    a2 = jnp.where(near_zero_m, a2_series, a2)
    a2_m = jnp.where(near_zero_m, a2_m_series, a2_m)

    e_mm = (e_m - k_m) / (2.0 * safe_m) - (e - k) / (2.0 * safe_m * safe_m)
    k_mm = (
        e_m / (2.0 * safe_m * (1.0 - evaluation_m))
        - e
        * (1.0 - 2.0 * evaluation_m)
        / (2.0 * safe_m * safe_m * (1.0 - evaluation_m) ** 2)
        - k_m / (2.0 * safe_m)
        + k / (2.0 * safe_m * safe_m)
    )
    e_mmm = (
        (e_mm - k_mm) / (2.0 * safe_m)
        - (e_m - k_m) / (safe_m * safe_m)
        + (e - k) / (safe_m * safe_m * safe_m)
    )
    line_derivative = (2.0 / 3.0) * (
        e_m / (1.0 - evaluation_m) + e / ((1.0 - evaluation_m) ** 2)
    )
    line_derivative_m = (2.0 / 3.0) * (
        e_mm / (1.0 - evaluation_m)
        + 2.0 * e_m / ((1.0 - evaluation_m) ** 2)
        + 2.0 * e / ((1.0 - evaluation_m) ** 3)
    )
    line_derivative_mm = (2.0 / 3.0) * (
        e_mmm / (1.0 - evaluation_m)
        + 3.0 * e_mm / ((1.0 - evaluation_m) ** 2)
        + 6.0 * e_m / ((1.0 - evaluation_m) ** 3)
        + 6.0 * e / ((1.0 - evaluation_m) ** 4)
    )
    line_series = pi / 4.0 + 15.0 * pi * evaluation_m / 32.0 + 175.0 * pi * m2 / 256.0
    line_m_series = 15.0 * pi / 32.0 + 175.0 * pi * evaluation_m / 128.0
    line_mm_series = 175.0 * pi / 128.0 + 11025.0 * pi * evaluation_m / 2048.0
    line_derivative = jnp.where(near_zero_m, line_series, line_derivative)
    line_derivative_m = jnp.where(near_zero_m, line_m_series, line_derivative_m)
    line_derivative_mm = jnp.where(near_zero_m, line_mm_series, line_derivative_mm)

    zero_n_derivative = a1 + 2.0 * n * a2
    zero_m_derivative = k_m + n * a1_m + n * n * a2_m
    delta = n - m
    line_n_derivative = line_derivative + 0.8 * delta * line_derivative_m
    line_m_derivative = (
        0.5 * line_derivative
        + 0.2 * delta * line_derivative_m
        + 0.4 * delta * delta * line_derivative_mm
    )
    derivative_n = jnp.where(near_zero_n, zero_n_derivative, derivative_n)
    derivative_m = jnp.where(near_zero_n, zero_m_derivative, derivative_m)
    derivative_n = jnp.where(near_line, line_n_derivative, derivative_n)
    derivative_m = jnp.where(near_line, line_m_derivative, derivative_m)
    derivative_n = jnp.where(invalid, jnp.full_like(derivative_n, jnp.nan), derivative_n)
    derivative_m = jnp.where(invalid, jnp.full_like(derivative_m, jnp.nan), derivative_m)
    direct_tangent = derivative_n * n_dot + derivative_m * m_dot
    negative_n = jnp.where(transformed, n, jnp.full_like(n, -2.0))
    negative_m = jnp.where(transformed, m, jnp.zeros_like(m))
    _, negative_tangent = jax.jvp(
        _complete_pi_negative,
        (negative_n, negative_m),
        (n_dot, m_dot),
    )
    return value, jnp.where(transformed, negative_tangent, direct_tangent)


def _reduce_amplitude(phi: Array) -> tuple[Array, Array]:
    period = jnp.floor((phi + 0.5 * math.pi) / math.pi)
    reduced = phi - period * math.pi
    return period, reduced


def ellipkinc(phi: ArrayLike, m: ArrayLike) -> Array:
    """Incomplete elliptic integral of the first kind ``F(phi | m)``."""
    phi, m = promote_real("ellipkinc", phi, m)
    phi, m = jnp.broadcast_arrays(phi, m)
    invalid = m > 1.0
    singular = (m == 1.0) & (jnp.abs(phi) >= 0.5 * math.pi)
    period, reduced = _reduce_amplitude(phi)
    sine = jnp.sin(reduced)
    cosine = jnp.cos(reduced)
    safe_m = jnp.where(invalid | singular, jnp.zeros_like(m), m)
    local = sine * elliprf(cosine * cosine, 1.0 - safe_m * sine * sine, jnp.ones_like(m))
    periodic = (period != 0.0) & ~invalid & ~singular
    periodic_m = jnp.where(periodic, safe_m, jnp.full_like(safe_m, 0.5))
    value = local + 2.0 * period * _ellipk_array(periodic_m)
    value = jnp.where(singular, jnp.copysign(jnp.full_like(value, jnp.inf), phi), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _ellipeinc_primal(phi: Array, m: Array) -> Array:
    invalid = m > 1.0
    period, reduced = _reduce_amplitude(phi)
    sine = jnp.sin(reduced)
    cosine = jnp.cos(reduced)
    safe_m = jnp.where(invalid, jnp.zeros_like(m), m)
    y = 1.0 - safe_m * sine * sine
    rf = elliprf(cosine * cosine, y, jnp.ones_like(m))
    rd = elliprd(cosine * cosine, y, jnp.ones_like(m))
    local = sine * rf - safe_m * sine * sine * sine * rd / 3.0
    periodic = (period != 0.0) & ~invalid
    periodic_m = jnp.where(periodic, safe_m, jnp.full_like(safe_m, 0.5))
    value = local + 2.0 * period * _ellipe_array(periodic_m)
    endpoint = m == 1.0
    endpoint_value = jnp.sin(reduced) + 2.0 * period
    value = jnp.where(endpoint, endpoint_value, value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@jax.custom_jvp
def _ellipeinc_array(phi: Array, m: Array) -> Array:
    return _ellipeinc_primal(phi, m)


@partial(_ellipeinc_array.defjvp, symbolic_zeros=True)
def _ellipeinc_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    phi, m = primals
    phi_dot, m_dot = tangents
    value = _ellipeinc_array(phi, m)
    invalid = m > 1.0
    near_zero = jnp.abs(m) < (1e-3 if m.dtype == jnp.float32 else 1e-7)
    safe_m = jnp.where(invalid | near_zero, jnp.full_like(m, 0.5), m)
    first = ellipkinc(phi, safe_m)
    parameter_derivative = (value - first) / (2.0 * safe_m)

    sine = jnp.sin(phi)
    cosine = jnp.cos(phi)
    fourth_moment = 3.0 * phi / 8.0 - jnp.sin(2.0 * phi) / 4.0 + jnp.sin(4.0 * phi) / 32.0
    zero_series = -0.25 * (phi - sine * cosine) - 0.25 * m * fourth_moment
    parameter_derivative = jnp.where(near_zero, zero_series, parameter_derivative)
    root = jnp.sqrt(jnp.maximum(0.0, 1.0 - m * sine * sine))
    zero_tangent = jnp.zeros_like(value)
    phi_tangent = (
        zero_tangent
        if isinstance(phi_dot, jax.custom_derivatives.SymbolicZero)
        else root * phi_dot
    )
    parameter_tangent = (
        zero_tangent
        if isinstance(m_dot, jax.custom_derivatives.SymbolicZero)
        else parameter_derivative * m_dot
    )
    tangent = phi_tangent + parameter_tangent
    tangent = jnp.where(invalid, jnp.full_like(tangent, jnp.nan), tangent)
    return value, tangent


def ellipeinc(phi: ArrayLike, m: ArrayLike) -> Array:
    """Incomplete elliptic integral of the second kind ``E(phi | m)``."""
    phi, m = promote_real("ellipeinc", phi, m)
    phi, m = jnp.broadcast_arrays(phi, m)
    return _ellipeinc_array(phi, m)


def ellippi(n: ArrayLike, m: ArrayLike) -> Array:
    """Complete elliptic integral of the third kind ``Pi(n | m)``."""
    n, m = promote_real("ellippi", n, m)
    n, m = jnp.broadcast_arrays(n, m)
    return _complete_pi_array(n, m)


def ellippiinc(n: ArrayLike, phi: ArrayLike, m: ArrayLike) -> Array:
    """Incomplete elliptic integral of the third kind ``Pi(n; phi | m)``."""
    n, phi, m = promote_real("ellippiinc", n, phi, m)
    n, phi, m = jnp.broadcast_arrays(n, phi, m)
    invalid = (n >= 1.0) | (m > 1.0)
    singular = (m == 1.0) & (jnp.abs(phi) >= 0.5 * math.pi)
    period, reduced = _reduce_amplitude(phi)
    sine = jnp.sin(reduced)
    cosine = jnp.cos(reduced)
    safe_n = jnp.where(invalid | singular, jnp.zeros_like(n), n)
    safe_m = jnp.where(invalid | singular, jnp.zeros_like(m), m)
    transformed = _use_negative_pi_transform(safe_n, safe_m)

    direct_n = jnp.where(transformed, jnp.zeros_like(safe_n), safe_n)
    x = cosine * cosine
    y = 1.0 - safe_m * sine * sine
    p = 1.0 - direct_n * sine * sine
    rf = elliprf(x, y, jnp.ones_like(m))
    rj = elliprj(x, y, jnp.ones_like(m), p)
    direct_local = sine * rf + direct_n * sine * sine * sine * rj / 3.0

    negative_n = jnp.where(transformed, safe_n, jnp.full_like(safe_n, -2.0))
    negative_m = jnp.where(transformed, safe_m, jnp.zeros_like(safe_m))
    transformed_n = negative_m / negative_n
    elementary = _negative_pi_elementary(negative_n, reduced, negative_m)
    transformed_rj = elliprj(
        x,
        1.0 - negative_m * sine * sine,
        jnp.ones_like(m),
        1.0 - transformed_n * sine * sine,
    )
    transformed_local = (
        elementary - transformed_n * sine * sine * sine * transformed_rj / 3.0
    )
    local = jnp.where(transformed, transformed_local, direct_local)
    periodic = (period != 0.0) & ~invalid & ~singular
    periodic_n = jnp.where(periodic, safe_n, jnp.full_like(safe_n, 0.25))
    periodic_m = jnp.where(periodic, safe_m, jnp.full_like(safe_m, 0.5))
    complete = _complete_pi_array(periodic_n, periodic_m)
    value = local + 2.0 * period * complete
    value = jnp.where(singular, jnp.copysign(jnp.full_like(value, jnp.inf), phi), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


__all__ = [
    "ellipe",
    "ellipeinc",
    "ellipk",
    "ellipkinc",
    "ellipkm1",
    "ellippi",
    "ellippiinc",
]
