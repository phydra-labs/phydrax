#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real Jacobi elliptic functions and amplitude."""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._dtype import promote_real
from ._legendre import ellipeinc


def _steps(dtype: jnp.dtype) -> int:
    return 7 if dtype == jnp.float32 else 11


def _sech(x: Array) -> Array:
    exponential = jnp.exp(-jnp.abs(x))
    return 2.0 * exponential / (1.0 + exponential * exponential)


def _agm_jacobi(
    u: Array, complementary_parameter: Array
) -> tuple[Array, Array, Array, Array]:
    count = _steps(u.dtype)
    a = jnp.ones_like(u)
    b = jnp.sqrt(complementary_parameter)

    def ascend(
        state: tuple[Array, Array], _: None
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        current_a, current_b = state
        c = 0.5 * (current_a - current_b)
        next_a = 0.5 * (current_a + current_b)
        next_b = jnp.sqrt(current_a * current_b)
        return (next_a, next_b), (c, next_a)

    (a, _), history = jax.lax.scan(ascend, (a, b), None, length=count)
    amplitude = jnp.ldexp(a * u, count)

    def descend(phi: Array, data: tuple[Array, Array]) -> tuple[Array, None]:
        c, next_a = data
        argument = c * jnp.sin(phi) / next_a
        next_phi = 0.5 * (phi + jnp.arcsin(jnp.clip(argument, -1.0, 1.0)))
        return next_phi, None

    amplitude, _ = jax.lax.scan(descend, amplitude, history, reverse=True)
    sn = jnp.sin(amplitude)
    cn = jnp.cos(amplitude)
    dn = jnp.sqrt(
        jnp.maximum(
            0.0,
            cn * cn + complementary_parameter * sn * sn,
        )
    )
    return sn, cn, dn, amplitude


def _jacobi_primal(u: Array, m: Array) -> tuple[Array, Array, Array, Array]:
    invalid = m > 1.0
    endpoint_zero = m == 0.0
    endpoint_one = m == 1.0
    negative = m < 0.0

    safe_m = jnp.where(invalid | endpoint_one, jnp.zeros_like(m), m)
    complementary_parameter = jnp.where(negative, 1.0 / (1.0 - safe_m), 1.0 - safe_m)
    scale = jnp.where(negative, jnp.sqrt(1.0 - safe_m), jnp.ones_like(m))
    base_sn, base_cn, base_dn, base_amplitude = _agm_jacobi(
        scale * u, complementary_parameter
    )

    sn = jnp.where(negative, base_sn / (scale * base_dn), base_sn)
    cn = jnp.where(negative, base_cn / base_dn, base_cn)
    dn = jnp.where(negative, 1.0 / base_dn, base_dn)

    transformed_principal = jnp.arctan2(base_sn, scale * base_cn)
    cycles = jnp.floor((base_amplitude + math.pi) / (2.0 * math.pi))
    amplitude = jnp.where(
        negative, transformed_principal + cycles * (2.0 * math.pi), base_amplitude
    )

    sn = jnp.where(endpoint_zero, jnp.sin(u), sn)
    cn = jnp.where(endpoint_zero, jnp.cos(u), cn)
    dn = jnp.where(endpoint_zero, jnp.ones_like(u), dn)
    amplitude = jnp.where(endpoint_zero, u, amplitude)

    hyperbolic_sn = jnp.tanh(u)
    hyperbolic_cn = _sech(u)
    sn = jnp.where(endpoint_one, hyperbolic_sn, sn)
    cn = jnp.where(endpoint_one, hyperbolic_cn, cn)
    dn = jnp.where(endpoint_one, hyperbolic_cn, dn)
    amplitude = jnp.where(
        endpoint_one, jnp.arctan2(hyperbolic_sn, hyperbolic_cn), amplitude
    )

    nan = jnp.full_like(u, jnp.nan)
    return (
        jnp.where(invalid, nan, sn),
        jnp.where(invalid, nan, cn),
        jnp.where(invalid, nan, dn),
        jnp.where(invalid, nan, amplitude),
    )


@jax.custom_jvp
def _ellipj_array(u: Array, m: Array) -> tuple[Array, Array, Array, Array]:
    return _jacobi_primal(u, m)


@partial(_ellipj_array.defjvp, symbolic_zeros=True)
def _ellipj_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[tuple[Array, Array, Array, Array], tuple[Array, Array, Array, Array]]:
    u, m = primals
    u_dot, m_dot = tangents
    sn, cn, dn, amplitude = _ellipj_array(u, m)

    threshold = 1e-3 if m.dtype == jnp.float32 else 1e-7
    near_zero = jnp.abs(m) < threshold
    complement = 1.0 - m
    absolute_u = jnp.abs(u)
    log_cosh = absolute_u + jnp.log1p(jnp.exp(-2.0 * absolute_u)) - math.log(2.0)
    positive_complement = complement > 0.0
    safe_complement = jnp.where(
        positive_complement, complement, jnp.ones_like(complement)
    )
    endpoint_series = (m == 1.0) | (
        positive_complement
        & (jnp.log(safe_complement) + 2.0 * log_cosh < math.log(threshold))
    )
    safe_m = jnp.where(near_zero | endpoint_series | (m > 1.0), jnp.full_like(m, 0.5), m)
    e = ellipeinc(amplitude, safe_m)
    f_m = (
        e / (2.0 * safe_m * (1.0 - safe_m))
        - u / (2.0 * safe_m)
        - sn * cn / (2.0 * (1.0 - safe_m) * dn)
    )
    amplitude_m = -dn * f_m
    sine_u = jnp.sin(u)
    cosine_u = jnp.cos(u)
    zero_integral = 0.25 * (u - sine_u * cosine_u)
    zero_second_integral = (
        9.0 * u / 64.0 - 3.0 * jnp.sin(2.0 * u) / 32.0 + 3.0 * jnp.sin(4.0 * u) / 256.0
    )
    zero_quadratic = 0.5 * zero_integral * sine_u * sine_u - zero_second_integral
    zero_series = -zero_integral + 2.0 * m * zero_quadratic
    amplitude_m = jnp.where(near_zero, zero_series, amplitude_m)

    series_u = jnp.where(endpoint_series, u, jnp.zeros_like(u))
    hyperbolic_sn = jnp.tanh(series_u)
    hyperbolic_cn = _sech(series_u)
    sinh_u = jnp.sinh(series_u)

    # This is the endpoint expansion
    #   am_m = -(sinh(u) - u sech(u))/4 - 2 (1-m) Q(u),
    # with the terms combined so an exact endpoint never forms 0 * Q when
    # Q has overflowed but the first derivative is itself unrepresentable.
    one_series = sinh_u * (
        -0.25 - complement * (9.0 - 4.0 * series_u * hyperbolic_sn) / 32.0
    ) + hyperbolic_cn * (
        0.25 * series_u
        + complement * (9.0 * series_u + 2.0 * series_u * series_u * hyperbolic_sn) / 32.0
    )
    amplitude_m = jnp.where(endpoint_series, one_series, amplitude_m)
    endpoint_fallback = (m == 1.0) & ~jnp.isfinite(amplitude_m)
    formula_amplitude_m = jnp.where(
        endpoint_fallback, jnp.zeros_like(amplitude_m), amplitude_m
    )
    generic_sn_m = cn * formula_amplitude_m
    generic_cn_m = -sn * formula_amplitude_m
    safe_dn = jnp.where(dn == 0.0, jnp.ones_like(dn), dn)
    generic_dn_m = -(sn * sn + 2.0 * m * sn * generic_sn_m) / (2.0 * safe_dn)

    endpoint_sn_m = -0.25 * (hyperbolic_sn - series_u * hyperbolic_cn * hyperbolic_cn)
    endpoint_cn_m = 0.25 * hyperbolic_sn * (sinh_u - series_u * hyperbolic_cn)
    endpoint_dn_m = -0.25 * hyperbolic_sn * (sinh_u + series_u * hyperbolic_cn)
    sn_m = jnp.where(endpoint_fallback, endpoint_sn_m, generic_sn_m)
    cn_m = jnp.where(endpoint_fallback, endpoint_cn_m, generic_cn_m)
    dn_m = jnp.where(endpoint_fallback, endpoint_dn_m, generic_dn_m)

    invalid = m > 1.0
    nan = jnp.full_like(u, jnp.nan)
    zero_tangent = jnp.zeros_like(u)
    if isinstance(u_dot, jax.custom_derivatives.SymbolicZero):
        argument_tangents = (zero_tangent,) * 4
    else:
        argument_tangents = (
            cn * dn * u_dot,
            -sn * dn * u_dot,
            -m * sn * cn * u_dot,
            dn * u_dot,
        )
    if isinstance(m_dot, jax.custom_derivatives.SymbolicZero):
        parameter_tangents = (zero_tangent,) * 4
    else:
        parameter_tangents = (
            sn_m * m_dot,
            cn_m * m_dot,
            dn_m * m_dot,
            amplitude_m * m_dot,
        )
    sn_tangent = argument_tangents[0] + parameter_tangents[0]
    cn_tangent = argument_tangents[1] + parameter_tangents[1]
    dn_tangent = argument_tangents[2] + parameter_tangents[2]
    amplitude_tangent = argument_tangents[3] + parameter_tangents[3]
    tangents_out = (
        jnp.where(invalid, nan, sn_tangent),
        jnp.where(invalid, nan, cn_tangent),
        jnp.where(invalid, nan, dn_tangent),
        jnp.where(invalid, nan, amplitude_tangent),
    )
    return (sn, cn, dn, amplitude), tangents_out


def ellipj(u: ArrayLike, m: ArrayLike) -> tuple[Array, Array, Array, Array]:
    """Return ``sn(u|m)``, ``cn(u|m)``, ``dn(u|m)``, and ``am(u|m)``."""
    u, m = promote_real("ellipj", u, m)
    u, m = jnp.broadcast_arrays(u, m)
    return _ellipj_array(u, m)


def ellipam(u: ArrayLike, m: ArrayLike) -> Array:
    """Return the unwrapped Jacobi amplitude ``am(u | m)``."""
    return ellipj(u, m)[3]


__all__ = ["ellipam", "ellipj"]
