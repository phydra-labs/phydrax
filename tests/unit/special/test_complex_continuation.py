#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

import phydrax as phx


def test_airy_complex_continuation_satisfies_ode_and_preserves_dtype():
    z = jnp.asarray(0.4 + 0.7j, dtype=jnp.complex64)
    ai, aip, bi, bip = phx.special.airy(z)
    second_ai = jax.jvp(
        lambda value: phx.special.airy(value)[1], (z,), (jnp.ones_like(z),)
    )[1]
    second_bi = jax.jvp(
        lambda value: phx.special.airy(value)[3], (z,), (jnp.ones_like(z),)
    )[1]
    assert ai.dtype == z.dtype
    assert jnp.allclose(second_ai, z * ai, rtol=2e-4, atol=2e-5)
    assert jnp.allclose(second_bi, z * bi, rtol=2e-4, atol=2e-5)
    assert jnp.allclose(ai * bip - aip * bi, 1.0 / jnp.pi, rtol=2e-4)


def test_bessel_complex_wronskian_recurrence_and_order_derivatives():
    order = jnp.asarray(0.7)
    z = jnp.asarray(1.2 + 0.4j)
    jv = phx.special.jv(order, z)
    yv = phx.special.yv(order, z)
    jvp1 = phx.special.jv(order + 1.0, z)
    jvm1 = phx.special.jv(order - 1.0, z)
    assert jnp.allclose(jvm1 + jvp1, 2.0 * order * jv / z, rtol=2e-4, atol=2e-5)
    derivative = phx.special.jv_order_derivative(order, z)
    step = 1e-3
    finite_difference = (
        phx.special.jv(order + step, z) - phx.special.jv(order - step, z)
    ) / (2.0 * step)
    assert jnp.allclose(derivative, finite_difference, rtol=2e-3, atol=2e-4)
    assert jnp.allclose(phx.special.hankel1(order, z), jv + 1j * yv)
    assert jnp.all(jnp.isfinite(phx.special.yv_order_derivative(1.0, z)))
    _, real_order_tangent = jax.jvp(
        phx.special.jv,
        (jnp.asarray(0.7), jnp.asarray(1.2)),
        (jnp.asarray(1.0), jnp.asarray(0.0)),
    )
    assert jnp.allclose(
        real_order_tangent,
        jnp.real(phx.special.jv_order_derivative(0.7, 1.2)),
        rtol=2e-4,
    )


def test_negative_integer_complex_jv_uses_reflection_limit_and_finite_derivative():
    orders = np.asarray([-1.0, -2.0, -3.0])
    argument = 1.2 + 0.4j
    actual = np.asarray(phx.special.jv(jnp.asarray(orders), argument))
    reflected = np.asarray([-1.0, 1.0, -1.0]) * np.asarray(
        phx.special.jv(jnp.asarray(-orders), argument)
    )
    np.testing.assert_allclose(actual, reflected, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        actual,
        scipy.special.jv(orders, argument),
        rtol=2e-12,
        atol=2e-13,
    )

    derivative = np.asarray(
        phx.special.jv_order_derivative(jnp.asarray(orders), argument)
    )
    step = np.cbrt(np.finfo(np.float64).eps) * (1.0 + np.abs(orders))
    reference = (
        scipy.special.jv(orders + step, argument)
        - scipy.special.jv(orders - step, argument)
    ) / (2.0 * step)
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, reference, rtol=2e-7, atol=2e-9)


def test_modified_bessel_complex_connection_and_integer_order_limit():
    z = jnp.asarray(1.1 + 0.2j)
    order = jnp.asarray(0.4)
    iv = phx.special.iv(order, z)
    kv = phx.special.kv(order, z)
    expected = 0.5 * jnp.pi * (phx.special.iv(-order, z) - iv) / jnp.sin(jnp.pi * order)
    assert jnp.allclose(kv, expected, rtol=2e-4, atol=2e-5)
    assert jnp.all(jnp.isfinite(phx.special.kv(2.0, z)))
    assert jnp.all(jnp.isfinite(phx.special.kv_order_derivative(2.0, z)))
    assert jnp.allclose(phx.special.ive(order, z), jnp.exp(-jnp.abs(jnp.real(z))) * iv)
    assert jnp.allclose(phx.special.kve(order, z), jnp.exp(z) * kv)


def test_carlson_legendre_jacobi_and_dawson_complex_identities():
    value = jnp.asarray(0.8 + 0.3j)
    assert jnp.allclose(
        phx.special.elliprf(value, value, value), 1.0 / jnp.sqrt(value), rtol=2e-5
    )
    parameter = jnp.asarray(0.2 + 0.1j)
    expected_e = (
        phx.special.elliprf(0.0j, 1.0 - parameter, 1.0 + 0.0j)
        - parameter * phx.special.elliprd(0.0j, 1.0 - parameter, 1.0 + 0.0j) / 3.0
    )
    assert jnp.allclose(phx.special.ellipe(parameter), expected_e, rtol=2e-5)
    sn, cn, dn, amplitude = phx.special.ellipj(0.3 + 0.2j, parameter)
    assert jnp.allclose(sn * sn + cn * cn, 1.0, rtol=2e-4, atol=2e-5)
    assert jnp.allclose(dn * dn + parameter * sn * sn, 1.0, rtol=2e-4, atol=2e-5)
    assert jnp.allclose(jnp.sin(amplitude), sn, rtol=2e-4)
    z = jnp.asarray(0.2 + 0.1j)
    dawson = phx.special.dawsn(z)
    derivative = jax.jvp(phx.special.dawsn, (z,), (jnp.ones_like(z),))[1]
    assert jnp.allclose(derivative, 1.0 - 2.0 * z * dawson, rtol=2e-4, atol=2e-5)


def test_principal_log_signed_zero_lips_are_explicit():
    upper = phx.special.principal_log(jnp.asarray(complex(-1.0, 0.0)))
    lower = phx.special.principal_log(jnp.asarray(complex(-1.0, -0.0)))
    assert jnp.imag(upper) > 0.0
    assert jnp.imag(lower) < 0.0
    assert jnp.allclose(
        phx.special.principal_sqrt(jnp.asarray(3.0 + 4.0j)) ** 2, 3.0 + 4.0j
    )
