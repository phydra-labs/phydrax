#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import scipy.special

import phydrax as phx


def _right_half_plane_grid() -> np.ndarray:
    """``|z|`` in ``[1e-3, 200]`` and ``arg z`` in ``[-pi/2, pi/2]`` with exact axes."""
    radii = np.geomspace(1e-3, 200.0, 15)
    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 9)
    grid = radii[:, None] * np.exp(1j * angles[None, :])
    grid[:, 0] = -1j * radii
    grid[:, -1] = 1j * radii
    return grid.ravel()


def test_complex_continuation_scenario_1() -> None:
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


def test_complex_continuation_scenario_2() -> None:
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
    upper = phx.special.principal_log(jnp.asarray(complex(-1.0, 0.0)))
    lower = phx.special.principal_log(jnp.asarray(complex(-1.0, -0.0)))
    assert jnp.imag(upper) > 0.0
    assert jnp.imag(lower) < 0.0
    assert jnp.allclose(
        phx.special.principal_sqrt(jnp.asarray(3.0 + 4.0j)) ** 2, 3.0 + 4.0j
    )


@pytest.mark.parametrize(
    "order",
    [0.0, 1.0, 2.0, 0.3, 1.7, -5.0 / 3.0],
    ids=["zero", "one", "two", "three-tenths", "one-point-seven", "minus-five-thirds"],
)
def test_complex_kv_and_kve_match_mpmath_on_closed_right_half_plane(
    order: float,
) -> None:
    # Independent 30-digit mpmath reference; K_{-v} = K_v covers negative orders.
    z = _right_half_plane_grid()
    with mp.workdps(30):
        reference_kv = np.asarray(
            [complex(mp.besselk(order, mp.mpc(value.real, value.imag))) for value in z]
        )
        reference_kve = np.asarray(
            [
                complex(
                    mp.besselk(order, mp.mpc(value.real, value.imag))
                    * mp.exp(mp.mpc(value.real, value.imag))
                )
                for value in z
            ]
        )
    argument = jnp.asarray(z, dtype=jnp.complex128)
    orders = jnp.full(z.shape, order, dtype=jnp.complex128)
    kv = np.asarray(phx.special.kv(orders, argument))
    kve = np.asarray(phx.special.kve(orders, argument))
    assert kv.dtype == np.complex128
    assert kve.dtype == np.complex128
    np.testing.assert_allclose(kve, reference_kve, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(kv, reference_kv, rtol=1e-12, atol=0.0)


def test_complex_kv_on_imaginary_axis_matches_hankel_functions() -> None:
    # K_0(-ix) = (i pi/2) H_0^(1)(x) and K_1(-ix) = -(pi/2) H_1^(1)(x) for x > 0.
    x = np.geomspace(0.01, 150.0, 200)
    argument = jnp.asarray(-1j * x, dtype=jnp.complex128)
    k0 = np.asarray(phx.special.kv(jnp.asarray(0.0, dtype=jnp.complex128), argument))
    k1 = np.asarray(phx.special.kv(jnp.asarray(1.0, dtype=jnp.complex128), argument))
    np.testing.assert_allclose(
        k0, 0.5j * np.pi * scipy.special.hankel1(0.0, x), rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        k1, -0.5 * np.pi * scipy.special.hankel1(1.0, x), rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize(
    "order",
    [0.0, 1.0, 0.3, 5.0 / 3.0],
    ids=["zero", "one", "three-tenths", "five-thirds"],
)
def test_complex_kv_argument_jvp_matches_bessel_derivative(order: float) -> None:
    # K_v'(z) = -(K_{v-1}(z) + K_{v+1}(z)) / 2, referenced by scipy's AMOS kvp;
    # d kve / dz = kve + e^z K_v'.
    z = _right_half_plane_grid()
    argument = jnp.asarray(z, dtype=jnp.complex128)
    orders = jnp.full(z.shape, order, dtype=jnp.complex128)
    tangent = jnp.ones_like(argument)
    _, kv_tangent = jax.jvp(
        lambda value: phx.special.kv(orders, value), (argument,), (tangent,)
    )
    kve, kve_tangent = jax.jvp(
        lambda value: phx.special.kve(orders, value), (argument,), (tangent,)
    )
    derivative = scipy.special.kvp(order, z)
    np.testing.assert_allclose(np.asarray(kv_tangent), derivative, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(kve_tangent),
        np.asarray(kve) + np.exp(z) * derivative,
        rtol=1e-10,
        atol=1e-300,
    )
