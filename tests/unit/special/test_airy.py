import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import scipy.special

import phydrax as phx


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float32, 2e-4, 3e-6),
        (jnp.float64, 2e-11, 3e-13),
    ],
)
def test_airy_and_scaled_airy_match_scipy(dtype, rtol, atol):
    values = np.concatenate(
        [
            np.linspace(-100.0, -8.0, 100),
            np.linspace(-8.0, 5.0, 180),
            np.geomspace(5.01, 100.0, 80),
        ]
    ).astype(np.dtype(dtype))
    actual = np.stack(
        [np.asarray(item) for item in phx.special.airy(jnp.asarray(values))]
    )
    expected = np.stack(scipy.special.airy(values))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    positive = values >= 0.0
    scaled = np.stack(
        [np.asarray(item) for item in phx.special.airye(jnp.asarray(values))]
    )
    expected_scaled = np.stack(scipy.special.airye(values[positive]))
    np.testing.assert_allclose(scaled[:, positive], expected_scaled, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(scaled[:, ~positive], actual[:, ~positive])


def test_airy_large_negative_phase_preserves_quarter_pi_rotation():
    argument = -1e12
    values = phx.special.airy(jnp.asarray(argument))

    with mp.workdps(80):
        x = mp.mpf("-1e12")
        expected = [mp.airyai(x), mp.airyai(x, 1), mp.airybi(x), mp.airybi(x, 1)]
    np.testing.assert_allclose(
        [float(value) for value in values],
        [float(value) for value in expected],
        rtol=3e-13,
    )

    derivatives = [
        jax.grad(lambda value: phx.special.airy(value)[component])(jnp.asarray(argument))
        for component in range(4)
    ]
    expected_derivatives = (
        values[1],
        argument * values[0],
        values[3],
        argument * values[2],
    )
    np.testing.assert_allclose(derivatives, expected_derivatives, rtol=3e-13)


def test_airy_scaling_and_wronskian_identities():
    positive = jnp.geomspace(1e-8, 100.0, 100)
    ai, aip, bi, bip = phx.special.airy(positive)
    aie, aipe, bie, bipe = phx.special.airye(positive)
    zeta = 2.0 * positive * jnp.sqrt(positive) / 3.0
    np.testing.assert_allclose(
        np.asarray(aie), np.asarray(ai * jnp.exp(zeta)), rtol=3e-13
    )
    np.testing.assert_allclose(
        np.asarray(aipe), np.asarray(aip * jnp.exp(zeta)), rtol=3e-13
    )
    np.testing.assert_allclose(
        np.asarray(bie), np.asarray(bi * jnp.exp(-zeta)), rtol=3e-13
    )
    np.testing.assert_allclose(
        np.asarray(bipe), np.asarray(bip * jnp.exp(-zeta)), rtol=3e-13
    )

    values = jnp.linspace(-50.0, 20.0, 200)
    ai, aip, bi, bip = phx.special.airy(values)
    wronskian = ai * bip - aip * bi
    np.testing.assert_allclose(
        np.asarray(wronskian), 1.0 / math.pi, rtol=2e-11, atol=2e-13
    )


def test_airy_forward_reverse_and_second_derivatives_obey_ode():
    values = jnp.asarray([-20.0, -5.0, -0.2, 0.0, 2.0, 8.0, 40.0])
    for component in range(4):
        function = lambda value: phx.special.airy(value)[component]
        forward = jax.vmap(jax.jacfwd(function))(values)
        reverse = jax.vmap(jax.jacrev(function))(values)
        np.testing.assert_allclose(
            np.asarray(forward), np.asarray(reverse), rtol=3e-13, atol=3e-14
        )

    for component in (0, 2):
        function = lambda value: phx.special.airy(value)[component]
        second = jax.vmap(jax.grad(jax.grad(function)))(values)
        value = jax.vmap(function)(values)
        np.testing.assert_allclose(
            np.asarray(second), np.asarray(values * value), rtol=3e-11, atol=5e-13
        )


@pytest.mark.parametrize(
    ("dtype", "argument", "rtol"),
    [
        (jnp.float32, 1e5, 2e-5),
        (jnp.float64, 1e12, 2e-13),
    ],
)
def test_scaled_airy_extreme_derivatives_remain_representable(dtype, argument, rtol):
    derivatives = [
        float(
            jax.grad(lambda value: phx.special.airye(value)[component])(
                jnp.asarray(argument, dtype=dtype)
            )
        )
        for component in range(4)
    ]
    with mp.workdps(80):
        x = mp.mpf(str(argument))
        ai = mp.airyai(x)
        aip = mp.airyai(x, 1)
        bi = mp.airybi(x)
        bip = mp.airybi(x, 1)
        zeta = 2 * x ** mp.mpf("1.5") / 3
        expected = [
            mp.exp(zeta) * (aip + mp.sqrt(x) * ai),
            mp.exp(zeta) * (x * ai + mp.sqrt(x) * aip),
            mp.exp(-zeta) * (bip - mp.sqrt(x) * bi),
            mp.exp(-zeta) * (x * bi - mp.sqrt(x) * bip),
        ]
    np.testing.assert_allclose(
        derivatives,
        [float(value) for value in expected],
        rtol=rtol,
    )


def test_airy_boundary_and_dtype_contracts():
    ordinary = phx.special.airy(jnp.asarray([jnp.inf, -jnp.inf, jnp.nan]))
    assert ordinary[0].dtype == jnp.float64
    assert np.asarray(ordinary[0])[0] == 0.0
    assert np.asarray(ordinary[1])[0] == 0.0
    assert np.isposinf(np.asarray(ordinary[2])[0])
    assert np.isposinf(np.asarray(ordinary[3])[0])
    for value in ordinary:
        assert np.isnan(np.asarray(value)[1:]).all()

    scaled = phx.special.airye(jnp.asarray(jnp.inf))
    assert scaled[0] == 0.0
    assert np.isneginf(scaled[1])
    assert scaled[2] == 0.0
    assert np.isposinf(scaled[3])

    for component in (0, 1):
        assert jax.grad(lambda value: phx.special.airy(value)[component])(jnp.inf) == 0.0
        assert jax.grad(lambda value: phx.special.airye(value)[component])(jnp.inf) == 0.0

    for function in (phx.special.airy, phx.special.airye):
        assert function(jnp.asarray(0.5, dtype=jnp.float16))[0].dtype == jnp.float32
        with pytest.raises(TypeError, match="does not support complex-valued inputs"):
            function(0.5 + 0.2j)
