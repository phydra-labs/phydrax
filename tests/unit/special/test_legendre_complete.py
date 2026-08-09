import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import scipy.special

import phydrax as phx


def test_complete_legendre_integrals_match_scipy():
    parameters = np.concatenate(
        [
            -np.geomspace(1e-12, 1e200, 80),
            np.linspace(0.0, 0.99, 80),
            1.0 - np.geomspace(1e-15, 1e-2, 80),
        ]
    )
    for function, reference in [
        (phx.special.ellipk, scipy.special.ellipk),
        (phx.special.ellipe, scipy.special.ellipe),
    ]:
        actual = np.asarray(function(jnp.asarray(parameters)))
        expected = reference(parameters)
        np.testing.assert_allclose(actual, expected, rtol=8e-13, atol=2e-15)


def test_complete_third_kind_matches_high_precision_reference_and_broadcasts():
    characteristics = np.asarray([-3.0, -0.2, 0.0, 0.4, 0.9])[:, None]
    parameters = np.asarray([-1.0, 0.0, 0.3, 0.8, 0.99])[None, :]
    with mp.workdps(70):
        expected = np.asarray(
            [
                [float(mp.ellippi(n, m)) for m in parameters[0]]
                for n in characteristics[:, 0]
            ]
        )
    actual = np.asarray(
        phx.special.ellippi(jnp.asarray(characteristics), jnp.asarray(parameters))
    )
    assert actual.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=3e-14)


def test_complete_third_kind_large_negative_characteristic_is_factored():
    characteristics = np.asarray([-1e10, -1e100, -1e300])
    expected = math.pi / (2.0 * np.sqrt(1.0 - characteristics))
    actual = phx.special.ellippi(jnp.asarray(characteristics), 0.0)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-15)

    characteristic = -1e100
    expected_derivative = math.pi / (4.0 * (1.0 - characteristic) ** 1.5)
    function = lambda n: phx.special.ellippi(n, 0.0)
    np.testing.assert_allclose(
        jax.jacfwd(function)(characteristic),
        expected_derivative,
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        jax.jacrev(function)(characteristic),
        expected_derivative,
        rtol=3e-15,
    )


def test_complete_third_kind_derivatives_match_high_precision():
    characteristic = mp.mpf("0.2")
    parameter = mp.mpf("0.4")
    with mp.workdps(70):
        reference = lambda n, m: mp.ellippi(n, m)
        expected_gradient = np.asarray(
            [
                float(mp.diff(lambda n: reference(n, parameter), characteristic)),
                float(mp.diff(lambda m: reference(characteristic, m), parameter)),
            ]
        )
        expected_hessian = np.asarray(
            [
                [
                    float(
                        mp.diff(
                            lambda n: reference(n, parameter),
                            characteristic,
                            2,
                        )
                    ),
                    float(
                        mp.diff(
                            lambda n: mp.diff(lambda m: reference(n, m), parameter),
                            characteristic,
                        )
                    ),
                ],
                [
                    float(
                        mp.diff(
                            lambda m: mp.diff(lambda n: reference(n, m), characteristic),
                            parameter,
                        )
                    ),
                    float(
                        mp.diff(
                            lambda m: reference(characteristic, m),
                            parameter,
                            2,
                        )
                    ),
                ],
            ]
        )

    point = jnp.asarray([float(characteristic), float(parameter)])
    function = lambda arguments: phx.special.ellippi(arguments[0], arguments[1])
    np.testing.assert_allclose(
        np.asarray(jax.grad(function)(point)),
        expected_gradient,
        rtol=3e-13,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jacfwd(jax.jacrev(function))(point)),
        expected_hessian,
        rtol=3e-12,
        atol=3e-13,
    )


def test_ellipkm1_is_accurate_across_its_positive_domain():
    parameters = np.geomspace(1e-300, 1e200, 180)
    actual = np.asarray(phx.special.ellipkm1(jnp.asarray(parameters)))
    expected = scipy.special.ellipkm1(parameters)
    np.testing.assert_allclose(actual, expected, rtol=7e-13, atol=2e-15)
    np.testing.assert_allclose(phx.special.ellipkm1(1.0), math.pi / 2.0, rtol=2e-15)


def test_ellipkm1_preserves_smallest_positive_values_and_signed_zero_boundary():
    for dtype, value_tolerance, derivative_tolerance in (
        (np.float32, 3e-6, 3e-6),
        (np.float64, 3e-15, 3e-14),
    ):
        parameter = np.nextafter(dtype(0.0), dtype(1.0), dtype=dtype)
        expected = math.log(4.0) - 0.5 * math.log(float(parameter))
        actual = phx.special.ellipkm1(jnp.asarray(parameter))
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, expected, rtol=value_tolerance)

        largest_subnormal = np.nextafter(np.finfo(dtype).tiny, dtype(0.0), dtype=dtype)
        expected_derivative = -0.5 / float(largest_subnormal)
        argument = jnp.asarray(largest_subnormal)
        np.testing.assert_allclose(
            jax.jacfwd(phx.special.ellipkm1)(argument),
            expected_derivative,
            rtol=derivative_tolerance,
        )
        np.testing.assert_allclose(
            jax.jacrev(phx.special.ellipkm1)(argument),
            expected_derivative,
            rtol=derivative_tolerance,
        )

    assert np.isposinf(phx.special.ellipkm1(-0.0))


def test_ellipkm1_derivatives_cover_singular_and_transformed_regimes():
    parameters = np.asarray([1e-200, 1e-12, 0.2, 1.0, 10.0, 1e100, 1e200])
    with mp.workdps(250):
        expected = []
        for parameter in parameters:
            p = mp.mpf(str(parameter))
            if p == 1:
                expected.append(float(-mp.pi / 8))
                continue
            m = 1 - p
            derivative = -(mp.ellipe(m) / (2 * m * p) - mp.ellipk(m) / (2 * m))
            expected.append(float(derivative))

    arguments = jnp.asarray(parameters)
    forward = jax.vmap(jax.jacfwd(phx.special.ellipkm1))(arguments)
    reverse = jax.vmap(jax.jacrev(phx.special.ellipkm1))(arguments)
    np.testing.assert_allclose(np.asarray(forward), expected, rtol=2e-12, atol=1e-300)
    np.testing.assert_allclose(np.asarray(reverse), expected, rtol=2e-12, atol=1e-300)
    np.testing.assert_allclose(
        jax.grad(phx.special.ellipk)(-1e200),
        -expected[-1],
        rtol=2e-12,
        atol=1e-300,
    )


def test_complete_legendre_values_and_derivatives_obey_identities():
    parameters = jnp.asarray([-10.0, -0.5, 0.0, 0.2, 0.8, 1.0 - 1e-10])
    k = phx.special.ellipk(parameters)
    e = phx.special.ellipe(parameters)
    k_derivative = jax.vmap(jax.grad(phx.special.ellipk))(parameters)
    e_derivative = jax.vmap(jax.grad(phx.special.ellipe))(parameters)

    parameters_np = np.asarray(parameters)
    safe = parameters_np != 0.0
    expected_k = np.asarray(e)[safe] / (
        2.0 * parameters_np[safe] * (1.0 - parameters_np[safe])
    ) - np.asarray(k)[safe] / (2.0 * parameters_np[safe])
    expected_e = (np.asarray(e)[safe] - np.asarray(k)[safe]) / (2.0 * parameters_np[safe])
    np.testing.assert_allclose(
        np.asarray(k_derivative)[safe], expected_k, rtol=3e-12, atol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(e_derivative)[safe], expected_e, rtol=3e-12, atol=2e-14
    )
    np.testing.assert_allclose(k_derivative[2], math.pi / 8.0, rtol=2e-15)
    np.testing.assert_allclose(e_derivative[2], -math.pi / 8.0, rtol=2e-15)

    point = jnp.asarray(0.4)
    np.testing.assert_allclose(
        jax.jacfwd(phx.special.ellipk)(point),
        jax.jacrev(phx.special.ellipk)(point),
        rtol=2e-15,
    )
    assert np.isfinite(jax.grad(jax.grad(phx.special.ellipk))(point))


def test_complete_legendre_degenerate_hessians_are_analytic():
    np.testing.assert_allclose(
        jax.grad(jax.grad(phx.special.ellipk))(0.0), 9.0 * math.pi / 64.0
    )
    np.testing.assert_allclose(
        jax.grad(jax.grad(phx.special.ellipkm1))(1.0), 9.0 * math.pi / 64.0
    )

    function = lambda arguments: phx.special.ellippi(arguments[0], arguments[1])
    for point in (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 0.4]),
        jnp.asarray([0.4, 0.0]),
    ):
        forward = jax.jacfwd(jax.jacrev(function))(point)
        reverse = jax.jacrev(jax.jacfwd(function))(point)
        assert np.isfinite(np.asarray(forward)).all()
        np.testing.assert_allclose(
            np.asarray(forward), np.asarray(reverse), rtol=3e-12, atol=3e-13
        )


def test_complete_legendre_boundaries_and_invalid_lanes():
    k = np.asarray(phx.special.ellipk(jnp.asarray([0.0, 1.0, 2.0, np.nan])))
    e = np.asarray(phx.special.ellipe(jnp.asarray([0.0, 1.0, 2.0, np.nan])))
    np.testing.assert_allclose(k[0], math.pi / 2.0)
    assert np.isposinf(k[1])
    assert np.isnan(k[2:]).all()
    np.testing.assert_allclose(e[:2], [math.pi / 2.0, 1.0])
    assert np.isnan(e[2:]).all()

    third = np.asarray(
        phx.special.ellippi(
            jnp.asarray([0.0, 1.0, 0.2, 0.2, np.nan]),
            jnp.asarray([0.0, 0.5, 1.0, 1.1, 0.5]),
        )
    )
    np.testing.assert_allclose(third[0], math.pi / 2.0)
    assert np.isnan(third[1])
    assert np.isposinf(third[2])
    assert np.isnan(third[3:]).all()

    km1 = np.asarray(phx.special.ellipkm1(jnp.asarray([0.0, -1.0, 1.0])))
    assert np.isposinf(km1[0])
    assert np.isnan(km1[1])
    np.testing.assert_allclose(km1[2], math.pi / 2.0)
