import itertools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

import phydrax as phx


def test_carlson_forms_match_scipy_across_scales():
    x = np.asarray([0.0, 1e-100, 0.2, 1.0, 1e100])
    y = np.asarray([0.5, 2e-100, 3.0, 2.0, 2e100])
    z = np.asarray([1.0, 5e-100, 0.7, 4.0, 4e100])
    p = np.asarray([0.8, 3e-100, 1.7, 0.9, 3e100])
    cases = [
        (phx.special.elliprc, scipy.special.elliprc, (x, y)),
        (phx.special.elliprf, scipy.special.elliprf, (x, y, z)),
        (phx.special.elliprd, scipy.special.elliprd, (x, y, z)),
        (phx.special.elliprj, scipy.special.elliprj, (x, y, z, p)),
        (phx.special.elliprg, scipy.special.elliprg, (x, y, z)),
    ]
    for function, reference, arguments in cases:
        actual = np.asarray(function(*(jnp.asarray(value) for value in arguments)))
        expected = reference(*arguments)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=2e-15)


def test_carlson_symmetry_homogeneity_and_degeneracies():
    values = (0.2, 1.3, 4.7)
    rf_values = [
        float(phx.special.elliprf(*permutation))
        for permutation in itertools.permutations(values)
    ]
    np.testing.assert_allclose(rf_values, rf_values[0], rtol=3e-15)

    rj_values = [
        float(phx.special.elliprj(permutation[0], permutation[1], permutation[2], 0.8))
        for permutation in itertools.permutations(values)
    ]
    np.testing.assert_allclose(rj_values, rj_values[0], rtol=4e-15)

    scale = 1e80
    np.testing.assert_allclose(
        phx.special.elliprf(*(scale * value for value in values)),
        phx.special.elliprf(*values) / np.sqrt(scale),
        rtol=3e-15,
    )
    np.testing.assert_allclose(
        phx.special.elliprd(*(scale * value for value in values)),
        phx.special.elliprd(*values) / scale**1.5,
        rtol=4e-15,
    )
    np.testing.assert_allclose(
        phx.special.elliprj(
            scale * values[0],
            scale * values[1],
            scale * values[2],
            scale * 0.8,
        ),
        phx.special.elliprj(values[0], values[1], values[2], 0.8) / scale**1.5,
        rtol=5e-15,
    )
    np.testing.assert_allclose(
        phx.special.elliprg(*(scale * value for value in values)),
        phx.special.elliprg(*values) * np.sqrt(scale),
        rtol=4e-15,
    )

    np.testing.assert_allclose(phx.special.elliprf(2.0, 2.0, 2.0), 1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(phx.special.elliprd(2.0, 2.0, 2.0), 2.0**-1.5)
    np.testing.assert_allclose(phx.special.elliprj(2.0, 2.0, 2.0, 2.0), 2.0**-1.5)
    np.testing.assert_allclose(phx.special.elliprg(2.0, 2.0, 2.0), np.sqrt(2.0))


def test_carlson_derivatives_compose_in_forward_and_reverse_modes():
    point = jnp.asarray([0.3, 1.2, 2.4, 0.8])

    def observable(arguments):
        x, y, z, p = arguments
        return (
            phx.special.elliprf(x, y, z)
            + 0.2 * phx.special.elliprd(x, y, z)
            + 0.1 * phx.special.elliprj(x, y, z, p)
            + 0.3 * phx.special.elliprg(x, y, z)
        )

    forward = jax.jacfwd(observable)(point)
    reverse = jax.jacrev(observable)(point)
    np.testing.assert_allclose(
        np.asarray(forward), np.asarray(reverse), rtol=3e-13, atol=2e-14
    )
    assert np.all(np.isfinite(np.asarray(forward)))


def test_carlson_invalid_lanes_do_not_poison_valid_lanes():
    values = np.asarray(
        phx.special.elliprf(
            jnp.asarray([0.0, -1.0, 1.0]),
            jnp.asarray([1.0, 1.0, 2.0]),
            jnp.asarray([2.0, 2.0, 3.0]),
        )
    )
    assert np.isfinite(values[[0, 2]]).all()
    assert np.isnan(values[1])

    assert np.isposinf(phx.special.elliprf(0.0, 0.0, 1.0))
    assert np.isposinf(phx.special.elliprd(0.0, 1.0, 0.0))
    assert np.isposinf(phx.special.elliprj(0.0, 0.0, 1.0, 1.0))
    assert np.isnan(phx.special.elliprc(1.0, 0.0))


def test_carlson_zero_boundary_derivatives_are_finite_for_active_arguments():
    cases = [
        (lambda y: phx.special.elliprc(0.0, y), -np.pi / 4.0),
        (lambda y: phx.special.elliprf(0.0, y, y), -np.pi / 4.0),
        (lambda y: phx.special.elliprd(0.0, y, y), -9.0 * np.pi / 8.0),
        (lambda y: phx.special.elliprj(0.0, y, y, y), -9.0 * np.pi / 8.0),
    ]
    for function, expected in cases:
        _, forward = jax.jvp(function, (1.0,), (1.0,))
        reverse = jax.grad(function)(1.0)
        np.testing.assert_allclose(forward, expected, rtol=4e-13, atol=2e-15)
        np.testing.assert_allclose(reverse, expected, rtol=4e-13, atol=2e-15)


def test_carlson_rd_rj_retain_representable_values_at_extreme_dynamic_range():
    rd_arguments = (1.0, 1e300, 1.0)
    rj_arguments = (1.0, 1e300, 1.0, 1.0)
    rd = phx.special.elliprd(*rd_arguments)
    rj = phx.special.elliprj(*rj_arguments)

    assert np.isfinite(rd) and rd > 0.0
    assert np.isfinite(rj) and rj > 0.0
    np.testing.assert_allclose(
        rd, scipy.special.elliprd(*rd_arguments), rtol=5e-13, atol=0.0
    )
    np.testing.assert_allclose(
        rj, scipy.special.elliprd(*rd_arguments), rtol=5e-13, atol=0.0
    )


def test_carlson_rg_equal_arguments_preserve_extreme_scales():
    arguments = jnp.asarray([1e-300, 1e300])
    actual = phx.special.elliprg(arguments, arguments, arguments)
    np.testing.assert_allclose(actual, np.sqrt(np.asarray(arguments)), rtol=3e-15)


def test_carlson_positive_infinity_limits_and_precedence():
    infinity = jnp.asarray(jnp.inf)
    assert phx.special.elliprc(infinity, 1.0) == 0.0
    assert phx.special.elliprf(infinity, 1.0, 1.0) == 0.0
    assert phx.special.elliprd(infinity, 1.0, 1.0) == 0.0
    assert phx.special.elliprj(infinity, 1.0, 1.0, 1.0) == 0.0
    assert np.isposinf(phx.special.elliprg(infinity, 1.0, 1.0))

    lanes = np.asarray(
        jax.jit(phx.special.elliprf)(
            jnp.asarray([jnp.inf, 1.0, -1.0]),
            jnp.asarray([1.0, 2.0, 1.0]),
            jnp.asarray([1.0, 3.0, 1.0]),
        )
    )
    assert lanes[0] == 0.0
    assert np.isfinite(lanes[1])
    assert np.isnan(lanes[2])

    assert np.isposinf(phx.special.elliprf(infinity, 0.0, 0.0))
    assert np.isposinf(phx.special.elliprd(infinity, 1.0, 0.0))
    assert np.isposinf(phx.special.elliprj(infinity, 0.0, 0.0, 1.0))
    assert np.isnan(phx.special.elliprf(infinity, -1.0, 1.0))
    assert np.isnan(phx.special.elliprg(infinity, -1.0, 1.0))
