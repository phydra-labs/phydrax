import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sc

from phydrax.special._normal import (
    normal_cdf,
    normal_logcdf,
    normal_logpdf,
    normal_logsurvival,
    normal_pdf,
    normal_quantile,
    normal_survival,
)


def test_normal_density_and_probability_boundaries_are_exact():
    x = jnp.asarray([-jnp.inf, -2.0, -0.0, 0.0, 2.0, jnp.inf, jnp.nan])
    pdf = np.asarray(normal_pdf(x))
    logpdf = np.asarray(normal_logpdf(x))
    cdf = np.asarray(normal_cdf(x))
    logcdf = np.asarray(normal_logcdf(x))
    survival = np.asarray(normal_survival(x))
    logsurvival = np.asarray(normal_logsurvival(x))

    np.testing.assert_array_equal(pdf[[0, 5]], np.zeros(2))
    assert np.isneginf(logpdf[0]) and np.isneginf(logpdf[5])
    np.testing.assert_array_equal(cdf[[0, 5]], np.asarray([0.0, 1.0]))
    assert np.isneginf(logcdf[0]) and logcdf[5] == 0.0
    np.testing.assert_array_equal(survival[[0, 5]], np.asarray([1.0, 0.0]))
    assert logsurvival[0] == 0.0 and np.isneginf(logsurvival[5])
    assert pdf[2] == pdf[3] == 1.0 / math.sqrt(2.0 * math.pi)
    assert np.isnan(
        np.asarray(
            [pdf[-1], logpdf[-1], cdf[-1], logcdf[-1], survival[-1], logsurvival[-1]]
        )
    ).all()


def test_normal_log_probabilities_and_survival_preserve_deep_tails():
    x = np.asarray([-38.0, -20.0, -12.0, 0.0, 12.0, 20.0, 38.0])
    np.testing.assert_allclose(
        np.asarray(normal_logcdf(jnp.asarray(x))),
        sc.log_ndtr(x),
        rtol=5e-11,
        atol=5e-11,
    )
    np.testing.assert_allclose(
        np.asarray(normal_logsurvival(jnp.asarray(x))),
        sc.log_ndtr(-x),
        rtol=5e-11,
        atol=5e-11,
    )

    right_tail = np.asarray(normal_survival(jnp.asarray([8.0, 12.0, 20.0])))
    np.testing.assert_allclose(
        right_tail, sc.ndtr(-np.asarray([8.0, 12.0, 20.0])), rtol=2e-14
    )
    assert np.all(right_tail > 0.0)
    assert 1.0 - float(normal_cdf(12.0)) == 0.0


def test_normal_quantile_handles_deep_probabilities_endpoints_and_domain():
    probabilities = np.asarray([1e-300, 1e-100, 1e-20, 0.5, 1.0 - 1e-12])
    quantiles = np.asarray(normal_quantile(jnp.asarray(probabilities)))
    np.testing.assert_allclose(quantiles, sc.ndtri(probabilities), rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        np.asarray(normal_logcdf(jnp.asarray(quantiles[:-1]))),
        np.log(probabilities[:-1]),
        rtol=5e-11,
        atol=5e-11,
    )

    boundary = np.asarray(normal_quantile(jnp.asarray([0.0, 1.0, -0.1, 1.1, jnp.nan])))
    assert np.isneginf(boundary[0])
    assert np.isposinf(boundary[1])
    assert np.isnan(boundary[2:]).all()


def test_normal_functions_promote_real_dtypes_and_reject_complex_values():
    functions = (
        normal_pdf,
        normal_logpdf,
        normal_cdf,
        normal_logcdf,
        normal_survival,
        normal_logsurvival,
        normal_quantile,
    )
    for function in functions:
        assert function(jnp.asarray(0.5, dtype=jnp.float16)).dtype == jnp.float32
        assert function(jnp.asarray(0.5, dtype=jnp.bfloat16)).dtype == jnp.float32
        assert function(jnp.asarray(0.5, dtype=jnp.float32)).dtype == jnp.float32
        assert function(jnp.asarray(0.5, dtype=jnp.float64)).dtype == jnp.float64
        with pytest.raises(TypeError, match="does not support complex-valued inputs"):
            function(0.5 + 0.1j)


def test_normal_functions_compose_under_jit_vmap_and_grad():
    x = jnp.asarray([-3.0, -0.5, 0.0, 1.25, 4.0])
    compiled = jax.jit(
        lambda value: (
            normal_pdf(value),
            normal_logpdf(value),
            normal_cdf(value),
            normal_logcdf(value),
            normal_survival(value),
            normal_logsurvival(value),
        )
    )
    outputs = compiled(x)
    assert all(output.shape == x.shape for output in outputs)

    cdf_gradients = jax.vmap(jax.grad(normal_cdf))(x)
    logpdf_gradients = jax.vmap(jax.grad(normal_logpdf))(x)
    np.testing.assert_allclose(
        np.asarray(cdf_gradients), np.asarray(normal_pdf(x)), rtol=2e-14
    )
    np.testing.assert_allclose(
        np.asarray(logpdf_gradients), -np.asarray(x), rtol=0.0, atol=0.0
    )

    probabilities = jnp.asarray([1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8])
    quantile_gradients = jax.jit(jax.vmap(jax.grad(normal_quantile)))(probabilities)
    quantiles = normal_quantile(probabilities)
    np.testing.assert_allclose(
        np.asarray(quantile_gradients),
        np.asarray(1.0 / normal_pdf(quantiles)),
        rtol=8e-14,
    )

    infinity_gradients = jax.vmap(jax.grad(normal_pdf))(jnp.asarray([-jnp.inf, jnp.inf]))
    np.testing.assert_array_equal(np.asarray(infinity_gradients), np.zeros(2))
