#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def test_correlated_observable_fft_matches_direct_autocovariance():
    samples = jnp.asarray([[[1.0, 3.0], [2.0, 1.0], [4.0, 0.0], [7.0, 2.0], [9.0, 5.0]]])
    result = phx.uq.correlated_observable_diagnostics(
        samples,
        policy=phx.uq.CorrelatedObservablePolicy(max_lag=3, minimum_draws=4),
    )

    values = np.asarray(samples)
    centered = values - values.mean(axis=1, keepdims=True)
    expected = []
    for lag in range(4):
        products = centered[:, : values.shape[1] - lag] * centered[:, lag:]
        expected.append(products.mean(axis=(0, 1)))
    expected = np.asarray(expected)
    expected_correlation = expected / expected[0]

    assert jnp.all(result.valid)
    assert jnp.allclose(result.autocorrelation, expected_correlation, atol=1e-6)


def test_correlated_observable_ar1_recovers_integrated_time():
    rho = 0.7
    innovations = jr.normal(jr.key(1), (4, 8192))

    def step(previous, innovation):
        value = rho * previous + jnp.sqrt(1.0 - rho**2) * innovation
        return value, value

    _, samples = jax.vmap(lambda row: jax.lax.scan(step, 0.0, row))(innovations)
    result = phx.uq.correlated_observable_diagnostics(
        samples[:, 512:],
        policy=phx.uq.CorrelatedObservablePolicy(max_lag=256),
    )

    expected = (1.0 + rho) / (1.0 - rho)
    assert result.valid
    assert jnp.isclose(
        result.integrated_autocorrelation_time,
        expected,
        rtol=0.25,
    )
    assert jnp.isclose(
        result.effective_sample_size,
        samples[:, 512:].size / result.integrated_autocorrelation_time,
    )


def test_correlated_observable_statuses_are_explicit():
    short = phx.uq.correlated_observable_diagnostics(
        jnp.arange(6.0)[None, :],
        policy=phx.uq.CorrelatedObservablePolicy(minimum_draws=8),
    )
    constant = phx.uq.correlated_observable_diagnostics(jnp.ones((2, 16)))
    nonfinite = phx.uq.correlated_observable_diagnostics(
        jnp.asarray([[0.0, 1.0, jnp.nan, 2.0], [0.0, 1.0, 2.0, 3.0]])
    )

    assert int(short.status) == phx.uq.CORRELATED_OBSERVABLE_INSUFFICIENT_DRAWS
    assert int(constant.status) == phx.uq.CORRELATED_OBSERVABLE_ZERO_VARIANCE
    assert int(nonfinite.status) == phx.uq.CORRELATED_OBSERVABLE_NONFINITE
    assert not short.valid
    assert not constant.valid
    assert not nonfinite.valid


def test_correlated_observable_preserves_outputs_and_jit():
    samples = jr.normal(jr.key(2), (3, 64, 2, 3))
    policy = phx.uq.CorrelatedObservablePolicy(max_lag=16)
    eager = phx.uq.correlated_observable_diagnostics(samples, policy=policy)
    compiled = jax.jit(
        lambda value: phx.uq.correlated_observable_diagnostics(value, policy=policy)
    )(samples)

    assert eager.mean.shape == (2, 3)
    assert eager.autocorrelation.shape == (17, 2, 3)
    assert jnp.allclose(compiled.mean, eager.mean)
    assert jnp.allclose(
        compiled.integrated_autocorrelation_time,
        eager.integrated_autocorrelation_time,
    )


def test_correlated_observable_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="chain and draw"):
        phx.uq.correlated_observable_diagnostics(jnp.ones((8,)))
    with pytest.raises(TypeError, match="real samples"):
        phx.uq.correlated_observable_diagnostics(jnp.ones((2, 8), dtype=complex))
    with pytest.raises(ValueError, match="max_lag"):
        phx.uq.CorrelatedObservablePolicy(max_lag=0)
