#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_lie_algebra_metric_matches_hat_basis_frobenius_pairing():
    for dimension in (2, 3):
        group = phx.metrix.SpecialUnitaryGroup(dimension)
        metric = phx.metrix.LieAlgebraCoordinateMetric(group)
        expected = jnp.real(
            phx.ein.contract("aij,bij->ab", jnp.conj(metric.basis), metric.basis)
        )

        assert metric.valid
        assert jnp.allclose(metric.gram, expected)
        assert jnp.allclose(metric.gram, metric.gram.T)
        assert metric.reconstruction_residual < 1e-10


def test_lie_algebra_metric_solve_and_kinetic_energy_are_consistent():
    metric = phx.metrix.LieAlgebraCoordinateMetric(phx.metrix.SpecialUnitaryGroup(3))
    momentum = jnp.arange(16.0).reshape((2, metric.dimension)) / 10.0
    velocity = metric.solve(momentum)

    assert jnp.allclose(velocity @ metric.gram.T, momentum)
    assert jnp.allclose(
        metric.kinetic_energy(momentum),
        0.5 * jnp.sum(momentum * velocity),
    )


def test_lie_algebra_metric_momentum_covariance_matches_gram():
    metric = phx.metrix.LieAlgebraCoordinateMetric(phx.metrix.SpecialUnitaryGroup(2))
    samples = metric.sample_momentum(jax.random.key(4), (32768,))
    covariance = samples.T @ samples / samples.shape[0]

    assert jnp.allclose(covariance, metric.gram, rtol=0.04, atol=0.04)
