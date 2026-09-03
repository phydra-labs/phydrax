#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.uq._constraint_conditioning import ConstraintLikelihoodTerm


def _normal_log_density(residual, covariance):
    sign, log_determinant = jnp.linalg.slogdet(covariance)
    assert sign > 0.0
    return -0.5 * (
        residual @ jnp.linalg.solve(covariance, residual)
        + log_determinant
        + residual.size * jnp.log(2.0 * jnp.pi)
    )


def test_diagonal_constraint_likelihood_is_normalized():
    observed = jnp.asarray([1.5, -0.25])
    prediction = jnp.asarray([0.9, 0.5])
    scales = jnp.asarray([0.4, 0.7])
    likelihood = ConstraintLikelihoodTerm(observed, noise_scale=scales)
    residual = observed - prediction
    expected = _normal_log_density(residual, jnp.diag(scales**2))

    assert jnp.allclose(likelihood.log_likelihood(prediction), expected)
    assert jnp.array_equal(likelihood.physical_noise.covariance, jnp.diag(scales**2))
    assert int(likelihood.physical_noise_rank) == 2


def test_correlated_constraint_likelihood_matches_dense_normal_density():
    observed = jnp.asarray([0.5, -1.0])
    prediction = jnp.asarray([-0.1, -0.4])
    covariance = jnp.asarray([[0.8, 0.3], [0.3, 0.5]])
    likelihood = ConstraintLikelihoodTerm(
        observed,
        noise_covariance=covariance,
        rank_tolerance=1.0e-8,
    )

    assert jnp.allclose(
        likelihood.log_likelihood(prediction),
        _normal_log_density(observed - prediction, covariance),
    )
    assert jnp.allclose(likelihood.physical_noise.covariance, covariance)


def test_singular_constraint_likelihood_uses_its_intrinsic_support_density():
    covariance = jnp.asarray([[1.0, 1.0], [1.0, 1.0]])
    likelihood = ConstraintLikelihoodTerm(
        jnp.asarray([1.0, 1.0]),
        noise_covariance=covariance,
        rank_tolerance=1.0e-7,
        support_tolerance=1.0e-7,
    )
    expected_on_support = -0.5 * (1.0 + jnp.log(2.0) + jnp.log(2.0 * jnp.pi))

    assert int(likelihood.physical_noise_rank) == 1
    assert jnp.allclose(likelihood.log_likelihood(jnp.zeros(2)), expected_on_support)
    assert jnp.isneginf(likelihood.log_likelihood(jnp.asarray([0.0, 2.0])))


def test_zero_noise_constraint_likelihood_is_an_exact_support_measure():
    observed = jnp.asarray([1.25, -0.5])
    likelihood = ConstraintLikelihoodTerm(observed)

    assert likelihood.physical_noise.factor.shape == (2, 0)
    assert int(likelihood.physical_noise_rank) == 0
    assert likelihood.log_likelihood(observed) == 0.0
    assert jnp.isneginf(likelihood.log_likelihood(observed + jnp.asarray([0.0, 1.0e-3])))
