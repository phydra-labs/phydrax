#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.linalg._constraint_operators import prepare_constraint_operator
from phydrax.linalg._operators import DenseLinearOperator
from phydrax.uq._constraint_conditioning import (
    ApproximateGaussianConstraintConditioner,
    build_constraint_posterior,
    CONSTRAINT_CONDITIONING_INCONSISTENT_SUPPORT,
    CONSTRAINT_CONDITIONING_SUCCESS,
    ConstraintLikelihoodTerm,
    LinearGaussianConstraintConditioner,
)
from phydrax.uq._gaussian_factor import GaussianFactor


def test_exact_linear_gaussian_conditioning_matches_dense_closed_form():
    prior_mean = jnp.asarray([0.5, -0.25])
    prior_covariance = jnp.asarray([[2.0, 0.3], [0.3, 1.0]])
    prior_factor = GaussianFactor(jnp.linalg.cholesky(prior_covariance))
    matrix = jnp.asarray([[1.0, 2.0]])
    noise_variance = 0.5
    observed = jnp.asarray([1.2])
    likelihood = ConstraintLikelihoodTerm(
        observed,
        noise_scale=jnp.sqrt(noise_variance),
    )

    result = LinearGaussianConstraintConditioner().condition(
        prior_mean,
        prior_factor,
        matrix,
        likelihood,
    )

    predictive_mean = matrix @ prior_mean
    predictive_covariance = matrix @ prior_covariance @ matrix.T + noise_variance
    gain = prior_covariance @ matrix.T @ jnp.linalg.inv(predictive_covariance)
    expected_mean = prior_mean + gain @ (observed - predictive_mean)
    expected_covariance = prior_covariance - gain @ matrix @ prior_covariance
    expected_log_evidence = -0.5 * (
        (observed - predictive_mean)
        @ jnp.linalg.solve(predictive_covariance, observed - predictive_mean)
        + jnp.log(predictive_covariance[0, 0])
        + jnp.log(2.0 * jnp.pi)
    )

    assert bool(result.valid)
    assert int(result.status) == CONSTRAINT_CONDITIONING_SUCCESS
    assert result.approximation == "exact-linear"
    assert jnp.allclose(result.posterior_mean, expected_mean)
    assert jnp.allclose(result.posterior_covariance, expected_covariance)
    assert jnp.allclose(result.log_evidence, expected_log_evidence)
    assert jnp.allclose(result.evidence.log_normalizer, result.log_evidence)


@pytest.mark.parametrize(
    "method",
    ["first-order", "cubature", "unscented", "gauss-hermite"],
)
def test_nonlinear_conditioning_reports_the_selected_approximation(method):
    prior_mean = jnp.asarray([0.4])
    prior_factor = GaussianFactor(jnp.asarray([[0.3]]))
    likelihood = ConstraintLikelihoodTerm(
        jnp.asarray([0.25]),
        noise_scale=jnp.asarray([0.2]),
    )
    conditioner = ApproximateGaussianConstraintConditioner(method)

    result = conditioner.condition(
        prior_mean,
        prior_factor,
        lambda value: jnp.asarray([value[0] ** 2]),
        likelihood,
    )

    assert bool(result.valid)
    assert int(result.status) == CONSTRAINT_CONDITIONING_SUCCESS
    assert result.approximation == method
    assert not result.zero_noise_bridge


def test_zero_noise_hard_bridge_lifts_an_unchanged_coordinate_posterior():
    hard_matrix = jnp.asarray([[1.0, 1.0]])
    hard_operator = prepare_constraint_operator(DenseLinearOperator(hard_matrix))
    coordinate_mean = jnp.asarray([0.4])
    coordinate_factor = GaussianFactor(jnp.asarray([[0.6]]))
    condition = ConstraintLikelihoodTerm(jnp.asarray([3.0]))

    result = build_constraint_posterior(
        (coordinate_mean, coordinate_factor),
        condition,
        hard_operator=hard_operator,
    )

    reconstructed_mean = (
        result.feasible_origin + result.feasible_basis @ result.coordinate_mean
    )
    assert bool(result.valid)
    assert int(result.status) == CONSTRAINT_CONDITIONING_SUCCESS
    assert result.zero_noise_bridge
    assert result.evidence.stamp.exact
    assert result.log_evidence == 0.0
    assert jnp.array_equal(result.coordinate_mean, coordinate_mean)
    assert jnp.allclose(result.coordinate_covariance, coordinate_factor.covariance)
    assert jnp.allclose(result.posterior_mean, reconstructed_mean)
    assert jnp.allclose(hard_matrix @ result.posterior_mean, condition.observed)
    assert jnp.allclose(
        hard_matrix @ result.posterior_covariance @ hard_matrix.T,
        jnp.zeros((1, 1)),
        atol=1.0e-7,
    )


def test_numerical_jitter_cannot_turn_inconsistent_zero_noise_into_evidence():
    prior_mean = jnp.asarray([0.0])
    prior_factor = GaussianFactor(jnp.asarray([[1.0]]))
    matrix = jnp.asarray([[1.0], [1.0]])
    likelihood = ConstraintLikelihoodTerm(
        jnp.asarray([1.0, 2.0]),
        support_tolerance=1.0e-8,
    )
    conditioner = LinearGaussianConstraintConditioner(
        numerical_jitter=1.0e-3,
        rank_tolerance=1.0e-8,
        support_tolerance=1.0e-8,
    )

    result = conditioner.condition(prior_mean, prior_factor, matrix, likelihood)

    assert not bool(result.valid)
    assert int(result.status) == CONSTRAINT_CONDITIONING_INCONSISTENT_SUPPORT
    assert jnp.isneginf(result.log_evidence)
    assert jnp.isneginf(result.evidence.log_normalizer)
    assert result.evidence.accepted_probability == 0.0
    assert int(result.physical_noise_rank) == 0
    assert jnp.allclose(result.numerical_jitter, 1.0e-3)
    assert "physical-noise+separate-numerical-jitter" in result.evidence.evidence_id
