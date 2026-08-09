#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.uq import (
    BernoulliFamily,
    empirical_controllability_directions,
    empirical_observability_directions,
    experiment_design_objective,
    exponential_family_fisher_action,
    EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN,
    exponential_family_parameter_fisher_action,
    ExponentialRateFamily,
    fisher_information_action,
    fixed_noise_pathwise_gradient,
    gauss_newton_action,
    likelihood_ratio_gradient,
    NormalFamily,
    PoissonFamily,
    resampling_score_gradient,
)


def test_likelihood_ratio_matches_common_random_number_finite_difference():
    theta = 0.7
    noise = jr.normal(jr.key(21), (100_000,))
    samples = theta + noise
    values = samples**2
    scores = samples - theta
    estimate = likelihood_ratio_gradient(
        values,
        scores,
        noise_id="normal-draws-21",
    )
    epsilon = 1e-4
    plus = jnp.mean((theta + epsilon + noise) ** 2)
    minus = jnp.mean((theta - epsilon + noise) ** 2)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(estimate.gradient, finite_difference, atol=1.5e-2)
    assert bool(estimate.valid)
    assert estimate.estimator_id == "likelihood_ratio"
    assert estimate.noise_id == "normal-draws-21"


def test_fixed_noise_pathwise_gradient_matches_finite_difference():
    parameters = jnp.asarray([0.4, -0.7])
    noise = jnp.asarray([1.2, -0.3])

    def response(value, fixed_noise):
        return jnp.sum(jnp.sin(value + fixed_noise) ** 2)

    result = fixed_noise_pathwise_gradient(
        response,
        parameters,
        noise,
        noise_id="fixed-epsilon",
    )
    epsilon = 1e-5
    basis = jnp.eye(2)
    finite_difference = jax.vmap(
        lambda direction: (
            (
                response(parameters + epsilon * direction, noise)
                - response(parameters - epsilon * direction, noise)
            )
            / (2.0 * epsilon)
        )
    )(basis)

    np.testing.assert_allclose(result.gradient, finite_difference, atol=2e-9)
    assert result.method_id == "jax_jacrev_fixed_noise"
    assert result.approximation == "exact_autodiff_for_fixed_realization"


def test_resampling_scores_include_normalizer_and_obey_score_identity():
    features = jnp.asarray([-1.0, 0.5, 2.0, 3.0])
    theta = 0.4
    log_weights = theta * features
    ancestors = jnp.asarray([0, 1, 2, 3, 3, 2, 3, 1])
    values = features**2
    result = resampling_score_gradient(
        values,
        log_weights,
        features,
        ancestors,
        resampling_id="systematic-step-7",
        noise_id="particle-cloud-4",
    )

    np.testing.assert_allclose(result.expected_centered_score, 0.0, atol=2e-15)
    expected_centered = features - jnp.sum(result.normalized_weights * features)
    np.testing.assert_allclose(result.centered_scores, expected_centered, atol=2e-15)
    assert bool(result.valid)
    assert result.estimator_id == "resampling_score"
    assert result.resampling_id == "systematic-step-7"


def test_fisher_and_gauss_newton_actions_match_dense_products():
    scores = jnp.asarray([[1.0, -2.0, 0.5], [0.2, 0.7, -1.0], [-0.4, 1.1, 0.3]])
    vector = jnp.asarray([0.6, -0.2, 0.9])
    fisher = fisher_information_action(scores, vector, regularization=0.15)
    dense_fisher = scores.T @ scores / scores.shape[0] + 0.15 * jnp.eye(3)
    np.testing.assert_allclose(fisher.action, dense_fisher @ vector, atol=2e-15)

    matrix = jnp.asarray([[1.0, 2.0, -0.5], [0.3, -0.4, 1.2]])
    residual = lambda parameters: jnp.tanh(matrix @ parameters)
    parameters = jnp.asarray([0.2, -0.1, 0.4])
    gauss_newton = gauss_newton_action(
        residual,
        parameters,
        vector,
        regularization=0.05,
    )
    jacobian = jax.jacrev(residual)(parameters)
    expected = (jacobian.T @ jacobian + 0.05 * jnp.eye(3)) @ vector
    np.testing.assert_allclose(gauss_newton.action, expected, atol=2e-15)
    assert bool(fisher.valid)
    assert bool(gauss_newton.valid)


def test_matrix_free_actions_are_jit_compatible():
    scores = jnp.asarray([[1.0, 2.0], [-0.5, 0.3], [0.2, -0.7]])
    vector = jnp.asarray([0.4, -0.6])
    action = jax.jit(
        lambda sample_scores, direction: (
            fisher_information_action(sample_scores, direction).action
        )
    )(scores, vector)
    np.testing.assert_allclose(action, scores.T @ scores @ vector / 3.0, atol=2e-15)


def test_exact_exponential_family_fisher_actions_match_dense_hessians():
    cases = (
        (BernoulliFamily(), jnp.asarray([0.3])),
        (PoissonFamily(), jnp.asarray([jnp.log(1.7)])),
        (ExponentialRateFamily(), jnp.asarray([-1.4])),
        (NormalFamily(), jnp.asarray([0.2, -0.7])),
    )
    for family, natural_values in cases:
        direction = jnp.linspace(0.2, 0.8, family.signature.dimension)
        natural = family.natural(natural_values)
        result = exponential_family_fisher_action(
            family,
            natural,
            direction,
            regularization=0.07,
        )
        hessian = jax.hessian(
            lambda values: family.log_normalizer(family.natural(values))
        )(natural_values)
        expected = hessian @ direction + 0.07 * direction
        np.testing.assert_allclose(result.action, expected, rtol=2e-12, atol=2e-12)
        assert bool(result.valid)
        assert result.operator_id == "fisher_information"
        assert result.approximation == "exact_exponential_family"


def test_parameter_space_family_fisher_pullback_matches_dense_product_and_jit():
    family = PoissonFamily()
    matrix = jnp.asarray([[1.0, -0.4, 0.2], [0.3, 0.7, -0.5]])
    offset = jnp.asarray([-0.2, 0.4])
    parameters = jnp.asarray([0.1, -0.3, 0.5])
    direction = jnp.asarray([0.6, -0.2, 0.9])
    natural_fn = lambda values: (matrix @ values + offset)[..., None]
    result = exponential_family_parameter_fisher_action(
        family,
        natural_fn,
        parameters,
        direction,
        regularization=0.03,
    )
    natural_values = matrix @ parameters + offset
    expected = (
        matrix.T @ (jnp.exp(natural_values) * (matrix @ direction)) + 0.03 * direction
    )
    compiled = jax.jit(
        lambda values, vector: (
            exponential_family_parameter_fisher_action(
                family,
                natural_fn,
                values,
                vector,
            ).action
        )
    )(parameters, direction)

    np.testing.assert_allclose(result.action, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        compiled,
        matrix.T @ (jnp.exp(natural_values) * (matrix @ direction)),
        rtol=2e-12,
        atol=2e-12,
    )
    assert bool(result.valid)
    assert result.operator_id == "fisher_information_pullback"


def test_exact_family_fisher_reports_invalid_and_nonfinite_coordinates():
    family = ExponentialRateFamily()
    invalid = exponential_family_fisher_action(
        family,
        family.natural(jnp.asarray([0.0])),
        jnp.asarray([1.0]),
    )
    nonfinite = exponential_family_fisher_action(
        family,
        family.natural(jnp.asarray([jnp.nan])),
        jnp.asarray([1.0]),
    )
    assert not bool(invalid.valid)
    assert int(invalid.status) == 2
    assert int(family.natural_domain(family.natural(jnp.asarray([0.0]))).status) == (
        EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN
    )
    assert not bool(nonfinite.valid)
    assert int(nonfinite.status) == 1


def test_empirical_directions_match_dense_observability_and_controllability():
    observation = jnp.asarray([[2.0, 0.0], [0.0, 0.5], [1.0, -1.0]])
    observability = empirical_observability_directions(
        lambda state: observation @ state,
        jnp.asarray([0.2, -0.4]),
        rank=2,
    )
    expected_observability = jnp.linalg.eigvalsh(observation.T @ observation)[::-1]
    np.testing.assert_allclose(
        observability.strengths, expected_observability, atol=2e-15
    )

    response = jnp.asarray([[1.0, 2.0], [-0.5, 0.3], [0.2, 1.1]])
    controllability = empirical_controllability_directions(
        lambda controls: response @ controls,
        jnp.asarray([0.0, 0.0]),
        rank=2,
    )
    expected_controllability = jnp.linalg.eigvalsh(response @ response.T)[::-1][:2]
    np.testing.assert_allclose(
        controllability.strengths, expected_controllability, atol=2e-15
    )
    assert observability.quantity == "observability"
    assert controllability.quantity == "controllability"
    assert bool(observability.valid)
    assert bool(controllability.valid)


def test_experiment_design_objectives_and_invalid_information_status():
    information = jnp.asarray([[4.0, 0.5], [0.5, 2.0]])
    d_optimal = experiment_design_objective(
        information,
        criterion="d_optimal",
    )
    matrix_free = experiment_design_objective(
        lambda vector: information @ vector,
        criterion="d_optimal",
        dimension=2,
    )
    a_optimal = experiment_design_objective(
        information,
        criterion="a_optimal",
    )
    e_optimal = experiment_design_objective(
        information,
        criterion="e_optimal",
    )
    np.testing.assert_allclose(d_optimal.value, jnp.linalg.slogdet(information)[1])
    np.testing.assert_allclose(matrix_free.value, d_optimal.value)
    np.testing.assert_allclose(a_optimal.value, -jnp.trace(jnp.linalg.inv(information)))
    np.testing.assert_allclose(e_optimal.value, jnp.linalg.eigvalsh(information)[0])

    invalid = experiment_design_objective(
        jnp.asarray([[1.0, 0.0], [0.0, -1.0]]),
        criterion="d_optimal",
    )
    assert not bool(invalid.valid)
    assert bool(jnp.isnan(invalid.value))
    assert int(invalid.status) != 0


def test_guarded_dense_direction_and_design_materialization_reject_large_spaces():
    with pytest.raises(ValueError, match="max_dimension"):
        empirical_observability_directions(
            lambda state: state,
            jnp.ones(2),
            rank=1,
            max_dimension=1,
        )
    with pytest.raises(ValueError, match="max_dimension"):
        experiment_design_objective(
            jnp.eye(2),
            criterion="d_optimal",
            max_dimension=1,
        )
