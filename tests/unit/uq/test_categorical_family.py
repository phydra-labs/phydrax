#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def test_categorical_duality_normalization_kl_and_fisher():
    family = phx.uq.CategoricalFamily(4)
    natural_values = jnp.asarray([0.7, -0.4, 0.2])
    natural = family.natural(natural_values)
    mean = family.mean_from_natural(natural)
    probabilities = family.probabilities_from_natural(natural)
    gradient = jax.grad(lambda value: family.log_normalizer(family.natural(value)))(
        natural_values
    )
    direction = jnp.asarray([0.3, -0.2, 0.5])
    hessian = jax.hessian(lambda value: family.log_normalizer(family.natural(value)))(
        natural_values
    )
    conversion = family.natural_from_mean(mean)
    labels = jnp.arange(4)
    other = family.natural(jnp.asarray([-0.2, 0.5, -0.7]))
    other_probabilities = family.probabilities_from_natural(other)
    expected_kl = jnp.sum(
        probabilities * (jnp.log(probabilities) - jnp.log(other_probabilities))
    )

    np.testing.assert_allclose(probabilities[:-1], mean.values, atol=2e-15)
    np.testing.assert_allclose(mean.values, gradient, atol=2e-15)
    np.testing.assert_allclose(conversion.natural.values, natural_values, atol=2e-15)
    np.testing.assert_allclose(
        family.fisher_action(natural, direction), hessian @ direction, atol=2e-15
    )
    np.testing.assert_allclose(jnp.sum(jnp.exp(family.log_prob(natural, labels))), 1.0)
    np.testing.assert_allclose(family.kl_divergence(natural, natural), 0.0, atol=2e-15)
    np.testing.assert_allclose(
        family.kl_divergence(natural, other), expected_kl, atol=2e-15
    )
    assert bool(conversion.valid)
    assert int(conversion.iterations) == 0
    assert conversion.method_id == "categorical-analytic"


def test_categorical_full_logits_are_identified_and_support_is_explicit():
    family = phx.uq.CategoricalFamily(3)
    logits = jnp.asarray([1.2, -0.3, 0.7])
    shifted = logits + 8.0
    natural = family.natural_from_logits(logits)
    shifted_natural = family.natural_from_logits(shifted)

    np.testing.assert_allclose(natural.values, shifted_natural.values, atol=2e-15)
    assert jnp.all(jnp.isfinite(family.log_prob(natural, jnp.arange(3))))
    assert jnp.isneginf(family.log_prob(natural, -1))
    assert jnp.isneginf(family.log_prob(natural, 3))
    assert jnp.isneginf(family.log_prob(natural, 1.5))
    with pytest.raises(ValueError, match="num_categories"):
        phx.uq.CategoricalFamily(1)
    with pytest.raises(ValueError, match="full logits"):
        family.natural_from_logits(jnp.ones((2,)))


def test_categorical_gathered_log_prob_preserves_family_invalid_contract():
    family = phx.uq.CategoricalFamily(3)
    logits = jnp.asarray(
        [
            [10_000.0, -10_000.0, 0.0],
            [0.2, 0.3, -0.4],
            [-0.5, 0.8, 0.1],
            [1.0, -0.7, 0.2],
        ]
    )
    labels = jnp.asarray([0.0, -1.0, 3.0, 1.5])
    natural = family.natural_from_logits(logits)
    gathered = family.log_prob_from_logits(logits, labels)
    generic = family.log_prob(natural, labels)

    np.testing.assert_allclose(gathered, generic, atol=2e-15)
    assert jnp.isfinite(gathered[0])
    assert jnp.all(jnp.isneginf(gathered[1:]))
    np.testing.assert_allclose(
        family.log_prob_from_logits(logits + 13.0, labels),
        gathered,
        atol=2e-12,
    )


def test_categorical_mean_domain_and_projection_report_missing_categories():
    family = phx.uq.CategoricalFamily(3)
    interior = family.mean_domain(family.mean(jnp.asarray([0.2, 0.3])))
    boundary = family.mean_domain(family.mean(jnp.asarray([0.5, 0.5])))
    exterior = family.mean_domain(family.mean(jnp.asarray([0.7, 0.5])))
    projected = phx.uq.fit_exponential_family(
        family, jnp.asarray([0, 1, 0, 1]), sample_axes=0
    )

    assert bool(interior.valid)
    assert int(boundary.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert int(exterior.status) == phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN
    assert not bool(projected.valid)
    assert int(projected.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY


def test_categorical_projection_is_weighted_mergeable_and_sampled():
    family = phx.uq.CategoricalFamily(3)
    labels = jnp.asarray([0, 1, 2, 1, 0, 2, 1, 2])
    log_weights = jnp.asarray([-0.5, 0.3, 0.7, -0.1, 0.2, 0.9, -0.4, 0.1])
    one_shot = phx.uq.project_exponential_family(family, labels, log_weights=log_weights)
    left = phx.uq.ExponentialFamilyProjectionAccumulator.from_log_weights(
        family, labels[:4], log_weights[:4]
    )
    right = phx.uq.ExponentialFamilyProjectionAccumulator.from_log_weights(
        family, labels[4:], log_weights[4:]
    )
    merged = left.merge(right).finalize()
    samples = one_shot.law.sample(jr.key(3), sample_shape=(20_000,))
    expected = jax.nn.softmax(log_weights) @ jax.nn.one_hot(labels, 3)

    np.testing.assert_allclose(one_shot.mean_coordinates.values, expected[:-1])
    np.testing.assert_allclose(
        merged.law.natural.values, one_shot.law.natural.values, atol=2e-15
    )
    np.testing.assert_allclose(
        jnp.bincount(samples, length=3) / samples.size, expected, atol=0.015
    )
    assert samples.shape == (20_000,)
    assert bool(one_shot.valid)


def test_categorical_likelihood_declares_coordinate_and_target_axes():
    family = phx.uq.CategoricalFamily(3)
    full = phx.uq.CategoricalExponentialFamilyLikelihood(
        family, prediction_coordinates="full_logits"
    )
    minimal = phx.uq.CategoricalExponentialFamilyLikelihood(
        family, prediction_coordinates="natural"
    )
    logits = jnp.asarray([[1.0, -0.2, 0.4], [-0.5, 0.8, 0.1]])
    labels = jnp.asarray([0, 1])
    natural = family.natural_from_logits(logits).values
    expected = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(2), labels]

    np.testing.assert_allclose(full.log_prob(logits, labels), expected, atol=2e-15)
    np.testing.assert_allclose(minimal.log_prob(natural, labels), expected, atol=2e-15)
    np.testing.assert_allclose(
        full.class_probabilities(logits),
        minimal.class_probabilities(natural),
        atol=2e-15,
    )
    np.testing.assert_allclose(
        jnp.sum(full.class_probabilities(logits), axis=-1),
        jnp.ones((2,)),
        atol=2e-15,
    )
    assert full.sample(jr.key(4), logits).shape == labels.shape
    with pytest.raises(ValueError, match="coordinate dimension"):
        full.align_observations(jnp.ones((2, 2)), labels)
    with pytest.raises(ValueError, match="incompatible"):
        full.align_observations(logits, jnp.ones((2, 2)))
    with pytest.raises(TypeError, match="unknown parameters"):
        full.log_prob(logits, labels, scale=1.0)


def test_categorical_likelihood_integrates_with_fixed_posterior_terms():
    family = phx.uq.CategoricalFamily(3)
    likelihood = phx.uq.CategoricalExponentialFamilyLikelihood(
        family, prediction_coordinates="full_logits"
    )
    logits = jnp.asarray([[0.1, 1.0, -0.2], [0.7, -0.3, 0.4], [-0.5, 0.2, 0.9]])
    labels = jnp.asarray([1, 0, 2])
    term = phx.uq.FixedObservationLikelihood(
        lambda parameters: parameters,
        labels,
        likelihood,
    )

    per_case = term.per_case_log_prob(logits)
    expected = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(3), labels]
    np.testing.assert_allclose(per_case, expected, atol=2e-15)
    np.testing.assert_allclose(term.log_prob(logits), jnp.sum(expected), atol=2e-15)


def test_categorical_likelihood_matches_multinomial_glm_score_equations():
    family = phx.uq.CategoricalFamily(4)
    likelihood = phx.uq.CategoricalExponentialFamilyLikelihood(
        family, prediction_coordinates="full_logits"
    )
    logits = jnp.asarray([[0.2, -0.7, 1.1, 0.4], [-0.1, 0.8, 0.3, -0.6]])
    labels = jnp.asarray([2, 1])
    one_hot = jax.nn.one_hot(labels, 4)

    family_loss = -jnp.sum(likelihood.log_prob(logits, labels))
    glm_loss = jnp.sum(
        jax.nn.logsumexp(logits, axis=-1) - jnp.sum(one_hot * logits, axis=-1)
    )
    family_gradient = jax.grad(
        lambda value: -jnp.sum(likelihood.log_prob(value, labels))
    )(logits)
    glm_gradient = jax.nn.softmax(logits, axis=-1) - one_hot

    np.testing.assert_allclose(family_loss, glm_loss, atol=2e-15)
    np.testing.assert_allclose(family_gradient, glm_gradient, atol=2e-15)
