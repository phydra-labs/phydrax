#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import scipy.stats as stats

import phydrax as phx


def test_gamma_poisson_update_evidence_predictive_and_merge_are_exact():
    pair = phx.uq.GammaPoissonConjugacy(2.5, 1.7)
    counts = jnp.asarray([0.0, 3.0, 1.0, 4.0])
    exposure = jnp.asarray([0.5, 1.2, 2.0, 0.7])
    update = pair.update(counts, exposure=exposure)
    posterior_shape, posterior_rate = update.posterior_shape_rate
    expected_shape = 2.5 + jnp.sum(counts)
    expected_rate = 1.7 + jnp.sum(exposure)
    expected_evidence = (
        jsp.special.gammaln(expected_shape)
        - jsp.special.gammaln(2.5)
        + 2.5 * jnp.log(1.7)
        - expected_shape * jnp.log(expected_rate)
        + jnp.sum(counts * jnp.log(exposure) - jsp.special.gammaln(counts + 1.0))
    )
    first = pair.summarize(counts[:2], exposure=exposure[:2])
    second = pair.summarize(counts[2:], exposure=exposure[2:])
    merged = pair.update_statistics(first.merge(second))
    predictive_count = 3
    predictive_exposure = 1.4
    expected_predictive = stats.nbinom.logpmf(
        predictive_count,
        float(expected_shape),
        float(expected_rate / (expected_rate + predictive_exposure)),
    )

    np.testing.assert_allclose(posterior_shape, expected_shape, atol=2e-15)
    np.testing.assert_allclose(posterior_rate, expected_rate, atol=2e-15)
    np.testing.assert_allclose(update.log_evidence, expected_evidence, atol=2e-15)
    np.testing.assert_allclose(
        update.predictive_log_prob(predictive_count, exposure=predictive_exposure),
        expected_predictive,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        merged.posterior_natural.values, update.posterior_natural.values, atol=2e-15
    )
    np.testing.assert_allclose(merged.log_evidence, update.log_evidence, atol=2e-15)
    assert bool(update.valid)


def test_gamma_poisson_sequential_updates_and_predictive_sampling():
    pair = phx.uq.GammaPoissonConjugacy(1.8, 2.2)
    first = pair.update(jnp.asarray([1, 0, 2]), exposure=jnp.asarray([0.5, 1.0, 1.5]))
    first_shape, first_rate = first.posterior_shape_rate
    continuation = phx.uq.GammaPoissonConjugacy(first_shape, first_rate)
    second = continuation.update(jnp.asarray([3, 1]), exposure=jnp.asarray([2.0, 0.7]))
    direct = pair.update(
        jnp.asarray([1, 0, 2, 3, 1]),
        exposure=jnp.asarray([0.5, 1.0, 1.5, 2.0, 0.7]),
    )
    samples = direct.sample_predictive(jr.key(40), (80_000,), exposure=1.3)
    direct_shape, direct_rate = direct.posterior_shape_rate

    np.testing.assert_allclose(
        second.posterior_natural.values, direct.posterior_natural.values, atol=3e-15
    )
    np.testing.assert_allclose(
        first.log_evidence + second.log_evidence, direct.log_evidence, atol=3e-15
    )
    np.testing.assert_allclose(
        jnp.mean(samples), 1.3 * direct_shape / direct_rate, atol=0.025
    )
    assert samples.shape == (80_000,)


def test_gamma_poisson_batched_jit_and_invalid_observations():
    pair = phx.uq.GammaPoissonConjugacy(jnp.asarray([1.0, 2.0]), 3.0)
    counts = jnp.asarray([[0.0, 1.0, 2.0], [3.0, 1.0, 0.0]])
    update = jax.jit(lambda values: pair.update(values, sample_axes=1))(counts)
    invalid = pair.update(jnp.asarray([[0.0, -1.0], [1.5, 2.0]]), sample_axes=1)
    malformed_statistics = phx.uq.GammaPoissonStatistics(
        jnp.asarray(-1.0),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1),
        jnp.asarray(True),
    )
    malformed = pair.update_statistics(malformed_statistics)
    compensated = pair.update_statistics(
        malformed_statistics.merge(
            phx.uq.GammaPoissonStatistics(
                jnp.asarray(1.0),
                jnp.asarray(1.0),
                jnp.asarray(0.0),
                jnp.asarray(1),
                jnp.asarray(True),
            )
        )
    )
    shapes, rates = update.posterior_shape_rate

    np.testing.assert_allclose(shapes, jnp.asarray([4.0, 6.0]), atol=2e-15)
    np.testing.assert_allclose(rates, jnp.asarray([6.0, 6.0]), atol=2e-15)
    assert jnp.all(update.valid)
    assert not jnp.any(invalid.valid)
    assert jnp.all(jnp.isnan(invalid.posterior_natural.values))
    assert not jnp.any(malformed.valid)
    assert jnp.all(jnp.isnan(malformed.posterior_natural.values))
    assert not jnp.any(compensated.valid)
    assert jnp.all(jnp.isnan(compensated.posterior_natural.values))


def test_dirichlet_categorical_update_evidence_predictive_and_merge_are_exact():
    concentration = jnp.asarray([0.7, 1.2, 2.1, 3.0])
    labels = jnp.asarray([0, 3, 1, 3, 2, 3, 1, 3])
    pair = phx.uq.DirichletCategoricalConjugacy(concentration)
    update = pair.update(labels)
    counts = jnp.bincount(labels, length=4)
    expected = concentration + counts

    def log_beta(value):
        return jnp.sum(jsp.special.gammaln(value)) - jsp.special.gammaln(jnp.sum(value))

    first = pair.summarize(labels[:3])
    second = pair.summarize(labels[3:])
    merged = pair.update_statistics(first.merge(second))

    np.testing.assert_allclose(update.posterior_concentration, expected, atol=2e-15)
    np.testing.assert_allclose(
        update.log_evidence, log_beta(expected) - log_beta(concentration), atol=2e-15
    )
    np.testing.assert_allclose(
        update.predictive_probabilities, expected / jnp.sum(expected), atol=2e-15
    )
    np.testing.assert_allclose(
        update.predictive_log_prob(3),
        jnp.log(expected[3] / jnp.sum(expected)),
        atol=2e-15,
    )
    np.testing.assert_allclose(
        merged.posterior_natural.values, update.posterior_natural.values, atol=2e-15
    )
    np.testing.assert_allclose(merged.log_evidence, update.log_evidence, atol=2e-15)
    assert bool(update.valid)


def test_dirichlet_categorical_sequential_updates_and_predictive_sampling():
    concentration = jnp.asarray([1.0, 2.0, 1.5])
    pair = phx.uq.DirichletCategoricalConjugacy(concentration)
    first = pair.update(jnp.asarray([0, 2, 2, 1]))
    continuation = phx.uq.DirichletCategoricalConjugacy(first.posterior_concentration)
    second = continuation.update(jnp.asarray([2, 1, 1]))
    direct = pair.update(jnp.asarray([0, 2, 2, 1, 2, 1, 1]))
    samples = direct.sample_predictive(jr.key(41), (100_000,))
    frequencies = jnp.bincount(samples, length=3) / samples.size

    np.testing.assert_allclose(
        second.posterior_natural.values, direct.posterior_natural.values, atol=3e-15
    )
    np.testing.assert_allclose(
        first.log_evidence + second.log_evidence, direct.log_evidence, atol=3e-15
    )
    np.testing.assert_allclose(frequencies, direct.predictive_probabilities, atol=0.008)
    assert samples.shape == (100_000,)


def test_dirichlet_categorical_batched_jit_and_invalid_labels():
    pair = phx.uq.DirichletCategoricalConjugacy(jnp.asarray([1.0, 1.0, 1.0]))
    labels = jnp.asarray([[0, 1, 2, 2], [1, 1, 0, 1]])
    update = jax.jit(lambda values: pair.update(values, sample_axes=1))(labels)
    invalid = pair.update(jnp.asarray([[0.0, 3.0], [0.5, 1.0]]), sample_axes=1)
    malformed_statistics = phx.uq.DirichletCategoricalStatistics(
        jnp.asarray([-1.0, 1.0, 1.0]),
        jnp.asarray(1),
        jnp.asarray(True),
    )
    malformed = pair.update_statistics(malformed_statistics)
    compensated = pair.update_statistics(
        malformed_statistics.merge(
            phx.uq.DirichletCategoricalStatistics(
                jnp.asarray([2.0, 0.0, 0.0]),
                jnp.asarray(2),
                jnp.asarray(True),
            )
        )
    )

    np.testing.assert_allclose(
        update.posterior_concentration,
        jnp.asarray([[2.0, 2.0, 3.0], [2.0, 4.0, 1.0]]),
        atol=2e-15,
    )
    assert jnp.all(update.valid)
    assert not jnp.any(invalid.valid)
    assert jnp.all(jnp.isnan(invalid.posterior_natural.values))
    assert not bool(malformed.valid)
    assert jnp.all(jnp.isnan(malformed.posterior_natural.values))
    assert not bool(compensated.valid)
    assert jnp.all(jnp.isnan(compensated.posterior_natural.values))
