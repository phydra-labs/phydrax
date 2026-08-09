#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import metrics


def test_hard_calibration_bins_return_auditable_statistics():
    target = jnp.array([0, 0, 1, 1])
    probability = jnp.array([0.1, 0.2, 0.8, 0.9])

    expected = metrics.expected_calibration_error(target, probability, num_bins=2)
    maximum = metrics.maximum_calibration_error(target, probability, num_bins=2)

    assert expected.hard_binning
    assert maximum.hard_binning
    assert jnp.allclose(expected.bin_weight, jnp.array([2.0, 2.0]))
    assert jnp.allclose(expected.mean_probability, jnp.array([0.15, 0.85]))
    assert jnp.allclose(expected.empirical_frequency, jnp.array([0.0, 1.0]))
    assert jnp.allclose(expected.value, 0.15)
    assert jnp.allclose(maximum.value, 0.15)


def test_smooth_calibration_surrogates_are_distinct_and_differentiable():
    target = jnp.array([0, 0, 1, 1])
    probability = jnp.array([0.1, 0.2, 0.8, 0.9])
    hard = metrics.expected_calibration_error(target, probability, num_bins=2)
    smooth = metrics.smooth_expected_calibration_error(
        target,
        probability,
        num_bins=2,
        bin_temperature=0.2,
    )
    smooth_maximum = metrics.smooth_maximum_calibration_error(
        target,
        probability,
        num_bins=2,
        bin_temperature=0.2,
        maximum_temperature=0.1,
    )
    gradient = jax.grad(
        lambda values: (
            metrics.smooth_expected_calibration_error(
                target,
                values,
                num_bins=2,
                bin_temperature=0.2,
            ).value
        )
    )(probability)

    assert hard.hard_binning
    assert not smooth.hard_binning
    assert not smooth_maximum.hard_binning
    assert jnp.isfinite(smooth.value)
    assert jnp.isfinite(smooth_maximum.value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_calibration_invalid_probability_is_explicit():
    result = metrics.expected_calibration_error(
        jnp.array([0, 1]), jnp.array([0.1, 1.2]), num_bins=2
    )
    assert not bool(result.valid)
    assert int(result.status) == metrics.METRIC_INVALID_INPUT


def test_gaussian_interval_and_ordered_categorical_scores():
    target = jnp.array([0.0, 0.0])
    mean = jnp.zeros(2)
    variance = jnp.ones(2)

    negative_log_likelihood = metrics.gaussian_negative_log_likelihood(
        target, mean, variance
    )
    dawid_sebastiani = metrics.dawid_sebastiani_score(target, mean, variance)
    gaussian_crps = metrics.gaussian_crps(target, mean, jnp.ones(2))
    interval = metrics.interval_score(target, -jnp.ones(2), jnp.ones(2), alpha=0.1)

    assert jnp.allclose(negative_log_likelihood.value, 0.5 * jnp.log(2.0 * jnp.pi))
    assert jnp.allclose(dawid_sebastiani.value, 0.0)
    assert jnp.allclose(gaussian_crps.value, (jnp.sqrt(2.0) - 1.0) / jnp.sqrt(jnp.pi))
    assert jnp.allclose(interval.value, 2.0)

    labels = jnp.array([0, 2])
    perfect = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ranked = metrics.ranked_probability_score(labels, perfect)
    spherical = metrics.spherical_score(labels, perfect)
    assert jnp.allclose(ranked.value, 0.0)
    assert jnp.allclose(spherical.value, 1.0)


def test_empirical_crps_and_energy_score_definitions():
    observation = jnp.array([0.0])
    ensemble = jnp.array([[-1.0, 1.0]])
    crps = metrics.crps_ensemble(observation, ensemble)
    smooth_crps = metrics.smooth_crps_ensemble(observation, ensemble, smoothing=0.1)

    vector_observation = jnp.array([[0.0, 0.0]])
    vector_ensemble = jnp.array([[[-1.0, 1.0], [0.0, 0.0]]])
    energy = metrics.energy_score(vector_observation, vector_ensemble)

    assert jnp.allclose(crps.value, 0.5)
    assert float(smooth_crps.value) < float(crps.value)
    assert jnp.allclose(energy.value, 0.5)


def test_probabilistic_scores_jit_and_grad_through_forecasts():
    observation = jnp.array([0.0, 0.5])
    ensemble = jnp.array([[-1.0, 1.0], [0.0, 1.0]])

    compiled = jax.jit(
        lambda values: (
            metrics.smooth_crps_ensemble(observation, values, smoothing=0.1).value
        )
    )(ensemble)
    gradient = jax.grad(
        lambda values: (
            metrics.smooth_crps_ensemble(observation, values, smoothing=0.1).value
        )
    )(ensemble)

    assert jnp.isfinite(compiled)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_probabilistic_complex_policy_supports_energy_but_rejects_gaussian():
    observation = jnp.array([[0.0 + 0.0j]])
    ensemble = jnp.array([[[1.0 + 1.0j, -1.0 - 1.0j]]])
    energy = metrics.energy_score(observation, ensemble)
    assert bool(energy.valid)
    assert jnp.isfinite(energy.value)

    with pytest.raises(TypeError, match="complex"):
        metrics.gaussian_negative_log_likelihood(
            jnp.array([0.0 + 1.0j]),
            jnp.array([0.0 + 0.0j]),
            jnp.ones(1),
        )


def test_calibration_norm_bins_weights_masks_case_axes_and_classwise_score():
    target = jnp.array([0, 1, 1])
    probability = jnp.array([0.1, 0.6, jnp.nan])
    weight = jnp.array([1.0, 3.0, 9.0])
    mask = jnp.array([True, True, False])

    l1 = metrics.expected_calibration_error(
        target,
        probability,
        num_bins=2,
        norm="l1",
        sample_weight=weight,
        mask=mask,
    )
    l2 = metrics.expected_calibration_error(
        target,
        probability,
        num_bins=2,
        norm="l2",
        sample_weight=weight,
        mask=mask,
    )
    maximum = metrics.maximum_calibration_error(
        target,
        probability,
        num_bins=2,
        sample_weight=weight,
        mask=mask,
    )

    assert jnp.allclose(l1.bin_weight, jnp.array([1.0, 3.0]))
    assert jnp.allclose(l1.mean_probability, jnp.array([0.1, 0.6]))
    assert jnp.allclose(l1.empirical_frequency, jnp.array([0.0, 1.0]))
    assert jnp.allclose(l1.value, 0.325)
    assert jnp.allclose(l2.value, 0.35)
    assert jnp.allclose(maximum.value, 0.4)
    assert jnp.allclose(l1.effective_weight, 4.0)

    boundary = metrics.expected_calibration_error(
        jnp.array([0, 1, 1]),
        jnp.array([0.0, 0.5, 1.0]),
        num_bins=2,
    )
    assert jnp.allclose(boundary.bin_weight, jnp.array([1.0, 2.0]))
    assert jnp.allclose(boundary.value, 1.0 / 6.0)

    perfect_classwise = metrics.classwise_expected_calibration_error(
        jnp.array([0, 2]),
        jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        num_bins=2,
    )
    assert bool(perfect_classwise.valid)
    assert jnp.allclose(perfect_classwise.value, 0.0)

    target_cases = jnp.stack((target, 1 - target))
    probability_cases = jnp.stack((probability, 1.0 - probability))
    mask_cases = jnp.stack((mask, mask))
    batched = metrics.expected_calibration_error(
        target_cases,
        probability_cases,
        num_bins=2,
        sample_weight=weight,
        mask=mask_cases,
        sample_axis=-1,
    )
    mapped = jax.vmap(
        lambda labels, values, included: (
            metrics.expected_calibration_error(
                labels,
                values,
                num_bins=2,
                sample_weight=weight,
                mask=included,
            ).value
        )
    )(target_cases, probability_cases, mask_cases)
    compiled = jax.jit(
        lambda values: (
            metrics.expected_calibration_error(
                target_cases,
                values,
                num_bins=2,
                sample_weight=weight,
                mask=mask_cases,
            ).value
        )
    )(probability_cases)

    assert batched.value.shape == (2,)
    assert batched.bin_weight.shape == (2, 2)
    assert jnp.allclose(batched.value, mapped)
    assert jnp.allclose(compiled, mapped)


def test_hard_calibration_is_local_while_smooth_calibration_crosses_bins():
    target = jnp.array([0, 1])
    probability = jnp.array([0.1, 0.6])
    hard_gradient = jax.grad(
        lambda values: (
            metrics.expected_calibration_error(target, values, num_bins=2).value
        )
    )(probability)
    smooth_gradient = jax.grad(
        lambda values: (
            metrics.smooth_maximum_calibration_error(
                target,
                values,
                num_bins=2,
                bin_temperature=0.2,
                maximum_temperature=0.1,
            ).value
        )
    )(probability)

    assert jnp.all(jnp.isfinite(hard_gradient))
    assert jnp.any(jnp.abs(hard_gradient) > 0.0)
    assert jnp.all(jnp.isfinite(smooth_gradient))
    assert jnp.any(jnp.abs(smooth_gradient) > 0.0)

    empty = metrics.expected_calibration_error(
        target, probability, mask=jnp.zeros(2, dtype=bool)
    )
    single_class = metrics.expected_calibration_error(
        jnp.zeros(2, dtype=jnp.int32), probability, num_bins=2
    )
    assert int(empty.status) == metrics.METRIC_EMPTY
    assert bool(single_class.valid)
    with pytest.raises(TypeError, match="complex"):
        metrics.expected_calibration_error(
            target,
            probability.astype(jnp.complex64),
            num_bins=2,
        )


def test_probabilistic_output_axes_and_gaussian_gradients():
    target = jnp.zeros((2, 2))
    mean = jnp.array([[0.0, 1.0], [2.0, 0.0]])
    variance = jnp.ones((2, 2))
    constant = 0.5 * jnp.log(2.0 * jnp.pi)

    likelihood_raw = metrics.gaussian_negative_log_likelihood(
        target,
        mean,
        variance,
        sample_axis=0,
        output_reduction="raw_values",
    )
    likelihood_uniform = metrics.gaussian_negative_log_likelihood(
        target, mean, variance, sample_axis=0
    )
    dawid_raw = metrics.dawid_sebastiani_score(
        target,
        mean,
        variance,
        sample_axis=0,
        output_reduction="raw_values",
    )
    crps_raw = metrics.gaussian_crps(
        target,
        mean,
        jnp.ones_like(mean),
        sample_axis=0,
        output_reduction="raw_values",
    )
    interval_raw = metrics.interval_score(
        target,
        -jnp.ones_like(target),
        jnp.ones_like(target),
        alpha=0.2,
        sample_axis=0,
        output_reduction="raw_values",
    )

    assert jnp.allclose(
        likelihood_raw.value, jnp.array([constant + 1.0, constant + 0.25])
    )
    assert jnp.allclose(likelihood_uniform.value, jnp.mean(likelihood_raw.value))
    assert jnp.allclose(dawid_raw.value, jnp.array([2.0, 0.5]))
    assert crps_raw.value.shape == (2,)
    assert jnp.all(jnp.isfinite(crps_raw.value))
    assert jnp.allclose(interval_raw.value, jnp.full(2, 2.0))

    gradient = jax.grad(
        lambda values: (
            metrics.gaussian_negative_log_likelihood(
                target, values, variance, sample_axis=0
            ).value
            + metrics.dawid_sebastiani_score(
                target, values, variance, sample_axis=0
            ).value
            + metrics.gaussian_crps(
                target, values, jnp.ones_like(values), sample_axis=0
            ).value
        )
    )(mean)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)


def test_probabilistic_member_weights_masks_and_smooth_energy_gradient():
    observation = jnp.array([0.0])
    ensemble = jnp.array([[-1.0, 100.0, 1.0]])
    member_mask = jnp.array([True, False, True])
    member_weight = jnp.array([1.0, 50.0, 1.0])
    crps = metrics.crps_ensemble(
        observation,
        ensemble,
        member_weight=member_weight,
        member_mask=member_mask,
    )
    smooth_crps = metrics.smooth_crps_ensemble(
        observation,
        ensemble,
        smoothing=0.1,
        member_weight=member_weight,
        member_mask=member_mask,
    )
    assert jnp.allclose(crps.value, 0.5)
    assert bool(smooth_crps.valid)
    assert jnp.isfinite(smooth_crps.value)

    vector_observation = jnp.array([[0.0, 0.0]])
    vector_ensemble = jnp.array([[[-1.0, 100.0, 1.0], [0.0, 100.0, 0.0]]])
    energy = metrics.energy_score(
        vector_observation,
        vector_ensemble,
        member_weight=member_weight,
        member_mask=member_mask,
    )
    smooth_energy = metrics.smooth_energy_score(
        vector_observation,
        vector_ensemble,
        smoothing=0.1,
        member_weight=member_weight,
        member_mask=member_mask,
    )
    assert jnp.allclose(energy.value, 0.5)
    assert bool(smooth_energy.valid)
    assert jnp.isfinite(smooth_energy.value)

    active_ensemble = jnp.array([[[-1.0, 1.0], [0.0, 0.0]]])
    smooth_gradient = jax.grad(
        lambda values: (
            metrics.smooth_energy_score(vector_observation, values, smoothing=0.1).value
        )
    )(active_ensemble)
    crps_gradient = jax.grad(
        lambda values: metrics.crps_ensemble(observation, values).value
    )(jnp.array([[-1.0, 1.0]]))

    assert jnp.all(jnp.isfinite(smooth_gradient))
    assert jnp.any(jnp.abs(smooth_gradient) > 0.0)
    assert jnp.all(jnp.isfinite(crps_gradient))
    assert jnp.any(jnp.abs(crps_gradient) > 0.0)


def test_probabilistic_invalid_empty_and_zero_denominator_states():
    target = jnp.array([0.0, 1.0])
    mean = jnp.zeros(2)
    invalid_likelihood = metrics.gaussian_negative_log_likelihood(
        target, mean, jnp.array([1.0, 0.0])
    )
    invalid_dawid = metrics.dawid_sebastiani_score(target, mean, jnp.array([1.0, -1.0]))
    invalid_crps = metrics.gaussian_crps(target, mean, jnp.array([1.0, 0.0]))
    invalid_interval = metrics.interval_score(
        target,
        jnp.array([0.0, 2.0]),
        jnp.array([1.0, 1.0]),
    )
    invalid_categorical = metrics.ranked_probability_score(
        jnp.array([0, 1]),
        jnp.array([[0.7, 0.3], [0.2, 1.2]]),
    )
    invalid_spherical = metrics.spherical_score(jnp.array([0, 1]), jnp.zeros((2, 2)))
    empty = metrics.gaussian_negative_log_likelihood(
        target, mean, jnp.ones(2), mask=jnp.zeros(2, dtype=bool)
    )
    empty_members = metrics.crps_ensemble(
        jnp.array([0.0]),
        jnp.array([[-1.0, 1.0]]),
        member_mask=jnp.zeros(2, dtype=bool),
    )
    empty_energy_members = metrics.energy_score(
        jnp.array([[0.0, 0.0]]),
        jnp.array([[[-1.0, 1.0], [0.0, 0.0]]]),
        member_mask=jnp.zeros(2, dtype=bool),
    )

    for result in (
        invalid_likelihood,
        invalid_dawid,
        invalid_crps,
        invalid_interval,
        invalid_categorical,
        invalid_spherical,
    ):
        assert int(result.status) == metrics.METRIC_INVALID_INPUT
        assert not bool(result.valid)
    assert int(empty.status) == metrics.METRIC_EMPTY
    assert int(empty_members.status) == metrics.METRIC_ZERO_DENOMINATOR
    assert int(empty_energy_members.status) == metrics.METRIC_ZERO_DENOMINATOR


def test_ordered_categorical_probabilistic_scores_are_differentiable():
    target = jnp.array([0, 2])
    probability = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])

    def combined(values):
        return (
            metrics.ranked_probability_score(target, values).value
            + metrics.spherical_score(target, values).value
        )

    compiled = jax.jit(combined)(probability)
    gradient = jax.grad(combined)(probability)
    assert jnp.isfinite(compiled)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0.0)
