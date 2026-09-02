#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.observation import (
    AutocorrelationPlan,
    BrightnessConditionedTransportPlan,
    DiffusionModelPlan,
    DwellTimeLikelihoodPlan,
    FluorescenceCorrelationPlan,
    FluorescencePhotonPlan,
    IVReversalPlan,
    MeanSquareDisplacementPlan,
    PairCorrelationPlan,
)


def test_msd_acf_and_fcs_match_analytic_lag_estimators_with_explicit_units():
    positions_m = jnp.stack((jnp.arange(6.0), jnp.zeros(6)), axis=-1)
    msd_plan = MeanSquareDisplacementPlan(6, 2, 3, 0.25, distance_unit="m", time_unit="s")
    msd = msd_plan.prepare().forward(positions_m)
    np.testing.assert_allclose(msd.lag_times, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(msd.values, [0.0, 1.0, 4.0, 9.0])
    np.testing.assert_array_equal(msd.pair_counts, [6, 5, 4, 3])
    assert bool(msd.successful)
    assert msd_plan.distance_unit == "m"
    assert msd_plan.time_unit == "s"
    assert (
        msd_plan.plan_id
        == MeanSquareDisplacementPlan(
            6, 2, 3, 0.25, distance_unit="m", time_unit="s"
        ).plan_id
    )

    alternating = jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    acf = (
        AutocorrelationPlan(6, 2, 0.5, normalized=True, time_unit="s", signal_unit="V")
        .prepare()
        .forward(alternating)
    )
    np.testing.assert_allclose(acf.values, [1.0, -1.0, 1.0])
    assert bool(acf.successful)

    intensity = alternating + 2.0
    fcs_runtime = FluorescenceCorrelationPlan(
        6, 2, 0.5, time_unit="s", intensity_unit="count/s"
    ).prepare()
    fcs = eqx.filter_jit(fcs_runtime.forward)(intensity)
    np.testing.assert_allclose(fcs.correlation, [0.25, -0.25, 0.25])
    assert bool(fcs.successful)

    sample = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    centered = np.asarray(sample) - float(jnp.mean(sample))
    sample_centered = AutocorrelationPlan(4, 2, 1.0).prepare().forward(sample)
    expected = [
        np.mean(centered * centered),
        np.sum(centered[:-1] * centered[1:]) / 3.0,
        np.sum(centered[:-2] * centered[2:]) / 2.0,
    ]
    np.testing.assert_allclose(sample_centered.variance, expected[0])
    np.testing.assert_allclose(sample_centered.values, np.asarray(expected) / expected[0])


def test_normalized_correlations_are_invariant_to_signal_unit_scale():
    alternating = jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    acf_runtime = AutocorrelationPlan(6, 2, 1.0).prepare()
    base_acf = acf_runtime.forward(alternating)
    scaled_acf = acf_runtime.forward(1.0e-8 * alternating)
    np.testing.assert_allclose(scaled_acf.values, base_acf.values)
    assert bool(scaled_acf.successful)

    intensity = alternating + 2.0
    fcs_runtime = FluorescenceCorrelationPlan(6, 2, 1.0).prepare()
    base_fcs = fcs_runtime.forward(intensity)
    scaled_fcs = fcs_runtime.forward(1.0e-8 * intensity)
    np.testing.assert_allclose(scaled_fcs.correlation, base_fcs.correlation)
    assert bool(scaled_fcs.successful)

    leading = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    trailing = jnp.roll(leading, 2)
    pair_runtime = PairCorrelationPlan(8, 3, 1.0).prepare()
    base_pair = pair_runtime.forward(leading, trailing)
    scaled_pair = pair_runtime.forward(1.0e-8 * leading, 1.0e-8 * trailing)
    np.testing.assert_allclose(scaled_pair.correlation, base_pair.correlation)
    np.testing.assert_allclose(scaled_pair.peak_lag, base_pair.peak_lag)
    assert bool(scaled_pair.successful)


def test_correlation_evidence_fails_closed_for_nonfinite_or_constant_data():
    acf_runtime = AutocorrelationPlan(4, 2, 1.0).prepare()
    constant = acf_runtime.forward(jnp.ones(4))
    assert not bool(constant.finite)
    assert not bool(constant.identifiable)
    assert not bool(constant.successful)

    invalid = acf_runtime.forward(jnp.asarray([1.0, jnp.nan, 2.0, 3.0]))
    assert not bool(invalid.finite)
    assert not bool(invalid.successful)

    dark = FluorescenceCorrelationPlan(4, 2, 1.0).prepare().forward(jnp.zeros(4))
    assert not bool(dark.identifiable)
    assert not bool(dark.successful)


def test_pair_correlation_peak_lag_sign_and_directionality_follow_leader():
    leading = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    trailing = jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    runtime = PairCorrelationPlan(8, 3, 0.2, time_unit="s").prepare()

    forward = runtime.forward(leading, trailing)
    reverse = runtime.forward(trailing, leading)

    np.testing.assert_allclose(forward.peak_lag, 0.4)
    np.testing.assert_allclose(reverse.peak_lag, -0.4)
    assert float(forward.directionality) > 0.0
    assert float(reverse.directionality) < 0.0
    assert bool(forward.successful)
    assert bool(reverse.successful)

    tied = runtime.forward(leading, leading)
    assert bool(tied.finite)
    assert not bool(tied.identifiable)
    assert not bool(tied.successful)
    assert np.isnan(float(tied.peak_lag))
    assert np.isnan(float(tied.directionality))


def test_anomalous_and_confined_diffusion_forward_and_evaluation_are_analytic():
    lag_s = jnp.asarray([0.0, 1.0, 4.0])
    anomalous = DiffusionModelPlan(
        lag_s, 2, "anomalous", distance_unit="m", time_unit="s"
    ).prepare()
    prediction = anomalous.forward(2.0, exponent=0.5, localization_variance=0.1)
    np.testing.assert_allclose(
        prediction.mean_squared_displacement,
        8.0 * np.sqrt([0.0, 1.0, 4.0]) + np.asarray([0.0, 0.4, 0.4]),
    )
    evaluation = anomalous.evaluate(
        prediction.mean_squared_displacement,
        0.25,
        2.0,
        exponent=0.5,
        localization_variance=0.1,
    )
    np.testing.assert_allclose(evaluation.residual, 0.0)
    np.testing.assert_allclose(evaluation.chi_square, 0.0)
    assert bool(evaluation.successful)

    confined = (
        DiffusionModelPlan(lag_s, 2, "confined")
        .prepare()
        .forward(2.0, confinement_time=3.0)
    )
    np.testing.assert_allclose(
        confined.mean_squared_displacement,
        24.0 * (1.0 - np.exp(-np.asarray([0.0, 1.0, 4.0]) / 3.0)),
        rtol=2.0e-6,
    )

    unresolved_anomalous = (
        DiffusionModelPlan([0.0, 1.0], 2, "anomalous")
        .prepare()
        .forward(2.0, exponent=0.5)
    )
    unresolved_confined = (
        DiffusionModelPlan([0.0, 1.0], 2, "confined")
        .prepare()
        .forward(2.0, confinement_time=3.0)
    )
    assert bool(unresolved_anomalous.finite)
    assert not bool(unresolved_anomalous.identifiable)
    assert not bool(unresolved_anomalous.successful)
    assert bool(unresolved_confined.finite)
    assert not bool(unresolved_confined.identifiable)
    assert not bool(unresolved_confined.successful)


def test_brightness_conditioned_transport_recovers_each_bin_and_capacity_evidence():
    runtime = BrightnessConditionedTransportPlan(
        [0.0, 5.0, 10.0],
        4,
        2,
        0.5,
        minimum_count=2,
        brightness_unit="count/s",
        distance_unit="m",
        time_unit="s",
    ).prepare()
    result = runtime.evaluate(
        jnp.asarray([1.0, 2.0, 7.0, 10.0]),
        jnp.asarray([[np.sqrt(2.0), 0.0], [np.sqrt(2.0), 0.0], [2.0, 0.0], [2.0, 0.0]]),
    )
    np.testing.assert_array_equal(result.counts, [2, 2])
    np.testing.assert_allclose(result.diffusion_coefficient, [1.0, 2.0])
    assert bool(result.successful)

    underfilled = runtime.evaluate(
        jnp.asarray([1.0, 2.0, 7.0, 10.0]),
        jnp.zeros((4, 2)),
        active=jnp.asarray([True, True, True, False]),
    )
    np.testing.assert_array_equal(underfilled.identifiable_bins, [True, False])
    assert not bool(underfilled.identifiable)
    assert not bool(underfilled.successful)


@pytest.mark.parametrize("outside_brightness", [0.5, 11.0])
def test_brightness_conditioning_fails_closed_outside_recorded_bins(
    outside_brightness,
):
    runtime = BrightnessConditionedTransportPlan(
        [1.0, 5.0, 10.0], 5, 2, 0.5, minimum_count=2
    ).prepare()
    result = runtime.evaluate(
        jnp.asarray([2.0, 3.0, 7.0, 8.0, outside_brightness]),
        jnp.ones((5, 2)),
    )
    np.testing.assert_array_equal(result.counts, [2, 2])
    assert not bool(result.finite)
    assert not bool(result.successful)


def test_lifetime_fret_irf_limits_and_poisson_draws_are_reproducible():
    plan = FluorescencePhotonPlan(
        jnp.arange(5.0),
        jnp.asarray([0.0, 1.0, 0.0, 0.0]),
        time_unit="ns",
        count_unit="photon",
    )
    runtime = plan.prepare()
    donor = runtime.expected(2.0, 100.0, fret_efficiency=0.0)
    full_transfer = runtime.expected(2.0, 100.0, fret_efficiency=1.0)
    np.testing.assert_allclose(donor.effective_lifetime, 2.0)
    np.testing.assert_allclose(full_transfer.effective_lifetime, 0.0)
    np.testing.assert_allclose(full_transfer.detected_probability, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(jnp.sum(donor.expected_counts), 100.0)
    assert bool(donor.successful)
    assert bool(full_transfer.finite)
    assert not bool(full_transfer.identifiable)

    key = jax.random.key(91)
    first = runtime.forward(key, 2.0, 100.0, fret_efficiency=0.25)
    second = runtime.forward(key, 2.0, 100.0, fret_efficiency=0.25)
    np.testing.assert_array_equal(first.photon_counts, second.photon_counts)
    likelihood = eqx.filter_jit(runtime.evaluate)(
        first.photon_counts, 2.0, 100.0, fret_efficiency=0.25
    )
    assert bool(likelihood.successful)

    invalid = runtime.expected(jnp.nan, 100.0)
    assert not bool(invalid.finite)
    assert not bool(invalid.successful)

    long_lifetime = runtime.expected(1.0e12, 100.0)
    np.testing.assert_allclose(jnp.sum(long_lifetime.expected_counts), 100.0)
    assert bool(long_lifetime.finite)
    assert bool(long_lifetime.successful)


def test_censored_dwell_likelihood_counts_only_observed_exits():
    runtime = DwellTimeLikelihoodPlan(3, time_unit="s").prepare()
    result = eqx.filter_jit(runtime.evaluate)(
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray([True, False, True]),
        0.5,
    )
    np.testing.assert_allclose(result.log_likelihood, 2.0 * np.log(0.5) - 3.0)
    np.testing.assert_allclose(result.maximum_likelihood_rate, 2.0 / 6.0)
    assert int(result.event_count) == 2
    assert int(result.censored_count) == 1
    assert bool(result.successful)

    all_censored = runtime.evaluate(
        jnp.asarray([1.0, 2.0, 3.0]), jnp.zeros(3, dtype=bool), 0.5
    )
    assert bool(all_censored.finite)
    assert not bool(all_censored.identifiable)
    assert np.isnan(float(all_censored.maximum_likelihood_rate))


def test_iv_reversal_inference_recovers_conductance_and_reports_flat_curve():
    voltages_v = jnp.asarray([-0.1, 0.0, 0.1, 0.2, 0.3])
    runtime = IVReversalPlan(
        voltages_v,
        minimum_conductance=1.0e-9,
        voltage_unit="V",
        current_unit="A",
    ).prepare()
    currents_a = 2.0 * (voltages_v - 0.075)
    result = eqx.filter_jit(runtime.evaluate)(currents_a)
    np.testing.assert_allclose(result.conductance, 2.0)
    np.testing.assert_allclose(result.reversal_potential, 0.075, atol=1.0e-7)
    np.testing.assert_allclose(result.weighted_residual_sum_squares, 0.0, atol=1.0e-12)
    assert bool(result.successful)

    rescaled_weights = (
        IVReversalPlan(
            voltages_v,
            weights=jnp.full_like(voltages_v, 1.0e-12),
            minimum_conductance=1.0e-9,
        )
        .prepare()
        .evaluate(currents_a)
    )
    np.testing.assert_allclose(rescaled_weights.conductance, result.conductance)
    np.testing.assert_allclose(
        rescaled_weights.reversal_potential, result.reversal_potential
    )
    assert bool(rescaled_weights.successful)

    flat = runtime.evaluate(jnp.ones_like(voltages_v))
    assert bool(flat.finite)
    assert not bool(flat.identifiable)
    assert np.isnan(float(flat.reversal_potential))

    extreme_runtime = IVReversalPlan(
        jnp.asarray([0.0, 1.0e38], dtype=jnp.float32)
    ).prepare()
    next_float = jnp.nextafter(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    overflowing_reversal = extreme_runtime.evaluate(
        jnp.asarray([1.0, next_float], dtype=jnp.float32)
    )
    assert not bool(overflowing_reversal.identifiable)
    assert not bool(overflowing_reversal.successful)
    assert np.isnan(float(overflowing_reversal.reversal_potential))


def test_plan_constructors_reject_ambiguous_static_configuration():
    with pytest.raises(ValueError, match="max_lag"):
        MeanSquareDisplacementPlan(4, 2, 4, 1.0)
    with pytest.raises(TypeError, match="normalized"):
        AutocorrelationPlan(4, 2, 1.0, normalized=1)
    with pytest.raises(ValueError, match="model"):
        DiffusionModelPlan([0.0, 1.0], 2, "free")
    with pytest.raises(ValueError, match="minimum_conductance"):
        IVReversalPlan([0.0, 1.0], minimum_conductance=-1.0)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="uniformly spaced"):
        invalid = FluorescencePhotonPlan([0.0, 1.0, 3.0], [1.0, 0.0])
        jax.block_until_ready(invalid.bin_edges)
