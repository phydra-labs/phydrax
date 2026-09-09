import jax.numpy as jnp
import jax.random as jr

from phydrax.ml.covariance._random_matrix import (
    clean_covariance_spectrum,
    marchenko_pastur_diagnostics,
)
from phydrax.stochastic._point_process import (
    evaluate_hawkes_likelihood,
    ExponentialHawkesProcess,
    HAWKES_UNSTABLE,
    PointProcessObservation,
    prepare_hawkes_likelihood,
    simulate_hawkes,
)


def test_univariate_hawkes_likelihood_matches_compensator_oracle():
    process = ExponentialHawkesProcess(
        jnp.asarray([0.5]), jnp.asarray([[0.2]]), jnp.asarray([[1.0]])
    )
    observation = PointProcessObservation(
        jnp.asarray([1.0, 2.0, 0.0]),
        jnp.asarray([0, 0, 0]),
        jnp.asarray([True, True, False]),
        start_time=0.0,
        end_time=3.0,
        channel_count=1,
    )

    result = evaluate_hawkes_likelihood(observation, process)

    expected_intensity = jnp.asarray([0.5, 0.5 + 0.2 * jnp.exp(-1.0)])
    expected_compensator = 1.5 + 0.2 * (1.0 - jnp.exp(-2.0)) + 0.2 * (1.0 - jnp.exp(-1.0))
    expected = jnp.sum(jnp.log(expected_intensity)) - expected_compensator
    assert result.successful
    assert jnp.allclose(result.event_intensity[:2], expected_intensity)
    assert jnp.isclose(result.compensator, expected_compensator)
    assert jnp.isclose(result.log_likelihood, expected)


def test_simultaneous_tie_policy_does_not_create_artificial_excitation():
    process = ExponentialHawkesProcess(
        jnp.asarray([0.4]), jnp.asarray([[0.3]]), jnp.asarray([[1.2]])
    )
    observation = PointProcessObservation(
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([0, 0]),
        jnp.asarray([True, True]),
        start_time=0.0,
        end_time=2.0,
        channel_count=1,
    )
    simultaneous = prepare_hawkes_likelihood(
        observation, process, tie_policy="simultaneous"
    )
    ordered = prepare_hawkes_likelihood(observation, process, tie_policy="ordered")

    simultaneous_result = evaluate_hawkes_likelihood(
        observation, process, plan=simultaneous
    )
    ordered_result = evaluate_hawkes_likelihood(observation, process, plan=ordered)

    assert jnp.allclose(simultaneous_result.event_intensity, jnp.asarray([0.4, 0.4]))
    assert jnp.allclose(ordered_result.event_intensity, jnp.asarray([0.4, 0.7]))
    assert simultaneous_result.tied_event_count == 1


def test_hawkes_stability_evidence_and_seeded_prefix_are_explicit():
    unstable = ExponentialHawkesProcess(
        jnp.asarray([0.2]), jnp.asarray([[2.0]]), jnp.asarray([[1.0]])
    )
    observation = PointProcessObservation(
        jnp.asarray([0.5]),
        jnp.asarray([0]),
        jnp.asarray([True]),
        start_time=0.0,
        end_time=1.0,
        channel_count=1,
    )
    plan = prepare_hawkes_likelihood(observation, unstable, require_stable=True)
    assert (
        evaluate_hawkes_likelihood(observation, unstable, plan=plan).status
        == HAWKES_UNSTABLE
    )

    stable = ExponentialHawkesProcess(
        jnp.asarray([2.0]), jnp.asarray([[0.2]]), jnp.asarray([[1.0]])
    )
    small = simulate_hawkes(
        stable, jr.key(11), start_time=0.0, end_time=4.0, capacity=4, max_proposals=128
    )
    large = simulate_hawkes(
        stable, jr.key(11), start_time=0.0, end_time=4.0, capacity=12, max_proposals=128
    )
    stored = int(small.stored_event_count)
    assert jnp.array_equal(
        small.observation.times[:stored], large.observation.times[:stored]
    )
    assert jnp.array_equal(
        small.observation.channels[:stored], large.observation.channels[:stored]
    )


def test_rmt_cleaning_is_psd_trace_preserving_and_diagnosed():
    covariance = jnp.diag(jnp.asarray([0.1, 0.5, 1.0, 10.0]))

    diagnostics = marchenko_pastur_diagnostics(covariance, 100)
    cleaned = clean_covariance_spectrum(covariance, 100, preserve_trace=True)

    assert jnp.array_equal(
        diagnostics.noise_eigenvalue_mask, jnp.asarray([True, True, True, False])
    )
    assert jnp.all(jnp.linalg.eigvalsh(cleaned.covariance) >= 0.0)
    assert cleaned.psd
    assert jnp.isclose(cleaned.cleaned_trace, cleaned.raw_trace)
    assert jnp.isclose(jnp.trace(cleaned.covariance), jnp.trace(covariance))
    assert jnp.isfinite(cleaned.condition_number)
