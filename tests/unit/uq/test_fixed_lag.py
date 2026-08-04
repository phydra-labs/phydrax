import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _problem():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0, 1.5]),
        jnp.asarray([[0.5], [1.0], [1.5]]),
        case_ids=("only",),
        sequence_id="fixed-lag-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="fixed-lag-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="fixed-lag-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="fixed-lag-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="fixed-lag-problem",
    )


def test_full_lag_kalman_smoother_matches_batch_rts_and_zero_lag_filter():
    filtered = phx.uq.kalman_filter(_problem())
    zero = phx.uq.fixed_lag_kalman_smoother(filtered, 0)
    full = phx.uq.fixed_lag_kalman_smoother(filtered, 10)
    expected = phx.uq.rts_smoother(filtered)

    assert jnp.allclose(zero.means, filtered.filtered_means)
    assert jnp.allclose(zero.covariances, filtered.filtered_covariances)
    assert jnp.allclose(full.means, expected.means)
    assert jnp.allclose(full.covariances, expected.covariances)
    assert jnp.array_equal(full.horizons, jnp.asarray([2, 2, 2]))
    assert jnp.all(full.valid)


def test_particle_fixed_lag_weights_are_normalized_and_trace_genealogy():
    filtered = phx.uq.bootstrap_particle_filter(
        jr.key(9),
        _problem(),
        num_particles=64,
        resampling_policy="always",
    )
    zero = phx.uq.fixed_lag_particle_smoother(filtered, 0)
    full = phx.uq.fixed_lag_particle_smoother(filtered, 10)

    assert jnp.allclose(zero.log_weights, filtered.log_weights)
    assert jnp.allclose(jnp.sum(jnp.exp(full.log_weights), axis=-1), 1.0)
    assert jnp.array_equal(full.horizons, jnp.asarray([2, 2, 2]))
    assert jnp.array_equal(full.lineage_indices[-1], jnp.arange(64))
    assert full.means.shape == (3, 1)
    assert jnp.all(full.valid)
