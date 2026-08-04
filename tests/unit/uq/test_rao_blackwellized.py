import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _observations():
    return phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]),
        case_ids=("only",),
        sequence_id="rb-sequence",
    )


def _rao_blackwellized_problem():
    modes = jnp.asarray([[0]])
    nonlinear_prior = phx.stochastic.CategoricalStatePrior(
        modes,
        jnp.asarray([1.0]),
        prior_id="mode-prior",
    )
    nonlinear_transition = phx.stochastic.CallableTransitionKernel(
        lambda key, state, t0, t1: state,
        state_shape=(1,),
        process_id="constant-mode",
        approximation_id="exact-constant-mode",
    )
    model = phx.uq.RaoBlackwellizedStateSpaceModel(
        nonlinear_prior,
        nonlinear_transition,
        lambda mode, args: (jnp.asarray([0.0]), jnp.asarray([[1.0]])),
        lambda previous_mode, mode, t0, t1, args: (
            jnp.asarray([[1.0]]),
            jnp.asarray([0.0]),
            jnp.asarray([[0.1]]),
        ),
        lambda mode, time, args: (
            jnp.asarray([[1.0]]),
            jnp.asarray([0.0]),
            jnp.asarray([[0.2]]),
        ),
        linear_state_shape=(1,),
        observation_shape=(1,),
        model_id="conditionally-linear",
    )
    return phx.uq.RaoBlackwellizedStateSpaceProblem(
        model,
        _observations(),
        initial_time=0.0,
        problem_id="rb-problem",
    )


def _kalman_problem():
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="linear-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="linear-process",
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
        model_id="linear-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        _observations(),
        initial_time=0.0,
        problem_id="linear-problem",
    )


def test_single_mode_rao_blackwellized_filter_matches_exact_kalman_filter():
    result = phx.uq.rao_blackwellized_particle_filter(
        jr.key(5),
        _rao_blackwellized_problem(),
        num_particles=8,
        resampling_policy="never",
    )
    expected = phx.uq.kalman_filter(_kalman_problem())

    assert result.successful
    assert jnp.allclose(
        result.linear_means,
        jnp.broadcast_to(expected.filtered_means[:, None, :], (2, 8, 1)),
    )
    assert jnp.allclose(
        result.linear_covariances,
        jnp.broadcast_to(expected.filtered_covariances[:, None, :, :], (2, 8, 1, 1)),
    )
    assert jnp.allclose(
        result.final_state.log_likelihood,
        expected.final_state.log_likelihood,
    )
    assert jnp.allclose(jnp.exp(result.log_weights), 1.0 / 8.0)
    assert jnp.all(result.nonlinear_particles == 0)
