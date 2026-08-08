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


def _rao_blackwellized_problem(*, args=None, input_signal=None):
    if args is None:
        args = {
            "initial_mean": 0.0,
            "transition_input_scale": 0.0,
            "observation_input_scale": 0.0,
        }
    if input_signal is None:
        input_signal = phx.stochastic.SampledStateSpaceInput(
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.zeros((3, 1)),
            interpolation="linear",
            input_id="rb-input",
        )
    modes = jnp.asarray([[0]])
    nonlinear_prior = phx.stochastic.CategoricalStatePrior(
        modes,
        jnp.asarray([1.0]),
        prior_id="mode-prior",
    )
    nonlinear_transition = phx.stochastic.CallableTransitionKernel(
        lambda key, state, t0, t1, context: state,
        state_shape=(1,),
        process_id="constant-mode",
        approximation_id="exact-constant-mode",
    )
    model = phx.uq.RaoBlackwellizedStateSpaceModel(
        nonlinear_prior,
        nonlinear_transition,
        lambda mode, args: (
            jnp.asarray(args["initial_mean"]).reshape((1,)),
            jnp.asarray([[1.0]]),
        ),
        lambda previous_mode, mode, t0, t1, context: (
            jnp.asarray([[1.0]]),
            context.args["transition_input_scale"] * context.transition_end_input,
            jnp.asarray([[0.1]]),
        ),
        lambda mode, time, context: (
            jnp.asarray([[1.0]]),
            context.args["observation_input_scale"] * context.observation_input,
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
        args=args,
        input_signal=input_signal,
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


def test_rao_callbacks_observe_typed_input_context():
    input_signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([[0.0], [2.0], [4.0]]),
        interpolation="linear",
        input_id="varying-rb-input",
    )
    problem = _rao_blackwellized_problem(
        args={
            "initial_mean": 5.0,
            "transition_input_scale": 2.0,
            "observation_input_scale": 3.0,
        },
        input_signal=input_signal,
    )
    context = problem.step_context(0, 0)
    initial_mean, _ = problem.model.initial_linear_gaussian(
        jnp.asarray([0]), problem.args
    )
    _, transition_offset, _ = problem.model.linear_transition_parameters(
        jnp.asarray([0]),
        jnp.asarray([0]),
        jnp.asarray(0.0),
        jnp.asarray(0.5),
        context,
    )
    _, observation_offset, _ = problem.model.observation_parameters(
        jnp.asarray([0]), jnp.asarray(0.5), context
    )
    evaluated = context.evaluate_input(jnp.asarray(0.25))

    assert isinstance(evaluated, phx.stochastic.InputEvaluation)
    assert jnp.allclose(
        jnp.concatenate(
            (initial_mean, transition_offset, observation_offset, evaluated.value)
        ),
        jnp.asarray([5.0, 4.0, 6.0, 1.0]),
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
