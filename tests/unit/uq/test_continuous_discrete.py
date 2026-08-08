import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _affine_problem(*, mask=None, times=None, values=None):
    resolved_times = (
        jnp.asarray([0.15, 0.6, 1.4]) if times is None else jnp.asarray(times)
    )
    resolved_values = (
        jnp.asarray([[0.4], [-0.2], [0.7]]) if values is None else jnp.asarray(values)
    )
    sequence = phx.stochastic.ObservationSequence(
        resolved_times,
        resolved_values,
        observation_mask=mask,
        sequence_id="continuous-discrete-affine-data",
        discretization_id="irregular-observation-grid",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.3, -0.1]),
        jnp.asarray([[0.8, 0.15], [0.15, 0.5]]),
        state_shape=(2,),
        prior_id="continuous-prior",
    )
    dynamics = phx.stochastic.LinearGaussianDynamics(
        jnp.asarray([[-0.4, 0.2], [-0.1, -0.25]]),
        jnp.asarray([[0.3, 0.0], [0.1, 0.2]]),
        state_shape=(2,),
        offset=jnp.asarray([0.1, -0.05]),
        dynamics_id="affine-sde",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(dynamics)
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0, -0.3]]),
        jnp.asarray([[0.2]]),
        state_shape=(2,),
        observation_shape=(1,),
        observation_id="affine-sensor",
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="continuous-affine-model",
        parameter_id="affine-parameters",
        discretization_id="exact-lti-discretization",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=-0.2,
        problem_id="continuous-affine-problem",
    )


@pytest.mark.parametrize("method", ["extended", "cubature", "unscented"])
def test_affine_continuous_discrete_oracle_and_smoother(method):
    problem = _affine_problem()
    expected_filter = phx.uq.kalman_filter(problem)
    expected_smoother = phx.uq.rts_smoother(expected_filter)

    result = phx.uq.continuous_discrete_gaussian_filter(problem, method=method)
    smoother = phx.uq.continuous_discrete_gaussian_smoother(result)

    assert jnp.allclose(
        result.predicted_means, expected_filter.predicted_means, atol=2e-10
    )
    assert jnp.allclose(
        result.predicted_covariances,
        expected_filter.predicted_covariances,
        atol=2e-10,
    )
    assert jnp.allclose(result.filtered_means, expected_filter.filtered_means, atol=2e-10)
    assert jnp.allclose(
        result.filtered_covariances,
        expected_filter.filtered_covariances,
        atol=2e-10,
    )
    assert jnp.allclose(
        result.incremental_log_likelihood,
        expected_filter.incremental_log_likelihood,
        atol=2e-10,
    )
    assert jnp.allclose(smoother.smoothed_means, expected_smoother.means, atol=3e-10)
    assert jnp.allclose(
        smoother.smoothed_covariances,
        expected_smoother.covariances,
        atol=3e-10,
    )
    assert result.method == method
    assert result.method_id == f"continuous-discrete-gaussian-filter:{method}"
    assert result.solver_id == "analytic-linear-gaussian"
    assert "van-loan" in result.transition_method
    assert result.discretization_id == "exact-lti-discretization"
    assert jnp.all(result.status == phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS)
    assert jnp.all(result.solver_status == 0)
    assert jnp.all(result.successful)


def test_irregular_typed_inputs_preserve_case_axes_and_physical_times():
    times = jnp.asarray([[0.3, 0.9], [0.2, 0.8]])
    sequence = phx.stochastic.ObservationSequence(
        times,
        jnp.zeros((2, 2, 1)),
        case_axes=("experiment",),
        case_shape=(2,),
        case_ids=("first", "second"),
        observation_mask=jnp.zeros((2, 2, 1), dtype=bool),
        sequence_id="typed-input-data",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2, 1)),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
    )
    signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([[0.0, 0.4, 1.0], [0.0, 0.5, 1.0]]),
        jnp.asarray(
            [
                [[1.0], [1.4], [2.0]],
                [[2.0], [2.5], [3.0]],
            ]
        ),
        interpolation="linear",
        input_id="case-forcing",
    )

    def drift(time, state, context):
        del state
        return context.evaluate_input(time).value

    transition = phx.stochastic.DifferentialTransitionKernel(
        drift,
        state_shape=(1,),
        process_id="input-driven-flow",
        rtol=1e-9,
        atol=1e-11,
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="input-driven-model"
    )
    problem = phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=0.0,
        problem_id="input-driven-problem",
        input_signal=signal,
    )

    result = phx.uq.continuous_discrete_gaussian_filter(problem, method="cubature")
    expected = jnp.asarray(
        [
            [0.3 + 0.5 * 0.3**2, 0.9 + 0.5 * 0.9**2],
            [2.0 * 0.2 + 0.5 * 0.2**2, 2.0 * 0.8 + 0.5 * 0.8**2],
        ]
    )

    assert result.case_shape == (2,)
    assert result.case_axes == ("experiment",)
    assert result.case_ids == ("first", "second")
    assert result.input_id == "case-forcing"
    assert jnp.array_equal(result.times, times)
    assert jnp.allclose(result.predicted_means[..., 0], expected, atol=2e-7)
    assert jnp.allclose(result.filtered_means, result.predicted_means)
    assert jnp.all(result.successful)


def test_missing_observations_are_forecast_only_and_likelihood_increments_accumulate():
    mask = jnp.asarray([[True], [False], [True]])
    result = phx.uq.continuous_discrete_gaussian_filter(
        _affine_problem(mask=mask), method="extended"
    )

    assert jnp.allclose(result.filtered_means[1], result.predicted_means[1])
    assert jnp.allclose(result.filtered_covariances[1], result.predicted_covariances[1])
    assert result.observed_counts[1] == 0
    assert result.incremental_log_likelihood[1] == 0.0
    assert jnp.allclose(
        result.cumulative_log_likelihood,
        jnp.cumsum(result.incremental_log_likelihood),
    )


def test_nonlinear_observation_uses_declared_gaussian_transform():
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5]),
        jnp.asarray([[1.2]]),
        sequence_id="quadratic-observation-data",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([1.0]),
        jnp.asarray([[0.25]]),
        state_shape=(1,),
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.0]]),
        state_shape=(1,),
    )
    observation = phx.stochastic.GaussianObservationModel(
        lambda state, time, context: state**2 + 0.0 * time + 0.0 * context.step_index,
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="quadratic-observation-model"
    )
    problem = phx.stochastic.StateSpaceProblem(
        model, sequence, initial_time=0.0, problem_id="quadratic-observation-problem"
    )

    extended = phx.uq.continuous_discrete_gaussian_filter(problem, method="extended")
    cubature = phx.uq.continuous_discrete_gaussian_filter(problem, method="cubature")
    unscented = phx.uq.continuous_discrete_gaussian_filter(problem, method="unscented")

    assert jnp.allclose(extended.predicted_observation_means[0, 0], 1.0)
    assert jnp.allclose(cubature.predicted_observation_means[0, 0], 1.25)
    assert jnp.allclose(unscented.predicted_observation_means[0, 0], 1.25)
    assert jnp.allclose(extended.predicted_observation_covariances[0, 0, 0], 1.1)
    assert jnp.allclose(cubature.predicted_observation_covariances[0, 0, 0], 1.1)
    assert jnp.allclose(unscented.predicted_observation_covariances[0, 0, 0], 1.225)
    assert extended.observation_transform_method == "first-order-jvp-vjp"
    assert cubature.observation_transform_method == "spherical-radial-cubature"
    assert unscented.observation_transform_method == "scaled-unscented"


def _differential_problem(
    *,
    max_steps=4096,
    input_signal=None,
    times=None,
    solver=None,
):
    resolved_times = jnp.asarray([0.25]) if times is None else jnp.asarray(times)
    sequence = phx.stochastic.ObservationSequence(
        resolved_times,
        jnp.zeros((resolved_times.shape[0], 1)),
        sequence_id="differential-data",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.5]),
        jnp.asarray([[0.15]]),
        state_shape=(1,),
    )

    def drift(time, state, context):
        forcing = (
            jnp.asarray([0.0])
            if context.input_signal is None
            else context.evaluate_input(time).value
        )
        return context.args * state + forcing

    transition = phx.stochastic.DifferentialTransitionKernel(
        drift,
        state_shape=(1,),
        process_id="parameterized-flow",
        max_steps=max_steps,
        rtol=1e-9,
        atol=1e-11,
        solver=solver,
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="parameterized-flow-model"
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=0.0,
        problem_id="parameterized-flow-problem",
        args=jnp.asarray(0.4),
        input_signal=input_signal,
    )


def test_nonfinite_solver_output_has_precedence_without_fallback():
    problem = _differential_problem(max_steps=1)
    problem = eqx.tree_at(
        lambda node: node.args,
        problem,
        jnp.asarray(100.0),
    )
    problem = eqx.tree_at(
        lambda node: node.observations.times,
        problem,
        jnp.asarray([10.0]),
    )
    result = phx.uq.continuous_discrete_gaussian_filter(problem, method="extended")

    assert result.status[0] == phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE
    assert result.solver_status[0] == 1
    assert not result.valid[0]
    assert result.incremental_log_likelihood[0] == -jnp.inf
    assert not result.successful


def test_jit_parameter_and_typed_input_gradients_are_supported():
    signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([0.0, 0.25]),
        jnp.asarray([[0.2], [0.4]]),
        interpolation="linear",
        input_id="differentiable-forcing",
    )
    problem = _differential_problem(input_signal=signal)
    compiled = eqx.filter_jit(
        lambda value: phx.uq.continuous_discrete_gaussian_filter(value, method="extended")
    )(problem)

    def parameter_objective(rate):
        changed = eqx.tree_at(lambda node: node.args, problem, rate)
        return phx.uq.continuous_discrete_gaussian_filter(
            changed, method="extended"
        ).cumulative_log_likelihood[-1]

    def input_objective(values):
        changed = eqx.tree_at(
            lambda node: node.input_signal.values,
            problem,
            values,
        )
        return phx.uq.continuous_discrete_gaussian_filter(
            changed, method="extended"
        ).cumulative_log_likelihood[-1]

    parameter_gradient = jax.grad(parameter_objective)(problem.args)
    input_gradient = jax.grad(input_objective)(problem.input_signal.values)

    assert jnp.all(compiled.successful)
    assert jnp.isfinite(parameter_gradient)
    assert parameter_gradient != 0.0
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.linalg.norm(input_gradient) > 0.0


def _provided_transition_problem(covariance):
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5]),
        jnp.zeros((1, 1)),
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2,)),
        0.5 * jnp.eye(2),
        state_shape=(2,),
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.eye(2),
        covariance,
        state_shape=(2,),
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(2,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="provided-transition-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=0.0,
        problem_id="provided-transition-problem",
    )


def test_incompatible_pathwise_sde_solver_is_rejected_before_dispatch():
    problem = _differential_problem(solver=dfx.ItoMilstein())

    with pytest.raises(ValueError, match="deterministic ODE-compatible"):
        phx.uq.continuous_discrete_gaussian_filter(problem)


def test_diffrax_backend_result_code_is_retained():
    signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([0.0, 5.0, 10.0]),
        jnp.zeros((3, 1)),
        interpolation="linear",
        input_id="segmented-backend-code",
    )
    problem = _differential_problem(
        max_steps=1,
        times=jnp.asarray([10.0]),
        input_signal=signal,
    )
    problem = eqx.tree_at(
        lambda node: node.args,
        problem,
        jnp.asarray(100.0),
    )

    result = phx.uq.continuous_discrete_gaussian_filter(
        problem,
        method="cubature",
    )

    assert result.solver_status[0] == dfx.RESULTS.max_steps_reached._value
    assert result.status[0] == phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE


def test_nonfinite_analytic_covariance_has_precedence_and_no_solver_code():
    problem = _provided_transition_problem(
        lambda start, end, context: jnp.full((2, 2), jnp.nan)
    )

    result = phx.uq.continuous_discrete_gaussian_filter(problem)

    assert result.status[0] == phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE
    assert result.solver_status[0] == 0
    assert not result.valid[0]


@pytest.mark.parametrize(
    "covariance",
    (
        jnp.asarray([[0.1, 0.2], [0.0, 0.1]]),
        jnp.asarray([[-1.0, 0.0], [0.0, 0.0]]),
    ),
)
def test_invalid_analytic_transition_covariance_is_a_transform_failure(covariance):
    result = phx.uq.continuous_discrete_gaussian_filter(
        _provided_transition_problem(covariance)
    )

    assert result.status[0] == phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE
    assert result.solver_status[0] == 0
    assert not result.valid[0]
    if covariance[0, 1] != covariance[1, 0]:
        assert not jnp.array_equal(
            result.predicted_covariances[0],
            result.predicted_covariances[0].T,
        )


def test_nonsymmetric_nonlinear_observation_covariance_is_not_repaired():
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5]),
        jnp.zeros((1, 2)),
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2,)),
        0.5 * jnp.eye(2),
        state_shape=(2,),
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.eye(2),
        jnp.zeros((2, 2)),
        state_shape=(2,),
    )
    observation = phx.stochastic.GaussianObservationModel(
        lambda state, time, context: state,
        jnp.asarray([[0.1, 0.2], [0.0, 0.1]]),
        state_shape=(2,),
        observation_shape=(2,),
    )
    problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="nonsymmetric-observation-model",
        ),
        sequence,
        initial_time=0.0,
        problem_id="nonsymmetric-observation-problem",
    )

    result = phx.uq.continuous_discrete_gaussian_filter(problem)

    assert result.status[0] == phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE
    assert result.solver_status[0] == 0
    assert not jnp.array_equal(
        result.predicted_observation_covariances[0],
        result.predicted_observation_covariances[0].T,
    )


def test_skipped_active_step_preserves_original_backend_provenance():
    problem = _differential_problem(
        max_steps=1,
        times=jnp.asarray([10.0, 20.0]),
    )
    problem = eqx.tree_at(
        lambda node: node.args,
        problem,
        jnp.asarray(100.0),
    )

    result = phx.uq.continuous_discrete_gaussian_filter(problem)

    expected = dfx.RESULTS.max_steps_reached._value
    assert jnp.array_equal(result.solver_status, jnp.asarray([expected, expected]))
    assert jnp.array_equal(
        result.status,
        jnp.asarray(
            [
                phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE,
                phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE,
            ]
        ),
    )


def test_backward_smoothing_failure_invalidates_every_dependent_step():
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray([0.2, 0.5, 0.9]),
        jnp.zeros((3, 1)),
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((1,)),
        jnp.zeros((1, 1)),
        state_shape=(1,),
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.ones((1, 1)),
        jnp.zeros((1, 1)),
        state_shape=(1,),
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.ones((1, 1)),
        jnp.ones((1, 1)),
        state_shape=(1,),
        observation_shape=(1,),
    )
    problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="singular-smoothing-model",
        ),
        sequence,
        initial_time=0.0,
        problem_id="singular-smoothing-problem",
    )
    filtered = phx.uq.continuous_discrete_gaussian_filter(problem)

    smoothed = phx.uq.continuous_discrete_gaussian_smoother(filtered)

    assert jnp.array_equal(smoothed.valid, jnp.asarray([False, False, True]))
    assert jnp.array_equal(
        smoothed.status[:2],
        jnp.full(
            (2,),
            phx.uq.CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE,
        ),
    )
