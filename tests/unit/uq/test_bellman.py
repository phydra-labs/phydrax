import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _scalar_linear_problem(*, values=None, mask=None, step_valid=None, problem_id="scalar"):
    if values is None:
        values = jnp.asarray([[0.4], [-0.2]])
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        values,
        observation_mask=mask,
        step_valid=step_valid,
        case_ids=("only",),
        sequence_id=f"{problem_id}-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.1]),
        jnp.asarray([[0.7]]),
        state_shape=(1,),
        prior_id=f"{problem_id}-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        offset=jnp.asarray([0.05]),
        process_id=f"{problem_id}-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.2]]),
        jnp.asarray([[0.3]]),
        state_shape=(1,),
        observation_shape=(1,),
        offset=jnp.asarray([-0.1]),
        observation_id=f"{problem_id}-observation",
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior, transition, observation, model_id=f"{problem_id}-model"
        ),
        observations,
        initial_time=0.0,
        problem_id=problem_id,
    )


def _time_varying_masked_problem():
    times = jnp.asarray([[0.4, 0.9, 1.5], [0.4, 0.9, 0.9]])
    values = jnp.asarray(
        [
            [[0.5, -0.1], [0.2, 0.7], [-0.3, 0.4]],
            [[-0.2, 0.1], [0.8, -0.4], [0.0, 0.0]],
        ]
    )
    mask = jnp.asarray(
        [
            [[True, True], [True, False], [False, False]],
            [[True, True], [False, True], [False, False]],
        ]
    )
    observations = phx.stochastic.ObservationSequence(
        times,
        values,
        case_axes=("case",),
        case_shape=(2,),
        observation_mask=mask,
        step_valid=jnp.asarray([[True, True, True], [True, True, False]]),
        case_ids=("first", "second"),
        sequence_id="time-varying-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([[0.1, -0.2], [-0.3, 0.4]]),
        jnp.asarray([[0.8, 0.15], [0.15, 0.6]]),
        state_shape=(2,),
        prior_id="time-varying-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        lambda t0, t1, context: jnp.asarray(
            [[0.75 + 0.05 * t1, 0.1], [-0.05, 0.85 - 0.03 * t0]]
        ),
        lambda t0, t1, context: jnp.asarray(
            [[0.18 + 0.02 * (t1 - t0), 0.03], [0.03, 0.14]]
        ),
        state_shape=(2,),
        offset=lambda t0, t1, context: jnp.asarray([0.02 * t1, -0.01 * t0]),
        process_id="time-varying-process",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        lambda time, context: jnp.asarray(
            [[1.0, 0.15 * time], [-0.1 * time, 0.9]]
        ),
        lambda time, context: jnp.asarray(
            [[0.25 + 0.02 * time, 0.04], [0.04, 0.3]]
        ),
        state_shape=(2,),
        observation_shape=(2,),
        observation_id="time-varying-observation",
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="time-varying-model"
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=jnp.zeros(2),
        problem_id="time-varying-problem",
    )


def _poisson_problem():
    base = _scalar_linear_problem(values=jnp.asarray([[3.0], [2.0]]), problem_id="poisson")

    def log_prob(value, state, time, mask, context):
        del time, context
        terms = value * state[0] - jnp.exp(state[0]) - jax.scipy.special.gammaln(
            value + 1.0
        )
        return jnp.sum(jnp.where(mask, terms, 0.0))

    observation = phx.stochastic.CallableObservationModel(
        lambda state, time, context: jnp.asarray([jnp.exp(state[0])]),
        log_prob,
        lambda key, state, time, sample_shape, context: jnp.broadcast_to(
            jnp.asarray([jnp.exp(state[0])]), sample_shape + (1,)
        ),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="poisson-log-link",
    )
    model = phx.stochastic.StateSpaceModel(
        base.model.prior,
        base.model.transition,
        observation,
        model_id="poisson-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        base.observations,
        initial_time=base.initial_time,
        problem_id="poisson-problem",
    )


def _state_dependent_transition_problem():
    beta = 0.4

    def variance(state):
        return jnp.exp(beta * state[0])

    def sample(key, state, t0, t1, context):
        del t0, t1, context
        return state + jnp.sqrt(variance(state)) * jr.normal(key, state.shape)

    def log_prob(next_state, state, t0, t1, context):
        del t0, t1, context
        residual = next_state[0] - state[0]
        q = variance(state)
        return -0.5 * (residual**2 / q + jnp.log(2.0 * jnp.pi * q))

    transition = phx.stochastic.CallableTransitionKernel(
        sample,
        state_shape=(1,),
        process_id="heteroscedastic-process",
        approximation_id="normalized-heteroscedastic",
        log_prob_fn=log_prob,
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([1.0]),
        jnp.zeros((1, 1)),
        observation_mask=jnp.zeros((1, 1), dtype=bool),
        case_ids=("only",),
        sequence_id="heteroscedastic-sequence",
    )
    model = phx.stochastic.StateSpaceModel(
        phx.stochastic.GaussianStatePrior(
            jnp.zeros(1),
            jnp.eye(1),
            state_shape=(1,),
            prior_id="heteroscedastic-prior",
        ),
        transition,
        phx.stochastic.LinearGaussianObservationModel(
            jnp.eye(1),
            jnp.eye(1),
            state_shape=(1,),
            observation_shape=(1,),
        ),
        model_id="heteroscedastic-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model, observations, initial_time=0.0, problem_id="heteroscedastic-problem"
    )


def _nonconcave_observation_problem():
    base = _scalar_linear_problem(
        values=jnp.asarray([[1.0], [1.0]]), problem_id="curvature"
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros(1),
        jnp.asarray([[0.7]]),
        state_shape=(1,),
        prior_id="nonconcave-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.eye(1),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        process_id="nonconcave-process",
    )

    def log_prob(value, state, time, mask, context):
        del time, context
        residual = value - state[0] ** 2
        terms = -0.5 * (residual**2 + jnp.log(2.0 * jnp.pi))
        return jnp.sum(jnp.where(mask, terms, 0.0))

    observation = phx.stochastic.CallableObservationModel(
        lambda state, time, context: jnp.asarray([state[0] ** 2]),
        log_prob,
        lambda key, state, time, sample_shape, context: jnp.broadcast_to(
            jnp.asarray([state[0] ** 2]), sample_shape + (1,)
        ),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="nonconcave-observation",
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="nonconcave-model",
        ),
        base.observations,
        initial_time=0.0,
        problem_id="nonconcave-problem",
    )


def test_linear_gaussian_exact_limit_matches_kalman_and_rts_with_masks():
    problem = _time_varying_masked_problem()
    bellman = phx.uq.bellman_filter(problem, method="analytic")
    kalman = phx.uq.kalman_filter(problem, method="sequential")
    bellman_smooth = phx.uq.bellman_smoother(bellman)
    kalman_smooth = phx.uq.rts_smoother(kalman, method="sequential")

    assert bellman.execution_method == "analytic"
    assert jnp.allclose(bellman.filtered_modes, kalman.filtered_means)
    assert jnp.allclose(bellman.filtered_covariances, kalman.filtered_covariances)
    assert jnp.allclose(bellman.predicted_modes, kalman.predicted_means)
    assert jnp.allclose(bellman.predicted_covariances, kalman.predicted_covariances)
    assert jnp.allclose(
        bellman.cumulative_pseudo_log_likelihood,
        kalman.cumulative_log_likelihood,
    )
    assert jnp.allclose(bellman_smooth.modes, kalman_smooth.means)
    assert jnp.allclose(bellman_smooth.covariances, kalman_smooth.covariances)
    assert jnp.allclose(bellman_smooth.gains, kalman_smooth.gains)
    assert bellman.observed_counts[0, 1] == 1
    assert bellman.observed_counts[0, 2] == 0
    assert jnp.allclose(bellman.filtered_modes[1, 2], bellman.filtered_modes[1, 1])
    assert jnp.all(bellman.successful)


def test_forced_optimization_matches_the_exact_linear_gaussian_engine():
    problem = _scalar_linear_problem()
    exact = phx.uq.bellman_filter(problem, method="analytic")
    optimized = phx.uq.bellman_filter(problem, method="optimization")

    assert optimized.execution_method == "optimization"
    assert jnp.all(optimized.prediction_converged)
    assert jnp.all(optimized.update_converged)
    assert jnp.allclose(optimized.filtered_modes, exact.filtered_modes, atol=2e-6)
    assert jnp.allclose(
        optimized.filtered_covariances, exact.filtered_covariances, atol=2e-6
    )
    assert jnp.allclose(
        optimized.cumulative_pseudo_log_likelihood,
        exact.cumulative_pseudo_log_likelihood,
        atol=2e-6,
    )


def test_streaming_steps_reproduce_batch_histories():
    problem = _scalar_linear_problem()
    batch = phx.uq.bellman_filter(problem)
    state = phx.uq.initialize_bellman_filter(problem)
    records = []
    for _ in range(problem.observations.num_steps):
        state, record = phx.uq.bellman_filter_step(problem, state)
        records.append(record)

    assert jnp.allclose(
        jnp.stack([record.filtered_mode for record in records]), batch.filtered_modes
    )
    assert jnp.allclose(state.mode, batch.final_state.mode)
    assert jnp.allclose(
        state.pseudo_log_likelihood, batch.final_state.pseudo_log_likelihood
    )


def test_poisson_update_solves_stationarity_and_reports_observed_curvature():
    result = phx.uq.bellman_filter(_poisson_problem(), method="optimization")
    mode = result.filtered_modes[..., 0]
    predicted_mode = result.predicted_modes[..., 0]
    predicted_information = result.predicted_information[..., 0, 0]
    expected_score = predicted_information * (mode - predicted_mode) + jnp.exp(mode)
    observations = jnp.asarray([3.0, 2.0])

    assert jnp.allclose(expected_score, observations, atol=2e-6)
    assert jnp.allclose(
        result.filtered_information[..., 0, 0],
        predicted_information + jnp.exp(mode),
        atol=2e-6,
    )
    assert jnp.all(result.successful)


def test_normalized_state_dependent_transition_uses_log_determinant_and_schur_profile():
    problem = _state_dependent_transition_problem()
    result = phx.uq.bellman_filter(problem, method="optimization")
    joint_mode = jnp.concatenate(
        [result.revised_previous_modes[0], result.predicted_modes[0]]
    )
    previous_information = jnp.eye(1)
    context = problem.step_context(0, 0)

    def objective(joint):
        previous = joint[:1]
        current = joint[1:]
        return 0.5 * previous @ previous_information @ previous - problem.model.transition.log_prob(
            current, previous, 0.0, 1.0, context
        )

    hessian = jax.hessian(objective)(joint_mode)
    expected = hessian[1:, 1:] - hessian[1:, :1] @ jnp.linalg.solve(
        hessian[:1, :1], hessian[:1, 1:]
    )

    assert jnp.allclose(result.revised_previous_modes[0, 0], -0.2, atol=2e-6)
    assert jnp.allclose(result.predicted_modes[0, 0], -0.2, atol=2e-6)
    assert jnp.allclose(result.predicted_information[0], expected, atol=2e-6)
    assert result.observed_counts[0] == 0
    assert result.incremental_pseudo_log_likelihood[0] == 0.0


def test_curvature_failure_is_visible_and_declared_damping_repairs_it():
    problem = _nonconcave_observation_problem()
    failed = phx.uq.bellman_filter(problem, method="optimization")
    damped = phx.uq.bellman_filter(
        problem, method="optimization", curvature_damping=2.0
    )

    assert failed.status[0] == phx.uq.BELLMAN_UPDATE_CURVATURE_FAILURE
    assert not failed.mode_valid[0]
    assert jnp.all(damped.mode_valid)
    assert damped.curvature_damping == 2.0
    assert jnp.all(jnp.linalg.eigvalsh(damped.filtered_information) > 0.0)


def test_solver_failure_freezes_state_and_dimension_and_density_guards_reject():
    problem = _scalar_linear_problem()
    failed = phx.uq.bellman_filter(
        problem, method="optimization", optimizer_max_steps=1
    )

    assert failed.status[0] == phx.uq.BELLMAN_INITIALIZATION_OPTIMIZER_FAILURE
    assert jnp.allclose(failed.filtered_modes[0], problem.model.prior.location)
    with pytest.raises(ValueError, match="max_dimension"):
        phx.uq.bellman_filter(problem, max_dimension=0)

    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.eye(1),
        jnp.eye(1),
        state_shape=(1,),
        has_log_density=False,
    )
    no_density = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            problem.model.prior,
            transition,
            problem.model.observation,
            model_id="no-density-model",
        ),
        problem.observations,
        initial_time=0.0,
        problem_id="no-density-problem",
    )
    with pytest.raises(ValueError, match="normalized transition"):
        phx.uq.bellman_filter(no_density, method="optimization")


def test_bellman_pseudo_likelihood_gradient_matches_central_difference():
    def objective(value):
        problem = _scalar_linear_problem(
            values=jnp.asarray([[value], [-0.2]]), problem_id="gradient"
        )
        return phx.uq.bellman_filter(
            problem, method="optimization"
        ).cumulative_pseudo_log_likelihood[-1]

    point = jnp.asarray(0.4)
    automatic = jax.grad(objective)(point)
    step = 1e-4
    finite_difference = (objective(point + step) - objective(point - step)) / (2.0 * step)

    assert jnp.isfinite(automatic)
    assert jnp.allclose(automatic, finite_difference, rtol=2e-3, atol=2e-4)
