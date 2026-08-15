import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
import pytest

import phydrax as phx


def _constant_mode_problems():
    times = jnp.asarray([[0.5, 1.0, 1.5], [0.5, 1.0, 1.0]])
    values = jnp.asarray([[[1.0], [0.0], [1.5]], [[-0.5], [0.7], [0.0]]])
    masks = jnp.asarray([[[True], [False], [True]], [[True], [True], [False]]])
    step_valid = jnp.asarray([[True, True, True], [True, True, False]])
    observations = phx.stochastic.ObservationSequence(
        times,
        values,
        case_axes=("case",),
        case_shape=(2,),
        observation_mask=masks,
        step_valid=step_valid,
        case_ids=("long", "short"),
        sequence_id="constant-mode-sequence",
    )
    nonlinear_prior = phx.stochastic.CategoricalStatePrior(
        jnp.asarray([[0]]),
        jnp.ones((2, 1)),
        prior_id="constant-mode-prior",
    )
    nonlinear_transition = phx.stochastic.CallableTransitionKernel(
        lambda key, state, t0, t1, context: state,
        state_shape=(1,),
        process_id="constant-mode",
        approximation_id="exact-constant-mode",
        log_prob_fn=lambda next_state, state, t0, t1, context: jnp.where(
            jnp.all(next_state == state), 0.0, -jnp.inf
        ),
    )
    observation_noise = phx.uq.DiagonalCovariance(jnp.asarray([0.2]))
    rb_model = phx.uq.RaoBlackwellizedStateSpaceModel(
        nonlinear_prior,
        nonlinear_transition,
        lambda mode, args: (jnp.zeros(1), jnp.asarray([[1.0]])),
        lambda previous_mode, mode, t0, t1, context: (
            jnp.asarray([[1.0]]),
            jnp.zeros(1),
            jnp.asarray([[0.1]]),
        ),
        lambda mode, time, context: (
            jnp.asarray([[1.0]]),
            jnp.zeros(1),
            observation_noise,
        ),
        linear_state_shape=(1,),
        observation_shape=(1,),
        model_id="constant-mode-rb-model",
    )
    rb_problem = phx.uq.RaoBlackwellizedStateSpaceProblem(
        rb_model,
        observations,
        initial_time=jnp.zeros(2),
        problem_id="constant-mode-rb-problem",
    )

    kalman_model = phx.stochastic.StateSpaceModel(
        phx.stochastic.GaussianStatePrior(
            jnp.zeros((2, 1)),
            jnp.asarray([[1.0]]),
            state_shape=(1,),
            prior_id="constant-mode-linear-prior",
        ),
        phx.stochastic.LinearGaussianTransitionKernel(
            jnp.asarray([[1.0]]),
            jnp.asarray([[0.1]]),
            state_shape=(1,),
            process_id="constant-mode-linear-process",
        ),
        phx.stochastic.LinearGaussianObservationModel(
            jnp.asarray([[1.0]]),
            jnp.asarray([[0.2]]),
            state_shape=(1,),
            observation_shape=(1,),
        ),
        model_id="constant-mode-linear-model",
    )
    kalman_problem = phx.stochastic.StateSpaceProblem(
        kalman_model,
        observations,
        initial_time=jnp.zeros(2),
        problem_id="constant-mode-linear-problem",
    )
    return rb_problem, kalman_problem


def _initial_mode_dependent_problems():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[0.8], [2.2]]),
        case_ids=("only",),
        sequence_id="initial-dependent-sequence",
    )
    nonlinear_prior = phx.stochastic.CategoricalStatePrior(
        jnp.asarray([[0]]), jnp.asarray([1.0]), prior_id="initial-dependent-prior"
    )
    nonlinear_transition = phx.stochastic.CallableTransitionKernel(
        lambda key, state, t0, t1, context: state + 1,
        state_shape=(1,),
        process_id="incrementing-mode",
        approximation_id="exact-incrementing-mode",
        log_prob_fn=lambda next_state, state, t0, t1, context: jnp.where(
            jnp.all(next_state == state + 1), 0.0, -jnp.inf
        ),
    )
    rb_model = phx.uq.RaoBlackwellizedStateSpaceModel(
        nonlinear_prior,
        nonlinear_transition,
        lambda mode, args: (jnp.zeros(1), jnp.eye(1)),
        lambda previous_mode, mode, t0, t1, context: (
            jnp.eye(1),
            (previous_mode + mode).astype(float),
            jnp.asarray([[0.1]]),
        ),
        lambda mode, time, context: (jnp.eye(1), jnp.zeros(1), jnp.asarray([[0.2]])),
        linear_state_shape=(1,),
        observation_shape=(1,),
        model_id="initial-dependent-rb-model",
    )
    rb_problem = phx.uq.RaoBlackwellizedStateSpaceProblem(
        rb_model,
        observations,
        initial_time=0.0,
        problem_id="initial-dependent-rb-problem",
    )
    linear_model = phx.stochastic.StateSpaceModel(
        phx.stochastic.GaussianStatePrior(
            jnp.zeros(1), jnp.eye(1), state_shape=(1,), prior_id="initial-linear-prior"
        ),
        phx.stochastic.LinearGaussianTransitionKernel(
            jnp.eye(1),
            jnp.asarray([[0.1]]),
            state_shape=(1,),
            offset=lambda t0, t1, context: jnp.where(
                t1 == 0.5, jnp.asarray([1.0]), jnp.asarray([3.0])
            ),
            process_id="initial-linear-process",
        ),
        phx.stochastic.LinearGaussianObservationModel(
            jnp.eye(1),
            jnp.asarray([[0.2]]),
            state_shape=(1,),
            observation_shape=(1,),
        ),
        model_id="initial-dependent-linear-model",
    )
    linear_problem = phx.stochastic.StateSpaceProblem(
        linear_model,
        observations,
        initial_time=0.0,
        problem_id="initial-dependent-linear-problem",
    )
    return rb_problem, linear_problem


def _correlated_nonlinear_problem(*, normalized=True):
    covariance = jnp.asarray([[1.0, 0.94], [0.94, 1.0]])
    inverse = jnp.linalg.inv(covariance)
    log_determinant = jnp.linalg.slogdet(covariance)[1]

    def sample(key, state, t0, t1, context):
        del t0, t1, context
        return 0.55 * state + jr.multivariate_normal(key, jnp.zeros(2), covariance)

    def log_prob(next_state, state, t0, t1, context):
        del t0, t1, context
        residual = next_state - 0.55 * state
        return -0.5 * (
            residual @ inverse @ residual + log_determinant + 2.0 * jnp.log(2.0 * jnp.pi)
        )

    transition = phx.stochastic.CallableTransitionKernel(
        sample,
        state_shape=(2,),
        process_id="correlated-nonlinear",
        approximation_id="full-correlated-density",
        log_prob_fn=log_prob if normalized else None,
    )
    model = phx.uq.RaoBlackwellizedStateSpaceModel(
        phx.stochastic.GaussianStatePrior(
            jnp.zeros(2),
            jnp.eye(2),
            state_shape=(2,),
            prior_id="correlated-nonlinear-prior",
        ),
        transition,
        lambda mode, args: (jnp.zeros(1), jnp.eye(1)),
        lambda previous_mode, mode, t0, t1, context: (
            jnp.eye(1),
            jnp.zeros(1),
            jnp.asarray([[0.1]]),
        ),
        lambda mode, time, context: (jnp.eye(1), jnp.zeros(1), jnp.eye(1)),
        linear_state_shape=(1,),
        observation_shape=(1,),
        model_id="correlated-rb-model",
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.zeros((2, 1)),
        observation_mask=jnp.zeros((2, 1), dtype=bool),
        case_ids=("only",),
        sequence_id="correlated-sequence",
    )
    return phx.uq.RaoBlackwellizedStateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="correlated-rb-problem",
    )


def test_single_mode_full_smoother_matches_kalman_rts_for_masks_and_padding():
    rb_problem, kalman_problem = _constant_mode_problems()
    filtered = phx.uq.rao_blackwellized_particle_filter(
        jr.key(1), rb_problem, num_particles=4, resampling_policy="never"
    )
    result = phx.uq.rao_blackwellized_particle_smoother(
        jr.key(2), filtered, sample_shape=(3,)
    )
    expected = phx.uq.rts_smoother(
        phx.uq.kalman_filter(kalman_problem), method="sequential"
    )
    active = rb_problem.observations.step_valid
    active_mean = jnp.broadcast_to(active, result.linear_means.shape[:-1])
    active_covariance = jnp.broadcast_to(
        active[..., None, None], result.linear_covariances.shape
    )
    expected_means = jnp.broadcast_to(expected.means, result.linear_means.shape)
    expected_covariances = jnp.broadcast_to(
        expected.covariances, result.linear_covariances.shape
    )
    expected_lag_one = expected.covariances[..., 1:, :, :] @ jnp.swapaxes(
        expected.gains, -1, -2
    )

    assert jnp.allclose(
        jnp.where(active_mean[..., None], result.linear_means, 0.0),
        jnp.where(active_mean[..., None], expected_means, 0.0),
    )
    assert jnp.allclose(
        jnp.where(active_covariance, result.linear_covariances, 0.0),
        jnp.where(active_covariance, expected_covariances, 0.0),
    )
    assert jnp.allclose(
        result.lag_one_covariances,
        jnp.broadcast_to(expected_lag_one, result.lag_one_covariances.shape),
    )
    path_active = jnp.broadcast_to(
        active, result.backward_simulation.nonlinear_paths.shape[:-1]
    )
    assert jnp.all(
        jnp.where(
            path_active[..., None],
            result.backward_simulation.nonlinear_paths,
            0,
        )
        == 0
    )
    assert jnp.all(result.successful)


def test_first_conditional_transition_uses_sampled_initial_nonlinear_state():
    rb_problem, linear_problem = _initial_mode_dependent_problems()
    filtered = phx.uq.rao_blackwellized_particle_filter(
        jr.key(3), rb_problem, num_particles=4, resampling_policy="never"
    )
    result = phx.uq.rao_blackwellized_particle_smoother(
        jr.key(4), filtered, sample_shape=(2,)
    )
    expected = phx.uq.rts_smoother(phx.uq.kalman_filter(linear_problem))

    assert jnp.all(filtered.initial_nonlinear_particles == 0)
    assert jnp.all(result.backward_simulation.initial_nonlinear_states == 0)
    assert jnp.all(result.backward_simulation.nonlinear_paths[..., 0, :] == 1)
    assert jnp.allclose(
        result.linear_means,
        jnp.broadcast_to(expected.means, result.linear_means.shape),
    )
    assert jnp.allclose(
        result.linear_covariances,
        jnp.broadcast_to(expected.covariances, result.linear_covariances.shape),
    )


def test_backward_path_prefixes_are_stable_and_resampling_policies_are_coherent():
    problem, _ = _constant_mode_problems()
    never = phx.uq.rao_blackwellized_particle_filter(
        jr.key(5), problem, num_particles=4, resampling_policy="never"
    )
    always = phx.uq.rao_blackwellized_particle_filter(
        jr.key(5), problem, num_particles=4, resampling_policy="always"
    )
    short = phx.uq.rao_blackwellized_backward_simulation(
        jr.key(6), never, sample_shape=(3,)
    )
    long = phx.uq.rao_blackwellized_backward_simulation(
        jr.key(6), never, sample_shape=(5,)
    )
    smoothed_always = phx.uq.rao_blackwellized_particle_smoother(
        jr.key(7), always, sample_shape=(2,)
    )

    assert jnp.allclose(short.nonlinear_paths, long.nonlinear_paths[:3], equal_nan=True)
    assert jnp.array_equal(short.particle_indices, long.particle_indices[:3])
    assert jnp.all(short.valid)
    assert jnp.all(smoothed_always.successful)
    assert jnp.all(
        jax.lax.stop_gradient(short.particle_indices) == short.particle_indices
    )


def test_backward_simulation_uses_full_correlated_transition_density():
    problem = _correlated_nonlinear_problem()
    filtered = phx.uq.rao_blackwellized_particle_filter(
        jr.key(8), problem, num_particles=4, resampling_policy="never"
    )
    backward = phx.uq.rao_blackwellized_backward_simulation(
        jr.key(9), filtered, sample_shape=(1024,)
    )
    particles = filtered.nonlinear_particles
    previous_particles = particles[0]
    context = problem.step_context(0, 1)
    covariance_diagonal = jnp.diag(jnp.diag(jnp.asarray([[1.0, 0.94], [0.94, 1.0]])))
    diagonal_inverse = jnp.linalg.inv(covariance_diagonal)
    full_probabilities = []
    diagonal_probabilities = []
    for terminal_index in range(4):
        terminal_state = particles[1, terminal_index]
        full_log_density = jnp.asarray(
            [
                problem.model.nonlinear_transition.log_prob(
                    terminal_state,
                    previous,
                    filtered.times[0],
                    filtered.times[1],
                    context,
                )
                for previous in previous_particles
            ]
        )
        full_probabilities.append(
            jax.nn.softmax(filtered.log_weights[0] + full_log_density)
        )
        residual = terminal_state - 0.55 * previous_particles
        diagonal_log_density = -0.5 * oe.contract(
            "pi,ij,pj->p", residual, diagonal_inverse, residual
        )
        diagonal_probabilities.append(
            jax.nn.softmax(filtered.log_weights[0] + diagonal_log_density)
        )
    full_probabilities = jnp.stack(full_probabilities)
    diagonal_probabilities = jnp.stack(diagonal_probabilities)
    separations = jnp.sum(jnp.abs(full_probabilities - diagonal_probabilities), axis=-1)
    terminal = int(jnp.argmax(separations))
    full_probability = full_probabilities[terminal]
    diagonal_probability = diagonal_probabilities[terminal]
    indices = backward.particle_indices
    selected = indices[:, 2] == terminal
    empirical = jnp.bincount(indices[selected, 1], length=4) / jnp.sum(selected)
    full_error = jnp.sum(jnp.abs(empirical - full_probability))
    diagonal_error = jnp.sum(jnp.abs(empirical - diagonal_probability))

    assert jnp.sum(selected) > 150
    assert separations[terminal] > 0.15
    assert full_error < diagonal_error
    assert jnp.max(jnp.abs(empirical - full_probability)) < 0.1


def test_backward_simulation_rejects_missing_normalized_transition_density():
    filtered = phx.uq.rao_blackwellized_particle_filter(
        jr.key(10),
        _correlated_nonlinear_problem(normalized=False),
        num_particles=4,
        resampling_policy="never",
    )

    with pytest.raises(ValueError, match="normalized nonlinear transition"):
        phx.uq.rao_blackwellized_backward_simulation(jr.key(11), filtered)
