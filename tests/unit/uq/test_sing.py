import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
import pytest

import phydrax as phx


def _linear_problems():
    drift = jnp.asarray([[-0.35, 0.12], [-0.08, -0.22]])
    dispersion = jnp.asarray([[0.32, 0.04], [0.0, 0.27]])
    prior_mean = jnp.asarray([0.15, -0.2])
    prior_covariance = jnp.asarray([[0.7, 0.12], [0.12, 0.55]])
    observation_matrix = jnp.asarray([[1.0, -0.25], [0.15, 0.8]])
    observation_covariance = jnp.asarray([[0.18, 0.03], [0.03, 0.24]])
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.25, 0.7, 1.15]),
        jnp.asarray([[0.3, -0.1], [0.05, 0.4], [-0.2, 0.15]]),
        sequence_id="sing-linear-observations",
    )
    prior = phx.stochastic.GaussianStatePrior(
        prior_mean,
        prior_covariance,
        state_shape=(2,),
        prior_id="sing-linear-prior",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        observation_matrix,
        observation_covariance,
        state_shape=(2,),
        observation_shape=(2,),
        observation_id="sing-linear-observation",
    )
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: drift @ state,
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="sing-linear-system",
    )
    noise = phx.solver.WienerTerm(
        "sing-linear-noise",
        lambda time, state, args: dispersion,
        (2,),
        structure="additive",
        basis_id="sing-linear-basis",
    )
    euler_transition = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(2,),
        noise_shape=(2,),
        process_id="sing-linear-euler",
    )
    euler_problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            euler_transition,
            observation,
            model_id="sing-linear-euler-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="sing-linear-euler-problem",
    )
    identity = jnp.eye(2)
    process_covariance = dispersion @ dispersion.T
    linear_transition = phx.stochastic.LinearGaussianTransitionKernel(
        lambda start, end, context: identity + (end - start) * drift,
        lambda start, end, context: (end - start) * process_covariance,
        state_shape=(2,),
        process_id="sing-linear-reference",
    )
    linear_problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            linear_transition,
            observation,
            model_id="sing-linear-reference-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="sing-linear-reference-problem",
    )
    return euler_problem, linear_problem


def _nonlinear_problem():
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: -0.3 * state - 0.08 * state**3,
        state_layout=phx.dynamics.StateLayout((1,)),
        system_id="sing-nonlinear-system",
    )
    noise = phx.solver.WienerTerm(
        "sing-nonlinear-noise",
        lambda time, state, args: jnp.full((1, 1), 0.35),
        (1,),
        structure="additive",
        basis_id="sing-nonlinear-basis",
    )
    transition = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(1,),
        noise_shape=(1,),
        process_id="sing-nonlinear-process",
    )
    observation = phx.stochastic.GaussianObservationModel(
        lambda state, time, context: state + 0.12 * state**3,
        jnp.asarray([[0.16]]),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="sing-nonlinear-observation",
    )
    sequence = phx.stochastic.ObservationSequence(
        jnp.asarray([0.2, 0.55, 0.95]),
        jnp.asarray([[0.5], [0.15], [-0.25]]),
        sequence_id="sing-nonlinear-sequence",
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            phx.stochastic.GaussianStatePrior(
                jnp.asarray([0.1]),
                jnp.asarray([[0.8]]),
                state_shape=(1,),
                prior_id="sing-nonlinear-prior",
            ),
            transition,
            observation,
            model_id="sing-nonlinear-model",
        ),
        sequence,
        initial_time=0.0,
        problem_id="sing-nonlinear-problem",
    )


def _masked_case_problem():
    case_shape = (2,)
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: -0.2 * state,
        state_layout=phx.dynamics.StateLayout((1,)),
        system_id="sing-masked-system",
    )
    noise = phx.solver.WienerTerm(
        "sing-masked-noise",
        lambda time, state, args: jnp.full((1, 1), 0.25),
        (1,),
        structure="additive",
        basis_id="sing-masked-basis",
    )
    transition = phx.stochastic.EulerMaruyamaTransitionKernel(
        system,
        (noise,),
        state_shape=(1,),
        noise_shape=(1,),
        process_id="sing-masked-process",
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([[0.0, 0.4, 0.9], [0.0, 0.6, 0.6]]),
        jnp.asarray([[[0.1], [0.3], [-0.2]], [[-0.1], [0.25], [0.0]]]),
        case_axes=("trial",),
        case_shape=case_shape,
        case_ids=("left", "right"),
        step_valid=jnp.asarray([[True, True, True], [True, True, False]]),
        observation_mask=jnp.asarray(
            [[[True], [False], [True]], [[True], [True], [False]]]
        ),
        sequence_id="sing-masked-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([[0.0], [0.2]]),
        jnp.broadcast_to(jnp.asarray([[0.6]]), case_shape + (1, 1)),
        state_shape=(1,),
        prior_id="sing-masked-prior",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="sing-masked-observation",
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="sing-masked-model",
        ),
        observations,
        initial_time=jnp.zeros(case_shape),
        problem_id="sing-masked-problem",
    )


def test_linear_gaussian_one_step_matches_exact_smoother_elbo_and_gradients(tmp_path):
    problem, reference_problem = _linear_problems()
    initial = phx.uq.initialize_sing(problem)
    update = phx.uq.sing_step(problem, initial, max_backtracks=0)
    result = phx.uq.sing_smoother(
        problem,
        max_iterations=1,
        max_backtracks=0,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
    )
    filtered = phx.uq.kalman_filter(reference_problem, method="sequential")
    reference = phx.uq.rts_smoother(filtered, method="sequential")

    assert update.accepted
    assert update.valid
    assert jnp.allclose(update.accepted_step_size, 1.0)
    assert jnp.allclose(result.observation_means, reference.means, atol=2e-9, rtol=2e-9)
    assert jnp.allclose(
        result.observation_covariances,
        reference.covariances,
        atol=2e-9,
        rtol=2e-9,
    )
    assert jnp.allclose(
        result.elbo.total_elbo,
        jnp.sum(filtered.incremental_log_likelihood),
        atol=2e-9,
        rtol=2e-9,
    )
    assert result.converged
    assert result.status == phx.uq.SING_SUCCESS
    exported = phx.uq.export_result(result, tmp_path / "sing.phxresult")
    archive = phx.uq.read_result_archive(exported)
    assert archive.kind == "sing_smoother"
    assert archive.metadata["problem_id"] == problem.problem_id
    assert archive.metadata["expectation_method"] == "cubature"
    assert "state.expectation_key" in archive.excluded
    assert jnp.array_equal(archive.array("means"), result.means)
    assert jnp.array_equal(
        archive.array("information.diagonal_precision"),
        result.state.information.diagonal_precision,
    )

    compiled = jax.jit(
        lambda current: phx.uq.sing_step(problem, current, max_backtracks=0)
    )(initial)
    assert compiled.valid
    assert jnp.allclose(compiled.moments.means, update.moments.means)

    def fixed_posterior_objective(offset):
        candidate = eqx.tree_at(
            lambda value: value.model.observation.offset,
            problem,
            offset,
        )
        return phx.uq.sing_elbo(candidate, result.state).total_elbo

    offset_gradient = jax.grad(fixed_posterior_objective)(
        problem.model.observation.offset
    )
    assert jnp.all(jnp.isfinite(offset_gradient))
    assert jnp.linalg.norm(offset_gradient) > 0.0


def test_sing_samples_are_coherent_and_recover_posterior_moments():
    problem, _ = _linear_problems()
    result = phx.uq.sing_smoother(problem, max_iterations=1, max_backtracks=0)
    samples = phx.uq.sample_sing_paths(jr.key(902), result, sample_shape=(8192,))
    sample_mean = jnp.mean(samples, axis=0)
    centered = samples - sample_mean
    sample_covariance = oe.contract("toi,toj->oij", centered, centered) / samples.shape[0]
    sample_cross = (
        oe.contract("toi,toj->oij", centered[:, :-1], centered[:, 1:]) / samples.shape[0]
    )
    observation_nodes = result.state.grid.observation_node_indices
    reference_cross = result.transition_cross_covariances[observation_nodes[:-1]]

    assert samples.shape == (8192, 3, 2)
    assert jnp.allclose(sample_mean, result.observation_means, atol=3.5e-2)
    assert jnp.allclose(sample_covariance, result.observation_covariances, atol=4.5e-2)
    assert jnp.allclose(sample_cross, reference_cross, atol=4.5e-2)


def test_nonlinear_sing_is_monotone_and_monte_carlo_is_reproducible():
    problem = _nonlinear_problem()
    result = phx.uq.sing_smoother(
        problem,
        max_iterations=5,
        max_backtracks=5,
        acceptance_tolerance=1e-9,
    )
    increments = jnp.diff(result.elbo_history)

    assert result.valid
    assert jnp.all(jnp.isfinite(result.elbo_history))
    assert jnp.all(increments >= -1e-8)
    assert jnp.all(result.step_size_history >= 0.0)
    assert result.observation_means.shape == (3, 1)

    first = phx.uq.initialize_sing(
        problem,
        key=jr.key(71),
        expectation_method="monte-carlo",
        num_samples=32,
    )
    repeated = phx.uq.initialize_sing(
        problem,
        key=jr.key(71),
        expectation_method="monte-carlo",
        num_samples=32,
    )
    changed = phx.uq.initialize_sing(
        problem,
        key=jr.key(72),
        expectation_method="monte-carlo",
        num_samples=32,
    )
    assert jnp.array_equal(
        first.information.diagonal_precision,
        repeated.information.diagonal_precision,
    )
    assert not jnp.array_equal(
        first.information.information_vector,
        changed.information.information_vector,
    )


def test_case_masks_padding_statuses_and_model_guards_are_explicit():
    problem = _masked_case_problem()
    state = phx.uq.initialize_sing(problem)
    result = phx.uq.sing_smoother(
        problem, state=state, max_iterations=2, max_backtracks=2
    )

    assert jnp.array_equal(
        state.grid.node_valid,
        jnp.asarray([[True, True, True, False], [True, True, False, False]]),
    )
    assert jnp.array_equal(
        state.grid.observation_node_indices,
        jnp.asarray([[0, 1, 2], [0, 1, 1]], dtype=jnp.int32),
    )
    assert result.valid.shape == (2,)
    assert jnp.all(result.valid)
    assert result.observation_means.shape == (2, 3, 1)
    assert result.elbo.expected_observation_log_density[0, 1] == 0.0
    assert result.elbo.expected_observation_log_density[1, 2] == 0.0

    failed = phx.uq.sing_step(problem, state, step_size=-1.0, max_backtracks=1)
    assert not jnp.any(failed.accepted)
    assert jnp.all(failed.status == phx.uq.SING_LINE_SEARCH_FAILURE)

    invalid_noise = phx.solver.WienerTerm(
        "undeclared-noise",
        lambda time, value, args: jnp.full((1, 1), 0.2),
        (1,),
    )
    invalid_transition = phx.stochastic.EulerMaruyamaTransitionKernel(
        problem.model.transition.system,
        (invalid_noise,),
        state_shape=(1,),
        noise_shape=(1,),
        process_id="undeclared-process",
    )
    invalid_problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            problem.model.prior,
            invalid_transition,
            problem.model.observation,
            model_id="undeclared-model",
        ),
        problem.observations,
        initial_time=problem.initial_time,
        problem_id="undeclared-problem",
    )
    with pytest.raises(ValueError, match="structure='additive'"):
        phx.uq.initialize_sing(invalid_problem)
    with pytest.raises(ValueError, match="key is required"):
        phx.uq.initialize_sing(problem, expectation_method="monte-carlo")
