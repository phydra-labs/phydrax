import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _problem(*, num_steps=3, transition_value=0.8, has_density=True):
    times = jnp.linspace(0.5, 0.5 * num_steps, num_steps)
    values = jnp.linspace(0.4, 1.2, num_steps)[:, None]
    observations = phx.stochastic.ObservationSequence(
        times,
        values,
        case_ids=("only",),
        sequence_id="particle-smoothing-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.asarray([0.0]),
        jnp.asarray([[0.7]]),
        state_shape=(1,),
        prior_id="particle-smoothing-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[transition_value]]),
        jnp.asarray([[0.3]]),
        state_shape=(1,),
        process_id="particle-smoothing-process",
        approximation_id="particle-smoothing-transition",
        has_log_density=has_density,
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.15]]),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="particle-smoothing-observation",
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="particle-smoothing-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="particle-smoothing-problem",
    )


def _masked_case_problem():
    times = jnp.asarray([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]])
    values = jnp.asarray([[[0.2], [0.5], [0.9]], [[-0.1], [0.3], [0.0]]])
    step_valid = jnp.asarray([[True, True, True], [True, True, False]])
    observations = phx.stochastic.ObservationSequence(
        times,
        values,
        case_axes=("experiment",),
        case_shape=(2,),
        step_valid=step_valid,
        observation_mask=step_valid[..., None],
        case_ids=("case-a", "case-b"),
        sequence_id="masked-particle-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2, 1)),
        jnp.asarray([[0.5]]),
        state_shape=(1,),
        prior_id="masked-particle-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        process_id="masked-particle-process",
        approximation_id="masked-particle-transition",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="masked-particle-observation",
    )
    input_signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([[0.0, 0.5, 1.0, 1.5], [0.0, 0.5, 1.0, 1.5]]),
        jnp.zeros((2, 4, 1)),
        interpolation="linear",
        input_id="masked-particle-input",
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="masked-particle-model",
        ),
        observations,
        initial_time=jnp.zeros((2,)),
        input_signal=input_signal,
        problem_id="masked-particle-problem",
    )


def _initial_pair_probabilities(result, smoother):
    count = result.num_particles
    initial_particles = jnp.stack(
        [
            result.problem.model.prior.sample(
                phx.stochastic.state_space_key(
                    result.final_state.root_key,
                    "particle-filter-prior",
                    result.case_ids[0],
                    0,
                    member=particle_index,
                )
            )
            for particle_index in range(count)
        ]
    )
    transition = result.problem.model.transition
    density = jnp.stack(
        [
            jnp.stack(
                [
                    transition.log_prob(
                        result.predicted_particles[0, next_index],
                        initial_particles[previous_index],
                        result.problem.initial_time,
                        result.times[0],
                        result.problem.step_context(0, 0),
                    )
                    for previous_index in range(count)
                ]
            )
            for next_index in range(count)
        ]
    )
    backward = density - jax.scipy.special.logsumexp(density, axis=-1)[:, None]
    predicted_weights = jax.scipy.special.logsumexp(
        jnp.where(
            result.ancestor_indices[0][None, :]
            == jnp.arange(count, dtype=jnp.int32)[:, None],
            smoother.log_weights[0][None, :],
            -jnp.inf,
        ),
        axis=-1,
    )
    return (
        initial_particles,
        jax.lax.stop_gradient(jnp.exp(predicted_weights[:, None] + backward)),
    )


def _fixed_smoothing_transition_objective(value, result, smoother):
    transition = eqx.tree_at(
        lambda kernel: kernel.parameterization.transition,
        result.problem.model.transition,
        jnp.asarray([[value]]),
    )
    initial_particles, initial_pairs = _initial_pair_probabilities(result, smoother)
    total = 0.0
    for next_index in range(result.num_particles):
        for previous_index in range(result.num_particles):
            total = total + initial_pairs[
                next_index, previous_index
            ] * transition.log_prob(
                result.predicted_particles[0, next_index],
                initial_particles[previous_index],
                result.problem.initial_time,
                result.times[0],
                result.problem.step_context(0, 0),
            )
    pair_probabilities = jax.lax.stop_gradient(jnp.exp(smoother.pair_log_weights))
    for step in range(result.problem.observations.num_steps - 1):
        for next_index in range(result.num_particles):
            for previous_index in range(result.num_particles):
                total = total + pair_probabilities[
                    step, next_index, previous_index
                ] * transition.log_prob(
                    result.particles[step + 1, next_index],
                    result.particles[step, previous_index],
                    result.times[step],
                    result.times[step + 1],
                    result.problem.step_context(0, step + 1),
                )
    return total


@pytest.mark.parametrize("method", ["systematic", "multinomial"])
def test_resampling_ancestry_is_the_exact_post_resampling_genealogy(method):
    result = phx.uq.bootstrap_particle_filter(
        jr.key(1),
        _problem(),
        num_particles=24,
        resampling_method=method,
        resampling_policy="always",
    )

    for step in range(result.problem.observations.num_steps):
        assert jnp.array_equal(
            result.particles[step],
            result.predicted_particles[step, result.ancestor_indices[step]],
        )

    smoother = phx.uq.full_particle_smoother(result)
    terminal = result.problem.observations.num_steps - 1
    lineage = jnp.arange(result.num_particles, dtype=jnp.int32)
    for step in range(terminal, 0, -1):
        lineage = result.ancestor_indices[step, lineage]
    assert jnp.array_equal(smoother.lineage_indices[0], lineage)
    expected = jax.scipy.special.logsumexp(
        jnp.where(
            lineage[None, :] == jnp.arange(result.num_particles)[:, None],
            result.log_weights[terminal][None, :],
            -jnp.inf,
        ),
        axis=-1,
    )
    expected = expected - jax.scipy.special.logsumexp(expected)
    assert jnp.allclose(smoother.log_weights[0], expected)
    assert smoother.ancestry_gradient == "stop"


def test_no_resampling_paths_keep_identity_genealogy():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(2),
        _problem(),
        num_particles=16,
        resampling_policy="never",
    )
    identity = jnp.arange(result.num_particles, dtype=jnp.int32)
    assert jnp.array_equal(
        result.ancestor_indices,
        jnp.broadcast_to(identity, result.ancestor_indices.shape),
    )

    smoother = phx.uq.full_particle_smoother(result)
    assert jnp.array_equal(
        smoother.lineage_indices,
        jnp.broadcast_to(identity, smoother.lineage_indices.shape),
    )
    assert jnp.allclose(
        smoother.log_weights,
        jnp.broadcast_to(result.log_weights[-1], smoother.log_weights.shape),
    )
    paths = phx.uq.sample_particle_ancestry_paths(jr.key(3), result, sample_shape=(8,))
    for path in paths:
        candidate = jnp.all(result.particles == path[:, None, :], axis=(0, 2))
        assert jnp.any(candidate)


def test_backward_probabilities_match_direct_enumeration():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(4),
        _problem(),
        num_particles=7,
        resampling_policy="never",
    )
    smoother = phx.uq.particle_backward_smoother(result)
    transition = result.problem.model.transition
    step = 0
    rows = []
    for next_index in range(result.num_particles):
        row = []
        for previous_index in range(result.num_particles):
            row.append(
                result.log_weights[step, previous_index]
                + transition.log_prob(
                    result.particles[step + 1, next_index],
                    result.particles[step, previous_index],
                    result.times[step],
                    result.times[step + 1],
                    result.problem.step_context(0, step + 1),
                )
            )
        row = jnp.stack(row)
        rows.append(row - jax.scipy.special.logsumexp(row))
    expected = jnp.stack(rows)

    assert jnp.allclose(smoother.backward_log_probabilities[step], expected)
    assert jnp.allclose(
        jnp.sum(jnp.exp(smoother.backward_log_probabilities), axis=-1), 1.0
    )
    assert jnp.allclose(jnp.sum(jnp.exp(smoother.pair_log_weights), axis=(-1, -2)), 1.0)

    simulation = phx.uq.particle_backward_simulation(
        jr.key(5), result, sample_shape=(12,)
    )
    gathered = jnp.take_along_axis(
        result.particles[None, ...],
        simulation.particle_indices[..., None, None],
        axis=2,
    )[..., 0, :]
    assert jnp.array_equal(simulation.paths, gathered)
    assert simulation.ancestry_gradient == "stop"


def test_full_smoothing_is_not_the_fixed_lag_zero_approximation():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(6),
        _problem(),
        num_particles=32,
        resampling_policy="always",
    )
    full = phx.uq.full_particle_smoother(result)
    fixed = phx.uq.fixed_lag_particle_smoother(result, 0)

    assert jnp.array_equal(full.horizons, jnp.full((3,), 2, dtype=jnp.int32))
    assert jnp.array_equal(fixed.horizons, jnp.arange(3, dtype=jnp.int32))
    assert not jnp.allclose(full.log_weights[0], fixed.log_weights[0])
    assert full.method_id == "full-particle-ancestry"


def test_density_methods_reject_density_free_transitions_without_fallback():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(7),
        _problem(has_density=False),
        num_particles=8,
        resampling_policy="never",
    )
    genealogy = phx.uq.full_particle_smoother(result)
    assert jnp.all(genealogy.valid)

    with pytest.raises(ValueError, match="normalized transition density"):
        phx.uq.particle_backward_smoother(result)
    with pytest.raises(ValueError, match="normalized transition density"):
        phx.uq.particle_backward_simulation(jr.key(8), result)
    with pytest.raises(ValueError, match="normalized transition density"):
        phx.uq.sample_particle_backward_paths(jr.key(8), result)


def test_particle_fisher_score_matches_fixed_smoothing_finite_difference():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(9),
        _problem(transition_value=0.8),
        num_particles=10,
        resampling_policy="never",
    )
    smoother = phx.uq.particle_backward_smoother(result)
    score = phx.uq.particle_fisher_score(smoother)

    def fixed_smoothing_objective(value):
        return _fixed_smoothing_transition_objective(value, result, smoother)

    automatic = score.transition_score.parameterization.transition[0, 0]
    finite_difference = (
        fixed_smoothing_objective(0.8 + 1e-3) - fixed_smoothing_objective(0.8 - 1e-3)
    ) / (2e-3)
    assert jnp.allclose(automatic, finite_difference, rtol=2e-3, atol=2e-3)

    fisher = phx.uq.particle_fisher_information(smoother)
    assert fisher.information.shape == (score.parameter_size, score.parameter_size)
    assert jnp.allclose(fisher.information, fisher.information.T)
    assert jnp.all(jnp.linalg.eigvalsh(fisher.information) >= -1e-8)


def test_one_observation_fisher_score_counts_initial_transition_once():
    result = phx.uq.bootstrap_particle_filter(
        jr.key(91),
        _problem(num_steps=1, transition_value=0.8),
        num_particles=12,
        resampling_policy="never",
    )
    smoother = phx.uq.particle_backward_smoother(result)
    score = phx.uq.particle_fisher_score(smoother)
    fisher = phx.uq.particle_fisher_information(smoother)

    automatic = score.transition_score.parameterization.transition[0, 0]
    finite_difference = (
        _fixed_smoothing_transition_objective(0.8 + 1e-3, result, smoother)
        - _fixed_smoothing_transition_objective(0.8 - 1e-3, result, smoother)
    ) / (2e-3)

    assert jnp.abs(automatic) > 1e-3
    assert jnp.allclose(automatic, finite_difference, rtol=2e-3, atol=2e-3)
    assert jnp.any(jnp.abs(fisher.information) > 1e-6)


def test_zero_mass_singular_transition_pairs_have_finite_fisher_score():
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[0.4], [0.8]]),
        case_ids=("singular",),
        sequence_id="singular-particle-sequence",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2,)),
        jnp.eye(2),
        state_shape=(2,),
        prior_id="singular-particle-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[0.8, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0.3, 0.0], [0.0, 0.0]]),
        state_shape=(2,),
        process_id="singular-particle-process",
        approximation_id="singular-particle-transition",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([[0.15]]),
        state_shape=(2,),
        observation_shape=(1,),
        observation_id="singular-particle-observation",
    )
    problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior,
            transition,
            observation,
            model_id="singular-particle-model",
        ),
        observations,
        initial_time=0.0,
        problem_id="singular-particle-problem",
    )
    result = phx.uq.bootstrap_particle_filter(
        jr.key(92),
        problem,
        num_particles=8,
        resampling_policy="never",
    )
    smoother = phx.uq.particle_backward_smoother(result)
    score = phx.uq.particle_fisher_score(smoother)

    assert jnp.any(jnp.isneginf(smoother.pair_log_weights))
    assert jnp.all(jnp.isfinite(score.flat_score))
    assert jnp.all(jnp.isfinite(score.case_scores))


def test_full_genealogy_stays_invalid_after_filter_failure():
    base = _problem()

    def fail_middle_step(key, state, t0, t1, context):
        del key, t0, context
        return jnp.where(t1 == 1.0, jnp.full_like(state, jnp.nan), state)

    transition = phx.stochastic.CallableTransitionKernel(
        fail_middle_step,
        state_shape=(1,),
        process_id="locally-recovering-particle-process",
        approximation_id="locally-recovering-particle-transition",
    )
    problem = phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            base.model.prior,
            transition,
            base.model.observation,
            model_id="locally-recovering-particle-model",
        ),
        base.observations,
        initial_time=base.initial_time,
        problem_id="locally-recovering-particle-problem",
    )
    result = phx.uq.bootstrap_particle_filter(
        jr.key(93),
        problem,
        num_particles=6,
        resampling_policy="never",
    )
    smoother = phx.uq.full_particle_smoother(result)

    assert jnp.array_equal(result.valid, jnp.asarray([True, False, True]))
    assert not bool(result.final_state.valid)
    assert not jnp.any(smoother.valid)


def test_checkpoint_resume_preserves_next_ancestry(tmp_path):
    problem = _problem()
    state = phx.uq.initialize_particle_filter(
        jr.key(10),
        problem,
        num_particles=20,
        resampling_method="multinomial",
        resampling_policy="always",
    )
    state, _ = phx.uq.particle_filter_step(problem, state)
    destination = tmp_path / "particle-smoothing-checkpoint.phxckpt"
    phx.uq.write_particle_filter_checkpoint(destination, problem, state)
    restored = phx.uq.read_particle_filter_checkpoint(
        destination,
        problem,
        num_particles=20,
        resampling_method="multinomial",
        resampling_policy="always",
    )

    direct_state, direct_step = phx.uq.particle_filter_step(problem, state)
    resumed_state, resumed_step = phx.uq.particle_filter_step(problem, restored)
    assert jnp.array_equal(direct_step.ancestor_indices, resumed_step.ancestor_indices)
    assert jnp.array_equal(direct_step.particles, resumed_step.particles)
    assert jnp.array_equal(direct_state.root_key, resumed_state.root_key)


def test_masks_cases_provenance_and_result_export(tmp_path):
    problem = _masked_case_problem()
    result = phx.uq.bootstrap_particle_filter(
        jr.key(11),
        problem,
        num_particles=12,
        resampling_policy="never",
    )
    genealogy = phx.uq.full_particle_smoother(result)
    backward = phx.uq.particle_backward_smoother(result)

    for smoother in (genealogy, backward):
        assert smoother.case_shape == (2,)
        assert smoother.case_axes == ("experiment",)
        assert smoother.case_ids == ("case-a", "case-b")
        assert smoother.problem_id == problem.problem_id
        assert smoother.sequence_id == problem.observations.sequence_id
        assert smoother.input_id == problem.input_signal.input_id
        assert jnp.array_equal(smoother.step_valid, problem.observations.step_valid)
        assert not smoother.valid[1, 2]
        assert jnp.isnan(smoother.means[1, 2]).all()

    simulation = phx.uq.particle_backward_simulation(
        jr.key(12), result, sample_shape=(5,)
    )
    assert jnp.all(simulation.particle_indices[:, 1, 2] == -1)
    assert jnp.isnan(simulation.paths[:, 1, 2]).all()

    archive_path = phx.uq.export_result(
        backward, tmp_path / "particle-backward-smoother.phxresult"
    )
    archive = phx.uq.read_result_archive(archive_path)
    assert archive.kind == "particle_backward_smoother"
    assert archive.metadata["problem_id"] == problem.problem_id
    assert archive.metadata["sequence_id"] == problem.observations.sequence_id
    assert archive.metadata["input_id"] == problem.input_signal.input_id
    assert jnp.array_equal(
        archive.array("backward_log_probabilities"),
        backward.backward_log_probabilities,
    )


def test_resampling_indices_have_zero_forward_sensitivity():
    weights = jnp.log(jnp.asarray([0.6, 0.3, 0.1]))
    for method in ("systematic", "multinomial"):
        _, tangent = jax.jvp(
            lambda values: phx.uq.resample_indices(
                jr.key(13), values, method=method
            ).astype(float),
            (weights,),
            (jnp.ones_like(weights),),
        )
        assert jnp.array_equal(tangent, jnp.zeros_like(tangent))
