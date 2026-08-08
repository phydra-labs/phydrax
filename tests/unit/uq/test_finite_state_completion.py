from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


EMISSION = jnp.asarray([[0.85, 0.15], [0.25, 0.75]])


def _generator(*, forward=0.7, backward=0.4):
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray(
            [
                jnp.where(state[0] == 0, forward, 0.0),
                jnp.where(state[0] == 1, backward, 0.0),
            ]
        ),
        lambda state, channel, mark, args: jnp.where(
            channel == 0,
            jnp.ones_like(state),
            jnp.zeros_like(state),
        ),
        state_shape=(1,),
        num_channels=2,
        process_id="finite-completion-chain",
    )
    return phx.solver.finite_state_generator(
        process,
        jnp.asarray([[0], [1]]),
    )


def _observation_model(emission=EMISSION):
    def log_prob(value, state, time, mask, context):
        del time, context
        probability = emission[state[0], value[0].astype(jnp.int32)]
        exact = jnp.where(probability > 0.0, jnp.log(probability), -jnp.inf)
        return jnp.where(mask[0], exact, 0.0)

    return phx.stochastic.CallableObservationModel(
        lambda state, time, context: state.astype(float),
        log_prob,
        lambda key, state, time, sample_shape, context: jnp.broadcast_to(
            jnp.zeros((1,), dtype=jnp.int32),
            tuple(sample_shape) + (1,),
        ),
        state_shape=(1,),
        observation_shape=(1,),
        observation_id="finite-completion-observation",
    )


def _problem(
    *,
    times=jnp.asarray([0.4, 1.1, 1.8]),
    values=jnp.asarray([[0], [1], [1]]),
    probabilities=jnp.asarray([0.6, 0.4]),
    step_valid=None,
    observation_mask=None,
    generator=None,
    emission=EMISSION,
    initial_time=0.0,
    input_signal=None,
    case_ids=("only",),
):
    values = jnp.asarray(values)
    probabilities = jnp.asarray(probabilities)
    case_shape = tuple(probabilities.shape[:-1])
    case_axes = ("case",) if case_shape else ()
    sequence = phx.stochastic.ObservationSequence(
        times,
        values,
        case_axes=case_axes,
        case_shape=case_shape,
        observation_axes=("sensor",),
        step_valid=step_valid,
        observation_mask=observation_mask,
        case_ids=case_ids,
        sequence_id="finite-completion-sequence",
    )
    states = jnp.asarray([[0], [1]])
    prior = phx.stochastic.CategoricalStatePrior(
        states,
        probabilities,
        prior_id="finite-completion-prior",
    )
    transition = phx.stochastic.FiniteStateTransitionKernel(
        _generator() if generator is None else generator
    )
    model = phx.stochastic.StateSpaceModel(
        prior,
        transition,
        _observation_model(emission),
        model_id="finite-completion-model",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=initial_time,
        input_signal=input_signal,
        problem_id="finite-completion-problem",
    )


def _enumerate_single_case(problem, filter_result, case_index=None):
    prior = np.asarray(problem.model.prior.probabilities)
    matrices = np.asarray(filter_result.transition_matrices)
    values = np.asarray(problem.observations.values)
    masks = np.asarray(problem.observations.observation_mask)
    active = np.asarray(problem.observations.step_valid)
    if case_index is not None:
        prior = prior[case_index]
        matrices = matrices[case_index]
        values = values[case_index]
        masks = masks[case_index]
        active = active[case_index]
    values = np.asarray(values[:, 0], dtype=int)
    masks = np.asarray(masks[:, 0], dtype=bool)
    active_steps = int(np.sum(active))
    paths = list(product(range(2), repeat=active_steps + 1))
    masses = []
    for path in paths:
        mass = prior[path[0]]
        for step in range(active_steps):
            mass *= matrices[step, path[step], path[step + 1]]
            if masks[step]:
                mass *= float(EMISSION[path[step + 1], values[step]])
        masses.append(mass)
    masses = np.asarray(masses)
    posterior = masses / np.sum(masses)
    smoothed = np.zeros((active_steps, 2))
    initial = np.zeros((2,))
    pairwise = np.zeros((active_steps, 2, 2))
    for path, probability in zip(paths, posterior, strict=True):
        initial[path[0]] += probability
        for step in range(active_steps):
            smoothed[step, path[step + 1]] += probability
            pairwise[step, path[step], path[step + 1]] += probability
    map_index = int(np.argmax(masses))
    return initial, smoothed, pairwise, paths[map_index], masses[map_index]


def test_smoothing_viterbi_and_counts_match_complete_path_enumeration():
    problem = _problem(observation_mask=jnp.asarray([[True], [False], [True]]))
    likelihood = phx.uq.exact_state_space_log_likelihood(problem)
    filtered = likelihood.backend
    smoother = phx.uq.finite_state_backward_smoother(filtered)
    viterbi = phx.uq.finite_state_viterbi(filtered)
    counts = phx.uq.finite_state_expected_transition_counts(smoother)
    initial, marginals, pairwise, map_path, map_mass = _enumerate_single_case(
        problem, filtered
    )

    assert likelihood.successful
    assert smoother.successful
    assert viterbi.successful
    assert jnp.allclose(smoother.initial_probabilities, initial)
    assert jnp.allclose(smoother.smoothed_probabilities, marginals)
    assert jnp.allclose(smoother.transition_probabilities, pairwise)
    assert jnp.allclose(counts.per_case_counts, jnp.sum(pairwise, axis=0))
    assert jnp.allclose(counts.total_counts, counts.per_case_counts)
    assert int(viterbi.initial_state_indices) == map_path[0]
    assert tuple(np.asarray(viterbi.state_indices)) == map_path[1:]
    assert viterbi.joint_log_probability == pytest.approx(np.log(map_mass))


def test_viterbi_ties_and_zero_probability_transitions_are_exact():
    zero_generator = _generator(forward=0.0, backward=0.0)
    problem = _problem(
        generator=zero_generator,
        probabilities=jnp.asarray([0.5, 0.5]),
        observation_mask=jnp.zeros((3, 1), dtype=bool),
    )
    filtered = phx.uq.exact_state_space_log_likelihood(problem).backend
    smoother = phx.uq.finite_state_backward_smoother(filtered)
    viterbi = phx.uq.finite_state_viterbi(filtered)
    kernel = problem.model.transition
    context = phx.stochastic.StateSpaceStepContext.empty()

    assert int(viterbi.initial_state_indices) == 0
    assert jnp.array_equal(viterbi.state_indices, jnp.zeros((3,), dtype=jnp.int32))
    assert jnp.all(smoother.transition_probabilities[:, 0, 1] == 0.0)
    assert jnp.all(smoother.transition_probabilities[:, 1, 0] == 0.0)
    assert (
        kernel.log_prob(
            jnp.asarray([1]),
            jnp.asarray([0]),
            0.0,
            1.0,
            context,
        )
        == -jnp.inf
    )


def test_zero_mass_pairs_mask_nonfinite_sufficient_statistics():
    problem = _problem(
        generator=_generator(forward=0.0, backward=0.0),
        probabilities=jnp.asarray([0.5, 0.5]),
        step_valid=jnp.asarray([True, True, False]),
        observation_mask=jnp.zeros((3, 1), dtype=bool),
    )
    filtered = phx.uq.exact_state_space_log_likelihood(problem).backend
    smoother = phx.uq.finite_state_backward_smoother(filtered)

    def statistic(previous_state, state, t0, t1, context):
        del t0, t1
        finite_pair = (previous_state[0] == state[0]) & (context.step_index < 2)
        nonfinite = jnp.where(previous_state[0] < state[0], jnp.nan, jnp.inf)
        return jnp.where(finite_pair, 2.0, nonfinite)

    statistics = phx.uq.finite_state_expected_sufficient_statistics(smoother, statistic)

    assert jnp.all(jnp.isfinite(statistics.per_step_statistics))
    assert jnp.allclose(statistics.per_step_statistics, jnp.asarray([2.0, 2.0, 0.0]))
    assert statistics.per_case_statistics == pytest.approx(4.0)
    assert statistics.total_statistics == pytest.approx(4.0)


def test_masked_observation_value_has_no_effect_on_exact_posteriors():
    first = _problem(
        values=jnp.asarray([[0], [0], [1]]),
        observation_mask=jnp.asarray([[True], [False], [True]]),
    )
    second = _problem(
        values=jnp.asarray([[0], [1], [1]]),
        observation_mask=jnp.asarray([[True], [False], [True]]),
    )
    first_likelihood = phx.uq.exact_state_space_log_likelihood(first)
    second_likelihood = phx.uq.exact_state_space_log_likelihood(second)
    first_smoother = phx.uq.finite_state_backward_smoother(first_likelihood.backend)
    second_smoother = phx.uq.finite_state_backward_smoother(second_likelihood.backend)

    assert first_likelihood.total_log_likelihood == pytest.approx(
        second_likelihood.total_log_likelihood
    )
    assert jnp.allclose(
        first_smoother.smoothed_probabilities,
        second_smoother.smoothed_probabilities,
    )
    assert jnp.allclose(
        first_smoother.transition_probabilities,
        second_smoother.transition_probabilities,
    )


def test_cases_padding_ids_and_sufficient_statistic_context_are_preserved():
    times = jnp.asarray([[0.4, 1.0, 1.6], [0.5, 1.1, 1.8]])
    values = jnp.asarray([[[0], [1], [1]], [[1], [0], [1]]])
    step_valid = jnp.asarray([[True, True, True], [True, True, False]])
    mask = jnp.asarray([[[True], [False], [True]], [[True], [True], [False]]])
    signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([[0.0, 0.5, 1.2, 2.0], [0.0, 0.5, 1.2, 2.0]]),
        jnp.asarray(
            [
                [[0.0], [0.5], [1.2], [2.0]],
                [[10.0], [10.5], [11.2], [12.0]],
            ]
        ),
        interpolation="linear",
        input_id="finite-context-input",
    )
    problem = _problem(
        times=times,
        values=values,
        probabilities=jnp.asarray([[0.7, 0.3], [0.2, 0.8]]),
        step_valid=step_valid,
        observation_mask=mask,
        initial_time=jnp.asarray([0.0, 0.1]),
        input_signal=signal,
        case_ids=("first", "second"),
    )
    likelihood = phx.uq.exact_state_space_log_likelihood(problem)
    filtered = likelihood.backend
    smoother = phx.uq.finite_state_backward_smoother(filtered)
    viterbi = phx.uq.finite_state_viterbi(filtered)
    counts = phx.uq.finite_state_expected_transition_counts(smoother)

    def statistic(previous_state, state, t0, t1, context):
        return {
            "changed": (previous_state[0] != state[0]).astype(float),
            "context": jnp.stack(
                [
                    context.case_index.astype(float),
                    context.step_index.astype(float),
                    t0,
                    t1,
                    context.transition_start_input[0],
                    context.transition_end_input[0],
                ]
            ),
        }

    statistics = phx.uq.finite_state_expected_sufficient_statistics(smoother, statistic)
    for case_index in range(2):
        initial, marginals, pairwise, map_path, _ = _enumerate_single_case(
            problem,
            filtered,
            case_index,
        )
        active_steps = int(np.sum(np.asarray(step_valid[case_index])))
        assert jnp.allclose(smoother.initial_probabilities[case_index], initial)
        assert jnp.allclose(
            smoother.smoothed_probabilities[case_index, :active_steps],
            marginals,
        )
        assert jnp.allclose(
            smoother.transition_probabilities[case_index, :active_steps],
            pairwise,
        )
        assert jnp.allclose(
            counts.per_case_counts[case_index],
            jnp.sum(pairwise, axis=0),
        )
        assert int(viterbi.initial_state_indices[case_index]) == map_path[0]
        assert (
            tuple(np.asarray(viterbi.state_indices[case_index, :active_steps]))
            == map_path[1:]
        )

    assert filtered.case_shape == (2,)
    assert filtered.case_ids == ("first", "second")
    assert likelihood.input_id == "finite-context-input"
    assert filtered.input_id == "finite-context-input"
    assert smoother.input_id == "finite-context-input"
    assert viterbi.input_id == "finite-context-input"
    assert counts.input_id == "finite-context-input"
    assert statistics.input_id == "finite-context-input"
    assert jnp.all(filtered.status[step_valid] == phx.uq.EXACT_STATE_SPACE_SUCCESS)
    assert jnp.all(smoother.transition_probabilities[1, 2] == 0.0)
    assert jnp.allclose(
        smoother.smoothed_probabilities[1, 2],
        smoother.smoothed_probabilities[1, 1],
    )
    assert jnp.allclose(
        jnp.sum(counts.per_case_counts, axis=(-1, -2)),
        jnp.asarray([3.0, 2.0]),
    )
    assert jnp.allclose(
        statistics.per_case_statistics["changed"],
        counts.per_case_counts[:, 0, 1] + counts.per_case_counts[:, 1, 0],
    )
    assert jnp.allclose(
        statistics.per_step_statistics["context"][0, 0],
        jnp.asarray([0.0, 0.0, 0.0, 0.4, 0.0, 0.4]),
    )
    assert jnp.allclose(
        statistics.per_step_statistics["context"][1, 1],
        jnp.asarray([1.0, 1.0, 0.5, 1.1, 10.5, 11.1]),
    )
    assert jnp.all(statistics.per_step_statistics["context"][1, 2] == 0.0)


def test_generator_semigroup_and_exact_completion_are_jittable():
    generator = _generator()
    transition_matrix = eqx.filter_jit(generator.transition_matrix)
    first = transition_matrix(jnp.asarray(0.35))
    second = transition_matrix(jnp.asarray(0.8))
    combined = transition_matrix(jnp.asarray(1.15))
    assert jnp.allclose(first @ second, combined, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(jnp.sum(combined, axis=-1), 1.0)

    problem = _problem(generator=generator)
    likelihood = eqx.filter_jit(
        lambda candidate: phx.uq.exact_state_space_log_likelihood(
            candidate, method="finite-state"
        )
    )(problem)
    smoother = eqx.filter_jit(phx.uq.finite_state_backward_smoother)(likelihood.backend)
    viterbi = eqx.filter_jit(phx.uq.finite_state_viterbi)(likelihood.backend)
    counts = eqx.filter_jit(phx.uq.finite_state_expected_transition_counts)(smoother)
    statistics = eqx.filter_jit(
        lambda candidate: phx.uq.finite_state_expected_sufficient_statistics(
            candidate,
            lambda previous, state, t0, t1, context: jnp.stack(
                [
                    (previous[0] != state[0]).astype(float),
                    t1 - t0,
                    context.step_index.astype(float),
                ]
            ),
        )
    )(smoother)

    assert likelihood.successful
    assert smoother.successful
    assert viterbi.successful
    assert jnp.all(jnp.isfinite(counts.total_counts))
    assert jnp.all(jnp.isfinite(statistics.total_statistics))
    assert jnp.allclose(jnp.sum(smoother.smoothed_probabilities, axis=-1), 1.0)
