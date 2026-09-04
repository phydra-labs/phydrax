import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.control.games._mean_field import (
    FROZEN_LAW_BEST_RESPONSE,
    FrozenLawBestResponseProblem,
    FrozenLawBestResponseStatus,
    solve_frozen_law_best_response,
)
from phydrax.stochastic import (
    adapt_mean_field_control_bsde,
    BSDEPathBatch,
    EmpiricalMeanField,
    evaluate_bsde,
    evaluate_mean_field_bsde_control,
    MeanFieldBSDEControlAdapter,
)


def _paths() -> BSDEPathBatch:
    return BSDEPathBatch(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray(
            [
                [[0.0], [0.25], [0.5]],
                [[0.0], [-0.25], [-0.5]],
            ]
        ),
        jnp.asarray([[[0.25], [0.25]], [[-0.25], [-0.25]]]),
        sample_shape=(2,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id="candidate-paths",
        process_id="candidate-process",
    )


def _adapter() -> MeanFieldBSDEControlAdapter:
    return MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, args: law.mean - z.reshape((1,)),
        lambda time, state, law, action, args: 0.5 * action**2,
        lambda time, state, law, action, args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="frozen-law-hamiltonian",
    )


def _facade(
    particles,
    *,
    weights=None,
    valid=None,
    supplied_law_id="supplied-law",
    flow_id="flow",
):
    paths = _paths()
    law = EmpiricalMeanField(
        paths.times,
        jnp.asarray(particles),
        sample_shape=(2,),
        state_shape=(1,),
        mean_field_id=flow_id,
        weights=weights,
        valid=valid,
        source_path_id="law-source-paths",
    )
    adapter = _adapter()
    base = adapt_mean_field_control_bsde(
        lambda key: paths,
        law,
        lambda time, state, snapshot, args: jnp.zeros((1,)),
        lambda time, state, snapshot, args: jnp.ones((1, 1)),
        lambda state, snapshot, args: jnp.zeros((1,)),
        adapter,
        state_shape=(1,),
        problem_id=f"base:{flow_id}",
        process_id=paths.process_id,
    )
    facade = FrozenLawBestResponseProblem(
        base,
        adapter,
        supplied_law_id=supplied_law_id,
        problem_id=f"facade:{flow_id}",
    )
    return facade, paths, base, adapter, law


def _solve(facade, paths):
    return solve_frozen_law_best_response(
        facade,
        paths,
        lambda time, state: jnp.zeros((1,)),
        control_predictor=lambda time, state: jnp.ones((1, 1)),
        key=jr.key(7),
    )


def test_frozen_law_facade_preserves_problem_law_and_evidence_identity():
    facade, paths, base, adapter, law = _facade(
        [[[0.0], [0.0], [0.0]], [[2.0], [2.0], [2.0]]]
    )

    result = _solve(facade, paths)

    assert facade.base_problem is base
    assert facade.adapter is adapter
    assert facade.mean_field is law
    assert result.problem is facade
    assert result.bsde_evaluation.paths is paths
    assert result.paths is paths
    assert result.supplied_law_id == "supplied-law"
    assert result.flow_id == "flow"
    assert result.process_id == paths.process_id
    assert result.support == (0.0, 1.0)
    assert result.source_path_id == "law-source-paths"
    assert result.adapter_id == adapter.adapter_id
    assert result.base_problem_id == base.problem_id
    assert jnp.array_equal(result.law_weights, law.weights)
    assert jnp.array_equal(result.law_particle_validity, law.valid)


def test_frozen_law_facade_delegates_bsde_and_hamiltonian_evaluation():
    facade, paths, base, _, _ = _facade([[[0.0], [0.0], [0.0]], [[2.0], [2.0], [2.0]]])
    value = lambda time, state: jnp.zeros((1,))
    control = lambda time, state: jnp.ones((1, 1))

    result = solve_frozen_law_best_response(
        facade,
        paths,
        value,
        control_predictor=control,
        key=jr.key(3),
    )
    direct = evaluate_bsde(
        base.as_bsde_problem(),
        paths,
        value,
        control_predictor=control,
        key=jr.key(3),
    )
    direct_action = evaluate_mean_field_bsde_control(
        base,
        paths.times[0],
        paths.states[0, 0],
        direct.values[0, 0],
        direct.controls[0, 0],
    )

    assert jnp.array_equal(result.bsde_evaluation.local_residuals, direct.local_residuals)
    assert jnp.array_equal(result.hamiltonian_values, direct.generator_values)
    assert jnp.array_equal(result.selected_controls[0, 0], direct_action)
    assert result.hamiltonian_evidence.adapter_id == base.control_adapter.adapter_id
    assert result.hamiltonian_evidence.finite
    assert result.status == FrozenLawBestResponseStatus.SUCCESS
    assert result.valid


def test_frozen_law_facade_fails_closed_for_invalid_and_degenerate_law_evidence():
    particles = [[[0.0], [0.0], [0.0]], [[2.0], [2.0], [2.0]]]
    invalid, paths, _, _, _ = _facade(
        particles,
        valid=jnp.asarray([[True, True, True], [False, False, False]]),
        flow_id="invalid-flow",
    )
    degenerate, _, _, _, _ = _facade(
        particles,
        weights=jnp.asarray([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        flow_id="degenerate-flow",
    )

    invalid_result = _solve(invalid, paths)
    degenerate_result = _solve(degenerate, paths)

    assert not invalid_result.law_evidence_valid
    assert invalid_result.status == FrozenLawBestResponseStatus.INVALID_LAW_EVIDENCE
    assert not invalid_result.valid
    assert degenerate_result.law_evidence_valid
    assert jnp.allclose(degenerate_result.law_effective_sample_sizes, 1.0)
    assert not degenerate_result.effective_sample_size_sufficient
    assert (
        degenerate_result.status == FrozenLawBestResponseStatus.LOW_EFFECTIVE_SAMPLE_SIZE
    )
    assert not degenerate_result.valid


def test_frozen_law_label_is_only_a_candidate_evaluation_certificate():
    facade, paths, _, _, _ = _facade([[[0.0], [0.0], [0.0]], [[2.0], [2.0], [2.0]]])
    result = _solve(facade, paths)

    assert facade.certificate_label == FROZEN_LAW_BEST_RESPONSE
    assert result.certificate_label == "FROZEN_LAW_BEST_RESPONSE"
    assert result.candidate_evaluation_only
    assert not result.law_consistency_evaluated
    assert not result.best_response_optimality_evaluated
    assert not result.mean_field_game_equilibrium_claimed
    assert not result.mean_field_control_optimum_claimed
    assert not result.finite_population_game_claimed


def test_changing_only_supplied_frozen_law_changes_candidate_without_consistency_claim():
    low, paths, _, _, _ = _facade(
        [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]],
        supplied_law_id="law:low",
        flow_id="flow:low",
    )
    high, _, _, _, _ = _facade(
        [[[2.0], [2.0], [2.0]], [[2.0], [2.0], [2.0]]],
        supplied_law_id="law:high",
        flow_id="flow:high",
    )

    low_result = _solve(low, paths)
    high_result = _solve(high, paths)

    assert not jnp.allclose(low_result.selected_controls, high_result.selected_controls)
    assert low_result.supplied_law_id != high_result.supplied_law_id
    assert low_result.flow_id != high_result.flow_id
    assert not low_result.law_consistency_evaluated
    assert not high_result.law_consistency_evaluated


def test_frozen_law_facade_rejects_unsupported_semantics_before_prediction():
    facade, paths, base, _, _ = _facade([[[0.0], [0.0], [0.0]], [[2.0], [2.0], [2.0]]])
    calls = []

    def predictor(time, state):
        calls.append((time, state))
        return jnp.zeros((1,))

    mismatched = BSDEPathBatch(
        paths.times,
        paths.states,
        paths.wiener_increments,
        sample_shape=paths.sample_shape,
        state_shape=paths.state_shape,
        noise_shape=paths.noise_shape,
        path_id="wrong-process-paths",
        process_id="wrong-process",
    )
    with pytest.raises(ValueError, match="process IDs"):
        solve_frozen_law_best_response(
            facade,
            mismatched,
            predictor,
            control_predictor=lambda time, state: jnp.ones((1, 1)),
        )
    assert calls == []
    assert base.process_id != mismatched.process_id
