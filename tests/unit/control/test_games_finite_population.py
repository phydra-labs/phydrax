from functools import cache

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.control.games._finite_population import (
    evaluate_finite_population_continuation,
    FINITE_POPULATION_CONTINUATION_EVALUATION,
    FINITE_POPULATION_EPSILON_NASH_EVIDENCE,
    FinitePopulationBestResponseEvidence,
    FinitePopulationContinuationPlan,
    FinitePopulationContinuationStatus,
    FinitePopulationGameProblem,
    FinitePopulationJointPolicyEvaluation,
)
from phydrax.control.games._mean_field import (
    FrozenLawBestResponseProblem,
    solve_frozen_law_best_response,
)
from phydrax.control.games._mean_field_fixed_point import (
    MeanFieldGameFixedPointPlan,
    MeanFieldGameFixedPointProblem,
    solve_mean_field_game_fixed_point,
)
from phydrax.stochastic import (
    adapt_mean_field_control_bsde,
    BSDEPathBatch,
    EmpiricalMeanField,
    MeanFieldBSDEControlAdapter,
)


def _law(value, law_id, source_id, *, population=2):
    return EmpiricalMeanField(
        jnp.asarray([0.0, 1.0]),
        jnp.full((population, 2, 1), value),
        sample_shape=(population,),
        state_shape=(1,),
        mean_field_id=law_id,
        source_path_id=source_id,
    )


def _response(flow):
    paths = BSDEPathBatch(
        flow.times,
        jnp.zeros((2, 2, 1)),
        jnp.zeros((2, 1, 1)),
        sample_shape=(2,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id=f"response-paths:{flow.mean_field_id}",
        process_id="finite-population-test-process",
    )
    adapter = MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, args: -z.reshape((1,)),
        lambda time, state, law, action, args: 0.5 * action**2,
        lambda time, state, law, action, args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="finite-population-test-adapter",
    )
    base = adapt_mean_field_control_bsde(
        lambda key: paths,
        flow,
        lambda time, state, law, args: jnp.zeros((1,)),
        lambda time, state, law, args: jnp.ones((1, 1)),
        lambda state, law, args: jnp.zeros((1,)),
        adapter,
        state_shape=(1,),
        problem_id=f"finite-population-base:{flow.mean_field_id}",
        process_id=paths.process_id,
    )
    frozen = FrozenLawBestResponseProblem(
        base,
        adapter,
        supplied_law_id=f"supplied:{flow.mean_field_id}",
        problem_id=f"frozen:{flow.mean_field_id}",
    )
    return solve_frozen_law_best_response(
        frozen,
        paths,
        lambda time, state: jnp.zeros((1,)),
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
        key=jr.key(7),
    )


@cache
def _fixed_point():
    initial = _law(0.0, "mfg-law", "mfg-paths")
    problem = MeanFieldGameFixedPointProblem(
        initial,
        lambda flow, args: _response(flow),
        lambda response, args: _law(0.0, "induced-law", "induced-paths"),
        lambda current, induced, args: jnp.asarray(0.0),
        best_response_id="mfg-best-response",
        induced_flow_id="mfg-independent-forward-law",
        law_distance_id="mfg-law-distance",
        problem_id="analytic-mfg",
    )
    plan = MeanFieldGameFixedPointPlan(
        maximum_iterations=1,
        consistency_tolerance=0.0,
        minimum_effective_sample_size=2.0,
        problem_id="analytic-mfg",
    )
    return solve_mean_field_game_fixed_point(problem, plan)


def _best_response(
    player,
    value,
    *,
    numerical=0.0,
    statistical=0.0,
    valid=True,
    method="exact",
    clusters=1,
    simultaneous=True,
):
    def callback(joint, args):
        return FinitePopulationBestResponseEvidence(
            value,
            player_index=player,
            numerical_error_bound=numerical,
            statistical_error_bound=statistical,
            best_response_id=f"br:{player}",
            feasible_deviation_id=f"feasible:{player}",
            deviation_policy_id=f"deviation-policy:{player}",
            coverage_method=method,
            confidence=0.95,
            independent_cluster_count=clusters,
            simultaneous=simultaneous,
            valid=valid,
            certified=valid,
            failure_reason=None if valid else "solver did not certify its result",
        )

    return callback


def _case(
    costs,
    responses,
    *,
    epsilon=10.0,
    finite_law_value=0.0,
    law_tolerance=0.0,
    method="exact",
    statistically_exact=True,
    labels=None,
    reuse_mfg_law=False,
):
    costs = jnp.asarray(costs, dtype=float)
    players, paths = costs.shape
    policy_ids = tuple(f"policy:{player}" for player in range(players))
    path_ids = tuple(f"finite-path:{path}" for path in range(paths))
    if labels is None:
        labels = jnp.arange(paths)
    evaluation = FinitePopulationJointPolicyEvaluation(
        costs,
        path_ids=path_ids,
        cluster_labels=labels,
        coupling_id="finite-coupling",
        policy_ids=policy_ids,
        evaluation_id="finite-joint-evaluator",
        statistically_exact=statistically_exact,
    )
    problem = FinitePopulationGameProblem(
        _fixed_point(),
        players,
        policy_ids,
        lambda fixed_point, args: evaluation,
        responses,
        lambda joint, args: (
            _fixed_point().flow
            if reuse_mfg_law
            else _law(
                finite_law_value,
                f"finite-law:N={players}",
                f"finite-path-batch:N={players}",
                population=players,
            )
        ),
        lambda finite, mfg, args: jnp.max(
            jnp.abs(
                jnp.stack([finite.snapshot(time).mean for time in finite.times])
                - jnp.stack([mfg.snapshot(time).mean for time in mfg.times])
            )
        ),
        joint_profile_evaluator_id="finite-joint-evaluator",
        best_response_ids=tuple(f"br:{player}" for player in range(players)),
        feasible_deviation_ids=tuple(f"feasible:{player}" for player in range(players)),
        finite_law_builder_id="independent-finite-law-builder",
        law_distance_id="node-mean-distance",
        problem_id=f"analytic-finite-game:N={players}",
    )
    plan = FinitePopulationContinuationPlan(
        epsilon=epsilon,
        law_tolerance=law_tolerance,
        confidence=0.95,
        coverage_method=method,
        minimum_clusters=1,
        problem_id=problem.problem_id,
    )
    return problem, plan


def test_analytic_finite_game_computes_minimizer_exploitability_and_provenance():
    problem, plan = _case(
        [[4.0, 4.0], [3.0, 3.0]],
        [_best_response(0, 1.0), _best_response(1, 3.0)],
        epsilon=3.0,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    assert result.status == FinitePopulationContinuationStatus.SUCCESS
    np.testing.assert_allclose(result.profile_values, [4.0, 3.0])
    np.testing.assert_allclose(result.best_response_values, [1.0, 3.0])
    np.testing.assert_allclose(result.exploitabilities, [3.0, 0.0])
    assert result.epsilon_upper_bound == 3.0
    assert result.certificate_label == FINITE_POPULATION_EPSILON_NASH_EVIDENCE
    assert result.path_ids == ("finite-path:0", "finite-path:1")
    assert result.coupling_id == "finite-coupling"
    assert result.policy_ids == ("policy:0", "policy:1")
    assert result.mfg_law_id == "mfg-law"
    assert result.finite_law_id == "finite-law:N=2"
    assert result.finite_source_path_id == "finite-path-batch:N=2"


def test_deliberately_exploitable_profile_proves_sign_is_not_reversed():
    problem, plan = _case(
        [[9.0], [1.0]],
        [_best_response(0, 2.0), _best_response(1, 2.0)],
        epsilon=6.9,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    np.testing.assert_allclose(result.raw_improvements, [7.0, -1.0])
    np.testing.assert_allclose(result.exploitabilities, [7.0, 0.0])
    assert result.status == FinitePopulationContinuationStatus.EPSILON_EXCEEDED
    assert not result.epsilon_nash_claimed


def test_exact_finite_equilibrium_has_zero_exploitability():
    problem, plan = _case(
        [[2.0], [5.0]],
        [_best_response(0, 2.0), _best_response(1, 5.0)],
        epsilon=0.0,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    np.testing.assert_allclose(result.exploitability_upper_bounds, 0.0)
    assert result.epsilon_upper_bound == 0.0
    assert result.valid


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        (
            [None, _best_response(1, 1.0)],
            FinitePopulationContinuationStatus.MISSING_BEST_RESPONSE,
        ),
        (
            [_best_response(0, 1.0, valid=False), _best_response(1, 1.0)],
            FinitePopulationContinuationStatus.FAILED_BEST_RESPONSE,
        ),
    ],
)
def test_missing_or_failed_best_response_fails_closed(responses, expected):
    problem, plan = _case([[1.0], [1.0]], responses)

    result = evaluate_finite_population_continuation(problem, plan)

    assert result.status == expected
    assert not result.valid
    assert not result.all_best_responses_valid
    assert result.certificate_label == FINITE_POPULATION_CONTINUATION_EVALUATION


def test_numerical_and_simultaneous_statistical_errors_are_added():
    problem, plan = _case(
        [[4.0], [2.0]],
        [
            _best_response(0, 3.0, numerical=0.2, statistical=0.3),
            _best_response(1, 2.0, numerical=0.1, statistical=0.4),
        ],
        epsilon=1.5,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    np.testing.assert_allclose(result.exploitabilities, [1.0, 0.0])
    np.testing.assert_allclose(result.exploitability_upper_bounds, [1.5, 0.5])
    np.testing.assert_allclose(result.epsilon_upper_bound, 1.5)
    assert result.all_bounds_available
    assert result.status == FinitePopulationContinuationStatus.SUCCESS


def test_finite_empirical_law_mismatch_blocks_nash_label():
    problem, plan = _case(
        [[1.0], [1.0]],
        [_best_response(0, 1.0), _best_response(1, 1.0)],
        finite_law_value=0.25,
        law_tolerance=0.1,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    assert result.status == FinitePopulationContinuationStatus.LAW_MISMATCH
    np.testing.assert_allclose(result.law_distance, 0.25)
    assert not result.law_matches
    assert not result.epsilon_nash_claimed


def test_reusing_the_mfg_law_is_not_finite_population_evidence():
    problem, plan = _case(
        [[1.0], [1.0]],
        [_best_response(0, 1.0), _best_response(1, 1.0)],
        reuse_mfg_law=True,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    assert result.status == FinitePopulationContinuationStatus.INVALID_FINITE_LAW
    assert not result.finite_law_valid
    assert not result.epsilon_nash_claimed


@pytest.mark.parametrize("population_size", [2, 5])
def test_population_scaling_metadata_retains_every_player(population_size):
    problem, plan = _case(
        jnp.zeros((population_size, 2)),
        [_best_response(player, 0.0) for player in range(population_size)],
        epsilon=0.0,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    assert result.population_size == population_size
    assert len(result.policy_ids) == population_size
    assert len(result.best_response_ids) == population_size
    assert len(result.feasible_deviation_ids) == population_size
    assert result.exploitability_upper_bounds.shape == (population_size,)
    assert f"N={population_size}" in result.finite_law_id


def test_no_epsilon_claim_without_every_numerical_and_statistical_bound():
    incomplete = [
        _best_response(0, 1.0, statistical=None),
        _best_response(1, 1.0),
    ]
    problem, plan = _case([[1.0], [1.0]], incomplete, epsilon=100.0)

    result = evaluate_finite_population_continuation(problem, plan)

    assert result.status == FinitePopulationContinuationStatus.INCOMPLETE_BOUNDS
    assert not result.all_bounds_available
    assert jnp.isnan(result.epsilon_upper_bound)
    assert not result.valid
    assert not result.finite_population_game_claimed
    assert result.certificate_label != FINITE_POPULATION_EPSILON_NASH_EVIDENCE


def test_non_simultaneous_or_wrong_coverage_evidence_is_incomplete():
    responses = [
        _best_response(0, 0.0, method="asymptotic-normal", clusters=8),
        _best_response(
            1,
            0.0,
            method="asymptotic-normal",
            clusters=8,
            simultaneous=False,
        ),
    ]
    problem, plan = _case(
        [[0.0, 0.0], [0.0, 0.0]],
        responses,
        method="asymptotic-normal",
        statistically_exact=False,
    )

    result = evaluate_finite_population_continuation(problem, plan)

    assert result.status == FinitePopulationContinuationStatus.INCOMPLETE_BOUNDS
    assert not result.bound_available[1]
    assert result.coverage_method == "asymptotic-normal"


def test_profile_values_average_declared_independent_clusters():
    problem, plan = _case(
        [[0.0, 2.0, 10.0], [4.0, 8.0, 2.0]],
        [_best_response(0, 5.5), _best_response(1, 4.0)],
        epsilon=0.0,
        labels=jnp.asarray([0, 0, 1]),
    )

    result = evaluate_finite_population_continuation(problem, plan)

    # Equal cluster weighting gives ((0 + 2) / 2 + 10) / 2 = 5.5,
    # rather than the path-weighted value 4.
    np.testing.assert_allclose(result.profile_values, [5.5, 4.0])
    assert result.valid
