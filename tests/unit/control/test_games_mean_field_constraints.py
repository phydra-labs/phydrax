#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.control._trajectory_optimization import (
    BoundedTrajectoryConstraint,
    TrajectoryOptimizationView,
)
from phydrax.control.games._constraints import (
    evaluate_game_feasibility,
    GameConstraintBlock,
    GameConstraintScope,
    GameConstraintSite,
    OpenLoopGameConstraints,
)
from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._mean_field import (
    FrozenLawBestResponseProblem,
    solve_frozen_law_best_response,
)
from phydrax.control.games._mean_field_constraints import (
    CONSTRAINED_MEAN_FIELD_GAME_KKT_CANDIDATE,
    ConstrainedMeanFieldGamePlan,
    ConstrainedMeanFieldGameProblem,
    ConstrainedMeanFieldGameStatus,
    MeanFieldAggregateConstraintDerivativeEvidence,
    MeanFieldConstraintConcept,
    MeanFieldIndividualConstraintEvidence,
    solve_constrained_mean_field_game,
)
from phydrax.control.games._mean_field_fixed_point import (
    MeanFieldGameFixedPointProblem,
    MeanFieldGameFixedPointStatus,
)
from phydrax.stochastic import (
    adapt_mean_field_control_bsde,
    BSDEPathBatch,
    EmpiricalMeanField,
    MeanFieldBSDEControlAdapter,
)


def _law(
    value: float,
    flow_id: str,
    source_path_id: str,
    *,
    weights=None,
    sample_shape=(2,),
) -> EmpiricalMeanField:
    return EmpiricalMeanField(
        jnp.asarray([0.0, 1.0]),
        jnp.full(sample_shape + (2, 1), value),
        sample_shape=sample_shape,
        state_shape=(1,),
        mean_field_id=flow_id,
        weights=weights,
        source_path_id=source_path_id,
    )


def _law_means(flow: EmpiricalMeanField):
    return jnp.stack([flow.snapshot(time).mean[0] for time in flow.times])


def _response(flow: EmpiricalMeanField):
    path_id = f"best-response-evaluation:{flow.mean_field_id}"
    paths = BSDEPathBatch(
        flow.times,
        jnp.zeros(flow.sample_shape + (2, 1)),
        jnp.zeros(flow.sample_shape + (1, 1)),
        sample_shape=flow.sample_shape,
        state_shape=(1,),
        noise_shape=(1,),
        path_id=path_id,
        process_id="one-period-process",
    )
    adapter = MeanFieldBSDEControlAdapter(
        lambda time, state, law, value, z, args: -z.reshape((1,)),
        lambda time, state, law, action, args: 0.5 * action**2,
        lambda time, state, law, action, args: action,
        control_shape=(1,),
        output_shape=(1,),
        noise_shape=(1,),
        adapter_id="analytic-best-response",
    )
    base = adapt_mean_field_control_bsde(
        lambda key: paths,
        flow,
        lambda time, state, law, args: jnp.zeros((1,)),
        lambda time, state, law, args: jnp.ones((1, 1)),
        lambda state, law, args: jnp.zeros((1,)),
        adapter,
        state_shape=(1,),
        problem_id=f"base:{flow.mean_field_id}",
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
        key=jr.key(19),
        minimum_effective_sample_size=2.0,
    )


def _normalised_weights(flow: EmpiricalMeanField):
    weights = flow.weights.reshape((flow.num_particles, flow.times.size))
    return weights / jnp.sum(weights, axis=0, keepdims=True)


def _exact_law_mixture(current, induced, damping, iteration, args):
    del args
    time_count = current.times.size
    particles = jnp.concatenate(
        (
            current.particles.reshape(
                (current.num_particles, time_count) + current.state_shape
            ),
            induced.particles.reshape(
                (induced.num_particles, time_count) + induced.state_shape
            ),
        )
    )
    weights = jnp.concatenate(
        (
            (1.0 - damping) * _normalised_weights(current),
            damping * _normalised_weights(induced),
        )
    )
    valid = jnp.concatenate(
        (
            current.valid.reshape((current.num_particles, time_count)),
            induced.valid.reshape((induced.num_particles, time_count)),
        )
    )
    return EmpiricalMeanField(
        current.times,
        particles,
        sample_shape=(current.num_particles + induced.num_particles,),
        state_shape=current.state_shape,
        mean_field_id=(
            f"constraint-union-mixture:{iteration}:"
            f"{current.mean_field_id}+{induced.mean_field_id}"
        ),
        weights=weights,
        valid=valid,
        source_path_id=None,
    )


def _fixed_point_problem(
    initial,
    induced_value,
    *,
    induced_weights=None,
    law_mixture=None,
):
    def induced(response, args):
        return _law(
            induced_value,
            f"induced:{response.flow_id}",
            f"new-forward-paths:{response.flow_id}",
            weights=induced_weights,
            sample_shape=response.mean_field.sample_shape,
        )

    return MeanFieldGameFixedPointProblem(
        initial,
        lambda flow, args: _response(flow),
        induced,
        lambda current, candidate, args: jnp.max(
            jnp.abs(_law_means(current) - _law_means(candidate))
        ),
        best_response_id="analytic-frozen-best-response",
        induced_flow_id="independent-forward-law",
        law_mixture=law_mixture,
        law_mixture_id=(
            "exact-constraint-union-law-mixture" if law_mixture is not None else None
        ),
        law_distance_id="maximum-node-mean-distance",
        problem_id="one-period-mean-field-game",
    )


def _block(
    constraint_id: str,
    *,
    scope: GameConstraintScope,
    participants: tuple[str, ...],
    owner: str | None,
) -> GameConstraintBlock:
    return GameConstraintBlock(
        BoundedTrajectoryConstraint(
            lambda trajectory, args: trajectory.final_state[..., 0],
            lower=-jnp.inf,
            upper=0.0,
            constraint_id=constraint_id,
        ),
        scope=scope,
        participants=participants,
        owner=owner,
        site=GameConstraintSite.TERMINAL,
        equality=False,
        residual_shape=(),
        time_dependent=False,
        state_dependent=True,
        control_dependencies=(),
    )


def _individual_evidence(
    response,
    constraints: OpenLoopGameConstraints,
    *,
    residual: float = 0.0,
    stationarity: float = 0.0,
    evidence_id: str = "sampled-individual-kkt",
):
    trajectory = TrajectoryOptimizationView(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [residual]]),
        jnp.zeros((1, constraints.partition.joint_control_size)),
        case_shape=(),
        state_shape=(1,),
        control_shape=(constraints.partition.joint_control_size,),
        approximation_id="one-period-best-response-trajectory",
    )
    feasibility = evaluate_game_feasibility(constraints, trajectory)
    return MeanFieldIndividualConstraintEvidence(
        feasibility,
        stationarity,
        best_response_flow_id=response.flow_id,
        best_response_path_id=response.paths.path_id,
        evidence_id=evidence_id,
    )


def _problem(
    initial,
    blocks,
    concept,
    *,
    induced_value=0.0,
    aggregate_residual=0.0,
    multipliers=(),
    multiplier_ids=(),
    individual_residual=0.0,
    stationarity=0.0,
    induced_weights=None,
    multiplier_layout=None,
    aggregate_jacobian=None,
    derivative_multipliers=None,
    derivative_induced_flow_id=None,
    law_mixture=None,
):
    partition = PlayerControlPartition(("alpha", "beta"), (1, 1))
    constraints = OpenLoopGameConstraints(partition, blocks)
    fixed_point = _fixed_point_problem(
        initial,
        induced_value,
        induced_weights=induced_weights,
        law_mixture=law_mixture,
    )
    individual = OpenLoopGameConstraints(
        partition,
        tuple(block for block in blocks if block.scope is not GameConstraintScope.SHARED),
    )
    if multiplier_layout is None:
        layout = constraints.layout(num_path_sites=1)
        multiplier_layout = layout.multiplier_layout(
            variational=concept is MeanFieldConstraintConcept.AGGREGATE_VARIATIONAL
        )
    has_constraints = bool(blocks)
    has_aggregate = any(block.scope is GameConstraintScope.SHARED for block in blocks)
    return ConstrainedMeanFieldGameProblem(
        fixed_point,
        constraints,
        concept=concept,
        multiplier_layout=multiplier_layout,
        multiplier_ids=multiplier_ids,
        individual_evidence=(
            (
                lambda response, args: _individual_evidence(
                    response,
                    individual,
                    residual=individual_residual,
                    stationarity=stationarity,
                )
            )
            if has_constraints
            else None
        ),
        aggregate_law_residuals=(
            (lambda induced, args: (jnp.asarray(aggregate_residual),))
            if has_aggregate
            else None
        ),
        aggregate_derivative_evidence=(
            (
                lambda response, induced, prices, args: (
                    MeanFieldAggregateConstraintDerivativeEvidence(
                        (
                            jnp.zeros((prices.size, 1))
                            if aggregate_jacobian is None
                            else jnp.asarray(aggregate_jacobian)
                        ),
                        (
                            prices
                            if derivative_multipliers is None
                            else jnp.asarray(derivative_multipliers)
                        ),
                        best_response_flow_id=response.flow_id,
                        best_response_path_id=response.paths.path_id,
                        induced_flow_id=(
                            induced.mean_field_id
                            if derivative_induced_flow_id is None
                            else derivative_induced_flow_id
                        ),
                        aggregate_law_constraints_id="induced-law-capacity",
                        multiplier_ids=multiplier_ids,
                        evidence_id="aggregate-law-jacobian",
                    )
                )
            )
            if has_aggregate
            else None
        ),
        multipliers=(
            (lambda response, induced, args: jnp.asarray(multipliers))
            if has_constraints
            else None
        ),
        individual_evidence_id=("sampled-individual-kkt" if has_constraints else None),
        aggregate_law_constraints_id=("induced-law-capacity" if has_aggregate else None),
        aggregate_derivative_evidence_id=(
            "aggregate-law-jacobian" if has_aggregate else None
        ),
        multiplier_callback_id=("declared-kkt-multipliers" if has_constraints else None),
        problem_id="constrained-one-period-mean-field-game",
    )


def _plan(maximum_iterations=2, *, damping=1.0):
    return ConstrainedMeanFieldGamePlan(
        maximum_iterations=maximum_iterations,
        consistency_tolerance=1.0e-8,
        feasibility_tolerance=1.0e-8,
        kkt_tolerance=1.0e-8,
        damping=damping,
        minimum_effective_sample_size=2.0,
        problem_id="constrained-one-period-mean-field-game",
    )


def test_constrained_outer_damping_uses_exact_law_mixture_callback():
    initial = _law(-1.0, "damped-initial", "damped-input-paths")
    problem = _problem(
        initial,
        (),
        MeanFieldConstraintConcept.INDIVIDUAL,
        induced_value=1.0,
        law_mixture=_exact_law_mixture,
    )

    result = solve_constrained_mean_field_game(
        problem,
        _plan(2, damping=0.5),
    )

    assert result.status == ConstrainedMeanFieldGameStatus.MAX_ITERATIONS
    assert (
        result.fixed_point_results[0].law_mixture_id
        == "exact-constraint-union-law-mixture"
    )
    assert result.flow.num_particles == 4
    np.testing.assert_allclose(_law_means(result.flow), [0.0, 0.0])
    np.testing.assert_allclose(
        jnp.sort(result.flow.particles.reshape((-1,))),
        [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )


def test_unconstrained_reduction_preserves_current_fixed_point_evidence():
    initial = _law(0.0, "unconstrained-initial", "unconstrained-input-paths")
    problem = _problem(
        initial,
        (),
        MeanFieldConstraintConcept.INDIVIDUAL,
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.status == ConstrainedMeanFieldGameStatus.SUCCESS
    assert result.successful
    assert result.fixed_point_result is not None
    assert result.fixed_point_result.status == MeanFieldGameFixedPointStatus.SUCCESS
    assert result.best_response_validity_history[0]
    assert result.induced_law_validity_history[0]
    assert result.law_consistency_history[0]
    assert result.physical_constraint_residual_history.shape == (2, 0)
    assert result.multiplier_history.shape == (2, 0)
    np.testing.assert_allclose(result.final_original_kkt_residual, 0.0)
    np.testing.assert_allclose(result.current_effective_sample_size_history[0], 2.0)
    np.testing.assert_allclose(result.induced_effective_sample_size_history[0], 2.0)
    assert result.current_flow_ids[0] == "unconstrained-initial"
    assert result.induced_source_path_ids[0] == (
        "new-forward-paths:unconstrained-initial"
    )


def test_individual_constraint_requires_feasibility_and_original_kkt_evidence():
    initial = _law(0.0, "individual-initial", "individual-input-paths")
    local = _block(
        "alpha-action-limit",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("alpha",),
        owner="alpha",
    )
    feasible = _problem(
        initial,
        (local,),
        MeanFieldConstraintConcept.INDIVIDUAL,
        multipliers=(0.0,),
        multiplier_ids=("lambda-alpha-action-limit",),
    )
    nonstationary = _problem(
        initial,
        (local,),
        MeanFieldConstraintConcept.INDIVIDUAL,
        multipliers=(0.0,),
        multiplier_ids=("lambda-alpha-action-limit",),
        stationarity=0.25,
    )

    feasible_result = solve_constrained_mean_field_game(feasible, _plan())
    nonstationary_result = solve_constrained_mean_field_game(nonstationary, _plan())

    assert feasible_result.status == ConstrainedMeanFieldGameStatus.SUCCESS
    assert feasible_result.individual_evidence is not None
    assert feasible_result.individual_evidence.feasibility.sampled_only
    assert feasible_result.individual_constraint_ids == ("alpha-action-limit",)
    np.testing.assert_allclose(feasible_result.final_individual_primal_violation, 0.0)
    np.testing.assert_allclose(feasible_result.final_stationarity_residual, 0.0)
    assert (
        nonstationary_result.status
        == ConstrainedMeanFieldGameStatus.INDIVIDUAL_KKT_FAILURE
    )
    np.testing.assert_allclose(nonstationary_result.final_stationarity_residual, 0.25)


def test_aggregate_capacity_distinguishes_generic_population_multipliers():
    initial = _law(0.0, "generic-initial", "generic-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        multipliers=(0.25, 0.75),
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.status == ConstrainedMeanFieldGameStatus.SUCCESS
    assert not result.problem.multiplier_layout.variational
    assert result.aggregate_constraint_ids == ("population-capacity",)
    assert result.multiplier_ids == (
        "lambda-alpha-capacity",
        "lambda-beta-capacity",
    )
    np.testing.assert_allclose(result.population_multipliers[0], [0.25])
    np.testing.assert_allclose(result.population_multipliers[1], [0.75])
    assert result.common_multipliers.shape == (0,)


def test_aggregate_capacity_variational_mode_has_one_declared_common_multiplier():
    initial = _law(0.0, "variational-initial", "variational-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_VARIATIONAL,
        multipliers=(0.5,),
        multiplier_ids=("lambda-common-capacity",),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.status == ConstrainedMeanFieldGameStatus.SUCCESS
    assert result.problem.multiplier_layout.variational
    assert result.multiplier_ids == ("lambda-common-capacity",)
    np.testing.assert_allclose(result.common_multipliers, [0.5])
    assert result.population_multipliers[0].shape == (0,)
    assert result.population_multipliers[1].shape == (0,)


def test_positive_aggregate_prices_require_complete_stationarity():
    initial = _law(0.0, "nonstationary-price-initial", "nonstationary-price-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        multipliers=(0.25, 0.75),
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
        aggregate_jacobian=((1.0,), (1.0,)),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.status == ConstrainedMeanFieldGameStatus.INDIVIDUAL_KKT_FAILURE
    np.testing.assert_allclose(result.final_stationarity_residual, 1.0)
    assert not result.kkt_validity_history[0]


def test_binding_aggregate_price_is_added_to_original_stationarity():
    initial = _law(0.0, "binding-price-initial", "binding-price-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        multipliers=(1.0, 0.0),
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
        stationarity=-1.0,
        aggregate_jacobian=((1.0,), (0.0,)),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.status == ConstrainedMeanFieldGameStatus.SUCCESS
    np.testing.assert_allclose(
        result.individual_evidence.original_stationarity_residual,
        1.0,
    )
    np.testing.assert_allclose(result.final_stationarity_residual, 0.0)
    assert result.aggregate_derivative_evidence is not None
    assert (
        result.aggregate_derivative_evidence.induced_flow_id
        == result.induced_flow.mean_field_id
    )


def test_aggregate_derivative_evidence_rejects_wrong_price_vector():
    initial = _law(0.0, "wrong-price-initial", "wrong-price-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        multipliers=(1.0, 0.0),
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
        stationarity=-1.0,
        aggregate_jacobian=((1.0,), (0.0,)),
        derivative_multipliers=(0.5, 0.0),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert (
        result.status
        == ConstrainedMeanFieldGameStatus.INVALID_AGGREGATE_DERIVATIVE_EVIDENCE
    )
    assert not result.valid


def test_generic_aggregate_problem_rejects_a_common_multiplier_claim():
    initial = _law(0.0, "wrong-common-initial", "wrong-common-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    partition = PlayerControlPartition(("alpha", "beta"), (1, 1))
    constraints = OpenLoopGameConstraints(partition, (capacity,))
    wrong_layout = constraints.layout(num_path_sites=1).multiplier_layout(
        variational=True
    )

    with pytest.raises(ValueError, match="population-specific"):
        _problem(
            initial,
            (capacity,),
            MeanFieldConstraintConcept.AGGREGATE_GENERIC,
            multipliers=(0.5,),
            multiplier_ids=("lambda-common-capacity",),
            multiplier_layout=wrong_layout,
        )


def test_law_consistent_but_population_infeasible_candidate_is_rejected():
    initial = _law(0.0, "infeasible-initial", "infeasible-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        aggregate_residual=0.2,
        multipliers=(0.0, 0.0),
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.law_consistency_history[0]
    assert not result.population_feasibility_history[0]
    assert result.status == ConstrainedMeanFieldGameStatus.POPULATION_INFEASIBLE
    np.testing.assert_allclose(result.final_population_primal_violation, 0.2)
    assert not result.valid


def test_population_feasible_but_law_inconsistent_candidate_is_rejected():
    initial = _law(0.0, "law-mismatch-initial", "law-mismatch-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        induced_value=1.0,
        aggregate_residual=0.0,
        multipliers=(0.0, 0.0),
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
    )

    result = solve_constrained_mean_field_game(problem, _plan(1))

    assert result.population_feasibility_history[0]
    assert not result.law_consistency_history[0]
    assert result.status == ConstrainedMeanFieldGameStatus.MAX_ITERATIONS
    np.testing.assert_allclose(result.final_law_distance, 1.0)
    assert not result.valid


@pytest.mark.parametrize(
    ("aggregate_residual", "multipliers", "expected_status"),
    [
        (
            -1.0,
            (1.0, 0.0),
            ConstrainedMeanFieldGameStatus.COMPLEMENTARITY_FAILURE,
        ),
        (
            0.0,
            (-1.0, 0.0),
            ConstrainedMeanFieldGameStatus.DUAL_INFEASIBLE,
        ),
    ],
)
def test_aggregate_dual_and_complementarity_failures_are_separate(
    aggregate_residual,
    multipliers,
    expected_status,
):
    initial = _law(0.0, "kkt-failure-initial", "kkt-failure-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        aggregate_residual=aggregate_residual,
        multipliers=multipliers,
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.population_feasibility_history[0]
    assert result.status == expected_status
    assert not result.kkt_validity_history[0]


def test_low_effective_sample_size_fails_before_constraint_acceptance():
    initial = _law(0.0, "low-ess-initial", "low-ess-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_GENERIC,
        induced_weights=jnp.asarray([[1.0, 1.0], [0.0, 0.0]]),
        multipliers=(0.0, 0.0),
        multiplier_ids=("lambda-alpha-capacity", "lambda-beta-capacity"),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.status == ConstrainedMeanFieldGameStatus.LOW_EFFECTIVE_SAMPLE_SIZE
    np.testing.assert_allclose(result.induced_effective_sample_size_history[0], 1.0)
    assert not result.individual_evidence_validity_history[0]
    assert not result.valid


def test_candidate_label_retains_sampling_scope_and_separates_stronger_claims():
    initial = _law(0.0, "claim-initial", "claim-input-paths")
    capacity = _block(
        "population-capacity",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
    )
    problem = _problem(
        initial,
        (capacity,),
        MeanFieldConstraintConcept.AGGREGATE_VARIATIONAL,
        multipliers=(0.0,),
        multiplier_ids=("lambda-common-capacity",),
    )

    result = solve_constrained_mean_field_game(problem, _plan())

    assert result.status == ConstrainedMeanFieldGameStatus.SUCCESS
    assert result.certificate_label == CONSTRAINED_MEAN_FIELD_GAME_KKT_CANDIDATE
    assert result.candidate_evaluation_only
    assert result.sampled_only
    assert result.frozen_law_best_response_evaluated
    assert result.law_consistency_evaluated
    assert result.best_response_kkt_evaluated
    assert result.aggregate_feasibility_evaluated
    assert "sampled" in result.sampling_scope
    assert not result.best_response_optimality_evaluated
    assert not result.continuous_safety_claimed
    assert not result.mean_field_game_equilibrium_claimed
    assert not result.generalized_mean_field_equilibrium_claimed
    assert not result.mean_field_control_optimum_claimed
    assert not result.master_equation_claimed
    assert not result.finite_population_game_claimed
