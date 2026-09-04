#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.control._trajectory_optimization import BoundedPathConstraint
from phydrax.control.games import _generalized_nash as _gne
from phydrax.control.games._constraints import (
    GameConstraintBlock,
    GameConstraintScope,
    GameConstraintSite,
    OpenLoopGameConstraints,
)
from phydrax.control.games._generalized_nash import (
    FiniteHorizonLQOpenLoopGNEProblem,
    GLOBAL_CONVEX_GNE_GAP_EVIDENCE,
    OPEN_LOOP_GENERALIZED_NASH_KKT,
    OpenLoopGNEStatus,
    plan_open_loop_gne,
    prepare_open_loop_gne,
    refresh_open_loop_gne,
    solve_open_loop_gne,
    solve_prepared_open_loop_gne,
)
from phydrax.control.games._layout import PlayerControlPartition
from phydrax.optim import ConvexProgramStatus


def _path_inequality(
    function,
    constraint_id: str,
    *,
    scope: GameConstraintScope,
    participants: tuple[str, ...],
    owner: str | None,
    control_dependencies: tuple[str, ...],
) -> GameConstraintBlock:
    return GameConstraintBlock(
        BoundedPathConstraint(
            function,
            lower=-jnp.inf,
            upper=0.0,
            constraint_id=constraint_id,
        ),
        scope=scope,
        participants=participants,
        owner=owner,
        site=GameConstraintSite.PATH,
        equality=False,
        residual_shape=(),
        time_dependent=False,
        state_dependent=False,
        control_dependencies=control_dependencies,
    )


def _separable_problem(
    targets,
    *,
    constraints: OpenLoopGameConstraints | None = None,
    problem_id: str = "test:open-loop-gne",
) -> FiniteHorizonLQOpenLoopGNEProblem:
    targets = jnp.asarray(targets)
    players = int(targets.shape[-1])
    partition = (
        PlayerControlPartition(
            tuple(f"player-{player}" for player in range(players)),
            (1,) * players,
        )
        if constraints is None
        else constraints.partition
    )
    case_shape = tuple(targets.shape[:-1])
    control_costs = jnp.zeros(
        case_shape + (players, 1, players, players), dtype=targets.dtype
    )
    control_linear = jnp.zeros(case_shape + (players, 1, players), dtype=targets.dtype)
    for player in range(players):
        control_costs = control_costs.at[..., player, 0, player, player].set(1.0)
        control_linear = control_linear.at[..., player, 0, player].set(
            -targets[..., player]
        )
    return FiniteHorizonLQOpenLoopGNEProblem(
        jnp.zeros(case_shape + (1, 1, 1)),
        jnp.zeros(case_shape + (1, 1, players)),
        jnp.zeros(case_shape + (1,)),
        jnp.zeros(case_shape + (players, 1, 1, 1)),
        control_costs,
        jnp.zeros(case_shape + (players, 1, 1)),
        partition,
        constraints=constraints,
        control_linear=control_linear,
        problem_id=problem_id,
    )


def _two_player_shared_resource(
    *, include_private: bool = False
) -> OpenLoopGameConstraints:
    partition = PlayerControlPartition(("player-0", "player-1"), (1, 1))
    blocks = []
    if include_private:
        blocks.extend(
            (
                _path_inequality(
                    lambda time, state, control, args: -control[0] - 2.0,
                    "player-zero-private-lower",
                    scope=GameConstraintScope.PLAYER_LOCAL,
                    participants=("player-0",),
                    owner="player-0",
                    control_dependencies=("player-0",),
                ),
                _path_inequality(
                    lambda time, state, control, args: -control[1] - 2.0,
                    "player-one-private-lower",
                    scope=GameConstraintScope.PLAYER_LOCAL,
                    participants=("player-1",),
                    owner="player-1",
                    control_dependencies=("player-1",),
                ),
            )
        )
    blocks.append(
        _path_inequality(
            lambda time, state, control, args: control[0] + control[1] - 1.0,
            "shared-resource",
            scope=GameConstraintScope.SHARED,
            participants=("player-0", "player-1"),
            owner=None,
            control_dependencies=("player-0", "player-1"),
        )
    )
    return OpenLoopGameConstraints(partition, tuple(blocks))


def _solve_exact_profile(problem, controls, inequality_multipliers):
    plan = plan_open_loop_gne(problem)
    prepared = prepare_open_loop_gne(
        plan,
        problem,
        jnp.asarray((controls,)),
        initial_inequality_multipliers=jnp.asarray(inequality_multipliers),
    )
    return solve_prepared_open_loop_gne(prepared)


def test_shared_resource_continuum_endpoints_keep_unequal_player_multiplier_copies():
    constraints = _two_player_shared_resource()
    problem = _separable_problem(
        (2.0, 2.0), constraints=constraints, problem_id="test:gne-endpoints"
    )

    left = _solve_exact_profile(problem, (0.0, 1.0), (2.0, 1.0))
    right = _solve_exact_profile(problem, (1.0, 0.0), (1.0, 2.0))

    np.testing.assert_allclose(left.controls[0], (0.0, 1.0), atol=2.0e-6)
    np.testing.assert_allclose(right.controls[0], (1.0, 0.0), atol=2.0e-6)
    np.testing.assert_allclose(
        left.player_shared_multiplier_copies[0], (2.0,), atol=2.0e-6
    )
    np.testing.assert_allclose(
        left.player_shared_multiplier_copies[1], (1.0,), atol=2.0e-6
    )
    np.testing.assert_allclose(
        right.player_shared_multiplier_copies[0], (1.0,), atol=2.0e-6
    )
    np.testing.assert_allclose(
        right.player_shared_multiplier_copies[1], (2.0,), atol=2.0e-6
    )
    assert bool(left.valid) and bool(right.valid)
    assert left.physical_shared_residuals.shape == (1,)
    assert left.multiplier_layout.num_multipliers == 2
    assert left.multiplier_layout.shared_slice == (2, 2)


def test_variational_midpoint_is_one_generic_gne_without_common_multiplier_claim():
    constraints = _two_player_shared_resource()
    problem = _separable_problem(
        (2.0, 2.0), constraints=constraints, problem_id="test:ve-is-gne"
    )

    midpoint = _solve_exact_profile(problem, (0.5, 0.5), (1.5, 1.5))

    np.testing.assert_allclose(midpoint.controls[0], (0.5, 0.5), atol=2.0e-6)
    np.testing.assert_allclose(
        midpoint.player_shared_multiplier_copies[0], (1.5,), atol=2.0e-6
    )
    np.testing.assert_allclose(
        midpoint.player_shared_multiplier_copies[1], (1.5,), atol=2.0e-6
    )
    assert midpoint.certificate_label == OPEN_LOOP_GENERALIZED_NASH_KKT
    assert midpoint.global_gap_certificate_label == GLOBAL_CONVEX_GNE_GAP_EVIDENCE
    assert not midpoint.multiplier_layout.variational
    assert not midpoint.common_multiplier_imposed
    assert not midpoint.variational_equilibrium_claimed
    assert "variational" not in midpoint.certification_claim.lower()
    assert not bool(midpoint.global_gap_evidence_available)
    assert bool(midpoint.nonuniqueness_evidence)
    assert int(midpoint.branch_dimension) == 1
    assert int(midpoint.status) == int(OpenLoopGNEStatus.RESIDUAL_VALID_NONISOLATED)


def test_shared_participant_subset_allocates_no_multiplier_to_nonparticipant():
    partition = PlayerControlPartition(("player-0", "player-1", "player-2"), (1, 1, 1))
    resource = _path_inequality(
        lambda time, state, control, args: control[0] + control[1] - 1.0,
        "two-of-three-resource",
        scope=GameConstraintScope.SHARED,
        participants=("player-0", "player-1"),
        owner=None,
        control_dependencies=("player-0", "player-1"),
    )
    constraints = OpenLoopGameConstraints(partition, (resource,))
    problem = _separable_problem(
        (2.0, 2.0, 3.0),
        constraints=constraints,
        problem_id="test:participant-subset-gne",
    )
    plan = plan_open_loop_gne(problem)
    prepared = prepare_open_loop_gne(
        plan,
        problem,
        jnp.asarray(((0.5, 0.5, 3.0),)),
        initial_inequality_multipliers=jnp.asarray((1.5, 1.5)),
    )
    result = solve_prepared_open_loop_gne(prepared)

    np.testing.assert_allclose(result.controls[0], (0.5, 0.5, 3.0), atol=3.0e-6)
    assert result.player_multipliers[2].shape == (0,)
    assert result.player_shared_multiplier_copies[2].shape == (0,)
    assert plan.multiplier_layout.player_slices == ((0, 1), (1, 2), (2, 2))
    assert result.physical_constraint_residuals.shape == (1,)
    assert bool(result.original_kkt_valid)


def test_private_and_shared_constraints_retain_one_physical_copy_and_player_blocks():
    constraints = _two_player_shared_resource(include_private=True)
    problem = _separable_problem(
        (2.0, 2.0), constraints=constraints, problem_id="test:private-shared-gne"
    )
    # Multiplier order is player 0 private/shared, then player 1 private/shared.
    result = _solve_exact_profile(problem, (0.5, 0.5), (0.0, 1.5, 0.0, 1.5))

    assert result.physical_constraint_residuals.shape == (3,)
    assert result.physical_shared_residuals.shape == (1,)
    assert result.player_multipliers[0].shape == (2,)
    assert result.player_multipliers[1].shape == (2,)
    np.testing.assert_allclose(result.player_multipliers[0], (0.0, 1.5), atol=3.0e-6)
    np.testing.assert_allclose(result.player_multipliers[1], (0.0, 1.5), atol=3.0e-6)
    np.testing.assert_array_equal(result.player_constraint_qualification, (True, True))
    assert bool(result.original_kkt_valid)


def test_best_response_gap_uses_minimizer_sign_and_complete_audit_enables_global_bound(
    monkeypatch,
):
    problem = _separable_problem((1.0, 2.0), problem_id="test:best-response-sign")
    prepared = prepare_open_loop_gne(
        plan_open_loop_gne(problem, audit_best_responses=True),
        problem,
        jnp.zeros((1, 2)),
    )
    original = _gne.solve_prepared_variational_inequality

    def leave_non_equilibrium_profile(candidate, *, termination=None):
        result = original(candidate, termination=termination)
        controls, equality, inequality = result.state
        return eqx.tree_at(
            lambda item: item.state,
            result,
            (jnp.zeros_like(controls), equality, inequality),
        )

    monkeypatch.setattr(
        _gne,
        "solve_prepared_variational_inequality",
        leave_non_equilibrium_profile,
    )
    result = solve_prepared_open_loop_gne(prepared)

    np.testing.assert_allclose(result.player_best_response_gaps, (0.5, 2.0), atol=2.0e-5)
    assert bool(jnp.all(result.player_best_response_gaps > 0.0))
    assert bool(jnp.all(result.best_response_numerical_errors >= 0.0))
    assert bool(result.best_response_audit_complete)
    # The KKT candidate was deliberately corrupted, so complete BR solves alone
    # are insufficient to publish global equilibrium-gap evidence.
    assert not bool(result.global_gap_evidence_available)
    assert int(result.status) == int(OpenLoopGNEStatus.ORIGINAL_KKT_FAILURE)


def test_complete_best_response_audits_publish_separate_global_convex_gap_evidence():
    problem = _separable_problem((1.0, 2.0), problem_id="test:global-gap-gne")

    result = solve_open_loop_gne(problem, audit_best_responses=True)

    assert bool(result.best_response_audit_complete)
    assert bool(result.global_gap_evidence_available)
    assert float(result.global_gne_gap_bound) >= 0.0
    assert float(result.global_gne_gap_bound) <= 2.0e-5
    assert result.certificate_label == OPEN_LOOP_GENERALIZED_NASH_KKT
    assert result.global_gap_certificate_label == GLOBAL_CONVEX_GNE_GAP_EVIDENCE


def test_failed_inner_best_response_solve_has_stable_status_and_no_global_bound(
    monkeypatch,
):
    problem = _separable_problem((1.0, 2.0), problem_id="test:failed-br-gne")
    prepared = prepare_open_loop_gne(
        plan_open_loop_gne(problem, audit_best_responses=True),
        problem,
        jnp.zeros((1, 2)),
    )
    original = _gne.solve_quadratic_program

    def fail_second_player(program, *, policy=None, warm_start=None):
        result = original(program, policy=policy, warm_start=warm_start)
        if program.problem_id.endswith("player-1:best-response"):
            return eqx.tree_at(
                lambda item: (item.status, item.valid),
                result,
                (
                    jnp.full_like(
                        result.status, int(ConvexProgramStatus.ITERATION_LIMIT)
                    ),
                    jnp.zeros_like(result.valid),
                ),
            )
        return result

    monkeypatch.setattr(_gne, "solve_quadratic_program", fail_second_player)
    result = solve_prepared_open_loop_gne(prepared)

    np.testing.assert_array_equal(result.best_response_successful, (True, False))
    assert not bool(result.best_response_audit_complete)
    assert not bool(result.global_gap_evidence_available)
    assert int(result.status) == int(OpenLoopGNEStatus.BEST_RESPONSE_FAILURE)
    assert not bool(result.valid)


def test_zero_game_reports_nonisolated_branch_without_fabricating_uniqueness():
    partition = PlayerControlPartition(("player-0", "player-1"), (1, 1))
    problem = FiniteHorizonLQOpenLoopGNEProblem(
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 2)),
        jnp.zeros((1,)),
        jnp.zeros((2, 1, 1, 1)),
        jnp.zeros((2, 1, 2, 2)),
        jnp.zeros((2, 1, 1)),
        partition,
        problem_id="test:nonunique-gne",
    )

    result = solve_open_loop_gne(problem, jnp.asarray(((0.25, -0.75),)))

    np.testing.assert_allclose(result.controls[0], (0.25, -0.75), atol=1.0e-7)
    assert bool(result.nonuniqueness_evidence)
    assert int(result.branch_dimension) == 2
    assert not bool(result.branch_isolated)
    assert bool(result.valid)
    assert int(result.status) == int(OpenLoopGNEStatus.RESIDUAL_VALID_NONISOLATED)


def test_player_owned_coupled_constraint_has_only_the_owner_multiplier():
    partition = PlayerControlPartition(("player-0", "player-1"), (1, 1))
    unilateral = _path_inequality(
        lambda time, state, control, args: control[0] + control[1] - 1.0,
        "player-zero-coupled-feasible-set",
        scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
        participants=("player-0", "player-1"),
        owner="player-0",
        control_dependencies=("player-0", "player-1"),
    )
    constraints = OpenLoopGameConstraints(partition, (unilateral,))
    problem = _separable_problem(
        (2.0, 0.0),
        constraints=constraints,
        problem_id="test:player-owned-coupled-gne",
    )
    result = _solve_exact_profile(problem, (1.0, 0.0), (1.0,))

    np.testing.assert_allclose(result.controls[0], (1.0, 0.0), atol=2.0e-6)
    np.testing.assert_allclose(result.player_multipliers[0], (1.0,), atol=2.0e-6)
    assert result.player_multipliers[1].shape == (0,)
    assert result.multiplier_layout.num_multipliers == 1
    assert result.physical_constraint_residuals.shape == (1,)
    assert bool(result.original_kkt_valid)


def test_dependent_active_rows_report_failed_player_cq_and_singular_branch():
    partition = PlayerControlPartition(("player-0",), (1,))
    first = _path_inequality(
        lambda time, state, control, args: control[0],
        "duplicate-upper-one",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("player-0",),
        owner="player-0",
        control_dependencies=("player-0",),
    )
    second = _path_inequality(
        lambda time, state, control, args: control[0],
        "duplicate-upper-two",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("player-0",),
        owner="player-0",
        control_dependencies=("player-0",),
    )
    problem = _separable_problem(
        (1.0,),
        constraints=OpenLoopGameConstraints(partition, (first, second)),
        problem_id="test:dependent-cq-gne",
    )
    result = _solve_exact_profile(problem, (0.0,), (0.5, 0.5))

    np.testing.assert_array_equal(result.player_active_constraint_rank, (1,))
    np.testing.assert_array_equal(result.player_active_constraint_count, (2,))
    np.testing.assert_array_equal(result.player_constraint_qualification, (False,))
    assert int(result.branch_dimension) == 1
    assert not bool(result.branch_regular)
    assert not bool(result.regularity_certified)


def test_case_axes_jit_and_refresh_preserve_topology_and_change_numeric_solution():
    first = _separable_problem(
        jnp.asarray(((1.0, 2.0), (3.0, -1.0))),
        problem_id="test:case-refresh-gne",
    )
    plan = plan_open_loop_gne(first)
    prepared = prepare_open_loop_gne(plan, first, jnp.zeros((2, 1, 2)))
    first_result = eqx.filter_jit(solve_prepared_open_loop_gne)(prepared)

    second = _separable_problem(
        jnp.asarray(((2.0, -2.0), (0.5, 4.0))),
        problem_id="test:case-refresh-gne",
    )
    refreshed = refresh_open_loop_gne(prepared, second)
    second_result = eqx.filter_jit(solve_prepared_open_loop_gne)(refreshed)

    np.testing.assert_allclose(
        first_result.controls[:, 0, :], ((1.0, 2.0), (3.0, -1.0)), atol=4.0e-6
    )
    np.testing.assert_allclose(
        second_result.controls[:, 0, :], ((2.0, -2.0), (0.5, 4.0)), atol=4.0e-6
    )
    assert first_result.status.shape == (2,)
    np.testing.assert_array_equal(first_result.valid, (True, True))
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.vi_prepared.topology_id == prepared.vi_prepared.topology_id
    assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1
