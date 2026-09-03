#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.control._trajectory_optimization import (
    BoundedPathConstraint,
    BoundedTrajectoryConstraint,
)
from phydrax.control.games._constraints import (
    GameConstraintBlock,
    GameConstraintScope,
    GameConstraintSite,
    OpenLoopGameConstraints,
)
from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._variational import (
    FiniteHorizonLQOpenLoopVEProblem,
    OpenLoopVEStatus,
    plan_open_loop_ve,
    prepare_open_loop_ve,
    refresh_open_loop_ve,
    solve_open_loop_ve,
    solve_prepared_open_loop_ve,
)


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


def _trajectory_constraint(
    function,
    constraint_id: str,
    *,
    scope: GameConstraintScope,
    participants: tuple[str, ...],
    owner: str | None,
    equality: bool,
    state_dependent: bool,
    control_dependencies: tuple[str, ...] = (),
) -> GameConstraintBlock:
    return GameConstraintBlock(
        BoundedTrajectoryConstraint(
            function,
            lower=0.0 if equality else -jnp.inf,
            upper=0.0,
            constraint_id=constraint_id,
        ),
        scope=scope,
        participants=participants,
        owner=owner,
        site=GameConstraintSite.TRAJECTORY,
        equality=equality,
        residual_shape=(),
        time_dependent=False,
        state_dependent=state_dependent,
        control_dependencies=control_dependencies,
    )


def _two_player_problem(
    linear,
    *,
    constraints: OpenLoopGameConstraints | None = None,
    dynamics_matrices=None,
    control_matrices=None,
    initial_state=None,
    problem_id: str = "test:open-loop-ve",
) -> FiniteHorizonLQOpenLoopVEProblem:
    partition = (
        PlayerControlPartition(("one", "two"), (1, 1))
        if constraints is None
        else constraints.partition
    )
    a = jnp.zeros((1, 1, 1)) if dynamics_matrices is None else dynamics_matrices
    b = jnp.zeros((1, 1, 2)) if control_matrices is None else control_matrices
    x0 = jnp.zeros((1,)) if initial_state is None else initial_state
    q = jnp.zeros((2, 1, 1, 1))
    r = jnp.asarray(
        (
            (((1.0, 0.0), (0.0, 0.0)),),
            (((0.0, 0.0), (0.0, 1.0)),),
        )
    )
    q_terminal = jnp.zeros((2, 1, 1))
    return FiniteHorizonLQOpenLoopVEProblem(
        a,
        b,
        x0,
        q,
        r,
        q_terminal,
        partition,
        constraints=constraints,
        control_linear=jnp.asarray(linear),
        problem_id=problem_id,
    )


def _shared_resource_constraints() -> OpenLoopGameConstraints:
    partition = PlayerControlPartition(("one", "two"), (1, 1))
    lower_one = _path_inequality(
        lambda time, state, control, args: -control[0],
        "one-nonnegative",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("one",),
        owner="one",
        control_dependencies=("one",),
    )
    lower_two = _path_inequality(
        lambda time, state, control, args: -control[1],
        "two-nonnegative",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("two",),
        owner="two",
        control_dependencies=("two",),
    )
    resource = _path_inequality(
        lambda time, state, control, args: control[0] + control[1] - 1.0,
        "shared-resource",
        scope=GameConstraintScope.SHARED,
        participants=("one", "two"),
        owner=None,
        control_dependencies=("one", "two"),
    )
    return OpenLoopGameConstraints(partition, (lower_one, lower_two, resource))


def test_shared_resource_selects_unique_ve_from_continuum_gne():
    constraints = _shared_resource_constraints()
    problem = _two_player_problem(
        (((-2.0, 0.0),), ((0.0, -2.0),)),
        constraints=constraints,
    )

    result = solve_open_loop_ve(problem, jnp.asarray(((0.2, 0.2),)))

    np.testing.assert_allclose(result.controls[0], (0.5, 0.5), atol=3.0e-5)
    np.testing.assert_allclose(result.shared_multipliers, (1.5,), atol=3.0e-5)
    np.testing.assert_allclose(result.private_multipliers[0], (0.0,), atol=3.0e-5)
    np.testing.assert_allclose(result.private_multipliers[1], (0.0,), atol=3.0e-5)
    assert bool(result.valid)
    assert int(result.status) == int(OpenLoopVEStatus.SUCCESS)
    assert result.certificate_label == "OPEN_LOOP_VARIATIONAL_GNE"
    assert result.certification_claim == "numerically certified convex open-loop VE"
    assert result.prepared_id
    assert bool(result.convexity_certified)
    assert bool(result.strongly_monotone)
    assert bool(result.vi_result.certificate.certified)


def test_endpoint_gne_does_not_satisfy_one_common_shared_multiplier_kkt():
    constraints = _shared_resource_constraints()
    problem = _two_player_problem(
        (((-2.0, 0.0),), ((0.0, -2.0),)),
        constraints=constraints,
    )
    prepared = prepare_open_loop_ve(
        plan_open_loop_ve(problem), problem, jnp.asarray(((1.0, 0.0),))
    )
    endpoint = jnp.asarray((1.0, 0.0))
    # Player one forces the common multiplier to one. Player two would then
    # require a negative multiplier on its private nonnegativity constraint.
    equality = jnp.zeros((1,))  # fixed internal dummy: there are no equalities
    inequality = jnp.asarray((0.0, 0.0, 1.0))
    stationarity = prepared.vi_problem.evaluate(
        (endpoint, equality, inequality), prepared.vi_prepared.args
    )[0]

    np.testing.assert_allclose(stationarity, (0.0, -1.0), atol=1.0e-7)
    assert prepared.plan.multiplier_layout.shared_slice == (2, 3)
    assert prepared.plan.multiplier_layout.num_multipliers == 3


def test_private_only_game_reduces_to_open_loop_nash_and_retains_cross_costs():
    partition = PlayerControlPartition(("one", "two"), (1, 1))
    problem = _two_player_problem(
        (((-1.0, 4.0),), ((-3.0, -2.0),)),
        constraints=OpenLoopGameConstraints(partition),
    )
    # The non-owned linear entries remain in player costs but not in owned rows
    # of the pseudogradient: the Nash controls are therefore (1, 2).
    result = solve_open_loop_ve(problem)

    np.testing.assert_allclose(result.controls[0], (1.0, 2.0), atol=2.0e-6)
    np.testing.assert_allclose(result.player_costs, (7.5, -5.0), atol=2.0e-6)
    assert result.shared_multipliers.shape == (0,)
    assert bool(result.valid)


def test_affine_dynamics_condensation_enforces_terminal_budget_and_reconstructs_states():
    partition = PlayerControlPartition(("one", "two"), (1, 1))
    terminal_budget = _trajectory_constraint(
        lambda trajectory, args: trajectory.final_state[..., 0] - 2.0,
        "terminal-budget",
        scope=GameConstraintScope.SHARED,
        participants=("one", "two"),
        owner=None,
        equality=True,
        state_dependent=True,
    )
    constraints = OpenLoopGameConstraints(partition, (terminal_budget,))
    problem = _two_player_problem(
        (((-3.0, 0.0),), ((0.0, -3.0),)),
        constraints=constraints,
        dynamics_matrices=jnp.ones((1, 1, 1)),
        control_matrices=jnp.asarray((((1.0, 1.0),),)),
        initial_state=jnp.asarray((0.25,)),
    )

    result = solve_open_loop_ve(problem)

    np.testing.assert_allclose(result.controls[0], (0.875, 0.875), atol=4.0e-5)
    np.testing.assert_allclose(result.states, ((0.25,), (2.0,)), atol=4.0e-5)
    np.testing.assert_allclose(result.shared_multipliers, (2.125,), atol=4.0e-5)
    assert bool(result.valid)


def test_inactive_active_degenerate_and_nonisolated_evidence_are_distinct():
    partition = PlayerControlPartition(("one", "two"), (1, 1))
    inactive_constraint = _path_inequality(
        lambda time, state, control, args: control[0] + control[1] - 1.0,
        "inactive-resource",
        scope=GameConstraintScope.SHARED,
        participants=("one", "two"),
        owner=None,
        control_dependencies=("one", "two"),
    )
    inactive_problem = _two_player_problem(
        (((-0.2, 0.0),), ((0.0, -0.2),)),
        constraints=OpenLoopGameConstraints(partition, (inactive_constraint,)),
        problem_id="test:inactive-ve",
    )
    inactive = solve_open_loop_ve(inactive_problem)
    np.testing.assert_allclose(inactive.controls[0], (0.2, 0.2), atol=2.0e-5)
    np.testing.assert_allclose(inactive.shared_multipliers, (0.0,), atol=2.0e-5)

    duplicate_one = _path_inequality(
        lambda time, state, control, args: control[0] + control[1] - 1.0,
        "duplicate-resource-one",
        scope=GameConstraintScope.SHARED,
        participants=("one", "two"),
        owner=None,
        control_dependencies=("one", "two"),
    )
    duplicate_two = _path_inequality(
        lambda time, state, control, args: control[0] + control[1] - 1.0,
        "duplicate-resource-two",
        scope=GameConstraintScope.SHARED,
        participants=("one", "two"),
        owner=None,
        control_dependencies=("one", "two"),
    )
    degenerate_problem = _two_player_problem(
        (((-2.0, 0.0),), ((0.0, -2.0),)),
        constraints=OpenLoopGameConstraints(partition, (duplicate_one, duplicate_two)),
        problem_id="test:degenerate-ve",
    )
    degenerate = solve_open_loop_ve(degenerate_problem)
    assert not bool(degenerate.regularity_certified)
    assert int(degenerate.active_constraint_count) == 2
    assert int(degenerate.active_constraint_rank) == 1

    zero_costs = FiniteHorizonLQOpenLoopVEProblem(
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 2)),
        jnp.zeros((1,)),
        jnp.zeros((2, 1, 1, 1)),
        jnp.zeros((2, 1, 2, 2)),
        jnp.zeros((2, 1, 1)),
        partition,
        problem_id="test:nonisolated-ve",
    )
    nonisolated = solve_open_loop_ve(zero_costs, jnp.asarray(((0.25, -0.75),)))
    np.testing.assert_allclose(nonisolated.controls[0], (0.25, -0.75), atol=1.0e-7)
    assert bool(nonisolated.valid)
    assert int(nonisolated.status) == int(OpenLoopVEStatus.RESIDUAL_VALID_NONISOLATED)
    assert bool(nonisolated.nonuniqueness_evidence)
    assert int(nonisolated.nonisolation_dimension) == 2


def test_audited_phase_i_certifies_infeasible_original_polyhedron():
    partition = PlayerControlPartition(("one",), (1,))
    upper = _path_inequality(
        lambda time, state, control, args: control[0],
        "upper-zero",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("one",),
        owner="one",
        control_dependencies=("one",),
    )
    lower = _path_inequality(
        lambda time, state, control, args: 1.0 - control[0],
        "lower-one",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("one",),
        owner="one",
        control_dependencies=("one",),
    )
    constraints = OpenLoopGameConstraints(partition, (upper, lower))
    problem = FiniteHorizonLQOpenLoopVEProblem(
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1, 1, 1)),
        jnp.ones((1, 1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        partition,
        constraints=constraints,
        problem_id="test:infeasible-ve",
    )

    result = solve_open_loop_ve(problem)

    assert int(result.status) == int(OpenLoopVEStatus.CERTIFIED_INFEASIBILITY)
    assert bool(result.phase_one_result.certificate.dual_ray_valid)
    assert not bool(result.valid)


def test_case_axes_are_preserved_in_condensation_solve_and_status():
    partition = PlayerControlPartition(("one", "two"), (1, 1))
    a = jnp.zeros((2, 1, 1, 1))
    b = jnp.zeros((2, 1, 1, 2))
    x0 = jnp.zeros((2, 1))
    q = jnp.zeros((2, 2, 1, 1, 1))
    r_single = jnp.asarray(
        (
            (((1.0, 0.0), (0.0, 0.0)),),
            (((0.0, 0.0), (0.0, 1.0)),),
        )
    )
    r = jnp.broadcast_to(r_single, (2,) + r_single.shape)
    q_terminal = jnp.zeros((2, 2, 1, 1))
    linear = jnp.asarray(
        (
            (((-1.0, 0.0),), ((0.0, -2.0),)),
            (((-3.0, 0.0),), ((0.0, 1.0),)),
        )
    )
    problem = FiniteHorizonLQOpenLoopVEProblem(
        a,
        b,
        x0,
        q,
        r,
        q_terminal,
        partition,
        control_linear=linear,
        problem_id="test:case-axis-ve",
    )

    result = solve_open_loop_ve(problem)

    np.testing.assert_allclose(
        result.controls[:, 0, :], ((1.0, 2.0), (3.0, -1.0)), atol=3.0e-6
    )
    assert result.status.shape == (2,)
    np.testing.assert_array_equal(result.valid, (True, True))
    assert result.player_costs.shape == (2, 2)


def test_jitted_prepared_solve_and_refresh_preserve_topology_identity():
    partition = PlayerControlPartition(("one", "two"), (1, 1))
    constraints = OpenLoopGameConstraints(partition)
    first = _two_player_problem(
        (((-1.0, 0.0),), ((0.0, -2.0),)),
        constraints=constraints,
        problem_id="test:refresh-ve",
    )
    plan = plan_open_loop_ve(first)
    prepared = prepare_open_loop_ve(plan, first, jnp.zeros((1, 2)))
    first_result = eqx.filter_jit(solve_prepared_open_loop_ve)(prepared)

    second = _two_player_problem(
        (((-2.0, 0.0),), ((0.0, 1.0),)),
        constraints=constraints,
        problem_id="test:refresh-ve",
    )
    refreshed = refresh_open_loop_ve(prepared, second)
    second_result = eqx.filter_jit(solve_prepared_open_loop_ve)(refreshed)

    np.testing.assert_allclose(first_result.controls[0], (1.0, 2.0), atol=3.0e-6)
    np.testing.assert_allclose(second_result.controls[0], (2.0, -1.0), atol=3.0e-6)
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.vi_prepared.topology_id == prepared.vi_prepared.topology_id
    assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1


def test_natural_residual_uses_independent_projection_onto_original_polyhedron():
    constraints = _shared_resource_constraints()
    problem = _two_player_problem(
        (((-2.0, 0.0),), ((0.0, -2.0),)),
        constraints=constraints,
        problem_id="test:projection-certificate-ve",
    )
    plan = plan_open_loop_ve(problem, natural_step=0.37)
    prepared = prepare_open_loop_ve(plan, problem, jnp.zeros((1, 2)))
    result = solve_prepared_open_loop_ve(prepared)
    flat = result.controls.reshape((2,))

    np.testing.assert_allclose(
        result.natural_residual,
        jnp.linalg.norm(flat - result.projection_result.primal),
        atol=1.0e-8,
    )
    assert result.projection_result.provenance.problem_id.endswith(
        ":independent-natural-projection"
    )
    assert bool(result.projection_result.successful)


def test_nonlinear_declared_constraint_is_structurally_invalid_not_certified_ve():
    partition = PlayerControlPartition(("one", "two"), (1, 1))
    nonlinear = _path_inequality(
        lambda time, state, control, args: control[0] ** 2 - 1.0,
        "non-polyhedral",
        scope=GameConstraintScope.SHARED,
        participants=("one", "two"),
        owner=None,
        control_dependencies=("one",),
    )
    problem = _two_player_problem(
        (((-0.2, 0.0),), ((0.0, -0.2),)),
        constraints=OpenLoopGameConstraints(partition, (nonlinear,)),
        problem_id="test:invalid-polyhedral-claim",
    )

    result = solve_open_loop_ve(problem)

    assert int(result.status) == int(OpenLoopVEStatus.STRUCTURAL_INVALIDITY)
    assert not bool(result.valid)


def test_nonfinite_problem_data_has_a_distinct_case_status():
    problem = _two_player_problem(
        (((-0.2, 0.0),), ((0.0, -0.2),)),
        dynamics_matrices=jnp.full((1, 1, 1), jnp.nan),
        problem_id="test:nonfinite-ve",
    )

    result = solve_open_loop_ve(problem)

    assert int(result.status) == int(OpenLoopVEStatus.NONFINITE)
    assert not bool(result.finite)
    assert not bool(result.valid)
