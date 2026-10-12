#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.control._trajectory_optimization import (
    BoundedPathConstraint,
    BoundedTrajectoryConstraint,
)
from phydrax.control.games._constraints import (
    GameConstraintBlock,
    GameConstraintScope,
    GameConstraintSite,
    GameFeasibilityStatus,
    OpenLoopGameConstraints,
)
from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._open_loop_kkt import (
    LOCAL_NOMINAL_GNE_STATIONARY,
    LOCAL_NOMINAL_NASH_STATIONARY,
    NonlinearOpenLoopGameProblem,
    OpenLoopGameKKTStatus,
    plan_open_loop_game_kkt,
    prepare_open_loop_game_kkt,
    refresh_open_loop_game_kkt,
    solve_open_loop_game_kkt,
    solve_prepared_open_loop_game_kkt,
)
from phydrax.nonlinear import NonlinearTermination


def _path_block(
    function: Any,
    constraint_id: Any,
    *,
    owner: Any = "one",
    participants: Any = ("one",),
    scope: Any = GameConstraintScope.PLAYER_LOCAL,
    equality: Any = False,
    state_dependent: Any = False,
    control_dependencies: Any = ("one",),
) -> Any:
    return GameConstraintBlock(
        BoundedPathConstraint(
            function,
            lower=0.0 if equality else -jnp.inf,
            upper=0.0,
            constraint_id=constraint_id,
        ),
        scope=scope,
        participants=participants,
        owner=owner,
        site=GameConstraintSite.PATH,
        equality=equality,
        residual_shape=(),
        time_dependent=False,
        state_dependent=state_dependent,
        control_dependencies=control_dependencies,
    )


def _trajectory_block(
    function: Any,
    constraint_id: Any,
    *,
    site: Any,
    owner: Any = "one",
    participants: Any = ("one",),
    scope: Any = GameConstraintScope.PLAYER_LOCAL,
    equality: Any = False,
    state_dependent: Any = True,
    control_dependencies: Any = (),
) -> Any:
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
        site=site,
        equality=equality,
        residual_shape=(),
        time_dependent=False,
        state_dependent=state_dependent,
        control_dependencies=control_dependencies,
    )


def _problem(
    partition: Any,
    constraints: Any,
    stage_costs: Any,
    terminal_costs: Any,
    *,
    horizon: Any = 1,
    args: Any = None,
    initial_state: Any = 0.0,
    problem_id: Any = "nonlinear-open-loop-test",
    nonlinear_dynamics: Any = False,
) -> Any:
    state_layout = phx.dynamics.StateLayout((1,))
    input_layout = phx.dynamics.InputLayout(
        (partition.joint_control_size,), roles="control"
    )

    def transition(context: Any, state: Any, control: Any, callback_args: Any) -> Any:
        del context, callback_args
        increment = jnp.sum(control)
        if nonlinear_dynamics:
            increment = increment + 0.05 * state[0] ** 2
        return jnp.asarray([state[0] + increment])

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id=f"{problem_id}:dynamics",
    )
    return NonlinearOpenLoopGameProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(
            jnp.arange(horizon + 1, dtype="float64"),
            time_id=f"{problem_id}:time",
        ),
        jnp.asarray([initial_state]),
        partition,
        stage_costs=stage_costs,
        terminal_costs=terminal_costs,
        constraints=constraints,
        args=args,
        problem_id=problem_id,
    )


def _zero_terminal(time: Any, state: Any, args: Any) -> float:
    del time, state, args
    return 0.0


def _problem_with_state_layout(
    state_layout: Any, initial_state: Any, problem_id: Any
) -> Any:
    partition = PlayerControlPartition(("one",), (1,))
    input_layout = phx.dynamics.InputLayout((1,), roles="control")

    def transition(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, control, args
        return state

    def stage_cost(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state, args
        return 0.5 * control[0] ** 2

    system = phx.dynamics.DiscreteSystem(
        transition,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id=f"{problem_id}:dynamics",
    )
    return NonlinearOpenLoopGameProblem(
        phx.control.DiscreteControlDynamics(system),
        phx.dynamics.TimeGrid(
            jnp.asarray([0.0, 1.0]),
            time_id=f"{problem_id}:time",
        ),
        initial_state,
        partition,
        stage_costs=(stage_cost,),
        terminal_costs=(_zero_terminal,),
        problem_id=problem_id,
    )


def test_games_open_loop_kkt_scenario_1() -> None:
    local_space = phx.linalg.ArraySpace((6,), dtype=jnp.float32)
    state_layout = phx.dynamics.StateLayout(
        (7,),
        geometry=phx.metrix.QuaternionPoseStateGeometry(),
        local_space=local_space,
        tangent_space=local_space,
        layout_id="test:open-loop-game:quaternion-pose",
    )
    problem = _problem_with_state_layout(
        state_layout,
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7]),
        "open-loop-game-quaternion-rejection",
    )
    plan = plan_open_loop_game_kkt(problem)

    with pytest.raises(
        ValueError,
        match="storage-coordinate trajectories require trivial Euclidean geometry",
    ) as error:
        prepare_open_loop_game_kkt(plan, problem, jnp.zeros((1, 1)))

    message = str(error.value)
    assert "trivial=False" in message
    assert "point_size=7, local_size=6, tangent_size=6" in message
    state_layout = phx.dynamics.StateLayout(
        (2,),
        local_space=phx.linalg.ArraySpace((1,), dtype=jnp.float32),
        tangent_space=phx.linalg.ArraySpace((2,), dtype=jnp.float32),
        layout_id="test:open-loop-game:unequal-euclidean-spaces",
    )
    problem = _problem_with_state_layout(
        state_layout,
        jnp.zeros((2,)),
        "open-loop-game-unequal-space-rejection",
    )
    plan = plan_open_loop_game_kkt(problem)

    with pytest.raises(
        ValueError,
        match="equal point/local/tangent sizes",
    ) as error:
        prepare_open_loop_game_kkt(plan, problem, jnp.zeros((1, 1)))

    message = str(error.value)
    assert "trivial=True" in message
    assert "point_size=2, local_size=1, tangent_size=2" in message
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    shared = _trajectory_block(
        lambda trajectory, args: trajectory.final_state[0],
        "shared-terminal",
        site=GameConstraintSite.TERMINAL,
        owner=None,
        participants=("left", "right"),
        scope=GameConstraintScope.SHARED,
        equality=True,
    )
    constraints = OpenLoopGameConstraints(partition, (shared,))
    with pytest.raises(ValueError, match="only private"):
        _problem(
            partition,
            constraints,
            (lambda c, x, u, a: u[0] ** 2, lambda c, x, u, a: u[1] ** 2),
            (_zero_terminal, _zero_terminal),
            problem_id="reject-shared",
        )
    partition = PlayerControlPartition(("one",), (1,))
    constraints = OpenLoopGameConstraints(
        partition,
        (
            _path_block(
                lambda time, state, control, args: control[0],
                "u-at-most-zero",
            ),
            _path_block(
                lambda time, state, control, args: 1.0 - control[0],
                "u-at-least-one",
            ),
        ),
    )
    problem = _problem(
        partition,
        constraints,
        (lambda context, state, control, args: 0.5 * control[0] ** 2,),
        (_zero_terminal,),
        problem_id="infeasible-private",
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 1)))

    assert result.status == int(OpenLoopGameKKTStatus.PRIMAL_INFEASIBLE)
    assert not result.valid
    assert not result.feasible
    assert result.feasibility.maximum_violation > 0.0
    assert result.feasibility.status == int(GameFeasibilityStatus.INFEASIBLE)
    assert np.isfinite(np.asarray(result.feasibility.maximum_violation))


def test_one_player_active_inequality_has_original_private_kkt_evidence() -> None:
    partition = PlayerControlPartition(("one",), (1,))
    block = _path_block(
        lambda time, state, control, args: control[0] - args["limit"],
        "upper-control",
    )
    constraints = OpenLoopGameConstraints(partition, (block,))

    def stage(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state
        return 0.5 * (control[0] - args["target"]) ** 2

    problem = _problem(
        partition,
        constraints,
        (stage,),
        (_zero_terminal,),
        args={"target": jnp.asarray(2.0), "limit": jnp.asarray(1.0)},
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 1)))

    assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
    assert result.certificate_label == LOCAL_NOMINAL_NASH_STATIONARY
    np.testing.assert_allclose(result.controls, [[1.0]], atol=2.0e-5)
    np.testing.assert_allclose(result.inequality_multipliers, [1.0], atol=2.0e-5)
    np.testing.assert_allclose(result.private_multipliers[0], [1.0], atol=2.0e-5)
    assert result.original_stationarity_residual < 1.0e-6
    assert result.original_inequality_violation < 1.0e-7
    assert result.original_dual_violation < 1.0e-7
    assert result.original_ncp_residual < 1.0e-6
    assert result.original_complementarity_residual < 1.0e-6
    assert result.original_kkt_residual < 1.0e-6
    assert result.feasibility.status == int(GameFeasibilityStatus.FEASIBLE)
    assert result.dynamics_valid
    assert result.constraint_qualification_satisfied


def test_two_player_product_game_uses_owned_rows_not_a_summed_objective() -> None:
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    constraints = OpenLoopGameConstraints(partition)

    def left_cost(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state
        return 0.5 * (control[0] - args["targets"][0]) ** 2 + 9.0 * control[1]

    def right_cost(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state
        return 0.5 * (control[1] - args["targets"][1]) ** 2 - 7.0 * control[0]

    problem = _problem(
        partition,
        constraints,
        (left_cost, right_cost),
        (_zero_terminal, _zero_terminal),
        args={"targets": jnp.asarray([1.25, -0.75])},
        problem_id="two-player-product",
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 2)))

    assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
    assert result.certificate_label == LOCAL_NOMINAL_NASH_STATIONARY
    assert result.constraint_scope == "product-local-private-player-feasible-sets"
    np.testing.assert_allclose(result.controls, [[1.25, -0.75]], atol=2.0e-5)
    np.testing.assert_allclose(result.original_stationarity, [0.0, 0.0], atol=1.0e-6)
    assert result.multipliers.shape == (0,)


def test_opponent_dependent_player_owned_constraint_is_private_gne() -> None:
    partition = PlayerControlPartition(("left", "right"), (1, 1))
    coupled = _path_block(
        lambda time, state, control, args: control[0] + control[1] - 1.0,
        "left-owned-capacity",
        owner="left",
        participants=("left", "right"),
        scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
        control_dependencies=("left", "right"),
    )
    constraints = OpenLoopGameConstraints(partition, (coupled,))

    def left_cost(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state, args
        return 0.5 * (control[0] - 2.0) ** 2

    def right_cost(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state, args
        return 0.5 * control[1] ** 2

    problem = _problem(
        partition,
        constraints,
        (left_cost, right_cost),
        (_zero_terminal, _zero_terminal),
        problem_id="player-owned-gne",
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 2)))

    assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
    assert result.certificate_label == LOCAL_NOMINAL_GNE_STATIONARY
    assert result.constraint_scope == "opponent-dependent-private-player-feasible-sets"
    np.testing.assert_allclose(result.controls, [[1.0, 0.0]], atol=3.0e-5)
    np.testing.assert_allclose(result.private_multipliers[0], [1.0], atol=3.0e-5)
    assert result.private_multipliers[1].shape == (0,)


def test_path_terminal_and_whole_trajectory_constraints() -> None:
    for kind in ("path-equality", "terminal-equality", "trajectory-inequality"):
        partition = PlayerControlPartition(("one",), (1,))
        if kind == "path-equality":
            block = _path_block(
                lambda time, state, control, args: control[0] - 0.25,
                "path-equality",
                equality=True,
            )
        elif kind == "terminal-equality":
            block = _trajectory_block(
                lambda trajectory, args: trajectory.final_state[0] - 1.0,
                "terminal-equality",
                site=GameConstraintSite.TERMINAL,
                equality=True,
            )
        else:
            block = _trajectory_block(
                lambda trajectory, args: jnp.sum(trajectory.controls[..., 0]) - 0.5,
                "whole-trajectory-cap",
                site=GameConstraintSite.TRAJECTORY,
                state_dependent=False,
                control_dependencies=("one",),
            )
        constraints = OpenLoopGameConstraints(partition, (block,))

        def stage(context: Any, state: Any, control: Any, args: Any) -> Any:
            del context, state, args
            return 0.5 * (control[0] - 1.0) ** 2

        problem = _problem(
            partition,
            constraints,
            (stage,),
            (_zero_terminal,),
            horizon=2,
            problem_id=f"site-{kind}",
        )
        result = solve_open_loop_game_kkt(problem, jnp.zeros((2, 1)))

        assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
        assert result.feasibility.feasible
        assert result.original_equality_residual < 2.0e-6
        assert result.original_inequality_violation < 2.0e-6
        if kind == "path-equality":
            np.testing.assert_allclose(result.controls[:, 0], [0.25, 0.25], atol=2.0e-5)
            assert result.equality_multipliers.shape == (2,)
        elif kind == "terminal-equality":
            np.testing.assert_allclose(result.states[-1, 0], 1.0, atol=2.0e-5)
            assert result.equality_multipliers.shape == (1,)
        else:
            np.testing.assert_allclose(jnp.sum(result.controls), 0.5, atol=2.0e-5)
            assert result.inequality_multipliers[0] > 0.0


def test_inactive_and_active_inequality_multipliers() -> None:
    for constraint, target, expected_control, active in (
        ("inactive", 0.0, 0.0, False),
        ("active", 2.0, 1.0, True),
    ):
        partition = PlayerControlPartition(("one",), (1,))
        block = _path_block(
            lambda time, state, control, args: control[0] - 1.0,
            f"{constraint}-upper",
        )

        def stage(context: Any, state: Any, control: Any, args: Any) -> Any:
            del context, state
            return 0.5 * (control[0] - args) ** 2

        problem = _problem(
            partition,
            OpenLoopGameConstraints(partition, (block,)),
            (stage,),
            (_zero_terminal,),
            args=jnp.asarray(target),
            problem_id=f"multiplier-{constraint}",
        )
        result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 1)))

        assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
        np.testing.assert_allclose(result.controls[0, 0], expected_control, atol=2.0e-5)
        if active:
            assert result.inequality_multipliers[0] > 0.9
        else:
            np.testing.assert_allclose(result.inequality_multipliers, [0.0], atol=2.0e-6)


def test_degenerate_active_constraint_reports_failed_constraint_qualification() -> None:
    partition = PlayerControlPartition(("one",), (1,))
    degenerate = _path_block(
        lambda time, state, control, args: control[0] ** 2,
        "degenerate-feasible-set",
    )

    def stage(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state, args
        return 0.5 * control[0] ** 2

    problem = _problem(
        partition,
        OpenLoopGameConstraints(partition, (degenerate,)),
        (stage,),
        (_zero_terminal,),
        problem_id="degenerate-cq",
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 1)))

    assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
    np.testing.assert_array_equal(result.active_constraint_count, [1])
    np.testing.assert_array_equal(result.active_constraint_rank, [0])
    np.testing.assert_array_equal(result.constraint_qualification, [False])
    assert not result.constraint_qualification_satisfied
    np.testing.assert_allclose(result.inequality_multipliers, [0.0], atol=1.0e-8)


def test_games_open_loop_kkt_scenario_2() -> None:
    partition = PlayerControlPartition(("one",), (1,))
    nonfinite = _path_block(
        lambda time, state, control, args: jnp.asarray(jnp.nan),
        "nonfinite-residual",
    )
    problem = _problem(
        partition,
        OpenLoopGameConstraints(partition, (nonfinite,)),
        (lambda context, state, control, args: control[0] ** 2,),
        (_zero_terminal,),
        problem_id="nonfinite-private",
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 1)))

    assert result.status == int(OpenLoopGameKKTStatus.NONFINITE)
    assert not result.finite
    assert not result.valid
    assert result.feasibility.status == int(GameFeasibilityStatus.NONFINITE_RESIDUAL)
    assert np.isinf(np.asarray(result.feasibility.maximum_violation))
    partition = PlayerControlPartition(("one",), (1,))
    problem = _problem(
        partition,
        OpenLoopGameConstraints(partition),
        (lambda context, state, control, args: 0.5 * control[0] ** 2,),
        (_zero_terminal,),
        problem_id="claim-scope",
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((1, 1)))

    assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
    assert result.certification_claim == (
        "local nominal open-loop first-order KKT stationarity"
    )
    assert not result.feedback_claim
    assert not result.global_equilibrium_claim
    assert "feedback" not in result.certification_claim
    assert "global" not in result.certification_claim


def test_nested_root_failure_is_not_promoted_to_stationarity() -> None:
    partition = PlayerControlPartition(("one",), (1,))

    def quartic(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state, args
        return 0.25 * control[0] ** 4

    problem = _problem(
        partition,
        OpenLoopGameConstraints(partition),
        (quartic,),
        (_zero_terminal,),
        problem_id="root-failure",
    )
    result = solve_open_loop_game_kkt(
        problem,
        jnp.asarray([[3.0]]),
        termination=NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            maximum_steps=1,
        ),
    )

    assert result.status == int(OpenLoopGameKKTStatus.ROOT_FAILURE)
    assert not result.valid
    assert not result.vi_result.successful
    assert result.original_stationarity_residual > 0.0


def test_whole_horizon_ad_differentiates_future_state_cost_through_rollout() -> None:
    partition = PlayerControlPartition(("one",), (1,))

    def stage(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state, args
        return 0.5 * control[0] ** 2

    def terminal(time: Any, state: Any, args: Any) -> Any:
        del time
        return 0.5 * (state[0] - args["target"]) ** 2

    problem = _problem(
        partition,
        OpenLoopGameConstraints(partition),
        (stage,),
        (terminal,),
        horizon=2,
        args={"target": jnp.asarray(2.0)},
        problem_id="whole-horizon-ad",
        nonlinear_dynamics=True,
    )
    result = solve_open_loop_game_kkt(problem, jnp.zeros((2, 1)))

    assert result.status == int(OpenLoopGameKKTStatus.SUCCESS)
    assert result.controls[0, 0] > 0.0
    assert result.controls[1, 0] > 0.0
    assert result.states[-1, 0] > result.states[1, 0]
    assert result.original_stationarity_residual < 1.0e-6


def test_refresh_and_filtered_jit_preserve_topology_and_change_numeric_solution() -> None:
    partition = PlayerControlPartition(("one",), (1,))

    def stage(context: Any, state: Any, control: Any, args: Any) -> Any:
        del context, state
        return 0.5 * (control[0] - args["target"]) ** 2

    def make_problem(target: Any) -> Any:
        return _problem(
            partition,
            OpenLoopGameConstraints(partition),
            (stage,),
            (_zero_terminal,),
            args={"target": jnp.asarray(target)},
            problem_id="refresh-private-kkt",
        )

    first = make_problem(1.0)
    plan = plan_open_loop_game_kkt(first)
    prepared = prepare_open_loop_game_kkt(plan, first, jnp.zeros((1, 1)))
    first_result = eqx.filter_jit(solve_prepared_open_loop_game_kkt)(prepared)
    refreshed = refresh_open_loop_game_kkt(
        prepared,
        make_problem(-2.0),
        jnp.zeros((1, 1)),
    )
    second_result = eqx.filter_jit(solve_prepared_open_loop_game_kkt)(refreshed)

    np.testing.assert_allclose(first_result.controls, [[1.0]], atol=2.0e-5)
    np.testing.assert_allclose(second_result.controls, [[-2.0]], atol=2.0e-5)
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.numeric_version == prepared.numeric_version + 1


def test_weighted_stationarity_and_owner_qualification_match_finite_differences() -> None:
    from phydrax.control.games._open_loop_kkt import (
        _constraint_qualification,
        _kkt_quantities,
        _KKTArguments,
        _single_case_values,
    )

    partition = PlayerControlPartition(("one", "two"), (1, 1))
    constraints = OpenLoopGameConstraints(
        partition,
        (
            _path_block(
                lambda time, state, control, args: (
                    control[0] + 2.0 * control[1] + 0.1 * state[0]
                ),
                "first-owner-equality",
                owner="one",
                participants=("one", "two"),
                scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
                equality=True,
                state_dependent=True,
                control_dependencies=("one", "two"),
            ),
            _path_block(
                lambda time, state, control, args: 3.0 * control[0] + control[1] - 0.5,
                "second-owner-inequality",
                owner="two",
                participants=("one", "two"),
                scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
                control_dependencies=("one", "two"),
            ),
        ),
    )
    problem = _problem(
        partition,
        constraints,
        (
            lambda context, state, control, args: (
                control[0] ** 2 + 0.3 * control[1] ** 2 + state[0] ** 2
            ),
            lambda context, state, control, args: (
                2.0 * control[1] ** 2
                + 0.7 * control[0] * control[1]
                + 0.2 * state[0] ** 2
            ),
        ),
        (_zero_terminal, _zero_terminal),
        horizon=2,
        nonlinear_dynamics=True,
    )
    problem = NonlinearOpenLoopGameProblem(
        problem.dynamics,
        problem.time_grid,
        jnp.asarray([[0.1], [0.4]], dtype=jnp.float64),
        partition,
        stage_costs=problem.stage_costs,
        terminal_costs=problem.terminal_costs,
        constraints=constraints,
        problem_id="owner-oracle-batched",
    )
    plan = plan_open_loop_game_kkt(problem)
    arguments = _KKTArguments(plan, problem, None)
    controls = jnp.asarray(
        [[0.1, 0.2, -0.1, 0.5], [0.1, 0.2, -0.1, 0.5]], dtype=jnp.float64
    )
    equalities = jnp.asarray([[0.4, -0.7], [-0.2, 0.9]], dtype=jnp.float64)
    inequalities = jnp.asarray([[0.8, 0.3], [0.1, 0.6]], dtype=jnp.float64)
    operator, auxiliary = _kkt_quantities((controls, equalities, inequalities), arguments)
    raw, combined = auxiliary[1], auxiliary[4]
    reference = np.zeros(controls.shape, dtype=np.float64)
    expected_rank = np.zeros((2, 2), dtype=np.int32)
    expected_count = np.zeros((2, 2), dtype=np.int32)
    for case in range(2):
        primal = np.asarray(controls[case])
        derivative = np.empty((2 + raw.shape[-1], primal.size), dtype=np.float64)
        for column in range(primal.size):
            delta = np.zeros_like(primal)
            delta[column] = 1e-5
            plus, _ = _single_case_values(
                problem, None, problem.initial_state[case], jnp.asarray(primal + delta)
            )
            minus, _ = _single_case_values(
                problem, None, problem.initial_state[case], jnp.asarray(primal - delta)
            )
            derivative[:, column] = (np.asarray(plus) - np.asarray(minus)) / 2e-5
        for player, owners in enumerate(plan.owned_control_indices):
            seed = np.zeros(derivative.shape[0], dtype=np.float64)
            seed[player] = 1.0
            rows = []
            flags = []
            for block_index in plan.multiplier_layout.player_block_indices[player]:
                start, stop = plan.constraint_layout.block_slices[block_index]
                rows.extend(range(start, stop))
                flags.extend(
                    [plan.constraint_layout.equalities[block_index]] * (stop - start)
                )
            for multiplier, (row, owner_indices) in enumerate(
                zip(
                    plan.multiplier_physical_rows,
                    plan.multiplier_owner_indices,
                    strict=True,
                )
            ):
                if owner_indices == owners:
                    seed[2 + row] += float(combined[case, multiplier])
            reference[case, list(owners)] = (seed @ derivative)[list(owners)]
            active = np.asarray(flags) | (
                np.abs(np.asarray(raw[case])[rows])
                <= max(
                    plan.feasibility_tolerance, plan.constraint_qualification_tolerance
                )
            )
            expected_count[case, player] = np.sum(active)
            expected_rank[case, player] = np.linalg.matrix_rank(
                derivative[2 + np.asarray(rows), :][:, list(owners)] * active[:, None],
                tol=1e-6,
            )
    np.testing.assert_allclose(operator[0], reference, atol=2e-8, rtol=2e-8)
    rank, count, qualified, finite = _constraint_qualification(arguments, controls, raw)
    np.testing.assert_array_equal(rank, expected_rank)
    np.testing.assert_array_equal(count, expected_count)
    np.testing.assert_array_equal(qualified, expected_rank == expected_count)
    assert jnp.all(finite)
