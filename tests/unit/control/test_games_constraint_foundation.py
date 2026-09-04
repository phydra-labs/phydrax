#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control._trajectory_optimization import (
    BoundedPathConstraint,
    BoundedTrajectoryConstraint,
    TrajectoryOptimizationView,
)
from phydrax.control.games._constraints import (
    evaluate_game_feasibility,
    GameConstraintBlock,
    GameConstraintLayout,
    GameConstraintScope,
    GameConstraintSite,
    GameFeasibilityStatus,
    GameMultiplierLayout,
    OpenLoopGameConstraints,
)
from phydrax.control.games._layout import PlayerControlPartition


def _path_block(
    function,
    constraint_id: str,
    *,
    scope: GameConstraintScope,
    participants: tuple[str, ...],
    owner: str | None,
    equality: bool,
    residual_shape: tuple[int, ...] = (),
    control_dependencies: tuple[str, ...] = (),
    time_dependent: bool = False,
    state_dependent: bool = False,
) -> GameConstraintBlock:
    constraint = BoundedPathConstraint(
        function,
        lower=0.0 if equality else -jnp.inf,
        upper=0.0,
        constraint_id=constraint_id,
    )
    return GameConstraintBlock(
        constraint,
        scope=scope,
        participants=participants,
        owner=owner,
        site=GameConstraintSite.PATH,
        equality=equality,
        residual_shape=residual_shape,
        time_dependent=time_dependent,
        state_dependent=state_dependent,
        control_dependencies=control_dependencies,
    )


def _trajectory_block(
    function,
    constraint_id: str,
    *,
    scope: GameConstraintScope,
    participants: tuple[str, ...],
    owner: str | None,
    site: GameConstraintSite,
    equality: bool,
    residual_shape: tuple[int, ...] = (),
    control_dependencies: tuple[str, ...] = (),
    time_dependent: bool = False,
    state_dependent: bool = False,
) -> GameConstraintBlock:
    constraint = BoundedTrajectoryConstraint(
        function,
        lower=0.0 if equality else -jnp.inf,
        upper=0.0,
        constraint_id=constraint_id,
    )
    return GameConstraintBlock(
        constraint,
        scope=scope,
        participants=participants,
        owner=owner,
        site=site,
        equality=equality,
        residual_shape=residual_shape,
        time_dependent=time_dependent,
        state_dependent=state_dependent,
        control_dependencies=control_dependencies,
    )


def test_constraint_and_multiplier_layouts_distinguish_ownership_concepts():
    partition = PlayerControlPartition(("alpha", "beta", "gamma"), (1, 1, 1))
    local = _path_block(
        lambda time, state, control, args: state[0],
        "local-alpha",
        scope=GameConstraintScope.PLAYER_LOCAL,
        participants=("alpha",),
        owner="alpha",
        equality=True,
        state_dependent=True,
    )
    owned_coupled = _path_block(
        lambda time, state, control, args: control[:2],
        "owned-alpha-beta",
        scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
        participants=("alpha", "beta"),
        owner="alpha",
        equality=False,
        residual_shape=(2,),
        control_dependencies=("alpha", "beta"),
    )
    shared_subset = _trajectory_block(
        lambda trajectory, args: trajectory.final_state[..., 0],
        "shared-beta-gamma",
        scope=GameConstraintScope.SHARED,
        participants=("beta", "gamma"),
        owner=None,
        site=GameConstraintSite.TERMINAL,
        equality=False,
        state_dependent=True,
    )
    constraints = OpenLoopGameConstraints(
        partition,
        (local, owned_coupled, shared_subset),
    )

    layout = GameConstraintLayout(constraints, num_path_sites=4)
    assert layout.block_ids == (
        "local-alpha",
        "owned-alpha-beta",
        "shared-beta-gamma",
    )
    assert layout.block_output_shapes == ((4,), (4, 2), ())
    assert layout.block_slices == ((0, 4), (4, 12), (12, 13))
    assert layout.num_residuals == 13
    np.testing.assert_array_equal(
        layout.feasibility_incidence,
        np.asarray(
            (
                (True, False, False),
                (True, False, False),
                (False, True, True),
            )
        ),
    )
    assert constraints.layout(num_path_sites=4).layout_id == layout.layout_id

    generalized = GameMultiplierLayout(layout, variational=False)
    assert generalized.player_block_indices == ((0, 1), (2,), (2,))
    assert generalized.player_slices == ((0, 12), (12, 13), (13, 14))
    assert generalized.player_residual_slices == (
        ((0, 4), (4, 12)),
        ((12, 13),),
        ((12, 13),),
    )
    assert generalized.shared_block_indices == ()
    assert generalized.shared_slice == (14, 14)
    assert generalized.num_multipliers == 14

    variational = layout.multiplier_layout(variational=True)
    assert variational.player_block_indices == ((0, 1), (), ())
    assert variational.player_slices == ((0, 12), (12, 12), (12, 12))
    assert variational.shared_block_indices == (2,)
    assert variational.shared_residual_slices == ((12, 13),)
    assert variational.shared_slice == (12, 13)
    assert variational.num_multipliers == 13
    assert variational.layout_id != generalized.layout_id


def test_constraint_metadata_rejects_ambiguous_ownership_and_bound_forms():
    path = BoundedPathConstraint(
        lambda time, state, control, args: control[0],
        lower=-jnp.inf,
        upper=0.0,
        constraint_id="ambiguous",
    )
    with pytest.raises(ValueError, match="participants to contain only owner"):
        GameConstraintBlock(
            path,
            scope=GameConstraintScope.PLAYER_LOCAL,
            participants=("alpha", "beta"),
            owner="alpha",
            site=GameConstraintSite.PATH,
            equality=False,
            residual_shape=(),
            time_dependent=False,
            state_dependent=False,
            control_dependencies=("alpha",),
        )
    with pytest.raises(ValueError, match="at least two participants"):
        GameConstraintBlock(
            path,
            scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
            participants=("alpha",),
            owner="alpha",
            site=GameConstraintSite.PATH,
            equality=False,
            residual_shape=(),
            time_dependent=False,
            state_dependent=False,
            control_dependencies=("alpha",),
        )

    non_residual_bound = BoundedPathConstraint(
        lambda time, state, control, args: control[0],
        lower=-1.0,
        upper=1.0,
        constraint_id="not-residual-form",
    )
    with pytest.raises(ValueError, match="residual <= 0 bounds"):
        GameConstraintBlock(
            non_residual_bound,
            scope=GameConstraintScope.SHARED,
            participants=("alpha",),
            owner=None,
            site=GameConstraintSite.PATH,
            equality=False,
            residual_shape=(),
            time_dependent=False,
            state_dependent=False,
            control_dependencies=("alpha",),
        )


def test_ordered_constraints_validate_partition_members_and_order():
    partition = PlayerControlPartition(("alpha", "beta"), (1, 1))
    unknown = _trajectory_block(
        lambda trajectory, args: trajectory.final_state[..., 0],
        "unknown-player",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "gamma"),
        owner=None,
        site=GameConstraintSite.TERMINAL,
        equality=True,
        state_dependent=True,
    )
    with pytest.raises(ValueError, match="unknown participant"):
        OpenLoopGameConstraints(partition, (unknown,))

    reversed_participants = _trajectory_block(
        lambda trajectory, args: trajectory.final_state[..., 0],
        "reversed",
        scope=GameConstraintScope.SHARED,
        participants=("beta", "alpha"),
        owner=None,
        site=GameConstraintSite.TERMINAL,
        equality=True,
        state_dependent=True,
    )
    with pytest.raises(ValueError, match="partition order"):
        OpenLoopGameConstraints(partition, (reversed_participants,))


def test_evaluation_preserves_path_terminal_and_trajectory_axes_and_evaluates_shared_once():
    partition = PlayerControlPartition(("alpha", "beta"), (1, 1))
    calls = {"shared_terminal": 0}

    def shared_terminal(trajectory, args):
        calls["shared_terminal"] += 1
        return trajectory.final_state[..., 0]

    blocks = (
        _path_block(
            lambda time, state, control, args: state[0],
            "local-path-equality",
            scope=GameConstraintScope.PLAYER_LOCAL,
            participants=("alpha",),
            owner="alpha",
            equality=True,
            state_dependent=True,
        ),
        _path_block(
            lambda time, state, control, args: jnp.stack(
                (control[0] - args[0], control[1] - args[1])
            ),
            "owned-coupled-path",
            scope=GameConstraintScope.PLAYER_OWNED_COUPLED,
            participants=("alpha", "beta"),
            owner="alpha",
            equality=False,
            residual_shape=(2,),
            control_dependencies=("alpha", "beta"),
        ),
        _trajectory_block(
            shared_terminal,
            "shared-terminal-equality",
            scope=GameConstraintScope.SHARED,
            participants=("alpha", "beta"),
            owner=None,
            site=GameConstraintSite.TERMINAL,
            equality=True,
            state_dependent=True,
        ),
        _trajectory_block(
            lambda trajectory, args: jnp.max(trajectory.controls[..., 1], axis=-1) - 1.0,
            "shared-beta-trajectory",
            scope=GameConstraintScope.SHARED,
            participants=("beta",),
            owner=None,
            site=GameConstraintSite.TRAJECTORY,
            equality=False,
            control_dependencies=("beta",),
        ),
    )
    constraints = OpenLoopGameConstraints(partition, blocks)
    generalized = constraints.layout(num_path_sites=2).multiplier_layout(
        variational=False
    )
    assert generalized.player_block_indices == ((0, 1, 2), (2, 3))

    trajectory = TrajectoryOptimizationView(
        jnp.asarray((0.0, 0.5, 1.0)),
        jnp.asarray(
            (
                ((0.0,), (0.0,), (0.0,)),
                ((0.0,), (0.0,), (0.25,)),
            )
        ),
        jnp.asarray(
            (
                ((0.1, 0.2), (0.2, 0.3)),
                ((0.1, 0.2), (0.2, 0.3)),
            )
        ),
        case_shape=(2,),
        state_shape=(1,),
        control_shape=(2,),
    )
    evidence = evaluate_game_feasibility(
        constraints,
        trajectory,
        jnp.asarray((0.5, 0.5)),
    )

    assert calls["shared_terminal"] == 1
    assert tuple(value.shape for value in evidence.raw_residuals) == (
        (2, 2),
        (2, 2, 2),
        (2,),
        (2,),
    )
    np.testing.assert_allclose(evidence.raw_residuals[2], np.asarray((0.0, 0.25)))
    np.testing.assert_allclose(evidence.violations[1], 0.0)
    assert tuple(value.shape for value in evidence.equality_violations) == (
        (2, 2),
        (2,),
    )
    assert tuple(value.shape for value in evidence.positive_inequality_violations) == (
        (2, 2, 2),
        (2,),
    )
    np.testing.assert_array_equal(evidence.feasible, np.asarray((True, False)))
    np.testing.assert_array_equal(
        evidence.status,
        np.asarray(
            (
                GameFeasibilityStatus.FEASIBLE,
                GameFeasibilityStatus.INFEASIBLE,
            )
        ),
    )
    assert evidence.sampled_only
    assert not evidence.certified
    assert evidence.feasibility_scope == (
        "declared-open-loop-blocks-at-supplied-trajectory-sites"
    )


def test_nonfinite_residual_is_case_local_and_scoped_to_shared_participants():
    partition = PlayerControlPartition(("alpha", "beta"), (1, 1))
    shared_beta = _trajectory_block(
        lambda trajectory, args: trajectory.final_state[..., 0],
        "shared-beta-terminal",
        scope=GameConstraintScope.SHARED,
        participants=("beta",),
        owner=None,
        site=GameConstraintSite.TERMINAL,
        equality=True,
        state_dependent=True,
    )
    constraints = OpenLoopGameConstraints(partition, (shared_beta,))
    trajectory = TrajectoryOptimizationView(
        jnp.asarray((0.0, 1.0)),
        jnp.asarray((((0.0,), (0.25,)), ((0.0,), (jnp.nan,)))),
        jnp.zeros((2, 1, 2)),
        case_shape=(2,),
        state_shape=(1,),
        control_shape=(2,),
    )

    evidence = evaluate_game_feasibility(constraints, trajectory)

    np.testing.assert_array_equal(evidence.valid, np.asarray((True, False)))
    np.testing.assert_array_equal(evidence.feasible, np.asarray((False, False)))
    np.testing.assert_array_equal(
        evidence.status,
        np.asarray(
            (
                GameFeasibilityStatus.INFEASIBLE,
                GameFeasibilityStatus.NONFINITE_RESIDUAL,
            )
        ),
    )
    np.testing.assert_allclose(evidence.maximum_violation[0], 0.25)
    assert np.isinf(np.asarray(evidence.maximum_violation[1]))
    np.testing.assert_array_equal(
        evidence.player_valid,
        np.asarray(((True, True), (True, False))),
    )
    np.testing.assert_array_equal(
        evidence.player_feasible,
        np.asarray(((True, False), (True, False))),
    )
    assert np.isnan(np.asarray(evidence.raw_residuals[0][1]))
    assert np.isinf(np.asarray(evidence.violations[0][1]))


def test_schema_checks_precede_callbacks_and_return_shapes_are_enforced():
    partition = PlayerControlPartition(("alpha", "beta"), (1, 1))
    calls = {"path": 0}

    def path_callback(time, state, control, args):
        calls["path"] += 1
        return control[0]

    block = _path_block(
        path_callback,
        "schema-path",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
        equality=False,
        control_dependencies=("alpha",),
    )
    constraints = OpenLoopGameConstraints(partition, (block,))
    wrong_controls = TrajectoryOptimizationView(
        jnp.asarray((0.0, 1.0)),
        jnp.zeros((2, 1)),
        jnp.zeros((1, 1)),
        case_shape=(),
        state_shape=(1,),
        control_shape=(1,),
    )
    with pytest.raises(ValueError, match="joint-control partition"):
        evaluate_game_feasibility(constraints, wrong_controls)
    assert calls["path"] == 0

    wrong_shape = _trajectory_block(
        lambda trajectory, args: trajectory.final_state[..., 0],
        "wrong-result-shape",
        scope=GameConstraintScope.SHARED,
        participants=("alpha", "beta"),
        owner=None,
        site=GameConstraintSite.TRAJECTORY,
        equality=True,
        residual_shape=(2,),
        state_dependent=True,
    )
    shaped_constraints = OpenLoopGameConstraints(partition, (wrong_shape,))
    valid_trajectory = TrajectoryOptimizationView(
        jnp.asarray((0.0, 1.0)),
        jnp.zeros((2, 1)),
        jnp.zeros((1, 2)),
        case_shape=(),
        state_shape=(1,),
        control_shape=(2,),
    )
    with pytest.raises(ValueError, match="callback must return shape"):
        evaluate_game_feasibility(shaped_constraints, valid_trajectory)
