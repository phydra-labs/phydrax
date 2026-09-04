#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private-constraint nonlinear open-loop game KKT solves.

The dynamics are eliminated by an exact discrete single-shooting rollout.  The
remaining mixed-complementarity variables are the joint open-loop controls and
one private multiplier for every declared player-owned constraint residual.
This module provides local nominal first-order evidence only.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite, prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._bounds import Bounds
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import DiscreteStepContext, TimeGrid
from ...nonlinear import (
    NonlinearTermination,
    prepare_variational_inequality,
    PreparedVariationalInequalitySolve,
    refresh_variational_inequality,
    SemismoothNewton,
    solve_prepared_variational_inequality,
    VariationalInequalityProblem,
    VariationalInequalityResult,
)
from .._dynamics import DiscreteControlDynamics
from .._trajectory_optimization import TrajectoryOptimizationView
from ._constraints import (
    evaluate_game_feasibility,
    GameConstraintLayout,
    GameConstraintScope,
    GameFeasibilityEvidence,
    GameFeasibilityStatus,
    GameMultiplierLayout,
    OpenLoopGameConstraints,
)
from ._layout import PlayerControlPartition
from ._nonlinear import GameStageCost, GameTerminalCost


LOCAL_NOMINAL_NASH_STATIONARY = "LOCAL_NOMINAL_NASH_STATIONARY"
LOCAL_NOMINAL_GNE_STATIONARY = "LOCAL_NOMINAL_GNE_STATIONARY"
_UNSET = object()
_METHOD_ID = "control:game:nonlinear-open-loop-private-kkt:single-shooting:v1"
_STAGE_COST_SEMANTICS = "unweighted-discrete-stage-sum"


class OpenLoopGameKKTStatus(IntEnum):
    """Stable outcomes for a private-constraint nonlinear game KKT solve."""

    SUCCESS = 0
    ROOT_FAILURE = 1
    ORIGINAL_KKT_FAILURE = 2
    PRIMAL_INFEASIBLE = 3
    NONFINITE = 4
    DYNAMICS_FAILURE = 5


class NonlinearOpenLoopGameProblem(StrictModule):
    """A deterministic nonlinear finite-horizon open-loop game.

    Each player's complete objective is the unweighted sum of its scalar stage
    callbacks plus its scalar terminal callback.  Constraints may be
    ``PLAYER_LOCAL`` or ``PLAYER_OWNED_COUPLED``.  Physically shared blocks are
    intentionally rejected: generic GNEs in this module always carry private,
    player-owned multipliers and never a common variational multiplier.
    """

    dynamics: DiscreteControlDynamics
    time_grid: TimeGrid
    initial_state: Array
    partition: PlayerControlPartition
    constraints: OpenLoopGameConstraints
    stage_costs: tuple[GameStageCost, ...] = eqx.field(static=True)
    terminal_costs: tuple[GameTerminalCost, ...] = eqx.field(static=True)
    args: Any
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    stage_cost_semantics: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: DiscreteControlDynamics,
        time_grid: TimeGrid,
        initial_state: ArrayLike,
        partition: PlayerControlPartition,
        /,
        *,
        stage_costs: Sequence[GameStageCost],
        terminal_costs: Sequence[GameTerminalCost],
        constraints: OpenLoopGameConstraints | None = None,
        args: Any = None,
        problem_id: str,
    ):
        if not isinstance(dynamics, DiscreteControlDynamics):
            raise TypeError("dynamics must be DiscreteControlDynamics.")
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if not isinstance(partition, PlayerControlPartition):
            raise TypeError("partition must be a PlayerControlPartition.")
        if len(dynamics.state_shape) != 1:
            raise ValueError("Nonlinear open-loop games require rank-one states.")
        if len(dynamics.control_shape) != 1:
            raise ValueError("Nonlinear open-loop games require rank-one controls.")
        if dynamics.control_shape != (partition.joint_control_size,):
            raise ValueError(
                "partition joint control size must match dynamics control_shape."
            )

        state = jnp.asarray(initial_state)
        if state.ndim < 1 or tuple(state.shape[-1:]) != dynamics.state_shape:
            raise ValueError(
                "initial_state must have shape case_shape + "
                f"{dynamics.state_shape}; got {state.shape}."
            )
        if jnp.issubdtype(state.dtype, jnp.complexfloating):
            raise TypeError("initial_state must be real-valued.")
        if not jnp.issubdtype(state.dtype, jnp.inexact):
            state = state.astype(float)
        cases = tuple(int(size) for size in state.shape[:-1])
        if any(size <= 0 for size in cases):
            raise ValueError("case_shape dimensions must be positive.")

        stage = tuple(stage_costs)
        terminal = tuple(terminal_costs)
        if len(stage) != partition.num_players:
            raise ValueError("stage_costs must provide exactly one callback per player.")
        if len(terminal) != partition.num_players:
            raise ValueError(
                "terminal_costs must provide exactly one callback per player."
            )
        if any(not callable(callback) for callback in stage):
            raise TypeError("Every stage_costs entry must be callable.")
        if any(not callable(callback) for callback in terminal):
            raise TypeError("Every terminal_costs entry must be callable.")

        constraints_ = (
            OpenLoopGameConstraints(partition) if constraints is None else constraints
        )
        if not isinstance(constraints_, OpenLoopGameConstraints):
            raise TypeError("constraints must be OpenLoopGameConstraints or None.")
        if constraints_.partition.partition_id != partition.partition_id:
            raise ValueError("constraints and problem must use the same partition.")
        if any(
            block.scope is GameConstraintScope.SHARED for block in constraints_.blocks
        ):
            raise ValueError(
                "Nonlinear open-loop KKT accepts only private PLAYER_LOCAL or "
                "PLAYER_OWNED_COUPLED blocks; SHARED constraints require an "
                "explicit shared-constraint formulation."
            )
        identifier = _identifier(problem_id, "problem_id")

        self.dynamics = dynamics
        self.time_grid = time_grid
        self.initial_state = state
        self.partition = partition
        self.constraints = constraints_
        self.stage_costs = stage
        self.terminal_costs = terminal
        self.args = args
        self.case_shape = cases
        self.state_size = dynamics.state_shape[0]
        self.control_size = dynamics.control_shape[0]
        self.horizon = time_grid.num_steps
        self.num_players = partition.num_players
        self.problem_id = identifier
        self.dynamics_id = dynamics.dynamics_id
        self.stage_cost_semantics = _STAGE_COST_SEMANTICS


class OpenLoopGameKKTPlan(StrictModule):
    """Fixed private-multiplier topology and semismooth solver policy."""

    partition: PlayerControlPartition
    constraint_layout: GameConstraintLayout
    multiplier_layout: GameMultiplierLayout
    method: SemismoothNewton
    termination: NonlinearTermination
    owned_control_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    multiplier_physical_rows: tuple[int, ...] = eqx.field(static=True)
    multiplier_owner_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    equality_positions: tuple[int, ...] = eqx.field(static=True)
    inequality_positions: tuple[int, ...] = eqx.field(static=True)
    equality_physical_rows: tuple[int, ...] = eqx.field(static=True)
    inequality_physical_rows: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_control_variables: int = eqx.field(static=True)
    num_equalities: int = eqx.field(static=True)
    num_inequalities: int = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    kkt_tolerance: float = eqx.field(static=True)
    constraint_qualification_tolerance: float = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    constraint_scope: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedOpenLoopGameKKT(StrictModule):
    """Prepared nonlinear private KKT mixed-complementarity solve."""

    plan: OpenLoopGameKKTPlan
    problem: NonlinearOpenLoopGameProblem
    initial_controls: Array
    initial_equality_multipliers: Array
    initial_inequality_multipliers: Array
    initial_finite: Array
    vi_problem: VariationalInequalityProblem
    vi_prepared: PreparedVariationalInequalitySolve
    constraint_args: Any
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class OpenLoopGameKKTResult(StrictModule):
    """Candidate trajectory and original unscaled local KKT evidence.

    ``valid`` certifies only the reported local nominal first-order system.  It
    is neither a feedback-equilibrium claim nor a global equilibrium claim.
    """

    partition: PlayerControlPartition
    multiplier_layout: GameMultiplierLayout
    time_grid: TimeGrid
    controls: Array
    states: Array
    player_costs: Array
    equality_multipliers: Array
    inequality_multipliers: Array
    multipliers: Array
    private_multipliers: tuple[Array, ...]
    original_stationarity: Array
    original_constraint_residuals: Array
    original_equality_residuals: Array
    original_inequality_residuals: Array
    original_primal_residuals: Array
    original_dual_residuals: Array
    original_dual_violations: Array
    original_ncp_residuals: Array
    original_complementarity: Array
    original_stationarity_residual: Array
    original_equality_residual: Array
    original_inequality_violation: Array
    original_primal_residual: Array
    original_dual_violation: Array
    original_ncp_residual: Array
    original_complementarity_residual: Array
    original_kkt_residual: Array
    feasibility: GameFeasibilityEvidence
    active_constraint_rank: Array
    active_constraint_count: Array
    constraint_qualification: Array
    constraint_qualification_satisfied: Array
    dynamics_valid: Array
    finite: Array
    feasible: Array
    valid: Array
    status: Array
    vi_result: VariationalInequalityResult
    certificate_label: str = eqx.field(static=True)
    certification_claim: str = eqx.field(static=True)
    constraint_scope: str = eqx.field(static=True)
    feedback_claim: bool = eqx.field(static=True)
    global_equilibrium_claim: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


class _KKTArguments(StrictModule):
    plan: OpenLoopGameKKTPlan
    problem: NonlinearOpenLoopGameProblem
    constraint_args: Any


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _positive_tolerance(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _exact_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
    dtype: jnp.dtype,
    /,
) -> Array:
    array = _real_array(value, name).astype(dtype)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array


def _real_scalar(value: ArrayLike, owner: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{owner} must return one scalar.")
    if jnp.issubdtype(scalar.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must return a real scalar.")
    return scalar if jnp.issubdtype(scalar.dtype, jnp.inexact) else scalar.astype(float)


def _owned_indices(
    partition: PlayerControlPartition, horizon: int, /
) -> tuple[tuple[int, ...], ...]:
    width = partition.joint_control_size
    return tuple(
        tuple(
            stage * width + component
            for stage in range(horizon)
            for component in range(start, stop)
        )
        for start, stop in partition.control_slices
    )


def _multiplier_metadata(
    layout: GameMultiplierLayout,
    owned: tuple[tuple[int, ...], ...],
    /,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    constraint_layout = layout.constraint_layout
    blocks = constraint_layout.constraints.blocks
    rows: list[int] = []
    owners: list[tuple[int, ...]] = []
    equalities: list[bool] = []
    for player, block_indices in enumerate(layout.player_block_indices):
        for block_index in block_indices:
            start, stop = constraint_layout.block_slices[block_index]
            for physical_row in range(start, stop):
                rows.append(physical_row)
                owners.append(owned[player])
                equalities.append(blocks[block_index].equality)
    if layout.shared_block_indices or layout.shared_slice[0] != layout.shared_slice[1]:
        raise ValueError("Private nonlinear KKT cannot allocate shared multipliers.")
    if len(rows) != constraint_layout.num_residuals or len(set(rows)) != len(rows):
        raise ValueError(
            "Private multiplier allocation must contain every physical constraint "
            "residual exactly once."
        )
    equality_positions = tuple(i for i, equality in enumerate(equalities) if equality)
    inequality_positions = tuple(
        i for i, equality in enumerate(equalities) if not equality
    )
    return (
        tuple(rows),
        tuple(owners),
        equality_positions,
        inequality_positions,
        tuple(rows[i] for i in equality_positions),
        tuple(rows[i] for i in inequality_positions),
    )


def plan_open_loop_game_kkt(
    problem: NonlinearOpenLoopGameProblem,
    /,
    *,
    method: SemismoothNewton | None = None,
    termination: NonlinearTermination | None = None,
    feasibility_tolerance: float = 1.0e-7,
    kkt_tolerance: float = 1.0e-6,
    constraint_qualification_tolerance: float = 1.0e-9,
) -> OpenLoopGameKKTPlan:
    """Plan a private constrained nonlinear open-loop Nash/GNE KKT solve."""

    if not isinstance(problem, NonlinearOpenLoopGameProblem):
        raise TypeError("problem must be NonlinearOpenLoopGameProblem.")
    tolerances = (
        _positive_tolerance(feasibility_tolerance, "feasibility_tolerance"),
        _positive_tolerance(kkt_tolerance, "kkt_tolerance"),
        _positive_tolerance(
            constraint_qualification_tolerance,
            "constraint_qualification_tolerance",
        ),
    )
    method_ = (
        SemismoothNewton(
            feasibility="preserve-box",
            certification_tolerance=tolerances[1],
        )
        if method is None
        else method
    )
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(method_, SemismoothNewton):
        raise TypeError("method must be SemismoothNewton or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")

    layout = problem.constraints.layout(num_path_sites=problem.horizon)
    multipliers = layout.multiplier_layout(variational=False)
    owned = _owned_indices(problem.partition, problem.horizon)
    metadata = _multiplier_metadata(multipliers, owned)
    coupled = any(
        block.scope is GameConstraintScope.PLAYER_OWNED_COUPLED
        for block in problem.constraints.blocks
    )
    certificate = (
        LOCAL_NOMINAL_GNE_STATIONARY if coupled else LOCAL_NOMINAL_NASH_STATIONARY
    )
    scope = (
        "opponent-dependent-private-player-feasible-sets"
        if coupled
        else "product-local-private-player-feasible-sets"
    )
    payload = {
        "kind": "nonlinear-open-loop-private-game-kkt",
        "problem_id": problem.problem_id,
        "dynamics_id": problem.dynamics_id,
        "constraints_id": problem.constraints.constraints_id,
        "time_id": problem.time_grid.time_id,
        "partition_id": problem.partition.partition_id,
        "case_shape": problem.case_shape,
        "horizon": problem.horizon,
        "state_size": problem.state_size,
        "control_size": problem.control_size,
        "multiplier_layout": multipliers.layout_id,
        "formulation": method_.formulation,
        "feasibility": method_.feasibility,
        "tolerances": tolerances,
        "certificate": certificate,
    }
    return OpenLoopGameKKTPlan(
        problem.partition,
        layout,
        multipliers,
        method_,
        termination_,
        owned,
        *metadata,
        problem.case_shape,
        problem.horizon,
        problem.state_size,
        problem.control_size,
        problem.horizon * problem.control_size,
        len(metadata[2]),
        len(metadata[3]),
        *tolerances,
        certificate,
        scope,
        problem.problem_id,
        problem.dynamics_id,
        problem.constraints.constraints_id,
        problem.time_grid.time_id,
        f"open-loop-game-kkt-plan:{canonical_fingerprint(payload)}",
    )


def _validate_topology(
    plan: OpenLoopGameKKTPlan,
    problem: NonlinearOpenLoopGameProblem,
    /,
) -> None:
    if not isinstance(plan, OpenLoopGameKKTPlan):
        raise TypeError("plan must be OpenLoopGameKKTPlan.")
    if not isinstance(problem, NonlinearOpenLoopGameProblem):
        raise TypeError("problem must be NonlinearOpenLoopGameProblem.")
    valid = (
        plan.problem_id == problem.problem_id
        and plan.dynamics_id == problem.dynamics_id
        and plan.constraints_id == problem.constraints.constraints_id
        and plan.time_id == problem.time_grid.time_id
        and plan.partition.partition_id == problem.partition.partition_id
        and plan.case_shape == problem.case_shape
        and plan.horizon == problem.horizon
        and plan.state_size == problem.state_size
        and plan.control_size == problem.control_size
    )
    if not valid:
        raise ValueError(
            "Open-loop game KKT plan and problem topology identities do not match."
        )


def _validate_storage_coordinate_geometry(
    problem: NonlinearOpenLoopGameProblem,
    /,
) -> None:
    layout = problem.dynamics.system.state_layout
    if (
        not layout.geometry.trivial
        or layout.size != layout.local_size
        or layout.size != layout.tangent_size
    ):
        raise ValueError(
            "Open-loop game KKT storage-coordinate trajectories require trivial "
            "Euclidean geometry and equal point/local/tangent sizes; got "
            f"geometry_id={layout.geometry.geometry_id!r}, "
            f"trivial={layout.geometry.trivial}, point_size={layout.size}, "
            f"local_size={layout.local_size}, tangent_size={layout.tangent_size}."
        )


def _rollout_and_costs_single(
    problem: NonlinearOpenLoopGameProblem,
    initial_state: Array,
    controls: Array,
    /,
) -> tuple[Array, Array]:
    def step(state: Array, stage_data: tuple[Array, Array]):
        step_index, control = stage_data
        context = DiscreteStepContext(
            problem.time_grid.times[step_index],
            problem.time_grid.times[step_index + 1],
            step_index,
        )
        costs = jnp.stack(
            tuple(
                _real_scalar(
                    callback(context, state, control, problem.args),
                    f"stage_costs[{player}]",
                )
                for player, callback in enumerate(problem.stage_costs)
            )
        )
        next_state = problem.dynamics.system.evaluate(
            context,
            state,
            problem.args,
            inputs=control,
        )
        if next_state.shape != (problem.state_size,):
            raise ValueError("dynamics must return the declared rank-one state shape.")
        return next_state, (next_state, costs)

    terminal_state, (state_tail, stage_costs) = jax.lax.scan(
        step,
        initial_state,
        (jnp.arange(problem.horizon, dtype=jnp.int32), controls),
    )
    terminal_costs = jnp.stack(
        tuple(
            _real_scalar(
                callback(problem.time_grid.times[-1], terminal_state, problem.args),
                f"terminal_costs[{player}]",
            )
            for player, callback in enumerate(problem.terminal_costs)
        )
    )
    states = jnp.concatenate((initial_state[None, :], state_tail), axis=0)
    return states, jnp.sum(stage_costs, axis=0) + terminal_costs


def _single_case_values(
    problem: NonlinearOpenLoopGameProblem,
    constraint_args: Any,
    initial_state: Array,
    flat_controls: Array,
    /,
):
    controls = flat_controls.reshape((problem.horizon, problem.control_size))
    states, costs = _rollout_and_costs_single(problem, initial_state, controls)
    trajectory = TrajectoryOptimizationView(
        problem.time_grid.times,
        states,
        controls,
        case_shape=(),
        state_shape=(problem.state_size,),
        control_shape=(problem.control_size,),
        approximation_id="control:game:exact-discrete-single-shooting",
    )
    evidence = evaluate_game_feasibility(
        problem.constraints,
        trajectory,
        constraint_args,
    )
    raw = (
        jnp.concatenate(
            tuple(
                value.reshape((size,))
                for value, size in zip(
                    evidence.raw_residuals,
                    evidence.layout.block_sizes,
                    strict=True,
                )
            ),
            axis=0,
        )
        if evidence.layout.num_blocks
        else jnp.zeros((0,), dtype=flat_controls.dtype)
    )
    values = jnp.concatenate((costs, raw), axis=0)
    return values, (values, states, evidence.block_finite)


def _differentiate_candidate(
    problem: NonlinearOpenLoopGameProblem,
    flat_controls: Array,
    constraint_args: Any,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    count = prod(problem.case_shape) if problem.case_shape else 1
    initial = problem.initial_state.reshape((count, problem.state_size))
    controls = flat_controls.reshape((count, -1))

    def evaluate_case(case_initial: Array, case_controls: Array):
        return _single_case_values(
            problem,
            constraint_args,
            case_initial,
            case_controls,
        )

    differentiate = jax.jacrev(evaluate_case, argnums=1, has_aux=True)
    jacobian, (values, states, block_finite) = jax.vmap(differentiate)(initial, controls)
    values = values.reshape(problem.case_shape + (values.shape[-1],))
    jacobian = jacobian.reshape(
        problem.case_shape + (values.shape[-1], controls.shape[-1])
    )
    states = states.reshape(
        problem.case_shape + (problem.horizon + 1, problem.state_size)
    )
    block_finite = block_finite.reshape(
        problem.case_shape + (len(problem.constraints.blocks),)
    )
    costs = values[..., : problem.num_players]
    raw = values[..., problem.num_players :]
    return costs, raw, jacobian, states, block_finite


def _kkt_quantities(
    state: tuple[Array, Array, Array],
    arguments: _KKTArguments,
    /,
):
    controls, equality_variables, inequality_variables = state
    plan = arguments.plan
    costs, raw, jacobian, states, block_finite = _differentiate_candidate(
        arguments.problem,
        controls,
        arguments.constraint_args,
    )
    cost_jacobian = jacobian[..., : arguments.problem.num_players, :]
    constraint_jacobian = jacobian[..., arguments.problem.num_players :, :]
    stationarity = jnp.zeros_like(controls)
    for player, indices in enumerate(plan.owned_control_indices):
        index_array = jnp.asarray(indices, dtype=jnp.int32)
        stationarity = stationarity.at[..., index_array].set(
            jnp.take(cost_jacobian[..., player, :], index_array, axis=-1)
        )

    equality_multipliers = equality_variables[..., : plan.num_equalities]
    inequality_multipliers = inequality_variables[..., : plan.num_inequalities]
    combined = jnp.zeros(
        plan.case_shape + (plan.multiplier_layout.num_multipliers,),
        dtype=controls.dtype,
    )
    combined = combined.at[
        ..., jnp.asarray(plan.equality_positions, dtype=jnp.int32)
    ].set(equality_multipliers)
    combined = combined.at[
        ..., jnp.asarray(plan.inequality_positions, dtype=jnp.int32)
    ].set(inequality_multipliers)
    for multiplier, (physical_row, owner_indices) in enumerate(
        zip(
            plan.multiplier_physical_rows,
            plan.multiplier_owner_indices,
            strict=True,
        )
    ):
        if owner_indices:
            owners = jnp.asarray(owner_indices, dtype=jnp.int32)
            contribution = combined[..., multiplier, None] * jnp.take(
                constraint_jacobian[..., physical_row, :], owners, axis=-1
            )
            stationarity = stationarity.at[..., owners].add(contribution)

    equality = jnp.take(
        raw,
        jnp.asarray(plan.equality_physical_rows, dtype=jnp.int32),
        axis=-1,
    )
    inequality = jnp.take(
        raw,
        jnp.asarray(plan.inequality_physical_rows, dtype=jnp.int32),
        axis=-1,
    )
    equality_operator = (
        equality if plan.num_equalities else jnp.zeros_like(equality_variables)
    )
    slack_operator = (
        -inequality if plan.num_inequalities else jnp.zeros_like(inequality_variables)
    )
    return (
        stationarity,
        equality_operator,
        slack_operator,
    ), (
        costs,
        raw,
        constraint_jacobian,
        states,
        block_finite,
        combined,
        equality,
        inequality,
    )


def _kkt_operator(state, arguments, /):
    operator, _ = _kkt_quantities(state, arguments)
    return operator


def _vi_problem_and_state(
    plan: OpenLoopGameKKTPlan,
    problem: NonlinearOpenLoopGameProblem,
    controls: Array,
    equality_multipliers: Array,
    inequality_multipliers: Array,
    constraint_args: Any,
    /,
):
    flat_controls = controls.reshape(plan.case_shape + (plan.num_control_variables,))
    equality_variables = (
        equality_multipliers
        if plan.num_equalities
        else jnp.zeros(plan.case_shape + (1,), dtype=controls.dtype)
    )
    inequality_variables = (
        inequality_multipliers
        if plan.num_inequalities
        else jnp.zeros(plan.case_shape + (1,), dtype=controls.dtype)
    )
    lower = (
        jnp.full_like(flat_controls, -jnp.inf),
        (
            jnp.full_like(equality_variables, -jnp.inf)
            if plan.num_equalities
            else jnp.zeros_like(equality_variables)
        ),
        jnp.zeros_like(inequality_variables),
    )
    upper = (
        jnp.full_like(flat_controls, jnp.inf),
        (
            jnp.full_like(equality_variables, jnp.inf)
            if plan.num_equalities
            else jnp.zeros_like(equality_variables)
        ),
        (
            jnp.full_like(inequality_variables, jnp.inf)
            if plan.num_inequalities
            else jnp.zeros_like(inequality_variables)
        ),
    )
    vi_problem = VariationalInequalityProblem(
        _kkt_operator,
        Bounds(lower, upper),
        problem_id=f"{plan.problem_id}:private-open-loop-game-kkt",
    )
    return (
        vi_problem,
        (flat_controls, equality_variables, inequality_variables),
        _KKTArguments(plan, problem, constraint_args),
    )


def _case_all_finite(value: Array, case_rank: int, /) -> Array:
    axes = tuple(range(case_rank, value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes) if axes else jnp.isfinite(value)


def _initial_finite(
    problem: NonlinearOpenLoopGameProblem,
    controls: Array,
    equality_multipliers: Array,
    inequality_multipliers: Array,
    /,
) -> Array:
    rank = len(problem.case_shape)
    finite = _case_all_finite(problem.initial_state, rank)
    for value in (controls, equality_multipliers, inequality_multipliers):
        finite = finite & _case_all_finite(value, rank)
    return finite


def prepare_open_loop_game_kkt(
    plan: OpenLoopGameKKTPlan,
    problem: NonlinearOpenLoopGameProblem,
    initial_controls: ArrayLike,
    /,
    *,
    initial_equality_multipliers: ArrayLike | None = None,
    initial_inequality_multipliers: ArrayLike | None = None,
    constraint_args: Any = _UNSET,
) -> PreparedOpenLoopGameKKT:
    """Prepare the private nonlinear KKT mixed-complementarity solve."""

    _validate_topology(plan, problem)
    _validate_storage_coordinate_geometry(problem)
    resolved_constraint_args = (
        problem.args if constraint_args is _UNSET else constraint_args
    )
    dtype = problem.initial_state.dtype
    controls = _exact_array(
        initial_controls,
        problem.case_shape + (problem.horizon, problem.control_size),
        "initial_controls",
        dtype,
    )
    equality = (
        jnp.zeros(problem.case_shape + (plan.num_equalities,), dtype=dtype)
        if initial_equality_multipliers is None
        else _exact_array(
            initial_equality_multipliers,
            problem.case_shape + (plan.num_equalities,),
            "initial_equality_multipliers",
            dtype,
        )
    )
    inequality = (
        jnp.zeros(problem.case_shape + (plan.num_inequalities,), dtype=dtype)
        if initial_inequality_multipliers is None
        else _exact_array(
            initial_inequality_multipliers,
            problem.case_shape + (plan.num_inequalities,),
            "initial_inequality_multipliers",
            dtype,
        )
    )
    vi_problem, initial_state, arguments = _vi_problem_and_state(
        plan,
        problem,
        controls,
        equality,
        inequality,
        resolved_constraint_args,
    )
    vi_prepared = prepare_variational_inequality(
        vi_problem,
        initial_state,
        method=plan.method,
        termination=plan.termination,
        args=arguments,
    )
    prepared_id = "prepared-open-loop-game-kkt:" + canonical_fingerprint(
        {"plan_id": plan.plan_id, "dtype": str(dtype)}
    )
    return PreparedOpenLoopGameKKT(
        plan,
        problem,
        controls,
        equality,
        inequality,
        _initial_finite(problem, controls, equality, inequality),
        vi_problem,
        vi_prepared,
        resolved_constraint_args,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_open_loop_game_kkt(
    prepared: PreparedOpenLoopGameKKT,
    problem: NonlinearOpenLoopGameProblem,
    initial_controls: ArrayLike | None = None,
    /,
    *,
    initial_equality_multipliers: ArrayLike | None = None,
    initial_inequality_multipliers: ArrayLike | None = None,
    constraint_args: Any = _UNSET,
) -> PreparedOpenLoopGameKKT:
    """Refresh numeric data while preserving KKT and multiplier topology."""

    if not isinstance(prepared, PreparedOpenLoopGameKKT):
        raise TypeError("prepared must be PreparedOpenLoopGameKKT.")
    _validate_topology(prepared.plan, problem)
    _validate_storage_coordinate_geometry(problem)
    dtype = problem.initial_state.dtype
    controls = (
        prepared.initial_controls
        if initial_controls is None
        else _exact_array(
            initial_controls,
            problem.case_shape + (problem.horizon, problem.control_size),
            "initial_controls",
            dtype,
        )
    )
    equality = (
        prepared.initial_equality_multipliers
        if initial_equality_multipliers is None
        else _exact_array(
            initial_equality_multipliers,
            problem.case_shape + (prepared.plan.num_equalities,),
            "initial_equality_multipliers",
            dtype,
        )
    )
    inequality = (
        prepared.initial_inequality_multipliers
        if initial_inequality_multipliers is None
        else _exact_array(
            initial_inequality_multipliers,
            problem.case_shape + (prepared.plan.num_inequalities,),
            "initial_inequality_multipliers",
            dtype,
        )
    )
    args = prepared.constraint_args if constraint_args is _UNSET else constraint_args
    vi_problem, initial_state, arguments = _vi_problem_and_state(
        prepared.plan,
        problem,
        controls,
        equality,
        inequality,
        args,
    )
    vi_prepared = refresh_variational_inequality(
        prepared.vi_prepared,
        vi_problem,
        initial_state,
        args=arguments,
    )
    return PreparedOpenLoopGameKKT(
        prepared.plan,
        problem,
        controls,
        equality,
        inequality,
        _initial_finite(problem, controls, equality, inequality),
        vi_problem,
        vi_prepared,
        args,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def _maximum_abs(value: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.zeros(value.shape[:-1], dtype=value.dtype)
    return jnp.max(jnp.abs(value), axis=-1)


def _feasibility_evidence(
    plan: OpenLoopGameKKTPlan,
    raw: Array,
    block_finite: Array,
    /,
) -> GameFeasibilityEvidence:
    raw_blocks = tuple(
        raw[..., start:stop].reshape(plan.case_shape + output_shape)
        for (start, stop), output_shape in zip(
            plan.constraint_layout.block_slices,
            plan.constraint_layout.block_output_shapes,
            strict=True,
        )
    )
    violations = tuple(
        jnp.where(
            jnp.isfinite(value),
            jnp.abs(value) if equality else jnp.maximum(value, 0.0),
            jnp.inf,
        )
        for value, equality in zip(
            raw_blocks,
            plan.constraint_layout.equalities,
            strict=True,
        )
    )
    block_maximum = []
    for value in violations:
        axes = tuple(range(len(plan.case_shape), value.ndim))
        block_maximum.append(jnp.max(value, axis=axes))
    if plan.constraint_layout.num_blocks:
        maximum_by_block = jnp.stack(block_maximum, axis=-1)
        maximum_by_block = jnp.where(block_finite, maximum_by_block, jnp.inf)
        feasible_by_block = block_finite & (
            maximum_by_block <= plan.feasibility_tolerance
        )
        finite = jnp.all(block_finite, axis=-1)
        feasible = jnp.all(feasible_by_block, axis=-1)
        maximum = jnp.max(maximum_by_block, axis=-1)
    else:
        dtype = raw.dtype
        maximum_by_block = jnp.zeros(plan.case_shape + (0,), dtype=dtype)
        block_finite = jnp.ones(plan.case_shape + (0,), dtype=bool)
        feasible_by_block = jnp.ones(plan.case_shape + (0,), dtype=bool)
        finite = jnp.ones(plan.case_shape, dtype=bool)
        feasible = jnp.ones(plan.case_shape, dtype=bool)
        maximum = jnp.zeros(plan.case_shape, dtype=dtype)
    incidence = plan.constraint_layout.feasibility_incidence.astype(jnp.int32)
    player_valid = (
        ein.contract("...b,bp->...p", (~block_finite).astype(jnp.int32), incidence) == 0
    )
    player_feasible = (
        ein.contract("...b,bp->...p", (~feasible_by_block).astype(jnp.int32), incidence)
        == 0
    )
    status = jnp.where(
        ~finite,
        int(GameFeasibilityStatus.NONFINITE_RESIDUAL),
        jnp.where(
            feasible,
            int(GameFeasibilityStatus.FEASIBLE),
            int(GameFeasibilityStatus.INFEASIBLE),
        ),
    ).astype(jnp.int32)
    return GameFeasibilityEvidence(
        layout=plan.constraint_layout,
        raw_residuals=raw_blocks,
        violations=violations,
        block_maximum_violation=maximum_by_block,
        block_finite=block_finite,
        block_feasible=feasible_by_block,
        player_valid=player_valid,
        player_feasible=player_feasible,
        maximum_violation=maximum,
        finite=finite,
        feasible=feasible,
        valid=finite,
        status=status,
        case_shape=plan.case_shape,
        tolerance=plan.feasibility_tolerance,
    )


def _rank(matrix: Array, tolerance: float, /) -> Array:
    if matrix.shape[-2] == 0 or matrix.shape[-1] == 0:
        return jnp.zeros(matrix.shape[:-2], dtype=jnp.int32)
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    maximum = jnp.max(singular, axis=-1, keepdims=True)
    threshold = jnp.asarray(tolerance, dtype=matrix.dtype) * (1.0 + maximum)
    return jnp.sum(singular > threshold, axis=-1, dtype=jnp.int32)


def _constraint_qualification(
    plan: OpenLoopGameKKTPlan,
    raw: Array,
    constraint_jacobian: Array,
    /,
) -> tuple[Array, Array, Array]:
    ranks = []
    counts = []
    qualifications = []
    blocks = plan.constraint_layout.constraints.blocks
    active_tolerance = max(
        plan.feasibility_tolerance,
        plan.constraint_qualification_tolerance,
    )
    for player, owner_indices in enumerate(plan.owned_control_indices):
        physical_rows: list[int] = []
        equality_flags: list[bool] = []
        for block_index in plan.multiplier_layout.player_block_indices[player]:
            start, stop = plan.constraint_layout.block_slices[block_index]
            physical_rows.extend(range(start, stop))
            equality_flags.extend([blocks[block_index].equality] * (stop - start))
        rows = jnp.asarray(tuple(physical_rows), dtype=jnp.int32)
        owners = jnp.asarray(owner_indices, dtype=jnp.int32)
        matrix = jnp.take(
            jnp.take(constraint_jacobian, rows, axis=-2),
            owners,
            axis=-1,
        )
        equality = jnp.asarray(tuple(equality_flags), dtype=bool)
        values = jnp.take(raw, rows, axis=-1)
        active = equality | (jnp.abs(values) <= active_tolerance)
        masked = matrix * active[..., :, None]
        rank = _rank(masked, plan.constraint_qualification_tolerance)
        count = jnp.sum(active, axis=-1, dtype=jnp.int32)
        ranks.append(rank)
        counts.append(count)
        qualifications.append(rank == count)
    return (
        jnp.stack(ranks, axis=-1),
        jnp.stack(counts, axis=-1),
        jnp.stack(qualifications, axis=-1),
    )


def solve_prepared_open_loop_game_kkt(
    prepared: PreparedOpenLoopGameKKT,
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> OpenLoopGameKKTResult:
    """Solve and certify one prepared local nominal private KKT candidate."""

    if not isinstance(prepared, PreparedOpenLoopGameKKT):
        raise TypeError("prepared must be PreparedOpenLoopGameKKT.")
    if termination is not None and not isinstance(termination, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    vi_result = solve_prepared_variational_inequality(
        prepared.vi_prepared,
        termination=termination,
    )
    flat_controls, equality_variables, inequality_variables = vi_result.state
    plan = prepared.plan
    equality_multipliers = equality_variables[..., : plan.num_equalities]
    inequality_multipliers = inequality_variables[..., : plan.num_inequalities]
    arguments = _KKTArguments(plan, prepared.problem, prepared.constraint_args)
    stationarity_operator, auxiliary = _kkt_quantities(
        (flat_controls, equality_variables, inequality_variables),
        arguments,
    )
    stationarity = stationarity_operator[0]
    (
        costs,
        raw,
        constraint_jacobian,
        states,
        block_finite,
        multipliers,
        equality,
        inequality,
    ) = auxiliary
    controls = flat_controls.reshape(plan.case_shape + (plan.horizon, plan.control_size))
    private = tuple(
        multipliers[..., start:stop]
        for start, stop in plan.multiplier_layout.player_slices
    )
    primal_residuals = jnp.zeros_like(raw)
    for (start, stop), equality_block in zip(
        plan.constraint_layout.block_slices,
        plan.constraint_layout.equalities,
        strict=True,
    ):
        values = raw[..., start:stop]
        violation = jnp.abs(values) if equality_block else jnp.maximum(values, 0.0)
        primal_residuals = primal_residuals.at[..., start:stop].set(violation)
    dual_residuals = -inequality_multipliers
    dual_violations = jnp.maximum(dual_residuals, 0.0)
    slack = -inequality
    ncp = jnp.hypot(inequality_multipliers, slack) - inequality_multipliers - slack
    complementarity = inequality_multipliers * inequality
    stationarity_residual = _maximum_abs(stationarity)
    equality_residual = _maximum_abs(equality)
    inequality_violation = _maximum_abs(jnp.maximum(inequality, 0.0))
    primal_residual = jnp.maximum(equality_residual, inequality_violation)
    dual_violation = _maximum_abs(dual_violations)
    ncp_residual = _maximum_abs(ncp)
    complementarity_residual = _maximum_abs(complementarity)
    original_kkt = jnp.maximum(
        jnp.maximum(stationarity_residual, equality_residual),
        jnp.maximum(
            inequality_violation,
            jnp.maximum(
                dual_violation,
                jnp.maximum(ncp_residual, complementarity_residual),
            ),
        ),
    )
    feasibility = _feasibility_evidence(plan, raw, block_finite)
    active_rank, active_count, cq = _constraint_qualification(
        plan,
        raw,
        constraint_jacobian,
    )
    cq_satisfied = jnp.all(cq, axis=-1)

    case_rank = len(plan.case_shape)
    dynamics_valid = _case_all_finite(controls, case_rank) & _case_all_finite(
        states, case_rank
    )
    output_finite = jnp.ones(plan.case_shape, dtype=bool)
    for value in (
        controls,
        states,
        costs,
        equality_multipliers,
        inequality_multipliers,
        stationarity,
        raw,
        constraint_jacobian,
        ncp,
        original_kkt,
    ):
        output_finite = output_finite & _case_all_finite(value, case_rank)
    nested_finite = vi_result.certificate.finite
    finite = prepared.initial_finite & output_finite & feasibility.finite & nested_finite
    vi_ok = vi_result.successful & vi_result.certificate.certified
    kkt_ok = original_kkt <= plan.kkt_tolerance
    feasible = feasibility.feasible

    status = jnp.full(
        plan.case_shape,
        int(OpenLoopGameKKTStatus.SUCCESS),
        dtype=jnp.int32,
    )
    status = jnp.where(
        ~kkt_ok,
        int(OpenLoopGameKKTStatus.ORIGINAL_KKT_FAILURE),
        status,
    )
    status = jnp.where(
        ~vi_ok,
        int(OpenLoopGameKKTStatus.ROOT_FAILURE),
        status,
    )
    status = jnp.where(
        ~feasible,
        int(OpenLoopGameKKTStatus.PRIMAL_INFEASIBLE),
        status,
    )
    status = jnp.where(
        ~finite,
        int(OpenLoopGameKKTStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        ~dynamics_valid,
        int(OpenLoopGameKKTStatus.DYNAMICS_FAILURE),
        status,
    ).astype(jnp.int32)
    valid = status == int(OpenLoopGameKKTStatus.SUCCESS)
    return OpenLoopGameKKTResult(
        prepared.problem.partition,
        plan.multiplier_layout,
        prepared.problem.time_grid,
        controls,
        states,
        costs,
        equality_multipliers,
        inequality_multipliers,
        multipliers,
        private,
        stationarity,
        raw,
        equality,
        inequality,
        primal_residuals,
        dual_residuals,
        dual_violations,
        ncp,
        complementarity,
        stationarity_residual,
        equality_residual,
        inequality_violation,
        primal_residual,
        dual_violation,
        ncp_residual,
        complementarity_residual,
        original_kkt,
        feasibility,
        active_rank,
        active_count,
        cq,
        cq_satisfied,
        dynamics_valid,
        finite,
        feasible,
        valid,
        status,
        vi_result,
        plan.certificate_label,
        "local nominal open-loop first-order KKT stationarity",
        plan.constraint_scope,
        False,
        False,
        plan.problem_id,
        plan.dynamics_id,
        plan.constraints_id,
        plan.time_id,
        plan.plan_id,
        prepared.prepared_id,
        _METHOD_ID,
    )


def solve_open_loop_game_kkt(
    problem: NonlinearOpenLoopGameProblem,
    initial_controls: ArrayLike | None = None,
    /,
    *,
    initial_equality_multipliers: ArrayLike | None = None,
    initial_inequality_multipliers: ArrayLike | None = None,
    constraint_args: Any = _UNSET,
    method: SemismoothNewton | None = None,
    termination: NonlinearTermination | None = None,
    feasibility_tolerance: float = 1.0e-7,
    kkt_tolerance: float = 1.0e-6,
    constraint_qualification_tolerance: float = 1.0e-9,
) -> OpenLoopGameKKTResult:
    """Plan, prepare, solve, and certify a private nonlinear open-loop KKT."""

    if not isinstance(problem, NonlinearOpenLoopGameProblem):
        raise TypeError("problem must be NonlinearOpenLoopGameProblem.")
    plan = plan_open_loop_game_kkt(
        problem,
        method=method,
        termination=termination,
        feasibility_tolerance=feasibility_tolerance,
        kkt_tolerance=kkt_tolerance,
        constraint_qualification_tolerance=constraint_qualification_tolerance,
    )
    controls = (
        jnp.zeros(
            problem.case_shape + (problem.horizon, problem.control_size),
            dtype=problem.initial_state.dtype,
        )
        if initial_controls is None
        else initial_controls
    )
    prepared = prepare_open_loop_game_kkt(
        plan,
        problem,
        controls,
        initial_equality_multipliers=initial_equality_multipliers,
        initial_inequality_multipliers=initial_inequality_multipliers,
        constraint_args=constraint_args,
    )
    return solve_prepared_open_loop_game_kkt(prepared)


__all__ = [
    "LOCAL_NOMINAL_GNE_STATIONARY",
    "LOCAL_NOMINAL_NASH_STATIONARY",
    "NonlinearOpenLoopGameProblem",
    "OpenLoopGameKKTPlan",
    "OpenLoopGameKKTResult",
    "OpenLoopGameKKTStatus",
    "PreparedOpenLoopGameKKT",
    "plan_open_loop_game_kkt",
    "prepare_open_loop_game_kkt",
    "refresh_open_loop_game_kkt",
    "solve_open_loop_game_kkt",
    "solve_prepared_open_loop_game_kkt",
]
