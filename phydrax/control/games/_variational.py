#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Convex finite-horizon affine LQ open-loop variational equilibria."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._bounds import Bounds
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import TimeGrid
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
from ...optim import (
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexTermination,
    QuadraticProgram,
    solve_quadratic_program,
)
from .._trajectory_optimization import TrajectoryOptimizationView
from ._constraints import (
    evaluate_game_feasibility,
    GameConstraintLayout,
    GameConstraintScope,
    GameMultiplierLayout,
    OpenLoopGameConstraints,
)
from ._layout import PlayerControlPartition


OPEN_LOOP_VARIATIONAL_GNE = "OPEN_LOOP_VARIATIONAL_GNE"
_CERTIFICATION_CLAIM = "numerically certified convex open-loop VE"
_UNSET = object()


class OpenLoopVEStatus(IntEnum):
    """Stable case-local outcomes for an open-loop variational-equilibrium solve."""

    SUCCESS = 0
    STRUCTURAL_INVALIDITY = 1
    CERTIFIED_INFEASIBILITY = 2
    VI_FAILURE = 3
    ORIGINAL_KKT_FAILURE = 4
    PROJECTION_FAILURE = 5
    NONFINITE = 6
    RESIDUAL_VALID_NONISOLATED = 7
    PHASE_I_FAILURE = 8


class FiniteHorizonLQOpenLoopVEProblem(StrictModule):
    r"""A convex affine-dynamics LQ game with one variational shared feasible set.

    The dynamics are ``x[t+1] = A[t] x[t] + B[t] u[t] + c[t]``. Player ``i``
    has the discrete objective

    ``sum_t 1/2 x'Q[i,t]x + x'N[i,t]u + 1/2 u'R[i,t]u``
    ``+ q[i,t]'x + r[i,t]'u + d[i,t]``

    plus the analogous quadratic-affine terminal term. ``R[i,t]`` and
    ``N[i,t]`` act on the complete joint control, so costs may retain arbitrary
    cross-player control terms. Constraint callbacks are required to be affine
    after exact dynamics condensation; preparation audits that requirement.
    """

    dynamics_matrices: Array
    control_matrices: Array
    initial_state: Array
    state_costs: Array
    control_costs: Array
    terminal_state_costs: Array
    dynamics_bias: Array
    state_control_cross: Array
    state_linear: Array
    control_linear: Array
    stage_constants: Array
    terminal_linear: Array
    terminal_constants: Array
    partition: PlayerControlPartition
    constraints: OpenLoopGameConstraints
    time_grid: TimeGrid
    case_shape: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics_matrices: ArrayLike,
        control_matrices: ArrayLike,
        initial_state: ArrayLike,
        state_costs: ArrayLike,
        control_costs: ArrayLike,
        terminal_state_costs: ArrayLike,
        partition: PlayerControlPartition,
        /,
        *,
        constraints: OpenLoopGameConstraints | None = None,
        dynamics_bias: ArrayLike | None = None,
        state_control_cross: ArrayLike | None = None,
        state_linear: ArrayLike | None = None,
        control_linear: ArrayLike | None = None,
        stage_constants: ArrayLike | None = None,
        terminal_linear: ArrayLike | None = None,
        terminal_constants: ArrayLike | None = None,
        time_grid: TimeGrid | None = None,
        problem_id: str = "control:game:lq-open-loop-ve",
        dynamics_id: str = "control:game:dynamics:affine-discrete",
    ):
        if not isinstance(partition, PlayerControlPartition):
            raise TypeError("partition must be a PlayerControlPartition.")
        a = _real_array(dynamics_matrices, "dynamics_matrices")
        if a.ndim < 3 or a.shape[-1] != a.shape[-2]:
            raise ValueError(
                "dynamics_matrices must have shape "
                "case_shape + (horizon, state_size, state_size)."
            )
        case_shape = tuple(int(size) for size in a.shape[:-3])
        horizon = int(a.shape[-3])
        state_size = int(a.shape[-1])
        if horizon < 1 or state_size < 1:
            raise ValueError("horizon and state_size must be positive.")
        b = _real_array(control_matrices, "control_matrices")
        if (
            b.ndim < 3
            or tuple(b.shape[:-3]) != case_shape
            or tuple(b.shape[-3:-1]) != (horizon, state_size)
        ):
            raise ValueError(
                "control_matrices must have shape "
                "case_shape + (horizon, state_size, control_size)."
            )
        control_size = int(b.shape[-1])
        if partition.joint_control_size != control_size:
            raise ValueError(
                "partition joint control size must match control_matrices; "
                f"got {partition.joint_control_size} and {control_size}."
            )
        players = partition.num_players
        required = (
            a,
            b,
            _real_array(initial_state, "initial_state"),
            _real_array(state_costs, "state_costs"),
            _real_array(control_costs, "control_costs"),
            _real_array(terminal_state_costs, "terminal_state_costs"),
        )
        dtype = jnp.result_type(*(value.dtype for value in required), jnp.float32)
        a = _exact_array(
            a,
            case_shape + (horizon, state_size, state_size),
            "dynamics_matrices",
            dtype,
        )
        b = _exact_array(
            b,
            case_shape + (horizon, state_size, control_size),
            "control_matrices",
            dtype,
        )
        initial = _exact_array(
            required[2], case_shape + (state_size,), "initial_state", dtype
        )
        q = _exact_array(
            required[3],
            case_shape + (players, horizon, state_size, state_size),
            "state_costs",
            dtype,
        )
        r = _exact_array(
            required[4],
            case_shape + (players, horizon, control_size, control_size),
            "control_costs",
            dtype,
        )
        q_terminal = _exact_array(
            required[5],
            case_shape + (players, state_size, state_size),
            "terminal_state_costs",
            dtype,
        )
        zeros = lambda shape: jnp.zeros(shape, dtype=dtype)
        c = (
            zeros(case_shape + (horizon, state_size))
            if dynamics_bias is None
            else _exact_array(
                _real_array(dynamics_bias, "dynamics_bias"),
                case_shape + (horizon, state_size),
                "dynamics_bias",
                dtype,
            )
        )
        cross = (
            zeros(case_shape + (players, horizon, state_size, control_size))
            if state_control_cross is None
            else _exact_array(
                _real_array(state_control_cross, "state_control_cross"),
                case_shape + (players, horizon, state_size, control_size),
                "state_control_cross",
                dtype,
            )
        )
        q_linear = (
            zeros(case_shape + (players, horizon, state_size))
            if state_linear is None
            else _exact_array(
                _real_array(state_linear, "state_linear"),
                case_shape + (players, horizon, state_size),
                "state_linear",
                dtype,
            )
        )
        r_linear = (
            zeros(case_shape + (players, horizon, control_size))
            if control_linear is None
            else _exact_array(
                _real_array(control_linear, "control_linear"),
                case_shape + (players, horizon, control_size),
                "control_linear",
                dtype,
            )
        )
        constants = (
            zeros(case_shape + (players, horizon))
            if stage_constants is None
            else _exact_array(
                _real_array(stage_constants, "stage_constants"),
                case_shape + (players, horizon),
                "stage_constants",
                dtype,
            )
        )
        terminal_linear_value = (
            zeros(case_shape + (players, state_size))
            if terminal_linear is None
            else _exact_array(
                _real_array(terminal_linear, "terminal_linear"),
                case_shape + (players, state_size),
                "terminal_linear",
                dtype,
            )
        )
        if terminal_constants is None:
            terminal_constant_value = zeros(case_shape + (players,))
        else:
            terminal_constant_raw = _real_array(terminal_constants, "terminal_constants")
            if terminal_constant_raw.shape == ():
                terminal_constant_raw = jnp.broadcast_to(
                    terminal_constant_raw, case_shape + (players,)
                )
            terminal_constant_value = _exact_array(
                terminal_constant_raw,
                case_shape + (players,),
                "terminal_constants",
                dtype,
            )
        constraints_ = (
            OpenLoopGameConstraints(partition) if constraints is None else constraints
        )
        if not isinstance(constraints_, OpenLoopGameConstraints):
            raise TypeError("constraints must be OpenLoopGameConstraints or None.")
        if constraints_.partition.partition_id != partition.partition_id:
            raise ValueError("constraints and problem must use the same partition.")
        identifier = _identifier(problem_id, "problem_id")
        dynamics_identifier = _identifier(dynamics_id, "dynamics_id")
        if time_grid is None:
            time_grid = TimeGrid(
                jnp.arange(horizon + 1, dtype=dtype),
                time_id=f"{identifier}:time",
            )
        elif not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid or None.")
        if time_grid.num_steps != horizon:
            raise ValueError(
                f"time_grid must contain {horizon + 1} times for this horizon."
            )

        self.dynamics_matrices = a
        self.control_matrices = b
        self.initial_state = initial
        self.state_costs = q
        self.control_costs = r
        self.terminal_state_costs = q_terminal
        self.dynamics_bias = c
        self.state_control_cross = cross
        self.state_linear = q_linear
        self.control_linear = r_linear
        self.stage_constants = constants
        self.terminal_linear = terminal_linear_value
        self.terminal_constants = terminal_constant_value
        self.partition = partition
        self.constraints = constraints_
        self.time_grid = time_grid
        self.case_shape = case_shape
        self.horizon = horizon
        self.state_size = state_size
        self.control_size = control_size
        self.num_players = players
        self.problem_id = identifier
        self.dynamics_id = dynamics_identifier


class OpenLoopVEPlan(StrictModule):
    """Fixed topology and solver policies for one variational GNE tranche."""

    partition: PlayerControlPartition
    constraint_layout: GameConstraintLayout
    multiplier_layout: GameMultiplierLayout
    method: SemismoothNewton
    termination: NonlinearTermination
    phase_one_policy: ConvexSolvePolicy
    projection_policy: ConvexSolvePolicy
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
    structural_tolerance: float = eqx.field(static=True)
    convexity_tolerance: float = eqx.field(static=True)
    monotonicity_tolerance: float = eqx.field(static=True)
    regularity_tolerance: float = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    kkt_tolerance: float = eqx.field(static=True)
    natural_residual_tolerance: float = eqx.field(static=True)
    natural_step: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedOpenLoopVE(StrictModule):
    """Numeric condensed game, audited phase I, and prepared box VI."""

    plan: OpenLoopVEPlan
    problem: FiniteHorizonLQOpenLoopVEProblem
    initial_controls: Array
    state_maps: Array
    state_offsets: Array
    player_hessians: Array
    player_linear: Array
    player_constants: Array
    pseudogradient_matrix: Array
    pseudogradient_linear: Array
    constraint_matrix: Array
    constraint_offset: Array
    equality_matrix: Array
    equality_rhs: Array
    inequality_matrix: Array
    inequality_rhs: Array
    equality_stationarity: Array
    inequality_stationarity: Array
    cost_symmetry_residual: Array
    constraint_affinity_residual: Array
    minimum_own_control_eigenvalues: Array
    minimum_monotonicity_eigenvalue: Array
    input_finite: Array
    structural_valid: Array
    phase_one_result: ConvexProgramResult
    phase_one_residual: Array
    vi_problem: VariationalInequalityProblem
    vi_prepared: PreparedVariationalInequalitySolve
    constraint_args: Any
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class OpenLoopVEResult(StrictModule):
    """Candidate trajectory plus original-scale VE and nonuniqueness evidence."""

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
    shared_multipliers: Array
    original_stationarity_residual: Array
    original_equality_residual: Array
    original_inequality_violation: Array
    original_dual_violation: Array
    original_complementarity_residual: Array
    original_kkt_residual: Array
    natural_residual: Array
    phase_one_residual: Array
    phase_one_certified_infeasible: Array
    cost_symmetry_residual: Array
    constraint_affinity_residual: Array
    structural_valid: Array
    minimum_own_control_eigenvalues: Array
    minimum_monotonicity_eigenvalue: Array
    player_convex: Array
    convexity_certified: Array
    monotone: Array
    strongly_monotone: Array
    active_constraint_rank: Array
    active_constraint_count: Array
    regularity_certified: Array
    isolation_rank: Array
    nonisolation_dimension: Array
    isolation_certified: Array
    nonuniqueness_evidence: Array
    finite: Array
    valid: Array
    status: Array
    vi_result: VariationalInequalityResult
    phase_one_result: ConvexProgramResult
    projection_result: ConvexProgramResult
    certificate_label: str = eqx.field(static=True)
    certification_claim: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    return array


def _exact_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
    dtype: jnp.dtype,
    /,
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array


def _positive_tolerance(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _case_all_finite(value: Array, case_rank: int, /) -> Array:
    axes = tuple(range(case_rank, value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes) if axes else jnp.isfinite(value)


def _maximum_abs(value: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.zeros(value.shape[:-1], dtype=value.dtype)
    return jnp.max(jnp.abs(value), axis=-1)


def _normalized_symmetry(value: Array, event_axes: tuple[int, ...], /) -> Array:
    difference = value - jnp.swapaxes(value, -1, -2)
    numerator = jnp.max(jnp.abs(difference), axis=event_axes)
    scale = jnp.max(jnp.abs(value), axis=event_axes)
    return numerator / (1.0 + scale)


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
    for block_index in layout.shared_block_indices:
        block = blocks[block_index]
        participating = tuple(
            index
            for player, player_id in enumerate(
                layout.constraint_layout.constraints.partition.player_ids
            )
            if player_id in block.participants
            for index in owned[player]
        )
        start, stop = constraint_layout.block_slices[block_index]
        for physical_row in range(start, stop):
            rows.append(physical_row)
            owners.append(participating)
            equalities.append(block.equality)
    if len(rows) != constraint_layout.num_residuals:
        raise ValueError(
            "Variational multiplier allocation must contain every physical constraint "
            "exactly once."
        )
    if len(set(rows)) != len(rows):
        raise ValueError("Variational multiplier allocation duplicated a physical row.")
    equality_positions = tuple(i for i, equality in enumerate(equalities) if equality)
    inequality_positions = tuple(
        i for i, equality in enumerate(equalities) if not equality
    )
    equality_rows = tuple(rows[i] for i in equality_positions)
    inequality_rows = tuple(rows[i] for i in inequality_positions)
    return (
        tuple(rows),
        tuple(owners),
        equality_positions,
        inequality_positions,
        equality_rows,
        inequality_rows,
    )


def plan_open_loop_ve(
    problem: FiniteHorizonLQOpenLoopVEProblem,
    /,
    *,
    method: SemismoothNewton | None = None,
    termination: NonlinearTermination | None = None,
    phase_one_policy: ConvexSolvePolicy | None = None,
    projection_policy: ConvexSolvePolicy | None = None,
    structural_tolerance: float = 1.0e-9,
    convexity_tolerance: float = 1.0e-9,
    monotonicity_tolerance: float = 1.0e-9,
    regularity_tolerance: float = 1.0e-9,
    feasibility_tolerance: float = 1.0e-7,
    kkt_tolerance: float = 1.0e-6,
    natural_residual_tolerance: float = 1.0e-6,
    natural_step: float = 1.0,
) -> OpenLoopVEPlan:
    """Plan one fixed-topology, explicitly variational open-loop GNE solve."""
    if not isinstance(problem, FiniteHorizonLQOpenLoopVEProblem):
        raise TypeError("problem must be FiniteHorizonLQOpenLoopVEProblem.")
    if any(
        block.scope is GameConstraintScope.PLAYER_OWNED_COUPLED
        for block in problem.constraints.blocks
    ):
        raise ValueError(
            "Convex open-loop VE supports player-local and physically shared "
            "polyhedra; player-owned coupled constraints require a generalized "
            "Nash formulation."
        )
    tolerances = (
        _positive_tolerance(structural_tolerance, "structural_tolerance"),
        _positive_tolerance(convexity_tolerance, "convexity_tolerance"),
        _positive_tolerance(monotonicity_tolerance, "monotonicity_tolerance"),
        _positive_tolerance(regularity_tolerance, "regularity_tolerance"),
        _positive_tolerance(feasibility_tolerance, "feasibility_tolerance"),
        _positive_tolerance(kkt_tolerance, "kkt_tolerance"),
        _positive_tolerance(natural_residual_tolerance, "natural_residual_tolerance"),
    )
    step = _positive_tolerance(natural_step, "natural_step")
    method_ = (
        SemismoothNewton(
            feasibility="preserve-box",
            certification_tolerance=tolerances[5],
        )
        if method is None
        else method
    )
    termination_ = NonlinearTermination() if termination is None else termination
    default_qp_policy = ConvexSolvePolicy(
        termination=ConvexTermination(absolute=1.0e-9, maximum_steps=200)
    )
    phase_policy = default_qp_policy if phase_one_policy is None else phase_one_policy
    projection_policy_ = (
        default_qp_policy if projection_policy is None else projection_policy
    )
    if not isinstance(method_, SemismoothNewton):
        raise TypeError("method must be SemismoothNewton or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    if not isinstance(phase_policy, ConvexSolvePolicy):
        raise TypeError("phase_one_policy must be ConvexSolvePolicy or None.")
    if not isinstance(projection_policy_, ConvexSolvePolicy):
        raise TypeError("projection_policy must be ConvexSolvePolicy or None.")
    constraint_layout = problem.constraints.layout(num_path_sites=problem.horizon)
    multiplier_layout = constraint_layout.multiplier_layout(variational=True)
    if not multiplier_layout.variational:
        raise RuntimeError("Open-loop VE planning requires variational multipliers.")
    owned = _owned_indices(problem.partition, problem.horizon)
    metadata = _multiplier_metadata(multiplier_layout, owned)
    num_variables = problem.horizon * problem.control_size
    payload = {
        "kind": "finite-horizon-lq-open-loop-variational-gne",
        "problem_id": problem.problem_id,
        "dynamics_id": problem.dynamics_id,
        "constraints_id": problem.constraints.constraints_id,
        "time_id": problem.time_grid.time_id,
        "partition_id": problem.partition.partition_id,
        "case_shape": problem.case_shape,
        "horizon": problem.horizon,
        "state_size": problem.state_size,
        "control_size": problem.control_size,
        "multiplier_layout": multiplier_layout.layout_id,
        "method_formulation": method_.formulation,
        "phase_policy": phase_policy.policy_id,
        "projection_policy": projection_policy_.policy_id,
        "tolerances": tolerances,
        "natural_step": step,
    }
    return OpenLoopVEPlan(
        problem.partition,
        constraint_layout,
        multiplier_layout,
        method_,
        termination_,
        phase_policy,
        projection_policy_,
        owned,
        *metadata,
        problem.case_shape,
        problem.horizon,
        problem.state_size,
        problem.control_size,
        num_variables,
        len(metadata[2]),
        len(metadata[3]),
        *tolerances,
        step,
        problem.problem_id,
        problem.dynamics_id,
        problem.constraints.constraints_id,
        problem.time_grid.time_id,
        f"open-loop-ve-plan:{canonical_fingerprint(payload)}",
    )


def _validate_topology(
    plan: OpenLoopVEPlan,
    problem: FiniteHorizonLQOpenLoopVEProblem,
    /,
) -> None:
    if not isinstance(plan, OpenLoopVEPlan):
        raise TypeError("plan must be OpenLoopVEPlan.")
    if not isinstance(problem, FiniteHorizonLQOpenLoopVEProblem):
        raise TypeError("problem must be FiniteHorizonLQOpenLoopVEProblem.")
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
            "Open-loop VE plan and problem topology identities do not match."
        )


def _condense_dynamics(
    problem: FiniteHorizonLQOpenLoopVEProblem, /
) -> tuple[Array, Array]:
    n = problem.state_size
    variables = problem.horizon * problem.control_size
    case_shape = problem.case_shape
    dtype = problem.dynamics_matrices.dtype
    mapping = jnp.zeros(case_shape + (n, variables), dtype=dtype)
    offset = problem.initial_state
    mappings = [mapping]
    offsets = [offset]
    for stage in range(problem.horizon):
        mapping = problem.dynamics_matrices[..., stage, :, :] @ mapping
        column = stage * problem.control_size
        mapping = mapping.at[..., :, column : column + problem.control_size].add(
            problem.control_matrices[..., stage, :, :]
        )
        offset = (
            ein.contract(
                "...ij,...j->...i",
                problem.dynamics_matrices[..., stage, :, :],
                offset,
            )
            + problem.dynamics_bias[..., stage, :]
        )
        mappings.append(mapping)
        offsets.append(offset)
    return jnp.stack(mappings, axis=-3), jnp.stack(offsets, axis=-2)


def _condense_costs(
    problem: FiniteHorizonLQOpenLoopVEProblem,
    state_maps: Array,
    state_offsets: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    cases = problem.case_shape
    players = problem.num_players
    variables = problem.horizon * problem.control_size
    dtype = problem.dynamics_matrices.dtype
    hessians = jnp.zeros(cases + (players, variables, variables), dtype=dtype)
    linear = jnp.zeros(cases + (players, variables), dtype=dtype)
    constants = jnp.zeros(cases + (players,), dtype=dtype)
    symmetry = jnp.zeros(cases + (players,), dtype=dtype)
    identity = jnp.eye(variables, dtype=dtype)
    for player in range(players):
        hessian = jnp.zeros(cases + (variables, variables), dtype=dtype)
        affine = jnp.zeros(cases + (variables,), dtype=dtype)
        constant = jnp.zeros(cases, dtype=dtype)
        player_symmetry = jnp.zeros(cases, dtype=dtype)
        for stage in range(problem.horizon):
            state_map = state_maps[..., stage, :, :]
            state_offset = state_offsets[..., stage, :]
            selector = identity[
                stage * problem.control_size : (stage + 1) * problem.control_size
            ]
            q_raw = problem.state_costs[..., player, stage, :, :]
            r_raw = problem.control_costs[..., player, stage, :, :]
            q = 0.5 * (q_raw + jnp.swapaxes(q_raw, -1, -2))
            r = 0.5 * (r_raw + jnp.swapaxes(r_raw, -1, -2))
            cross = problem.state_control_cross[..., player, stage, :, :]
            q_linear = problem.state_linear[..., player, stage, :]
            r_linear = problem.control_linear[..., player, stage, :]
            hessian = hessian + ein.contract(
                "...ai,...ab,...bj->...ij", state_map, q, state_map
            )
            cross_hessian = ein.contract(
                "...ai,...ab,bj->...ij", state_map, cross, selector
            )
            hessian = hessian + cross_hessian + jnp.swapaxes(cross_hessian, -1, -2)
            hessian = hessian + ein.contract("ai,...ab,bj->...ij", selector, r, selector)
            affine = affine + ein.contract(
                "...ai,...ab,...b->...i", state_map, q, state_offset
            )
            affine = affine + ein.contract("...ai,...a->...i", state_map, q_linear)
            affine = affine + ein.contract(
                "ai,...ab,...b->...i",
                selector,
                jnp.swapaxes(cross, -1, -2),
                state_offset,
            )
            affine = affine + ein.contract("ai,...a->...i", selector, r_linear)
            constant = constant + 0.5 * ein.contract(
                "...a,...ab,...b->...", state_offset, q, state_offset
            )
            constant = constant + ein.contract("...a,...a->...", q_linear, state_offset)
            constant = constant + problem.stage_constants[..., player, stage]
            q_residual = _normalized_symmetry(q_raw, (-2, -1))
            r_residual = _normalized_symmetry(r_raw, (-2, -1))
            player_symmetry = jnp.maximum(
                player_symmetry, jnp.maximum(q_residual, r_residual)
            )
        terminal_map = state_maps[..., -1, :, :]
        terminal_offset = state_offsets[..., -1, :]
        terminal_raw = problem.terminal_state_costs[..., player, :, :]
        terminal = 0.5 * (terminal_raw + jnp.swapaxes(terminal_raw, -1, -2))
        terminal_linear = problem.terminal_linear[..., player, :]
        hessian = hessian + ein.contract(
            "...ai,...ab,...bj->...ij", terminal_map, terminal, terminal_map
        )
        affine = affine + ein.contract(
            "...ai,...ab,...b->...i", terminal_map, terminal, terminal_offset
        )
        affine = affine + ein.contract("...ai,...a->...i", terminal_map, terminal_linear)
        constant = constant + 0.5 * ein.contract(
            "...a,...ab,...b->...", terminal_offset, terminal, terminal_offset
        )
        constant = constant + ein.contract(
            "...a,...a->...", terminal_linear, terminal_offset
        )
        constant = constant + problem.terminal_constants[..., player]
        player_symmetry = jnp.maximum(
            player_symmetry, _normalized_symmetry(terminal_raw, (-2, -1))
        )
        hessians = hessians.at[..., player, :, :].set(hessian)
        linear = linear.at[..., player, :].set(affine)
        constants = constants.at[..., player].set(constant)
        symmetry = symmetry.at[..., player].set(player_symmetry)

    pseudogradient = jnp.zeros(cases + (variables, variables), dtype=dtype)
    pseudolinear = jnp.zeros(cases + (variables,), dtype=dtype)
    minimum_own = []
    for player, indices in enumerate(_owned_indices(problem.partition, problem.horizon)):
        index_array = jnp.asarray(indices, dtype=jnp.int32)
        pseudogradient = pseudogradient.at[..., index_array, :].set(
            jnp.take(hessians[..., player, :, :], index_array, axis=-2)
        )
        pseudolinear = pseudolinear.at[..., index_array].set(
            jnp.take(linear[..., player, :], index_array, axis=-1)
        )
        own_hessian = jnp.take(
            jnp.take(hessians[..., player, :, :], index_array, axis=-2),
            index_array,
            axis=-1,
        )
        minimum_own.append(jnp.min(jnp.linalg.eigvalsh(own_hessian), axis=-1))
    symmetric_pseudogradient = 0.5 * (
        pseudogradient + jnp.swapaxes(pseudogradient, -1, -2)
    )
    minimum_monotonicity = jnp.min(jnp.linalg.eigvalsh(symmetric_pseudogradient), axis=-1)
    return (
        hessians,
        linear,
        constants,
        pseudogradient,
        pseudolinear,
        symmetry,
        jnp.stack(minimum_own, axis=-1),
        minimum_monotonicity,
    )


def _trajectory(
    problem: FiniteHorizonLQOpenLoopVEProblem,
    state_maps: Array,
    state_offsets: Array,
    flat_controls: Array,
    /,
) -> TrajectoryOptimizationView:
    controls = flat_controls.reshape(
        problem.case_shape + (problem.horizon, problem.control_size)
    )
    states = ein.contract("...tij,...j->...ti", state_maps, flat_controls) + state_offsets
    return TrajectoryOptimizationView(
        problem.time_grid.times,
        states,
        controls,
        case_shape=problem.case_shape,
        state_shape=(problem.state_size,),
        control_shape=(problem.control_size,),
        approximation_id="control:game:exact-affine-condensation",
    )


def _constraint_values(
    problem: FiniteHorizonLQOpenLoopVEProblem,
    state_maps: Array,
    state_offsets: Array,
    flat_controls: Array,
    constraint_args: Any,
    /,
) -> Array:
    if not problem.constraints.blocks:
        return jnp.zeros(problem.case_shape + (0,), dtype=problem.dynamics_matrices.dtype)
    evidence = evaluate_game_feasibility(
        problem.constraints,
        _trajectory(problem, state_maps, state_offsets, flat_controls),
        constraint_args,
    )
    flattened = tuple(
        raw.reshape(problem.case_shape + (size,))
        for raw, size in zip(
            evidence.raw_residuals,
            evidence.layout.block_sizes,
            strict=True,
        )
    )
    return jnp.concatenate(flattened, axis=-1)


def _constraint_linearization(
    problem: FiniteHorizonLQOpenLoopVEProblem,
    state_maps: Array,
    state_offsets: Array,
    constraint_args: Any,
    /,
) -> tuple[Array, Array, Array]:
    variables = problem.horizon * problem.control_size
    layout = problem.constraints.layout(num_path_sites=problem.horizon)
    residuals = layout.num_residuals
    dtype = problem.dynamics_matrices.dtype
    zero = jnp.zeros(problem.case_shape + (variables,), dtype=dtype)
    if residuals == 0:
        return (
            jnp.zeros(problem.case_shape + (0, variables), dtype=dtype),
            jnp.zeros(problem.case_shape + (0,), dtype=dtype),
            jnp.zeros(problem.case_shape, dtype=dtype),
        )

    evaluate = lambda controls: _constraint_values(
        problem, state_maps, state_offsets, controls, constraint_args
    )
    offset = evaluate(zero)
    full_jacobian = jax.jacfwd(evaluate)(zero)
    count = int(np.prod(problem.case_shape)) if problem.case_shape else 1
    if problem.case_shape:
        full = full_jacobian.reshape((count, residuals, count, variables))
        matrix = jnp.stack(tuple(full[index, :, index, :] for index in range(count)))
        off_diagonal = full * (1.0 - jnp.eye(count, dtype=dtype)[:, None, :, None])
        cross_case = jnp.max(jnp.abs(off_diagonal), axis=(1, 2, 3))
        matrix = matrix.reshape(problem.case_shape + (residuals, variables))
        cross_case = cross_case.reshape(problem.case_shape)
    else:
        matrix = full_jacobian
        cross_case = jnp.asarray(0.0, dtype=dtype)
    base = jnp.linspace(-0.73, 0.91, variables, dtype=dtype)
    probe_one = jnp.broadcast_to(base, problem.case_shape + (variables,))
    probe_two = jnp.broadcast_to(-0.37 * base[::-1] + 0.19, probe_one.shape)
    actual_one = evaluate(probe_one)
    actual_two = evaluate(probe_two)
    predicted_one = ein.contract("...ij,...j->...i", matrix, probe_one) + offset
    predicted_two = ein.contract("...ij,...j->...i", matrix, probe_two) + offset
    residual_one = _maximum_abs(actual_one - predicted_one) / (
        1.0 + _maximum_abs(actual_one)
    )
    residual_two = _maximum_abs(actual_two - predicted_two) / (
        1.0 + _maximum_abs(actual_two)
    )
    ownership = jnp.zeros(problem.case_shape, dtype=dtype)
    owned = _owned_indices(problem.partition, problem.horizon)
    for block, (start, stop) in zip(
        problem.constraints.blocks, layout.block_slices, strict=True
    ):
        relevant_players = (
            (problem.partition.player_ids.index(block.owner),)
            if block.scope is GameConstraintScope.PLAYER_LOCAL
            else tuple(
                player
                for player, player_id in enumerate(problem.partition.player_ids)
                if player_id in block.participants
            )
        )
        relevant = {index for player in relevant_players for index in owned[player]}
        irrelevant = tuple(index for index in range(variables) if index not in relevant)
        if irrelevant:
            block_matrix = matrix[..., start:stop, :]
            outside = jnp.take(
                block_matrix, jnp.asarray(irrelevant, dtype=jnp.int32), axis=-1
            )
            ownership = jnp.maximum(
                ownership,
                jnp.max(jnp.abs(outside), axis=(-2, -1))
                / (1.0 + jnp.max(jnp.abs(block_matrix), axis=(-2, -1))),
            )
    affinity = jnp.maximum(
        ownership,
        jnp.maximum(cross_case, jnp.maximum(residual_one, residual_two)),
    )
    return matrix, offset, affinity


def _lower_constraints(
    plan: OpenLoopVEPlan,
    matrix: Array,
    offset: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    cases = plan.case_shape
    variables = plan.num_control_variables
    dtype = matrix.dtype
    combined_stationarity = jnp.zeros(
        cases + (variables, plan.multiplier_layout.num_multipliers), dtype=dtype
    )
    for multiplier, (physical_row, owner_indices) in enumerate(
        zip(
            plan.multiplier_physical_rows,
            plan.multiplier_owner_indices,
            strict=True,
        )
    ):
        if owner_indices:
            owners = jnp.asarray(owner_indices, dtype=jnp.int32)
            values = jnp.take(matrix[..., physical_row, :], owners, axis=-1)
            combined_stationarity = combined_stationarity.at[..., owners, multiplier].set(
                values
            )
    equality_positions = jnp.asarray(plan.equality_positions, dtype=jnp.int32)
    inequality_positions = jnp.asarray(plan.inequality_positions, dtype=jnp.int32)
    equality_rows = jnp.asarray(plan.equality_physical_rows, dtype=jnp.int32)
    inequality_rows = jnp.asarray(plan.inequality_physical_rows, dtype=jnp.int32)
    equality_matrix = jnp.take(matrix, equality_rows, axis=-2)
    inequality_matrix = jnp.take(matrix, inequality_rows, axis=-2)
    equality_rhs = -jnp.take(offset, equality_rows, axis=-1)
    inequality_rhs = -jnp.take(offset, inequality_rows, axis=-1)
    equality_stationarity = jnp.take(combined_stationarity, equality_positions, axis=-1)
    inequality_stationarity = jnp.take(
        combined_stationarity, inequality_positions, axis=-1
    )
    return (
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        equality_stationarity,
        inequality_stationarity,
    )


def _constraint_residual_norm(
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    controls: Array,
    /,
) -> Array:
    equality = ein.contract("...ij,...j->...i", equality_matrix, controls) - equality_rhs
    inequality = (
        ein.contract("...ij,...j->...i", inequality_matrix, controls) - inequality_rhs
    )
    return jnp.maximum(
        _maximum_abs(equality),
        _maximum_abs(jnp.maximum(inequality, 0.0)),
    )


def _phase_one(
    plan: OpenLoopVEPlan,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    dtype: jnp.dtype,
    /,
) -> ConvexProgramResult:
    program = QuadraticProgram(
        jnp.eye(plan.num_control_variables, dtype=dtype),
        jnp.zeros(plan.case_shape + (plan.num_control_variables,), dtype=dtype),
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_rhs=inequality_rhs,
        problem_id=f"{plan.plan_id}:phase-i-feasibility",
        convexity_evidence="identity phase-I objective; original unscaled constraints",
    )
    return solve_quadratic_program(program, policy=plan.phase_one_policy)


def _kkt_operator(state, args, /):
    controls, equality_variables, inequality_variables = state
    (
        pseudogradient,
        pseudolinear,
        equality_stationarity,
        inequality_stationarity,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
    ) = args
    num_equalities = equality_matrix.shape[-2]
    num_inequalities = inequality_matrix.shape[-2]
    equality_multipliers = equality_variables[..., :num_equalities]
    inequality_multipliers = inequality_variables[..., :num_inequalities]
    stationarity = (
        ein.contract("...ij,...j->...i", pseudogradient, controls)
        + pseudolinear
        + ein.contract("...ij,...j->...i", equality_stationarity, equality_multipliers)
        + ein.contract(
            "...ij,...j->...i", inequality_stationarity, inequality_multipliers
        )
    )
    equality = ein.contract("...ij,...j->...i", equality_matrix, controls) - equality_rhs
    slack = inequality_rhs - ein.contract("...ij,...j->...i", inequality_matrix, controls)
    if num_equalities == 0:
        equality = jnp.zeros_like(equality_variables)
    if num_inequalities == 0:
        slack = jnp.zeros_like(inequality_variables)
    return stationarity, equality, slack


def _safe(value: Array, /) -> Array:
    return jnp.where(jnp.isfinite(value), value, 0.0)


def _numeric_preparation(
    plan: OpenLoopVEPlan,
    problem: FiniteHorizonLQOpenLoopVEProblem,
    initial_controls: Array,
    initial_equality_multipliers: Array | None,
    initial_inequality_multipliers: Array | None,
    constraint_args: Any,
    /,
):
    state_maps, state_offsets = _condense_dynamics(problem)
    (
        player_hessians,
        player_linear,
        player_constants,
        pseudogradient,
        pseudolinear,
        symmetry,
        minimum_own,
        minimum_monotonicity,
    ) = _condense_costs(problem, state_maps, state_offsets)
    constraint_matrix, constraint_offset, affinity = _constraint_linearization(
        problem, state_maps, state_offsets, constraint_args
    )
    lowered = _lower_constraints(plan, constraint_matrix, constraint_offset)
    equality_matrix, equality_rhs, inequality_matrix, inequality_rhs = lowered[:4]
    case_rank = len(problem.case_shape)
    finite = jnp.ones(problem.case_shape, dtype=bool)
    for value in (
        problem.dynamics_matrices,
        problem.control_matrices,
        problem.initial_state,
        problem.state_costs,
        problem.control_costs,
        problem.terminal_state_costs,
        problem.dynamics_bias,
        problem.state_control_cross,
        problem.state_linear,
        problem.control_linear,
        problem.stage_constants,
        problem.terminal_linear,
        problem.terminal_constants,
        constraint_matrix,
        constraint_offset,
    ):
        finite = finite & _case_all_finite(value, case_rank)
    player_convex = minimum_own >= -plan.convexity_tolerance
    structural_valid = (
        jnp.all(symmetry <= plan.structural_tolerance, axis=-1)
        & jnp.all(player_convex, axis=-1)
        & (affinity <= plan.structural_tolerance)
    )
    safe_lowered = tuple(_safe(value) for value in lowered)
    safe_pseudogradient = _safe(pseudogradient)
    safe_pseudolinear = _safe(pseudolinear)
    phase_result = _phase_one(
        plan,
        *safe_lowered[:4],
        problem.dynamics_matrices.dtype,
    )
    phase_residual = _constraint_residual_norm(*safe_lowered[:4], phase_result.primal)
    flat_initial = initial_controls.reshape(
        problem.case_shape + (plan.num_control_variables,)
    )
    initial_residual = _constraint_residual_norm(*safe_lowered[:4], flat_initial)
    feasible_initial = jnp.isfinite(initial_residual) & (
        initial_residual <= plan.feasibility_tolerance
    )
    controls = jnp.where(
        feasible_initial[..., None],
        flat_initial,
        jnp.where(phase_result.successful[..., None], phase_result.primal, flat_initial),
    )
    equality_initial = (
        jnp.zeros(problem.case_shape + (plan.num_equalities,), dtype=controls.dtype)
        if initial_equality_multipliers is None
        else initial_equality_multipliers
    )
    inequality_initial = (
        jnp.zeros(problem.case_shape + (plan.num_inequalities,), dtype=controls.dtype)
        if initial_inequality_multipliers is None
        else initial_inequality_multipliers
    )
    equality_variables = (
        equality_initial
        if plan.num_equalities
        else jnp.zeros(problem.case_shape + (1,), dtype=controls.dtype)
    )
    inequality_variables = (
        inequality_initial
        if plan.num_inequalities
        else jnp.zeros(problem.case_shape + (1,), dtype=controls.dtype)
    )
    initial_state = (controls, equality_variables, inequality_variables)
    vi_args = (
        safe_pseudogradient,
        safe_pseudolinear,
        safe_lowered[4],
        safe_lowered[5],
        safe_lowered[0],
        safe_lowered[1],
        safe_lowered[2],
        safe_lowered[3],
    )
    lower = (
        jnp.full_like(controls, -jnp.inf),
        (
            jnp.full_like(equality_variables, -jnp.inf)
            if plan.num_equalities
            else jnp.zeros_like(equality_variables)
        ),
        jnp.zeros_like(inequality_variables),
    )
    upper = (
        jnp.full_like(controls, jnp.inf),
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
        problem_id=f"{plan.problem_id}:open-loop-variational-kkt",
    )
    return (
        state_maps,
        state_offsets,
        player_hessians,
        player_linear,
        player_constants,
        pseudogradient,
        pseudolinear,
        constraint_matrix,
        constraint_offset,
        *lowered,
        symmetry,
        affinity,
        minimum_own,
        minimum_monotonicity,
        finite,
        structural_valid,
        phase_result,
        phase_residual,
        vi_problem,
        initial_state,
        vi_args,
    )


def _initial_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
    dtype: jnp.dtype,
    /,
) -> Array:
    array = _real_array(value, name)
    return _exact_array(array, shape, name, dtype)


def prepare_open_loop_ve(
    plan: OpenLoopVEPlan,
    problem: FiniteHorizonLQOpenLoopVEProblem,
    initial_controls: ArrayLike,
    /,
    *,
    initial_equality_multipliers: ArrayLike | None = None,
    initial_inequality_multipliers: ArrayLike | None = None,
    constraint_args: Any = None,
) -> PreparedOpenLoopVE:
    """Condense, audit phase-I feasibility, and prepare the semismooth box VI."""
    _validate_topology(plan, problem)
    dtype = problem.dynamics_matrices.dtype
    controls = _initial_array(
        initial_controls,
        problem.case_shape + (problem.horizon, problem.control_size),
        "initial_controls",
        dtype,
    )
    equality_initial = (
        None
        if initial_equality_multipliers is None
        else _initial_array(
            initial_equality_multipliers,
            problem.case_shape + (plan.num_equalities,),
            "initial_equality_multipliers",
            dtype,
        )
    )
    inequality_initial = (
        None
        if initial_inequality_multipliers is None
        else _initial_array(
            initial_inequality_multipliers,
            problem.case_shape + (plan.num_inequalities,),
            "initial_inequality_multipliers",
            dtype,
        )
    )
    numeric = _numeric_preparation(
        plan,
        problem,
        controls,
        equality_initial,
        inequality_initial,
        constraint_args,
    )
    vi_prepared = prepare_variational_inequality(
        numeric[-3],
        numeric[-2],
        method=plan.method,
        termination=plan.termination,
        args=numeric[-1],
    )
    prepared_payload = {
        "plan_id": plan.plan_id,
        "dtype": np.dtype(dtype).str,
    }
    prepared_id = f"prepared-open-loop-ve:{canonical_fingerprint(prepared_payload)}"
    return PreparedOpenLoopVE(
        plan,
        problem,
        controls,
        *numeric[:-3],
        numeric[-3],
        vi_prepared,
        constraint_args,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_open_loop_ve(
    prepared: PreparedOpenLoopVE,
    problem: FiniteHorizonLQOpenLoopVEProblem,
    initial_controls: ArrayLike | None = None,
    /,
    *,
    initial_equality_multipliers: ArrayLike | None = None,
    initial_inequality_multipliers: ArrayLike | None = None,
    constraint_args: Any = _UNSET,
) -> PreparedOpenLoopVE:
    """Refresh numeric data while preserving the planned VI and multiplier topology."""
    if not isinstance(prepared, PreparedOpenLoopVE):
        raise TypeError("prepared must be PreparedOpenLoopVE.")
    _validate_topology(prepared.plan, problem)
    dtype = problem.dynamics_matrices.dtype
    controls = (
        prepared.initial_controls
        if initial_controls is None
        else _initial_array(
            initial_controls,
            problem.case_shape + (problem.horizon, problem.control_size),
            "initial_controls",
            dtype,
        )
    )
    equality_initial = (
        None
        if initial_equality_multipliers is None
        else _initial_array(
            initial_equality_multipliers,
            problem.case_shape + (prepared.plan.num_equalities,),
            "initial_equality_multipliers",
            dtype,
        )
    )
    inequality_initial = (
        None
        if initial_inequality_multipliers is None
        else _initial_array(
            initial_inequality_multipliers,
            problem.case_shape + (prepared.plan.num_inequalities,),
            "initial_inequality_multipliers",
            dtype,
        )
    )
    args = prepared.constraint_args if constraint_args is _UNSET else constraint_args
    numeric = _numeric_preparation(
        prepared.plan,
        problem,
        controls,
        equality_initial,
        inequality_initial,
        args,
    )
    vi_prepared = refresh_variational_inequality(
        prepared.vi_prepared,
        numeric[-3],
        numeric[-2],
        args=numeric[-1],
    )
    return PreparedOpenLoopVE(
        prepared.plan,
        problem,
        controls,
        *numeric[:-3],
        numeric[-3],
        vi_prepared,
        args,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def _player_costs(
    problem: FiniteHorizonLQOpenLoopVEProblem,
    states: Array,
    controls: Array,
    /,
) -> Array:
    values = []
    for player in range(problem.num_players):
        stage_states = states[..., :-1, :]
        q = 0.5 * (
            problem.state_costs[..., player, :, :, :]
            + jnp.swapaxes(problem.state_costs[..., player, :, :, :], -1, -2)
        )
        r = 0.5 * (
            problem.control_costs[..., player, :, :, :]
            + jnp.swapaxes(problem.control_costs[..., player, :, :, :], -1, -2)
        )
        stage = 0.5 * ein.contract(
            "...ta,...tab,...tb->...t", stage_states, q, stage_states
        )
        stage = stage + ein.contract(
            "...ta,...tab,...tb->...t",
            stage_states,
            problem.state_control_cross[..., player, :, :, :],
            controls,
        )
        stage = stage + 0.5 * ein.contract(
            "...ta,...tab,...tb->...t", controls, r, controls
        )
        stage = stage + ein.contract(
            "...ta,...ta->...t",
            problem.state_linear[..., player, :, :],
            stage_states,
        )
        stage = stage + ein.contract(
            "...ta,...ta->...t",
            problem.control_linear[..., player, :, :],
            controls,
        )
        stage = stage + problem.stage_constants[..., player, :]
        terminal_state = states[..., -1, :]
        terminal_q = 0.5 * (
            problem.terminal_state_costs[..., player, :, :]
            + jnp.swapaxes(problem.terminal_state_costs[..., player, :, :], -1, -2)
        )
        terminal = 0.5 * ein.contract(
            "...a,...ab,...b->...", terminal_state, terminal_q, terminal_state
        )
        terminal = terminal + ein.contract(
            "...a,...a->...",
            problem.terminal_linear[..., player, :],
            terminal_state,
        )
        terminal = terminal + problem.terminal_constants[..., player]
        values.append(jnp.sum(stage, axis=-1) + terminal)
    return jnp.stack(values, axis=-1)


def _rank(matrix: Array, tolerance: float, /) -> Array:
    if matrix.shape[-2] == 0:
        return jnp.zeros(matrix.shape[:-2], dtype=jnp.int32)
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    maximum = jnp.max(singular, axis=-1, keepdims=True)
    threshold = jnp.asarray(tolerance, dtype=matrix.dtype) * (1.0 + maximum)
    return jnp.sum(singular > threshold, axis=-1, dtype=jnp.int32)


def _projection(
    prepared: PreparedOpenLoopVE,
    controls: Array,
    /,
) -> tuple[ConvexProgramResult, Array]:
    gradient = (
        ein.contract("...ij,...j->...i", prepared.pseudogradient_matrix, controls)
        + prepared.pseudogradient_linear
    )
    trial = controls - prepared.plan.natural_step * gradient
    program = QuadraticProgram(
        jnp.eye(prepared.plan.num_control_variables, dtype=controls.dtype),
        -trial,
        equality_matrix=_safe(prepared.equality_matrix),
        equality_rhs=_safe(prepared.equality_rhs),
        inequality_matrix=_safe(prepared.inequality_matrix),
        inequality_rhs=_safe(prepared.inequality_rhs),
        problem_id=f"{prepared.plan.plan_id}:independent-natural-projection",
        convexity_evidence="identity projection Hessian over original polyhedron",
    )
    result = solve_quadratic_program(program, policy=prepared.plan.projection_policy)
    residual = jnp.linalg.norm(controls - result.primal, axis=-1)
    return result, residual


def solve_prepared_open_loop_ve(
    prepared: PreparedOpenLoopVE,
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> OpenLoopVEResult:
    """Solve and independently certify one prepared convex open-loop VE candidate."""
    if not isinstance(prepared, PreparedOpenLoopVE):
        raise TypeError("prepared must be PreparedOpenLoopVE.")
    if termination is not None and not isinstance(termination, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    vi_result = solve_prepared_variational_inequality(
        prepared.vi_prepared, termination=termination
    )
    flat_controls, equality_variables, inequality_variables = vi_result.state
    equality_multipliers = equality_variables[..., : prepared.plan.num_equalities]
    inequality_multipliers = inequality_variables[..., : prepared.plan.num_inequalities]
    controls = flat_controls.reshape(
        prepared.plan.case_shape + (prepared.plan.horizon, prepared.plan.control_size)
    )
    trajectory = _trajectory(
        prepared.problem,
        prepared.state_maps,
        prepared.state_offsets,
        flat_controls,
    )
    states = trajectory.states
    costs = _player_costs(prepared.problem, states, controls)
    original_raw = _constraint_values(
        prepared.problem,
        prepared.state_maps,
        prepared.state_offsets,
        flat_controls,
        prepared.constraint_args,
    )
    equality_rows = jnp.asarray(prepared.plan.equality_physical_rows, dtype=jnp.int32)
    inequality_rows = jnp.asarray(prepared.plan.inequality_physical_rows, dtype=jnp.int32)
    equality_raw = jnp.take(original_raw, equality_rows, axis=-1)
    inequality_raw = jnp.take(original_raw, inequality_rows, axis=-1)
    stationarity = (
        ein.contract("...ij,...j->...i", prepared.pseudogradient_matrix, flat_controls)
        + prepared.pseudogradient_linear
        + ein.contract(
            "...ij,...j->...i",
            prepared.equality_stationarity,
            equality_multipliers,
        )
        + ein.contract(
            "...ij,...j->...i",
            prepared.inequality_stationarity,
            inequality_multipliers,
        )
    )
    stationarity_residual = _maximum_abs(stationarity)
    equality_residual = _maximum_abs(equality_raw)
    inequality_violation = _maximum_abs(jnp.maximum(inequality_raw, 0.0))
    dual_violation = _maximum_abs(jnp.maximum(-inequality_multipliers, 0.0))
    complementarity_residual = _maximum_abs(inequality_multipliers * inequality_raw)
    original_kkt = jnp.maximum(
        jnp.maximum(stationarity_residual, equality_residual),
        jnp.maximum(
            inequality_violation,
            jnp.maximum(dual_violation, complementarity_residual),
        ),
    )
    projection_result, natural_residual = _projection(prepared, flat_controls)

    combined = jnp.zeros(
        prepared.plan.case_shape + (prepared.plan.multiplier_layout.num_multipliers,),
        dtype=flat_controls.dtype,
    )
    combined = combined.at[
        ..., jnp.asarray(prepared.plan.equality_positions, dtype=jnp.int32)
    ].set(equality_multipliers)
    combined = combined.at[
        ..., jnp.asarray(prepared.plan.inequality_positions, dtype=jnp.int32)
    ].set(inequality_multipliers)
    private = tuple(
        combined[..., start:stop]
        for start, stop in prepared.plan.multiplier_layout.player_slices
    )
    shared_start, shared_stop = prepared.plan.multiplier_layout.shared_slice
    shared = combined[..., shared_start:shared_stop]

    active_tolerance = max(
        prepared.plan.regularity_tolerance, prepared.plan.kkt_tolerance
    )
    active_inequality = jnp.abs(inequality_raw) <= active_tolerance
    masked_inequality = prepared.inequality_matrix * active_inequality[..., :, None]
    active_matrix = jnp.concatenate(
        (prepared.equality_matrix, masked_inequality), axis=-2
    )
    active_rank = _rank(active_matrix, prepared.plan.regularity_tolerance)
    active_count = prepared.plan.num_equalities + jnp.sum(
        active_inequality, axis=-1, dtype=jnp.int32
    )
    regularity = active_rank == active_count
    isolation_matrix = jnp.concatenate(
        (prepared.pseudogradient_matrix, prepared.equality_matrix), axis=-2
    )
    isolation_rank = _rank(isolation_matrix, prepared.plan.regularity_tolerance)
    nonisolation_dimension = prepared.plan.num_control_variables - isolation_rank
    strongly_monotone = (
        prepared.minimum_monotonicity_eigenvalue > prepared.plan.monotonicity_tolerance
    )
    # Strong monotonicity is the only uniqueness/isolation theorem claimed here.
    # The rank is retained as local evidence, not promoted to a global result.
    isolation = strongly_monotone
    # A null direction of both the affine pseudogradient and equalities gives
    # an actual local continuum only when every inequality is strictly inactive.
    # Active one-sided geometry needs more than a rank test, so no claim is made.
    nonuniqueness_evidence = (nonisolation_dimension > 0) & ~jnp.any(
        active_inequality, axis=-1
    )
    player_convex = (
        prepared.minimum_own_control_eigenvalues >= -prepared.plan.convexity_tolerance
    )
    convexity = jnp.all(player_convex, axis=-1) & jnp.all(
        prepared.cost_symmetry_residual <= prepared.plan.structural_tolerance,
        axis=-1,
    )
    monotone = (
        prepared.minimum_monotonicity_eigenvalue >= -prepared.plan.monotonicity_tolerance
    )

    phase_certified_infeasible = (
        prepared.phase_one_result.status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
    ) & prepared.phase_one_result.certificate.dual_ray_valid
    phase_feasible = prepared.phase_one_result.successful & (
        prepared.phase_one_residual <= prepared.plan.feasibility_tolerance
    )
    projection_ok = projection_result.successful
    vi_ok = vi_result.successful & vi_result.certificate.certified
    kkt_ok = (original_kkt <= prepared.plan.kkt_tolerance) & (
        natural_residual <= prepared.plan.natural_residual_tolerance
    )
    case_rank = len(prepared.plan.case_shape)
    output_finite = jnp.ones(prepared.plan.case_shape, dtype=bool)
    for value in (
        flat_controls,
        states,
        costs,
        equality_multipliers,
        inequality_multipliers,
        original_kkt,
        natural_residual,
    ):
        output_finite = output_finite & _case_all_finite(value, case_rank)
    nested_finite = vi_result.certificate.finite
    finite = prepared.input_finite & output_finite & nested_finite

    status = jnp.full(
        prepared.plan.case_shape, int(OpenLoopVEStatus.SUCCESS), dtype=jnp.int32
    )
    status = jnp.where(
        nonuniqueness_evidence,
        int(OpenLoopVEStatus.RESIDUAL_VALID_NONISOLATED),
        status,
    )
    status = jnp.where(~kkt_ok, int(OpenLoopVEStatus.ORIGINAL_KKT_FAILURE), status)
    status = jnp.where(~projection_ok, int(OpenLoopVEStatus.PROJECTION_FAILURE), status)
    status = jnp.where(~vi_ok, int(OpenLoopVEStatus.VI_FAILURE), status)
    status = jnp.where(~output_finite, int(OpenLoopVEStatus.NONFINITE), status)
    status = jnp.where(
        ~phase_feasible,
        int(OpenLoopVEStatus.PHASE_I_FAILURE),
        status,
    )
    status = jnp.where(
        phase_certified_infeasible,
        int(OpenLoopVEStatus.CERTIFIED_INFEASIBILITY),
        status,
    )
    status = jnp.where(
        ~prepared.structural_valid,
        int(OpenLoopVEStatus.STRUCTURAL_INVALIDITY),
        status,
    )
    status = jnp.where(
        ~prepared.input_finite, int(OpenLoopVEStatus.NONFINITE), status
    ).astype(jnp.int32)
    valid = (status == int(OpenLoopVEStatus.SUCCESS)) | (
        status == int(OpenLoopVEStatus.RESIDUAL_VALID_NONISOLATED)
    )
    return OpenLoopVEResult(
        prepared.problem.partition,
        prepared.plan.multiplier_layout,
        prepared.problem.time_grid,
        controls,
        states,
        costs,
        equality_multipliers,
        inequality_multipliers,
        combined,
        private,
        shared,
        stationarity_residual,
        equality_residual,
        inequality_violation,
        dual_violation,
        complementarity_residual,
        original_kkt,
        natural_residual,
        prepared.phase_one_residual,
        phase_certified_infeasible,
        prepared.cost_symmetry_residual,
        prepared.constraint_affinity_residual,
        prepared.structural_valid,
        prepared.minimum_own_control_eigenvalues,
        prepared.minimum_monotonicity_eigenvalue,
        player_convex,
        convexity,
        monotone,
        strongly_monotone,
        active_rank,
        active_count,
        regularity,
        isolation_rank,
        nonisolation_dimension,
        isolation,
        nonuniqueness_evidence,
        finite,
        valid,
        status,
        vi_result,
        prepared.phase_one_result,
        projection_result,
        OPEN_LOOP_VARIATIONAL_GNE,
        _CERTIFICATION_CLAIM,
        prepared.prepared_id,
    )


def solve_open_loop_ve(
    problem: FiniteHorizonLQOpenLoopVEProblem,
    initial_controls: ArrayLike | None = None,
    /,
    *,
    constraint_args: Any = None,
    method: SemismoothNewton | None = None,
    termination: NonlinearTermination | None = None,
    phase_one_policy: ConvexSolvePolicy | None = None,
    projection_policy: ConvexSolvePolicy | None = None,
    structural_tolerance: float = 1.0e-9,
    convexity_tolerance: float = 1.0e-9,
    monotonicity_tolerance: float = 1.0e-9,
    regularity_tolerance: float = 1.0e-9,
    feasibility_tolerance: float = 1.0e-7,
    kkt_tolerance: float = 1.0e-6,
    natural_residual_tolerance: float = 1.0e-6,
    natural_step: float = 1.0,
) -> OpenLoopVEResult:
    """Plan, prepare, solve, and certify a convex open-loop variational GNE."""
    if not isinstance(problem, FiniteHorizonLQOpenLoopVEProblem):
        raise TypeError("problem must be FiniteHorizonLQOpenLoopVEProblem.")
    plan = plan_open_loop_ve(
        problem,
        method=method,
        termination=termination,
        phase_one_policy=phase_one_policy,
        projection_policy=projection_policy,
        structural_tolerance=structural_tolerance,
        convexity_tolerance=convexity_tolerance,
        monotonicity_tolerance=monotonicity_tolerance,
        regularity_tolerance=regularity_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        kkt_tolerance=kkt_tolerance,
        natural_residual_tolerance=natural_residual_tolerance,
        natural_step=natural_step,
    )
    controls = (
        jnp.zeros(
            problem.case_shape + (problem.horizon, problem.control_size),
            dtype=problem.dynamics_matrices.dtype,
        )
        if initial_controls is None
        else initial_controls
    )
    prepared = prepare_open_loop_ve(
        plan, problem, controls, constraint_args=constraint_args
    )
    return solve_prepared_open_loop_ve(prepared)


__all__ = [
    "FiniteHorizonLQOpenLoopVEProblem",
    "OPEN_LOOP_VARIATIONAL_GNE",
    "OpenLoopVEPlan",
    "OpenLoopVEResult",
    "OpenLoopVEStatus",
    "PreparedOpenLoopVE",
    "plan_open_loop_ve",
    "prepare_open_loop_ve",
    "refresh_open_loop_ve",
    "solve_open_loop_ve",
    "solve_prepared_open_loop_ve",
]
