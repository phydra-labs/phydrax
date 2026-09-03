#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-horizon affine-LQ open-loop generalized Nash equilibria."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._bounds import Bounds
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import TimeGrid
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
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
from ._constraints import (
    GameConstraintLayout,
    GameConstraintScope,
    GameMultiplierLayout,
    OpenLoopGameConstraints,
)
from ._layout import PlayerControlPartition
from ._variational import (
    _case_all_finite,
    _condense_costs,
    _condense_dynamics,
    _constraint_linearization,
    _constraint_values,
    _exact_array,
    _identifier,
    _initial_array,
    _maximum_abs,
    _owned_indices,
    _player_costs,
    _positive_tolerance,
    _rank,
    _real_array,
    _safe,
    _trajectory,
)


OPEN_LOOP_GENERALIZED_NASH_KKT = "OPEN_LOOP_GENERALIZED_NASH_KKT"
GLOBAL_CONVEX_GNE_GAP_EVIDENCE = "GLOBAL_CONVEX_GNE_GAP_EVIDENCE"
_KKT_CLAIM = "numerically certified open-loop generalized Nash KKT candidate"
_UNSET = object()


class OpenLoopGNEStatus(IntEnum):
    """Stable case-local outcomes for a generic open-loop GNE solve."""

    SUCCESS = 0
    STRUCTURAL_INVALIDITY = 1
    CERTIFIED_INFEASIBILITY = 2
    VI_FAILURE = 3
    ORIGINAL_KKT_FAILURE = 4
    NONFINITE = 5
    RESIDUAL_VALID_NONISOLATED = 6
    PHASE_I_FAILURE = 7
    BEST_RESPONSE_FAILURE = 8


class FiniteHorizonLQOpenLoopGNEProblem(StrictModule):
    r"""A convex affine-dynamics LQ game with player-specific feasible sets.

    A physically shared constraint is evaluated once, but its KKT multiplier is
    copied once for every declared participant. Player-local and player-owned
    coupled blocks are allocated only to their owner. No common multiplier or
    variational-equilibrium restriction is introduced.
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
        problem_id: str = "control:game:lq-open-loop-gne",
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
        initial_raw = _real_array(initial_state, "initial_state")
        state_cost_raw = _real_array(state_costs, "state_costs")
        control_cost_raw = _real_array(control_costs, "control_costs")
        terminal_cost_raw = _real_array(terminal_state_costs, "terminal_state_costs")
        dtype = jnp.result_type(
            a.dtype,
            b.dtype,
            initial_raw.dtype,
            state_cost_raw.dtype,
            control_cost_raw.dtype,
            terminal_cost_raw.dtype,
            jnp.float32,
        )
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
            initial_raw, case_shape + (state_size,), "initial_state", dtype
        )
        q = _exact_array(
            state_cost_raw,
            case_shape + (players, horizon, state_size, state_size),
            "state_costs",
            dtype,
        )
        r = _exact_array(
            control_cost_raw,
            case_shape + (players, horizon, control_size, control_size),
            "control_costs",
            dtype,
        )
        q_terminal = _exact_array(
            terminal_cost_raw,
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


class OpenLoopGNEPlan(StrictModule):
    """Fixed generic-GNE multiplier topology and numerical policies."""

    partition: PlayerControlPartition
    constraint_layout: GameConstraintLayout
    multiplier_layout: GameMultiplierLayout
    method: SemismoothNewton
    termination: NonlinearTermination
    phase_one_policy: ConvexSolvePolicy
    best_response_policy: ConvexSolvePolicy
    owned_control_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    multiplier_physical_rows: tuple[int, ...] = eqx.field(static=True)
    multiplier_owner_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    equality_positions: tuple[int, ...] = eqx.field(static=True)
    inequality_positions: tuple[int, ...] = eqx.field(static=True)
    equality_multiplier_physical_rows: tuple[int, ...] = eqx.field(static=True)
    inequality_multiplier_physical_rows: tuple[int, ...] = eqx.field(static=True)
    physical_equality_rows: tuple[int, ...] = eqx.field(static=True)
    physical_inequality_rows: tuple[int, ...] = eqx.field(static=True)
    shared_physical_rows: tuple[int, ...] = eqx.field(static=True)
    player_equality_physical_rows: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    player_inequality_physical_rows: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    player_shared_multiplier_positions: tuple[tuple[int, ...], ...] = eqx.field(
        static=True
    )
    case_shape: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_control_variables: int = eqx.field(static=True)
    num_equalities: int = eqx.field(static=True)
    num_inequalities: int = eqx.field(static=True)
    num_physical_equalities: int = eqx.field(static=True)
    num_physical_inequalities: int = eqx.field(static=True)
    audit_best_responses: bool = eqx.field(static=True)
    structural_tolerance: float = eqx.field(static=True)
    convexity_tolerance: float = eqx.field(static=True)
    regularity_tolerance: float = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    kkt_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedOpenLoopGNE(StrictModule):
    """Condensed player KKT system, phase-I audit, and prepared box VI."""

    plan: OpenLoopGNEPlan
    problem: FiniteHorizonLQOpenLoopGNEProblem
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
    physical_equality_matrix: Array
    physical_equality_rhs: Array
    physical_inequality_matrix: Array
    physical_inequality_rhs: Array
    equality_matrix: Array
    equality_rhs: Array
    inequality_matrix: Array
    inequality_rhs: Array
    equality_stationarity: Array
    inequality_stationarity: Array
    cost_symmetry_residual: Array
    constraint_affinity_residual: Array
    minimum_own_control_eigenvalues: Array
    input_finite: Array
    structural_valid: Array
    phase_one_result: ConvexProgramResult
    phase_one_residual: Array
    vi_problem: VariationalInequalityProblem
    vi_prepared: PreparedVariationalInequalitySolve
    constraint_args: Any
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class OpenLoopGNEResult(StrictModule):
    """Candidate with physical feasibility and player-specific KKT evidence."""

    partition: PlayerControlPartition
    multiplier_layout: GameMultiplierLayout
    time_grid: TimeGrid
    controls: Array
    states: Array
    player_costs: Array
    physical_constraint_residuals: Array
    physical_shared_residuals: Array
    equality_multipliers: Array
    inequality_multipliers: Array
    multipliers: Array
    player_multipliers: tuple[Array, ...]
    player_shared_multiplier_copies: tuple[Array, ...]
    player_stationarity_residuals: Array
    player_dual_violations: Array
    player_complementarity_residuals: Array
    original_stationarity_residual: Array
    original_equality_residual: Array
    original_inequality_violation: Array
    original_dual_violation: Array
    original_complementarity_residual: Array
    original_kkt_residual: Array
    phase_one_residual: Array
    phase_one_certified_infeasible: Array
    cost_symmetry_residual: Array
    constraint_affinity_residual: Array
    structural_valid: Array
    minimum_own_control_eigenvalues: Array
    player_convex: Array
    convexity_certified: Array
    player_active_constraint_rank: Array
    player_active_constraint_count: Array
    player_constraint_qualification: Array
    strict_complementarity: Array
    branch_jacobian_rank: Array
    branch_dimension: Array
    branch_regular: Array
    branch_isolated: Array
    regularity_certified: Array
    nonuniqueness_evidence: Array
    best_response_values: Array
    player_best_response_gaps: Array
    best_response_numerical_errors: Array
    player_gap_upper_bounds: Array
    best_response_successful: Array
    best_response_audit_complete: Array
    global_gne_gap_bound: Array
    global_gap_evidence_available: Array
    original_kkt_valid: Array
    finite: Array
    valid: Array
    status: Array
    vi_result: VariationalInequalityResult
    phase_one_result: ConvexProgramResult
    best_response_results: tuple[ConvexProgramResult, ...]
    certificate_label: str = eqx.field(static=True)
    certification_claim: str = eqx.field(static=True)
    global_gap_certificate_label: str = eqx.field(static=True)
    common_multiplier_imposed: bool = eqx.field(static=True)
    variational_equilibrium_claimed: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


def _physical_metadata(
    layout: GameConstraintLayout, /
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    equalities: list[int] = []
    inequalities: list[int] = []
    shared: list[int] = []
    for block, (start, stop) in zip(
        layout.constraints.blocks, layout.block_slices, strict=True
    ):
        destination = equalities if block.equality else inequalities
        destination.extend(range(start, stop))
        if block.scope is GameConstraintScope.SHARED:
            shared.extend(range(start, stop))
    return tuple(equalities), tuple(inequalities), tuple(shared)


def _gne_multiplier_metadata(
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
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    if layout.variational:
        raise ValueError("Generic GNE planning requires private multiplier copies.")
    constraint_layout = layout.constraint_layout
    blocks = constraint_layout.constraints.blocks
    rows: list[int] = []
    owners: list[tuple[int, ...]] = []
    equalities: list[bool] = []
    player_equalities: list[tuple[int, ...]] = []
    player_inequalities: list[tuple[int, ...]] = []
    player_shared_positions: list[tuple[int, ...]] = []
    cursor = 0
    for player, block_indices in enumerate(layout.player_block_indices):
        equality_rows: list[int] = []
        inequality_rows: list[int] = []
        shared_positions: list[int] = []
        for block_index in block_indices:
            block = blocks[block_index]
            start, stop = constraint_layout.block_slices[block_index]
            block_rows = tuple(range(start, stop))
            if block.equality:
                equality_rows.extend(block_rows)
            else:
                inequality_rows.extend(block_rows)
            for physical_row in block_rows:
                rows.append(physical_row)
                owners.append(owned[player])
                equalities.append(block.equality)
                if block.scope is GameConstraintScope.SHARED:
                    shared_positions.append(cursor)
                cursor += 1
        player_equalities.append(tuple(equality_rows))
        player_inequalities.append(tuple(inequality_rows))
        player_shared_positions.append(tuple(shared_positions))
    if len(rows) != layout.num_multipliers or cursor != layout.num_multipliers:
        raise RuntimeError("GNE multiplier metadata does not match its fixed layout.")
    equality_positions = tuple(i for i, value in enumerate(equalities) if value)
    inequality_positions = tuple(i for i, value in enumerate(equalities) if not value)
    equality_rows = tuple(rows[i] for i in equality_positions)
    inequality_rows = tuple(rows[i] for i in inequality_positions)
    return (
        tuple(rows),
        tuple(owners),
        equality_positions,
        inequality_positions,
        equality_rows,
        inequality_rows,
        tuple(player_equalities),
        tuple(player_inequalities),
        tuple(player_shared_positions),
    )


def plan_open_loop_gne(
    problem: FiniteHorizonLQOpenLoopGNEProblem,
    /,
    *,
    method: SemismoothNewton | None = None,
    termination: NonlinearTermination | None = None,
    phase_one_policy: ConvexSolvePolicy | None = None,
    audit_best_responses: bool = False,
    best_response_policy: ConvexSolvePolicy | None = None,
    structural_tolerance: float = 1.0e-9,
    convexity_tolerance: float = 1.0e-9,
    regularity_tolerance: float = 1.0e-9,
    feasibility_tolerance: float = 1.0e-7,
    kkt_tolerance: float = 1.0e-6,
) -> OpenLoopGNEPlan:
    """Plan a generic GNE KKT solve with one shared multiplier per player."""
    if not isinstance(problem, FiniteHorizonLQOpenLoopGNEProblem):
        raise TypeError("problem must be FiniteHorizonLQOpenLoopGNEProblem.")
    if not isinstance(audit_best_responses, bool):
        raise TypeError("audit_best_responses must be a bool.")
    tolerances = (
        _positive_tolerance(structural_tolerance, "structural_tolerance"),
        _positive_tolerance(convexity_tolerance, "convexity_tolerance"),
        _positive_tolerance(regularity_tolerance, "regularity_tolerance"),
        _positive_tolerance(feasibility_tolerance, "feasibility_tolerance"),
        _positive_tolerance(kkt_tolerance, "kkt_tolerance"),
    )
    method_ = (
        SemismoothNewton(
            feasibility="preserve-box", certification_tolerance=tolerances[4]
        )
        if method is None
        else method
    )
    termination_ = NonlinearTermination() if termination is None else termination
    default_qp_policy = ConvexSolvePolicy(
        termination=ConvexTermination(absolute=1.0e-9, maximum_steps=200)
    )
    phase_policy = default_qp_policy if phase_one_policy is None else phase_one_policy
    response_policy = (
        default_qp_policy if best_response_policy is None else best_response_policy
    )
    if not isinstance(method_, SemismoothNewton):
        raise TypeError("method must be SemismoothNewton or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    if not isinstance(phase_policy, ConvexSolvePolicy):
        raise TypeError("phase_one_policy must be ConvexSolvePolicy or None.")
    if not isinstance(response_policy, ConvexSolvePolicy):
        raise TypeError("best_response_policy must be ConvexSolvePolicy or None.")
    constraint_layout = problem.constraints.layout(num_path_sites=problem.horizon)
    multiplier_layout = constraint_layout.multiplier_layout(variational=False)
    if (
        multiplier_layout.variational
        or multiplier_layout.shared_slice[0] != multiplier_layout.shared_slice[1]
    ):
        raise RuntimeError("Generic GNE planning must not allocate common multipliers.")
    owned = _owned_indices(problem.partition, problem.horizon)
    multiplier = _gne_multiplier_metadata(multiplier_layout, owned)
    physical = _physical_metadata(constraint_layout)
    num_variables = problem.horizon * problem.control_size
    payload = {
        "kind": "finite-horizon-lq-open-loop-generalized-nash",
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
        "best_response_policy": response_policy.policy_id,
        "audit_best_responses": audit_best_responses,
        "tolerances": tolerances,
    }
    return OpenLoopGNEPlan(
        problem.partition,
        constraint_layout,
        multiplier_layout,
        method_,
        termination_,
        phase_policy,
        response_policy,
        owned,
        *multiplier[:6],
        *physical,
        *multiplier[6:],
        problem.case_shape,
        problem.horizon,
        problem.state_size,
        problem.control_size,
        num_variables,
        len(multiplier[2]),
        len(multiplier[3]),
        len(physical[0]),
        len(physical[1]),
        audit_best_responses,
        *tolerances,
        problem.problem_id,
        problem.dynamics_id,
        problem.constraints.constraints_id,
        problem.time_grid.time_id,
        f"open-loop-gne-plan:{canonical_fingerprint(payload)}",
    )


def _validate_topology(
    plan: OpenLoopGNEPlan,
    problem: FiniteHorizonLQOpenLoopGNEProblem,
    /,
) -> None:
    if not isinstance(plan, OpenLoopGNEPlan):
        raise TypeError("plan must be OpenLoopGNEPlan.")
    if not isinstance(problem, FiniteHorizonLQOpenLoopGNEProblem):
        raise TypeError("problem must be FiniteHorizonLQOpenLoopGNEProblem.")
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
            "Open-loop GNE plan and problem topology identities do not match."
        )


def _lower_constraints(
    plan: OpenLoopGNEPlan,
    matrix: Array,
    offset: Array,
    /,
) -> tuple[Array, ...]:
    dtype = matrix.dtype
    combined_stationarity = jnp.zeros(
        plan.case_shape
        + (plan.num_control_variables, plan.multiplier_layout.num_multipliers),
        dtype=dtype,
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

    def lower(rows: tuple[int, ...]) -> tuple[Array, Array]:
        indices = jnp.asarray(rows, dtype=jnp.int32)
        return jnp.take(matrix, indices, axis=-2), -jnp.take(offset, indices, axis=-1)

    physical_equality = lower(plan.physical_equality_rows)
    physical_inequality = lower(plan.physical_inequality_rows)
    equality = lower(plan.equality_multiplier_physical_rows)
    inequality = lower(plan.inequality_multiplier_physical_rows)
    equality_stationarity = jnp.take(
        combined_stationarity,
        jnp.asarray(plan.equality_positions, dtype=jnp.int32),
        axis=-1,
    )
    inequality_stationarity = jnp.take(
        combined_stationarity,
        jnp.asarray(plan.inequality_positions, dtype=jnp.int32),
        axis=-1,
    )
    return (
        *physical_equality,
        *physical_inequality,
        *equality,
        *inequality,
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
    plan: OpenLoopGNEPlan,
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
        problem_id=f"{plan.plan_id}:physical-phase-i-feasibility",
        convexity_evidence="identity objective over each physical constraint once",
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


def _numeric_preparation(
    plan: OpenLoopGNEPlan,
    problem: FiniteHorizonLQOpenLoopGNEProblem,
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
        _,
    ) = _condense_costs(problem, state_maps, state_offsets)
    constraint_matrix, constraint_offset, affinity = _constraint_linearization(
        problem, state_maps, state_offsets, constraint_args
    )
    lowered = _lower_constraints(plan, constraint_matrix, constraint_offset)
    physical = lowered[:4]
    multiplier = lowered[4:]
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
    safe_physical = tuple(_safe(value) for value in physical)
    safe_multiplier = tuple(_safe(value) for value in multiplier)
    safe_pseudogradient = _safe(pseudogradient)
    safe_pseudolinear = _safe(pseudolinear)
    phase_result = _phase_one(plan, *safe_physical, problem.dynamics_matrices.dtype)
    phase_residual = _constraint_residual_norm(*safe_physical, phase_result.primal)
    flat_initial = initial_controls.reshape(
        problem.case_shape + (plan.num_control_variables,)
    )
    initial_residual = _constraint_residual_norm(*safe_physical, flat_initial)
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
    vi_args = (
        safe_pseudogradient,
        safe_pseudolinear,
        safe_multiplier[4],
        safe_multiplier[5],
        safe_multiplier[0],
        safe_multiplier[1],
        safe_multiplier[2],
        safe_multiplier[3],
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
        problem_id=f"{plan.problem_id}:open-loop-generalized-nash-kkt",
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
        finite,
        structural_valid,
        phase_result,
        phase_residual,
        vi_problem,
        (controls, equality_variables, inequality_variables),
        vi_args,
    )


def prepare_open_loop_gne(
    plan: OpenLoopGNEPlan,
    problem: FiniteHorizonLQOpenLoopGNEProblem,
    initial_controls: ArrayLike,
    /,
    *,
    initial_equality_multipliers: ArrayLike | None = None,
    initial_inequality_multipliers: ArrayLike | None = None,
    constraint_args: Any = None,
) -> PreparedOpenLoopGNE:
    """Condense and prepare the player-specific KKT complementarity system."""
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
    prepared_id = "prepared-open-loop-gne:" + canonical_fingerprint(
        {"plan_id": plan.plan_id, "dtype": np.dtype(dtype).str}
    )
    return PreparedOpenLoopGNE(
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


def refresh_open_loop_gne(
    prepared: PreparedOpenLoopGNE,
    problem: FiniteHorizonLQOpenLoopGNEProblem,
    initial_controls: ArrayLike | None = None,
    /,
    *,
    initial_equality_multipliers: ArrayLike | None = None,
    initial_inequality_multipliers: ArrayLike | None = None,
    constraint_args: Any = _UNSET,
) -> PreparedOpenLoopGNE:
    """Refresh numeric data without changing player or multiplier topology."""
    if not isinstance(prepared, PreparedOpenLoopGNE):
        raise TypeError("prepared must be PreparedOpenLoopGNE.")
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
        prepared.vi_prepared, numeric[-3], numeric[-2], args=numeric[-1]
    )
    return PreparedOpenLoopGNE(
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


def _player_constraint_evidence(
    prepared: PreparedOpenLoopGNE,
    physical_raw: Array,
    /,
) -> tuple[Array, Array, Array]:
    ranks = []
    counts = []
    qualifications = []
    tolerance = max(prepared.plan.regularity_tolerance, prepared.plan.kkt_tolerance)
    for player, owned in enumerate(prepared.plan.owned_control_indices):
        owned_indices = jnp.asarray(owned, dtype=jnp.int32)
        equality_rows = prepared.plan.player_equality_physical_rows[player]
        inequality_rows = prepared.plan.player_inequality_physical_rows[player]
        equality_matrix = jnp.take(
            prepared.constraint_matrix,
            jnp.asarray(equality_rows, dtype=jnp.int32),
            axis=-2,
        )
        equality_matrix = jnp.take(equality_matrix, owned_indices, axis=-1)
        inequality_matrix = jnp.take(
            prepared.constraint_matrix,
            jnp.asarray(inequality_rows, dtype=jnp.int32),
            axis=-2,
        )
        inequality_matrix = jnp.take(inequality_matrix, owned_indices, axis=-1)
        inequality_raw = jnp.take(
            physical_raw,
            jnp.asarray(inequality_rows, dtype=jnp.int32),
            axis=-1,
        )
        active = jnp.abs(inequality_raw) <= tolerance
        active_matrix = jnp.concatenate(
            (equality_matrix, inequality_matrix * active[..., :, None]), axis=-2
        )
        rank = _rank(active_matrix, prepared.plan.regularity_tolerance)
        count = len(equality_rows) + jnp.sum(active, axis=-1, dtype=jnp.int32)
        ranks.append(rank)
        counts.append(count)
        qualifications.append(rank == count)
    return (
        jnp.stack(ranks, axis=-1),
        jnp.stack(counts, axis=-1),
        jnp.stack(qualifications, axis=-1),
    )


def _branch_evidence(
    prepared: PreparedOpenLoopGNE,
    inequality_raw: Array,
    inequality_multipliers: Array,
    player_cq: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    plan = prepared.plan
    tolerance = max(plan.regularity_tolerance, plan.kkt_tolerance)
    active = jnp.abs(inequality_raw) <= tolerance
    strict_active = jnp.all((~active) | (inequality_multipliers > tolerance), axis=-1)
    strict_inactive = jnp.all(active | (inequality_raw < -tolerance), axis=-1)
    strict = strict_active & strict_inactive
    zeros_top = jnp.zeros(
        plan.case_shape + (plan.num_control_variables, 0),
        dtype=prepared.pseudogradient_matrix.dtype,
    )
    equality_stationarity = (
        prepared.equality_stationarity if plan.num_equalities else zeros_top
    )
    inequality_stationarity = (
        prepared.inequality_stationarity * active[..., None, :]
        if plan.num_inequalities
        else zeros_top
    )
    top = jnp.concatenate(
        (
            prepared.pseudogradient_matrix,
            equality_stationarity,
            inequality_stationarity,
        ),
        axis=-1,
    )
    multiplier_width = plan.num_equalities + plan.num_inequalities
    middle = jnp.concatenate(
        (
            prepared.equality_matrix,
            jnp.zeros(
                plan.case_shape + (plan.num_equalities, multiplier_width),
                dtype=top.dtype,
            ),
        ),
        axis=-1,
    )
    inactive_diagonal = (
        jnp.eye(plan.num_inequalities, dtype=top.dtype) * (~active)[..., None, :]
    )
    bottom = jnp.concatenate(
        (
            prepared.inequality_matrix * active[..., :, None],
            jnp.zeros(
                plan.case_shape + (plan.num_inequalities, plan.num_equalities),
                dtype=top.dtype,
            ),
            inactive_diagonal,
        ),
        axis=-1,
    )
    branch_matrix = jnp.concatenate((top, middle, bottom), axis=-2)
    rank = _rank(branch_matrix, plan.regularity_tolerance)
    size = plan.num_control_variables + multiplier_width
    dimension = size - rank
    regular = rank == size
    cq = jnp.all(player_cq, axis=-1)
    isolated = regular & cq & strict
    regularity = regular & cq & strict
    nonunique = (dimension > 0) & cq & strict
    return strict, rank, dimension, regular, isolated, regularity, nonunique


def _quadratic_value(
    hessian: Array, linear: Array, constant: Array, point: Array, /
) -> Array:
    return (
        0.5 * ein.contract("...i,...ij,...j->...", point, hessian, point)
        + ein.contract("...i,...i->...", linear, point)
        + constant
    )


def _best_response_audit(
    prepared: PreparedOpenLoopGNE,
    controls: Array,
    profile_costs: Array,
    /,
) -> tuple[tuple[ConvexProgramResult, ...], Array, Array, Array, Array]:
    results: list[ConvexProgramResult] = []
    values = []
    gaps = []
    errors = []
    successful = []
    for player, owned in enumerate(prepared.plan.owned_control_indices):
        indices = jnp.asarray(owned, dtype=jnp.int32)
        hessian = prepared.player_hessians[..., player, :, :]
        linear = prepared.player_linear[..., player, :]
        own_hessian = jnp.take(jnp.take(hessian, indices, axis=-2), indices, axis=-1)
        current_own = jnp.take(controls, indices, axis=-1)
        full_gradient = ein.contract("...ij,...j->...i", hessian, controls) + linear
        own_linear = jnp.take(full_gradient, indices, axis=-1) - ein.contract(
            "...ij,...j->...i", own_hessian, current_own
        )

        equality_rows = prepared.plan.player_equality_physical_rows[player]
        inequality_rows = prepared.plan.player_inequality_physical_rows[player]

        def unilateral_constraints(rows: tuple[int, ...]) -> tuple[Array, Array]:
            row_indices = jnp.asarray(rows, dtype=jnp.int32)
            complete_matrix = jnp.take(prepared.constraint_matrix, row_indices, axis=-2)
            own_matrix = jnp.take(complete_matrix, indices, axis=-1)
            offset = jnp.take(prepared.constraint_offset, row_indices, axis=-1)
            fixed = (
                ein.contract("...ij,...j->...i", complete_matrix, controls)
                + offset
                - ein.contract("...ij,...j->...i", own_matrix, current_own)
            )
            return own_matrix, -fixed

        equality_matrix, equality_rhs = unilateral_constraints(equality_rows)
        inequality_matrix, inequality_rhs = unilateral_constraints(inequality_rows)
        program = QuadraticProgram(
            own_hessian,
            own_linear,
            equality_matrix=equality_matrix,
            equality_rhs=equality_rhs,
            inequality_matrix=inequality_matrix,
            inequality_rhs=inequality_rhs,
            problem_id=f"{prepared.plan.plan_id}:player-{player}:best-response",
            convexity_evidence=(
                "audited condensed own-control Hessian and unilateral polyhedron"
            ),
        )
        result = solve_quadratic_program(
            program, policy=prepared.plan.best_response_policy
        )
        response = controls.at[..., indices].set(result.primal)
        value = _quadratic_value(
            hessian,
            linear,
            prepared.player_constants[..., player],
            response,
        )
        gap = profile_costs[..., player] - value
        # A nonnegative inequality dual and positive-definite own Hessian give
        # an explicit global dual lower bound. Thus ``value - lower_bound`` is
        # an objective-unit numerical allowance, not a stationarity norm passed
        # off as a gap. Semidefinite cases retain the audited minimizer estimate
        # but cannot publish a finite bound without additional recession data.
        nonnegative_dual = jnp.maximum(result.inequality_dual, 0.0)
        dual_gradient = (
            ein.contract("...ij,...j->...i", own_hessian, result.primal)
            + own_linear
            + ein.contract("...ji,...j->...i", equality_matrix, result.equality_dual)
            + ein.contract("...ji,...j->...i", inequality_matrix, nonnegative_dual)
        )
        lagrangian = (
            result.objective
            + ein.contract(
                "...i,...i->...",
                result.equality_dual,
                ein.contract("...ij,...j->...i", equality_matrix, result.primal)
                - equality_rhs,
            )
            + ein.contract(
                "...i,...i->...",
                nonnegative_dual,
                ein.contract("...ij,...j->...i", inequality_matrix, result.primal)
                - inequality_rhs,
            )
        )
        minimum_curvature = jnp.min(jnp.linalg.eigvalsh(own_hessian), axis=-1)
        curvature_certified = minimum_curvature > prepared.plan.convexity_tolerance
        identity = jnp.eye(own_hessian.shape[-1], dtype=own_hessian.dtype)
        safe_hessian = jnp.where(
            curvature_certified[..., None, None], own_hessian, identity
        )
        correction_result = solve(
            LinearSystem(
                DenseLinearOperator(
                    safe_hessian,
                    operator_id=(
                        f"{prepared.plan.plan_id}:player-{player}:dual-error-hessian"
                    ),
                ),
                problem_id=f"{prepared.plan.plan_id}:player-{player}:dual-error",
            ),
            dual_gradient,
            policy=LinearSolvePolicy(
                DenseLU(),
                failure=FailurePolicy("status"),
            ),
        )
        correction_direction = correction_result.value
        dual_lower_bound = lagrangian - 0.5 * ein.contract(
            "...i,...i->...", dual_gradient, correction_direction
        )
        finite_error = jnp.maximum(result.objective - dual_lower_bound, 0.0)
        error = jnp.where(
            curvature_certified & result.successful & correction_result.successful,
            finite_error,
            jnp.asarray(jnp.inf, dtype=finite_error.dtype),
        )
        results.append(result)
        values.append(value)
        gaps.append(gap)
        errors.append(error)
        successful.append(result.successful)
    return (
        tuple(results),
        jnp.stack(values, axis=-1),
        jnp.stack(gaps, axis=-1),
        jnp.stack(errors, axis=-1),
        jnp.stack(successful, axis=-1),
    )


def solve_prepared_open_loop_gne(
    prepared: PreparedOpenLoopGNE,
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> OpenLoopGNEResult:
    """Solve and certify player-specific open-loop GNE KKT conditions."""
    if not isinstance(prepared, PreparedOpenLoopGNE):
        raise TypeError("prepared must be PreparedOpenLoopGNE.")
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

    # This is the sole original-scale callback evaluation used by certification.
    physical_raw = _constraint_values(
        prepared.problem,
        prepared.state_maps,
        prepared.state_offsets,
        flat_controls,
        prepared.constraint_args,
    )
    equality_raw = jnp.take(
        physical_raw,
        jnp.asarray(prepared.plan.physical_equality_rows, dtype=jnp.int32),
        axis=-1,
    )
    physical_inequality_raw = jnp.take(
        physical_raw,
        jnp.asarray(prepared.plan.physical_inequality_rows, dtype=jnp.int32),
        axis=-1,
    )
    multiplier_inequality_raw = jnp.take(
        physical_raw,
        jnp.asarray(prepared.plan.inequality_multiplier_physical_rows, dtype=jnp.int32),
        axis=-1,
    )
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
    inequality_violation = _maximum_abs(jnp.maximum(physical_inequality_raw, 0.0))
    dual_violation = _maximum_abs(jnp.maximum(-inequality_multipliers, 0.0))
    complementarity_residual = _maximum_abs(
        inequality_multipliers * multiplier_inequality_raw
    )
    original_kkt = jnp.maximum(
        jnp.maximum(stationarity_residual, equality_residual),
        jnp.maximum(
            inequality_violation,
            jnp.maximum(dual_violation, complementarity_residual),
        ),
    )

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
    player_multipliers = tuple(
        combined[..., start:stop]
        for start, stop in prepared.plan.multiplier_layout.player_slices
    )
    player_shared = tuple(
        jnp.take(combined, jnp.asarray(positions, dtype=jnp.int32), axis=-1)
        for positions in prepared.plan.player_shared_multiplier_positions
    )
    player_stationarity_residuals = jnp.stack(
        tuple(
            _maximum_abs(
                jnp.take(
                    stationarity,
                    jnp.asarray(indices, dtype=jnp.int32),
                    axis=-1,
                )
            )
            for indices in prepared.plan.owned_control_indices
        ),
        axis=-1,
    )
    inequality_position_set = set(prepared.plan.inequality_positions)
    player_dual_violations = []
    player_complementarity_residuals = []
    for start, stop in prepared.plan.multiplier_layout.player_slices:
        positions = tuple(
            position
            for position in range(start, stop)
            if position in inequality_position_set
        )
        indices = jnp.asarray(positions, dtype=jnp.int32)
        player_duals = jnp.take(combined, indices, axis=-1)
        player_raw = jnp.take(
            physical_raw,
            jnp.asarray(
                tuple(
                    prepared.plan.multiplier_physical_rows[position]
                    for position in positions
                ),
                dtype=jnp.int32,
            ),
            axis=-1,
        )
        player_dual_violations.append(_maximum_abs(jnp.maximum(-player_duals, 0.0)))
        player_complementarity_residuals.append(_maximum_abs(player_duals * player_raw))
    player_dual_violations = jnp.stack(player_dual_violations, axis=-1)
    player_complementarity_residuals = jnp.stack(
        player_complementarity_residuals, axis=-1
    )
    shared_raw = jnp.take(
        physical_raw,
        jnp.asarray(prepared.plan.shared_physical_rows, dtype=jnp.int32),
        axis=-1,
    )

    cq_rank, cq_count, player_cq = _player_constraint_evidence(prepared, physical_raw)
    branch = _branch_evidence(
        prepared,
        multiplier_inequality_raw,
        inequality_multipliers,
        player_cq,
    )
    (
        strict_complementarity,
        branch_rank,
        branch_dimension,
        branch_regular,
        branch_isolated,
        regularity,
        nonuniqueness,
    ) = branch
    player_convex = (
        prepared.minimum_own_control_eigenvalues >= -prepared.plan.convexity_tolerance
    )
    convexity = jnp.all(player_convex, axis=-1) & jnp.all(
        prepared.cost_symmetry_residual <= prepared.plan.structural_tolerance,
        axis=-1,
    )

    if prepared.plan.audit_best_responses:
        (
            response_results,
            response_values,
            response_gaps,
            response_errors,
            response_successful,
        ) = _best_response_audit(prepared, flat_controls, costs)
        response_complete = jnp.all(response_successful, axis=-1) & jnp.all(
            jnp.isfinite(response_values)
            & jnp.isfinite(response_gaps)
            & jnp.isfinite(response_errors),
            axis=-1,
        )
    else:
        response_results = ()
        response_values = jnp.full_like(costs, jnp.nan)
        response_gaps = jnp.full_like(costs, jnp.nan)
        response_errors = jnp.full_like(costs, jnp.nan)
        response_successful = jnp.zeros_like(costs, dtype=bool)
        response_complete = jnp.zeros(prepared.plan.case_shape, dtype=bool)
    gap_upper = jnp.maximum(response_gaps, 0.0) + response_errors
    global_gap = jnp.max(gap_upper, axis=-1)

    phase_certified_infeasible = (
        prepared.phase_one_result.status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
    ) & prepared.phase_one_result.certificate.dual_ray_valid
    phase_feasible = prepared.phase_one_result.successful & (
        prepared.phase_one_residual <= prepared.plan.feasibility_tolerance
    )
    vi_ok = vi_result.successful & vi_result.certificate.certified
    kkt_ok = original_kkt <= prepared.plan.kkt_tolerance
    case_rank = len(prepared.plan.case_shape)
    output_finite = jnp.ones(prepared.plan.case_shape, dtype=bool)
    for value in (
        flat_controls,
        states,
        costs,
        equality_multipliers,
        inequality_multipliers,
        physical_raw,
        original_kkt,
    ):
        output_finite = output_finite & _case_all_finite(value, case_rank)
    finite = prepared.input_finite & output_finite & vi_result.certificate.finite
    original_kkt_valid = (
        prepared.structural_valid & phase_feasible & vi_ok & kkt_ok & finite
    )
    global_gap_available = (
        response_complete & original_kkt_valid & convexity & jnp.isfinite(global_gap)
    )

    status = jnp.full(
        prepared.plan.case_shape, int(OpenLoopGNEStatus.SUCCESS), dtype=jnp.int32
    )
    status = jnp.where(
        nonuniqueness,
        int(OpenLoopGNEStatus.RESIDUAL_VALID_NONISOLATED),
        status,
    )
    if prepared.plan.audit_best_responses:
        status = jnp.where(
            ~response_complete,
            int(OpenLoopGNEStatus.BEST_RESPONSE_FAILURE),
            status,
        )
    status = jnp.where(~kkt_ok, int(OpenLoopGNEStatus.ORIGINAL_KKT_FAILURE), status)
    status = jnp.where(~vi_ok, int(OpenLoopGNEStatus.VI_FAILURE), status)
    status = jnp.where(~output_finite, int(OpenLoopGNEStatus.NONFINITE), status)
    status = jnp.where(~phase_feasible, int(OpenLoopGNEStatus.PHASE_I_FAILURE), status)
    status = jnp.where(
        phase_certified_infeasible,
        int(OpenLoopGNEStatus.CERTIFIED_INFEASIBILITY),
        status,
    )
    status = jnp.where(
        ~prepared.structural_valid,
        int(OpenLoopGNEStatus.STRUCTURAL_INVALIDITY),
        status,
    )
    status = jnp.where(
        ~prepared.input_finite, int(OpenLoopGNEStatus.NONFINITE), status
    ).astype(jnp.int32)
    valid = (status == int(OpenLoopGNEStatus.SUCCESS)) | (
        status == int(OpenLoopGNEStatus.RESIDUAL_VALID_NONISOLATED)
    )
    return OpenLoopGNEResult(
        prepared.problem.partition,
        prepared.plan.multiplier_layout,
        prepared.problem.time_grid,
        controls,
        states,
        costs,
        physical_raw,
        shared_raw,
        equality_multipliers,
        inequality_multipliers,
        combined,
        player_multipliers,
        player_shared,
        player_stationarity_residuals,
        player_dual_violations,
        player_complementarity_residuals,
        stationarity_residual,
        equality_residual,
        inequality_violation,
        dual_violation,
        complementarity_residual,
        original_kkt,
        prepared.phase_one_residual,
        phase_certified_infeasible,
        prepared.cost_symmetry_residual,
        prepared.constraint_affinity_residual,
        prepared.structural_valid,
        prepared.minimum_own_control_eigenvalues,
        player_convex,
        convexity,
        cq_rank,
        cq_count,
        player_cq,
        strict_complementarity,
        branch_rank,
        branch_dimension,
        branch_regular,
        branch_isolated,
        regularity,
        nonuniqueness,
        response_values,
        response_gaps,
        response_errors,
        gap_upper,
        response_successful,
        response_complete,
        global_gap,
        global_gap_available,
        original_kkt_valid,
        finite,
        valid,
        status,
        vi_result,
        prepared.phase_one_result,
        response_results,
        OPEN_LOOP_GENERALIZED_NASH_KKT,
        _KKT_CLAIM,
        GLOBAL_CONVEX_GNE_GAP_EVIDENCE,
        False,
        False,
        prepared.prepared_id,
    )


def solve_open_loop_gne(
    problem: FiniteHorizonLQOpenLoopGNEProblem,
    initial_controls: ArrayLike | None = None,
    /,
    *,
    constraint_args: Any = None,
    method: SemismoothNewton | None = None,
    termination: NonlinearTermination | None = None,
    phase_one_policy: ConvexSolvePolicy | None = None,
    audit_best_responses: bool = False,
    best_response_policy: ConvexSolvePolicy | None = None,
    structural_tolerance: float = 1.0e-9,
    convexity_tolerance: float = 1.0e-9,
    regularity_tolerance: float = 1.0e-9,
    feasibility_tolerance: float = 1.0e-7,
    kkt_tolerance: float = 1.0e-6,
) -> OpenLoopGNEResult:
    """Plan, prepare, solve, and audit a finite-dimensional open-loop GNE."""
    if not isinstance(problem, FiniteHorizonLQOpenLoopGNEProblem):
        raise TypeError("problem must be FiniteHorizonLQOpenLoopGNEProblem.")
    plan = plan_open_loop_gne(
        problem,
        method=method,
        termination=termination,
        phase_one_policy=phase_one_policy,
        audit_best_responses=audit_best_responses,
        best_response_policy=best_response_policy,
        structural_tolerance=structural_tolerance,
        convexity_tolerance=convexity_tolerance,
        regularity_tolerance=regularity_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        kkt_tolerance=kkt_tolerance,
    )
    controls = (
        jnp.zeros(
            problem.case_shape + (problem.horizon, problem.control_size),
            dtype=problem.dynamics_matrices.dtype,
        )
        if initial_controls is None
        else initial_controls
    )
    prepared = prepare_open_loop_gne(
        plan, problem, controls, constraint_args=constraint_args
    )
    return solve_prepared_open_loop_gne(prepared)


__all__ = [
    "FiniteHorizonLQOpenLoopGNEProblem",
    "GLOBAL_CONVEX_GNE_GAP_EVIDENCE",
    "OPEN_LOOP_GENERALIZED_NASH_KKT",
    "OpenLoopGNEPlan",
    "OpenLoopGNEResult",
    "OpenLoopGNEStatus",
    "PreparedOpenLoopGNE",
    "plan_open_loop_gne",
    "prepare_open_loop_gne",
    "refresh_open_loop_gne",
    "solve_open_loop_gne",
    "solve_prepared_open_loop_gne",
]
