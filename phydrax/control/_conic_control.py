#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._bounds import Bounds
from .._strict import StrictModule
from ..optim import (
    ConicProgram,
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    NonnegativeCone,
    ProductCone,
    SecondOrderCone,
    solve_conic_program,
    ZeroCone,
)
from ._parameterization import PiecewiseConstantControlParameterization
from ._problem import _identifier
from ._qp_compiler import (
    compile_linear_quadratic_control,
    LinearControlDecisionLayout,
    LinearControlQPCompilation,
    LinearQuadraticControlProblem,
)
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_INFEASIBLE,
    CONTROL_SUCCESS,
    ControlTrajectory,
)


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


class StageSecondOrderConstraint(StrictModule):
    """Stagewise affine SOC ``||F_x x + F_u u + f|| <= g_x x + g_u u + g0``."""

    left_state: Array
    left_control: Array
    left_offset: Array
    right_state: Array
    right_control: Array
    right_offset: Array
    label: str = eqx.field(static=True)

    def __init__(
        self,
        left_state: ArrayLike,
        left_control: ArrayLike,
        left_offset: ArrayLike,
        right_state: ArrayLike,
        right_control: ArrayLike,
        right_offset: ArrayLike,
        /,
        *,
        label: str = "stage-soc",
    ):
        identifier = str(label)
        if not identifier:
            raise ValueError("label must be non-empty.")
        self.left_state = _real_array(left_state, "left_state")
        self.left_control = _real_array(left_control, "left_control")
        self.left_offset = _real_array(left_offset, "left_offset")
        self.right_state = _real_array(right_state, "right_state")
        self.right_control = _real_array(right_control, "right_control")
        self.right_offset = _real_array(right_offset, "right_offset")
        self.label = identifier


class TerminalSecondOrderConstraint(StrictModule):
    """Terminal affine SOC ``||F x + f|| <= g x + g0``."""

    left_state: Array
    left_offset: Array
    right_state: Array
    right_offset: Array
    label: str = eqx.field(static=True)

    def __init__(
        self,
        left_state: ArrayLike,
        left_offset: ArrayLike,
        right_state: ArrayLike,
        right_offset: ArrayLike,
        /,
        *,
        label: str = "terminal-soc",
    ):
        identifier = str(label)
        if not identifier:
            raise ValueError("label must be non-empty.")
        self.left_state = _real_array(left_state, "left_state")
        self.left_offset = _real_array(left_offset, "left_offset")
        self.right_state = _real_array(right_state, "right_state")
        self.right_offset = _real_array(right_offset, "right_offset")
        self.label = identifier


class LinearControlConicCompilation(StrictModule):
    """Quadratic-conic control program with decision and SOC block provenance."""

    quadratic_compilation: LinearControlQPCompilation
    conic_program: ConicProgram
    stage_soc_slices: tuple[tuple[slice, ...], ...] = eqx.field(static=True)
    terminal_soc_slices: tuple[slice, ...] = eqx.field(static=True)
    compiler_id: str = eqx.field(static=True)

    @property
    def decision_layout(self) -> LinearControlDecisionLayout:
        return self.quadratic_compilation.decision_layout

    def decode(self, primal: ArrayLike, /) -> tuple[Array, Array]:
        return self.decision_layout.decode(primal)


class LinearControlConicSolution(StrictModule):
    """Decoded conic-control solution and complete conic solver evidence."""

    compilation: LinearControlConicCompilation
    conic_result: ConvexProgramResult
    trajectory: ControlTrajectory
    policy: PiecewiseConstantControlParameterization
    parameters: Array
    objective: Array
    valid: Array
    status: Array
    solution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def states(self) -> Array:
        return self.trajectory.states

    @property
    def controls(self) -> Array:
        return self.parameters

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(ConvexProgramStatus.OPTIMAL))


def _validate_stage_constraint(
    constraint: StageSecondOrderConstraint,
    problem: LinearQuadraticControlProblem,
    /,
) -> int:
    batch = problem.case_shape
    horizon = problem.horizon
    state = problem.state_size
    control = problem.control_size
    if constraint.left_state.ndim < 3:
        raise ValueError("stage left_state must include stage, cone, and state axes.")
    cone_dimension = int(constraint.left_state.shape[-2])
    expected = batch + (horizon, cone_dimension, state)
    if tuple(constraint.left_state.shape) != expected:
        raise ValueError(f"stage left_state must have shape {expected}.")
    if tuple(constraint.left_control.shape) != batch + (
        horizon,
        cone_dimension,
        control,
    ):
        raise ValueError("stage left_control has incompatible shape.")
    if tuple(constraint.left_offset.shape) != batch + (horizon, cone_dimension):
        raise ValueError("stage left_offset has incompatible shape.")
    if tuple(constraint.right_state.shape) != batch + (horizon, state):
        raise ValueError("stage right_state has incompatible shape.")
    if tuple(constraint.right_control.shape) != batch + (horizon, control):
        raise ValueError("stage right_control has incompatible shape.")
    if tuple(constraint.right_offset.shape) != batch + (horizon,):
        raise ValueError("stage right_offset has incompatible shape.")
    return cone_dimension + 1


def _validate_terminal_constraint(
    constraint: TerminalSecondOrderConstraint,
    problem: LinearQuadraticControlProblem,
    /,
) -> int:
    batch = problem.case_shape
    state = problem.state_size
    if constraint.left_state.ndim < 2:
        raise ValueError("terminal left_state must include cone and state axes.")
    cone_dimension = int(constraint.left_state.shape[-2])
    if tuple(constraint.left_state.shape) != batch + (cone_dimension, state):
        raise ValueError("terminal left_state has incompatible shape.")
    if tuple(constraint.left_offset.shape) != batch + (cone_dimension,):
        raise ValueError("terminal left_offset has incompatible shape.")
    if tuple(constraint.right_state.shape) != batch + (state,):
        raise ValueError("terminal right_state has incompatible shape.")
    if tuple(constraint.right_offset.shape) != batch:
        raise ValueError("terminal right_offset has incompatible shape.")
    return cone_dimension + 1


def compile_linear_conic_control(
    problem: LinearQuadraticControlProblem,
    /,
    *,
    stage_constraints: Sequence[StageSecondOrderConstraint] = (),
    terminal_constraints: Sequence[TerminalSecondOrderConstraint] = (),
    cost_tolerance: float = 1e-10,
) -> LinearControlConicCompilation:
    """Compile affine dynamics, polyhedra, native bounds, and exact SOC constraints."""

    quadratic = compile_linear_quadratic_control(problem, cost_tolerance=cost_tolerance)
    qp = quadratic.quadratic_program
    stages = tuple(stage_constraints)
    terminals = tuple(terminal_constraints)
    if any(not isinstance(value, StageSecondOrderConstraint) for value in stages):
        raise TypeError(
            "stage_constraints must contain StageSecondOrderConstraint values."
        )
    if any(not isinstance(value, TerminalSecondOrderConstraint) for value in terminals):
        raise TypeError(
            "terminal_constraints must contain TerminalSecondOrderConstraint values."
        )
    stage_dimensions = tuple(
        _validate_stage_constraint(value, problem) for value in stages
    )
    terminal_dimensions = tuple(
        _validate_terminal_constraint(value, problem) for value in terminals
    )
    base_rows = qp.num_user_equalities + qp.num_user_inequalities
    total_soc_rows = problem.horizon * sum(stage_dimensions) + sum(terminal_dimensions)
    matrix = jnp.zeros(
        problem.case_shape
        + (base_rows + total_soc_rows, quadratic.decision_layout.num_variables),
        dtype=qp.linear.dtype,
    )
    rhs = jnp.zeros(
        problem.case_shape + (base_rows + total_soc_rows,), dtype=qp.linear.dtype
    )
    equality_rows = slice(0, qp.num_user_equalities)
    inequality_rows = slice(qp.num_user_equalities, base_rows)
    matrix = matrix.at[..., equality_rows, :].set(
        qp.equality_matrix[..., : qp.num_user_equalities, :]
    )
    rhs = rhs.at[..., equality_rows].set(qp.equality_rhs[..., : qp.num_user_equalities])
    matrix = matrix.at[..., inequality_rows, :].set(
        qp.inequality_matrix[..., : qp.num_user_inequalities, :]
    )
    rhs = rhs.at[..., inequality_rows].set(
        qp.inequality_rhs[..., : qp.num_user_inequalities]
    )
    cones = [ZeroCone(qp.num_user_equalities), NonnegativeCone(qp.num_user_inequalities)]
    cursor = base_rows
    stage_slices: list[tuple[slice, ...]] = []
    for constraint, dimension in zip(stages, stage_dimensions, strict=True):
        constraint_slices = []
        for stage in range(problem.horizon):
            rows = slice(cursor, cursor + dimension)
            constraint_slices.append(rows)
            state_slice = quadratic.decision_layout.state_slice(stage)
            control_slice = quadratic.decision_layout.control_slice(stage)
            matrix = matrix.at[..., rows.start, state_slice].set(
                -constraint.right_state[..., stage, :]
            )
            matrix = matrix.at[..., rows.start, control_slice].set(
                -constraint.right_control[..., stage, :]
            )
            matrix = matrix.at[..., rows.start + 1 : rows.stop, state_slice].set(
                -constraint.left_state[..., stage, :, :]
            )
            matrix = matrix.at[..., rows.start + 1 : rows.stop, control_slice].set(
                -constraint.left_control[..., stage, :, :]
            )
            rhs = rhs.at[..., rows.start].set(constraint.right_offset[..., stage])
            rhs = rhs.at[..., rows.start + 1 : rows.stop].set(
                constraint.left_offset[..., stage, :]
            )
            cones.append(SecondOrderCone(dimension))
            cursor += dimension
        stage_slices.append(tuple(constraint_slices))
    terminal_slices = []
    terminal_state = quadratic.decision_layout.state_slice(problem.horizon)
    for constraint, dimension in zip(terminals, terminal_dimensions, strict=True):
        rows = slice(cursor, cursor + dimension)
        terminal_slices.append(rows)
        matrix = matrix.at[..., rows.start, terminal_state].set(-constraint.right_state)
        matrix = matrix.at[..., rows.start + 1 : rows.stop, terminal_state].set(
            -constraint.left_state
        )
        rhs = rhs.at[..., rows.start].set(constraint.right_offset)
        rhs = rhs.at[..., rows.start + 1 : rows.stop].set(constraint.left_offset)
        cones.append(SecondOrderCone(dimension))
        cursor += dimension
    conic = ConicProgram(
        qp.quadratic,
        qp.linear,
        matrix,
        rhs,
        ProductCone(tuple(cones)),
        bounds=Bounds(qp.lower_bounds, qp.upper_bounds),
        problem_id=f"{problem.problem_id}:conic",
        convexity_evidence=qp.convexity_evidence,
    )
    return LinearControlConicCompilation(
        quadratic_compilation=quadratic,
        conic_program=conic,
        stage_soc_slices=tuple(stage_slices),
        terminal_soc_slices=tuple(terminal_slices),
        compiler_id="control:conic-compiler:linear-socp",
    )


def solve_linear_conic_control(
    problem: LinearQuadraticControlProblem,
    policy: ConvexSolvePolicy,
    /,
    *,
    stage_constraints: Sequence[StageSecondOrderConstraint] = (),
    terminal_constraints: Sequence[TerminalSecondOrderConstraint] = (),
    cost_tolerance: float = 1e-10,
    solution_id: str | None = None,
) -> LinearControlConicSolution:
    """Compile, solve, and decode one finite-horizon quadratic SOCP."""

    compilation = compile_linear_conic_control(
        problem,
        stage_constraints=stage_constraints,
        terminal_constraints=terminal_constraints,
        cost_tolerance=cost_tolerance,
    )
    result = solve_conic_program(compilation.conic_program, policy=policy)
    states, controls = compilation.decode(result.primal)
    finite_nodes = jnp.all(jnp.isfinite(states), axis=-1)
    trajectory_valid = result.valid[..., None] & finite_nodes
    control_status = jnp.where(
        result.status == int(ConvexProgramStatus.OPTIMAL),
        CONTROL_SUCCESS,
        jnp.where(
            result.status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE),
            CONTROL_INFEASIBLE,
            CONTROL_DYNAMICS_FAILED,
        ),
    ).astype(jnp.int32)
    policy_id = f"{problem.problem_id}:conic-policy"
    control_policy = PiecewiseConstantControlParameterization(
        problem.time_grid,
        (problem.control_size,),
        parameterization_id=policy_id,
    )
    trajectory = ControlTrajectory(
        time_grid=problem.time_grid,
        states=states,
        controls=controls,
        valid=trajectory_valid,
        status=control_status,
        backend_status=result.status,
        case_shape=problem.case_shape,
        state_shape=(problem.state_size,),
        control_shape=(problem.control_size,),
        problem_id=problem.problem_id,
        dynamics_id=problem.dynamics_id,
        control_id=policy_id,
        backend_id=result.backend,
        method_id=f"control:conic:{result.method}",
        discretization_id="control:discrete:exact-affine",
        approximation_id=control_policy.approximation_id,
    )
    identifier = (
        f"{problem.problem_id}:conic-solution"
        if solution_id is None
        else _identifier(solution_id, "solution_id")
    )
    return LinearControlConicSolution(
        compilation=compilation,
        conic_result=result,
        trajectory=trajectory,
        policy=control_policy,
        parameters=controls,
        objective=result.objective + compilation.quadratic_compilation.objective_constant,
        valid=result.valid,
        status=result.status,
        solution_id=identifier,
        method_id=f"control:conic:{result.method}",
    )


__all__ = [
    "LinearControlConicCompilation",
    "LinearControlConicSolution",
    "StageSecondOrderConstraint",
    "TerminalSecondOrderConstraint",
    "compile_linear_conic_control",
    "solve_linear_conic_control",
]
