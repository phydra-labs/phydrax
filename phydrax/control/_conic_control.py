#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field
from ..optim import (
    ConicProgram,
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ProductCone,
    SecondOrderCone,
    solve_conic_program,
)
from ..sparse import EdgeRelation, SparseLinearMap
from ._parameterization import PiecewiseConstantControlParameterization
from ._problem import _identifier
from ._qp_compiler import (
    _append_sparse_block,
    compile_linear_quadratic_control,
    LinearControlCompilationPolicy,
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
        array = array.astype("float64")
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
    conic_result: ConvexProgramResult = fixed_field()
    trajectory: ControlTrajectory
    policy: PiecewiseConstantControlParameterization
    parameters: Array = fixed_field()
    objective: Array = fixed_field()
    valid: Array = fixed_field()
    status: Array = fixed_field()
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
    cone_dimension = constraint.left_state.shape[-2]
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
    cone_dimension = constraint.left_state.shape[-2]
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
    compilation_policy: LinearControlCompilationPolicy | None = None,
) -> LinearControlConicCompilation:
    """Compile affine dynamics, polyhedra, bounds, and SOCs without densifying."""
    selected_compilation = (
        LinearControlCompilationPolicy("sparse")
        if compilation_policy is None
        else compilation_policy
    )
    if not isinstance(selected_compilation, LinearControlCompilationPolicy):
        raise TypeError(
            "compilation_policy must be a LinearControlCompilationPolicy or None."
        )
    if selected_compilation.representation != "sparse":
        raise ValueError("Linear conic control requires sparse base compilation.")
    quadratic = compile_linear_quadratic_control(
        problem,
        cost_tolerance=cost_tolerance,
        compilation_policy=selected_compilation,
    )
    base = quadratic.program
    if not isinstance(base, ConicProgram):
        raise RuntimeError("Sparse control compilation did not produce a ConicProgram.")
    if not isinstance(base.constraint_matrix, SparseLinearMap):
        raise RuntimeError("Sparse control constraints require a SparseLinearMap.")
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
    base_rows = base.constraint_matrix.target.size
    total_soc_rows = problem.horizon * sum(stage_dimensions) + sum(terminal_dimensions)
    relation = base.constraint_matrix.relation
    relation_valid = np.asarray(relation.valid)
    row_routes = [np.asarray(relation.target_indices, dtype=np.int32)[relation_valid]]
    column_routes = [np.asarray(relation.source_indices, dtype=np.int32)[relation_valid]]
    coefficient_blocks = [base.constraint_matrix.coefficients[..., relation_valid]]
    rhs_blocks = [base.constraint_rhs]
    if not isinstance(base.cone, ProductCone):
        raise RuntimeError("Sparse control base cone must be a ProductCone.")
    cones = list(base.cone.cones)
    cursor = base_rows
    stage_slices: list[tuple[slice, ...]] = []
    for constraint, dimension in zip(stages, stage_dimensions, strict=True):
        constraint_slices = []
        for stage in range(problem.horizon):
            rows = slice(cursor, cursor + dimension)
            constraint_slices.append(rows)
            state_slice = quadratic.decision_layout.state_slice(stage)
            control_slice = quadratic.decision_layout.control_slice(stage)
            scalar_row = slice(rows.start, rows.start + 1)
            vector_rows = slice(rows.start + 1, rows.stop)
            _append_sparse_block(
                row_routes,
                column_routes,
                coefficient_blocks,
                scalar_row,
                state_slice,
                -constraint.right_state[..., stage, None, :],
            )
            _append_sparse_block(
                row_routes,
                column_routes,
                coefficient_blocks,
                scalar_row,
                control_slice,
                -constraint.right_control[..., stage, None, :],
            )
            _append_sparse_block(
                row_routes,
                column_routes,
                coefficient_blocks,
                vector_rows,
                state_slice,
                -constraint.left_state[..., stage, :, :],
            )
            _append_sparse_block(
                row_routes,
                column_routes,
                coefficient_blocks,
                vector_rows,
                control_slice,
                -constraint.left_control[..., stage, :, :],
            )
            rhs_blocks.append(
                jnp.concatenate(
                    (
                        constraint.right_offset[..., stage, None],
                        constraint.left_offset[..., stage, :],
                    ),
                    axis=-1,
                )
            )
            cones.append(SecondOrderCone(dimension))
            cursor += dimension
        stage_slices.append(tuple(constraint_slices))
    terminal_slices = []
    terminal_state = quadratic.decision_layout.state_slice(problem.horizon)
    for constraint, dimension in zip(terminals, terminal_dimensions, strict=True):
        rows = slice(cursor, cursor + dimension)
        terminal_slices.append(rows)
        scalar_row = slice(rows.start, rows.start + 1)
        vector_rows = slice(rows.start + 1, rows.stop)
        _append_sparse_block(
            row_routes,
            column_routes,
            coefficient_blocks,
            scalar_row,
            terminal_state,
            -constraint.right_state[..., None, :],
        )
        _append_sparse_block(
            row_routes,
            column_routes,
            coefficient_blocks,
            vector_rows,
            terminal_state,
            -constraint.left_state,
        )
        rhs_blocks.append(
            jnp.concatenate(
                (
                    constraint.right_offset[..., None],
                    constraint.left_offset,
                ),
                axis=-1,
            )
        )
        cones.append(SecondOrderCone(dimension))
        cursor += dimension
    rows = np.concatenate(row_routes)
    columns = np.concatenate(column_routes)
    coefficients = jnp.concatenate(tuple(coefficient_blocks), axis=-1)
    relation = EdgeRelation(
        jnp.asarray(columns),
        jnp.asarray(rows),
        source_size=quadratic.decision_layout.num_variables,
        target_size=base_rows + total_soc_rows,
    )
    constraint_matrix = SparseLinearMap(
        relation,
        coefficients,
        operator_id=f"{problem.problem_id}:conic-constraints",
    )
    conic = ConicProgram(
        base.quadratic,
        base.linear,
        constraint_matrix,
        jnp.concatenate(tuple(rhs_blocks), axis=-1),
        ProductCone(tuple(cones)),
        bounds=base.bounds,
        problem_id=f"{problem.problem_id}:conic",
        convexity_evidence=base.convexity_evidence,
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
    compilation_policy: LinearControlCompilationPolicy | None = None,
    solution_id: str | None = None,
) -> LinearControlConicSolution:
    """Compile, solve, and decode one finite-horizon quadratic SOCP."""

    compilation = compile_linear_conic_control(
        problem,
        stage_constraints=stage_constraints,
        terminal_constraints=terminal_constraints,
        cost_tolerance=cost_tolerance,
        compilation_policy=compilation_policy,
    )
    result = solve_conic_program(compilation.conic_program, policy=policy)
    states, controls = compilation.decode(result.primal)
    finite_nodes = jnp.all(jnp.isfinite(states), axis=-1)
    finite_controls = jnp.all(jnp.isfinite(controls), axis=-1)
    trajectory_valid = (
        jnp.concatenate(
            (
                result.valid[..., None] & finite_controls,
                result.valid[..., None],
            ),
            axis=-1,
        )
        & finite_nodes
    )
    decoded_valid = (
        result.valid & jnp.all(finite_nodes, axis=-1) & jnp.all(finite_controls, axis=-1)
    )
    decoded_status = jnp.where(
        (result.status == int(ConvexProgramStatus.OPTIMAL)) & ~decoded_valid,
        int(ConvexProgramStatus.NONFINITE_OUTPUT),
        result.status,
    ).astype(jnp.int32)
    control_status = jnp.where(
        decoded_valid,
        CONTROL_SUCCESS,
        jnp.where(
            decoded_status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE),
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
        backend_status=decoded_status,
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
        "control-conic-solution:"
        + canonical_fingerprint(
            {
                "specification": problem.specification_id,
                "numeric_binding": result.provenance.numeric_binding_id,
                "method": result.provenance.method_id,
                "result": array_tree_fingerprint(
                    {
                        "primal": result.primal,
                        "objective": result.objective,
                        "status": decoded_status,
                        "valid": decoded_valid,
                    }
                ),
            }
        )
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
        valid=decoded_valid,
        status=decoded_status,
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
