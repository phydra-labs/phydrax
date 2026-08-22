#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..optim import (
    ConvexProgramResult,
    ConvexSolvePolicy,
    ConvexTermination,
    DensePrimalDualQP,
    QuadraticProgram,
    solve_quadratic_program,
)
from ..solver import DifferentialProblem, solve_diffrax
from ._constraints import evaluate_sampled_feasibility
from ._cost import evaluate_sampled_cost
from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics
from ._parameterization import PiecewiseConstantControlParameterization
from ._problem import ControlProblem
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_SUCCESS,
    ControlResult,
    ControlTrajectory,
)


MULTIPLE_SHOOTING_SUCCESS = 0
MULTIPLE_SHOOTING_MAX_ITERATIONS = 1
MULTIPLE_SHOOTING_QP_FAILED = 2
MULTIPLE_SHOOTING_LINE_SEARCH_FAILED = 3
MULTIPLE_SHOOTING_INTEGRATION_FAILED = 4
MULTIPLE_SHOOTING_ROLLOUT_FAILED = 5
MULTIPLE_SHOOTING_NONFINITE = 6


class MultipleShootingDecisionLayout(StrictModule):
    """Immutable state-first layout for one slack-free shooting decision vector."""

    num_steps: int = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_state_variables: int = eqx.field(static=True)
    num_control_variables: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    state_slice: tuple[int, int] = eqx.field(static=True)
    control_slice: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        num_steps: int,
        state_shape: tuple[int, ...],
        control_shape: tuple[int, ...],
        /,
    ):
        steps = int(num_steps)
        states = tuple(int(size) for size in state_shape)
        controls = tuple(int(size) for size in control_shape)
        if steps < 1:
            raise ValueError("Multiple shooting requires at least one segment.")
        if any(size <= 0 for size in states + controls):
            raise ValueError("State and control dimensions must be positive.")
        state_size = prod(states)
        control_size = prod(controls)
        state_variables = (steps + 1) * state_size
        control_variables = steps * control_size
        self.num_steps = steps
        self.state_shape = states
        self.control_shape = controls
        self.state_size = state_size
        self.control_size = control_size
        self.num_state_variables = state_variables
        self.num_control_variables = control_variables
        self.num_variables = state_variables + control_variables
        self.state_slice = (0, state_variables)
        self.control_slice = (state_variables, state_variables + control_variables)

    def pack(self, state_nodes: ArrayLike, control_nodes: ArrayLike, /) -> Array:
        states = jnp.asarray(state_nodes)
        controls = jnp.asarray(control_nodes)
        expected_states = (self.num_steps + 1,) + self.state_shape
        expected_controls = (self.num_steps,) + self.control_shape
        if tuple(states.shape) != expected_states:
            raise ValueError(
                f"state_nodes must have shape {expected_states}; got {states.shape}."
            )
        if tuple(controls.shape) != expected_controls:
            raise ValueError(
                f"control_nodes must have shape {expected_controls}; "
                f"got {controls.shape}."
            )
        dtype = jnp.result_type(states, controls, jnp.float32)
        return jnp.concatenate(
            (states.astype(dtype).reshape(-1), controls.astype(dtype).reshape(-1))
        )

    def unpack(self, decision: ArrayLike, /) -> tuple[Array, Array]:
        vector = jnp.asarray(decision)
        if vector.shape != (self.num_variables,):
            raise ValueError(
                f"decision must have shape ({self.num_variables},); got {vector.shape}."
            )
        state_start, state_stop = self.state_slice
        control_start, control_stop = self.control_slice
        states = vector[state_start:state_stop].reshape(
            (self.num_steps + 1,) + self.state_shape
        )
        controls = vector[control_start:control_stop].reshape(
            (self.num_steps,) + self.control_shape
        )
        return states, controls


class MultipleShootingLinearization(StrictModule):
    """Exact residuals and local derivatives defining one dense SQP subproblem."""

    quadratic_program: QuadraticProgram
    layout: MultipleShootingDecisionLayout
    objective: Array
    objective_gradient: Array
    objective_hessian: Array
    boundary_defect: Array
    continuity_defects: Array
    path_residuals: Array
    terminal_residuals: Array
    equality_residuals: Array
    inequality_residuals: Array
    equality_jacobian: Array
    inequality_jacobian: Array
    integration_valid: Array
    equality_provenance: tuple[str, ...] = eqx.field(static=True)
    inequality_provenance: tuple[str, ...] = eqx.field(static=True)
    hessian_regularization: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        quadratic_program: QuadraticProgram,
        layout: MultipleShootingDecisionLayout,
        objective: ArrayLike,
        objective_gradient: ArrayLike,
        objective_hessian: ArrayLike,
        boundary_defect: ArrayLike,
        continuity_defects: ArrayLike,
        path_residuals: ArrayLike,
        terminal_residuals: ArrayLike,
        equality_residuals: ArrayLike,
        inequality_residuals: ArrayLike,
        equality_jacobian: ArrayLike,
        inequality_jacobian: ArrayLike,
        integration_valid: ArrayLike,
        equality_provenance: tuple[str, ...],
        inequality_provenance: tuple[str, ...],
        hessian_regularization: float,
    ):
        self.quadratic_program = quadratic_program
        self.layout = layout
        self.objective = jnp.asarray(objective)
        self.objective_gradient = jnp.asarray(objective_gradient)
        self.objective_hessian = jnp.asarray(objective_hessian)
        self.boundary_defect = jnp.asarray(boundary_defect)
        self.continuity_defects = jnp.asarray(continuity_defects)
        self.path_residuals = jnp.asarray(path_residuals)
        self.terminal_residuals = jnp.asarray(terminal_residuals)
        self.equality_residuals = jnp.asarray(equality_residuals)
        self.inequality_residuals = jnp.asarray(inequality_residuals)
        self.equality_jacobian = jnp.asarray(equality_jacobian)
        self.inequality_jacobian = jnp.asarray(inequality_jacobian)
        self.integration_valid = jnp.asarray(integration_valid, dtype=bool)
        self.equality_provenance = equality_provenance
        self.inequality_provenance = inequality_provenance
        self.hessian_regularization = float(hessian_regularization)


class MultipleShootingHistory(StrictModule):
    """One deterministic record per dense QP attempted by multiple shooting."""

    objective: Array
    merit: Array
    maximum_defect: Array
    maximum_constraint_violation: Array
    kkt_residual_norm: Array
    step_size: Array
    accepted: Array
    qp_status: Array

    def __init__(
        self,
        *,
        objective: ArrayLike,
        merit: ArrayLike,
        maximum_defect: ArrayLike,
        maximum_constraint_violation: ArrayLike,
        kkt_residual_norm: ArrayLike,
        step_size: ArrayLike,
        accepted: ArrayLike,
        qp_status: ArrayLike,
    ):
        objective_ = jnp.asarray(objective)
        merit_ = jnp.asarray(merit)
        maximum_defect_ = jnp.asarray(maximum_defect)
        maximum_constraint_violation_ = jnp.asarray(maximum_constraint_violation)
        kkt_residual_norm_ = jnp.asarray(kkt_residual_norm)
        step_size_ = jnp.asarray(step_size)
        accepted_ = jnp.asarray(accepted, dtype=bool)
        qp_status_ = jnp.asarray(qp_status, dtype=jnp.int32)
        lengths = {
            value.shape
            for value in (
                objective_,
                merit_,
                maximum_defect_,
                maximum_constraint_violation_,
                kkt_residual_norm_,
                step_size_,
                accepted_,
                qp_status_,
            )
        }
        if len(lengths) != 1 or objective_.ndim != 1:
            raise ValueError(
                "All multiple-shooting history arrays must be vectors of equal length."
            )
        self.objective = objective_
        self.merit = merit_
        self.maximum_defect = maximum_defect_
        self.maximum_constraint_violation = maximum_constraint_violation_
        self.kkt_residual_norm = kkt_residual_norm_
        self.step_size = step_size_
        self.accepted = accepted_
        self.qp_status = qp_status_

    @property
    def num_iterations(self) -> int:
        return int(self.objective.shape[0])


class MultipleShootingResult(StrictModule):
    """Multiple-shooting nodes, exact defects, rollout audit, and solver status."""

    state_nodes: Array
    control_nodes: Array
    boundary_defect: Array
    continuity_defects: Array
    path_residuals: Array
    terminal_residuals: Array
    objective: Array
    maximum_defect: Array
    maximum_constraint_violation: Array
    kkt_residual_norm: Array
    rollout_state_error: Array
    rollout_result: ControlResult
    history: MultipleShootingHistory
    last_qp_result: ConvexProgramResult | None
    iterations: Array
    valid: Array
    status: Array
    layout: MultipleShootingDecisionLayout
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_nodes: ArrayLike,
        control_nodes: ArrayLike,
        boundary_defect: ArrayLike,
        continuity_defects: ArrayLike,
        path_residuals: ArrayLike,
        terminal_residuals: ArrayLike,
        objective: ArrayLike,
        maximum_defect: ArrayLike,
        maximum_constraint_violation: ArrayLike,
        kkt_residual_norm: ArrayLike,
        rollout_state_error: ArrayLike,
        rollout_result: ControlResult,
        history: MultipleShootingHistory,
        last_qp_result: ConvexProgramResult | None,
        iterations: int,
        status: int,
        layout: MultipleShootingDecisionLayout,
    ):
        self.state_nodes = jnp.asarray(state_nodes)
        self.control_nodes = jnp.asarray(control_nodes)
        self.boundary_defect = jnp.asarray(boundary_defect)
        self.continuity_defects = jnp.asarray(continuity_defects)
        self.path_residuals = jnp.asarray(path_residuals)
        self.terminal_residuals = jnp.asarray(terminal_residuals)
        self.objective = jnp.asarray(objective)
        self.maximum_defect = jnp.asarray(maximum_defect)
        self.maximum_constraint_violation = jnp.asarray(maximum_constraint_violation)
        self.kkt_residual_norm = jnp.asarray(kkt_residual_norm)
        self.rollout_state_error = jnp.asarray(rollout_state_error)
        self.rollout_result = rollout_result
        self.history = history
        self.last_qp_result = last_qp_result
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.valid = jnp.asarray(status == MULTIPLE_SHOOTING_SUCCESS, dtype=bool)
        self.layout = layout
        self.method_id = "control:multiple-shooting:dense-sqp"

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == MULTIPLE_SHOOTING_SUCCESS)

    @property
    def trajectory(self) -> ControlTrajectory:
        return self.rollout_result.trajectory

    @property
    def control_result(self) -> ControlResult:
        return self.rollout_result

    @property
    def sampled_loss(self) -> Any:
        return self.rollout_result.sampled_loss

    @property
    def feasibility(self) -> Any:
        return self.rollout_result.feasibility

    @property
    def parameters(self) -> Array:
        return self.control_nodes


class _NonlinearValues(StrictModule):
    objective: Array
    boundary_defect: Array
    continuity_defects: Array
    path_residuals: Array
    terminal_residuals: Array
    equality_residuals: Array
    inequality_residuals: Array
    integration_valid: Array


def _validate_problem(problem: ControlProblem, /) -> None:
    if not isinstance(problem, ControlProblem):
        raise TypeError("problem must be a ControlProblem.")
    if problem.case_shape:
        raise ValueError(
            "Multiple shooting currently supports one optimization case only; "
            f"got case_shape={problem.case_shape}."
        )


def _solver_options(
    *,
    solver: Any | None,
    stepsize_controller: Any | None,
    adjoint: Any | None,
    dt0: ArrayLike | None,
    event: Any | None,
    rtol: float,
    atol: float,
    max_steps: int | None,
    throw: bool,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "stepsize_controller": stepsize_controller,
        "adjoint": adjoint,
        "dt0": dt0,
        "event": event,
        "rtol": rtol,
        "atol": atol,
        "max_steps": max_steps,
        "throw": throw,
    }


def _rollout_solver_options(
    problem: ControlProblem, solver_options: dict[str, Any], /
) -> dict[str, Any]:
    if isinstance(problem.dynamics, DifferentialControlDynamics):
        return solver_options
    return {}


def _evaluate_held_control(
    problem: ControlProblem,
    parameterization: PiecewiseConstantControlParameterization,
    coefficients: ArrayLike,
    /,
    *,
    solver_options: dict[str, Any],
) -> ControlResult:
    """Evaluate differential held controls by solving each declared interval."""
    if not isinstance(problem.dynamics, DifferentialControlDynamics):
        return problem.evaluate(
            parameterization,
            coefficients,
            **solver_options,
        )

    values = jnp.asarray(coefficients)
    expected = problem.case_shape + parameterization.parameter_shape
    if tuple(values.shape) != expected:
        raise ValueError(
            f"Piecewise-constant coefficients must have shape {expected}; "
            f"got {values.shape}."
        )
    if parameterization.time_grid.time_id != problem.time_grid.time_id:
        return problem.evaluate(
            parameterization,
            values,
            **solver_options,
        )

    time_axis = len(problem.case_shape)
    states = [problem.initial_state]
    controls = []
    validity = [jnp.ones(problem.case_shape, dtype=bool)]
    backend_statuses = []
    method_id = ""
    current = problem.initial_state
    for segment in range(problem.time_grid.num_steps):
        segment_grid = eqx.tree_at(
            lambda grid: grid.times,
            problem.time_grid,
            problem.time_grid.times[segment : segment + 2],
        )
        segment_parameterization = PiecewiseConstantControlParameterization(
            segment_grid,
            problem.control_shape,
            parameterization_id=parameterization.parameterization_id,
        )
        segment_control = jnp.take(values, segment, axis=time_axis)
        segment_coefficients = jnp.expand_dims(segment_control, axis=time_axis)
        trajectory = problem.dynamics.rollout(
            segment_grid,
            current,
            segment_parameterization,
            segment_coefficients,
            args=problem.args,
            problem_id=problem.problem_id,
            **solver_options,
        )
        current = trajectory.final_state
        states.append(current)
        controls.append(jnp.take(trajectory.controls, 0, axis=time_axis))
        validity.append(
            jnp.take(trajectory.valid, 1, axis=time_axis) & trajectory.successful
        )
        backend_statuses.append(trajectory.backend_status)
        method_id = trajectory.method_id

    state_values = jnp.stack(states, axis=time_axis)
    control_values = jnp.stack(controls, axis=time_axis)
    valid_values = jnp.stack(validity, axis=time_axis)
    successful = jnp.all(valid_values, axis=-1)
    status = jnp.where(
        successful,
        CONTROL_SUCCESS,
        CONTROL_DYNAMICS_FAILED,
    ).astype(jnp.int32)
    trajectory = ControlTrajectory(
        time_grid=problem.time_grid,
        states=state_values,
        controls=control_values,
        valid=valid_values,
        status=status,
        backend_status=tuple(backend_statuses),
        case_shape=problem.case_shape,
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        problem_id=problem.problem_id,
        dynamics_id=problem.dynamics.dynamics_id,
        control_id=parameterization.parameterization_id,
        backend_id="backend:diffrax",
        method_id=method_id,
        discretization_id=problem.time_grid.time_id,
        approximation_id=parameterization.approximation_id,
    )
    sampled_loss = evaluate_sampled_cost(problem, trajectory)
    feasibility = evaluate_sampled_feasibility(problem, trajectory)
    return ControlResult(
        trajectory=trajectory,
        parameters=values,
        sampled_loss=sampled_loss,
        feasibility=feasibility,
        result_id=f"control-result:{problem.problem_id}",
        method_id=trajectory.method_id,
    )


def _segment_state(
    problem: ControlProblem,
    segment: int,
    state: Array,
    control: Array,
    /,
    *,
    solver_options: dict[str, Any],
) -> Array:
    dynamics = problem.dynamics
    time0 = problem.time_grid.times[segment]
    time1 = problem.time_grid.times[segment + 1]
    if isinstance(dynamics, DiscreteControlDynamics):
        next_state = dynamics.system.evaluate(
            time0,
            state,
            problem.args,
            inputs=control,
        )
        if tuple(next_state.shape) != problem.state_shape:
            raise ValueError(
                "DiscreteControlDynamics transition returned the wrong state shape."
            )
        return next_state

    if not isinstance(dynamics, DifferentialControlDynamics):
        raise TypeError("Unsupported control dynamics type.")

    def controlled_field(time: Array, current: Array, args: Any) -> Array:
        value = dynamics.system.evaluate(time, current, args, inputs=control)
        if tuple(value.shape) != problem.state_shape:
            raise ValueError(
                "DifferentialControlDynamics vector_field returned the wrong state shape."
            )
        return value

    differential = DifferentialProblem(
        controlled_field,
        state,
        t0=time0,
        t1=time1,
        args=problem.args,
    )
    solution = solve_diffrax(
        differential,
        save_times=jnp.stack((time0, time1)),
        dense=False,
        **solver_options,
    )
    return solution.states[-1]


def _segment_state_and_validity(
    problem: ControlProblem,
    segment: int,
    state: Array,
    control: Array,
    /,
    *,
    solver_options: dict[str, Any],
) -> tuple[Array, Array]:
    dynamics = problem.dynamics
    if isinstance(dynamics, DiscreteControlDynamics):
        next_state = _segment_state(
            problem,
            segment,
            state,
            control,
            solver_options=solver_options,
        )
        return next_state, jnp.all(jnp.isfinite(next_state))

    time0 = problem.time_grid.times[segment]
    time1 = problem.time_grid.times[segment + 1]

    def controlled_field(time: Array, current: Array, args: Any) -> Array:
        value = dynamics.system.evaluate(time, current, args, inputs=control)
        if tuple(value.shape) != problem.state_shape:
            raise ValueError(
                "DifferentialControlDynamics vector_field returned the wrong state shape."
            )
        return value

    differential = DifferentialProblem(
        controlled_field,
        state,
        t0=time0,
        t1=time1,
        args=problem.args,
    )
    solution = solve_diffrax(
        differential,
        save_times=jnp.stack((time0, time1)),
        dense=False,
        **solver_options,
    )
    next_state = solution.states[-1]
    valid = (
        jnp.all(jnp.asarray(solution.valid, dtype=bool))
        & jnp.asarray(solution.backend_result == dfx.RESULTS.successful, dtype=bool)
        & jnp.all(jnp.isfinite(next_state))
    )
    return next_state, valid


def _objective_function(
    problem: ControlProblem,
    layout: MultipleShootingDecisionLayout,
    decision: Array,
    /,
) -> Array:
    states, controls = layout.unpack(decision)
    dtype = decision.dtype
    running = jnp.asarray(0.0, dtype=dtype)
    if problem.running_cost is not None:
        samples = []
        for segment in range(layout.num_steps):
            value = jnp.asarray(
                problem.running_cost(
                    problem.time_grid.times[segment],
                    states[segment],
                    controls[segment],
                    problem.args,
                )
            )
            if value.shape != ():
                raise ValueError("RunningCost must return a scalar.")
            samples.append(value)
        running = jnp.sum(jnp.stack(samples) * problem.time_grid.durations)
    terminal = jnp.asarray(0.0, dtype=dtype)
    if problem.terminal_cost is not None:
        terminal = jnp.asarray(
            problem.terminal_cost(problem.time_grid.times[-1], states[-1], problem.args)
        )
        if terminal.shape != ():
            raise ValueError("TerminalCost must return a scalar.")
    return running + terminal


def _equality_function(
    problem: ControlProblem,
    layout: MultipleShootingDecisionLayout,
    decision: Array,
    /,
    *,
    solver_options: dict[str, Any],
) -> Array:
    states, controls = layout.unpack(decision)
    rows = [(states[0] - problem.initial_state).reshape(-1)]
    for segment in range(layout.num_steps):
        predicted = _segment_state(
            problem,
            segment,
            states[segment],
            controls[segment],
            solver_options=solver_options,
        )
        rows.append((predicted - states[segment + 1]).reshape(-1))
    return jnp.concatenate(rows)


def _inequality_function(
    problem: ControlProblem,
    layout: MultipleShootingDecisionLayout,
    decision: Array,
    /,
) -> Array:
    states, controls = layout.unpack(decision)
    rows = []
    for segment in range(layout.num_steps):
        for constraint in problem.path_constraints:
            value = jnp.asarray(
                constraint(
                    problem.time_grid.times[segment],
                    states[segment],
                    controls[segment],
                    problem.args,
                )
            )
            if value.shape != ():
                raise ValueError("PathConstraint must return a scalar.")
            rows.append(value)
    for constraint in problem.terminal_constraints:
        value = jnp.asarray(
            constraint(problem.time_grid.times[-1], states[-1], problem.args)
        )
        if value.shape != ():
            raise ValueError("TerminalConstraint must return a scalar.")
        rows.append(value)
    if rows:
        return jnp.stack(rows)
    return jnp.zeros((0,), dtype=decision.dtype)


def _nonlinear_values(
    problem: ControlProblem,
    layout: MultipleShootingDecisionLayout,
    decision: Array,
    /,
    *,
    solver_options: dict[str, Any],
) -> _NonlinearValues:
    states, controls = layout.unpack(decision)
    boundary = states[0] - problem.initial_state
    defects = []
    valid = []
    for segment in range(layout.num_steps):
        predicted, segment_valid = _segment_state_and_validity(
            problem,
            segment,
            states[segment],
            controls[segment],
            solver_options=solver_options,
        )
        defects.append(predicted - states[segment + 1])
        valid.append(segment_valid)
    continuity = jnp.stack(defects)
    integration_valid = jnp.stack(valid)

    path_rows = []
    for segment in range(layout.num_steps):
        columns = []
        for constraint in problem.path_constraints:
            value = jnp.asarray(
                constraint(
                    problem.time_grid.times[segment],
                    states[segment],
                    controls[segment],
                    problem.args,
                )
            )
            if value.shape != ():
                raise ValueError("PathConstraint must return a scalar.")
            columns.append(value)
        path_rows.append(
            jnp.stack(columns) if columns else jnp.zeros((0,), dtype=decision.dtype)
        )
    path = jnp.stack(path_rows)

    terminal_columns = []
    for constraint in problem.terminal_constraints:
        value = jnp.asarray(
            constraint(problem.time_grid.times[-1], states[-1], problem.args)
        )
        if value.shape != ():
            raise ValueError("TerminalConstraint must return a scalar.")
        terminal_columns.append(value)
    terminal = (
        jnp.stack(terminal_columns)
        if terminal_columns
        else jnp.zeros((0,), dtype=decision.dtype)
    )
    equality = jnp.concatenate((boundary.reshape(-1), continuity.reshape(-1)))
    inequality = jnp.concatenate((path.reshape(-1), terminal.reshape(-1)))
    return _NonlinearValues(
        objective=_objective_function(problem, layout, decision),
        boundary_defect=boundary,
        continuity_defects=continuity,
        path_residuals=path,
        terminal_residuals=terminal,
        equality_residuals=equality,
        inequality_residuals=inequality,
        integration_valid=integration_valid,
    )


def _constraint_provenance(
    problem: ControlProblem, layout: MultipleShootingDecisionLayout, /
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    equality = tuple(
        f"boundary:initial:state[{component}]" for component in range(layout.state_size)
    ) + tuple(
        f"continuity:segment[{segment}]:state[{component}]"
        for segment in range(layout.num_steps)
        for component in range(layout.state_size)
    )
    inequality = tuple(
        f"path:segment[{segment}]:constraint[{constraint}]"
        for segment in range(layout.num_steps)
        for constraint in range(len(problem.path_constraints))
    ) + tuple(
        f"terminal:constraint[{constraint}]"
        for constraint in range(len(problem.terminal_constraints))
    )
    return equality, inequality


def _configuration(
    *,
    hessian_regularization: float,
    max_iterations: int,
    constraint_tolerance: float,
    optimality_tolerance: float,
    rollout_tolerance: float,
    merit_penalty: float,
    armijo_fraction: float,
    line_search_contraction: float,
    max_line_search_iterations: int,
) -> tuple[float, int, float, float, float, float, float, float, int]:
    regularization = float(hessian_regularization)
    iterations = int(max_iterations)
    constraint = float(constraint_tolerance)
    optimality = float(optimality_tolerance)
    rollout = float(rollout_tolerance)
    penalty = float(merit_penalty)
    armijo = float(armijo_fraction)
    contraction = float(line_search_contraction)
    line_iterations = int(max_line_search_iterations)
    if not isfinite(regularization) or regularization < 0.0:
        raise ValueError("hessian_regularization must be finite and nonnegative.")
    if iterations < 1:
        raise ValueError("max_iterations must be positive.")
    for name, value in (
        ("constraint_tolerance", constraint),
        ("optimality_tolerance", optimality),
        ("rollout_tolerance", rollout),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not isfinite(penalty) or penalty <= 0.0:
        raise ValueError("merit_penalty must be finite and positive.")
    if not isfinite(armijo) or not 0.0 < armijo < 1.0:
        raise ValueError("armijo_fraction must lie strictly between zero and one.")
    if not isfinite(contraction) or not 0.0 < contraction < 1.0:
        raise ValueError(
            "line_search_contraction must lie strictly between zero and one."
        )
    if line_iterations < 1:
        raise ValueError("max_line_search_iterations must be positive.")
    return (
        regularization,
        iterations,
        constraint,
        optimality,
        rollout,
        penalty,
        armijo,
        contraction,
        line_iterations,
    )


def linearize_multiple_shooting(
    problem: ControlProblem,
    state_nodes: ArrayLike,
    control_nodes: ArrayLike,
    /,
    *,
    hessian_regularization: float = 0.0,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    event: Any | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    integration_max_steps: int | None = 4096,
    integration_throw: bool = False,
) -> MultipleShootingLinearization:
    """Quadratize cost and linearize exact shooting/constraint residuals."""
    _validate_problem(problem)
    regularization = float(hessian_regularization)
    if not isfinite(regularization) or regularization < 0.0:
        raise ValueError("hessian_regularization must be finite and nonnegative.")
    layout = MultipleShootingDecisionLayout(
        problem.time_grid.num_steps, problem.state_shape, problem.control_shape
    )
    decision = layout.pack(state_nodes, control_nodes)
    options = _solver_options(
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        dt0=dt0,
        event=event,
        rtol=rtol,
        atol=atol,
        max_steps=integration_max_steps,
        throw=integration_throw,
    )
    values = _nonlinear_values(problem, layout, decision, solver_options=options)
    objective, gradient = jax.value_and_grad(_objective_function, argnums=2)(
        problem, layout, decision
    )
    hessian = jax.hessian(_objective_function, argnums=2)(problem, layout, decision)
    hessian = 0.5 * (hessian + hessian.T)
    quadratic = hessian + regularization * jnp.eye(
        layout.num_variables, dtype=hessian.dtype
    )
    equality_jacobian = jax.jacrev(_equality_function, argnums=2)(
        problem, layout, decision, solver_options=options
    )
    inequality_jacobian = jax.jacrev(_inequality_function, argnums=2)(
        problem, layout, decision
    )
    qp = QuadraticProgram(
        quadratic,
        gradient,
        equality_matrix=equality_jacobian,
        equality_rhs=-values.equality_residuals,
        inequality_matrix=inequality_jacobian,
        inequality_rhs=-values.inequality_residuals,
    )
    equality_provenance, inequality_provenance = _constraint_provenance(problem, layout)
    return MultipleShootingLinearization(
        quadratic_program=qp,
        layout=layout,
        objective=objective,
        objective_gradient=gradient,
        objective_hessian=hessian,
        boundary_defect=values.boundary_defect,
        continuity_defects=values.continuity_defects,
        path_residuals=values.path_residuals,
        terminal_residuals=values.terminal_residuals,
        equality_residuals=values.equality_residuals,
        inequality_residuals=values.inequality_residuals,
        equality_jacobian=equality_jacobian,
        inequality_jacobian=inequality_jacobian,
        integration_valid=values.integration_valid,
        equality_provenance=equality_provenance,
        inequality_provenance=inequality_provenance,
        hessian_regularization=regularization,
    )


def _quadratic_is_positive_semidefinite(quadratic: Array, /) -> bool:
    eigenvalues = jnp.linalg.eigvalsh(quadratic)
    return bool(
        np.asarray(
            jnp.all(jnp.isfinite(eigenvalues)) & (jnp.min(eigenvalues) >= -1.0e-10)
        )
    )


def _maximum_abs(values: Array, /) -> Array:
    if values.size == 0:
        return jnp.asarray(0.0, dtype=values.dtype)
    return jnp.max(jnp.abs(values))


def _maximum_violation(values: Array, /) -> Array:
    if values.size == 0:
        return jnp.asarray(0.0, dtype=values.dtype)
    return jnp.maximum(jnp.max(values), 0.0)


def _merit(values: _NonlinearValues, penalty: float, /) -> Array:
    return values.objective + penalty * (
        jnp.sum(jnp.abs(values.equality_residuals))
        + jnp.sum(jnp.maximum(values.inequality_residuals, 0.0))
    )


def _linearized_merit(
    linearization: MultipleShootingLinearization,
    direction: Array,
    penalty: float,
    /,
) -> Array:
    local_objective = (
        linearization.objective
        + jnp.vdot(linearization.objective_gradient, direction)
        + 0.5 * jnp.vdot(direction, linearization.objective_hessian @ direction)
    )
    equalities = (
        linearization.equality_residuals + linearization.equality_jacobian @ direction
    )
    inequalities = (
        linearization.inequality_residuals + linearization.inequality_jacobian @ direction
    )
    return local_objective + penalty * (
        jnp.sum(jnp.abs(equalities)) + jnp.sum(jnp.maximum(inequalities, 0.0))
    )


def _kkt_residual(
    problem: ControlProblem,
    layout: MultipleShootingDecisionLayout,
    decision: Array,
    values: _NonlinearValues,
    qp_result: ConvexProgramResult,
    /,
    *,
    solver_options: dict[str, Any],
) -> Array:
    gradient = jax.grad(_objective_function, argnums=2)(problem, layout, decision)
    equality_jacobian = jax.jacrev(_equality_function, argnums=2)(
        problem, layout, decision, solver_options=solver_options
    )
    inequality_jacobian = jax.jacrev(_inequality_function, argnums=2)(
        problem, layout, decision
    )
    stationarity = (
        gradient
        + equality_jacobian.T @ qp_result.equality_dual
        + inequality_jacobian.T @ qp_result.inequality_dual
    )
    complementarity = qp_result.inequality_dual * values.inequality_residuals
    dual_violation = jnp.maximum(-qp_result.inequality_dual, 0.0)
    return jnp.maximum(
        jnp.maximum(
            _maximum_abs(values.equality_residuals),
            _maximum_violation(values.inequality_residuals),
        ),
        jnp.maximum(
            _maximum_abs(stationarity),
            jnp.maximum(_maximum_abs(complementarity), _maximum_abs(dual_violation)),
        ),
    )


def _empty_history(dtype: jnp.dtype) -> dict[str, list[Array]]:
    del dtype
    return {
        "objective": [],
        "merit": [],
        "maximum_defect": [],
        "maximum_constraint_violation": [],
        "kkt_residual_norm": [],
        "step_size": [],
        "accepted": [],
        "qp_status": [],
    }


def _history(
    values: dict[str, list[Array]], dtype: jnp.dtype, /
) -> MultipleShootingHistory:
    def vector(name: str, *, requested_dtype: Any = None) -> Array:
        sequence = values[name]
        selected_dtype = dtype if requested_dtype is None else requested_dtype
        if sequence:
            return jnp.asarray(sequence, dtype=selected_dtype)
        return jnp.zeros((0,), dtype=selected_dtype)

    return MultipleShootingHistory(
        objective=vector("objective"),
        merit=vector("merit"),
        maximum_defect=vector("maximum_defect"),
        maximum_constraint_violation=vector("maximum_constraint_violation"),
        kkt_residual_norm=vector("kkt_residual_norm"),
        step_size=vector("step_size"),
        accepted=vector("accepted", requested_dtype=bool),
        qp_status=vector("qp_status", requested_dtype=jnp.int32),
    )


def _seed_nodes(
    problem: ControlProblem,
    layout: MultipleShootingDecisionLayout,
    initial_states: ArrayLike | ControlTrajectory | None,
    initial_controls: ArrayLike | None,
    initial_trajectory: ControlTrajectory | None,
    /,
    *,
    solver_options: dict[str, Any],
) -> tuple[Array, Array]:
    trajectory = initial_trajectory
    states_input = initial_states
    controls_input = initial_controls
    if isinstance(states_input, ControlTrajectory):
        if trajectory is not None or controls_input is not None:
            raise ValueError(
                "A positional ControlTrajectory seed cannot be combined with other seeds."
            )
        trajectory = states_input
        states_input = None
    if trajectory is not None:
        if states_input is not None or controls_input is not None:
            raise ValueError(
                "initial_trajectory cannot be combined with initial state/control nodes."
            )
        if not isinstance(trajectory, ControlTrajectory):
            raise TypeError("initial_trajectory must be a ControlTrajectory.")
        if trajectory.problem_id != problem.problem_id:
            raise ValueError("initial_trajectory problem_id does not match problem.")
        if trajectory.discretization_id != problem.time_grid.time_id:
            raise ValueError(
                "initial_trajectory discretization_id does not match problem."
            )
        if trajectory.dynamics_id != problem.dynamics.dynamics_id:
            raise ValueError("initial_trajectory dynamics_id does not match problem.")
        if trajectory.case_shape:
            raise ValueError("A multiple-shooting seed trajectory cannot be batched.")
        states_input = trajectory.states
        controls_input = trajectory.controls

    if controls_input is None:
        controls = jnp.zeros(
            (layout.num_steps,) + layout.control_shape,
            dtype=problem.initial_state.dtype,
        )
    else:
        controls = jnp.asarray(controls_input)
    expected_controls = (layout.num_steps,) + layout.control_shape
    if tuple(controls.shape) != expected_controls:
        raise ValueError(
            f"initial_controls must have shape {expected_controls}; got {controls.shape}."
        )
    if not jnp.issubdtype(controls.dtype, jnp.inexact):
        controls = controls.astype(float)

    if states_input is None:
        parameterization = PiecewiseConstantControlParameterization(
            problem.time_grid,
            problem.control_shape,
            parameterization_id=f"multiple-shooting-seed:{problem.problem_id}",
        )
        trajectory_seed = problem.rollout(
            parameterization,
            controls,
            **_rollout_solver_options(problem, solver_options),
        )
        states = trajectory_seed.states
    else:
        states = jnp.asarray(states_input)
    expected_states = (layout.num_steps + 1,) + layout.state_shape
    if tuple(states.shape) != expected_states:
        raise ValueError(
            f"initial_states must have shape {expected_states}; got {states.shape}."
        )
    if not jnp.issubdtype(states.dtype, jnp.inexact):
        states = states.astype(float)
    return states, controls


def solve_multiple_shooting(
    problem: ControlProblem,
    /,
    initial_states: ArrayLike | ControlTrajectory | None = None,
    initial_controls: ArrayLike | None = None,
    *,
    initial_trajectory: ControlTrajectory | None = None,
    hessian_regularization: float = 0.0,
    max_iterations: int = 25,
    constraint_tolerance: float = 1.0e-6,
    optimality_tolerance: float = 1.0e-6,
    rollout_tolerance: float = 1.0e-5,
    merit_penalty: float = 10.0,
    armijo_fraction: float = 1.0e-4,
    line_search_contraction: float = 0.5,
    max_line_search_iterations: int = 12,
    qp_tolerance: float = 1.0e-7,
    qp_max_iterations: int = 100,
    qp_regularization: float = 0.0,
    qp_step_fraction: float = 0.995,
    max_dense_dimension: int = 512,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    event: Any | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    integration_max_steps: int | None = 4096,
    integration_throw: bool = False,
) -> MultipleShootingResult:
    """Solve one nonlinear finite-horizon problem by dense SQP multiple shooting.

    State nodes are independent decisions. Segment continuity, the initial boundary,
    and every declared path/terminal residual are imposed in each local QP. No
    projection, feasibility repair, elastic variable, fallback, or implicit
    regularization is used.
    """
    _validate_problem(problem)
    (
        hessian_regularization,
        max_iterations,
        constraint_tolerance,
        optimality_tolerance,
        rollout_tolerance,
        merit_penalty,
        armijo_fraction,
        line_search_contraction,
        max_line_search_iterations,
    ) = _configuration(
        hessian_regularization=hessian_regularization,
        max_iterations=max_iterations,
        constraint_tolerance=constraint_tolerance,
        optimality_tolerance=optimality_tolerance,
        rollout_tolerance=rollout_tolerance,
        merit_penalty=merit_penalty,
        armijo_fraction=armijo_fraction,
        line_search_contraction=line_search_contraction,
        max_line_search_iterations=max_line_search_iterations,
    )
    layout = MultipleShootingDecisionLayout(
        problem.time_grid.num_steps, problem.state_shape, problem.control_shape
    )
    options = _solver_options(
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        dt0=dt0,
        event=event,
        rtol=rtol,
        atol=atol,
        max_steps=integration_max_steps,
        throw=integration_throw,
    )
    state_nodes, control_nodes = _seed_nodes(
        problem,
        layout,
        initial_states,
        initial_controls,
        initial_trajectory,
        solver_options=options,
    )
    decision = layout.pack(state_nodes, control_nodes)
    dtype = decision.dtype
    records = _empty_history(dtype)
    last_qp_result = None
    status = MULTIPLE_SHOOTING_MAX_ITERATIONS
    kkt_norm = jnp.asarray(jnp.inf, dtype=dtype)

    values = _nonlinear_values(problem, layout, decision, solver_options=options)
    initial_finite = (
        jnp.isfinite(values.objective)
        & jnp.all(jnp.isfinite(values.equality_residuals))
        & jnp.all(jnp.isfinite(values.inequality_residuals))
    )
    if not bool(np.asarray(jnp.all(values.integration_valid))):
        status = MULTIPLE_SHOOTING_INTEGRATION_FAILED
    elif not bool(np.asarray(initial_finite)):
        status = MULTIPLE_SHOOTING_NONFINITE
    else:
        for _ in range(max_iterations):
            linearization = linearize_multiple_shooting(
                problem,
                *layout.unpack(decision),
                hessian_regularization=hessian_regularization,
                solver=solver,
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
                dt0=dt0,
                event=event,
                rtol=rtol,
                atol=atol,
                integration_max_steps=integration_max_steps,
                integration_throw=integration_throw,
            )
            if not bool(np.asarray(jnp.all(linearization.integration_valid))):
                status = MULTIPLE_SHOOTING_INTEGRATION_FAILED
                break
            if not _quadratic_is_positive_semidefinite(
                linearization.quadratic_program.quadratic
            ):
                status = MULTIPLE_SHOOTING_QP_FAILED
                break
            last_qp_result = solve_quadratic_program(
                linearization.quadratic_program,
                policy=ConvexSolvePolicy(
                    DensePrimalDualQP(
                        step_fraction=qp_step_fraction,
                        max_kkt_dimension=max_dense_dimension,
                    ),
                    termination=ConvexTermination(
                        absolute=qp_tolerance,
                        maximum_steps=qp_max_iterations,
                    ),
                    regularization=qp_regularization,
                ),
            )
            current_merit = _merit(values, merit_penalty)
            if not bool(np.asarray(last_qp_result.successful)):
                records["objective"].append(values.objective)
                records["merit"].append(current_merit)
                records["maximum_defect"].append(_maximum_abs(values.equality_residuals))
                records["maximum_constraint_violation"].append(
                    _maximum_violation(values.inequality_residuals)
                )
                records["kkt_residual_norm"].append(jnp.asarray(jnp.inf, dtype=dtype))
                kkt_norm = jnp.asarray(jnp.inf, dtype=dtype)
                records["step_size"].append(jnp.asarray(0.0, dtype=dtype))
                records["accepted"].append(jnp.asarray(False))
                records["qp_status"].append(last_qp_result.status)
                status = MULTIPLE_SHOOTING_QP_FAILED
                break

            current_kkt = _kkt_residual(
                problem,
                layout,
                decision,
                values,
                last_qp_result,
                solver_options=options,
            )
            kkt_norm = current_kkt
            current_maximum_defect = _maximum_abs(values.equality_residuals)
            current_maximum_violation = _maximum_violation(values.inequality_residuals)
            current_converged = (
                (current_maximum_defect <= constraint_tolerance)
                & (current_maximum_violation <= constraint_tolerance)
                & (current_kkt <= optimality_tolerance)
            )
            if bool(np.asarray(current_converged)):
                kkt_norm = current_kkt
                records["objective"].append(values.objective)
                records["merit"].append(current_merit)
                records["maximum_defect"].append(current_maximum_defect)
                records["maximum_constraint_violation"].append(current_maximum_violation)
                records["kkt_residual_norm"].append(current_kkt)
                records["step_size"].append(jnp.asarray(0.0, dtype=dtype))
                records["accepted"].append(jnp.asarray(True))
                records["qp_status"].append(last_qp_result.status)
                status = MULTIPLE_SHOOTING_SUCCESS
                break

            direction = last_qp_result.primal
            predicted_reduction = current_merit - _linearized_merit(
                linearization, direction, merit_penalty
            )
            accepted = False
            accepted_step = jnp.asarray(0.0, dtype=dtype)
            candidate_values = values
            candidate_decision = decision
            all_trial_integrations_failed = False
            if bool(np.asarray(jnp.isfinite(predicted_reduction))) and bool(
                np.asarray(predicted_reduction > 0.0)
            ):
                step = 1.0
                all_trial_integrations_failed = True
                for _ in range(max_line_search_iterations):
                    trial_decision = decision + step * direction
                    trial_values = _nonlinear_values(
                        problem,
                        layout,
                        trial_decision,
                        solver_options=options,
                    )
                    trial_integration_valid = bool(
                        np.asarray(jnp.all(trial_values.integration_valid))
                    )
                    if trial_integration_valid:
                        all_trial_integrations_failed = False
                    trial_merit = _merit(trial_values, merit_penalty)
                    finite_trial = (
                        jnp.isfinite(trial_merit)
                        & jnp.all(jnp.isfinite(trial_values.equality_residuals))
                        & jnp.all(jnp.isfinite(trial_values.inequality_residuals))
                    )
                    sufficient_decrease = trial_merit <= (
                        current_merit - armijo_fraction * step * predicted_reduction
                    )
                    if trial_integration_valid and bool(
                        np.asarray(finite_trial & sufficient_decrease)
                    ):
                        accepted = True
                        accepted_step = jnp.asarray(step, dtype=dtype)
                        candidate_values = trial_values
                        candidate_decision = trial_decision
                        break
                    step *= line_search_contraction

            if not accepted:
                records["objective"].append(values.objective)
                records["merit"].append(current_merit)
                records["maximum_defect"].append(_maximum_abs(values.equality_residuals))
                records["maximum_constraint_violation"].append(
                    _maximum_violation(values.inequality_residuals)
                )
                records["kkt_residual_norm"].append(jnp.asarray(jnp.inf, dtype=dtype))
                records["step_size"].append(accepted_step)
                records["accepted"].append(jnp.asarray(False))
                records["qp_status"].append(last_qp_result.status)
                status = (
                    MULTIPLE_SHOOTING_INTEGRATION_FAILED
                    if all_trial_integrations_failed
                    else MULTIPLE_SHOOTING_LINE_SEARCH_FAILED
                )
                break

            decision = candidate_decision
            values = candidate_values
            kkt_norm = _kkt_residual(
                problem,
                layout,
                decision,
                values,
                last_qp_result,
                solver_options=options,
            )
            maximum_defect = _maximum_abs(values.equality_residuals)
            maximum_violation = _maximum_violation(values.inequality_residuals)
            records["objective"].append(values.objective)
            records["merit"].append(_merit(values, merit_penalty))
            records["maximum_defect"].append(maximum_defect)
            records["maximum_constraint_violation"].append(maximum_violation)
            records["kkt_residual_norm"].append(kkt_norm)
            records["step_size"].append(accepted_step)
            records["accepted"].append(jnp.asarray(True))
            records["qp_status"].append(last_qp_result.status)
            converged = (
                (maximum_defect <= constraint_tolerance)
                & (maximum_violation <= constraint_tolerance)
                & (kkt_norm <= optimality_tolerance)
            )
            if bool(np.asarray(converged)):
                status = MULTIPLE_SHOOTING_SUCCESS
                break

    state_nodes, control_nodes = layout.unpack(decision)
    values = _nonlinear_values(problem, layout, decision, solver_options=options)
    parameterization = PiecewiseConstantControlParameterization(
        problem.time_grid,
        problem.control_shape,
        parameterization_id=f"multiple-shooting:{problem.problem_id}",
    )
    rollout_result = _evaluate_held_control(
        problem,
        parameterization,
        control_nodes,
        solver_options=_rollout_solver_options(problem, options),
    )
    rollout_state_error = _maximum_abs(rollout_result.trajectory.states - state_nodes)
    rollout_valid = (
        jnp.all(rollout_result.trajectory.successful)
        & jnp.all(rollout_result.sampled_loss.valid)
        & (rollout_result.feasibility.maximum_violation <= constraint_tolerance)
        & jnp.isfinite(rollout_state_error)
        & (rollout_state_error <= rollout_tolerance)
    )
    if status == MULTIPLE_SHOOTING_SUCCESS and not bool(np.asarray(rollout_valid)):
        status = MULTIPLE_SHOOTING_ROLLOUT_FAILED

    history = _history(records, dtype)
    return MultipleShootingResult(
        state_nodes=state_nodes,
        control_nodes=control_nodes,
        boundary_defect=values.boundary_defect,
        continuity_defects=values.continuity_defects,
        path_residuals=values.path_residuals,
        terminal_residuals=values.terminal_residuals,
        objective=values.objective,
        maximum_defect=_maximum_abs(values.equality_residuals),
        maximum_constraint_violation=_maximum_violation(values.inequality_residuals),
        kkt_residual_norm=kkt_norm,
        rollout_state_error=rollout_state_error,
        rollout_result=rollout_result,
        history=history,
        last_qp_result=last_qp_result,
        iterations=history.num_iterations,
        status=status,
        layout=layout,
    )


__all__ = [
    "MULTIPLE_SHOOTING_INTEGRATION_FAILED",
    "MULTIPLE_SHOOTING_LINE_SEARCH_FAILED",
    "MULTIPLE_SHOOTING_MAX_ITERATIONS",
    "MULTIPLE_SHOOTING_NONFINITE",
    "MULTIPLE_SHOOTING_QP_FAILED",
    "MULTIPLE_SHOOTING_ROLLOUT_FAILED",
    "MULTIPLE_SHOOTING_SUCCESS",
    "MultipleShootingDecisionLayout",
    "MultipleShootingHistory",
    "MultipleShootingLinearization",
    "MultipleShootingResult",
    "linearize_multiple_shooting",
    "solve_multiple_shooting",
]
