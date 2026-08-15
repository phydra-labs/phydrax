#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Receding-horizon MPC for canonical finite linear-control QPs."""

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ..optim._quadratic_program import (
    QP_INFEASIBLE,
    QP_NONFINITE,
    QP_SUCCESS,
    QPMethod,
    QuadraticProgramResult,
)
from ._parameterization import PiecewiseConstantControlParameterization
from ._problem import _identifier
from ._qp_compiler import (
    LinearControlQPSolution,
    LinearQuadraticControlProblem,
    solve_linear_quadratic_control,
)
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_INFEASIBLE,
    CONTROL_SUCCESS,
    ControlTrajectory,
)


MPCTerminalPolicy: TypeAlias = Literal["global", "always", "none"]


class RecedingHorizonMPCResult(StrictModule):
    """Applied MPC rollout and every local QP result, without hidden repair."""

    trajectory: ControlTrajectory
    policy: PiecewiseConstantControlParameterization
    parameters: Array
    subproblem_solutions: tuple[LinearControlQPSolution, ...]
    qp_results: tuple[QuadraticProgramResult, ...]
    objective: Array
    stage_valid: Array
    valid: Array
    status: Array
    prediction_horizon: int = eqx.field(static=True)
    terminal_policy: MPCTerminalPolicy = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def states(self) -> Array:
        return self.trajectory.states

    @property
    def controls(self) -> Array:
        return self.parameters

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == QP_SUCCESS)


class RecedingHorizonMPC(StrictModule):
    """A configured linear MPC controller with an explicit terminal policy.

    ``terminal_policy="global"`` applies terminal cost and terminal constraints
    only when a prediction window reaches the specification's final node.
    ``"always"`` applies them at every prediction endpoint, and ``"none"``
    omits them even at the global endpoint. Warm starts are not implemented and
    passing one raises instead of silently ignoring it.
    """

    specification: LinearQuadraticControlProblem
    prediction_horizon: int = eqx.field(static=True)
    terminal_policy: MPCTerminalPolicy = eqx.field(static=True)
    method: QPMethod = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    cost_tolerance: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    step_fraction: float = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)

    def __init__(
        self,
        specification: LinearQuadraticControlProblem,
        /,
        *,
        prediction_horizon: int,
        terminal_policy: MPCTerminalPolicy,
        method: QPMethod = "dense-primal-dual",
        tolerance: float = 1e-7,
        cost_tolerance: float = 1e-10,
        max_iterations: int = 100,
        regularization: float = 0.0,
        step_fraction: float = 0.995,
        max_dense_dimension: int = 512,
        controller_id: str = "control:mpc:receding-horizon",
    ):
        if not isinstance(specification, LinearQuadraticControlProblem):
            raise TypeError("specification must be a LinearQuadraticControlProblem.")
        if (
            not isinstance(prediction_horizon, int)
            or not 1 <= prediction_horizon <= specification.horizon
        ):
            raise ValueError(
                f"prediction_horizon must be an integer in [1, {specification.horizon}]."
            )
        if terminal_policy not in ("global", "always", "none"):
            raise ValueError("terminal_policy must be 'global', 'always', or 'none'.")
        self.specification = specification
        self.prediction_horizon = prediction_horizon
        self.terminal_policy = terminal_policy
        self.method = method
        self.tolerance = float(tolerance)
        self.cost_tolerance = float(cost_tolerance)
        self.max_iterations = int(max_iterations)
        self.regularization = float(regularization)
        self.step_fraction = float(step_fraction)
        self.max_dense_dimension = int(max_dense_dimension)
        self.controller_id = _identifier(controller_id, "controller_id")

    def solve(
        self,
        /,
        *,
        initial_state: ArrayLike | None = None,
        warm_start: ArrayLike | None = None,
    ) -> RecedingHorizonMPCResult:
        """Solve each local QP, hand off the exact state, and roll out controls."""
        if warm_start is not None:
            raise NotImplementedError(
                "RecedingHorizonMPC does not implement warm starts; "
                "warm_start must be None."
            )
        specification = self.specification
        if initial_state is None:
            current_state = specification.initial_state
        else:
            current_state = jnp.asarray(initial_state)
            if jnp.issubdtype(current_state.dtype, jnp.complexfloating):
                raise TypeError("initial_state must be real-valued.")
            current_state = current_state.astype(specification.initial_state.dtype)
        expected_initial = specification.case_shape + (specification.state_size,)
        if tuple(current_state.shape) != expected_initial:
            raise ValueError(
                f"initial_state must have shape {expected_initial}; "
                f"got {current_state.shape}."
            )

        applied_controls: list[Array] = []
        subproblem_solutions: list[LinearControlQPSolution] = []
        online_states = [current_state]
        for stage in range(specification.horizon):
            local_horizon = min(self.prediction_horizon, specification.horizon - stage)
            local_problem = self._subproblem(stage, local_horizon, current_state)
            local_solution = solve_linear_quadratic_control(
                local_problem,
                method=self.method,
                tolerance=self.tolerance,
                cost_tolerance=self.cost_tolerance,
                max_iterations=self.max_iterations,
                regularization=self.regularization,
                step_fraction=self.step_fraction,
                max_dense_dimension=self.max_dense_dimension,
            )
            applied_control = local_solution.controls[..., 0, :]
            next_state = (
                oe.contract(
                    "...ij,...j->...i",
                    specification.dynamics_matrices[..., stage, :, :],
                    current_state,
                )
                + oe.contract(
                    "...ij,...j->...i",
                    specification.control_matrices[..., stage, :, :],
                    applied_control,
                )
                + specification.dynamics_bias[..., stage, :]
            )
            subproblem_solutions.append(local_solution)
            applied_controls.append(applied_control)
            online_states.append(next_state)
            current_state = next_state

        case_axis = len(specification.case_shape)
        controls = jnp.stack(applied_controls, axis=case_axis)

        # The handed-off states are the exact full affine rollout of the applied
        # controls, not copied prediction nodes from any local QP.
        states = jnp.stack(online_states, axis=case_axis)

        qp_results = tuple(solution.qp_result for solution in subproblem_solutions)
        stage_valid_values = []
        node_valid_values = [jnp.all(jnp.isfinite(online_states[0]), axis=-1)]
        status = jnp.zeros(specification.case_shape, dtype=jnp.int32)
        cumulative_valid = node_valid_values[0]
        for stage, result in enumerate(qp_results):
            control = jnp.take(controls, stage, axis=case_axis)
            next_state = online_states[stage + 1]
            finite_step = jnp.all(jnp.isfinite(control), axis=-1) & jnp.all(
                jnp.isfinite(next_state), axis=-1
            )
            local_valid = result.valid & finite_step
            local_status = jnp.where(
                result.valid & ~finite_step,
                QP_NONFINITE,
                result.status,
            ).astype(jnp.int32)
            status = jnp.where(
                (status == QP_SUCCESS) & (local_status != QP_SUCCESS),
                local_status,
                status,
            )
            cumulative_valid = cumulative_valid & local_valid
            stage_valid_values.append(local_valid)
            node_valid_values.append(cumulative_valid)
        stage_valid = jnp.stack(stage_valid_values, axis=case_axis)
        trajectory_valid = jnp.stack(node_valid_values, axis=case_axis)
        valid = jnp.all(stage_valid, axis=-1) & jnp.all(trajectory_valid, axis=-1)
        status = jnp.where(
            (status == QP_SUCCESS) & ~valid,
            QP_NONFINITE,
            status,
        ).astype(jnp.int32)

        policy_id = f"{self.controller_id}:policy"
        policy = PiecewiseConstantControlParameterization(
            specification.time_grid,
            (specification.control_size,),
            parameterization_id=policy_id,
        )
        control_status = jnp.where(
            status == QP_SUCCESS,
            CONTROL_SUCCESS,
            jnp.where(
                status == QP_INFEASIBLE,
                CONTROL_INFEASIBLE,
                CONTROL_DYNAMICS_FAILED,
            ),
        ).astype(jnp.int32)
        backend_status = jnp.stack(
            tuple(result.status for result in qp_results), axis=case_axis
        )
        trajectory = ControlTrajectory(
            time_grid=specification.time_grid,
            states=states,
            controls=controls,
            valid=trajectory_valid,
            status=control_status,
            backend_status=backend_status,
            case_shape=specification.case_shape,
            state_shape=(specification.state_size,),
            control_shape=(specification.control_size,),
            problem_id=specification.problem_id,
            dynamics_id=specification.dynamics_id,
            control_id=policy_id,
            backend_id=qp_results[0].backend,
            method_id=f"control:mpc:{self.method}",
            discretization_id="control:discrete:exact-affine",
            approximation_id=policy.approximation_id,
        )
        objective = _realized_objective(specification, states, controls)
        return RecedingHorizonMPCResult(
            trajectory=trajectory,
            policy=policy,
            parameters=controls,
            subproblem_solutions=tuple(subproblem_solutions),
            qp_results=qp_results,
            objective=objective,
            stage_valid=stage_valid,
            valid=valid,
            status=status,
            prediction_horizon=self.prediction_horizon,
            terminal_policy=self.terminal_policy,
            result_id=f"{self.controller_id}:result",
            method_id=f"control:mpc:{self.method}",
        )

    __call__ = solve

    def _subproblem(
        self,
        stage: int,
        local_horizon: int,
        initial_state: Array,
        /,
    ) -> LinearQuadraticControlProblem:
        specification = self.specification
        end = stage + local_horizon
        apply_terminal = self.terminal_policy == "always" or (
            self.terminal_policy == "global" and end == specification.horizon
        )
        batch = specification.case_shape
        dtype = specification.dynamics_matrices.dtype
        state_size = specification.state_size
        terminal_state_cost = (
            specification.terminal_state_cost
            if apply_terminal
            else jnp.zeros(batch + (state_size, state_size), dtype=dtype)
        )
        terminal_linear = (
            specification.terminal_linear
            if apply_terminal
            else jnp.zeros(batch + (state_size,), dtype=dtype)
        )
        terminal_constant = (
            specification.terminal_constant
            if apply_terminal
            else jnp.zeros(batch, dtype=dtype)
        )
        terminal_equality_matrix = (
            specification.terminal_equality_matrix if apply_terminal else None
        )
        terminal_equality_rhs = (
            specification.terminal_equality_rhs if apply_terminal else None
        )
        terminal_inequality_matrix = (
            specification.terminal_inequality_matrix if apply_terminal else None
        )
        terminal_inequality_rhs = (
            specification.terminal_inequality_rhs if apply_terminal else None
        )
        local_time_grid = TimeGrid(
            specification.time_grid.times[stage : end + 1],
            time_id=f"{specification.time_grid.time_id}:mpc:{stage}:{end}",
        )
        stage_slice = slice(stage, end)
        state_slice = slice(stage, end + 1)
        return LinearQuadraticControlProblem(
            specification.dynamics_matrices[..., stage_slice, :, :],
            specification.control_matrices[..., stage_slice, :, :],
            initial_state,
            specification.state_costs[..., stage_slice, :, :],
            specification.control_costs[..., stage_slice, :, :],
            terminal_state_cost,
            dynamics_bias=specification.dynamics_bias[..., stage_slice, :],
            state_control_cross=specification.state_control_cross[..., stage_slice, :, :],
            state_linear=specification.state_linear[..., stage_slice, :],
            control_linear=specification.control_linear[..., stage_slice, :],
            stage_constants=specification.stage_constants[..., stage_slice],
            terminal_linear=terminal_linear,
            terminal_constant=terminal_constant,
            state_lower_bounds=(
                None
                if specification.state_lower_bounds is None
                else specification.state_lower_bounds[..., state_slice, :]
            ),
            state_upper_bounds=(
                None
                if specification.state_upper_bounds is None
                else specification.state_upper_bounds[..., state_slice, :]
            ),
            control_lower_bounds=(
                None
                if specification.control_lower_bounds is None
                else specification.control_lower_bounds[..., stage_slice, :]
            ),
            control_upper_bounds=(
                None
                if specification.control_upper_bounds is None
                else specification.control_upper_bounds[..., stage_slice, :]
            ),
            stage_equality_state_matrix=(
                None
                if specification.stage_equality_state_matrix is None
                else specification.stage_equality_state_matrix[..., stage_slice, :, :]
            ),
            stage_equality_control_matrix=(
                None
                if specification.stage_equality_control_matrix is None
                else specification.stage_equality_control_matrix[..., stage_slice, :, :]
            ),
            stage_equality_rhs=(
                None
                if specification.stage_equality_rhs is None
                else specification.stage_equality_rhs[..., stage_slice, :]
            ),
            stage_inequality_state_matrix=(
                None
                if specification.stage_inequality_state_matrix is None
                else specification.stage_inequality_state_matrix[..., stage_slice, :, :]
            ),
            stage_inequality_control_matrix=(
                None
                if specification.stage_inequality_control_matrix is None
                else specification.stage_inequality_control_matrix[..., stage_slice, :, :]
            ),
            stage_inequality_rhs=(
                None
                if specification.stage_inequality_rhs is None
                else specification.stage_inequality_rhs[..., stage_slice, :]
            ),
            terminal_equality_matrix=terminal_equality_matrix,
            terminal_equality_rhs=terminal_equality_rhs,
            terminal_inequality_matrix=terminal_inequality_matrix,
            terminal_inequality_rhs=terminal_inequality_rhs,
            time_grid=local_time_grid,
            problem_id=f"{specification.problem_id}:mpc:{stage}:{end}",
            dynamics_id=specification.dynamics_id,
        )


def _realized_objective(
    specification: LinearQuadraticControlProblem,
    states: Array,
    controls: Array,
    /,
) -> Array:
    stages = states[..., :-1, :]
    state_quadratic = 0.5 * oe.contract(
        "...ti,...tij,...tj->...t",
        stages,
        specification.state_costs,
        stages,
    )
    control_quadratic = 0.5 * oe.contract(
        "...ti,...tij,...tj->...t",
        controls,
        specification.control_costs,
        controls,
    )
    cross = oe.contract(
        "...ti,...tij,...tj->...t",
        stages,
        specification.state_control_cross,
        controls,
    )
    stage_linear = oe.contract(
        "...ti,...ti->...t", specification.state_linear, stages
    ) + oe.contract("...ti,...ti->...t", specification.control_linear, controls)
    final_state = states[..., -1, :]
    terminal = (
        0.5
        * oe.contract(
            "...i,...ij,...j->...",
            final_state,
            specification.terminal_state_cost,
            final_state,
        )
        + oe.contract("...i,...i->...", specification.terminal_linear, final_state)
        + specification.terminal_constant
    )
    return (
        jnp.sum(
            state_quadratic
            + control_quadratic
            + cross
            + stage_linear
            + specification.stage_constants,
            axis=-1,
        )
        + terminal
    )


def solve_receding_horizon_mpc(
    specification: LinearQuadraticControlProblem,
    /,
    *,
    prediction_horizon: int,
    terminal_policy: MPCTerminalPolicy,
    initial_state: ArrayLike | None = None,
    warm_start: ArrayLike | None = None,
    method: QPMethod = "dense-primal-dual",
    tolerance: float = 1e-7,
    cost_tolerance: float = 1e-10,
    max_iterations: int = 100,
    regularization: float = 0.0,
    step_fraction: float = 0.995,
    max_dense_dimension: int = 512,
    controller_id: str = "control:mpc:receding-horizon",
) -> RecedingHorizonMPCResult:
    """Configure and execute receding-horizon MPC over the full time grid."""
    controller = RecedingHorizonMPC(
        specification,
        prediction_horizon=prediction_horizon,
        terminal_policy=terminal_policy,
        method=method,
        tolerance=tolerance,
        cost_tolerance=cost_tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
        max_dense_dimension=max_dense_dimension,
        controller_id=controller_id,
    )
    return controller.solve(initial_state=initial_state, warm_start=warm_start)


__all__ = [
    "MPCTerminalPolicy",
    "RecedingHorizonMPC",
    "RecedingHorizonMPCResult",
    "solve_receding_horizon_mpc",
]
