#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Receding-horizon MPC for canonical finite linear-control QPs."""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ..optim._programming import (
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexWarmStart,
)
from ._parameterization import PiecewiseConstantControlParameterization
from ._problem import _identifier
from ._qp_compiler import (
    LinearControlQPSolution,
    LinearQuadraticControlProblem,
    prepare_linear_quadratic_control,
    PreparedLinearControlQP,
    refresh_linear_quadratic_control,
    solve_prepared_linear_quadratic_control,
)
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_INFEASIBLE,
    CONTROL_SUCCESS,
    ControlTrajectory,
)


MPCTerminalPolicy: TypeAlias = Literal["global", "always", "none"]


class MPCWarmStartPolicy(StrictModule):
    """Explicit primal/dual shift and interiorization policy between MPC windows."""

    terminal_control: Literal["hold", "zero"] = eqx.field(static=True)
    interior_margin: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        terminal_control: Literal["hold", "zero"] = "hold",
        interior_margin: float = 1e-7,
    ):
        if terminal_control not in ("hold", "zero"):
            raise ValueError("terminal_control must be 'hold' or 'zero'.")
        margin = float(interior_margin)
        if not isfinite(margin) or margin <= 0.0:
            raise ValueError("interior_margin must be finite and positive.")
        self.terminal_control = terminal_control
        self.interior_margin = margin


class RecedingHorizonMPCResult(StrictModule):
    """Applied MPC rollout and every local QP result, without hidden repair."""

    trajectory: ControlTrajectory
    policy: PiecewiseConstantControlParameterization
    parameters: Array
    subproblem_solutions: tuple[LinearControlQPSolution, ...]
    qp_results: tuple[ConvexProgramResult, ...]
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
        return self.valid & (self.status == int(ConvexProgramStatus.OPTIMAL))


class RecedingHorizonMPC(StrictModule):
    """A configured linear MPC controller with an explicit terminal policy.

    ``terminal_policy="global"`` applies terminal cost and terminal constraints
    only when a prediction window reaches the specification's final node.
    ``"always"`` applies them at every prediction endpoint, and ``"none"``
    omits them. Warm starts are enabled only through an explicit
    `MPCWarmStartPolicy` and a selected QP method that declares support.
    """

    specification: LinearQuadraticControlProblem
    prediction_horizon: int = eqx.field(static=True)
    terminal_policy: MPCTerminalPolicy = eqx.field(static=True)
    qp_policy: ConvexSolvePolicy
    warm_start_policy: MPCWarmStartPolicy | None
    cost_tolerance: float = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)

    def __init__(
        self,
        specification: LinearQuadraticControlProblem,
        /,
        *,
        prediction_horizon: int,
        terminal_policy: MPCTerminalPolicy,
        cost_tolerance: float = 1e-10,
        policy: ConvexSolvePolicy | None = None,
        warm_start_policy: MPCWarmStartPolicy | None = None,
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
        policy = ConvexSolvePolicy() if policy is None else policy
        if not isinstance(policy, ConvexSolvePolicy):
            raise TypeError("policy must be a ConvexSolvePolicy or None.")
        if warm_start_policy is not None and not isinstance(
            warm_start_policy, MPCWarmStartPolicy
        ):
            raise TypeError("warm_start_policy must be an MPCWarmStartPolicy or None.")
        if warm_start_policy is not None and not policy.method.capabilities.warm_start:
            raise ValueError(
                f"Method {policy.method.method_id!r} does not support MPC warm starts."
            )
        self.qp_policy = policy
        self.warm_start_policy = warm_start_policy
        self.cost_tolerance = float(cost_tolerance)
        self.controller_id = _identifier(controller_id, "controller_id")

    def solve(
        self,
        /,
        *,
        initial_state: ArrayLike | None = None,
        warm_start: LinearControlQPSolution | None = None,
    ) -> RecedingHorizonMPCResult:
        """Solve each local QP, hand off the exact state, and roll out controls."""
        if warm_start is not None and not isinstance(warm_start, LinearControlQPSolution):
            raise TypeError("warm_start must be a LinearControlQPSolution or None.")
        if warm_start is not None and self.warm_start_policy is None:
            raise ValueError("warm_start requires an explicit MPCWarmStartPolicy.")
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
        prepared_by_topology: dict[tuple[int, bool], PreparedLinearControlQP] = {}
        previous_solution = warm_start
        for stage in range(specification.horizon):
            local_horizon = min(self.prediction_horizon, specification.horizon - stage)
            end = stage + local_horizon
            apply_terminal = self.terminal_policy == "always" or (
                self.terminal_policy == "global" and end == specification.horizon
            )
            topology = (local_horizon, apply_terminal)
            local_problem = self._subproblem(stage, local_horizon, current_state)
            if topology in prepared_by_topology:
                prepared = refresh_linear_quadratic_control(
                    prepared_by_topology[topology],
                    local_problem,
                    cost_tolerance=self.cost_tolerance,
                )
            else:
                prepared = prepare_linear_quadratic_control(
                    local_problem,
                    policy=self.qp_policy,
                    cost_tolerance=self.cost_tolerance,
                )
            prepared_by_topology[topology] = prepared
            convex_warm = (
                None
                if self.warm_start_policy is None or previous_solution is None
                else self._shift_warm_start(
                    previous_solution,
                    local_problem,
                    prepared.compilation,
                )
            )
            local_solution = solve_prepared_linear_quadratic_control(
                prepared,
                warm_start=convex_warm,
            )
            previous_solution = local_solution
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
                int(ConvexProgramStatus.NONFINITE_OUTPUT),
                result.status,
            ).astype(jnp.int32)
            status = jnp.where(
                (status == int(ConvexProgramStatus.OPTIMAL))
                & (local_status != int(ConvexProgramStatus.OPTIMAL)),
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
            (status == int(ConvexProgramStatus.OPTIMAL)) & ~valid,
            int(ConvexProgramStatus.NONFINITE_OUTPUT),
            status,
        ).astype(jnp.int32)

        policy_id = f"{self.controller_id}:policy"
        policy = PiecewiseConstantControlParameterization(
            specification.time_grid,
            (specification.control_size,),
            parameterization_id=policy_id,
        )
        control_status = jnp.where(
            status == int(ConvexProgramStatus.OPTIMAL),
            CONTROL_SUCCESS,
            jnp.where(
                status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE),
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
            method_id=f"control:mpc:{self.qp_policy.method.method_id}",
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
            method_id=f"control:mpc:{self.qp_policy.method.method_id}",
        )

    __call__ = solve

    def _shift_warm_start(
        self,
        previous: LinearControlQPSolution,
        problem: LinearQuadraticControlProblem,
        compilation,
        /,
    ) -> ConvexWarmStart:
        policy = self.warm_start_policy
        if policy is None:
            raise RuntimeError("Warm-start shifting requires MPCWarmStartPolicy.")
        horizon = problem.horizon
        dtype = problem.initial_state.dtype
        previous_controls = previous.controls
        shifted_controls = [
            previous_controls[..., stage, :]
            for stage in range(1, min(previous_controls.shape[-2], horizon + 1))
        ]
        fill_control = (
            jnp.zeros(problem.case_shape + (problem.control_size,), dtype=dtype)
            if policy.terminal_control == "zero"
            else previous_controls[..., -1, :]
        )
        while len(shifted_controls) < horizon:
            shifted_controls.append(fill_control)
        controls = jnp.stack(tuple(shifted_controls[:horizon]), axis=-2)

        states = [problem.initial_state]
        current = problem.initial_state
        for stage in range(horizon):
            current = (
                oe.contract(
                    "...ij,...j->...i",
                    problem.dynamics_matrices[..., stage, :, :],
                    current,
                )
                + oe.contract(
                    "...ij,...j->...i",
                    problem.control_matrices[..., stage, :, :],
                    controls[..., stage, :],
                )
                + problem.dynamics_bias[..., stage, :]
            )
            states.append(current)
        primal = compilation.decision_layout.encode(
            jnp.stack(tuple(states), axis=-2),
            controls,
        )
        qp = compilation.quadratic_program
        margin = jnp.asarray(policy.interior_margin, dtype=dtype)
        lower_finite = jnp.isfinite(qp.lower_bounds)
        upper_finite = jnp.isfinite(qp.upper_bounds)
        fixed = lower_finite & upper_finite & (qp.lower_bounds == qp.upper_bounds)
        narrow = (
            lower_finite
            & upper_finite
            & ((qp.upper_bounds - qp.lower_bounds) <= 2.0 * margin)
        )
        primal = jnp.where(
            lower_finite,
            jnp.maximum(primal, qp.lower_bounds + margin),
            primal,
        )
        primal = jnp.where(
            upper_finite,
            jnp.minimum(primal, qp.upper_bounds - margin),
            primal,
        )
        primal = jnp.where(
            narrow,
            0.5 * (qp.lower_bounds + qp.upper_bounds),
            primal,
        )
        primal = jnp.where(fixed, qp.lower_bounds, primal)

        old_compilation = previous.compilation
        old_constraints = old_compilation.constraint_layout
        new_constraints = compilation.constraint_layout
        old_result = previous.qp_result
        equality_dual = jnp.zeros(
            problem.case_shape + (qp.num_user_equalities,), dtype=dtype
        )
        for stage, target in enumerate(new_constraints.dynamics_slices):
            source_stage = stage + 1
            if source_stage < len(old_constraints.dynamics_slices):
                equality_dual = equality_dual.at[..., target].set(
                    old_result.equality_dual[
                        ..., old_constraints.dynamics_slices[source_stage]
                    ]
                )
        for stage, target in enumerate(new_constraints.stage_equality_slices):
            source_stage = stage + 1
            if source_stage < len(old_constraints.stage_equality_slices):
                equality_dual = equality_dual.at[..., target].set(
                    old_result.equality_dual[
                        ..., old_constraints.stage_equality_slices[source_stage]
                    ]
                )
        if (
            new_constraints.terminal_equality_slice is not None
            and old_constraints.terminal_equality_slice is not None
        ):
            equality_dual = equality_dual.at[
                ..., new_constraints.terminal_equality_slice
            ].set(old_result.equality_dual[..., old_constraints.terminal_equality_slice])

        inequality_dual = jnp.full(
            problem.case_shape + (qp.num_user_inequalities,),
            margin,
            dtype=dtype,
        )
        for stage, target in enumerate(new_constraints.stage_inequality_slices):
            source_stage = stage + 1
            if source_stage < len(old_constraints.stage_inequality_slices):
                inequality_dual = inequality_dual.at[..., target].set(
                    jnp.maximum(
                        old_result.inequality_dual[
                            ..., old_constraints.stage_inequality_slices[source_stage]
                        ],
                        margin,
                    )
                )
        if (
            new_constraints.terminal_inequality_slice is not None
            and old_constraints.terminal_inequality_slice is not None
        ):
            inequality_dual = inequality_dual.at[
                ..., new_constraints.terminal_inequality_slice
            ].set(
                jnp.maximum(
                    old_result.inequality_dual[
                        ..., old_constraints.terminal_inequality_slice
                    ],
                    margin,
                )
            )
        inequality_slack = jnp.maximum(
            qp.inequality_rhs[..., : qp.num_user_inequalities]
            - oe.contract(
                "...ij,...j->...i",
                qp.inequality_matrix[..., : qp.num_user_inequalities, :],
                primal,
            ),
            margin,
        )

        def shift_bound_dual(values):
            old_states, old_controls = old_compilation.decision_layout.decode(values)
            state_values = [
                old_states[..., stage, :]
                for stage in range(1, min(old_states.shape[-2], horizon + 2))
            ]
            while len(state_values) < horizon + 1:
                state_values.append(
                    jnp.full(
                        problem.case_shape + (problem.state_size,), margin, dtype=dtype
                    )
                )
            control_values = [
                old_controls[..., stage, :]
                for stage in range(1, min(old_controls.shape[-2], horizon + 1))
            ]
            while len(control_values) < horizon:
                control_values.append(
                    jnp.full(
                        problem.case_shape + (problem.control_size,),
                        margin,
                        dtype=dtype,
                    )
                )
            return compilation.decision_layout.encode(
                jnp.stack(tuple(state_values[: horizon + 1]), axis=-2),
                jnp.stack(tuple(control_values[:horizon]), axis=-2),
            )

        lower_bound_dual = jnp.maximum(
            shift_bound_dual(old_result.lower_bound_dual), margin
        )
        upper_bound_dual = jnp.maximum(
            shift_bound_dual(old_result.upper_bound_dual), margin
        )
        return ConvexWarmStart(
            primal=primal,
            equality_dual=equality_dual,
            inequality_dual=inequality_dual,
            inequality_slack=inequality_slack,
            lower_bound_dual=lower_bound_dual,
            upper_bound_dual=upper_bound_dual,
            structure_id=qp.structure_id,
        )

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
            problem_id=(
                f"{specification.problem_id}:mpc-window:"
                f"{local_horizon}:terminal-{int(apply_terminal)}"
            ),
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
    warm_start: LinearControlQPSolution | None = None,
    cost_tolerance: float = 1e-10,
    policy: ConvexSolvePolicy | None = None,
    warm_start_policy: MPCWarmStartPolicy | None = None,
    controller_id: str = "control:mpc:receding-horizon",
) -> RecedingHorizonMPCResult:
    """Configure and execute receding-horizon MPC over the full time grid."""
    controller = RecedingHorizonMPC(
        specification,
        prediction_horizon=prediction_horizon,
        terminal_policy=terminal_policy,
        cost_tolerance=cost_tolerance,
        policy=policy,
        warm_start_policy=warm_start_policy,
        controller_id=controller_id,
    )
    return controller.solve(initial_state=initial_state, warm_start=warm_start)


__all__ = [
    "MPCTerminalPolicy",
    "MPCWarmStartPolicy",
    "RecedingHorizonMPC",
    "RecedingHorizonMPCResult",
    "solve_receding_horizon_mpc",
]
