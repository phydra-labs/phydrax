#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Receding-horizon MPC for canonical finite linear-control QPs."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field
from ..dynamics import TimeGrid
from ..linalg import prepare_linearization, PreparedLinearization
from ..optim._programming import (
    ClarabelInteriorPoint,
    ConvexDifferentiationPolicy,
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexWarmStart,
    DensePrimalDualQP,
    MPAXraPDHG,
    prepare_qp_sensitivity,
    PreparedQPSensitivity,
    QuadraticProgram,
)
from ._parameterization import PiecewiseConstantControlParameterization
from ._problem import _identifier
from ._qp_compiler import (
    _rebind_dense_control_program,
    LinearControlCompilationPolicy,
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
    parameters: Array = fixed_field()
    subproblem_solutions: tuple[LinearControlQPSolution, ...]
    qp_results: tuple[ConvexProgramResult, ...] = fixed_field()
    objective: Array = fixed_field()
    stage_valid: Array = fixed_field()
    valid: Array = fixed_field()
    status: Array = fixed_field()
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
    compilation_policy: LinearControlCompilationPolicy = eqx.field(static=True)
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
        compilation_policy: LinearControlCompilationPolicy | None = None,
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
        selected_compilation = (
            LinearControlCompilationPolicy()
            if compilation_policy is None
            else compilation_policy
        )
        if not isinstance(selected_compilation, LinearControlCompilationPolicy):
            raise TypeError(
                "compilation_policy must be a LinearControlCompilationPolicy or None."
            )
        selected_policy = (
            ConvexSolvePolicy(ClarabelInteriorPoint())
            if policy is None and selected_compilation.representation == "sparse"
            else ConvexSolvePolicy()
            if policy is None
            else policy
        )
        if not isinstance(selected_policy, ConvexSolvePolicy):
            raise TypeError("policy must be a ConvexSolvePolicy or None.")
        if warm_start_policy is not None and not isinstance(
            warm_start_policy, MPCWarmStartPolicy
        ):
            raise TypeError("warm_start_policy must be an MPCWarmStartPolicy or None.")
        if (
            warm_start_policy is not None
            and not selected_policy.method.capabilities.warm_start
        ):
            raise ValueError(
                f"Method {selected_policy.method.method_id!r} does not support MPC warm starts."
            )
        self.compilation_policy = selected_compilation
        self.qp_policy = selected_policy
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
        if (
            warm_start is not None
            and warm_start.compilation.specification.specification_id
            != specification.specification_id
        ):
            raise ValueError(
                "warm_start specification does not match this MPC controller."
            )
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
                f"initial_state must have shape {expected_initial}; got {current_state.shape}."
            )

        applied_controls: list[Array] = []
        subproblem_solutions: list[LinearControlQPSolution] = []
        online_states = [current_state]
        prepared_by_topology: dict[tuple[int, bool], PreparedLinearControlQP] = {}
        previous_solution = warm_start
        for stage in range(specification.horizon):
            topology = self._window(stage)
            local_problem = self._subproblem(stage, current_state)
            if topology in prepared_by_topology:
                prepared = refresh_linear_quadratic_control(
                    prepared_by_topology[topology],
                    local_problem,
                    cost_tolerance=self.cost_tolerance,
                    compilation_policy=self.compilation_policy,
                )
            else:
                prepared = prepare_linear_quadratic_control(
                    local_problem,
                    policy=self.qp_policy,
                    cost_tolerance=self.cost_tolerance,
                    compilation_policy=self.compilation_policy,
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
            next_state = _handoff(specification, stage, current_state, applied_control)
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

        run_id = "control-mpc-result:" + canonical_fingerprint(
            {
                "controller": self.controller_id,
                "specification": specification.specification_id,
                "prediction_horizon": self.prediction_horizon,
                "terminal_policy": self.terminal_policy,
                "method": self.qp_policy.method.method_id,
                "subproblems": [
                    solution.solution_id for solution in subproblem_solutions
                ],
                "realized": array_tree_fingerprint(
                    {
                        "states": states,
                        "controls": controls,
                        "stage_valid": stage_valid,
                        "status": status,
                    }
                ),
            }
        )
        policy_id = f"{run_id}:policy"
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
            result_id=run_id,
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
            current = _handoff(problem, stage, current, controls[..., stage, :])
            states.append(current)
        primal = compilation.decision_layout.encode(
            jnp.stack(tuple(states), axis=-2),
            controls,
        )
        qp = compilation.program
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
            - ein.contract(
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

    def _window(self, stage: int, /) -> tuple[int, bool]:
        """Return the local horizon and terminal flag of the window at ``stage``."""
        horizon = self.specification.horizon
        local_horizon = min(self.prediction_horizon, horizon - stage)
        apply_terminal = self.terminal_policy == "always" or (
            self.terminal_policy == "global" and stage + local_horizon == horizon
        )
        return local_horizon, apply_terminal

    def _subproblem(
        self,
        stage: int,
        initial_state: Array,
        /,
    ) -> LinearQuadraticControlProblem:
        specification = self.specification
        local_horizon, apply_terminal = self._window(stage)
        end = stage + local_horizon
        fields = _window_fields(
            specification, stage, local_horizon, apply_terminal, initial_state
        )
        positional = tuple(fields.pop(name) for name in _WINDOW_POSITIONAL_FIELDS)
        return LinearQuadraticControlProblem(
            *positional,
            **fields,
            time_grid=TimeGrid(
                specification.time_grid.times[stage : end + 1],
                time_id=f"{specification.time_grid.time_id}:mpc:{stage}:{end}",
            ),
            problem_id=(
                f"{specification.problem_id}:mpc-window:{local_horizon}:terminal-{int(apply_terminal)}"
            ),
            dynamics_id=specification.dynamics_id,
        )


_WINDOW_POSITIONAL_FIELDS = (
    "dynamics_matrices",
    "control_matrices",
    "initial_state",
    "state_costs",
    "control_costs",
    "terminal_state_cost",
)

_PROBLEM_FIELDS = _WINDOW_POSITIONAL_FIELDS + (
    "dynamics_bias",
    "state_control_cross",
    "state_linear",
    "control_linear",
    "stage_constants",
    "terminal_linear",
    "terminal_constant",
    "state_lower_bounds",
    "state_upper_bounds",
    "control_lower_bounds",
    "control_upper_bounds",
    "stage_equality_state_matrix",
    "stage_equality_control_matrix",
    "stage_equality_rhs",
    "stage_inequality_state_matrix",
    "stage_inequality_control_matrix",
    "stage_inequality_rhs",
    "terminal_equality_matrix",
    "terminal_equality_rhs",
    "terminal_inequality_matrix",
    "terminal_inequality_rhs",
)


def _numeric_fields(problem: LinearQuadraticControlProblem, /) -> dict[str, Array]:
    """Present numeric coefficient arrays; the time grid is metadata."""
    return {
        name: value
        for name in _PROBLEM_FIELDS
        if (value := getattr(problem, name)) is not None
    }


def _with_fields(
    problem: LinearQuadraticControlProblem,
    fields: dict[str, Array | None],
    /,
) -> LinearQuadraticControlProblem:
    """Rebind numeric leaves without reconstructing (and re-admitting) a problem.

    Every static field and the None pattern stay those of ``problem``, so the
    rebinding is traceable; admission already happened for ``problem``.
    """
    names = tuple(name for name, value in fields.items() if value is not None)
    return eqx.tree_at(
        lambda item: tuple(getattr(item, name) for name in names),
        problem,
        tuple(fields[name] for name in names),
    )


def _window_fields(
    specification: LinearQuadraticControlProblem,
    stage: int,
    local_horizon: int,
    apply_terminal: bool,
    initial_state: Array,
    /,
) -> dict[str, Array | None]:
    """Slice every numeric field of one prediction window.

    One pure slicing map drives both the audited window problems and the traced
    closed-loop sensitivity, so both see identical window data.
    """
    stages = slice(stage, stage + local_horizon)
    nodes = slice(stage, stage + local_horizon + 1)
    batch = specification.case_shape
    dtype = specification.dynamics_matrices.dtype
    state_size = specification.state_size

    def stage_matrix(value: Array | None) -> Array | None:
        return None if value is None else value[..., stages, :, :]

    def stage_vector(value: Array | None) -> Array | None:
        return None if value is None else value[..., stages, :]

    def node_vector(value: Array | None) -> Array | None:
        return None if value is None else value[..., nodes, :]

    def terminal(value: Array | None, shape: tuple[int, ...] | None) -> Array | None:
        if apply_terminal:
            return value
        return None if shape is None else jnp.zeros(batch + shape, dtype=dtype)

    return {
        "dynamics_matrices": stage_matrix(specification.dynamics_matrices),
        "control_matrices": stage_matrix(specification.control_matrices),
        "initial_state": initial_state,
        "state_costs": stage_matrix(specification.state_costs),
        "control_costs": stage_matrix(specification.control_costs),
        "terminal_state_cost": terminal(
            specification.terminal_state_cost, (state_size, state_size)
        ),
        "dynamics_bias": stage_vector(specification.dynamics_bias),
        "state_control_cross": stage_matrix(specification.state_control_cross),
        "state_linear": stage_vector(specification.state_linear),
        "control_linear": stage_vector(specification.control_linear),
        "stage_constants": specification.stage_constants[..., stages],
        "terminal_linear": terminal(specification.terminal_linear, (state_size,)),
        "terminal_constant": terminal(specification.terminal_constant, ()),
        "state_lower_bounds": node_vector(specification.state_lower_bounds),
        "state_upper_bounds": node_vector(specification.state_upper_bounds),
        "control_lower_bounds": stage_vector(specification.control_lower_bounds),
        "control_upper_bounds": stage_vector(specification.control_upper_bounds),
        "stage_equality_state_matrix": stage_matrix(
            specification.stage_equality_state_matrix
        ),
        "stage_equality_control_matrix": stage_matrix(
            specification.stage_equality_control_matrix
        ),
        "stage_equality_rhs": stage_vector(specification.stage_equality_rhs),
        "stage_inequality_state_matrix": stage_matrix(
            specification.stage_inequality_state_matrix
        ),
        "stage_inequality_control_matrix": stage_matrix(
            specification.stage_inequality_control_matrix
        ),
        "stage_inequality_rhs": stage_vector(specification.stage_inequality_rhs),
        "terminal_equality_matrix": terminal(
            specification.terminal_equality_matrix, None
        ),
        "terminal_equality_rhs": terminal(specification.terminal_equality_rhs, None),
        "terminal_inequality_matrix": terminal(
            specification.terminal_inequality_matrix, None
        ),
        "terminal_inequality_rhs": terminal(specification.terminal_inequality_rhs, None),
    }


def _handoff(
    specification: LinearQuadraticControlProblem,
    stage: int,
    state: Array,
    control: Array,
    /,
) -> Array:
    """Exact affine state handoff ``A[t] x + B[t] u + c[t]``."""
    return (
        ein.contract(
            "...ij,...j->...i",
            specification.dynamics_matrices[..., stage, :, :],
            state,
        )
        + ein.contract(
            "...ij,...j->...i",
            specification.control_matrices[..., stage, :, :],
            control,
        )
        + specification.dynamics_bias[..., stage, :]
    )


def _realized_objective(
    specification: LinearQuadraticControlProblem,
    states: Array,
    controls: Array,
    /,
) -> Array:
    stages = states[..., :-1, :]
    state_quadratic = 0.5 * ein.contract(
        "...ti,...tij,...tj->...t",
        stages,
        specification.state_costs,
        stages,
    )
    control_quadratic = 0.5 * ein.contract(
        "...ti,...tij,...tj->...t",
        controls,
        specification.control_costs,
        controls,
    )
    cross = ein.contract(
        "...ti,...tij,...tj->...t",
        stages,
        specification.state_control_cross,
        controls,
    )
    stage_linear = ein.contract(
        "...ti,...ti->...t", specification.state_linear, stages
    ) + ein.contract("...ti,...ti->...t", specification.control_linear, controls)
    final_state = states[..., -1, :]
    terminal = (
        0.5
        * ein.contract(
            "...i,...ij,...j->...",
            final_state,
            specification.terminal_state_cost,
            final_state,
        )
        + ein.contract("...i,...i->...", specification.terminal_linear, final_state)
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


class PreparedMPCSensitivity(StrictModule):
    """Closed-loop MPC sensitivity composed from per-window dense QP sensitivities.

    The primal is the audited `RecedingHorizonMPC.solve` result with
    cold-started windows. Each window contributes one `PreparedQPSensitivity`
    at its own QP; the exact affine state handoffs compose those window maps
    into one linear map from a `LinearQuadraticControlProblem`-shaped tangent
    to the handed-off states and applied controls. Window QP data is rebound
    through the window's compiled static layout, so tangents of the initial
    state, dynamics, costs, bounds, and constraints all propagate.

    The complete derivative is refused unless every window is valid, OPTIMAL,
    and regular (strictly complementary with a nonsingular reduced KKT
    system). No partial or stage-truncated derivative is returned.
    """

    specification: LinearQuadraticControlProblem
    result: RecedingHorizonMPCResult
    window_sensitivities: tuple[PreparedQPSensitivity, ...]
    linearization: PreparedLinearization | None
    stage_optimal: Array = fixed_field()
    stage_regular: Array = fixed_field()
    regular: Array = fixed_field()
    differentiation: ConvexDifferentiationPolicy = eqx.field(static=True)
    refusal: str | None = eqx.field(static=True)
    sensitivity_id: str = eqx.field(static=True)

    @property
    def states(self) -> Array:
        return self.result.states

    @property
    def controls(self) -> Array:
        return self.result.controls

    def jvp(self, tangent: LinearQuadraticControlProblem, /) -> tuple[Array, Array]:
        """Push a specification tangent to state and applied-control tangents.

        ``tangent`` has the structure of the specification; its time grid
        carries no derivative.
        """
        linearization = self._require_linearization()
        if not isinstance(tangent, LinearQuadraticControlProblem):
            raise TypeError("tangent must be LinearQuadraticControlProblem-shaped.")
        return linearization.jvp(_numeric_fields(tangent))

    def vjp(
        self,
        states_cotangent: ArrayLike,
        controls_cotangent: ArrayLike,
        /,
    ) -> LinearQuadraticControlProblem:
        """Pull state and applied-control cotangents back to the specification."""
        linearization = self._require_linearization()
        cotangent = linearization.vjp(
            (
                jnp.asarray(states_cotangent, dtype=self.states.dtype),
                jnp.asarray(controls_cotangent, dtype=self.controls.dtype),
            )
        )
        return _with_fields(jax.tree.map(jnp.zeros_like, self.specification), cotangent)

    def _require_linearization(self) -> PreparedLinearization:
        if self.linearization is None:
            raise ValueError(self.refusal)
        return self.linearization


def _sensitivity_differentiation(
    controller: RecedingHorizonMPC,
    differentiation: ConvexDifferentiationPolicy | None,
    /,
) -> ConvexDifferentiationPolicy:
    if controller.compilation_policy.representation != "dense":
        raise ValueError(
            "MPC sensitivity is dense-only; sparse control compilations have no "
            "QP sensitivity."
        )
    if controller.warm_start_policy is not None:
        raise ValueError(
            "MPC sensitivity has no warm-start derivative; configure the "
            "controller without a warm_start_policy."
        )
    policy = controller.qp_policy
    if policy.regularization != 0.0:
        raise ValueError("MPC sensitivity requires zero solver regularization.")
    if differentiation is not None and not isinstance(
        differentiation, ConvexDifferentiationPolicy
    ):
        raise TypeError("differentiation must be a ConvexDifferentiationPolicy or None.")
    method = policy.method
    match method:
        case DensePrimalDualQP():
            selected = (
                ConvexDifferentiationPolicy()
                if differentiation is None
                else differentiation
            )
            if selected.mode not in ("active-set-kkt", "barrier-kkt"):
                raise ValueError(
                    "DensePrimalDualQP MPC sensitivity requires active-set-kkt or "
                    "barrier-kkt differentiation."
                )
        case MPAXraPDHG():
            if not method.plan.unroll or method.plan.representation != "dense":
                raise ValueError(
                    "MPAXraPDHG MPC sensitivity requires a dense unrolled plan."
                )
            selected = (
                ConvexDifferentiationPolicy("algorithmic")
                if differentiation is None
                else differentiation
            )
            if selected.mode != "algorithmic":
                raise ValueError(
                    "MPAXraPDHG MPC sensitivity requires algorithmic differentiation."
                )
        case _:
            raise ValueError(
                f"Method {method.method_id!r} has no dense QP sensitivity; use "
                "DensePrimalDualQP or MPAXraPDHG(unroll=True)."
            )
    return selected


def _stored_window_solution(
    primal: Array,
    sensitivity: PreparedQPSensitivity,
    /,
) -> Callable[[QuadraticProgram], Array]:
    """Window solution map at its audited primal with the prepared QP tangent."""

    @jax.custom_jvp
    def solution(program: QuadraticProgram) -> Array:
        del program
        return primal

    @solution.defjvp
    def solution_jvp(primals, tangents):
        del primals
        (tangent,) = tangents
        return primal, sensitivity.jvp(tangent)

    return solution


def _closed_loop_map(
    controller: RecedingHorizonMPC,
    result: RecedingHorizonMPCResult,
    sensitivities: tuple[PreparedQPSensitivity, ...],
    /,
) -> Callable[[dict[str, Array]], tuple[Array, Array]]:
    """Closed loop as a function of the specification's numeric fields.

    Window QPs are rebound through each window's compiled static layout; the
    window solve is the audited primal with its prepared QP tangent, and the
    state handoffs are the same exact affine map as `RecedingHorizonMPC.solve`.
    """
    specification = controller.specification
    windows = tuple(
        (
            stage,
            *controller._window(stage),
            solution.compilation,
            _stored_window_solution(solution.qp_result.primal, sensitivity),
        )
        for stage, (solution, sensitivity) in enumerate(
            zip(result.subproblem_solutions, sensitivities, strict=True)
        )
    )
    case_axis = len(specification.case_shape)

    def closed_loop(fields: dict[str, Array]) -> tuple[Array, Array]:
        problem = _with_fields(specification, fields)
        state = problem.initial_state
        states = [state]
        controls = []
        for stage, local_horizon, apply_terminal, compilation, solution in windows:
            window = _with_fields(
                compilation.specification,
                _window_fields(problem, stage, local_horizon, apply_terminal, state),
            )
            primal = solution(_rebind_dense_control_program(compilation, window))
            _, window_controls = compilation.decode(primal)
            control = window_controls[..., 0, :]
            state = _handoff(problem, stage, state, control)
            controls.append(control)
            states.append(state)
        return (
            jnp.stack(states, axis=case_axis),
            jnp.stack(controls, axis=case_axis),
        )

    return closed_loop


def _sensitivity_refusal(stage_optimal: Array, stage_regular: Array, /) -> str | None:
    # Host evidence boundary: the prepared sensitivity is an eager artifact
    # whose admission is decided once, after every window has been solved.
    case_axes = tuple(range(stage_optimal.ndim - 1))
    optimal = np.asarray(jnp.all(stage_optimal, axis=case_axes))
    regular = np.asarray(stage_regular)
    reasons = []
    if not optimal.all():
        reasons.append(
            f"windows {np.flatnonzero(~optimal).tolist()} are not valid and OPTIMAL"
        )
    if not regular.all():
        reasons.append(
            f"windows {np.flatnonzero(~regular).tolist()} have nonregular QP "
            "sensitivities (weak complementarity or a singular reduced KKT system)"
        )
    if not reasons:
        return None
    return "MPC sensitivity is refused: " + "; ".join(reasons) + "."


def prepare_receding_horizon_mpc_sensitivity(
    controller: RecedingHorizonMPC,
    /,
    *,
    differentiation: ConvexDifferentiationPolicy | None = None,
) -> PreparedMPCSensitivity:
    """Solve cold-started MPC windows and prepare the composed dense sensitivity.

    Only dense affine-quadratic controllers without warm starts or solver
    regularization are admitted. ``DensePrimalDualQP`` uses implicit
    active-set or barrier KKT differentiation; ``MPAXraPDHG(unroll=True)``
    differentiates its unrolled iterations.
    """
    if not isinstance(controller, RecedingHorizonMPC):
        raise TypeError("controller must be a RecedingHorizonMPC.")
    derivative = _sensitivity_differentiation(controller, differentiation)
    result = controller.solve()
    sensitivities = tuple(
        prepare_qp_sensitivity(
            solution.compilation.program,
            policy=controller.qp_policy,
            differentiation=derivative,
        )
        for solution in result.subproblem_solutions
    )
    case_axis = len(controller.specification.case_shape)
    statuses = jnp.stack(
        tuple(qp_result.status for qp_result in result.qp_results), axis=case_axis
    )
    stage_optimal = result.stage_valid & (statuses == int(ConvexProgramStatus.OPTIMAL))
    stage_regular = jnp.stack(tuple(item.regular for item in sensitivities))
    refusal = _sensitivity_refusal(stage_optimal, stage_regular)
    sensitivity_id = "control-mpc-sensitivity:" + canonical_fingerprint(
        {
            "result": result.result_id,
            "method": controller.qp_policy.method.method_id,
            "differentiation": derivative.mode,
        }
    )
    linearization = (
        None
        if refusal is not None
        else prepare_linearization(
            _closed_loop_map(controller, result, sensitivities),
            _numeric_fields(controller.specification),
            linearization_id=sensitivity_id,
        )
    )
    return PreparedMPCSensitivity(
        specification=controller.specification,
        result=result,
        window_sensitivities=sensitivities,
        linearization=linearization,
        stage_optimal=stage_optimal,
        stage_regular=stage_regular,
        regular=jnp.all(stage_optimal) & jnp.all(stage_regular),
        differentiation=derivative,
        refusal=refusal,
        sensitivity_id=sensitivity_id,
    )


__all__ = [
    "MPCTerminalPolicy",
    "MPCWarmStartPolicy",
    "PreparedMPCSensitivity",
    "RecedingHorizonMPC",
    "RecedingHorizonMPCResult",
    "prepare_receding_horizon_mpc_sensitivity",
    "solve_receding_horizon_mpc",
]
