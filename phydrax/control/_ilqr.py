#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Iterative LQR for a single finite-horizon control case."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..dynamics import DiscreteStepContext, StateLayout, TimeGrid
from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics
from ._parameterization import AbstractControlParameterization
from ._problem import _identifier, ControlProblem
from ._trajectory import ControlResult, ControlTrajectory


DifferentialFlowStep: TypeAlias = Callable[[Array, Array, Array, Array, Any], ArrayLike]
ILQRFlow: TypeAlias = Callable[[Array, Array, Array, Array, Array], Array]


class ILQRStatus(IntEnum):
    """Stable termination codes for :func:`solve_ilqr`."""

    SUCCESS = 0
    MAX_ITERATIONS = 1
    INITIAL_ROLLOUT_FAILED = 2
    BACKWARD_PASS_NOT_POSITIVE_DEFINITE = 3
    LINE_SEARCH_FAILED = 4


class DifferentialControlFlow(StrictModule):
    """Explicit one-step flow selected for differential-dynamics iLQR.

    ``step(t0, t1, state, control, args)`` must return the state at ``t1``
    under the interval's held control. Nonfinite returned states are reported as
    failed integration; they are never repaired or replaced by another method.
    """

    step: DifferentialFlowStep
    flow_id: str = eqx.field(static=True)

    def __init__(self, step: DifferentialFlowStep, /, *, flow_id: str):
        if not callable(step):
            raise TypeError("DifferentialControlFlow step must be callable.")
        self.step = step
        self.flow_id = _identifier(flow_id, "DifferentialControlFlow flow_id")

    def __call__(
        self,
        t0: Array,
        t1: Array,
        state: Array,
        control: Array,
        args: Any,
        /,
    ) -> Array:
        return jnp.asarray(self.step(t0, t1, state, control, args))


class ILQRPolicy(AbstractControlParameterization):
    """Time-indexed affine feedback around an iLQR nominal trajectory.

    The policy has no free coefficients: pass an empty array to ``evaluate`` or
    ``ControlProblem.rollout``. Without a state, evaluation returns the nominal
    open-loop controls. With a state it applies feedback to
    ``inverse_retract(state_nominal, state)``. No control clipping is performed.
    """

    time_grid: TimeGrid
    nominal_states: Array
    nominal_controls: Array
    feedback: Array
    state_layout: StateLayout
    state_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        time_grid: TimeGrid,
        nominal_states: ArrayLike,
        nominal_controls: ArrayLike,
        feedback: ArrayLike,
        /,
        *,
        state_layout: StateLayout,
        control_shape: tuple[int, ...],
        policy_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("ILQRPolicy time_grid must be a TimeGrid.")
        states = jnp.asarray(nominal_states)
        controls = jnp.asarray(nominal_controls)
        gains = jnp.asarray(feedback)
        if not isinstance(state_layout, StateLayout):
            raise TypeError("ILQRPolicy state_layout must be a StateLayout.")
        if not state_layout.geometry.supports_exact_inverse:
            raise ValueError("ILQRPolicy requires exact inverse-retraction geometry.")
        state_shape_ = state_layout.shape
        control_shape_ = tuple(int(size) for size in control_shape)
        state_size = state_layout.local_size
        control_size = int(np.prod(control_shape_))
        trailing_states = (time_grid.num_times,) + state_shape_
        if (
            states.ndim < len(trailing_states)
            or tuple(states.shape[-len(trailing_states) :]) != trailing_states
        ):
            raise ValueError(
                "ILQRPolicy nominal_states must end with "
                f"{trailing_states}; got {states.shape}."
            )
        case_shape_ = tuple(states.shape[: -len(trailing_states)])
        expected_controls = case_shape_ + (time_grid.num_steps,) + control_shape_
        expected_feedback = case_shape_ + (
            time_grid.num_steps,
            control_size,
            state_size,
        )
        if tuple(controls.shape) != expected_controls:
            raise ValueError(
                f"ILQRPolicy nominal_controls must have shape {expected_controls}; "
                f"got {controls.shape}."
            )
        if tuple(gains.shape) != expected_feedback:
            raise ValueError(
                f"ILQRPolicy feedback must have shape {expected_feedback}; "
                f"got {gains.shape}."
            )
        self.time_grid = time_grid
        self.nominal_states = states
        self.nominal_controls = controls
        self.feedback = gains
        self.state_layout = state_layout
        self.state_shape = state_shape_
        self.case_shape = case_shape_
        self.control_shape = control_shape_
        self.parameter_shape = (0,)
        self.parameterization_id = _identifier(policy_id, "ILQRPolicy policy_id")
        self.approximation_id = "control:ilqr:affine-feedback"

    @property
    def feedforward(self) -> Array:
        """Affine intercepts, using local state coordinates for feedback."""
        control_size = int(np.prod(self.control_shape))
        controls = self.nominal_controls.reshape(
            self.case_shape + (self.time_grid.num_steps, control_size)
        )
        if not self.state_layout.geometry.trivial:
            return controls.reshape(
                self.case_shape + (self.time_grid.num_steps,) + self.control_shape
            )
        states = jnp.take(
            self.nominal_states,
            jnp.arange(self.time_grid.num_steps),
            axis=len(self.case_shape),
        ).reshape(
            self.case_shape + (self.time_grid.num_steps, self.state_layout.local_size)
        )
        intercept = controls - ein.contract("...tij,...tj->...ti", self.feedback, states)
        return intercept.reshape(
            self.case_shape + (self.time_grid.num_steps,) + self.control_shape
        )

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        cases = tuple(int(size) for size in case_shape)
        if cases != self.case_shape:
            raise ValueError(
                f"ILQRPolicy case_shape must be {self.case_shape}; got {cases}."
            )
        parameters = jnp.asarray(coefficients)
        expected_parameters = self.case_shape + self.parameter_shape
        if tuple(parameters.shape) not in (self.parameter_shape, expected_parameters):
            raise ValueError(
                "ILQRPolicy coefficients must have an empty trailing parameter axis."
            )
        query = jnp.asarray(time)
        if jnp.issubdtype(query.dtype, jnp.complexfloating):
            raise TypeError("ILQRPolicy evaluation times must be real-valued.")
        query = query.astype(jnp.result_type(query, float))
        query = eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query))
            | jnp.any(query < self.time_grid.t0)
            | jnp.any(query > self.time_grid.t1),
            "ILQRPolicy evaluation time lies outside its physical grid.",
        )
        indices = jnp.searchsorted(self.time_grid.times, query, side="right") - 1
        indices = jnp.minimum(indices, self.time_grid.num_steps - 1)
        nominal_controls = jnp.take(
            self.nominal_controls, indices, axis=len(self.case_shape)
        )
        if state is None:
            return nominal_controls

        states = jnp.asarray(state)
        expected_state_shape = self.case_shape + tuple(query.shape) + self.state_shape
        if tuple(states.shape) != expected_state_shape:
            raise ValueError(
                f"ILQRPolicy state must have shape {expected_state_shape}; "
                f"got {states.shape}."
            )
        nominal_nodes = jnp.take(
            self.nominal_states,
            jnp.arange(self.time_grid.num_steps),
            axis=len(self.case_shape),
        )
        nominal_states = jnp.take(
            nominal_nodes,
            indices,
            axis=len(self.case_shape),
        )
        gains = jnp.take(self.feedback, indices, axis=len(self.case_shape))
        local_size = self.state_layout.local_size
        control_size = int(np.prod(self.control_shape))
        prefix = self.case_shape + tuple(query.shape)
        sample_count = int(np.prod(prefix)) if prefix else 1
        flat_states = states.reshape((sample_count,) + self.state_shape)
        flat_nominal = nominal_states.reshape((sample_count,) + self.state_shape)
        flat_delta = jax.vmap(
            lambda nominal, point: jnp.asarray(
                self.state_layout.geometry.inverse_retract(nominal, point)
            ).reshape((local_size,))
        )(flat_nominal, flat_states).reshape(prefix + (local_size,))
        flat_gains = gains.reshape(prefix + (control_size, local_size))
        correction = ein.contract("...ij,...j->...i", flat_gains, flat_delta)
        return nominal_controls + correction.reshape(prefix + self.control_shape)

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)

    def __call__(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        return self.evaluate(
            jnp.empty(self.case_shape + (0,)),
            time,
            case_shape=self.case_shape,
            state=state,
        )


class ILQRDiagnostics(StrictModule):
    """Convergence and failure evidence retained by an iLQR solve."""

    objective_history: Array
    gradient_norm_history: Array
    regularized_minimum_curvature_history: Array
    step_size_history: Array
    expected_reduction_history: Array
    actual_reduction_history: Array
    line_search_evaluations_history: Array
    regularization: Array
    status: Array
    iterations: Array
    accepted_iterations: Array
    failed_step: Array
    converged: Array
    method_id: str = eqx.field(static=True)


class ILQRResult(StrictModule):
    """A foundation-compatible control result plus policy and iLQR evidence."""

    control_result: ControlResult
    policy: ILQRPolicy
    diagnostics: ILQRDiagnostics

    @property
    def trajectory(self) -> ControlTrajectory:
        return self.control_result.trajectory

    @property
    def sampled_loss(self):
        return self.control_result.sampled_loss

    @property
    def feasibility(self):
        return self.control_result.feasibility

    @property
    def parameters(self) -> Array:
        return self.control_result.parameters

    @property
    def status(self) -> Array:
        return self.diagnostics.status

    @property
    def successful(self) -> Array:
        return self.control_result.successful & self.diagnostics.converged


class _LocalModel(StrictModule):
    dynamics_state: Array
    dynamics_control: Array
    running_state_gradient: Array
    running_control_gradient: Array
    running_state_hessian: Array
    running_control_hessian: Array
    running_control_state_hessian: Array
    terminal_gradient: Array
    terminal_hessian: Array
    control_gradient: Array


def _validate_solver_options(
    *,
    max_iterations: int,
    regularization: float,
    gradient_tolerance: float,
    cost_tolerance: float,
    line_search_steps: int,
    line_search_decay: float,
    initial_step_size: float,
    armijo: float,
) -> tuple[int, float, float, float, int, float, float, float]:
    iterations = int(max_iterations)
    searches = int(line_search_steps)
    regularization_ = float(regularization)
    gradient_tolerance_ = float(gradient_tolerance)
    cost_tolerance_ = float(cost_tolerance)
    decay = float(line_search_decay)
    initial_step = float(initial_step_size)
    armijo_ = float(armijo)
    if iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if searches <= 0:
        raise ValueError("line_search_steps must be positive.")
    if not isfinite(regularization_) or regularization_ < 0.0:
        raise ValueError("regularization must be finite and nonnegative.")
    if not isfinite(gradient_tolerance_) or gradient_tolerance_ < 0.0:
        raise ValueError("gradient_tolerance must be finite and nonnegative.")
    if not isfinite(cost_tolerance_) or cost_tolerance_ < 0.0:
        raise ValueError("cost_tolerance must be finite and nonnegative.")
    if not isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("line_search_decay must lie strictly between zero and one.")
    if not isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step_size must be finite and positive.")
    if not isfinite(armijo_) or not 0.0 <= armijo_ < 1.0:
        raise ValueError("armijo must lie in [0, 1).")
    return (
        iterations,
        regularization_,
        gradient_tolerance_,
        cost_tolerance_,
        searches,
        decay,
        initial_step,
        armijo_,
    )


def _flow_map(
    problem: ControlProblem,
    differential_flow: DifferentialControlFlow | None,
    /,
) -> tuple[ILQRFlow | None, str, str]:
    dynamics = problem.dynamics
    if isinstance(dynamics, DiscreteControlDynamics):
        if differential_flow is not None:
            raise ValueError(
                "differential_flow must be None for DiscreteControlDynamics."
            )
        return None, problem.time_grid.time_id, "backend:jax:discrete-flow-jvp"

    if not isinstance(dynamics, DifferentialControlDynamics):
        raise TypeError("Unsupported control dynamics type for iLQR.")
    if not isinstance(differential_flow, DifferentialControlFlow):
        raise ValueError(
            "DifferentialControlDynamics requires an explicit "
            "DifferentialControlFlow for iLQR."
        )

    def differential_step(
        t0: Array,
        t1: Array,
        step_index: Array,
        state: Array,
        control: Array,
    ) -> Array:
        del step_index
        return differential_flow(t0, t1, state, control, problem.args)

    return (
        differential_step,
        differential_flow.flow_id,
        f"backend:jax:{differential_flow.flow_id}",
    )


def _trajectory_cost(
    problem: ControlProblem,
    states: Array,
    controls: Array,
    /,
) -> tuple[Array, Array]:
    def running_term(time: Array, duration: Array, state: Array, control: Array):
        if problem.running_cost is None:
            value = jnp.asarray(0.0, dtype=states.dtype)
        else:
            value = jnp.asarray(problem.running_cost(time, state, control, problem.args))
            if value.shape != ():
                raise ValueError("RunningCost must return a scalar during iLQR.")
        return duration * value

    running = jax.vmap(running_term)(
        problem.time_grid.times[:-1],
        problem.time_grid.durations,
        states[:-1],
        controls,
    )
    if problem.terminal_cost is None:
        terminal = jnp.asarray(0.0, dtype=states.dtype)
    else:
        terminal = jnp.asarray(
            problem.terminal_cost(problem.time_grid.times[-1], states[-1], problem.args)
        )
        if terminal.shape != ():
            raise ValueError("TerminalCost must return a scalar during iLQR.")
    total = jnp.sum(running) + terminal
    valid = jnp.all(jnp.isfinite(running)) & jnp.isfinite(terminal) & jnp.isfinite(total)
    return total, valid


def _evaluate_ilqr_flow(
    problem: ControlProblem,
    flow: ILQRFlow | None,
    step: int | Array,
    state: Array,
    control: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    time = problem.time_grid.times[step]
    target_time = problem.time_grid.times[step + 1]
    step_index = jnp.asarray(step, dtype=jnp.int32)
    if isinstance(problem.dynamics, DiscreteControlDynamics):
        result = problem.dynamics.system.evaluate_result(
            DiscreteStepContext(time, target_time, step_index),
            state,
            problem.args,
            inputs=control,
        )
        return (
            result.candidate_state,
            result.accepted_state,
            result.successful,
            result.status,
        )
    if flow is None:
        raise ValueError("Differential iLQR evaluation requires an explicit flow.")
    value = jnp.asarray(flow(time, target_time, step_index, state, control))
    return (
        value,
        value,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _local_model(
    problem: ControlProblem,
    states: Array,
    controls: Array,
    flow: ILQRFlow | None,
    /,
) -> _LocalModel:
    state_layout = problem.dynamics.system.state_layout
    geometry = state_layout.geometry
    state_size = state_layout.local_size
    control_size = int(np.prod(problem.control_shape))
    local_template = jnp.asarray(geometry.inverse_retract(states[0], states[0]))
    if local_template.size != state_size:
        raise ValueError(
            "iLQR geometry local coordinates must match state_layout.local_size."
        )
    local_shape = local_template.shape
    basis_state = jnp.zeros((state_size,), dtype=states.dtype)

    def stage_model(
        step: Array,
        anchor: Array,
        nominal_next: Array,
        nominal_control: Array,
    ):
        nominal = jnp.concatenate((basis_state, nominal_control.reshape((control_size,))))

        def flattened_flow(joint: Array) -> Array:
            state = jnp.asarray(
                geometry.retract(anchor, joint[:state_size].reshape(local_shape))
            )
            control = joint[state_size:].reshape(problem.control_shape)
            _, accepted_state, successful, _ = _evaluate_ilqr_flow(
                problem,
                flow,
                step,
                state,
                control,
            )
            next_error = jnp.asarray(
                geometry.inverse_retract(nominal_next, accepted_state)
            ).reshape((state_size,))
            return jnp.where(
                successful & jnp.all(jnp.isfinite(accepted_state)),
                next_error,
                jnp.full_like(next_error, jnp.nan),
            )

        jacobian = jax.jacfwd(flattened_flow)(nominal)

        def stage_cost(joint: Array) -> Array:
            if problem.running_cost is None:
                return jnp.asarray(0.0, dtype=joint.dtype)
            state = jnp.asarray(
                geometry.retract(anchor, joint[:state_size].reshape(local_shape))
            )
            control = joint[state_size:].reshape(problem.control_shape)
            value = jnp.asarray(
                problem.running_cost(
                    problem.time_grid.times[step],
                    state,
                    control,
                    problem.args,
                )
            )
            if value.shape != ():
                raise ValueError("RunningCost must return a scalar during iLQR.")
            return problem.time_grid.durations[step] * value

        gradient, hessian = jax.jacfwd(jax.value_and_grad(stage_cost))(nominal)
        hessian = 0.5 * (hessian + hessian.T)
        return (
            jacobian[:, :state_size],
            jacobian[:, state_size:],
            gradient[:state_size],
            gradient[state_size:],
            hessian[:state_size, :state_size],
            hessian[state_size:, state_size:],
            hessian[state_size:, :state_size],
        )

    (
        dynamics_state,
        dynamics_control,
        running_state_gradient,
        running_control_gradient,
        running_state_hessian,
        running_control_hessian,
        running_control_state_hessian,
    ) = jax.vmap(stage_model)(
        jnp.arange(problem.time_grid.num_steps, dtype=jnp.int32),
        states[:-1],
        states[1:],
        controls,
    )

    terminal_anchor = states[-1]
    terminal_local = jnp.zeros((state_size,), dtype=states.dtype)
    terminal_template = jnp.asarray(
        geometry.inverse_retract(terminal_anchor, terminal_anchor)
    )

    def terminal_cost(local_coordinates: Array) -> Array:
        if problem.terminal_cost is None:
            return jnp.asarray(0.0, dtype=local_coordinates.dtype)
        terminal_state = jnp.asarray(
            geometry.retract(
                terminal_anchor,
                local_coordinates.reshape(terminal_template.shape),
            )
        )
        value = jnp.asarray(
            problem.terminal_cost(
                problem.time_grid.times[-1],
                terminal_state,
                problem.args,
            )
        )
        if value.shape != ():
            raise ValueError("TerminalCost must return a scalar during iLQR.")
        return value

    terminal_gradient, terminal_hessian = jax.jacfwd(jax.value_and_grad(terminal_cost))(
        terminal_local
    )
    terminal_hessian = 0.5 * (terminal_hessian + terminal_hessian.T)

    def adjoint_step(costate: Array, inputs: tuple[Array, Array, Array, Array]):
        dynamics_state_step, dynamics_control_step, state_gradient, control_gradient = (
            inputs
        )
        reduced_control_gradient = control_gradient + dynamics_control_step.T @ costate
        next_costate = state_gradient + dynamics_state_step.T @ costate
        return next_costate, reduced_control_gradient

    _, reverse_control_gradient = jax.lax.scan(
        adjoint_step,
        terminal_gradient,
        (
            dynamics_state[::-1],
            dynamics_control[::-1],
            running_state_gradient[::-1],
            running_control_gradient[::-1],
        ),
    )

    return _LocalModel(
        dynamics_state=dynamics_state,
        dynamics_control=dynamics_control,
        running_state_gradient=running_state_gradient,
        running_control_gradient=running_control_gradient,
        running_state_hessian=running_state_hessian,
        running_control_hessian=running_control_hessian,
        running_control_state_hessian=running_control_state_hessian,
        terminal_gradient=terminal_gradient,
        terminal_hessian=terminal_hessian,
        control_gradient=reverse_control_gradient[::-1],
    )


def solve_ilqr(
    problem: ControlProblem,
    initial_controls: ArrayLike,
    /,
    *,
    differential_flow: DifferentialControlFlow | None = None,
    max_iterations: int = 100,
    regularization: float = 1.0e-6,
    gradient_tolerance: float = 1.0e-6,
    cost_tolerance: float = 1.0e-9,
    line_search_steps: int = 10,
    line_search_decay: float = 0.5,
    initial_step_size: float = 1.0,
    armijo: float = 1.0e-4,
    policy_id: str | None = None,
    result_id: str | None = None,
) -> ILQRResult:
    """Solve a finite-horizon case batch through the prepared iLQR kernel."""
    from ._prepared_ilqr import (
        _compiled_solve_prepared_ilqr,
        plan_ilqr,
        prepare_ilqr,
    )

    plan = plan_ilqr(
        problem,
        max_iterations=max_iterations,
        regularization=regularization,
        gradient_tolerance=gradient_tolerance,
        cost_tolerance=cost_tolerance,
        line_search_steps=line_search_steps,
        line_search_decay=line_search_decay,
        initial_step_size=initial_step_size,
        armijo=armijo,
    )
    prepared = prepare_ilqr(
        plan,
        problem,
        initial_controls,
        differential_flow=differential_flow,
    )
    result = _compiled_solve_prepared_ilqr(
        prepared,
        policy_id=policy_id,
        result_id=result_id,
    )
    if problem.case_shape:
        return result
    attempted = int(jax.device_get(result.diagnostics.iterations))
    accepted = int(jax.device_get(result.diagnostics.accepted_iterations))
    diagnostics = eqx.tree_at(
        lambda value: (
            value.objective_history,
            value.gradient_norm_history,
            value.regularized_minimum_curvature_history,
            value.step_size_history,
            value.expected_reduction_history,
            value.actual_reduction_history,
            value.line_search_evaluations_history,
        ),
        result.diagnostics,
        (
            result.diagnostics.objective_history[: accepted + 1],
            result.diagnostics.gradient_norm_history[:attempted],
            result.diagnostics.regularized_minimum_curvature_history[:attempted],
            result.diagnostics.step_size_history[:attempted],
            result.diagnostics.expected_reduction_history[:attempted],
            result.diagnostics.actual_reduction_history[:attempted],
            result.diagnostics.line_search_evaluations_history[:attempted],
        ),
    )
    return eqx.tree_at(lambda value: value.diagnostics, result, diagnostics)


__all__ = [
    "DifferentialControlFlow",
    "DifferentialFlowStep",
    "ILQRDiagnostics",
    "ILQRPolicy",
    "ILQRResult",
    "ILQRStatus",
    "solve_ilqr",
]
