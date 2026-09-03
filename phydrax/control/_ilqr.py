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
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..dynamics import DiscreteStepContext, TimeGrid
from ..dynamics._system import DiscreteTransitionEvidence
from ._constraints import evaluate_sampled_feasibility
from ._cost import evaluate_sampled_cost
from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics
from ._parameterization import AbstractControlParameterization
from ._problem import _identifier, ControlProblem
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_SUCCESS,
    ControlResult,
    ControlTrajectory,
)


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
    open-loop controls. With a state it applies
    ``u_nominal + feedback @ (state - state_nominal)``. No control clipping is
    performed.
    """

    time_grid: TimeGrid
    nominal_states: Array
    nominal_controls: Array
    feedback: Array
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
        state_shape: tuple[int, ...],
        control_shape: tuple[int, ...],
        policy_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("ILQRPolicy time_grid must be a TimeGrid.")
        states = jnp.asarray(nominal_states)
        controls = jnp.asarray(nominal_controls)
        gains = jnp.asarray(feedback)
        state_shape_ = tuple(int(size) for size in state_shape)
        control_shape_ = tuple(int(size) for size in control_shape)
        state_size = int(np.prod(state_shape_))
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
        self.state_shape = state_shape_
        self.case_shape = case_shape_
        self.control_shape = control_shape_
        self.parameter_shape = (0,)
        self.parameterization_id = _identifier(policy_id, "ILQRPolicy policy_id")
        self.approximation_id = "control:ilqr:affine-feedback"

    @property
    def feedforward(self) -> Array:
        """Affine intercepts in the equivalent ``feedback @ state + b`` form."""
        state_size = int(np.prod(self.state_shape))
        control_size = int(np.prod(self.control_shape))
        states = self.nominal_states[..., :-1, :].reshape(
            self.case_shape + (self.time_grid.num_steps, state_size)
        )
        controls = self.nominal_controls.reshape(
            self.case_shape + (self.time_grid.num_steps, control_size)
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
        nominal_states = jnp.take(
            self.nominal_states[..., :-1, :],
            indices,
            axis=len(self.case_shape),
        )
        gains = jnp.take(self.feedback, indices, axis=len(self.case_shape))
        state_size = int(np.prod(self.state_shape))
        control_size = int(np.prod(self.control_shape))
        prefix = self.case_shape + tuple(query.shape)
        flat_delta = (states - nominal_states).reshape(prefix + (state_size,))
        flat_gains = gains.reshape(prefix + (control_size, state_size))
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


class _BackwardPass(StrictModule):
    feedforward: Array
    feedback: Array
    linear_reduction: Array
    quadratic_reduction: Array
    minimum_curvature: Array
    positive_definite: Array
    failed_step: Array


def _host_bool(value: ArrayLike, /) -> bool:
    return bool(np.asarray(value))


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
) -> tuple[ILQRFlow, str, str]:
    dynamics = problem.dynamics
    if isinstance(dynamics, DiscreteControlDynamics):
        if differential_flow is not None:
            raise ValueError(
                "differential_flow must be None for DiscreteControlDynamics."
            )

        def discrete_step(
            t0: Array,
            t1: Array,
            step_index: Array,
            state: Array,
            control: Array,
        ) -> Array:
            context = DiscreteStepContext(t0, t1, step_index)
            result = dynamics.system.evaluate_result(
                context,
                state,
                problem.args,
                inputs=control,
            )
            return jnp.where(
                result.successful,
                result.accepted_state,
                jnp.full_like(result.accepted_state, jnp.nan),
            )

        return discrete_step, problem.time_grid.time_id, "backend:jax:discrete-flow-jvp"

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
    running_terms: list[Array] = []
    for step in range(problem.time_grid.num_steps):
        if problem.running_cost is None:
            value = jnp.asarray(0.0, dtype=states.dtype)
        else:
            value = jnp.asarray(
                problem.running_cost(
                    problem.time_grid.times[step],
                    states[step],
                    controls[step],
                    problem.args,
                )
            )
            if value.shape != ():
                raise ValueError("RunningCost must return a scalar during iLQR.")
        running_terms.append(problem.time_grid.durations[step] * value)
    if problem.terminal_cost is None:
        terminal = jnp.asarray(0.0, dtype=states.dtype)
    else:
        terminal = jnp.asarray(
            problem.terminal_cost(problem.time_grid.times[-1], states[-1], problem.args)
        )
        if terminal.shape != ():
            raise ValueError("TerminalCost must return a scalar during iLQR.")
    running = jnp.stack(running_terms)
    total = jnp.sum(running) + terminal
    valid = jnp.all(jnp.isfinite(running)) & jnp.isfinite(terminal) & jnp.isfinite(total)
    return total, valid


def _evaluate_ilqr_flow(
    problem: ControlProblem,
    flow: ILQRFlow,
    step: int,
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
    value = jnp.asarray(flow(time, target_time, step_index, state, control))
    return (
        value,
        value,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _open_loop_rollout(
    problem: ControlProblem,
    controls: Array,
    flow: ILQRFlow,
    /,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    DiscreteTransitionEvidence | None,
]:
    states: list[Array] = [problem.initial_state]
    valid: list[Array] = [jnp.all(jnp.isfinite(problem.initial_state))]
    candidates: list[Array] = []
    accepted: list[Array] = []
    transition_successful: list[Array] = []
    transition_status: list[Array] = []
    failed_step = -1
    active = _host_bool(valid[0])
    for step in range(problem.time_grid.num_steps):
        if active:
            candidate, next_state, successful, backend_status = _evaluate_ilqr_flow(
                problem,
                flow,
                step,
                states[-1],
                controls[step],
            )
            if tuple(next_state.shape) != problem.state_shape:
                raise ValueError(
                    "The selected iLQR flow must return exactly dynamics state_shape."
                )
            next_valid = (
                jnp.all(jnp.isfinite(controls[step]))
                & successful
                & jnp.all(jnp.isfinite(next_state))
            )
            active = _host_bool(next_valid)
            if not active:
                failed_step = step
        else:
            candidate = jnp.full(
                problem.state_shape, jnp.nan, dtype=problem.initial_state.dtype
            )
            next_state = jnp.full(
                problem.state_shape, jnp.nan, dtype=problem.initial_state.dtype
            )
            successful = jnp.asarray(False)
            backend_status = jnp.asarray(0, dtype=jnp.int32)
            next_valid = jnp.asarray(False)
        candidates.append(candidate)
        accepted.append(next_state)
        transition_successful.append(successful)
        transition_status.append(backend_status)
        states.append(next_state)
        valid.append(next_valid)
    state_array = jnp.stack(states)
    valid_array = jnp.stack(valid)
    if _host_bool(jnp.all(valid_array)):
        objective, cost_valid = _trajectory_cost(problem, state_array, controls)
    else:
        objective = jnp.asarray(jnp.inf, dtype=state_array.real.dtype)
        cost_valid = jnp.asarray(False)
    if not _host_bool(cost_valid):
        objective = jnp.asarray(jnp.inf, dtype=state_array.real.dtype)
    evidence = (
        DiscreteTransitionEvidence(
            jnp.stack(candidates),
            jnp.stack(accepted),
            jnp.stack(transition_successful),
            jnp.stack(transition_status),
        )
        if isinstance(problem.dynamics, DiscreteControlDynamics)
        else None
    )
    return (
        state_array,
        controls,
        valid_array,
        objective,
        jnp.asarray(failed_step, dtype=jnp.int32),
        evidence,
    )


def _feedback_rollout(
    problem: ControlProblem,
    nominal_states: Array,
    nominal_controls: Array,
    feedforward: Array,
    feedback: Array,
    step_size: float,
    flow: ILQRFlow,
    /,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    DiscreteTransitionEvidence | None,
]:
    state_size = int(np.prod(problem.state_shape))
    control_size = int(np.prod(problem.control_shape))
    states: list[Array] = [problem.initial_state]
    controls: list[Array] = []
    valid: list[Array] = [jnp.all(jnp.isfinite(problem.initial_state))]
    candidates: list[Array] = []
    accepted: list[Array] = []
    transition_successful: list[Array] = []
    transition_status: list[Array] = []
    failed_step = -1
    active = _host_bool(valid[0])
    for step in range(problem.time_grid.num_steps):
        if active:
            state_delta = states[-1].reshape((state_size,)) - nominal_states[
                step
            ].reshape((state_size,))
            control = (
                nominal_controls[step].reshape((control_size,))
                + step_size * feedforward[step]
                + feedback[step] @ state_delta
            ).reshape(problem.control_shape)
            candidate, next_state, successful, backend_status = _evaluate_ilqr_flow(
                problem,
                flow,
                step,
                states[-1],
                control,
            )
            if tuple(next_state.shape) != problem.state_shape:
                raise ValueError(
                    "The selected iLQR flow must return exactly dynamics state_shape."
                )
            next_valid = (
                jnp.all(jnp.isfinite(control))
                & successful
                & jnp.all(jnp.isfinite(next_state))
            )
            active = _host_bool(next_valid)
            if not active:
                failed_step = step
        else:
            control = jnp.full(
                problem.control_shape, jnp.nan, dtype=nominal_controls.dtype
            )
            candidate = jnp.full(
                problem.state_shape, jnp.nan, dtype=nominal_states.dtype
            )
            next_state = jnp.full(
                problem.state_shape, jnp.nan, dtype=nominal_states.dtype
            )
            successful = jnp.asarray(False)
            backend_status = jnp.asarray(0, dtype=jnp.int32)
            next_valid = jnp.asarray(False)
        controls.append(control)
        candidates.append(candidate)
        accepted.append(next_state)
        transition_successful.append(successful)
        transition_status.append(backend_status)
        states.append(next_state)
        valid.append(next_valid)
    state_array = jnp.stack(states)
    control_array = jnp.stack(controls)
    valid_array = jnp.stack(valid)
    if _host_bool(jnp.all(valid_array)):
        objective, cost_valid = _trajectory_cost(problem, state_array, control_array)
    else:
        objective = jnp.asarray(jnp.inf, dtype=state_array.real.dtype)
        cost_valid = jnp.asarray(False)
    if not _host_bool(cost_valid):
        objective = jnp.asarray(jnp.inf, dtype=state_array.real.dtype)
    evidence = (
        DiscreteTransitionEvidence(
            jnp.stack(candidates),
            jnp.stack(accepted),
            jnp.stack(transition_successful),
            jnp.stack(transition_status),
        )
        if isinstance(problem.dynamics, DiscreteControlDynamics)
        else None
    )
    return (
        state_array,
        control_array,
        valid_array,
        objective,
        jnp.asarray(failed_step, dtype=jnp.int32),
        evidence,
    )


def _local_model(
    problem: ControlProblem,
    states: Array,
    controls: Array,
    flow: ILQRFlow,
    /,
) -> _LocalModel:
    state_size = int(np.prod(problem.state_shape))
    control_size = int(np.prod(problem.control_shape))
    total_size = state_size + control_size
    dynamics_state: list[Array] = []
    dynamics_control: list[Array] = []
    running_state_gradient: list[Array] = []
    running_control_gradient: list[Array] = []
    running_state_hessian: list[Array] = []
    running_control_hessian: list[Array] = []
    running_control_state_hessian: list[Array] = []

    for step in range(problem.time_grid.num_steps):
        time = problem.time_grid.times[step]
        next_time = problem.time_grid.times[step + 1]
        nominal = jnp.concatenate(
            (
                states[step].reshape((state_size,)),
                controls[step].reshape((control_size,)),
            )
        )

        def flattened_flow(joint: Array) -> Array:
            state = joint[:state_size].reshape(problem.state_shape)
            control = joint[state_size:].reshape(problem.control_shape)
            return flow(
                time,
                next_time,
                jnp.asarray(step, dtype=jnp.int32),
                state,
                control,
            ).reshape((state_size,))

        basis = jnp.eye(total_size, dtype=nominal.dtype)
        columns = jax.vmap(
            lambda tangent: jax.jvp(flattened_flow, (nominal,), (tangent,))[1]
        )(basis)
        jacobian = jnp.swapaxes(columns, 0, 1)
        dynamics_state.append(jacobian[:, :state_size])
        dynamics_control.append(jacobian[:, state_size:])

        def stage_cost(joint: Array) -> Array:
            if problem.running_cost is None:
                return jnp.asarray(0.0, dtype=joint.dtype)
            state = joint[:state_size].reshape(problem.state_shape)
            control = joint[state_size:].reshape(problem.control_shape)
            value = jnp.asarray(problem.running_cost(time, state, control, problem.args))
            if value.shape != ():
                raise ValueError("RunningCost must return a scalar during iLQR.")
            return problem.time_grid.durations[step] * value

        gradient = jax.grad(stage_cost)(nominal)
        hessian = jax.hessian(stage_cost)(nominal)
        hessian = 0.5 * (hessian + hessian.T)
        running_state_gradient.append(gradient[:state_size])
        running_control_gradient.append(gradient[state_size:])
        running_state_hessian.append(hessian[:state_size, :state_size])
        running_control_hessian.append(hessian[state_size:, state_size:])
        running_control_state_hessian.append(hessian[state_size:, :state_size])

    terminal_state = states[-1].reshape((state_size,))

    def terminal_cost(flat_state: Array) -> Array:
        if problem.terminal_cost is None:
            return jnp.asarray(0.0, dtype=flat_state.dtype)
        value = jnp.asarray(
            problem.terminal_cost(
                problem.time_grid.times[-1],
                flat_state.reshape(problem.state_shape),
                problem.args,
            )
        )
        if value.shape != ():
            raise ValueError("TerminalCost must return a scalar during iLQR.")
        return value

    terminal_gradient = jax.grad(terminal_cost)(terminal_state)
    terminal_hessian = jax.hessian(terminal_cost)(terminal_state)
    terminal_hessian = 0.5 * (terminal_hessian + terminal_hessian.T)
    a = jnp.stack(dynamics_state)
    b = jnp.stack(dynamics_control)
    lx = jnp.stack(running_state_gradient)
    lu = jnp.stack(running_control_gradient)

    costate = terminal_gradient
    control_gradient: list[Array] = [
        jnp.zeros_like(lu[0]) for _ in range(problem.time_grid.num_steps)
    ]
    for step in range(problem.time_grid.num_steps - 1, -1, -1):
        control_gradient[step] = lu[step] + b[step].T @ costate
        costate = lx[step] + a[step].T @ costate

    return _LocalModel(
        dynamics_state=a,
        dynamics_control=b,
        running_state_gradient=lx,
        running_control_gradient=lu,
        running_state_hessian=jnp.stack(running_state_hessian),
        running_control_hessian=jnp.stack(running_control_hessian),
        running_control_state_hessian=jnp.stack(running_control_state_hessian),
        terminal_gradient=terminal_gradient,
        terminal_hessian=terminal_hessian,
        control_gradient=jnp.stack(control_gradient),
    )


def _backward_pass(model: _LocalModel, regularization: float, /) -> _BackwardPass:
    num_steps = int(model.dynamics_state.shape[0])
    control_size = int(model.dynamics_control.shape[-1])
    state_size = int(model.dynamics_state.shape[-1])
    value_gradient = model.terminal_gradient
    value_hessian = model.terminal_hessian
    feedforward = [
        jnp.zeros((control_size,), dtype=value_gradient.dtype) for _ in range(num_steps)
    ]
    feedback = [
        jnp.zeros((control_size, state_size), dtype=value_gradient.dtype)
        for _ in range(num_steps)
    ]
    linear_reduction = jnp.asarray(0.0, dtype=value_gradient.dtype)
    quadratic_reduction = jnp.asarray(0.0, dtype=value_gradient.dtype)
    minimum_curvature = jnp.asarray(jnp.inf, dtype=value_gradient.real.dtype)
    failed_step = -1
    positive_definite = True
    identity = jnp.eye(control_size, dtype=value_hessian.dtype)

    for step in range(num_steps - 1, -1, -1):
        a = model.dynamics_state[step]
        b = model.dynamics_control[step]
        qx = model.running_state_gradient[step] + a.T @ value_gradient
        qu = model.running_control_gradient[step] + b.T @ value_gradient
        qxx = model.running_state_hessian[step] + a.T @ value_hessian @ a
        quu = model.running_control_hessian[step] + b.T @ value_hessian @ b
        qux = model.running_control_state_hessian[step] + b.T @ value_hessian @ a
        qxx = 0.5 * (qxx + qxx.T)
        regularized_quu = 0.5 * (quu + quu.T) + regularization * identity
        curvature = jnp.min(jnp.linalg.eigvalsh(regularized_quu))
        minimum_curvature = jnp.minimum(minimum_curvature, curvature)
        if not _host_bool(jnp.isfinite(curvature) & (curvature > 0.0)):
            positive_definite = False
            failed_step = step
            break
        factor = jnp.linalg.cholesky(regularized_quu)
        solved_gradient = jsp_linalg.solve_triangular(
            factor.T,
            jsp_linalg.solve_triangular(factor, qu, lower=True),
            lower=False,
        )
        solved_cross = jsp_linalg.solve_triangular(
            factor.T,
            jsp_linalg.solve_triangular(factor, qux, lower=True),
            lower=False,
        )
        k = -solved_gradient
        gain = -solved_cross
        feedforward[step] = k
        feedback[step] = gain
        linear_reduction = linear_reduction + qu @ k
        quadratic_reduction = quadratic_reduction + 0.5 * k @ regularized_quu @ k
        value_gradient = qx + gain.T @ qu + qux.T @ k + gain.T @ regularized_quu @ k
        value_hessian = (
            qxx + gain.T @ qux + qux.T @ gain + gain.T @ regularized_quu @ gain
        )
        value_hessian = 0.5 * (value_hessian + value_hessian.T)

    return _BackwardPass(
        feedforward=jnp.stack(feedforward),
        feedback=jnp.stack(feedback),
        linear_reduction=linear_reduction,
        quadratic_reduction=quadratic_reduction,
        minimum_curvature=minimum_curvature,
        positive_definite=jnp.asarray(positive_definite),
        failed_step=jnp.asarray(failed_step, dtype=jnp.int32),
    )


def _history(values: list[Array], dtype: Any, /) -> Array:
    return jnp.stack(values) if values else jnp.empty((0,), dtype=dtype)


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
    """Solve one homogeneous finite-horizon case batch by iterative LQR.

    Nonempty case axes route through the prepared fixed-capacity JAX kernel.
    Unbatched calls retain the established convenience implementation.
    """
    if not isinstance(problem, ControlProblem):
        raise TypeError("solve_ilqr problem must be a ControlProblem.")
    if problem.case_shape:
        from ._batched_trajectory import (
            plan_ilqr,
            prepare_ilqr,
            solve_prepared_ilqr,
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
        return solve_prepared_ilqr(
            prepared,
            policy_id=policy_id,
            result_id=result_id,
        )
    if problem.path_constraints or problem.terminal_constraints:
        raise ValueError(
            "solve_ilqr is unconstrained; constrained ControlProblems are unsupported."
        )
    (
        max_iterations_,
        regularization_,
        gradient_tolerance_,
        cost_tolerance_,
        line_search_steps_,
        line_search_decay_,
        initial_step_size_,
        armijo_,
    ) = _validate_solver_options(
        max_iterations=max_iterations,
        regularization=regularization,
        gradient_tolerance=gradient_tolerance,
        cost_tolerance=cost_tolerance,
        line_search_steps=line_search_steps,
        line_search_decay=line_search_decay,
        initial_step_size=initial_step_size,
        armijo=armijo,
    )
    controls = jnp.asarray(initial_controls)
    expected_controls = (problem.time_grid.num_steps,) + problem.control_shape
    if tuple(controls.shape) != expected_controls:
        raise ValueError(
            f"initial_controls must have shape {expected_controls}; got {controls.shape}."
        )
    if not jnp.issubdtype(controls.dtype, jnp.inexact):
        controls = controls.astype(float)
    flow, discretization_id, backend_id = _flow_map(problem, differential_flow)
    (
        states,
        controls,
        valid,
        objective,
        failed_step,
        transition_evidence,
    ) = _open_loop_rollout(problem, controls, flow)

    objective_history: list[Array] = [objective]
    gradient_history: list[Array] = []
    curvature_history: list[Array] = []
    step_history: list[Array] = []
    expected_history: list[Array] = []
    actual_history: list[Array] = []
    evaluations_history: list[Array] = []
    accepted_iterations = 0
    status = ILQRStatus.MAX_ITERATIONS
    final_feedback = jnp.zeros(
        (
            problem.time_grid.num_steps,
            int(np.prod(problem.control_shape)),
            int(np.prod(problem.state_shape)),
        ),
        dtype=controls.dtype,
    )

    if not _host_bool(jnp.all(valid) & jnp.isfinite(objective)):
        status = ILQRStatus.INITIAL_ROLLOUT_FAILED
    else:
        for _ in range(max_iterations_):
            model = _local_model(problem, states, controls, flow)
            gradient_norm = jnp.linalg.norm(model.control_gradient.reshape((-1,)))
            backward = _backward_pass(model, regularization_)
            gradient_history.append(gradient_norm)
            curvature_history.append(backward.minimum_curvature)
            if not _host_bool(backward.positive_definite):
                status = ILQRStatus.BACKWARD_PASS_NOT_POSITIVE_DEFINITE
                failed_step = backward.failed_step
                step_history.append(jnp.asarray(0.0, dtype=objective.dtype))
                expected_history.append(jnp.asarray(jnp.nan, dtype=objective.dtype))
                actual_history.append(jnp.asarray(jnp.nan, dtype=objective.dtype))
                evaluations_history.append(jnp.asarray(0, dtype=jnp.int32))
                break
            final_feedback = backward.feedback
            if _host_bool(gradient_norm <= gradient_tolerance_):
                status = ILQRStatus.SUCCESS
                step_history.append(jnp.asarray(0.0, dtype=objective.dtype))
                expected_history.append(jnp.asarray(0.0, dtype=objective.dtype))
                actual_history.append(jnp.asarray(0.0, dtype=objective.dtype))
                evaluations_history.append(jnp.asarray(0, dtype=jnp.int32))
                break

            accepted = False
            last_expected = jnp.asarray(jnp.nan, dtype=objective.dtype)
            last_actual = jnp.asarray(jnp.nan, dtype=objective.dtype)
            candidate_failed_step = jnp.asarray(-1, dtype=jnp.int32)
            for search in range(line_search_steps_):
                step_size = initial_step_size_ * line_search_decay_**search
                (
                    candidate_states,
                    candidate_controls,
                    candidate_valid,
                    candidate_cost,
                    candidate_failure,
                    candidate_evidence,
                ) = _feedback_rollout(
                    problem,
                    states,
                    controls,
                    backward.feedforward,
                    backward.feedback,
                    step_size,
                    flow,
                )
                expected_reduction = -(
                    step_size * backward.linear_reduction
                    + step_size**2 * backward.quadratic_reduction
                )
                actual_reduction = objective - candidate_cost
                last_expected = expected_reduction
                last_actual = actual_reduction
                if int(np.asarray(candidate_failure)) >= 0:
                    candidate_failed_step = candidate_failure
                acceptable = (
                    jnp.all(candidate_valid)
                    & jnp.isfinite(candidate_cost)
                    & jnp.isfinite(expected_reduction)
                    & (expected_reduction > 0.0)
                    & (actual_reduction > 0.0)
                    & (actual_reduction >= armijo_ * expected_reduction)
                )
                if _host_bool(acceptable):
                    previous_objective = objective
                    states = candidate_states
                    controls = candidate_controls
                    valid = candidate_valid
                    objective = candidate_cost
                    transition_evidence = candidate_evidence
                    objective_history.append(objective)
                    step_history.append(jnp.asarray(step_size, dtype=objective.dtype))
                    expected_history.append(expected_reduction)
                    actual_history.append(actual_reduction)
                    evaluations_history.append(jnp.asarray(search + 1, dtype=jnp.int32))
                    accepted_iterations += 1
                    accepted = True
                    if _host_bool(
                        actual_reduction
                        <= cost_tolerance_
                        * jnp.maximum(jnp.asarray(1.0), jnp.abs(previous_objective))
                    ):
                        status = ILQRStatus.SUCCESS
                    break
            if not accepted:
                status = ILQRStatus.LINE_SEARCH_FAILED
                failed_step = candidate_failed_step
                step_history.append(jnp.asarray(0.0, dtype=objective.dtype))
                expected_history.append(last_expected)
                actual_history.append(last_actual)
                evaluations_history.append(
                    jnp.asarray(line_search_steps_, dtype=jnp.int32)
                )
                break
            if status == ILQRStatus.SUCCESS:
                break

    policy_name = f"ilqr-policy:{problem.problem_id}" if policy_id is None else policy_id
    policy = ILQRPolicy(
        problem.time_grid,
        states,
        controls,
        final_feedback,
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        policy_id=policy_name,
    )
    trajectory_status = jnp.asarray(
        CONTROL_SUCCESS if _host_bool(jnp.all(valid)) else CONTROL_DYNAMICS_FAILED,
        dtype=jnp.int32,
    )
    trajectory_backend_status = (
        transition_evidence.first_failure_status
        if transition_evidence is not None
        else trajectory_status
    )
    trajectory = ControlTrajectory(
        time_grid=problem.time_grid,
        states=states,
        controls=controls,
        valid=valid,
        status=trajectory_status,
        backend_status=trajectory_backend_status,
        transition_evidence=transition_evidence,
        case_shape=(),
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        problem_id=problem.problem_id,
        dynamics_id=problem.dynamics.dynamics_id,
        control_id=policy.parameterization_id,
        backend_id=backend_id,
        method_id="iterative-lqr:explicit-flow-map-jvp",
        discretization_id=discretization_id,
        approximation_id=policy.approximation_id,
    )
    sampled_loss = evaluate_sampled_cost(problem, trajectory)
    feasibility = evaluate_sampled_feasibility(problem, trajectory)
    control_result = ControlResult(
        trajectory=trajectory,
        parameters=controls,
        sampled_loss=sampled_loss,
        feasibility=feasibility,
        result_id=(
            f"ilqr-result:{problem.problem_id}" if result_id is None else result_id
        ),
        method_id=trajectory.method_id,
    )
    diagnostics = ILQRDiagnostics(
        objective_history=_history(objective_history, objective.dtype),
        gradient_norm_history=_history(gradient_history, objective.dtype),
        regularized_minimum_curvature_history=_history(
            curvature_history, objective.dtype
        ),
        step_size_history=_history(step_history, objective.dtype),
        expected_reduction_history=_history(expected_history, objective.dtype),
        actual_reduction_history=_history(actual_history, objective.dtype),
        line_search_evaluations_history=_history(evaluations_history, jnp.int32),
        regularization=jnp.asarray(regularization_, dtype=objective.dtype),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        iterations=jnp.asarray(len(gradient_history), dtype=jnp.int32),
        accepted_iterations=jnp.asarray(accepted_iterations, dtype=jnp.int32),
        failed_step=failed_step,
        converged=jnp.asarray(status == ILQRStatus.SUCCESS),
        method_id="iterative-lqr:regularized-backward+backtracking",
    )
    return ILQRResult(
        control_result=control_result,
        policy=policy,
        diagnostics=diagnostics,
    )


__all__ = [
    "DifferentialControlFlow",
    "DifferentialFlowStep",
    "ILQRDiagnostics",
    "ILQRPolicy",
    "ILQRResult",
    "ILQRStatus",
    "solve_ilqr",
]
