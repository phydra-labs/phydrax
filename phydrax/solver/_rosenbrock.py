#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import TimeGrid
from ..linalg import (
    ArraySpace,
    FGMRES,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    solve,
    TolerancePolicy,
)
from ._differential import DifferentialProblem, DifferentialSolution
from ._temporal_method import (
    configuration_id,
    TemporalMethodCapabilities,
    TemporalSolveEvidence,
)


_DEFAULT_ARGS = object()


class RosenbrockWMethod(StrictModule, NonTrainableState):
    """Four-stage, third-order L-stable RA34PW2 Rosenbrock-W method."""

    capabilities: TemporalMethodCapabilities
    propagation: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    stage: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    weights: tuple[float, ...] = eqx.field(static=True)
    embedded_weights: tuple[float, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self):
        self.propagation = (
            (0.0, 0.0, 0.0, 0.0),
            (8.7173304301691801e-01, 0.0, 0.0, 0.0),
            (8.4457060015369423e-01, -1.1299064236484185e-01, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        )
        self.stage = (
            (4.3586652150845900e-01, 0.0, 0.0, 0.0),
            (-8.7173304301691801e-01, 4.3586652150845900e-01, 0.0, 0.0),
            (
                -9.0338057013044082e-01,
                5.4180672388095326e-02,
                4.3586652150845900e-01,
                0.0,
            ),
            (
                2.4212380706095346e-01,
                -1.2232505839045147,
                5.4526025533510214e-01,
                4.3586652150845900e-01,
            ),
        )
        self.weights = (
            2.4212380706095346e-01,
            -1.2232505839045147,
            1.5452602553351020,
            4.3586652150845900e-01,
        )
        self.embedded_weights = (
            3.7810903145819369e-01,
            -9.6042292212423178e-02,
            5.0000000000000000e-01,
            2.1793326075422950e-01,
        )
        self.method_id = "temporal:rosenbrock-w:ra34pw2"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="rosenbrock-w",
            order=3,
            embedded_order=2,
            dense_order=None,
            adaptive=True,
            history_depth=1,
            stage_abscissae=tuple(sum(row) for row in self.propagation),
            causal_stage_extent=1.0,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
            verified=True,
            method_id=self.method_id,
        )


class RosenbrockAdaptivePolicy(StrictModule, NonTrainableState):
    """Bounded accept/reject control for the embedded RA34PW2 pair."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float | None = eqx.field(static=True)
    safety: float = eqx.field(static=True)
    minimum_factor: float = eqx.field(static=True)
    maximum_factor: float = eqx.field(static=True)
    maximum_accepted_steps: int = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-8,
        initial_step: float = 1e-2,
        minimum_step: float = 1e-12,
        maximum_step: float | None = None,
        safety: float = 0.9,
        minimum_factor: float = 0.2,
        maximum_factor: float = 5.0,
        maximum_accepted_steps: int = 4096,
        maximum_attempts: int = 8192,
    ):
        values = tuple(
            float(value)
            for value in (
                relative_tolerance,
                absolute_tolerance,
                initial_step,
                minimum_step,
                safety,
                minimum_factor,
                maximum_factor,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Rosenbrock adaptive controls must be finite and positive.")
        relative, absolute, initial, minimum, safety_, factor_min, factor_max = values
        maximum = None if maximum_step is None else float(maximum_step)
        if maximum is not None and (
            not isfinite(maximum) or maximum <= 0.0 or maximum < minimum
        ):
            raise ValueError("maximum_step must be finite and at least minimum_step.")
        if initial < minimum or (maximum is not None and initial > maximum):
            raise ValueError("initial_step must lie inside the configured step bounds.")
        if not 0.0 < safety_ <= 1.0 or not 0.0 < factor_min <= 1.0 <= factor_max:
            raise ValueError("Adaptive safety and factor bounds are invalid.")
        accepted = int(maximum_accepted_steps)
        attempts = int(maximum_attempts)
        if accepted < 1 or attempts < accepted:
            raise ValueError(
                "Adaptive capacities must be positive and attempts cover accepts."
            )
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.initial_step = initial
        self.minimum_step = minimum
        self.maximum_step = maximum
        self.safety = safety_
        self.minimum_factor = factor_min
        self.maximum_factor = factor_max
        self.maximum_accepted_steps = accepted
        self.maximum_attempts = attempts


class _JacobianAction(eqx.Module):
    drift: Any
    time: Array
    state: Array
    args: Any

    def __call__(self, direction: Array, /) -> Array:
        return jax.jvp(
            lambda value: jnp.asarray(self.drift(self.time, value, self.args)),
            (self.state,),
            (direction,),
        )[1]


class _ShiftedJacobianAction(eqx.Module):
    jacobian: _JacobianAction
    scale: Array

    def __call__(self, direction: Array, /) -> Array:
        return direction - self.scale * self.jacobian(direction)


def _time_derivative(problem: DifferentialProblem, time: Array, state: Array, args: Any):
    return jax.jvp(
        lambda value: jnp.asarray(problem.drift(value, state, args)),
        (time,),
        (jnp.ones_like(time),),
    )[1]


def _default_linear_policy(state: Array, /) -> LinearSolvePolicy:
    restart = max(1, min(20, int(state.size) if state.shape else 1))
    return LinearSolvePolicy(
        FGMRES(restart=restart),
        tolerance=TolerancePolicy(relative=1e-8, absolute=1e-10, max_steps=64),
    )


def _rosenbrock_step(
    problem: DifferentialProblem,
    method: RosenbrockWMethod,
    policy: LinearSolvePolicy,
    space: ArraySpace,
    time: Array,
    state: Array,
    step_size: Array,
    args: Any,
    /,
) -> tuple[Array, Array, Array, Array]:
    propagation = jnp.asarray(method.propagation, dtype=state.real.dtype)
    stage_matrix = jnp.asarray(method.stage, dtype=state.real.dtype)
    weights = jnp.asarray(method.weights, dtype=state.real.dtype)
    embedded = jnp.asarray(method.embedded_weights, dtype=state.real.dtype)
    jacobian = _JacobianAction(problem.drift, time, state, args)
    time_derivative = _time_derivative(problem, time, state, args)
    increments: list[Array] = []
    successful = jnp.asarray(True)
    iterations = jnp.asarray(0, dtype=jnp.int32)
    for index in range(4):
        stage_state = state
        correction = jnp.zeros_like(state)
        for previous in range(index):
            stage_state = (
                stage_state + propagation[index, previous] * increments[previous]
            )
            correction = correction + stage_matrix[index, previous] * increments[previous]
        stage_time = time + step_size * jnp.sum(propagation[index])
        rhs = step_size * jnp.asarray(problem.drift(stage_time, stage_state, args))
        rhs = rhs + step_size * jacobian(correction)
        rhs = rhs + step_size**2 * jnp.sum(stage_matrix[index]) * time_derivative
        gamma = stage_matrix[index, index]
        operator = FunctionLinearOperator(
            _ShiftedJacobianAction(jacobian, step_size * gamma),
            source=space,
            target=space,
            operator_id=f"{method.method_id}:shifted-jacobian",
        )
        linear_result = solve(LinearSystem(operator), rhs, policy=policy)
        increments.append(jnp.asarray(linear_result.value))
        successful = successful & linear_result.successful
        iterations = iterations + jnp.asarray(
            linear_result.diagnostics.iterations, dtype=jnp.int32
        )
    stacked = jnp.stack(increments)
    next_state = state + jnp.tensordot(weights, stacked, axes=1)
    embedded_state = state + jnp.tensordot(embedded, stacked, axes=1)
    error = jnp.sqrt(jnp.mean(jnp.abs(next_state - embedded_state) ** 2))
    finite = jnp.all(jnp.isfinite(next_state)) & jnp.isfinite(error)
    return next_state, successful & finite, error, iterations


def solve_rosenbrock(
    problem: DifferentialProblem,
    time_grid: TimeGrid,
    /,
    *,
    method: RosenbrockWMethod | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    args: Any = _DEFAULT_ARGS,
) -> DifferentialSolution:
    """Integrate one deterministic ODE on a fixed grid with matrix-free RA34PW2."""
    if not isinstance(problem, DifferentialProblem) or problem.stochastic:
        raise TypeError("solve_rosenbrock requires a deterministic DifferentialProblem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    geometry = problem.state_geometry
    if geometry is not None and not geometry.trivial:
        raise ValueError("Rosenbrock-W currently requires Euclidean state geometry.")
    times = lax.stop_gradient(time_grid.times)
    times = eqx.error_if(
        times,
        ~jnp.isclose(times[0], problem.t0) | ~jnp.isclose(times[-1], problem.t1),
        "TimeGrid endpoints must match the differential problem.",
    )
    selected = RosenbrockWMethod() if method is None else method
    if not isinstance(selected, RosenbrockWMethod):
        raise TypeError("method must be RosenbrockWMethod or None.")
    policy = (
        _default_linear_policy(problem.initial_state)
        if linear_policy is None
        else linear_policy
    )
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be LinearSolvePolicy or None.")
    runtime_args = problem.args if args is _DEFAULT_ARGS else args
    space = ArraySpace(problem.initial_state.shape, dtype=problem.initial_state.dtype)

    def advance(carry, values):
        state, prior_valid = carry
        time, step_size = values

        def solve_step(_):
            return _rosenbrock_step(
                problem,
                selected,
                policy,
                space,
                time,
                state,
                step_size,
                runtime_args,
            )

        def skip_step(_):
            return (
                jnp.full_like(state, jnp.nan),
                jnp.asarray(False),
                jnp.asarray(jnp.inf, dtype=state.real.dtype),
                jnp.asarray(0, dtype=jnp.int32),
            )

        next_state, valid, error, iterations = lax.cond(
            prior_valid, solve_step, skip_step, operand=None
        )
        return (next_state, valid), (next_state, valid, error, iterations)

    (_, _), (step_states, step_valid, errors, iterations) = lax.scan(
        advance,
        (problem.initial_state, jnp.asarray(True)),
        (times[:-1], jnp.diff(times)),
    )
    states = jnp.concatenate((problem.initial_state[None, ...], step_states), axis=0)
    valid = jnp.concatenate((jnp.asarray([True]), step_valid))
    evidence = TemporalSolveEvidence(
        selected.capabilities,
        equation_form="explicit-ode",
        backend_id="backend:phydrax:rosenbrock-w",
        configuration_id=configuration_id(
            (selected, policy, time_grid.time_id), prefix="temporal-configuration"
        ),
        controller_id=f"controller:fixed-grid:{time_grid.time_id}",
        adjoint_id="adjoint:jax-discrete-linear-solves",
        event_id=None,
        adaptive=False,
        dense=False,
        maximum_steps=time_grid.num_steps,
    )
    successful = jnp.all(valid)
    return DifferentialSolution(
        times=times,
        states=states,
        valid=valid,
        backend_result=jnp.where(successful, 0, 1),
        stats={
            "num_steps": jnp.asarray(time_grid.num_steps, dtype=jnp.int32),
            "linear_iterations": jnp.sum(iterations),
            "embedded_error": errors,
        },
        solver_name="RA34PW2",
        interpretation=problem.interpretation,
        state_geometry_id=problem.state_geometry_id,
        solver_id=selected.method_id,
        resolved_method="RA34PW2:matrix-free-exact-jacobian",
        discretization_bundle=problem.discretization_bundle,
        backend_successful=successful,
        temporal_evidence=evidence,
        problem_id=problem.problem_id,
    )


class _RosenbrockAdaptiveCarry(StrictModule):
    time: Array
    state: Array
    step_size: Array
    accepted_count: Array
    attempt_count: Array
    save_index: Array
    step_sizes: Array
    step_valid: Array
    save_steps: Array
    successful: Array


def solve_rosenbrock_adaptive(
    problem: DifferentialProblem,
    time_grid: TimeGrid,
    /,
    *,
    method: RosenbrockWMethod | None = None,
    adaptive: RosenbrockAdaptivePolicy | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    args: Any = _DEFAULT_ARGS,
) -> DifferentialSolution:
    """Realize and replay an adaptive RA34PW2 solve with frozen-grid derivatives."""
    if not isinstance(problem, DifferentialProblem) or problem.stochastic:
        raise TypeError(
            "solve_rosenbrock_adaptive requires a deterministic DifferentialProblem."
        )
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    geometry = problem.state_geometry
    if geometry is not None and not geometry.trivial:
        raise ValueError("Rosenbrock-W currently requires Euclidean state geometry.")
    times = lax.stop_gradient(time_grid.times)
    times = eqx.error_if(
        times,
        ~jnp.isclose(times[0], problem.t0) | ~jnp.isclose(times[-1], problem.t1),
        "TimeGrid endpoints must match the differential problem.",
    )
    selected = RosenbrockWMethod() if method is None else method
    controller = RosenbrockAdaptivePolicy() if adaptive is None else adaptive
    policy = (
        _default_linear_policy(problem.initial_state)
        if linear_policy is None
        else linear_policy
    )
    if not isinstance(selected, RosenbrockWMethod):
        raise TypeError("method must be RosenbrockWMethod or None.")
    if not isinstance(controller, RosenbrockAdaptivePolicy):
        raise TypeError("adaptive must be RosenbrockAdaptivePolicy or None.")
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be LinearSolvePolicy or None.")
    runtime_args = problem.args if args is _DEFAULT_ARGS else args
    space = ArraySpace(problem.initial_state.shape, dtype=problem.initial_state.dtype)
    initial_step = jnp.minimum(
        jnp.asarray(controller.initial_step, dtype=times.dtype),
        times[1] - times[0],
    )
    if controller.maximum_step is not None:
        initial_step = jnp.minimum(initial_step, controller.maximum_step)
    initial = _RosenbrockAdaptiveCarry(
        time=times[0],
        state=problem.initial_state,
        step_size=initial_step,
        accepted_count=jnp.asarray(0, dtype=jnp.int32),
        attempt_count=jnp.asarray(0, dtype=jnp.int32),
        save_index=jnp.asarray(0, dtype=jnp.int32),
        step_sizes=jnp.zeros((controller.maximum_accepted_steps,), dtype=times.dtype),
        step_valid=jnp.zeros((controller.maximum_accepted_steps,), dtype=bool),
        save_steps=jnp.full((time_grid.num_steps,), -1, dtype=jnp.int32),
        successful=jnp.asarray(True),
    )

    def condition(current):
        return (
            current.successful
            & (current.save_index < time_grid.num_steps)
            & (current.accepted_count < controller.maximum_accepted_steps)
            & (current.attempt_count < controller.maximum_attempts)
        )

    def body(current):
        target = times[current.save_index + 1]
        remaining = target - current.time
        step_size = jnp.minimum(current.step_size, remaining)
        if controller.maximum_step is not None:
            step_size = jnp.minimum(step_size, controller.maximum_step)
        next_state, step_ok, error, _ = _rosenbrock_step(
            problem,
            selected,
            policy,
            space,
            current.time,
            current.state,
            step_size,
            runtime_args,
        )
        scale = (
            controller.absolute_tolerance
            + controller.relative_tolerance
            * jnp.maximum(jnp.abs(current.state), jnp.abs(next_state))
        )
        error_ratio = jnp.sqrt(
            jnp.mean(
                jnp.abs(
                    (next_state - current.state) * 0.0
                    + error / jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)
                )
                ** 2
            )
        )
        accepted = step_ok & (error_ratio <= 1.0)
        accepted_index = current.accepted_count
        lands_on_save = step_size == remaining
        step_sizes = lax.cond(
            accepted,
            lambda values: values.at[accepted_index].set(step_size),
            lambda values: values,
            current.step_sizes,
        )
        step_valid = lax.cond(
            accepted,
            lambda values: values.at[accepted_index].set(True),
            lambda values: values,
            current.step_valid,
        )
        save_steps = lax.cond(
            accepted & lands_on_save,
            lambda values: values.at[current.save_index].set(accepted_index),
            lambda values: values,
            current.save_steps,
        )
        exponent = 1.0 / 3.0
        raw_factor = controller.safety * jnp.maximum(
            error_ratio, jnp.finfo(step_size.dtype).tiny
        ) ** (-exponent)
        accepted_factor = jnp.clip(
            raw_factor, controller.minimum_factor, controller.maximum_factor
        )
        rejected_factor = jnp.clip(raw_factor, controller.minimum_factor, 1.0)
        factor = jnp.where(accepted, accepted_factor, rejected_factor)
        next_step = step_size * factor
        if controller.maximum_step is not None:
            next_step = jnp.minimum(next_step, controller.maximum_step)
        next_successful = current.successful & (
            accepted | (next_step >= controller.minimum_step)
        )
        return _RosenbrockAdaptiveCarry(
            time=jnp.where(accepted, current.time + step_size, current.time),
            state=jnp.where(accepted, next_state, current.state),
            step_size=next_step,
            accepted_count=current.accepted_count + accepted.astype(jnp.int32),
            attempt_count=current.attempt_count + 1,
            save_index=current.save_index + (accepted & lands_on_save).astype(jnp.int32),
            step_sizes=step_sizes,
            step_valid=step_valid,
            save_steps=save_steps,
            successful=next_successful,
        )

    schedule = lax.while_loop(condition, body, initial)
    completed = schedule.successful & (schedule.save_index == time_grid.num_steps)
    frozen_steps = lax.stop_gradient(schedule.step_sizes)
    frozen_valid = lax.stop_gradient(schedule.step_valid)
    frozen_save_steps = lax.stop_gradient(schedule.save_steps)

    def replay(carry, values):
        time, state = carry
        step_size, active = values

        def execute(_):
            next_state, valid, error, iterations = _rosenbrock_step(
                problem,
                selected,
                policy,
                space,
                time,
                state,
                step_size,
                runtime_args,
            )
            return (
                time + step_size,
                next_state,
                valid,
                error,
                iterations,
            )

        def skip(_):
            return (
                time,
                state,
                jnp.asarray(True),
                jnp.asarray(0.0, dtype=state.real.dtype),
                jnp.asarray(0, dtype=jnp.int32),
            )

        output = lax.cond(active, execute, skip, operand=None)
        next_time, next_state, *_ = output
        return (next_time, next_state), output

    _, replayed = lax.scan(
        replay,
        (times[0], problem.initial_state),
        (frozen_steps, frozen_valid),
    )
    _, accepted_states, replay_valid, replay_errors, replay_iterations = replayed
    safe_save_steps = jnp.clip(frozen_save_steps, 0, accepted_states.shape[0] - 1)
    saved_states = accepted_states[safe_save_steps]
    node_valid = (frozen_save_steps >= 0) & completed & replay_valid[safe_save_steps]
    states = jnp.concatenate((problem.initial_state[None, ...], saved_states), axis=0)
    valid = jnp.concatenate((jnp.asarray([True]), node_valid))
    successful = completed & jnp.all(valid)
    evidence = TemporalSolveEvidence(
        selected.capabilities,
        equation_form="explicit-ode",
        backend_id="backend:phydrax:rosenbrock-w",
        configuration_id=configuration_id(
            (selected, policy, controller, time_grid.time_id),
            prefix="temporal-configuration",
        ),
        controller_id=configuration_id(controller, prefix="controller"),
        adjoint_id="adjoint:frozen-accepted-grid-linear-solves",
        event_id=None,
        adaptive=True,
        dense=False,
        maximum_steps=controller.maximum_attempts,
    )
    return DifferentialSolution(
        times=times,
        states=states,
        valid=valid,
        backend_result=jnp.where(successful, 0, 1),
        stats={
            "accepted_steps": schedule.accepted_count,
            "attempts": schedule.attempt_count,
            "rejected_steps": schedule.attempt_count - schedule.accepted_count,
            "accepted_step_sizes": frozen_steps,
            "accepted_step_mask": frozen_valid,
            "embedded_error": replay_errors,
            "linear_iterations": jnp.sum(replay_iterations),
        },
        solver_name="RA34PW2",
        interpretation=problem.interpretation,
        state_geometry_id=problem.state_geometry_id,
        solver_id=selected.method_id,
        resolved_method="RA34PW2:adaptive-frozen-grid",
        discretization_bundle=problem.discretization_bundle,
        backend_successful=successful,
        temporal_evidence=evidence,
        problem_id=problem.problem_id,
    )


__all__ = [
    "RosenbrockAdaptivePolicy",
    "RosenbrockWMethod",
    "solve_rosenbrock",
    "solve_rosenbrock_adaptive",
]
