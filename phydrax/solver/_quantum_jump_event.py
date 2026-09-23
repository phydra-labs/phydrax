#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._geometry_precision import GeometryPrecisionPolicy
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..nonlinear import (
    Bisection,
    Brent,
    NonlinearTermination,
    NonlinearWork,
    scalar_root,
    ScalarRootProblem,
    TOMS748,
)
from ._quantum_jump import QuantumJumpProblem
from ._quantum_trajectory_contract import QuantumTrajectoryPlan, QuantumTrajectoryStatus


class QuantumJumpEventTable(StrictModule):
    times: Array
    channels: Array
    root_residuals: Array
    thresholds: Array
    active: Array

    def __init__(
        self,
        times: ArrayLike,
        channels: ArrayLike,
        root_residuals: ArrayLike,
        thresholds: ArrayLike,
        active: ArrayLike,
        /,
    ):
        self.times = jnp.asarray(times)
        self.channels = jnp.asarray(channels, dtype=jnp.int32)
        self.root_residuals = jnp.asarray(root_residuals)
        self.thresholds = jnp.asarray(thresholds)
        self.active = jnp.asarray(active, dtype=jnp.bool_)


class EventDrivenQuantumJumpResult(StrictModule):
    states: Array
    times: Array
    events: QuantumJumpEventTable
    norm_residual: Array
    valid: Array
    saturated: Array
    successful: Array
    status: Array
    work: NonlinearWork
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: ArrayLike,
        times: ArrayLike,
        events: QuantumJumpEventTable,
        /,
        *,
        problem_id: str,
        work: NonlinearWork,
        status: int | Array = int(QuantumTrajectoryStatus.SUCCESS),
        saturated: bool = False,
        successful: bool = True,
    ):
        self.states = jnp.asarray(states)
        self.times = jnp.asarray(times)
        self.events = events
        self.status = jnp.asarray(status, dtype=jnp.int32)
        if not isinstance(work, NonlinearWork):
            raise TypeError("work must be NonlinearWork.")
        self.work = work
        self.saturated = jnp.asarray(saturated, dtype=jnp.bool_)
        self.successful = jnp.asarray(successful, dtype=jnp.bool_) & (
            self.status == int(QuantumTrajectoryStatus.SUCCESS)
        )
        self.norm_residual = jnp.max(jnp.abs(jnp.linalg.norm(self.states, axis=-1) - 1.0))
        self.valid = (
            jnp.all(jnp.isfinite(self.states))
            & (self.norm_residual <= 1e-6)
            & jnp.all(jnp.where(events.active, events.root_residuals <= 1e-5, True))
            & ~self.saturated
            & self.successful
        )
        self.problem_id = str(problem_id)


def _effective_rhs(problem: QuantumJumpProblem, state: Array, /) -> Array:
    result = -1j * problem.hamiltonian(state)
    for operator in problem.collapse_operators:
        result = result - 0.5 * operator.adjoint(operator(state))
    return result


def _rk4(problem: QuantumJumpProblem, state: Array, step: Array, /) -> Array:
    first = _effective_rhs(problem, state)
    second = _effective_rhs(problem, state + 0.5 * step * first)
    third = _effective_rhs(problem, state + 0.5 * step * second)
    fourth = _effective_rhs(problem, state + step * third)
    return state + step * (first + 2 * second + 2 * third + fourth) / 6.0


def solve_event_driven_quantum_jump(
    problem: QuantumJumpProblem,
    key: Array,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    maximum_events: int = 128,
    bisection_iterations: int = 24,
    trajectory_plan: QuantumTrajectoryPlan | None = None,
) -> EventDrivenQuantumJumpResult:
    """Bracket survival events and refine with the shared scalar-root substrate."""
    if not isinstance(problem, QuantumJumpProblem):
        raise TypeError("problem must be a QuantumJumpProblem.")
    step = jnp.asarray(step_size, dtype=jnp.float64).reshape(())
    count = int(steps)
    if count < 0 or not bool(jnp.isfinite(step) & (step > 0.0)):
        raise ValueError("steps and step_size must be nonnegative/positive and finite.")
    plan = (
        QuantumTrajectoryPlan(
            maximum_events=int(maximum_events),
            root_iterations=int(bisection_iterations),
        )
        if trajectory_plan is None
        else trajectory_plan
    )
    if not isinstance(plan, QuantumTrajectoryPlan):
        raise TypeError("trajectory_plan must be QuantumTrajectoryPlan or None.")
    event_capacity = plan.maximum_events
    unnormalized = problem.initial_state
    threshold_address = SampleAddress(
        "quantum-trajectory",
        "jump-threshold",
        target=problem.problem_id,
        role="threshold",
    )
    channel_address = SampleAddress(
        "quantum-trajectory",
        "jump-channel",
        target=problem.problem_id,
        role="channel",
    )
    threshold = jax.random.uniform(derive_key(key, threshold_address, 0))
    saved = [problem.initial_state]
    event_times = jnp.zeros((event_capacity,), dtype=step.dtype)
    event_channels = -jnp.ones((event_capacity,), dtype=jnp.int32)
    root_residuals = jnp.zeros((event_capacity,), dtype=step.dtype)
    thresholds = jnp.zeros((event_capacity,), dtype=step.dtype)
    active = jnp.zeros((event_capacity,), dtype=jnp.bool_)
    event_count = 0
    saturated = False
    successful = True
    segment_count = 0
    status = QuantumTrajectoryStatus.SUCCESS
    work = NonlinearWork.zero()
    budget = plan.work_budget
    maximum_root_work = NonlinearWork(
        residual_evaluations=plan.root_iterations + 3,
        validity_evaluations=plan.root_iterations + 3,
    )

    precision = GeometryPrecisionPolicy()
    for index in range(count):
        current = unnormalized
        elapsed = jnp.asarray(0.0, dtype=step.dtype)
        remaining = step
        while float(jax.device_get(remaining)) > 0.0:
            segment_count += 1
            if segment_count > plan.maximum_segments:
                saturated = True
                status = QuantumTrajectoryStatus.TRUNCATION_BUDGET_EXCEEDED
                break
            candidate = _rk4(problem, current, remaining)
            start_survival = jnp.real(jnp.vdot(current, current))
            end_survival = jnp.real(jnp.vdot(candidate, candidate))
            crossing = (start_survival >= threshold) & (end_survival <= threshold)
            if not bool(jax.device_get(crossing)):
                current = candidate
                remaining = jnp.asarray(0.0, dtype=step.dtype)
                break
            if event_count >= event_capacity:
                saturated = True
                status = QuantumTrajectoryStatus.EVENT_CAPACITY_EXHAUSTED
                current = candidate
                break
            if not bool(jax.device_get(budget.permits(maximum_root_work))):
                saturated = True
                status = QuantumTrajectoryStatus.TRUNCATION_BUDGET_EXCEEDED
                break
            root_problem = ScalarRootProblem(
                lambda duration, args: (
                    jnp.real(
                        jnp.vdot(
                            _rk4(problem, current, duration),
                            _rk4(problem, current, duration),
                        )
                    )
                    - threshold
                ),
                bracket=(jnp.asarray(0.0, dtype=step.dtype), remaining),
                validity=lambda duration, value, args: jnp.isfinite(value),
                problem_id=f"{problem.problem_id}:survival-root",
            )
            method = {
                "toms748": TOMS748(),
                "brent": Brent(),
                "bisection": Bisection(),
            }[plan.root_method]
            root = scalar_root(
                root_problem,
                method=method,
                termination=NonlinearTermination(
                    absolute_residual=plan.root_tolerance,
                    relative_residual=0.0,
                    maximum_steps=plan.root_iterations,
                ),
            )
            diagnostics = root.nonlinear_result.diagnostics
            root_work = NonlinearWork(
                residual_evaluations=diagnostics.residual_evaluations,
                validity_evaluations=diagnostics.residual_evaluations,
                jvp_evaluations=diagnostics.jvp_evaluations,
            )
            if not bool(jax.device_get(budget.permits(root_work))):
                saturated = True
                status = QuantumTrajectoryStatus.TRUNCATION_BUDGET_EXCEEDED
                break
            work = work + root_work
            budget = budget.consume(root_work)
            if not bool(jax.device_get(root.successful)):
                successful = False
                status = QuantumTrajectoryStatus.ROOT_NOT_CONVERGED
                current = candidate
                break
            duration = root.nonlinear_result.state
            event_state = _rk4(problem, current, duration)
            norm = precision.norm(event_state)
            if not bool(jax.device_get(jnp.isfinite(norm) & (norm > 0.0))):
                successful = False
                status = QuantumTrajectoryStatus.INVALID_SURVIVAL
                current = candidate
                break
            normalized = event_state / norm
            collapsed = jnp.stack(
                [operator(normalized) for operator in problem.collapse_operators]
            )
            rates = jnp.real(ein.contract("ki,ki->k", jnp.conj(collapsed), collapsed))
            total_rate = jnp.sum(rates)
            if not bool(
                jax.device_get(
                    jnp.all(jnp.isfinite(rates) & (rates >= 0.0))
                    & jnp.isfinite(total_rate)
                    & (total_rate > 0.0)
                )
            ):
                successful = False
                status = QuantumTrajectoryStatus.INVALID_RATES
                current = candidate
                break
            probabilities = rates / total_rate
            local_key = derive_key(key, channel_address, event_count)
            channel = jax.random.categorical(
                local_key, jnp.log(jnp.maximum(probabilities, 1e-30))
            )
            selected = collapsed[channel]
            selected_norm = precision.norm(selected)
            if not bool(
                jax.device_get(jnp.isfinite(selected_norm) & (selected_norm > 0.0))
            ):
                successful = False
                status = QuantumTrajectoryStatus.ZERO_JUMP_STATE
                current = candidate
                break
            current = selected / selected_norm
            event_times = event_times.at[event_count].set(
                index * step + elapsed + duration
            )
            event_channels = event_channels.at[event_count].set(
                jnp.asarray(channel, dtype=jnp.int32)
            )
            root_residuals = root_residuals.at[event_count].set(
                jnp.abs(jnp.real(jnp.vdot(event_state, event_state)) - threshold)
            )
            thresholds = thresholds.at[event_count].set(threshold)
            active = active.at[event_count].set(True)
            event_count += 1
            elapsed = elapsed + duration
            remaining = jnp.maximum(remaining - duration, 0.0)
            threshold = jax.random.uniform(
                derive_key(key, threshold_address, event_count)
            )
        unnormalized = current
        state_norm = precision.norm(unnormalized)
        saved.append(
            unnormalized
            / jnp.where(
                jnp.isfinite(state_norm) & (state_norm > 0.0),
                state_norm,
                1.0,
            )
        )
        if saturated or not successful:
            break

    saved_count = len(saved)
    return EventDrivenQuantumJumpResult(
        jnp.stack(saved),
        step * jnp.arange(saved_count),
        QuantumJumpEventTable(
            event_times, event_channels, root_residuals, thresholds, active
        ),
        problem_id=problem.problem_id,
        saturated=saturated,
        successful=successful,
        status=int(status),
        work=work,
    )


__all__ = [
    "EventDrivenQuantumJumpResult",
    "QuantumJumpEventTable",
    "solve_event_driven_quantum_jump",
]
