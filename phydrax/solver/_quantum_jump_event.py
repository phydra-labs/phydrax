#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..nonlinear import Bisection, Brent, scalar_root, ScalarRootProblem, TOMS748
from ._quantum_jump import QuantumJumpProblem
from ._quantum_trajectory_contract import QuantumTrajectoryPlan


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
        self.active = jnp.asarray(active, dtype=bool)


class EventDrivenQuantumJumpResult(StrictModule):
    states: Array
    times: Array
    events: QuantumJumpEventTable
    norm_residual: Array
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: ArrayLike,
        times: ArrayLike,
        events: QuantumJumpEventTable,
        /,
        *,
        problem_id: str,
    ):
        self.states = jnp.asarray(states)
        self.times = jnp.asarray(times)
        self.events = events
        self.norm_residual = jnp.max(jnp.abs(jnp.linalg.norm(self.states, axis=-1) - 1.0))
        self.valid = (
            jnp.all(jnp.isfinite(self.states))
            & (self.norm_residual <= 1e-6)
            & jnp.all(jnp.where(events.active, events.root_residuals <= 1e-5, True))
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
    step = jnp.asarray(step_size, dtype=float).reshape(())
    count = int(steps)
    event_capacity = int(maximum_events)
    if count < 0 or event_capacity < 1 or float(step) <= 0.0:
        raise ValueError("steps, maximum_events, and step_size must be positive.")
    plan = (
        QuantumTrajectoryPlan(
            maximum_events=event_capacity,
            root_iterations=bisection_iterations,
        )
        if trajectory_plan is None
        else trajectory_plan
    )
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
    active = jnp.zeros((event_capacity,), dtype=bool)
    event_count = 0

    for index in range(count):
        start = unnormalized
        candidate = _rk4(problem, start, step)
        start_survival = jnp.real(jnp.vdot(start, start))
        end_survival = jnp.real(jnp.vdot(candidate, candidate))
        crossing = (start_survival >= threshold) & (end_survival <= threshold)
        if bool(jax.device_get(crossing)) and event_count < event_capacity:
            root_problem = ScalarRootProblem(
                lambda duration, args: (
                    jnp.real(
                        jnp.vdot(
                            _rk4(problem, start, duration),
                            _rk4(problem, start, duration),
                        )
                    )
                    - threshold
                ),
                bracket=(jnp.asarray(0.0, dtype=step.dtype), step),
                validity=lambda duration, value, args: jnp.isfinite(value),
                problem_id=f"{problem.problem_id}:survival-root",
            )
            method = {
                "toms748": TOMS748(),
                "brent": Brent(),
                "bisection": Bisection(),
            }[plan.root_method]
            root = scalar_root(root_problem, method=method)
            upper = root.nonlinear_result.state
            event_state = _rk4(problem, start, upper)
            normalized = event_state / jnp.linalg.norm(event_state)
            collapsed = jnp.stack(
                [operator(normalized) for operator in problem.collapse_operators]
            )
            rates = jnp.real(oe.contract("ki,ki->k", jnp.conj(collapsed), collapsed))
            probabilities = rates / jnp.sum(rates)
            local_key = derive_key(key, channel_address, event_count)
            channel = jax.random.categorical(
                local_key, jnp.log(jnp.maximum(probabilities, 1e-30))
            )
            selected = collapsed[channel]
            post_jump = selected / jnp.linalg.norm(selected)
            remaining = step - upper
            unnormalized = _rk4(problem, post_jump, remaining)
            event_times = event_times.at[event_count].set(index * step + upper)
            event_channels = event_channels.at[event_count].set(
                jnp.asarray(channel, dtype=jnp.int32)
            )
            root_residuals = root_residuals.at[event_count].set(
                jnp.abs(jnp.real(jnp.vdot(event_state, event_state)) - threshold)
            )
            thresholds = thresholds.at[event_count].set(threshold)
            active = active.at[event_count].set(True)
            event_count += 1
            threshold = jax.random.uniform(
                derive_key(key, threshold_address, event_count)
            )
        else:
            unnormalized = candidate
        saved.append(unnormalized / jnp.linalg.norm(unnormalized))

    return EventDrivenQuantumJumpResult(
        jnp.stack(saved),
        step * jnp.arange(count + 1),
        QuantumJumpEventTable(
            event_times, event_channels, root_residuals, thresholds, active
        ),
        problem_id=problem.problem_id,
    )


__all__ = [
    "EventDrivenQuantumJumpResult",
    "QuantumJumpEventTable",
    "solve_event_driven_quantum_jump",
]
