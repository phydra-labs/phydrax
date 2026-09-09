#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-capacity reference rollouts for feedback-controlled jump processes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite, prod
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics._grid import TimeGrid
from ...stochastic._jump import (
    AbstractJumpProcess,
    JUMP_INVALID_INTENSITY,
    JUMP_MAX_EVENTS,
    JUMP_SOLVER_FAILURE,
    JUMP_SUCCESS,
    JumpEventBatch,
    PoissonClockRealization,
)


ControlledJumpPolicy: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _shape(value: Sequence[int], owner: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _finite_real(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.number)
        and not jnp.issubdtype(array.dtype, jnp.complexfloating)
    ):
        raise TypeError(f"{owner} must be a real numeric array.")
    array = array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{owner} must be finite.")
    return array


class ControlledJumpProblem(StrictModule):
    """A supplied jump process controlled through its explicit ``args`` channel.

    During a rollout, each process callback receives ``(action, problem.args)`` as
    its ``args`` value. This keeps the stochastic jump definition and solver owner
    unchanged while making the left-continuous control visible to intensities, marks,
    and jump maps.
    """

    process: AbstractJumpProcess
    initial_state: Array
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    action_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        process: AbstractJumpProcess,
        initial_state: ArrayLike,
        /,
        *,
        action_shape: Sequence[int],
        args: Any = None,
        problem_id: str,
    ):
        if not isinstance(process, AbstractJumpProcess):
            raise TypeError("process must implement AbstractJumpProcess.")
        state = _finite_real(initial_state, "initial_state")
        if tuple(state.shape) != process.state_shape:
            raise ValueError(
                "initial_state must have the process state_shape "
                f"{process.state_shape}; got {state.shape}."
            )
        self.process = process
        self.initial_state = state
        self.args = args
        self.state_shape = process.state_shape
        self.action_shape = _shape(action_shape, "action_shape")
        self.problem_id = _identifier(problem_id, "problem_id")


class ControlledJumpPlan(StrictModule):
    """A fixed observation grid and event-capacity contract for one rollout."""

    time_grid: TimeGrid
    intensity_tolerance: float = eqx.field(static=True)
    same_time_order: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_grid: TimeGrid,
        /,
        *,
        intensity_tolerance: float = 1.0e-12,
        plan_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        tolerance = float(intensity_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("intensity_tolerance must be finite and nonnegative.")
        self.time_grid = time_grid
        self.intensity_tolerance = tolerance
        self.same_time_order = (
            "event-time-then-channel-then-channel-event-index;"
            "boundary-events-before-next-control"
        )
        self.plan_id = _identifier(plan_id, "plan_id")


class ControlledJumpEvidence(StrictModule):
    """Pathwise finite-intensity and finite-capacity evidence."""

    minimum_intensity: Array
    maximum_intensity: Array
    event_counts: Array
    finite: Array
    capacity_complete: Array
    successful: Array
    scope: str = eqx.field(static=True)


class ControlledJumpPathBatch(StrictModule):
    """Grid states, predictable controls, events, and explicit path statuses."""

    time_grid: TimeGrid
    states: Array
    actions: Array
    events: JumpEventBatch
    evidence: ControlledJumpEvidence
    valid: Array
    status: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    action_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    same_time_order: str = eqx.field(static=True)

    @property
    def path_count(self) -> int:
        return prod(self.batch_shape) if self.batch_shape else 1


def _callback_action(
    policy: ControlledJumpPolicy,
    time: float,
    state: np.ndarray,
    args: Any,
    action_shape: tuple[int, ...],
    /,
) -> np.ndarray:
    value = np.asarray(policy(jnp.asarray(time), jnp.asarray(state), args))
    if value.shape != action_shape:
        raise ValueError(
            f"policy must return action_shape {action_shape}; got {value.shape}."
        )
    if np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError("policy must return real actions.")
    return np.asarray(value, dtype=float)


def rollout_controlled_jumps_reference(
    problem: ControlledJumpProblem,
    plan: ControlledJumpPlan,
    realization: PoissonClockRealization,
    policy: ControlledJumpPolicy,
    /,
    *,
    policy_id: str,
) -> ControlledJumpPathBatch:
    """Roll out piecewise-constant feedback with a next-reaction reference method.

    Controls are sampled at each grid left endpoint and held through that grid cell.
    Simultaneous candidate events are ordered by channel index. Events exactly on a
    grid boundary use the preceding action and are applied before the next control is
    observed. Capacity exhaustion is reported as an unsuccessful truncated path.
    """

    if not isinstance(problem, ControlledJumpProblem):
        raise TypeError("problem must be a ControlledJumpProblem.")
    if not isinstance(plan, ControlledJumpPlan):
        raise TypeError("plan must be a ControlledJumpPlan.")
    if not isinstance(realization, PoissonClockRealization):
        raise TypeError("realization must be a PoissonClockRealization.")
    if not callable(policy):
        raise TypeError("policy must be callable.")
    identifier = _identifier(policy_id, "policy_id")
    if realization.process_id != problem.process.process_id:
        raise ValueError("realization process_id does not match the jump process.")
    support = (float(plan.time_grid.times[0]), float(plan.time_grid.times[-1]))
    if not np.allclose(realization.support, support, rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("realization support must equal the plan time-grid support.")
    if realization.num_channels != problem.process.num_channels:
        raise ValueError("realization channel count does not match the jump process.")

    batch_shape = realization.sample_shape
    path_count = realization.num_paths
    channels = problem.process.num_channels
    per_channel_capacity = realization.max_events_per_channel
    event_capacity = channels * per_channel_capacity
    state_shape = problem.state_shape
    mark_shape = problem.process.mark_shape
    steps = plan.time_grid.num_steps
    times = np.asarray(plan.time_grid.times, dtype=float)
    thresholds = np.asarray(realization.thresholds, dtype=float).reshape(
        (path_count, channels, per_channel_capacity)
    )
    mark_keys = realization.mark_keys.reshape(
        (path_count, channels, per_channel_capacity) + tuple(realization.root_key.shape)
    )

    states = np.zeros((path_count, steps + 1) + state_shape, dtype=float)
    actions = np.zeros((path_count, steps) + problem.action_shape, dtype=float)
    event_times = np.zeros((path_count, event_capacity), dtype=float)
    event_channels = np.zeros((path_count, event_capacity), dtype=np.int32)
    event_marks = np.zeros((path_count, event_capacity) + mark_shape, dtype=float)
    event_valid = np.zeros((path_count, event_capacity), dtype=bool)
    pre_states = np.zeros((path_count, event_capacity) + state_shape, dtype=float)
    post_states = np.zeros((path_count, event_capacity) + state_shape, dtype=float)
    statuses = np.full((path_count,), JUMP_SUCCESS, dtype=np.int32)
    minimum_intensity = np.full((path_count,), np.inf, dtype=float)
    maximum_intensity = np.zeros((path_count,), dtype=float)
    finite = np.ones((path_count,), dtype=bool)

    initial = np.asarray(problem.initial_state, dtype=float)
    tolerance = plan.intensity_tolerance
    for path_index in range(path_count):
        state = initial.copy()
        states[path_index, 0] = state
        integrated = np.zeros((channels,), dtype=float)
        threshold_index = np.zeros((channels,), dtype=np.int32)
        event_index = 0
        failed = False
        for step in range(steps):
            action = _callback_action(
                policy,
                times[step],
                state,
                problem.args,
                problem.action_shape,
            )
            actions[path_index, step] = action
            if not np.all(np.isfinite(action)):
                statuses[path_index] = JUMP_SOLVER_FAILURE
                finite[path_index] = False
                failed = True
                break
            current_time = times[step]
            end_time = times[step + 1]
            while current_time <= end_time:
                controlled_args = (jnp.asarray(action), problem.args)
                raw_rates = np.asarray(
                    problem.process.intensities(
                        jnp.asarray(current_time), jnp.asarray(state), controlled_args
                    )
                )
                if np.issubdtype(raw_rates.dtype, np.complexfloating):
                    statuses[path_index] = JUMP_INVALID_INTENSITY
                    finite[path_index] = False
                    failed = True
                    break
                rates = np.asarray(raw_rates, dtype=float)
                if rates.shape != (channels,):
                    raise ValueError(
                        "process intensities must return exactly one channel vector "
                        "for an unbatched controlled state."
                    )
                if not np.all(np.isfinite(rates)) or np.any(rates < -tolerance):
                    statuses[path_index] = JUMP_INVALID_INTENSITY
                    finite[path_index] = bool(np.all(np.isfinite(rates)))
                    failed = True
                    break
                rates = np.maximum(rates, 0.0)
                minimum_intensity[path_index] = min(
                    minimum_intensity[path_index], float(np.min(rates))
                )
                maximum_intensity[path_index] = max(
                    maximum_intensity[path_index], float(np.max(rates))
                )
                candidates = np.full((channels,), np.inf, dtype=float)
                for channel in range(channels):
                    index = int(threshold_index[channel])
                    if index < per_channel_capacity and rates[channel] > 0.0:
                        remaining = (
                            thresholds[path_index, channel, index] - integrated[channel]
                        )
                        candidates[channel] = (
                            current_time + max(remaining, 0.0) / rates[channel]
                        )
                channel = int(np.argmin(candidates))
                candidate_time = float(candidates[channel])
                if candidate_time > end_time + tolerance or not np.isfinite(
                    candidate_time
                ):
                    integrated += rates * (end_time - current_time)
                    current_time = end_time
                    break
                resolved_event_time = min(candidate_time, end_time)
                elapsed = max(resolved_event_time - current_time, 0.0)
                integrated += rates * elapsed
                current_time = resolved_event_time
                if event_index >= event_capacity:
                    statuses[path_index] = JUMP_MAX_EVENTS
                    failed = True
                    break
                channel_event_index = int(threshold_index[channel])
                if channel_event_index >= per_channel_capacity:
                    statuses[path_index] = JUMP_MAX_EVENTS
                    failed = True
                    break
                key = mark_keys[path_index, channel, channel_event_index]
                mark = np.asarray(
                    problem.process.sample_mark(
                        key,
                        jnp.asarray(current_time),
                        jnp.asarray(state),
                        jnp.asarray(channel, dtype=jnp.int32),
                        controlled_args,
                    )
                )
                if (
                    mark.shape != mark_shape
                    or np.issubdtype(mark.dtype, np.complexfloating)
                    or not np.all(np.isfinite(mark))
                ):
                    statuses[path_index] = JUMP_SOLVER_FAILURE
                    finite[path_index] = False
                    failed = True
                    break
                raw_post = np.asarray(
                    problem.process.jump(
                        jnp.asarray(state),
                        jnp.asarray(channel, dtype=jnp.int32),
                        jnp.asarray(mark),
                        controlled_args,
                    )
                )
                if np.issubdtype(raw_post.dtype, np.complexfloating):
                    statuses[path_index] = JUMP_SOLVER_FAILURE
                    finite[path_index] = False
                    failed = True
                    break
                post = np.asarray(raw_post, dtype=float)
                if post.shape != state_shape or not np.all(np.isfinite(post)):
                    statuses[path_index] = JUMP_SOLVER_FAILURE
                    finite[path_index] = False
                    failed = True
                    break
                event_times[path_index, event_index] = current_time
                event_channels[path_index, event_index] = channel
                event_marks[path_index, event_index] = mark
                event_valid[path_index, event_index] = True
                pre_states[path_index, event_index] = state
                post_states[path_index, event_index] = post
                state = post
                event_index += 1
                threshold_index[channel] += 1
                if (
                    event_index == event_capacity
                    or threshold_index[channel] == per_channel_capacity
                ):
                    statuses[path_index] = JUMP_MAX_EVENTS
                    failed = True
                    break
            states[path_index, step + 1] = state
            if failed:
                states[path_index, step + 1 :] = state
                break
        if np.isinf(minimum_intensity[path_index]):
            minimum_intensity[path_index] = 0.0

    reshaped_events = JumpEventBatch(
        jnp.asarray(event_times.reshape(batch_shape + (event_capacity,))),
        jnp.asarray(event_channels.reshape(batch_shape + (event_capacity,))),
        jnp.asarray(event_marks.reshape(batch_shape + (event_capacity,) + mark_shape)),
        jnp.asarray(event_valid.reshape(batch_shape + (event_capacity,))),
        jnp.asarray(statuses.reshape(batch_shape)),
        mark_shape=mark_shape,
        state_shape=state_shape,
        pre_states=jnp.asarray(
            pre_states.reshape(batch_shape + (event_capacity,) + state_shape)
        ),
        post_states=jnp.asarray(
            post_states.reshape(batch_shape + (event_capacity,) + state_shape)
        ),
    )
    capacity_complete = statuses != JUMP_MAX_EVENTS
    successful = statuses == JUMP_SUCCESS
    evidence = ControlledJumpEvidence(
        minimum_intensity=jnp.asarray(minimum_intensity.reshape(batch_shape)),
        maximum_intensity=jnp.asarray(maximum_intensity.reshape(batch_shape)),
        event_counts=reshaped_events.counts,
        finite=jnp.asarray(finite.reshape(batch_shape)),
        capacity_complete=jnp.asarray(capacity_complete.reshape(batch_shape)),
        successful=jnp.asarray(successful.reshape(batch_shape)),
        scope="declared-grid-piecewise-constant-control-finite-capacity-reference",
    )
    return ControlledJumpPathBatch(
        time_grid=plan.time_grid,
        states=jnp.asarray(states.reshape(batch_shape + (steps + 1,) + state_shape)),
        actions=jnp.asarray(
            actions.reshape(batch_shape + (steps,) + problem.action_shape)
        ),
        events=reshaped_events,
        evidence=evidence,
        valid=jnp.asarray(successful.reshape(batch_shape)),
        status=jnp.asarray(statuses.reshape(batch_shape)),
        batch_shape=batch_shape,
        state_shape=state_shape,
        action_shape=problem.action_shape,
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        policy_id=identifier,
        process_id=problem.process.process_id,
        realization_id=realization.realization_id,
        coupling_id=realization.coupling_id,
        same_time_order=plan.same_time_order,
    )


__all__ = [
    "ControlledJumpEvidence",
    "ControlledJumpPathBatch",
    "ControlledJumpPlan",
    "ControlledJumpPolicy",
    "ControlledJumpProblem",
    "rollout_controlled_jumps_reference",
]
