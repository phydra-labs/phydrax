#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._hybrid_event import (
    empty_hybrid_event_tape,
    HybridEventPlan,
    HybridEventTape,
    HybridReplayPolicy,
    HybridReplayResult,
    localize_hybrid_event,
    replay_hybrid_events,
)


class ScheduledHybridEvent(StrictModule, NonTrainableState):
    event: HybridEventPlan
    direction: int = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    terminal: bool = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self, event: HybridEventPlan, /, *, direction=0, priority=0, terminal=False
    ):
        if not isinstance(event, HybridEventPlan):
            raise TypeError("event must be a HybridEventPlan.")
        if direction not in (-1, 0, 1):
            raise ValueError("Scheduled event direction must be -1, zero, or +1.")
        self.event = event
        self.direction = int(direction)
        self.priority = int(priority)
        self.terminal = bool(terminal)
        self.event_id = canonical_fingerprint(
            {
                "kind": "scheduled-hybrid-event",
                "event": event.plan_id,
                "direction": int(direction),
                "priority": int(priority),
                "terminal": bool(terminal),
            }
        )


class HybridScheduleResult(StrictModule):
    event_times: Array
    event_indices: Array
    event_states_before: Array
    event_states_after: Array
    valid: Array
    terminal: Array
    event_count: Array
    capacity_exceeded: Array
    tape: HybridEventTape
    plan_id: str = eqx.field(static=True)


class HybridSchedulePlan(StrictModule, NonTrainableState):
    events: tuple[ScheduledHybridEvent, ...]
    maximum_events: int = eqx.field(static=True)
    simultaneous_tolerance: float = eqx.field(static=True)
    minimum_event_separation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        events: Sequence[ScheduledHybridEvent],
        /,
        *,
        maximum_events=64,
        simultaneous_tolerance=1.0e-10,
        minimum_event_separation=1.0e-12,
    ):
        items = tuple(events)
        if not items or any(not isinstance(item, ScheduledHybridEvent) for item in items):
            raise ValueError("Hybrid schedule requires scheduled event values.")
        if len({item.event_id for item in items}) != len(items):
            raise ValueError("Hybrid schedule requires unique events.")
        if not isinstance(maximum_events, int) or isinstance(maximum_events, bool):
            raise TypeError("maximum_events must be an integer.")
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive.")
        simultaneous = float(simultaneous_tolerance)
        separation = float(minimum_event_separation)
        if (
            not np.isfinite(simultaneous)
            or simultaneous <= 0.0
            or not np.isfinite(separation)
            or separation < 0.0
        ):
            raise ValueError("Hybrid schedule tolerances are invalid.")
        self.events = items
        self.maximum_events = maximum_events
        self.simultaneous_tolerance = simultaneous
        self.minimum_event_separation = separation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-schedule",
                "events": [item.event_id for item in items],
                "capacity": maximum_events,
                "simultaneous_tolerance": simultaneous,
                "minimum_event_separation": separation,
            }
        )

    def localize(
        self,
        state_at_time: Callable[[Array, Any], Array],
        brackets: ArrayLike,
        args: Any = None,
        /,
        *,
        replay_policy: HybridReplayPolicy | None = None,
    ) -> HybridScheduleResult:
        """Localize and commit the earliest event in each fixed-shape bracket.

        A committed reset ends that bracket. The next bracket therefore represents
        the continuous segment restarted from the recorded post-reset state.
        """

        intervals = jnp.asarray(brackets)
        if intervals.ndim != 2 or intervals.shape[1] != 2 or intervals.shape[0] == 0:
            raise ValueError("Event brackets must have nonempty shape (N,2).")
        if not callable(state_at_time):
            raise TypeError("state_at_time must be callable.")
        sample = jnp.asarray(state_at_time(intervals[0, 0], args))
        policy = (
            HybridReplayPolicy(
                self.maximum_events,
                grazing_tolerance=min(
                    item.event.grazing_tolerance for item in self.events
                ),
                simultaneous_tolerance=self.simultaneous_tolerance,
                event_tolerance=min(item.event.event_tolerance for item in self.events),
            )
            if replay_policy is None
            else replay_policy
        )
        if not isinstance(policy, HybridReplayPolicy):
            raise TypeError("replay_policy must be a HybridReplayPolicy or None.")
        if policy.maximum_events != self.maximum_events:
            raise ValueError("Schedule and replay event capacities must match exactly.")
        tape = empty_hybrid_event_tape(
            policy,
            sample,
            schedule_id=self.plan_id,
        )
        count = jnp.asarray(0, dtype=jnp.int32)
        last_time = jnp.asarray(-jnp.inf, dtype=intervals.dtype)
        stopped = jnp.asarray(False)
        capacity_exceeded = jnp.asarray(False)

        for bracket_index in range(intervals.shape[0]):
            left, right = intervals[bracket_index]
            candidates = []
            for event_index, scheduled in enumerate(self.events):
                left_state = state_at_time(left, args)
                right_state = state_at_time(right, args)
                left_guard = scheduled.event.guard(left, left_state, args)
                right_guard = scheduled.event.guard(right, right_state, args)
                crossed = left_guard * right_guard <= 0.0
                direction_ok = (
                    (scheduled.direction == 0)
                    | ((scheduled.direction > 0) & (right_guard > left_guard))
                    | ((scheduled.direction < 0) & (right_guard < left_guard))
                )
                localized = localize_hybrid_event(
                    scheduled.event,
                    state_at_time,
                    left,
                    right,
                    args=args,
                )
                candidates.append((localized, crossed & direction_ok))

            first_result = candidates[0][0]
            winner_exists = jnp.asarray(False)
            winner_index = jnp.asarray(0, dtype=jnp.int32)
            winner_time = jnp.asarray(jnp.inf, dtype=intervals.dtype)
            winner_priority = jnp.asarray(np.iinfo(np.int32).min, dtype=jnp.int32)
            winner_values = (
                first_result.event_time,
                first_result.state_before,
                first_result.state_after,
                first_result.guard_residual,
                first_result.transversality,
                first_result.successful,
                first_result.determinant_sign,
                first_result.log_abs_determinant,
                first_result.log_jacobian_valid,
            )
            winner_terminal = jnp.asarray(False)
            for event_index, (result, crossed) in enumerate(candidates):
                scheduled = self.events[event_index]
                eligible = crossed & result.successful
                simultaneous = (
                    jnp.abs(result.event_time - winner_time)
                    <= self.simultaneous_tolerance
                )
                earlier = result.event_time < winner_time - self.simultaneous_tolerance
                higher_priority = scheduled.priority > winner_priority
                choose = eligible & (
                    (~winner_exists) | earlier | (simultaneous & higher_priority)
                )
                winner_exists = winner_exists | eligible
                winner_index = jnp.where(choose, event_index, winner_index)
                winner_time = jnp.where(choose, result.event_time, winner_time)
                winner_priority = jnp.where(choose, scheduled.priority, winner_priority)
                winner_terminal = jnp.where(choose, scheduled.terminal, winner_terminal)
                candidate_values = (
                    result.event_time,
                    result.state_before,
                    result.state_after,
                    result.guard_residual,
                    result.transversality,
                    result.successful,
                    result.determinant_sign,
                    result.log_abs_determinant,
                    result.log_jacobian_valid,
                )
                winner_values = tuple(
                    jnp.where(choose, candidate, current)
                    for candidate, current in zip(
                        candidate_values, winner_values, strict=True
                    )
                )

            (
                winner_event_time,
                winner_state_before,
                winner_state_after,
                winner_guard_residual,
                winner_transversality,
                winner_successful,
                winner_determinant_sign,
                winner_log_abs_determinant,
                winner_log_jacobian_valid,
            ) = winner_values
            separated = winner_event_time - last_time >= self.minimum_event_separation
            accepted = (~stopped) & winner_exists & separated
            room = count < self.maximum_events
            capacity_exceeded = capacity_exceeded | (accepted & (~room))
            write = accepted & room
            slot = jnp.minimum(count, self.maximum_events - 1)
            tape = HybridEventTape(
                tape.event_indices.at[slot].set(
                    jnp.where(write, winner_index, tape.event_indices[slot])
                ),
                tape.event_times.at[slot].set(
                    jnp.where(write, winner_event_time, tape.event_times[slot])
                ),
                tape.states_before.at[slot].set(
                    jnp.where(write, winner_state_before, tape.states_before[slot])
                ),
                tape.states_after.at[slot].set(
                    jnp.where(write, winner_state_after, tape.states_after[slot])
                ),
                tape.guard_residuals.at[slot].set(
                    jnp.where(
                        write,
                        winner_guard_residual,
                        tape.guard_residuals[slot],
                    )
                ),
                tape.transversality.at[slot].set(
                    jnp.where(
                        write,
                        winner_transversality,
                        tape.transversality[slot],
                    )
                ),
                tape.saltation_valid.at[slot].set(
                    jnp.where(
                        write,
                        winner_successful,
                        tape.saltation_valid[slot],
                    )
                ),
                tape.determinant_signs.at[slot].set(
                    jnp.where(
                        write,
                        winner_determinant_sign,
                        tape.determinant_signs[slot],
                    )
                ),
                tape.log_abs_determinants.at[slot].set(
                    jnp.where(
                        write,
                        winner_log_abs_determinant,
                        tape.log_abs_determinants[slot],
                    )
                ),
                tape.log_jacobian_valid.at[slot].set(
                    jnp.where(
                        write,
                        winner_log_jacobian_valid,
                        tape.log_jacobian_valid[slot],
                    )
                ),
                tape.active.at[slot].set(write | tape.active[slot]),
                count + write.astype(jnp.int32),
                stopped | (write & winner_terminal),
                capacity_exceeded,
                jnp.where(capacity_exceeded, policy.failure, tape.status),
                tape.policy_id,
                tape.schedule_id,
            )
            count = tape.event_count
            last_time = jnp.where(write, winner_event_time, last_time)
            stopped = tape.terminal

        tape = eqx.tree_at(
            lambda value: (value.capacity_exceeded, value.status),
            tape,
            (
                capacity_exceeded,
                jnp.where(capacity_exceeded, policy.failure, tape.status),
            ),
        )
        return HybridScheduleResult(
            tape.event_times,
            tape.event_indices,
            tape.states_before,
            tape.states_after,
            tape.active,
            tape.active
            & jnp.arange(self.maximum_events)
            .astype(jnp.int32)
            .__eq__(jnp.maximum(tape.event_count - 1, 0))
            & tape.terminal,
            tape.event_count,
            tape.capacity_exceeded,
            tape,
            self.plan_id,
        )


class PreparedHybridSchedule(StrictModule, NonTrainableState):
    """Schedule bound to one state shape/dtype and canonical replay policy."""

    plan: HybridSchedulePlan
    replay_policy: HybridReplayPolicy
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dtype: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    @property
    def schedule_id(self) -> str:
        return self.plan.plan_id


def prepare_hybrid_schedule(
    plan: HybridSchedulePlan,
    state_template: ArrayLike,
    /,
    *,
    replay_policy: HybridReplayPolicy | None = None,
) -> PreparedHybridSchedule:
    """Bind exact event, capacity, state-shape, and replay identities."""

    if not isinstance(plan, HybridSchedulePlan):
        raise TypeError("plan must be a HybridSchedulePlan.")
    state = jnp.asarray(state_template)
    if not jnp.issubdtype(state.dtype, jnp.inexact):
        raise TypeError("Hybrid schedule states must use an inexact dtype.")
    policy = (
        HybridReplayPolicy(
            plan.maximum_events,
            grazing_tolerance=min(item.event.grazing_tolerance for item in plan.events),
            simultaneous_tolerance=plan.simultaneous_tolerance,
            event_tolerance=min(item.event.event_tolerance for item in plan.events),
        )
        if replay_policy is None
        else replay_policy
    )
    if not isinstance(policy, HybridReplayPolicy):
        raise TypeError("replay_policy must be a HybridReplayPolicy or None.")
    if policy.maximum_events != plan.maximum_events:
        raise ValueError("Schedule and replay event capacities must match exactly.")
    preparation_id = canonical_fingerprint(
        {
            "kind": "prepared-hybrid-schedule",
            "schedule": plan.plan_id,
            "replay": policy.policy_id,
            "state_shape": list(state.shape),
            "state_dtype": np.dtype(state.dtype).str,
        }
    )
    return PreparedHybridSchedule(
        plan,
        policy,
        tuple(state.shape),
        np.dtype(state.dtype).str,
        preparation_id,
    )


def execute_hybrid_schedule(
    prepared: PreparedHybridSchedule,
    state_at_time: Callable[[Array, Any], Array],
    brackets: ArrayLike,
    /,
    *,
    args: Any = None,
) -> HybridScheduleResult:
    """Run one prepared fixed-epoch schedule and return its canonical tape."""

    if not isinstance(prepared, PreparedHybridSchedule):
        raise TypeError("prepared must be a PreparedHybridSchedule.")
    sample = jnp.asarray(state_at_time(jnp.asarray(brackets)[0, 0], args))
    if (
        tuple(sample.shape) != prepared.state_shape
        or np.dtype(sample.dtype).str != prepared.state_dtype
    ):
        raise ValueError("Runtime state shape/dtype does not match prepared schedule.")
    result = prepared.plan.localize(
        state_at_time,
        brackets,
        args,
        replay_policy=prepared.replay_policy,
    )
    if result.tape.policy_id != prepared.replay_policy.policy_id:
        raise ValueError(
            "Runtime tape identity does not match the prepared replay policy."
        )
    return result


def replay_hybrid_schedule(
    prepared: PreparedHybridSchedule,
    tape: HybridEventTape,
    initial_state: ArrayLike,
    /,
    *,
    args: Any = None,
) -> HybridReplayResult:
    """Replay only an identical prepared schedule/tape identity."""

    if not isinstance(prepared, PreparedHybridSchedule):
        raise TypeError("prepared must be a PreparedHybridSchedule.")
    if (
        tape.schedule_id != prepared.schedule_id
        or tape.policy_id != prepared.replay_policy.policy_id
    ):
        raise ValueError("Prepared schedule and HybridEventTape identities do not match.")
    return replay_hybrid_events(
        tuple(item.event for item in prepared.plan.events),
        tape,
        initial_state,
        args=args,
    )


__all__ = [
    "HybridSchedulePlan",
    "HybridScheduleResult",
    "PreparedHybridSchedule",
    "ScheduledHybridEvent",
    "execute_hybrid_schedule",
    "prepare_hybrid_schedule",
    "replay_hybrid_schedule",
]
