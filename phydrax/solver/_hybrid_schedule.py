#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._hybrid_event import HybridEventPlan, localize_hybrid_event


class ScheduledHybridEvent(StrictModule, NonTrainableState):
    event: HybridEventPlan
    direction: int = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    terminal: bool = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self, event: HybridEventPlan, /, *, direction=0, priority=0, terminal=False
    ):
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
    plan_id: str = eqx.field(static=True)


class HybridSchedulePlan(StrictModule, NonTrainableState):
    events: tuple[ScheduledHybridEvent, ...]
    maximum_events: int = eqx.field(static=True)
    simultaneous_tolerance: float = eqx.field(static=True)
    minimum_event_separation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        events: tuple[ScheduledHybridEvent, ...],
        /,
        *,
        maximum_events=64,
        simultaneous_tolerance=1.0e-10,
        minimum_event_separation=1.0e-12,
    ):
        items = tuple(events)
        if not items or len({item.event_id for item in items}) != len(items):
            raise ValueError("Hybrid schedule requires unique events.")
        if maximum_events <= 0:
            raise ValueError("maximum_events must be positive.")
        self.events = items
        self.maximum_events = int(maximum_events)
        self.simultaneous_tolerance = float(simultaneous_tolerance)
        self.minimum_event_separation = float(minimum_event_separation)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-schedule",
                "events": [item.event_id for item in items],
                "capacity": int(maximum_events),
            }
        )

    def localize(
        self,
        state_at_time: Callable[[Array, Any], Array],
        brackets: ArrayLike,
        args: Any = None,
        /,
    ) -> HybridScheduleResult:
        intervals = jnp.asarray(brackets)
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError("Event brackets must have shape (N,2).")
        sample = jnp.asarray(state_at_time(intervals[0, 0], args))
        state_shape = sample.shape
        times = jnp.full((self.maximum_events,), jnp.nan, dtype=intervals.dtype)
        indices = jnp.full((self.maximum_events,), -1, dtype=jnp.int32)
        before = jnp.zeros((self.maximum_events, *state_shape), dtype=sample.dtype)
        after = jnp.zeros_like(before)
        valid = jnp.zeros((self.maximum_events,), dtype=bool)
        terminal = jnp.zeros_like(valid)
        count = 0
        last_time = -jnp.inf
        for bracket_index in range(int(intervals.shape[0])):
            left, right = intervals[bracket_index]
            candidates = []
            for event_index, scheduled in enumerate(self.events):
                left_guard = scheduled.event.guard(state_at_time(left, args), args)
                right_guard = scheduled.event.guard(state_at_time(right, args), args)
                crossed = left_guard * right_guard <= 0.0
                direction_ok = (scheduled.direction == 0) or (
                    (scheduled.direction > 0 and right_guard > left_guard)
                    or (scheduled.direction < 0 and right_guard < left_guard)
                )
                result = localize_hybrid_event(
                    scheduled.event, state_at_time, left, right, args=args
                )
                candidates.append(
                    (event_index, scheduled, result, crossed & direction_ok)
                )
            selected = sorted(candidates, key=lambda item: (-item[1].priority, item[0]))
            for event_index, scheduled, result, crossed in selected:
                if count >= self.maximum_events:
                    break
                separated = result.event_time - last_time >= self.minimum_event_separation
                accepted = crossed & result.successful & separated
                times = times.at[count].set(
                    jnp.where(accepted, result.event_time, jnp.nan)
                )
                indices = indices.at[count].set(jnp.where(accepted, event_index, -1))
                before = before.at[count].set(
                    jnp.where(accepted, result.state_before, 0.0)
                )
                after = after.at[count].set(jnp.where(accepted, result.state_after, 0.0))
                valid = valid.at[count].set(accepted)
                terminal = terminal.at[count].set(accepted & scheduled.terminal)
                last_time = jnp.where(accepted, result.event_time, last_time)
                count += 1
                if bool(accepted & scheduled.terminal):
                    break
        event_count = jnp.sum(valid.astype(jnp.int32))
        capacity_exceeded = (
            int(intervals.shape[0]) * len(self.events)
        ) > self.maximum_events
        return HybridScheduleResult(
            times,
            indices,
            before,
            after,
            valid,
            terminal,
            event_count,
            jnp.asarray(capacity_exceeded),
            self.plan_id,
        )


__all__ = ["HybridSchedulePlan", "HybridScheduleResult", "ScheduledHybridEvent"]
