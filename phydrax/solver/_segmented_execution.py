#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


Carry = TypeVar("Carry")


class FixedCapacitySegmentPolicy(StrictModule, NonTrainableState):
    """Static whole-solve capacities for one compiled segmented execution."""

    maximum_segments: int = eqx.field(static=True)
    maximum_steps_per_segment: int = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    failure: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_segments: int,
        maximum_steps_per_segment: int,
        maximum_events: int = 0,
        /,
        *,
        failure: int = -1,
    ):
        capacities = (
            maximum_segments,
            maximum_steps_per_segment,
            maximum_events,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in capacities
        ):
            raise TypeError("segment capacities must be integers.")
        if maximum_segments <= 0 or maximum_steps_per_segment <= 0:
            raise ValueError("segment and step capacities must be positive.")
        if maximum_events < 0:
            raise ValueError("maximum_events must be nonnegative.")
        if not isinstance(failure, int) or isinstance(failure, bool):
            raise TypeError("failure must be an integer terminal status.")
        self.maximum_segments = maximum_segments
        self.maximum_steps_per_segment = maximum_steps_per_segment
        self.maximum_events = maximum_events
        self.failure = failure
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-segments",
                "maximum_segments": maximum_segments,
                "maximum_steps_per_segment": maximum_steps_per_segment,
                "maximum_events": maximum_events,
                "failure": failure,
            }
        )


class FixedCapacitySegmentStep(StrictModule, Generic[Carry]):
    """One physical owner's same-structure segment transition."""

    carry: Carry
    segment_start: Array
    segment_end: Array
    step_count: Array
    event_count: Array
    terminal: Array
    status: Array

    def __init__(
        self,
        carry: Carry,
        segment_start: ArrayLike,
        segment_end: ArrayLike,
        step_count: ArrayLike,
        event_count: ArrayLike = 0,
        terminal: ArrayLike = False,
        status: ArrayLike = 0,
        /,
    ):
        start = jnp.asarray(segment_start)
        end = jnp.asarray(segment_end)
        steps = jnp.asarray(step_count, dtype=jnp.int32)
        events = jnp.asarray(event_count, dtype=jnp.int32)
        terminal_ = jnp.asarray(terminal, dtype=bool)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if (
            start.shape
            or end.shape
            or steps.shape
            or events.shape
            or terminal_.shape
            or status_.shape
        ):
            raise ValueError("segment step metadata must be scalar arrays.")
        self.carry = carry
        self.segment_start = start
        self.segment_end = end
        self.step_count = steps
        self.event_count = events
        self.terminal = terminal_
        self.status = status_


class FixedCapacitySegmentEvidence(StrictModule, NonTrainableState):
    """Fixed-shape work and terminal evidence for a bounded segment run."""

    segment_starts: Array
    segment_ends: Array
    step_counts: Array
    event_counts: Array
    active: Array
    segment_count: Array
    terminal_status: Array
    capacity_exceeded: Array
    policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (~self.capacity_exceeded) & (self.terminal_status >= 0)


def run_fixed_capacity_segments(
    policy: FixedCapacitySegmentPolicy,
    initial_carry: Carry,
    advance_segment: Callable[[Carry, Array], FixedCapacitySegmentStep[Carry]],
    /,
) -> tuple[Carry, FixedCapacitySegmentEvidence]:
    """Execute a runtime number of segments inside one fixed-capacity JAX loop.

    ``advance_segment`` is called only while the physical solve is active. It must
    return a same-PyTree-structure carry and scalar metadata. Reaching the static
    segment, step, or event capacity before a terminal transition is a failed solve,
    never a partial success.
    """

    if not isinstance(policy, FixedCapacitySegmentPolicy):
        raise TypeError("policy must be a FixedCapacitySegmentPolicy.")
    if not callable(advance_segment):
        raise TypeError("advance_segment must be callable.")

    n = policy.maximum_segments
    starts = jnp.zeros((n,))
    ends = jnp.zeros((n,))
    step_counts = jnp.zeros((n,), dtype=jnp.int32)
    event_counts = jnp.zeros((n,), dtype=jnp.int32)
    active = jnp.zeros((n,), dtype=bool)
    initial = (
        initial_carry,
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        starts,
        ends,
        step_counts,
        event_counts,
        active,
        jnp.asarray(False),
    )

    def body(index: int, loop_state: tuple[Any, ...]) -> tuple[Any, ...]:
        (
            carry,
            terminal,
            status,
            count,
            total_events,
            starts_,
            ends_,
            steps_,
            events_,
            active_,
            failed,
        ) = loop_state

        def advance(_: None) -> tuple[Any, ...]:
            step = advance_segment(carry, jnp.asarray(index, dtype=jnp.int32))
            if not isinstance(step, FixedCapacitySegmentStep):
                raise TypeError("advance_segment must return FixedCapacitySegmentStep.")
            within_steps = (step.step_count >= 0) & (
                step.step_count <= policy.maximum_steps_per_segment
            )
            next_total_events = total_events + step.event_count
            within_events = (step.event_count >= 0) & (
                next_total_events <= policy.maximum_events
            )
            finite = jnp.isfinite(step.segment_start) & jnp.isfinite(step.segment_end)
            monotone = step.segment_end >= step.segment_start
            valid = within_steps & within_events & finite & monotone
            next_failed = failed | (~valid)
            next_terminal = step.terminal & valid
            next_status = jnp.where(valid, step.status, policy.failure)
            return (
                step.carry,
                next_terminal,
                next_status,
                count + jnp.asarray(1, dtype=jnp.int32),
                jnp.where(valid, next_total_events, total_events),
                starts_.at[index].set(step.segment_start),
                ends_.at[index].set(step.segment_end),
                steps_.at[index].set(step.step_count),
                events_.at[index].set(step.event_count),
                active_.at[index].set(True),
                next_failed,
            )

        return jax.lax.cond(terminal | failed, lambda _: loop_state, advance, None)

    final = jax.lax.fori_loop(0, n, body, initial)
    (
        carry,
        terminal,
        status,
        count,
        _total_events,
        starts,
        ends,
        step_counts,
        event_counts,
        active,
        failed,
    ) = final
    capacity_exceeded = failed | (~terminal)
    terminal_status = jnp.where(capacity_exceeded, policy.failure, status)
    evidence = FixedCapacitySegmentEvidence(
        segment_starts=starts,
        segment_ends=ends,
        step_counts=step_counts,
        event_counts=event_counts,
        active=active,
        segment_count=count,
        terminal_status=terminal_status,
        capacity_exceeded=capacity_exceeded,
        policy_id=policy.policy_id,
    )
    return carry, evidence


__all__ = [
    "FixedCapacitySegmentEvidence",
    "FixedCapacitySegmentPolicy",
    "FixedCapacitySegmentStep",
    "run_fixed_capacity_segments",
]
