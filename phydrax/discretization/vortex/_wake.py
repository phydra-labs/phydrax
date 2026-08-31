#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._filament import VortexFilamentState, VortexFilamentTopology


class VortexWakeState(StrictModule):
    start: Array
    end: Array
    circulation: Array
    core_radius: Array
    age: Array
    active: Array
    bound_circulation: Array
    step_index: Array

    def as_filaments(self, /) -> VortexFilamentState:
        capacity = int(self.start.shape[0])
        vertices = jnp.concatenate((self.start, self.end), axis=0)
        segment_index = jnp.arange(capacity, dtype=jnp.int32)
        start_index = jnp.where(self.active, segment_index, 0)
        end_index = jnp.where(self.active, capacity + segment_index, 0)
        topology = VortexFilamentTopology(
            2 * capacity,
            start_index,
            end_index,
            active=self.active,
            segment_ids=jnp.where(
                self.active,
                jnp.arange(capacity, dtype=jnp.int64),
                -1,
            ),
        )
        return VortexFilamentState(
            topology,
            vertices,
            jnp.where(self.active, self.circulation, 0.0),
            jnp.where(self.active, self.core_radius, 1.0),
        )


class VortexWakeTransition(StrictModule):
    candidate: VortexWakeState
    accepted: VortexWakeState
    emitted_circulation: Array
    overflow_count: Array
    circulation_residual: Array
    successful: Array
    wake_id: str = eqx.field(static=True)


class VortexWakePlan(StrictModule, NonTrainableState):
    """Fixed-capacity trailing-filament wake with atomic emission."""

    segment_capacity: int = eqx.field(static=True)
    source_count: int = eqx.field(static=True)
    core_radius: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, segment_capacity: int, source_count: int, core_radius: float, /):
        capacity = int(segment_capacity)
        sources = int(source_count)
        core = float(core_radius)
        if capacity <= 0 or sources <= 0 or sources > capacity or core <= 0.0:
            raise ValueError("Vortex wake capacities/core radius are invalid.")
        self.segment_capacity = capacity
        self.source_count = sources
        self.core_radius = core
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-wake-plan",
                "segment_capacity": capacity,
                "source_count": sources,
                "core_radius": core,
            }
        )

    def initialize(
        self, bound_circulation: ArrayLike, /, *, dtype=None
    ) -> VortexWakeState:
        bound = jnp.asarray(bound_circulation, dtype=dtype)
        if bound.shape != (self.source_count,):
            raise ValueError("bound_circulation must have source_count shape.")
        zeros = jnp.zeros((self.segment_capacity, 3), dtype=bound.dtype)
        return VortexWakeState(
            zeros,
            zeros,
            jnp.zeros((self.segment_capacity,), dtype=bound.dtype),
            jnp.full((self.segment_capacity,), self.core_radius, dtype=bound.dtype),
            jnp.zeros((self.segment_capacity,), dtype=bound.dtype),
            jnp.zeros((self.segment_capacity,), dtype=bool),
            bound,
            jnp.asarray(0, dtype=jnp.int32),
        )

    def shed(
        self,
        state: VortexWakeState,
        trailing_start: ArrayLike,
        trailing_end: ArrayLike,
        new_bound_circulation: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> VortexWakeTransition:
        start = jnp.asarray(trailing_start, dtype=state.start.dtype)
        end = jnp.asarray(trailing_end, dtype=state.end.dtype)
        bound = jnp.asarray(new_bound_circulation, dtype=state.circulation.dtype)
        dt = jnp.asarray(time_step, dtype=state.age.dtype)
        if (
            start.shape != (self.source_count, 3)
            or end.shape != start.shape
            or bound.shape != (self.source_count,)
        ):
            raise ValueError("Wake emission arrays do not match source_count.")
        if dt.shape != ():
            raise ValueError("time_step must be scalar.")
        emission = state.bound_circulation - bound
        free_slots = ~state.active
        free_order = jnp.nonzero(free_slots, size=self.segment_capacity, fill_value=-1)[0]
        selected = free_order[: self.source_count]
        valid_selected = selected >= 0
        overflow = jnp.sum(~valid_selected, dtype=jnp.int32)
        safe_selected = jnp.where(valid_selected, selected, 0)
        candidate_start = state.start.at[safe_selected].set(
            jnp.where(valid_selected[:, None], start, state.start[safe_selected])
        )
        candidate_end = state.end.at[safe_selected].set(
            jnp.where(valid_selected[:, None], end, state.end[safe_selected])
        )
        candidate_gamma = state.circulation.at[safe_selected].set(
            jnp.where(valid_selected, emission, state.circulation[safe_selected])
        )
        candidate_core = state.core_radius.at[safe_selected].set(
            jnp.where(valid_selected, self.core_radius, state.core_radius[safe_selected])
        )
        candidate_active = state.active.at[safe_selected].set(
            jnp.where(valid_selected, True, state.active[safe_selected])
        )
        candidate_age = jnp.where(state.active, state.age + dt, state.age)
        candidate_age = candidate_age.at[safe_selected].set(
            jnp.where(valid_selected, 0.0, candidate_age[safe_selected])
        )
        finite = (
            jnp.all(jnp.isfinite(start))
            & jnp.all(jnp.isfinite(end))
            & jnp.all(jnp.isfinite(bound))
            & jnp.isfinite(dt)
            & (dt > 0.0)
        )
        residual = jnp.sum(emission) + jnp.sum(bound) - jnp.sum(state.bound_circulation)
        successful = (
            (overflow == 0)
            & finite
            & (
                jnp.abs(residual)
                <= 64
                * jnp.finfo(bound.dtype).eps
                * jnp.maximum(jnp.sum(jnp.abs(state.bound_circulation)), 1.0)
            )
        )
        candidate = VortexWakeState(
            candidate_start,
            candidate_end,
            candidate_gamma,
            candidate_core,
            candidate_age,
            candidate_active,
            bound,
            state.step_index + 1,
        )
        accepted = jax_tree_select(successful, candidate, state)
        return VortexWakeTransition(
            candidate,
            accepted,
            jnp.sum(emission),
            overflow,
            residual,
            successful,
            self.plan_id,
        )


def jax_tree_select(
    condition: ArrayLike, candidate: VortexWakeState, previous: VortexWakeState, /
) -> VortexWakeState:
    condition_ = jnp.asarray(condition, dtype=bool)
    return VortexWakeState(
        jnp.where(condition_, candidate.start, previous.start),
        jnp.where(condition_, candidate.end, previous.end),
        jnp.where(condition_, candidate.circulation, previous.circulation),
        jnp.where(condition_, candidate.core_radius, previous.core_radius),
        jnp.where(condition_, candidate.age, previous.age),
        jnp.where(condition_, candidate.active, previous.active),
        jnp.where(condition_, candidate.bound_circulation, previous.bound_circulation),
        jnp.where(condition_, candidate.step_index, previous.step_index),
    )


__all__ = ["VortexWakePlan", "VortexWakeState", "VortexWakeTransition"]
