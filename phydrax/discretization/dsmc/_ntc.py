#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCStructuredCellPlan


class DSMCNTCReason(IntFlag):
    INVALID_PARTICLE = 1 << 8
    CELL_OCCUPANCY_EXCEEDED = 1 << 9
    UNEQUAL_CELL_WEIGHTS = 1 << 10
    EVENT_CAPACITY_EXCEEDED = 1 << 11
    INVALID_MAJORANT = 1 << 12
    INVALID_STEP_SIZE = 1 << 13


class DSMCCellOccupancy(StrictModule):
    slots: Array
    counts: Array
    simulator_weight: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DSMCNTCState(StrictModule):
    majorant_sigma_speed: Array
    fractional_remainder: Array
    epoch: Array
    plan_id: str = eqx.field(static=True)


class DSMCNTCSchedule(StrictModule):
    first_indices: Array
    second_indices: Array
    event_cells: Array
    valid_events: Array
    candidate_counts: Array
    occupancy: DSMCCellOccupancy
    candidate_state: DSMCNTCState
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DSMCNTCSchedulePlan(StrictModule, NonTrainableState):
    """Fixed-capacity, cell-local no-time-counter collision scheduler."""

    cells: DSMCStructuredCellPlan
    maximum_particles_per_cell: int = eqx.field(static=True)
    maximum_events_per_cell: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cells: DSMCStructuredCellPlan,
        /,
        *,
        maximum_particles_per_cell: int,
        maximum_events_per_cell: int,
    ) -> None:
        particle_capacity = int(maximum_particles_per_cell)
        event_capacity = int(maximum_events_per_cell)
        if (
            not isinstance(cells, DSMCStructuredCellPlan)
            or particle_capacity < 2
            or event_capacity <= 0
        ):
            raise ValueError("DSMC NTC cell and event capacities are invalid.")
        self.cells = cells
        self.maximum_particles_per_cell = particle_capacity
        self.maximum_events_per_cell = event_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-ntc-schedule",
                "cells": cells.plan_id,
                "maximum_particles_per_cell": particle_capacity,
                "maximum_events_per_cell": event_capacity,
            }
        )

    @property
    def event_capacity(self) -> int:
        return self.cells.cell_count * self.maximum_events_per_cell

    def initialize(self, majorant_sigma_speed: ArrayLike, /) -> DSMCNTCState:
        majorant = jnp.asarray(majorant_sigma_speed)
        if majorant.shape == ():
            majorant = jnp.full((self.cells.cell_count,), majorant)
        if majorant.shape != (self.cells.cell_count,):
            raise ValueError("DSMC NTC majorant must be scalar or one value per cell.")
        return DSMCNTCState(
            majorant,
            jnp.zeros_like(majorant),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def occupancy(self, particles: DSMCParticleState, /) -> DSMCCellOccupancy:
        capacity = particles.capacity
        cell_count = self.cells.cell_count
        index = jnp.arange(capacity, dtype=jnp.int32)
        valid_cell = (particles.cell_id >= 0) & (particles.cell_id < cell_count)
        finite_particle = (
            jnp.all(jnp.isfinite(particles.position), axis=-1)
            & jnp.all(jnp.isfinite(particles.velocity), axis=-1)
            & jnp.isfinite(particles.rotational_energy)
            & jnp.isfinite(particles.vibrational_energy)
            & jnp.isfinite(particles.statistical_weight)
        )
        valid_active = (
            particles.active
            & valid_cell
            & finite_particle
            & (particles.statistical_weight > 0.0)
        )
        invalid_active = particles.active & ~valid_active
        safe_cell = jnp.where(valid_active, particles.cell_id, 0)
        counts = (
            jnp.zeros((cell_count,), dtype=jnp.int32)
            .at[safe_cell]
            .add(valid_active.astype(jnp.int32))
        )
        earlier = index[None, :] < index[:, None]
        same_cell = safe_cell[None, :] == safe_cell[:, None]
        active_pair = valid_active[None, :] & valid_active[:, None]
        rank = jnp.sum(earlier & same_cell & active_pair, axis=1).astype(jnp.int32)
        storable = valid_active & (rank < self.maximum_particles_per_cell)
        safe_rank = jnp.clip(rank, 0, self.maximum_particles_per_cell - 1)
        stored_index = jnp.where(storable, index, -1)
        slots = -jnp.ones((cell_count, self.maximum_particles_per_cell), dtype=jnp.int32)
        slots = slots.at[safe_cell, safe_rank].max(stored_index)

        dtype = particles.statistical_weight.dtype
        infinity = jnp.asarray(jnp.inf, dtype=dtype)
        minimum = (
            jnp.full((cell_count,), infinity)
            .at[safe_cell]
            .min(jnp.where(valid_active, particles.statistical_weight, infinity))
        )
        maximum = (
            jnp.full((cell_count,), -infinity)
            .at[safe_cell]
            .max(jnp.where(valid_active, particles.statistical_weight, -infinity))
        )
        simulator_weight = jnp.where(counts > 0, minimum, 0.0)
        tolerance = 64.0 * jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(maximum), 1.0)
        equal_weight = (counts <= 1) | (jnp.abs(maximum - minimum) <= tolerance)
        occupancy_ok = counts <= self.maximum_particles_per_cell
        particles_ok = ~jnp.any(invalid_active)
        finite = jnp.all(jnp.isfinite(simulator_weight)) & particles_ok
        reasons = jnp.zeros((cell_count,), dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            particles_ok,
            reasons,
            reasons | jnp.asarray(int(DSMCNTCReason.INVALID_PARTICLE), jnp.uint32),
        )
        reasons = jnp.where(
            occupancy_ok,
            reasons,
            reasons | jnp.asarray(int(DSMCNTCReason.CELL_OCCUPANCY_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            equal_weight,
            reasons,
            reasons | jnp.asarray(int(DSMCNTCReason.UNEQUAL_CELL_WEIGHTS), jnp.uint32),
        )
        margin = jnp.minimum(
            (self.maximum_particles_per_cell - counts).astype(dtype)
            / self.maximum_particles_per_cell,
            jnp.where(equal_weight, 1.0, -1.0),
        )
        header = AdmissibilityHeader(
            margin,
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dsmc-cell-occupancy-evidence", "plan": self.plan_id}
            ),
        )
        return DSMCCellOccupancy(slots, counts, simulator_weight, header, self.plan_id)

    def schedule(
        self,
        particles: DSMCParticleState,
        state: DSMCNTCState,
        step_size: ArrayLike,
        key: PRNGKeyArray,
        /,
    ) -> DSMCNTCSchedule:
        if not isinstance(state, DSMCNTCState) or state.plan_id != self.plan_id:
            raise ValueError("DSMC NTC state does not belong to this schedule plan.")
        if jax.random.key_data(key).shape != (2,):
            raise ValueError("DSMC NTC scheduling requires one PRNG key.")
        occupancy = self.occupancy(particles)
        dtype = particles.velocity.dtype
        step = jnp.asarray(step_size, dtype=dtype)
        if step.shape != ():
            raise ValueError("DSMC NTC step size must be scalar.")
        majorant = state.majorant_sigma_speed.astype(dtype)
        count = occupancy.counts.astype(dtype)
        pair_measure = 0.5 * count * (count - 1.0)
        expected = pair_measure * occupancy.simulator_weight.astype(
            dtype
        ) * step * majorant / jnp.asarray(
            self.cells.cell_volumes, dtype=dtype
        ) + state.fractional_remainder.astype(dtype)
        finite_expected = jnp.isfinite(expected)
        maximum_integer = jnp.asarray(np.iinfo(np.int32).max, dtype=dtype)
        bounded = jnp.where(
            finite_expected,
            jnp.minimum(jnp.floor(expected), maximum_integer),
            maximum_integer,
        )
        candidate_counts = bounded.astype(jnp.int32)
        event_capacity_ok = candidate_counts <= self.maximum_events_per_cell
        majorant_ok = jnp.isfinite(majorant) & (majorant > 0.0)
        step_ok = jnp.isfinite(step) & (step > 0.0)
        reasons = occupancy.header.reason_bits
        reasons = jnp.where(
            finite_expected,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            event_capacity_ok,
            reasons,
            reasons | jnp.asarray(int(DSMCNTCReason.EVENT_CAPACITY_EXCEEDED), jnp.uint32),
        )
        reasons = jnp.where(
            majorant_ok,
            reasons,
            reasons | jnp.asarray(int(DSMCNTCReason.INVALID_MAJORANT), jnp.uint32),
        )
        reasons = jnp.where(
            step_ok,
            reasons,
            reasons | jnp.asarray(int(DSMCNTCReason.INVALID_STEP_SIZE), jnp.uint32),
        )
        capacity_margin = (self.maximum_events_per_cell - candidate_counts).astype(
            dtype
        ) / self.maximum_events_per_cell
        margin = jnp.minimum(occupancy.header.margin, capacity_margin)
        margin = jnp.minimum(margin, jnp.where(majorant_ok & step_ok, 1.0, -1.0))
        header = AdmissibilityHeader(
            margin,
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dsmc-ntc-schedule-evidence", "plan": self.plan_id}
            ),
        )

        random = jax.random.uniform(
            key,
            (self.cells.cell_count, self.maximum_events_per_cell, 2),
            dtype=dtype,
        )
        local_event = jnp.arange(self.maximum_events_per_cell, dtype=jnp.int32)
        local_event = jnp.broadcast_to(
            local_event[None, :],
            (self.cells.cell_count, self.maximum_events_per_cell),
        )
        cell = jnp.broadcast_to(
            jnp.arange(self.cells.cell_count, dtype=jnp.int32)[:, None],
            local_event.shape,
        )
        particle_count = occupancy.counts[:, None]
        first_rank = jnp.floor(random[..., 0] * jnp.maximum(particle_count, 1)).astype(
            jnp.int32
        )
        second_raw = jnp.floor(
            random[..., 1] * jnp.maximum(particle_count - 1, 1)
        ).astype(jnp.int32)
        second_rank = second_raw + (second_raw >= first_rank).astype(jnp.int32)
        first_rank = jnp.clip(first_rank, 0, self.maximum_particles_per_cell - 1)
        second_rank = jnp.clip(second_rank, 0, self.maximum_particles_per_cell - 1)
        first = occupancy.slots[cell, first_rank]
        second = occupancy.slots[cell, second_rank]
        valid = (
            (local_event < candidate_counts[:, None])
            & (particle_count >= 2)
            & header.eligible[:, None]
            & (first >= 0)
            & (second >= 0)
            & (first != second)
        )
        remainder = jnp.where(
            header.eligible,
            expected - candidate_counts.astype(dtype),
            state.fractional_remainder,
        )
        candidate_state = DSMCNTCState(
            majorant,
            remainder,
            state.epoch,
            self.plan_id,
        )
        return DSMCNTCSchedule(
            first.reshape((-1,)),
            second.reshape((-1,)),
            cell.reshape((-1,)),
            valid.reshape((-1,)),
            candidate_counts,
            occupancy,
            candidate_state,
            header,
            self.plan_id,
        )

    def update_majorant(
        self,
        state: DSMCNTCState,
        required_sigma_speed: ArrayLike,
        /,
        *,
        safety_factor: float = 1.05,
    ) -> DSMCNTCState:
        if not isinstance(state, DSMCNTCState) or state.plan_id != self.plan_id:
            raise ValueError("DSMC NTC state does not belong to this schedule plan.")
        factor = float(safety_factor)
        if not np.isfinite(factor) or factor <= 1.0:
            raise ValueError("DSMC majorant safety_factor must exceed one.")
        required = jnp.asarray(
            required_sigma_speed, dtype=state.majorant_sigma_speed.dtype
        )
        if required.shape != state.majorant_sigma_speed.shape:
            raise ValueError("Required DSMC majorants must have one value per cell.")
        majorant = jnp.maximum(state.majorant_sigma_speed, factor * required)
        return DSMCNTCState(
            majorant,
            state.fractional_remainder,
            state.epoch + 1,
            self.plan_id,
        )


__all__ = [
    "DSMCCellOccupancy",
    "DSMCNTCReason",
    "DSMCNTCSchedule",
    "DSMCNTCSchedulePlan",
    "DSMCNTCState",
]
