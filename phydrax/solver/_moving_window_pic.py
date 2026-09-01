#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ..discretization.particle import (
    ParticleAllocationRequest,
    ParticlePopulationPlan,
    ParticlePopulationState,
)
from ..discretization.pic import PICParticleState
from ._maxwell import CompatibleMaxwellState, MaxwellPrimaryState


def _shift_without_wrap(value: Array, axis: int, cells: int, /) -> Array:
    shifted = jnp.roll(value, -cells, axis=axis)
    index = [slice(None)] * value.ndim
    index[axis] = slice(value.shape[axis] - cells, value.shape[axis])
    return shifted.at[tuple(index)].set(0.0)


class PICMovingWindowState(StrictModule):
    particles: PICParticleState
    population: ParticlePopulationState
    maxwell: CompatibleMaxwellState
    origin: Array
    cumulative_cells: Array
    shift_epoch: Array


class PICMovingWindowResult(StrictModule):
    candidate_state: PICMovingWindowState
    accepted_state: PICMovingWindowState
    shifted: Array
    outflow_mask: Array
    outflow_mass: Array
    outflow_charge: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PICMovingWindowPlan(StrictModule, NonTrainableState):
    bridge: StructuredCochainBridge
    axis: int = eqx.field(static=True)
    shift_cells: int = eqx.field(static=True)
    interval: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        axis: int,
        /,
        *,
        shift_cells: int = 1,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        selected = int(axis)
        cells = int(shift_cells)
        if selected < 0 or selected >= bridge.dimension or cells <= 0:
            raise ValueError("Moving-window axis/cell shift is invalid.")
        widths = np.asarray(bridge.grid.structured_axes[selected].interval_widths)
        if not np.allclose(widths, widths[0]) or cells >= widths.size:
            raise ValueError(
                "Moving window requires a uniform axis and a bounded cell shift."
            )
        self.bridge = bridge
        self.axis = selected
        self.shift_cells = cells
        self.interval = float(widths[0])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-moving-window",
                "bridge": bridge.bridge_id,
                "axis": selected,
                "shift_cells": cells,
            }
        )

    def initialize(
        self,
        particles: PICParticleState,
        population: ParticlePopulationState,
        maxwell: CompatibleMaxwellState,
        /,
    ) -> PICMovingWindowState:
        return PICMovingWindowState(
            particles,
            population,
            maxwell,
            jnp.asarray(0.0, dtype=particles.position.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def _shift_cochain(self, degree: int, value: Array, /) -> Array:
        components = self.bridge.unpack(degree, value)
        shifted = tuple(
            _shift_without_wrap(component, self.axis, self.shift_cells)
            for component in components
        )
        return self.bridge.pack(degree, shifted)

    def _shift_auxiliary_leaf(self, value):
        if not eqx.is_array(value):
            return value
        if value.shape == (self.bridge.cochain.cell_counts[0],):
            return self._shift_cochain(0, value)
        if value.shape == (self.bridge.cochain.cell_counts[1],):
            return self._shift_cochain(1, value)
        if value.shape == (self.bridge.cochain.cell_counts[2],):
            return self._shift_cochain(2, value)
        return value

    def shift(
        self,
        state: PICMovingWindowState,
        population_plan: ParticlePopulationPlan,
        macrocharge: ArrayLike,
        /,
        *,
        apply_shift: ArrayLike = True,
        injection_request: ParticleAllocationRequest | None = None,
        injection_position: ArrayLike | None = None,
        injection_velocity: ArrayLike | None = None,
    ) -> PICMovingWindowResult:
        predicate = jnp.asarray(apply_shift, dtype=bool).reshape(())
        distance = self.shift_cells * self.interval
        position = state.particles.position.at[:, self.axis].add(-distance)
        lower = self.bridge.grid.structured_axes[self.axis].bounds[0]
        outflow = state.population.active & (position[:, self.axis] < lower)
        deactivated = population_plan.deactivate(state.population, outflow)
        shifted_position = jnp.where(
            deactivated.accepted_state.active[:, None], position, 0.0
        )
        shifted_velocity = jnp.where(
            deactivated.accepted_state.active[:, None],
            state.particles.proper_velocity,
            0.0,
        )
        next_population = deactivated.accepted_state
        injection_success = jnp.asarray(True)
        if injection_request is not None:
            if injection_position is None or injection_velocity is None:
                raise ValueError(
                    "Moving-window injection requires position and velocity payloads."
                )
            injected_position = jnp.asarray(
                injection_position, dtype=shifted_position.dtype
            )
            injected_velocity = jnp.asarray(
                injection_velocity, dtype=shifted_velocity.dtype
            )
            width = injection_request.valid.shape[0]
            if injected_position.shape != (
                width,
                shifted_position.shape[1],
            ) or injected_velocity.shape != (width, 3):
                raise ValueError(
                    "Moving-window injection payloads must match request capacity."
                )
            allocation = population_plan.allocate(
                deactivated.accepted_state, injection_request
            )
            slots = jnp.maximum(allocation.slots, 0)
            use = allocation.allocated
            shifted_position = shifted_position.at[slots].set(
                jnp.where(
                    use[:, None], injected_position, shifted_position[slots]
                )
            )
            shifted_velocity = shifted_velocity.at[slots].set(
                jnp.where(
                    use[:, None], injected_velocity, shifted_velocity[slots]
                )
            )
            next_population = allocation.accepted_state
            injection_success = allocation.successful
        shifted_particles = PICParticleState(
            shifted_position,
            shifted_velocity,
        )
        primary = MaxwellPrimaryState(
            self._shift_cochain(1, state.maxwell.primary.electric_displacement),
            self._shift_cochain(2, state.maxwell.primary.magnetic_flux),
            self._shift_cochain(0, state.maxwell.primary.charge),
        )
        auxiliary = jax.tree.map(self._shift_auxiliary_leaf, state.maxwell.auxiliary)
        observations = jax.tree.map(
            self._shift_auxiliary_leaf, state.maxwell.observations
        )
        shifted_maxwell = CompatibleMaxwellState(primary, auxiliary, observations)
        candidate = PICMovingWindowState(
            shifted_particles,
            next_population,
            shifted_maxwell,
            state.origin + distance,
            state.cumulative_cells + self.shift_cells,
            state.shift_epoch + 1,
        )
        charge = jnp.asarray(macrocharge, dtype=state.origin.dtype)
        outflow_mass = jnp.sum(jnp.where(outflow, state.population.mass, 0.0))
        outflow_charge = jnp.sum(jnp.where(outflow, charge, 0.0))
        finite = (
            jnp.all(jnp.isfinite(candidate.particles.position))
            & jnp.all(jnp.isfinite(candidate.maxwell.primary.electric_displacement))
            & jnp.all(jnp.isfinite(candidate.maxwell.primary.magnetic_flux))
        )
        successful = deactivated.successful & injection_success & finite
        select = predicate & successful
        accepted = jax.tree.map(
            lambda proposed, old: jnp.where(select, proposed, old), candidate, state
        )
        return PICMovingWindowResult(
            candidate,
            accepted,
            select,
            outflow,
            outflow_mass,
            outflow_charge,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "PICMovingWindowPlan",
    "PICMovingWindowResult",
    "PICMovingWindowState",
]
