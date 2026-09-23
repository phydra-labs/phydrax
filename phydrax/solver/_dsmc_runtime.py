#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from .._admissibility import AdmissibilityHeader, AdmissibilityReason
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.dsmc._boundaries import (
    DSMCReservoirFacePlan,
    DSMCReservoirState,
    DSMCSurfaceBoundaryPlan,
)
from ..discretization.dsmc._chemistry import DSMCInternalReactionPlan
from ..discretization.dsmc._collisions import (
    DSMCElasticCollisionPlan,
    DSMCVHSCollisionPlan,
    DSMCVSSCollisionPlan,
)
from ..discretization.dsmc._core import DSMCParticleState, DSMCStreamingPlan
from ..discretization.dsmc._moments import (
    DSMCMomentAccumulatorState,
    DSMCMomentEvaluation,
    DSMCMomentPlan,
    DSMCStatisticalEvidence,
)
from ..discretization.dsmc._ntc import (
    DSMCNTCReason,
    DSMCNTCSchedule,
    DSMCNTCSchedulePlan,
    DSMCNTCState,
)


class DSMCBoundaryExchangeLedger(StrictModule):
    removed_count: Array
    removed_mass: Array
    removed_momentum: Array
    removed_energy: Array
    injected_count: Array
    injected_mass: Array
    injected_momentum: Array
    injected_energy: Array
    surface_mass: Array
    surface_momentum: Array
    surface_energy: Array
    net_mass: Array
    net_momentum: Array
    net_energy: Array


class DSMCRuntimeState(StrictModule):
    particles: DSMCParticleState
    ntc: DSMCNTCState
    moment_accumulator: DSMCMomentAccumulatorState
    reservoir_states: tuple[DSMCReservoirState, ...]
    time: Array
    accepted_steps: Array
    key: Array
    runtime_id: str = eqx.field(static=True)


class DSMCStepResult(StrictModule):
    candidate: DSMCRuntimeState
    accepted: DSMCRuntimeState
    moments: DSMCMomentEvaluation
    statistics: DSMCStatisticalEvidence
    schedule: DSMCNTCSchedule
    boundary_exchange: DSMCBoundaryExchangeLedger
    required_majorant_sigma_speed: Array
    collision_count: Array
    reaction_count: Array
    relaxation_count: Array
    finite: Array
    header: AdmissibilityHeader
    successful: Array
    plan_id: str = eqx.field(static=True)


class DSMCProductionPlan(StrictModule, NonTrainableState):
    """Cell-local NTC DSMC with interleaved accepted-pair internal physics."""

    streaming: DSMCStreamingPlan
    collisions: DSMCElasticCollisionPlan
    internal: DSMCInternalReactionPlan
    ntc: DSMCNTCSchedulePlan
    moments: DSMCMomentPlan
    reservoirs: tuple[DSMCReservoirFacePlan, ...]
    surface_boundaries: tuple[DSMCSurfaceBoundaryPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        streaming: DSMCStreamingPlan,
        collisions: DSMCElasticCollisionPlan,
        internal: DSMCInternalReactionPlan,
        ntc: DSMCNTCSchedulePlan,
        moments: DSMCMomentPlan,
        /,
        *,
        surface_boundaries: Sequence[DSMCSurfaceBoundaryPlan] = (),
        reservoirs: Sequence[DSMCReservoirFacePlan] = (),
    ) -> None:
        reservoirs_ = tuple(reservoirs)
        surfaces = tuple(surface_boundaries)
        if (
            not isinstance(streaming, DSMCStreamingPlan)
            or not isinstance(collisions, (DSMCVHSCollisionPlan, DSMCVSSCollisionPlan))
            or not isinstance(internal, DSMCInternalReactionPlan)
            or not isinstance(ntc, DSMCNTCSchedulePlan)
            or not isinstance(moments, DSMCMomentPlan)
            or collisions.species.plan_id != internal.species.plan_id
            or collisions.species.plan_id != moments.species.plan_id
            or streaming.cells.plan_id != ntc.cells.plan_id
            or streaming.cells.plan_id != moments.cells.plan_id
            or any(not isinstance(value, DSMCSurfaceBoundaryPlan) for value in surfaces)
            or any(
                value.interaction.species.plan_id != collisions.species.plan_id
                for value in surfaces
            )
            or any(
                value.face.cells.plan_id != streaming.cells.plan_id for value in surfaces
            )
            or any(not isinstance(value, DSMCReservoirFacePlan) for value in reservoirs_)
            or any(
                value.species.plan_id != collisions.species.plan_id
                for value in reservoirs_
            )
            or any(
                value.face.cells.plan_id != streaming.cells.plan_id
                for value in reservoirs_
            )
        ):
            raise ValueError("DSMC production plans are incompatible.")
        face_keys = tuple(
            (value.face.axis, value.face.side) for value in (*surfaces, *reservoirs_)
        )
        if len(set(face_keys)) != len(face_keys):
            raise ValueError("A DSMC boundary face may own at most one physical plan.")
        declared_surfaces = {
            (axis, side)
            for axis, pair in enumerate(streaming.boundary_kinds)
            for side, kind in zip(("lower", "upper"), pair, strict=True)
            if kind == "surface"
        }
        declared_reservoirs = {
            (axis, side)
            for axis, pair in enumerate(streaming.boundary_kinds)
            for side, kind in zip(("lower", "upper"), pair, strict=True)
            if kind == "reservoir"
        }
        registered_surfaces = {(value.face.axis, value.face.side) for value in surfaces}
        registered_reservoirs = {
            (value.face.axis, value.face.side) for value in reservoirs_
        }
        if (
            registered_surfaces != declared_surfaces
            or registered_reservoirs != declared_reservoirs
        ):
            raise ValueError(
                "Every DSMC surface and reservoir stream face requires one physical plan."
            )
        for reservoir in reservoirs_:
            boundary = streaming.boundary_kinds[reservoir.face.axis]
            kind = boundary[0 if reservoir.face.side == "lower" else 1]
            if kind != "reservoir":
                raise ValueError(
                    "Every DSMC reservoir must bind a reservoir stream face."
                )
        for surface in surfaces:
            boundary = streaming.boundary_kinds[surface.face.axis]
            kind = boundary[0 if surface.face.side == "lower" else 1]
            if kind != "surface":
                raise ValueError(
                    "Every DSMC wall kernel must bind a surface stream face."
                )
        self.streaming = streaming
        self.collisions = collisions
        self.internal = internal
        self.ntc = ntc
        self.moments = moments
        self.surface_boundaries = surfaces
        self.reservoirs = reservoirs_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-production",
                "streaming": streaming.plan_id,
                "collisions": collisions.plan_id,
                "internal": internal.plan_id,
                "ntc": ntc.plan_id,
                "moments": moments.plan_id,
                "reservoirs": tuple(value.plan_id for value in reservoirs_),
                "surface_boundaries": tuple(value.plan_id for value in surfaces),
            }
        )

    def initialize(
        self,
        particles: DSMCParticleState,
        key: PRNGKeyArray,
        /,
        *,
        majorant_sigma_speed: ArrayLike,
    ) -> DSMCRuntimeState:
        key_data = jax.random.key_data(key)
        if particles.position.shape[0] <= 1 or key_data.shape != (2,):
            raise ValueError(
                "DSMC initialization requires particle capacity and PRNG key."
            )
        active = np.asarray(particles.active)
        species = np.asarray(particles.species_index)
        cell = np.asarray(particles.cell_id)
        weight = np.asarray(particles.statistical_weight)
        finite = (
            np.all(np.isfinite(np.asarray(particles.position)))
            and np.all(np.isfinite(np.asarray(particles.velocity)))
            and np.all(np.isfinite(np.asarray(particles.rotational_energy)))
            and np.all(np.isfinite(np.asarray(particles.vibrational_energy)))
            and np.all(np.isfinite(weight))
        )
        active_valid = (
            np.all(
                (species[active] >= 0)
                & (species[active] < self.collisions.species.species_count)
            )
            and np.all(
                (cell[active] >= 0) & (cell[active] < self.streaming.cells.cell_count)
            )
            and np.all(weight[active] > 0.0)
        )
        inactive_valid = np.all(cell[~active] == -1) and np.all(weight[~active] == 0.0)
        if not finite or not active_valid or not inactive_valid:
            raise ValueError(
                "DSMC particle slots violate the active/inactive state contract."
            )
        ntc_state = self.ntc.initialize(majorant_sigma_speed)
        accumulator = self.moments.initialize_accumulator(particles.position.dtype)
        reservoir_states = tuple(
            value.initialize(particles.position.dtype) for value in self.reservoirs
        )
        return DSMCRuntimeState(
            particles,
            ntc_state,
            accumulator,
            reservoir_states,
            jnp.asarray(0.0, dtype=particles.position.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            key_data,
            canonical_fingerprint(
                {
                    "kind": "dsmc-runtime",
                    "plan": self.plan_id,
                    "capacity": particles.capacity,
                }
            ),
        )

    def update_majorant(
        self,
        state: DSMCRuntimeState,
        required_sigma_speed: ArrayLike,
        /,
        *,
        safety_factor: float = 1.05,
    ) -> DSMCRuntimeState:
        if state.runtime_id != self._runtime_id(state.particles.capacity):
            raise ValueError("DSMC runtime state does not belong to this plan.")
        return DSMCRuntimeState(
            state.particles,
            self.ntc.update_majorant(
                state.ntc,
                required_sigma_speed,
                safety_factor=safety_factor,
            ),
            state.moment_accumulator,
            state.reservoir_states,
            state.time,
            state.accepted_steps,
            state.key,
            state.runtime_id,
        )

    def _runtime_id(self, capacity: int, /) -> str:
        return canonical_fingerprint(
            {"kind": "dsmc-runtime", "plan": self.plan_id, "capacity": capacity}
        )

    def _removed_exchange(
        self, particles: DSMCParticleState, removed: Array, /
    ) -> tuple[Array, Array, Array]:
        safe_species = jnp.clip(
            particles.species_index, 0, self.collisions.species.species_count - 1
        )
        molecular_mass = self.collisions.species.molecular_masses[safe_species]
        represented_mass = particles.statistical_weight * molecular_mass
        mask = removed.astype(particles.position.dtype)
        mass = jnp.sum(mask * represented_mass)
        momentum = jnp.sum(
            mask[:, None] * represented_mass[:, None] * particles.velocity, axis=0
        )
        energy = jnp.sum(
            mask
            * particles.statistical_weight
            * (
                0.5 * molecular_mass * jnp.sum(particles.velocity**2, axis=-1)
                + particles.rotational_energy
                + particles.vibrational_energy
            )
        )
        return mass, momentum, energy

    def _apply_surfaces(
        self,
        incoming: DSMCParticleState,
        streamed,
        step: Array,
        key: PRNGKeyArray,
        /,
    ) -> tuple[DSMCParticleState, Array, Array, Array, Array, Array]:
        dimension = incoming.velocity.shape[-1]
        dtype = incoming.position.dtype
        mass = jnp.asarray(0.0, dtype=dtype)
        impulse = jnp.zeros((dimension,), dtype=dtype)
        heat = jnp.asarray(0.0, dtype=dtype)
        finite = jnp.asarray(True)
        successful = jnp.asarray(True)
        if not self.surface_boundaries:
            return streamed.state, mass, impulse, heat, finite, successful
        surface_crossed = jnp.zeros((incoming.capacity,), dtype=jnp.bool_)
        for surface in self.surface_boundaries:
            face_index = 2 * surface.face.axis + (
                1 if surface.face.side == "upper" else 0
            )
            surface_crossed = surface_crossed | (streamed.crossed_face == face_index)
        displacement_cells = jnp.max(
            jnp.abs(step * incoming.velocity) / self.streaming.cells.cell_widths,
            axis=-1,
        )
        crossing_count = jnp.sum(streamed.crossed_boundary, axis=-1)
        topology_supported = jnp.all(
            ~surface_crossed | ((crossing_count == 1) & (displacement_cells <= 1.0))
        )
        successful = successful & topology_supported
        current = streamed.state
        surface_keys = jax.random.split(key, len(self.surface_boundaries))
        slot_indices = jnp.arange(incoming.capacity, dtype=jnp.int32)
        inert_position = 0.5 * (self.streaming.cells.lower + self.streaming.cells.upper)
        for surface_index, surface in enumerate(self.surface_boundaries):
            axis = surface.face.axis
            face_number = 2 * axis + (1 if surface.face.side == "upper" else 0)
            selected = incoming.active & (streamed.crossed_face == face_number)
            boundary_coordinate = (
                self.streaming.cells.lower[axis]
                if surface.face.side == "lower"
                else self.streaming.cells.upper[axis]
            ).astype(dtype)
            normal_velocity = incoming.velocity[:, axis]
            safe_velocity = jnp.where(
                jnp.abs(normal_velocity) > jnp.finfo(dtype).tiny,
                normal_velocity,
                1.0,
            )
            hit_time = (boundary_coordinate - incoming.position[:, axis]) / safe_velocity
            hit_valid = (
                jnp.isfinite(hit_time)
                & (hit_time >= 0.0)
                & (hit_time <= step)
                & (
                    normal_velocity * surface.face.outward_normal[axis].astype(dtype)
                    > 0.0
                )
            )
            event_mask = selected & hit_valid
            successful = successful & jnp.all(~selected | hit_valid)
            hit_position = incoming.position + hit_time[:, None] * incoming.velocity
            event_state = DSMCParticleState(
                jnp.where(event_mask[:, None], hit_position, current.position),
                jnp.where(event_mask[:, None], incoming.velocity, current.velocity),
                jnp.where(event_mask, incoming.species_index, current.species_index),
                jnp.where(
                    event_mask,
                    incoming.rotational_energy,
                    current.rotational_energy,
                ),
                jnp.where(
                    event_mask,
                    incoming.vibrational_energy,
                    current.vibrational_energy,
                ),
                jnp.where(
                    event_mask,
                    incoming.statistical_weight,
                    current.statistical_weight,
                ),
                jnp.where(event_mask, incoming.cell_id, current.cell_id),
                jnp.where(event_mask, incoming.active, current.active),
                jnp.where(event_mask, incoming.incarnation, current.incarnation),
            )
            event_indices = jnp.where(event_mask, slot_indices, -1)
            normals = jnp.broadcast_to(
                surface.face.outward_normal.astype(dtype),
                (incoming.capacity, dimension),
            )
            event_keys = jax.random.split(surface_keys[surface_index], incoming.capacity)
            result = surface.interaction.interact(
                event_state,
                event_indices,
                normals,
                event_keys,
            )
            remaining = jnp.maximum(step - hit_time, 0.0)
            propagated = hit_position + remaining[:, None] * result.state.velocity
            propagated_cell, inside = self.streaming.cells.locate(propagated)
            surviving_event = event_mask & result.state.active
            event_supported = ~event_mask | ~result.state.active | inside
            successful = successful & result.successful & jnp.all(event_supported)
            position = jnp.where(
                surviving_event[:, None],
                propagated,
                result.state.position,
            )
            position = jnp.where(
                (event_mask & ~result.state.active)[:, None],
                inert_position,
                position,
            )
            cell_id = jnp.where(
                surviving_event,
                propagated_cell,
                result.state.cell_id,
            )
            current = DSMCParticleState(
                position,
                result.state.velocity,
                result.state.species_index,
                result.state.rotational_energy,
                result.state.vibrational_energy,
                result.state.statistical_weight,
                cell_id,
                result.state.active,
                result.state.incarnation,
            )
            mass = mass + result.total_mass
            impulse = impulse + result.total_force_impulse
            heat = heat + result.total_heat
            finite = finite & jnp.all(result.finite) & jnp.all(jnp.isfinite(position))
        return current, mass, impulse, heat, finite, successful

    def advance(self, state: DSMCRuntimeState, step_size: ArrayLike, /) -> DSMCStepResult:
        if state.runtime_id != self._runtime_id(state.particles.capacity):
            raise ValueError("DSMC runtime state does not belong to this plan.")
        step = jnp.asarray(step_size, dtype=state.particles.position.dtype)
        if step.shape != ():
            raise ValueError("DSMC production step size must be scalar.")
        typed_key = jax.random.wrap_key_data(state.key)
        keys = jax.random.split(typed_key, 4 + len(self.reservoirs))
        key_next = jax.random.key_data(keys[0])
        schedule_key = keys[1]
        collision_key = keys[2]
        surface_key = keys[3]
        stream = self.streaming.advance(state.particles, step)
        removed_mass, removed_momentum, removed_energy = self._removed_exchange(
            state.particles, stream.deactivated_mask
        )
        (
            particles,
            surface_mass,
            surface_momentum,
            surface_energy,
            surface_finite,
            surface_ok,
        ) = self._apply_surfaces(state.particles, stream, step, surface_key)
        candidate_reservoir_states: list[DSMCReservoirState] = []
        injected_count = jnp.asarray(0, dtype=jnp.int32)
        injected_mass = jnp.asarray(0.0, dtype=particles.position.dtype)
        injected_momentum = jnp.zeros(
            (particles.velocity.shape[-1],), dtype=particles.velocity.dtype
        )
        injected_energy = jnp.asarray(0.0, dtype=particles.position.dtype)
        reservoir_ok = jnp.asarray(True)
        reservoir_finite = jnp.asarray(True)
        reservoir_margin = jnp.asarray(1.0, dtype=particles.position.dtype)
        reservoir_reasons = jnp.asarray(0, dtype=jnp.uint32)
        for index, (reservoir, reservoir_state) in enumerate(
            zip(self.reservoirs, state.reservoir_states, strict=True)
        ):
            result = reservoir.inject(
                particles,
                reservoir_state,
                step,
                keys[4 + index],
            )
            particles = result.state
            candidate_reservoir_states.append(result.candidate_reservoir_state)
            injected_count = injected_count + result.injected_count
            injected_mass = injected_mass + result.injected_mass
            injected_momentum = injected_momentum + result.injected_momentum
            injected_energy = injected_energy + result.injected_energy
            reservoir_ok = reservoir_ok & result.header.globally_eligible
            reservoir_finite = reservoir_finite & jnp.all(
                jnp.isfinite(
                    jnp.concatenate(
                        (
                            result.injected_mass[None],
                            result.injected_momentum,
                            result.injected_energy[None],
                        )
                    )
                )
            )
            reservoir_margin = jnp.minimum(
                reservoir_margin, jnp.min(result.header.margin)
            )
            reservoir_reasons = reservoir_reasons | jnp.bitwise_or.reduce(
                result.header.reason_bits
            )

        schedule = self.ntc.schedule(
            particles,
            state.ntc,
            step,
            schedule_key,
        )
        event_keys = jax.random.split(collision_key, self.ntc.event_capacity)
        event_majorant = state.ntc.majorant_sigma_speed[schedule.event_cells]

        def event_body(current, event):
            first, second, valid, event_cell, majorant, key = event
            collision_uniforms = jax.random.uniform(
                jax.random.fold_in(key, 0),
                (3,),
                dtype=current.velocity.dtype,
            )
            collision = self.collisions.collide_one(
                current,
                first,
                second,
                valid,
                collision_uniforms,
                majorant,
            )
            internal = self.internal.apply_one(
                collision.state,
                first,
                second,
                collision.accepted,
                jax.random.fold_in(key, 1),
            )
            return internal.state, (
                collision.accepted,
                collision.sigma_speed,
                collision.momentum_defect,
                collision.energy_defect,
                collision.majorant_violation,
                collision.finite,
                internal.reacted,
                internal.relaxed,
                internal.energy_defect,
                internal.finite,
                event_cell,
                valid,
            )

        collided, event_diagnostics = jax.lax.scan(
            event_body,
            particles,
            (
                schedule.first_indices,
                schedule.second_indices,
                schedule.valid_events,
                schedule.event_cells,
                event_majorant,
                event_keys,
            ),
        )
        (
            collision_accepted,
            sigma_speed,
            momentum_defect,
            collision_energy_defect,
            majorant_violation,
            collision_finite,
            reacted,
            relaxed,
            internal_energy_defect,
            internal_finite,
            event_cells,
            valid_events,
        ) = event_diagnostics
        required_majorant = (
            jnp.zeros_like(state.ntc.majorant_sigma_speed)
            .at[event_cells]
            .max(jnp.where(valid_events, sigma_speed, 0.0))
        )
        energy_scale = jnp.maximum(
            jnp.max(jnp.abs(collision_energy_defect))
            + jnp.max(jnp.abs(internal_energy_defect)),
            1.0,
        )
        tolerance = 1024.0 * jnp.finfo(collided.velocity.dtype).eps * energy_scale
        event_ok = (
            jnp.all(collision_finite)
            & jnp.all(internal_finite)
            & ~jnp.any(majorant_violation)
            & jnp.all(jnp.abs(collision_energy_defect) <= tolerance)
            & jnp.all(jnp.abs(internal_energy_defect) <= tolerance)
            & jnp.all(jnp.isfinite(momentum_defect))
        )
        finite = (
            stream.finite
            & surface_finite
            & reservoir_finite
            & jnp.all(collision_finite)
            & jnp.all(internal_finite)
            & jnp.all(jnp.isfinite(collided.position))
            & jnp.all(jnp.isfinite(collided.velocity))
        )
        physical_ok = (
            stream.successful
            & surface_ok
            & reservoir_ok
            & schedule.header.globally_eligible
            & event_ok
            & finite
        )
        reasons = jnp.bitwise_or.reduce(schedule.header.reason_bits)
        reasons = reasons | reservoir_reasons
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            stream.successful & surface_ok & event_ok,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        reasons = jnp.where(
            ~jnp.any(majorant_violation),
            reasons,
            reasons | jnp.asarray(int(DSMCNTCReason.INVALID_MAJORANT), jnp.uint32),
        )
        margin = jnp.minimum(
            jnp.min(schedule.header.margin),
            reservoir_margin,
        )
        margin = jnp.minimum(
            margin,
            jnp.where(stream.successful & surface_ok & event_ok, 1.0, -1.0),
        )
        header = AdmissibilityHeader(
            margin,
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "dsmc-step-evidence", "plan": self.plan_id}),
        )
        successful = physical_ok & header.globally_eligible
        candidate_moments = self.moments.evaluate(collided)
        candidate_accumulator = self.moments.accumulate(
            state.moment_accumulator, candidate_moments
        )
        candidate = DSMCRuntimeState(
            collided,
            schedule.candidate_state,
            candidate_accumulator,
            tuple(candidate_reservoir_states),
            state.time + step,
            state.accepted_steps + 1,
            key_next,
            state.runtime_id,
        )
        accepted = jax.tree.map(
            lambda new, old: (
                jnp.where(successful, new, old) if isinstance(new, jax.Array) else new
            ),
            candidate,
            state,
        )
        accepted_moments = self.moments.evaluate(accepted.particles)
        statistics = self.moments.statistical_evidence(accepted.moment_accumulator)
        boundary = DSMCBoundaryExchangeLedger(
            stream.deactivated_count,
            removed_mass,
            removed_momentum,
            removed_energy,
            injected_count,
            injected_mass,
            injected_momentum,
            injected_energy,
            surface_mass,
            surface_momentum,
            surface_energy,
            injected_mass - removed_mass - surface_mass,
            injected_momentum - removed_momentum - surface_momentum,
            injected_energy - removed_energy - surface_energy,
        )
        return DSMCStepResult(
            candidate,
            accepted,
            accepted_moments,
            statistics,
            schedule,
            boundary,
            required_majorant,
            jnp.sum(collision_accepted),
            jnp.sum(reacted),
            jnp.sum(relaxed),
            finite,
            header,
            successful,
            self.plan_id,
        )


__all__ = [
    "DSMCBoundaryExchangeLedger",
    "DSMCProductionPlan",
    "DSMCRuntimeState",
    "DSMCStepResult",
]
