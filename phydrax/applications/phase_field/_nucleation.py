#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...stochastic import PoissonClockRealization


class ClassicalNucleationEvaluation(StrictModule):
    rate: Array
    critical_radius: Array
    barrier: Array
    finite: Array
    admissible: Array
    successful: Array
    law_id: str = eqx.field(static=True)


class ClassicalNucleationRateLaw(StrictModule, NonTrainableState):
    surface_energy: Array
    kinetic_prefactor: Array
    boltzmann_constant: Array
    heterogeneous_factor: Array
    spatial_dimension: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        surface_energy: ArrayLike,
        kinetic_prefactor: ArrayLike,
        boltzmann_constant: ArrayLike,
        spatial_dimension: int = 3,
        heterogeneous_factor: ArrayLike = 1.0,
        law_id: str,
    ):
        values = tuple(
            np.asarray(value)
            for value in (
                surface_energy,
                kinetic_prefactor,
                boltzmann_constant,
                heterogeneous_factor,
            )
        )
        dimension = int(spatial_dimension)
        identifier = str(law_id)
        if (
            any(value.shape != () or not np.isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] < 0.0
            or values[2] <= 0.0
            or values[3] <= 0.0
            or values[3] > 1.0
            or dimension not in (2, 3)
            or not identifier
        ):
            raise ValueError("Classical nucleation law is invalid.")
        self.surface_energy = jnp.asarray(values[0])
        self.kinetic_prefactor = jnp.asarray(values[1])
        self.boltzmann_constant = jnp.asarray(values[2])
        self.heterogeneous_factor = jnp.asarray(values[3])
        self.spatial_dimension = dimension
        self.law_id = canonical_fingerprint(
            {
                "kind": "classical-nucleation-rate-law",
                "declared_id": identifier,
                "surface_energy": float(values[0]),
                "kinetic_prefactor": float(values[1]),
                "boltzmann_constant": float(values[2]),
                "spatial_dimension": dimension,
                "heterogeneous_factor": float(values[3]),
            }
        )

    def evaluate(
        self,
        driving_force: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> ClassicalNucleationEvaluation:
        driving = jnp.asarray(driving_force)
        thermal = jnp.asarray(temperature, dtype=driving.dtype)
        if driving.shape != thermal.shape:
            raise ValueError("Nucleation driving force and temperature must align.")
        positive = driving > 0.0
        safe_driving = jnp.where(positive, driving, 1.0)
        surface = self.surface_energy.astype(driving.dtype)
        if self.spatial_dimension == 3:
            radius = 2.0 * surface / safe_driving
            barrier = 16.0 * jnp.pi * surface**3 / (3.0 * safe_driving**2)
        else:
            radius = surface / safe_driving
            barrier = jnp.pi * surface**2 / safe_driving
        barrier = barrier * self.heterogeneous_factor.astype(driving.dtype)
        rate = self.kinetic_prefactor.astype(driving.dtype) * jnp.exp(
            -barrier / (self.boltzmann_constant.astype(driving.dtype) * thermal)
        )
        radius = jnp.where(positive, radius, jnp.inf)
        barrier = jnp.where(positive, barrier, jnp.inf)
        rate = jnp.where(positive, rate, 0.0)
        finite = jnp.all(jnp.isfinite(rate)) & jnp.all(
            jnp.isfinite(thermal) & (thermal > 0.0)
        )
        admissible = jnp.all(driving >= 0.0)
        return ClassicalNucleationEvaluation(
            rate,
            radius,
            barrier,
            finite,
            admissible,
            finite & admissible,
            self.law_id,
        )


class NucleationClockState(StrictModule):
    integrated_hazard: Array
    event_counts: Array
    accepted_events: Array
    state_id: str = eqx.field(static=True)


class NucleationProposal(StrictModule):
    event_times: Array
    channels: Array
    event_indices: Array
    valid: Array
    critical_radius: Array
    proposed_radius: Array
    orientation: Array
    required_component: Array
    required_energy: Array
    capacity_overflow: Array
    finite: Array
    successful: Array
    proposal_id: str = eqx.field(static=True)


class NucleationTransaction(StrictModule):
    previous: NucleationClockState
    candidate: NucleationClockState
    proposal: NucleationProposal
    accepted_mask: Array
    component_used: Array
    energy_used: Array
    remaining_component: Array
    remaining_energy: Array
    successful: Array


class NucleationEventPlan(StrictModule, NonTrainableState):
    realization: PoissonClockRealization
    rate_law: ClassicalNucleationRateLaw
    channel_measures: Array
    component_cost_density: Array
    energy_cost_density: Array
    radius_overshoot: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        realization: PoissonClockRealization,
        rate_law: ClassicalNucleationRateLaw,
        channel_measures: ArrayLike,
        /,
        *,
        component_cost_density: ArrayLike,
        energy_cost_density: ArrayLike,
        radius_overshoot: float = 0.05,
    ):
        if not isinstance(realization, PoissonClockRealization):
            raise TypeError("realization must be PoissonClockRealization.")
        if not isinstance(rate_law, ClassicalNucleationRateLaw):
            raise TypeError("rate_law must be ClassicalNucleationRateLaw.")
        measures = np.asarray(channel_measures)
        component_cost = np.asarray(component_cost_density)
        energy_cost = np.asarray(energy_cost_density)
        overshoot = float(radius_overshoot)
        if (
            measures.shape != (realization.num_channels,)
            or np.any(~np.isfinite(measures))
            or np.any(measures <= 0.0)
            or component_cost.shape != ()
            or energy_cost.shape != ()
            or not np.isfinite(component_cost)
            or not np.isfinite(energy_cost)
            or component_cost < 0.0
            or energy_cost < 0.0
            or not np.isfinite(overshoot)
            or overshoot < 0.0
        ):
            raise ValueError("Nucleation event-plan parameters are invalid.")
        self.realization = realization
        self.rate_law = rate_law
        self.channel_measures = jnp.asarray(measures)
        self.component_cost_density = jnp.asarray(component_cost)
        self.energy_cost_density = jnp.asarray(energy_cost)
        self.radius_overshoot = overshoot
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nucleation-event-plan",
                "realization": realization.realization_id,
                "rate_law": rate_law.law_id,
                "channel_measures": measures.tolist(),
                "component_cost_density": float(component_cost),
                "energy_cost_density": float(energy_cost),
                "radius_overshoot": overshoot,
            }
        )

    def initialize(self) -> NucleationClockState:
        return NucleationClockState(
            jnp.zeros((self.realization.num_channels,), dtype=jnp.float64),
            jnp.zeros((self.realization.num_channels,), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def propose(
        self,
        state: NucleationClockState,
        start_time: ArrayLike,
        step_size: ArrayLike,
        driving_force: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> tuple[NucleationClockState, NucleationProposal]:
        if not isinstance(state, NucleationClockState) or state.state_id != self.plan_id:
            raise ValueError("Nucleation clock state is incompatible with its plan.")
        start = jnp.asarray(start_time)
        step = jnp.asarray(step_size, dtype=start.dtype)
        driving = jnp.asarray(driving_force, dtype=start.dtype)
        thermal = jnp.asarray(temperature, dtype=start.dtype)
        if driving.shape != (self.realization.num_channels,) or thermal.shape != (
            self.realization.num_channels,
        ):
            raise ValueError("Nucleation rate fields must have one value per channel.")
        evaluation = self.rate_law.evaluate(driving, thermal)
        hazard_increment = evaluation.rate * self.channel_measures * step
        candidate_hazard = state.integrated_hazard + hazard_increment
        thresholds = self.realization.thresholds
        previous_crossed = thresholds <= state.integrated_hazard[:, None]
        candidate_crossed = thresholds <= candidate_hazard[:, None]
        newly_crossed = candidate_crossed & ~previous_crossed
        event_counts = jnp.sum(candidate_crossed, axis=-1, dtype=jnp.int32)
        overflow = jnp.any(
            (event_counts >= self.realization.max_events_per_channel)
            & (
                candidate_hazard
                >= thresholds[:, self.realization.max_events_per_channel - 1]
            )
        )
        channels = jnp.broadcast_to(
            jnp.arange(self.realization.num_channels, dtype=jnp.int32)[:, None],
            thresholds.shape,
        )
        event_indices = jnp.broadcast_to(
            jnp.arange(self.realization.max_events_per_channel, dtype=jnp.int32)[None, :],
            thresholds.shape,
        )
        safe_increment = jnp.where(hazard_increment > 0.0, hazard_increment, 1.0)
        fraction = jnp.clip(
            (thresholds - state.integrated_hazard[:, None]) / safe_increment[:, None],
            0.0,
            1.0,
        )
        event_times = start + step * fraction
        mark_keys = self.realization.mark_keys.reshape(
            (-1,) + tuple(self.realization.root_key.shape)
        )
        orientation = jax.vmap(lambda key: jax.random.uniform(key))(mark_keys).reshape(
            thresholds.shape
        )
        orientation = 2.0 * jnp.pi * orientation
        critical = jnp.broadcast_to(evaluation.critical_radius[:, None], thresholds.shape)
        proposed = critical * (1.0 + self.radius_overshoot)
        measure_factor = (
            jnp.pi * proposed**2
            if self.rate_law.spatial_dimension == 2
            else (4.0 / 3.0) * jnp.pi * proposed**3
        )
        component = self.component_cost_density.astype(start.dtype) * measure_factor
        energy = self.energy_cost_density.astype(start.dtype) * measure_factor
        finite = (
            evaluation.finite
            & jnp.all(~newly_crossed | jnp.isfinite(event_times))
            & jnp.all(~newly_crossed | jnp.isfinite(proposed))
            & jnp.all(~newly_crossed | jnp.isfinite(component))
            & jnp.all(~newly_crossed | jnp.isfinite(energy))
        )
        successful = finite & ~overflow
        candidate_state = NucleationClockState(
            candidate_hazard,
            event_counts,
            state.accepted_events + jnp.sum(newly_crossed, dtype=jnp.int32),
            self.plan_id,
        )
        proposal = NucleationProposal(
            event_times.reshape((-1,)),
            channels.reshape((-1,)),
            event_indices.reshape((-1,)),
            newly_crossed.reshape((-1,)),
            critical.reshape((-1,)),
            proposed.reshape((-1,)),
            orientation.reshape((-1,)),
            component.reshape((-1,)),
            energy.reshape((-1,)),
            overflow,
            finite,
            successful,
            self.plan_id,
        )
        return candidate_state, proposal

    def transact(
        self,
        state: NucleationClockState,
        candidate_state: NucleationClockState,
        proposal: NucleationProposal,
        /,
        *,
        available_component: ArrayLike,
        available_energy: ArrayLike,
    ) -> NucleationTransaction:
        component = jnp.asarray(available_component)
        energy = jnp.asarray(available_energy, dtype=component.dtype)
        if component.shape != () or energy.shape != ():
            raise ValueError("Nucleation inventories must be scalar.")
        sort_time = jnp.where(proposal.valid, proposal.event_times, jnp.inf)
        order = jnp.lexsort(
            (
                proposal.event_indices,
                proposal.channels,
                sort_time,
            )
        )
        component_sorted = proposal.required_component[order]
        energy_sorted = proposal.required_energy[order]
        valid_sorted = proposal.valid[order]
        component_prefix = jnp.cumsum(jnp.where(valid_sorted, component_sorted, 0.0))
        energy_prefix = jnp.cumsum(jnp.where(valid_sorted, energy_sorted, 0.0))
        accepted_sorted = (
            valid_sorted & (component_prefix <= component) & (energy_prefix <= energy)
        )
        accepted = jnp.zeros_like(accepted_sorted).at[order].set(accepted_sorted)
        component_used = jnp.sum(jnp.where(accepted, proposal.required_component, 0.0))
        energy_used = jnp.sum(jnp.where(accepted, proposal.required_energy, 0.0))
        all_admissible = jnp.all(~proposal.valid | accepted)
        successful = (
            proposal.successful
            & all_admissible
            & jnp.isfinite(component_used)
            & jnp.isfinite(energy_used)
        )
        committed = NucleationClockState(
            candidate_state.integrated_hazard,
            candidate_state.event_counts,
            state.accepted_events + jnp.sum(accepted, dtype=jnp.int32),
            self.plan_id,
        )
        selected = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), committed, state
        )
        return NucleationTransaction(
            state,
            selected,
            proposal,
            accepted,
            component_used,
            energy_used,
            component - component_used,
            energy - energy_used,
            successful,
        )


__all__ = [
    "ClassicalNucleationEvaluation",
    "ClassicalNucleationRateLaw",
    "NucleationClockState",
    "NucleationEventPlan",
    "NucleationProposal",
    "NucleationTransaction",
]
