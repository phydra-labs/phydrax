#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...particle import (
    ParticleAllocationRequest,
    ParticlePopulationPlan,
    ParticlePopulationState,
)
from .._charge_state import PICChargeModelPlan, PICChargeState
from .._types import PICParticleState
from ._types import PICIonizationResult


class ElectronImpactIonizationPlan(StrictModule, NonTrainableState):
    energy_grid: jnp.ndarray
    cross_section: jnp.ndarray
    ionization_energy: float = eqx.field(static=True)
    rate_scale: float = eqx.field(static=True)
    maximum_probability: float = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_grid: ArrayLike,
        cross_section: ArrayLike,
        /,
        *,
        ionization_energy: float,
        rate_scale: float = 1.0,
        maximum_probability: float = 0.25,
        maximum_events: int,
    ):
        energy = np.asarray(energy_grid, dtype=float)
        section = np.asarray(cross_section, dtype=float)
        threshold = float(ionization_energy)
        scale = float(rate_scale)
        probability = float(maximum_probability)
        events = int(maximum_events)
        if (
            energy.ndim != 1
            or energy.size < 2
            or section.shape != energy.shape
            or np.any(~np.isfinite(energy))
            or np.any(~np.isfinite(section))
            or np.any(np.diff(energy) <= 0.0)
            or np.any(section < 0.0)
        ):
            raise ValueError("Ionization cross-section table is invalid.")
        if threshold <= 0.0 or scale < 0.0 or not 0.0 < probability <= 1.0 or events <= 0:
            raise ValueError("Ionization policy values are invalid.")
        self.energy_grid = jnp.asarray(energy)
        self.cross_section = jnp.asarray(section)
        self.ionization_energy = threshold
        self.rate_scale = scale
        self.maximum_probability = probability
        self.maximum_events = events
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electron-impact-ionization",
                "energy": array_tree_fingerprint(energy),
                "section": array_tree_fingerprint(section),
                "threshold": threshold,
                "rate_scale": scale,
                "maximum_probability": probability,
                "maximum_events": events,
            }
        )

    def apply(
        self,
        ion_model: PICChargeModelPlan,
        ion_population: ParticlePopulationState,
        ion_charge: PICChargeState,
        ion_particles: PICParticleState,
        electron_model: PICChargeModelPlan,
        electron_population_plan: ParticlePopulationPlan,
        electron_population: ParticlePopulationState,
        electron_charge: PICChargeState,
        electron_particles: PICParticleState,
        ion_indices: ArrayLike,
        electron_indices: ArrayLike,
        key,
        step_size: ArrayLike,
        step_index: ArrayLike,
        /,
    ) -> PICIonizationResult:
        ions = jnp.asarray(ion_indices, dtype=jnp.int32)
        electrons = jnp.asarray(electron_indices, dtype=jnp.int32)
        if ions.shape != (self.maximum_events,) or electrons.shape != ions.shape:
            raise ValueError("Ionization pair arrays must match maximum_events.")
        safe_ions = jnp.clip(ions, 0, ion_population.active.size - 1)
        safe_electrons = jnp.clip(electrons, 0, electron_population.active.size - 1)
        valid_pair = (
            (ions >= 0)
            & (electrons >= 0)
            & ion_population.active[safe_ions]
            & electron_population.active[safe_electrons]
            & (ion_charge.charge_number[safe_ions] < ion_model.maximum_charge_number)
        )
        ion_velocity = ion_particles.proper_velocity[safe_ions]
        electron_velocity = electron_particles.proper_velocity[safe_electrons]
        ion_mass = ion_population.mass[safe_ions]
        electron_mass = electron_population.mass[safe_electrons]
        relative = electron_velocity - ion_velocity
        reduced_mass = (
            ion_mass * electron_mass / jnp.maximum(ion_mass + electron_mass, 1.0e-30)
        )
        energy = 0.5 * reduced_mass * jnp.sum(relative * relative, axis=-1)
        section = jnp.interp(
            energy,
            self.energy_grid,
            self.cross_section,
            left=0.0,
            right=self.cross_section[-1],
        )
        dt = jnp.asarray(step_size, dtype=energy.dtype).reshape(())
        probability = 1.0 - jnp.exp(
            -self.rate_scale
            * section
            * jnp.sqrt(
                jnp.maximum(2.0 * energy / jnp.maximum(reduced_mass, 1.0e-30), 0.0)
            )
            * dt
        )
        stable = jnp.all(
            jnp.where(valid_pair, probability <= self.maximum_probability, True)
        )
        event = (
            valid_pair
            & (energy >= self.ionization_energy)
            & (jr.uniform(key, probability.shape, dtype=probability.dtype) < probability)
            & stable
        )
        requested_masses = jnp.where(event, electron_mass, 1.0)
        allocation = electron_population_plan.allocate(
            electron_population,
            ParticleAllocationRequest(
                jnp.arange(self.maximum_events, dtype=jnp.int64),
                requested_masses,
                event,
            ),
        )
        use = event & allocation.allocated
        slots = jnp.maximum(allocation.slots, 0)
        delta = (
            jnp.zeros_like(ion_charge.charge_number)
            .at[safe_ions]
            .add(use.astype(jnp.int16))
        )
        new_electron_charge_value = (
            allocation.accepted_state.mass[slots]
            * electron_model.base_specific_charge
            * electron_model.initial_charge_number
        )
        compensating = jnp.sum(jnp.where(use, new_electron_charge_value, 0.0))
        transition = ion_model.transition(
            ion_population,
            ion_charge,
            delta,
            step_index,
            compensating_charge=compensating,
        )
        total_mass = ion_mass + 2.0 * electron_mass
        center = (
            ion_mass[:, None] * ion_velocity + electron_mass[:, None] * electron_velocity
        ) / jnp.maximum(total_mass[:, None], 1.0e-30)
        center_energy = 0.5 * total_mass * jnp.sum(center * center, axis=-1)
        initial_energy = 0.5 * ion_mass * jnp.sum(
            ion_velocity**2, axis=-1
        ) + 0.5 * electron_mass * jnp.sum(electron_velocity**2, axis=-1)
        available = jnp.maximum(
            initial_energy - center_energy - self.ionization_energy, 0.0
        )
        relative_speed = jnp.sqrt(available / jnp.maximum(electron_mass, 1.0e-30))
        direction_key = jr.fold_in(key, 1)
        direction = jr.normal(direction_key, (self.maximum_events, 3), dtype=center.dtype)
        direction = direction / jnp.maximum(
            jnp.sqrt(jnp.sum(direction**2, axis=-1))[:, None], 1.0e-30
        )
        primary_velocity = center + relative_speed[:, None] * direction
        secondary_velocity = center - relative_speed[:, None] * direction
        ion_candidate_velocity = ion_particles.proper_velocity.at[safe_ions].set(
            jnp.where(use[:, None], center, ion_velocity)
        )
        electron_candidate_velocity = electron_particles.proper_velocity.at[
            safe_electrons
        ].set(jnp.where(use[:, None], primary_velocity, electron_velocity))
        electron_candidate_velocity = electron_candidate_velocity.at[slots].set(
            jnp.where(
                use[:, None], secondary_velocity, electron_candidate_velocity[slots]
            )
        )
        electron_candidate_position = electron_particles.position.at[slots].set(
            jnp.where(
                use[:, None],
                ion_particles.position[safe_ions],
                electron_particles.position[slots],
            )
        )
        electron_number = electron_charge.charge_number.at[slots].set(
            jnp.where(
                use,
                electron_model.initial_charge_number,
                electron_charge.charge_number[slots],
            )
        )
        electron_charge_candidate = PICChargeState(
            electron_number,
            electron_charge.transition_count.at[slots].add(use.astype(jnp.int32)),
            electron_charge.last_transition_step.at[slots].set(
                jnp.where(
                    use,
                    jnp.asarray(step_index, dtype=jnp.int32),
                    electron_charge.last_transition_step[slots],
                )
            ),
        )
        ion_particle_candidate = PICParticleState(
            ion_particles.position, ion_candidate_velocity
        )
        electron_particle_candidate = PICParticleState(
            electron_candidate_position, electron_candidate_velocity
        )
        success = allocation.successful & transition.successful & stable
        ion_particle_accepted = PICParticleState(
            jnp.where(success, ion_particle_candidate.position, ion_particles.position),
            jnp.where(
                success,
                ion_particle_candidate.proper_velocity,
                ion_particles.proper_velocity,
            ),
        )
        electron_particle_accepted = PICParticleState(
            jnp.where(
                success, electron_particle_candidate.position, electron_particles.position
            ),
            jnp.where(
                success,
                electron_particle_candidate.proper_velocity,
                electron_particles.proper_velocity,
            ),
        )
        electron_charge_accepted = PICChargeState(
            jnp.where(
                success,
                electron_charge_candidate.charge_number,
                electron_charge.charge_number,
            ),
            jnp.where(
                success,
                electron_charge_candidate.transition_count,
                electron_charge.transition_count,
            ),
            jnp.where(
                success,
                electron_charge_candidate.last_transition_step,
                electron_charge.last_transition_step,
            ),
        )
        momentum_before = jnp.sum(
            ion_mass[:, None] * ion_velocity + electron_mass[:, None] * electron_velocity,
            axis=0,
        )
        momentum_after = jnp.sum(
            ion_mass[:, None] * jnp.where(use[:, None], center, ion_velocity)
            + electron_mass[:, None]
            * jnp.where(
                use[:, None], primary_velocity + secondary_velocity, electron_velocity
            ),
            axis=0,
        )
        momentum_defect = jnp.sqrt(jnp.sum((momentum_after - momentum_before) ** 2))
        finite = jnp.all(jnp.isfinite(ion_particle_candidate.proper_velocity)) & jnp.all(
            jnp.isfinite(electron_particle_candidate.proper_velocity)
        )
        return PICIonizationResult(
            transition.accepted_state,
            ion_particle_accepted,
            allocation.accepted_state,
            electron_particle_accepted,
            electron_charge_accepted,
            use,
            jnp.sum(use, dtype=jnp.int32),
            transition.charge_source,
            momentum_defect,
            jnp.asarray(0.0, dtype=energy.dtype),
            self.ionization_energy * jnp.sum(use, dtype=energy.dtype),
            allocation.capacity_available,
            finite,
            success & finite,
            self.plan_id,
        )


__all__ = ["ElectronImpactIonizationPlan"]
