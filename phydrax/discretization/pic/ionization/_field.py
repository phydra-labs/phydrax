#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import ArrayLike

from ...._fingerprint import canonical_fingerprint
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


class FieldIonizationPlan(StrictModule, NonTrainableState):
    """Bounded field-rate ionization with collocated electron creation."""

    rate_coefficient: float = eqx.field(static=True)
    field_power: float = eqx.field(static=True)
    ionization_energy: float = eqx.field(static=True)
    maximum_probability: float = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate_coefficient: float,
        /,
        *,
        field_power: float,
        ionization_energy: float,
        maximum_probability: float = 0.25,
        maximum_events: int,
    ):
        rate = float(rate_coefficient)
        power = float(field_power)
        energy = float(ionization_energy)
        probability = float(maximum_probability)
        events = int(maximum_events)
        if (
            not np.isfinite(rate)
            or rate < 0.0
            or not np.isfinite(power)
            or power <= 0.0
            or not np.isfinite(energy)
            or energy <= 0.0
            or not 0.0 < probability <= 1.0
            or events <= 0
        ):
            raise ValueError("Field ionization parameters are invalid.")
        self.rate_coefficient = rate
        self.field_power = power
        self.ionization_energy = energy
        self.maximum_probability = probability
        self.maximum_events = events
        self.plan_id = canonical_fingerprint(
            {
                "kind": "field-ionization",
                "rate": rate,
                "power": power,
                "energy": energy,
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
        electric_field: ArrayLike,
        electron_model: PICChargeModelPlan,
        electron_population_plan: ParticlePopulationPlan,
        electron_population: ParticlePopulationState,
        electron_charge: PICChargeState,
        electron_particles: PICParticleState,
        key,
        step_size: ArrayLike,
        step_index: ArrayLike,
        /,
    ) -> PICIonizationResult:
        field = jnp.asarray(electric_field, dtype=ion_particles.position.dtype)
        if field.shape != (ion_population.active.size, 3):
            raise ValueError("electric_field must have ion-capacity by three shape.")
        magnitude = jnp.sqrt(jnp.sum(field * field, axis=-1))
        dt = jnp.asarray(step_size, dtype=magnitude.dtype).reshape(())
        probability = 1.0 - jnp.exp(
            -self.rate_coefficient * magnitude**self.field_power * dt
        )
        eligible = ion_population.active & (
            ion_charge.charge_number < ion_model.maximum_charge_number
        )
        stable = jnp.all(
            jnp.where(eligible, probability <= self.maximum_probability, True)
        )
        sampled = (
            eligible
            & (jr.uniform(key, probability.shape, dtype=probability.dtype) < probability)
            & stable
        )
        order = jnp.argsort(jnp.where(sampled, -magnitude, jnp.inf))
        selected = order[: self.maximum_events]
        event = sampled[selected]
        selected_mass = ion_population.mass[selected]
        electron_mass = jnp.where(event, selected_mass, 1.0)
        allocation = electron_population_plan.allocate(
            electron_population,
            ParticleAllocationRequest(
                jnp.arange(self.maximum_events, dtype=jnp.int64),
                electron_mass,
                event,
            ),
        )
        use = event & allocation.allocated
        slots = jnp.maximum(allocation.slots, 0)
        delta = (
            jnp.zeros_like(ion_charge.charge_number)
            .at[selected]
            .add(use.astype(jnp.int16))
        )
        electron_macrocharge = (
            allocation.accepted_state.mass[slots]
            * electron_model.base_specific_charge
            * electron_model.initial_charge_number
        )
        transition = ion_model.transition(
            ion_population,
            ion_charge,
            delta,
            step_index,
            compensating_charge=jnp.sum(jnp.where(use, electron_macrocharge, 0.0)),
        )
        electron_position = electron_particles.position.at[slots].set(
            jnp.where(
                use[:, None],
                ion_particles.position[selected],
                electron_particles.position[slots],
            )
        )
        born_velocity = ion_particles.proper_velocity[selected]
        electron_velocity = electron_particles.proper_velocity.at[slots].set(
            jnp.where(
                use[:, None], born_velocity, electron_particles.proper_velocity[slots]
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
        successful = allocation.successful & transition.successful & stable
        electron_state = PICParticleState(electron_position, electron_velocity)
        accepted_electron = PICParticleState(
            jnp.where(successful, electron_state.position, electron_particles.position),
            jnp.where(
                successful,
                electron_state.proper_velocity,
                electron_particles.proper_velocity,
            ),
        )
        accepted_charge = PICChargeState(
            jnp.where(
                successful,
                electron_charge_candidate.charge_number,
                electron_charge.charge_number,
            ),
            jnp.where(
                successful,
                electron_charge_candidate.transition_count,
                electron_charge.transition_count,
            ),
            jnp.where(
                successful,
                electron_charge_candidate.last_transition_step,
                electron_charge.last_transition_step,
            ),
        )
        finite = jnp.all(jnp.isfinite(electron_state.position)) & jnp.all(
            jnp.isfinite(electron_state.proper_velocity)
        )
        return PICIonizationResult(
            transition.accepted_state,
            ion_particles,
            allocation.accepted_state,
            accepted_electron,
            accepted_charge,
            use,
            jnp.sum(use, dtype=jnp.int32),
            transition.charge_source,
            jnp.asarray(0.0, dtype=magnitude.dtype),
            self.ionization_energy * jnp.sum(use, dtype=magnitude.dtype),
            self.ionization_energy * jnp.sum(use, dtype=magnitude.dtype),
            allocation.capacity_available,
            finite,
            successful & finite,
            self.plan_id,
        )


__all__ = ["FieldIonizationPlan"]
