#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..particle import ParticlePopulationPlan, ParticlePopulationState
from ._types import PICParticleState


class PICChargeState(StrictModule):
    charge_number: Array
    transition_count: Array
    last_transition_step: Array


class PICSpeciesState(StrictModule):
    particles: PICParticleState
    population: ParticlePopulationState
    charge: PICChargeState


class PICChargeTransitionResult(StrictModule):
    candidate_state: PICChargeState
    accepted_state: PICChargeState
    old_macrocharge: Array
    new_macrocharge: Array
    charge_source: Array
    changed_count: Array
    bounded: Array
    neutral_total: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PICChargeModelPlan(StrictModule, NonTrainableState):
    """Integer charge-number model over one runtime particle population."""

    base_specific_charge: float = eqx.field(static=True)
    minimum_charge_number: int = eqx.field(static=True)
    maximum_charge_number: int = eqx.field(static=True)
    initial_charge_number: int = eqx.field(static=True)
    species_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base_specific_charge: float,
        species_id: str,
        /,
        *,
        minimum_charge_number: int,
        maximum_charge_number: int,
        initial_charge_number: int,
    ) -> None:
        base = float(base_specific_charge)
        minimum = int(minimum_charge_number)
        maximum = int(maximum_charge_number)
        initial = int(initial_charge_number)
        identifier = str(species_id)
        if not np.isfinite(base) or base == 0.0:
            raise ValueError("base_specific_charge must be finite and nonzero.")
        if minimum > maximum or not minimum <= initial <= maximum:
            raise ValueError("Charge-number bounds or initial value are invalid.")
        if not identifier:
            raise ValueError("species_id must be nonempty.")
        self.base_specific_charge = base
        self.minimum_charge_number = minimum
        self.maximum_charge_number = maximum
        self.initial_charge_number = initial
        self.species_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-charge-model",
                "base_specific_charge": base,
                "minimum": minimum,
                "maximum": maximum,
                "initial": initial,
                "species": identifier,
            }
        )

    def initialize(self, population: ParticlePopulationState, /) -> PICChargeState:
        number = jnp.where(population.active, self.initial_charge_number, 0).astype(
            jnp.int16
        )
        return PICChargeState(
            number,
            jnp.zeros_like(number, dtype=jnp.int32),
            jnp.full_like(number, -1, dtype=jnp.int32),
        )

    def macrocharge(
        self,
        population: ParticlePopulationState,
        charge: PICChargeState,
        /,
    ) -> Array:
        if charge.charge_number.shape != population.mass.shape:
            raise ValueError("Charge and population capacity disagree.")
        return jnp.where(
            population.active,
            population.mass
            * self.base_specific_charge
            * charge.charge_number.astype(population.mass.dtype),
            0.0,
        )

    def transition(
        self,
        population: ParticlePopulationState,
        state: PICChargeState,
        delta: ArrayLike,
        step_index: ArrayLike,
        /,
        *,
        compensating_charge: ArrayLike = 0.0,
        tolerance: float = 1.0e-12,
    ) -> PICChargeTransitionResult:
        increment = jnp.asarray(delta, dtype=jnp.int16)
        if increment.shape != state.charge_number.shape:
            raise ValueError("Charge transition delta must preserve capacity.")
        proposed = state.charge_number.astype(jnp.int32) + increment.astype(jnp.int32)
        bounded = jnp.all(
            jnp.where(
                population.active,
                (proposed >= self.minimum_charge_number)
                & (proposed <= self.maximum_charge_number),
                increment == 0,
            )
        )
        candidate_number = jnp.where(
            bounded & population.active,
            proposed,
            state.charge_number.astype(jnp.int32),
        ).astype(jnp.int16)
        changed = population.active & (candidate_number != state.charge_number)
        candidate = PICChargeState(
            candidate_number,
            state.transition_count + changed.astype(jnp.int32),
            jnp.where(
                changed,
                jnp.asarray(step_index, dtype=jnp.int32),
                state.last_transition_step,
            ),
        )
        old_charge = self.macrocharge(population, state)
        new_charge = self.macrocharge(population, candidate)
        source = jnp.sum(new_charge - old_charge) + jnp.asarray(
            compensating_charge, dtype=new_charge.dtype
        )
        scale = jnp.maximum(
            1.0, jnp.sum(jnp.abs(new_charge)) + jnp.sum(jnp.abs(old_charge))
        )
        neutral = jnp.abs(source) <= float(tolerance) * scale
        successful = bounded & neutral
        accepted = PICChargeState(
            jnp.where(successful, candidate.charge_number, state.charge_number),
            jnp.where(successful, candidate.transition_count, state.transition_count),
            jnp.where(
                successful, candidate.last_transition_step, state.last_transition_step
            ),
        )
        return PICChargeTransitionResult(
            candidate,
            accepted,
            old_charge,
            new_charge,
            source,
            jnp.sum(changed, dtype=jnp.int32),
            bounded,
            neutral,
            successful,
            self.plan_id,
        )


class PICSpeciesPlan(StrictModule, NonTrainableState):
    """One PIC species: runtime population with persistent identities and charge.

    Macroparticle charge is ``mass * base_specific_charge * charge_number`` of the
    runtime population, so ionization and creation events change charge through
    the population and charge state rather than through prepared constants.
    """

    population: ParticlePopulationPlan
    charge_model: PICChargeModelPlan
    species_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        population: ParticlePopulationPlan,
        charge_model: PICChargeModelPlan,
        /,
    ) -> None:
        if not isinstance(population, ParticlePopulationPlan):
            raise TypeError("population must be ParticlePopulationPlan.")
        if not isinstance(charge_model, PICChargeModelPlan):
            raise TypeError("charge_model must be PICChargeModelPlan.")
        self.population = population
        self.charge_model = charge_model
        self.species_id = charge_model.species_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-species",
                "population": population.plan_id,
                "charge_model": charge_model.plan_id,
            }
        )

    @property
    def capacity(self) -> int:
        return self.population.particles.capacity

    def initialize(
        self,
        position: ArrayLike,
        proper_velocity: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        masses: ArrayLike | None = None,
    ) -> PICSpeciesState:
        """Initial species state; active particles receive identities in slot order."""
        population = self.population.initialize(active_mask=active_mask, masses=masses)
        position_ = jnp.asarray(position)
        velocity = jnp.asarray(proper_velocity, dtype=position_.dtype)
        if (
            position_.ndim != 2
            or position_.shape[0] != self.capacity
            or velocity.shape != (self.capacity, 3)
        ):
            raise ValueError(
                "Species position/proper velocity must have capacity-by-d and "
                "capacity-by-three shapes."
            )
        active = population.active[:, None]
        return PICSpeciesState(
            PICParticleState(
                jnp.where(active, position_, 0.0), jnp.where(active, velocity, 0.0)
            ),
            population,
            self.charge_model.initialize(population),
        )

    def macrocharge(self, state: PICSpeciesState, /) -> Array:
        return self.charge_model.macrocharge(state.population, state.charge)

    def specific_charge(self, state: PICSpeciesState, /) -> Array:
        """Runtime charge-to-mass ratio; zero on inactive slots."""
        return jnp.where(
            state.population.active,
            self.charge_model.base_specific_charge
            * state.charge.charge_number.astype(state.population.mass.dtype),
            0.0,
        )


__all__ = [
    "PICChargeModelPlan",
    "PICChargeState",
    "PICChargeTransitionResult",
    "PICSpeciesPlan",
    "PICSpeciesState",
]
