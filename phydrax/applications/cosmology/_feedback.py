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
from ...events import (
    EVENT_COMMITTED,
    EVENT_DEFERRED,
    EVENT_REJECTED,
    FixedCapacityEventState,
)


POPULATION_INACTIVE = 0
POPULATION_DARK_MATTER = 1
POPULATION_STAR = 2
POPULATION_BLACK_HOLE = 3


class CosmologicalPopulationState(StrictModule):
    global_ids: Array
    generations: Array
    kinds: Array
    active_mask: Array
    positions: Array
    canonical_momenta: Array
    gravitating_masses: Array
    birth_scale_factors: Array
    initial_masses: Array
    returned_masses: Array
    metallicities: Array
    energy_reservoirs: Array


class CosmologicalPopulationPlan(StrictModule, NonTrainableState):
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, dimension: int, /):
        capacity_ = int(capacity)
        dimension_ = int(dimension)
        if capacity_ <= 0 or dimension_ not in (1, 2, 3):
            raise ValueError("Cosmological population capacity/dimension is invalid.")
        self.capacity = capacity_
        self.dimension = dimension_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-population",
                "capacity": capacity_,
                "dimension": dimension_,
            }
        )

    def empty(self, *, dtype=jnp.float64) -> CosmologicalPopulationState:
        return CosmologicalPopulationState(
            jnp.arange(self.capacity, dtype=jnp.int64),
            jnp.zeros((self.capacity,), dtype=jnp.int32),
            jnp.zeros((self.capacity,), dtype=jnp.int8),
            jnp.zeros((self.capacity,), dtype=bool),
            jnp.zeros((self.capacity, self.dimension), dtype=dtype),
            jnp.zeros((self.capacity, self.dimension), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=dtype),
            jnp.zeros((self.capacity,), dtype=dtype),
        )


class FeedbackEventLedger(StrictModule):
    source_ids: Array
    recipient_ids: Array
    channels: Array
    requested_mass: Array
    coupled_mass: Array
    requested_energy: Array
    coupled_energy: Array
    committed: Array
    deferred: Array
    rejected: Array
    event_count: Array
    overflow: Array

    def as_event_state(self, /) -> FixedCapacityEventState:
        active = self.committed | self.deferred | self.rejected
        statuses = jnp.where(
            self.committed,
            EVENT_COMMITTED,
            jnp.where(self.deferred, EVENT_DEFERRED, EVENT_REJECTED),
        )
        return FixedCapacityEventState(
            self.source_ids,
            self.recipient_ids,
            self.channels,
            statuses,
            active,
            self.overflow,
        )


class StarFormationResult(StrictModule):
    gas_masses: Array
    gas_momenta: Array
    gas_energies: Array
    population: CosmologicalPopulationState
    ledger: FeedbackEventLedger
    mass_defect: Array
    momentum_defect: Array
    energy_defect: Array
    successful: Array


class StochasticStarFormationPlan(StrictModule, NonTrainableState):
    star_mass: float = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    process_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        star_mass: float,
        maximum_events: int,
        process_id: int = 1,
    ):
        mass = float(star_mass)
        events = int(maximum_events)
        process = int(process_id)
        if not np.isfinite(mass) or mass <= 0.0 or events <= 0 or process < 0:
            raise ValueError("Star-formation event policy is invalid.")
        self.star_mass = mass
        self.maximum_events = events
        self.process_id = process
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stochastic-star-formation",
                "star_mass": mass,
                "maximum_events": events,
                "process_id": process,
            }
        )

    def apply(
        self,
        population: CosmologicalPopulationState,
        gas_masses: ArrayLike,
        gas_momenta: ArrayLike,
        gas_energies: ArrayLike,
        gas_metallicities: ArrayLike,
        cell_positions: ArrayLike,
        eligible: ArrayLike,
        scale_factor: ArrayLike,
        key: Array,
        macroepoch: int,
        /,
    ) -> StarFormationResult:
        masses = jnp.asarray(gas_masses)
        momenta = jnp.asarray(gas_momenta, dtype=masses.dtype)
        energies = jnp.asarray(gas_energies, dtype=masses.dtype)
        metallicities = jnp.asarray(gas_metallicities, dtype=masses.dtype)
        positions = jnp.asarray(cell_positions, dtype=masses.dtype)
        eligible_ = jnp.asarray(eligible, dtype=bool)
        scale = jnp.asarray(scale_factor, dtype=masses.dtype)
        if (
            masses.ndim != 1
            or momenta.shape != positions.shape
            or momenta.shape[0] != masses.size
            or energies.shape != masses.shape
            or metallicities.shape != masses.shape
            or eligible_.shape != masses.shape
            or positions.shape[1] != population.positions.shape[1]
            or scale.shape != ()
        ):
            raise ValueError("Star-formation gas/population shapes are inconsistent.")
        probability = jnp.clip(masses / self.star_mass, 0.0, 1.0)
        cell_keys = jax.random.split(
            jax.random.fold_in(key, int(macroepoch)), masses.size
        )
        draws = jax.vmap(lambda local_key: jax.random.uniform(local_key))(cell_keys)
        requested = eligible_ & (masses >= self.star_mass) & (draws < probability)
        requested_indices = jnp.nonzero(
            requested, size=self.maximum_events, fill_value=-1
        )[0]
        free_indices = jnp.nonzero(
            ~population.active_mask, size=self.maximum_events, fill_value=-1
        )[0]
        valid = (requested_indices >= 0) & (free_indices >= 0)
        event_count = jnp.sum(valid)
        overflow = jnp.sum(requested) > self.maximum_events
        safe_cell = jnp.where(valid, requested_indices, 0)
        safe_slot = jnp.where(valid, free_indices, 0)
        coupled_mass = jnp.where(valid, self.star_mass, 0.0)
        gas_velocity = momenta[safe_cell] / masses[safe_cell, None]
        specific_energy = energies[safe_cell] / masses[safe_cell]
        coupled_momentum = coupled_mass[:, None] * gas_velocity
        coupled_energy = coupled_mass * specific_energy
        delta_mass = jnp.zeros_like(masses).at[safe_cell].add(-coupled_mass)
        delta_momentum = jnp.zeros_like(momenta).at[safe_cell].add(-coupled_momentum)
        delta_energy = jnp.zeros_like(energies).at[safe_cell].add(-coupled_energy)
        candidate_mass = masses + delta_mass
        candidate_momentum = momenta + delta_momentum
        candidate_energy = energies + delta_energy
        admissible = (
            ~overflow
            & jnp.all(candidate_mass >= 0.0)
            & jnp.all(candidate_energy >= 0.0)
            & jnp.all(jnp.isfinite(candidate_momentum))
        )
        active = population.active_mask.at[safe_slot].set(
            population.active_mask[safe_slot] | (valid & admissible)
        )
        kinds = population.kinds.at[safe_slot].set(
            jnp.where(valid & admissible, POPULATION_STAR, population.kinds[safe_slot])
        )
        generations = population.generations.at[safe_slot].add(
            (valid & admissible).astype(jnp.int32)
        )
        star_mass = population.gravitating_masses.at[safe_slot].add(
            jnp.where(admissible, coupled_mass, 0.0)
        )
        star_initial = population.initial_masses.at[safe_slot].add(
            jnp.where(admissible, coupled_mass, 0.0)
        )
        star_position = population.positions.at[safe_slot].set(
            jnp.where(
                (valid & admissible)[:, None],
                positions[safe_cell],
                population.positions[safe_slot],
            )
        )
        star_momentum = population.canonical_momenta.at[safe_slot].add(
            jnp.where(admissible, coupled_momentum, 0.0)
        )
        star_birth = population.birth_scale_factors.at[safe_slot].set(
            jnp.where(
                valid & admissible, scale, population.birth_scale_factors[safe_slot]
            )
        )
        star_metallicity = population.metallicities.at[safe_slot].set(
            jnp.where(
                valid & admissible,
                metallicities[safe_cell],
                population.metallicities[safe_slot],
            )
        )
        updated_population = CosmologicalPopulationState(
            population.global_ids,
            generations,
            kinds,
            active,
            star_position,
            star_momentum,
            star_mass,
            star_birth,
            star_initial,
            population.returned_masses,
            star_metallicity,
            population.energy_reservoirs,
        )
        accepted_mass = jnp.where(admissible, candidate_mass, masses)
        accepted_momentum = jnp.where(admissible, candidate_momentum, momenta)
        accepted_energy = jnp.where(admissible, candidate_energy, energies)
        ledger = FeedbackEventLedger(
            population.global_ids[safe_slot],
            safe_cell,
            jnp.full((self.maximum_events,), self.process_id, dtype=jnp.int32),
            jnp.where(valid, self.star_mass, 0.0),
            jnp.where(admissible, coupled_mass, 0.0),
            coupled_energy,
            jnp.where(admissible, coupled_energy, 0.0),
            valid & admissible,
            valid & ~admissible,
            ~valid,
            event_count,
            overflow,
        )
        mass_defect = (
            jnp.sum(accepted_mass)
            + jnp.sum(updated_population.gravitating_masses)
            - jnp.sum(masses)
            - jnp.sum(population.gravitating_masses)
        )
        momentum_defect = (
            jnp.sum(accepted_momentum, axis=0)
            + jnp.sum(updated_population.canonical_momenta, axis=0)
            - jnp.sum(momenta, axis=0)
            - jnp.sum(population.canonical_momenta, axis=0)
        )
        energy_defect = (
            jnp.sum(accepted_energy)
            + jnp.sum(jnp.where(ledger.committed, ledger.coupled_energy, 0.0))
            - jnp.sum(energies)
        )
        successful = admissible & jnp.all(jnp.isfinite(momentum_defect))
        return StarFormationResult(
            accepted_mass,
            accepted_momentum,
            accepted_energy,
            updated_population,
            ledger,
            mass_defect,
            momentum_defect,
            energy_defect,
            successful,
        )


class ThermalFeedbackResult(StrictModule):
    gas_energies: Array
    population: CosmologicalPopulationState
    ledger: FeedbackEventLedger
    energy_defect: Array
    successful: Array


class StochasticThermalFeedbackPlan(StrictModule, NonTrainableState):
    heating_energy_per_mass: float = eqx.field(static=True)
    maximum_events: int = eqx.field(static=True)
    process_id: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        heating_energy_per_mass: float,
        maximum_events: int,
        process_id: int = 2,
    ):
        energy = float(heating_energy_per_mass)
        events = int(maximum_events)
        process = int(process_id)
        if not np.isfinite(energy) or energy <= 0.0 or events <= 0:
            raise ValueError("Thermal-feedback policy is invalid.")
        self.heating_energy_per_mass = energy
        self.maximum_events = events
        self.process_id = process

    def apply(
        self,
        population: CosmologicalPopulationState,
        gas_masses: ArrayLike,
        gas_energies: ArrayLike,
        neighbor_indices: ArrayLike,
        key: Array,
        macroepoch: int,
        /,
    ) -> ThermalFeedbackResult:
        masses = jnp.asarray(gas_masses)
        energies = jnp.asarray(gas_energies, dtype=masses.dtype)
        neighbors = jnp.asarray(neighbor_indices, dtype=jnp.int32)
        star_mask = population.active_mask & (population.kinds == POPULATION_STAR)
        star_indices = jnp.nonzero(star_mask, size=self.maximum_events, fill_value=-1)[0]
        valid_star = star_indices >= 0
        safe_star = jnp.where(valid_star, star_indices, 0)
        local_neighbors = neighbors[safe_star]
        event_keys = jax.random.split(
            jax.random.fold_in(key, int(macroepoch)), self.maximum_events
        )
        rank = jax.vmap(
            lambda local_key: jax.random.randint(local_key, (), 0, neighbors.shape[1])
        )(event_keys)
        recipient = local_neighbors[jnp.arange(self.maximum_events), rank]
        valid_recipient = valid_star & (recipient >= 0) & (recipient < masses.size)
        safe_recipient = jnp.where(valid_recipient, recipient, 0)
        requested = masses[safe_recipient] * self.heating_energy_per_mass
        available = population.energy_reservoirs[safe_star]
        committed_energy = jnp.where(
            valid_recipient & (available >= requested), requested, 0.0
        )
        candidate_energy = energies.at[safe_recipient].add(committed_energy)
        reservoir = population.energy_reservoirs.at[safe_star].add(-committed_energy)
        updated_population = CosmologicalPopulationState(
            population.global_ids,
            population.generations,
            population.kinds,
            population.active_mask,
            population.positions,
            population.canonical_momenta,
            population.gravitating_masses,
            population.birth_scale_factors,
            population.initial_masses,
            population.returned_masses,
            population.metallicities,
            reservoir,
        )
        committed = committed_energy > 0.0
        ledger = FeedbackEventLedger(
            population.global_ids[safe_star],
            safe_recipient,
            jnp.full((self.maximum_events,), self.process_id, dtype=jnp.int32),
            jnp.zeros((self.maximum_events,), dtype=masses.dtype),
            jnp.zeros((self.maximum_events,), dtype=masses.dtype),
            requested,
            committed_energy,
            committed,
            valid_recipient & ~committed,
            ~valid_recipient,
            jnp.sum(committed),
            jnp.asarray(False),
        )
        energy_defect = (
            jnp.sum(candidate_energy)
            + jnp.sum(reservoir)
            - jnp.sum(energies)
            - jnp.sum(population.energy_reservoirs)
        )
        successful = jnp.all(jnp.isfinite(candidate_energy)) & jnp.all(reservoir >= 0.0)
        return ThermalFeedbackResult(
            candidate_energy,
            updated_population,
            ledger,
            energy_defect,
            successful,
        )


__all__ = [
    "CosmologicalPopulationPlan",
    "CosmologicalPopulationState",
    "FeedbackEventLedger",
    "POPULATION_BLACK_HOLE",
    "POPULATION_DARK_MATTER",
    "POPULATION_INACTIVE",
    "POPULATION_STAR",
    "StarFormationResult",
    "StochasticStarFormationPlan",
    "StochasticThermalFeedbackPlan",
    "ThermalFeedbackResult",
]
