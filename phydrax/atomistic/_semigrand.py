#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleNeighborhoodState
from ._potential_program import PreparedAtomisticPotentialProgram


class VarianceConstrainedSemiGrandPlan(StrictModule, NonTrainableState):
    temperature: float = eqx.field(static=True)
    chemical_potentials: Array
    target_fractions: Array
    kappa: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperature: float,
        chemical_potentials: ArrayLike,
        target_fractions: ArrayLike,
        kappa: float,
        /,
        *,
        realization_id: int = 0,
    ):
        thermal = float(temperature)
        chemical = np.asarray(chemical_potentials, dtype=float)
        target = np.asarray(target_fractions, dtype=float)
        strength = float(kappa)
        realization = int(realization_id)
        if (
            not math.isfinite(thermal)
            or thermal <= 0.0
            or chemical.ndim != 1
            or chemical.size < 2
            or target.shape != chemical.shape
            or np.any(~np.isfinite(chemical))
            or np.any(~np.isfinite(target))
            or np.any(target < 0.0)
            or not np.isclose(np.sum(target), 1.0)
            or not math.isfinite(strength)
            or strength < 0.0
            or realization < 0
        ):
            raise ValueError("Variance-constrained semi-grand parameters are invalid.")
        self.temperature = thermal
        self.chemical_potentials = jnp.asarray(chemical)
        self.target_fractions = jnp.asarray(target)
        self.kappa = strength
        self.realization_id = realization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variance-constrained-semi-grand",
                "temperature": thermal,
                "chemical_potentials": chemical.tolist(),
                "target_fractions": target.tolist(),
                "kappa": strength,
                "realization_id": realization,
            }
        )


class SemiGrandTransition(StrictModule):
    species: Array
    accepted: Array
    particle_index: Array
    previous_species: Array
    proposed_species: Array
    energy_delta: Array
    bias_delta: Array
    log_acceptance_probability: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def variance_constrained_semigrand_step(
    potential: PreparedAtomisticPotentialProgram,
    positions: ArrayLike,
    neighborhood: ParticleNeighborhoodState,
    species: ArrayLike,
    key_data: ArrayLike,
    step_index: ArrayLike,
    plan: VarianceConstrainedSemiGrandPlan,
    /,
) -> SemiGrandTransition:
    if not isinstance(potential, PreparedAtomisticPotentialProgram):
        raise TypeError("potential must be PreparedAtomisticPotentialProgram.")
    if not isinstance(plan, VarianceConstrainedSemiGrandPlan):
        raise TypeError("plan must be VarianceConstrainedSemiGrandPlan.")
    if not potential.plan.capabilities.dynamic_species:
        raise ValueError("Potential program does not support dynamic species.")
    current = jnp.asarray(species, dtype=jnp.int32)
    capacity = potential.system.capacity
    if current.shape != (capacity,):
        raise ValueError("species must match the atomistic capacity.")
    key = jr.wrap_key_data(jnp.asarray(key_data, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.asarray(plan.realization_id, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.asarray(step_index, dtype=jnp.uint32))
    site_key, species_key, accept_key = jr.split(key, 3)
    active_indices = jnp.nonzero(
        potential.system.active_mask, size=capacity, fill_value=0
    )[0]
    active_count = jnp.sum(potential.system.active_mask, dtype=jnp.int32)
    selected_rank = jr.randint(site_key, (), 0, active_count)
    particle_index = active_indices[selected_rank]
    previous_species = current[particle_index]
    species_count = int(plan.chemical_potentials.size)
    offset = jr.randint(species_key, (), 1, species_count, dtype=jnp.int32)
    proposed_species = (
        (previous_species + offset) % jnp.asarray(species_count, dtype=jnp.int32)
    ).astype(jnp.int32)
    proposed = current.at[particle_index].set(proposed_species)
    current_energy = potential.energy(positions, neighborhood, species=current)[0]
    proposed_energy = potential.energy(positions, neighborhood, species=proposed)[0]
    energy_delta = proposed_energy - current_energy
    counts = jnp.bincount(current, length=species_count).astype(
        jnp.asarray(positions).dtype
    )
    proposed_counts = counts.at[previous_species].add(-1.0).at[proposed_species].add(1.0)
    fraction = counts / active_count
    proposed_fraction = proposed_counts / active_count
    current_bias = (
        plan.kappa * active_count * jnp.sum((fraction - plan.target_fractions) ** 2)
    )
    proposed_bias = (
        plan.kappa
        * active_count
        * jnp.sum((proposed_fraction - plan.target_fractions) ** 2)
    )
    chemical_delta = (
        plan.chemical_potentials[proposed_species]
        - plan.chemical_potentials[previous_species]
    )
    bias_delta = proposed_bias - current_bias - chemical_delta
    beta = 1.0 / (potential.system.plan.units.boltzmann_constant * plan.temperature)
    log_acceptance = -beta * (energy_delta + bias_delta)
    accepted = jnp.log(jr.uniform(accept_key, ())) < jnp.minimum(log_acceptance, 0.0)
    successful = (
        jnp.isfinite(energy_delta)
        & jnp.isfinite(bias_delta)
        & jnp.isfinite(log_acceptance)
    )
    accepted = accepted & successful
    return SemiGrandTransition(
        jnp.where(accepted, proposed, current),
        accepted,
        particle_index,
        previous_species,
        proposed_species,
        energy_delta,
        bias_delta,
        log_acceptance,
        successful,
        plan.plan_id,
    )


__all__ = [
    "SemiGrandTransition",
    "VarianceConstrainedSemiGrandPlan",
    "variance_constrained_semigrand_step",
]
