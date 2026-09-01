#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._particle_internal_mesh import (
    ParticleInternalGeometry,
    PreparedParticleInternalBatch,
)
from ._particle_internal_state import (
    ParticleConversionState,
    ParticleInternalBatchState,
)
from ._population import ParticlePopulationState


class ParticleDynamicBodyProperties(StrictModule):
    population: ParticlePopulationState
    inverse_masses: Array
    radii: Array
    inertias: Array
    inverse_inertias: Array

    @property
    def masses(self) -> Array:
        return self.population.mass

    @property
    def active(self) -> Array:
        return self.population.active


class ParticleMorphologyEvaluation(StrictModule):
    batch_states: tuple[ParticleInternalBatchState, ...]
    body_properties: ParticleDynamicBodyProperties
    displaced_volume: Array
    mass_residual: Array
    volume_residual: Array
    neighborhood_rebuild_required: Array
    minimum_scale_margin: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DensityPorosityMorphologyPlan(StrictModule, NonTrainableState):
    solid_density: tuple[Array, ...]
    neighborhood_skin: float = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    maximum_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        solid_density,
        /,
        *,
        neighborhood_skin: float,
        minimum_scale: float = 1.0e-12,
        maximum_scale: float = 1.0e12,
        plan_id: str | None = None,
    ):
        densities = tuple(np.asarray(value, dtype=float) for value in solid_density)
        skin = float(neighborhood_skin)
        minimum = float(minimum_scale)
        maximum = float(maximum_scale)
        if not densities or any(
            value.ndim != 1
            or value.size == 0
            or np.any(~np.isfinite(value))
            or np.any(value <= 0.0)
            for value in densities
        ):
            raise ValueError("solid_density must contain positive species arrays.")
        if (
            not np.isfinite(skin)
            or skin <= 0.0
            or not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or not 0.0 < minimum < maximum
        ):
            raise ValueError("Morphology skin and scale bounds are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "density-porosity-morphology",
                "solid_density": [array_tree_fingerprint(value) for value in densities],
                "neighborhood_skin": skin,
                "bounds": [minimum, maximum],
            }
        )
        self.solid_density = tuple(jnp.asarray(value) for value in densities)
        self.neighborhood_skin = skin
        self.minimum_scale = minimum
        self.maximum_scale = maximum
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def evaluate(
        self,
        batches: tuple[PreparedParticleInternalBatch, ...],
        state: ParticleConversionState,
        molar_masses,
        /,
    ) -> ParticleMorphologyEvaluation:
        prepared_values = tuple(batches)
        mass_values = tuple(jnp.asarray(value) for value in molar_masses)
        if (
            len(prepared_values) != len(state.batches)
            or len(prepared_values) != len(self.solid_density)
            or len(prepared_values) != len(mass_values)
        ):
            raise ValueError("Morphology batch schemas must match conversion state.")
        capacity = prepared_values[0].particles.capacity
        owner_mass = jnp.zeros((capacity,), dtype=state.batches[0].internal_energy.dtype)
        owner_radius = jnp.zeros_like(owner_mass)
        owner_inertia = jnp.ones_like(owner_mass)
        owner_active = jnp.zeros((capacity,), dtype=bool)
        owner_volume = jnp.zeros_like(owner_mass)
        batch_states = []
        mass_residual = jnp.zeros((), dtype=owner_mass.dtype)
        volume_residual = jnp.zeros_like(mass_residual)
        rebuild = jnp.asarray(False)
        minimum_margin = jnp.asarray(jnp.inf, dtype=owner_mass.dtype)
        successful = jnp.asarray(True)
        for prepared, batch_state, density, masses in zip(
            prepared_values,
            state.batches,
            self.solid_density,
            mass_values,
            strict=True,
        ):
            if masses.shape != (prepared.species_count,) or density.shape != masses.shape:
                raise ValueError("Morphology species properties do not match batch.")
            species_mass = batch_state.species_amount * masses
            particle_mass = jnp.sum(species_mass, axis=(1, 2))
            species_volume = species_mass / density
            solid_volume = jnp.sum(species_volume, axis=(1, 2))
            metrics = prepared.mesh.metrics(batch_state.outer_scale)
            pore_weight = metrics.cell_measures
            average_porosity = jnp.sum(
                batch_state.porosity * pore_weight, axis=1
            ) / jnp.maximum(jnp.sum(pore_weight, axis=1), 1.0e-30)
            target_volume = solid_volume / jnp.maximum(1.0 - average_porosity, 1.0e-30)
            geometry = prepared.mesh.plan.geometry
            if geometry is ParticleInternalGeometry.SLAB:
                scale = target_volume / prepared.mesh.plan.transverse_measure
            elif geometry is ParticleInternalGeometry.CYLINDER:
                scale = jnp.sqrt(
                    target_volume / (jnp.pi * prepared.mesh.plan.transverse_measure)
                )
            else:
                scale = (3.0 * target_volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
            scale = jnp.where(batch_state.active, scale, 1.0)
            admissible_scale = (
                jnp.isfinite(scale)
                & (scale >= self.minimum_scale)
                & (scale <= self.maximum_scale)
            )
            new_metrics = prepared.mesh.metrics(scale)
            represented_volume = jnp.sum(new_metrics.cell_measures, axis=1)
            local_volume_residual = jnp.where(
                batch_state.active, represented_volume - target_volume, 0.0
            )
            local_mass_residual = jnp.where(
                batch_state.active,
                particle_mass - jnp.sum(species_mass, axis=(1, 2)),
                0.0,
            )
            changed = jnp.abs(scale - batch_state.outer_scale)
            rebuild = rebuild | jnp.any(
                batch_state.active & (changed > 0.5 * self.neighborhood_skin)
            )
            minimum_margin = jnp.minimum(
                minimum_margin,
                jnp.min(
                    jnp.where(
                        batch_state.active,
                        jnp.minimum(
                            scale - self.minimum_scale,
                            self.maximum_scale - scale,
                        ),
                        jnp.inf,
                    )
                ),
            )
            updated = eqx.tree_at(
                lambda value: value.outer_scale,
                batch_state,
                scale,
            )
            batch_states.append(updated)
            owner_mass = owner_mass.at[prepared.owner_indices].set(particle_mass)
            owner_radius = owner_radius.at[prepared.owner_indices].set(scale)
            owner_inertia = owner_inertia.at[prepared.owner_indices].set(
                0.4 * particle_mass * scale**2
            )
            owner_volume = owner_volume.at[prepared.owner_indices].set(target_volume)
            owner_active = owner_active.at[prepared.owner_indices].set(batch_state.active)
            mass_residual = mass_residual + jnp.sum(local_mass_residual)
            volume_residual = volume_residual + jnp.sum(local_volume_residual)
            successful = (
                successful
                & metrics.successful
                & new_metrics.successful
                & jnp.all(~batch_state.active | admissible_scale)
                & jnp.all(jnp.isfinite(particle_mass) & (particle_mass >= 0.0))
            )
        inverse_mass = jnp.where(owner_active & (owner_mass > 0.0), 1.0 / owner_mass, 0.0)
        inverse_inertia = jnp.where(
            owner_active & (owner_inertia > 0.0), 1.0 / owner_inertia, 0.0
        )
        population = ParticlePopulationState(
            owner_active,
            owner_mass,
            jnp.where(owner_active, 1, 0).astype(jnp.int32),
            owner_active,
            jnp.zeros_like(owner_active),
        )
        properties = ParticleDynamicBodyProperties(
            population,
            inverse_mass,
            owner_radius,
            owner_inertia,
            inverse_inertia,
        )
        tolerance = 128.0 * jnp.finfo(owner_mass.dtype).eps
        successful = (
            successful
            & (
                jnp.abs(mass_residual)
                <= tolerance * jnp.maximum(jnp.sum(owner_mass), 1.0)
            )
            & (
                jnp.abs(volume_residual)
                <= tolerance * jnp.maximum(jnp.sum(owner_volume), 1.0)
            )
        )
        return ParticleMorphologyEvaluation(
            tuple(batch_states),
            properties,
            owner_volume,
            mass_residual,
            volume_residual,
            rebuild,
            minimum_margin,
            successful,
            self.plan_id,
        )

    def apply(
        self,
        state: ParticleConversionState,
        evaluation: ParticleMorphologyEvaluation,
        /,
    ) -> ParticleConversionState:
        if evaluation.plan_id != self.plan_id:
            raise ValueError("Morphology evaluation does not match plan.")
        return ParticleConversionState(
            evaluation.batch_states,
            state.ledger,
            state.state_id,
        )


class ThermochemicalFragmentationPlan(StrictModule, NonTrainableState):
    maximum_children: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_children: int, /, *, tolerance: float = 1.0e-10):
        children = int(maximum_children)
        tolerance_ = float(tolerance)
        if children < 2 or not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Fragmentation child capacity/tolerance is invalid.")
        self.maximum_children = children
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "thermochemical-fragmentation-plan",
                "maximum_children": children,
                "tolerance": tolerance_,
            }
        )


class ThermochemicalFragmentationEvaluation(StrictModule):
    candidate_state: ParticleInternalBatchState
    mass_residual: Array
    energy_residual: Array
    species_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def fragment_particle_internal_batch(
    plan: ThermochemicalFragmentationPlan,
    state: ParticleInternalBatchState,
    source_index: Array,
    child_indices: ArrayLike,
    child_valid: ArrayLike,
    child_masses: ArrayLike,
    child_outer_scale: ArrayLike,
    molar_masses: ArrayLike,
    /,
) -> ThermochemicalFragmentationEvaluation:
    if not isinstance(plan, ThermochemicalFragmentationPlan):
        raise TypeError("plan must be ThermochemicalFragmentationPlan.")
    source = jnp.asarray(source_index, dtype=jnp.int32)
    children = jnp.asarray(child_indices, dtype=jnp.int32)
    valid = jnp.asarray(child_valid, dtype=bool)
    masses = jnp.asarray(child_masses, dtype=state.internal_energy.dtype)
    scales = jnp.asarray(child_outer_scale, dtype=state.outer_scale.dtype)
    molar = jnp.asarray(molar_masses, dtype=state.species_amount.dtype)
    expected = (plan.maximum_children,)
    if (
        children.shape != expected
        or valid.shape != expected
        or masses.shape != expected
        or scales.shape != expected
        or molar.shape != (state.species_amount.shape[-1],)
    ):
        raise ValueError("Fragmentation arrays do not match plan/schema.")
    safe_children = jnp.where(valid, children, 0)
    indices_valid = (children >= 0) & (children < state.active.shape[0]) & valid
    source_mass = jnp.sum(state.species_amount[source] * molar)
    assigned_mass = jnp.sum(jnp.where(valid, masses, 0.0))
    weights = jnp.where(
        valid,
        masses / jnp.maximum(assigned_mass, 1.0e-30),
        0.0,
    )
    child_energy = weights[:, None] * state.internal_energy[source][None, :]
    child_species = weights[:, None, None] * state.species_amount[source][None, :, :]
    energy = state.internal_energy.at[source].set(0.0)
    species = state.species_amount.at[source].set(0.0)
    active = state.active.at[source].set(False)
    porosity = state.porosity
    internal_area = state.internal_surface_area
    outer_scale = state.outer_scale
    front = state.reaction_front
    for index in range(plan.maximum_children):
        child = safe_children[index]
        child_active = valid[index] & indices_valid[index]
        energy = energy.at[child].set(
            jnp.where(child_active, child_energy[index], energy[child])
        )
        species = species.at[child].set(
            jnp.where(
                child_active,
                child_species[index],
                species[child],
            )
        )
        porosity = porosity.at[child].set(
            jnp.where(child_active, state.porosity[source], porosity[child])
        )
        internal_area = internal_area.at[child].set(
            jnp.where(
                child_active,
                weights[index] * state.internal_surface_area[source],
                internal_area[child],
            )
        )
        outer_scale = outer_scale.at[child].set(
            jnp.where(child_active, scales[index], outer_scale[child])
        )
        front = front.at[child].set(
            jnp.where(child_active, state.reaction_front[source], front[child])
        )
        active = active.at[child].set(child_active | active[child])
    candidate = ParticleInternalBatchState(
        energy,
        species,
        porosity,
        internal_area,
        outer_scale,
        front,
        active,
        state.batch_id,
    )
    mass_residual = assigned_mass - source_mass
    energy_residual = jnp.sum(jnp.where(valid[:, None], child_energy, 0.0)) - jnp.sum(
        state.internal_energy[source]
    )
    species_residual = jnp.sum(
        jnp.where(valid[:, None, None], child_species, 0.0), axis=0
    ) - jnp.sum(state.species_amount[source], axis=0)
    successful = (
        state.active[source]
        & jnp.all(indices_valid | ~valid)
        & jnp.all(~state.active[safe_children] | ~valid)
        & jnp.all(jnp.isfinite(masses) & (~valid | (masses > 0.0)))
        & jnp.all(jnp.isfinite(scales) & (~valid | (scales > 0.0)))
        & (jnp.abs(mass_residual) <= plan.tolerance * jnp.maximum(source_mass, 1.0))
        & (
            jnp.abs(energy_residual)
            <= plan.tolerance
            * jnp.maximum(jnp.sum(jnp.abs(state.internal_energy[source])), 1.0)
        )
        & jnp.all(
            jnp.abs(species_residual)
            <= plan.tolerance
            * jnp.maximum(
                jnp.sum(jnp.abs(state.species_amount[source]), axis=0),
                1.0,
            )
        )
    )
    return ThermochemicalFragmentationEvaluation(
        candidate,
        mass_residual,
        energy_residual,
        species_residual,
        successful,
        plan.plan_id,
    )


class ParticleDeactivationResult(StrictModule):
    candidate_state: ParticleInternalBatchState
    released_internal_energy: Array
    released_species_amount: Array
    successful: Array


def deactivate_particle_internal_state(
    state: ParticleInternalBatchState,
    particle_index: Array,
    /,
) -> ParticleDeactivationResult:
    index = jnp.asarray(particle_index, dtype=jnp.int32)
    energy = jnp.sum(state.internal_energy[index])
    species = jnp.sum(state.species_amount[index], axis=0)
    candidate = ParticleInternalBatchState(
        state.internal_energy.at[index].set(0.0),
        state.species_amount.at[index].set(0.0),
        state.porosity.at[index].set(0.0),
        state.internal_surface_area.at[index].set(0.0),
        state.outer_scale.at[index].set(1.0),
        state.reaction_front.at[index].set(0.0),
        state.active.at[index].set(False),
        state.batch_id,
    )
    successful = (
        state.active[index] & jnp.isfinite(energy) & jnp.all(jnp.isfinite(species))
    )
    return ParticleDeactivationResult(candidate, energy, species, successful)


__all__ = [
    "DensityPorosityMorphologyPlan",
    "ParticleDeactivationResult",
    "ParticleDynamicBodyProperties",
    "ParticleMorphologyEvaluation",
    "ThermochemicalFragmentationEvaluation",
    "ThermochemicalFragmentationPlan",
    "deactivate_particle_internal_state",
    "fragment_particle_internal_batch",
]
