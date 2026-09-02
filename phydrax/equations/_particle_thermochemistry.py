#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle._particle_internal_mesh import (
    PreparedParticleInternalBatch,
)
from ..discretization.particle._particle_internal_state import ParticleInternalBatchState
from ..discretization.particle._particle_internal_unstructured import (
    PreparedUnstructuredParticleInternalMesh,
)
from ._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from ._chemical_thermodynamics import AbstractSpeciesThermodynamicsPlan


_UNIVERSAL_GAS_CONSTANT = 8.31446261815324


@jax.custom_jvp
def _implicit_temperature_value(
    temperature,
    internal_energy,
    species_amount,
    molar_internal_energy,
    molar_heat_capacity,
):
    del internal_energy, species_amount, molar_internal_energy, molar_heat_capacity
    return temperature


@_implicit_temperature_value.defjvp
def _implicit_temperature_value_jvp(primals, tangents):
    temperature, internal_energy, species_amount, molar_energy, molar_capacity = primals
    _, energy_tangent, amount_tangent, molar_energy_tangent, _ = tangents
    capacity = jnp.sum(species_amount * molar_capacity, axis=-1)
    tangent = (
        energy_tangent
        - jnp.sum(amount_tangent * molar_energy, axis=-1)
        - jnp.sum(species_amount * molar_energy_tangent, axis=-1)
    ) / jnp.maximum(capacity, jnp.finfo(capacity.dtype).tiny)
    return temperature, tangent


class ParticleThermodynamicState(StrictModule):
    temperature: Array
    heat_capacity: Array
    gas_pressure: Array
    gas_amount: Array
    energy_residual: Array
    temperature_margin: Array
    successful: Array


class ParticleThermodynamicMaterialPlan(StrictModule, NonTrainableState):
    schema: ChemicalSpeciesSchema
    species_thermodynamics: AbstractSpeciesThermodynamicsPlan
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    inversion_iterations: int = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_thermodynamics: AbstractSpeciesThermodynamicsPlan,
        /,
        *,
        minimum_temperature: float | None = None,
        maximum_temperature: float | None = None,
        inversion_iterations: int = 48,
        material_id: str | None = None,
    ):
        if not isinstance(species_thermodynamics, AbstractSpeciesThermodynamicsPlan):
            raise TypeError(
                "species_thermodynamics must implement AbstractSpeciesThermodynamicsPlan."
            )
        t_min = (
            species_thermodynamics.minimum_temperature
            if minimum_temperature is None
            else float(minimum_temperature)
        )
        t_max = (
            species_thermodynamics.maximum_temperature
            if maximum_temperature is None
            else float(maximum_temperature)
        )
        iterations = int(inversion_iterations)
        if (
            not np.isfinite(t_min)
            or not np.isfinite(t_max)
            or not 0.0 < t_min < t_max
            or t_min < species_thermodynamics.minimum_temperature
            or t_max > species_thermodynamics.maximum_temperature
            or iterations <= 0
        ):
            raise ValueError("Thermodynamic inversion controls are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "particle-thermodynamic-material",
                "thermodynamics": species_thermodynamics.thermodynamics_id,
                "bounds": [t_min, t_max],
                "iterations": iterations,
            }
        )
        self.schema = species_thermodynamics.schema
        self.species_thermodynamics = species_thermodynamics
        self.minimum_temperature = t_min
        self.maximum_temperature = t_max
        self.inversion_iterations = iterations
        self.material_id = generated if material_id is None else str(material_id)
        if not self.material_id:
            raise ValueError("material_id must be nonempty.")

    def molar_heat_capacity(self, temperature: ArrayLike, /) -> Array:
        return self.species_thermodynamics.evaluate(
            temperature
        ).molar_heat_capacity_volume

    def molar_internal_energy(self, temperature: ArrayLike, /) -> Array:
        return self.species_thermodynamics.evaluate(temperature).molar_internal_energy

    def energy_from_temperature(
        self, temperature: ArrayLike, species_amount: ArrayLike, /
    ) -> Array:
        amount = jnp.asarray(species_amount)
        return jnp.sum(amount * self.molar_internal_energy(temperature), axis=-1)

    def state(
        self,
        internal_energy: ArrayLike,
        species_amount: ArrayLike,
        cell_measure: ArrayLike,
        porosity: ArrayLike,
        /,
    ) -> ParticleThermodynamicState:
        energy = jnp.asarray(internal_energy)
        amount = jnp.asarray(species_amount, dtype=energy.dtype)
        measure = jnp.asarray(cell_measure, dtype=energy.dtype)
        pore = jnp.asarray(porosity, dtype=energy.dtype)
        if amount.shape != energy.shape + (self.schema.species_count,):
            raise ValueError("species_amount must extend internal-energy shape.")
        if measure.shape != energy.shape or pore.shape != energy.shape:
            raise ValueError("Cell measure and porosity must match energy shape.")
        low = jnp.full_like(energy, self.minimum_temperature)
        high = jnp.full_like(energy, self.maximum_temperature)
        low_energy = self.energy_from_temperature(low, amount)
        high_energy = self.energy_from_temperature(high, amount)
        energy_admissible = (
            jnp.isfinite(energy)
            & jnp.isfinite(low_energy)
            & jnp.isfinite(high_energy)
            & (energy >= low_energy)
            & (energy <= high_energy)
            & jnp.all(amount >= 0.0, axis=-1)
            & (jnp.sum(amount, axis=-1) > 0.0)
        )
        target = jnp.where(
            energy_admissible,
            energy,
            0.5 * (low_energy + high_energy),
        )

        def iteration(_, bracket):
            lower, upper = bracket
            midpoint = 0.5 * (lower + upper)
            midpoint_energy = self.energy_from_temperature(midpoint, amount)
            upper = jnp.where(midpoint_energy > target, midpoint, upper)
            lower = jnp.where(midpoint_energy > target, lower, midpoint)
            return lower, upper

        low, high = jax.lax.fori_loop(
            0,
            self.inversion_iterations,
            iteration,
            (low, high),
        )
        raw_temperature = 0.5 * (low + high)
        raw_thermo = self.species_thermodynamics.evaluate(raw_temperature)
        temperature = _implicit_temperature_value(
            raw_temperature,
            energy,
            amount,
            raw_thermo.molar_internal_energy,
            raw_thermo.molar_heat_capacity_volume,
        )
        thermo = self.species_thermodynamics.evaluate(temperature)
        capacity = jnp.sum(
            amount * thermo.molar_heat_capacity_volume,
            axis=-1,
        )
        recovered = jnp.sum(amount * thermo.molar_internal_energy, axis=-1)
        residual = recovered - energy
        gas_mask = self.schema.phase_mask(ChemicalPhaseKind.GAS).astype(amount.dtype)
        gas_amount = jnp.sum(amount * gas_mask, axis=-1)
        pore_volume = pore * measure
        pressure = (
            gas_amount
            * _UNIVERSAL_GAS_CONSTANT
            * temperature
            / jnp.maximum(pore_volume, jnp.finfo(energy.dtype).tiny)
        )
        bracket_error = capacity * (high - low)
        scale = jnp.maximum(jnp.abs(energy), 1.0)
        successful = (
            jnp.all(energy_admissible)
            & jnp.all(thermo.successful)
            & jnp.all(capacity > 0.0)
            & jnp.all(
                jnp.abs(residual)
                <= bracket_error + 256.0 * jnp.finfo(energy.dtype).eps * scale
            )
            & jnp.all(jnp.isfinite(pressure) & (pressure >= 0.0))
        )
        margin = jnp.minimum(
            temperature - self.minimum_temperature,
            self.maximum_temperature - temperature,
        )
        return ParticleThermodynamicState(
            temperature,
            capacity,
            pressure,
            gas_amount,
            residual,
            margin,
            successful,
        )


class ParticleTransportMaterialPlan(StrictModule, NonTrainableState):
    schema: ChemicalSpeciesSchema
    thermal_conductivity: Array
    species_diffusivity: Array
    tortuosity_exponent: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        thermal_conductivity: ArrayLike,
        species_diffusivity: ArrayLike,
        /,
        *,
        tortuosity_exponent: float = 1.0,
        material_id: str | None = None,
    ):
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be a ChemicalSpeciesSchema.")
        conductivity = np.asarray(thermal_conductivity, dtype=float)
        diffusivity = np.asarray(species_diffusivity, dtype=float)
        exponent = float(tortuosity_exponent)
        if (
            conductivity.shape != (schema.species_count,)
            or diffusivity.shape != (schema.species_count,)
            or np.any(~np.isfinite(conductivity))
            or np.any(conductivity < 0.0)
            or np.any(~np.isfinite(diffusivity))
            or np.any(diffusivity < 0.0)
            or not np.isfinite(exponent)
            or exponent < 0.0
        ):
            raise ValueError("Particle transport properties are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "particle-transport-material",
                "schema": schema.schema_id,
                "conductivity": array_tree_fingerprint(conductivity),
                "diffusivity": array_tree_fingerprint(diffusivity),
                "tortuosity_exponent": exponent,
            }
        )
        self.schema = schema
        self.thermal_conductivity = jnp.asarray(conductivity)
        self.species_diffusivity = jnp.asarray(diffusivity)
        self.tortuosity_exponent = exponent
        self.material_id = generated if material_id is None else str(material_id)
        if not self.material_id:
            raise ValueError("material_id must be nonempty.")


class ParticleThermochemicalMaterialBundle(StrictModule, NonTrainableState):
    thermodynamics: ParticleThermodynamicMaterialPlan
    transport: ParticleTransportMaterialPlan
    bundle_id: str = eqx.field(static=True)

    def __init__(self, thermodynamics, transport, /):
        if not isinstance(thermodynamics, ParticleThermodynamicMaterialPlan):
            raise TypeError("thermodynamics must be a ParticleThermodynamicMaterialPlan.")
        if not isinstance(transport, ParticleTransportMaterialPlan):
            raise TypeError("transport must be a ParticleTransportMaterialPlan.")
        if thermodynamics.schema.schema_id != transport.schema.schema_id:
            raise ValueError("Thermodynamic and transport species schemas must match.")
        self.thermodynamics = thermodynamics
        self.transport = transport
        self.bundle_id = canonical_fingerprint(
            {
                "kind": "particle-thermochemical-material-bundle",
                "thermodynamics": thermodynamics.material_id,
                "transport": transport.material_id,
            }
        )


class ParticleTransportBoundary(StrictModule):
    temperature: Array
    species_concentration: Array
    heat_transfer_coefficient: Array
    mass_transfer_coefficient: Array
    prescribed_heat_rate: Array
    prescribed_species_rate: Array


class ParticleTransportEvaluation(StrictModule):
    internal_energy_rate: Array
    species_amount_rate: Array
    boundary_heat_rate: Array
    boundary_species_rate: Array
    internal_energy_residual: Array
    internal_species_residual: Array
    entropy_production: Array
    explicit_step_restriction: Array
    thermodynamic_state: ParticleThermodynamicState
    successful: Array
    material_id: str = eqx.field(static=True)


def evaluate_particle_transport(
    batch: PreparedParticleInternalBatch,
    state: ParticleInternalBatchState,
    material: ParticleThermochemicalMaterialBundle,
    boundary: ParticleTransportBoundary,
    /,
) -> ParticleTransportEvaluation:
    if batch.prepared_id != state.batch_id:
        raise ValueError("Particle transport state does not match prepared batch.")
    if material.thermodynamics.schema.species_count != batch.species_count:
        raise ValueError("Particle transport species schema does not match batch.")
    if isinstance(batch.mesh, PreparedUnstructuredParticleInternalMesh):
        return _evaluate_unstructured_particle_transport(batch, state, material, boundary)
    metrics = batch.mesh.metrics(state.outer_scale)
    thermodynamics = material.thermodynamics.state(
        state.internal_energy,
        state.species_amount,
        metrics.cell_measures,
        state.porosity,
    )
    _validate_boundary(boundary, batch, state.internal_energy.dtype)
    concentration = state.species_amount / metrics.cell_measures[:, :, None]
    amount_sum = jnp.sum(state.species_amount, axis=-1, keepdims=True)
    fraction = state.species_amount / jnp.maximum(amount_sum, 1.0e-30)
    conductivity_cell = jnp.sum(
        fraction * material.transport.thermal_conductivity,
        axis=-1,
    )
    conductivity_face = _harmonic_mean(
        conductivity_cell[:, :-1], conductivity_cell[:, 1:]
    )
    heat_conductance = (
        conductivity_face * metrics.face_measures[:, 1:-1] / metrics.center_distances
    )
    heat_flux = heat_conductance * (
        thermodynamics.temperature[:, :-1] - thermodynamics.temperature[:, 1:]
    )
    energy_rate = jnp.zeros_like(state.internal_energy)
    energy_rate = energy_rate.at[:, :-1].add(-heat_flux)
    energy_rate = energy_rate.at[:, 1:].add(heat_flux)
    effective_diffusivity = (
        material.transport.species_diffusivity[None, None, :]
        * state.porosity[:, :, None] ** material.transport.tortuosity_exponent
    )
    diffusivity_face = _harmonic_mean(
        effective_diffusivity[:, :-1, :],
        effective_diffusivity[:, 1:, :],
    )
    species_conductance = (
        diffusivity_face
        * metrics.face_measures[:, 1:-1, None]
        / metrics.center_distances[:, :, None]
    )
    species_flux = species_conductance * (
        concentration[:, :-1, :] - concentration[:, 1:, :]
    )
    species_rate = jnp.zeros_like(state.species_amount)
    species_rate = species_rate.at[:, :-1, :].add(-species_flux)
    species_rate = species_rate.at[:, 1:, :].add(species_flux)
    surface_temperature = thermodynamics.temperature[:, -1]
    boundary_heat = (
        boundary.heat_transfer_coefficient
        * metrics.surface_measure
        * (boundary.temperature - surface_temperature)
        + boundary.prescribed_heat_rate
    )
    boundary_species = (
        boundary.mass_transfer_coefficient
        * metrics.surface_measure[:, None]
        * (boundary.species_concentration - concentration[:, -1, :])
        + boundary.prescribed_species_rate
    )
    boundary_heat = jnp.where(state.active, boundary_heat, 0.0)
    boundary_species = jnp.where(state.active[:, None], boundary_species, 0.0)
    energy_rate = energy_rate.at[:, -1].add(boundary_heat)
    species_rate = species_rate.at[:, -1, :].add(boundary_species)
    energy_residual = jnp.sum(energy_rate) - jnp.sum(boundary_heat)
    species_residual = jnp.sum(species_rate, axis=(0, 1)) - jnp.sum(
        boundary_species, axis=0
    )
    temperature_jump = (
        thermodynamics.temperature[:, :-1] - thermodynamics.temperature[:, 1:]
    )
    entropy = jnp.sum(
        heat_conductance
        * temperature_jump**2
        / jnp.maximum(
            thermodynamics.temperature[:, :-1] * thermodynamics.temperature[:, 1:],
            1.0e-30,
        )
    )
    heat_degree = jnp.zeros_like(state.internal_energy)
    heat_degree = heat_degree.at[:, :-1].add(heat_conductance)
    heat_degree = heat_degree.at[:, 1:].add(heat_conductance)
    heat_degree = heat_degree.at[:, -1].add(
        boundary.heat_transfer_coefficient * metrics.surface_measure
    )
    thermal_limit = jnp.where(
        heat_degree > 0.0,
        thermodynamics.heat_capacity / heat_degree,
        jnp.inf,
    )
    species_degree = jnp.zeros_like(state.species_amount)
    species_degree = species_degree.at[:, :-1, :].add(species_conductance)
    species_degree = species_degree.at[:, 1:, :].add(species_conductance)
    species_degree = species_degree.at[:, -1, :].add(
        boundary.mass_transfer_coefficient * metrics.surface_measure[:, None]
    )
    species_limit = jnp.where(
        species_degree > 0.0,
        metrics.cell_measures[:, :, None] / species_degree,
        jnp.inf,
    )
    boundary_valid = (
        jnp.all(jnp.isfinite(boundary.temperature) & (boundary.temperature > 0.0))
        & jnp.all(
            jnp.isfinite(boundary.heat_transfer_coefficient)
            & (boundary.heat_transfer_coefficient >= 0.0)
        )
        & jnp.all(
            jnp.isfinite(boundary.species_concentration)
            & (boundary.species_concentration >= 0.0)
        )
        & jnp.all(
            jnp.isfinite(boundary.mass_transfer_coefficient)
            & (boundary.mass_transfer_coefficient >= 0.0)
        )
        & jnp.all(jnp.isfinite(boundary.prescribed_heat_rate))
        & jnp.all(jnp.isfinite(boundary.prescribed_species_rate))
    )
    restriction = jnp.minimum(jnp.min(thermal_limit), jnp.min(species_limit))
    tolerance = 128.0 * jnp.finfo(state.internal_energy.dtype).eps
    successful = (
        metrics.successful
        & thermodynamics.successful
        & boundary_valid
        & jnp.all(jnp.isfinite(energy_rate))
        & jnp.all(jnp.isfinite(species_rate))
        & (
            jnp.abs(energy_residual)
            <= tolerance * jnp.maximum(jnp.sum(jnp.abs(boundary_heat)), 1.0)
        )
        & jnp.all(
            jnp.abs(species_residual)
            <= tolerance * jnp.maximum(jnp.sum(jnp.abs(boundary_species), axis=0), 1.0)
        )
        & jnp.isfinite(entropy)
        & (entropy >= 0.0)
        & ~jnp.isnan(restriction)
    )
    return ParticleTransportEvaluation(
        energy_rate,
        species_rate,
        boundary_heat,
        boundary_species,
        energy_residual,
        species_residual,
        entropy,
        restriction,
        thermodynamics,
        successful,
        material.bundle_id,
    )


def _evaluate_unstructured_particle_transport(
    batch,
    state,
    material,
    boundary,
):
    _validate_boundary(boundary, batch, state.internal_energy.dtype)
    active_cells = jnp.broadcast_to(
        state.active[:, None], (batch.particle_count, batch.cell_capacity)
    )
    metrics = batch.mesh.metrics(state.outer_scale, active_cells=active_cells)
    thermodynamics = material.thermodynamics.state(
        state.internal_energy,
        state.species_amount,
        metrics.cell_measures,
        state.porosity,
    )
    concentration = state.species_amount / metrics.cell_measures[:, :, None]
    amount_sum = jnp.sum(state.species_amount, axis=-1, keepdims=True)
    fraction = state.species_amount / jnp.maximum(amount_sum, 1.0e-30)
    conductivity_cell = jnp.sum(
        fraction * material.transport.thermal_conductivity, axis=-1
    )
    owner = metrics.owner_cells
    neighbour = metrics.neighbour_cells
    safe_neighbour = jnp.maximum(neighbour, 0)
    interior = (~metrics.boundary_faces)[None, :] & metrics.active_faces
    conductivity_face = _harmonic_mean(
        conductivity_cell[:, owner], conductivity_cell[:, safe_neighbour]
    )
    heat_conductance = (
        conductivity_face * metrics.face_measures / metrics.center_distances
    )
    heat_flux = jnp.where(
        interior,
        heat_conductance
        * (
            thermodynamics.temperature[:, owner]
            - thermodynamics.temperature[:, safe_neighbour]
        ),
        0.0,
    )
    energy_rate = jnp.zeros_like(state.internal_energy)
    energy_rate = energy_rate.at[:, owner].add(-heat_flux)
    energy_rate = energy_rate.at[:, safe_neighbour].add(heat_flux)
    effective_diffusivity = (
        material.transport.species_diffusivity[None, None, :]
        * state.porosity[:, :, None] ** material.transport.tortuosity_exponent
    )
    diffusivity_face = _harmonic_mean(
        effective_diffusivity[:, owner, :],
        effective_diffusivity[:, safe_neighbour, :],
    )
    species_conductance = (
        diffusivity_face
        * metrics.face_measures[:, :, None]
        / metrics.center_distances[:, :, None]
    )
    species_flux = jnp.where(
        interior[:, :, None],
        species_conductance
        * (concentration[:, owner, :] - concentration[:, safe_neighbour, :]),
        0.0,
    )
    species_rate = jnp.zeros_like(state.species_amount)
    species_rate = species_rate.at[:, owner, :].add(-species_flux)
    species_rate = species_rate.at[:, safe_neighbour, :].add(species_flux)
    boundary_faces = metrics.boundary_faces[None, :] & metrics.active_faces
    boundary_owner_temperature = thermodynamics.temperature[:, owner]
    boundary_owner_species = concentration[:, owner, :]
    area_fraction = metrics.face_measures / jnp.maximum(
        metrics.surface_measure[:, None], 1.0e-30
    )
    boundary_face_heat = jnp.where(
        boundary_faces,
        boundary.heat_transfer_coefficient[:, None]
        * metrics.face_measures
        * (boundary.temperature[:, None] - boundary_owner_temperature)
        + boundary.prescribed_heat_rate[:, None] * area_fraction,
        0.0,
    )
    boundary_face_species = jnp.where(
        boundary_faces[:, :, None],
        boundary.mass_transfer_coefficient[:, None, :]
        * metrics.face_measures[:, :, None]
        * (boundary.species_concentration[:, None, :] - boundary_owner_species)
        + boundary.prescribed_species_rate[:, None, :] * area_fraction[:, :, None],
        0.0,
    )
    energy_rate = energy_rate.at[:, owner].add(boundary_face_heat)
    species_rate = species_rate.at[:, owner, :].add(boundary_face_species)
    boundary_heat = jnp.sum(boundary_face_heat, axis=1)
    boundary_species = jnp.sum(boundary_face_species, axis=1)
    energy_residual = jnp.sum(energy_rate) - jnp.sum(boundary_heat)
    species_residual = jnp.sum(species_rate, axis=(0, 1)) - jnp.sum(
        boundary_species, axis=0
    )
    temperature_jump = (
        thermodynamics.temperature[:, owner]
        - thermodynamics.temperature[:, safe_neighbour]
    )
    entropy = jnp.sum(
        jnp.where(
            interior,
            heat_conductance
            * temperature_jump**2
            / jnp.maximum(
                thermodynamics.temperature[:, owner]
                * thermodynamics.temperature[:, safe_neighbour],
                1.0e-30,
            ),
            0.0,
        )
    )
    heat_degree = jnp.zeros_like(state.internal_energy)
    heat_degree = heat_degree.at[:, owner].add(
        jnp.where(interior, heat_conductance, 0.0)
        + jnp.where(
            boundary_faces,
            boundary.heat_transfer_coefficient[:, None] * metrics.face_measures,
            0.0,
        )
    )
    heat_degree = heat_degree.at[:, safe_neighbour].add(
        jnp.where(interior, heat_conductance, 0.0)
    )
    species_degree = jnp.zeros_like(state.species_amount)
    species_degree = species_degree.at[:, owner, :].add(
        jnp.where(interior[:, :, None], species_conductance, 0.0)
        + jnp.where(
            boundary_faces[:, :, None],
            boundary.mass_transfer_coefficient[:, None, :]
            * metrics.face_measures[:, :, None],
            0.0,
        )
    )
    species_degree = species_degree.at[:, safe_neighbour, :].add(
        jnp.where(interior[:, :, None], species_conductance, 0.0)
    )
    thermal_limit = jnp.where(
        heat_degree > 0.0,
        thermodynamics.heat_capacity / heat_degree,
        jnp.inf,
    )
    species_limit = jnp.where(
        species_degree > 0.0,
        metrics.cell_measures[:, :, None] / species_degree,
        jnp.inf,
    )
    restriction = jnp.minimum(jnp.min(thermal_limit), jnp.min(species_limit))
    tolerance = 128.0 * jnp.finfo(state.internal_energy.dtype).eps
    successful = (
        metrics.successful
        & thermodynamics.successful
        & jnp.all(jnp.isfinite(energy_rate))
        & jnp.all(jnp.isfinite(species_rate))
        & (
            jnp.abs(energy_residual)
            <= tolerance * jnp.maximum(jnp.sum(jnp.abs(boundary_heat)), 1.0)
        )
        & jnp.all(
            jnp.abs(species_residual)
            <= tolerance * jnp.maximum(jnp.sum(jnp.abs(boundary_species), axis=0), 1.0)
        )
        & jnp.isfinite(entropy)
        & (entropy >= -tolerance)
        & ~jnp.isnan(restriction)
    )
    return ParticleTransportEvaluation(
        energy_rate,
        species_rate,
        boundary_heat,
        boundary_species,
        energy_residual,
        species_residual,
        entropy,
        restriction,
        thermodynamics,
        successful,
        material.bundle_id,
    )


def _validate_boundary(boundary, batch, dtype):
    if not isinstance(boundary, ParticleTransportBoundary):
        raise TypeError("boundary must be a ParticleTransportBoundary.")
    particle_shape = (batch.particle_count,)
    species_shape = particle_shape + (batch.species_count,)
    values = (
        jnp.asarray(boundary.temperature, dtype=dtype),
        jnp.asarray(boundary.heat_transfer_coefficient, dtype=dtype),
        jnp.asarray(boundary.prescribed_heat_rate, dtype=dtype),
    )
    species_values = (
        jnp.asarray(boundary.species_concentration, dtype=dtype),
        jnp.asarray(boundary.mass_transfer_coefficient, dtype=dtype),
        jnp.asarray(boundary.prescribed_species_rate, dtype=dtype),
    )
    if any(value.shape != particle_shape for value in values) or any(
        value.shape != species_shape for value in species_values
    ):
        raise ValueError("Particle transport boundary shapes are invalid.")


def _harmonic_mean(left, right):
    denominator = left + right
    return jnp.where(denominator > 0.0, 2.0 * left * right / denominator, 0.0)


__all__ = [
    "ParticleThermochemicalMaterialBundle",
    "ParticleThermodynamicMaterialPlan",
    "ParticleThermodynamicState",
    "ParticleTransportBoundary",
    "ParticleTransportEvaluation",
    "ParticleTransportMaterialPlan",
    "evaluate_particle_transport",
]
