#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

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


_UNIVERSAL_GAS_CONSTANT = 8.31446261815324


class ParticlePhase(StrEnum):
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    INERT = "inert"


class ParticleSpeciesSchema(StrictModule, NonTrainableState):
    species_names: tuple[str, ...] = eqx.field(static=True)
    phase_ids: Array
    molar_masses: Array
    element_names: tuple[str, ...] = eqx.field(static=True)
    element_composition: Array
    species_count: int = eqx.field(static=True)
    element_count: int = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_names,
        phases,
        molar_masses: ArrayLike,
        element_names,
        element_composition: ArrayLike,
        /,
        *,
        schema_id: str | None = None,
    ):
        names = tuple(str(value) for value in species_names)
        phase_values = tuple(phases)
        masses = np.asarray(molar_masses, dtype=float)
        elements = tuple(str(value) for value in element_names)
        composition = np.asarray(element_composition)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
            or masses.shape != (len(names),)
            or np.any(~np.isfinite(masses))
            or np.any(masses <= 0.0)
        ):
            raise ValueError("Species names and molar masses are invalid.")
        if len(phase_values) != len(names) or any(
            not isinstance(value, ParticlePhase) for value in phase_values
        ):
            raise TypeError("phases must contain one ParticlePhase per species.")
        if (
            not elements
            or any(not value for value in elements)
            or len(set(elements)) != len(elements)
            or composition.shape != (len(elements), len(names))
            or not np.issubdtype(composition.dtype, np.integer)
            or np.any(composition < 0)
        ):
            raise ValueError("Element schema/composition is invalid.")
        phase_ids = np.asarray(
            [list(ParticlePhase).index(value) for value in phase_values], dtype=np.int32
        )
        generated = canonical_fingerprint(
            {
                "kind": "particle-species-schema",
                "species": list(names),
                "phases": [value.value for value in phase_values],
                "molar_masses": array_tree_fingerprint(masses),
                "elements": list(elements),
                "composition": array_tree_fingerprint(composition),
            }
        )
        self.species_names = names
        self.phase_ids = jnp.asarray(phase_ids)
        self.molar_masses = jnp.asarray(masses)
        self.element_names = elements
        self.element_composition = jnp.asarray(composition, dtype=jnp.int32)
        self.species_count = len(names)
        self.element_count = len(elements)
        self.schema_id = generated if schema_id is None else str(schema_id)
        if not self.schema_id:
            raise ValueError("schema_id must be nonempty.")

    def phase_mask(self, phase: ParticlePhase, /) -> Array:
        if not isinstance(phase, ParticlePhase):
            raise TypeError("phase must be a ParticlePhase.")
        return self.phase_ids == list(ParticlePhase).index(phase)

    def element_amount(self, species_amount: ArrayLike, /) -> Array:
        value = jnp.asarray(species_amount)
        if value.shape[-1] != self.species_count:
            raise ValueError("species_amount must end in species axis.")
        return contract("es,...s->...e", self.element_composition, value)


class ParticleThermodynamicState(StrictModule):
    temperature: Array
    heat_capacity: Array
    gas_pressure: Array
    gas_amount: Array
    energy_residual: Array
    temperature_margin: Array
    successful: Array


class ParticleThermodynamicMaterialPlan(StrictModule, NonTrainableState):
    schema: ParticleSpeciesSchema
    heat_capacity_coefficients: Array
    reference_molar_internal_energy: Array
    reference_temperature: float = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    inversion_iterations: int = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ParticleSpeciesSchema,
        molar_heat_capacity: ArrayLike,
        reference_molar_internal_energy: ArrayLike,
        /,
        *,
        reference_temperature: float = 298.15,
        minimum_temperature: float = 1.0,
        maximum_temperature: float = 5000.0,
        inversion_iterations: int = 16,
        material_id: str | None = None,
    ):
        if not isinstance(schema, ParticleSpeciesSchema):
            raise TypeError("schema must be a ParticleSpeciesSchema.")
        coefficients = np.asarray(molar_heat_capacity, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients[:, None]
        reference = np.asarray(reference_molar_internal_energy, dtype=float)
        t_ref = float(reference_temperature)
        t_min = float(minimum_temperature)
        t_max = float(maximum_temperature)
        iterations = int(inversion_iterations)
        if (
            coefficients.ndim != 2
            or coefficients.shape[0] != schema.species_count
            or reference.shape != (schema.species_count,)
            or np.any(~np.isfinite(coefficients))
            or np.any(~np.isfinite(reference))
            or not np.isfinite(t_ref)
            or not np.isfinite(t_min)
            or not np.isfinite(t_max)
            or not 0.0 < t_min < t_ref < t_max
            or iterations <= 0
        ):
            raise ValueError("Thermodynamic material inputs are invalid.")
        sample = np.asarray((t_min, t_ref, t_max))
        powers = sample[:, None] ** np.arange(coefficients.shape[1])[None, :]
        sampled_capacity = powers @ coefficients.T
        if np.any(sampled_capacity <= 0.0):
            raise ValueError("Molar heat capacity must remain positive over bounds.")
        generated = canonical_fingerprint(
            {
                "kind": "particle-thermodynamic-material",
                "schema": schema.schema_id,
                "heat_capacity": array_tree_fingerprint(coefficients),
                "reference_energy": array_tree_fingerprint(reference),
                "reference_temperature": t_ref,
                "bounds": [t_min, t_max],
                "iterations": iterations,
            }
        )
        self.schema = schema
        self.heat_capacity_coefficients = jnp.asarray(coefficients)
        self.reference_molar_internal_energy = jnp.asarray(reference)
        self.reference_temperature = t_ref
        self.minimum_temperature = t_min
        self.maximum_temperature = t_max
        self.inversion_iterations = iterations
        self.material_id = generated if material_id is None else str(material_id)
        if not self.material_id:
            raise ValueError("material_id must be nonempty.")

    def molar_heat_capacity(self, temperature: ArrayLike, /) -> Array:
        value = jnp.asarray(temperature)
        powers = value[..., None] ** jnp.arange(
            self.heat_capacity_coefficients.shape[1], dtype=value.dtype
        )
        return contract("...k,sk->...s", powers, self.heat_capacity_coefficients)

    def molar_internal_energy(self, temperature: ArrayLike, /) -> Array:
        value = jnp.asarray(temperature)
        order = jnp.arange(
            1, self.heat_capacity_coefficients.shape[1] + 1, dtype=value.dtype
        )
        integral = (
            value[..., None] ** order
            - jnp.asarray(self.reference_temperature, dtype=value.dtype) ** order
        ) / order
        return self.reference_molar_internal_energy + contract(
            "...k,sk->...s", integral, self.heat_capacity_coefficients
        )

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
        reference_energy = jnp.sum(amount * self.reference_molar_internal_energy, axis=-1)
        reference_capacity = jnp.sum(
            amount * self.molar_heat_capacity(self.reference_temperature), axis=-1
        )
        initial = self.reference_temperature + (energy - reference_energy) / jnp.maximum(
            reference_capacity, 1.0e-30
        )
        initial = jnp.clip(initial, self.minimum_temperature, self.maximum_temperature)

        def iteration(_, temperature):
            residual = self.energy_from_temperature(temperature, amount) - energy
            capacity = jnp.sum(amount * self.molar_heat_capacity(temperature), axis=-1)
            candidate = temperature - residual / jnp.maximum(capacity, 1.0e-30)
            return jnp.clip(candidate, self.minimum_temperature, self.maximum_temperature)

        temperature = jax.lax.fori_loop(0, self.inversion_iterations, iteration, initial)
        capacity = jnp.sum(amount * self.molar_heat_capacity(temperature), axis=-1)
        recovered = self.energy_from_temperature(temperature, amount)
        residual = recovered - energy
        gas_mask = self.schema.phase_mask(ParticlePhase.GAS).astype(amount.dtype)
        gas_amount = jnp.sum(amount * gas_mask, axis=-1)
        pore_volume = pore * measure
        pressure = (
            gas_amount
            * _UNIVERSAL_GAS_CONSTANT
            * temperature
            / jnp.maximum(pore_volume, 1.0e-30)
        )
        scale = jnp.maximum(jnp.abs(energy), 1.0)
        successful = (
            jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature >= self.minimum_temperature)
            & jnp.all(temperature <= self.maximum_temperature)
            & jnp.all(capacity > 0.0)
            & jnp.all(jnp.abs(residual) <= 128.0 * jnp.finfo(energy.dtype).eps * scale)
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
    schema: ParticleSpeciesSchema
    thermal_conductivity: Array
    species_diffusivity: Array
    tortuosity_exponent: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ParticleSpeciesSchema,
        thermal_conductivity: ArrayLike,
        species_diffusivity: ArrayLike,
        /,
        *,
        tortuosity_exponent: float = 1.0,
        material_id: str | None = None,
    ):
        if not isinstance(schema, ParticleSpeciesSchema):
            raise TypeError("schema must be a ParticleSpeciesSchema.")
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
    "ParticlePhase",
    "ParticleSpeciesSchema",
    "ParticleThermochemicalMaterialBundle",
    "ParticleThermodynamicMaterialPlan",
    "ParticleThermodynamicState",
    "ParticleTransportBoundary",
    "ParticleTransportEvaluation",
    "ParticleTransportMaterialPlan",
    "evaluate_particle_transport",
]
