#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
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
from ._particle_thermochemistry import (
    ParticlePhase,
    ParticleSpeciesSchema,
    ParticleThermodynamicMaterialPlan,
    ParticleThermodynamicState,
)


_UNIVERSAL_GAS_CONSTANT = 8.31446261815324


class ParticleReactionLocation(StrEnum):
    BULK = "bulk"
    INTERNAL_SURFACE = "internal_surface"
    OUTER_SURFACE = "outer_surface"


class ParticleReactionEvaluation(StrictModule):
    extent_rate: Array
    species_amount_rate: Array
    internal_energy_rate: Array
    element_residual: Array
    reactant_margin: Array
    explicit_step_restriction: Array
    successful: Array
    network_id: str = eqx.field(static=True)


class ParticleReactionNetworkPlan(StrictModule, NonTrainableState):
    schema: ParticleSpeciesSchema
    reactant_stoichiometry: Array
    product_stoichiometry: Array
    net_stoichiometry: Array
    pre_exponential: Array
    temperature_exponent: Array
    activation_energy: Array
    location_ids: Array
    reaction_count: int = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ParticleSpeciesSchema,
        reactant_stoichiometry: ArrayLike,
        product_stoichiometry: ArrayLike,
        pre_exponential: ArrayLike,
        activation_energy: ArrayLike,
        /,
        *,
        temperature_exponent: ArrayLike | None = None,
        locations=None,
        network_id: str | None = None,
    ):
        if not isinstance(schema, ParticleSpeciesSchema):
            raise TypeError("schema must be a ParticleSpeciesSchema.")
        reactant = np.asarray(reactant_stoichiometry)
        product = np.asarray(product_stoichiometry)
        pre = np.asarray(pre_exponential, dtype=float)
        activation = np.asarray(activation_energy, dtype=float)
        if (
            reactant.ndim != 2
            or reactant.shape != product.shape
            or reactant.shape[1] != schema.species_count
            or reactant.shape[0] == 0
            or not np.issubdtype(reactant.dtype, np.integer)
            or not np.issubdtype(product.dtype, np.integer)
            or np.any(reactant < 0)
            or np.any(product < 0)
        ):
            raise ValueError(
                "Reaction stoichiometry must be nonnegative integer matrices."
            )
        count = reactant.shape[0]
        exponent = (
            np.zeros((count,), dtype=float)
            if temperature_exponent is None
            else np.asarray(temperature_exponent, dtype=float)
        )
        location_values = (
            (ParticleReactionLocation.BULK,) * count
            if locations is None
            else tuple(locations)
        )
        if (
            pre.shape != (count,)
            or activation.shape != (count,)
            or exponent.shape != (count,)
            or np.any(~np.isfinite(pre))
            or np.any(pre < 0.0)
            or np.any(~np.isfinite(activation))
            or np.any(activation < 0.0)
            or np.any(~np.isfinite(exponent))
            or len(location_values) != count
            or any(
                not isinstance(value, ParticleReactionLocation)
                for value in location_values
            )
        ):
            raise ValueError("Reaction kinetic parameters are invalid.")
        net = product.astype(np.int64) - reactant.astype(np.int64)
        element_residual = np.asarray(schema.element_composition) @ net.T
        if np.any(element_residual != 0):
            raise ValueError("Reaction network must conserve every declared element.")
        location_ids = np.asarray(
            [list(ParticleReactionLocation).index(value) for value in location_values],
            dtype=np.int32,
        )
        generated = canonical_fingerprint(
            {
                "kind": "particle-reaction-network",
                "schema": schema.schema_id,
                "reactant": array_tree_fingerprint(reactant),
                "product": array_tree_fingerprint(product),
                "pre_exponential": array_tree_fingerprint(pre),
                "temperature_exponent": array_tree_fingerprint(exponent),
                "activation_energy": array_tree_fingerprint(activation),
                "locations": [value.value for value in location_values],
            }
        )
        self.schema = schema
        self.reactant_stoichiometry = jnp.asarray(reactant, dtype=jnp.int32)
        self.product_stoichiometry = jnp.asarray(product, dtype=jnp.int32)
        self.net_stoichiometry = jnp.asarray(net, dtype=jnp.int32)
        self.pre_exponential = jnp.asarray(pre)
        self.temperature_exponent = jnp.asarray(exponent)
        self.activation_energy = jnp.asarray(activation)
        self.location_ids = jnp.asarray(location_ids)
        self.reaction_count = count
        self.network_id = generated if network_id is None else str(network_id)
        if not self.network_id:
            raise ValueError("network_id must be nonempty.")

    def evaluate(
        self,
        batch: PreparedParticleInternalBatch,
        state: ParticleInternalBatchState,
        thermodynamics: ParticleThermodynamicMaterialPlan,
        /,
    ) -> ParticleReactionEvaluation:
        if batch.prepared_id != state.batch_id:
            raise ValueError("Reaction state does not match prepared batch.")
        if self.schema.schema_id != thermodynamics.schema.schema_id:
            raise ValueError("Reaction and thermodynamic species schemas must match.")
        metrics = batch.mesh.metrics(state.outer_scale)
        thermo = thermodynamics.state(
            state.internal_energy,
            state.species_amount,
            metrics.cell_measures,
            state.porosity,
        )
        concentration = state.species_amount / metrics.cell_measures[:, :, None]
        reactant = self.reactant_stoichiometry.astype(concentration.dtype)
        log_concentration = jnp.log(jnp.maximum(concentration, 1.0e-300))
        log_mass_action = contract("...s,rs->...r", log_concentration, reactant)
        temperature = thermo.temperature[..., None]
        log_rate = (
            jnp.log(jnp.maximum(self.pre_exponential, 1.0e-300))
            + self.temperature_exponent * jnp.log(temperature)
            - self.activation_energy / (_UNIVERSAL_GAS_CONSTANT * temperature)
            + log_mass_action
        )
        local_rate = jnp.exp(jnp.clip(log_rate, -700.0, 700.0))
        bulk_id = list(ParticleReactionLocation).index(ParticleReactionLocation.BULK)
        internal_id = list(ParticleReactionLocation).index(
            ParticleReactionLocation.INTERNAL_SURFACE
        )
        outer_id = list(ParticleReactionLocation).index(
            ParticleReactionLocation.OUTER_SURFACE
        )
        measure = jnp.where(
            self.location_ids[None, None, :] == bulk_id,
            metrics.cell_measures[:, :, None],
            jnp.where(
                self.location_ids[None, None, :] == internal_id,
                state.internal_surface_area[:, :, None],
                jnp.zeros_like(local_rate),
            ),
        )
        measure = measure.at[:, -1, :].add(
            jnp.where(
                self.location_ids[None, :] == outer_id,
                metrics.surface_measure[:, None],
                0.0,
            )
        )
        feasible = jnp.all(
            (state.species_amount[..., None, :] > 0.0)
            | (self.reactant_stoichiometry[None, None, :, :] == 0),
            axis=-1,
        )
        extent_rate = jnp.where(feasible, local_rate * measure, 0.0)
        species_rate = contract("...r,rs->...s", extent_rate, self.net_stoichiometry)
        molar_energy = thermodynamics.molar_internal_energy(thermo.temperature)
        reaction_energy = -jnp.sum(species_rate * molar_energy, axis=-1)
        element_residual = contract(
            "es,...s->...e", self.schema.element_composition, species_rate
        )
        consumption = jnp.maximum(-species_rate, 0.0)
        restriction = jnp.min(
            jnp.where(
                consumption > 0.0,
                state.species_amount / consumption,
                jnp.inf,
            )
        )
        margin = jnp.min(
            jnp.where(
                self.reactant_stoichiometry[None, None, :, :] > 0,
                state.species_amount[..., None, :],
                jnp.inf,
            )
        )
        tolerance = 128.0 * jnp.finfo(state.internal_energy.dtype).eps
        successful = (
            metrics.successful
            & thermo.successful
            & jnp.all(jnp.isfinite(extent_rate) & (extent_rate >= 0.0))
            & jnp.all(jnp.isfinite(species_rate))
            & jnp.all(jnp.isfinite(reaction_energy))
            & jnp.all(jnp.abs(element_residual) <= tolerance)
            & ~jnp.isnan(restriction)
        )
        return ParticleReactionEvaluation(
            extent_rate,
            species_rate,
            reaction_energy,
            element_residual,
            margin,
            restriction,
            successful,
            self.network_id,
        )


class AntoineSaturationPressurePlan(StrictModule, NonTrainableState):
    coefficient_a: float = eqx.field(static=True)
    coefficient_b: float = eqx.field(static=True)
    coefficient_c: float = eqx.field(static=True)
    pressure_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, coefficient_a, coefficient_b, coefficient_c, /, *, pressure_scale=133.322
    ):
        values = tuple(
            float(value)
            for value in (coefficient_a, coefficient_b, coefficient_c, pressure_scale)
        )
        if any(not np.isfinite(value) for value in values) or values[3] <= 0.0:
            raise ValueError("Antoine saturation-pressure parameters are invalid.")
        (
            self.coefficient_a,
            self.coefficient_b,
            self.coefficient_c,
            self.pressure_scale,
        ) = values
        self.plan_id = canonical_fingerprint(
            {"kind": "antoine-saturation-pressure", "values": values}
        )

    def pressure(self, temperature: ArrayLike, /) -> Array:
        value = jnp.asarray(temperature)
        celsius = value - 273.15
        denominator = self.coefficient_c + celsius
        return self.pressure_scale * 10.0 ** (
            self.coefficient_a - self.coefficient_b / denominator
        )


class ParticlePhaseChangeEvaluation(StrictModule):
    species_amount_rate: Array
    internal_energy_rate: Array
    extent_rate: Array
    saturation_margin: Array
    phase_margin: Array
    explicit_step_restriction: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class EvaporationPhaseChangePlan(StrictModule, NonTrainableState):
    schema: ParticleSpeciesSchema
    liquid_species: int = eqx.field(static=True)
    vapor_species: int = eqx.field(static=True)
    mass_transfer_coefficient: float = eqx.field(static=True)
    latent_heat: float = eqx.field(static=True)
    saturation_pressure: AntoineSaturationPressurePlan
    allow_condensation: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema,
        liquid_species,
        vapor_species,
        mass_transfer_coefficient,
        latent_heat,
        saturation_pressure,
        /,
        *,
        allow_condensation=False,
    ):
        if not isinstance(schema, ParticleSpeciesSchema):
            raise TypeError("schema must be a ParticleSpeciesSchema.")
        liquid = int(liquid_species)
        vapor = int(vapor_species)
        coefficient = float(mass_transfer_coefficient)
        latent = float(latent_heat)
        if not isinstance(saturation_pressure, AntoineSaturationPressurePlan):
            raise TypeError(
                "saturation_pressure must be an AntoineSaturationPressurePlan."
            )
        if (
            liquid == vapor
            or liquid < 0
            or vapor < 0
            or liquid >= schema.species_count
            or vapor >= schema.species_count
            or not bool(schema.phase_mask(ParticlePhase.LIQUID)[liquid])
            or not bool(schema.phase_mask(ParticlePhase.GAS)[vapor])
            or not np.isclose(
                float(schema.molar_masses[liquid]), float(schema.molar_masses[vapor])
            )
            or not np.array_equal(
                np.asarray(schema.element_composition)[:, liquid],
                np.asarray(schema.element_composition)[:, vapor],
            )
            or not np.isfinite(coefficient)
            or coefficient < 0.0
            or not np.isfinite(latent)
            or latent <= 0.0
        ):
            raise ValueError("Evaporation species and material parameters are invalid.")
        self.schema = schema
        self.liquid_species = liquid
        self.vapor_species = vapor
        self.mass_transfer_coefficient = coefficient
        self.latent_heat = latent
        self.saturation_pressure = saturation_pressure
        self.allow_condensation = bool(allow_condensation)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "evaporation-phase-change",
                "schema": schema.schema_id,
                "liquid": liquid,
                "vapor": vapor,
                "coefficient": coefficient,
                "latent_heat": latent,
                "saturation": saturation_pressure.plan_id,
                "allow_condensation": bool(allow_condensation),
            }
        )

    def evaluate(
        self,
        batch,
        state,
        thermodynamics: ParticleThermodynamicState,
        metrics,
        /,
    ):
        pore_volume = state.porosity * metrics.cell_measures
        vapor_concentration = state.species_amount[..., self.vapor_species] / jnp.maximum(
            pore_volume, 1.0e-30
        )
        saturation_pressure = self.saturation_pressure.pressure(
            thermodynamics.temperature
        )
        saturation_concentration = saturation_pressure / (
            _UNIVERSAL_GAS_CONSTANT * thermodynamics.temperature
        )
        driving = saturation_concentration - vapor_concentration
        if not self.allow_condensation:
            driving = jnp.maximum(driving, 0.0)
        extent_rate = (
            self.mass_transfer_coefficient * state.internal_surface_area * driving
        )
        liquid = state.species_amount[..., self.liquid_species]
        extent_rate = jnp.where((liquid > 0.0) | (extent_rate < 0.0), extent_rate, 0.0)
        species_rate = jnp.zeros_like(state.species_amount)
        species_rate = species_rate.at[..., self.liquid_species].add(-extent_rate)
        species_rate = species_rate.at[..., self.vapor_species].add(extent_rate)
        energy_rate = -self.latent_heat * extent_rate
        restriction = jnp.min(jnp.where(extent_rate > 0.0, liquid / extent_rate, jnp.inf))
        saturation_margin = jnp.min(jnp.abs(driving))
        phase_margin = jnp.min(liquid)
        element_residual = self.schema.element_amount(species_rate)
        tolerance = 128.0 * jnp.finfo(state.internal_energy.dtype).eps
        successful = (
            thermodynamics.successful
            & metrics.successful
            & jnp.all(jnp.isfinite(extent_rate))
            & jnp.all(jnp.isfinite(energy_rate))
            & jnp.all(jnp.abs(element_residual) <= tolerance)
            & ~jnp.isnan(restriction)
        )
        return ParticlePhaseChangeEvaluation(
            species_rate,
            energy_rate,
            extent_rate,
            saturation_margin,
            phase_margin,
            restriction,
            successful,
            self.plan_id,
        )


class ShrinkingCoreState(StrictModule):
    normalized_core_radius: Array


class ShrinkingCoreEvaluation(StrictModule):
    core_radius_rate: Array
    gas_consumption_rate: Array
    solid_consumption_rate: Array
    conversion: Array
    film_resistance: Array
    ash_resistance: Array
    reaction_resistance: Array
    explicit_step_restriction: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ShrinkingCoreConversionPlan(StrictModule, NonTrainableState):
    gas_stoichiometry: float = eqx.field(static=True)
    solid_stoichiometry: float = eqx.field(static=True)
    solid_molar_density: float = eqx.field(static=True)
    film_transfer_coefficient: float = eqx.field(static=True)
    ash_diffusivity: float = eqx.field(static=True)
    surface_rate_coefficient: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gas_stoichiometry,
        solid_stoichiometry,
        solid_molar_density,
        film_transfer_coefficient,
        ash_diffusivity,
        surface_rate_coefficient,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                gas_stoichiometry,
                solid_stoichiometry,
                solid_molar_density,
                film_transfer_coefficient,
                ash_diffusivity,
                surface_rate_coefficient,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Shrinking-core parameters must be finite and positive.")
        (
            self.gas_stoichiometry,
            self.solid_stoichiometry,
            self.solid_molar_density,
            self.film_transfer_coefficient,
            self.ash_diffusivity,
            self.surface_rate_coefficient,
        ) = values
        self.plan_id = canonical_fingerprint(
            {"kind": "shrinking-core-conversion", "values": values}
        )

    def evaluate(
        self,
        state: ShrinkingCoreState,
        particle_radius: ArrayLike,
        bulk_gas_concentration: ArrayLike,
        /,
    ) -> ShrinkingCoreEvaluation:
        core = jnp.asarray(state.normalized_core_radius)
        radius = jnp.asarray(particle_radius, dtype=core.dtype)
        concentration = jnp.asarray(bulk_gas_concentration, dtype=core.dtype)
        safe_core = jnp.maximum(core, 1.0e-12)
        film = 1.0 / (self.film_transfer_coefficient * radius)
        ash = (1.0 / safe_core - 1.0) / self.ash_diffusivity
        reaction = 1.0 / (
            self.gas_stoichiometry * self.surface_rate_coefficient * safe_core**2
        )
        total = film + ash + reaction
        molar_rate = 4.0 * jnp.pi * radius**2 * concentration / total
        core_rate = -(
            self.solid_stoichiometry
            * molar_rate
            / (
                self.gas_stoichiometry
                * self.solid_molar_density
                * 4.0
                * jnp.pi
                * radius**3
                * safe_core**2
            )
        )
        restriction = jnp.min(jnp.where(core_rate < 0.0, core / -core_rate, jnp.inf))
        conversion = 1.0 - core**3
        successful = (
            jnp.all(jnp.isfinite(core) & (core >= 0.0) & (core <= 1.0))
            & jnp.all(jnp.isfinite(radius) & (radius > 0.0))
            & jnp.all(jnp.isfinite(concentration) & (concentration >= 0.0))
            & jnp.all(jnp.isfinite(core_rate))
            & ~jnp.isnan(restriction)
        )
        return ShrinkingCoreEvaluation(
            core_rate,
            -self.gas_stoichiometry * molar_rate,
            -self.solid_stoichiometry * molar_rate,
            conversion,
            film,
            ash,
            reaction,
            restriction,
            successful,
            self.plan_id,
        )


__all__ = [
    "AntoineSaturationPressurePlan",
    "EvaporationPhaseChangePlan",
    "ParticlePhaseChangeEvaluation",
    "ParticleReactionEvaluation",
    "ParticleReactionLocation",
    "ParticleReactionNetworkPlan",
    "ShrinkingCoreConversionPlan",
    "ShrinkingCoreEvaluation",
    "ShrinkingCoreState",
]
