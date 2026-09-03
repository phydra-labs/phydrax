#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._gas_dynamics import HomogeneousMixtureEulerSystem


class ReactiveClosureTargets(StrictModule):
    """Instantaneous conservative targets for reactive closure learning."""

    species_mass_source: Array
    energy_source: Array
    diagnostic_heat_release_rate: Array
    species_diffusive_flux: Array
    heat_flux: Array
    scalar_dissipation_rate: Array
    net_species_source: Array
    element_source: Array
    charge_source: Array
    net_diffusive_mass_flux: Array
    successful: Array
    system_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)


class ReactiveClosureTargetPlan(StrictModule, NonTrainableState):
    system: HomogeneousMixtureEulerSystem
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: HomogeneousMixtureEulerSystem,
        /,
        *,
        conservation_tolerance: float = 1.0e-10,
    ):
        if not isinstance(system, HomogeneousMixtureEulerSystem):
            raise TypeError("system must be HomogeneousMixtureEulerSystem.")
        tolerance = float(conservation_tolerance)
        if not 0.0 < tolerance < 1.0:
            raise ValueError(
                "conservation_tolerance must lie strictly between zero and one."
            )
        self.system = system
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-closure-target-plan",
                "system": system.system_id,
                "conservation_tolerance": tolerance,
            }
        )

    def build(
        self,
        species_mass_source: ArrayLike,
        diagnostic_heat_release_rate: ArrayLike,
        species_diffusive_flux: ArrayLike,
        heat_flux: ArrayLike,
        scalar_dissipation_rate: ArrayLike,
        /,
    ) -> ReactiveClosureTargets:
        source = jnp.asarray(species_mass_source)
        heat_release = jnp.asarray(diagnostic_heat_release_rate, dtype=source.dtype)
        flux = jnp.asarray(species_diffusive_flux, dtype=source.dtype)
        heat_flux_ = jnp.asarray(heat_flux, dtype=source.dtype)
        dissipation = jnp.asarray(scalar_dissipation_rate, dtype=source.dtype)
        species_count = self.system.species_count
        if source.ndim < 1 or source.shape[-1] != species_count:
            raise ValueError("species_mass_source must end in the full species axis.")
        cell_shape = source.shape[:-1]
        if heat_release.shape != cell_shape or dissipation.shape != cell_shape:
            raise ValueError("Heat release/dissipation must match target cell shape.")
        if (
            flux.ndim != source.ndim + 1
            or flux.shape[:-2] != cell_shape
            or flux.shape[-2] != species_count
        ):
            raise ValueError(
                "species_diffusive_flux must end in species and spatial axes."
            )
        if heat_flux_.shape != cell_shape + (flux.shape[-1],):
            raise ValueError("heat_flux must match target cell and spatial shape.")
        schema = self.system.thermodynamics.schema
        amount_source = source / schema.molar_masses.astype(source.dtype)
        net_source = jnp.sum(source, axis=-1)
        element_source = schema.element_amount(amount_source)
        charge_source = schema.charge_amount(amount_source)
        net_flux = jnp.sum(flux, axis=-2)
        energy_source = jnp.zeros(cell_shape, dtype=source.dtype)
        source_scale = jnp.maximum(jnp.max(jnp.abs(source), axis=-1), 1.0)
        amount_scale = jnp.maximum(jnp.max(jnp.abs(amount_source), axis=-1), 1.0)
        flux_scale = jnp.maximum(jnp.max(jnp.abs(flux), axis=(-2, -1)), 1.0)
        successful = (
            jnp.all(jnp.isfinite(source), axis=-1)
            & jnp.isfinite(heat_release)
            & jnp.all(jnp.isfinite(flux), axis=(-2, -1))
            & jnp.all(jnp.isfinite(heat_flux_), axis=-1)
            & jnp.isfinite(dissipation)
            & (dissipation >= 0.0)
            & (jnp.abs(net_source) <= self.conservation_tolerance * source_scale)
            & jnp.all(
                jnp.abs(element_source)
                <= self.conservation_tolerance * amount_scale[..., None],
                axis=-1,
            )
            & (jnp.abs(charge_source) <= self.conservation_tolerance * amount_scale)
            & jnp.all(
                jnp.abs(net_flux) <= self.conservation_tolerance * flux_scale[..., None],
                axis=-1,
            )
        )
        target_id = canonical_fingerprint(
            {
                "kind": "reactive-closure-target",
                "plan": self.plan_id,
                "species_count": species_count,
                "dimension": flux.shape[-1],
                "chemical_energy_source": "zero-canonical-total-energy",
            }
        )
        return ReactiveClosureTargets(
            source,
            energy_source,
            heat_release,
            flux,
            heat_flux_,
            dissipation,
            net_source,
            element_source,
            charge_source,
            net_flux,
            successful,
            self.system.system_id,
            target_id,
        )


class ReactiveFlowStatistics(StrictModule):
    mean_density: Array
    favre_velocity: Array
    favre_reynolds_stress: Array
    favre_temperature: Array
    favre_temperature_variance: Array
    favre_species_mass_fractions: Array
    favre_species_covariance: Array
    favre_temperature_species_covariance: Array
    mean_element_amount_per_mass: Array
    favre_specific_internal_energy: Array
    favre_specific_enthalpy: Array
    mean_diagnostic_heat_release_rate: Array
    total_weight: Array
    total_mass_weight: Array
    successful: Array
    statistics_id: str = eqx.field(static=True)


class ReactiveFlowStatisticsPlan(StrictModule, NonTrainableState):
    """Volume and Favre statistics retaining species/element/energy structure."""

    system: HomogeneousMixtureEulerSystem
    plan_id: str = eqx.field(static=True)

    def __init__(self, system: HomogeneousMixtureEulerSystem, /):
        if not isinstance(system, HomogeneousMixtureEulerSystem):
            raise TypeError("system must be HomogeneousMixtureEulerSystem.")
        self.system = system
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-flow-statistics",
                "system": system.system_id,
                "weighting": "volume-and-favre",
            }
        )

    def evaluate(
        self,
        conserved: ArrayLike,
        cell_weights: ArrayLike,
        /,
        *,
        closure_targets: ReactiveClosureTargets | None = None,
    ) -> ReactiveFlowStatistics:
        state = jnp.asarray(conserved)
        weights = jnp.asarray(cell_weights, dtype=state.dtype)
        if state.ndim < 2 or state.shape[-1] != self.system.component_count:
            raise ValueError("conserved must contain at least one cell axis.")
        cell_shape = state.shape[:-1]
        if weights.shape != cell_shape:
            raise ValueError("cell_weights must match the conserved cell shape.")
        species_count = self.system.species_count
        species_density = state[..., :species_count]
        density = jnp.sum(species_density, axis=-1)
        velocity = state[..., species_count:-1] / density[..., None]
        mass_fractions = species_density / density[..., None]
        recovered = self.system.recover_thermodynamics(state)
        thermo = recovered.state
        axes = tuple(range(len(cell_shape)))
        total_weight = jnp.sum(weights)
        mass_weights = weights * density
        total_mass = jnp.sum(mass_weights)
        mean_density = jnp.sum(weights * density) / total_weight
        favre_velocity = (
            jnp.sum(mass_weights[..., None] * velocity, axis=axes) / total_mass
        )
        velocity_fluctuation = velocity - favre_velocity
        reynolds = (
            contract(
                "...,...i,...j->ij",
                mass_weights,
                velocity_fluctuation,
                velocity_fluctuation,
                backend="jax",
            )
            / total_mass
        )
        favre_temperature = jnp.sum(mass_weights * thermo.temperature) / total_mass
        temperature_fluctuation = thermo.temperature - favre_temperature
        temperature_variance = (
            jnp.sum(mass_weights * temperature_fluctuation**2) / total_mass
        )
        favre_species = (
            jnp.sum(mass_weights[..., None] * mass_fractions, axis=axes) / total_mass
        )
        species_fluctuation = mass_fractions - favre_species
        species_covariance = (
            contract(
                "...,...s,...t->st",
                mass_weights,
                species_fluctuation,
                species_fluctuation,
                backend="jax",
            )
            / total_mass
        )
        temperature_species = (
            contract(
                "...,...,...s->s",
                mass_weights,
                temperature_fluctuation,
                species_fluctuation,
                backend="jax",
            )
            / total_mass
        )
        schema = self.system.thermodynamics.schema
        element_amount_per_mass = contract(
            "es,...s,s->...e",
            schema.element_composition,
            mass_fractions,
            1.0 / schema.molar_masses.astype(state.dtype),
            backend="jax",
        )
        mean_element = (
            jnp.sum(mass_weights[..., None] * element_amount_per_mass, axis=axes)
            / total_mass
        )
        internal_energy = (
            jnp.sum(mass_weights * (thermo.molar_internal_energy / thermo.molar_mass))
            / total_mass
        )
        enthalpy = (
            jnp.sum(mass_weights * (thermo.molar_enthalpy / thermo.molar_mass))
            / total_mass
        )
        if closure_targets is None:
            heat_release = jnp.asarray(0.0, dtype=state.dtype)
            closure_success = jnp.asarray(True)
        else:
            if not isinstance(closure_targets, ReactiveClosureTargets):
                raise TypeError("closure_targets must be ReactiveClosureTargets or None.")
            if closure_targets.system_id != self.system.system_id:
                raise ValueError(
                    "Closure targets and statistics must bind the same system."
                )
            if closure_targets.diagnostic_heat_release_rate.shape != cell_shape:
                raise ValueError("Closure targets do not match statistics cell shape.")
            heat_release = (
                jnp.sum(weights * closure_targets.diagnostic_heat_release_rate)
                / total_weight
            )
            closure_success = jnp.all(closure_targets.successful)
        successful = (
            jnp.isfinite(total_weight)
            & (total_weight > 0.0)
            & jnp.all(jnp.isfinite(weights) & (weights > 0.0))
            & jnp.isfinite(total_mass)
            & (total_mass > 0.0)
            & jnp.all(recovered.successful)
            & closure_success
            & jnp.isfinite(mean_density)
            & jnp.all(jnp.isfinite(reynolds))
            & jnp.all(jnp.isfinite(species_covariance))
        )
        return ReactiveFlowStatistics(
            mean_density,
            favre_velocity,
            reynolds,
            favre_temperature,
            temperature_variance,
            favre_species,
            species_covariance,
            temperature_species,
            mean_element,
            internal_energy,
            enthalpy,
            heat_release,
            total_weight,
            total_mass,
            successful,
            self.plan_id,
        )


__all__ = [
    "ReactiveClosureTargetPlan",
    "ReactiveClosureTargets",
    "ReactiveFlowStatistics",
    "ReactiveFlowStatisticsPlan",
]
