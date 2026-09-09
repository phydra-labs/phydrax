#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT


class AblatingMaterialState(StrictModule):
    solid_component_densities: Array
    pore_gas_molar_densities: Array
    energy_density: Array
    porosity: Array
    finite: Array


class AblatingMaterialEvaluation(StrictModule):
    temperature: Array
    solid_rate: Array
    pore_gas_rate: Array
    energy_rate: Array
    porosity_rate: Array
    conductivity: Array
    permeability: Array
    mass_defect: Array
    finite: Array
    successful: Array
    material_id: str = eqx.field(static=True)


class AblatingMaterialAdvance(StrictModule):
    candidate: AblatingMaterialState
    accepted: AblatingMaterialState
    evaluation: AblatingMaterialEvaluation
    mass_defect: Array
    energy_defect: Array
    finite: Array
    successful: Array
    material_id: str = eqx.field(static=True)


class PorousAblatingMaterialPlan(StrictModule, NonTrainableState):
    """Finite-rate porous solid decomposition with pore-gas production."""

    component_heat_capacities: Array
    component_conductivities: Array
    component_reference_energies: Array
    pre_exponentials: Array
    temperature_exponents: Array
    activation_energies: Array
    solid_yields: Array
    pore_gas_yields: Array
    pore_gas_molar_masses: Array
    reaction_heats: Array
    virgin_porosity: float = eqx.field(static=True)
    char_porosity: float = eqx.field(static=True)
    reference_permeability: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_heat_capacities: ArrayLike,
        component_conductivities: ArrayLike,
        component_reference_energies: ArrayLike,
        pre_exponentials: ArrayLike,
        temperature_exponents: ArrayLike,
        activation_energies: ArrayLike,
        solid_yields: ArrayLike,
        pore_gas_yields: ArrayLike,
        pore_gas_molar_masses: ArrayLike,
        reaction_heats: ArrayLike,
        /,
        *,
        virgin_porosity: float,
        char_porosity: float,
        reference_permeability: float,
        reference_temperature: float = 300.0,
        minimum_temperature: float = 50.0,
        maximum_temperature: float = 5000.0,
    ):
        cp = np.asarray(component_heat_capacities, dtype=float)
        conductivity = np.asarray(component_conductivities, dtype=float)
        reference_energy = np.asarray(component_reference_energies, dtype=float)
        pre = np.asarray(pre_exponentials, dtype=float)
        exponent = np.asarray(temperature_exponents, dtype=float)
        activation = np.asarray(activation_energies, dtype=float)
        solid_yield = np.asarray(solid_yields, dtype=float)
        gas_yield = np.asarray(pore_gas_yields, dtype=float)
        gas_masses = np.asarray(pore_gas_molar_masses, dtype=float)
        heats = np.asarray(reaction_heats, dtype=float)
        component_count = cp.size
        reaction_count = pre.size
        if (
            cp.ndim != 1
            or component_count == 0
            or conductivity.shape != cp.shape
            or reference_energy.shape != cp.shape
            or exponent.shape != pre.shape
            or activation.shape != pre.shape
            or heats.shape != pre.shape
            or solid_yield.shape != (reaction_count, component_count)
            or gas_yield.ndim != 2
            or gas_yield.shape[0] != reaction_count
            or gas_masses.shape != (gas_yield.shape[1],)
            or np.any(~np.isfinite(cp))
            or np.any(cp <= 0.0)
            or np.any(~np.isfinite(conductivity))
            or np.any(conductivity < 0.0)
            or np.any(~np.isfinite(pre))
            or np.any(pre < 0.0)
            or np.any(~np.isfinite(exponent))
            or np.any(~np.isfinite(activation))
            or np.any(activation < 0.0)
            or np.any(~np.isfinite(solid_yield))
            or np.any(solid_yield < 0.0)
            or np.any(~np.isfinite(gas_yield))
            or np.any(gas_yield < 0.0)
            or np.any(~np.isfinite(gas_masses))
            or np.any(gas_masses <= 0.0)
        ):
            raise ValueError("Porous ablating material arrays are invalid.")
        scalar_values = tuple(
            float(value)
            for value in (
                virgin_porosity,
                char_porosity,
                reference_permeability,
                reference_temperature,
                minimum_temperature,
                maximum_temperature,
            )
        )
        if (
            not 0.0 <= scalar_values[0] < 1.0
            or not 0.0 <= scalar_values[1] < 1.0
            or scalar_values[2] <= 0.0
            or not 0.0 < scalar_values[4] < scalar_values[3] < scalar_values[5]
            or any(not np.isfinite(value) for value in scalar_values)
        ):
            raise ValueError("Porosity, permeability, or temperature bounds are invalid.")
        produced_mass = contract(
            "rg,g->r", jnp.asarray(gas_yield), jnp.asarray(gas_masses), backend="jax"
        ) + jnp.sum(jnp.asarray(solid_yield), axis=-1)
        if np.any(np.abs(np.asarray(produced_mass) - 1.0) > 1.0e-12):
            raise ValueError("Material reaction yields must conserve closed-system mass.")
        self.component_heat_capacities = jnp.asarray(cp)
        self.component_conductivities = jnp.asarray(conductivity)
        self.component_reference_energies = jnp.asarray(reference_energy)
        self.pre_exponentials = jnp.asarray(pre)
        self.temperature_exponents = jnp.asarray(exponent)
        self.activation_energies = jnp.asarray(activation)
        self.solid_yields = jnp.asarray(solid_yield)
        self.pore_gas_yields = jnp.asarray(gas_yield)
        self.pore_gas_molar_masses = jnp.asarray(gas_masses)
        self.reaction_heats = jnp.asarray(heats)
        self.virgin_porosity = scalar_values[0]
        self.char_porosity = scalar_values[1]
        self.reference_permeability = scalar_values[2]
        self.reference_temperature = scalar_values[3]
        self.minimum_temperature = scalar_values[4]
        self.maximum_temperature = scalar_values[5]
        self.material_id = canonical_fingerprint(
            {
                "kind": "porous-ablating-material",
                "cp": array_tree_fingerprint(self.component_heat_capacities),
                "conductivity": array_tree_fingerprint(self.component_conductivities),
                "reference_energy": array_tree_fingerprint(
                    self.component_reference_energies
                ),
                "pre_exponentials": array_tree_fingerprint(self.pre_exponentials),
                "temperature_exponents": array_tree_fingerprint(
                    self.temperature_exponents
                ),
                "activation_energies": array_tree_fingerprint(self.activation_energies),
                "solid_yields": array_tree_fingerprint(self.solid_yields),
                "pore_gas_yields": array_tree_fingerprint(self.pore_gas_yields),
                "pore_gas_molar_masses": array_tree_fingerprint(
                    self.pore_gas_molar_masses
                ),
                "reaction_heats": array_tree_fingerprint(self.reaction_heats),
                "porosity": scalar_values[:2],
                "permeability": scalar_values[2],
                "temperature_bounds": scalar_values[3:],
            }
        )

    @property
    def component_count(self) -> int:
        return self.component_heat_capacities.size

    @property
    def gas_species_count(self) -> int:
        return self.pore_gas_molar_masses.size

    @property
    def reaction_count(self) -> int:
        return self.pre_exponentials.size

    def temperature(self, state: AblatingMaterialState, /) -> Array:
        solid = jnp.asarray(state.solid_component_densities)
        heat_capacity = contract(
            "...s,s->...", solid, self.component_heat_capacities, backend="jax"
        )
        reference_energy = contract(
            "...s,s->...", solid, self.component_reference_energies, backend="jax"
        )
        return self.reference_temperature + (
            state.energy_density - reference_energy
        ) / jnp.maximum(heat_capacity, jnp.finfo(solid.dtype).tiny)

    def evaluate(self, state: AblatingMaterialState, /) -> AblatingMaterialEvaluation:
        solid = jnp.asarray(state.solid_component_densities)
        gas = jnp.asarray(state.pore_gas_molar_densities, dtype=solid.dtype)
        temperature = self.temperature(state)
        if solid.shape[-1] != self.component_count or gas.shape != solid.shape[:-1] + (
            self.gas_species_count,
        ):
            raise ValueError("Ablating material state shapes are invalid.")
        safe_temperature = jnp.maximum(temperature, self.minimum_temperature)
        rates = (
            self.pre_exponentials
            * safe_temperature[..., None] ** self.temperature_exponents
            * jnp.exp(
                -self.activation_energies
                / (UNIVERSAL_GAS_CONSTANT * safe_temperature[..., None])
            )
        )
        limiting = solid[..., : self.reaction_count]
        progress = rates * limiting
        consumed = jnp.zeros_like(solid).at[..., : self.reaction_count].add(-progress)
        produced_solid = contract(
            "...r,rs->...s", progress, self.solid_yields, backend="jax"
        )
        solid_rate = consumed + produced_solid
        gas_rate = contract(
            "...r,rg->...g", progress, self.pore_gas_yields, backend="jax"
        )
        energy_rate = -contract(
            "...r,r->...", progress, self.reaction_heats, backend="jax"
        )
        initial_solid = jnp.sum(solid, axis=-1)
        char_fraction = jnp.where(
            initial_solid > 0.0, solid[..., -1] / initial_solid, 0.0
        )
        target_porosity = (
            self.virgin_porosity
            + (self.char_porosity - self.virgin_porosity) * char_fraction
        )
        porosity_rate = target_porosity - state.porosity
        conductivity = contract(
            "...s,s->...", solid, self.component_conductivities, backend="jax"
        ) / jnp.maximum(initial_solid, jnp.finfo(solid.dtype).tiny)
        permeability = (
            self.reference_permeability
            * jnp.maximum(state.porosity, 1.0e-8) ** 3
            / jnp.maximum((1.0 - state.porosity) ** 2, 1.0e-8)
        )
        mass_defect = jnp.sum(solid_rate, axis=-1) + contract(
            "...g,g->...", gas_rate, self.pore_gas_molar_masses, backend="jax"
        )
        finite = (
            jnp.all(jnp.isfinite(solid_rate), axis=-1)
            & jnp.all(jnp.isfinite(gas_rate), axis=-1)
            & jnp.isfinite(energy_rate)
            & jnp.isfinite(temperature)
        )
        successful = (
            finite
            & jnp.all(solid >= 0.0, axis=-1)
            & jnp.all(gas >= 0.0, axis=-1)
            & (temperature >= self.minimum_temperature)
            & (temperature <= self.maximum_temperature)
            & (state.porosity >= 0.0)
            & (state.porosity < 1.0)
            & (
                jnp.abs(mass_defect)
                <= 512.0
                * jnp.finfo(solid.dtype).eps
                * jnp.maximum(jnp.max(jnp.abs(progress), axis=-1), 1.0)
            )
        )
        return AblatingMaterialEvaluation(
            temperature,
            solid_rate,
            gas_rate,
            energy_rate,
            porosity_rate,
            conductivity,
            permeability,
            mass_defect,
            finite,
            successful,
            self.material_id,
        )

    def advance(
        self, state: AblatingMaterialState, step_size: ArrayLike, /, *, subcycles: int = 8
    ) -> AblatingMaterialAdvance:
        step = jnp.asarray(step_size, dtype=state.energy_density.dtype)
        count = int(subcycles)
        if step.shape != () or count <= 0:
            raise ValueError("Material step and subcycles are invalid.")
        initial = state

        def body(_, current):
            evaluation = self.evaluate(current)
            fraction = step / count
            return AblatingMaterialState(
                current.solid_component_densities + fraction * evaluation.solid_rate,
                current.pore_gas_molar_densities + fraction * evaluation.pore_gas_rate,
                current.energy_density + fraction * evaluation.energy_rate,
                jnp.clip(
                    current.porosity + fraction * evaluation.porosity_rate,
                    0.0,
                    1.0 - 1.0e-8,
                ),
                evaluation.finite,
            )

        candidate = jax.lax.fori_loop(0, count, body, state)
        final = self.evaluate(candidate)
        mass_before = jnp.sum(initial.solid_component_densities, axis=-1) + contract(
            "...g,g->...",
            initial.pore_gas_molar_densities,
            self.pore_gas_molar_masses,
            backend="jax",
        )
        mass_after = jnp.sum(candidate.solid_component_densities, axis=-1) + contract(
            "...g,g->...",
            candidate.pore_gas_molar_densities,
            self.pore_gas_molar_masses,
            backend="jax",
        )
        mass_defect = mass_after - mass_before
        energy_defect = jnp.zeros_like(candidate.energy_density)
        successful = jnp.all(
            final.successful
            & jnp.isfinite(mass_defect)
            & (jnp.abs(mass_defect) <= 1.0e-8 * jnp.maximum(jnp.abs(mass_before), 1.0))
        )
        accepted = AblatingMaterialState(
            jnp.where(
                successful,
                candidate.solid_component_densities,
                initial.solid_component_densities,
            ),
            jnp.where(
                successful,
                candidate.pore_gas_molar_densities,
                initial.pore_gas_molar_densities,
            ),
            jnp.where(successful, candidate.energy_density, initial.energy_density),
            jnp.where(successful, candidate.porosity, initial.porosity),
            jnp.asarray(successful),
        )
        return AblatingMaterialAdvance(
            candidate,
            accepted,
            final,
            mass_defect,
            energy_defect,
            jnp.all(final.finite),
            successful,
            self.material_id,
        )


__all__ = [
    "AblatingMaterialAdvance",
    "AblatingMaterialEvaluation",
    "AblatingMaterialState",
    "PorousAblatingMaterialPlan",
]
