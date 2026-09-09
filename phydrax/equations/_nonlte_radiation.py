#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT


AVOGADRO_CONSTANT = 6.02214076e23
PLANCK_CONSTANT = 6.62607015e-34


class NonLTEPopulationEvaluation(StrictModule):
    level_molar_densities: Array
    level_number_densities: Array
    partition_functions: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class NonLTELevelPopulationPlan(StrictModule, NonTrainableState):
    """Species-resolved discrete electronic populations with departure factors."""

    level_species: Array
    level_energies: Array
    level_degeneracies: Array
    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_count: int,
        level_species: ArrayLike,
        level_energies: ArrayLike,
        level_degeneracies: ArrayLike,
        /,
    ):
        count = int(species_count)
        owners = np.asarray(level_species, dtype=np.int32)
        energies = np.asarray(level_energies, dtype=float)
        degeneracies = np.asarray(level_degeneracies, dtype=float)
        if (
            count <= 0
            or owners.ndim != 1
            or energies.shape != owners.shape
            or degeneracies.shape != owners.shape
            or owners.size == 0
            or np.any((owners < 0) | (owners >= count))
            or np.any(~np.isfinite(energies))
            or np.any(energies < 0.0)
            or np.any(~np.isfinite(degeneracies))
            or np.any(degeneracies <= 0.0)
            or any(np.count_nonzero(owners == index) == 0 for index in range(count))
        ):
            raise ValueError(
                "Non-LTE level ownership, energies, or degeneracies are invalid."
            )
        self.species_count = count
        self.level_species = jnp.asarray(owners)
        self.level_energies = jnp.asarray(energies)
        self.level_degeneracies = jnp.asarray(degeneracies)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "non-lte-level-populations",
                "species_count": count,
                "level_species": array_tree_fingerprint(self.level_species),
                "level_energies": array_tree_fingerprint(self.level_energies),
                "level_degeneracies": array_tree_fingerprint(self.level_degeneracies),
            }
        )

    @property
    def level_count(self) -> int:
        return self.level_species.size

    def evaluate(
        self,
        species_molar_densities: ArrayLike,
        electron_temperature: ArrayLike,
        /,
        *,
        departure_coefficients: ArrayLike | None = None,
    ) -> NonLTEPopulationEvaluation:
        species = jnp.asarray(species_molar_densities)
        temperature = jnp.asarray(electron_temperature, dtype=species.dtype)
        cell_shape = species.shape[:-1]
        if species.shape[-1] != self.species_count or temperature.shape != cell_shape:
            raise ValueError("Non-LTE species and temperature shapes are incompatible.")
        departure = (
            jnp.ones(cell_shape + (self.level_count,), dtype=species.dtype)
            if departure_coefficients is None
            else jnp.asarray(departure_coefficients, dtype=species.dtype)
        )
        if departure.shape != cell_shape + (self.level_count,):
            raise ValueError("Departure coefficients must match all electronic levels.")
        boltzmann = self.level_degeneracies.astype(species.dtype) * jnp.exp(
            -self.level_energies.astype(species.dtype)
            / (UNIVERSAL_GAS_CONSTANT * temperature[..., None])
        )
        weighted = departure * boltzmann
        partitions = []
        populations = []
        for species_index in range(self.species_count):
            mask = self.level_species == species_index
            partition = jnp.sum(jnp.where(mask, weighted, 0.0), axis=-1)
            partitions.append(partition)
            populations.append(
                jnp.where(
                    mask,
                    species[..., species_index, None]
                    * weighted
                    / jnp.maximum(partition[..., None], jnp.finfo(species.dtype).tiny),
                    0.0,
                )
            )
        level_molar = jnp.sum(jnp.stack(tuple(populations), axis=-2), axis=-2)
        partition = jnp.stack(tuple(partitions), axis=-1)
        level_number = AVOGADRO_CONSTANT * level_molar
        recovered_species = jnp.stack(
            tuple(
                jnp.sum(
                    jnp.where(self.level_species == index, level_molar, 0.0),
                    axis=-1,
                )
                for index in range(self.species_count)
            ),
            axis=-1,
        )
        scale = jnp.maximum(jnp.max(jnp.abs(species), axis=-1), 1.0)
        finite = (
            jnp.all(jnp.isfinite(level_molar), axis=-1)
            & jnp.all(jnp.isfinite(partition), axis=-1)
            & jnp.all(jnp.isfinite(departure), axis=-1)
        )
        successful = (
            finite
            & (temperature > 0.0)
            & jnp.all(species >= 0.0, axis=-1)
            & jnp.all(departure >= 0.0, axis=-1)
            & jnp.all(partition > 0.0, axis=-1)
            & jnp.all(
                jnp.abs(recovered_species - species)
                <= 256.0 * jnp.finfo(species.dtype).eps * scale[..., None],
                axis=-1,
            )
        )
        return NonLTEPopulationEvaluation(
            level_molar,
            level_number,
            partition,
            finite,
            successful,
            self.plan_id,
        )


class NonLTERadiationCoefficientEvaluation(StrictModule):
    absorption: Array
    emission_power_density: Array
    spontaneous_power_density: Array
    level_populations: NonLTEPopulationEvaluation
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class NonLTERadiationCoefficientPlan(StrictModule, NonTrainableState):
    """Multigroup bound-bound coefficients from explicit level populations."""

    populations: NonLTELevelPopulationPlan
    lower_levels: Array
    upper_levels: Array
    group_indices: Array
    photon_frequencies: Array
    spontaneous_rates: Array
    absorption_cross_sections: Array
    group_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        populations: NonLTELevelPopulationPlan,
        group_count: int,
        lower_levels: ArrayLike,
        upper_levels: ArrayLike,
        group_indices: ArrayLike,
        photon_frequencies: ArrayLike,
        spontaneous_rates: ArrayLike,
        absorption_cross_sections: ArrayLike,
        /,
    ):
        groups = int(group_count)
        lower = np.asarray(lower_levels, dtype=np.int32)
        upper = np.asarray(upper_levels, dtype=np.int32)
        group = np.asarray(group_indices, dtype=np.int32)
        frequency = np.asarray(photon_frequencies, dtype=float)
        rates = np.asarray(spontaneous_rates, dtype=float)
        cross_sections = np.asarray(absorption_cross_sections, dtype=float)
        shape = lower.shape
        if (
            not isinstance(populations, NonLTELevelPopulationPlan)
            or groups <= 0
            or lower.ndim != 1
            or lower.size == 0
            or upper.shape != shape
            or group.shape != shape
            or frequency.shape != shape
            or rates.shape != shape
            or cross_sections.shape != shape
            or np.any((lower < 0) | (lower >= populations.level_count))
            or np.any((upper < 0) | (upper >= populations.level_count))
            or np.any((group < 0) | (group >= groups))
            or np.any(frequency <= 0.0)
            or np.any(rates < 0.0)
            or np.any(cross_sections < 0.0)
            or np.any(~np.isfinite(frequency))
            or np.any(~np.isfinite(rates))
            or np.any(~np.isfinite(cross_sections))
        ):
            raise ValueError("Non-LTE radiative transition data are invalid.")
        self.populations = populations
        self.group_count = groups
        self.lower_levels = jnp.asarray(lower)
        self.upper_levels = jnp.asarray(upper)
        self.group_indices = jnp.asarray(group)
        self.photon_frequencies = jnp.asarray(frequency)
        self.spontaneous_rates = jnp.asarray(rates)
        self.absorption_cross_sections = jnp.asarray(cross_sections)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "non-lte-radiation-coefficients",
                "populations": populations.plan_id,
                "group_count": groups,
                "lower": array_tree_fingerprint(self.lower_levels),
                "upper": array_tree_fingerprint(self.upper_levels),
                "groups": array_tree_fingerprint(self.group_indices),
                "frequencies": array_tree_fingerprint(self.photon_frequencies),
                "rates": array_tree_fingerprint(self.spontaneous_rates),
                "cross_sections": array_tree_fingerprint(self.absorption_cross_sections),
            }
        )

    def evaluate(
        self,
        species_molar_densities: ArrayLike,
        electron_temperature: ArrayLike,
        /,
        *,
        departure_coefficients: ArrayLike | None = None,
    ) -> NonLTERadiationCoefficientEvaluation:
        population = self.populations.evaluate(
            species_molar_densities,
            electron_temperature,
            departure_coefficients=departure_coefficients,
        )
        lower = population.level_number_densities[..., self.lower_levels]
        upper = population.level_number_densities[..., self.upper_levels]
        absorption_transition = lower * self.absorption_cross_sections.astype(lower.dtype)
        spontaneous_transition = (
            upper
            * self.spontaneous_rates.astype(upper.dtype)
            * PLANCK_CONSTANT
            * self.photon_frequencies.astype(upper.dtype)
        )
        absorption_groups = []
        emission_groups = []
        for group in range(self.group_count):
            mask = self.group_indices == group
            absorption_groups.append(
                jnp.sum(jnp.where(mask, absorption_transition, 0.0), axis=-1)
            )
            emission_groups.append(
                jnp.sum(jnp.where(mask, spontaneous_transition, 0.0), axis=-1)
            )
        absorption = jnp.stack(tuple(absorption_groups), axis=-1)
        emission = jnp.stack(tuple(emission_groups), axis=-1)
        finite = (
            population.finite
            & jnp.all(jnp.isfinite(absorption), axis=-1)
            & jnp.all(jnp.isfinite(emission), axis=-1)
        )
        successful = (
            finite
            & population.successful
            & jnp.all(absorption >= 0.0, axis=-1)
            & jnp.all(emission >= 0.0, axis=-1)
        )
        return NonLTERadiationCoefficientEvaluation(
            absorption,
            emission,
            emission,
            population,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "AVOGADRO_CONSTANT",
    "NonLTELevelPopulationPlan",
    "NonLTEPopulationEvaluation",
    "NonLTERadiationCoefficientEvaluation",
    "NonLTERadiationCoefficientPlan",
    "PLANCK_CONSTANT",
]
