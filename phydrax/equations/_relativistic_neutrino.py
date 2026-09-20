#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import (
    ADMGridGeometry,
    combine_stress_energy_projections,
    StressEnergyProjection,
)
from ..metrix._spacetime_conventions import RelativityConvention
from ._relativistic_multigroup_radiation import (
    GRMultigroupM1ClosureEvaluation,
    GRMultigroupM1RadiationSystem,
    GRMultigroupRadiationInteractionPlan,
    GRMultigroupRadiationMatterExchange,
)


NeutrinoSpecies = Literal["electron_neutrino", "electron_antineutrino", "heavy_lepton"]
_VALID_SPECIES = frozenset(("electron_neutrino", "electron_antineutrino", "heavy_lepton"))
_LEPTON_SIGN = {
    "electron_neutrino": 1.0,
    "electron_antineutrino": -1.0,
    "heavy_lepton": 0.0,
}


class GRNeutrinoM1ClosureEvaluation(StrictModule):
    species: tuple[GRMultigroupM1ClosureEvaluation, ...]
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)


class GRNeutrinoM1System(StrictModule, NonTrainableState):
    """Energy-group M1 moments for explicitly named neutrino species."""

    scale: RelativityScaleContract
    convention: RelativityConvention
    frequency_edges: Array
    species: tuple[str, ...] = eqx.field(static=True)
    species_systems: tuple[GRMultigroupM1RadiationSystem, ...]
    species_count: int = eqx.field(static=True)
    group_count: int = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        frequency_edges: ArrayLike,
        /,
        *,
        species: tuple[NeutrinoSpecies, ...] = (
            "electron_neutrino",
            "electron_antineutrino",
            "heavy_lepton",
        ),
        reduced_light_speed: float | None = None,
        energy_floor: float = 1.0e-12,
        metric_tolerance: float = 1.0e-9,
    ) -> None:
        names = tuple(species)
        if (
            not names
            or len(set(names)) != len(names)
            or any(value not in _VALID_SPECIES for value in names)
        ):
            raise ValueError("Neutrino species must be unique supported names.")
        systems = tuple(
            GRMultigroupM1RadiationSystem(
                scale,
                convention,
                frequency_edges,
                reduced_light_speed=reduced_light_speed,
                energy_floor=energy_floor,
                metric_tolerance=metric_tolerance,
            )
            for _ in names
        )
        self.scale = scale
        self.convention = convention
        self.frequency_edges = systems[0].frequency_edges
        self.species = names
        self.species_systems = systems
        self.species_count = len(names)
        self.group_count = systems[0].group_count
        self.component_names = tuple(
            f"{species_name}_{component}"
            for species_name in names
            for component in systems[0].component_names
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "gr-neutrino-m1",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "frequency_edges": array_tree_fingerprint(
                    np.asarray(frequency_edges, dtype=np.float64)
                ),
                "species": names,
                "systems": [value.system_id for value in systems],
            }
        )

    def species_group_moments(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        count = 4 * self.species_count * self.group_count
        if value.shape[-1:] != (count,):
            raise ValueError("GR neutrino state has the wrong component count.")
        return value.reshape(value.shape[:-1] + (self.species_count, self.group_count, 4))

    def flatten_moments(self, moments: ArrayLike, /) -> Array:
        value = jnp.asarray(moments)
        expected = (self.species_count, self.group_count, 4)
        if value.shape[-3:] != expected:
            raise ValueError(f"GR neutrino moments must end in {expected}.")
        return value.reshape(
            value.shape[:-3] + (4 * self.species_count * self.group_count,)
        )

    def closure(
        self, state: ArrayLike, geometry: ADMGridGeometry, /
    ) -> GRNeutrinoM1ClosureEvaluation:
        moments = self.species_group_moments(state)
        evaluations = tuple(
            system.closure(system.flatten_groups(moments[..., index, :, :]), geometry)
            for index, system in enumerate(self.species_systems)
        )
        finite = jnp.all(jnp.stack(tuple(value.finite for value in evaluations)), axis=0)
        physical = jnp.all(
            jnp.stack(tuple(value.physically_valid for value in evaluations)), axis=0
        )
        qualified = jnp.all(
            jnp.stack(tuple(value.qualified for value in evaluations)), axis=0
        )
        derivative = jnp.all(
            jnp.stack(tuple(value.derivative_valid for value in evaluations)), axis=0
        )
        return GRNeutrinoM1ClosureEvaluation(
            evaluations,
            finite,
            physical,
            qualified,
            derivative,
            self.system_id,
        )

    def stress_energy_projection(
        self, state: ArrayLike, geometry: ADMGridGeometry, /
    ) -> StressEnergyProjection:
        moments = self.species_group_moments(state)
        projections = tuple(
            system.stress_energy_projection(
                system.flatten_groups(moments[..., index, :, :]), geometry
            )
            for index, system in enumerate(self.species_systems)
        )
        return combine_stress_energy_projections(projections)

    def radiation_lepton_number(self, state: ArrayLike, /) -> Array:
        moments = self.species_group_moments(state)
        mean_energies = jnp.sqrt(self.frequency_edges[:-1] * self.frequency_edges[1:])
        signs = jnp.asarray(tuple(_LEPTON_SIGN[value] for value in self.species))
        number = moments[..., :, :, 0] / mean_energies
        return jnp.sum(number * signs[:, None], axis=(-2, -1))


class GRNeutrinoMatterExchange(StrictModule):
    species_exchanges: tuple[GRMultigroupRadiationMatterExchange, ...]
    radiation_energy_source: Array
    radiation_flux_source: Array
    matter_energy_source: Array
    matter_momentum_source: Array
    radiation_lepton_number_source: Array
    matter_lepton_number_source: Array
    electron_fraction_source: Array
    energy_balance_residual: Array
    momentum_balance_residual: Array
    lepton_balance_residual: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class GRNeutrinoInteractionPlan(StrictModule, NonTrainableState):
    neutrinos: GRNeutrinoM1System
    species_interactions: tuple[GRMultigroupRadiationInteractionPlan, ...]
    mean_group_energies: Array
    lepton_signs: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        neutrinos: GRNeutrinoM1System,
        species_interactions: tuple[GRMultigroupRadiationInteractionPlan, ...],
        /,
    ) -> None:
        if not isinstance(neutrinos, GRNeutrinoM1System):
            raise TypeError("neutrinos must be GRNeutrinoM1System.")
        interactions = tuple(species_interactions)
        if len(interactions) != neutrinos.species_count or any(
            not isinstance(value, GRMultigroupRadiationInteractionPlan)
            for value in interactions
        ):
            raise TypeError(
                "One multigroup interaction is required per neutrino species."
            )
        for system, interaction in zip(
            neutrinos.species_systems, interactions, strict=True
        ):
            if interaction.radiation.system_id != system.system_id:
                raise ValueError("Neutrino interaction and species systems differ.")
        self.neutrinos = neutrinos
        self.species_interactions = interactions
        self.mean_group_energies = jnp.sqrt(
            neutrinos.frequency_edges[:-1] * neutrinos.frequency_edges[1:]
        )
        self.lepton_signs = jnp.asarray(
            tuple(_LEPTON_SIGN[value] for value in neutrinos.species)
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gr-neutrino-interaction",
                "neutrinos": neutrinos.system_id,
                "interactions": [value.plan_id for value in interactions],
            }
        )

    def matter_exchange(
        self,
        state: ArrayLike,
        rest_mass_density: ArrayLike,
        fluid_velocity: ArrayLike,
        matter_temperature: ArrayLike,
        baryon_number_density: ArrayLike,
        electron_fraction: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        magnetic_squared: ArrayLike = 0.0,
    ) -> GRNeutrinoMatterExchange:
        moments = self.neutrinos.species_group_moments(state)
        exchanges = tuple(
            interaction.matter_exchange(
                system.flatten_groups(moments[..., index, :, :]),
                rest_mass_density,
                fluid_velocity,
                matter_temperature,
                geometry,
                magnetic_squared=magnetic_squared,
                composition=electron_fraction,
            )
            for index, (system, interaction) in enumerate(
                zip(
                    self.neutrinos.species_systems,
                    self.species_interactions,
                    strict=True,
                )
            )
        )
        energy = jnp.stack(
            tuple(value.radiation_energy_source for value in exchanges), axis=-2
        )
        flux = jnp.stack(
            tuple(value.radiation_flux_source for value in exchanges), axis=-3
        )
        radiation_momentum = jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        group.radiation_momentum_source for group in value.group_exchanges
                    ),
                    axis=-2,
                )
                for value in exchanges
            ),
            axis=-3,
        )
        matter_energy = -jnp.sum(energy, axis=(-2, -1))
        matter_momentum = -jnp.sum(radiation_momentum, axis=(-3, -2))
        radiation_lepton = jnp.sum(
            energy / self.mean_group_energies * self.lepton_signs[:, None],
            axis=(-2, -1),
        )
        matter_lepton = -radiation_lepton
        baryon_density = jnp.asarray(baryon_number_density)
        safe_baryon_density = jnp.where(baryon_density > 0.0, baryon_density, 1.0)
        electron_fraction_source = matter_lepton / safe_baryon_density
        electron_fraction_source = jnp.where(
            baryon_density > 0.0, electron_fraction_source, 0.0
        )
        energy_residual = jnp.sum(energy, axis=(-2, -1)) + matter_energy
        momentum_residual = jnp.sum(radiation_momentum, axis=(-3, -2)) + matter_momentum
        lepton_residual = radiation_lepton + matter_lepton
        finite = jnp.all(jnp.stack(tuple(value.finite for value in exchanges)), axis=0)
        physical = (
            jnp.all(
                jnp.stack(tuple(value.physically_valid for value in exchanges)), axis=0
            )
            & jnp.isfinite(baryon_density)
            & (baryon_density >= 0.0)
            & jnp.isfinite(jnp.asarray(electron_fraction))
            & (jnp.asarray(electron_fraction) >= 0.0)
            & (jnp.asarray(electron_fraction) <= 1.0)
        )
        qualified = (
            jnp.all(jnp.stack(tuple(value.qualified for value in exchanges)), axis=0)
            & physical
        )
        fraction = jnp.asarray(electron_fraction)
        derivative = (
            jnp.all(
                jnp.stack(tuple(value.derivative_valid for value in exchanges), axis=0)
            )
            & qualified
            & (baryon_density > 0.0)
            & (fraction > 0.0)
            & (fraction < 1.0)
        )
        return GRNeutrinoMatterExchange(
            exchanges,
            energy,
            flux,
            matter_energy,
            matter_momentum,
            radiation_lepton,
            matter_lepton,
            electron_fraction_source,
            energy_residual,
            momentum_residual,
            lepton_residual,
            finite,
            physical,
            qualified,
            derivative,
            self.plan_id,
        )


__all__ = [
    "GRNeutrinoInteractionPlan",
    "GRNeutrinoM1ClosureEvaluation",
    "GRNeutrinoM1System",
    "GRNeutrinoMatterExchange",
    "NeutrinoSpecies",
]
