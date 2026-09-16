#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
from ._relativistic_radiation import (
    GRGreyM1ClosureEvaluation,
    GRGreyM1RadiationSystem,
)
from ._relativistic_radiation_interaction import (
    GRGreyRadiationInteractionPlan,
    GRRadiationMatterExchange,
)


class GRMultigroupM1ClosureEvaluation(StrictModule):
    groups: tuple[GRGreyM1ClosureEvaluation, ...]
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    system_id: str = eqx.field(static=True)


class GRMultigroupM1RadiationSystem(StrictModule, NonTrainableState):
    """Frequency-bounded collection of metric-aware GR M1 moment groups."""

    scale: RelativityScaleContract
    convention: RelativityConvention
    frequency_edges: Array
    groups: tuple[GRGreyM1RadiationSystem, ...]
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
        reduced_light_speed: float | None = None,
        energy_floor: float = 1.0e-12,
        metric_tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be RelativityConvention.")
        edges = np.asarray(frequency_edges, dtype=float)
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(edges <= 0.0)
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("GR multigroup frequency edges are invalid.")
        count = int(edges.size - 1)
        groups = tuple(
            GRGreyM1RadiationSystem(
                scale,
                convention,
                reduced_light_speed=reduced_light_speed,
                energy_floor=energy_floor,
                metric_tolerance=metric_tolerance,
            )
            for _ in range(count)
        )
        self.scale = scale
        self.convention = convention
        self.frequency_edges = jnp.asarray(edges)
        self.groups = groups
        self.group_count = count
        self.component_names = tuple(
            name
            for group in range(count)
            for name in (
                f"radiation_energy_{group}",
                f"radiation_flux_x_{group}",
                f"radiation_flux_y_{group}",
                f"radiation_flux_z_{group}",
            )
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "gr-multigroup-m1-radiation",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "frequency_edges": array_tree_fingerprint(edges),
                "groups": [value.system_id for value in groups],
            }
        )

    def group_moments(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[-1:] != (4 * self.group_count,):
            raise ValueError("GR multigroup state has the wrong component count.")
        return value.reshape(value.shape[:-1] + (self.group_count, 4))

    def flatten_groups(self, groups: ArrayLike, /) -> Array:
        value = jnp.asarray(groups)
        if value.shape[-2:] != (self.group_count, 4):
            raise ValueError("GR multigroup moments must end in (group_count, 4).")
        return value.reshape(value.shape[:-2] + (4 * self.group_count,))

    def closure(
        self, state: ArrayLike, geometry: ADMGridGeometry, /
    ) -> GRMultigroupM1ClosureEvaluation:
        moments = self.group_moments(state)
        groups = tuple(
            system.closure(moments[..., index, 0], moments[..., index, 1:], geometry)
            for index, system in enumerate(self.groups)
        )
        finite = jnp.all(jnp.stack(tuple(value.finite for value in groups)), axis=0)
        physical = jnp.all(
            jnp.stack(tuple(value.physically_valid for value in groups)), axis=0
        )
        qualified = jnp.all(jnp.stack(tuple(value.qualified for value in groups)), axis=0)
        derivative = jnp.all(
            jnp.stack(tuple(value.derivative_valid for value in groups)), axis=0
        )
        return GRMultigroupM1ClosureEvaluation(
            groups, finite, physical, qualified, derivative, self.system_id
        )

    def coordinate_flux(
        self,
        state: ArrayLike,
        axis: int,
        geometry: ADMGridGeometry,
        /,
    ) -> Array:
        moments = self.group_moments(state)
        fluxes = jnp.stack(
            tuple(
                system.coordinate_flux(
                    moments[..., index, 0],
                    moments[..., index, 1:],
                    axis,
                    geometry,
                )
                for index, system in enumerate(self.groups)
            ),
            axis=-2,
        )
        return self.flatten_groups(fluxes)

    def coordinate_characteristic_bounds(
        self,
        unit_covector: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> tuple[Array, Array]:
        bounds = tuple(
            system.coordinate_characteristic_bounds(unit_covector, geometry)
            for system in self.groups
        )
        lower = jnp.min(jnp.stack(tuple(value[0] for value in bounds)), axis=0)
        upper = jnp.max(jnp.stack(tuple(value[1] for value in bounds)), axis=0)
        return lower, upper

    def stress_energy_projection(
        self,
        state: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> StressEnergyProjection:
        moments = self.group_moments(state)
        projections = tuple(
            system.stress_energy_projection(
                moments[..., index, 0], moments[..., index, 1:], geometry
            )
            for index, system in enumerate(self.groups)
        )
        return combine_stress_energy_projections(projections)

    def admissible(self, state: ArrayLike, geometry: ADMGridGeometry, /) -> Array:
        return self.closure(state, geometry).physically_valid


class GRMultigroupRadiationMatterExchange(StrictModule):
    group_exchanges: tuple[GRRadiationMatterExchange, ...]
    radiation_energy_source: Array
    radiation_flux_source: Array
    matter_energy_source: Array
    matter_momentum_source: Array
    energy_balance_residual: Array
    momentum_balance_residual: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class GRMultigroupRadiationInteractionPlan(StrictModule, NonTrainableState):
    radiation: GRMultigroupM1RadiationSystem
    group_interactions: tuple[GRGreyRadiationInteractionPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radiation: GRMultigroupM1RadiationSystem,
        group_interactions: tuple[GRGreyRadiationInteractionPlan, ...],
        /,
    ) -> None:
        if not isinstance(radiation, GRMultigroupM1RadiationSystem):
            raise TypeError("radiation must be GRMultigroupM1RadiationSystem.")
        interactions = tuple(group_interactions)
        if len(interactions) != radiation.group_count or any(
            not isinstance(value, GRGreyRadiationInteractionPlan)
            for value in interactions
        ):
            raise TypeError("One grey interaction is required per radiation group.")
        for system, interaction in zip(radiation.groups, interactions, strict=True):
            if interaction.radiation.system_id != system.system_id:
                raise ValueError("Multigroup interaction radiation systems differ.")
        self.radiation = radiation
        self.group_interactions = interactions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gr-multigroup-radiation-interaction",
                "radiation": radiation.system_id,
                "interactions": [value.interaction_id for value in interactions],
            }
        )

    def matter_exchange(
        self,
        state: ArrayLike,
        rest_mass_density: ArrayLike,
        fluid_velocity: ArrayLike,
        matter_temperature: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        magnetic_squared: ArrayLike = 0.0,
        composition: ArrayLike | None = None,
    ) -> GRMultigroupRadiationMatterExchange:
        moments = self.radiation.group_moments(state)
        exchanges = tuple(
            interaction.matter_exchange(
                moments[..., index, 0],
                moments[..., index, 1:],
                rest_mass_density,
                fluid_velocity,
                matter_temperature,
                geometry,
                magnetic_squared=magnetic_squared,
                composition=composition,
            )
            for index, interaction in enumerate(self.group_interactions)
        )
        energy_sources = jnp.stack(
            tuple(value.radiation_energy_source for value in exchanges), axis=-1
        )
        flux_sources = jnp.stack(
            tuple(value.radiation_flux_source for value in exchanges), axis=-2
        )
        matter_energy = -jnp.sum(energy_sources, axis=-1)
        matter_momentum = -jnp.sum(
            jnp.stack(
                tuple(value.radiation_momentum_source for value in exchanges),
                axis=-2,
            ),
            axis=-2,
        )
        energy_residual = jnp.sum(energy_sources, axis=-1) + matter_energy
        momentum_residual = (
            jnp.sum(
                jnp.stack(
                    tuple(value.radiation_momentum_source for value in exchanges),
                    axis=-2,
                ),
                axis=-2,
            )
            + matter_momentum
        )
        finite = jnp.all(jnp.stack(tuple(value.finite for value in exchanges)), axis=0)
        physical = jnp.all(
            jnp.stack(tuple(value.physically_valid for value in exchanges)), axis=0
        )
        qualified = jnp.all(
            jnp.stack(tuple(value.qualified for value in exchanges)), axis=0
        )
        derivative = jnp.all(
            jnp.stack(tuple(value.derivative_valid for value in exchanges)), axis=0
        )
        return GRMultigroupRadiationMatterExchange(
            exchanges,
            energy_sources,
            flux_sources,
            matter_energy,
            matter_momentum,
            energy_residual,
            momentum_residual,
            finite,
            physical,
            qualified,
            derivative,
            self.plan_id,
        )


__all__ = [
    "GRMultigroupM1ClosureEvaluation",
    "GRMultigroupM1RadiationSystem",
    "GRMultigroupRadiationInteractionPlan",
    "GRMultigroupRadiationMatterExchange",
]
