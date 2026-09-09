#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_rates import ChemicalRateRuntime
from ...equations._electrochemistry import FARADAY_CONSTANT
from ...equations._ionized_gas import (
    IonizedMultitemperatureEulerSystem,
    IonizedMultitemperatureNavierStokesSystem,
)
from ...equations._surface_chemistry import (
    GasSurfaceChemicalEvaluation,
    PreparedGasSurfaceMechanism,
    SurfaceChemicalState,
)


class ReactingPlasmaWallExchange(StrictModule):
    exterior_state: Array
    conservative_outward_flux: Array
    gas_species_mass_flux: Array
    surface_candidate: SurfaceChemicalState
    surface_accepted: SurfaceChemicalState
    blowing_velocity: Array
    electric_current_density: Array
    catalytic_heat_flux: Array
    net_heat_to_material: Array
    chemistry: GasSurfaceChemicalEvaluation
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReactingPlasmaWallPlan(StrictModule, NonTrainableState):
    """Finite-rate plasma wall with one gas/surface mass, charge, and heat ledger."""

    mechanism: PreparedGasSurfaceMechanism
    wall_velocity: Array
    surface_mass_per_recession: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedGasSurfaceMechanism,
        wall_velocity: ArrayLike,
        /,
        *,
        surface_mass_per_recession: float = 1.0,
    ):
        velocity = jnp.asarray(wall_velocity)
        scale = float(surface_mass_per_recession)
        if (
            not isinstance(mechanism, PreparedGasSurfaceMechanism)
            or velocity.ndim != 1
            or velocity.size not in (1, 2, 3)
            or np.any(~np.isfinite(np.asarray(velocity)))
            or not np.isfinite(scale)
            or scale <= 0.0
        ):
            raise ValueError("Reacting plasma wall inputs are invalid.")
        self.mechanism = mechanism
        self.wall_velocity = velocity
        self.surface_mass_per_recession = scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reacting-plasma-wall",
                "mechanism": mechanism.mechanism_id,
                "wall_velocity": array_tree_fingerprint(velocity),
                "surface_mass_per_recession": scale,
            }
        )

    def evaluate(
        self,
        system: IonizedMultitemperatureEulerSystem
        | IonizedMultitemperatureNavierStokesSystem,
        interior_state: ArrayLike,
        outward_normal: ArrayLike,
        surface_state: SurfaceChemicalState,
        step_size: ArrayLike,
        /,
        *,
        conductive_heat_to_material: ArrayLike = 0.0,
        radiative_heat_to_material: ArrayLike = 0.0,
        runtime: ChemicalRateRuntime | None = None,
    ) -> ReactingPlasmaWallExchange:
        if not isinstance(
            system,
            (
                IonizedMultitemperatureEulerSystem,
                IonizedMultitemperatureNavierStokesSystem,
            ),
        ):
            raise TypeError("Reacting plasma wall requires an ionized gas system.")
        interior = jnp.asarray(interior_state)
        normal = jnp.asarray(outward_normal, dtype=interior.dtype)
        step = jnp.asarray(step_size, dtype=interior.dtype)
        if (
            interior.shape[-1] != system.component_count
            or normal.shape[-1] != system.dimension
            or self.wall_velocity.shape != (system.dimension,)
        ):
            raise ValueError("Wall state, normal, or velocity shape is invalid.")
        recovered = system.recover_thermodynamics(interior)
        concentration = interior[
            ..., : system.species_count
        ] / system.thermodynamics.schema.molar_masses.astype(interior.dtype)
        chemistry = self.mechanism.evaluate(
            concentration,
            surface_state,
            recovered.total_pressure,
            runtime=runtime,
        )
        molar_masses = system.thermodynamics.schema.molar_masses.astype(interior.dtype)
        species_mass_flux = chemistry.gas_amount_flux * molar_masses
        total_mass_flux = jnp.sum(species_mass_flux, axis=-1)
        density = system.density(interior)
        blowing_velocity = -total_mass_flux / jnp.maximum(density, system.density_floor)
        current = FARADAY_CONSTANT * contract(
            "s,...s->...",
            system.thermodynamics.schema.charges.astype(interior.dtype),
            chemistry.gas_amount_flux,
            backend="jax",
        )
        catalytic_heat = -chemistry.reaction_heat_flux
        net_heat = (
            jnp.asarray(conductive_heat_to_material, dtype=interior.dtype)
            + jnp.asarray(radiative_heat_to_material, dtype=interior.dtype)
            + catalytic_heat
        )
        surface_amount_candidate = (
            surface_state.amounts + step[..., None] * chemistry.surface_amount_rate
        )
        recession_mass = surface_state.cumulative_recession_mass + step * jnp.maximum(
            -total_mass_flux, 0.0
        )
        surface_candidate = SurfaceChemicalState(
            surface_amount_candidate,
            surface_state.temperature,
            recession_mass,
            jnp.all(jnp.isfinite(surface_amount_candidate), axis=-1),
        )
        site_total = contract(
            "s,...s->...",
            self.mechanism.surface_schema.site_occupancy,
            surface_amount_candidate,
            backend="jax",
        )
        surface_successful = (
            surface_candidate.finite
            & jnp.all(surface_amount_candidate >= 0.0, axis=-1)
            & (
                site_total
                <= self.mechanism.surface_schema.site_capacity * (1.0 + 1.0e-10)
            )
        )
        successful = jnp.all(chemistry.successful & surface_successful)
        surface_accepted = SurfaceChemicalState(
            jnp.where(successful, surface_candidate.amounts, surface_state.amounts),
            surface_state.temperature,
            jnp.where(
                successful, recession_mass, surface_state.cumulative_recession_mass
            ),
            jnp.asarray(successful),
        )
        primitive = system.conserved_to_primitive(interior)
        reflected_velocity = 2.0 * self.wall_velocity - system.primitive_velocity(
            primitive
        )
        exterior_primitive = system.with_primitive_velocity(primitive, reflected_velocity)
        exterior_primitive = system.with_primitive_temperature(
            exterior_primitive,
            2.0 * surface_state.temperature - system.temperature(interior),
        )
        exterior = system.primitive_to_conserved(exterior_primitive)
        flux = jnp.zeros_like(interior)
        flux = flux.at[..., : system.species_count].set(species_mass_flux)
        pressure_traction = recovered.total_pressure[..., None] * normal
        flux = flux.at[..., system.momentum_slice].set(
            pressure_traction + total_mass_flux[..., None] * self.wall_velocity
        )
        flux = flux.at[..., system.energy_index].set(net_heat)
        finite = (
            jnp.all(jnp.isfinite(exterior), axis=-1)
            & jnp.all(jnp.isfinite(flux), axis=-1)
            & jnp.isfinite(current)
            & jnp.isfinite(net_heat)
        )
        success = finite & successful & jnp.all(system.admissible(exterior))
        return ReactingPlasmaWallExchange(
            exterior,
            flux,
            species_mass_flux,
            surface_candidate,
            surface_accepted,
            blowing_velocity,
            current,
            catalytic_heat,
            net_heat,
            chemistry,
            finite,
            success,
            self.plan_id,
        )


__all__ = ["ReactingPlasmaWallExchange", "ReactingPlasmaWallPlan"]
