#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Surface energy balance with finite-thickness ablation recession."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._core import STEFAN_BOLTZMANN_W_M2_K4


@dataclass(frozen=True, slots=True)
class AblationSurfaceState:
    remaining_thickness_m: Array
    surface_temperature_k: Array


@dataclass(frozen=True, slots=True)
class AblationSurfaceStep:
    state: AblationSurfaceState
    net_heat_flux_w_m2: Array
    recession_m: Array
    removed_mass_kg_m2: Array
    rejected_energy_j_m2: Array
    energy_balance_residual_j_m2: Array
    candidate_state: AblationSurfaceState
    successful: Array
    exhausted: Array


@dataclass(frozen=True, slots=True)
class AblationSurfaceModel:
    density_kg_m3: float
    areal_heat_capacity_j_m2_k: float
    ablation_temperature_k: float
    effective_heat_of_ablation_j_kg: float
    emissivity: float
    absorptivity: float

    def __post_init__(self):
        positive = (
            self.density_kg_m3,
            self.areal_heat_capacity_j_m2_k,
            self.ablation_temperature_k,
            self.effective_heat_of_ablation_j_kg,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Ablation material parameters must be finite and positive.")
        if (
            not np.isfinite(self.emissivity)
            or not np.isfinite(self.absorptivity)
            or not 0 <= self.emissivity <= 1
            or not 0 <= self.absorptivity <= 1
        ):
            raise ValueError(
                "Ablation radiative properties must be finite and lie in [0, 1]."
            )

    def advance(
        self,
        state: AblationSurfaceState,
        convective_heat_flux_w_m2: float,
        incident_radiative_flux_w_m2: float,
        conductive_heat_loss_w_m2: float,
        environment_temperature_k: float,
        step_size_s: float,
        /,
    ) -> AblationSurfaceStep:
        thickness = jnp.asarray(state.remaining_thickness_m)
        temperature = jnp.asarray(state.surface_temperature_k)
        dtype = jnp.result_type(thickness, temperature)
        convective = jnp.asarray(convective_heat_flux_w_m2, dtype=dtype)
        incident = jnp.asarray(incident_radiative_flux_w_m2, dtype=dtype)
        conductive = jnp.asarray(conductive_heat_loss_w_m2, dtype=dtype)
        environment = jnp.asarray(environment_temperature_k, dtype=dtype)
        dt = jnp.asarray(step_size_s, dtype=dtype)
        input_valid = (
            jnp.all(jnp.isfinite(thickness))
            & jnp.all(jnp.isfinite(temperature))
            & jnp.all(thickness >= 0)
            & jnp.all(temperature > 0)
            & jnp.isfinite(convective)
            & jnp.isfinite(incident)
            & jnp.isfinite(conductive)
            & jnp.isfinite(environment)
            & (environment > 0)
            & jnp.isfinite(dt)
            & (dt > 0)
        )
        radiative_loss = (
            self.emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (temperature**4 - environment**4)
        )
        net_flux = convective + self.absorptivity * incident - conductive - radiative_loss
        supplied = dt * net_flux
        cooling = supplied <= 0
        cooling_temperature = temperature + supplied / self.areal_heat_capacity_j_m2_k
        heating_need = self.areal_heat_capacity_j_m2_k * jnp.maximum(
            self.ablation_temperature_k - temperature, 0
        )
        sensible = jnp.minimum(jnp.maximum(supplied, 0), heating_need)
        heated_temperature = temperature + sensible / self.areal_heat_capacity_j_m2_k
        available_ablation = jnp.maximum(supplied - sensible, 0)
        requested_mass = available_ablation / self.effective_heat_of_ablation_j_kg
        available_mass = self.density_kg_m3 * thickness
        removed_mass = jnp.minimum(requested_mass, available_mass)
        recession = removed_mass / self.density_kg_m3
        next_thickness = jnp.maximum(thickness - recession, 0)
        candidate_temperature = jnp.where(
            cooling, cooling_temperature, heated_temperature
        )
        candidate = AblationSurfaceState(next_thickness, candidate_temperature)
        successful = (
            input_valid
            & jnp.all(jnp.isfinite(next_thickness))
            & jnp.all(jnp.isfinite(candidate_temperature))
            & jnp.all(candidate_temperature > 0)
        )
        accepted = AblationSurfaceState(
            jnp.where(successful, next_thickness, thickness),
            jnp.where(successful, candidate_temperature, temperature),
        )
        actual_sensible = self.areal_heat_capacity_j_m2_k * (
            candidate_temperature - temperature
        )
        ablation_energy = removed_mass * self.effective_heat_of_ablation_j_kg
        rejected = supplied - actual_sensible - ablation_energy
        residual = supplied - actual_sensible - ablation_energy - rejected
        return AblationSurfaceStep(
            accepted,
            net_flux,
            recession,
            removed_mass,
            rejected,
            residual,
            candidate,
            successful,
            accepted.remaining_thickness_m <= 0,
        )


__all__ = ["AblationSurfaceModel", "AblationSurfaceState", "AblationSurfaceStep"]
