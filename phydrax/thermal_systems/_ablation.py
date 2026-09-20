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
        if not 0 <= self.emissivity <= 1 or not 0 <= self.absorptivity <= 1:
            raise ValueError("Ablation radiative properties must lie in [0, 1].")

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
        if bool((thickness < 0) | (temperature <= 0)) or step_size_s <= 0:
            raise ValueError("Ablation surface state or step size is invalid.")
        if environment_temperature_k <= 0:
            raise ValueError("Ablation environment temperature must be positive.")
        radiative_loss = (
            self.emissivity
            * STEFAN_BOLTZMANN_W_M2_K4
            * (temperature**4 - float(environment_temperature_k) ** 4)
        )
        net_flux = (
            float(convective_heat_flux_w_m2)
            + self.absorptivity * float(incident_radiative_flux_w_m2)
            - float(conductive_heat_loss_w_m2)
            - radiative_loss
        )
        supplied = float(step_size_s) * net_flux
        if bool(supplied <= 0):
            sensible = supplied
            next_temperature = jnp.maximum(
                temperature + sensible / self.areal_heat_capacity_j_m2_k, 0
            )
            actual_sensible = self.areal_heat_capacity_j_m2_k * (
                next_temperature - temperature
            )
            rejected = supplied - actual_sensible
            return AblationSurfaceStep(
                AblationSurfaceState(thickness, next_temperature),
                net_flux,
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                rejected,
                supplied - actual_sensible - rejected,
                thickness <= 0,
            )

        heating_need = self.areal_heat_capacity_j_m2_k * jnp.maximum(
            self.ablation_temperature_k - temperature, 0
        )
        sensible = jnp.minimum(supplied, heating_need)
        next_temperature = temperature + sensible / self.areal_heat_capacity_j_m2_k
        available_ablation = supplied - sensible
        requested_mass = available_ablation / self.effective_heat_of_ablation_j_kg
        available_mass = self.density_kg_m3 * thickness
        removed_mass = jnp.minimum(requested_mass, available_mass)
        recession = removed_mass / self.density_kg_m3
        next_thickness = jnp.maximum(thickness - recession, 0)
        ablation_energy = removed_mass * self.effective_heat_of_ablation_j_kg
        rejected = available_ablation - ablation_energy
        residual = supplied - sensible - ablation_energy - rejected
        return AblationSurfaceStep(
            AblationSurfaceState(next_thickness, next_temperature),
            net_flux,
            recession,
            removed_mass,
            rejected,
            residual,
            next_thickness <= 0,
        )


__all__ = ["AblationSurfaceModel", "AblationSurfaceState", "AblationSurfaceStep"]
