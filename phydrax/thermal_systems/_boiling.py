#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous pool-boiling regime map and wall energy balance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


BoilingRegime: TypeAlias = Literal["single-phase", "nucleate", "transition", "film"]


@dataclass(frozen=True, slots=True)
class BoilingEvaluation:
    heat_flux_w_m2: Array
    vapor_mass_flux_kg_m2_s: Array
    superheat_k: Array
    regime: BoilingRegime


@dataclass(frozen=True, slots=True)
class BoilingWallStep:
    temperature_k: Array
    boiling: BoilingEvaluation
    energy_balance_residual_j_m2: Array


@dataclass(frozen=True, slots=True)
class PoolBoilingCurve:
    saturation_temperature_k: float
    single_phase_coefficient_w_m2_k: float
    nucleate_coefficient_w_m2_k3: float
    critical_heat_flux_w_m2: float
    minimum_film_heat_flux_w_m2: float
    leidenfrost_superheat_k: float
    film_coefficient_w_m2_k: float
    latent_heat_j_kg: float

    def __post_init__(self):
        values = (
            self.saturation_temperature_k,
            self.single_phase_coefficient_w_m2_k,
            self.nucleate_coefficient_w_m2_k3,
            self.critical_heat_flux_w_m2,
            self.minimum_film_heat_flux_w_m2,
            self.leidenfrost_superheat_k,
            self.film_coefficient_w_m2_k,
            self.latent_heat_j_kg,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Pool-boiling parameters must be finite and positive.")
        if self.minimum_film_heat_flux_w_m2 >= self.critical_heat_flux_w_m2:
            raise ValueError(
                "Minimum film-boiling flux must be below critical heat flux."
            )
        if self.critical_superheat_k >= self.leidenfrost_superheat_k:
            raise ValueError("Leidenfrost superheat must exceed critical superheat.")

    @property
    def critical_superheat_k(self) -> float:
        return (self.critical_heat_flux_w_m2 / self.nucleate_coefficient_w_m2_k3) ** (
            1.0 / 3.0
        )

    def evaluate(self, wall_temperature_k: float, /) -> BoilingEvaluation:
        if not np.isfinite(wall_temperature_k) or wall_temperature_k <= 0:
            raise ValueError("Boiling wall temperature must be finite and positive.")
        superheat = float(wall_temperature_k) - self.saturation_temperature_k
        if superheat <= 0:
            flux = self.single_phase_coefficient_w_m2_k * superheat
            regime: BoilingRegime = "single-phase"
        elif superheat <= self.critical_superheat_k:
            flux = self.nucleate_coefficient_w_m2_k3 * superheat**3
            regime = "nucleate"
        elif superheat < self.leidenfrost_superheat_k:
            fraction = (superheat - self.critical_superheat_k) / (
                self.leidenfrost_superheat_k - self.critical_superheat_k
            )
            flux = self.critical_heat_flux_w_m2 + fraction * (
                self.minimum_film_heat_flux_w_m2 - self.critical_heat_flux_w_m2
            )
            regime = "transition"
        else:
            flux = self.minimum_film_heat_flux_w_m2 + self.film_coefficient_w_m2_k * (
                superheat - self.leidenfrost_superheat_k
            )
            regime = "film"
        heat_flux = jnp.asarray(flux)
        vapor_flux = jnp.maximum(heat_flux, 0) / self.latent_heat_j_kg
        return BoilingEvaluation(heat_flux, vapor_flux, jnp.asarray(superheat), regime)

    def advance_wall(
        self,
        wall_temperature_k: float,
        imposed_heat_flux_w_m2: float,
        areal_heat_capacity_j_m2_k: float,
        step_size_s: float,
        /,
    ) -> BoilingWallStep:
        if areal_heat_capacity_j_m2_k <= 0 or step_size_s <= 0:
            raise ValueError("Boiling wall capacity and step size must be positive.")
        boiling = self.evaluate(wall_temperature_k)
        net_flux = float(imposed_heat_flux_w_m2) - boiling.heat_flux_w_m2
        temperature_change = (
            float(step_size_s) * net_flux / float(areal_heat_capacity_j_m2_k)
        )
        next_temperature = jnp.asarray(wall_temperature_k) + temperature_change
        stored = float(areal_heat_capacity_j_m2_k) * temperature_change
        supplied = float(step_size_s) * float(imposed_heat_flux_w_m2)
        removed = float(step_size_s) * boiling.heat_flux_w_m2
        return BoilingWallStep(
            next_temperature,
            boiling,
            supplied - removed - stored,
        )


__all__ = [
    "BoilingEvaluation",
    "BoilingRegime",
    "BoilingWallStep",
    "PoolBoilingCurve",
]
