#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Reciprocal diffuse-gray enclosure radiation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._core import STEFAN_BOLTZMANN_W_M2_K4


@dataclass(frozen=True, slots=True)
class EnclosureRadiationResult:
    radiosity_w_m2: Array
    irradiation_w_m2: Array
    outward_heat_flux_w_m2: Array
    surface_power_w: Array
    enclosure_balance_w: Array
    residual_norm: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class DiffuseGrayEnclosure:
    surface_areas_m2: Array
    emissivity: Array
    view_factors: Array

    @classmethod
    def create(
        cls,
        surface_areas_m2: ArrayLike,
        emissivity: ArrayLike,
        view_factors: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> DiffuseGrayEnclosure:
        areas = np.asarray(surface_areas_m2, dtype=float)
        emissivity_ = np.asarray(emissivity, dtype=float)
        factors = np.asarray(view_factors, dtype=float)
        if areas.ndim != 1 or areas.size < 2 or np.any(areas <= 0):
            raise ValueError("Radiation enclosure areas must be a positive vector.")
        if emissivity_.shape != areas.shape or np.any(
            (emissivity_ <= 0) | (emissivity_ > 1)
        ):
            raise ValueError("Radiation emissivities must lie in (0, 1].")
        if factors.shape != (areas.size, areas.size) or np.any(factors < 0):
            raise ValueError("Radiation view factors have incompatible shape or signs.")
        if not np.allclose(factors.sum(axis=1), 1, atol=tolerance, rtol=0):
            raise ValueError("Closed-enclosure view-factor rows must sum to one.")
        reciprocity = areas[:, None] * factors - areas[None, :] * factors.T
        if not np.allclose(reciprocity, 0, atol=tolerance, rtol=tolerance):
            raise ValueError("Radiation view factors must satisfy area reciprocity.")
        return cls(jnp.asarray(areas), jnp.asarray(emissivity_), jnp.asarray(factors))

    def solve(self, temperature_k: ArrayLike, /) -> EnclosureRadiationResult:
        temperature = jnp.asarray(temperature_k)
        if temperature.shape != self.surface_areas_m2.shape or bool(
            jnp.any(temperature <= 0)
        ):
            raise ValueError(
                "Radiation temperatures must be positive and surface aligned."
            )
        matrix = (
            jnp.eye(temperature.size, dtype=temperature.dtype)
            - (1 - self.emissivity)[:, None] * self.view_factors
        )
        emitted = self.emissivity * STEFAN_BOLTZMANN_W_M2_K4 * temperature**4
        solved = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            emitted,
            policy=LinearSolvePolicy(DenseLU()),
        )
        irradiation = self.view_factors @ solved.value
        flux = solved.value - irradiation
        power = self.surface_areas_m2 * flux
        residual = matrix @ solved.value - emitted
        residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(residual), residual))
        )
        balance = jnp.sum(power)
        return EnclosureRadiationResult(
            solved.value,
            irradiation,
            flux,
            power,
            balance,
            residual_norm,
            solved.successful & jnp.all(jnp.isfinite(power)),
        )


__all__ = ["DiffuseGrayEnclosure", "EnclosureRadiationResult"]
