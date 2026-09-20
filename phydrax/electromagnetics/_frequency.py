#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Frequency-domain curl–curl Maxwell systems with energy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class MaxwellFrequencyResult:
    electric_field_coefficients: Array
    residual_norm: Array
    electric_energy_j: Array
    magnetic_energy_j: Array
    ohmic_dissipation_w: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class MaxwellFrequencySystem:
    curl_curl_operator: Array
    permittivity_mass: Array
    conductivity_mass: Array

    @classmethod
    def create(
        cls,
        curl_curl_operator: ArrayLike,
        permittivity_mass: ArrayLike,
        conductivity_mass: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> MaxwellFrequencySystem:
        curl_curl = np.asarray(curl_curl_operator, dtype=np.float64)
        permittivity = np.asarray(permittivity_mass, dtype=np.float64)
        conductivity = np.asarray(conductivity_mass, dtype=np.float64)
        if curl_curl.ndim != 2 or curl_curl.shape[0] != curl_curl.shape[1]:
            raise ValueError("Maxwell operators must be square.")
        if permittivity.shape != curl_curl.shape or conductivity.shape != curl_curl.shape:
            raise ValueError("Maxwell operators must share one edge basis.")
        if any(
            not np.allclose(value, value.T, atol=tolerance, rtol=0)
            for value in (curl_curl, permittivity, conductivity)
        ):
            raise ValueError("Maxwell energy and loss operators must be symmetric.")
        if np.min(np.linalg.eigvalsh(curl_curl)) < -tolerance:
            raise ValueError("Maxwell curl-curl operator must be positive semidefinite.")
        if np.min(np.linalg.eigvalsh(permittivity)) <= 0:
            raise ValueError("Maxwell permittivity mass must be positive definite.")
        if np.min(np.linalg.eigvalsh(conductivity)) < -tolerance:
            raise ValueError("Maxwell conductivity mass must be positive semidefinite.")
        return cls(
            jnp.asarray(curl_curl),
            jnp.asarray(permittivity),
            jnp.asarray(conductivity),
        )

    def solve(
        self,
        angular_frequency_rad_s: float,
        impressed_current_coefficients_a_m2: ArrayLike,
        /,
        *,
        relative_tolerance: float = 1e-9,
    ) -> MaxwellFrequencyResult:
        if angular_frequency_rad_s <= 0:
            raise ValueError(
                "Frequency-domain Maxwell solve requires positive frequency."
            )
        current = jnp.asarray(impressed_current_coefficients_a_m2)
        if current.shape != (self.curl_curl_operator.shape[0],):
            raise ValueError("Maxwell impressed current has incompatible shape.")
        omega = float(angular_frequency_rad_s)
        matrix = (
            self.curl_curl_operator.astype("complex128")
            - omega**2 * self.permittivity_mass
            + 1j * omega * self.conductivity_mass
        )
        right = 1j * omega * current.astype(matrix.dtype)
        solved = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        electric = solved.value
        residual = matrix @ electric - right
        residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(residual), residual))
        )
        right_norm = jnp.sqrt(jnp.real(contract("i,i->", jnp.conj(right), right)))
        electric_energy = 0.25 * jnp.real(
            contract("i,ij,j->", jnp.conj(electric), self.permittivity_mass, electric)
        )
        magnetic_energy = (
            0.25
            / omega**2
            * jnp.real(
                contract(
                    "i,ij,j->", jnp.conj(electric), self.curl_curl_operator, electric
                )
            )
        )
        ohmic = 0.5 * jnp.real(
            contract("i,ij,j->", jnp.conj(electric), self.conductivity_mass, electric)
        )
        successful = solved.successful & (
            residual_norm <= float(relative_tolerance) * jnp.maximum(right_norm, 1.0)
        )
        return MaxwellFrequencyResult(
            electric,
            residual_norm,
            electric_energy,
            magnetic_energy,
            ohmic,
            successful,
        )


__all__ = ["MaxwellFrequencyResult", "MaxwellFrequencySystem"]
