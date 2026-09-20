#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Radiation–diffraction marine frequency response and absorbed power."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...ein import contract
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class MarineFrequencyResult:
    angular_frequency_rad_s: Array
    response_amplitude: Array
    residual_norm: Array
    radiation_dissipation_w: Array
    absorbed_power_w: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class MarineFrequencySystem:
    angular_frequency_rad_s: Array
    rigid_mass: Array
    structural_damping: Array
    hydrostatic_mooring_stiffness: Array
    added_mass: Array
    radiation_damping: Array
    pto_damping: Array

    @classmethod
    def create(
        cls,
        angular_frequency_rad_s: ArrayLike,
        rigid_mass: ArrayLike,
        structural_damping: ArrayLike,
        hydrostatic_mooring_stiffness: ArrayLike,
        added_mass: ArrayLike,
        radiation_damping: ArrayLike,
        pto_damping: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> MarineFrequencySystem:
        frequency = np.asarray(angular_frequency_rad_s, dtype=np.float64)
        mass = np.asarray(rigid_mass, dtype=np.float64)
        damping = np.asarray(structural_damping, dtype=np.float64)
        stiffness = np.asarray(hydrostatic_mooring_stiffness, dtype=np.float64)
        added = np.asarray(added_mass, dtype=np.float64)
        radiation = np.asarray(radiation_damping, dtype=np.float64)
        pto = np.asarray(pto_damping, dtype=np.float64)
        if (
            frequency.ndim != 1
            or frequency.size == 0
            or np.any(frequency <= 0)
            or np.any(np.diff(frequency) <= 0)
        ):
            raise ValueError(
                "Marine frequencies must be positive and strictly increasing."
            )
        if mass.ndim != 2 or mass.shape[0] != mass.shape[1]:
            raise ValueError("Marine rigid mass must be square.")
        size = mass.shape[0]
        if any(value.shape != (size, size) for value in (damping, stiffness, pto)):
            raise ValueError("Marine fixed operators must share one motion basis.")
        if added.shape != (frequency.size, size, size) or radiation.shape != added.shape:
            raise ValueError(
                "Marine hydrodynamic coefficients must be frequency aligned."
            )
        for value in (mass, damping, stiffness, pto, *added, *radiation):
            if not np.allclose(value, value.T, atol=tolerance, rtol=0):
                raise ValueError("Marine operators must be symmetric.")
        if np.min(np.linalg.eigvalsh(mass)) <= 0:
            raise ValueError("Marine rigid mass must be positive definite.")
        return cls(
            jnp.asarray(frequency),
            jnp.asarray(mass),
            jnp.asarray(damping),
            jnp.asarray(stiffness),
            jnp.asarray(added),
            jnp.asarray(radiation),
            jnp.asarray(pto),
        )

    def solve(self, wave_excitation_n: ArrayLike, /) -> MarineFrequencyResult:
        excitation = jnp.asarray(wave_excitation_n)
        count = self.angular_frequency_rad_s.size
        size = self.rigid_mass.shape[0]
        if excitation.shape != (count, size):
            raise ValueError(
                "Marine wave excitation must have shape (frequency, motion)."
            )
        response = []
        residuals = []
        radiation_power = []
        absorbed_power = []
        successes = []
        for index in range(count):
            omega = self.angular_frequency_rad_s[index]
            total_mass = self.rigid_mass + self.added_mass[index]
            total_damping = (
                self.structural_damping + self.radiation_damping[index] + self.pto_damping
            )
            dynamic = (
                self.hydrostatic_mooring_stiffness.astype("complex128")
                - omega**2 * total_mass
                + 1j * omega * total_damping
            )
            right = excitation[index].astype(dynamic.dtype)
            solved = solve(
                LinearSystem(DenseLinearOperator(dynamic)),
                right,
                policy=LinearSolvePolicy(DenseLU()),
            )
            residual = dynamic @ solved.value - right
            residual_norm = jnp.sqrt(
                jnp.real(contract("i,i->", jnp.conj(residual), residual))
            )
            velocity = 1j * omega * solved.value
            radiation_value = 0.5 * jnp.real(
                contract(
                    "i,ij,j->",
                    jnp.conj(velocity),
                    self.radiation_damping[index],
                    velocity,
                )
            )
            absorbed_value = 0.5 * jnp.real(
                contract("i,ij,j->", jnp.conj(velocity), self.pto_damping, velocity)
            )
            response.append(solved.value)
            residuals.append(residual_norm)
            radiation_power.append(radiation_value)
            absorbed_power.append(absorbed_value)
            successes.append(solved.successful)
        return MarineFrequencyResult(
            self.angular_frequency_rad_s,
            jnp.stack(response),
            jnp.stack(residuals),
            jnp.stack(radiation_power),
            jnp.stack(absorbed_power),
            jnp.stack(successes),
        )


__all__ = ["MarineFrequencyResult", "MarineFrequencySystem"]
