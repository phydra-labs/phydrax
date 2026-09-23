#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Reciprocal finite-dimensional vibroacoustic frequency systems."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class VibroacousticResult:
    displacement_m: Array
    acoustic_pressure_pa: Array
    structural_residual_norm: Array
    acoustic_residual_norm: Array
    interface_power_w: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class VibroacousticSystem:
    structural_mass: Array
    structural_damping: Array
    structural_stiffness: Array
    acoustic_mass: Array
    acoustic_damping: Array
    acoustic_stiffness: Array
    reciprocal_coupling: Array

    @classmethod
    def create(
        cls,
        structural_mass: ArrayLike,
        structural_damping: ArrayLike,
        structural_stiffness: ArrayLike,
        acoustic_mass: ArrayLike,
        acoustic_damping: ArrayLike,
        acoustic_stiffness: ArrayLike,
        reciprocal_coupling: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> VibroacousticSystem:
        raw_values = (
            structural_mass,
            structural_damping,
            structural_stiffness,
            acoustic_mass,
            acoustic_damping,
            acoustic_stiffness,
            reciprocal_coupling,
        )
        if any(np.iscomplexobj(value) for value in raw_values):
            raise TypeError("Vibroacoustic systems require real-valued operators.")
        structural = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (structural_mass, structural_damping, structural_stiffness)
        )
        acoustic = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (acoustic_mass, acoustic_damping, acoustic_stiffness)
        )
        coupling = np.asarray(reciprocal_coupling, dtype=np.float64)
        if structural[0].ndim != 2 or structural[0].shape[0] != structural[0].shape[1]:
            raise ValueError("Vibroacoustic structural matrices must be square.")
        if acoustic[0].ndim != 2 or acoustic[0].shape[0] != acoustic[0].shape[1]:
            raise ValueError("Vibroacoustic acoustic matrices must be square.")
        if any(value.shape != structural[0].shape for value in structural[1:]):
            raise ValueError("Vibroacoustic structural matrices must align.")
        if any(value.shape != acoustic[0].shape for value in acoustic[1:]):
            raise ValueError("Vibroacoustic acoustic matrices must align.")
        if coupling.shape != (structural[0].shape[0], acoustic[0].shape[0]):
            raise ValueError("Vibroacoustic reciprocal coupling has incompatible shape.")
        if not all(
            np.all(np.isfinite(value)) for value in (*structural, *acoustic, coupling)
        ):
            raise ValueError("Vibroacoustic operators must be finite.")
        if any(
            not np.allclose(value, value.T, atol=tolerance, rtol=0)
            for value in (*structural, *acoustic)
        ):
            raise ValueError("Vibroacoustic diagonal operators must be symmetric.")
        if (
            np.min(np.linalg.eigvalsh(structural[0])) <= tolerance
            or np.min(np.linalg.eigvalsh(acoustic[0])) <= tolerance
            or any(
                np.min(np.linalg.eigvalsh(value)) < -tolerance
                for value in (
                    structural[1],
                    structural[2],
                    acoustic[1],
                    acoustic[2],
                )
            )
        ):
            raise ValueError(
                "Vibroacoustic mass must be positive definite and damping/stiffness positive semidefinite."
            )
        return cls(
            *(jnp.asarray(value) for value in structural),
            *(jnp.asarray(value) for value in acoustic),
            jnp.asarray(coupling),
        )

    def solve(
        self,
        angular_frequency_rad_s: float,
        structural_force_n: ArrayLike,
        acoustic_source: ArrayLike,
        /,
    ) -> VibroacousticResult:
        if not isfinite(angular_frequency_rad_s) or angular_frequency_rad_s < 0:
            raise ValueError("Vibroacoustic frequency must be finite and non-negative.")
        force = jnp.asarray(structural_force_n)
        source = jnp.asarray(acoustic_source)
        structural_size = self.structural_mass.shape[0]
        acoustic_size = self.acoustic_mass.shape[0]
        if force.shape != (structural_size,) or source.shape != (acoustic_size,):
            raise ValueError("Vibroacoustic loads have incompatible shapes.")
        if not bool(jnp.all(jnp.isfinite(force))) or not bool(
            jnp.all(jnp.isfinite(source))
        ):
            raise ValueError("Vibroacoustic loads must be finite.")
        omega = float(angular_frequency_rad_s)
        structural_dynamic = (
            self.structural_stiffness.astype("complex128")
            - omega**2 * self.structural_mass
            + 1j * omega * self.structural_damping
        )
        acoustic_dynamic = (
            self.acoustic_stiffness.astype("complex128")
            - omega**2 * self.acoustic_mass
            + 1j * omega * self.acoustic_damping
        )
        matrix = jnp.block(
            [
                [structural_dynamic, -self.reciprocal_coupling],
                [-self.reciprocal_coupling.T, acoustic_dynamic],
            ]
        )
        right = jnp.concatenate((force, source)).astype(matrix.dtype)
        solved = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        displacement = solved.value[:structural_size]
        pressure = solved.value[structural_size:]
        structural_residual = (
            structural_dynamic @ displacement
            - self.reciprocal_coupling @ pressure
            - force
        )
        acoustic_residual = (
            acoustic_dynamic @ pressure
            - self.reciprocal_coupling.T @ displacement
            - source
        )
        structural_norm = jnp.sqrt(
            jnp.real(
                contract("i,i->", jnp.conj(structural_residual), structural_residual)
            )
        )
        acoustic_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(acoustic_residual), acoustic_residual))
        )
        coupling_force = self.reciprocal_coupling @ pressure
        interface_power = 0.5 * jnp.real(
            contract("i,i->", coupling_force, jnp.conj(1j * omega * displacement))
        )
        return VibroacousticResult(
            displacement,
            pressure,
            structural_norm,
            acoustic_norm,
            interface_power,
            solved.successful
            & jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(pressure)),
        )


__all__ = ["VibroacousticResult", "VibroacousticSystem"]
