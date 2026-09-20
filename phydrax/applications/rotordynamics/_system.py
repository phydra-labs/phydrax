#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Gyroscopic rotor-bearing frequency response and unbalance loading."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...ein import contract
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class RotorFrequencyResult:
    spin_speed_rad_s: Array
    displacement: Array
    residual_norm: Array
    bearing_dissipation_w: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class RotorSystem:
    mass: Array
    damping: Array
    gyroscopic: Array
    stiffness: Array

    @classmethod
    def create(
        cls,
        mass: ArrayLike,
        damping: ArrayLike,
        gyroscopic: ArrayLike,
        stiffness: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> RotorSystem:
        mass_ = np.asarray(mass, dtype=np.float64)
        damping_ = np.asarray(damping, dtype=np.float64)
        gyroscopic_ = np.asarray(gyroscopic, dtype=np.float64)
        stiffness_ = np.asarray(stiffness, dtype=np.float64)
        if mass_.ndim != 2 or mass_.shape[0] != mass_.shape[1]:
            raise ValueError("Rotor matrices must be square.")
        if any(
            value.shape != mass_.shape for value in (damping_, gyroscopic_, stiffness_)
        ):
            raise ValueError("Rotor matrices must share one coordinate basis.")
        if not np.allclose(mass_, mass_.T, atol=tolerance, rtol=0) or not np.allclose(
            stiffness_, stiffness_.T, atol=tolerance, rtol=0
        ):
            raise ValueError("Rotor mass and stiffness must be symmetric.")
        if not np.allclose(damping_, damping_.T, atol=tolerance, rtol=0):
            raise ValueError("Rotor damping must be symmetric.")
        if not np.allclose(gyroscopic_, -gyroscopic_.T, atol=tolerance, rtol=0):
            raise ValueError("Rotor gyroscopic matrix must be skew-symmetric.")
        if (
            np.min(np.linalg.eigvalsh(mass_)) <= 0
            or np.min(np.linalg.eigvalsh(stiffness_)) <= 0
        ):
            raise ValueError("Rotor mass and stiffness must be positive definite.")
        return cls(
            jnp.asarray(mass_),
            jnp.asarray(damping_),
            jnp.asarray(gyroscopic_),
            jnp.asarray(stiffness_),
        )

    def unbalance_force(
        self,
        spin_speed_rad_s: ArrayLike,
        unbalance_kg_m: ArrayLike,
        phase_rad: ArrayLike,
        /,
    ) -> Array:
        speed = jnp.asarray(spin_speed_rad_s)
        unbalance = jnp.asarray(unbalance_kg_m)
        phase = jnp.asarray(phase_rad)
        if unbalance.shape != (self.mass.shape[0],) or phase.shape != unbalance.shape:
            raise ValueError(
                "Rotor unbalance magnitude and phase must align with coordinates."
            )
        return speed[..., None] ** 2 * unbalance * jnp.exp(1j * phase)

    def frequency_response(
        self,
        spin_speed_rad_s: ArrayLike,
        force_n: ArrayLike,
        /,
        *,
        excitation_order: float = 1.0,
        relative_tolerance: float = 1e-9,
    ) -> RotorFrequencyResult:
        speed = jnp.atleast_1d(jnp.asarray(spin_speed_rad_s))
        force = jnp.asarray(force_n)
        size = self.mass.shape[0]
        if force.shape == (size,):
            force = jnp.broadcast_to(force, (speed.size, size))
        if force.shape != (speed.size, size) or bool(jnp.any(speed < 0)):
            raise ValueError("Rotor speeds or forces have incompatible shapes.")
        responses = []
        residuals = []
        successes = []
        dissipations = []
        for index in range(speed.size):
            excitation = float(excitation_order) * speed[index]
            dynamic = (
                self.stiffness.astype("complex128")
                - excitation**2 * self.mass
                + 1j * excitation * (self.damping + speed[index] * self.gyroscopic)
            )
            right = force[index].astype(dynamic.dtype)
            result = solve(
                LinearSystem(DenseLinearOperator(dynamic)),
                right,
                policy=LinearSolvePolicy(DenseLU()),
            )
            residual = dynamic @ result.value - right
            residual_norm = jnp.sqrt(
                jnp.real(contract("i,i->", jnp.conj(residual), residual))
            )
            force_norm = jnp.sqrt(jnp.real(contract("i,i->", jnp.conj(right), right)))
            velocity = 1j * excitation * result.value
            dissipation = 0.5 * jnp.real(
                contract("i,ij,j->", jnp.conj(velocity), self.damping, velocity)
            )
            responses.append(result.value)
            residuals.append(residual_norm)
            dissipations.append(dissipation)
            successes.append(
                result.successful
                & (
                    residual_norm
                    <= float(relative_tolerance) * jnp.maximum(force_norm, 1.0)
                )
            )
        return RotorFrequencyResult(
            speed,
            jnp.stack(responses),
            jnp.stack(residuals),
            jnp.stack(dissipations),
            jnp.stack(successes),
        )


__all__ = ["RotorFrequencyResult", "RotorSystem"]
