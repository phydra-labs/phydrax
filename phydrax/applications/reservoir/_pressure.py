#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Implicit finite-volume slightly-compressible reservoir pressure workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...ein import contract
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class ReservoirPressureState:
    pressure_pa: Array
    time_s: Array
    cumulative_production_m3: Array
    cumulative_injection_m3: Array


@dataclass(frozen=True, slots=True)
class ReservoirPressureStep:
    state: ReservoirPressureState
    well_production_rate_m3_s: Array
    storage_change_m3: Array
    volume_balance_residual_m3: Array
    pressure_residual_norm_m3_s: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class ReservoirPressureSystem:
    storage_m3_pa: Array
    transmissibility_m3_pa_s: Array

    @classmethod
    def create(
        cls,
        storage_m3_pa: ArrayLike,
        transmissibility_m3_pa_s: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> ReservoirPressureSystem:
        storage = np.asarray(storage_m3_pa, dtype=float)
        transmissibility = np.asarray(transmissibility_m3_pa_s, dtype=float)
        if storage.ndim != 1 or storage.size == 0 or np.any(storage <= 0):
            raise ValueError("Reservoir storage coefficients must be a positive vector.")
        if transmissibility.shape != (storage.size, storage.size):
            raise ValueError("Reservoir transmissibility has incompatible shape.")
        if not np.allclose(transmissibility, transmissibility.T, atol=tolerance, rtol=0):
            raise ValueError("Reservoir transmissibility must be symmetric.")
        if not np.allclose(transmissibility.sum(axis=0), 0, atol=tolerance, rtol=0):
            raise ValueError("Closed reservoir transmissibility must conserve volume.")
        if np.min(np.linalg.eigvalsh(transmissibility)) < -tolerance:
            raise ValueError(
                "Reservoir transmissibility Laplacian must be positive semidefinite."
            )
        return cls(jnp.asarray(storage), jnp.asarray(transmissibility))

    def advance(
        self,
        state: ReservoirPressureState,
        external_injection_rate_m3_s: ArrayLike,
        well_productivity_m3_pa_s: ArrayLike,
        bottom_hole_pressure_pa: ArrayLike,
        step_size_s: float,
        /,
    ) -> ReservoirPressureStep:
        pressure = jnp.asarray(state.pressure_pa)
        injection = jnp.asarray(external_injection_rate_m3_s)
        productivity = jnp.asarray(well_productivity_m3_pa_s)
        bottom_hole = jnp.asarray(bottom_hole_pressure_pa)
        shape = self.storage_m3_pa.shape
        if any(
            value.shape != shape
            for value in (pressure, injection, productivity, bottom_hole)
        ):
            raise ValueError("Reservoir state, sources, and wells must be cell aligned.")
        if bool(jnp.any(productivity < 0)) or step_size_s <= 0:
            raise ValueError("Reservoir productivity and step size are invalid.")
        dt = float(step_size_s)
        matrix = (
            jnp.diag(self.storage_m3_pa / dt + productivity)
            + self.transmissibility_m3_pa_s
        )
        right = (
            self.storage_m3_pa / dt * pressure + injection + productivity * bottom_hole
        )
        solved = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        well_rate = productivity * (solved.value - bottom_hole)
        storage_change = contract("q,q->", self.storage_m3_pa, solved.value - pressure)
        volume_balance = (
            storage_change + dt * jnp.sum(well_rate) - dt * jnp.sum(injection)
        )
        residual = matrix @ solved.value - right
        residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(residual), residual))
        )
        produced = dt * jnp.sum(jnp.maximum(well_rate, 0))
        injected = dt * (
            jnp.sum(jnp.maximum(injection, 0)) + jnp.sum(jnp.maximum(-well_rate, 0))
        )
        next_state = ReservoirPressureState(
            solved.value,
            jnp.asarray(state.time_s) + dt,
            jnp.asarray(state.cumulative_production_m3) + produced,
            jnp.asarray(state.cumulative_injection_m3) + injected,
        )
        return ReservoirPressureStep(
            next_state,
            well_rate,
            storage_change,
            volume_balance,
            residual_norm,
            solved.successful & jnp.all(jnp.isfinite(solved.value)),
        )


__all__ = [
    "ReservoirPressureState",
    "ReservoirPressureStep",
    "ReservoirPressureSystem",
]
