#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compiled frequency-domain systems and native complex solves."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class FrequencyDomainResult:
    angular_frequency_rad_s: Array
    response: Array
    residual_norm: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class CompiledFrequencySystem:
    """Second-order system K x + C ẋ + M ẍ = f in a fixed coordinate basis."""

    mass: Array
    damping: Array
    stiffness: Array

    @classmethod
    def create(
        cls,
        mass: ArrayLike,
        damping: ArrayLike,
        stiffness: ArrayLike,
        /,
    ) -> CompiledFrequencySystem:
        matrices = tuple(np.asarray(value) for value in (mass, damping, stiffness))
        if matrices[0].ndim != 2 or matrices[0].shape[0] != matrices[0].shape[1]:
            raise ValueError("Frequency system matrices must be square.")
        if any(value.shape != matrices[0].shape for value in matrices[1:]):
            raise ValueError("Frequency system matrices must share one coordinate basis.")
        if any(not np.all(np.isfinite(value)) for value in matrices):
            raise ValueError("Frequency system matrices must be finite.")
        return cls(*(jnp.asarray(value) for value in matrices))

    @property
    def size(self) -> int:
        return int(self.mass.shape[0])

    def dynamic_stiffness(self, angular_frequency_rad_s: ArrayLike, /) -> Array:
        omega = jnp.asarray(angular_frequency_rad_s)
        return (
            self.stiffness
            + 1j * omega[..., None, None] * self.damping
            - omega[..., None, None] ** 2 * self.mass
        )

    def solve(
        self,
        angular_frequency_rad_s: ArrayLike,
        forcing: ArrayLike,
        /,
        *,
        relative_tolerance: float = 1e-9,
    ) -> FrequencyDomainResult:
        omega = jnp.atleast_1d(jnp.asarray(angular_frequency_rad_s))
        loads = jnp.asarray(forcing)
        if loads.shape == (self.size,):
            loads = jnp.broadcast_to(loads, (omega.size, self.size))
        if loads.shape != (omega.size, self.size):
            raise ValueError("Frequency forcing must have shape (frequency, coordinate).")
        if bool(jnp.any(omega < 0)):
            raise ValueError("Angular frequencies must be non-negative.")

        matrices = self.dynamic_stiffness(omega)
        loads = loads.astype(matrices.dtype)
        solved = tuple(
            solve(
                LinearSystem(DenseLinearOperator(matrices[index])),
                loads[index],
                policy=LinearSolvePolicy(DenseLU()),
            )
            for index in range(omega.size)
        )
        response = jnp.stack(tuple(value.value for value in solved))
        residual = contract("fij,fj->fi", matrices, response) - loads
        residual_norm = jnp.sqrt(
            jnp.real(contract("fi,fi->f", jnp.conj(residual), residual))
        )
        load_norm = jnp.sqrt(jnp.real(contract("fi,fi->f", jnp.conj(loads), loads)))
        native_success = jnp.stack(tuple(value.successful for value in solved))
        successful = native_success & (
            residual_norm <= float(relative_tolerance) * jnp.maximum(load_norm, 1.0)
        )
        return FrequencyDomainResult(omega, response, residual_norm, successful)


__all__ = ["CompiledFrequencySystem", "FrequencyDomainResult"]
