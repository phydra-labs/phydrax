#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Monolithic spatial smart-material field systems."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class PiezoelectricSolveResult:
    displacement: Array
    electric_potential_v: Array
    residual_norm: Array
    electric_enthalpy_j: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SpatialPiezoelectricSystem:
    """Reciprocal finite-dimensional piezoelectric enthalpy system."""

    mechanical_stiffness: Array
    dielectric_stiffness: Array
    electromechanical_coupling: Array

    @classmethod
    def create(
        cls,
        mechanical_stiffness: ArrayLike,
        dielectric_stiffness: ArrayLike,
        electromechanical_coupling: ArrayLike,
        /,
        *,
        symmetry_tolerance: float = 1e-10,
    ) -> SpatialPiezoelectricSystem:
        mechanical = np.asarray(mechanical_stiffness, dtype=float)
        dielectric = np.asarray(dielectric_stiffness, dtype=float)
        coupling = np.asarray(electromechanical_coupling, dtype=float)
        if mechanical.ndim != 2 or mechanical.shape[0] != mechanical.shape[1]:
            raise ValueError("Piezoelectric mechanical stiffness must be square.")
        if dielectric.ndim != 2 or dielectric.shape[0] != dielectric.shape[1]:
            raise ValueError("Piezoelectric dielectric stiffness must be square.")
        if coupling.shape != (mechanical.shape[0], dielectric.shape[0]):
            raise ValueError("Piezoelectric coupling does not match field coordinates.")
        if not np.allclose(mechanical, mechanical.T, atol=symmetry_tolerance, rtol=0):
            raise ValueError("Piezoelectric mechanical stiffness must be symmetric.")
        if not np.allclose(dielectric, dielectric.T, atol=symmetry_tolerance, rtol=0):
            raise ValueError("Piezoelectric dielectric stiffness must be symmetric.")
        if (
            np.min(np.linalg.eigvalsh(mechanical)) <= 0
            or np.min(np.linalg.eigvalsh(dielectric)) <= 0
        ):
            raise ValueError(
                "Piezoelectric diagonal energy blocks must be positive definite."
            )
        return cls(
            jnp.asarray(mechanical), jnp.asarray(dielectric), jnp.asarray(coupling)
        )

    def block_matrix(self, /) -> Array:
        return jnp.block(
            [
                [self.mechanical_stiffness, -self.electromechanical_coupling],
                [
                    -self.electromechanical_coupling.T,
                    -self.dielectric_stiffness,
                ],
            ]
        )

    def solve(
        self, mechanical_load_n: ArrayLike, free_charge_c: ArrayLike, /
    ) -> PiezoelectricSolveResult:
        force = jnp.asarray(mechanical_load_n)
        charge = jnp.asarray(free_charge_c)
        if force.shape != (self.mechanical_stiffness.shape[0],):
            raise ValueError("Piezoelectric mechanical load has incompatible shape.")
        if charge.shape != (self.dielectric_stiffness.shape[0],):
            raise ValueError("Piezoelectric free charge has incompatible shape.")
        matrix = self.block_matrix()
        right = jnp.concatenate((force, charge)).astype(matrix.dtype)
        result = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        mechanical_size = force.size
        displacement = result.value[:mechanical_size]
        potential = result.value[mechanical_size:]
        residual = matrix @ result.value - right
        residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(residual), residual))
        )
        enthalpy = (
            0.5
            * contract("i,ij,j->", displacement, self.mechanical_stiffness, displacement)
            - contract(
                "i,ij,j->", displacement, self.electromechanical_coupling, potential
            )
            - 0.5 * contract("i,ij,j->", potential, self.dielectric_stiffness, potential)
        )
        return PiezoelectricSolveResult(
            displacement,
            potential,
            residual_norm,
            enthalpy,
            result.successful & jnp.isfinite(enthalpy),
        )


__all__ = ["PiezoelectricSolveResult", "SpatialPiezoelectricSystem"]
