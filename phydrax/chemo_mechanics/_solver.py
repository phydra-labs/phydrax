#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Monolithic conservative diffusion–mechanics coupling."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


@dataclass(frozen=True, slots=True)
class ChemoMechanicalStep:
    displacement: Array
    concentration: Array
    chemical_potential_j_mol: Array
    mass_balance_residual: Array
    dissipation_rate: Array
    residual_norm: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SpatialChemoMechanicalSystem:
    """Linear reciprocal free energy with conservative Onsager transport."""

    mechanical_stiffness: Array
    chemical_hessian: Array
    chemical_expansion_coupling: Array
    mobility_laplacian: Array
    storage_weights: Array

    @classmethod
    def create(
        cls,
        mechanical_stiffness: ArrayLike,
        chemical_hessian: ArrayLike,
        chemical_expansion_coupling: ArrayLike,
        mobility_laplacian: ArrayLike,
        storage_weights: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> SpatialChemoMechanicalSystem:
        mechanical = np.asarray(mechanical_stiffness, dtype=float)
        chemical = np.asarray(chemical_hessian, dtype=float)
        coupling = np.asarray(chemical_expansion_coupling, dtype=float)
        mobility = np.asarray(mobility_laplacian, dtype=float)
        weights = np.asarray(storage_weights, dtype=float)
        if mechanical.ndim != 2 or mechanical.shape[0] != mechanical.shape[1]:
            raise ValueError("Chemo-mechanical stiffness must be square.")
        if chemical.ndim != 2 or chemical.shape[0] != chemical.shape[1]:
            raise ValueError("Chemical Hessian must be square.")
        if coupling.shape != (mechanical.shape[0], chemical.shape[0]):
            raise ValueError("Chemical expansion coupling has incompatible shape.")
        if mobility.shape != chemical.shape or weights.shape != (chemical.shape[0],):
            raise ValueError("Chemical transport operators have incompatible shapes.")
        if np.any(weights <= 0):
            raise ValueError("Chemical storage weights must be positive.")
        if any(
            not np.allclose(value, value.T, atol=tolerance, rtol=0)
            for value in (mechanical, chemical, mobility)
        ):
            raise ValueError(
                "Chemo-mechanical energy and mobility blocks must be symmetric."
            )
        if (
            np.min(np.linalg.eigvalsh(mechanical)) <= 0
            or np.min(np.linalg.eigvalsh(chemical)) <= 0
        ):
            raise ValueError(
                "Chemo-mechanical free-energy diagonal blocks must be positive."
            )
        if np.min(np.linalg.eigvalsh(mobility)) < -tolerance:
            raise ValueError("Chemical mobility Laplacian must be positive semidefinite.")
        if not np.allclose(mobility.sum(axis=0), 0, atol=tolerance, rtol=0):
            raise ValueError("Chemical mobility Laplacian must conserve total species.")
        return cls(
            jnp.asarray(mechanical),
            jnp.asarray(chemical),
            jnp.asarray(coupling),
            jnp.asarray(mobility),
            jnp.asarray(weights),
        )

    def advance(
        self,
        concentration: ArrayLike,
        mechanical_load: ArrayLike,
        chemical_source_mol_s: ArrayLike,
        step_size_s: float,
        /,
    ) -> ChemoMechanicalStep:
        old = jnp.asarray(concentration)
        force = jnp.asarray(mechanical_load)
        source = jnp.asarray(chemical_source_mol_s)
        mechanical_size = self.mechanical_stiffness.shape[0]
        chemical_size = self.chemical_hessian.shape[0]
        if old.shape != (chemical_size,) or source.shape != (chemical_size,):
            raise ValueError("Chemical state and source have incompatible shapes.")
        if force.shape != (mechanical_size,) or step_size_s <= 0:
            raise ValueError("Mechanical load or chemo-mechanical step is invalid.")
        storage = jnp.diag(self.storage_weights / float(step_size_s))
        transport_hessian = self.mobility_laplacian @ self.chemical_hessian
        matrix = jnp.block(
            [
                [self.mechanical_stiffness, -self.chemical_expansion_coupling],
                [
                    -self.mobility_laplacian @ self.chemical_expansion_coupling.T,
                    storage + transport_hessian,
                ],
            ]
        )
        right = jnp.concatenate(
            (force, self.storage_weights / float(step_size_s) * old + source)
        ).astype(matrix.dtype)
        result = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        displacement = result.value[:mechanical_size]
        next_concentration = result.value[mechanical_size:]
        potential = (
            self.chemical_hessian @ next_concentration
            - self.chemical_expansion_coupling.T @ displacement
        )
        mass_residual = contract(
            "i,i->", self.storage_weights, next_concentration - old
        ) - float(step_size_s) * jnp.sum(source)
        dissipation = contract("i,ij,j->", potential, self.mobility_laplacian, potential)
        residual = matrix @ result.value - right
        residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(residual), residual))
        )
        successful = (
            result.successful
            & jnp.all(next_concentration >= -1e-12)
            & (dissipation >= -1e-12)
        )
        return ChemoMechanicalStep(
            displacement,
            next_concentration,
            potential,
            mass_residual,
            dissipation,
            residual_norm,
            successful,
        )


__all__ = ["ChemoMechanicalStep", "SpatialChemoMechanicalSystem"]
