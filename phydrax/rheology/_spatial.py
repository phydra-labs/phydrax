#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Spatial transport of viscoelastic conformation tensors."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._core import ViscoelasticLaw


@dataclass(frozen=True, slots=True)
class SpatialConformationStep:
    conformation: Array
    polymer_stress_pa: Array
    minimum_eigenvalue: Array
    transport_balance_residual: Array
    stabilization_correction_norm: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SpatialConformationSolver:
    """Implicit conservative transport plus objective constitutive evolution."""

    measure_weights: Array
    transport_generator_s_inv: Array
    law: ViscoelasticLaw
    eigenvalue_floor: float = 1e-10

    @classmethod
    def create(
        cls,
        measure_weights: ArrayLike,
        transport_generator_s_inv: ArrayLike,
        law: ViscoelasticLaw,
        /,
        *,
        eigenvalue_floor: float = 1e-10,
        conservation_tolerance: float = 1e-10,
    ) -> SpatialConformationSolver:
        weights = np.asarray(measure_weights, dtype=np.float64)
        generator = np.asarray(transport_generator_s_inv, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0 or np.any(weights <= 0):
            raise ValueError("Rheology measures must be a positive vector.")
        if generator.shape != (weights.size, weights.size):
            raise ValueError(
                "Rheology transport generator must be square on spatial cells."
            )
        if not np.allclose(
            weights @ generator,
            0,
            atol=conservation_tolerance,
            rtol=conservation_tolerance,
        ):
            raise ValueError(
                "Rheology transport generator must conserve weighted content."
            )
        if eigenvalue_floor <= 0:
            raise ValueError("Conformation eigenvalue floor must be positive.")
        return cls(jnp.asarray(weights), jnp.asarray(generator), law, eigenvalue_floor)

    def advance(
        self,
        conformation: ArrayLike,
        velocity_gradient_s_inv: ArrayLike,
        step_size_s: float,
        /,
    ) -> SpatialConformationStep:
        value = jnp.asarray(conformation)
        gradient = jnp.asarray(velocity_gradient_s_inv)
        if (
            value.ndim != 3
            or value.shape[0] != self.measure_weights.size
            or value.shape[-1] != value.shape[-2]
            or gradient.shape != value.shape
        ):
            raise ValueError("Spatial conformation and velocity gradients must align.")
        if step_size_s <= 0:
            raise ValueError("Rheology step size must be positive.")
        symmetric = 0.5 * (value + jnp.swapaxes(value, -1, -2))
        local_rate = (
            contract("qik,qkj->qij", gradient, symmetric)
            + contract("qik,qjk->qij", symmetric, gradient)
            + self.law.relaxation(symmetric)
        )
        right = symmetric + float(step_size_s) * local_rate
        spatial_matrix = jnp.eye(value.shape[0], dtype=value.dtype) - float(
            step_size_s
        ) * self.transport_generator_s_inv.astype(value.dtype)
        components = []
        native_success = []
        for row in range(value.shape[-2]):
            columns = []
            for column in range(value.shape[-1]):
                result = solve(
                    LinearSystem(DenseLinearOperator(spatial_matrix)),
                    right[:, row, column],
                    policy=LinearSolvePolicy(DenseLU()),
                )
                columns.append(result.value)
                native_success.append(result.successful)
            components.append(jnp.stack(columns, axis=-1))
        transported = jnp.stack(components, axis=-2)
        transported = 0.5 * (transported + jnp.swapaxes(transported, -1, -2))
        eigenvalues, eigenvectors = jnp.linalg.eigh(transported)
        bounded = jnp.maximum(eigenvalues, self.eigenvalue_floor)
        stabilized = contract("qik,qk,qjk->qij", eigenvectors, bounded, eigenvectors)
        correction = stabilized - transported
        transport_balance = contract(
            "q,qij->ij", self.measure_weights, transported - right
        )
        correction_norm = jnp.sqrt(
            jnp.real(contract("qij,qij->", jnp.conj(correction), correction))
        )
        minimum = jnp.min(jnp.linalg.eigvalsh(stabilized))
        successful = (
            jnp.all(jnp.stack(native_success))
            & jnp.all(jnp.isfinite(stabilized))
            & (minimum >= 0)
        )
        return SpatialConformationStep(
            stabilized,
            self.law.stress(stabilized),
            minimum,
            transport_balance,
            correction_norm,
            successful,
        )


__all__ = ["SpatialConformationSolver", "SpatialConformationStep"]
