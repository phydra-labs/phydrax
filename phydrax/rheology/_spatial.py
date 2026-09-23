#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Spatial transport of viscoelastic conformation tensors."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._core import ViscoelasticLaw


@dataclass(frozen=True, slots=True)
class SpatialConformationStep:
    conformation: Array
    candidate_conformation: Array
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
    conservation_tolerance: float = 1e-10

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
        if not isinstance(law, ViscoelasticLaw):
            raise TypeError("Rheology law must be ViscoelasticLaw.")
        if (
            not isfinite(eigenvalue_floor)
            or eigenvalue_floor <= 0
            or not isfinite(conservation_tolerance)
            or conservation_tolerance <= 0
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(generator))
        ):
            raise ValueError("Rheology stabilization/conservation data are invalid.")
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
        return cls(
            jnp.asarray(weights),
            jnp.asarray(generator),
            law,
            eigenvalue_floor,
            conservation_tolerance,
        )

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
        if not isfinite(step_size_s) or step_size_s <= 0:
            raise ValueError("Rheology step size must be finite and positive.")
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | ~jnp.isfinite(gradient)),
            "Conformation and velocity gradients must be finite.",
        )
        value = self.law._validated_conformation(value)
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
        eigenvalues = jnp.linalg.eigvalsh(transported)
        minimum = jnp.min(eigenvalues)
        transport_balance = contract(
            "q,qij->ij", self.measure_weights, transported - right
        )
        correction_norm = jnp.asarray(0.0, dtype=value.dtype)
        balance_norm = jnp.linalg.norm(transport_balance)
        successful = (
            jnp.all(jnp.stack(native_success))
            & jnp.all(jnp.isfinite(transported))
            & (minimum >= self.eigenvalue_floor)
            & jnp.isfinite(balance_norm)
            & (balance_norm <= self.conservation_tolerance)
        )
        accepted = jnp.where(successful, transported, value)
        return SpatialConformationStep(
            accepted,
            transported,
            self.law.stress(accepted),
            minimum,
            transport_balance,
            correction_norm,
            successful,
        )


__all__ = ["SpatialConformationSolver", "SpatialConformationStep"]
