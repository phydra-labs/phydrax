#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Modal and transient execution for linear structural systems."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..frequency import CompiledFrequencySystem, FrequencyDomainResult
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ..linalg.eigen import (
    DenseEigh,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)


@dataclass(frozen=True, slots=True)
class ModalAnalysisResult:
    eigenvalues_rad2_s2: Array
    angular_frequencies_rad_s: Array
    modes: Array
    eigen_residual_norms: Array
    mass_orthogonality_error: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class StructuralDynamicState:
    displacement: Array
    velocity: Array
    acceleration: Array


@dataclass(frozen=True, slots=True)
class StructuralDynamicStep:
    state: StructuralDynamicState
    equilibrium_residual_norm: Array
    mechanical_energy_j: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class LinearStructuralSystem:
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
        *,
        symmetry_tolerance: float = 1e-10,
    ) -> LinearStructuralSystem:
        matrices = tuple(
            np.asarray(value, dtype=np.float64) for value in (mass, damping, stiffness)
        )
        if matrices[0].ndim != 2 or matrices[0].shape[0] != matrices[0].shape[1]:
            raise ValueError("Structural matrices must be square.")
        if any(value.shape != matrices[0].shape for value in matrices[1:]):
            raise ValueError("Structural matrices must share one coordinate basis.")
        if any(
            not np.allclose(value, value.T, atol=symmetry_tolerance, rtol=0)
            for value in matrices
        ):
            raise ValueError("Structural mass, damping, and stiffness must be symmetric.")
        if np.min(np.linalg.eigvalsh(matrices[0])) <= 0:
            raise ValueError("Structural mass must be positive definite.")
        return cls(*(jnp.asarray(value) for value in matrices))

    @property
    def size(self) -> int:
        return self.mass.shape[0]

    def modal_analysis(self, count: int | None = None, /) -> ModalAnalysisResult:
        requested = self.size if count is None else min(int(count), self.size)
        if requested <= 0:
            raise ValueError("At least one structural mode must be requested.")
        self_adjoint = OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        )
        positive_mass = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "verified",
            },
        )
        result = eigensolve(
            GeneralizedEigenproblem(
                DenseLinearOperator(self.stiffness, properties=self_adjoint),
                DenseLinearOperator(self.mass, properties=positive_mass),
            ),
            policy=EigenSolvePolicy(
                DenseEigh(), count=requested, which="smallest-algebraic"
            ),
        )
        modes = result.eigenvectors
        gram = jnp.conj(modes).T @ self.mass @ modes
        diagonal = jnp.real(jnp.diag(gram))
        modes = (
            modes
            / jnp.sqrt(jnp.maximum(diagonal, jnp.finfo(diagonal.dtype).tiny))[None, :]
        )
        residual = (
            self.stiffness @ modes - (self.mass @ modes) * result.eigenvalues[None, :]
        )
        residual_norms = jnp.sqrt(
            jnp.real(contract("im,im->m", jnp.conj(residual), residual))
        )
        orthogonality = jnp.conj(modes).T @ self.mass @ modes - jnp.eye(requested)
        orthogonality_error = jnp.sqrt(
            jnp.real(contract("ij,ij->", jnp.conj(orthogonality), orthogonality))
        )
        nonnegative = jnp.all(result.eigenvalues >= -1e-12)
        frequencies = jnp.sqrt(jnp.maximum(result.eigenvalues, 0))
        return ModalAnalysisResult(
            result.eigenvalues,
            frequencies,
            modes,
            residual_norms,
            orthogonality_error,
            result.successful & nonnegative,
        )

    def harmonic_response(
        self, angular_frequency_rad_s: ArrayLike, force: ArrayLike, /
    ) -> FrequencyDomainResult:
        return CompiledFrequencySystem.create(
            self.mass, self.damping, self.stiffness
        ).solve(angular_frequency_rad_s, force)

    def newmark_step(
        self,
        state: StructuralDynamicState,
        force: ArrayLike,
        step_size_s: float,
        /,
        *,
        beta: float = 0.25,
        gamma: float = 0.5,
        relative_tolerance: float = 1e-9,
    ) -> StructuralDynamicStep:
        if step_size_s <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("Newmark integration controls must be positive.")
        displacement = jnp.asarray(state.displacement)
        velocity = jnp.asarray(state.velocity)
        acceleration = jnp.asarray(state.acceleration)
        load = jnp.asarray(force)
        if any(
            value.shape != (self.size,)
            for value in (displacement, velocity, acceleration, load)
        ):
            raise ValueError("Structural state and load must match the coordinate basis.")
        dt = float(step_size_s)
        predicted_displacement = (
            displacement + dt * velocity + dt**2 * (0.5 - beta) * acceleration
        )
        predicted_velocity = velocity + dt * (1.0 - gamma) * acceleration
        effective = self.mass + gamma * dt * self.damping + beta * dt**2 * self.stiffness
        right = (
            load
            - self.damping @ predicted_velocity
            - self.stiffness @ predicted_displacement
        )
        acceleration_new = solve(
            LinearSystem(DenseLinearOperator(effective)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        displacement_new = predicted_displacement + beta * dt**2 * acceleration_new.value
        velocity_new = predicted_velocity + gamma * dt * acceleration_new.value
        residual = (
            self.mass @ acceleration_new.value
            + self.damping @ velocity_new
            + self.stiffness @ displacement_new
            - load
        )
        residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(residual), residual))
        )
        load_norm = jnp.sqrt(jnp.real(contract("i,i->", jnp.conj(load), load)))
        energy = 0.5 * (
            contract("i,ij,j->", velocity_new, self.mass, velocity_new)
            + contract("i,ij,j->", displacement_new, self.stiffness, displacement_new)
        )
        successful = acceleration_new.successful & (
            residual_norm <= float(relative_tolerance) * jnp.maximum(load_norm, 1.0)
        )
        return StructuralDynamicStep(
            StructuralDynamicState(
                displacement_new, velocity_new, acceleration_new.value
            ),
            residual_norm,
            energy,
            successful,
        )


__all__ = [
    "LinearStructuralSystem",
    "ModalAnalysisResult",
    "StructuralDynamicState",
    "StructuralDynamicStep",
]
