#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""MECI/MECP branching-space evidence and minimum-energy crossing optimization."""

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy


class CrossingKind(StrEnum):
    MECI = "meci"
    MECP = "mecp"


class TwoStateSurfaceEvaluation(StrictModule, NonTrainableState):
    energies: Array
    gradients: Array
    energy_weighted_coupling: Array
    successful: Array
    state_ids: tuple[str, str] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies: ArrayLike,
        gradients: ArrayLike,
        energy_weighted_coupling: ArrayLike,
        successful: ArrayLike,
        state_ids: tuple[str, str],
        provider_id: str,
        /,
    ):
        energy = jnp.asarray(energies)
        gradient = jnp.asarray(gradients, dtype=energy.dtype)
        coupling = jnp.asarray(energy_weighted_coupling, dtype=energy.dtype)
        states = tuple(str(value).strip() for value in state_ids)
        provider = str(provider_id).strip()
        if (
            energy.shape != (2,)
            or gradient.ndim != 3
            or gradient.shape[0] != 2
            or coupling.shape != gradient.shape[1:]
        ):
            raise ValueError("Two-state energies, gradients, and coupling do not align.")
        if (
            len(states) != 2
            or any(not value for value in states)
            or states[0] == states[1]
            or not provider
        ):
            raise ValueError(
                "Two-state identities and provider must be distinct and non-empty."
            )
        self.energies = energy
        self.gradients = gradient
        self.energy_weighted_coupling = coupling
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.state_ids = states
        self.provider_id = provider
        self.result_id = canonical_fingerprint(
            {
                "kind": "two-state-surface-evaluation",
                "states": list(states),
                "provider": provider,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energies": np.asarray(energy),
                        "gradients": np.asarray(gradient),
                        "energy_weighted_coupling": np.asarray(coupling),
                    }
                ),
            }
        )


class AbstractTwoStateSurfaceProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(self, positions: ArrayLike, /) -> TwoStateSurfaceEvaluation:
        raise NotImplementedError


TwoStateEvaluator = Callable[[ArrayLike], TwoStateSurfaceEvaluation]


class CallableTwoStateSurfaceProvider(AbstractTwoStateSurfaceProvider):
    evaluator: TwoStateEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: TwoStateEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(self, positions: ArrayLike, /) -> TwoStateSurfaceEvaluation:
        result = self.evaluator(positions)
        if (
            not isinstance(result, TwoStateSurfaceEvaluation)
            or result.provider_id != self.provider_id
        ):
            raise ValueError("Two-state provider changed result type or identity.")
        return result


class BranchingPlaneResult(StrictModule):
    gradient_difference: Array
    energy_weighted_coupling: Array
    orthonormal_basis: Array
    gram_eigenvalues: Array
    rank: Array
    orthogonality_residual: Array
    successful: Array


def branching_plane(
    evaluation: TwoStateSurfaceEvaluation,
    /,
    *,
    rank_tolerance: float = 1.0e-10,
) -> BranchingPlaneResult:
    if not isinstance(evaluation, TwoStateSurfaceEvaluation):
        raise TypeError("evaluation must be TwoStateSurfaceEvaluation.")
    g = 0.5 * (evaluation.gradients[1] - evaluation.gradients[0])
    h = evaluation.energy_weighted_coupling
    vectors = jnp.stack((g.reshape((-1,)), h.reshape((-1,))), axis=1)
    gram = jnp.real(jnp.conj(vectors.T) @ vectors)
    spectrum = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                gram,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
            )
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=2, which="smallest-algebraic"),
    )
    eigenvalues, eigenvectors = spectrum.eigenvalues, spectrum.eigenvectors
    retained = eigenvalues > float(rank_tolerance)
    safe = jnp.where(retained, eigenvalues, 1.0)
    basis = vectors @ (eigenvectors / jnp.sqrt(safe)[None, :])
    basis = jnp.where(retained[None, :], basis, 0.0)
    residual = jnp.max(
        jnp.abs(jnp.conj(basis.T) @ basis - jnp.diag(retained.astype(basis.real.dtype))),
        initial=0.0,
    )
    rank = jnp.sum(retained.astype(jnp.int32))
    successful = (
        evaluation.successful & spectrum.successful & (rank > 0) & jnp.isfinite(residual)
    )
    return BranchingPlaneResult(g, h, basis, eigenvalues, rank, residual, successful)


class CrossingOptimizationResult(StrictModule, NonTrainableState):
    positions: Array
    energies: Array
    energy_gap: Array
    projected_gradient_norm: Array
    branching: BranchingPlaneResult
    trajectory: Array
    iterations: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions,
        energies,
        projected_gradient_norm,
        branching,
        trajectory,
        iterations,
        successful,
        plan_id,
        provider_id,
        /,
    ):
        position = jnp.asarray(positions)
        energy = jnp.asarray(energies, dtype=position.dtype)
        trajectory_ = jnp.asarray(trajectory, dtype=position.dtype)
        self.positions = position
        self.energies = energy
        self.energy_gap = jnp.abs(energy[1] - energy[0])
        self.projected_gradient_norm = jnp.asarray(
            projected_gradient_norm, dtype=position.dtype
        ).reshape(())
        self.branching = branching
        self.trajectory = trajectory_
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.plan_id = str(plan_id)
        self.provider_id = str(provider_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "crossing-optimization-result",
                "plan": self.plan_id,
                "provider": self.provider_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "positions": np.asarray(position),
                        "energies": np.asarray(energy),
                        "projected_gradient_norm": np.asarray(
                            self.projected_gradient_norm
                        ),
                        "trajectory": np.asarray(trajectory_),
                    }
                ),
            }
        )


class MinimumEnergyCrossingPlan(StrictModule, NonTrainableState):
    kind: CrossingKind = eqx.field(static=True)
    provider: AbstractTwoStateSurfaceProvider
    gap_tolerance: float = eqx.field(static=True)
    gradient_tolerance: float = eqx.field(static=True)
    geometry_step: float = eqx.field(static=True)
    gap_step: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CrossingKind,
        provider: AbstractTwoStateSurfaceProvider,
        /,
        *,
        gap_tolerance: float = 1.0e-5,
        gradient_tolerance: float = 1.0e-4,
        geometry_step: float = 0.05,
        gap_step: float = 0.5,
        maximum_iterations: int = 200,
    ):
        if not isinstance(kind, CrossingKind) or not isinstance(
            provider, AbstractTwoStateSurfaceProvider
        ):
            raise TypeError("Crossing optimization requires typed kind and provider.")
        values = tuple(
            float(value)
            for value in (gap_tolerance, gradient_tolerance, geometry_step, gap_step)
        )
        iterations = int(maximum_iterations)
        if (
            any(not isfinite(value) or value <= 0.0 for value in values)
            or iterations <= 0
        ):
            raise ValueError("Crossing tolerances, steps, or work limit are invalid.")
        self.kind = kind
        self.provider = provider
        self.gap_tolerance, self.gradient_tolerance, self.geometry_step, self.gap_step = (
            values
        )
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "minimum-energy-crossing-plan",
                "crossing_kind": kind.value,
                "provider": provider.provider_id,
                "gap_tolerance": values[0],
                "gradient_tolerance": values[1],
                "geometry_step": values[2],
                "gap_step": values[3],
                "maximum_iterations": iterations,
            }
        )

    def run(self, initial_positions: ArrayLike, /) -> CrossingOptimizationResult:
        positions = jnp.asarray(initial_positions)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Crossing geometry must have shape (N, 3).")
        trajectory = [positions]
        converged = False
        projected_norm = jnp.asarray(jnp.inf, dtype=positions.dtype)
        final = self.provider.evaluate(positions)
        final_branching = branching_plane(final)
        completed = 0
        for iteration in range(self.maximum_iterations):
            evaluation = self.provider.evaluate(positions)
            branching = branching_plane(evaluation)
            average_gradient = 0.5 * (
                evaluation.gradients[0] + evaluation.gradients[1]
            ).reshape((-1,))
            basis = branching.orthonormal_basis
            if self.kind is CrossingKind.MECP:
                basis = basis[:, :1]
            projected = average_gradient - basis @ (jnp.conj(basis.T) @ average_gradient)
            projected_norm = jnp.sqrt(jnp.real(jnp.vdot(projected, projected)))
            gap = evaluation.energies[1] - evaluation.energies[0]
            g = branching.gradient_difference.reshape((-1,))
            g_norm_squared = jnp.real(jnp.vdot(g, g))
            successful = (
                evaluation.successful & branching.successful & (g_norm_squared > 0.0)
            )
            if bool(
                successful
                & (jnp.abs(gap) <= self.gap_tolerance)
                & (projected_norm <= self.gradient_tolerance)
            ):
                converged = True
                final = evaluation
                final_branching = branching
                completed = iteration
                break
            step = (
                -self.geometry_step * projected - self.gap_step * gap * g / g_norm_squared
            ).reshape(positions.shape)
            positions = positions + step
            trajectory.append(positions)
            final = evaluation
            final_branching = branching
            completed = iteration + 1
        final = self.provider.evaluate(positions)
        final_branching = branching_plane(final)
        final_average_gradient = 0.5 * (final.gradients[0] + final.gradients[1]).reshape(
            (-1,)
        )
        final_basis = final_branching.orthonormal_basis
        if self.kind is CrossingKind.MECP:
            final_basis = final_basis[:, :1]
        final_projected = final_average_gradient - final_basis @ (
            jnp.conj(final_basis.T) @ final_average_gradient
        )
        projected_norm = jnp.sqrt(jnp.real(jnp.vdot(final_projected, final_projected)))
        final_gap = jnp.abs(final.energies[1] - final.energies[0])
        converged = converged or bool(
            final.successful
            & final_branching.successful
            & (final_gap <= self.gap_tolerance)
            & (projected_norm <= self.gradient_tolerance)
        )
        return CrossingOptimizationResult(
            positions,
            final.energies,
            projected_norm,
            final_branching,
            jnp.stack(tuple(trajectory)),
            completed,
            converged & bool(final.successful),
            self.plan_id,
            self.provider.provider_id,
        )


__all__ = [
    "AbstractTwoStateSurfaceProvider",
    "BranchingPlaneResult",
    "CallableTwoStateSurfaceProvider",
    "CrossingKind",
    "CrossingOptimizationResult",
    "MinimumEnergyCrossingPlan",
    "TwoStateSurfaceEvaluation",
    "branching_plane",
]
