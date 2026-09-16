#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generalized periodic eigenspaces and bounded Chebyshev spectral moments."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._reciprocal import ReciprocalMeshPlan, ReciprocalPathPlan
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    DenseEigh,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)
from ...operators.periodic._family import PeriodicResourceError
from ...units import UnitDefinition
from ._orbital_model import PreparedPeriodicOrbitalPencil


ReciprocalSupport = ReciprocalMeshPlan | ReciprocalPathPlan


class PeriodicSpectrumResult(StrictModule, NonTrainableState):
    fractional_points: Array
    weights: Array
    distances: Array
    energies: Array
    coefficients: Array
    eigen_residuals: Array
    metric_residuals: Array
    overlap_minimum_eigenvalues: Array
    overlap_condition_numbers: Array
    successful: Array
    energy_unit: UnitDefinition
    cell_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    pencil_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        fractional_points,
        weights,
        distances,
        energies,
        coefficients,
        eigen_residuals,
        metric_residuals,
        overlap_minimum_eigenvalues,
        overlap_condition_numbers,
        successful,
        energy_unit,
        /,
        *,
        cell_id: str,
        basis_id: str,
        support_id: str,
        pencil_id: str,
    ):
        points = jnp.asarray(fractional_points)
        weight = jnp.asarray(weights, dtype=points.dtype)
        distance = jnp.asarray(distances, dtype=points.dtype)
        energy = jnp.asarray(energies)
        coefficient = jnp.asarray(coefficients)
        eigen = jnp.asarray(eigen_residuals, dtype=energy.real.dtype)
        metric = jnp.asarray(metric_residuals, dtype=energy.real.dtype)
        if (
            points.ndim != 2
            or weight.shape != (points.shape[0],)
            or distance.shape != (points.shape[0],)
            or energy.ndim != 2
            or energy.shape[0] != points.shape[0]
            or coefficient.shape[:1] != (points.shape[0],)
            or coefficient.shape[2] != energy.shape[1]
            or eigen.shape != energy.shape
            or metric.shape != (points.shape[0],)
        ):
            raise ValueError("Periodic spectrum arrays do not align.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        self.fractional_points = points
        self.weights = weight
        self.distances = distance
        self.energies = energy
        self.coefficients = coefficient
        self.eigen_residuals = eigen
        self.metric_residuals = metric
        self.overlap_minimum_eigenvalues = jnp.asarray(overlap_minimum_eigenvalues)
        self.overlap_condition_numbers = jnp.asarray(overlap_condition_numbers)
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.energy_unit = energy_unit
        self.cell_id = str(cell_id)
        self.basis_id = str(basis_id)
        self.support_id = str(support_id)
        self.pencil_id = str(pencil_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-spectrum-result",
                "cell": self.cell_id,
                "basis": self.basis_id,
                "support": self.support_id,
                "pencil": self.pencil_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "points": np.asarray(points),
                        "weights": np.asarray(weight),
                        "distances": np.asarray(distance),
                        "energies": np.asarray(energy),
                        "coefficients": np.asarray(coefficient),
                        "eigen_residuals": np.asarray(eigen),
                        "metric_residuals": np.asarray(metric),
                    }
                ),
            }
        )


class PeriodicSpectrumPlan(StrictModule, NonTrainableState):
    """Dense bounded generalized H C = S C epsilon solve on one support."""

    pencil: PreparedPeriodicOrbitalPencil
    support: ReciprocalSupport
    band_count: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        support: ReciprocalSupport,
        /,
        *,
        band_count: int | None = None,
        residual_tolerance: float = 1.0e-9,
        maximum_eigenpairs: int = 1_000_000,
    ):
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil):
            raise TypeError("pencil must be PreparedPeriodicOrbitalPencil.")
        if not isinstance(support, (ReciprocalMeshPlan, ReciprocalPathPlan)):
            raise TypeError("support must be ReciprocalMeshPlan or ReciprocalPathPlan.")
        support.require_cell(pencil.plan.basis.cell)
        orbitals = pencil.plan.basis.orbital_count
        count = orbitals if band_count is None else int(band_count)
        tolerance = float(residual_tolerance)
        if count <= 0 or count > orbitals:
            raise ValueError("band_count must lie within the orbital dimension.")
        if int(support.fractional_points.shape[0]) * count > int(maximum_eigenpairs):
            raise PeriodicResourceError("Periodic eigensolve exceeds maximum_eigenpairs.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive finite.")
        self.pencil = pencil
        self.support = support
        self.band_count = count
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-spectrum-plan",
                "pencil": pencil.prepared_id,
                "support": support.mesh_id
                if isinstance(support, ReciprocalMeshPlan)
                else support.path_id,
                "band_count": count,
                "residual_tolerance": tolerance,
            }
        )

    def evaluate(self, /) -> PeriodicSpectrumResult:
        evaluation = self.pencil.evaluate(self.support.fractional_points)
        if not bool(evaluation.successful):
            raise ValueError(
                "Periodic overlap pencil is not positive definite within policy."
            )
        properties = OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        )
        metric_properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        problem = GeneralizedEigenproblem(
            DenseLinearOperator(evaluation.hamiltonians, properties=properties),
            DenseLinearOperator(evaluation.overlaps, properties=metric_properties),
            problem_id=self.plan_id,
        )
        solved = eigensolve(
            problem,
            policy=EigenSolvePolicy(
                DenseEigh(),
                count=self.band_count,
                which="smallest-algebraic",
            ),
        )
        values = solved.eigenvalues.real
        vectors = solved.eigenvectors
        h_action = evaluation.hamiltonians @ vectors
        s_action = evaluation.overlaps @ vectors
        residual = jnp.sqrt(
            jnp.sum(jnp.abs(h_action - s_action * values[:, None, :]) ** 2, axis=1)
        )
        metric_gram = jnp.conj(jnp.swapaxes(vectors, -1, -2)) @ s_action
        metric_residual = jnp.max(
            jnp.abs(
                metric_gram
                - jnp.eye(self.band_count, dtype=metric_gram.dtype)[None, :, :]
            ),
            axis=(-2, -1),
        )
        successful = (
            jnp.all(solved.successful)
            & jnp.all(residual <= self.residual_tolerance)
            & jnp.all(metric_residual <= self.residual_tolerance)
        )
        if isinstance(self.support, ReciprocalMeshPlan):
            weights = self.support.weights
            distances = jnp.zeros_like(weights)
            support_id = self.support.mesh_id
        else:
            weights = jnp.full(
                (self.support.fractional_points.shape[0],),
                1.0 / self.support.fractional_points.shape[0],
                dtype=self.support.fractional_points.dtype,
            )
            distances = self.support.distances
            support_id = self.support.path_id
        return PeriodicSpectrumResult(
            self.support.fractional_points,
            weights,
            distances,
            values,
            vectors,
            residual,
            metric_residual,
            evaluation.overlap_minimum_eigenvalues,
            evaluation.overlap_condition_numbers,
            successful,
            self.pencil.plan.energy_unit,
            cell_id=self.pencil.plan.basis.cell_id,
            basis_id=self.pencil.plan.basis.basis_id,
            support_id=support_id,
            pencil_id=self.pencil.plan.pencil_id,
        )


class ChebyshevMomentResult(StrictModule, NonTrainableState):
    moments: Array
    lower_bound: Array
    upper_bound: Array
    recurrence_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ChebyshevMomentPlan(StrictModule, NonTrainableState):
    """Candidate trace moments using sparse family actions, never a dense fallback."""

    pencil: PreparedPeriodicOrbitalPencil
    mesh: ReciprocalMeshPlan
    order: int = eqx.field(static=True)
    lower_bound: float = eqx.field(static=True)
    upper_bound: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        mesh: ReciprocalMeshPlan,
        order: int,
        lower_bound: float,
        upper_bound: float,
        /,
        *,
        maximum_operator_applications: int = 1_000_000,
    ):
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil) or not isinstance(
            mesh, ReciprocalMeshPlan
        ):
            raise TypeError(
                "Chebyshev moments require a prepared pencil and reciprocal mesh."
            )
        mesh.require_cell(pencil.plan.basis.cell)
        order_ = int(order)
        lower = float(lower_bound)
        upper = float(upper_bound)
        applications = (
            int(mesh.fractional_points.shape[0])
            * pencil.plan.basis.orbital_count
            * max(order_ - 1, 0)
        )
        if order_ < 2 or not isfinite(lower) or not isfinite(upper) or upper <= lower:
            raise ValueError("Chebyshev order and spectral bounds are invalid.")
        if applications > int(maximum_operator_applications):
            raise PeriodicResourceError(
                "Chebyshev recurrence exceeds maximum_operator_applications."
            )
        overlap_evaluation = pencil.evaluate(mesh.fractional_points)
        overlap_matrices = np.asarray(overlap_evaluation.overlaps)
        identity = np.broadcast_to(
            np.eye(pencil.plan.basis.orbital_count, dtype=overlap_matrices.dtype),
            overlap_matrices.shape,
        )
        if not bool(overlap_evaluation.successful) or not np.allclose(
            overlap_matrices, identity, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError(
                "Chebyshev moment candidate currently requires orthonormal S=I."
            )
        self.pencil = pencil
        self.mesh = mesh
        self.order = order_
        self.lower_bound = lower
        self.upper_bound = upper
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-chebyshev-moment-plan",
                "pencil": pencil.prepared_id,
                "mesh": mesh.mesh_id,
                "order": order_,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )

    def evaluate(self, /) -> ChebyshevMomentResult:
        points = self.mesh.fractional_points
        dimension = self.pencil.plan.basis.orbital_count
        center = 0.5 * (self.upper_bound + self.lower_bound)
        half_width = 0.5 * (self.upper_bound - self.lower_bound)
        identity = jnp.eye(dimension, dtype=self.pencil.hamiltonian.state.values.dtype)

        def scaled_action(vector):
            action = self.pencil.hamiltonian.apply(points, vector)
            return (action - center * vector) / half_width

        t0 = jnp.broadcast_to(
            identity[:, None, :], (dimension, points.shape[0], dimension)
        )
        t1 = jax.vmap(scaled_action)(identity)
        moments = [
            jnp.sum(
                self.mesh.weights * jnp.trace(jnp.swapaxes(t0, 0, 1), axis1=-2, axis2=-1)
            ),
            jnp.sum(
                self.mesh.weights * jnp.trace(jnp.swapaxes(t1, 0, 1), axis1=-2, axis2=-1)
            ),
        ]
        previous, current = t0, t1
        maximum_residual = jnp.asarray(0.0, dtype=points.dtype)
        for _ in range(2, self.order):
            applied = jax.vmap(scaled_action)(current)
            next_value = 2.0 * applied - previous
            recurrence = jnp.max(
                jnp.abs(next_value - (2.0 * applied - previous)), initial=0.0
            )
            maximum_residual = jnp.maximum(maximum_residual, recurrence)
            matrix = jnp.swapaxes(next_value, 0, 1)
            moments.append(
                jnp.sum(self.mesh.weights * jnp.trace(matrix, axis1=-2, axis2=-1))
            )
            previous, current = current, next_value
        moment_values = jnp.real(jnp.stack(tuple(moments)))
        successful = jnp.all(jnp.isfinite(moment_values)) & jnp.isfinite(maximum_residual)
        return ChebyshevMomentResult(
            moment_values,
            jnp.asarray(self.lower_bound),
            jnp.asarray(self.upper_bound),
            maximum_residual,
            successful,
            self.plan_id,
        )


__all__ = [
    "ChebyshevMomentPlan",
    "ChebyshevMomentResult",
    "PeriodicSpectrumPlan",
    "PeriodicSpectrumResult",
    "ReciprocalSupport",
]
