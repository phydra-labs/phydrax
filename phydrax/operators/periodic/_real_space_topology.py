#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded real-space Bott topology for finite periodic orbital systems."""

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ._finite import PeriodicFiniteOrbitalRealization


class BottIndexResult(StrictModule, NonTrainableState):
    raw_index: Array
    nearest_integer: Array
    quantization_residual: Array
    spectral_gap: Array
    projector_residual: Array
    unitary_residual: Array
    successful: Array
    realization_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class BottIndexPlan(StrictModule, NonTrainableState):
    realization: PeriodicFiniteOrbitalRealization
    occupied_count: int = eqx.field(static=True)
    spectral_gap_tolerance: float = eqx.field(static=True)
    quantization_tolerance: float = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        realization: PeriodicFiniteOrbitalRealization,
        occupied_count: int,
        /,
        *,
        spectral_gap_tolerance: float = 1.0e-8,
        quantization_tolerance: float = 1.0e-6,
        maximum_matrix_elements: int = 4_000_000,
    ):
        if not isinstance(realization, PeriodicFiniteOrbitalRealization):
            raise TypeError("realization must be PeriodicFiniteOrbitalRealization.")
        occupied = int(occupied_count)
        dimension = realization.hamiltonian.input_size
        gap = float(spectral_gap_tolerance)
        quantization = float(quantization_tolerance)
        maximum = int(maximum_matrix_elements)
        if (
            occupied < 1
            or occupied >= dimension
            or any(not isfinite(value) or value < 0.0 for value in (gap, quantization))
            or maximum < dimension * dimension
        ):
            raise ValueError(
                "Bott index occupation, tolerances, or resource limit is invalid."
            )
        if len(realization.boundary.supercell_shape) != 2:
            raise ValueError(
                "Bott index currently requires a two-dimensional finite realization."
            )
        self.realization = realization
        self.occupied_count = occupied
        self.spectral_gap_tolerance = gap
        self.quantization_tolerance = quantization
        self.maximum_matrix_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bott-index-plan",
                "realization": realization.realization_id,
                "occupied_count": occupied,
                "spectral_gap_tolerance": gap,
                "quantization_tolerance": quantization,
                "maximum_matrix_elements": maximum,
            }
        )

    def evaluate(self, /) -> BottIndexResult:
        hamiltonian = self.realization.hamiltonian.to_dense()
        overlap = self.realization.overlap.to_dense()
        identity = jnp.eye(hamiltonian.shape[0], dtype=overlap.dtype)
        overlap_residual = jnp.max(jnp.abs(overlap - identity))
        overlap_residual = eqx.error_if(
            overlap_residual,
            overlap_residual > 1.0e-10,
            "Bott index currently requires an orthonormal finite basis.",
        )
        solved = eigensolve(
            Eigenproblem(
                DenseLinearOperator(
                    hamiltonian,
                    properties=OperatorProperties(
                        self_adjoint=True,
                        evidence={"self_adjoint": "construction"},
                    ),
                ),
                problem_id=self.plan_id,
            ),
            policy=EigenSolvePolicy(
                DenseEigh(),
                count=hamiltonian.shape[0],
                which="smallest-algebraic",
            ),
        )
        values = np.asarray(solved.eigenvalues.real)
        vectors = np.asarray(solved.eigenvectors)
        occupied = vectors[:, : self.occupied_count]
        projector = occupied @ np.conj(occupied.T)
        dimension = projector.shape[0]
        complement = np.eye(dimension, dtype=projector.dtype) - projector
        coordinates = np.asarray(self.realization.cell_coordinates, dtype=np.float64)
        orbital_count = len(self.realization.orbital_labels)
        fractional = np.repeat(coordinates, orbital_count, axis=0)
        shape = np.asarray(self.realization.boundary.supercell_shape, dtype=np.float64)
        phases = np.exp(2.0j * pi * fractional / shape[None, :])
        projected = []
        unitarity = 0.0
        for axis in range(2):
            candidate = projector @ np.diag(phases[:, axis]) @ projector + complement
            left, _, right_h = np.linalg.svd(candidate)
            unitary = left @ right_h
            projected.append(unitary)
            unitarity = max(
                unitarity,
                float(np.max(np.abs(np.conj(unitary.T) @ unitary - np.eye(dimension)))),
            )
        first, second = projected
        commutator = second @ first @ np.conj(second.T) @ np.conj(first.T)
        raw = float(np.sum(np.angle(np.linalg.eigvals(commutator))) / (2.0 * pi))
        nearest = int(np.rint(raw))
        quantization = abs(raw - nearest)
        spectral_gap = float(
            values[self.occupied_count] - values[self.occupied_count - 1]
        )
        projector_residual = float(np.max(np.abs(projector @ projector - projector)))
        successful = bool(
            np.all(np.asarray(solved.successful))
            and spectral_gap > self.spectral_gap_tolerance
            and quantization <= self.quantization_tolerance
            and np.isfinite(raw)
            and unitarity <= 1.0e-8
        )
        result_id = canonical_fingerprint(
            {
                "kind": "bott-index-result",
                "plan": self.plan_id,
                "arrays": array_tree_fingerprint(
                    {
                        "raw_index": np.asarray(raw),
                        "nearest_integer": np.asarray(nearest),
                        "spectral_gap": np.asarray(spectral_gap),
                    }
                ),
            }
        )
        return BottIndexResult(
            jnp.asarray(raw),
            jnp.asarray(nearest, dtype=jnp.int32),
            jnp.asarray(quantization),
            jnp.asarray(spectral_gap),
            jnp.asarray(projector_residual),
            jnp.asarray(unitarity),
            jnp.asarray(successful),
            self.realization.realization_id,
            result_id,
        )


__all__ = ["BottIndexPlan", "BottIndexResult"]
