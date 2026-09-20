#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Density-fitting and pivoted-Cholesky two-electron representations."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ....linalg import DenseLinearOperator, OperatorProperties
from ....linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy


class FactorizedERITensor(StrictModule, NonTrainableState):
    """Three-index factors satisfying (μν|κλ) ≈ ΣP Lᴾμν Lᴾκλ."""

    factors: Array
    residual_bound: Array
    source_id: str = eqx.field(static=True)
    representation: str = eqx.field(static=True)
    tensor_id: str = eqx.field(static=True)

    def __init__(
        self,
        factors: ArrayLike,
        residual_bound: ArrayLike,
        source_id: str,
        representation: str,
        /,
    ):
        values = jnp.asarray(factors)
        residual = jnp.asarray(residual_bound, dtype=values.real.dtype).reshape(())
        source = str(source_id).strip()
        representation_ = str(representation).strip()
        if (
            values.ndim != 3
            or values.shape[1] != values.shape[2]
            or not source
            or not representation_
        ):
            raise ValueError(
                "Factorized ERIs require shape (rank, AO, AO) and exact identities."
            )
        if not np.isfinite(float(residual)) or float(residual) < 0.0:
            raise ValueError(
                "Factorized ERI residual bound must be finite and non-negative."
            )
        self.factors = values
        self.residual_bound = residual
        self.source_id = source
        self.representation = representation_
        self.tensor_id = canonical_fingerprint(
            {
                "kind": "factorized-electron-repulsion-tensor",
                "source": source,
                "representation": representation_,
                "residual_bound": float(residual),
                "arrays": array_tree_fingerprint(np.asarray(values)),
            }
        )

    @property
    def rank(self) -> int:
        return self.factors.shape[0]

    @property
    def orbital_count(self) -> int:
        return self.factors.shape[1]

    def reconstruct(self, /) -> Array:
        return contract("Pab,Pcd->abcd", self.factors, self.factors)

    def coulomb(self, density: ArrayLike, /) -> Array:
        density_ = jnp.asarray(density, dtype=self.factors.dtype)
        if density_.shape != (self.orbital_count, self.orbital_count):
            raise ValueError("Density matrix does not align with factorized ERIs.")
        projected = contract("Pcd,cd->P", self.factors, density_)
        return contract("Pab,P->ab", self.factors, projected)

    def exchange(self, density: ArrayLike, /) -> Array:
        density_ = jnp.asarray(density, dtype=self.factors.dtype)
        if density_.shape != (self.orbital_count, self.orbital_count):
            raise ValueError("Density matrix does not align with factorized ERIs.")
        return contract("Pac,cd,Pbd->ab", self.factors, density_, self.factors)


class PivotedCholeskyERIPlan(StrictModule, NonTrainableState):
    tolerance: float = eqx.field(static=True)
    maximum_rank: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, tolerance: float = 1.0e-10, maximum_rank: int = 4096, /):
        tolerance_ = float(tolerance)
        rank = int(maximum_rank)
        if not isfinite(tolerance_) or tolerance_ <= 0.0 or rank <= 0:
            raise ValueError("Cholesky tolerance and maximum rank must be positive.")
        self.tolerance = tolerance_
        self.maximum_rank = rank
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pivoted-cholesky-eri-plan",
                "tolerance": tolerance_,
                "maximum_rank": rank,
            }
        )

    def factorize(self, electron_repulsion: ArrayLike, /) -> FactorizedERITensor:
        eri = np.asarray(electron_repulsion)
        if eri.ndim != 4 or len(set(eri.shape)) != 1 or np.any(~np.isfinite(eri)):
            raise ValueError(
                "Electron-repulsion tensor must have finite shape (N,N,N,N)."
            )
        orbital_count = eri.shape[0]
        matrix = eri.reshape((orbital_count**2, orbital_count**2))
        matrix = 0.5 * (matrix + matrix.T.conj())
        diagonal = np.real(np.diag(matrix)).copy()
        if np.min(diagonal) < -10.0 * self.tolerance:
            raise ValueError(
                "ERI pair matrix is not positive semidefinite within tolerance."
            )
        diagonal = np.maximum(diagonal, 0.0)
        rank_limit = min(self.maximum_rank, matrix.shape[0])
        columns = np.zeros((matrix.shape[0], rank_limit), dtype=matrix.dtype)
        completed = 0
        for column in range(rank_limit):
            pivot = int(np.argmax(diagonal))
            pivot_value = float(diagonal[pivot])
            if pivot_value <= self.tolerance:
                break
            previous = columns[:, :column] @ np.conj(columns[pivot, :column])
            columns[:, column] = (matrix[:, pivot] - previous) / np.sqrt(pivot_value)
            diagonal = np.maximum(
                diagonal - np.real(columns[:, column] * np.conj(columns[:, column])),
                0.0,
            )
            completed = column + 1
        residual = float(np.max(diagonal, initial=0.0))
        return FactorizedERITensor(
            np.moveaxis(
                columns[:, :completed].reshape((orbital_count, orbital_count, completed)),
                -1,
                0,
            ),
            residual,
            self.plan_id,
            "pivoted-cholesky",
        )


class DensityFittingPlan(StrictModule, NonTrainableState):
    metric_tolerance: float = eqx.field(static=True)
    maximum_rank: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, metric_tolerance: float = 1.0e-10, maximum_rank: int = 4096, /):
        tolerance = float(metric_tolerance)
        rank = int(maximum_rank)
        if not isfinite(tolerance) or tolerance <= 0.0 or rank <= 0:
            raise ValueError(
                "Density-fitting metric tolerance and rank must be positive."
            )
        self.metric_tolerance = tolerance
        self.maximum_rank = rank
        self.plan_id = canonical_fingerprint(
            {
                "kind": "density-fitting-plan",
                "metric_tolerance": tolerance,
                "maximum_rank": rank,
            }
        )

    def factorize(
        self,
        metric: ArrayLike,
        three_index: ArrayLike,
        /,
        *,
        source_id: str,
    ) -> FactorizedERITensor:
        metric_ = jnp.asarray(metric)
        three = jnp.asarray(three_index, dtype=metric_.dtype)
        if metric_.ndim != 2 or metric_.shape[0] != metric_.shape[1]:
            raise ValueError("Density-fitting metric must be square.")
        if (
            three.ndim != 3
            or three.shape[0] != metric_.shape[0]
            or three.shape[1] != three.shape[2]
        ):
            raise ValueError("Three-index integrals must have shape (aux, AO, AO).")
        solve = eigensolve(
            Eigenproblem(
                DenseLinearOperator(
                    0.5 * (metric_ + jnp.conj(metric_.T)),
                    properties=OperatorProperties(
                        self_adjoint=True,
                        evidence={"self_adjoint": "construction"},
                    ),
                )
            ),
            policy=EigenSolvePolicy(
                DenseEigh(),
                count=metric_.shape[0],
                which="smallest-algebraic",
            ),
        )
        if not bool(solve.successful):
            raise ValueError("Density-fitting metric eigensolve failed.")
        eigenvalues = np.asarray(solve.eigenvalues)
        eigenvectors = np.asarray(solve.eigenvectors)
        retained = np.flatnonzero(eigenvalues > self.metric_tolerance)
        if retained.size == 0 or retained.size > self.maximum_rank:
            raise ValueError(
                "Density-fitting metric rank lies outside the plan capacity."
            )
        whitener = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])[None, :]
        factors = contract("Pr,Pab->rab", jnp.asarray(whitener), three)
        omitted = eigenvalues[eigenvalues <= self.metric_tolerance]
        residual = float(np.max(np.abs(omitted), initial=0.0))
        return FactorizedERITensor(
            factors,
            residual,
            str(source_id),
            "density-fitting",
        )


__all__ = [
    "DensityFittingPlan",
    "FactorizedERITensor",
    "PivotedCholeskyERIPlan",
]
