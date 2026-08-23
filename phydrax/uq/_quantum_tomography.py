#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import TracelessHermitianSpace


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


class QuantumPOVM(StrictModule):
    effects: Array
    valid: Array
    hermiticity_residual: Array
    completeness_residual: Array
    minimum_eigenvalue: Array
    outcome_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    povm_id: str = eqx.field(static=True)

    def __init__(
        self,
        effects: ArrayLike,
        /,
        *,
        povm_id: str,
        tolerance: float = 1e-9,
    ):
        values = jnp.asarray(effects)
        if values.ndim != 3 or values.shape[-2] != values.shape[-1]:
            raise ValueError("POVM effects must have shape (outcomes, n, n).")
        dimension = values.shape[-1]
        hermitian = 0.5 * (values + _adjoint(values))
        asymmetry = jnp.max(jnp.abs(values - _adjoint(values)))
        eigenvalues = jnp.linalg.eigvalsh(hermitian)
        minimum = jnp.min(eigenvalues)
        completeness = jnp.max(
            jnp.abs(jnp.sum(hermitian, axis=0) - jnp.eye(dimension, dtype=values.dtype))
        )
        identifier = str(povm_id)
        if not identifier:
            raise ValueError("povm_id must be non-empty.")
        self.effects = hermitian
        self.hermiticity_residual = asymmetry
        self.completeness_residual = completeness
        self.minimum_eigenvalue = minimum
        self.valid = (
            jnp.all(jnp.isfinite(values))
            & (asymmetry <= tolerance)
            & (completeness <= tolerance)
            & (minimum >= -tolerance)
        )
        self.outcome_count = values.shape[0]
        self.dimension = dimension
        self.povm_id = identifier

    def probabilities(self, density: ArrayLike, /) -> Array:
        rho = jnp.asarray(density)
        if rho.shape != (self.dimension, self.dimension):
            raise ValueError("Density shape does not match POVM dimension.")
        probabilities = jnp.real(jnp.einsum("kij,ji->k", self.effects, rho))
        return probabilities

    def identifiability_rank(self, /, *, tolerance: float = 1e-9) -> Array:
        space = TracelessHermitianSpace(self.dimension)
        basis = []
        for row in range(self.dimension):
            for column in range(row, self.dimension):
                real = (
                    jnp.zeros(space.shape, dtype=self.effects.dtype)
                    .at[row, column]
                    .set(1.0)
                )
                real = real.at[column, row].set(1.0 if row != column else 1.0)
                projected = space.project(real)
                if float(space.inner(projected, projected)) > tolerance:
                    basis.append(projected)
                if row != column:
                    imaginary = jnp.zeros(space.shape, dtype=self.effects.dtype)
                    imaginary = imaginary.at[row, column].set(1j)
                    imaginary = imaginary.at[column, row].set(-1j)
                    basis.append(space.project(imaginary))
        design = jnp.stack(
            [
                jnp.asarray([jnp.real(jnp.trace(effect @ vector)) for vector in basis])
                for effect in self.effects
            ]
        )
        singular_values = jnp.linalg.svd(design, compute_uv=False)
        return jnp.sum(singular_values > tolerance)


class QuantumTomographyData(StrictModule):
    counts: Array
    shots: Array
    valid: Array
    data_id: str = eqx.field(static=True)

    def __init__(self, counts: ArrayLike, /, *, data_id: str):
        values = jnp.asarray(counts, dtype=float)
        if values.ndim != 1:
            raise ValueError("Tomography counts must be a vector.")
        identifier = str(data_id)
        if not identifier:
            raise ValueError("data_id must be non-empty.")
        self.counts = values
        self.shots = jnp.sum(values)
        self.valid = jnp.all(jnp.isfinite(values) & (values >= 0.0)) & (self.shots > 0.0)
        self.data_id = identifier


class TomographyLikelihoodResult(StrictModule):
    log_likelihood: Array
    probabilities: Array
    normalization_residual: Array
    valid: Array

    def __init__(
        self,
        log_likelihood: ArrayLike,
        probabilities: ArrayLike,
        normalization_residual: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        self.log_likelihood = jnp.asarray(log_likelihood)
        self.probabilities = jnp.asarray(probabilities)
        self.normalization_residual = jnp.asarray(normalization_residual)
        self.valid = jnp.asarray(valid, dtype=bool)


def tomography_log_likelihood(
    povm: QuantumPOVM,
    data: QuantumTomographyData,
    density: ArrayLike,
    /,
) -> TomographyLikelihoodResult:
    if data.counts.shape != (povm.outcome_count,):
        raise ValueError("Tomography counts must match POVM outcomes.")
    probabilities = povm.probabilities(density)
    normalization = jnp.abs(jnp.sum(probabilities) - 1.0)
    safe = jnp.maximum(probabilities, jnp.finfo(probabilities.dtype).tiny)
    log_likelihood = jnp.sum(data.counts * jnp.log(safe))
    valid = (
        povm.valid
        & data.valid
        & jnp.all(jnp.isfinite(probabilities) & (probabilities >= 0.0))
        & (normalization <= 1e-8)
        & jnp.isfinite(log_likelihood)
    )
    return TomographyLikelihoodResult(log_likelihood, probabilities, normalization, valid)


def tetrahedral_qubit_povm() -> QuantumPOVM:
    directions = jnp.asarray(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    ) / jnp.sqrt(3.0)
    sigma = jnp.asarray(
        [
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=complex,
    )
    identity = jnp.eye(2, dtype=complex)
    effects = jnp.stack(
        [
            0.25 * (identity + jnp.einsum("a,aij->ij", direction, sigma))
            for direction in directions
        ]
    )
    return QuantumPOVM(effects, povm_id="qubit-tetrahedral")


__all__ = [
    "QuantumPOVM",
    "QuantumTomographyData",
    "TomographyLikelihoodResult",
    "tetrahedral_qubit_povm",
    "tomography_log_likelihood",
]
