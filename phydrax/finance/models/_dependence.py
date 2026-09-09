#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Validated multi-asset dependence and lognormal factor records."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    HermitianSpectrum,
    OperatorProperties,
)


def _correlation_matrix(value: ArrayLike, /) -> Array:
    matrix = jnp.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("correlation must be a non-empty square matrix.")
    matrix = eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix))
        | jnp.any(jnp.abs(matrix - matrix.T) > 1.0e-7)
        | jnp.any(jnp.abs(jnp.diag(matrix) - 1.0) > 1.0e-7)
        | jnp.any(jnp.abs(matrix) > 1.0 + 1.0e-7),
        "correlation must be finite, symmetric, unit diagonal, and bounded by one.",
    )
    spectrum = HermitianSpectrum(matrix, tolerance=1.0e-12)
    return eqx.error_if(
        matrix,
        jnp.min(spectrum.eigenvalues) <= 0.0,
        "correlation must be positive definite; singular dependence is not implicit.",
    )


def _certified_correlation_operator(matrix: Array, /) -> DenseLinearOperator:
    properties = OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "validated-correlation-symmetry",
            "positive_semidefinite": "validated-correlation-spectrum",
            "positive_definite": "validated-correlation-spectrum",
        },
    )
    return DenseLinearOperator(matrix, properties=properties)


class CorrelationMatrix(StrictModule):
    """Positive-definite correlation matrix with native factorization."""

    matrix: Array
    dimension: int = eqx.field(static=True)

    def __init__(self, matrix: ArrayLike, /):
        matrix_ = _correlation_matrix(matrix)
        self.matrix = matrix_
        self.dimension = int(matrix_.shape[0])

    def cholesky_factor(self) -> Array:
        prepared = factorize(
            _certified_correlation_operator(self.matrix),
            FactorizationPolicy("cholesky"),
        )
        return prepared.prepared_solve.state.factor


class MultiAssetLognormalModel(StrictModule):
    """Measure-neutral lognormal diffusion scales and their dependence."""

    volatilities: Array
    dependence: CorrelationMatrix
    asset_count: int = eqx.field(static=True)

    def __init__(self, volatilities: ArrayLike, dependence: CorrelationMatrix, /):
        if not isinstance(dependence, CorrelationMatrix):
            raise TypeError("dependence must be a CorrelationMatrix.")
        volatility = jnp.asarray(volatilities, dtype=float)
        if volatility.shape != (dependence.dimension,):
            raise ValueError(
                "volatilities must contain one entry per dependence dimension."
            )
        volatility = eqx.error_if(
            volatility,
            jnp.any(~jnp.isfinite(volatility)) | jnp.any(volatility <= 0.0),
            "volatilities must be finite and strictly positive.",
        )
        self.volatilities = volatility
        self.dependence = dependence
        self.asset_count = dependence.dimension

    def covariance(self) -> Array:
        return (
            self.volatilities[:, None]
            * self.dependence.matrix
            * self.volatilities[None, :]
        )

    def diffusion_factor(self) -> Array:
        return self.volatilities[:, None] * self.dependence.cholesky_factor()


class FactorDependenceModel(StrictModule):
    """Low-rank factor covariance plus strictly positive idiosyncratic variance."""

    loadings: Array
    idiosyncratic_variances: Array
    asset_count: int = eqx.field(static=True)
    factor_count: int = eqx.field(static=True)

    def __init__(self, loadings: ArrayLike, idiosyncratic_variances: ArrayLike, /):
        loadings_ = jnp.asarray(loadings, dtype=float)
        residual = jnp.asarray(idiosyncratic_variances, dtype=float)
        if loadings_.ndim != 2 or loadings_.shape[0] < 1 or loadings_.shape[1] < 1:
            raise ValueError("loadings must be a non-empty asset-by-factor matrix.")
        if residual.shape != (loadings_.shape[0],):
            raise ValueError("idiosyncratic_variances must contain one entry per asset.")
        loadings_ = eqx.error_if(
            loadings_, jnp.any(~jnp.isfinite(loadings_)), "loadings must be finite."
        )
        residual = eqx.error_if(
            residual,
            jnp.any(~jnp.isfinite(residual)) | jnp.any(residual <= 0.0),
            "idiosyncratic_variances must be finite and strictly positive.",
        )
        self.loadings = loadings_
        self.idiosyncratic_variances = residual
        self.asset_count, self.factor_count = map(int, loadings_.shape)

    def covariance(self) -> Array:
        return self.loadings @ self.loadings.T + jnp.diag(self.idiosyncratic_variances)

    def correlation(self) -> CorrelationMatrix:
        covariance = self.covariance()
        standard_deviation = jnp.sqrt(jnp.diag(covariance))
        return CorrelationMatrix(
            covariance / (standard_deviation[:, None] * standard_deviation[None, :])
        )


class GaussianCopulaModel(StrictModule):
    dependence: CorrelationMatrix

    def __init__(self, dependence: CorrelationMatrix, /):
        if not isinstance(dependence, CorrelationMatrix):
            raise TypeError("dependence must be a CorrelationMatrix.")
        self.dependence = dependence


class StudentTCopulaModel(StrictModule):
    dependence: CorrelationMatrix
    degrees_of_freedom: Array

    def __init__(self, dependence: CorrelationMatrix, degrees_of_freedom: ArrayLike, /):
        if not isinstance(dependence, CorrelationMatrix):
            raise TypeError("dependence must be a CorrelationMatrix.")
        degrees = jnp.asarray(degrees_of_freedom, dtype=float)
        if degrees.shape != ():
            raise ValueError("degrees_of_freedom must be scalar.")
        degrees = eqx.error_if(
            degrees,
            ~jnp.isfinite(degrees) | (degrees <= 2.0),
            "degrees_of_freedom must exceed two.",
        )
        self.dependence = dependence
        self.degrees_of_freedom = degrees


__all__ = [
    "CorrelationMatrix",
    "FactorDependenceModel",
    "GaussianCopulaModel",
    "MultiAssetLognormalModel",
    "StudentTCopulaModel",
]
