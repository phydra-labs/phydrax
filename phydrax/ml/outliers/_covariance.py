#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract, ML_NONCONVERGED
from .._numerics import weighted_covariance
from ._common import (
    _BLOCKWISE_BINDING,
    _case_count,
    _fit_arrays,
    _fit_status,
    _prepare_queries,
    _restore_scores,
    _score_bounds,
    _weighted_threshold,
    OutlierDiagnostics,
)


def _regularize_covariance(
    covariance: Array,
    *,
    shrinkage: float,
    ridge: float,
) -> tuple[Array, Array, Array, Array]:
    features = int(covariance.shape[-1])
    trace_scale = jnp.maximum(
        jnp.real(jnp.trace(covariance)) / features, jnp.finfo(covariance.real.dtype).eps
    )
    regularized = (
        (1.0 - float(shrinkage)) * covariance
        + float(shrinkage) * trace_scale * jnp.eye(features, dtype=covariance.dtype)
        + float(ridge) * trace_scale * jnp.eye(features, dtype=covariance.dtype)
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(regularized)
    cutoff = jnp.max(eigenvalues) * features * jnp.finfo(eigenvalues.dtype).eps
    retained = eigenvalues > cutoff
    inverse_values = jnp.where(retained, 1.0 / jnp.maximum(eigenvalues, cutoff), 0.0)
    precision = (eigenvectors * inverse_values[None, :]) @ jnp.conj(eigenvectors).T
    rank = jnp.sum(retained, dtype=jnp.int32)
    condition = jnp.max(eigenvalues) / jnp.maximum(
        jnp.min(jnp.where(retained, eigenvalues, jnp.inf)),
        jnp.finfo(eigenvalues.dtype).tiny,
    )
    log_determinant = jnp.sum(
        jnp.log(jnp.maximum(eigenvalues, jnp.finfo(eigenvalues.dtype).tiny))
    )
    return precision, rank, condition, log_determinant


def _mahalanobis_one(query: Array, location: Array, precision: Array) -> Array:
    delta = query - location
    return jnp.real(oe.contract("qi,ij,qj->q", jnp.conj(delta), precision, delta))


class CovarianceOutlierModel(AbstractArrayModel):
    """Smooth squared-Mahalanobis score with separate hard and relaxed decisions."""

    location: Array
    precision: Array
    threshold: Array
    log_determinant: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        location: ArrayLike,
        precision: ArrayLike,
        threshold: ArrayLike,
        log_determinant: ArrayLike,
        *,
        case_shape: tuple[int, ...],
    ):
        location_ = jnp.asarray(location)
        self.location = location_
        self.precision = jnp.asarray(precision)
        self.threshold = jnp.asarray(threshold)
        self.log_determinant = jnp.asarray(log_determinant)
        self.case_shape = tuple(case_shape)
        self.in_size = int(location_.shape[-1])
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        scores = jax.vmap(_mahalanobis_one)(
            queries,
            self.location.reshape((cases, self.in_size)),
            self.precision.reshape((cases, self.in_size, self.in_size)),
        )
        return _restore_scores(
            scores, case_shape=self.case_shape, query_shape=query_shape
        )

    def predict(self, x: Any, /) -> Array:
        """Return hard anomaly indicators; this comparison is nondifferentiable."""
        scores = self(x)
        threshold = self.threshold.reshape(
            self.case_shape + (1,) * (scores.ndim - len(self.case_shape))
        )
        return jax.lax.stop_gradient(scores > threshold)

    def smooth_membership(self, x: Any, /, *, temperature: ArrayLike = 1.0) -> Array:
        scores = self(x)
        threshold = self.threshold.reshape(
            self.case_shape + (1,) * (scores.ndim - len(self.case_shape))
        )
        scale = jnp.maximum(jnp.asarray(temperature), jnp.finfo(float).tiny)
        return jax.nn.sigmoid((scores - threshold) / scale)


class CovarianceOutlierRecipe(AbstractRecipe):
    """Weighted covariance/Mahalanobis anomaly scoring with calibrated contamination."""

    contamination: float = eqx.field(static=True)
    shrinkage: float = eqx.field(static=True)
    ridge: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        shrinkage: float = 0.0,
        ridge: float = 1e-6,
    ):
        if not 0.0 < float(contamination) < 0.5:
            raise ValueError("contamination must lie in (0, 0.5).")
        if not 0.0 <= float(shrinkage) <= 1.0 or float(ridge) <= 0.0:
            raise ValueError("shrinkage must lie in [0, 1] and ridge must be positive.")
        self.contamination = float(contamination)
        self.shrinkage = float(shrinkage)
        self.ridge = float(ridge)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, active = _fit_arrays(batch)
        location, covariance, covariance_valid = weighted_covariance(x, weights)
        cases = _case_count(batch.case_shape)
        precision, rank, condition, log_determinant = jax.vmap(
            lambda covariance_: _regularize_covariance(
                covariance_, shrinkage=self.shrinkage, ridge=self.ridge
            )
        )(covariance.reshape((cases, batch.feature_count, batch.feature_count)))
        precision = precision.reshape(
            batch.case_shape + (batch.feature_count, batch.feature_count)
        )
        rank = rank.reshape(batch.case_shape)
        condition = condition.reshape(batch.case_shape)
        log_determinant = log_determinant.reshape(batch.case_shape)
        scores = jax.vmap(_mahalanobis_one)(
            x.reshape((cases, batch.sample_count, batch.feature_count)),
            location.reshape((cases, batch.feature_count)),
            precision.reshape((cases, batch.feature_count, batch.feature_count)),
        ).reshape(batch.case_shape + (batch.sample_count,))
        scores = jnp.where(active, scores, jnp.inf)
        threshold = _weighted_threshold(scores, weights, self.contamination)
        minimum, maximum = _score_bounds(scores, active)
        effective = jnp.sum(active, axis=-1)
        finite = (
            covariance_valid
            & jnp.all(jnp.isfinite(precision), axis=(-2, -1))
            & jnp.isfinite(threshold)
        )
        enough = effective > batch.feature_count
        valid = finite & enough
        status = _fit_status(finite, enough)
        diagnostics = OutlierDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.sum(jnp.where(active, weights * scores, 0.0), axis=-1)
            / jnp.maximum(jnp.sum(weights, axis=-1), jnp.finfo(float).tiny),
            iterations=1,
            effective_samples=effective,
            threshold=threshold,
            score_minimum=minimum,
            score_maximum=maximum,
            rank=rank,
            condition=condition,
            converged=True,
            method="covariance-mahalanobis",
        )
        model = CovarianceOutlierModel(
            location,
            precision,
            threshold,
            log_determinant,
            case_shape=batch.case_shape,
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="direct",
            nondifferentiable_outputs=("predict", "threshold", "rank", "valid", "status"),
            conditions=(
                "covariance rank is fixed",
                "score ordering at contamination threshold is fixed",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="covariance-mahalanobis",
            gradient_contract=contract,
        )


class EllipticEnvelopeModel(AbstractArrayModel):
    """Continuously robust Mahalanobis score with explicit hard envelope membership."""

    location: Array
    precision: Array
    threshold: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        location: ArrayLike,
        precision: ArrayLike,
        threshold: ArrayLike,
        *,
        case_shape: tuple[int, ...],
    ):
        location_ = jnp.asarray(location)
        self.location = location_
        self.precision = jnp.asarray(precision)
        self.threshold = jnp.asarray(threshold)
        self.case_shape = tuple(case_shape)
        self.in_size = int(location_.shape[-1])
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        scores = jax.vmap(_mahalanobis_one)(
            queries,
            self.location.reshape((cases, self.in_size)),
            self.precision.reshape((cases, self.in_size, self.in_size)),
        )
        return _restore_scores(
            scores, case_shape=self.case_shape, query_shape=query_shape
        )

    def predict(self, x: Any, /) -> Array:
        scores = self(x)
        threshold = self.threshold.reshape(
            self.case_shape + (1,) * (scores.ndim - len(self.case_shape))
        )
        return jax.lax.stop_gradient(scores > threshold)

    def smooth_membership(self, x: Any, /, *, temperature: ArrayLike = 1.0) -> Array:
        scores = self(x)
        threshold = self.threshold.reshape(
            self.case_shape + (1,) * (scores.ndim - len(self.case_shape))
        )
        return jax.nn.sigmoid(
            (scores - threshold)
            / jnp.maximum(jnp.asarray(temperature), jnp.finfo(float).tiny)
        )


def _robust_covariance_one(
    x: Array,
    base_weights: Array,
    iterations: int,
    tuning: float,
    shrinkage: float,
    ridge: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    location, covariance, _valid = weighted_covariance(x, base_weights)
    precision, rank, condition, _logdet = _regularize_covariance(
        covariance, shrinkage=shrinkage, ridge=ridge
    )

    def step(_iteration, state):
        current_location, current_precision, _rank, _condition, _delta = state
        scores = _mahalanobis_one(x, current_location, current_precision)
        robust_weight = 1.0 / (1.0 + scores / (tuning * tuning))
        combined = base_weights * robust_weight
        next_location, next_covariance, _next_valid = weighted_covariance(x, combined)
        next_precision, next_rank, next_condition, _next_logdet = _regularize_covariance(
            next_covariance, shrinkage=shrinkage, ridge=ridge
        )
        delta = jnp.linalg.norm(next_location - current_location) + jnp.linalg.norm(
            next_precision - current_precision
        ) / jnp.maximum(jnp.linalg.norm(current_precision), jnp.finfo(float).tiny)
        return next_location, next_precision, next_rank, next_condition, delta

    location, precision, rank, condition, delta = jax.lax.fori_loop(
        0,
        iterations,
        step,
        (location, precision, rank, condition, jnp.asarray(jnp.inf, dtype=x.real.dtype)),
    )
    scores = _mahalanobis_one(x, location, precision)
    objective = jnp.sum(
        base_weights * jnp.log1p(scores / (tuning * tuning))
    ) / jnp.maximum(jnp.sum(base_weights), jnp.finfo(float).tiny)
    return location, precision, scores, rank, condition, delta, objective


class EllipticEnvelopeRecipe(AbstractRecipe):
    """Fixed-iteration Cauchy-IRLS elliptic envelope with continuous robust weights."""

    contamination: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    tuning: float = eqx.field(static=True)
    shrinkage: float = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        iterations: int = 25,
        tuning: float = 2.5,
        shrinkage: float = 0.05,
        ridge: float = 1e-6,
        tolerance: float = 1e-5,
    ):
        if not 0.0 < float(contamination) < 0.5:
            raise ValueError("contamination must lie in (0, 0.5).")
        if int(iterations) <= 0 or float(tuning) <= 0.0:
            raise ValueError("iterations and tuning must be positive.")
        if not 0.0 <= float(shrinkage) <= 1.0 or float(ridge) <= 0.0:
            raise ValueError("shrinkage must lie in [0, 1] and ridge must be positive.")
        if float(tolerance) <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.contamination = float(contamination)
        self.iterations = int(iterations)
        self.tuning = float(tuning)
        self.shrinkage = float(shrinkage)
        self.ridge = float(ridge)
        self.tolerance = float(tolerance)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, active = _fit_arrays(batch)
        cases = _case_count(batch.case_shape)
        outputs = jax.vmap(
            lambda x_, w_: _robust_covariance_one(
                x_,
                w_,
                self.iterations,
                self.tuning,
                self.shrinkage,
                self.ridge,
            )
        )(
            x.reshape((cases, batch.sample_count, batch.feature_count)),
            weights.reshape((cases, batch.sample_count)),
        )
        location, precision, scores, rank, condition, residual, objective = outputs
        location = location.reshape(batch.case_shape + (batch.feature_count,))
        precision = precision.reshape(
            batch.case_shape + (batch.feature_count, batch.feature_count)
        )
        scores = scores.reshape(batch.case_shape + (batch.sample_count,))
        rank = rank.reshape(batch.case_shape)
        condition = condition.reshape(batch.case_shape)
        residual = residual.reshape(batch.case_shape)
        objective = objective.reshape(batch.case_shape)
        scores = jnp.where(active, scores, jnp.inf)
        threshold = _weighted_threshold(scores, weights, self.contamination)
        minimum, maximum = _score_bounds(scores, active)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(precision), axis=(-2, -1)) & jnp.isfinite(threshold)
        enough = effective > batch.feature_count
        converged = residual <= self.tolerance
        valid = finite & enough & converged
        status = _fit_status(finite, enough)
        status = jnp.where(finite & enough & ~converged, ML_NONCONVERGED, status).astype(
            jnp.int32
        )
        diagnostics = OutlierDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.iterations,
            effective_samples=effective,
            threshold=threshold,
            score_minimum=minimum,
            score_maximum=maximum,
            rank=rank,
            condition=condition,
            converged=converged,
            method="elliptic-envelope-cauchy",
        )
        model = EllipticEnvelopeModel(
            location, precision, threshold, case_shape=batch.case_shape
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("predict", "threshold", "rank", "valid", "status"),
            conditions=(
                "fixed IRLS iteration count",
                "covariance rank and score ordering at threshold are fixed",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="elliptic-envelope-cauchy",
            gradient_contract=contract,
        )


__all__ = [
    "CovarianceOutlierModel",
    "CovarianceOutlierRecipe",
    "EllipticEnvelopeModel",
    "EllipticEnvelopeRecipe",
]
