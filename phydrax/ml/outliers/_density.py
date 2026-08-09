#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch
from .._contracts import AbstractRecipe, FitResult, GradientContract, ML_NONCONVERGED
from .._numerics import pairwise_distances, weighted_mean
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


def _kde_scores_one(
    queries: Array,
    training: Array,
    weights: Array,
    bandwidth: float,
) -> Array:
    squared = pairwise_distances(queries, training, metric="squared-euclidean")
    log_weights = jnp.where(
        weights > 0.0,
        jnp.log(jnp.maximum(weights, jnp.finfo(float).tiny)),
        -jnp.inf,
    )
    if jnp.issubdtype(training.dtype, jnp.complexfloating):
        log_kernel = -squared / (bandwidth * bandwidth) + log_weights[None, :]
        log_normalizer = training.shape[-1] * jnp.log(jnp.pi * bandwidth * bandwidth)
    else:
        log_kernel = -0.5 * squared / (bandwidth * bandwidth) + log_weights[None, :]
        log_normalizer = training.shape[-1] * jnp.log(bandwidth * jnp.sqrt(2.0 * jnp.pi))
    log_density = (
        jax.scipy.special.logsumexp(log_kernel, axis=-1)
        - jnp.log(jnp.maximum(jnp.sum(weights), jnp.finfo(float).tiny))
        - log_normalizer
    )
    return -log_density


def _kde_leave_one_out_one(
    training: Array,
    weights: Array,
    bandwidth: float,
) -> Array:
    n = int(training.shape[0])
    squared = pairwise_distances(training, metric="squared-euclidean")
    eligible = (weights[None, :] > 0.0) & ~jnp.eye(n, dtype=bool)
    if jnp.issubdtype(training.dtype, jnp.complexfloating):
        logits = -squared / (bandwidth * bandwidth) + jnp.log(
            jnp.maximum(weights[None, :], jnp.finfo(float).tiny)
        )
        log_normalizer = training.shape[-1] * jnp.log(jnp.pi * bandwidth * bandwidth)
    else:
        logits = -0.5 * squared / (bandwidth * bandwidth) + jnp.log(
            jnp.maximum(weights[None, :], jnp.finfo(float).tiny)
        )
        log_normalizer = training.shape[-1] * jnp.log(bandwidth * jnp.sqrt(2.0 * jnp.pi))
    logits = jnp.where(eligible, logits, -jnp.inf)
    denominator = jnp.maximum(jnp.sum(weights) - weights, jnp.finfo(float).tiny)
    log_density = (
        jax.scipy.special.logsumexp(logits, axis=-1)
        - jnp.log(denominator)
        - log_normalizer
    )
    return -log_density


class KernelDensityOutlierModel(AbstractArrayModel):
    """Gaussian KDE negative-log-density anomaly score."""

    training_features: Array
    training_weights: Array
    threshold: Array
    bandwidth: float = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        training_features: ArrayLike,
        training_weights: ArrayLike,
        threshold: ArrayLike,
        *,
        bandwidth: float,
        case_shape: tuple[int, ...],
    ):
        train = jnp.asarray(training_features)
        self.training_features = train
        self.training_weights = jnp.asarray(training_weights)
        self.threshold = jnp.asarray(threshold)
        self.bandwidth = float(bandwidth)
        self.case_shape = tuple(case_shape)
        self.in_size = int(train.shape[-1])
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        scores = jax.vmap(lambda q_, t_, w_: _kde_scores_one(q_, t_, w_, self.bandwidth))(
            queries,
            self.training_features.reshape((cases,) + self.training_features.shape[-2:]),
            self.training_weights.reshape((cases, self.training_weights.shape[-1])),
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


class KernelDensityOutlierRecipe(AbstractRecipe):
    """Weighted Gaussian KDE with leave-one-out anomaly-threshold calibration."""

    bandwidth: float = eqx.field(static=True)
    contamination: float = eqx.field(static=True)

    def __init__(self, *, bandwidth: float = 1.0, contamination: float = 0.1):
        if float(bandwidth) <= 0.0:
            raise ValueError("bandwidth must be positive.")
        if not 0.0 < float(contamination) < 0.5:
            raise ValueError("contamination must lie in (0, 0.5).")
        self.bandwidth = float(bandwidth)
        self.contamination = float(contamination)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, active = _fit_arrays(batch)
        cases = _case_count(batch.case_shape)
        scores = jax.vmap(lambda x_, w_: _kde_leave_one_out_one(x_, w_, self.bandwidth))(
            x.reshape((cases, batch.sample_count, batch.feature_count)),
            weights.reshape((cases, batch.sample_count)),
        ).reshape(batch.case_shape + (batch.sample_count,))
        scores = jnp.where(active, scores, jnp.inf)
        threshold = _weighted_threshold(scores, weights, self.contamination)
        minimum, maximum = _score_bounds(scores, active)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.isfinite(threshold) & jnp.all(
            jnp.where(active, jnp.isfinite(scores), True), axis=-1
        )
        enough = effective >= 2
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
            rank=-1,
            condition=jnp.nan,
            converged=True,
            method="kernel-density-outlier",
        )
        model = KernelDensityOutlierModel(
            x,
            weights,
            threshold,
            bandwidth=self.bandwidth,
            case_shape=batch.case_shape,
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="smooth",
            fit_mode="direct",
            nondifferentiable_outputs=("predict", "threshold", "valid", "status"),
            conditions=(
                "leave-one-out score ordering at contamination threshold is fixed",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="kernel-density-outlier",
            gradient_contract=contract,
        )


def _robust_score_one(
    queries: Array,
    location: Array,
    scale: Array,
    tuning: float,
) -> Array:
    standardized = (queries - location) / scale
    magnitude_squared = jnp.real(standardized * jnp.conj(standardized))
    return jnp.sum(
        tuning * tuning * (jnp.sqrt(1.0 + magnitude_squared / (tuning * tuning)) - 1.0),
        axis=-1,
    )


def _fit_robust_one(
    x: Array,
    weights: Array,
    iterations: int,
    tuning: float,
    scale_floor: float,
) -> tuple[Array, Array, Array, Array, Array]:
    location = weighted_mean(x, weights)
    centered = x - location
    variance = weighted_mean(jnp.real(centered * jnp.conj(centered)), weights)
    scale = jnp.sqrt(jnp.maximum(variance, scale_floor * scale_floor))

    def step(_iteration, state):
        current_location, current_scale, _delta = state
        standardized = (x - current_location) / current_scale
        influence = 1.0 / jnp.sqrt(
            1.0 + jnp.real(standardized * jnp.conj(standardized)) / (tuning * tuning)
        )
        coordinate_weights = weights[:, None] * influence
        mass = jnp.sum(coordinate_weights, axis=0)
        next_location = jnp.sum(coordinate_weights * x, axis=0) / jnp.maximum(
            mass, jnp.finfo(float).tiny
        )
        residual = x - next_location
        next_variance = jnp.sum(
            coordinate_weights * jnp.real(residual * jnp.conj(residual)), axis=0
        ) / jnp.maximum(mass, jnp.finfo(float).tiny)
        next_scale = jnp.sqrt(jnp.maximum(next_variance, scale_floor * scale_floor))
        delta = jnp.linalg.norm(next_location - current_location) + jnp.linalg.norm(
            next_scale - current_scale
        )
        return next_location, next_scale, delta

    location, scale, residual = jax.lax.fori_loop(
        0,
        iterations,
        step,
        (location, scale, jnp.asarray(jnp.inf, dtype=x.real.dtype)),
    )
    scores = _robust_score_one(x, location, scale, tuning)
    objective = jnp.sum(weights * scores) / jnp.maximum(
        jnp.sum(weights), jnp.finfo(float).tiny
    )
    return location, scale, scores, residual, objective


class RobustNoveltyModel(AbstractArrayModel):
    """Smooth pseudo-Huber standardized novelty score with a separate hard cutoff."""

    location: Array
    scale: Array
    threshold: Array
    tuning: float = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    _input_binding: ClassVar[ModelBinding] = _BLOCKWISE_BINDING

    def __init__(
        self,
        location: ArrayLike,
        scale: ArrayLike,
        threshold: ArrayLike,
        *,
        tuning: float,
        case_shape: tuple[int, ...],
    ):
        location_ = jnp.asarray(location)
        self.location = location_
        self.scale = jnp.asarray(scale)
        self.threshold = jnp.asarray(threshold)
        self.tuning = float(tuning)
        self.case_shape = tuple(case_shape)
        self.in_size = int(location_.shape[-1])
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        queries, query_shape = _prepare_queries(
            x, case_shape=self.case_shape, feature_count=self.in_size
        )
        cases = _case_count(self.case_shape)
        scores = jax.vmap(lambda q_, l_, s_: _robust_score_one(q_, l_, s_, self.tuning))(
            queries,
            self.location.reshape((cases, self.in_size)),
            self.scale.reshape((cases, self.in_size)),
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


class RobustNoveltyRecipe(AbstractRecipe):
    """Fixed-iteration featurewise smooth robust location/scale novelty fitting."""

    contamination: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    tuning: float = eqx.field(static=True)
    scale_floor: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        iterations: int = 25,
        tuning: float = 1.345,
        scale_floor: float = 1e-6,
        tolerance: float = 1e-5,
    ):
        if not 0.0 < float(contamination) < 0.5:
            raise ValueError("contamination must lie in (0, 0.5).")
        if int(iterations) <= 0 or float(tuning) <= 0.0:
            raise ValueError("iterations and tuning must be positive.")
        if float(scale_floor) <= 0.0 or float(tolerance) <= 0.0:
            raise ValueError("scale_floor and tolerance must be positive.")
        self.contamination = float(contamination)
        self.iterations = int(iterations)
        self.tuning = float(tuning)
        self.scale_floor = float(scale_floor)
        self.tolerance = float(tolerance)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, weights, active = _fit_arrays(batch)
        cases = _case_count(batch.case_shape)
        location, scale, scores, residual, objective = jax.vmap(
            lambda x_, w_: _fit_robust_one(
                x_, w_, self.iterations, self.tuning, self.scale_floor
            )
        )(
            x.reshape((cases, batch.sample_count, batch.feature_count)),
            weights.reshape((cases, batch.sample_count)),
        )
        location = location.reshape(batch.case_shape + (batch.feature_count,))
        scale = scale.reshape(batch.case_shape + (batch.feature_count,))
        scores = scores.reshape(batch.case_shape + (batch.sample_count,))
        residual = residual.reshape(batch.case_shape)
        objective = objective.reshape(batch.case_shape)
        scores = jnp.where(active, scores, jnp.inf)
        threshold = _weighted_threshold(scores, weights, self.contamination)
        minimum, maximum = _score_bounds(scores, active)
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(scale), axis=-1) & jnp.isfinite(threshold)
        enough = effective >= 2
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
            rank=batch.feature_count,
            condition=jnp.max(scale, axis=-1)
            / jnp.maximum(jnp.min(scale, axis=-1), jnp.finfo(float).tiny),
            converged=converged,
            method="robust-pseudo-huber-novelty",
        )
        model = RobustNoveltyModel(
            location,
            scale,
            threshold,
            tuning=self.tuning,
            case_shape=batch.case_shape,
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("predict", "threshold", "valid", "status"),
            conditions=(
                "fixed IRLS iteration count",
                "scale-floor branches and score ordering at threshold are held fixed",
            ),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="robust-pseudo-huber-novelty",
            gradient_contract=contract,
        )


__all__ = [
    "KernelDensityOutlierModel",
    "KernelDensityOutlierRecipe",
    "RobustNoveltyModel",
    "RobustNoveltyRecipe",
]
