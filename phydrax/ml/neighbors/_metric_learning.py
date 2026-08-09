#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel
from ..._model._binding import ModelBinding
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from ._utils import masked_softmax, size, validated_weights


class LinearMetricModel(AbstractArrayModel):
    """Learned linear embedding inducing a positive-semidefinite metric."""

    factor: Array
    feature_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        factor: ArrayLike,
        feature_count: int,
        component_count: int,
        case_shape: tuple[int, ...],
        method: str,
    ):
        factor_ = jnp.asarray(factor)
        features = int(feature_count)
        components = int(component_count)
        cases = tuple(int(value) for value in case_shape)
        if features <= 0 or components <= 0:
            raise ValueError("feature_count and component_count must be positive.")
        if factor_.shape != cases + (components, features):
            raise ValueError(
                "Metric factor must have shape case_shape + (component_count, feature_count)."
            )
        self.factor = factor_
        self.feature_count = features
        self.component_count = components
        self.case_shape = cases
        self.method = str(method)
        self.in_size = features
        self.out_size = components

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        points = jnp.asarray(x)
        if points.shape[-1] != self.feature_count:
            raise ValueError("Query feature size does not match metric factor.")
        if not self.case_shape:
            return points @ self.factor.T
        if points.shape[: len(self.case_shape)] != self.case_shape:
            raise ValueError(
                f"Query must begin with fitted case shape {self.case_shape}."
            )
        query_shape = points.shape[len(self.case_shape) : -1]
        cases = size(self.case_shape)
        q = size(tuple(int(s) for s in query_shape)) if query_shape else 1
        output = jax.vmap(lambda x_, a_: x_ @ a_.T)(
            points.reshape((cases, q, self.feature_count)),
            self.factor.reshape((cases, self.component_count, self.feature_count)),
        )
        return output.reshape(self.case_shape + query_shape + (self.component_count,))

    def squared_distance(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        delta = self(left) - self(right)
        return jnp.sum(delta * delta, axis=-1)

    @property
    def metric_matrix(self) -> Array:
        return jnp.swapaxes(self.factor, -1, -2) @ self.factor


class NeighborhoodComponentsAnalysisRecipe(AbstractRecipe):
    component_count: int | None = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    learning_rate: Array
    temperature: Array
    ridge: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        component_count: int | None = None,
        iterations: int = 100,
        learning_rate: ArrayLike = 0.01,
        temperature: ArrayLike = 1.0,
        ridge: ArrayLike = 1e-4,
        weight_policy: WeightPolicy = "statistical",
    ):
        if component_count is not None and int(component_count) <= 0:
            raise ValueError("component_count must be positive.")
        if int(iterations) <= 0:
            raise ValueError("iterations must be positive.")
        self.component_count = None if component_count is None else int(component_count)
        self.iterations = int(iterations)
        learning_rate_ = jnp.asarray(learning_rate, dtype=float)
        temperature_ = jnp.asarray(temperature, dtype=float)
        ridge_ = jnp.asarray(ridge, dtype=float)
        if any(value.ndim != 0 for value in (learning_rate_, temperature_, ridge_)):
            raise ValueError("learning_rate, temperature, and ridge must be scalars.")
        self.learning_rate = eqx.error_if(
            learning_rate_,
            ~jnp.isfinite(learning_rate_) | (learning_rate_ <= 0.0),
            "learning_rate must be finite and positive.",
        )
        self.temperature = eqx.error_if(
            temperature_,
            ~jnp.isfinite(temperature_) | (temperature_ <= 0.0),
            "temperature must be finite and positive.",
        )
        self.ridge = eqx.error_if(
            ridge_,
            ~jnp.isfinite(ridge_) | (ridge_ < 0.0),
            "ridge must be finite and nonnegative.",
        )
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        labels = batch.require_targets()
        if labels.shape != batch.case_shape + (batch.sample_count,) or not jnp.issubdtype(
            labels.dtype, jnp.integer
        ):
            raise ValueError("NCA requires one integer label per sample.")
        x = batch.dense_features()
        if jnp.iscomplexobj(x):
            raise TypeError("NCA currently supports real feature geometry only.")
        compute_dtype = jnp.result_type(
            x.dtype, self.learning_rate.dtype, self.temperature.dtype, self.ridge.dtype
        )
        x = jnp.asarray(x, dtype=compute_dtype)
        feature_valid = jnp.all(jnp.isfinite(x), axis=-1)
        x = jnp.where(jnp.isfinite(x), x, 0.0)
        weights = (
            validated_weights(batch.effective_weight(self.weight_policy)) * feature_valid
        )
        if batch.target_mask is not None:
            weights = weights * batch.target_mask
        labels = jnp.where(weights > 0, labels, 0)
        components = (
            batch.feature_count if self.component_count is None else self.component_count
        )
        if components > batch.feature_count:
            raise ValueError("component_count cannot exceed feature_count.")
        cases = size(batch.case_shape)
        x_cases = x.reshape((cases, batch.sample_count, batch.feature_count))
        y_cases = labels.reshape((cases, batch.sample_count))
        w_cases = weights.reshape((cases, batch.sample_count))
        initial = jnp.eye(components, batch.feature_count, dtype=x.dtype)

        def fit_one(points, target, weight):
            same = target[:, None] == target[None, :]
            active = weight > 0
            pair_mask = (
                active[:, None] & active[None, :] & ~jnp.eye(points.shape[0], dtype=bool)
            )

            def objective(factor):
                embedded = points @ factor.T
                delta = embedded[:, None, :] - embedded[None, :, :]
                distance = jnp.sum(delta * delta, axis=-1)
                logits = jnp.where(
                    pair_mask,
                    -distance / self.temperature
                    + jnp.log(jnp.maximum(weight[None, :], jnp.finfo(weight.dtype).tiny)),
                    -jnp.inf,
                )
                probability = masked_softmax(logits, pair_mask)
                success = jnp.sum(jnp.where(same, probability, 0.0), axis=-1)
                data_loss = -jnp.sum(
                    weight * jnp.log(jnp.maximum(success, jnp.finfo(weight.dtype).tiny))
                ) / jnp.maximum(jnp.sum(weight), 1.0)
                return data_loss + self.ridge * jnp.sum(factor * factor)

            gradient = jax.grad(objective)
            step_size = jnp.asarray(self.learning_rate, dtype=initial.dtype)
            factor = jax.lax.fori_loop(
                0,
                self.iterations,
                lambda _, current: current - step_size * gradient(current),
                initial,
            )
            return factor, objective(factor)

        factor, objective = jax.vmap(fit_one)(x_cases, y_cases, w_cases)
        factor = factor.reshape(batch.case_shape + (components, batch.feature_count))
        objective = objective.reshape(batch.case_shape)
        effective = jnp.sum(weights > 0, axis=-1)
        finite = jnp.all(jnp.isfinite(factor), axis=(-2, -1)) & jnp.isfinite(objective)
        valid = finite & (effective > 1)
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective > 1, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = LinearMetricModel(
            factor=factor,
            feature_count=batch.feature_count,
            component_count=components,
            case_shape=batch.case_shape,
            method="neighborhood-components-analysis",
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=self.iterations,
            effective_samples=effective,
            rank=jnp.linalg.matrix_rank(factor),
            method="unrolled-neighborhood-components",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="smooth",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="smooth",
            fit_mode="unrolled",
            conditions=("Discrete labels and active sample mask are fixed.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="neighborhood-components-analysis",
            gradient_contract=contract,
        )


class MahalanobisMetricRecipe(AbstractRecipe):
    ridge: Array
    component_count: int | None = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        ridge: ArrayLike = 1e-6,
        component_count: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        ridge_ = jnp.asarray(ridge, dtype=float)
        if ridge_.ndim != 0:
            raise ValueError("ridge must be scalar.")
        self.ridge = eqx.error_if(
            ridge_,
            ~jnp.isfinite(ridge_) | (ridge_ <= 0.0),
            "ridge must be finite and positive.",
        )
        if component_count is not None and int(component_count) <= 0:
            raise ValueError("component_count must be positive.")
        self.component_count = None if component_count is None else int(component_count)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x = batch.dense_features()
        if jnp.iscomplexobj(x):
            raise TypeError(
                "Mahalanobis metric learning currently supports real features only."
            )
        feature_valid = jnp.all(jnp.isfinite(x), axis=-1)
        x = jnp.where(jnp.isfinite(x), x, 0.0)
        weights = (
            validated_weights(batch.effective_weight(self.weight_policy)) * feature_valid
        )
        labels = batch.targets
        total = jnp.sum(weights, axis=-1, keepdims=True)
        if labels is None:
            mean = jnp.sum(weights[..., :, None] * x, axis=-2) / jnp.maximum(
                total, jnp.finfo(weights.dtype).tiny
            )
            centered = x - mean[..., None, :]
        else:
            if labels.shape != batch.case_shape + (
                batch.sample_count,
            ) or not jnp.issubdtype(labels.dtype, jnp.integer):
                raise ValueError(
                    "Supervised Mahalanobis fitting requires scalar integer labels."
                )
            if batch.target_mask is not None:
                weights = weights * batch.target_mask
                total = jnp.sum(weights, axis=-1, keepdims=True)
            labels = jnp.where(weights > 0, labels, 0)
            same = labels[..., :, None] == labels[..., None, :]
            class_weight = jnp.sum(weights[..., None, :] * same, axis=-1)
            class_mean = jnp.einsum(
                "...ij,...jf->...if", weights[..., None, :] * same, x
            ) / jnp.maximum(class_weight[..., :, None], jnp.finfo(weights.dtype).tiny)
            centered = x - class_mean
        covariance = jnp.einsum(
            "...n,...nf,...ng->...fg", weights, centered, centered
        ) / jnp.maximum(total[..., None], 1.0)
        covariance = covariance + self.ridge * jnp.eye(batch.feature_count, dtype=x.dtype)
        values, vectors = jnp.linalg.eigh(covariance)
        components = (
            batch.feature_count if self.component_count is None else self.component_count
        )
        if components > batch.feature_count:
            raise ValueError("component_count cannot exceed feature_count.")
        selected_values = values[..., :components]
        selected_vectors = vectors[..., :, :components]
        factor = jnp.swapaxes(
            selected_vectors / jnp.sqrt(selected_values)[..., None, :], -1, -2
        )
        effective = jnp.sum(weights > 0, axis=-1)
        finite = jnp.all(jnp.isfinite(factor), axis=(-2, -1))
        valid = finite & (effective > 1)
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective > 1, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = LinearMetricModel(
            factor=factor,
            feature_count=batch.feature_count,
            component_count=components,
            case_shape=batch.case_shape,
            method="mahalanobis-whitening",
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=effective,
            rank=jnp.linalg.matrix_rank(covariance),
            condition=jnp.max(values, axis=-1) / jnp.min(values, axis=-1),
            method="weighted-covariance-eigh",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="spectral",
            conditions=("Selected eigenspace is separated and labels are fixed.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="mahalanobis-metric",
            gradient_contract=contract,
        )


__all__ = [
    "LinearMetricModel",
    "MahalanobisMetricRecipe",
    "NeighborhoodComponentsAnalysisRecipe",
]
