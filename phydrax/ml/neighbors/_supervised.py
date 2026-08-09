#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar, Literal

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
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    ML_SUCCESS,
)
from ._utils import (
    broadcast_support,
    case_distances,
    chunked_call,
    gather_support,
    masked_softmax,
    pad_support,
    validate_metric,
    validated_weights,
)


def _positive_scalar(value: ArrayLike, /, *, name: str) -> Array:
    scalar = jnp.asarray(value, dtype=float)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        scalar,
        ~jnp.isfinite(scalar) | (scalar <= 0.0),
        f"{name} must be finite and positive.",
    )


def _model_support(
    support: ArrayLike,
    support_weight: ArrayLike,
    support_mask: ArrayLike,
    feature_count: int,
    case_shape: tuple[int, ...],
) -> tuple[Array, Array, Array, int, tuple[int, ...]]:
    values = jnp.asarray(support)
    weight = jnp.asarray(support_weight)
    mask = jnp.asarray(support_mask, dtype=bool)
    features = int(feature_count)
    cases = tuple(int(value) for value in case_shape)
    if (
        values.ndim != len(cases) + 2
        or values.shape[: len(cases)] != cases
        or values.shape[-1] != features
    ):
        raise ValueError("Support must have shape case_shape + (support, feature_count).")
    support_shape = cases + (values.shape[-2],)
    if weight.shape != support_shape or mask.shape != support_shape:
        raise ValueError(
            "Support weights and masks must have shape case_shape + (support,)."
        )
    return values, weight, mask, features, cases


def _target_width(output_shape: tuple[int, ...]) -> int:
    width = 1
    for dimension in output_shape:
        width *= int(dimension)
    return width


class ExactNeighborRegressorModel(AbstractArrayModel):
    support: Array
    targets: Array
    support_weight: Array
    support_mask: Array
    metric: Any
    neighbor_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        targets: ArrayLike,
        support_weight: ArrayLike,
        support_mask: ArrayLike,
        metric: Any,
        neighbor_count: int,
        feature_count: int,
        output_shape: tuple[int, ...],
        case_shape: tuple[int, ...],
    ):
        support_, weight_, mask_, features, cases = _model_support(
            support, support_weight, support_mask, feature_count, case_shape
        )
        targets_ = jnp.asarray(targets)
        outputs = tuple(int(value) for value in output_shape)
        if targets_.shape != cases + (support_.shape[-2], _target_width(outputs)):
            raise ValueError("Flattened targets must align with case and support axes.")
        neighbors = int(neighbor_count)
        if neighbors <= 0 or neighbors > support_.shape[-2]:
            raise ValueError(
                "neighbor_count must be positive and cannot exceed support capacity."
            )
        self.support = support_
        self.targets = targets_
        self.support_weight = weight_
        self.support_mask = mask_
        self.metric = validate_metric(metric)
        self.neighbor_count = neighbors
        self.feature_count = features
        self.output_shape = outputs
        self.case_shape = cases
        self.in_size = features
        self.out_size = "scalar" if not outputs else outputs

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def neighbor_indices(self, x: ArrayLike, /) -> tuple[Array, Array]:
        distances, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, self.metric
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape)
        safe = jnp.where(mask, distances, jnp.inf)
        negative, indices = jax.lax.top_k(-safe, self.neighbor_count)
        return indices, -negative

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        indices, distances = self.neighbor_indices(x)
        values = gather_support(self.targets, indices, self.case_shape)
        weight = gather_support(self.support_weight[..., None], indices, self.case_shape)[
            ..., 0
        ]
        weight = jnp.where(jnp.isfinite(distances), weight, 0.0)
        denominator = jnp.sum(weight, axis=-1)
        prediction = (
            jnp.sum(weight[..., None] * values, axis=-2)
            / jnp.where(denominator > 0, denominator, 1.0)[..., None]
        )
        prediction = jnp.where((denominator > 0)[..., None], prediction, jnp.nan)
        if not self.output_shape:
            return prediction[..., 0]
        return prediction.reshape(prediction.shape[:-1] + self.output_shape)

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class ExactNeighborClassifierModel(AbstractArrayModel):
    support: Array
    labels: Array
    support_weight: Array
    support_mask: Array
    metric: Any
    neighbor_count: int = eqx.field(static=True)
    class_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        labels: ArrayLike,
        support_weight: ArrayLike,
        support_mask: ArrayLike,
        metric: Any,
        neighbor_count: int,
        class_count: int,
        feature_count: int,
        case_shape: tuple[int, ...],
    ):
        support_, weight_, mask_, features, cases = _model_support(
            support, support_weight, support_mask, feature_count, case_shape
        )
        labels_ = jnp.asarray(labels, dtype=jnp.int32)
        if labels_.shape != cases + (support_.shape[-2],):
            raise ValueError("Labels must align with case and support axes.")
        neighbors = int(neighbor_count)
        classes = int(class_count)
        if neighbors <= 0 or neighbors > support_.shape[-2]:
            raise ValueError(
                "neighbor_count must be positive and cannot exceed support capacity."
            )
        if classes < 2:
            raise ValueError("class_count must be at least two.")
        self.support = support_
        self.labels = labels_
        self.support_weight = weight_
        self.support_mask = mask_
        self.metric = validate_metric(metric)
        self.neighbor_count = neighbors
        self.class_count = classes
        self.feature_count = features
        self.case_shape = cases
        self.in_size = features
        self.out_size = classes

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def probabilities(self, x: ArrayLike, /) -> Array:
        distances, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, self.metric
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape)
        negative, indices = jax.lax.top_k(
            -jnp.where(mask, distances, jnp.inf), self.neighbor_count
        )
        labels = gather_support(self.labels[..., None], indices, self.case_shape)[..., 0]
        weight = gather_support(self.support_weight[..., None], indices, self.case_shape)[
            ..., 0
        ]
        weight = jnp.where(jnp.isfinite(-negative), weight, 0.0)
        scores = jnp.sum(
            weight[..., None] * jax.nn.one_hot(labels, self.class_count), axis=-2
        )
        total = jnp.sum(scores, axis=-1, keepdims=True)
        return jnp.where(total > 0, scores / jnp.where(total > 0, total, 1.0), 0.0)

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        return self.probabilities(x)

    def predict(self, x: ArrayLike, /) -> Array:
        probability = self.probabilities(x)
        valid = jnp.sum(probability, axis=-1) > 0
        return jnp.where(valid, jnp.argmax(probability, axis=-1), -1).astype(jnp.int32)

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class KernelNeighborRegressorModel(AbstractArrayModel):
    """Smooth all-support kernel weighting, distinct from hard top-k selection."""

    support: Array
    targets: Array
    support_weight: Array
    support_mask: Array
    metric: Any
    neighbor_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    temperature: Array
    in_size: int = eqx.field(static=True)
    out_size: tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        targets: ArrayLike,
        support_weight: ArrayLike,
        support_mask: ArrayLike,
        metric: Any,
        neighbor_count: int,
        feature_count: int,
        output_shape: tuple[int, ...],
        case_shape: tuple[int, ...],
        temperature: ArrayLike,
    ):
        support_, weight_, mask_, features, cases = _model_support(
            support, support_weight, support_mask, feature_count, case_shape
        )
        targets_ = jnp.asarray(targets)
        outputs = tuple(int(value) for value in output_shape)
        if targets_.shape != cases + (support_.shape[-2], _target_width(outputs)):
            raise ValueError("Flattened targets must align with case and support axes.")
        neighbors = int(neighbor_count)
        if neighbors <= 0:
            raise ValueError("neighbor_count must be positive.")
        self.support = support_
        self.targets = targets_
        self.support_weight = weight_
        self.support_mask = mask_
        self.metric = validate_metric(metric)
        self.neighbor_count = neighbors
        self.feature_count = features
        self.output_shape = outputs
        self.case_shape = cases
        self.temperature = _positive_scalar(temperature, name="temperature")
        self.in_size = features
        self.out_size = "scalar" if not outputs else outputs

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def weights(self, x: ArrayLike, /) -> Array:
        distances, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, self.metric
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape)
        support_weight = broadcast_support(
            self.support_weight, len(query_shape), self.case_shape
        )
        active = mask & (support_weight > 0)
        logits = -distances / self.temperature + jnp.log(
            jnp.maximum(support_weight, jnp.finfo(float).tiny)
        )
        return masked_softmax(logits, active)

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        weight = self.weights(x)
        query_ndim = weight.ndim - len(self.case_shape) - 1
        targets = broadcast_support(self.targets, query_ndim, self.case_shape)
        prediction = jnp.sum(weight[..., None] * targets, axis=-2)
        if not self.output_shape:
            return prediction[..., 0]
        return prediction.reshape(prediction.shape[:-1] + self.output_shape)

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class KernelNeighborClassifierModel(AbstractArrayModel):
    """Smooth all-support class probabilities with no hard top-k operation."""

    support: Array
    labels: Array
    support_weight: Array
    support_mask: Array
    metric: Any
    neighbor_count: int = eqx.field(static=True)
    class_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    temperature: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        labels: ArrayLike,
        support_weight: ArrayLike,
        support_mask: ArrayLike,
        metric: Any,
        neighbor_count: int,
        class_count: int,
        feature_count: int,
        case_shape: tuple[int, ...],
        temperature: ArrayLike,
    ):
        support_, weight_, mask_, features, cases = _model_support(
            support, support_weight, support_mask, feature_count, case_shape
        )
        labels_ = jnp.asarray(labels, dtype=jnp.int32)
        if labels_.shape != cases + (support_.shape[-2],):
            raise ValueError("Labels must align with case and support axes.")
        neighbors = int(neighbor_count)
        classes = int(class_count)
        if neighbors <= 0:
            raise ValueError("neighbor_count must be positive.")
        if classes < 2:
            raise ValueError("class_count must be at least two.")
        self.support = support_
        self.labels = labels_
        self.support_weight = weight_
        self.support_mask = mask_
        self.metric = validate_metric(metric)
        self.neighbor_count = neighbors
        self.class_count = classes
        self.feature_count = features
        self.case_shape = cases
        self.temperature = _positive_scalar(temperature, name="temperature")
        self.in_size = features
        self.out_size = classes

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def probabilities(self, x: ArrayLike, /) -> Array:
        distances, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, self.metric
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape)
        support_weight = broadcast_support(
            self.support_weight, len(query_shape), self.case_shape
        )
        active = mask & (support_weight > 0)
        logits = -distances / self.temperature + jnp.log(
            jnp.maximum(support_weight, jnp.finfo(float).tiny)
        )
        weight = masked_softmax(logits, active)
        labels = broadcast_support(self.labels, len(query_shape), self.case_shape)
        return jnp.sum(
            weight[..., None] * jax.nn.one_hot(labels, self.class_count), axis=-2
        )

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        return self.probabilities(x)

    def predict(self, x: ArrayLike, /) -> Array:
        probability = self.probabilities(x)
        valid = jnp.sum(probability, axis=-1) > 0
        return jnp.where(valid, jnp.argmax(probability, axis=-1), -1).astype(jnp.int32)

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class RadiusNeighborRegressorModel(AbstractArrayModel):
    support: Array
    targets: Array
    support_weight: Array
    support_mask: Array
    metric: Any
    neighbor_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    radius: Array
    in_size: int = eqx.field(static=True)
    out_size: tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        targets: ArrayLike,
        support_weight: ArrayLike,
        support_mask: ArrayLike,
        metric: Any,
        neighbor_count: int,
        feature_count: int,
        output_shape: tuple[int, ...],
        case_shape: tuple[int, ...],
        radius: ArrayLike,
    ):
        support_, weight_, mask_, features, cases = _model_support(
            support, support_weight, support_mask, feature_count, case_shape
        )
        targets_ = jnp.asarray(targets)
        outputs = tuple(int(value) for value in output_shape)
        if targets_.shape != cases + (support_.shape[-2], _target_width(outputs)):
            raise ValueError("Flattened targets must align with case and support axes.")
        neighbors = int(neighbor_count)
        if neighbors <= 0:
            raise ValueError("neighbor_count must be positive.")
        self.support = support_
        self.targets = targets_
        self.support_weight = weight_
        self.support_mask = mask_
        self.metric = validate_metric(metric)
        self.neighbor_count = neighbors
        self.feature_count = features
        self.output_shape = outputs
        self.case_shape = cases
        self.radius = _positive_scalar(radius, name="radius")
        self.in_size = features
        self.out_size = "scalar" if not outputs else outputs

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        distances, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, self.metric
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape) & (
            distances <= self.radius
        )
        weight = (
            broadcast_support(self.support_weight, len(query_shape), self.case_shape)
            * mask
        )
        targets = broadcast_support(self.targets, len(query_shape), self.case_shape)
        denominator = jnp.sum(weight, axis=-1)
        prediction = (
            jnp.sum(weight[..., None] * targets, axis=-2)
            / jnp.where(denominator > 0, denominator, 1.0)[..., None]
        )
        prediction = jnp.where((denominator > 0)[..., None], prediction, jnp.nan)
        if not self.output_shape:
            return prediction[..., 0]
        return prediction.reshape(prediction.shape[:-1] + self.output_shape)

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class RadiusNeighborClassifierModel(AbstractArrayModel):
    support: Array
    labels: Array
    support_weight: Array
    support_mask: Array
    metric: Any
    neighbor_count: int = eqx.field(static=True)
    class_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    radius: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        labels: ArrayLike,
        support_weight: ArrayLike,
        support_mask: ArrayLike,
        metric: Any,
        neighbor_count: int,
        class_count: int,
        feature_count: int,
        case_shape: tuple[int, ...],
        radius: ArrayLike,
    ):
        support_, weight_, mask_, features, cases = _model_support(
            support, support_weight, support_mask, feature_count, case_shape
        )
        labels_ = jnp.asarray(labels, dtype=jnp.int32)
        if labels_.shape != cases + (support_.shape[-2],):
            raise ValueError("Labels must align with case and support axes.")
        neighbors = int(neighbor_count)
        classes = int(class_count)
        if neighbors <= 0:
            raise ValueError("neighbor_count must be positive.")
        if classes < 2:
            raise ValueError("class_count must be at least two.")
        self.support = support_
        self.labels = labels_
        self.support_weight = weight_
        self.support_mask = mask_
        self.metric = validate_metric(metric)
        self.neighbor_count = neighbors
        self.class_count = classes
        self.feature_count = features
        self.case_shape = cases
        self.radius = _positive_scalar(radius, name="radius")
        self.in_size = features
        self.out_size = classes

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def probabilities(self, x: ArrayLike, /) -> Array:
        distances, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, self.metric
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape) & (
            distances <= self.radius
        )
        weight = (
            broadcast_support(self.support_weight, len(query_shape), self.case_shape)
            * mask
        )
        labels = broadcast_support(self.labels, len(query_shape), self.case_shape)
        scores = jnp.sum(
            weight[..., None] * jax.nn.one_hot(labels, self.class_count), axis=-2
        )
        total = jnp.sum(scores, axis=-1, keepdims=True)
        return jnp.where(total > 0, scores / jnp.where(total > 0, total, 1.0), 0.0)

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        return self.probabilities(x)

    def predict(self, x: ArrayLike, /) -> Array:
        probability = self.probabilities(x)
        valid = jnp.sum(probability, axis=-1) > 0
        return jnp.where(valid, jnp.argmax(probability, axis=-1), -1).astype(jnp.int32)

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class NearestCentroidModel(AbstractArrayModel):
    centroids: Array
    class_mask: Array
    metric: Any
    feature_count: int = eqx.field(static=True)
    class_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    temperature: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        centroids: ArrayLike,
        class_mask: ArrayLike,
        metric: Any,
        feature_count: int,
        class_count: int,
        case_shape: tuple[int, ...],
        temperature: ArrayLike,
    ):
        centroids_ = jnp.asarray(centroids)
        mask_ = jnp.asarray(class_mask, dtype=bool)
        features = int(feature_count)
        classes = int(class_count)
        cases = tuple(int(value) for value in case_shape)
        if centroids_.shape != cases + (classes, features):
            raise ValueError(
                "Centroids must have shape case_shape + (class_count, feature_count)."
            )
        if mask_.shape != cases + (classes,):
            raise ValueError("Class masks must have shape case_shape + (class_count,).")
        if classes < 2:
            raise ValueError("class_count must be at least two.")
        self.centroids = centroids_
        self.class_mask = mask_
        self.metric = validate_metric(metric)
        self.feature_count = features
        self.class_count = classes
        self.case_shape = cases
        self.temperature = _positive_scalar(temperature, name="temperature")
        self.in_size = features
        self.out_size = classes

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def probabilities(self, x: ArrayLike, /) -> Array:
        distances, query_shape = case_distances(
            jnp.asarray(x), self.centroids, self.case_shape, self.metric
        )
        mask = broadcast_support(self.class_mask, len(query_shape), self.case_shape)
        return masked_softmax(-distances / self.temperature, mask)

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        return self.probabilities(x)

    def predict(self, x: ArrayLike, /) -> Array:
        probability = self.probabilities(x)
        valid = jnp.sum(probability, axis=-1) > 0
        return jnp.where(valid, jnp.argmax(probability, axis=-1), -1).astype(jnp.int32)


def _prepare_support(batch: MLBatch, capacity: int | None, policy: WeightPolicy):
    cap = batch.sample_count if capacity is None else int(capacity)
    if cap <= 0:
        raise ValueError("capacity must be positive.")
    axis = len(batch.case_shape)
    x = pad_support(batch.dense_features(), cap, axis, 0.0)
    mask = pad_support(batch.sample_mask, cap, axis, False)
    weight = pad_support(
        validated_weights(batch.effective_weight(policy)), cap, axis, 0.0
    )
    feature_valid = jnp.all(
        jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), axis=-1
    )
    x = jnp.where(jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), x, 0.0)
    return x, mask & feature_valid & (weight > 0), weight, cap


def _fit_status(batch: MLBatch, support_mask: Array, capacity: int, minimum_support: int):
    effective = jnp.sum(support_mask, axis=-1)
    exhausted = capacity < batch.sample_count
    sufficient = effective >= minimum_support
    status = jnp.where(
        sufficient,
        ML_CAPACITY_EXHAUSTED if exhausted else ML_SUCCESS,
        ML_INSUFFICIENT_DATA,
    )
    valid = sufficient & (not exhausted)
    return valid, status, effective


def _diagnostics_method(diagnostics: FitDiagnostics, method: str) -> FitDiagnostics:
    return FitDiagnostics(
        valid=diagnostics.valid,
        status=diagnostics.status,
        objective=diagnostics.objective,
        iterations=diagnostics.iterations,
        effective_samples=diagnostics.effective_samples,
        rank=diagnostics.rank,
        condition=diagnostics.condition,
        method=method,
    )


class KNeighborsRegressorRecipe(AbstractRecipe):
    neighbor_count: int = eqx.field(static=True)
    metric: Any
    capacity: int | None = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        neighbor_count: int = 5,
        /,
        *,
        metric: Any = "euclidean",
        capacity: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(neighbor_count) <= 0:
            raise ValueError("neighbor_count must be positive.")
        self.neighbor_count = int(neighbor_count)
        self.metric = validate_metric(metric)
        self.capacity = None if capacity is None else int(capacity)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        target = batch.require_targets()
        x, support_mask, weight, capacity = _prepare_support(
            batch, self.capacity, self.weight_policy
        )
        if self.neighbor_count > capacity:
            raise ValueError("neighbor_count cannot exceed fixed support capacity.")
        sample_shape = batch.case_shape + (batch.sample_count,)
        output_shape = tuple(int(s) for s in target.shape[len(sample_shape) :])
        target_flat = target.reshape(sample_shape + (-1,))
        target_flat = pad_support(target_flat, capacity, len(batch.case_shape), 0.0)
        target_finite = jnp.all(
            jnp.isfinite(jnp.real(target_flat)) & jnp.isfinite(jnp.imag(target_flat)),
            axis=-1,
        )
        support_mask = support_mask & target_finite
        if batch.target_mask is not None:
            target_valid = jnp.all(
                batch.target_mask.reshape(sample_shape + (-1,)), axis=-1
            )
            support_mask = support_mask & pad_support(
                target_valid, capacity, len(batch.case_shape), False
            )
        model = ExactNeighborRegressorModel(
            support=x,
            targets=target_flat,
            support_weight=weight,
            support_mask=support_mask,
            metric=self.metric,
            neighbor_count=self.neighbor_count,
            feature_count=batch.feature_count,
            output_shape=output_shape,
            case_shape=batch.case_shape,
        )
        valid, status, effective = _fit_status(
            batch, support_mask, capacity, self.neighbor_count
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=effective,
            method="exact-top-k-neighbors",
        )
        contract = GradientContract(
            prediction_inputs="almost-everywhere",
            prediction_parameters="almost-everywhere",
            fit_mode="stopped",
            nondifferentiable_outputs=("neighbor_indices",),
            conditions=("Top-k ordering is locally constant and tie-free.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="knn-regression",
            gradient_contract=contract,
        )


class KNeighborsClassifierRecipe(AbstractRecipe):
    neighbor_count: int = eqx.field(static=True)
    class_count: int = eqx.field(static=True)
    metric: Any
    capacity: int | None = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        neighbor_count: int = 5,
        /,
        *,
        class_count: int,
        metric: Any = "euclidean",
        capacity: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(neighbor_count) <= 0 or int(class_count) < 2:
            raise ValueError(
                "neighbor_count must be positive and class_count at least two."
            )
        self.neighbor_count = int(neighbor_count)
        self.class_count = int(class_count)
        self.metric = validate_metric(metric)
        self.capacity = None if capacity is None else int(capacity)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        labels = batch.require_targets()
        if labels.shape != batch.case_shape + (batch.sample_count,) or not jnp.issubdtype(
            labels.dtype, jnp.integer
        ):
            raise ValueError(
                "Neighbor classification requires one integer label per sample."
            )
        x, support_mask, weight, capacity = _prepare_support(
            batch, self.capacity, self.weight_policy
        )
        if self.neighbor_count > capacity:
            raise ValueError("neighbor_count cannot exceed fixed support capacity.")
        labels = pad_support(labels.astype(jnp.int32), capacity, len(batch.case_shape), 0)
        if batch.target_mask is not None:
            support_mask = support_mask & pad_support(
                batch.target_mask, capacity, len(batch.case_shape), False
            )
        support_mask = support_mask & (labels >= 0) & (labels < self.class_count)
        model = ExactNeighborClassifierModel(
            support=x,
            labels=labels,
            support_weight=weight,
            support_mask=support_mask,
            metric=self.metric,
            neighbor_count=self.neighbor_count,
            class_count=self.class_count,
            feature_count=batch.feature_count,
            case_shape=batch.case_shape,
        )
        valid, status, effective = _fit_status(
            batch, support_mask, capacity, self.neighbor_count
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=effective,
            method="exact-top-k-neighbors",
        )
        contract = GradientContract(
            prediction_inputs="almost-everywhere",
            prediction_parameters="almost-everywhere",
            fit_targets="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("neighbor_indices", "predict"),
            conditions=("Top-k ordering is locally constant and tie-free.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="knn-classification",
            gradient_contract=contract,
        )


class KernelNeighborsRegressorRecipe(AbstractRecipe):
    recipe: KNeighborsRegressorRecipe
    temperature: Array

    def __init__(
        self,
        /,
        *,
        temperature: ArrayLike = 1.0,
        metric: Any = "squared-euclidean",
        capacity: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.recipe = KNeighborsRegressorRecipe(
            1, metric=metric, capacity=capacity, weight_policy=weight_policy
        )
        self.temperature = _positive_scalar(temperature, name="temperature")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        result = self.recipe.fit_batch(batch, key=key)
        raw = result.as_trainable()
        model = KernelNeighborRegressorModel(
            support=raw.support,
            targets=raw.targets,
            support_weight=raw.support_weight,
            support_mask=raw.support_mask,
            metric=raw.metric,
            neighbor_count=raw.neighbor_count,
            feature_count=raw.feature_count,
            output_shape=raw.output_shape,
            case_shape=raw.case_shape,
            temperature=self.temperature,
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="smooth",
            fit_targets="smooth",
            fit_weights="smooth",
            fit_hyperparameters="smooth",
            fit_mode="relaxed",
            conditions=("At least one positive support weight.",),
        )
        return FitResult(
            model,
            _diagnostics_method(result.diagnostics, "soft-kernel-neighbors"),
            valid=result.valid,
            status=result.status,
            method="kernel-neighbor-regression",
            gradient_contract=contract,
        )


class KernelNeighborsClassifierRecipe(AbstractRecipe):
    recipe: KNeighborsClassifierRecipe
    temperature: Array

    def __init__(
        self,
        /,
        *,
        class_count: int,
        temperature: ArrayLike = 1.0,
        metric: Any = "squared-euclidean",
        capacity: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.recipe = KNeighborsClassifierRecipe(
            1,
            class_count=class_count,
            metric=metric,
            capacity=capacity,
            weight_policy=weight_policy,
        )
        self.temperature = _positive_scalar(temperature, name="temperature")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        result = self.recipe.fit_batch(batch, key=key)
        raw = result.as_trainable()
        model = KernelNeighborClassifierModel(
            support=raw.support,
            labels=raw.labels,
            support_weight=raw.support_weight,
            support_mask=raw.support_mask,
            metric=raw.metric,
            neighbor_count=raw.neighbor_count,
            class_count=raw.class_count,
            feature_count=raw.feature_count,
            case_shape=raw.case_shape,
            temperature=self.temperature,
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="smooth",
            fit_targets="none",
            fit_weights="smooth",
            fit_hyperparameters="smooth",
            fit_mode="relaxed",
            nondifferentiable_outputs=("predict",),
            conditions=(
                "Class labels are fixed and at least one support weight is positive.",
            ),
        )
        return FitResult(
            model,
            _diagnostics_method(result.diagnostics, "soft-kernel-neighbors"),
            valid=result.valid,
            status=result.status,
            method="kernel-neighbor-classification",
            gradient_contract=contract,
        )


class RadiusNeighborsRegressorRecipe(AbstractRecipe):
    recipe: KNeighborsRegressorRecipe
    radius: Array

    def __init__(
        self,
        radius: ArrayLike,
        /,
        *,
        metric: Any = "euclidean",
        capacity: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.recipe = KNeighborsRegressorRecipe(
            1, metric=metric, capacity=capacity, weight_policy=weight_policy
        )
        self.radius = _positive_scalar(radius, name="radius")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        result = self.recipe.fit_batch(batch, key=key)
        raw = result.as_trainable()
        model = RadiusNeighborRegressorModel(
            support=raw.support,
            targets=raw.targets,
            support_weight=raw.support_weight,
            support_mask=raw.support_mask,
            metric=raw.metric,
            neighbor_count=raw.neighbor_count,
            feature_count=raw.feature_count,
            output_shape=raw.output_shape,
            case_shape=raw.case_shape,
            radius=self.radius,
        )
        contract = GradientContract(
            prediction_inputs="almost-everywhere",
            prediction_parameters="almost-everywhere",
            fit_mode="stopped",
            nondifferentiable_outputs=("radius_membership",),
            conditions=("No distance lies on the radius boundary.",),
        )
        return FitResult(
            model,
            _diagnostics_method(result.diagnostics, "hard-radius-neighbors"),
            valid=result.valid,
            status=result.status,
            method="radius-neighbor-regression",
            gradient_contract=contract,
        )


class RadiusNeighborsClassifierRecipe(AbstractRecipe):
    recipe: KNeighborsClassifierRecipe
    radius: Array

    def __init__(
        self,
        radius: ArrayLike,
        /,
        *,
        class_count: int,
        metric: Any = "euclidean",
        capacity: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.recipe = KNeighborsClassifierRecipe(
            1,
            class_count=class_count,
            metric=metric,
            capacity=capacity,
            weight_policy=weight_policy,
        )
        self.radius = _positive_scalar(radius, name="radius")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        result = self.recipe.fit_batch(batch, key=key)
        raw = result.as_trainable()
        model = RadiusNeighborClassifierModel(
            support=raw.support,
            labels=raw.labels,
            support_weight=raw.support_weight,
            support_mask=raw.support_mask,
            metric=raw.metric,
            neighbor_count=raw.neighbor_count,
            class_count=raw.class_count,
            feature_count=raw.feature_count,
            case_shape=raw.case_shape,
            radius=self.radius,
        )
        contract = GradientContract(
            prediction_inputs="almost-everywhere",
            prediction_parameters="almost-everywhere",
            fit_targets="none",
            fit_mode="stopped",
            nondifferentiable_outputs=("radius_membership", "predict"),
            conditions=("No distance lies on the radius boundary.",),
        )
        return FitResult(
            model,
            _diagnostics_method(result.diagnostics, "hard-radius-neighbors"),
            valid=result.valid,
            status=result.status,
            method="radius-neighbor-classification",
            gradient_contract=contract,
        )


class NearestCentroidRecipe(AbstractRecipe):
    class_count: int = eqx.field(static=True)
    metric: Any
    temperature: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        class_count: int,
        metric: Any = "squared-euclidean",
        temperature: ArrayLike = 1.0,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(class_count) < 2:
            raise ValueError("class_count must be at least two.")
        self.class_count = int(class_count)
        self.metric = validate_metric(metric)
        self.temperature = _positive_scalar(temperature, name="temperature")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        labels = batch.require_targets()
        if labels.shape != batch.case_shape + (batch.sample_count,) or not jnp.issubdtype(
            labels.dtype, jnp.integer
        ):
            raise ValueError("Nearest centroid requires one integer label per sample.")
        x = batch.dense_features()
        feature_valid = jnp.all(
            jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), axis=-1
        )
        x = jnp.where(jnp.isfinite(jnp.real(x)) & jnp.isfinite(jnp.imag(x)), x, 0.0)
        weights = (
            validated_weights(batch.effective_weight(self.weight_policy)) * feature_valid
        )
        if batch.target_mask is not None:
            weights = weights * batch.target_mask
        labels = jnp.where(weights > 0, labels, 0)
        labels = eqx.error_if(
            labels,
            jnp.any((labels < 0) | (labels >= self.class_count)),
            "Nearest-centroid labels exceed the configured class capacity.",
        )
        one_hot = jax.nn.one_hot(labels, self.class_count)
        class_weight = jnp.sum(weights[..., :, None] * one_hot, axis=-2)
        centroids = jnp.einsum(
            "...nc,...nf->...cf", weights[..., :, None] * one_hot, x
        ) / jnp.maximum(class_weight[..., :, None], jnp.finfo(weights.dtype).tiny)
        class_mask = class_weight > 0
        valid = jnp.sum(class_mask, axis=-1) >= 2
        status = jnp.where(valid, ML_SUCCESS, ML_INSUFFICIENT_DATA)
        model = NearestCentroidModel(
            centroids=centroids,
            class_mask=class_mask,
            metric=self.metric,
            feature_count=batch.feature_count,
            class_count=self.class_count,
            case_shape=batch.case_shape,
            temperature=self.temperature,
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=jnp.sum(weights > 0, axis=-1),
            method="weighted-nearest-centroid",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="smooth",
            fit_targets="none",
            fit_weights="conditional",
            fit_mode="direct",
            nondifferentiable_outputs=("predict",),
            conditions=("Class membership and nonempty classes are fixed.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="nearest-centroid",
            gradient_contract=contract,
        )


__all__ = [
    "ExactNeighborClassifierModel",
    "ExactNeighborRegressorModel",
    "KNeighborsClassifierRecipe",
    "KNeighborsRegressorRecipe",
    "KernelNeighborClassifierModel",
    "KernelNeighborRegressorModel",
    "KernelNeighborsClassifierRecipe",
    "KernelNeighborsRegressorRecipe",
    "NearestCentroidModel",
    "NearestCentroidRecipe",
    "RadiusNeighborClassifierModel",
    "RadiusNeighborRegressorModel",
    "RadiusNeighborsClassifierRecipe",
    "RadiusNeighborsRegressorRecipe",
]
