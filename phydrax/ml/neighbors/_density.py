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
    pad_support,
    validate_metric,
    validated_weights,
)


class KernelDensityModel(AbstractArrayModel):
    support: Array
    support_weight: Array
    support_mask: Array
    bandwidth: Array
    feature_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        support_weight: ArrayLike,
        support_mask: ArrayLike,
        bandwidth: ArrayLike,
        feature_count: int,
        case_shape: tuple[int, ...],
    ):
        support_ = jnp.asarray(support)
        weight_ = jnp.asarray(support_weight)
        mask_ = jnp.asarray(support_mask, dtype=bool)
        features = int(feature_count)
        cases = tuple(int(value) for value in case_shape)
        if (
            support_.ndim != len(cases) + 2
            or support_.shape[: len(cases)] != cases
            or support_.shape[-1] != features
        ):
            raise ValueError(
                "Support must have shape case_shape + (support, feature_count)."
            )
        if weight_.shape != cases + (support_.shape[-2],) or mask_.shape != weight_.shape:
            raise ValueError(
                "Support weights and masks must align with the support axis."
            )
        bandwidth_ = jnp.asarray(bandwidth, dtype=float)
        if bandwidth_.ndim != 0:
            raise ValueError("bandwidth must be scalar.")
        self.support = support_
        self.support_weight = weight_
        self.support_mask = mask_
        self.bandwidth = eqx.error_if(
            bandwidth_,
            ~jnp.isfinite(bandwidth_) | (bandwidth_ <= 0.0),
            "bandwidth must be finite and positive.",
        )
        self.feature_count = features
        self.case_shape = cases
        self.in_size = features
        self.out_size = "scalar"

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def log_density(self, x: ArrayLike, /) -> Array:
        squared, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, "squared-euclidean"
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape)
        weight = broadcast_support(self.support_weight, len(query_shape), self.case_shape)
        active = mask & (weight > 0)
        logits = jnp.log(jnp.maximum(weight, jnp.finfo(float).tiny))
        logits = logits - 0.5 * squared / (self.bandwidth * self.bandwidth)
        any_active = jnp.any(active, axis=-1, keepdims=True)
        safe_logits = jnp.where(active, logits, -jnp.inf)
        safe_logits = jnp.where(any_active, safe_logits, 0.0)
        log_sum = jax.scipy.special.logsumexp(safe_logits, axis=-1)
        total = jnp.sum(jnp.where(self.support_mask, self.support_weight, 0.0), axis=-1)
        total = total.reshape(self.case_shape + (1,) * len(query_shape))
        normalizer = (
            jnp.log(jnp.maximum(total, jnp.finfo(float).tiny))
            + self.feature_count * jnp.log(self.bandwidth)
            + 0.5 * self.feature_count * jnp.log(2.0 * jnp.pi)
        )
        return jnp.where(total > 0, log_sum - normalizer, -jnp.inf)

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        return jnp.exp(self.log_density(x))

    def score_samples(self, x: ArrayLike, /) -> Array:
        return self.log_density(x)

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class KernelDensityRecipe(AbstractRecipe):
    bandwidth: Array
    capacity: int | None = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        bandwidth: ArrayLike = 1.0,
        /,
        *,
        capacity: int | None = None,
        weight_policy: WeightPolicy = "measure",
    ):
        bandwidth_ = jnp.asarray(bandwidth, dtype=float)
        if bandwidth_.ndim != 0:
            raise ValueError("bandwidth must be scalar.")
        self.bandwidth = eqx.error_if(
            bandwidth_,
            ~jnp.isfinite(bandwidth_) | (bandwidth_ <= 0.0),
            "bandwidth must be finite and positive.",
        )
        self.capacity = None if capacity is None else int(capacity)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        capacity = batch.sample_count if self.capacity is None else self.capacity
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        axis = len(batch.case_shape)
        support = pad_support(batch.dense_features(), capacity, axis, 0.0)
        weight = pad_support(
            validated_weights(batch.effective_weight(self.weight_policy)),
            capacity,
            axis,
            0.0,
        )
        feature_valid = jnp.all(
            jnp.isfinite(jnp.real(support)) & jnp.isfinite(jnp.imag(support)),
            axis=-1,
        )
        support = jnp.where(
            jnp.isfinite(jnp.real(support)) & jnp.isfinite(jnp.imag(support)),
            support,
            0.0,
        )
        mask = (
            pad_support(batch.sample_mask, capacity, axis, False)
            & (weight > 0)
            & feature_valid
        )
        effective = jnp.sum(mask, axis=-1)
        exhausted = capacity < batch.sample_count
        valid = (effective > 0) & (not exhausted)
        status = jnp.where(
            effective > 0,
            ML_CAPACITY_EXHAUSTED if exhausted else ML_SUCCESS,
            ML_INSUFFICIENT_DATA,
        )
        model = KernelDensityModel(
            support=support,
            support_weight=weight,
            support_mask=mask,
            bandwidth=self.bandwidth,
            feature_count=batch.feature_count,
            case_shape=batch.case_shape,
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=effective,
            method="gaussian-kernel-density",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="smooth",
            fit_weights="smooth",
            fit_hyperparameters="smooth",
            fit_mode="direct",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="kernel-density",
            gradient_contract=contract,
        )


class LocalOutlierFactorModel(AbstractArrayModel):
    support: Array
    support_mask: Array
    support_weight: Array
    local_reachability_density: Array
    k_distance: Array
    metric: Any
    neighbor_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        support: ArrayLike,
        support_mask: ArrayLike,
        support_weight: ArrayLike,
        local_reachability_density: ArrayLike,
        k_distance: ArrayLike,
        metric: Any,
        neighbor_count: int,
        feature_count: int,
        case_shape: tuple[int, ...],
    ):
        support_ = jnp.asarray(support)
        mask_ = jnp.asarray(support_mask, dtype=bool)
        weight_ = jnp.asarray(support_weight)
        lrd_ = jnp.asarray(local_reachability_density)
        k_distance_ = jnp.asarray(k_distance)
        features = int(feature_count)
        cases = tuple(int(value) for value in case_shape)
        if (
            support_.ndim != len(cases) + 2
            or support_.shape[: len(cases)] != cases
            or support_.shape[-1] != features
        ):
            raise ValueError(
                "Support must have shape case_shape + (support, feature_count)."
            )
        support_shape = cases + (support_.shape[-2],)
        if any(
            value.shape != support_shape for value in (mask_, weight_, lrd_, k_distance_)
        ):
            raise ValueError("LOF support statistics must align with the support axis.")
        neighbors = int(neighbor_count)
        if neighbors <= 0 or neighbors > support_.shape[-2]:
            raise ValueError(
                "neighbor_count must be positive and cannot exceed support capacity."
            )
        self.support = support_
        self.support_mask = mask_
        self.support_weight = weight_
        self.local_reachability_density = lrd_
        self.k_distance = k_distance_
        self.metric = validate_metric(metric)
        self.neighbor_count = neighbors
        self.feature_count = features
        self.case_shape = cases
        self.in_size = features
        self.out_size = "scalar"

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def score_samples(self, x: ArrayLike, /) -> Array:
        distances, query_shape = case_distances(
            jnp.asarray(x), self.support, self.case_shape, self.metric
        )
        mask = broadcast_support(self.support_mask, len(query_shape), self.case_shape)
        negative, indices = jax.lax.top_k(
            -jnp.where(mask, distances, jnp.inf), self.neighbor_count
        )
        selected_distance = -negative
        neighbor_k_distance = gather_support(
            self.k_distance[..., None], indices, self.case_shape
        )[..., 0]
        neighbor_lrd = gather_support(
            self.local_reachability_density[..., None], indices, self.case_shape
        )[..., 0]
        neighbor_weight = gather_support(
            self.support_weight[..., None], indices, self.case_shape
        )[..., 0]
        neighbor_weight = jnp.where(jnp.isfinite(selected_distance), neighbor_weight, 0.0)
        reachability = jnp.where(
            jnp.isfinite(selected_distance),
            jnp.maximum(selected_distance, neighbor_k_distance),
            0.0,
        )
        total_weight = jnp.sum(neighbor_weight, axis=-1)
        mean_reachability = jnp.sum(
            neighbor_weight * reachability, axis=-1
        ) / jnp.maximum(total_weight, jnp.finfo(reachability.dtype).tiny)
        query_lrd = 1.0 / jnp.maximum(
            mean_reachability, jnp.finfo(reachability.dtype).tiny
        )
        mean_neighbor_lrd = jnp.sum(
            neighbor_weight * neighbor_lrd, axis=-1
        ) / jnp.maximum(total_weight, jnp.finfo(reachability.dtype).tiny)
        return mean_neighbor_lrd / query_lrd

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        return self.score_samples(x)

    def predict(self, x: ArrayLike, /, *, threshold: ArrayLike = 1.5) -> Array:
        return jnp.where(self.score_samples(x) <= jnp.asarray(threshold), 1, -1).astype(
            jnp.int32
        )

    def predict_chunked(self, x: ArrayLike, /, *, chunk_size: int) -> Array:
        if self.case_shape:
            raise ValueError("predict_chunked requires an unbatched fitted case.")
        return chunked_call(self, jnp.asarray(x), chunk_size)


class LocalOutlierFactorRecipe(AbstractRecipe):
    neighbor_count: int = eqx.field(static=True)
    metric: Any
    chunk_size: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        neighbor_count: int = 20,
        /,
        *,
        metric: Any = "euclidean",
        chunk_size: int = 128,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(neighbor_count) <= 0 or int(chunk_size) <= 0:
            raise ValueError("neighbor_count and chunk_size must be positive.")
        self.neighbor_count = int(neighbor_count)
        self.metric = validate_metric(metric)
        self.chunk_size = int(chunk_size)
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if self.neighbor_count >= batch.sample_count:
            raise ValueError("neighbor_count must be smaller than sample capacity.")
        support = batch.dense_features()
        support_weight = validated_weights(batch.effective_weight(self.weight_policy))
        feature_valid = jnp.all(
            jnp.isfinite(jnp.real(support)) & jnp.isfinite(jnp.imag(support)),
            axis=-1,
        )
        support = jnp.where(
            jnp.isfinite(jnp.real(support)) & jnp.isfinite(jnp.imag(support)),
            support,
            0.0,
        )
        support_mask = batch.sample_mask & (support_weight > 0) & feature_valid
        index_parts = []
        distance_parts = []
        sample_axis = len(batch.case_shape)
        for start in range(0, batch.sample_count, self.chunk_size):
            stop = min(start + self.chunk_size, batch.sample_count)
            query = jnp.take(support, jnp.arange(start, stop), axis=sample_axis)
            distances, query_shape = case_distances(
                query, support, batch.case_shape, self.metric
            )
            mask = broadcast_support(support_mask, len(query_shape), batch.case_shape)
            local_rows = jnp.arange(stop - start)
            global_rows = jnp.arange(start, stop)
            distances = distances.at[(..., local_rows, global_rows)].set(jnp.inf)
            negative, indices = jax.lax.top_k(
                -jnp.where(mask, distances, jnp.inf), self.neighbor_count
            )
            index_parts.append(indices)
            distance_parts.append(-negative)
        indices = jnp.concatenate(tuple(index_parts), axis=sample_axis)
        distances = jnp.concatenate(tuple(distance_parts), axis=sample_axis)
        k_distance = distances[..., -1]
        neighbor_k = gather_support(k_distance[..., None], indices, batch.case_shape)[
            ..., 0
        ]
        neighbor_weight = gather_support(
            support_weight[..., None], indices, batch.case_shape
        )[..., 0]
        neighbor_weight = jnp.where(jnp.isfinite(distances), neighbor_weight, 0.0)
        reachability = jnp.where(
            jnp.isfinite(distances), jnp.maximum(distances, neighbor_k), 0.0
        )
        lrd = 1.0 / jnp.maximum(
            jnp.sum(neighbor_weight * reachability, axis=-1)
            / jnp.maximum(
                jnp.sum(neighbor_weight, axis=-1), jnp.finfo(reachability.dtype).tiny
            ),
            jnp.finfo(reachability.dtype).tiny,
        )
        effective = jnp.sum(support_mask, axis=-1)
        valid = effective > self.neighbor_count
        status = jnp.where(valid, ML_SUCCESS, ML_INSUFFICIENT_DATA)
        model = LocalOutlierFactorModel(
            support=support,
            support_mask=support_mask,
            support_weight=support_weight,
            local_reachability_density=lrd,
            k_distance=k_distance,
            metric=self.metric,
            neighbor_count=self.neighbor_count,
            feature_count=batch.feature_count,
            case_shape=batch.case_shape,
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=effective,
            method="chunked-local-outlier-factor",
        )
        contract = GradientContract(
            prediction_inputs="almost-everywhere",
            prediction_parameters="almost-everywhere",
            fit_mode="stopped",
            nondifferentiable_outputs=("neighbor_indices", "predict"),
            conditions=("Neighbor ordering is fixed and tie-free.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="local-outlier-factor",
            gradient_contract=contract,
        )


__all__ = [
    "KernelDensityModel",
    "KernelDensityRecipe",
    "LocalOutlierFactorModel",
    "LocalOutlierFactorRecipe",
]
