#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._model._array import value_derivative_contract
from ..._model._binding import ModelBinding
from ..._trainable import fixed_field
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    ML_SUCCESS,
    prediction_fit_contract,
)
from .._numerics._pairwise import blocked_top_k
from .._schema import AbstractFittedModel
from ._utils import (
    blocked_case_logsumexp,
    blocked_case_top_k,
    chunked_call,
    gather_support,
    metric_regularity,
    pad_support,
    size,
    validate_metric,
    validated_weights,
)


# A Gaussian mixture density: exp of a log-sum-exp of quadratics.
_KERNEL_DENSITY_CONTRACT = value_derivative_contract(DerivativeRegularity.smooth())


class KernelDensityModel(AbstractFittedModel):
    support: Array = fixed_field()
    support_weight: Array = fixed_field()
    support_mask: Array
    bandwidth: Array = fixed_field()
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
    ) -> None:
        support_ = jnp.asarray(support)
        weight_ = jnp.asarray(support_weight)
        mask_ = jnp.asarray(support_mask, dtype=jnp.bool_)
        features = int(feature_count)
        cases = tuple(case_shape)
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
        bandwidth_ = jnp.asarray(bandwidth, dtype=jnp.float64)
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

    def _prediction_contract(self) -> DerivativeContract:
        return _KERNEL_DENSITY_CONTRACT

    def log_density(
        self, x: ArrayLike, /, *, workspace_bytes: int = 4 * 1024 * 1024
    ) -> Array:
        active = self.support_mask & (self.support_weight > 0)
        log_weight = jnp.where(
            active,
            jnp.log(jnp.maximum(self.support_weight, jnp.finfo(jnp.float64).tiny)),
            -jnp.inf,
        )
        log_sum, query_shape = blocked_case_logsumexp(
            jnp.asarray(x),
            self.support,
            log_weight,
            0.5 / (self.bandwidth * self.bandwidth),
            self.case_shape,
            workspace_bytes=workspace_bytes,
        )
        total = jnp.sum(jnp.where(self.support_mask, self.support_weight, 0.0), axis=-1)
        total = total.reshape(self.case_shape + (1,) * len(query_shape))
        normalizer = (
            jnp.log(jnp.maximum(total, jnp.finfo(jnp.float64).tiny))
            + self.feature_count * jnp.log(self.bandwidth)
            + 0.5 * self.feature_count * jnp.log(2.0 * jnp.pi)
        )
        return jnp.where(total > 0, log_sum - normalizer, -jnp.inf)

    def __call__(
        self, x: ArrayLike, /, *, key: Any = None, workspace_bytes: int = 4 * 1024 * 1024
    ) -> Array:
        del key
        return jnp.exp(self.log_density(x, workspace_bytes=workspace_bytes))

    def score_samples(
        self, x: ArrayLike, /, *, workspace_bytes: int = 4 * 1024 * 1024
    ) -> Array:
        return self.log_density(x, workspace_bytes=workspace_bytes)

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
    ) -> None:
        bandwidth_ = jnp.asarray(bandwidth, dtype=jnp.float64)
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
        contract = prediction_fit_contract(
            model._prediction_contract(),
            (
                SurfaceDerivative(DerivativeSurface.FIT_FEATURES, GradientLevel.SMOOTH),
                SurfaceDerivative(DerivativeSurface.FIT_WEIGHTS, GradientLevel.SMOOTH),
                SurfaceDerivative(
                    DerivativeSurface.FIT_HYPERPARAMETERS, GradientLevel.SMOOTH
                ),
            ),
            route=DerivativeRoute.DIRECT,
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="kernel-density",
            derivative_contract=contract,
        )


class LocalOutlierFactorModel(AbstractFittedModel):
    support: Array = fixed_field()
    support_mask: Array
    support_weight: Array = fixed_field()
    local_reachability_density: Array = fixed_field()
    k_distance: Array = fixed_field()
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
    ) -> None:
        support_ = jnp.asarray(support)
        mask_ = jnp.asarray(support_mask, dtype=jnp.bool_)
        weight_ = jnp.asarray(support_weight)
        lrd_ = jnp.asarray(local_reachability_density)
        k_distance_ = jnp.asarray(k_distance)
        features = int(feature_count)
        cases = tuple(case_shape)
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

    def _prediction_contract(self) -> DerivativeContract:
        # The neighbor-density average jumps when the top-k set changes; between
        # changes the score is the metric's reachability distances, rescaled.
        regularity = (
            DerivativeRegularity.discontinuous()
            if metric_regularity(self.metric) is None
            else DerivativeRegularity.piecewise_smooth(continuity=-1)
        )
        return prediction_fit_contract(
            value_derivative_contract(regularity),
            route=DerivativeRoute.DIRECT,
            nondifferentiable_outputs=("neighbor_indices", "predict"),
        )

    def score_samples(
        self, x: ArrayLike, /, *, workspace_bytes: int = 4 * 1024 * 1024
    ) -> Array:
        indices, selected_distance = blocked_case_top_k(
            jnp.asarray(x),
            self.support,
            self.support_mask,
            self.case_shape,
            self.metric,
            self.neighbor_count,
            workspace_bytes=workspace_bytes,
        )
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

    def __call__(
        self, x: ArrayLike, /, *, key: Any = None, workspace_bytes: int = 4 * 1024 * 1024
    ) -> Array:
        del key
        return self.score_samples(x, workspace_bytes=workspace_bytes)

    def predict(
        self,
        x: ArrayLike,
        /,
        *,
        threshold: ArrayLike = 1.5,
        workspace_bytes: int = 4 * 1024 * 1024,
    ) -> Array:
        return jnp.where(
            self.score_samples(x, workspace_bytes=workspace_bytes)
            <= jnp.asarray(threshold),
            1,
            -1,
        ).astype(jnp.int32)

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
    ) -> None:
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
        indices, distances = blocked_top_k(
            support,
            support,
            support_mask,
            self.neighbor_count,
            metric=self.metric,
            block_size=self.chunk_size,
            workspace_bytes=(4 * 1024 * 1024) // size(batch.case_shape),
            exclude_self=True,
        )
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
        contract = prediction_fit_contract(
            model._prediction_contract(),
            route=DerivativeRoute.STOPPED,
            conditions=("Neighbor ordering is fixed and tie-free.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="local-outlier-factor",
            derivative_contract=contract,
        )


__all__ = [
    "KernelDensityModel",
    "KernelDensityRecipe",
    "LocalOutlierFactorModel",
    "LocalOutlierFactorRecipe",
]
