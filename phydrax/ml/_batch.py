#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._schema import FeatureSchema, TargetSchema
from ._sparse_features import FeatureArray, SparseFeatures


WeightPolicy: TypeAlias = Literal["none", "statistical", "measure", "product"]


def _broadcast(value: ArrayLike | None, shape: tuple[int, ...], *, dtype, fill) -> Array:
    if value is None:
        return jnp.full(shape, fill, dtype=dtype)
    return jnp.broadcast_to(jnp.asarray(value, dtype=dtype), shape)


class MLBatch(StrictModule):
    """Canonical supervised or unsupervised feature batch for native ML fits."""

    features: FeatureArray
    targets: Array | None
    feature_mask: Array
    target_mask: Array | None
    sample_mask: Array
    sample_weight: Array
    measure_weight: Array
    groups: Array | None
    feature_schema: FeatureSchema
    target_schema: TargetSchema
    case_shape: tuple[int, ...] = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    target_shape: tuple[int, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        features: ArrayLike | SparseFeatures,
        targets: ArrayLike | None = None,
        /,
        *,
        feature_mask: ArrayLike | None = None,
        target_mask: ArrayLike | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        measure_weight: ArrayLike | None = None,
        groups: ArrayLike | None = None,
        feature_schema: FeatureSchema | None = None,
        target_schema: TargetSchema | None = None,
    ):
        if isinstance(features, SparseFeatures):
            features_ = features
            case_shape = features.case_shape
            sample_count = features.sample_count
            feature_count = features.feature_count
            feature_shape = features.shape
            if feature_mask is not None:
                raise ValueError(
                    "SparseFeatures encode entry validity internally; feature_mask is unsupported."
                )
            feature_mask_ = jnp.ones(feature_shape, dtype=bool)
        else:
            features_ = jnp.asarray(features)
            if features_.ndim < 2:
                raise ValueError(
                    "features must have shape case_shape + (sample, feature)."
                )
            case_shape = tuple(int(size) for size in features_.shape[:-2])
            sample_count = int(features_.shape[-2])
            feature_count = int(features_.shape[-1])
            if sample_count <= 0 or feature_count <= 0:
                raise ValueError(
                    "Feature sample and feature dimensions must be positive."
                )
            feature_shape = tuple(int(size) for size in features_.shape)
            feature_mask_ = _broadcast(feature_mask, feature_shape, dtype=bool, fill=True)

        sample_shape = case_shape + (sample_count,)
        targets_ = None if targets is None else jnp.asarray(targets)
        if targets_ is None:
            if target_mask is not None:
                raise ValueError("target_mask requires targets.")
            target_shape = None
            target_mask_ = None
        else:
            prefix = targets_.shape[: len(sample_shape)]
            if tuple(int(size) for size in prefix) != sample_shape:
                raise ValueError(
                    f"targets must begin with case/sample shape {sample_shape}; "
                    f"got {targets_.shape}."
                )
            target_shape = tuple(
                int(size) for size in targets_.shape[len(sample_shape) :]
            )
            target_mask_ = _broadcast(
                target_mask,
                tuple(int(size) for size in targets_.shape),
                dtype=bool,
                fill=True,
            )

        sample_mask_ = _broadcast(sample_mask, sample_shape, dtype=bool, fill=True)
        sample_weight_ = _broadcast(sample_weight, sample_shape, dtype=float, fill=1.0)
        measure_weight_ = _broadcast(measure_weight, sample_shape, dtype=float, fill=1.0)
        groups_ = None
        if groups is not None:
            groups_ = jnp.broadcast_to(jnp.asarray(groups), sample_shape)
            if not jnp.issubdtype(groups_.dtype, jnp.integer):
                raise TypeError("groups must use an integer dtype.")
            groups_ = groups_.astype(jnp.int32)

        feature_schema_ = (
            FeatureSchema.anonymous(feature_count)
            if feature_schema is None
            else feature_schema
        )
        if len(feature_schema_.names) != feature_count:
            raise ValueError("Feature schema length must match the feature axis.")
        target_schema_ = TargetSchema() if target_schema is None else target_schema

        self.features = features_
        self.targets = targets_
        self.feature_mask = feature_mask_
        self.target_mask = target_mask_
        self.sample_mask = sample_mask_
        self.sample_weight = sample_weight_
        self.measure_weight = measure_weight_
        self.groups = groups_
        self.feature_schema = feature_schema_
        self.target_schema = target_schema_
        self.case_shape = case_shape
        self.sample_count = sample_count
        self.feature_count = feature_count
        self.target_shape = target_shape

    @property
    def supervised(self) -> bool:
        return self.targets is not None

    def require_targets(self, /) -> Array:
        if self.targets is None:
            raise ValueError("This ML recipe requires targets.")
        return self.targets

    def effective_weight(self, policy: WeightPolicy = "statistical", /) -> Array:
        if policy == "none":
            weights = jnp.ones_like(self.sample_weight)
        elif policy == "statistical":
            weights = self.sample_weight
        elif policy == "measure":
            weights = self.measure_weight
        elif policy == "product":
            weights = self.sample_weight * self.measure_weight
        else:
            raise ValueError(f"Unsupported weight policy {policy!r}.")
        return jnp.where(self.sample_mask, weights, 0.0)

    def weights_valid(self, policy: WeightPolicy = "statistical", /) -> Array:
        weights = self.effective_weight(policy)
        return jnp.all(jnp.isfinite(weights) & (weights >= 0.0), axis=-1) & (
            jnp.sum(weights, axis=-1) > 0.0
        )

    def dense_features(self, /, *, fill_value: float = 0.0) -> Array:
        """Return masked dense features without materializing sparse storage."""
        if isinstance(self.features, SparseFeatures):
            raise TypeError(
                "This recipe requires dense features; call SparseFeatures.to_dense() "
                "explicitly before fitting if materialization is intended."
            )
        return jnp.where(self.feature_mask, self.features, fill_value)

    def take_samples(self, indices: ArrayLike, /) -> "MLBatch":
        selected = jnp.asarray(indices, dtype=jnp.int32)
        if selected.ndim != 1:
            raise ValueError("Sample indices must be one-dimensional.")
        feature_axis = len(self.case_shape)
        target_axis = len(self.case_shape)
        features = (
            self.features.take_rows(selected)
            if isinstance(self.features, SparseFeatures)
            else jnp.take(self.features, selected, axis=feature_axis)
        )
        return MLBatch(
            features,
            None
            if self.targets is None
            else jnp.take(self.targets, selected, axis=target_axis),
            feature_mask=(
                None
                if isinstance(features, SparseFeatures)
                else jnp.take(self.feature_mask, selected, axis=feature_axis)
            ),
            target_mask=(
                None
                if self.target_mask is None
                else jnp.take(self.target_mask, selected, axis=target_axis)
            ),
            sample_mask=jnp.take(self.sample_mask, selected, axis=feature_axis),
            sample_weight=jnp.take(self.sample_weight, selected, axis=feature_axis),
            measure_weight=jnp.take(self.measure_weight, selected, axis=feature_axis),
            groups=(
                None
                if self.groups is None
                else jnp.take(self.groups, selected, axis=feature_axis)
            ),
            feature_schema=self.feature_schema,
            target_schema=self.target_schema,
        )

    def with_features(
        self,
        features: ArrayLike | SparseFeatures,
        /,
        *,
        feature_schema: FeatureSchema | None = None,
        feature_mask: ArrayLike | None = None,
    ) -> "MLBatch":
        return MLBatch(
            features,
            self.targets,
            feature_mask=feature_mask,
            target_mask=self.target_mask,
            sample_mask=self.sample_mask,
            sample_weight=self.sample_weight,
            measure_weight=self.measure_weight,
            groups=self.groups,
            feature_schema=feature_schema,
            target_schema=self.target_schema,
        )


__all__ = ["MLBatch", "WeightPolicy"]
