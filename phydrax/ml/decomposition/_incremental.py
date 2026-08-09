#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel, ModelBinding
from .._batch import MLBatch, WeightPolicy
from .._contracts import AbstractRecipe, FitResult, GradientContract
from ._subspace import _fit_subspace, SubspaceModel


class IncrementalPCAModel(AbstractArrayModel):
    """Principal subspace summary that can be immutably merged with later batches."""

    subspace: SubspaceModel
    total_weight: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    chunks_seen: int = eqx.field(static=True)

    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(
        self,
        model: SubspaceModel,
        /,
        *,
        total_weight,
        chunks_seen: int,
    ):
        self.subspace = model
        self.total_weight = jnp.asarray(total_weight)
        self.in_size = model.in_size
        self.out_size = model.out_size
        self.case_shape = model.case_shape
        self.chunks_seen = int(chunks_seen)

    @property
    def offset(self) -> Array:
        return self.subspace.offset

    @property
    def components(self) -> Array:
        return self.subspace.components

    @property
    def weighted_components(self) -> Array:
        return self.subspace.weighted_components

    @property
    def feature_metric(self) -> Array:
        return self.subspace.feature_metric

    @property
    def feature_support(self) -> Array:
        return self.subspace.feature_support

    @property
    def centered(self) -> bool:
        return self.subspace.centered

    @property
    def weighting_provenance(self) -> str:
        return self.subspace.weighting_provenance

    @property
    def centering_provenance(self) -> str:
        return self.subspace.centering_provenance

    @property
    def mask_provenance(self) -> str:
        return self.subspace.mask_provenance

    @property
    def query_layout_provenance(self) -> tuple[str, ...]:
        return self.subspace.query_layout_provenance

    @property
    def sign_phase_convention(self) -> str:
        return self.subspace.sign_phase_convention

    @property
    def singular_values(self) -> Array:
        return self.subspace.singular_values

    def transform(self, x, /) -> Array:
        return self.subspace.transform(x)

    def inverse_transform(self, scores, /) -> Array:
        return self.subspace.inverse_transform(scores)

    def project(self, x, /) -> Array:
        return self.subspace.project(x)

    def projector(self, /) -> Array:
        return self.subspace.projector()

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)

    def update_recipe(
        self,
        /,
        *,
        chunk_size: int | None = None,
        weight_policy: WeightPolicy = "statistical",
    ) -> "IncrementalPCA":
        return IncrementalPCA(
            self.out_size,
            chunk_size=chunk_size,
            weight_policy=weight_policy,
            previous=self,
        )


def _wrap_incremental(
    result: FitResult,
    /,
    *,
    total_weight,
    chunks_seen: int,
) -> FitResult:
    base = result.as_trainable()
    if not isinstance(base, SubspaceModel):
        raise TypeError("Incremental PCA expected a fitted SubspaceModel.")
    model = IncrementalPCAModel(
        base,
        total_weight=total_weight,
        chunks_seen=chunks_seen,
    )
    return FitResult(
        model,
        result.diagnostics,
        valid=result.valid,
        status=result.status,
        method="incremental-pca-merge-svd",
        gradient_contract=GradientContract(
            fit_features="conditional",
            fit_weights="conditional",
            fit_mode="spectral",
            conditions=(
                "each merge differentiates through its rank-truncated covariance summary",
                "projector gradients require retained/discarded spectral separation at every merge",
                "basis gradients additionally require non-repeated retained spectra",
            ),
        ),
    )


def _initial_fit(
    batch: MLBatch,
    /,
    *,
    rank: int,
    policy: WeightPolicy,
) -> FitResult:
    result = _fit_subspace(
        batch,
        rank=rank,
        centered=True,
        weight_policy=policy,
        physical_weights=None,
        differentiate="projector",
        method="incremental-pca-initial-svd",
        query_layout_provenance=(),
    )
    return _wrap_incremental(
        result,
        total_weight=jnp.sum(batch.effective_weight(policy), axis=-1),
        chunks_seen=1,
    )


def _merge_fit(
    previous: IncrementalPCAModel,
    batch: MLBatch,
    /,
    *,
    policy: WeightPolicy,
) -> FitResult:
    if previous.case_shape != batch.case_shape:
        raise ValueError(
            "Incremental PCA updates must preserve the fitted case shape; got "
            f"{previous.case_shape} and {batch.case_shape}."
        )
    if previous.in_size != batch.feature_count:
        raise ValueError("Incremental PCA updates must preserve feature width.")
    rank = previous.out_size
    repeated_mean = previous.offset[..., None, :]
    scaled_modes = (
        previous.weighted_components
        * (previous.singular_values * jnp.sqrt(float(rank)))[..., :, None]
    )
    plus = repeated_mean + scaled_modes
    minus = repeated_mean - scaled_modes
    pseudo_values = jnp.concatenate((plus, minus), axis=-2)
    pseudo_weight = jnp.broadcast_to(
        previous.total_weight[..., None] / float(2 * rank),
        batch.case_shape + (2 * rank,),
    )
    pseudo_mask = jnp.broadcast_to(
        previous.feature_support[..., None, :], pseudo_values.shape
    )
    current_values = batch.dense_features()
    current_weight = batch.effective_weight(policy)
    values = jnp.concatenate((pseudo_values, current_values), axis=-2)
    feature_mask = jnp.concatenate((pseudo_mask, batch.feature_mask), axis=-2)
    sample_weight = jnp.concatenate((pseudo_weight, current_weight), axis=-1)
    sample_mask = jnp.concatenate(
        (
            jnp.ones(batch.case_shape + (2 * rank,), dtype=bool),
            batch.sample_mask,
        ),
        axis=-1,
    )
    combined = MLBatch(
        values,
        feature_mask=feature_mask,
        sample_mask=sample_mask,
        sample_weight=sample_weight,
        feature_schema=batch.feature_schema,
    )
    result = _fit_subspace(
        combined,
        rank=rank,
        centered=True,
        weight_policy="statistical",
        physical_weights=None,
        differentiate="projector",
        method="incremental-pca-merge-svd",
        query_layout_provenance=(),
    )
    return _wrap_incremental(
        result,
        total_weight=previous.total_weight + jnp.sum(current_weight, axis=-1),
        chunks_seen=previous.chunks_seen + 1,
    )


class IncrementalPCA(AbstractRecipe):
    """Chunked immutable PCA using rank-truncated moment-preserving SVD merges."""

    n_components: int = eqx.field(static=True)
    chunk_size: int | None = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    previous: IncrementalPCAModel | None

    def __init__(
        self,
        n_components: int,
        /,
        *,
        chunk_size: int | None = None,
        weight_policy: WeightPolicy = "statistical",
        previous: IncrementalPCAModel | None = None,
    ):
        self.n_components = int(n_components)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.weight_policy = weight_policy
        self.previous = previous
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided.")
        if previous is not None and previous.out_size != self.n_components:
            raise ValueError("previous model rank must match n_components.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        chunk = batch.sample_count if self.chunk_size is None else self.chunk_size
        start = 0
        current = self.previous
        result: FitResult | None = None
        if current is None:
            first_size = min(batch.sample_count, max(chunk, self.n_components))
            first = batch.take_samples(jnp.arange(first_size, dtype=jnp.int32))
            result = _initial_fit(
                first,
                rank=self.n_components,
                policy=self.weight_policy,
            )
            fitted = result.as_trainable()
            assert isinstance(fitted, IncrementalPCAModel)
            current = fitted
            start = first_size
        while start < batch.sample_count:
            stop = min(batch.sample_count, start + chunk)
            selected = batch.take_samples(jnp.arange(start, stop, dtype=jnp.int32))
            result = _merge_fit(current, selected, policy=self.weight_policy)
            fitted = result.as_trainable()
            if not isinstance(fitted, IncrementalPCAModel):
                raise TypeError("Incremental PCA merge returned an invalid fitted model.")
            current = fitted
            start = stop
        if result is None:
            result = _merge_fit(current, batch, policy=self.weight_policy)
        return result


__all__ = [
    "IncrementalPCA",
    "IncrementalPCAModel",
]
