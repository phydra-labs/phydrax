#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from phydrax.coresets import (
    CoresetSelection,
    kernel_herd,
    KernelHerding,
    moment_recombine,
    MomentRecombination,
)
from phydrax.domain import PointBatch

from .._strict import StrictModule
from ._batches import PointIntegrationBatch, WeightedSampleBatch
from ._targets import (
    DiscreteMeasureTarget,
    ProbabilityTarget,
    WeightedSampleTarget,
)


CompressionMethod = MomentRecombination | KernelHerding
FeatureMap = Callable[[Any], Any]


class MeasureCompressionDiagnostics(StrictModule):
    """Selection evidence and source identity for one compressed realization."""

    selection: Any
    source_mass: Array
    source_points: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)


class CompressedIntegrationDiagnostics(StrictModule):
    """Compression evidence paired with the downstream measure reduction."""

    compression: MeasureCompressionDiagnostics
    reduction: Any


def _single_named_axis(batch: PointIntegrationBatch, /) -> tuple[str, int]:
    if len(batch.axes) != 1:
        raise ValueError("Compression currently requires exactly one integration axis.")
    axis = batch.axes[0]
    if batch.weights.dims != (axis,):
        raise ValueError(
            "Compression currently requires a one-dimensional point-weight field."
        )
    if batch.stratum_indices is not None:
        raise ValueError(
            "Stratified realizations must be compressed within strata, which is not "
            "yet supported."
        )
    if "antithetic" in batch.provenance:
        raise ValueError(
            "Antithetic realizations cannot be compressed without preserving pairs."
        )
    return axis, int(batch.weights.shape[0])


def _single_weighted_axis(
    batch: WeightedSampleBatch,
    /,
) -> tuple[str | int, int]:
    if len(batch.sample_axes) != 1:
        raise ValueError("Compression currently requires exactly one sample axis.")
    if batch.stratum_ids is not None:
        raise ValueError(
            "Weighted strata must be compressed independently, which is not yet supported."
        )
    if batch.pair_ids is not None:
        raise ValueError(
            "Paired samples cannot be compressed without preserving their pair blocks."
        )
    if batch.replicate_ids is not None:
        raise ValueError(
            "Replicated samples must be compressed independently by replicate."
        )
    axis = batch.sample_axes[0]
    if isinstance(batch.log_weights, cx.Field):
        if not isinstance(axis, str) or batch.log_weights.dims != (axis,):
            raise ValueError(
                "Named compression currently requires a one-dimensional log-weight field."
            )
        return axis, int(batch.log_weights.shape[0])
    if not isinstance(axis, int) or batch.log_weights.ndim != 1 or axis != 0:
        raise ValueError(
            "Raw compression currently requires a leading one-dimensional sample axis."
        )
    return axis, int(batch.log_weights.shape[0])


def _matrix_leaf(value: Any, axis: str | int, count: int, /) -> Array | None:
    if isinstance(value, cx.Field):
        if isinstance(axis, str):
            if axis not in value.named_dims:
                return None
            position = value.dims.index(axis)
        else:
            if axis >= value.data.ndim:
                return None
            position = axis
        data = jnp.moveaxis(jnp.asarray(value.data, dtype=float), position, 0)
    elif isinstance(value, (jax.Array, jax.core.Tracer)):
        data = jnp.asarray(value, dtype=float)
        position = axis if isinstance(axis, int) else 0
        if data.ndim == 0 or position >= data.ndim or data.shape[position] != count:
            return None
        data = jnp.moveaxis(data, position, 0)
    else:
        return None
    if data.shape[0] != count:
        raise ValueError("Sample fields disagree on the compression-axis size.")
    return data.reshape((count, -1))


def _feature_matrix(value: Any, axis: str | int, count: int, /) -> Array:
    if isinstance(value, cx.Field):
        leaves = (value,)
    elif isinstance(value, (jax.Array, jax.core.Tracer)):
        leaves = (value,)
    else:
        leaves = tuple(
            jtu.tree_leaves(value, is_leaf=lambda leaf: isinstance(leaf, cx.Field))
        )
    matrices = tuple(
        matrix
        for leaf in leaves
        if (matrix := _matrix_leaf(leaf, axis, count)) is not None
    )
    if not matrices:
        raise ValueError("Could not derive finite-dimensional compression features.")
    return jnp.concatenate(matrices, axis=1)


def _take_samples(value: Any, axis: str | int, indices: Array, /) -> Any:
    def take(leaf: Any) -> Any:
        if isinstance(leaf, cx.Field):
            if isinstance(axis, str):
                if axis not in leaf.named_dims:
                    return leaf
                position = leaf.dims.index(axis)
            else:
                if axis >= leaf.data.ndim:
                    return leaf
                position = axis
            return cx.Field(jnp.take(leaf.data, indices, axis=position), dims=leaf.dims)
        if isinstance(leaf, (jax.Array, jax.core.Tracer)):
            position = axis if isinstance(axis, int) else 0
            if leaf.ndim == 0 or position >= leaf.ndim:
                return leaf
            return jnp.take(leaf, indices, axis=position)
        return leaf

    if isinstance(value, PointBatch):
        points = jtu.tree_map(
            take,
            value.points,
            is_leaf=lambda leaf: isinstance(leaf, cx.Field),
        )
        metadata = jtu.tree_map(
            take,
            value.metadata,
            is_leaf=lambda leaf: isinstance(leaf, cx.Field),
        )
        return PointBatch(points, value.structure, metadata=metadata)
    return jtu.tree_map(take, value, is_leaf=lambda leaf: isinstance(leaf, cx.Field))


def _select(
    feature_values: Array,
    method: CompressionMethod,
    /,
    *,
    log_weights: Array,
    mask: Array | None,
) -> CoresetSelection:
    if isinstance(method, MomentRecombination):
        return moment_recombine(
            feature_values,
            method,
            log_weights=log_weights,
            mask=mask,
        )
    if isinstance(method, KernelHerding):
        return kernel_herd(
            feature_values,
            method,
            log_weights=log_weights,
            mask=mask,
        )
    raise TypeError("method must be MomentRecombination or KernelHerding.")


def _point_source(
    realization: Any,
    batch: PointIntegrationBatch,
    /,
) -> tuple[Any, str, int, Array, Array | None, bool, Array]:
    target = realization.target
    if not isinstance(target, (DiscreteMeasureTarget, ProbabilityTarget)):
        raise TypeError(
            "Point-realization compression currently supports probability and "
            "external discrete targets only."
        )
    axis, count = _single_named_axis(batch)
    weights = jnp.asarray(batch.weights.data, dtype=float)
    mask = None if batch.mask is None else jnp.asarray(batch.mask.data, dtype=bool)
    included = jnp.ones((count,), dtype=bool) if mask is None else mask
    if bool(jnp.any(included & (~jnp.isfinite(weights) | (weights < 0.0)))):
        raise ValueError("Compression requires finite nonnegative source weights.")
    active_weights = jnp.where(included & (weights > 0.0), weights, 0.0)
    source_mass = jnp.sum(active_weights)
    if not bool(jnp.isfinite(source_mass) & (source_mass > 0.0)):
        raise ValueError("Compression requires positive source mass.")
    log_weights = jnp.where(active_weights > 0.0, jnp.log(active_weights), -jnp.inf)
    normalized = target.normalized
    return batch.points, axis, count, log_weights, mask, normalized, source_mass


def _weighted_source(
    realization: Any,
    batch: WeightedSampleBatch,
    /,
) -> tuple[Any, str | int, int, Array, Array | None, bool, Array]:
    target = realization.target
    if not isinstance(target, WeightedSampleTarget):
        raise TypeError(
            "Weighted compression currently supports externally materialized "
            "WeightedSampleTarget realizations only."
        )
    axis, count = _single_weighted_axis(batch)
    if isinstance(batch.log_weights, cx.Field):
        log_weights = jnp.asarray(batch.log_weights.data, dtype=float)
        mask = None if batch.mask is None else jnp.asarray(cast(cx.Field, batch.mask).data)
    else:
        log_weights = jnp.asarray(batch.log_weights, dtype=float)
        mask = None if batch.mask is None else jnp.asarray(batch.mask, dtype=bool)
    included = jnp.ones((count,), dtype=bool) if mask is None else mask
    admissible = jnp.isfinite(log_weights) | jnp.isneginf(log_weights)
    if bool(jnp.any(included & ~admissible)):
        raise ValueError("Compression requires finite or negative-infinite log weights.")
    finite = included & jnp.isfinite(log_weights)
    if not bool(jnp.any(finite)):
        raise ValueError("Compression requires positive source mass.")
    log_mass = jax.scipy.special.logsumexp(jnp.where(finite, log_weights, -jnp.inf))
    if target.normalized:
        source_mass = jnp.asarray(1.0)
    elif target.target_mass is not None:
        source_mass = jnp.asarray(target.target_mass, dtype=float)
    else:
        source_mass = jnp.exp(log_mass)
    return (
        batch.samples,
        axis,
        count,
        log_weights,
        mask,
        target.normalized,
        source_mass,
    )


def compress(
    realization: Any,
    method: CompressionMethod,
    /,
    *,
    features: FeatureMap | Any | None = None,
):
    """Compress a finite positive realization before evaluating its integrand."""
    from ._api import IntegrationRealization

    if not isinstance(realization, IntegrationRealization):
        raise TypeError("compress expects an IntegrationRealization from materialize().")
    if isinstance(realization.batch, tuple):
        raise ValueError(
            "Replicated or component-union realizations must be compressed per member."
        )
    if isinstance(realization.batch, PointIntegrationBatch):
        source = _point_source(realization, realization.batch)
        source_provenance = realization.batch.provenance
    elif isinstance(realization.batch, WeightedSampleBatch):
        source = _weighted_source(realization, realization.batch)
        source_provenance = realization.batch.provenance
    else:
        raise TypeError(
            "Compression requires a one-axis PointIntegrationBatch or "
            "WeightedSampleBatch."
        )
    samples, axis, count, source_log_weights, source_mask, normalized, source_mass = source
    support_valid = (
        realization.batch.support_valid
        if isinstance(realization.batch, WeightedSampleBatch)
        else None
    )
    raw_features = (
        samples
        if features is None
        else features(samples)
        if callable(features)
        else features
    )
    feature_values = _feature_matrix(raw_features, axis, count)
    selection = _select(
        feature_values,
        method,
        log_weights=source_log_weights,
        mask=source_mask,
    )
    if not bool(selection.diagnostics.valid):
        raise ValueError("Coreset selection failed for the supplied measure.")
    selected_samples = _take_samples(samples, axis, selection.indices)
    selected_source_ancestry = (
        None
        if not isinstance(realization.batch, WeightedSampleBatch)
        or realization.batch.ancestry_ids is None
        else _take_samples(
            realization.batch.ancestry_ids,
            axis,
            selection.indices,
        )
    )
    if isinstance(axis, str):
        selected_log_weights: Array | cx.Field = cx.Field(
            selection.log_weights,
            dims=(axis,),
        )
        selected_mask: Array | cx.Field = cx.Field(selection.mask, dims=(axis,))
        ancestry: Array | cx.Field = (
            cx.Field(selection.indices, dims=(axis,))
            if selected_source_ancestry is None
            else cast(cx.Field, selected_source_ancestry)
        )
        sample_axes: int | str = axis
    else:
        selected_log_weights = selection.log_weights
        selected_mask = selection.mask
        ancestry = (
            selection.indices
            if selected_source_ancestry is None
            else jnp.asarray(selected_source_ancestry, dtype=jnp.int32)
        )
        sample_axes = 0
    target_mass = None if normalized else source_mass
    compressed_target = WeightedSampleTarget(
        selected_samples,
        selected_log_weights,
        normalized=normalized,
        target_mass=target_mass,
        independent=False,
        ancestry=ancestry,
        support_valid=support_valid,
        mask=selected_mask,
        sample_axes=sample_axes,
        provenance=f"compressed:{selection.method}:{source_provenance}",
    )
    compressed_batch = WeightedSampleBatch(
        compressed_target.samples,
        compressed_target.log_weights,
        mask=compressed_target.mask,
        target_mass=compressed_target.target_mass,
        ancestry_ids=compressed_target.ancestry,
        support_valid=compressed_target.support_valid,
        sample_axes=compressed_target.sample_axes,
        independent=False,
        provenance=compressed_target.provenance,
    )
    diagnostics = MeasureCompressionDiagnostics(
        selection=selection.diagnostics,
        source_mass=source_mass,
        source_points=count,
        feature_count=int(feature_values.shape[1]),
        source_provenance=source_provenance,
    )
    return IntegrationRealization(
        compressed_target,
        None,
        compressed_batch,
        realization.key,
        diagnostics,
    )


__all__ = [
    "CompressedIntegrationDiagnostics",
    "CompressionMethod",
    "MeasureCompressionDiagnostics",
    "compress",
]
