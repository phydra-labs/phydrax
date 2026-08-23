#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import core as jax_core
from jaxtyping import Array

from phydrax.domain import PointBatch

from .._measure_weights import log_weights_from_normalized, normalized_weights
from .._strict import StrictModule
from ._batches import PointIntegrationBatch, WeightedSampleBatch
from ._targets import DiscreteMeasureTarget, ProbabilityTarget, WeightedSampleTarget
from ._transformations import MeasureTransformationRecord


class FiniteMeasureRealization(StrictModule):
    """Canonical one-axis finite positive measure for support transformations."""

    samples: Any
    log_weights: Array
    mask: Array
    physical_mass: Array
    ancestry: Any
    support_valid: Any
    axis: str | int = eqx.field(static=True)
    count: int = eqx.field(static=True)
    normalized: bool = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)


def lower_finite_measure(realization: Any, /) -> FiniteMeasureRealization:
    """Lower a supported realization to normalized probabilities and physical mass."""

    if isinstance(realization.batch, tuple):
        raise TypeError(
            "Replicated or component-union realizations must be transformed per member."
        )
    if isinstance(realization.batch, PointIntegrationBatch):
        return _lower_point_measure(realization, realization.batch)
    if isinstance(realization.batch, WeightedSampleBatch):
        return _lower_weighted_measure(realization, realization.batch)
    raise TypeError(
        "Finite-measure transformations require a one-axis PointIntegrationBatch "
        "or WeightedSampleBatch."
    )


def feature_matrix(value: Any, axis: str | int, count: int, /) -> Array:
    """Canonicalize array or named-field feature PyTrees to shape ``(count, m)``."""

    if isinstance(value, (cx.Field, jax.Array, jax_core.Tracer)):
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
        raise ValueError("Could not derive finite-dimensional measure features.")
    result = jnp.concatenate(matrices, axis=1)
    if not bool(jnp.all(jnp.isfinite(result))):
        raise ValueError("Measure features must be finite on every source point.")
    return result


def take_samples(value: Any, axis: str | int, indices: Array, /) -> Any:
    """Take finite-measure sample leaves while preserving named structure."""

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
        if isinstance(leaf, (jax.Array, jax_core.Tracer)):
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


def transformed_weighted_realization(
    realization: Any,
    measure: FiniteMeasureRealization,
    log_weights: Array,
    /,
    *,
    transformation_kind: str,
    transformation_diagnostics: Any,
    provenance: str,
    indices: Array | None = None,
):
    """Rebuild a weighted realization and append ordered transformation evidence."""

    from ._api import IntegrationRealization

    if not isinstance(realization, IntegrationRealization):
        raise TypeError("realization must be an IntegrationRealization.")
    log_weights_ = jnp.asarray(log_weights, dtype=measure.log_weights.dtype)
    if log_weights_.shape != ((measure.count,) if indices is None else indices.shape):
        raise ValueError("Transformed log weights disagree with the output support size.")
    if indices is None:
        samples = measure.samples
        mask = measure.mask
        ancestry = measure.ancestry
        support_valid = measure.support_valid
    else:
        samples = take_samples(measure.samples, measure.axis, indices)
        mask = jnp.take(measure.mask, indices)
        ancestry = (
            indices
            if measure.ancestry is None
            else take_samples(measure.ancestry, measure.axis, indices)
        )
        support_valid = measure.support_valid
    if ancestry is None:
        ancestry = jnp.arange(log_weights_.shape[0], dtype=jnp.int32)
    if isinstance(measure.axis, str):
        target_log_weights: Array | cx.Field = cx.Field(
            log_weights_,
            dims=(measure.axis,),
        )
        target_mask: Array | cx.Field = cx.Field(mask, dims=(measure.axis,))
        if not isinstance(ancestry, cx.Field):
            ancestry = cx.Field(jnp.asarray(ancestry), dims=(measure.axis,))
        sample_axes: int | str = measure.axis
    else:
        target_log_weights = log_weights_
        target_mask = mask
        ancestry = jnp.asarray(ancestry, dtype=jnp.int32)
        sample_axes = 0
    target_mass = None if measure.normalized else measure.physical_mass
    target = WeightedSampleTarget(
        samples,
        target_log_weights,
        normalized=measure.normalized,
        target_mass=target_mass,
        independent=False,
        ancestry=ancestry,
        support_valid=support_valid,
        mask=target_mask,
        sample_axes=sample_axes,
        provenance=provenance,
    )
    batch = WeightedSampleBatch(
        target.samples,
        target.log_weights,
        mask=target.mask,
        target_mass=target.target_mass,
        ancestry_ids=target.ancestry,
        support_valid=target.support_valid,
        sample_axes=target.sample_axes,
        independent=False,
        provenance=target.provenance,
    )
    record = MeasureTransformationRecord(
        transformation_kind,
        transformation_diagnostics,
        source_provenance=measure.source_provenance,
        target_provenance=provenance,
    )
    return IntegrationRealization(
        target,
        None,
        batch,
        realization.key,
        realization.transformations + (record,),
        precision=realization.precision,
    )


def _single_named_axis(batch: PointIntegrationBatch, /) -> tuple[str, int]:
    if len(batch.axes) != 1:
        raise ValueError("Finite-measure transformations require exactly one axis.")
    axis = batch.axes[0]
    if batch.weights.dims != (axis,):
        raise ValueError(
            "Finite-measure transformations require one-dimensional point weights."
        )
    if batch.stratum_indices is not None:
        raise ValueError("Stratified realizations must be transformed within strata.")
    if "antithetic" in batch.provenance:
        raise ValueError("Antithetic realizations must preserve their paired blocks.")
    return axis, int(batch.weights.shape[0])


def _single_weighted_axis(batch: WeightedSampleBatch, /) -> tuple[str | int, int]:
    if len(batch.sample_axes) != 1:
        raise ValueError("Finite-measure transformations require one sample axis.")
    if batch.stratum_ids is not None:
        raise ValueError("Weighted strata must be transformed independently.")
    if batch.pair_ids is not None:
        raise ValueError("Paired samples must be transformed in paired blocks.")
    if batch.replicate_ids is not None:
        raise ValueError("Replicated samples must be transformed independently.")
    axis = batch.sample_axes[0]
    if isinstance(batch.log_weights, cx.Field):
        if not isinstance(axis, str) or batch.log_weights.dims != (axis,):
            raise ValueError("Named transformations require one-dimensional log weights.")
        return axis, int(batch.log_weights.shape[0])
    if not isinstance(axis, int) or batch.log_weights.ndim != 1 or axis != 0:
        raise ValueError("Raw transformations require a leading one-dimensional axis.")
    return axis, int(batch.log_weights.shape[0])


def _lower_point_measure(
    realization: Any,
    batch: PointIntegrationBatch,
    /,
) -> FiniteMeasureRealization:
    target = realization.target
    if not isinstance(target, (DiscreteMeasureTarget, ProbabilityTarget)):
        raise TypeError(
            "Point transformations currently support probability and external "
            "discrete targets only."
        )
    axis, count = _single_named_axis(batch)
    values = jnp.asarray(batch.weights.data, dtype=float)
    mask = (
        jnp.ones((count,), dtype=bool)
        if batch.mask is None
        else jnp.asarray(batch.mask.data, dtype=bool)
    )
    if bool(jnp.any(mask & (~jnp.isfinite(values) | (values < 0.0)))):
        raise ValueError("Finite measures require finite nonnegative weights.")
    positive = mask & (values > 0.0)
    raw_log_weights = jnp.where(positive, jnp.log(values), -jnp.inf)
    weights, active, valid, log_mass = normalized_weights(
        count,
        log_weights=raw_log_weights,
        mask=mask,
    )
    if not bool(valid):
        raise ValueError("Finite measures require positive source mass.")
    normalized_log_weights = log_weights_from_normalized(weights, active)
    normalized = target.normalized
    physical_mass = (
        jnp.asarray(1.0, dtype=weights.dtype)
        if normalized
        else jnp.asarray(batch.target_mass, dtype=weights.dtype)
        if batch.target_mass is not None
        else jnp.exp(log_mass)
    )
    return FiniteMeasureRealization(
        samples=batch.points,
        log_weights=normalized_log_weights,
        mask=mask,
        physical_mass=physical_mass,
        ancestry=None,
        support_valid=None,
        axis=axis,
        count=count,
        normalized=normalized,
        source_provenance=batch.provenance,
    )


def _lower_weighted_measure(
    realization: Any,
    batch: WeightedSampleBatch,
    /,
) -> FiniteMeasureRealization:
    target = realization.target
    if not isinstance(target, WeightedSampleTarget):
        raise TypeError(
            "Weighted transformations require an externally materialized "
            "WeightedSampleTarget."
        )
    axis, count = _single_weighted_axis(batch)
    if isinstance(batch.log_weights, cx.Field):
        values = jnp.asarray(batch.log_weights.data, dtype=float)
        mask = (
            jnp.ones((count,), dtype=bool)
            if batch.mask is None
            else jnp.asarray(cast(cx.Field, batch.mask).data, dtype=bool)
        )
    else:
        values = jnp.asarray(batch.log_weights, dtype=float)
        mask = (
            jnp.ones((count,), dtype=bool)
            if batch.mask is None
            else jnp.asarray(batch.mask, dtype=bool)
        )
    weights, active, valid, log_mass = normalized_weights(
        count,
        log_weights=values,
        mask=mask,
    )
    if not bool(valid):
        raise ValueError(
            "Finite measures require finite or negative-infinite log weights and "
            "positive source mass."
        )
    normalized_log_weights = log_weights_from_normalized(weights, active)
    physical_mass = (
        jnp.asarray(1.0, dtype=weights.dtype)
        if target.normalized
        else jnp.asarray(target.target_mass, dtype=weights.dtype)
        if target.target_mass is not None
        else jnp.exp(log_mass)
    )
    return FiniteMeasureRealization(
        samples=batch.samples,
        log_weights=normalized_log_weights,
        mask=mask,
        physical_mass=physical_mass,
        ancestry=batch.ancestry_ids,
        support_valid=batch.support_valid,
        axis=axis,
        count=count,
        normalized=target.normalized,
        source_provenance=batch.provenance,
    )


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
    elif isinstance(value, (jax.Array, jax_core.Tracer)):
        data = jnp.asarray(value, dtype=float)
        position = axis if isinstance(axis, int) else 0
        if data.ndim == 0 or position >= data.ndim or data.shape[position] != count:
            return None
        data = jnp.moveaxis(data, position, 0)
    else:
        return None
    if data.shape[0] != count:
        raise ValueError("Sample fields disagree on the transformation-axis size.")
    return data.reshape((count, -1))


__all__ = [
    "FiniteMeasureRealization",
    "feature_matrix",
    "lower_finite_measure",
    "take_samples",
    "transformed_weighted_realization",
]
