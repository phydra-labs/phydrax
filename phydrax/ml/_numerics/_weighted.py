#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike


def _expanded_weights(weights: Array, values: Array, /) -> Array:
    if weights.ndim > values.ndim:
        raise ValueError("weights rank cannot exceed values rank.")
    return weights.reshape(weights.shape + (1,) * (values.ndim - weights.ndim))


def safe_weighted_values(
    values: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
) -> tuple[Array, Array]:
    values_ = jnp.asarray(values)
    weights_ = jnp.asarray(weights, dtype=float)
    if values_.shape[: weights_.ndim] != weights_.shape:
        raise ValueError("weights must match the leading value axes.")
    included = jnp.ones(weights_.shape, dtype=bool)
    if mask is not None:
        included = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), weights_.shape)
    valid_weight = included & jnp.isfinite(weights_) & (weights_ >= 0.0)
    safe_weights = jnp.where(valid_weight, weights_, 0.0)
    contributing = valid_weight & (weights_ > 0.0)
    safe_values = jnp.where(_expanded_weights(contributing, values_), values_, 0)
    return safe_values, safe_weights


def weighted_sum(
    values: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    axis: int = -2,
    mask: ArrayLike | None = None,
) -> Array:
    values_, weights_ = safe_weighted_values(values, weights, mask=mask)
    return jnp.sum(_expanded_weights(weights_, values_) * values_, axis=axis)


def weighted_mean(
    values: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    axis: int = -2,
    mask: ArrayLike | None = None,
) -> Array:
    values_, weights_ = safe_weighted_values(values, weights, mask=mask)
    value_axis = int(axis) % values_.ndim
    if value_axis >= weights_.ndim:
        raise ValueError("Weighted reduction axis must belong to the weight prefix.")
    numerator = jnp.sum(
        _expanded_weights(weights_, values_) * values_,
        axis=value_axis,
    )
    denominator = jnp.sum(weights_, axis=value_axis)
    expanded = denominator.reshape(
        denominator.shape + (1,) * (numerator.ndim - denominator.ndim)
    )
    return jnp.where(
        expanded > 0.0,
        numerator / jnp.maximum(expanded, jnp.finfo(float).tiny),
        0,
    )


def effective_sample_size(weights: ArrayLike, /, *, axis: int = -1) -> Array:
    weights_ = jnp.asarray(weights, dtype=float)
    valid = jnp.all(
        jnp.isfinite(weights_) & (weights_ >= 0.0),
        axis=axis,
    )
    safe = jnp.where(jnp.isfinite(weights_) & (weights_ >= 0.0), weights_, 0.0)
    first = jnp.sum(safe, axis=axis)
    second = jnp.sum(safe * safe, axis=axis)
    value = jnp.where(second > 0.0, first * first / second, 0.0)
    return jnp.where(valid, value, 0.0)


def weighted_covariance(
    features: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    center: bool = True,
    correction: float = 0.0,
) -> tuple[Array, Array, Array]:
    x = jnp.asarray(features)
    w = jnp.asarray(weights, dtype=float)
    if x.ndim < 2 or w.shape != x.shape[:-1]:
        raise ValueError("features and weights must end in (sample, feature) and sample.")
    correction_ = float(correction)
    if not isfinite(correction_) or correction_ < 0.0:
        raise ValueError("correction must be finite and non-negative.")
    safe_x, safe_w = safe_weighted_values(x, w)
    mean = (
        weighted_mean(safe_x, safe_w, axis=-2)
        if center
        else jnp.zeros(x.shape[:-2] + (x.shape[-1],), dtype=x.dtype)
    )
    centered = safe_x - mean[..., None, :]
    centered = jnp.where(_expanded_weights(safe_w > 0.0, centered), centered, 0)
    scatter = oe.contract("...ni,...n,...nj->...ij", jnp.conj(centered), safe_w, centered)
    total = jnp.sum(safe_w, axis=-1)
    denominator = jnp.maximum(total - correction_, jnp.finfo(float).tiny)
    covariance = scatter / denominator[..., None, None]
    valid_weights = jnp.all(jnp.isfinite(w) & (w >= 0.0), axis=-1)
    valid = (
        valid_weights
        & (total > correction_)
        & jnp.all(jnp.isfinite(covariance), axis=(-2, -1))
    )
    return mean, covariance, valid


def segmented_weighted_sum(
    values: ArrayLike,
    segment_ids: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    num_segments: int,
    mask: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Reduce weighted samples into a fixed number of segments."""
    x = jnp.asarray(values)
    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    w = jnp.asarray(weights, dtype=float)
    if x.ndim < 1 or segments.shape != w.shape or x.shape[: w.ndim] != w.shape:
        raise ValueError("values, segment_ids, and weights must share case/sample axes.")
    count = int(num_segments)
    if count <= 0:
        raise ValueError("num_segments must be positive.")

    safe_x, safe_w = safe_weighted_values(x, w, mask=mask)
    in_range = (segments >= 0) & (segments < count)
    safe_w = jnp.where(in_range, safe_w, 0.0)
    safe_x = jnp.where(_expanded_weights(in_range, safe_x), safe_x, 0)
    membership = jax.nn.one_hot(segments, count, dtype=safe_w.dtype)
    membership = membership * safe_w[..., None]

    case_shape = w.shape[:-1]
    sample_count = w.shape[-1]
    value_shape = x.shape[w.ndim :]
    case_count = prod(case_shape)
    value_count = prod(value_shape)
    flat_membership = membership.reshape((case_count, sample_count, count))
    flat_values = safe_x.reshape((case_count, sample_count, value_count))
    totals = oe.contract("cnk,cnp->ckp", flat_membership, flat_values)
    mass = jnp.sum(flat_membership, axis=1)
    return (
        totals.reshape(case_shape + (count,) + value_shape),
        mass.reshape(case_shape + (count,)),
    )


def segmented_weighted_mean(
    values: ArrayLike,
    segment_ids: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    num_segments: int,
    mask: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Return fixed-capacity segment means and their total weights."""
    totals, mass = segmented_weighted_sum(
        values,
        segment_ids,
        weights,
        num_segments=num_segments,
        mask=mask,
    )
    value_rank = totals.ndim - mass.ndim
    expanded_mass = mass.reshape(mass.shape + (1,) * value_rank)
    means = jnp.where(
        expanded_mass > 0.0,
        totals / jnp.maximum(expanded_mass, jnp.finfo(float).tiny),
        0,
    )
    return means, mass


def class_weighted_moments(
    features: ArrayLike,
    labels: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    num_classes: int,
) -> tuple[Array, Array, Array, Array]:
    x = jnp.asarray(features)
    y = jnp.asarray(labels, dtype=jnp.int32)
    w = jnp.asarray(weights, dtype=float)
    if y.shape != x.shape[:-1] or w.shape != y.shape:
        raise ValueError("Class moments require aligned sample labels and weights.")
    classes = int(num_classes)
    if classes < 2:
        raise ValueError("num_classes must be at least two.")
    safe_x, safe_w = safe_weighted_values(x, w)
    label_valid = (y >= 0) & (y < classes)
    safe_w = jnp.where(label_valid, safe_w, 0.0)
    membership = jax.nn.one_hot(y, classes, dtype=safe_w.dtype)
    class_weights = safe_w[..., :, None] * membership
    mass = jnp.sum(class_weights, axis=-2)
    means = oe.contract("...nc,...nf->...cf", class_weights, safe_x)
    means = means / jnp.maximum(mass[..., :, None], jnp.finfo(float).tiny)
    centered = safe_x[..., :, None, :] - means[..., None, :, :]
    active = class_weights[..., :, :, None] > 0.0
    squared = jnp.real(centered * jnp.conj(centered))
    variance = jnp.sum(
        class_weights[..., :, :, None] * jnp.where(active, squared, 0),
        axis=-3,
    ) / jnp.maximum(mass[..., :, None], jnp.finfo(float).tiny)
    prior = mass / jnp.maximum(
        jnp.sum(mass, axis=-1, keepdims=True),
        jnp.finfo(float).tiny,
    )
    return mass, means, variance, prior


__all__ = [
    "class_weighted_moments",
    "effective_sample_size",
    "segmented_weighted_mean",
    "segmented_weighted_sum",
    "safe_weighted_values",
    "weighted_covariance",
    "weighted_mean",
    "weighted_sum",
]
