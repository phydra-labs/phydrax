#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def quantile_bin_edges(features: ArrayLike, /, *, num_bins: int) -> Array:
    """Deterministic hard quantile edges for each feature."""
    x = jnp.asarray(features)
    bins = int(num_bins)
    if x.ndim != 2 or bins < 2:
        raise ValueError("features must be rank-2 and num_bins at least two.")
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        raise TypeError("Quantile binning requires real-valued features.")
    quantiles = jnp.linspace(0.0, 1.0, bins + 1, dtype=x.real.dtype)[1:-1]
    return jnp.quantile(x, quantiles, axis=0).T


def assign_bins(features: ArrayLike, edges: ArrayLike, /) -> Array:
    x = jnp.asarray(features)
    edges_ = jnp.asarray(edges)
    if x.ndim != 2 or edges_.ndim != 2 or x.shape[1] != edges_.shape[0]:
        raise ValueError("features and edges must align by feature.")
    return jax.vmap(lambda row: jax.vmap(jnp.searchsorted)(edges_, row))(x).astype(
        jnp.int32
    )


def histogram_gradient_statistics(
    bins: ArrayLike,
    gradients: ArrayLike,
    hessians: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    num_bins: int,
) -> tuple[Array, Array, Array]:
    """Aggregate weighted gradient, Hessian, and count per feature/bin."""
    bin_indices = jnp.asarray(bins, dtype=jnp.int32)
    gradient = jnp.asarray(gradients)
    hessian = jnp.asarray(hessians)
    weight = jnp.asarray(weights, dtype=float)
    if bin_indices.ndim != 2 or gradient.shape[0] != bin_indices.shape[0]:
        raise ValueError("Histogram statistics require aligned samples.")
    if hessian.shape != gradient.shape or weight.shape != (bin_indices.shape[0],):
        raise ValueError("Gradient, Hessian, and weights must align.")
    if int(num_bins) <= 0:
        raise ValueError("num_bins must be positive.")
    output_shape = gradient.shape[1:]
    features = bin_indices.shape[1]

    def one_feature(feature_bins):
        active = (
            jnp.isfinite(weight)
            & (weight >= 0.0)
            & (feature_bins >= 0)
            & (feature_bins < int(num_bins))
        )
        safe_bins = jnp.clip(feature_bins, 0, int(num_bins) - 1)
        safe_weight = jnp.where(active, weight, 0.0)
        contributing = active & (weight > 0.0)
        expanded_active = contributing.reshape(
            contributing.shape + (1,) * len(output_shape)
        )
        safe_gradient = jnp.where(expanded_active, gradient, 0)
        safe_hessian = jnp.where(expanded_active, hessian, 0)
        expanded_weight = safe_weight.reshape(
            safe_weight.shape + (1,) * len(output_shape)
        )
        g = jnp.zeros((num_bins,) + output_shape, dtype=gradient.dtype)
        h = jnp.zeros((num_bins,) + output_shape, dtype=hessian.dtype)
        c = jnp.zeros((num_bins,), dtype=weight.dtype)
        g = g.at[safe_bins].add(expanded_weight * safe_gradient)
        h = h.at[safe_bins].add(expanded_weight * safe_hessian)
        c = c.at[safe_bins].add(safe_weight)
        return g, h, c

    gradients_by_bin, hessians_by_bin, counts = jax.vmap(
        one_feature, in_axes=1, out_axes=0
    )(bin_indices)
    return (
        gradients_by_bin.reshape((features, num_bins) + output_shape),
        hessians_by_bin.reshape((features, num_bins) + output_shape),
        counts.reshape((features, num_bins)),
    )


def xgboost_leaf_weight(
    gradient_sum: ArrayLike,
    hessian_sum: ArrayLike,
    /,
    *,
    l2_regularization: ArrayLike,
    l1_regularization: ArrayLike = 0.0,
    max_delta_step: ArrayLike = 0.0,
) -> Array:
    gradient = jnp.asarray(gradient_sum)
    hessian = jnp.asarray(hessian_sum)
    l1 = jnp.asarray(l1_regularization)
    shrunk = jnp.sign(gradient) * jnp.maximum(jnp.abs(gradient) - l1, 0.0)
    value = -shrunk / jnp.maximum(
        hessian + jnp.asarray(l2_regularization), jnp.finfo(float).tiny
    )
    limit = jnp.asarray(max_delta_step)
    return jnp.where(limit > 0.0, jnp.clip(value, -limit, limit), value)


def xgboost_split_gain(
    left_gradient: ArrayLike,
    left_hessian: ArrayLike,
    right_gradient: ArrayLike,
    right_hessian: ArrayLike,
    /,
    *,
    l2_regularization: ArrayLike,
    minimum_gain: ArrayLike = 0.0,
    l1_regularization: ArrayLike = 0.0,
) -> Array:
    lg = jnp.asarray(left_gradient)
    lh = jnp.asarray(left_hessian)
    rg = jnp.asarray(right_gradient)
    rh = jnp.asarray(right_hessian)
    regularization = jnp.asarray(l2_regularization)
    l1 = jnp.asarray(l1_regularization)

    def score(g, h):
        shrunk = jnp.sign(g) * jnp.maximum(jnp.abs(g) - l1, 0.0)
        return jnp.sum(
            shrunk * shrunk / jnp.maximum(h + regularization, jnp.finfo(float).tiny)
        )

    return 0.5 * (score(lg, lh) + score(rg, rh) - score(lg + rg, lh + rh)) - jnp.asarray(
        minimum_gain
    )


__all__ = [
    "assign_bins",
    "histogram_gradient_statistics",
    "quantile_bin_edges",
    "xgboost_leaf_weight",
    "xgboost_split_gain",
]
