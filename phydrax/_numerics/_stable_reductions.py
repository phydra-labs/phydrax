#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def log_normalize(
    log_weights: Array,
    /,
    *,
    axes: int | tuple[int, ...] = 0,
    mask: Array | None = None,
    keepdims: bool = False,
) -> tuple[Array, Array, Array]:
    """Normalize masked log weights independently over explicit sample axes."""
    log_weights_ = jnp.asarray(log_weights, dtype=float)
    raw_axes = (axes,) if isinstance(axes, int) else tuple(axes)
    if not raw_axes:
        raise ValueError("axes must contain at least one reduction axis.")
    resolved_axes = tuple(
        axis + log_weights_.ndim if axis < 0 else axis for axis in raw_axes
    )
    if any(axis < 0 or axis >= log_weights_.ndim for axis in resolved_axes):
        raise ValueError("axes contains an out-of-range reduction axis.")
    if len(set(resolved_axes)) != len(resolved_axes):
        raise ValueError("axes must not contain duplicates.")
    included = (
        jnp.ones(log_weights_.shape, dtype=bool)
        if mask is None
        else jnp.broadcast_to(jnp.asarray(mask, dtype=bool), log_weights_.shape)
    )
    finite = jnp.isfinite(log_weights_)
    admissible = finite | jnp.isneginf(log_weights_)
    inputs_valid = jnp.all(~included | admissible, axis=resolved_axes, keepdims=True)
    positive = jnp.any(included & finite, axis=resolved_axes, keepdims=True)
    valid = inputs_valid & positive
    active = included & finite
    maximum = jnp.max(
        jnp.where(active, log_weights_, -jnp.inf),
        axis=resolved_axes,
        keepdims=True,
        initial=-jnp.inf,
    )
    safe_maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    scaled = jnp.where(active, jnp.exp(log_weights_ - safe_maximum), 0.0)
    total = jnp.sum(scaled, axis=resolved_axes, keepdims=True)
    safe_total = jnp.maximum(total, jnp.finfo(float).tiny)
    normalized = jnp.where(valid, scaled / safe_total, 0.0)
    log_sum = jnp.where(valid, safe_maximum + jnp.log(safe_total), -jnp.inf)
    if keepdims:
        return normalized, log_sum, valid
    return (
        normalized,
        jnp.squeeze(log_sum, axis=resolved_axes),
        jnp.squeeze(valid, axis=resolved_axes),
    )


def signed_logsumexp(
    log_magnitudes: Array,
    signs: Array,
    /,
    *,
    axis: int = 0,
) -> tuple[Array, Array]:
    """Return sign and log magnitude of a cancellation-aware signed sum."""
    log_magnitudes_ = jnp.asarray(log_magnitudes, dtype=float)
    signs_ = jnp.asarray(signs, dtype=float)
    maximum = jnp.max(log_magnitudes_, axis=axis, keepdims=True)
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    scaled = jnp.sum(signs_ * jnp.exp(log_magnitudes_ - maximum), axis=axis)
    scale = jnp.squeeze(maximum, axis=axis)
    return jnp.sign(scaled), scale + jnp.log(jnp.abs(scaled))


def weight_ess(normalized_weights: Array, /, *, axis: int = 0) -> Array:
    """Effective sample size for already-normalized nonnegative weights."""
    weights = jnp.asarray(normalized_weights, dtype=float)
    return 1.0 / jnp.maximum(jnp.sum(weights * weights, axis=axis), jnp.finfo(float).tiny)


__all__ = ["log_normalize", "signed_logsumexp", "weight_ess"]
