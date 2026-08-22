#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


def normalized_log_weights(log_weights: Array, valid: Array, /) -> Array:
    """Normalize finite log weights over valid entries only."""
    values = jnp.asarray(log_weights, dtype=float)
    validity = jnp.asarray(valid, dtype=bool)
    if values.shape != validity.shape:
        raise ValueError("log_weights and valid must have identical shapes.")
    valid_count = jnp.sum(validity)
    valid_count = eqx.error_if(
        valid_count,
        valid_count <= 0,
        "Weighted sample batch contains no valid entries.",
    )
    safe = jnp.where(validity, values, -jnp.inf)
    reference = jnp.max(safe) + 0.0 * valid_count
    unnormalized = jnp.where(validity, jnp.exp(values - reference), 0.0)
    mass = jnp.sum(unnormalized)
    mass = eqx.error_if(
        mass,
        ~(jnp.isfinite(mass) & (mass > 0.0)),
        "Weighted sample batch has zero finite mass.",
    )
    return unnormalized / mass


def weighted_mean(values: Array, weights: Array, /) -> Array:
    """Reduce every weight axis while preserving trailing value axes."""
    array = jnp.asarray(values)
    mass = jnp.asarray(weights, dtype=float)
    if array.shape[: mass.ndim] != mass.shape:
        raise ValueError("values must begin with the complete weight shape.")
    expanded = mass.reshape(mass.shape + (1,) * (array.ndim - mass.ndim))
    return jnp.sum(expanded * array, axis=tuple(range(mass.ndim)))


def effective_sample_size(weights: Array, /) -> Array:
    """Return inverse squared mass for already-normalized nonnegative weights."""
    mass = jnp.asarray(weights, dtype=float)
    return 1.0 / jnp.sum(mass**2)


def clustered_standard_error(
    values: Array,
    weights: Array,
    cluster_indices: Array,
    num_clusters: int,
    /,
) -> Array:
    """Estimate uncertainty from weighted independent cluster aggregates."""
    node_values = jnp.asarray(values)
    mass = jnp.asarray(weights, dtype=float)
    indices = jnp.asarray(cluster_indices, dtype=jnp.int32)
    count = int(num_clusters)
    if node_values.shape != mass.shape or indices.shape != mass.shape:
        raise ValueError("Cluster values, weights, and indices must have one shape.")
    if count <= 0:
        raise ValueError("num_clusters must be positive.")
    cluster_weight = jax.ops.segment_sum(mass, indices, count)
    cluster_total = jax.ops.segment_sum(mass * node_values, indices, count)
    active = cluster_weight > 0.0
    cluster_mean = jnp.where(
        active,
        cluster_total / jnp.maximum(cluster_weight, jnp.finfo(mass.dtype).tiny),
        0.0,
    )
    active_count = jnp.sum(active)
    mean = jnp.sum(jnp.where(active, cluster_mean, 0.0)) / jnp.maximum(active_count, 1)
    squared = jnp.sum(jnp.where(active, (cluster_mean - mean) ** 2, 0.0))
    variance = jnp.where(active_count > 1, squared / (active_count - 1), jnp.nan)
    return jnp.sqrt(variance / jnp.maximum(active_count, 1))


__all__ = [
    "clustered_standard_error",
    "effective_sample_size",
    "normalized_log_weights",
    "weighted_mean",
]
