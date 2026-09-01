#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def clamped_log_barrier(value: ArrayLike, activation: ArrayLike, /) -> Array:
    """C2-clamped logarithmic barrier on a positive scalar coordinate."""
    coordinate = jnp.asarray(value)
    threshold = jnp.asarray(activation, dtype=coordinate.dtype)
    safe = jnp.maximum(coordinate, jnp.finfo(coordinate.dtype).tiny)
    interior = -((coordinate - threshold) ** 2) * jnp.log(safe / threshold)
    return jnp.where(
        coordinate <= 0.0,
        jnp.asarray(jnp.inf, dtype=coordinate.dtype),
        jnp.where(coordinate < threshold, interior, 0.0),
    )


def clamped_log_barrier_first_derivative(
    value: ArrayLike,
    activation: ArrayLike,
    /,
) -> Array:
    coordinate = jnp.asarray(value)
    threshold = jnp.asarray(activation, dtype=coordinate.dtype)
    safe = jnp.maximum(coordinate, jnp.finfo(coordinate.dtype).tiny)
    derivative = (threshold - coordinate) * (
        2.0 * jnp.log(safe / threshold) - threshold / safe + 1.0
    )
    return jnp.where(coordinate < threshold, derivative, 0.0)


def clamped_log_barrier_second_derivative(
    value: ArrayLike,
    activation: ArrayLike,
    /,
) -> Array:
    coordinate = jnp.asarray(value)
    threshold = jnp.asarray(activation, dtype=coordinate.dtype)
    safe = jnp.maximum(coordinate, jnp.finfo(coordinate.dtype).tiny)
    ratio = threshold / safe
    derivative = (ratio + 2.0) * ratio - 2.0 * jnp.log(safe / threshold) - 3.0
    return jnp.where(coordinate < threshold, derivative, 0.0)


def physical_barrier_scale(
    activation_distance: ArrayLike,
    minimum_separation: ArrayLike,
    /,
) -> Array:
    """Scale a squared-distance clamped barrier to one physical length unit."""
    distance = jnp.asarray(activation_distance)
    separation = jnp.asarray(minimum_separation, dtype=distance.dtype)
    squared_activation = (2.0 * separation + distance) * distance
    return distance / (squared_activation * squared_activation)


def physical_clamped_log_barrier(
    distance_squared: ArrayLike,
    activation_distance: ArrayLike,
    minimum_separation: ArrayLike = 0.0,
    /,
) -> Array:
    squared = jnp.asarray(distance_squared)
    distance = jnp.asarray(activation_distance, dtype=squared.dtype)
    separation = jnp.asarray(minimum_separation, dtype=squared.dtype)
    shifted = squared - separation * separation
    threshold = (2.0 * separation + distance) * distance
    return physical_barrier_scale(distance, separation) * clamped_log_barrier(
        shifted, threshold
    )


__all__ = [
    "clamped_log_barrier",
    "clamped_log_barrier_first_derivative",
    "clamped_log_barrier_second_derivative",
    "physical_barrier_scale",
    "physical_clamped_log_barrier",
]
