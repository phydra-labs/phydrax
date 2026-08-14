#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array

from ._costs import GroundCost, PrecomputedCost


def cost_matrix(cost: GroundCost, source_points: Array, target_points: Array, /) -> Array:
    """Materialize a finite cost geometry independently of a transport problem."""
    if isinstance(cost, PrecomputedCost):
        return cost.values
    return cost.matrix(source_points, target_points)


def cost_block(
    cost: GroundCost,
    source_points: Array,
    target_points: Array,
    source_indices: Array,
    target_indices: Array,
    /,
) -> Array:
    """Evaluate one rectangular cost block without owning measure semantics."""
    if isinstance(cost, PrecomputedCost):
        return cost.values[source_indices[:, None], target_indices[None, :]]
    return cost.matrix(source_points[source_indices], target_points[target_indices])


def row_logsumexp(
    cost: GroundCost,
    source_points: Array,
    target_points: Array,
    log_values: Array,
    epsilon: Array,
    /,
    *,
    block_size: int | None,
) -> Array:
    """Reduce ``log_values - cost / epsilon`` over target atoms."""
    values = jnp.asarray(log_values)
    source_count = int(source_points.shape[0])
    target_count = int(target_points.shape[0])
    if values.shape != (target_count,):
        raise ValueError("row_logsumexp values must match target atom count.")
    if block_size is None:
        return logsumexp(
            values[None, :] - cost_matrix(cost, source_points, target_points) / epsilon,
            axis=1,
        )
    size = int(block_size)
    source_blocks = block_count(source_count, size)
    target_blocks = block_count(target_count, size)
    output = jnp.full((source_blocks * size,), -jnp.inf, dtype=values.dtype)

    def source_body(source_block, result):
        source_start = source_block * size
        source_indices, source_valid = indices(source_start, size, source_count)
        accumulator = jnp.full((size,), -jnp.inf, dtype=values.dtype)

        def target_body(target_block, current):
            target_start = target_block * size
            target_indices, target_valid = indices(target_start, size, target_count)
            costs = cost_block(
                cost,
                source_points,
                target_points,
                source_indices,
                target_indices,
            )
            block_values = jnp.take(values, target_indices, axis=0)
            terms = block_values[None, :] - costs / epsilon
            terms = jnp.where(
                source_valid[:, None] & target_valid[None, :], terms, -jnp.inf
            )
            return jnp.logaddexp(current, logsumexp(terms, axis=1))

        accumulator = jax.lax.fori_loop(0, target_blocks, target_body, accumulator)
        accumulator = jnp.where(source_valid, accumulator, -jnp.inf)
        return jax.lax.dynamic_update_slice(result, accumulator, (source_start,))

    output = jax.lax.fori_loop(0, source_blocks, source_body, output)
    return output[:source_count]


def column_logsumexp(
    cost: GroundCost,
    source_points: Array,
    target_points: Array,
    log_values: Array,
    epsilon: Array,
    /,
    *,
    block_size: int | None,
) -> Array:
    """Reduce ``log_values - cost / epsilon`` over source atoms."""
    values = jnp.asarray(log_values)
    source_count = int(source_points.shape[0])
    target_count = int(target_points.shape[0])
    if values.shape != (source_count,):
        raise ValueError("column_logsumexp values must match source atom count.")
    if block_size is None:
        return logsumexp(
            values[:, None] - cost_matrix(cost, source_points, target_points) / epsilon,
            axis=0,
        )
    size = int(block_size)
    source_blocks = block_count(source_count, size)
    target_blocks = block_count(target_count, size)
    output = jnp.full((target_blocks * size,), -jnp.inf, dtype=values.dtype)

    def target_body(target_block, result):
        target_start = target_block * size
        target_indices, target_valid = indices(target_start, size, target_count)
        accumulator = jnp.full((size,), -jnp.inf, dtype=values.dtype)

        def source_body(source_block, current):
            source_start = source_block * size
            source_indices, source_valid = indices(source_start, size, source_count)
            costs = cost_block(
                cost,
                source_points,
                target_points,
                source_indices,
                target_indices,
            )
            block_values = jnp.take(values, source_indices, axis=0)
            terms = block_values[:, None] - costs / epsilon
            terms = jnp.where(
                source_valid[:, None] & target_valid[None, :], terms, -jnp.inf
            )
            return jnp.logaddexp(current, logsumexp(terms, axis=0))

        accumulator = jax.lax.fori_loop(0, source_blocks, source_body, accumulator)
        accumulator = jnp.where(target_valid, accumulator, -jnp.inf)
        return jax.lax.dynamic_update_slice(result, accumulator, (target_start,))

    output = jax.lax.fori_loop(0, target_blocks, target_body, output)
    return output[:target_count]


def indices(start: Array, size: int, count: int, /) -> tuple[Array, Array]:
    """Return clipped static block indices and their validity mask."""
    raw = start + jnp.arange(size, dtype=jnp.int32)
    return jnp.minimum(raw, count - 1), raw < count


def block_count(count: int, block_size: int, /) -> int:
    """Return the number of fixed-size blocks needed for a finite axis."""
    return (int(count) + int(block_size) - 1) // int(block_size)


__all__ = [
    "block_count",
    "column_logsumexp",
    "cost_block",
    "cost_matrix",
    "indices",
    "row_logsumexp",
]
