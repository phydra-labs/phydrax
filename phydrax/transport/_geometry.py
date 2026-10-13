#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from ..typing import parse
from ._costs import GroundCost, PrecomputedCost


AutomaticSinkhornBlockSize: TypeAlias = Literal["auto"]
SinkhornBlockSize: TypeAlias = AutomaticSinkhornBlockSize | int | None
DEFAULT_SINKHORN_WORKSPACE_BYTES = 128 * 1024 * 1024


def validate_sinkhorn_workspace_bytes(workspace_bytes: int, /) -> None:
    """Validate the shared logical cost/kernel working-set budget."""
    if not isinstance(workspace_bytes, int) or isinstance(workspace_bytes, bool):
        raise TypeError("sinkhorn_workspace_bytes must be an integer.")
    if workspace_bytes < 1:
        raise ValueError("sinkhorn_workspace_bytes must be positive.")


def validate_sinkhorn_block_size(block_size: SinkhornBlockSize, /) -> SinkhornBlockSize:
    """Distinguish automatic admission from explicit dense and tile overrides."""
    if isinstance(block_size, str):
        return parse(block_size, AutomaticSinkhornBlockSize, "block_size")
    if block_size is not None:
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise TypeError("block_size must be 'auto', a positive integer, or None.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")
    return block_size


def resolve_sinkhorn_block_size(
    block_size: SinkhornBlockSize,
    workspace_bytes: int,
    source_count: int,
    target_count: int,
    itemsize: int,
    /,
) -> int | None:
    """Admit dense cost/kernel scratch or conservatively bound square tiles.

    Four scalar pair arrays account for costs, log coupling, coupling, and
    reduction scratch. The linear extent bound also bounds each square tile.
    Inputs, outputs, custom cost internals, and autodiff storage are separate.
    """
    if block_size != "auto":
        return block_size
    bytes_per_pair = 4 * itemsize
    if source_count * target_count * bytes_per_pair <= workspace_bytes:
        return None
    extent = max(source_count, target_count)
    return min(extent, max(1, workspace_bytes // (extent * bytes_per_pair)))


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
    source_count = source_points.shape[0]
    target_count = target_points.shape[0]
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

    def source_body(source_block: Array, result: Array) -> Array:
        source_start = source_block * size
        source_indices, source_valid = indices(source_start, size, source_count)
        accumulator = jnp.full((size,), -jnp.inf, dtype=values.dtype)

        def target_body(target_block: Array, current: Array) -> Array:
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
    source_count = source_points.shape[0]
    target_count = target_points.shape[0]
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

    def target_body(target_block: Array, result: Array) -> Array:
        target_start = target_block * size
        target_indices, target_valid = indices(target_start, size, target_count)
        accumulator = jnp.full((size,), -jnp.inf, dtype=values.dtype)

        def source_body(source_block: Array, current: Array) -> Array:
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
