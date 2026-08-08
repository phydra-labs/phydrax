#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.kernels import AbstractPositiveDefiniteKernel


def _weighted_kernel_sum(
    kernel: AbstractPositiveDefiniteKernel,
    left: Array,
    left_weights: Array,
    right: Array,
    right_weights: Array,
    /,
    *,
    block_size: int,
) -> Array:
    left_count, coordinate_size = left.shape
    right_count = int(right.shape[0])
    block = int(block_size)
    left_blocks = (left_count + block - 1) // block
    right_blocks = (right_count + block - 1) // block
    left_padding = left_blocks * block - left_count
    right_padding = right_blocks * block - right_count
    padded_left = jnp.pad(left, ((0, left_padding), (0, 0)))
    padded_right = jnp.pad(right, ((0, right_padding), (0, 0)))
    padded_left_weights = jnp.pad(left_weights, (0, left_padding))
    padded_right_weights = jnp.pad(right_weights, (0, right_padding))

    def left_body(left_index, total):
        left_start = left_index * block
        left_points = jax.lax.dynamic_slice(
            padded_left,
            (left_start, 0),
            (block, coordinate_size),
        )
        left_mass = jax.lax.dynamic_slice(
            padded_left_weights,
            (left_start,),
            (block,),
        )

        def right_body(right_index, subtotal):
            right_start = right_index * block
            right_points = jax.lax.dynamic_slice(
                padded_right,
                (right_start, 0),
                (block, coordinate_size),
            )
            right_mass = jax.lax.dynamic_slice(
                padded_right_weights,
                (right_start,),
                (block,),
            )
            values = kernel.matrix(left_points, right_points)
            return subtotal + jnp.sum(left_mass[:, None] * values * right_mass[None, :])

        return jax.lax.fori_loop(
            0,
            right_blocks,
            right_body,
            total,
        )

    return jax.lax.fori_loop(
        0,
        left_blocks,
        left_body,
        jnp.asarray(0.0, dtype=left.dtype),
    )


def _weighted_kernel_mean(
    kernel: AbstractPositiveDefiniteKernel,
    points: Array,
    weights: Array,
    /,
    *,
    block_size: int,
) -> Array:
    count, coordinate_size = points.shape
    block = int(block_size)
    blocks = (count + block - 1) // block
    padding = blocks * block - count
    padded_points = jnp.pad(points, ((0, padding), (0, 0)))
    padded_weights = jnp.pad(weights, (0, padding))
    output = jnp.zeros((blocks * block,), dtype=points.dtype)

    def outer_body(outer_index, means):
        outer_start = outer_index * block
        outer_points = jax.lax.dynamic_slice(
            padded_points,
            (outer_start, 0),
            (block, coordinate_size),
        )

        def inner_body(inner_index, subtotal):
            inner_start = inner_index * block
            inner_points = jax.lax.dynamic_slice(
                padded_points,
                (inner_start, 0),
                (block, coordinate_size),
            )
            inner_weights = jax.lax.dynamic_slice(
                padded_weights,
                (inner_start,),
                (block,),
            )
            return subtotal + kernel.matrix(outer_points, inner_points) @ inner_weights

        block_mean = jax.lax.fori_loop(
            0,
            blocks,
            inner_body,
            jnp.zeros((block,), dtype=points.dtype),
        )
        return jax.lax.dynamic_update_slice(means, block_mean, (outer_start,))

    return jax.lax.fori_loop(0, blocks, outer_body, output)[:count]
