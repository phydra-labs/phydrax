#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array


CheckpointedScanMode: TypeAlias = Literal["full", "step", "block"]


def _reshape_blocks(value: Array, block_count: int, block_size: int, /) -> Array:
    return value.reshape((block_count, block_size) + value.shape[1:])


def _flatten_blocks(value: Array, /) -> Array:
    return value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:])


def checkpointed_scan(
    body: Callable[[Any, Any], tuple[Any, Any]],
    initial: Any,
    xs: Any,
    /,
    *,
    length: int,
    mode: CheckpointedScanMode,
    block_size: int | None = None,
) -> tuple[Any, Any]:
    """Run a fixed scan with full, per-step, or block rematerialization.

    Block mode checkpoints each complete inner scan and therefore retains only block
    boundary carries for reverse-mode replay. ``xs`` must be a PyTree whose leaves have
    the declared leading ``length``; callers with no inputs should pass ``arange(length)``.
    """

    count = int(length)
    if count <= 0:
        raise ValueError("checkpointed_scan length must be positive.")
    if mode not in ("full", "step", "block"):
        raise ValueError("Unknown checkpointed scan mode.")
    if mode != "block":
        if block_size is not None:
            raise ValueError("block_size is valid only for block checkpointing.")
        selected = jax.checkpoint(body) if mode == "step" else body
        return jax.lax.scan(selected, initial, xs, length=count)

    size = None if block_size is None else int(block_size)
    if size is None or size <= 0:
        raise ValueError("Block checkpointing requires a positive block_size.")
    if size > count:
        size = count

    leaves = jax.tree.leaves(xs)
    if not leaves:
        raise ValueError("Block checkpointing requires explicit scan inputs.")
    if any(value.shape[0] != count for value in leaves):
        raise ValueError("Every checkpointed scan input must match length.")

    complete_count = count // size
    complete_length = complete_count * size
    remainder = count - complete_length
    carry = initial
    complete_outputs = None

    if complete_count:
        complete_inputs = jax.tree.map(
            lambda value: _reshape_blocks(value[:complete_length], complete_count, size),
            xs,
        )

        def run_block(block_carry, block_inputs):
            return jax.lax.scan(body, block_carry, block_inputs, length=size)

        carry, block_outputs = jax.lax.scan(
            jax.checkpoint(run_block),
            carry,
            complete_inputs,
            length=complete_count,
        )
        complete_outputs = (
            None
            if block_outputs is None
            else jax.tree.map(_flatten_blocks, block_outputs)
        )

    if not remainder:
        return carry, complete_outputs

    remainder_inputs = jax.tree.map(lambda value: value[complete_length:], xs)
    carry, remainder_outputs = jax.lax.scan(
        jax.checkpoint(body),
        carry,
        remainder_inputs,
        length=remainder,
    )
    if complete_outputs is None:
        return carry, remainder_outputs
    if remainder_outputs is None:
        return carry, complete_outputs
    return carry, jax.tree.map(
        lambda complete, remaining: jnp.concatenate((complete, remaining), axis=0),
        complete_outputs,
        remainder_outputs,
    )


__all__ = ["CheckpointedScanMode", "checkpointed_scan"]
