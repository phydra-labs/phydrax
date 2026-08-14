from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

from ._kernels import (
    scatter_add,
    scatter_max,
    scatter_mean,
    scatter_min,
    segment_max,
    segment_max_or_constant,
    segment_mean,
    segment_min,
    segment_min_or_constant,
    segment_normalize,
    segment_softmax,
    segment_sum,
    segment_variance,
)


ArrayTree = Any


def partition_softmax(
    logits: jnp.ndarray,
    partitions: jnp.ndarray,
    *,
    sum_partitions: int | None = None,
) -> jnp.ndarray:
    n_partitions = int(partitions.shape[0])
    segment_ids = jnp.repeat(
        jnp.arange(n_partitions, dtype=jnp.int32),
        partitions,
        axis=0,
        total_repeat_length=sum_partitions,
    )
    return segment_softmax(logits, segment_ids, n_partitions)


def concatenated_args(
    update: Callable[..., ArrayTree] | None = None,
    *,
    axis: int = -1,
) -> Callable[..., ArrayTree]:
    """Decorator concatenating flattened arg/kwarg leaves into one array."""

    def _decorate(fn: Callable[..., ArrayTree]) -> Callable[..., ArrayTree]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> ArrayTree:
            leaves = jtu.tree_flatten(args)[0] + jtu.tree_flatten(kwargs)[0]
            if len(leaves) == 0:
                raise ValueError(
                    "concatenated_args received no array leaves to concatenate."
                )
            return fn(jnp.concatenate(leaves, axis=axis))

        return wrapper

    if update is not None:
        return _decorate(update)
    return _decorate


__all__ = [
    "segment_sum",
    "segment_mean",
    "segment_variance",
    "segment_normalize",
    "segment_max",
    "segment_max_or_constant",
    "segment_min_or_constant",
    "segment_min",
    "segment_softmax",
    "scatter_add",
    "scatter_mean",
    "scatter_max",
    "scatter_min",
    "partition_softmax",
    "concatenated_args",
]
