#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reusable global-array placement for independent scientific axes."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._execution_runtime import ExecutionGroup
from ._trainable import place_array_leaves


def shard_array_axis(
    value: ArrayLike,
    execution_group: ExecutionGroup,
    /,
    *,
    axis: int = 0,
    mesh_axis: str | None = None,
    process_local: bool = False,
    global_axis_size: int | None = None,
) -> Array:
    """Place one global or process-local array along one assigned mesh axis."""

    array = jnp.asarray(value)
    array_axis = int(axis)
    if array_axis < 0:
        array_axis += array.ndim
    if not 0 <= array_axis < array.ndim:
        raise ValueError("axis is outside the array rank")
    selected_mesh_axis = (
        execution_group.mesh.axis_names[0] if mesh_axis is None else str(mesh_axis)
    )
    if selected_mesh_axis not in execution_group.mesh.axis_names:
        raise ValueError("mesh_axis is outside the execution-group mesh")
    partitions: list[str | None] = [None] * array.ndim
    partitions[array_axis] = selected_mesh_axis
    sharding = execution_group.named_sharding(partitions)
    if process_local:
        if global_axis_size is None or global_axis_size <= 0:
            raise ValueError("process-local array placement requires global_axis_size")
        global_shape = list(array.shape)
        global_shape[array_axis] = int(global_axis_size)
        if global_shape[array_axis] % int(execution_group.mesh.shape[selected_mesh_axis]):
            raise ValueError("global axis size must divide across the mesh axis")
        return jax.make_array_from_process_local_data(
            sharding,
            np.asarray(jax.device_get(array)),
            tuple(global_shape),
        )
    if array.shape[array_axis] % int(execution_group.mesh.shape[selected_mesh_axis]):
        raise ValueError("array axis size must divide across the mesh axis")
    return jax.device_put(array, sharding)


def shard_tree_axis(
    tree: Any,
    execution_group: ExecutionGroup,
    /,
    *,
    axis: int = 0,
    mesh_axis: str | None = None,
    process_local: bool = False,
    global_axis_size: int | None = None,
) -> Any:
    """Place every array leaf sharing one independent leading-axis contract."""

    return jax.tree.map(
        lambda leaf: (
            shard_array_axis(
                leaf,
                execution_group,
                axis=axis,
                mesh_axis=mesh_axis,
                process_local=process_local,
                global_axis_size=global_axis_size,
            )
            if eqx.is_array(leaf)
            else leaf
        ),
        tree,
    )


def replicate_tree(tree: Any, execution_group: ExecutionGroup, /) -> Any:
    """Replicate array leaves on an assigned execution-group mesh."""

    sharding = execution_group.named_sharding(())
    return place_array_leaves(tree, sharding)


def global_weighted_mean(
    values: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    mask: ArrayLike | None = None,
) -> Array:
    """Reduce a globally sharded weighted numerator and support exactly once."""

    value = jnp.asarray(values)
    weight = jnp.ones(value.shape, dtype=value.real.dtype)
    if weights is not None:
        weight = jnp.broadcast_to(jnp.asarray(weights, dtype=weight.dtype), value.shape)
    if mask is not None:
        valid = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), value.shape)
        weight = jnp.where(valid, weight, 0)
    numerator = jnp.sum(weight * value)
    support = jnp.sum(weight)
    safe_support = jnp.where(support > 0, support, 1)
    result = numerator / safe_support
    return jnp.where(support > 0, result, jnp.asarray(jnp.nan, dtype=result.dtype))


__all__ = (
    "global_weighted_mean",
    "replicate_tree",
    "shard_array_axis",
    "shard_tree_axis",
)
