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
from jax import Array
from jax.sharding import Sharding
from jax.typing import ArrayLike

from ._execution_runtime import ExecutionGroup


@jax.jit
def _slice_local_leading_axes(
    values: tuple[Array, ...], indices: tuple[int, ...]
) -> tuple[Array, ...]:
    """Project local numerical leaves with dynamic leading-axis indices."""
    return tuple(value[index] for value, index in zip(values, indices, strict=True))


def _project_local_leading_axes(
    payloads: tuple[tuple[Array, int, jax.Device], ...],
) -> tuple[Array, ...]:
    """Project admitted local shards without moving leaves between devices."""
    grouped: dict[jax.Device, list[tuple[int, Array, int]]] = {}
    for index, (value, local_index, device) in enumerate(payloads):
        grouped.setdefault(device, []).append((index, value, local_index))
    result = [payload[0] for payload in payloads]
    for entries in grouped.values():
        selected = _slice_local_leading_axes(
            tuple(entry[1] for entry in entries),
            tuple(entry[2] for entry in entries),
        )
        for entry, value in zip(entries, selected, strict=True):
            result[entry[0]] = value
    return tuple(result)


def _place_process_local_array(value: Array, sharding: Sharding, /) -> Array:
    """Assemble current global placement from a complete process-local operand.

    Only current addressable device buffers are produced. No host fetch,
    cross-host device transfer, or global bank materialization is performed.
    Callers agree local admission before entering a numerical collective.
    """
    if not value.is_fully_addressable or value.is_deleted():
        raise ValueError(
            "collective.operand must retain a complete live process-local array."
        )
    if sharding.is_fully_addressable:
        return jax.device_put(value, sharding)
    indices = sharding.addressable_devices_indices_map(value.shape)
    if not indices or any(index is None for index in indices.values()):
        raise ValueError(
            "collective.placement must identify current process-addressable slices."
        )
    buffers = [
        jax.device_put(value[index], device) for device, index in indices.items()
    ]
    return jax.make_array_from_single_device_arrays(value.shape, sharding, buffers)


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
        if global_shape[array_axis] % execution_group.mesh.shape[selected_mesh_axis]:
            raise ValueError("global axis size must divide across the mesh axis")
        return jax.make_array_from_process_local_data(
            sharding,
            np.asarray(jax.device_get(array)),
            tuple(global_shape),
        )
    if array.shape[array_axis] % execution_group.mesh.shape[selected_mesh_axis]:
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
    """Replicate every array leaf on an assigned execution-group mesh.

    Placement is independent of array roles: FIXED data is replicated like
    parameters, and non-array leaves are returned unchanged.
    """

    arrays, rest = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.device_put(arrays, execution_group.named_sharding(())), rest)


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
        valid = jnp.broadcast_to(jnp.asarray(mask, dtype=jnp.bool_), value.shape)
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
