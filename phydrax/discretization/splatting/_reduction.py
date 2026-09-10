#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._interpolation import GatherStencil
from ..._numerics._compensated import compensated_sum
from ...sparse import canonical_row_route_ids, RelationExecutionPlan
from ._types import SplatAccumulation


def stage_dtype(value: ArrayLike, real_dtype: str, /) -> jnp.dtype:
    """Resolve a real precision stage while preserving complex payloads."""
    array = jnp.asarray(value)
    real = jnp.dtype(real_dtype)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        return jnp.dtype(
            jnp.complex64 if real == jnp.dtype(jnp.float32) else jnp.complex128
        )
    return real


def cast_stage(value: ArrayLike, real_dtype: str, /) -> Array:
    """Cast one real or complex payload into a declared real precision stage."""
    return jnp.asarray(value).astype(stage_dtype(value, real_dtype))


def _reduce_routes(
    stencil: GatherStencil,
    route_values: Array,
    stable_source_order: Array,
    target_size: int,
    accumulation: SplatAccumulation,
    /,
) -> Array:
    edge = stencil.relation.as_edge_relation().transpose()
    if edge.target_size != int(target_size):
        raise ValueError("target_size does not match the stencil source space.")
    stable_route_ids = canonical_row_route_ids(
        stable_source_order, stencil.indices.shape[1]
    )
    execution = RelationExecutionPlan().prepare(edge, stable_route_ids=stable_route_ids)
    reduced, _ = execution.reduce(
        route_values.reshape((edge.capacity,) + route_values.shape[2:]),
        accumulation=accumulation,
        output="dense",
    )
    return reduced


def deposit_routes(
    stencil: GatherStencil,
    source_values: ArrayLike,
    stable_source_order: ArrayLike,
    target_size: int,
    accumulation: SplatAccumulation,
    /,
) -> Array:
    """Apply the masked source-to-target transpose under one accumulation policy."""
    values = jnp.asarray(source_values)
    if values.ndim < 1 or int(values.shape[0]) != stencil.relation.targets_per_case:
        raise ValueError("Splat source values must begin with the stencil source count.")
    order = jnp.asarray(stable_source_order, dtype=jnp.int32)
    if order.shape != (values.shape[0],):
        raise ValueError("Stable source order must contain every source exactly once.")
    payload = values[:, None] * stencil.weights.astype(values.dtype).reshape(
        stencil.weights.shape + (1,) * (values.ndim - 1)
    )
    return _reduce_routes(stencil, payload, order, target_size, accumulation)


def _scatter_route_payload(
    stencil: GatherStencil,
    route_values: ArrayLike,
    stable_source_order: ArrayLike,
    target_size: int,
    accumulation: SplatAccumulation,
    /,
) -> Array:
    """Reduce preweighted per-route payloads onto the stencil source layout."""
    values = jnp.asarray(route_values)
    route_shape = stencil.indices.shape
    if values.ndim < 2 or tuple(int(size) for size in values.shape[:2]) != route_shape:
        raise ValueError(
            f"Route payload must begin with route shape {route_shape}; got {values.shape}."
        )
    order = jnp.asarray(stable_source_order, dtype=jnp.int32)
    if order.shape != (route_shape[0],):
        raise ValueError("Stable source order must contain every source exactly once.")
    return _reduce_routes(stencil, values, order, target_size, accumulation)


def certified_sum(value: ArrayLike, real_dtype: str, /, *, axis: int) -> Array:
    """Reduce one payload axis in certification precision with compensation."""
    return compensated_sum(cast_stage(value, real_dtype), axis=axis)


__all__ = ["cast_stage", "certified_sum", "deposit_routes", "stage_dtype"]
