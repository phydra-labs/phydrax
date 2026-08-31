#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._interpolation import GatherStencil
from ..._numerics._compensated import compensated_sum, two_sum
from ...sparse import linear_transpose_apply, route_reduce
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


def _canonical_route_sum(
    stencil: GatherStencil,
    values: Array,
    stable_source_order: Array,
    target_size: int,
    /,
    *,
    compensated: bool,
) -> Array:
    indices = stencil.indices
    valid = stencil.valid
    weights = stencil.weights.astype(values.dtype)
    source_count, width = indices.shape
    output_shape = (int(target_size),) + values.shape[1:]
    zeros = jnp.zeros(output_shape, dtype=values.dtype)

    def step(route_index: int, carry):
        source_rank = route_index // width
        slot = route_index - source_rank * width
        source = stable_source_order[source_rank]
        target = indices[source, slot]
        route_valid = valid[source, slot]
        weight = weights[source, slot]
        payload = values[source] * weight
        payload = jnp.where(route_valid, payload, jnp.zeros((), dtype=payload.dtype))
        if not compensated:
            return carry.at[target].add(payload)
        total, correction = carry
        next_total, error = two_sum(total[target], payload)
        total = total.at[target].set(next_total)
        correction = correction.at[target].add(error)
        return total, correction

    route_count = source_count * width
    if compensated:
        total, correction = jax.lax.fori_loop(
            0,
            route_count,
            step,
            (zeros, zeros),
        )
        return total + correction
    return jax.lax.fori_loop(0, route_count, step, zeros)


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
    if accumulation == "fast":
        return linear_transpose_apply(stencil.relation, stencil.weights, values)
    if accumulation == "deterministic":
        return _canonical_route_sum(
            stencil,
            values,
            order,
            target_size,
            compensated=False,
        )
    if accumulation == "compensated":
        return _canonical_route_sum(
            stencil,
            values,
            order,
            target_size,
            compensated=True,
        )
    raise ValueError("Unknown splat accumulation policy.")


def _canonical_route_payload_sum(
    stencil: GatherStencil,
    route_values: Array,
    stable_source_order: Array,
    target_size: int,
    /,
    *,
    compensated: bool,
) -> Array:
    source_count, width = stencil.indices.shape
    payload_shape = route_values.shape[2:]
    zeros = jnp.zeros((int(target_size),) + payload_shape, dtype=route_values.dtype)

    def step(route_index: int, carry):
        source_rank = route_index // width
        slot = route_index - source_rank * width
        source = stable_source_order[source_rank]
        target = stencil.indices[source, slot]
        route_valid = stencil.valid[source, slot]
        payload = jnp.where(
            route_valid,
            route_values[source, slot],
            jnp.zeros((), dtype=route_values.dtype),
        )
        if not compensated:
            return carry.at[target].add(payload)
        total, correction = carry
        next_total, error = two_sum(total[target], payload)
        total = total.at[target].set(next_total)
        correction = correction.at[target].add(error)
        return total, correction

    route_count = source_count * width
    if compensated:
        total, correction = jax.lax.fori_loop(
            0,
            route_count,
            step,
            (zeros, zeros),
        )
        return total + correction
    return jax.lax.fori_loop(0, route_count, step, zeros)


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
    if accumulation == "fast":
        edge = stencil.relation.as_edge_relation().transpose()
        flattened = values.reshape((edge.capacity,) + values.shape[2:])
        return route_reduce(edge, flattened)
    if accumulation == "deterministic":
        return _canonical_route_payload_sum(
            stencil,
            values,
            order,
            target_size,
            compensated=False,
        )
    if accumulation == "compensated":
        return _canonical_route_payload_sum(
            stencil,
            values,
            order,
            target_size,
            compensated=True,
        )
    raise ValueError("Unknown splat accumulation policy.")


def certified_sum(value: ArrayLike, real_dtype: str, /, *, axis: int) -> Array:
    """Reduce one payload axis in certification precision with compensation."""
    return compensated_sum(cast_stage(value, real_dtype), axis=axis)


__all__ = ["cast_stage", "certified_sum", "deposit_routes", "stage_dtype"]
