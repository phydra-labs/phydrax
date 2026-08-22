#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike

from ._relation import EdgeRelation, RowRelation, SparseRelation


RouteReduction: TypeAlias = Literal["sum", "mean", "max", "min"]


def _require_prefix(
    name: str,
    array: Array,
    prefix: tuple[int, ...],
    /,
) -> tuple[int, ...]:
    if (
        array.ndim < len(prefix)
        or tuple(int(size) for size in array.shape[: len(prefix)]) != prefix
    ):
        raise ValueError(f"{name} must begin with shape {prefix}; got {array.shape}.")
    return tuple(int(size) for size in array.shape[len(prefix) :])


def _expanded_mask(valid: Array, payload_ndim: int, /) -> Array:
    return valid.reshape(valid.shape + (1,) * int(payload_ndim))


def mask_routes(relation: SparseRelation, values: Any, /) -> Any:
    """Make invalid route payloads numerically inert without changing their layout."""
    route_shape = relation.route_shape

    def mask_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_shape = _require_prefix("Route values", array, route_shape)
        valid = _expanded_mask(relation.valid, len(payload_shape))
        return jnp.where(valid, array, jnp.zeros((), dtype=array.dtype))

    return jtu.tree_map(mask_leaf, values)


def _gather_edge_leaf(relation: EdgeRelation, value: Any, /) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Edge source values", array, relation.input_shape)
    safe = jnp.where(relation.valid, relation.source_indices, 0)
    gathered = array[safe]
    valid = _expanded_mask(relation.valid, len(payload_shape))
    return jnp.where(valid, gathered, jnp.zeros((), dtype=gathered.dtype))


def _gather_row_leaf(relation: RowRelation, value: Any, /) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Row source values", array, relation.input_shape)
    cases = relation.num_cases
    targets = relation.targets_per_case
    width = relation.width
    flattened = array.reshape((cases, relation.source_size) + payload_shape)
    safe = jnp.where(relation.valid, relation.source_indices, 0).reshape(
        (cases, targets, width)
    )
    case_indices = jnp.arange(cases, dtype=jnp.int32).reshape((cases, 1, 1))
    gathered = flattened[case_indices, safe].reshape(relation.route_shape + payload_shape)
    valid = _expanded_mask(relation.valid, len(payload_shape))
    return jnp.where(valid, gathered, jnp.zeros((), dtype=gathered.dtype))


def gather_routes(relation: SparseRelation, values: Any, /) -> Any:
    """Gather source payloads onto routes using the relation's canonical source axis."""
    if isinstance(relation, EdgeRelation):
        return jtu.tree_map(lambda value: _gather_edge_leaf(relation, value), values)
    if isinstance(relation, RowRelation):
        return jtu.tree_map(lambda value: _gather_row_leaf(relation, value), values)
    raise TypeError("relation must be an EdgeRelation or RowRelation.")


def _extreme_fill(dtype: jnp.dtype, reduction: RouteReduction, /) -> Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise TypeError(f"Route reduction {reduction!r} does not support complex values.")
    if jnp.issubdtype(dtype, jnp.bool_):
        return jnp.asarray(reduction == "min", dtype=dtype)
    if jnp.issubdtype(dtype, jnp.inexact):
        value = -jnp.inf if reduction == "max" else jnp.inf
        return jnp.asarray(value, dtype=dtype)
    limits = jnp.iinfo(dtype)
    value = limits.min if reduction == "max" else limits.max
    return jnp.asarray(value, dtype=dtype)


def _finalize_reduction(
    reduced: Array,
    counts: Array,
    payload_ndim: int,
    reduction: RouteReduction,
    /,
) -> Array:
    expanded_counts = counts.reshape(counts.shape + (1,) * payload_ndim)
    if reduction == "mean":
        denominator = jnp.maximum(expanded_counts, 1).astype(reduced.dtype)
        return reduced / denominator
    if reduction in ("max", "min"):
        return jnp.where(expanded_counts > 0, reduced, jnp.zeros_like(reduced))
    return reduced


def _reduce_edge_leaf(
    relation: EdgeRelation,
    value: Any,
    reduction: RouteReduction,
    /,
) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Edge route values", array, relation.route_shape)
    payload_ndim = len(payload_shape)
    valid = _expanded_mask(relation.valid, payload_ndim)
    safe_target = jnp.where(relation.valid, relation.target_indices, 0)
    counts = jax.ops.segment_sum(
        relation.valid.astype(jnp.int32),
        safe_target,
        relation.target_size,
    )
    if reduction in ("sum", "mean"):
        material = jnp.where(valid, array, jnp.zeros((), dtype=array.dtype))
        reduced = jax.ops.segment_sum(material, safe_target, relation.target_size)
    elif reduction == "max":
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jax.ops.segment_max(material, safe_target, relation.target_size)
    else:
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jax.ops.segment_min(material, safe_target, relation.target_size)
    return _finalize_reduction(reduced, counts, payload_ndim, reduction)


def _reduce_row_leaf(
    relation: RowRelation,
    value: Any,
    reduction: RouteReduction,
    /,
) -> Array:
    array = jnp.asarray(value)
    payload_shape = _require_prefix("Row route values", array, relation.route_shape)
    payload_ndim = len(payload_shape)
    valid = _expanded_mask(relation.valid, payload_ndim)
    route_axis = len(relation.route_shape) - 1
    counts = jnp.sum(relation.valid, axis=-1, dtype=jnp.int32)
    if reduction in ("sum", "mean"):
        material = jnp.where(valid, array, jnp.zeros((), dtype=array.dtype))
        reduced = jnp.sum(material, axis=route_axis)
    elif reduction == "max":
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jnp.max(material, axis=route_axis)
    else:
        material = jnp.where(valid, array, _extreme_fill(array.dtype, reduction))
        reduced = jnp.min(material, axis=route_axis)
    return _finalize_reduction(reduced, counts, payload_ndim, reduction)


def route_reduce(
    relation: SparseRelation,
    values: Any,
    /,
    *,
    reduction: RouteReduction = "sum",
) -> Any:
    """Reduce route payloads only onto the relation's declared target axis."""
    if reduction not in ("sum", "mean", "max", "min"):
        raise ValueError("reduction must be 'sum', 'mean', 'max', or 'min'.")
    if isinstance(relation, EdgeRelation):
        return jtu.tree_map(
            lambda value: _reduce_edge_leaf(relation, value, reduction), values
        )
    if isinstance(relation, RowRelation):
        return jtu.tree_map(
            lambda value: _reduce_row_leaf(relation, value, reduction), values
        )
    raise TypeError("relation must be an EdgeRelation or RowRelation.")


def _coefficient_array(
    relation: SparseRelation,
    coefficients: ArrayLike,
    /,
) -> Array:
    values = jnp.asarray(coefficients)
    route_ndim = len(relation.route_shape)
    if (
        values.ndim < route_ndim
        or tuple(values.shape[-route_ndim:]) != relation.route_shape
    ):
        raise ValueError(
            f"Sparse coefficients must end in route shape {relation.route_shape}; "
            f"got {values.shape}."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    batch_ndim = values.ndim - route_ndim
    valid = relation.valid.reshape((1,) * batch_ndim + relation.valid.shape)
    return jnp.where(valid, values, jnp.zeros((), dtype=values.dtype))


def _flatten_operator_batch(
    values: Any,
    batch_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
    name: str,
    /,
) -> Any:
    count = 1
    for size in batch_shape:
        count *= size

    def flatten_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        prefix = batch_shape + event_shape
        payload_shape = _require_prefix(name, array, prefix)
        return array.reshape((count,) + event_shape + payload_shape)

    return jtu.tree_map(flatten_leaf, values)


def _restore_operator_batch(values: Any, batch_shape: tuple[int, ...], /) -> Any:
    return jtu.tree_map(
        lambda value: jnp.asarray(value).reshape(batch_shape + value.shape[1:]),
        values,
    )


def linear_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
) -> Any:
    """Apply a scalar-coefficient sparse linear map to source payloads."""
    weights = _coefficient_array(relation, coefficients)
    route_ndim = len(relation.route_shape)
    batch_shape = tuple(int(size) for size in weights.shape[:-route_ndim])
    if batch_shape:
        flattened_weights = weights.reshape((-1,) + relation.route_shape)
        flattened_values = _flatten_operator_batch(
            values,
            batch_shape,
            relation.input_shape,
            "Batched sparse source values",
        )
        output = jax.vmap(lambda weight, value: linear_apply(relation, weight, value))(
            flattened_weights,
            flattened_values,
        )
        return _restore_operator_batch(output, batch_shape)
    gathered = gather_routes(relation, values)

    def weight_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_ndim = array.ndim - len(relation.route_shape)
        expanded = weights.reshape(weights.shape + (1,) * payload_ndim)
        return array * expanded

    messages = jtu.tree_map(weight_leaf, gathered)
    return route_reduce(relation, messages)


def _flatten_row_targets(relation: RowRelation, values: Any, /) -> Any:
    def flatten_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_shape = _require_prefix("Row target values", array, relation.output_shape)
        return array.reshape(
            (relation.num_cases * relation.targets_per_case,) + payload_shape
        )

    return jtu.tree_map(flatten_leaf, values)


def _restore_row_sources(relation: RowRelation, values: Any, /) -> Any:
    def restore_leaf(value: Any, /) -> Array:
        array = jnp.asarray(value)
        payload_shape = _require_prefix(
            "Flattened row source values",
            array,
            (relation.num_cases * relation.source_size,),
        )
        return array.reshape(relation.input_shape + payload_shape)

    return jtu.tree_map(restore_leaf, values)


def _linear_reverse_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
    *,
    conjugate: bool,
) -> Any:
    weights = _coefficient_array(relation, coefficients)
    route_ndim = len(relation.route_shape)
    batch_shape = tuple(int(size) for size in weights.shape[:-route_ndim])
    if batch_shape:
        flattened_weights = weights.reshape((-1,) + relation.route_shape)
        flattened_values = _flatten_operator_batch(
            values,
            batch_shape,
            relation.output_shape,
            "Batched sparse target values",
        )
        output = jax.vmap(
            lambda weight, value: _linear_reverse_apply(
                relation,
                weight,
                value,
                conjugate=conjugate,
            )
        )(flattened_weights, flattened_values)
        return _restore_operator_batch(output, batch_shape)
    if conjugate:
        weights = jnp.conj(weights)
    if isinstance(relation, EdgeRelation):
        return linear_apply(relation.transpose(), weights, values)
    if isinstance(relation, RowRelation):
        edge_relation = relation.as_edge_relation()
        flattened_targets = _flatten_row_targets(relation, values)
        flattened_weights = weights.reshape((-1,))
        sources = linear_apply(
            edge_relation.transpose(),
            flattened_weights,
            flattened_targets,
        )
        return _restore_row_sources(relation, sources)
    raise TypeError("relation must be an EdgeRelation or RowRelation.")


def linear_transpose_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
) -> Any:
    """Apply the algebraic transpose of a sparse linear map."""
    return _linear_reverse_apply(
        relation,
        coefficients,
        values,
        conjugate=False,
    )


def linear_adjoint_apply(
    relation: SparseRelation,
    coefficients: ArrayLike,
    values: Any,
    /,
) -> Any:
    """Apply the conjugate adjoint of a sparse linear map."""
    return _linear_reverse_apply(
        relation,
        coefficients,
        values,
        conjugate=True,
    )


__all__ = [
    "RouteReduction",
    "gather_routes",
    "linear_adjoint_apply",
    "linear_apply",
    "linear_transpose_apply",
    "mask_routes",
    "route_reduce",
]
