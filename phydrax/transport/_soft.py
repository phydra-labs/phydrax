#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._costs import SquaredEuclideanCost
from ._measure import _FiniteTransportMeasure
from ._problem import DiscreteTransportProblem
from ._results import (
    AbstractBalancedTransportPlan,
    AbstractBalancedTransportSolver,
    require_converged,
)
from ._sinkhorn import Sinkhorn
from ._univariate import _probabilities


Value = ArrayLike | cx.Field


def soft_order_transport(
    values: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    num_targets: int | None = None,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> AbstractBalancedTransportPlan:
    """Fit one soft monotone coupling from values to ordered mass bins."""
    source_values = _vector(values, name="values")
    count = source_values.shape[0]
    targets = count if num_targets is None else int(num_targets)
    if targets < 1:
        raise ValueError("num_targets must be positive.")
    source_probabilities = _probabilities(
        weights,
        count,
        name="weights",
        dtype=source_values.dtype,
    )
    target_probabilities = _probabilities(
        target_weights,
        targets,
        name="target_weights",
        dtype=source_values.dtype,
    )
    source_locations = _squash(source_values, source_probabilities)
    target_locations = (
        jnp.cumsum(target_probabilities) - 0.5 * target_probabilities
    )
    source_measure = _FiniteTransportMeasure(
        source_locations[:, None],
        source_probabilities,
        jnp.asarray(1.0, dtype=source_values.dtype),
        source_probabilities > 0.0,
        event_shape=(),
        normalized=True,
        provenance="soft-order-source",
    )
    target_measure = _FiniteTransportMeasure(
        target_locations[:, None],
        target_probabilities,
        jnp.asarray(1.0, dtype=source_values.dtype),
        target_probabilities > 0.0,
        event_shape=(),
        normalized=True,
        provenance="soft-order-target",
    )
    problem = DiscreteTransportProblem(
        source_measure,
        target_measure,
        SquaredEuclideanCost(),
    )
    configured = _soft_solver(epsilon, solver)
    return require_converged(configured(problem))


def soft_sort(
    values: Value,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> Array | cx.Field:
    """Differentiably sort values along one array axis or named field dimension."""
    data, position, dims = _data_axis(values, axis=axis, name="values")
    weight_data = _weight_data(weights, data, position, dims=dims)
    configured = _soft_solver(epsilon, solver)

    def one(vector, vector_weights):
        result = soft_order_transport(
            vector,
            weights=vector_weights,
            solver=configured,
        )
        return result.barycentric_source_to_target(vector)

    output = _map_same_axis(data, weight_data, position, one)
    return _restore(output, dims)


def soft_rank(
    values: Value,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> Array | cx.Field:
    """Return zero-based differentiable ranks along one axis."""
    data, position, dims = _data_axis(values, axis=axis, name="values")
    weight_data = _weight_data(weights, data, position, dims=dims)
    configured = _soft_solver(epsilon, solver)
    count = data.shape[position]
    target_ranks = jnp.arange(count, dtype=data.dtype)

    def one(vector, vector_weights):
        result = soft_order_transport(
            vector,
            weights=vector_weights,
            solver=configured,
        )
        return result.barycentric_target_to_source(target_ranks)

    output = _map_same_axis(data, weight_data, position, one)
    return _restore(output, dims)


def soft_sort_by(
    criterion: Value,
    payload: Value,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> Array | cx.Field:
    """Differentiably reorder a same-shaped payload by a scalar criterion."""
    criterion_data, position, dims = _data_axis(
        criterion, axis=axis, name="criterion"
    )
    payload_data, payload_position, payload_dims = _data_axis(
        payload, axis=axis, name="payload"
    )
    if payload_data.shape != criterion_data.shape or payload_position != position:
        raise ValueError("criterion and payload must have identical shape and axis.")
    if payload_dims != dims:
        raise ValueError("Named criterion and payload fields must have equal dims.")
    weight_data = _weight_data(weights, criterion_data, position, dims=dims)
    configured = _soft_solver(epsilon, solver)
    moved_criterion = jnp.moveaxis(criterion_data, position, -1)
    moved_payload = jnp.moveaxis(payload_data, position, -1)
    moved_weights = jnp.moveaxis(weight_data, position, -1)
    leading_shape = moved_criterion.shape[:-1]
    count = moved_criterion.shape[-1]
    criteria = moved_criterion.reshape((-1, count))
    payloads = moved_payload.reshape((-1, count))
    weight_rows = moved_weights.reshape((-1, count))

    def one(vector, values_, vector_weights):
        result = soft_order_transport(
            vector,
            weights=vector_weights,
            solver=configured,
        )
        return result.barycentric_source_to_target(values_)

    output = jax.vmap(one)(criteria, payloads, weight_rows)
    output = output.reshape(leading_shape + (count,))
    output = jnp.moveaxis(output, -1, position)
    return _restore(output, dims)


def soft_topk_mask(
    values: Value,
    k: int,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> Array | cx.Field:
    """Return differentiable membership in the largest ``k`` ordered bins."""
    data, position, dims = _data_axis(values, axis=axis, name="values")
    count = data.shape[position]
    selected = int(k)
    if selected < 0 or selected > count:
        raise ValueError("k must lie in [0, axis_size].")
    if selected == 0:
        return _restore(jnp.zeros_like(data), dims)
    if selected == count:
        return _restore(jnp.ones_like(data), dims)
    weight_data = _weight_data(weights, data, position, dims=dims)
    configured = _soft_solver(epsilon, solver)
    target_mask = jnp.concatenate(
        [
            jnp.zeros((count - selected,), dtype=data.dtype),
            jnp.ones((selected,), dtype=data.dtype),
        ]
    )

    def one(vector, vector_weights):
        result = soft_order_transport(
            vector,
            weights=vector_weights,
            solver=configured,
        )
        return result.barycentric_target_to_source(target_mask)

    output = _map_same_axis(data, weight_data, position, one)
    return _restore(output, dims)


def soft_topk_values(
    values: Value,
    k: int,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> Array | cx.Field:
    """Return differentiable ascending values in the largest ``k`` bins."""
    data, position, dims = _data_axis(values, axis=axis, name="values")
    count = data.shape[position]
    selected = int(k)
    if selected < 0 or selected > count:
        raise ValueError("k must lie in [0, axis_size].")
    sorted_values = soft_sort(
        values,
        weights=weights,
        axis=axis,
        epsilon=epsilon,
        solver=solver,
    )
    sorted_data = sorted_values.data if isinstance(sorted_values, cx.Field) else sorted_values
    indices = jnp.arange(count - selected, count, dtype=jnp.int32)
    output = jnp.take(sorted_data, indices, axis=position)
    return _restore(output, dims)


def soft_quantile(
    values: Value,
    q: ArrayLike,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
    quantile_dim: str = "quantile",
) -> Array | cx.Field:
    """Return differentiable weighted quantiles in caller-specified order."""
    data, position, dims = _data_axis(values, axis=axis, name="values")
    quantiles = jnp.asarray(q, dtype=data.dtype)
    quantiles = eqx.error_if(
        quantiles,
        jnp.any(~jnp.isfinite(quantiles))
        | jnp.any(quantiles < 0.0)
        | jnp.any(quantiles > 1.0),
        "q must contain only finite values in [0, 1].",
    )
    sorted_values = soft_sort(
        values,
        weights=weights,
        axis=axis,
        epsilon=epsilon,
        solver=solver,
    )
    sorted_data = sorted_values.data if isinstance(sorted_values, cx.Field) else sorted_values
    moved = jnp.moveaxis(sorted_data, position, -1)
    moved_original = jnp.moveaxis(data, position, -1)
    weight_data = _weight_data(weights, data, position, dims=dims)
    moved_weights = jnp.moveaxis(weight_data, position, -1)
    leading_shape = moved.shape[:-1]
    count = moved.shape[-1]
    rows = moved.reshape((-1, count))
    original_rows = moved_original.reshape((-1, count))
    weight_rows = moved_weights.reshape((-1, count))
    positions = quantiles.reshape((-1,)) * float(max(count - 1, 0))
    grid = jnp.arange(count, dtype=data.dtype)

    def interpolate(row, original, vector_weights):
        values_ = jnp.interp(positions, grid, row)
        lower = jnp.min(jnp.where(vector_weights > 0.0, original, jnp.inf))
        upper = jnp.max(jnp.where(vector_weights > 0.0, original, -jnp.inf))
        values_ = jnp.where(quantiles.reshape((-1,)) == 0.0, lower, values_)
        values_ = jnp.where(quantiles.reshape((-1,)) == 1.0, upper, values_)
        return values_

    output = jax.vmap(interpolate)(rows, original_rows, weight_rows)
    output_shape = leading_shape + quantiles.shape
    output = output.reshape(output_shape)
    if dims is None:
        return output
    retained_dims = tuple(dim for index, dim in enumerate(dims) if index != position)
    if quantiles.ndim == 0:
        output_dims = retained_dims
    elif quantiles.ndim == 1:
        if not quantile_dim:
            raise ValueError("quantile_dim must be nonempty for vector quantiles.")
        if quantile_dim in retained_dims:
            raise ValueError("quantile_dim collides with an existing field dimension.")
        output_dims = retained_dims + (quantile_dim,)
    else:
        raise ValueError("Named-field quantiles must be scalar or rank one.")
    return cx.Field(output, dims=output_dims)


def soft_quantile_normalize(
    values: Value,
    reference: ArrayLike,
    /,
    *,
    weights: Value | None = None,
    reference_weights: ArrayLike | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> Array | cx.Field:
    """Differentiably map values to the ordered empirical reference law."""
    data, position, dims = _data_axis(values, axis=axis, name="values")
    reference_values = _vector(reference, name="reference")
    count = data.shape[position]
    configured = _soft_solver(epsilon, solver)
    reference_result = soft_order_transport(
        reference_values,
        weights=reference_weights,
        num_targets=count,
        solver=configured,
    )
    reference_bins = reference_result.barycentric_source_to_target(reference_values)
    weight_data = _weight_data(weights, data, position, dims=dims)

    def one(vector, vector_weights):
        result = soft_order_transport(
            vector,
            weights=vector_weights,
            solver=configured,
        )
        return result.barycentric_target_to_source(reference_bins)

    output = _map_same_axis(data, weight_data, position, one)
    return _restore(output, dims)


def soft_quantize(
    values: Value,
    num_levels: int,
    /,
    *,
    weights: Value | None = None,
    axis: int | str = -1,
    epsilon: float = 0.1,
    solver: AbstractBalancedTransportSolver | None = None,
) -> Array | cx.Field:
    """Differentiably quantize values through learned ordered barycentric levels."""
    data, position, dims = _data_axis(values, axis=axis, name="values")
    levels = int(num_levels)
    if levels < 1:
        raise ValueError("num_levels must be positive.")
    weight_data = _weight_data(weights, data, position, dims=dims)
    configured = _soft_solver(epsilon, solver)

    def one(vector, vector_weights):
        result = soft_order_transport(
            vector,
            weights=vector_weights,
            num_targets=levels,
            solver=configured,
        )
        centers = result.barycentric_source_to_target(vector)
        return result.barycentric_target_to_source(centers)

    output = _map_same_axis(data, weight_data, position, one)
    return _restore(output, dims)


def _soft_solver(
    epsilon: float, solver: AbstractBalancedTransportSolver | None, /
) -> AbstractBalancedTransportSolver:
    if solver is not None:
        if not isinstance(solver, AbstractBalancedTransportSolver):
            raise TypeError(
                "solver must implement the balanced transport solver contract or be None."
            )
        return solver
    value = float(epsilon)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    return Sinkhorn(
        value,
        max_iterations=300,
        min_iterations=1,
        tolerance=1e-7,
        check_every=5,
        early_stop=False,
    )


def _squash(values: Array, probabilities: Array, /) -> Array:
    center = jnp.sum(probabilities * values)
    variance = jnp.sum(probabilities * (values - center) ** 2)
    scale = jnp.sqrt(variance + jnp.finfo(values.dtype).eps)
    return jax.nn.sigmoid((values - center) / scale)


def _data_axis(
    value: Value,
    /,
    *,
    axis: int | str,
    name: str,
) -> tuple[Array, int, tuple[Any, ...] | None]:
    if isinstance(value, cx.Field):
        if not isinstance(axis, str):
            raise TypeError(f"Named {name} fields require a named axis.")
        if axis not in value.named_dims:
            raise ValueError(f"{name} is missing named axis {axis!r}.")
        data = jnp.asarray(value.data, dtype=float)
        position = value.dims.index(axis)
        return data, position, value.dims
    if not isinstance(axis, int):
        raise TypeError(f"Raw {name} arrays require an integer axis.")
    data = jnp.asarray(value, dtype=float)
    if data.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension.")
    position = axis + data.ndim if axis < 0 else axis
    if position < 0 or position >= data.ndim:
        raise ValueError(f"{name} axis is out of range.")
    if data.shape[position] < 1:
        raise ValueError(f"{name} ordering axis must be nonempty.")
    data = eqx.error_if(
        data,
        jnp.any(~jnp.isfinite(data)),
        f"{name} must contain only finite values.",
    )
    return data, position, None


def _weight_data(
    weights: Value | None,
    data: Array,
    position: int,
    /,
    *,
    dims: tuple[Any, ...] | None,
) -> Array:
    if weights is None:
        shape = [1] * data.ndim
        shape[position] = data.shape[position]
        base = jnp.full(
            (data.shape[position],),
            1.0 / float(data.shape[position]),
            dtype=data.dtype,
        ).reshape(tuple(shape))
        return jnp.broadcast_to(base, data.shape)
    if isinstance(weights, cx.Field):
        if dims is None:
            raw = jnp.asarray(weights.data, dtype=data.dtype)
        else:
            template = cx.Field(jnp.ones(data.shape, dtype=data.dtype), dims=dims)
            raw = jnp.asarray((weights * template).data, dtype=data.dtype)
    else:
        raw = jnp.asarray(weights, dtype=data.dtype)
        if raw.ndim == 1 and raw.shape[0] == data.shape[position]:
            shape = [1] * data.ndim
            shape[position] = raw.shape[0]
            raw = raw.reshape(tuple(shape))
    return jnp.broadcast_to(raw, data.shape)


def _map_same_axis(data, weights, position, operation):
    moved = jnp.moveaxis(data, position, -1)
    moved_weights = jnp.moveaxis(weights, position, -1)
    leading_shape = moved.shape[:-1]
    count = moved.shape[-1]
    rows = moved.reshape((-1, count))
    weight_rows = moved_weights.reshape((-1, count))
    output = jax.vmap(operation)(rows, weight_rows)
    output_count = output.shape[-1]
    output = output.reshape(leading_shape + (output_count,))
    return jnp.moveaxis(output, -1, position)


def _restore(data: Array, dims: tuple[Any, ...] | None, /) -> Array | cx.Field:
    return data if dims is None else cx.Field(data, dims=dims)


def _vector(values: ArrayLike, /, *, name: str) -> Array:
    result = jnp.asarray(values, dtype=float)
    if result.ndim != 1 or result.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty rank-one array.")
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite values.",
    )


__all__ = [
    "soft_order_transport",
    "soft_quantile",
    "soft_quantile_normalize",
    "soft_quantize",
    "soft_rank",
    "soft_sort",
    "soft_sort_by",
    "soft_topk_mask",
    "soft_topk_values",
]
