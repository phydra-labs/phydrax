#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..integration import discrete
from ..transport import (
    AbstractBalancedTransportPlan,
    AbstractBalancedTransportSolver,
    AbstractGroundCost,
    discrete_problem,
    require_converged,
    Sinkhorn,
    SquaredEuclideanCost,
)


class OptimalTransportEnsembleTransformResult(StrictModule):
    """Equal-weight barycentric ensemble transform and coupling diagnostics."""

    particles: Array
    source_weights: Array
    source_mean: Array
    transformed_mean: Array
    mean_error: Array
    transport: AbstractBalancedTransportPlan
    particle_axis: int = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)


def optimal_transport_ensemble_transform(
    particles: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    particle_axis: int = 0,
    cost: AbstractGroundCost | None = None,
    solver: AbstractBalancedTransportSolver | None = None,
    epsilon: float = 0.5,
) -> OptimalTransportEnsembleTransformResult:
    """Deterministically transform weighted particles into an equal-weight ensemble.

    Dimensions before ``particle_axis`` are independent physical cases; dimensions
    after it form one particle event. The returned particles preserve the input axis
    order and expose the raw native coupling for sensitivity or genealogy analysis.
    """
    values = jnp.asarray(particles, dtype=float)
    if values.ndim < 1:
        raise ValueError("particles must have at least one dimension.")
    position = particle_axis + values.ndim if particle_axis < 0 else particle_axis
    if position < 0 or position >= values.ndim:
        raise ValueError("particle_axis is out of range.")
    count = values.shape[position]
    if count < 1:
        raise ValueError("The particle axis must be nonempty.")
    leading_shape = values.shape[:position]
    event_shape = values.shape[position + 1 :]
    expected_weights = leading_shape + (count,)
    probabilities = jnp.asarray(weights, dtype=values.dtype)
    if probabilities.shape != expected_weights:
        raise ValueError(
            f"weights must have shape {expected_weights}; got {probabilities.shape}."
        )
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        "particles must contain only finite values.",
    )
    probabilities = _normalize_weights(probabilities)
    configured_cost = SquaredEuclideanCost() if cost is None else cost
    if not isinstance(configured_cost, AbstractGroundCost):
        raise TypeError("cost must be an AbstractGroundCost or None.")
    configured_solver = _solver(epsilon, solver)

    flattened = values.reshape((-1, count) + event_shape)
    point_rows = flattened.reshape((flattened.shape[0], count, -1))
    weight_rows = probabilities.reshape((-1, count))
    uniform = jnp.full((count,), 1.0 / float(count), dtype=values.dtype)

    def transform_case(points, source_weights):
        source = discrete(
            points,
            cx.Field(source_weights, dims=("particle",)),
            axes="particle",
            normalized=True,
            provenance="optimal-transport-ensemble-source",
        )
        target = discrete(
            points,
            cx.Field(uniform, dims=("particle",)),
            axes="particle",
            normalized=True,
            provenance="optimal-transport-ensemble-target",
        )
        problem = discrete_problem(source, target, cost=configured_cost)
        result = require_converged(configured_solver(problem))
        transformed = result.barycentric_source_to_target(points)
        source_mean = jnp.sum(source_weights[:, None] * points, axis=0)
        transformed_mean = jnp.mean(transformed, axis=0)
        return transformed, source_mean, transformed_mean, result

    transformed, source_mean, transformed_mean, transport = jax.vmap(transform_case)(
        point_rows,
        weight_rows,
    )
    transport = jax.tree.map(
        lambda leaf: (
            leaf.reshape(leading_shape + leaf.shape[1:])
            if eqx.is_array(leaf)
            else leaf
        ),
        transport,
    )
    transformed = transformed.reshape(leading_shape + (count,) + event_shape)
    source_mean = source_mean.reshape(leading_shape + event_shape)
    transformed_mean = transformed_mean.reshape(leading_shape + event_shape)
    return OptimalTransportEnsembleTransformResult(
        particles=transformed,
        source_weights=probabilities,
        source_mean=source_mean,
        transformed_mean=transformed_mean,
        mean_error=transformed_mean - source_mean,
        transport=transport,
        particle_axis=position,
        event_shape=event_shape,
    )


def _normalize_weights(weights: Array, /) -> Array:
    total = jnp.sum(weights, axis=-1, keepdims=True)
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights))
        | jnp.any(weights < 0.0)
        | jnp.any(total <= 0.0),
        "weights must be finite, nonnegative, and have positive casewise mass.",
    )
    return weights / total


def _solver(
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
        max_iterations=1000,
        min_iterations=1,
        tolerance=1e-7,
        check_every=5,
        early_stop=False,
    )


__all__ = [
    "OptimalTransportEnsembleTransformResult",
    "optimal_transport_ensemble_transform",
]
