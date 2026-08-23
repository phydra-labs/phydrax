#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Segmented metric reductions for sampled cochain values."""

from __future__ import annotations

from typing import Literal, TypeAlias

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy


CochainMetricReduction: TypeAlias = Literal[
    "graph_mean",
    "metric_mean",
    "metric_sum",
]


def cochain_metric_reduce(
    values: ArrayLike,
    hodge_star: ArrayLike,
    graph_index: ArrayLike,
    /,
    *,
    n_graph: int,
    reduction: CochainMetricReduction = "graph_mean",
    segment_weight: ArrayLike | None = None,
    entity_mask: ArrayLike | None = None,
    precision: GeometryPrecisionPolicy | None = None,
) -> Array:
    """Reduce per-cell scalar values without allowing mesh size to bias cases.

    Each non-empty graph segment contributes equally. ``graph_mean`` uses an
    arithmetic cell mean, ``metric_mean`` normalizes by the segment's Hodge-star
    mass, and ``metric_sum`` retains that physical mass. ``entity_mask`` excludes
    padding or other cells without changing static shapes. ``segment_weight`` may
    supply one constant measure factor per segment, repeated over its cells.
    """
    if reduction not in ("graph_mean", "metric_mean", "metric_sum"):
        raise ValueError(
            "reduction must be 'graph_mean', 'metric_mean', or 'metric_sum'."
        )
    count = int(n_graph)
    if count <= 0:
        raise ValueError("n_graph must be positive.")
    precision_ = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")

    value_array = jnp.asarray(values)
    precision_.validate_coordinates(value_array)
    value_array = precision_.compute(value_array)
    metric = precision_.compute(jnp.asarray(hodge_star, dtype=value_array.real.dtype))
    graph_id = jnp.asarray(graph_index, dtype=jnp.int32)
    if value_array.ndim != 1:
        raise ValueError(f"values must be rank-1, got shape {value_array.shape!r}.")
    if metric.shape != value_array.shape:
        raise ValueError(
            f"hodge_star shape {metric.shape!r} must match values {value_array.shape!r}."
        )
    if graph_id.shape != value_array.shape:
        raise ValueError(
            f"graph_index shape {graph_id.shape!r} must match values {value_array.shape!r}."
        )
    if entity_mask is None:
        active = graph_id >= 0
    else:
        active = jnp.asarray(entity_mask, dtype=bool)
        if active.shape != value_array.shape:
            raise ValueError(
                f"entity_mask shape {active.shape!r} must match values "
                f"{value_array.shape!r}."
            )
        active = active & (graph_id >= 0)
    safe_graph_id = jnp.where(active, graph_id, 0)
    active_weight = precision_.accumulation(active.astype(metric.dtype))

    if segment_weight is None:
        time_weight = precision_.accumulation(jnp.ones_like(metric))
    else:
        time_weight = precision_.accumulation(
            jnp.asarray(segment_weight, dtype=metric.dtype)
        )
        if time_weight.shape != value_array.shape:
            raise ValueError(
                "segment_weight shape must match values; "
                f"got {time_weight.shape!r} and {value_array.shape!r}."
            )

    reduced_values = precision_.accumulation(value_array)
    reduced_metric = precision_.accumulation(metric)
    cells = jnp.bincount(safe_graph_id, weights=active_weight, length=count)
    unweighted_sum = jnp.bincount(
        safe_graph_id,
        weights=reduced_values * active_weight,
        length=count,
    )
    metric_mass = jnp.bincount(
        safe_graph_id,
        weights=reduced_metric * active_weight,
        length=count,
    )
    metric_sum = jnp.bincount(
        safe_graph_id,
        weights=reduced_metric * reduced_values * active_weight,
        length=count,
    )
    time_sum = jnp.bincount(
        safe_graph_id,
        weights=time_weight * active_weight,
        length=count,
    )
    valid = cells > 0

    if reduction == "graph_mean":
        per_graph = unweighted_sum / jnp.maximum(cells, 1)
    elif reduction == "metric_mean":
        per_graph = metric_sum / jnp.maximum(metric_mass, jnp.finfo(metric.dtype).tiny)
    else:
        per_graph = metric_sum

    per_graph_weight = time_sum / jnp.maximum(cells, 1)
    weighted = jnp.where(valid, per_graph * per_graph_weight, 0)
    return precision_.decision(
        precision_.sum(weighted) / jnp.maximum(precision_.sum(valid), 1)
    )


__all__ = ["CochainMetricReduction", "cochain_metric_reduce"]
