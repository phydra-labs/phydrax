#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MultiHistogramPlan(StrictModule, NonTrainableState):
    edges: tuple[Array, ...]
    axis_names: tuple[str, ...] = eqx.field(static=True)
    unit_ids: tuple[str, ...] = eqx.field(static=True)
    bin_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        edges: Sequence[ArrayLike],
        /,
        *,
        axis_names: Sequence[str],
        unit_ids: Sequence[str],
    ):
        edges_ = tuple(np.asarray(value, dtype=np.float64) for value in edges)
        names = tuple(str(value).strip() for value in axis_names)
        units = tuple(str(value).strip() for value in unit_ids)
        if (
            not edges_
            or len(edges_) != len(names)
            or len(names) != len(units)
            or any(not value for value in names + units)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Histogram axes require aligned unique names and units.")
        if any(
            value.ndim != 1
            or value.size < 2
            or np.any(~np.isfinite(value))
            or np.any(np.diff(value) <= 0.0)
            for value in edges_
        ):
            raise ValueError("Every histogram edge vector must be finite and increasing.")
        self.edges = tuple(jnp.asarray(value) for value in edges_)
        self.axis_names = names
        self.unit_ids = units
        self.bin_shape = tuple(value.size - 1 for value in edges_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multidimensional-histogram-plan",
                "edges": [array_tree_fingerprint(value) for value in edges_],
                "names": list(names),
                "units": list(units),
            }
        )


class MultiHistogram(StrictModule, NonTrainableState):
    sum_weights: Array
    sum_squared_weights: Array
    entry_count: Array
    flow_sum_weights: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


def fill_multidimensional_histogram(
    plan: MultiHistogramPlan,
    coordinates: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    active: ArrayLike | None = None,
) -> MultiHistogram:
    if not isinstance(plan, MultiHistogramPlan):
        raise TypeError("plan must be MultiHistogramPlan.")
    coordinates_ = jnp.asarray(coordinates)
    weights_ = jnp.asarray(weights, dtype=coordinates_.dtype)
    if (
        coordinates_.ndim != 2
        or coordinates_.shape[1] != len(plan.edges)
        or weights_.shape != (coordinates_.shape[0],)
    ):
        raise ValueError("Coordinates and weights must align with histogram dimensions.")
    active_ = (
        jnp.ones(weights_.shape, dtype=jnp.bool_)
        if active is None
        else jnp.asarray(active, dtype=jnp.bool_)
    )
    if active_.shape != weights_.shape:
        raise ValueError("active must align with histogram entries.")
    bin_indices = []
    in_range = (
        active_ & jnp.all(jnp.isfinite(coordinates_), axis=1) & jnp.isfinite(weights_)
    )
    for axis, edges in enumerate(plan.edges):
        index = jnp.searchsorted(edges, coordinates_[:, axis], side="right") - 1
        in_range &= (index >= 0) & (index < plan.bin_shape[axis])
        bin_indices.append(jnp.clip(index, 0, plan.bin_shape[axis] - 1))
    strides = tuple(
        int(np.prod(plan.bin_shape[axis + 1 :], dtype=np.int64))
        for axis in range(len(plan.bin_shape))
    )
    flat_index = sum(
        index * stride for index, stride in zip(bin_indices, strides, strict=True)
    )
    flat_count = int(np.prod(plan.bin_shape, dtype=np.int64))
    membership = (
        jax.nn.one_hot(flat_index, flat_count, dtype=weights_.dtype) * in_range[:, None]
    )
    sum_weights = jnp.sum(membership * weights_[:, None], axis=0).reshape(plan.bin_shape)
    sum_squared = jnp.sum(membership * (weights_ * weights_)[:, None], axis=0).reshape(
        plan.bin_shape
    )
    counts = jnp.sum(membership.astype(jnp.int32), axis=0).reshape(plan.bin_shape)
    flow = jnp.sum(jnp.where(active_ & ~in_range, weights_, 0.0))
    finite = jnp.all(
        jnp.where(
            active_,
            jnp.isfinite(weights_) & jnp.all(jnp.isfinite(coordinates_), axis=1),
            True,
        )
    )
    return MultiHistogram(sum_weights, sum_squared, counts, flow, finite, plan.plan_id)


__all__ = ["MultiHistogram", "MultiHistogramPlan", "fill_multidimensional_histogram"]
