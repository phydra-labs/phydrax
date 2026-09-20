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
from ...particle_physics import EventWeightSet


class HistogramPlan(StrictModule, NonTrainableState):
    edges: Array
    observable_id: str = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    bin_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, edges: ArrayLike, /, *, observable_id: str, unit_id: str):
        edges_ = np.asarray(edges, dtype=np.float64)
        observable = str(observable_id).strip()
        unit = str(unit_id).strip()
        if (
            edges_.ndim != 1
            or edges_.size < 2
            or np.any(~np.isfinite(edges_))
            or np.any(np.diff(edges_) <= 0.0)
        ):
            raise ValueError("Histogram edges must be finite and strictly increasing.")
        if not observable or not unit:
            raise ValueError("Histogram observable and unit identities are required.")
        self.edges = jnp.asarray(edges_)
        self.observable_id = observable
        self.unit_id = unit
        self.bin_count = edges_.size - 1
        self.plan_id = canonical_fingerprint(
            {
                "kind": "collider-histogram-plan",
                "edges": array_tree_fingerprint(edges_),
                "observable": observable,
                "unit": unit,
            }
        )


class WeightedHistogram(StrictModule, NonTrainableState):
    sum_weights: Array
    sum_squared_weights: Array
    entry_count: Array
    underflow_sum_weights: Array
    overflow_sum_weights: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


def fill_weighted_histogram(
    plan: HistogramPlan,
    values: ArrayLike,
    weights: ArrayLike,
    /,
    *,
    active: ArrayLike | None = None,
) -> WeightedHistogram:
    if not isinstance(plan, HistogramPlan):
        raise TypeError("plan must be HistogramPlan.")
    values_ = jnp.asarray(values)
    weights_ = jnp.asarray(weights, dtype=values_.dtype)
    if values_.ndim != 1 or weights_.shape != values_.shape:
        raise ValueError("Histogram values and weights must be aligned vectors.")
    active_ = (
        jnp.ones(values_.shape, dtype=jnp.bool_)
        if active is None
        else jnp.asarray(active, dtype=jnp.bool_)
    )
    if active_.shape != values_.shape:
        raise ValueError("active must align with histogram values.")
    finite_entries = jnp.isfinite(values_) & jnp.isfinite(weights_)
    admitted = active_ & finite_entries
    indices = jnp.searchsorted(plan.edges, values_, side="right") - 1
    in_range = admitted & (indices >= 0) & (indices < plan.bin_count)
    safe_indices = jnp.clip(indices, 0, plan.bin_count - 1)
    membership = jax.nn.one_hot(safe_indices, plan.bin_count, dtype=weights_.dtype)
    weighted_membership = membership * in_range[:, None]
    return WeightedHistogram(
        jnp.sum(weighted_membership * weights_[:, None], axis=0),
        jnp.sum(weighted_membership * (weights_ * weights_)[:, None], axis=0),
        jnp.sum(weighted_membership.astype(jnp.int32), axis=0),
        jnp.sum(jnp.where(admitted & (values_ < plan.edges[0]), weights_, 0.0)),
        jnp.sum(jnp.where(admitted & (values_ >= plan.edges[-1]), weights_, 0.0)),
        jnp.all(jnp.where(active_, finite_entries, True)),
        plan.plan_id,
    )


class CutflowResult(StrictModule, NonTrainableState):
    sum_weights: Array
    sum_squared_weights: Array
    event_counts: Array
    nested: Array
    finite: Array
    cut_names: tuple[str, ...] = eqx.field(static=True)
    weight_set_id: str = eqx.field(static=True)


def build_cutflow(
    weights: EventWeightSet,
    cumulative_masks: ArrayLike,
    /,
    *,
    cut_names: Sequence[str],
) -> CutflowResult:
    """Accumulate a precomputed cumulative cut matrix and verify nesting."""
    if not isinstance(weights, EventWeightSet):
        raise TypeError("weights must be EventWeightSet.")
    masks = jnp.asarray(cumulative_masks, dtype=jnp.bool_)
    names = tuple(str(value).strip() for value in cut_names)
    if (
        masks.shape != (weights.event_capacity, len(names))
        or not names
        or any(not value for value in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Cut masks and distinct names must align.")
    masks = masks & weights.event_active[:, None]
    nested = (
        jnp.all(~masks[:, 1:] | masks[:, :-1]) if len(names) > 1 else jnp.asarray(True)
    )
    nominal = weights.nominal[:, None]
    active = masks & weights.finite[:, None]
    return CutflowResult(
        jnp.sum(jnp.where(active, nominal, 0.0), axis=0),
        jnp.sum(jnp.where(active, nominal * nominal, 0.0), axis=0),
        jnp.sum(active, axis=0, dtype=jnp.int32),
        nested,
        jnp.all(jnp.where(weights.event_active, weights.finite, True)),
        names,
        weights.weight_set_id,
    )


class SystematicHistogramSet(StrictModule, NonTrainableState):
    sum_weights: Array
    sum_squared_weights: Array
    finite: Array
    weight_names: tuple[str, ...] = eqx.field(static=True)
    variation_kinds: tuple[str, ...] = eqx.field(static=True)
    correlation_groups: tuple[str, ...] = eqx.field(static=True)
    histogram_plan_id: str = eqx.field(static=True)


def histogram_weight_variations(
    plan: HistogramPlan,
    values: ArrayLike,
    weights: EventWeightSet,
    /,
    *,
    selected: ArrayLike | None = None,
) -> SystematicHistogramSet:
    values_ = jnp.asarray(values)
    if values_.shape != (weights.event_capacity,):
        raise ValueError("values must align with event weights.")
    active = (
        weights.event_active
        if selected is None
        else weights.event_active & jnp.asarray(selected, dtype=jnp.bool_)
    )
    histograms = tuple(
        fill_weighted_histogram(plan, values_, weights.values[:, index], active=active)
        for index in range(weights.weight_count)
    )
    return SystematicHistogramSet(
        jnp.stack(tuple(value.sum_weights for value in histograms)),
        jnp.stack(tuple(value.sum_squared_weights for value in histograms)),
        jnp.all(jnp.stack(tuple(value.finite for value in histograms))),
        weights.names,
        tuple(value.value for value in weights.variation_kinds),
        weights.correlation_groups,
        plan.plan_id,
    )


__all__ = [
    "CutflowResult",
    "HistogramPlan",
    "SystematicHistogramSet",
    "WeightedHistogram",
    "build_cutflow",
    "fill_weighted_histogram",
    "histogram_weight_variations",
]
