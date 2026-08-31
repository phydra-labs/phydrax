#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..topology import FieldTopologySnapshot


class TopologyEnsembleSummary(StrictModule, NonTrainableState):
    """Weighted posterior summaries of exact per-realization topology snapshots."""

    mean_betti: Array
    variance_betti: Array
    event_probability: Array
    weights: Array
    sample_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    summary_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshots: Sequence[FieldTopologySnapshot],
        /,
        *,
        weights: ArrayLike | None = None,
    ):
        values = tuple(snapshots)
        if not values:
            raise ValueError("Topology ensemble summaries require snapshots.")
        topology_ids = {value.filtration.complex.topology.topology_id for value in values}
        threshold_shapes = {value.betti_counts.shape for value in values}
        if len(topology_ids) != 1 or len(threshold_shapes) != 1:
            raise ValueError("Topology ensemble snapshots must share topology and axes.")
        weight_values = (
            jnp.ones((len(values),), dtype=float)
            if weights is None
            else jnp.asarray(weights, dtype=float)
        )
        if weight_values.shape != (len(values),):
            raise ValueError("Topology ensemble weights do not match sample count.")
        if not bool(jnp.all(jnp.isfinite(weight_values))) or bool(
            jnp.any(weight_values < 0)
        ):
            raise ValueError("Topology ensemble weights must be finite and non-negative.")
        total = jnp.sum(weight_values)
        if not bool(total > 0):
            raise ValueError("Topology ensemble weights require positive total mass.")
        normalized = weight_values / total
        counts = jnp.stack(tuple(value.betti_counts for value in values))
        mean = jnp.sum(normalized[:, None, None] * counts, axis=0)
        variance = jnp.sum(
            normalized[:, None, None] * (counts - mean) ** 2,
            axis=0,
        )
        baseline = counts[0]
        event = jnp.any(counts != baseline[None, ...], axis=(-2, -1))
        probability = jnp.sum(normalized * event.astype(normalized.dtype))
        topology_id = next(iter(topology_ids))
        self.mean_betti = mean
        self.variance_betti = variance
        self.event_probability = probability
        self.weights = normalized
        self.sample_count = len(values)
        self.topology_id = topology_id
        self.summary_id = canonical_fingerprint(
            {
                "kind": "topology-ensemble-summary",
                "topology": topology_id,
                "snapshots": [value.snapshot_id for value in values],
                "weights": array_tree_fingerprint(normalized),
            }
        )


__all__ = ["TopologyEnsembleSummary"]
