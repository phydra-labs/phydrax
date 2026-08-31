#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._complex import CellSubcomplex, CellVertexSupport
from ._filtration import (
    CellFiltration,
    lower_star_filtration,
    PreparedVertexFiltration,
    upper_star_filtration,
)
from ._persistence import compute_persistence, PersistenceResult
from ._resources import TopologyResourcePolicy


class FieldTopologySnapshot(StrictModule, NonTrainableState):
    """Exact finite-complex topology of one explicitly scalarized field snapshot."""

    filtration: CellFiltration
    persistence: PersistenceResult
    thresholds: Array
    betti_counts: Array
    field_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        filtration: CellFiltration,
        persistence: PersistenceResult,
        thresholds: ArrayLike,
        betti_counts: ArrayLike,
        /,
        *,
        field_id: str,
    ):
        threshold_values = jnp.asarray(thresholds)
        counts = jnp.asarray(betti_counts, dtype=jnp.int32)
        if threshold_values.ndim != 1 or counts.ndim != 2:
            raise ValueError(
                "Field topology thresholds and Betti counts have invalid ranks."
            )
        if counts.shape[0] != threshold_values.shape[0]:
            raise ValueError("Field topology thresholds and counts do not align.")
        identifier = str(field_id)
        if not identifier:
            raise ValueError("Field topology field_id must be non-empty.")
        self.filtration = filtration
        self.persistence = persistence
        self.thresholds = threshold_values
        self.betti_counts = counts
        self.field_id = identifier
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "field-topology-snapshot",
                "field": identifier,
                "filtration": filtration.filtration_id,
                "persistence": persistence.result_id,
                "thresholds": array_tree_fingerprint(threshold_values),
                "betti": array_tree_fingerprint(counts),
            }
        )


class FieldTopologyPlan(StrictModule, NonTrainableState):
    """Prepared scalar-field filtration and exact persistence policy."""

    complex: CellSubcomplex
    support: CellVertexSupport
    coefficients: PrimeField
    resources: TopologyResourcePolicy
    thresholds: Array
    direction: Literal["sublevel", "superlevel"] = eqx.field(static=True)
    max_degree: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: CellSubcomplex,
        support: CellVertexSupport,
        coefficients: PrimeField,
        thresholds: ArrayLike,
        /,
        *,
        direction: Literal["sublevel", "superlevel"] = "sublevel",
        max_degree: int | None = None,
        resources: TopologyResourcePolicy | None = None,
    ):
        if direction not in ("sublevel", "superlevel"):
            raise ValueError("Field topology direction must be sublevel or superlevel.")
        threshold_values = jnp.asarray(thresholds)
        if threshold_values.ndim != 1 or not bool(
            jnp.all(jnp.isfinite(threshold_values))
        ):
            raise ValueError("Field topology thresholds must be one finite vector.")
        maximum = complex.max_degree if max_degree is None else int(max_degree)
        if maximum < 0 or maximum > complex.max_degree:
            raise ValueError("Field topology max_degree is outside the complex.")
        policy = TopologyResourcePolicy() if resources is None else resources
        PreparedVertexFiltration(complex, support, direction=direction)
        self.complex = complex
        self.support = support
        self.coefficients = coefficients
        self.resources = policy
        self.thresholds = threshold_values
        self.direction = direction
        self.max_degree = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "field-topology-plan",
                "complex": complex.subcomplex_id,
                "support": support.support_id,
                "field": coefficients.field_id,
                "thresholds": array_tree_fingerprint(threshold_values),
                "direction": direction,
                "max_degree": maximum,
                "resources": policy.policy_id,
            }
        )

    def snapshot(
        self, vertex_values: ArrayLike, /, *, field_id: str
    ) -> FieldTopologySnapshot:
        filtration = (
            lower_star_filtration(
                self.complex,
                self.support,
                vertex_values,
                source_id=field_id,
            )
            if self.direction == "sublevel"
            else upper_star_filtration(
                self.complex,
                self.support,
                vertex_values,
                source_id=field_id,
            )
        )
        persistence = compute_persistence(
            filtration,
            coefficients=self.coefficients,
            max_degree=self.max_degree,
            resources=self.resources,
        )
        diagram = persistence.diagram(include_zero_length=True)
        degrees = np.asarray(diagram.degrees)
        births = np.asarray(diagram.birth_values)
        deaths = np.asarray(diagram.death_values)
        finite = np.asarray(diagram.has_finite_death)
        counts = np.zeros(
            (int(self.thresholds.shape[0]), self.max_degree + 1), dtype=np.int32
        )
        for threshold_index, threshold in enumerate(np.asarray(self.thresholds)):
            alive = (
                (births <= threshold) & (~finite | (deaths > threshold))
                if self.direction == "sublevel"
                else (births >= threshold) & (~finite | (deaths < threshold))
            )
            for degree in range(self.max_degree + 1):
                counts[threshold_index, degree] = int(
                    np.count_nonzero(alive & (degrees == degree))
                )
        return FieldTopologySnapshot(
            filtration,
            persistence,
            self.thresholds,
            counts,
            field_id=field_id,
        )


class FieldTopologySeries(StrictModule, NonTrainableState):
    """Identity-safe sequence of topology snapshots on one fixed layout."""

    snapshots: tuple[FieldTopologySnapshot, ...]
    times: Array
    topology_id: str = eqx.field(static=True)
    series_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshots: Sequence[FieldTopologySnapshot],
        times: ArrayLike,
        /,
    ):
        values = tuple(snapshots)
        time_values = jnp.asarray(times)
        if not values or time_values.shape != (len(values),):
            raise ValueError("Field topology series snapshots and times do not align.")
        if not bool(jnp.all(jnp.isfinite(time_values))) or not bool(
            jnp.all(jnp.diff(time_values) > 0)
        ):
            raise ValueError("Field topology series times must be finite and increasing.")
        topology_ids = {value.filtration.complex.topology.topology_id for value in values}
        layout_ids = {value.filtration.complex.layout.layout_id for value in values}
        if len(topology_ids) != 1 or len(layout_ids) != 1:
            raise ValueError(
                "Field topology series requires one fixed topology and layout."
            )
        topology_id = next(iter(topology_ids))
        self.snapshots = values
        self.times = time_values
        self.topology_id = topology_id
        self.series_id = canonical_fingerprint(
            {
                "kind": "field-topology-series",
                "topology": topology_id,
                "snapshots": [value.snapshot_id for value in values],
                "times": array_tree_fingerprint(time_values),
            }
        )

    @property
    def betti_history(self) -> Array:
        return jnp.stack(tuple(value.betti_counts for value in self.snapshots))


__all__ = ["FieldTopologyPlan", "FieldTopologySeries", "FieldTopologySnapshot"]
