#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...graph import GraphIR
from ...sparse import gather_routes, RowRelation
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._assay import SpatialAssay


NeighborMode = Literal["radius", "knn"]
NeighborWeight = Literal["binary", "inverse_distance", "gaussian"]


class NeighborGraphStatus(IntEnum):
    OK = 0
    CAPACITY_EXCEEDED = 1


@dataclass(frozen=True, slots=True)
class SpatialNeighborPlan:
    """Host plan for a deterministic fixed-capacity spatial graph."""

    mode: NeighborMode
    capacity: int
    radius: float | None = None
    k: int | None = None
    weight: NeighborWeight = "binary"
    bandwidth: float | None = None
    exclude_self: bool = True
    row_normalize: bool = True

    def __post_init__(self):
        if self.mode not in ("radius", "knn"):
            raise ValueError("mode must be 'radius' or 'knn'.")
        if int(self.capacity) <= 0:
            raise ValueError("capacity must be positive.")
        object.__setattr__(self, "capacity", int(self.capacity))
        if self.mode == "radius":
            if self.radius is None or not np.isfinite(self.radius) or self.radius <= 0.0:
                raise ValueError("Radius graphs require a finite positive radius.")
            if self.k is not None:
                raise ValueError("Radius graphs do not accept k.")
            object.__setattr__(self, "radius", float(self.radius))
        else:
            if self.k is None or int(self.k) <= 0:
                raise ValueError("kNN graphs require a positive k.")
            if self.radius is not None:
                raise ValueError("kNN graphs do not accept radius.")
            object.__setattr__(self, "k", int(self.k))
        if self.weight not in ("binary", "inverse_distance", "gaussian"):
            raise ValueError("Unknown spatial neighbor weight.")
        if self.weight == "gaussian":
            if (
                self.bandwidth is None
                or not np.isfinite(self.bandwidth)
                or self.bandwidth <= 0.0
            ):
                raise ValueError("Gaussian graph weights require a positive bandwidth.")
            object.__setattr__(self, "bandwidth", float(self.bandwidth))
        elif self.bandwidth is not None:
            raise ValueError("bandwidth is only defined for gaussian weights.")


class NeighborGraphEvidence(StrictModule):
    required_capacity: Array
    configured_capacity: Array
    edge_count: Array
    maximum_degree: Array
    tied_distance_pairs: Array
    section_count: Array


_NEIGHBOR_CONTRACT = BioinformaticsMethodContract(
    "fixed_capacity_spatial_neighbor_graph",
    MethodKind.EXACT_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.GRAPH,
    conditioning_statement=(
        "Euclidean distances are evaluated in one explicit common spatial frame; "
        "section indices define disconnected graph components."
    ),
    truncation_statement=(
        "No valid graph is returned when the preflight degree exceeds route capacity."
    ),
    capacity_semantics=(
        "The route width is fixed by SpatialNeighborPlan.capacity and overflow is an "
        "observable invalid result with the required capacity."
    ),
    assumptions=("Coordinates use commensurate axes and units.",),
    nondifferentiable_outputs=(
        "relation.source_indices",
        "relation.valid",
        "status",
        "evidence",
    ),
)


class SpatialNeighborGraph(StrictModule):
    """Native sparse row relation with geometry, weights, and capacity evidence."""

    relation: RowRelation
    native_graph: GraphIR
    displacement: Array
    distance: Array
    weight: Array
    valid: Array
    status: Array
    evidence: NeighborGraphEvidence
    method_contract: BioinformaticsMethodContract

    @property
    def indices(self) -> Array:
        return self.relation.source_indices

    @property
    def mask(self) -> Array:
        return self.relation.valid

    @property
    def capacity(self) -> int:
        return self.relation.width

    def gather(self, values: Any, /) -> Any:
        checked = eqx.error_if(values, ~self.valid, "Spatial neighbor graph is invalid.")
        return gather_routes(self.relation, checked)

    def lag(self, values: Any, /) -> Array:
        gathered = self.gather(jnp.asarray(values))
        trailing = (1,) * max(gathered.ndim - self.weight.ndim, 0)
        weights = self.weight.reshape(self.weight.shape + trailing)
        return jnp.sum(weights * gathered, axis=1)


def _padded_sorted_indices(ordered: Array, capacity: int, /) -> Array:
    point_count = int(ordered.shape[1])
    if capacity <= point_count:
        return ordered[:, :capacity]
    padding = jnp.zeros((point_count, capacity - point_count), dtype=ordered.dtype)
    return jnp.concatenate((ordered, padding), axis=1)


def _distinct_count(labels: Array, active: Array, /) -> Array:
    same_prior = (labels[:, None] == labels[None, :]) & active[None, :]
    first = active & ~jnp.any(jnp.tril(same_prior, k=-1), axis=1)
    return jnp.sum(first, dtype=jnp.int32)


def build_spatial_neighbor_graph(
    coordinates: Any,
    plan: SpatialNeighborPlan,
    /,
    *,
    valid_spots: Any | None = None,
    section_index: Any | None = None,
) -> SpatialNeighborGraph:
    """Build a section-isolated graph after a full degree-capacity preflight."""
    if not isinstance(plan, SpatialNeighborPlan):
        raise TypeError("plan must be a SpatialNeighborPlan.")
    points = jnp.asarray(coordinates, dtype=float)
    if points.ndim != 2 or int(points.shape[0]) < 1 or int(points.shape[1]) < 1:
        raise ValueError("coordinates must have shape (spot, coordinate).")
    count = int(points.shape[0])
    valid = (
        jnp.ones((count,), dtype=bool)
        if valid_spots is None
        else jnp.asarray(valid_spots, dtype=bool)
    )
    sections = (
        jnp.zeros((count,), dtype=jnp.int32)
        if section_index is None
        else jnp.asarray(section_index, dtype=jnp.int32)
    )
    if valid.shape != (count,) or sections.shape != (count,):
        raise ValueError("valid_spots and section_index must have shape (spot,).")
    if np.any(np.asarray(sections) < 0):
        raise ValueError("section_index entries must be non-negative.")
    if np.any(~np.isfinite(np.asarray(points)[np.asarray(valid)])):
        raise ValueError("Valid spatial graph coordinates must be finite.")

    displacement_all = points[:, None, :] - points[None, :, :]
    distance_squared = jnp.sum(displacement_all * displacement_all, axis=-1)
    candidates = (
        valid[:, None] & valid[None, :] & (sections[:, None] == sections[None, :])
    )
    if plan.exclude_self:
        candidates = candidates & ~jnp.eye(count, dtype=bool)
    k_value: int | None = None
    if plan.mode == "radius":
        radius = plan.radius
        if radius is None:
            raise ValueError("Radius neighbor plans require radius.")
        candidates = candidates & (distance_squared <= radius**2)
        wanted = jnp.sum(candidates, axis=1, dtype=jnp.int32)
    else:
        k_value = plan.k
        if k_value is None:
            raise ValueError("kNN neighbor plans require k.")
        available = jnp.sum(candidates, axis=1, dtype=jnp.int32)
        wanted = jnp.minimum(available, k_value)

    required_capacity = jnp.max(wanted, initial=0)
    overflow = required_capacity > plan.capacity
    sortable = jnp.where(candidates, distance_squared, jnp.inf)
    ordered = jnp.argsort(sortable, axis=1, stable=True)
    indices = _padded_sorted_indices(ordered, plan.capacity)
    selected_distance_squared = jnp.take_along_axis(distance_squared, indices, axis=1)
    selected_candidates = jnp.take_along_axis(candidates, indices, axis=1)
    slot = jnp.arange(plan.capacity)[None, :]
    if k_value is not None:
        selected_candidates = selected_candidates & (slot < k_value)
    selected_candidates = selected_candidates & (slot < wanted[:, None])
    selected_candidates = selected_candidates & ~overflow
    selected_displacement = jnp.take_along_axis(
        displacement_all,
        indices[..., None],
        axis=1,
    )
    selected_distance = jnp.sqrt(jnp.maximum(selected_distance_squared, 0.0))
    selected_displacement = jnp.where(
        selected_candidates[..., None], selected_displacement, 0.0
    )
    selected_distance = jnp.where(selected_candidates, selected_distance, 0.0)

    if plan.weight == "binary":
        weights = selected_candidates.astype(points.dtype)
    elif plan.weight == "inverse_distance":
        safe_distance = jnp.maximum(
            selected_distance, jnp.finfo(selected_distance.dtype).tiny
        )
        weights = jnp.where(selected_candidates, 1.0 / safe_distance, 0.0)
    else:
        bandwidth = plan.bandwidth
        if bandwidth is None:
            raise ValueError("Gaussian neighbor weights require bandwidth.")
        weights = jnp.where(
            selected_candidates,
            jnp.exp(-0.5 * (selected_distance / bandwidth) ** 2),
            0.0,
        )
    if plan.row_normalize:
        row_total = jnp.sum(weights, axis=1, keepdims=True)
        weights = jnp.where(
            row_total > 0.0, weights / jnp.where(row_total > 0.0, row_total, 1.0), 0.0
        )

    sorted_distance = jnp.take_along_axis(distance_squared, ordered, axis=1)
    sorted_valid = jnp.take_along_axis(candidates, ordered, axis=1)
    tied = jnp.sum(
        sorted_valid[:, 1:]
        & sorted_valid[:, :-1]
        & (sorted_distance[:, 1:] == sorted_distance[:, :-1]),
        dtype=jnp.int32,
    )
    relation = RowRelation(
        indices.astype(jnp.int32),
        source_size=count,
        valid=selected_candidates,
    )
    receivers = jnp.broadcast_to(
        jnp.arange(count, dtype=jnp.int32)[:, None], indices.shape
    )
    native_graph = GraphIR(
        nodes=points,
        edges={
            "displacement": selected_displacement.reshape((-1, int(points.shape[1]))),
            "distance": selected_distance.reshape((-1,)),
            "weight": weights.reshape((-1,)),
        },
        senders=indices.reshape((-1,)),
        receivers=receivers.reshape((-1,)),
        n_node=jnp.asarray([count], dtype=jnp.int32),
        n_edge=jnp.asarray([count * plan.capacity], dtype=jnp.int32),
        node_mask=valid,
        edge_mask=selected_candidates.reshape((-1,)),
        graph_mask=(~overflow).reshape((1,)),
    )
    evidence = NeighborGraphEvidence(
        required_capacity=required_capacity.astype(jnp.int32),
        configured_capacity=jnp.asarray(plan.capacity, dtype=jnp.int32),
        edge_count=jnp.sum(selected_candidates, dtype=jnp.int32),
        maximum_degree=jnp.max(jnp.sum(selected_candidates, axis=1), initial=0).astype(
            jnp.int32
        ),
        tied_distance_pairs=tied,
        section_count=_distinct_count(sections, valid),
    )
    return SpatialNeighborGraph(
        relation=relation,
        native_graph=native_graph,
        displacement=selected_displacement,
        distance=selected_distance,
        weight=weights,
        valid=~overflow,
        status=jnp.where(
            overflow,
            int(NeighborGraphStatus.CAPACITY_EXCEEDED),
            int(NeighborGraphStatus.OK),
        ).astype(jnp.int32),
        evidence=evidence,
        method_contract=_NEIGHBOR_CONTRACT,
    )


def assay_neighbor_graph(
    assay: SpatialAssay,
    plan: SpatialNeighborPlan,
    /,
) -> SpatialNeighborGraph:
    """Build a graph whose components cannot cross tissue-section boundaries."""
    if not isinstance(assay, SpatialAssay):
        raise TypeError("assay must be a SpatialAssay.")
    return build_spatial_neighbor_graph(
        assay.data.coordinates.values,
        plan,
        valid_spots=assay.data.valid_spots,
        section_index=assay.section_index(),
    )


__all__ = [
    "NeighborGraphEvidence",
    "NeighborGraphStatus",
    "SpatialNeighborGraph",
    "SpatialNeighborPlan",
    "assay_neighbor_graph",
    "build_spatial_neighbor_graph",
]
