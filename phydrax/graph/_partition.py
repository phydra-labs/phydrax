#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from math import gcd
from operator import index
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import graph_partition as _native_graph_partition, meshcore_identity
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked


_INT64_MAX = int(np.iinfo(np.int64).max)
_INT32_MAX = int(np.iinfo(np.int32).max)
_NATIVE_WEIGHT_TOTAL_MAX = 2**53


def _integer_vector(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size and array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers.")
    if array.dtype.kind == "u" and array.size and int(np.max(array)) > _INT64_MAX:
        raise ValueError(f"{name} must fit in int64.")
    return array.astype(np.int64, copy=False)


def _positive_exact_weights(value: ArrayLike | None, count: int, /) -> np.ndarray:
    if value is None:
        return np.ones((count,), dtype=np.int64)
    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in "iuf":
        raise ValueError("vertex_weights must be one finite positive weight per vertex.")
    if raw.dtype.kind == "f":
        if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
            raise ValueError("vertex_weights must be finite and positive.")
        # The native kernel consumes integers. Scale the represented binary
        # weights by their exact common power-of-two denominator, then reduce
        # their common divisor; never silently quantize balancing semantics.
        ratios = [Fraction.from_float(float(item)) for item in raw]
        denominator = max(item.denominator for item in ratios)
        integers = [item.numerator * (denominator // item.denominator) for item in ratios]
    else:
        integers = [int(item) for item in raw]
        if any(item <= 0 for item in integers):
            raise ValueError("vertex_weights must be finite and positive.")
    divisor = 0
    for item in integers:
        divisor = gcd(divisor, item)
    reduced = [item // divisor for item in integers]
    if (
        any(item > _INT64_MAX for item in reduced)
        or sum(reduced) > _NATIVE_WEIGHT_TOTAL_MAX
    ):
        raise ValueError(
            "vertex_weights cannot be represented exactly by the native integer-weight partition kernel."
        )
    return np.asarray(reduced, dtype=np.int64)


def _canonical_csr(
    offsets: ArrayLike,
    neighbors: ArrayLike,
    edge_weights: ArrayLike | None,
    vertex_weights: ArrayLike | None,
    vertex_ids: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_offsets = _integer_vector(offsets, "offsets")
    adjacency = _integer_vector(neighbors, "neighbors")
    count = row_offsets.size - 1
    if count < 1 or count >= _INT32_MAX:
        raise ValueError("CSR graph vertex count must fit the native int32 index range.")
    if (
        row_offsets[0] != 0
        or np.any(np.diff(row_offsets) < 0)
        or row_offsets[-1] != adjacency.size
    ):
        raise ValueError(
            "offsets must start at zero, be monotone, and end at the adjacency length."
        )
    if adjacency.size and (np.any(adjacency < 0) or np.any(adjacency >= count)):
        raise ValueError("neighbors contain an out-of-range vertex index.")
    if adjacency.size and np.any(adjacency >= _INT32_MAX):
        raise ValueError("neighbors exceed the native int32 index range.")
    edges = (
        np.ones((adjacency.size,), dtype=np.int64)
        if edge_weights is None
        else _integer_vector(edge_weights, "edge_weights")
    )
    if edges.shape != adjacency.shape or np.any(edges < 0):
        raise ValueError(
            "edge_weights must be nonnegative and align with adjacency entries."
        )
    if sum(map(int, edges)) > _NATIVE_WEIGHT_TOTAL_MAX:
        raise ValueError(
            "edge_weights exceed the exact integer range of the native kernel."
        )
    vertices = _positive_exact_weights(vertex_weights, count)
    if sum(map(int, vertices)) > _NATIVE_WEIGHT_TOTAL_MAX:
        raise ValueError(
            "vertex_weights exceed the exact integer range of the native kernel."
        )
    ids = (
        np.arange(count, dtype=np.int64)
        if vertex_ids is None
        else _integer_vector(vertex_ids, "vertex_ids")
    )
    if ids.shape != (count,) or np.any(ids < 0) or np.unique(ids).size != count:
        raise ValueError("vertex_ids must be unique nonnegative IDs, one per CSR vertex.")

    rows = np.repeat(np.arange(count, dtype=np.int64), np.diff(row_offsets))
    if np.any(rows == adjacency):
        raise ValueError("CSR graph cannot contain self loops.")
    order = np.lexsort((adjacency, rows))
    rows, adjacency, edges = rows[order], adjacency[order], edges[order]
    if rows.size > 1 and np.any(
        (rows[1:] == rows[:-1]) & (adjacency[1:] == adjacency[:-1])
    ):
        raise ValueError("CSR graph cannot repeat an adjacency entry.")
    reverse = np.lexsort((rows, adjacency))
    if not np.array_equal(adjacency[reverse], rows) or not np.array_equal(
        rows[reverse], adjacency
    ):
        raise ValueError("CSR graph must be symmetric.")
    if not np.array_equal(edges[reverse], edges):
        raise ValueError("Both directions of an edge must have equal weights.")

    canonical_offsets = np.zeros((count + 1,), dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=count), out=canonical_offsets[1:])
    return canonical_offsets, adjacency.astype(np.int32), edges, vertices, ids


@final
class WeightedCSRGraph(StrictModule, NonTrainableState):
    """Canonical symmetric weighted CSR graph with stable vertex identities."""

    offsets: Array
    neighbors: Array
    edge_weights: Array
    vertex_weights: Array
    entity_ids: Array
    vertex_count: int = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        offsets: ArrayLike,
        neighbors: ArrayLike,
        /,
        *,
        edge_weights: ArrayLike | None = None,
        vertex_weights: ArrayLike | None = None,
        entity_ids: ArrayLike | None = None,
    ) -> None:
        rows, adjacent, edges, vertices, ids = _canonical_csr(
            offsets, neighbors, edge_weights, vertex_weights, entity_ids
        )
        offsets_, neighbors_, edges_, vertices_, ids_ = (
            np.asarray(rows, dtype=np.int64),
            np.asarray(adjacent, dtype=np.int64),
            np.asarray(edges, dtype=np.int64),
            np.asarray(vertices, dtype=np.int64),
            np.asarray(ids, dtype=np.int64),
        )
        source_rows = np.repeat(np.arange(ids_.size, dtype=np.int64), np.diff(offsets_))
        upper = source_rows < neighbors_
        edge_ids = np.sort(
            np.stack((ids_[source_rows[upper]], ids_[neighbors_[upper]]), axis=1),
            axis=1,
        )
        edge_order = np.lexsort((edge_ids[:, 1], edge_ids[:, 0]))
        logical_edges = np.concatenate(
            (edge_ids[edge_order], edges_[upper][edge_order, None]), axis=1
        )
        id_order = np.argsort(ids_, kind="stable")
        self.graph_id = canonical_fingerprint(
            {
                "kind": "weighted-csr-graph",
                "entity_ids": array_tree_fingerprint(ids_[id_order]),
                "vertex_weights": array_tree_fingerprint(vertices_[id_order]),
                "undirected_edges": array_tree_fingerprint(logical_edges),
            }
        )
        self.offsets = jnp.asarray(offsets_, dtype=jnp.int64)
        self.neighbors = jnp.asarray(neighbors_, dtype=jnp.int32)
        self.edge_weights = jnp.asarray(edges_, dtype=jnp.int64)
        self.vertex_weights = jnp.asarray(vertices_, dtype=jnp.int64)
        self.entity_ids = jnp.asarray(ids_, dtype=jnp.int64)
        self.vertex_count = int(vertices_.size)

    @classmethod
    def from_edges(
        cls,
        vertex_count: int,
        edges: ArrayLike,
        /,
        *,
        edge_weights: ArrayLike | None = None,
        vertex_weights: ArrayLike | None = None,
        entity_ids: ArrayLike | None = None,
    ) -> WeightedCSRGraph:
        count = index(vertex_count)
        if isinstance(vertex_count, bool) or count <= 0 or count >= _INT32_MAX:
            raise ValueError("vertex_count must lie within the native int32 index range.")
        pairs = np.asarray(edges)
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
            raise ValueError("edges must be an integer array with shape (edge_count, 2).")
        pair_ids = _integer_vector(pairs.reshape(-1), "edges").reshape((-1, 2))
        weights = (
            np.ones((pair_ids.shape[0],), dtype=np.int64)
            if edge_weights is None
            else _integer_vector(edge_weights, "edge_weights")
        )
        if weights.shape != (pair_ids.shape[0],):
            raise ValueError("edge_weights must align with undirected edges.")
        directed = np.concatenate((pair_ids, pair_ids[:, ::-1]), axis=0)
        directed_weights = np.concatenate((weights, weights))
        offsets = np.zeros((count + 1,), dtype=np.int64)
        if directed.size:
            if np.any(directed < 0) or np.any(directed >= count):
                raise ValueError("edges contain an out-of-range vertex index.")
            order = np.lexsort((directed[:, 1], directed[:, 0]))
            directed, directed_weights = directed[order], directed_weights[order]
            np.cumsum(np.bincount(directed[:, 0], minlength=count), out=offsets[1:])
        return cls(
            offsets,
            directed[:, 1],
            edge_weights=directed_weights,
            vertex_weights=vertex_weights,
            entity_ids=entity_ids,
        )


@final
class GraphPartitionPlan(StrictModule, NonTrainableState):
    """Target shares and numerical refinement schedule for one graph partition."""

    target_shares: tuple[float, ...] = eqx.field(static=True)
    maximum_imbalance: float = eqx.field(static=True)
    require_nonempty: bool = eqx.field(static=True)
    refinement_passes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_count: int,
        /,
        *,
        target_shares: ArrayLike | None = None,
        maximum_imbalance: float = 1.05,
        require_nonempty: bool = True,
        refinement_passes: int = 4,
    ) -> None:
        try:
            count = index(part_count)
        except TypeError as error:
            raise TypeError("part_count must be an integer.") from error
        if isinstance(part_count, bool) or count <= 0:
            raise ValueError("part_count must be a positive integer.")
        shares = (
            np.ones((count,), dtype=np.float64)
            if target_shares is None
            else np.asarray(target_shares, dtype=np.float64)
        )
        imbalance = float(maximum_imbalance)
        try:
            passes = index(refinement_passes)
        except TypeError as error:
            raise TypeError("refinement_passes must be an integer.") from error
        if (
            shares.shape != (count,)
            or np.any(~np.isfinite(shares))
            or np.any(shares < 0)
            or not np.any(shares > 0)
        ):
            raise ValueError(
                "target_shares must be finite, nonnegative, and include positive load."
            )
        if not np.isfinite(imbalance) or imbalance < 1.0:
            raise ValueError("maximum_imbalance must be finite and at least one.")
        if isinstance(refinement_passes, bool):
            raise TypeError("refinement_passes must be an integer.")
        if passes < 0 or passes > np.iinfo(np.int32).max:
            raise ValueError("refinement_passes must be a nonnegative int32 value.")
        if not isinstance(require_nonempty, bool):
            raise TypeError("require_nonempty must be bool.")
        share_tuple = tuple(float(value) for value in shares)
        self.target_shares = share_tuple
        self.maximum_imbalance = imbalance
        self.require_nonempty = require_nonempty
        self.refinement_passes = passes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-graph-partition-plan",
                "shares": list(share_tuple),
                "maximum_imbalance": imbalance,
                "require_nonempty": require_nonempty,
                "refinement_passes": passes,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[GraphPartitionPlan], value: GraphPartitionPlan, /
    ) -> GraphPartitionPlan:
        if type(value) is not cls:
            raise TypeError("value must be GraphPartitionPlan.")
        if type(value.target_shares) is not tuple:
            raise TypeError("Stored graph partition target_shares must be a tuple.")
        rebuilt = cls(
            len(value.target_shares),
            target_shares=np.asarray(value.target_shares, dtype=np.float64),
            maximum_imbalance=value.maximum_imbalance,
            require_nonempty=value.require_nonempty,
            refinement_passes=value.refinement_passes,
        )
        if (
            rebuilt.plan_id != value.plan_id
            or rebuilt.target_shares != value.target_shares
            or rebuilt.maximum_imbalance != value.maximum_imbalance
            or rebuilt.require_nonempty is not value.require_nonempty
            or rebuilt.refinement_passes != value.refinement_passes
        ):
            raise ValueError(
                "Stored graph partition plan differs from its canonical fields."
            )
        return rebuilt

    @property
    def part_count(self) -> int:
        return len(self.target_shares)


def _targets_and_capacities(
    weights: np.ndarray, plan: GraphPartitionPlan, /
) -> tuple[np.ndarray, np.ndarray]:
    total = sum(map(int, weights))
    fractions = tuple(Fraction.from_float(value) for value in plan.target_shares)
    fraction_total = sum(fractions, Fraction())
    exact = tuple(Fraction(total) * value / fraction_total for value in fractions)
    targets = np.asarray(
        [value.numerator // value.denominator for value in exact], dtype=np.int64
    )
    remainder = total - sum(map(int, targets))
    priority = sorted(
        range(plan.part_count),
        key=lambda index: (-(exact[index] - int(targets[index])), index),
    )
    for part in priority[:remainder]:
        targets[part] += 1
    if plan.require_nonempty and np.any(targets == 0):
        raise ValueError(
            "The native graph partition cannot make every requested part nonempty."
        )
    capacities = np.empty((plan.part_count,), dtype=np.int64)
    for part, share in enumerate(fractions):
        capacity = (
            Fraction(total)
            * share
            / fraction_total
            * Fraction.from_float(plan.maximum_imbalance)
        )
        rounded = (capacity.numerator + capacity.denominator - 1) // capacity.denominator
        capacities[part] = min(_NATIVE_WEIGHT_TOTAL_MAX, max(int(targets[part]), rounded))
    return targets, capacities


def _id_ordered_csr(
    graph: WeightedCSRGraph, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(graph.offsets, dtype=np.int64)
    neighbors = np.asarray(graph.neighbors, dtype=np.int32)
    edges = np.asarray(graph.edge_weights, dtype=np.int64)
    ids = np.asarray(graph.entity_ids, dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.int64)
    rows: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    new_offsets = np.zeros((order.size + 1,), dtype=np.int64)
    for new_row, old_row in enumerate(order):
        start, stop = int(offsets[old_row]), int(offsets[old_row + 1])
        mapped = inverse[neighbors[start:stop]]
        local_order = np.argsort(mapped, kind="stable")
        rows.append(mapped[local_order].astype(np.int32, copy=False))
        edge_rows.append(edges[start:stop][local_order])
        new_offsets[new_row + 1] = new_offsets[new_row] + stop - start
    new_neighbors = np.concatenate(rows) if rows else np.zeros((0,), dtype=np.int32)
    new_edges = np.concatenate(edge_rows) if edge_rows else np.zeros((0,), dtype=np.int64)
    return new_offsets, new_neighbors, new_edges, order


@final
class GraphPartitionResult(StrictModule, NonTrainableState):
    """Native partition labels and measured cut/load feasibility."""

    owners: Array
    part_weights: Array
    target_weights: Array
    capacity_weights: Array
    edge_cut: int = eqx.field(static=True)
    imbalance: float = eqx.field(static=True)
    feasible: bool = eqx.field(static=True)
    feasibility_status: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        owners: np.ndarray,
        part_weights: np.ndarray,
        target_weights: np.ndarray,
        capacity_weights: np.ndarray,
        edge_cut: int,
        imbalance: float,
        feasible: bool,
        provenance: str,
        graph_id: str,
        plan_id: str,
        /,
    ) -> None:
        self.owners = jnp.asarray(owners, dtype=jnp.int32)
        self.part_weights = jnp.asarray(part_weights, dtype=jnp.int64)
        self.target_weights = jnp.asarray(target_weights, dtype=jnp.int64)
        self.capacity_weights = jnp.asarray(capacity_weights, dtype=jnp.int64)
        self.edge_cut = int(edge_cut)
        self.imbalance = float(imbalance)
        self.feasible = bool(feasible)
        self.feasibility_status = "feasible" if feasible else "balance_not_reached"
        self.provenance = provenance
        self.result_id = canonical_fingerprint(
            {
                "kind": "native-graph-partition-result",
                "graph": graph_id,
                "plan": plan_id,
                "owners": array_tree_fingerprint(owners),
                "part_weights": array_tree_fingerprint(part_weights),
                "target_weights": array_tree_fingerprint(target_weights),
                "capacity_weights": array_tree_fingerprint(capacity_weights),
                "edge_cut": int(edge_cut),
                "feasible": bool(feasible),
                "provenance": provenance,
            }
        )


@checked
def partition_graph(
    graph: WeightedCSRGraph, plan: GraphPartitionPlan, /
) -> GraphPartitionResult:
    if not isinstance(graph, WeightedCSRGraph) or not isinstance(
        plan, GraphPartitionPlan
    ):
        raise TypeError(
            "partition_graph requires WeightedCSRGraph and GraphPartitionPlan values."
        )
    if plan.require_nonempty and plan.part_count > graph.vertex_count:
        raise ValueError("part_count exceeds the graph vertex count.")
    vertices = np.asarray(graph.vertex_weights, dtype=np.int64)
    targets, capacities = _targets_and_capacities(vertices, plan)
    offsets, neighbors, edge_weights, order = _id_ordered_csr(graph)
    ordered_vertices = vertices[order]
    ordered_targets = np.ascontiguousarray(targets)
    ordered_capacities = np.ascontiguousarray(capacities)
    labels = np.asarray(
        _native_graph_partition(
            offsets,
            neighbors,
            edge_weights,
            ordered_vertices,
            ordered_targets,
            ordered_capacities,
            require_nonempty=plan.require_nonempty,
            refinement_passes=plan.refinement_passes,
        ),
        dtype=np.int32,
    )
    if (
        labels.shape != (graph.vertex_count,)
        or np.any(labels < 0)
        or np.any(labels >= plan.part_count)
    ):
        raise RuntimeError(
            "The native graph partition kernel returned invalid part labels."
        )
    owners = np.empty_like(labels)
    owners[order] = labels
    loads = np.bincount(owners, weights=vertices, minlength=plan.part_count).astype(
        np.int64
    )
    if plan.require_nonempty and np.any(loads == 0):
        raise RuntimeError(
            "The native graph partition kernel returned an empty requested part."
        )
    offsets_source = np.asarray(graph.offsets, dtype=np.int64)
    neighbors_source = np.asarray(graph.neighbors, dtype=np.int64)
    rows = np.repeat(
        np.arange(graph.vertex_count, dtype=np.int64), np.diff(offsets_source)
    )
    cut = int(
        np.count_nonzero(
            (owners[rows] != owners[neighbors_source]) & (rows < neighbors_source)
        )
    )
    positive = targets > 0
    imbalance = (
        float(np.max(loads[positive] / targets[positive]))
        if np.any(positive)
        else float("inf")
    )
    feasible = bool(np.all(loads <= capacities))
    return GraphPartitionResult(
        owners,
        loads,
        targets,
        capacities,
        cut,
        imbalance,
        feasible,
        meshcore_identity(),
        graph.graph_id,
        plan.plan_id,
    )


__all__ = [
    "GraphPartitionPlan",
    "GraphPartitionResult",
    "WeightedCSRGraph",
    "partition_graph",
]
