#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic multilevel k-way partitioning of weighted CSR graphs.

`phydrax.graph` owns graph validation and sorted CSR form, target shares and
balance capacities, the nonempty-part policy, work
budgets, and the evidence measured on the result. The serial compiled multilevel
kernel ships in meshcore. Owner-local globally logical CSR views use the
Phydrax-owned XLA multilevel route: two heavy-edge contraction levels, scalar
capacity-first weighted allocation, projection and bounded gain refinement.
Only bounded sparse neighbor queries circulate; CSR and candidate tables are
never gathered. Both routes measure evidence from returned parts and never
claim an optimal cut or unmeasured scaling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import fields
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import graph_partition, GraphPartitionCounters, meshcore_identity
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, nonnegative_integer, positive_integer
from ..typing import (
    AnyShape,
    checked,
    ConvertibleToArray,
    Dim,
    Float64,
    HostFloat64,
    HostInt32,
    HostInt64,
    Identifier,
    Int32,
    Int64,
    parse,
    Scope,
)
from ._distributed_partition import execute, logical_id, PartitionMeasurements


GraphEmptyPartPolicy: TypeAlias = Literal["require_nonempty", "permit_empty"]
GraphPartitionStatus: TypeAlias = Literal[
    "balanced", "indivisible_vertex_overload", "balance_not_reached"
]

# Weight totals stay exactly representable in binary64 and free of int64
# overflow in every native gain.
_WEIGHT_LIMIT = 2**53
_VERTEX_LIMIT = 2**31 - 2
_DETERMINISM = (
    "sequential multilevel route: integer gains and weights; seeded rank then "
    "vertex-index ties, destination gain then remaining room then part-index ties; "
    "canonical-input reproducibility under IEEE binary64 target scaling without contraction"
)


class _GraphVertexDim(Dim, minimum=1):
    """Vertices of one partitioned graph."""


class _GraphOffsetDim(Dim, minimum=2):
    """CSR row offsets: one more than the vertex count."""


class _GraphEntryDim(Dim):
    """Directed CSR adjacency entries."""


class _GraphPartDim(Dim, minimum=1):
    """Parts of one partition."""


class _GraphIndivisibleDim(Dim):
    """Vertices heavier than every part capacity."""


def _integer_vector(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(
            f"WeightedCSRGraph.{name} must be one-dimensional; got shape {array.shape}"
        )
    # An empty sequence carries no dtype of its own.
    if array.dtype.kind not in "iu" and array.size:
        raise TypeError(
            f"WeightedCSRGraph.{name} must be an integer array; got dtype {array.dtype}"
        )
    if array.dtype == np.uint64 and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(
            f"WeightedCSRGraph.{name} must fit the int64 range; got overflowing uint64 values"
        )
    return array.astype(np.int64)


def _weights(value: ConvertibleToArray | None, size: int, name: str, /) -> np.ndarray:
    if value is None:
        return np.ones((size,), dtype=np.int64)
    weights = _integer_vector(value, name)
    if weights.shape != (size,):
        raise ValueError(
            f"WeightedCSRGraph.{name} must have shape ({size},); got {weights.shape}"
        )
    if np.any(weights < 0):
        raise ValueError(
            f"WeightedCSRGraph.{name} must be nonnegative; got negative weights"
        )
    if sum(map(int, weights)) > _WEIGHT_LIMIT:
        raise ValueError(
            f"WeightedCSRGraph.{name} must total at most 2**53; got an oversized total"
        )
    return weights


def _canonical_rows(
    offsets: np.ndarray, neighbors: np.ndarray, edge_weights: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort every row by neighbor and verify a simple symmetric graph."""
    count = offsets.size - 1
    rows = np.repeat(np.arange(count, dtype=np.int64), np.diff(offsets))
    if np.any((neighbors < 0) | (neighbors >= count)):
        raise ValueError(
            "WeightedCSRGraph.neighbors must be vertex indices; got out-of-range adjacency"
        )
    if np.any(rows == neighbors):
        raise ValueError(
            "WeightedCSRGraph.neighbors must exclude self loops; got a self loop"
        )
    order = np.lexsort((neighbors, rows))
    neighbors, edge_weights = neighbors[order], edge_weights[order]
    keys = rows * count + neighbors
    if np.any(keys[1:] == keys[:-1]):
        raise ValueError(
            "WeightedCSRGraph.neighbors must not repeat a neighbor; got duplicate adjacency"
        )
    mirror = np.argsort(neighbors * count + rows, kind="stable")
    if not (
        np.array_equal(neighbors[mirror] * count + rows[mirror], keys)
        and np.array_equal(edge_weights[mirror], edge_weights)
    ):
        raise ValueError(
            "WeightedCSRGraph.adjacency must be symmetric with equal weights; got unmatched directed edges"
        )
    return rows, neighbors, edge_weights


class _OwnerCSR(StrictModule):
    offsets: Array
    neighbors: Array
    edges: Array
    vertices: Array
    vertex_ids: Array
    vertex_owners: Array
    vertex_valid: Array
    vertex_count: int = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)


def _prepare_owner_csr(
    offsets: ConvertibleToArray,
    neighbors: ConvertibleToArray,
    edge_weights: ConvertibleToArray | None,
    vertex_weights: ConvertibleToArray | None,
    vertex_ids: Array | None,
    vertex_owners: Array | None,
    vertex_valid: Array | None,
    mesh: Mesh,
    axis_name: str | None,
) -> _OwnerCSR:
    if axis_name is None or len(mesh.axis_names) != 1 or axis_name not in mesh.axis_names:
        raise ValueError(
            f"WeightedCSRGraph.axis_name must name one execution mesh axis; got {axis_name!r}"
        )
    match offsets, neighbors:
        case Array(), Array():
            pass
        case _:
            raise TypeError(
                "WeightedCSRGraph.rows must use globally logical JAX arrays; got non-array rows"
            )
    if vertex_ids is None or vertex_owners is None or vertex_valid is None:
        raise TypeError(
            "WeightedCSRGraph.ownership must include ID, owner and validity arrays; got missing arrays"
        )
    if any(
        value.ndim != 2
        for value in (offsets, neighbors, vertex_ids, vertex_owners, vertex_valid)
    ):
        raise ValueError(
            "WeightedCSRGraph.rows must align with mesh and ID rows; got non-rank-2 arrays"
        )
    ranks, capacity = vertex_ids.shape
    entries = neighbors.shape[1]
    if (
        ranks != mesh.shape[axis_name]
        or offsets.shape != (ranks, capacity + 1)
        or vertex_owners.shape != (ranks, capacity)
        or vertex_valid.shape != (ranks, capacity)
    ):
        raise ValueError(
            "WeightedCSRGraph.rows must align with mesh and ID rows; got mismatched shapes"
        )
    if (
        capacity < 1
        or entries < 1
        or neighbors.shape[0] != ranks
        or ranks * capacity > _VERTEX_LIMIT
        or ranks * entries > _VERTEX_LIMIT
    ):
        raise ValueError(
            "WeightedCSRGraph.capacity must be positive and int32-addressable; got invalid capacities"
        )
    match edge_weights:
        case None:
            admitted_edges = None
        case Array():
            admitted_edges = edge_weights
        case _:
            raise TypeError(
                "WeightedCSRGraph.edge_weights must be logical JAX arrays; got non-array weights"
            )
    match vertex_weights:
        case None:
            admitted_vertices = None
        case Array():
            admitted_vertices = vertex_weights
        case _:
            raise TypeError(
                "WeightedCSRGraph.vertex_weights must be logical JAX arrays; got non-array weights"
            )
    edges = jnp.ones_like(neighbors) if admitted_edges is None else admitted_edges
    vertices = (
        jnp.ones_like(vertex_ids) if admitted_vertices is None else admitted_vertices
    )
    for value in (offsets, neighbors, edges, vertices, vertex_ids):
        if value.dtype != jnp.int64:
            raise TypeError(
                f"WeightedCSRGraph.rows must use int64 IDs, offsets and weights; got {value.dtype}"
            )
    if vertex_owners.dtype != jnp.int32 or vertex_valid.dtype != jnp.bool_:
        raise TypeError(
            "WeightedCSRGraph.ownership must use int32 owners and bool masks; got invalid dtypes"
        )
    if edges.shape != neighbors.shape or vertices.shape != vertex_ids.shape:
        raise ValueError(
            "WeightedCSRGraph.weights must align with CSR rows; got mismatched shapes"
        )
    placement = NamedSharding(mesh, PartitionSpec(axis_name))
    arrays = (
        offsets,
        neighbors,
        edges,
        vertices,
        vertex_ids,
        vertex_owners,
        vertex_valid,
    )
    if any(not value.sharding.is_equivalent_to(placement, ndim=2) for value in arrays):
        raise ValueError(
            "WeightedCSRGraph.rows must use declared rank-row sharding; got different placement"
        )
    return _OwnerCSR(
        offsets=offsets,
        neighbors=neighbors,
        edges=edges,
        vertices=vertices,
        vertex_ids=vertex_ids,
        vertex_owners=vertex_owners,
        vertex_valid=vertex_valid,
        vertex_count=ranks * capacity,
        graph_id=logical_id(arrays),
    )


@final
class WeightedCSRGraph(StrictModule, NonTrainableState):
    """Simple undirected graph with nonnegative integer vertex and edge weights.

    Row ``v`` lists the neighbors of vertex ``v`` in ``neighbors[offsets[v]:
    offsets[v + 1]]``. Every undirected edge appears in both rows with the same
    weight; self loops and repeated neighbors are refused. Rows are stored
    sorted by neighbor to define graph identity. Weights default to one;
    each weight total is at most ``2**53`` and the vertex total is positive.
    """

    __strict_contract__ = True

    offsets: HostInt64[_GraphOffsetDim] | Int64[AnyShape]
    neighbors: HostInt32[_GraphEntryDim] | Int64[AnyShape]
    edge_weights: HostInt64[_GraphEntryDim] | Int64[AnyShape]
    vertex_weights: HostInt64[_GraphVertexDim] | Int64[AnyShape]
    vertex_count: int = eqx.field(static=True)
    graph_id: Identifier = eqx.field(static=True)
    vertex_ids: Array | None
    vertex_owners: Array | None
    vertex_valid: Array | None
    mesh: Mesh | None = eqx.field(static=True)
    axis_name: str | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        offsets: ConvertibleToArray,
        neighbors: ConvertibleToArray,
        /,
        *,
        edge_weights: ConvertibleToArray | None = None,
        vertex_weights: ConvertibleToArray | None = None,
        vertex_ids: Array | None = None,
        vertex_owners: Array | None = None,
        vertex_valid: Array | None = None,
        mesh: Mesh | None = None,
        axis_name: str | None = None,
    ) -> None:
        if mesh is not None:
            prepared = _prepare_owner_csr(
                offsets,
                neighbors,
                edge_weights,
                vertex_weights,
                vertex_ids,
                vertex_owners,
                vertex_valid,
                mesh,
                axis_name,
            )
            self.offsets = prepared.offsets
            self.neighbors = prepared.neighbors
            self.edge_weights = prepared.edges
            self.vertex_weights = prepared.vertices
            self.vertex_ids = prepared.vertex_ids
            self.vertex_owners = prepared.vertex_owners
            self.vertex_valid = prepared.vertex_valid
            self.mesh = mesh
            self.axis_name = axis_name
            self.vertex_count = prepared.vertex_count
            self.graph_id = prepared.graph_id
            return
        if any(
            value is not None
            for value in (vertex_ids, vertex_owners, vertex_valid, axis_name)
        ):
            raise ValueError(
                "WeightedCSRGraph.ownership must include an execution mesh; got ID/ownership views without mesh"
            )
        row_offsets = _integer_vector(offsets, "offsets")
        adjacency = _integer_vector(neighbors, "neighbors")
        count = row_offsets.size - 1
        if count < 1 or count > _VERTEX_LIMIT:
            raise ValueError(
                f"WeightedCSRGraph.vertex_count must lie between 1 and {_VERTEX_LIMIT}; got {count}"
            )
        if (
            row_offsets[0] != 0
            or np.any(np.diff(row_offsets) < 0)
            or row_offsets[-1] != adjacency.size
        ):
            raise ValueError(
                "WeightedCSRGraph.offsets must start at 0, not decrease, and end at the neighbor count; got invalid CSR offsets"
            )
        edges = _weights(edge_weights, adjacency.size, "edge_weights")
        vertices = _weights(vertex_weights, count, "vertex_weights")
        if not np.any(vertices > 0):
            raise ValueError(
                "WeightedCSRGraph.vertex_weights must have a positive total; got zero total"
            )
        _, adjacency, edges = _canonical_rows(row_offsets, adjacency, edges)
        scope = Scope()
        self.offsets = parse(
            _frozen(row_offsets), HostInt64[_GraphOffsetDim], "offsets", scope=scope
        )
        self.neighbors = parse(
            _frozen(adjacency.astype(np.int32)),
            HostInt32[_GraphEntryDim],
            "neighbors",
            scope=scope,
        )
        self.edge_weights = parse(
            _frozen(edges), HostInt64[_GraphEntryDim], "edge_weights", scope=scope
        )
        self.vertex_weights = parse(
            _frozen(vertices), HostInt64[_GraphVertexDim], "vertex_weights", scope=scope
        )
        self.vertex_count = count
        self.vertex_ids = vertex_ids
        self.vertex_owners = vertex_owners
        self.vertex_valid = vertex_valid
        self.mesh = mesh
        self.axis_name = axis_name
        self.graph_id = canonical_fingerprint(
            {
                "kind": "weighted-csr-graph",
                "offsets": array_tree_fingerprint(self.offsets),
                "neighbors": array_tree_fingerprint(self.neighbors),
                "edge_weights": array_tree_fingerprint(self.edge_weights),
                "vertex_weights": array_tree_fingerprint(self.vertex_weights),
            }
        )

    @classmethod
    @checked
    def owner_local(
        cls,
        offsets: Array,
        neighbor_ids: Array,
        vertex_ids: Array,
        vertex_owners: Array,
        vertex_valid: Array,
        /,
        *,
        mesh: Mesh,
        axis_name: str,
        edge_weights: Array | None = None,
        vertex_weights: Array | None = None,
    ) -> WeightedCSRGraph:
        """Bind a globally logical, capacity-bounded owner-local CSR view.

        Shapes are ``[rank, row+1]``, ``[rank, entry]`` and ``[rank, row]``.
        Neighbors are scientific int64 IDs, not slot indices. Valid rows whose
        owner differs from the execution rank are ghosts and do not contribute
        weight or adjacency. Padding entries lie beyond the last row offset.
        Content validation is collective inside :func:`partition_graph`; invalid
        input or exhausted work returns rejected evidence and unchanged owners.
        """
        return cls(
            offsets,
            neighbor_ids,
            edge_weights=edge_weights,
            vertex_weights=vertex_weights,
            vertex_ids=vertex_ids,
            vertex_owners=vertex_owners,
            vertex_valid=vertex_valid,
            mesh=mesh,
            axis_name=axis_name,
        )

    @property
    def neighbor_ids(self) -> Array:
        if self.mesh is None:
            raise ValueError(
                "WeightedCSRGraph.neighbor_ids must belong to an owner-local view; got serial graph"
            )
        match self.neighbors:
            case Array():
                return self.neighbors
            case _:
                raise RuntimeError(
                    "WeightedCSRGraph.neighbors must retain logical JAX arrays; got host adjacency"
                )


def _frozen(array: np.ndarray | Array, /) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@final
class GraphPartitionPlan(StrictModule, NonTrainableState):
    """Part count, balance tolerance, part shares, empty-part policy and budget.

    Part ``p`` targets ``t_p = W * share_p`` of the total vertex weight ``W``
    (shares default to equal and are normalized to sum to one). Its capacity is
    ``min(W, max(floor(maximum_imbalance * t_p), ceil(t_p)))``: no move on the
    input graph exceeds it (coarse levels relax it by their heaviest merged
    vertex), and the partition is balanced exactly when every part weight
    is within its capacity. ``require_nonempty`` needs ``part_count`` at most
    the vertex count and gives every part a vertex; ``permit_empty`` allows
    empty parts. ``refinement_passes`` bounds refinement passes per level and
    ``work_limit`` bounds adjacency visits plus candidate evaluations
    (``None`` is unbounded). Serial exhaustion raises
    :class:`phydrax._meshcore.MeshcoreError` with status ``CAPACITY_EXCEEDED``.
    Owner-local exhaustion collectively returns ``evidence.accepted=False``,
    ``resource_status & 2`` and the unchanged incoming ownership map.
    """

    __strict_contract__ = True

    part_count: int = eqx.field(static=True)
    maximum_imbalance: float = eqx.field(static=True)
    part_shares: tuple[float, ...] = eqx.field(static=True)
    empty_parts: GraphEmptyPartPolicy = eqx.field(static=True)
    refinement_passes: int = eqx.field(static=True)
    work_limit: int | None = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        part_count: int,
        /,
        *,
        maximum_imbalance: float = 1.03,
        part_shares: Sequence[float] | None = None,
        empty_parts: GraphEmptyPartPolicy = "require_nonempty",
        refinement_passes: int = 8,
        work_limit: int | None = None,
    ) -> None:
        parts = positive_integer(part_count, "GraphPartitionPlan.part_count")
        tolerance = finite_real_scalar(
            maximum_imbalance, "GraphPartitionPlan.maximum_imbalance"
        )
        if tolerance < 1.0:
            raise ValueError(
                "GraphPartitionPlan.maximum_imbalance must be at least one; got a value below one"
            )
        policy = parse(
            empty_parts, GraphEmptyPartPolicy, "GraphPartitionPlan.empty_parts"
        )
        passes = nonnegative_integer(
            refinement_passes, "GraphPartitionPlan.refinement_passes"
        )
        budget = (
            None
            if work_limit is None
            else nonnegative_integer(work_limit, "GraphPartitionPlan.work_limit")
        )
        self.part_count = parts
        self.maximum_imbalance = tolerance
        self.part_shares = _shares(part_shares, parts)
        self.empty_parts = policy
        self.refinement_passes = passes
        self.work_limit = budget
        self.plan_id = canonical_fingerprint(
            {
                "kind": "graph-partition-plan",
                "part_count": parts,
                "maximum_imbalance": tolerance,
                "part_shares": list(self.part_shares),
                "empty_parts": policy,
                "refinement_passes": passes,
                "work_limit": budget,
            }
        )


def _shares(value: Sequence[float] | None, parts: int, /) -> tuple[float, ...]:
    if value is None:
        return (1.0 / parts,) * parts
    if isinstance(value, (str, bytes)) or len(value) != parts:
        raise ValueError(
            "GraphPartitionPlan.part_shares must hold one share per part; got mismatched shares"
        )
    shares = tuple(
        finite_real_scalar(share, f"GraphPartitionPlan.part_shares[{index}]")
        for index, share in enumerate(value)
    )
    if any(share <= 0.0 for share in shares):
        raise ValueError(
            "GraphPartitionPlan.part_shares must be positive; got a nonpositive share"
        )
    total = math.fsum(shares)
    return tuple(share / total for share in shares)


@final
class GraphPartitionWork(StrictModule, NonTrainableState):
    """Native work: literal adjacency visits and vertex/candidate evaluations.

    ``work_limit`` bounds their sum, including isolated-vertex attempts.
    A candidate evaluation is a vertex scan, heap attempt or exchange attempt;
    it is not an elapsed-time or byte count.
    """

    coarsening_levels: int = eqx.field(static=True)
    coarsest_vertex_count: int = eqx.field(static=True)
    matched_pairs: int = eqx.field(static=True)
    bisection_trials: int = eqx.field(static=True)
    refinement_passes: int = eqx.field(static=True)
    moves_committed: int = eqx.field(static=True)
    moves_rolled_back: int = eqx.field(static=True)
    balance_moves: int = eqx.field(static=True)
    nonempty_repairs: int = eqx.field(static=True)
    adjacency_visits: int = eqx.field(static=True)
    candidate_evaluations: int = eqx.field(static=True)

    @checked
    def __init__(self, counters: GraphPartitionCounters, /) -> None:
        if any(
            type(value) is not int or value < 0 or value > np.iinfo(np.int64).max
            for _, value in counters.items()
        ):
            raise ValueError(
                "GraphPartitionWork.counters must be nonnegative int64 values; got invalid counters"
            )
        self.coarsening_levels = counters.coarsening_levels
        self.coarsest_vertex_count = counters.coarsest_vertex_count
        self.matched_pairs = counters.matched_pairs
        self.bisection_trials = counters.bisection_trials
        self.refinement_passes = counters.refinement_passes
        self.moves_committed = counters.moves_committed
        self.moves_rolled_back = counters.moves_rolled_back
        self.balance_moves = counters.balance_moves
        self.nonempty_repairs = counters.nonempty_repairs
        self.adjacency_visits = counters.adjacency_visits
        self.candidate_evaluations = counters.candidate_evaluations

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[GraphPartitionWork], value: GraphPartitionWork, /
    ) -> GraphPartitionWork:
        """Reconstruct every retained native counter through its owning contract."""
        return cls(GraphPartitionCounters(**{
            member.name: getattr(value, member.name)
            for member in fields(GraphPartitionCounters)
        }))


class _CollectiveHostMeasurements(StrictModule):
    weights: np.ndarray
    counts: np.ndarray
    targets: np.ndarray
    capacities: np.ndarray
    resource: int = eqx.field(static=True)
    cut: int = eqx.field(static=True)
    cut_count: int = eqx.field(static=True)
    boundary: int = eqx.field(static=True)
    migration: int = eqx.field(static=True)
    ghosts: int = eqx.field(static=True)
    maximum: int = eqx.field(static=True)


def _collective_host_measurements(
    measured: PartitionMeasurements,
) -> _CollectiveHostMeasurements:
    weights, counts, targets, capacities, *scalars = jax.device_get(
        (
            measured.part_weights,
            measured.part_vertex_counts,
            measured.targets,
            measured.capacities,
            measured.resource_status,
            measured.edge_cut,
            measured.cut_edge_count,
            measured.boundary_vertex_count,
            measured.migration_vertex_count,
            measured.ghost_entry_count,
            measured.maximum_vertex_weight,
        )
    )
    resource, cut, cut_count, boundary, migration, ghosts, maximum = (
        int(value) for value in scalars
    )
    if resource == 0 and (
        np.any(weights < 0)
        or np.any(counts < 0)
        or np.any(~np.isfinite(targets))
        or np.any(targets <= 0)
        or any(
            value < 0 for value in (cut, cut_count, boundary, migration, ghosts, maximum)
        )
    ):
        raise ValueError(
            "GraphPartitionEvidence.measurements must be nonnegative with positive finite targets on acceptance; got invalid measurements"
        )
    return _CollectiveHostMeasurements(
        weights=weights,
        counts=counts,
        targets=targets,
        capacities=capacities,
        resource=resource,
        cut=cut,
        cut_count=cut_count,
        boundary=boundary,
        migration=migration,
        ghosts=ghosts,
        maximum=maximum,
    )


@final
class GraphPartitionEvidence(StrictModule, NonTrainableState):
    """Cut, balance and work of one partition, measured from its parts.

    ``edge_cut`` is the total weight of edges joining different parts and
    ``cut_edge_count`` their number; ``boundary_vertex_count`` counts vertices
    with a neighbor in another part. ``imbalance`` is ``max_p w_p / t_p`` and
    ``imbalance_lower_bound = max(1, max_v w_v / max_p t_p)`` bounds it for every
    partition. ``status`` is ``balanced`` when every part is within capacity,
    ``indivisible_vertex_overload`` when ``indivisible_vertices`` (vertices
    heavier than every capacity) make that impossible, and ``balance_not_reached``
    otherwise. The cut is not claimed optimal.

    For owner-local views, ``accepted`` distinguishes a completed partition
    from collective refusal: ``resource_status`` has invalid-input bit 1 and
    work-capacity bit 2. Rejection preserves incoming ownership on every row.
    ``indivisible_vertices`` is an aligned int64 scientific-ID table with max
    int64 padding rather than serial vertex indices. ``migration_vertex_count``
    counts source-owned vertices whose accepted owner changed;
    ``ghost_entry_count`` counts directed crossing-edge incidences, not unique
    halo vertices. No graph cut or scaling optimality is asserted.
    """

    __strict_contract__ = True

    part_weights: HostInt64[_GraphPartDim] | Int64[AnyShape]
    part_targets: HostFloat64[_GraphPartDim] | Float64[AnyShape]
    part_capacities: HostInt64[_GraphPartDim] | Int64[AnyShape]
    part_vertex_counts: HostInt64[_GraphPartDim] | Int64[AnyShape]
    indivisible_vertices: HostInt64[_GraphIndivisibleDim] | Int64[AnyShape]
    status: GraphPartitionStatus = eqx.field(static=True)
    imbalance: float = eqx.field(static=True)
    imbalance_lower_bound: float = eqx.field(static=True)
    edge_cut: int = eqx.field(static=True)
    cut_edge_count: int = eqx.field(static=True)
    boundary_vertex_count: int = eqx.field(static=True)
    empty_part_count: int = eqx.field(static=True)
    work: GraphPartitionWork
    determinism: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    evidence_id: Identifier = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    resource_status: int = eqx.field(static=True)
    migration_vertex_count: int = eqx.field(static=True)
    ghost_entry_count: int = eqx.field(static=True)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[GraphPartitionEvidence], value: GraphPartitionEvidence, /
    ) -> GraphPartitionEvidence:
        """Restore the descriptor; its source graph is re-earned by its owner."""
        return cls(
            part_weights=value.part_weights,
            part_targets=value.part_targets,
            part_capacities=value.part_capacities,
            part_vertex_counts=value.part_vertex_counts,
            indivisible_vertices=value.indivisible_vertices,
            status=value.status,
            imbalance=value.imbalance,
            imbalance_lower_bound=value.imbalance_lower_bound,
            edge_cut=value.edge_cut,
            cut_edge_count=value.cut_edge_count,
            boundary_vertex_count=value.boundary_vertex_count,
            empty_part_count=value.empty_part_count,
            work=GraphPartitionWork.from_stored_fields(value.work),
            determinism=value.determinism,
            backend=value.backend,
            evidence_id=value.evidence_id,
            accepted=value.accepted,
            resource_status=value.resource_status,
            migration_vertex_count=value.migration_vertex_count,
            ghost_entry_count=value.ghost_entry_count,
        )

    @classmethod
    @checked
    def serial(
        cls,
        graph: WeightedCSRGraph,
        parts: np.ndarray,
        targets: np.ndarray,
        capacities: np.ndarray,
        work: GraphPartitionWork,
        backend: str,
        /,
    ) -> GraphPartitionEvidence:
        """Measure cut, balance and work from a serial partition."""
        if graph.mesh is not None:
            raise ValueError(
                "GraphPartitionEvidence.graph must be serial; got owner-local graph"
            )
        accepted, resource_status = True, 0
        migration_vertex_count, ghost_entry_count = 0, 0
        count = targets.size
        if parts.dtype != np.int32 or parts.shape != (graph.vertex_count,):
            raise ValueError(
                "GraphPartitionEvidence.parts must be int32 with one entry per vertex; got invalid dtype or shape"
            )
        if np.any((parts < 0) | (parts >= count)):
            raise ValueError(
                "GraphPartitionEvidence.parts must contain in-range part indices; got out-of-range part index"
            )
        if capacities.dtype != np.int64 or capacities.shape != (count,):
            raise ValueError(
                "GraphPartitionEvidence.capacities must be int64 with one entry per part; got invalid dtype or shape"
            )
        if (
            targets.dtype != np.float64
            or np.any(~np.isfinite(targets))
            or np.any(targets <= 0)
        ):
            raise ValueError(
                "GraphPartitionEvidence.targets must be positive finite float64 values; got invalid targets"
            )
        weights = np.zeros((count,), dtype=np.int64)
        np.add.at(weights, parts, graph.vertex_weights)
        rows = np.repeat(
            np.arange(graph.vertex_count, dtype=np.int64), np.diff(graph.offsets)
        )
        crossing = parts[rows] != parts[graph.neighbors]
        indivisible = np.flatnonzero(graph.vertex_weights > np.max(capacities))
        if indivisible.size:
            status: GraphPartitionStatus = "indivisible_vertex_overload"
        elif np.all(weights <= capacities):
            status = "balanced"
        else:
            status = "balance_not_reached"
        scope = Scope()
        part_weights = parse(
            _frozen(weights), HostInt64[_GraphPartDim], "part_weights", scope=scope
        )
        part_targets = parse(
            _frozen(targets), HostFloat64[_GraphPartDim], "part_targets", scope=scope
        )
        part_capacities = parse(
            _frozen(capacities), HostInt64[_GraphPartDim], "part_capacities", scope=scope
        )
        part_vertex_counts = parse(
            _frozen(np.bincount(parts, minlength=count).astype(np.int64)),
            HostInt64[_GraphPartDim],
            "part_vertex_counts",
            scope=scope,
        )
        indivisible_vertices = parse(
            _frozen(indivisible.astype(np.int64)),
            HostInt64[_GraphIndivisibleDim],
            "indivisible_vertices",
        )
        imbalance = float(np.max(weights / targets))
        imbalance_lower_bound = max(
            1.0, float(np.max(graph.vertex_weights)) / float(np.max(targets))
        )
        edge_cut = int(np.sum(graph.edge_weights[crossing])) // 2
        cut_edge_count = int(np.count_nonzero(crossing)) // 2
        boundary_vertex_count = np.unique(rows[crossing]).size
        empty_part_count = int(np.count_nonzero(part_vertex_counts == 0))
        determinism = _DETERMINISM
        evidence_id = canonical_fingerprint(
            {
                "kind": "graph-partition-evidence",
                "status": status,
                "part_weights": array_tree_fingerprint(weights),
                "part_capacities": array_tree_fingerprint(capacities),
                "edge_cut": edge_cut,
                "backend": backend,
            }
        )
        return cls(
            part_weights=part_weights,
            part_targets=part_targets,
            part_capacities=part_capacities,
            part_vertex_counts=part_vertex_counts,
            indivisible_vertices=indivisible_vertices,
            status=status,
            imbalance=imbalance,
            imbalance_lower_bound=imbalance_lower_bound,
            edge_cut=edge_cut,
            cut_edge_count=cut_edge_count,
            boundary_vertex_count=boundary_vertex_count,
            empty_part_count=empty_part_count,
            work=work,
            determinism=determinism,
            backend=backend,
            evidence_id=evidence_id,
            accepted=accepted,
            resource_status=resource_status,
            migration_vertex_count=migration_vertex_count,
            ghost_entry_count=ghost_entry_count,
        )

    @classmethod
    @checked
    def collective(
        cls,
        graph: WeightedCSRGraph,
        parts: Array,
        measurements: PartitionMeasurements,
        indivisible_ids: Array,
        work: GraphPartitionWork,
        backend: str,
        /,
    ) -> GraphPartitionEvidence:
        """Build evidence from the collective kernel's measured output."""
        if graph.mesh is None or graph.vertex_ids is None:
            raise ValueError(
                "GraphPartitionEvidence.graph must be owner-local; got serial graph"
            )
        return cls.logical(
            graph.graph_id, graph.vertex_ids, parts, measurements,
            indivisible_ids, work, backend,
        )

    @classmethod
    @checked
    def logical(
        cls,
        graph_id: Identifier,
        vertex_ids: Array,
        parts: Array,
        measurements: PartitionMeasurements,
        indivisible_ids: Array,
        work: GraphPartitionWork,
        backend: str,
        /,
    ) -> GraphPartitionEvidence:
        """Measure logical outputs whose actual graph is authenticated by the caller."""
        targets, capacities = measurements.targets, measurements.capacities
        weights, counts = measurements.part_weights, measurements.part_vertex_counts
        count = targets.size
        if (
            parts.dtype != jnp.int32
            or parts.shape != vertex_ids.shape
            or indivisible_ids.dtype != jnp.int64
            or indivisible_ids.shape != parts.shape
        ):
            raise ValueError(
                "GraphPartitionEvidence.parts must align with ID rows using int32 parts and int64 indivisible IDs; got invalid dtype or shape"
            )
        if (
            any(
                value.dtype != jnp.int64 or value.shape != (count,)
                for value in (weights, counts, capacities)
            )
            or targets.dtype != jnp.float64
            or targets.shape != (count,)
        ):
            raise ValueError(
                "GraphPartitionEvidence.measurements must have aligned int64 weights, counts, capacities and float64 targets; got invalid dtype or shape"
            )
        host = _collective_host_measurements(measurements)
        resource, cut, cut_count = host.resource, host.cut, host.cut_count
        boundary, migration, ghosts, maximum = (
            host.boundary,
            host.migration,
            host.ghosts,
            host.maximum,
        )
        host_weights, host_counts, host_targets, host_capacities = (
            host.weights,
            host.counts,
            host.targets,
            host.capacities,
        )
        part_weights = weights
        part_vertex_counts = counts
        part_targets = targets
        part_capacities = capacities
        indivisible_vertices = indivisible_ids
        status: GraphPartitionStatus = (
            "indivisible_vertex_overload"
            if maximum > np.max(host_capacities)
            else "balanced"
            if np.all(host_weights <= host_capacities)
            else "balance_not_reached"
        )
        imbalance = (
            float(np.max(host_weights / host_targets))
            if np.all(host_targets > 0)
            else math.inf
        )
        imbalance_lower_bound = (
            max(1.0, maximum / float(np.max(host_targets)))
            if np.max(host_targets) > 0
            else math.inf
        )
        edge_cut = cut
        cut_edge_count = cut_count
        boundary_vertex_count = boundary
        empty_part_count = int(np.count_nonzero(host_counts == 0))
        determinism = "owner-local multilevel heavy-edge matching; scientific-ID scalar priorities; exact integer gains; capacity-first weighted allocation; bounded gain refinement"
        accepted = resource == 0
        resource_status = resource
        migration_vertex_count = migration
        ghost_entry_count = ghosts
        evidence_id = canonical_fingerprint(
            dict(
                kind="owner-local-graph-partition-evidence",
                graph=graph_id,
                parts=_logical_parts_id(parts),
                resource_status=resource,
                weights=host_weights.tolist(),
                cut=cut,
            )
        )
        return cls(
            part_weights=part_weights,
            part_targets=part_targets,
            part_capacities=part_capacities,
            part_vertex_counts=part_vertex_counts,
            indivisible_vertices=indivisible_vertices,
            status=status,
            imbalance=imbalance,
            imbalance_lower_bound=imbalance_lower_bound,
            edge_cut=edge_cut,
            cut_edge_count=cut_edge_count,
            boundary_vertex_count=boundary_vertex_count,
            empty_part_count=empty_part_count,
            work=work,
            determinism=determinism,
            backend=backend,
            evidence_id=evidence_id,
            accepted=accepted,
            resource_status=resource_status,
            migration_vertex_count=migration_vertex_count,
            ghost_entry_count=ghost_entry_count,
        )


@final
class GraphPartitionResult(StrictModule, NonTrainableState):
    """Part of every vertex of one graph under one plan, with its evidence."""

    __strict_contract__ = True

    parts: HostInt32[_GraphVertexDim] | Int32[AnyShape]
    evidence: GraphPartitionEvidence
    graph_id: Identifier = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)
    result_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        graph: WeightedCSRGraph,
        plan: GraphPartitionPlan,
        parts: np.ndarray | Array,
        evidence: GraphPartitionEvidence,
        /,
    ) -> None:
        self.parts = (
            parts
            if graph.mesh is not None
            else parse(_frozen(parts), HostInt32[_GraphVertexDim], "parts")
        )
        self.evidence = evidence
        self.graph_id = graph.graph_id
        self.plan_id = plan.plan_id
        identity = {
            "kind": "graph-partition-result",
            "graph": graph.graph_id,
            "plan": plan.plan_id,
            "evidence": evidence.evidence_id,
        }
        if graph.mesh is None:
            identity["parts"] = array_tree_fingerprint(self.parts)
        # Collective evidence binds the parts digest, avoiding a second hash.
        self.result_id = canonical_fingerprint(identity)


def _logical_parts_id(parts: Array, /) -> str:

    return logical_id((parts,))


def _partition_owner_local(
    graph: WeightedCSRGraph, plan: GraphPartitionPlan, /
) -> GraphPartitionResult:

    if graph.vertex_ids is None:
        raise RuntimeError(
            "WeightedCSRGraph.vertex_ids must be present for owner-local partitioning; got None"
        )
    if plan.part_count != graph.vertex_ids.shape[0]:
        raise ValueError(
            "GraphPartitionPlan.part_count must match the execution mesh; got mismatched count"
        )
    parts, heavy, measured = execute(graph, plan)
    backend = "xla/compiled-local-shard"
    evidence = GraphPartitionEvidence.collective(
        graph,
        parts,
        measured,
        heavy,
        GraphPartitionWork(
            GraphPartitionCounters.from_slots(
                np.asarray(jax.device_get(measured.counters.slots))
            )
        ),
        backend,
    )
    return GraphPartitionResult(graph, plan, parts, evidence)


def _part_table(
    total: int, plan: GraphPartitionPlan, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real targets, integer targets summing to ``total``, and capacities."""
    shares = np.asarray(plan.part_shares, dtype=np.float64)
    targets = total * shares
    # Rounded cumulative boundaries: integer targets sum exactly to ``total``
    # and each lies within one unit of its real target.
    boundaries = np.minimum(np.floor(total * np.cumsum(shares)), total).astype(np.int64)
    boundaries[-1] = total
    whole = np.diff(boundaries, prepend=0)
    # No part can hold more than the total, so larger capacities are clamped.
    capacities = np.minimum(
        np.maximum(np.floor(plan.maximum_imbalance * targets), np.ceil(targets)), total
    ).astype(np.int64)
    return targets, whole, np.maximum(capacities, whole)


@checked
def partition_graph(
    graph: WeightedCSRGraph, plan: GraphPartitionPlan, /
) -> GraphPartitionResult:
    """Partition ``graph`` by ``plan`` with the native multilevel k-way route.

    Refuses ``require_nonempty`` plans with more parts than vertices. The
    result always carries measured evidence; an unbalanced outcome is reported
    by its status, never hidden or repaired by relaxing the capacities.
    On an owner-local view the result remains aligned to its input scientific
    ID rows, including updated valid ghost copies. Invalid graph content,
    nonempty-policy infeasibility or work exhaustion collectively reject the
    candidate and retain incoming owners; inspect ``evidence.accepted`` before
    using its map for a migration transaction.
    """
    if graph.mesh is not None:
        return _partition_owner_local(graph, plan)
    require_nonempty = plan.empty_parts == "require_nonempty"
    if require_nonempty and plan.part_count > graph.vertex_count:
        raise ValueError(
            "GraphPartitionPlan.part_count must not exceed the vertex count of a nonempty partition; got too many parts"
        )
    offsets, neighbors = graph.offsets, graph.neighbors
    edge_weights, vertex_weights = graph.edge_weights, graph.vertex_weights
    match offsets, neighbors, edge_weights, vertex_weights:
        case np.ndarray(), np.ndarray(), np.ndarray(), np.ndarray():
            pass
        case _:
            raise RuntimeError(
                "WeightedCSRGraph.rows must be prepared host CSR arrays for serial partitioning; got device arrays"
            )
    targets, whole, capacities = _part_table(int(np.sum(graph.vertex_weights)), plan)
    native = graph_partition(
        offsets,
        neighbors,
        edge_weights,
        vertex_weights,
        whole,
        capacities,
        require_nonempty=require_nonempty,
        refinement_passes=plan.refinement_passes,
        work_limit=np.iinfo(np.int64).max if plan.work_limit is None else plan.work_limit,
    )
    parts = native.parts
    counters = native.counters
    evidence = GraphPartitionEvidence.serial(
        graph,
        parts,
        targets,
        capacities,
        GraphPartitionWork(counters),
        meshcore_identity(),
    )
    if require_nonempty and evidence.empty_part_count:
        raise RuntimeError(
            "GraphPartitionResult.part_vertex_counts must be positive for required parts; got an empty part"
        )
    return GraphPartitionResult(graph, plan, parts, evidence)


__all__ = [
    "GraphEmptyPartPolicy",
    "GraphPartitionEvidence",
    "GraphPartitionPlan",
    "GraphPartitionResult",
    "GraphPartitionStatus",
    "GraphPartitionWork",
    "WeightedCSRGraph",
    "partition_graph",
]
