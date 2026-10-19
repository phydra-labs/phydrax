# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compiled owner-local CSR partition kernels, with no replicated graph table.

Only fixed-capacity sparse ID/edge queries circulate. Global decisions use
scalar reductions; no vertex candidates or adjacency are all-gathered.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, PartitionSpec
from jax.typing import ArrayLike

from .._fingerprint import logical_array_value_collection_digest
from .._strict import StrictModule
from ..typing import checked


class PartitionPolicy(Protocol):
    """Static partition policy consumed by the numerical kernel."""

    @property
    def part_count(self) -> int: ...

    @property
    def part_shares(self) -> tuple[float, ...]: ...

    @property
    def maximum_imbalance(self) -> float: ...

    @property
    def empty_parts(self) -> str: ...

    @property
    def refinement_passes(self) -> int: ...

    @property
    def work_limit(self) -> int | None: ...


class OwnerCSRView(Protocol):
    """Prepared row arrays and physical execution placement."""

    @property
    def offsets(self) -> ArrayLike: ...

    @property
    def neighbor_ids(self) -> Array: ...

    @property
    def edge_weights(self) -> ArrayLike: ...

    @property
    def vertex_weights(self) -> ArrayLike: ...

    @property
    def vertex_ids(self) -> Array | None: ...

    @property
    def vertex_owners(self) -> Array | None: ...

    @property
    def vertex_valid(self) -> Array | None: ...

    @property
    def mesh(self) -> Mesh | None: ...

    @property
    def axis_name(self) -> str | None: ...


_CoarseningCarry: TypeAlias = tuple[Array, Array]
_QueryPacket: TypeAlias = tuple[Array, Array, Array, Array, Array]
_AllocationCarry: TypeAlias = tuple[Array, Array, Array, Array, Array]
_RefinementCarry: TypeAlias = tuple[Array, Array, Array, Array]
_MoveCarry: TypeAlias = tuple[Array, Array, Array, Array, Array]


class PartitionKernelCounters(StrictModule):
    """Named work slots of the compiled partition route."""

    slots: Array

    @property
    def coarsening_levels(self) -> Array:
        return self.slots[_CounterSlot.COARSENING_LEVELS]

    @property
    def coarsest_vertex_count(self) -> Array:
        return self.slots[_CounterSlot.COARSEST_VERTEX_COUNT]

    @property
    def matched_pairs(self) -> Array:
        return self.slots[_CounterSlot.MATCHED_PAIRS]

    @property
    def bisection_trials(self) -> Array:
        return self.slots[_CounterSlot.BISECTION_TRIALS]

    @property
    def refinement_passes(self) -> Array:
        return self.slots[_CounterSlot.REFINEMENT_PASSES]

    @property
    def moves_committed(self) -> Array:
        return self.slots[_CounterSlot.MOVES_COMMITTED]

    @property
    def moves_rolled_back(self) -> Array:
        return self.slots[_CounterSlot.MOVES_ROLLED_BACK]

    @property
    def balance_moves(self) -> Array:
        return self.slots[_CounterSlot.BALANCE_MOVES]

    @property
    def nonempty_repairs(self) -> Array:
        return self.slots[_CounterSlot.NONEMPTY_REPAIRS]

    @property
    def adjacency_visits(self) -> Array:
        return self.slots[_CounterSlot.ADJACENCY_VISITS]

    @property
    def candidate_evaluations(self) -> Array:
        return self.slots[_CounterSlot.CANDIDATE_EVALUATIONS]


class PartitionMeasurements(StrictModule):
    """Collective measurements aligned to the returned partition."""

    part_weights: Array
    part_vertex_counts: Array
    targets: Array
    capacities: Array
    counters: PartitionKernelCounters
    resource_status: Array
    edge_cut: Array
    cut_edge_count: Array
    boundary_vertex_count: Array
    migration_vertex_count: Array
    ghost_entry_count: Array
    maximum_vertex_weight: Array


class _CounterSlot(IntEnum):
    COARSENING_LEVELS = 0
    COARSEST_VERTEX_COUNT = 1
    MATCHED_PAIRS = 2
    BISECTION_TRIALS = 3
    REFINEMENT_PASSES = 4
    MOVES_COMMITTED = 5
    MOVES_ROLLED_BACK = 6
    BALANCE_MOVES = 7
    NONEMPTY_REPAIRS = 8
    ADJACENCY_VISITS = 9
    CANDIDATE_EVALUATIONS = 10


class _PartitionRows(StrictModule):
    offsets: Array
    neighbors: Array
    edges: Array
    weights: Array
    ids: Array
    owners: Array
    valid: Array
    row: Array
    active_edge: Array
    owned: Array
    id_order: Array
    id_keys: Array
    edge_order: Array
    sorted_keys: Array


class _PreparedPartition(StrictModule):
    rows: _PartitionRows
    status: Array


class _ValidatedPartition(StrictModule):
    status: Array
    vertices: Array
    targets: Array
    capacities: Array
    counters: Array


class _CoarsePartition(StrictModule):
    leaders: Array
    cluster_weights: Array
    remaining: Array
    parts: Array
    part_weights: Array
    part_counts: Array
    counters: Array


class _AllocatedPartition(StrictModule):
    parts: Array
    part_weights: Array
    part_counts: Array
    counters: Array


class _PartitionQuery(Protocol):
    def __call__(
        self, query_ids: Array, source_ids: Array, values: Array, reverse: bool = False
    ) -> tuple[Array, Array, Array]: ...


_PartitionOutput: TypeAlias = tuple[Array, Array, PartitionMeasurements]

SENTINEL = 2**63 - 1
INVALID = 1
RESOURCE = 2


def logical_id(arrays: tuple[Array, ...]) -> str:
    """Return the scientific content digest of logical arrays.

    Identity hashing streams bounded logical byte chunks. It never retains a
    whole host CSR. This explicit identity traffic is separate from the sparse
    partition algorithm and is not a parallel scaling claim.
    """
    return logical_array_value_collection_digest(
        {str(index): value for index, value in enumerate(arrays)},
        maximum_chunk_bytes=65536,
    )


def _lookup(keys: Array, queries: Array) -> tuple[Array, Array]:
    """Lexicographic lower bound without a query-by-table allocation."""
    size = keys.shape[0]
    lo = jnp.zeros(queries.shape[0], jnp.int32)
    hi = jnp.full_like(lo, size)

    def step(_: int | Array, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        low, high = bounds
        mid = (low + high) // 2
        value = keys[jnp.minimum(mid, size - 1)]
        less = (value[:, 0] < queries[:, 0]) | (
            (value[:, 0] == queries[:, 0]) & (value[:, 1] < queries[:, 1])
        )
        return jnp.where((low < high) & less, mid + 1, low), jnp.where(
            (low < high) & ~less, mid, high
        )

    lo, _ = jax.lax.fori_loop(0, size.bit_length() + 1, step, (lo, hi))
    found = (lo < size) & jnp.all(keys[jnp.minimum(lo, size - 1)] == queries, axis=1)
    return jnp.minimum(lo, size - 1), found


def _collective_status(status: Array, axis: str) -> Array:
    return jax.lax.psum((status & INVALID != 0).astype(jnp.int32), axis).astype(
        jnp.bool_
    ).astype(jnp.int32) | (
        jax.lax.psum((status & RESOURCE != 0).astype(jnp.int32), axis)
        .astype(jnp.bool_)
        .astype(jnp.int32)
        << 1
    )


def _prepare_rows(
    offsets: Array,
    neighbors: Array,
    edges: Array,
    weights: Array,
    ids: Array,
    owners: Array,
    valid: Array,
    axis: str,
    ranks: int,
) -> _PreparedPartition:
    rank = jax.lax.axis_index(axis)
    c, e = ids.shape[0], neighbors.shape[0]
    owned = valid & (owners == rank)
    row = jnp.searchsorted(offsets[1:], jnp.arange(e), side="right")
    row = jnp.minimum(row, c - 1)
    active_edge = (jnp.arange(e) < offsets[-1]) & owned[row]
    safe_ids = jnp.where(owned, ids, SENTINEL)
    id_order = jnp.argsort(safe_ids, stable=True)
    id_keys = jnp.stack((safe_ids[id_order], jnp.zeros(c, jnp.int64)), axis=1)
    edge_keys = jnp.stack(
        (
            jnp.where(active_edge, ids[row], SENTINEL),
            jnp.where(active_edge, neighbors, SENTINEL),
        ),
        axis=1,
    )
    edge_order = jnp.lexsort((edge_keys[:, 1], edge_keys[:, 0]))
    sorted_keys = edge_keys[edge_order]
    duplicate = jnp.any(
        owned[id_order[1:]] & (safe_ids[id_order[1:]] == safe_ids[id_order[:-1]])
    )
    duplicate_edges = jnp.any(
        active_edge[edge_order[1:]] & jnp.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    )
    valid_ids = jnp.sort(jnp.where(valid, ids, SENTINEL))
    duplicate |= jnp.any((valid_ids[1:] != SENTINEL) & (valid_ids[1:] == valid_ids[:-1]))
    invalid = (
        duplicate
        | duplicate_edges
        | (offsets[0] != 0)
        | jnp.any(offsets[1:] < offsets[:-1])
        | (offsets[-1] > e)
        | jnp.any(
            valid & ((ids < 0) | (ids == SENTINEL) | (weights < 0) | (weights > 2**53))
        )
        | jnp.any(valid & ((owners < 0) | (owners >= ranks)))
        | jnp.any(
            active_edge
            & ((neighbors < 0) | (neighbors == ids[row]) | (edges < 0) | (edges > 2**53))
        )
    )
    status = jnp.where(invalid, INVALID, 0).astype(jnp.int32)
    return _PreparedPartition(
        rows=_PartitionRows(
            offsets=offsets,
            neighbors=neighbors,
            edges=edges,
            weights=weights,
            ids=ids,
            owners=owners,
            valid=valid,
            row=row,
            active_edge=active_edge,
            owned=owned,
            id_order=id_order,
            id_keys=id_keys,
            edge_order=edge_order,
            sorted_keys=sorted_keys,
        ),
        status=status,
    )


def _query_operation(rows: _PartitionRows, axis: str, ranks: int) -> _PartitionQuery:
    edges = rows.edges
    id_order = rows.id_order
    id_keys = rows.id_keys
    edge_order = rows.edge_order
    sorted_keys = rows.sorted_keys
    ring = tuple((rank, (rank + 1) % ranks) for rank in range(ranks))

    def query(
        query_ids: Array,
        source_ids: Array,
        values: Array,
        reverse: bool = False,
    ) -> tuple[Array, Array, Array]:
        packet = (
            query_ids,
            source_ids,
            jnp.zeros(query_ids.shape, values.dtype),
            jnp.zeros(query_ids.shape, jnp.int32),
            jnp.zeros(query_ids.shape, jnp.int64),
        )

        def hop(_: int | Array, packet: _QueryPacket) -> _QueryPacket:
            target, source, result, found_count, reverse_weight = packet
            pos, found = _lookup(
                id_keys, jnp.stack((target, jnp.zeros_like(target)), axis=1)
            )
            found &= target != SENTINEL
            result = jnp.where(found, values[id_order[pos]], result)
            found_count += found.astype(jnp.int32)
            if reverse:
                ep, ef = _lookup(sorted_keys, jnp.stack((target, source), axis=1))
                reverse_weight = jnp.where(
                    ef & found, edges[edge_order[ep]], reverse_weight
                )
                found_count = jnp.where(found & ~ef, -ranks - 1, found_count)
            return (
                jax.lax.ppermute(target, axis, ring),
                jax.lax.ppermute(source, axis, ring),
                jax.lax.ppermute(result, axis, ring),
                jax.lax.ppermute(found_count, axis, ring),
                jax.lax.ppermute(reverse_weight, axis, ring),
            )

        _, _, result, found, reverse_weight = jax.lax.fori_loop(0, ranks, hop, packet)
        return result, found, reverse_weight

    return query


def _validate_collective(
    rows: _PartitionRows,
    plan: PartitionPolicy,
    query: _PartitionQuery,
    status: Array,
    axis: str,
    ranks: int,
) -> _ValidatedPartition:
    neighbors = rows.neighbors
    edges = rows.edges
    weights = rows.weights
    ids = rows.ids
    owners = rows.owners
    valid = rows.valid
    row = rows.row
    active_edge = rows.active_edge
    owned = rows.owned
    k = plan.part_count
    nonempty = plan.empty_parts == "require_nonempty"
    limit = 2**63 - 1 if plan.work_limit is None else plan.work_limit
    resolved_owners, unique_count, _ = query(jnp.where(valid, ids, SENTINEL), ids, owners)
    resolved_weights, _, _ = query(jnp.where(valid, ids, SENTINEL), ids, weights)
    neighbor_parts, neighbor_count, mirror_weights = query(
        jnp.where(active_edge, neighbors, SENTINEL),
        jnp.where(active_edge, ids[row], SENTINEL),
        owners,
        True,
    )
    status |= jnp.where(
        jnp.any(
            valid
            & (
                (unique_count != 1)
                | (resolved_owners != owners)
                | (resolved_weights != weights)
            )
        )
        | jnp.any(active_edge & ((neighbor_count != 1) | (mirror_weights != edges))),
        INVALID,
        0,
    )
    total = jax.lax.psum(jnp.sum(jnp.where(owned, weights, 0)), axis)
    vertices = jax.lax.psum(jnp.sum(owned, dtype=jnp.int64), axis)

    # Split sums keep invalid oversized totals from wrapping int64 before
    # the collective validity decision.
    def exceeds_weight_limit(values: Array) -> Array:
        high = jax.lax.psum(jnp.sum(values // 2**26), axis)
        low = jax.lax.psum(jnp.sum(values % 2**26), axis)
        high += low // 2**26
        return (high > 2**27) | ((high == 2**27) & (low % 2**26 != 0))

    oversized = exceeds_weight_limit(
        jnp.where(owned, jnp.clip(weights, 0, 2**53), 0)
    ) | exceeds_weight_limit(jnp.where(active_edge, jnp.clip(edges, 0, 2**53), 0))
    status |= jnp.where(
        (total <= 0) | oversized | (nonempty & (vertices < k)), INVALID, 0
    )
    targets = total * jnp.asarray(plan.part_shares, jnp.float64)
    capacities = jnp.minimum(
        jnp.maximum(jnp.floor(plan.maximum_imbalance * targets), jnp.ceil(targets)),
        total,
    ).astype(jnp.int64)
    counters = jnp.zeros(len(_CounterSlot), jnp.int64)
    counters = counters.at[_CounterSlot.ADJACENCY_VISITS].set(
        jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64) * ranks, axis)
    )
    counters = counters.at[_CounterSlot.CANDIDATE_EVALUATIONS].set(
        jax.lax.psum(jnp.sum(valid, dtype=jnp.int64), axis) * ranks * 2
    )
    status |= jnp.where(
        counters[_CounterSlot.ADJACENCY_VISITS]
        + counters[_CounterSlot.CANDIDATE_EVALUATIONS]
        > limit,
        RESOURCE,
        0,
    )
    status = _collective_status(status, axis).astype(jnp.int32) | (
        jax.lax.psum((status & RESOURCE != 0).astype(jnp.int32), axis)
        .astype(jnp.bool_)
        .astype(jnp.int32)
        << 1
    )
    return _ValidatedPartition(
        status=status,
        vertices=vertices,
        targets=targets,
        capacities=capacities,
        counters=counters,
    )


def _coarsen_partition(
    rows: _PartitionRows,
    plan: PartitionPolicy,
    status: Array,
    vertices: Array,
    capacities: Array,
    counters: Array,
    axis: str,
) -> _CoarsePartition:
    neighbors = rows.neighbors
    edges = rows.edges
    weights = rows.weights
    ids = rows.ids
    owners = rows.owners
    row = rows.row
    active_edge = rows.active_edge
    owned = rows.owned
    id_order = rows.id_order
    id_keys = rows.id_keys
    c, e = ids.shape[0], neighbors.shape[0]
    k = plan.part_count
    nonempty = plan.empty_parts == "require_nonempty"
    limit = 2**63 - 1 if plan.work_limit is None else plan.work_limit
    # Independent heavy-edge matchings on each owner. Contracted vertices
    # retain the minimum scientific ID; a second level matches aggregates.
    leaders = jnp.arange(c, dtype=jnp.int32)
    local_neighbor, local_found = _lookup(
        id_keys, jnp.stack((neighbors, jnp.zeros_like(neighbors)), axis=1)
    )
    local_neighbor = id_order[local_neighbor]
    local_found &= active_edge & (neighbors != SENTINEL)

    def coarsen(_: int | Array, state: _CoarseningCarry) -> _CoarseningCarry:
        leaders, counters = state
        cluster_weights = (
            jnp.zeros(c, jnp.int64).at[leaders].add(jnp.where(owned, weights, 0))
        )
        live = owned & (leaders == jnp.arange(c))
        # Aggregate parallel edges in the contracted local CSR.
        left, right = leaders[row], leaders[local_neighbor]
        edge_live = local_found & (left != right)
        key = left.astype(jnp.int64) * c + right
        order = jnp.argsort(jnp.where(edge_live, key, SENTINEL), stable=True)
        ordered_key = key[order]
        start = jnp.concatenate(
            (jnp.ones(1, jnp.bool_), ordered_key[1:] != ordered_key[:-1])
        )
        segment = jnp.cumsum(start, dtype=jnp.int32) - 1
        sums = (
            jnp.zeros(e, jnp.int64)
            .at[segment]
            .add(jnp.where(edge_live[order], edges[order], 0))
        )
        affinity = jnp.where(edge_live[order], sums[segment], -1)
        best_weight = jnp.full(c, -1, jnp.int64).at[left[order]].max(affinity)
        candidate = edge_live[order] & (affinity == best_weight[left[order]])
        best_id = (
            jnp.full(c, SENTINEL, jnp.int64)
            .at[left[order]]
            .min(jnp.where(candidate, ids[right[order]], SENTINEL))
        )
        partner_pos, partner_found = _lookup(
            id_keys, jnp.stack((best_id, jnp.zeros_like(best_id)), axis=1)
        )
        partner = id_order[partner_pos]
        mutual = (
            live
            & partner_found
            & (best_id[partner] == ids)
            & (cluster_weights + cluster_weights[partner] <= jnp.max(capacities))
        )
        matched = jnp.where(
            mutual,
            jnp.where(ids < ids[partner], jnp.arange(c), partner),
            jnp.arange(c),
        ).astype(jnp.int32)
        updated = matched[leaders]
        pairs = jax.lax.psum(
            jnp.sum(mutual & (ids < ids[partner]), dtype=jnp.int64), axis
        )
        counters = (
            counters.at[_CounterSlot.MATCHED_PAIRS]
            .add(pairs)
            .at[_CounterSlot.COARSENING_LEVELS]
            .add((pairs > 0).astype(jnp.int64))
        )
        counters = (
            counters.at[_CounterSlot.ADJACENCY_VISITS]
            .add(jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64), axis))
            .at[_CounterSlot.CANDIDATE_EVALUATIONS]
            .add(vertices)
        )
        return updated, counters

    def bounded_coarsen(index: int | Array, state: _CoarseningCarry) -> _CoarseningCarry:
        return jax.lax.cond(
            (status == 0)
            & (
                state[-1][_CounterSlot.ADJACENCY_VISITS]
                + state[-1][_CounterSlot.CANDIDATE_EVALUATIONS]
                <= limit
            ),
            lambda value: coarsen(index, value),
            lambda value: value,
            state,
        )

    leaders, counters = jax.lax.fori_loop(0, 2, bounded_coarsen, (leaders, counters))
    cluster_weights = (
        jnp.zeros(c, jnp.int64).at[leaders].add(jnp.where(owned, weights, 0))
    )
    remaining = owned & (leaders == jnp.arange(c))
    coarse_count = jax.lax.psum(jnp.sum(remaining, dtype=jnp.int64), axis)
    counters = counters.at[_CounterSlot.COARSEST_VERTEX_COUNT].set(coarse_count)
    # Nonempty allocation needs enough atoms. Undo matching collectively
    # rather than pretending a coarse atom can fill several parts.
    undo = nonempty & (coarse_count < k)
    leaders = jnp.where(undo, jnp.arange(c), leaders)
    cluster_weights = jnp.where(undo, jnp.where(owned, weights, 0), cluster_weights)
    remaining = jnp.where(undo, owned, remaining)
    parts = jnp.where(owned, -1, owners).astype(jnp.int32)
    part_weights, part_counts = jnp.zeros(k, jnp.int64), jnp.zeros(k, jnp.int64)
    return _CoarsePartition(
        leaders=leaders,
        cluster_weights=cluster_weights,
        remaining=remaining,
        parts=parts,
        part_weights=part_weights,
        part_counts=part_counts,
        counters=counters,
    )


def _allocate_partition(
    rows: _PartitionRows,
    plan: PartitionPolicy,
    coarse: _CoarsePartition,
    query: _PartitionQuery,
    status: Array,
    targets: Array,
    capacities: Array,
    axis: str,
    ranks: int,
) -> _AllocatedPartition:
    neighbors = rows.neighbors
    edges = rows.edges
    ids = rows.ids
    row = rows.row
    active_edge = rows.active_edge
    owned = rows.owned
    k = plan.part_count
    nonempty = plan.empty_parts == "require_nonempty"
    limit = 2**63 - 1 if plan.work_limit is None else plan.work_limit
    leaders, cluster_weights, remaining = (
        coarse.leaders,
        coarse.cluster_weights,
        coarse.remaining,
    )
    parts, part_weights, part_counts, counters = (
        coarse.parts,
        coarse.part_weights,
        coarse.part_counts,
        coarse.counters,
    )

    def allocate(_: int | Array, state: _AllocationCarry) -> _AllocationCarry:
        parts, remaining, pw, pc, counters = state
        maximum = jax.lax.pmax(jnp.max(jnp.where(remaining, cluster_weights, -1)), axis)
        selected = jax.lax.pmin(
            jnp.min(jnp.where(remaining & (cluster_weights == maximum), ids, SENTINEL)),
            axis,
        )
        chosen = remaining & (ids == selected)
        members = owned & chosen[leaders]
        neighbor_parts, _, _ = query(
            jnp.where(active_edge, neighbors, SENTINEL), ids[row], parts
        )
        affinities = (
            jnp.zeros(k, jnp.int64)
            .at[jnp.clip(neighbor_parts, 0, k - 1)]
            .add(jnp.where(active_edge & members[row] & (neighbor_parts >= 0), edges, 0))
        )
        affinities = jax.lax.psum(affinities, axis)
        feasible = pw + maximum <= capacities
        empty = pc == 0
        require_empty = nonempty & jnp.any(empty)
        allowed = jnp.where(require_empty, empty, feasible)
        allowed = jnp.where(jnp.any(allowed), allowed, jnp.ones(k, jnp.bool_))
        overload = jnp.maximum(pw + maximum - capacities, 0)
        # Capacity first, then heavy-edge affinity, normalized load, part ID.
        order = jnp.lexsort(
            (jnp.arange(k), (pw + maximum) / targets, -affinities, overload, ~allowed)
        )
        destination = order[0].astype(jnp.int32)
        count = jax.lax.psum(jnp.sum(members, dtype=jnp.int64), axis)
        active = selected != SENTINEL
        parts = jnp.where(members & active, destination, parts)
        remaining &= ~chosen
        pw = pw.at[destination].add(jnp.where(active, maximum, 0))
        pc = pc.at[destination].add(count)
        counters = (
            counters.at[_CounterSlot.ADJACENCY_VISITS]
            .add(jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64) * ranks, axis))
            .at[_CounterSlot.CANDIDATE_EVALUATIONS]
            .add(jnp.where(active, k, 0))
        )
        return parts, remaining, pw, pc, counters

    allocation_steps = jax.lax.psum(jnp.sum(remaining, dtype=jnp.int64), axis)

    def bounded_allocate(index: int | Array, state: _AllocationCarry) -> _AllocationCarry:
        return jax.lax.cond(
            (status == 0)
            & (
                state[-1][_CounterSlot.ADJACENCY_VISITS]
                + state[-1][_CounterSlot.CANDIDATE_EVALUATIONS]
                <= limit
            ),
            lambda value: allocate(index, value),
            lambda value: value,
            state,
        )

    parts, remaining, part_weights, part_counts, counters = jax.lax.fori_loop(
        jnp.int64(0),
        allocation_steps,
        bounded_allocate,
        (parts, remaining, part_weights, part_counts, counters),
    )
    return _AllocatedPartition(
        parts=parts, part_weights=part_weights, part_counts=part_counts, counters=counters
    )


def _move_operation(
    rows: _PartitionRows,
    plan: PartitionPolicy,
    query: _PartitionQuery,
    targets: Array,
    capacities: Array,
    axis: str,
    ranks: int,
) -> Callable[[int | Array, _MoveCarry], _MoveCarry]:
    neighbors, edges, weights, ids = rows.neighbors, rows.edges, rows.weights, rows.ids
    owned, active_edge, row = rows.owned, rows.active_edge, rows.row
    k = plan.part_count
    nonempty = plan.empty_parts == "require_nonempty"

    def move(_: int | Array, carry: _MoveCarry) -> _MoveCarry:
        parts, pw, pc, visited, counters = carry
        selected = jax.lax.pmin(jnp.min(jnp.where(owned & ~visited, ids, SENTINEL)), axis)
        chosen = owned & (ids == selected)
        weight = jax.lax.psum(jnp.sum(jnp.where(chosen, weights, 0)), axis)
        source = jax.lax.pmax(jnp.max(jnp.where(chosen, parts, -1)), axis)
        source = jnp.maximum(source, 0)
        neighbor_parts, _, _ = query(
            jnp.where(active_edge, neighbors, SENTINEL), ids[row], parts
        )
        affinities = (
            jnp.zeros(k, jnp.int64)
            .at[jnp.clip(neighbor_parts, 0, k - 1)]
            .add(jnp.where(active_edge & chosen[row], edges, 0))
        )
        affinities = jax.lax.psum(affinities, axis)
        gain = affinities - affinities[source]
        reducing = jnp.maximum(pw[source] - capacities[source], 0) - jnp.maximum(
            pw[source] - weight - capacities[source], 0
        )
        destination_overload = jnp.maximum(pw + weight - capacities, 0) - jnp.maximum(
            pw - capacities, 0
        )
        balance_gain = reducing - destination_overload
        allowed = (
            (jnp.arange(k) != source)
            & ((pw + weight <= capacities) | (balance_gain > 0))
            & ((not nonempty) | (pc[source] > 1))
        )
        allowed &= (balance_gain > 0) | ((balance_gain == 0) & (gain > 0))
        order = jnp.lexsort((jnp.arange(k), pw / targets, -gain, -balance_gain, ~allowed))
        destination = order[0].astype(jnp.int32)
        commit = (selected != SENTINEL) & allowed[destination]
        parts = jnp.where(chosen & commit, destination, parts)
        pw = (
            pw.at[source]
            .add(jnp.where(commit, -weight, 0))
            .at[destination]
            .add(jnp.where(commit, weight, 0))
        )
        pc = (
            pc.at[source]
            .add(jnp.where(commit, -1, 0))
            .at[destination]
            .add(commit.astype(jnp.int64))
        )
        visited |= chosen
        counters = (
            counters.at[_CounterSlot.MOVES_COMMITTED]
            .add(commit.astype(jnp.int64))
            .at[_CounterSlot.BALANCE_MOVES]
            .add((commit & (balance_gain[destination] > 0)).astype(jnp.int64))
        )
        counters = (
            counters.at[_CounterSlot.ADJACENCY_VISITS]
            .add(jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64) * ranks, axis))
            .at[_CounterSlot.CANDIDATE_EVALUATIONS]
            .add(jnp.where(selected != SENTINEL, k, 0))
        )
        return parts, pw, pc, visited, counters

    return move


def _refine_partition(
    rows: _PartitionRows,
    plan: PartitionPolicy,
    allocated: _AllocatedPartition,
    query: _PartitionQuery,
    status: Array,
    vertices: Array,
    targets: Array,
    capacities: Array,
    axis: str,
    ranks: int,
) -> _AllocatedPartition:
    ids = rows.ids
    c = ids.shape[0]
    limit = 2**63 - 1 if plan.work_limit is None else plan.work_limit
    parts, part_weights, part_counts, counters = (
        allocated.parts,
        allocated.part_weights,
        allocated.part_counts,
        allocated.counters,
    )

    def refine_pass(_: int | Array, state: _RefinementCarry) -> _RefinementCarry:
        parts, pw, pc, counters = state
        visited = jnp.zeros(c, jnp.bool_)

        move = _move_operation(rows, plan, query, targets, capacities, axis, ranks)

        def bounded_move(index: int | Array, state: _MoveCarry) -> _MoveCarry:
            return jax.lax.cond(
                (status == 0)
                & (
                    state[-1][_CounterSlot.ADJACENCY_VISITS]
                    + state[-1][_CounterSlot.CANDIDATE_EVALUATIONS]
                    <= limit
                ),
                lambda value: move(index, value),
                lambda value: value,
                state,
            )

        parts, pw, pc, _, counters = jax.lax.fori_loop(
            jnp.int64(0), vertices, bounded_move, (parts, pw, pc, visited, counters)
        )
        return parts, pw, pc, counters.at[_CounterSlot.REFINEMENT_PASSES].add(1)

    def bounded_refine(index: int | Array, state: _RefinementCarry) -> _RefinementCarry:
        return jax.lax.cond(
            (status == 0)
            & (
                state[-1][_CounterSlot.ADJACENCY_VISITS]
                + state[-1][_CounterSlot.CANDIDATE_EVALUATIONS]
                <= limit
            ),
            lambda value: refine_pass(index, value),
            lambda value: value,
            state,
        )

    parts, part_weights, part_counts, counters = jax.lax.fori_loop(
        0,
        plan.refinement_passes,
        bounded_refine,
        (parts, part_weights, part_counts, counters),
    )
    return _AllocatedPartition(
        parts=parts, part_weights=part_weights, part_counts=part_counts, counters=counters
    )


def _measure_partition(
    rows: _PartitionRows,
    plan: PartitionPolicy,
    refined: _AllocatedPartition,
    query: _PartitionQuery,
    status: Array,
    targets: Array,
    capacities: Array,
    axis: str,
) -> _PartitionOutput:
    neighbors = rows.neighbors
    edges = rows.edges
    weights = rows.weights
    ids = rows.ids
    owners = rows.owners
    valid = rows.valid
    row = rows.row
    active_edge = rows.active_edge
    owned = rows.owned
    c = ids.shape[0]
    k = plan.part_count
    nonempty = plan.empty_parts == "require_nonempty"
    limit = 2**63 - 1 if plan.work_limit is None else plan.work_limit
    parts, part_counts, counters = refined.parts, refined.part_counts, refined.counters
    status |= jnp.where(
        counters[_CounterSlot.ADJACENCY_VISITS]
        + counters[_CounterSlot.CANDIDATE_EVALUATIONS]
        > limit,
        RESOURCE,
        0,
    )
    status |= jnp.where(
        (status == 0)
        & (
            counters[_CounterSlot.ADJACENCY_VISITS]
            + counters[_CounterSlot.CANDIDATE_EVALUATIONS]
            <= limit
        )
        & (jnp.any(owned & (parts < 0)) | (nonempty & jnp.any(part_counts == 0))),
        INVALID,
        0,
    )
    status = _collective_status(status, axis).astype(jnp.int32) | (
        jax.lax.psum((status & RESOURCE != 0).astype(jnp.int32), axis)
        .astype(jnp.bool_)
        .astype(jnp.int32)
        << 1
    )
    accepted = status == 0
    parts = jnp.where(accepted, parts, owners)
    resolved_parts, _, _ = query(jnp.where(valid, ids, SENTINEL), ids, parts)
    parts = jnp.where(accepted & valid, resolved_parts, parts)
    neighbor_parts, _, _ = query(
        jnp.where(active_edge, neighbors, SENTINEL), ids[row], parts
    )
    crossing = active_edge & (parts[row] != neighbor_parts)
    pw = jax.lax.psum(
        jnp.zeros(k, jnp.int64)
        .at[jnp.clip(parts, 0, k - 1)]
        .add(jnp.where(owned, weights, 0)),
        axis,
    )
    pc = jax.lax.psum(
        jnp.zeros(k, jnp.int64)
        .at[jnp.clip(parts, 0, k - 1)]
        .add(owned.astype(jnp.int64)),
        axis,
    )
    cut = jax.lax.psum(jnp.sum(jnp.where(crossing, edges, 0)), axis) // 2
    cut_count = jax.lax.psum(jnp.sum(crossing, dtype=jnp.int64), axis) // 2
    boundary = jax.lax.psum(
        jnp.sum(jnp.zeros(c, jnp.bool_).at[row].max(crossing), dtype=jnp.int64), axis
    )
    migration = jax.lax.psum(jnp.sum(owned & (parts != owners), dtype=jnp.int64), axis)
    ghosts = jax.lax.psum(
        jnp.sum(active_edge & (neighbor_parts != parts[row]), dtype=jnp.int64), axis
    )
    heavy = jnp.where(owned & (weights > jnp.max(capacities)), ids, SENTINEL)
    max_weight = jax.lax.pmax(jnp.max(jnp.where(owned, weights, 0)), axis)
    return (
        parts,
        heavy,
        PartitionMeasurements(
            part_weights=pw,
            part_vertex_counts=pc,
            targets=targets,
            capacities=capacities,
            counters=PartitionKernelCounters(slots=counters),
            resource_status=status,
            edge_cut=cut,
            cut_edge_count=cut_count,
            boundary_vertex_count=boundary,
            migration_vertex_count=migration,
            ghost_entry_count=ghosts,
            maximum_vertex_weight=max_weight,
        ),
    )


def _part_operation(
    plan: PartitionPolicy, axis: str, ranks: int
) -> Callable[[Array, Array, Array, Array, Array, Array, Array], _PartitionOutput]:
    """Prepare the shared compiled operation for live and archived graph rows."""

    def local(
        offsets: Array,
        neighbors: Array,
        edges: Array,
        weights: Array,
        ids: Array,
        owners: Array,
        valid: Array,
    ) -> _PartitionOutput:
        prepared = _prepare_rows(
            offsets, neighbors, edges, weights, ids, owners, valid, axis, ranks
        )
        rows, status = prepared.rows, prepared.status
        query = _query_operation(rows, axis, ranks)
        validated = _validate_collective(rows, plan, query, status, axis, ranks)
        status, vertices = validated.status, validated.vertices
        targets, capacities, counters = (
            validated.targets,
            validated.capacities,
            validated.counters,
        )
        coarse = _coarsen_partition(
            rows, plan, status, vertices, capacities, counters, axis
        )
        allocated = _allocate_partition(
            rows, plan, coarse, query, status, targets, capacities, axis, ranks
        )
        refined = _refine_partition(
            rows,
            plan,
            allocated,
            query,
            status,
            vertices,
            targets,
            capacities,
            axis,
            ranks,
        )
        return _measure_partition(
            rows, plan, refined, query, status, targets, capacities, axis
        )

    return local


@eqx.filter_jit
@checked
def execute(view: OwnerCSRView, plan: PartitionPolicy) -> _PartitionOutput:
    """Partition physical owner shards with the shared mathematical operation."""
    mesh, axis = view.mesh, view.axis_name
    ids, owners, valid = view.vertex_ids, view.vertex_owners, view.vertex_valid
    if mesh is None or axis is None or ids is None or owners is None or valid is None:
        raise ValueError(
            "WeightedCSRGraph.mesh must bind an owner-local execution view; got missing mesh, axis, IDs, owners or validity"
        )
    offsets = jnp.asarray(view.offsets)
    edges = jnp.asarray(view.edge_weights)
    weights = jnp.asarray(view.vertex_weights)
    operation = _part_operation(plan, axis, ids.shape[0])
    spec = PartitionSpec(axis)

    def local(
        offset_block: Array,
        neighbor_block: Array,
        edge_block: Array,
        weight_block: Array,
        id_block: Array,
        owner_block: Array,
        valid_block: Array,
    ) -> _PartitionOutput:
        parts, heavy, measured = operation(
            offset_block[0],
            neighbor_block[0],
            edge_block[0],
            weight_block[0],
            id_block[0],
            owner_block[0],
            valid_block[0],
        )
        return parts[None], heavy[None], measured

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(spec,) * 7,
        out_specs=(spec, spec, PartitionSpec()),
        check_vma=False,
    )
    return mapped(
        offsets,
        view.neighbor_ids,
        edges,
        weights,
        ids,
        owners,
        valid,
    )


@eqx.filter_jit
@checked
def replay_logical_partition(
    offsets: Array,
    neighbor_ids: Array,
    edge_weights: Array,
    vertex_weights: Array,
    vertex_ids: Array,
    vertex_owners: Array,
    vertex_valid: Array,
    plan: PartitionPolicy,
    /,
    *,
    axis_name: str,
) -> _PartitionOutput:
    """Recompute an archived logical graph on the current devices.

    Named numerical batching is not an old device-mesh stand-in. Matching,
    coarsening, allocation, bounded gain refinement, status and literal work
    measurements are the same operation used by physical execution.
    """
    if vertex_ids.ndim != 2 or not isinstance(axis_name, str) or not axis_name:
        raise ValueError(
            "GraphPartitionReplay.rows must be rank-2 with an explicit batch axis; got invalid rows or axis"
        )
    ranks, capacity = vertex_ids.shape
    if (
        ranks < 1
        or capacity < 1
        or offsets.shape != (ranks, capacity + 1)
        or neighbor_ids.ndim != 2
        or neighbor_ids.shape[0] != ranks
        or neighbor_ids.shape[1] < 1
        or edge_weights.shape != neighbor_ids.shape
        or vertex_weights.shape != vertex_ids.shape
        or vertex_owners.shape != vertex_ids.shape
        or vertex_valid.shape != vertex_ids.shape
        or any(
            value.dtype != jnp.int64
            for value in (offsets, neighbor_ids, edge_weights, vertex_weights, vertex_ids)
        )
        or vertex_owners.dtype != jnp.int32
        or vertex_valid.dtype != jnp.bool_
    ):
        raise ValueError(
            "GraphPartitionReplay.rows must align CSR, ID, owner and weight axes with int64 data, int32 owners and bool validity; got invalid shapes or dtypes"
        )
    operation = _part_operation(plan, axis_name, ranks)
    mapped = jax.vmap(
        operation, in_axes=(0,) * 7, out_axes=(0, 0, 0), axis_name=axis_name
    )
    parts, heavy, measured = mapped(
        offsets,
        neighbor_ids,
        edge_weights,
        vertex_weights,
        vertex_ids,
        vertex_owners,
        vertex_valid,
    )
    return parts, heavy, jax.tree.map(lambda value: value[0], measured)
