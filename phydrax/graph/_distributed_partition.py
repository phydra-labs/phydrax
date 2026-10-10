# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compiled owner-local CSR partition kernels, with no replicated graph table.

Only fixed-capacity sparse ID/edge queries circulate. Global decisions use
scalar reductions; no vertex candidates or adjacency are all-gathered.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING, TypeAlias

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec

from .._fingerprint import logical_array_value_collection_digest


if TYPE_CHECKING:
    from ._partition import GraphPartitionPlan, WeightedCSRGraph

_CoarseningCarry: TypeAlias = tuple[Array, Array]
_QueryPacket: TypeAlias = tuple[Array, Array, Array, Array, Array]
_AllocationCarry: TypeAlias = tuple[Array, Array, Array, Array, Array]
_RefinementCarry: TypeAlias = tuple[Array, Array, Array, Array]
_MoveCarry: TypeAlias = tuple[Array, Array, Array, Array, Array]
_PartitionMeasurements: TypeAlias = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]
_PartitionOutput: TypeAlias = tuple[Array, Array, _PartitionMeasurements]

SENTINEL = 2**63 - 1
INVALID = 1
RESOURCE = 2


def logical_id(arrays: tuple[Array, ...]) -> str:
    """Use the canonical scientific content owner, not a hash-of-rank-hashes.

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


def _part_operation(
    plan: GraphPartitionPlan,
    axis: str,
    ranks: int,
) -> Callable[[Array, Array, Array, Array, Array, Array, Array], _PartitionOutput]:
    """The identical multilevel mathematical part operation for live or saved rows."""
    ring = tuple((rank, (rank + 1) % ranks) for rank in range(ranks))
    k = plan.part_count
    nonempty = plan.empty_parts == "require_nonempty"
    limit = 2**63 - 1 if plan.work_limit is None else plan.work_limit

    def local(
        offsets: Array,
        neighbors: Array,
        edges: Array,
        weights: Array,
        ids: Array,
        owners: Array,
        valid: Array,
    ) -> _PartitionOutput:
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
            active_edge[edge_order[1:]]
            & jnp.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
        )
        valid_ids = jnp.sort(jnp.where(valid, ids, SENTINEL))
        duplicate |= jnp.any(
            (valid_ids[1:] != SENTINEL) & (valid_ids[1:] == valid_ids[:-1])
        )
        invalid = (
            duplicate
            | duplicate_edges
            | (offsets[0] != 0)
            | jnp.any(offsets[1:] < offsets[:-1])
            | (offsets[-1] > e)
            | jnp.any(
                valid
                & ((ids < 0) | (ids == SENTINEL) | (weights < 0) | (weights > 2**53))
            )
            | jnp.any(valid & ((owners < 0) | (owners >= ranks)))
            | jnp.any(
                active_edge
                & (
                    (neighbors < 0)
                    | (neighbors == ids[row])
                    | (edges < 0)
                    | (edges > 2**53)
                )
            )
        )
        status = jnp.where(invalid, INVALID, 0).astype(jnp.int32)

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

        resolved_owners, unique_count, _ = query(
            jnp.where(valid, ids, SENTINEL), ids, owners
        )
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
        counters = jnp.zeros(11, jnp.int64)
        counters = counters.at[9].set(
            jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64) * ranks, axis)
        )
        counters = counters.at[10].set(
            jax.lax.psum(jnp.sum(valid, dtype=jnp.int64), axis) * ranks * 2
        )
        status |= jnp.where(counters[9] + counters[10] > limit, RESOURCE, 0)
        status = jax.lax.psum((status & INVALID != 0).astype(jnp.int32), axis).astype(
            jnp.bool_
        ).astype(jnp.int32) | (
            jax.lax.psum((status & RESOURCE != 0).astype(jnp.int32), axis)
            .astype(jnp.bool_)
            .astype(jnp.int32)
            << 1
        )

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
            counters = counters.at[2].add(pairs).at[0].add((pairs > 0).astype(jnp.int64))
            counters = (
                counters.at[9]
                .add(jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64), axis))
                .at[10]
                .add(vertices)
            )
            return updated, counters

        def bounded_coarsen(
            index: int | Array, state: _CoarseningCarry
        ) -> _CoarseningCarry:
            return jax.lax.cond(
                (status == 0) & (state[-1][9] + state[-1][10] <= limit),
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
        counters = counters.at[1].set(coarse_count)
        # Nonempty allocation needs enough atoms. Undo matching collectively
        # rather than pretending a coarse atom can fill several parts.
        undo = nonempty & (coarse_count < k)
        leaders = jnp.where(undo, jnp.arange(c), leaders)
        cluster_weights = jnp.where(undo, jnp.where(owned, weights, 0), cluster_weights)
        remaining = jnp.where(undo, owned, remaining)
        parts = jnp.where(owned, -1, owners).astype(jnp.int32)
        part_weights, part_counts = jnp.zeros(k, jnp.int64), jnp.zeros(k, jnp.int64)

        def allocate(_: int | Array, state: _AllocationCarry) -> _AllocationCarry:
            parts, remaining, pw, pc, counters = state
            maximum = jax.lax.pmax(
                jnp.max(jnp.where(remaining, cluster_weights, -1)), axis
            )
            selected = jax.lax.pmin(
                jnp.min(
                    jnp.where(remaining & (cluster_weights == maximum), ids, SENTINEL)
                ),
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
                .add(
                    jnp.where(
                        active_edge & members[row] & (neighbor_parts >= 0), edges, 0
                    )
                )
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
                counters.at[9]
                .add(jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64) * ranks, axis))
                .at[10]
                .add(jnp.where(active, k, 0))
            )
            return parts, remaining, pw, pc, counters

        allocation_steps = jax.lax.psum(jnp.sum(remaining, dtype=jnp.int64), axis)

        def bounded_allocate(
            index: int | Array, state: _AllocationCarry
        ) -> _AllocationCarry:
            return jax.lax.cond(
                (status == 0) & (state[-1][9] + state[-1][10] <= limit),
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

        def refine_pass(_: int | Array, state: _RefinementCarry) -> _RefinementCarry:
            parts, pw, pc, counters = state
            visited = jnp.zeros(c, jnp.bool_)

            def move(_: int | Array, carry: _MoveCarry) -> _MoveCarry:
                parts, pw, pc, visited, counters = carry
                selected = jax.lax.pmin(
                    jnp.min(jnp.where(owned & ~visited, ids, SENTINEL)), axis
                )
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
                destination_overload = jnp.maximum(
                    pw + weight - capacities, 0
                ) - jnp.maximum(pw - capacities, 0)
                balance_gain = reducing - destination_overload
                allowed = (
                    (jnp.arange(k) != source)
                    & ((pw + weight <= capacities) | (balance_gain > 0))
                    & ((not nonempty) | (pc[source] > 1))
                )
                allowed &= (balance_gain > 0) | ((balance_gain == 0) & (gain > 0))
                order = jnp.lexsort(
                    (jnp.arange(k), pw / targets, -gain, -balance_gain, ~allowed)
                )
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
                    counters.at[5]
                    .add(commit.astype(jnp.int64))
                    .at[7]
                    .add((commit & (balance_gain[destination] > 0)).astype(jnp.int64))
                )
                counters = (
                    counters.at[9]
                    .add(
                        jax.lax.psum(jnp.sum(active_edge, dtype=jnp.int64) * ranks, axis)
                    )
                    .at[10]
                    .add(jnp.where(selected != SENTINEL, k, 0))
                )
                return parts, pw, pc, visited, counters

            def bounded_move(index: int | Array, state: _MoveCarry) -> _MoveCarry:
                return jax.lax.cond(
                    (status == 0) & (state[-1][9] + state[-1][10] <= limit),
                    lambda value: move(index, value),
                    lambda value: value,
                    state,
                )

            parts, pw, pc, _, counters = jax.lax.fori_loop(
                jnp.int64(0), vertices, bounded_move, (parts, pw, pc, visited, counters)
            )
            return parts, pw, pc, counters.at[4].add(1)

        def bounded_refine(
            index: int | Array, state: _RefinementCarry
        ) -> _RefinementCarry:
            return jax.lax.cond(
                (status == 0) & (state[-1][9] + state[-1][10] <= limit),
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
        status |= jnp.where(counters[9] + counters[10] > limit, RESOURCE, 0)
        status |= jnp.where(
            (status == 0)
            & (counters[9] + counters[10] <= limit)
            & (jnp.any(owned & (parts < 0)) | (nonempty & jnp.any(part_counts == 0))),
            INVALID,
            0,
        )
        status = jax.lax.psum((status & INVALID != 0).astype(jnp.int32), axis).astype(
            jnp.bool_
        ).astype(jnp.int32) | (
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
        migration = jax.lax.psum(
            jnp.sum(owned & (parts != owners), dtype=jnp.int64), axis
        )
        ghosts = jax.lax.psum(
            jnp.sum(active_edge & (neighbor_parts != parts[row]), dtype=jnp.int64), axis
        )
        heavy = jnp.where(owned & (weights > jnp.max(capacities)), ids, SENTINEL)
        max_weight = jax.lax.pmax(jnp.max(jnp.where(owned, weights, 0)), axis)
        return (
            parts,
            heavy,
            (
                pw,
                pc,
                targets,
                capacities,
                counters,
                status,
                cut,
                cut_count,
                boundary,
                migration,
                ghosts,
                max_weight,
            ),
        )

    return local


def execute(view: WeightedCSRGraph, plan: GraphPartitionPlan) -> _PartitionOutput:
    """Run real physical owner shards with the canonical mathematical operation."""
    if view.mesh is None or view.axis_name is None or view.vertex_ids is None:
        raise ValueError("Distributed execution requires an owner-local graph view.")
    axis, mesh = view.axis_name, view.mesh
    operation = _part_operation(plan, axis, view.vertex_ids.shape[0])
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
    return jax.jit(mapped)(
        view.offsets,
        view.neighbor_ids,
        view.edge_weights,
        view.vertex_weights,
        view.vertex_ids,
        view.vertex_owners,
        view.vertex_valid,
    )


def replay_logical_partition(
    offsets: Array,
    neighbor_ids: Array,
    edge_weights: Array,
    vertex_weights: Array,
    vertex_ids: Array,
    vertex_owners: Array,
    vertex_valid: Array,
    plan: GraphPartitionPlan,
    /,
    *,
    axis_name: str,
) -> _PartitionOutput:
    """Recompute an archived logical graph on actual current devices.

    Named numerical batching is not an old device-mesh stand-in. Matching,
    coarsening, allocation, bounded gain refinement, status and literal work
    measurements are the same operation used by physical execution.
    """
    if vertex_ids.ndim != 2 or not isinstance(axis_name, str) or not axis_name:
        raise ValueError(
            "Logical graph replay requires saved source rows and an explicit batch axis."
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
            "Logical graph replay requires exact canonical CSR/ID/owner/weight axes and dtypes."
        )
    operation = _part_operation(plan, axis_name, ranks)
    mapped = jax.vmap(
        operation, in_axes=(0,) * 7, out_axes=(0, 0, 0), axis_name=axis_name
    )
    parts, heavy, measured = jax.jit(mapped)(
        offsets,
        neighbor_ids,
        edge_weights,
        vertex_weights,
        vertex_ids,
        vertex_owners,
        vertex_valid,
    )
    return (
        parts,
        heavy,
        (
            measured[0][0],
            measured[1][0],
            measured[2][0],
            measured[3][0],
            measured[4][0],
            measured[5][0],
            measured[6][0],
            measured[7][0],
            measured[8][0],
            measured[9][0],
            measured[10][0],
            measured[11][0],
        ),
    )
