#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic bounded migration of native owner-local simplex packets.

The generic WeightedCSRGraph owner partitions the actual dual-facet graph.
This module lowers adjacency and consumes graph-produced target owners in
canonical semantic-ID row order; it does not own another partition algorithm.
Capacity-sized packets circulate without gathering geometry, CSR or candidates.

Complete raw bisection forests travel as canonical AdaptiveSimplexState leaves.
Stable scientific IDs replace slot references in flight; family execution
ownership is reconstructed separately from solver graph ownership.
After acceptance, solvers must rebind their ownership, closure and field views.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
)
from ..graph._partition import GraphPartitionPlan


if TYPE_CHECKING:
    from ..graph._partition import WeightedCSRGraph
    from ._device_adaptation import PartitionedAdaptiveSimplex
    from ._distribution import MeshPartitionPolicy, SimplexNeighborhoodWorkset


_SENTINEL = jnp.iinfo(jnp.int64).max
_CAPACITY = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
_INVALID = int(AdaptiveSimplexStatus.INVALID_GEOMETRY)


class OwnerLocalMigrationState(StrictModule, NonTrainableState):
    """Canonical cell packets with aligned numerical content and epoch IDs.

    Scientific identifiers/epochs are int64, owners are int32. Vertex fields
    and vertex owners have one row per cell corner, including shared copies.
    """

    workset: SimplexNeighborhoodWorkset
    cell_fields: Array
    vertex_fields: Array
    cell_epochs: Array
    vertex_owners: Array
    forest: AdaptiveSimplexState | None = None
    forest_cell_owners: Array | None = None
    forest_vertex_owners: Array | None = None
    cell_history: tuple[Array, ...] = ()
    vertex_history: tuple[Array, ...] = ()


class OwnerLocalMigrationEvidence(StrictModule, NonTrainableState):
    """Measured collective resource/atomic decision, never an asserted report."""

    status: Array
    accepted: Array
    owned_before: Array
    owned_after: Array
    resident_after: Array
    vertices_after: Array
    sent_cells: Array
    received_cells: Array
    forest_cells_before: Array
    forest_cells_after: Array
    forest_vertices_before: Array
    forest_vertices_after: Array
    halo_width: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    forest_cell_capacity: int = eqx.field(static=True)
    forest_vertex_capacity: int = eqx.field(static=True)
    route_metadata_bytes: int = eqx.field(static=True)


class PreparedOwnerLocalMigration(StrictModule, NonTrainableState):
    """A staged transaction; only accept_ownerlocal_migration publishes it.

    ``target_parts`` describes measured proposed ownership adjacency. Consumers
    must bind it, rebuild solver halos and lifecycle views only after collective
    acceptance; rejection retains the original execution binding and arrays.
    """

    original: OwnerLocalMigrationState
    candidate: OwnerLocalMigrationState
    evidence: OwnerLocalMigrationEvidence
    target_parts: AdaptiveSimplexParts
    forest_parts: AdaptiveSimplexParts | None = None


def _collective_status(status: Array, axis: str) -> Array:
    shifts = jnp.arange(8, dtype=jnp.int32)
    bits = (status >> shifts) & 1
    return jnp.sum(
        (jax.lax.psum(bits, axis) > 0).astype(jnp.int32) << shifts, dtype=jnp.int32
    )


_WORKSET_FIELDS = (
    "cell_ids",
    "cell_vertices",
    "cell_coordinates",
    "cell_owner",
    "cell_classes",
    "cell_exterior",
    "cell_valid",
    "root_cell_ids",
    "source_cell_ids",
    "source_vertex_ids",
    "source_weights",
    "status",
)


def _replace_work(
    work: SimplexNeighborhoodWorkset, **changes: Array
) -> SimplexNeighborhoodWorkset:
    # The distribution owner imports this module's restart consumer; bind its
    # workset type at call time so either module can be imported first.
    from ._distribution import SimplexNeighborhoodWorkset

    return SimplexNeighborhoodWorkset(
        *(changes.get(name, getattr(work, name)) for name in _WORKSET_FIELDS)
    )


def _migrate_owned_packets(
    parts: AdaptiveSimplexParts,
    workset: SimplexNeighborhoodWorkset,
    targets: Array,
    payloads: tuple[Array, ...],
) -> tuple[SimplexNeighborhoodWorkset, tuple[Array, ...], Array, Array, Array]:
    """Transport geometry and only the actual supplied cell-aligned payloads."""
    from ._distribution import SimplexNeighborhoodWorkset

    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(
        block: SimplexNeighborhoodWorkset,
        target_block: Array,
        payload_block: tuple[Array, ...],
    ) -> tuple[SimplexNeighborhoodWorkset, tuple[Array, ...], Array, Array, Array]:
        work, values = jax.tree_util.tree_map(
            lambda value: value[0], (block, payload_block)
        )
        rank = jax.lax.axis_index(axis)
        target = target_block[0]
        owned = work.cell_valid & (work.cell_owner == rank)
        invalid = jnp.any(owned & ((target < 0) | (target >= parts.part_count)))
        status = work.status | jnp.where(invalid, _INVALID, 0)
        packet = _replace_work(work, cell_valid=owned, cell_owner=target, status=status)
        resident = _replace_work(
            work,
            cell_valid=jnp.zeros_like(owned),
            cell_ids=jnp.full_like(work.cell_ids, _SENTINEL),
            status=status,
        )
        capacity = work.cell_ids.shape[0]

        def merge_step(
            _: int,
            carry: tuple[
                SimplexNeighborhoodWorkset,
                tuple[Array, ...],
                SimplexNeighborhoodWorkset,
                tuple[Array, ...],
            ],
        ) -> tuple[
            SimplexNeighborhoodWorkset,
            tuple[Array, ...],
            SimplexNeighborhoodWorkset,
            tuple[Array, ...],
        ]:
            incoming, incoming_values, current, current_values = carry
            active = incoming.cell_valid & (incoming.cell_owner == rank)
            keys = jnp.concatenate(
                (current.cell_ids, jnp.where(active, incoming.cell_ids, _SENTINEL))
            )
            order = jnp.argsort(keys, stable=True)
            sorted_keys = keys[order]
            duplicate = jnp.any(
                (sorted_keys[1:] == sorted_keys[:-1]) & (sorted_keys[1:] != _SENTINEL)
            )
            count = jnp.sum(current.cell_valid, dtype=jnp.int32) + jnp.sum(
                active, dtype=jnp.int32
            )
            selected = order[:capacity]

            def merge(left: Array, right: Array) -> Array:
                return jnp.concatenate((left, right), axis=0)[selected]

            merged = SimplexNeighborhoodWorkset(
                cell_ids=sorted_keys[:capacity],
                cell_vertices=merge(current.cell_vertices, incoming.cell_vertices),
                cell_coordinates=merge(
                    current.cell_coordinates, incoming.cell_coordinates
                ),
                cell_owner=merge(current.cell_owner, incoming.cell_owner),
                cell_classes=merge(current.cell_classes, incoming.cell_classes),
                cell_exterior=merge(current.cell_exterior, incoming.cell_exterior),
                cell_valid=jnp.arange(capacity) < count,
                root_cell_ids=merge(current.root_cell_ids, incoming.root_cell_ids),
                source_cell_ids=merge(current.source_cell_ids, incoming.source_cell_ids),
                source_vertex_ids=merge(
                    current.source_vertex_ids, incoming.source_vertex_ids
                ),
                source_weights=merge(current.source_weights, incoming.source_weights),
                status=current.status
                | incoming.status
                | jnp.where(count > capacity, _CAPACITY, 0)
                | jnp.where(duplicate, _INVALID, 0),
            )
            merged_values = tuple(
                merge(left, right)
                for left, right in zip(current_values, incoming_values, strict=True)
            )
            incoming, incoming_values = jax.tree_util.tree_map(
                lambda value: jax.lax.ppermute(value, axis, ring),
                (incoming, incoming_values),
            )
            return incoming, incoming_values, merged, merged_values

        _, _, candidate, candidate_values = jax.lax.fori_loop(
            0,
            parts.part_count,
            merge_step,
            (packet, values, resident, values),
        )
        before = jnp.sum(owned, dtype=jnp.int32)
        after = jnp.sum(candidate.cell_valid, dtype=jnp.int32)
        retained = jnp.sum(owned & (target == rank), dtype=jnp.int32)
        status = candidate.status | jnp.where(
            jax.lax.psum(before, axis) != jax.lax.psum(after, axis), _INVALID, 0
        )
        candidate = _replace_work(candidate, status=_collective_status(status, axis))
        return jax.tree_util.tree_map(
            lambda value: value[None],
            (candidate, candidate_values, before, before - retained, after - retained),
        )

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec),
        out_specs=(spec, spec, spec, spec, spec),
        check_vma=False,
    )(workset, targets, payloads)


_compiled_owned_packets = eqx.filter_jit(_migrate_owned_packets)


def _owned_migration(
    parts: AdaptiveSimplexParts,
    original: OwnerLocalMigrationState,
    targets: Array,
) -> tuple[OwnerLocalMigrationState, Array, Array, Array]:
    work, payloads, before, sent, received = _migrate_owned_packets(
        parts,
        original.workset,
        targets,
        (
            original.cell_fields,
            original.vertex_fields,
            original.cell_epochs,
            original.vertex_owners,
        ),
    )
    fields, vertices, epochs, owners = payloads
    return (
        OwnerLocalMigrationState(work, fields, vertices, epochs, owners),
        before,
        sent,
        received,
    )


_compiled_owned_migration = eqx.filter_jit(_owned_migration)


def _hydrate_packet_closure(
    parts: AdaptiveSimplexParts,
    owned: SimplexNeighborhoodWorkset,
    payloads: tuple[Array, ...],
    closure: SimplexNeighborhoodWorkset,
    corner_payloads: tuple[int, ...] = (),
) -> tuple[SimplexNeighborhoodWorkset, tuple[Array, ...], Array, Array]:
    """Hydrate exact-ID geometry and actual supplied payloads into a solver halo."""
    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(
        owner_block: SimplexNeighborhoodWorkset,
        payload_block: tuple[Array, ...],
        closure_block: SimplexNeighborhoodWorkset,
    ) -> tuple[SimplexNeighborhoodWorkset, tuple[Array, ...], Array, Array]:
        source, source_values, work = jax.tree_util.tree_map(
            lambda value: value[0], (owner_block, payload_block, closure_block)
        )
        capacity, width = work.cell_vertices.shape
        ids = work.cell_vertices.reshape(-1)
        valid = jnp.repeat(work.cell_valid, width)
        vertex_owners = jnp.full(ids.shape, parts.part_count, dtype=jnp.int32)
        values = tuple(jnp.zeros_like(value) for value in source_values)
        covered = jnp.zeros_like(work.cell_valid)
        status = work.status | source.status
        coords = work.cell_coordinates.reshape((-1, work.cell_coordinates.shape[-1]))

        def hydrate(
            _: int,
            carry: tuple[
                SimplexNeighborhoodWorkset,
                tuple[Array, ...],
                tuple[Array, ...],
                Array,
                Array,
                Array,
            ],
        ) -> tuple[
            SimplexNeighborhoodWorkset,
            tuple[Array, ...],
            tuple[Array, ...],
            Array,
            Array,
            Array,
        ]:
            packet, packet_values, values, covered, vertex_owners, status = carry
            position = jnp.minimum(
                jnp.searchsorted(packet.cell_ids, work.cell_ids), capacity - 1
            )
            matched = (
                work.cell_valid
                & packet.cell_valid[position]
                & (packet.cell_ids[position] == work.cell_ids)
            )

            def select(old: Array, new: Array) -> Array:
                mask = matched.reshape((capacity,) + (1,) * (old.ndim - 1))
                return jnp.where(mask, new[position], old)

            values = tuple(
                select(old, new) for old, new in zip(values, packet_values, strict=True)
            )
            covered |= matched
            packet_ids = jnp.where(
                packet.cell_valid[:, None], packet.cell_vertices, _SENTINEL
            ).reshape(-1)
            order = jnp.argsort(packet_ids, stable=True)
            sorted_ids = packet_ids[order]
            vpos = jnp.minimum(jnp.searchsorted(sorted_ids, ids), sorted_ids.shape[0] - 1)
            vmatch = valid & (sorted_ids[vpos] == ids)
            lanes = order[vpos]
            packet_coords = packet.cell_coordinates.reshape(coords.shape)[lanes]
            conflict = jnp.any(vmatch & jnp.any(coords != packet_coords, axis=1))
            owners = jnp.repeat(packet.cell_owner, width)[lanes]
            vertex_owners = jnp.minimum(
                vertex_owners, jnp.where(vmatch, owners, parts.part_count)
            )
            status |= packet.status | jnp.where(conflict, _INVALID, 0)
            packet, packet_values = jax.tree_util.tree_map(
                lambda value: jax.lax.ppermute(value, axis, ring),
                (packet, packet_values),
            )
            return packet, packet_values, values, covered, vertex_owners, status

        _, _, values, covered, vertex_owners, status = jax.lax.fori_loop(
            0,
            parts.part_count,
            hydrate,
            (source, source_values, values, covered, vertex_owners, status),
        )
        status |= jnp.where(jnp.any(work.cell_valid & ~covered), _INVALID, 0)
        if corner_payloads:
            flat = tuple(
                values[index].reshape((capacity * width, -1)) for index in corner_payloads
            )

            def check_fields(
                _: int,
                carry: tuple[SimplexNeighborhoodWorkset, tuple[Array, ...], Array],
            ) -> tuple[SimplexNeighborhoodWorkset, tuple[Array, ...], Array]:
                packet, packet_values, status = carry
                keys = jnp.where(
                    packet.cell_valid[:, None], packet.cell_vertices, _SENTINEL
                ).reshape(-1)
                order = jnp.argsort(keys, stable=True)
                keys = keys[order]
                pos = jnp.minimum(jnp.searchsorted(keys, ids), keys.shape[0] - 1)
                match = valid & (keys[pos] == ids)
                for original, index in zip(flat, corner_payloads, strict=True):
                    other = packet_values[index].reshape((capacity * width, -1))[
                        order[pos]
                    ]
                    status |= jnp.where(
                        jnp.any(match & jnp.any(original != other, axis=1)), _INVALID, 0
                    )
                packet, packet_values = jax.tree_util.tree_map(
                    lambda value: jax.lax.ppermute(value, axis, ring),
                    (packet, packet_values),
                )
                return packet, packet_values, status

            _, _, status = jax.lax.fori_loop(
                0, parts.part_count, check_fields, (source, source_values, status)
            )
        sorted_ids = jnp.sort(jnp.where(valid, ids, _SENTINEL))
        fresh = (sorted_ids != _SENTINEL) & jnp.concatenate(
            (jnp.ones((1,), dtype=jnp.bool_), sorted_ids[1:] != sorted_ids[:-1])
        )
        vertices = jnp.sum(fresh, dtype=jnp.int32)
        return jax.tree_util.tree_map(
            lambda value: value[None],
            (
                _replace_work(work, status=status),
                values,
                vertex_owners.reshape((capacity, width)),
                vertices,
            ),
        )

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec),
        out_specs=(spec, spec, spec, spec),
        check_vma=False,
    )(owned, payloads, closure)


_compiled_hydrate_packets = eqx.filter_jit(_hydrate_packet_closure)


def _hydrate_closure(
    parts: AdaptiveSimplexParts,
    owned: OwnerLocalMigrationState,
    closure: SimplexNeighborhoodWorkset,
) -> tuple[OwnerLocalMigrationState, Array]:
    work, payloads, owners, vertices = _hydrate_packet_closure(
        parts,
        owned.workset,
        (owned.cell_fields, owned.vertex_fields, owned.cell_epochs),
        closure,
        (1,),
    )
    fields, vertex_fields, epochs = payloads
    result = eqx.tree_at(
        lambda state: (
            state.workset,
            state.cell_fields,
            state.vertex_fields,
            state.cell_epochs,
            state.vertex_owners,
        ),
        owned,
        (work, fields, vertex_fields, epochs, owners),
    )
    return result, vertices


_compiled_hydrate_closure = eqx.filter_jit(_hydrate_closure)


def _migration_routes(
    parts: AdaptiveSimplexParts, workset: SimplexNeighborhoodWorkset
) -> Array:
    """Replicated bounded routing metadata, derived without gathering geometry."""
    axis = parts.axis_name
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(block: SimplexNeighborhoodWorkset) -> Array:
        work = jax.tree_util.tree_map(lambda value: value[0], block)
        rank = jax.lax.axis_index(axis)
        ids = jnp.where(work.cell_valid[:, None], work.cell_vertices, _SENTINEL).reshape(
            -1
        )
        neighbors = jnp.zeros((parts.part_count,), dtype=jnp.int32)

        def visit(
            _: int,
            carry: tuple[SimplexNeighborhoodWorkset, Array, Array],
        ) -> tuple[SimplexNeighborhoodWorkset, Array, Array]:
            packet, source_rank, neighbors = carry
            keys = jnp.sort(
                jnp.where(
                    packet.cell_valid[:, None], packet.cell_vertices, _SENTINEL
                ).reshape(-1)
            )
            pos = jnp.minimum(jnp.searchsorted(keys, ids), keys.shape[0] - 1)
            shared = jnp.any((ids != _SENTINEL) & (keys[pos] == ids)) & (
                source_rank != rank
            )
            neighbors = neighbors.at[source_rank].set(shared.astype(jnp.int32))
            packet = jax.tree_util.tree_map(
                lambda value: jax.lax.ppermute(value, axis, ring), packet
            )
            source_rank = jax.lax.ppermute(source_rank, axis, ring)
            return packet, source_rank, neighbors

        _, _, neighbors = jax.lax.fori_loop(
            0, parts.part_count, visit, (work, rank, neighbors)
        )
        matrix = (
            jnp.zeros((parts.part_count, parts.part_count), dtype=jnp.int32)
            .at[rank]
            .set(neighbors)
        )
        return jax.lax.psum(matrix, axis) > 0

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(PartitionSpec(axis),),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(workset)


_compiled_migration_routes = eqx.filter_jit(_migration_routes)


def _simplex_graph_part(
    work: SimplexNeighborhoodWorkset,
    weights: Array,
    axis: str,
    part_count: int,
) -> tuple[Array, Array, Array, Array]:
    from ._distribution import _lexicographic_bound, _neighborhood_facet_keys

    ring = tuple((rank, (rank + 1) % part_count) for rank in range(part_count))
    rank = jax.lax.axis_index(axis)
    owned = work.cell_valid & (work.cell_owner == rank)
    source = _replace_work(work, cell_valid=owned)
    keys = _neighborhood_facet_keys(source)
    capacity, width = work.cell_vertices.shape
    self_ids = jnp.repeat(work.cell_ids, width)
    neighbors = jnp.full((capacity * width,), _SENTINEL, jnp.int64)

    def visit(
        _: int, carry: tuple[SimplexNeighborhoodWorkset, Array]
    ) -> tuple[SimplexNeighborhoodWorkset, Array]:
        packet, neighbors = carry
        facets = _neighborhood_facet_keys(packet)
        order = jnp.lexsort(
            tuple(facets[:, column] for column in range(width - 2, -1, -1))
        )
        facets = facets[order]
        ids = jnp.repeat(packet.cell_ids, width)[order]
        lower = jnp.minimum(
            _lexicographic_bound(facets, keys, False), facets.shape[0] - 1
        )
        upper = jnp.maximum(_lexicographic_bound(facets, keys, True) - 1, 0)
        candidate = jnp.where(ids[lower] != self_ids, lower, upper)
        matched = (
            jnp.repeat(owned, width)
            & jnp.all(facets[candidate] == keys, axis=1)
            & (ids[candidate] != self_ids)
        )
        neighbors = jnp.minimum(neighbors, jnp.where(matched, ids[candidate], _SENTINEL))
        return jax.tree_util.tree_map(
            lambda value: jax.lax.ppermute(value, axis, ring), packet
        ), neighbors

    _, neighbors = jax.lax.fori_loop(0, part_count, visit, (source, neighbors))
    present = neighbors != _SENTINEL
    counts = jnp.sum(present.reshape((capacity, width)), axis=1, dtype=jnp.int64)
    offsets = jnp.concatenate(
        (jnp.zeros(1, jnp.int64), jnp.cumsum(counts, dtype=jnp.int64))
    )
    lanes = jnp.nonzero(present, size=capacity * width, fill_value=0)[0]
    valid_entries = jnp.arange(capacity * width) < offsets[-1]
    return (
        offsets,
        jnp.where(valid_entries, neighbors[lanes], -1),
        jnp.where(valid_entries, weights.reshape(-1)[lanes], 0),
        owned,
    )


@eqx.filter_jit
def _simplex_graph_packets(
    parts: AdaptiveSimplexParts,
    workset: SimplexNeighborhoodWorkset,
    edge_weights: Array,
) -> tuple[Array, Array, Array, Array]:
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def local(
        block: SimplexNeighborhoodWorkset, weight_block: Array
    ) -> tuple[Array, Array, Array, Array]:
        work = jax.tree_util.tree_map(lambda value: value[0], block)
        output = _simplex_graph_part(work, weight_block[0], axis, parts.part_count)
        return jax.tree_util.tree_map(lambda value: value[None], output)

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec),
        out_specs=(spec, spec, spec, spec),
        check_vma=False,
    )(workset, edge_weights)


@eqx.filter_jit
def replay_logical_simplex_graph(
    workset: SimplexNeighborhoodWorkset,
    edge_weights: Array,
    /,
    *,
    axis_name: str,
) -> tuple[Array, Array, Array, Array]:
    """Recompute actual dual-facet CSR on saved logical parts without old devices."""
    if (
        workset.cell_ids.ndim != 2
        or edge_weights.shape != workset.cell_vertices.shape
        or edge_weights.dtype != jnp.int64
        or not isinstance(axis_name, str)
        or not axis_name
    ):
        raise ValueError(
            "Logical dual-graph replay requires canonical saved geometry/facet-weight rows."
        )
    count = workset.cell_ids.shape[0]

    def local(
        work: SimplexNeighborhoodWorkset, weights: Array
    ) -> tuple[Array, Array, Array, Array]:
        return _simplex_graph_part(work, weights, axis_name, count)

    return jax.vmap(local, in_axes=(0, 0), out_axes=(0, 0, 0, 0), axis_name=axis_name)(
        workset, edge_weights
    )


def ownerlocal_simplex_graph(
    parts: AdaptiveSimplexParts,
    state: OwnerLocalMigrationState,
    /,
    *,
    vertex_weights: Array,
    edge_weights: Array | None = None,
) -> WeightedCSRGraph:
    """Lower the actual owner-local dual facet graph to the canonical graph owner."""
    from ..graph._partition import WeightedCSRGraph

    shape = state.workset.cell_ids.shape
    edge_shape = state.workset.cell_vertices.shape
    if vertex_weights.shape != shape or vertex_weights.dtype != jnp.int64:
        raise ValueError(
            "Graph cell work weights must be int64 and aligned with scientific cell-ID rows."
        )
    if edge_weights is None:
        edge_weights = jnp.ones(edge_shape, jnp.int64)
    if edge_weights.shape != edge_shape or edge_weights.dtype != jnp.int64:
        raise ValueError(
            "Graph facet communication weights must be int64 and aligned with cell corners."
        )
    offsets, neighbors, weights, valid = _simplex_graph_packets(
        parts, state.workset, edge_weights
    )
    return WeightedCSRGraph.owner_local(
        offsets,
        neighbors,
        state.workset.cell_ids,
        state.workset.cell_owner,
        valid,
        mesh=parts.mesh,
        axis_name=parts.axis_name,
        edge_weights=weights,
        vertex_weights=vertex_weights,
    )


def _prepare_migration_routes(
    parts: AdaptiveSimplexParts, route_mask: Array
) -> AdaptiveSimplexParts:
    """Validate the sole bounded replicated metadata read at the prepare barrier."""
    if route_mask.is_fully_addressable:
        mask = np.asarray(jax.device_get(route_mask), dtype=np.bool_)
    else:
        from jax.experimental import multihost_utils

        mask = np.asarray(
            multihost_utils.process_allgather(route_mask, tiled=True), dtype=np.bool_
        )
    if mask.shape != (parts.part_count, parts.part_count) or not np.array_equal(
        mask, mask.T
    ):
        raise ValueError(
            "Migration route metadata must be a single reciprocal global rank matrix."
        )
    routes = tuple((int(source), int(target)) for source, target in np.argwhere(mask))
    return AdaptiveSimplexParts(
        tuple(parts.mesh.devices.flat), axis_name=parts.axis_name, neighbor_pairs=routes
    )


@eqx.filter_jit
def _forest_migration_routes(
    parts: AdaptiveSimplexParts, forest: AdaptiveSimplexState
) -> Array:
    axis = parts.axis_name
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(block: Array) -> Array:
        ids = block[0]
        keys = jnp.where(ids >= 0, ids, _SENTINEL)
        rank = jax.lax.axis_index(axis)
        neighbors = jnp.zeros(parts.part_count, jnp.int32)

        def visit(
            _: int, carry: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            packet, source_rank, neighbors = carry
            pos = jnp.minimum(jnp.searchsorted(packet, keys), packet.shape[0] - 1)
            shared = jnp.any((keys != _SENTINEL) & (packet[pos] == keys)) & (
                source_rank != rank
            )
            neighbors = neighbors.at[source_rank].set(shared.astype(jnp.int32))
            return (
                jax.lax.ppermute(packet, axis, ring),
                jax.lax.ppermute(source_rank, axis, ring),
                neighbors,
            )

        _, _, neighbors = jax.lax.fori_loop(
            0, parts.part_count, visit, (keys, rank, neighbors)
        )
        return (
            jax.lax.psum(
                jnp.zeros((parts.part_count, parts.part_count), jnp.int32)
                .at[rank]
                .set(neighbors),
                axis,
            )
            > 0
        )

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=PartitionSpec(axis),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(forest.mesh.vertex_ids)


@eqx.filter_jit
def _bind_migration_state(
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    workset: SimplexNeighborhoodWorkset,
    cell_field_ids: Array,
    cell_fields: Array,
    vertex_field_ids: Array,
    vertex_fields: Array,
    cell_epochs: Array,
    cell_history: tuple[Array, ...],
    vertex_history: tuple[Array, ...],
) -> OwnerLocalMigrationState:
    from ._distribution_forest import _vertex_record_owners

    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(
        raw_block: AdaptiveSimplexState,
        work_block: SimplexNeighborhoodWorkset,
        ci_block: Array,
        cf_block: Array,
        vi_block: Array,
        vf_block: Array,
        epoch_block: Array,
        ch_block: tuple[Array, ...],
        vh_block: tuple[Array, ...],
    ) -> OwnerLocalMigrationState:
        raw, work, ci, cf, vi, vf, epochs, ch, vh = jax.tree_util.tree_map(
            lambda value: value[0],
            (
                raw_block,
                work_block,
                ci_block,
                cf_block,
                vi_block,
                vf_block,
                epoch_block,
                ch_block,
                vh_block,
            ),
        )
        rank = jax.lax.axis_index(axis)

        def query(
            ids: Array, values: Array, requested: Array, valid: Array
        ) -> tuple[Array, Array]:
            result = jnp.zeros(requested.shape + values.shape[1:], values.dtype)
            covered = jnp.zeros(requested.shape, jnp.bool_)

            def visit(
                _: int, carry: tuple[Array, Array, Array, Array, Array]
            ) -> tuple[Array, Array, Array, Array, Array]:
                keys, payload, result, covered, status = carry
                order = jnp.argsort(jnp.where(keys >= 0, keys, _SENTINEL), stable=True)
                sorted_ids = jnp.where(keys[order] >= 0, keys[order], _SENTINEL)
                pos = jnp.minimum(
                    jnp.searchsorted(sorted_ids, requested), keys.shape[0] - 1
                )
                match = valid & (requested >= 0) & (sorted_ids[pos] == requested)
                next_values = payload[order[pos]]
                mismatch = result != next_values
                if mismatch.ndim > requested.ndim:
                    mismatch = jnp.any(
                        mismatch, axis=tuple(range(requested.ndim, mismatch.ndim))
                    )
                status |= jnp.where(jnp.any(covered & match & mismatch), _INVALID, 0)
                mask = match.reshape(requested.shape + (1,) * (values.ndim - 1))
                result = jnp.where(mask, next_values, result)
                covered |= match
                return (
                    jax.lax.ppermute(keys, axis, ring),
                    jax.lax.ppermute(payload, axis, ring),
                    result,
                    covered,
                    status,
                )

            _, _, result, covered, status = jax.lax.fori_loop(
                0,
                parts.part_count,
                visit,
                (ids, values, result, covered, jnp.asarray(0, jnp.int32)),
            )
            return result, status | jnp.where(jnp.any(valid & ~covered), _INVALID, 0)

        cells, cell_status = query(ci, cf, work.cell_ids, work.cell_valid)
        vertices, vertex_status = query(
            vi, vf, work.cell_vertices, work.cell_valid[:, None]
        )
        epoch_values, epoch_status = query(
            raw.mesh.cell_ids, epochs, work.cell_ids, work.cell_valid
        )
        owners = _vertex_record_owners(parts, raw.mesh.vertex_ids)
        status = _collective_status(
            work.status | cell_status | vertex_status | epoch_status, axis
        )
        state = OwnerLocalMigrationState(
            _replace_work(work, status=status),
            cells,
            vertices,
            epoch_values,
            jnp.zeros_like(work.cell_vertices, jnp.int32),
            raw,
            jnp.where(raw.mesh.cell_ids >= 0, rank, -1).astype(jnp.int32),
            jnp.where(raw.mesh.vertex_ids >= 0, owners, -1).astype(jnp.int32),
            ch,
            vh,
        )
        return jax.tree_util.tree_map(lambda value: value[None], state)

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec, spec, spec, spec, spec, spec, spec),
        out_specs=spec,
        check_vma=False,
    )(
        states,
        workset,
        cell_field_ids,
        cell_fields,
        vertex_field_ids,
        vertex_fields,
        cell_epochs,
        cell_history,
        vertex_history,
    )


def prepare_ownerlocal_migration_state(
    partitioned: PartitionedAdaptiveSimplex,
    states: AdaptiveSimplexState,
    /,
    *,
    cell_field_ids: Array,
    cell_fields: Array,
    vertex_field_ids: Array,
    vertex_fields: Array,
    cell_epochs: Array,
    cell_history: tuple[Array, ...] = (),
    vertex_history: tuple[Array, ...] = (),
) -> OwnerLocalMigrationState:
    """Bind transferred scientific fields to actual source/forest packets.

    Field IDs and values are capacity-padded part-sharded arrays (unused IDs -1).
    Cells/vertices are matched by exact int64 identity, not shape/order. Cell
    history is aligned with all raw allocated cell slots; vertex history with
    all raw allocated vertex slots, including inactive/removed records.
    """
    from ._device_adaptation import PartitionedAdaptiveSimplex
    from ._distribution import expand_partitioned_simplex_neighborhood

    if not isinstance(partitioned, PartitionedAdaptiveSimplex) or not isinstance(
        states, AdaptiveSimplexState
    ):
        raise TypeError(
            "Migration binding requires the canonical partitioned/raw simplex epoch."
        )
    count = partitioned.parts.part_count
    if (
        cell_field_ids.ndim != 2
        or vertex_field_ids.ndim != 2
        or cell_field_ids.shape[0] != count
        or vertex_field_ids.shape[0] != count
        or cell_field_ids.shape[1] == 0
        or vertex_field_ids.shape[1] == 0
        or cell_field_ids.dtype != jnp.int64
        or vertex_field_ids.dtype != jnp.int64
        or cell_fields.shape[:2] != cell_field_ids.shape
        or vertex_fields.shape[:2] != vertex_field_ids.shape
        or cell_epochs.shape != states.mesh.cell_ids.shape
        or cell_epochs.dtype != jnp.int64
        or any(value.shape[:2] != states.mesh.cell_ids.shape for value in cell_history)
        or any(
            value.shape[:2] != states.mesh.vertex_ids.shape for value in vertex_history
        )
    ):
        raise ValueError(
            "Scientific field IDs, epochs and raw history must have explicit matched capacity axes."
        )
    policy = partitioned.prepared.adaptation.policy.distribution
    if policy is None:
        raise ValueError("Migration binding requires the actual prepared distribution.")
    work = expand_partitioned_simplex_neighborhood(
        partitioned.parts,
        states,
        partitioned.states,
        partitioned.source_exterior,
        halo_width=policy.halo_width,
        cell_capacity=partitioned.layout.cell_capacity,
        vertex_capacity=partitioned.layout.vertex_capacity,
    )
    bound = _bind_migration_state(
        partitioned.parts,
        states,
        work,
        cell_field_ids,
        cell_fields,
        vertex_field_ids,
        vertex_fields,
        cell_epochs,
        cell_history,
        vertex_history,
    )
    hydrated, _ = _compiled_hydrate_closure(partitioned.parts, bound, work)
    return hydrated


@eqx.filter_jit
def ownerlocal_forest_marks(
    parts: AdaptiveSimplexParts,
    state: OwnerLocalMigrationState,
    marks: Array,
    /,
) -> tuple[Array, Array]:
    """Route solver-owned requests to reassembled raw families by scientific ID.

    Returns the raw-slot marks and collective status; unknown/lost requests
    reject the operation rather than disappearing at an ownership boundary.
    """
    if state.forest is None or state.forest_cell_owners is None:
        raise ValueError("Raw family marks require the accepted complete forest.")
    if marks.shape != state.workset.cell_ids.shape or marks.dtype != jnp.bool_:
        raise ValueError(
            "Solver marks must be bool and aligned with canonical workset cell IDs."
        )
    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(
        work_block: SimplexNeighborhoodWorkset,
        raw_block: Array,
        owner_block: Array,
        mark_block: Array,
    ) -> tuple[Array, Array]:
        work = jax.tree_util.tree_map(lambda value: value[0], work_block)
        raw_ids, owners, requested = raw_block[0], owner_block[0], mark_block[0]
        rank = jax.lax.axis_index(axis)
        requested &= work.cell_valid & (work.cell_owner == rank)
        result = jnp.zeros(raw_ids.shape, jnp.bool_)

        def visit(
            _: int, carry: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            ids, packet_marks, result = carry
            pos = jnp.minimum(jnp.searchsorted(ids, raw_ids), ids.shape[0] - 1)
            found = (raw_ids >= 0) & (owners == rank) & (ids[pos] == raw_ids)
            result |= found & packet_marks[pos]
            return (
                jax.lax.ppermute(ids, axis, ring),
                jax.lax.ppermute(packet_marks, axis, ring),
                result,
            )

        _, _, result = jax.lax.fori_loop(
            0, parts.part_count, visit, (work.cell_ids, requested, result)
        )
        before = jax.lax.psum(jnp.sum(requested, dtype=jnp.int32), axis)
        after = jax.lax.psum(jnp.sum(result, dtype=jnp.int32), axis)
        status = _collective_status(
            work.status | jnp.where(before != after, _INVALID, 0), axis
        )
        return result[None], status[None]

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec, spec),
        out_specs=(spec, spec),
        check_vma=False,
    )(state.workset, state.forest.mesh.cell_ids, state.forest_cell_owners, marks)


def _solver_owner_part(
    work: SimplexNeighborhoodWorkset,
    proposed: Array,
    raw_ids: Array,
    active: Array,
    authority: Array,
    axis: str,
    part_count: int,
    target_count: int,
) -> tuple[Array, Array]:
    """The identical scientific-ID projection kernel for live or saved logical parts."""
    ring = tuple((rank, (rank + 1) % part_count) for rank in range(part_count))
    rank = jax.lax.axis_index(axis)
    owned = work.cell_valid & (work.cell_owner == rank)
    owners = jnp.full(raw_ids.shape, -1, jnp.int32)
    covered = jnp.zeros(raw_ids.shape, jnp.int32)
    order = jnp.argsort(jnp.where(owned, work.cell_ids, _SENTINEL), stable=True)
    packet_ids = jnp.where(owned[order], work.cell_ids[order], _SENTINEL)

    def visit(
        _: int, carry: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array]:
        ids, valid, source_owners, owners, covered = carry
        position = jnp.minimum(jnp.searchsorted(ids, raw_ids), ids.shape[0] - 1)
        match = active & valid[position] & (ids[position] == raw_ids)
        owners = jnp.where(match, source_owners[position], owners)
        covered += match.astype(jnp.int32)
        return (
            jax.lax.ppermute(ids, axis, ring),
            jax.lax.ppermute(valid, axis, ring),
            jax.lax.ppermute(source_owners, axis, ring),
            owners,
            covered,
        )

    _, _, _, owners, covered = jax.lax.fori_loop(
        0, part_count, visit, (packet_ids, owned[order], proposed[order], owners, covered)
    )
    authoritative = active & (authority == rank)
    before = jax.lax.psum(jnp.sum(owned, dtype=jnp.int32), axis)
    after = jax.lax.psum(jnp.sum(authoritative & (covered == 1), dtype=jnp.int32), axis)
    invalid = jnp.any(active & (covered != 1)) | (before != after)
    invalid |= jnp.any(owned & ((proposed < 0) | (proposed >= target_count)))
    return owners, _collective_status(work.status | jnp.where(invalid, _INVALID, 0), axis)


@eqx.filter_jit
def _project_workset_solver_owners(
    parts: AdaptiveSimplexParts,
    workset: SimplexNeighborhoodWorkset,
    proposals: Array,
    raw_ids: Array,
    raw_active: Array,
    raw_authority: Array,
    target_count: int,
) -> tuple[Array, Array]:
    """One exact scientific-ID projection for current or proposed solver ownership."""
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def local(
        work_block: SimplexNeighborhoodWorkset,
        proposal_block: Array,
        raw_id_block: Array,
        raw_active_block: Array,
        authority_block: Array,
    ) -> tuple[Array, Array]:
        work = jax.tree_util.tree_map(lambda value: value[0], work_block)
        owners, status = _solver_owner_part(
            work,
            proposal_block[0],
            raw_id_block[0],
            raw_active_block[0],
            authority_block[0],
            axis,
            parts.part_count,
            target_count,
        )
        return owners[None], status[None]

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec, spec, spec),
        out_specs=(spec, spec),
        check_vma=False,
    )(workset, proposals, raw_ids, raw_active, raw_authority)


@eqx.filter_jit
def _replay_logical_solver_owners(
    workset: SimplexNeighborhoodWorkset,
    proposals: Array,
    raw_ids: Array,
    raw_active: Array,
    target_count: int,
    axis_name: str,
) -> tuple[Array, Array]:
    """Recompute the saved restart proposal projection without the old devices."""
    count = workset.cell_ids.shape[0]
    authority = jnp.where(raw_ids >= 0, jnp.arange(count, dtype=jnp.int32)[:, None], -1)

    def local(
        work: SimplexNeighborhoodWorkset,
        proposed: Array,
        ids: Array,
        active: Array,
        owners: Array,
    ) -> tuple[Array, Array]:
        return _solver_owner_part(
            work, proposed, ids, active, owners, axis_name, count, target_count
        )

    return jax.vmap(local, axis_name=axis_name)(
        workset, proposals, raw_ids, raw_active, authority
    )


def ownerlocal_solver_cell_owners(
    parts: AdaptiveSimplexParts,
    state: OwnerLocalMigrationState,
    /,
) -> tuple[Array, Array]:
    """Bind solver ownership to actual raw active scientific cells, inactive -1."""
    if state.forest is None or state.forest_cell_owners is None:
        raise ValueError(
            "Solver-owner projection requires the complete accepted raw forest."
        )
    return _project_workset_solver_owners(
        parts,
        state.workset,
        state.workset.cell_owner,
        state.forest.mesh.cell_ids,
        state.forest.mesh.cell_active,
        state.forest_cell_owners,
        parts.part_count,
    )


def restart_solver_target_owners(
    parts: AdaptiveSimplexParts,
    saved: AdaptiveSimplexState,
    workset: SimplexNeighborhoodWorkset,
    requested_solver_owners: Array,
    /,
    *,
    target_part_count: int | None = None,
) -> tuple[Array, Array]:
    """Map the actual graph proposal into complete raw-slot order without slot guesses.

    Graph rows align workset.cell_ids; only its current source-owned rows propose.
    Scientific coverage must match every raw active record exactly once. Inactive
    raw rows are -1 and do not supply family execution ownership.
    """
    count = (
        parts.part_count
        if target_part_count is None
        else operator.index(target_part_count)
    )
    if (
        isinstance(target_part_count, bool)
        or count < 1
        or saved.mesh.cell_ids.shape[0] != parts.part_count
        or workset.cell_ids.shape[0] != parts.part_count
        or requested_solver_owners.shape != workset.cell_ids.shape
        or requested_solver_owners.dtype != jnp.int32
    ):
        raise ValueError(
            "Graph restart proposals require the actual scientific workset rows and target count."
        )
    authority = jnp.where(
        saved.mesh.cell_ids >= 0,
        jnp.arange(parts.part_count, dtype=jnp.int32)[:, None],
        -1,
    )
    return _project_workset_solver_owners(
        parts,
        workset,
        requested_solver_owners,
        saved.mesh.cell_ids,
        saved.mesh.cell_active,
        authority,
        count,
    )


@dataclass(frozen=True, slots=True)
class SimplexGraphRestartProposal:
    """An executed native dual-graph partition bound to its raw-slot proposal.

    ``parts`` is the accepted owner-local partition, aligned with
    ``workset.cell_ids``; ``requested_target_owners`` is its exact projection
    into saved raw-cell order. Restart validation replays the same dual-graph
    lowering, multilevel partition operation and projection from these saved
    logical rows, so a GRAPH ownership claim is never a caller label.
    """

    workset: SimplexNeighborhoodWorkset
    vertex_weights: Array
    edge_weights: Array
    parts: Array
    requested_target_owners: Array
    maximum_imbalance: float

    @property
    def plan(self) -> GraphPartitionPlan:
        """The canonical mesh GRAPH plan of this proposal's owner count and tolerance."""
        return GraphPartitionPlan(
            self.parts.shape[0], maximum_imbalance=self.maximum_imbalance
        )


def prepare_graph_restart_proposal(
    parts: AdaptiveSimplexParts,
    saved: AdaptiveSimplexState,
    state: OwnerLocalMigrationState,
    policy: MeshPartitionPolicy,
    /,
    *,
    vertex_weights: Array,
    edge_weights: Array | None = None,
) -> SimplexGraphRestartProposal:
    """Execute the native GRAPH route on the actual dual graph of an accepted epoch.

    The owner-local partition runs on the current execution parts, so the
    requested owner count equals the current one. A collectively refused
    partition or an incomplete scientific projection raises; its measured
    graph evidence is reported, never replaced by incoming ownership.
    """
    from ..graph._partition import partition_graph
    from ._distribution import MeshPartitionKind, MeshPartitionPolicy

    if (
        not isinstance(policy, MeshPartitionPolicy)
        or policy.kind is not MeshPartitionKind.GRAPH
    ):
        raise ValueError(
            "A graph restart proposal requires a GRAPH mesh partition policy."
        )
    if policy.part_count != parts.part_count:
        raise ValueError(
            "The owner-local GRAPH route partitions onto its current execution parts."
        )
    if edge_weights is None:
        edge_weights = jnp.ones(state.workset.cell_vertices.shape, jnp.int64)
    graph = ownerlocal_simplex_graph(
        parts, state, vertex_weights=vertex_weights, edge_weights=edge_weights
    )
    plan = GraphPartitionPlan(
        parts.part_count, maximum_imbalance=policy.maximum_imbalance
    )
    result = partition_graph(graph, plan)
    evidence = result.evidence
    if not evidence.accepted or not isinstance(result.parts, Array):
        raise ValueError(
            "The native GRAPH partition collectively refused the actual dual graph "
            f"(resource status {evidence.resource_status}, evidence {evidence.evidence_id})."
        )
    requested, status = restart_solver_target_owners(
        parts, saved, state.workset, result.parts
    )
    if bool(jax.device_get(jnp.any(status != 0))):
        raise ValueError(
            "The GRAPH proposal does not cover every raw active scientific cell exactly once."
        )
    return SimplexGraphRestartProposal(
        state.workset,
        vertex_weights,
        edge_weights,
        result.parts,
        requested,
        policy.maximum_imbalance,
    )


def replay_graph_restart_proposal(
    proposal: SimplexGraphRestartProposal,
    saved: AdaptiveSimplexState,
    /,
    *,
    axis_name: str,
) -> None:
    """Recompute the saved GRAPH proposal exactly on the current devices."""
    from .._fingerprint import logical_array_value_collection_digest
    from ..graph._distributed_partition import replay_logical_partition

    def digest(value: Array) -> str:
        return logical_array_value_collection_digest({"value": value})

    work = proposal.workset
    if (
        work.cell_ids.shape != proposal.parts.shape
        or proposal.parts.dtype != jnp.int32
        or proposal.vertex_weights.shape != work.cell_ids.shape
        or proposal.vertex_weights.dtype != jnp.int64
        or proposal.edge_weights.shape != work.cell_vertices.shape
        or proposal.edge_weights.dtype != jnp.int64
        or proposal.requested_target_owners.shape != saved.mesh.cell_ids.shape
        or work.cell_ids.shape[0] != saved.mesh.cell_ids.shape[0]
    ):
        raise ValueError(
            "A GRAPH restart proposal must align with its saved dual-graph rows and raw forest."
        )
    offsets, neighbors, weights, valid = replay_logical_simplex_graph(
        work, proposal.edge_weights, axis_name=axis_name
    )
    parts, _, measured = replay_logical_partition(
        offsets,
        neighbors,
        weights,
        proposal.vertex_weights,
        work.cell_ids,
        work.cell_owner,
        valid,
        proposal.plan,
        axis_name=axis_name,
    )
    if bool(jax.device_get(measured[5] != 0)) or digest(parts) != digest(proposal.parts):
        raise ValueError(
            "Saved GRAPH ownership differs from the replayed native dual-graph partition."
        )
    requested, status = _replay_logical_solver_owners(
        work,
        parts,
        saved.mesh.cell_ids,
        saved.mesh.cell_active,
        parts.shape[0],
        axis_name,
    )
    if bool(jax.device_get(jnp.any(status != 0))) or digest(requested) != digest(
        proposal.requested_target_owners
    ):
        raise ValueError(
            "The raw-slot proposal is not the exact projection of its native GRAPH partition."
        )


def ownerlocal_solver_forest(
    parts: AdaptiveSimplexParts,
    state: OwnerLocalMigrationState,
    /,
) -> tuple[AdaptiveSimplexState, tuple[Array, ...], tuple[Array, ...], Array]:
    """Receive complete family records needed by each solver resident closure.

    This bounded projection is a hierarchy-construction view, never an execution
    ownership replacement. Raw active flags remain global family facts; callers
    select solver labels by their resident scientific IDs. The original accepted
    execution forest and its owners remain unchanged.
    """
    from ._distribution_forest import migrate_simplex_forest

    if (
        state.forest is None
        or state.forest_cell_owners is None
        or state.forest_vertex_owners is None
    ):
        raise ValueError(
            "Solver hierarchy closure requires the complete accepted raw forest and record authority."
        )
    forest, _, _, cells, vertices, status = migrate_simplex_forest(
        parts,
        state.forest,
        state.forest_cell_owners,
        state.forest_vertex_owners,
        state.cell_history,
        state.vertex_history,
        state.workset,
        include_ghost_families=True,
    )
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def collective(block: Array) -> Array:
        return _collective_status(block[0], axis)[None]

    status = jax.shard_map(
        collective, mesh=parts.mesh, in_specs=spec, out_specs=spec, check_vma=False
    )(status)
    return forest, cells, vertices, status


def ownerlocal_solver_forest_geometry(
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    solver_workset: SimplexNeighborhoodWorkset,
    /,
) -> tuple[AdaptiveSimplexState, Array]:
    """Receive geometry/history families without claiming any physical payloads."""
    from ._distribution_forest import _vertex_record_owners, migrate_simplex_forest

    if (
        states.mesh.cell_ids.shape[0] != parts.part_count
        or solver_workset.cell_ids.shape[0] != parts.part_count
    ):
        raise ValueError(
            "Solver hierarchy projection requires the actual execution and solver placements."
        )
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def authority(cell_block: Array, vertex_block: Array) -> tuple[Array, Array]:
        cells, vertices = cell_block[0], vertex_block[0]
        rank = jax.lax.axis_index(axis)
        return jnp.where(cells >= 0, rank, -1).astype(jnp.int32)[
            None
        ], _vertex_record_owners(parts, vertices)[None]

    cell_owners, vertex_owners = jax.shard_map(
        authority,
        mesh=parts.mesh,
        in_specs=(spec, spec),
        out_specs=(spec, spec),
        check_vma=False,
    )(states.mesh.cell_ids, states.mesh.vertex_ids)
    forest, _, _, _, _, status = migrate_simplex_forest(
        parts,
        states,
        cell_owners,
        vertex_owners,
        (),
        (),
        solver_workset,
        include_ghost_families=True,
    )

    def collective(block: Array) -> Array:
        return _collective_status(block[0], axis)[None]

    status = jax.shard_map(
        collective, mesh=parts.mesh, in_specs=spec, out_specs=spec, check_vma=False
    )(status)
    return forest, status


def relocate_solver_workset(
    parts: AdaptiveSimplexParts,
    raw_workset: SimplexNeighborhoodWorkset,
    requested_solver_owners: Array,
    /,
    *,
    halo_width: int,
    vertex_capacity: int,
    route_metadata_capacity_bytes: int = 1 << 20,
) -> tuple[SimplexNeighborhoodWorkset, AdaptiveSimplexParts, Array]:
    """Stage exact geometry-only solver ownership/closure using the migration core.

    No physical field or epoch payload exists on this path. Nothing is published:
    the caller must reject the complete epoch on any returned nonzero status.
    Target rows align with raw_workset.cell_ids; only raw-owned rows send.
    """
    from ._distribution import expand_simplex_neighborhood_workset

    halo = operator.index(halo_width)
    vertices = operator.index(vertex_capacity)
    budget = operator.index(route_metadata_capacity_bytes)
    shape = raw_workset.cell_ids.shape
    if (
        len(shape) != 2
        or shape[0] != parts.part_count
        or requested_solver_owners.shape != shape
        or requested_solver_owners.dtype != jnp.int32
        or raw_workset.cell_ids.dtype != jnp.int64
        or raw_workset.cell_vertices.dtype != jnp.int64
        or raw_workset.cell_owner.dtype != jnp.int32
        or raw_workset.cell_valid.dtype != jnp.bool_
        or isinstance(halo_width, bool)
        or isinstance(vertex_capacity, bool)
        or halo < 0
        or vertices <= 0
    ):
        raise ValueError(
            "Geometry-only relocation requires canonical matched packets, owners and capacities."
        )
    if (
        isinstance(route_metadata_capacity_bytes, bool)
        or budget < parts.part_count * parts.part_count
    ):
        raise ValueError(
            "Geometry-only route metadata exceeds its explicit prepare-barrier byte budget."
        )
    owned, _, _, _, _ = _compiled_owned_packets(
        parts, raw_workset, requested_solver_owners, ()
    )
    target_parts = _prepare_migration_routes(
        parts, _compiled_migration_routes(parts, owned)
    )
    closure = expand_simplex_neighborhood_workset(
        target_parts, owned, halo_width=halo, vertex_capacity=vertices
    )
    hydrated, _, _, vertex_counts = _compiled_hydrate_packets(
        target_parts, owned, (), closure
    )
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def finish(
        block: SimplexNeighborhoodWorkset, counts: Array
    ) -> SimplexNeighborhoodWorkset:
        work = jax.tree_util.tree_map(lambda value: value[0], block)
        status = _collective_status(
            work.status | jnp.where(counts[0] > vertices, _CAPACITY, 0), axis
        )
        return jax.tree_util.tree_map(
            lambda value: value[None], _replace_work(work, status=status)
        )

    staged = jax.shard_map(
        finish, mesh=parts.mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
    )(
        hydrated,
        vertex_counts,
    )
    return staged, target_parts, staged.status


def prepare_ownerlocal_migration(
    parts: AdaptiveSimplexParts,
    original: OwnerLocalMigrationState,
    requested_target_owners: Array,
    /,
    *,
    halo_width: int,
    vertex_capacity: int,
    route_metadata_capacity_bytes: int = 1 << 20,
) -> PreparedOwnerLocalMigration:
    """Prepare geometry/state migration and actual requested ghost closure.

    Target owners are int32 [parts,C] in the workset's semantic cell-ID order;
    only source-owned rows send. A retained raw forest carries explicit
    authoritative cell/vertex owners and aligned dtype-preserving history leaves.

    ``route_metadata_capacity_bytes`` bounds the host prepare-barrier reads:
    one replicated bool [parts,parts] solver-routing matrix, plus a separate
    family-execution matrix when history is retained. Geometry, field content
    and ancestry remain in capacity-sized device packets.
    """
    from ._distribution import expand_simplex_neighborhood_workset

    halo = operator.index(halo_width)
    vertices = operator.index(vertex_capacity)
    route_bytes = (
        parts.part_count * parts.part_count * (2 if original.forest is not None else 1)
    )
    route_budget = operator.index(route_metadata_capacity_bytes)
    if isinstance(route_metadata_capacity_bytes, bool) or route_budget < route_bytes:
        raise ValueError(
            "Migration routing metadata exceeds the explicit prepare-barrier byte budget."
        )
    shape = original.workset.cell_ids.shape
    width = original.workset.cell_vertices.shape[-1]
    if (
        len(shape) != 2
        or shape[0] != parts.part_count
        or requested_target_owners.shape != shape
        or requested_target_owners.dtype != jnp.int32
        or original.cell_epochs.shape != shape
        or original.cell_epochs.dtype != jnp.int64
        or original.vertex_owners.shape != (*shape, width)
        or original.vertex_owners.dtype != jnp.int32
        or original.cell_fields.shape[:2] != shape
        or original.vertex_fields.shape[:3] != (*shape, width)
        or original.workset.cell_ids.dtype != jnp.int64
        or original.workset.cell_vertices.dtype != jnp.int64
        or original.workset.cell_owner.dtype != jnp.int32
        or original.workset.root_cell_ids.dtype != jnp.int64
        or original.workset.source_vertex_ids.dtype != jnp.int64
        or original.workset.source_cell_ids.dtype != jnp.int64
        or original.workset.source_cell_ids.shape[:2] != shape
        or isinstance(halo_width, bool)
        or isinstance(vertex_capacity, bool)
        or halo < 0
        or vertices <= 0
    ):
        raise ValueError(
            "Migration requires matched explicit-dtype canonical packets and positive capacity."
        )
    if original.forest is None:
        if (
            original.forest_cell_owners is not None
            or original.forest_vertex_owners is not None
            or original.cell_history
            or original.vertex_history
        ):
            raise ValueError(
                "Forest ownership and history require the canonical raw forest."
            )
    else:
        forest = original.forest
        cell_owners, vertex_owners = (
            original.forest_cell_owners,
            original.forest_vertex_owners,
        )
        if cell_owners is None or vertex_owners is None:
            raise ValueError(
                "Raw forest migration requires explicit authoritative record ownership."
            )
        if (
            forest.mesh.cell_ids.shape[0] != parts.part_count
            or cell_owners.shape != forest.mesh.cell_ids.shape
            or cell_owners.dtype != jnp.int32
            or vertex_owners.shape != forest.mesh.vertex_ids.shape
            or vertex_owners.dtype != jnp.int32
            or any(
                value.shape[:2] != cell_owners.shape for value in original.cell_history
            )
            or any(
                value.shape[:2] != vertex_owners.shape
                for value in original.vertex_history
            )
        ):
            raise ValueError(
                "Forest records and typed history must match their canonical raw capacities."
            )
    packets = OwnerLocalMigrationState(
        original.workset,
        original.cell_fields,
        original.vertex_fields,
        original.cell_epochs,
        original.vertex_owners,
    )
    owned, before, sent, received = _compiled_owned_migration(
        parts, packets, requested_target_owners
    )
    route_mask = _compiled_migration_routes(parts, owned.workset)
    # This is routing metadata only, already replicated by device psum. No
    # cell, vertex, candidate, ancestry or field arrays cross this host barrier.
    target_parts = _prepare_migration_routes(parts, route_mask)
    closure = expand_simplex_neighborhood_workset(
        target_parts, owned.workset, halo_width=halo, vertex_capacity=vertices
    )
    candidate, vertex_counts = _compiled_hydrate_closure(target_parts, owned, closure)
    forest_parts = None
    if original.forest is not None:
        from ._distribution_forest import migrate_simplex_forest

        if original.forest_cell_owners is None or original.forest_vertex_owners is None:
            raise ValueError("Raw forest ownership was not validated.")
        (
            forest,
            cell_owners,
            vertex_owners,
            cell_history,
            vertex_history,
            forest_status,
        ) = migrate_simplex_forest(
            parts,
            original.forest,
            original.forest_cell_owners,
            original.forest_vertex_owners,
            original.cell_history,
            original.vertex_history,
            owned.workset,
        )
        candidate = OwnerLocalMigrationState(
            _replace_work(
                candidate.workset, status=candidate.workset.status | forest_status
            ),
            candidate.cell_fields,
            candidate.vertex_fields,
            candidate.cell_epochs,
            candidate.vertex_owners,
            forest,
            cell_owners,
            vertex_owners,
            cell_history,
            vertex_history,
        )
        forest_parts = _prepare_migration_routes(
            parts, _forest_migration_routes(parts, forest)
        )
    # All reductions below are device collectives, including hydration refusal.
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def finish(
        block: OwnerLocalMigrationState, vertex_block: Array
    ) -> OwnerLocalMigrationState:
        state = jax.tree_util.tree_map(lambda value: value[0], block)
        status = state.workset.status | jnp.where(
            vertex_block[0] > vertices, _CAPACITY, 0
        )
        status = _collective_status(status, axis)
        return jax.tree_util.tree_map(
            lambda value: value[None],
            eqx.tree_at(
                lambda value: value.workset,
                state,
                _replace_work(state.workset, status=status),
            ),
        )

    candidate = jax.shard_map(
        finish, mesh=parts.mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
    )(candidate, vertex_counts)
    rank = jnp.arange(parts.part_count, dtype=jnp.int32)[:, None]
    after = jnp.sum(
        candidate.workset.cell_valid & (candidate.workset.cell_owner == rank),
        axis=1,
        dtype=jnp.int32,
    )
    resident = jnp.sum(candidate.workset.cell_valid, axis=1, dtype=jnp.int32)
    forest_before = jnp.zeros(parts.part_count, jnp.int32)
    forest_after = jnp.zeros_like(forest_before)
    forest_vertices_before = jnp.zeros_like(forest_before)
    forest_vertices_after = jnp.zeros_like(forest_before)
    forest_cell_capacity = forest_vertex_capacity = 0
    if original.forest is not None and candidate.forest is not None:
        forest_before = jnp.sum(
            original.forest.mesh.cell_ids >= 0, axis=1, dtype=jnp.int32
        )
        forest_after = jnp.sum(
            candidate.forest.mesh.cell_ids >= 0, axis=1, dtype=jnp.int32
        )
        forest_vertices_before = jnp.sum(
            original.forest.mesh.vertex_ids >= 0, axis=1, dtype=jnp.int32
        )
        forest_vertices_after = jnp.sum(
            candidate.forest.mesh.vertex_ids >= 0, axis=1, dtype=jnp.int32
        )
        forest_cell_capacity = original.forest.mesh.cell_ids.shape[1]
        forest_vertex_capacity = original.forest.mesh.vertex_ids.shape[1]
    evidence = OwnerLocalMigrationEvidence(
        candidate.workset.status,
        candidate.workset.status == 0,
        before,
        after,
        resident,
        vertex_counts,
        sent,
        received,
        forest_before,
        forest_after,
        forest_vertices_before,
        forest_vertices_after,
        halo,
        shape[1],
        vertices,
        forest_cell_capacity,
        forest_vertex_capacity,
        route_bytes,
    )
    return PreparedOwnerLocalMigration(
        original, candidate, evidence, target_parts, forest_parts
    )


@eqx.filter_jit
def accept_ownerlocal_migration(
    prepared: PreparedOwnerLocalMigration, /
) -> OwnerLocalMigrationState:
    """Atomically publish all staged arrays, or preserve all original arrays."""
    accepted = prepared.evidence.accepted

    def select(original: Array, candidate: Array) -> Array:
        mask = accepted.reshape((accepted.shape[0],) + (1,) * (original.ndim - 1))
        return jnp.where(mask, candidate, original)

    return jax.tree_util.tree_map(select, prepared.original, prepared.candidate)
