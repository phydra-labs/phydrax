#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic bounded migration of native owner-local simplex packets.

The generic WeightedCSRGraph owner partitions the dual-facet graph.
This module lowers adjacency and consumes graph-produced target owners in
semantic-ID row order; it does not own another partition algorithm.
Capacity-sized packets circulate without gathering geometry, CSR or candidates.

Raw bisection forest records travel as AdaptiveSimplexState leaves.
Stable scientific IDs replace slot references in flight; family execution
ownership is reconstructed separately from solver graph ownership.
After acceptance, solvers must rebind their ownership, closure and field views.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec

from .._execution_array import _place_process_local_array
from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    masked_simplex_facet_neighbors,
    vertex_half_facets,
)
from ..graph._distributed_partition import replay_logical_partition
from ..graph._partition import GraphPartitionPlan, partition_graph, WeightedCSRGraph
from ..typing import checked

# Module references defer symbol resolution across mutually dependent execution owners.
from . import (
    _device_adaptation as device_adaptation_owner,
    _distribution as distribution_owner,
    _restart_distribution as restart_distribution_owner,
)
from ._collective_failure import agreed_local, agreed_local_step
from ._distribution_forest import (
    migrate_simplex_forest,
    restore_forest_records,
    slots,
    vertex_record_owners,
)
from ._migration_execution import MigrationParts, partition_map, with_migration_routes


if TYPE_CHECKING:
    from ..graph._partition import WeightedCSRGraph
    from ._device_adaptation import PartitionedAdaptiveSimplex, PreparedAdaptiveSimplex
    from ._distribution import SimplexNeighborhoodWorkset


_SENTINEL = jnp.iinfo(jnp.int64).max
_CAPACITY = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
_INVALID = int(AdaptiveSimplexStatus.INVALID_GEOMETRY)


class OwnerLocalMigrationState(StrictModule, NonTrainableState):
    """Cell packets with aligned numerical content and epoch IDs.

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
    acceptance; rejection retains the source execution binding and arrays.
    """

    original: OwnerLocalMigrationState
    candidate: OwnerLocalMigrationState
    evidence: OwnerLocalMigrationEvidence
    target_parts: MigrationParts
    forest_parts: MigrationParts | None = None


def collective_axis_status_union(status: Array, axis: str) -> Array:
    shifts = jnp.arange(8, dtype=jnp.int32)
    bits = (status >> shifts) & 1
    return jnp.sum(
        (jax.lax.psum(bits, axis) > 0).astype(jnp.int32) << shifts, dtype=jnp.int32
    )


WORKSET_FIELDS = (
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
    work: distribution_owner.SimplexNeighborhoodWorkset, **changes: Array
) -> distribution_owner.SimplexNeighborhoodWorkset:
    return distribution_owner.SimplexNeighborhoodWorkset(**(work.arrays() | changes))


def _migrate_owned_packets(
    parts: MigrationParts,
    workset: distribution_owner.SimplexNeighborhoodWorkset,
    targets: Array,
    payloads: tuple[Array, ...],
) -> tuple[
    distribution_owner.SimplexNeighborhoodWorkset, tuple[Array, ...], Array, Array, Array
]:
    """Transport geometry and supplied cell-aligned payloads."""

    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )
    transported_payloads = tuple(value for value in payloads if value.size)

    def local(
        block: distribution_owner.SimplexNeighborhoodWorkset,
        target_block: Array,
        payload_block: tuple[Array, ...],
    ) -> tuple[
        distribution_owner.SimplexNeighborhoodWorkset,
        tuple[Array, ...],
        Array,
        Array,
        Array,
    ]:
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
                distribution_owner.SimplexNeighborhoodWorkset,
                tuple[Array, ...],
                distribution_owner.SimplexNeighborhoodWorkset,
                tuple[Array, ...],
            ],
        ) -> tuple[
            distribution_owner.SimplexNeighborhoodWorkset,
            tuple[Array, ...],
            distribution_owner.SimplexNeighborhoodWorkset,
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

            merged = distribution_owner.SimplexNeighborhoodWorkset(
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
        candidate = _replace_work(
            candidate, status=collective_axis_status_union(status, axis)
        )
        return jax.tree_util.tree_map(
            lambda value: value[None],
            (candidate, candidate_values, before, before - retained, after - retained),
        )

    work, transported, before, sent, received = partition_map(
            local,
            parts=parts,
            in_specs=(spec, spec, spec),
            out_specs=(spec, spec, spec, spec, spec),
            check_vma=False,
        )(workset, targets, transported_payloads)
    values = iter(transported)
    return (
        work,
        tuple(next(values) if value.size else value for value in payloads),
        before,
        sent,
        received,
    )


_compiled_owned_packets = eqx.filter_jit(_migrate_owned_packets)


def _owned_migration(
    parts: MigrationParts,
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
    parts: MigrationParts,
    owned: SimplexNeighborhoodWorkset,
    payloads: tuple[Array, ...],
    closure: SimplexNeighborhoodWorkset,
    corner_payloads: tuple[int, ...] = (),
) -> tuple[SimplexNeighborhoodWorkset, tuple[Array, ...], Array, Array]:
    """Hydrate geometry and supplied payloads into a solver halo by stable IDs."""
    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )
    transported_indices = tuple(
        index for index, value in enumerate(payloads) if value.size
    )
    transported_payloads = tuple(payloads[index] for index in transported_indices)
    corner_payloads = tuple(
        transported_indices.index(index)
        for index in corner_payloads
        if index in transported_indices
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

    work, transported, owners, vertices = partition_map(
            local,
            parts=parts,
            in_specs=(spec, spec, spec),
            out_specs=(spec, spec, spec, spec),
            check_vma=False,
        )(owned, transported_payloads, closure)
    values = iter(transported)
    return (
        work,
        tuple(
            next(values)
            if value.size
            else jnp.empty((*closure.cell_ids.shape, *value.shape[2:]), dtype=value.dtype)
            for value in payloads
        ),
        owners,
        vertices,
    )


_compiled_hydrate_packets = eqx.filter_jit(_hydrate_packet_closure)


def _hydrate_closure(
    parts: MigrationParts,
    owned: OwnerLocalMigrationState,
    closure: SimplexNeighborhoodWorkset,
) -> tuple[OwnerLocalMigrationState, Array]:
    work, payloads, owners, vertices = _compiled_hydrate_packets(
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


def _migration_routes(
    parts: MigrationParts, workset: SimplexNeighborhoodWorkset
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

    return partition_map(
            local,
            parts=parts,
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

    ring = tuple((rank, (rank + 1) % part_count) for rank in range(part_count))
    rank = jax.lax.axis_index(axis)
    owned = work.cell_valid & (work.cell_owner == rank)
    source = _replace_work(work, cell_valid=owned)
    keys = distribution_owner.neighborhood_facet_keys(source)
    capacity, width = work.cell_vertices.shape
    self_ids = jnp.repeat(work.cell_ids, width)
    neighbors = jnp.full((capacity * width,), _SENTINEL, jnp.int64)

    def visit(
        _: int, carry: tuple[SimplexNeighborhoodWorkset, Array]
    ) -> tuple[SimplexNeighborhoodWorkset, Array]:
        packet, neighbors = carry
        facets = distribution_owner.neighborhood_facet_keys(packet)
        order = jnp.lexsort(
            tuple(facets[:, column] for column in range(width - 2, -1, -1))
        )
        facets = facets[order]
        ids = jnp.repeat(packet.cell_ids, width)[order]
        lower = jnp.minimum(
            distribution_owner.lexicographic_bound(facets, keys, False),
            facets.shape[0] - 1,
        )
        upper = jnp.maximum(
            distribution_owner.lexicographic_bound(facets, keys, True) - 1, 0
        )
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
    parts: MigrationParts,
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

    return partition_map(
            local,
            parts=parts,
            in_specs=(spec, spec),
            out_specs=(spec, spec, spec, spec),
            check_vma=False,
        )(workset, edge_weights)


def _validate_simplex_graph_replay(
    workset: SimplexNeighborhoodWorkset, edge_weights: Array, axis_name: str
) -> None:
    if (
        workset.cell_ids.ndim != 2
        or edge_weights.shape != workset.cell_vertices.shape
        or edge_weights.dtype != jnp.int64
        or not isinstance(axis_name, str)
        or not axis_name
    ):
        raise ValueError(
            "migration.graph_replay must bind part-shaped geometry, int64 facet weights, and a nonempty axis; got incompatible rows, dtype, or axis"
        )


def replay_ownerlocal_simplex_graph(
    parts: MigrationParts,
    workset: SimplexNeighborhoodWorkset,
    edge_weights: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Replay graph packets on an explicitly supplied execution owner.

    No scientific owner/predecessor DAG enters the numerical program. Physical
    and historical execution retain the same ordered dual-facet packet kernel;
    their identity comes from ``parts``, never from the workset's shape.
    """
    _validate_simplex_graph_replay(workset, edge_weights, parts.axis_name)
    if workset.cell_ids.shape[0] != parts.part_count:
        raise ValueError(
            "migration.graph_replay.parts must preserve the explicit execution rank rows; got incompatible owner count"
        )
    return _simplex_graph_packets(parts, workset, edge_weights)


def place_logical_simplex_graph_weights(
    workset: SimplexNeighborhoodWorkset,
    vertex_weights: Array,
    edge_weights: Array,
    /,
) -> tuple[Array, Array]:
    """Admit complete local weights on the authenticated workset placement.

    Shape-derived weights are complete process-local source rows, not historical
    owner shards. Slice those rows into current addressable buffers before JIT;
    never fetch or gather the restored global topology to manufacture weights.
    """
    def place() -> tuple[Array, Array]:
        placed = []
        for value, reference in (
            (vertex_weights, workset.cell_ids),
            (edge_weights, workset.cell_vertices),
        ):
            if value.shape != reference.shape or value.dtype != jnp.int64:
                raise ValueError(
                    "migration.graph_replay.weights must contain complete int64 source rows; got missing or incompatible rows"
                )
            if reference.is_deleted() or value.is_deleted():
                raise ValueError(
                    "migration.graph_replay.weights must retain live source rows and placement"
                )
            if value.is_fully_addressable:
                placed.append(_place_process_local_array(value, reference.sharding))
            elif value.sharding.is_equivalent_to(
                reference.sharding, ndim=value.ndim
            ):
                placed.append(value)
            else:
                raise ValueError(
                    "migration.graph_replay.weights must use current workset placement; got foreign source rows"
                )
        return placed[0], placed[1]

    return agreed_local(place)


@eqx.filter_jit
def replay_logical_simplex_graph(
    workset: SimplexNeighborhoodWorkset,
    edge_weights: Array,
    /,
    *,
    axis_name: str,
) -> tuple[Array, Array, Array, Array]:
    """Recompute dual-facet CSR from saved logical parts on the current devices."""
    _validate_simplex_graph_replay(workset, edge_weights, axis_name)
    count = workset.cell_ids.shape[0]

    def local(
        work: SimplexNeighborhoodWorkset, weights: Array
    ) -> tuple[Array, Array, Array, Array]:
        return _simplex_graph_part(work, weights, axis_name, count)

    return jax.vmap(local, in_axes=(0, 0), out_axes=(0, 0, 0, 0), axis_name=axis_name)(
        workset, edge_weights
    )


def ownerlocal_simplex_graph(
    parts: MigrationParts,
    state: OwnerLocalMigrationState,
    /,
    *,
    vertex_weights: Array,
    edge_weights: Array | None = None,
) -> WeightedCSRGraph:
    """Lower the owner-local dual-facet graph to WeightedCSRGraph."""

    shape = state.workset.cell_ids.shape
    edge_shape = state.workset.cell_vertices.shape
    if vertex_weights.shape != shape or vertex_weights.dtype != jnp.int64:
        raise ValueError(
            f"migration.graph.vertex_weights must be int64 and align with cell-ID rows; got shape={vertex_weights.shape}, dtype={vertex_weights.dtype}, expected_shape={shape}"
        )
    if edge_weights is None:
        edge_weights = jnp.ones(edge_shape, jnp.int64)
    if edge_weights.shape != edge_shape or edge_weights.dtype != jnp.int64:
        raise ValueError(
            f"migration.graph.edge_weights must be int64 and align with cell corners; got shape={edge_weights.shape}, dtype={edge_weights.dtype}, expected_shape={edge_shape}"
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


def prepare_migration_routes(
    parts: MigrationParts, route_mask: Array
) -> MigrationParts:
    """Validate the sole bounded replicated metadata read at the prepare barrier."""
    if route_mask.is_fully_addressable:
        mask = np.asarray(jax.device_get(route_mask), dtype=np.bool_)
    else:
        mask = np.asarray(
            multihost_utils.process_allgather(route_mask, tiled=True), dtype=np.bool_
        )
    if mask.shape != (parts.part_count, parts.part_count) or not np.array_equal(
        mask, mask.T
    ):
        raise ValueError(
            f"migration.route_mask must be a reciprocal global rank matrix; got shape={mask.shape}, part_count={parts.part_count}, incompatible shape or asymmetric routes"
        )
    routes = tuple((int(source), int(target)) for source, target in np.argwhere(mask))
    return with_migration_routes(parts, routes)


def forest_migration_routes(
    parts: MigrationParts, forest: AdaptiveSimplexState
) -> Array:
    """Discover reciprocal shared-vertex routes independently of existing routes."""
    return _forest_migration_route_values(forest.mesh.vertex_ids, parts)


@eqx.filter_jit
def _forest_migration_route_values(
    vertex_ids: Array, parts: MigrationParts
) -> Array:
    """Compile route discovery on only its vertex inventory and rank identity."""
    axis, part_count = parts.axis_name, parts.part_count
    ring = tuple((rank, (rank + 1) % part_count) for rank in range(part_count))

    def local(block: Array) -> Array:
        ids = block[0]
        keys = jnp.where(ids >= 0, ids, _SENTINEL)
        rank = jax.lax.axis_index(axis)
        neighbors = jnp.zeros(part_count, jnp.int32)

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

        _, _, neighbors = jax.lax.fori_loop(0, part_count, visit, (keys, rank, neighbors))
        return (
            jax.lax.psum(
                jnp.zeros((part_count, part_count), jnp.int32).at[rank].set(neighbors),
                axis,
            )
            > 0
        )

    return partition_map(
            local,
            parts=parts,
            in_specs=PartitionSpec(axis),
            out_specs=PartitionSpec(),
            check_vma=False,
        )(vertex_ids)


@eqx.filter_jit
def _bind_migration_state(
    parts: MigrationParts,
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
        owners = vertex_record_owners(parts, raw.mesh.vertex_ids)
        status = collective_axis_status_union(
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

    return partition_map(
            local,
            parts=parts,
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


@checked
def prepare_ownerlocal_migration_state(
    partitioned: device_adaptation_owner.PartitionedAdaptiveSimplex,
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
    """Bind transferred scientific fields to source and forest packets.

    Field IDs and values are capacity-padded part-sharded arrays (unused IDs -1).
    Cells/vertices are matched by int64 scientific identity, not shape/order. Cell
    history is aligned with all raw allocated cell slots; vertex history with
    all raw allocated vertex slots, including inactive/removed records.
    """

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
            "migration.fields must bind nonempty int64 ID and epoch axes, aligned field values, and raw history capacities; got incompatible dtype or capacity axes"
        )
    policy = partitioned.prepared.adaptation.policy.distribution
    if policy is None:
        raise ValueError(
            "migration.distribution must match the prepared distribution; got missing policy or owner-count mismatch"
        )
    work = distribution_owner.expand_partitioned_simplex_neighborhood(
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
    hydrated, _ = _hydrate_closure(partitioned.parts, bound, work)
    return hydrated


@eqx.filter_jit
def ownerlocal_forest_marks(
    parts: MigrationParts,
    state: OwnerLocalMigrationState,
    marks: Array,
    /,
) -> tuple[Array, Array]:
    """Route solver-owned requests to reassembled raw families by scientific ID.

    Returns the raw-slot marks and collective status; unknown/lost requests
    reject the operation rather than disappearing at an ownership boundary.
    """
    if state.forest is None or state.forest_cell_owners is None:
        raise ValueError(
            "migration.forest_marks must bind the accepted raw forest; got missing forest or ownership"
        )
    if marks.shape != state.workset.cell_ids.shape or marks.dtype != jnp.bool_:
        raise ValueError(
            f"migration.forest_marks must be bool and align with workset cell IDs; got shape={marks.shape}, dtype={marks.dtype}, expected_shape={state.workset.cell_ids.shape}"
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
        status = collective_axis_status_union(
            work.status | jnp.where(before != after, _INVALID, 0), axis
        )
        return result[None], status[None]

    return partition_map(
            local,
            parts=parts,
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
    return owners, collective_axis_status_union(
        work.status | jnp.where(invalid, _INVALID, 0), axis
    )


@eqx.filter_jit
def _project_workset_solver_owners(
    parts: MigrationParts,
    workset: SimplexNeighborhoodWorkset,
    proposals: Array,
    raw_ids: Array,
    raw_active: Array,
    raw_authority: Array,
    target_count: int,
) -> tuple[Array, Array]:
    """Project current or proposed solver ownership by scientific cell IDs."""
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

    return partition_map(
            local,
            parts=parts,
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
    parts: MigrationParts,
    state: OwnerLocalMigrationState,
    /,
) -> tuple[Array, Array]:
    """Bind solver ownership to active raw scientific cells, using -1 for inactive rows."""
    if state.forest is None or state.forest_cell_owners is None:
        raise ValueError(
            "migration.solver_owners must bind the accepted raw forest; got missing forest or ownership"
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
    parts: MigrationParts,
    saved: AdaptiveSimplexState,
    workset: SimplexNeighborhoodWorkset,
    requested_solver_owners: Array,
    /,
    *,
    target_part_count: int | None = None,
) -> tuple[Array, Array]:
    """Map the graph proposal into raw-slot order without slot guesses.

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
            "migration.graph_restart must match scientific workset rows and target count; got incompatible shape or count"
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


class SimplexGraphRestartProposal(StrictModule, NonTrainableState):
    """An executed native dual-graph partition bound to its raw-slot proposal.

    ``parts`` is the accepted owner-local partition, aligned with
    ``workset.cell_ids``; ``requested_target_owners`` is its scientific-ID projection
    into saved raw-cell order. Restart validation replays the same dual-graph
    lowering, multilevel partition operation and projection from these saved
    logical rows, so a GRAPH ownership claim is never a caller label.
    """

    workset: distribution_owner.SimplexNeighborhoodWorkset
    vertex_weights: Array
    edge_weights: Array
    parts: Array
    requested_target_owners: Array
    maximum_imbalance: float = eqx.field(static=True)

    @property
    def plan(self) -> GraphPartitionPlan:
        """Return the GRAPH plan for the proposal owner count and imbalance tolerance."""
        return GraphPartitionPlan(
            self.parts.shape[0], maximum_imbalance=self.maximum_imbalance
        )


@checked
def prepare_graph_restart_proposal(
    parts: MigrationParts,
    saved: AdaptiveSimplexState,
    state: OwnerLocalMigrationState,
    policy: distribution_owner.MeshPartitionPolicy,
    /,
    *,
    vertex_weights: Array,
    edge_weights: Array | None = None,
) -> SimplexGraphRestartProposal:
    """Execute the native GRAPH route on the dual graph of an accepted epoch.

    The owner-local partition runs on the current execution parts, so the
    requested owner count equals the current one. A collectively refused
    partition or an incomplete scientific projection raises; its measured
    graph evidence is reported, never replaced by incoming ownership.
    """

    if policy.kind is not distribution_owner.MeshPartitionKind.GRAPH:
        raise ValueError(
            f"migration.graph_restart.policy.kind must be GRAPH; got {policy.kind}"
        )
    if policy.part_count != parts.part_count:
        raise ValueError(
            f"migration.graph_restart.policy.part_count must match current execution owners; got requested={policy.part_count}, execution={parts.part_count}"
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
            f"migration.graph_restart.partition must accept the dual graph; got resource_status={evidence.resource_status}, evidence_id={evidence.evidence_id}"
        )
    requested, status = restart_solver_target_owners(
        parts, saved, state.workset, result.parts
    )
    if bool(jax.device_get(jnp.any(status != 0))):
        raise ValueError(
            "migration.graph_restart.requested_target_owners must cover every active raw cell once; got nonzero projection status"
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
    collective: bool = False,
) -> None:
    """Recompute the saved GRAPH proposal exactly on the current devices.

    Cold restore opts into collective local-phase agreement before replay
    kernels and after their numerical digests. Both digests are computed before
    inspecting status, so a local refusal cannot skip another rank's collective.
    """
    from contextlib import nullcontext

    from ..lifecycle._meshing_sources import meshing_source_local_restore_phase

    local_phase = meshing_source_local_restore_phase if collective else nullcontext

    def digest(value: Array) -> str:
        return logical_array_value_collection_digest({"value": value})

    with local_phase():
        work = proposal.workset
        _validate_simplex_graph_replay(work, proposal.edge_weights, axis_name)
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
                "migration.graph_restart must align int32 proposal owners and int64 weights with saved graph and raw-forest rows; got incompatible shapes or dtypes"
            )
        plan = proposal.plan
    vertex_weights, edge_weights = place_logical_simplex_graph_weights(
        work, proposal.vertex_weights, proposal.edge_weights
    )
    offsets, neighbors, weights, valid = replay_logical_simplex_graph(
        work, edge_weights, axis_name=axis_name
    )
    parts, _, measured = replay_logical_partition(
        offsets,
        neighbors,
        weights,
        vertex_weights,
        work.cell_ids,
        work.cell_owner,
        valid,
        plan,
        axis_name=axis_name,
    )
    parts_digest = digest(parts)
    expected_parts_digest = digest(proposal.parts)
    with local_phase():
        if bool(jax.device_get(measured.resource_status != 0)) or (
            parts_digest != expected_parts_digest
        ):
            raise ValueError(
                "migration.graph_restart.parts must match an accepted replayed native partition; got nonzero replay status or different ownership digest"
            )
    requested, status = _replay_logical_solver_owners(
        work,
        parts,
        saved.mesh.cell_ids,
        saved.mesh.cell_active,
        parts.shape[0],
        axis_name,
    )
    requested_digest = digest(requested)
    expected_requested_digest = digest(proposal.requested_target_owners)
    with local_phase():
        if bool(jax.device_get(jnp.any(status != 0))) or (
            requested_digest != expected_requested_digest
        ):
            raise ValueError(
                "migration.graph_restart.requested_target_owners must match its native partition projection; got different raw-slot proposal"
            )


@eqx.filter_jit
def _collective_migration_status(parts: MigrationParts, status: Array) -> Array:
    """Compile the unchanged status-bit union on the explicit execution axis."""
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def collective(block: Array) -> Array:
        return collective_axis_status_union(block[0], axis)[None]

    return partition_map(
        collective,
        parts=parts,
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )(status)


def ownerlocal_solver_forest(
    parts: MigrationParts,
    state: OwnerLocalMigrationState,
    /,
) -> tuple[AdaptiveSimplexState, tuple[Array, ...], tuple[Array, ...], Array]:
    """Receive all family records needed by each solver resident closure.

    This bounded projection is a hierarchy-construction view, never an execution
    ownership replacement. Raw active flags remain global family facts; callers
    select solver labels by their resident scientific IDs. The accepted source
    execution forest and its owners remain unchanged.
    """

    if (
        state.forest is None
        or state.forest_cell_owners is None
        or state.forest_vertex_owners is None
    ):
        raise ValueError(
            "migration.solver_closure must bind the accepted forest and record ownership; got missing forest or owners"
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
    status = _collective_migration_status(parts, status)
    return forest, cells, vertices, status


def ownerlocal_solver_forest_geometry(
    parts: MigrationParts,
    states: AdaptiveSimplexState,
    solver_workset: SimplexNeighborhoodWorkset,
    /,
) -> tuple[AdaptiveSimplexState, Array]:
    """Receive geometry/history families without claiming any physical payloads."""

    if (
        states.mesh.cell_ids.shape[0] != parts.part_count
        or solver_workset.cell_ids.shape[0] != parts.part_count
    ):
        raise ValueError(
            "migration.solver_projection must match forest and solver placements; got incompatible owner count"
        )
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def authority(cell_block: Array, vertex_block: Array) -> tuple[Array, Array]:
        cells, vertices = cell_block[0], vertex_block[0]
        rank = jax.lax.axis_index(axis)
        return jnp.where(cells >= 0, rank, -1).astype(jnp.int32)[
            None
        ], vertex_record_owners(parts, vertices)[None]

    cell_owners, vertex_owners = partition_map(
            authority,
            parts=parts,
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

    status = _collective_migration_status(parts, status)
    return forest, status


@eqx.filter_jit
def _restart_solver_workset_owners(
    raw_ids: Array, solver_owners: Array, work: SimplexNeighborhoodWorkset,
) -> tuple[Array, Array]:
    """Project authenticated raw-slot solver labels by exact scientific cell ID."""
    table = jnp.where(raw_ids >= 0, raw_ids, jnp.iinfo(jnp.int64).max)
    order = jnp.argsort(table, axis=1, stable=True)
    table = jnp.take_along_axis(table, order, axis=1)
    owners = jnp.take_along_axis(solver_owners, order, axis=1)
    positions = jax.vmap(jnp.searchsorted)(table, work.cell_ids)
    safe = jnp.minimum(positions, table.shape[1] - 1)
    found = jnp.take_along_axis(table, safe, axis=1) == work.cell_ids
    selected = jnp.take_along_axis(owners, safe, axis=1)
    own = work.cell_valid & (
        work.cell_owner == jnp.arange(raw_ids.shape[0], dtype=jnp.int32)[:, None]
    )
    valid = jnp.all(~own | (found & (selected >= 0) & (selected < raw_ids.shape[0])))
    return jnp.where(own, selected, work.cell_owner).astype(jnp.int32), valid


def restart_solver_family_view(
    partitioned: PartitionedAdaptiveSimplex,
    states: AdaptiveSimplexState,
    /,
) -> tuple[SimplexNeighborhoodWorkset, AdaptiveSimplexState]:
    """Reconstruct solver-family geometry without changing physical restart state."""
    with agreed_local_step(stage="restart-solver-closure"):
        receipt, proof = partitioned.restart_repack, partitioned.restart_proof
        policy = partitioned.prepared.adaptation.policy.distribution
        if receipt is None or proof is None or policy is None or partitioned.migration is not None:
            raise ValueError("restart solver closure requires its explicit accepted restart transaction")
        type(proof).require_receipt(proof, receipt)
    raw_workset = distribution_owner.expand_partitioned_simplex_neighborhood(
        partitioned.parts, states, partitioned.states, partitioned.source_exterior,
        halo_width=policy.halo_width,
        cell_capacity=partitioned.layout.cell_capacity,
        vertex_capacity=partitioned.layout.vertex_capacity,
    )
    targets, valid = _restart_solver_workset_owners(
        states.mesh.cell_ids, receipt.solver_cell_owners, raw_workset,
    )
    with agreed_local_step(stage="restart-solver-closure"):
        if not bool(jax.device_get(valid)):
            raise ValueError("restart solver labels must cover every current scientific cell")
    workset, solver_parts, status = relocate_solver_workset(
        partitioned.parts, raw_workset, targets, halo_width=policy.halo_width,
        vertex_capacity=partitioned.layout.vertex_capacity,
    )
    with agreed_local_step(stage="restart-solver-closure"):
        if bool(jax.device_get(jnp.any(status != 0))):
            raise ValueError("restart solver geometry transport must be accepted by every owner")
    forest, status = ownerlocal_solver_forest_geometry(solver_parts, states, workset)
    workset, family_status = _complete_solver_family_neighborhood(
        solver_parts, workset, forest, partitioned.layout.vertex_capacity,
    )
    with agreed_local_step(stage="restart-solver-closure"):
        if bool(jax.device_get(jnp.any((status | family_status) != 0))):
            raise ValueError("restart solver families must cover the accepted solver neighborhood")
    return workset, forest


def relocate_solver_workset(
    parts: MigrationParts,
    raw_workset: SimplexNeighborhoodWorkset,
    requested_solver_owners: Array,
    /,
    *,
    halo_width: int,
    vertex_capacity: int,
    route_metadata_capacity_bytes: int = 1 << 20,
) -> tuple[SimplexNeighborhoodWorkset, MigrationParts, Array]:
    """Stage geometry-only solver ownership/closure using the migration core.

    No physical field or epoch payload exists on this path. Nothing is published:
    the caller must reject the epoch on any returned nonzero status.
    Target rows align with raw_workset.cell_ids; only raw-owned rows send.
    """

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
            "migration.geometry_relocation must bind int64 geometry IDs, int32 owners, bool validity, matched execution rows, and positive capacities; got incompatible packet fields or capacities"
        )
    if (
        isinstance(route_metadata_capacity_bytes, bool)
        or budget < parts.part_count * parts.part_count
    ):
        raise ValueError(
            f"migration.geometry_relocation.route_metadata_capacity_bytes must cover the rank routing matrix; got capacity={route_metadata_capacity_bytes!r}, required={parts.part_count * parts.part_count}"
        )
    owned, _, _, _, _ = _compiled_owned_packets(
        parts, raw_workset, requested_solver_owners, ()
    )
    target_parts = prepare_migration_routes(
        parts, _compiled_migration_routes(parts, owned)
    )
    closure = distribution_owner.expand_simplex_neighborhood_workset(
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
        status = collective_axis_status_union(
            work.status | jnp.where(counts[0] > vertices, _CAPACITY, 0), axis
        )
        return jax.tree_util.tree_map(
            lambda value: value[None], _replace_work(work, status=status)
        )

    staged = partition_map(
            finish,
            parts=parts,
            in_specs=(spec, spec),
            out_specs=spec,
            check_vma=False,
        )(
        hydrated,
        vertex_counts,
    )
    return staged, target_parts, staged.status


class _MigrationForest(StrictModule):
    state: AdaptiveSimplexState
    cell_owners: Array
    vertex_owners: Array


class _MigrationPacketInputs(StrictModule):
    halo: int = eqx.field(static=True)
    vertices: int = eqx.field(static=True)
    route_bytes: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)


class _MigrationInputs(StrictModule):
    halo: int = eqx.field(static=True)
    vertices: int = eqx.field(static=True)
    route_bytes: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    forest: _MigrationForest | None


class _MigrationPackets(StrictModule):
    owned: OwnerLocalMigrationState
    candidate: OwnerLocalMigrationState
    target_parts: MigrationParts
    before: Array
    sent: Array
    received: Array
    vertex_counts: Array


class _MigrationForestStage(StrictModule):
    candidate: OwnerLocalMigrationState
    parts: MigrationParts | None


def _prepare_migration_packet_inputs(
    parts: MigrationParts,
    original: OwnerLocalMigrationState,
    requested_target_owners: Array,
    /,
    *,
    halo_width: int,
    vertex_capacity: int,
    route_metadata_capacity_bytes: int = 1 << 20,
) -> _MigrationPacketInputs:
    halo = operator.index(halo_width)
    vertices = operator.index(vertex_capacity)
    route_bytes = (
        parts.part_count * parts.part_count * (2 if original.forest is not None else 1)
    )
    route_budget = operator.index(route_metadata_capacity_bytes)
    if isinstance(route_metadata_capacity_bytes, bool) or route_budget < route_bytes:
        raise ValueError(
            f"migration.route_metadata_capacity_bytes must cover solver and retained-family routing matrices; got capacity={route_metadata_capacity_bytes!r}, required={route_bytes}"
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
            "migration.packet_inputs must bind matched ID, owner, epoch, field, and source-support axes with explicit dtypes and valid capacities; got incompatible packet metadata or capacities"
        )
    return _MigrationPacketInputs(halo, vertices, route_bytes, shape[1])


def _prepare_migration_history(
    parts: MigrationParts, original: OwnerLocalMigrationState, /
) -> _MigrationForest | None:
    if original.forest is None:
        if (
            original.forest_cell_owners is not None
            or original.forest_vertex_owners is not None
            or original.cell_history
            or original.vertex_history
        ):
            raise ValueError(
                "migration.forest must be present when record ownership or history is supplied; got None with retained ownership or history"
            )
        return None
    forest = original.forest
    cell_owners, vertex_owners = (
        original.forest_cell_owners,
        original.forest_vertex_owners,
    )
    if cell_owners is None or vertex_owners is None:
        raise ValueError(
            "migration.forest_owners must assign cell and vertex records; got missing ownership arrays"
        )
    if (
        forest.mesh.cell_ids.shape[0] != parts.part_count
        or cell_owners.shape != forest.mesh.cell_ids.shape
        or cell_owners.dtype != jnp.int32
        or vertex_owners.shape != forest.mesh.vertex_ids.shape
        or vertex_owners.dtype != jnp.int32
        or any(value.shape[:2] != cell_owners.shape for value in original.cell_history)
        or any(
            value.shape[:2] != vertex_owners.shape for value in original.vertex_history
        )
    ):
        raise ValueError(
            "migration.forest_history must align int32 record ownership and history leaves with raw part capacities; got incompatible owner dtype or history shape"
        )
    return _MigrationForest(forest, cell_owners, vertex_owners)


def _stage_migration_packets(
    parts: MigrationParts,
    original: OwnerLocalMigrationState,
    requested_target_owners: Array,
    inputs: _MigrationInputs,
    /,
) -> _MigrationPackets:
    halo, vertices = inputs.halo, inputs.vertices
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
    target_parts = prepare_migration_routes(parts, route_mask)
    closure = distribution_owner.expand_simplex_neighborhood_workset(
        target_parts, owned.workset, halo_width=halo, vertex_capacity=vertices
    )
    candidate, vertex_counts = _hydrate_closure(target_parts, owned, closure)
    return _MigrationPackets(
        owned, candidate, target_parts, before, sent, received, vertex_counts
    )


def _stage_migration_forest(
    parts: MigrationParts,
    original: OwnerLocalMigrationState,
    inputs: _MigrationInputs,
    packets: _MigrationPackets,
    /,
) -> _MigrationForestStage:
    owned, candidate = packets.owned, packets.candidate
    forest_parts = None
    if inputs.forest is not None:
        (
            forest,
            cell_owners,
            vertex_owners,
            cell_history,
            vertex_history,
            forest_status,
        ) = migrate_simplex_forest(
            parts,
            inputs.forest.state,
            inputs.forest.cell_owners,
            inputs.forest.vertex_owners,
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
        forest_parts = prepare_migration_routes(
            parts, forest_migration_routes(parts, forest)
        )
    return _MigrationForestStage(candidate, forest_parts)


@eqx.filter_jit
def _finalize_migration_status(
    parts: MigrationParts,
    status: Array,
    vertex_counts: Array,
    vertices: int,
    /,
) -> Array:
    """Reduce hydration refusal without mapping unrelated forest/history leaves."""
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def finish(status_block: Array, vertex_block: Array) -> Array:
        status = status_block[0] | jnp.where(vertex_block[0] > vertices, _CAPACITY, 0)
        return collective_axis_status_union(status, axis)[None]

    return partition_map(
        finish,
        parts=parts,
        in_specs=(spec, spec),
        out_specs=spec,
        check_vma=False,
    )(status, vertex_counts)


def _prepare_migration_evidence(
    parts: MigrationParts,
    original: OwnerLocalMigrationState,
    candidate: OwnerLocalMigrationState,
    inputs: _MigrationInputs,
    packets: _MigrationPackets,
    /,
) -> OwnerLocalMigrationEvidence:
    forest_ids = None
    if original.forest is not None and candidate.forest is not None:
        forest_ids = (
            original.forest.mesh.cell_ids,
            candidate.forest.mesh.cell_ids,
            original.forest.mesh.vertex_ids,
            candidate.forest.mesh.vertex_ids,
        )
    return _prepare_migration_evidence_values(
        parts.part_count,
        candidate.workset.cell_valid,
        candidate.workset.cell_owner,
        candidate.workset.status,
        forest_ids,
        packets.before,
        packets.sent,
        packets.received,
        packets.vertex_counts,
        inputs.halo,
        inputs.cell_capacity,
        inputs.vertices,
        inputs.route_bytes,
    )


@eqx.filter_jit
def _prepare_migration_evidence_values(
    part_count: int,
    cell_valid: Array,
    cell_owner: Array,
    status: Array,
    forest_ids: tuple[Array, Array, Array, Array] | None,
    before: Array,
    sent: Array,
    received: Array,
    vertex_counts: Array,
    halo: int,
    cell_capacity: int,
    vertices: int,
    route_bytes: int,
) -> OwnerLocalMigrationEvidence:
    """Compute every evidence count from arrays without eager leaf programs."""
    rank = jnp.arange(part_count, dtype=jnp.int32)[:, None]
    after = jnp.sum(
        cell_valid & (cell_owner == rank),
        axis=1,
        dtype=jnp.int32,
    )
    resident = jnp.sum(cell_valid, axis=1, dtype=jnp.int32)
    forest_before = jnp.zeros(part_count, jnp.int32)
    forest_after = jnp.zeros_like(forest_before)
    forest_vertices_before = jnp.zeros_like(forest_before)
    forest_vertices_after = jnp.zeros_like(forest_before)
    forest_cell_capacity = forest_vertex_capacity = 0
    if forest_ids is not None:
        before_cells, after_cells, before_vertices, after_vertices = forest_ids
        forest_before = jnp.sum(before_cells >= 0, axis=1, dtype=jnp.int32)
        forest_after = jnp.sum(after_cells >= 0, axis=1, dtype=jnp.int32)
        forest_vertices_before = jnp.sum(before_vertices >= 0, axis=1, dtype=jnp.int32)
        forest_vertices_after = jnp.sum(after_vertices >= 0, axis=1, dtype=jnp.int32)
        forest_cell_capacity = before_cells.shape[1]
        forest_vertex_capacity = before_vertices.shape[1]
    return OwnerLocalMigrationEvidence(
        status,
        status == 0,
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
        cell_capacity,
        vertices,
        forest_cell_capacity,
        forest_vertex_capacity,
        route_bytes,
    )


@checked
def prepare_ownerlocal_migration(
    parts: MigrationParts,
    original: OwnerLocalMigrationState,
    requested_target_owners: Array,
    /,
    *,
    halo_width: int,
    vertex_capacity: int,
    route_metadata_capacity_bytes: int = 1 << 20,
) -> PreparedOwnerLocalMigration:
    """Prepare geometry/state migration and requested ghost closure.

    Target owners are int32 [parts,C] in the workset's semantic cell-ID order;
    only source-owned rows send. A retained raw forest carries explicit
    recorded cell/vertex ranks and aligned dtype-preserving history leaves.

    ``route_metadata_capacity_bytes`` bounds the host prepare-barrier reads:
    one replicated bool [parts,parts] solver-routing matrix, plus a separate
    family-execution matrix when history is retained. Geometry, field content
    and ancestry remain in capacity-sized device packets.
    """

    capacity = _prepare_migration_packet_inputs(
        parts,
        original,
        requested_target_owners,
        halo_width=halo_width,
        vertex_capacity=vertex_capacity,
        route_metadata_capacity_bytes=route_metadata_capacity_bytes,
    )
    retained_forest = _prepare_migration_history(parts, original)
    inputs = _MigrationInputs(
        capacity.halo,
        capacity.vertices,
        capacity.route_bytes,
        capacity.cell_capacity,
        retained_forest,
    )
    packets = _stage_migration_packets(parts, original, requested_target_owners, inputs)
    forest = _stage_migration_forest(parts, original, inputs, packets)
    status = _finalize_migration_status(
        parts, forest.candidate.workset.status, packets.vertex_counts, inputs.vertices
    )
    candidate = eqx.tree_at(
        lambda state: state.workset,
        forest.candidate,
        _replace_work(forest.candidate.workset, status=status),
    )
    evidence = _prepare_migration_evidence(parts, original, candidate, inputs, packets)
    return PreparedOwnerLocalMigration(
        original, candidate, evidence, packets.target_parts, forest.parts
    )


def accept_ownerlocal_migration(
    prepared: PreparedOwnerLocalMigration, /
) -> OwnerLocalMigrationState:
    """Atomically publish arrays; only operands enter the compiled selection."""
    return _accept_ownerlocal_migration_values(
        prepared.original, prepared.candidate, prepared.evidence.accepted
    )


@eqx.filter_jit
def _accept_ownerlocal_migration_values(
    original: OwnerLocalMigrationState,
    candidate: OwnerLocalMigrationState,
    accepted: Array,
) -> OwnerLocalMigrationState:

    def select(original: Array, candidate: Array) -> Array:
        mask = accepted.reshape((accepted.shape[0],) + (1,) * (original.ndim - 1))
        return jnp.where(mask, candidate, original)

    return jax.tree_util.tree_map(select, original, candidate)


class PartitionedEpochMigration(StrictModule, NonTrainableState):
    """Aligned source roots and current families from an accepted bounded migration.

    The source preparation is retained unchanged. Solver ownership follows the
    requested scientific-cell proposal; physical execution keeps each raw family
    together. Initial records occupy the same prefix slots as the current forest.
    """

    source_partitioned: PartitionedAdaptiveSimplex
    saved_states: AdaptiveSimplexState
    saved_migration_state: OwnerLocalMigrationState
    requested_target_owners: Array
    states: AdaptiveSimplexState
    initial_states: AdaptiveSimplexState
    source_exterior: Array
    slot_origins: Array
    parts: MigrationParts
    solver_cell_owners: Array
    initial_solver_cell_owners: Array
    solver_neighborhood: SimplexNeighborhoodWorkset
    solver_forest: AdaptiveSimplexState
    solver_initial_states: AdaptiveSimplexState
    solver_source_exterior: Array
    solver_slot_origins: Array
    migration: PreparedOwnerLocalMigration


def _partitioned_epoch_payloads(
    partitioned: PartitionedAdaptiveSimplex,
    states: AdaptiveSimplexState,
    migration_state: OwnerLocalMigrationState,
) -> tuple[tuple[Array, ...], tuple[Array, ...], Array]:
    """Bind only numerical source records, not the preparation/predecessor DAG."""
    forest = migration_state.forest
    if forest is None:
        raise ValueError(
            "PartitionedEpochMigration.forest must retain the current raw epoch; got none"
        )
    return _partitioned_epoch_payload_values(
        partitioned.parts,
        partitioned.states,
        states,
        forest,
        partitioned.source_exterior,
        jnp.asarray(partitioned.slot_origins, dtype=jnp.int64),
        partitioned.solver_cell_owners,
    )


@eqx.filter_jit
def _partitioned_epoch_payload_values(
    parts: MigrationParts,
    initial: AdaptiveSimplexState,
    states: AdaptiveSimplexState,
    forest: AdaptiveSimplexState,
    source_exterior: Array,
    slot_origins: Array,
    solver_cell_owners: Array,
) -> tuple[tuple[Array, ...], tuple[Array, ...], Array]:
    """Encode initial records as stable-ID histories on the current raw forest."""
    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(
        initial_block: AdaptiveSimplexState,
        current_block: AdaptiveSimplexState,
        bound_block: AdaptiveSimplexState,
        exterior_block: Array,
        origins_block: Array,
        solver_block: Array,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...], Array]:
        initial, current, bound, exterior, origins, solver = jax.tree_util.tree_map(
            lambda value: value[0],
            (
                initial_block,
                current_block,
                bound_block,
                exterior_block,
                origins_block,
                solver_block,
            ),
        )
        cells, vertices, status = restart_distribution_owner.packet(
            initial, (exterior, origins, solver), ()
        )
        # JAX reconstructs dict PyTrees in sorted-key order. The transported
        # tuple must retain the packet schema's order across the ring loop.
        vertex_names = tuple(vertices)
        roots, descendants, midpoints, _ = distribution_owner.subdivision_checks(
            current, initial, exterior
        )
        same_current = jnp.asarray(True)
        for left, right in zip(
            jax.tree_util.tree_leaves(current),
            jax.tree_util.tree_leaves(bound),
            strict=True,
        ):
            same_current &= jnp.array_equal(left, right)
        status |= jnp.where(roots & descendants & midpoints & same_current, 0, _INVALID)
        # Source vertices can have several current copies. Query source packets
        # before choosing the current vertex record owner, so a copy introduced
        # by closure cannot discard the source vertex's initial history.
        requested = current.mesh.vertex_ids
        result = {
            name: jnp.full_like(value, -1) if name == "ids" else jnp.zeros_like(value)
            for name, value in vertices.items()
        }
        covered = jnp.zeros(requested.shape, jnp.bool_)

        def visit(
            _: int,
            carry: tuple[dict[str, Array], dict[str, Array], Array, Array],
        ) -> tuple[dict[str, Array], dict[str, Array], Array, Array]:
            local_packet, result, covered, status = carry
            order = jnp.argsort(
                jnp.where(
                    local_packet["ids"] >= 0,
                    local_packet["ids"],
                    _SENTINEL,
                ),
                stable=True,
            )
            keys = jnp.where(
                local_packet["ids"][order] >= 0,
                local_packet["ids"][order],
                _SENTINEL,
            )
            position = jnp.minimum(
                jnp.searchsorted(keys, requested), requested.shape[0] - 1
            )
            match = (requested >= 0) & (keys[position] == requested)
            conflict = jnp.asarray(False)
            for name, value in local_packet.items():
                incoming = value[order[position]]
                different = incoming != result[name]
                if different.ndim > 1:
                    different = jnp.any(different, axis=tuple(range(1, different.ndim)))
                conflict |= jnp.any(covered & match & different)
                mask = match.reshape(match.shape + (1,) * (incoming.ndim - 1))
                result[name] = jnp.where(mask, incoming, result[name])
            return (
                jax.tree_util.tree_map(
                    lambda value: jax.lax.ppermute(value, axis, ring),
                    local_packet,
                ),
                result,
                covered | match,
                status | jnp.where(conflict, _INVALID, 0),
            )

        _, vertices, _, status = jax.lax.fori_loop(
            0, parts.part_count, visit, (vertices, result, covered, status)
        )
        status = collective_axis_status_union(status, axis)
        return (
            tuple(value[None] for value in cells.values()),
            tuple(vertices[name][None] for name in vertex_names),
            status[None],
        )

    # The host binding checks forest presence before entering this program.
    return partition_map(
            local,
            parts=parts,
            in_specs=(spec, spec, spec, spec, spec, spec),
            out_specs=(spec, spec, spec),
            check_vma=False,
        )(
        initial,
        states,
        forest,
        source_exterior,
        slot_origins,
        solver_cell_owners,
    )


def _reconstruct_partitioned_initial(
    partitioned: PartitionedAdaptiveSimplex,
    accepted: OwnerLocalMigrationState,
    cell_payload_count: int,
    vertex_payload_count: int,
    *,
    include_ghost_families: bool = False,
) -> tuple[AdaptiveSimplexState, Array, Array, Array, Array, Array]:
    """Bind source schema and accepted histories without static owner capture."""
    forest = accepted.forest
    if forest is None:
        raise ValueError(
            "PartitionedEpochMigration.forest must retain accepted raw families; got none"
        )
    return _reconstruct_partitioned_initial_values(
        partitioned.parts,
        partitioned.states,
        forest,
        accepted.cell_history[-cell_payload_count:],
        accepted.vertex_history[-vertex_payload_count:],
        partitioned.source_exterior,
        jnp.asarray(partitioned.slot_origins, dtype=jnp.int64),
        partitioned.solver_cell_owners,
        include_ghost_families=include_ghost_families,
    )


@eqx.filter_jit
def _reconstruct_partitioned_initial_values(
    parts: MigrationParts,
    initial: AdaptiveSimplexState,
    forest: AdaptiveSimplexState,
    cell_history: tuple[Array, ...],
    vertex_history: tuple[Array, ...],
    source_exterior: Array,
    slot_origins: Array,
    solver_cell_owners: Array,
    *,
    include_ghost_families: bool = False,
) -> tuple[AdaptiveSimplexState, Array, Array, Array, Array, Array]:
    """Decode source records into the accepted current forest's slot numbering."""
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def local(
        template_block: AdaptiveSimplexState,
        current_block: AdaptiveSimplexState,
        cell_block: tuple[Array, ...],
        vertex_block: tuple[Array, ...],
        old_exterior_block: Array,
        old_origins_block: Array,
        old_solver_block: Array,
    ) -> tuple[AdaptiveSimplexState, Array, Array, Array, Array, Array]:
        template, current, ch, vh, old_exterior, old_origins, old_solver = (
            jax.tree_util.tree_map(
                lambda value: value[0],
                (
                    template_block,
                    current_block,
                    cell_block,
                    vertex_block,
                    old_exterior_block,
                    old_origins_block,
                    old_solver_block,
                ),
            )
        )
        cell_schema, vertex_schema, _ = restart_distribution_owner.packet(
            template, (old_exterior, old_origins, old_solver), ()
        )
        cells = dict(zip(cell_schema, ch, strict=True))
        vertices = dict(zip(vertex_schema, vh, strict=True))
        cell_ids = jnp.where(current.mesh.cell_ids >= 0, cells["ids"], -1)
        vertex_ids = jnp.where(current.mesh.vertex_ids >= 0, vertices["ids"], -1)
        used, vertex_used = cell_ids >= 0, vertex_ids >= 0
        c, v = cell_ids.shape[0], vertex_ids.shape[0]
        cell_count = jnp.sum(used, dtype=jnp.int64)
        vertex_count = jnp.sum(vertex_used, dtype=jnp.int64)
        checks = [
            ~jnp.any(used != (jnp.arange(c) < cell_count)),
            ~jnp.any(vertex_used != (jnp.arange(v) < vertex_count)),
            ~jnp.any(used & (cell_ids != current.mesh.cell_ids)),
            ~jnp.any(vertex_used & (vertex_ids != current.mesh.vertex_ids)),
        ]
        # Only the physical execution forest owns the global once-count.
        # Solver closure replicates all family records, with unchanged
        # scientific payloads and independently checked local slot alignment.
        conserved = jnp.asarray(True)
        if not include_ghost_families:
            conserved = jax.lax.psum(cell_count, axis) == jax.lax.psum(
                jnp.sum(template.mesh.cell_ids >= 0, dtype=jnp.int64), axis
            )
        checks.append(conserved)
        missing_vertices = jnp.asarray(False)
        for name in ("cells", "tuples", "bisection"):
            missing_vertices |= restart_distribution_owner.missing(
                cells[name], vertex_ids, used
            )
        missing_cells = jnp.asarray(False)
        for name in ("parents", "children"):
            missing_cells |= restart_distribution_owner.missing(
                cells[name], cell_ids, used
            )
        checks.extend(
            (
                ~missing_vertices,
                ~missing_cells,
                ~restart_distribution_owner.missing(
                    vertices["parents"], vertex_ids, vertex_used
                ),
            )
        )
        rows = jnp.maximum(slots(vertex_ids, cells["cells"]), 0)
        active = used & cells["active"]
        neighbors = masked_simplex_facet_neighbors(rows, active)
        mesh = eqx.tree_at(
            lambda value: (
                value.coordinates,
                value.vertex_ids,
                value.vertex_active,
                value.cells,
                value.cell_ids,
                value.cell_active,
                value.facet_neighbors,
            ),
            template.mesh,
            (
                vertices["coordinates"],
                vertex_ids,
                vertex_used & vertices["active"],
                rows,
                cell_ids,
                active,
                neighbors,
            ),
        )
        parent_slots = slots(cell_ids, cells["parents"])
        initial = restore_forest_records(
            template,
            cells,
            vertices,
            mesh=mesh,
            tuples=jnp.maximum(slots(vertex_ids, cells["tuples"]), 0),
            parents=jnp.where(
                parent_slots >= 0, parent_slots * 2 + cells["ordinal"], -1
            ).astype(jnp.int32),
            children=slots(cell_ids, cells["children"]),
            bisection_vertices=slots(vertex_ids, cells["bisection"]),
            vertex_parents=slots(vertex_ids, vertices["parents"]),
            protected_codes=current.protected_codes,
            vertex_half_facets=vertex_half_facets(rows, active, neighbors, v),
            cursors=jnp.stack(
                (
                    vertex_count,
                    cell_count,
                    jax.lax.pmax(template.cursors[2], axis),
                    jax.lax.pmax(template.cursors[3], axis),
                )
            ),
        )
        exterior = jnp.where(used[:, None], cells["payload_0"], False)
        origins = jnp.where(used, cells["payload_1"], -1)
        solver = jnp.where(active, cells["payload_2"], -1).astype(jnp.int32)
        roots, descendants, midpoints, _ = distribution_owner.subdivision_checks(
            current, initial, exterior
        )
        checks.extend((roots, descendants, midpoints))
        records = (current.mesh.cell_ids >= 0) & ~current.retired
        tuple_slots = jnp.clip(current.tuples, 0, v - 1)
        tuple_ids = current.mesh.vertex_ids[tuple_slots]
        tuple_valid = jnp.all(
            (current.tuples >= 0)
            & (current.tuples < current.cursors[0])
            & (tuple_ids >= 0),
            axis=1,
        )
        tuple_valid &= ~current.mesh.cell_active | jnp.all(
            current.mesh.vertex_active[tuple_slots], axis=1
        )
        ordered_tuples = jnp.sort(tuple_ids, axis=1)
        ordered_cells = jnp.sort(current.mesh.vertex_ids[current.mesh.cells], axis=1)
        tuple_vertices = jnp.all(ordered_tuples[:, 1:] > ordered_tuples[:, :-1], axis=1)
        tuple_vertices &= jnp.all(ordered_tuples == ordered_cells, axis=1)
        checks.extend(
            (
                jnp.all(~records | tuple_valid),
                jnp.all(~records | tuple_vertices),
            )
        )
        validity = jnp.stack(checks)
        status = collective_axis_status_union(
            jnp.where(jnp.all(validity), 0, _INVALID), axis
        )
        return jax.tree_util.tree_map(
            lambda value: value[None],
            (initial, exterior, origins, solver, status, validity),
        )

    # The host binding checks forest presence before entering this program.
    return partition_map(
            local,
            parts=parts,
            in_specs=(spec, spec, spec, spec, spec, spec, spec),
            out_specs=(spec, spec, spec, spec, spec, spec),
            check_vma=False,
        )(
        initial,
        forest,
        cell_history,
        vertex_history,
        source_exterior,
        slot_origins,
        solver_cell_owners,
    )


@eqx.filter_jit
def _complete_solver_family_neighborhood(
    parts: MigrationParts,
    work: SimplexNeighborhoodWorkset,
    solver_forest: AdaptiveSimplexState,
    vertex_capacity: int,
) -> tuple[SimplexNeighborhoodWorkset, Array]:
    """Hydrate every active solver-family row from accepted scientific packets."""
    if solver_forest.mesh.cell_ids.shape != work.cell_ids.shape:
        raise ValueError(
            "PartitionedEpochMigration.solver_neighborhood must share bounded family capacity; got "
            f"{solver_forest.mesh.cell_ids.shape}, {work.cell_ids.shape}"
        )
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def request(
        forest_block: AdaptiveSimplexState,
        work_block: SimplexNeighborhoodWorkset,
    ) -> SimplexNeighborhoodWorkset:
        forest, template = jax.tree_util.tree_map(
            lambda value: value[0], (forest_block, work_block)
        )
        active = forest.mesh.cell_active & (forest.mesh.cell_ids >= 0)
        order = jnp.argsort(
            jnp.where(active, forest.mesh.cell_ids, _SENTINEL), stable=True
        )
        valid = active[order]
        rows = forest.mesh.cells[order]
        closure = _replace_work(
            template,
            cell_ids=jnp.where(valid, forest.mesh.cell_ids[order], _SENTINEL),
            cell_valid=valid,
            cell_vertices=forest.mesh.vertex_ids[rows],
            cell_coordinates=forest.mesh.coordinates[rows],
            status=template.status | forest.clocks[2],
        )
        return jax.tree_util.tree_map(lambda value: value[None], closure)

    closure = partition_map(
            request,
            parts=parts,
            in_specs=(spec, spec),
            out_specs=spec,
            check_vma=False,
        )(solver_forest, work)
    # The same bounded query hydrates geometry, organization, solver ownership,
    # exterior flags, and source supports. Only semantic IDs select the rows.
    arrays = {
        name: value
        for name, value in work.arrays().items()
        if name not in ("cell_ids", "cell_valid", "status")
    }
    names = tuple(arrays)
    hydrated, values, _, vertex_counts = _compiled_hydrate_packets(
        parts, work, tuple(arrays.values()), closure
    )
    hydrated = _replace_work(hydrated, **dict(zip(names, values, strict=True)))

    def finish(
        work_block: SimplexNeighborhoodWorkset, count_block: Array
    ) -> SimplexNeighborhoodWorkset:
        local = jax.tree_util.tree_map(lambda value: value[0], work_block)
        status = collective_axis_status_union(
            local.status | jnp.where(count_block[0] > vertex_capacity, _CAPACITY, 0),
            axis,
        )
        return jax.tree_util.tree_map(
            lambda value: value[None], _replace_work(local, status=status)
        )

    result = partition_map(
            finish,
            parts=parts,
            in_specs=(spec, spec),
            out_specs=spec,
            check_vma=False,
        )(hydrated, vertex_counts)
    return result, result.status


@eqx.filter_jit
def _simplex_predecessor_witness_valid(
    parts: MigrationParts,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    cell_ids: Array,
    cell_valid: Array,
    source_cell_ids: Array,
    support: tuple[Array, Array, Array],
    /,
) -> Array:
    """Reprove complete ID lists from raw numerical ancestry, including ghosts.

    Only raw IDs, masks and compact predecessor lists circulate. Packets retain
    their bounded local extent on physical and historical logical execution;
    neither geometry nor a global numerical host bank participates.
    """
    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )
    source_width = source_cell_ids.shape[2]

    def local(
        state_block: AdaptiveSimplexState,
        initial_block: AdaptiveSimplexState,
        ids_block: Array,
        valid_block: Array,
        sources_block: Array,
        support_block: tuple[Array, Array, Array],
    ) -> Array:
        state, initial = jax.tree.map(
            lambda value: value[0], (state_block, initial_block)
        )
        ids, valid, sources = ids_block[0], valid_block[0], sources_block[0]
        support = (support_block[0][0], support_block[1][0], support_block[2][0])
        expected = distribution_owner._simplex_predecessor_ids(
            state, initial, support, source_width
        )
        invalid = jnp.any(
            state.mesh.cell_active & (support[2] == 0)
        )
        # Validity breaks sentinel ties so an actual selected MAX ID cannot be
        # confused with unused padding. ID values are never validity predicates.
        order = jnp.lexsort(
            (
                ~state.mesh.cell_active,
                jnp.where(state.mesh.cell_active, state.mesh.cell_ids, _SENTINEL),
            )
        )
        raw_valid = state.mesh.cell_active[order]
        raw_ids = jnp.where(raw_valid, state.mesh.cell_ids[order], _SENTINEL)
        invalid |= jnp.any(
            raw_valid[1:] & raw_valid[:-1] & (raw_ids[1:] == raw_ids[:-1])
        )
        work_order = jnp.lexsort((~valid, jnp.where(valid, ids, _SENTINEL)))
        work_valid = valid[work_order]
        work_ids = jnp.where(work_valid, ids[work_order], _SENTINEL)

        def visit(
            _: Array, carry: tuple[Array, Array, Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array, Array, Array]:
            packet_ids, packet_valid, packet_sources, packet_covered, covered, invalid = (
                carry
            )
            position = jnp.minimum(
                jnp.searchsorted(packet_ids, ids), packet_ids.shape[0] - 1
            )
            match = valid & packet_valid[position] & (packet_ids[position] == ids)
            invalid |= jnp.any(
                match & jnp.any(packet_sources[position] != sources, axis=1)
            )
            covered += match.astype(jnp.int32)
            position = jnp.minimum(
                jnp.searchsorted(work_ids, packet_ids), work_ids.shape[0] - 1
            )
            packet_covered |= (
                packet_valid
                & work_valid[position]
                & (work_ids[position] == packet_ids)
            )
            packet = jax.tree.map(
                lambda value: jax.lax.ppermute(value, axis, ring),
                (packet_ids, packet_valid, packet_sources, packet_covered),
            )
            return (*packet, covered, invalid)

        _, raw_valid, _, raw_covered, covered, invalid = jax.lax.fori_loop(
            0,
            parts.part_count,
            visit,
            (
                raw_ids,
                raw_valid,
                expected[order],
                jnp.zeros(raw_valid.shape, dtype=jnp.bool_),
                jnp.zeros(valid.shape, dtype=jnp.int32),
                invalid,
            ),
        )
        invalid |= jnp.any(valid & (covered != 1))
        invalid |= jnp.any(raw_valid & ~raw_covered)
        return jax.lax.pmin((~invalid).astype(jnp.int32), axis).astype(jnp.bool_)

    return partition_map(
        local,
        parts=parts,
        in_specs=(spec, spec, spec, spec, spec, spec),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(states, initial_states, cell_ids, cell_valid, source_cell_ids, support)


def validate_simplex_predecessor_witness(
    parts: MigrationParts,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    workset: SimplexNeighborhoodWorkset,
    /,
) -> None:
    """Require independently reconstructed complete canonical predecessor lists.

    A payload digest or self-consistent saved migration ID does not authenticate
    ancestry. Actual raw support determines the collective width and every list;
    all supplied valid rows, including remote ghosts, must match exactly.
    """
    shape = states.mesh.cell_ids.shape
    if (
        len(shape) != 2
        or shape[0] != parts.part_count
        or shape[1] == 0
        or initial_states.mesh.cell_ids.shape != shape
        or states.mesh.cells.shape[:2] != shape
        or initial_states.mesh.cells.shape != states.mesh.cells.shape
        or states.mesh.cell_active.shape != shape
        or initial_states.mesh.cell_active.shape != shape
        or states.parents.shape != shape
        or workset.cell_ids.ndim != 2
        or workset.cell_ids.shape[0] != parts.part_count
        or workset.cell_ids.shape[1] == 0
        or workset.cell_ids.dtype != jnp.int64
        or workset.cell_valid.shape != workset.cell_ids.shape
        or workset.cell_valid.dtype != jnp.bool_
        or workset.source_cell_ids.ndim != 3
        or workset.source_cell_ids.shape[:2] != workset.cell_ids.shape
        or workset.source_cell_ids.shape[2] == 0
        or workset.source_cell_ids.dtype != jnp.int64
    ):
        raise ValueError(
            "migration.predecessor must bind matching raw ancestry and positive int64 ID-list packets; got incompatible capacities, masks, or dtypes"
        )
    support, maximum = distribution_owner._prepare_partitioned_simplex_predecessors(
        parts, states, initial_states
    )
    if int(jax.device_get(maximum)) != workset.source_cell_ids.shape[2]:
        raise ValueError(
            "migration.predecessor width must equal the actual complete collective support maximum; got a changed predecessor width"
        )
    valid = _simplex_predecessor_witness_valid(
        parts,
        states,
        initial_states,
        workset.cell_ids,
        workset.cell_valid,
        workset.source_cell_ids,
        support,
    )
    if not bool(jax.device_get(valid)):
        raise ValueError(
            "migration.predecessor must reproduce the actual complete collective width, canonical ID lists, multiplicity and raw-cell coverage; got changed or incomplete predecessor evidence"
        )


def migrate_partitioned_epoch(
    partitioned: PartitionedAdaptiveSimplex,
    states: AdaptiveSimplexState,
    migration_state: OwnerLocalMigrationState,
    requested_target_owners: Array,
    /,
) -> PartitionedEpochMigration:
    """Migrate current families and their prepared source records atomically.

    Target rows belong to ``migration_state.workset``. A family executes on its
    smallest active proposed owner, while each active solver row retains its
    requested owner. Rejection raises before any migrated epoch is returned.
    """
    policy = partitioned.prepared.adaptation.policy.distribution
    if policy is None:
        raise ValueError(
            "PartitionedEpochMigration.distribution must retain a prepared policy; got none"
        )
    if (
        states.mesh.cell_ids.shape != partitioned.states.mesh.cell_ids.shape
        or states.mesh.vertex_ids.shape != partitioned.states.mesh.vertex_ids.shape
    ):
        raise ValueError(
            "PartitionedEpochMigration.states must share prepared raw capacities; got "
            f"{states.mesh.cell_ids.shape}, {states.mesh.vertex_ids.shape}"
        )
    cells, vertices, status = _partitioned_epoch_payloads(
        partitioned, states, migration_state
    )
    bound = OwnerLocalMigrationState(
        _replace_work(
            migration_state.workset, status=migration_state.workset.status | status
        ),
        migration_state.cell_fields,
        migration_state.vertex_fields,
        migration_state.cell_epochs,
        migration_state.vertex_owners,
        migration_state.forest,
        migration_state.forest_cell_owners,
        migration_state.forest_vertex_owners,
        migration_state.cell_history + cells,
        migration_state.vertex_history + vertices,
    )
    migration = prepare_ownerlocal_migration(
        partitioned.parts,
        bound,
        requested_target_owners,
        halo_width=policy.halo_width,
        vertex_capacity=partitioned.layout.vertex_capacity,
    )
    if not bool(jax.device_get(jnp.all(migration.evidence.accepted))):
        raise ValueError(
            "PartitionedEpochMigration.transport must be collectively accepted; got rejection"
        )
    validate_simplex_predecessor_witness(
        partitioned.parts, states, partitioned.states, migration_state.workset
    )
    accepted = accept_ownerlocal_migration(migration)
    forest, parts = accepted.forest, migration.forest_parts
    if forest is None or parts is None:
        raise ValueError(
            "PartitionedEpochMigration.transport must retain physical families and their history; got none"
        )
    initial, exterior, origins, initial_solver, initial_status, initial_validity = (
        _reconstruct_partitioned_initial(partitioned, accepted, len(cells), len(vertices))
    )
    solver, solver_status = ownerlocal_solver_cell_owners(parts, accepted)
    solver_forest, solver_cells, solver_vertices, closure_status = (
        ownerlocal_solver_forest(parts, accepted)
    )
    solver_neighborhood, family_status = _complete_solver_family_neighborhood(
        parts, accepted.workset, solver_forest, partitioned.layout.vertex_capacity
    )
    solver_state = OwnerLocalMigrationState(
        accepted.workset,
        accepted.cell_fields,
        accepted.vertex_fields,
        accepted.cell_epochs,
        accepted.vertex_owners,
        solver_forest,
        accepted.forest_cell_owners,
        accepted.forest_vertex_owners,
        solver_cells,
        solver_vertices,
    )
    (
        solver_initial,
        solver_exterior,
        solver_origins,
        _,
        solver_initial_status,
        solver_initial_validity,
    ) = _reconstruct_partitioned_initial(
        partitioned,
        solver_state,
        len(cells),
        len(vertices),
        include_ghost_families=True,
    )
    status = (
        initial_status
        | solver_status
        | closure_status
        | solver_initial_status
        | family_status
    )
    if bool(jax.device_get(jnp.any(status != 0))):
        stage_values = jnp.stack(
            (
                initial_status,
                solver_status,
                closure_status,
                solver_initial_status,
                family_status,
            ),
            axis=-1,
        )
        # Refusal diagnostics are owner-local too: fetching the global arrays
        # here would mask the actual coverage refusal on a multihost runtime.
        stage_status, physical_validity, ghost_validity = (
            tuple(np.asarray(shard.data).tolist() for shard in value.addressable_shards)
            for value in (stage_values, initial_validity, solver_initial_validity)
        )
        raise ValueError(
            "PartitionedEpochMigration.records must preserve source and solver coverage; got "
            "owner-local [initial, solver, closure, solver_initial, neighborhood] status "
            f"{stage_status}; initial validity "
            "[cell_prefix, vertex_prefix, cell_ids, vertex_ids, conserved, "
            "cell_vertex_refs, cell_family_refs, vertex_parents, roots, descendants, midpoints, "
            "tuple_slots, tuple_vertices] "
            f"physical={physical_validity}, solver={ghost_validity}"
        )
    return PartitionedEpochMigration(
        partitioned,
        states,
        migration_state,
        requested_target_owners,
        forest,
        initial,
        exterior,
        origins,
        parts,
        solver,
        initial_solver,
        solver_neighborhood,
        solver_forest,
        solver_initial,
        solver_exterior,
        solver_origins,
        migration,
    )


@eqx.filter_jit
def _migration_array_pairs_equal(
    left: tuple[Array | np.ndarray, ...], right: tuple[Array | np.ndarray, ...]
) -> Array:
    """Replay every numerical equality in one program, without owner captures.

    ``array_equal`` deliberately retains its existing NaN and signed-zero
    semantics. Ordered structural/capacity/dtype refusals stay with the caller.
    """
    same = jnp.asarray(True)
    for before, after in zip(left, right, strict=True):
        same &= jnp.array_equal(before, after)
    return same


def _migration_placed_pairs_equal(
    left: tuple[Array | np.ndarray, ...],
    right: tuple[Array | np.ndarray, ...],
    /,
) -> bool:
    """Compare aligned replay leaves on their declared placement, then agree.

    Record IDs and slot references remain ordinary compared leaves. No row is
    reordered, no payload is gathered, and local source records never enter a
    global executable alongside the distributed forest.
    """
    groups: dict[
        tuple[bool, tuple[jax.Device, ...]],
        list[tuple[Array | np.ndarray, Array | np.ndarray]],
    ] = {}
    with agreed_local_step(stage="migration-record-comparison"):
        if len(left) != len(right):
            raise ValueError("migration comparison must retain every replay leaf")
        for before, after in zip(left, right, strict=True):
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError("migration comparison must preserve leaf shape and dtype")
            placed = tuple(value for value in (before, after) if isinstance(value, Array))
            devices = () if not placed else tuple(sorted(
                placed[0].devices(), key=lambda device: (device.platform, device.process_index, device.id)
            ))
            local = all(value.is_fully_addressable for value in placed)
            if any(value.devices() != set(devices) for value in placed) or (
                not local and len(placed) != 2
            ):
                raise ValueError("migration comparison leaves must share their declared owner placement")
            groups.setdefault((local, devices), []).append((before, after))
    global_keys = sorted(
        (key for key in groups if not key[0]),
        key=lambda key: tuple((device.platform, device.process_index, device.id) for device in key[1]),
    )
    declaration = canonical_fingerprint(tuple(
        (
            tuple((device.platform, device.process_index, device.id) for device in key[1]),
            tuple((before.shape, str(before.dtype)) for before, _ in groups[key]),
        )
        for key in global_keys
    ))
    declarations = np.asarray(multihost_utils.process_allgather(
        np.frombuffer(bytes.fromhex(declaration), dtype=np.uint8), tiled=False,
    )).reshape((-1, 32))
    if np.any(declarations != declarations[0]):
        raise ValueError("migration comparison must agree on the complete distributed leaf inventory")
    same = True
    with agreed_local_step(stage="migration-record-comparison"):
        for key, pairs in groups.items():
            if key[0]:
                result = _migration_array_pairs_equal(
                    tuple(before for before, _ in pairs), tuple(after for _, after in pairs)
                )
                same = bool(jax.device_get(result)) and same
    for key in global_keys:
        pairs = groups[key]
        result = _migration_array_pairs_equal(
            tuple(before for before, _ in pairs), tuple(after for _, after in pairs)
        )
        same = bool(jax.device_get(result.addressable_shards[0].data)) and same
    return bool(np.all(multihost_utils.process_allgather(
        np.asarray(same, dtype=np.bool_), tiled=False,
    )))


def validate_migrated_epoch(
    receipt: PartitionedEpochMigration,
    prepared: PreparedAdaptiveSimplex,
    current: AdaptiveSimplexState,
    initial: AdaptiveSimplexState,
    exterior: Array,
    origins: Array | np.ndarray,
    /,
) -> None:
    """Replay bounded transport and bind every published source/current record.

    The preparation owner validates the retained source epoch's scientific
    anchor. This transport record binds the allocation of all retained records,
    aligned slot references, constraints, and requested solver ownership.
    """
    if receipt.source_partitioned.prepared.prepared_id != prepared.prepared_id:
        raise ValueError(
            "PartitionedEpochMigration.prepared must match the retained source; got mismatch"
        )
    replayed = migrate_partitioned_epoch(
        receipt.source_partitioned,
        receipt.saved_states,
        receipt.saved_migration_state,
        receipt.requested_target_owners,
    )
    pairs = (
        (receipt.source_partitioned.prepared.state, prepared.state),
        (receipt.states, current),
        (receipt.initial_states, initial),
        (receipt.source_exterior, exterior),
        (receipt.slot_origins, origins),
        (replayed.states, receipt.states),
        (replayed.initial_states, receipt.initial_states),
        (replayed.source_exterior, receipt.source_exterior),
        (replayed.slot_origins, receipt.slot_origins),
        (replayed.solver_cell_owners, receipt.solver_cell_owners),
        (replayed.initial_solver_cell_owners, receipt.initial_solver_cell_owners),
        (replayed.solver_neighborhood, receipt.solver_neighborhood),
        (replayed.solver_forest, receipt.solver_forest),
        (replayed.solver_initial_states, receipt.solver_initial_states),
        (replayed.solver_source_exterior, receipt.solver_source_exterior),
        (replayed.solver_slot_origins, receipt.solver_slot_origins),
        (replayed.migration, receipt.migration),
    )
    expected_arrays = []
    actual_arrays = []
    with agreed_local_step(stage="migration-record-comparison"):
        for expected, actual in pairs:
            if jax.tree_util.tree_structure(expected) != jax.tree_util.tree_structure(actual):
                raise ValueError(
                    "PartitionedEpochMigration.records must match replayed metadata; got mismatch"
                )
            left = jax.tree_util.tree_leaves(expected)
            right = jax.tree_util.tree_leaves(actual)
            if len(left) != len(right):
                raise ValueError(
                    "PartitionedEpochMigration.records must match replayed leaves; got mismatch"
                )
            for before, after in zip(left, right, strict=True):
                if before.shape != after.shape or before.dtype != after.dtype:
                    raise ValueError(
                        "PartitionedEpochMigration.records must match replayed capacities/dtypes; got mismatch"
                    )
                expected_arrays.append(before)
                actual_arrays.append(after)
    same = _migration_placed_pairs_equal(tuple(expected_arrays), tuple(actual_arrays))
    if (
        receipt.parts.part_count != replayed.parts.part_count
        or receipt.parts.axis_name != replayed.parts.axis_name
        or receipt.parts.neighbor_pairs != replayed.parts.neighbor_pairs
        or not same
    ):
        raise ValueError(
            "PartitionedEpochMigration.records must equal accepted transport replay; got mismatch"
        )
