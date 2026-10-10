#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Topology-changing placement of saved, complete simplex forest records.

Only numerical stable-ID packets are repacked here. Scientific source recipes,
accepted certificates and predecessor proofs remain with the checkpoint/epoch
owners; a repack receipt is not an acceptance report.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    _vertex_half_facets,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    masked_simplex_facet_neighbors,
    MaskedSimplexMesh,
)
from ._distribution_forest import (
    _CELL_FIELDS,
    _ids,
    _members,
    _slots,
    _validated_forest_roots,
    _VERTEX_FIELDS,
)
from ._distribution_migration import (
    _forest_migration_routes,
    _prepare_migration_routes,
    _WORKSET_FIELDS,
    replay_graph_restart_proposal,
    SimplexGraphRestartProposal,
)
from ._result import CellMeshingResult, CollectiveMeshEvidence


_SENTINEL = jnp.iinfo(jnp.int64).max
_INVALID = int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
_CAPACITY = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)


@dataclass(frozen=True, slots=True)
class _SimplexRestartInventory:
    """Scientific old/new records, independent of where their archive is restored."""

    axis_name: str
    partition_count: int
    neighbor_pairs: tuple[tuple[int, int], ...]
    layout: AdaptiveSimplexLayout
    states: AdaptiveSimplexState
    cell_owners: Array
    vertex_owners: Array
    cell_history: tuple[Array, ...]
    vertex_history: tuple[Array, ...]
    status: Array
    cells_before: Array
    cells_after: Array
    vertices_before: Array
    vertices_after: Array
    saved_cursors: Array
    saved_clocks: Array
    saved_counters: Array
    saved_states: AdaptiveSimplexState
    saved_cell_owners: Array
    saved_vertex_owners: Array
    source_result: CellMeshingResult | None
    cell_saved_locations: Array
    vertex_saved_locations: Array
    requested_target_owners: Array
    saved_cell_history: tuple[Array, ...]
    saved_vertex_history: tuple[Array, ...]
    solver_cell_owners: Array
    saved_solver_cell_owners: Array | None
    cell_migration_counts: Array | None
    graph_proposal: SimplexGraphRestartProposal | None = None


@dataclass(frozen=True, slots=True)
class SimplexRestartRepack:
    """Concrete new-placement records and lossless original control inventory."""

    parts: AdaptiveSimplexParts
    layout: AdaptiveSimplexLayout
    states: AdaptiveSimplexState
    cell_owners: Array
    vertex_owners: Array
    cell_history: tuple[Array, ...]
    vertex_history: tuple[Array, ...]
    status: Array
    cells_before: Array
    cells_after: Array
    vertices_before: Array
    vertices_after: Array
    saved_cursors: Array
    saved_clocks: Array
    saved_counters: Array
    saved_states: AdaptiveSimplexState
    saved_cell_owners: Array
    saved_vertex_owners: Array
    source_result: CellMeshingResult | None
    cell_saved_locations: Array
    vertex_saved_locations: Array
    requested_target_owners: Array
    saved_cell_history: tuple[Array, ...]
    saved_vertex_history: tuple[Array, ...]
    solver_cell_owners: Array
    saved_solver_cell_owners: Array | None
    cell_migration_counts: Array | None
    graph_proposal: SimplexGraphRestartProposal | None = None


@final
class SimplexRestartRepackProof(StrictModule, NonTrainableState):
    """All-rank numerical proof consumed later without owner-local collectives."""

    arrays: tuple[tuple[str, Array], ...]
    source: CellMeshingResult
    layout: AdaptiveSimplexLayout
    content_id: str = eqx.field(static=True)
    source_result_id: str = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    neighbor_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(self, receipt: SimplexRestartRepack, /) -> None:
        if not isinstance(receipt, SimplexRestartRepack):
            raise TypeError("Restart proof requires the owning actual numerical receipt.")
        identity = simplex_restart_repack_content_id(receipt)
        source = receipt.source_result
        if source is None:
            raise ValueError("Validated restart proof lost its actual scientific source.")
        arrays = tuple(sorted(_restart_repack_arrays(receipt).items()))
        self.arrays = arrays
        self.source = source
        self.layout = receipt.layout
        self.content_id = identity
        self.source_result_id = source.result_id
        self.axis_name = receipt.parts.axis_name
        self.partition_count = receipt.parts.part_count
        self.neighbor_pairs = receipt.parts.neighbor_pairs

    def require_receipt(self, receipt: SimplexRestartRepack, /) -> None:
        """Require exact immutable array/source bindings using only local metadata."""
        if not isinstance(receipt, SimplexRestartRepack):
            raise TypeError(
                "Local restart construction requires its actual numerical receipt."
            )
        if (
            receipt.source_result is not self.source
            or receipt.layout is not self.layout
            or receipt.parts.axis_name != self.axis_name
            or receipt.parts.part_count != self.partition_count
            or receipt.parts.neighbor_pairs != self.neighbor_pairs
        ):
            raise ValueError(
                "Restart proof has a different scientific source or reconstruction placement."
            )
        placement = NamedSharding(receipt.parts.mesh, PartitionSpec(self.axis_name))
        if not receipt.states.mesh.cell_ids.sharding.is_equivalent_to(
            placement, receipt.states.mesh.cell_ids.ndim
        ):
            raise ValueError(
                "Restart arrays have a different actual device placement from the validated receipt."
            )
        actual = _restart_repack_arrays(receipt)
        if len(actual) != len(self.arrays) or any(
            name not in actual or actual[name] is not value for name, value in self.arrays
        ):
            raise ValueError(
                "Restart arrays differ from the validated immutable numerical bindings."
            )


@eqx.filter_jit
def _restart_solver_owners(active: Array, locations: Array, requested: Array) -> Array:
    parts = jnp.clip(locations[:, :, 0], 0, requested.shape[0] - 1)
    slots = jnp.clip(locations[:, :, 1], 0, requested.shape[1] - 1)
    found = jnp.all(locations >= 0, axis=-1)
    return jnp.where(active & found, requested[parts, slots], -1).astype(jnp.int32)


def _source_solver_owners(
    source: CellMeshingResult, saved: AdaptiveSimplexState
) -> Array:
    """Bind old solver senders to the actual accepted source entity bank."""
    evidence = source.collective_evidence
    storage = source.mesh.storage
    if evidence is None or storage is None:
        raise ValueError(
            "Accepted restart ownership requires the original logical mesh proof and storage."
        )
    count = saved.mesh.cell_ids.shape[0]
    if (
        evidence.partition_count != count
        or storage.partition_count != count
        or evidence.mesh_id != source.mesh.mesh_id
        or evidence.evidence_id != storage.evidence_id
    ):
        raise ValueError("The accepted source does not bind the saved logical placement.")
    identifiers, owners = evidence.entity_ids[-1], evidence.entity_owners[-1]
    total = evidence.global_entity_counts[-1]
    valid = jnp.arange(identifiers.shape[0]) < total
    order = jnp.argsort(jnp.where(valid, identifiers, _SENTINEL), stable=True)
    keys = jnp.where(valid[order], identifiers[order], _SENTINEL)
    position = jnp.minimum(jnp.searchsorted(keys, saved.mesh.cell_ids), keys.shape[0] - 1)
    matched = keys[position] == saved.mesh.cell_ids
    selected = owners[order[position]]
    invalid = jnp.any(
        saved.mesh.cell_active & (~matched | (selected < 0) | (selected >= count))
    )
    invalid |= jnp.any((keys[1:] == keys[:-1]) & (keys[1:] != _SENTINEL))
    invalid |= jnp.sum(saved.mesh.cell_active, dtype=jnp.int64) != total
    if bool(jax.device_get(invalid)):
        raise ValueError(
            "Saved active records do not cover the accepted scientific source ownership."
        )
    return jnp.where(saved.mesh.cell_active, selected, -1).astype(jnp.int32)


@eqx.filter_jit
def _restart_cell_traffic(
    saved: AdaptiveSimplexState,
    authority: Array,
    solver_owners: Array,
    requested: Array,
    count: int,
) -> Array:
    old_count = saved.mesh.cell_ids.shape[0]
    active = saved.mesh.cell_active & (
        authority == jnp.arange(old_count, dtype=jnp.int32)[:, None]
    )
    senders = jnp.where(active, solver_owners, old_count)
    receivers = jnp.where(active, requested, count)
    return (
        jnp.zeros((old_count, count), jnp.int64)
        .at[senders, receivers]
        .add(
            active.astype(jnp.int64),
            mode="drop",
        )
    )


def _numerical_digest(tree: object) -> str:
    """Use the canonical logical-byte owner for exact numerical leaf comparison."""
    arrays = {
        "value" + jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
        if isinstance(value, Array)
    }
    return logical_array_value_collection_digest(arrays)


@eqx.filter_jit
def _logical_restart_routes(vertex_ids: Array, /) -> Array:
    """Replay shared allocated-ID routes without assuming one device per saved row."""
    count = vertex_ids.shape[0]
    keys = jnp.sort(jnp.where(vertex_ids >= 0, vertex_ids, _SENTINEL), axis=1)

    def source_row(source: Array, mask: Array) -> Array:
        def target_row(target: Array, row_mask: Array) -> Array:
            positions = jnp.minimum(
                jnp.searchsorted(keys[target], keys[source]), keys.shape[1] - 1
            )
            shared = (source != target) & jnp.any(
                (keys[source] != _SENTINEL) & (keys[target, positions] == keys[source])
            )
            return row_mask.at[source, target].set(shared)

        return jax.lax.fori_loop(0, count, target_row, mask)

    return jax.lax.fori_loop(
        0, count, source_row, jnp.zeros((count, count), dtype=jnp.bool_)
    )


def validate_simplex_restart_repack(
    receipt: SimplexRestartRepack | _SimplexRestartInventory,
    requested_target_owners: Array,
    /,
) -> None:
    """Consume an actual accepted-source, complete-history rectangular receipt.

    Cached status, source presence and rank metadata are not proof. Recompute
    old-to-new scientific records and every supplied history leaf, rebind the
    saved source through its numerical archive theorem, and check exact controls,
    locations, solver senders/receivers and measured target execution routes.
    """
    from ._device_adaptation import restore_partitioned_mesh_state

    if not isinstance(receipt, (SimplexRestartRepack, _SimplexRestartInventory)):
        raise TypeError(
            "Restart validation requires the owning numerical repack receipt."
        )
    source = receipt.source_result
    if source is None:
        raise ValueError(
            "A raw staged repack is not an accepted scientific source receipt."
        )
    storage, evidence = source.mesh.storage, source.collective_evidence
    if storage is None or not isinstance(evidence, CollectiveMeshEvidence):
        raise ValueError(
            "Restart validation requires the source's actual retained-forest inventory and theorem."
        )
    saved_layout, _, _ = restore_partitioned_mesh_state(storage, evidence)
    # The storage epoch is the packed publication view; migration consumes the
    # independently retained raw compiled forest proved by that same theorem.
    accepted_saved = evidence.compiled_states
    if _numerical_digest(receipt.saved_states) != _numerical_digest(accepted_saved):
        raise ValueError(
            "The saved forest differs from the actual accepted source theorem."
        )
    if _numerical_digest(requested_target_owners) != _numerical_digest(
        receipt.requested_target_owners
    ):
        raise ValueError(
            "The rectangular placement proposal differs from its retained numerical request."
        )
    placement = receipt.parts if isinstance(receipt, SimplexRestartRepack) else receipt
    count = (
        placement.part_count
        if isinstance(placement, AdaptiveSimplexParts)
        else placement.partition_count
    )
    if (
        requested_target_owners.shape != receipt.saved_states.mesh.cell_ids.shape
        or requested_target_owners.dtype != jnp.int32
        or receipt.states.mesh.cell_ids.shape != (count, receipt.layout.cell_capacity)
        or receipt.states.mesh.vertex_ids.shape != (count, receipt.layout.vertex_capacity)
        or receipt.layout.dimension != saved_layout.dimension
        or receipt.layout.ambient_dimension != saved_layout.ambient_dimension
        or receipt.layout.coordinate_dtype != saved_layout.coordinate_dtype
    ):
        raise ValueError(
            "Restart receipt dimensions do not describe the actual old/new execution placements."
        )
    expected = _repack(
        receipt.layout,
        count,
        receipt.saved_states,
        receipt.saved_cell_owners,
        receipt.saved_vertex_owners,
        requested_target_owners,
        receipt.saved_cell_history,
        receipt.saved_vertex_history,
    )
    if expected[0].mesh.signature_id != receipt.states.mesh.signature_id:
        raise ValueError(
            "Restart raw layout differs from its actual coordinate/cell capacity identity."
        )
    actual = (
        receipt.states,
        receipt.cell_owners,
        receipt.vertex_owners,
        receipt.cell_history,
        receipt.vertex_history,
        receipt.status,
    )
    if _numerical_digest(expected) != _numerical_digest(actual):
        raise ValueError(
            "Restart records, typed histories or numerical verdict differ from exact repacking."
        )
    states, co, vo, _, _, status = expected
    if bool(jax.device_get(jnp.any(status != 0))):
        raise ValueError(
            "The complete-history placement failed its actual numerical packet proof."
        )
    cells = _saved_locations(
        receipt.saved_states.mesh.cell_ids,
        receipt.saved_cell_owners,
        states.mesh.cell_ids,
    )
    vertices = _saved_locations(
        receipt.saved_states.mesh.vertex_ids,
        receipt.saved_vertex_owners,
        states.mesh.vertex_ids,
    )
    before_ranks = jnp.arange(
        receipt.saved_states.mesh.cell_ids.shape[0], dtype=jnp.int32
    )[:, None]
    after_ranks = jnp.arange(count, dtype=jnp.int32)[:, None]
    controls = (
        receipt.saved_states.cursors,
        receipt.saved_states.clocks,
        receipt.saved_states.counters,
    )
    counts = (
        jnp.sum(
            (receipt.saved_states.mesh.cell_ids >= 0)
            & (receipt.saved_cell_owners == before_ranks),
            axis=1,
            dtype=jnp.int64,
        ),
        jnp.sum(states.mesh.cell_ids >= 0, axis=1, dtype=jnp.int64),
        jnp.sum(
            (receipt.saved_states.mesh.vertex_ids >= 0)
            & (receipt.saved_vertex_owners == before_ranks),
            axis=1,
            dtype=jnp.int64,
        ),
        jnp.sum(
            (states.mesh.vertex_ids >= 0) & (vo == after_ranks), axis=1, dtype=jnp.int64
        ),
    )
    if _numerical_digest((cells, vertices, controls, counts)) != _numerical_digest(
        (
            receipt.cell_saved_locations,
            receipt.vertex_saved_locations,
            (receipt.saved_cursors, receipt.saved_clocks, receipt.saved_counters),
            (
                receipt.cells_before,
                receipt.cells_after,
                receipt.vertices_before,
                receipt.vertices_after,
            ),
        )
    ):
        raise ValueError(
            "Restart locations, original controls or measured record counts were corrupted."
        )
    saved_solver = _source_solver_owners(source, receipt.saved_states)
    solver = _restart_solver_owners(
        states.mesh.cell_active, cells, requested_target_owners
    )
    traffic = _restart_cell_traffic(
        receipt.saved_states,
        receipt.saved_cell_owners,
        saved_solver,
        requested_target_owners,
        count,
    )
    if _numerical_digest((saved_solver, solver, traffic)) != _numerical_digest(
        (
            receipt.saved_solver_cell_owners,
            receipt.solver_cell_owners,
            receipt.cell_migration_counts,
        )
    ):
        raise ValueError(
            "Rectangular solver ownership/traffic differs from exact scientific cell correspondence."
        )
    if isinstance(receipt, SimplexRestartRepack):
        routes = _prepare_migration_routes(
            receipt.parts, _forest_migration_routes(receipt.parts, receipt.states)
        )
        neighbor_pairs = routes.neighbor_pairs
    else:
        mask = np.asarray(
            jax.device_get(_logical_restart_routes(receipt.states.mesh.vertex_ids)),
            dtype=np.bool_,
        )
        neighbor_pairs = tuple(
            (int(source), int(target)) for source, target in np.argwhere(mask)
        )
    if neighbor_pairs != placement.neighbor_pairs:
        raise ValueError(
            "Restart execution routes differ from actual shared allocated vertices."
        )
    graph = receipt.graph_proposal
    if graph is not None:
        if graph.parts.shape[0] != count or _numerical_digest(
            graph.requested_target_owners
        ) != _numerical_digest(requested_target_owners):
            raise ValueError(
                "A GRAPH restart must place exactly its executed partition's projected owners."
            )
        replay_graph_restart_proposal(
            graph, receipt.saved_states, axis_name=placement.axis_name
        )


def _restart_repack_arrays(
    receipt: SimplexRestartRepack | _SimplexRestartInventory,
) -> dict[str, Array]:
    """Explicit complete numerical inventory shared by identity and local binding."""
    old_solver = receipt.saved_solver_cell_owners
    traffic = receipt.cell_migration_counts
    if receipt.source_result is None or old_solver is None or traffic is None:
        raise ValueError(
            "Validated restart content requires its actual scientific source and solver transport."
        )
    arrays = {
        "proposal/target_owners": receipt.requested_target_owners,
        "source/raw_cell_owners": receipt.saved_cell_owners,
        "source/raw_vertex_owners": receipt.saved_vertex_owners,
        "source/solver_cell_owners": old_solver,
        "source/cursors": receipt.saved_cursors,
        "source/clocks": receipt.saved_clocks,
        "source/counters": receipt.saved_counters,
        "target/raw_cell_owners": receipt.cell_owners,
        "target/raw_vertex_owners": receipt.vertex_owners,
        "target/solver_cell_owners": receipt.solver_cell_owners,
        "target/cell_saved_locations": receipt.cell_saved_locations,
        "target/vertex_saved_locations": receipt.vertex_saved_locations,
        "target/cell_migration_counts": traffic,
        "target/status": receipt.status,
        "counts/cells_before": receipt.cells_before,
        "counts/cells_after": receipt.cells_after,
        "counts/vertices_before": receipt.vertices_before,
        "counts/vertices_after": receipt.vertices_after,
    }
    mesh_names = (
        "coordinates",
        "vertex_ids",
        "vertex_active",
        "cells",
        "cell_ids",
        "cell_active",
        "facet_neighbors",
    )
    state_names = (
        "vertex_half_facets",
        "tuples",
        "tags",
        "blocks",
        "generations",
        "parents",
        "children",
        "bisection_vertices",
        "retired",
        "cell_classes",
        "facet_classes",
        "vertex_parents",
        "vertex_levels",
        "vertex_removal",
        "vertex_protected",
        "protected_codes",
        "refine_rejected",
        "coarsen_marked",
        "cursors",
        "clocks",
        "counters",
    )
    for prefix, state in (("source", receipt.saved_states), ("target", receipt.states)):
        arrays.update(
            {f"{prefix}/mesh/{name}": getattr(state.mesh, name) for name in mesh_names}
        )
        arrays.update(
            {f"{prefix}/state/{name}": getattr(state, name) for name in state_names}
        )
    for prefix, cells, vertices in (
        ("source", receipt.saved_cell_history, receipt.saved_vertex_history),
        ("target", receipt.cell_history, receipt.vertex_history),
    ):
        arrays.update(
            {f"{prefix}/cell_history/{index}": value for index, value in enumerate(cells)}
        )
        arrays.update(
            {
                f"{prefix}/vertex_history/{index}": value
                for index, value in enumerate(vertices)
            }
        )
    graph = receipt.graph_proposal
    if graph is not None:
        arrays.update(
            {
                "graph/parts": graph.parts,
                "graph/vertex_weights": graph.vertex_weights,
                "graph/edge_weights": graph.edge_weights,
                **{
                    f"graph/workset/{name}": getattr(graph.workset, name)
                    for name in _WORKSET_FIELDS
                },
            }
        )
    return arrays


def simplex_restart_repack_content_id(
    receipt: SimplexRestartRepack | _SimplexRestartInventory, /
) -> str:
    """Validate once and identify complete scientific placement without live caches."""
    validate_simplex_restart_repack(receipt, receipt.requested_target_owners)
    source = receipt.source_result
    if source is None:
        raise ValueError("Validated restart content lost its actual accepted source.")
    arrays = _restart_repack_arrays(receipt)
    placement = receipt.parts if isinstance(receipt, SimplexRestartRepack) else receipt
    return canonical_fingerprint(
        {
            "kind": "complete-simplex-restart-repack",
            "source_result": source.result_id,
            "source_mesh": source.mesh.mesh_id,
            "source_coordinate_contract": source.coordinate_contract.spatial_id,
            "source_layout": receipt.saved_states.mesh.signature_id,
            "target_layout": receipt.layout.signature_id,
            "old_partition_count": receipt.saved_states.mesh.cell_ids.shape[0],
            "new_partition_count": placement.part_count
            if isinstance(placement, AdaptiveSimplexParts)
            else placement.partition_count,
            "axis_name": placement.axis_name,
            "neighbor_pairs": placement.neighbor_pairs,
            "proposal_route": None
            if receipt.graph_proposal is None
            else {
                "kind": "native-graph",
                "plan": receipt.graph_proposal.plan.plan_id,
            },
            "numerical_content": logical_array_value_collection_digest(arrays),
        }
    )


@eqx.filter_jit
def restart_forest_target_owners(
    ancestor_cell_ids: Array,
    repacked_cell_ids: Array,
    repacked_cell_owners: Array,
    /,
) -> tuple[Array, Array]:
    """Route a saved theorem's initial forest by retained scientific cell IDs.

    The ancestor forest can have different capacities and old part count.
    Every allocated ancestor record must have exactly one authoritative record
    in the repacked forest; unknown/duplicated records refuse the whole mapping.
    These owners permit an actual second repack of the old initial states and
    their geometry/exterior payloads, not a relabeling of the predecessor proof.
    """
    if (
        ancestor_cell_ids.ndim != 2
        or repacked_cell_ids.ndim != 2
        or repacked_cell_owners.shape != repacked_cell_ids.shape
    ):
        raise ValueError(
            "Ancestor and repacked forest IDs require their actual part/capacity axes."
        )
    if (
        ancestor_cell_ids.dtype != jnp.int64
        or repacked_cell_ids.dtype != jnp.int64
        or repacked_cell_owners.dtype != jnp.int32
    ):
        raise TypeError("Forest correspondence requires int64 IDs and int32 owners.")
    count = repacked_cell_ids.shape[0]
    result = jnp.full(ancestor_cell_ids.shape, count, jnp.int32)
    occurrences = jnp.zeros(ancestor_cell_ids.shape, jnp.int32)

    def visit(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        result, occurrences = carry
        keys = jnp.where(
            repacked_cell_ids[index] >= 0, repacked_cell_ids[index], _SENTINEL
        )
        left = jnp.searchsorted(keys, ancestor_cell_ids, side="left")
        right = jnp.searchsorted(keys, ancestor_cell_ids, side="right")
        matches = jnp.where(ancestor_cell_ids >= 0, right - left, 0).astype(jnp.int32)
        return jnp.minimum(
            result, jnp.where(matches > 0, index, count)
        ), occurrences + matches

    result, occurrences = jax.lax.fori_loop(0, count, visit, (result, occurrences))
    invalid = jnp.any((ancestor_cell_ids >= 0) & (occurrences != 1))
    invalid |= jnp.any(
        (repacked_cell_ids >= 0)
        & (repacked_cell_owners != jnp.arange(count, dtype=jnp.int32)[:, None])
    )
    return jnp.where(ancestor_cell_ids >= 0, result, -1).astype(jnp.int32), jnp.where(
        invalid, _INVALID, 0
    ).astype(jnp.int32)


@eqx.filter_jit
def _saved_locations(saved_ids: Array, owners: Array, target_ids: Array) -> Array:
    """Exact new-slot to authoritative old-part/slot correspondence."""
    result = jnp.full((*target_ids.shape, 2), -1, jnp.int32)

    def visit(index: int, result: Array) -> Array:
        keys = jnp.where(saved_ids[index] >= 0, saved_ids[index], _SENTINEL)
        slots = jnp.minimum(jnp.searchsorted(keys, target_ids), keys.shape[0] - 1)
        found = (
            (target_ids >= 0)
            & (keys[slots] == target_ids)
            & (owners[index, slots] == index)
        )
        return jnp.where(
            found[:, :, None],
            jnp.stack((jnp.full_like(slots, index), slots), axis=-1).astype(jnp.int32),
            result,
        )

    return jax.lax.fori_loop(0, saved_ids.shape[0], visit, result)


def _merge_packet(
    current: dict[str, Array],
    packet: dict[str, Array],
    selected: Array,
) -> tuple[dict[str, Array], Array]:
    capacity = current["ids"].shape[0]
    keys = jnp.concatenate(
        (current["ids"], jnp.where(selected, packet["ids"], _SENTINEL))
    )
    order = jnp.argsort(keys, stable=True)
    ordered = keys[order]
    unique = (ordered != _SENTINEL) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), ordered[1:] != ordered[:-1])
    )
    count = jnp.sum(unique, dtype=jnp.int32)
    pick = jnp.nonzero(unique, size=capacity, fill_value=keys.shape[0] - 1)[0]
    result = {
        name: jnp.concatenate((current[name], packet[name]))[order[pick]]
        for name in current
    }
    result["ids"] = jnp.where(jnp.arange(capacity) < count, ordered[pick], _SENTINEL)
    duplicate = (ordered[1:] == ordered[:-1]) & (ordered[1:] != _SENTINEL)
    conflict = jnp.asarray(False)
    for name in current:
        values = jnp.concatenate((current[name], packet[name]))[order]
        different = values[1:] != values[:-1]
        if different.ndim > 1:
            different = jnp.any(different, axis=tuple(range(1, different.ndim)))
        conflict |= jnp.any(duplicate & different)
    return result, jnp.where(count > capacity, _CAPACITY, 0) | jnp.where(
        conflict, _INVALID, 0
    )


def _empty(packet: dict[str, Array], capacity: int) -> dict[str, Array]:
    return {
        name: jnp.full((capacity, *value.shape[1:]), _SENTINEL, jnp.int64)
        if name == "ids"
        else jnp.zeros((capacity, *value.shape[1:]), value.dtype)
        for name, value in packet.items()
    }


def _packet(
    raw: AdaptiveSimplexState,
    cells: tuple[Array, ...],
    vertices: tuple[Array, ...],
) -> tuple[dict[str, Array], dict[str, Array], Array]:
    mesh = raw.mesh
    c, v = mesh.cell_ids.shape[0], mesh.vertex_ids.shape[0]
    cell = {
        "ids": mesh.cell_ids,
        "cells": _ids(mesh.vertex_ids, mesh.cells),
        "tuples": _ids(mesh.vertex_ids, raw.tuples),
        "active": mesh.cell_active,
        "parents": _ids(mesh.cell_ids, raw.parents // 2),
        "ordinal": raw.parents % 2,
        "children": _ids(mesh.cell_ids, raw.children),
        "bisection": _ids(mesh.vertex_ids, raw.bisection_vertices),
        **{name: getattr(raw, name) for name in _CELL_FIELDS},
        **{f"payload_{i}": value for i, value in enumerate(cells)},
    }
    vertex = {
        "ids": mesh.vertex_ids,
        "coordinates": mesh.coordinates,
        "active": mesh.vertex_active,
        "parents": _ids(mesh.vertex_ids, raw.vertex_parents),
        **{name: getattr(raw, name) for name in _VERTEX_FIELDS},
        **{f"payload_{i}": value for i, value in enumerate(vertices)},
    }
    invalid = jnp.any(
        (mesh.cell_ids >= 0) & ((raw.parents < -1) | (raw.parents >= 2 * c))
    )
    for references in (mesh.cells, raw.tuples, raw.bisection_vertices):
        mask = (mesh.cell_ids >= 0).reshape((c,) + (1,) * (references.ndim - 1))
        invalid |= jnp.any(mask & ((references < -1) | (references >= v)))
    invalid |= jnp.any(
        (mesh.cell_ids >= 0)[:, None] & ((raw.children < -1) | (raw.children >= c))
    )
    invalid |= jnp.any(
        (mesh.vertex_ids >= 0)[:, None]
        & ((raw.vertex_parents < -1) | (raw.vertex_parents >= v))
    )
    allocated = mesh.cell_ids >= 0
    invalid |= jnp.any(allocated[:, None] & ((mesh.cells < 0) | (raw.tuples < 0)))
    invalid |= jnp.any(allocated[:, None] & ((cell["cells"] < 0) | (cell["tuples"] < 0)))
    for slots, references in (
        (raw.parents // 2, cell["parents"]),
        (raw.children, cell["children"]),
        (raw.bisection_vertices, cell["bisection"]),
    ):
        mask = allocated.reshape((c,) + (1,) * (slots.ndim - 1))
        invalid |= jnp.any(mask & (slots >= 0) & (references < 0))
    invalid |= jnp.any(
        (mesh.vertex_ids >= 0)[:, None]
        & (raw.vertex_parents >= 0)
        & (vertex["parents"] < 0)
    )
    protected = raw.protected_codes != _SENTINEL
    endpoints = jnp.stack((raw.protected_codes // v, raw.protected_codes % v), axis=1)
    invalid |= jnp.any(
        protected[:, None] & (_ids(mesh.vertex_ids, jnp.clip(endpoints, 0, v - 1)) < 0)
    )
    invalid |= jnp.any(
        (mesh.vertex_ids >= 0) & ~jnp.all(jnp.isfinite(mesh.coordinates), axis=1)
    )
    invalid |= jnp.any(
        (raw.protected_codes != _SENTINEL)
        & ((raw.protected_codes < 0) | (raw.protected_codes >= v * v))
    )
    invalid |= (raw.cursors[2] <= jnp.max(mesh.vertex_ids)) | (
        raw.cursors[3] <= jnp.max(mesh.cell_ids)
    )
    return cell, vertex, jnp.where(invalid, _INVALID, 0)


def _missing(references: Array, ids: Array, valid: Array) -> Array:
    keys = jnp.where(ids >= 0, ids, _SENTINEL)
    pos = jnp.minimum(jnp.searchsorted(keys, references), ids.shape[0] - 1)
    mask = valid.reshape((valid.shape[0],) + (1,) * (references.ndim - 1))
    return jnp.any(mask & (references >= 0) & (ids[pos] != references))


@eqx.filter_jit
def _repack(
    layout: AdaptiveSimplexLayout,
    count: int,
    old: AdaptiveSimplexState,
    cell_owners: Array,
    vertex_owners: Array,
    requested: Array,
    ch: tuple[Array, ...],
    vh: tuple[Array, ...],
) -> tuple[
    AdaptiveSimplexState, Array, Array, tuple[Array, ...], tuple[Array, ...], Array
]:
    old_count = old.mesh.cell_ids.shape[0]
    ranks = jnp.arange(old_count, dtype=jnp.int32)[:, None]
    authoritative = (old.mesh.cell_ids >= 0) & (cell_owners == ranks)
    vertex_authoritative = (old.mesh.vertex_ids >= 0) & (vertex_owners == ranks)

    def complete_authority(ids: Array, owners: Array, mask: Array) -> Array:
        order = jnp.argsort(jnp.where(mask, ids, _SENTINEL).reshape(-1), stable=True)
        keys = jnp.where(mask, ids, _SENTINEL).reshape(-1)[order]
        source_rank = jnp.broadcast_to(ranks, ids.shape).reshape(-1)[order]
        position = jnp.minimum(jnp.searchsorted(keys, ids), keys.shape[0] - 1)
        unique = jnp.all((keys[1:] != keys[:-1]) | (keys[1:] == _SENTINEL))
        covered = jnp.all(
            (ids < 0) | ((keys[position] == ids) & (source_rank[position] == owners))
        )
        return unique & covered

    invalid = ~complete_authority(old.mesh.cell_ids, cell_owners, authoritative)
    invalid |= ~complete_authority(
        old.mesh.vertex_ids, vertex_owners, vertex_authoritative
    )
    invalid |= jnp.any(
        (old.mesh.cell_ids >= 0) & ((cell_owners < 0) | (cell_owners >= old_count))
    )
    invalid |= jnp.any(
        (old.mesh.vertex_ids >= 0) & ((vertex_owners < 0) | (vertex_owners >= old_count))
    )
    invalid |= jnp.any(old.clocks[:, 2] != 0)

    def local(
        rank: Array,
    ) -> tuple[
        AdaptiveSimplexState, Array, Array, tuple[Array, ...], tuple[Array, ...], Array
    ]:
        first = jax.tree_util.tree_map(lambda value: value[0], old)
        cell, vertex, _ = _packet(
            first, tuple(value[0] for value in ch), tuple(value[0] for value in vh)
        )
        resident = _empty(cell, layout.cell_capacity)
        vertices = _empty(vertex, layout.vertex_capacity)
        status = jnp.where(invalid, _INVALID, 0)

        def cell_visit(
            index: int, carry: tuple[dict[str, Array], Array]
        ) -> tuple[dict[str, Array], Array]:
            resident, status = carry
            raw = jax.tree_util.tree_map(lambda value: value[index], old)
            packet, _, flags = _packet(
                raw,
                tuple(value[index] for value in ch),
                tuple(value[index] for value in vh),
            )
            roots, valid = _validated_forest_roots(raw)
            target = jnp.full(raw.mesh.cell_ids.shape, count, jnp.int32)
            proposal = requested[index]
            active = raw.mesh.cell_active & (cell_owners[index] == index)
            status |= flags | jnp.where(
                valid & ~jnp.any(active & ((proposal < 0) | (proposal >= count))),
                0,
                _INVALID,
            )
            target = target.at[roots].min(jnp.where(active, proposal, count))[roots]
            target = jnp.where(target < count, target, raw.mesh.cell_ids[roots] % count)
            selected = (
                (raw.mesh.cell_ids >= 0)
                & (cell_owners[index] == index)
                & (target == rank)
            )
            resident, flags = _merge_packet(resident, packet, selected)
            return resident, status | flags

        resident, status = jax.lax.fori_loop(0, old_count, cell_visit, (resident, status))
        used = resident["ids"] != _SENTINEL
        required = jnp.concatenate(
            (
                jnp.where(used[:, None], resident["cells"], -1).reshape(-1),
                jnp.where(used[:, None], resident["tuples"], -1).reshape(-1),
                jnp.where(used, resident["bisection"], -1),
            )
        )

        def vertex_round(
            carry: tuple[dict[str, Array], Array, Array, Array],
        ) -> tuple[dict[str, Array], Array, Array, Array]:
            vertices, status, iteration, _ = carry
            before = vertices["ids"]
            refs = jnp.sort(
                jnp.concatenate(
                    (
                        required,
                        jnp.where(
                            (before != _SENTINEL)[:, None], vertices["parents"], -1
                        ).reshape(-1),
                        jnp.where(before != _SENTINEL, before, -1),
                    )
                )
            )

            def visit(
                index: int, carry: tuple[dict[str, Array], Array]
            ) -> tuple[dict[str, Array], Array]:
                result, status = carry
                raw = jax.tree_util.tree_map(lambda value: value[index], old)
                _, packet, flags = _packet(
                    raw,
                    tuple(value[index] for value in ch),
                    tuple(value[index] for value in vh),
                )
                v = raw.mesh.vertex_ids.shape[0]
                codes = raw.protected_codes
                edges = jnp.stack((codes // v, codes % v), axis=1)
                edges = jnp.where(
                    (codes != _SENTINEL)[:, None],
                    _ids(raw.mesh.vertex_ids, jnp.clip(edges, 0, v - 1)),
                    -1,
                )
                touching = jnp.any(_members(refs, edges), axis=1)
                edge_refs = jnp.sort(jnp.where(touching[:, None], edges, -1).reshape(-1))
                selected = (packet["ids"] >= 0) & (
                    _members(refs, packet["ids"])
                    | _members(edge_refs, packet["ids"])
                    | (packet["ids"] % count == rank)
                )
                result, merged_status = _merge_packet(result, packet, selected)
                return result, status | flags | merged_status

            vertices, status = jax.lax.fori_loop(0, old_count, visit, (vertices, status))
            return vertices, status, iteration + 1, jnp.any(vertices["ids"] != before)

        vertices, status, _, changed = jax.lax.while_loop(
            lambda carry: carry[3] & (carry[2] <= old.mesh.vertex_ids.size),
            vertex_round,
            (vertices, status, jnp.asarray(0, jnp.int32), jnp.asarray(True)),
        )
        status |= jnp.where(changed, _INVALID, 0)
        cids = jnp.where(used, resident["ids"], -1)
        vids = jnp.where(vertices["ids"] != _SENTINEL, vertices["ids"], -1)
        for name in ("cells", "tuples", "bisection"):
            status |= jnp.where(_missing(resident[name], vids, used), _INVALID, 0)
        for name in ("parents", "children"):
            status |= jnp.where(_missing(resident[name], cids, used), _INVALID, 0)
        status |= jnp.where(_missing(vertices["parents"], vids, vids >= 0), _INVALID, 0)
        rows = jnp.maximum(_slots(vids, resident["cells"]), 0)
        active = used & resident["active"]
        neighbors = masked_simplex_facet_neighbors(rows, active)
        mesh = MaskedSimplexMesh(
            vertices["coordinates"],
            vids,
            (vids >= 0) & vertices["active"],
            rows,
            cids,
            active,
            neighbors,
        )
        codes = jnp.full((layout.protected_edge_capacity,), _SENTINEL, jnp.int64)

        def constraints(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            codes, status = carry
            raw = jax.tree_util.tree_map(lambda value: value[index], old)
            v = raw.mesh.vertex_ids.shape[0]
            edges = jnp.stack((raw.protected_codes // v, raw.protected_codes % v), axis=1)
            edges = jnp.where(
                (raw.protected_codes != _SENTINEL)[:, None],
                _ids(raw.mesh.vertex_ids, jnp.clip(edges, 0, v - 1)),
                -1,
            )
            present = jnp.all(
                _members(jnp.where(vids >= 0, vids, _SENTINEL), edges), axis=1
            )
            slots = _slots(vids, edges).astype(jnp.int64)
            incoming = jnp.where(
                present,
                jnp.min(slots, axis=1) * layout.vertex_capacity + jnp.max(slots, axis=1),
                _SENTINEL,
            )
            ordered = jnp.sort(jnp.concatenate((codes, incoming)))
            fresh = (ordered != _SENTINEL) & jnp.concatenate(
                (jnp.ones(1, jnp.bool_), ordered[1:] != ordered[:-1])
            )
            pick = jnp.nonzero(
                fresh, size=codes.shape[0], fill_value=ordered.shape[0] - 1
            )[0]
            n = jnp.sum(fresh, dtype=jnp.int32)
            return jnp.where(
                jnp.arange(codes.shape[0]) < n, ordered[pick], _SENTINEL
            ), status | jnp.where(n > codes.shape[0], _CAPACITY, 0)

        codes, status = jax.lax.fori_loop(0, old_count, constraints, (codes, status))
        parent = _slots(cids, resident["parents"])

        def accumulate(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            counters, status = carry
            incoming = jnp.where(index % count == rank, old.counters[index], 0)
            overflow = jnp.any((incoming < 0) | (counters > _SENTINEL - incoming))
            return counters + incoming, status | jnp.where(overflow, _CAPACITY, 0)

        counters, status = jax.lax.fori_loop(
            0,
            old_count,
            accumulate,
            (jnp.zeros(old.counters.shape[1:], jnp.int64), status),
        )
        clocks = jnp.max(old.clocks, axis=0).at[2].set(status)
        state = AdaptiveSimplexState(
            mesh,
            vertex_half_facets=_vertex_half_facets(
                rows, active, neighbors, layout.vertex_capacity
            ),
            tuples=jnp.maximum(_slots(vids, resident["tuples"]), 0),
            parents=jnp.where(parent >= 0, parent * 2 + resident["ordinal"], -1),
            children=_slots(cids, resident["children"]),
            bisection_vertices=_slots(vids, resident["bisection"]),
            vertex_parents=_slots(vids, vertices["parents"]),
            protected_codes=codes,
            cursors=jnp.stack(
                (
                    jnp.sum(vids >= 0, dtype=jnp.int64),
                    jnp.sum(used, dtype=jnp.int64),
                    jnp.max(old.cursors[:, 2]),
                    jnp.max(old.cursors[:, 3]),
                )
            ),
            clocks=clocks,
            counters=counters,
            tags=resident["tags"],
            blocks=resident["blocks"],
            generations=resident["generations"],
            retired=resident["retired"],
            cell_classes=resident["cell_classes"],
            facet_classes=resident["facet_classes"],
            refine_rejected=resident["refine_rejected"],
            coarsen_marked=resident["coarsen_marked"],
            vertex_levels=vertices["vertex_levels"],
            vertex_removal=vertices["vertex_removal"],
            vertex_protected=vertices["vertex_protected"],
        )
        return (
            state,
            jnp.where(used, rank, -1).astype(jnp.int32),
            jnp.where(vids >= 0, vids % count, -1).astype(jnp.int32),
            tuple(resident[f"payload_{i}"] for i in range(len(ch))),
            tuple(vertices[f"payload_{i}"] for i in range(len(vh))),
            status,
        )

    states, co, vo, ch, vh, status = jax.vmap(local)(jnp.arange(count, dtype=jnp.int32))
    before = jnp.sum(authoritative, dtype=jnp.int64)
    after = jnp.sum(states.mesh.cell_ids >= 0, dtype=jnp.int64)
    vb = jnp.sum(vertex_authoritative, dtype=jnp.int64)
    va = jnp.sum(
        (states.mesh.vertex_ids >= 0)
        & (vo == jnp.arange(count, dtype=jnp.int32)[:, None]),
        dtype=jnp.int64,
    )
    verdict = jnp.bitwise_or.reduce(status) | jnp.where(
        (before != after) | (vb != va), _INVALID, 0
    )
    status = jnp.full((count,), verdict, jnp.int32)
    states = eqx.tree_at(
        lambda value: value.clocks, states, states.clocks.at[:, 2].set(verdict)
    )
    return states, co, vo, ch, vh, status


def repack_simplex_restart(
    target_group: ExecutionGroup,
    layout: AdaptiveSimplexLayout,
    saved: AdaptiveSimplexState,
    cell_owners: Array,
    vertex_owners: Array,
    requested_target_owners: Array,
    /,
    *,
    cell_history: tuple[Array, ...] = (),
    vertex_history: tuple[Array, ...] = (),
    source_result: CellMeshingResult | None = None,
    graph_proposal: SimplexGraphRestartProposal | None = None,
    route_metadata_capacity_bytes: int = 1 << 20,
) -> SimplexRestartRepack:
    """Reassemble complete old logical records on actual new execution devices.

    ``requested_target_owners`` is the placement proposal in saved raw-cell ID
    order. When it comes from the native GRAPH route, ``graph_proposal`` binds
    the executed partition and must hold this exact proposal; restart
    validation replays it. A family stays together; its smallest active proposed
    owner wins.
    Entirely inactive families use their root scientific ID. Cumulative counters
    retain their total, clocks use their global maximum and allocator ID cursors
    retain their maximum. Exact former rank controls are also returned unchanged.
    Any numerical refusal applies collectively; no result acceptance is implied.
    """
    if (
        not isinstance(target_group, ExecutionGroup)
        or not isinstance(layout, AdaptiveSimplexLayout)
        or not isinstance(saved, AdaptiveSimplexState)
    ):
        raise TypeError(
            "Restart repack requires an explicit execution group, layout and saved raw state."
        )
    if len(target_group.mesh.axis_names) != 1 or not target_group.is_member:
        raise ValueError(
            "Restart repack requires membership in a one-axis target execution group."
        )
    old_shape = saved.mesh.cell_ids.shape
    if (
        len(old_shape) != 2
        or cell_owners.shape != old_shape
        or requested_target_owners.shape != old_shape
        or vertex_owners.shape != saved.mesh.vertex_ids.shape
    ):
        raise ValueError(
            "Saved owner/proposal arrays must align with the old logical raw capacities."
        )
    if (
        cell_owners.dtype != jnp.int32
        or vertex_owners.dtype != jnp.int32
        or requested_target_owners.dtype != jnp.int32
    ):
        raise TypeError("Raw owners and graph proposals must be int32.")
    if (
        saved.mesh.cells.shape[-1] - 1 != layout.dimension
        or saved.mesh.ambient_dimension != layout.ambient_dimension
        or saved.mesh.coordinates.dtype.name != layout.coordinate_dtype
    ):
        raise ValueError(
            "Restart placement cannot change the saved geometry or coordinate type."
        )
    if any(value.shape[:2] != old_shape for value in cell_history) or any(
        value.shape[:2] != saved.mesh.vertex_ids.shape for value in vertex_history
    ):
        raise ValueError("Typed histories must bind every allocated old raw record.")
    if source_result is not None and not isinstance(source_result, CellMeshingResult):
        raise TypeError("source_result must be the actual accepted predecessor result.")
    if (
        graph_proposal is not None
        and graph_proposal.requested_target_owners is not requested_target_owners
    ):
        raise ValueError(
            "A GRAPH restart must place exactly its executed partition's projected owners."
        )
    axis = target_group.mesh.axis_names[0]
    count = len(target_group.devices)
    route_budget = operator.index(route_metadata_capacity_bytes)
    if isinstance(route_metadata_capacity_bytes, bool) or route_budget < count * count:
        raise ValueError(
            "Restart shared-vertex routing exceeds its explicit prepare-barrier byte budget."
        )
    parts = AdaptiveSimplexParts(target_group.devices, axis_name=axis, neighbor_pairs=())
    output = _repack(
        layout,
        count,
        saved,
        cell_owners,
        vertex_owners,
        requested_target_owners,
        cell_history,
        vertex_history,
    )
    logical_states = output[0]
    cell_locations = _saved_locations(
        saved.mesh.cell_ids, cell_owners, logical_states.mesh.cell_ids
    )
    vertex_locations = _saved_locations(
        saved.mesh.vertex_ids, vertex_owners, logical_states.mesh.vertex_ids
    )
    solver_owners = _restart_solver_owners(
        logical_states.mesh.cell_active, cell_locations, requested_target_owners
    )
    # Complete correspondence math runs in the old logical array placement;
    # the single explicit lowering then moves all new-slot records together.
    output, cell_locations, vertex_locations, solver_owners = jax.tree_util.tree_map(
        lambda value: jax.device_put(
            value, NamedSharding(parts.mesh, PartitionSpec(axis))
        ),
        (output, cell_locations, vertex_locations, solver_owners),
    )
    states, co, vo, ch, vh, status = output
    parts = _prepare_migration_routes(parts, _forest_migration_routes(parts, states))
    ranks = jnp.arange(old_shape[0], dtype=jnp.int32)[:, None]
    new_ranks = jnp.arange(count, dtype=jnp.int32)[:, None]
    saved_solver_owners = (
        None if source_result is None else _source_solver_owners(source_result, saved)
    )
    traffic = (
        None
        if saved_solver_owners is None
        else _restart_cell_traffic(
            saved,
            cell_owners,
            saved_solver_owners,
            requested_target_owners,
            count,
        )
    )
    return SimplexRestartRepack(
        parts=parts,
        layout=layout,
        states=states,
        cell_owners=co,
        vertex_owners=vo,
        cell_history=ch,
        vertex_history=vh,
        status=status,
        cells_before=jnp.sum(
            (saved.mesh.cell_ids >= 0) & (cell_owners == ranks), axis=1, dtype=jnp.int64
        ),
        cells_after=jnp.sum(states.mesh.cell_ids >= 0, axis=1, dtype=jnp.int64),
        vertices_before=jnp.sum(
            (saved.mesh.vertex_ids >= 0) & (vertex_owners == ranks),
            axis=1,
            dtype=jnp.int64,
        ),
        vertices_after=jnp.sum(
            (states.mesh.vertex_ids >= 0) & (vo == new_ranks), axis=1, dtype=jnp.int64
        ),
        saved_cursors=saved.cursors,
        saved_clocks=saved.clocks,
        saved_counters=saved.counters,
        saved_states=saved,
        saved_cell_owners=cell_owners,
        saved_vertex_owners=vertex_owners,
        source_result=source_result,
        cell_saved_locations=cell_locations,
        vertex_saved_locations=vertex_locations,
        requested_target_owners=requested_target_owners,
        saved_cell_history=cell_history,
        saved_vertex_history=vertex_history,
        solver_cell_owners=solver_owners,
        saved_solver_cell_owners=saved_solver_owners,
        cell_migration_counts=traffic,
        graph_proposal=graph_proposal,
    )
