#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded stable-ID transport of the canonical raw simplex forest.

Slot references never cross a rank boundary. Families are assigned an execution
owner independently of solver-cell ownership, because raw bisection/coarsening
operates on complete families. The solver workset retains graph ownership.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec

from ..discretization._adaptive_simplex import (
    _global_flags,
    _vertex_half_facets,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    masked_simplex_facet_neighbors,
)


if TYPE_CHECKING:
    from ._distribution import SimplexNeighborhoodWorkset


_SENTINEL = jnp.iinfo(jnp.int64).max
_INVALID = int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
_CAPACITY = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
_CELL_FIELDS = (
    "tags",
    "blocks",
    "generations",
    "retired",
    "cell_classes",
    "facet_classes",
    "refine_rejected",
    "coarsen_marked",
)
_VERTEX_FIELDS = ("vertex_levels", "vertex_removal", "vertex_protected")


def _ids(ids: Array, slots: Array) -> Array:
    return jnp.where(slots >= 0, ids[jnp.maximum(slots, 0)], -1)


def _slots(ids: Array, references: Array) -> Array:
    keys = jnp.where(ids >= 0, ids, _SENTINEL)
    position = jnp.minimum(jnp.searchsorted(keys, references), ids.shape[0] - 1)
    return jnp.where(references >= 0, position, -1).astype(jnp.int32)


def _members(sorted_ids: Array, references: Array) -> Array:
    position = jnp.minimum(
        jnp.searchsorted(sorted_ids, references), sorted_ids.shape[0] - 1
    )
    return (
        (references >= 0)
        & (references != _SENTINEL)
        & (sorted_ids[position] == references)
    )


def _vertex_record_owners(parts: AdaptiveSimplexParts, vertex_ids: Array) -> Array:
    """Local shard kernel assigning one authority to each allocated raw vertex."""
    axis = parts.axis_name
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )
    rank = jax.lax.axis_index(axis)
    owners = jnp.full(vertex_ids.shape, parts.part_count, jnp.int32)

    def visit(_: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        ids, source_rank, owners = carry
        keys = jnp.where(ids >= 0, ids, _SENTINEL)
        position = jnp.minimum(jnp.searchsorted(keys, vertex_ids), ids.shape[0] - 1)
        match = (vertex_ids >= 0) & (ids[position] == vertex_ids)
        owners = jnp.minimum(owners, jnp.where(match, source_rank, parts.part_count))
        return (
            jax.lax.ppermute(ids, axis, ring),
            jax.lax.ppermute(source_rank, axis, ring),
            owners,
        )

    _, _, owners = jax.lax.fori_loop(
        0, parts.part_count, visit, (vertex_ids, rank, owners)
    )
    return jnp.where(vertex_ids >= 0, owners, -1)


def _family_roots(parents: Array) -> tuple[Array, Array]:
    capacity = parents.shape[0]
    roots = jnp.arange(capacity, dtype=jnp.int32)

    def proceed(carry: tuple[Array, Array, Array]) -> Array:
        _, iterations, changed = carry
        return changed & (iterations < capacity)

    def step(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        roots, iterations, _ = carry
        parent = jnp.clip(parents[roots] // 2, -1, capacity - 1)
        next_roots = jnp.where(parent >= 0, parent, roots)
        return next_roots, iterations + 1, jnp.any(next_roots != roots)

    roots, _, changed = jax.lax.while_loop(
        proceed, step, (roots, jnp.asarray(0, jnp.int32), jnp.asarray(True))
    )
    valid = jnp.all((parents >= -1) & (parents < 2 * capacity)) & ~changed
    return roots, valid


def _validated_forest_roots(state: AdaptiveSimplexState) -> tuple[Array, Array]:
    """Validate allocated scientific parent/child links, ignoring unused padding."""
    ids, parents, children = state.mesh.cell_ids, state.parents, state.children
    allocated = ids >= 0
    capacity = ids.shape[0]
    roots, roots_valid = _family_roots(jnp.where(allocated, parents, -1))
    father = jnp.clip(parents // 2, 0, capacity - 1)
    parent_valid = (parents == -1) | (
        (parents >= 0) & (parents < 2 * capacity) & allocated[father]
    )
    slots = jnp.clip(children, 0, capacity - 1)
    pair = jnp.all(children == -1, axis=1) | jnp.all(
        (children >= 0)
        & (children < capacity)
        & allocated[slots]
        & (
            parents[slots]
            == 2 * jnp.arange(capacity, dtype=jnp.int32)[:, None]
            + jnp.arange(2, dtype=jnp.int32)
        ),
        axis=1,
    )
    ordered = jnp.all(~(allocated[1:] & ~allocated[:-1]))
    ordered &= jnp.all(~(allocated[1:] & allocated[:-1]) | (ids[1:] > ids[:-1]))
    active_valid = jnp.all(~state.mesh.cell_active | (allocated & ~state.retired))
    return roots, roots_valid & ordered & active_valid & jnp.all(
        ~allocated | (parent_valid & pair)
    )


def _source_owner_support(
    ids: Array,
    known: Array,
    active: Array,
    owners: Array,
    children: Array,
) -> tuple[Array, Array]:
    """Resolve complete source-epoch leaf coverage and its canonical minimum."""
    capacity = ids.shape[0]
    keys = jnp.where(ids >= 0, ids, _SENTINEL)
    positions = jnp.minimum(jnp.searchsorted(keys, children), capacity - 1)
    present = (children >= 0) & (ids[positions] == children)
    pair = jnp.all(present, axis=1) & known
    missing = jnp.any(known[:, None] & (children >= 0) & ~present)
    source = known & active
    minimum = jnp.where(source, owners, jnp.iinfo(jnp.int32).max)
    count = source.astype(jnp.int32)
    complete = source

    def proceed(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
        return carry[4] & (carry[3] < capacity)

    def step(
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        complete, minimum, count, iteration, _ = carry
        next_complete = source | (pair & jnp.all(complete[positions], axis=1))
        next_count = source.astype(jnp.int32) + jnp.where(
            pair, jnp.sum(count[positions], axis=1, dtype=jnp.int32), 0
        )
        next_minimum = jnp.where(
            source,
            owners,
            jnp.where(
                next_complete,
                jnp.min(minimum[positions], axis=1),
                jnp.iinfo(jnp.int32).max,
            ),
        )
        changed = jnp.any(
            (next_complete != complete)
            | (next_minimum != minimum)
            | (next_count != count)
        )
        return next_complete, next_minimum, next_count, iteration + 1, changed

    complete, minimum, count, _, changed = jax.lax.while_loop(
        proceed,
        step,
        (complete, minimum, count, jnp.asarray(0, jnp.int32), jnp.asarray(True)),
    )
    valid = ~missing & ~changed & ~jnp.any(source & (count != 1))
    return jnp.where(complete, minimum, -1), valid


def _inherit_tree_owners(
    parents: Array, source: Array, allocated: Array
) -> tuple[Array, Array]:
    capacity = parents.shape[0]
    father = jnp.clip(parents // 2, 0, capacity - 1)
    owners = jnp.where(allocated, source, -1)

    def proceed(carry: tuple[Array, Array, Array]) -> Array:
        return carry[2] & (carry[1] < capacity)

    def step(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        owners, iteration, _ = carry
        inherited = jnp.where((parents >= 0) & allocated, owners[father], -1)
        next_owners = jnp.where(allocated & (source >= 0), source, inherited)
        return next_owners, iteration + 1, jnp.any(next_owners != owners)

    owners, _, changed = jax.lax.while_loop(
        proceed,
        step,
        (owners, jnp.asarray(0, jnp.int32), jnp.asarray(True)),
    )
    return owners, ~changed


def _inherit_solver_owners_local(
    current: AdaptiveSimplexState,
    source: AdaptiveSimplexState,
    source_owners: Array,
    part_count: int,
    axis: str,
) -> tuple[Array, Array]:
    """One numerical part operation shared by physical and archived-logical execution."""
    ring = tuple((rank, (rank + 1) % part_count) for rank in range(part_count))
    ids = current.mesh.cell_ids
    allocated = ids >= 0
    source_allocated = source.mesh.cell_ids >= 0
    capacity = ids.shape[0]
    initial_parent_ids = _ids(
        source.mesh.cell_ids, jnp.clip(source.parents // 2, -1, capacity - 1)
    )
    current_parent_ids = _ids(ids, jnp.clip(current.parents // 2, -1, capacity - 1))
    initial_children_ids = _ids(
        source.mesh.cell_ids, jnp.clip(source.children, -1, capacity - 1)
    )
    packet = (
        source.mesh.cell_ids,
        source.mesh.cell_active,
        source_owners,
        initial_parent_ids,
        initial_children_ids,
    )
    source_active = jnp.zeros_like(allocated)
    owners = jnp.full(ids.shape, -1, jnp.int32)
    children = jnp.full((capacity, 2), -1, jnp.int64)
    coverage = jnp.zeros(ids.shape, jnp.int32)
    _, current_valid = _validated_forest_roots(current)
    _, source_valid = _validated_forest_roots(source)
    valid = current_valid & source_valid
    valid &= ~jnp.any(
        source.mesh.cell_active & ((source_owners < 0) | (source_owners >= part_count))
    )

    def visit(
        _: int,
        carry: tuple[
            tuple[Array, Array, Array, Array, Array], Array, Array, Array, Array, Array
        ],
    ) -> tuple[
        tuple[Array, Array, Array, Array, Array], Array, Array, Array, Array, Array
    ]:
        incoming, active, owners, children, coverage, valid = carry
        (
            incoming_ids,
            incoming_active,
            incoming_owners,
            incoming_parents,
            incoming_children,
        ) = incoming
        keys = jnp.where(incoming_ids >= 0, incoming_ids, _SENTINEL)
        position = jnp.minimum(jnp.searchsorted(keys, ids), capacity - 1)
        match = allocated & (incoming_ids[position] == ids)
        valid &= ~jnp.any(match & (incoming_parents[position] != current_parent_ids))
        active |= match & incoming_active[position]
        owners = jnp.where(match, incoming_owners[position], owners)
        children = jnp.where(match[:, None], incoming_children[position], children)
        coverage += match.astype(jnp.int32)
        incoming = jax.tree_util.tree_map(
            lambda value: jax.lax.ppermute(value, axis, ring), incoming
        )
        return incoming, active, owners, children, coverage, valid

    _, source_active, owners, children, coverage, valid = jax.lax.fori_loop(
        0,
        part_count,
        visit,
        (packet, source_active, owners, children, coverage, valid),
    )
    before = jax.lax.psum(jnp.sum(source_allocated, dtype=jnp.int32), axis)
    after = jax.lax.psum(jnp.sum(coverage == 1, dtype=jnp.int32), axis)
    valid &= (before == after) & ~jnp.any(coverage > 1)
    support, complete = _source_owner_support(
        ids, coverage == 1, source_active, owners, children
    )
    inherited, converged = _inherit_tree_owners(current.parents, support, allocated)
    valid &= complete & converged & ~jnp.any(current.mesh.cell_active & (inherited < 0))
    status = _global_flags(
        current.status_flags | source.status_flags | jnp.where(valid, 0, _INVALID), axis
    )
    return jnp.where(current.mesh.cell_active, inherited, -1), status


@eqx.filter_jit
def inherit_solver_cell_owners(
    parts: AdaptiveSimplexParts,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    initial_solver_cell_owners: Array,
    /,
) -> tuple[Array, Array]:
    """Preserve graph ownership through actual refinement/coarsening history.

    Initial active cells retain ownership precedence. Restored source parents
    consume complete source-child support and take its minimum owner. New
    descendants inherit the nearest covered source ownership, including a parent
    coarsened and refined again before publication. No graph is repartitioned.
    """
    shape = states.mesh.cell_ids.shape
    if (
        len(shape) != 2
        or shape[0] != parts.part_count
        or initial_states.mesh.cell_ids.shape != shape
        or initial_solver_cell_owners.shape != shape
        or initial_solver_cell_owners.dtype != jnp.int32
    ):
        raise ValueError(
            "Solver ownership inheritance requires matched part-sharded raw cell capacities."
        )
    axis = parts.axis_name
    spec = PartitionSpec(axis)

    def local(
        current_block: AdaptiveSimplexState,
        source_block: AdaptiveSimplexState,
        owner_block: Array,
    ) -> tuple[Array, Array]:
        current, source, source_owners = jax.tree_util.tree_map(
            lambda value: value[0], (current_block, source_block, owner_block)
        )
        owners, status = _inherit_solver_owners_local(
            current, source, source_owners, parts.part_count, axis
        )
        return owners[None], status[None]

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec),
        out_specs=(spec, spec),
        check_vma=False,
    )(states, initial_states, initial_solver_cell_owners)


@eqx.filter_jit
def replay_logical_solver_cell_owners(
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    initial_solver_cell_owners: Array,
    /,
    *,
    neighbor_pairs: tuple[tuple[int, int], ...],
    axis_name: str,
) -> tuple[Array, Array]:
    """Revalidate archived logical ownership on the actual current devices.

    The saved leading part axis is a named numerical batch, not a fabricated
    device mesh. This invokes the identical part operation used by shard_map.
    It is cold logical revalidation, never evidence of distributed execution.
    """
    shape = states.mesh.cell_ids.shape
    if (
        len(shape) != 2
        or shape[0] < 1
        or initial_states.mesh.cell_ids.shape != shape
        or initial_solver_cell_owners.shape != shape
        or initial_solver_cell_owners.dtype != jnp.int32
        or not isinstance(axis_name, str)
        or not axis_name
    ):
        raise ValueError(
            "Logical solver ownership replay requires matched saved part arrays and an explicit axis."
        )
    count = shape[0]
    if (
        len(set(neighbor_pairs)) != len(neighbor_pairs)
        or any(
            source == target
            or source < 0
            or target < 0
            or source >= count
            or target >= count
            for source, target in neighbor_pairs
        )
        or any(
            (target, source) not in neighbor_pairs for source, target in neighbor_pairs
        )
    ):
        raise ValueError("Saved logical routes must be valid, unique and reciprocal.")

    def local(
        current: AdaptiveSimplexState, source: AdaptiveSimplexState, owners: Array
    ) -> tuple[Array, Array]:
        return _inherit_solver_owners_local(current, source, owners, count, axis_name)

    return jax.vmap(local, in_axes=(0, 0, 0), out_axes=(0, 0), axis_name=axis_name)(
        states,
        initial_states,
        initial_solver_cell_owners,
    )


def _merge(
    current: dict[str, Array],
    incoming: dict[str, Array],
    selected: Array,
) -> tuple[dict[str, Array], Array]:
    capacity = current["ids"].shape[0]
    keys = jnp.concatenate(
        (current["ids"], jnp.where(selected, incoming["ids"], _SENTINEL))
    )
    order = jnp.argsort(keys, stable=True)
    ordered = keys[order]
    unique = (ordered != _SENTINEL) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), ordered[1:] != ordered[:-1])
    )
    pick = jnp.nonzero(unique, size=capacity, fill_value=2 * capacity - 1)[0]
    lanes = order[pick]
    count = jnp.sum(unique, dtype=jnp.int32)
    merged = {
        name: jnp.concatenate((current[name], incoming[name]), axis=0)[lanes]
        for name in current
    }
    merged["ids"] = jnp.where(jnp.arange(capacity) < count, ordered[pick], _SENTINEL)
    # Authoritative packets must agree on every replica, including history.
    duplicate = (ordered[1:] == ordered[:-1]) & (ordered[1:] != _SENTINEL)
    conflict = jnp.asarray(False)
    for name in current:
        values = jnp.concatenate((current[name], incoming[name]), axis=0)[order]
        different = values[1:] != values[:-1]
        if different.ndim > 1:
            different = jnp.any(different, axis=tuple(range(1, different.ndim)))
        conflict |= jnp.any(duplicate & different)
    return merged, jnp.where(count > capacity, _CAPACITY, 0) | jnp.where(
        conflict, _INVALID, 0
    )


@eqx.filter_jit
def migrate_simplex_forest(
    parts: AdaptiveSimplexParts,
    forest: AdaptiveSimplexState,
    cell_owners: Array,
    vertex_owners: Array,
    cell_history: tuple[Array, ...],
    vertex_history: tuple[Array, ...],
    target: SimplexNeighborhoodWorkset,
    *,
    include_ghost_families: bool = False,
) -> tuple[
    AdaptiveSimplexState, Array, Array, tuple[Array, ...], tuple[Array, ...], Array
]:
    """Relocate all allocated records, including retired/removed records.

    Only bounded owner packets circulate. Controls retain their rank-local
    cumulative evidence; allocator ID cursors are collectively reconciled.
    Geometry, family references, constraints and typed payloads use stable IDs.
    """
    axis = parts.axis_name
    spec = PartitionSpec(axis)
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )

    def local(
        block: AdaptiveSimplexState,
        cell_owner_block: Array,
        vertex_owner_block: Array,
        cell_payload_block: tuple[Array, ...],
        vertex_payload_block: tuple[Array, ...],
        target_block: SimplexNeighborhoodWorkset,
    ) -> tuple[
        AdaptiveSimplexState, Array, Array, tuple[Array, ...], tuple[Array, ...], Array
    ]:
        state, co, vo, ch, vh, work = jax.tree_util.tree_map(
            lambda value: value[0],
            (
                block,
                cell_owner_block,
                vertex_owner_block,
                cell_payload_block,
                vertex_payload_block,
                target_block,
            ),
        )
        rank = jax.lax.axis_index(axis)
        mesh = state.mesh
        c, v = mesh.cell_ids.shape[0], mesh.vertex_ids.shape[0]
        roots, root_valid = _validated_forest_roots(state)
        recipients = None
        if include_ghost_families:
            recipients = jnp.zeros((c, parts.part_count), jnp.bool_)

            def request(
                _: int,
                carry: tuple[SimplexNeighborhoodWorkset, Array, Array],
            ) -> tuple[SimplexNeighborhoodWorkset, Array, Array]:
                packet, source_rank, requested = carry
                pos = jnp.minimum(
                    jnp.searchsorted(packet.cell_ids, mesh.cell_ids),
                    packet.cell_ids.shape[0] - 1,
                )
                match = (
                    (mesh.cell_ids >= 0)
                    & packet.cell_valid[pos]
                    & (packet.cell_ids[pos] == mesh.cell_ids)
                )
                family = jnp.zeros((c,), jnp.bool_).at[roots].max(match)[roots]
                requested = requested.at[:, source_rank].set(family)
                return (
                    jax.tree_util.tree_map(
                        lambda value: jax.lax.ppermute(value, axis, ring), packet
                    ),
                    jax.lax.ppermute(source_rank, axis, ring),
                    requested,
                )

            _, _, recipients = jax.lax.fori_loop(
                0, parts.part_count, request, (work, rank, recipients)
            )
            owner = co
        else:
            owner = jnp.full((c,), parts.part_count, jnp.int32)

            def assign(
                _: int, carry: tuple[SimplexNeighborhoodWorkset, Array]
            ) -> tuple[SimplexNeighborhoodWorkset, Array]:
                packet, owner = carry
                pos = jnp.minimum(
                    jnp.searchsorted(packet.cell_ids, mesh.cell_ids),
                    packet.cell_ids.shape[0] - 1,
                )
                match = (
                    (mesh.cell_ids >= 0)
                    & packet.cell_valid[pos]
                    & (packet.cell_ids[pos] == mesh.cell_ids)
                )
                owner = jnp.minimum(
                    owner, jnp.where(match, packet.cell_owner[pos], parts.part_count)
                )
                return jax.tree_util.tree_map(
                    lambda value: jax.lax.ppermute(value, axis, ring), packet
                ), owner

            _, owner = jax.lax.fori_loop(0, parts.part_count, assign, (work, owner))
            owner = (
                jnp.full((c,), parts.part_count, jnp.int32).at[roots].min(owner)[roots]
            )
            # Entirely inactive families remain authoritative on their former owner.
            owner = jnp.where(owner == parts.part_count, co[roots], owner)
        cells = {
            "ids": jnp.where(
                (mesh.cell_ids >= 0) & (co == rank), mesh.cell_ids, _SENTINEL
            ),
            "owner": owner,
            "cells": _ids(mesh.vertex_ids, mesh.cells),
            "tuples": _ids(mesh.vertex_ids, state.tuples),
            "active": mesh.cell_active,
            "parents": _ids(mesh.cell_ids, state.parents // 2),
            "ordinal": state.parents % 2,
            "children": _ids(mesh.cell_ids, state.children),
            "bisection": _ids(mesh.vertex_ids, state.bisection_vertices),
            **{name: getattr(state, name) for name in _CELL_FIELDS},
            **{f"payload_{i}": value for i, value in enumerate(ch)},
        }
        if recipients is not None:
            cells["recipients"] = recipients
        empty = {**cells, "ids": jnp.full_like(cells["ids"], _SENTINEL)}

        def cell_visit(
            _: int, carry: tuple[dict[str, Array], dict[str, Array], Array]
        ) -> tuple[dict[str, Array], dict[str, Array], Array]:
            packet, resident, status = carry
            selected = (
                packet["recipients"][:, rank]
                if include_ghost_families
                else packet["owner"] == rank
            )
            resident, flags = _merge(
                resident, packet, (packet["ids"] != _SENTINEL) & selected
            )
            return (
                jax.tree_util.tree_map(
                    lambda value: jax.lax.ppermute(value, axis, ring), packet
                ),
                resident,
                status | flags,
            )

        _, resident, status = jax.lax.fori_loop(
            0, parts.part_count, cell_visit, (cells, empty, state.clocks[2] | work.status)
        )
        used = resident["ids"] != _SENTINEL
        before_cells = jnp.sum((mesh.cell_ids >= 0) & (co == rank), dtype=jnp.int32)
        after_cells = jnp.sum(used, dtype=jnp.int32)
        invalid_owners = jnp.any(
            (mesh.cell_ids >= 0) & ((co < 0) | (co >= parts.part_count))
        )
        invalid_owners |= jnp.any(
            (mesh.vertex_ids >= 0) & ((vo < 0) | (vo >= parts.part_count))
        )
        lost_cells = (
            False
            if include_ghost_families
            else jax.lax.psum(before_cells, axis) != jax.lax.psum(after_cells, axis)
        )
        status |= jnp.where(invalid_owners | lost_cells, _INVALID, 0)
        required = jnp.concatenate(
            (
                jnp.where(used[:, None], resident["cells"], -1).reshape(-1),
                jnp.where(used[:, None], resident["tuples"], -1).reshape(-1),
                jnp.where(used, resident["bisection"], -1),
            )
        )
        edge_slots = jnp.stack(
            (state.protected_codes // v, state.protected_codes % v), axis=1
        )
        edge_ids = jnp.where(
            (state.protected_codes != _SENTINEL)[:, None],
            _ids(mesh.vertex_ids, jnp.minimum(edge_slots, v - 1)),
            -1,
        )
        referenced = jnp.zeros((v,), jnp.bool_)

        def references(
            _: int, carry: tuple[dict[str, Array], Array]
        ) -> tuple[dict[str, Array], Array]:
            packet, seen = carry
            live = packet["ids"] != _SENTINEL
            refs = jnp.concatenate(
                (
                    jnp.where(live[:, None], packet["cells"], -1).reshape(-1),
                    jnp.where(live[:, None], packet["tuples"], -1).reshape(-1),
                    jnp.where(live, packet["bisection"], -1),
                )
            )
            seen |= _members(jnp.sort(refs), mesh.vertex_ids)
            return jax.tree_util.tree_map(
                lambda value: jax.lax.ppermute(value, axis, ring), packet
            ), seen

        _, referenced = jax.lax.fori_loop(
            0, parts.part_count, references, (cells, referenced)
        )
        required = jnp.concatenate(
            (required, jnp.where((vo == rank) & ~referenced, mesh.vertex_ids, -1))
        )
        vertices = {
            "ids": jnp.where(
                (mesh.vertex_ids >= 0) & (vo == rank), mesh.vertex_ids, _SENTINEL
            ),
            "coordinates": mesh.coordinates,
            "active": mesh.vertex_active,
            "parents": _ids(mesh.vertex_ids, state.vertex_parents),
            **{name: getattr(state, name) for name in _VERTEX_FIELDS},
            **{f"payload_{i}": value for i, value in enumerate(vh)},
        }
        vertex_empty = {**vertices, "ids": jnp.full_like(vertices["ids"], _SENTINEL)}

        def vertex_round(
            carry: tuple[dict[str, Array], Array, Array, Array],
        ) -> tuple[dict[str, Array], Array, Array, Array]:
            result, status, iterations, _ = carry
            previous_ids = result["ids"]
            refs = jnp.concatenate(
                (
                    required,
                    jnp.where(
                        (result["ids"] != _SENTINEL)[:, None], result["parents"], -1
                    ).reshape(-1),
                )
            )
            sorted_refs = jnp.sort(refs)

            def visit(
                _: int, carry: tuple[dict[str, Array], Array, dict[str, Array], Array]
            ) -> tuple[dict[str, Array], Array, dict[str, Array], Array]:
                packet, edges, current, status = carry

                def edge_visit(_: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
                    constraint_packet, selected = carry
                    touching = jnp.any(
                        _members(sorted_refs, constraint_packet), axis=1
                    ) & jnp.all(constraint_packet >= 0, axis=1)
                    constraint_ids = jnp.sort(
                        jnp.where(touching[:, None], constraint_packet, -1).reshape(-1)
                    )
                    selected |= _members(constraint_ids, packet["ids"])
                    return jax.lax.ppermute(constraint_packet, axis, ring), selected

                selected = _members(sorted_refs, packet["ids"])
                _, selected = jax.lax.fori_loop(
                    0, parts.part_count, edge_visit, (edges, selected)
                )
                selected &= packet["ids"] != _SENTINEL
                current, flags = _merge(current, packet, selected)
                return (
                    jax.tree_util.tree_map(
                        lambda value: jax.lax.ppermute(value, axis, ring), packet
                    ),
                    jax.lax.ppermute(edges, axis, ring),
                    current,
                    status | flags,
                )

            _, _, result, status = jax.lax.fori_loop(
                0, parts.part_count, visit, (vertices, edge_ids, result, status)
            )
            changed = (
                jax.lax.psum(
                    jnp.any(result["ids"] != previous_ids).astype(jnp.int32), axis
                )
                > 0
            )
            return result, status, iterations + 1, changed

        def vertex_proceed(carry: tuple[dict[str, Array], Array, Array, Array]) -> Array:
            return carry[3] & (carry[2] < v * parts.part_count)

        retained_vertices, status, _, _ = jax.lax.while_loop(
            vertex_proceed,
            vertex_round,
            (
                vertex_empty,
                status | jnp.where(root_valid, 0, _INVALID),
                jnp.asarray(0, jnp.int32),
                jnp.asarray(True),
            ),
        )
        cell_ids = jnp.where(used, resident["ids"], -1)
        vertex_ids = jnp.where(
            retained_vertices["ids"] != _SENTINEL, retained_vertices["ids"], -1
        )
        rows = _slots(vertex_ids, resident["cells"])
        tuples = _slots(vertex_ids, resident["tuples"])
        parent_slots = _slots(cell_ids, resident["parents"])
        child_slots = _slots(cell_ids, resident["children"])
        bisect_slots = _slots(vertex_ids, resident["bisection"])
        vertex_parents = _slots(vertex_ids, retained_vertices["parents"])

        # Padding references are immaterial and must not veto an empty part.
        def missing(references: Array, ids: Array, mask: Array) -> Array:
            keys = jnp.where(ids >= 0, ids, _SENTINEL)
            pos = jnp.minimum(jnp.searchsorted(keys, references), ids.shape[0] - 1)
            shape = (mask.shape[0],) + (1,) * (references.ndim - 1)
            return jnp.any(
                mask.reshape(shape) & (references >= 0) & (ids[pos] != references)
            )

        invalid = missing(resident["cells"], vertex_ids, used) | missing(
            resident["tuples"], vertex_ids, used
        )
        invalid |= missing(resident["parents"], cell_ids, used) | missing(
            resident["children"], cell_ids, used
        )
        invalid |= missing(resident["bisection"], vertex_ids, used)
        invalid |= missing(retained_vertices["parents"], vertex_ids, vertex_ids >= 0)
        status |= jnp.where(invalid, _INVALID, 0)
        protected = jnp.full_like(state.protected_codes, _SENTINEL)

        def constraints(
            _: int, carry: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            edges, codes, status = carry
            slots = _slots(vertex_ids, edges)
            keys = jnp.where(vertex_ids >= 0, vertex_ids, _SENTINEL)
            positions = jnp.minimum(jnp.searchsorted(keys, edges), v - 1)
            present = jnp.all((edges >= 0) & (vertex_ids[positions] == edges), axis=1)
            incoming = jnp.where(
                present,
                jnp.minimum(slots[:, 0], slots[:, 1]).astype(jnp.int64) * v
                + jnp.maximum(slots[:, 0], slots[:, 1]),
                _SENTINEL,
            )
            sorted_codes = jnp.sort(jnp.concatenate((codes, incoming)))
            fresh = (sorted_codes != _SENTINEL) & jnp.concatenate(
                (jnp.ones(1, jnp.bool_), sorted_codes[1:] != sorted_codes[:-1])
            )
            selected = jnp.nonzero(
                fresh, size=codes.shape[0], fill_value=sorted_codes.shape[0] - 1
            )[0]
            count = jnp.sum(fresh, dtype=jnp.int32)
            codes = jnp.where(
                jnp.arange(codes.shape[0]) < count, sorted_codes[selected], _SENTINEL
            )
            return (
                jax.lax.ppermute(edges, axis, ring),
                codes,
                status | jnp.where(count > codes.shape[0], _CAPACITY, 0),
            )

        _, protected, status = jax.lax.fori_loop(
            0, parts.part_count, constraints, (edge_ids, protected, status)
        )
        new_vertex_owner = _vertex_record_owners(parts, vertex_ids)
        before_vertices = jnp.sum((mesh.vertex_ids >= 0) & (vo == rank), dtype=jnp.int32)
        after_vertices = jnp.sum(
            (vertex_ids >= 0) & (new_vertex_owner == rank), dtype=jnp.int32
        )
        if include_ghost_families:
            position = jnp.minimum(
                jnp.searchsorted(resident["ids"], work.cell_ids), c - 1
            )
            missing_cells = jnp.any(
                work.cell_valid & (resident["ids"][position] != work.cell_ids)
            )
            status |= jnp.where(missing_cells, _INVALID, 0)
        status |= jnp.where(
            jax.lax.psum(before_vertices, axis) != jax.lax.psum(after_vertices, axis),
            _INVALID,
            0,
        )
        active = used & resident["active"]
        rows = jnp.maximum(rows, 0)
        neighbors = masked_simplex_facet_neighbors(rows, active)
        new_mesh = eqx.tree_at(
            lambda value: (
                value.coordinates,
                value.vertex_ids,
                value.vertex_active,
                value.cells,
                value.cell_ids,
                value.cell_active,
                value.facet_neighbors,
            ),
            mesh,
            (
                retained_vertices["coordinates"],
                vertex_ids,
                (vertex_ids >= 0) & retained_vertices["active"],
                rows,
                cell_ids,
                active,
                neighbors,
            ),
        )
        changes = {name: resident[name] for name in _CELL_FIELDS}
        changes.update({name: retained_vertices[name] for name in _VERTEX_FIELDS})
        changes.update(
            mesh=new_mesh,
            tuples=jnp.maximum(tuples, 0),
            parents=jnp.where(
                parent_slots >= 0, parent_slots * 2 + resident["ordinal"], -1
            ),
            children=child_slots,
            bisection_vertices=bisect_slots,
            vertex_parents=vertex_parents,
            protected_codes=protected,
            vertex_half_facets=_vertex_half_facets(rows, active, neighbors, v),
            cursors=jnp.stack(
                (
                    jnp.sum(vertex_ids >= 0, dtype=jnp.int64),
                    jnp.sum(used, dtype=jnp.int64),
                    jax.lax.pmax(state.cursors[2], axis),
                    jax.lax.pmax(state.cursors[3], axis),
                )
            ),
        )
        names = tuple(changes)
        result = eqx.tree_at(
            lambda value: tuple(getattr(value, name) for name in names),
            state,
            tuple(changes.values()),
        )
        return jax.tree_util.tree_map(
            lambda value: value[None],
            (
                result,
                jnp.where(used, rank, -1).astype(jnp.int32),
                jnp.where(vertex_ids >= 0, new_vertex_owner, -1).astype(jnp.int32),
                tuple(resident[f"payload_{i}"] for i in range(len(ch))),
                tuple(retained_vertices[f"payload_{i}"] for i in range(len(vh))),
                status,
            ),
        )

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec, spec, spec, spec),
        out_specs=(spec, spec, spec, spec, spec, spec),
        check_vma=False,
    )(forest, cell_owners, vertex_owners, cell_history, vertex_history, target)
