#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._morton import canonical_morton_order, MortonAddressPlan


_UINT64_MAX = np.iinfo(np.uint64).max


class MortonHierarchyBuildEvidence(NonTrainableState, StrictModule):
    """Completion and resource evidence for one point hierarchy build."""

    successful: jax.Array
    active_points: jax.Array
    invalid_points: jax.Array
    stable_ids_unique: jax.Array
    required_nodes: jax.Array
    node_capacity: jax.Array
    active_nodes: jax.Array
    active_leaves: jax.Array
    maximum_leaf_occupancy: jax.Array


class MortonPointHierarchyState(NonTrainableState, StrictModule):
    """Packed fixed-capacity sparse quadtree/octree topology."""

    sorted_codes: jax.Array
    sorted_stable_ids: jax.Array
    sorted_active: jax.Array
    storage_to_logical: jax.Array
    logical_to_storage: jax.Array
    sorted_point_leaf_slots: jax.Array
    logical_point_leaf_slots: jax.Array
    node_prefixes: jax.Array
    node_levels: jax.Array
    node_active: jax.Array
    node_is_leaf: jax.Array
    node_parents: jax.Array
    node_children: jax.Array
    node_item_starts: jax.Array
    node_item_counts: jax.Array
    node_centers: jax.Array
    node_half_widths: jax.Array
    root_slot: jax.Array
    epoch: jax.Array
    evidence: MortonHierarchyBuildEvidence


class MortonHierarchyTransition(NonTrainableState, StrictModule):
    """Atomic point-hierarchy refresh result."""

    candidate: MortonPointHierarchyState
    accepted: MortonPointHierarchyState
    accepted_candidate: jax.Array
    refitted: jax.Array
    rebuilt: jax.Array


class MortonPointHierarchyPlan(StrictModule):
    """Build deterministic fixed-capacity sparse Morton point hierarchies."""

    address_plan: MortonAddressPlan
    point_capacity: int = eqx.field(static=True)
    node_capacity: int = eqx.field(static=True)
    target_leaf_occupancy: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        point_capacity: int,
        *,
        node_capacity: int | None = None,
        target_leaf_occupancy: int = 16,
    ) -> None:
        points = int(point_capacity)
        if points < 1:
            raise ValueError("point_capacity must be positive.")
        target = int(target_leaf_occupancy)
        if target < 0:
            raise ValueError("target_leaf_occupancy must be nonnegative.")
        rectangular_capacity = (address_plan.maximum_depth + 1) * points
        nodes = rectangular_capacity if node_capacity is None else int(node_capacity)
        if nodes < 1 or nodes > rectangular_capacity:
            raise ValueError(
                "node_capacity must be positive and no larger than "
                "(maximum_depth + 1) * point_capacity."
            )
        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "point_capacity", points)
        object.__setattr__(self, "node_capacity", nodes)
        object.__setattr__(self, "target_leaf_occupancy", target)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "morton-point-hierarchy-plan",
                    "address_plan_id": address_plan.plan_id,
                    "point_capacity": points,
                    "node_capacity": nodes,
                    "target_leaf_occupancy": target,
                }
            ),
        )

    def build(
        self,
        points: jax.Array,
        *,
        active_mask: jax.Array | None = None,
        stable_ids: jax.Array | None = None,
        epoch: int | jax.Array = 0,
    ) -> MortonPointHierarchyState:
        positions = jnp.asarray(points)
        expected_shape = (self.point_capacity, self.address_plan.dimension)
        if positions.shape != expected_shape:
            raise ValueError(
                f"points must have shape {expected_shape}; got {positions.shape}."
            )
        if active_mask is None:
            active = jnp.ones((self.point_capacity,), dtype=bool)
        else:
            active = jnp.asarray(active_mask, dtype=bool)
            if active.shape != (self.point_capacity,):
                raise ValueError("active_mask must match point_capacity.")
        if stable_ids is None:
            identifiers = jnp.arange(self.point_capacity, dtype=jnp.int64)
        else:
            identifiers = jnp.asarray(stable_ids)
            if identifiers.shape != (self.point_capacity,):
                raise ValueError("stable_ids must match point_capacity.")
            if not jnp.issubdtype(identifiers.dtype, jnp.integer):
                raise TypeError("stable_ids must have integer dtype.")

        encoding = self.address_plan.encode(positions)
        valid = active & encoding.in_domain
        order = canonical_morton_order(encoding.codes, identifiers, valid)
        inverse = (
            jnp.zeros_like(order)
            .at[order]
            .set(jnp.arange(self.point_capacity, dtype=jnp.int32))
        )
        sorted_codes = encoding.codes[order]
        sorted_ids = identifiers[order]
        sorted_valid = valid[order]
        active_count = jnp.sum(sorted_valid, dtype=jnp.int32)
        slots = jnp.arange(self.point_capacity, dtype=jnp.int32)

        identifier_order = jnp.lexsort((identifiers, (~active).astype(jnp.int32)))
        identifiers_by_id = identifiers[identifier_order]
        active_by_id = active[identifier_order]
        duplicate_id = (
            active_by_id[1:]
            & active_by_id[:-1]
            & (identifiers_by_id[1:] == identifiers_by_id[:-1])
        )
        stable_ids_unique = ~jnp.any(duplicate_id)

        prefixes_by_level: list[jax.Array] = []
        starts_by_level: list[jax.Array] = []
        counts_by_level: list[jax.Array] = []
        unique_by_level: list[jax.Array] = []
        for level in range(self.address_plan.maximum_depth + 1):
            point_prefixes = self.address_plan.prefix(sorted_codes, level)
            boundary = sorted_valid & jnp.concatenate(
                (
                    jnp.asarray([True]),
                    (~sorted_valid[:-1]) | (point_prefixes[1:] != point_prefixes[:-1]),
                )
            )
            unique_count = jnp.sum(boundary, dtype=jnp.int32)
            starts = jnp.nonzero(
                boundary,
                size=self.point_capacity,
                fill_value=self.point_capacity,
            )[0].astype(jnp.int32)
            unique = slots < unique_count
            safe_starts = jnp.minimum(starts, self.point_capacity - 1)
            prefixes = jnp.where(
                unique,
                point_prefixes[safe_starts],
                jnp.asarray(_UINT64_MAX, dtype=jnp.uint64),
            )
            next_starts = jnp.concatenate((starts[1:], active_count[None]))
            counts = jnp.where(unique, next_starts - starts, 0).astype(jnp.int32)
            prefixes_by_level.append(prefixes)
            starts_by_level.append(jnp.where(unique, starts, 0))
            counts_by_level.append(counts)
            unique_by_level.append(unique)

        active_by_level: list[jax.Array] = []
        leaf_by_level: list[jax.Array] = []
        parent_positions_by_level: list[jax.Array] = []
        root_active = unique_by_level[0]
        active_by_level.append(root_active)
        root_split = root_active & (counts_by_level[0] > self.target_leaf_occupancy)
        leaf_by_level.append(root_active & ~root_split)
        parent_positions_by_level.append(
            jnp.full((self.point_capacity,), -1, dtype=jnp.int32)
        )
        previous_split = root_split
        for level in range(1, self.address_plan.maximum_depth + 1):
            parent_prefixes = prefixes_by_level[level] >> self.address_plan.dimension
            parent_positions = jnp.searchsorted(
                prefixes_by_level[level - 1], parent_prefixes, side="left"
            ).astype(jnp.int32)
            safe_parent = jnp.minimum(parent_positions, self.point_capacity - 1)
            parent_matches = (parent_positions < self.point_capacity) & (
                prefixes_by_level[level - 1][safe_parent] == parent_prefixes
            )
            node_active = (
                unique_by_level[level] & parent_matches & previous_split[safe_parent]
            )
            node_split = (
                node_active
                & (counts_by_level[level] > self.target_leaf_occupancy)
                & (level < self.address_plan.maximum_depth)
            )
            active_by_level.append(node_active)
            leaf_by_level.append(node_active & ~node_split)
            parent_positions_by_level.append(jnp.where(node_active, parent_positions, -1))
            previous_split = node_split

        branching = 1 << self.address_plan.dimension
        rectangular_capacity = (self.address_plan.maximum_depth + 1) * self.point_capacity
        rectangular_sentinel = rectangular_capacity
        children_by_level: list[jax.Array] = []
        digits = jnp.arange(branching, dtype=jnp.uint64)
        for level in range(self.address_plan.maximum_depth):
            child_prefixes = (
                prefixes_by_level[level][:, None] << self.address_plan.dimension
            ) | digits[None, :]
            child_positions = jnp.searchsorted(
                prefixes_by_level[level + 1], child_prefixes, side="left"
            ).astype(jnp.int32)
            safe_children = jnp.minimum(child_positions, self.point_capacity - 1)
            child_matches = (child_positions < self.point_capacity) & (
                prefixes_by_level[level + 1][safe_children] == child_prefixes
            )
            child_active = child_matches & active_by_level[level + 1][safe_children]
            child_rectangular = (level + 1) * self.point_capacity + child_positions
            children_by_level.append(
                jnp.where(child_active, child_rectangular, rectangular_sentinel)
            )
        children_by_level.append(
            jnp.full(
                (self.point_capacity, branching),
                rectangular_sentinel,
                dtype=jnp.int32,
            )
        )

        flat_active = jnp.concatenate(active_by_level)
        required_nodes = jnp.sum(flat_active, dtype=jnp.int32)
        selected_rectangular = jnp.nonzero(
            flat_active,
            size=self.node_capacity,
            fill_value=rectangular_sentinel,
        )[0].astype(jnp.int32)
        packed_slots = jnp.arange(self.node_capacity, dtype=jnp.int32)
        packed_active = packed_slots < jnp.minimum(required_nodes, self.node_capacity)
        safe_selected = jnp.minimum(selected_rectangular, rectangular_capacity - 1)

        flat_prefixes = jnp.concatenate(prefixes_by_level)
        flat_levels = jnp.repeat(
            jnp.arange(self.address_plan.maximum_depth + 1, dtype=jnp.int32),
            self.point_capacity,
        )
        flat_leaves = jnp.concatenate(leaf_by_level)
        flat_starts = jnp.concatenate(starts_by_level)
        flat_counts = jnp.concatenate(counts_by_level)
        parent_rectangular_parts = [
            jnp.full((self.point_capacity,), rectangular_sentinel, dtype=jnp.int32)
        ]
        for level in range(1, self.address_plan.maximum_depth + 1):
            parent_position = parent_positions_by_level[level]
            parent_rectangular_parts.append(
                jnp.where(
                    active_by_level[level],
                    (level - 1) * self.point_capacity + parent_position,
                    rectangular_sentinel,
                )
            )
        flat_parents = jnp.concatenate(parent_rectangular_parts)
        flat_children = jnp.concatenate(children_by_level, axis=0)

        rectangular_to_packed = jnp.full((rectangular_capacity + 1,), -1, dtype=jnp.int32)
        rectangular_to_packed = rectangular_to_packed.at[
            jnp.where(packed_active, selected_rectangular, rectangular_sentinel)
        ].set(jnp.where(packed_active, packed_slots, -1))

        node_prefixes = jnp.where(
            packed_active,
            flat_prefixes[safe_selected],
            jnp.asarray(0, dtype=jnp.uint64),
        )
        node_levels = jnp.where(packed_active, flat_levels[safe_selected], 0)
        node_is_leaf = packed_active & flat_leaves[safe_selected]
        node_item_starts = jnp.where(packed_active, flat_starts[safe_selected], 0)
        node_item_counts = jnp.where(packed_active, flat_counts[safe_selected], 0)
        parent_rectangular = jnp.where(
            packed_active,
            flat_parents[safe_selected],
            rectangular_sentinel,
        )
        node_parents = rectangular_to_packed[parent_rectangular]
        child_rectangular = jnp.where(
            packed_active[:, None],
            flat_children[safe_selected],
            rectangular_sentinel,
        )
        node_children = rectangular_to_packed[child_rectangular]
        leaf_slots = jnp.nonzero(
            node_is_leaf,
            size=self.node_capacity,
            fill_value=self.node_capacity,
        )[0].astype(jnp.int32)
        leaf_count = jnp.sum(node_is_leaf, dtype=jnp.int32)
        leaf_slot_valid = packed_slots < leaf_count
        safe_leaf_slots = jnp.minimum(leaf_slots, self.node_capacity - 1)
        leaf_starts = jnp.where(
            leaf_slot_valid,
            node_item_starts[safe_leaf_slots],
            self.point_capacity,
        )
        leaf_order = jnp.argsort(leaf_starts, stable=True)
        ordered_leaf_slots = leaf_slots[leaf_order]
        ordered_leaf_starts = leaf_starts[leaf_order]
        storage_slots = jnp.arange(self.point_capacity, dtype=jnp.int32)
        leaf_rank = jnp.searchsorted(ordered_leaf_starts, storage_slots, side="right") - 1
        sorted_point_leaf_slots = jnp.where(
            sorted_valid,
            ordered_leaf_slots[jnp.maximum(leaf_rank, 0)],
            -1,
        ).astype(jnp.int32)
        logical_point_leaf_slots = (
            jnp.full((self.point_capacity,), -1, dtype=jnp.int32)
            .at[order]
            .set(sorted_point_leaf_slots)
        )
        geometry = self.address_plan.cell_geometry(node_prefixes, node_levels)
        leaf_occupancy = jnp.where(node_is_leaf, node_item_counts, 0)
        successful = (
            (jnp.sum(active & ~encoding.in_domain, dtype=jnp.int32) == 0)
            & stable_ids_unique
            & (required_nodes <= self.node_capacity)
        )
        evidence = MortonHierarchyBuildEvidence(
            successful=successful,
            active_points=active_count,
            invalid_points=jnp.sum(active & ~encoding.in_domain, dtype=jnp.int32),
            stable_ids_unique=stable_ids_unique,
            required_nodes=required_nodes,
            node_capacity=jnp.asarray(self.node_capacity, dtype=jnp.int32),
            active_nodes=jnp.minimum(required_nodes, self.node_capacity),
            active_leaves=jnp.sum(node_is_leaf, dtype=jnp.int32),
            maximum_leaf_occupancy=jnp.max(leaf_occupancy, initial=0),
        )
        return MortonPointHierarchyState(
            sorted_codes=sorted_codes,
            sorted_stable_ids=sorted_ids,
            sorted_active=sorted_valid,
            storage_to_logical=order,
            logical_to_storage=inverse,
            sorted_point_leaf_slots=sorted_point_leaf_slots,
            logical_point_leaf_slots=logical_point_leaf_slots,
            node_prefixes=node_prefixes,
            node_levels=node_levels,
            node_active=packed_active,
            node_is_leaf=node_is_leaf,
            node_parents=node_parents,
            node_children=node_children,
            node_item_starts=node_item_starts,
            node_item_counts=node_item_counts,
            node_centers=jnp.where(
                packed_active[:, None],
                geometry.center,
                jnp.asarray(0, geometry.center.dtype),
            ),
            node_half_widths=jnp.where(
                packed_active[:, None],
                geometry.half_width,
                jnp.asarray(0, geometry.half_width.dtype),
            ),
            root_slot=jnp.where(active_count > 0, 0, -1).astype(jnp.int32),
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            evidence=evidence,
        )

    def refresh(
        self,
        previous: MortonPointHierarchyState,
        points: jax.Array,
        *,
        active_mask: jax.Array | None = None,
        stable_ids: jax.Array | None = None,
    ) -> MortonHierarchyTransition:
        candidate = self.build(
            points,
            active_mask=active_mask,
            stable_ids=stable_ids,
            epoch=previous.epoch + 1,
        )
        same_topology = (
            jnp.array_equal(candidate.node_prefixes, previous.node_prefixes)
            & jnp.array_equal(candidate.node_levels, previous.node_levels)
            & jnp.array_equal(candidate.node_active, previous.node_active)
            & jnp.array_equal(candidate.sorted_stable_ids, previous.sorted_stable_ids)
        )
        accepted_candidate = candidate.evidence.successful
        accepted = jax.lax.cond(
            accepted_candidate,
            lambda _: candidate,
            lambda _: previous,
            operand=None,
        )
        return MortonHierarchyTransition(
            candidate=candidate,
            accepted=accepted,
            accepted_candidate=accepted_candidate,
            refitted=accepted_candidate & same_topology,
            rebuilt=accepted_candidate & ~same_topology,
        )


__all__ = [
    "MortonHierarchyBuildEvidence",
    "MortonHierarchyTransition",
    "MortonPointHierarchyPlan",
    "MortonPointHierarchyState",
]
