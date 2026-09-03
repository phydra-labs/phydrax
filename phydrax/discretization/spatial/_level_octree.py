#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._morton import morton_decode_integer, morton_encode_integer, MortonAddressPlan
from ._point_hierarchy import MortonPointHierarchyPlan, MortonPointHierarchyState


_UINT64_MAX = np.iinfo(np.uint64).max


class SparseLevelOctreeEvidence(NonTrainableState, StrictModule):
    """Capacity and completion evidence for levelwise tree relations."""

    required_far_interactions: jax.Array
    far_interaction_capacity: jax.Array
    required_near_interactions: jax.Array
    near_interaction_capacity: jax.Array
    active_nodes: jax.Array
    active_leaves: jax.Array
    successful: jax.Array


class SparseLevelOctree(NonTrainableState, StrictModule):
    """All occupied Morton prefixes and bounded FMM interaction relations."""

    hierarchy: MortonPointHierarchyState
    far_targets: jax.Array
    far_sources: jax.Array
    far_active: jax.Array
    near_targets: jax.Array
    near_sources: jax.Array
    near_active: jax.Array
    evidence: SparseLevelOctreeEvidence
    tree_id: str = eqx.field(static=True)


class SparseLevelOctreePlan(StrictModule):
    """Prepare an occupied level octree without complete-grid allocation."""

    address_plan: MortonAddressPlan
    point_capacity: int = eqx.field(static=True)
    far_interaction_capacity: int = eqx.field(static=True)
    near_interaction_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        point_capacity: int,
        *,
        far_interaction_capacity: int,
        near_interaction_capacity: int,
    ) -> None:
        points = int(point_capacity)
        far = int(far_interaction_capacity)
        near = int(near_interaction_capacity)
        if points < 1 or far < 1 or near < 1:
            raise ValueError("Point and interaction capacities must be positive.")
        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "point_capacity", points)
        object.__setattr__(self, "far_interaction_capacity", far)
        object.__setattr__(self, "near_interaction_capacity", near)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "sparse-level-octree-plan",
                    "address_plan_id": address_plan.plan_id,
                    "point_capacity": points,
                    "far_interaction_capacity": far,
                    "near_interaction_capacity": near,
                }
            ),
        )

    def prepare(
        self,
        points: jax.Array,
        *,
        active_mask: jax.Array | None = None,
        stable_ids: jax.Array | None = None,
    ) -> SparseLevelOctree:
        hierarchy = MortonPointHierarchyPlan(
            self.address_plan,
            self.point_capacity,
            node_capacity=(self.address_plan.maximum_depth + 1) * self.point_capacity,
            target_leaf_occupancy=0,
        ).build(
            points,
            active_mask=active_mask,
            stable_ids=stable_ids,
        )
        node_capacity = hierarchy.node_active.size
        node_slots = jnp.arange(node_capacity, dtype=jnp.int32)
        level_prefixes = []
        level_slots = []
        for level in range(self.address_plan.maximum_depth + 1):
            at_level = hierarchy.node_active & (hierarchy.node_levels == level)
            sortable = jnp.where(
                at_level,
                hierarchy.node_prefixes,
                jnp.asarray(_UINT64_MAX, dtype=jnp.uint64),
            )
            order = jnp.argsort(sortable, stable=True)
            level_prefixes.append(sortable[order])
            level_slots.append(jnp.where(at_level[order], node_slots[order], -1))
        prefix_table = jnp.stack(level_prefixes)
        slot_table = jnp.stack(level_slots)
        periodic = jnp.asarray(self.address_plan.periodic_axes, dtype=bool)
        parent_offsets = jnp.asarray(
            tuple(product((-1, 0, 1), repeat=self.address_plan.dimension)),
            dtype=jnp.int64,
        )
        child_offsets = jnp.asarray(
            tuple(product((0, 1), repeat=self.address_plan.dimension)),
            dtype=jnp.int64,
        )
        near_offsets = parent_offsets

        def lookup(level, candidate_coordinates):
            candidate_prefixes = morton_encode_integer(
                candidate_coordinates, self.address_plan.maximum_depth
            )
            row_prefixes = prefix_table[level]
            row_slots = slot_table[level]
            positions = jnp.searchsorted(
                row_prefixes, candidate_prefixes, side="left"
            ).astype(jnp.int32)
            safe_positions = jnp.minimum(positions, node_capacity - 1)
            matches = (
                (positions < node_capacity)
                & (row_prefixes[safe_positions] == candidate_prefixes)
                & (row_slots[safe_positions] >= 0)
            )
            return jnp.where(matches, row_slots[safe_positions], -1), matches

        def far_candidates(node):
            level = hierarchy.node_levels[node]
            prefix = hierarchy.node_prefixes[node]
            target_coordinate = morton_decode_integer(
                prefix, self.address_plan.dimension, self.address_plan.maximum_depth
            )
            parent_coordinate = target_coordinate >> 1
            parent_level = jnp.maximum(level - 1, 0)
            parent_resolution = jnp.left_shift(
                jnp.asarray(1, dtype=jnp.int64), parent_level
            )
            neighbor_parents = parent_coordinate[None, :] + parent_offsets
            parent_in_bounds = jnp.all(
                periodic
                | ((neighbor_parents >= 0) & (neighbor_parents < parent_resolution)),
                axis=-1,
            )
            neighbor_parents = jnp.where(
                periodic,
                jnp.mod(neighbor_parents, parent_resolution),
                neighbor_parents,
            )
            candidate_coordinates = (
                neighbor_parents[:, None, :] * 2 + child_offsets[None, :, :]
            ).reshape((-1, self.address_plan.dimension))
            parent_valid = jnp.repeat(parent_in_bounds, child_offsets.shape[0])
            source_slots, source_found = lookup(level, candidate_coordinates)
            level_resolution = jnp.left_shift(jnp.asarray(1, dtype=jnp.int64), level)
            coordinate_delta = jnp.abs(candidate_coordinates - target_coordinate[None, :])
            coordinate_delta = jnp.where(
                periodic,
                jnp.minimum(coordinate_delta, level_resolution - coordinate_delta),
                coordinate_delta,
            )
            separated = jnp.any(coordinate_delta > 1, axis=-1)
            valid = (
                hierarchy.node_active[node]
                & (level >= 2)
                & parent_valid
                & source_found
                & separated
            )
            return source_slots, valid

        far_sources_matrix, far_valid_matrix = jax.vmap(far_candidates)(node_slots)
        far_width = far_sources_matrix.shape[1]
        flat_far_valid = far_valid_matrix.reshape((-1,))
        required_far = jnp.sum(flat_far_valid, dtype=jnp.int32)
        selected_far = jnp.nonzero(
            flat_far_valid,
            size=self.far_interaction_capacity,
            fill_value=flat_far_valid.size,
        )[0].astype(jnp.int32)
        far_active = jnp.arange(
            self.far_interaction_capacity, dtype=jnp.int32
        ) < jnp.minimum(required_far, self.far_interaction_capacity)
        safe_far = jnp.minimum(selected_far, flat_far_valid.size - 1)
        far_targets = jnp.where(far_active, safe_far // far_width, -1)
        far_sources = jnp.where(
            far_active, far_sources_matrix.reshape((-1,))[safe_far], -1
        )

        def near_candidates(node):
            level = hierarchy.node_levels[node]
            prefix = hierarchy.node_prefixes[node]
            target_coordinate = morton_decode_integer(
                prefix, self.address_plan.dimension, self.address_plan.maximum_depth
            )
            resolution = jnp.left_shift(jnp.asarray(1, dtype=jnp.int64), level)
            candidate_coordinates = target_coordinate[None, :] + near_offsets
            in_bounds = jnp.all(
                periodic
                | ((candidate_coordinates >= 0) & (candidate_coordinates < resolution)),
                axis=-1,
            )
            candidate_coordinates = jnp.where(
                periodic,
                jnp.mod(candidate_coordinates, resolution),
                candidate_coordinates,
            )
            source_slots, source_found = lookup(level, candidate_coordinates)
            valid = (
                hierarchy.node_is_leaf[node]
                & in_bounds
                & source_found
                & hierarchy.node_is_leaf[jnp.maximum(source_slots, 0)]
            )
            return source_slots, valid

        near_sources_matrix, near_valid_matrix = jax.vmap(near_candidates)(node_slots)
        near_width = near_sources_matrix.shape[1]
        flat_near_valid = near_valid_matrix.reshape((-1,))
        required_near = jnp.sum(flat_near_valid, dtype=jnp.int32)
        selected_near = jnp.nonzero(
            flat_near_valid,
            size=self.near_interaction_capacity,
            fill_value=flat_near_valid.size,
        )[0].astype(jnp.int32)
        near_active = jnp.arange(
            self.near_interaction_capacity, dtype=jnp.int32
        ) < jnp.minimum(required_near, self.near_interaction_capacity)
        safe_near = jnp.minimum(selected_near, flat_near_valid.size - 1)
        near_targets = jnp.where(near_active, safe_near // near_width, -1)
        near_sources = jnp.where(
            near_active, near_sources_matrix.reshape((-1,))[safe_near], -1
        )
        successful = (
            hierarchy.evidence.successful
            & (required_far <= self.far_interaction_capacity)
            & (required_near <= self.near_interaction_capacity)
        )
        evidence = SparseLevelOctreeEvidence(
            required_far_interactions=required_far,
            far_interaction_capacity=jnp.asarray(
                self.far_interaction_capacity, dtype=jnp.int32
            ),
            required_near_interactions=required_near,
            near_interaction_capacity=jnp.asarray(
                self.near_interaction_capacity, dtype=jnp.int32
            ),
            active_nodes=hierarchy.evidence.active_nodes,
            active_leaves=hierarchy.evidence.active_leaves,
            successful=successful,
        )
        return SparseLevelOctree(
            hierarchy=hierarchy,
            far_targets=far_targets,
            far_sources=far_sources,
            far_active=far_active,
            near_targets=near_targets,
            near_sources=near_sources,
            near_active=near_active,
            evidence=evidence,
            tree_id=canonical_fingerprint(
                {
                    "kind": "sparse-level-octree",
                    "plan": self.plan_id,
                }
            ),
        )


__all__ = [
    "SparseLevelOctree",
    "SparseLevelOctreeEvidence",
    "SparseLevelOctreePlan",
]
