#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._point_hierarchy import MortonPointHierarchyState


class MortonPrimitiveBoundsEvidence(NonTrainableState, StrictModule):
    active_items: Array
    bounded_leaves: Array
    bounded_nodes: Array
    invalid_items: Array
    maximum_extent: Array
    finite: Array
    successful: Array


class MortonPrimitiveBoundsState(StrictModule):
    """Per-item and bottom-up aggregate AABBs on a fixed point hierarchy."""

    hierarchy: MortonPointHierarchyState
    item_lower: Array
    item_upper: Array
    node_lower: Array
    node_upper: Array
    node_bounded: Array
    evidence: MortonPrimitiveBoundsEvidence
    epoch: Array
    bounds_id: str = eqx.field(static=True)


class MortonPrimitiveBoundsPlan(NonTrainableState, StrictModule):
    """Refit extended primitive AABBs over an accepted Morton hierarchy."""

    hierarchy: MortonPointHierarchyState
    maximum_depth: int = eqx.field(static=True)
    point_capacity: int = eqx.field(static=True)
    node_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: MortonPointHierarchyState,
        ambient_dimension: int,
        /,
    ) -> None:
        if not isinstance(hierarchy, MortonPointHierarchyState):
            raise TypeError("hierarchy must be MortonPointHierarchyState.")
        if not bool(hierarchy.evidence.successful):
            raise ValueError("hierarchy must be successful before bounds preparation.")
        dimension = int(ambient_dimension)
        if dimension not in (1, 2, 3):
            raise ValueError("Primitive bounds require dimension one, two, or three.")
        maximum_depth = int(np.max(np.asarray(hierarchy.node_levels)))
        self.hierarchy = hierarchy
        self.maximum_depth = maximum_depth
        self.point_capacity = int(hierarchy.sorted_codes.shape[0])
        self.node_capacity = int(hierarchy.node_active.shape[0])
        self.ambient_dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "morton-primitive-bounds-plan",
                "stable_ids": array_tree_fingerprint(
                    np.asarray(hierarchy.sorted_stable_ids)
                ),
                "node_prefixes": array_tree_fingerprint(
                    np.asarray(hierarchy.node_prefixes)
                ),
                "node_levels": array_tree_fingerprint(np.asarray(hierarchy.node_levels)),
                "ambient_dimension": dimension,
            }
        )

    def refit(
        self,
        item_lower: ArrayLike,
        item_upper: ArrayLike,
        /,
        *,
        epoch: int | Array = 0,
    ) -> MortonPrimitiveBoundsState:
        lower = jnp.asarray(item_lower)
        upper = jnp.asarray(item_upper, dtype=lower.dtype)
        expected = (self.point_capacity, self.ambient_dimension)
        if lower.shape != expected or upper.shape != expected:
            raise ValueError(f"Primitive bounds must both have shape {expected}.")
        hierarchy = self.hierarchy
        item_active = hierarchy.logical_point_leaf_slots >= 0
        finite_item = jnp.all(jnp.isfinite(lower), axis=-1) & jnp.all(
            jnp.isfinite(upper), axis=-1
        )
        ordered_item = jnp.all(lower <= upper, axis=-1)
        valid_item = finite_item & ordered_item
        safe_lower = jnp.where((item_active & valid_item)[:, None], lower, 0.0)
        safe_upper = jnp.where((item_active & valid_item)[:, None], upper, 0.0)
        sorted_lower = safe_lower[hierarchy.storage_to_logical]
        sorted_upper = safe_upper[hierarchy.storage_to_logical]
        sorted_valid = (
            item_active[hierarchy.storage_to_logical]
            & valid_item[hierarchy.storage_to_logical]
        )
        leaf_slot = hierarchy.sorted_point_leaf_slots
        safe_leaf = jnp.maximum(leaf_slot, 0)
        lower_contribution = jnp.where(sorted_valid[:, None], sorted_lower, jnp.inf)
        upper_contribution = jnp.where(sorted_valid[:, None], sorted_upper, -jnp.inf)
        node_lower = (
            jnp.full(
                (self.node_capacity, self.ambient_dimension),
                jnp.inf,
                dtype=lower.dtype,
            )
            .at[safe_leaf]
            .min(lower_contribution)
        )
        node_upper = (
            jnp.full(
                (self.node_capacity, self.ambient_dimension),
                -jnp.inf,
                dtype=upper.dtype,
            )
            .at[safe_leaf]
            .max(upper_contribution)
        )
        node_bounded = hierarchy.node_is_leaf & jnp.all(
            jnp.isfinite(node_lower) & jnp.isfinite(node_upper), axis=-1
        )
        for level in range(self.maximum_depth - 1, -1, -1):
            internal = (
                hierarchy.node_active
                & ~hierarchy.node_is_leaf
                & (hierarchy.node_levels == level)
            )
            children = hierarchy.node_children
            child_valid = children >= 0
            safe_children = jnp.maximum(children, 0)
            child_bounded = child_valid & node_bounded[safe_children]
            child_lower = jnp.where(
                child_bounded[..., None], node_lower[safe_children], jnp.inf
            )
            child_upper = jnp.where(
                child_bounded[..., None], node_upper[safe_children], -jnp.inf
            )
            reduced_lower = jnp.min(child_lower, axis=1)
            reduced_upper = jnp.max(child_upper, axis=1)
            reduced_bounded = jnp.any(child_bounded, axis=1)
            node_lower = jnp.where(internal[:, None], reduced_lower, node_lower)
            node_upper = jnp.where(internal[:, None], reduced_upper, node_upper)
            node_bounded = jnp.where(internal, reduced_bounded, node_bounded)
        node_lower = jnp.where(node_bounded[:, None], node_lower, 0.0)
        node_upper = jnp.where(node_bounded[:, None], node_upper, 0.0)
        invalid_items = jnp.sum(item_active & ~valid_item, dtype=jnp.int32)
        finite = jnp.all(
            jnp.where(node_bounded[:, None], jnp.isfinite(node_lower), True)
        ) & jnp.all(jnp.where(node_bounded[:, None], jnp.isfinite(node_upper), True))
        successful = hierarchy.evidence.successful & (invalid_items == 0) & finite
        extent = jnp.where(node_bounded[:, None], node_upper - node_lower, 0.0)
        evidence = MortonPrimitiveBoundsEvidence(
            active_items=jnp.sum(item_active, dtype=jnp.int32),
            bounded_leaves=jnp.sum(
                hierarchy.node_is_leaf & node_bounded, dtype=jnp.int32
            ),
            bounded_nodes=jnp.sum(node_bounded, dtype=jnp.int32),
            invalid_items=invalid_items,
            maximum_extent=jnp.max(extent, axis=0),
            finite=finite,
            successful=successful,
        )
        return MortonPrimitiveBoundsState(
            hierarchy=hierarchy,
            item_lower=safe_lower,
            item_upper=safe_upper,
            node_lower=node_lower,
            node_upper=node_upper,
            node_bounded=node_bounded,
            evidence=evidence,
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            bounds_id=canonical_fingerprint(
                {
                    "kind": "morton-primitive-bounds-state",
                    "plan": self.plan_id,
                }
            ),
        )


__all__ = [
    "MortonPrimitiveBoundsEvidence",
    "MortonPrimitiveBoundsPlan",
    "MortonPrimitiveBoundsState",
]
