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
from ._core import BlockMetadata


class RefinementDecision(StrictModule):
    """Fixed-capacity child activation and explicit overflow status."""

    selected_parents: Array
    child_active: Array
    overflow: Array
    plan_id: str = eqx.field(static=True)


class FixedCapacityRefinementPlan(StrictModule, NonTrainableState):
    """Preallocated parent-to-child slot map for JIT-stable refinement decisions."""

    parent_to_children: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, parent_to_children: ArrayLike, /):
        mapping = np.asarray(parent_to_children, dtype=np.int32)
        if mapping.ndim != 2 or mapping.shape[0] == 0 or mapping.shape[1] == 0:
            raise ValueError("parent_to_children must be one non-empty rank-2 map.")
        self.parent_to_children = jnp.asarray(mapping)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-refinement",
                "mapping": array_tree_fingerprint(mapping),
            }
        )

    def decide(
        self,
        parent_metadata: BlockMetadata,
        child_metadata: BlockMetadata,
        indicators: ArrayLike,
        threshold: ArrayLike,
    ) -> RefinementDecision:
        if not isinstance(parent_metadata, BlockMetadata) or not isinstance(
            child_metadata, BlockMetadata
        ):
            raise TypeError("Refinement decision requires parent and child metadata.")
        values = jnp.asarray(indicators)
        threshold_ = jnp.asarray(threshold)
        if values.shape != parent_metadata.active.shape or threshold_.shape != ():
            raise ValueError("Indicators/threshold do not match parent capacity.")
        selected = parent_metadata.active & (values > threshold_)
        mapping = self.parent_to_children
        if mapping.shape[0] != selected.shape[0]:
            raise ValueError("Refinement map parent capacity does not match metadata.")
        valid = mapping >= 0
        selected_routes = selected[:, None] & valid
        overflow = jnp.any(selected[:, None] & ~valid)
        safe_slots = jnp.where(valid, mapping, 0)
        if child_metadata.active.shape[0] == 0:
            raise ValueError("Child metadata has no capacity.")
        overflow = overflow | jnp.any(
            selected_routes & (safe_slots >= child_metadata.active.shape[0])
        )
        bounded_slots = jnp.clip(safe_slots, 0, child_metadata.active.shape[0] - 1)
        requested = (
            jnp.zeros_like(child_metadata.active).at[bounded_slots].max(selected_routes)
        )
        return RefinementDecision(
            selected_parents=selected,
            child_active=child_metadata.active | requested,
            overflow=overflow,
            plan_id=self.plan_id,
        )


__all__ = ["FixedCapacityRefinementPlan", "RefinementDecision"]
