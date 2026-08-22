#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class BlockLevelPlan(StrictModule, NonTrainableState):
    """Static block shape, capacity, halo, and refinement ratio for one AMR level."""

    level: int = eqx.field(static=True)
    block_shape: tuple[int, ...] = eqx.field(static=True)
    halo_width: tuple[int, ...] = eqx.field(static=True)
    maximum_blocks: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        block_shape: Sequence[int],
        maximum_blocks: int,
        /,
        *,
        halo_width: int | Sequence[int] = 1,
        refinement_ratio: int = 2,
        spacing: Sequence[float],
    ):
        level_ = int(level)
        shape = tuple(int(size) for size in block_shape)
        capacity = int(maximum_blocks)
        ratio = int(refinement_ratio)
        halo = (
            (int(halo_width),) * len(shape)
            if isinstance(halo_width, int)
            else tuple(int(value) for value in halo_width)
        )
        spacing_ = tuple(float(value) for value in spacing)
        if (
            level_ < 0
            or not shape
            or any(size <= 0 for size in shape)
            or len(halo) != len(shape)
            or any(
                value < 0 or 2 * value >= size
                for value, size in zip(halo, shape, strict=True)
            )
            or capacity <= 0
            or ratio <= 1
            or len(spacing_) != len(shape)
            or any(not np.isfinite(value) or value <= 0.0 for value in spacing_)
        ):
            raise ValueError(
                "Invalid AMR level shape, halo, capacity, ratio, or spacing."
            )
        self.level = level_
        self.block_shape = shape
        self.halo_width = halo
        self.maximum_blocks = capacity
        self.refinement_ratio = ratio
        self.spacing = spacing_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "amr-level-plan",
                "level": level_,
                "block_shape": list(shape),
                "halo_width": list(halo),
                "maximum_blocks": capacity,
                "refinement_ratio": ratio,
                "spacing": list(spacing_),
            }
        )


class BlockMetadata(StrictModule, NonTrainableState):
    """Fixed-capacity active, hierarchy, logical-index, and neighbor metadata."""

    active: Array
    block_ids: Array
    parent_ids: Array
    logical_indices: Array
    neighbor_slots: Array
    metadata_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BlockLevelPlan,
        /,
        *,
        active: ArrayLike,
        block_ids: ArrayLike,
        parent_ids: ArrayLike,
        logical_indices: ArrayLike,
        neighbor_slots: ArrayLike,
    ):
        if not isinstance(plan, BlockLevelPlan):
            raise TypeError("plan must be a BlockLevelPlan.")
        mask = np.asarray(active, dtype=bool)
        ids = np.asarray(block_ids, dtype=np.int64)
        parents = np.asarray(parent_ids, dtype=np.int64)
        logical = np.asarray(logical_indices, dtype=np.int32)
        neighbors = np.asarray(neighbor_slots, dtype=np.int32)
        capacity = plan.maximum_blocks
        dimension = len(plan.block_shape)
        if (
            mask.shape != (capacity,)
            or ids.shape != (capacity,)
            or parents.shape != (capacity,)
            or logical.shape != (capacity, dimension)
            or neighbors.shape != (capacity, dimension, 2)
        ):
            raise ValueError("AMR metadata arrays do not match level capacity/dimension.")
        active_ids = ids[mask]
        if np.any(active_ids < 0) or np.unique(active_ids).size != active_ids.size:
            raise ValueError("Active AMR block IDs must be unique and non-negative.")
        if np.any(ids[~mask] != -1):
            raise ValueError("Inactive AMR block IDs must use -1 metadata sentinel.")
        valid_neighbors = neighbors >= 0
        if np.any(neighbors[valid_neighbors] >= capacity):
            raise ValueError("AMR neighbor slots are out of capacity bounds.")
        if np.any(valid_neighbors & ~mask[neighbors.clip(min=0)]):
            raise ValueError("Active AMR neighbor routes must target active blocks.")
        self.active = jnp.asarray(mask)
        self.block_ids = jnp.asarray(ids)
        self.parent_ids = jnp.asarray(parents)
        self.logical_indices = jnp.asarray(logical)
        self.neighbor_slots = jnp.asarray(neighbors)
        self.metadata_id = canonical_fingerprint(
            {
                "kind": "amr-block-metadata",
                "plan": plan.plan_id,
                "active": array_tree_fingerprint(mask),
                "block_ids": array_tree_fingerprint(ids),
                "parent_ids": array_tree_fingerprint(parents),
                "logical_indices": array_tree_fingerprint(logical),
                "neighbors": array_tree_fingerprint(neighbors),
            }
        )


class BlockLevelState(StrictModule):
    """Fixed-capacity block values whose inactive payload is always masked before use."""

    plan: BlockLevelPlan
    metadata: BlockMetadata
    values: Array

    def __init__(
        self,
        plan: BlockLevelPlan,
        metadata: BlockMetadata,
        values: ArrayLike,
        /,
    ):
        if not isinstance(plan, BlockLevelPlan) or not isinstance(
            metadata, BlockMetadata
        ):
            raise TypeError("Invalid AMR level state plan/metadata.")
        array = jnp.asarray(values)
        expected_prefix = (plan.maximum_blocks,) + plan.block_shape
        if array.shape[: len(expected_prefix)] != expected_prefix:
            raise ValueError("AMR block values do not match capacity and block shape.")
        self.plan = plan
        self.metadata = metadata
        self.values = array

    def safe_values(self, /) -> Array:
        mask = self.metadata.active.reshape(
            (self.plan.maximum_blocks,) + (1,) * (self.values.ndim - 1)
        )
        return jnp.where(mask, self.values, jnp.zeros((), dtype=self.values.dtype))

    def fill_same_level_halo_1d(self, /) -> Array:
        if len(self.plan.block_shape) != 1:
            raise ValueError(
                "Initial same-level halo fill supports one-dimensional blocks."
            )
        width = self.plan.halo_width[0]
        values = self.safe_values()
        padded = jnp.pad(values, ((0, 0), (width, width)) + ((0, 0),) * (values.ndim - 2))
        if width == 0:
            return padded
        left_slots = self.metadata.neighbor_slots[:, 0, 0]
        right_slots = self.metadata.neighbor_slots[:, 0, 1]
        safe_left = jnp.where(left_slots >= 0, left_slots, 0)
        safe_right = jnp.where(right_slots >= 0, right_slots, 0)
        left_data = values[safe_left, -width:]
        right_data = values[safe_right, :width]
        left_mask = (left_slots >= 0).reshape((-1,) + (1,) * (left_data.ndim - 1))
        right_mask = (right_slots >= 0).reshape((-1,) + (1,) * (right_data.ndim - 1))
        padded = padded.at[:, :width].set(jnp.where(left_mask, left_data, 0.0))
        padded = padded.at[:, -width:].set(jnp.where(right_mask, right_data, 0.0))
        return padded


class BlockHierarchyPlan(StrictModule, NonTrainableState):
    """Ordered fixed-capacity AMR level plans."""

    levels: tuple[BlockLevelPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, levels: Sequence[BlockLevelPlan], /):
        values = tuple(levels)
        if not values or not all(isinstance(level, BlockLevelPlan) for level in values):
            raise TypeError("levels must contain BlockLevelPlan values.")
        if tuple(level.level for level in values) != tuple(range(len(values))):
            raise ValueError("AMR level numbers must be contiguous from zero.")
        for coarse, fine in zip(values[:-1], values[1:], strict=True):
            expected = tuple(value / coarse.refinement_ratio for value in coarse.spacing)
            if not np.allclose(fine.spacing, expected):
                raise ValueError("AMR fine spacing must follow refinement ratio.")
        self.levels = values
        self.plan_id = canonical_fingerprint(
            {"kind": "amr-hierarchy-plan", "levels": [level.plan_id for level in values]}
        )


class BlockHierarchyState(StrictModule):
    """Tuple of static levels plus the realized active-block trace."""

    plan: BlockHierarchyPlan
    levels: tuple[BlockLevelState, ...]
    refinement_trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BlockHierarchyPlan,
        levels: Sequence[BlockLevelState],
        /,
    ):
        values = tuple(levels)
        if not isinstance(plan, BlockHierarchyPlan) or len(values) != len(plan.levels):
            raise TypeError("AMR hierarchy state must match its hierarchy plan.")
        if any(
            level.plan.plan_id != expected.plan_id
            for level, expected in zip(values, plan.levels, strict=True)
        ):
            raise ValueError("AMR hierarchy level state/plan mismatch.")
        self.plan = plan
        self.levels = values
        self.refinement_trace_id = canonical_fingerprint(
            {
                "kind": "amr-refinement-trace",
                "plan": plan.plan_id,
                "metadata": [level.metadata.metadata_id for level in values],
            }
        )


__all__ = [
    "BlockHierarchyPlan",
    "BlockHierarchyState",
    "BlockLevelPlan",
    "BlockLevelState",
    "BlockMetadata",
]
