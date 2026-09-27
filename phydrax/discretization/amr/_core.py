#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import KeyGroupLookup, KeyGroupPlan, KeyGroupState
from .._tensor_support import PreparedTensorGrid
from .._topology_epoch import TopologyEpoch
from ._patches import LogicalPatchBox


class BlockLevelPlan(StrictModule, NonTrainableState):
    """Static fixed-block storage and refinement policy for one AMR level."""

    level: int = eqx.field(static=True)
    block_shape: tuple[int, ...] = eqx.field(static=True)
    halo_width: tuple[int, ...] = eqx.field(static=True)
    maximum_blocks: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
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
    ) -> None:
        level_ = int(level)
        shape = tuple(block_shape)
        capacity = int(maximum_blocks)
        ratio = int(refinement_ratio)
        halo = (
            (int(halo_width),) * len(shape)
            if isinstance(halo_width, int)
            else tuple(halo_width)
        )
        if (
            level_ < 0
            or not shape
            or any(size <= 0 for size in shape)
            or len(halo) != len(shape)
            or any(value < 0 for value in halo)
            or capacity <= 0
            or ratio <= 1
        ):
            raise ValueError("Invalid AMR level shape, halo, capacity, or ratio.")
        self.level = level_
        self.block_shape = shape
        self.halo_width = halo
        self.maximum_blocks = capacity
        self.refinement_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {
                "kind": "amr-level-plan",
                "level": level_,
                "block_shape": list(shape),
                "halo_width": list(halo),
                "maximum_blocks": capacity,
                "refinement_ratio": ratio,
            }
        )


class BlockHierarchyPlan(StrictModule, NonTrainableState):
    """Fixed-block hierarchy bound to one uniform cell-centered tensor geometry.

    The prepared tensor grid is the sole geometry source.  Every finer cell lattice,
    physical spacing, block lattice, and parent/child alignment is derived exactly
    from it and the adjacent level plans.
    """

    grid: PreparedTensorGrid
    levels: tuple[BlockLevelPlan, ...]
    global_cell_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    block_lattice_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    level_spacings: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    children_per_parent: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    block_id_offsets: tuple[int, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        levels: Sequence[BlockLevelPlan],
        /,
    ) -> None:
        values = tuple(levels)
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        if not values or not all(isinstance(level, BlockLevelPlan) for level in values):
            raise TypeError("levels must contain BlockLevelPlan values.")
        if tuple(level.level for level in values) != tuple(range(len(values))):
            raise ValueError("AMR level numbers must be contiguous from zero.")
        dimension = len(grid.shape)
        if any(len(level.block_shape) != dimension for level in values):
            raise ValueError("Every AMR block shape must match the tensor-grid rank.")
        if any(axis.primary_entity != "interval" for axis in grid.axes):
            raise ValueError("Block AMR requires an interval-primary tensor grid.")
        widths = tuple(
            np.asarray(axis.interval_widths, dtype=np.float64)
            for axis in grid.structured_axes
        )
        if any(
            axis.basis != "uniform" or width.size == 0 or not np.all(width == width[0])
            for axis, width in zip(grid.axes, widths, strict=True)
        ):
            raise ValueError("Block AMR requires uniform tensor-grid axes.")

        global_shapes: list[tuple[int, ...]] = [tuple(grid.shape)]
        spacings: list[tuple[float, ...]] = [tuple(float(width[0]) for width in widths)]
        for coarse in values[:-1]:
            global_shapes.append(
                tuple(size * coarse.refinement_ratio for size in global_shapes[-1])
            )
            spacings.append(
                tuple(value / coarse.refinement_ratio for value in spacings[-1])
            )
        lattices: list[tuple[int, ...]] = []
        for level, global_shape in zip(values, global_shapes, strict=True):
            if any(
                cells % block != 0
                for cells, block in zip(global_shape, level.block_shape, strict=True)
            ):
                raise ValueError(
                    "Each level block shape must divide its derived global cell lattice exactly."
                )
            lattices.append(
                tuple(
                    cells // block
                    for cells, block in zip(global_shape, level.block_shape, strict=True)
                )
            )
        children: list[tuple[int, ...]] = []
        for coarse, fine in zip(values[:-1], values[1:], strict=True):
            refined_parent = tuple(
                size * coarse.refinement_ratio for size in coarse.block_shape
            )
            if any(
                parent % child != 0
                or child > parent
                or child % coarse.refinement_ratio != 0
                for parent, child in zip(refined_parent, fine.block_shape, strict=True)
            ):
                raise ValueError(
                    "Adjacent fixed blocks must have exact parent/child face alignment."
                )
            children.append(
                tuple(
                    parent // child
                    for parent, child in zip(
                        refined_parent, fine.block_shape, strict=True
                    )
                )
            )
        if values[0].maximum_blocks < prod(lattices[0]):
            raise ValueError("Level-zero capacity must hold complete base-grid coverage.")
        offsets: list[int] = []
        next_offset = 0
        for lattice in lattices:
            offsets.append(next_offset)
            next_offset += prod(lattice)
        if next_offset > np.iinfo(np.int32).max:
            raise ValueError("Canonical AMR block IDs exceed int32 range.")

        self.grid = grid
        self.levels = values
        self.global_cell_shapes = tuple(global_shapes)
        self.block_lattice_shapes = tuple(lattices)
        self.level_spacings = tuple(spacings)
        self.children_per_parent = tuple(children)
        self.block_id_offsets = tuple(offsets)
        self.periodic_axes = tuple(bool(axis.periodic) for axis in grid.axes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "amr-hierarchy-plan",
                "grid": grid.prepared_id,
                "levels": [level.plan_id for level in values],
                "global_cells": global_shapes,
                "block_lattices": lattices,
                "children_per_parent": children,
            }
        )

    @property
    def geometry_id(self) -> str:
        return self.grid.support.embedding_id

    def block_id(self, level: int, logical_index: Sequence[int], /) -> int:
        level_ = int(level)
        if level_ < 0 or level_ >= len(self.levels):
            raise ValueError("AMR level is out of range.")
        logical = tuple(logical_index)
        lattice = self.block_lattice_shapes[level_]
        if len(logical) != len(lattice) or any(
            value < 0 or value >= extent
            for value, extent in zip(logical, lattice, strict=True)
        ):
            raise ValueError("Logical block index is outside its level lattice.")
        return self.block_id_offsets[level_] + int(np.ravel_multi_index(logical, lattice))


class BlockMetadata(StrictModule, NonTrainableState):
    """Fixed-capacity active, hierarchy, logical-index, and neighbor metadata."""

    active: Array
    block_ids: Array
    parent_ids: Array
    logical_indices: Array
    neighbor_slots: Array
    block_groups: KeyGroupState
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
    ) -> None:
        if not isinstance(plan, BlockLevelPlan):
            raise TypeError("plan must be a BlockLevelPlan.")
        mask = np.asarray(active, dtype=np.bool_)
        ids = np.asarray(block_ids, dtype=np.int32)
        parents = np.asarray(parent_ids, dtype=np.int32)
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
        if np.any(parents[~mask] != -1) or np.any(logical[~mask] != -1):
            raise ValueError("Inactive AMR hierarchy metadata must use -1 sentinels.")
        valid_neighbors = neighbors >= 0
        if np.any(neighbors[valid_neighbors] >= capacity):
            raise ValueError("AMR neighbor slots are out of capacity bounds.")
        if np.any(valid_neighbors & ~mask[neighbors.clip(min=0)]):
            raise ValueError("AMR neighbor routes must target active blocks.")
        if np.any(neighbors[~mask] != -1):
            raise ValueError("Inactive AMR neighbor metadata must use -1 sentinels.")
        key_upper_bound = int(np.max(active_ids, initial=0))
        block_groups = KeyGroupPlan(
            capacity,
            capacity,
            key_upper_bound,
            maximum_group_size=1,
        ).build(
            jnp.asarray(ids),
            jnp.asarray(mask),
            stable_ids=jnp.arange(capacity, dtype=jnp.int32),
        )
        self.active = jnp.asarray(mask)
        self.block_ids = jnp.asarray(ids)
        self.parent_ids = jnp.asarray(parents)
        self.logical_indices = jnp.asarray(logical)
        self.neighbor_slots = jnp.asarray(neighbors)
        self.block_groups = block_groups
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

    def lookup_block_ids(self, block_ids: ArrayLike, /) -> KeyGroupLookup:
        """Resolve stable int32 block IDs to canonical current capacity slots."""
        lookup = self.block_groups.lookup(jnp.asarray(block_ids, dtype=jnp.int32))
        sorted_slots = lookup.group_slots
        storage_slots = self.block_groups.storage_to_logical[
            self.block_groups.group_starts[sorted_slots]
        ]
        return KeyGroupLookup(
            group_slots=jnp.where(lookup.supported, storage_slots, 0),
            supported=lookup.supported,
        )


class BlockLevelState(StrictModule):
    """Numeric fixed-capacity block payload bound to realized level metadata."""

    plan: BlockLevelPlan
    metadata: BlockMetadata
    values: Array

    def __init__(
        self,
        plan: BlockLevelPlan,
        metadata: BlockMetadata,
        values: ArrayLike,
        /,
    ) -> None:
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


def _lattice_linear(logical: np.ndarray, lattice: Sequence[int], /) -> np.ndarray:
    """Row-major int64 lattice keys of in-range ``(n, d)`` logical rows."""
    rows = np.asarray(logical, dtype=np.int64)
    if rows.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.ravel_multi_index(tuple(rows.T), tuple(lattice)).astype(np.int64)


def _sorted_membership(sorted_keys: np.ndarray, keys: np.ndarray, /) -> np.ndarray:
    """Slots of ``keys`` in a strictly increasing key array, ``-1`` when absent."""
    query = np.asarray(keys, dtype=np.int64)
    if sorted_keys.size == 0:
        return np.full(query.shape, -1, dtype=np.int64)
    position = np.searchsorted(sorted_keys, query).clip(max=sorted_keys.size - 1)
    return np.where(sorted_keys[position] == query, position, -1)


def _canonical_block_routes(
    logical: np.ndarray,
    lattice: Sequence[int],
    periodic: Sequence[bool],
    capacity: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Face-neighbor slots and coarse/fine interface flags of canonical blocks.

    ``logical`` is the active prefix in strictly increasing lattice order, so each
    face lookup is a binary search over that prefix and no level lattice is ever
    materialized.  Returns ``(capacity, d, 2)`` int32 slots (``-1`` when absent)
    and Boolean flags marking in-domain faces whose same-level neighbor is absent.
    """
    rows = np.asarray(logical, dtype=np.int64)
    count, dimension = rows.shape
    extents = np.asarray(lattice, dtype=np.int64)
    linear = _lattice_linear(rows, lattice)
    neighbors = np.full((capacity, dimension, 2), -1, dtype=np.int32)
    interfaces = np.zeros((capacity, dimension, 2), dtype=np.bool_)
    for axis in range(dimension):
        for side, delta in enumerate((-1, 1)):
            shifted = rows.copy()
            shifted[:, axis] += delta
            if periodic[axis]:
                shifted[:, axis] %= extents[axis]
            inside = (shifted[:, axis] >= 0) & (shifted[:, axis] < extents[axis])
            slots = _sorted_membership(
                linear, _lattice_linear(np.where(inside[:, None], shifted, 0), lattice)
            )
            found = inside & (slots >= 0)
            neighbors[:count, axis, side] = np.where(found, slots, -1)
            interfaces[:count, axis, side] = inside & ~found
    return neighbors, interfaces


def canonical_block_metadata(
    plan: BlockHierarchyPlan,
    level: int,
    logical_indices: Sequence[Sequence[int]],
    /,
) -> BlockMetadata:
    """Canonical sorted fixed-capacity metadata for one level's active blocks."""
    level_plan = plan.levels[level]
    capacity = level_plan.maximum_blocks
    dimension = len(level_plan.block_shape)
    lattice = plan.block_lattice_shapes[level]
    rows = np.asarray(tuple(logical_indices), dtype=np.int64).reshape(-1, dimension)
    if rows.shape[0] > capacity:
        raise ValueError("Internal topology metadata construction exceeded capacity.")
    linear = _lattice_linear(rows, lattice)
    order = np.argsort(linear, kind="stable")
    rows = rows[order]
    count = rows.shape[0]
    active = np.zeros((capacity,), dtype=np.bool_)
    active[:count] = True
    block_ids = np.full((capacity,), -1, dtype=np.int32)
    block_ids[:count] = plan.block_id_offsets[level] + linear[order]
    parent_ids = np.full((capacity,), -1, dtype=np.int32)
    logical_array = np.full((capacity, dimension), -1, dtype=np.int32)
    logical_array[:count] = rows
    if level > 0 and count:
        children = np.asarray(plan.children_per_parent[level - 1], dtype=np.int64)
        parent_ids[:count] = plan.block_id_offsets[level - 1] + _lattice_linear(
            rows // children, plan.block_lattice_shapes[level - 1]
        )
    neighbors, _ = _canonical_block_routes(rows, lattice, plan.periodic_axes, capacity)
    return BlockMetadata(
        level_plan,
        active=active,
        block_ids=block_ids,
        parent_ids=parent_ids,
        logical_indices=logical_array,
        neighbor_slots=neighbors,
    )


class BlockHierarchyTopology(StrictModule, NonTrainableState):
    """One immutable sparse realized block topology and canonical epoch.

    ``logical_boxes`` and per-patch ``covered_cells`` are the sole coverage
    representation.  The class deliberately never materializes a full finest-level
    Boolean lattice: host preparation and execution storage scale with active patch
    capacity, not the enclosing reference domain.
    """

    plan: BlockHierarchyPlan
    levels: tuple[BlockMetadata, ...]
    logical_boxes: tuple[tuple[LogicalPatchBox, ...], ...] = eqx.field(static=True)
    covered_cells: tuple[Array, ...]
    interfaces: tuple[Array, ...]
    epoch: TopologyEpoch
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BlockHierarchyPlan,
        levels: Sequence[BlockMetadata],
        /,
        *,
        epoch: TopologyEpoch | None = None,
    ) -> None:
        metadata = tuple(levels)
        if not isinstance(plan, BlockHierarchyPlan) or len(metadata) != len(plan.levels):
            raise TypeError("Block hierarchy topology must match one hierarchy plan.")
        dimension = len(plan.grid.shape)
        partition_levels: list[list[int]] = []
        linear_by_level: list[np.ndarray] = []
        logical_by_level: list[np.ndarray] = []
        boxes_by_level: list[tuple[LogicalPatchBox, ...]] = []
        interfaces: list[Array] = []
        for level_index, (level_plan, level_metadata, lattice) in enumerate(
            zip(plan.levels, metadata, plan.block_lattice_shapes, strict=True)
        ):
            if not isinstance(level_metadata, BlockMetadata):
                raise TypeError("Topology levels must contain BlockMetadata values.")
            active = np.asarray(level_metadata.active, dtype=np.bool_)
            ids = np.asarray(level_metadata.block_ids, dtype=np.int32)
            logical = np.asarray(level_metadata.logical_indices, dtype=np.int32)
            count = int(np.count_nonzero(active))
            if np.any(active[count:]) or np.any(~active[:count]):
                raise ValueError("Active AMR slots must be the canonical compact prefix.")
            active_logical = logical[:count]
            if np.any(active_logical < 0) or np.any(
                active_logical >= np.asarray(lattice, dtype=np.int64)
            ):
                raise ValueError(
                    "Active logical block indices must be unique and in range."
                )
            linear = _lattice_linear(active_logical, lattice)
            if np.unique(linear).size != count:
                raise ValueError(
                    "Active logical block indices must be unique and in range."
                )
            if np.any(np.diff(linear) <= 0) or not np.array_equal(
                ids[:count], plan.block_id_offsets[level_index] + linear
            ):
                raise ValueError(
                    "AMR block IDs and slots must use canonical lattice order."
                )
            linear_by_level.append(linear)
            logical_by_level.append(active_logical)
            boxes_by_level.append(
                tuple(
                    LogicalPatchBox(
                        level_index,
                        tuple(
                            index * size
                            for index, size in zip(
                                row, level_plan.block_shape, strict=True
                            )
                        ),
                        tuple(
                            (index + 1) * size
                            for index, size in zip(
                                row, level_plan.block_shape, strict=True
                            )
                        ),
                    )
                    for row in active_logical
                )
            )
            parents = np.asarray(level_metadata.parent_ids, dtype=np.int32)[:count]
            if level_index == 0:
                if count != prod(lattice):
                    raise ValueError(
                        "Level zero must provide complete base-grid coverage."
                    )
                if np.any(parents != -1):
                    raise ValueError("Level-zero AMR blocks cannot have parents.")
            elif count:
                children = np.asarray(
                    plan.children_per_parent[level_index - 1], dtype=np.int32
                )
                parent_linear = _lattice_linear(
                    active_logical // children,
                    plan.block_lattice_shapes[level_index - 1],
                )
                missing = (
                    _sorted_membership(linear_by_level[level_index - 1], parent_linear)
                    < 0
                )
                rejected = missing | (
                    parents != plan.block_id_offsets[level_index - 1] + parent_linear
                )
                if np.any(rejected):
                    if missing[np.argmax(rejected)]:
                        raise ValueError(
                            "Fine AMR blocks require active aligned parents."
                        )
                    raise ValueError("Fine AMR parent IDs must use canonical identity.")
            expected_neighbors, level_interfaces = _canonical_block_routes(
                active_logical,
                lattice,
                plan.periodic_axes,
                level_plan.maximum_blocks,
            )
            if not np.array_equal(
                np.asarray(level_metadata.neighbor_slots), expected_neighbors
            ):
                raise ValueError(
                    "AMR neighbor slots must match canonical topology routes."
                )
            interfaces.append(jnp.asarray(level_interfaces))
            partition_levels.append(ids.tolist())

        covered_cells: list[Array] = []
        for level_index, level_plan in enumerate(plan.levels):
            mask = np.zeros(
                (level_plan.maximum_blocks,) + level_plan.block_shape,
                dtype=np.bool_,
            )
            active_logical = logical_by_level[level_index]
            count = active_logical.shape[0]
            if level_index + 1 < len(plan.levels) and count:
                # Fine block extents are multiples of the refinement ratio, so the
                # ratio**d children of one coarse cell always share one fine block.
                block = np.asarray(level_plan.block_shape, dtype=np.int64)
                local = np.indices(level_plan.block_shape).reshape(dimension, -1).T
                cells = active_logical[:, None, :] * block + local[None, :, :]
                fine_rows = (cells * level_plan.refinement_ratio) // np.asarray(
                    plan.levels[level_index + 1].block_shape, dtype=np.int64
                )
                fine_linear = _lattice_linear(
                    fine_rows.reshape(-1, dimension),
                    plan.block_lattice_shapes[level_index + 1],
                )
                mask[:count] = (
                    _sorted_membership(linear_by_level[level_index + 1], fine_linear) >= 0
                ).reshape((count,) + level_plan.block_shape)
            covered_cells.append(jnp.asarray(mask))
        topology_id = canonical_fingerprint(
            {
                "kind": "block-hierarchy-topology",
                "plan": plan.plan_id,
                "logical_boxes": [
                    [
                        {
                            "lower": box.lower,
                            "upper": box.upper,
                        }
                        for box in boxes
                    ]
                    for boxes in boxes_by_level
                ],
            }
        )
        partition_id = canonical_fingerprint(
            {
                "kind": "block-hierarchy-partition",
                "plan": plan.plan_id,
                "canonical_slots": partition_levels,
            }
        )
        epoch_ = (
            TopologyEpoch(0, plan.geometry_id, topology_id, partition_id)
            if epoch is None
            else epoch
        )
        if not isinstance(epoch_, TopologyEpoch) or (
            epoch_.geometry_id != plan.geometry_id
            or epoch_.topology_id != topology_id
            or epoch_.partition_id != partition_id
        ):
            raise ValueError(
                "Topology epoch identities do not match realized block topology."
            )
        self.plan = plan
        self.levels = metadata
        self.logical_boxes = tuple(boxes_by_level)
        self.covered_cells = tuple(covered_cells)
        self.interfaces = tuple(interfaces)
        self.epoch = epoch_
        self.topology_id = topology_id
        self.partition_id = partition_id

    def patch_boxes(self, level: int, /) -> tuple[LogicalPatchBox, ...]:
        level_ = int(level)
        if level_ < 0 or level_ >= len(self.logical_boxes):
            raise ValueError("AMR level is out of range.")
        return self.logical_boxes[level_]

    def cell_slot(self, level: int, coordinate: Sequence[int], /) -> int | None:
        values = tuple(coordinate)
        if level < 0 or level >= len(self.logical_boxes):
            raise ValueError("AMR level is out of range.")
        if len(values) != len(self.plan.grid.shape):
            raise ValueError("AMR cell coordinate rank does not match hierarchy.")
        for slot, box in enumerate(self.logical_boxes[level]):
            if box.contains_cell(values):
                return slot
        return None

    def cell_is_active(self, level: int, coordinate: Sequence[int], /) -> bool:
        return self.cell_slot(level, coordinate) is not None

    def covered_mask(self, level: int, slot: int, /) -> Array:
        level_ = int(level)
        slot_ = int(slot)
        if (
            level_ < 0
            or level_ >= len(self.covered_cells)
            or slot_ < 0
            or slot_ >= self.plan.levels[level_].maximum_blocks
        ):
            raise ValueError("AMR covered-cell route is out of range.")
        return self.covered_cells[level_][slot_]


class BlockHierarchyState(StrictModule):
    """Numeric hierarchy payload bound to exactly one realized block topology."""

    topology: BlockHierarchyTopology
    levels: tuple[BlockLevelState, ...]
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology,
        levels: Sequence[BlockLevelState],
        /,
    ) -> None:
        values = tuple(levels)
        if not isinstance(topology, BlockHierarchyTopology) or len(values) != len(
            topology.plan.levels
        ):
            raise TypeError("AMR hierarchy state must bind one hierarchy topology.")
        if any(
            not isinstance(level, BlockLevelState)
            or level.plan.plan_id != expected.plan_id
            or level.metadata.metadata_id != metadata.metadata_id
            for level, expected, metadata in zip(
                values, topology.plan.levels, topology.levels, strict=True
            )
        ):
            raise ValueError(
                "AMR hierarchy payload does not match its realized topology."
            )
        self.topology = topology
        self.levels = values
        self.payload_id = canonical_fingerprint(
            {
                "kind": "amr-hierarchy-payload",
                "epoch": topology.epoch.epoch_id,
                "shapes": [list(level.values.shape) for level in values],
                "dtypes": [str(level.values.dtype) for level in values],
            }
        )

    @property
    def plan(self) -> BlockHierarchyPlan:
        return self.topology.plan


__all__ = [
    "BlockHierarchyPlan",
    "BlockHierarchyState",
    "BlockHierarchyTopology",
    "BlockLevelPlan",
    "BlockLevelState",
    "BlockMetadata",
]
