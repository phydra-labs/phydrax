#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

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
    ):
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
    ):
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
    ):
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
    ):
        metadata = tuple(levels)
        if not isinstance(plan, BlockHierarchyPlan) or len(metadata) != len(plan.levels):
            raise TypeError("Block hierarchy topology must match one hierarchy plan.")
        dimension = len(plan.grid.shape)
        topology_levels: list[list[int]] = []
        partition_levels: list[list[int]] = []
        active_logical_by_level: list[dict[tuple[int, ...], int]] = []
        boxes_by_level: list[tuple[LogicalPatchBox, ...]] = []
        active_rows_by_level: list[tuple[tuple[int, ...], ...]] = []
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
            active_rows = tuple(tuple(row) for row in logical[:count])
            if len(set(active_rows)) != count or any(
                any(
                    value < 0 or value >= extent
                    for value, extent in zip(row, lattice, strict=True)
                )
                for row in active_rows
            ):
                raise ValueError(
                    "Active logical block indices must be unique and in range."
                )
            expected_ids = np.asarray(
                [plan.block_id(level_index, row) for row in active_rows],
                dtype=np.int32,
            )
            order = np.argsort(expected_ids, kind="stable")
            if not np.array_equal(order, np.arange(count)) or not np.array_equal(
                ids[:count], expected_ids
            ):
                raise ValueError(
                    "AMR block IDs and slots must use canonical lattice order."
                )
            logical_to_slot = {row: slot for slot, row in enumerate(active_rows)}
            active_logical_by_level.append(logical_to_slot)
            active_rows_by_level.append(active_rows)
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
                    for row in active_rows
                )
            )
            if level_index == 0:
                expected_base = set(np.ndindex(lattice))
                if set(active_rows) != expected_base:
                    raise ValueError(
                        "Level zero must provide complete base-grid coverage."
                    )
                if np.any(np.asarray(level_metadata.parent_ids)[:count] != -1):
                    raise ValueError("Level-zero AMR blocks cannot have parents.")
            else:
                parent_map = active_logical_by_level[level_index - 1]
                children = plan.children_per_parent[level_index - 1]
                parents = np.asarray(level_metadata.parent_ids, dtype=np.int32)
                for slot, row in enumerate(active_rows):
                    parent_logical = tuple(
                        value // child for value, child in zip(row, children, strict=True)
                    )
                    if parent_logical not in parent_map:
                        raise ValueError(
                            "Fine AMR blocks require active aligned parents."
                        )
                    expected_parent = plan.block_id(level_index - 1, parent_logical)
                    if int(parents[slot]) != expected_parent:
                        raise ValueError(
                            "Fine AMR parent IDs must use canonical identity."
                        )
            expected_neighbors = np.full(
                (level_plan.maximum_blocks, dimension, 2), -1, dtype=np.int32
            )
            for slot, row in enumerate(active_rows):
                for axis in range(dimension):
                    for side, delta in enumerate((-1, 1)):
                        neighbor = list(row)
                        neighbor[axis] += delta
                        if plan.periodic_axes[axis]:
                            neighbor[axis] %= lattice[axis]
                        neighbor_tuple = tuple(neighbor)
                        if (
                            0 <= neighbor[axis] < lattice[axis]
                            and neighbor_tuple in logical_to_slot
                        ):
                            expected_neighbors[slot, axis, side] = logical_to_slot[
                                neighbor_tuple
                            ]
            if not np.array_equal(
                np.asarray(level_metadata.neighbor_slots), expected_neighbors
            ):
                raise ValueError(
                    "AMR neighbor slots must match canonical topology routes."
                )
            level_interfaces = np.zeros(
                (level_plan.maximum_blocks, dimension, 2), dtype=np.bool_
            )
            for slot, row in enumerate(active_rows):
                for axis in range(dimension):
                    for side, delta in enumerate((-1, 1)):
                        neighbor = list(row)
                        neighbor[axis] += delta
                        inside = 0 <= neighbor[axis] < lattice[axis]
                        if plan.periodic_axes[axis]:
                            neighbor[axis] %= lattice[axis]
                            inside = True
                        if inside and tuple(neighbor) not in logical_to_slot:
                            level_interfaces[slot, axis, side] = True
            interfaces.append(jnp.asarray(level_interfaces))
            topology_levels.append(ids[:count].tolist())
            partition_levels.append(ids.tolist())

        covered_cells: list[Array] = []
        for level_index, (level_plan, active_rows) in enumerate(
            zip(plan.levels, active_rows_by_level, strict=True)
        ):
            mask = np.zeros(
                (level_plan.maximum_blocks,) + level_plan.block_shape,
                dtype=np.bool_,
            )
            if level_index + 1 < len(plan.levels):
                ratio = plan.levels[level_index].refinement_ratio
                fine_boxes = boxes_by_level[level_index + 1]
                for slot, row in enumerate(active_rows):
                    origin = tuple(
                        index * size
                        for index, size in zip(row, level_plan.block_shape, strict=True)
                    )
                    for local in np.ndindex(level_plan.block_shape):
                        lower = tuple(
                            (start + value) * ratio
                            for start, value in zip(origin, local, strict=True)
                        )
                        upper = tuple(value + ratio for value in lower)
                        child_offsets = tuple(
                            np.ndindex(
                                tuple(
                                    stop - start
                                    for start, stop in zip(lower, upper, strict=True)
                                )
                            )
                        )
                        mask[(slot,) + local] = all(
                            any(
                                box.contains_cell(
                                    tuple(
                                        start + offset
                                        for start, offset in zip(
                                            lower, point, strict=True
                                        )
                                    )
                                )
                                for box in fine_boxes
                            )
                            for point in child_offsets
                        )
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
    ):
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
