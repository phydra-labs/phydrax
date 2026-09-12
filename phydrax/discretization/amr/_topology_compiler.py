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
from .._topology_epoch import TopologyEpoch
from ._core import (
    BlockHierarchyPlan,
    BlockHierarchyTopology,
    BlockMetadata,
)


class BlockTopologyCompileStatus(StrictModule, NonTrainableState):
    """Atomic host topology-selection outcome."""

    code: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    changed: bool = eqx.field(static=True)
    message: str = eqx.field(static=True)
    status_id: str = eqx.field(static=True)

    def __init__(self, code: str, successful: bool, changed: bool, message: str, /):
        code_ = str(code)
        message_ = str(message)
        if code_ not in (
            "initialized",
            "success",
            "unchanged",
            "capacity_exceeded",
            "proper_nesting_failed",
        ):
            raise ValueError("Unknown block-topology compilation status.")
        if not message_:
            raise ValueError("Block-topology status message must be non-empty.")
        self.code = code_
        self.successful = bool(successful)
        self.changed = bool(changed)
        self.message = message_
        self.status_id = canonical_fingerprint(
            {
                "kind": "block-topology-compile-status",
                "code": code_,
                "successful": bool(successful),
                "changed": bool(changed),
                "message": message_,
            }
        )


class BlockTopologyCompileEvidence(StrictModule, NonTrainableState):
    """Auditable block counts, tag counts, nesting rejections, and capacity."""

    requested_blocks: tuple[int, ...] = eqx.field(static=True)
    realized_blocks: tuple[int, ...] = eqx.field(static=True)
    capacities: tuple[int, ...] = eqx.field(static=True)
    buffered_tagged_cells: tuple[int, ...] = eqx.field(static=True)
    proper_nesting_rejections: tuple[int, ...] = eqx.field(static=True)
    overflow_level: int | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        requested_blocks: Sequence[int],
        realized_blocks: Sequence[int],
        capacities: Sequence[int],
        buffered_tagged_cells: Sequence[int],
        proper_nesting_rejections: Sequence[int],
        overflow_level: int | None,
        /,
    ):
        requested = tuple(int(value) for value in requested_blocks)
        realized = tuple(int(value) for value in realized_blocks)
        capacities_ = tuple(int(value) for value in capacities)
        tagged = tuple(int(value) for value in buffered_tagged_cells)
        rejected = tuple(int(value) for value in proper_nesting_rejections)
        if not (
            len(requested) == len(realized) == len(capacities_)
            and len(tagged) == len(rejected) == max(0, len(requested) - 1)
        ):
            raise ValueError(
                "Topology compilation evidence has inconsistent level counts."
            )
        overflow = None if overflow_level is None else int(overflow_level)
        self.requested_blocks = requested
        self.realized_blocks = realized
        self.capacities = capacities_
        self.buffered_tagged_cells = tagged
        self.proper_nesting_rejections = rejected
        self.overflow_level = overflow
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "block-topology-compile-evidence",
                "requested": requested,
                "realized": realized,
                "capacities": capacities_,
                "tagged": tagged,
                "nesting_rejections": rejected,
                "overflow_level": overflow,
            }
        )


class BlockTopologyRouteGraph(StrictModule, NonTrainableState):
    """Canonical old/new slots plus fine-parent routes for one topology result."""

    old_to_new_slots: tuple[Array, ...]
    new_to_old_slots: tuple[Array, ...]
    parent_slots: tuple[Array, ...]
    child_offsets: tuple[Array, ...]
    route_graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        old_to_new_slots: Sequence[ArrayLike],
        new_to_old_slots: Sequence[ArrayLike],
        parent_slots: Sequence[ArrayLike],
        child_offsets: Sequence[ArrayLike],
        /,
    ):
        old_to_new = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in old_to_new_slots
        )
        new_to_old = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in new_to_old_slots
        )
        parents = tuple(jnp.asarray(value, dtype=jnp.int32) for value in parent_slots)
        offsets = tuple(jnp.asarray(value, dtype=jnp.int32) for value in child_offsets)
        if not (len(old_to_new) == len(new_to_old) == len(parents) == len(offsets)):
            raise ValueError("Topology route graph level counts must agree.")
        self.old_to_new_slots = old_to_new
        self.new_to_old_slots = new_to_old
        self.parent_slots = parents
        self.child_offsets = offsets
        self.route_graph_id = canonical_fingerprint(
            {
                "kind": "block-topology-route-graph",
                "old_to_new": [array_tree_fingerprint(value) for value in old_to_new],
                "new_to_old": [array_tree_fingerprint(value) for value in new_to_old],
                "parent_slots": [array_tree_fingerprint(value) for value in parents],
                "child_offsets": [array_tree_fingerprint(value) for value in offsets],
            }
        )


class BlockTopologyCompileResult(StrictModule, NonTrainableState):
    """Atomic topology, routes, status, and evidence returned by the host compiler."""

    topology: BlockHierarchyTopology
    routes: BlockTopologyRouteGraph
    status: BlockTopologyCompileStatus
    evidence: BlockTopologyCompileEvidence
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology,
        routes: BlockTopologyRouteGraph,
        status: BlockTopologyCompileStatus,
        evidence: BlockTopologyCompileEvidence,
        /,
    ):
        if not isinstance(topology, BlockHierarchyTopology):
            raise TypeError("Compiled topology result requires BlockHierarchyTopology.")
        self.topology = topology
        self.routes = routes
        self.status = status
        self.evidence = evidence
        self.result_id = canonical_fingerprint(
            {
                "kind": "block-topology-compile-result",
                "epoch": topology.epoch.epoch_id,
                "routes": routes.route_graph_id,
                "status": status.status_id,
                "evidence": evidence.evidence_id,
            }
        )


def _active_tagged_cells(
    topology: BlockHierarchyTopology,
    level: int,
    values: np.ndarray,
    /,
) -> set[tuple[int, ...]]:
    """Expand local tags only over active patch storage, never a full level lattice."""
    metadata = topology.levels[level]
    level_plan = topology.plan.levels[level]
    logical = np.asarray(metadata.logical_indices, dtype=np.int32)
    result: set[tuple[int, ...]] = set()
    for slot in np.flatnonzero(np.asarray(metadata.active, dtype=bool)):
        origin = tuple(
            int(index) * size
            for index, size in zip(logical[slot], level_plan.block_shape, strict=True)
        )
        for local in np.argwhere(values[slot]):
            result.add(
                tuple(
                    start + int(offset)
                    for start, offset in zip(origin, local, strict=True)
                )
            )
    return result


def _buffer_cells(
    cells: set[tuple[int, ...]],
    width: int,
    global_shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> set[tuple[int, ...]]:
    result = set(cells)
    for _ in range(width):
        expanded = set(result)
        for cell in result:
            for axis in range(len(global_shape)):
                for delta in (-1, 1):
                    shifted = list(cell)
                    shifted[axis] += delta
                    if periodic[axis]:
                        shifted[axis] %= global_shape[axis]
                    if 0 <= shifted[axis] < global_shape[axis]:
                        expanded.add(tuple(shifted))
        result = expanded
    return result


def _candidate_fine_blocks(
    buffered_cells: set[tuple[int, ...]],
    ratio: int,
    fine_shape: tuple[int, ...],
    /,
) -> set[tuple[int, ...]]:
    result: set[tuple[int, ...]] = set()
    for coarse_cell in buffered_cells:
        for child in np.ndindex((ratio,) * len(fine_shape)):
            fine_cell = tuple(
                int(cell) * ratio + int(offset)
                for cell, offset in zip(coarse_cell, child, strict=True)
            )
            result.add(
                tuple(
                    coordinate // extent
                    for coordinate, extent in zip(fine_cell, fine_shape, strict=True)
                )
            )
    return result


def _region_is_nested(
    coarse_rows: set[tuple[int, ...]],
    coarse_shape: tuple[int, ...],
    global_shape: tuple[int, ...],
    starts: tuple[int, ...],
    stops: tuple[int, ...],
    width: int,
    periodic: tuple[bool, ...],
    /,
) -> bool:
    lower = tuple(start - width for start in starts)
    upper = tuple(stop + width for stop in stops)
    for offset in np.ndindex(
        tuple(stop - start for start, stop in zip(lower, upper, strict=True))
    ):
        coordinate = tuple(
            start + value for start, value in zip(lower, offset, strict=True)
        )
        mapped = list(coordinate)
        for axis, extent in enumerate(global_shape):
            if periodic[axis]:
                mapped[axis] %= extent
            elif mapped[axis] < 0 or mapped[axis] >= extent:
                return False
        row = tuple(
            value // extent for value, extent in zip(mapped, coarse_shape, strict=True)
        )
        if row not in coarse_rows:
            return False
    return True


def _metadata_from_logical(
    plan: BlockHierarchyPlan,
    level: int,
    logical_indices: Sequence[tuple[int, ...]],
) -> BlockMetadata:
    level_plan = plan.levels[level]
    capacity = level_plan.maximum_blocks
    logical = sorted(
        (tuple(int(value) for value in row) for row in logical_indices),
        key=lambda row: plan.block_id(level, row),
    )
    if len(logical) > capacity:
        raise ValueError("Internal topology metadata construction exceeded capacity.")
    count = len(logical)
    active = np.zeros((capacity,), dtype=bool)
    active[:count] = True
    block_ids = np.full((capacity,), -1, dtype=np.int32)
    block_ids[:count] = [plan.block_id(level, row) for row in logical]
    parent_ids = np.full((capacity,), -1, dtype=np.int32)
    logical_array = np.full((capacity, len(level_plan.block_shape)), -1, dtype=np.int32)
    if count:
        logical_array[:count] = logical
    if level > 0:
        children = plan.children_per_parent[level - 1]
        parent_ids[:count] = [
            plan.block_id(
                level - 1,
                tuple(value // child for value, child in zip(row, children, strict=True)),
            )
            for row in logical
        ]
    logical_to_slot = {row: slot for slot, row in enumerate(logical)}
    lattice = plan.block_lattice_shapes[level]
    neighbors = np.full((capacity, len(level_plan.block_shape), 2), -1, dtype=np.int32)
    for slot, row in enumerate(logical):
        for axis in range(len(row)):
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
                    neighbors[slot, axis, side] = logical_to_slot[neighbor_tuple]
    return BlockMetadata(
        level_plan,
        active=active,
        block_ids=block_ids,
        parent_ids=parent_ids,
        logical_indices=logical_array,
        neighbor_slots=neighbors,
    )


def _route_graph(
    source: BlockHierarchyTopology,
    target: BlockHierarchyTopology,
) -> BlockTopologyRouteGraph:
    plan = source.plan
    old_to_new: list[np.ndarray] = []
    new_to_old: list[np.ndarray] = []
    parent_slots: list[np.ndarray] = []
    child_offsets: list[np.ndarray] = []
    for level, level_plan in enumerate(plan.levels):
        source_ids = np.asarray(source.levels[level].block_ids, dtype=np.int32)
        target_ids = np.asarray(target.levels[level].block_ids, dtype=np.int32)
        source_by_id = {
            int(value): slot for slot, value in enumerate(source_ids) if value >= 0
        }
        target_by_id = {
            int(value): slot for slot, value in enumerate(target_ids) if value >= 0
        }
        old_route = np.asarray(
            [
                target_by_id.get(int(value), -1) if value >= 0 else -1
                for value in source_ids
            ],
            dtype=np.int32,
        )
        new_route = np.asarray(
            [
                source_by_id.get(int(value), -1) if value >= 0 else -1
                for value in target_ids
            ],
            dtype=np.int32,
        )
        parents = np.full((level_plan.maximum_blocks,), -1, dtype=np.int32)
        offsets = np.full(
            (level_plan.maximum_blocks, len(level_plan.block_shape)),
            -1,
            dtype=np.int32,
        )
        if level > 0:
            coarse_ids = np.asarray(target.levels[level - 1].block_ids, dtype=np.int32)
            coarse_by_id = {
                int(value): slot for slot, value in enumerate(coarse_ids) if value >= 0
            }
            count = int(np.count_nonzero(np.asarray(target.levels[level].active)))
            logical = np.asarray(target.levels[level].logical_indices)
            target_parents = np.asarray(target.levels[level].parent_ids)
            children = plan.children_per_parent[level - 1]
            for slot in range(count):
                parents[slot] = coarse_by_id[int(target_parents[slot])]
                offsets[slot] = tuple(
                    int(value) % child
                    for value, child in zip(logical[slot], children, strict=True)
                )
        old_to_new.append(old_route)
        new_to_old.append(new_route)
        parent_slots.append(parents)
        child_offsets.append(offsets)
    return BlockTopologyRouteGraph(old_to_new, new_to_old, parent_slots, child_offsets)


class BlockTopologyCompiler(StrictModule, NonTrainableState):
    """Host-only fixed-block compiler for buffered cell tags and nested coverage."""

    plan: BlockHierarchyPlan
    tag_buffer: int = eqx.field(static=True)
    proper_nesting: int = eqx.field(static=True)
    compiler_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: BlockHierarchyPlan,
        /,
        *,
        tag_buffer: int = 0,
        proper_nesting: int = 0,
    ):
        if not isinstance(plan, BlockHierarchyPlan):
            raise TypeError("Block topology compiler requires BlockHierarchyPlan.")
        buffer_ = int(tag_buffer)
        nesting = int(proper_nesting)
        if buffer_ < 0 or nesting < 0:
            raise ValueError(
                "Tag buffering and proper-nesting widths must be non-negative."
            )
        self.plan = plan
        self.tag_buffer = buffer_
        self.proper_nesting = nesting
        self.compiler_id = canonical_fingerprint(
            {
                "kind": "block-topology-compiler",
                "plan": plan.plan_id,
                "tag_buffer": buffer_,
                "proper_nesting": nesting,
            }
        )

    def initial_topology(self, /) -> BlockHierarchyTopology:
        base = tuple(np.ndindex(self.plan.block_lattice_shapes[0]))
        logical = (base,) + ((),) * (len(self.plan.levels) - 1)
        metadata = tuple(
            _metadata_from_logical(self.plan, level, rows)
            for level, rows in enumerate(logical)
        )
        return BlockHierarchyTopology(self.plan, metadata)

    def initialize(self, /) -> BlockTopologyCompileResult:
        topology = self.initial_topology()
        counts = tuple(
            int(np.count_nonzero(np.asarray(level.active))) for level in topology.levels
        )
        evidence = BlockTopologyCompileEvidence(
            counts,
            counts,
            tuple(level.maximum_blocks for level in self.plan.levels),
            (0,) * (len(self.plan.levels) - 1),
            (0,) * (len(self.plan.levels) - 1),
            None,
        )
        return BlockTopologyCompileResult(
            topology,
            _route_graph(topology, topology),
            BlockTopologyCompileStatus(
                "initialized", True, True, "Complete base coverage initialized."
            ),
            evidence,
        )

    def compile(
        self,
        source: BlockHierarchyTopology,
        block_tags: Sequence[ArrayLike],
        /,
    ) -> BlockTopologyCompileResult:
        if (
            not isinstance(source, BlockHierarchyTopology)
            or source.plan.plan_id != self.plan.plan_id
        ):
            raise ValueError(
                "Topology compiler source does not match its hierarchy plan."
            )
        tags = tuple(np.asarray(value) for value in block_tags)
        if any(value.dtype != np.dtype(bool) for value in tags):
            raise TypeError("Block refinement tags must have exact Boolean dtype.")
        if len(tags) != len(self.plan.levels) - 1:
            raise ValueError(
                "One block-local Boolean tag field is required per refinable level."
            )
        for level, value in enumerate(tags):
            expected = (self.plan.levels[level].maximum_blocks,) + self.plan.levels[
                level
            ].block_shape
            if value.shape != expected:
                raise ValueError(
                    f"Level {level} tags must have fixed-capacity block shape {expected}."
                )
            active = np.asarray(source.levels[level].active, dtype=bool)
            inactive_shape = (active.size,) + (1,) * len(
                self.plan.levels[level].block_shape
            )
            if np.any(value & ~active.reshape(inactive_shape)):
                raise ValueError("Inactive block slots cannot carry refinement tags.")

        desired: list[tuple[tuple[int, ...], ...]] = [
            tuple(np.ndindex(self.plan.block_lattice_shapes[0]))
        ]
        requested = [len(desired[0])]
        tagged_counts: list[int] = []
        nesting_rejections: list[int] = []
        for level, local_tags in enumerate(tags):
            level_plan = self.plan.levels[level]
            fine_plan = self.plan.levels[level + 1]
            desired_rows = set(desired[level])
            tagged = {
                coordinate
                for coordinate in _active_tagged_cells(source, level, local_tags)
                if tuple(
                    value // extent
                    for value, extent in zip(
                        coordinate, level_plan.block_shape, strict=True
                    )
                )
                in desired_rows
            }
            buffered = _buffer_cells(
                tagged,
                self.tag_buffer,
                self.plan.global_cell_shapes[level],
                self.plan.periodic_axes,
            )
            tagged_counts.append(len(buffered))
            ratio = level_plan.refinement_ratio
            candidates = {
                row
                for row in _candidate_fine_blocks(
                    buffered,
                    ratio,
                    fine_plan.block_shape,
                )
                if all(
                    0 <= value < extent
                    for value, extent in zip(
                        row,
                        self.plan.block_lattice_shapes[level + 1],
                        strict=True,
                    )
                )
            }
            requested_candidates = len(candidates)
            accepted: list[tuple[int, ...]] = []
            rejected = 0
            for row in sorted(
                candidates,
                key=lambda value: self.plan.block_id(level + 1, value),
            ):
                fine_starts = tuple(
                    index * size
                    for index, size in zip(row, fine_plan.block_shape, strict=True)
                )
                fine_stops = tuple(
                    start + size
                    for start, size in zip(
                        fine_starts, fine_plan.block_shape, strict=True
                    )
                )
                coarse_starts = tuple(start // ratio for start in fine_starts)
                coarse_stops = tuple(stop // ratio for stop in fine_stops)
                if _region_is_nested(
                    desired_rows,
                    level_plan.block_shape,
                    self.plan.global_cell_shapes[level],
                    coarse_starts,
                    coarse_stops,
                    self.proper_nesting,
                    self.plan.periodic_axes,
                ):
                    accepted.append(row)
                else:
                    rejected += 1
            desired.append(tuple(accepted))
            requested.append(requested_candidates)
            nesting_rejections.append(rejected)

        capacities = tuple(level.maximum_blocks for level in self.plan.levels)
        overflow_level = next(
            (
                level
                for level, (count, capacity) in enumerate(
                    zip(requested, capacities, strict=True)
                )
                if count > capacity
            ),
            None,
        )
        source_counts = tuple(
            int(np.count_nonzero(np.asarray(level.active))) for level in source.levels
        )
        if any(nesting_rejections):
            evidence = BlockTopologyCompileEvidence(
                requested,
                source_counts,
                capacities,
                tagged_counts,
                nesting_rejections,
                None,
            )
            first_rejected = next(
                level + 1 for level, count in enumerate(nesting_rejections) if count
            )
            return BlockTopologyCompileResult(
                source,
                _route_graph(source, source),
                BlockTopologyCompileStatus(
                    "proper_nesting_failed",
                    False,
                    False,
                    f"Level {first_rejected} requested blocks outside proper nesting.",
                ),
                evidence,
            )
        if overflow_level is not None:
            evidence = BlockTopologyCompileEvidence(
                requested,
                source_counts,
                capacities,
                tagged_counts,
                nesting_rejections,
                overflow_level,
            )
            return BlockTopologyCompileResult(
                source,
                _route_graph(source, source),
                BlockTopologyCompileStatus(
                    "capacity_exceeded",
                    False,
                    False,
                    f"Level {overflow_level} requested more blocks than fixed capacity.",
                ),
                evidence,
            )

        metadata = tuple(
            _metadata_from_logical(self.plan, level, rows)
            for level, rows in enumerate(desired)
        )
        candidate = BlockHierarchyTopology(self.plan, metadata)
        evidence = BlockTopologyCompileEvidence(
            requested,
            tuple(len(rows) for rows in desired),
            capacities,
            tagged_counts,
            nesting_rejections,
            None,
        )
        if (
            candidate.topology_id == source.topology_id
            and candidate.partition_id == source.partition_id
        ):
            return BlockTopologyCompileResult(
                source,
                _route_graph(source, source),
                BlockTopologyCompileStatus(
                    "unchanged",
                    True,
                    False,
                    "Refinement tags preserve the current epoch.",
                ),
                evidence,
            )
        epoch = TopologyEpoch(
            source.epoch.index + 1,
            self.plan.geometry_id,
            candidate.topology_id,
            candidate.partition_id,
        )
        target = BlockHierarchyTopology(self.plan, metadata, epoch=epoch)
        return BlockTopologyCompileResult(
            target,
            _route_graph(source, target),
            BlockTopologyCompileStatus(
                "success", True, True, "Refinement tags compiled into a successor epoch."
            ),
            evidence,
        )


__all__ = [
    "BlockTopologyCompileEvidence",
    "BlockTopologyCompileResult",
    "BlockTopologyCompileStatus",
    "BlockTopologyCompiler",
    "BlockTopologyRouteGraph",
]
