#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Variable logical-patch AMR topology over finite static execution buckets.

The module is host-only by design.  It owns reference boxes, clustering, bucket
admission, and epoch identities; numerical kernels receive only the resulting fixed
bucket tensors and route indices.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_support import PreparedTensorGrid
from .._topology_epoch import TopologyEpoch
from ._patches import LogicalPatchBox, PatchBucketPlan


class VariablePatchLevelPlan(StrictModule, NonTrainableState):
    """Finite shape catalog and lane capacities at one reference AMR level."""

    level: int = eqx.field(static=True)
    buckets: tuple[PatchBucketPlan, ...]
    refinement_ratio: int = eqx.field(static=True)
    patch_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        buckets: Sequence[PatchBucketPlan],
        /,
        *,
        refinement_ratio: int = 2,
    ):
        level_ = int(level)
        buckets_ = tuple(buckets)
        ratio = int(refinement_ratio)
        if (
            level_ < 0
            or not buckets_
            or not all(isinstance(bucket, PatchBucketPlan) for bucket in buckets_)
            or ratio <= 1
        ):
            raise ValueError("Variable patch level plan is invalid.")
        dimension = len(buckets_[0].signature.envelope_shape)
        if any(
            len(bucket.signature.envelope_shape) != dimension for bucket in buckets_
        ) or len({bucket.signature.signature_id for bucket in buckets_}) != len(buckets_):
            raise ValueError(
                "Variable patch bucket signatures must be unique and same-rank."
            )
        self.level = level_
        self.buckets = buckets_
        self.refinement_ratio = ratio
        self.patch_capacity = sum(bucket.lane_capacity for bucket in buckets_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-level-plan",
                "level": level_,
                "buckets": [bucket.bucket_id for bucket in buckets_],
                "refinement_ratio": ratio,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.buckets[0].signature.envelope_shape)

    def bucket_for_extent(self, extent: Sequence[int], /) -> int | None:
        values = tuple(int(value) for value in extent)
        candidates = tuple(
            (prod(bucket.signature.envelope_shape), bucket.signature.signature_id, index)
            for index, bucket in enumerate(self.buckets)
            if bucket.signature.admits(values)
        )
        return None if not candidates else min(candidates)[2]


class VariablePatchHierarchyPlan(StrictModule, NonTrainableState):
    """Reference hierarchy with an explicit complete base patch partition."""

    grid: PreparedTensorGrid
    levels: tuple[VariablePatchLevelPlan, ...]
    base_boxes: tuple[LogicalPatchBox, ...] = eqx.field(static=True)
    global_cell_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    level_spacings: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        levels: Sequence[VariablePatchLevelPlan],
        base_boxes: Sequence[LogicalPatchBox],
        /,
    ):
        levels_ = tuple(levels)
        boxes = tuple(base_boxes)
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be PreparedTensorGrid.")
        if (
            not levels_
            or not all(isinstance(level, VariablePatchLevelPlan) for level in levels_)
            or tuple(level.level for level in levels_) != tuple(range(len(levels_)))
            or not boxes
            or not all(
                isinstance(box, LogicalPatchBox) and box.level == 0 for box in boxes
            )
        ):
            raise ValueError(
                "Variable patch hierarchy levels and base boxes are invalid."
            )
        dimension = len(grid.shape)
        if any(level.dimension != dimension for level in levels_):
            raise ValueError("Variable patch signatures must match tensor-grid rank.")
        if any(axis.primary_entity != "interval" for axis in grid.axes):
            raise ValueError("Variable patch AMR requires interval-primary tensor axes.")
        widths = tuple(
            np.asarray(axis.interval_widths, dtype=float) for axis in grid.structured_axes
        )
        if any(
            axis.basis != "uniform" or width.size == 0 or not np.all(width == width[0])
            for axis, width in zip(grid.axes, widths, strict=True)
        ):
            raise ValueError("Variable patch AMR requires uniform reference tensor axes.")
        base_shape = tuple(int(value) for value in grid.shape)
        if any(
            any(
                start < 0 or stop > extent
                for start, stop, extent in zip(
                    box.lower, box.upper, base_shape, strict=True
                )
            )
            for box in boxes
        ):
            raise ValueError("Base patch boxes must lie inside the reference grid.")
        if any(
            left.overlaps(right)
            for index, left in enumerate(boxes)
            for right in boxes[index + 1 :]
        ):
            raise ValueError("Base patch boxes must not overlap.")
        if sum(box.cell_count for box in boxes) != prod(base_shape):
            raise ValueError("Base patch boxes must exactly cover the reference grid.")
        if any(levels_[0].bucket_for_extent(box.extent) is None for box in boxes):
            raise ValueError("Every base patch box must fit one level-zero bucket.")
        global_shapes = [base_shape]
        spacings = [tuple(float(width[0]) for width in widths)]
        for level in levels_[:-1]:
            global_shapes.append(
                tuple(value * level.refinement_ratio for value in global_shapes[-1])
            )
            spacings.append(
                tuple(value / level.refinement_ratio for value in spacings[-1])
            )
        self.grid = grid
        self.levels = levels_
        self.base_boxes = tuple(sorted(boxes, key=lambda box: box.box_id))
        self.global_cell_shapes = tuple(global_shapes)
        self.level_spacings = tuple(spacings)
        self.periodic_axes = tuple(bool(axis.periodic) for axis in grid.axes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-hierarchy-plan",
                "grid": grid.prepared_id,
                "levels": [level.plan_id for level in levels_],
                "base_boxes": [box.box_id for box in self.base_boxes],
            }
        )

    @property
    def geometry_id(self) -> str:
        return self.grid.support.embedding_id


class VariablePatchLevelMetadata(StrictModule, NonTrainableState):
    """Bucket/lane realization of ordered logical boxes at one level."""

    plan: VariablePatchLevelPlan
    active: tuple[Array, ...]
    lower: tuple[Array, ...]
    extent: tuple[Array, ...]
    route_indices: tuple[Array, ...]
    boxes: tuple[tuple[LogicalPatchBox | None, ...], ...] = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: VariablePatchLevelPlan,
        boxes_by_bucket: Sequence[Sequence[LogicalPatchBox]],
        /,
    ):
        if not isinstance(plan, VariablePatchLevelPlan):
            raise TypeError("Variable patch metadata requires VariablePatchLevelPlan.")
        groups = tuple(tuple(group) for group in boxes_by_bucket)
        if len(groups) != len(plan.buckets):
            raise ValueError("Variable patch metadata requires one group per bucket.")
        active: list[Array] = []
        lower: list[Array] = []
        extent: list[Array] = []
        route_indices: list[Array] = []
        storage_boxes: list[tuple[LogicalPatchBox | None, ...]] = []
        ordered = sorted(
            (
                (bucket_index, box)
                for bucket_index, group in enumerate(groups)
                for box in group
            ),
            key=lambda item: item[1].box_id,
        )
        route_by_box = {box.box_id: route for route, (_, box) in enumerate(ordered)}
        for bucket_index, (bucket, group) in enumerate(
            zip(plan.buckets, groups, strict=True)
        ):
            if len(group) > bucket.lane_capacity:
                raise ValueError("Variable patch bucket lane capacity is exceeded.")
            if any(
                not isinstance(box, LogicalPatchBox)
                or box.level != plan.level
                or not bucket.signature.admits(box.extent)
                for box in group
            ):
                raise ValueError("Variable patch box does not fit its assigned bucket.")
            canonical = tuple(sorted(group, key=lambda box: box.box_id))
            active_array = np.zeros((bucket.lane_capacity,), dtype=bool)
            lower_array = np.zeros((bucket.lane_capacity, plan.dimension), dtype=np.int32)
            extent_array = np.zeros(
                (bucket.lane_capacity, plan.dimension), dtype=np.int32
            )
            route_array = np.full((bucket.lane_capacity,), -1, dtype=np.int32)
            padded: list[LogicalPatchBox | None] = [None] * bucket.lane_capacity
            for lane, box in enumerate(canonical):
                active_array[lane] = True
                lower_array[lane] = box.lower
                extent_array[lane] = box.extent
                route_array[lane] = route_by_box[box.box_id]
                padded[lane] = box
            active.append(jnp.asarray(active_array))
            lower.append(jnp.asarray(lower_array))
            extent.append(jnp.asarray(extent_array))
            route_indices.append(jnp.asarray(route_array))
            storage_boxes.append(tuple(padded))
        self.plan = plan
        self.active = tuple(active)
        self.lower = tuple(lower)
        self.extent = tuple(extent)
        self.route_indices = tuple(route_indices)
        self.boxes = tuple(storage_boxes)
        self.metadata_id = canonical_fingerprint(
            {
                "kind": "variable-patch-level-metadata",
                "plan": plan.plan_id,
                "boxes": [
                    [None if box is None else box.box_id for box in group]
                    for group in storage_boxes
                ],
            }
        )

    def active_boxes(self, /) -> tuple[tuple[int, int, LogicalPatchBox], ...]:
        return tuple(
            (bucket_index, lane, box)
            for bucket_index, group in enumerate(self.boxes)
            for lane, box in enumerate(group)
            if box is not None
        )

    def locate(
        self, coordinate: Sequence[int], /
    ) -> tuple[int, int, LogicalPatchBox] | None:
        point = tuple(int(value) for value in coordinate)
        for bucket, lane, box in self.active_boxes():
            if box.contains_cell(point):
                return bucket, lane, box
        return None


class VariablePatchHierarchyTopology(StrictModule, NonTrainableState):
    """Sparse logical-patch topology, layout and canonical epoch."""

    plan: VariablePatchHierarchyPlan
    levels: tuple[VariablePatchLevelMetadata, ...]
    logical_boxes: tuple[tuple[LogicalPatchBox, ...], ...] = eqx.field(static=True)
    epoch: TopologyEpoch
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: VariablePatchHierarchyPlan,
        levels: Sequence[VariablePatchLevelMetadata],
        /,
        *,
        epoch: TopologyEpoch | None = None,
    ):
        levels_ = tuple(levels)
        if (
            not isinstance(plan, VariablePatchHierarchyPlan)
            or len(levels_) != len(plan.levels)
            or any(
                not isinstance(metadata, VariablePatchLevelMetadata)
                or metadata.plan.plan_id != level.plan_id
                for metadata, level in zip(levels_, plan.levels, strict=True)
            )
        ):
            raise ValueError("Variable patch topology metadata does not match its plan.")
        boxes_by_level = tuple(
            tuple(
                sorted(
                    (box for _, _, box in metadata.active_boxes()),
                    key=lambda box: (box.level, box.lower, box.upper),
                )
            )
            for metadata in levels_
        )
        for level, (boxes, global_shape) in enumerate(
            zip(boxes_by_level, plan.global_cell_shapes, strict=True)
        ):
            if any(
                any(
                    start < 0 or stop > maximum
                    for start, stop, maximum in zip(
                        box.lower, box.upper, global_shape, strict=True
                    )
                )
                for box in boxes
            ) or any(
                left.overlaps(right)
                for index, left in enumerate(boxes)
                for right in boxes[index + 1 :]
            ):
                raise ValueError("Variable patch boxes are out of range or overlap.")
            if level == 0 and sum(box.cell_count for box in boxes) != prod(global_shape):
                raise ValueError(
                    "Variable level-zero patches must completely cover the grid."
                )
            if level > 0:
                ratio = plan.levels[level - 1].refinement_ratio
                coarse_boxes = boxes_by_level[level - 1]
                for box in boxes:
                    if any(value % ratio for value in (*box.lower, *box.upper)):
                        raise ValueError(
                            "Fine variable patch boxes must align to complete coarse cells."
                        )
                    parent = box.coarsen(ratio)
                    if not all(
                        any(coarse.contains_cell(point) for coarse in coarse_boxes)
                        for point in (
                            tuple(
                                start + offset
                                for start, offset in zip(parent.lower, local, strict=True)
                            )
                            for local in np.ndindex(parent.extent)
                        )
                    ):
                        raise ValueError(
                            "Fine variable patch boxes require complete coarse parent coverage."
                        )
        active_cell_keys = tuple(
            tuple(
                sorted(
                    tuple(
                        start + offset
                        for start, offset in zip(
                            box.lower,
                            local,
                            strict=True,
                        )
                    )
                    for box in boxes
                    for local in np.ndindex(box.extent)
                )
            )
            for boxes in boxes_by_level
        )
        topology_id = canonical_fingerprint(
            {
                "kind": "variable-patch-hierarchy-topology",
                "plan": plan.plan_id,
                "active_cell_keys": active_cell_keys,
            }
        )
        layout_id = canonical_fingerprint(
            {
                "kind": "variable-patch-hierarchy-layout",
                "metadata": [metadata.metadata_id for metadata in levels_],
            }
        )
        partition_id = canonical_fingerprint(
            {
                "kind": "variable-patch-hierarchy-partition",
                "layout": layout_id,
            }
        )
        epoch_ = (
            TopologyEpoch(0, plan.geometry_id, topology_id, partition_id)
            if epoch is None
            else epoch
        )
        if (
            not isinstance(epoch_, TopologyEpoch)
            or epoch_.geometry_id != plan.geometry_id
            or epoch_.topology_id != topology_id
            or epoch_.partition_id != partition_id
        ):
            raise ValueError("Variable patch topology epoch identities do not match.")
        self.plan = plan
        self.levels = levels_
        self.logical_boxes = boxes_by_level
        self.epoch = epoch_
        self.topology_id = topology_id
        self.layout_id = layout_id
        self.partition_id = partition_id

    def locate(
        self, level: int, coordinate: Sequence[int], /
    ) -> tuple[int, int, LogicalPatchBox] | None:
        level_ = int(level)
        if level_ < 0 or level_ >= len(self.levels):
            raise ValueError("Variable patch level is out of range.")
        return self.levels[level_].locate(coordinate)

    def covers_region(
        self, level: int, lower: Sequence[int], upper: Sequence[int], /
    ) -> bool:
        lower_ = tuple(int(value) for value in lower)
        upper_ = tuple(int(value) for value in upper)
        if len(lower_) != len(upper_) or any(
            stop <= start for start, stop in zip(lower_, upper_, strict=True)
        ):
            return False
        return all(
            self.locate(
                level,
                tuple(
                    start + offset for start, offset in zip(lower_, point, strict=True)
                ),
            )
            is not None
            for point in np.ndindex(
                tuple(stop - start for start, stop in zip(lower_, upper_, strict=True))
            )
        )


class VariablePatchCompileStatus(StrictModule, NonTrainableState):
    """Atomic result status for host patch clustering and bucket admission."""

    code: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    changed: bool = eqx.field(static=True)
    message: str = eqx.field(static=True)
    status_id: str = eqx.field(static=True)

    def __init__(self, code: str, successful: bool, changed: bool, message: str, /):
        if code not in {
            "initialized",
            "success",
            "unchanged",
            "bucket_capacity_exceeded",
            "proper_nesting_failed",
            "unsupported_extent",
        }:
            raise ValueError("Unknown variable patch compilation status.")
        if not str(message):
            raise ValueError("Variable patch compilation status message is required.")
        self.code = code
        self.successful = bool(successful)
        self.changed = bool(changed)
        self.message = str(message)
        self.status_id = canonical_fingerprint(
            {
                "kind": "variable-patch-compile-status",
                "code": code,
                "successful": bool(successful),
                "changed": bool(changed),
                "message": str(message),
            }
        )


class VariablePatchCompileEvidence(StrictModule, NonTrainableState):
    """Capacity, clustering and sparse-box evidence for one candidate."""

    tag_counts: tuple[int, ...] = eqx.field(static=True)
    patch_counts: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    bucket_capacities: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    padding_cells: tuple[int, ...] = eqx.field(static=True)
    route_capacity_required: tuple[int, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        tag_counts: Sequence[int],
        patch_counts: Sequence[Sequence[int]],
        bucket_capacities: Sequence[Sequence[int]],
        padding_cells: Sequence[int],
        route_capacity_required: Sequence[int],
        /,
    ):
        tags = tuple(int(value) for value in tag_counts)
        counts = tuple(tuple(int(value) for value in row) for row in patch_counts)
        capacities = tuple(
            tuple(int(value) for value in row) for row in bucket_capacities
        )
        padding = tuple(int(value) for value in padding_cells)
        routes = tuple(int(value) for value in route_capacity_required)
        if (
            len(tags) != len(counts)
            or len(counts) != len(capacities)
            or len(padding) != len(counts)
            or len(routes) != len(counts)
            or any(
                len(left) != len(right)
                for left, right in zip(counts, capacities, strict=True)
            )
        ):
            raise ValueError("Variable patch evidence dimensions are inconsistent.")
        self.tag_counts = tags
        self.patch_counts = counts
        self.bucket_capacities = capacities
        self.padding_cells = padding
        self.route_capacity_required = routes
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "variable-patch-compile-evidence",
                "tags": tags,
                "counts": counts,
                "capacities": capacities,
                "padding": padding,
                "routes": routes,
            }
        )


class VariablePatchCompileResult(StrictModule, NonTrainableState):
    topology: VariablePatchHierarchyTopology
    status: VariablePatchCompileStatus
    evidence: VariablePatchCompileEvidence
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: VariablePatchHierarchyTopology,
        status: VariablePatchCompileStatus,
        evidence: VariablePatchCompileEvidence,
        /,
    ):
        if not isinstance(topology, VariablePatchHierarchyTopology):
            raise TypeError("Variable patch compile result requires topology.")
        self.topology = topology
        self.status = status
        self.evidence = evidence
        self.result_id = canonical_fingerprint(
            {
                "kind": "variable-patch-compile-result",
                "epoch": topology.epoch.epoch_id,
                "status": status.status_id,
                "evidence": evidence.evidence_id,
            }
        )


def _buffer_cells(
    cells: set[tuple[int, ...]],
    width: int,
    shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> set[tuple[int, ...]]:
    result = set(cells)
    for _ in range(width):
        expanded = set(result)
        for cell in result:
            for axis in range(len(shape)):
                for direction in (-1, 1):
                    shifted = list(cell)
                    shifted[axis] += direction
                    if periodic[axis]:
                        shifted[axis] %= shape[axis]
                    if 0 <= shifted[axis] < shape[axis]:
                        expanded.add(tuple(shifted))
        result = expanded
    return result


def _components(cells: set[tuple[int, ...]], /) -> tuple[set[tuple[int, ...]], ...]:
    remaining = set(cells)
    result: list[set[tuple[int, ...]]] = []
    while remaining:
        root = min(remaining)
        component = {root}
        queue = deque((root,))
        remaining.remove(root)
        while queue:
            cell = queue.popleft()
            for axis in range(len(cell)):
                for direction in (-1, 1):
                    neighbour = list(cell)
                    neighbour[axis] += direction
                    neighbour_ = tuple(neighbour)
                    if neighbour_ in remaining:
                        remaining.remove(neighbour_)
                        component.add(neighbour_)
                        queue.append(neighbour_)
        result.append(component)
    return tuple(result)


def _aligned_component_box(
    level: int,
    component: set[tuple[int, ...]],
    alignment: tuple[int, ...],
    shape: tuple[int, ...],
    /,
) -> LogicalPatchBox:
    lower = tuple(
        min(cell[axis] for cell in component) // alignment[axis] * alignment[axis]
        for axis in range(len(shape))
    )
    upper = tuple(
        min(
            shape[axis],
            (max(cell[axis] for cell in component) + 1 + alignment[axis] - 1)
            // alignment[axis]
            * alignment[axis],
        )
        for axis in range(len(shape))
    )
    return LogicalPatchBox(level, lower, upper)


def _split_to_catalog(
    level_plan: VariablePatchLevelPlan,
    box: LogicalPatchBox,
    /,
) -> tuple[tuple[int, LogicalPatchBox], ...] | None:
    bucket = level_plan.bucket_for_extent(box.extent)
    if bucket is not None:
        return ((bucket, box),)
    extent = box.extent
    axis = max(range(box.dimension), key=lambda value: (extent[value], -value))
    minimum = min(bucket.signature.alignment[axis] for bucket in level_plan.buckets)
    split = box.lower[axis] + (extent[axis] // 2 // minimum) * minimum
    if split <= box.lower[axis] or split >= box.upper[axis]:
        return None
    lower_upper = list(box.upper)
    lower_upper[axis] = split
    upper_lower = list(box.lower)
    upper_lower[axis] = split
    left = LogicalPatchBox(box.level, box.lower, tuple(lower_upper))
    right = LogicalPatchBox(box.level, tuple(upper_lower), box.upper)
    left_split = _split_to_catalog(level_plan, left)
    right_split = _split_to_catalog(level_plan, right)
    if left_split is None or right_split is None:
        return None
    return left_split + right_split


class VariablePatchTopologyCompiler(StrictModule, NonTrainableState):
    """Deterministic host clustering into finite static patch buckets."""

    plan: VariablePatchHierarchyPlan
    tag_buffer: int = eqx.field(static=True)
    proper_nesting: int = eqx.field(static=True)
    compiler_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: VariablePatchHierarchyPlan,
        /,
        *,
        tag_buffer: int = 0,
        proper_nesting: int = 0,
    ):
        if not isinstance(plan, VariablePatchHierarchyPlan):
            raise TypeError(
                "Variable patch compiler requires VariablePatchHierarchyPlan."
            )
        if int(tag_buffer) < 0 or int(proper_nesting) < 0:
            raise ValueError("Variable patch buffering and nesting must be non-negative.")
        self.plan = plan
        self.tag_buffer = int(tag_buffer)
        self.proper_nesting = int(proper_nesting)
        self.compiler_id = canonical_fingerprint(
            {
                "kind": "variable-patch-topology-compiler",
                "plan": plan.plan_id,
                "tag_buffer": self.tag_buffer,
                "proper_nesting": self.proper_nesting,
            }
        )

    def initial_topology(self, /) -> VariablePatchHierarchyTopology:
        groups = [[] for _ in self.plan.levels[0].buckets]
        for box in self.plan.base_boxes:
            bucket = self.plan.levels[0].bucket_for_extent(box.extent)
            if bucket is None:
                raise RuntimeError("Validated base box lost its shape-bucket assignment.")
            groups[bucket].append(box)
        metadata = [VariablePatchLevelMetadata(self.plan.levels[0], groups)]
        metadata.extend(
            VariablePatchLevelMetadata(level, [[] for _ in level.buckets])
            for level in self.plan.levels[1:]
        )
        return VariablePatchHierarchyTopology(self.plan, metadata)

    def initialize(self, /) -> VariablePatchCompileResult:
        topology = self.initial_topology()
        counts = tuple(
            tuple(int(jnp.count_nonzero(active)) for active in level.active)
            for level in topology.levels
        )
        evidence = VariablePatchCompileEvidence(
            (0,) * len(topology.levels),
            counts,
            tuple(
                tuple(bucket.lane_capacity for bucket in level.buckets)
                for level in self.plan.levels
            ),
            (0,) * len(topology.levels),
            tuple(sum(sum(row) for row in counts) for _ in topology.levels),
        )
        return VariablePatchCompileResult(
            topology,
            VariablePatchCompileStatus(
                "initialized", True, True, "Complete base patch partition initialized."
            ),
            evidence,
        )

    def _tagged_cells(
        self,
        topology: VariablePatchHierarchyTopology,
        level: int,
        tag_buckets: Sequence[ArrayLike],
        /,
    ) -> set[tuple[int, ...]]:
        metadata = topology.levels[level]
        plan = self.plan.levels[level]
        values = tuple(np.asarray(value) for value in tag_buckets)
        if len(values) != len(plan.buckets):
            raise ValueError("Variable patch tags require one tensor per bucket.")
        result: set[tuple[int, ...]] = set()
        for bucket_index, (bucket, tags, active, lower, extent) in enumerate(
            zip(
                plan.buckets,
                values,
                metadata.active,
                metadata.lower,
                metadata.extent,
                strict=True,
            )
        ):
            expected = (bucket.lane_capacity,) + bucket.signature.envelope_shape
            if tags.shape != expected or tags.dtype != np.dtype(bool):
                raise ValueError(
                    "Variable patch tags must have exact Boolean bucket-envelope shapes."
                )
            active_host = np.asarray(active, dtype=bool)
            lower_host = np.asarray(lower, dtype=np.int32)
            extent_host = np.asarray(extent, dtype=np.int32)
            for lane in range(bucket.lane_capacity):
                valid = tuple(slice(0, int(value)) for value in extent_host[lane])
                invalid = np.ones(bucket.signature.envelope_shape, dtype=bool)
                if active_host[lane]:
                    invalid[valid] = False
                if np.any(tags[lane] & invalid):
                    raise ValueError(
                        "Variable patch tags cannot occupy inactive or envelope-padding cells."
                    )
                if not active_host[lane]:
                    continue
                for local in np.argwhere(tags[lane]):
                    result.add(
                        tuple(
                            int(start) + int(offset)
                            for start, offset in zip(lower_host[lane], local, strict=True)
                        )
                    )
        return result

    def compile(
        self,
        source: VariablePatchHierarchyTopology,
        tags: Sequence[Sequence[ArrayLike]],
        /,
    ) -> VariablePatchCompileResult:
        if (
            not isinstance(source, VariablePatchHierarchyTopology)
            or source.plan.plan_id != self.plan.plan_id
            or len(tags) != len(self.plan.levels) - 1
        ):
            raise ValueError("Variable patch topology compiler inputs are incompatible.")
        groups: list[list[list[LogicalPatchBox]]] = [
            [[] for _ in level.buckets] for level in self.plan.levels
        ]
        for bucket, _, box in source.levels[0].active_boxes():
            groups[0][bucket].append(box)
        tag_counts: list[int] = [0]
        padding: list[int] = [0]
        route_counts: list[int] = [sum(len(group) for group in groups[0])]
        failure: tuple[str, str] | None = None
        for level, tag_values in enumerate(tags):
            tagged = {
                coordinate
                for coordinate in self._tagged_cells(source, level, tag_values)
                if any(
                    box.contains_cell(coordinate)
                    for group in groups[level]
                    for box in group
                )
            }
            buffered = _buffer_cells(
                tagged,
                self.tag_buffer,
                self.plan.global_cell_shapes[level],
                self.plan.periodic_axes,
            )
            tag_counts.append(len(buffered))
            level_plan = self.plan.levels[level + 1]
            alignment = tuple(
                int(
                    np.lcm(
                        np.gcd.reduce(
                            [
                                bucket.signature.alignment[axis]
                                for bucket in level_plan.buckets
                            ]
                        ),
                        self.plan.levels[level].refinement_ratio,
                    )
                )
                for axis in range(level_plan.dimension)
            )
            components = _components(buffered)
            candidate_pairs: list[tuple[int, LogicalPatchBox]] = []
            for component in components:
                refined = {
                    tuple(
                        cell[axis] * self.plan.levels[level].refinement_ratio
                        + child[axis]
                        for axis in range(level_plan.dimension)
                    )
                    for cell in component
                    for child in np.ndindex(
                        (self.plan.levels[level].refinement_ratio,) * level_plan.dimension
                    )
                }
                box = _aligned_component_box(
                    level + 1,
                    refined,
                    alignment,
                    self.plan.global_cell_shapes[level + 1],
                )
                split = _split_to_catalog(level_plan, box)
                if split is None:
                    failure = (
                        "unsupported_extent",
                        "Tagged component cannot fit any admitted patch signature.",
                    )
                    break
                candidate_pairs.extend(split)
            if failure is not None:
                break
            candidate_boxes = tuple(box for _, box in candidate_pairs)
            if any(
                left.overlaps(right)
                for index, left in enumerate(candidate_boxes)
                for right in candidate_boxes[index + 1 :]
            ):
                failure = (
                    "unsupported_extent",
                    "Aligned variable patch boxes overlap after clustering.",
                )
                break
            coarse_boxes = tuple(box for group in groups[level] for box in group)
            ratio = self.plan.levels[level].refinement_ratio
            nested_pairs: list[tuple[int, LogicalPatchBox]] = []
            for bucket, box in candidate_pairs:
                if any(value % ratio for value in (*box.lower, *box.upper)):
                    failure = (
                        "proper_nesting_failed",
                        "Fine patch bounds do not coarsen to complete parent cells.",
                    )
                    break
                parent = box.coarsen(ratio)
                support_lower = tuple(
                    value - self.proper_nesting for value in parent.lower
                )
                support_upper = tuple(
                    value + self.proper_nesting for value in parent.upper
                )
                parent_supported = True
                for offset in np.ndindex(
                    tuple(
                        stop - start
                        for start, stop in zip(support_lower, support_upper, strict=True)
                    )
                ):
                    point = tuple(
                        start + value
                        for start, value in zip(support_lower, offset, strict=True)
                    )
                    mapped = list(point)
                    for axis, extent in enumerate(self.plan.global_cell_shapes[level]):
                        if self.plan.periodic_axes[axis]:
                            mapped[axis] %= extent
                        elif mapped[axis] < 0 or mapped[axis] >= extent:
                            parent_supported = False
                            break
                    if not parent_supported or not any(
                        coarse.contains_cell(tuple(mapped)) for coarse in coarse_boxes
                    ):
                        parent_supported = False
                        break
                if not parent_supported:
                    failure = (
                        "proper_nesting_failed",
                        "Fine patch lacks admitted coarse interpolation support.",
                    )
                    break
                nested_pairs.append((bucket, box))
            if failure is not None:
                break
            for bucket, box in nested_pairs:
                groups[level + 1][bucket].append(box)
            counts = tuple(len(group) for group in groups[level + 1])
            capacities = tuple(bucket.lane_capacity for bucket in level_plan.buckets)
            if any(
                count > capacity
                for count, capacity in zip(counts, capacities, strict=True)
            ):
                failure = (
                    "bucket_capacity_exceeded",
                    "Variable patch candidate exceeds one or more bucket lane capacities.",
                )
                break
            padding.append(
                sum(
                    prod(level_plan.buckets[bucket].signature.envelope_shape)
                    - box.cell_count
                    for bucket, box in nested_pairs
                )
            )
            route_counts.append(sum(counts))
        if failure is not None:
            topology = source
            status = VariablePatchCompileStatus(failure[0], False, False, failure[1])
        else:
            metadata = tuple(
                VariablePatchLevelMetadata(level, bucket_boxes)
                for level, bucket_boxes in zip(self.plan.levels, groups, strict=True)
            )
            candidate = VariablePatchHierarchyTopology(self.plan, metadata)
            if (
                candidate.topology_id == source.topology_id
                and candidate.layout_id == source.layout_id
            ):
                topology = source
                status = VariablePatchCompileStatus(
                    "unchanged", True, False, "Tags preserve the current patch topology."
                )
            else:
                topology = VariablePatchHierarchyTopology(
                    self.plan,
                    metadata,
                    epoch=TopologyEpoch(
                        source.epoch.index + 1,
                        self.plan.geometry_id,
                        candidate.topology_id,
                        candidate.partition_id,
                    ),
                )
                status = VariablePatchCompileStatus(
                    "success", True, True, "Tags compiled into a successor patch epoch."
                )
        patch_counts = tuple(
            tuple(len(group) for group in bucket_boxes) for bucket_boxes in groups
        )
        capacities = tuple(
            tuple(bucket.lane_capacity for bucket in level.buckets)
            for level in self.plan.levels
        )
        while len(tag_counts) < len(self.plan.levels):
            tag_counts.append(0)
        while len(padding) < len(self.plan.levels):
            padding.append(0)
        while len(route_counts) < len(self.plan.levels):
            route_counts.append(0)
        evidence = VariablePatchCompileEvidence(
            tag_counts,
            patch_counts,
            capacities,
            padding,
            route_counts,
        )
        return VariablePatchCompileResult(topology, status, evidence)


class VariablePatchFieldState(StrictModule):
    """One static bucketed field payload over a realized variable patch level."""

    metadata: VariablePatchLevelMetadata
    values: tuple[Array, ...]
    component_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        metadata: VariablePatchLevelMetadata,
        values: Sequence[ArrayLike],
        /,
    ):
        arrays = tuple(jnp.asarray(value) for value in values)
        if not isinstance(metadata, VariablePatchLevelMetadata) or len(arrays) != len(
            metadata.plan.buckets
        ):
            raise ValueError(
                "Variable patch field state does not match metadata buckets."
            )
        component_shape: tuple[int, ...] | None = None
        for bucket, array in zip(metadata.plan.buckets, arrays, strict=True):
            prefix = (bucket.lane_capacity,) + bucket.signature.envelope_shape
            if array.shape[: len(prefix)] != prefix:
                raise ValueError(
                    "Variable patch field array does not match bucket envelope."
                )
            suffix = array.shape[len(prefix) :]
            if component_shape is None:
                component_shape = suffix
            elif suffix != component_shape:
                raise ValueError(
                    "Variable patch field buckets require one component shape."
                )
        self.metadata = metadata
        self.values = arrays
        self.component_shape = () if component_shape is None else component_shape

    def safe_values(self, /) -> tuple[Array, ...]:
        result = []
        for value, active, extent in zip(
            self.values,
            self.metadata.active,
            self.metadata.extent,
            strict=True,
        ):
            envelope = value.shape[1 : 1 + extent.shape[1]]
            lane_count = int(active.shape[0])
            valid = jnp.asarray(active).reshape((lane_count,) + (1,) * len(envelope))
            for axis, size in enumerate(envelope):
                coordinate = jnp.arange(size).reshape(
                    (1,) + (1,) * axis + (size,) + (1,) * (len(envelope) - axis - 1)
                )
                limit = jnp.asarray(extent[:, axis]).reshape(
                    (lane_count,) + (1,) * len(envelope)
                )
                valid = valid & (coordinate < limit)
            valid = valid.reshape(valid.shape + (1,) * (value.ndim - valid.ndim))
            result.append(jnp.where(valid, value, jnp.zeros((), dtype=value.dtype)))
        return tuple(result)


__all__ = [
    "VariablePatchCompileEvidence",
    "VariablePatchCompileResult",
    "VariablePatchCompileStatus",
    "VariablePatchFieldState",
    "VariablePatchHierarchyPlan",
    "VariablePatchHierarchyTopology",
    "VariablePatchLevelMetadata",
    "VariablePatchLevelPlan",
    "VariablePatchTopologyCompiler",
]
