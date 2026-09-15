#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical bucketed view and resource contract for block AMR hierarchies."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import BlockHierarchyTopology
from ._patches import (
    BlockHierarchyCapacityPlan,
    LogicalPatchBox,
    PatchBucketPlan,
    PatchShapeSignature,
)
from ._variable import VariablePatchHierarchyTopology


HierarchyTopology = BlockHierarchyTopology | VariablePatchHierarchyTopology


class CanonicalPatchBucket(StrictModule, NonTrainableState):
    """One statically shaped bucket in the canonical hierarchy view."""

    plan: PatchBucketPlan
    active: Array
    lower: Array
    extent: Array
    route_indices: Array
    cell_active: Array
    leaf_active: Array
    boxes: tuple[LogicalPatchBox | None, ...] = eqx.field(static=True)
    bucket_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PatchBucketPlan,
        *,
        active: np.ndarray,
        lower: np.ndarray,
        extent: np.ndarray,
        route_indices: np.ndarray,
        cell_active: np.ndarray,
        leaf_active: np.ndarray,
        boxes: Sequence[LogicalPatchBox | None],
    ):
        if not isinstance(plan, PatchBucketPlan):
            raise TypeError("Canonical patch buckets require PatchBucketPlan.")
        lane_count = plan.lane_capacity
        dimension = len(plan.signature.envelope_shape)
        lane_shape = (lane_count,)
        metadata_shape = (lane_count, dimension)
        storage_shape = lane_shape + plan.signature.envelope_shape
        active_ = np.asarray(active, dtype=bool)
        lower_ = np.asarray(lower)
        extent_ = np.asarray(extent)
        routes_ = np.asarray(route_indices)
        cell_active_ = np.asarray(cell_active, dtype=bool)
        leaf_active_ = np.asarray(leaf_active, dtype=bool)
        boxes_ = tuple(boxes)
        if (
            active_.shape != lane_shape
            or lower_.shape != metadata_shape
            or extent_.shape != metadata_shape
            or routes_.shape != lane_shape
            or cell_active_.shape != storage_shape
            or leaf_active_.shape != storage_shape
            or len(boxes_) != lane_count
            or lower_.dtype.kind not in "iu"
            or extent_.dtype.kind not in "iu"
            or routes_.dtype.kind not in "iu"
        ):
            raise ValueError("Canonical patch bucket arrays do not match their envelope.")
        if np.any(leaf_active_ & ~cell_active_):
            raise ValueError("Leaf-active cells must be active patch cells.")
        for lane, box in enumerate(boxes_):
            if bool(active_[lane]) != (box is not None):
                raise ValueError("Canonical active lanes and logical boxes disagree.")
            if box is None:
                if (
                    np.any(lower_[lane] != 0)
                    or np.any(extent_[lane] != 0)
                    or int(routes_[lane]) != -1
                    or np.any(cell_active_[lane])
                    or np.any(leaf_active_[lane])
                ):
                    raise ValueError(
                        "Inactive canonical lanes require zero/sentinel data."
                    )
                continue
            if (
                tuple(int(value) for value in lower_[lane]) != box.lower
                or tuple(int(value) for value in extent_[lane]) != box.extent
            ):
                raise ValueError(
                    "Canonical lane metadata does not match its logical box."
                )
            if not plan.signature.admits(box.extent) or int(routes_[lane]) < 0:
                raise ValueError("Canonical active lane is not admitted by its bucket.")
        self.plan = plan
        self.active = jnp.asarray(active_)
        self.lower = jnp.asarray(lower_, dtype=jnp.int32)
        self.extent = jnp.asarray(extent_, dtype=jnp.int32)
        self.route_indices = jnp.asarray(routes_, dtype=jnp.int32)
        self.cell_active = jnp.asarray(cell_active_)
        self.leaf_active = jnp.asarray(leaf_active_)
        self.boxes = boxes_
        self.bucket_id = canonical_fingerprint(
            {
                "kind": "canonical-amr-patch-bucket",
                "plan": plan.bucket_id,
                "boxes": [None if box is None else box.box_id for box in boxes_],
                "routes": routes_.tolist(),
                "leaf_count": int(np.count_nonzero(leaf_active_)),
            }
        )


class CanonicalPatchLevel(StrictModule, NonTrainableState):
    """All execution buckets for one physical refinement level."""

    level: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    global_cell_shape: tuple[int, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    buckets: tuple[CanonicalPatchBucket, ...]
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: int,
        refinement_ratio: int,
        global_cell_shape: Sequence[int],
        spacing: Sequence[float],
        buckets: Sequence[CanonicalPatchBucket],
        /,
    ):
        level_ = int(level)
        ratio = int(refinement_ratio)
        shape = tuple(int(value) for value in global_cell_shape)
        spacing_ = tuple(float(value) for value in spacing)
        buckets_ = tuple(buckets)
        if (
            level_ < 0
            or ratio <= 1
            or not shape
            or any(value <= 0 for value in shape)
            or len(spacing_) != len(shape)
            or any(not np.isfinite(value) or value <= 0.0 for value in spacing_)
            or not buckets_
            or any(
                not isinstance(bucket, CanonicalPatchBucket)
                or len(bucket.plan.signature.envelope_shape) != len(shape)
                for bucket in buckets_
            )
        ):
            raise ValueError("Canonical patch level is invalid.")
        self.level = level_
        self.refinement_ratio = ratio
        self.global_cell_shape = shape
        self.spacing = spacing_
        self.buckets = buckets_
        self.level_id = canonical_fingerprint(
            {
                "kind": "canonical-amr-patch-level",
                "level": level_,
                "ratio": ratio,
                "shape": shape,
                "spacing": spacing_,
                "buckets": [bucket.bucket_id for bucket in buckets_],
            }
        )

    @property
    def active_patch_count(self) -> int:
        return sum(
            int(np.count_nonzero(np.asarray(bucket.active))) for bucket in self.buckets
        )

    @property
    def active_cell_count(self) -> int:
        return sum(
            int(np.count_nonzero(np.asarray(bucket.cell_active)))
            for bucket in self.buckets
        )

    @property
    def leaf_cell_count(self) -> int:
        return sum(
            int(np.count_nonzero(np.asarray(bucket.leaf_active)))
            for bucket in self.buckets
        )


class CanonicalPatchHierarchy(StrictModule, NonTrainableState):
    """One canonical downstream view of fixed and variable patch hierarchies."""

    topology: HierarchyTopology
    levels: tuple[CanonicalPatchLevel, ...]
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: HierarchyTopology,
        levels: Sequence[CanonicalPatchLevel],
        periodic_axes: Sequence[bool],
        layout_id: str,
        /,
    ):
        levels_ = tuple(levels)
        periodic = tuple(bool(value) for value in periodic_axes)
        if (
            not isinstance(
                topology, (BlockHierarchyTopology, VariablePatchHierarchyTopology)
            )
            or not levels_
            or tuple(level.level for level in levels_) != tuple(range(len(levels_)))
            or len(periodic) != len(levels_[0].global_cell_shape)
        ):
            raise ValueError(
                "Canonical hierarchy topology, levels, or periodicity is invalid."
            )
        self.topology = topology
        self.levels = levels_
        self.periodic_axes = periodic
        self.topology_id = topology.topology_id
        self.layout_id = str(layout_id)
        self.partition_id = topology.partition_id
        self.hierarchy_id = canonical_fingerprint(
            {
                "kind": "canonical-amr-patch-hierarchy",
                "topology": topology.topology_id,
                "layout": self.layout_id,
                "partition": topology.partition_id,
                "levels": [level.level_id for level in levels_],
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.periodic_axes)

    @property
    def geometry_id(self) -> str:
        return self.topology.epoch.geometry_id


class BlockAMRResourceEvidence(StrictModule, NonTrainableState):
    """Exact static-slot accounting for a canonical hierarchy preparation."""

    patch_slots: int = eqx.field(static=True)
    cell_slots: int = eqx.field(static=True)
    leaf_cells: int = eqx.field(static=True)
    control_volume_slots: int = eqx.field(static=True)
    face_slots: int = eqx.field(static=True)
    entity_slots: int = eqx.field(static=True)
    route_slots: int = eqx.field(static=True)
    reserved_host_bytes: int = eqx.field(static=True)
    reserved_device_bytes: int = eqx.field(static=True)
    host_byte_limit: int | None = eqx.field(static=True)
    device_byte_limit: int | None = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    message: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        patch_slots: int,
        cell_slots: int,
        leaf_cells: int,
        control_volume_slots: int,
        face_slots: int,
        entity_slots: int,
        route_slots: int,
        reserved_host_bytes: int,
        reserved_device_bytes: int,
        host_byte_limit: int | None,
        device_byte_limit: int | None,
    ):
        values = (
            patch_slots,
            cell_slots,
            leaf_cells,
            control_volume_slots,
            face_slots,
            entity_slots,
            route_slots,
            reserved_host_bytes,
            reserved_device_bytes,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError(
                "AMR resource counts and byte reservations must be nonnegative."
            )
        host_limit = None if host_byte_limit is None else int(host_byte_limit)
        device_limit = None if device_byte_limit is None else int(device_byte_limit)
        if (host_limit is not None and host_limit <= 0) or (
            device_limit is not None and device_limit <= 0
        ):
            raise ValueError("AMR resource byte limits must be positive when supplied.")
        host_ok = host_limit is None or int(reserved_host_bytes) <= host_limit
        device_ok = device_limit is None or int(reserved_device_bytes) <= device_limit
        valid = host_ok and device_ok
        message = (
            "AMR static resource reservation is admissible."
            if valid
            else "AMR static resource reservation exceeds its declared byte limit."
        )
        self.patch_slots = int(patch_slots)
        self.cell_slots = int(cell_slots)
        self.leaf_cells = int(leaf_cells)
        self.control_volume_slots = int(control_volume_slots)
        self.face_slots = int(face_slots)
        self.entity_slots = int(entity_slots)
        self.route_slots = int(route_slots)
        self.reserved_host_bytes = int(reserved_host_bytes)
        self.reserved_device_bytes = int(reserved_device_bytes)
        self.host_byte_limit = host_limit
        self.device_byte_limit = device_limit
        self.valid = valid
        self.message = message
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "block-amr-resource-evidence",
                "counts": values,
                "host_limit": host_limit,
                "device_limit": device_limit,
                "valid": valid,
            }
        )


class BlockAMRResourcePlan(StrictModule, NonTrainableState):
    """Finite resource envelope shared by geometry, numerics, and execution."""

    maximum_components_per_cell: int = eqx.field(static=True)
    maximum_apertures_per_face: int = eqx.field(static=True)
    maximum_embedded_faces_per_cell: int = eqx.field(static=True)
    maximum_mortars: int = eqx.field(static=True)
    maximum_redistribution_routes: int = eqx.field(static=True)
    maximum_topology_events: int = eqx.field(static=True)
    maximum_communication_peers: int = eqx.field(static=True)
    host_byte_limit: int | None = eqx.field(static=True)
    device_byte_limit: int | None = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_components_per_cell: int = 1,
        maximum_apertures_per_face: int = 1,
        maximum_embedded_faces_per_cell: int = 1,
        maximum_mortars: int = 1,
        maximum_redistribution_routes: int = 1,
        maximum_topology_events: int = 1,
        maximum_communication_peers: int = 1,
        host_byte_limit: int | None = None,
        device_byte_limit: int | None = None,
    ):
        capacities = tuple(
            int(value)
            for value in (
                maximum_components_per_cell,
                maximum_apertures_per_face,
                maximum_embedded_faces_per_cell,
                maximum_mortars,
                maximum_redistribution_routes,
                maximum_topology_events,
                maximum_communication_peers,
            )
        )
        if any(value <= 0 for value in capacities):
            raise ValueError("All block-AMR production capacities must be positive.")
        host_limit = None if host_byte_limit is None else int(host_byte_limit)
        device_limit = None if device_byte_limit is None else int(device_byte_limit)
        if (host_limit is not None and host_limit <= 0) or (
            device_limit is not None and device_limit <= 0
        ):
            raise ValueError("AMR resource byte limits must be positive when supplied.")
        (
            self.maximum_components_per_cell,
            self.maximum_apertures_per_face,
            self.maximum_embedded_faces_per_cell,
            self.maximum_mortars,
            self.maximum_redistribution_routes,
            self.maximum_topology_events,
            self.maximum_communication_peers,
        ) = capacities
        self.host_byte_limit = host_limit
        self.device_byte_limit = device_limit
        self.resource_id = canonical_fingerprint(
            {
                "kind": "block-amr-resource-plan",
                "capacities": capacities,
                "host_byte_limit": host_limit,
                "device_byte_limit": device_limit,
            }
        )

    def preflight(
        self,
        hierarchy: CanonicalPatchHierarchy,
        /,
        *,
        physical_component_count: int,
        dtype: np.dtype | type,
        entity_capacity: BlockHierarchyCapacityPlan | None = None,
        additional_host_bytes: int = 0,
        additional_device_bytes: int = 0,
    ) -> BlockAMRResourceEvidence:
        if not isinstance(hierarchy, CanonicalPatchHierarchy):
            raise TypeError("AMR resource preflight requires CanonicalPatchHierarchy.")
        components = int(physical_component_count)
        if components <= 0:
            raise ValueError("physical_component_count must be positive.")
        scalar_bytes = int(np.dtype(dtype).itemsize)
        patch_slots = sum(
            bucket.plan.lane_capacity
            for level in hierarchy.levels
            for bucket in level.buckets
        )
        cell_slots = sum(
            bucket.plan.lane_capacity * prod(bucket.plan.signature.envelope_shape)
            for level in hierarchy.levels
            for bucket in level.buckets
        )
        leaf_cells = sum(level.leaf_cell_count for level in hierarchy.levels)
        control_slots = cell_slots * self.maximum_components_per_cell
        face_slots = cell_slots * (
            2 * hierarchy.dimension * self.maximum_apertures_per_face
            + self.maximum_embedded_faces_per_cell
        )
        entity_slots = (
            0
            if entity_capacity is None
            else sum(sum(row) for row in entity_capacity.entity_capacities)
        )
        route_slots = (
            self.maximum_mortars
            + self.maximum_redistribution_routes
            + self.maximum_communication_peers
            + (0 if entity_capacity is None else sum(entity_capacity.route_capacities))
        )
        field_bytes = control_slots * components * scalar_bytes
        geometry_bytes = scalar_bytes * (
            control_slots * (hierarchy.dimension + 1)
            + face_slots * (3 * hierarchy.dimension + 2)
        )
        route_bytes = 4 * (5 * face_slots + 3 * route_slots)
        mask_bytes = control_slots + face_slots
        device_bytes = (
            field_bytes
            + geometry_bytes
            + route_bytes
            + mask_bytes
            + int(additional_device_bytes)
        )
        host_bytes = (
            2 * geometry_bytes
            + 2 * route_bytes
            + 32 * (patch_slots + entity_slots)
            + int(additional_host_bytes)
        )
        return BlockAMRResourceEvidence(
            patch_slots=patch_slots,
            cell_slots=cell_slots,
            leaf_cells=leaf_cells,
            control_volume_slots=control_slots,
            face_slots=face_slots,
            entity_slots=entity_slots,
            route_slots=route_slots,
            reserved_host_bytes=host_bytes,
            reserved_device_bytes=device_bytes,
            host_byte_limit=self.host_byte_limit,
            device_byte_limit=self.device_byte_limit,
        )


def _rectangular_cell_mask(
    active: np.ndarray,
    extents: np.ndarray,
    envelope: tuple[int, ...],
) -> np.ndarray:
    mask = np.zeros((active.size,) + envelope, dtype=bool)
    for lane in np.flatnonzero(active):
        extent = tuple(int(value) for value in extents[lane])
        mask[(int(lane),) + tuple(slice(0, value) for value in extent)] = True
    return mask


def _covered_by_fine_boxes(
    box: LogicalPatchBox,
    local: tuple[int, ...],
    ratio: int,
    fine_boxes: tuple[LogicalPatchBox, ...],
) -> bool:
    coarse = tuple(start + offset for start, offset in zip(box.lower, local, strict=True))
    fine_lower = tuple(value * ratio for value in coarse)
    return all(
        any(
            candidate.contains_cell(
                tuple(
                    start + offset
                    for start, offset in zip(fine_lower, child, strict=True)
                )
            )
            for candidate in fine_boxes
        )
        for child in np.ndindex((ratio,) * len(coarse))
    )


def _leaf_mask(
    cell_active: np.ndarray,
    boxes: tuple[LogicalPatchBox | None, ...],
    ratio: int,
    fine_boxes: tuple[LogicalPatchBox, ...],
) -> np.ndarray:
    leaf = cell_active.copy()
    if not fine_boxes:
        return leaf
    for lane, box in enumerate(boxes):
        if box is None:
            continue
        for local in np.ndindex(box.extent):
            if _covered_by_fine_boxes(box, local, ratio, fine_boxes):
                leaf[(lane,) + local] = False
    return leaf


def canonicalize_patch_hierarchy(
    topology: HierarchyTopology,
    /,
) -> CanonicalPatchHierarchy:
    """Lower fixed and variable hierarchy frontends to one bucketed view."""

    if isinstance(topology, BlockHierarchyTopology):
        levels: list[CanonicalPatchLevel] = []
        for level_index, (level_plan, metadata, boxes) in enumerate(
            zip(
                topology.plan.levels,
                topology.levels,
                topology.logical_boxes,
                strict=True,
            )
        ):
            signature = PatchShapeSignature(
                level_plan.block_shape,
                halo_width=level_plan.halo_width,
            )
            bucket_plan = PatchBucketPlan(signature, level_plan.maximum_blocks)
            active = np.asarray(metadata.active, dtype=bool)
            lower = np.zeros(
                (level_plan.maximum_blocks, len(level_plan.block_shape)), dtype=np.int32
            )
            extent = np.zeros_like(lower)
            routes = np.full((level_plan.maximum_blocks,), -1, dtype=np.int32)
            padded_boxes: list[LogicalPatchBox | None] = [
                None
            ] * level_plan.maximum_blocks
            for lane, box in enumerate(boxes):
                lower[lane] = box.lower
                extent[lane] = box.extent
                routes[lane] = lane
                padded_boxes[lane] = box
            cell_active = _rectangular_cell_mask(active, extent, signature.envelope_shape)
            leaf_active = cell_active & ~np.asarray(topology.covered_cells[level_index])
            bucket = CanonicalPatchBucket(
                bucket_plan,
                active=active,
                lower=lower,
                extent=extent,
                route_indices=routes,
                cell_active=cell_active,
                leaf_active=leaf_active,
                boxes=padded_boxes,
            )
            levels.append(
                CanonicalPatchLevel(
                    level_index,
                    level_plan.refinement_ratio,
                    topology.plan.global_cell_shapes[level_index],
                    topology.plan.level_spacings[level_index],
                    (bucket,),
                )
            )
        layout_id = canonical_fingerprint(
            {
                "kind": "fixed-block-canonical-layout",
                "metadata": [metadata.metadata_id for metadata in topology.levels],
            }
        )
        return CanonicalPatchHierarchy(
            topology,
            levels,
            topology.plan.periodic_axes,
            layout_id,
        )

    if not isinstance(topology, VariablePatchHierarchyTopology):
        raise TypeError("Unsupported block-AMR hierarchy topology.")
    levels = []
    for level_index, (level_plan, metadata) in enumerate(
        zip(topology.plan.levels, topology.levels, strict=True)
    ):
        fine_boxes = (
            ()
            if level_index + 1 == len(topology.levels)
            else topology.logical_boxes[level_index + 1]
        )
        buckets = []
        for bucket_index, bucket_plan in enumerate(level_plan.buckets):
            active = np.asarray(metadata.active[bucket_index], dtype=bool)
            lower = np.asarray(metadata.lower[bucket_index], dtype=np.int32)
            extent = np.asarray(metadata.extent[bucket_index], dtype=np.int32)
            routes = np.asarray(metadata.route_indices[bucket_index], dtype=np.int32)
            boxes = metadata.boxes[bucket_index]
            cell_active = _rectangular_cell_mask(
                active,
                extent,
                bucket_plan.signature.envelope_shape,
            )
            leaf_active = _leaf_mask(
                cell_active,
                boxes,
                level_plan.refinement_ratio,
                fine_boxes,
            )
            buckets.append(
                CanonicalPatchBucket(
                    bucket_plan,
                    active=active,
                    lower=lower,
                    extent=extent,
                    route_indices=routes,
                    cell_active=cell_active,
                    leaf_active=leaf_active,
                    boxes=boxes,
                )
            )
        levels.append(
            CanonicalPatchLevel(
                level_index,
                level_plan.refinement_ratio,
                topology.plan.global_cell_shapes[level_index],
                topology.plan.level_spacings[level_index],
                buckets,
            )
        )
    return CanonicalPatchHierarchy(
        topology,
        levels,
        topology.plan.periodic_axes,
        topology.layout_id,
    )


__all__ = [
    "BlockAMRResourceEvidence",
    "BlockAMRResourcePlan",
    "CanonicalPatchBucket",
    "CanonicalPatchHierarchy",
    "CanonicalPatchLevel",
    "HierarchyTopology",
    "canonicalize_patch_hierarchy",
]
