#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Portable restart and visualization snapshots for multivalued block AMR."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import TensorGridPlan, TopologyEpoch, UniformCellAxisSpec
from ..discretization.amr import (
    BlockAMRResourcePlan,
    EmbeddedLevelSetBody,
    EmbeddedLevelSetBodySet,
    LogicalPatchBox,
    MultivaluedCutCellPlan,
    PatchBucketPlan,
    PatchCoordinateMapSet,
    PatchShapeSignature,
    VariablePatchHierarchyPlan,
    VariablePatchHierarchyTopology,
    VariablePatchLevelMetadata,
    VariablePatchLevelPlan,
)
from ._moving_cut_cell import MovingCutCellState


CoordinateMap = Callable[[Any, Any, Any], Any]
LevelSet = Callable[[Any, Any, Any], Any]


class CutCellRestartRegistry(StrictModule, NonTrainableState):
    """Explicit callable registry used to reconstruct persisted geometry providers."""

    coordinate_map_ids: tuple[str, ...] = eqx.field(static=True)
    coordinate_maps: tuple[CoordinateMap | PatchCoordinateMapSet, ...] = eqx.field(
        static=True
    )
    level_set_ids: tuple[str, ...] = eqx.field(static=True)
    level_sets: tuple[LevelSet, ...] = eqx.field(static=True)
    registry_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate_maps: Mapping[str, CoordinateMap | PatchCoordinateMapSet],
        level_sets: Mapping[str, LevelSet],
        /,
    ):
        map_entries = tuple(
            sorted((str(key), value) for key, value in coordinate_maps.items())
        )
        level_entries = tuple(
            sorted((str(key), value) for key, value in level_sets.items())
        )
        if (
            not map_entries
            or not level_entries
            or any(
                not key
                or (not callable(value) and not isinstance(value, PatchCoordinateMapSet))
                for key, value in map_entries
            )
            or any(not key or not callable(value) for key, value in level_entries)
        ):
            raise ValueError("Cut-cell restart registry entries must be named callables.")
        self.coordinate_map_ids = tuple(key for key, _ in map_entries)
        self.coordinate_maps = tuple(value for _, value in map_entries)
        self.level_set_ids = tuple(key for key, _ in level_entries)
        self.level_sets = tuple(value for _, value in level_entries)
        self.registry_id = canonical_fingerprint(
            {
                "kind": "cut-cell-restart-registry",
                "coordinate_maps": self.coordinate_map_ids,
                "level_sets": self.level_set_ids,
            }
        )

    def coordinate_map(self, identifier: str, /) -> CoordinateMap | PatchCoordinateMapSet:
        if identifier not in self.coordinate_map_ids:
            raise ValueError(f"Checkpoint coordinate map {identifier!r} is unavailable.")
        return self.coordinate_maps[self.coordinate_map_ids.index(identifier)]

    def level_set(self, identifier: str, /) -> LevelSet:
        if identifier not in self.level_set_ids:
            raise ValueError(f"Checkpoint level set {identifier!r} is unavailable.")
        return self.level_sets[self.level_set_ids.index(identifier)]


class MultivaluedBlockAMRCheckpointPlan(StrictModule, NonTrainableState):
    """Canonical archive policy independent of device/process partition."""

    cut_plan: MultivaluedCutCellPlan
    component_names: tuple[str, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        cut_plan: MultivaluedCutCellPlan,
        component_names: Sequence[str],
        /,
        *,
        dtype: str | np.dtype = np.float64,
    ):
        names = tuple(str(value) for value in component_names)
        dtype_ = np.dtype(dtype)
        if (
            not isinstance(cut_plan, MultivaluedCutCellPlan)
            or not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
            or dtype_.kind not in "fc"
        ):
            raise ValueError("Multivalued block-AMR checkpoint plan is invalid.")
        self.cut_plan = cut_plan
        self.component_names = names
        self.dtype = dtype_.name
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "multivalued-block-amr-checkpoint-plan",
                "cut_plan": cut_plan.plan_id,
                "component_names": names,
                "dtype": dtype_.name,
            }
        )


class MultivaluedBlockAMRCheckpoint(StrictModule, NonTrainableState):
    """Restored moving state, geometry plan, manifest, and payload identity."""

    state: MovingCutCellState
    cut_plan: MultivaluedCutCellPlan
    manifest: dict[str, Any] = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


class CutCellOutputSnapshot(StrictModule, NonTrainableState):
    """Partition-independent polyhedral output manifest and payload identity."""

    manifest: dict[str, Any] = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


def _hierarchy_record(cut_plan: MultivaluedCutCellPlan, /) -> dict[str, Any]:
    hierarchy = cut_plan.hierarchy
    topology = hierarchy.topology
    grid = topology.plan.grid
    levels = []
    for level in hierarchy.levels:
        levels.append(
            {
                "level": level.level,
                "refinement_ratio": level.refinement_ratio,
                "buckets": [
                    {
                        "envelope_shape": list(bucket.plan.signature.envelope_shape),
                        "halo_width": list(bucket.plan.signature.halo_width),
                        "alignment": list(bucket.plan.signature.alignment),
                        "lane_capacity": bucket.plan.lane_capacity,
                        "boxes": [
                            None
                            if box is None
                            else {
                                "lower": list(box.lower),
                                "upper": list(box.upper),
                            }
                            for box in bucket.boxes
                        ],
                    }
                    for bucket in level.buckets
                ],
            }
        )
    resources = cut_plan.resources
    return {
        "axis_names": list(grid.axis_names),
        "shape": list(grid.shape),
        "bounds": [
            [float(axis.bounds[0]), float(axis.bounds[1])]
            for axis in grid.structured_axes
        ],
        "periodic_axes": list(hierarchy.periodic_axes),
        "levels": levels,
        "epoch_index": topology.epoch.index,
        "resources": {
            "maximum_components_per_cell": resources.maximum_components_per_cell,
            "maximum_apertures_per_face": resources.maximum_apertures_per_face,
            "maximum_embedded_faces_per_cell": resources.maximum_embedded_faces_per_cell,
            "maximum_mortars": resources.maximum_mortars,
            "maximum_redistribution_routes": resources.maximum_redistribution_routes,
            "maximum_topology_events": resources.maximum_topology_events,
            "maximum_communication_peers": resources.maximum_communication_peers,
            "host_byte_limit": resources.host_byte_limit,
            "device_byte_limit": resources.device_byte_limit,
        },
    }


def _cut_plan_record(plan: MultivaluedBlockAMRCheckpointPlan, /) -> dict[str, Any]:
    cut = plan.cut_plan
    return {
        "checkpoint_id": plan.checkpoint_id,
        "hierarchy": _hierarchy_record(cut),
        "coordinate_map_id": cut.coordinate_map_id,
        "body_set_id": cut.bodies.body_set_id,
        "body_operation": cut.bodies.operation,
        "body_signs": list(cut.bodies.body_signs),
        "bodies": [
            {"field_id": body.field_id, "body_tag": body.body_tag}
            for body in cut.bodies.bodies
        ],
        "subdivision": cut.subdivision,
        "predicate_tolerance": cut.predicate_tolerance,
        "component_names": list(plan.component_names),
        "dtype": plan.dtype,
    }


def _state_arrays(state: MovingCutCellState, dtype: np.dtype, /) -> dict[str, np.ndarray]:
    complex_ = state.complex
    connectivity = complex_.mesh.connectivity
    return {
        "content": np.asarray(state.content, dtype=dtype),
        "component_active": np.asarray(complex_.component_active, dtype=np.bool_),
        "component_levels": np.asarray(complex_.component_levels, dtype=np.int32),
        "component_cell_coordinates": np.asarray(
            complex_.component_cell_coordinates, dtype=np.int32
        ),
        "component_slots": np.asarray(complex_.component_slots, dtype=np.int32),
        "component_volumes": np.asarray(complex_.component_volumes),
        "component_centers": np.asarray(complex_.component_centers),
        "component_volume_fractions": np.asarray(complex_.component_volume_fractions),
        "face_active": np.asarray(complex_.face_active, dtype=np.bool_),
        "face_owner_components": np.asarray(
            complex_.face_owner_components, dtype=np.int32
        ),
        "face_neighbor_components": np.asarray(
            complex_.face_neighbor_components, dtype=np.int32
        ),
        "face_kinds": np.asarray(complex_.face_kinds, dtype=np.int32),
        "face_body_tags": np.asarray(complex_.face_body_tags, dtype=np.int32),
        "face_centers": np.asarray(complex_.face_centers),
        "face_area_vectors": np.asarray(complex_.face_area_vectors),
        "mesh_coordinates": np.asarray(complex_.mesh.coordinates),
        "mesh_face_vertex_offsets": np.asarray(
            connectivity.face_vertex_offsets, dtype=np.int32
        ),
        "mesh_face_vertex_values": np.asarray(
            connectivity.face_vertex_values, dtype=np.int32
        ),
        "mesh_cell_face_offsets": np.asarray(
            connectivity.cell_face_offsets, dtype=np.int32
        ),
        "mesh_cell_face_values": np.asarray(
            connectivity.cell_face_values, dtype=np.int32
        ),
    }


def _payload_id(record: dict[str, Any], arrays: Mapping[str, np.ndarray], /) -> str:
    return canonical_fingerprint(
        {
            "manifest": record,
            "arrays": {
                name: array_tree_fingerprint(value)
                for name, value in sorted(arrays.items())
            },
        }
    )


def write_multivalued_block_amr_checkpoint(
    path: str | Path,
    plan: MultivaluedBlockAMRCheckpointPlan,
    state: MovingCutCellState,
    /,
) -> MultivaluedBlockAMRCheckpoint:
    if not isinstance(plan, MultivaluedBlockAMRCheckpointPlan) or not isinstance(
        state, MovingCutCellState
    ):
        raise TypeError("Multivalued checkpoint write requires plan and moving state.")
    if state.complex.hierarchy.hierarchy_id != plan.cut_plan.hierarchy.hierarchy_id:
        raise ValueError("Checkpoint state belongs to another hierarchy plan.")
    arrays = _state_arrays(state, np.dtype(plan.dtype))
    record = {
        "archive_kind": "multivalued-block-amr-checkpoint",
        "plan": _cut_plan_record(plan),
        "time": float(state.time),
        "revision": int(state.revision),
        "topology_id": state.complex.topology_id,
        "geometry_id": state.complex.geometry_id,
        "component_shape": list(state.content.shape[1:]),
    }
    payload = _payload_id(record, arrays)
    manifest = {**record, "payload_id": payload}
    write_array_archive(path, manifest=manifest, arrays=arrays)
    return MultivaluedBlockAMRCheckpoint(state, plan.cut_plan, manifest, payload)


def _restore_cut_plan(
    record: Mapping[str, Any],
    registry: CutCellRestartRegistry,
    /,
) -> MultivaluedCutCellPlan:
    hierarchy_record = record["hierarchy"]
    shape = tuple(hierarchy_record["shape"])
    bounds_by_axis = tuple(
        tuple(float(value) for value in pair) for pair in hierarchy_record["bounds"]
    )
    grid = TensorGridPlan(
        tuple(
            UniformCellAxisSpec(count, periodic=bool(periodic))
            for count, periodic in zip(
                shape, hierarchy_record["periodic_axes"], strict=True
            )
        ),
        axis_names=tuple(str(value) for value in hierarchy_record["axis_names"]),
    ).prepare(
        jnp.asarray(
            [
                [pair[0] for pair in bounds_by_axis],
                [pair[1] for pair in bounds_by_axis],
            ]
        )
    )
    level_plans = []
    metadata_groups = []
    for level_record in hierarchy_record["levels"]:
        buckets = []
        groups = []
        for bucket_record in level_record["buckets"]:
            signature = PatchShapeSignature(
                tuple(bucket_record["envelope_shape"]),
                halo_width=tuple(bucket_record["halo_width"]),
                alignment=tuple(bucket_record["alignment"]),
            )
            buckets.append(
                PatchBucketPlan(signature, int(bucket_record["lane_capacity"]))
            )
            groups.append(
                tuple(
                    LogicalPatchBox(
                        int(level_record["level"]),
                        tuple(box["lower"]),
                        tuple(box["upper"]),
                    )
                    for box in bucket_record["boxes"]
                    if box is not None
                )
            )
        level_plans.append(
            VariablePatchLevelPlan(
                int(level_record["level"]),
                buckets,
                refinement_ratio=int(level_record["refinement_ratio"]),
            )
        )
        metadata_groups.append(tuple(groups))
    base_boxes = tuple(box for group in metadata_groups[0] for box in group)
    hierarchy_plan = VariablePatchHierarchyPlan(grid, level_plans, base_boxes)
    metadata = tuple(
        VariablePatchLevelMetadata(level, groups)
        for level, groups in zip(level_plans, metadata_groups, strict=True)
    )
    candidate = VariablePatchHierarchyTopology(hierarchy_plan, metadata)
    topology = VariablePatchHierarchyTopology(
        hierarchy_plan,
        metadata,
        epoch=TopologyEpoch(
            int(hierarchy_record["epoch_index"]),
            hierarchy_plan.geometry_id,
            candidate.topology_id,
            candidate.partition_id,
        ),
    )
    resource_record = hierarchy_record["resources"]
    resources = BlockAMRResourcePlan(**resource_record)
    bodies = EmbeddedLevelSetBodySet(
        tuple(
            EmbeddedLevelSetBody(
                registry.level_set(str(body["field_id"])),
                str(body["field_id"]),
                int(body["body_tag"]),
            )
            for body in record["bodies"]
        ),
        operation=str(record["body_operation"]),
        body_signs=tuple(record["body_signs"]),
    )
    if bodies.body_set_id != record["body_set_id"]:
        raise ValueError("Restart body registry does not reproduce body identity.")
    map_id = str(record["coordinate_map_id"])
    return MultivaluedCutCellPlan(
        topology,
        registry.coordinate_map(map_id),
        map_id,
        bodies,
        resources,
        subdivision=int(record["subdivision"]),
        predicate_tolerance=float(record["predicate_tolerance"]),
    )


def read_multivalued_block_amr_checkpoint(
    path: str | Path,
    registry: CutCellRestartRegistry,
    /,
    *,
    args: Any = None,
) -> MultivaluedBlockAMRCheckpoint:
    if not isinstance(registry, CutCellRestartRegistry):
        raise TypeError("Checkpoint restore requires CutCellRestartRegistry.")
    manifest, arrays = read_array_archive(path)
    record = {
        key: value
        for key, value in manifest.items()
        if key not in ("payload_id", "arrays")
    }
    if manifest.get("archive_kind") != "multivalued-block-amr-checkpoint" or _payload_id(
        record, arrays
    ) != manifest.get("payload_id"):
        raise ValueError("Multivalued block-AMR checkpoint is corrupted.")
    cut_plan = _restore_cut_plan(manifest["plan"], registry)
    complex_ = cut_plan.prepare(manifest["time"], args)
    expected = _state_arrays(
        MovingCutCellState(
            complex_,
            np.zeros(
                (complex_.component_capacity,) + tuple(manifest["component_shape"]),
                dtype=np.dtype(manifest["plan"]["dtype"]),
            ),
            manifest["time"],
            manifest["revision"],
        ),
        np.dtype(manifest["plan"]["dtype"]),
    )
    for name, value in expected.items():
        if name == "content":
            continue
        if name not in arrays or array_tree_fingerprint(value) != array_tree_fingerprint(
            arrays[name]
        ):
            raise ValueError(f"Restart geometry payload {name!r} is incompatible.")
    content = jnp.asarray(arrays["content"], dtype=np.dtype(manifest["plan"]["dtype"]))
    state = MovingCutCellState(
        complex_,
        content,
        manifest["time"],
        manifest["revision"],
    )
    if (
        complex_.topology_id != manifest["topology_id"]
        or complex_.geometry_id != manifest["geometry_id"]
    ):
        raise ValueError("Restarted cut topology or geometry identity changed.")
    return MultivaluedBlockAMRCheckpoint(
        state,
        cut_plan,
        manifest,
        manifest["payload_id"],
    )


def write_multivalued_cut_cell_output(
    path: str | Path,
    state: MovingCutCellState,
    component_names: Sequence[str],
    /,
) -> CutCellOutputSnapshot:
    if not isinstance(state, MovingCutCellState):
        raise TypeError("Cut-cell output requires MovingCutCellState.")
    names = tuple(str(value) for value in component_names)
    if (
        len(names) != (1 if state.content.ndim == 1 else state.content.shape[-1])
        or any(not value for value in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Cut-cell output component names do not match state.")
    arrays = _state_arrays(state, np.asarray(state.content).dtype)
    arrays["cell_average"] = np.asarray(state.cell_average())
    record = {
        "archive_kind": "multivalued-block-amr-output",
        "time": float(state.time),
        "revision": int(state.revision),
        "topology_id": state.complex.topology_id,
        "geometry_id": state.complex.geometry_id,
        "component_names": list(names),
        "body_set_id": state.complex.body_set_id,
    }
    payload = _payload_id(record, arrays)
    manifest = {**record, "payload_id": payload}
    write_array_archive(path, manifest=manifest, arrays=arrays)
    return CutCellOutputSnapshot(manifest, payload)


__all__ = [
    "CutCellOutputSnapshot",
    "CutCellRestartRegistry",
    "MultivaluedBlockAMRCheckpoint",
    "MultivaluedBlockAMRCheckpointPlan",
    "read_multivalued_block_amr_checkpoint",
    "write_multivalued_block_amr_checkpoint",
    "write_multivalued_cut_cell_output",
]
