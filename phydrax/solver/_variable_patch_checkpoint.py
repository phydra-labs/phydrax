#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pickle-free canonical variable-patch state and explicit repartition archive."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import TopologyEpoch
from ..discretization.amr import (
    PreparedVariablePatchPartition,
    VariablePatchFieldState,
    VariablePatchHierarchyState,
    VariablePatchHierarchyTopology,
    VariablePatchLevelMetadata,
)


class VariablePatchCheckpointPlan(StrictModule, NonTrainableState):
    topology: VariablePatchHierarchyTopology
    component_names: tuple[str, ...]
    dtype: Any = jnp.float64
    partition: PreparedVariablePatchPartition | None
    checkpoint_id: str

    def __init__(
        self,
        topology: VariablePatchHierarchyTopology,
        component_names: Sequence[str],
        /,
        *,
        dtype=jnp.float64,
        partition: PreparedVariablePatchPartition | None = None,
    ):
        names = tuple(str(value) for value in component_names)
        dtype_ = jnp.dtype(dtype)
        if (
            not isinstance(topology, VariablePatchHierarchyTopology)
            or not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
            or not jnp.issubdtype(dtype_, jnp.inexact)
            or (
                partition is not None
                and (
                    not isinstance(partition, PreparedVariablePatchPartition)
                    or partition.topology.epoch.epoch_id != topology.epoch.epoch_id
                )
            )
        ):
            raise ValueError("Variable patch checkpoint plan is invalid.")
        self.topology = topology
        self.component_names = names
        self.dtype = dtype_
        self.partition = partition
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "variable-patch-checkpoint-plan",
                "topology": topology.topology_id,
                "layout": topology.layout_id,
                "components": names,
                "dtype": str(dtype_),
                "partition": None if partition is None else partition.partition_id,
            }
        )


class VariablePatchCheckpoint(StrictModule, NonTrainableState):
    state: VariablePatchHierarchyState
    manifest: dict[str, Any]
    payload_id: str


def _arrays(state: VariablePatchHierarchyState, dtype, /) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for level, (metadata, field) in enumerate(
        zip(state.topology.levels, state.levels, strict=True)
    ):
        for bucket, (values, active, lower, extent, routes) in enumerate(
            zip(
                field.safe_values(),
                metadata.active,
                metadata.lower,
                metadata.extent,
                metadata.route_indices,
                strict=True,
            )
        ):
            prefix = f"levels/{level:04d}/buckets/{bucket:04d}"
            arrays[f"{prefix}/values"] = np.asarray(values, dtype=np.dtype(dtype))
            arrays[f"{prefix}/active"] = np.asarray(active, dtype=np.bool_)
            arrays[f"{prefix}/lower"] = np.asarray(lower, dtype=np.int32)
            arrays[f"{prefix}/extent"] = np.asarray(extent, dtype=np.int32)
            arrays[f"{prefix}/route_indices"] = np.asarray(routes, dtype=np.int32)
    return arrays


def _payload_id(manifest: dict[str, Any], arrays: dict[str, np.ndarray], /) -> str:
    return canonical_fingerprint(
        {
            "manifest": manifest,
            "arrays": {
                name: array_tree_fingerprint(value)
                for name, value in sorted(arrays.items())
            },
        }
    )


def write_variable_patch_checkpoint(
    path: str | Path,
    plan: VariablePatchCheckpointPlan,
    state: VariablePatchHierarchyState,
    /,
) -> VariablePatchCheckpoint:
    if (
        not isinstance(plan, VariablePatchCheckpointPlan)
        or not isinstance(state, VariablePatchHierarchyState)
        or state.topology.epoch.epoch_id != plan.topology.epoch.epoch_id
    ):
        raise ValueError("Variable patch checkpoint state does not match plan epoch.")
    arrays = _arrays(state, plan.dtype)
    record = {
        "archive_kind": "variable-patch-checkpoint",
        "checkpoint_id": plan.checkpoint_id,
        "epoch": state.topology.epoch.to_archive_record(),
        "topology_id": state.topology.topology_id,
        "layout_id": state.topology.layout_id,
        "component_names": list(plan.component_names),
        "dtype": str(np.dtype(plan.dtype)),
        "partition_id": None if plan.partition is None else plan.partition.partition_id,
    }
    payload = _payload_id(record, arrays)
    manifest = {**record, "payload_id": payload}
    write_array_archive(path, manifest=manifest, arrays=arrays)
    return VariablePatchCheckpoint(state, manifest, payload)


def read_variable_patch_checkpoint(
    path: str | Path,
    plan: VariablePatchCheckpointPlan,
    /,
) -> VariablePatchCheckpoint:
    if not isinstance(plan, VariablePatchCheckpointPlan):
        raise TypeError("plan must be VariablePatchCheckpointPlan.")
    manifest, arrays = read_array_archive(path)
    record = {
        key: value
        for key, value in manifest.items()
        if key not in ("payload_id", "arrays")
    }
    if (
        manifest.get("archive_kind") != "variable-patch-checkpoint"
        or manifest.get("checkpoint_id") != plan.checkpoint_id
        or manifest.get("topology_id") != plan.topology.topology_id
        or manifest.get("layout_id") != plan.topology.layout_id
        or manifest.get("component_names") != list(plan.component_names)
        or manifest.get("dtype") != str(np.dtype(plan.dtype))
        or manifest.get("partition_id")
        != (None if plan.partition is None else plan.partition.partition_id)
        or _payload_id(record, arrays) != manifest.get("payload_id")
    ):
        raise ValueError("Variable patch checkpoint is incompatible or corrupted.")
    epoch = TopologyEpoch.from_archive_record(manifest["epoch"])
    metadata = []
    fields = []
    for level, level_plan in enumerate(plan.topology.plan.levels):
        groups = []
        value_arrays = []
        for bucket_index, bucket in enumerate(level_plan.buckets):
            prefix = f"levels/{level:04d}/buckets/{bucket_index:04d}"
            active = np.asarray(arrays[f"{prefix}/active"], dtype=np.bool_)
            lower = np.asarray(arrays[f"{prefix}/lower"], dtype=np.int32)
            extent = np.asarray(arrays[f"{prefix}/extent"], dtype=np.int32)
            routes = np.asarray(arrays[f"{prefix}/route_indices"], dtype=np.int32)
            boxes = tuple(
                plan.topology.levels[level].boxes[bucket_index][lane]
                for lane in np.flatnonzero(active)
            )
            if any(box is None for box in boxes):
                raise ValueError(
                    "Variable patch checkpoint active lane lost its logical box."
                )
            groups.append(boxes)
            expected_metadata = plan.topology.levels[level]
            if (
                not np.array_equal(lower, expected_metadata.lower[bucket_index])
                or not np.array_equal(extent, expected_metadata.extent[bucket_index])
                or not np.array_equal(
                    routes, expected_metadata.route_indices[bucket_index]
                )
            ):
                raise ValueError("Variable patch checkpoint bucket metadata changed.")
            values = jnp.asarray(arrays[f"{prefix}/values"], dtype=plan.dtype)
            expected_prefix = (bucket.lane_capacity,) + bucket.signature.envelope_shape
            if values.shape[: len(expected_prefix)] != expected_prefix:
                raise ValueError("Variable patch checkpoint field shape changed.")
            value_arrays.append(values)
        level_metadata = VariablePatchLevelMetadata(level_plan, groups)
        metadata.append(level_metadata)
        fields.append(VariablePatchFieldState(level_metadata, value_arrays))
    topology = VariablePatchHierarchyTopology(plan.topology.plan, metadata, epoch=epoch)
    state = VariablePatchHierarchyState(topology, fields)
    return VariablePatchCheckpoint(state, manifest, manifest["payload_id"])


__all__ = [
    "VariablePatchCheckpoint",
    "VariablePatchCheckpointPlan",
    "read_variable_patch_checkpoint",
    "write_variable_patch_checkpoint",
]
