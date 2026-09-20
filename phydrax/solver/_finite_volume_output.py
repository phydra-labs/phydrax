#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import html
import json
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._publication import publish_bytes
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.amr import PreparedDistributedBlockAMRHierarchy
from ..discretization.finite_volume import (
    FiniteVolumeDiscretization,
    FiniteVolumePrecisionPolicy,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeGeometryState,
)
from ._block_amr_runtime import BlockAMRRuntimeState, PreparedBlockAMRRuntime
from ._finite_volume_runtime import FiniteVolumeRuntimeState


if TYPE_CHECKING:
    from ..discretization.finite_volume import ShallowWaterObservables


OutputDiscretization = (
    FiniteVolumeDiscretization
    | UnstructuredFiniteVolumeDiscretization
    | PreparedBlockAMRRuntime
)


def _h5py():
    if find_spec("h5py") is None:
        raise ImportError(
            "Finite-volume HDF5 output requires the optional 'h5py' package."
        )
    return import_module("h5py")


def _atomic_text(path: Path, payload: str, /) -> None:
    publish_bytes(
        path,
        payload.encode("utf-8"),
        maximum_bytes=256 * 1024 * 1024,
        mode="atomic_replace",
    )


def _accepted_points(
    discretization: OutputDiscretization,
    runtime_state: FiniteVolumeRuntimeState,
    accepted_geometry: UnstructuredFiniteVolumeGeometryState | ArrayLike | None,
    /,
) -> np.ndarray | None:
    if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
        if accepted_geometry is not None:
            raise TypeError("Structured output does not accept unstructured geometry.")
        return None
    content = runtime_state.content_state
    version = int(np.asarray(content.geometry_version))
    if accepted_geometry is None:
        if version != 0:
            raise ValueError(
                "Moving finite-volume output requires accepted geometry points."
            )
        points = np.asarray(discretization.vertices)
    elif isinstance(accepted_geometry, UnstructuredFiniteVolumeGeometryState):
        if (
            accepted_geometry.topology_id != discretization.topology_id
            or accepted_geometry.geometry_layout_id != content.geometry_layout_id
            or int(np.asarray(accepted_geometry.geometry_version)) != version
            or not np.array_equal(
                np.asarray(accepted_geometry.time),
                np.asarray(content.time),
            )
        ):
            raise ValueError("Accepted output geometry is stale for the runtime content.")
        points = np.asarray(accepted_geometry.vertices)
    else:
        points = np.asarray(accepted_geometry)
    expected_shape = tuple(discretization.vertices.shape)
    if (
        points.shape != expected_shape
        or points.dtype.kind not in "fc"
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "Accepted output geometry points must be finite with the mesh vertex shape."
        )
    return points


class FiniteVolumeOutputPlan(StrictModule, NonTrainableState):
    """Host-side HDF5/XDMF time series and meshio VTK snapshots."""

    hdf5_path: str = eqx.field(static=True)
    xdmf_path: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    geometry_kind: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    precision: FiniteVolumePrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope
    output_id: str = eqx.field(static=True)
    route_ids_json: str = eqx.field(static=True)
    distributed_partition_json: str = eqx.field(static=True)

    def __init__(
        self,
        path: str | Path,
        discretization: OutputDiscretization,
        /,
        *,
        precision: FiniteVolumePrecisionPolicy | None = None,
        partition: PreparedDistributedBlockAMRHierarchy | None = None,
    ):
        if isinstance(discretization, PreparedBlockAMRRuntime):
            block_runtime = discretization
            topology = block_runtime.dynamics.topology
            if partition is not None and (
                not isinstance(partition, PreparedDistributedBlockAMRHierarchy)
                or partition.topology.epoch.epoch_id != topology.epoch.epoch_id
                or partition.fd_hierarchy.prepared_id
                != block_runtime.plan.finite_volume.hierarchy.prepared_id
            ):
                raise ValueError(
                    "Output partition identity must match the prepared block runtime."
                )
            precision_ = (
                block_runtime.dynamics.plan.precision if precision is None else precision
            )
            if (
                not isinstance(precision_, FiniteVolumePrecisionPolicy)
                or precision_.policy_id != block_runtime.dynamics.plan.precision.policy_id
            ):
                raise ValueError(
                    "Block output must use the prepared runtime precision policy."
                )
            geometry_kind = "block_amr"
            discretization_id = block_runtime.prepared_id
            topology_id = topology.topology_id
            geometry_id = topology.epoch.geometry_id
            component_names = tuple(block_runtime.dynamics.plan.system.component_names)
            route_ids = {
                "fill_patch_plan_ids": [
                    fill.plan_id for fill in block_runtime.dynamics.fill_patch_plans
                ],
                "face_route_ids": [
                    route.block_id for route in block_runtime.dynamics.face_routes
                ],
                "edge_route_ids": [
                    route.route_plan_id for route in block_runtime.edge_routes
                ],
                "conservation_plan_id": block_runtime.conservation.plan_id,
                "topology_artifacts_id": block_runtime.topology_artifacts.artifacts_id,
            }
            partition_record = (
                None if partition is None else partition.manifest_compatibility_data()
            )
        else:
            if not isinstance(
                discretization,
                (FiniteVolumeDiscretization, UnstructuredFiniteVolumeDiscretization),
            ):
                raise TypeError(
                    "Output requires structured, unstructured, or prepared block finite-volume geometry."
                )
            if partition is not None:
                raise ValueError(
                    "Ordinary finite-volume output does not take a block partition."
                )
            precision_ = (
                FiniteVolumePrecisionPolicy(
                    np.asarray(discretization.cell_volumes).dtype.name
                )
                if precision is None
                else precision
            )
            if not isinstance(precision_, FiniteVolumePrecisionPolicy):
                raise TypeError("precision must be a FiniteVolumePrecisionPolicy.")
            if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
                geometry_kind = "unstructured"
                topology_id = discretization.topology_id
                geometry_id = discretization.geometry_id
            else:
                geometry_kind = "structured"
                topology_id = discretization.prepared_id
                geometry_id = discretization.prepared_id
            discretization_id = discretization.prepared_id
            component_names = discretization.component_names
            route_ids = {}
            partition_record = None
        target = Path(path)
        hdf5_path = target if target.suffix == ".h5" else target.with_suffix(".h5")
        xdmf_path = hdf5_path.with_suffix(".xdmf")
        self.hdf5_path = str(hdf5_path)
        self.xdmf_path = str(xdmf_path)
        self.discretization_id = discretization_id
        self.geometry_kind = geometry_kind
        self.topology_id = topology_id
        self.geometry_id = geometry_id
        self.component_names = component_names
        self.precision = precision_
        self.precision_evidence = precision_.evidence()
        self.route_ids_json = json.dumps(route_ids, sort_keys=True, separators=(",", ":"))
        self.distributed_partition_json = json.dumps(
            partition_record, sort_keys=True, separators=(",", ":")
        )
        self.output_id = canonical_fingerprint(
            {
                "kind": "finite-volume-output",
                "hdf5": str(hdf5_path),
                "geometry_kind": geometry_kind,
                "discretization": discretization_id,
                "topology": topology_id,
                "geometry": geometry_id,
                "components": list(component_names),
                "precision": precision_.policy_id,
                "precision_evidence": self.precision_evidence.evidence_id,
                "routes": route_ids,
                "distributed_partition": partition_record,
            }
        )

    def _validate_discretization(self, discretization: OutputDiscretization, /) -> None:
        if self.geometry_kind == "block_amr":
            if not isinstance(discretization, PreparedBlockAMRRuntime):
                raise TypeError("Block output requires PreparedBlockAMRRuntime.")
            topology = discretization.dynamics.topology
            if (
                discretization.prepared_id != self.discretization_id
                or topology.topology_id != self.topology_id
                or topology.epoch.geometry_id != self.geometry_id
                or json.dumps(
                    {
                        "fill_patch_plan_ids": [
                            fill.plan_id
                            for fill in discretization.dynamics.fill_patch_plans
                        ],
                        "face_route_ids": [
                            route.block_id
                            for route in discretization.dynamics.face_routes
                        ],
                        "edge_route_ids": [
                            route.route_plan_id for route in discretization.edge_routes
                        ],
                        "conservation_plan_id": discretization.conservation.plan_id,
                        "topology_artifacts_id": (
                            discretization.topology_artifacts.artifacts_id
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                != self.route_ids_json
            ):
                raise ValueError("Block output topology or route identity changed.")
            return
        if isinstance(discretization, PreparedBlockAMRRuntime):
            raise TypeError("Ordinary output does not accept a block runtime.")
        if discretization.prepared_id != self.discretization_id:
            raise ValueError("Output discretization identity changed.")
        if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            if (
                discretization.topology_id != self.topology_id
                or discretization.geometry_id != self.geometry_id
            ):
                raise ValueError("Output unstructured mesh identity changed.")

    def _initialize_block_geometry(
        self,
        handle: Any,
        runtime: PreparedBlockAMRRuntime,
        /,
    ) -> None:
        topology = runtime.dynamics.topology
        hierarchy = topology.plan
        block_group = handle.create_group("block_hierarchy")
        block_group.attrs["hierarchy_plan_id"] = hierarchy.plan_id
        block_group.attrs["topology_epoch_id"] = topology.epoch.epoch_id
        block_group.attrs["topology_epoch_index"] = topology.epoch.index
        block_group.attrs["canonical_partition_id"] = topology.partition_id
        block_group.attrs["topology_artifacts_id"] = (
            runtime.topology_artifacts.artifacts_id
        )
        block_group.attrs["route_ids_json"] = self.route_ids_json
        block_group.attrs["distributed_partition_json"] = self.distributed_partition_json
        levels = block_group.create_group("levels")
        lower = np.asarray(
            [float(np.asarray(axis.bounds)[0]) for axis in hierarchy.grid.structured_axes]
        )
        for level, (
            level_plan,
            metadata,
            covered_cells,
            interfaces,
            spacing,
        ) in enumerate(
            zip(
                hierarchy.levels,
                topology.levels,
                topology.covered_cells,
                topology.interfaces,
                hierarchy.level_spacings,
                strict=True,
            )
        ):
            group = levels.create_group(f"{level:04d}")
            group.attrs["level"] = level
            group.attrs["level_plan_id"] = level_plan.plan_id
            group.attrs["metadata_id"] = metadata.metadata_id
            group.attrs["covered_cells_id"] = array_tree_fingerprint(
                np.asarray(covered_cells, dtype=np.bool_)
            )["sha256"]
            group.attrs["interface_id"] = array_tree_fingerprint(
                np.asarray(interfaces, dtype=np.bool_)
            )["sha256"]
            group.attrs["maximum_blocks"] = level_plan.maximum_blocks
            group.attrs["block_shape"] = np.asarray(
                level_plan.block_shape, dtype=np.int32
            )
            group.attrs["spacing"] = np.asarray(spacing, dtype=np.float64)
            active = np.asarray(metadata.active, dtype=np.bool_)
            logical = np.asarray(metadata.logical_indices, dtype=np.int32)
            origins = np.zeros(
                (level_plan.maximum_blocks, len(level_plan.block_shape)),
                dtype=np.float64,
            )
            origins[active] = lower + logical[active] * (
                np.asarray(level_plan.block_shape) * np.asarray(spacing)
            )
            for name, value in (
                ("active", active),
                ("stable_block_ids", np.asarray(metadata.block_ids, dtype=np.int32)),
                ("parent_ids", np.asarray(metadata.parent_ids, dtype=np.int32)),
                ("logical_indices", logical),
                ("neighbor_slots", np.asarray(metadata.neighbor_slots, dtype=np.int32)),
                ("covered_cells", np.asarray(covered_cells, dtype=np.bool_)),
                ("interfaces", np.asarray(interfaces, dtype=np.bool_)),
                ("origins", origins),
            ):
                group.create_dataset(name, data=value)

    def initialize(self, discretization: OutputDiscretization, /) -> None:
        self._validate_discretization(discretization)
        h5py = _h5py()
        target = Path(self.hdf5_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(target, "w") as handle:
            handle.attrs["geometry_kind"] = self.geometry_kind
            handle.attrs["discretization_id"] = self.discretization_id
            handle.attrs["topology_id"] = self.topology_id
            handle.attrs["geometry_id"] = self.geometry_id
            handle.attrs["precision_policy_id"] = self.precision.policy_id
            handle.attrs["precision_evidence_id"] = self.precision_evidence.evidence_id
            handle.attrs["component_names"] = np.asarray(
                self.component_names, dtype=h5py.string_dtype()
            )
            if isinstance(discretization, PreparedBlockAMRRuntime):
                self._initialize_block_geometry(handle, discretization)
            elif isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
                mesh = handle.create_group("mesh")
                mesh.create_dataset("points", data=np.asarray(discretization.vertices))
                mesh.create_dataset(
                    "triangles", data=np.asarray(discretization.triangles, dtype=np.int32)
                )
                mesh.create_dataset(
                    "quadrilaterals",
                    data=np.asarray(discretization.quadrilaterals, dtype=np.int32),
                )
                mesh.create_dataset(
                    "tetrahedra",
                    data=np.asarray(discretization.tetrahedra, dtype=np.int32),
                )
                mesh.create_dataset(
                    "vertex_global_ids",
                    data=np.asarray(discretization.vertex_global_ids, dtype=np.int64),
                )
                mesh.create_dataset(
                    "cell_global_ids",
                    data=np.asarray(discretization.cell_global_ids, dtype=np.int64),
                )
                handle.create_group("geometry_epochs")
            else:
                coordinates = handle.create_group("coordinates")
                for axis_name, axis in zip(
                    discretization.grid.axis_names,
                    discretization.grid.structured_axes,
                    strict=True,
                ):
                    coordinates.create_dataset(
                        axis_name, data=np.asarray(axis.point_coordinates)
                    )
            handle.create_group("steps")
        self._write_xdmf(discretization)

    def _store_geometry_points(
        self,
        handle: Any,
        runtime_state: FiniteVolumeRuntimeState,
        points: np.ndarray | None,
        /,
    ) -> str | None:
        if points is None:
            return None
        if "geometry_epochs" not in handle:
            raise ValueError("Finite-volume output geometry epoch inventory changed.")
        content = runtime_state.content_state
        epoch = runtime_state.topology_journal.epoch_table[-1]
        artifacts = runtime_state.topology_journal.artifact_table[-1]
        epoch_key = canonical_fingerprint(
            {
                "kind": "finite-volume-output-geometry-epoch",
                "topology_epoch_id": content.topology_epoch_id,
            }
        )
        epoch_group = handle["geometry_epochs"].require_group(epoch_key)
        expected_epoch_attrs = {
            "topology_epoch_id": epoch.epoch_id,
            "topology_epoch_index": epoch.index,
            "geometry_id": epoch.geometry_id,
            "topology_id": epoch.topology_id,
            "partition_id": epoch.partition_id,
            "topology_artifacts_id": artifacts.artifacts_id,
        }
        for name, value in expected_epoch_attrs.items():
            if name in epoch_group.attrs and epoch_group.attrs[name] != value:
                raise ValueError("Finite-volume output geometry epoch identity changed.")
            epoch_group.attrs[name] = value
        versions = epoch_group.require_group("versions")
        version = int(np.asarray(content.geometry_version))
        version_group = versions.require_group(f"{version:010d}")
        expected_version_attrs = {
            "geometry_layout_id": content.geometry_layout_id,
            "geometry_version": version,
        }
        for name, value in expected_version_attrs.items():
            if name in version_group.attrs and version_group.attrs[name] != value:
                raise ValueError(
                    "Finite-volume output geometry version identity changed."
                )
            version_group.attrs[name] = value
        if "points" in version_group:
            if not np.array_equal(np.asarray(version_group["points"]), points):
                raise ValueError("Finite-volume output geometry version points changed.")
        else:
            version_group.create_dataset("points", data=points)
        return version_group["points"].name

    def _write_block_snapshot(
        self,
        runtime: PreparedBlockAMRRuntime,
        state: BlockAMRRuntimeState,
        /,
    ) -> int:
        self._validate_discretization(runtime)
        if not isinstance(state, BlockAMRRuntimeState):
            raise TypeError("Block output state must be BlockAMRRuntimeState.")
        topology = state.hierarchy_state.topology
        if (
            topology.epoch.epoch_id != runtime.dynamics.topology.epoch.epoch_id
            or state.topology_journal.current_epoch_id != topology.epoch.epoch_id
        ):
            raise ValueError("Block output state has an incompatible topology epoch.")
        for level_state in state.hierarchy_state.levels:
            self.precision.validate_state(level_state.values)
        h5py = _h5py()
        target = Path(self.hdf5_path)
        if not target.exists():
            self.initialize(runtime)
        with h5py.File(target, "a") as handle:
            expected_attrs = {
                "geometry_kind": "block_amr",
                "discretization_id": self.discretization_id,
                "topology_id": self.topology_id,
                "geometry_id": self.geometry_id,
                "precision_policy_id": self.precision.policy_id,
                "precision_evidence_id": self.precision_evidence.evidence_id,
            }
            if any(
                handle.attrs.get(name) != value for name, value in expected_attrs.items()
            ):
                raise ValueError("Block finite-volume output identity changed.")
            hierarchy_group = handle.get("block_hierarchy")
            if hierarchy_group is None or (
                hierarchy_group.attrs.get("topology_epoch_id") != topology.epoch.epoch_id
                or hierarchy_group.attrs.get("route_ids_json") != self.route_ids_json
                or hierarchy_group.attrs.get("distributed_partition_json")
                != self.distributed_partition_json
            ):
                raise ValueError("Block finite-volume output topology inventory changed.")
            steps = handle["steps"]
            index = len(steps)
            group = steps.create_group(f"{index:08d}")
            group.attrs["time"] = float(state.time)
            group.attrs["accepted_step"] = int(state.accepted_step)
            group.attrs["level_accepted_steps"] = np.asarray(
                state.level_accepted_steps, dtype=np.int32
            )
            group.attrs["status"] = int(state.last_status)
            group.attrs["topology_epoch_id"] = topology.epoch.epoch_id
            group.attrs["topology_epoch_index"] = topology.epoch.index
            group.attrs["topology_id"] = topology.topology_id
            group.attrs["geometry_id"] = topology.epoch.geometry_id
            group.attrs["canonical_partition_id"] = topology.partition_id
            group.attrs["topology_artifacts_id"] = runtime.topology_artifacts.artifacts_id
            group.attrs["hierarchy_payload_id"] = state.hierarchy_state.payload_id
            levels = group.create_group("levels")
            output_dtype = self.precision.numpy_dtype("output")
            for level, level_state in enumerate(state.hierarchy_state.levels):
                metadata = level_state.metadata
                expected_metadata = runtime.dynamics.topology.levels[level]
                if metadata.metadata_id != expected_metadata.metadata_id:
                    raise ValueError("Block output level metadata identity changed.")
                level_group = levels.create_group(f"{level:04d}")
                level_group.attrs["metadata_id"] = metadata.metadata_id
                active = np.asarray(metadata.active, dtype=np.bool_)
                level_group.attrs["active_stable_block_ids"] = np.asarray(
                    metadata.block_ids, dtype=np.int32
                )[active]
                level_group.create_dataset(
                    "cell_average",
                    data=np.asarray(level_state.safe_values(), dtype=output_dtype),
                    compression="gzip",
                    shuffle=True,
                )
        self._write_xdmf(runtime)
        return index

    def write_snapshot(
        self,
        discretization: OutputDiscretization,
        runtime_state: FiniteVolumeRuntimeState | BlockAMRRuntimeState,
        /,
        *,
        accepted_geometry: UnstructuredFiniteVolumeGeometryState
        | ArrayLike
        | None = None,
        shallow_water: ShallowWaterObservables | None = None,
    ) -> int:
        if isinstance(discretization, PreparedBlockAMRRuntime):
            if accepted_geometry is not None or shallow_water is not None:
                raise ValueError(
                    "Block snapshots do not accept unstructured geometry or shallow-water views."
                )
            if not isinstance(runtime_state, BlockAMRRuntimeState):
                raise TypeError("Block output state must be BlockAMRRuntimeState.")
            return self._write_block_snapshot(discretization, runtime_state)
        self._validate_discretization(discretization)
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        content_state = runtime_state.content_state
        self.precision.validate_state(content_state.conservative_content)
        cell_count = np.asarray(discretization.cell_volumes).size
        content_shape = (cell_count, discretization.component_count)
        if content_state.conservative_content.shape != content_shape:
            raise ValueError("Output conservative content shape changed.")
        cell_average = runtime_state.cell_average()
        if cell_average.shape != content_shape:
            raise ValueError("Output cell-average shape changed.")
        points = _accepted_points(discretization, runtime_state, accepted_geometry)
        output_dtype = self.precision.numpy_dtype("output")
        h5py = _h5py()
        target = Path(self.hdf5_path)
        if not target.exists():
            self.initialize(discretization)
        with h5py.File(target, "a") as handle:
            geometry_points_path = self._store_geometry_points(
                handle,
                runtime_state,
                points,
            )
            steps = handle["steps"]
            index = len(steps)
            group = steps.create_group(f"{index:08d}")
            epoch = runtime_state.topology_journal.epoch_table[-1]
            artifacts = runtime_state.topology_journal.artifact_table[-1]
            group.attrs["time"] = float(runtime_state.time)
            group.attrs["accepted_step"] = int(runtime_state.accepted_step)
            group.attrs["status"] = int(runtime_state.last_status)
            group.attrs["topology_epoch_id"] = content_state.topology_epoch_id
            group.attrs["topology_epoch_index"] = epoch.index
            group.attrs["topology_id"] = epoch.topology_id
            group.attrs["geometry_id"] = epoch.geometry_id
            group.attrs["partition_id"] = epoch.partition_id
            group.attrs["topology_artifacts_id"] = artifacts.artifacts_id
            group.attrs["geometry_layout_id"] = content_state.geometry_layout_id
            group.attrs["geometry_version"] = int(content_state.geometry_version)
            group.attrs["evidence_policy_id"] = content_state.evidence_policy_id
            group.attrs["evidence_version"] = int(content_state.evidence_version)
            if geometry_points_path is not None:
                group.attrs["geometry_points_path"] = geometry_points_path
            group.create_dataset(
                "conservative_content",
                data=np.asarray(
                    content_state.conservative_content,
                    dtype=output_dtype,
                ),
                compression="gzip",
                shuffle=True,
            )
            group.create_dataset(
                "cell_average",
                data=np.asarray(
                    cell_average.reshape(discretization.state_shape),
                    dtype=output_dtype,
                ),
                compression="gzip",
                shuffle=True,
            )
            group.create_dataset(
                "effective_cell_volumes",
                data=np.asarray(
                    content_state.effective_cell_volumes,
                    dtype=output_dtype,
                ),
                compression="gzip",
                shuffle=True,
            )
            group.create_dataset(
                "active_cell_mask",
                data=np.asarray(content_state.active_cell_mask, dtype=np.bool_),
                compression="gzip",
                shuffle=True,
            )
            if shallow_water is not None:
                from ..discretization.finite_volume import ShallowWaterObservables

                if not isinstance(shallow_water, ShallowWaterObservables):
                    raise TypeError(
                        "shallow_water must be ShallowWaterObservables or None."
                    )
                expected_shape = discretization.cell_shape
                if (
                    shallow_water.depth.shape != expected_shape
                    or shallow_water.bathymetry.shape != expected_shape
                    or shallow_water.surface.shape != expected_shape
                    or shallow_water.velocity.shape
                    != expected_shape + (discretization.component_count - 1,)
                ):
                    raise ValueError(
                        "Shallow-water observable shapes must match output geometry."
                    )
                shallow_group = group.create_group("shallow_water")
                shallow_group.attrs["bed_id"] = shallow_water.bed_id
                shallow_group.attrs["precision_id"] = shallow_water.precision_id
                for name, field in (
                    ("depth", shallow_water.depth),
                    ("bathymetry", shallow_water.bathymetry),
                    ("surface", shallow_water.surface),
                    ("momentum", shallow_water.momentum),
                    ("velocity", shallow_water.velocity),
                    ("energy_density", shallow_water.energy_density),
                ):
                    shallow_group.create_dataset(
                        name,
                        data=np.asarray(field, dtype=output_dtype),
                        compression="gzip",
                        shuffle=True,
                    )
                shallow_group.create_dataset(
                    "wet_mask",
                    data=np.asarray(shallow_water.wet_mask, dtype=np.bool_),
                    compression="gzip",
                    shuffle=True,
                )
        self._write_xdmf(discretization)
        return index

    def write_vtk_snapshot(
        self,
        path: str | Path,
        discretization: UnstructuredFiniteVolumeDiscretization,
        runtime_state: FiniteVolumeRuntimeState,
        /,
        *,
        accepted_geometry: UnstructuredFiniteVolumeGeometryState
        | ArrayLike
        | None = None,
    ) -> Path:
        """Write one meshio-readable VTK sidecar; never use it for restart."""

        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("VTK snapshots require unstructured finite-volume geometry.")
        self._validate_discretization(discretization)
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        content_state = runtime_state.content_state
        self.precision.validate_state(content_state.conservative_content)
        if content_state.conservative_content.shape != discretization.state_shape:
            raise ValueError("Output cell-average shape changed.")
        cell_average = np.asarray(
            runtime_state.cell_average(),
            dtype=self.precision.numpy_dtype("output"),
        )
        blocks: dict[str, Any] = {}
        block_counts: list[int] = []
        for name, cells in (
            ("triangle", discretization.triangles),
            ("quad", discretization.quadrilaterals),
            ("tetra", discretization.tetrahedra),
        ):
            values = np.asarray(cells, dtype=np.int32)
            if values.shape[0]:
                blocks[name] = values
                block_counts.append(values.shape[0])
        cell_data = {component: [] for component in self.component_names}
        cell_data["cell_global_id"] = []
        offset = 0
        global_ids = np.asarray(discretization.cell_global_ids, dtype=np.int64)
        for count in block_counts:
            for component_index, component in enumerate(self.component_names):
                cell_data[component].append(
                    cell_average[offset : offset + count, component_index]
                )
            cell_data["cell_global_id"].append(global_ids[offset : offset + count])
            offset += count
        points = _accepted_points(discretization, runtime_state, accepted_geometry)
        if points is None:
            raise RuntimeError("Unstructured output geometry resolution failed.")
        if points.shape[1] == 2:
            points = np.pad(points, ((0, 0), (0, 1)))
        meshio = import_module("meshio")
        mesh = meshio.Mesh(
            points,
            blocks,
            point_data={
                "vertex_global_id": np.asarray(
                    discretization.vertex_global_ids, dtype=np.int64
                )
            },
            cell_data=cell_data,
        )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mesh.write(target)
        return target

    def _write_xdmf(self, discretization: OutputDiscretization, /) -> None:
        hdf5_path = Path(self.hdf5_path)
        if not hdf5_path.exists():
            return
        if isinstance(discretization, PreparedBlockAMRRuntime):
            payload = self._block_xdmf(discretization)
        elif isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            payload = self._unstructured_xdmf(discretization)
        else:
            payload = self._structured_xdmf(discretization)
        _atomic_text(Path(self.xdmf_path), payload)

    def _block_xdmf(self, runtime: PreparedBlockAMRRuntime, /) -> str:
        topology = runtime.dynamics.topology
        hierarchy = topology.plan
        rank = len(hierarchy.grid.shape)
        geometry_type = {1: "ORIGIN_DX", 2: "ORIGIN_DXDY", 3: "ORIGIN_DXDYDZ"}[rank]
        topology_type = f"{rank}DCoRectMesh"
        hdf5_name = Path(self.hdf5_path).name
        h5py = _h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            records = tuple(
                (name, float(group.attrs["time"]))
                for name, group in handle["steps"].items()
            )
            output_precision = (
                handle[f"steps/{records[0][0]}/levels/0000/cell_average"].dtype.itemsize
                if records
                else self.precision.numpy_dtype("output").itemsize
            )
        step_grids = []
        for step_name, time in records:
            block_grids = []
            for level, (level_plan, metadata, spacing) in enumerate(
                zip(
                    hierarchy.levels,
                    topology.levels,
                    hierarchy.level_spacings,
                    strict=True,
                )
            ):
                active = np.asarray(metadata.active, dtype=np.bool_)
                logical = np.asarray(metadata.logical_indices, dtype=np.int32)
                stable_ids = np.asarray(metadata.block_ids, dtype=np.int32)
                lower = np.asarray(
                    [
                        float(np.asarray(axis.bounds)[0])
                        for axis in hierarchy.grid.structured_axes
                    ]
                )
                for slot in np.flatnonzero(active):
                    origin = lower + logical[slot] * (
                        np.asarray(level_plan.block_shape) * np.asarray(spacing)
                    )
                    point_dimensions = " ".join(
                        str(size + 1) for size in level_plan.block_shape[::-1]
                    )
                    attribute_dimensions = " ".join(
                        [
                            *(str(size) for size in level_plan.block_shape[::-1]),
                            str(len(self.component_names)),
                        ]
                    )
                    source_dimensions = " ".join(
                        str(value)
                        for value in (
                            level_plan.maximum_blocks,
                            *level_plan.block_shape,
                            len(self.component_names),
                        )
                    )
                    start = (int(slot),) + (0,) * rank + (0,)
                    stride = (1,) * (rank + 2)
                    count = (
                        (1,)
                        + tuple(level_plan.block_shape)
                        + (len(self.component_names),)
                    )
                    hyperslab = "  ".join(
                        " ".join(str(value) for value in row)
                        for row in (start, stride, count)
                    )
                    state_path = (
                        f"{hdf5_name}:/steps/{step_name}/levels/{level:04d}/cell_average"
                    )
                    grid_open = f'        <Grid Name="level-{level}-block-{int(stable_ids[slot])}" GridType="Uniform">'
                    topology_element = f'          <Topology TopologyType="{topology_type}" Dimensions="{point_dimensions}"/>'
                    origin_item = (
                        f'            <DataItem Dimensions="{rank}" '
                        'NumberType="Float" Precision="8" Format="XML">'
                        + " ".join(f"{value:.17g}" for value in origin)
                        + "</DataItem>"
                    )
                    spacing_item = (
                        f'            <DataItem Dimensions="{rank}" '
                        'NumberType="Float" Precision="8" Format="XML">'
                        + " ".join(f"{value:.17g}" for value in spacing)
                        + "</DataItem>"
                    )
                    attribute_open = '          <Attribute Name="cell_average" AttributeType="Vector" Center="Cell">'
                    hyperslab_open = (
                        '            <DataItem ItemType="HyperSlab" '
                        f'Dimensions="{attribute_dimensions}" Type="HyperSlab">'
                    )
                    selector_item = f'              <DataItem Dimensions="3 {rank + 2}" Format="XML">{hyperslab}</DataItem>'
                    source_item = (
                        f'              <DataItem Dimensions="{source_dimensions}" '
                        f'NumberType="Float" Precision="{output_precision}" '
                        f'Format="HDF">{state_path}</DataItem>'
                    )
                    block_grids.append(
                        "\n".join(
                            (
                                grid_open,
                                topology_element,
                                f'          <Geometry GeometryType="{geometry_type}">',
                                origin_item,
                                spacing_item,
                                "          </Geometry>",
                                attribute_open,
                                hyperslab_open,
                                selector_item,
                                source_item,
                                "            </DataItem>",
                                "          </Attribute>",
                                "        </Grid>",
                            )
                        )
                    )
            step_grids.append(
                "\n".join(
                    (
                        f'      <Grid Name="step-{step_name}" GridType="Collection" CollectionType="Spatial">',
                        f'        <Time Value="{time:.17g}"/>',
                        *block_grids,
                        "      </Grid>",
                    )
                )
            )
        return "\n".join(
            (
                '<?xml version="1.0" ?>',
                '<Xdmf Version="3.0">',
                "  <Domain>",
                '    <Grid Name="block-finite-volume" GridType="Collection" CollectionType="Temporal">',
                *step_grids,
                "    </Grid>",
                "  </Domain>",
                "</Xdmf>",
                "",
            )
        )

    def _records(self):
        h5py = _h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            records = []
            for name, group in handle["steps"].items():
                geometry_path = group.attrs.get("geometry_points_path")
                geometry_precision = None
                if self.geometry_kind == "unstructured":
                    if (
                        not isinstance(geometry_path, str)
                        or geometry_path not in handle
                        or tuple(handle[geometry_path].shape)
                        != tuple(handle["mesh/points"].shape)
                    ):
                        raise ValueError(
                            "Unstructured output step has no accepted geometry."
                        )
                    geometry_precision = handle[geometry_path].dtype.itemsize
                records.append(
                    (
                        name,
                        float(group.attrs["time"]),
                        geometry_path,
                        geometry_precision,
                    )
                )
            records = tuple(records)
            state_precision = (
                handle["steps"][records[0][0]]["cell_average"].dtype.itemsize
                if records
                else self.precision.numpy_dtype("output").itemsize
            )
        return records, state_precision

    def _unstructured_xdmf(
        self, discretization: UnstructuredFiniteVolumeDiscretization, /
    ) -> str:
        records, state_precision = self._records()
        hdf5_name = Path(self.hdf5_path).name
        points = np.asarray(discretization.vertices)
        geometry_type = "XY" if points.shape[1] == 2 else "XYZ"
        cell_count = discretization.cell_count
        component_count = discretization.component_count
        blocks = []
        offset = 0
        for name, topology_type, cells in (
            ("triangles", "Triangle", discretization.triangles),
            ("quadrilaterals", "Quadrilateral", discretization.quadrilaterals),
            ("tetrahedra", "Tetrahedron", discretization.tetrahedra),
        ):
            count = cells.shape[0]
            if count:
                blocks.append((name, topology_type, count, offset, cells.shape[1]))
                offset += count
        grids = []
        for step_name, time, geometry_path, geometry_precision in records:
            children = []
            state_path = f"{hdf5_name}:/steps/{step_name}/cell_average"
            for block_name, topology_type, count, block_offset, arity in blocks:
                attributes = []
                for component_index, component in enumerate(self.component_names):
                    selection = f"{block_offset} {component_index}  1 1  {count} 1"
                    state_item = (
                        f'              <DataItem Dimensions="{cell_count} '
                        f'{component_count}" NumberType="Float" '
                        f'Precision="{state_precision}" Format="HDF">'
                        f"{state_path}</DataItem>"
                    )
                    attributes.append(
                        "\n".join(
                            (
                                f'          <Attribute Name="{html.escape(component)}" '
                                'AttributeType="Scalar" Center="Cell">',
                                f'            <DataItem ItemType="HyperSlab" Dimensions="{count}" Type="HyperSlab">',
                                f'              <DataItem Dimensions="3 2" Format="XML">{selection}</DataItem>',
                                state_item,
                                "            </DataItem>",
                                "          </Attribute>",
                            )
                        )
                    )
                topology_item = (
                    f'            <DataItem Dimensions="{count} {arity}" '
                    'NumberType="Int" Precision="4" Format="HDF">'
                    f"{hdf5_name}:/mesh/{block_name}</DataItem>"
                )
                if geometry_path is None or geometry_precision is None:
                    raise ValueError("Unstructured output step has no accepted geometry.")
                geometry_item = (
                    f'            <DataItem Dimensions="{points.shape[0]} '
                    f'{points.shape[1]}" NumberType="Float" '
                    f'Precision="{geometry_precision}" Format="HDF">'
                    f"{hdf5_name}:{geometry_path}</DataItem>"
                )
                children.append(
                    "\n".join(
                        (
                            f'        <Grid Name="{block_name}" GridType="Uniform">',
                            f'          <Topology TopologyType="{topology_type}" NumberOfElements="{count}">',
                            topology_item,
                            "          </Topology>",
                            f'          <Geometry GeometryType="{geometry_type}">',
                            geometry_item,
                            "          </Geometry>",
                            *attributes,
                            "        </Grid>",
                        )
                    )
                )
            grids.append(
                f'''      <Grid Name="step-{step_name}" GridType="Collection" CollectionType="Spatial">
        <Time Value="{time:.17g}"/>
{chr(10).join(children)}
      </Grid>'''
            )
        return "\n".join(
            (
                '<?xml version="1.0" ?>',
                '<Xdmf Version="3.0">',
                "  <Domain>",
                '    <Grid Name="finite-volume" GridType="Collection" CollectionType="Temporal">',
                *grids,
                "    </Grid>",
                "  </Domain>",
                "</Xdmf>",
                "",
            )
        )

    def _structured_xdmf(self, discretization: FiniteVolumeDiscretization, /) -> str:
        records, state_precision = self._records()
        hdf5_path = Path(self.hdf5_path)
        h5py = _h5py()
        with h5py.File(hdf5_path, "r") as handle:
            coordinate_precision = {
                name: dataset.dtype.itemsize
                for name, dataset in handle["coordinates"].items()
            }
        vertex_counts = [
            axis.point_coordinates.size for axis in discretization.grid.structured_axes
        ]
        while len(vertex_counts) < 3:
            vertex_counts.append(1)
        topology_dimensions = " ".join(str(int(value)) for value in vertex_counts[::-1])
        axis_names = list(discretization.grid.axis_names)
        while len(axis_names) < 3:
            axis_names.append(f"inactive_{len(axis_names)}")
        geometry_items = []
        for index, axis_name in enumerate(axis_names[:3]):
            if index < len(discretization.cell_shape):
                size = discretization.grid.structured_axes[index].point_coordinates.size
                geometry_items.append(
                    f'<DataItem Dimensions="{size}" NumberType="Float" '
                    f'Precision="{coordinate_precision[axis_name]}" Format="HDF">'
                    f"{hdf5_path.name}:/coordinates/{axis_name}</DataItem>"
                )
            else:
                geometry_items.append(
                    '<DataItem Dimensions="1" NumberType="Float" Precision="8" Format="XML">0</DataItem>'
                )
        state_dimensions = " ".join(
            str(int(value))
            for value in (
                *discretization.cell_shape[::-1],
                len(self.component_names),
            )
        )
        grids = []
        for name, time, _, _ in records:
            state_path = f"{hdf5_path.name}:/steps/{name}/cell_average"
            state_item = (
                f'          <DataItem Dimensions="{state_dimensions}" '
                f'NumberType="Float" Precision="{state_precision}" Format="HDF">'
                f"{state_path}</DataItem>"
            )
            grids.append(
                "\n".join(
                    (
                        f'      <Grid Name="step-{name}" GridType="Uniform">',
                        f'        <Time Value="{time:.17g}"/>',
                        f'        <Topology TopologyType="3DRectMesh" Dimensions="{topology_dimensions}"/>',
                        '        <Geometry GeometryType="VXVYVZ">',
                        f"          {geometry_items[0]}",
                        f"          {geometry_items[1]}",
                        f"          {geometry_items[2]}",
                        "        </Geometry>",
                        '        <Attribute Name="cell_average" AttributeType="Vector" Center="Cell">',
                        state_item,
                        "        </Attribute>",
                        "      </Grid>",
                    )
                )
            )
        return "\n".join(
            (
                '<?xml version="1.0" ?>',
                '<Xdmf Version="3.0">',
                "  <Domain>",
                '    <Grid Name="finite-volume" GridType="Collection" CollectionType="Temporal">',
                *grids,
                "    </Grid>",
                "  </Domain>",
                "</Xdmf>",
                "",
            )
        )


__all__ = ["FiniteVolumeOutputPlan"]
