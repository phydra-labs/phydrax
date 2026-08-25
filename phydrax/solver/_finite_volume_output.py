#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import html
import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FiniteVolumeDiscretization,
    FiniteVolumePrecisionPolicy,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeGeometryState,
)
from ._finite_volume_runtime import FiniteVolumeRuntimeState


OutputDiscretization = FiniteVolumeDiscretization | UnstructuredFiniteVolumeDiscretization
_OUTPUT_SCHEMA_VERSION = 4


def _h5py():
    if find_spec("h5py") is None:
        raise ImportError(
            "Finite-volume HDF5 output requires the optional 'h5py' package."
        )
    return import_module("h5py")


def _atomic_text(path: Path, payload: str, /) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


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

    def __init__(
        self,
        path: str | Path,
        discretization: OutputDiscretization,
        /,
        *,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        if not isinstance(
            discretization,
            (FiniteVolumeDiscretization, UnstructuredFiniteVolumeDiscretization),
        ):
            raise TypeError(
                "Output requires structured or unstructured finite-volume geometry."
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
        target = Path(path)
        hdf5_path = target if target.suffix == ".h5" else target.with_suffix(".h5")
        xdmf_path = hdf5_path.with_suffix(".xdmf")
        self.hdf5_path = str(hdf5_path)
        self.xdmf_path = str(xdmf_path)
        self.discretization_id = discretization.prepared_id
        self.geometry_kind = geometry_kind
        self.topology_id = topology_id
        self.geometry_id = geometry_id
        self.component_names = discretization.component_names
        self.precision = precision_
        self.precision_evidence = precision_.evidence()
        self.output_id = canonical_fingerprint(
            {
                "kind": "finite-volume-output",
                "schema_version": _OUTPUT_SCHEMA_VERSION,
                "hdf5": str(hdf5_path),
                "discretization": discretization.prepared_id,
                "topology": topology_id,
                "geometry": geometry_id,
                "components": list(discretization.component_names),
                "precision": precision_.policy_id,
                "precision_evidence": self.precision_evidence.evidence_id,
            }
        )

    def _validate_discretization(self, discretization: OutputDiscretization, /) -> None:
        if discretization.prepared_id != self.discretization_id:
            raise ValueError("Output discretization identity changed.")
        if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            if (
                discretization.topology_id != self.topology_id
                or discretization.geometry_id != self.geometry_id
            ):
                raise ValueError("Output unstructured mesh identity changed.")

    def initialize(self, discretization: OutputDiscretization, /) -> None:
        self._validate_discretization(discretization)
        h5py = _h5py()
        target = Path(self.hdf5_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(target, "w") as handle:
            handle.attrs["schema_version"] = _OUTPUT_SCHEMA_VERSION
            handle.attrs["geometry_kind"] = self.geometry_kind
            handle.attrs["discretization_id"] = self.discretization_id
            handle.attrs["topology_id"] = self.topology_id
            handle.attrs["geometry_id"] = self.geometry_id
            handle.attrs["precision_policy_id"] = self.precision.policy_id
            handle.attrs["precision_evidence_id"] = self.precision_evidence.evidence_id
            handle.attrs["component_names"] = np.asarray(
                self.component_names, dtype=h5py.string_dtype()
            )
            if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
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
        epoch_key = canonical_fingerprint(
            {
                "kind": "finite-volume-output-geometry-epoch",
                "topology_epoch_id": content.topology_epoch_id,
            }
        )
        epoch_group = handle["geometry_epochs"].require_group(epoch_key)
        expected_epoch_attrs = {
            "topology_epoch_id": content.topology_epoch_id,
            "topology_id": self.topology_id,
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

    def write_snapshot(
        self,
        discretization: OutputDiscretization,
        runtime_state: FiniteVolumeRuntimeState,
        /,
        *,
        accepted_geometry: UnstructuredFiniteVolumeGeometryState
        | ArrayLike
        | None = None,
    ) -> int:
        self._validate_discretization(discretization)
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        content_state = runtime_state.content_state
        self.precision.validate_state(content_state.conservative_content)
        cell_count = int(np.asarray(discretization.cell_volumes).size)
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
            if int(handle.attrs.get("schema_version", -1)) != _OUTPUT_SCHEMA_VERSION:
                raise ValueError("Unsupported finite-volume output schema.")
            geometry_points_path = self._store_geometry_points(
                handle,
                runtime_state,
                points,
            )
            steps = handle["steps"]
            index = len(steps)
            group = steps.create_group(f"{index:08d}")
            group.attrs["time"] = float(runtime_state.time)
            group.attrs["accepted_step"] = int(runtime_state.accepted_step)
            group.attrs["status"] = int(runtime_state.last_status)
            group.attrs["topology_epoch_id"] = content_state.topology_epoch_id
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
        if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            payload = self._unstructured_xdmf(discretization)
        else:
            payload = self._structured_xdmf(discretization)
        _atomic_text(Path(self.xdmf_path), payload)

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
                    geometry_precision = int(handle[geometry_path].dtype.itemsize)
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
                int(handle["steps"][records[0][0]]["cell_average"].dtype.itemsize)
                if records
                else int(self.precision.numpy_dtype("output").itemsize)
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
            count = int(cells.shape[0])
            if count:
                blocks.append((name, topology_type, count, offset, int(cells.shape[1])))
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
                                f'            <DataItem ItemType="HyperSlab" '
                                f'Dimensions="{count}" Type="HyperSlab">',
                                '              <DataItem Dimensions="3 2" '
                                f'Format="XML">{selection}</DataItem>',
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
                            f'          <Topology TopologyType="{topology_type}" '
                            f'NumberOfElements="{count}">',
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
                name: int(dataset.dtype.itemsize)
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
                size = int(
                    discretization.grid.structured_axes[index].point_coordinates.size
                )
                geometry_items.append(
                    f'<DataItem Dimensions="{size}" NumberType="Float" '
                    f'Precision="{coordinate_precision[axis_name]}" Format="HDF">'
                    f"{hdf5_path.name}:/coordinates/{axis_name}</DataItem>"
                )
            else:
                geometry_items.append(
                    '<DataItem Dimensions="1" NumberType="Float" Precision="8" '
                    'Format="XML">0</DataItem>'
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
                        f'        <Topology TopologyType="3DRectMesh" '
                        f'Dimensions="{topology_dimensions}"/>',
                        '        <Geometry GeometryType="VXVYVZ">',
                        f"          {geometry_items[0]}",
                        f"          {geometry_items[1]}",
                        f"          {geometry_items[2]}",
                        "        </Geometry>",
                        '        <Attribute Name="cell_average" '
                        'AttributeType="Vector" Center="Cell">',
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
