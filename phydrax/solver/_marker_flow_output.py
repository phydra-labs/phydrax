#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import html
import os
from collections.abc import Mapping
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import FiniteVolumeDiscretization


def _h5py():
    if find_spec("h5py") is None:
        raise ImportError("Marker-flow output requires the optional 'h5py' package.")
    return import_module("h5py")


def _mapping_arrays(
    name: str, values: Mapping[str, ArrayLike], /
) -> dict[str, np.ndarray]:
    output = {}
    for field_name, value in values.items():
        identifier = str(field_name)
        array = np.asarray(value)
        if not identifier or array.dtype.hasobject or np.any(~np.isfinite(array)):
            raise ValueError(f"{name} field {identifier!r} is invalid.")
        output[identifier] = array
    return output


class MarkerFlowOutputPlan(StrictModule, NonTrainableState):
    """Accepted-state HDF5/XDMF series plus optional VTK point snapshots."""

    hdf5_path: str = eqx.field(static=True)
    xdmf_path: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    marker_ids: object
    output_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: str | Path,
        discretization: FiniteVolumeDiscretization,
        marker_ids: ArrayLike,
        /,
    ):
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        ids = np.asarray(marker_ids)
        if ids.ndim != 1 or np.unique(ids).size != ids.size:
            raise ValueError("marker_ids must be a unique rank-one array.")
        target = Path(path)
        hdf5 = target if target.suffix == ".h5" else target.with_suffix(".h5")
        self.hdf5_path = str(hdf5)
        self.xdmf_path = str(hdf5.with_suffix(".xdmf"))
        self.discretization_id = discretization.prepared_id
        self.marker_ids = ids
        self.output_id = canonical_fingerprint(
            {
                "kind": "marker-flow-output",
                "discretization": discretization.prepared_id,
                "marker_ids": ids.tolist(),
                "path": str(hdf5),
            }
        )

    def initialize(self, discretization: FiniteVolumeDiscretization, /) -> None:
        if discretization.prepared_id != self.discretization_id:
            raise ValueError("Marker-flow output discretization changed.")
        h5py = _h5py()
        path = Path(self.hdf5_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["output_id"] = self.output_id
            handle.attrs["discretization_id"] = self.discretization_id
            mesh = handle.create_group("eulerian_mesh")
            mesh.create_dataset(
                "cell_centers", data=np.asarray(discretization.cell_centers)
            )
            mesh.create_dataset(
                "cell_volumes", data=np.asarray(discretization.cell_volumes)
            )
            faces = mesh.create_group("face_centers")
            for axis, value in enumerate(discretization.face_centers):
                faces.create_dataset(str(axis), data=np.asarray(value))
            handle.create_dataset("marker_ids", data=self.marker_ids)
            handle.create_group("steps")
        self._write_xdmf()

    def write_snapshot(
        self,
        discretization: FiniteVolumeDiscretization,
        time: ArrayLike,
        accepted_step: ArrayLike,
        marker_position: ArrayLike,
        /,
        *,
        eulerian_fields: Mapping[str, ArrayLike],
        face_fields: Mapping[str, ArrayLike] | None = None,
        marker_fields: Mapping[str, ArrayLike] | None = None,
        rigid_fields: Mapping[str, ArrayLike] | None = None,
        deformable_fields: Mapping[str, ArrayLike] | None = None,
        contact_fields: Mapping[str, ArrayLike] | None = None,
        diagnostics: Mapping[str, ArrayLike] | None = None,
        write_vtk: bool = False,
    ) -> int:
        if discretization.prepared_id != self.discretization_id:
            raise ValueError("Marker-flow output discretization changed.")
        time_ = np.asarray(time)
        step = int(np.asarray(accepted_step))
        markers = np.asarray(marker_position)
        if time_.shape != () or not np.isfinite(time_) or step < 0:
            raise ValueError("Output time and accepted step are invalid.")
        if markers.shape != (
            self.marker_ids.size,
            len(discretization.cell_shape),
        ) or np.any(~np.isfinite(markers)):
            raise ValueError(
                "marker_position has an incompatible shape or nonfinite data."
            )
        groups = {
            "eulerian": _mapping_arrays("eulerian", eulerian_fields),
            "faces": _mapping_arrays("faces", {} if face_fields is None else face_fields),
            "markers": _mapping_arrays(
                "markers", {} if marker_fields is None else marker_fields
            ),
            "rigid": _mapping_arrays(
                "rigid", {} if rigid_fields is None else rigid_fields
            ),
            "deformable": _mapping_arrays(
                "deformable", {} if deformable_fields is None else deformable_fields
            ),
            "contact": _mapping_arrays(
                "contact", {} if contact_fields is None else contact_fields
            ),
            "diagnostics": _mapping_arrays(
                "diagnostics", {} if diagnostics is None else diagnostics
            ),
        }
        h5py = _h5py()
        with h5py.File(self.hdf5_path, "a") as handle:
            if handle.attrs.get("output_id") != self.output_id:
                raise ValueError("Marker-flow output archive identity changed.")
            steps = handle["steps"]
            key = f"{step:010d}"
            if key in steps:
                raise ValueError("Accepted output step already exists.")
            group = steps.create_group(key)
            group.attrs["time"] = time_.item()
            group.attrs["accepted_step"] = step
            group.create_dataset("marker_position", data=markers)
            for group_name, fields in groups.items():
                target = group.create_group(group_name)
                for field_name, value in fields.items():
                    target.create_dataset(field_name, data=value)
        self._write_xdmf()
        if write_vtk:
            self._write_vtk(discretization, step, markers, groups)
        return step

    def _write_xdmf(self, /) -> None:
        h5py = _h5py()
        hdf5_name = html.escape(Path(self.hdf5_path).name)
        grids = []

        def attributes(group, prefix):
            output = []
            for field_name, dataset in group.items():
                if field_name in ("position", "connectivity"):
                    continue
                shape = tuple(dataset.shape)
                dimensions = " ".join(str(value) for value in shape)
                attribute_type = "Vector" if shape and shape[-1] in (2, 3) else "Scalar"
                output.append(
                    f'<Attribute Name="{html.escape(field_name)}" '
                    f'AttributeType="{attribute_type}" Center="Node">'
                    f'<DataItem Dimensions="{dimensions}" Format="HDF">'
                    f"{hdf5_name}:{prefix}/{html.escape(field_name)}"
                    "</DataItem></Attribute>"
                )
            return "".join(output)

        with h5py.File(self.hdf5_path, "r") as handle:
            cell_shape = tuple(handle["eulerian_mesh/cell_centers"].shape)
            dimension = cell_shape[-1]
            geometry_type = "XY" if dimension == 2 else "XYZ"
            cell_count = int(np.prod(cell_shape[:-1]))
            for key in sorted(handle["steps"]):
                group = handle[f"steps/{key}"]
                time = float(group.attrs["time"])
                eulerian_attributes = attributes(
                    group["eulerian"], f"/steps/{key}/eulerian"
                )
                grids.append(
                    f'<Grid Name="eulerian-{key}" GridType="Uniform">'
                    f'<Time Value="{time:.17g}"/>'
                    f'<Topology TopologyType="Polyvertex" '
                    f'NumberOfElements="{cell_count}"/>'
                    f'<Geometry GeometryType="{geometry_type}">'
                    f'<DataItem Dimensions="{cell_count} {dimension}" Format="HDF">'
                    f"{hdf5_name}:/eulerian_mesh/cell_centers"
                    f"</DataItem></Geometry>{eulerian_attributes}</Grid>"
                )
                marker_attributes = attributes(group["markers"], f"/steps/{key}/markers")
                grids.append(
                    f'<Grid Name="markers-{key}" GridType="Uniform">'
                    f'<Time Value="{time:.17g}"/>'
                    f'<Topology TopologyType="Polyvertex" '
                    f'NumberOfElements="{self.marker_ids.size}"/>'
                    f'<Geometry GeometryType="{geometry_type}">'
                    f'<DataItem Dimensions="{self.marker_ids.size} {dimension}" '
                    f'Format="HDF">{hdf5_name}:/steps/{key}/marker_position'
                    f"</DataItem></Geometry>{marker_attributes}</Grid>"
                )
                for cloud in ("rigid", "deformable", "contact"):
                    cloud_group = group[cloud]
                    if "position" not in cloud_group:
                        continue
                    point_shape = tuple(cloud_group["position"].shape)
                    if len(point_shape) != 2 or point_shape[1] != dimension:
                        raise ValueError(
                            f"{cloud} position must have shape (count, dimension)."
                        )
                    grids.append(
                        f'<Grid Name="{cloud}-{key}" GridType="Uniform">'
                        f'<Time Value="{time:.17g}"/>'
                        f'<Topology TopologyType="Polyvertex" '
                        f'NumberOfElements="{point_shape[0]}"/>'
                        f'<Geometry GeometryType="{geometry_type}">'
                        f'<DataItem Dimensions="{point_shape[0]} {dimension}" '
                        f'Format="HDF">{hdf5_name}:/steps/{key}/{cloud}/position'
                        f"</DataItem></Geometry>"
                        f"{attributes(cloud_group, f'/steps/{key}/{cloud}')}</Grid>"
                    )
        payload = (
            '<?xml version="1.0"?><Xdmf Version="3.0"><Domain>'
            '<Grid Name="marker-flow" GridType="Collection" '
            'CollectionType="Temporal">' + "".join(grids) + "</Grid></Domain></Xdmf>"
        )
        target = Path(self.xdmf_path)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(payload)
        os.replace(temporary, target)

    def _write_vtk(
        self,
        discretization: FiniteVolumeDiscretization,
        step: int,
        marker_position: np.ndarray,
        groups: Mapping[str, Mapping[str, np.ndarray]],
        /,
    ) -> None:
        if find_spec("meshio") is None:
            raise ImportError(
                "VTK marker-flow output requires the optional 'meshio' package."
            )
        meshio = import_module("meshio")
        root = Path(self.hdf5_path).with_suffix("")

        def points3(value):
            return np.pad(value, ((0, 0), (0, 1))) if value.shape[1] == 2 else value

        def write_cloud(name, position, fields):
            point_data = {
                field_name: value
                for field_name, value in fields.items()
                if field_name not in ("position", "connectivity")
                and value.shape[:1] == position.shape[:1]
            }
            meshio.write_points_cells(
                f"{root}-{name}-{step:010d}.vtu",
                points3(position),
                [("vertex", np.arange(position.shape[0])[:, None])],
                point_data=point_data,
            )

        write_cloud("markers", marker_position, groups["markers"])
        centers = np.asarray(discretization.cell_centers).reshape(
            (-1, len(discretization.cell_shape))
        )
        cell_data = {
            name: value.reshape(
                (centers.shape[0],) + value.shape[len(discretization.cell_shape) :]
            )
            for name, value in groups["eulerian"].items()
        }
        write_cloud("eulerian", centers, cell_data)
        for cloud in ("rigid", "deformable", "contact"):
            fields = groups[cloud]
            if "position" in fields:
                position = fields["position"]
                if position.ndim != 2:
                    raise ValueError(f"{cloud} position must be rank two.")
                write_cloud(cloud, position, fields)


__all__ = ["MarkerFlowOutputPlan"]
