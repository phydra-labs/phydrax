#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import FiniteVolumeDiscretization
from ._finite_volume_runtime import FiniteVolumeRuntimeState


def _h5py():
    if find_spec("h5py") is None:
        raise ImportError(
            "Finite-volume HDF5 output requires the optional 'h5py' package."
        )
    return import_module("h5py")


class FiniteVolumeOutputPlan(StrictModule, NonTrainableState):
    """Host-side HDF5 snapshots and XDMF temporal metadata."""

    hdf5_path: str = eqx.field(static=True)
    xdmf_path: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    output_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: str | Path,
        discretization: FiniteVolumeDiscretization,
        /,
    ):
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError(
                "XDMF output currently requires Cartesian finite-volume geometry."
            )
        target = Path(path)
        hdf5_path = target if target.suffix == ".h5" else target.with_suffix(".h5")
        xdmf_path = hdf5_path.with_suffix(".xdmf")
        self.hdf5_path = str(hdf5_path)
        self.xdmf_path = str(xdmf_path)
        self.discretization_id = discretization.prepared_id
        self.component_names = discretization.component_names
        self.output_id = canonical_fingerprint(
            {
                "kind": "finite-volume-output",
                "hdf5": str(hdf5_path),
                "discretization": discretization.prepared_id,
                "components": list(discretization.component_names),
            }
        )

    def initialize(self, discretization: FiniteVolumeDiscretization, /) -> None:
        if discretization.prepared_id != self.discretization_id:
            raise ValueError("Output discretization identity changed.")
        h5py = _h5py()
        target = Path(self.hdf5_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(target, "w") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["discretization_id"] = self.discretization_id
            handle.attrs["component_names"] = np.asarray(
                self.component_names, dtype=h5py.string_dtype()
            )
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

    def write_snapshot(
        self,
        discretization: FiniteVolumeDiscretization,
        runtime_state: FiniteVolumeRuntimeState,
        /,
    ) -> int:
        if discretization.prepared_id != self.discretization_id:
            raise ValueError("Output discretization identity changed.")
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        h5py = _h5py()
        target = Path(self.hdf5_path)
        if not target.exists():
            self.initialize(discretization)
        with h5py.File(target, "a") as handle:
            steps = handle["steps"]
            index = len(steps)
            group = steps.create_group(f"{index:08d}")
            group.attrs["time"] = float(runtime_state.time)
            group.attrs["accepted_step"] = int(runtime_state.accepted_step)
            group.attrs["status"] = int(runtime_state.last_status)
            group.create_dataset(
                "conservative_state",
                data=np.asarray(runtime_state.conservative_state),
                compression="gzip",
                shuffle=True,
            )
        self._write_xdmf(discretization)
        return index

    def _write_xdmf(self, discretization: FiniteVolumeDiscretization, /) -> None:
        hdf5_path = Path(self.hdf5_path)
        if not hdf5_path.exists():
            return
        h5py = _h5py()
        with h5py.File(hdf5_path, "r") as handle:
            records = tuple(
                (name, float(group.attrs["time"]))
                for name, group in handle["steps"].items()
            )
        vertex_counts = [
            axis.point_coordinates.size
            for axis in discretization.grid.structured_axes
        ]
        while len(vertex_counts) < 3:
            vertex_counts.append(1)
        topology_dimensions = " ".join(
            str(int(value)) for value in vertex_counts[::-1]
        )
        axis_names = list(discretization.grid.axis_names)
        while len(axis_names) < 3:
            axis_names.append(f"inactive_{len(axis_names)}")
        geometry_items = []
        for index, axis_name in enumerate(axis_names[:3]):
            if index < len(discretization.cell_shape):
                size = int(
                    discretization.grid.structured_axes[
                        index
                    ].point_coordinates.size
                )
                geometry_items.append(
                    f'<DataItem Dimensions="{size}" NumberType="Float" '
                    f'Precision="8" Format="HDF">{hdf5_path.name}:'
                    f'/coordinates/{axis_name}</DataItem>'
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
        for name, time in records:
            state_path = (
                f"{hdf5_path.name}:/steps/{name}/conservative_state"
            )
            grids.append(
                f'''      <Grid Name="step-{name}" GridType="Uniform">
        <Time Value="{time:.17g}"/>
        <Topology TopologyType="3DRectMesh" Dimensions="{topology_dimensions}"/>
        <Geometry GeometryType="VXVYVZ">
          {geometry_items[0]}
          {geometry_items[1]}
          {geometry_items[2]}
        </Geometry>
        <Attribute Name="conservative_state" AttributeType="Vector" Center="Cell">
          <DataItem Dimensions="{state_dimensions}" NumberType="Float"
                    Precision="8" Format="HDF">{state_path}</DataItem>
        </Attribute>
      </Grid>'''
            )
        payload = "\n".join(
            (
                '<?xml version="1.0" ?>',
                '<Xdmf Version="3.0">',
                '  <Domain>',
                '    <Grid Name="finite-volume" GridType="Collection" CollectionType="Temporal">',
                *grids,
                '    </Grid>',
                '  </Domain>',
                '</Xdmf>',
                '',
            )
        )
        Path(self.xdmf_path).write_text(payload)


__all__ = ["FiniteVolumeOutputPlan"]
