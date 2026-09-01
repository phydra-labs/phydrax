#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections import deque
from importlib import import_module
from pathlib import Path

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.mpm import MPMRuntimeState
from ..equations import CompiledMaterialPointProblem


_OUTPUT_SCHEMA_VERSION = 1


def _h5py():
    return import_module("h5py")


class MPMOutputManifest(StrictModule, NonTrainableState):
    schema_version: int = eqx.field(static=True)
    output_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    accepted_steps: int = eqx.field(static=True)
    last_time_hex: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)


class MPMOutputPlan(StrictModule, NonTrainableState):
    compiled: CompiledMaterialPointProblem
    hdf5_path: str = eqx.field(static=True)
    xdmf_path: str = eqx.field(static=True)
    output_id: str = eqx.field(static=True)

    def __init__(
        self,
        compiled: CompiledMaterialPointProblem,
        target: str | Path,
        /,
    ):
        if not isinstance(compiled, CompiledMaterialPointProblem):
            raise TypeError("compiled must be CompiledMaterialPointProblem.")
        path = Path(target)
        hdf5 = path if path.suffix == ".h5" else path.with_suffix(".h5")
        self.compiled = compiled
        self.hdf5_path = str(hdf5)
        self.xdmf_path = str(hdf5.with_suffix(".xdmf"))
        self.output_id = canonical_fingerprint(
            {
                "kind": "mpm-output-plan",
                "schema_version": _OUTPUT_SCHEMA_VERSION,
                "compilation": compiled.compilation_id,
                "hdf5_path": str(hdf5),
            }
        )

    def initialize(self):
        h5py = _h5py()
        path = Path(self.hdf5_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = _OUTPUT_SCHEMA_VERSION
            handle.attrs["output_id"] = self.output_id
            handle.attrs["compilation_id"] = self.compiled.compilation_id
            handle.attrs["claim_id"] = self.compiled.claim_id
            handle.create_group("steps")
        self._write_xdmf()

    def _validate(self, handle):
        if (
            int(handle.attrs.get("schema_version", -1)) != _OUTPUT_SCHEMA_VERSION
            or handle.attrs.get("output_id") != self.output_id
            or handle.attrs.get("compilation_id") != self.compiled.compilation_id
        ):
            raise ValueError("MPM output archive identity is incompatible.")

    def append(self, state: MPMRuntimeState, /):
        if not isinstance(state, MPMRuntimeState):
            raise TypeError("state must be MPMRuntimeState.")
        h5py = _h5py()
        path = Path(self.hdf5_path)
        if not path.exists():
            self.initialize()
        step = int(np.asarray(state.accepted_step))
        name = f"{step:08d}"
        with h5py.File(path, "a") as handle:
            self._validate(handle)
            steps = handle["steps"]
            if name in steps:
                raise ValueError(f"MPM output step {step} already exists.")
            group = steps.create_group(name)
            group.attrs["time_hex"] = float(np.asarray(state.time)).hex()
            group.attrs["topology_generation"] = int(
                np.asarray(state.topology_generation)
            )
            particles = state.particles
            for field, value in (
                ("position", particles.position),
                ("velocity", particles.velocity),
                ("deformation_gradient", particles.deformation_gradient),
                ("affine_velocity", particles.affine_velocity),
                ("reference_volume", particles.reference_volume),
                ("first_piola", particles.first_piola),
                (
                    "reference_energy_density",
                    particles.reference_energy_density,
                ),
                ("maximum_wave_speed", particles.maximum_wave_speed),
                ("material_state", particles.material_state),
            ):
                group.create_dataset(field, data=np.asarray(value))
            visualization_dimension = max(2, particles.position.shape[1])
            visualization_padding = visualization_dimension - particles.position.shape[1]
            group.create_dataset(
                "position_visualization",
                data=np.pad(
                    np.asarray(particles.position),
                    ((0, 0), (0, visualization_padding)),
                ),
            )
            group.create_dataset(
                "velocity_visualization",
                data=np.pad(
                    np.asarray(particles.velocity),
                    ((0, 0), (0, visualization_padding)),
                ),
            )
            group.create_dataset("material_slots", data=np.asarray(state.material_slots))
            group.create_dataset("body_ids", data=np.asarray(state.body_ids))
            group.create_dataset(
                "velocity_field_slots", data=np.asarray(state.velocity_field_slots)
            )
            handle.flush()
        self._write_xdmf()
        return name

    def manifest(self):
        h5py = _h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            self._validate(handle)
            names = sorted(handle["steps"].keys())
            if not names:
                count = 0
                time_hex = float("nan").hex()
            else:
                count = len(names)
                time_hex = str(handle["steps"][names[-1]].attrs["time_hex"])
        identifier = canonical_fingerprint(
            {
                "kind": "mpm-output-manifest",
                "schema_version": _OUTPUT_SCHEMA_VERSION,
                "output_id": self.output_id,
                "compilation": self.compiled.compilation_id,
                "accepted_steps": count,
                "last_time_hex": time_hex,
            }
        )
        return MPMOutputManifest(
            _OUTPUT_SCHEMA_VERSION,
            self.output_id,
            self.compiled.compilation_id,
            count,
            time_hex,
            identifier,
        )

    def _write_xdmf(self):
        hdf5 = Path(self.hdf5_path)
        if not hdf5.exists():
            return
        h5py = _h5py()
        with h5py.File(hdf5, "r") as handle:
            names = sorted(handle["steps"].keys())
            records = []
            for name in names:
                group = handle["steps"][name]
                position_shape = group["position_visualization"].shape
                particle_count = group["position"].shape[0]
                visualization_dimension = position_shape[1]
                time_value = float.fromhex(str(group.attrs["time_hex"]))
                geometry = "XY" if visualization_dimension == 2 else "XYZ"
                attributes = []
                for field, source, kind, components in (
                    (
                        "velocity",
                        "velocity_visualization",
                        "Vector",
                        visualization_dimension,
                    ),
                    ("reference_energy_density", "reference_energy_density", "Scalar", 1),
                    ("maximum_wave_speed", "maximum_wave_speed", "Scalar", 1),
                ):
                    data_shape = (
                        f"{particle_count}"
                        if components == 1
                        else f"{particle_count} {components}"
                    )
                    attributes.append(
                        f'<Attribute Name="{field}" AttributeType="{kind}" Center="Node">'
                        f'<DataItem Dimensions="{data_shape}" Format="HDF">'
                        f"{hdf5.name}:/steps/{name}/{source}</DataItem></Attribute>"
                    )
                records.append(
                    f'<Grid Name="step-{name}" GridType="Uniform">'
                    f'<Time Value="{time_value:.17g}"/>'
                    f'<Topology TopologyType="Polyvertex" NumberOfElements="{particle_count}"/>'
                    f'<Geometry GeometryType="{geometry}">'
                    f'<DataItem Dimensions="{particle_count} '
                    f'{visualization_dimension}" Format="HDF">'
                    f"{hdf5.name}:/steps/{name}/position_visualization</DataItem></Geometry>"
                    + "".join(attributes)
                    + "</Grid>"
                )
        content = (
            '<?xml version="1.0"?><Xdmf Version="3.0"><Domain>'
            '<Grid Name="MPM" GridType="Collection" CollectionType="Temporal">'
            + "".join(records)
            + "</Grid></Domain></Xdmf>\n"
        )
        path = Path(self.xdmf_path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)

    def write_vtk_snapshot(self, path: str | Path, state: MPMRuntimeState, /):
        meshio = import_module("meshio")
        points = np.asarray(state.particles.position)
        if points.shape[1] < 3:
            points = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
        cells = [("vertex", np.arange(points.shape[0], dtype=np.int32)[:, None])]
        point_data = {
            "velocity": np.pad(
                np.asarray(state.particles.velocity),
                ((0, 0), (0, 3 - state.particles.velocity.shape[1])),
            ),
            "material_slot": np.asarray(state.material_slots),
            "body_id": np.asarray(state.body_ids),
            "velocity_field_slot": np.asarray(state.velocity_field_slots),
        }
        meshio.write(Path(path), meshio.Mesh(points, cells, point_data=point_data))
        return Path(path)


class MPMBoundedOutputBuffer:
    """Host-side accepted-output buffer with explicit backpressure."""

    def __init__(self, maximum_items: int):
        maximum = int(maximum_items)
        if maximum <= 0:
            raise ValueError("maximum_items must be positive.")
        self.maximum_items = maximum
        self._queue = deque()

    @property
    def size(self):
        return len(self._queue)

    def push(self, state: MPMRuntimeState):
        if len(self._queue) >= self.maximum_items:
            raise BufferError("MPM output backpressure capacity reached.")
        self._queue.append(state)

    def pop(self):
        if not self._queue:
            raise IndexError("MPM output buffer is empty.")
        return self._queue.popleft()

    def manifest(self):
        return json.dumps(
            {
                "maximum_items": self.maximum_items,
                "size": self.size,
                "accepted_steps": [
                    int(np.asarray(value.accepted_step)) for value in self._queue
                ],
            },
            sort_keys=True,
        )


__all__ = [
    "MPMBoundedOutputBuffer",
    "MPMOutputManifest",
    "MPMOutputPlan",
]
