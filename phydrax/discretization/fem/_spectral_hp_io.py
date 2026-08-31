#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_mesh import CellBlock, CellMesh
from ._hp_runtime import FiniteElementHPEpoch
from ._mortar import FiniteElementMortarMetricData, FiniteElementMortarPlan


class PersistentSemanticCache(StrictModule, NonTrainableState):
    directory: str = eqx.field(static=True)

    def __init__(self, directory: str | Path, /):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.directory = str(path)

    def store(
        self,
        semantic_id: str,
        arrays: Mapping[str, ArrayLike],
        metadata: Mapping[str, object],
        /,
    ) -> Path:
        identifier = str(semantic_id)
        if not identifier:
            raise ValueError("Semantic cache identity must be non-empty.")
        path = Path(self.directory) / f"{identifier}.npz"
        payload = {str(name): np.asarray(value) for name, value in arrays.items()}
        payload["metadata"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
        np.savez(path, allow_pickle=False, **payload)
        return path

    def load(self, semantic_id: str, /) -> tuple[dict[str, Array], dict[str, object]]:
        path = Path(self.directory) / f"{semantic_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            arrays = {
                name: jnp.asarray(archive[name])
                for name in archive.files
                if name != "metadata"
            }
        return arrays, metadata


class HeterogeneousSignatureSchedule(StrictModule, NonTrainableState):
    ordered_signatures: tuple[str, ...] = eqx.field(static=True)
    bucket_routes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, signatures: Sequence[str], costs: ArrayLike, /):
        signatures_ = tuple(str(value) for value in signatures)
        costs_ = np.asarray(costs, dtype=float)
        if (
            costs_.shape != (len(signatures_),)
            or any(not value for value in signatures_)
            or np.any(costs_ < 0.0)
        ):
            raise ValueError("Signature identities and costs are incompatible.")
        unique = sorted(
            set(signatures_),
            key=lambda value: (
                -float(np.sum(costs_[np.asarray(signatures_) == value])),
                value,
            ),
        )
        routes = tuple(
            tuple(np.flatnonzero(np.asarray(signatures_) == value).tolist())
            for value in unique
        )
        self.ordered_signatures = tuple(unique)
        self.bucket_routes = routes
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "heterogeneous-signature-schedule",
                "signatures": list(signatures_),
                "costs": costs_.tolist(),
            }
        )


class FusedMortarAction(StrictModule, NonTrainableState):
    mortar: FiniteElementMortarPlan

    def apply(
        self,
        left: ArrayLike,
        right: ArrayLike,
        metric: FiniteElementMortarMetricData,
        flux: Callable,
        /,
    ) -> tuple[Array, Array]:
        left_q = self.mortar.interpolate_left(left)
        right_q = self.mortar.interpolate_right(right)
        normal = metric.owner_scaled_normals / jnp.linalg.norm(
            metric.owner_scaled_normals, axis=-1, keepdims=True
        )
        numerical_flux = flux(left_q, right_q, metric.physical_coordinates, normal)
        return self.mortar.conservative_flux_contributions(numerical_flux, metric)


class FusedTensorTransfer(StrictModule, NonTrainableState):
    factors: tuple[Array, ...]

    def __init__(self, factors: Sequence[ArrayLike], /):
        factors_ = tuple(jnp.asarray(value) for value in factors)
        if not factors_ or any(value.ndim != 2 for value in factors_):
            raise ValueError("Fused tensor transfers require rank-2 axis factors.")
        self.factors = factors_

    def apply(self, values: ArrayLike, /) -> Array:
        result = jnp.asarray(values)
        for axis, factor in enumerate(self.factors):
            result = jnp.moveaxis(result, axis, 0)
            result = oe.contract("ij,j...->i...", factor, result)
            result = jnp.moveaxis(result, 0, axis)
        return result

    def pullback(self, dual: ArrayLike, /) -> Array:
        result = jnp.asarray(dual)
        for axis in reversed(range(len(self.factors))):
            result = jnp.moveaxis(result, axis, 0)
            result = oe.contract("ji,j...->i...", self.factors[axis], result)
            result = jnp.moveaxis(result, 0, axis)
        return result


class HPMixedPrecisionPolicy(StrictModule, NonTrainableState):
    storage_dtype: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)

    def __init__(
        self, storage_dtype: str, compute_dtype: str, accumulation_dtype: str, /
    ):
        dtypes = tuple(
            np.dtype(value)
            for value in (storage_dtype, compute_dtype, accumulation_dtype)
        )
        if any(not np.issubdtype(value, np.inexact) for value in dtypes):
            raise ValueError("hp mixed precision requires inexact dtypes.")
        if (
            dtypes[2].itemsize < dtypes[1].itemsize
            or dtypes[1].itemsize < dtypes[0].itemsize
        ):
            raise ValueError(
                "hp precision must not narrow from storage to compute to accumulation."
            )
        self.storage_dtype, self.compute_dtype, self.accumulation_dtype = tuple(
            value.name for value in dtypes
        )


class HPWorksetMemoryPlan(StrictModule, NonTrainableState):
    entity_bytes: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    planned_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        local_widths: Sequence[int],
        component_count: int,
        dtype: str,
        maximum_bytes: int,
        /,
    ):
        widths = tuple(int(value) for value in local_widths)
        components = int(component_count)
        budget = int(maximum_bytes)
        if not widths or min(widths) <= 0 or components <= 0 or budget <= 0:
            raise ValueError("Workset width, components, or memory budget is invalid.")
        entity_bytes = sum(widths) * components * np.dtype(dtype).itemsize
        capacity = budget // entity_bytes
        if capacity < 1:
            raise ValueError("Memory budget cannot hold one hp workset entity.")
        self.entity_bytes = entity_bytes
        self.capacity = capacity
        self.planned_bytes = capacity * entity_bytes


def write_adaptive_vtk(
    path: str | Path,
    epoch: FiniteElementHPEpoch,
    point_values: ArrayLike | None = None,
    /,
) -> None:
    mesh = epoch.mesh
    points = np.asarray(mesh.coordinates)
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks)
    )
    cell_kind = epoch.topology.cell_kind
    vtk_type = 9 if cell_kind == "quadrilateral" else 12
    with Path(path).open("w") as stream:
        stream.write(
            "# vtk DataFile Version 3.0\nPhydrax adaptive hp\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        )
        stream.write(f"POINTS {points.shape[0]} float\n")
        padded = np.pad(points, ((0, 0), (0, max(0, 3 - points.shape[1]))))
        for point in padded:
            stream.write(" ".join(str(float(value)) for value in point[:3]) + "\n")
        stream.write(f"CELLS {cells.shape[0]} {cells.size + cells.shape[0]}\n")
        for cell in cells:
            stream.write(
                f"{cell.size} " + " ".join(str(int(value)) for value in cell) + "\n"
            )
        stream.write(
            f"CELL_TYPES {cells.shape[0]}\n"
            + "\n".join((str(vtk_type),) * cells.shape[0])
            + "\n"
        )
        if point_values is not None:
            values = np.asarray(point_values)
            if values.shape[0] != points.shape[0]:
                raise ValueError("VTK point values must match mesh points.")
            stream.write(
                f"POINT_DATA {points.shape[0]}\nSCALARS field float 1\nLOOKUP_TABLE default\n"
            )
            for value in values.reshape((values.shape[0], -1))[:, 0]:
                stream.write(f"{float(value)}\n")


def write_adaptive_xdmf(path: str | Path, epoch: FiniteElementHPEpoch, /) -> None:
    mesh = epoch.mesh
    points = np.asarray(mesh.coordinates)
    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks)
    )
    topology = (
        "Quadrilateral" if epoch.topology.cell_kind == "quadrilateral" else "Hexahedron"
    )
    geometry = "XY" if points.shape[1] == 2 else "XYZ"
    cell_text = " ".join(str(int(value)) for value in cells.reshape(-1))
    point_text = " ".join(str(float(value)) for value in points.reshape(-1))
    content = (
        f'<Xdmf Version="3.0"><Domain><Grid Name="hp">'
        f'<Topology TopologyType="{topology}" NumberOfElements="{cells.shape[0]}">'
        f'<DataItem Dimensions="{cells.shape[0]} {cells.shape[1]}" Format="XML">'
        f"{cell_text}</DataItem></Topology>"
        f'<Geometry GeometryType="{geometry}">'
        f'<DataItem Dimensions="{points.shape[0]} {points.shape[1]}" Format="XML">'
        f"{point_text}</DataItem></Geometry></Grid></Domain></Xdmf>\n"
    )
    Path(path).write_text(content)


def write_hp_forest(path: str | Path, epoch: FiniteElementHPEpoch, /) -> None:
    topology = epoch.topology
    payload = {
        "cell_kind": topology.cell_kind,
        "epoch_id": epoch.epoch_id,
        "cell_global_ids": np.asarray(topology.cell_global_ids).tolist(),
        "allocated": np.asarray(topology.allocated).tolist(),
        "active": np.asarray(topology.active).tolist(),
        "degrees": np.asarray(topology.cell_degrees).tolist(),
        "root_ids": np.asarray(topology.root_cell_ids).tolist(),
        "path_codes": np.asarray(topology.path_codes).tolist(),
        "levels": np.asarray(topology.levels).tolist(),
        "interface_ids": list(epoch.interfaces.interface_ids),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def read_gmsh_high_order(path: str | Path, /) -> CellMesh:
    lines = Path(path).read_text().splitlines()
    node_start = lines.index("$Nodes")
    node_count = int(lines[node_start + 1])
    nodes = {}
    for line in lines[node_start + 2 : node_start + 2 + node_count]:
        values = line.split()
        nodes[int(values[0])] = tuple(float(value) for value in values[1:4])
    element_start = lines.index("$Elements")
    element_count = int(lines[element_start + 1])
    cells = []
    kind = None
    for line in lines[element_start + 2 : element_start + 2 + element_count]:
        values = [int(value) for value in line.split()]
        element_type = values[1]
        tag_count = values[2]
        connectivity = values[3 + tag_count :]
        if element_type in (3, 10):
            kind = "quadrilateral"
            cells.append(connectivity[:4])
        elif element_type in (5, 12):
            kind = "hexahedron"
            cells.append(connectivity[:8])
    if kind is None or not cells:
        raise ValueError("Gmsh file contains no supported quad/hex elements.")
    ordered_ids = sorted(nodes)
    local = {value: index for index, value in enumerate(ordered_ids)}
    coordinates = np.asarray([nodes[value] for value in ordered_ids])
    if np.allclose(coordinates[:, 2], 0.0):
        coordinates = coordinates[:, :2]
    routes = np.asarray(
        [[local[value] for value in cell] for cell in cells], dtype=np.int32
    )
    return CellMesh(coordinates, (CellBlock("gmsh", kind, routes),))


def read_exodus_high_order_arrays(
    coordinates: ArrayLike, connectivity: ArrayLike, cell_kind: str, /
) -> CellMesh:
    return CellMesh(
        jnp.asarray(coordinates),
        (
            CellBlock(
                "exodus", str(cell_kind), jnp.asarray(connectivity, dtype=jnp.int32)
            ),
        ),
    )


__all__ = [
    "FusedMortarAction",
    "FusedTensorTransfer",
    "HeterogeneousSignatureSchedule",
    "HPMixedPrecisionPolicy",
    "HPWorksetMemoryPlan",
    "PersistentSemanticCache",
    "read_exodus_high_order_arrays",
    "read_gmsh_high_order",
    "write_adaptive_vtk",
    "write_adaptive_xdmf",
    "write_hp_forest",
]
