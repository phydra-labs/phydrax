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
from .._cell_complex import PolyhedralConnectivity
from .._cell_mesh import CellBlock, CellMesh
from ._generic import FiniteElementCoordinateSpec
from ._hp_runtime import FiniteElementHPEpoch
from ._mortar import FiniteElementMortarMetricData, FiniteElementMortarPlan
from ._reference import lagrange_element


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


def _canonical_source_metadata(
    metadata: Mapping[str, object], /
) -> tuple[tuple[str, str], ...]:
    def portable(value):
        if isinstance(value, Mapping):
            return {
                str(key): portable(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, np.ndarray):
            return portable(value.tolist())
        if isinstance(value, np.generic):
            return portable(value.item())
        if isinstance(value, (tuple, list)):
            return [portable(item) for item in value]
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(
            "Finite-element source metadata must contain portable deterministic values."
        )

    return tuple(
        (
            str(name),
            json.dumps(portable(value), sort_keys=True, separators=(",", ":")),
        )
        for name, value in sorted(metadata.items(), key=lambda pair: str(pair[0]))
    )


def _entity_counts(
    values: Mapping[str, int] | Sequence[tuple[str, int]], /
) -> tuple[tuple[str, int], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized = tuple(sorted((str(name), int(count)) for name, count in items))
    if any(not name or count < 0 for name, count in normalized):
        raise ValueError("Import entity counts must be named and non-negative.")
    return normalized


def _group_values(
    groups: Mapping[str, Sequence[int]], /
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                str(name),
                tuple(sorted(int(value) for value in entities)),
            )
            for name, entities in groups.items()
        )
    )


def _named_group(
    groups: tuple[tuple[str, tuple[int, ...]], ...], name: str, kind: str, /
) -> tuple[int, ...]:
    requested = str(name)
    for group_name, entities in groups:
        if group_name == requested:
            return entities
    raise ValueError(f"Unknown imported {kind} group {requested!r}.")


class FiniteElementMeshImportReport(StrictModule, NonTrainableState):
    block_names: tuple[str, ...] = eqx.field(static=True)
    cell_kinds: tuple[str, ...] = eqx.field(static=True)
    geometry_orders: tuple[int, ...] = eqx.field(static=True)
    volume_names: tuple[str, ...] = eqx.field(static=True)
    boundary_names: tuple[str, ...] = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    curved: bool = eqx.field(static=True)
    source_path: str = eqx.field(static=True)
    source_format: str = eqx.field(static=True)
    source_metadata: tuple[tuple[str, str], ...] = eqx.field(static=True)
    lossy: bool = eqx.field(static=True)
    losses: tuple[str, ...] = eqx.field(static=True)
    source_entity_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    imported_entity_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    dropped_entity_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        block_names: Sequence[str],
        cell_kinds: Sequence[str],
        geometry_orders: Sequence[int],
        boundary_names: Sequence[str],
        coordinate_count: int,
        /,
        *,
        volume_names: Sequence[str] = (),
        source_path: str = "",
        source_format: str = "",
        source_metadata: Mapping[str, object] | Sequence[tuple[str, str]] = (),
        losses: Sequence[str] = (),
        source_entity_counts: Mapping[str, int] | Sequence[tuple[str, int]] = (),
        imported_entity_counts: Mapping[str, int] | Sequence[tuple[str, int]] = (),
        dropped_entity_counts: Mapping[str, int] | Sequence[tuple[str, int]] = (),
    ):
        names = tuple(str(value) for value in block_names)
        kinds = tuple(str(value) for value in cell_kinds)
        orders = tuple(int(value) for value in geometry_orders)
        volumes = tuple(sorted(str(value) for value in volume_names))
        boundaries = tuple(sorted(str(value) for value in boundary_names))
        path = str(source_path)
        format_ = str(source_format).lower()
        metadata = (
            _canonical_source_metadata(source_metadata)
            if isinstance(source_metadata, Mapping)
            else tuple(sorted((str(name), str(value)) for name, value in source_metadata))
        )
        losses_ = tuple(sorted(set(str(value) for value in losses)))
        source_counts = _entity_counts(source_entity_counts)
        imported_counts = _entity_counts(imported_entity_counts)
        dropped_counts = _entity_counts(dropped_entity_counts)
        if (
            not names
            or len(names) != len(kinds)
            or len(names) != len(orders)
            or any(value < 1 for value in orders)
            or int(coordinate_count) <= 0
            or any(not value for value in (*volumes, *boundaries, *losses_))
            or any(not key for key, _value in metadata)
        ):
            raise ValueError("Finite-element mesh import report is inconsistent.")
        source_by_name = dict(source_counts)
        imported_by_name = dict(imported_counts)
        dropped_by_name = dict(dropped_counts)
        count_names = (
            source_by_name.keys() | imported_by_name.keys() | dropped_by_name.keys()
        )
        if any(
            source_by_name.get(name, 0)
            != imported_by_name.get(name, 0) + dropped_by_name.get(name, 0)
            for name in count_names
        ) or (any(dropped_by_name.values()) and not losses_):
            raise ValueError(
                "Finite-element import entity loss accounting is inconsistent."
            )
        self.block_names = names
        self.cell_kinds = kinds
        self.geometry_orders = orders
        self.volume_names = volumes
        self.boundary_names = boundaries
        self.coordinate_count = int(coordinate_count)
        self.curved = any(value > 1 for value in orders)
        self.source_path = path
        self.source_format = format_
        self.source_metadata = metadata
        self.lossy = bool(losses_)
        self.losses = losses_
        self.source_entity_counts = source_counts
        self.imported_entity_counts = imported_counts
        self.dropped_entity_counts = dropped_counts
        self.report_id = canonical_fingerprint(
            {
                "kind": "finite-element-mesh-import-report",
                "blocks": names,
                "cell_kinds": kinds,
                "geometry_orders": orders,
                "volumes": volumes,
                "boundaries": boundaries,
                "coordinate_count": int(coordinate_count),
                "source_path": path,
                "source_format": format_,
                "source_metadata": metadata,
                "losses": losses_,
                "source_entity_counts": source_counts,
                "imported_entity_counts": imported_counts,
                "dropped_entity_counts": dropped_counts,
            }
        )


class FiniteElementMeshImport(StrictModule, NonTrainableState):
    mesh: CellMesh
    coordinate_spec: FiniteElementCoordinateSpec
    volume_groups: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    boundary_groups: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    report: FiniteElementMeshImportReport
    import_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        coordinate_spec: FiniteElementCoordinateSpec,
        boundary_groups: Mapping[str, Sequence[int]],
        report: FiniteElementMeshImportReport,
        /,
        *,
        volume_groups: Mapping[str, Sequence[int]] | None = None,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if not isinstance(coordinate_spec, FiniteElementCoordinateSpec):
            raise TypeError("coordinate_spec must be FiniteElementCoordinateSpec.")
        volumes = _group_values({} if volume_groups is None else volume_groups)
        boundaries = _group_values(boundary_groups)
        if not isinstance(report, FiniteElementMeshImportReport):
            raise TypeError("report must be FiniteElementMeshImportReport.")
        if (
            report.block_names != tuple(block.name for block in mesh.blocks)
            or report.cell_kinds != tuple(block.cell_kind for block in mesh.blocks)
            or report.volume_names != tuple(name for name, _entities in volumes)
            or report.boundary_names != tuple(name for name, _entities in boundaries)
        ):
            raise ValueError("Imported entity groups do not match the import report.")
        if any(
            not name or not entities or len(set(entities)) != len(entities)
            for name, entities in (*volumes, *boundaries)
        ):
            raise ValueError("Imported entity groups must be nonempty and unique.")
        cell_count = sum(block.cell_count for block in mesh.blocks)
        facet_count = mesh.topology.entity_sets[mesh.topological_dimension - 1].count
        if any(
            value < 0 or value >= cell_count
            for _name, entities in volumes
            for value in entities
        ):
            raise ValueError("Imported volume group contains an invalid cell index.")
        if any(
            value < 0 or value >= facet_count
            for _name, entities in boundaries
            for value in entities
        ):
            raise ValueError("Imported boundary group contains an invalid facet index.")
        self.mesh = mesh
        self.coordinate_spec = coordinate_spec
        self.volume_groups = volumes
        self.boundary_groups = boundaries
        self.report = report
        self.import_id = canonical_fingerprint(
            {
                "kind": "finite-element-mesh-import",
                "mesh": mesh.mesh_id,
                "coordinate_spec": coordinate_spec.coordinate_spec_id,
                "volumes": volumes,
                "boundaries": boundaries,
                "report": report.report_id,
            }
        )

    def volume_cells(self, name: str, /) -> tuple[int, ...]:
        return _named_group(self.volume_groups, name, "volume")

    def boundary_facets(self, name: str, /) -> tuple[int, ...]:
        return _named_group(self.boundary_groups, name, "boundary")


_MESHIO_VOLUME_TYPES = {
    "triangle": ("triangle", 1, 3),
    "triangle6": ("triangle", 2, 3),
    "quad": ("quadrilateral", 1, 4),
    "quad9": ("quadrilateral", 2, 4),
    "tetra": ("tetrahedron", 1, 4),
    "tetra10": ("tetrahedron", 2, 4),
    "hexahedron": ("hexahedron", 1, 8),
    "hexahedron27": ("hexahedron", 2, 8),
    "wedge": ("prism", 1, 6),
    "wedge18": ("prism", 2, 6),
    "pyramid": ("pyramid", 1, 5),
    "pyramid14": ("pyramid", 2, 5),
}


def _meshio_reference_nodes(cell_type: str, /) -> np.ndarray:
    values = {
        "triangle": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        "triangle6": (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5),
        ),
        "quad": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        "quad9": (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (1.0, 0.5),
            (0.5, 1.0),
            (0.0, 0.5),
            (0.5, 0.5),
        ),
        "tetra": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        "tetra10": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        "hexahedron": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        "hexahedron27": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.5, 0.0, 1.0),
            (1.0, 0.5, 1.0),
            (0.5, 1.0, 1.0),
            (0.0, 0.5, 1.0),
            (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.5),
            (1.0, 1.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 1.0),
            (0.5, 0.0, 0.5),
            (1.0, 0.5, 0.5),
            (0.5, 1.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
        "wedge": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        "wedge18": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0),
            (0.5, 0.0, 1.0),
            (0.5, 0.5, 1.0),
            (0.0, 0.5, 1.0),
            (0.0, 0.0, 0.5),
            (1.0, 0.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.5),
            (0.0, 0.5, 0.5),
        ),
        "pyramid": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
        ),
        "pyramid14": (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
            (0.5, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (0.5, 1.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.25, 0.25, 0.5),
            (0.75, 0.25, 0.5),
            (0.75, 0.75, 0.5),
            (0.25, 0.75, 0.5),
            (0.5, 0.5, 0.0),
        ),
    }
    if cell_type not in values:
        raise ValueError(f"Unsupported high-order mesh cell type {cell_type!r}.")
    return np.asarray(values[cell_type], dtype=float)


def _geometry_permutation(cell_type: str, cell_kind: str, order: int, /) -> np.ndarray:
    source = _meshio_reference_nodes(cell_type)
    target = np.asarray(lagrange_element(cell_kind, order).reference_nodes)
    if source.shape != target.shape:
        raise ValueError("Imported and Phydrax geometry node counts differ.")
    permutation = []
    for point in target:
        matches = np.flatnonzero(np.max(np.abs(source - point), axis=1) <= 2.0e-12)
        if matches.size != 1:
            raise ValueError("High-order geometry node ordering is ambiguous.")
        permutation.append(int(matches[0]))
    return np.asarray(permutation, dtype=np.int32)


def read_finite_element_mesh(path: str | Path, /) -> FiniteElementMeshImport:
    import meshio

    source_path = Path(path)
    source = meshio.read(source_path)
    volume_blocks = []
    for source_index, cell_block in enumerate(source.cells):
        if cell_block.type in _MESHIO_VOLUME_TYPES:
            volume_blocks.append((source_index, cell_block))
    if not volume_blocks:
        raise ValueError("Mesh contains no supported finite-element volume cells.")
    topological_dimensions = {
        2 if _MESHIO_VOLUME_TYPES[block.type][0] in ("triangle", "quadrilateral") else 3
        for _index, block in volume_blocks
    }
    if len(topological_dimensions) != 1:
        raise ValueError("Imported volume blocks must share one topological dimension.")
    topological_dimension = topological_dimensions.pop()
    points = np.asarray(source.points, dtype=float)
    ambient_dimension = points.shape[1]
    while ambient_dimension > topological_dimension and np.allclose(
        points[:, ambient_dimension - 1], 0.0
    ):
        ambient_dimension -= 1
    points = points[:, :ambient_dimension]

    corner_ids = sorted(
        {
            int(value)
            for _source_index, cell_block in volume_blocks
            for value in np.asarray(cell_block.data)[
                :, : _MESHIO_VOLUME_TYPES[cell_block.type][2]
            ].reshape((-1,))
        }
    )
    compact = {value: index for index, value in enumerate(corner_ids)}
    mesh_coordinates = points[corner_ids]
    blocks = []
    coordinate_elements = {}
    coordinate_routes = {}
    volume_routes: dict[int, np.ndarray] = {}
    next_cell_id = 0
    for block_index, (source_index, cell_block) in enumerate(volume_blocks):
        cell_kind, order, corner_count = _MESHIO_VOLUME_TYPES[cell_block.type]
        name = f"{cell_kind}_{block_index}"
        data = np.asarray(cell_block.data, dtype=np.int32)
        corners = np.asarray(
            [[compact[int(value)] for value in row[:corner_count]] for row in data],
            dtype=np.int32,
        )
        global_ids = np.arange(next_cell_id, next_cell_id + data.shape[0], dtype=np.int64)
        volume_routes[source_index] = global_ids
        next_cell_id += data.shape[0]
        blocks.append(
            CellBlock(
                name,
                cell_kind,
                corners,
                global_ids=global_ids,
            )
        )
        element = lagrange_element(cell_kind, order)
        permutation = _geometry_permutation(cell_block.type, cell_kind, order)
        coordinate_elements[name] = element
        coordinate_routes[name] = data[:, permutation]
    mesh = CellMesh(mesh_coordinates, tuple(blocks))
    coordinate_spec = FiniteElementCoordinateSpec(
        coordinate_elements,
        coordinate_routes,
        points,
    )

    facet_vertices = (
        np.asarray(mesh.connectivity.edges)
        if topological_dimension == 2
        else np.asarray(mesh.connectivity.faces)
    )
    if isinstance(mesh.connectivity, PolyhedralConnectivity):
        arities = np.asarray(mesh.connectivity.face_arities, dtype=np.int32)
        facets_by_key = {
            tuple(sorted(int(value) for value in vertices[: int(arities[index])])): index
            for index, vertices in enumerate(facet_vertices)
        }
    else:
        facets_by_key = {
            tuple(sorted(int(value) for value in vertices)): index
            for index, vertices in enumerate(facet_vertices)
        }
    volume_groups: dict[str, set[int]] = {}
    boundary_groups: dict[str, set[int]] = {}
    selected_source_facets: set[tuple[int, int]] = set()
    for group_name, selections in source.cell_sets.items():
        group = str(group_name)
        selected_volumes = volume_groups.setdefault(group, set())
        selected_facets = boundary_groups.setdefault(group, set())
        for source_index, selected in enumerate(selections):
            if selected is None or len(selected) == 0:
                continue
            selected_indices = np.asarray(selected, dtype=np.int32)
            if source_index in volume_routes:
                selected_volumes.update(
                    int(value) for value in volume_routes[source_index][selected_indices]
                )
                continue
            cell_block = source.cells[source_index]
            boundary_arity = (
                2
                if topological_dimension == 2 and cell_block.type in ("line", "line3")
                else 3
                if topological_dimension == 3
                and cell_block.type in ("triangle", "triangle6")
                else 4
                if topological_dimension == 3 and cell_block.type in ("quad", "quad9")
                else 0
            )
            if boundary_arity == 0:
                continue
            for selected_index in selected_indices:
                row = np.asarray(cell_block.data)[int(selected_index)]
                source_vertices = tuple(int(value) for value in row[:boundary_arity])
                if any(value not in compact for value in source_vertices):
                    raise ValueError("Boundary group references a non-volume vertex.")
                compact_vertices = tuple(
                    sorted(compact[value] for value in source_vertices)
                )
                if compact_vertices not in facets_by_key:
                    raise ValueError("Boundary group references a non-volume facet.")
                selected_facets.add(facets_by_key[compact_vertices])
                selected_source_facets.add((source_index, int(selected_index)))
    normalized_volumes = {
        name: tuple(sorted(values)) for name, values in volume_groups.items() if values
    }
    normalized_boundaries = {
        name: tuple(sorted(values)) for name, values in boundary_groups.items() if values
    }

    source_volume_count = sum(
        np.asarray(block.data).shape[0] for _index, block in volume_blocks
    )
    source_facet_count = 0
    for cell_block in source.cells:
        is_facet = (
            topological_dimension == 2 and cell_block.type in ("line", "line3")
        ) or (
            topological_dimension == 3
            and cell_block.type in ("triangle", "triangle6", "quad", "quad9")
        )
        if is_facet:
            source_facet_count += np.asarray(cell_block.data).shape[0]
    source_cell_count = sum(np.asarray(block.data).shape[0] for block in source.cells)
    imported_facet_count = len(selected_source_facets)
    other_cell_count = source_cell_count - source_volume_count - source_facet_count
    dropped_facet_count = source_facet_count - imported_facet_count
    losses = []
    if dropped_facet_count:
        losses.append(f"dropped_ungrouped_facet_cells:{dropped_facet_count}")
    if other_cell_count:
        losses.append(f"dropped_unsupported_cells:{other_cell_count}")
    if ambient_dimension != np.asarray(source.points).shape[1]:
        losses.append(
            "dropped_zero_coordinate_axes:"
            f"{np.asarray(source.points).shape[1] - ambient_dimension}"
        )
    for name, values in (
        ("point_data", tuple(sorted(str(value) for value in source.point_data))),
        ("cell_data", tuple(sorted(str(value) for value in source.cell_data))),
        ("point_sets", tuple(sorted(str(value) for value in source.point_sets))),
    ):
        if values:
            losses.append(f"{name}_not_imported:{','.join(values)}")
    if source.gmsh_periodic:
        losses.append(f"gmsh_periodic_not_imported:{len(source.gmsh_periodic)}")
    metadata = {
        "field_data": source.field_data,
        "info": source.info,
        "gmsh_periodic": source.gmsh_periodic,
        "cell_set_names": tuple(sorted(str(value) for value in source.cell_sets)),
        "point_set_names": tuple(sorted(str(value) for value in source.point_sets)),
        "point_data_names": tuple(sorted(str(value) for value in source.point_data)),
        "cell_data_names": tuple(sorted(str(value) for value in source.cell_data)),
    }
    source_counts = {
        "points": points.shape[0],
        "volume_cells": source_volume_count,
        "facet_cells": source_facet_count,
        "other_cells": other_cell_count,
    }
    imported_counts = {
        "points": points.shape[0],
        "volume_cells": source_volume_count,
        "facet_cells": imported_facet_count,
        "other_cells": 0,
    }
    dropped_counts = {
        "points": 0,
        "volume_cells": 0,
        "facet_cells": dropped_facet_count,
        "other_cells": other_cell_count,
    }
    report = FiniteElementMeshImportReport(
        tuple(block.name for block in blocks),
        tuple(block.cell_kind for block in blocks),
        tuple(_MESHIO_VOLUME_TYPES[block.type][1] for _index, block in volume_blocks),
        tuple(normalized_boundaries),
        points.shape[0],
        volume_names=tuple(normalized_volumes),
        source_path=str(source_path),
        source_format=source_path.suffix.lower().lstrip("."),
        source_metadata=metadata,
        losses=losses,
        source_entity_counts=source_counts,
        imported_entity_counts=imported_counts,
        dropped_entity_counts=dropped_counts,
    )
    return FiniteElementMeshImport(
        mesh,
        coordinate_spec,
        normalized_boundaries,
        report,
        volume_groups=normalized_volumes,
    )


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
    "FiniteElementMeshImport",
    "FiniteElementMeshImportReport",
    "FusedMortarAction",
    "FusedTensorTransfer",
    "HPMixedPrecisionPolicy",
    "HPWorksetMemoryPlan",
    "PersistentSemanticCache",
    "read_exodus_high_order_arrays",
    "read_finite_element_mesh",
    "write_adaptive_vtk",
    "write_adaptive_xdmf",
    "write_hp_forest",
]
