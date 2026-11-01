#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Triangle-to-prism and quadrilateral-to-hex sweep construction."""

from __future__ import annotations

from enum import StrEnum
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometryRestrictionSource,
    swept_coordinate_element,
)
from ..discretization._nested_reference_types import (
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._reference import FiniteElementSpec
from ..typing import checked
from ._controls import LayerSchedule


class SweepMapKind(StrEnum):
    EXTRUSION = "extrusion"
    REVOLUTION = "revolution"
    FRAMES = "frames"


@final
class SweepControl(StrictModule, NonTrainableState):
    """A physical station schedule and an extrusion, revolution, or frame map."""

    schedule: LayerSchedule
    origin: Array
    axis: Array
    frames: Array
    translations: Array
    kind: SweepMapKind = eqx.field(static=True)
    map_id: str = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        kind: SweepMapKind,
        schedule: LayerSchedule,
        map_id: str,
        /,
        *,
        origin: ArrayLike | None = None,
        axis: ArrayLike | None = None,
        frames: ArrayLike | None = None,
        translations: ArrayLike | None = None,
        closed: bool = False,
        tolerance: float = 1.0e-12,
    ) -> None:
        if not isinstance(kind, SweepMapKind):
            raise TypeError("kind must be a SweepMapKind.")
        if not isinstance(map_id, str) or not map_id.strip():
            raise ValueError("map_id must be nonempty.")
        if not isinstance(closed, bool):
            raise TypeError("closed must be bool.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        center = np.zeros((3,), dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
        direction = np.asarray((0.0, 0.0, 1.0), dtype=np.float64) if axis is None else np.asarray(axis, dtype=np.float64)
        if center.shape != (3,) or direction.shape != (3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(direction)):
            raise ValueError("origin and axis must be finite three-vectors.")
        axis_norm = float(np.linalg.norm(direction))
        if not np.isclose(axis_norm, 1.0, rtol=0.0, atol=8.0 * np.finfo(np.float64).eps):
            raise ValueError(f"axis must be a unit vector; got norm {axis_norm}.")
        station_count = schedule.layer_count + 1
        match kind:
            case SweepMapKind.EXTRUSION | SweepMapKind.REVOLUTION:
                if frames is not None or translations is not None:
                    raise ValueError("Analytic sweep maps cannot carry frame or translation banks.")
                rotations = np.zeros((0, 3, 3), dtype=np.float64)
                offsets = np.zeros((0, 3), dtype=np.float64)
                if kind is SweepMapKind.EXTRUSION and closed:
                    raise ValueError("A straight extrusion cannot declare a closed seam.")
                if kind is SweepMapKind.REVOLUTION and closed:
                    turn = float(np.sum(np.asarray(schedule.thicknesses, dtype=np.float64)))
                    if abs(turn - 2.0 * np.pi) > tolerance:
                        raise ValueError("A closed revolution schedule must span one full turn.")
            case SweepMapKind.FRAMES:
                if frames is None or translations is None:
                    raise ValueError("Frame sweeps require rotations and translations at every station.")
                rotations = np.asarray(frames, dtype=np.float64)
                offsets = np.asarray(translations, dtype=np.float64)
                if rotations.shape != (station_count, 3, 3) or offsets.shape != (station_count, 3):
                    raise ValueError("Frame and translation banks must match the station count.")
                if not np.all(np.isfinite(rotations)) or not np.all(np.isfinite(offsets)):
                    raise ValueError("Frame and translation banks must be finite.")
                gram = np.swapaxes(rotations, -1, -2) @ rotations
                if not np.allclose(gram, np.eye(3, dtype=np.float64), rtol=0.0, atol=tolerance):
                    raise ValueError("Sweep frames must be orthonormal.")
                if np.any(np.linalg.det(rotations) <= 0.0):
                    raise ValueError("Sweep frames must preserve orientation.")
                if closed and (not np.array_equal(rotations[-1], rotations[0]) or not np.array_equal(offsets[-1], offsets[0])):
                    raise ValueError("A closed frame sweep requires identical endpoint transforms.")
            case _:
                raise ValueError(f"Unsupported sweep map kind {kind!r}.")
        self.schedule = schedule
        self.kind = kind
        self.map_id = map_id
        self.closed = closed
        self.tolerance = float(tolerance)
        self.origin, self.axis, self.frames, self.translations = jax.device_put(
            (center, direction, rotations, offsets)
        )
        self.control_id = canonical_fingerprint(
            {
                "kind": "sweep-control",
                "map": map_id,
                "map_kind": kind.value,
                "schedule": schedule.schedule_id,
                "origin": center,
                "axis": direction,
                "frames": array_tree_fingerprint(rotations),
                "translations": offsets,
                "closed": closed,
                "tolerance": float(tolerance),
            }
        )

    @classmethod
    def from_stored_fields(cls: type[SweepControl], value: SweepControl, /) -> SweepControl:
        if type(value) is not cls:
            raise TypeError("Stored sweep control has the wrong registered type.")
        if value.kind is SweepMapKind.FRAMES:
            fresh = cls(
                value.kind,
                value.schedule,
                value.map_id,
                origin=value.origin,
                axis=value.axis,
                frames=value.frames,
                translations=value.translations,
                closed=value.closed,
                tolerance=value.tolerance,
            )
        else:
            fresh = cls(
                value.kind,
                value.schedule,
                value.map_id,
                origin=value.origin,
                axis=value.axis,
                closed=value.closed,
                tolerance=value.tolerance,
            )
        if fresh.control_id != value.control_id or array_tree_fingerprint(fresh) != array_tree_fingerprint(value):
            raise ValueError("Stored sweep control fields do not match constructor-defined state.")
        return fresh


@final
class SweepConstruction(StrictModule):
    """Swept mesh, mapped coordinates, physical stations, and source ancestry."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    source_mesh_id: str = eqx.field(static=True)
    source_vertex_ids: np.ndarray
    layer_vertex_ids: np.ndarray
    source_cell_ids: tuple[tuple[str, np.ndarray], ...]
    cell_parent_ids: tuple[tuple[str, np.ndarray], ...]
    reference_map: tuple[ReferenceCellMapWitnesses, ...]
    stations: np.ndarray
    measured_layer_thicknesses: np.ndarray
    control_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source_mesh_id: str,
        source_vertex_ids: np.ndarray,
        layer_vertex_ids: np.ndarray,
        source_cell_ids: tuple[tuple[str, np.ndarray], ...],
        cell_parent_ids: tuple[tuple[str, np.ndarray], ...],
        reference_map: tuple[ReferenceCellMapWitnesses, ...],
        stations: np.ndarray,
        measured_layer_thicknesses: np.ndarray,
        control_id: str,
    ) -> None:
        parents = dict(cell_parent_ids)
        if set(parents) != {block.name for block in mesh.blocks}:
            raise ValueError("Sweep parent-cell IDs must cover every output block exactly.")
        if len(reference_map) != len(mesh.blocks):
            raise ValueError("Sweep reference maps must cover every output block.")
        for block, witnesses in zip(mesh.blocks, reference_map, strict=True):
            validate_reference_cell_map_witnesses(witnesses)
            if (
                witnesses.target_cell_kind != block.cell_kind
                or not np.array_equal(
                    np.asarray(witnesses.target_cell_ids),
                    np.asarray(block.global_ids),
                )
            ):
                raise ValueError("Sweep reference maps must bind each output cell block.")
            if not np.array_equal(
                np.asarray(witnesses.source_cell_ids),
                np.asarray(parents[block.name]),
            ):
                raise ValueError("Sweep reference maps must preserve actual source cell IDs.")
        self.mesh = mesh
        self.geometry = geometry
        self.source_mesh_id = source_mesh_id
        self.source_vertex_ids = source_vertex_ids
        self.layer_vertex_ids = layer_vertex_ids
        self.source_cell_ids = source_cell_ids
        self.cell_parent_ids = cell_parent_ids
        self.reference_map = reference_map
        self.stations = stations
        self.measured_layer_thicknesses = measured_layer_thicknesses
        self.control_id = control_id


def mapped_stations(points: np.ndarray, control: SweepControl, /) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the declared physical map at every scheduled station."""
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("Sweep points must be finite coordinates with shape (node_count, 2 or 3).")
    if coordinates.shape[1] == 2:
        points3 = np.zeros((coordinates.shape[0], 3), dtype=np.float64)
        points3[:, :2] = coordinates
    else:
        points3 = coordinates
    increments = np.asarray(control.schedule.thicknesses, dtype=np.float64)
    stations = np.empty((increments.size + 1,), dtype=np.float64)
    stations[0] = 0.0
    np.cumsum(increments, out=stations[1:])
    mapped = np.empty((stations.size, points3.shape[0], 3), dtype=np.float64)
    origin = np.asarray(control.origin, dtype=np.float64)
    axis = np.asarray(control.axis, dtype=np.float64)
    match control.kind:
        case SweepMapKind.EXTRUSION:
            for station, distance in enumerate(stations):
                mapped[station] = points3 + distance * axis
        case SweepMapKind.REVOLUTION:
            displacement = points3 - origin
            parallel = (displacement @ axis)[:, None] * axis
            radial = displacement - parallel
            if np.any(np.all(radial == 0.0, axis=-1)):
                raise ValueError("A revolution profile cannot contain a point on the sweep axis.")
            tangent = np.cross(axis, radial)
            for station, angle in enumerate(stations):
                mapped[station] = origin + parallel + np.cos(angle) * radial + np.sin(angle) * tangent
            if control.closed:
                defect = float(np.max(np.linalg.norm(mapped[-1] - mapped[0], axis=-1), initial=0.0))
                scale = max(1.0, float(np.max(np.linalg.norm(points3, axis=-1), initial=0.0)))
                if defect > control.tolerance * scale:
                    raise ValueError(f"The revolution seam does not close; residual {defect}.")
                mapped[-1] = mapped[0]
        case SweepMapKind.FRAMES:
            rotations = np.asarray(control.frames, dtype=np.float64)
            translations = np.asarray(control.translations, dtype=np.float64)
            for station in range(stations.size):
                mapped[station] = points3 @ rotations[station].T + translations[station]
        case _:
            raise ValueError(f"Unsupported sweep map kind {control.kind!r}.")
    if control.closed:
        mapped[-1] = mapped[0]
    return mapped, stations


def column_connectivity(rows: np.ndarray, layers: int, stride: int, closed: bool, /) -> np.ndarray:
    """Extrude oriented triangle/quad rows into prism/hexahedron rows."""
    source_rows = np.asarray(rows)
    if source_rows.ndim != 2 or source_rows.shape[1] < 3 or source_rows.shape[0] == 0:
        raise ValueError("Sweep profile and geometry routes must be nonempty row arrays of width at least three.")
    if source_rows.dtype.kind not in "iu":
        raise TypeError("Sweep connectivity inputs must contain integer node identities.")
    if np.any(source_rows < 0) or np.any(source_rows > np.iinfo(np.int32).max):
        raise OverflowError("Sweep profile node identity is outside int32 connectivity range.")
    cells = source_rows.astype(np.int32, copy=False)
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("layers must be a positive integer.")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if not isinstance(closed, bool):
        raise TypeError("closed must be bool.")
    if np.any(cells >= stride):
        raise ValueError("Sweep connectivity rows must index the declared profile-node stride.")
    node_layers = layers if closed else layers + 1
    if node_layers * stride - 1 > np.iinfo(np.int32).max:
        raise OverflowError("Sweep node identity exceeds int32 connectivity range.")
    result = np.empty((layers * cells.shape[0], 2 * cells.shape[1]), dtype=np.int32)
    for layer in range(layers):
        lower = result[layer * cells.shape[0] : (layer + 1) * cells.shape[0], : cells.shape[1]]
        upper = result[layer * cells.shape[0] : (layer + 1) * cells.shape[0], cells.shape[1] :]
        np.add(cells, layer * stride, out=lower)
        next_layer = 0 if closed and layer == layers - 1 else (layer + 1) * stride
        np.add(cells, next_layer, out=upper)
    return result


def _corner_local_indices(element: FiniteElementSpec, cell_kind: str, /) -> tuple[int, ...]:
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    vertices = reference_cell_topology(cell_kind).vertices
    indices: list[int] = []
    for vertex in vertices:
        matches = np.flatnonzero(np.all(reference_nodes == np.asarray(vertex), axis=1))
        if matches.size != 1:
            raise ValueError("A polynomial sweep source element must include every reference vertex exactly once.")
        indices.append(int(matches[0]))
    return tuple(indices)


def _require_profile_corner_maps(
    profile: CellMesh,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[ArrayLike, ...],
    coordinates: np.ndarray,
    /,
) -> None:
    mesh_points = np.asarray(profile.coordinates, dtype=np.float64)
    for block, element, route in zip(profile.blocks, elements, routes, strict=True):
        if not isinstance(element, FiniteElementSpec):
            raise ValueError("A sweep requires polynomial finite-element profile geometry.")
        corner_indices = _corner_local_indices(element, block.cell_kind)
        corner_routes = np.asarray(route, dtype=np.int64)[:, corner_indices]
        source_corners = coordinates[corner_routes]
        mesh_corners = mesh_points[np.asarray(block.vertices, dtype=np.int64)]
        if not np.array_equal(source_corners, mesh_corners):
            raise ValueError(
                f"Sweep profile coordinate geometry must retain the exact topology corners in block {block.name!r}."
            )


@checked
def construct_sweep(
    profile: CellMesh,
    control: SweepControl,
    /,
    *,
    name: str = "sweep",
    profile_geometry: CellGeometrySpec | None = None,
    root_correspondence: CellGeometryRestrictionSource | None = None,
) -> SweepConstruction:
    """Carry the complete profile coordinate map through swept cells."""
    if profile.topological_dimension != 2 or profile.ambient_dimension not in (2, 3):
        raise ValueError("A sweep profile must be a two-dimensional mesh embedded in 2D or 3D.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be nonempty.")
    if any(block.cell_kind not in ("triangle", "quadrilateral") for block in profile.blocks):
        raise ValueError("A sweep profile may contain only triangle or quadrilateral blocks.")
    source_geometry = CellGeometrySpec.affine(profile) if profile_geometry is None else profile_geometry
    source_elements, source_routes, source_coordinates = source_geometry.resolve(profile)
    source_points = np.asarray(source_coordinates, dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1] != profile.ambient_dimension:
        raise ValueError("Profile geometry coordinates must match the mesh ambient dimension.")
    if any(not isinstance(element, FiniteElementSpec) for element in source_elements):
        raise ValueError("A sweep requires polynomial finite-element profile geometry.")
    _require_profile_corner_maps(profile, source_elements, source_routes, source_points)
    source_coordinates3 = np.zeros((source_points.shape[0], 3), dtype=np.float64)
    source_coordinates3[:, : source_points.shape[1]] = source_points
    mapped_geometry, stations = mapped_stations(source_coordinates3, control)
    mapped_vertices, vertex_stations = mapped_stations(np.asarray(profile.coordinates), control)
    if not np.array_equal(stations, vertex_stations):
        raise RuntimeError("Sweep station construction changed between coordinate banks.")
    layers = stations.size - 1
    if control.closed and layers < 3:
        raise ValueError("A closed sweep requires at least three layers.")
    if control.closed:
        mapped_vertices[-1] = mapped_vertices[0]
        mapped_geometry[-1] = mapped_geometry[0]
    node_layers = layers if control.closed else layers + 1
    coordinates = mapped_vertices[:node_layers].reshape(-1, 3)
    output_blocks: list[CellBlock] = []
    output_elements: dict[str, CellGeometryElement] = {}
    output_routes: dict[str, np.ndarray] = {}
    parent_cells: list[tuple[str, np.ndarray]] = []
    source_cells: list[tuple[str, np.ndarray]] = []
    reference_maps: list[ReferenceCellMapWitnesses] = []
    vertex_stride = profile.coordinates.shape[0]
    offset = 0
    for source_block, element, source_route in zip(profile.blocks, source_elements, source_routes, strict=True):
        source_vertices = np.asarray(source_block.vertices, dtype=np.int32)
        columns = column_connectivity(source_vertices, layers, vertex_stride, control.closed)
        kind = "prism" if source_block.cell_kind == "triangle" else "hexahedron"
        block_name = f"{name}:{source_block.name}"
        identifiers = np.arange(offset, offset + columns.shape[0], dtype=np.int64)
        output_blocks.append(CellBlock(block_name, kind, columns, global_ids=identifiers))
        parent_ids = np.tile(np.asarray(source_block.global_ids, dtype=np.int64), layers)
        source_reference_corners = np.asarray(
            reference_cell_topology(source_block.cell_kind).vertices,
            dtype=np.float64,
        )
        reference_vertices = np.concatenate(
            (source_reference_corners, source_reference_corners), axis=0
        )
        reference_maps.append(
            ReferenceCellMapWitnesses(
                identifiers,
                parent_ids,
                kind,
                source_block.cell_kind,
                np.broadcast_to(
                    reference_vertices,
                    (columns.shape[0], *reference_vertices.shape),
                ).copy(),
            )
        )
        offset += columns.shape[0]
        parent_cells.append((block_name, np.tile(np.asarray(source_block.global_ids, dtype=np.int64), layers)))
        source_cells.append((source_block.name, np.asarray(source_block.global_ids, dtype=np.int64).copy()))
        if not isinstance(element, FiniteElementSpec):
            raise ValueError("A sweep requires polynomial finite-element profile geometry.")
        output_elements[block_name] = swept_coordinate_element(element)
        route_columns = column_connectivity(
            np.asarray(source_route, dtype=np.int32),
            layers,
            source_points.shape[0],
            control.closed,
        )
        output_routes[block_name] = route_columns
    if root_correspondence is not None:
        if name != "sweep":
            raise ValueError("A mapped sweep root correspondence requires canonical sweep block names.")
        if set(root_correspondence.block_names) != set(output_elements):
            raise ValueError("Sweep root correspondence must name every output block exactly.")
        for block in profile.blocks:
            target_name = f"{name}:{block.name}"
            expected = layers * block.cell_count
            if (
                root_correspondence.block_parent_cell_ids[target_name].shape != (expected,)
                or root_correspondence.block_parent_vertex_ids[target_name].shape[0] != expected
            ):
                raise ValueError("Sweep root correspondence must name every layer-major output cell.")
    geometry_points = mapped_geometry[:node_layers].reshape(-1, 3)
    geometry = CellGeometrySpec(
        output_elements,
        output_routes,
        geometry_points,
        restriction_source=root_correspondence,
    )
    layer_ids = np.arange(node_layers * vertex_stride, dtype=np.int64).reshape(node_layers, vertex_stride)
    seam_ids = np.concatenate((layer_ids, layer_ids[:1]), axis=0) if control.closed else layer_ids
    thickness = np.linalg.norm(mapped_vertices[1:] - mapped_vertices[:-1], axis=-1)
    mesh = CellMesh(
        coordinates,
        tuple(output_blocks),
        vertex_global_ids=np.arange(coordinates.shape[0], dtype=np.int64),
    )
    return SweepConstruction(
        mesh,
        geometry,
        profile.mesh_id,
        np.asarray(profile.vertex_global_ids, dtype=np.int64).copy(),
        seam_ids,
        tuple(source_cells),
        tuple(parent_cells),
        tuple(reference_maps),
        stations,
        thickness,
        control.control_id,
    )


@checked
def generate_sweep(
    profile: CellMesh,
    control: SweepControl,
    /,
    *,
    name: str = "sweep",
    profile_geometry: CellGeometrySpec | None = None,
    root_correspondence: CellGeometryRestrictionSource | None = None,
) -> SweepConstruction:
    """Construct swept cells and geometry for canonical native acceptance."""
    return construct_sweep(
        profile,
        control,
        name=name,
        profile_geometry=profile_geometry,
        root_correspondence=root_correspondence,
    )


__all__ = [
    "SweepMapKind",
    "SweepControl",
    "SweepConstruction",
    "mapped_stations",
    "column_connectivity",
    "construct_sweep",
    "generate_sweep",
]
