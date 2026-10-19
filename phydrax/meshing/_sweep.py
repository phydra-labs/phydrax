#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native extrusion/revolution and explicitly scheduled rigid-frame sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as _dependency_phydrax_geometry__global_embedding
import phydrax.geometry._mapped_reference_domain as _dependency_phydrax_geometry__mapped_reference_domain
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._family_execution as family_execution_owner
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._structured as _dependency_phydrax_meshing__structured
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.discretization.fem._reference_types import FiniteElementSpec

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeExecutionScope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..typing import checked, parse
from ._contracts import MeshingLimits


class SweepMapKind(StrEnum):
    EXTRUSION = "extrusion"
    REVOLUTION = "revolution"
    FRAMES = "frames"


@final
class SweepControl(StrictModule, NonTrainableState):
    """Store physical layer stations and a declared sweep map.

    Extrusion thicknesses are distances along the unit axis. Revolution
    thicknesses are angles in radians with positive direction around the unit
    axis. Frame schedules are absolute rigid transforms; coordinate controls
    interpolate affinely between stations. ``closed=True`` declares the
    last-to-first source-vertex seam.
    """

    schedule: _dependency_phydrax_meshing__controls.LayerSchedule
    origin: Array
    axis: Array
    frames: Array
    translations: Array
    kind: SweepMapKind = eqx.field(static=True)
    map_id: str = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    control_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        kind: SweepMapKind,
        schedule: _dependency_phydrax_meshing__controls.LayerSchedule,
        map_id: str,
        /,
        *,
        origin: ArrayLike | None = None,
        axis: ArrayLike | None = None,
        frames: ArrayLike | None = None,
        translations: ArrayLike | None = None,
        closed: bool = False,
    ) -> None:
        kind = parse(kind, SweepMapKind, "SweepControl.kind")
        if not isinstance(closed, bool):
            raise TypeError(
                f"SweepControl.closed must be bool; got {type(closed).__name__}"
            )
        if not map_id.strip():
            raise ValueError(f"SweepControl.map_id must be nonempty; got {map_id!r}")
        center = (
            np.zeros((3,), dtype=np.float64)
            if origin is None
            else np.asarray(origin, dtype=np.float64)
        )
        direction = (
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            if axis is None
            else np.asarray(axis, dtype=np.float64)
        )
        if (
            center.shape != (3,)
            or direction.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(direction))
        ):
            raise ValueError(
                f"SweepControl.origin_axis must be finite three-vectors; got origin shape {center.shape}, axis shape {direction.shape}"
            )
        if abs(np.linalg.norm(direction) - 1.0) > 1e-12:
            raise ValueError(
                f"SweepControl.axis must be normalized; got norm {np.linalg.norm(direction)}"
            )
        count = len(schedule.thicknesses) + 1
        match kind:
            case SweepMapKind.EXTRUSION | SweepMapKind.REVOLUTION:
                if frames is not None or translations is not None:
                    raise ValueError(
                        "SweepControl.frames must be absent for analytic maps; got a frame or translation override"
                    )
                rotations = np.zeros((0, 3, 3), dtype=np.float64)
                offsets = np.zeros((0, 3), dtype=np.float64)
                if kind is SweepMapKind.EXTRUSION and closed:
                    raise ValueError(
                        f"SweepControl.closed must be false for a straight extrusion; got {closed}"
                    )
                if kind is SweepMapKind.REVOLUTION:
                    angle = schedule.total_thickness
                    if closed and abs(angle - 2.0 * np.pi) > 1e-12:
                        raise ValueError(
                            f"SweepControl.schedule must span one full turn for a closed revolution; got angle {angle}"
                        )
                    if not closed and angle >= 2.0 * np.pi:
                        raise ValueError(
                            f"SweepControl.closed must be true for a full turn; got false at angle {angle}"
                        )
            case SweepMapKind.FRAMES:
                if frames is None or translations is None:
                    raise ValueError(
                        "SweepControl.frames must provide rotations and translations; got a missing transform bank"
                    )
                rotations = np.asarray(frames, dtype=np.float64)
                offsets = np.asarray(translations, dtype=np.float64)
                if (
                    rotations.shape != (count, 3, 3)
                    or offsets.shape != (count, 3)
                    or not np.all(np.isfinite(rotations))
                    or not np.all(np.isfinite(offsets))
                ):
                    raise ValueError(
                        f"SweepControl.frames must match the physical stations; got rotations {rotations.shape}, translations {offsets.shape}, stations {count}"
                    )
                gram = np.swapaxes(rotations, -1, -2) @ rotations
                if not np.allclose(
                    gram, np.eye(3, dtype=np.float64), atol=1e-12, rtol=0.0
                ) or np.any(np.linalg.det(rotations) <= 0.0):
                    raise ValueError(
                        "SweepControl.frames must be proper orthonormal rotations; got a nonorthonormal or orientation-reversing frame"
                    )
                if closed and (
                    not np.array_equal(rotations[-1], rotations[0])
                    or not np.array_equal(offsets[-1], offsets[0])
                ):
                    raise ValueError(
                        "SweepControl.frames must have equal first and last transforms for a closed sweep; got different endpoint transforms"
                    )
            case _:
                raise ValueError(
                    f"SweepControl.kind must be a supported SweepMapKind; got {kind}"
                )
        self.kind = kind
        self.schedule = schedule
        self.map_id = map_id
        self.closed = closed
        _host_origin = np.asarray(center, dtype=np.float64)
        _host_axis = np.asarray(direction, dtype=np.float64)
        _host_frames = np.asarray(rotations, dtype=np.float64)
        _host_translations = np.asarray(offsets, dtype=np.float64)
        (
            self.origin,
            self.axis,
            self.frames,
            self.translations,
        ) = jax.device_put(
            (
                _host_origin,
                _host_axis,
                _host_frames,
                _host_translations,
            )
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
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SweepControl], value: SweepControl, /
    ) -> SweepControl:
        """Revalidate stored sweep inputs and their control identity."""
        if type(value) is not cls:
            raise TypeError(
                f"SweepControl.stored_fields must have the registered type; got {type(value).__name__}"
            )
        match value.kind:
            case SweepMapKind.FRAMES:
                fresh = cls(
                    value.kind,
                    value.schedule,
                    value.map_id,
                    origin=value.origin,
                    axis=value.axis,
                    frames=value.frames,
                    translations=value.translations,
                    closed=value.closed,
                )
            case SweepMapKind.EXTRUSION | SweepMapKind.REVOLUTION:
                fresh = cls(
                    value.kind,
                    value.schedule,
                    value.map_id,
                    origin=value.origin,
                    axis=value.axis,
                    closed=value.closed,
                )
            case _:
                raise ValueError(
                    f"SweepControl.kind must be a supported SweepMapKind; got {value.kind}"
                )
        if fresh.control_id != value.control_id:
            raise ValueError(
                f"SweepControl.control_id must match its stored inputs; got {value.control_id}"
            )
        if array_tree_fingerprint(fresh) != array_tree_fingerprint(value):
            raise ValueError(
                "SweepControl.stored_fields must match constructor-defined state; got changed numerical or selector fields"
            )
        return fresh


@dataclass(frozen=True, slots=True)
class SweepConstruction:
    mesh: CellMesh
    geometry: CellGeometrySpec
    source_vertex_ids: np.ndarray
    layer_vertex_ids: np.ndarray
    stations: np.ndarray
    measured_layer_thicknesses: np.ndarray
    validity: (
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    )
    embedding: _dependency_phydrax_geometry__global_embedding.GlobalEmbeddingCertificate
    control_id: str


@checked
def mapped_stations(
    points: np.ndarray,
    control: SweepControl,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[np.ndarray, np.ndarray]:
    stations = _dependency_phydrax_meshing__host_banks.family_host_array(
        (control.schedule.layer_count + 1,), np.float64, scope=scope
    )
    stations[0] = 0.0
    np.cumsum(
        np.asarray(control.schedule.thicknesses, dtype=np.float64), out=stations[1:]
    )
    mapped = _dependency_phydrax_meshing__host_banks.family_host_array(
        (stations.size, points.shape[0], points.shape[1]), np.float64, scope=scope
    )
    origin = np.asarray(control.origin, dtype=np.float64)
    axis = np.asarray(control.axis, dtype=np.float64)
    map_count = stations.size * points.shape[0]
    ticket = scope.batch(map_count, queries_upper=map_count)
    with ticket:
        ticket.consume(map_count, geometry_queries=map_count)
        match control.kind:
            case SweepMapKind.EXTRUSION:
                mapped[:] = (
                    points[None, :, :] + stations[:, None, None] * axis[None, None, :]
                )
            case SweepMapKind.REVOLUTION:
                displacement = points - origin
                parallel = (displacement @ axis)[:, None] * axis
                radial = displacement - parallel
                if np.any(np.linalg.norm(radial, axis=-1) <= 1e-12):
                    raise ValueError(
                        "sweep.profile must stay off the revolution axis; got core correspondence collapses"
                    )
                angles = stations[:, None, None]
                mapped[:] = (
                    origin
                    + parallel[None, :, :]
                    + np.cos(angles) * radial[None, :, :]
                    + np.sin(angles) * np.cross(axis, radial)[None, :, :]
                )
                mapped[0] = points
            case SweepMapKind.FRAMES:
                np.matmul(
                    points[None, :, :],
                    np.swapaxes(np.asarray(control.frames), -1, -2),
                    out=mapped,
                )
                mapped += np.asarray(control.translations)[:, None, :]
            case _:
                raise ValueError(
                    f"sweep.kind must be a supported SweepMapKind; got {control.kind}"
                )
    return mapped, stations


@checked
def column_connectivity(
    rows: np.ndarray,
    layers: int,
    stride: int,
    closed: bool,
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    node_layers = layers if closed else layers + 1
    if node_layers * stride - 1 > np.iinfo(np.int32).max:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"sweep.connectivity must fit the int32 node range; got {node_layers * stride} nodes",
            stage="sweep_topology",
        )
    count, arity = rows.shape
    columns = _dependency_phydrax_meshing__host_banks.family_host_array(
        (layers * count, 2 * arity), np.int32, scope=scope
    )
    ticket = scope.batch(columns.size)
    with ticket:
        ticket.consume(columns.size)
        for layer in range(layers):
            lower = columns[layer * count : (layer + 1) * count, :arity]
            upper = columns[layer * count : (layer + 1) * count, arity:]
            np.add(rows, layer * stride, out=lower)
            np.add(
                rows,
                0 if closed and layer == layers - 1 else (layer + 1) * stride,
                out=upper,
            )
    return columns


class _PolynomialSweepSource(StrictModule):
    elements: tuple[FiniteElementSpec, ...]


def _require_sweep_elements(
    elements: tuple[CellGeometryElement, ...],
    /,
) -> _PolynomialSweepSource:
    retained: list[FiniteElementSpec] = []
    for element in elements:
        match element:
            case FiniteElementSpec():
                retained.append(element)
            case _:
                raise ValueError(
                    f"sweep.profile_geometry.elements must be unrestricted polynomial elements; got {type(element).__name__}"
                )
    return _PolynomialSweepSource(tuple(retained))


@checked
def construct_sweep(
    profile: CellMesh,
    control: SweepControl,
    /,
    *,
    name: str = "sweep",
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    profile_geometry: CellGeometrySpec | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> SweepConstruction:
    """Carry full polynomial source coordinates into hex/prism columns.

    Revolution evaluates the analytic map at stations; intervals retain the
    explicitly linear station coordinate space. Fidelity certification bounds
    the deviation between each linear interval and the curved revolution.
    """

    if profile.topological_dimension != 2 or profile.ambient_dimension not in (2, 3):
        raise ValueError(
            f"sweep.profile must have topological dimension two and ambient dimension two or three; got topological={profile.topological_dimension}, ambient={profile.ambient_dimension}"
        )
    if profile.periodic_topology is not None:
        raise ValueError(
            "sweep.profile.periodic_topology must provide lifted correspondence; got a periodic profile without lift correspondence"
        )
    if any(
        block.cell_kind not in ("triangle", "quadrilateral") for block in profile.blocks
    ):
        raise ValueError(
            f"sweep.profile.blocks must contain triangle or quadrilateral cells; got {tuple(block.cell_kind for block in profile.blocks)}"
        )
    source_geometry = (
        CellGeometrySpec.affine(profile) if profile_geometry is None else profile_geometry
    )

    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        source_elements, source_routes, source_coordinates = source_geometry.resolve(
            profile, budget=budget
        )
        corner_count = sum(block.vertices.size for block in profile.blocks)
        family_execution_owner.count_geometry_queries(
            corner_count, work_units=corner_count, budget=budget
        )
        source_corners = _dependency_phydrax_geometry__mapped_reference_domain.mapped_source_corner_coordinates(
            profile, source_geometry, budget=budget
        )
    if not np.array_equal(source_corners, np.asarray(profile.coordinates)):
        raise ValueError(
            "sweep.profile_geometry must retain profile topology corners; got different coordinate-map corner images"
        )
    prepared_source = _require_sweep_elements(source_elements)
    source_routes, source_coordinates = jax.device_get(
        (source_routes, source_coordinates)
    )
    points = np.asarray(profile.coordinates, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    mapped, stations = mapped_stations(points, control, scope=scope)
    count = points.shape[0]
    layers = stations.size - 1
    if control.closed:
        if layers < 3:
            raise ValueError(
                f"sweep.layers must have at least three layers when closed; got {layers}"
            )
        seam_residual = np.max(np.linalg.norm(mapped[-1] - mapped[0], axis=-1))
        if seam_residual > 1e-12:
            raise ValueError(
                f"sweep.closed_seam must retain source/cap correspondence; got residual {seam_residual}"
            )
        mapped[-1] = mapped[0]
    blocks = []
    cell_offset = 0
    for block in profile.blocks:
        cells = np.asarray(block.vertices, dtype=np.int32)
        columns = column_connectivity(cells, layers, count, control.closed, scope=scope)
        kind = "prism" if block.cell_kind == "triangle" else "hexahedron"
        blocks.append(
            CellBlock(
                f"{name}:{block.name}",
                kind,
                columns,
                global_ids=np.arange(
                    cell_offset, cell_offset + columns.shape[0], dtype=np.int64
                ),
            )
        )
        cell_offset += columns.shape[0]
    mesh_points = mapped[:-1] if control.closed else mapped
    mesh = CellMesh(mesh_points.reshape(-1, 3), tuple(blocks))
    controls = np.asarray(source_coordinates, dtype=np.float64)
    if controls.shape[1] == 2:
        controls = np.pad(controls, ((0, 0), (0, 1)))
    mapped_controls, _ = mapped_stations(controls, control, scope=scope)
    if control.closed and not np.allclose(
        mapped_controls[-1], mapped_controls[0], atol=1e-12, rtol=0.0
    ):
        raise ValueError(
            "sweep.closed_seam must retain source coordinate controls; got different endpoint coordinate banks"
        )
    control_count = controls.shape[0]
    elements = {}
    geometry_routes = {}
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        for block, source_element in zip(
            mesh.blocks, prepared_source.elements, strict=True
        ):
            elements[block.name] = (
                _dependency_phydrax_discretization__cell_geometry.swept_coordinate_element(
                    source_element, budget=budget
                )
            )
    for block, source_route in zip(mesh.blocks, source_routes, strict=True):
        geometry_routes[block.name] = column_connectivity(
            np.asarray(source_route, dtype=np.int32),
            layers,
            control_count,
            control.closed,
            scope=scope,
        )
    geometry_points = mapped_controls[:-1] if control.closed else mapped_controls
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        geometry = CellGeometrySpec(
            elements, geometry_routes, geometry_points.reshape(-1, 3), budget=budget
        )
    validity, embedding = (
        _dependency_phydrax_meshing__structured.certify_constructed_mesh(
            mesh,
            geometry=geometry,
            certificate_limits=certificate_limits,
            scope=scope,
            budget=budget,
        )
    )
    layer_ids = np.arange(mapped.shape[0] * count, dtype=np.int64).reshape(
        mapped.shape[:2]
    )
    if control.closed:
        layer_ids[-1] = layer_ids[0]
    differences = _dependency_phydrax_meshing__host_banks.family_host_array(
        (layers, count, 3), np.float64, scope=scope
    )
    measured = _dependency_phydrax_meshing__host_banks.family_host_array(
        (layers, count), np.float64, scope=scope
    )
    np.subtract(mapped[1:], mapped[:-1], out=differences)
    np.square(differences, out=differences)
    np.sum(differences, axis=-1, out=measured)
    np.sqrt(measured, out=measured)
    return SweepConstruction(
        mesh,
        geometry,
        np.asarray(profile.vertex_global_ids, dtype=np.int64),
        layer_ids,
        stations,
        measured,
        validity,
        embedding,
        control.control_id,
    )


@checked
def generate_sweep(
    profile: CellMesh,
    control: SweepControl,
    /,
    *,
    name: str = "sweep",
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    profile_geometry: CellGeometrySpec | None = None,
    limits: MeshingLimits = MeshingLimits(),
) -> SweepConstruction:
    """Construct and certify swept columns within the declared meshing limits."""
    with _dependency_phydrax_meshing__volume_generation.native_volume_execution_budget(
        limits
    ) as scope:
        ticket = scope.batch(0)
        ticket.settle()
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.retained_storage(lambda n: scope.batch(0, bytes_upper=n)):
            return construct_sweep(
                profile,
                control,
                name=name,
                certificate_limits=certificate_limits,
                profile_geometry=profile_geometry,
                scope=scope,
                budget=budget,
            )


__all__ = ["SweepMapKind", "SweepControl", "SweepConstruction", "generate_sweep"]
