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
import phydrax.geometry._mapped_reference_coverage as _dependency_phydrax_geometry__mapped_reference_coverage
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
from ..discretization._cell_geometry import (
    affine_geometry_construction_bounds,
    CellGeometryRestrictionSource,
)
from ..discretization._cell_geometry_predecessor import (
    CellGeometryPredecessor,
    prepare_cell_geometry_predecessor,
)
from ..discretization._cell_mesh import (
    cell_block_construction_bounds,
    cell_mesh_construction_bounds,
)
from ..discretization._coordinate_enclosure import (
    coordinate_polynomials,
    expression_equal,
)
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
    layer_count = control.schedule.layer_count
    count = points.shape[0]
    map_count = (layer_count + 1) * count
    # One map evaluation per station node.
    with (
        scope.batch(map_count, queries_upper=map_count) as ticket,
        scope.host_workspace() as workspace,
    ):
        # Broadcast map terms: at most three station-by-node coordinate banks,
        # eight node coordinate banks and two station vectors coexist.
        width = points.shape[1]
        workspace.set_bound(
            8 * (3 * width * map_count + 8 * width * count + 2 * layer_count + 2)
            + 4096
        )
        stations = _dependency_phydrax_meshing__host_banks.family_host_array(
            (layer_count + 1,), np.float64, scope=scope
        )
        stations[0] = 0.0
        np.cumsum(
            np.asarray(control.schedule.thicknesses, dtype=np.float64), out=stations[1:]
        )
        mapped = _dependency_phydrax_meshing__host_banks.family_host_array(
            (stations.size, count, points.shape[1]), np.float64, scope=scope
        )
        origin = np.asarray(control.origin, dtype=np.float64)
        axis = np.asarray(control.axis, dtype=np.float64)
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
    with scope.batch(layers * count * 2 * arity) as ticket:
        columns = _dependency_phydrax_meshing__host_banks.family_host_array(
            (layers * count, 2 * arity), np.int32, scope=scope
        )
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


def _bind_root_geometry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    correspondence: CellGeometryRestrictionSource,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellGeometrySpec:
    "Retain an authored root basis only after source-expression equality."
    root_elements, root_routes, root_coordinates = domain.source_geometry.resolve(
        domain.reference_mesh, budget=budget
    )
    root_values = np.asarray(root_coordinates, dtype=np.float64)
    root_cells = sum(block.cell_count for block in domain.reference_mesh.blocks)
    root_width = max((np.shape(routes)[1] for routes in root_routes), default=0)
    with scope.host_workspace() as workspace:
        # CPython root entries (identifier, record tuple, route row view and
        # dictionary slot) of at most 384 bytes per root cell.
        retained = 384 * root_cells
        workspace.set_bound(retained + 4096)
        roots = {
            int(identifier): (
                element,
                np.asarray(route, dtype=np.int32),
                block.cell_kind,
            )
            for block, element, routes in zip(
                domain.reference_mesh.blocks, root_elements, root_routes, strict=True
            )
            for identifier, route in zip(
                np.asarray(block.global_ids), np.asarray(routes), strict=True
            )
        }
        elements, routes, coordinates = geometry.resolve(mesh, budget=budget)
        values = np.asarray(coordinates, dtype=np.float64)
        cells = sum(block.cell_count for block in mesh.blocks)
        width = max((np.shape(route)[1] for route in routes), default=0)
        gathered = width * values.shape[1] + root_width * root_values.shape[1]
        # Selected row slots and stacked root routes stay live; one cell's two
        # coefficient gathers coexist.
        workspace.set_bound(
            retained + cells * (8 + 4 * root_width) + 8 * gathered + 4096
        )
        bound_elements: dict[str, CellGeometryElement] = {}
        bound_routes: dict[str, np.ndarray] = {}
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            parents = correspondence.block_parent_cell_ids[block.name]
            selected = []
            for cell_id, local_route, parent in zip(
                np.asarray(block.global_ids),
                np.asarray(route),
                np.asarray(parents),
                strict=True,
            ):
                if int(parent) not in roots:
                    raise ValueError(
                        "sweep.root_correspondence must identify declared source roots; got an absent root"
                    )
                root_element, root_route, root_kind = roots[int(parent)]
                if root_kind != block.cell_kind:
                    raise ValueError(
                        "sweep.root_correspondence.reference_family must equal the generated cell family; got mismatched families"
                    )
                generated = coordinate_polynomials(
                    element, values[local_route], budget=budget
                )
                authored = coordinate_polynomials(
                    root_element, root_values[root_route], budget=budget
                )
                if (
                    generated is None
                    or authored is None
                    or len(generated) != len(authored)
                    or not all(
                        expression_equal(first, second, budget=budget)
                        for first, second in zip(generated, authored, strict=True)
                    )
                ):
                    raise _leaf_phydrax_meshing__failures.MeshingFailure(
                        _leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED,
                        "sweep.root_correspondence.coordinate_map must equal the declared source-root map; got a different expression",
                        stage="sweep_source_map",
                        entity_ids=(int(cell_id),),
                    )
                previous = bound_elements.get(block.name)
                if (
                    previous is not None
                    and previous.element_id != root_element.element_id
                ):
                    raise ValueError(
                        "sweep.block.root_basis must be uniform within each block; got differing source-root bases"
                    )
                bound_elements[block.name] = root_element
                selected.append(root_route)
            bound_routes[block.name] = np.stack(selected)
        return CellGeometrySpec(
            bound_elements,
            bound_routes,
            root_coordinates,
            restriction_source=correspondence,
            budget=budget,
        )


def _root_predecessor(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain,
    limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> CellGeometryPredecessor | None:
    """Bind the declared roots' independently certified physical source proof.

    This is the request-retained source evidence domain coverage consumes;
    restricted certification imports its root theorem only after comparing
    every root map coefficientwise. Uncertified sources bind nothing.
    """
    # Coverage retains the earned source certificates, not the polynomial
    # workspace that computed them. Construction and publication consume those
    # actual results through the same request ledger.
    with budget.polynomial_preparation_scope():
        _dependency_phydrax_geometry__mapped_reference_coverage.certify_mapped_reference_source(
            domain, limits, budget=budget
        )
    evidence = _dependency_phydrax_geometry__mapped_reference_coverage.retained_mapped_reference_source_evidence(
        domain, limits, budget=budget
    )
    if evidence is None:
        return None
    return prepare_cell_geometry_predecessor(
        evidence.physical_mesh,
        domain.source_geometry,
        evidence.physical_validity,
        evidence.physical_embedding,
        mesh,
        geometry,
        source_limits=limits,
        budget=budget,
    )


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
    root_domain: _dependency_phydrax_geometry__mapped_reference_domain.MappedReferenceDomain
    | None = None,
    root_correspondence: CellGeometryRestrictionSource | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> SweepConstruction:
    """Carry full polynomial source coordinates into hex/prism columns.

    Revolution evaluates the analytic map at stations; intervals retain the
    explicitly linear station coordinate space. Fidelity certification bounds
    the deviation between each linear interval and the curved revolution.

    ``root_domain`` with ``root_correspondence`` binds every column to its
    declared source root's authored basis after coefficientwise equality of
    the two coordinate maps; the returned geometry and both certificates then
    belong to that published geometry.
    """

    if (root_domain is None) != (root_correspondence is None):
        raise ValueError(
            "sweep.root_correspondence must accompany a mapped root domain; got only one of them"
        )
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
    if profile_geometry is None:
        profile_shapes = tuple(
            (block.cell_kind, block.cell_count, block.arity) for block in profile.blocks
        )
        with scope.host_workspace() as workspace:
            # The profile is not periodic (refused above).
            workspace.set_bound(
                affine_geometry_construction_bounds(
                    profile.coordinates.shape[0],
                    profile.ambient_dimension,
                    profile_shapes,
                )
            )
            source_geometry = CellGeometrySpec.affine(profile)
    else:
        source_geometry = profile_geometry

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
        # Corner equality compares every coordinate through one boolean
        # temporary.
        with scope.host_workspace() as workspace:
            workspace.set_bound(profile.coordinates.size + 4096)
            corners_retained = np.array_equal(
                source_corners, np.asarray(profile.coordinates)
            )
    if not corners_retained:
        raise ValueError(
            "sweep.profile_geometry must retain profile topology corners; got different coordinate-map corner images"
        )
    prepared_source = _require_sweep_elements(source_elements)
    source_routes, source_coordinates = jax.device_get(
        (source_routes, source_coordinates)
    )
    points = np.asarray(profile.coordinates, dtype=np.float64)
    count = points.shape[0]
    if points.shape[1] == 2:
        # Planar profiles embed at zero height in a pooled spatial bank.
        planar = points
        points = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count, 3), np.float64, scope=scope
        )
        points[:, :2] = planar
        points[:, 2] = 0.0
    mapped, stations = mapped_stations(points, control, scope=scope)
    layers = stations.size - 1
    if control.closed:
        if layers < 3:
            raise ValueError(
                f"sweep.layers must have at least three layers when closed; got {layers}"
            )
        seam = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count, 3), np.float64, scope=scope
        )
        seam_distance = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count,), np.float64, scope=scope
        )
        np.subtract(mapped[-1], mapped[0], out=seam)
        np.square(seam, out=seam)
        np.sum(seam, axis=-1, out=seam_distance)
        np.sqrt(seam_distance, out=seam_distance)
        seam_residual = np.max(seam_distance)
        if seam_residual > 1e-12:
            raise ValueError(
                f"sweep.closed_seam must retain source/cap correspondence; got residual {seam_residual}"
            )
        mapped[-1] = mapped[0]
    blocks = []
    cell_offset = 0
    column_cells = layers * sum(block.cell_count for block in profile.blocks)
    column_shapes = tuple(
        (
            "prism" if block.cell_kind == "triangle" else "hexahedron",
            layers * block.cell_count,
            2 * block.arity,
        )
        for block in profile.blocks
    )
    mesh_points = mapped[:-1] if control.closed else mapped
    mesh_vertices = mesh_points.shape[0] * mesh_points.shape[1]
    # One identity range per column cell; device transfer may retain every
    # range. Each column block and then the mesh are admitted before
    # construction.
    with scope.host_workspace() as workspace:
        workspace.set_bound(8 * column_cells + 4096)
        for block, (kind, column_count, arity) in zip(
            profile.blocks, column_shapes, strict=True
        ):
            cells = np.asarray(block.vertices, dtype=np.int32)
            columns = column_connectivity(
                cells, layers, count, control.closed, scope=scope
            )
            workspace.set_bound(
                8 * column_cells
                + 4096
                + cell_block_construction_bounds(column_count, arity)
            )
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
        workspace.set_bound(
            8 * column_cells
            + 4096
            + cell_mesh_construction_bounds(mesh_vertices, 3, column_shapes)
        )
        mesh = CellMesh(mesh_points.reshape(-1, 3), tuple(blocks))
    controls = np.asarray(source_coordinates, dtype=np.float64)
    control_count = controls.shape[0]
    if controls.shape[1] == 2:
        planar = controls
        controls = _dependency_phydrax_meshing__host_banks.family_host_array(
            (control_count, 3), np.float64, scope=scope
        )
        controls[:, :2] = planar
        controls[:, 2] = 0.0
    mapped_controls, _ = mapped_stations(controls, control, scope=scope)
    if control.closed:
        endpoint = mapped_controls[0].size
        # Closeness owns at most five coordinate-sized and five boolean
        # temporaries.
        with scope.host_workspace() as workspace:
            workspace.set_bound(48 * endpoint + 4096)
            controls_closed = np.allclose(
                mapped_controls[-1], mapped_controls[0], atol=1e-12, rtol=0.0
            )
        if not controls_closed:
            raise ValueError(
                "sweep.closed_seam must retain source coordinate controls; got different endpoint coordinate banks"
            )
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
    predecessor: CellGeometryPredecessor | None = None
    with family_execution_owner.coordinate_phase(scope=scope, budget=budget):
        geometry = CellGeometrySpec(
            elements, geometry_routes, geometry_points.reshape(-1, 3), budget=budget
        )
        if root_domain is not None and root_correspondence is not None:
            geometry = _bind_root_geometry(
                mesh,
                geometry,
                root_domain,
                root_correspondence,
                scope=scope,
                budget=budget,
            )
            predecessor = _root_predecessor(
                mesh,
                geometry,
                root_domain,
                _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits()
                if certificate_limits is None
                else certificate_limits,
                budget=budget,
            )
    validity, embedding = (
        _dependency_phydrax_meshing__structured.certify_constructed_mesh(
            mesh,
            geometry=geometry,
            certificate_limits=certificate_limits,
            predecessor=predecessor,
            scope=scope,
            budget=budget,
        )
    )
    node_layers = mapped.shape[0]
    with scope.host_workspace() as workspace:
        # Layer offsets and profile identities are the unpooled temporaries.
        workspace.set_bound(8 * (2 * node_layers + count) + 4096)
        layer_ids = _dependency_phydrax_meshing__host_banks.family_host_array(
            (node_layers, count), np.int64, scope=scope
        )
        np.add(
            np.arange(node_layers, dtype=np.int64)[:, None] * count,
            np.arange(count, dtype=np.int64)[None, :],
            out=layer_ids,
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
