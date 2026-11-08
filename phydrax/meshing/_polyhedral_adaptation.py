#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Plane splitting, same-region agglomeration, and restricted-power regeneration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import StrEnum
from fractions import Fraction
from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec, CellVertexGeometryElement
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ..discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from ..geometry._source_boundary_records import SourceBoundaryDistanceWitness
from ..geometry._supermesh import (
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
    PreparedCommonRefinement,
)
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
from ..geometry.implicit._restricted_delaunay_coverage import (
    RestrictedDelaunayBoundaryWitness,
    RestrictedDelaunayCoverageProof,
    RestrictedDelaunayCoverageTransportWitness,
    RestrictedDelaunayCoverageWitness,
    validate_restricted_delaunay_coverage_witness,
)
from ..typing import checked, ConvertibleToArray, parse
from ._contracts import MeshingFailureCategory, VolumeMeshingSpec
from ._lineage import EntityLineageKind, MeshLineage, VertexInterpolationStencil
from ._polyhedral_adaptation_shared import (
    build_polyhedral_topology_edit,
    polyhedral_face_loops,
    polyhedral_signed_volume,
    PolyhedralAdaptationError,
)
from ._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
)
from ._polyhedral_plane_restriction import restrict_polyhedral_plane
from ._topology_edit import assemble_topology_edit, CellTopologyEdit
from ._volume_generation import _shape_targets_gated, NativeVolumeSchedule
from .providers._native_sources import NativePolyhedralSource


class PolyhedralAdaptationOperation(StrEnum):
    """One closed set of concrete edits over an existing polyhedral mesh."""

    PLANE_SPLIT = "plane_split"
    AGGLOMERATE = "agglomerate"
    REGENERATE = "regenerate"


@final
class PolyhedralAdaptationRequest(StrictModule, NonTrainableState):
    """Controls exactly one polyhedral adaptation operation.

    Regeneration receives an authored ``NativePolyhedralSource`` with the
    successor sites and weights; it never interprets site IDs as nodal parents.
    """

    operation: PolyhedralAdaptationOperation = eqx.field(static=True)
    planes: Array
    agglomerations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    regeneration_source: NativePolyhedralSource | None
    volume_specification: VolumeMeshingSpec | None
    schedule: NativePolyhedralSchedule = eqx.field(static=True)
    volume_schedule: NativeVolumeSchedule | None = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        operation: PolyhedralAdaptationOperation,
        /,
        *,
        planes: ConvertibleToArray = (),
        agglomerations: Sequence[Sequence[int]] = (),
        regeneration_source: NativePolyhedralSource | None = None,
        volume_specification: VolumeMeshingSpec | None = None,
        schedule: NativePolyhedralSchedule | None = None,
        volume_schedule: NativeVolumeSchedule | None = None,
        frame_id: str = "polyhedral-adaptation",
    ) -> None:
        operation_ = parse(operation, PolyhedralAdaptationOperation, "operation")
        plane_array = np.asarray(planes, dtype=np.float64)
        if plane_array.size == 0:
            plane_array = np.empty((0, 4), dtype=np.float64)
        elif plane_array.ndim == 1:
            plane_array = plane_array.reshape((1, 4))
        if plane_array.ndim != 2 or plane_array.shape[1] != 4:
            raise ValueError("planes must have shape (count, 4).")
        if not np.all(np.isfinite(plane_array)) or np.any(
            ~np.any(plane_array[:, :3] != 0.0, axis=1)
        ):
            raise ValueError(
                "Every plane requires finite coefficients and a nonzero normal."
            )
        groups: list[tuple[int, ...]] = []
        for group in agglomerations:
            values = tuple(group)
            if len(values) < 2 or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
                for value in values
            ):
                raise ValueError(
                    "Each agglomeration needs at least two nonnegative cell IDs."
                )
            normalized = tuple(sorted(int(value) for value in values))
            if len(set(normalized)) != len(normalized):
                raise ValueError("Agglomeration cell IDs must be distinct.")
            groups.append(normalized)
        normalized_groups = tuple(sorted(groups))
        all_group_ids = tuple(value for group in normalized_groups for value in group)
        if len(set(all_group_ids)) != len(all_group_ids):
            raise ValueError("Agglomeration groups must be disjoint.")
        schedule_ = NativePolyhedralSchedule() if schedule is None else schedule
        if not isinstance(schedule_, NativePolyhedralSchedule):
            raise TypeError("schedule must be NativePolyhedralSchedule.")
        if volume_schedule is not None and not isinstance(
            volume_schedule, NativeVolumeSchedule
        ):
            raise TypeError("volume_schedule must be NativeVolumeSchedule or None.")
        if volume_schedule is not None and not _shape_targets_gated(volume_schedule):
            raise ValueError(
                "Polyhedral regeneration carriers require a hard-strength volume_schedule."
            )
        if volume_specification is not None and not isinstance(
            volume_specification, VolumeMeshingSpec
        ):
            raise TypeError("volume_specification must be VolumeMeshingSpec or None.")
        if regeneration_source is not None and not isinstance(
            regeneration_source, NativePolyhedralSource
        ):
            raise TypeError("regeneration_source must be NativePolyhedralSource or None.")
        frame = str(frame_id).strip()
        if not frame:
            raise ValueError("frame_id must be non-empty.")
        match operation_:
            case PolyhedralAdaptationOperation.PLANE_SPLIT:
                valid = (
                    bool(plane_array.shape[0])
                    and not normalized_groups
                    and (regeneration_source is None and volume_specification is None)
                )
            case PolyhedralAdaptationOperation.AGGLOMERATE:
                valid = (
                    bool(normalized_groups)
                    and not plane_array.shape[0]
                    and (regeneration_source is None and volume_specification is None)
                )
            case PolyhedralAdaptationOperation.REGENERATE:
                valid = (
                    regeneration_source is not None
                    and volume_specification is not None
                    and not normalized_groups
                    and not plane_array.shape[0]
                )
            case _:
                raise ValueError("operation is not a supported polyhedral operation.")
        if not valid:
            raise ValueError(
                "Request controls must match exactly one polyhedral operation."
            )
        self.operation = operation_
        self.planes = jnp.asarray(plane_array)
        self.agglomerations = normalized_groups
        self.regeneration_source = regeneration_source
        self.volume_specification = volume_specification
        self.schedule = schedule_
        self.volume_schedule = volume_schedule
        self.frame_id = frame
        self.request_id = canonical_fingerprint(
            {
                "kind": "polyhedral-adaptation-request",
                "operation": operation_.value,
                "planes": array_tree_fingerprint(plane_array),
                "agglomerations": normalized_groups,
                "regeneration_source": (
                    None
                    if regeneration_source is None
                    else regeneration_source.binding_id
                ),
                "volume_specification": (
                    None
                    if volume_specification is None
                    else volume_specification.specification_id
                ),
                "schedule": schedule_.schedule_id,
                "volume_schedule": (
                    None if volume_schedule is None else volume_schedule.schedule_id
                ),
                "frame": frame,
            }
        )


class PolyhedralAdaptationHierarchy(NamedTuple):
    source_mesh_id: str
    target_mesh_id: str
    source_cell_ids: np.ndarray
    target_cell_ids: np.ndarray
    parent_offsets: np.ndarray
    parent_cell_ids: np.ndarray
    region_classes: np.ndarray
    hierarchy_id: str


class PolyhedralAdaptationEvidence(NamedTuple):
    operation: PolyhedralAdaptationOperation
    source_mesh_id: str
    target_mesh_id: str
    changed_cell_ids: np.ndarray
    target_region_classes: np.ndarray
    common_refinement_id: str
    conservative_remap_id: str
    maximum_measure_residual: float
    maximum_coordinate_rounding_error: float
    construction_id: str | None
    site_ids: tuple[str, ...]
    site_points: np.ndarray | None
    site_weights: np.ndarray | None
    coverage_renewal_id: str | None
    evidence_id: str


class PolyhedralCoverageRenewal(NamedTuple):
    source_witness_id: str
    source_distance_witness_id: str
    target_witness: RestrictedDelaunayBoundaryWitness
    target_distance_witness: SourceBoundaryDistanceWitness
    common_refinement_id: str
    proof: RestrictedDelaunayCoverageProof
    renewal_id: str


class PolyhedralAdaptationOutcome(NamedTuple):
    edit: CellTopologyEdit
    target_mesh: CellMesh
    target_geometry: CellGeometrySpec
    lineage: MeshLineage
    vertex_stencil: VertexInterpolationStencil
    hierarchy: PolyhedralAdaptationHierarchy
    evidence: PolyhedralAdaptationEvidence
    common_refinement: PreparedCommonRefinement
    conservative_remap: UnstructuredConservativeRemapPlan
    coverage_renewal: PolyhedralCoverageRenewal | None
    construction: PolyhedralConstruction | None


@final
class _PolyhedralExactCoordinateSource(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)
    rounded_vertices: Array
    exact_vertices: tuple[tuple[Fraction, Fraction, Fraction], ...] = eqx.field(
        static=True
    )
    parent_map_id: str = eqx.field(static=True)
    vertex_ids: Array

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        exact_vertices: tuple[tuple[Fraction, Fraction, Fraction], ...],
        /,
        *,
        parent_map_id: str,
        vertex_ids: np.ndarray,
    ) -> None:
        rounded = np.asarray(
            [[float(value) for value in point] for point in exact_vertices],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(rounded)):
            raise ValueError("Exact polyhedral coordinates need finite binary64 shadows.")
        self.source_id = source_id
        self.source_revision = source_revision
        self.exact_vertices = exact_vertices
        self.rounded_vertices = jnp.asarray(rounded)
        self.parent_map_id = parent_map_id
        self.vertex_ids = jnp.asarray(vertex_ids, dtype=jnp.int64)
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "polyhedral-exact-coordinate-restriction",
                "source": source_id,
                "revision": source_revision,
                "parent_map": parent_map_id,
                "vertex_ids": array_tree_fingerprint(vertex_ids),
                "exact_coordinates": tuple(
                    tuple((value.numerator, value.denominator) for value in point)
                    for point in exact_vertices
                ),
            }
        )

    @classmethod
    def from_stored_fields(
        cls, value: _PolyhedralExactCoordinateSource, /, *, mesh: CellMesh
    ) -> _PolyhedralExactCoordinateSource:
        """Recompute the restriction identity from its stored exact inputs."""
        if type(value) is not cls or any(
            len(point) != 3 or any(type(item) is not Fraction for item in point)
            for point in value.exact_vertices
        ):
            raise TypeError("Stored polyhedral coordinates must be exact rationals.")
        vertex_ids = np.asarray(value.vertex_ids, dtype=np.int64)
        rebuilt = cls(
            value.source_id,
            value.source_revision,
            tuple(tuple(point) for point in value.exact_vertices),
            parent_map_id=value.parent_map_id,
            vertex_ids=vertex_ids,
        )
        if (
            rebuilt.source_map_id != value.source_map_id
            or not np.array_equal(
                np.asarray(rebuilt.rounded_vertices), np.asarray(value.rounded_vertices)
            )
            or not np.array_equal(vertex_ids, np.asarray(mesh.vertex_global_ids))
            or not np.array_equal(
                np.asarray(rebuilt.rounded_vertices), np.asarray(mesh.coordinates)
            )
        ):
            raise ValueError("Stored polyhedral exact coordinate source changed.")
        return rebuilt

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
        if indices is None:
            rows = np.arange(len(self.exact_vertices), dtype=np.int64)
        else:
            values = np.asarray(indices)
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise TypeError("coordinate indices must be an integer vector.")
            rows = values.astype(np.int64, copy=False)
            if np.any(rows < 0) or np.any(rows >= len(self.exact_vertices)):
                raise IndexError("coordinate index lies outside the exact vertex bank.")
        return tuple(self.exact_vertices[int(row)] for row in rows)


def _exact_coordinate_bank(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    if geometry.coordinates.shape != mesh.coordinates.shape or not np.array_equal(
        np.asarray(geometry.coordinates), np.asarray(mesh.coordinates)
    ):
        raise ValueError(
            "Polyhedral adaptation requires one P1 coordinate per mesh vertex."
        )
    bank = geometry.source_coordinates()
    message = (
        "Exact polyhedral plane edits require rational represented source coordinates."
    )
    if len(bank) != mesh.coordinates.shape[0]:
        raise ValueError(message)
    coordinates: list[tuple[Fraction, Fraction, Fraction]] = []
    for point in bank:
        if len(point) != 3:
            raise ValueError(message)
        x, y, z = point
        if (
            not isinstance(x, Fraction)
            or not isinstance(y, Fraction)
            or not isinstance(z, Fraction)
        ):
            raise ValueError(message)
        coordinates.append((x, y, z))
    return tuple(coordinates)


def _require_piecewise_linear_geometry(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> None:
    elements, _, _ = geometry.resolve(mesh)
    if len(elements) != len(mesh.blocks) or any(
        type(element) is not CellVertexGeometryElement for element in elements
    ):
        raise ValueError(
            "Polyhedral adaptation requires straight-sided P1 cell geometry."
        )


def _bind_exact_polyhedral_geometry(
    mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    exact_coordinates: tuple[tuple[Fraction, Fraction, Fraction], ...],
    /,
) -> CellGeometrySpec:
    base = CellGeometrySpec.affine(mesh)
    source_map = source_geometry.source_map
    source_id = (
        f"polyhedral-mesh-{source_geometry.source_mesh_id or mesh.topology_id}"
        if source_map is None
        else source_map.source_id
    )
    source_revision = (
        mesh.numeric_version if source_map is None else source_map.source_revision
    )
    parent_map_id = mesh.geometry_id if source_map is None else source_map.source_map_id
    exact_source = _PolyhedralExactCoordinateSource(
        source_id,
        source_revision,
        exact_coordinates,
        parent_map_id=parent_map_id,
        vertex_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64),
    )
    return CellGeometrySpec(
        dict(zip(base.block_names, base.elements, strict=True)),
        {
            name: np.asarray(route, dtype=np.int32)
            for name, route in zip(base.block_names, base.geometry_dofs, strict=True)
        },
        mesh.coordinates,
        source_map=exact_source,
        source_mesh=mesh,
    )


def _classes_by_cell(
    mesh: CellMesh,
    cell_classes: ArrayLike | None,
    /,
) -> dict[int, int]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("mesh must have packed polyhedral connectivity.")
    identifiers = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    classes = (
        np.zeros((identifiers.size,), dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes)
    )
    if classes.ndim != 1 or classes.shape != identifiers.shape:
        raise ValueError("cell_classes must contain one value per polyhedral cell.")
    if classes.dtype.kind not in "iu" or np.any(classes < 0):
        raise TypeError("cell_classes must be nonnegative integer labels.")
    return {
        int(identifier): int(value)
        for identifier, value in zip(identifiers, classes, strict=True)
    }


def _protected_faces(mesh: CellMesh, values: ConvertibleToArray, /) -> set[int]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("mesh must have packed polyhedral connectivity.")
    supplied = np.asarray(values)
    if supplied.size == 0:
        return set()
    if supplied.ndim != 1 or supplied.dtype.kind not in "iu":
        raise TypeError("protected_face_ids must be an integer vector.")
    identifiers = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    requested = set(int(value) for value in supplied)
    if any(value < 0 for value in requested) or not requested.issubset(
        set(int(value) for value in identifiers)
    ):
        raise ValueError("protected_face_ids must name current polyhedral faces.")
    return requested


def _split_candidate(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    request: PolyhedralAdaptationRequest,
    protected: set[int],
    classes: Mapping[int, int],
    /,
) -> tuple[
    CellMesh,
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    dict[int, dict[int, Fraction]],
    dict[int, tuple[int, ...]],
    dict[int, int],
    tuple[int, ...],
    float,
]:
    current = mesh
    exact_coordinates = _exact_coordinate_bank(mesh, geometry)
    vertex_support: dict[int, dict[int, Fraction]] = {
        int(identifier): {int(identifier): Fraction(1)}
        for identifier in np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    }
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("mesh must have packed polyhedral connectivity.")
    source_ids = tuple(int(value) for value in np.asarray(connectivity.cell_global_ids))
    ancestors: dict[int, tuple[int, ...]] = {
        identifier: (identifier,) for identifier in source_ids
    }
    maximum_error = 0.0
    for plane in np.asarray(request.planes, dtype=np.float64):
        restricted = restrict_polyhedral_plane(
            current,
            plane,
            protected,
            vertex_support,
            exact_coordinates=exact_coordinates,
        )
        if restricted.mesh is current:
            continue
        parent_rows = dict(restricted.child_parent_ids)
        next_ancestors: dict[int, tuple[int, ...]] = {}
        next_connectivity = restricted.mesh.connectivity
        if not isinstance(next_connectivity, PolyhedralConnectivity):
            raise TypeError("A split target lost polyhedral connectivity.")
        for identifier_raw in np.asarray(next_connectivity.cell_global_ids):
            identifier = int(identifier_raw)
            parent = parent_rows.get(identifier)
            next_ancestors[identifier] = (
                ancestors[identifier] if parent is None else ancestors[parent]
            )
        ancestors = next_ancestors
        current = restricted.mesh
        connectivity = next_connectivity
        exact_coordinates = restricted.exact_coordinates
        maximum_error = max(maximum_error, restricted.maximum_coordinate_rounding_error)
    source_ids_set = set(source_ids)
    changed = tuple(
        sorted(
            int(identifier)
            for identifier in np.asarray(connectivity.cell_global_ids)
            if int(identifier) not in source_ids_set
        )
    )
    target_classes = {
        identifier: classes[parents[0]] for identifier, parents in ancestors.items()
    }
    return (
        current,
        exact_coordinates,
        vertex_support,
        ancestors,
        target_classes,
        changed,
        maximum_error,
    )


def _agglomerate_candidate(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    request: PolyhedralAdaptationRequest,
    protected: set[int],
    classes: Mapping[int, int],
    /,
) -> tuple[
    CellMesh,
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    dict[int, dict[int, Fraction]],
    dict[int, tuple[int, ...]],
    dict[int, int],
    tuple[int, ...],
]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("mesh must have packed polyhedral connectivity.")
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    row_by_id = {int(identifier): row for row, identifier in enumerate(cell_ids)}
    loops = polyhedral_face_loops(mesh)
    offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    groups = request.agglomerations
    removed = {identifier for group in groups for identifier in group}
    if not removed.issubset(row_by_id):
        raise ValueError("Agglomeration groups must name current polyhedral cells.")
    for group in groups:
        rows = {row_by_id[identifier] for identifier in group}
        if len({classes[identifier] for identifier in group}) != 1:
            raise PolyhedralAdaptationError(
                "Agglomeration cannot cross a material or region class.",
                category=MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                entity_ids=tuple(group),
            )
        visited: set[int] = set()
        frontier = [min(rows)]
        while frontier:
            row = frontier.pop()
            if row in visited:
                continue
            visited.add(row)
            for face in faces[offsets[row] : offsets[row + 1]]:
                left, right = int(owner[face]), int(neighbor[face])
                adjacent = right if left == row else left
                if adjacent in rows and adjacent not in visited:
                    frontier.append(adjacent)
        if visited != rows:
            raise PolyhedralAdaptationError(
                "Agglomeration groups must be face-connected.", entity_ids=tuple(group)
            )
    output_cells: list[tuple[np.ndarray, ...]] = []
    output_ids: list[int] = []
    parent_ids: dict[int, tuple[int, ...]] = {}
    target_classes: dict[int, int] = {}
    for row, identifier_raw in enumerate(cell_ids):
        identifier = int(identifier_raw)
        if identifier not in removed:
            output_cells.append(loops[row])
            output_ids.append(identifier)
            parent_ids[identifier] = (identifier,)
            target_classes[identifier] = classes[identifier]
    next_cell = int(np.max(cell_ids)) + 1
    for group in groups:
        rows = {row_by_id[identifier] for identifier in group}
        boundary: list[np.ndarray] = []
        for row in sorted(rows):
            for local, face in enumerate(loops[row]):
                face_row = int(faces[offsets[row] + local])
                left, right = int(owner[face_row]), int(neighbor[face_row])
                internal = right >= 0 and left in rows and right in rows
                if internal:
                    if int(face_ids[face_row]) in protected:
                        raise PolyhedralAdaptationError(
                            "A protected face cannot be removed by agglomeration.",
                            entity_ids=(int(face_ids[face_row]),),
                        )
                    continue
                boundary.append(face)
        if not boundary:
            raise PolyhedralAdaptationError(
                "Agglomeration produced an empty shell.", entity_ids=tuple(group)
            )
        if next_cell > np.iinfo(np.int64).max:
            raise OverflowError("Polyhedral cell global IDs exceed int64.")
        output_cells.append(tuple(boundary))
        output_ids.append(next_cell)
        parent_ids[next_cell] = tuple(group)
        target_classes[next_cell] = classes[group[0]]
        next_cell += 1
    used = np.unique(np.concatenate([face for cell in output_cells for face in cell]))
    remap = np.full(mesh.coordinates.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    compact_cells = tuple(
        tuple(np.asarray(remap[face], dtype=np.int32) for face in cell)
        for cell in output_cells
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)[used]
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[used]
    target = CellMesh.from_polyhedra(
        coordinates,
        compact_cells,
        block_name=mesh.blocks[0].name,
        vertex_global_ids=vertex_ids,
        cell_global_ids=np.asarray(output_ids, dtype=np.int64),
        numeric_version=mesh.numeric_version,
    )
    exact_source = _exact_coordinate_bank(mesh, geometry)
    exact_coordinates = tuple(exact_source[int(row)] for row in used)
    supports = {
        int(identifier): {int(identifier): Fraction(1)} for identifier in vertex_ids
    }
    for row, identifier in enumerate(output_ids):
        if len(parent_ids[identifier]) > 1:
            volume = polyhedral_signed_volume(coordinates, compact_cells[row])
            if volume <= 0:
                raise PolyhedralAdaptationError(
                    "Agglomeration produced a nonpositive whole-cell volume.",
                    entity_ids=parent_ids[identifier],
                )
    changed = tuple(
        sorted(identifier for identifier in output_ids if identifier not in row_by_id)
    )
    return target, exact_coordinates, supports, parent_ids, target_classes, changed


def _regenerate_candidate(
    mesh: CellMesh,
    request: PolyhedralAdaptationRequest,
    /,
) -> tuple[CellMesh, PolyhedralConstruction]:
    source = request.regeneration_source
    specification = request.volume_specification
    if source is None or specification is None:
        raise ValueError(
            "Regeneration requires an authored source and volume specification."
        )
    construction = generate_polyhedral_volume(
        source,
        specification,
        request.schedule,
        frame_id=request.frame_id,
        volume_schedule=request.volume_schedule,
    )
    generated = construction.mesh
    generated_connectivity = generated.connectivity
    source_connectivity = mesh.connectivity
    if not isinstance(generated_connectivity, PolyhedralConnectivity) or not isinstance(
        source_connectivity, PolyhedralConnectivity
    ):
        raise TypeError("Polyhedral regeneration requires polyhedral endpoint meshes.")
    vertex_start = int(np.max(np.asarray(mesh.vertex_global_ids, dtype=np.int64))) + 1
    cell_start = (
        int(np.max(np.asarray(source_connectivity.cell_global_ids, dtype=np.int64))) + 1
    )
    vertex_count = generated.coordinates.shape[0]
    cell_count = generated_connectivity.cell_count
    if (
        vertex_start + vertex_count - 1 > np.iinfo(np.int64).max
        or cell_start + cell_count - 1 > np.iinfo(np.int64).max
    ):
        raise OverflowError("Regenerated polyhedral global IDs exceed int64.")
    vertex_ids = np.arange(vertex_start, vertex_start + vertex_count, dtype=np.int64)
    cell_ids = np.arange(cell_start, cell_start + cell_count, dtype=np.int64)
    target = CellMesh.from_polyhedra(
        np.asarray(generated.coordinates, dtype=np.float64),
        polyhedral_face_loops(generated),
        block_name=generated.blocks[0].name,
        vertex_global_ids=vertex_ids,
        cell_global_ids=cell_ids,
        numeric_version=mesh.numeric_version,
    )
    return target, construction


def _region_classes_from_overlap(
    source: CellMesh,
    target: CellMesh,
    refinement: PreparedCommonRefinement,
    source_classes: Mapping[int, int],
    /,
) -> tuple[
    dict[int, int],
    dict[int, tuple[int, ...]],
    tuple[tuple[int, int, EntityLineageKind], ...],
]:
    source_connectivity = source.connectivity
    target_connectivity = target.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity) or not isinstance(
        target_connectivity, PolyhedralConnectivity
    ):
        raise TypeError("Overlap region transfer requires polyhedral endpoint meshes.")
    source_ids = np.asarray(source_connectivity.cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(target_connectivity.cell_global_ids, dtype=np.int64)
    source_rows = np.asarray(refinement.source_cells, dtype=np.int64)
    target_offsets = np.asarray(refinement.target_offsets, dtype=np.int64)
    target_classes: dict[int, int] = {}
    parents: dict[int, tuple[int, ...]] = {}
    relations: list[tuple[int, int, EntityLineageKind]] = []
    for target_row, target_id_raw in enumerate(target_ids):
        target_id = int(target_id_raw)
        rows = source_rows[target_offsets[target_row] : target_offsets[target_row + 1]]
        parent_ids = tuple(sorted(set(int(source_ids[row]) for row in rows)))
        if not parent_ids:
            raise PolyhedralAdaptationError(
                "A regenerated cell has no positive physical source overlap.",
                category=MeshingFailureCategory.TRANSFER_FAILED,
                entity_ids=(target_id,),
            )
        region_values = {source_classes[parent] for parent in parent_ids}
        if len(region_values) != 1:
            raise PolyhedralAdaptationError(
                "A regenerated cell crosses source material or region classes.",
                category=MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                entity_ids=(target_id, *parent_ids),
            )
        target_classes[target_id] = next(iter(region_values))
        parents[target_id] = parent_ids
        for parent in parent_ids:
            relations.append((parent, target_id, EntityLineageKind.SWAPPED_FROM))
    return target_classes, parents, tuple(relations)


def _hierarchy(
    source: CellMesh,
    target: CellMesh,
    parents: Mapping[int, tuple[int, ...]],
    region_classes: Mapping[int, int],
    /,
) -> PolyhedralAdaptationHierarchy:
    source_connectivity = source.connectivity
    target_connectivity = target.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity) or not isinstance(
        target_connectivity, PolyhedralConnectivity
    ):
        raise TypeError("Hierarchy endpoints must have polyhedral connectivity.")
    source_ids = np.asarray(source_connectivity.cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(target_connectivity.cell_global_ids, dtype=np.int64)
    offsets = np.zeros((target_ids.size + 1,), dtype=np.int64)
    rows: list[int] = []
    classes = np.empty((target_ids.size,), dtype=np.int64)
    for row, identifier_raw in enumerate(target_ids):
        identifier = int(identifier_raw)
        source_parents = parents.get(identifier)
        if source_parents is None or not source_parents:
            raise ValueError("Every target cell requires stable source ancestry.")
        rows.extend(source_parents)
        offsets[row + 1] = len(rows)
        classes[row] = region_classes[identifier]
    parent_ids = np.asarray(rows, dtype=np.int64)
    identity = canonical_fingerprint(
        {
            "kind": "polyhedral-adaptation-hierarchy",
            "source_mesh": source.mesh_id,
            "target_mesh": target.mesh_id,
            "source_cell_ids": array_tree_fingerprint(source_ids),
            "target_cell_ids": array_tree_fingerprint(target_ids),
            "parent_offsets": array_tree_fingerprint(offsets),
            "parent_cell_ids": array_tree_fingerprint(parent_ids),
            "region_classes": array_tree_fingerprint(classes),
        }
    )
    return PolyhedralAdaptationHierarchy(
        source.mesh_id,
        target.mesh_id,
        source_ids,
        target_ids,
        offsets,
        parent_ids,
        classes,
        identity,
    )


def _physical_transfer(
    source: CellMesh,
    target: CellMesh,
    policy: CommonRefinementPolicy,
    /,
) -> tuple[
    PreparedCommonRefinement,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredConservativeRemapPlan,
]:
    if policy.coverage_tolerance <= 0.0:
        raise ValueError(
            "Physical conservative remapping requires positive coverage tolerance."
        )
    refinement = prepare_common_refinement(source, target, policy=policy)
    if refinement.status is not CommonRefinementStatus.SUCCESS:
        raise PolyhedralAdaptationError(
            f"Physical common refinement failed: {refinement.evidence.reason}",
            category=MeshingFailureCategory.TRANSFER_FAILED,
        )
    source_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(source).prepare(
        numeric_version=source.numeric_version
    )
    target_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(target).prepare(
        numeric_version=target.numeric_version
    )
    remap = UnstructuredConservativeRemapPlan(
        source_fv,
        target_fv,
        refinement.target_offsets,
        refinement.source_cells,
        refinement.volumes,
        method="physical-overlap",
        provenance=refinement.refinement_id,
        tolerance=policy.coverage_tolerance,
        require_complete=True,
    )
    return refinement, source_fv, target_fv, remap


def _coverage_region_rows(
    mesh: CellMesh,
    classes: Mapping[int, int],
    /,
) -> np.ndarray:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Restricted coverage requires polyhedral connectivity.")
    return np.asarray(
        [
            classes[int(identifier)]
            for identifier in np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        ],
        dtype=np.int64,
    )


def _renew_restricted_coverage(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    source_classes: Mapping[int, int],
    source_witness: RestrictedDelaunayCoverageWitness,
    source_distance: SourceBoundaryDistanceWitness,
    profile: AnalyticImplicitProfile,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    target_classes: Mapping[int, int],
    common_refinement: PreparedCommonRefinement,
    /,
) -> PolyhedralCoverageRenewal:
    if common_refinement.status is not CommonRefinementStatus.SUCCESS:
        raise PolyhedralAdaptationError(
            "Restricted coverage renewal requires complete physical overlap.",
            category=MeshingFailureCategory.TRANSFER_FAILED,
        )
    source_proof = validate_restricted_delaunay_coverage_witness(
        source,
        source_geometry,
        profile,
        _coverage_region_rows(source, source_classes),
        source_distance,
        source_witness,
    )
    target_connectivity = target.connectivity
    if not isinstance(target_connectivity, PolyhedralConnectivity):
        raise TypeError(
            "Restricted coverage renewal requires polyhedral target incidence."
        )
    boundary = np.asarray(target_connectivity.face_neighbor, dtype=np.int64) < 0
    target_face_ids = np.sort(
        np.asarray(target_connectivity.face_global_ids, dtype=np.int64)[boundary]
    )
    target_geometry_id = cell_geometry_id(target_geometry)
    target_distance = SourceBoundaryDistanceWitness(
        mesh_id=target.mesh_id,
        geometry_id=target_geometry_id,
        topology_id=target.topology_id,
        source_id=profile.source_id,
        source_revision=profile.source_revision,
        source_query_id=profile.profile_id,
        sample_count=source_distance.sample_count,
        cover_radius=source_distance.cover_radius,
        target_facet_global_ids=target_face_ids,
    )
    target_witness = RestrictedDelaunayBoundaryWitness(
        mesh_id=target.mesh_id,
        geometry_id=target_geometry_id,
        topology_id=target.topology_id,
        profile_id=profile.profile_id,
        source_id=profile.source_id,
        source_revision=profile.source_revision,
        domain_id=profile.domain_id,
        distance_witness_id=target_distance.witness_id,
    )
    proof = validate_restricted_delaunay_coverage_witness(
        target,
        target_geometry,
        profile,
        _coverage_region_rows(target, target_classes),
        target_distance,
        target_witness,
    )
    renewal_id = canonical_fingerprint(
        {
            "kind": "polyhedral-coverage-overlap-renewal",
            "source_witness": source_witness.witness_id,
            "source_distance": source_distance.witness_id,
            "source_degree": source_proof.degree_id,
            "target_witness": target_witness.witness_id,
            "target_distance": target_distance.witness_id,
            "target_degree": proof.degree_id,
            "common_refinement": common_refinement.refinement_id,
        }
    )
    return PolyhedralCoverageRenewal(
        source_witness.witness_id,
        source_distance.witness_id,
        target_witness,
        target_distance,
        common_refinement.refinement_id,
        proof,
        renewal_id,
    )


def _measure_residual(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    tolerance: float,
    /,
) -> float:
    source_measure = float(np.sum(np.asarray(source.cell_volumes), dtype=np.float64))
    target_measure = float(np.sum(np.asarray(target.cell_volumes), dtype=np.float64))
    residual = abs(source_measure - target_measure)
    allowed = tolerance * max(abs(source_measure), abs(target_measure))
    if residual > allowed:
        raise PolyhedralAdaptationError(
            "The topology edit changed total physical polyhedral measure.",
            category=MeshingFailureCategory.TRANSFER_FAILED,
        )
    return residual


def _cell_relations_from_parents(
    source: CellMesh,
    target: CellMesh,
    parents: Mapping[int, tuple[int, ...]],
    operation: PolyhedralAdaptationOperation,
    /,
) -> tuple[tuple[int, int, EntityLineageKind], ...]:
    source_connectivity = source.connectivity
    target_connectivity = target.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity) or not isinstance(
        target_connectivity, PolyhedralConnectivity
    ):
        raise TypeError("Cell relations require polyhedral endpoint meshes.")
    source_ids = set(
        int(value) for value in np.asarray(source_connectivity.cell_global_ids)
    )
    relations: list[tuple[int, int, EntityLineageKind]] = []
    match operation:
        case PolyhedralAdaptationOperation.PLANE_SPLIT:
            kind = EntityLineageKind.REFINED_FROM
        case PolyhedralAdaptationOperation.AGGLOMERATE:
            kind = EntityLineageKind.MERGED_INTO
        case PolyhedralAdaptationOperation.REGENERATE:
            return ()
        case _:
            raise ValueError("Unsupported polyhedral operation.")
    for target_id, source_parents in parents.items():
        if (
            len(source_parents) == 1
            and target_id == source_parents[0]
            and target_id in source_ids
        ):
            continue
        relations.extend((source_id, target_id, kind) for source_id in source_parents)
    return tuple(relations)


@checked
def adapt_polyhedral_mesh(
    mesh: CellMesh,
    request: PolyhedralAdaptationRequest,
    /,
    *,
    source_geometry: CellGeometrySpec | None = None,
    cell_classes: ArrayLike | None = None,
    protected_face_ids: ConvertibleToArray = (),
    common_refinement_policy: CommonRefinementPolicy | None = None,
    source_coverage_witness: RestrictedDelaunayCoverageWitness | None = None,
    source_distance_witness: SourceBoundaryDistanceWitness | None = None,
    source_coverage_profile: AnalyticImplicitProfile | None = None,
) -> PolyhedralAdaptationOutcome:
    """Build a certified polyhedral edit and its physical-overlap transfer.

    Plane cuts are exact for rational P1 vertex coordinates and edited cells
    must be convex with planar faces. ``cell_classes=None`` denotes one region;
    callers carrying material labels must provide the source cell classes.
    Optional restricted-coverage inputs are fully revalidated at both endpoints:
    volume overlap alone is never treated as a boundary-degree proof.
    """
    if not isinstance(mesh, CellMesh) or not isinstance(
        request, PolyhedralAdaptationRequest
    ):
        raise TypeError(
            "mesh and request must be CellMesh and PolyhedralAdaptationRequest."
        )
    connectivity = mesh.connectivity
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not isinstance(connectivity, PolyhedralConnectivity)
    ):
        raise ValueError(
            "Polyhedral adaptation requires a three-dimensional polyhedral mesh."
        )
    geometry = (
        CellGeometrySpec.affine(mesh) if source_geometry is None else source_geometry
    )
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be CellGeometrySpec or None.")
    _require_piecewise_linear_geometry(mesh, geometry)
    coverage_inputs = (
        source_coverage_witness,
        source_distance_witness,
        source_coverage_profile,
    )
    if any(value is not None for value in coverage_inputs) and not all(
        value is not None for value in coverage_inputs
    ):
        raise ValueError(
            "Coverage renewal requires the retained witness, distance witness, and analytic profile together."
        )
    if source_coverage_witness is not None and not isinstance(
        source_coverage_witness,
        (RestrictedDelaunayBoundaryWitness, RestrictedDelaunayCoverageTransportWitness),
    ):
        raise TypeError(
            "source_coverage_witness has an unsupported restricted coverage type."
        )
    if source_distance_witness is not None and not isinstance(
        source_distance_witness, SourceBoundaryDistanceWitness
    ):
        raise TypeError("source_distance_witness must be SourceBoundaryDistanceWitness.")
    if source_coverage_profile is not None and not isinstance(
        source_coverage_profile, AnalyticImplicitProfile
    ):
        raise TypeError("source_coverage_profile must be AnalyticImplicitProfile.")
    geometry.resolve(mesh)
    classes = _classes_by_cell(mesh, cell_classes)
    protected = _protected_faces(mesh, protected_face_ids)
    policy = (
        CommonRefinementPolicy()
        if common_refinement_policy is None
        else common_refinement_policy
    )
    if not isinstance(policy, CommonRefinementPolicy):
        raise TypeError(
            "common_refinement_policy must be CommonRefinementPolicy or None."
        )
    construction: PolyhedralConstruction | None = None
    changed: tuple[int, ...]
    maximum_coordinate_rounding_error = 0.0
    if request.operation is PolyhedralAdaptationOperation.PLANE_SPLIT:
        (
            candidate,
            exact_coordinates,
            supports,
            parent_map,
            target_classes,
            changed,
            maximum_coordinate_rounding_error,
        ) = _split_candidate(mesh, geometry, request, protected, classes)
        relations = _cell_relations_from_parents(
            mesh, candidate, parent_map, request.operation
        )
        target_geometry_source = geometry
    elif request.operation is PolyhedralAdaptationOperation.AGGLOMERATE:
        (
            candidate,
            exact_coordinates,
            supports,
            parent_map,
            target_classes,
            changed,
        ) = _agglomerate_candidate(mesh, geometry, request, protected, classes)
        relations = _cell_relations_from_parents(
            mesh, candidate, parent_map, request.operation
        )
        target_geometry_source = geometry
    else:
        if protected:
            raise PolyhedralAdaptationError(
                "Site regeneration cannot preserve selected individual face IDs.",
                category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                entity_ids=tuple(sorted(protected)),
            )
        candidate, construction = _regenerate_candidate(mesh, request)
        exact_coordinates = ()
        supports = None
        parent_map = {}
        target_classes = {}
        changed_connectivity = candidate.connectivity
        if not isinstance(changed_connectivity, PolyhedralConnectivity):
            raise TypeError("Regenerated target lost polyhedral connectivity.")
        changed = tuple(
            int(value) for value in np.asarray(changed_connectivity.cell_global_ids)
        )
        relations = ()
    edit = build_polyhedral_topology_edit(mesh, candidate, supports, relations)
    target, lineage, stencil = assemble_topology_edit(
        mesh,
        edit,
        numeric_version=candidate.numeric_version,
    )
    if construction is not None:
        target_geometry = CellGeometrySpec.power(target, construction.coordinate_source)
    else:
        target_geometry = _bind_exact_polyhedral_geometry(
            target, target_geometry_source, exact_coordinates
        )
    refinement, source_fv, target_fv, remap = _physical_transfer(mesh, target, policy)
    if construction is not None:
        target_classes, parent_map, overlap_relations = _region_classes_from_overlap(
            mesh, target, refinement, classes
        )
        if overlap_relations:
            edit = build_polyhedral_topology_edit(
                mesh,
                candidate,
                None,
                overlap_relations,
            )
            target, lineage, stencil = assemble_topology_edit(
                mesh,
                edit,
                numeric_version=candidate.numeric_version,
            )
            target_geometry = CellGeometrySpec.power(
                target, construction.coordinate_source
            )
            refinement, source_fv, target_fv, remap = _physical_transfer(
                mesh, target, policy
            )
    if construction is not None:
        candidate_connectivity = candidate.connectivity
        target_connectivity = target.connectivity
        if (
            not isinstance(candidate_connectivity, PolyhedralConnectivity)
            or not isinstance(target_connectivity, PolyhedralConnectivity)
            or not np.array_equal(
                candidate_connectivity.cell_global_ids,
                target_connectivity.cell_global_ids,
            )
            or not np.array_equal(candidate.coordinates, target.coordinates)
            or not np.array_equal(candidate.vertex_global_ids, target.vertex_global_ids)
        ):
            raise ValueError(
                "Regenerated construction rows do not match the target mesh."
            )
        generated_connectivity = construction.mesh.connectivity
        if not isinstance(generated_connectivity, PolyhedralConnectivity) or any(
            not np.array_equal(
                getattr(generated_connectivity, field),
                getattr(target_connectivity, field),
            )
            for field in (
                "edges",
                "face_vertex_offsets",
                "face_vertex_values",
                "cell_face_offsets",
                "cell_face_values",
                "cell_face_sign_values",
                "face_owner",
                "face_neighbor",
            )
        ):
            raise ValueError(
                "Regenerated source incidence changed during target binding."
            )
        construction = replace(construction, mesh=target, geometry=target_geometry)
    target_connectivity = target.connectivity
    if not isinstance(target_connectivity, PolyhedralConnectivity):
        raise TypeError("An adapted target lost polyhedral connectivity.")
    target_ids = np.asarray(target_connectivity.cell_global_ids, dtype=np.int64)
    if not target_classes:
        target_classes = {
            int(identifier): classes[parent_map[int(identifier)][0]]
            for identifier in target_ids
        }
    if not parent_map:
        parent_map = {
            int(identifier): (int(identifier),)
            for identifier in target_ids
            if int(identifier) in classes
        }
    hierarchy = _hierarchy(mesh, target, parent_map, target_classes)
    if (
        source_coverage_witness is None
        or source_distance_witness is None
        or source_coverage_profile is None
    ):
        coverage_renewal = None
    else:
        coverage_renewal = _renew_restricted_coverage(
            mesh,
            geometry,
            classes,
            source_coverage_witness,
            source_distance_witness,
            source_coverage_profile,
            target,
            target_geometry,
            target_classes,
            refinement,
        )
    residual = _measure_residual(source_fv, target_fv, policy.coverage_tolerance)
    site_ids = () if construction is None else construction.site_ids
    site_points = (
        None
        if construction is None
        else np.asarray(construction.site_points, dtype=np.float64)
    )
    site_weights = (
        None
        if construction is None
        else np.asarray(construction.site_weights, dtype=np.float64)
    )
    construction_id = None if construction is None else construction.construction_id
    evidence_id = canonical_fingerprint(
        {
            "kind": "polyhedral-adaptation-evidence",
            "request": request.request_id,
            "source_mesh": mesh.mesh_id,
            "target_mesh": target.mesh_id,
            "changed_cell_ids": array_tree_fingerprint(
                np.asarray(changed, dtype=np.int64)
            ),
            "target_region_classes": array_tree_fingerprint(hierarchy.region_classes),
            "common_refinement": refinement.refinement_id,
            "conservative_remap": remap.plan_id,
            "measure_residual": residual,
            "coordinate_rounding_error": maximum_coordinate_rounding_error,
            "construction": construction_id,
            "site_ids": site_ids,
            "sites": array_tree_fingerprint((site_points, site_weights)),
            "coverage_renewal": (
                None if coverage_renewal is None else coverage_renewal.renewal_id
            ),
        }
    )
    evidence = PolyhedralAdaptationEvidence(
        request.operation,
        mesh.mesh_id,
        target.mesh_id,
        np.asarray(changed, dtype=np.int64),
        hierarchy.region_classes,
        refinement.refinement_id,
        remap.plan_id,
        residual,
        maximum_coordinate_rounding_error,
        construction_id,
        site_ids,
        site_points,
        site_weights,
        None if coverage_renewal is None else coverage_renewal.renewal_id,
        evidence_id,
    )
    return PolyhedralAdaptationOutcome(
        edit,
        target,
        target_geometry,
        lineage,
        stencil,
        hierarchy,
        evidence,
        refinement,
        remap,
        coverage_renewal,
        construction,
    )


__all__ = [
    "PolyhedralAdaptationEvidence",
    "PolyhedralCoverageRenewal",
    "PolyhedralAdaptationHierarchy",
    "PolyhedralAdaptationOperation",
    "PolyhedralAdaptationOutcome",
    "PolyhedralAdaptationRequest",
    "adapt_polyhedral_mesh",
]
