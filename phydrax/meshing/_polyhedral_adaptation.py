#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native polyhedral edits with exact source and periodic quotient ownership.

Splitting/agglomeration retain actual source restrictions and scientific orbits.
Regeneration admits a new power source only through complete common refinement
and unchanged original domain/material/periodic-control authority. Scalar nodal
stencils are restricted to known source-vertex/edge functionals; geometric convex
actions do not evaluate a general virtual-element face or interior field.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from enum import StrEnum
from fractions import Fraction
from typing import assert_never, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import clip_box_halfspaces, MeshcoreError
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import polyhedral_connectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
    PreparedCommonRefinement,
)
from ..typing import ConvertibleToArray
from ._association import GeometryAssociation, GeometryAssociationKind
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._controls import PeriodicConstraint
from ._lineage import EntityLineageKind
from ._polyhedral_generation import (
    _polyhedral_connectivity,
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
    PolyhedralGenerationError,
)
from ._result import CellMeshingResult
from ._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    PeriodicNonnestedGeometryAuthority,
    PolyhedralTopologyEditBlock,
)
from ._volume_generation import PiecewiseLinearComplex


class PolyhedralAdaptationOperation(StrEnum):
    PLANE_SPLIT = "plane_split"
    AGGLOMERATE = "agglomerate"
    REGENERATE = "regenerate"


class PolyhedralMeshAdaptation(StrictModule, NonTrainableState):
    """Explicit plane, connected-cell or site-regeneration request.

    Plane subdivision propagates each plane through the source, so a shared face
    is never split only on one side. Site insertion/removal is expressed by the
    actual successor site/weight arrays; site identity is not a nodal stencil.
    """

    operation: PolyhedralAdaptationOperation = eqx.field(static=True)
    planes: Array
    agglomerations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    complex_: PiecewiseLinearComplex | None
    sites: Array | None
    weights: Array | None
    periodic_constraints: tuple[PeriodicConstraint, ...]
    schedule: NativePolyhedralSchedule = eqx.field(static=True)
    coordinate_tolerance: float = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        operation: PolyhedralAdaptationOperation,
        *,
        planes: ConvertibleToArray = (),
        agglomerations: tuple[tuple[int, ...], ...] = (),
        complex_: PiecewiseLinearComplex | None = None,
        sites: ConvertibleToArray | None = None,
        weights: ConvertibleToArray | None = None,
        schedule: NativePolyhedralSchedule | None = None,
        coordinate_tolerance: float = 0.0,
        periodic_constraints: tuple[PeriodicConstraint, ...] = (),
    ) -> None:
        if not isinstance(operation, PolyhedralAdaptationOperation):
            raise TypeError("operation must be PolyhedralAdaptationOperation.")
        constraints = tuple(periodic_constraints)
        if not all(isinstance(value, PeriodicConstraint) for value in constraints):
            raise TypeError(
                "periodic_constraints must contain PeriodicConstraint values."
            )
        if constraints and operation is not PolyhedralAdaptationOperation.REGENERATE:
            raise ValueError(
                "Explicit periodic construction constraints belong to site regeneration."
            )
        planes_ = np.asarray(planes, dtype=np.float64).reshape((-1, 4))
        groups = tuple(tuple(sorted(group)) for group in agglomerations)
        if any(
            len(group) < 2
            or len(set(group)) != len(group)
            or any(identifier < 0 for identifier in group)
            for group in groups
        ):
            raise ValueError(
                "Agglomerations need at least two distinct nonnegative cell IDs."
            )
        if len({identifier for group in groups for identifier in group}) != sum(
            map(len, groups)
        ):
            raise ValueError("Agglomeration groups must be disjoint.")
        if not np.all(np.isfinite(planes_)) or np.any(
            np.linalg.norm(planes_[:, :3], axis=1) == 0
        ):
            raise ValueError("Split planes need finite nonzero normals and offsets.")
        if not np.isfinite(coordinate_tolerance) or coordinate_tolerance < 0:
            raise ValueError("coordinate_tolerance must be finite and nonnegative.")
        sites_ = None if sites is None else np.asarray(sites, dtype=np.float64)
        weights_ = None if weights is None else np.asarray(weights, dtype=np.float64)
        if sites_ is not None and (
            sites_.ndim != 2
            or sites_.shape[1] != 3
            or sites_.shape[0] == 0
            or not np.all(np.isfinite(sites_))
        ):
            raise ValueError("Regenerated sites require finite (N,3) coordinates.")
        if weights_ is not None and (
            sites_ is None
            or weights_.shape != (sites_.shape[0],)
            or not np.all(np.isfinite(weights_))
        ):
            raise ValueError("Weights require one finite value per regenerated site.")
        schedule_ = NativePolyhedralSchedule() if schedule is None else schedule
        if not isinstance(schedule_, NativePolyhedralSchedule):
            raise TypeError("schedule must be NativePolyhedralSchedule.")
        match operation:
            case PolyhedralAdaptationOperation.PLANE_SPLIT:
                valid = (
                    planes_.shape[0] > 0
                    and not groups
                    and complex_ is None
                    and sites_ is None
                )
            case PolyhedralAdaptationOperation.AGGLOMERATE:
                valid = (
                    bool(groups)
                    and planes_.shape[0] == 0
                    and complex_ is None
                    and sites_ is None
                )
            case PolyhedralAdaptationOperation.REGENERATE:
                valid = (
                    isinstance(complex_, PiecewiseLinearComplex)
                    and sites_ is not None
                    and not groups
                    and planes_.shape[0] == 0
                )
        if not valid:
            raise ValueError(
                "The selected polyhedral operation requires only its own controls."
            )
        self.operation = operation
        self.planes = jnp.asarray(planes_)
        self.agglomerations = tuple(sorted(groups))
        self.complex_ = complex_
        self.sites = None if sites_ is None else jnp.asarray(sites_)
        self.weights = None if weights_ is None else jnp.asarray(weights_)
        self.periodic_constraints = constraints
        self.schedule = schedule_
        self.coordinate_tolerance = float(coordinate_tolerance)
        self.request_id = canonical_fingerprint(
            {
                "kind": "polyhedral-adaptation-request",
                "operation": operation.value,
                "planes": array_tree_fingerprint(planes_),
                "agglomerations": self.agglomerations,
                "source": None if complex_ is None else complex_.complex_id,
                "sites": None if sites_ is None else array_tree_fingerprint(sites_),
                "weights": None if weights_ is None else array_tree_fingerprint(weights_),
                "schedule": asdict(schedule_),
                "coordinate_tolerance": self.coordinate_tolerance,
                "periodic_constraints": tuple(
                    value.constraint_id for value in constraints
                ),
            }
        )


class PolyhedralAdaptationEvidence(NamedTuple):
    operation: PolyhedralAdaptationOperation
    source_cells: int
    target_cells: int
    plane_construction_residual: float
    common_refinement_id: str
    changed_cell_ids: np.ndarray
    construction_id: str | None
    site_points: np.ndarray | None
    site_weights: np.ndarray | None
    site_parents: np.ndarray | None
    cell_sites: np.ndarray | None
    cell_components: np.ndarray | None
    cell_regions: np.ndarray | None
    face_facets: np.ndarray | None
    feature_edge_sources: np.ndarray | None

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "polyhedral-adaptation-evidence",
                "operation": self.operation.value,
                "source_cells": self.source_cells,
                "target_cells": self.target_cells,
                "plane_construction_residual": self.plane_construction_residual,
                "common_refinement": self.common_refinement_id,
                "changed_cells": array_tree_fingerprint(self.changed_cell_ids),
                "construction": self.construction_id,
                "ancestry": array_tree_fingerprint(
                    (
                        self.site_points,
                        self.site_weights,
                        self.site_parents,
                        self.cell_sites,
                        self.cell_components,
                        self.cell_regions,
                        self.face_facets,
                        self.feature_edge_sources,
                    )
                ),
            }
        )


class PolyhedralAdaptationOutcome(NamedTuple):
    edit: CellTopologyEdit
    common_refinement: PreparedCommonRefinement
    evidence: PolyhedralAdaptationEvidence
    construction: PolyhedralConstruction | None
    geometry: CellGeometrySpec


class PolyhedralAdaptationError(MeshingFailure):
    """A rejected candidate retains scientific failure status and accepted state."""

    def __init__(
        self,
        message: str,
        /,
        *,
        category: MeshingFailureCategory = MeshingFailureCategory.COMPLIANCE_FAILED,
    ) -> None:
        super().__init__(category, message, stage="polyhedral_adaptation")


def regenerated_polyhedral_associations(
    source: CellMeshingResult,
    request: PolyhedralMeshAdaptation,
    construction: PolyhedralConstruction,
    target: CellMesh,
    /,
) -> tuple[GeometryAssociation, ...]:
    """Reissue source strata from the actual restricted-diagram carrier witness."""
    from .providers._native_polyhedral import _organization
    from .providers._native_sources import NativePlcSource

    if request.complex_ is None or not source.associations:
        raise ValueError(
            "Associated regeneration requires its authoritative PLC and source associations."
        )
    certificate = source.certification
    if (
        certificate is None
        or not certificate.passed
        or certificate.request.domain is None
    ):
        raise ValueError(
            "Associated regeneration requires current independent PLC coverage."
        )
    if certificate.request.domain.domain_id != construction.domain.domain_id:
        raise ValueError(
            "Regeneration cannot replace the accepted authoritative PLC domain."
        )
    first = source.associations[0]
    for association in source.associations:
        if (
            association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != first.source_id
            or association.source_revision != first.source_revision
        ):
            raise ValueError(
                "Polyhedral regeneration requires one represented PLC source revision."
            )
        dimensions = [
            dimension
            for dimension in range(4)
            if source.mesh.entity_set(dimension).entity_set_id
            == association.target_entity_set_id
        ]
        if len(dimensions) != 1:
            raise ValueError(
                "Source association lacks an authoritative mesh entity binding."
            )
        association.validate_target(source.mesh.entity_set(dimensions[0]))
        if not np.all(np.asarray(association.resolved)) or np.any(
            np.asarray(association.ambiguous)
        ):
            raise ValueError("Regeneration requires resolved unambiguous source strata.")
    authority = NativePlcSource(request.complex_, first.source_id, first.source_revision)
    _, _, _, associations = _organization(
        authority,
        None,
        replace(construction, mesh=target),
        request.schedule.feature_tolerance,
    )
    return associations


def _stratum_overlap(
    source_points: np.ndarray,
    target_points: np.ndarray,
    dimension: int,
    work: list[int],
    /,
) -> bool:
    """Deciding surface/curve ancestry requires positive physical overlap."""
    if dimension == 0:
        return np.array_equal(source_points, target_points)
    if np.any(
        np.maximum(np.min(source_points, axis=0), np.min(target_points, axis=0))
        > np.minimum(np.max(source_points, axis=0), np.max(target_points, axis=0))
    ):
        return False
    if dimension == 1:
        a, b = source_points
        origin = tuple(
            value if isinstance(value, Fraction) else Fraction(float(value))
            for value in a
        )
        direction = tuple(
            (value if isinstance(value, Fraction) else Fraction(float(value))) - previous
            for value, previous in zip(b, origin, strict=True)
        )
        axis = next(index for index, value in enumerate(direction) if value != 0)
        parameters = []
        for point in target_points:
            delta = tuple(
                (value if isinstance(value, Fraction) else Fraction(float(value)))
                - previous
                for value, previous in zip(point, origin, strict=True)
            )
            parameter = delta[axis] / direction[axis]
            if any(
                value != parameter * component
                for value, component in zip(delta, direction, strict=True)
            ):
                return False
            parameters.append(parameter)
        return max(Fraction(0), min(parameters)) < min(Fraction(1), max(parameters))
    from .._meshcore import MeshcoreStatus, triangle_intersections
    from ..geometry._mesh_certificates import _loop_triangles

    triangles = []
    for points in (source_points, target_points):
        loops = np.arange(points.shape[0], dtype=np.int64)[None, :]
        indices, _, nonplanar, undecided = _loop_triangles(points, loops, work[0])
        if np.any(nonplanar | undecided):
            raise PolyhedralAdaptationError(
                "Source-stratum overlap requires decided planar faces."
            )
        triangles.append(points[indices])
    pairs = triangles[0].shape[0] * triangles[1].shape[0]
    if pairs > work[0]:
        raise PolyhedralAdaptationError(
            "Source-stratum overlap exceeds the geometry query budget.",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    work[0] -= pairs
    if source_points.dtype == object or target_points.dtype == object:
        from ..geometry._planar_coverage import (
            intersection,
            plane_key,
            project,
            rational_points,
        )

        for first in triangles[0]:
            for second in triangles[1]:
                a, b = rational_points(first), rational_points(second)
                key = plane_key(a)
                other = plane_key(b)
                if (
                    key is not None
                    and other is not None
                    and key[0] == other[0]
                    and intersection(project(a, key[1]), project(b, key[1])) > 0
                ):
                    return True
        return False
    for first in triangles[0]:
        for second in triangles[1]:
            _, counts, _, _, _, status = triangle_intersections(
                first[None, :], second[None, :]
            )
            if np.any(status != int(MeshcoreStatus.OK)):
                raise PolyhedralAdaptationError(
                    "Source-stratum overlap predicates are unresolved."
                )
            if counts[0] >= 3:
                return True
    return False


def _stratum_vertex_rows(mesh: CellMesh, dimension: int, /) -> tuple[np.ndarray, ...]:
    """Keep packed cyclic face order; lineage keys deliberately discard it."""
    c = mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError("Regenerated strata require canonical packed connectivity.")
    identifiers = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if dimension == 0:
        rows = key_rows(np.asarray(mesh.vertex_global_ids)[:, None], identifiers[:, None])
        return tuple(row[None] for row in rows)
    if dimension == 1:
        rows = key_rows(np.asarray(c.edge_global_ids)[:, None], identifiers[:, None])
        edges = np.asarray(c.edges, dtype=np.int64)
        return tuple(edges[row] for row in rows)
    rows = key_rows(np.asarray(c.face_global_ids)[:, None], identifiers[:, None])
    offsets, vertices = (
        np.asarray(c.face_vertex_offsets),
        np.asarray(c.face_vertex_values),
    )
    return tuple(vertices[offsets[row] : offsets[row + 1]] for row in rows)


def regenerated_polyhedral_relations(
    source: CellMeshingResult,
    target: CellMesh,
    edit: CellTopologyEdit,
    associations: tuple[GeometryAssociation, ...],
    maximum_work_units: int,
    /,
    *,
    maximum_geometry_queries: int,
    target_geometry: CellGeometrySpec,
) -> CellTopologyEdit:
    """Carry complete stratum membership; inconsistent partial scopes fail inheritance.

    Regenerated cells use independently measured physical overlap. Lower strata
    use explicit source-entity identity, not nearest coordinates or display names.
    Membership of a remeshed stratum must agree among every deciding predecessor.
    """
    relations = list(edit.relations)
    work = 0
    queries = [min(maximum_work_units, maximum_geometry_queries)]
    source.geometry.resolve(source.mesh)
    target_geometry.resolve(target)
    source_coordinates = np.asarray(source.geometry.source_coordinates(), dtype=object)
    target_coordinates = np.asarray(target_geometry.source_coordinates(), dtype=object)
    for dimension in range(3):
        old_keys, new_keys = (
            entity_keys(source.mesh, dimension),
            entity_keys(target, dimension),
        )
        old_ids = np.asarray(source.mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        new_ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
        source_vertices = _stratum_vertex_rows(source.mesh, dimension)
        target_vertices = _stratum_vertex_rows(target, dimension)
        previous = [
            value
            for value in source.associations
            if value.target_entity_set_id
            == source.mesh.entity_set(dimension).entity_set_id
        ]
        current = [
            value
            for value in associations
            if value.target_entity_set_id == target.entity_set(dimension).entity_set_id
        ]
        if not previous or not current:
            continue
        if len(previous) != 1 or len(current) != 1:
            raise ValueError(
                "Regeneration requires one source association per mesh dimension."
            )
        old, new = previous[0], current[0]
        parent_rows, child_rows = [], []
        bank: dict[str, list[int]] = {}
        old_positions = key_rows(
            old_ids[:, None], np.asarray(old.target_global_ids)[:, None]
        )
        new_positions = key_rows(
            new_ids[:, None], np.asarray(new.target_global_ids)[:, None]
        )
        if np.any(old_positions < 0) or np.any(new_positions < 0):
            raise ValueError(
                "Source-stratum associations reference undeclared mesh entities."
            )
        for identity, position in zip(
            old.source_entity_ids, old_positions.tolist(), strict=True
        ):
            bank.setdefault(identity, []).append(position)
        work += old_positions.size + new_positions.size
        for row, identity in enumerate(new.source_entity_ids):
            parents = bank.get(identity, [])
            work += len(parents)
            if work > maximum_work_units:
                raise PolyhedralAdaptationError(
                    "Source-stratum lineage exceeds the work budget.",
                    category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                )
            child = int(new_positions[row])
            target_points = target_coordinates[target_vertices[child]]
            for parent in parents:
                if queries[0] < 1:
                    raise PolyhedralAdaptationError(
                        "Source-stratum lineage exceeds the geometry query budget.",
                        category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    )
                queries[0] -= 1
                if _stratum_overlap(
                    source_coordinates[source_vertices[parent]],
                    target_points,
                    dimension,
                    queries,
                ):
                    parent_rows.append(parent)
                    child_rows.append(child)
        relations[dimension] = EntityRelations(
            dimension,
            old_keys[np.asarray(parent_rows, dtype=np.int64)],
            new_keys[np.asarray(child_rows, dtype=np.int64)],
            np.full(
                len(parent_rows), int(EntityLineageKind.SWAPPED_FROM), dtype=np.int32
            ),
        )
    return edit._replace(relations=tuple(relations))


def _loops(mesh: CellMesh) -> tuple[tuple[np.ndarray, ...], ...]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise ValueError("Polyhedral adaptation requires packed oriented connectivity.")
    co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int32)
    fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    fv = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
    return tuple(
        tuple(
            fv[fo[cf[position]] : fo[cf[position] + 1]][:: int(signs[position])]
            for position in range(co[row], co[row + 1])
        )
        for row in range(co.size - 1)
    )


def _split(
    mesh: CellMesh,
    plane: np.ndarray,
    protected: set[int],
    tolerance: float,
    maximum_cells: int,
    maximum_vertices: int,
    vertex_support: dict[int, set[int]],
) -> tuple[CellMesh, float]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise ValueError("Plane subdivision requires polyhedral connectivity.")
    loops = _loops(mesh)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    signed = coordinates @ plane[:3] - plane[3]
    cut = [
        index
        for index, cell in enumerate(loops)
        if np.min(signed[np.unique(np.concatenate(cell))])
        < 0
        < np.max(signed[np.unique(np.concatenate(cell))])
    ]
    if len(loops) + len(cut) > maximum_cells:
        raise PolyhedralAdaptationError(
            "Plane splitting exceeds the target cell budget.",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    for index in cut:
        for local, face in enumerate(loops[index]):
            if (
                np.min(signed[face]) < 0 < np.max(signed[face])
                and int(face_ids[cf[co[index] + local]]) in protected
            ):
                raise PolyhedralAdaptationError(
                    "Plane closure would subdivide a protected face."
                )
    normals, offsets, counts = [], [], []
    for index in cut:
        cell_normals, cell_offsets = [], []
        vertices = np.unique(np.concatenate(loops[index]))
        for face in loops[index]:
            points = coordinates[face]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            offset = float(normal @ points[0])
            residual = coordinates[vertices] @ normal - offset
            if np.any(residual > tolerance * np.linalg.norm(normal)):
                raise PolyhedralAdaptationError(
                    "Native halfspace subdivision requires convex planar source cells."
                )
            cell_normals.append(normal)
            cell_offsets.append(offset)
        for sign in (1.0, -1.0):
            normals.append(
                np.concatenate((np.asarray(cell_normals), plane[None, :3] * sign))
            )
            offsets.append(np.asarray([*cell_offsets, plane[3] * sign], dtype=np.float64))
            counts.append(len(cell_normals) + 1)
    if not cut:
        return mesh, 0.0
    width = max(counts)
    n = np.zeros((len(normals), width, 3), dtype=np.float64)
    h = np.zeros((len(normals), width), dtype=np.float64)
    for index, (normal, offset) in enumerate(zip(normals, offsets, strict=True)):
        n[index, : normal.shape[0]], h[index, : offset.size] = normal, offset
    lower, upper = np.min(coordinates, axis=0), np.max(coordinates, axis=0)
    extent = np.max(upper - lower)
    native = clip_box_halfspaces(
        lower - extent, upper + extent, n, h, np.asarray(counts, dtype=np.int32)
    )
    (
        points,
        point_counts,
        face_offsets,
        face_labels,
        face_vertices,
        face_counts,
        _,
        _,
        status,
    ) = native
    if np.any(status != 0):
        raise PolyhedralAdaptationError(
            f"Native plane clipping failed with statuses {tuple(int(value) for value in status)}."
        )
    global_points = list(coordinates)
    global_ids = list(vertex_ids)
    point_keys: dict[tuple[int, ...], int] = {}
    next_vertex = int(np.max(vertex_ids)) + 1
    next_cell = int(np.max(cell_ids)) + 1
    new_loops, new_ids = [], []
    residual = 0.0
    cut_positions = {cell: index for index, cell in enumerate(cut)}
    for cell, original in enumerate(loops):
        if cell not in cut_positions:
            new_loops.append(original)
            new_ids.append(int(cell_ids[cell]))
            continue
        for side in range(2):
            item = 2 * cut_positions[cell] + side
            incident: list[list[int]] = [[] for _ in range(point_counts[item])]
            for face in range(face_counts[item]):
                label = int(face_labels[item, face])
                for vertex in face_vertices[
                    item, face_offsets[item, face] : face_offsets[item, face + 1]
                ]:
                    incident[int(vertex)].append(label)
            local_to_global = []
            for vertex, labels in enumerate(incident):
                source_faces = [
                    original[label] for label in labels if 0 <= label < len(original)
                ]
                if not source_faces:
                    raise PolyhedralAdaptationError(
                        "A clipped vertex lacks authoritative source-face ancestry."
                    )
                support = set(int(value) for value in source_faces[0])
                for face in source_faces[1:]:
                    support.intersection_update(int(value) for value in face)
                if len(support) == 1:
                    global_vertex = next(iter(support))
                    error = float(
                        np.max(np.abs(points[item, vertex] - coordinates[global_vertex]))
                    )
                elif len(support) == 2 and len(original) in labels:
                    key = tuple(sorted(int(vertex_ids[value]) for value in support))
                    first, second = sorted(support)
                    origin = tuple(Fraction(float(value)) for value in coordinates[first])
                    direction = tuple(
                        Fraction(float(value)) - start
                        for value, start in zip(coordinates[second], origin, strict=True)
                    )
                    normal = tuple(Fraction(float(value)) for value in plane[:3])
                    denominator = sum(
                        value * delta
                        for value, delta in zip(normal, direction, strict=True)
                    )
                    if denominator == 0:
                        raise PolyhedralAdaptationError(
                            "A cut edge is exactly parallel to its plane."
                        )
                    parameter = (
                        Fraction(float(plane[3]))
                        - sum(
                            value * start
                            for value, start in zip(normal, origin, strict=True)
                        )
                    ) / denominator
                    exact = tuple(
                        start + parameter * delta
                        for start, delta in zip(origin, direction, strict=True)
                    )
                    construction_error = max(
                        abs(Fraction(float(value)) - expected)
                        for value, expected in zip(
                            points[item, vertex], exact, strict=True
                        )
                    )
                    if construction_error > Fraction(tolerance):
                        raise PolyhedralAdaptationError(
                            "A native cut point exceeds its exact source-edge construction bound."
                        )
                    construction_bound = float(construction_error)
                    if Fraction(construction_bound) < construction_error:
                        construction_bound = float(
                            np.nextafter(construction_bound, np.inf)
                        )
                    if key not in point_keys:
                        if len(global_points) >= maximum_vertices:
                            raise PolyhedralAdaptationError(
                                "Plane splitting exceeds the target vertex budget.",
                                category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                            )
                        point_keys[key] = len(global_points)
                        global_points.append(points[item, vertex])
                        global_ids.append(next_vertex)
                        vertex_support[next_vertex] = set().union(
                            *(vertex_support[int(vertex_ids[value])] for value in support)
                        )
                        next_vertex += 1
                    global_vertex = point_keys[key]
                    error = max(
                        construction_bound,
                        float(
                            np.max(
                                np.abs(
                                    points[item, vertex] - global_points[global_vertex]
                                )
                            )
                        ),
                    )
                else:
                    raise PolyhedralAdaptationError(
                        "Clipping construction has ambiguous source vertex/edge ancestry."
                    )
                residual = max(residual, error)
                local_to_global.append(global_vertex)
            faces = tuple(
                np.asarray(
                    [
                        local_to_global[int(value)]
                        for value in face_vertices[
                            item, face_offsets[item, face] : face_offsets[item, face + 1]
                        ]
                    ],
                    dtype=np.int32,
                )
                for face in range(face_counts[item])
            )
            new_loops.append(faces)
            new_ids.append(next_cell)
            next_cell += 1
    if residual > tolerance:
        raise PolyhedralAdaptationError(
            "Shared-edge constructions exceed the explicit coordinate tolerance."
        )
    return CellMesh.from_polyhedra(
        np.asarray(global_points, dtype=np.float64),
        new_loops,
        vertex_global_ids=np.asarray(global_ids, dtype=np.int64),
        cell_global_ids=np.asarray(new_ids, dtype=np.int64),
    ), residual


def _agglomerate(
    mesh: CellMesh,
    groups: tuple[tuple[int, ...], ...],
    classes: np.ndarray,
    protected: set[int],
) -> CellMesh:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise ValueError("Agglomeration requires packed oriented incidence.")
    loops = _loops(mesh)
    ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    index = {int(identifier): row for row, identifier in enumerate(ids)}
    co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    owner, neighbor = (
        np.asarray(connectivity.face_owner),
        np.asarray(connectivity.face_neighbor),
    )
    removed = set(identifier for group in groups for identifier in group)
    if not removed.issubset(index):
        raise ValueError("Agglomeration IDs must name current cells.")
    cells, cell_ids = (
        [cell for row, cell in enumerate(loops) if int(ids[row]) not in removed],
        [int(identifier) for identifier in ids if int(identifier) not in removed],
    )
    next_cell = int(np.max(ids)) + 1
    for group in groups:
        rows = {index[identifier] for identifier in group}
        if np.unique(classes[list(rows)]).size != 1:
            raise PolyhedralAdaptationError(
                "Agglomeration crosses a region or feature class."
            )
        visited, frontier = set(), [min(rows)]
        while frontier:
            row = frontier.pop()
            if row in visited:
                continue
            visited.add(row)
            for face in cf[co[row] : co[row + 1]]:
                adjacent = int(neighbor[face]) if owner[face] == row else int(owner[face])
                if adjacent in rows and adjacent not in visited:
                    frontier.append(adjacent)
        if visited != rows:
            raise PolyhedralAdaptationError("Agglomeration must be face-connected.")
        boundary = []
        for row in sorted(rows):
            for local, face in enumerate(cf[co[row] : co[row + 1]]):
                if int(owner[face]) in rows and int(neighbor[face]) in rows:
                    if int(face_ids[face]) in protected:
                        raise PolyhedralAdaptationError(
                            "Agglomeration removes a protected face."
                        )
                else:
                    boundary.append(loops[row][local])
        cells.append(tuple(boundary))
        cell_ids.append(next_cell)
        next_cell += 1
    used = np.unique(np.concatenate([face for cell in cells for face in cell]))
    remap = np.full(mesh.coordinates.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return CellMesh.from_polyhedra(
        np.asarray(mesh.coordinates)[used],
        tuple(tuple(remap[face] for face in cell) for cell in cells),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids)[used],
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
    )


def _topology_edit(
    source: CellMesh,
    target: CellMesh,
    refinement: PreparedCommonRefinement | None,
    vertex_support: dict[int, set[int]],
) -> CellTopologyEdit:
    loops = _loops(target)
    blocks = []
    offset = 0
    source_kinds = {block.name: block.cell_kind for block in source.blocks}
    for block in target.blocks:
        connectivity = polyhedral_connectivity(
            loops[offset : offset + block.cell_count],
            target.coordinates.shape[0],
            vertex_global_ids=target.vertex_global_ids,
            cell_global_ids=block.global_ids,
        )
        blocks.append(
            PolyhedralTopologyEditBlock(
                block.name,
                source_kinds.get(block.name),
                connectivity,
                np.asarray(block.global_ids, dtype=np.int64),
            )
        )
        offset += block.cell_count
    source_ids = (
        np.zeros(0, dtype=np.int64)
        if refinement is None
        else np.asarray(refinement.source_cell_global_ids, dtype=np.int64)
    )
    target_ids = (
        np.zeros(0, dtype=np.int64)
        if refinement is None
        else np.asarray(refinement.target_cell_global_ids, dtype=np.int64)
    )
    relations = []
    for dimension in range(4):
        keys = entity_keys(source, dimension)
        if dimension == 3:
            source_cells = (
                np.zeros(0, dtype=np.int64)
                if refinement is None
                else np.asarray(refinement.source_cells)
            )
            target_cells = (
                np.zeros(0, dtype=np.int64)
                if refinement is None
                else np.asarray(refinement.target_cells)
            )
            relations.append(
                EntityRelations(
                    3,
                    source_ids[source_cells][:, None],
                    target_ids[target_cells][:, None],
                    np.full(
                        source_cells.size,
                        int(EntityLineageKind.SWAPPED_FROM),
                        dtype=np.int32,
                    ),
                )
            )
        else:
            target_keys = entity_keys(target, dimension)
            parents, children = [], []
            if dimension in (1, 2):
                for target_key in target_keys:
                    values = [int(value) for value in target_key if value >= 0]
                    support = set().union(
                        *(vertex_support.get(value, set()) for value in values)
                    )
                    if not support:
                        continue
                    for source_key in keys:
                        if support.issubset(
                            set(int(value) for value in source_key if value >= 0)
                        ) and set(values) != set(
                            int(value) for value in source_key if value >= 0
                        ):
                            parents.append(source_key)
                            children.append(target_key)
            relations.append(
                EntityRelations(
                    dimension,
                    np.asarray(parents, dtype=np.int64).reshape((-1, keys.shape[1])),
                    np.asarray(children, dtype=np.int64).reshape(
                        (-1, target_keys.shape[1])
                    ),
                    np.full(
                        len(parents), int(EntityLineageKind.REFINED_FROM), dtype=np.int32
                    ),
                )
            )
    source_vertices = set(int(value) for value in np.asarray(source.vertex_global_ids))
    vertices = np.asarray(target.vertex_global_ids, dtype=np.int64)
    valid = np.asarray(
        [int(value) in source_vertices for value in vertices], dtype=np.bool_
    )[:, None]
    return CellTopologyEdit(
        "local_reconnection",
        np.asarray(target.coordinates),
        vertices,
        tuple(blocks),
        np.where(valid, vertices[:, None], -1),
        valid.astype(np.float64),
        valid,
        tuple(relations),
        target_periodic_topology=target.periodic_topology,
    )


def _power_source(
    geometry: CellGeometrySpec, /
) -> (
    ExactPowerCellGeometrySource
    | ExactPowerCellGeometryRestrictionSource
    | ExactPowerCellGeometryLinearActionSource
    | None
):
    """The power construction of a polyhedral geometry; PLC sources are not power polyhedra."""
    match geometry.exact_source:
        case (
            None
            | ExactPowerCellGeometrySource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource() as source
        ):
            return source
        case ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource():
            raise ValueError(
                "Polyhedral adaptation refuses exact PLC sources; they are not power polyhedra."
            )
        case invalid:
            assert_never(invalid)


def _edge_trace_stencil(
    source: CellMesh,
    target: CellMesh,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    edit: CellTopologyEdit,
    *,
    maximum_work: int,
    maximum_bytes: int,
) -> CellTopologyEdit:
    """Compose actual source restrictions only where degree-one edge traces are known.

    A geometric convex action on a face/interior is NOT a virtual-element field
    evaluation. Only retained vertices and original source edges are admitted.
    Site regeneration has no such source restriction and keeps unknown rows.
    """
    from ..discretization._coordinate_enclosure import coordinate_enclosure_budget
    from ..geometry._planar_coverage import _reserve_fraction_work

    original, current = _power_source(source_geometry), _power_source(target_geometry)
    if original is None or current is None:
        return edit
    chain = []
    visited = set()
    while current.source_id != original.source_id:
        if id(current) in visited:
            raise ValueError("Polyhedral edge trace source ancestry contains a cycle.")
        visited.add(id(current))
        if not isinstance(current, ExactPowerCellGeometryRestrictionSource):
            return edit
        chain.append(current)
        current = current.parent
    ledger = coordinate_enclosure_budget(maximum_work, maximum_bytes)
    source_ids = np.asarray(source.vertex_global_ids)
    ledger.reserve(len(source_ids), 256 + 256 * len(source_ids))
    actions = [{row: Fraction(1)} for row in range(len(source_ids))]
    for node in reversed(chain):
        prepared = node.prepare()
        parents = np.asarray(node.vertex_parents)
        ledger.reserve(len(parents), 256 + 384 * len(parents))
        successor = []
        for indices, coefficients in zip(parents, prepared.barycentric, strict=True):
            row = {}
            for parent, coefficient in zip(
                indices[indices >= 0], coefficients, strict=True
            ):
                for origin, weight in actions[int(parent)].items():
                    if coefficient == 0 or weight == 0:
                        continue
                    if origin not in row and len(row) == 2:
                        return edit
                    accumulated = row.get(origin, Fraction(0))
                    _reserve_fraction_work(((coefficient, weight, accumulated),), 2, 1, 2)
                    value = accumulated + coefficient * weight
                    if (
                        max(value.numerator.bit_length(), value.denominator.bit_length())
                        > original.maximum_bits
                    ):
                        raise PolyhedralAdaptationError(
                            "Edge trace action exceeds original integer-bit capacity.",
                            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        )
                    row[origin] = value
            successor.append({origin: value for origin, value in row.items() if value})
        actions = successor
    if len(actions) != target.coordinates.shape[0]:
        raise ValueError("Source edge trace actions omit target coordinate rows.")
    edge_rows = np.asarray(_polyhedral_connectivity(source).edges)
    ledger.reserve(edge_rows.size, 256 + 192 * len(edge_rows))
    edges = {frozenset(int(value) for value in edge) for edge in edge_rows}
    if any(len(row) != 1 and frozenset(row) not in edges for row in actions):
        return edit
    ledger.reserve(len(actions), 64 * len(actions))
    sources = np.full((len(actions), 2), -1, dtype=np.int64)
    weights = np.zeros((len(actions), 2), dtype=np.float64)
    for target_row, row in enumerate(actions):
        if sum(row.values(), Fraction(0)) != 1 or min(row.values()) < 0:
            raise ValueError("Source edge trace action is not a convex unit functional.")
        for slot, (origin, coefficient) in enumerate(sorted(row.items())):
            sources[target_row, slot] = source_ids[origin]
            weights[target_row, slot] = float(coefficient)
    return edit._replace(
        stencil_sources=sources, stencil_weights=weights, stencil_valid=sources >= 0
    )


def _periodic_source_target(
    source: CellMesh,
    target: CellMesh,
    geometry: CellGeometrySpec,
    *,
    maximum_work: int,
    maximum_bytes: int,
) -> CellMesh:
    """Reuse W11's exact seam-orbit owner for source-backed split/agglomeration."""
    from ..discretization._coordinate_enclosure import (
        coordinate_enclosure_budget,
        prepared_coordinate_source_bank,
    )
    from ..discretization._periodic_topology import PeriodicMeshTopology
    from ..geometry._planar_coverage import _reserve_fraction_work
    from ..geometry._triangulation import PeriodicPowerPreparation
    from ._polyhedral_generation import _seam_vertex_orbits
    from ._topology_edit import _power_domain_root

    periodic = source.periodic_topology
    if periodic is None:
        return target
    ledger = coordinate_enclosure_budget(maximum_work, maximum_bytes)
    with ledger.activate(), ledger.bound_stage(maximum_work, maximum_bytes):
        root = _power_domain_root(geometry)
        preparation = root.periodic_preparation
        if not isinstance(preparation, PeriodicPowerPreparation):
            raise ValueError(
                "Periodic polyhedral edits require the original image preparation."
            )
        points = prepared_coordinate_source_bank(geometry)
        ledger.reserve(0, 256 + 128 * len(points))
        lookup = {point: row for row, point in enumerate(points)}
        matrices = tuple(
            tuple(tuple(Fraction(float(value)) for value in row) for row in matrix)
            for matrix in preparation.generators
        )
        pairs = []
        for generator, matrix in enumerate(matrices):
            for row, point in enumerate(points):
                _reserve_fraction_work((point, *matrix), 36, 3, 8)
                image = tuple(
                    sum(matrix[axis][column] * point[column] for column in range(3))
                    + matrix[axis][3]
                    for axis in range(3)
                )
                other = lookup.get(image)
                if other is not None:
                    ledger.reserve(1, 128)
                    pairs.append((row, other, generator))
        representatives, shifts = _seam_vertex_orbits(
            points,
            np.asarray(pairs, dtype=np.int64).reshape((-1, 3)),
            np.asarray(target.vertex_global_ids),
            preparation,
            ledger,
        )
        descriptor = PeriodicMeshTopology(
            target,
            periodic.cell,
            representatives,
            shifts,
            actual_geometry=geometry,
        )
        ledger.charge_native_work(ledger.work_units - ledger.native_charged_work_units)
    return CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        polyhedral_connectivity=_polyhedral_connectivity(target),
        periodic_topology=descriptor,
        numeric_version=target.numeric_version,
    )


def _stage_polyhedral_target(
    source: CellMesh,
    target: CellMesh,
    geometry: CellGeometrySpec,
    support: dict[int, set[int]],
    *,
    numeric_version: str,
    maximum_work: int,
    maximum_bytes: int,
) -> CellMesh:
    from ._periodic import prepare_periodic_edit_target
    from ._topology_edit import prepare_topology_edit_target

    if source.periodic_topology is not None and target.periodic_topology is None:
        target = _periodic_source_target(
            source,
            target,
            geometry,
            maximum_work=maximum_work,
            maximum_bytes=maximum_bytes,
        )
    target = prepare_topology_edit_target(
        source,
        _topology_edit(source, target, None, support),
        numeric_version=numeric_version,
    )
    if source.periodic_topology is not None:
        target = prepare_periodic_edit_target(source, target)
    elif target.periodic_topology is not None:
        raise ValueError(
            "Polyhedral regeneration cannot silently introduce a periodic source law."
        )
    return target


def _periodic_polyhedral_edit(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    authority: PeriodicNonnestedGeometryAuthority | None = None,
) -> CellTopologyEdit:
    from ._periodic import periodic_vertex_orbit_witness

    descriptor = target.periodic_topology
    if source.periodic_topology is None:
        return edit
    if descriptor is None:
        raise ValueError(
            "Polyhedral periodic publication omits the actual target quotient source."
        )
    witness = periodic_vertex_orbit_witness(
        source,
        target,
        np.asarray(target.vertex_global_ids)[
            np.asarray(descriptor.vertex_representatives)
        ],
        np.asarray(descriptor.vertex_shifts),
        None,
        nonnested_geometry=authority,
    )
    return edit._replace(periodic_orbits=witness)


def _regenerated_target(
    source: CellMesh, construction: PolyhedralConstruction
) -> tuple[CellMesh, CellGeometrySpec]:
    """Rebind scientific IDs without rebuilding the producer's exact source/actions."""
    from ..discretization._periodic_topology import PeriodicMeshTopology

    generated = construction.mesh
    source_connectivity = source.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral regeneration requires packed source identities.")
    first_vertex = int(np.max(np.asarray(source.vertex_global_ids))) + 1
    first_cell = int(np.max(np.asarray(source_connectivity.cell_global_ids))) + 1
    vertex_count = generated.coordinates.shape[0]
    cell_count = generated.connectivity.cell_count
    if (
        max(first_vertex + vertex_count - 1, first_cell + cell_count - 1)
        > np.iinfo(np.int64).max
    ):
        raise PolyhedralAdaptationError(
            "Regeneration exceeds scientific identity capacity.",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    target = CellMesh.from_polyhedra(
        generated.coordinates,
        _loops(generated),
        vertex_global_ids=np.arange(
            first_vertex, first_vertex + vertex_count, dtype=np.int64
        ),
        cell_global_ids=np.arange(first_cell, first_cell + cell_count, dtype=np.int64),
    )
    exact = _power_source(construction.geometry)
    if exact is None:
        raise ValueError("Regeneration requires its actual exact power source.")
    geometry = CellGeometrySpec.power(target, exact)
    periodic = generated.periodic_topology
    if periodic is not None:
        descriptor = PeriodicMeshTopology(
            target,
            periodic.cell,
            periodic.vertex_representatives,
            periodic.vertex_shifts,
            actual_geometry=geometry,
        )
        target = CellMesh(
            target.coordinates,
            target.blocks,
            vertex_global_ids=target.vertex_global_ids,
            polyhedral_connectivity=_polyhedral_connectivity(target),
            periodic_topology=descriptor,
        )
        geometry = CellGeometrySpec.power(target, exact)
    return target, geometry


def adapt_polyhedral_mesh(
    mesh: CellMesh,
    request: PolyhedralMeshAdaptation,
    *,
    source_geometry: CellGeometrySpec | None = None,
    cell_classes: ConvertibleToArray | None = None,
    protected_face_ids: ConvertibleToArray = (),
    maximum_cells: int = 1 << 24,
    maximum_vertices: int = 1 << 22,
    maximum_work_units: int = 1 << 26,
    maximum_scratch_bytes: int = 8_000_000_000,
    source_id: str = "native-polyhedral",
    common_refinement_policy: CommonRefinementPolicy | None = None,
    numeric_version: str = "0",
) -> PolyhedralAdaptationOutcome:
    """Keep exact source/orbit/overlap preparations under one original coordinate ledger."""
    from ..discretization._coordinate_enclosure import coordinate_enclosure_budget

    ledger = coordinate_enclosure_budget(maximum_work_units, maximum_scratch_bytes)
    with ledger.activate(), ledger.bound_stage(maximum_work_units, maximum_scratch_bytes):
        outcome = _adapt_polyhedral_mesh(
            mesh,
            request,
            source_geometry=source_geometry,
            cell_classes=cell_classes,
            protected_face_ids=protected_face_ids,
            maximum_cells=maximum_cells,
            maximum_vertices=maximum_vertices,
            maximum_work_units=maximum_work_units,
            maximum_scratch_bytes=maximum_scratch_bytes,
            source_id=source_id,
            common_refinement_policy=common_refinement_policy,
            numeric_version=numeric_version,
        )
        ledger.charge_native_work(ledger.work_units - ledger.native_charged_work_units)
    return outcome


def _adapt_polyhedral_mesh(
    mesh: CellMesh,
    request: PolyhedralMeshAdaptation,
    *,
    source_geometry: CellGeometrySpec | None,
    cell_classes: ConvertibleToArray | None,
    protected_face_ids: ConvertibleToArray,
    maximum_cells: int,
    maximum_vertices: int,
    maximum_work_units: int,
    maximum_scratch_bytes: int,
    source_id: str,
    common_refinement_policy: CommonRefinementPolicy | None,
    numeric_version: str,
) -> PolyhedralAdaptationOutcome:
    """Prepare a bounded candidate and real complete overlap, never a P1 fiction."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("Polyhedral adaptation requires a packed CellMesh.")
    if not isinstance(request, PolyhedralMeshAdaptation):
        raise TypeError("request must be PolyhedralMeshAdaptation.")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral adaptation requires a packed CellMesh.")
    count = connectivity.cell_count
    classes = (
        np.zeros(count, dtype=np.int64)
        if cell_classes is None
        else np.asarray(cell_classes, dtype=np.int64)
    )
    if classes.shape != (count,):
        raise ValueError("cell_classes must align with concatenated polyhedral cells.")
    protected = set(
        int(value) for value in np.asarray(protected_face_ids, dtype=np.int64).reshape(-1)
    )
    residual = 0.0
    construction = None
    vertex_support = {
        int(identifier): {int(identifier)}
        for identifier in np.asarray(mesh.vertex_global_ids)
    }
    source_geometry = (
        CellGeometrySpec.affine(mesh) if source_geometry is None else source_geometry
    )
    source_geometry.resolve(mesh)
    target_geometry = source_geometry
    source_power = _power_source(source_geometry)
    match request.operation:
        case PolyhedralAdaptationOperation.PLANE_SPLIT:
            target = mesh
            for plane in np.asarray(request.planes):
                if _power_source(target_geometry) is None:
                    candidate, error = _split(
                        target,
                        plane,
                        protected,
                        request.coordinate_tolerance,
                        maximum_cells,
                        maximum_vertices,
                        vertex_support,
                    )
                    target_geometry = CellGeometrySpec.affine(candidate)
                    residual = max(residual, error)
                else:
                    from ._polyhedral_plane_restriction import restrict_polyhedral_plane

                    candidate, target_geometry = restrict_polyhedral_plane(
                        target,
                        target_geometry,
                        plane,
                        protected,
                        vertex_support,
                        maximum_cells=maximum_cells,
                        maximum_vertices=maximum_vertices,
                        maximum_work=maximum_work_units,
                        maximum_bytes=maximum_scratch_bytes,
                    )
                support = {
                    int(identifier): {int(identifier)}
                    for identifier in np.asarray(target.vertex_global_ids)
                }
                staged = _stage_polyhedral_target(
                    target,
                    candidate,
                    target_geometry,
                    support,
                    numeric_version="polyhedral-plane-candidate",
                    maximum_work=maximum_work_units,
                    maximum_bytes=maximum_scratch_bytes,
                )
                candidate_edit = _periodic_polyhedral_edit(
                    target, staged, _topology_edit(target, staged, None, support)
                )
                target, _, _ = assemble_topology_edit(
                    target, candidate_edit, numeric_version="polyhedral-plane-candidate"
                )
        case PolyhedralAdaptationOperation.AGGLOMERATE:
            target = _agglomerate(mesh, request.agglomerations, classes, protected)
            if source_power is None:
                target_geometry = CellGeometrySpec.affine(target)
            else:
                target_connectivity = target.connectivity
                if not isinstance(target_connectivity, PolyhedralConnectivity):
                    raise TypeError(
                        "Exact agglomeration requires packed polyhedral connectivity."
                    )
                used = np.unique(np.asarray(target_connectivity.cell_vertex_values))
                lowering = np.full(target.coordinates.shape[0], -1, dtype=np.int32)
                lowering[used] = np.arange(used.size, dtype=np.int32)
                # Agglomeration compacts away interior vertices. The exact
                # carrier still uses source rows, not the target's compact rows.
                source_rows = {
                    int(identifier): row
                    for row, identifier in enumerate(np.asarray(mesh.vertex_global_ids))
                }
                retained_source_rows = np.asarray(
                    [
                        source_rows[int(identifier)]
                        for identifier in np.asarray(target.vertex_global_ids)[used]
                    ],
                    dtype=np.int64,
                )
                source = ExactPowerCellGeometryRestrictionSource(
                    source_power,
                    np.empty((0, 4), dtype=np.float64),
                    np.column_stack(
                        (retained_source_rows, np.full(used.size, -1, dtype=np.int64))
                    ),
                    np.full(used.size, -1, dtype=np.int64),
                )
                target = CellMesh.from_polyhedra(
                    source.prepare().rounded_vertices,
                    tuple(
                        tuple(lowering[row] for row in cell) for cell in _loops(target)
                    ),
                    vertex_global_ids=np.asarray(target.vertex_global_ids)[used],
                    cell_global_ids=np.asarray(target_connectivity.cell_global_ids),
                )
                target_geometry = CellGeometrySpec.power(target, source)
        case PolyhedralAdaptationOperation.REGENERATE:
            if request.complex_ is None or request.sites is None:
                raise ValueError(
                    "Site regeneration requires an authoritative PLC and explicit sites."
                )
            if protected:
                raise PolyhedralAdaptationError(
                    "Site regeneration lacks a fixed protected-face construction contract."
                )
            schedule = replace(
                request.schedule,
                maximum_cells=min(maximum_cells, request.schedule.maximum_cells),
                maximum_vertices=min(maximum_vertices, request.schedule.maximum_vertices),
                maximum_work_units=min(
                    maximum_work_units, request.schedule.maximum_work_units
                ),
                maximum_scratch_bytes=min(
                    maximum_scratch_bytes, request.schedule.maximum_scratch_bytes
                ),
            )
            from .providers._native_polyhedral import _construction_failure

            try:
                construction = generate_polyhedral_volume(
                    request.complex_,
                    sites=request.sites,
                    weights=request.weights,
                    schedule=schedule,
                    source_id=source_id,
                    periodic_constraints=request.periodic_constraints,
                )
            except (MeshcoreError, PolyhedralGenerationError) as error:
                raise _construction_failure(error, "polyhedral_adaptation") from error
            target, target_geometry = _regenerated_target(mesh, construction)
    if target.connectivity.cell_count > maximum_cells:
        raise PolyhedralAdaptationError(
            "The polyhedral candidate exceeds the target cell budget.",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    if target.coordinates.shape[0] > maximum_vertices:
        raise PolyhedralAdaptationError(
            "The polyhedral candidate exceeds the target vertex budget.",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED,
        )
    policy = (
        CommonRefinementPolicy(overlap_simplices=True)
        if common_refinement_policy is None
        else common_refinement_policy
    )
    if (
        not isinstance(policy, CommonRefinementPolicy)
        or policy.coverage is not CommonRefinementCoverage.COMPLETE
    ):
        raise ValueError(
            "Polyhedral adaptation requires complete source and target overlap coverage."
        )
    target = _stage_polyhedral_target(
        mesh,
        target,
        target_geometry,
        vertex_support,
        numeric_version=numeric_version,
        maximum_work=maximum_work_units,
        maximum_bytes=maximum_scratch_bytes,
    )
    if construction is not None:
        power_source = _power_source(construction.geometry)
        if power_source is None:
            raise ValueError(
                "Regenerated polyhedra require their exact power source construction."
            )
    else:
        power_source = _power_source(target_geometry)
    target_geometry = (
        CellGeometrySpec.affine(target)
        if power_source is None
        else CellGeometrySpec.power(target, power_source)
    )
    common = prepare_common_refinement(
        mesh,
        target,
        policy=policy,
        source_geometry=source_geometry if source_power is not None else None,
        target_geometry=target_geometry if power_source is not None else None,
    )
    if common.status != CommonRefinementStatus.SUCCESS:
        raise PolyhedralAdaptationError(
            f"Polyhedral common refinement failed: {common.status.name}.",
            category=MeshingFailureCategory.RESOURCE_EXHAUSTED
            if common.status is CommonRefinementStatus.RESOURCE_LIMIT
            else MeshingFailureCategory.AUDIT_FAILED,
        )
    for row in range(common.target_cell_count):
        begin, end = np.asarray(common.target_offsets)[row : row + 2]
        owners = np.asarray(common.source_cells)[begin:end]
        if np.unique(classes[owners]).size != 1:
            raise PolyhedralAdaptationError(
                "The candidate mixes scientific cell classes."
            )
    edit = _topology_edit(mesh, target, common, vertex_support)
    edit = _edge_trace_stencil(
        mesh,
        target,
        source_geometry,
        target_geometry,
        edit,
        maximum_work=maximum_work_units,
        maximum_bytes=maximum_scratch_bytes,
    )
    if mesh.periodic_topology is not None:
        from ._topology_edit import prepare_periodic_nonnested_geometry

        authority = (
            None
            if construction is None
            else prepare_periodic_nonnested_geometry(
                mesh,
                target,
                source_geometry,
                target_geometry,
                common,
            )
        )
        edit = _periodic_polyhedral_edit(mesh, target, edit, authority)
        accepted, _, _ = assemble_topology_edit(
            mesh, edit, numeric_version=numeric_version
        )
        if accepted.mesh_id != target.mesh_id:
            raise ValueError(
                "Periodic publication changed the certified overlap target identity."
            )
    target_connectivity = target.connectivity
    if not isinstance(target_connectivity, PolyhedralConnectivity):
        raise TypeError(
            "Polyhedral adaptation must produce packed polyhedral connectivity."
        )
    source_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(target_connectivity.cell_global_ids, dtype=np.int64)
    evidence = PolyhedralAdaptationEvidence(
        request.operation,
        count,
        target_connectivity.cell_count,
        residual,
        common.refinement_id,
        np.setdiff1d(source_ids, target_ids),
        None if construction is None else construction.construction_id,
        None if construction is None else construction.diagram.points,
        None if construction is None else construction.diagram.weights,
        None if construction is None else construction.site_parents,
        None if construction is None else construction.cell_sites,
        None if construction is None else construction.component_ids,
        None if construction is None else construction.cell_regions,
        None if construction is None else construction.face_facets,
        None if construction is None else construction.feature_edge_sources,
    )
    return PolyhedralAdaptationOutcome(
        edit, common, evidence, construction, target_geometry
    )
