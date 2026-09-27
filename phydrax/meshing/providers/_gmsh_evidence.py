#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh source association, semantic organization, and connectivity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import jax.numpy as jnp
import numpy as np

from ...discretization import CellMesh
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...geometry.brep import BRepModel
from ...geometry.simplicial import TriangleMesh
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from .._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import MeshingStageKind
from ._gmsh_elements import _curve_corner_rows, _ElementRows, _local_connectivity
from ._gmsh_import import _CadEntityMap


@dataclass(frozen=True, slots=True)
class _EvidenceSection:
    """Ordered requested/achieved compliance entries and soft issues of one stage."""

    requested: tuple[tuple[str, float], ...] = ()
    achieved: tuple[tuple[str, float], ...] = ()
    issues: tuple[str, ...] = ()


def _boundary_association(
    source: BRepModel,
    boundary: CellMesh,
    tolerance_factor: float,
    /,
) -> tuple[GeometryAssociation, tuple[MeshZone, ...], MeshAttribute]:
    points = np.asarray(boundary.coordinates, dtype=np.float64)
    centroids = np.concatenate(
        [
            np.mean(points[np.asarray(block.vertices, dtype=np.int32)], axis=1)
            for block in boundary.blocks
        ]
    )
    query_mesh = TriangleMesh(
        source.mesh_vertices,
        source.mesh_faces,
        source_id=f"{source.report.source_id}:association-query",
    )
    query = query_mesh.query_index().query(jnp.asarray(centroids))
    triangle_ids = np.asarray(query.face_index, dtype=np.int32)
    source_faces = np.asarray(source.triangle_face_ids, dtype=np.int32)[triangle_ids]
    residuals = np.asarray(query.distance, dtype=np.float64)
    tolerance = max(
        source.report.linear_deflection * float(tolerance_factor),
        256.0 * np.finfo(np.float64).eps,
    )
    resolved = residuals <= tolerance
    target_set = boundary.entity_set(2)
    source_ids = tuple(
        f"{source.report.source_revision}:face:{int(index)}" for index in source_faces
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        target_set.entity_set_id,
        target_set.entity_ids,
        source_ids,
        residuals,
        resolved=resolved,
        exact=False,
        source_dimensions=np.full(source_faces.shape, 2, dtype=np.int8),
        source_indices=source_faces,
    )
    if not association.complete:
        failed = tuple(np.asarray(target_set.entity_ids)[~resolved])
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "Generated boundary faces could not be uniquely matched within tolerance.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            entity_ids=failed,
        )
    zones = []
    for face_id in np.unique(source_faces):
        selected = np.flatnonzero(source_faces == face_id)
        scope = MeshingScope(
            boundary.mesh_id,
            boundary.numeric_version,
            MeshingEntityKind.MESH,
            2,
            target_set.entity_set_id,
            np.asarray(target_set.entity_ids)[selected],
        )
        zones.append(MeshZone(f"brep-face-{int(face_id)}", MeshZoneRole.BOUNDARY, scope))
    all_scope = MeshingScope(
        boundary.mesh_id,
        boundary.numeric_version,
        MeshingEntityKind.MESH,
        2,
        target_set.entity_set_id,
        target_set.entity_ids,
    )
    attribute = MeshAttribute(
        "brep_face_index",
        MeshAttributeRole.GEOMETRY_CLASSIFICATION,
        all_scope,
        source_faces,
    )
    return association, tuple(zones), attribute


@dataclass(frozen=True, slots=True)
class _PlanarSurfaceEvidence:
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    associations: tuple[GeometryAssociation, ...]
    attributes: tuple[MeshAttribute, ...]
    cell_face_ids: np.ndarray
    mesh_edge_source: np.ndarray


def _polygon_edge_incidents(
    connectivity: PolygonalConnectivity, /
) -> tuple[tuple[int, ...], ...]:
    incidents = [[] for _ in np.asarray(connectivity.edges)]
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
    for cell_index, row in enumerate(cell_edges):
        for edge_index in row[valid[cell_index]]:
            incidents[int(edge_index)].append(cell_index)
    return tuple(tuple(values) for values in incidents)


def _planar_edge_patch_connected(
    connectivity: PolygonalConnectivity, edge_indices: np.ndarray, /
) -> bool:
    if edge_indices.size <= 1:
        return True
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    vertex_edges: dict[int, set[int]] = {}
    selected = {int(value) for value in edge_indices}
    for edge_index in selected:
        for vertex in edges[edge_index]:
            vertex_edges.setdefault(int(vertex), set()).add(edge_index)
    pending = [next(iter(selected))]
    visited = set()
    while pending:
        edge_index = pending.pop()
        if edge_index in visited:
            continue
        visited.add(edge_index)
        for vertex in edges[edge_index]:
            pending.extend(vertex_edges[int(vertex)] - visited)
    return visited == selected


def _planar_surface_evidence(
    gmsh: Any,
    source: BRepModel,
    mesh: CellMesh,
    specification: SurfaceMeshingSpec,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    node_tags: np.ndarray,
    source_to_corner: np.ndarray,
    cad_entities: _CadEntityMap,
    geometry_order: int,
    /,
) -> _PlanarSurfaceEvidence:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Semantic planar meshing requires PolygonalConnectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    surface_to_face = {
        surface: face for face, surface in enumerate(cad_entities.face_to_surface)
    }
    rows_by_name = {row.block_name: row for row in rows}
    owner_chunks = []
    for block in mesh.blocks:
        row = rows_by_name[block.name]
        entity_tags = row.entity_tags[row_orders[block.name]]
        owner_chunks.append(
            np.asarray(
                tuple(surface_to_face.get(int(tag), -1) for tag in entity_tags.tolist()),
                dtype=np.int32,
            )
        )
    cell_face_ids = np.concatenate(owner_chunks)
    cell_entity_set = mesh.entity_set(2)
    cell_ids = np.asarray(cell_entity_set.entity_ids, dtype=np.int64)
    if (
        cell_face_ids.shape != cell_ids.shape
        or np.any(cell_face_ids < 0)
        or not np.array_equal(
            cell_ids,
            np.concatenate(
                tuple(
                    np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks
                )
            ),
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CANONICALIZATION_FAILED,
            "Canonical planar cells lost their source-face ownership.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )

    curve_nodes, curve_tags = _curve_corner_rows(gmsh, geometry_order)
    local_nodes = _local_connectivity(node_tags, curve_nodes)
    corners = source_to_corner[local_nodes]
    if np.any(corners < 0):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "A source curve corner is absent from the canonical planar mesh.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    edge_rows = np.asarray(connectivity.edges, dtype=np.int32)
    edge_lookup: dict[tuple[int, int], int] = {}
    for index in range(edge_rows.shape[0]):
        first, second = np.take(edge_rows, index, axis=0)
        first, second = sorted((int(first), int(second)))
        edge_lookup[(first, second)] = index
    curve_to_edge = {curve: edge for edge, curve in enumerate(cad_entities.edge_to_curve)}
    mesh_edge_source = np.full((edge_rows.shape[0],), -1, dtype=np.int32)
    for row_index in range(corners.shape[0]):
        first, second = np.take(corners, row_index, axis=0)
        first, second = sorted((int(first), int(second)))
        mesh_edge = edge_lookup.get((first, second))
        source_edge = curve_to_edge.get(int(curve_tags[row_index]))
        if mesh_edge is None or source_edge is None or mesh_edge_source[mesh_edge] >= 0:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "A Gmsh curve element does not map uniquely to one canonical mesh edge.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        mesh_edge_source[mesh_edge] = source_edge
    if {int(value) for value in mesh_edge_source if value >= 0} != set(
        range(source.report.num_edges)
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Generated planar edges do not cover every source BRep edge.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    incidents = _polygon_edge_incidents(connectivity)
    mapped = np.flatnonzero(mesh_edge_source >= 0)
    for mesh_edge in mapped:
        source_edge = int(mesh_edge_source[mesh_edge])
        expected = set(source.topology.edge_faces[source_edge])
        actual = {int(cell_face_ids[cell]) for cell in incidents[int(mesh_edge)]}
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Canonical planar edge adjacency differs from source BRep incidence.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
    expected_boundary = np.zeros((edge_rows.shape[0],), dtype=np.bool_)
    expected_boundary[mapped] = np.asarray(
        [
            len(source.topology.edge_faces[int(mesh_edge_source[index])]) == 1
            for index in mapped
        ]
    )
    if not np.array_equal(
        np.asarray(connectivity.boundary_edges, dtype=np.bool_), expected_boundary
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Planar mesh boundary is not exactly the singly incident source CAD edges.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )

    face_association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        cell_entity_set.entity_set_id,
        cell_ids,
        tuple(
            f"{source.report.source_revision}:face:{int(face)}" for face in cell_face_ids
        ),
        np.zeros(cell_ids.shape, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(cell_ids.shape, 2, dtype=np.int8),
        source_indices=np.asarray(cell_face_ids, dtype=np.int64),
    )
    edge_entity_set = mesh.entity_set(1)
    edge_ids = np.asarray(edge_entity_set.entity_ids, dtype=np.int64)
    edge_association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        edge_entity_set.entity_set_id,
        edge_ids[mapped],
        tuple(
            f"{source.report.source_revision}:edge:{int(mesh_edge_source[index])}"
            for index in mapped
        ),
        np.zeros(mapped.shape, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(mapped.shape, 1, dtype=np.int8),
        source_indices=np.asarray(mesh_edge_source[mapped], dtype=np.int64),
    )

    face_regions = np.empty((source.report.num_faces,), dtype=object)
    face_regions[:] = None
    zones = []
    for control in specification.region_controls:
        source_faces = np.asarray(control.scope.entity_ids, dtype=np.int32)
        face_regions[source_faces] = control.region_name
        selected = np.isin(cell_face_ids, source_faces)
        if not np.any(selected):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Region {control.region_name!r} has no generated planar cells.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            cell_entity_set.entity_set_id,
            cell_ids[selected],
        )
        zones.append(
            MeshZone(
                control.region_name,
                MeshZoneRole.REGION,
                scope,
                material_id=control.material_id,
                region_role=control.role,
            )
        )
    if any(value is None for value in face_regions):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Planar cell ownership is not an exhaustive region partition.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    region_by_cell = face_regions[cell_face_ids]
    zone_by_name = {zone.name: zone for zone in zones}
    patches = []
    claimed: set[int] = set()
    for control in specification.patch_controls:
        source_edges = np.asarray(control.scope.entity_ids, dtype=np.int32)
        selected = np.flatnonzero(np.isin(mesh_edge_source, source_edges))
        if {int(value) for value in mesh_edge_source[selected]} != {
            int(value) for value in source_edges
        }:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated patch {control.name!r} does not cover its exact source scope.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        for mesh_edge in selected:
            adjacent = incidents[int(mesh_edge)]
            actual = tuple(sorted(str(region_by_cell[cell]) for cell in adjacent))
            if (
                len(adjacent) not in (1, 2)
                or len(set(actual)) != len(actual)
                or actual != control.adjacent_region_names
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Generated patch {control.name!r} has incorrect planar region adjacency.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                    entity_ids=(int(edge_ids[int(mesh_edge)]),),
                )
        if not selected.size:
            if control.required:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Required patch {control.name!r} is absent.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            continue
        claimed.update(int(value) for value in selected)
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            1,
            edge_entity_set.entity_set_id,
            edge_ids[selected],
        )
        patches.append(
            MeshPatch(
                control.name,
                scope,
                connected=_planar_edge_patch_connected(connectivity, selected),
                adjacent_zone_ids=tuple(
                    zone_by_name[name].zone_id for name in control.adjacent_region_names
                ),
            )
        )
    for mesh_edge, adjacent in enumerate(incidents):
        if len(adjacent) != 2:
            continue
        regions = {
            str(region_by_cell[adjacent[0]]),
            str(region_by_cell[adjacent[1]]),
        }
        if len(regions) == 2 and mesh_edge not in claimed:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated planar inter-region edge {mesh_edge} is undeclared.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                entity_ids=(int(edge_ids[mesh_edge]),),
            )

    face_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        2,
        cell_entity_set.entity_set_id,
        cell_ids,
    )
    edge_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        edge_entity_set.entity_set_id,
        edge_ids[mapped],
    )
    attributes = (
        MeshAttribute(
            "brep_face_index",
            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
            face_scope,
            cell_face_ids,
        ),
        MeshAttribute(
            "brep_edge_index",
            MeshAttributeRole.GEOMETRY_CLASSIFICATION,
            edge_scope,
            mesh_edge_source[mapped],
        ),
    )
    return _PlanarSurfaceEvidence(
        tuple(zones),
        tuple(patches),
        (face_association, edge_association),
        attributes,
        cell_face_ids,
        mesh_edge_source,
    )


def _canonical_cell_solid_ids(
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    cad_entities: _CadEntityMap,
    /,
) -> np.ndarray:
    volume_to_solid = {
        volume: solid for solid, volume in enumerate(cad_entities.solid_to_volume)
    }
    rows_by_name = {row.block_name: row for row in rows}
    owner_chunks = []
    for block in mesh.blocks:
        row = rows_by_name[block.name]
        entity_tags = row.entity_tags[row_orders[block.name]]
        owner_chunks.append(
            np.asarray(
                tuple(volume_to_solid.get(int(tag), -1) for tag in entity_tags),
                dtype=np.int32,
            )
        )
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    if not np.array_equal(cell_ids, np.asarray(mesh.entity_set(3).entity_ids)):
        raise MeshingFailure(
            MeshingFailureCategory.CANONICALIZATION_FAILED,
            "Canonical cell entity ordering does not match canonical mesh blocks.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    owners = np.concatenate(owner_chunks)
    if np.any(owners < 0):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "A generated top cell has unknown source-solid ownership.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return owners


@dataclass(frozen=True, slots=True)
class _SemanticSurfaceEvidence:
    boundary: SurfaceModel
    association: GeometryAssociation
    zones: tuple[MeshZone, ...]
    attribute: MeshAttribute
    mesh_face_source: np.ndarray


def _connectivity_face_rows(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity, /
) -> tuple[np.ndarray, ...]:
    if isinstance(connectivity, TetrahedralConnectivity):
        return tuple(np.asarray(connectivity.faces, dtype=np.int32))
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int32)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
    return tuple(values[int(start) : int(stop)] for start, stop in pairwise(offsets))


def _connectivity_face_incidents(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity, /
) -> tuple[tuple[int, ...], ...]:
    if isinstance(connectivity, PolyhedralConnectivity):
        owner = np.asarray(connectivity.face_owner, dtype=np.int32)
        neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32)
        return tuple(
            (int(first),) if int(second) < 0 else (int(first), int(second))
            for first, second in zip(owner, neighbor, strict=True)
        )
    incidents = [[] for _ in np.asarray(connectivity.faces)]
    cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
    for cell_index in range(cell_faces.shape[0]):
        face_row = np.take(cell_faces, cell_index, axis=0)
        for face_index in face_row:
            incidents[int(face_index)].append(cell_index)
    return tuple(tuple(row) for row in incidents)


def _connectivity_face_edge_rows(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity, /
) -> tuple[np.ndarray, ...]:
    if isinstance(connectivity, TetrahedralConnectivity):
        return tuple(np.asarray(connectivity.face_edges, dtype=np.int32))
    offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int32)
    values = np.asarray(connectivity.face_edge_values, dtype=np.int32)
    return tuple(values[int(start) : int(stop)] for start, stop in pairwise(offsets))


def _semantic_surface_evidence(
    gmsh: Any,
    source: BRepModel,
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    node_tags: np.ndarray,
    source_to_corner: np.ndarray,
    cell_solid_ids: np.ndarray,
    cad_entities: _CadEntityMap,
    plan_id: str,
    /,
) -> _SemanticSurfaceEvidence:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, (TetrahedralConnectivity, PolyhedralConnectivity)):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Semantic BRep volume meshing requires tetrahedral or mixed polyhedral connectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    faces = _connectivity_face_rows(connectivity)
    face_lookup = {
        tuple(sorted(int(value) for value in face)): index
        for index, face in enumerate(faces)
    }
    surface_to_face = {
        surface: face for face, surface in enumerate(cad_entities.face_to_surface)
    }
    face_source = np.full((len(faces),), -1, dtype=np.int32)
    for block in rows:
        if block.cell_kind not in ("triangle", "quadrilateral"):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Semantic CAD surfaces require triangle or swept-quad elements.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        local_nodes = _local_connectivity(
            node_tags, block.vertices[:, : block.corner_count]
        )
        corners = source_to_corner[local_nodes]
        if np.any(corners < 0):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "A CAD surface corner is absent from the canonical volume mesh.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        for values, surface_tag in zip(corners, block.entity_tags, strict=True):
            face_index = face_lookup.get(tuple(sorted(int(value) for value in values)))
            source_face = surface_to_face.get(int(surface_tag))
            if face_index is None or source_face is None or face_source[face_index] >= 0:
                raise MeshingFailure(
                    MeshingFailureCategory.CONVERSION_FAILED,
                    "A Gmsh CAD surface element does not map uniquely to one canonical mesh face.",
                    stage=MeshingStageKind.CANONICALIZATION.value,
                )
            face_source[face_index] = source_face

    incidents = _connectivity_face_incidents(connectivity)
    boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    mapped = np.flatnonzero(face_source >= 0)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    for face_index in mapped:
        source_face = int(face_source[face_index])
        expected = set(source.topology.face_solids[source_face])
        actual = {int(cell_solid_ids[cell]) for cell in incidents[face_index]}
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Canonical CAD face adjacency does not match source solid incidence.",
                stage=MeshingStageKind.CANONICALIZATION.value,
                entity_ids=(int(face_ids[face_index]),),
            )
    expected_exterior = face_source >= 0
    expected_exterior[mapped] = np.asarray(
        [
            len(source.topology.face_solids[int(face_source[index])]) == 1
            for index in mapped
        ]
    )
    if not np.array_equal(boundary_mask, expected_exterior):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Exterior mesh faces are not exactly the singly incident source CAD surfaces.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )

    face_entity_set = mesh.entity_set(2)
    exterior = np.flatnonzero(boundary_mask)
    boundary_triangles = []
    boundary_sources = []
    for face_index in exterior:
        face = faces[int(face_index)]
        for local in range(1, face.size - 1):
            boundary_triangles.append(
                (int(face[0]), int(face[local]), int(face[local + 1]))
            )
            boundary_sources.append(int(face_source[face_index]))
    boundary_triangles_array = np.asarray(boundary_triangles, dtype=np.int32)
    if all(faces[int(index)].size == 3 for index in exterior):
        boundary_order = np.arange(exterior.size, dtype=np.int64)
        boundary_ids = face_ids[exterior]
    else:
        boundary_keys = np.sort(boundary_triangles_array, axis=1)
        boundary_order = np.lexsort(
            tuple(
                boundary_keys[:, column]
                for column in range(boundary_keys.shape[1] - 1, -1, -1)
            )
        )
        boundary_ids = np.arange(boundary_order.size, dtype=np.int64)
    ordered_sources = np.asarray(boundary_sources, dtype=np.int32)[boundary_order]
    boundary_metadata = SurfaceMetadata(
        source_id=source.report.source_id,
        source_revision=source.report.source_revision,
        coordinate_contract=source.coordinate_contract,
        provenance=("gmsh-occ", plan_id),
        cell_tags=tuple(f"brep-face:{int(value)}" for value in ordered_sources),
    )
    boundary = SurfaceModel.from_triangles(
        mesh.coordinates,
        boundary_triangles_array[boundary_order],
        boundary_metadata,
        vertex_global_ids=mesh.vertex_global_ids,
        cell_global_ids=boundary_ids,
        numeric_version=mesh.numeric_version,
        repair_orientation=True,
        orient_closed_outward=True,
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source.report.source_id,
        source.report.source_revision,
        face_entity_set.entity_set_id,
        face_ids[mapped],
        tuple(
            f"{source.report.source_revision}:face:{int(face_source[index])}"
            for index in mapped
        ),
        np.zeros((mapped.size,), dtype=np.float64),
        exact=True,
        source_dimensions=np.full((mapped.size,), 2, dtype=np.int8),
        source_indices=np.asarray(face_source[mapped], dtype=np.int64),
    )
    zones = []
    for source_face in np.unique(face_source[exterior]):
        selected = exterior[face_source[exterior] == source_face]
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            face_entity_set.entity_set_id,
            face_ids[selected],
        )
        zones.append(
            MeshZone(f"brep-face-{int(source_face)}", MeshZoneRole.BOUNDARY, scope)
        )
    attribute_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        2,
        face_entity_set.entity_set_id,
        face_ids[mapped],
    )
    attribute = MeshAttribute(
        "brep_face_index",
        MeshAttributeRole.GEOMETRY_CLASSIFICATION,
        attribute_scope,
        face_source[mapped],
    )
    return _SemanticSurfaceEvidence(
        boundary,
        association,
        tuple(zones),
        attribute,
        face_source,
    )


def _patch_is_connected(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity,
    face_indices: np.ndarray,
    /,
) -> bool:
    if face_indices.size <= 1:
        return True
    face_edges = _connectivity_face_edge_rows(connectivity)
    edge_faces: dict[int, set[int]] = {}
    for face_index in face_indices:
        for edge_index in face_edges[int(face_index)]:
            edge_faces.setdefault(int(edge_index), set()).add(int(face_index))
    pending = [int(face_indices[0])]
    visited = set()
    while pending:
        face_index = pending.pop()
        if face_index in visited:
            continue
        visited.add(face_index)
        for edge_index in face_edges[face_index]:
            pending.extend(edge_faces[int(edge_index)] - visited)
    return len(visited) == face_indices.size


def _region_evidence(
    source: BRepModel,
    mesh: CellMesh,
    specification: VolumeMeshingSpec,
    cell_solid_ids: np.ndarray,
    mesh_face_source: np.ndarray,
    /,
) -> tuple[tuple[MeshZone, ...], tuple[MeshPatch, ...]]:
    cell_entity_set = mesh.entity_set(3)
    cell_ids = np.asarray(cell_entity_set.entity_ids, dtype=np.int64)
    solid_regions = np.empty((source.topology.num_solids,), dtype=object)
    solid_regions[:] = None
    zones = []
    for control in specification.region_controls:
        solid_ids = np.asarray(control.scope.entity_ids, dtype=np.int64)
        solid_regions[solid_ids] = control.region_name
        selected = np.isin(cell_solid_ids, solid_ids)
        if not np.any(selected):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Region {control.region_name!r} has no generated cells.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            3,
            cell_entity_set.entity_set_id,
            cell_ids[selected],
        )
        zones.append(
            MeshZone(
                control.region_name,
                MeshZoneRole.REGION,
                scope,
                material_id=control.material_id,
                region_role=control.role,
            )
        )
    if any(value is None for value in solid_regions):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Generated cell ownership does not resolve to an exhaustive region partition.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    region_by_cell = solid_regions[cell_solid_ids]
    zone_by_name = {zone.name: zone for zone in zones}
    connectivity = mesh.connectivity
    if not isinstance(connectivity, (TetrahedralConnectivity, PolyhedralConnectivity)):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Semantic region patches require tetrahedral or mixed polyhedral connectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    face_source = np.asarray(mesh_face_source, dtype=np.int32)
    face_rows = _connectivity_face_rows(connectivity)
    if face_source.shape != (len(face_rows),):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "CAD face evidence does not align with canonical mesh faces.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    incidents = _connectivity_face_incidents(connectivity)
    face_entity_set = mesh.entity_set(2)
    face_ids = np.asarray(face_entity_set.entity_ids, dtype=np.int64)
    claimed: set[int] = set()
    patches = []
    for control in specification.patch_controls:
        selected = np.flatnonzero(
            np.isin(
                face_source,
                np.asarray(control.scope.entity_ids, dtype=np.int32),
            )
        )
        requested_source_faces = {
            int(value) for value in np.asarray(control.scope.entity_ids)
        }
        mapped_source_faces = {int(value) for value in face_source[selected]}
        if selected.size and mapped_source_faces != requested_source_faces:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated patch {control.name!r} does not cover its exact source scope.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        valid = []
        for face_index in selected:
            adjacent = incidents[int(face_index)]
            actual = tuple(sorted(str(region_by_cell[cell]) for cell in adjacent))
            if (
                len(adjacent) not in (1, 2)
                or len(set(actual)) != len(actual)
                or actual != control.adjacent_region_names
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Generated patch {control.name!r} has incorrect region adjacency.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                    entity_ids=(int(face_ids[int(face_index)]),),
                )
            valid.append(int(face_index))
        face_indices = np.asarray(valid, dtype=np.int64)
        if not face_indices.size:
            if control.required:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Required patch {control.name!r} is absent.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            continue
        claimed.update(valid)
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            face_entity_set.entity_set_id,
            face_ids[face_indices],
        )
        patches.append(
            MeshPatch(
                control.name,
                scope,
                connected=_patch_is_connected(connectivity, face_indices),
                adjacent_zone_ids=tuple(
                    zone_by_name[name].zone_id for name in control.adjacent_region_names
                ),
            )
        )
    for face_index, adjacent in enumerate(incidents):
        if len(adjacent) != 2:
            continue
        regions = {
            str(region_by_cell[adjacent[0]]),
            str(region_by_cell[adjacent[1]]),
        }
        if len(regions) == 2 and face_index not in claimed:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated inter-region mesh face {face_index} is not declared by a PatchControl.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                entity_ids=(int(face_ids[face_index]),),
            )
    return tuple(zones), tuple(patches)
