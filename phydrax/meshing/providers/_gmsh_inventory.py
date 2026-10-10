#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact authored occurrence inventory at the optional Gmsh comparison boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...geometry.brep._model import (
    BRepEntityId,
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
    BRepQualifiedIncidence,
    BRepTopology,
)
from .._contracts import MeshingFailure, MeshingFailureCategory
from .._trace import MeshingStageKind


_KINDS = ("vertex", "edge", "face", "solid")


def _exact_geometry(source: BRepModel, /) -> BRepGeometry:
    geometry = source.geometry
    if geometry is None:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "External comparison requires authoritative native CAD carriers and incidence.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    return geometry


def _cad_scope_set(source: BRepModel, dimension: int, /) -> str:
    return f"{source.model_id}:cad-occurrences:{dimension}"


def _belongs_to_solid(
    entity: BRepEntityId, solid: int, geometry: BRepGeometry, topology: BRepTopology, /
) -> bool:
    faces = topology.solid_faces[solid]
    match entity.kind:
        case "solid":
            return entity.index == solid
        case "face":
            return entity.index in faces
        case "edge":
            return any(entity.index in topology.face_edges[face] for face in faces)
        case "vertex":
            return any(
                entity.index in geometry.edge_vertices[edge]
                for face in faces
                for edge in topology.face_edges[face]
            )
        case _:
            raise ValueError(
                "Comparison inventories admit vertex, edge, face and solid IDs."
            )


def _placement(
    entity: BRepEntityId, geometry: BRepGeometry, topology: BRepTopology, /
) -> BRepOccurrence | None:
    if not entity.occurrence_path:
        return None
    for occurrence in geometry.occurrences:
        if occurrence.path == entity.occurrence_path:
            if not _belongs_to_solid(entity, occurrence.solid, geometry, topology):
                raise ValueError(
                    "An authored entity is absent from its declared occurrence."
                )
            return occurrence
    for container in geometry.assembly_containers:
        if container.path == entity.occurrence_path:
            owners = tuple(
                occurrence
                for occurrence in geometry.occurrences
                if occurrence.path in container.member_paths
                and _belongs_to_solid(entity, occurrence.solid, geometry, topology)
            )
            if not owners:
                raise ValueError("A shared stratum has no authored container member.")
            first = owners[0]
            if any(
                first.rotation != other.rotation or first.translation != other.translation
                for other in owners[1:]
            ):
                raise ValueError(
                    "A shared stratum has contradictory authored placements."
                )
            return first
    raise ValueError("An entity occurrence path has no declared placement or container.")


@dataclass(frozen=True, slots=True)
class _CadOccurrenceInventory:
    """Numeric scope rows backed by exact IDs, frames and oriented authored links."""

    entities: tuple[tuple[BRepEntityId, ...], ...]
    placements: tuple[tuple[BRepOccurrence | None, ...], ...]
    topology: BRepTopology
    edge_vertices: tuple[tuple[int, int], ...]
    incidence: tuple[BRepQualifiedIncidence, ...]

    def select(
        self, selected: tuple[BRepEntityId, ...], dimension: int, /
    ) -> tuple[int, ...]:
        rows = self.entities[dimension]
        result: set[int] = set()
        for entity in selected:
            matches = tuple(
                row
                for row, candidate in enumerate(rows)
                if candidate.source_revision == entity.source_revision
                and candidate.kind == entity.kind
                and candidate.index == entity.index
                and (
                    not entity.occurrence_path
                    or candidate.occurrence_path == entity.occurrence_path
                )
            )
            if not matches:
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "The selector has no authored native definition or occurrence member.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
            result.update(matches)
        return tuple(sorted(result))

    def world_points(self, dimension: int, row: int, points: np.ndarray, /) -> np.ndarray:
        placement = self.placements[dimension][row]
        return points if placement is None else placement.place(points)


def _face_wires(
    faces: tuple[BRepEntityId, ...],
    edges: dict[BRepEntityId, int],
    geometry: BRepGeometry,
    incidence: tuple[BRepQualifiedIncidence, ...],
    /,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    coedges = {
        (record.container, record.use_index): record
        for record in incidence
        if record.container.kind == "face" and record.member.kind == "edge"
    }
    result = []
    for face in faces:
        wires = []
        for loop in geometry.face_loops[face.index]:
            wire = []
            for use in loop:
                record = coedges.get((face, use))
                if record is None:
                    raise ValueError(
                        "An authored face loop lost its qualified coedge use."
                    )
                wire.append(record.orientation * (edges[record.member] + 1))
            wires.append(tuple(wire))
        result.append(tuple(wires))
    return tuple(result)


def _solid_faces(
    solids: tuple[BRepEntityId, ...],
    faces: dict[BRepEntityId, int],
    topology: BRepTopology,
    incidence: tuple[BRepQualifiedIncidence, ...],
    /,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    result, signs = [], []
    for solid in solids:
        members = {
            record.member.index: record
            for record in incidence
            if record.container == solid and record.member.kind == "face"
        }
        native_faces = topology.solid_faces[solid.index]
        if set(members) != set(native_faces):
            raise ValueError(
                "A qualified solid lost its exact definition face incidence."
            )
        result.append(tuple(faces[members[face].member] for face in native_faces))
        signs.append(tuple(members[face].orientation for face in native_faces))
    return tuple(result), tuple(signs)


def _cad_occurrence_inventory(source: BRepModel, /) -> _CadOccurrenceInventory:
    geometry = _exact_geometry(source)
    incidence = geometry.qualified_entity_incidence(source.source_revision)
    authored = {record.container for record in incidence} | {
        record.member for record in incidence
    }
    entities = tuple(
        tuple(sorted(entity for entity in authored if entity.kind == kind))
        for kind in _KINDS
    )
    vertices, edges, faces, solids = entities
    edge_rows = {entity: row for row, entity in enumerate(edges)}
    vertex_rows = {entity: row for row, entity in enumerate(vertices)}
    face_rows = {entity: row for row, entity in enumerate(faces)}
    wires = _face_wires(faces, edge_rows, geometry, incidence)
    face_edges = tuple(
        tuple(dict.fromkeys(abs(edge) - 1 for wire in loops for edge in wire))
        for loops in wires
    )
    edge_faces: list[list[int]] = [[] for _ in edges]
    for face, members in enumerate(face_edges):
        for edge in members:
            edge_faces[edge].append(face)
    endpoints = {
        (record.container, record.use_index): record.member
        for record in incidence
        if record.container.kind == "edge" and record.member.kind == "vertex"
    }
    edge_vertices = tuple(
        (vertex_rows[endpoints[(edge, 0)]], vertex_rows[endpoints[(edge, 1)]])
        for edge in edges
    )
    solid_faces, solid_signs = _solid_faces(solids, face_rows, source.topology, incidence)
    topology = BRepTopology(
        face_edges=face_edges,
        edge_faces=tuple(tuple(rows) for rows in edge_faces),
        face_wires=wires,
        solid_faces=solid_faces,
        solid_face_orientations=solid_signs,
        num_vertices=len(vertices),
    )
    placements = tuple(
        tuple(_placement(entity, geometry, source.topology) for entity in rows)
        for rows in entities
    )
    return _CadOccurrenceInventory(
        entities, placements, topology, edge_vertices, incidence
    )
