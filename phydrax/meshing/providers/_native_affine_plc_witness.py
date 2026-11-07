#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source-chart restrictions for global-affine PLC boundaries."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._affine_plc_boundary import AffinePlcBoundarySource
from ...geometry._planar_primitives import (
    area2,
    cross,
    intersection,
    plane_key,
    Point,
    Polygon,
    project_polygon,
    projection_axes,
)
from ...geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
)
from .._volume_generation import VolumeConstruction
from ._native_plc_geometry import simplex_volume_entities, volume_cell_entity_rows
from ._native_sources import NativePlcSource


def _parameters(point: Point, triangle: Polygon, /) -> Point:
    determinant = cross(triangle[0], triangle[1], triangle[2])
    if not determinant:
        raise ValueError("An affine PLC boundary chart is degenerate.")
    return (
        cross(triangle[0], point, triangle[2]) / determinant,
        cross(triangle[0], triangle[1], point) / determinant,
    )


def affine_plc_boundary_chart_witness(
    source: NativePlcSource,
    query: AffinePlcBoundarySource,
    construction: VolumeConstruction,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> SourceBoundaryChartWitness:
    """Restrict exact source triangles to the physical target boundary.

    ``mesh`` has the construction's vertex and cell row ancestry, but carries
    the physical image. ``geometry`` is its exact physical coordinate map;
    rounded mesh coordinates are never used to infer source-point ancestry.
    """
    if query.owner is not source or (
        construction.source_id,
        construction.source_revision,
    ) != (source.source_id, source.source_revision):
        raise ValueError(
            "Affine PLC boundary witness requires its original source authority."
        )
    query.require_current()
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind not in {
        "simplex:3",
        "tetrahedron",
    }:
        raise ValueError(
            "Affine PLC boundary witnesses require one tetrahedral target block."
        )
    block = mesh.blocks[0]
    if len(construction.mesh.blocks) != 1:
        raise ValueError("The construction must have one tetrahedral target block.")
    original = construction.mesh.blocks[0]
    if (
        not np.array_equal(block.global_ids, original.global_ids)
        or not np.array_equal(
            np.sort(block.vertices, axis=1), np.sort(original.vertices, axis=1)
        )
        or mesh.coordinates.shape != construction.mesh.coordinates.shape
    ):
        raise ValueError(
            "The physical target must preserve construction cell and vertex ancestry."
        )
    topology = reference_cell_topology(block.cell_kind)
    reference = tuple(
        tuple(Fraction.from_float(float(x)) for x in point) for point in topology.vertices
    )
    cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    exact_corners = geometry.evaluate_exact(mesh, cell_ids, (reference,) * len(cell_ids))
    _, faces = simplex_volume_entities(mesh)
    face_rows = {frozenset(map(int, face)): row for row, face in enumerate(faces)}
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    _, cell_faces = volume_cell_entity_rows(mesh, cell_ids)
    owners = {}
    for cell_row, rows in enumerate(cell_faces):
        for local_face, face_row in enumerate(rows):
            owners.setdefault(int(face_row), []).append((cell_row, local_face))

    # The supporting plane and original facet identity jointly determine the
    # candidate bank; triangulation diagonals are deliberately not assumed to
    # survive recovery.
    charts = {}
    for row, chart_id in enumerate(query.face_chart_ids):
        triangle = tuple(
            query.domain.exact_vertices[int(v)] for v in query.domain.facets[row]
        )
        key = plane_key(triangle)
        source_key = query.chart_source_key(2, chart_id)
        charts.setdefault((key, source_key), []).append((chart_id, triangle))

    cell_out, face_out, local_out = [], [], []
    target_out, parameter_out = [], []
    chart_out, entity_out, occurrence_out, orientation_out = [], [], [], []
    if not (
        len(construction.boundary_faces)
        == len(construction.face_source_ids)
        == len(construction.face_source_occurrence_paths)
    ):
        raise ValueError("Recovered boundary face identities must align with its faces.")
    for face, entity, occurrence in zip(
        construction.boundary_faces,
        construction.face_source_ids,
        construction.face_source_occurrence_paths,
        strict=True,
    ):
        face_row = face_rows.get(frozenset(map(int, face)))
        if face_row is None:
            raise ValueError(
                "A recovered boundary face is absent from the physical target."
            )
        face_owners = owners.get(face_row, ())
        if not 1 <= len(face_owners) <= 2:
            raise ValueError("A recovered face must have one or two target owners.")
        # An internal interface has two incident cells, but certification binds
        # all pieces of a face to one consistent side.
        cell_row, local_face = face_owners[0]
        local_vertices = tuple(int(v) for v in topology.entities[2][local_face])
        target_ref = tuple(reference[v] for v in local_vertices)
        target_xyz = tuple(exact_corners[cell_row][v] for v in local_vertices)
        plane = plane_key(target_xyz)
        axes = projection_axes(plane)
        target_uv = project_polygon(target_xyz, axes)
        target_area = area2(target_uv)
        if not target_area:
            raise ValueError("A physical target boundary face is exactly degenerate.")
        candidates = charts.get((plane, (entity, occurrence)), ())
        covered = Fraction(0)
        for chart_id, source_xyz in candidates:
            source_uv = project_polygon(source_xyz, axes)
            polygon = intersection(target_uv, source_uv)
            covered += area2(polygon)
            for index in range(1, len(polygon) - 1):
                piece = (polygon[0], polygon[index], polygon[index + 1])
                if not cross(*piece):
                    continue
                target_parameters = tuple(
                    _parameters(point, target_uv) for point in piece
                )
                source_parameters = tuple(
                    _parameters(point, source_uv) for point in piece
                )
                target_piece = tuple(
                    tuple(
                        target_ref[0][axis]
                        + u * (target_ref[1][axis] - target_ref[0][axis])
                        + v * (target_ref[2][axis] - target_ref[0][axis])
                        for axis in range(3)
                    )
                    for u, v in target_parameters
                )
                target_sign = cross(*target_parameters)
                source_sign = cross(*source_parameters)
                cell_out.append(int(cell_ids[cell_row]))
                face_out.append(int(face_ids[face_row]))
                local_out.append(local_face)
                target_out.append(target_piece)
                parameter_out.append(source_parameters)
                chart_out.append(chart_id)
                entity_out.append(entity)
                occurrence_out.append(occurrence)
                orientation_out.append(1 if target_sign * source_sign > 0 else -1)
        if covered != target_area:
            raise ValueError(
                "Exact source charts do not partition a physical boundary face."
            )
    if not cell_out:
        raise ValueError("Affine PLC boundary query produced no target charts.")
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(cell_out, dtype=np.int64),
        entity_global_ids=np.asarray(face_out, dtype=np.int64),
        local_entity_indices=np.asarray(local_out, dtype=np.int32),
        target_reference_vertices=np.asarray(target_out, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(target_out),
        source_chart_ids=tuple(chart_out),
        source_entity_ids=tuple(entity_out),
        occurrence_paths=tuple(occurrence_out),
        source_parameters=np.asarray(parameter_out, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(parameter_out),
        orientations=np.asarray(orientation_out, dtype=np.int8),
    )
    query.validate_chart_witness(witness)
    return witness


__all__ = ["affine_plc_boundary_chart_witness"]
