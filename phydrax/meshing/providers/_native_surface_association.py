# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Map native surface construction entities to authoritative source strata."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.meshing._association as association_owner

from ...typing import checked
from .._surface_construction import SurfaceConstruction


def simplex_rows(canonical: np.ndarray, pairs: np.ndarray, /) -> np.ndarray:
    """Find canonical simplex rows by vertex incidence, independent of order."""
    first = np.asarray(canonical)
    second = np.asarray(pairs)
    if (
        first.ndim != 2
        or first.shape[1] == 0
        or second.ndim != 2
        or second.shape[1] != first.shape[1]
        or first.dtype.kind not in "iu"
        or second.dtype.kind not in "iu"
    ):
        raise ValueError(
            "NativeSurfaceAssociation.simplex_rows must receive aligned integer vertex-incidence tables; got incompatible arity."
        )
    keys = np.sort(first.astype(np.int64, copy=False), axis=1)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError(
            "NativeSurfaceAssociation.simplex_rows canonical incidence must be unique; got duplicate simplex."
        )
    lookup = {tuple(pair): row for row, pair in enumerate(keys.tolist())}
    rows: list[int] = []
    for key in map(tuple, np.sort(second.astype(np.int64, copy=False), axis=1).tolist()):
        if key not in lookup:
            raise ValueError(
                "NativeSurfaceAssociation.simplex_rows must match every construction simplex to mesh incidence; got absent simplex."
            )
        rows.append(lookup[key])
    return np.asarray(rows, dtype=np.int64)


def _source_metadata(
    domain: domain_owner.MeshingDomain,
    dimensions: np.ndarray,
    indices: np.ndarray,
    /,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[tuple[str, ...], ...],
    tuple[association_owner.GeometrySourceEntityRole, ...],
]:
    """Resolve source indices, occurrence paths, and explicit entity roles."""
    dimension_values = np.asarray(dimensions, dtype=np.int8)
    index_values = np.asarray(indices, dtype=np.int64)
    if dimension_values.shape != index_values.shape:
        raise ValueError(
            "NativeSurfaceAssociation.source_metadata must align source dimensions and indices; got mismatched rows."
        )
    if domain.source_kinds not in (
        ("corner", "curve", "surface"),
        ("vertex", "edge", "face"),
        ("reconstructed_vertex", "reconstructed_edge", "reconstructed_triangle"),
    ):
        raise ValueError(
            "NativeSurfaceAssociation.domain must retain canonical source strata; got unsupported source kind table."
        )
    source_indices = np.asarray(
        [
            domain.source_indices[dimension][index]
            for dimension, index in zip(
                dimension_values.tolist(), index_values.tolist(), strict=True
            )
        ],
        dtype=np.int64,
    )
    entities = domain.entity_ids(dimension_values, index_values)
    occurrences = tuple(
        domain.source_occurrences[dimension][index]
        for dimension, index in zip(
            dimension_values.tolist(), index_values.tolist(), strict=True
        )
    )
    role_by_dimension = (
        association_owner.GeometrySourceEntityRole.VERTEX,
        association_owner.GeometrySourceEntityRole.EDGE,
        association_owner.GeometrySourceEntityRole.FACET,
    )
    roles = tuple(role_by_dimension[dimension] for dimension in dimension_values.tolist())
    return entities, dimension_values, source_indices, occurrences, roles


def _surface_edges(mesh: cell_mesh_owner.CellMesh) -> np.ndarray:
    """Return canonical vertex-row pairs for every declared surface edge."""
    match mesh.connectivity:
        case cell_mesh_owner.SimplicialConnectivity() as connectivity:
            if connectivity.dimension != 2:
                raise ValueError(
                    "NativeSurfaceAssociation.mesh must have triangular surface incidence; got different simplex dimension."
                )
            return np.asarray(connectivity.entities[1], dtype=np.int64)
        case cell_complex_owner.PolygonalConnectivity() as connectivity:
            return np.asarray(connectivity.edges, dtype=np.int64)
        case _:
            raise ValueError(
                "NativeSurfaceAssociation.mesh must expose canonical surface-edge incidence; got unsupported connectivity."
            )


@checked
def _validate_shared_junctions(
    curves: tuple[domain_owner.MeshingDomainCurve, ...],
    construction: SurfaceConstruction,
    /,
) -> None:
    """Bind each shared source-curve chain to its authored corner identities."""
    corner_rows: dict[int, int] = {}
    vertex_corners: dict[int, int] = {}
    vertex_curves: dict[int, int] = {}
    dimensions = np.asarray(construction.vertex_source_dimensions, dtype=np.int8)
    indices = np.asarray(construction.vertex_source_indices, dtype=np.int64)
    if dimensions.ndim != 1 or indices.shape != dimensions.shape:
        raise ValueError(
            "NativeSurfaceAssociation.vertices must retain aligned source strata; got malformed vertex ancestry."
        )
    for curve, vertices in construction.curve_vertex_chains():
        if curve >= len(curves):
            raise ValueError(
                "NativeSurfaceAssociation.curve_parameters must name a declared source curve; got foreign curve ID."
            )
        source_curve = curves[curve]
        for position, vertex_value in enumerate(vertices.tolist()):
            vertex = int(vertex_value)
            if not 0 <= vertex < dimensions.size:
                raise ValueError(
                    "NativeSurfaceAssociation.curve vertices must index the retained source vertex bank."
                )
            if position == 0:
                corner = source_curve.start
            elif position == vertices.size - 1:
                corner = source_curve.end
            else:
                if dimensions[vertex] != 1 or indices[vertex] != curve:
                    raise ValueError(
                        "NativeSurfaceAssociation.curve interiors must retain their canonical source-curve identity; got a foreign or welded vertex."
                    )
                if vertex in vertex_curves and vertex_curves[vertex] != curve:
                    raise ValueError(
                        "NativeSurfaceAssociation.distinct source curves must not share interior vertex IDs."
                    )
                vertex_curves[vertex] = curve
                continue
            if dimensions[vertex] != 0 or indices[vertex] != corner:
                raise ValueError(
                    "NativeSurfaceAssociation.curve endpoints must retain their authored source-corner identities; got mismatched junction."
                )
            if corner in corner_rows and corner_rows[corner] != vertex:
                raise ValueError(
                    "NativeSurfaceAssociation.shared curve uses must reuse one source-junction vertex ID; got duplicated junction."
                )
            if vertex in vertex_corners and vertex_corners[vertex] != corner:
                raise ValueError(
                    "NativeSurfaceAssociation.source corners must retain distinct topological vertex IDs; got coordinate-based junction welding."
                )
            corner_rows[corner] = vertex
            vertex_corners[vertex] = corner
        if source_curve.start == source_curve.end and vertices[0] != vertices[-1]:
            raise ValueError(
                "NativeSurfaceAssociation.closed source curves must close on their authored junction ID; got open chain."
            )


def _face_orientations(
    points: np.ndarray, faces: np.ndarray, normals: np.ndarray, /
) -> np.ndarray:
    """Relate each oriented triangle to its native source-surface normal."""
    coordinates = np.asarray(points, dtype=np.float64)
    cells = np.asarray(faces)
    directions = np.asarray(normals, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or cells.ndim != 2
        or cells.shape[1] != 3
        or cells.dtype.kind not in "iu"
        or directions.shape != (cells.shape[0], 3)
    ):
        raise ValueError(
            "NativeSurfaceAssociation.face_orientation inputs must align triangle and source-normal rows."
        )
    flat = np.cross(
        coordinates[cells[:, 1]] - coordinates[cells[:, 0]],
        coordinates[cells[:, 2]] - coordinates[cells[:, 0]],
    )
    return np.sign(np.sum(flat * directions, axis=1)).astype(np.int8)


def _curve_orientations(
    points: np.ndarray,
    canonical_edges: np.ndarray,
    curve_edges: np.ndarray,
    source_senses: np.ndarray,
    /,
) -> np.ndarray:
    """Relate canonical mesh edges to canonical source-curve direction."""
    coordinates = np.asarray(points, dtype=np.float64)
    canonical = np.asarray(canonical_edges)
    curves = np.asarray(curve_edges)
    senses = np.asarray(source_senses, dtype=np.int8)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or canonical.ndim != 2
        or canonical.shape[1] != 2
        or canonical.dtype.kind not in "iu"
        or curves.shape != canonical.shape
        or curves.dtype.kind not in "iu"
        or senses.shape != (canonical.shape[0],)
        or np.any(~np.isin(senses, (-1, 1)))
    ):
        raise ValueError(
            "NativeSurfaceAssociation.curve_orientation inputs must align canonical mesh and source-curve edges."
        )
    mesh_directions = coordinates[canonical[:, 1]] - coordinates[canonical[:, 0]]
    curve_directions = coordinates[curves[:, 1]] - coordinates[curves[:, 0]]
    return (
        np.sign(np.sum(mesh_directions * curve_directions, axis=1)).astype(np.int8)
        * senses
    )


@checked
def associations(
    domain: domain_owner.MeshingDomain,
    mesh: cell_mesh_owner.CellMesh,
    construction: SurfaceConstruction,
    /,
) -> tuple[association_owner.GeometryAssociation, ...]:
    """Associate faces, authored physical curves, and vertices with source strata.

    Curve edges are exactly the canonical source-curve chains emitted once by
    construction. Local extraction and T-junction partitions contribute no
    physical edge associations.
    """
    if not domain.source_id or not domain.source_revision:
        raise ValueError(
            "NativeSurfaceAssociation.domain must retain source identity and revision."
        )
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        raise ValueError(
            "NativeSurfaceAssociation.mesh must be a surface in three-space; got incompatible dimensions."
        )
    if not mesh.blocks or any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError(
            "NativeSurfaceAssociation.mesh must contain triangular cells; got an empty or non-triangle block."
        )
    kind = (
        association_owner.GeometryAssociationKind.BREP
        if domain.source_kinds == ("vertex", "edge", "face")
        else association_owner.GeometryAssociationKind.SURFACE
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    faces = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks),
        axis=0,
    )
    face_set = mesh.entity_set(2)
    if faces.shape != (face_set.count, 3):
        raise ValueError(
            "NativeSurfaceAssociation.mesh faces must align with the top-dimensional entity set; got mismatched rows."
        )
    triangles = np.asarray(construction.triangles)
    triangle_patches = np.asarray(construction.triangle_patches)
    triangle_charts = np.asarray(construction.triangle_charts, dtype=np.float64)
    triangle_deviations = np.asarray(construction.triangle_deviations, dtype=np.float64)
    source_cells = np.asarray(construction.triangle_source_cells)
    source_pieces = construction.triangle_source_pieces
    source_parameters = np.asarray(construction.triangle_parameters, dtype=np.float64)
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or triangles.dtype.kind not in "iu"
        or triangle_patches.shape != (triangles.shape[0],)
        or triangle_patches.dtype.kind not in "iu"
        or np.any(triangle_patches < 0)
        or triangle_charts.shape != (triangles.shape[0], 2)
        or not np.all(np.isfinite(triangle_charts))
        or triangle_deviations.shape != (triangles.shape[0],)
        or source_cells.shape != (triangles.shape[0],)
        or source_cells.dtype.kind not in "iu"
        or np.any(source_cells < 0)
        or len(source_pieces) != triangles.shape[0]
        or any(
            not piece
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in piece
            )
            for piece in source_pieces
        )
        or source_parameters.shape != (triangles.shape[0], 3, 2)
        or not np.all(np.isfinite(source_parameters))
    ):
        raise ValueError(
            "NativeSurfaceAssociation.triangles must retain aligned source-cell and original UV ancestry; got incomplete construction data."
        )
    _validate_shared_junctions(domain.curves, construction)
    rows = simplex_rows(triangles, faces)
    patches = triangle_patches.astype(np.int64, copy=False)[rows]
    charts = triangle_charts[rows]
    normals, _ = domain.oriented_normals(patches, charts)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    face_sources, face_dimensions, face_indices, face_occurrences, face_roles = (
        _source_metadata(
            domain,
            np.full(patches.shape, 2, dtype=np.int8),
            patches,
        )
    )
    face_association = association_owner.GeometryAssociation(
        kind,
        domain.source_id,
        domain.source_revision,
        face_set.entity_set_id,
        face_ids,
        face_sources,
        np.asarray(construction.triangle_deviations, dtype=np.float64)[rows],
        exact=False,
        source_dimensions=face_dimensions,
        source_roles=face_roles,
        source_indices=face_indices,
        source_occurrence_paths=face_occurrences,
        source_parameter_intervals=(None,) * face_ids.size,
        parameters=charts,
        orientations=_face_orientations(points, faces, normals),
    )
    associations = [face_association]

    if construction.curve_edges.shape[0]:
        edge_set = mesh.entity_set(1)
        canonical_edges = _surface_edges(mesh)
        if canonical_edges.shape != (edge_set.count, 2):
            raise ValueError(
                "NativeSurfaceAssociation.mesh edge incidence must align with its entity set; got mismatched rows."
            )
        edge_rows = simplex_rows(canonical_edges, construction.curve_edges)
        edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[edge_rows]
        if np.unique(edge_ids).size != edge_ids.size:
            raise ValueError(
                "NativeSurfaceAssociation.curve_edges must retain distinct topological source strata; got shared mesh-edge identity for separate curves."
            )
        order = np.argsort(edge_ids, kind="stable")
        edge_ids = edge_ids[order]
        curves = np.asarray(construction.curve_edge_curves, dtype=np.int64)[order]
        along = np.asarray(construction.curve_edges, dtype=np.int64)[order]
        canonical = canonical_edges[edge_rows[order]]
        source_senses = np.ones(curves.shape, dtype=np.int8)
        if kind is association_owner.GeometryAssociationKind.BREP:
            for row, curve in enumerate(curves.tolist()):
                patch, loop, position = map(int, domain.curve_owners[curve])
                use = domain.patches[patch].loops[loop][position]
                if use.last == use.first:
                    raise ValueError(
                        "NativeSurfaceAssociation.curve must retain a nonzero authored orientation; got collapsed coedge interval."
                    )
                source_senses[row] = 1 if use.last > use.first else -1
        (
            edge_sources,
            edge_dimensions,
            edge_indices,
            edge_occurrences,
            edge_roles,
        ) = _source_metadata(
            domain,
            np.full(curves.shape, 1, dtype=np.int8),
            curves,
        )
        edge_parameter_intervals = construction.curve_edge_parameter_intervals()
        if len(edge_parameter_intervals) != edge_ids.size:
            raise ValueError(
                "NativeSurfaceAssociation.curve_parameters must align exact physical-edge intervals; got mismatched rows."
            )
        associations.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.CURVE,
                domain.source_id,
                domain.source_revision,
                edge_set.entity_set_id,
                edge_ids,
                edge_sources,
                np.asarray(construction.curve_edge_deviations, dtype=np.float64)[order],
                exact=False,
                source_dimensions=edge_dimensions,
                source_roles=edge_roles,
                source_indices=edge_indices,
                source_occurrence_paths=edge_occurrences,
                source_parameter_intervals=tuple(
                    edge_parameter_intervals[int(row)] for row in order
                ),
                orientations=_curve_orientations(points, canonical, along, source_senses),
            )
        )

    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    (
        vertex_sources,
        vertex_dimensions,
        vertex_indices,
        vertex_occurrences,
        vertex_roles,
    ) = _source_metadata(
        domain,
        np.asarray(construction.vertex_source_dimensions, dtype=np.int8),
        np.asarray(construction.vertex_source_indices, dtype=np.int64),
    )
    if vertex_ids.size != len(vertex_sources):
        raise ValueError(
            "NativeSurfaceAssociation.vertices must align source classification with mesh vertex IDs; got mismatched rows."
        )
    parameters = np.array(construction.vertex_parameters, dtype=np.float64, copy=True)
    curve_rows = np.asarray(construction.vertex_source_indices, dtype=np.int64)
    exact_by_vertex: dict[tuple[int, int], Fraction] = {}
    for edge, curve, interval in zip(
        np.asarray(construction.curve_edges, dtype=np.int64).tolist(),
        np.asarray(construction.curve_edge_curves, dtype=np.int64),
        construction.curve_edge_parameter_intervals(),
        strict=True,
    ):
        for vertex, parameter in zip(edge, interval, strict=True):
            if int(vertex_dimensions[int(vertex)]) != 1:
                continue
            key = (int(vertex), int(curve))
            previous = exact_by_vertex.setdefault(key, parameter)
            if previous != parameter:
                raise ValueError("Curve vertex has conflicting exact source parameters.")
    for row in np.flatnonzero(vertex_dimensions == 1):
        curve = int(curve_rows[row])
        parameter = exact_by_vertex.get((int(row), curve))
        if parameter is None:
            raise ValueError("Curve vertex lacks its exact source parameter.")
        parameters[row, 0] = float(
            domain.curve_native_parameter(curve, parameter)
            if kind is association_owner.GeometryAssociationKind.BREP
            else parameter
        )
    associations.append(
        association_owner.GeometryAssociation(
            kind,
            domain.source_id,
            domain.source_revision,
            vertex_set.entity_set_id,
            vertex_ids,
            vertex_sources,
            np.zeros((vertex_ids.size,), dtype=np.float64),
            # Numeric construction parameters do not prove exact source equality.
            exact=False,
            source_dimensions=vertex_dimensions,
            source_roles=vertex_roles,
            source_indices=vertex_indices,
            source_occurrence_paths=vertex_occurrences,
            source_parameter_intervals=(None,) * vertex_ids.size,
            parameters=parameters,
        )
    )
    return tuple(associations)


__all__ = ["associations", "simplex_rows"]
