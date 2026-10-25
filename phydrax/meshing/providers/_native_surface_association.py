#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-stratum associations for generated surface topology."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

import phydrax.discretization as cell_mesh_owner
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...geometry._meshing_domain import MeshingDomain
from ...typing import checked
from .._organization_records import MeshLabel, MeshPatch
from .._surface_construction import SurfaceConstruction
from . import _native_sources as sources_owner
from ._native_route_metadata import mesh_entity_scope


def simplex_rows(canonical: np.ndarray, pairs: np.ndarray, /) -> np.ndarray:
    """Return simplex rows matching the vertex pairs in any vertex order."""

    keys = {key: row for row, key in enumerate(map(tuple, np.sort(canonical, axis=1)))}
    return np.asarray(
        [keys[key] for key in map(tuple, np.sort(pairs, axis=1).tolist())],
        dtype=np.int64,
    )


class _SourceMetadata(TypedDict):
    source_dimensions: np.ndarray
    source_indices: np.ndarray
    source_occurrence_paths: tuple[tuple[str, ...], ...]


def _source_metadata(
    domain: MeshingDomain, dimensions: np.ndarray, indices: np.ndarray, /
) -> _SourceMetadata:
    """Retain authored strata for both analytic and native B-Rep publications."""
    return {
        "source_dimensions": dimensions,
        "source_indices": np.asarray(
            [
                domain.source_indices[int(dimension)][int(index)]
                for dimension, index in zip(dimensions, indices, strict=True)
            ],
            dtype=np.int64,
        ),
        "source_occurrence_paths": tuple(
            domain.source_occurrences[int(dimension)][int(index)]
            for dimension, index in zip(dimensions, indices, strict=True)
        ),
    }


@checked
def associations(
    source: sources_owner.NativeSurfaceSource,
    mesh: cell_mesh_owner.CellMesh,
    construction: SurfaceConstruction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[GeometryAssociation, ...], tuple[MeshPatch, ...], tuple[MeshLabel, ...]]:
    domain = source.domain
    revision = source.source_revision
    kind = (
        GeometryAssociationKind.BREP
        if domain.source_kinds == ("vertex", "edge", "face")
        else GeometryAssociationKind.SURFACE
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    face_set = mesh.entity_set(2)
    faces = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    )
    rows = simplex_rows(construction.triangles, faces)
    patches = construction.triangle_patches[rows]
    charts = construction.triangle_charts[rows]
    normals, _ = domain.oriented_normals(patches, charts, budget=budget)
    flat = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    face_association = GeometryAssociation(
        kind,
        source.source_id,
        revision,
        face_set.entity_set_id,
        face_ids,
        domain.entity_ids(np.full(patches.shape, 2, dtype=np.int64), patches),
        construction.triangle_deviations[rows],
        exact=False,
        parameters=charts,
        orientations=np.sign(np.sum(flat * normals, axis=1)).astype(np.int8),
        **_source_metadata(domain, np.full(patches.shape, 2, dtype=np.int8), patches),
    )
    edge_set = mesh.entity_set(1)
    # ty: ignore[unresolved-attribute]
    canonical_edges = np.asarray(mesh.connectivity.edges, dtype=np.int64)
    edge_rows = simplex_rows(canonical_edges, construction.curve_edges)
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[edge_rows]
    order = np.argsort(edge_ids, kind="stable")
    edge_ids = edge_ids[order]
    curves = construction.curve_edge_curves[order]
    along = construction.curve_edges[order]
    canonical = canonical_edges[edge_rows[order]]
    tangent = points[along[:, 1]] - points[along[:, 0]]
    sense = np.ones(curves.shape, dtype=np.int8)
    if kind is GeometryAssociationKind.BREP:
        for row, curve in enumerate(curves):
            patch, loop, position = domain.curve_owners[curve]
            use = domain.patches[patch].loops[loop][position]
            sense[row] = np.sign(use.last - use.first)
    edge_association = GeometryAssociation(
        kind,
        source.source_id,
        revision,
        edge_set.entity_set_id,
        edge_ids,
        domain.entity_ids(np.full(curves.shape, 1, dtype=np.int64), curves),
        construction.curve_edge_deviations[order],
        exact=False,
        orientations=np.sign(
            np.sum((points[canonical[:, 1]] - points[canonical[:, 0]]) * tangent, axis=1)
        ).astype(np.int8)
        * sense,
        **_source_metadata(domain, np.full(curves.shape, 1, dtype=np.int8), curves),
    )

    surface_patches = tuple(
        MeshPatch(
            f"surface:{patch}",
            mesh_entity_scope(mesh, 2, face_ids[patches == patch]),
        )
        for patch in np.unique(patches).tolist()
    )
    curve_labels = tuple(
        MeshLabel(
            f"curve:{curve}",
            mesh_entity_scope(mesh, 1, edge_ids[curves == curve]),
        )
        for curve in np.unique(curves).tolist()
    )
    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    vertex_association = GeometryAssociation(
        kind,
        source.source_id,
        revision,
        vertex_set.entity_set_id,
        vertex_ids,
        domain.entity_ids(
            construction.vertex_source_dimensions,
            construction.vertex_source_indices,
        ),
        np.zeros((vertex_ids.size,)),
        parameters=construction.vertex_parameters,
        exact=kind is not GeometryAssociationKind.BREP,
        **_source_metadata(
            domain,
            construction.vertex_source_dimensions,
            construction.vertex_source_indices,
        ),
    )
    return (
        (face_association, edge_association, vertex_association),
        surface_patches,
        curve_labels,
    )
