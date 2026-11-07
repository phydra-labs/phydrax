#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact physical layer boundary footprints on the original planar CAD atlas."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions
from ..geometry._planar_primitives import plane_key
from ..geometry._source_fidelity import _require_reference_chart_partition
from ._layer_core_edge_policy import _PlanarCadStrata, prepare_planar_cad_strata
from ._layer_core_topology import triangles
from .providers._native_layer import _layer_output_boundary_faces


if TYPE_CHECKING:
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry._surface_source_support import SurfaceSourceCharts
    from ._boundary_layer import BoundaryLayerMesh


type _Point = tuple[Fraction, Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]


@dataclass(frozen=True, slots=True)
class LayerPhysicalTriangle:
    """An oriented physical triangle and its actual layer-cell face incidence."""

    vertices: _Triangle
    face_id: int
    cell_id: int
    local_face: int
    cell_kind: str
    vertex_ids: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class LayerPlanarCadFootprints:
    """Canonical CAD strata and physical triangles aligned to its patch rows."""

    strata: _PlanarCadStrata
    patch_triangles: tuple[tuple[LayerPhysicalTriangle, ...], ...]


def layer_planar_cad_footprints(
    domain: MeshingDomain, charts: SurfaceSourceCharts, layers: BoundaryLayerMesh, /
) -> LayerPlanarCadFootprints | None:
    """Prove each source-touching layer triangle belongs to one actual CAD patch.

    The unassigned, zero-overlap exterior triangles include the internal core
    cap. They acquire no CAD identity. Nonzero partial or multi-patch overlap
    is not sufficient authority and is rejected.
    """
    from ._boundary_layer import BoundaryLayerMesh

    strata = prepare_planar_cad_strata(domain, charts)
    if strata is None:
        return None
    if not isinstance(layers, BoundaryLayerMesh):
        raise TypeError("Planar CAD footprints require the actual boundary layer mesh.")
    mesh = layers.mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if coordinates.shape != (len(identifiers), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("Layer vertices must have finite three-dimensional coordinates.")
    row_by_id = {int(identifier): row for row, identifier in enumerate(identifiers)}
    if len(row_by_id) != len(identifiers):
        raise ValueError("Layer vertex identities are not unique.")
    points = tuple(
        (
            Fraction.from_float(float(point[0])),
            Fraction.from_float(float(point[1])),
            Fraction.from_float(float(point[2])),
        )
        for point in coordinates
    )
    cap_triangles: set[tuple[int, int, int]] = set()
    if layers.cap is not None:
        cap_rows = np.asarray(layers.cap_vertices, dtype=np.int64)
        if cap_rows.shape != (len(layers.cap.vertex_global_ids),) or np.any(
            (cap_rows < 0) | (cap_rows >= len(identifiers))
        ):
            raise ValueError("The layer cap has invalid layer-vertex ancestry.")
        for block in layers.cap.blocks:
            for face in np.asarray(block.vertices, dtype=np.int64):
                if np.any((face < 0) | (face >= len(cap_rows))):
                    raise ValueError("The layer cap has invalid face incidence.")
                for part in triangles(tuple(int(cap_rows[row]) for row in face)):
                    ordered = sorted(int(identifiers[row]) for row in part)
                    cap_triangles.add((ordered[0], ordered[1], ordered[2]))
    exterior = _layer_output_boundary_faces(mesh)
    boundary_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)[
        np.asarray(mesh.boundary_masks[2], dtype=np.bool_)
    ]
    actual_ids = [row[0][0] for row in exterior.values() if len(row) == 1]
    if len(actual_ids) != len(exterior) or set(actual_ids) != set(map(int, boundary_ids)):
        raise ValueError("Layer exterior faces lack unique volume-cell incidence.")

    physical: list[LayerPhysicalTriangle] = []
    for incident in exterior.values():
        face_id, cell_id, local_face, kind, order, _ = incident[0]
        rows = tuple(row_by_id[identifier] for identifier in order)
        parts = triangles(rows)
        if len(rows) == 4:
            plane = plane_key(
                (points[parts[0][0]], points[parts[0][1]], points[parts[0][2]])
            )
            if any(
                sum(plane[axis] * points[row][axis] for axis in range(3)) + plane[3]
                for row in rows
            ):
                raise ValueError("A nonplanar layer quad has no exact planar footprint.")
        for part in parts:
            vertices = (points[part[0]], points[part[1]], points[part[2]])
            # Check even unassigned cap triangles for a valid physical face.
            plane_key(vertices)
            physical.append(
                LayerPhysicalTriangle(
                    vertices,
                    face_id,
                    cell_id,
                    local_face,
                    kind,
                    (
                        int(identifiers[part[0]]),
                        int(identifiers[part[1]]),
                        int(identifiers[part[2]]),
                    ),
                )
            )

    pieces = exact_coplanar_triangle_partitions(
        strata.triangles,
        tuple(triangle.vertices for triangle in physical),
        require_source=False,
        require_target=False,
    )
    by_target = [[] for _ in physical]
    for piece in pieces:
        by_target[piece.target_row].append(piece)
    patch_index = {patch.patch_row: index for index, patch in enumerate(strata.patches)}
    if len(patch_index) != len(strata.patches):
        raise ValueError("The CAD strata contain ambiguous patch rows.")
    groups: list[list[LayerPhysicalTriangle]] = [[] for _ in strata.patches]
    standard = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    for target_row, overlaps in enumerate(by_target):
        if not overlaps:
            if tuple(sorted(physical[target_row].vertex_ids)) not in cap_triangles:
                raise ValueError("A non-cap layer exterior triangle has no CAD overlap.")
            continue
        owners = {strata.charts[piece.source_row][0] for piece in overlaps}
        if len(owners) != 1:
            raise ValueError("A layer triangle overlaps multiple original CAD patches.")
        owner = next(iter(owners))
        if owner not in patch_index:
            raise ValueError("A layer triangle overlaps an unproved CAD patch.")
        _require_reference_chart_partition(
            tuple(piece.target_chart for piece in overlaps), standard
        )
        groups[patch_index[owner]].append(physical[target_row])
    return LayerPlanarCadFootprints(strata, tuple(tuple(group) for group in groups))
