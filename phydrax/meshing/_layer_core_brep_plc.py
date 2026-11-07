#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Assemble a fixed layer cap and exact remaining original CAD faces."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions
from ..geometry._planar_primitives import plane_orientation, signed_area2
from ._layer_core_brep_arrangement import remaining_brep_face_triangles


if TYPE_CHECKING:
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry._surface_source_support import SurfaceSourceCharts
    from ._boundary_layer import BoundaryLayerMesh
    from ._layer_core_cad_footprints import LayerPlanarCadFootprints


type _Point = tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class BrepLayerCorePlcRows:
    """Host-only PLC rows; cap polygons and vertices retain their original order.

    ``points`` are exact binary64-representable physical coordinates. A vertex
    with both ancestry rows -1 is a new exact arrangement vertex. A polygon
    with wall cell -1 has no single original wall-cell owner.
    """

    points: tuple[_Point, ...]
    polygons: tuple[tuple[int, int, int], ...]
    polygon_facets: tuple[int, ...]
    vertex_layer_ids: tuple[int, ...]
    vertex_wall_rows: tuple[int, ...]
    polygon_wall_cells: tuple[int, ...]
    facet_keys: tuple[tuple[str, int], ...]
    cap_count: int


def remaining_brep_core_plc_rows(
    domain: MeshingDomain,
    charts: SurfaceSourceCharts,
    layers: BoundaryLayerMesh,
    footprints: LayerPlanarCadFootprints,
    /,
) -> BrepLayerCorePlcRows:
    """Bind the unchanged actual cap to arranged original source-face regions."""
    wall = charts.mesh
    if layers.source_wall.mesh_id != wall.mesh_id:
        raise ValueError("Layer cap ancestry must use the actual charted wall.")
    if layers.cap is None:
        raise ValueError("A fixed layer/core source requires an actual free cap.")
    if len(domain.regions) != 1:
        raise ValueError("Planar layer/core assembly requires one source solid.")
    cap = layers.cap
    if any(block.cell_kind != "triangle" for block in cap.blocks) or any(
        block.cell_kind != "triangle" for block in wall.blocks
    ):
        raise ValueError("Fixed layer/core cap and source wall must be triangular.")
    cap_rows = tuple(
        (int(row[0]), int(row[1]), int(row[2]))
        for block in cap.blocks
        for row in np.asarray(block.vertices, dtype=np.int64)
    )
    wall_rows = tuple(
        (int(identifier), tuple(int(value) for value in row))
        for block in wall.blocks
        for identifier, row in zip(block.global_ids, block.vertices, strict=True)
    )
    cap_layer = tuple(
        int(value) for value in np.asarray(layers.cap_vertices, dtype=np.int64)
    )
    layer_coordinates = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    wall_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        wall.coordinates, dtype=np.float64
    )
    cap_coordinates = np.asarray(cap.coordinates, dtype=np.float64)
    if (
        len(cap_layer) != len(cap_coordinates)
        or len(set(cap_layer)) != len(cap_layer)
        or any(row < 0 or row >= len(layer_coordinates) for row in cap_layer)
    ):
        raise ValueError("The fixed cap lacks unique layer-vertex ancestry.")
    if not np.array_equal(
        cap_coordinates, layer_coordinates[np.asarray(cap_layer, dtype=np.int64)]
    ):
        raise ValueError("The fixed cap changed its layer-vertex coordinates.")
    if any(any(row < 0 or row >= len(cap_layer) for row in face) for face in cap_rows):
        raise ValueError("A fixed cap triangle has invalid vertex incidence.")

    def exact(point: np.ndarray) -> _Point:
        if len(point) != 3 or not np.all(np.isfinite(point)):
            raise ValueError(
                "Core vertices must have finite three-dimensional coordinates."
            )
        return (
            Fraction.from_float(float(point[0])),
            Fraction.from_float(float(point[1])),
            Fraction.from_float(float(point[2])),
        )

    points = [exact(point) for point in cap_coordinates]
    if len(set(points)) != len(points):
        raise ValueError("Distinct fixed cap vertices have coincident coordinates.")
    point_rows = {point: row for row, point in enumerate(points)}
    vertex_layer_ids = list(cap_layer)
    vertex_wall_rows = [-1] * len(points)
    wall_point_rows: dict[_Point, list[int]] = {}
    for row, point in enumerate(wall_coordinates):
        wall_point_rows.setdefault(exact(point), []).append(row)

    wall_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    layer_sources = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    if layer_sources.shape != (len(layer_coordinates),):
        raise ValueError("Layer vertices lack source-wall ancestry.")
    wall_by_vertices: dict[tuple[int, ...], int] = {}
    for identifier, rows in wall_rows:
        key = tuple(sorted(int(wall_ids[row]) for row in rows))
        if key in wall_by_vertices:
            raise ValueError("Wall cells have ambiguous vertex ancestry.")
        wall_by_vertices[key] = identifier
    patch_of_cell = dict(
        zip(map(int, charts.cell_global_ids), map(int, charts.root_patches), strict=True)
    )
    cap_patches = []
    for face in cap_rows:
        key = tuple(sorted(int(layer_sources[cap_layer[row]]) for row in face))
        cell = wall_by_vertices.get(key)
        if cell is None or cell not in patch_of_cell:
            raise ValueError(
                "Each fixed cap triangle must cap one actual charted wall cell."
            )
        cap_patches.append(patch_of_cell[cell])
    cap_edges = tuple(
        (points[face[corner]], points[face[(corner + 1) % 3]])
        for face in cap_rows
        for corner in range(3)
    )
    remaining = remaining_brep_face_triangles(footprints, cap_edges)
    source = footprints.strata.triangles
    source_charts = footprints.strata.charts
    if len(source) != len(wall_rows) or len(source_charts) != len(wall_rows):
        raise ValueError("The proved atlas does not match the actual wall cells.")
    for _, triangle in remaining:
        for point in triangle:
            for first, last in cap_edges:
                direction = tuple(last[axis] - first[axis] for axis in range(3))
                axis = next((axis for axis, value in enumerate(direction) if value), None)
                if axis is None:
                    raise ValueError("A fixed cap edge is degenerate.")
                parameter = (point[axis] - first[axis]) / direction[axis]
                if 0 < parameter < 1 and all(
                    point[index] == first[index] + parameter * direction[index]
                    for index in range(3)
                ):
                    raise ValueError("A remaining face would split a fixed cap edge.")
    for (cell, rows), triangle, (patch, _) in zip(
        wall_rows, source, source_charts, strict=True
    ):
        if (
            patch_of_cell.get(cell) != patch
            or tuple(exact(wall_coordinates[row]) for row in rows) != triangle
        ):
            raise ValueError(
                "The proved atlas changed its source wall-cell geometry or patch."
            )
    region_sign: dict[int, int] = {}
    for patch, sign in domain.regions[0].boundary:
        if patch in region_sign or sign not in (-1, 1):
            raise ValueError("The source solid has ambiguous patch incidence.")
        region_sign[patch] = int(sign)
    orientation_by_patch: dict[int, int] = {}
    for triangle, (patch, chart) in zip(source, source_charts, strict=True):
        area = signed_area2(chart)
        if not area or patch not in region_sign:
            raise ValueError("The source chart has no oriented solid incidence.")
        orientation = (
            plane_orientation(triangle)
            * (1 if area > 0 else -1)
            * int(domain.patches[patch].orientation)
            * region_sign[patch]
        )
        if patch in orientation_by_patch and orientation_by_patch[patch] != orientation:
            raise ValueError("A source patch has inconsistent exact orientation.")
        orientation_by_patch[patch] = orientation
    oriented = []
    for patch, triangle in remaining:
        if patch not in orientation_by_patch:
            raise ValueError("A remaining face has no proved source orientation.")
        if plane_orientation(triangle) != orientation_by_patch[patch]:
            triangle = (triangle[0], triangle[2], triangle[1])
        oriented.append((patch, triangle))
    overlap = (
        exact_coplanar_triangle_partitions(
            source,
            tuple(triangle for _, triangle in oriented),
            require_source=False,
            require_target=True,
        )
        if oriented
        else ()
    )
    owners: list[set[int]] = [set() for _ in oriented]
    for piece in overlap:
        if source_charts[piece.source_row][0] != oriented[piece.target_row][0]:
            raise ValueError("A remaining face overlaps a different source patch.")
        owners[piece.target_row].add(wall_rows[piece.source_row][0])
    polygons = list(cap_rows)
    for _, triangle in oriented:
        rows = []
        for point in triangle:
            row = point_rows.get(point)
            if row is None:
                represented = tuple(float(value) for value in point)
                if any(
                    not np.isfinite(value) or Fraction.from_float(value) != coordinate
                    for value, coordinate in zip(represented, point, strict=True)
                ):
                    raise ValueError(
                        "An arrangement vertex is not exactly representable in binary64."
                    )
                candidates = wall_point_rows.get(point, ())
                if len(candidates) > 1:
                    raise ValueError("An arrangement vertex has ambiguous wall ancestry.")
                row = len(points)
                point_rows[point] = row
                points.append(point)
                vertex_layer_ids.append(-1)
                vertex_wall_rows.append(candidates[0] if candidates else -1)
            rows.append(row)
        polygons.append((rows[0], rows[1], rows[2]))
    facet_keys = tuple(
        sorted({("cap", patch) for patch in cap_patches})
        + sorted({("face", patch) for patch, _ in oriented})
    )
    facet_rows = {key: row for row, key in enumerate(facet_keys)}
    polygon_facets = tuple(facet_rows[("cap", patch)] for patch in cap_patches) + tuple(
        facet_rows[("face", patch)] for patch, _ in oriented
    )
    return BrepLayerCorePlcRows(
        tuple(points),
        tuple(polygons),
        polygon_facets,
        tuple(vertex_layer_ids),
        tuple(vertex_wall_rows),
        tuple(
            [-1] * len(cap_rows)
            + [next(iter(group)) if len(group) == 1 else -1 for group in owners]
        ),
        facet_keys,
        len(cap_rows),
    )
