#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent exact staged boundary proof against retained planar CAD charts."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..discretization._reference_cell import reference_cell_topology
from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._domain_coverage import PiecewiseLinearDomain
from ..geometry._meshing_domain import MeshingDomain
from ..geometry._planar_primitives import signed_area2
from ..geometry._source_fidelity import reprove_actual_planar_chart_rows
from ..geometry._surface_source_support import SurfaceSourceCharts
from ._layer_core_association import composite_region_identity
from ._layer_core_topology import combined_domain, prepare_identity

if TYPE_CHECKING:
    from .providers._native_sources import NativeLayerCoreSource


type _Point = tuple[Fraction, ...]
type _Triangle = tuple[_Point, ...]
type _Chart = tuple[tuple[Fraction, Fraction], ...]


def exact_planar_atlas_triangles(
    atlas: SurfaceSourceCharts, query: MeshingDomainBoundarySource, /
) -> tuple[tuple[_Triangle, ...], tuple[tuple[int, _Chart], ...]]:
    """Reprove complete original source charts and return their actual exact images."""
    atlas.require_current()
    atlas.require_root(atlas.mesh, atlas.geometry)
    atlas.require_bound(atlas.mesh, atlas.geometry, atlas.boundary_source)
    if atlas.domain.domain_id != query.domain.domain_id or not set(query.patches) <= set(
        atlas.boundary_source.patches
    ):
        raise ValueError("Planar atlas changes the original source authority or scope.")
    actual = reprove_actual_planar_chart_rows(atlas)
    if actual is None:
        raise ValueError(
            "Staged CAD boundary requires an independently proved exact planar atlas."
        )
    rows = {int(identifier): row for row, identifier in enumerate(atlas.cell_global_ids)}
    corners = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(reference_cell_topology("triangle").vertices)
    )
    triangles: list[_Triangle] = []
    metadata: list[tuple[int, _Chart]] = []
    for block in atlas.mesh.blocks:
        if block.cell_kind != "triangle":
            raise ValueError("Staged CAD boundary requires triangular atlas cells.")
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        exact_cells = atlas.geometry.evaluate_exact(
            atlas.mesh, identifiers, tuple(corners for _ in identifiers)
        )
        for identifier, triangle in zip(identifiers, exact_cells, strict=True):
            row = rows[int(identifier)]
            patch = int(atlas.root_patches[row])
            if patch not in query.patches:
                continue
            exact_points: list[_Point] = []
            for point in triangle:
                exact_coordinates: list[Fraction] = []
                for value in point:
                    if not isinstance(value, Fraction):
                        raise ValueError(
                            "Staged CAD charts require exact rational source coordinates."
                        )
                    exact_coordinates.append(value)
                exact_points.append(tuple(exact_coordinates))
            triangles.append(tuple(exact_points))
            metadata.append((patch, actual[row]))
    return tuple(triangles), tuple(metadata)


def _normal(triangle: _Triangle, /) -> _Point:
    first = tuple(triangle[1][axis] - triangle[0][axis] for axis in range(3))
    second = tuple(triangle[2][axis] - triangle[0][axis] for axis in range(3))
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _source_sides(domain: MeshingDomain, patch: int, /) -> tuple[str | None, str | None]:
    result: list[str | None] = [None, None]
    for row, region in enumerate(domain.regions):
        for region_patch, sign in region.boundary:
            if region_patch != patch:
                continue
            slot = 0 if sign > 0 else 1
            identity = domain.entity_id(3, row)
            if result[slot] not in (None, identity):
                raise ValueError("Original CAD patch has ambiguous material incidence.")
            result[slot] = identity
    return result[0], result[1]


def validate_brep_staged_layer_domain(
    source: NativeLayerCoreSource, /
) -> PiecewiseLinearDomain:
    """Reconstruct and prove the staged PLC/layer boundary against original CAD.

    Cap incidence is proved by the combined-domain construction, not promoted to
    an original CAD face. Every exterior piece is independently partitioned in
    both the retained source atlas and the actual combined carrier.
    """
    from .providers._native_sources import NativeLayerCoreSource

    if type(source) is not NativeLayerCoreSource:
        raise TypeError("Expected a native layer/core source.")
    source.validate_restored()
    domain, query, atlas = (
        source.source_domain,
        source.fidelity_source,
        source.source_boundary_charts,
    )
    if (
        not isinstance(domain, MeshingDomain)
        or not isinstance(query, MeshingDomainBoundarySource)
        or not isinstance(atlas, SurfaceSourceCharts)
    ):
        raise TypeError(
            "Staged CAD proof requires the retained domain, boundary query and atlas."
        )
    domain.require_current(source.source_id, source.source_revision)
    query.require_current()
    if query.domain.domain_id != domain.domain_id:
        raise ValueError("Staged CAD query changes the original source domain.")
    mapping, cap, polygons, points = prepare_identity(
        source.layers, source.complex, source.vertex_layer_ids, source.cap_polygon_ids
    )
    combined, _ = combined_domain(
        source.layers,
        source.complex,
        source.layer_regions,
        mapping,
        cap,
        polygons,
        points,
        source.source_id,
        source.region_ids,
        source.core_region_map,
        source.region_occurrence_paths,
    )
    source_triangles, metadata = exact_planar_atlas_triangles(atlas, query)
    coordinates = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(combined.vertices, dtype=np.float64)
    )
    facets = np.asarray(combined.facets, dtype=np.int64)
    incidence = np.asarray(combined.facet_regions, dtype=np.int64)
    if any(int(pair[0]) >= 0 and int(pair[1]) >= 0 for pair in incidence):
        raise ValueError(
            "Staged planar CAD proof requires explicit original material-interface charts."
        )
    target = tuple(
        tuple(coordinates[int(vertex)] for vertex in facet) for facet in facets
    )
    pieces = exact_coplanar_triangle_partitions(source_triangles, target)

    cap_rows = frozenset(int(value) for value in cap)
    core_by_vertices: dict[tuple[int, ...], int] = {}
    for row, polygon in enumerate(polygons):
        facet = int(source.complex.polygon_facets[row])
        key = tuple(sorted(int(value) for value in mapping[polygon]))
        if row in cap_rows:
            if (
                source.core_facet_source_ids[facet] is not None
                or source.core_facet_source_occurrence_paths[facet] is not None
            ):
                raise ValueError(
                    "Internal layer/core cap must not acquire a CAD facet identity."
                )
            continue
        if key in core_by_vertices:
            raise ValueError("Staged core repeats a boundary polygon.")
        core_by_vertices[key] = facet
    for piece in pieces:
        patch, uv = metadata[piece.source_row]
        source_triangle, target_triangle = (
            source_triangles[piece.source_row],
            target[piece.target_row],
        )
        source_normal, target_normal = _normal(source_triangle), _normal(target_triangle)
        dot = sum(
            (
                left * right
                for left, right in zip(source_normal, target_normal, strict=True)
            ),
            Fraction(0),
        )
        orientation = (1 if signed_area2(uv) > 0 else -1) * domain.patches[
            patch
        ].orientation
        if not dot:
            raise ValueError("Staged boundary has a degenerate oriented source overlap.")
        sides = _source_sides(domain, patch)
        pair = incidence[piece.target_row]
        actual = tuple(
            None if int(value) < 0 else composite_region_identity(source, int(value))
            for value in pair
        )
        if (sides if dot * orientation > 0 else sides[::-1]) != actual:
            raise ValueError(
                "Staged boundary changes the original oriented material sides."
            )
        key = tuple(sorted(int(value) for value in facets[piece.target_row]))
        facet = core_by_vertices.get(key)
        if facet is not None and (
            source.core_facet_source_ids[facet] != domain.entity_id(2, patch)
            or source.core_facet_source_occurrence_paths[facet]
            != tuple(domain.source_occurrences[2][patch])
        ):
            raise ValueError("Staged core facet changes its original CAD occurrence.")
    return combined
