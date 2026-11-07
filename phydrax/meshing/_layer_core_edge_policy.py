#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-derived required core-edge strata for fixed and conforming layers."""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

from ..geometry._planar_primitives import plane_key, plane_orientation
from ._volume_generation import _authored_coplanar_edge_rows, _source_rows

if TYPE_CHECKING:
    from ._contracts import VolumeMeshingSpec
    from ._plc_source import PiecewiseLinearComplex
    from ._layer_core_subdivision import NestedLayerPartition
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry._surface_source_support import SurfaceSourceCharts
    from .providers._native_sources import NativeLayerCoreSource

type _PlaneSides = tuple[tuple[int, int, int, int], tuple[int, int]]


def _facet_plane_sides(
    complex_: PiecewiseLinearComplex, /
) -> tuple[_PlaneSides | None, ...]:
    """Return exact oriented plane/material signatures, or None when nonplanar."""
    points = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(complex_.vertices, dtype=np.float64)
    )
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    sides = np.asarray(complex_.facet_regions, dtype=np.int64)
    signatures: list[_PlaneSides | None] = [None] * len(sides)
    invalid = set()
    for polygon, facet in enumerate(facets):
        facet = int(facet)
        loop = tuple(
            points[int(row)] for row in vertices[offsets[polygon] : offsets[polygon + 1]]
        )
        triangle = None
        for corner in range(2, len(loop)):
            candidate = (loop[0], loop[1], loop[corner])
            try:
                plane = plane_key(candidate)
            except ValueError:
                continue
            triangle = candidate
            break
        if triangle is None:
            raise ValueError("A source PLC polygon has no noncollinear corners.")
        if any(
            sum(plane[axis] * point[axis] for axis in range(3)) + plane[3] != 0
            for point in loop
        ):
            invalid.add(facet)
            continue
        region_pair = (int(sides[facet, 0]), int(sides[facet, 1]))
        signature = (
            plane,
            region_pair if plane_orientation(triangle) > 0 else region_pair[::-1],
        )
        previous = signatures[facet]
        if previous is not None and previous != signature:
            invalid.add(facet)
        signatures[facet] = signature
    return tuple(
        None if row in invalid else signature for row, signature in enumerate(signatures)
    )


def conforming_physical_edge_rows(complex_: PiecewiseLinearComplex, /) -> tuple[int, ...]:
    """Identify original noncoplanar and material-discontinuous PLC edges."""
    signatures = _facet_plane_sides(complex_)
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    directed_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for polygon, facet in enumerate(polygon_facets):
        loop = tuple(
            int(value) for value in vertices[offsets[polygon] : offsets[polygon + 1]]
        )
        for index, first in enumerate(loop):
            last = loop[(index + 1) % len(loop)]
            key = (min(first, last), max(first, last))
            directed_uses.setdefault(key, []).append(
                (int(facet), 1 if first < last else -1)
            )
    explicit = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in np.asarray(complex_.segments, dtype=np.int64)
    }
    required = set()
    for row, facets in enumerate(complex_.plc_edge_facets):
        facet_rows = tuple(int(facet) for facet in facets)
        first, last = complex_.plc_edge_vertices[row]
        key = (min(first, last), max(first, last))
        if key in explicit:
            required.add(row)
            continue
        if len(facet_rows) == 1:
            uses = directed_uses.get(key, ())
            if (
                signatures[facet_rows[0]] is not None
                and len(uses) == 2
                and uses[0][0] == uses[1][0] == facet_rows[0]
                and uses[0][1] == -uses[1][1]
            ):
                continue
        if (
            len(facet_rows) != 2
            or signatures[facet_rows[0]] is None
            or signatures[facet_rows[0]] != signatures[facet_rows[1]]
        ):
            required.add(row)
    return tuple(sorted(required))


def _segment_relation(
    first: tuple[Fraction, ...],
    last: tuple[Fraction, ...],
    cad_first: tuple[Fraction, ...],
    cad_last: tuple[Fraction, ...],
    /,
) -> tuple[tuple[Fraction, Fraction] | None, bool]:
    """Return positive collinear support and an isolated interior intersection."""
    direction = tuple(b - a for a, b in zip(first, last, strict=True))
    other = tuple(b - a for a, b in zip(cad_first, cad_last, strict=True))
    delta = tuple(b - a for a, b in zip(first, cad_first, strict=True))
    axis = next((i for i, value in enumerate(direction) if value), None)
    if axis is None or not any(other):
        raise ValueError("Source and CAD edges must be nondegenerate.")
    for i in range(3):
        for j in range(i + 1, 3):
            determinant = direction[i] * other[j] - direction[j] * other[i]
            if not determinant:
                continue
            t = (delta[i] * other[j] - delta[j] * other[i]) / determinant
            u = (delta[i] * direction[j] - delta[j] * direction[i]) / determinant
            if all(
                first[k] + t * direction[k] == cad_first[k] + u * other[k]
                for k in range(3)
            ):
                return None, bool(0 < t < 1 and 0 <= u <= 1)
            return None, False
    if any(delta[i] * direction[axis] != delta[axis] * direction[i] for i in range(3)):
        return None, False
    a = delta[axis] / direction[axis]
    b = (cad_last[axis] - first[axis]) / direction[axis]
    lower, upper = max(Fraction(0), min(a, b)), min(Fraction(1), max(a, b))
    return ((lower, upper), False) if lower < upper else (None, False)


class _PlanarCadPatch(NamedTuple):
    patch_row: int
    entity_id: str
    occurrence: tuple[str, ...]
    material_sides: tuple[str | None, str | None]


class _PlanarCadCurveUse(NamedTuple):
    patch_row: int
    loop_index: int
    use_index: int
    curve_row: int
    entity_id: str
    occurrence: tuple[str, ...]
    parameters: tuple[Fraction, Fraction]


class _PlanarCadCurveSegment(NamedTuple):
    first: tuple[Fraction, ...]
    last: tuple[Fraction, ...]
    uses: tuple[_PlanarCadCurveUse, ...]


class _PlanarCadStrata(NamedTuple):
    triangles: tuple[tuple[tuple[Fraction, ...], ...], ...]
    charts: tuple[tuple[int, tuple[tuple[Fraction, Fraction], ...]], ...]
    patches: tuple[_PlanarCadPatch, ...]
    segments: tuple[_PlanarCadCurveSegment, ...]


def prepare_planar_cad_strata(
    domain: MeshingDomain, charts: SurfaceSourceCharts, /
) -> _PlanarCadStrata | None:
    """Reprove a complete planar B-Rep or native surface atlas and its curves."""
    from ..discretization._cell_geometry_transition_types import _geometry_map_expressions
    from ..discretization._coordinate_enclosure import (
        compose_expression,
        evaluate_expression,
    )
    from ..discretization._nested_reference import _chart_from_corners
    from ..discretization._reference_cell import reference_cell_topology
    from ..geometry._mesh_certificates import _same_tree_content
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry._source_fidelity import reprove_actual_planar_chart_rows
    from ..geometry._surface_source_support import (
        SurfaceSourceCharts,
        _cad_root_edges,
        validate_cad_curve_ancestry,
        validate_surface_curve_ancestry,
    )
    from ._layer_core_staged_domain import _source_sides, exact_planar_atlas_triangles
    from ._layer_core_strata_replay import source_line_interval

    if not isinstance(domain, MeshingDomain) or not isinstance(
        charts, SurfaceSourceCharts
    ):
        raise TypeError(
            "Planar CAD strata require the actual source domain and surface charts."
        )
    if domain.brep_authority is None and domain.source_kinds != ("corner", "curve", "surface"):
        return None
    if not _same_tree_content(charts.domain, domain):
        raise ValueError("CAD strata changed the original source domain.")
    if tuple(charts.boundary_source.patches) != tuple(range(len(domain.patches))):
        return None
    if reprove_actual_planar_chart_rows(charts) is None:
        return None
    if set(int(value) for value in np.asarray(charts.chart_witness.patches)) != set(
        charts.boundary_source.patches
    ):
        return None
    triangles, chart_rows = exact_planar_atlas_triangles(charts, charts.boundary_source)
    patches = tuple(
        _PlanarCadPatch(
            row,
            domain.entity_id(2, row),
            tuple(domain.source_occurrences[2][row]),
            _source_sides(domain, row),
        )
        for row in charts.boundary_source.patches
    )
    curve_edges = (
        validate_cad_curve_ancestry(charts, 0.0)
        if domain.brep_authority is not None
        else validate_surface_curve_ancestry(charts, 0.0)
    )
    root_edges, incident = _cad_root_edges(charts)
    root_mesh, root_geometry = charts.mesh, charts.geometry
    resolved = root_geometry.resolve(root_mesh)
    maps = _geometry_map_expressions(root_mesh, root_geometry, (resolved[0], resolved[1]))
    root_cells = {
        int(identifier): (block.cell_kind, tuple(int(value) for value in vertices))
        for block in root_mesh.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    patch_by_cell = dict(zip(charts.cell_global_ids, charts.root_patches, strict=True))
    segments = []
    for edge_id in sorted(curve_edges):
        if edge_id not in root_edges or edge_id not in incident or not incident[edge_id]:
            raise ValueError("A proved CAD coedge lacks its root edge incidence.")
        cell_id = incident[edge_id][0][0]
        kind, vertex_rows = root_cells[cell_id]
        topology = reference_cell_topology(kind)
        corners = tuple(
            tuple(
                Fraction.from_float(float(value))
                for value in topology.vertices[vertex_rows.index(vertex)]
            )
            for vertex in root_edges[edge_id]
        )
        arguments = _chart_from_corners("interval", corners)
        image = tuple(
            compose_expression(value, arguments) for value in maps.expressions(cell_id)
        )
        endpoints = tuple(
            tuple(evaluate_expression(value, (Fraction(parameter),)) for value in image)
            for parameter in (0, 1)
        )
        rational_endpoints: list[tuple[Fraction, ...]] = []
        for point in endpoints:
            values: list[Fraction] = []
            for value in point:
                if not isinstance(value, Fraction):
                    return None
                values.append(value)
            rational_endpoints.append(tuple(values))
        first, last = rational_endpoints
        if source_line_interval(image, first, last) != (Fraction(0), Fraction(1)):
            return None
        uses = tuple(
            _PlanarCadCurveUse(
                int(patch_by_cell[binding.root_cell_id]),
                binding.loop_index,
                binding.use_index,
                binding.curve_row,
                domain.entity_id(1, binding.curve_row),
                tuple(domain.source_occurrences[1][binding.curve_row]),
                binding.parameters,
            )
            for binding in curve_edges[edge_id]
        )
        if not uses:
            raise ValueError("A proved CAD coedge has no original curve use.")
        segments.append(_PlanarCadCurveSegment(first, last, uses))
    return _PlanarCadStrata(triangles, chart_rows, patches, tuple(segments))


def _mapped_noncap_source_edge_classification(
    source: NativeLayerCoreSource, partition: NestedLayerPartition | None, /
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    """Unlock only reference diagonals proved to share one actual mapped root face."""
    from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
    from .providers._native_sources import (
        NativeMappedHexSource, NativePlcSource, _mapped_layer_core_facet_relation,
    )

    reference = source.reference_source
    query = source.fidelity_source
    if reference is None or not isinstance(query, MappedReferenceBoundarySource):
        return None
    authority = query.authority
    if not isinstance(authority, (NativeMappedHexSource, NativePlcSource)):
        return None
    full = authority.complex
    if full.cad_source_charts is not None or not query.is_full_surface_scope:
        return None
    source.validate_restored()
    query.require_current()
    _mapped_layer_core_facet_relation(source, authority)
    roots_by_key: dict[tuple[str, tuple[str, ...]], set[tuple[int, int, int]]] = {}
    for row, chart_id in enumerate(query.face_chart_ids):
        key = query.chart_source_key(2, chart_id)
        roots_by_key.setdefault(key, set()).add((
            int(query.polygon_root_face_global_ids[row]),
            int(query.polygon_root_cell_global_ids[row]),
            int(query.polygon_root_local_face_indices[row]),
        ))

    complex_ = reference.complex
    points = tuple(tuple(Fraction.from_float(float(value)) for value in point)
                   for point in np.asarray(complex_.vertices, dtype=np.float64))
    full_points = tuple(tuple(Fraction.from_float(float(value)) for value in point)
                        for point in np.asarray(full.vertices, dtype=np.float64))
    full_segments = tuple(
        (full_points[int(a)], full_points[int(b)])
        for a, b in np.asarray(full.segments, dtype=np.int64)
    )
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    cap_polygons = set(int(value) for value in np.asarray(reference.cap_polygon_ids, dtype=np.int64))
    seam_polygons = set()
    if reference.core_seam_polygon_pairs is not None:
        seam_polygons.update(
            int(value) for value in np.asarray(reference.core_seam_polygon_pairs, dtype=np.int64).reshape(-1)
        )
    proven_seam_diagonals: set[tuple[int, int]] = set()
    if seam_polygons and partition is not None:
        from ._layer_core_periodic import partition_periodic_seams

        def polygon_edges(row: int) -> set[tuple[int, int]]:
            polygon = tuple(int(value) for value in vertices[offsets[row]:offsets[row + 1]])
            return {
                (min(first, polygon[(index + 1) % len(polygon)]),
                 max(first, polygon[(index + 1) % len(polygon)]))
                for index, first in enumerate(polygon)
            }

        for seam in partition_periodic_seams(reference, partition):
            for first, second in seam.source_polygons:
                shared = polygon_edges(first) & polygon_edges(second)
                if len(shared) != 1:
                    raise ValueError("A proved periodic seam quad lacks one internal source diagonal.")
                proven_seam_diagonals.update(shared)
    locked: set[tuple[int, int]] = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in np.asarray(complex_.segments, dtype=np.int64)
    }
    for row in cap_polygons | seam_polygons:
        polygon = tuple(int(value) for value in vertices[offsets[row]:offsets[row + 1]])
        for index, first in enumerate(polygon):
            last = polygon[(index + 1) % len(polygon)]
            edge = (min(first, last), max(first, last))
            if row in cap_polygons or edge not in proven_seam_diagonals:
                locked.add(edge)
    physical_rows = set(conforming_physical_edge_rows(complex_))
    removable: set[tuple[int, int]] = set()
    internal_roots: dict[tuple[int, ...], dict[int, set[tuple[int, int, int]]]] = {}
    authored_segments: set[tuple[int, int]] = set()
    for edge_row, raw_edge in enumerate(complex_.plc_edge_vertices):
        pair = (min(raw_edge), max(raw_edge))
        intervals = []
        crossing = False
        for first, last in full_segments:
            interval, crosses = _segment_relation(points[pair[0]], points[pair[1]], first, last)
            if interval is not None:
                intervals.append(interval)
            crossing |= crosses
        cursor = Fraction(0)
        for lower, upper in sorted(intervals):
            if lower > cursor:
                break
            cursor = max(cursor, upper)
        if cursor == 1:
            authored_segments.add(pair)
            continue
        if intervals or crossing:
            raise ValueError("An authored source segment crosses only part of a mapped PLC edge.")
        if pair in locked or edge_row in physical_rows:
            continue
        facets = tuple(int(value) for value in complex_.plc_edge_facets[edge_row])
        keys = tuple(
            (reference.core_facet_source_ids[facet], reference.core_facet_source_occurrence_paths[facet])
            for facet in facets
        )
        if not keys:
            continue
        if all(identity is None and path is None for identity, path in keys):
            sides = np.asarray(complex_.facet_regions, dtype=np.int64)
            if any(sides[facet, 0] < 0 or sides[facet, 0] != sides[facet, 1] for facet in facets):
                continue
            selection = tuple(sorted(set(facets)))
            if selection not in internal_roots:
                from ._layer_core_facet_authority import MappedCoreFacetAuthority

                relation = MappedCoreFacetAuthority(source, selection).root_relation
                root_sets_by_facet: dict[int, set[tuple[int, int, int]]] = {
                    facet: set() for facet in selection
                }
                for facet, face, cells, locals_ in zip(
                    relation.polygon_source_facet_rows,
                    relation.polygon_root_face_global_ids,
                    relation.polygon_root_cell_global_ids,
                    relation.polygon_root_local_face_indices,
                    strict=True,
                ):
                    root_sets_by_facet[int(facet)].add(
                        (int(face), int(cells[0]), int(locals_[0]))
                    )
                internal_roots[selection] = root_sets_by_facet
            root_sets = tuple(internal_roots[selection][facet] for facet in facets)
        elif any(identity is None or path is None for identity, path in keys):
            continue
        else:
            root_sets = tuple(roots_by_key.get((identity, path), set()) for identity, path in keys)
        if all(len(roots) == 1 for roots in root_sets) and all(roots == root_sets[0] for roots in root_sets):
            removable.add(pair)
    return frozenset(removable), frozenset(authored_segments)


def noncap_source_edge_classification(
    source: NativeLayerCoreSource, /, *, partition: NestedLayerPartition | None = None,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    """Prove removable PLC diagonals and whole-edge CAD coedge coverage.

    None means the actual CAD atlas does not admit this exact straight-edge
    proof. Callers must then preserve all execution edges or refuse the route.
    """
    from ..geometry._meshing_domain import MeshingDomain

    if source.mapped_domain is not None:
        return _mapped_noncap_source_edge_classification(source, partition)
    source.validate_restored()
    authored = source if source.reference_source is None else source.reference_source
    charts = authored.source_boundary_charts
    domain = authored.source_domain
    if charts is None or not isinstance(domain, MeshingDomain):
        return None
    strata = prepare_planar_cad_strata(domain, charts)
    if strata is None:
        return None
    cad_segments = tuple((segment.first, segment.last) for segment in strata.segments)

    complex_ = authored.complex
    points = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(complex_.vertices, dtype=np.float64)
    )
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    cap_edges: set[tuple[int, int]] = set()
    for raw in np.asarray(authored.cap_polygon_ids, dtype=np.int64):
        row = int(raw)
        polygon = tuple(
            int(value) for value in polygon_vertices[offsets[row] : offsets[row + 1]]
        )
        for index, vertex in enumerate(polygon):
            other = polygon[(index + 1) % len(polygon)]
            cap_edges.add((min(vertex, other), max(vertex, other)))
    explicit = {
        tuple(sorted((int(a), int(b))))
        for a, b in np.asarray(complex_.segments, dtype=np.int64)
    }
    physical_rows = set(conforming_physical_edge_rows(complex_))
    removable: set[tuple[int, int]] = set()
    cad: set[tuple[int, int]] = set()
    for edge_row, raw_edge in enumerate(complex_.plc_edge_vertices):
        pair = (min(raw_edge), max(raw_edge))
        intervals = []
        crossing = False
        for first, last in cad_segments:
            interval, crosses = _segment_relation(
                points[pair[0]], points[pair[1]], first, last
            )
            if interval is not None:
                intervals.append(interval)
            crossing |= crosses
        cursor = Fraction(0)
        for lower, upper in sorted(intervals):
            if lower > cursor:
                break
            cursor = max(cursor, upper)
        if cursor == 1:
            cad.add(pair)
            continue
        if intervals or crossing:
            raise ValueError("A CAD coedge crosses only part of an original PLC edge.")
        facets = tuple(int(value) for value in complex_.plc_edge_facets[edge_row])
        if pair in cap_edges or pair in explicit or edge_row in physical_rows:
            continue
        if not facets:
            raise ValueError("A proved virtual source edge has no facet incidence.")
        identities = tuple(
            (
                authored.core_facet_source_ids[row],
                authored.core_facet_source_occurrence_paths[row],
            )
            for row in facets
        )
        if identities[0][0] is not None and all(identity == identities[0] for identity in identities):
            removable.add(pair)
    return frozenset(removable), frozenset(cad)


def required_core_source_edge_rows(
    source: NativeLayerCoreSource, specification: VolumeMeshingSpec, /
) -> tuple[int, ...]:
    """Recompute required physical and protected edges under the source policy."""
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    source_rows = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(complex_.plc_edge_ids, complex_.plc_edge_occurrence_paths, strict=True)
        )
    }
    required = (
        set(conforming_physical_edge_rows(complex_))
        if source.conforming_execution
        else set(int(row) for row in _authored_coplanar_edge_rows(complex_))
    )
    for identifier, path in zip(
        complex_.segment_ids, complex_.segment_occurrence_paths, strict=True
    ):
        row = source_rows.get((identifier, tuple(path)))
        if row is None:
            raise ValueError("An explicit source segment lacks its canonical PLC edge.")
        required.add(row)
    from .providers._native_layer import _lower_core_specification

    core_specification = _lower_core_specification(source, specification)
    for feature in core_specification.protected_features:
        if feature.scope.entity_dimension == 1 and feature.feature_kind.value == "curve":
            required.update(
                int(row)
                for row in _source_rows(
                    np.asarray(feature.scope.entity_ids, dtype=np.int64),
                    complex_.plc_edge_entity_ids,
                    "PLC edge",
                )
            )
    return tuple(sorted(required))
