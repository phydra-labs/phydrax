# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Exact tensor-spline quadrilaterals on rectangular native surface charts."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ...discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    elevate_spline_extraction,
    PerCellRestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import CellBlock, CellMesh
from ...discretization._coordinate_enclosure import (
    coordinate_map_expressions,
    evaluate_expression,
    rounded_point,
)
from ...discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from ...discretization._surface_chart_records import (
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from ...geometry._meshing_domain import MeshingDomain, PatchCurveUse, SourcePreparation
from ...geometry._rational import fraction_endpoint
from ...geometry._surface_source_support import (
    _SplineSurfaceCoordinateMap,
    PreparedSurfaceSourceSupport,
    SurfaceNativeRestrictionBoundarySource,
    SurfaceSourceCharts,
)
from ...geometry.brep._patches import BSplineSurfacePatch, LineCurve
from .._association import GeometryAssociation, GeometryAssociationKind
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import MeshingFailureCategory
from .._controls import FeatureKind
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._quality import evaluate_cell_quality
from .._result import MeshingComplianceReport
from ._native_surface_association import _source_metadata, _surface_edges


if TYPE_CHECKING:
    from ._native_surface import _PreparedNativeSurface

type _ChartPoint = tuple[Fraction, Fraction]
type _TrimUse = tuple[PatchCurveUse, tuple[_ChartPoint, _ChartPoint]]


def _exact_line_endpoints(
    use: PatchCurveUse, /
) -> tuple[_ChartPoint, _ChartPoint] | None:
    curve = use.pcurve
    if not isinstance(curve, LineCurve) or any(
        not isinstance(value, Fraction)
        for value in (*curve.origin_exact, *curve.direction_exact)
    ):
        return None
    return tuple(
        tuple(
            a + Fraction.from_float(float(t)) * b
            for a, b in zip(curve.origin_exact, curve.direction_exact, strict=True)
        )
        for t in (use.first, use.last)
    )


def execute_tensor_quads(
    prepared: _PreparedNativeSurface, preparation: SourcePreparation, /
) -> NativeMeshConstruction:
    from . import _native_surface as owner

    domain = prepared.domain
    specification = prepared.specification
    selected = prepared.selected_patches
    atlas = prepared.roots.spline_root_atlas
    if (
        len(selected) != 1
        or atlas is None
        or not isinstance(domain.patches[selected[0]].surface, BSplineSurfacePatch)
        or domain.patches[selected[0]].orientation != 1
        or domain.regions
        or specification.maximum_normal_deviation is not None
        or prepared.schedule.normal_deviation_aim is not None
        or any(
            feature.feature_kind is not FeatureKind.SURFACE
            for feature in specification.protected_features
        )
    ):
        raise owner._failure(
            "exact tensor quadrilaterals require one positively oriented tensor patch without region, normal, or non-surface feature controls",
            category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
        )
    patch = selected[0]
    surface = domain.patches[patch].surface
    loops = domain.patches[patch].loops
    if (
        len(loops) != 1
        or len(loops[0]) != 4
        or any(not isinstance(use, PatchCurveUse) for use in loops[0])
    ):
        raise owner._failure(
            "exact tensor quadrilaterals require one rectangular linear trim",
            category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
        )
    root_bounds = atlas.root_parameter_bounds
    low_u = min(corner[0] for chart in root_bounds for corner in chart)
    high_u = max(corner[0] for chart in root_bounds for corner in chart)
    low_v = min(corner[1] for chart in root_bounds for corner in chart)
    high_v = max(corner[1] for chart in root_bounds for corner in chart)
    corners = ((low_u, low_v), (high_u, low_v), (high_u, high_v), (low_u, high_v))
    sides = {frozenset((corners[i], corners[(i + 1) % 4])) for i in range(4)}
    uses = []
    for use in loops[0]:
        endpoints = _exact_line_endpoints(use)
        if endpoints is None or frozenset(endpoints) not in sides:
            raise owner._failure(
                "exact tensor quadrilaterals require the complete authored rectangular trim",
                category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            )
        sides.remove(frozenset(endpoints))
        uses.append((use, endpoints))
    if sides or len({use.curve for use, _ in uses}) != 4:
        raise owner._failure(
            "rectangular trim curves must be distinct",
            category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
        )

    limit = prepared.schedule.spacing_fraction * float(prepared.size_values[0])
    n = 1
    while n < prepared.schedule.minimum_edge_samples - 1:
        n *= 2
    chords = owner._CurveChordBounds(preparation)
    while True:
        acceptable = True
        for bounds in root_bounds:
            u0, v0 = bounds[0]
            u1, v1 = bounds[2]
            for i in range(n):
                for j in range(n):
                    a, b = (
                        u0 + (u1 - u0) * Fraction(i, n),
                        u0 + (u1 - u0) * Fraction(i + 1, n),
                    )
                    c, d = (
                        v0 + (v1 - v0) * Fraction(j, n),
                        v0 + (v1 - v0) * Fraction(j + 1, n),
                    )
                    box = np.asarray(
                        (
                            (
                                fraction_endpoint(a, upper=False),
                                fraction_endpoint(c, upper=False),
                            ),
                            (
                                fraction_endpoint(b, upper=True),
                                fraction_endpoint(d, upper=True),
                            ),
                        )
                    )
                    enclosure = np.asarray(
                        preparation.surface_box(surface, box), dtype=np.float64
                    )
                    if (
                        enclosure.shape != (2, 3)
                        or not np.all(np.isfinite(enclosure))
                        or np.any(enclosure[0] > enclosure[1])
                    ):
                        raise owner._failure(
                            "surface sizing enclosure is unavailable",
                            category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        )
                    squared_diameter = sum(
                        (
                            Fraction.from_float(float(enclosure[1, axis]))
                            - Fraction.from_float(float(enclosure[0, axis]))
                        )
                        ** 2
                        for axis in range(3)
                    )
                    if squared_diameter > Fraction.from_float(limit) ** 2:
                        acceptable = False
                        break
                if not acceptable:
                    break
            if not acceptable:
                break
        if acceptable:
            for use, (first, last) in uses:
                axis = 0 if first[0] != last[0] else 1
                other = 1 - axis
                values = {Fraction(0), Fraction(1)}
                for bounds in root_bounds:
                    if first[other] not in (bounds[0][other], bounds[2][other]):
                        continue
                    lower, upper = bounds[0][axis], bounds[2][axis]
                    for index in range(n + 1):
                        coordinate = lower + (upper - lower) * Fraction(index, n)
                        values.add(
                            (coordinate - first[axis]) / (last[axis] - first[axis])
                        )
                ordered = sorted(values)
                for lower, upper in zip(ordered, ordered[1:]):
                    _, length = chords.interval(use.curve, lower, upper)
                    if length > limit:
                        acceptable = False
                        break
                if not acceptable:
                    break
        if acceptable:
            break
        if float(Fraction(1, n * 2)) == 0.0:
            raise owner._failure(
                "surface sizing reached parameter representability",
                category=MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            )
        n *= 2

    root_elements, root_routes, _ = atlas.geometry.resolve(atlas.mesh)
    coefficients = atlas.geometry.source_coordinates()
    positions = {}
    points = []
    point_uv = []
    exact_points = []
    blocks = []
    element_map = {}
    route_map = {}
    parent_cells = {}
    parent_vertices = {}
    parent_maps = {}
    source_blocks = {}
    charts = []
    pieces = []
    source_ids = []
    bindings = []
    next_id = 0
    for root_index, (root_block, root_element, root_route, bounds) in enumerate(
        zip(atlas.mesh.blocks, root_elements, root_routes, root_bounds, strict=True)
    ):
        if not isinstance(root_element, SplineCellGeometryElement):
            raise owner._failure(
                "tensor root lacks exact spline extraction",
                category=MeshingFailureCategory.INVALID_SOURCE,
            )
        order = specification.target.geometry_order
        if any(degree > order for degree in root_element.degrees):
            raise owner._failure(
                "requested geometry order is below a source tensor degree",
                category=MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            )
        elevated = elevate_spline_extraction(root_element, (order, order))
        route = np.asarray(root_route, dtype=np.int64)[0]
        local = tuple(coefficients[int(row)] for row in route)
        expressions = coordinate_map_expressions(root_element, local, cell_row=0)
        root_id = int(np.asarray(root_block.global_ids)[0])
        u0, v0 = bounds[0]
        u1, v1 = bounds[2]
        cell_vertices = []
        chart_arguments = []
        reference_corners = []
        for i in range(n):
            for j in range(n):
                r0, r1 = Fraction(i, n), Fraction(i + 1, n)
                s0, s1 = Fraction(j, n), Fraction(j + 1, n)
                local_corners = ((r0, s0), (r1, s0), (r1, s1), (r0, s1))
                uv = tuple(
                    (u0 + (u1 - u0) * r, v0 + (v1 - v0) * s) for r, s in local_corners
                )
                vertices = []
                for reference, point in zip(local_corners, uv, strict=True):
                    image = tuple(
                        evaluate_expression(expression, reference)
                        for expression in expressions
                    )
                    row = positions.get(point)
                    if row is None:
                        row = len(points)
                        positions[point] = row
                        points.append(rounded_point(image))
                        exact_points.append(image)
                        point_uv.append(point)
                    elif exact_points[row] != image:
                        raise owner._failure(
                            "adjacent spline spans disagree at a shared source vertex",
                            category=MeshingFailureCategory.INVALID_SOURCE,
                        )
                    vertices.append(row)
                cell_vertices.append(vertices)
                charts.append(uv)
                pieces.append(tuple(atlas.root_source_cell_ids[root_index]))
                source_ids.append(prepared.roots.root_cell_ids[root_index])
                bindings.append(
                    SurfaceRootMeshCellBinding(
                        cell_geometry_id(atlas.geometry), atlas.mesh.topology_id, root_id
                    )
                )
                reference_corners.append(local_corners)
                chart_arguments.append(
                    ({(0, 0): r0, (1, 0): r1 - r0}, {(0, 0): s0, (0, 1): s1 - s0})
                )
        count = len(cell_vertices)
        ids = np.arange(next_id, next_id + count, dtype=np.int64)
        next_id += count
        name = f"tensor_span_{root_index:08d}"
        blocks.append(
            CellBlock(
                name,
                "quadrilateral",
                np.asarray(cell_vertices, dtype=np.int64),
                global_ids=ids,
            )
        )
        element_map[name] = PerCellRestrictedCellGeometryElement(
            elevated,
            "quadrilateral",
            ids,
            np.full(count, root_id, dtype=np.int64),
            chart_arguments,
        )
        route_map[name] = np.broadcast_to(route, (count, route.size)).copy()
        parent_cells[name] = np.full(count, root_id, dtype=np.int64)
        parent_vertices[name] = np.full((count, 4), -1, dtype=np.int64)
        parent_maps[name] = ReferenceCellMapWitnesses(
            np.arange(count, dtype=np.int64),
            parent_cells[name],
            "quadrilateral",
            "quadrilateral",
            np.asarray(reference_corners, dtype=np.float64),
            tuple(reference_corners),
            freeze_reference_chart_arguments(chart_arguments),
        )
        source_blocks[name] = root_block.name
    mesh = CellMesh(
        np.asarray(points, dtype=np.float64),
        tuple(blocks),
        numeric_version=domain.source_revision,
    )
    root_map = atlas.geometry.source_map
    if not isinstance(root_map, _SplineSurfaceCoordinateMap):
        raise owner._failure(
            "tensor root lacks its exact source control bank",
            category=MeshingFailureCategory.INVALID_SOURCE,
        )
    source_map = _SplineSurfaceCoordinateMap(
        domain.source_id,
        domain.source_revision,
        mesh.coordinates,
        coefficients,
        atlas.geometry.coordinates,
        root_map.coefficient_keys,
        root_map.coefficient_authorities,
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(atlas.geometry),
        atlas.mesh.topology_id,
        parent_cells,
        parent_vertices,
        block_source_blocks=source_blocks,
        block_parent_reference_maps=parent_maps,
    )
    geometry = CellGeometrySpec(
        element_map,
        route_map,
        atlas.geometry.coordinates,
        source_map=source_map,
        source_mesh=mesh,
        restriction_source=restriction,
    )

    associations, junctions = _associations(domain, mesh, patch, uses, point_uv, charts)
    patch_rows = np.full(next_id, patch, dtype=np.int64)
    source_charts = SurfaceSourceCharts(
        domain,
        mesh,
        geometry,
        prepared.coordinate_contract,
        np.asarray(charts, dtype=np.float64),
        patch_rows,
        associations,
        prepared.boundary_source,
        source_root_cell_ids=np.asarray(source_ids, dtype=np.int64),
        source_piece_ids=tuple(pieces),
        root_parameter_exact=tuple(charts),
    )
    chart = source_charts.chart_witness
    witness = SurfaceChartWitness(
        geometry_id=chart.geometry_id,
        topology_id=chart.topology_id,
        domain_id=chart.domain_id,
        cell_kind="quadrilateral",
        cell_global_ids=chart.cell_global_ids,
        patches=chart.patches,
        charts=chart.charts,
        source_root_cell_ids=chart.source_root_cell_ids,
        source_piece_ids=chart.source_piece_ids,
        geometry_entity_ids=chart.geometry_entity_ids,
        occurrence_paths=chart.occurrence_paths,
        chart_rational_tokens=chart.chart_rational_tokens,
        root_mesh_cell_bindings=tuple(bindings),
    )
    support = PreparedSurfaceSourceSupport(
        domain, atlas, atlas.root_parameters, prepared.boundary_source
    )
    query = SurfaceNativeRestrictionBoundarySource(
        prepared.boundary_source, support, mesh, geometry
    )
    support.prove_native_restrictions(mesh, geometry, chart_witness=witness)
    base = owner._organization(
        domain, specification, prepared.scope_ids, mesh, patch_rows, associations
    )
    organization = NativeMeshOrganization(
        patches=base.patches,
        labels=base.labels,
        associations=associations,
        surface_source=source_charts,
    )
    quality = evaluate_cell_quality(mesh)
    minimum_angle = float(np.min(np.asarray(quality.minimum_angle, dtype=np.float64)))
    if not math.isfinite(minimum_angle):
        raise owner._failure(
            "quadrilateral quality evaluation is unresolved",
            category=MeshingFailureCategory.COMPLIANCE_FAILED,
        )
    if (
        specification.quality_target is not None
        and specification.quality_target.hard
        and minimum_angle < specification.quality_target.minimum_angle
    ):
        raise owner._failure(
            "quadrilateral mesh misses the hard minimum-angle target",
            category=MeshingFailureCategory.COMPLIANCE_FAILED,
        )
    requested = [
        ("maximum_edge_length", limit),
        ("minimum_angle_degrees", prepared.schedule.quality_angle_degrees),
    ]
    mesh_edges = _surface_edges(mesh)
    edge_lengths = np.linalg.norm(
        np.asarray(mesh.coordinates)[mesh_edges[:, 1]]
        - np.asarray(mesh.coordinates)[mesh_edges[:, 0]],
        axis=1,
    )
    achieved = [
        ("maximum_edge_length", float(np.max(edge_lengths))),
        ("maximum_deviation", 0.0),
        ("minimum_angle_degrees", math.degrees(minimum_angle)),
    ]
    if specification.maximum_deviation is not None:
        requested.append(("maximum_deviation", specification.maximum_deviation))
    if specification.quality_target is not None:
        requested.append(
            (
                "quality_target_degrees",
                math.degrees(specification.quality_target.minimum_angle),
            )
        )
        achieved.append(("quality_target_degrees", math.degrees(minimum_angle)))
    for feature in specification.protected_features:
        requested.append(
            (f"feature:{feature.feature_id}:maximum_deviation", feature.maximum_deviation)
        )
        achieved.append((f"feature:{feature.feature_id}:maximum_deviation", 0.0))
    compliance = MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(requested),
        achieved=tuple(achieved),
    )
    scoped = tuple(
        (query, witness, feature.maximum_deviation)
        for feature in specification.protected_features
        if feature.hard
    )
    fidelity_tolerance = owner._surface_fidelity_tolerance(domain, specification)
    if fidelity_tolerance is None:
        fidelity_tolerance = 0.0
    certification = NativeCertificationRequest(
        MeshCertificationSchedule("surface"),
        domain.source_id,
        domain.source_revision,
        fidelity_source=query,
        fidelity_tolerance=fidelity_tolerance,
        junction_vertices=junctions,
        scoped_fidelity=scoped,
        protected_features=specification.protected_features,
        fidelity_witness=witness,
        feature_source=domain,
    )
    return NativeMeshConstruction(
        mesh,
        prepared.coordinate_contract,
        compliance,
        certification,
        geometry=geometry,
        organization=organization,
        source_mesh=atlas.mesh,
        source_geometry=atlas.geometry,
    )


def _associations(
    domain: MeshingDomain,
    mesh: CellMesh,
    patch: int,
    uses: list[_TrimUse],
    point_uv: list[_ChartPoint],
    charts: list[tuple[_ChartPoint, ...]],
    /,
) -> tuple[tuple[GeometryAssociation, ...], np.ndarray]:
    from . import _native_surface as owner
    from ._native_surface_association import simplex_rows

    kind = (
        GeometryAssociationKind.BREP
        if domain.source_kinds == ("vertex", "edge", "face")
        else GeometryAssociationKind.SURFACE
    )
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    patch_rows = np.full(cell_ids.size, patch, dtype=np.int64)
    face_sources, face_dims, face_indices, face_paths, face_roles = _source_metadata(
        domain, np.full(cell_ids.size, 2, dtype=np.int8), patch_rows
    )
    face_charts = np.asarray(
        [
            [float(sum(point[axis] for point in row) / 4) for axis in range(2)]
            for row in charts
        ]
    )
    faces = GeometryAssociation(
        kind,
        domain.source_id,
        domain.source_revision,
        mesh.entity_set(2).entity_set_id,
        cell_ids,
        face_sources,
        np.zeros(cell_ids.size),
        exact=False,
        source_dimensions=face_dims,
        source_roles=face_roles,
        source_indices=face_indices,
        source_occurrence_paths=face_paths,
        source_parameter_intervals=(None,) * cell_ids.size,
        parameters=face_charts,
        orientations=np.ones(cell_ids.size, dtype=np.int8),
    )
    vertex_dimensions = np.full(len(point_uv), 2, dtype=np.int8)
    vertex_indices = np.full(len(point_uv), patch, dtype=np.int64)
    edge_records = []
    junctions = {}
    for use, (first, last) in uses:
        axis = 0 if first[0] != last[0] else 1
        other = 1 - axis
        rows = sorted(
            ((point[axis] - first[axis]) / (last[axis] - first[axis]), row)
            for row, point in enumerate(point_uv)
            if point[other] == first[other]
            and min(first[axis], last[axis])
            <= point[axis]
            <= max(first[axis], last[axis])
        )
        if not rows or rows[0][0] != 0 or rows[-1][0] != 1:
            raise owner._failure("source trim schedule is incomplete")
        for parameter, row in rows:
            if parameter == 0 or parameter == 1:
                corner = (
                    domain.curves[use.curve].start
                    if parameter == 0
                    else domain.curves[use.curve].end
                )
                previous = junctions.setdefault(row, corner)
                if previous != corner:
                    raise owner._failure("source trim corners disagree")
                vertex_dimensions[row], vertex_indices[row] = 0, corner
            elif vertex_dimensions[row] != 0:
                vertex_dimensions[row], vertex_indices[row] = 1, use.curve
        for (a, first_row), (b, last_row) in zip(rows, rows[1:]):
            edge_records.append((first_row, last_row, use.curve, a, b))
    canonical = _surface_edges(mesh)
    edge_rows = simplex_rows(
        canonical, np.asarray([(a, b) for a, b, _, _, _ in edge_records], dtype=np.int64)
    )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)[edge_rows]
    order = np.argsort(edge_ids, kind="stable")
    edge_ids = edge_ids[order]
    curves = np.asarray([edge_records[index][2] for index in order], dtype=np.int64)
    edge_sources, edge_dims, edge_indices, edge_paths, edge_roles = _source_metadata(
        domain, np.full(edge_ids.size, 1, dtype=np.int8), curves
    )
    directions = np.asarray(
        [
            1
            if tuple(canonical[edge_rows[index]])
            == (edge_records[index][0], edge_records[index][1])
            else -1
            for index in order
        ],
        dtype=np.int8,
    )
    if kind is GeometryAssociationKind.BREP:
        senses = {use.curve: 1 if use.last > use.first else -1 for use, _ in uses}
        directions *= np.asarray([senses[int(curve)] for curve in curves], dtype=np.int8)
    edges = GeometryAssociation(
        GeometryAssociationKind.CURVE,
        domain.source_id,
        domain.source_revision,
        mesh.entity_set(1).entity_set_id,
        edge_ids,
        edge_sources,
        np.zeros(edge_ids.size),
        exact=False,
        source_dimensions=edge_dims,
        source_roles=edge_roles,
        source_indices=edge_indices,
        source_occurrence_paths=edge_paths,
        source_parameter_intervals=tuple(
            (edge_records[index][3], edge_records[index][4]) for index in order
        ),
        orientations=directions,
    )
    vertex_ids = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
    vertex_sources, vertex_dims, vertex_indices_, vertex_paths, vertex_roles = (
        _source_metadata(domain, vertex_dimensions, vertex_indices)
    )
    vertices = GeometryAssociation(
        kind,
        domain.source_id,
        domain.source_revision,
        mesh.entity_set(0).entity_set_id,
        vertex_ids,
        vertex_sources,
        np.zeros(vertex_ids.size),
        exact=False,
        source_dimensions=vertex_dims,
        source_roles=vertex_roles,
        source_indices=vertex_indices_,
        source_occurrence_paths=vertex_paths,
        source_parameter_intervals=(None,) * vertex_ids.size,
        parameters=np.asarray(point_uv, dtype=np.float64),
    )
    return (faces, edges, vertices), vertex_ids[
        np.asarray(sorted(junctions), dtype=np.int64)
    ]
