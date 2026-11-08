#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact constrained-Delaunay route for planar regions and embedded features."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import numpy as np

import phydrax.discretization as cell_mesh_owner
import phydrax.meshing._association as association_owner
from phydrax.geometry import ConstrainedDelaunayTriangulation
from phydrax.geometry._curve_boundary import (
    ParametricCurveBoundarySource,
    SegmentChartCurve,
)
from phydrax.geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from phydrax.meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
)
from phydrax.meshing._result import MeshingComplianceReport
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl

from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._reference_cell import reference_cell_topology
from ...typing import checked
from .._certification_schedule import MeshCertificationSchedule
from .._controls import FeatureKind
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._organization import MeshLabel, MeshPatch
from .._plc_planar_support import planar_region_domain
from .._scope import MeshingEntityKind, MeshingScope
from . import _native_sources as sources_owner
from ._native_options import NativeMeshingOptions
from ._native_planar_domain import (
    constrained_edges,
    PreparedPlanarDomain,
    scoped_triangulation,
    size_control_issues,
    source_edge_parameters,
    source_entity_arrays,
)
from ._native_surface_association import _surface_edges


def planar_support_issues(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    /,
) -> list[str]:
    """Return source, target, and physical-control requests outside this route."""
    unsupported: list[str] = []
    target = specification.target
    family = target.cell_families
    if (
        target.topological_dimension != 2
        or target.ambient_dimension != 2
        or target.geometry_order != 1
        or set((*family.required, *family.preferred)) != {"triangle"}
        or family.allow_mixed
        or family.allowed_transitions
    ):
        unsupported.append(
            "a conforming affine triangular target in ambient dimension two"
        )
    if specification.background_metric is not None:
        unsupported.append("background metrics outside the isotropic planar size field")
    scope = specification.scope
    region_scope_ids = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
    if (
        scope.source_id != source.source_id
        or scope.source_revision != source.source_revision
        or scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension != 2
        or scope.entity_set_id != source.scope_entity_set_id(2)
        or not np.array_equal(np.asarray(scope.entity_ids), region_scope_ids)
    ):
        unsupported.append(
            "the source-owned planar region scope for the current source revision"
        )
    vertex_entity_ids, edge_entity_ids = source_entity_arrays(source)
    unsupported.extend(
        size_control_issues(
            specification,
            edge_entity_ids,
            vertex_entity_ids,
            region_scope_ids,
            source.scope_entity_set_id(2),
            source.scope_entity_set_id(1),
            source.scope_entity_set_id(0),
        )
    )
    return list(dict.fromkeys(unsupported))


def _failure(message: str, /, *, entity_ids: tuple[int, ...] = ()) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
        message,
        provider_code="native_planar_unresolved",
        stage="surface_meshing",
        entity_ids=entity_ids,
    )


def _validate_constraint_chains(
    prepared: PreparedPlanarDomain,
    triangulation: ConstrainedDelaunayTriangulation,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    /,
) -> None:
    """Require each recovered mesh-edge chain to cover its original source edge."""
    point_sources, _, exact_parameters = source_edge_parameters(prepared, triangulation)
    spans: list[list[tuple[Fraction, Fraction]]] = [
        [] for _ in range(prepared.source_edges.shape[0])
    ]

    def parameter(vertex: int, edge: int, /) -> Fraction:
        if point_sources[vertex] == edge:
            return exact_parameters[vertex]
        first, last = prepared.source_edges[edge]
        if vertex == first:
            return Fraction(0)
        if vertex == last:
            return Fraction(1)
        raise ValueError(
            f"Native CDT edge endpoint {vertex} lacks ancestry on source edge {edge}."
        )

    for pair, edge_value in zip(
        constrained.tolist(), constrained_sources.tolist(), strict=True
    ):
        edge = int(edge_value)
        if not 0 <= edge < len(spans):
            raise ValueError(f"Native CDT constraint names absent source edge {edge}.")
        first, last = (parameter(int(vertex), edge) for vertex in pair)
        if first == last:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                f"native CDT recovered a zero-length piece of source edge {edge}",
                provider_code="native_planar",
                stage="surface_meshing",
                entity_ids=(int(prepared.source_edge_ids[edge]),),
            )
        spans[edge].append((min(first, last), max(first, last)))
    for edge, pieces in enumerate(spans):
        pieces.sort()
        if (
            not pieces
            or pieces[0][0] != 0
            or pieces[-1][1] != 1
            or any(
                first[1] != second[0] for first, second in zip(pieces[:-1], pieces[1:])
            )
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                f"native CDT did not recover the complete source edge {edge} without gaps or overlaps",
                provider_code="native_planar",
                stage="surface_meshing",
                entity_ids=(int(prepared.source_edge_ids[edge]),),
            )


def _unique_edges(triangles: np.ndarray, /) -> np.ndarray:
    """Return lexicographically unique undirected triangle edges."""
    sides = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    return np.unique(np.sort(sides, axis=1), axis=0)


def _refine(
    prepared: PreparedPlanarDomain,
    specification: SurfaceMeshingSpec,
    /,
) -> ConstrainedDelaunayTriangulation:
    """Refine free edges until the declared physical size field is satisfied."""
    quality = specification.quality_target
    minimum_angle = (
        0.0
        if quality is None or not quality.hard
        else math.degrees(quality.minimum_angle)
    )
    extra = np.empty((0, prepared.points.shape[1]), dtype=np.float64)
    while True:
        inputs = (
            np.concatenate((prepared.points, extra), axis=0)
            if extra.size
            else prepared.points
        )
        triangulation = scoped_triangulation(
            inputs,
            prepared.segments,
            holes=prepared.holes_for_cdt(),
            min_angle=minimum_angle,
        )
        if str(triangulation.evidence.status) != "ok":
            raise _failure(
                f"native CDT returned {triangulation.evidence.status!s}; requested planar size field remains unresolved"
            )
        triangles = np.asarray(triangulation.triangles, dtype=np.int64)
        if not triangles.size:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "native CDT produced no triangles for the declared planar region",
                provider_code="native_planar",
                stage="surface_meshing",
            )

        points = np.asarray(triangulation.points, dtype=np.float64)
        edges = _unique_edges(triangles)
        constrained, _ = constrained_edges(
            triangles,
            np.asarray(triangulation.segment_ids, dtype=np.int64),
            prepared.segment_sources,
        )
        constraint_set = {tuple(pair) for pair in constrained.tolist()}
        free = np.fromiter(
            (tuple(pair) not in constraint_set for pair in edges.tolist()),
            dtype=np.bool_,
            count=edges.shape[0],
        )
        endpoints = points[edges]
        lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
        middle = 0.5 * (endpoints[:, 0] + endpoints[:, 1])
        field_samples = prepared.sizes(
            np.concatenate((endpoints[:, 0], endpoints[:, 1], middle))
        )
        local = np.min(field_samples.reshape((3, edges.shape[0])), axis=0)
        long = free & (lengths > local)
        if not np.any(long):
            return triangulation
        selected = edges[long]
        midpoints = 0.5 * (points[selected[:, 0]] + points[selected[:, 1]])
        if np.any(
            np.all(midpoints == points[selected[:, 0]], axis=1)
            | np.all(midpoints == points[selected[:, 1]], axis=1)
        ):
            raise _failure(
                "planar refinement reached coordinate representability before meeting the physical size field"
            )
        extra = np.concatenate((extra, midpoints), axis=0)


def _source_deviations(
    prepared: PreparedPlanarDomain,
    triangulation: ConstrainedDelaunayTriangulation,
    used: np.ndarray,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[Fraction, ...],
]:
    """Produce exact source-entity ancestry and physical carrier deviations."""
    source_edges, source_parameters, exact_parameters = source_edge_parameters(
        prepared, triangulation
    )
    source_points = prepared.source_points
    all_points = np.asarray(triangulation.points, dtype=np.float64)
    source_vertex_count = source_points.shape[0]
    dimensions = np.full(all_points.shape[0], 2, dtype=np.int8)
    indices = np.zeros(all_points.shape[0], dtype=np.int64)
    parameters = np.full((all_points.shape[0], 2), np.nan, dtype=np.float64)
    dimensions[:source_vertex_count] = 0
    indices[:source_vertex_count] = np.arange(source_vertex_count, dtype=np.int64)
    split_input = np.flatnonzero(prepared.point_sources >= 0)
    dimensions[split_input] = 1
    indices[split_input] = prepared.point_sources[split_input]
    parameters[split_input, 0] = prepared.point_parameters[split_input]
    edge_rows = np.flatnonzero(source_edges >= 0)
    dimensions[edge_rows] = 1
    indices[edge_rows] = source_edges[edge_rows]
    parameters[edge_rows, 0] = source_parameters[edge_rows]
    edge_deviations = np.zeros((prepared.source_edges.shape[0],), dtype=np.float64)
    point_deviations = np.zeros((all_points.shape[0],), dtype=np.float64)
    for vertex in edge_rows.tolist():
        edge = int(source_edges[vertex])
        first, last = prepared.source_edges[edge]
        start = tuple(Fraction(float(value)) for value in source_points[first])
        end = tuple(Fraction(float(value)) for value in source_points[last])
        parameter = exact_parameters[vertex]
        exact = tuple(a + parameter * (b - a) for a, b in zip(start, end, strict=True))
        squared = sum(
            (Fraction(float(value)) - expected) ** 2
            for value, expected in zip(all_points[vertex], exact, strict=True)
        )
        deviation = (
            0.0
            if squared == 0
            else float(np.nextafter(math.sqrt(float(squared)), math.inf))
        )
        point_deviations[vertex] = deviation
        edge_deviations[edge] = max(edge_deviations[edge], deviation)
    return (
        dimensions[used],
        indices[used],
        parameters[used],
        edge_deviations,
        point_deviations[used],
        tuple(exact_parameters[int(row)] for row in used.tolist()),
    )


def _planar_curve_source_query(
    prepared: PreparedPlanarDomain,
    selected_entity_ids: np.ndarray | None = None,
    /,
) -> ParametricCurveBoundarySource:
    """Bind complete authored polygon and embedded edges to their exact charts."""
    source = prepared.source
    edge_ids = np.asarray(prepared.source_edge_ids, dtype=np.int64)
    selected_rows = (
        np.arange(edge_ids.size, dtype=np.int64)
        if selected_entity_ids is None
        else np.flatnonzero(
            np.isin(edge_ids, np.asarray(selected_entity_ids, dtype=np.int64))
        )
    )
    if selected_rows.size == 0:
        raise ValueError("A planar curve query requires selected authored source edges.")
    edge_rows = np.asarray(prepared.source_edges, dtype=np.int64)[selected_rows]
    if edge_rows.shape != (selected_rows.size, 2):
        raise ValueError("Planar source edge identities must retain two endpoint rows.")
    points = np.asarray(prepared.source_points, dtype=np.float64)
    vertex_ids = np.asarray(prepared.source_vertex_ids, dtype=np.int64)
    source_edge_ids = edge_ids[selected_rows]
    keys = tuple(
        source.source_entity_key_for_scope(1, int(identifier))
        for identifier in source_edge_ids.tolist()
    )
    endpoint_keys: list[
        tuple[tuple[str, tuple[str, ...]], tuple[str, tuple[str, ...]]]
    ] = []
    endpoint_bounds: list[
        tuple[
            tuple[tuple[float, ...], tuple[float, ...]],
            tuple[tuple[float, ...], tuple[float, ...]],
        ]
    ] = []
    curves = []
    chart_ids = []
    for edge_row, (entity_id, occurrence) in zip(
        range(edge_rows.shape[0]), keys, strict=True
    ):
        first, last = int(edge_rows[edge_row, 0]), int(edge_rows[edge_row, 1])
        curves.append(SegmentChartCurve(points[first], points[last], 0.0, 1.0))
        endpoint_keys.append(
            (
                source.source_entity_key_for_scope(0, int(vertex_ids[first])),
                source.source_entity_key_for_scope(0, int(vertex_ids[last])),
            )
        )
        first_point = tuple(float(value) for value in points[first])
        last_point = tuple(float(value) for value in points[last])
        endpoint_bounds.append(((first_point, first_point), (last_point, last_point)))
        chart_ids.append(entity_id)
    ranges = tuple((0.0, 1.0) for _ in curves)
    return ParametricCurveBoundarySource(
        tuple(curves),
        ranges,
        source_id=source.source_id,
        source_revision=source.source_revision,
        curve_entity_ids=tuple(entity_id for entity_id, _ in keys),
        occurrence_paths=tuple(occurrence for _, occurrence in keys),
        chart_ids=tuple(chart_ids),
        endpoint_source_keys=tuple(endpoint_keys),
        endpoint_point_bounds=tuple(endpoint_bounds),
    )


def _compliance(
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    vertices: np.ndarray,
    triangles: np.ndarray,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    source_deviations: np.ndarray,
    source_dimensions: np.ndarray,
    source_indices: np.ndarray,
    minimum_angle_degrees: float,
    /,
) -> MeshingComplianceReport:
    """Measure physical edge, growth, and protected-source contracts."""
    edge_pairs = _unique_edges(triangles)
    edge_lengths = np.linalg.norm(
        vertices[edge_pairs[:, 1]] - vertices[edge_pairs[:, 0]], axis=1
    )
    pair_rows = {tuple(pair): row for row, pair in enumerate(edge_pairs.tolist())}
    constrained_rows = np.asarray(
        [pair_rows[tuple(pair)] for pair in constrained.tolist()], dtype=np.int64
    )
    vertex_sources = np.where(source_dimensions == 1, source_indices, -1)
    constrained_lookup = {
        tuple(pair): int(source)
        for pair, source in zip(
            constrained.tolist(), constrained_sources.tolist(), strict=True
        )
    }
    edge_sources = np.asarray(
        [constrained_lookup.get(tuple(pair), -1) for pair in edge_pairs.tolist()],
        dtype=np.int64,
    )
    first, last = vertices[edge_pairs[:, 0]], vertices[edge_pairs[:, 1]]
    middle = 0.5 * (first + last)
    samples = np.concatenate((first, last, middle), axis=0)
    sample_sources = np.concatenate(
        (
            vertex_sources[edge_pairs[:, 0]],
            vertex_sources[edge_pairs[:, 1]],
            edge_sources,
        )
    )
    local_sizes = prepared.sizes(samples, point_sources=sample_sources).reshape(
        (3, edge_pairs.shape[0])
    )
    local = np.min(local_sizes, axis=0)
    vertex_sizes = prepared.sizes(vertices, point_sources=vertex_sources)
    growth = 1.0 + float(
        np.max(
            np.abs(vertex_sizes[edge_pairs[:, 1]] - vertex_sizes[edge_pairs[:, 0]])
            / edge_lengths,
            initial=0.0,
        )
    )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    policy = specification.size_compliance
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        key = f"size:{control.control_id}"
        if control.scope.entity_dimension == 1:
            selected = np.isin(
                prepared.source_edge_ids[constrained_sources],
                np.asarray(control.scope.entity_ids, dtype=np.int64),
            )
            rows = constrained_rows[selected]
        else:
            rows = np.arange(edge_pairs.shape[0], dtype=np.int64)
        lengths = edge_lengths[rows]
        requested.append((f"{key}:target_size", control.target_size))
        if control.minimum_size is not None:
            requested.append((f"{key}:minimum_size", control.minimum_size))
        if control.maximum_size is not None:
            requested.append((f"{key}:maximum_size", control.maximum_size))
        if control.maximum_growth_rate is not None:
            requested.append((f"{key}:maximum_growth_rate", control.maximum_growth_rate))
        if not lengths.size:
            issues.append(f"missing_controlled_edges:{control.control_id}")
            continue
        statistics = {
            name: float(np.quantile(lengths, 0.5 if name == "p50" else 0.95))
            for name in policy.target_statistics
        }
        achieved.extend(
            (
                (f"{key}:minimum_edge", float(np.min(lengths))),
                (f"{key}:maximum_edge", float(np.max(lengths))),
                (
                    f"{key}:maximum_edge_to_local_size",
                    float(np.max(lengths / local[rows])),
                ),
                (f"{key}:size_field_growth_rate", growth),
            )
        )
        achieved.extend(
            (f"{key}:{name}_edge", value) for name, value in statistics.items()
        )
        tolerance = (
            policy.absolute_tolerance + policy.relative_tolerance * control.target_size
        )
        if control.strength is SizeControlStrength.HARD:
            issues.extend(
                f"target_size_{name}:{control.control_id}"
                for name, value in statistics.items()
                if abs(value - control.target_size) > tolerance
            )
            if (
                control.minimum_size is not None
                and float(np.min(lengths)) < control.minimum_size - tolerance
            ):
                issues.append(f"minimum_size:{control.control_id}")
            if (
                control.maximum_size is not None
                and float(np.max(lengths)) > control.maximum_size + tolerance
            ):
                issues.append(f"maximum_size:{control.control_id}")
            if np.any(
                lengths
                > local[rows]
                + policy.absolute_tolerance
                + policy.relative_tolerance * local[rows]
            ):
                issues.append(f"local_size:{control.control_id}")
            if (
                control.maximum_growth_rate is not None
                and growth
                > control.maximum_growth_rate
                + policy.absolute_tolerance
                + policy.relative_tolerance * control.maximum_growth_rate
            ):
                issues.append(f"maximum_growth_rate:{control.control_id}")
    for feature in specification.protected_features:
        key = f"protected:{feature.feature_id}:maximum_deviation"
        if feature.feature_kind is FeatureKind.CURVE:
            rows = np.flatnonzero(
                np.isin(
                    prepared.source_edge_ids,
                    np.asarray(feature.scope.entity_ids, dtype=np.int64),
                )
            )
            achieved_deviation = float(np.max(source_deviations[rows], initial=0.0))
            if not rows.size:
                issues.append(f"missing_protected_edges:{feature.feature_id}")
        else:
            achieved_deviation = 0.0
        requested.append((key, feature.maximum_deviation))
        achieved.append((key, achieved_deviation))
        if feature.hard and achieved_deviation > feature.maximum_deviation:
            issues.append(key)
    if specification.maximum_deviation is not None:
        achieved_deviation = float(np.max(source_deviations, initial=0.0))
        requested.append(("maximum_deviation", specification.maximum_deviation))
        achieved.append(("maximum_deviation", achieved_deviation))
        if achieved_deviation > specification.maximum_deviation:
            issues.append("maximum_deviation")
    if specification.maximum_normal_deviation is not None:
        requested.append(
            ("maximum_normal_deviation", specification.maximum_normal_deviation)
        )
        achieved.append(("maximum_normal_deviation", 0.0))
    if specification.quality_target is not None:
        target = specification.quality_target
        achieved_angle = math.radians(minimum_angle_degrees)
        requested.append(("minimum_angle", target.minimum_angle))
        achieved.append(("minimum_angle", achieved_angle))
        if target.hard and achieved_angle < target.minimum_angle:
            issues.append(f"minimum_angle:{target.target_id}")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _mesh_scope(
    mesh: cell_mesh_owner.CellMesh, dimension: int, ids: np.ndarray, /
) -> MeshingScope:
    entity_set = mesh.entity_set(dimension)
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        ids,
    )


def _organization(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    mesh: cell_mesh_owner.CellMesh,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    used: np.ndarray,
    source_dimensions: np.ndarray,
    source_indices: np.ndarray,
    parameters: np.ndarray,
    exact_parameters: tuple[Fraction, ...],
    point_deviations: np.ndarray,
    /,
) -> NativeMeshOrganization:
    """Preserve source-region, boundary-loop, feature, and exact edge ancestry."""
    edge_set = mesh.entity_set(1)
    mesh_edges = np.sort(_surface_edges(mesh), axis=1)
    mesh_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    used_rows = {int(vertex): row for row, vertex in enumerate(used.tolist())}
    compact_constrained = np.asarray(
        [[used_rows[int(vertex)] for vertex in pair] for pair in constrained],
        dtype=np.int64,
    )
    edge_lookup = {tuple(pair): row for row, pair in enumerate(mesh_edges.tolist())}
    constrained_rows = np.asarray(
        [edge_lookup[tuple(pair)] for pair in compact_constrained.tolist()],
        dtype=np.int64,
    )
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[constrained_rows]
    order = np.argsort(edge_ids, kind="stable")
    constrained_rows, edge_ids = constrained_rows[order], edge_ids[order]
    constrained = constrained[order]
    compact_constrained = compact_constrained[order]
    constrained_sources = constrained_sources[order]
    loops = prepared.source_loops[constrained_sources]
    patches = tuple(
        MeshPatch(
            f"loop:{loop}",
            _mesh_scope(mesh, 1, edge_ids[loops == loop]),
        )
        for loop in np.unique(loops[loops >= 0]).tolist()
    )
    labels: list[MeshLabel] = []
    if np.any(loops < 0):
        labels.append(MeshLabel("embedded", _mesh_scope(mesh, 1, edge_ids[loops < 0])))
    cell_set = mesh.entity_set(2)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    labels.append(MeshLabel("region:0", _mesh_scope(mesh, 2, cell_ids)))
    for feature in specification.protected_features:
        if feature.feature_kind is FeatureKind.CURVE:
            selected = np.isin(
                prepared.source_edge_ids[constrained_sources],
                np.asarray(feature.scope.entity_ids, dtype=np.int64),
            )
            labels.append(
                MeshLabel(
                    f"feature:{feature.feature_id}",
                    _mesh_scope(mesh, 1, edge_ids[selected]),
                )
            )
        elif feature.feature_kind is FeatureKind.CORNER:
            selected = (source_dimensions == 0) & np.isin(
                prepared.source_vertex_ids[source_indices],
                np.asarray(feature.scope.entity_ids, dtype=np.int64),
            )
            rows = np.flatnonzero(selected)
            vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[rows]
            labels.append(
                MeshLabel(
                    f"feature:{feature.feature_id}",
                    _mesh_scope(mesh, 0, vertex_ids),
                )
            )

    region_scope_id = int(source.scope_entity_ids(2)[0])
    region_source_key = source.source_entity_key_for_scope(2, region_scope_id)
    if region_source_key[0] != source.region_entity_id:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "the source region scope key disagrees with its canonical region identity",
            provider_code="native_planar_identity",
            stage="geometry_association",
            entity_ids=(region_scope_id,),
        )
    cell_association = association_owner.GeometryAssociation(
        association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        cell_set.entity_set_id,
        cell_ids,
        (source.region_entity_id,) * cell_ids.size,
        np.zeros((cell_ids.size,), dtype=np.float64),
        exact=True,
        source_roles=(association_owner.GeometrySourceEntityRole.REGION,) * cell_ids.size,
        source_occurrence_paths=(region_source_key[1],) * cell_ids.size,
    )

    def edge_parameter(vertex: int, edge: int, /) -> Fraction:
        row = used_rows[vertex]
        if source_dimensions[row] == 1 and source_indices[row] == edge:
            return exact_parameters[row]
        first, last = prepared.source_edges[edge]
        if source_dimensions[row] == 0 and source_indices[row] == first:
            return Fraction(0)
        if source_dimensions[row] == 0 and source_indices[row] == last:
            return Fraction(1)
        raise ValueError(
            f"Planar constraint endpoint {vertex} has no exact ancestry on source edge {edge}."
        )

    raw_edge_parameters = tuple(
        (
            edge_parameter(int(pair[0]), int(edge)),
            edge_parameter(int(pair[1]), int(edge)),
        )
        for pair, edge in zip(
            constrained.tolist(), constrained_sources.tolist(), strict=True
        )
    )
    edge_directions = (
        mesh_coordinates[compact_constrained[:, 1]]
        - mesh_coordinates[compact_constrained[:, 0]]
    )
    source_directions = (
        prepared.source_points[prepared.source_edges[constrained_sources, 1]]
        - prepared.source_points[prepared.source_edges[constrained_sources, 0]]
    )
    edge_orientations = np.sign(
        np.sum(edge_directions * source_directions, axis=1)
    ).astype(np.int8)
    edge_parameters = tuple(
        (min(first, last), max(first, last)) for first, last in raw_edge_parameters
    )
    edge_parameter_values = np.asarray(
        [[float(first), float(last)] for first, last in edge_parameters],
        dtype=np.float64,
    )
    edge_residuals = np.max(point_deviations[compact_constrained], axis=1)
    curve_edge_association = association_owner.GeometryAssociation(
        association_owner.GeometryAssociationKind.CURVE,
        source.source_id,
        source.source_revision,
        edge_set.entity_set_id,
        edge_ids,
        tuple(
            source.source_entity_key_for_scope(1, int(prepared.source_edge_ids[edge]))[0]
            for edge in constrained_sources.tolist()
        ),
        edge_residuals,
        exact=bool(np.all(edge_residuals == 0.0)),
        source_roles=(association_owner.GeometrySourceEntityRole.EDGE,) * edge_ids.size,
        parameters=edge_parameter_values,
        source_parameter_intervals=edge_parameters,
        orientations=edge_orientations,
        source_occurrence_paths=tuple(
            source.source_entity_key_for_scope(1, int(prepared.source_edge_ids[edge]))[1]
            for edge in constrained_sources.tolist()
        ),
    )
    all_edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)
    free_edge_ids = np.setdiff1d(all_edge_ids, edge_ids)
    associations = [cell_association, curve_edge_association]
    if free_edge_ids.size:
        associations.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                edge_set.entity_set_id,
                free_edge_ids,
                (source.region_entity_id,) * free_edge_ids.size,
                np.zeros((free_edge_ids.size,), dtype=np.float64),
                exact=True,
                source_roles=(association_owner.GeometrySourceEntityRole.REGION,)
                * free_edge_ids.size,
                source_occurrence_paths=(region_source_key[1],) * free_edge_ids.size,
            )
        )
    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)

    def vertex_source_key(dimension: int, index: int, /) -> tuple[str, tuple[str, ...]]:
        if dimension == 0:
            return source.source_entity_key_for_scope(
                0, int(prepared.source_vertex_ids[index])
            )
        if dimension == 2:
            return source.region_entity_id, region_source_key[1]
        raise ValueError(f"Unsupported non-curve planar source dimension {dimension}.")

    edge_vertices = np.flatnonzero(source_dimensions == 1)
    if edge_vertices.size:
        edge_point_keys = tuple(
            source.source_entity_key_for_scope(
                1, int(prepared.source_edge_ids[source_indices[row]])
            )
            for row in edge_vertices.tolist()
        )
        edge_point_sources = tuple(key[0] for key in edge_point_keys)
        associations.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.CURVE,
                source.source_id,
                source.source_revision,
                vertex_set.entity_set_id,
                vertex_ids[edge_vertices],
                edge_point_sources,
                point_deviations[edge_vertices],
                source_occurrence_paths=tuple(key[1] for key in edge_point_keys),
                exact=bool(np.all(point_deviations[edge_vertices] == 0.0)),
                source_roles=(association_owner.GeometrySourceEntityRole.EDGE,)
                * edge_vertices.size,
                parameters=parameters[edge_vertices],
                source_parameter_points=tuple(
                    exact_parameters[row] for row in edge_vertices.tolist()
                ),
            )
        )
    noncurve_vertices = np.flatnonzero(source_dimensions != 1)
    if noncurve_vertices.size:
        dimensions = source_dimensions[noncurve_vertices]
        vertex_source_keys = tuple(
            vertex_source_key(int(dimension), int(index))
            for dimension, index in zip(
                dimensions.tolist(),
                source_indices[noncurve_vertices].tolist(),
                strict=True,
            )
        )
        associations.append(
            association_owner.GeometryAssociation(
                association_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                vertex_set.entity_set_id,
                vertex_ids[noncurve_vertices],
                tuple(key[0] for key in vertex_source_keys),
                point_deviations[noncurve_vertices],
                exact=bool(np.all(point_deviations[noncurve_vertices] == 0.0)),
                source_roles=tuple(
                    (
                        association_owner.GeometrySourceEntityRole.VERTEX,
                        association_owner.GeometrySourceEntityRole.REGION,
                    )[int(dimension) if int(dimension) == 0 else 1]
                    for dimension in dimensions.tolist()
                ),
                parameters=parameters[noncurve_vertices],
                source_occurrence_paths=tuple(key[1] for key in vertex_source_keys),
            )
        )
    return NativeMeshOrganization(
        patches=patches,
        labels=tuple(labels),
        associations=tuple(associations),
    )


def _planar_curve_chart_witness(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    query: ParametricCurveBoundarySource,
    prepared: PreparedPlanarDomain,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    used: np.ndarray,
    source_dimensions: np.ndarray,
    source_indices: np.ndarray,
    exact_parameters: tuple[Fraction, ...],
    requested_entity_ids: np.ndarray,
    /,
) -> SourceBoundaryChartWitness:
    selected_source_ids = set(int(value) for value in requested_entity_ids.tolist())
    tri_to_mesh = {int(identifier): row for row, identifier in enumerate(used.tolist())}
    target_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    target_edges = _surface_edges(mesh)
    edge_id_by_vertices = {
        tuple(sorted(map(int, pair))): int(identifier)
        for pair, identifier in zip(
            target_edges.tolist(), target_edge_ids.tolist(), strict=True
        )
    }
    source_row_by_target_edge: dict[int, int] = {}
    expected_target_edges: set[int] = set()
    constrained_rows = np.asarray(constrained, dtype=np.int64)
    constrained_source_rows = np.asarray(constrained_sources, dtype=np.int64)
    if (
        constrained_rows.ndim != 2
        or constrained_rows.shape[1] != 2
        or constrained_source_rows.shape != (constrained_rows.shape[0],)
    ):
        raise ValueError(
            "Planar source constraints require aligned edge endpoint and source rows."
        )
    for constraint_row in range(constrained_rows.shape[0]):
        first = int(constrained_rows[constraint_row, 0])
        second = int(constrained_rows[constraint_row, 1])
        source_row = int(constrained_source_rows[constraint_row])
        if int(prepared.source_edge_ids[source_row]) not in selected_source_ids:
            continue
        first_target = tri_to_mesh[first]
        second_target = tri_to_mesh[second]
        target_pair = (min(first_target, second_target), max(first_target, second_target))
        try:
            target_edge_id = edge_id_by_vertices[target_pair]
        except KeyError as error:
            raise ValueError(
                "Planar source constraint has no target mesh edge identity."
            ) from error
        previous = source_row_by_target_edge.get(target_edge_id)
        if previous is not None and previous != source_row:
            raise ValueError(
                "One target mesh edge has conflicting planar source ancestry."
            )
        source_row_by_target_edge[target_edge_id] = source_row
        expected_target_edges.add(target_edge_id)

    topology = reference_cell_topology("triangle")
    reference_vertices = np.asarray(topology.vertices, dtype=np.float64)
    local_facets = np.asarray(topology.entities[1], dtype=np.int64)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    triangles = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    witness_cell_ids: list[int] = []
    witness_edge_ids: list[int] = []
    witness_local_facets: list[int] = []
    target_references: list[np.ndarray] = []
    target_reference_rational_rows = []
    chart_ids: list[str] = []
    entity_ids: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    source_parameters: list[tuple[float, float]] = []
    source_parameter_rational_rows = []
    orientations: list[int] = []
    covered: set[int] = set()
    source_edges = np.asarray(prepared.source_edges, dtype=np.int64)

    def parameter(vertex_row: int, source_row: int, /) -> Fraction:
        if (
            source_dimensions[vertex_row] == 1
            and source_indices[vertex_row] == source_row
        ):
            return exact_parameters[vertex_row]
        first, last = int(source_edges[source_row, 0]), int(source_edges[source_row, 1])
        if source_dimensions[vertex_row] == 0:
            source_vertex = int(source_indices[vertex_row])
            if source_vertex == first:
                return Fraction(0)
            if source_vertex == last:
                return Fraction(1)
        raise ValueError(
            "Planar target edge endpoint lacks its authored source-chart parameter."
        )

    for cell_row in range(triangles.shape[0]):
        for local_facet in range(local_facets.shape[0]):
            first_local = int(local_facets[local_facet, 0])
            second_local = int(local_facets[local_facet, 1])
            first_target_row = int(triangles[cell_row, first_local])
            second_target_row = int(triangles[cell_row, second_local])
            target_pair = (
                min(first_target_row, second_target_row),
                max(first_target_row, second_target_row),
            )
            target_edge_id = edge_id_by_vertices[target_pair]
            source_row = source_row_by_target_edge.get(target_edge_id)
            if source_row is None or target_edge_id in covered:
                continue
            chart_key = prepared.source.source_entity_key_for_scope(
                1, int(prepared.source_edge_ids[source_row])
            )
            target_rows = (first_target_row, second_target_row)
            exact_parameters_for_edge = tuple(
                parameter(vertex_row, source_row) for vertex_row in target_rows
            )
            source_parameters.append(
                (float(exact_parameters_for_edge[0]), float(exact_parameters_for_edge[1]))
            )
            source_parameter_rational_rows.append(
                tuple((value,) for value in exact_parameters_for_edge)
            )
            witness_cell_ids.append(int(cell_ids[cell_row]))
            witness_edge_ids.append(target_edge_id)
            witness_local_facets.append(local_facet)
            target_reference_row = reference_vertices[
                np.asarray((first_local, second_local), dtype=np.int64)
            ]
            target_references.append(target_reference_row)
            target_reference_rational_rows.append(
                tuple(
                    tuple(Fraction.from_float(float(value)) for value in point)
                    for point in target_reference_row
                )
            )
            chart_ids.append(chart_key[0])
            entity_ids.append(chart_key[0])
            occurrence_paths.append(chart_key[1])
            orientations.append(
                1 if exact_parameters_for_edge[1] > exact_parameters_for_edge[0] else -1
            )
            covered.add(target_edge_id)
    if covered != expected_target_edges:
        raise ValueError(
            "Planar witness does not cover every target edge in the requested source chart."
        )
    if not witness_edge_ids:
        raise ValueError("Planar source chart witness cannot be empty.")
    return SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.query_id,
        chart_dimension=1,
        reference_dimension=2,
        cell_global_ids=np.asarray(witness_cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(witness_edge_ids, dtype=np.int64),
        local_entity_indices=np.asarray(witness_local_facets, dtype=np.int32),
        target_reference_vertices=np.asarray(target_references, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(
            target_reference_rational_rows
        ),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(entity_ids),
        occurrence_paths=tuple(occurrence_paths),
        source_parameters=np.asarray(source_parameters, dtype=np.float64)[:, :, None],
        source_parameter_rational_tokens=rational_chart_tokens(
            source_parameter_rational_rows
        ),
        orientations=np.asarray(orientations, dtype=np.int8),
    )


def _planar_point_witness(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    query: ParametricCurveBoundarySource,
    source: sources_owner.NativePlanarSource,
    prepared: PreparedPlanarDomain,
    requested_point_ids: np.ndarray,
    source_dimensions: np.ndarray,
    source_indices: np.ndarray,
    /,
) -> SourceBoundaryPointWitness:
    identifiers = np.asarray(requested_point_ids, dtype=np.int64)
    if (
        identifiers.ndim != 1
        or identifiers.size == 0
        or np.unique(identifiers).size != identifiers.size
    ):
        raise ValueError("Planar point witness needs unique authored source point IDs.")
    target_rows = []
    entity_ids = []
    occurrence_paths = []
    for identifier in identifiers.tolist():
        matches = np.flatnonzero(
            (np.asarray(source_dimensions, dtype=np.int64) == 0)
            & (
                np.asarray(prepared.source_vertex_ids, dtype=np.int64)[
                    np.asarray(source_indices, dtype=np.int64)
                ]
                == int(identifier)
            )
        )
        if matches.size != 1:
            raise ValueError(
                "Every protected planar point must map to one target vertex."
            )
        target_rows.append(int(matches[0]))
        entity_id, occurrence = source.source_entity_key_for_scope(0, int(identifier))
        entity_ids.append(entity_id)
        occurrence_paths.append(occurrence)
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
        np.asarray(target_rows, dtype=np.int64)
    ]
    if np.unique(target_ids).size != target_ids.size:
        raise ValueError(
            "Distinct planar source points cannot share one target vertex identity."
        )
    return SourceBoundaryPointWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.query_id,
        vertex_global_ids=target_ids,
        source_entity_ids=tuple(entity_ids),
        occurrence_paths=tuple(occurrence_paths),
    )


def _planar_junction_vertices(
    source: sources_owner.NativePlanarSource,
    prepared: PreparedPlanarDomain,
    mesh: cell_mesh_owner.CellMesh,
    source_dimensions: np.ndarray,
    source_indices: np.ndarray,
    /,
) -> np.ndarray:
    """Map authored vertices with at least three distinct source-edge incidents."""
    incident_edges: dict[
        tuple[str, tuple[str, ...]], set[tuple[str, tuple[str, ...]]]
    ] = {}
    edge_ids = np.asarray(prepared.source_edge_ids, dtype=np.int64)
    endpoints = np.asarray(prepared.source_edges, dtype=np.int64)
    if endpoints.shape != (edge_ids.size, 2):
        raise ValueError("Planar source edges must retain two endpoint rows.")
    for edge_row in range(edge_ids.size):
        edge_key = source.source_entity_key_for_scope(1, int(edge_ids[edge_row]))
        for endpoint_column in range(2):
            vertex_row = int(endpoints[edge_row, endpoint_column])
            vertex_id = int(prepared.source_vertex_ids[vertex_row])
            vertex_key = source.source_entity_key_for_scope(0, vertex_id)
            incident_edges.setdefault(vertex_key, set()).add(edge_key)
    junction_keys = {
        key for key, incident in incident_edges.items() if len(incident) >= 3
    }
    if not junction_keys:
        return np.empty((0,), dtype=np.int64)
    target_row_by_source_key: dict[tuple[str, tuple[str, ...]], int] = {}
    target_dimensions = np.asarray(source_dimensions, dtype=np.int64)
    target_indices = np.asarray(source_indices, dtype=np.int64)
    if target_dimensions.shape != target_indices.shape:
        raise ValueError("Planar source ancestry dimensions and indices must align.")
    for target_row in range(target_dimensions.size):
        dimension = int(target_dimensions[target_row])
        index = int(target_indices[target_row])
        if dimension != 0:
            continue
        vertex_id = int(prepared.source_vertex_ids[int(index)])
        key = source.source_entity_key_for_scope(0, vertex_id)
        if key in junction_keys:
            if key in target_row_by_source_key:
                raise ValueError(
                    "An authored planar junction has multiple target vertex rows."
                )
            target_row_by_source_key[key] = target_row
    if set(target_row_by_source_key) != junction_keys:
        raise ValueError(
            "An authored planar junction lacks a unique compact target vertex."
        )
    return np.asarray(
        sorted(
            int(mesh.vertex_global_ids[target_row_by_source_key[key]])
            for key in junction_keys
        ),
        dtype=np.int64,
    )


@final
class _PreparedPlanarRoute(StrictModule, NonTrainableState):
    source: sources_owner.NativePlanarSource
    specification: SurfaceMeshingSpec
    prepared: PreparedPlanarDomain
    coordinate_contract: SpatialCoordinateContract

    def __init__(
        self,
        source: sources_owner.NativePlanarSource,
        specification: SurfaceMeshingSpec,
        prepared: PreparedPlanarDomain,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.prepared = prepared
        self.coordinate_contract = coordinate_contract

    def execute(self, /) -> NativeMeshConstruction:
        source, specification, prepared = self.source, self.specification, self.prepared
        triangulation = _refine(prepared, specification)
        triangles = np.asarray(triangulation.triangles, dtype=np.int64)
        used, compact = np.unique(triangles, return_inverse=True)
        coordinates = np.asarray(triangulation.points, dtype=np.float64)[used]
        compact_triangles = compact.reshape(triangles.shape).astype(np.int32)
        cell_segments = np.asarray(triangulation.segment_ids, dtype=np.int64)
        constrained, constrained_sources = constrained_edges(
            triangles, cell_segments, prepared.segment_sources
        )
        missing = np.setdiff1d(
            np.arange(prepared.source_edges.shape[0]), constrained_sources
        )
        if missing.size:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "native CDT did not recover every authored planar constraint",
                provider_code="native_planar",
                stage="surface_meshing",
                entity_ids=tuple(
                    int(value) for value in prepared.source_edge_ids[missing].tolist()
                ),
            )
        _validate_constraint_chains(
            prepared, triangulation, constrained, constrained_sources
        )
        (
            dimensions,
            indices,
            parameters,
            source_deviations,
            point_deviations,
            exact_parameters,
        ) = _source_deviations(prepared, triangulation, used)
        used_rows = {int(vertex): row for row, vertex in enumerate(used.tolist())}
        compact_constrained = np.asarray(
            [[used_rows[int(vertex)] for vertex in pair] for pair in constrained],
            dtype=np.int64,
        )
        compliance = _compliance(
            specification,
            prepared,
            coordinates,
            compact_triangles,
            compact_constrained,
            constrained_sources,
            source_deviations,
            dimensions,
            indices,
            float(triangulation.evidence.minimum_angle_degrees),
        )
        excessive = np.flatnonzero(source_deviations > prepared.source_bounds)
        if excessive.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "planar source constraints exceed a hard physical deviation tolerance",
                provider_code="native_planar_fidelity",
                stage="geometry_association",
                entity_ids=tuple(
                    int(value) for value in prepared.source_edge_ids[excessive].tolist()
                ),
            )
        mesh = cell_mesh_owner.CellMesh.from_triangles(
            coordinates,
            compact_triangles,
            numeric_version=source.source_revision,
        )
        organization = _organization(
            source,
            specification,
            prepared,
            mesh,
            constrained,
            constrained_sources,
            used,
            dimensions,
            indices,
            parameters,
            exact_parameters,
            point_deviations,
        )
        geometry = CellGeometrySpec.affine(mesh)
        scoped_fidelity = []
        for feature in specification.protected_features:
            if not feature.hard:
                continue
            feature_ids = np.asarray(feature.scope.entity_ids, dtype=np.int64)
            if feature.feature_kind is FeatureKind.CURVE:
                feature_query = _planar_curve_source_query(prepared, feature_ids)
                feature_witness = _planar_curve_chart_witness(
                    mesh,
                    geometry,
                    feature_query,
                    prepared,
                    constrained,
                    constrained_sources,
                    used,
                    dimensions,
                    indices,
                    exact_parameters,
                    feature_ids,
                )
                scoped_fidelity.append(
                    (feature_query, feature_witness, feature.maximum_deviation)
                )
            elif feature.feature_kind is FeatureKind.CORNER:
                feature_query = _planar_curve_source_query(prepared)
                feature_witness = _planar_point_witness(
                    mesh,
                    geometry,
                    feature_query,
                    source,
                    prepared,
                    feature_ids,
                    dimensions,
                    indices,
                )
                scoped_fidelity.append(
                    (feature_query, feature_witness, feature.maximum_deviation)
                )
        junction_vertices = _planar_junction_vertices(
            source, prepared, mesh, dimensions, indices
        )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("planar"),
            source.source_id,
            source.source_revision,
            domain=planar_region_domain(source),
            cell_regions=np.zeros((compact_triangles.shape[0],), dtype=np.int64),
            protected_features=specification.protected_features,
            maximum_normal_deviation=specification.maximum_normal_deviation,
            scoped_fidelity=tuple(scoped_fidelity),
            feature_source=source,
            junction_vertices=junction_vertices,
        )
        return NativeMeshConstruction(
            mesh,
            self.coordinate_contract,
            compliance,
            certification,
            geometry=geometry,
            organization=organization,
            planar_embedding=specification.planar_embedding,
        )


@checked
def prepare_native_planar(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedPlanarRoute:
    """Prepare a source-bound planar construction for central acceptance."""
    del options
    unsupported = planar_support_issues(source, specification)
    if unsupported:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(unsupported),
            provider_code="native_planar_preflight",
            stage="source_inspection",
        )
    prepared = PreparedPlanarDomain(source, specification)
    return _PreparedPlanarRoute(source, specification, prepared, coordinate_contract)


__all__ = ["planar_support_issues", "prepare_native_planar"]
