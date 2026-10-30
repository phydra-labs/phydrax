#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-faithful planar constraints, size fields, and CDT evidence."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

import phydrax.geometry as triangulation_owner
from phydrax.meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
)
from phydrax.meshing._sizing import (
    resolve_size_controls,
    SizeCombinationPolicy,
    SizeFieldDomain,
    UniformSizeControl,
)

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.simplicial import PlanarMeshRegion, SegmentMesh
from ...typing import checked
from .._controls import FeatureKind
from .._scope import MeshingEntityKind
from . import _native_sources as sources_owner
from ._native_planar_construction import uniform_construction_size


@checked
def source_arrays(
    source: sources_owner.NativePlanarSource, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source vertices, constraints, and loop index per constraint."""
    if not isinstance(source.region, PlanarMeshRegion):
        raise TypeError("NativePlanarSource.region must be PlanarMeshRegion.")
    vertices = np.asarray(source.region.vertices, dtype=np.float64)
    edges = np.asarray(source.region.edges, dtype=np.int64)
    offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
    loops = np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))
    if source.embedded is not None:
        if not isinstance(source.embedded, SegmentMesh):
            raise TypeError("NativePlanarSource.embedded must be SegmentMesh or None.")
        embedded_vertices = np.asarray(source.embedded.vertices, dtype=np.float64)
        embedded_edges = np.asarray(source.embedded.edges, dtype=np.int64)
        edges = np.concatenate((edges, embedded_edges + vertices.shape[0]), axis=0)
        loops = np.concatenate((loops, np.full(embedded_edges.shape[0], -1, dtype=np.int64)))
        vertices = np.concatenate((vertices, embedded_vertices), axis=0)
    return vertices, edges, loops


def source_entity_arrays(
    source: sources_owner.NativePlanarSource, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable vertex and edge IDs aligned with the combined source banks."""
    vertices = np.asarray(source.vertex_entity_ids, dtype=np.int64)
    edges = np.asarray(source.edge_entity_ids, dtype=np.int64)
    if source.embedded is not None:
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(source.embedded_vertex_entity_ids, dtype=np.int64),
            )
        )
        edges = np.concatenate(
            (
                edges,
                np.asarray(source.embedded_edge_entity_ids, dtype=np.int64),
            )
        )
    return vertices, edges




def size_control_issues(
    specification: SurfaceMeshingSpec,
    source_edge_ids: np.ndarray,
    source_vertex_ids: np.ndarray,
    region_scope_entity_ids: np.ndarray,
    region_scope_entity_set_id: str,
    source_edge_entity_set_id: str,
    source_vertex_entity_set_id: str,
    /,
) -> list[str]:
    """Describe planar controls not bound to stable source entity identities."""
    controls = specification.size_controls
    issues: list[str] = []
    if not controls:
        return ["an explicit physical resolution request for the planar source region"]
    uniform_controls = tuple(
        control for control in controls if isinstance(control, UniformSizeControl)
    )
    if len(uniform_controls) != len(controls):
        return ["uniform size controls on planar regions and source edges"]
    scope = specification.scope
    for control in uniform_controls:
        control_scope = control.scope
        if (control_scope.source_id, control_scope.source_revision) != (
            scope.source_id,
            scope.source_revision,
        ):
            issues.append("size-control scopes bound to the planar source revision")
            continue
        if control_scope.entity_kind is not MeshingEntityKind.GEOMETRY:
            issues.append("size controls on planar geometry entities")
            continue
        identifiers = np.asarray(control_scope.entity_ids, dtype=np.int64)
        if control_scope.entity_dimension == 2:
            if (
                control_scope.entity_set_id != region_scope_entity_set_id
                or not np.array_equal(identifiers, region_scope_entity_ids)
            ):
                issues.append("the source-owned planar region scope")
        elif (
            control_scope.entity_dimension != 1
            or control_scope.entity_set_id != source_edge_entity_set_id
            or not np.all(np.isin(identifiers, source_edge_ids))
        ):
            issues.append("size controls scoped to the source region or existing edges")
    if not any(control.scope.entity_dimension == 2 for control in uniform_controls):
        issues.append("a region-scoped physical resolution control for the source-owned planar region")
    for feature in specification.protected_features:
        feature_scope = feature.scope
        if (feature_scope.source_id, feature_scope.source_revision) != (
            scope.source_id,
            scope.source_revision,
        ):
            issues.append("protected features bound to the planar source revision")
        elif feature_scope.entity_kind is not MeshingEntityKind.GEOMETRY:
            issues.append("protected features on planar geometry entities")
        elif feature.feature_kind is FeatureKind.CURVE:
            if (
                feature_scope.entity_dimension != 1
                or feature_scope.entity_set_id != source_edge_entity_set_id
                or not np.all(
                    np.isin(
                        np.asarray(feature.scope.entity_ids, dtype=np.int64),
                        source_edge_ids,
                    )
                )
            ):
                issues.append("protected planar curves on existing source edges")
        elif feature.feature_kind is FeatureKind.CORNER:
            if (
                feature_scope.entity_dimension != 0
                or feature_scope.entity_set_id != source_vertex_entity_set_id
                or not np.all(
                    np.isin(
                        np.asarray(feature.scope.entity_ids, dtype=np.int64),
                        source_vertex_ids,
                    )
                )
            ):
                issues.append("protected planar corners on existing source vertices")
        else:
            issues.append(f"{feature.feature_kind.value} protected features")
    if specification.region_controls:
        issues.append("region controls not represented by the single-region planar source")
    if specification.patch_controls:
        issues.append("patch/interface controls not represented by the planar source")
    if specification.periodic_constraints:
        issues.append("periodic constraints not represented by this planar source")
    if specification.layer_controls:
        issues.append("boundary-layer controls not represented by this planar route")
    return issues


def scoped_triangulation(
    points: np.ndarray,
    segments: np.ndarray,
    /,
    *,
    holes: np.ndarray | None = None,
    min_angle: float = 0.0,
    max_area: float = math.inf,
) -> triangulation_owner.ConstrainedDelaunayTriangulation:
    """Build the exact constrained triangulation without resource ceilings."""
    return triangulation_owner.ConstrainedDelaunayTriangulation(
        points,
        segments,
        holes=holes,
        keep_convex_hull=False,
        min_angle=min_angle,
        max_area=max_area,
    )


def interior_point(points: np.ndarray, loop: np.ndarray, /) -> np.ndarray:
    """Choose an interior point from an exact CDT of one simple source loop."""
    local_edges = np.stack((np.arange(loop.size), np.roll(np.arange(loop.size), -1)), axis=1)
    triangulation = scoped_triangulation(points[loop], local_edges.astype(np.int32))
    if triangulation.triangles.shape[0] == 0:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "planar.loop_interior_point could not triangulate the source loop",
            provider_code="native_planar",
            stage="source_inspection",
        )
    triangle = np.asarray(triangulation.triangles[0], dtype=np.int64)
    return np.mean(np.asarray(triangulation.points)[triangle], axis=0)


def _segment_distances(points: np.ndarray, segments: np.ndarray, /) -> np.ndarray:
    """Compute point-to-closed-segment distances with linear scratch storage."""
    count = points.shape[0]
    result = np.empty((count, segments.shape[0]), dtype=np.float64)
    for column, segment in enumerate(segments):
        start, end = segment
        direction = end - start
        squared = float(np.dot(direction, direction))
        if squared == 0.0:
            raise ValueError("Planar size-field source edges must be nondegenerate.")
        along = np.clip(((points - start) @ direction) / squared, 0.0, 1.0)
        result[:, column] = np.linalg.norm(points - (start + along[:, None] * direction), axis=1)
    return result


def _field_sizes(
    points: np.ndarray,
    region_size: float,
    patch_segments: np.ndarray,
    patch_sizes: np.ndarray,
    slope: float,
    /,
    *,
    point_sources: np.ndarray | None = None,
    patch_source_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the declared Euclidean graded field without a scratch cap."""
    sizes = np.full((points.shape[0],), region_size, dtype=np.float64)
    if not patch_sizes.size or not points.shape[0]:
        return sizes
    if patch_segments.shape != (patch_sizes.size, 2, points.shape[1]):
        raise ValueError("Planar patch segments and patch sizes must align.")
    for column, (segment, target) in enumerate(zip(patch_segments, patch_sizes, strict=True)):
        distance = _segment_distances(points, segment[None, ...])[:, 0]
        if point_sources is not None and patch_source_rows is not None:
            if point_sources.shape != (points.shape[0],):
                raise ValueError("point_sources must align with size-field samples.")
            distance = np.where(point_sources == patch_source_rows[column], 0.0, distance)
        if math.isfinite(slope):
            sizes = np.minimum(sizes, target + slope * distance)
        else:
            sizes = np.minimum(sizes, np.where(distance == 0.0, target, np.inf))
    if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise MeshingFailure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            "planar.size_field must remain positive and finite",
            provider_code="native_planar",
            stage="source_inspection",
        )
    return sizes


def _resolve_field(
    specification: SurfaceMeshingSpec,
    vertices: np.ndarray,
    edges: np.ndarray,
    interior: np.ndarray,
    region_scope_entity_ids: np.ndarray,
    edge_entity_ids: np.ndarray,
    /,
) -> tuple[float, np.ndarray, np.ndarray, float, tuple[str, ...]]:
    """Resolve region and source-edge controls while retaining stable entity IDs."""
    controls = tuple(specification.size_controls)
    uniform_controls = tuple(
        control for control in controls if isinstance(control, UniformSizeControl)
    )
    if len(uniform_controls) != len(controls):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "planar field resolution requires uniform source-region and edge controls",
            provider_code="native_planar_sizing",
            stage="source_inspection",
        )
    region_controls = tuple(
        control for control in uniform_controls if control.scope.entity_dimension == 2
    )
    edge_controls = tuple(
        control for control in uniform_controls if control.scope.entity_dimension == 1
    )
    combination: SizeCombinationPolicy = specification.size_combination
    try:
        region_field, region_report = resolve_size_controls(
            region_controls,
            interior[None, :],
            region_scope_entity_ids,
            SizeFieldDomain.EUCLIDEAN_VOLUME,
            combination=combination,
        )
    except ValueError as error:
        raise MeshingFailure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            f"planar region size controls do not resolve: {error}",
            provider_code="native_planar_sizing",
            stage="source_inspection",
            entity_ids=tuple(
                int(value) for value in region_scope_entity_ids.tolist()
            ),
        ) from error
    selected_ids = (
        np.unique(
            np.concatenate(
                [
                    np.asarray(control.scope.entity_ids, dtype=np.int64)
                    for control in edge_controls
                ]
            )
        )
        if edge_controls
        else np.empty((0,), dtype=np.int64)
    )
    rows = np.flatnonzero(np.isin(edge_entity_ids, selected_ids))
    sizes = np.empty((0,), dtype=np.float64)
    reports = [region_report.report_id]
    if rows.size:
        midpoints = 0.5 * (
            vertices[edges[rows, 0]] + vertices[edges[rows, 1]]
        )
        try:
            patch_field, patch_report = resolve_size_controls(
                edge_controls,
                midpoints,
                edge_entity_ids[rows],
                SizeFieldDomain.EUCLIDEAN_VOLUME,
                combination=combination,
            )
        except ValueError as error:
            raise MeshingFailure(
                MeshingFailureCategory.CONTROL_CONFLICT,
                f"planar source-edge size controls do not resolve: {error}",
                provider_code="native_planar_sizing",
                stage="source_inspection",
                entity_ids=tuple(
                    int(value) for value in edge_entity_ids[rows].tolist()
                ),
            ) from error
        sizes = np.asarray(patch_field.values, dtype=np.float64)
        reports.append(patch_report.report_id)
    growth_rates = [
        control.maximum_growth_rate
        for control in uniform_controls
        if control.maximum_growth_rate is not None
    ]
    slope = min(growth_rates) - 1.0 if growth_rates else math.inf
    return float(np.asarray(region_field.values)[0]), rows, sizes, slope, tuple(reports)
@final
class PreparedPlanarDomain(StrictModule, NonTrainableState):
    """Immutable native input complex with per-vertex source ancestry."""

    source: sources_owner.NativePlanarSource
    points: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    point_sources: np.ndarray
    point_parameters: np.ndarray
    source_points: np.ndarray
    source_edges: np.ndarray
    source_loops: np.ndarray
    source_vertex_ids: np.ndarray
    source_edge_ids: np.ndarray
    holes: np.ndarray
    patch_segments: np.ndarray
    patch_sizes: np.ndarray
    patch_source_rows: np.ndarray
    source_bounds: np.ndarray
    region_size: float = eqx.field(static=True)
    gradation_slope: float = eqx.field(static=True)
    construction_size: float = eqx.field(static=True)

    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, source: sources_owner.NativePlanarSource, specification: SurfaceMeshingSpec, /
    ) -> None:
        vertices, edges, loops = source_arrays(source)
        vertex_ids, edge_ids = source_entity_arrays(source)
        issues = size_control_issues(
            specification,
            edge_ids,
            vertex_ids,
            np.asarray(source.scope_entity_ids(2), dtype=np.int64),
            source.scope_entity_set_id(2),
            source.scope_entity_set_id(1),
            source.scope_entity_set_id(0),
        )
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(dict.fromkeys(issues)),
                provider_code="native_planar_preflight",
                stage="source_inspection",
            )
        region_edges = np.asarray(source.region.edges, dtype=np.int64)
        offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
        loop_vertices = [
            region_edges[offsets[index] : offsets[index + 1], 0]
            for index in range(offsets.size - 1)
        ]
        interior_points = [
            interior_point(vertices, loop) for loop in loop_vertices
        ]
        region_size, patch_rows, patch_sizes, slope, resolutions = _resolve_field(
            specification,
            vertices,
            edges,
            interior_points[0],
            np.asarray(source.scope_entity_ids(2), dtype=np.int64),
            edge_ids,
        )
        patch_segments = (
            np.stack(
                (vertices[edges[patch_rows, 0]], vertices[edges[patch_rows, 1]]),
                axis=1,
            )
            if patch_rows.size
            else np.empty((0, 2, vertices.shape[1]), dtype=np.float64)
        )
        construction_size = (
            uniform_construction_size(specification, region_size)
            if patch_sizes.size == 0
            else region_size
        )
        points, segments, segment_sources, point_sources, point_parameters = (
            split_constraints(
                vertices,
                edges,
                edge_ids,
                construction_size if patch_sizes.size == 0 else region_size,
                patch_segments,
                patch_sizes,
                patch_rows,
                slope,
            )
        )
        bounds = np.full((edges.shape[0],), np.inf, dtype=np.float64)
        if specification.maximum_deviation is not None:
            bounds.fill(specification.maximum_deviation)

        for feature in specification.protected_features:
            if feature.feature_kind is FeatureKind.CURVE and feature.hard:
                selected = np.isin(
                    edge_ids, np.asarray(feature.scope.entity_ids, dtype=np.int64)
                )
                bounds[selected] = np.minimum(
                    bounds[selected], feature.maximum_deviation
                )
        holes = np.asarray(interior_points[1:], dtype=np.float64).reshape((-1, 2))
        self.source = source
        self.points = points
        self.segments = segments
        self.segment_sources = segment_sources
        self.point_sources = point_sources
        self.point_parameters = point_parameters
        self.source_points = vertices
        self.source_edges = edges
        self.source_loops = loops
        self.source_vertex_ids = vertex_ids
        self.source_edge_ids = edge_ids
        self.holes = holes
        self.patch_segments = patch_segments
        self.patch_sizes = patch_sizes
        self.patch_source_rows = patch_rows
        self.source_bounds = bounds
        self.region_size = region_size
        self.gradation_slope = slope
        self.construction_size = construction_size
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-planar-domain",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "size_resolutions": resolutions,
                "points": array_tree_fingerprint(points),
                "segments": array_tree_fingerprint(segments),
                "segment_sources": array_tree_fingerprint(segment_sources),
                "point_sources": array_tree_fingerprint(point_sources),
                "point_parameters": array_tree_fingerprint(point_parameters),
                "source_vertex_ids": array_tree_fingerprint(vertex_ids),
                "source_edge_ids": array_tree_fingerprint(edge_ids),
                "holes": array_tree_fingerprint(holes),
                "source_bounds": array_tree_fingerprint(bounds),
                "construction_size": construction_size,
            }
        )

    def sizes(
        self, points: np.ndarray, /, *, point_sources: np.ndarray | None = None
    ) -> np.ndarray:
        """Evaluate the prepared graded field at arbitrary source-plane points."""
        return _field_sizes(
            points,
            self.region_size,
            self.patch_segments,
            self.patch_sizes,
            self.gradation_slope,
            point_sources=point_sources,
            patch_source_rows=self.patch_source_rows,
        )

    def holes_for_cdt(self, /) -> np.ndarray | None:
        return self.holes if self.holes.size else None




def split_constraints(
    vertices: np.ndarray,
    edges: np.ndarray,
    source_edge_ids: np.ndarray,
    region_size: float,
    patch_segments: np.ndarray,
    patch_sizes: np.ndarray,
    patch_source_rows: np.ndarray,
    slope: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if vertices.ndim != 2 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Planar vertices and edges must have shapes (V, 2) and (E, 2).")
    if edges.size and (np.any(edges < 0) or np.any(edges >= vertices.shape[0])):
        raise ValueError("Planar source edges must index the source vertex bank.")
    if source_edge_ids.shape != (edges.shape[0],):
        raise ValueError("Planar source edge IDs must align with source constraints.")
    start = vertices[edges[:, 0]]
    direction = vertices[edges[:, 1]] - start
    lengths = np.linalg.norm(direction, axis=1)
    zero_length = np.flatnonzero(lengths <= 0.0)
    if zero_length.size:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "planar.constraints must contain no zero-length source edges",
            provider_code="native_planar",
            stage="source_inspection",
            entity_ids=tuple(
                int(value) for value in source_edge_ids[zero_length].tolist()
            ),
        )
    owner = np.arange(edges.shape[0], dtype=np.int64)
    lower = np.zeros((owner.size,), dtype=np.float64)
    upper = np.ones((owner.size,), dtype=np.float64)

    while True:
        middle = lower + 0.5 * (upper - lower)
        representable = (middle > lower) & (middle < upper)
        corners = np.concatenate(
            [
                start[owner] + parameter[:, None] * direction[owner]
                for parameter in (lower, middle, upper)
            ]
        )
        corner_sources = np.tile(owner, 3)
        sizes = _field_sizes(
            corners,
            region_size,
            patch_segments,
            patch_sizes,
            slope,
            point_sources=corner_sources,
            patch_source_rows=patch_source_rows,
        ).reshape((3, owner.size))
        required = np.min(sizes, axis=0)
        long = (upper - lower) * lengths[owner] > required
        unresolved = long & ~representable
        if np.any(unresolved):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "planar.constraints reached binary64 parameter resolution before meeting the physical size field",
                provider_code="native_planar_unresolved",
                stage="source_inspection",
                entity_ids=tuple(
                    int(value)
                    for value in np.unique(
                        source_edge_ids[owner[unresolved]]
                    ).tolist()
                ),
            )
        if not np.any(long):
            break
        owner = np.concatenate((owner[~long], owner[long], owner[long]))
        lower, upper = (
            np.concatenate((lower[~long], lower[long], middle[long])),
            np.concatenate((upper[~long], middle[long], upper[long])),
        )
    order = np.lexsort((lower, owner))
    owner, lower, upper = owner[order], lower[order], upper[order]
    split = lower > 0.0
    first = np.where(
        split,
        vertices.shape[0] + np.cumsum(split, dtype=np.int64) - 1,
        edges[owner, 0],
    )
    last = np.where(
        upper == 1.0,
        edges[owner, 1],
        np.concatenate((first[1:], np.zeros((1,), dtype=np.int64))),
    )
    points = np.concatenate(
        (vertices, start[owner[split]] + lower[split, None] * direction[owner[split]])
    )
    point_sources = np.concatenate(
        (np.full(vertices.shape[0], -1, dtype=np.int64), owner[split])
    )
    point_parameters = np.concatenate(
        (np.zeros(vertices.shape[0], dtype=np.float64), lower[split])
    )
    return points, np.stack((first, last), axis=1), owner, point_sources, point_parameters


def constrained_edges(
    triangles: np.ndarray, cell_segments: np.ndarray, segment_sources: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Recover unique mesh constraints without erasing conflicting source owners."""
    carried = cell_segments >= 0
    triangle_rows, opposite = np.nonzero(carried)
    if not triangle_rows.size:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
    pairs = np.sort(
        np.stack(
            (
                triangles[triangle_rows, (opposite + 1) % 3],
                triangles[triangle_rows, (opposite + 2) % 3],
            ),
            axis=1,
        ),
        axis=1,
    )
    source_rows = segment_sources[cell_segments[triangle_rows, opposite]]
    order = np.lexsort((source_rows, pairs[:, 1], pairs[:, 0]))
    pairs, source_rows = pairs[order], source_rows[order]
    first = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1,
        )
    )
    following = np.concatenate((first[1:], np.asarray((pairs.shape[0],))))
    for start, stop in zip(first, following, strict=True):
        if np.any(source_rows[start:stop] != source_rows[start]):
            raise ValueError(
                "Native CDT assigned one mesh constraint edge to multiple source entities."
            )
    return pairs[first], source_rows[first]





def source_edge_parameters(
    prepared: PreparedPlanarDomain,
    triangulation: triangulation_owner.ConstrainedDelaunayTriangulation,
    /,
) -> tuple[np.ndarray, np.ndarray, tuple[Fraction, ...]]:
    """Compose constraint splits while preserving the prepared source-point prefix."""
    points = np.asarray(triangulation.points, dtype=np.float64)
    input_count = triangulation.input_point_count
    prepared_count = prepared.points.shape[0]
    if (
        input_count < prepared_count
        or points.shape[0] < input_count
        or not np.array_equal(
            points[:prepared_count], np.asarray(prepared.points, dtype=np.float64)
        )
    ):
        raise ValueError("Native CDT output must preserve every prepared source point in its input prefix.")
    sources = np.full(points.shape[0], -1, dtype=np.int64)
    parameters = [Fraction(0) for _ in range(points.shape[0])]
    generated = np.flatnonzero(prepared.point_sources >= 0)
    sources[generated] = prepared.point_sources[generated]
    for vertex in generated.tolist():
        parameters[vertex] = Fraction(float(prepared.point_parameters[vertex]))

    def parameter_on_edge(vertex: int, edge: int) -> Fraction:
        if sources[vertex] == edge:
            return parameters[vertex]
        first, last = prepared.source_edges[edge]
        if vertex == first:
            return Fraction(0)
        if vertex == last:
            return Fraction(1)
        raise ValueError(
            f"Native CDT constraint ancestry for vertex {vertex} does not lie on source edge entity {int(prepared.source_edge_ids[edge])}."
        )
    native_sources = np.asarray(triangulation.constraint_sources, dtype=np.int64)
    native_parents = np.asarray(triangulation.constraint_parents, dtype=np.int64)
    native_ratios = np.asarray(triangulation.constraint_ratios, dtype=np.float64)
    if native_sources.shape != (points.shape[0],) or native_parents.shape != (points.shape[0], 2) or native_ratios.shape != (points.shape[0],):
        raise ValueError("Native CDT constraint ancestry arrays must align with output points.")
    for vertex in range(input_count, points.shape[0]):
        segment = int(native_sources[vertex])
        if segment < 0:
            continue
        first, last = native_parents[vertex]
        if not 0 <= first < vertex or not 0 <= last < vertex:
            raise ValueError(f"Native CDT constraint ancestry is cyclic at vertex {vertex}.")
        edge = int(prepared.segment_sources[segment])
        ratio = Fraction(float(native_ratios[vertex]))
        low = parameter_on_edge(int(first), edge)
        high = parameter_on_edge(int(last), edge)
        parameter = low + ratio * (high - low)
        if not 0 <= ratio <= 1 or not min(low, high) <= parameter <= max(low, high):
            raise ValueError(
                f"Native CDT constraint ancestry leaves source edge entity {int(prepared.source_edge_ids[edge])}."
            )
        sources[vertex] = edge
        parameters[vertex] = parameter
    return sources, np.asarray([float(value) for value in parameters]), tuple(parameters)


__all__ = [
    "PreparedPlanarDomain",
    "constrained_edges",
    "interior_point",
    "scoped_triangulation",
    "size_control_issues",
    "source_arrays",
    "source_entity_arrays",
    "source_edge_parameters",
    "split_constraints",
]
