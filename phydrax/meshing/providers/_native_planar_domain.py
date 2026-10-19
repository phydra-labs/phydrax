#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared planar size fields, constraints, and publication evidence."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

import phydrax.discretization as cell_mesh_owner
import phydrax.geometry as triangulation_owner
from phydrax.geometry._planar_primitives import rational_points
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._contracts import MeshingLimits, SurfaceMeshingSpec
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    exact_orient2d,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    NativeResourceFailure,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ...geometry._domain_coverage import PiecewiseLinearDomain
from ...geometry._source_fidelity import fidelity_norm_bound
from ...typing import checked
from .._controls import FeatureKind
from .._organization_records import MeshLabel, MeshPatch
from .._result import MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    resolve_size_controls,
    size_statistic_quantile,
    SizeCompliancePolicy,
    SizeControlStrength,
    SizeFieldDomain,
    UniformSizeControl,
)
from .._trace import MeshingStageKind
from . import _native_sources as sources_owner
from ._native_route_checks import check_deadline, unique_edges


# Point-by-segment distance entries evaluated per chunk of size-field queries.
_FIELD_WORKING_ENTRIES = 1 << 20
# Host bytes one CDT input point costs across one rebuild: its coordinates,
# about two triangles of vertex, neighbor and segment-id rows, and the edge
# table built from them.
_CDT_BYTES_PER_POINT = 16 + 2 * 3 * 3 * 8 + 3 * 2 * 8
# Default source-carrier fidelity bound in units of eps * coordinate scale.
# Entity support uses rational ancestry, never this distance allowance.
_CONSTRAINT_ROUNDING = 1024.0


@checked
def source_arrays(
    source: sources_owner.NativePlanarSource, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source vertices, source edges, and each edge's loop (-1 when embedded)."""

    region = source.region
    vertices = np.asarray(region.vertices, dtype=np.float64)
    edges = np.asarray(region.edges, dtype=np.int64)
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    loops = np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))
    if source.embedded is not None:
        embedded_vertices = np.asarray(source.embedded.vertices, dtype=np.float64)
        embedded_edges = np.asarray(source.embedded.edges, dtype=np.int64)
        edges = np.concatenate((edges, embedded_edges + vertices.shape[0]), axis=0)
        loops = np.concatenate(
            (loops, np.full((embedded_edges.shape[0],), -1, dtype=np.int64))
        )
        vertices = np.concatenate((vertices, embedded_vertices), axis=0)
    return vertices, edges, loops


def size_control_issues(
    specification: SurfaceMeshingSpec, edge_count: int, /
) -> list[str]:
    """Size controls must be uniform and scope the region or source edges."""

    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if not controls or len(controls) != len(specification.size_controls):
        return ["uniform size controls only"]
    scope = specification.scope
    issues: list[str] = []
    region_scoped = False
    for control in controls:
        control_scope = control.scope
        identifiers = np.asarray(control_scope.entity_ids)
        if (control_scope.source_id, control_scope.source_revision) != (
            scope.source_id,
            scope.source_revision,
        ):
            issues.append("size-control scopes of the meshed source revision")
        elif control_scope.entity_kind is not MeshingEntityKind.GEOMETRY:
            issues.append("size controls on planar geometry entities")
        elif control_scope.entity_dimension == 2 and np.array_equal(identifiers, (0,)):
            region_scoped = True
        elif control_scope.entity_dimension != 1 or np.any(identifiers >= edge_count):
            issues.append("size controls scoped to region:0 or to source edges")
    if not region_scoped:
        issues.append("a size control on the planar region")
    return issues


def interior_point(
    points: np.ndarray, loop: np.ndarray, limits: MeshingLimits, /
) -> np.ndarray:
    """A point strictly inside one simple loop: a centroid of its CDT."""

    local = np.arange(loop.size, dtype=np.int64)
    try:
        inside = triangulation_owner.ConstrainedDelaunayTriangulation(
            points[loop], np.stack((local, np.roll(local, -1)), axis=1)
        )
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        raise _capacity_failure(error, limits) from error
    return np.mean(inside.points[inside.triangles[0]], axis=0)


def _segment_distances(points: np.ndarray, segments: np.ndarray, /) -> np.ndarray:
    """Euclidean distances ``(points, segments)`` to closed segments."""

    start = segments[None, :, 0]
    direction = segments[None, :, 1] - start
    offset = points[:, None, :] - start
    squared = np.maximum(np.sum(direction * direction, axis=2), np.finfo(np.float64).tiny)
    along = np.clip(np.sum(offset * direction, axis=2) / squared, 0.0, 1.0)
    return np.linalg.norm(offset - along[..., None] * direction, axis=2)


def _field_sizes(
    points: np.ndarray,
    region_size: float,
    patch_segments: np.ndarray,
    patch_sizes: np.ndarray,
    slope: float,
    /,
) -> np.ndarray:
    """Graded target size ``min(h_region, min_e h_e + slope d_e)`` at points.

    An infinite slope (no declared growth rate) applies a patch size on its own
    edge only.
    """

    sizes = np.full((points.shape[0],), region_size, dtype=np.float64)
    if patch_sizes.size == 0 or points.shape[0] == 0:
        return sizes
    chunk = max(1, _FIELD_WORKING_ENTRIES // patch_sizes.size)
    for begin in range(0, points.shape[0], chunk):
        distance = _segment_distances(points[begin : begin + chunk], patch_segments)
        if math.isfinite(slope):
            graded = patch_sizes[None, :] + slope * distance
        else:
            graded = np.where(distance == 0.0, patch_sizes[None, :], np.inf)
        sizes[begin : begin + chunk] = np.minimum(
            sizes[begin : begin + chunk], np.min(graded, axis=1)
        )
    return sizes


def _resolve_field(
    specification: SurfaceMeshingSpec,
    vertices: np.ndarray,
    edges: np.ndarray,
    interior: np.ndarray,
    /,
) -> tuple[float, np.ndarray, np.ndarray, float, tuple[str, ...]]:
    """Region size, controlled edge rows and sizes, slope, and resolution ids.

    Region-scoped and edge-scoped controls resolve separately (their entity ids
    live in different entity sets) through the size resolution, which
    intersects hard intervals and applies priorities.
    """

    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    region = tuple(value for value in controls if value.scope.entity_dimension == 2)
    patch = tuple(value for value in controls if value.scope.entity_dimension == 1)
    combination = specification.size_combination
    region_field, region_report = resolve_size_controls(
        region,
        interior[None, :],
        np.zeros((1,), dtype=np.int64),
        SizeFieldDomain.EUCLIDEAN_VOLUME,
        combination=combination,
    )
    reports = [region_report.report_id]
    rows = np.zeros((0,), dtype=np.int64)
    sizes = np.zeros((0,), dtype=np.float64)
    if patch:
        rows = np.unique(
            np.concatenate(
                [np.asarray(value.scope.entity_ids, dtype=np.int64) for value in patch]
            )
        )
        midpoints = 0.5 * (vertices[edges[rows, 0]] + vertices[edges[rows, 1]])
        patch_field, patch_report = resolve_size_controls(
            patch,
            midpoints,
            rows,
            SizeFieldDomain.EUCLIDEAN_VOLUME,
            combination=combination,
        )
        sizes = np.asarray(patch_field.values, dtype=np.float64)
        reports.append(patch_report.report_id)
    rates = [
        value.maximum_growth_rate
        for value in controls
        if value.maximum_growth_rate is not None
    ]
    slope = min(rates) - 1.0 if rates else math.inf
    region_size = float(np.asarray(region_field.values)[0])
    return region_size, rows, sizes, slope, tuple(reports)


def split_constraints(
    vertices: np.ndarray,
    edges: np.ndarray,
    region_size: float,
    patch_segments: np.ndarray,
    patch_sizes: np.ndarray,
    slope: float,
    limits: MeshingLimits,
    /,
    *,
    balanced: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split source edges, retaining each point's binary64 rational parameter.

    Balanced initial pieces avoid power-of-two overshoot for a uniform
    statistical size target; subsequent field-driven subdivisions remain dyadic.
    """

    start = vertices[edges[:, 0]]
    direction = vertices[edges[:, 1]] - start
    if balanced:
        # Match boundary spacing to a square lattice whose diagonals fit h.
        counts = np.minimum(
            np.ceil(np.linalg.norm(direction, axis=1) / (region_size / math.sqrt(2.0))),
            limits.maximum_vertices + 1,
        ).astype(np.int64)
        point_count = vertices.shape[0] + int(np.sum(counts)) - edges.shape[0]
        if point_count > limits.maximum_vertices:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "planar.constraints must respect limits.maximum_vertices; got excess vertices",
                stage=MeshingStageKind.CURVE_MESHING.value,
                requested=(("maximum_vertices", limits.maximum_vertices),),
                achieved=(("constraint_vertices", point_count),),
            )
        owner = np.repeat(np.arange(edges.shape[0], dtype=np.int64), counts)
        lower = np.concatenate([np.arange(count) / count for count in counts])
        upper = np.concatenate([np.arange(1, count + 1) / count for count in counts])
    else:
        owner = np.arange(edges.shape[0], dtype=np.int64)
        lower = np.zeros((owner.size,), dtype=np.float64)
        upper = np.ones((owner.size,), dtype=np.float64)
    while True:
        middle = 0.5 * (lower + upper)
        corners = np.concatenate(
            [start[owner] + t[:, None] * direction[owner] for t in (lower, middle, upper)]
        )
        size = np.min(
            _field_sizes(
                corners, region_size, patch_segments, patch_sizes, slope
            ).reshape(3, -1),
            axis=0,
        )
        long = (upper - lower) * np.linalg.norm(direction[owner], axis=1) > size
        if not np.any(long):
            break
        point_count = vertices.shape[0] + owner.size + int(np.sum(long)) - edges.shape[0]
        if point_count > limits.maximum_vertices:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "planar.constraints must respect limits.maximum_vertices; got excess vertices",
                stage=MeshingStageKind.CURVE_MESHING.value,
                requested=(("maximum_vertices", limits.maximum_vertices),),
                achieved=(("constraint_vertices", point_count),),
            )
        owner = np.concatenate((owner[~long], owner[long], owner[long]))
        lower, upper = (
            np.concatenate((lower[~long], lower[long], middle[long])),
            np.concatenate((upper[~long], middle[long], upper[long])),
        )
    order = np.lexsort((lower, owner))
    owner, lower, upper = owner[order], lower[order], upper[order]
    split = lower > 0.0
    index = np.full((owner.size,), -1, dtype=np.int64)
    index[split] = vertices.shape[0] + np.arange(int(np.sum(split)), dtype=np.int64)
    first = np.where(split, index, edges[owner, 0])
    following = np.concatenate((first[1:], np.zeros((1,), dtype=np.int64)))
    last = np.where(upper == 1.0, edges[owner, 1], following)
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


def _loop_signs(source: sources_owner.NativePlanarSource, /) -> np.ndarray:
    """Orientation (+1 counterclockwise) of every region loop by signed area."""

    vertices = np.asarray(source.region.vertices, dtype=np.float64)
    edges = np.asarray(source.region.edges, dtype=np.int64)
    offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
    a = vertices[edges[:, 0]]
    b = vertices[edges[:, 1]]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.sign(np.add.reduceat(cross, offsets[:-1])).astype(np.int64)


@final
class PreparedPlanarDomain(StrictModule, NonTrainableState):
    """Graded size field and size-split constraint complex of one planar source.

    ``points`` are the source vertices followed by boundary split points;
    ``segments`` are split pieces and ``segment_sources`` their source edge IDs.
    ``point_sources`` and ``point_parameters`` retain each generated point's
    source edge and binary64 rational parameter; ``source_bounds`` limits carrier
    deviations, tightened by hard protected-feature declarations.
    ``patch_segments``/``patch_sizes`` are the size-controlled source edges and
    their resolved sizes; ``region_size`` the resolved region size and
    ``gradation_slope`` the declared growth rate minus one (``inf`` if none).
    """

    points: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    point_sources: np.ndarray
    point_parameters: np.ndarray
    source_edges: np.ndarray
    source_points: np.ndarray
    source_bounds: np.ndarray
    source_directions: np.ndarray
    source_loops: np.ndarray
    loop_signs: np.ndarray
    holes: np.ndarray
    patch_segments: np.ndarray
    patch_sizes: np.ndarray
    region_size: float = eqx.field(static=True)
    gradation_slope: float = eqx.field(static=True)
    maximum_area: float = eqx.field(static=True)
    minimum_angle_degrees: float = eqx.field(static=True)
    maximum_steiner: int = eqx.field(static=True)
    maximum_triangles: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlanarSource,
        specification: SurfaceMeshingSpec,
        /,
        *,
        balanced: bool = False,
    ) -> None:
        limits = specification.limits
        vertices, edges, loops = source_arrays(source)
        offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
        region_edges = np.asarray(source.region.edges, dtype=np.int64)
        interior = [
            interior_point(vertices, region_edges[offsets[k] : offsets[k + 1], 0], limits)
            for k in range(offsets.size - 1)
        ]
        region_size, rows, sizes, slope, resolutions = _resolve_field(
            specification, vertices, edges, interior[0]
        )
        patch_segments = np.stack((vertices[edges[rows, 0]], vertices[edges[rows, 1]]), 1)
        points, segments, sources, point_sources, point_parameters = split_constraints(
            vertices,
            edges,
            region_size,
            patch_segments,
            sizes,
            slope,
            limits,
            balanced=balanced
            and not sizes.size
            and _uniform_median_strategy(specification, region_size),
        )
        holes = np.asarray(interior[1:], dtype=np.float64).reshape((-1, 2))
        steiner = min(
            limits.maximum_work_units, limits.maximum_vertices - points.shape[0]
        )
        quality = specification.quality_target
        self.points = points
        self.segments = segments
        self.segment_sources = sources
        self.point_sources = point_sources
        self.point_parameters = point_parameters
        self.source_edges = edges
        self.source_points = vertices
        source_bound = (
            _CONSTRAINT_ROUNDING
            * np.finfo(np.float64).eps
            * max(float(np.max(np.abs(points))), 1.0)
        )
        bounds = np.full(edges.shape[0], source_bound, dtype=np.float64)
        for feature in specification.protected_features:
            if feature.hard and feature.feature_kind is not FeatureKind.CORNER:
                rows = np.asarray(feature.scope.entity_ids, dtype=np.int64)
                bounds[rows] = np.minimum(bounds[rows], feature.maximum_deviation)
        self.source_bounds = bounds
        self.source_directions = vertices[edges[:, 1]] - vertices[edges[:, 0]]
        self.source_loops = loops
        self.loop_signs = _loop_signs(source)
        self.holes = holes
        self.patch_segments = patch_segments
        self.patch_sizes = sizes
        self.region_size = region_size
        self.gradation_slope = slope
        self.maximum_area = 0.25 * math.sqrt(3.0) * region_size * region_size
        self.minimum_angle_degrees = (
            0.0 if quality is None else math.degrees(quality.minimum_angle)
        )
        self.maximum_steiner = steiner
        self.maximum_triangles = min(limits.maximum_cells, limits.maximum_faces)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-planar-domain",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "size_resolutions": resolutions,
                "points": array_tree_fingerprint(self.points),
                "segments": array_tree_fingerprint(self.segments),
                "segment_sources": array_tree_fingerprint(self.segment_sources),
                "point_sources": array_tree_fingerprint(self.point_sources),
                "point_parameters": array_tree_fingerprint(self.point_parameters),
                "source_bounds": array_tree_fingerprint(self.source_bounds),
                "holes": array_tree_fingerprint(holes),
                "minimum_angle_degrees": self.minimum_angle_degrees,
                "maximum_steiner": steiner,
                "maximum_triangles": self.maximum_triangles,
            }
        )

    def sizes(self, points: np.ndarray, /) -> np.ndarray:
        """Target size of the graded field at points."""

        return _field_sizes(
            points,
            self.region_size,
            self.patch_segments,
            self.patch_sizes,
            self.gradation_slope,
        )

    def domain(
        self, source: sources_owner.NativePlanarSource, /
    ) -> PiecewiseLinearDomain:
        """Return the authored planar region with outward-oriented source loops."""
        loop = self.source_loops
        on_loop = loop >= 0
        facets = self.source_edges[on_loop]
        wanted = np.where(loop[on_loop] == 0, 1, -1)
        forward = self.loop_signs[loop[on_loop]] == wanted
        facets = np.where(forward[:, None], facets, facets[:, ::-1])
        return PiecewiseLinearDomain(
            self.source_points,
            facets,
            np.tile(np.asarray(((0, -1),), dtype=np.int64), (facets.shape[0], 1)),
            (sources_owner.source_entity_id(source.source_revision, "region", 0),),
            source_id=source.source_id,
        )


def constrained_edges(
    triangles: np.ndarray, segment_ids: np.ndarray, sources: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted vertex pairs of constrained mesh edges and their source edges."""

    carried = segment_ids >= 0
    rows, opposite = np.nonzero(carried)
    first = triangles[rows, (opposite + 1) % 3]
    second = triangles[rows, (opposite + 2) % 3]
    pairs = np.sort(np.stack((first, second), axis=1), axis=1)
    edge_sources = sources[segment_ids[rows, opposite]]
    keys, index = np.unique(pairs, axis=0, return_index=True)
    return keys, edge_sources[index]


def _association_distances(
    points: np.ndarray,
    source_points: np.ndarray,
    source_edges: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Return outward-certified nearest-source projection residuals."""
    original = rational_points(source_points, budget=budget)
    distances = np.empty(points.shape[0], dtype=np.float64)
    for row, point in enumerate(rational_points(points, budget=budget)):
        index = int(indices[row])
        match int(dimensions[row]):
            case 0:
                closest = original[index]
            case 1:
                first, last = source_edges[index].tolist()
                start, end = original[first], original[last]
                direction = tuple(b - a for a, b in zip(start, end, strict=True))
                squared = sum(value * value for value in direction)
                parameter = (
                    sum(
                        (p - a) * d
                        for p, a, d in zip(point, start, direction, strict=True)
                    )
                    / squared
                )
                parameter = min(max(parameter, 0), 1)
                closest = tuple(
                    a + parameter * d for a, d in zip(start, direction, strict=True)
                )
            case invalid:
                raise ValueError(
                    f"association.source_dimension must be zero or one; got {invalid}"
                )
        difference = tuple(p - q for p, q in zip(point, closest, strict=True))
        distances[row] = (
            fidelity_norm_bound(
                tuple(({(0,): value} for value in difference)), 1, budget=budget
            )
            if any(difference)
            else 0.0
        )
    return distances


@checked
def organization_and_associations(
    mesh: cell_mesh_owner.CellMesh,
    source: sources_owner.NativePlanarSource,
    prepared: PreparedPlanarDomain,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    input_vertex_rows: np.ndarray,
    authority: ExactPlcCellGeometrySource,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshLabel, ...], tuple[GeometryAssociation, ...]]:
    """Emit source authority from source input rows and recovered constraints.

    The CDT preserves input points before Steiner points; compaction's ``used``
    rows therefore identify authored vertices, including embedded endpoints.
    Geometry validates their support, never reconstructs their identity. Planar
    region rows retain geometric dimension two and an explicit REGION role.
    Residuals are outward bounds against the identified source primitives;
    flags require zero deviation and unique source classification.
    """
    revision = source.source_revision
    edge_set = mesh.entity_set(1)
    # ty: ignore[unresolved-attribute]
    canonical = np.asarray(mesh.connectivity.edges, dtype=np.int64)
    rows = {pair: row for row, pair in enumerate(map(tuple, np.sort(canonical, axis=1)))}
    edge_rows = np.asarray(
        [rows[tuple(pair)] for pair in constrained.tolist()], dtype=np.int64
    )
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[edge_rows]
    order = np.argsort(edge_ids, kind="stable")
    edge_ids, edge_rows = edge_ids[order], edge_rows[order]
    edge_sources = constrained_sources[order]
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    source_points, source_edges, _ = source_arrays(source)
    input_rows = np.asarray(input_vertex_rows)
    if not np.issubdtype(input_rows.dtype, np.integer):
        raise TypeError(
            "triangulation.input_rows must contain integer source rows; got a noninteger dtype"
        )
    input_rows = input_rows.astype(np.int64, copy=False)
    if (
        input_rows.shape != (points.shape[0],)
        or np.any(input_rows < 0)
        or np.unique(input_rows).size != input_rows.size
    ):
        raise ValueError(
            "triangulation.input_rows must match compacted points and source rows; got inconsistent rows"
        )
    endpoint_rows = canonical[edge_rows].reshape(-1)
    endpoint_sources = np.repeat(edge_sources, 2)
    edge_distances = (
        _association_distances(
            points[endpoint_rows],
            source_points,
            source_edges,
            np.ones(endpoint_sources.shape, dtype=np.int8),
            endpoint_sources,
            budget=budget,
        )
        .reshape((-1, 2))
        .max(axis=1)
    )
    boundary = prepared.source_loops[edge_sources] >= 0
    region_distance = float(np.max(edge_distances[boundary], initial=0.0))
    vertex_dims = np.full(points.shape[0], 2, dtype=np.int8)
    vertex_sources = np.zeros(points.shape[0], dtype=np.int64)
    vertex_ambiguous = np.zeros(points.shape[0], dtype=np.bool_)
    original = input_rows < source_points.shape[0]
    vertex_dims[original], vertex_sources[original] = 0, input_rows[original]
    for row in np.unique(endpoint_rows).tolist():
        if original[row]:
            continue
        candidates = np.unique(endpoint_sources[endpoint_rows == row])
        vertex_dims[row], vertex_sources[row] = 1, candidates[0]
        vertex_ambiguous[row] = candidates.size != 1
    vertex_distances = np.full(points.shape[0], region_distance, dtype=np.float64)
    constrained_vertices = vertex_dims < 2
    vertex_distances[constrained_vertices] = _association_distances(
        points[constrained_vertices],
        source_points,
        source_edges,
        vertex_dims[constrained_vertices],
        vertex_sources[constrained_vertices],
        budget=budget,
    )
    vertex_parameters = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    edge_vertices = vertex_dims == 1
    witness_rows = np.asarray(authority.vertex_rows, dtype=np.int64)
    if np.any(witness_rows[edge_vertices] != vertex_sources[edge_vertices]):
        raise ValueError(
            "association.source_edges must agree with retained ancestry; got conflicting rows"
        )
    vertex_parameters[edge_vertices] = np.asarray(authority.vertex_parameters)[
        edge_vertices
    ]
    along = points[canonical[edge_rows, 1]] - points[canonical[edge_rows, 0]]
    orientations = np.sign(
        np.sum(along * prepared.source_directions[edge_sources], axis=1)
    ).astype(np.int8)
    cell_set = mesh.entity_set(2)
    cell_count = cell_set.count
    associations = (
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            cell_set.entity_set_id,
            cell_set.entity_ids,
            tuple(
                sources_owner.source_entity_id(revision, "region", 0)
                for _ in range(cell_count)
            ),
            np.full((cell_count,), region_distance, dtype=np.float64),
            exact=region_distance == 0.0,
            source_dimensions=np.full(cell_count, 2, dtype=np.int8),
            source_indices=np.zeros(cell_count, dtype=np.int64),
            source_entity_roles=(GeometrySourceEntityRole.REGION,) * cell_count,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            edge_set.entity_set_id,
            edge_ids,
            tuple(
                sources_owner.source_entity_id(revision, "edge", int(edge))
                for edge in edge_sources.tolist()
            ),
            edge_distances,
            exact=bool(np.all(edge_distances == 0.0)),
            orientations=orientations,
            source_dimensions=np.full(edge_ids.size, 1, dtype=np.int8),
            source_indices=edge_sources,
            source_entity_roles=(GeometrySourceEntityRole.EDGE,) * edge_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(0).entity_set_id,
            mesh.vertex_global_ids,
            tuple(
                sources_owner.source_entity_id(
                    revision, ("vertex", "edge", "region")[int(kind)], int(index)
                )
                for kind, index in zip(vertex_dims, vertex_sources, strict=True)
            ),
            vertex_distances,
            resolved=~vertex_ambiguous,
            ambiguous=vertex_ambiguous,
            exact=not bool(np.any(vertex_ambiguous))
            and bool(np.all(vertex_distances == 0.0)),
            source_dimensions=vertex_dims,
            source_indices=vertex_sources,
            parameters=vertex_parameters,
            source_entity_roles=tuple(
                (
                    GeometrySourceEntityRole.VERTEX,
                    GeometrySourceEntityRole.EDGE,
                    GeometrySourceEntityRole.REGION,
                )[int(kind)]
                for kind in vertex_dims
            ),
        ),
    )

    def edge_scope(selected: np.ndarray, /) -> MeshingScope:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            1,
            edge_set.entity_set_id,
            selected,
        )

    loops = prepared.source_loops[edge_sources]
    patches = tuple(
        MeshPatch(f"loop:{loop}", edge_scope(edge_ids[loops == loop]))
        for loop in np.unique(loops[loops >= 0]).tolist()
    )
    labels = (
        (MeshLabel("embedded", edge_scope(edge_ids[loops < 0])),)
        if np.any(loops < 0)
        else ()
    )
    return patches, labels, associations


@checked
def graded_size_compliance(
    control: UniformSizeControl,
    policy: SizeCompliancePolicy,
    lengths: np.ndarray,
    local: np.ndarray,
    gradation: float,
    /,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    """Requested, achieved, and failed quantities of one control's edges.

    ``local`` is the field target at each selected edge; a hard control fails
    when a target statistic, bound, the local target, or the field growth rate
    is missed beyond the compliance tolerance.
    """

    key = f"size:{control.control_id}"
    requested = [(f"{key}:target_size", control.target_size)]
    requested.extend(
        (f"{key}:{name}", value)
        for name, value in (
            ("minimum_size", control.minimum_size),
            ("maximum_size", control.maximum_size),
            ("maximum_growth_rate", control.maximum_growth_rate),
        )
        if value is not None
    )
    ratio = lengths / local
    achieved = [
        (f"{key}:edge_count", float(lengths.size)),
        (f"{key}:minimum_edge", float(np.min(lengths))),
        (f"{key}:maximum_edge", float(np.max(lengths))),
        (f"{key}:maximum_edge_to_local_size", float(np.max(ratio))),
        (f"{key}:size_field_growth_rate", gradation),
    ]
    statistics = {
        name: float(np.quantile(lengths, size_statistic_quantile(name)))
        for name in policy.target_statistics
    }
    achieved.extend((f"{key}:{name}_edge", value) for name, value in statistics.items())
    if control.strength is SizeControlStrength.SOFT:
        return requested, achieved, []

    tolerance = policy.tolerance

    issues = [
        f"target_size_{name}:{control.control_id}"
        for name, value in statistics.items()
        if abs(value - control.target_size) > tolerance(control.target_size)
    ]
    checks = (
        (
            "minimum_size",
            control.minimum_size is not None
            and float(np.min(lengths))
            < control.minimum_size - tolerance(control.minimum_size),
        ),
        (
            "maximum_size",
            control.maximum_size is not None
            and float(np.max(lengths))
            > control.maximum_size + tolerance(control.maximum_size),
        ),
        (
            "local_size",
            bool(
                np.any(
                    lengths
                    > local
                    + policy.absolute_tolerance
                    + policy.relative_tolerance * local
                )
            ),
        ),
        (
            "maximum_growth_rate",
            control.maximum_growth_rate is not None
            and gradation
            > control.maximum_growth_rate + tolerance(control.maximum_growth_rate),
        ),
    )
    issues.extend(f"{name}:{control.control_id}" for name, failed in checks if failed)
    return requested, achieved, issues


@checked
def control_edges(
    control: UniformSizeControl,
    prepared: PreparedPlanarDomain,
    edges: np.ndarray,
    local: np.ndarray,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    /,
) -> np.ndarray:
    """Rows of the mesh edges a control governs.

    Edge-scoped controls govern the constrained edges on their source edges; a
    region control governs the edges where its size is the active target.
    """

    if control.scope.entity_dimension == 1:
        pairs = constrained[
            np.isin(constrained_sources, np.asarray(control.scope.entity_ids))
        ]
        width = np.int64(max(int(np.max(edges, initial=0)), 1) + 1)
        return np.flatnonzero(
            np.isin(edges[:, 0] * width + edges[:, 1], pairs[:, 0] * width + pairs[:, 1])
        )
    governed = np.flatnonzero(local >= prepared.region_size)
    return governed if governed.size else np.arange(edges.shape[0])


@checked
def planar_compliance(
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    vertices: np.ndarray,
    triangles: np.ndarray,
    constrained: tuple[np.ndarray, np.ndarray],
    refinement: tuple[float, int, str],
    constraint_rounding: tuple[float, float],
    /,
) -> MeshingComplianceReport:
    minimum_angle_degrees, steiner_count, status = refinement
    edges = unique_edges(triangles, "triangle")
    a = vertices[edges[:, 0]]
    b = vertices[edges[:, 1]]
    lengths = np.linalg.norm(b - a, axis=1)
    local = np.min(
        prepared.sizes(np.concatenate((a, b, 0.5 * (a + b)))).reshape(3, -1), axis=0
    )
    endpoint = prepared.sizes(vertices)
    gradation = 1.0 + float(
        np.max(np.abs(endpoint[edges[:, 1]] - endpoint[edges[:, 0]]) / lengths)
    )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    for control in specification.size_controls:
        match control:
            case UniformSizeControl():
                rows = control_edges(control, prepared, edges, local, *constrained)
                control_requested, control_achieved, control_issues = (
                    graded_size_compliance(
                        control,
                        specification.size_compliance,
                        lengths[rows],
                        local[rows],
                        gradation,
                    )
                )
                requested.extend(control_requested)
                achieved.extend(control_achieved)
                issues.extend(control_issues)
            case _:
                raise TypeError(
                    "specification.size_controls must contain UniformSizeControl; got an unsupported control"
                )
    corners = vertices[triangles]
    areas = 0.5 * np.abs(
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    )
    requested.extend(
        (
            ("maximum_triangle_area", prepared.maximum_area),
            ("maximum_steiner_points", prepared.maximum_steiner),
        )
    )
    achieved.extend(
        (
            ("maximum_triangle_area", float(np.max(areas))),
            ("minimum_angle", math.radians(minimum_angle_degrees)),
            ("steiner_points", steiner_count),
        )
    )
    quality = specification.quality_target
    if quality is not None:
        requested.append(("minimum_angle", quality.minimum_angle))
        if quality.hard and math.radians(minimum_angle_degrees) < quality.minimum_angle:
            issues.append(f"minimum_angle:{quality.target_id}")
    if status != "ok":
        issues.append(f"refinement_limit:{status}")
    rounding, admissible = constraint_rounding
    requested.append(("maximum_constraint_rounding", admissible))
    achieved.append(("maximum_constraint_rounding", rounding))
    if rounding > admissible:
        issues.append("constraint_rounding")
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        # Source vertices are retained without rounding. Split curve vertices
        # retain the independently measured source-segment rounding bound.
        deviation = 0.0 if feature.feature_kind is FeatureKind.CORNER else rounding
        key = f"protected:{feature.feature_id}:maximum_deviation"
        achieved.append((key, deviation))
        if feature.hard and deviation > feature.maximum_deviation:
            issues.append(key)
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _refinement_budget(
    prepared: PreparedPlanarDomain, limits: MeshingLimits, inserted: int, /
) -> None:
    """Refuse a CDT rebuild whose host scratch would exceed the declared budget."""

    count = prepared.points.shape[0] + inserted
    scratch = count * _CDT_BYTES_PER_POINT
    if scratch > limits.maximum_scratch_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "planar.refinement must respect limits.maximum_scratch_bytes; got excess scratch storage",
            stage=MeshingStageKind.SURFACE_MESHING.value,
            requested=(("maximum_scratch_bytes", limits.maximum_scratch_bytes),),
            achieved=(("scratch_bytes", scratch), ("input_points", count)),
        )


def _uniform_median_strategy(specification: SurfaceMeshingSpec, size: float, /) -> bool:
    """Capability of the fixed-spacing construction, not an acceptance policy.

    The lattice is designed for a single median with an interval containing
    [h/2, h]. Tight/additional statistics or a hard lower edge bound must retain
    the established construction; their contracts cannot be inferred from a median.
    """
    hard = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
        and control.strength is SizeControlStrength.HARD
    )
    return (
        bool(hard)
        and specification.size_compliance.target_statistics == ("p50",)
        and all(
            control.scope.entity_dimension == 2
            and control.target_size == size
            and control.minimum_size is None
            and specification.size_compliance.tolerance(size) >= 0.5 * size
            for control in hard
        )
    )


def _distribution_seeds(
    prepared: PreparedPlanarDomain, specification: SurfaceMeshingSpec, /
) -> np.ndarray:
    """Interior sites for a uniform hard median, charged as refinement points.

    Quality-only insertion resolves small features without populating the coarse
    interior. A square lattice at 0.55 h keeps its diagonals below h and avoids
    the h/2 endpoint of a 50-percent median interval. A 0.2 h constraint clearance
    avoids nearly coincident sites. Neither choice changes the declared field,
    statistics, geometry, quality target, or final certification.
    """
    if prepared.patch_sizes.size or not _uniform_median_strategy(
        specification, prepared.region_size
    ):
        return np.empty((0, 2), dtype=np.float64)
    limits = specification.limits
    step = 0.55 * prepared.region_size
    # Scalar reductions precede the admission boundary; no proposal/edge arrays
    # are allocated until both work and all host working storage fit.
    lower = tuple(float(np.min(prepared.source_points[:, axis])) for axis in (0, 1))
    upper = tuple(float(np.max(prepared.source_points[:, axis])) for axis in (0, 1))
    shape = tuple(
        max(math.ceil((b - a) / step) - 1, 0) for a, b in zip(lower, upper, strict=True)
    )
    proposals = shape[0] * shape[1]
    if not proposals:
        return np.empty((0, 2), dtype=np.float64)
    edges = prepared.source_edges.shape[0]
    boundary_count = sum(int(loop) >= 0 for loop in prepared.source_loops)
    # One site construction and one visit per orientation/distance edge.
    work = proposals * (1 + boundary_count + edges)
    capacity = min(proposals, prepared.maximum_steiner)
    retained = capacity * 2 * np.dtype(np.float64).itemsize
    # Bounds cover gathered source rows, three contiguous broadcast predicate
    # operands, signs/masks, all distance intermediates and selected-point copies.
    # The output is a separately tracked native buffer, never kept+concatenated.
    fixed = 128 + 64 * (edges + boundary_count)
    per_point = 128 * (1 + edges + boundary_count)
    with NativeExecutionScope(limits.native_limits()) as scope:
        allowance = scope.remaining()
        available = min(limits.maximum_scratch_bytes, allowance.remaining_managed_bytes)
        minimum_scratch = fixed + retained + per_point
        if minimum_scratch > available:
            # Preserve the owner's raw refusal before any array allocation.
            with scope.batch(0, bytes_upper=minimum_scratch):
                pass
        chunk = min(
            proposals,
            max(1, _FIELD_WORKING_ENTRIES // max(1, edges)),
            max(1, (available - fixed - retained) // per_point),
        )
        scratch = fixed + chunk * per_point
        with scope.batch(work, bytes_upper=scratch) as ticket:
            output = scope.allocate_host_array((capacity, 2), np.float64)
            boundary = prepared.source_edges[prepared.source_loops >= 0]
            a = prepared.source_points[boundary[:, 0]]
            b = prepared.source_points[boundary[:, 1]]
            segments = np.stack(
                (
                    prepared.source_points[prepared.source_edges[:, 0]],
                    prepared.source_points[prepared.source_edges[:, 1]],
                ),
                axis=1,
            )
            count = 0
            for begin in range(0, proposals, chunk):
                batch_size = min(chunk, proposals - begin)
                ticket.consume(batch_size * (1 + boundary_count + edges))
                index = np.arange(begin, begin + batch_size, dtype=np.int64)
                points = np.asarray(lower) + step * np.stack(
                    (index // shape[1] + 1, index % shape[1] + 1), axis=1
                )
                signs = exact_orient2d(a[None, :, :], b[None, :, :], points[:, None, :])
                rising = (a[None, :, 1] <= points[:, None, 1]) & (
                    points[:, None, 1] < b[None, :, 1]
                )
                falling = (b[None, :, 1] <= points[:, None, 1]) & (
                    points[:, None, 1] < a[None, :, 1]
                )
                inside = (
                    np.count_nonzero(
                        (rising & (signs > 0)) | (falling & (signs < 0)), axis=1
                    )
                    % 2
                    == 1
                )
                clearance = np.min(_segment_distances(points, segments), axis=1)
                selected = inside & (clearance > 0.2 * prepared.region_size)
                end = count + int(np.count_nonzero(selected))
                if end > capacity:
                    raise MeshingFailure(
                        MeshingFailureCategory.RESOURCE_EXHAUSTED,
                        "planar.interior_sites must respect the refinement point budget; got excess sites",
                        stage=MeshingStageKind.SURFACE_MESHING.value,
                        requested=(("maximum_steiner_points", prepared.maximum_steiner),),
                        achieved=(("interior_sites", end),),
                    )
                output[count:end] = points[selected]
                count = end
            return output[:count]


def _capacity_failure(error: MeshcoreError, limits: MeshingLimits, /) -> MeshingFailure:
    """Translate capacity admission without discarding the native demand ledger."""
    demand = error.batch_requested_upper
    achieved = [("native_status", int(error.status))]
    if demand is not None:
        achieved.extend((("work_units", demand[0]), ("scratch_bytes", demand[3])))
        achieved.extend(
            (f"native_batch:requested_{name}", value)
            for name, value in zip(
                ("work_units", "geometry_queries", "cavity_cells", "managed_bytes"),
                demand,
                strict=True,
            )
        )
    if error.batch_refusal_evidence is not None:
        achieved.extend(
            (f"native_batch:refusal:{index}", int(value))
            for index, value in enumerate(error.batch_refusal_evidence)
        )
    if error.memory_counters is not None:
        achieved.extend(
            (f"native_memory:{name}", value)
            for name, value in error.memory_counters.items()
        )
    if isinstance(error, NativeResourceFailure) and error.work_evidence_kind == "planar":
        achieved.extend(
            (f"native_planar:{name}", int(value))
            for name, value in zip(
                (
                    "work_units",
                    "orient_requests",
                    "incircle_requests",
                    "power_requests",
                    "insertions",
                    "flips",
                    "peak_cavity_cells",
                    "work_refusals",
                    "cavity_refusals",
                ),
                error.work_evidence,
                strict=True,
            )
        )
    return MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED,
        "planar.refinement must respect the declared work and scratch allowances; got excess resource demand",
        stage=MeshingStageKind.SURFACE_MESHING.value,
        requested=(
            ("maximum_work_units", limits.maximum_work_units),
            ("maximum_scratch_bytes", limits.maximum_scratch_bytes),
        ),
        achieved=tuple(achieved),
    )


@checked
def triangulate(
    prepared: PreparedPlanarDomain,
    started: float,
    specification: SurfaceMeshingSpec,
    /,
    *,
    distribution_sites: bool = False,
) -> tuple[triangulation_owner.ConstrainedDelaunayTriangulation, int]:
    """Retain one cumulative native allowance across every CDT rebuild."""
    try:
        with NativeExecutionScope(specification.limits.native_limits()):
            return _triangulate(
                prepared, started, specification, distribution_sites=distribution_sites
            )
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        raise _capacity_failure(error, specification.limits) from error


def _triangulate(
    prepared: PreparedPlanarDomain,
    started: float,
    specification: SurfaceMeshingSpec,
    /,
    *,
    distribution_sites: bool = False,
) -> tuple[triangulation_owner.ConstrainedDelaunayTriangulation, int]:
    """Refined CDT whose unconstrained edges respect the local target size.

    Area refinement does not bound edge length, so every unconstrained edge
    longer than the field at its ends or midpoint contributes its midpoint as
    a free input point and the CDT is rebuilt; lengths halve per round.
    Inserted points are charged to the work budget together with the
    refinement Steiner points.
    """

    limits = specification.limits
    extra = (
        _distribution_seeds(prepared, specification)
        if distribution_sites
        else np.empty((0, 2), dtype=np.float64)
    )
    while True:
        _refinement_budget(prepared, limits, extra.shape[0])
        triangulation = triangulation_owner.ConstrainedDelaunayTriangulation(
            np.concatenate((prepared.points, extra), axis=0),
            prepared.segments,
            holes=prepared.holes if prepared.holes.shape[0] else None,
            keep_convex_hull=False,
            min_angle=prepared.minimum_angle_degrees,
            max_area=prepared.maximum_area,
            max_steiner=max(prepared.maximum_steiner - extra.shape[0], 0),
            max_triangles=prepared.maximum_triangles,
        )
        check_deadline(started, limits, MeshingStageKind.SURFACE_MESHING)
        points = np.asarray(triangulation.points, dtype=np.float64)
        triangles = np.asarray(triangulation.triangles, dtype=np.int64)
        edges = unique_edges(triangles, "triangle")
        constrained, _ = constrained_edges(
            triangles,
            np.asarray(triangulation.segment_ids, dtype=np.int64),
            prepared.segment_sources,
        )
        width = np.int64(points.shape[0])
        free = ~np.isin(
            edges[:, 0] * width + edges[:, 1],
            constrained[:, 0] * width + constrained[:, 1],
        )
        a = points[edges[:, 0]]
        b = points[edges[:, 1]]
        lengths = np.linalg.norm(b - a, axis=1)
        local = np.min(
            prepared.sizes(np.concatenate((a, b, 0.5 * (a + b)))).reshape(3, -1), axis=0
        )
        long = free & (lengths > local)
        if not np.any(long):
            return triangulation, extra.shape[0]
        midpoints = 0.5 * (a[long] + b[long])
        if extra.shape[0] + midpoints.shape[0] > prepared.maximum_steiner:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "planar.refinement must respect limits.maximum_work_units; got excess edge-length work",
                stage=MeshingStageKind.SURFACE_MESHING.value,
                requested=(("maximum_steiner_points", prepared.maximum_steiner),),
                achieved=(
                    ("maximum_edge_to_local_size", float(np.max(lengths / local))),
                    ("steiner_points", extra.shape[0] + midpoints.shape[0]),
                ),
            )
        extra = np.concatenate((extra, midpoints), axis=0)


def _median_repair_needed(
    prepared: PreparedPlanarDomain,
    specification: SurfaceMeshingSpec,
    triangulation: triangulation_owner.ConstrainedDelaunayTriangulation,
    scope: NativeExecutionScope,
    /,
) -> bool:
    """Retain compliant generation; repair only an eligible undersized median."""
    if (
        prepared.patch_sizes.size
        or not _uniform_median_strategy(specification, prepared.region_size)
        or triangulation.evidence.status != "ok"
    ):
        return False
    quality = specification.quality_target
    if (
        quality is not None
        and quality.hard
        and math.radians(triangulation.evidence.minimum_angle_degrees)
        < quality.minimum_angle
    ):
        return False
    cells = triangulation.triangles.shape[0]
    vertices = triangulation.points.shape[0]
    edge_upper = 3 * cells
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
        and control.strength is SizeControlStrength.HARD
    )
    work = edge_upper * (edge_upper.bit_length() + 4) * len(controls)
    # Edge extraction/sorting, vector norms and quantile copies coexist with
    # the retained initial triangulation, whose owner has its own live bound.
    scratch = 512 + 512 * cells + 64 * vertices + 256 * len(controls)
    with scope.batch(work, bytes_upper=scratch) as ticket:
        ticket.consume(work)
        edges = unique_edges(triangulation.triangles, "triangle")
        points = triangulation.points
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        local = np.full(lengths.shape, prepared.region_size, dtype=np.float64)
        issues = []
        for control in controls:
            _, _, failed = graded_size_compliance(
                control, specification.size_compliance, lengths, local, 1.0
            )
            issues.extend(failed)
        return (
            bool(issues)
            and all(issue.startswith("target_size_p50:") for issue in issues)
            and float(np.quantile(lengths, 0.5)) < prepared.region_size
        )


def select_triangulation(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    started: float,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    /,
) -> tuple[
    PreparedPlanarDomain, triangulation_owner.ConstrainedDelaunayTriangulation, int
]:
    """Choose once under the same request budget, retaining the selected ancestry.

    The plan retains its established preparation identity. A necessary median
    repair has its own preparation identity, emitted by execution's source and
    meshing stages; identical source controls do not imply identical plan IDs.
    """
    try:
        with scope.host_workspace() as initial_storage:
            triangulation, inserted = triangulate(prepared, started, specification)
            initial_storage.retain_owner(triangulation)
            repair = _median_repair_needed(prepared, specification, triangulation, scope)
        if not repair:
            workspace.retain_owner(triangulation)
            return prepared, triangulation, inserted
        # Closing its workspace and dropping the last local reference releases
        # the obsolete native/NumPy buffers before preparing another complex.
        del triangulation
        check_deadline(started, specification.limits, MeshingStageKind.SURFACE_MESHING)
        spacing = prepared.region_size / math.sqrt(2.0)
        # One extra piece per edge conservatively covers final-bit norm rounding.
        pieces = sum(
            math.ceil(math.hypot(float(dx), float(dy)) / spacing) + 1
            for dx, dy in prepared.source_directions
        )
        vertices, edges = prepared.source_points.shape[0], prepared.source_edges.shape[0]
        preparation_work = vertices + edges + pieces * (8 + pieces.bit_length())
        preparation_bytes = 512 + 256 * (vertices + edges + pieces)
        with scope.batch(
            scope.remaining().remaining_work_units, bytes_upper=preparation_bytes
        ) as ticket:
            if preparation_work > ticket.remaining_work_units:
                ticket.refuse(preparation_work)
            ticket.consume(preparation_work)
            with ticket.native_target():
                prepared = PreparedPlanarDomain(source, specification, balanced=True)
        workspace.retain_owner(prepared)
        check_deadline(started, specification.limits, MeshingStageKind.SURFACE_MESHING)
        triangulation, inserted = triangulate(
            prepared, started, specification, distribution_sites=True
        )
        workspace.retain_owner(triangulation)
        return prepared, triangulation, inserted
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        raise _capacity_failure(error, specification.limits) from error


@checked
def constraint_coordinate_source(
    source: sources_owner.NativePlanarSource,
    prepared: PreparedPlanarDomain,
    triangulation: triangulation_owner.ConstrainedDelaunayTriangulation,
    used: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactPlcCellGeometrySource, np.ndarray, np.ndarray]:
    """Compose source-edge rational ancestry and certify their carrier deviations."""
    count = triangulation.points.shape[0]
    strata = np.zeros(count, dtype=np.int64)
    rows = np.full(count, -1, dtype=np.int64)
    weights = [Fraction(0) for _ in range(count)]
    generated = np.flatnonzero(prepared.point_sources >= 0)
    strata[generated] = 1
    rows[generated] = prepared.point_sources[generated]
    for vertex in generated.tolist():
        weights[vertex] = Fraction(float(prepared.point_parameters[vertex]))

    def endpoint_parameter(vertex: int, edge: int) -> Fraction:
        if rows[vertex] == edge:
            return weights[vertex]
        first, last = prepared.source_edges[edge].tolist()
        if vertex == first:
            return Fraction(0)
        if vertex == last:
            return Fraction(1)
        raise ValueError(
            f"constraint.ancestry must bind source edge {edge}; got vertex {vertex}"
        )

    for vertex in range(triangulation.input_point_count, count):
        segment = int(triangulation.constraint_sources[vertex])
        if segment < 0:
            continue
        first, last = triangulation.constraint_parents[vertex].tolist()
        if not 0 <= first < vertex or not 0 <= last < vertex:
            raise ValueError(
                f"constraint.ancestry must be acyclic; got vertex {vertex} parents {(first, last)}"
            )
        edge = int(prepared.segment_sources[segment])
        ratio = Fraction(float(triangulation.constraint_ratios[vertex]))
        lower, upper = endpoint_parameter(first, edge), endpoint_parameter(last, edge)
        parameter = lower + ratio * (upper - lower)
        if not 0 <= ratio <= 1 or not min(lower, upper) <= parameter <= max(lower, upper):
            raise ValueError(
                f"constraint.ancestry must stay in its source interval; got vertex {vertex}"
            )
        strata[vertex], rows[vertex], weights[vertex] = 1, edge, parameter
    parameters = np.zeros((used.size, 2), dtype=np.float64)
    parameters[:, 0] = np.asarray(
        [float(weights[index]) for index in used], dtype=np.float64
    )
    ratios = tuple(
        ((weights[index].numerator, weights[index].denominator), (0, 1))
        for index in used.tolist()
    )
    authority = ExactPlcCellGeometrySource(
        prepared.source_points,
        np.empty((0, 3), dtype=np.int64),
        prepared.source_edges,
        strata[used],
        rows[used],
        parameters,
        domain_source_id=source.source_id,
        domain_source_revision=source.source_revision,
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.arange(prepared.source_edges.shape[0], dtype=np.int64),
        source_segment_bounds=prepared.source_bounds,
        vertex_parameter_ratios=ratios,
        budget=budget,
    )
    carriers = np.asarray(triangulation.points, dtype=np.float64)[used]
    exact_source_points = rational_points(prepared.source_points, budget=budget)
    achieved = np.zeros(prepared.source_edges.shape[0], dtype=np.float64)
    carrier_deviations = np.zeros(used.size, dtype=np.float64)
    for compact, vertex in enumerate(used.tolist()):
        edge = int(rows[vertex])
        if edge < 0:
            continue
        first, last = prepared.source_edges[edge].tolist()
        exact = tuple(
            a + weights[vertex] * (b - a)
            for a, b in zip(
                exact_source_points[first], exact_source_points[last], strict=True
            )
        )
        carrier = tuple(Fraction(float(value)) for value in carriers[compact])
        difference = tuple(a - b for a, b in zip(exact, carrier, strict=True))
        bound = fidelity_norm_bound(
            tuple(({(0,): value} for value in difference)), 1, budget=budget
        )
        carrier_deviations[compact] = bound
        achieved[edge] = max(achieved[edge], bound)
        if bound > prepared.source_bounds[edge]:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"source.edge_deviation must respect its source-fidelity bound; got edge {edge} deviation {bound}",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                entity_ids=(edge,),
                requested=(
                    (
                        f"source:edge:{edge}:maximum_deviation",
                        float(prepared.source_bounds[edge]),
                    ),
                ),
                achieved=((f"source:edge:{edge}:certified_deviation", bound),),
            )
    authority.prepare(carriers, budget=budget)
    return authority, achieved, carrier_deviations


@checked
def source_carrier_evidence(
    prepared: PreparedPlanarDomain,
    authority: ExactPlcCellGeometrySource,
    vertex_ids: np.ndarray,
    carrier_deviations: np.ndarray,
    source_deviations: np.ndarray,
    /,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    """Bind source-edge and per-carrier certified deviations to their requested bounds."""
    requested = [
        (f"source:edge:{edge}:maximum_deviation", float(bound))
        for edge, bound in enumerate(prepared.source_bounds)
    ]
    achieved = [
        (f"source:edge:{edge}:certified_deviation", float(bound))
        for edge, bound in enumerate(source_deviations)
    ]
    for vertex, edge in enumerate(np.asarray(authority.vertex_rows, dtype=np.int64)):
        if edge < 0:
            continue
        key = f"source:edge:{edge}:vertex:{vertex_ids[vertex]}"
        requested.append(
            (f"{key}:maximum_deviation", float(prepared.source_bounds[edge]))
        )
        achieved.append((f"{key}:certified_deviation", float(carrier_deviations[vertex])))
    return tuple(requested), tuple(achieved)
