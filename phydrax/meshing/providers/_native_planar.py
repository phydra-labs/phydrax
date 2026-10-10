#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native planar route: graded size fields and refined constrained Delaunay.

Uniform size controls scoped to the planar region (``region:0``) or to source
edges (boundary/embedded patches) are resolved through `resolve_size_controls`
into one size field ``h(x) = min(h_region, min_e h_e + (g - 1) d_e(x))`` where
``d_e`` is the exact distance to the controlled source edge ``e`` and ``g`` the
smallest declared growth rate, so ``h`` is ``(g - 1)``-Lipschitz by
construction. Every source edge (region loops and embedded segments) is bisected
once until each piece is no longer than ``h`` at its ends and midpoint; the
split chains are shared by construction rather than welded afterwards. The
exact meshcore constrained Delaunay triangulation then removes holes and
exterior, Ruppert/Chew refinement enforces the area and minimum-angle
requests, and free edges longer than the local target are split until the
field is met within the work and scratch budgets. Publication certifies global
embedding and exact coverage of the declared region loops (the split
constraint polygon) together with the region measure.
"""

from __future__ import annotations

import math
from time import monotonic
from typing import final

import equinox as eqx
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh
from ...geometry import ConstrainedDelaunayTriangulation
from ...geometry._mesh_certificates import _fidelity_norm_bound, PiecewiseLinearDomain
from ...geometry._planar_coverage import rational_points
from .._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._canonical import canonicalize_cell_mesh
from .._certification import MeshCertificationSchedule
from .._contracts import (
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
)
from .._controls import FeatureKind
from .._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed
from .._organization import MeshLabel, MeshPatch
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    resolve_size_controls,
    SizeCompliancePolicy,
    SizeControlStrength,
    SizeFieldDomain,
    UniformSizeControl,
)
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from ._native_publication import (
    check_deadline,
    NativeCertificationRequest,
    publish_native_result,
    simplex_entity_limits,
    unique_edges,
)
from ._native_sources import NativePlanarSource, source_entity_id


# Point-by-segment distance entries evaluated per chunk of size-field queries.
_FIELD_WORKING_ENTRIES = 1 << 20
# Host bytes one CDT input point costs across one rebuild: its coordinates,
# about two triangles of vertex, neighbor and segment-id rows, and the edge
# table built from them.
_CDT_BYTES_PER_POINT = 16 + 2 * 3 * 3 * 8 + 3 * 2 * 8
# Admissible distance, in units of eps * coordinate scale, of a constraint split
# point from its source edge: each nested midpoint split adds one rounding.
_CONSTRAINT_ROUNDING = 1024.0


def _source_arrays(
    source: NativePlanarSource, /
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


def _size_control_issues(
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
            issues.append("size controls on authoritative planar geometry entities")
        elif control_scope.entity_dimension == 2 and np.array_equal(identifiers, (0,)):
            region_scoped = True
        elif control_scope.entity_dimension != 1 or np.any(identifiers >= edge_count):
            issues.append("size controls scoped to region:0 or to source edges")
    if not region_scoped:
        issues.append("a size control on the planar region")
    return issues


def planar_support_issues(
    source: NativePlanarSource, specification: SurfaceMeshingSpec, /
) -> list[str]:
    """Physical requests of one planar specification this route cannot enforce."""

    unsupported: list[str] = []
    target = specification.target
    families = target.cell_families
    if set((*families.required, *families.preferred)) != {"triangle"}:
        unsupported.append("a triangular surface target")
    if target.ambient_dimension != 2 or target.geometry_order != 1:
        unsupported.append("affine planar triangles in ambient dimension two")
    if families.allowed_transitions or families.allow_mixed:
        unsupported.append("mixed-cell transition policies")
    if not np.array_equal(np.asarray(specification.scope.entity_ids), (0,)):
        unsupported.append("the single planar region entity region:0")
    if (
        specification.scope.entity_dimension != 2
        or specification.scope.entity_kind is not MeshingEntityKind.GEOMETRY
    ):
        unsupported.append("the authoritative planar geometry region scope")
    vertices, edges, _ = _source_arrays(source)
    unsupported.extend(_size_control_issues(specification, edges.shape[0]))
    counts = {0: vertices.shape[0], 1: edges.shape[0]}
    for feature in specification.protected_features:
        dimension = feature.scope.entity_dimension
        identifiers = np.asarray(feature.scope.entity_ids)
        if (feature.scope.source_id, feature.scope.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            unsupported.append("protected features bound to the planar source revision")
        elif feature.scope.entity_kind is not MeshingEntityKind.GEOMETRY:
            unsupported.append(
                "protected features on authoritative planar geometry entities"
            )
        elif feature.feature_kind not in (FeatureKind.CORNER, FeatureKind.CURVE):
            unsupported.append(f"{feature.feature_kind.value} protected features")
        elif dimension != (0 if feature.feature_kind is FeatureKind.CORNER else 1):
            unsupported.append(
                "protected feature kinds matching planar source dimensions"
            )
        elif np.any(identifiers >= counts[dimension]):
            unsupported.append("protected features outside the planar source entities")
    if specification.region_controls:
        unsupported.append("region controls")
    if specification.patch_controls:
        unsupported.append("patch/interface controls")
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")
    return unsupported


def _interior_point(points: np.ndarray, loop: np.ndarray, /) -> np.ndarray:
    """A point strictly inside one simple loop: a centroid of its exact CDT."""

    local = np.arange(loop.size, dtype=np.int64)
    inside = ConstrainedDelaunayTriangulation(
        points[loop], np.stack((local, np.roll(local, -1)), axis=1)
    )
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
    live in different entity sets) through the canonical size resolution, which
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


def _split_constraints(
    vertices: np.ndarray,
    edges: np.ndarray,
    region_size: float,
    patch_segments: np.ndarray,
    patch_sizes: np.ndarray,
    slope: float,
    limits: MeshingLimits,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bisect source edges to the size field: points, pieces, source edges."""

    start = vertices[edges[:, 0]]
    direction = vertices[edges[:, 1]] - start
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
                "Planar constraint discretization exceeds the vertex budget.",
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
    return points, np.stack((first, last), axis=1), owner


def _loop_signs(source: NativePlanarSource, /) -> np.ndarray:
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
    ``patch_segments``/``patch_sizes`` are the size-controlled source edges and
    their resolved sizes; ``region_size`` the resolved region size and
    ``gradation_slope`` the declared growth rate minus one (``inf`` if none).
    """

    points: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    source_starts: np.ndarray
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

    def __init__(
        self, source: NativePlanarSource, specification: SurfaceMeshingSpec, /
    ) -> None:
        limits = specification.limits
        vertices, edges, loops = _source_arrays(source)
        offsets = np.asarray(source.region.loop_offsets, dtype=np.int64)
        region_edges = np.asarray(source.region.edges, dtype=np.int64)
        interior = [
            _interior_point(vertices, region_edges[offsets[k] : offsets[k + 1], 0])
            for k in range(offsets.size - 1)
        ]
        region_size, rows, sizes, slope, resolutions = _resolve_field(
            specification, vertices, edges, interior[0]
        )
        patch_segments = np.stack((vertices[edges[rows, 0]], vertices[edges[rows, 1]]), 1)
        points, segments, sources = _split_constraints(
            vertices, edges, region_size, patch_segments, sizes, slope, limits
        )
        holes = np.asarray(interior[1:], dtype=np.float64).reshape((-1, 2))
        steiner = min(
            limits.maximum_work_units, limits.maximum_vertices - points.shape[0]
        )
        quality = specification.quality_target
        self.points = points
        self.segments = segments
        self.segment_sources = sources
        self.source_starts = vertices[edges[:, 0]]
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
        self,
        source: NativePlanarSource,
        vertices: np.ndarray,
        constrained: np.ndarray,
        constrained_sources: np.ndarray,
        /,
    ) -> tuple[PiecewiseLinearDomain, float, float]:
        """Declared region, constraint rounding, and its admissible bound.

        The declared boundary is the recovered constraint chain on the region
        loops, oriented outward (outer loop counterclockwise, holes clockwise).
        Refinement splits constraints at rounded points, so every chain vertex
        is checked against its source edge: the chain is the source loop up to
        the returned rounding. Embedded constraints separate the region from
        itself, which a declared domain cannot express; they are recovered and
        rounding-checked the same way.
        """

        start = self.source_starts[constrained_sources]
        direction = self.source_directions[constrained_sources]
        squared = np.sum(direction * direction, axis=1)
        rounding = 0.0
        for column in range(2):
            offset = vertices[constrained[:, column]] - start
            along = np.clip(np.sum(offset * direction, axis=1) / squared, 0.0, 1.0)
            distance = np.linalg.norm(offset - along[:, None] * direction, axis=1)
            rounding = max(rounding, float(np.max(distance, initial=0.0)))
        scale = max(float(np.max(np.abs(self.points))), 1.0)
        admissible = _CONSTRAINT_ROUNDING * np.finfo(np.float64).eps * scale
        loop = self.source_loops[constrained_sources]
        on_loop = loop >= 0
        facets = constrained[on_loop]
        # Orient each chain edge along its source edge, then outward.
        along = np.sum(
            (vertices[facets[:, 1]] - vertices[facets[:, 0]]) * direction[on_loop],
            axis=1,
        )
        wanted = np.where(loop[on_loop] == 0, 1, -1)
        forward = (along > 0.0) == (self.loop_signs[loop[on_loop]] == wanted)
        facets = np.where(forward[:, None], facets, facets[:, ::-1])
        domain = PiecewiseLinearDomain(
            vertices,
            facets,
            np.tile(np.asarray(((0, -1),), dtype=np.int64), (facets.shape[0], 1)),
            (source_entity_id(source.source_revision, "region", 0),),
            source_id=source.source_id,
        )
        return domain, rounding, admissible


def _constrained_edges(
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
) -> np.ndarray:
    """Exact known-entity projection with the geometry owner's outward norm bound."""
    original = rational_points(source_points)
    distances = np.empty((points.shape[0],), dtype=np.float64)
    for row, point in enumerate(rational_points(points)):
        index = int(indices[row])
        if dimensions[row] == 0:
            closest = original[index]
        elif dimensions[row] == 1:
            first, last = source_edges[index].tolist()
            start, end = original[first], original[last]
            direction = tuple(b - a for a, b in zip(start, end, strict=True))
            squared = sum(value * value for value in direction)
            parameter = (
                sum((p - a) * d for p, a, d in zip(point, start, direction, strict=True))
                / squared
            )
            parameter = min(max(parameter, 0), 1)
            closest = tuple(
                a + parameter * d for a, d in zip(start, direction, strict=True)
            )
        else:
            raise ValueError(
                "Planar projection residuals require an identified source vertex or edge."
            )
        difference = tuple(p - q for p, q in zip(point, closest, strict=True))
        distances[row] = (
            _fidelity_norm_bound(tuple({(0,): value} for value in difference), 1)
            if any(difference)
            else 0.0
        )
    return distances


def _organization_and_associations(
    mesh: CellMesh,
    source: NativePlanarSource,
    prepared: PreparedPlanarDomain,
    constrained: np.ndarray,
    constrained_sources: np.ndarray,
    input_vertex_rows: np.ndarray,
    /,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshLabel, ...], tuple[GeometryAssociation, ...]]:
    """Emit source authority from original input rows and recovered constraints.

    The CDT preserves input points before Steiner points; compaction's ``used``
    rows therefore identify authored vertices, including embedded endpoints.
    Geometry validates their support, never reconstructs their identity. Planar
    region rows retain geometric dimension two and an explicit REGION role.
    Residuals are outward bounds against the identified source primitives;
    exact flags require actual zero deviation and unique source classification.
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
    source_points, source_edges, _ = _source_arrays(source)
    input_rows = np.asarray(input_vertex_rows)
    if not np.issubdtype(input_rows.dtype, np.integer):
        raise TypeError("The CDT input-row witness must contain integer source rows.")
    input_rows = input_rows.astype(np.int64, copy=False)
    if (
        input_rows.shape != (points.shape[0],)
        or np.any(input_rows < 0)
        or np.unique(input_rows).size != input_rows.size
    ):
        raise ValueError(
            "Planar associations require the actual compacted CDT input-row witness."
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
    )
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
            tuple(source_entity_id(revision, "region", 0) for _ in range(cell_count)),
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
                source_entity_id(revision, "edge", int(edge))
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
                source_entity_id(
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


def _graded_size_compliance(
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
        name: float(np.quantile(lengths, {"p50": 0.5, "p95": 0.95}[name]))
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


def _control_edges(
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


def _planar_compliance(
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
        if not isinstance(control, UniformSizeControl):
            raise TypeError("Admitted planar size controls are UniformSizeControl.")
        rows = _control_edges(control, prepared, edges, local, *constrained)
        control_requested, control_achieved, control_issues = _graded_size_compliance(
            control, specification.size_compliance, lengths[rows], local[rows], gradation
        )
        requested.extend(control_requested)
        achieved.extend(control_achieved)
        issues.extend(control_issues)
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
        # Original source vertices are retained exactly. Split curve vertices
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
            "Planar refinement exceeds its scratch budget.",
            stage=MeshingStageKind.SURFACE_MESHING.value,
            requested=(("maximum_scratch_bytes", limits.maximum_scratch_bytes),),
            achieved=(("scratch_bytes", scratch), ("input_points", count)),
        )


def _triangulate(
    prepared: PreparedPlanarDomain, started: float, specification: SurfaceMeshingSpec, /
) -> tuple[ConstrainedDelaunayTriangulation, int]:
    """Refined CDT whose unconstrained edges respect the local target size.

    Area refinement does not bound edge length, so every unconstrained edge
    longer than the field at its ends or midpoint contributes its midpoint as
    a free input point and the exact CDT is rebuilt; lengths halve per round.
    Inserted points are charged to the work budget together with the
    refinement Steiner points.
    """

    limits = specification.limits
    extra = np.zeros((0, 2), dtype=np.float64)
    while True:
        _refinement_budget(prepared, limits, extra.shape[0])
        triangulation = ConstrainedDelaunayTriangulation(
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
        constrained, _ = _constrained_edges(
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
                "Planar edge-length refinement exceeds its work budget.",
                stage=MeshingStageKind.SURFACE_MESHING.value,
                requested=(("maximum_steiner_points", prepared.maximum_steiner),),
                achieved=(
                    ("maximum_edge_to_local_size", float(np.max(lengths / local))),
                    ("steiner_points", extra.shape[0] + midpoints.shape[0]),
                ),
            )
        extra = np.concatenate((extra, midpoints), axis=0)


def execute_planar_route(
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Triangulate, refine, associate, certify, and publish one planar domain."""

    started = monotonic()
    limits = specification.limits
    phase_start = phase_started(record_phase)
    triangulation, inserted = _triangulate(prepared, started, specification)
    record_elapsed(record_phase, "refinement", phase_start)
    phase_start = phase_started(record_phase)
    evidence = triangulation.evidence
    triangles = np.asarray(triangulation.triangles, dtype=np.int64)
    used, compact = np.unique(triangles, return_inverse=True)
    vertices = np.asarray(triangulation.points, dtype=np.float64)[used]
    triangles = compact.reshape(triangles.shape).astype(np.int32)
    segment_ids = np.asarray(triangulation.segment_ids, dtype=np.int64)
    constrained, constrained_sources = _constrained_edges(
        triangles, segment_ids, prepared.segment_sources
    )
    missing = np.setdiff1d(
        np.arange(prepared.source_directions.shape[0]), constrained_sources
    )
    if missing.size:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Source edges lie outside the meshed planar region.",
            stage=MeshingStageKind.SURFACE_MESHING.value,
            entity_ids=tuple(int(value) for value in missing),
        )
    simplex_entity_limits(
        vertices,
        triangles,
        limits,
        MeshingStageKind.SURFACE_MESHING,
        cell_kind="triangle",
    )
    mesh = canonicalize_cell_mesh(
        CellMesh.from_triangles(
            vertices, triangles, numeric_version=source.source_revision
        )
    )
    record_elapsed(record_phase, "construction", phase_start)
    phase_start = phase_started(record_phase)
    patches, labels, associations = _organization_and_associations(
        mesh, source, prepared, constrained, constrained_sources, used
    )
    domain, rounding, admissible = prepared.domain(
        source, vertices, constrained, constrained_sources
    )
    record_elapsed(record_phase, "geometry_association", phase_start)
    phase_start = phase_started(record_phase)
    compliance = _planar_compliance(
        specification,
        prepared,
        vertices,
        triangles,
        (constrained, constrained_sources),
        (
            evidence.minimum_angle_degrees,
            evidence.steiner_count + inserted,
            evidence.status,
        ),
        (rounding, admissible),
    )
    record_elapsed(record_phase, "compliance", phase_start)
    check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
    construction = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.binding_id,),
            output_ids=(prepared.prepared_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING,
            MeshingStageStatus.PASSED,
            input_ids=(prepared.prepared_id,),
            output_ids=(evidence.evidence_id, mesh.mesh_id),
            created_count=triangles.shape[0],
        ),
    )
    return publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        construction,
        provider,
        {
            "kind": "native-planar-cell-mesh",
            "route": "planar_constrained_delaunay",
            "source": source.binding_id,
            "plan": plan_id,
            "specification": specification.specification_id,
            "triangulation": evidence.evidence_id,
        },
        NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            source.source_id,
            source.source_revision,
            limits,
            domain=domain,
            cell_regions=np.zeros((mesh.blocks[0].cell_count,), dtype=np.int64),
        ),
        # A planar region mesh has a closed boundary cycle; coverage needs it.
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=CellMeshAuditDisposition.REJECT,
        ),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "work_units",
            "geometry_queries",
            "scratch_bytes",
            "wall_seconds",
        ),
        # The CDT rebuilds rather than edits cavities; no cavity is formed.
        unenforced_limits=("cavity_cells",),
        patches=patches,
        labels=labels,
        associations=associations,
        record_phase=record_phase,
    )


__all__ = [
    "PreparedPlanarDomain",
    "execute_planar_route",
    "planar_support_issues",
]
