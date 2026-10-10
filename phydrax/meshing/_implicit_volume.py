#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native certified implicit-domain preparation and source-faithful volumes.

The full-box Delaunay worklist is preparation, never source-domain publication.
The native Delaunay hull contains the *entire* declared query box. Each cell is
classified using a source interval on its bounding box; a vertex, barycenter,
or sign-change sample never establishes cell membership. Unknown cells remain
in the carrier and drive bounded native point insertion. Consequently even a
component invisible to every sampled vertex remains covered by pending work.

Native primal/dual incidence, expansion-bounded circumcenters and hull ray
enclosures support batched certified source root queries. Nominal sphere and
full ring-torus profiles additionally admit actual primal restriction, protected
native region flood and tetrahedral quality refinement. Publication independently
checks approximate-domain coverage, global embedding and degree-one continuous
source projection within the established reach tube. The distinct adaptive
extraction route accepts general sound scalar enclosures only after complete
regular topology discovery, independently bracketed cycle roots, native PLC
coverage and continuous two-sided source bounds. It does not infer signed
distance, reach, hidden-component absence or material interfaces from samples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum
from time import monotonic
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    charge_native_geometry_queries,
    current_native_execution_budget,
    exact_orient3d,
    IncrementalDelaunay3D,
    MeshcoreStatus,
    NativeExecutionBudget,
    restricted_centers_3d,
    restricted_dual_3d,
    restricted_rays_3d,
    TET_MESH_EXUDE_COUNTERS,
    TET_MESH_IMPROVE_COUNTERS,
    TET_MESH_REFINE_COUNTERS,
    TetMesh3D,
    TetMeshRun,
    TRIANGULATION_3D_STATISTICS,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, TetrahedralConnectivity
from ..discretization._cell_geometry_validity import CellValidityPolicy
from ..geometry._certified_implicit import implicit_state_id
from ..geometry._contracts import CompiledGeometry
from ..geometry._mesh_certificates import (
    ImplicitProjectionBoundarySource,
    PiecewiseLinearDomain,
)
from ..geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitSurface,
    discover_adaptive_implicit_surface,
    ImplicitVolumeClass,
    ImplicitVolumeQuery,
)
from ..geometry.implicit._analytic_profile import (
    AnalyticBoundaryCoverCapacityError,
    AnalyticImplicitProfile,
)
from ..geometry.implicit._policy import AdaptiveImplicitSurfacePolicy
from ..geometry.simplicial._topology import TriangleTopology
from ._association import GeometryAssociation, GeometryAssociationKind
from ._canonical import canonicalize_cell_mesh
from ._contracts import (
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from ._controls import FeatureKind
from ._measurements import (
    measure_phase,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._organization import MeshPatch, MeshZone, MeshZoneRole, RegionRole
from ._result import CellMeshingResult, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import resolve_size_controls, SizeFieldDomain, UniformSizeControl
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from ._volume_generation import exude_native_volume, NativeVolumeSchedule


class ImplicitVolumeWorkStop(StrEnum):
    ALL_CLASSIFIED = "all_classified"
    REFINEMENT_LIMIT = "refinement_limit"
    GEOMETRY_QUERY_LIMIT = "geometry_query_limit"
    WORK_LIMIT = "work_limit"
    VERTEX_LIMIT = "vertex_limit"
    NATIVE_REFUSAL = "native_refusal"
    WALL_LIMIT = "wall_limit"


@dataclass(frozen=True, slots=True)
class ImplicitVolumeDomainWorkset:
    """Complete box carrier with source enclosures and explicit pending cells.

    ``classes`` are whole-cell classifications, not material labels. Unqueried
    cells have bounds [-inf, inf] and ``queried=False``. ``inside_tetrahedra``
    names a proven subset, never a mesh of the source domain. Physical controls
    are retained intact in ``specification``; none are claimed satisfied by
    this preparation. The coverage statement is only relative to query.domain,
    not a claim that the source has no components outside that declared box.

    Standalone preparation enforces vertex, cell, cavity, insertion work and
    source-query bounds. An actual ambient native scope additionally owns
    native initialization, managed storage and within-native deadlines.
    Unmanaged host/device/compiler work, storage and nonpreemptible phases
    remain explicit; this carrier is never eligible for publication by itself.
    """

    query: ImplicitVolumeQuery
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    points: np.ndarray
    tetrahedra: np.ndarray
    box_lower: np.ndarray
    box_upper: np.ndarray
    classes: np.ndarray
    value_lower: np.ndarray
    value_upper: np.ndarray
    queried: np.ndarray
    stop: ImplicitVolumeWorkStop
    geometry_queries: int
    refinement_rounds: int
    native_statistics: tuple[tuple[str, int], ...]
    unenforced_limits: tuple[str, ...]
    input_id: str
    workset_id: str

    @property
    def inside_tetrahedra(self) -> np.ndarray:
        """Rows whose entire physical tetrahedron is strictly inside."""
        return np.flatnonzero(self.classes == int(ImplicitVolumeClass.INSIDE))

    @property
    def unresolved_tetrahedra(self) -> np.ndarray:
        """All potentially intersected cells, including unqueried cells."""
        return np.flatnonzero(self.classes == int(ImplicitVolumeClass.UNKNOWN))

    def require_current(
        self, query: ImplicitVolumeQuery, specification: VolumeMeshingSpec, /
    ) -> None:
        """Refuse reuse after source state, domain or physical controls change."""
        if _input_id(query, specification, self.schedule) != self.input_id:
            raise ValueError("Implicit volume workset source or request is stale.")


def _input_id(
    query: ImplicitVolumeQuery,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "implicit-volume-domain-input",
            "query": query.query_id,
            "domain": array_tree_fingerprint(query.domain),
            "kernel": f"{type(query.bounds.kernel).__module__}.{type(query.bounds.kernel).__qualname__}",
            "state": array_tree_fingerprint(query.bounds.state),
            "specification": specification.specification_id,
            "schedule": schedule.schedule_id,
        }
    )


def _classify(
    query: ImplicitVolumeQuery,
    points: np.ndarray,
    cells: np.ndarray,
    budget: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = min(cells.shape[0], budget)
    execution = current_native_execution_budget()
    if execution is not None:
        execution.charge()
        if count > execution.remaining().remaining_geometry_queries:
            # The actual source batch cannot fit: refuse before its host banks.
            execution.charge(geometry_queries=count)
    vertices = points[cells]
    lower = np.min(vertices, axis=1)
    upper = np.max(vertices, axis=1)
    classes = np.full(cells.shape[0], int(ImplicitVolumeClass.UNKNOWN), dtype=np.int64)
    value_lower = np.full(cells.shape[0], -np.inf, dtype=np.float64)
    value_upper = np.full(cells.shape[0], np.inf, dtype=np.float64)
    queried = np.arange(cells.shape[0], dtype=np.int64) < count
    if count:
        result = query.classify_boxes(lower[:count], upper[:count])
        if not result.certified:
            raise ValueError("Implicit domain work requires rigorous value enclosures.")
        classes[:count] = result.classes
        value_lower[:count] = result.value_lower
        value_upper[:count] = result.value_upper
    return lower, upper, classes, value_lower, value_upper, queried


def _publish_workset(
    query: ImplicitVolumeQuery,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    points: np.ndarray,
    cells: np.ndarray,
    classification: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    stop: ImplicitVolumeWorkStop,
    queries: int,
    rounds: int,
    statistics: np.ndarray,
    /,
) -> ImplicitVolumeDomainWorkset:
    input_id = _input_id(query, specification, schedule)
    lower, upper, classes, value_lower, value_upper, queried = classification
    arrays = (points, cells, lower, upper, classes, value_lower, value_upper, queried)
    workset_id = canonical_fingerprint(
        {
            "kind": "implicit-volume-domain-workset",
            "input": input_id,
            "arrays": [array_tree_fingerprint(value) for value in arrays],
            "stop": stop.value,
            "queries": queries,
            "rounds": rounds,
        }
    )
    for value in arrays:
        value.setflags(write=False)
    unenforced = (
        "maximum_edges",
        "maximum_faces",
        "maximum_connectivity_entries",
        "maximum_data_bytes",
        "maximum_scratch_bytes",
        "maximum_work_units:unmetered_host_device_compiler",
        "maximum_wall_seconds:nonpreemptible_host_device_compiler",
    )
    if current_native_execution_budget() is None:
        unenforced += (
            "maximum_work_units:native_initialization",
            "maximum_wall_seconds:within_native_batch",
        )
    return ImplicitVolumeDomainWorkset(
        query,
        specification,
        schedule,
        *arrays,
        stop,
        queries,
        rounds,
        tuple(
            (name, int(value))
            for name, value in zip(TRIANGULATION_3D_STATISTICS, statistics, strict=True)
        ),
        unenforced,
        input_id,
        workset_id,
    )


def prepare_implicit_volume_domain(
    query: ImplicitVolumeQuery,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
) -> ImplicitVolumeDomainWorkset:
    """Refine a full-box native Delaunay workset using whole-cell intervals.

    Barycenters are insertion *sites*, never classification samples. Unknown
    tetrahedra are retained even when native insertion or a budget refuses
    further progress. No caller may turn this preparation into a successful
    implicit volume result without separately establishing restricted-boundary,
    source-domain coverage, feature and material-control obligations.
    """
    if not isinstance(query, ImplicitVolumeQuery):
        raise TypeError("query must be ImplicitVolumeQuery.")
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    if not isinstance(schedule, NativeVolumeSchedule):
        raise TypeError("schedule must be NativeVolumeSchedule.")
    if not query.certified:
        raise ValueError("Implicit domain work requires rigorous value enclosures.")
    domain = np.asarray(query.domain, dtype=np.float64)
    if not np.all(np.isfinite(domain)) or np.any(domain[0] >= domain[1]):
        raise ValueError("Implicit volume domain requires a finite positive box.")
    limits = specification.limits
    if limits.maximum_vertices < 8 or limits.maximum_cells < 6:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Implicit full-enclosure initialization requires eight vertices and six cells.",
            stage="implicit_domain_preparation",
        )
    corners = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ),
        dtype=np.int64,
    )
    points = np.where(corners, domain[1], domain[0])
    started = monotonic()
    state = IncrementalDelaunay3D(
        points,
        max_vertices=limits.maximum_vertices,
        max_tetrahedra=limits.maximum_cells,
        max_cavity=limits.maximum_cavity_cells,
    )
    queries = 0
    rounds = 0
    stop = ImplicitVolumeWorkStop.REFINEMENT_LIMIT
    try:
        while True:
            points, cells, _, _, _ = state.finalize()
            statistics = state.statistics()
            classification = _classify(
                query, points, cells, limits.maximum_geometry_queries - queries
            )
            queries += int(np.count_nonzero(classification[5]))
            unknown = np.flatnonzero(
                classification[2] == int(ImplicitVolumeClass.UNKNOWN)
            )
            if not unknown.size:
                stop = ImplicitVolumeWorkStop.ALL_CLASSIFIED
                break
            if queries >= limits.maximum_geometry_queries:
                stop = ImplicitVolumeWorkStop.GEOMETRY_QUERY_LIMIT
                break
            if rounds >= schedule.refinement_rounds:
                break
            if monotonic() - started >= limits.maximum_wall_seconds:
                stop = ImplicitVolumeWorkStop.WALL_LIMIT
                break
            remaining_work = limits.maximum_work_units - int(statistics[7])
            if remaining_work <= 0:
                stop = ImplicitVolumeWorkStop.WORK_LIMIT
                break
            remaining_vertices = limits.maximum_vertices - points.shape[0]
            if remaining_vertices <= 0:
                stop = ImplicitVolumeWorkStop.VERTEX_LIMIT
                break
            # Limit the submitted batch before native allocation. Native cavity,
            # cell and work refusals leave its previously covering hull intact.
            sites = np.mean(points[cells[unknown[:remaining_vertices]]], axis=1)
            _, status = state.insert(sites, work_limit=remaining_work)
            rounds += 1
            if np.any(status != int(MeshcoreStatus.OK)):
                stop = ImplicitVolumeWorkStop.NATIVE_REFUSAL
                # Reclassify the actual partially committed snapshot.
                points, cells, _, _, _ = state.finalize()
                statistics = state.statistics()
                classification = _classify(
                    query, points, cells, limits.maximum_geometry_queries - queries
                )
                queries += int(np.count_nonzero(classification[5]))
                break
        return _publish_workset(
            query,
            specification,
            schedule,
            points,
            cells,
            classification,
            stop,
            queries,
            rounds,
            statistics,
        )
    finally:
        state.close()


class ImplicitDualIntervalClass(IntEnum):
    """Scientific verdict on one terminal interval of a numerical dual."""

    EXCLUDED = 0
    UNIQUE_ROOT = 1
    UNRESOLVED = 2


@dataclass(frozen=True, slots=True)
class ImplicitDualRootWorkset:
    """An exhaustive terminal partition of every supplied numerical segment.

    Each UNIQUE_ROOT interval has rigorous endpoint sign separation and a
    strictly signed enclosed directional derivative for every line through the
    supplied endpoint bounds. Each such line contains exactly one source root.
    EXCLUDED intervals have a value enclosure excluding zero. All other
    intervals remain UNRESOLVED, including tangencies and resource exhaustion.
    A caller using exact-center bounds can certify finite-dual roots; it must
    independently establish that those bounds enclose the actual dual ends.
    """

    endpoints: np.ndarray
    endpoint_bounds: np.ndarray
    segment_ids: np.ndarray
    parameter_intervals: np.ndarray
    classes: np.ndarray
    value_lower: np.ndarray
    value_upper: np.ndarray
    derivative_lower: np.ndarray
    derivative_upper: np.ndarray
    endpoint_value_lower: np.ndarray
    endpoint_value_upper: np.ndarray
    geometry_queries: int
    query_budget_exhausted: bool
    maximum_depth_reached: bool
    query_id: str
    workset_id: str


def _line_boxes(
    endpoint_bounds: np.ndarray, segments: np.ndarray, parameters: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Outward boxes of all affine lines through enclosed endpoints."""
    first_lower = endpoint_bounds[segments, 0, 0]
    first_upper = endpoint_bounds[segments, 0, 1]
    second_lower = endpoint_bounds[segments, 1, 0]
    second_upper = endpoint_bounds[segments, 1, 1]
    difference_lower = np.nextafter(second_lower - first_upper, -np.inf)
    difference_upper = np.nextafter(second_upper - first_lower, np.inf)
    parameter = parameters[..., None]
    lower = np.nextafter(
        first_lower[:, None]
        + np.nextafter(difference_lower[:, None] * parameter, -np.inf),
        -np.inf,
    )
    upper = np.nextafter(
        first_upper[:, None]
        + np.nextafter(difference_upper[:, None] * parameter, np.inf),
        np.inf,
    )
    lower = np.where(parameter == 0.0, first_lower[:, None], lower)
    upper = np.where(parameter == 0.0, first_upper[:, None], upper)
    lower = np.where(parameter == 1.0, second_lower[:, None], lower)
    upper = np.where(parameter == 1.0, second_upper[:, None], upper)
    return lower, upper, difference_lower, difference_upper


def _length_upper(displacement: np.ndarray, /) -> np.ndarray:
    squared = np.nextafter(displacement * displacement, np.inf)
    total = np.zeros(displacement.shape[0], dtype=np.float64)
    for axis in range(3):
        total = np.nextafter(total + squared[:, axis], np.inf)
    return np.nextafter(np.sqrt(total), np.inf)


def _dual_interval_bounds(
    query: ImplicitVolumeQuery,
    endpoint_bounds: np.ndarray,
    segments: np.ndarray,
    parameters: np.ndarray,
    /,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    lower, upper, direction_lower, direction_upper = _line_boxes(
        endpoint_bounds, segments, parameters
    )
    box = query.bounds.boxes(np.min(lower, axis=1), np.max(upper, axis=1))
    point_bounds = query.classify_boxes(lower.reshape((-1, 3)), upper.reshape((-1, 3)))
    endpoint_lower = np.asarray(point_bounds.value_lower).reshape((-1, 2))
    endpoint_upper = np.asarray(point_bounds.value_upper).reshape((-1, 2))
    products = np.stack(
        (
            box.gradient_lower * direction_lower,
            box.gradient_lower * direction_upper,
            box.gradient_upper * direction_lower,
            box.gradient_upper * direction_upper,
        )
    )
    product_lower = np.nextafter(np.min(products, axis=0), -np.inf)
    product_upper = np.nextafter(np.max(products, axis=0), np.inf)
    # Accumulate outward per addition, rather than widening an ordinary sum.
    derivative_lower = np.zeros(segments.shape[0], dtype=np.float64)
    derivative_upper = np.zeros(segments.shape[0], dtype=np.float64)
    for axis in range(3):
        derivative_lower = np.nextafter(
            derivative_lower + product_lower[:, axis], -np.inf
        )
        derivative_upper = np.nextafter(derivative_upper + product_upper[:, axis], np.inf)
    if not query.bounds.gradient_rigorous:
        derivative_lower.fill(-np.inf)
        derivative_upper.fill(np.inf)
    regular = (derivative_lower > 0.0) | (derivative_upper < 0.0)
    # Axis boxes lose the correlation of an oblique dual's coordinates. When
    # its complete directional derivative has one strict sign, all field
    # values lie between the enclosed endpoint values for every allowed dual.
    # This is a source interval proof, not sign sampling or empty-box inference.
    value_lower = np.where(
        regular,
        np.maximum(box.value_lower, np.min(endpoint_lower, axis=1)),
        box.value_lower,
    )
    value_upper = np.where(
        regular,
        np.minimum(box.value_upper, np.max(endpoint_upper, axis=1)),
        box.value_upper,
    )
    return (
        value_lower,
        value_upper,
        derivative_lower,
        derivative_upper,
        endpoint_lower,
        endpoint_upper,
        _length_upper(
            np.nextafter(np.max(upper, axis=1) - np.min(lower, axis=1), np.inf)
        ),
    )


def _merge_dual_root_runs(
    query: ImplicitVolumeQuery,
    enclosures: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    spatial_tolerance: float,
    remaining_queries: int,
    /,
) -> tuple[tuple[np.ndarray, ...], int, bool]:
    """Prove roots jointly across shared uncertain subdivision endpoints.

    A root exactly on a bisection plane need not have a strictly signed value
    at that plane. Adjacent unresolved pieces are joined and re-enclosed;
    strict outer signs and a monotone derivative establish the same unique
    root without requiring a pointwise exact-zero assertion.
    """
    segments, parameters, classes = arrays[:3]
    candidates: list[tuple[int, int]] = []
    start = 0
    while start < segments.size:
        if classes[start] != int(ImplicitDualIntervalClass.UNRESOLVED):
            start += 1
            continue
        end = start + 1
        while (
            end < segments.size
            and segments[end] == segments[start]
            and classes[end] == int(ImplicitDualIntervalClass.UNRESOLVED)
            and parameters[end - 1, 1] == parameters[end, 0]
        ):
            end += 1
        if end - start > 1:
            candidates.append((start, end))
        start = end
    count = min(len(candidates), remaining_queries // 3)
    if not count:
        return arrays, 0, bool(candidates)
    chosen = candidates[:count]
    ids = np.asarray([segments[start] for start, _ in chosen], dtype=np.int64)
    spans = np.asarray(
        [(parameters[start, 0], parameters[end - 1, 1]) for start, end in chosen],
        dtype=np.float64,
    )
    bounds = _dual_interval_bounds(query, enclosures, ids, spans)
    value_low, value_high, derivative_low, derivative_high, low, high, width = bounds
    regular = (derivative_low > 0.0) | (derivative_high < 0.0)
    crossing = ((high[:, 0] < 0.0) & (low[:, 1] > 0.0)) | (
        (low[:, 0] > 0.0) & (high[:, 1] < 0.0)
    )
    unique = regular & crossing & (width <= spatial_tolerance)
    excluded = (value_low > 0.0) | (value_high < 0.0)
    merged_classes = np.where(
        excluded,
        int(ImplicitDualIntervalClass.EXCLUDED),
        np.where(
            unique,
            int(ImplicitDualIntervalClass.UNIQUE_ROOT),
            int(ImplicitDualIntervalClass.UNRESOLVED),
        ),
    ).astype(np.int64)
    keep = np.ones(segments.size, dtype=np.bool_)
    for start, end in chosen:
        keep[start:end] = False
    merged = (ids, spans, merged_classes, *bounds[:6])
    result = tuple(
        np.concatenate((old[keep], new)) for old, new in zip(arrays, merged, strict=True)
    )
    order = np.lexsort((result[1][:, 0], result[0]))
    return tuple(value[order] for value in result), 3 * count, count < len(candidates)


def isolate_implicit_dual_roots(
    query: ImplicitVolumeQuery,
    endpoints: ArrayLike,
    /,
    *,
    spatial_tolerance: float,
    maximum_depth: int,
    maximum_geometry_queries: int,
    endpoint_bounds: ArrayLike | None = None,
) -> ImplicitDualRootWorkset:
    """Bound roots on all numerical duals, without sampled-empty exclusions.

    Every split replaces one interval with its two closed children; terminal
    intervals cover [0,1] for every input segment, even on exhausted budgets.
    No singular, exact-zero or multiple-root interval is silently discarded.
    """
    if not isinstance(query, ImplicitVolumeQuery):
        raise TypeError("query must be ImplicitVolumeQuery.")
    if not query.certified:
        raise ValueError("Dual root isolation requires rigorous value enclosures.")
    points = np.asarray(endpoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (2, 3) or not np.all(np.isfinite(points)):
        raise ValueError("endpoints must be finite with shape (segments, 2, 3).")
    if not np.isfinite(spatial_tolerance) or spatial_tolerance <= 0.0:
        raise ValueError("spatial_tolerance must be positive and finite.")
    for value, name in (
        (maximum_depth, "maximum_depth"),
        (maximum_geometry_queries, "maximum_geometry_queries"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
        minimum = 0 if name == "maximum_geometry_queries" else 1
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
    enclosures = (
        np.stack((points, points), axis=2)
        if endpoint_bounds is None
        else np.asarray(endpoint_bounds, dtype=np.float64)
    )
    if (
        enclosures.shape != (points.shape[0], 2, 2, 3)
        or not np.all(np.isfinite(enclosures))
        or np.any(enclosures[:, :, 0] > enclosures[:, :, 1])
        or np.any(points < enclosures[:, :, 0])
        or np.any(points > enclosures[:, :, 1])
    ):
        raise ValueError(
            "endpoint_bounds must be finite enclosing (segments, 2, 2, 3) intervals."
        )
    segments = np.arange(points.shape[0], dtype=np.int64)
    parameters = np.tile(np.asarray((0.0, 1.0), dtype=np.float64), (segments.size, 1))
    depth = np.zeros(segments.size, dtype=np.int64)
    terminals: list[tuple[np.ndarray, ...]] = []
    queries = 0
    exhausted = False
    depth_reached = False
    while segments.size:
        count = min(segments.size, (maximum_geometry_queries - queries) // 3)
        if count < segments.size:
            pending = segments[count:]
            terminals.append(
                (
                    pending,
                    parameters[count:],
                    np.full(
                        pending.size,
                        int(ImplicitDualIntervalClass.UNRESOLVED),
                        dtype=np.int64,
                    ),
                    np.full(pending.size, -np.inf, dtype=np.float64),
                    np.full(pending.size, np.inf, dtype=np.float64),
                    np.full(pending.size, -np.inf, dtype=np.float64),
                    np.full(pending.size, np.inf, dtype=np.float64),
                    np.full((pending.size, 2), -np.inf, dtype=np.float64),
                    np.full((pending.size, 2), np.inf, dtype=np.float64),
                )
            )
            exhausted = True
        if not count:
            break
        segments, parameters, depth = segments[:count], parameters[:count], depth[:count]
        bounds = _dual_interval_bounds(query, enclosures, segments, parameters)
        queries += 3 * count
        value_lower, value_upper, derivative_lower, derivative_upper, low, high, width = (
            bounds
        )
        excluded = (value_lower > 0.0) | (value_upper < 0.0)
        regular = (derivative_lower > 0.0) | (derivative_upper < 0.0)
        crossing = ((high[:, 0] < 0.0) & (low[:, 1] > 0.0)) | (
            (low[:, 0] > 0.0) & (high[:, 1] < 0.0)
        )
        small = width <= 0.5 * spatial_tolerance
        isolated = regular & crossing & small
        stop = excluded | isolated | (depth >= maximum_depth)
        classes = np.where(
            excluded,
            int(ImplicitDualIntervalClass.EXCLUDED),
            np.where(
                isolated,
                int(ImplicitDualIntervalClass.UNIQUE_ROOT),
                int(ImplicitDualIntervalClass.UNRESOLVED),
            ),
        ).astype(np.int64)
        depth_reached |= bool(np.any((depth >= maximum_depth) & ~excluded & ~isolated))
        if np.any(stop):
            terminals.append(
                (
                    segments[stop],
                    parameters[stop],
                    classes[stop],
                    *(value[stop] for value in bounds[:6]),
                )
            )
        pending_segments = segments[~stop]
        pending_parameters = parameters[~stop]
        middle = np.mean(pending_parameters, axis=1)
        segments = np.repeat(pending_segments, 2)
        parameters = np.stack(
            (
                np.stack((pending_parameters[:, 0], middle), axis=1),
                np.stack((middle, pending_parameters[:, 1]), axis=1),
            ),
            axis=1,
        ).reshape((-1, 2))
        depth = np.repeat(depth[~stop] + 1, 2)
    shapes = ((0,), (0, 2), (0,), (0,), (0,), (0,), (0,), (0, 2), (0, 2))
    arrays = tuple(
        np.concatenate([terminal[column] for terminal in terminals])
        if terminals
        else np.empty(shape, dtype=np.int64 if column in (0, 2) else np.float64)
        for column, shape in enumerate(shapes)
    )
    order = np.lexsort((arrays[1][:, 0], arrays[0]))
    arrays = tuple(value[order] for value in arrays)
    arrays, merge_queries, merge_exhausted = _merge_dual_root_runs(
        query, enclosures, arrays, spatial_tolerance, maximum_geometry_queries - queries
    )
    queries += merge_queries
    exhausted |= merge_exhausted
    identity = canonical_fingerprint(
        {
            "kind": "implicit-dual-roots",
            "query": query.query_id,
            "source_state": array_tree_fingerprint(query.bounds.state),
            "endpoints": array_tree_fingerprint(points),
            "endpoint_bounds": array_tree_fingerprint(enclosures),
            "terminals": [array_tree_fingerprint(value) for value in arrays],
            "spatial_tolerance": spatial_tolerance,
            "maximum_depth": maximum_depth,
            "maximum_geometry_queries": maximum_geometry_queries,
        }
    )
    points = points.copy()
    enclosures = enclosures.copy()
    for value in (points, enclosures, *arrays):
        value.setflags(write=False)
    return ImplicitDualRootWorkset(
        endpoints=points,
        endpoint_bounds=enclosures,
        segment_ids=arrays[0],
        parameter_intervals=arrays[1],
        classes=arrays[2],
        value_lower=arrays[3],
        value_upper=arrays[4],
        derivative_lower=arrays[5],
        derivative_upper=arrays[6],
        endpoint_value_lower=arrays[7],
        endpoint_value_upper=arrays[8],
        geometry_queries=queries,
        query_budget_exhausted=exhausted,
        maximum_depth_reached=depth_reached,
        query_id=query.query_id,
        workset_id=identity,
    )


@dataclass(frozen=True, slots=True)
class ImplicitRestrictedQueryWorkset:
    """Source-bound native primal/dual incidence and dual/source root proofs.

    Every failed/unknown query remains in ``unresolved_facets``. Hull rays use
    rigorous finite ray covers, not numerical clipping or guessed endpoints.
    This is not a restricted surface or volume mesh; complete facet coverage
    cannot be claimed while any query obligation remains. The domain workset
    retains the exact physical request and all its unapplied controls.
    ``work_units`` sums measured native phase counters only; it does not
    assign cell/ray/query counts a fabricated instruction cost. Source query
    cardinalities remain in ``geometry_queries``. The original ambient
    execution owner, when present, enforces the cumulative allowance.
    """

    domain: ImplicitVolumeDomainWorkset
    facets: np.ndarray
    incident_cells: np.ndarray
    numerical_endpoints: np.ndarray
    numerical_kinds: np.ndarray
    numerical_status: np.ndarray
    center_bounds: np.ndarray
    center_status: np.ndarray
    ray_facets: np.ndarray
    ray_bounds: np.ndarray
    ray_kinds: np.ndarray
    ray_status: np.ndarray
    root_facets: np.ndarray
    roots: ImplicitDualRootWorkset
    unresolved_facets: np.ndarray
    work_units: int
    geometry_queries: int
    workset_id: str


def prepare_implicit_restricted_queries(
    domain: ImplicitVolumeDomainWorkset,
    /,
    *,
    spatial_tolerance: float,
    maximum_depth: int,
    additional_work_units: int = 0,
) -> ImplicitRestrictedQueryWorkset:
    """Recompute exact dual incidence and all dual/source root enclosures.

    Native transactional insertion is owned by domain preparation. This phase
    snapshots its current triangulation, certifies finite Voronoi endpoints
    with expansion-based rational bounds, encloses hull rays through the domain
    exit and executes batched interval roots.
    Numeric clipping never establishes absence of an exact dual/source root.
    """
    if not isinstance(domain, ImplicitVolumeDomainWorkset):
        raise TypeError("domain must be ImplicitVolumeDomainWorkset.")
    domain.require_current(domain.query, domain.specification)
    limits = domain.specification.limits
    if additional_work_units < 0:
        raise ValueError("additional_work_units must be nonnegative.")
    spent = dict(domain.native_statistics)["work"] + additional_work_units
    execution = current_native_execution_budget()
    if execution is not None:
        execution.charge()
    remaining = (
        limits.maximum_work_units - spent
        if execution is None
        else execution.remaining().remaining_work_units
    )
    if remaining <= 0:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Implicit dual preparation has no remaining native work budget.",
            stage="implicit_restricted_queries",
        )
    facets, cells, endpoints, kinds, status, counters = restricted_dual_3d(
        domain.points,
        domain.tetrahedra,
        domain.query.domain,
        limits.maximum_faces,
        remaining,
    )
    center_bounds, center_status = restricted_centers_3d(domain.points, domain.tetrahedra)
    ray_facets = np.flatnonzero(cells[:, 1] < 0)
    ray_bounds, ray_kinds, ray_status = restricted_rays_3d(
        domain.points,
        domain.tetrahedra,
        center_bounds,
        facets[ray_facets],
        cells[ray_facets, 0],
        domain.query.domain,
    )
    finite = cells[:, 1] >= 0
    good = center_status == int(MeshcoreStatus.OK)
    eligible = finite & good[cells[:, 0]] & good[np.maximum(cells[:, 1], 0)]
    excluded = np.zeros(facets.shape[0], dtype=np.bool_)
    excluded[ray_facets] = (ray_status == int(MeshcoreStatus.OK)) & (ray_kinds == 2)
    eligible[ray_facets] = (ray_status == int(MeshcoreStatus.OK)) & (ray_kinds == 1)
    root_facets = np.flatnonzero(eligible)
    all_bounds = np.zeros((facets.shape[0], 2, 2, 3), dtype=np.float64)
    all_bounds[finite] = center_bounds[cells[finite]]
    all_bounds[ray_facets] = ray_bounds
    endpoint_bounds = all_bounds[eligible]
    representative = 0.5 * endpoint_bounds[:, :, 0] + 0.5 * endpoint_bounds[:, :, 1]
    roots = isolate_implicit_dual_roots(
        domain.query,
        representative,
        spatial_tolerance=spatial_tolerance,
        maximum_depth=maximum_depth,
        maximum_geometry_queries=min(
            limits.maximum_geometry_queries - domain.geometry_queries,
            limits.maximum_geometry_queries - domain.geometry_queries
            if execution is None
            else execution.remaining().remaining_geometry_queries,
        ),
        endpoint_bounds=endpoint_bounds,
    )
    pending = roots.classes == int(ImplicitDualIntervalClass.UNRESOLVED)
    unresolved = np.union1d(
        np.flatnonzero(~eligible & ~excluded), root_facets[roots.segment_ids[pending]]
    ).astype(np.int64)
    arrays = (
        facets,
        cells,
        endpoints,
        kinds,
        status,
        center_bounds,
        center_status,
        ray_facets,
        ray_bounds,
        ray_kinds,
        ray_status,
        root_facets,
        unresolved,
    )
    identity = canonical_fingerprint(
        {
            "kind": "implicit-restricted-queries",
            "domain": domain.workset_id,
            "arrays": [array_tree_fingerprint(value) for value in arrays],
            "roots": roots.workset_id,
        }
    )
    for value in arrays:
        value.setflags(write=False)
    return ImplicitRestrictedQueryWorkset(
        domain,
        facets,
        cells,
        endpoints,
        kinds,
        status,
        center_bounds,
        center_status,
        ray_facets,
        ray_bounds,
        ray_kinds,
        ray_status,
        root_facets,
        roots,
        unresolved,
        spent + int(counters[0]),
        domain.geometry_queries + roots.geometry_queries,
        identity,
    )


@dataclass(frozen=True, slots=True)
class RestrictedImplicitTopology:
    """Native flood-classified approximate domain bounded by restricted facets."""

    points: np.ndarray
    tetrahedra: np.ndarray
    boundary_facets: np.ndarray
    boundary_root_facets: np.ndarray
    full_cell_regions: np.ndarray
    source_query_workset_id: str
    topology_id: str


def _restricted_root_sites(
    work: ImplicitRestrictedQueryWorkset, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(
        work.roots.classes == int(ImplicitDualIntervalClass.UNIQUE_ROOT)
    )
    segments = work.roots.segment_ids[rows]
    facets = work.root_facets[segments]
    parameter = np.mean(work.roots.parameter_intervals[rows], axis=1)
    endpoints = work.roots.endpoints[segments]
    sites = (1.0 - parameter[:, None]) * endpoints[:, 0] + parameter[:, None] * endpoints[
        :, 1
    ]
    counts = np.bincount(facets, minlength=work.facets.shape[0])
    return facets, sites, counts


def _boundary_normal_bounds(
    query: ImplicitVolumeQuery, triangles: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose exact oriented primal normals dotted with source gradients."""
    edge_low = np.nextafter(triangles[:, 1:] - triangles[:, :1], -np.inf)
    edge_high = np.nextafter(triangles[:, 1:] - triangles[:, :1], np.inf)

    def product_bounds(
        first_low: np.ndarray,
        first_high: np.ndarray,
        second_low: np.ndarray,
        second_high: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.stack(
            (
                first_low * second_low,
                first_low * second_high,
                first_high * second_low,
                first_high * second_high,
            )
        )
        return (
            np.nextafter(np.min(values, axis=0), -np.inf),
            np.nextafter(np.max(values, axis=0), np.inf),
        )

    normal_low = np.empty((triangles.shape[0], 3), dtype=np.float64)
    normal_high = np.empty_like(normal_low)
    for axis in range(3):
        first = (axis + 1) % 3
        second = (axis + 2) % 3
        low_first, high_first = product_bounds(
            edge_low[:, 0, first],
            edge_high[:, 0, first],
            edge_low[:, 1, second],
            edge_high[:, 1, second],
        )
        low_second, high_second = product_bounds(
            edge_low[:, 0, second],
            edge_high[:, 0, second],
            edge_low[:, 1, first],
            edge_high[:, 1, first],
        )
        normal_low[:, axis] = np.nextafter(low_first - high_second, -np.inf)
        normal_high[:, axis] = np.nextafter(high_first - low_second, np.inf)
    bounds = query.bounds.boxes(np.min(triangles, axis=1), np.max(triangles, axis=1))
    lower = np.zeros(triangles.shape[0], dtype=np.float64)
    upper = np.zeros(triangles.shape[0], dtype=np.float64)
    for axis in range(3):
        low, high = product_bounds(
            normal_low[:, axis],
            normal_high[:, axis],
            bounds.gradient_lower[:, axis],
            bounds.gradient_upper[:, axis],
        )
        lower = np.nextafter(lower + low, -np.inf)
        upper = np.nextafter(upper + high, np.inf)
    return lower, upper


def assemble_restricted_implicit_topology(
    state: IncrementalDelaunay3D,
    work: ImplicitRestrictedQueryWorkset,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> RestrictedImplicitTopology:
    """Protect actual restricted primal faces and flood their two sides natively.

    Whole-cell source enclosures outside the actual certified boundary-field
    band provide interior/exterior witnesses. Cells in that allowed approximation
    band follow the protected primal complex, never a centroid label. Independent
    degree-one source projection and approximate-domain coverage are mandatory
    before publication. Every restricted facet must separate the
    two regions, and every enclosed inside/outside cell must agree with it.
    """
    if work.unresolved_facets.size:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted dual/source intersections remain unresolved.",
            stage="implicit_restricted_topology",
            entity_ids=tuple(int(value) for value in work.unresolved_facets),
        )
    points, cells, _, _, _ = state.finalize()
    if not np.array_equal(points, work.domain.points) or not np.array_equal(
        cells, work.domain.tetrahedra
    ):
        raise ValueError("Restricted query workset binds another triangulation snapshot.")
    _, _, counts = _restricted_root_sites(work)
    if np.any(counts > 1):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A dual intersects the source more than once; native site refinement is required.",
            stage="implicit_restricted_topology",
        )
    selected = np.flatnonzero(counts == 1)
    if not selected.size:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "No restricted boundary exists; complete domain work is not certified empty.",
            stage="implicit_restricted_topology",
        )
    with measure_phase(record_phase, "boundary_recovery"):
        constrained = state.constrain_facets(work.facets[selected], selected)
    if np.any(constrained != int(MeshcoreStatus.OK)):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Native restriction could not protect every queried primal facet.",
            stage="implicit_restricted_topology",
        )
    triangles = points[work.facets[selected]]
    boundary_values = work.domain.query.classify_boxes(
        np.min(triangles, axis=1), np.max(triangles, axis=1)
    )
    if not boundary_values.certified:
        raise ValueError(
            "Restricted source consistency requires rigorous boundary intervals."
        )
    field_band = float(
        np.max(
            np.maximum(
                np.abs(boundary_values.value_lower), np.abs(boundary_values.value_upper)
            )
        )
    )
    # Exact source membership and affine approximation membership can differ
    # inside this measured band. Never alter the original source classes or
    # forgive a contradiction beyond it; the projection proof establishes the
    # continuous correspondence needed for the declared approximation.
    inside = np.flatnonzero(work.domain.value_upper < -field_band)
    outside = np.flatnonzero(work.domain.value_lower > field_band)
    witness_points: list[np.ndarray] = []
    if inside.size:
        witness_points.append(np.mean(points[cells[inside[0]]], axis=0))
    if outside.size:
        exterior = np.mean(points[cells[outside[0]]], axis=0)
    else:
        # A hull-adjacent tetrahedron's axis box can touch the source even
        # when its interior witness point is well outside. A certified point
        # seed drives the protected-complex flood; it never labels the cell
        # or establishes cell conformity from a centroid sign.
        hull_cells = np.unique(work.incident_cells[work.incident_cells[:, 1] < 0, 0])
        candidates = np.mean(points[cells[hull_cells]], axis=1)
        source_points = work.domain.query.classify_boxes(candidates, candidates)
        exterior_rows = np.flatnonzero(source_points.value_lower > field_band)
        if not exterior_rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "No source-enclosed exterior flood witness lies beyond the certified boundary band.",
                stage="implicit_restricted_topology",
            )
        exterior = candidates[exterior_rows[0]]
    if not witness_points:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "A source-enclosed interior cell beyond the measured approximation band is required.",
            stage="implicit_restricted_topology",
        )
    witnesses = np.stack((witness_points[0], exterior))
    with measure_phase(record_phase, "region_classification"):
        status = state.label_regions(witnesses, np.asarray((0, 1), dtype=np.int32))
    if np.any(status != int(MeshcoreStatus.OK)):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "Restricted facets do not separate source-enclosed interior and exterior witnesses.",
            stage="implicit_restricted_topology",
        )
    points, cells, _, _, regions = state.finalize()
    if (
        np.any(regions < 0)
        or np.any(regions[inside] != 0)
        or np.any(regions[outside] != 1)
    ):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "Native restricted flood contradicts complete source enclosures beyond the certified boundary band.",
            stage="implicit_restricted_topology",
        )
    owners = work.incident_cells[selected]
    first_inside = regions[owners[:, 0]] == 0
    second_inside = np.where(
        owners[:, 1] >= 0, regions[np.maximum(owners[:, 1], 0)] == 0, False
    )
    if np.any(first_inside == second_inside):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "A restricted primal facet does not separate the declared source regions.",
            stage="implicit_restricted_topology",
        )
    boundary = work.facets[selected].copy()
    adjacent = np.where(first_inside, owners[:, 0], owners[:, 1])
    apex = np.asarray(
        [
            next(int(value) for value in cells[cell] if value not in face)
            for cell, face in zip(adjacent, boundary, strict=True)
        ],
        dtype=np.int64,
    )
    corners = points[boundary]
    toward_inside = (
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], points[apex]) > 0
    )
    boundary[toward_inside] = boundary[toward_inside][:, (0, 2, 1)]
    used = np.unique(cells[regions == 0])
    reindex = np.full(points.shape[0], -1, dtype=np.int64)
    reindex[used] = np.arange(used.size, dtype=np.int64)
    result_points = points[used]
    result_cells = reindex[cells[regions == 0]]
    result_boundary = reindex[boundary]
    topology = TriangleTopology(result_boundary, num_vertices=result_points.shape[0])
    if not topology.watertight:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Restricted primal facets do not form a closed oriented manifold.",
            stage="implicit_restricted_topology",
        )
    identity = canonical_fingerprint(
        {
            "kind": "restricted-implicit-topology",
            "queries": work.workset_id,
            "points": array_tree_fingerprint(result_points),
            "cells": array_tree_fingerprint(result_cells),
            "boundary": array_tree_fingerprint(result_boundary),
            "regions": array_tree_fingerprint(regions),
        }
    )
    for value in (result_points, result_cells, result_boundary, selected, regions):
        value.setflags(write=False)
    return RestrictedImplicitTopology(
        result_points,
        result_cells,
        result_boundary,
        selected,
        regions,
        work.workset_id,
        identity,
    )


@dataclass(frozen=True, slots=True)
class PreparedImplicitVolume:
    """Qualified regular analytic source and exact physical request binding."""

    profile: AnalyticImplicitProfile
    query: ImplicitVolumeQuery
    specification: VolumeMeshingSpec
    schedule: NativeVolumeSchedule
    target_size: float
    fidelity_tolerance: float
    seed_cover_radius: float
    geometry_query_charge: int
    size_resolution_id: str
    prepared_id: str


def implicit_volume_support_issues(
    specification: VolumeMeshingSpec, /
) -> tuple[str, ...]:
    """Unsupported controls are refused, not omitted from a positive profile."""
    if not isinstance(specification, VolumeMeshingSpec):
        raise TypeError("specification must be VolumeMeshingSpec.")
    issues: list[str] = []
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"tetrahedron"}
        or families.allow_mixed
        or families.allowed_transitions
    ):
        issues.append("conforming affine tetrahedral targets")
    if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
        issues.append("the simplex fill strategy")
    scope = specification.boundary_scope
    if scope.entity_dimension != 2 or not np.array_equal(scope.entity_ids, [0]):
        issues.append("the complete implicit-zero-set boundary entity")
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append("nonuniform native restricted-domain sizing")
        elif not np.array_equal(control.scope.entity_ids, scope.entity_ids):
            issues.append("uniform controls on the entire implicit boundary")
    for feature in specification.protected_features:
        if feature.feature_kind is not FeatureKind.SURFACE or not np.array_equal(
            feature.scope.entity_ids, scope.entity_ids
        ):
            issues.append("sharp-feature/material-junction protected cavity execution")
        elif feature.hard and feature.maximum_deviation == 0.0:
            issues.append("exact curved geometry represented by affine boundary facets")
    if len(specification.region_controls) > 1:
        issues.append("multiple material regions on a single regular analytic source")
    for control in specification.region_controls:
        if (
            not control.meshing_enabled
            or control.scope.entity_dimension != 3
            or not np.array_equal(control.scope.entity_ids, [0])
        ):
            issues.append("the enabled source interior region")
    if specification.hole_seeds:
        issues.append("additional void components not present in the source profile")
    if specification.layer_controls:
        issues.append("boundary layers through the owning layer-core route")
    if specification.periodic_constraints:
        issues.append(
            "periodic restricted-domain construction through the quotient owner"
        )
    return tuple(sorted(set(issues)))


def prepare_restricted_implicit_volume(
    geometry: CompiledGeometry,
    domain: ArrayLike,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedImplicitVolume:
    """Prepare the certified analytic profile without extracting or filling a PLC."""
    issues = implicit_volume_support_issues(specification)
    if issues:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(issues),
            stage="implicit_volume_preparation",
        )
    if not isinstance(schedule, NativeVolumeSchedule):
        raise TypeError("schedule must be NativeVolumeSchedule.")
    _implicit_region_metadata(specification)
    if specification.limits.maximum_geometry_queries < 3:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Analytic source preparation requires three bounded geometry requests.",
            stage="implicit_volume_preparation",
        )
    profile = AnalyticImplicitProfile(
        geometry,
        coordinate_contract,
        source_id=specification.boundary_scope.source_id,
        source_revision=specification.boundary_scope.source_revision,
    )
    profile.require_bound(geometry, coordinate_contract)
    box = np.asarray(domain, dtype=np.float64)
    if (
        box.shape != (2, 3)
        or not np.all(np.isfinite(box))
        or np.any(box[0] > profile.boundary_bounds[0])
        or np.any(box[1] < profile.boundary_bounds[1])
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "The finite volume query domain must enclose the complete source boundary.",
            stage="implicit_volume_preparation",
        )
    # The physical source is unchanged. A buffered computational hull avoids
    # truncating true Voronoi rays at a source/domain tangency; the original
    # declared enclosure remains separately bound in prepared_id.
    margin = profile.reach_lower / 16.0
    carrier = np.stack(
        (
            np.nextafter(box[0] - margin, -np.inf),
            np.nextafter(box[1] + margin, np.inf),
        )
    )
    enclosures = profile.field_bounds.boxes(carrier[:1], carrier[1:])
    query = ImplicitVolumeQuery(
        profile.field_bounds,
        carrier[:1],
        carrier[1:],
        enclosures.value_lower,
        enclosures.value_upper,
        np.asarray((0,), dtype=np.int64),
        carrier,
        maximum_level=1,
    )
    charge_native_geometry_queries(1)
    point = np.asarray(
        geometry.boundary_atlas.map(
            jnp.asarray((0,), dtype=jnp.int32),
            jnp.asarray(((0.25, 0.25),), dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    field, resolution = resolve_size_controls(
        specification.size_controls,
        point,
        np.asarray((0,), dtype=np.int64),
        SizeFieldDomain.EUCLIDEAN_VOLUME,
        combination=specification.size_combination,
    )
    size = float(np.asarray(field.values)[0])
    tolerance = profile.tube_radius
    for feature in specification.protected_features:
        if feature.hard:
            tolerance = min(tolerance, feature.maximum_deviation)
    seed_queries = len(specification.region_seeds)
    if seed_queries + 3 > specification.limits.maximum_geometry_queries:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Declared material seed queries exceed the geometry budget.",
            stage="implicit_volume_preparation",
        )
    if seed_queries:
        seed_points = np.stack(
            tuple(np.asarray(seed.point) for seed in specification.region_seeds)
        )
        if seed_points.shape[1:] != (3,):
            raise ValueError(
                "Analytic volume region seeds require three physical coordinates."
            )
        seed_classes = query.classify_boxes(seed_points, seed_points)
        if np.any(seed_classes.classes != int(ImplicitVolumeClass.INSIDE)):
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "A declared material seed is not source-enclosed strictly inside the analytic body.",
                stage="implicit_volume_preparation",
            )
    cover = min(
        0.5 * size,
        0.25 * profile.reach_lower,
        0.5 * np.sqrt(profile.reach_lower * tolerance),
    )
    identity = canonical_fingerprint(
        {
            "kind": "prepared-restricted-implicit-volume",
            "profile": profile.profile_id,
            "domain_query": query.query_id,
            "domain": array_tree_fingerprint(box),
            "specification": specification.specification_id,
            "schedule": schedule.schedule_id,
            "sizing": resolution.report_id,
            "fidelity": tolerance,
            "seed_cover": cover,
        }
    )
    return PreparedImplicitVolume(
        profile,
        query,
        specification,
        schedule,
        size,
        tolerance,
        float(cover),
        3 + seed_queries,
        resolution.report_id,
        identity,
    )


def _implicit_seed_sites(prepared: PreparedImplicitVolume, /) -> tuple[np.ndarray, int]:
    """A complete source atlas cover plus source-enclosed interior lattice sites."""
    query = prepared.query
    profile = prepared.profile
    limits = prepared.specification.limits
    spacing = min(prepared.target_size, 0.5 * profile.reach_lower)
    counts = np.maximum(
        1,
        np.ceil(
            (profile.boundary_bounds[1] - profile.boundary_bounds[0]) / spacing
        ).astype(np.int64),
    )
    lattice_count = int(np.prod(counts))
    queries = prepared.geometry_query_charge
    execution = current_native_execution_budget()
    if execution is None:
        raise RuntimeError(
            "Implicit seeding requires its actual active execution budget."
        )
    execution.charge()
    allowance = execution.remaining()
    if (
        lattice_count + 8 >= limits.maximum_vertices
        or lattice_count
        >= min(
            limits.maximum_geometry_queries - queries,
            allowance.remaining_geometry_queries,
        )
        or lattice_count * 128 > limits.maximum_scratch_bytes
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The analytic interior work lattice exceeds its vertex/query/scratch budget.",
            stage="implicit_volume_seeding",
        )
    axes = tuple(
        profile.boundary_bounds[0, axis]
        + (np.arange(counts[axis], dtype=np.float64) + 0.5)
        * (profile.boundary_bounds[1, axis] - profile.boundary_bounds[0, axis])
        / counts[axis]
        for axis in range(3)
    )
    lattice = np.stack(
        [axis.reshape((-1,)) for axis in np.meshgrid(*axes, indexing="ij")], axis=1
    )
    classification = query.classify_boxes(lattice, lattice)
    queries += lattice_count
    interior = lattice[classification.classes == int(ImplicitVolumeClass.INSIDE)]
    cover_budget = min(
        limits.maximum_vertices - interior.shape[0] - 8,
        min(
            limits.maximum_geometry_queries - queries,
            execution.remaining().remaining_geometry_queries,
        )
        // 3,
        limits.maximum_scratch_bytes // 512,
    )
    if cover_budget < 4:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The complete source-boundary cover has no remaining bounded capacity.",
            stage="implicit_volume_seeding",
        )
    before_cover = execution.remaining().remaining_geometry_queries
    try:
        cover = profile.boundary_cover(prepared.seed_cover_radius, cover_budget)
    except AnalyticBoundaryCoverCapacityError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            stage="implicit_volume_seeding",
            requested=(("cover_radius", error.requested_radius),),
            achieved=(("cover_radius", error.achieved_radius),),
        ) from error
    if not cover.complete or cover.semantics != "certified":
        raise ValueError(
            "Analytic volume seeding requires an established complete source cover."
        )
    if np.max(cover.point_error) >= 0.25 * prepared.fidelity_tolerance:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Boundary seed realization error exceeds the source-fidelity budget.",
            stage="implicit_volume_seeding",
        )
    corners = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ),
        dtype=np.int64,
    )
    hull = np.where(corners, query.domain[1], query.domain[0])
    queries += before_cover - execution.remaining().remaining_geometry_queries
    return np.concatenate((hull, cover.points, interior)), queries


def _implicit_region_metadata(
    specification: VolumeMeshingSpec, /
) -> tuple[str, str | None, RegionRole | None]:
    controls = specification.region_controls
    seeds = specification.region_seeds
    name = (
        controls[0].region_name
        if controls
        else (seeds[0].region_name if seeds else "interior")
    )
    material = (
        controls[0].material_id if controls else (seeds[0].material_id if seeds else None)
    )
    role = controls[0].role if controls else (seeds[0].role if seeds else None)
    if any(
        (seed.region_name, seed.material_id, seed.role) != (name, material, role)
        for seed in seeds
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            "The regular analytic source has one interior region; seed/control material identities conflict.",
            stage="implicit_volume_preparation",
        )
    for patch in specification.patch_controls:
        if (
            patch.scope.entity_dimension != 2
            or not np.array_equal(
                patch.scope.entity_ids, specification.boundary_scope.entity_ids
            )
            or patch.adjacent_region_names != (name,)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.CONTROL_CONFLICT,
                "An analytic boundary patch must name the complete source boundary and its single interior region.",
                stage="implicit_volume_preparation",
            )
    return name, material, role


def _restricted_facet_quality(
    prepared: PreparedImplicitVolume,
    work: ImplicitRestrictedQueryWorkset,
    /,
) -> tuple[np.ndarray, float, int]:
    """Flag true restricted facets needing source/size/normal refinement."""
    _, _, counts = _restricted_root_sites(work)
    selected = np.flatnonzero(counts == 1)
    bad = np.flatnonzero(counts > 1)
    if not selected.size:
        return np.union1d(bad, work.unresolved_facets), np.inf, 0
    triangles = work.domain.points[work.facets[selected]]
    remaining = (
        prepared.specification.limits.maximum_geometry_queries - work.geometry_queries
    )
    if remaining < 2 * selected.size:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Restricted primal-facet source and normal queries exceed the geometry budget.",
            stage="implicit_restricted_refinement",
        )
    bounds = prepared.query.bounds.boxes(
        np.min(triangles, axis=1), np.max(triangles, axis=1)
    )
    deviation = np.maximum(np.abs(bounds.value_lower), np.abs(bounds.value_upper))
    low, high = _boundary_normal_bounds(prepared.query, triangles)
    orientable = (low > 0.0) | (high < 0.0)
    diameter = np.maximum.reduce(
        tuple(
            _length_upper(triangles[:, first] - triangles[:, second])
            for first, second in ((0, 1), (1, 2), (2, 0))
        )
    )
    maximum_size = min(prepared.target_size, prepared.fidelity_tolerance)
    failed = (
        ~orientable
        | (deviation >= 0.25 * prepared.fidelity_tolerance)
        | (diameter > maximum_size)
    )
    bad = np.union1d(bad, selected[failed])
    return (
        np.union1d(bad, work.unresolved_facets),
        float(np.max(deviation)),
        2 * selected.size,
    )


def _restricted_closed_profile(
    prepared: PreparedImplicitVolume,
    work: ImplicitRestrictedQueryWorkset,
    /,
) -> bool:
    """Check closed oriented source topology before any native protection edit."""
    _, _, counts = _restricted_root_sites(work)
    selected = np.flatnonzero(counts == 1)
    if not selected.size or np.any(counts > 1) or work.unresolved_facets.size:
        return False
    triangles = work.domain.points[work.facets[selected]]
    low, high = _boundary_normal_bounds(prepared.query, triangles)
    if np.any((low <= 0.0) & (high >= 0.0)):
        return False
    faces = work.facets[selected].copy()
    flip = high < 0.0
    faces[flip] = faces[flip][:, (0, 2, 1)]
    directed = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    _, count = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    if np.any(count != 2) or np.unique(directed, axis=0).shape[0] != directed.shape[0]:
        return False
    used = np.unique(faces)
    reindex = np.full(work.domain.points.shape[0], -1, dtype=np.int64)
    reindex[used] = np.arange(used.size, dtype=np.int64)
    # Primal geometry comes from exact native positive tetrahedra. This phase
    # owns combinatorial closure, not TriangleMesh's approximate area filter.
    # Actual coordinate validity and shape are independently audited later.
    topology = TriangleTopology(reindex[faces], num_vertices=used.size)
    return (
        topology.num_face_components == prepared.profile.component_count
        and topology.euler_characteristic
        == prepared.profile.boundary_euler_characteristic
    )


def _protected_insertion_batch(points: np.ndarray, radii: np.ndarray, /) -> np.ndarray:
    """Schedule separated empty-ball candidates from one triangulation epoch.

    Adjacent old duals can request practically the same surface site. Native
    insertion is sequential: after the first edit, their old empty-ball
    premises are obsolete. Defer conflicting candidates and recompute their
    actual duals next round, rather than constructing slivers between duplicate
    requests or repairing/welding the accepted mesh afterward.
    """
    if points.shape[0] < 2:
        return points
    protection = 0.25 * radii
    hierarchy = prepare_bvh(
        np.nextafter(points - protection[:, None], -np.inf),
        np.nextafter(points + protection[:, None], np.inf),
        dtype=np.float64,
    )
    neighbors: list[list[int]] = [[] for _ in range(points.shape[0])]
    for first, second in bvh_overlap_pair_blocks(
        hierarchy, hierarchy, include_touching=True
    ):
        selected = first < second
        first, second = first[selected], second[selected]
        threshold = np.maximum(protection[first], protection[second])
        near = (
            np.sum((points[first] - points[second]) ** 2, axis=1) < threshold * threshold
        )
        first, second = first[near], second[near]
        for one, two in zip(first.tolist(), second.tolist(), strict=True):
            neighbors[one].append(two)
            neighbors[two].append(one)
    blocked = np.zeros(points.shape[0], dtype=np.bool_)
    accepted = np.zeros(points.shape[0], dtype=np.bool_)
    order = np.argsort(-radii, kind="stable")
    for index in order.tolist():
        if not blocked[index]:
            accepted[index] = True
            blocked[neighbors[index]] = True
    return points[accepted]


def _construct_implicit_restriction(
    prepared: PreparedImplicitVolume,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    operation_started: float | None,
) -> tuple[RestrictedImplicitTopology, ImplicitRestrictedQueryWorkset, int, int, float]:
    """Native Delaunay site refinement followed by protected primal-region flood."""
    from ._volume_generation import native_volume_checkpoint

    profile = prepared.profile
    profile.require_bound(profile.geometry, profile.coordinate_contract)
    limits = prepared.specification.limits
    started = operation_started
    execution = current_native_execution_budget()
    if execution is None:
        raise RuntimeError(
            "Implicit restriction requires its actual active execution budget."
        )
    with measure_phase(record_phase, "source_preparation"):
        sites, queries = _implicit_seed_sites(prepared)
    with measure_phase(record_phase, "regular_triangulation"):
        state = IncrementalDelaunay3D(
            sites,
            max_vertices=limits.maximum_vertices,
            max_tetrahedra=limits.maximum_cells,
            max_cavity=limits.maximum_cavity_cells,
        )
    extra_work = 0
    try:
        for round_ in range(prepared.schedule.refinement_rounds + 1):
            native_volume_checkpoint(
                limits,
                MeshingStageKind.VOLUME_FILL,
                operation_started=started,
                execution_budget=execution,
            )
            points, cells, _, _, _ = state.finalize()
            statistics = state.statistics()
            classification = _classify(
                prepared.query, points, cells, limits.maximum_geometry_queries - queries
            )
            queries += int(np.count_nonzero(classification[5]))
            domain = _publish_workset(
                prepared.query,
                prepared.specification,
                prepared.schedule,
                points,
                cells,
                classification,
                ImplicitVolumeWorkStop.REFINEMENT_LIMIT,
                queries,
                round_,
                statistics,
            )
            work = prepare_implicit_restricted_queries(
                domain,
                spatial_tolerance=prepared.fidelity_tolerance * 1.0e-3,
                maximum_depth=32,
                additional_work_units=extra_work,
            )
            queries = work.geometry_queries
            extra_work = work.work_units - int(statistics[7])
            bad, deviation, facet_queries = _restricted_facet_quality(prepared, work)
            queries += facet_queries
            if not bad.size:
                before_closure = execution.remaining().remaining_geometry_queries
                closed = _restricted_closed_profile(prepared, work)
                queries += (
                    before_closure - execution.remaining().remaining_geometry_queries
                )
                if closed:
                    topology = assemble_restricted_implicit_topology(
                        state, work, record_phase=record_phase
                    )
                    region_work = int(state.statistics()[7]) - int(statistics[7])
                    return (
                        topology,
                        work,
                        work.work_units + region_work,
                        queries,
                        deviation,
                    )
                # All dual proofs remain meaningful, but not a source-bound
                # manifold yet. Refine their actual source roots, not PLC faces.
                _, _, counts = _restricted_root_sites(work)
                bad = np.flatnonzero(counts > 0)
            if round_ == prepared.schedule.refinement_rounds:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Restricted primal/source conformity remains unresolved after the native refinement budget.",
                    stage="implicit_restricted_refinement",
                    entity_ids=tuple(int(value) for value in bad),
                    requested=(("source_fidelity", prepared.fidelity_tolerance),),
                    achieved=(("facet_interval_deviation", deviation),),
                    checkpoint_id=work.workset_id,
                )
            root_facets, root_sites, _ = _restricted_root_sites(work)
            selected_roots = np.isin(root_facets, bad)
            additions = root_sites[selected_roots]
            addition_radii = np.min(
                np.linalg.norm(
                    points[work.facets[root_facets[selected_roots]]] - additions[:, None],
                    axis=2,
                ),
                axis=1,
            )
            unresolved_cells = work.domain.unresolved_tetrahedra
            if work.unresolved_facets.size or not additions.size:
                # Interval uncertainty is never discarded: subdivide the full
                # covering carrier near the exact unresolved dual incidences.
                owners = work.incident_cells[work.unresolved_facets].reshape((-1,))
                pending = np.unique(owners[owners >= 0])
                if not pending.size:
                    pending = unresolved_cells
                centers = np.mean(points[cells[pending]], axis=1)
                center_radii = np.min(
                    np.linalg.norm(points[cells[pending]] - centers[:, None], axis=2),
                    axis=1,
                )
                additions = np.concatenate((additions, centers))
                addition_radii = np.concatenate((addition_radii, center_radii))
            additions = _protected_insertion_batch(additions, addition_radii)
            remaining_vertices = limits.maximum_vertices - points.shape[0]
            if additions.shape[0] > remaining_vertices or not additions.shape[0]:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Restricted-domain insertion cannot fit its complete next site batch.",
                    stage="implicit_restricted_refinement",
                )
            before = int(statistics[1])
            started_insertion = phase_started(record_phase)
            _, status = state.insert(
                additions,
                work_limit=execution.remaining().remaining_work_units,
            )
            after = state.statistics()
            record_elapsed(
                record_phase,
                "site_refinement",
                started_insertion,
                work_units=int(after[7]) - int(statistics[7]),
            )
            if np.any(status != int(MeshcoreStatus.OK)) or int(after[1]) <= before:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Native restricted-domain insertion was refused or made no geometric progress.",
                    stage="implicit_restricted_refinement",
                    checkpoint_id=work.workset_id,
                )
        raise RuntimeError("Restricted refinement did not return or report its budget.")
    finally:
        state.close()


@dataclass(frozen=True, slots=True)
class ImplicitVolumeConstruction:
    """Actual native restricted-domain tetrahedra, geometry and phase evidence.

    ``exudation`` is the schedule's explicit weighted stage run (``None`` when
    disabled) and ``exudation_weights`` its accepted weights per published vertex.
    """

    mesh: CellMesh
    domain: PiecewiseLinearDomain
    cell_regions: np.ndarray
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    associations: tuple[GeometryAssociation, ...]
    stages: tuple[MeshingStageReport, ...]
    refinement: TetMeshRun
    improvement: TetMeshRun
    minimum_dihedral: float
    maximum_radius_edge: float
    source_deviation_upper: float
    work_units: int
    geometry_query_charge: int
    restricted_topology_id: str
    restricted_query_id: str
    exudation: TetMeshRun | None = None
    exudation_weights: np.ndarray | None = None


def _improve_implicit_volume(
    topology: RestrictedImplicitTopology,
    prepared: PreparedImplicitVolume,
    work_units: int,
    /,
    *,
    validity_policy: CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None,
    operation_started: float | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TetMeshRun,
    TetMeshRun,
    TetMeshRun | None,
    np.ndarray | None,
    float,
    float,
    int,
]:
    """Run the owning native tetrahedral quality machinery on the restricted body."""
    from ._volume_generation import native_volume_checkpoint
    from .providers._native_publication import unique_edges

    execution = current_native_execution_budget()
    if execution is None:
        raise RuntimeError(
            "Implicit quality requires its actual active execution budget."
        )

    native_volume_checkpoint(
        prepared.specification.limits,
        MeshingStageKind.OPTIMIZATION,
        operation_started=operation_started,
    )

    limits = prepared.specification.limits
    boundary = topology.boundary_facets
    edges = unique_edges(boundary, "triangle")
    with measure_phase(record_phase, "native_preparation"):
        state = TetMesh3D(
            topology.points,
            topology.tetrahedra,
            np.zeros(topology.tetrahedra.shape[0], dtype=np.int32),
            boundary,
            np.arange(boundary.shape[0], dtype=np.int32),
            edges,
            np.arange(edges.shape[0], dtype=np.int32),
            boundary_policy="fixed",
            max_vertices=limits.maximum_vertices,
            max_tetrahedra=limits.maximum_cells,
        )
    try:
        start = phase_started(record_phase)
        native_volume_checkpoint(
            limits, MeshingStageKind.OPTIMIZATION, operation_started=operation_started
        )
        refinement = state.refine(
            radius_edge_bound=prepared.schedule.radius_edge_bound,
            sizes=np.full(
                topology.points.shape[0], prepared.target_size, dtype=np.float64
            ),
            max_insertions=limits.maximum_vertices - topology.points.shape[0],
            work_limit=execution.remaining().remaining_work_units,
        )
        refinement_work = int(
            refinement.counters[TET_MESH_REFINE_COUNTERS.index("work_units")]
        )
        record_elapsed(record_phase, "refinement", start, work_units=refinement_work)
        work_units += refinement_work
        if (
            refinement.status == MeshcoreStatus.CAPACITY_EXCEEDED
            or work_units >= limits.maximum_work_units
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Restricted tetrahedral quality refinement exhausted its native capacity.",
                stage="implicit_volume_quality",
            )
        start = phase_started(record_phase)
        native_volume_checkpoint(
            limits, MeshingStageKind.OPTIMIZATION, operation_started=operation_started
        )
        improvement = state.improve(
            min_dihedral_degrees=prepared.schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=validity_policy.relative_determinant_floor,
            max_passes=prepared.schedule.improvement_passes,
            work_limit=execution.remaining().remaining_work_units,
        )
        improvement_work = int(
            improvement.counters[TET_MESH_IMPROVE_COUNTERS.index("work_units")]
        )
        record_elapsed(record_phase, "improvement", start, work_units=improvement_work)
        work_units += improvement_work
        if improvement.status == MeshcoreStatus.CAPACITY_EXCEEDED:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Restricted tetrahedral improvement exhausted its native capacity.",
                stage="implicit_volume_quality",
            )
        exudation, weights, work_units = exude_native_volume(
            state,
            prepared.schedule,
            limits,
            work_units,
            validity_policy=validity_policy,
            record_phase=record_phase,
        )
        native_volume_checkpoint(
            limits, MeshingStageKind.OPTIMIZATION, operation_started=operation_started
        )
        arrays = state.arrays()
        quality = state.quality(sliver_degrees=prepared.schedule.minimum_dihedral_degrees)
        used = np.unique(arrays.tetrahedra)
        reindex = np.full(arrays.points.shape[0], -1, dtype=np.int64)
        reindex[used] = np.arange(used.size, dtype=np.int64)
        return (
            arrays.points[used],
            reindex[arrays.tetrahedra],
            reindex[arrays.faces],
            refinement,
            improvement,
            exudation,
            None if weights is None else weights[used],
            quality.minimum_dihedral,
            quality.maximum_radius_edge,
            work_units,
        )
    finally:
        state.close()


def _implicit_organization(
    mesh: CellMesh,
    specification: VolumeMeshingSpec,
    source_deviation: float,
    source_id: str,
    source_revision: str,
    /,
) -> tuple[tuple[MeshZone, ...], tuple[MeshPatch, ...], tuple[GeometryAssociation, ...]]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("Implicit publication requires tetrahedral connectivity.")
    name, material, role = _implicit_region_metadata(specification)
    cells = mesh.entity_set(3)
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        3,
        cells.entity_set_id,
        cells.entity_ids,
    )
    zone = MeshZone(
        name, MeshZoneRole.REGION, scope, material_id=material, region_role=role
    )
    faces = mesh.entity_set(2)
    identifiers = np.asarray(faces.entity_ids)[np.asarray(connectivity.boundary_faces)]
    boundary_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        2,
        faces.entity_set_id,
        identifiers,
    )
    patch_names = tuple(patch.name for patch in specification.patch_controls) or (
        "implicit-boundary",
    )
    patches = tuple(
        MeshPatch(patch_name, boundary_scope, adjacent_zone_ids=(zone.zone_id,))
        for patch_name in patch_names
    )
    association = GeometryAssociation(
        GeometryAssociationKind.IMPLICIT,
        source_id,
        source_revision,
        faces.entity_set_id,
        identifiers,
        tuple("implicit-zero-set" for _ in identifiers),
        np.full(identifiers.size, source_deviation, dtype=np.float64),
        resolved=np.ones(identifiers.size, dtype=np.bool_),
        exact=False,
        source_dimensions=np.full(identifiers.size, 2, dtype=np.int32),
        source_indices=np.zeros(identifiers.size, dtype=np.int32),
    )
    return (zone,), patches, (association,)


def generate_restricted_implicit_volume(
    prepared: PreparedImplicitVolume,
    /,
    *,
    validity_policy: CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    execution_budget: NativeExecutionBudget | None = None,
    operation_started: float | None = None,
) -> ImplicitVolumeConstruction:
    """Construct an actual restricted-Delaunay body, never an extracted PLC fill.

    ``validity_policy`` is the publishing audit's cell validity policy; native
    improvement repairs or reports every cell below its determinant floor.
    """
    from ._volume_generation import (
        _native_volume_operation_started,
        native_volume_execution_budget,
    )

    if not isinstance(prepared, PreparedImplicitVolume):
        raise TypeError("prepared must be PreparedImplicitVolume.")
    if not isinstance(validity_policy, CellValidityPolicy):
        raise TypeError("validity_policy must be CellValidityPolicy.")
    active = current_native_execution_budget()
    if execution_budget is not None and execution_budget is not active:
        raise RuntimeError("Implicit generation must borrow the actual active budget.")
    started = _native_volume_operation_started(operation_started)
    if active is None and started is None:
        started = monotonic()
    with native_volume_execution_budget(
        prepared.specification.limits,
        operation_started=started,
        source_geometry_queries=prepared.geometry_query_charge,
    ) as execution:
        return _generate_restricted_implicit_volume_bound(
            prepared,
            validity_policy=validity_policy,
            record_phase=record_phase,
            execution_budget=execution,
            operation_started=started,
        )


def _generate_restricted_implicit_volume_bound(
    prepared: PreparedImplicitVolume,
    /,
    *,
    validity_policy: CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None,
    execution_budget: NativeExecutionBudget,
    operation_started: float | None,
) -> ImplicitVolumeConstruction:
    from ._volume_generation import native_volume_checkpoint

    topology, work, spent, queries, deviation = _construct_implicit_restriction(
        prepared,
        record_phase=record_phase,
        operation_started=operation_started,
    )
    (
        points,
        cells,
        boundary,
        refinement,
        improvement,
        exudation,
        exudation_weights,
        minimum_angle,
        maximum_ratio,
        spent,
    ) = _improve_implicit_volume(
        topology,
        prepared,
        spent,
        validity_policy=validity_policy,
        record_phase=record_phase,
        operation_started=operation_started,
    )
    with measure_phase(record_phase, "publication"):
        native_volume_checkpoint(
            prepared.specification.limits,
            MeshingStageKind.CANONICALIZATION,
            operation_started=operation_started,
            execution_budget=execution_budget,
        )
        mesh = canonicalize_cell_mesh(
            CellMesh.from_tetrahedra(
                points, cells, numeric_version=prepared.profile.source_revision
            )
        )
        name, _, _ = _implicit_region_metadata(prepared.specification)
        domain = PiecewiseLinearDomain(
            points,
            boundary,
            np.tile(np.asarray((0, -1), dtype=np.int64), (boundary.shape[0], 1)),
            (name,),
            source_id=f"{prepared.profile.source_id}:restricted-approximation",
        )
    with measure_phase(record_phase, "organization"):
        zones, patches, associations = _implicit_organization(
            mesh,
            prepared.specification,
            deviation,
            prepared.profile.source_id,
            prepared.profile.source_revision,
        )
    stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(prepared.prepared_id,),
            output_ids=(prepared.profile.profile_id, prepared.size_resolution_id),
        ),
        MeshingStageReport(
            MeshingStageKind.VOLUME_FILL,
            MeshingStageStatus.PASSED,
            input_ids=(work.workset_id,),
            output_ids=(topology.topology_id, domain.domain_id),
            created_count=topology.tetrahedra.shape[0],
        ),
        MeshingStageReport(
            MeshingStageKind.OPTIMIZATION,
            MeshingStageStatus.PASSED
            if refinement.status == improvement.status == MeshcoreStatus.OK
            and (exudation is None or exudation.status == MeshcoreStatus.OK)
            else MeshingStageStatus.WARNING,
            input_ids=(topology.topology_id,),
            output_ids=(mesh.mesh_id,),
            created_count=mesh.blocks[0].cell_count,
        ),
    )
    native_volume_checkpoint(
        prepared.specification.limits,
        MeshingStageKind.CANONICALIZATION,
        operation_started=operation_started,
    )
    return ImplicitVolumeConstruction(
        mesh,
        domain,
        np.zeros(mesh.blocks[0].cell_count, dtype=np.int64),
        zones,
        patches,
        associations,
        stages,
        refinement,
        improvement,
        minimum_angle,
        maximum_ratio,
        deviation,
        spent,
        queries,
        topology.topology_id,
        work.workset_id,
        exudation=exudation,
        exudation_weights=exudation_weights,
    )


def _implicit_volume_compliance(
    construction: ImplicitVolumeConstruction,
    prepared: PreparedImplicitVolume,
    /,
) -> MeshingComplianceReport:
    from .providers._native_publication import edge_size_evidence, uniform_size_compliance

    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("Implicit size compliance requires tetrahedral connectivity.")
    lengths, growth = edge_size_evidence(
        np.asarray(construction.mesh.coordinates),
        np.asarray(connectivity.edges)[np.asarray(connectivity.boundary_edges)],
    )
    requested: list[tuple[str, float]] = [
        ("source_fidelity", prepared.fidelity_tolerance),
        ("refinement:radius_edge_bound", prepared.schedule.radius_edge_bound),
        (
            "improvement:minimum_dihedral_degrees",
            prepared.schedule.minimum_dihedral_degrees,
        ),
    ]
    achieved: list[tuple[str, float]] = [
        (
            "source_fidelity:complete_facet_interval_upper",
            construction.source_deviation_upper,
        ),
        ("refinement:maximum_radius_edge", construction.maximum_radius_edge),
        ("improvement:minimum_dihedral_degrees", construction.minimum_dihedral),
        ("native:known_phase_work_units", float(construction.work_units)),
        ("source:geometry_queries", float(construction.geometry_query_charge)),
    ]
    issues: list[str] = []
    for control in prepared.specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise TypeError(
                "Prepared restricted volume contains an unsupported size field."
            )
        request, actual, failed = uniform_size_compliance(
            control, prepared.specification.size_compliance, lengths, growth
        )
        requested.extend(request)
        achieved.extend(actual)
        issues.extend(failed)
    for feature in prepared.specification.protected_features:
        key = f"protected:{feature.feature_id}:maximum_deviation"
        requested.append((key, feature.maximum_deviation))
        achieved.append((key, construction.source_deviation_upper))
        if (
            feature.hard
            and construction.source_deviation_upper > feature.maximum_deviation
        ):
            issues.append(f"protected_deviation:{feature.feature_id}")
    return MeshingComplianceReport(
        prepared.specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def execute_restricted_implicit_volume(
    geometry: CompiledGeometry,
    specification: VolumeMeshingSpec,
    prepared: PreparedImplicitVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    execution_budget: NativeExecutionBudget | None = None,
    operation_started: float | None = None,
) -> CellMeshingResult:
    """Certify and publish qualified analytic restricted-Delaunay generation."""
    from ._volume_generation import (
        _native_volume_operation_started,
        native_volume_execution_budget,
    )
    from .providers._native_publication import bind_native_execution_result

    prepared.profile.require_bound(geometry, coordinate_contract)
    if specification.specification_id != prepared.specification.specification_id:
        raise ValueError("The prepared restricted volume binds another physical request.")
    active = current_native_execution_budget()
    if execution_budget is not None and execution_budget is not active:
        raise RuntimeError("Implicit publication must borrow the actual active budget.")
    started = _native_volume_operation_started(operation_started)
    if active is None and started is None:
        started = monotonic()
    with native_volume_execution_budget(
        specification.limits,
        operation_started=started,
        source_geometry_queries=prepared.geometry_query_charge,
    ) as execution:
        result = _execute_restricted_implicit_volume_bound(
            geometry,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            record_phase=record_phase,
            execution_budget=execution,
            operation_started=started,
        )
    if active is not None:
        return result
    evidence = execution.evidence
    if evidence is None:
        raise RuntimeError(
            "Implicit publication lost its completed original execution evidence."
        )
    if started is None:
        raise RuntimeError("Owning implicit publication lost its original host clock.")
    return bind_native_execution_result(
        result,
        evidence,
        specification.limits,
        started,
        source_geometry_queries=prepared.geometry_query_charge,
    )


def _execute_restricted_implicit_volume_bound(
    geometry: CompiledGeometry,
    specification: VolumeMeshingSpec,
    prepared: PreparedImplicitVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    execution_budget: NativeExecutionBudget,
    operation_started: float | None,
) -> CellMeshingResult:
    from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
    from ._certification import MeshCertificationSchedule
    from ._volume_generation import native_volume_checkpoint
    from .providers._native_publication import (
        NativeCertificationRequest,
        publish_native_result,
        simplex_entity_limits,
    )

    # The native determinant-floor repair and the publication audit share one policy.
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    construction = generate_restricted_implicit_volume(
        prepared,
        validity_policy=audit_policy.validity_policy,
        record_phase=record_phase,
        execution_budget=execution_budget,
        operation_started=operation_started,
    )
    simplex_entity_limits(
        np.asarray(construction.mesh.coordinates),
        np.asarray(construction.mesh.blocks[0].vertices),
        specification.limits,
        MeshingStageKind.VOLUME_FILL,
        cell_kind=construction.mesh.blocks[0].cell_kind,
    )
    with measure_phase(record_phase, "compliance"):
        native_volume_checkpoint(
            specification.limits,
            MeshingStageKind.SPECIFICATION_COMPLIANCE,
            operation_started=operation_started,
        )
        compliance = _implicit_volume_compliance(construction, prepared)
    return publish_native_result(
        construction.mesh,
        coordinate_contract,
        compliance,
        construction.stages,
        provider,
        {
            "kind": "native-restricted-implicit-volume",
            "route": "implicit_restricted_delaunay",
            "plan": plan_id,
            "source_profile": prepared.profile.profile_id,
            "prepared": prepared.prepared_id,
            "source_state": prepared.profile.state_id,
            "source_schema": prepared.profile.schema_id,
            "source_kernel": prepared.profile.kernel_id,
            "restricted_topology": construction.restricted_topology_id,
            "restricted_queries": construction.restricted_query_id,
            "approximate_domain": construction.domain.domain_id,
            "source_fidelity_tolerance": prepared.fidelity_tolerance,
            "refinement_status": construction.refinement.status.name,
            "improvement_status": construction.improvement.status.name,
            "refinement_counters": tuple(
                zip(
                    TET_MESH_REFINE_COUNTERS,
                    construction.refinement.counters.tolist(),
                    strict=True,
                )
            ),
            "improvement_counters": tuple(
                zip(
                    TET_MESH_IMPROVE_COUNTERS,
                    construction.improvement.counters.tolist(),
                    strict=True,
                )
            ),
            "exudation_status": (
                None
                if construction.exudation is None
                else construction.exudation.status.name
            ),
            "exudation_counters": ()
            if construction.exudation is None
            else tuple(
                zip(
                    TET_MESH_EXUDE_COUNTERS,
                    construction.exudation.counters.tolist(),
                    strict=True,
                )
            ),
        },
        NativeCertificationRequest(
            MeshCertificationSchedule("volume_implicit"),
            prepared.profile.source_id,
            prepared.profile.source_revision,
            specification.limits,
            domain=construction.domain,
            cell_regions=construction.cell_regions,
            fidelity_source=ImplicitProjectionBoundarySource(prepared.profile),
            fidelity_tolerance=prepared.fidelity_tolerance,
        ),
        audit_policy=audit_policy,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "work_units:native_and_metered_host",
            "wall_seconds:native_and_host_boundaries",
            "geometry_queries:native_and_source_batches",
            "cavity_cells:native",
            "scratch_bytes:managed_native_and_bounded_host",
        ),
        unenforced_limits=(
            "scratch_bytes:unmanaged_host_device_compiler",
            "work_units:unmetered_host_device_compiler",
            "wall_seconds:nonpreemptible_host_device_compiler",
        ),
        patches=construction.patches,
        zones=construction.zones,
        associations=construction.associations,
        record_phase=record_phase,
        operation_started=operation_started,
    )


def _adaptive_source_state_id(geometry: CompiledGeometry, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "adaptive-implicit-source-state",
            "state": implicit_state_id(geometry),
            "kernel_structure": str(jax.tree_util.tree_structure(geometry.kernel)),
            "kernel_data": array_tree_fingerprint(geometry.kernel),
        }
    )


@final
class PreparedAdaptiveImplicitVolume(StrictModule, NonTrainableState):
    """Certified adaptive zero set and its exact extracted constrained carrier."""

    geometry: CompiledGeometry
    surface: AdaptiveImplicitSurface
    schedule: NativeVolumeSchedule
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    source_state_id: str = eqx.field(static=True)
    surface_numeric_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    fidelity_tolerance: float = eqx.field(static=True)
    geometry_queries: int = eqx.field(static=True)
    source_work_units: int = eqx.field(static=True)
    preparation_seconds: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CompiledGeometry,
        surface: AdaptiveImplicitSurface,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        source_revision: str,
        fidelity_tolerance: float,
        geometry_queries: int,
        source_work_units: int,
        preparation_seconds: float,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        AdaptiveImplicitBoundarySource(geometry, surface, source_revision)
        self.geometry, self.surface, self.schedule = geometry, surface, schedule
        self.coordinate_contract = coordinate_contract
        self.specification = specification
        self.source_state_id = _adaptive_source_state_id(geometry)
        self.surface_numeric_id = canonical_fingerprint(array_tree_fingerprint(surface))
        self.specification_id = specification.specification_id
        self.source_id, self.source_revision = surface.source_id, source_revision
        self.fidelity_tolerance = fidelity_tolerance
        self.geometry_queries, self.source_work_units = (
            geometry_queries,
            source_work_units,
        )
        self.preparation_seconds = preparation_seconds
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-adaptive-implicit-volume",
                "source": surface.source_id,
                "revision": source_revision,
                "state": self.source_state_id,
                "surface": surface.evidence.evidence_id,
                "surface_numerical_binding": self.surface_numeric_id,
                "witnesses": array_tree_fingerprint(surface.boundary_witness_boxes),
                "specification": self.specification_id,
                "schedule": schedule.schedule_id,
                "fidelity": fidelity_tolerance,
                "coordinates": coordinate_contract.spatial_id,
            }
        )


def adaptive_implicit_volume_support_issues(
    specification: VolumeMeshingSpec, /
) -> tuple[str, ...]:
    """Controls representable by one certified scalar negative-domain source."""
    issues = list(implicit_volume_support_issues(specification))
    issues = [
        issue
        for issue in issues
        if issue != "additional void components not present in the source profile"
    ]
    for patch in specification.patch_controls:
        if patch.scope.entity_dimension != 2 or not np.array_equal(
            patch.scope.entity_ids, [0]
        ):
            issues.append("patch scopes bound to the complete scalar zero set")
    return tuple(sorted(set(issues)))


def _classify_adaptive_volume_seeds(
    surface: AdaptiveImplicitSurface,
    specification: VolumeMeshingSpec,
    queries: int,
    /,
) -> int:
    """Admit authoritative point queries after genuine certified source discovery."""
    if not surface.evidence.certified or not surface.volume.certified:
        raise ValueError("Adaptive seed classification requires a certified source.")
    seed_queries = len(specification.region_seeds) + len(specification.hole_seeds)
    if queries + seed_queries >= specification.limits.maximum_geometry_queries:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Adaptive seed classification consumes the geometry query budget.",
            stage="implicit_volume_preparation",
            achieved=(
                ("geometry_queries", queries),
                ("requested_seed_queries", seed_queries),
            ),
        )
    for seed, expected in (
        *((seed, ImplicitVolumeClass.INSIDE) for seed in specification.region_seeds),
        *((seed, ImplicitVolumeClass.OUTSIDE) for seed in specification.hole_seeds),
    ):
        point = np.asarray(seed.point, dtype=np.float64).reshape((1, 3))
        lower, upper = surface.volume.bounds.point_values(point)
        admitted = (
            upper[0] < 0.0 if expected is ImplicitVolumeClass.INSIDE else lower[0] > 0.0
        )
        if not admitted:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "A declared region or hole seed contradicts the certified scalar source.",
                stage="implicit_volume_preparation",
            )
    return queries + seed_queries


def prepare_adaptive_implicit_volume(
    geometry: CompiledGeometry,
    domain: ArrayLike,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    policy: AdaptiveImplicitSurfacePolicy,
    source_revision: str,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> PreparedAdaptiveImplicitVolume:
    """Discover every possible component before recovering the bounded PLC."""
    from .providers._native_publication import check_deadline, simplex_entity_limits

    issues = adaptive_implicit_volume_support_issues(specification)
    if issues:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(issues),
            stage="implicit_volume_preparation",
        )
    _implicit_region_metadata(specification)
    if policy.enclosure == "sampled" or policy.allow_approximate_zero_set:
        raise ValueError(
            "Implicit volume publication requires sound exact-zero-set enclosures."
        )
    limits = specification.limits
    started = monotonic()
    # Bound the complete host tree and native predicates before discovery.
    # Each leaf can own 82 entity queries and their directional enclosures,
    # integer incidence, sorting and extraction buffers simultaneously.
    box_capacity = min(
        policy.maximum_boxes,
        limits.maximum_scratch_bytes // 32768,
        limits.maximum_data_bytes // 2048,
    )
    if box_capacity < 1 << (3 * policy.initial_level):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The initial adaptive cover exceeds its scratch/data/work capacity.",
            stage="implicit_volume_preparation",
        )
    bounded = replace(
        policy,
        maximum_boxes=box_capacity,
        maximum_evaluations=min(
            policy.maximum_evaluations, limits.maximum_geometry_queries
        ),
        maximum_faces=min(
            policy.maximum_faces,
            limits.maximum_faces,
            limits.maximum_connectivity_entries // 3,
        ),
        maximum_intersection_pairs=min(
            policy.maximum_intersection_pairs, limits.maximum_scratch_bytes // 128
        ),
    )
    surface = discover_adaptive_implicit_surface(
        geometry,
        domain=domain,
        policy=bounded,
        source_id=specification.boundary_scope.source_id,
        maximum_root_solves=limits.maximum_work_units,
    )
    check_deadline(started, limits, MeshingStageKind.SOURCE_INSPECTION)
    evidence = surface.evidence
    if (
        not evidence.certified
        or evidence.unresolved_count
        or surface.mesh is None
        or surface.topology is None
    ):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "The adaptive source retains unresolved, singular, hidden or unclosed zero-set components.",
            stage="implicit_volume_preparation",
            checkpoint_id=evidence.evidence_id,
            entity_ids=tuple(range(evidence.unresolved_count)),
            locations=tuple(
                tuple(point)
                for point in np.mean(evidence.unresolved_boxes, axis=1).tolist()
            ),
            achieved=(
                ("unresolved_boxes", evidence.unresolved_count),
                ("source_status", evidence.status),
                *(
                    (
                        f"unresolved_issue:{issue}",
                        int(np.count_nonzero(evidence.unresolved_issues == issue)),
                    )
                    for issue in np.unique(evidence.unresolved_issues).tolist()
                ),
            ),
        )
    points = np.asarray(surface.mesh.vertices, dtype=np.float64)
    faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    simplex_entity_limits(
        points, faces, limits, MeshingStageKind.SOURCE_INSPECTION, cell_kind="triangle"
    )
    # Recompute existence, not merely a nominal rigorous flag: the 3x3x3
    # lattice contains every endpoint of a balanced leaf's minimal edge.
    offsets = np.asarray(
        tuple(
            (x, y, z)
            for x in (0.0, 0.5, 1.0)
            for y in (0.0, 0.5, 1.0)
            for z in (0.0, 0.5, 1.0)
        ),
        dtype=np.float64,
    )
    witness = surface.boundary_witness_boxes
    witness_points = witness[:, None, 0] + offsets[None] * (
        witness[:, None, 1] - witness[:, None, 0]
    )
    queries = (
        evidence.box_evaluations
        + evidence.point_evaluations
        + evidence.extraction_evaluations
    )
    queries += witness_points.shape[0] * witness_points.shape[1]
    if queries >= limits.maximum_geometry_queries:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Adaptive cycle-root existence checks exceed the geometry query budget.",
            stage="implicit_volume_preparation",
        )
    lower, upper = surface.volume.bounds.point_values(witness_points.reshape((-1, 3)))
    negative = np.any(upper.reshape((-1, 27)) < 0.0, axis=1)
    positive = np.any(lower.reshape((-1, 27)) > 0.0, axis=1)
    if not np.all(negative & positive):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "An adaptive cycle has no independently source-enclosed root bracket.",
            stage="implicit_volume_preparation",
            entity_ids=tuple(np.flatnonzero(~(negative & positive)).tolist()),
        )
    source_work = evidence.root_solves
    if source_work >= limits.maximum_work_units:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Complete adaptive discovery consumes the native construction work budget.",
            stage="implicit_volume_preparation",
        )
    tolerance = min(
        (
            feature.maximum_deviation
            for feature in specification.protected_features
            if feature.hard
        ),
        default=evidence.maximum_surface_box_diagonal * 4.0,
    )
    if tolerance <= 0.0:
        raise ValueError(
            "Affine implicit boundary publication needs a positive source-fidelity tolerance."
        )
    queries = _classify_adaptive_volume_seeds(surface, specification, queries)
    check_deadline(started, limits, MeshingStageKind.SOURCE_INSPECTION)
    return PreparedAdaptiveImplicitVolume(
        geometry,
        surface,
        specification,
        schedule,
        source_revision,
        tolerance,
        queries,
        source_work,
        monotonic() - started,
        coordinate_contract,
    )


def execute_adaptive_implicit_volume(
    geometry: CompiledGeometry,
    specification: VolumeMeshingSpec,
    prepared: PreparedAdaptiveImplicitVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    execution_budget: NativeExecutionBudget | None = None,
    operation_started: float | None = None,
) -> CellMeshingResult:
    """Keep construction, independent queries and publication inside one allowance."""
    from ._volume_generation import (
        _native_volume_operation_started,
        native_volume_execution_budget,
    )
    from .providers._native_publication import bind_native_execution_result

    active = current_native_execution_budget()
    if execution_budget is not None and execution_budget is not active:
        raise RuntimeError(
            "Adaptive implicit publication must borrow the actual active budget."
        )
    execution_budget = active
    started = _native_volume_operation_started(operation_started)
    if active is None and started is None:
        started = monotonic() - prepared.preparation_seconds

    if execution_budget is not None:
        return _execute_adaptive_implicit_volume_bound(
            geometry,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            execution_budget=execution_budget,
            record_phase=record_phase,
            operation_started=started,
        )
    with native_volume_execution_budget(
        specification.limits,
        source_work_units=prepared.source_work_units,
        source_geometry_queries=prepared.geometry_queries,
        operation_started=started,
    ) as budget:
        result = _execute_adaptive_implicit_volume_bound(
            geometry,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            execution_budget=budget,
            record_phase=record_phase,
            operation_started=started,
        )
    evidence = budget.evidence
    if evidence is None:
        raise RuntimeError("Adaptive publication lost its original execution evidence.")
    if started is None:
        raise RuntimeError("Owning adaptive publication lost its original host clock.")
    return bind_native_execution_result(
        result,
        evidence,
        specification.limits,
        started,
        source_work_units=prepared.source_work_units,
        source_geometry_queries=prepared.geometry_queries,
        preparation_seconds=prepared.preparation_seconds,
    )


def _execute_adaptive_implicit_volume_bound(
    geometry: CompiledGeometry,
    specification: VolumeMeshingSpec,
    prepared: PreparedAdaptiveImplicitVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    execution_budget: NativeExecutionBudget,
    operation_started: float | None,
) -> CellMeshingResult:
    """Fill the actual extracted source and independently certify continuous fidelity."""
    from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
    from ._certification import MeshCertificationSchedule
    from ._volume_generation import (
        declared_plc_domain,
        generate_plc_volume,
        native_volume_checkpoint,
        PiecewiseLinearComplex,
    )
    from .providers._native_publication import (
        NativeCertificationRequest,
        publish_native_result,
    )
    from .providers._native_volume import _volume_compliance

    if (
        _adaptive_source_state_id(geometry) != prepared.source_state_id
        or specification.specification_id != prepared.specification_id
        or canonical_fingerprint(array_tree_fingerprint(prepared.surface))
        != prepared.surface_numeric_id
        or coordinate_contract.spatial_id != prepared.coordinate_contract.spatial_id
    ):
        raise ValueError("Prepared adaptive implicit source or specification is stale.")
    surface = prepared.surface
    if surface.mesh is None:
        raise RuntimeError("A prepared certified implicit source lost its boundary.")
    started = operation_started
    source = AdaptiveImplicitBoundarySource(
        geometry,
        surface,
        prepared.source_revision,
        maximum_distance_pairs=specification.limits.maximum_work_units,
        maximum_scratch_bytes=specification.limits.maximum_scratch_bytes,
    )
    faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    name, _, _ = _implicit_region_metadata(specification)
    complex_ = PiecewiseLinearComplex(
        surface.mesh.vertices,
        tuple(face for face in faces),
        np.zeros(faces.shape[0], dtype=np.int64),
        np.asarray(((-1, 0),), dtype=np.int64),
        (name,),
        boundary="conforming",
    )
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    construction = generate_plc_volume(
        complex_,
        specification,
        prepared.schedule,
        validity_policy=audit_policy.validity_policy,
        source_id=prepared.source_id,
        source_revision=prepared.source_revision,
        input_id=prepared.prepared_id,
        record_phase=record_phase,
        source_work_units=prepared.source_work_units,
        operation_started=started,
        execution_budget=execution_budget,
        source_geometry_queries=prepared.geometry_queries,
    )
    points = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    cells = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    compliance = _volume_compliance(
        specification, prepared.schedule, construction, points, cells
    )
    protected_quantities = {
        f"protected:{feature.feature_id}:maximum_deviation"
        for feature in specification.protected_features
    }
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=compliance.issues,
        requested=compliance.requested,
        achieved=tuple(
            (name, prepared.fidelity_tolerance if name in protected_quantities else value)
            for name, value in compliance.achieved
        ),
    )
    zones, patches, associations = _implicit_organization(
        construction.mesh,
        specification,
        prepared.fidelity_tolerance,
        prepared.source_id,
        prepared.source_revision,
    )
    native_volume_checkpoint(
        specification.limits, MeshingStageKind.CERTIFICATION, operation_started=started
    )
    result = publish_native_result(
        construction.mesh,
        coordinate_contract,
        compliance,
        construction.stages,
        provider,
        {
            "kind": "native-adaptive-implicit-volume",
            "route": "implicit_adaptive_tetrahedral",
            "plan": plan_id,
            "prepared": prepared.prepared_id,
            "source_state": prepared.source_state_id,
            "source_revision": prepared.source_revision,
            "cover": surface.cover.cover_id if surface.cover is not None else "",
            "topology": surface.topology.result_id
            if surface.topology is not None
            else "",
        },
        NativeCertificationRequest(
            MeshCertificationSchedule("volume_implicit"),
            prepared.source_id,
            prepared.source_revision,
            specification.limits,
            domain=declared_plc_domain(complex_, prepared.source_id),
            cell_regions=construction.cell_regions,
            fidelity_source=source,
            fidelity_tolerance=prepared.fidelity_tolerance,
        ),
        audit_policy=audit_policy,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "scratch_bytes:managed_native_and_bounded_host",
            "work_units:native_and_metered_host",
            "wall_seconds:native_and_host_boundaries",
            "geometry_queries:native_and_source_batches",
            "cavity_cells:native",
        ),
        unenforced_limits=(
            "scratch_bytes:unmanaged_host_device_compiler",
            "work_units:unmetered_host_device_compiler",
            "wall_seconds:nonpreemptible_host_device_compiler",
        ),
        zones=zones,
        patches=patches,
        associations=associations,
        record_phase=record_phase,
        operation_started=started,
    )
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.SPECIFICATION_COMPLIANCE,
        operation_started=started,
    )
    return result


__all__ = [
    "ImplicitVolumeDomainWorkset",
    "ImplicitVolumeWorkStop",
    "prepare_implicit_volume_domain",
    "ImplicitDualIntervalClass",
    "ImplicitDualRootWorkset",
    "isolate_implicit_dual_roots",
    "ImplicitRestrictedQueryWorkset",
    "prepare_implicit_restricted_queries",
    "RestrictedImplicitTopology",
    "assemble_restricted_implicit_topology",
    "PreparedImplicitVolume",
    "implicit_volume_support_issues",
    "prepare_restricted_implicit_volume",
    "ImplicitVolumeConstruction",
    "generate_restricted_implicit_volume",
    "execute_restricted_implicit_volume",
    "PreparedAdaptiveImplicitVolume",
    "adaptive_implicit_volume_support_issues",
    "prepare_adaptive_implicit_volume",
    "execute_adaptive_implicit_volume",
]
