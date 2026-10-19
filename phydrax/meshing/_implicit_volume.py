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
full ring-torus profiles additionally admit primal restriction, protected
native region flood and tetrahedral quality refinement. Publication independently
checks approximate-domain coverage, global embedding and degree-one continuous
source projection within the established reach tube. The distinct adaptive
extraction route accepts general sound scalar enclosures only after
regular topology discovery, independently bracketed cycle roots, native PLC
coverage and continuous two-sided source bounds. It does not infer signed
distance, reach, hidden-component absence or material interfaces from samples.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum
from time import monotonic
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._certified_implicit as _dependency_phydrax_geometry__certified_implicit
import phydrax.geometry._contracts as _dependency_phydrax_geometry__contracts
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry.implicit._adaptive_discovery as _dependency_phydrax_geometry_implicit__adaptive_discovery
import phydrax.geometry.implicit._analytic_profile as _dependency_phydrax_geometry_implicit__analytic_profile
import phydrax.geometry.implicit._enclosure as _owner_phydrax_geometry_implicit__enclosure
import phydrax.geometry.implicit._projection_cover as implicit_projection_owner
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._audit as _owner_phydrax_meshing__audit
import phydrax.meshing._canonical as _dependency_phydrax_meshing__canonical
import phydrax.meshing._certification_schedule as _leaf_phydrax_meshing__certification_schedule
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation
import phydrax.meshing._volume_generation as _owner_phydrax_meshing__volume_generation
import phydrax.meshing.providers._native_publication as _owner_phydrax_meshing_providers__native_publication
import phydrax.meshing.providers._native_route_checks as native_route_checks_owner
import phydrax.meshing.providers._native_volume as _owner_phydrax_meshing_providers__native_volume
from phydrax.meshing.providers._native_route_policy import NativePublicationPolicy

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    exact_orient3d,
    IncrementalDelaunay3D,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    restricted_centers_3d,
    restricted_dual_3d,
    restricted_rays_3d,
    TetMesh3D,
    TetMeshExudeCounters,
    TetMeshImproveCounters,
    TetMeshRefineCounters,
    TetMeshRun,
    Triangulation3DStatistics,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, positive_integer
from ..discretization import CellMesh, TetrahedralConnectivity
from ..geometry.implicit._policy import AdaptiveImplicitSurfacePolicy
from ..geometry.simplicial._topology import TriangleTopology
from ..typing import checked
from ._measurements import (
    measure_phase,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)


@contextmanager
@checked
def implicit_source_budget(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
    queries_upper: int,
) -> Iterator[
    _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
]:
    """Reserve a host source phase and settle its ticket before native execution."""
    remaining = scope.remaining()
    ticket = scope.batch(
        min(limits.maximum_work_units, remaining.remaining_work_units),
        queries_upper=queries_upper,
        bytes_upper=min(limits.maximum_scratch_bytes, remaining.remaining_managed_bytes),
    )
    with ticket:
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.activate():
            yield budget


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
    """Retain a box carrier, source enclosures, and pending cells.

    ``classes`` are whole-cell classifications, not material labels. Unqueried
    cells have bounds [-inf, inf] and ``queried=False``. ``inside_tetrahedra``
    names a proven subset, never a mesh of the source domain. Physical controls
    are retained intact in ``specification``; none are claimed satisfied by
    this preparation. The coverage statement is only relative to query.domain,
    not a claim that the source has no components outside that declared box.

    Preparation enforces vertex, cell, cavity, insertion work and source-query
    bounds through an explicit native scope, including native initialization,
    managed storage and within-native deadlines.
    Unmanaged host/device/compiler work, storage and nonpreemptible phases
    remain explicit; this carrier is never eligible for publication by itself.
    """

    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule
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
        return np.flatnonzero(
            self.classes
            == int(
                _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.INSIDE
            )
        )

    @property
    def unresolved_tetrahedra(self) -> np.ndarray:
        """All potentially intersected cells, including unqueried cells."""
        return np.flatnonzero(
            self.classes
            == int(
                _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.UNKNOWN
            )
        )

    @checked
    def require_current(
        self,
        query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
        specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
        /,
    ) -> None:
        """Refuse reuse after source state, domain or physical controls change."""
        if _input_id(query, specification, self.schedule) != self.input_id:
            raise ValueError(
                f"ImplicitVolumeDomainWorkset.input_id must match the source and request; got stale input_id={self.input_id!r}"
            )


def _input_id(
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
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
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    points: np.ndarray,
    cells: np.ndarray,
    budget: int,
    /,
    *,
    scope: NativeExecutionScope,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = min(cells.shape[0], budget)
    vertices = points[cells]
    lower = np.min(vertices, axis=1)
    upper = np.max(vertices, axis=1)
    classes = np.full(
        cells.shape[0],
        int(
            _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.UNKNOWN
        ),
        dtype=np.int64,
    )
    value_lower = np.full(cells.shape[0], -np.inf, dtype=np.float64)
    value_upper = np.full(cells.shape[0], np.inf, dtype=np.float64)
    queried = np.arange(cells.shape[0], dtype=np.int64) < count
    if count:
        with implicit_source_budget(
            limits, scope=scope, queries_upper=2 * count
        ) as source_budget:
            result = query.classify_boxes(
                lower[:count], upper[:count], budget=source_budget
            )
        if not result.certified:
            raise ValueError(
                "prepare_implicit_volume_domain.query must provide certified value enclosures; got certified=False"
            )
        classes[:count] = result.classes
        value_lower[:count] = result.value_lower
        value_upper[:count] = result.value_upper
    return lower, upper, classes, value_lower, value_upper, queried


def _publish_workset(
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
    points: np.ndarray,
    cells: np.ndarray,
    classification: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    stop: ImplicitVolumeWorkStop,
    queries: int,
    rounds: int,
    statistics: Triangulation3DStatistics,
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
    return ImplicitVolumeDomainWorkset(
        query,
        specification,
        schedule,
        *arrays,
        stop,
        queries,
        rounds,
        statistics.items(),
        unenforced,
        input_id,
        workset_id,
    )


@checked
def prepare_implicit_volume_domain(
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
    /,
    *,
    scope: NativeExecutionScope,
) -> ImplicitVolumeDomainWorkset:
    """Refine a full-box native Delaunay workset using whole-cell intervals.

    Barycenters are insertion *sites*, never classification samples. Unknown
    tetrahedra are retained even when native insertion or a budget refuses
    further progress. No caller may turn this preparation into a successful
    implicit volume result without separately establishing restricted-boundary,
    source-domain coverage, feature and material-control obligations.
    """
    if not query.certified:
        raise ValueError(
            "prepare_implicit_volume_domain.query must provide certified value enclosures; got certified=False"
        )
    domain = np.asarray(query.domain, dtype=np.float64)
    if not np.all(np.isfinite(domain)) or np.any(domain[0] >= domain[1]):
        raise ValueError(
            f"prepare_implicit_volume_domain.query.domain must be a finite positive box; got {domain!r}"
        )
    limits = specification.limits
    if limits.maximum_vertices < 8 or limits.maximum_cells < 6:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.initialization must admit at least eight vertices and six cells; got insufficient initialization capacity",
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
    started = scope.operation_started
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
            snapshot = state.finalize()
            points = snapshot.points
            cells = snapshot.cells
            statistics = state.statistics()
            classification = _classify(
                query,
                points,
                cells,
                limits.maximum_geometry_queries - queries,
                scope=scope,
                limits=limits,
            )
            queries += int(np.count_nonzero(classification[5]))
            unknown = np.flatnonzero(
                classification[2]
                == int(
                    _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.UNKNOWN
                )
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
            remaining_work = min(
                limits.maximum_work_units - statistics.work,
                scope.remaining().remaining_work_units,
            )
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
                # Reclassify the partially committed snapshot before reporting refusal.
                snapshot = state.finalize()
                points = snapshot.points
                cells = snapshot.cells
                statistics = state.statistics()
                classification = _classify(
                    query,
                    points,
                    cells,
                    limits.maximum_geometry_queries - queries,
                    scope=scope,
                    limits=limits,
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
    A caller using center-enclosure bounds can certify finite-dual roots; it must
    independently establish that those bounds enclose the dual endpoints.
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
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    endpoint_bounds: np.ndarray,
    segments: np.ndarray,
    parameters: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    lower, upper, direction_lower, direction_upper = _line_boxes(
        endpoint_bounds, segments, parameters
    )
    box = query.bounds.boxes(np.min(lower, axis=1), np.max(upper, axis=1), budget=budget)
    point_bounds = query.classify_boxes(
        lower.reshape((-1, 3)), upper.reshape((-1, 3)), budget=budget
    )
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
    # its enclosed directional derivative has one strict sign, all field
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
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    enclosures: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    spatial_tolerance: float,
    remaining_queries: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[np.ndarray, ...], int, bool]:
    """Prove roots jointly across shared uncertain subdivision endpoints.

    A root exactly on a bisection plane need not have a strictly signed value
    at that plane. Adjacent unresolved pieces are joined and re-enclosed;
    strict outer signs and a monotone derivative establish the same unique
    root without requiring a pointwise zero-value assertion.
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
    bounds = _dual_interval_bounds(query, enclosures, ids, spans, budget=budget)
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


@checked
def isolate_implicit_dual_roots(
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    endpoints: ArrayLike,
    /,
    *,
    spatial_tolerance: float,
    maximum_depth: int,
    maximum_geometry_queries: int,
    endpoint_bounds: ArrayLike | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
) -> ImplicitDualRootWorkset:
    """Bound roots on numerical duals under an explicit or host-owned ticket."""

    limits = _dependency_phydrax_meshing__contracts.MeshingLimits()
    with _owner_phydrax_geometry_implicit__enclosure.query_budget(
        budget,
        work_upper=limits.maximum_work_units,
        queries_upper=2 * maximum_geometry_queries,
        bytes_upper=limits.maximum_scratch_bytes,
    ) as source_budget:
        return _isolate_implicit_dual_roots(
            query,
            endpoints,
            spatial_tolerance=spatial_tolerance,
            maximum_depth=maximum_depth,
            maximum_geometry_queries=maximum_geometry_queries,
            endpoint_bounds=endpoint_bounds,
            budget=source_budget,
        )


def _isolate_implicit_dual_roots(
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    endpoints: ArrayLike,
    /,
    *,
    spatial_tolerance: float,
    maximum_depth: int,
    maximum_geometry_queries: int,
    endpoint_bounds: ArrayLike | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> ImplicitDualRootWorkset:
    """Bound roots on all numerical duals, without sampled-empty exclusions.

    Every split replaces one interval with its two closed children; terminal
    intervals cover [0,1] for every input segment, even on exhausted budgets.
    Singular, zero-valued and multiple-root intervals remain in the partition.
    """
    if not query.certified:
        raise ValueError(
            "isolate_implicit_dual_roots.query must provide certified value enclosures; got certified=False"
        )
    points = np.asarray(endpoints, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (2, 3) or not np.all(np.isfinite(points)):
        raise ValueError(
            f"isolate_implicit_dual_roots.endpoints must be finite with shape (segments, 2, 3); got shape={points.shape}"
        )
    if not np.isfinite(spatial_tolerance) or spatial_tolerance <= 0.0:
        raise ValueError(
            f"isolate_implicit_dual_roots.spatial_tolerance must be positive and finite; got {spatial_tolerance!r}"
        )
    for value, name in (
        (maximum_depth, "maximum_depth"),
        (maximum_geometry_queries, "maximum_geometry_queries"),
    ):
        value = (
            nonnegative_integer
            if name == "maximum_geometry_queries"
            else positive_integer
        )(value, name)
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
            f"isolate_implicit_dual_roots.endpoint_bounds must be finite enclosing (segments, 2, 2, 3) intervals; got shape={enclosures.shape}"
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
        bounds = _dual_interval_bounds(
            query, enclosures, segments, parameters, budget=budget
        )
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
        query,
        enclosures,
        arrays,
        spatial_tolerance,
        maximum_geometry_queries - queries,
        budget=budget,
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
    Facet coverage remains unresolved while any query obligation remains.
    The domain workset retains the physical request and unapplied controls.
    ``work_units`` sums measured native phase counters only; it does not
    assign cell/ray/query counts a fabricated instruction cost. Source query
    cardinalities remain in ``geometry_queries``. The ambient
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


@checked
def prepare_implicit_restricted_queries(
    domain: ImplicitVolumeDomainWorkset,
    /,
    *,
    spatial_tolerance: float,
    maximum_depth: int,
    additional_work_units: int = 0,
    scope: NativeExecutionScope,
) -> ImplicitRestrictedQueryWorkset:
    """Recompute dual incidence and dual/source root enclosures.

    Native transactional insertion is owned by domain preparation. This phase
    snapshots its current triangulation, certifies finite Voronoi endpoints
    with expansion-based rational bounds, encloses hull rays through the domain
    exit and executes batched interval roots.
    Root exclusion requires a source enclosure rather than clipping.
    """
    domain.require_current(domain.query, domain.specification)
    limits = domain.specification.limits
    if additional_work_units < 0:
        raise ValueError(
            f"prepare_implicit_restricted_queries.additional_work_units must be nonnegative; got {additional_work_units!r}"
        )
    spent = dict(domain.native_statistics)["work"] + additional_work_units
    remaining = scope.remaining().remaining_work_units
    if remaining <= 0:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.dual_preparation must have remaining native work budget; got no remaining work",
            stage="implicit_restricted_queries",
        )
    native = restricted_dual_3d(
        domain.points,
        domain.tetrahedra,
        domain.query.domain,
        limits.maximum_faces,
        remaining,
    )
    facets = native.facets
    cells = native.incident
    endpoints = native.endpoints
    kinds = native.kinds
    status = native.item_status
    counters = native.counters
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
    query_upper = min(
        limits.maximum_geometry_queries - domain.geometry_queries,
        scope.remaining().remaining_geometry_queries,
    )
    with implicit_source_budget(limits, scope=scope, queries_upper=query_upper) as budget:
        roots = isolate_implicit_dual_roots(
            domain.query,
            representative,
            spatial_tolerance=spatial_tolerance,
            maximum_depth=maximum_depth,
            maximum_geometry_queries=query_upper,
            endpoint_bounds=endpoint_bounds,
            budget=budget,
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
        spent + counters.work_units,
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
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery,
    triangles: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose the dot product of oriented primal normals and source gradients."""
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
    bounds = query.bounds.boxes(
        np.min(triangles, axis=1), np.max(triangles, axis=1), budget=budget
    )
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


@checked
def assemble_restricted_implicit_topology(
    state: IncrementalDelaunay3D,
    work: ImplicitRestrictedQueryWorkset,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_deviation_upper: float | None = None,
    scope: NativeExecutionScope,
) -> RestrictedImplicitTopology:
    """Protect restricted primal faces and flood their two sides natively.

    Whole-cell source enclosures outside the certified boundary-field band
    identify interior and exterior seed cells. Cells in the approximation
    band follow the protected primal complex, never a centroid label. Independent
    degree-one source projection and approximate-domain coverage are mandatory
    before publication. Every restricted facet must separate the
    two regions, and every enclosed inside/outside cell must agree with it.
    """
    if work.unresolved_facets.size:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            "implicit_volume.dual_roots must resolve every dual/source intersection; got unresolved intersections",
            stage="implicit_restricted_topology",
            entity_ids=tuple(int(value) for value in work.unresolved_facets),
        )
    snapshot = state.finalize()
    points = snapshot.points
    cells = snapshot.cells
    if not np.array_equal(points, work.domain.points) or not np.array_equal(
        cells, work.domain.tetrahedra
    ):
        raise ValueError(
            f"assemble_restricted_implicit_topology.work must bind the triangulation snapshot; got workset_id={work.workset_id!r} with changed points or cells"
        )
    _, _, counts = _restricted_root_sites(work)
    if np.any(counts > 1):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            "implicit_volume.dual_roots must isolate at most one source intersection per dual; got multiple intersections requiring native refinement",
            stage="implicit_restricted_topology",
        )
    selected = np.flatnonzero(counts == 1)
    if not selected.size:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            "implicit_volume.restricted_boundary must exist unless domain work certifies an empty source; got missing boundary without empty-source certification",
            stage="implicit_restricted_topology",
        )
    with measure_phase(record_phase, "boundary_recovery"):
        constrained = state.constrain_facets(work.facets[selected], selected)
    if np.any(constrained != int(MeshcoreStatus.OK)):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "implicit_volume.restricted_facets must protect every queried primal facet; got unprotected queried facets",
            stage="implicit_restricted_topology",
        )
    triangles = points[work.facets[selected]]
    with implicit_source_budget(
        work.domain.specification.limits,
        scope=scope,
        queries_upper=2 * triangles.shape[0],
    ) as budget:
        boundary_values = work.domain.query.classify_boxes(
            np.min(triangles, axis=1),
            np.max(triangles, axis=1),
            budget=budget,
        )
    if not boundary_values.certified:
        raise ValueError(
            "assemble_restricted_implicit_topology.boundary_values must provide certified boundary intervals; got certified=False"
        )
    box_band = float(
        np.max(
            np.maximum(
                np.abs(boundary_values.value_lower), np.abs(boundary_values.value_upper)
            )
        )
    )
    # Reference subdivision certifies the same unchanged physical facets,
    # rather than their larger axis boxes, before the protected-complex flood.
    if source_deviation_upper is not None and (
        not np.isfinite(source_deviation_upper) or source_deviation_upper < 0.0
    ):
        raise ValueError(
            f"implicit_volume.source_deviation_upper must be a finite nonnegative facet bound; got {source_deviation_upper!r}"
        )
    field_band = (
        box_band
        if source_deviation_upper is None
        else min(box_band, source_deviation_upper)
    )
    # Source membership and affine approximation membership can differ
    # inside this measured band. Keep the source classes unchanged and refuse
    # contradictions beyond it; the projection proof establishes the
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
        # when its enclosed interior point is well outside. A certified point
        # seed drives the protected-complex flood; it never labels the cell
        # or establishes cell conformity from a centroid sign.
        hull_cells = np.unique(work.incident_cells[work.incident_cells[:, 1] < 0, 0])
        candidates = np.mean(points[cells[hull_cells]], axis=1)
        with implicit_source_budget(
            work.domain.specification.limits,
            scope=scope,
            queries_upper=2 * candidates.shape[0],
        ) as budget:
            source_points = work.domain.query.classify_boxes(
                candidates, candidates, budget=budget
            )
        exterior_rows = np.flatnonzero(source_points.value_lower > field_band)
        if not exterior_rows.size:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
                "implicit_volume.flood.exterior must have a source-enclosed exterior cell beyond the certified boundary band; got no exterior cell beyond the band",
                stage="implicit_restricted_topology",
            )
        exterior = candidates[exterior_rows[0]]
    if not witness_points:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            "implicit_volume.flood.interior must have a source-enclosed interior cell beyond the measured approximation band; got no interior cell beyond the band",
            stage="implicit_restricted_topology",
        )
    witnesses = np.stack((witness_points[0], exterior))
    with measure_phase(record_phase, "region_classification"):
        status = state.label_regions(witnesses, np.asarray((0, 1), dtype=np.int32))
    if np.any(status != int(MeshcoreStatus.OK)):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "implicit_volume.restricted_facets must separate source-enclosed interior and exterior cells; got nonseparating facets",
            stage="implicit_restricted_topology",
        )
    snapshot = state.finalize()
    points = snapshot.points
    cells = snapshot.cells
    regions = snapshot.cell_regions
    if (
        np.any(regions < 0)
        or np.any(regions[inside] != 0)
        or np.any(regions[outside] != 1)
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "implicit_volume.flood must agree with source enclosures beyond the certified boundary band; got contradictory region assignments",
            stage="implicit_restricted_topology",
        )
    owners = work.incident_cells[selected]
    first_inside = regions[owners[:, 0]] == 0
    second_inside = np.where(
        owners[:, 1] >= 0, regions[np.maximum(owners[:, 1], 0)] == 0, False
    )
    if np.any(first_inside == second_inside):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "implicit_volume.restricted_facets must separate the declared source regions; got nonseparating primal facet",
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            "implicit_volume.restricted_facets must form a closed oriented manifold; got failed manifold closure or orientation",
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


class ImplicitVolumeSeedKind(StrEnum):
    SPHERE_RADIAL_CONES = "sphere_radial_cones"
    SOURCE_ATLAS_COVER = "source_atlas_cover"


@dataclass(frozen=True, slots=True)
class PreparedImplicitVolume:
    """Bind a certified regular analytic source to a physical volume request."""

    profile: (
        _dependency_phydrax_geometry_implicit__analytic_profile.AnalyticImplicitProfile
    )
    query: _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule
    target_size: float
    fidelity_tolerance: float
    seed_cover_radius: float
    seed_kind: ImplicitVolumeSeedKind
    geometry_query_charge: int
    size_resolution_id: str
    prepared_id: str


@checked
def implicit_volume_support_issues(
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec, /
) -> tuple[str, ...]:
    """Unsupported controls are refused, not omitted from a positive profile."""
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
    if (
        specification.fill_strategy
        is not _dependency_phydrax_meshing__contracts.VolumeFillStrategy.SIMPLEX
    ):
        issues.append("the simplex fill strategy")
    scope = specification.boundary_scope
    if scope.entity_dimension != 2 or not np.array_equal(scope.entity_ids, [0]):
        issues.append(
            "the implicit-zero-set boundary entity covering every source component"
        )
    for control in specification.size_controls:
        match control:
            case _dependency_phydrax_meshing__sizing.UniformSizeControl():
                if not np.array_equal(control.scope.entity_ids, scope.entity_ids):
                    issues.append("uniform controls on the entire implicit boundary")
            case _:
                issues.append("nonuniform native restricted-domain sizing")
    for feature in specification.protected_features:
        if (
            feature.feature_kind
            is not _dependency_phydrax_meshing__controls.FeatureKind.SURFACE
            or not np.array_equal(feature.scope.entity_ids, scope.entity_ids)
        ):
            issues.append("sharp-feature/material-junction protected cavity execution")
        elif feature.hard and feature.maximum_deviation == 0.0:
            issues.append(
                "zero-deviation curved geometry represented by affine boundary facets"
            )
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
        issues.append("boundary layers through the layer-core route")
    if specification.periodic_constraints:
        issues.append(
            "periodic restricted-domain construction through the quotient owner"
        )
    return tuple(sorted(set(issues)))


@checked
def prepare_restricted_implicit_volume(
    geometry: _dependency_phydrax_geometry__contracts.CompiledGeometry,
    domain: ArrayLike,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    scope: NativeExecutionScope,
) -> PreparedImplicitVolume:
    """Prepare the certified analytic profile without extracting or filling a PLC."""
    issues = implicit_volume_support_issues(specification)
    if issues:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            f"implicit_volume.specification must contain supported controls; got unsupported controls={issues!r}",
            stage="implicit_volume_preparation",
        )
    _implicit_region_metadata(specification)
    if specification.limits.maximum_geometry_queries < 3:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.preparation must admit three bounded geometry requests; got insufficient geometry query capacity",
            stage="implicit_volume_preparation",
        )
    with implicit_source_budget(
        specification.limits,
        scope=scope,
        queries_upper=5 + 2 * len(specification.region_seeds),
    ) as budget:
        profile = _dependency_phydrax_geometry_implicit__analytic_profile.AnalyticImplicitProfile(
            geometry,
            coordinate_contract,
            source_id=specification.boundary_scope.source_id,
            source_revision=specification.boundary_scope.source_revision,
            budget=budget,
        )
        profile.require_bound(geometry, coordinate_contract, budget=budget)
        box = np.asarray(domain, dtype=np.float64)
        if (
            box.shape != (2, 3)
            or not np.all(np.isfinite(box))
            or np.any(box[0] > profile.boundary_bounds[0])
            or np.any(box[1] < profile.boundary_bounds[1])
        ):
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SOURCE,
                "implicit_volume.query.domain must enclose every source boundary component; got source boundary outside the query domain",
                stage="implicit_volume_preparation",
            )
        # The physical source is unchanged. A buffered computational hull avoids
        # truncating Voronoi rays at a source/domain tangency; the
        # declared enclosure remains separately bound in prepared_id.
        margin = profile.reach_lower / 16.0
        carrier = np.stack(
            (
                np.nextafter(box[0] - margin, -np.inf),
                np.nextafter(box[1] + margin, np.inf),
            )
        )
        enclosures = profile.field_bounds.boxes(carrier[:1], carrier[1:], budget=budget)
        query = (
            _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeQuery(
                profile.field_bounds,
                carrier[:1],
                carrier[1:],
                enclosures.value_lower,
                enclosures.value_upper,
                np.asarray((0,), dtype=np.int64),
                carrier,
                maximum_level=1,
            )
        )
        budget.ticket.consume(geometry_queries=1)
        point = np.asarray(
            geometry.boundary_atlas.map(
                jnp.asarray((0,), dtype=jnp.int32),
                jnp.asarray(((0.25, 0.25),), dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        field, resolution = _dependency_phydrax_meshing__sizing.resolve_size_controls(
            specification.size_controls,
            point,
            np.asarray((0,), dtype=np.int64),
            _dependency_phydrax_meshing__sizing.SizeFieldDomain.EUCLIDEAN_VOLUME,
            combination=specification.size_combination,
        )
        size = float(np.asarray(field.values)[0])
        tolerance = profile.tube_radius
        for feature in specification.protected_features:
            if feature.hard:
                tolerance = min(tolerance, feature.maximum_deviation)
        seed_queries = len(specification.region_seeds)
        if seed_queries + 3 > specification.limits.maximum_geometry_queries:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "implicit_volume.region_seeds must fit the geometry query budget; got material seed query capacity exhausted",
                stage="implicit_volume_preparation",
            )
        if seed_queries:
            seed_points = np.stack(
                tuple(np.asarray(seed.point) for seed in specification.region_seeds)
            )
            if seed_points.shape[1:] != (3,):
                raise ValueError(
                    f"prepare_restricted_implicit_volume.region_seeds must have three physical coordinates; got shape={seed_points.shape}"
                )
            seed_classes = query.classify_boxes(seed_points, seed_points, budget=budget)
            if np.any(
                seed_classes.classes
                != int(
                    _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.INSIDE
                )
            ):
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "implicit_volume.region_seeds must be source-enclosed strictly inside the analytic body; got noninterior material seed",
                    stage="implicit_volume_preparation",
                )
        # Sampling is numerical preparation, not the source-fidelity proof.
        # Whole facets are certified against the authored tolerance below;
        # their fixed boundary survives subsequent volume improvement.
        cover = (
            min(size, profile.reach_lower)
            if profile.family == "sphere"
            else min(
                0.5 * size,
                0.25 * profile.reach_lower,
                0.5 * np.sqrt(profile.reach_lower * tolerance),
            )
        )
        seed_kind = (
            ImplicitVolumeSeedKind.SPHERE_RADIAL_CONES
            if profile.family == "sphere"
            else ImplicitVolumeSeedKind.SOURCE_ATLAS_COVER
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
                "seed_kind": seed_kind.value,
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
            seed_kind,
            3 + seed_queries,
            resolution.report_id,
            identity,
        )


def _sphere_seed_sites(
    prepared: PreparedImplicitVolume,
    maximum_samples: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, float]:
    """Choose on-source sites by cone fidelity and the requested cover scale.

    Integer frequencies avoid the forced factor-of-two resolution jump of
    midpoint subdivision. Facet-wide intervals and the final degree-one
    projection still independently decide source acceptance.
    """
    from ..geometry._domain_numerics import (
        interval_vector_norm,
        sphere_radial_degree_one,
    )
    from ..geometry._interval_enclosure import (
        interval_add,
        interval_multiply,
        interval_subtract,
    )
    from ..geometry._sphere_material_atlas import sphere_radial_triangle_bounds
    from ..geometry.multiregion_surface._seeding import sphere_direction_lattice

    profile = prepared.profile
    ratio = profile.radius / prepared.seed_cover_radius
    if not np.isfinite(ratio) or ratio > maximum_samples:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.sphere_seeds cannot admit the requested source resolution",
            stage="implicit_volume_seeding",
            requested=(("seed_cover_radius", prepared.seed_cover_radius),),
            achieved=(("maximum_source_sites", float(maximum_samples)),),
        )
    frequency = max(1, int(np.ceil(ratio)))
    while True:
        vertices, faces = 10 * frequency**2 + 2, 20 * frequency**2
        if vertices > maximum_samples:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "implicit_volume.sphere_seeds must fit the remaining source-site allowance",
                stage="implicit_volume_seeding",
                requested=(("source_sites", float(vertices)),),
                achieved=(("maximum_source_sites", float(maximum_samples)),),
            )
        with budget.temporary_scope():
            # Positions, oriented faces, incidence, stable IDs, construction
            # copies, and outward cone preparations coexist in this phase.
            budget.reserve(
                10 * (frequency + 1) * (frequency + 2) + 3 * faces,
                1024 * (vertices + faces),
            )
            directions, triangles = sphere_direction_lattice(frequency)
            floor, ceiling = np.inf, 0.0
            for face in triangles:
                bound = sphere_radial_triangle_bounds(directions[face], budget=budget)
                if bound is None:
                    raise RuntimeError(
                        f"implicit_volume.sphere_seed_cones must be nondegenerate; got no cone bound at frequency {frequency}"
                    )
                floor, ceiling = min(floor, bound[1]), max(ceiling, bound[2])
            with budget.ticket.native_target():
                degree_one = sphere_radial_degree_one(
                    (np.zeros((3,), dtype=np.float64), 1.0, 1.0),
                    directions,
                    triangles,
                    budget=budget,
                )
            if not degree_one:
                raise RuntimeError(
                    f"implicit_volume.sphere_seed_cones must cover the source with degree one; got no degree-one cover at frequency {frequency}"
                )
            edges = native_route_checks_owner.unique_edges(triangles, "triangle")
            diameter = float(
                np.max(
                    _length_upper(
                        np.nextafter(
                            np.abs(directions[edges[:, 1]] - directions[edges[:, 0]]),
                            np.inf,
                        )
                    )
                )
            )
            radial_error = np.nextafter(max(1.0 - floor, ceiling - 1.0), np.inf)
            points = profile.center + profile.radius * directions
            radius = np.asarray(profile.radius, dtype=np.float64)
            exact_points = interval_add(
                interval_multiply((directions, directions), (radius, radius)),
                (profile.center, profile.center),
            )
            difference = interval_subtract((points, points), exact_points)
            realization = float(
                np.max(
                    interval_vector_norm(
                        (
                            np.asarray(difference[0], dtype=np.float64),
                            np.asarray(difference[1], dtype=np.float64),
                        )
                    )[1]
                )
            )
            error = np.nextafter(
                np.nextafter(profile.radius * radial_error, np.inf) + realization,
                np.inf,
            )
            # For every convex triangle point y, some corner v satisfies
            # |y-v|² <= sum_i lambda_i|y-v_i|² <= diameter²/3.
            # The rational degree-one cone cover extends this to the whole sphere.
            chord_cover = np.nextafter(
                diameter / np.nextafter(np.sqrt(3.0), -np.inf), np.inf
            )
            cover = np.nextafter(
                np.nextafter(
                    profile.radius * np.nextafter(chord_cover + radial_error, np.inf),
                    np.inf,
                )
                + realization,
                np.inf,
            )
            if (
                error <= prepared.fidelity_tolerance
                and cover <= prepared.seed_cover_radius
            ):
                realized = profile.boundary_distance(points, budget=budget)
                if np.max(realized.upper) <= prepared.fidelity_tolerance:
                    return points, float(cover)
        frequency += 1


def _implicit_seed_sites(
    prepared: PreparedImplicitVolume, /, *, scope: NativeExecutionScope
) -> tuple[np.ndarray, int, tuple[tuple[str, float], ...]]:
    """Return source-controlled boundary proposals and enclosed interior sites."""
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
    allowance = scope.remaining()
    if (
        lattice_count + 8 >= limits.maximum_vertices
        or lattice_count
        >= min(
            limits.maximum_geometry_queries - queries,
            allowance.remaining_geometry_queries,
        )
        or lattice_count * 128 > limits.maximum_scratch_bytes
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.interior_lattice must fit vertex, query and scratch budgets; got interior lattice capacity exhausted",
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
    with implicit_source_budget(
        limits, scope=scope, queries_upper=2 * lattice_count
    ) as budget:
        classification = query.classify_boxes(lattice, lattice, budget=budget)
    queries += lattice_count
    interior = lattice[
        classification.classes
        == int(
            _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.INSIDE
        )
    ]
    cover_budget = min(
        limits.maximum_vertices - interior.shape[0] - 8,
        min(
            limits.maximum_geometry_queries - queries,
            scope.remaining().remaining_geometry_queries,
        )
        // 3,
        limits.maximum_scratch_bytes // 512,
    )
    if cover_budget < 4:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.source_cover must have sufficient remaining capacity; got source-boundary cover capacity exhausted",
            stage="implicit_volume_seeding",
        )
    before_cover = scope.remaining().remaining_geometry_queries
    try:
        with implicit_source_budget(
            limits,
            scope=scope,
            queries_upper=min(
                limits.maximum_geometry_queries - queries,
                scope.remaining().remaining_geometry_queries,
            ),
        ) as budget:
            with budget.measurement():
                match prepared.seed_kind:
                    case ImplicitVolumeSeedKind.SPHERE_RADIAL_CONES:
                        boundary_points, boundary_cover_radius = _sphere_seed_sites(
                            prepared, cover_budget, budget=budget
                        )
                    case ImplicitVolumeSeedKind.SOURCE_ATLAS_COVER:
                        cover = profile.boundary_cover(
                            prepared.seed_cover_radius, cover_budget, budget=budget
                        )
                        if not cover.complete or cover.semantics != "certified":
                            raise ValueError(
                                f"implicit_volume.source_cover must cover the source with certified bounds; got covered={cover.complete}, semantics={cover.semantics!r}"
                            )
                        if np.max(cover.point_error) > prepared.fidelity_tolerance:
                            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                                _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
                                "implicit_volume.boundary_seeds must respect source fidelity",
                                stage="implicit_volume_seeding",
                            )
                        boundary_points = cover.points
                        boundary_cover_radius = float(np.max(cover.covering_radius))
                    case _:
                        raise ValueError(
                            f"implicit_volume.seed_kind must select supported source preparation; got {prepared.seed_kind!r}"
                        )
    except _dependency_phydrax_geometry_implicit__analytic_profile.AnalyticBoundaryCoverCapacityError as error:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            stage="implicit_volume_seeding",
            requested=(("cover_radius", error.requested_radius),),
            achieved=(("cover_radius", error.achieved_radius),),
        ) from error
    inside_sites = interior.shape[0]
    # AnalyticImplicitProfile admits nominal signed-distance kernels only.
    # A body site farther from the whole source than its certified site cover
    # cannot own a source Voronoi region. Refinement fills the protected body
    # afterward, without creating accidental near-boundary source strata.
    if prepared.seed_kind is ImplicitVolumeSeedKind.SPHERE_RADIAL_CONES:
        interior = lattice[
            (
                classification.classes
                == int(
                    _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.INSIDE
                )
            )
            & (classification.value_upper < -boundary_cover_radius)
        ]
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
    queries += before_cover - scope.remaining().remaining_geometry_queries
    statistics = (
        ("body_candidates", float(lattice_count)),
        ("body_outside", float(lattice_count - inside_sites)),
        ("body_retained", float(interior.shape[0])),
        ("body_source_separation_refused", float(inside_sites - interior.shape[0])),
        ("boundary_cover_radius", boundary_cover_radius),
        ("boundary_sites", float(boundary_points.shape[0])),
    )
    return np.concatenate((hull, boundary_points, interior)), queries, statistics


def _implicit_region_metadata(
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec, /
) -> tuple[str, str | None, _dependency_phydrax_meshing__organization.RegionRole | None]:
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.CONTROL_CONFLICT,
            "implicit_volume.region_controls must agree with the single analytic interior region; got conflicting seed or control material identities",
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
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.CONTROL_CONFLICT,
                "implicit_volume.patch_controls must name every source boundary component and its single interior region; got incompatible patch scope or region identity",
                stage="implicit_volume_preparation",
            )
    return name, material, role


def _restricted_facet_deviation(
    profile: _dependency_phydrax_geometry_implicit__analytic_profile.AnalyticImplicitProfile,
    points: np.ndarray,
    facets: np.ndarray,
    box_deviation: np.ndarray,
    diameter: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Bound the triangle itself, not unrelated points in its axis box.

    Signed distance is one-Lipschitz, so vertex error plus diameter first
    establishes containment in the profile's certified Hessian tube. For
    barycentric weights, the Taylor remainder is at most
    H/2 * sum(i<j, lambda_i*lambda_j*edge_ij**2) <= H*diameter**2/6.
    Keep the box enclosure whenever that tube condition is not established.
    """
    used, inverse = np.unique(facets, return_inverse=True)
    lower, upper = profile.field_bounds.point_values(points[used], budget=budget)
    vertex_error = np.maximum(np.abs(lower), np.abs(upper))
    error = np.max(vertex_error[inverse.reshape(facets.shape)], axis=1)
    in_tube = np.nextafter(error + diameter, np.inf) < profile.tube_radius
    squared = np.nextafter(diameter * diameter, np.inf)
    remainder = np.nextafter(profile.hessian_upper * squared, np.inf)
    remainder = np.nextafter(remainder / 6.0, np.inf)
    interpolation = np.nextafter(error + remainder, np.inf)
    return np.where(in_tube, np.minimum(box_deviation, interpolation), box_deviation)


def _restricted_facet_quality(
    prepared: PreparedImplicitVolume,
    work: ImplicitRestrictedQueryWorkset,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[np.ndarray, float, int]:
    """Flag true restricted facets whose continuous source proof is unresolved."""
    _, _, counts = _restricted_root_sites(work)
    selected = np.flatnonzero(counts == 1)
    bad = np.flatnonzero(counts > 1)
    if not selected.size:
        return np.union1d(bad, work.unresolved_facets), np.inf, 0
    triangles = work.domain.points[work.facets[selected]]
    remaining = min(
        prepared.specification.limits.maximum_geometry_queries - work.geometry_queries,
        scope.remaining().remaining_geometry_queries,
    )
    selected_facets = work.facets[selected]
    vertex_queries = np.unique(selected_facets).size
    query_upper = 4 * selected.size + vertex_queries
    if remaining < query_upper:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.facet_queries must fit the geometry query budget; got facet source and normal query capacity exhausted",
            stage="implicit_restricted_refinement",
        )
    before_queries = scope.remaining().remaining_geometry_queries
    with implicit_source_budget(
        prepared.specification.limits, scope=scope, queries_upper=remaining
    ) as budget:
        bounds = prepared.query.bounds.boxes(
            np.min(triangles, axis=1), np.max(triangles, axis=1), budget=budget
        )
        low, high = _boundary_normal_bounds(prepared.query, triangles, budget=budget)
        box_deviation = np.maximum(np.abs(bounds.value_lower), np.abs(bounds.value_upper))
        diameter = np.maximum.reduce(
            tuple(
                _length_upper(
                    np.nextafter(
                        np.abs(triangles[:, first] - triangles[:, second]), np.inf
                    )
                )
                for first, second in ((0, 1), (1, 2), (2, 0))
            )
        )
        deviation = _restricted_facet_deviation(
            prepared.profile,
            work.domain.points,
            selected_facets,
            box_deviation,
            diameter,
            budget=budget,
        )
        orientable = (low > 0.0) | (high < 0.0)
        unresolved = ~orientable | (deviation > prepared.fidelity_tolerance)
        if np.any(unresolved):
            candidates = triangles[unresolved].copy()
            reverse = (low + high)[unresolved] < 0.0
            candidates[reverse] = candidates[reverse, ::-1]
            certificate = _owner_phydrax_meshing_providers__native_publication.NativeCertificationRequest(
                _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
                    "volume_implicit"
                ),
                prepared.profile.source_id,
                prepared.profile.source_revision,
                prepared.specification.limits,
            )
            continuous, normals, _ = (
                implicit_projection_owner.implicit_facet_projection_bounds(
                    candidates,
                    prepared.profile,
                    prepared.fidelity_tolerance,
                    certificate.certificate_limits,
                    budget=budget,
                )
            )
            deviation[unresolved] = continuous
            orientable[unresolved] = normals
    failed = ~orientable | (deviation > prepared.fidelity_tolerance)
    bad = np.union1d(bad, selected[failed])
    return (
        np.union1d(bad, work.unresolved_facets),
        float(np.max(deviation)),
        before_queries - scope.remaining().remaining_geometry_queries,
    )


def _restricted_closed_profile(
    prepared: PreparedImplicitVolume,
    work: ImplicitRestrictedQueryWorkset,
    /,
    *,
    scope: NativeExecutionScope,
) -> bool:
    """Check closed oriented source topology before any native protection edit."""
    _, _, counts = _restricted_root_sites(work)
    selected = np.flatnonzero(counts == 1)
    if not selected.size or np.any(counts > 1) or work.unresolved_facets.size:
        return False
    triangles = work.domain.points[work.facets[selected]]
    with implicit_source_budget(
        prepared.specification.limits, scope=scope, queries_upper=2 * selected.size
    ) as budget:
        low, high = _boundary_normal_bounds(prepared.query, triangles, budget=budget)
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
    # Primal geometry comes from positively oriented native tetrahedra.
    # This phase checks combinatorial closure; later audits check coordinate
    # validity and shape independently of TriangleMesh's area filter.
    topology = TriangleTopology(reindex[faces], num_vertices=used.size)
    return (
        topology.num_face_components == prepared.profile.component_count
        and topology.euler_characteristic
        == prepared.profile.boundary_euler_characteristic
    )


def _protected_insertion_batch(points: np.ndarray, radii: np.ndarray, /) -> np.ndarray:
    """Schedule separated empty-ball candidates from one triangulation epoch.

    Adjacent duals can request the same surface site. Native insertion changes
    the triangulation after the first edit, invalidating its previous empty-ball
    guarantees. Require each accepted site to lie outside every other
    accepted site's entire empty ball, and recompute deferred duals next round.
    """
    if points.shape[0] < 2:
        return points
    protection = radii
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
    scope: NativeExecutionScope,
) -> tuple[
    RestrictedImplicitTopology,
    ImplicitRestrictedQueryWorkset,
    int,
    int,
    float,
    tuple[tuple[str, float], ...],
]:
    """Native Delaunay site refinement followed by protected primal-region flood."""

    profile = prepared.profile
    with implicit_source_budget(
        prepared.specification.limits, scope=scope, queries_upper=1
    ) as budget:
        profile.require_bound(
            profile.geometry, profile.coordinate_contract, budget=budget
        )
    limits = prepared.specification.limits
    with measure_phase(record_phase, "source_preparation"):
        sites, queries, seed_statistics = _implicit_seed_sites(prepared, scope=scope)
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
            _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
                limits,
                _dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL,
                scope=scope,
            )
            snapshot = state.finalize()
            points = snapshot.points
            cells = snapshot.cells
            statistics = state.statistics()
            classification = _classify(
                prepared.query,
                points,
                cells,
                limits.maximum_geometry_queries - queries,
                scope=scope,
                limits=limits,
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
                scope=scope,
            )
            queries = work.geometry_queries
            extra_work = work.work_units - statistics.work
            bad, deviation, facet_queries = _restricted_facet_quality(
                prepared, work, scope=scope
            )
            queries += facet_queries
            if not bad.size:
                before_closure = scope.remaining().remaining_geometry_queries
                closed = _restricted_closed_profile(prepared, work, scope=scope)
                queries += before_closure - scope.remaining().remaining_geometry_queries
                if closed:
                    topology = assemble_restricted_implicit_topology(
                        state,
                        work,
                        record_phase=record_phase,
                        source_deviation_upper=deviation,
                        scope=scope,
                    )
                    region_work = state.statistics().work - statistics.work
                    return (
                        topology,
                        work,
                        work.work_units + region_work,
                        queries,
                        deviation,
                        seed_statistics,
                    )
                # All dual proofs remain meaningful, but not a source-bound
                # manifold yet. Refine their source roots rather than PLC faces.
                _, _, counts = _restricted_root_sites(work)
                bad = np.flatnonzero(counts > 0)
            if round_ == prepared.schedule.refinement_rounds:
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
                    "implicit_volume.refinement must certify primal/source conformity within the native refinement budget; got unresolved conformity after refinement",
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
                # covering carrier near the unresolved dual incidences.
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
            remaining_vertices = limits.maximum_vertices - statistics.vertex_ids
            if additions.shape[0] > remaining_vertices or not additions.shape[0]:
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "implicit_volume.insertion must admit the next site batch; got insufficient insertion capacity",
                    stage="implicit_restricted_refinement",
                )
            before = statistics.live_vertices
            started_insertion = phase_started(record_phase)
            _, status = state.insert(
                additions,
                work_limit=scope.remaining().remaining_work_units,
            )
            after = state.statistics()
            record_elapsed(
                record_phase,
                "site_refinement",
                started_insertion,
                work_units=after.work - statistics.work,
            )
            if np.any(status != int(MeshcoreStatus.OK)) or after.live_vertices <= before:
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "implicit_volume.insertion must be admitted and make geometric progress; got refused or nonprogressing insertion",
                    stage="implicit_restricted_refinement",
                    checkpoint_id=work.workset_id,
                    entity_ids=tuple(
                        int(value)
                        for value in np.flatnonzero(status != int(MeshcoreStatus.OK))
                    ),
                    requested=(("insertion_sites", additions.shape[0]),),
                    achieved=(
                        ("inserted_vertices", after.live_vertices - before),
                        (
                            "refused_sites",
                            np.count_nonzero(status != int(MeshcoreStatus.OK)),
                        ),
                        ("live_vertices", after.live_vertices),
                        ("finite_cells", after.finite_cells),
                        ("insertion_work_units", after.work - statistics.work),
                    ),
                )
        raise RuntimeError(
            "generate_restricted_implicit_volume.refinement must return a result or report budget refusal; got no terminal result"
        )
    finally:
        state.close()


@dataclass(frozen=True, slots=True)
class ImplicitVolumeConstruction:
    """Retain native restricted-domain tetrahedra, geometry and phase evidence.

    ``exudation`` is the schedule's explicit weighted stage run (``None`` when
    disabled) and ``exudation_weights`` its accepted weights per published vertex.
    """

    mesh: CellMesh
    domain: domain_coverage_owner.PiecewiseLinearDomain
    cell_regions: np.ndarray
    zones: tuple[_dependency_phydrax_meshing__organization.MeshZone, ...]
    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...]
    associations: tuple[association_state_owner.GeometryAssociation, ...]
    stages: tuple[_dependency_phydrax_meshing__trace.MeshingStageReport, ...]
    refinement: TetMeshRun[TetMeshRefineCounters]
    improvement: TetMeshRun[TetMeshImproveCounters]
    minimum_dihedral: float
    maximum_radius_edge: float
    source_deviation_upper: float
    work_units: int
    geometry_query_charge: int
    restricted_topology_id: str
    restricted_query_id: str
    exudation: TetMeshRun[TetMeshExudeCounters] | None = None
    exudation_weights: np.ndarray | None = None
    seed_statistics: tuple[tuple[str, float], ...] = ()


def _improve_implicit_volume(
    topology: RestrictedImplicitTopology,
    prepared: PreparedImplicitVolume,
    work_units: int,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TetMeshRun[TetMeshRefineCounters],
    TetMeshRun[TetMeshImproveCounters],
    TetMeshRun[TetMeshExudeCounters] | None,
    np.ndarray | None,
    float,
    float,
    int,
]:
    """Run native tetrahedral quality refinement and improvement on the restricted body."""

    _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
        prepared.specification.limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.OPTIMIZATION,
        scope=scope,
    )

    limits = prepared.specification.limits
    boundary = topology.boundary_facets
    edges = native_route_checks_owner.unique_edges(boundary, "triangle")
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
        _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
            limits,
            _dependency_phydrax_meshing__trace.MeshingStageKind.OPTIMIZATION,
            scope=scope,
        )
        refinement = state.refine(
            radius_edge_bound=prepared.schedule.radius_edge_bound,
            sizes=np.full(
                topology.points.shape[0], prepared.target_size, dtype=np.float64
            ),
            max_insertions=limits.maximum_vertices - topology.points.shape[0],
            work_limit=scope.remaining().remaining_work_units,
        )
        refinement_work = refinement.counters.work_units
        record_elapsed(record_phase, "refinement", start, work_units=refinement_work)
        work_units += refinement_work
        if (
            refinement.status == MeshcoreStatus.CAPACITY_EXCEEDED
            or work_units >= limits.maximum_work_units
        ):
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "implicit_volume.quality_refinement must fit native capacity; got quality refinement capacity exhausted",
                stage="implicit_volume_quality",
            )
        start = phase_started(record_phase)
        _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
            limits,
            _dependency_phydrax_meshing__trace.MeshingStageKind.OPTIMIZATION,
            scope=scope,
        )
        improvement = state.improve(
            min_dihedral_degrees=prepared.schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=validity_policy.relative_determinant_floor,
            max_passes=prepared.schedule.improvement_passes,
            work_limit=scope.remaining().remaining_work_units,
        )
        improvement_work = improvement.counters.work_units
        record_elapsed(record_phase, "improvement", start, work_units=improvement_work)
        work_units += improvement_work
        if improvement.status == MeshcoreStatus.CAPACITY_EXCEEDED:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "implicit_volume.improvement must fit native capacity; got tetrahedral improvement capacity exhausted",
                stage="implicit_volume_quality",
            )
        exudation, weights, work_units = (
            _dependency_phydrax_meshing__volume_generation.exude_native_volume(
                state,
                prepared.schedule,
                limits,
                work_units,
                validity_policy=validity_policy,
                record_phase=record_phase,
                scope=scope,
            )
        )
        _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
            limits,
            _dependency_phydrax_meshing__trace.MeshingStageKind.OPTIMIZATION,
            scope=scope,
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
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    source_deviation: float,
    source_id: str,
    source_revision: str,
    /,
) -> tuple[
    tuple[_dependency_phydrax_meshing__organization.MeshZone, ...],
    tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...],
    tuple[association_state_owner.GeometryAssociation, ...],
]:
    connectivity = mesh.connectivity
    match connectivity:
        case TetrahedralConnectivity():
            name, material, role = _implicit_region_metadata(specification)
        case _:
            raise TypeError(
                f"execute_restricted_implicit_volume.mesh.connectivity must be TetrahedralConnectivity; got {type(connectivity).__name__}"
            )
    cells = mesh.entity_set(3)
    scope = _dependency_phydrax_meshing__scope.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH,
        3,
        cells.entity_set_id,
        cells.entity_ids,
    )
    zone = _dependency_phydrax_meshing__organization.MeshZone(
        name,
        _dependency_phydrax_meshing__organization.MeshZoneRole.REGION,
        scope,
        material_id=material,
        region_role=role,
    )
    faces = mesh.entity_set(2)
    identifiers = np.asarray(faces.entity_ids)[np.asarray(connectivity.boundary_faces)]
    boundary_scope = _dependency_phydrax_meshing__scope.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH,
        2,
        faces.entity_set_id,
        identifiers,
    )
    patch_names = tuple(patch.name for patch in specification.patch_controls) or (
        "implicit-boundary",
    )
    patches = tuple(
        _dependency_phydrax_meshing__organization.MeshPatch(
            patch_name, boundary_scope, adjacent_zone_ids=(zone.zone_id,)
        )
        for patch_name in patch_names
    )
    association = association_state_owner.GeometryAssociation(
        association_state_owner.GeometryAssociationKind.IMPLICIT,
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


@checked
def generate_restricted_implicit_volume(
    prepared: PreparedImplicitVolume,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
) -> ImplicitVolumeConstruction:
    """Construct the restricted-Delaunay body under an explicit request scope."""

    topology, work, spent, queries, deviation, seed_statistics = (
        _construct_implicit_restriction(
            prepared,
            record_phase=record_phase,
            scope=scope,
        )
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
        scope=scope,
    )
    with measure_phase(record_phase, "publication"):
        _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
            prepared.specification.limits,
            _dependency_phydrax_meshing__trace.MeshingStageKind.CANONICALIZATION,
            scope=scope,
        )
        mesh = _dependency_phydrax_meshing__canonical.canonicalize_cell_mesh(
            CellMesh.from_tetrahedra(
                points, cells, numeric_version=prepared.profile.source_revision
            )
        )
        name, _, _ = _implicit_region_metadata(prepared.specification)
        domain = domain_coverage_owner.PiecewiseLinearDomain(
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
        _dependency_phydrax_meshing__trace.MeshingStageReport(
            _dependency_phydrax_meshing__trace.MeshingStageKind.SOURCE_INSPECTION,
            _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED,
            input_ids=(prepared.prepared_id,),
            output_ids=(prepared.profile.profile_id, prepared.size_resolution_id),
        ),
        _dependency_phydrax_meshing__trace.MeshingStageReport(
            _dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL,
            _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED,
            input_ids=(work.workset_id,),
            output_ids=(topology.topology_id, domain.domain_id),
            created_count=topology.tetrahedra.shape[0],
        ),
        _dependency_phydrax_meshing__trace.MeshingStageReport(
            _dependency_phydrax_meshing__trace.MeshingStageKind.OPTIMIZATION,
            _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED
            if refinement.status == improvement.status == MeshcoreStatus.OK
            and (exudation is None or exudation.status == MeshcoreStatus.OK)
            else _dependency_phydrax_meshing__trace.MeshingStageStatus.WARNING,
            input_ids=(topology.topology_id,),
            output_ids=(mesh.mesh_id,),
            created_count=mesh.blocks[0].cell_count,
        ),
    )
    _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
        prepared.specification.limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.CANONICALIZATION,
        scope=scope,
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
        seed_statistics=seed_statistics,
    )


def _implicit_volume_compliance(
    construction: ImplicitVolumeConstruction,
    prepared: PreparedImplicitVolume,
    /,
) -> _dependency_phydrax_meshing__result.MeshingComplianceReport:

    connectivity = construction.mesh.connectivity
    match connectivity:
        case TetrahedralConnectivity():
            lengths, growth = native_route_checks_owner.edge_size_evidence(
                np.asarray(construction.mesh.coordinates),
                np.asarray(connectivity.edges)[np.asarray(connectivity.boundary_edges)],
            )
        case _:
            raise TypeError(
                f"execute_restricted_implicit_volume.mesh.connectivity must be TetrahedralConnectivity; got {type(connectivity).__name__}"
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
        *(
            (f"source_seeding:{name}", value)
            for name, value in construction.seed_statistics
        ),
    ]
    issues: list[str] = []
    for control in prepared.specification.size_controls:
        match control:
            case _dependency_phydrax_meshing__sizing.UniformSizeControl():
                request, actual, failed = (
                    native_route_checks_owner.uniform_size_compliance(
                        control, prepared.specification.size_compliance, lengths, growth
                    )
                )
            case _:
                raise TypeError(
                    f"execute_restricted_implicit_volume.size_controls must contain UniformSizeControl values; got {type(control).__name__}"
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
    return _dependency_phydrax_meshing__result.MeshingComplianceReport(
        prepared.specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


@checked
def execute_restricted_implicit_volume(
    geometry: _dependency_phydrax_geometry__contracts.CompiledGeometry,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    prepared: PreparedImplicitVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> _dependency_phydrax_meshing__result.CellMeshingResult:
    """Certify and publish analytic restricted-Delaunay generation."""

    with implicit_source_budget(
        specification.limits, scope=scope, queries_upper=1
    ) as budget:
        prepared.profile.require_bound(geometry, coordinate_contract, budget=budget)
    if specification.specification_id != prepared.specification.specification_id:
        raise ValueError(
            f"execute_restricted_implicit_volume.prepared.specification must bind the physical request; got prepared_id={prepared.specification.specification_id!r}, requested_id={specification.specification_id!r}"
        )

    # The native determinant-floor repair and the publication audit share one policy.
    audit_policy = _owner_phydrax_meshing__audit.CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=_owner_phydrax_meshing__audit.CellMeshAuditDisposition.REJECT,
    )
    construction = generate_restricted_implicit_volume(
        prepared,
        validity_policy=audit_policy.validity_policy,
        record_phase=record_phase,
        scope=scope,
    )
    native_route_checks_owner.simplex_entity_limits(
        np.asarray(construction.mesh.coordinates),
        np.asarray(construction.mesh.blocks[0].vertices),
        specification.limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL,
        cell_kind=construction.mesh.blocks[0].cell_kind,
    )
    with measure_phase(record_phase, "compliance"):
        _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
            specification.limits,
            _dependency_phydrax_meshing__trace.MeshingStageKind.SPECIFICATION_COMPLIANCE,
            scope=scope,
        )
        compliance = _implicit_volume_compliance(construction, prepared)
    return _owner_phydrax_meshing_providers__native_publication.publish_native_result(
        construction.mesh,
        coordinate_contract,
        compliance,
        construction.stages,
        provider,
        {
            "kind": "native-restricted-implicit-volume",
            "route": "implicit_restricted_delaunay",
            "plan": plan_id,
            "specification": specification.specification_id,
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
            "refinement_counters": construction.refinement.counters.items(),
            "improvement_counters": construction.improvement.counters.items(),
            "exudation_status": (
                None
                if construction.exudation is None
                else construction.exudation.status.name
            ),
            "exudation_counters": ()
            if construction.exudation is None
            else construction.exudation.counters.items(),
        },
        _owner_phydrax_meshing_providers__native_publication.NativeCertificationRequest(
            _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
                "volume_implicit"
            ),
            prepared.profile.source_id,
            prepared.profile.source_revision,
            specification.limits,
            domain=construction.domain,
            cell_regions=construction.cell_regions,
            fidelity_source=source_fidelity_owner.ImplicitProjectionBoundarySource(
                prepared.profile
            ),
            fidelity_tolerance=prepared.fidelity_tolerance,
        ),
        policy=NativePublicationPolicy(
            audit_policy,
            _dependency_phydrax_meshing__contracts.MeshingDerivativeMode.NONDIFFERENTIABLE,
            (
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
            (
                "scratch_bytes:unmanaged_host_device_compiler",
                "work_units:unmetered_host_device_compiler",
                "wall_seconds:nonpreemptible_host_device_compiler",
            ),
        ),
        organization=_owner_phydrax_meshing_providers__native_publication.NativePublicationOrganization(
            patches=construction.patches,
            zones=construction.zones,
            associations=construction.associations,
        ),
        execution=_owner_phydrax_meshing_providers__native_publication.NativePublicationExecution(
            scope, workspace, record_phase, None
        ),
    )


def _adaptive_source_state_id(
    geometry: _dependency_phydrax_geometry__contracts.CompiledGeometry, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "adaptive-implicit-source-state",
            "state": _dependency_phydrax_geometry__certified_implicit.implicit_state_id(
                geometry
            ),
            "kernel_structure": str(jax.tree_util.tree_structure(geometry.kernel)),
            "kernel_data": array_tree_fingerprint(geometry.kernel),
        }
    )


@final
class PreparedAdaptiveImplicitVolume(StrictModule, NonTrainableState):
    """Bind a certified adaptive zero set to its extracted constrained carrier."""

    geometry: _dependency_phydrax_geometry__contracts.CompiledGeometry
    surface: (
        _dependency_phydrax_geometry_implicit__adaptive_discovery.AdaptiveImplicitSurface
    )
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec
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

    @checked
    def __init__(
        self,
        geometry: _dependency_phydrax_geometry__contracts.CompiledGeometry,
        surface: _dependency_phydrax_geometry_implicit__adaptive_discovery.AdaptiveImplicitSurface,
        specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
        schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
        source_revision: str,
        fidelity_tolerance: float,
        geometry_queries: int,
        source_work_units: int,
        preparation_seconds: float,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        _dependency_phydrax_geometry_implicit__adaptive_discovery.AdaptiveImplicitBoundarySource(
            geometry, surface, source_revision
        )
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


@checked
def adaptive_implicit_volume_support_issues(
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec, /
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
            issues.append("patch scopes covering every component of the scalar zero set")
    return tuple(sorted(set(issues)))


@checked
def classify_adaptive_volume_seeds(
    surface: _dependency_phydrax_geometry_implicit__adaptive_discovery.AdaptiveImplicitSurface,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    queries: int,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> int:
    """Classify region and hole seeds after certified source discovery."""
    if not surface.evidence.certified or not surface.volume.certified:
        raise ValueError(
            f"classify_adaptive_volume_seeds.surface must have certified surface and volume evidence; got surface_certified={surface.evidence.certified!r}, volume_certified={surface.volume.certified!r}"
        )
    seed_queries = len(specification.region_seeds) + len(specification.hole_seeds)
    if queries + seed_queries >= specification.limits.maximum_geometry_queries:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.adaptive_seeds must fit the geometry query budget; got adaptive seed query capacity exhausted",
            stage="implicit_volume_preparation",
            achieved=(
                ("geometry_queries", queries),
                ("requested_seed_queries", seed_queries),
            ),
        )
    for seed, expected in (
        *(
            (
                seed,
                _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.INSIDE,
            )
            for seed in specification.region_seeds
        ),
        *(
            (
                seed,
                _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.OUTSIDE,
            )
            for seed in specification.hole_seeds
        ),
    ):
        point = np.asarray(seed.point, dtype=np.float64).reshape((1, 3))
        lower, upper = surface.volume.bounds.point_values(point, budget=budget)
        admitted = (
            upper[0] < 0.0
            if expected
            is _dependency_phydrax_geometry_implicit__adaptive_discovery.ImplicitVolumeClass.INSIDE
            else lower[0] > 0.0
        )
        if not admitted:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "implicit_volume.adaptive_seeds must agree with the certified scalar source; got contradictory region or hole seed",
                stage="implicit_volume_preparation",
            )
    return queries + seed_queries


@checked
def prepare_adaptive_implicit_volume(
    geometry: _dependency_phydrax_geometry__contracts.CompiledGeometry,
    domain: ArrayLike,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: _dependency_phydrax_meshing__volume_generation.NativeVolumeSchedule,
    policy: AdaptiveImplicitSurfacePolicy,
    source_revision: str,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    scope: NativeExecutionScope,
) -> PreparedAdaptiveImplicitVolume:
    """Discover every possible component before recovering the bounded PLC."""

    issues = adaptive_implicit_volume_support_issues(specification)
    if issues:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            f"implicit_volume.specification must contain supported controls; got unsupported controls={issues!r}",
            stage="implicit_volume_preparation",
        )
    _implicit_region_metadata(specification)
    if policy.enclosure == "sampled" or policy.allow_approximate_zero_set:
        raise ValueError(
            f"prepare_adaptive_implicit_volume.policy must provide certified zero-set enclosures without approximate-zero-set admission; got enclosure={policy.enclosure!r}, allow_approximate_zero_set={policy.allow_approximate_zero_set!r}"
        )
    limits = specification.limits
    started = monotonic()
    # Bound the host tree and native predicates before discovery.
    # Each leaf can own 82 entity queries and their directional enclosures,
    # integer incidence, sorting and extraction buffers simultaneously.
    box_capacity = min(
        policy.maximum_boxes,
        limits.maximum_scratch_bytes // 32768,
        limits.maximum_data_bytes // 2048,
    )
    if box_capacity < 1 << (3 * policy.initial_level):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.adaptive_cover must fit scratch, retained-data and work capacity; got initial adaptive cover capacity exhausted",
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
    with implicit_source_budget(
        specification.limits, scope=scope, queries_upper=limits.maximum_geometry_queries
    ) as budget:
        surface = _dependency_phydrax_geometry_implicit__adaptive_discovery.discover_adaptive_implicit_surface(
            geometry,
            domain=domain,
            policy=bounded,
            source_id=specification.boundary_scope.source_id,
            maximum_root_solves=limits.maximum_work_units,
            budget=budget,
        )
    _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
        limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.SOURCE_INSPECTION,
        scope=scope,
    )
    evidence = surface.evidence
    if (
        not evidence.certified
        or evidence.unresolved_count
        or surface.mesh is None
        or surface.topology is None
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            "implicit_volume.adaptive_source must certify regular closed zero-set components; got unresolved, singular, hidden or unclosed components",
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
    native_route_checks_owner.simplex_entity_limits(
        points,
        faces,
        limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.SOURCE_INSPECTION,
        cell_kind="triangle",
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.cycle_roots must fit the geometry query budget; got cycle-root query capacity exhausted",
            stage="implicit_volume_preparation",
        )
    with implicit_source_budget(
        specification.limits,
        scope=scope,
        queries_upper=witness_points.shape[0] * witness_points.shape[1],
    ) as budget:
        lower, upper = surface.volume.bounds.point_values(
            witness_points.reshape((-1, 3)), budget=budget
        )
    negative = np.any(upper.reshape((-1, 27)) < 0.0, axis=1)
    positive = np.any(lower.reshape((-1, 27)) > 0.0, axis=1)
    if not np.all(negative & positive):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
            "implicit_volume.cycle_roots must have independently source-enclosed root brackets; got unbracketed adaptive cycle",
            stage="implicit_volume_preparation",
            entity_ids=tuple(np.flatnonzero(~(negative & positive)).tolist()),
        )
    source_work = evidence.root_solves
    if source_work >= limits.maximum_work_units:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "implicit_volume.adaptive_discovery must leave sufficient work for native construction; got native construction work capacity exhausted",
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
            f"prepare_adaptive_implicit_volume.fidelity_tolerance must be positive for affine boundary publication; got {tolerance!r}"
        )
    with implicit_source_budget(
        specification.limits,
        scope=scope,
        queries_upper=len(specification.region_seeds) + len(specification.hole_seeds),
    ) as budget:
        queries = classify_adaptive_volume_seeds(
            surface, specification, queries, budget=budget
        )
    _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
        limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.SOURCE_INSPECTION,
        scope=scope,
    )
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


@checked
def execute_adaptive_implicit_volume(
    geometry: _dependency_phydrax_geometry__contracts.CompiledGeometry,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    prepared: PreparedAdaptiveImplicitVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: _dependency_phydrax_meshing__contracts.MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> _dependency_phydrax_meshing__result.CellMeshingResult:
    """Fill the extracted source and certify continuous fidelity independently."""

    if (
        _adaptive_source_state_id(geometry) != prepared.source_state_id
        or specification.specification_id != prepared.specification_id
        or canonical_fingerprint(array_tree_fingerprint(prepared.surface))
        != prepared.surface_numeric_id
        or coordinate_contract.spatial_id != prepared.coordinate_contract.spatial_id
    ):
        raise ValueError(
            f"execute_adaptive_implicit_volume.prepared must bind the current source, specification and coordinate contract; got stale prepared_id={prepared.prepared_id!r}"
        )
    surface = prepared.surface
    if surface.mesh is None:
        raise RuntimeError(
            "execute_adaptive_implicit_volume.prepared.surface.mesh must retain the certified boundary; got None"
        )
    source = _dependency_phydrax_geometry_implicit__adaptive_discovery.AdaptiveImplicitBoundarySource(
        geometry,
        surface,
        prepared.source_revision,
        maximum_distance_pairs=specification.limits.maximum_work_units,
        maximum_scratch_bytes=specification.limits.maximum_scratch_bytes,
    )
    faces = np.asarray(surface.mesh.faces, dtype=np.int64)
    name, _, _ = _implicit_region_metadata(specification)
    complex_ = plc_source_owner.PiecewiseLinearComplex(
        surface.mesh.vertices,
        tuple(face for face in faces),
        np.zeros(faces.shape[0], dtype=np.int64),
        np.asarray(((-1, 0),), dtype=np.int64),
        (name,),
        boundary="conforming",
    )
    audit_policy = _owner_phydrax_meshing__audit.CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=_owner_phydrax_meshing__audit.CellMeshAuditDisposition.REJECT,
    )
    construction = _owner_phydrax_meshing__volume_generation.generate_plc_volume(
        complex_,
        specification,
        prepared.schedule,
        validity_policy=audit_policy.validity_policy,
        source_id=prepared.source_id,
        source_revision=prepared.source_revision,
        input_id=prepared.prepared_id,
        record_phase=record_phase,
        source_work_units=prepared.source_work_units,
        scope=scope,
    )
    points = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    cells = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    compliance = _owner_phydrax_meshing_providers__native_volume.volume_compliance(
        specification, prepared.schedule, construction, points, cells
    )
    protected_quantities = {
        f"protected:{feature.feature_id}:maximum_deviation"
        for feature in specification.protected_features
    }
    compliance = _dependency_phydrax_meshing__result.MeshingComplianceReport(
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
    _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
        specification.limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.CERTIFICATION,
        scope=scope,
    )
    result = _owner_phydrax_meshing_providers__native_publication.publish_native_result(
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
        _owner_phydrax_meshing_providers__native_publication.NativeCertificationRequest(
            _leaf_phydrax_meshing__certification_schedule.MeshCertificationSchedule(
                "volume_implicit"
            ),
            prepared.source_id,
            prepared.source_revision,
            specification.limits,
            domain=_owner_phydrax_meshing__volume_generation.declared_plc_domain(
                complex_, prepared.source_id
            ),
            cell_regions=construction.cell_regions,
            fidelity_source=source,
            fidelity_tolerance=prepared.fidelity_tolerance,
        ),
        policy=NativePublicationPolicy(
            audit_policy,
            _dependency_phydrax_meshing__contracts.MeshingDerivativeMode.NONDIFFERENTIABLE,
            (
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
            (
                "scratch_bytes:unmanaged_host_device_compiler",
                "work_units:unmetered_host_device_compiler",
                "wall_seconds:nonpreemptible_host_device_compiler",
            ),
        ),
        organization=_owner_phydrax_meshing_providers__native_publication.NativePublicationOrganization(
            patches=patches, zones=zones, associations=associations
        ),
        execution=_owner_phydrax_meshing_providers__native_publication.NativePublicationExecution(
            scope, workspace, record_phase, None
        ),
    )
    _owner_phydrax_meshing__volume_generation.native_volume_checkpoint(
        specification.limits,
        _dependency_phydrax_meshing__trace.MeshingStageKind.SPECIFICATION_COMPLIANCE,
        scope=scope,
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
