#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native constrained tetrahedral generation of oriented piecewise-linear complexes.

Phase order; each phase refuses with stage evidence before the next starts:

1. source contract (`PiecewiseLinearComplex`): structural validation of the
   oriented polygons, facet region incidence, internal curves and regions;
2. boundary recovery (`phydrax._meshcore.recover_plc_3d`): exact PLC
   validation, protected segment recovery, facet recovery by constrained
   cavity fills and region classification by flooding from both sides of every
   constrained facet, holes and seeds, under the fixed or conforming boundary
   policy of the source; conforming Steiner points are exact or, where a
   protected surface, material-interface or curve feature declares a positive
   ``maximum_deviation``, ancestry-backed carriers within that bound;
3. native shape refinement/improvement and the shared in-place metric epoch
   for admitted uniform statistical goals, under the original size request
   and bounded numerical schedule;
4. assembly of the canonical mesh with region zones, facet patches, interface
   and sheet labels and source associations, the native certified source
   witnesses of every published vertex (`PlcSourceFidelity`), plus the
   declared `PiecewiseLinearDomain` of the source polygons that independent
   domain coverage certifies the mesh against.

No external volume provider is invoked. Acceptance and publication belong to
the native provider route.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from time import monotonic
from typing import final, Literal, NoReturn, TYPE_CHECKING


if TYPE_CHECKING:
    from .providers._native_layer import PreparedLayerCore

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    charge_native_geometry_queries,
    constrained_delaunay_2d,
    current_native_execution_budget,
    current_native_host_workspace,
    exact_orient2d,
    exact_orient3d,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionBudget,
    NativeExecutionEvidence,
    NativeHostStorageWorkspace,
    PLC_3D_COUNTERS,
    plc_source_constraints,
    PlcRecovery3D,
    PlcRecoveryFailure,
    PlcSourceConstraints,
    point_triangle_locations,
    recover_plc_3d,
    TET_MESH_EXUDE_COUNTERS,
    TET_MESH_IMPROVE_COUNTERS,
    TET_MESH_REFINE_COUNTERS,
    TET_MESH_UNMET_CRITERIA,
    TET_MESH_UNMET_REASONS,
    TetMesh3D,
    TetMeshBoundaryPolicy,
    TetMeshQuality,
    TetMeshRun,
    TetMeshSourceComplex,
    TetMeshSourceEvidence,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, unique_identifiers
from ..discretization import CellMesh, TetrahedralConnectivity
from ..discretization._cell_geometry_validity import CellValidityPolicy
from ..geometry._mesh_certificates import PiecewiseLinearDomain
from ..typing import ConvertibleToArray, Dim, HostFloat64, HostInt64, parse, Scope
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
    PlcAssociationTransfer,
)
from ._canonical import canonicalize_cell_mesh
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    VolumeMeshingSpec,
)
from ._controls import FeatureKind
from ._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhase,
    NativeMeshingPhaseMeasurement,
    NativeMeshingPhaseRecorder,
)
from ._organization import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import (
    _select_control,
    SizeCombinationPolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from ._tetra_metric import (
    _physical_size_compliance,
    execute_native_tetra_metric,
    MetricRemeshingEvidence,
    MetricRemeshingStatus,
    NativeTetraMetricOutcome,
    TetraMetricSource,
    UniformSizeRemeshingGoal,
)
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus


class PlcVertexDim(Dim, minimum=4):
    """Vertices of one piecewise-linear complex."""


class PlcPolygonDim(Dim, minimum=1):
    """Oriented polygons of one piecewise-linear complex."""


class PlcPolygonOffsetDim(Dim, minimum=2):
    """Loop offsets of the polygons of one complex (polygons plus one)."""


class PlcLoopDim(Dim, minimum=3):
    """Concatenated polygon loop vertices of one complex."""


class PlcFacetDim(Dim, minimum=1):
    """Facets (region-incidence groups of polygons) of one complex."""


class PlcSegmentDim(Dim):
    """Explicit internal curve segments of one complex."""


class PlcSourceTriangleDim(Dim, minimum=1):
    """Exact original source triangles, including internal sheets."""


@final
class PiecewiseLinearComplex(StrictModule, NonTrainableState):
    """Oriented 3D piecewise-linear complex: the source of a PLC volume mesh.

    ``polygons`` are loops of distinct, exactly coplanar vertex indices;
    ``polygon_facets`` names each polygon's facet. ``facet_regions[f]`` is the
    pair (positive, negative): the index into ``region_ids`` of the region on
    the side of the loops' right-hand normal and on the opposite side; ``-1``
    is void (exterior or a hole) and equal entries declare an internal sheet.
    Entities meet only at shared vertices and edges, and every region's
    oriented facet chain must be closed; the native recovery certifies both
    exactly. ``segments`` are internal curves inside regions. ``boundary``
    ``"fixed"`` declares the supplied boundary immutable (no Steiner point on
    any facet or edge; triangle polygons are kept exactly); ``"conforming"``
    admits exact boundary subdivision with recorded ancestry.
    """

    __strict_contract__ = True

    vertices: HostFloat64[PlcVertexDim, Literal[3]]
    polygon_offsets: HostInt64[PlcPolygonOffsetDim]
    polygon_vertices: HostInt64[PlcLoopDim]
    polygon_facets: HostInt64[PlcPolygonDim]
    facet_regions: HostInt64[PlcFacetDim, Literal[2]]
    segments: HostInt64[PlcSegmentDim, Literal[2]]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    boundary: TetMeshBoundaryPolicy = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ConvertibleToArray,
        polygons: Sequence[ConvertibleToArray],
        polygon_facets: ConvertibleToArray,
        facet_regions: ConvertibleToArray,
        region_ids: tuple[str, ...],
        /,
        *,
        segments: ConvertibleToArray | None = None,
        boundary: TetMeshBoundaryPolicy = "conforming",
    ) -> None:
        if isinstance(polygons, np.ndarray) and polygons.ndim != 2:
            raise TypeError("polygons must be a sequence of index loops.")
        loops = [np.asarray(loop) for loop in polygons]
        if not loops:
            raise ValueError("A piecewise-linear complex needs at least one polygon.")
        for loop in loops:
            if loop.ndim != 1 or not np.issubdtype(loop.dtype, np.integer):
                raise TypeError("Every polygon must be a one-dimensional integer loop.")
            if loop.shape[0] < 3:
                raise ValueError("Every polygon needs at least three vertices.")
        scope = Scope()
        points = parse(
            np.asarray(vertices, dtype=np.float64),
            HostFloat64[PlcVertexDim, Literal[3]],
            "vertices",
            scope=scope,
        )
        execution = current_native_execution_budget()
        if execution is None:
            offsets_raw = np.concatenate(
                ([0], np.cumsum([loop.shape[0] for loop in loops]))
            ).astype(np.int64)
            flat_raw = np.concatenate(loops).astype(np.int64)
        else:
            offsets_raw = execution.allocate_host_array((len(loops) + 1,), np.int64)
            flat_raw = execution.allocate_host_array(
                (sum(loop.shape[0] for loop in loops),),
                np.int64,
            )
            offsets_raw[0] = 0
            offset = 0
            for index, loop in enumerate(loops):
                count = loop.shape[0]
                flat_raw[offset : offset + count] = loop
                offset += count
                offsets_raw[index + 1] = offset
        offsets = parse(
            offsets_raw,
            HostInt64[PlcPolygonOffsetDim],
            "polygon_offsets",
            scope=scope,
        )
        flat = parse(
            flat_raw,
            HostInt64[PlcLoopDim],
            "polygon_vertices",
            scope=scope,
        )
        owners = parse(
            np.asarray(polygon_facets, dtype=np.int64),
            HostInt64[PlcPolygonDim],
            "polygon_facets",
            scope=scope,
        )
        incidence = parse(
            np.asarray(facet_regions, dtype=np.int64),
            HostInt64[PlcFacetDim, Literal[2]],
            "facet_regions",
            scope=scope,
        )
        curves = parse(
            np.zeros((0, 2), dtype=np.int64)
            if segments is None
            else np.asarray(segments, dtype=np.int64),
            HostInt64[PlcSegmentDim, Literal[2]],
            "segments",
            scope=scope,
        )
        ids = unique_identifiers(region_ids, "region_ids")
        policy = parse(boundary, TetMeshBoundaryPolicy, "boundary")
        if owners.shape[0] != len(loops):
            raise ValueError("polygon_facets must name the facet of every polygon.")
        if not np.all(np.isfinite(points)):
            raise ValueError("PLC vertices must be finite.")
        vertex_count = points.shape[0]
        if np.any((flat < 0) | (flat >= vertex_count)) or np.any(
            (curves < 0) | (curves >= vertex_count)
        ):
            raise ValueError("Polygons and segments must index declared vertices.")
        facet_count = incidence.shape[0]
        if np.any((owners < 0) | (owners >= facet_count)):
            raise ValueError("polygon_facets must index declared facets.")
        if np.setdiff1d(np.arange(facet_count), owners).size:
            raise ValueError("Every facet must own at least one polygon.")
        if np.any((incidence < -1) | (incidence >= len(ids))) or np.any(
            np.all(incidence < 0, axis=1)
        ):
            raise ValueError(
                "facet_regions must index declared regions or be -1, and no "
                "facet may have void on both sides."
            )
        if np.setdiff1d(np.arange(len(ids)), incidence).size:
            raise ValueError("Every region must be bounded by at least one facet.")
        self.vertices = points
        self.polygon_offsets = offsets
        self.polygon_vertices = flat
        self.polygon_facets = owners
        self.facet_regions = incidence
        self.segments = curves
        self.region_ids = ids
        self.boundary = policy
        self.complex_id = canonical_fingerprint(
            {
                "kind": "piecewise-linear-complex",
                "vertices": array_tree_fingerprint(points),
                "polygon_offsets": array_tree_fingerprint(offsets),
                "polygon_vertices": array_tree_fingerprint(flat),
                "polygon_facets": array_tree_fingerprint(owners),
                "facet_regions": array_tree_fingerprint(incidence),
                "segments": array_tree_fingerprint(curves),
                "region_ids": ids,
                "boundary": policy,
            }
        )

    @property
    def facet_count(self) -> int:
        return self.facet_regions.shape[0]

    @property
    def region_count(self) -> int:
        return len(self.region_ids)


@final
class PreparedPlcSource(StrictModule, NonTrainableState):
    """Immutable source-only PLC preparation bound to source, revision and frame.

    Native source ordering is retained in the transfer's primitive tables.
    No volume mesh, opaque native handle or geometry proximity cache is kept.
    """

    __strict_contract__ = True
    complex: PiecewiseLinearComplex
    association_transfer: PlcAssociationTransfer
    input_polygons: HostInt64[PlcSourceTriangleDim]
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    preparation_counters: tuple[tuple[str, int], ...] = eqx.field(static=True)
    memory_evidence: tuple[int, ...] | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex_: PiecewiseLinearComplex,
        source_id: str,
        source_revision: str,
        coordinate_contract: SpatialCoordinateContract,
        constraints: PlcSourceConstraints,
        /,
        *,
        maximum_support_queries: int,
        execution_evidence: NativeExecutionEvidence | None = None,
        source_geometry_queries: int = 0,
    ) -> None:
        if not isinstance(complex_, PiecewiseLinearComplex):
            raise TypeError("complex_ must be PiecewiseLinearComplex.")
        if not isinstance(constraints, PlcSourceConstraints):
            raise TypeError("constraints must be real native PlcSourceConstraints.")
        if not np.array_equal(constraints.points, complex_.vertices):
            raise ValueError(
                "Native preparation must retain the original PLC point ordering."
            )
        polygons = parse(
            np.array(constraints.input_polygons, dtype=np.int64, copy=True),
            HostInt64[PlcSourceTriangleDim],
            "input_polygons",
        )
        if np.any(polygons < 0) or np.any(polygons >= complex_.polygon_facets.shape[0]):
            raise ValueError(
                "Native source-triangle polygons reference the wrong source."
            )
        polygons.setflags(write=False)
        transfer = PlcAssociationTransfer(
            _domain(complex_, constraints.input_triangles, polygons, source_id),
            coordinate_contract,
            source_revision,
            edge_vertices=constraints.plc_edges,
            triangle_vertices=constraints.input_triangles,
            triangle_facets=complex_.polygon_facets[polygons],
            facet_regions=complex_.facet_regions,
            vertex_indices=np.arange(complex_.vertices.shape[0], dtype=np.int64),
            edge_indices=np.arange(constraints.plc_edges.shape[0], dtype=np.int64),
            region_indices=np.arange(complex_.region_count, dtype=np.int64),
            maximum_support_queries=maximum_support_queries,
        )
        self.complex = complex_
        self.association_transfer = transfer
        self.input_polygons = polygons
        self.source_id = transfer.domain.source_id
        self.source_revision = transfer.source_revision
        source_counts = {
            "input_triangles",
            "facet_groups",
            "plc_edges",
            "contact_tests",
            "work_units",
            "peak_retained_bytes",
        }
        preparation_counters = tuple(
            (name, int(value))
            for name, value in zip(
                PLC_3D_COUNTERS, constraints.counters.tolist(), strict=True
            )
            if name in source_counts
        )
        if execution_evidence is not None:
            preparation_counters = tuple(
                (name, value)
                for name, value in preparation_counters
                if name != "work_units"
            ) + (
                ("work_units", int(execution_evidence.work_evidence[0])),
                (
                    "native_geometry_primitive_queries",
                    execution_evidence.native_primitive_queries,
                ),
                (
                    "source_geometry_queries",
                    source_geometry_queries
                    + execution_evidence.externally_charged_geometry_queries,
                ),
            )
        self.preparation_counters = preparation_counters
        self.memory_evidence = (
            tuple(int(value) for value in execution_evidence.memory_evidence)
            if execution_evidence is not None
            else (
                None
                if constraints.memory_evidence is None
                else tuple(int(value) for value in constraints.memory_evidence.tolist())
            )
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-plc-source",
                "source": self.source_id,
                "revision": self.source_revision,
                "complex": complex_.complex_id,
                "coordinate_contract": coordinate_contract.spatial_id,
                "association_transfer": transfer.transfer_id,
                "input_polygons": array_tree_fingerprint(polygons),
            }
        )


def prepare_plc_source(
    complex_: PiecewiseLinearComplex,
    source_id: str,
    source_revision: str,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    limits: MeshingLimits,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
    source_geometry_queries: int = 0,
    operation_started: float | None = None,
) -> PreparedPlcSource:
    """Prepare exact source strata without recovering or refining a volume."""
    if not isinstance(limits, MeshingLimits):
        raise TypeError("limits must be MeshingLimits.")
    remaining_work = _native_volume_remaining_work(limits, source_work_units)
    remaining_queries = limits.maximum_geometry_queries - source_geometry_queries
    if remaining_work <= 0 or remaining_queries <= 0:
        _exhausted(
            MeshingStageKind.SOURCE_INSPECTION,
            "Source preparation consumed the source-renewal work or query allowance.",
            None,
        )
    with native_volume_execution_budget(
        limits,
        source_work_units=source_work_units,
        source_geometry_queries=source_geometry_queries,
        operation_started=operation_started,
        stage=MeshingStageKind.SOURCE_INSPECTION,
    ) as budget:
        try:
            constraints = plc_source_constraints(
                complex_.vertices,
                complex_.polygon_offsets,
                complex_.polygon_vertices,
                complex_.polygon_facets,
                complex_.facet_regions,
                segments=complex_.segments,
                work_limit=remaining_work,
                max_scratch_bytes=limits.maximum_scratch_bytes,
                record_native_phase=_native_phase_recorder(record_phase),
            )
        except PlcRecoveryFailure as error:
            raise _recovery_failure(error) from error
    return PreparedPlcSource(
        complex_,
        source_id,
        source_revision,
        coordinate_contract,
        constraints,
        maximum_support_queries=remaining_queries,
        execution_evidence=budget.evidence,
        source_geometry_queries=source_geometry_queries,
    )


def _positive(value: int, name: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@final
class NativeVolumeSchedule(StrictModule, NonTrainableState):
    """Numerical schedule of native constrained tetrahedral generation.

    ``radius_edge_bound`` is the circumradius-to-shortest-edge aim of
    refinement and ``minimum_dihedral_degrees`` the sliver aim of improvement;
    unmet aims are published as evidence, never hidden. ``refinement_rounds``
    and ``improvement_passes`` bound the refinement batches and improvement
    sweeps. ``exudation_passes`` bounds the explicit weighted sliver-exudation
    stage after improvement (``0`` disables it); its weights stay below
    ``exudation_weight_fraction`` times the shortest incident edge squared and
    it is a bounded local stage, not a globally regular triangulation.
    ``metric_optimization_passes`` bounds the shared statistical-goal
    split/collapse/reconnection/relocation epoch; it never changes physical
    controls or their compliance tolerances.
    """

    radius_edge_bound: float = eqx.field(static=True)
    minimum_dihedral_degrees: float = eqx.field(static=True)
    refinement_rounds: int = eqx.field(static=True)
    improvement_passes: int = eqx.field(static=True)
    exudation_passes: int = eqx.field(static=True)
    exudation_weight_fraction: float = eqx.field(static=True)
    metric_optimization_passes: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        radius_edge_bound: float = 2.0,
        minimum_dihedral_degrees: float = 10.0,
        refinement_rounds: int = 8,
        improvement_passes: int = 8,
        exudation_passes: int = 0,
        exudation_weight_fraction: float = 0.1,
        metric_optimization_passes: int = 16,
    ) -> None:
        ratio = float(radius_edge_bound)
        angle = float(minimum_dihedral_degrees)
        if not np.isfinite(ratio) or ratio < 1.0:
            raise ValueError("radius_edge_bound must be finite and at least 1.")
        if not np.isfinite(angle) or not 0.0 <= angle < 70.0:
            raise ValueError("minimum_dihedral_degrees must lie in [0, 70).")
        rounds = _positive(refinement_rounds, "refinement_rounds")
        passes = _positive(improvement_passes, "improvement_passes")
        exudation = nonnegative_integer(exudation_passes, "exudation_passes")
        if exudation > np.iinfo(np.int32).max:
            raise ValueError("exudation_passes must fit signed int32.")
        weight_fraction = float(exudation_weight_fraction)
        if not np.isfinite(weight_fraction) or not 0.0 < weight_fraction <= 0.25:
            raise ValueError("exudation_weight_fraction must lie in (0, 0.25].")
        metric_passes = _positive(
            metric_optimization_passes, "metric_optimization_passes"
        )
        self.radius_edge_bound = ratio
        self.minimum_dihedral_degrees = angle
        self.refinement_rounds = rounds
        self.improvement_passes = passes
        self.exudation_passes = exudation
        self.exudation_weight_fraction = weight_fraction
        self.metric_optimization_passes = metric_passes
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-volume-schedule",
                "radius_edge_bound": ratio,
                "minimum_dihedral_degrees": angle,
                "refinement_rounds": rounds,
                "improvement_passes": passes,
                "exudation_passes": exudation,
                "exudation_weight_fraction": weight_fraction,
                "metric_optimization_passes": metric_passes,
            }
        )


@dataclass(frozen=True, slots=True)
class VolumeConstruction:
    """Assembled constrained tetrahedral mesh and the evidence of its phases.

    ``validity_policy_id`` identifies the cell validity policy whose relative
    determinant floor the native improvement and exudation enforced;
    publication must audit under that same policy. ``exudation`` is the run of
    the explicit weighted stage (``None`` when the schedule disables it) and
    ``exudation_weights`` its accepted power weights indexed by published mesh
    vertices; later-created vertices carry zero. They are exudation evidence,
    not a regular-triangulation certificate of the published mesh.
    """

    mesh: CellMesh
    cell_regions: np.ndarray
    domain: PiecewiseLinearDomain
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    labels: tuple[MeshLabel, ...]
    associations: tuple[GeometryAssociation, ...]
    stages: tuple[MeshingStageReport, ...]
    construction_counters: tuple[tuple[str, int], ...]
    refinement: TetMeshRun
    improvement: TetMeshRun
    quality: TetMeshQuality
    unmet: tuple[tuple[str, str, int], ...]
    steiner_points: int
    work_units: int
    validity_policy_id: str
    metric_optimization: MetricRemeshingEvidence | None = None
    native_execution_evidence: NativeExecutionEvidence | None = None
    source_fidelity: PlcSourceFidelity | None = None
    exudation: TetMeshRun | None = None
    exudation_weights: np.ndarray | None = None
    native_preparation_evidence: NativeExecutionRecord | None = None
    native_execution_record: NativeExecutionRecord | None = None


@final
@dataclass(frozen=True, slots=True)
class PlcSourceFidelity:
    """Declared and achieved source deviation of a PLC volume construction.

    ``facet_tolerances`` (F,) and ``segment_tolerances`` (S,) are the bounds
    declared by protected surface/material-interface (facets) and curve
    (explicit segments) features, the smallest one where several apply and
    zero (exact) elsewhere. Per published vertex, ``witness_strata`` (0 none,
    1 PLC edge, 2 input triangle), ``witness_entities`` (the ``plc_edges`` or
    input-triangle row) and ``witness_deviations`` are the natively certified
    ancestry; zero deviation is exact source membership. ``facet_achieved``
    and ``segment_achieved`` are the largest certified deviation of a
    published vertex on each facet or explicit segment, ``achieved_bound``
    their maximum and ``refused_carriers`` the bounded carriers refinement
    refused by their declared bound.
    """

    facet_tolerances: np.ndarray
    segment_tolerances: np.ndarray
    witness_strata: np.ndarray
    witness_entities: np.ndarray
    witness_parameters: np.ndarray
    witness_deviations: np.ndarray
    facet_achieved: np.ndarray
    segment_achieved: np.ndarray
    achieved_bound: float
    refused_carriers: int
    declared_source: TetMeshSourceComplex | None = None


def _declared_source_tolerances(
    complex_: PiecewiseLinearComplex, specification: VolumeMeshingSpec, /
) -> tuple[np.ndarray, np.ndarray]:
    """Facet and explicit-segment deviation bounds of the protected features."""
    facets = np.full((complex_.facet_count,), np.inf, dtype=np.float64)
    segments = np.full((complex_.segments.shape[0],), np.inf, dtype=np.float64)
    for feature in specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        match feature.scope.entity_dimension, feature.feature_kind:
            case 2, FeatureKind.SURFACE | FeatureKind.MATERIAL_INTERFACE:
                target = facets
            case 1, FeatureKind.CURVE:
                target = segments
            case _:
                continue
        if np.any(identifiers < 0) or np.any(identifiers >= target.shape[0]):
            raise ValueError(
                "Protected feature scopes must select PLC facets or segments."
            )
        target[identifiers] = np.minimum(target[identifiers], feature.maximum_deviation)
    facets[~np.isfinite(facets)] = 0.0
    segments[~np.isfinite(segments)] = 0.0
    return facets, segments


def _edge_facets(
    plc_edges: np.ndarray, input_triangles: np.ndarray, triangle_facets: np.ndarray, /
) -> list[np.ndarray]:
    """Facets of the input triangles bounded by each PLC edge."""
    keys = {tuple(sorted(edge)): row for row, edge in enumerate(plc_edges.tolist())}
    incident: list[list[int]] = [[] for _ in range(plc_edges.shape[0])]
    for triangle, facet in zip(
        input_triangles.tolist(), triangle_facets.tolist(), strict=True
    ):
        for k in range(3):
            row = keys.get(tuple(sorted((triangle[k], triangle[(k + 1) % 3]))))
            if row is not None:
                incident[row].append(facet)
    return [np.unique(np.asarray(values, dtype=np.int64)) for values in incident]


def _plc_source_rows(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    plc_edges: np.ndarray,
    diagonals: np.ndarray,
    /,
) -> tuple[np.ndarray, ...]:
    """Original row banks and scientific bounds shared by publication and restore."""
    facet_bounds, segment_bounds = _declared_source_tolerances(complex_, specification)
    triangle_facets = complex_.polygon_facets[input_polygons]
    incident = _edge_facets(plc_edges, input_triangles, triangle_facets)
    edge_bounds = np.empty((plc_edges.shape[0],), dtype=np.float64)
    for row, facets in enumerate(incident):
        bound = float(segment_bounds[row]) if row < segment_bounds.shape[0] else np.inf
        if facets.size:
            bound = min(bound, float(np.min(facet_bounds[facets])))
        edge_bounds[row] = bound if np.isfinite(bound) else 0.0
    return (
        np.asarray(complex_.vertices, dtype=np.float64),
        np.asarray(input_triangles, dtype=np.int32),
        np.asarray(triangle_facets, dtype=np.int64),
        facet_bounds[triangle_facets],
        np.concatenate((plc_edges, diagonals)).astype(np.int32),
        np.arange(plc_edges.shape[0] + diagonals.shape[0], dtype=np.int64),
        np.concatenate((edge_bounds, np.zeros((diagonals.shape[0],), dtype=np.float64))),
    )


def _plc_source_complex(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    recovery: PlcRecovery3D,
    diagonals: np.ndarray,
    /,
) -> TetMeshSourceComplex:
    """The PLC input triangles and edges as the carrier's declared source rows.

    A PLC edge row takes the smallest bound of its explicit segment and its
    incident facets; fixed-boundary diagonals are exact rows.
    """
    points, faces, triangle_facets, face_bounds, segments, segment_ids, edge_bounds = (
        _plc_source_rows(
            complex_,
            specification,
            recovery.input_triangles,
            recovery.input_polygons,
            recovery.plc_edges,
            diagonals,
        )
    )
    face_ids, face_groups = np.unique(triangle_facets, return_inverse=True)
    return TetMeshSourceComplex(
        points,
        faces,
        face_groups.astype(np.int32),
        face_bounds,
        segments,
        np.arange(segments.shape[0], dtype=np.int32),
        edge_bounds,
        recovery.witness_strata,
        recovery.witness_entities,
        recovery.witness_parameters,
        face_ids,
        segment_ids,
    )


def _entity(revision: str, kind: str, index: int, /) -> str:
    # Matches the native source entity convention of the provider bindings.
    return f"{revision}:{kind}:{index}"


def _seeds(
    complex_: PiecewiseLinearComplex, specification: VolumeMeshingSpec, /
) -> tuple[np.ndarray, np.ndarray]:
    """Seed points and their region indices (-1 for holes)."""

    points: list[np.ndarray] = []
    regions: list[int] = []
    for seed in specification.region_seeds:
        if seed.region_name not in complex_.region_ids:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                f"Region seed {seed.seed_id} names no region of the complex.",
                stage=MeshingStageKind.CONTROL_RESOLUTION.value,
            )
        points.append(np.asarray(seed.point, dtype=np.float64))
        regions.append(complex_.region_ids.index(seed.region_name))
    for hole in specification.hole_seeds:
        points.append(np.asarray(hole.point, dtype=np.float64))
        regions.append(-1)
    if not points:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    return np.stack(points).astype(np.float64), np.asarray(regions, dtype=np.int32)


def _recovery_failure(error: PlcRecoveryFailure, /) -> MeshingFailure:
    if error.status in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT):
        category = MeshingFailureCategory.RESOURCE_EXHAUSTED
    else:
        match error.reason:
            case (
                "vertex_budget"
                | "tetrahedron_budget"
                | "work_budget"
                | "scratch_byte_budget"
            ):
                category = MeshingFailureCategory.RESOURCE_EXHAUSTED
            case "invalid_seed" | "inconsistent_regions":
                category = MeshingFailureCategory.REGION_RESOLUTION_FAILED
            case "fixed_segment" | "fixed_facet" | "nonrepresentable_steiner":
                category = MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
            case "source_deviation":
                # The nearest ancestry-backed carrier exceeds the declared bound.
                category = MeshingFailureCategory.COMPLIANCE_FAILED
            case _:
                category = MeshingFailureCategory.INVALID_SOURCE
    described = ", ".join(f"{kind} {index}" for kind, index in error.entities)
    deviation, bound = error.source_refusal
    return MeshingFailure(
        category,
        f"PLC boundary recovery refused: {error.reason} ({described}).",
        provider_code=error.reason,
        stage=MeshingStageKind.VOLUME_FILL.value,
        entity_ids=tuple(index for _, index in error.entities),
        requested=(("source_deviation_bound", bound),)
        if error.reason == "source_deviation"
        else (),
        achieved=tuple(
            (name, int(value))
            for name, value in zip(PLC_3D_COUNTERS, error.counters.tolist(), strict=True)
        )
        + (
            (("certified_source_deviation", deviation),)
            if error.reason == "source_deviation"
            else ()
        ),
    )


_VOLUME_OPERATION_STARTED: ContextVar[float | None] = ContextVar(
    "native_volume_operation_started",
    default=None,
)


def _native_volume_remaining_work(limits: MeshingLimits, spent: int, /) -> int:
    """Forward the actual active allowance, not a renewed phase-local estimate."""
    execution = current_native_execution_budget()
    if execution is None:
        return limits.maximum_work_units - spent
    execution.charge()
    return execution.remaining().remaining_work_units


def _native_volume_operation_started(
    operation_started: float | None = None,
    /,
) -> float | None:
    """Borrow the actual original host clock; raw native scopes own their deadline."""
    started = _VOLUME_OPERATION_STARTED.get()
    return operation_started if started is None else started


def _native_volume_import_preparation(
    budget: NativeExecutionBudget,
    /,
    *,
    work: int,
    queries: int,
    seconds: float,
) -> None:
    """Import genuine prior duration into native deadlines and the original host clock."""
    budget.import_preparation(
        work=work, geometry_queries=queries, elapsed_seconds=seconds
    )
    started = _VOLUME_OPERATION_STARTED.get()
    if started is not None:
        _VOLUME_OPERATION_STARTED.set(started - seconds)


def _import_native_preparation(
    budget: NativeExecutionBudget,
    storage: NativeHostStorageWorkspace,
    receipt: NativeExecutionRecord,
    /,
) -> None:
    """Import each ended predecessor phase once, never replay a chain total."""
    receipt.require_valid()
    current: NativeExecutionRecord | None = receipt
    while current is not None:
        _remember_native_preparation(budget, storage, current, import_work=True)
        current = current.preparation_evidence


def _remember_native_preparation(
    budget: NativeExecutionBudget,
    storage: NativeHostStorageWorkspace,
    receipt: NativeExecutionRecord,
    /,
    *,
    import_work: bool,
) -> None:
    """Retain an actual phase identity in its bounded, single-use parent root."""
    ancestor: NativeExecutionBudget | None = budget
    while ancestor is not None:
        if id(receipt) in ancestor._imported_preparation_receipts:
            return
        ancestor = ancestor._parent
    receipts = budget._imported_preparation_receipts
    storage.retain_owner(receipt)
    storage.set_bound(storage.bound + 256)
    work = int(np.asarray(receipt.work[0])) if import_work else 0
    queries = int(np.asarray(receipt.work[1])) if import_work else 0
    if import_work and receipt.preparation_evidence is None:
        work += int(np.asarray(receipt.source_preparation_work_units))
        queries += int(np.asarray(receipt.source_preparation_geometry_queries))
    if import_work:
        seconds = float(np.asarray(receipt.elapsed_seconds)) + float(
            np.asarray(receipt.prior_elapsed_seconds)
        )
        if receipt.preparation_evidence is None:
            seconds += float(np.asarray(receipt.preparation_seconds))
        _native_volume_import_preparation(
            budget, work=work, queries=queries, seconds=seconds
        )
    budget.charge(work=1)
    receipts[id(receipt)] = receipt


def _native_live_preparation_is_active(prepared: PreparedLayerCore, /) -> bool:
    """Authorize only the actual successfully prepared owner in a live ancestry."""
    ancestor = current_native_execution_budget()
    while ancestor is not None:
        if ancestor._live_preparations.get(id(prepared)) is prepared:
            return True
        ancestor = ancestor._parent
    return False


def _remember_native_live_preparation(
    budget: NativeExecutionBudget,
    storage: NativeHostStorageWorkspace,
    prepared: PreparedLayerCore,
    /,
) -> None:
    """Retain a genuine completed preparation without replaying its work."""
    from .providers._native_layer import PreparedLayerCore

    if not isinstance(prepared, PreparedLayerCore):
        raise TypeError("Live preparation requires its canonical prepared native owner.")
    if budget is not current_native_execution_budget():
        raise RuntimeError(
            "Live preparation registration requires its actual active scope."
        )
    owner = budget
    while owner._parent is not None:
        owner = owner._parent
    if owner._live_preparations.get(id(prepared)) is prepared:
        return
    storage.retain_owner(prepared)
    storage.set_bound(storage.bound + 256)
    budget.charge(work=1)
    owner._live_preparations[id(prepared)] = prepared


def native_volume_checkpoint(
    limits: MeshingLimits,
    stage: MeshingStageKind,
    /,
    *,
    operation_started: float | None = None,
    execution_budget: NativeExecutionBudget | None = None,
) -> None:
    """Check the original clock and active native allowance before a host phase."""
    from .providers._native_publication import check_deadline

    active = current_native_execution_budget()
    if execution_budget is not None and execution_budget is not active:
        raise RuntimeError(
            "A volume phase must borrow its actual active execution budget."
        )
    started = _native_volume_operation_started(operation_started)
    if started is not None:
        check_deadline(started, limits, stage)
    if active is not None:
        active.charge()


@contextmanager
def native_volume_execution_budget(
    limits: MeshingLimits,
    /,
    *,
    source_work_units: int = 0,
    source_geometry_queries: int = 0,
    operation_started: float | None = None,
    mesh: TetMesh3D | None = None,
    stage: MeshingStageKind = MeshingStageKind.VOLUME_FILL,
    borrow_active: bool = True,
) -> Iterator[NativeExecutionBudget]:
    """Own one original cumulative allowance across all native producer phases.

    An active scope is borrowed by default. An imported ended preparation
    receipt may require a real child scope; only that child deducts its receipt.
    primitive work and queries include construction and initialization;
    separately owned host/device batches charge before dispatch, never repeat
    native mesh-counter deltas.
    """
    active = current_native_execution_budget()
    if active is not None and borrow_active:
        native_volume_checkpoint(
            limits, stage, operation_started=operation_started, execution_budget=active
        )
        yield active
        native_volume_checkpoint(
            limits, stage, operation_started=operation_started, execution_budget=active
        )
        return
    source_work = nonnegative_integer(source_work_units, "source_work_units")
    source_queries = nonnegative_integer(
        source_geometry_queries, "source_geometry_queries"
    )
    remaining_work = limits.maximum_work_units - source_work
    remaining_queries = limits.maximum_geometry_queries - source_queries
    started = monotonic() if operation_started is None else operation_started
    if remaining_work < 0 or remaining_queries < 0:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Source preparation exceeded the original native operation allowance.",
            stage=stage.value,
            requested=(
                ("maximum_work_units", limits.maximum_work_units),
                ("maximum_geometry_queries", limits.maximum_geometry_queries),
            ),
            achieved=(
                ("source_work_units", source_work),
                ("source_geometry_queries", source_queries),
            ),
        )
    budget = NativeExecutionBudget(
        max_work=remaining_work,
        max_geometry_queries=remaining_queries,
        max_cavity_cells=limits.maximum_cavity_cells,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        max_wall_seconds=max(0.0, limits.maximum_wall_seconds - (monotonic() - started)),
        mesh=mesh,
    )
    try:
        with budget:
            token = _VOLUME_OPERATION_STARTED.set(started)
            try:
                workspace = current_native_host_workspace()
                with (
                    budget.host_workspace()
                    if workspace is None
                    else nullcontext(workspace)
                ):
                    native_volume_checkpoint(limits, stage, execution_budget=budget)
                    yield budget
                    native_volume_checkpoint(limits, stage, execution_budget=budget)
            finally:
                _VOLUME_OPERATION_STARTED.reset(token)
    except (MeshcoreError, MeshingFailure) as error:
        evidence = budget.evidence
        status = (
            evidence.status
            if evidence is not None
            else (error.status if isinstance(error, MeshcoreError) else None)
        )
        phase_resource = isinstance(error, MeshingFailure) and (
            error.category
            in (
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                MeshingFailureCategory.TIMED_OUT,
            )
        )
        native_resource = status in (
            MeshcoreStatus.CAPACITY_EXCEEDED,
            MeshcoreStatus.TIMEOUT,
        )
        if not phase_resource and not native_resource:
            raise
        native_provider_code = (
            status.name.lower() if status is not None else "resource_exhausted"
        )
        if status is MeshcoreStatus.CAPACITY_EXCEEDED and evidence is not None:
            total_work = source_work + int(evidence.work_evidence[0])
            peak_bytes = (
                int(evidence.memory_evidence[2]) + evidence.host_storage_peak_bytes_upper
            )
            if total_work >= limits.maximum_work_units:
                native_provider_code = "work_budget"
            elif peak_bytes >= limits.maximum_scratch_bytes:
                native_provider_code = "scratch_byte_budget"
        provider_code = (
            native_provider_code
            if native_resource
            else (
                error.evidence.provider_code
                if isinstance(error, MeshingFailure)
                else "resource_exhausted"
            )
        )
        actual = (
            ()
            if evidence is None
            else (
                ("work_units", source_work + int(evidence.work_evidence[0])),
                ("geometry_queries", source_queries + int(evidence.work_evidence[1])),
                ("native_geometry_primitive_queries", evidence.native_primitive_queries),
                ("host_device_work_units", evidence.externally_charged_work),
                ("native_peak_cavity_cells", int(evidence.work_evidence[2])),
                *(
                    (f"native_execution:{index}", int(value))
                    for index, value in enumerate(evidence.work_evidence)
                ),
                *(
                    (f"native_memory:{index}", int(value))
                    for index, value in enumerate(evidence.memory_evidence)
                ),
            )
        )
        if isinstance(error, MeshingFailure):
            failure = error.evidence
            achieved = dict(failure.achieved)
            achieved.update((f"native_scope:{name}", value) for name, value in actual)
            achieved["native_scope:wall_seconds"] = monotonic() - started
            wrapped = MeshingFailure(
                failure.category,
                failure.message,
                stage=failure.stage,
                provider_code=failure.provider_code,
                entity_ids=failure.entity_ids,
                locations=failure.locations,
                requested=failure.requested,
                achieved=tuple(sorted(achieved.items())),
                checkpoint_id=failure.checkpoint_id,
                logical_findings=failure.logical_findings,
            )
            if error.cut_failure_prefix is not None:
                wrapped.cut_failure_prefix = error.cut_failure_prefix
            raise wrapped from error
        wrapped = MeshingFailure(
            MeshingFailureCategory.TIMED_OUT
            if status == MeshcoreStatus.TIMEOUT
            else MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The original native volume execution allowance was exhausted.",
            stage=stage.value,
            provider_code=provider_code,
            requested=(
                ("maximum_work_units", limits.maximum_work_units),
                ("maximum_geometry_queries", limits.maximum_geometry_queries),
                ("maximum_cavity_cells", limits.maximum_cavity_cells),
                ("maximum_scratch_bytes", limits.maximum_scratch_bytes),
                ("maximum_wall_seconds", limits.maximum_wall_seconds),
            ),
            achieved=(
                ("source_work_units", source_work),
                (
                    "source_geometry_queries",
                    source_queries
                    + (0 if evidence is None else int(evidence.work_evidence[1])),
                ),
                ("wall_seconds", monotonic() - started),
                *actual,
            ),
        )
        if error.cut_failure_prefix is not None:
            wrapped.cut_failure_prefix = error.cut_failure_prefix
        raise wrapped from error


def _require_native_preparation_allowance(
    limits: MeshingLimits,
    receipt: NativeExecutionRecord,
    /,
) -> None:
    """Admit actual prior phases; native payload and host upper bounds stay distinct."""
    receipt.require_valid()
    peak_bytes_upper = 0
    peak_cavity = 0
    current: NativeExecutionRecord | None = receipt
    while current is not None:
        peak_bytes_upper = max(
            peak_bytes_upper,
            int(np.asarray(current.memory)[2])
            + int(np.asarray(current.host_storage_peak_bytes_upper)),
        )
        peak_cavity = max(peak_cavity, int(np.asarray(current.work)[2]))
        current = current.preparation_evidence
    checks = (
        (
            "work_units",
            int(np.asarray(receipt.total_work_units)),
            limits.maximum_work_units,
        ),
        (
            "geometry_queries",
            int(np.asarray(receipt.total_geometry_queries)),
            limits.maximum_geometry_queries,
        ),
        ("cavity_cells", peak_cavity, limits.maximum_cavity_cells),
        ("scratch_bytes", peak_bytes_upper, limits.maximum_scratch_bytes),
        (
            "wall_seconds",
            float(np.asarray(receipt.total_elapsed_seconds)),
            limits.maximum_wall_seconds,
        ),
    )
    exceeded = tuple(name for name, actual, maximum in checks if actual > maximum)
    if exceeded:
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT
            if exceeded == ("wall_seconds",)
            else MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Actual source preparation exceeds the original volume allowance: "
            + ", ".join(exceeded)
            + ".",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
            requested=tuple(("maximum_" + name, maximum) for name, _, maximum in checks),
            achieved=tuple(
                (
                    "source_scratch_bytes_upper"
                    if name == "scratch_bytes"
                    else "source_" + name,
                    actual,
                )
                for name, actual, _ in checks
            ),
        )


def _native_volume_source_is_active(
    budget: NativeExecutionBudget,
    source_owner_id: str,
    /,
) -> bool:
    """Resolve an actual source identity through live native parent ownership."""
    current: NativeExecutionBudget | None = budget
    while current is not None:
        if source_owner_id in current._prepared_envelope_sources:
            return True
        if any(
            receipt.owner_id == source_owner_id
            for receipt in current._imported_preparation_receipts.values()
        ):
            return True
        current = current._parent
    return False


@contextmanager
def native_volume_source_execution_budget(
    limits: MeshingLimits,
    receipt: NativeExecutionRecord | None,
    source_owner_id: str,
    /,
) -> Iterator[NativeExecutionBudget]:
    """Borrow a live source owner or import its authentic ended predecessor once."""
    active = current_native_execution_budget()
    if receipt is None:
        if active is None or not _native_volume_source_is_active(active, source_owner_id):
            raise ValueError(
                "A prepared source requires its actual ended receipt or original active owner."
            )
        with native_volume_execution_budget(limits) as budget:
            yield budget
        return
    if receipt.owner_id != source_owner_id:
        raise ValueError(
            "Source preparation receipt binds another original source owner."
        )
    _require_native_preparation_allowance(limits, receipt)
    workspace = current_native_host_workspace()
    # Algebra reservations resize to their own scratch ledger. They cannot
    # also own persistent source trees whose bytes must remain charged.
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    coordinate = _COORDINATE_BUDGET.get()
    if coordinate is not None and workspace is coordinate._host_workspace:
        workspace = None
    scope = (
        active.host_workspace()
        if active is not None and workspace is None
        else nullcontext(workspace)
    )
    with scope as storage:
        if active is not None:
            if storage is None:
                raise RuntimeError(
                    "Source receipt import lost its owning host workspace."
                )
            _import_native_preparation(active, storage, receipt)
        seconds = float(np.asarray(receipt.total_elapsed_seconds))
        started = monotonic() - seconds
        parent_started = _native_volume_operation_started()
        if parent_started is not None:
            started = min(started, parent_started)
        with native_volume_execution_budget(
            limits,
            source_work_units=int(np.asarray(receipt.total_work_units)),
            source_geometry_queries=int(np.asarray(receipt.total_geometry_queries)),
            operation_started=started,
            borrow_active=False,
        ) as budget:
            persistent_storage = (
                storage if storage is not None else current_native_host_workspace()
            )
            if persistent_storage is None:
                raise RuntimeError(
                    "Source execution lost its actual persistent host workspace."
                )
            phase: NativeExecutionRecord | None = receipt
            while phase is not None:
                _remember_native_preparation(
                    budget,
                    persistent_storage,
                    phase,
                    import_work=False,
                )
                phase = phase.preparation_evidence
            yield budget


def _bind_volume_execution(
    construction: VolumeConstruction,
    budget: NativeExecutionBudget,
    source_work_units: int,
    source_geometry_queries: int,
    /,
    *,
    preparation_evidence: NativeExecutionRecord | None = None,
) -> VolumeConstruction:
    """Publish root measurements once; phase-local counters remain diagnostics."""
    evidence = budget.evidence
    if evidence is None:
        raise RuntimeError("Native construction has no completed execution evidence.")
    if preparation_evidence is not None:
        preparation_evidence.require_valid()
        if source_work_units != int(
            np.asarray(preparation_evidence.total_work_units)
        ) or source_geometry_queries != int(
            np.asarray(preparation_evidence.total_geometry_queries)
        ):
            raise ValueError(
                "Construction source charges differ from their actual ended preparation receipt."
            )
    counters = construction.construction_counters + (
        ("native_execution_work_units", int(evidence.work_evidence[0])),
        ("native_execution_geometry_queries", int(evidence.work_evidence[1])),
        ("native_geometry_primitive_queries", evidence.native_primitive_queries),
        ("host_device_work_units", evidence.externally_charged_work),
        (
            "source_geometry_queries",
            source_geometry_queries + evidence.externally_charged_geometry_queries,
        ),
        ("native_peak_cavity_cells", int(evidence.work_evidence[2])),
        *(
            (f"native_memory:{index}", int(value))
            for index, value in enumerate(evidence.memory_evidence)
        ),
    )
    return replace(
        construction,
        work_units=source_work_units + int(evidence.work_evidence[0]),
        construction_counters=counters,
        native_execution_evidence=evidence,
        native_preparation_evidence=preparation_evidence,
        native_execution_record=NativeExecutionRecord(
            evidence,
            source_preparation_work_units=source_work_units,
            source_preparation_geometry_queries=source_geometry_queries,
            preparation_evidence=preparation_evidence,
        ),
    )


def _native_phase_recorder(
    record_phase: NativeMeshingPhaseRecorder | None, /
) -> Callable[[str, float, int | None, int], None] | None:
    if record_phase is None:
        return None
    phases: dict[str, NativeMeshingPhase] = {
        "validation": "native_validation",
        "preparation": "native_preparation",
        "boundary_recovery": "boundary_recovery",
        "classification": "region_classification",
        "publication": "native_publication",
        "refinement": "refinement",
        "improvement": "improvement",
        "exudation": "exudation",
    }

    def record(phase: str, seconds: float, work: int | None, invocations: int) -> None:
        record_phase(
            NativeMeshingPhaseMeasurement(phases[phase], seconds, work, invocations)
        )

    return record


def _recover(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
) -> PlcRecovery3D:
    limits = specification.limits
    remaining_work = _native_volume_remaining_work(limits, source_work_units)
    if remaining_work <= 0:
        _exhausted(
            MeshingStageKind.SOURCE_INSPECTION,
            "Source preparation consumed the native construction work allowance.",
            None,
        )
    seeds, seed_regions = _seeds(complex_, specification)
    offsets = complex_.polygon_offsets
    facet_bounds, segment_bounds = _declared_source_tolerances(complex_, specification)
    try:
        return recover_plc_3d(
            complex_.vertices,
            offsets,
            complex_.polygon_vertices,
            complex_.polygon_facets,
            complex_.facet_regions,
            segments=complex_.segments,
            seeds=seeds,
            seed_regions=seed_regions,
            boundary_policy=complex_.boundary,
            facet_tolerances=facet_bounds,
            segment_tolerances=segment_bounds,
            max_vertices=limits.maximum_vertices,
            max_tetrahedra=limits.maximum_cells,
            work_limit=remaining_work,
            max_scratch_bytes=limits.maximum_scratch_bytes,
            record_native_phase=_native_phase_recorder(record_phase),
        )
    except PlcRecoveryFailure as error:
        # The native refusal is translated once into the meshing failure
        # contract with its reason, entities and counters.
        raise _recovery_failure(error) from error


def _satisfied_boundary_size_targets(
    specification: VolumeMeshingSpec,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    /,
) -> tuple[bool, ...]:
    """Freeze numerical target exemptions from the actual initial carrier.

    Only whole-boundary hard controls that pass every original size obligation
    qualify. This does not modify the specification or final acceptance.
    """
    from .providers._native_publication import (
        edge_size_evidence,
        uniform_size_compliance,
        unique_edges,
    )

    eligible = tuple(
        isinstance(control, UniformSizeControl)
        and control.strength is SizeControlStrength.HARD
        and np.array_equal(
            control.scope.entity_ids, specification.boundary_scope.entity_ids
        )
        for control in specification.size_controls
    )
    if not any(eligible):
        return eligible
    lengths, growth = edge_size_evidence(points, unique_edges(tetrahedra, "tetrahedron"))
    return tuple(
        admitted
        and isinstance(control, UniformSizeControl)
        and not uniform_size_compliance(
            control, specification.size_compliance, lengths, growth
        )[2]
        for control, admitted in zip(specification.size_controls, eligible, strict=True)
    )


def _uniform_background_goal(
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    validity_policy: CellValidityPolicy,
    /,
) -> UniformSizeRemeshingGoal | None:
    """The closed admitted profile for in-place statistical size optimization.

    Its proposals keep the publishing validity policy's determinant floor.
    """
    if len(specification.size_controls) != 1:
        return None
    control = specification.size_controls[0]
    if (
        not isinstance(control, UniformSizeControl)
        or control.strength is not SizeControlStrength.HARD
        or not np.array_equal(
            control.scope.entity_ids, specification.boundary_scope.entity_ids
        )
    ):
        return None
    return UniformSizeRemeshingGoal(
        control,
        specification.size_compliance,
        validity_policy.relative_determinant_floor,
        maximum_radius_edge=schedule.radius_edge_bound,
        minimum_dihedral_degrees=schedule.minimum_dihedral_degrees,
    )


@contextmanager
def _plc_metric_coordinate_scope(
    bounded: bool,
    work: int,
    memory: int,
    execution: NativeExecutionBudget | None,
    /,
) -> Iterator[None]:
    """Keep exact host source preparation inside the original native allowance."""
    if not bounded:
        yield
        return
    from ..discretization._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    if execution is not None:
        allowance = execution.remaining()
        work, memory = (
            min(work, allowance.remaining_work_units),
            min(memory, allowance.remaining_scratch_bytes),
        )
    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        ledger = CoordinateEnclosureBudget(work, memory)
    try:
        with ledger.activate(), ledger.bound_stage(work, memory):
            yield
            ledger.charge_native_work(
                ledger.work_units - ledger.native_charged_work_units
            )
    except CoordinateEnclosureResourceError as error:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The bounded PLC metric source exhausted its original exact-action allowance.",
            stage=MeshingStageKind.OPTIMIZATION.value,
            requested=(
                ("remaining_work_units", work),
                ("remaining_scratch_bytes", memory),
            ),
            achieved=(
                ("coordinate_work_units", ledger.work_units),
                ("coordinate_peak_bytes_upper", ledger.peak_bytes_upper),
            ),
        ) from error


def _optimize_uniform_background(
    state: TetMesh3D,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    goal: UniformSizeRemeshingGoal,
    original_vertex_count: int,
    spent: int,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    execution_budget: NativeExecutionBudget | None = None,
    operation_started: float | None = None,
    declared_source: TetMeshSourceComplex | None,
    source_id: str,
    source_revision: str,
) -> NativeTetraMetricOutcome:
    """Schedule the existing native metric epoch without reconstructing sources."""
    snapshot = state.arrays()
    control, policy = goal.control, goal.policy
    target = control.target_size
    lower = max(0.0, target - policy.tolerance(target))
    upper = target + policy.tolerance(target)
    if control.minimum_size is not None:
        lower = max(lower, control.minimum_size - policy.tolerance(control.minimum_size))
    if control.maximum_size is not None:
        upper = min(upper, control.maximum_size + policy.tolerance(control.maximum_size))
    if lower > upper:
        requested, achieved, issues = _physical_size_compliance(snapshot, goal)
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "The declared uniform target and hard edge bounds have no common admissible interval: "
            + "; ".join(issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            requested=tuple(requested),
            achieved=tuple(achieved),
        )
    limits = specification.limits
    remaining = _native_volume_remaining_work(limits, spent)
    source_evidence = state.source_evidence()
    with _plc_metric_coordinate_scope(
        source_evidence.achieved_bound > 0.0,
        remaining,
        limits.maximum_scratch_bytes,
        execution_budget,
    ):
        source_geometry = None
        if source_evidence.achieved_bound > 0.0:
            from ..discretization._cell_geometry import CellGeometrySpec
            from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource

            if declared_source is None:
                raise RuntimeError(
                    "A bounded native PLC epoch lost its original declared source bank."
                )
            exact = ExactPlcCellGeometrySource(
                declared_source.points,
                declared_source.faces,
                declared_source.segments,
                source_evidence.witness_strata,
                source_evidence.witness_entities,
                source_evidence.witness_parameters,
                domain_source_id=source_id,
                domain_source_revision=source_revision,
                source_triangle_ids=declared_source.face_ids[
                    declared_source.face_group_rows
                ],
                source_triangle_bounds=declared_source.face_tolerances,
                source_segment_ids=declared_source.segment_ids[
                    declared_source.segment_group_rows
                ],
                source_segment_bounds=declared_source.segment_tolerances,
                maximum_work=limits.maximum_work_units,
            )
            source_mesh = CellMesh.from_tetrahedra(snapshot.points, snapshot.tetrahedra)
            source_geometry = CellGeometrySpec.plc(source_mesh, exact)
        source = TetraMetricSource(
            snapshot.points,
            snapshot.tetrahedra,
            np.arange(snapshot.points.shape[0], dtype=np.int64),
            np.arange(snapshot.tetrahedra.shape[0], dtype=np.int64),
            source_geometry=source_geometry,
        )
        inverse = 1.0 / target
        metric = np.broadcast_to(
            np.eye(3, dtype=np.float64) * (inverse * inverse),
            (snapshot.points.shape[0], 3, 3),
        )
        fixed = np.zeros((snapshot.points.shape[0],), dtype=np.bool_)
        fixed[:original_vertex_count] = True
        return execute_native_tetra_metric(
            state,
            source,
            metric,
            fixed_vertices=fixed,
            protected_edges=set(),
            maximum_passes=schedule.metric_optimization_passes,
            topology_operations=True,
            relocation=True,
            minimum_metric_quality=0.05,
            lower_metric_length=lower / target,
            upper_metric_length=upper / target,
            maximum_vertices=limits.maximum_vertices,
            maximum_cells=limits.maximum_cells,
            maximum_operations=remaining,
            maximum_work_units=remaining,
            maximum_location_pairs=limits.maximum_geometry_queries // 4,
            maximum_cavity_cells=limits.maximum_cavity_cells,
            maximum_cavity_work=remaining,
            record_phase=record_phase,
            size_goal=goal,
            maximum_optimizer_scratch_bytes=limits.maximum_scratch_bytes,
            execution_budget=execution_budget,
            operation_started=operation_started,
            maximum_wall_seconds=None
            if operation_started is None
            else limits.maximum_wall_seconds,
        )


def _field_sizes(
    points: np.ndarray,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    /,
    *,
    deferred_boundary_targets: tuple[bool, ...] = (),
) -> np.ndarray:
    """Resolve native radius sizing on exact source-facet supports.

    A complete-boundary control supplies the volume-wide background field.
    Local controls apply only on their represented facets. A hard background
    already satisfied or deferred to the owning physical-statistic epoch does
    not also impose an extra circumradius preference; its explicit maximum
    remains binding. Original conflicts and winning priorities stay authoritative.
    """
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if len(controls) != len(specification.size_controls):
        raise TypeError("The admitted PLC field must contain uniform size controls.")
    if not controls:
        return np.zeros((points.shape[0],), dtype=np.float64)
    candidates = np.asarray(
        [
            np.clip(
                control.target_size,
                control.minimum_size if control.minimum_size is not None else 0.0,
                control.maximum_size if control.maximum_size is not None else np.inf,
            )
            for control in controls
        ],
        dtype=np.float64,
    )
    target_active = np.asarray(
        [not value for value in deferred_boundary_targets]
        if deferred_boundary_targets
        else [True] * len(controls),
        dtype=np.bool_,
    )
    if target_active.shape != (len(controls),):
        raise ValueError(
            "Size scheduling exemptions must align with the original controls."
        )
    background_disabled = not np.all(target_active)
    values = np.full(
        (points.shape[0],),
        0.0 if background_disabled else np.max(candidates),
        dtype=np.float64,
    )
    masks = np.zeros((len(controls), points.shape[0]), dtype=np.bool_)
    facet_sources = complex_.polygon_facets[input_polygons]
    for index, control in enumerate(controls):
        if np.array_equal(
            control.scope.entity_ids, specification.boundary_scope.entity_ids
        ):
            masks[index] = True
            continue
        triangles = complex_.vertices[
            input_triangles[np.isin(facet_sources, control.scope.entity_ids)]
        ]
        batch = max(1, 4096 // triangles.shape[0])
        for start in range(0, points.shape[0], batch):
            queries = points[start : start + batch]
            count = queries.shape[0]
            charge_native_geometry_queries(count)
            sides, features, status = point_triangle_locations(
                np.repeat(queries, triangles.shape[0], axis=0),
                np.tile(triangles, (count, 1, 1)),
            )
            masks[index, start : start + count] = np.any(
                ((status == MeshcoreStatus.OK) & (sides == 0) & (features >= 0)).reshape(
                    count, triangles.shape[0]
                ),
                axis=1,
            )
    hard = np.asarray(
        [control.strength is SizeControlStrength.HARD for control in controls],
        dtype=np.bool_,
    )
    priorities = np.asarray([control.priority for control in controls], dtype=np.int64)
    target_radius_factor = np.sqrt(6.0) / 4.0
    for point in range(points.shape[0]):
        active = masks[:, point]
        if not np.any(active):
            continue
        active_hard = [
            control
            for control, selected in zip(controls, active & hard, strict=True)
            if selected
        ]
        lower = max((control.minimum_size or 0.0 for control in active_hard), default=0.0)
        upper = min(
            (
                control.maximum_size if control.maximum_size is not None else np.inf
                for control in active_hard
            ),
            default=np.inf,
        )
        if lower > upper:
            raise ValueError("Overlapping hard PLC size intervals conflict.")
        # Keep the original controls authoritative for hard conflicts even when
        # an already-satisfied target no longer drives numerical insertion.
        winner = _select_control(
            candidates,
            active,
            hard,
            priorities,
            (lower, upper),
            specification.size_combination,
        )
        # Exempting a target cannot promote a soft or lower-priority control
        # that lost the original combination. Equivalent admitted local votes
        # still schedule their targets on their exact support.
        numerical = active & target_active & (hard == hard[winner])
        if specification.size_combination is SizeCombinationPolicy.EXPLICIT_PRIORITY:
            numerical &= priorities == priorities[winner]
        if np.any(numerical):
            if not target_active[winner]:
                winner = _select_control(
                    candidates,
                    numerical,
                    hard,
                    priorities,
                    (lower, upper),
                    specification.size_combination,
                )
            values[point] = np.clip(candidates[winner], lower, upper)
            if controls[winner].strength is SizeControlStrength.HARD:
                # Statistical targets remain preferences for unmet hard goals.
                values[point] *= target_radius_factor
        else:
            values[point] = 0.0
        if np.isfinite(upper):
            # A circumradius cap of half the physical maximum bounds every
            # tetrahedral edge, independent of the selected target preference.
            cap = 0.5 * upper
            values[point] = min(values[point], cap) if values[point] > 0.0 else cap
    return values


def _exhausted(
    stage: MeshingStageKind,
    message: str,
    run: TetMeshRun | None,
    counter_names: tuple[str, ...] = (),
    /,
) -> NoReturn:
    """Refuse with the stage's actual native counters under their vocabulary."""
    raise MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED,
        message,
        stage=stage.value,
        achieved=()
        if run is None
        else tuple(
            (name, int(value))
            for name, value in zip(counter_names, run.counters.tolist(), strict=True)
        ),
    )


def _facet_diagonals(
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    plc_edges: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Interior triangulation edges retained by the fixed-boundary contract.

    Fixed boundaries preserve the exact supplied triangles. Conforming facets
    instead retain their native constrained-face source identity; a coplanar
    diagonal is triangulation wiring, not a represented feature curve.
    """

    triangles = input_triangles.astype(np.int64)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    polygons = np.tile(input_polygons.astype(np.int64), 3)
    keys = np.sort(edges, axis=1)
    known = {tuple(pair) for pair in np.sort(plc_edges, axis=1).tolist()}
    interior = np.asarray(
        [tuple(pair) not in known for pair in keys.tolist()], dtype=np.bool_
    )
    unique, first = np.unique(keys[interior], axis=0, return_index=True)
    return unique, polygons[interior][first]


def _refine_and_improve(
    state: TetMesh3D,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    deferred_boundary_targets: tuple[bool, ...],
    initial_vertex_count: int,
    construction_work_units: int,
    /,
    *,
    validity_policy: CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> tuple[TetMesh3D, TetMeshRun, TetMeshRun, TetMeshRun | None, np.ndarray | None, int]:
    limits: MeshingLimits = specification.limits
    spent = construction_work_units
    insertions = limits.maximum_vertices - initial_vertex_count
    if spent >= limits.maximum_work_units:
        _exhausted(
            MeshingStageKind.VOLUME_FILL,
            "Initial volume construction consumed the work budget of refinement.",
            None,
        )
    refinement = state.refine(
        radius_edge_bound=schedule.radius_edge_bound,
        sizes=lambda points: _field_sizes(
            points,
            complex_,
            specification,
            input_triangles,
            input_polygons,
            deferred_boundary_targets=deferred_boundary_targets,
        ),
        max_insertions=insertions,
        work_limit=_native_volume_remaining_work(limits, spent),
        max_rounds=schedule.refinement_rounds,
        record_native_phase=_native_phase_recorder(record_phase),
    )
    if refinement.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        _exhausted(
            MeshingStageKind.VOLUME_FILL,
            "Tetrahedral refinement exceeded its declared budget.",
            refinement,
            TET_MESH_REFINE_COUNTERS,
        )
    spent += int(refinement.counters[TET_MESH_REFINE_COUNTERS.index("work_units")])
    if spent >= limits.maximum_work_units:
        _exhausted(
            MeshingStageKind.OPTIMIZATION,
            "Refinement consumed the work budget of improvement.",
            None,
        )
    state.measure_execution(record_phase is not None)
    try:
        improvement = state.improve(
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=validity_policy.relative_determinant_floor,
            max_passes=schedule.improvement_passes,
            work_limit=_native_volume_remaining_work(limits, spent),
        )
        if record_phase is not None:
            seconds, measured = state.execution_times()
            if measured[1]:
                record_phase(
                    NativeMeshingPhaseMeasurement(
                        "improvement",
                        float(seconds[1]),
                        int(
                            improvement.counters[
                                TET_MESH_IMPROVE_COUNTERS.index("work_units")
                            ]
                        ),
                    )
                )
    finally:
        state.measure_execution(False)
    if improvement.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        _exhausted(
            MeshingStageKind.OPTIMIZATION,
            "Tetrahedral improvement exceeded its declared budget.",
            improvement,
            TET_MESH_IMPROVE_COUNTERS,
        )
    spent += int(improvement.counters[TET_MESH_IMPROVE_COUNTERS.index("work_units")])
    exudation, weights, spent = exude_native_volume(
        state,
        schedule,
        limits,
        spent,
        validity_policy=validity_policy,
        record_phase=record_phase,
    )
    return state, refinement, improvement, exudation, weights, spent


def exude_native_volume(
    state: TetMesh3D,
    schedule: NativeVolumeSchedule,
    limits: MeshingLimits,
    spent: int,
    /,
    *,
    validity_policy: CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> tuple[TetMeshRun | None, np.ndarray | None, int]:
    """The explicit bounded weighted sliver stage after ordinary improvement.

    It shares the improvement's dihedral and radius-edge aims and the
    publishing determinant floor; only unprotected interior vertices receive
    weights. Unmet cells and refusals remain native evidence of this stage.
    """
    if schedule.exudation_passes == 0:
        return None, None, spent
    if spent >= limits.maximum_work_units:
        _exhausted(
            MeshingStageKind.OPTIMIZATION,
            "Improvement consumed the work budget of exudation.",
            None,
        )
    state.measure_execution(record_phase is not None)
    try:
        exudation, weights = state.exude(
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            max_weight_fraction=schedule.exudation_weight_fraction,
            radius_edge_bound=schedule.radius_edge_bound,
            minimum_relative_determinant=validity_policy.relative_determinant_floor,
            max_passes=schedule.exudation_passes,
            work_limit=_native_volume_remaining_work(limits, spent),
        )
        if record_phase is not None:
            seconds, measured = state.execution_times()
            if measured[2]:
                record_phase(
                    NativeMeshingPhaseMeasurement(
                        "exudation",
                        float(seconds[2]),
                        int(
                            exudation.counters[
                                TET_MESH_EXUDE_COUNTERS.index("work_units")
                            ]
                        ),
                    )
                )
    finally:
        state.measure_execution(False)
    if exudation.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        _exhausted(
            MeshingStageKind.OPTIMIZATION,
            "Weighted sliver exudation exceeded its declared budget.",
            exudation,
            TET_MESH_EXUDE_COUNTERS,
        )
    return (
        exudation,
        weights,
        spent + int(exudation.counters[TET_MESH_EXUDE_COUNTERS.index("work_units")]),
    )


def _rows(canonical: np.ndarray, entities: np.ndarray, /) -> np.ndarray:
    """Row of every entity (sorted vertex tuple) in a canonical incidence table."""

    lookup = {
        row: index for index, row in enumerate(map(tuple, np.sort(canonical, axis=1)))
    }
    return np.asarray(
        [lookup[row] for row in map(tuple, np.sort(entities, axis=1))], dtype=np.int64
    )


def _face_orientations(
    points: np.ndarray,
    rows: np.ndarray,
    sources: np.ndarray,
    cells: np.ndarray,
    cell_regions: np.ndarray,
    complex_: PiecewiseLinearComplex,
    /,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
) -> np.ndarray:
    """Orientation of every constrained face row relative to its source facet.

    The facet normal points into its positive region: a face bounding a cell
    of the negative region is aligned when its row normal leaves that cell.
    Internal sheets compare exact overlapping source-triangle planes. A
    conforming face may span coplanar triangulation wiring, but opposite
    winding or noncoplanar sheets must never share an averaged normal.
    """

    corners = points[rows]
    incidence = complex_.facet_regions[sources]
    keyed: dict[tuple[int, ...], int] = {}
    for cell, vertices in enumerate(cells.tolist()):
        for opposite in range(4):
            face = tuple(sorted(vertices[:opposite] + vertices[opposite + 1 :]))
            keyed.setdefault(face, cell)
    adjacent = np.asarray(
        [keyed[tuple(sorted(row))] for row in rows.tolist()], dtype=np.int64
    )
    apex = np.asarray(
        [
            next(v for v in cells[cell].tolist() if v not in row)
            for cell, row in zip(adjacent.tolist(), rows.tolist(), strict=True)
        ],
        dtype=np.int64,
    )
    toward_cell = (
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], points[apex]) > 0
    )
    in_positive = cell_regions[adjacent] == incidence[:, 0]
    aligned = np.where(in_positive, toward_cell, ~toward_cell)
    sheet = incidence[:, 0] == incidence[:, 1]
    result = np.where(aligned, 1, -1).astype(np.int8)
    triangle_facets = complex_.polygon_facets[input_polygons]
    for face in np.flatnonzero(sheet).tolist():
        candidates = input_triangles[triangle_facets == sources[face]]
        triangles = complex_.vertices[candidates]
        realized = corners[face]
        for axis in range(3):
            projection = [(axis + 1) % 3, (axis + 2) % 3]
            realized_2d = realized[:, projection]
            realized_sign = exact_orient2d(
                realized_2d[0], realized_2d[1], realized_2d[2]
            ).item()
            if realized_sign:
                break
        else:
            raise RuntimeError("A constrained sheet face is exactly degenerate.")
        charge_native_geometry_queries(3 * triangles.shape[0])
        sides, features, status = point_triangle_locations(
            np.tile(realized, (triangles.shape[0], 1)),
            np.repeat(triangles, 3, axis=0),
        )
        coplanar = np.all(
            ((status == MeshcoreStatus.OK) & (sides == 0)).reshape(-1, 3), axis=1
        )
        features = features.reshape(-1, 3)
        charge_native_geometry_queries(3 * triangles.shape[0])
        source_sides, source_features, source_status = point_triangle_locations(
            triangles.reshape(-1, 3),
            np.broadcast_to(realized, (3 * triangles.shape[0], 3, 3)),
        )
        source_features = source_features.reshape(-1, 3)
        source_valid = (
            (source_status == MeshcoreStatus.OK) & (source_sides == 0)
        ).reshape(-1, 3)
        overlap = np.any(features == 6, axis=1) | np.all(features >= 0, axis=1)
        overlap |= np.any(source_valid & (source_features == 6), axis=1)
        overlap |= np.all(source_valid & (source_features >= 0), axis=1)
        source_2d = triangles[:, :, projection]
        for first in range(3):
            a, b = source_2d[:, first], source_2d[:, (first + 1) % 3]
            for second in range(3):
                c, d = realized_2d[second], realized_2d[(second + 1) % 3]
                overlap |= (exact_orient2d(a, b, c) * exact_orient2d(a, b, d) < 0) & (
                    exact_orient2d(c, d, a) * exact_orient2d(c, d, b) < 0
                )
        matches = np.flatnonzero(coplanar & overlap)
        if not matches.size:
            raise RuntimeError("A recovered sheet face lacks exact source-plane overlap.")
        source_signs = exact_orient2d(
            source_2d[matches, 0], source_2d[matches, 1], source_2d[matches, 2]
        )
        if np.any(source_signs != source_signs[0]):
            raise RuntimeError(
                "A recovered sheet face crosses incompatible source winding."
            )
        result[face] = int(source_signs[0]) * realized_sign
    return result


def _organization(
    mesh: CellMesh,
    cell_regions: np.ndarray,
    faces: np.ndarray,
    face_sources: np.ndarray,
    segments: np.ndarray,
    segment_sources: np.ndarray,
    plc_edges: np.ndarray,
    diagonal_facets: np.ndarray,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    source_id: str,
    revision: str,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    input_vertices: np.ndarray,
    /,
) -> tuple[
    tuple[MeshZone, ...],
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[GeometryAssociation, ...],
]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError("The PLC volume mesh must carry tetrahedral connectivity.")
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    cell_set = mesh.entity_set(3)
    face_set = mesh.entity_set(2)
    edge_set = mesh.entity_set(1)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)

    def scope(
        dimension: int, entity_set_id: str, selected: np.ndarray, /
    ) -> MeshingScope:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            dimension,
            entity_set_id,
            np.sort(selected),
        )

    seeds = {seed.region_name: seed for seed in specification.region_seeds}
    controls = {control.region_name: control for control in specification.region_controls}
    zones = tuple(
        MeshZone(
            name,
            MeshZoneRole.REGION,
            scope(3, cell_set.entity_set_id, cell_ids[cell_regions == region]),
            material_id=(
                controls[name].material_id
                if name in controls
                else seeds[name].material_id
                if name in seeds
                else None
            ),
            region_role=(
                controls[name].role
                if name in controls
                else seeds[name].role
                if name in seeds
                else None
            ),
        )
        for region, name in enumerate(complex_.region_ids)
    )
    face_rows = _rows(np.asarray(connectivity.faces, dtype=np.int64), faces)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)[face_rows]
    face_order = np.argsort(face_ids, kind="stable")
    face_ids, face_rows = face_ids[face_order], face_rows[face_order]
    face_sources = face_sources[face_order]
    canonical_faces = np.asarray(connectivity.faces, dtype=np.int64)[face_rows]
    orientations = _face_orientations(
        points,
        canonical_faces,
        face_sources,
        cells,
        cell_regions,
        complex_,
        input_triangles,
        input_polygons,
    )
    # Subsegments carry their PLC edge (source dimension 1); every other edge
    # of a constrained face lies inside that face's facet (source dimension 2).
    face_edges = np.concatenate(
        (
            canonical_faces[:, [0, 1]],
            canonical_faces[:, [1, 2]],
            canonical_faces[:, [2, 0]],
        )
    )
    candidates = np.concatenate((segments, face_edges))
    # Subsegments of facet-triangulation diagonals lie inside their facet.
    edge_count = plc_edges.shape[0]
    on_diagonal = segment_sources >= edge_count
    segment_sources = np.where(
        on_diagonal,
        diagonal_facets[np.maximum(segment_sources - edge_count, 0)]
        if diagonal_facets.size
        else segment_sources,
        segment_sources,
    )
    candidate_dimensions = np.concatenate(
        (
            np.where(on_diagonal, 2, 1).astype(np.int64),
            np.full((face_edges.shape[0],), 2, dtype=np.int64),
        )
    )
    candidate_sources = np.concatenate((segment_sources, np.tile(face_sources, 3)))
    # np.unique keeps the first occurrence: a subsegment wins over its faces.
    edge_rows, first = np.unique(
        _rows(np.asarray(connectivity.edges, dtype=np.int64), candidates),
        return_index=True,
    )
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[edge_rows]
    edge_order = np.argsort(edge_ids, kind="stable")
    edge_ids, edge_rows = edge_ids[edge_order], edge_rows[edge_order]
    edge_dimensions = candidate_dimensions[first][edge_order]
    edge_sources = candidate_sources[first][edge_order]
    canonical_edges = np.asarray(connectivity.edges, dtype=np.int64)[edge_rows]
    along = points[canonical_edges[:, 1]] - points[canonical_edges[:, 0]]
    on_curve = edge_dimensions == 1
    source_direction = np.zeros_like(along)
    source_direction[on_curve] = (
        complex_.vertices[plc_edges[edge_sources[on_curve], 1]]
        - complex_.vertices[plc_edges[edge_sources[on_curve], 0]]
    )
    edge_orientations = np.where(
        on_curve, np.where(np.sum(along * source_direction, axis=1) > 0.0, 1, -1), 0
    ).astype(np.int8)
    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    parents: list[set[tuple[int, int]]] = [set() for _ in range(points.shape[0])]
    for row, dimension, source in zip(
        canonical_edges.tolist(),
        edge_dimensions.tolist(),
        edge_sources.tolist(),
        strict=True,
    ):
        for vertex in row:
            parents[vertex].add((dimension, source))
    for row, source in zip(canonical_faces.tolist(), face_sources.tolist(), strict=True):
        for vertex in row:
            parents[vertex].add((2, source))
    for row, region in zip(cells.tolist(), cell_regions.tolist(), strict=True):
        for vertex in row:
            parents[vertex].add((3, region))
    vertex_dimensions = np.empty((points.shape[0],), dtype=np.int64)
    vertex_sources = np.empty((points.shape[0],), dtype=np.int64)
    for vertex, original in enumerate(input_vertices.tolist()):
        if original >= 0 and np.array_equal(points[vertex], complex_.vertices[original]):
            vertex_dimensions[vertex], vertex_sources[vertex] = 0, original
            continue
        owners = parents[vertex]
        dimension = min(owner[0] for owner in owners)
        selected = sorted(source for kind, source in owners if kind == dimension)
        if len(selected) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                "A PLC vertex has ambiguous lowest-dimensional constraint ancestry.",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                entity_ids=(int(vertex_ids[vertex]),),
            )
        vertex_dimensions[vertex], vertex_sources[vertex] = dimension, selected[0]
    associations = (
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            vertex_set.entity_set_id,
            vertex_ids,
            tuple(
                _entity(
                    revision, ("vertex", "edge", "facet", "region")[dimension], source
                )
                for dimension, source in zip(
                    vertex_dimensions.tolist(), vertex_sources.tolist(), strict=True
                )
            ),
            np.zeros((vertex_ids.size,), dtype=np.float64),
            exact=True,
            source_dimensions=vertex_dimensions,
            source_indices=vertex_sources,
            source_entity_roles=tuple(
                (
                    GeometrySourceEntityRole.VERTEX,
                    GeometrySourceEntityRole.EDGE,
                    GeometrySourceEntityRole.FACET,
                    GeometrySourceEntityRole.REGION,
                )[dimension]
                for dimension in vertex_dimensions.tolist()
            ),
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            cell_set.entity_set_id,
            cell_ids,
            tuple(_entity(revision, "region", int(value)) for value in cell_regions),
            np.zeros((cell_ids.size,), dtype=np.float64),
            exact=True,
            source_dimensions=np.full(cell_ids.shape, 3, dtype=np.int64),
            source_indices=cell_regions,
            source_entity_roles=(GeometrySourceEntityRole.REGION,) * cell_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            face_set.entity_set_id,
            face_ids,
            tuple(_entity(revision, "facet", int(value)) for value in face_sources),
            np.zeros((face_ids.size,), dtype=np.float64),
            exact=True,
            orientations=orientations,
            source_dimensions=np.full(face_ids.shape, 2, dtype=np.int64),
            source_indices=face_sources,
            source_entity_roles=(GeometrySourceEntityRole.FACET,) * face_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            edge_set.entity_set_id,
            edge_ids,
            tuple(
                _entity(revision, "edge" if dimension == 1 else "facet", int(value))
                for dimension, value in zip(
                    edge_dimensions.tolist(), edge_sources.tolist(), strict=True
                )
            ),
            np.zeros((edge_ids.size,), dtype=np.float64),
            exact=True,
            orientations=edge_orientations,
            source_dimensions=edge_dimensions,
            source_indices=edge_sources,
            source_entity_roles=tuple(
                GeometrySourceEntityRole.EDGE
                if dimension == 1
                else GeometrySourceEntityRole.FACET
                for dimension in edge_dimensions.tolist()
            ),
        ),
    )
    patches = tuple(
        MeshPatch(
            f"facet:{facet}",
            scope(2, face_set.entity_set_id, face_ids[face_sources == facet]),
        )
        for facet in np.unique(face_sources).tolist()
    )
    patches += tuple(
        MeshPatch(
            control.name,
            scope(
                2,
                face_set.entity_set_id,
                face_ids[np.isin(face_sources, np.asarray(control.scope.entity_ids))],
            ),
        )
        for control in specification.patch_controls
    )
    incidence = complex_.facet_regions[face_sources]
    interface = (incidence[:, 0] >= 0) & (incidence[:, 1] >= 0)
    sheet = incidence[:, 0] == incidence[:, 1]
    curve = on_curve & (edge_sources < complex_.segments.shape[0])
    labels = tuple(
        MeshLabel(name, scope(dimension, entity_set_id, selected))
        for name, dimension, entity_set_id, selected in (
            ("interface", 2, face_set.entity_set_id, face_ids[interface & ~sheet]),
            ("internal_sheet", 2, face_set.entity_set_id, face_ids[sheet]),
            ("internal_curve", 1, edge_set.entity_set_id, edge_ids[curve]),
        )
        if selected.size
    )
    return zones, patches, labels, associations


def declared_plc_domain(
    complex_: PiecewiseLinearComplex, source_id: str, /
) -> PiecewiseLinearDomain:
    """Triangulate only authoritative source polygons for independent acceptance.

    This does not recover or fill the PLC. It prevents an already-built carrier
    from selecting its own smaller domain as the reference coverage oracle.
    """
    triangles: list[np.ndarray] = []
    polygons: list[np.ndarray] = []
    for polygon, (start, end) in enumerate(
        zip(complex_.polygon_offsets[:-1], complex_.polygon_offsets[1:], strict=True)
    ):
        loop = complex_.polygon_vertices[start:end]
        corners = complex_.vertices[loop]
        if loop.size == 3:
            triangles.append(loop[None, :])
            polygons.append(np.asarray([polygon], dtype=np.int64))
            continue
        projection = None
        for third in range(2, loop.size):
            for axis in range(3):
                selected = [(axis + 1) % 3, (axis + 2) % 3]
                p = corners[:, selected]
                if exact_orient2d(p[0], p[1], p[third]).item():
                    projection = selected
                    if np.any(
                        exact_orient3d(corners[0], corners[1], corners[third], corners)
                    ):
                        raise ValueError("PLC source polygons must be exactly planar.")
                    break
            if projection is not None:
                break
        if projection is None:
            raise ValueError("A PLC source polygon is degenerate.")
        projected = corners[:, projection]
        lowest = np.lexsort((projected[:, 1], projected[:, 0]))[0]
        sign = exact_orient2d(
            projected[(lowest - 1) % loop.size],
            projected[lowest],
            projected[(lowest + 1) % loop.size],
        ).item()
        boundary = np.column_stack(
            (
                np.arange(loop.size, dtype=np.int64),
                np.roll(np.arange(loop.size, dtype=np.int64), -1),
            )
        )
        _, local, _, _, status = constrained_delaunay_2d(
            projected, boundary, max_triangles=2 * loop.size + 8
        )
        if status != MeshcoreStatus.OK or not sign:
            raise ValueError("A PLC source polygon lacks a valid exact triangulation.")
        if sign < 0:
            local = local[:, [0, 2, 1]]
        triangles.append(loop[local])
        polygons.append(np.full((local.shape[0],), polygon, dtype=np.int64))
    return _domain(
        complex_, np.concatenate(triangles), np.concatenate(polygons), source_id
    )


def _domain(
    complex_: PiecewiseLinearComplex,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    source_id: str,
    /,
) -> PiecewiseLinearDomain:
    """Declared domain of the source polygons for independent coverage.

    Facet normals point from the negative into the positive region; internal
    sheets bound no region measure and are not domain facets.
    """

    polygons = input_polygons.astype(np.int64)
    incidence = complex_.facet_regions[complex_.polygon_facets[polygons]]
    kept = incidence[:, 0] != incidence[:, 1]
    return PiecewiseLinearDomain(
        complex_.vertices,
        input_triangles[kept].astype(np.int64),
        incidence[kept][:, ::-1],
        complex_.region_ids,
        source_id=source_id,
    )


def generate_plc_volume(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    source_id: str,
    source_revision: str,
    input_id: str,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
    source_geometry_queries: int = 0,
    operation_started: float | None = None,
    execution_budget: NativeExecutionBudget | None = None,
) -> VolumeConstruction:
    """Recover and finish under one original allowance, including initialization.

    ``validity_policy`` is the cell validity policy of the publishing audit; the
    native improvement repairs or reports every cell below its determinant floor.
    """
    if execution_budget is None:
        execution_budget = current_native_execution_budget()
    started = _native_volume_operation_started(operation_started)
    if execution_budget is None and started is None:
        started = monotonic()
    scope = (
        native_volume_execution_budget(
            specification.limits,
            source_work_units=source_work_units,
            source_geometry_queries=source_geometry_queries,
            operation_started=started,
        )
        if execution_budget is None
        else nullcontext(execution_budget)
    )
    with scope as budget:
        native_volume_checkpoint(
            specification.limits,
            MeshingStageKind.VOLUME_FILL,
            operation_started=started,
            execution_budget=budget,
        )
        construction = _construct_plc_volume(
            complex_,
            specification,
            schedule,
            validity_policy=validity_policy,
            source_id=source_id,
            source_revision=source_revision,
            input_id=input_id,
            record_phase=record_phase,
            source_work_units=source_work_units,
            operation_started=started,
            execution_budget=budget,
        )
    return (
        construction
        if execution_budget is not None
        else _bind_volume_execution(
            construction,
            budget,
            source_work_units,
            source_geometry_queries,
        )
    )


def _construct_plc_volume(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    source_id: str,
    source_revision: str,
    input_id: str,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
    operation_started: float | None = None,
    execution_budget: NativeExecutionBudget,
) -> VolumeConstruction:
    """Recover, refine, improve and assemble one oriented PLC volume mesh."""

    if not isinstance(complex_, PiecewiseLinearComplex):
        raise TypeError("complex_ must be PiecewiseLinearComplex.")
    if not isinstance(schedule, NativeVolumeSchedule):
        raise TypeError("schedule must be NativeVolumeSchedule.")
    if not isinstance(validity_policy, CellValidityPolicy):
        raise TypeError("validity_policy must be CellValidityPolicy.")
    if source_work_units < 0:
        raise ValueError("source_work_units must be nonnegative.")
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.SOURCE_INSPECTION,
        operation_started=operation_started,
        execution_budget=execution_budget,
    )
    recovery = _recover(
        complex_,
        specification,
        record_phase=record_phase,
        source_work_units=source_work_units,
    )
    construction_counters = tuple(
        (name, int(value))
        for name, value in zip(PLC_3D_COUNTERS, recovery.counters.tolist(), strict=True)
    )
    if source_work_units:
        construction_counters += (("source_preparation_work_units", source_work_units),)
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.VOLUME_FILL,
        operation_started=operation_started,
        execution_budget=execution_budget,
    )
    if complex_.boundary == "fixed":
        diagonals, diagonal_polygons = _facet_diagonals(
            recovery.input_triangles,
            recovery.input_polygons,
            recovery.plc_edges,
        )
        segments = np.concatenate((recovery.segments, diagonals))
        segment_sources = np.concatenate(
            (
                recovery.segment_sources,
                recovery.plc_edges.shape[0] + np.arange(diagonals.shape[0]),
            )
        )
    else:
        # The native constructor certifies every unprotected face edge as the
        # interior of two constrained faces of one source: coplanar, or folded
        # only by a certified bounded witness.
        segments = recovery.segments
        segment_sources = recovery.segment_sources
        diagonals = np.empty((0, 2), dtype=np.int32)
        diagonal_polygons = np.empty((0,), dtype=np.int64)
    limits = specification.limits
    declared = _plc_source_complex(complex_, specification, recovery, diagonals)
    state = TetMesh3D(
        recovery.points,
        recovery.tetrahedra,
        recovery.tetrahedron_regions,
        recovery.faces,
        recovery.face_sources,
        segments,
        segment_sources,
        boundary_policy=complex_.boundary,
        protection_radii=recovery.protection_radii,
        max_vertices=limits.maximum_vertices,
        max_tetrahedra=limits.maximum_cells,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        source=declared,
    )
    construction_id = canonical_fingerprint(
        {"kind": "plc-recovery", "input": input_id, "counters": construction_counters}
    )
    stage = MeshingStageReport(
        MeshingStageKind.VOLUME_FILL,
        MeshingStageStatus.PASSED,
        input_ids=(input_id,),
        output_ids=(construction_id,),
        created_count=recovery.tetrahedra.shape[0],
    )
    return finalize_plc_volume(
        state,
        complex_,
        specification,
        schedule,
        validity_policy=validity_policy,
        source_id=source_id,
        source_revision=source_revision,
        input_id=input_id,
        input_triangles=recovery.input_triangles.astype(np.int64),
        input_polygons=recovery.input_polygons.astype(np.int64),
        plc_edges=recovery.plc_edges.astype(np.int64),
        diagonal_polygons=diagonal_polygons,
        construction_stage=stage,
        construction_counters=construction_counters,
        construction_work_units=source_work_units
        + int(recovery.counters[PLC_3D_COUNTERS.index("work_units")]),
        record_phase=record_phase,
        operation_started=operation_started,
        execution_budget=execution_budget,
        source=declared,
    )


def finalize_plc_volume(
    state: TetMesh3D,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    source_id: str,
    source_revision: str,
    input_id: str,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    plc_edges: np.ndarray,
    diagonal_polygons: np.ndarray,
    construction_stage: MeshingStageReport,
    construction_counters: tuple[tuple[str, int], ...],
    construction_work_units: int,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_geometry_queries: int = 0,
    operation_started: float | None = None,
    execution_budget: NativeExecutionBudget | None = None,
    source: TetMeshSourceComplex | None = None,
) -> VolumeConstruction:
    """Consume an authoritative carrier without resetting its producer allowance.

    ``source`` is the declared source complex the carrier was built over (its
    witness rows); ``None`` means its initial faces and segments were their
    own exact source.
    """
    if not isinstance(validity_policy, CellValidityPolicy):
        state.close()
        raise TypeError("validity_policy must be CellValidityPolicy.")
    if execution_budget is None:
        execution_budget = current_native_execution_budget()
    started = _native_volume_operation_started(operation_started)
    if execution_budget is None and started is None:
        started = monotonic()
    scope = (
        native_volume_execution_budget(
            specification.limits,
            source_work_units=construction_work_units,
            source_geometry_queries=source_geometry_queries,
            operation_started=started,
            mesh=state,
        )
        if execution_budget is None
        else nullcontext(execution_budget)
    )
    consumed = False
    try:
        with scope as budget:
            native_volume_checkpoint(
                specification.limits,
                MeshingStageKind.VOLUME_FILL,
                operation_started=started,
                execution_budget=budget,
            )
            consumed = True
            construction = _finish_plc_volume(
                state,
                complex_,
                specification,
                schedule,
                validity_policy=validity_policy,
                source_id=source_id,
                source_revision=source_revision,
                input_id=input_id,
                input_triangles=input_triangles,
                input_polygons=input_polygons,
                plc_edges=plc_edges,
                diagonal_polygons=diagonal_polygons,
                construction_stage=construction_stage,
                construction_counters=construction_counters,
                construction_work_units=construction_work_units,
                record_phase=record_phase,
                operation_started=started,
                execution_budget=budget,
                source=source,
            )
    finally:
        if not consumed:
            state.close()
    return (
        construction
        if execution_budget is not None
        else _bind_volume_execution(
            construction,
            budget,
            construction_work_units,
            source_geometry_queries,
        )
    )


def _finish_plc_volume(
    state: TetMesh3D,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    source_id: str,
    source_revision: str,
    input_id: str,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    plc_edges: np.ndarray,
    diagonal_polygons: np.ndarray,
    construction_stage: MeshingStageReport,
    construction_counters: tuple[tuple[str, int], ...],
    construction_work_units: int,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    operation_started: float | None = None,
    execution_budget: NativeExecutionBudget,
    source: TetMeshSourceComplex | None = None,
) -> VolumeConstruction:
    """Finish an authoritative native carrier under the same PLC obligations.

    The mutable state is consumed and closed on every exit. Its initial source
    vertex prefix must be unchanged. Construction evidence describes the real
    producer (recovery, cutcells, or another native owner), never fabricated PLC
    recovery counters. This is not publication: independent domain/embedding
    and physical-compliance acceptance still runs in the provider.
    """
    try:
        native_volume_checkpoint(
            specification.limits,
            MeshingStageKind.VOLUME_FILL,
            operation_started=operation_started,
            execution_budget=execution_budget,
        )
        initial = state.arrays()
        if not np.array_equal(
            initial.points[: complex_.vertices.shape[0]], complex_.vertices
        ):
            raise ValueError(
                "The carrier must retain the authoritative PLC vertex prefix."
            )
        if construction_work_units < 0:
            raise ValueError("Construction work cannot be negative.")
        seeds, seed_regions = _seeds(complex_, specification)
        seed_work = 0
        if seeds.shape[0]:
            seed_queries_before = execution_budget.remaining().remaining_geometry_queries
            corners = initial.points[initial.tetrahedra]
            skin = initial.points[initial.faces]
            for index, (seed, expected) in enumerate(
                zip(seeds, seed_regions, strict=True)
            ):
                execution_budget.charge(geometry_queries=1, work=skin.shape[0])
                seed_work += skin.shape[0]
                _, features, status = point_triangle_locations(
                    np.repeat(seed[None, :], skin.shape[0], axis=0), skin
                )
                if np.any(status != MeshcoreStatus.OK) or np.any(features >= 0):
                    raise MeshingFailure(
                        MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                        "A carrier seed lies on a constrained facet.",
                        stage=MeshingStageKind.VOLUME_FILL.value,
                        entity_ids=(index,),
                    )
                contained = np.ones((corners.shape[0],), dtype=np.bool_)
                for slot in range(4):
                    execution_budget.charge(work=corners.shape[0])
                    seed_work += corners.shape[0]
                    operands = [corners[:, k] if k != slot else seed for k in range(4)]
                    contained &= exact_orient3d(*operands) >= 0
                labels = np.unique(initial.tetrahedron_regions[contained])
                valid = (
                    not labels.size
                    if expected < 0
                    else (labels.size == 1 and int(labels[0]) == int(expected))
                )
                if not valid:
                    raise MeshingFailure(
                        MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                        "A native carrier contradicts the declared region or void seed.",
                        stage=MeshingStageKind.VOLUME_FILL.value,
                        entity_ids=(index,),
                    )
            construction_counters += (
                (
                    "seed_geometry_queries",
                    seed_queries_before
                    - execution_budget.remaining().remaining_geometry_queries,
                ),
                ("seed_host_row_visits", seed_work),
            )
        construction_work_units += seed_work
        protected_vertices = [
            np.asarray(feature.scope.entity_ids, dtype=np.int64)
            for feature in specification.protected_features
            if feature.scope.entity_dimension == 0
        ]
        size_goal = _uniform_background_goal(specification, schedule, validity_policy)
        protection_work_start = state.work_units()
        state.set_work_limit(
            _native_volume_remaining_work(specification.limits, construction_work_units)
        )
        if size_goal is not None:
            originals = np.flatnonzero(
                initial.vertex_dimension[: complex_.vertices.shape[0]] >= 0
            ).astype(np.int64, copy=False)
            protected_vertices.append(originals)
        if protected_vertices:
            try:
                state.protect_vertices(np.unique(np.concatenate(protected_vertices)))
            except MeshcoreError as error:
                if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
                    raise
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Native original-vertex protection exhausted the remaining work allowance.",
                    stage=MeshingStageKind.VOLUME_FILL.value,
                    requested=(
                        ("maximum_work_units", specification.limits.maximum_work_units),
                    ),
                    achieved=(
                        (
                            "work_units",
                            construction_work_units
                            + state.work_units()
                            - protection_work_start,
                        ),
                        *(
                            (f"construction:{name}", value)
                            for name, value in construction_counters
                        ),
                        *(
                            (f"native_memory:{index}", int(value))
                            for index, value in enumerate(state.memory_evidence())
                        ),
                    ),
                ) from error
        protection_work = state.work_units() - protection_work_start
        construction_work_units += protection_work
        if protection_work:
            construction_counters += (("protected_vertex_work_units", protection_work),)
        native_volume_checkpoint(
            specification.limits,
            MeshingStageKind.VOLUME_FILL,
            operation_started=operation_started,
            execution_budget=execution_budget,
        )
        deferred_boundary_targets = (
            (True,)
            if size_goal is not None
            else _satisfied_boundary_size_targets(
                specification, initial.points, initial.tetrahedra
            )
        )
        state, refinement, improvement, exudation, exudation_weights, spent = (
            _refine_and_improve(
                state,
                complex_,
                specification,
                schedule,
                input_triangles,
                input_polygons,
                deferred_boundary_targets,
                initial.points.shape[0],
                construction_work_units,
                validity_policy=validity_policy,
                record_phase=record_phase,
            )
        )
        native_volume_checkpoint(
            specification.limits,
            MeshingStageKind.OPTIMIZATION,
            operation_started=operation_started,
            execution_budget=execution_budget,
        )
        unmet_cells = state.unmet()
        metric_outcome = (
            _optimize_uniform_background(
                state,
                specification,
                schedule,
                size_goal,
                complex_.vertices.shape[0],
                spent,
                record_phase=record_phase,
                execution_budget=execution_budget,
                operation_started=operation_started,
                declared_source=source,
                source_id=source_id,
                source_revision=source_revision,
            )
            if size_goal is not None
            else None
        )
        metric_optimization = (
            metric_outcome.evidence if metric_outcome is not None else None
        )
        if metric_optimization is not None:
            spent += metric_optimization.work_units
            if metric_optimization.status is MetricRemeshingStatus.RESOURCE_LIMIT:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    metric_optimization.resource_message
                    or "The native statistical-size epoch exhausted its declared resource allowance.",
                    stage=MeshingStageKind.OPTIMIZATION.value,
                    requested=(
                        ("maximum_work_units", specification.limits.maximum_work_units),
                        ("lower_metric_length", metric_optimization.lower_metric_length),
                        ("upper_metric_length", metric_optimization.upper_metric_length),
                        *(
                            (f"metric:resource:{name}", value)
                            for name, value in (metric_optimization.resource_requested)
                        ),
                    ),
                    achieved=(
                        ("work_units", spent),
                        *(
                            (f"construction:{name}", value)
                            for name, value in construction_counters
                        ),
                        *(
                            (f"refinement:{name}", int(value))
                            for name, value in zip(
                                TET_MESH_REFINE_COUNTERS,
                                refinement.counters.tolist(),
                                strict=True,
                            )
                        ),
                        *(
                            (f"improvement:{name}", int(value))
                            for name, value in zip(
                                TET_MESH_IMPROVE_COUNTERS,
                                improvement.counters.tolist(),
                                strict=True,
                            )
                        ),
                        *(
                            ()
                            if exudation is None
                            else (
                                (f"exudation:{name}", int(value))
                                for name, value in zip(
                                    TET_MESH_EXUDE_COUNTERS,
                                    exudation.counters.tolist(),
                                    strict=True,
                                )
                            )
                        ),
                        ("metric:passes", metric_optimization.passes),
                        ("metric:splits", metric_optimization.splits),
                        ("metric:collapses", metric_optimization.collapses),
                        ("metric:flips", metric_optimization.flips),
                        ("metric:relocations", metric_optimization.relocations),
                        (
                            "metric:rejected_operations",
                            metric_optimization.rejected_operations,
                        ),
                        (
                            "metric:source_candidate_pairs",
                            metric_optimization.source_candidate_pairs,
                        ),
                        *(
                            (f"metric:native_memory:{index}", value)
                            for index, value in enumerate(
                                metric_optimization.native_memory_evidence
                            )
                        ),
                        *metric_optimization.size_achieved,
                        *(
                            (f"metric:resource:{name}", value)
                            for name, value in (metric_optimization.resource_achieved)
                        ),
                    ),
                )
        arrays = metric_outcome.snapshot if metric_outcome is not None else state.arrays()
        source_evidence = state.source_evidence()
        if source_evidence.witness_strata.shape[0] != arrays.points.shape[0]:
            raise RuntimeError("Source witnesses must describe the published carrier.")
        quality = state.quality(sliver_degrees=schedule.minimum_dihedral_degrees)
        native_volume_checkpoint(
            specification.limits,
            MeshingStageKind.OPTIMIZATION,
            operation_started=operation_started,
            execution_budget=execution_budget,
        )
    finally:
        state.close()
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.CANONICALIZATION,
        operation_started=operation_started,
        execution_budget=execution_budget,
    )
    if complex_.boundary == "fixed":
        locked = np.unique(
            np.concatenate((complex_.polygon_vertices, complex_.segments.ravel()))
        )
        realized = np.unique(np.sort(arrays.faces, axis=1), axis=0)
        expected = np.unique(np.sort(input_triangles, axis=1), axis=0)
        if not np.array_equal(realized, expected) or not np.array_equal(
            arrays.points[locked], complex_.vertices[locked]
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The native carrier changed immutable PLC vertices or triangles.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
    for identifiers in protected_vertices:
        if np.any(arrays.vertex_dimension[identifiers] < 0) or not np.array_equal(
            arrays.points[identifiers], complex_.vertices[identifiers]
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A protected source point was moved or removed.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                entity_ids=tuple(int(v) for v in identifiers),
            )
    # The mutable native owner keeps reusable retired slots and may construct
    # void-side Steiner points. Publish only vertices incident to the accepted
    # domain complex, preserving source-index ancestry through this explicit map.
    used = np.unique(
        np.concatenate(
            (arrays.tetrahedra.ravel(), arrays.faces.ravel(), arrays.segments.ravel())
        )
    )
    remap = np.full((arrays.points.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    points = arrays.points[used]
    tetrahedra = remap[arrays.tetrahedra]
    faces = remap[arrays.faces]
    segments = remap[arrays.segments]
    input_vertices = np.where(used < complex_.vertices.shape[0], used, -1)
    # Accepted exudation weights by published vertex; later vertices carry zero.
    published_weights = (
        None
        if exudation_weights is None
        else np.concatenate(
            (
                exudation_weights,
                np.zeros(
                    (arrays.points.shape[0] - exudation_weights.shape[0],),
                    dtype=np.float64,
                ),
            )
        )[used]
    )
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.GEOMETRY_ASSOCIATION,
        operation_started=operation_started,
        execution_budget=execution_budget,
    )
    source_fidelity = _published_source_fidelity(
        complex_,
        specification,
        source_evidence,
        used,
        plc_edges,
        input_triangles,
        input_polygons,
        source,
    )
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.CANONICALIZATION,
        operation_started=operation_started,
        execution_budget=execution_budget,
    )
    mesh = canonicalize_cell_mesh(
        CellMesh.from_tetrahedra(points, tetrahedra, numeric_version=source_revision)
    )
    if not np.array_equal(
        np.sort(np.asarray(mesh.blocks[0].vertices, dtype=np.int64), axis=1),
        np.sort(tetrahedra, axis=1),
    ):
        raise RuntimeError("Canonicalization reordered the constructed cells.")
    cell_regions = np.asarray(arrays.tetrahedron_regions, dtype=np.int64)
    with measure_phase(record_phase, "organization"):
        native_volume_checkpoint(
            specification.limits,
            MeshingStageKind.CANONICALIZATION,
            operation_started=operation_started,
            execution_budget=execution_budget,
        )
        zones, patches, labels, associations = _organization(
            mesh,
            cell_regions,
            faces,
            np.asarray(arrays.face_sources, dtype=np.int64),
            segments,
            np.asarray(arrays.segment_sources, dtype=np.int64),
            plc_edges,
            complex_.polygon_facets[diagonal_polygons],
            complex_,
            specification,
            source_id,
            source_revision,
            input_triangles,
            input_polygons,
            input_vertices,
        )
    stages = (
        construction_stage,
        MeshingStageReport(
            MeshingStageKind.OPTIMIZATION,
            MeshingStageStatus.PASSED
            if refinement.status == MeshcoreStatus.OK
            and improvement.status == MeshcoreStatus.OK
            and (exudation is None or exudation.status == MeshcoreStatus.OK)
            and (metric_optimization is None or metric_optimization.converged)
            else MeshingStageStatus.WARNING,
            input_ids=tuple(dict.fromkeys((input_id, *construction_stage.output_ids))),
            output_ids=(
                (mesh.mesh_id,)
                if metric_optimization is None
                else (mesh.mesh_id, metric_optimization.evidence_id)
            ),
            created_count=mesh.blocks[0].vertices.shape[0],
        ),
    )
    unmet = tuple(
        (TET_MESH_UNMET_CRITERIA[criterion], TET_MESH_UNMET_REASONS[reason], count)
        for (criterion, reason), count in sorted(
            Counter(
                zip(
                    unmet_cells.criteria.tolist(),
                    unmet_cells.reasons.tolist(),
                    strict=True,
                )
            ).items()
        )
    )
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.CANONICALIZATION,
        operation_started=operation_started,
        execution_budget=execution_budget,
    )
    construction = VolumeConstruction(
        mesh,
        cell_regions,
        _domain(complex_, input_triangles, input_polygons, source_id),
        zones,
        patches,
        labels,
        associations,
        stages,
        construction_counters,
        refinement,
        improvement,
        quality,
        unmet,
        int(np.count_nonzero(used >= complex_.vertices.shape[0])),
        spent,
        validity_policy.policy_id,
        metric_optimization,
        source_fidelity=source_fidelity,
        exudation=exudation,
        exudation_weights=published_weights,
    )
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.CANONICALIZATION,
        operation_started=operation_started,
        execution_budget=execution_budget,
    )
    return construction


def _published_source_fidelity(
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    evidence: TetMeshSourceEvidence,
    used: np.ndarray,
    plc_edges: np.ndarray,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    source: TetMeshSourceComplex | None,
    /,
) -> PlcSourceFidelity:
    """Native certified witnesses of the published vertices per PLC entity.

    Over a declared PLC source complex, a bounded vertex on a PLC edge lies on
    that edge's explicit segment and on every facet the edge bounds, and one
    on an input triangle lies on its facet. Without one, the carrier's own
    rows are not PLC rows: every facet and segment conservatively reports the
    native achieved bound (exactly zero for an exact carrier).
    """
    facet_bounds, segment_bounds = _declared_source_tolerances(complex_, specification)
    triangle_facets = complex_.polygon_facets[input_polygons]
    incident = _edge_facets(plc_edges, input_triangles, triangle_facets)
    strata = evidence.witness_strata[used]
    entities = evidence.witness_entities[used]
    deviations = evidence.witness_deviations[used]
    if source is None:
        bound = float(evidence.achieved_bound)
        return PlcSourceFidelity(
            facet_bounds,
            segment_bounds,
            strata,
            entities,
            evidence.witness_parameters[used],
            deviations,
            np.full_like(facet_bounds, bound),
            np.full_like(segment_bounds, bound),
            bound,
            int(evidence.refused_edges.shape[0]),
        )
    facet_achieved = np.zeros_like(facet_bounds)
    segment_achieved = np.zeros_like(segment_bounds)
    for stratum, entity, deviation in zip(
        strata.tolist(), entities.tolist(), deviations.tolist(), strict=True
    ):
        if deviation == 0.0:
            continue
        if stratum == 2:
            facets = triangle_facets[entity : entity + 1]
        else:
            facets = (
                incident[entity] if entity < len(incident) else np.empty((0,), np.int64)
            )
            if entity < segment_achieved.shape[0]:
                segment_achieved[entity] = max(segment_achieved[entity], deviation)
        facet_achieved[facets] = np.maximum(facet_achieved[facets], deviation)
    return PlcSourceFidelity(
        facet_bounds,
        segment_bounds,
        strata,
        entities,
        evidence.witness_parameters[used],
        deviations,
        facet_achieved,
        segment_achieved,
        float(np.max(deviations, initial=0.0)),
        int(evidence.refused_edges.shape[0]),
        source,
    )


__all__ = [
    "NativeVolumeSchedule",
    "PiecewiseLinearComplex",
    "PlcSourceFidelity",
    "PreparedPlcSource",
    "prepare_plc_source",
    "VolumeConstruction",
    "declared_plc_domain",
    "exude_native_volume",
    "finalize_plc_volume",
    "generate_plc_volume",
    "native_volume_execution_budget",
]
