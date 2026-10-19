#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native constrained tetrahedral generation of oriented piecewise-linear complexes.

Phase order; each phase refuses with stage evidence before the next starts:

1. source contract (`PiecewiseLinearComplex`): structural validation of the
   oriented polygons, facet region incidence, internal curves and regions;
2. boundary recovery (`phydrax._meshcore.recover_plc_3d`): rational PLC
   validation, protected segment recovery, facet recovery by constrained
   cavity fills and region classification by flooding from both sides of every
   constrained facet, holes and seeds, under the fixed or conforming boundary
   policy of the source; conforming Steiner points are rational or, where a
   protected surface, material-interface or curve feature declares a positive
   ``maximum_deviation``, ancestry-backed carriers within that bound;
3. native shape refinement/improvement and the shared in-place metric epoch
   for admitted uniform statistical goals, under the source size request
   and bounded numerical schedule;
4. assembly of the normalized mesh with region zones, facet patches, interface
   and sheet labels and source associations, the native certified source
   witnesses of every published vertex (`PlcSourceFidelity`), plus the
   declared `PiecewiseLinearDomain` of the source polygons that independent
   domain coverage certifies the mesh against.

No external volume provider is invoked. Acceptance and publication belong to
the native provider route.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from time import monotonic
from typing import final, NoReturn

import equinox as eqx
import numpy as np

import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._exact_plc_geometry as _owner_phydrax_discretization__exact_plc_geometry
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._canonical as _dependency_phydrax_meshing__canonical
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
import phydrax.meshing._tetra_metric as _dependency_phydrax_meshing__tetra_metric
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    constrained_delaunay_2d,
    current_native_host_workspace,
    exact_orient2d,
    exact_orient3d,
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionEvidence,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    plc_source_constraints,
    PlcRecovery3D,
    PlcRecoveryFailure,
    PlcSourceConstraints,
    point_triangle_locations,
    recover_plc_3d,
    TET_MESH_UNMET_CRITERIA,
    TET_MESH_UNMET_REASONS,
    TetMesh3D,
    TetMeshArrays,
    TetMeshExudeCounters,
    TetMeshImproveCounters,
    TetMeshQuality,
    TetMeshRefineCounters,
    TetMeshRun,
    TetMeshSourceComplex,
    TetMeshSourceEvidence,
    TetMeshUnmet,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, nonnegative_integer, positive_integer
from ..discretization import CellMesh, TetrahedralConnectivity
from ..typing import checked, Dim, HostInt64, parse
from ._certified_unit_cells import (
    CertifiedSourceLattice,
    construct_unit_cells,
    UNIT_CELL_CONSTRUCTION,
    unit_cell_geometry,
    UnitCellArrays,
)
from ._compliance_failure import compliance_failure
from ._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhase,
    NativeMeshingPhaseMeasurement,
    NativeMeshingPhaseRecorder,
)
from ._native_preparation import native_preparation_ledger
from ._plc_source import PiecewiseLinearComplex
from ._scope import mesh_entity_scope
from ._trace import MeshingStageKind
from .providers._native_route_checks import (
    check_deadline,
    edge_size_evidence,
    uniform_size_compliance,
    unique_edges,
)


class PlcSourceTriangleDim(Dim, minimum=1):
    """Triangulated source polygons, including internal sheets."""


@final
class PreparedPlcSource(StrictModule, NonTrainableState):
    """Immutable source-only PLC preparation bound to source, revision and frame.

    Native source ordering is retained in the transfer's primitive tables.
    No volume mesh, opaque native handle or geometry proximity cache is kept.
    """

    __strict_contract__ = True
    complex: PiecewiseLinearComplex
    association_transfer: _dependency_phydrax_meshing__association.PlcAssociationTransfer
    input_polygons: HostInt64[PlcSourceTriangleDim]
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    preparation_counters: tuple[tuple[str, int], ...] = eqx.field(static=True)
    memory_counters: NativeExecutionMemoryCounters | None
    prepared_id: str = eqx.field(static=True)

    @checked
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
        if not np.array_equal(constraints.points, complex_.vertices):
            raise ValueError(
                "PreparedPlcSource.constraints must retain PLC point ordering; got reordered points"
            )
        polygons = parse(
            np.array(constraints.input_polygons, dtype=np.int64, copy=True),
            HostInt64[PlcSourceTriangleDim],
            "input_polygons",
        )
        if np.any(polygons < 0) or np.any(polygons >= complex_.polygon_facets.shape[0]):
            raise ValueError(
                "PreparedPlcSource.input_polygons must index PLC polygons; got out-of-range rows"
            )
        polygons.setflags(write=False)
        transfer = _dependency_phydrax_meshing__association.PlcAssociationTransfer(
            domain(complex_, constraints.input_triangles, polygons, source_id),
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
            for name, value in constraints.counters.items()
            if name in source_counts
        )
        if execution_evidence is not None:
            preparation_counters = tuple(
                (name, value)
                for name, value in preparation_counters
                if name != "work_units"
            ) + (
                ("work_units", int(execution_evidence.work_counters.construction_work)),
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
        self.memory_counters = (
            execution_evidence.memory_counters
            if execution_evidence is not None
            else constraints.memory_counters
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


@checked
def prepare_plc_source(
    complex_: PiecewiseLinearComplex,
    source_id: str,
    source_revision: str,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
    source_geometry_queries: int = 0,
) -> PreparedPlcSource:
    """Prepare rational source strata without recovering or refining a volume."""
    remaining_work = _native_volume_remaining_work(limits, source_work_units, scope=scope)
    remaining_queries = min(
        limits.maximum_geometry_queries - source_geometry_queries,
        scope.remaining().remaining_geometry_queries,
    )
    if remaining_work <= 0 or remaining_queries <= 0:
        _exhausted(
            MeshingStageKind.SOURCE_INSPECTION,
            "Source preparation consumed the source-renewal work or query allowance.",
            None,
        )
    with native_volume_execution_budget(
        limits,
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
            raise recovery_failure(error) from error
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
        ratio = finite_real_scalar(ratio, "NativeVolumeSchedule.radius_edge_bound")
        if ratio < 1.0:
            raise ValueError(
                "NativeVolumeSchedule.radius_edge_bound must be finite and at least 1; got an invalid ratio"
            )
        angle = finite_real_scalar(angle, "NativeVolumeSchedule.minimum_dihedral_degrees")
        if not 0.0 <= angle < 70.0:
            raise ValueError(
                "NativeVolumeSchedule.minimum_dihedral_degrees must lie in [0, 70); got an invalid angle"
            )
        rounds = positive_integer(
            refinement_rounds, "NativeVolumeSchedule.refinement_rounds"
        )
        passes = positive_integer(
            improvement_passes, "NativeVolumeSchedule.improvement_passes"
        )
        exudation = nonnegative_integer(
            exudation_passes, "NativeVolumeSchedule.exudation_passes"
        )
        if exudation > np.iinfo(np.int32).max:
            raise ValueError(
                "NativeVolumeSchedule.exudation_passes must fit signed int32; got an oversized count"
            )
        weight_fraction = finite_real_scalar(
            float(exudation_weight_fraction),
            "NativeVolumeSchedule.exudation_weight_fraction",
        )
        if not 0.0 < weight_fraction <= 0.25:
            raise ValueError(
                "NativeVolumeSchedule.exudation_weight_fraction must lie in (0, 0.25]; got an invalid fraction"
            )
        metric_passes = positive_integer(
            metric_optimization_passes, "NativeVolumeSchedule.metric_optimization_passes"
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

    ``validity_policy_id`` identifies the determinant floor certified by native
    shape inspection or executed improvement/exudation. Publication audits
    under that same policy. ``refinement`` and ``improvement`` are ``None``
    when the initial zero-deviation source, shape, and statistical goals are proved
    before those phases. ``exudation`` is ``None`` when not executed, and
    ``exudation_weights`` its accepted power weights indexed by published mesh
    vertices; later-created vertices carry zero. They are exudation evidence,
    not a regular-triangulation certificate of the published mesh.
    """

    mesh: CellMesh
    cell_regions: np.ndarray
    domain: domain_coverage_owner.PiecewiseLinearDomain
    zones: tuple[_dependency_phydrax_meshing__organization.MeshZone, ...]
    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...]
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...]
    associations: tuple[association_state_owner.GeometryAssociation, ...]
    stages: tuple[_dependency_phydrax_meshing__trace.MeshingStageReport, ...]
    construction_counters: tuple[tuple[str, int], ...]
    refinement: TetMeshRun[TetMeshRefineCounters] | None
    improvement: TetMeshRun[TetMeshImproveCounters] | None
    quality: TetMeshQuality
    unmet: tuple[tuple[str, str, int], ...]
    steiner_points: int
    work_units: int
    validity_policy_id: str
    metric_optimization: (
        _dependency_phydrax_meshing__tetra_metric.MetricRemeshingEvidence | None
    ) = None
    native_execution_evidence: NativeExecutionEvidence | None = None
    source_fidelity: PlcSourceFidelity | None = None
    exudation: TetMeshRun[TetMeshExudeCounters] | None = None
    exudation_weights: np.ndarray | None = None
    native_preparation_evidence: NativeExecutionRecord | None = None
    native_execution_record: NativeExecutionRecord | None = None
    geometry: _owner_phydrax_discretization__cell_geometry.CellGeometrySpec | None = None


@final
@dataclass(frozen=True, slots=True)
class PlcSourceFidelity:
    """Declared and achieved source deviation of a PLC volume construction.

    ``facet_tolerances`` (F,) and ``segment_tolerances`` (S,) are the bounds
    declared by protected surface/material-interface (facets) and curve
    (explicit segments) features, the smallest one where several apply and
    zero (rational) elsewhere. Per published vertex, ``witness_strata`` (0 none,
    1 PLC edge, 2 input triangle), ``witness_entities`` (the ``plc_edges`` or
    input-triangle row) and ``witness_deviations`` are the natively certified
    ancestry; zero deviation is rational source membership. ``facet_achieved``
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
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Facet and explicit-segment deviation bounds of the protected features."""
    facets = np.full((complex_.facet_count,), np.inf, dtype=np.float64)
    segments = np.full((complex_.segments.shape[0],), np.inf, dtype=np.float64)
    for feature in specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        match feature.scope.entity_dimension, feature.feature_kind:
            case (
                2,
                _dependency_phydrax_meshing__controls.FeatureKind.SURFACE
                | _dependency_phydrax_meshing__controls.FeatureKind.MATERIAL_INTERFACE,
            ):
                target = facets
            case 1, _dependency_phydrax_meshing__controls.FeatureKind.CURVE:
                target = segments
            case _:
                continue
        if np.any(identifiers < 0) or np.any(identifiers >= target.shape[0]):
            raise ValueError(
                "VolumeMeshingSpec.protected_features must select PLC facets or segments; got out-of-range IDs"
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


@final
class PlcSourceRows(StrictModule):
    """Declared source banks and per-row deviation bounds in native row order."""

    points: np.ndarray
    faces: np.ndarray
    triangle_facets: np.ndarray
    face_tolerances: np.ndarray
    segments: np.ndarray
    segment_ids: np.ndarray
    segment_tolerances: np.ndarray


@checked
def plc_source_rows(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    plc_edges: np.ndarray,
    diagonals: np.ndarray,
    /,
) -> PlcSourceRows:
    """source row banks and scientific bounds shared by publication and restore."""
    facet_bounds, segment_bounds = _declared_source_tolerances(complex_, specification)
    triangle_facets = complex_.polygon_facets[input_polygons]
    incident = _edge_facets(plc_edges, input_triangles, triangle_facets)
    edge_bounds = np.empty((plc_edges.shape[0],), dtype=np.float64)
    for row, facets in enumerate(incident):
        bound = float(segment_bounds[row]) if row < segment_bounds.shape[0] else np.inf
        if facets.size:
            bound = min(bound, float(np.min(facet_bounds[facets])))
        edge_bounds[row] = bound if np.isfinite(bound) else 0.0
    return PlcSourceRows(
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
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    recovery: PlcRecovery3D,
    diagonals: np.ndarray,
    /,
) -> TetMeshSourceComplex:
    """The PLC input triangles and edges as the carrier's declared source rows.

    A PLC edge row takes the smallest bound of its explicit segment and its
    incident facets; fixed-boundary diagonals are rational rows.
    """
    rows = plc_source_rows(
        complex_,
        specification,
        recovery.input_triangles,
        recovery.input_polygons,
        recovery.plc_edges,
        diagonals,
    )
    face_ids, face_groups = np.unique(rows.triangle_facets, return_inverse=True)
    return TetMeshSourceComplex(
        rows.points,
        rows.faces,
        face_groups.astype(np.int32),
        rows.face_tolerances,
        rows.segments,
        np.arange(rows.segments.shape[0], dtype=np.int32),
        rows.segment_tolerances,
        recovery.witness_strata,
        recovery.witness_entities,
        recovery.witness_parameters,
        face_ids,
        rows.segment_ids,
    )


@checked
def region_seed_points(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed points and their region indices (-1 for holes)."""

    points: list[np.ndarray] = []
    regions: list[int] = []
    for seed in specification.region_seeds:
        if seed.region_name not in complex_.region_ids:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                f"PiecewiseLinearComplex.region_seeds must name declared regions; got unknown seed {seed.seed_id!r}",
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


@checked
def recovery_failure(
    error: PlcRecoveryFailure, /
) -> _leaf_phydrax_meshing__failures.MeshingFailure:
    if error.status in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT):
        category = (
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
        )
    else:
        match error.reason:
            case (
                "vertex_budget"
                | "tetrahedron_budget"
                | "work_budget"
                | "scratch_byte_budget"
            ):
                category = _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
            case "invalid_seed" | "inconsistent_regions":
                category = _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED
            case "fixed_segment" | "fixed_facet" | "nonrepresentable_steiner":
                category = _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
            case "source_deviation":
                # The nearest ancestry-backed carrier exceeds the declared bound.
                category = _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED
            case _:
                category = (
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SOURCE
                )
    described = ", ".join(f"{kind} {index}" for kind, index in error.entities)
    deviation, bound = error.source_refusal
    return _leaf_phydrax_meshing__failures.MeshingFailure(
        category,
        f"PLC boundary recovery refused: {error.reason} ({described}).",
        provider_code=error.reason,
        stage=MeshingStageKind.VOLUME_FILL.value,
        entity_ids=tuple(index for _, index in error.entities),
        requested=(("source_deviation_bound", bound),)
        if error.reason == "source_deviation"
        else (),
        achieved=error.counters.items()
        + (
            (("certified_source_deviation", deviation),)
            if error.reason == "source_deviation"
            else ()
        ),
    )


def _native_volume_remaining_work(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    spent: int,
    /,
    *,
    scope: NativeExecutionScope,
) -> int:
    """Return the remaining cumulative allowance for an explicit producer."""
    return min(limits.maximum_work_units - spent, scope.remaining().remaining_work_units)


@checked
def native_volume_checkpoint(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    stage: MeshingStageKind,
    /,
    *,
    scope: NativeExecutionScope,
) -> None:
    """Check the request clock before a host phase."""

    check_deadline(scope.operation_started, limits, stage)
    with scope.batch(0) as ticket:
        ticket.checkpoint()


@final
class NativeVolumeRefusal(StrictModule):
    """Resource classification and cut-phase prefix of a request failure."""

    resource: bool
    native_resource: bool
    status: MeshcoreStatus | None
    provider_code: str
    cut_stage_records: tuple[_leaf_phydrax_meshing__failures.CutStageRecord, ...]


@final
class NativeVolumeWork(StrictModule):
    """Named cumulative scope consumption and native batch refusal quantities."""

    quantities: tuple[tuple[str, float], ...]


def _classify_native_volume_refusal(
    error: MeshcoreError | _leaf_phydrax_meshing__failures.MeshingFailure,
    evidence: NativeExecutionEvidence | None,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    source_work: int,
    cut_stage_records: tuple[_leaf_phydrax_meshing__failures.CutStageRecord, ...],
    /,
) -> NativeVolumeRefusal:
    """Classify resource status without replacing scientific capability failures."""
    match error:
        case MeshcoreError():
            own_status = error.status
            phase_resource = False
            failure_provider_code = "resource_exhausted"
        case _leaf_phydrax_meshing__failures.MeshingFailure():
            own_status = None
            phase_resource = error.category in (
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.TIMED_OUT,
            )
            failure_provider_code = error.evidence.provider_code
            if not cut_stage_records:
                cut_stage_records = error.cut_stage_records
        case _:
            raise error
    status = evidence.status if evidence is not None else own_status
    native_resource = status in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT)
    native_provider_code = (
        status.name.lower() if status is not None else "resource_exhausted"
    )
    if status is MeshcoreStatus.CAPACITY_EXCEEDED and evidence is not None:
        total_work = source_work + int(evidence.work_counters.construction_work)
        peak_bytes = (
            int(evidence.memory_counters.peak_bytes)
            + evidence.host_storage_peak_bytes_upper
        )
        if total_work >= limits.maximum_work_units:
            native_provider_code = "work_budget"
        elif peak_bytes >= limits.maximum_scratch_bytes:
            native_provider_code = "scratch_byte_budget"
    provider_code = native_provider_code if native_resource else failure_provider_code
    return NativeVolumeRefusal(
        phase_resource or native_resource,
        native_resource,
        status,
        provider_code,
        cut_stage_records,
    )


def _native_volume_actual_work(
    error: MeshcoreError | _leaf_phydrax_meshing__failures.MeshingFailure,
    evidence: NativeExecutionEvidence | None,
    source_work: int,
    source_queries: int,
    /,
) -> NativeVolumeWork:
    """Collect earned scope work and the native batch's refusal quantities."""
    actual = (
        ()
        if evidence is None
        else (
            ("work_units", source_work + int(evidence.work_counters.construction_work)),
            (
                "geometry_queries",
                source_queries + int(evidence.work_counters.source_queries),
            ),
            ("native_geometry_primitive_queries", evidence.native_primitive_queries),
            ("host_device_work_units", evidence.externally_charged_work),
            ("native_peak_cavity_cells", int(evidence.work_counters.peak_cavity_cells)),
            *(
                (f"native_execution:{name}", value)
                for name, value in evidence.work_counters.items()
            ),
            *(
                (f"native_memory:{name}", value)
                for name, value in evidence.memory_counters.items()
            ),
        )
    )
    if isinstance(error, MeshcoreError):
        requested_upper = error.batch_requested_upper
        refusal_evidence = error.batch_refusal_evidence
        if requested_upper is not None:
            actual += tuple(
                (f"native_batch:requested_{name}", value)
                for name, value in zip(
                    (
                        "work_units",
                        "geometry_queries",
                        "cavity_cells",
                        "managed_bytes",
                    ),
                    requested_upper,
                    strict=True,
                )
            )
        if refusal_evidence is not None:
            actual += tuple(
                (f"native_batch:refusal:{index}", int(value))
                for index, value in enumerate(refusal_evidence)
            )
    return NativeVolumeWork(actual)


def _native_volume_resource_failure(
    error: MeshcoreError | _leaf_phydrax_meshing__failures.MeshingFailure,
    refusal: NativeVolumeRefusal,
    work: NativeVolumeWork,
    evidence: NativeExecutionEvidence | None,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    source_work: int,
    source_queries: int,
    started: float,
    stage: MeshingStageKind,
    /,
) -> _leaf_phydrax_meshing__failures.MeshingFailure:
    """Publish a resource refusal with earned cut phases and measured scope charges."""
    actual, status, provider_code = work.quantities, refusal.status, refusal.provider_code
    requested_limits = (
        ("maximum_work_units", limits.maximum_work_units),
        ("maximum_geometry_queries", limits.maximum_geometry_queries),
        ("maximum_cavity_cells", limits.maximum_cavity_cells),
        ("maximum_scratch_bytes", limits.maximum_scratch_bytes),
        ("maximum_wall_seconds", limits.maximum_wall_seconds),
    )
    match error:
        case _leaf_phydrax_meshing__failures.MeshingFailure():
            failure = error.evidence
            requested = dict(requested_limits)
            requested.update(failure.requested)
            achieved = dict(failure.achieved)
            achieved.update(((f"native_scope:{name}", value) for name, value in actual))
            achieved["native_scope:wall_seconds"] = monotonic() - started
            wrapped = _leaf_phydrax_meshing__failures.MeshingFailure(
                failure.category,
                failure.message,
                stage=failure.stage,
                provider_code=failure.provider_code,
                entity_ids=failure.entity_ids,
                locations=failure.locations,
                requested=tuple(sorted(requested.items())),
                achieved=tuple(sorted(achieved.items())),
                checkpoint_id=failure.checkpoint_id,
                logical_findings=failure.logical_findings,
                cut_stage_records=refusal.cut_stage_records,
            )
            return wrapped
        case MeshcoreError():
            wrapped = _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.TIMED_OUT
                if status == MeshcoreStatus.TIMEOUT
                else _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "execution.native_volume must respect the request allowance; got resource exhaustion",
                stage=stage.value,
                provider_code=provider_code,
                requested=requested_limits,
                achieved=(
                    ("source_work_units", source_work),
                    (
                        "source_geometry_queries",
                        source_queries
                        + (
                            0
                            if evidence is None
                            else int(evidence.work_counters.source_queries)
                        ),
                    ),
                    ("wall_seconds", monotonic() - started),
                    *actual,
                ),
                cut_stage_records=refusal.cut_stage_records,
            )
            return wrapped
        case _:
            raise error


@contextmanager
@checked
def native_volume_execution_budget(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    source_work_units: int = 0,
    source_geometry_queries: int = 0,
    operation_started: float | None = None,
    mesh: TetMesh3D | None = None,
    stage: MeshingStageKind = MeshingStageKind.VOLUME_FILL,
) -> Iterator[NativeExecutionScope]:
    """Open a cumulative request; only its root imports anonymous source totals.

    Nested source metadata describes a prefix already accounted by the root.
    Ended phase evidence identities are imported separately by the preparation ledger.
    """
    source_work = nonnegative_integer(source_work_units, "source_work_units")
    source_queries = nonnegative_integer(
        source_geometry_queries, "source_geometry_queries"
    )
    remaining_work = limits.maximum_work_units - source_work
    remaining_queries = limits.maximum_geometry_queries - source_queries
    started = monotonic() if operation_started is None else operation_started
    if remaining_work < 0 or remaining_queries < 0:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "execution.source_preparation must respect the request allowance; got excess work or queries",
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
    budget = NativeExecutionScope(limits.native_limits(), mesh=mesh)
    cut_stage_records: tuple[_leaf_phydrax_meshing__failures.CutStageRecord, ...] = ()
    try:
        with budget, budget.host_workspace():
            prior_seconds = (
                0.0 if operation_started is None else max(0.0, monotonic() - started)
            )
            started = budget.operation_started
            if budget.parent is None and (source_work or source_queries or prior_seconds):
                budget.import_preparation(
                    work=source_work,
                    geometry_queries=source_queries,
                    elapsed_seconds=prior_seconds,
                )
                started = budget.operation_started
            native_volume_checkpoint(limits, stage, scope=budget)
            pending_error: (
                MeshcoreError | _leaf_phydrax_meshing__failures.MeshingFailure | None
            ) = None
            try:
                yield budget
            except _leaf_phydrax_meshing__failures.CutStageError as staged:
                match staged.cause:
                    case (
                        MeshcoreError() | _leaf_phydrax_meshing__failures.MeshingFailure()
                    ):
                        cut_stage_records = staged.stage_records
                        pending_error = staged.cause
                    case _:
                        raise
            if pending_error is not None:
                raise pending_error
            native_volume_checkpoint(limits, stage, scope=budget)
    except (MeshcoreError, _leaf_phydrax_meshing__failures.MeshingFailure) as error:
        evidence = budget.evidence
        refusal = _classify_native_volume_refusal(
            error, evidence, limits, source_work, cut_stage_records
        )
        if not refusal.resource:
            if refusal.cut_stage_records:
                raise _leaf_phydrax_meshing__failures.CutStageError(
                    error, refusal.cut_stage_records
                ) from error
            raise
        work = _native_volume_actual_work(error, evidence, source_work, source_queries)
        wrapped = _native_volume_resource_failure(
            error,
            refusal,
            work,
            evidence,
            limits,
            source_work,
            source_queries,
            started,
            stage,
        )
        raise wrapped from error


@checked
def require_native_preparation_allowance(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    receipt: NativeExecutionRecord,
    /,
) -> None:
    """Admit represented prior phases; native payload and host upper bounds stay distinct."""
    receipt.require_valid()
    peak_bytes_upper = 0
    peak_cavity = 0
    current: NativeExecutionRecord | None = receipt
    while current is not None:
        peak_bytes_upper = max(
            peak_bytes_upper,
            int(np.asarray(current.memory_counters.peak_bytes))
            + int(np.asarray(current.host_storage_peak_bytes_upper)),
        )
        peak_cavity = max(
            peak_cavity, int(np.asarray(current.work_counters.peak_cavity_cells))
        )
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.TIMED_OUT
            if exceeded == ("wall_seconds",)
            else _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "execution.source_preparation must respect the volume allowance; got excess "
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


@contextmanager
@checked
def native_volume_source_execution_budget(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    receipt: NativeExecutionRecord | None,
    source_owner_id: str,
    /,
    *,
    scope: NativeExecutionScope,
) -> Iterator[NativeExecutionScope]:
    """Use an explicit live source owner or import its ended predecessor once."""
    ledger = native_preparation_ledger(scope)
    if receipt is None:
        if not ledger.source_is_active(scope, source_owner_id):
            raise ValueError(
                "execution.prepared_source must bind ended phase evidence or a live source owner; got neither"
            )
        yield scope
        return
    if receipt.owner_id != source_owner_id:
        raise ValueError(
            "execution.preparation_evidence must bind the prepared source; got a different owner"
        )
    require_native_preparation_allowance(limits, receipt)
    ledger.import_receipt(scope, receipt)
    yield scope


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


@checked
def recover(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
) -> PlcRecovery3D:
    limits = specification.limits
    # The raw recovery API requires capacity for the immutable source prefix.
    # An insufficient public allowance is a resource refusal, not malformed PLC
    # input, and must be reported before calling that API.
    required_vertices = complex_.vertices.shape[0]
    if limits.maximum_vertices < required_vertices:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "VolumeMeshingSpec.limits.maximum_vertices must retain every PLC source vertex; "
            f"got required={required_vertices}, available={limits.maximum_vertices}",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
            provider_code="vertex_budget",
            requested=(("maximum_vertices", limits.maximum_vertices),),
            achieved=(
                ("required_vertices", required_vertices),
                ("available_vertices", limits.maximum_vertices),
            ),
        )
    remaining_work = _native_volume_remaining_work(limits, source_work_units, scope=scope)
    if remaining_work <= 0:
        _exhausted(
            MeshingStageKind.SOURCE_INSPECTION,
            "Source preparation consumed the native construction work allowance.",
            None,
        )
    seeds, seed_regions = region_seed_points(complex_, specification)
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
        raise recovery_failure(error) from error


def _satisfied_boundary_size_targets(
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    /,
) -> tuple[bool, ...]:
    """Freeze numerical target exemptions from the represented initial carrier.

    Only whole-boundary hard controls that pass every source size obligation
    qualify. This does not modify the specification or final acceptance.
    """

    eligible = tuple(
        isinstance(control, _dependency_phydrax_meshing__sizing.UniformSizeControl)
        and control.strength
        is _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
        and np.array_equal(
            control.scope.entity_ids, specification.boundary_scope.entity_ids
        )
        for control in specification.size_controls
    )
    if not any(eligible):
        return eligible
    lengths, growth = edge_size_evidence(
        points,
        unique_edges(tetrahedra, "tetrahedron"),
    )
    return tuple(
        admitted
        and isinstance(control, _dependency_phydrax_meshing__sizing.UniformSizeControl)
        and not uniform_size_compliance(
            control, specification.size_compliance, lengths, growth
        )[2]
        for control, admitted in zip(specification.size_controls, eligible, strict=True)
    )


def _uniform_background_goal(
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    /,
) -> _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal | None:
    """The closed admitted profile for in-place statistical size optimization.

    Its proposals keep the publishing validity policy's determinant floor.
    """
    if len(specification.size_controls) != 1:
        return None
    control = specification.size_controls[0]
    if (
        not isinstance(control, _dependency_phydrax_meshing__sizing.UniformSizeControl)
        or control.strength
        is not _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
        or not np.array_equal(
            control.scope.entity_ids, specification.boundary_scope.entity_ids
        )
    ):
        return None
    return _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal(
        control,
        specification.size_compliance,
        validity_policy.relative_determinant_floor,
        maximum_radius_edge=schedule.radius_edge_bound,
        minimum_dihedral_degrees=schedule.minimum_dihedral_degrees,
    )


@contextmanager
@checked
def native_volume_coordinate_budget(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
) -> Iterator[
    _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
]:
    """Prepare one rational source phase before the independent native epoch."""
    allowance = scope.remaining()
    work = min(limits.maximum_work_units, allowance.remaining_work_units)
    memory = min(limits.maximum_scratch_bytes, allowance.remaining_managed_bytes)
    with scope.batch(
        work,
        queries_upper=min(
            limits.maximum_geometry_queries, allowance.remaining_geometry_queries
        ),
        bytes_upper=memory,
    ) as ticket:
        ledger = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        try:
            with ledger.measurement(), ledger.activate():
                yield ledger
        except _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError as error:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "execution.metric_source must respect the coordinate action allowance; got resource exhaustion",
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
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    goal: _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal,
    original_vertex_count: int,
    spent: int,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    declared_source: TetMeshSourceComplex | None,
    source_id: str,
    source_revision: str,
    initial_geometry: _owner_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
) -> _dependency_phydrax_meshing__tetra_metric.NativeTetraMetricOutcome:
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
        requested, achieved, issues = (
            _dependency_phydrax_meshing__tetra_metric.physical_size_compliance(
                snapshot, goal
            )
        )
        raise compliance_failure(
            "VolumeMeshingSpec.size_controls must share an admissible uniform interval; got "
            + "; ".join(issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            requested=tuple(requested),
            achieved=tuple(achieved),
        )
    limits = specification.limits
    remaining = _native_volume_remaining_work(limits, spent, scope=scope)
    source_evidence = state.source_evidence()
    with native_volume_coordinate_budget(limits, scope=scope) as budget:
        source_geometry = None
        if source_evidence.achieved_bound > 0.0:
            if declared_source is None:
                raise RuntimeError(
                    "PlcCarrier.source must provide declared banks for bounded source ancestry; got None"
                )
            exact = _owner_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource(
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
                budget=budget,
            )
            source_mesh = CellMesh.from_tetrahedra(snapshot.points, snapshot.tetrahedra)
            source_geometry = (
                _owner_phydrax_discretization__cell_geometry.CellGeometrySpec.plc(
                    source_mesh, exact, budget=budget
                )
            )
        source = _dependency_phydrax_meshing__tetra_metric.TetraMetricSource(
            snapshot.points,
            snapshot.tetrahedra,
            np.arange(snapshot.points.shape[0], dtype=np.int64),
            np.arange(snapshot.tetrahedra.shape[0], dtype=np.int64),
            source_geometry=source_geometry,
            budget=budget,
        )
    inverse = 1.0 / target
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64) * (inverse * inverse),
        (snapshot.points.shape[0], 3, 3),
    )
    fixed = np.zeros((snapshot.points.shape[0],), dtype=np.bool_)
    fixed[:original_vertex_count] = True

    def retain_coordinate_storage(bytes_upper: int) -> NativeBatchTicket:
        return scope.batch(0, bytes_upper=bytes_upper)

    with budget.retained_storage(retain_coordinate_storage):
        return _dependency_phydrax_meshing__tetra_metric.execute_native_tetra_metric(
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
            scope=scope,
            initial_geometry=initial_geometry,
            budget=budget,
            operation_started=scope.operation_started,
            maximum_wall_seconds=limits.maximum_wall_seconds,
        )


@final
class PlcSizeField(StrictModule):
    """Uniform-control candidates and retained physical-target exemptions."""

    controls: tuple[_dependency_phydrax_meshing__sizing.UniformSizeControl, ...]
    candidates: np.ndarray
    target_active: np.ndarray
    values: np.ndarray


@final
class PlcSizeSupports(StrictModule):
    """rational represented-facet support of each prepared size-control candidate."""

    masks: np.ndarray


def _prepare_plc_size_field(
    points: np.ndarray,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    deferred_boundary_targets: tuple[bool, ...],
    /,
) -> PlcSizeField | None:
    """Prepare radius candidates without changing the physical control priority."""
    selected: list[_dependency_phydrax_meshing__sizing.UniformSizeControl] = []
    for control in specification.size_controls:
        match control:
            case _dependency_phydrax_meshing__sizing.UniformSizeControl():
                selected.append(control)
            case _:
                raise TypeError(
                    "VolumeMeshingSpec.size_controls must contain uniform size controls; got another control kind"
                )
    controls = tuple(selected)
    if not controls:
        return None
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
            "PlcSizeField.target_active must align with size_controls; got a different extent"
        )
    background_disabled = not np.all(target_active)
    values = np.full(
        (points.shape[0],),
        0.0 if background_disabled else np.max(candidates),
        dtype=np.float64,
    )
    return PlcSizeField(controls, candidates, target_active, values)


def _plc_size_supports(
    points: np.ndarray,
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    field: PlcSizeField,
    /,
    *,
    scope: NativeExecutionScope,
) -> PlcSizeSupports:
    """Classify size-control supports with bounded native point-triangle batches."""
    controls = field.controls
    masks = np.zeros((len(controls), points.shape[0]), dtype=np.bool_)
    facet_sources = complex_.polygon_facets[input_polygons]
    global_controls = tuple(
        np.array_equal(control.scope.entity_ids, specification.boundary_scope.entity_ids)
        for control in controls
    )
    with scope.batch(
        0, queries_upper=points.shape[0] * sum(not global_ for global_ in global_controls)
    ) as ticket:
        for index, (control, global_) in enumerate(
            zip(controls, global_controls, strict=True)
        ):
            if global_:
                masks[index] = True
                continue
            triangles = complex_.vertices[
                input_triangles[np.isin(facet_sources, control.scope.entity_ids)]
            ]
            batch = max(1, 4096 // triangles.shape[0])
            for start in range(0, points.shape[0], batch):
                queries = points[start : start + batch]
                count = queries.shape[0]
                ticket.consume(geometry_queries=count)
                sides, features, status = point_triangle_locations(
                    np.repeat(queries, triangles.shape[0], axis=0),
                    np.tile(triangles, (count, 1, 1)),
                )
                masks[index, start : start + count] = np.any(
                    (
                        (status == MeshcoreStatus.OK) & (sides == 0) & (features >= 0)
                    ).reshape(count, triangles.shape[0]),
                    axis=1,
                )
    return PlcSizeSupports(masks)


def _resolve_plc_size_field(
    field: PlcSizeField,
    supports: PlcSizeSupports,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
) -> np.ndarray:
    """Resolve hard intervals and the winning numerical radius at each point."""
    controls, candidates, target_active, values = (
        field.controls,
        field.candidates,
        field.target_active,
        field.values,
    )
    masks = supports.masks
    hard = np.asarray(
        [
            control.strength
            is _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
            for control in controls
        ],
        dtype=np.bool_,
    )
    priorities = np.asarray([control.priority for control in controls], dtype=np.int64)
    target_radius_factor = np.sqrt(6.0) / 4.0
    for point in range(values.shape[0]):
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
            raise ValueError(
                "VolumeMeshingSpec.size_controls must have intersecting hard intervals; got lower greater than upper"
            )
        # Keep the source controls declared for hard conflicts even when
        # an already-satisfied target no longer drives numerical insertion.
        winner = _dependency_phydrax_meshing__sizing.select_control(
            candidates,
            active,
            hard,
            priorities,
            (lower, upper),
            specification.size_combination,
        )
        # Exempting a target cannot promote a soft or lower-priority control
        # that lost the source combination. Equivalent admitted local votes
        # still schedule their targets on their rational support.
        numerical = active & target_active & (hard == hard[winner])
        if (
            specification.size_combination
            is _dependency_phydrax_meshing__sizing.SizeCombinationPolicy.EXPLICIT_PRIORITY
        ):
            numerical &= priorities == priorities[winner]
        if np.any(numerical):
            if not target_active[winner]:
                winner = _dependency_phydrax_meshing__sizing.select_control(
                    candidates,
                    numerical,
                    hard,
                    priorities,
                    (lower, upper),
                    specification.size_combination,
                )
            values[point] = np.clip(candidates[winner], lower, upper)
            if (
                controls[winner].strength
                is _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
            ):
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


def _field_sizes(
    points: np.ndarray,
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    deferred_boundary_targets: tuple[bool, ...] = (),
) -> np.ndarray:
    """Resolve native radius sizing on represented source-facet supports."""
    field = _prepare_plc_size_field(points, specification, deferred_boundary_targets)
    if field is None:
        return np.zeros((points.shape[0],), dtype=np.float64)
    supports = _plc_size_supports(
        points,
        complex_,
        specification,
        input_triangles,
        input_polygons,
        field,
        scope=scope,
    )
    return _resolve_plc_size_field(field, supports, specification)


def _exhausted(
    stage: MeshingStageKind,
    message: str,
    run: (
        TetMeshRun[TetMeshRefineCounters]
        | TetMeshRun[TetMeshImproveCounters]
        | TetMeshRun[TetMeshExudeCounters]
        | None
    ),
    /,
) -> NoReturn:
    """Refuse with the stage's represented native counters under their vocabulary."""
    raise _leaf_phydrax_meshing__failures.MeshingFailure(
        _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
        message,
        stage=stage.value,
        achieved=() if run is None else run.counters.items(),
    )


def facet_diagonals(
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    plc_edges: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Interior triangulation edges retained by the fixed-boundary contract.

    Fixed boundaries preserve the rational supplied triangles. Conforming facets
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
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    deferred_boundary_targets: tuple[bool, ...],
    initial_vertex_count: int,
    construction_work_units: int,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> PlcRefinement:
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits = specification.limits
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
            scope=scope,
            deferred_boundary_targets=deferred_boundary_targets,
        ),
        max_insertions=insertions,
        work_limit=_native_volume_remaining_work(limits, spent, scope=scope),
        max_rounds=schedule.refinement_rounds,
        record_native_phase=_native_phase_recorder(record_phase),
    )
    if refinement.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        _exhausted(
            MeshingStageKind.VOLUME_FILL,
            "Tetrahedral refinement exceeded its declared budget.",
            refinement,
        )
    spent += refinement.counters.work_units
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
            work_limit=_native_volume_remaining_work(limits, spent, scope=scope),
        )
        if record_phase is not None:
            seconds, measured = state.execution_times()
            if measured[1]:
                record_phase(
                    NativeMeshingPhaseMeasurement(
                        "improvement",
                        float(seconds[1]),
                        improvement.counters.work_units,
                    )
                )
    finally:
        state.measure_execution(False)
    if improvement.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        _exhausted(
            MeshingStageKind.OPTIMIZATION,
            "Tetrahedral improvement exceeded its declared budget.",
            improvement,
        )
    spent += improvement.counters.work_units
    exudation, weights, spent = exude_native_volume(
        state,
        schedule,
        limits,
        spent,
        validity_policy=validity_policy,
        scope=scope,
        record_phase=record_phase,
    )
    return PlcRefinement(refinement, improvement, exudation, weights, spent)


@checked
def exude_native_volume(
    state: TetMesh3D,
    schedule: NativeVolumeSchedule,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    spent: int,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> tuple[TetMeshRun[TetMeshExudeCounters] | None, np.ndarray | None, int]:
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
            work_limit=_native_volume_remaining_work(limits, spent, scope=scope),
        )
        if record_phase is not None:
            seconds, measured = state.execution_times()
            if measured[2]:
                record_phase(
                    NativeMeshingPhaseMeasurement(
                        "exudation",
                        float(seconds[2]),
                        exudation.counters.work_units,
                    )
                )
    finally:
        state.measure_execution(False)
    if exudation.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        _exhausted(
            MeshingStageKind.OPTIMIZATION,
            "Weighted sliver exudation exceeded its declared budget.",
            exudation,
        )
    return (
        exudation,
        weights,
        spent + exudation.counters.work_units,
    )


def _rows(canonical: np.ndarray, entities: np.ndarray, /) -> np.ndarray:
    """Row of every entity (sorted vertex tuple) in a normalized incidence table."""

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
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    """Orientation of every constrained face row relative to its source facet.

    The facet normal points into its positive region: a face bounding a cell
    of the negative region is aligned when its row normal leaves that cell.
    Internal sheets compare rational overlapping source-triangle planes. A
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
    queries_upper = sum(
        6 * int(np.count_nonzero(triangle_facets == sources[face]))
        for face in np.flatnonzero(sheet)
    )
    with scope.batch(0, queries_upper=queries_upper) as ticket:
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
                raise RuntimeError(
                    "PlcCarrier.faces must have nonzero constrained-sheet orientation; got a degenerate face"
                )
            ticket.consume(geometry_queries=3 * triangles.shape[0])
            sides, features, status = point_triangle_locations(
                np.tile(realized, (triangles.shape[0], 1)),
                np.repeat(triangles, 3, axis=0),
            )
            coplanar = np.all(
                ((status == MeshcoreStatus.OK) & (sides == 0)).reshape(-1, 3), axis=1
            )
            features = features.reshape(-1, 3)
            ticket.consume(geometry_queries=3 * triangles.shape[0])
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
                raise RuntimeError(
                    "PlcCarrier.faces must overlap the represented source sheet; got no source-plane overlap"
                )
            source_signs = exact_orient2d(
                source_2d[matches, 0], source_2d[matches, 1], source_2d[matches, 2]
            )
            if np.any(source_signs != source_signs[0]):
                raise RuntimeError(
                    "PlcCarrier.faces must retain source-sheet winding; got incompatible orientations"
                )
            result[face] = int(source_signs[0]) * realized_sign
    return result


@final
class PlcOrganizationContext(StrictModule):
    """Mesh/source identity and tetrahedral incidence used by organization phases."""

    mesh: CellMesh
    connectivity: TetrahedralConnectivity
    points: np.ndarray
    cells: np.ndarray
    cell_regions: np.ndarray
    carrier: PlcCarrier


@final
class PlcEntityBanks(StrictModule):
    """Sorted constrained edge/face IDs with represented source orientation."""

    face_ids: np.ndarray
    face_sources: np.ndarray
    canonical_faces: np.ndarray
    orientations: np.ndarray
    edge_ids: np.ndarray
    edge_dimensions: np.ndarray
    edge_sources: np.ndarray
    canonical_edges: np.ndarray
    edge_orientations: np.ndarray
    on_curve: np.ndarray


@final
class PlcVertexSources(StrictModule):
    """Lowest-dimensional unambiguous source ancestry of published vertices."""

    vertex_dimensions: np.ndarray
    vertex_sources: np.ndarray


@final
class PlcBoundaryOrganization(StrictModule):
    """Facet patches and material-interface/sheet/curve labels."""

    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...]
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...]


def _plc_organization_context(
    mesh: CellMesh, cell_regions: np.ndarray, carrier: PlcCarrier, /
) -> PlcOrganizationContext:
    """Narrow tetrahedral topology and prepare host mesh coordinate banks."""
    connectivity = mesh.connectivity
    match connectivity:
        case TetrahedralConnectivity():
            pass
        case _:
            raise TypeError(
                "PlcCarrier.mesh must carry tetrahedral connectivity; got another connectivity kind"
            )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    return PlcOrganizationContext(
        mesh, connectivity, points, cells, cell_regions, carrier
    )


def _plc_region_zones(
    context: PlcOrganizationContext,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
) -> tuple[_dependency_phydrax_meshing__organization.MeshZone, ...]:
    """Bind declared materials and region roles to ordered published cell IDs."""
    mesh, cell_regions, complex_ = (
        context.mesh,
        context.cell_regions,
        context.carrier.complex,
    )
    cell_set = mesh.entity_set(3)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    seeds = {seed.region_name: seed for seed in specification.region_seeds}
    controls = {control.region_name: control for control in specification.region_controls}
    zones = tuple(
        (
            _dependency_phydrax_meshing__organization.MeshZone(
                name,
                _dependency_phydrax_meshing__organization.MeshZoneRole.REGION,
                mesh_entity_scope(mesh, 3, np.sort(cell_ids[cell_regions == region])),
                material_id=controls[name].material_id
                if name in controls
                else seeds[name].material_id
                if name in seeds
                else None,
                region_role=controls[name].role
                if name in controls
                else seeds[name].role
                if name in seeds
                else None,
            )
            for region, name in enumerate(complex_.region_ids)
        )
    )
    return zones


def _plc_entity_banks(
    context: PlcOrganizationContext,
    published: PlcPublishedCarrier,
    arrays: TetMeshArrays,
    /,
    *,
    scope: NativeExecutionScope,
) -> PlcEntityBanks:
    """Resolve sorted constrained entity banks and source orientation."""
    mesh, connectivity, points, cells, cell_regions = (
        context.mesh,
        context.connectivity,
        context.points,
        context.cells,
        context.cell_regions,
    )
    complex_, plc_edges = context.carrier.complex, context.carrier.plc_edges
    input_triangles, input_polygons = (
        context.carrier.input_triangles,
        context.carrier.input_polygons,
    )
    faces, segments, face_sources, segment_sources = (
        published.faces,
        published.segments,
        np.asarray(arrays.face_sources, dtype=np.int64),
        np.asarray(arrays.segment_sources, dtype=np.int64),
    )
    diagonal_facets = complex_.polygon_facets[context.carrier.diagonal_polygons]
    face_set, edge_set = mesh.entity_set(2), mesh.entity_set(1)
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
        scope=scope,
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
    return PlcEntityBanks(
        face_ids,
        face_sources,
        canonical_faces,
        orientations,
        edge_ids,
        edge_dimensions,
        edge_sources,
        canonical_edges,
        edge_orientations,
        on_curve,
    )


def _plc_vertex_sources(
    context: PlcOrganizationContext,
    banks: PlcEntityBanks,
    published: PlcPublishedCarrier,
    /,
) -> PlcVertexSources:
    """Prove each vertex has one lowest-dimensional declared source ancestor."""
    mesh, points, cells, cell_regions = (
        context.mesh,
        context.points,
        context.cells,
        context.cell_regions,
    )
    complex_ = context.carrier.complex
    canonical_edges, edge_dimensions, edge_sources = (
        banks.canonical_edges,
        banks.edge_dimensions,
        banks.edge_sources,
    )
    canonical_faces, face_sources = banks.canonical_faces, banks.face_sources
    input_vertices = published.input_vertices
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
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.ASSOCIATION_FAILED,
                "PlcCarrier.vertices must have unambiguous lowest-dimensional source ancestry; got conflicting strata",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                entity_ids=(int(vertex_ids[vertex]),),
            )
        vertex_dimensions[vertex], vertex_sources[vertex] = dimension, selected[0]
    return PlcVertexSources(vertex_dimensions, vertex_sources)


def _plc_source_associations(
    context: PlcOrganizationContext, banks: PlcEntityBanks, vertices: PlcVertexSources, /
) -> tuple[association_state_owner.GeometryAssociation, ...]:
    """Publish represented source roles and IDs for every geometric stratum."""
    mesh, cell_regions = context.mesh, context.cell_regions
    source_id, revision = context.carrier.source_id, context.carrier.source_revision
    vertex_set, cell_set, face_set, edge_set = (
        mesh.entity_set(dimension) for dimension in (0, 3, 2, 1)
    )
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    cell_ids = np.asarray(cell_set.entity_ids, dtype=np.int64)
    vertex_dimensions, vertex_sources = (
        vertices.vertex_dimensions,
        vertices.vertex_sources,
    )
    face_ids, face_sources, orientations = (
        banks.face_ids,
        banks.face_sources,
        banks.orientations,
    )
    edge_ids, edge_dimensions, edge_sources, edge_orientations = (
        banks.edge_ids,
        banks.edge_dimensions,
        banks.edge_sources,
        banks.edge_orientations,
    )
    associations = (
        association_state_owner.GeometryAssociation(
            association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            vertex_set.entity_set_id,
            vertex_ids,
            tuple(
                _dependency_phydrax_meshing__host_banks.entity_key(
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
                    association_state_owner.GeometrySourceEntityRole.VERTEX,
                    association_state_owner.GeometrySourceEntityRole.EDGE,
                    association_state_owner.GeometrySourceEntityRole.FACET,
                    association_state_owner.GeometrySourceEntityRole.REGION,
                )[dimension]
                for dimension in vertex_dimensions.tolist()
            ),
        ),
        association_state_owner.GeometryAssociation(
            association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            cell_set.entity_set_id,
            cell_ids,
            tuple(
                _dependency_phydrax_meshing__host_banks.entity_key(
                    revision, "region", int(value)
                )
                for value in cell_regions
            ),
            np.zeros((cell_ids.size,), dtype=np.float64),
            exact=True,
            source_dimensions=np.full(cell_ids.shape, 3, dtype=np.int64),
            source_indices=cell_regions,
            source_entity_roles=(association_state_owner.GeometrySourceEntityRole.REGION,)
            * cell_ids.size,
        ),
        association_state_owner.GeometryAssociation(
            association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            face_set.entity_set_id,
            face_ids,
            tuple(
                _dependency_phydrax_meshing__host_banks.entity_key(
                    revision, "facet", int(value)
                )
                for value in face_sources
            ),
            np.zeros((face_ids.size,), dtype=np.float64),
            exact=True,
            orientations=orientations,
            source_dimensions=np.full(face_ids.shape, 2, dtype=np.int64),
            source_indices=face_sources,
            source_entity_roles=(association_state_owner.GeometrySourceEntityRole.FACET,)
            * face_ids.size,
        ),
        association_state_owner.GeometryAssociation(
            association_state_owner.GeometryAssociationKind.PIECEWISE_LINEAR,
            source_id,
            revision,
            edge_set.entity_set_id,
            edge_ids,
            tuple(
                _dependency_phydrax_meshing__host_banks.entity_key(
                    revision, "edge" if dimension == 1 else "facet", int(value)
                )
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
                association_state_owner.GeometrySourceEntityRole.EDGE
                if dimension == 1
                else association_state_owner.GeometrySourceEntityRole.FACET
                for dimension in edge_dimensions.tolist()
            ),
        ),
    )
    return associations


def _plc_boundary_organization(
    context: PlcOrganizationContext,
    banks: PlcEntityBanks,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    zones: tuple[_dependency_phydrax_meshing__organization.MeshZone, ...],
    /,
) -> PlcBoundaryOrganization:
    """Bind facet patches, requested zone adjacency, and source labels."""
    mesh, complex_ = context.mesh, context.carrier.complex
    face_set, edge_set = mesh.entity_set(2), mesh.entity_set(1)
    face_ids, face_sources, edge_ids, edge_sources, on_curve = (
        banks.face_ids,
        banks.face_sources,
        banks.edge_ids,
        banks.edge_sources,
        banks.on_curve,
    )
    patches = tuple(
        (
            _dependency_phydrax_meshing__organization.MeshPatch(
                f"facet:{facet}",
                mesh_entity_scope(mesh, 2, np.sort(face_ids[face_sources == facet])),
            )
            for facet in np.unique(face_sources).tolist()
        )
    )
    zone_ids = {zone.name: zone.zone_id for zone in zones}
    patches += tuple(
        (
            _dependency_phydrax_meshing__organization.MeshPatch(
                control.name,
                mesh_entity_scope(
                    mesh,
                    2,
                    np.sort(
                        face_ids[
                            np.isin(face_sources, np.asarray(control.scope.entity_ids))
                        ]
                    ),
                ),
                adjacent_zone_ids=tuple(
                    zone_ids[name] for name in control.adjacent_region_names
                ),
            )
            for control in specification.patch_controls
        )
    )
    incidence = complex_.facet_regions[face_sources]
    interface = (incidence[:, 0] >= 0) & (incidence[:, 1] >= 0)
    sheet = incidence[:, 0] == incidence[:, 1]
    curve = on_curve & (edge_sources < complex_.segments.shape[0])
    labels = tuple(
        (
            _dependency_phydrax_meshing__organization.MeshLabel(
                name, mesh_entity_scope(mesh, dimension, np.sort(selected))
            )
            for name, dimension, entity_set_id, selected in (
                ("interface", 2, face_set.entity_set_id, face_ids[interface & ~sheet]),
                ("internal_sheet", 2, face_set.entity_set_id, face_ids[sheet]),
                ("internal_curve", 1, edge_set.entity_set_id, edge_ids[curve]),
            )
            if selected.size
        )
    )
    return PlcBoundaryOrganization(patches, labels)


def _organization(
    mesh: CellMesh,
    cell_regions: np.ndarray,
    published: PlcPublishedCarrier,
    arrays: TetMeshArrays,
    carrier: PlcCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
) -> PlcOrganization:
    """Construct source organization in region, incidence, ancestry, and label order."""
    context = _plc_organization_context(mesh, cell_regions, carrier)
    zones = _plc_region_zones(context, specification)
    banks = _plc_entity_banks(context, published, arrays, scope=scope)
    vertices = _plc_vertex_sources(context, banks, published)
    associations = _plc_source_associations(context, banks, vertices)
    boundary = _plc_boundary_organization(context, banks, specification, zones)
    return PlcOrganization(zones, boundary.patches, boundary.labels, associations)


@checked
def declared_plc_domain(
    complex_: PiecewiseLinearComplex, source_id: str, /
) -> domain_coverage_owner.PiecewiseLinearDomain:
    """Triangulate only declared source polygons for independent acceptance.

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
                        raise ValueError(
                            "PiecewiseLinearComplex.polygons must be planar; got a noncoplanar vertex"
                        )
                    break
            if projection is not None:
                break
        if projection is None:
            raise ValueError(
                "PiecewiseLinearComplex.polygons must have nonzero area; got a degenerate polygon"
            )
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
        native = constrained_delaunay_2d(
            projected, boundary, max_triangles=2 * loop.size + 8
        )
        local = native.triangles
        status = native.status
        if status != MeshcoreStatus.OK or not sign:
            raise ValueError(
                "PiecewiseLinearComplex.polygons must admit predicate-certified triangulation; got an invalid polygon"
            )
        if sign < 0:
            local = local[:, [0, 2, 1]]
        triangles.append(loop[local])
        polygons.append(np.full((local.shape[0],), polygon, dtype=np.int64))
    return domain(
        complex_, np.concatenate(triangles), np.concatenate(polygons), source_id
    )


@checked
def domain(
    complex_: PiecewiseLinearComplex,
    input_triangles: np.ndarray,
    input_polygons: np.ndarray,
    source_id: str,
    /,
) -> domain_coverage_owner.PiecewiseLinearDomain:
    """Declared domain of the source polygons for independent coverage.

    Facet normals point from the negative into the positive region; internal
    sheets bound no region measure and are not domain facets.
    """

    polygons = input_polygons.astype(np.int64)
    incidence = complex_.facet_regions[complex_.polygon_facets[polygons]]
    kept = incidence[:, 0] != incidence[:, 1]
    return domain_coverage_owner.PiecewiseLinearDomain(
        complex_.vertices,
        input_triangles[kept].astype(np.int64),
        incidence[kept][:, ::-1],
        complex_.region_ids,
        source_id=source_id,
    )


@checked
def generate_plc_volume(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    source_id: str,
    source_revision: str,
    input_id: str,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
    certified_lattice: CertifiedSourceLattice | None = None,
    scope: NativeExecutionScope,
) -> VolumeConstruction:
    """Recover and finish under one source allowance, including initialization.

    ``validity_policy`` is the cell validity policy of the publishing audit; the
    native improvement repairs or reports every cell below its determinant floor.
    """

    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.VOLUME_FILL,
        scope=scope,
    )
    with native_volume_execution_budget(
        specification.limits, source_work_units=source_work_units
    ) as execution:
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
            certified_lattice=certified_lattice,
            scope=execution,
        )
    return bind_volume_execution(
        construction, execution.completed_evidence(), source_work_units
    )


@checked
def bind_volume_execution(
    construction: VolumeConstruction,
    evidence: NativeExecutionEvidence,
    source_work_units: int,
    source_geometry_queries: int = 0,
    /,
    *,
    preparation_evidence: NativeExecutionRecord | None = None,
) -> VolumeConstruction:
    """Publish measured construction work and retain its ended phase phase evidence."""
    if preparation_evidence is not None:
        preparation_evidence.require_valid()
        if source_work_units != int(
            np.asarray(preparation_evidence.total_work_units)
        ) or source_geometry_queries != int(
            np.asarray(preparation_evidence.total_geometry_queries)
        ):
            raise ValueError(
                "VolumeConstruction.source_charges must equal ended preparation evidence; got different totals"
            )
    execution_counters = (
        ("native_execution_work_units", int(evidence.work_counters.construction_work)),
        ("native_execution_geometry_queries", int(evidence.work_counters.source_queries)),
        ("native_geometry_primitive_queries", evidence.native_primitive_queries),
        ("host_device_work_units", evidence.externally_charged_work),
        (
            "source_geometry_queries",
            source_geometry_queries + evidence.externally_charged_geometry_queries,
        ),
        ("native_peak_cavity_cells", int(evidence.work_counters.peak_cavity_cells)),
        *(
            (f"native_memory:{name}", value)
            for name, value in evidence.memory_counters.items()
        ),
    )
    replaced_names = {name for name, _ in execution_counters}
    counters = (
        tuple(
            (name, value)
            for name, value in construction.construction_counters
            if name not in replaced_names
        )
        + execution_counters
    )
    return replace(
        construction,
        work_units=source_work_units + int(evidence.work_counters.construction_work),
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


def _construct_plc_volume(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    source_id: str,
    source_revision: str,
    input_id: str,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_work_units: int = 0,
    certified_lattice: CertifiedSourceLattice | None = None,
    scope: NativeExecutionScope,
) -> VolumeConstruction:
    """Recover, refine, improve and assemble one oriented PLC volume mesh."""

    if source_work_units < 0:
        raise ValueError(
            "execution.source_work_units must be nonnegative; got a negative count"
        )
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.SOURCE_INSPECTION,
        scope=scope,
    )
    if certified_lattice is not None and _certified_unit_cell_eligible(
        complex_, specification, schedule, validity_policy, certified_lattice
    ):
        carrier = _construct_certified_unit_cell_carrier(
            complex_,
            specification,
            certified_lattice,
            source_id=source_id,
            source_revision=source_revision,
            input_id=input_id,
            record_phase=record_phase,
            source_work_units=source_work_units,
            scope=scope,
        )
        return _finish_plc_volume(
            carrier,
            specification,
            schedule,
            validity_policy=validity_policy,
            record_phase=record_phase,
            scope=scope,
        )
    recovery = recover(
        complex_,
        specification,
        record_phase=record_phase,
        scope=scope,
        source_work_units=source_work_units,
    )
    construction_counters = recovery.counters.items()
    if source_work_units:
        construction_counters += (("source_preparation_work_units", source_work_units),)
    native_volume_checkpoint(
        specification.limits,
        MeshingStageKind.VOLUME_FILL,
        scope=scope,
    )
    if complex_.boundary == "fixed":
        diagonals, diagonal_polygons = facet_diagonals(
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
        # only by a certified bounded ancestry.
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
    stage = _dependency_phydrax_meshing__trace.MeshingStageReport(
        MeshingStageKind.VOLUME_FILL,
        _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED,
        input_ids=(input_id,),
        output_ids=(construction_id,),
        created_count=recovery.tetrahedra.shape[0],
    )
    carrier = PlcCarrier(
        state,
        complex_,
        source_id,
        source_revision,
        input_id,
        recovery.input_triangles.astype(np.int64),
        recovery.input_polygons.astype(np.int64),
        recovery.plc_edges.astype(np.int64),
        diagonal_polygons,
        stage,
        construction_counters,
        source_work_units + recovery.counters.work_units,
        declared,
    )
    return _finish_plc_volume(
        carrier,
        specification,
        schedule,
        validity_policy=validity_policy,
        record_phase=record_phase,
        scope=scope,
    )


def _certified_unit_cell_eligible(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    lattice: CertifiedSourceLattice | None,
    /,
) -> bool:
    """An explicit compiler certificate and the existing global HARD profile."""
    if lattice is None or complex_.boundary != "conforming":
        return False
    goal = _uniform_background_goal(specification, schedule, validity_policy)
    if goal is None:
        return False
    control = specification.size_controls[0]
    return (
        isinstance(control, _dependency_phydrax_meshing__sizing.UniformSizeControl)
        and control.scope.scope_id == specification.boundary_scope.scope_id
        and control.target_size == lattice.physical_cell_length
        and not specification.layer_controls
        and not specification.periodic_constraints
        and all(
            feature.maximum_deviation == 0.0
            and feature.scope.entity_dimension in (0, 1, 2)
            for feature in specification.protected_features
        )
    )


def _construct_certified_unit_cell_carrier(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    lattice: CertifiedSourceLattice,
    /,
    *,
    source_id: str,
    source_revision: str,
    input_id: str,
    record_phase: NativeMeshingPhaseRecorder | None,
    source_work_units: int,
    scope: NativeExecutionScope,
) -> PlcCarrier:
    """Validate the unchanged source, construct, and enter ordinary finishing."""
    limits = specification.limits
    storage = current_native_host_workspace()
    if storage is None:
        raise RuntimeError(
            "CertifiedSourceLattice.storage must have an active native host workspace; got none"
        )
    storage.retain_owner(lattice)
    remaining_work = _native_volume_remaining_work(limits, source_work_units, scope=scope)
    if remaining_work <= 0:
        _exhausted(
            MeshingStageKind.SOURCE_INSPECTION,
            "Source preparation consumed the native construction work allowance.",
            None,
        )
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
        raise recovery_failure(error) from error
    storage.retain_owner(constraints)
    count = len(lattice.cells)
    # Bound all integer identity dictionaries, incidence rows, simultaneous
    # host arrays, and predicate operands before allocating any template data.
    # Charge deterministic host visits, not fictitious recovery operations.
    host_work = 2048 * count + 16 * (
        len(lattice.vertex_keys)
        + constraints.input_triangles.shape[0]
        + constraints.plc_edges.shape[0]
    )
    host_bytes = 65536 * count + 1024 * (
        len(lattice.vertex_keys)
        + len(lattice.polygon_loops)
        + constraints.input_triangles.shape[0]
        + constraints.plc_edges.shape[0]
    )
    if 32 * count > limits.maximum_cells:
        _exhausted(
            MeshingStageKind.VOLUME_FILL,
            "Certified unit cells exceed the declared tetrahedron allowance.",
            None,
        )
    with measure_phase(record_phase, "construction"):
        with scope.batch(
            host_work,
            bytes_upper=host_bytes,
        ) as ticket:
            # Exact predicates are native primitive telemetry, not source
            # geometry queries (the existing meshcore accounting contract).
            ticket.consume(host_work, managed_bytes=host_bytes)
            arrays = construct_unit_cells(lattice, complex_, constraints)
            ticket.checkpoint()
    storage.retain_owner(arrays)
    rows = plc_source_rows(
        complex_,
        specification,
        constraints.input_triangles,
        constraints.input_polygons,
        constraints.plc_edges,
        np.empty((0, 2), dtype=np.int32),
    )
    face_ids, face_groups = np.unique(rows.triangle_facets, return_inverse=True)
    vertices = arrays.points.shape[0]
    declared = TetMeshSourceComplex(
        rows.points,
        rows.faces,
        face_groups.astype(np.int32),
        rows.face_tolerances,
        rows.segments,
        np.arange(rows.segments.shape[0], dtype=np.int32),
        rows.segment_tolerances,
        np.zeros(vertices, dtype=np.int8),
        np.full(vertices, -1, dtype=np.int32),
        np.zeros((vertices, 2), dtype=np.float64),
        face_ids.astype(np.int64),
        rows.segment_ids,
    )
    storage.retain_owner(declared)
    if vertices > limits.maximum_vertices:
        _exhausted(
            MeshingStageKind.VOLUME_FILL,
            "Certified unit cells exceed the declared vertex allowance.",
            None,
        )
    state = TetMesh3D(
        arrays.points,
        arrays.cells,
        arrays.regions,
        arrays.faces,
        arrays.face_sources,
        arrays.segments,
        arrays.segment_sources,
        boundary_policy=complex_.boundary,
        max_vertices=limits.maximum_vertices,
        max_tetrahedra=limits.maximum_cells,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        source=declared,
    )
    geometry_work_start = scope.remaining().remaining_work_units
    initial = state.arrays()
    with native_volume_coordinate_budget(limits, scope=scope) as budget:
        geometry = unit_cell_geometry(
            arrays,
            CellMesh.from_tetrahedra(initial.points, initial.tetrahedra),
            budget=budget,
        )
    geometry_work = geometry_work_start - scope.remaining().remaining_work_units
    storage.retain_owner(geometry)
    counters = (
        *constraints.counters.items(),
        (UNIT_CELL_CONSTRUCTION, 1),
        ("certified_unit_cell_count", count),
        ("certified_unit_cell_host_work_units", host_work),
        ("certified_unit_cell_orientation_predicates", 32 * count),
        ("source_preparation_work_units", source_work_units),
        ("certified_unit_cell_geometry_work_units", geometry_work),
    )
    construction_id = canonical_fingerprint(
        {
            "kind": UNIT_CELL_CONSTRUCTION,
            "input": input_id,
            "source": complex_.complex_id,
            "source_id": source_id,
            "source_revision": source_revision,
            "lattice_cells": lattice.cells,
            "lattice_affine": lattice.affine,
            "source_vertex_keys": lattice.vertex_keys,
            "coordinate_source": geometry.geometry_layout_id,
            "counters": counters,
        }
    )
    stage = _dependency_phydrax_meshing__trace.MeshingStageReport(
        MeshingStageKind.VOLUME_FILL,
        _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED,
        input_ids=(input_id,),
        output_ids=(construction_id,),
        created_count=arrays.cells.shape[0],
    )
    return PlcCarrier(
        state,
        complex_,
        source_id,
        source_revision,
        input_id,
        constraints.input_triangles.astype(np.int64),
        constraints.input_polygons.astype(np.int64),
        constraints.plc_edges.astype(np.int64),
        np.empty((0,), dtype=np.int64),
        stage,
        counters,
        source_work_units + constraints.counters.work_units + host_work + geometry_work,
        declared,
        geometry=geometry,
        unit_cells=arrays,
    )


@final
class PlcCarrier(StrictModule):
    """Native PLC carrier and the source/producer evidence consumed by finishing."""

    state: TetMesh3D
    complex: PiecewiseLinearComplex
    source_id: str
    source_revision: str
    input_id: str
    input_triangles: np.ndarray
    input_polygons: np.ndarray
    plc_edges: np.ndarray
    diagonal_polygons: np.ndarray
    construction_stage: _dependency_phydrax_meshing__trace.MeshingStageReport
    construction_counters: tuple[tuple[str, int], ...]
    construction_work_units: int
    source: TetMeshSourceComplex | None = None
    geometry: _owner_phydrax_discretization__cell_geometry.CellGeometrySpec | None = None
    unit_cells: UnitCellArrays | None = None


@final
class PlcSeededCarrier(StrictModule):
    """Initial snapshot and construction charges after region-seed validation."""

    initial: TetMeshArrays
    construction_counters: tuple[tuple[str, int], ...]
    construction_work_units: int


@final
class PlcProtectedCarrier(StrictModule):
    """Protected source vertices and the numerical size schedule for refinement."""

    initial: TetMeshArrays
    construction_counters: tuple[tuple[str, int], ...]
    construction_work_units: int
    protected_vertices: tuple[np.ndarray, ...]
    size_goal: _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal | None


@final
class PlcRefinement(StrictModule):
    """Runs, accepted weights, and cumulative charges of native shape operations."""

    refinement: TetMeshRun[TetMeshRefineCounters] | None
    improvement: TetMeshRun[TetMeshImproveCounters] | None
    exudation: TetMeshRun[TetMeshExudeCounters] | None
    exudation_weights: np.ndarray | None
    spent: int


@final
class PlcOptimizedCarrier(StrictModule):
    """Final native snapshot and source/shape evidence before owner closure."""

    arrays: TetMeshArrays
    source_evidence: TetMeshSourceEvidence
    quality: TetMeshQuality
    unmet_cells: TetMeshUnmet
    metric_optimization: (
        _dependency_phydrax_meshing__tetra_metric.MetricRemeshingEvidence | None
    )
    spent: int


@final
class PlcPublishedCarrier(StrictModule):
    """Compact incidence and source-index ancestry of the published domain."""

    points: np.ndarray
    tetrahedra: np.ndarray
    faces: np.ndarray
    segments: np.ndarray
    input_vertices: np.ndarray
    used: np.ndarray
    published_weights: np.ndarray | None


@final
class PlcOrganization(StrictModule):
    """Region, boundary, label, and association tables in publication order."""

    zones: tuple[_dependency_phydrax_meshing__organization.MeshZone, ...]
    patches: tuple[_dependency_phydrax_meshing__organization.MeshPatch, ...]
    labels: tuple[_dependency_phydrax_meshing__organization.MeshLabel, ...]
    associations: tuple[association_state_owner.GeometryAssociation, ...]


def _validate_plc_carrier_seeds(
    carrier: PlcCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
) -> PlcSeededCarrier:
    """Validate the source prefix and every declared region or void seed."""
    state, complex_ = carrier.state, carrier.complex
    construction_counters = carrier.construction_counters
    construction_work_units = carrier.construction_work_units
    initial = state.arrays()
    if not np.array_equal(
        initial.points[: complex_.vertices.shape[0]], complex_.vertices
    ):
        raise ValueError(
            "PlcCarrier.state must retain the PLC vertex prefix; got changed source points"
        )
    if construction_work_units < 0:
        raise ValueError(
            "PlcCarrier.construction_work_units must be nonnegative; got a negative count"
        )
    seeds, seed_regions = region_seed_points(complex_, specification)
    seed_work = 0
    if seeds.shape[0]:
        seed_queries_before = scope.remaining().remaining_geometry_queries
        corners = initial.points[initial.tetrahedra]
        skin = initial.points[initial.faces]
        with scope.batch(
            seeds.shape[0] * (skin.shape[0] + 4 * corners.shape[0]),
            queries_upper=seeds.shape[0],
        ) as ticket:
            for index, (seed, expected) in enumerate(
                zip(seeds, seed_regions, strict=True)
            ):
                ticket.consume(skin.shape[0], geometry_queries=1)
                seed_work += skin.shape[0]
                _, features, status = point_triangle_locations(
                    np.repeat(seed[None, :], skin.shape[0], axis=0), skin
                )
                if np.any(status != MeshcoreStatus.OK) or np.any(features >= 0):
                    raise _leaf_phydrax_meshing__failures.MeshingFailure(
                        _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                        "PlcCarrier.region_seeds must avoid constrained facets; got a boundary seed",
                        stage=MeshingStageKind.VOLUME_FILL.value,
                        entity_ids=(index,),
                    )
                contained = np.ones((corners.shape[0],), dtype=np.bool_)
                for slot in range(4):
                    ticket.consume(corners.shape[0])
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
                    raise _leaf_phydrax_meshing__failures.MeshingFailure(
                        _leaf_phydrax_meshing__failures.MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                        "PlcCarrier.region_seeds must agree with declared region or void labels; got a contradictory carrier",
                        stage=MeshingStageKind.VOLUME_FILL.value,
                        entity_ids=(index,),
                    )
        construction_counters += (
            (
                "seed_geometry_queries",
                seed_queries_before - scope.remaining().remaining_geometry_queries,
            ),
            ("seed_host_row_visits", seed_work),
        )
    construction_work_units += seed_work
    return PlcSeededCarrier(initial, construction_counters, construction_work_units)


def _protect_plc_carrier(
    carrier: PlcCarrier,
    seeded: PlcSeededCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    /,
    *,
    scope: NativeExecutionScope,
) -> PlcProtectedCarrier:
    """Protect source points before scheduling native shape and size operations."""
    state, complex_ = carrier.state, carrier.complex
    initial = seeded.initial
    construction_counters = seeded.construction_counters
    construction_work_units = seeded.construction_work_units
    protected_vertices = [
        np.asarray(feature.scope.entity_ids, dtype=np.int64)
        for feature in specification.protected_features
        if feature.scope.entity_dimension == 0
    ]
    size_goal = _uniform_background_goal(specification, schedule, validity_policy)
    protection_work_start = state.work_units()
    state.set_work_limit(
        _native_volume_remaining_work(
            specification.limits, construction_work_units, scope=scope
        )
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
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "PlcCarrier.protected_vertices must fit the remaining work allowance; got resource exhaustion",
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
                        (f"native_memory:{name}", value)
                        for name, value in state.memory_counters().items()
                    ),
                ),
            ) from error
    protection_work = state.work_units() - protection_work_start
    construction_work_units += protection_work
    if protection_work:
        construction_counters += (("protected_vertex_work_units", protection_work),)
    return PlcProtectedCarrier(
        initial,
        construction_counters,
        construction_work_units,
        tuple(protected_vertices),
        size_goal,
    )


def _optimize_plc_carrier(
    carrier: PlcCarrier,
    protected: PlcProtectedCarrier,
    runs: PlcRefinement,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> PlcOptimizedCarrier:
    """Run the statistical size epoch and capture native completion evidence."""
    state, complex_ = carrier.state, carrier.complex
    size_goal = protected.size_goal
    construction_counters = protected.construction_counters
    refinement, improvement, exudation = runs.refinement, runs.improvement, runs.exudation
    spent = runs.spent
    source, source_id, source_revision = (
        carrier.source,
        carrier.source_id,
        carrier.source_revision,
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
            scope=scope,
            declared_source=source,
            source_id=source_id,
            source_revision=source_revision,
            initial_geometry=carrier.geometry,
        )
        if size_goal is not None
        else None
    )
    metric_optimization = metric_outcome.evidence if metric_outcome is not None else None
    if metric_optimization is not None:
        spent += metric_optimization.work_units
        if (
            metric_optimization.status
            is _dependency_phydrax_meshing__tetra_metric.MetricRemeshingStatus.RESOURCE_LIMIT
        ):
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
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
                        for name, value in (
                            () if refinement is None else refinement.counters.items()
                        )
                    ),
                    *(
                        (f"improvement:{name}", int(value))
                        for name, value in (
                            () if improvement is None else improvement.counters.items()
                        )
                    ),
                    *(
                        ()
                        if exudation is None
                        else (
                            (f"exudation:{name}", int(value))
                            for name, value in exudation.counters.items()
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
                        (f"metric:native_memory:{name}", value)
                        for name, value in (
                            ()
                            if metric_optimization.memory.native_counters is None
                            else metric_optimization.memory.native_counters.items()
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
    quality = state.quality(sliver_degrees=schedule.minimum_dihedral_degrees)
    return PlcOptimizedCarrier(
        arrays, source_evidence, quality, unmet_cells, metric_optimization, spent
    )


def _compact_plc_carrier(
    carrier: PlcCarrier,
    protected: PlcProtectedCarrier,
    optimized: PlcOptimizedCarrier,
    runs: PlcRefinement,
    /,
) -> PlcPublishedCarrier:
    """Verify immutable source entities and compact vertices incident to the domain."""
    complex_, input_triangles = carrier.complex, carrier.input_triangles
    arrays = optimized.arrays
    protected_vertices = protected.protected_vertices
    exudation_weights = runs.exudation_weights
    if complex_.boundary == "fixed":
        locked = np.unique(
            np.concatenate((complex_.polygon_vertices, complex_.segments.ravel()))
        )
        realized = np.unique(np.sort(arrays.faces, axis=1), axis=0)
        expected = np.unique(np.sort(input_triangles, axis=1), axis=0)
        if not np.array_equal(realized, expected) or not np.array_equal(
            arrays.points[locked], complex_.vertices[locked]
        ):
            raise compliance_failure(
                "PlcCarrier.fixed_boundary must retain immutable PLC vertices and triangles; got changed boundary data",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
    for identifiers in protected_vertices:
        if np.any(arrays.vertex_dimension[identifiers] < 0) or not np.array_equal(
            arrays.points[identifiers], complex_.vertices[identifiers]
        ):
            raise compliance_failure(
                "PlcCarrier.protected_vertices must retain source positions; got a moved or removed point",
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
    return PlcPublishedCarrier(
        points, tetrahedra, faces, segments, input_vertices, used, published_weights
    )


def _assemble_plc_volume(
    carrier: PlcCarrier,
    protected: PlcProtectedCarrier,
    runs: PlcRefinement,
    optimized: PlcOptimizedCarrier,
    published: PlcPublishedCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    /,
    *,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> VolumeConstruction:
    """Construct source fidelity, organization, and phase reports for publication."""
    complex_, source = carrier.complex, carrier.source
    source_id, source_revision, input_id = (
        carrier.source_id,
        carrier.source_revision,
        carrier.input_id,
    )
    input_triangles, input_polygons, plc_edges = (
        carrier.input_triangles,
        carrier.input_polygons,
        carrier.plc_edges,
    )
    construction_stage = carrier.construction_stage
    construction_counters = protected.construction_counters
    arrays, source_evidence, quality, unmet_cells = (
        optimized.arrays,
        optimized.source_evidence,
        optimized.quality,
        optimized.unmet_cells,
    )
    metric_optimization, spent = optimized.metric_optimization, optimized.spent
    refinement, improvement, exudation = runs.refinement, runs.improvement, runs.exudation
    points, tetrahedra = published.points, published.tetrahedra
    used, published_weights = published.used, published.published_weights
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
    mesh = _dependency_phydrax_meshing__canonical.canonicalize_cell_mesh(
        CellMesh.from_tetrahedra(points, tetrahedra, numeric_version=source_revision)
    )
    geometry = None
    if carrier.unit_cells is not None:
        with native_volume_coordinate_budget(specification.limits, scope=scope) as budget:
            geometry = unit_cell_geometry(
                carrier.unit_cells, mesh, vertex_rows=used, budget=budget
            )
    cell_regions = np.asarray(arrays.tetrahedron_regions, dtype=np.int64)
    with measure_phase(record_phase, "organization"):
        organization = _organization(
            mesh, cell_regions, published, arrays, carrier, specification, scope=scope
        )
    stages = (
        construction_stage,
        _dependency_phydrax_meshing__trace.MeshingStageReport(
            MeshingStageKind.OPTIMIZATION,
            _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED
            if (refinement is None or refinement.status == MeshcoreStatus.OK)
            and (improvement is None or improvement.status == MeshcoreStatus.OK)
            and (exudation is None or exudation.status == MeshcoreStatus.OK)
            and (metric_optimization is None or metric_optimization.converged)
            else _dependency_phydrax_meshing__trace.MeshingStageStatus.WARNING,
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
    construction = VolumeConstruction(
        mesh,
        cell_regions,
        domain(complex_, input_triangles, input_polygons, source_id),
        organization.zones,
        organization.patches,
        organization.labels,
        organization.associations,
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
        geometry=geometry,
    )
    return construction


def finalize_plc_volume(
    carrier: PlcCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
) -> VolumeConstruction:
    """Consume an admitted PLC carrier, including refusals before execution starts."""
    # Cleanup belongs outside input checking because the native resource has
    # already been acquired. The class pattern admits only a closable carrier.
    match carrier:
        case PlcCarrier():
            try:
                return _run_plc_finalization(
                    carrier,
                    specification,
                    schedule,
                    validity_policy=validity_policy,
                    record_phase=record_phase,
                    scope=scope,
                )
            finally:
                carrier.state.close()
        case _:
            return _run_plc_finalization(
                carrier,
                specification,
                schedule,
                validity_policy=validity_policy,
                record_phase=record_phase,
                scope=scope,
            )


@checked
def _run_plc_finalization(
    carrier: PlcCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
) -> VolumeConstruction:
    """Bind finishing to the producer allowance and publish ended execution evidence."""
    native_volume_checkpoint(
        specification.limits, MeshingStageKind.VOLUME_FILL, scope=scope
    )
    with native_volume_execution_budget(
        specification.limits,
        source_work_units=carrier.construction_work_units,
        mesh=carrier.state,
    ) as execution:
        construction = _finish_plc_volume(
            carrier,
            specification,
            schedule,
            validity_policy=validity_policy,
            record_phase=record_phase,
            scope=execution,
        )
    return bind_volume_execution(
        construction, execution.completed_evidence(), carrier.construction_work_units
    )


def _initial_plc_goal_admitted(
    carrier: PlcCarrier,
    protected: PlcProtectedCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[bool, int]:
    """Prove unchanged physical goals before scheduling shape mutations.

    Zero-deviation source admission is established by the native constructor.
    Native shape predicates and the statistical goal's declared reducers are
    independent of Delaunayhood. The metric epoch still enforces its quality
    floor and publishes its own completion evidence.
    """
    goal, source = protected.size_goal, carrier.source
    if goal is None or source is None:
        return False, 0
    snapshot = protected.initial
    count = snapshot.tetrahedra.shape[0]
    host_work = (
        64 * count
        + 16 * snapshot.points.shape[0]
        + source.faces.size
        + source.segments.size
    )
    host_bytes = 1024 * count + 256 * snapshot.points.shape[0]
    before = scope.remaining().remaining_work_units
    with scope.batch(host_work, bytes_upper=host_bytes) as ticket:
        ticket.consume(host_work, managed_bytes=host_bytes)
        exact = bool(
            np.all(source.face_tolerances == 0.0)
            and np.all(source.segment_tolerances == 0.0)
        )
        admitted = exact
    if admitted:
        with native_volume_coordinate_budget(specification.limits, scope=scope) as budget:
            admitted = not (
                _dependency_phydrax_meshing__tetra_metric.physical_size_compliance(
                    snapshot, goal, geometry=carrier.geometry, budget=budget
                )[2]
            )
    if not admitted:
        return False, before - scope.remaining().remaining_work_units
    radius, angle, determinant_failures = carrier.state.proposal_shape(
        snapshot.points,
        snapshot.tetrahedra,
        vertex_ids=np.arange(snapshot.points.shape[0], dtype=np.int64),
        minimum_relative_determinant=validity_policy.relative_determinant_floor,
        work_limit=_native_volume_remaining_work(
            specification.limits, protected.construction_work_units, scope=scope
        ),
    )
    return (
        radius <= schedule.radius_edge_bound
        and angle >= schedule.minimum_dihedral_degrees
        and determinant_failures == 0,
        before - scope.remaining().remaining_work_units,
    )


def _finish_plc_volume(
    carrier: PlcCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
) -> VolumeConstruction:
    """Finish and close a native carrier, then assemble its immutable evidence."""
    try:
        seeded = _validate_plc_carrier_seeds(carrier, specification, scope=scope)
        protected = _protect_plc_carrier(
            carrier, seeded, specification, schedule, validity_policy, scope=scope
        )
        native_volume_checkpoint(
            specification.limits, MeshingStageKind.VOLUME_FILL, scope=scope
        )
        admitted, admission_work = _initial_plc_goal_admitted(
            carrier, protected, specification, schedule, validity_policy, scope=scope
        )
        protected = replace(
            protected,
            construction_counters=(
                *protected.construction_counters,
                ("initial_goal_shape_certified", int(admitted)),
                ("initial_goal_certification_work_units", admission_work),
            ),
            construction_work_units=protected.construction_work_units + admission_work,
        )
        if admitted:
            runs = PlcRefinement(
                None, None, None, None, protected.construction_work_units
            )
        else:
            if carrier.geometry is not None:
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
                    "PlcCarrier.geometry must meet initial source and shape goals before native mutations; got unresolved immutable source coordinates",
                    stage=MeshingStageKind.OPTIMIZATION.value,
                )
            deferred = (
                (True,)
                if protected.size_goal is not None
                else _satisfied_boundary_size_targets(
                    specification, protected.initial.points, protected.initial.tetrahedra
                )
            )
            runs = _refine_and_improve(
                carrier.state,
                carrier.complex,
                specification,
                schedule,
                carrier.input_triangles,
                carrier.input_polygons,
                deferred,
                protected.initial.points.shape[0],
                protected.construction_work_units,
                validity_policy=validity_policy,
                scope=scope,
                record_phase=record_phase,
            )
        native_volume_checkpoint(
            specification.limits, MeshingStageKind.OPTIMIZATION, scope=scope
        )
        optimized = _optimize_plc_carrier(
            carrier,
            protected,
            runs,
            specification,
            schedule,
            scope=scope,
            record_phase=record_phase,
        )
    finally:
        carrier.state.close()
    native_volume_checkpoint(
        specification.limits, MeshingStageKind.CANONICALIZATION, scope=scope
    )
    published = _compact_plc_carrier(carrier, protected, optimized, runs)
    native_volume_checkpoint(
        specification.limits, MeshingStageKind.GEOMETRY_ASSOCIATION, scope=scope
    )
    return _assemble_plc_volume(
        carrier,
        protected,
        runs,
        optimized,
        published,
        specification,
        validity_policy,
        scope=scope,
        record_phase=record_phase,
    )


def _published_source_fidelity(
    complex_: PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
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
    native achieved bound (exactly zero for an rational carrier).
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
    "PlcCarrier",
    "PlcSourceRows",
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
