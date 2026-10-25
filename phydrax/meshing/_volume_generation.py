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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from time import monotonic
from typing import final, NoReturn

import equinox as eqx
import numpy as np

import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._exact_plc_geometry as _owner_phydrax_discretization__exact_plc_geometry
import phydrax.discretization._exact_rational_geometry as rational_geometry_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._mapped_reference_domain as mapped_domain_owner
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._association_state as association_state_owner
import phydrax.meshing._canonical as _dependency_phydrax_meshing__canonical
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization
import phydrax.meshing._result as result_owner
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
import phydrax.meshing._tetra_metric as _dependency_phydrax_meshing__tetra_metric
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    constrained_delaunay_2d,
    current_native_execution_scope,
    current_native_host_workspace,
    exact_orient2d,
    exact_orient3d,
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionEvidence,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    NativeScopeTicket,
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
    TetMeshBoundaryPolicy,
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


def _matches_canonical_declared_plc_domain(
    complex_: PiecewiseLinearComplex,
    source_id: str,
    supplied: domain_coverage_owner.PiecewiseLinearDomain,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    /,
) -> bool:
    """Re-earn a complete source declaration, never portable mesh-proof authority."""
    ticket = budget.ticket
    if not isinstance(ticket, (NativeBatchTicket, NativeScopeTicket)) or ticket.settled:
        raise ValueError("Canonical PLC declaration reconstruction requires its live paid native ledger.")
    with budget.live_storage() as storage:
        storage.set_bound(
            65536 + 256 * (
                complex_.vertices.shape[0] + complex_.polygon_vertices.size
                + supplied.facets.size + supplied.facet_regions.size
            )
        )
        with ticket.native_target():
            canonical = declared_plc_domain(complex_, source_id)
        budget.reserve(4 + len(canonical.region_ids))
        matches = (
            canonical.source_id == supplied.source_id
            and canonical.source_revision == supplied.source_revision
            and canonical.domain_id == supplied.domain_id
            and canonical.region_ids == supplied.region_ids
        )
        for name in ("vertices", "facets", "facet_regions"):
            actual, expected = getattr(supplied, name), getattr(canonical, name)
            budget.reserve(actual.size + expected.size, actual.nbytes + expected.nbytes + 256)
            matches = matches and (
                actual.dtype == expected.dtype and actual.shape == expected.shape
                and actual.tobytes(order="C") == expected.tobytes(order="C")
            )
        del canonical
        return matches


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
        source_domain: domain_coverage_owner.PiecewiseLinearDomain | None = None,
        source_result: result_owner.CellMeshingResult | None = None,
        publication_coverage: domain_coverage_owner.DomainCoverageCertificate | None = None,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget | None = None,
    ) -> None:
        if publication_coverage is not None and source_result is None:
            raise ValueError("Publication coverage requires the actual current PLC source result.")
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
        if source_domain is None and source_result is not None:
            certificate = source_result.certification
            if certificate is None or not isinstance(
                certificate.request.domain, domain_coverage_owner.PiecewiseLinearDomain
            ):
                raise ValueError("Prepared PLC source must retain its actual published declared domain.")
            source_domain = certificate.request.domain
        if source_domain is None:
            source_domain = domain(complex_, constraints.input_triangles, polygons, source_id)
        else:
            if budget is None or source_result is None:
                raise ValueError("Prepared PLC domain operands require their actual issued source and paid coordinate ledger.")
            budget.reserve(
                complex_.vertices.size + len(complex_.region_ids)
                + len(source_result.associations)
            )
            certificate = source_result.certification
            if (
                source_result.coordinate_contract is not coordinate_contract
                or certificate is None or not certificate.passed
                or certificate.request.domain is not source_domain
                or not source_result.associations
                or any(
                    association.source_id != source_id
                    or association.source_revision != source_revision
                    for association in source_result.associations
                )
                or certificate.request.cell_regions is None
                or certificate.embedding is None
                or not isinstance(certificate.coverage, domain_coverage_owner.DomainCoverageCertificate)
                or source_domain.source_id != source_id
                or source_domain.region_ids != complex_.region_ids
                or source_domain.vertices.shape != complex_.vertices.shape
                or source_domain.vertices.dtype != complex_.vertices.dtype
                or source_domain.vertices.tobytes(order="C") != complex_.vertices.tobytes(order="C")
            ):
                raise ValueError("Prepared PLC domain operands must retain the actual issued domain owner and all current source coordinates and regions.")
            region_bank = certificate.request.cell_regions
            budget.reserve(region_bank.size, region_bank.nbytes + 256)
            regions = np.asarray(region_bank, dtype=np.int64)
            if publication_coverage is not None:
                authenticated = domain_coverage_owner.authenticate_issued_domain_coverage(
                    source_result.mesh, source_result.geometry, source_domain,
                    regions, certificate.embedding, certificate.request.limits,
                    certificate.coverage, publication_coverage, budget=budget,
                )
                if authenticated is not publication_coverage:
                    raise ValueError("Prepared PLC domain operands require the actual full current issued publication coverage.")
            elif not domain_coverage_owner.is_issued_domain_coverage(
                source_result.mesh, source_result.geometry, source_domain,
                regions, certificate.embedding,
                certificate.request.limits, certificate.coverage, budget=budget,
            ) and not _matches_canonical_declared_plc_domain(
                complex_, source_id, source_domain, budget,
            ):
                raise ValueError("Prepared PLC domain operands require full current issued coverage or the complete canonical oriented source declaration.")
        transfer = _dependency_phydrax_meshing__association.PlcAssociationTransfer(
            source_domain,
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
    source_result: result_owner.CellMeshingResult | None = None,
    publication_coverage: domain_coverage_owner.DomainCoverageCertificate | None = None,
    coordinate_budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget | None = None,
) -> PreparedPlcSource:
    """Prepare rational source strata without recovering or refining a volume."""
    if publication_coverage is not None and source_result is None:
        raise ValueError("Publication coverage requires the actual current PLC source result.")
    source_domain = None
    if source_result is not None:
        certificate = source_result.certification
        if (
            coordinate_budget is None or certificate is None
            or not isinstance(certificate.request.domain, domain_coverage_owner.PiecewiseLinearDomain)
        ):
            raise ValueError("Canonical PLC source preparation requires its actual declared domain and paid ledger.")
        source_domain = certificate.request.domain
    ticket = None if coordinate_budget is None else coordinate_budget.ticket
    if isinstance(ticket, (NativeBatchTicket, NativeScopeTicket)) and not ticket.settled:
        ticket_root, source_root = ticket.scope, scope
        while ticket_root.parent is not None:
            ticket_root = ticket_root.parent
        while source_root.parent is not None:
            source_root = source_root.parent
        if ticket_root is not source_root:
            raise ValueError("PLC source preparation must use the actual current native request's coordinate ledger.")
        assert coordinate_budget is not None
        remaining_work = min(
            limits.maximum_work_units - source_work_units,
            coordinate_budget.maximum_work_units - coordinate_budget.work_units,
            ticket.remaining_work_units,
        )
        remaining_queries = min(
            limits.maximum_geometry_queries - source_geometry_queries,
            ticket.remaining_geometry_queries,
        )
        native_owner = ticket.native_target()
    else:
        remaining_work = _native_volume_remaining_work(limits, source_work_units, scope=scope)
        with scope.ticket_view() as incoming:
            remaining_queries = min(
                limits.maximum_geometry_queries - source_geometry_queries,
                incoming.remaining_geometry_queries,
            )
        native_owner = nullcontext()
    if remaining_work <= 0 or remaining_queries <= 0:
        _exhausted(
            MeshingStageKind.SOURCE_INSPECTION,
            "Source preparation consumed the source-renewal work or query allowance.",
            None,
        )
    with (
        native_owner,
        native_volume_execution_budget(
            limits, stage=MeshingStageKind.SOURCE_INSPECTION,
        ) as budget,
    ):
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
        source_domain=source_domain,
        source_result=source_result,
        publication_coverage=publication_coverage,
        budget=coordinate_budget,
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
    ``prepare_fixed_boundary_adaptation`` explicitly permits interior stellar
    preparation for a subsequent fixed-boundary adaptation request; the default
    generation schedule preserves the original carrier and entity capacities.
    """

    radius_edge_bound: float = eqx.field(static=True)
    minimum_dihedral_degrees: float = eqx.field(static=True)
    refinement_rounds: int = eqx.field(static=True)
    improvement_passes: int = eqx.field(static=True)
    exudation_passes: int = eqx.field(static=True)
    exudation_weight_fraction: float = eqx.field(static=True)
    metric_optimization_passes: int = eqx.field(static=True)
    prepare_fixed_boundary_adaptation: bool = eqx.field(static=True)
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
        prepare_fixed_boundary_adaptation: bool = False,
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
        if not isinstance(prepare_fixed_boundary_adaptation, (bool, np.bool_)):
            raise TypeError(
                "NativeVolumeSchedule.prepare_fixed_boundary_adaptation must be a bool"
            )
        fixed_adaptation = bool(prepare_fixed_boundary_adaptation)
        self.radius_edge_bound = ratio
        self.minimum_dihedral_degrees = angle
        self.refinement_rounds = rounds
        self.improvement_passes = passes
        self.exudation_passes = exudation
        self.exudation_weight_fraction = weight_fraction
        self.metric_optimization_passes = metric_passes
        self.prepare_fixed_boundary_adaptation = fixed_adaptation
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
                "prepare_fixed_boundary_adaptation": fixed_adaptation,
            }
        )


@dataclass(frozen=True, slots=True)
class VolumeConstruction:
    """Assembled constrained tetrahedral mesh and the evidence of its phases.

    ``validity_policy_id`` identifies the determinant floor certified by native
    shape inspection or executed improvement/exudation. Publication audits
    under that same policy. ``refinement`` and ``improvement`` contain only
    executed native batched runs; a guarded hard-goal-first epoch retains its
    actual proposal counts in ``metric_optimization`` and its soft-AIM outcome
    in ``aim_requested``, ``aim_achieved`` and ``unmet``. A native improvement
    run may perform mandatory floor repair before that epoch, without claiming
    that the angular AIM was met. ``exudation`` is ``None`` when not executed;
    its weights are bounded local evidence, not a global regularity certificate.
    """

    mesh: CellMesh
    cell_regions: np.ndarray
    domain: domain_coverage_owner.PiecewiseLinearDomain | mapped_domain_owner.MappedReferenceDomain
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
    aim_requested: tuple[tuple[str, float], ...] = ()
    aim_achieved: tuple[tuple[str, float], ...] = ()
    source_constraints: PlcSourceConstraints | None = None
    source_input_points: np.ndarray | None = None


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
    with scope.ticket_view() as incoming:
        return min(limits.maximum_work_units - spent, incoming.remaining_work_units)


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
        if total_work >= limits.maximum_work_units or (
            isinstance(error, MeshcoreError)
            and error.batch_requested_upper is not None
            and error.batch_refusal_evidence is not None
            and error.batch_requested_upper[0] > error.batch_refusal_evidence[0]
        ):
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
    *,
    root_maximum_work_units: int | None = None,
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
            if root_maximum_work_units is not None:
                requested["maximum_work_units"] = root_maximum_work_units
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
    root_maximum_work_units: int | None = None,
) -> Iterator[NativeExecutionScope]:
    """Open a cumulative request; only its root imports anonymous source totals.

    Nested source metadata describes a prefix already accounted by the root.
    Ended phase evidence identities are imported separately by the preparation ledger.
    """
    if root_maximum_work_units is not None:
        if current_native_execution_scope() is not None:
            raise ValueError("Mechanical work admission must own the execution root.")
        if not 0 < root_maximum_work_units <= limits.maximum_work_units:
            raise ValueError("Mechanical work admission must only lower the original work limit.")
        limits = _dependency_phydrax_meshing__contracts.MeshingLimits(
            maximum_vertices=limits.maximum_vertices,
            maximum_edges=limits.maximum_edges,
            maximum_faces=limits.maximum_faces,
            maximum_cells=limits.maximum_cells,
            maximum_connectivity_entries=limits.maximum_connectivity_entries,
            maximum_data_bytes=limits.maximum_data_bytes,
            maximum_work_units=root_maximum_work_units,
            maximum_cavity_cells=limits.maximum_cavity_cells,
            maximum_geometry_queries=limits.maximum_geometry_queries,
            maximum_scratch_bytes=limits.maximum_scratch_bytes,
            maximum_wall_seconds=limits.maximum_wall_seconds,
        )
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
            root_maximum_work_units=root_maximum_work_units,
        )
        raise wrapped from error


@checked
def require_native_preparation_allowance(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    receipt: NativeExecutionRecord,
    /,
) -> None:
    """Admit represented prior phases; native payload and host upper bounds stay distinct."""
    scope = current_native_execution_scope()
    if scope is None:
        receipt.require_valid()
    else:
        native_preparation_ledger(scope).authenticate_receipt(scope, receipt)
        scope.charge(work=25)
    peak_bytes_upper = 0
    peak_cavity = 0
    current: NativeExecutionRecord | None = receipt
    while current is not None:
        if scope is not None:
            scope.charge(work=5)
        managed_peak = current.managed_admission_peak_bytes_upper
        phase_peak = (
            int(np.asarray(managed_peak))
            if managed_peak is not None else
            int(np.asarray(current.memory_counters.peak_bytes))
            + int(np.asarray(current.host_storage_peak_bytes_upper))
        )
        # An observed simultaneous admission envelope is authoritative for
        # this ended phase. The fallback remains conservative, not an exact
        # boundary invented by summing independently measured peaks.
        peak_bytes_upper = max(peak_bytes_upper, phase_peak)
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
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    /,
) -> _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal | None:
    """The closed admitted profile for in-place statistical size optimization.

    Only physical hard obligations belong here. Numerical refinement and
    improvement AIMs are attempted separately without becoming admission floors.
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
    with scope.ticket_view() as incoming:
        work = min(limits.maximum_work_units, incoming.remaining_work_units)
        memory = min(limits.maximum_scratch_bytes, incoming.remaining_managed_bytes)
        ticket = scope.batch(
            work,
            queries_upper=min(
                limits.maximum_geometry_queries, incoming.remaining_geometry_queries
            ),
            bytes_upper=memory,
        )
    with ticket:
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


# Mandatory finishing after the metric epoch (source evidence, exact geometry,
# organization and publication certification) is charged per published cell.
# The guarded shape AIM search is soft and must never consume that reserve.
_FINISHING_WORK_PER_CELL = 512


def _guarded_aim_allowance(remaining: int, cells: int, /) -> int:
    """Soft AIM search allowance that keeps an earned hard state publishable.

    Evaluated after the mandatory AIM entry measurement with the epoch's
    remaining work and current cell count. Only half of the unreserved work is
    offered: every accepted AIM edit adds cells whose own finishing must still
    fit inside the request allowance.
    """
    return max(0, (remaining - _FINISHING_WORK_PER_CELL * cells) // 2)


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
    source_evidence: TetMeshSourceEvidence | None,
    source_id: str,
    source_revision: str,
    initial_geometry: _owner_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    aim_improvement_passes: int | None = None,
) -> _dependency_phydrax_meshing__tetra_metric.NativeTetraMetricOutcome:
    """Schedule the existing native metric epoch without reconstructing sources.

    ``aim_improvement_passes`` requests the guarded soft shape-AIM search after
    the hard goal; ``None`` runs the hard epoch alone.
    """
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
    shape_aims = (
        None
        if aim_improvement_passes is None
        else _dependency_phydrax_meshing__tetra_metric.TetraShapeAims(
            radius_edge_bound=schedule.radius_edge_bound,
            minimum_dihedral_degrees=schedule.minimum_dihedral_degrees,
            refinement_rounds=schedule.refinement_rounds,
            improvement_passes=aim_improvement_passes,
            search_allowance=_guarded_aim_allowance,
        )
    )
    with native_volume_coordinate_budget(limits, scope=scope) as budget:
        source_geometry = None
        if source_evidence is not None and source_evidence.achieved_bound > 0.0:
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

    def retain_coordinate_storage(_bytes_upper: int) -> NativeScopeTicket:
        return scope.ticket_view()

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
            shape_aims=shape_aims,
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


def _run_native_improvement(
    state: TetMesh3D,
    /,
    *,
    minimum_dihedral_degrees: float,
    minimum_relative_determinant: float,
    maximum_passes: int,
    work_limit: int,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> TetMeshRun[TetMeshImproveCounters]:
    state.measure_execution(record_phase is not None)
    try:
        improvement = state.improve(
            min_dihedral_degrees=minimum_dihedral_degrees,
            minimum_relative_determinant=minimum_relative_determinant,
            max_passes=maximum_passes,
            work_limit=work_limit,
        )
        if record_phase is not None:
            seconds, measured = state.execution_times()
            if measured[1]:
                record_phase(
                    NativeMeshingPhaseMeasurement(
                        "improvement", float(seconds[1]), improvement.counters.work_units
                    )
                )
        return improvement
    finally:
        state.measure_execution(False)


# The native hard-size phase refines for size only: this finite bound never
# rejects a representable cell, so no radius-edge criterion drives insertion.
# The requested radius-edge and dihedral AIMs run later in the guarded epoch.
_SIZE_ONLY_RADIUS_EDGE_BOUND = float(np.finfo(np.float64).max)


def _uniform_edge_bounds(
    goal: _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal, /
) -> tuple[float, float, float]:
    """Hard statistic interval and the longest admitted edge of a uniform goal.

    Returns ``(lower, upper, longest)``: every requested size statistic must
    lie in ``[lower, upper]``; ``longest`` is the declared maximum size plus
    its tolerance, or ``upper`` when no maximum is declared.
    """
    control, policy = goal.control, goal.policy
    target = control.target_size
    lower = max(0.0, target - policy.tolerance(target))
    upper = target + policy.tolerance(target)
    longest = (
        upper
        if control.maximum_size is None
        else control.maximum_size + policy.tolerance(control.maximum_size)
    )
    return lower, upper, longest


def _closest_point_triangle_distances(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, /
) -> np.ndarray:
    """Euclidean point-triangle distances (rows, Voronoi-region classification)."""
    ab, ac, ap = b - a, c - a, points - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = points - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = points - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = 1.0 / (va + vb + vc)
        closest = a + ab * (vb * denominator)[:, None] + ac * (vc * denominator)[:, None]
        on_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        closest = np.where(on_ab[:, None], a + ab * (d1 / (d1 - d3))[:, None], closest)
        on_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        closest = np.where(on_ac[:, None], a + ac * (d2 / (d2 - d6))[:, None], closest)
        on_bc = (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
        closest = np.where(
            on_bc[:, None],
            b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None],
            closest,
        )
    closest = np.where(((d1 <= 0.0) & (d2 <= 0.0))[:, None], a, closest)
    closest = np.where(((d3 >= 0.0) & (d4 <= d3))[:, None], b, closest)
    closest = np.where(((d6 >= 0.0) & (d5 <= d6))[:, None], c, closest)
    return np.linalg.norm(points - closest, axis=1)


def _segment_segment_distances(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray, /
) -> np.ndarray:
    """Euclidean distances between rows of nondegenerate segments."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denominator = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denominator > 0.0, np.clip((b * f - c * e) / denominator, 0.0, 1.0), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]), axis=1)


def _sweep_ranges(
    lower: np.ndarray, upper: np.ndarray, queries_lower: np.ndarray, queries_upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate row ranges of ``x`` intervals sorted by ``lower`` per query interval.

    No interval longer than the longest one can start before ``queries_lower``
    minus that length and still meet the query, so each range is exact up to
    the end test applied by :func:`_sweep_pairs`.
    """
    span = float(np.max(upper - lower)) if lower.size else 0.0
    start = np.searchsorted(lower, queries_lower - span, side="left")
    stop = np.searchsorted(lower, queries_upper, side="right")
    return start, np.maximum(stop, start)


def _sweep_pairs(
    start: np.ndarray, stop: np.ndarray, upper: np.ndarray, queries_lower: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Query and sorted-row index pairs of the ranges whose ends meet the query."""
    counts = stop - start
    queries = np.repeat(np.arange(start.shape[0]), counts)
    rows = np.repeat(start, counts) + (
        np.arange(queries.shape[0]) - np.repeat(np.cumsum(counts) - counts, counts)
    )
    keep = upper[rows] >= queries_lower[queries]
    return queries[keep], rows[keep]


def _source_features_separated(
    carrier: PlcCarrier,
    threshold: float,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[bool, int]:
    """Whether distinct non-incident source features are at least ``threshold`` apart.

    The candidates are vertex-vertex, vertex-polygon (vertex not on the
    polygon) and PLC-edge pairs without a shared vertex; for polyhedral
    sources these attain the minimum local feature size. Conforming Delaunay
    size refinement resolves every narrower gap by encroachment, so it may
    over-resolve a hard lower size statistic there. Returns the decision and
    the charged host work.
    """
    points = np.asarray(carrier.complex.vertices, dtype=np.float64)
    triangles = carrier.input_triangles
    polygons = carrier.input_polygons
    edges = carrier.plc_edges
    count = points.shape[0]
    corners = points[triangles]
    triangle_lower, triangle_upper = corners.min(axis=1), corners.max(axis=1)
    edge_lower = np.minimum(points[edges[:, 0]], points[edges[:, 1]])
    edge_upper = np.maximum(points[edges[:, 0]], points[edges[:, 1]])
    features = count + triangles.shape[0] + edges.shape[0]
    before = scope.remaining().remaining_work_units
    # Sorting and interval search of every feature box on the sweep axis.
    search_work = 4 * features * max(1, features.bit_length())
    with scope.batch(search_work, bytes_upper=96 * features) as ticket:
        ticket.consume(search_work, managed_bytes=96 * features)
        order = np.argsort(points[:, 0], kind="stable")
        x = points[order, 0]
        edge_order = np.argsort(edge_lower[:, 0], kind="stable")
        sorted_lower, sorted_upper = edge_lower[edge_order, 0], edge_upper[edge_order, 0]
        sweeps = (
            (_sweep_ranges(x, x, x, x + threshold), x, x),
            (
                _sweep_ranges(
                    x, x, triangle_lower[:, 0] - threshold, triangle_upper[:, 0] + threshold
                ),
                x,
                triangle_lower[:, 0] - threshold,
            ),
            (
                _sweep_ranges(
                    sorted_lower,
                    sorted_upper,
                    sorted_lower - threshold,
                    sorted_upper + threshold,
                ),
                sorted_upper,
                sorted_lower - threshold,
            ),
        )
        candidates = int(sum(int(np.sum(stop - start)) for (start, stop), _, _ in sweeps))
    # Every sweep candidate costs one remaining-axis filter, and each
    # surviving pair one exact Euclidean distance; the first gap narrower than
    # the threshold decides, so later feature classes are not examined.
    with scope.batch(9 * candidates, bytes_upper=64 * candidates) as ticket:
        (vertex_ranges, vertex_upper, vertex_queries), triangle_sweep, edge_sweep = sweeps

        def separated() -> bool:
            # Distinct vertices.
            first, second = _sweep_pairs(*vertex_ranges, vertex_upper, vertex_queries)
            keep = first < second
            first, second = order[first[keep]], order[second[keep]]
            ticket.consume(9 * first.shape[0], managed_bytes=64 * candidates)
            if first.size and float(
                np.min(np.linalg.norm(points[first] - points[second], axis=1))
            ) < threshold:
                return False
            # Vertices against polygon triangles that they do not bound.
            (start, stop), upper, queries = triangle_sweep
            rows, vertices = _sweep_pairs(start, stop, upper, queries)
            vertices = order[vertices]
            polygon_keys = np.unique(
                np.repeat(polygons, 3).astype(np.int64) * count + triangles.reshape(-1)
            )
            near = (
                np.all(points[vertices] >= triangle_lower[rows] - threshold, axis=1)
                & np.all(points[vertices] <= triangle_upper[rows] + threshold, axis=1)
                & ~np.isin(polygons[rows].astype(np.int64) * count + vertices, polygon_keys)
            )
            rows, vertices = rows[near], vertices[near]
            ticket.consume(near.shape[0] + 8 * rows.shape[0])
            if rows.size and float(
                np.min(
                    _closest_point_triangle_distances(
                        points[vertices], corners[rows, 0], corners[rows, 1], corners[rows, 2]
                    )
                )
            ) < threshold:
                return False
            # PLC edges without a shared vertex.
            (start, stop), upper, queries = edge_sweep
            first, second = _sweep_pairs(start, stop, upper, queries)
            keep = first < second
            first, second = edge_order[first[keep]], edge_order[second[keep]]
            near = (
                np.all(edge_lower[first] <= edge_upper[second] + threshold, axis=1)
                & np.all(edge_lower[second] <= edge_upper[first] + threshold, axis=1)
                & np.all(edges[first][:, :, None] != edges[second][:, None, :], axis=(1, 2))
            )
            first, second = first[near], second[near]
            ticket.consume(near.shape[0] + 8 * first.shape[0])
            return not first.size or float(
                np.min(
                    _segment_segment_distances(
                        points[edges[first, 0]],
                        points[edges[first, 1]],
                        points[edges[second, 0]],
                        points[edges[second, 1]],
                    )
                )
            ) >= threshold

        decision = separated()
    return decision, before - scope.remaining().remaining_work_units


def _hard_size_compliance(
    state: TetMesh3D,
    goal: _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[list[tuple[str, float]], list[str]]:
    """Measured size statistics and unmet hard size issues of the current carrier."""
    # The coordinate phase reserves the whole remaining allowance, so the
    # native snapshot is read before it is admitted.
    snapshot = state.arrays()
    with native_volume_coordinate_budget(limits, scope=scope) as budget:
        _, achieved, issues = (
            _dependency_phydrax_meshing__tetra_metric.physical_size_compliance(
                snapshot, goal, budget=budget
            )
        )
    return achieved, issues


def _orthogonal_grid_witnesses(
    source: TetMeshSourceComplex,
    tables: PlcCarrierTables,
    source_keys: tuple[tuple[int, ...], ...],
    vertex_keys: dict[int, tuple[int, ...]],
    /,
    *,
    scope: NativeExecutionScope,
) -> TetMeshSourceComplex:
    """Supply exactly representable source parameters for the native verifier."""
    count, original = tables.points.shape[0], source.witness_strata.shape[0]
    storage = current_native_host_workspace()
    if storage is None:
        raise RuntimeError("Grid source witnesses require their request workspace.")
    scratch = 4096 + 128 * count + 1024 * (source.faces.shape[0] + source.segments.shape[0])
    with scope.batch(scope.remaining().remaining_work_units, bytes_upper=scratch) as ticket:
        ticket.consume(4 * count, managed_bytes=scratch)
        strata = np.zeros((count,), dtype=np.int8)
        entities = np.full((count,), -1, dtype=np.int32)
        parameters = np.zeros((count, 2), dtype=np.float64)
        strata[:original] = source.witness_strata
        entities[:original] = source.witness_entities
        parameters[:original] = source.witness_parameters
        segment_rows: dict[int, list[int]] = {}
        for row, group in enumerate(np.asarray(source.segment_group_rows).tolist()):
            ticket.consume(1)
            segment_rows.setdefault(int(source.segment_ids[group]), []).append(row)
        segment_hints: dict[int, int] = {}
        for edge, label in zip(tables.segments.tolist(), tables.segment_sources.tolist(), strict=True):
            ticket.consume(2)
            for vertex in edge:
                if vertex >= original:
                    segment_hints.setdefault(vertex, label)
        source_segments = np.asarray(source.segments).tolist()
        for vertex, label in segment_hints.items():
            point = vertex_keys[vertex]
            for row in segment_rows.get(label, ()):
                ticket.consume(22)
                a, b = (source_keys[index] for index in source_segments[row])
                direction = tuple(y - x for x, y in zip(a, b, strict=True))
                offset = tuple(y - x for x, y in zip(a, point, strict=True))
                axis = next((index for index, value in enumerate(direction) if value), None)
                if axis is None or any(
                    value * direction[axis] != offset[axis] * slope
                    for value, slope in zip(offset, direction, strict=True)
                ):
                    continue
                fraction = Fraction(offset[axis], direction[axis])
                numeric = float(fraction)
                if not (0 <= fraction <= 1) or Fraction(numeric) != fraction:
                    continue
                strata[vertex], entities[vertex] = 1, row
                parameters[vertex, 0] = numeric
                break
        triangle_rows: dict[tuple[int, int, int], list[tuple[int, tuple[int, ...]]]] = {}
        for row, corners in enumerate(np.asarray(source.faces).tolist()):
            ticket.consume(12)
            points = tuple(source_keys[index] for index in corners)
            axes = tuple(axis for axis in range(3) if len({point[axis] for point in points}) == 1)
            if len(axes) != 1:
                continue
            axis = axes[0]
            label = int(source.face_ids[int(source.face_group_rows[row])])
            ticket.consume(7)
            u, v = (axis + 1) % 3, (axis + 2) % 3
            a, b, c = points
            bu, bv, cu, cv = b[u] - a[u], b[v] - a[v], c[u] - a[u], c[v] - a[v]
            frame = a[u], a[v], bu, bv, cu, cv, bu * cv - bv * cu
            triangle_rows.setdefault((label, axis, points[0][axis]), []).append((row, frame))
        hints: dict[int, tuple[int, int, int]] = {}
        for face, label in zip(tables.faces.tolist(), tables.face_sources.tolist(), strict=True):
            ticket.consume(6)
            points = tuple(vertex_keys[vertex] for vertex in face)
            axis = next(axis for axis in range(3) if len({point[axis] for point in points}) == 1)
            for vertex in face:
                if vertex >= original and vertex not in segment_hints:
                    hints.setdefault(vertex, (label, axis, points[0][axis]))
        for vertex, hint in hints.items():
            point = vertex_keys[vertex]
            axis = hint[1]
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for row, (au, av, bu, bv, cu, cv, denominator) in triangle_rows.get(hint, ()):
                ticket.consume(11)
                du, dv = point[u] - au, point[v] - av
                if denominator == 0:
                    continue
                first = Fraction(du * cv - dv * cu, denominator)
                second = Fraction(bu * dv - bv * du, denominator)
                if first < 0 or second < 0 or first + second > 1:
                    continue
                numeric = float(first), float(second)
                if Fraction(numeric[0]) != first or Fraction(numeric[1]) != second:
                    continue
                strata[vertex], entities[vertex] = 2, row
                parameters[vertex] = numeric
                break
        # A nonrepresentable parameter stays unclassified; it is not a
        # rounded substitute. The native constructor completes and checks it.
        declared = TetMeshSourceComplex(
            source.points, source.faces, source.face_group_rows, source.face_tolerances,
            source.segments, source.segment_group_rows, source.segment_tolerances,
            strata, entities, parameters, source.face_ids, source.segment_ids,
        )
        storage.retain_owner(declared)
        return declared


def _orthogonal_plc_size_grid(
    carrier: PlcCarrier,
    protected: PlcProtectedCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    goal: _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal,
    /,
    *,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> tuple[PlcCarrier | None, PlcProtectedCarrier]:
    """Construct a source-conforming dyadic grid of an orthogonal PLC.

    Checkerboard five-tetrahedron cells share identical face diagonals. Cells
    traversed by a declared body-diagonal curve use a face-conforming twelve
    tetrahedron center star, retaining that curve as two exact edges. Source
    recognition is not an acceptance proof: the native constructor and the
    ordinary hard metric, source, embedding and coverage checks all still run.
    Other geometry, nonrepresentable nodes and coarser requests retain native
    refinement; every performed preparation visit remains on this request.
    """
    source, tables, complex_ = carrier.source, carrier.tables, carrier.complex
    if (
        source is None or tables is None or complex_.boundary != "conforming"
        or complex_.geometry is not None
        or np.any(source.face_tolerances != 0.0)
        or np.any(source.segment_tolerances != 0.0)
    ):
        return None, protected
    limits = specification.limits
    if not _hard_size_compliance(carrier.state, goal, limits, scope=scope)[1]:
        return carrier, protected
    initial = carrier.state.arrays()
    parent_storage = current_native_host_workspace()
    if parent_storage is None:
        raise RuntimeError("Orthogonal PLC construction requires its request workspace.")
    before = scope.remaining().remaining_work_units
    adopted: PlcCarrier | None = None
    grid_cells = 0
    cone_cells = 0
    source_keys: tuple[tuple[int, ...], ...] = ()
    vertex_keys: dict[int, tuple[int, ...]] = {}
    corner_bits = tuple(product((0, 1), repeat=3))
    face_bits = (
        (0, -1, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        (0, 1, ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        (1, -1, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        (1, 1, ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        (2, -1, ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        (2, 1, ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )
    with measure_phase(record_phase, "refinement"), NativeHostStorageWorkspace(scope) as storage:
        storage.set_bound(
            16384 + 4096 * initial.points.shape[0]
            + 1024 * complex_.polygon_vertices.shape[0]
        )
        allowance = scope.remaining().remaining_work_units
        with scope.batch(allowance) as ticket:
            def charge(visits: int) -> None:
                ticket.consume(visits)

            def build() -> PlcCarrierTables | None:
                nonlocal grid_cells, cone_cells, source_keys, vertex_keys
                charge(18 * initial.points.shape[0])
                rational = tuple(
                    tuple(Fraction(float(value)) for value in row) for row in initial.points
                )
                origin = tuple(min(row[axis] for row in rational) for axis in range(3))
                differences = tuple(value - origin[axis] for row in rational for axis, value in enumerate(row))
                powers = tuple(
                    (value.numerator & -value.numerator).bit_length() - value.denominator.bit_length()
                    for value in differences if value
                )
                if not powers:
                    return None
                step = Fraction(2) ** min(powers)
                if Fraction(goal.control.target_size) > step:
                    return None
                lower, upper, longest = _uniform_edge_bounds(goal)
                edge_limit = min(upper, longest) if "p95" in goal.policy.target_statistics else longest
                while step > Fraction(upper) or 2 * step * step > Fraction(edge_limit) ** 2:
                    charge(4)
                    step /= 2
                if step < Fraction(lower):
                    return None

                def lattice(row: tuple[Fraction, ...]) -> tuple[int, ...] | None:
                    charge(9)
                    indices = tuple((value - base) / step for value, base in zip(row, origin, strict=True))
                    if any(value.denominator != 1 for value in indices):
                        return None
                    return tuple(value.numerator for value in indices)

                keys = tuple(lattice(row) for row in rational)
                if any(row is None for row in keys):
                    return None
                grid_keys = tuple(row for row in keys if row is not None)
                if len(set(grid_keys)) != len(grid_keys):
                    return None
                shape = tuple(max(row[axis] for row in grid_keys) for axis in range(3))
                potential = shape[0] * shape[1] * shape[2]
                if min(shape) <= 0 or 5 * potential > limits.maximum_cells:
                    return None
                needed = storage.bound + 4096 * potential
                if needed > storage.bound + scope.remaining().remaining_managed_bytes:
                    return None
                storage.set_bound(needed)
                coordinate_axes: list[tuple[float, ...]] = []
                for axis in range(3):
                    charge(6 * (2 * shape[axis] + 1))
                    coefficients = tuple(
                        origin[axis] + step * Fraction(index, 2)
                        for index in range(2 * shape[axis] + 1)
                    )
                    values = tuple(float(value) for value in coefficients)
                    if any(Fraction(value) != exact for value, exact in zip(values, coefficients, strict=True)):
                        return None
                    coordinate_axes.append(values)
                # (projected loop, facet label, positive/negative side, normal).
                facets: dict[
                    tuple[int, int],
                    list[tuple[tuple[tuple[int, int], ...], int, tuple[int, int], int]],
                ] = {}
                polygon_vertices = np.asarray(complex_.polygon_vertices)
                for polygon, label in enumerate(np.asarray(complex_.polygon_facets).tolist()):
                    rows = polygon_vertices[
                        int(complex_.polygon_offsets[polygon]):int(complex_.polygon_offsets[polygon + 1])
                    ]
                    loop = tuple(grid_keys[int(index)] for index in rows)
                    charge(10 * len(loop))
                    axes = tuple(axis for axis in range(3) if len({point[axis] for point in loop}) == 1)
                    if len(axes) != 1:
                        return None
                    axis = axes[0]
                    if any(
                        sum(a != b for a, b in zip(first, second, strict=True)) != 1
                        for first, second in zip(loop, (*loop[1:], loop[0]), strict=True)
                    ):
                        return None
                    u, v = (axis + 1) % 3, (axis + 2) % 3
                    projected = tuple((point[u], point[v]) for point in loop)
                    area = sum(a * d - b * c for (a, b), (c, d) in zip(
                        projected, (*projected[1:], projected[0]), strict=True
                    ))
                    if not area:
                        return None
                    sides = tuple(int(value) for value in complex_.facet_regions[label])
                    facets.setdefault((axis, loop[0][axis]), []).append(
                        (projected, label, (sides[0], sides[1]), 1 if area > 0 else -1)
                    )


                face_owners: dict[
                    tuple[int, int, int, int],
                    tuple[tuple[tuple[int, int], ...], int, tuple[int, int], int],
                ] = {}
                for (axis, plane), descriptions in facets.items():
                    for description in descriptions:
                        loop = description[0]
                        # Integer orthogonal edges meet each unit-face center
                        # at a half-integer ordinate. A scanline's ordered
                        # vertical-edge crossings therefore describe exactly
                        # the same even/odd membership as individual rays,
                        # without revisiting every edge for every face.
                        charge(6 * len(loop))
                        vertical = tuple(
                            (a, min(b, d), max(b, d))
                            for (a, b), (c, d) in zip(
                                loop, (*loop[1:], loop[0]), strict=True
                            )
                            if a == c
                        )
                        for second in range(min(row[1] for row in loop), max(row[1] for row in loop)):
                            crossings: list[int] = []
                            for first, lower, upper_bound in vertical:
                                charge(3)
                                if not lower <= second < upper_bound:
                                    continue
                                position = 0
                                while position < len(crossings):
                                    charge(1)
                                    if crossings[position] >= first:
                                        break
                                    position += 1
                                charge(1 + len(crossings) - position)
                                crossings.insert(position, first)
                            if len(crossings) % 2:
                                return None
                            for slot in range(0, len(crossings), 2):
                                charge(2)
                                for first in range(crossings[slot], crossings[slot + 1]):
                                    charge(6)
                                    key = axis, plane, first, second
                                    if key in face_owners:
                                        return None
                                    face_owners[key] = description
                occupied: dict[tuple[int, ...], int] = {}
                x_planes = tuple(sorted(plane for axis, plane in facets if axis == 0))
                for j, k in product(range(shape[1]), range(shape[2])):
                    charge(len(x_planes))
                    events = [
                        (plane, description[2], description[3])
                        for plane in x_planes
                        if (description := face_owners.get((0, plane, j, k))) is not None
                    ]
                    cursor, region = 0, -1
                    for i in range(shape[0] + 1):
                        charge(1)
                        while cursor < len(events) and events[cursor][0] == i:
                            _, sides, normal = events[cursor]
                            previous, following = (sides[1], sides[0]) if normal > 0 else sides
                            if region != previous:
                                return None
                            region = following
                            cursor += 1
                        if i < shape[0] and region >= 0:
                            occupied[i, j, k] = region
                    if region != -1 or cursor != len(events):
                        return None
                grid_cells = len(occupied)
                coned: set[tuple[int, ...]] = set()
                curves: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
                charge(3 * source.segments.shape[0])
                for row, group in zip(
                    np.asarray(source.segments).tolist(),
                    np.asarray(source.segment_group_rows).tolist(), strict=True,
                ):
                    start, end = grid_keys[int(row[0])], grid_keys[int(row[1])]
                    delta = tuple(b - a for a, b in zip(start, end, strict=True))
                    spans = tuple(abs(value) for value in delta if value)
                    if not spans or len(spans) not in (1, 3) or len(set(spans)) != 1:
                        return None
                    direction = tuple((value > 0) - (value < 0) for value in delta)
                    current = start
                    for _ in range(spans[0]):
                        charge(6)
                        following = tuple(a + b for a, b in zip(current, direction, strict=True))
                        a = tuple(2 * value for value in current)
                        b = tuple(2 * value for value in following)
                        label = int(source.segment_ids[int(group)])
                        if len(spans) == 3:
                            cell = tuple(min(x, y) for x, y in zip(current, following, strict=True))
                            if cell not in occupied:
                                return None
                            coned.add(cell)
                            middle = tuple(2 * value + 1 for value in cell)
                            curves.extend(((a, middle, label), (middle, b, label)))
                        else:
                            curves.append((a, b, label))
                        current = following
                cone_cells = len(coned)
                if 5 * len(occupied) + 7 * len(coned) > limits.maximum_cells:
                    return None
                points = list(initial.points)
                identities = {tuple(2 * value for value in row): index for index, row in enumerate(grid_keys)}
                used_initial: set[int] = set()

                def point(index: tuple[int, ...]) -> int:
                    charge(1)
                    found = identities.get(index)
                    if found is not None:
                        return found
                    charge(8)
                    numeric = tuple(coordinate_axes[axis][value] for axis, value in enumerate(index))
                    found = len(points)
                    identities[index] = found
                    points.append(np.asarray(numeric, dtype=np.float64))
                    return found

                templates = []
                for parity in (0, 1):
                    central = tuple(index for index, corner in enumerate(corner_bits) if sum(corner) % 2 == parity)
                    rows = [central]
                    for index, corner in enumerate(corner_bits):
                        if index in central:
                            continue
                        neighbors = tuple(index ^ mask for mask in (4, 2, 1))
                        rows.append((index, *neighbors))
                    charge(len(rows))
                    # The central determinant is (-1)^parity * 2; a corner
                    # star's determinant is product(1 - 2*bit). Translation
                    # and the positive dyadic scale preserve these signs.
                    templates.append(tuple(
                        row if (parity == 0 if index == 0 else sum(corner_bits[row[0]]) % 2 == 0)
                        else (row[1], row[0], *row[2:])
                        for index, row in enumerate(rows)
                    ))
                tetrahedron_count = 5 * len(occupied) + 7 * len(coned)
                tetrahedra = np.empty((tetrahedron_count, 4), dtype=np.int32)
                tetrahedron_entries = tetrahedra.flat
                regions = np.empty((tetrahedron_count,), dtype=np.int32)
                output_row = 0
                boundary: list[tuple[int, ...]] = []
                labels: list[int] = []
                for cell, region in sorted(occupied.items()):
                    charge(20)
                    corners = tuple(tuple(2 * (a + b) for a, b in zip(cell, bits, strict=True)) for bits in corner_bits)
                    nodes = tuple(point(corner) for corner in corners)
                    used_initial.update(node for node in nodes if node < initial.points.shape[0])
                    center = point(tuple(2 * value + 1 for value in cell)) if cell in coned else -1
                    if center < 0:
                        for row in templates[sum(cell) % 2]:
                            charge(5)
                            for column, index in enumerate(row):
                                tetrahedron_entries[4 * output_row + column] = nodes[index]
                            regions[output_row] = region
                            output_row += 1
                    for axis, side, bits in face_bits:
                        charge(3)
                        neighbor = tuple(value + (side if index == axis else 0) for index, value in enumerate(cell))
                        other = occupied.get(neighbor, -1)
                        boundary_label = None
                        if not (
                            other >= 0
                            and (other < region or (other == region and neighbor < cell))
                        ):
                            u, v = (axis + 1) % 3, (axis + 2) % 3
                            plane = cell[axis] + int(side > 0)
                            charge(1)
                            description = face_owners.get((axis, plane, cell[u], cell[v]))
                            if description is None:
                                if other != region:
                                    return None
                            elif set(description[2]) != {region, other}:
                                return None
                            else:
                                boundary_label = description[1]
                        # Five-tetrahedron cells already own their complete
                        # volume rows. Only a declared source face needs a
                        # separate quad; center stars still need every face.
                        if center < 0 and boundary_label is None:
                            continue
                        face = tuple(tuple(a + b for a, b in zip(cell, corner, strict=True)) for corner in bits)
                        quad = tuple(point(tuple(2 * value for value in corner)) for corner in face)
                        triangles = (
                            ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3]))
                            if sum(face[0]) % 2 == 0
                            else ((quad[1], quad[2], quad[3]), (quad[1], quad[3], quad[0]))
                        )
                        if center >= 0:
                            for triangle in triangles:
                                charge(5)
                                tetrahedron_entries[4 * output_row] = center
                                for column, vertex in enumerate(triangle, start=1):
                                    tetrahedron_entries[4 * output_row + column] = vertex
                                regions[output_row] = region
                                output_row += 1
                        if boundary_label is not None:
                            boundary.extend(triangles)
                            labels.extend((boundary_label,) * 2)
                if output_row != tetrahedron_count or len(used_initial) != initial.points.shape[0] or len(points) > limits.maximum_vertices:
                    return None
                segment_labels: dict[tuple[int, int], int] = {}
                for start, end, label in curves:
                    a, b = point(start), point(end)
                    edge = min(a, b), max(a, b)
                    if edge in segment_labels and segment_labels[edge] != label:
                        return None
                    segment_labels[edge] = label
                charge(len(identities) + 12 * source.points.shape[0])
                vertex_keys = {index: key for key, index in identities.items()}
                source_lattice = tuple(lattice(tuple(Fraction(float(value)) for value in row)) for row in np.asarray(source.points).tolist())
                if any(row is None for row in source_lattice):
                    return None
                source_keys = tuple(tuple(2 * value for value in row) for row in source_lattice if row is not None)
                # Volume rows and regions were written once into their final
                # typed banks. Only the remaining list-backed tables need a
                # conversion visit; no second volume-table copy is performed.
                charge(3 * len(points) + 4 * len(boundary) + 3 * len(segment_labels))
                return PlcCarrierTables(
                    np.asarray(points, dtype=np.float64),
                    tetrahedra,
                    regions,
                    np.asarray(boundary, dtype=np.int32).reshape(-1, 3),
                    np.asarray(labels, dtype=np.int32),
                    np.asarray(tuple(segment_labels), dtype=np.int32).reshape(-1, 2),
                    np.asarray(tuple(segment_labels.values()), dtype=np.int32),
                    tables.boundary_policy,
                    np.concatenate((tables.protection_radii, np.zeros(len(points) - tables.points.shape[0]))),
                )

            result = build()
        if result is not None:
            declared = _orthogonal_grid_witnesses(
                source, result, source_keys, vertex_keys, scope=scope
            )
            parent_storage.retain_owner((result, declared))
            storage.set_bound(16384)
            state = _plc_carrier_state(result, limits, declared)
            try:
                if protected.protected_vertices:
                    state.protect_vertices(np.unique(np.concatenate(protected.protected_vertices)))
            except BaseException:
                state.close()
                raise
            adopted = replace(carrier, state=state, source=declared, tables=result)
            carrier.state.close()
    work = before - scope.remaining().remaining_work_units
    return adopted, replace(
        protected,
        construction_work_units=protected.construction_work_units + work,
        construction_counters=(
            *protected.construction_counters,
            ("hard_size_grid_cells", grid_cells),
            ("hard_size_grid_curve_cells", cone_cells),
            ("hard_size_grid_adopted", int(adopted is not None)),
            ("hard_size_grid_work_units", work),
        ),
    )


def _refine_hard_size(
    carrier: PlcCarrier,
    protected: PlcProtectedCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    goal: _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal,
    /,
    *,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> tuple[TetMeshRun[TetMeshRefineCounters] | None, int]:
    """Earn the hard edge bounds by native size-only Delaunay refinement.

    A carrier that already meets every hard size bound is not refined: the
    circumradius criterion below is only sufficient for the longest edge and
    would split flat cells whose edges already comply. Otherwise a
    circumradius at most half the longest admitted edge bounds every edge of
    the refined mesh, so the declared maximum holds by construction. When a
    requested statistic still exceeds its upper bound, one further monotone
    refinement aims at the target size. Statistics below their lower bound
    are left to the hard metric epoch, which also certifies the result.
    Returns ``None`` as the run when no refinement was needed.
    """
    limits = specification.limits
    state = carrier.state
    spent = protected.construction_work_units
    if not _hard_size_compliance(state, goal, limits, scope=scope)[1]:
        return None, spent
    _, upper, longest = _uniform_edge_bounds(goal)
    target = goal.control.target_size
    statistics = tuple(f":{name}_edge" for name in goal.policy.target_statistics)
    radius = 0.5 * longest
    insertions = limits.maximum_vertices - protected.initial.points.shape[0]
    total: TetMeshRefineCounters | None = None
    status = MeshcoreStatus.OK
    for correction in (False, True):

        def uniform(points: np.ndarray, /, size: float = radius) -> np.ndarray:
            return np.full((points.shape[0],), size, dtype=np.float64)

        run = state.refine(
            radius_edge_bound=_SIZE_ONLY_RADIUS_EDGE_BOUND,
            sizes=uniform,
            max_insertions=insertions,
            work_limit=_native_volume_remaining_work(limits, spent, scope=scope),
            max_rounds=1,
            record_native_phase=_native_phase_recorder(record_phase),
        )
        if run.status == MeshcoreStatus.CAPACITY_EXCEEDED:
            _exhausted(
                MeshingStageKind.VOLUME_FILL,
                "Native hard-size refinement exceeded its declared budget.",
                run,
            )
        counters = run.counters
        spent += counters.work_units
        insertions -= (
            counters.circumcenter_insertions
            + counters.subfacet_splits
            + counters.subsegment_splits
        )
        total = counters if total is None else total.accumulated(counters)
        status = run.status
        if correction or insertions <= 0:
            break
        achieved = _hard_size_compliance(state, goal, limits, scope=scope)[0]
        measured = max(
            (value for name, value in achieved if name.endswith(statistics)), default=0.0
        )
        if measured <= upper:
            break
        radius *= target / measured
    if total is None:
        raise RuntimeError("The native hard-size phase must execute one refinement run.")
    return TetMeshRun(status, total), spent


def _trial_hard_size(
    carrier: PlcCarrier,
    protected: PlcProtectedCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    goal: _dependency_phydrax_meshing__tetra_metric.UniformSizeRemeshingGoal,
    /,
    *,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> tuple[
    PlcCarrier | None, TetMeshRun[TetMeshRefineCounters] | None, PlcProtectedCarrier
]:
    """Run the native hard-size phase of a narrow source on an independent replica.

    The replica is the same recovered carrier with the same protected
    vertices. Its refinement is adopted only when the measured carrier meets
    every hard size bound; otherwise it is discarded and the untouched carrier
    keeps the metric route. Every unit spent by the trial stays charged and is
    reported. Returns the adopted carrier (``None`` when discarded), its
    refinement run, and the protected record with the trial evidence.
    """
    tables = carrier.tables
    if tables is None:
        return None, None, protected
    limits = specification.limits
    before = scope.remaining().remaining_work_units
    replica = _plc_carrier_state(tables, limits, carrier.source)
    trial: PlcCarrier | None = None
    refinement: TetMeshRun[TetMeshRefineCounters] | None = None
    try:
        replica.set_work_limit(
            _native_volume_remaining_work(
                limits, protected.construction_work_units, scope=scope
            )
        )
        if protected.protected_vertices:
            replica.protect_vertices(
                np.unique(np.concatenate(protected.protected_vertices))
            )
        candidate = replace(carrier, state=replica)
        refinement, _ = _refine_hard_size(
            candidate,
            protected,
            specification,
            goal,
            scope=scope,
            record_phase=record_phase,
        )
        if (
            refinement is not None
            and not _hard_size_compliance(replica, goal, limits, scope=scope)[1]
        ):
            trial = candidate
    finally:
        if trial is None:
            replica.close()
    work = before - scope.remaining().remaining_work_units
    protected = replace(
        protected,
        construction_counters=(
            *protected.construction_counters,
            ("hard_size_trial_work_units", work),
            ("hard_size_trial_adopted", int(trial is not None)),
        ),
        construction_work_units=protected.construction_work_units + work,
    )
    return trial, refinement if trial is not None else None, protected


def _repair_plc_validity(
    carrier: PlcCarrier,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    refinement: TetMeshRun[TetMeshRefineCounters] | None,
    spent: int,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy,
    scope: NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> PlcRefinement:
    """Repair the mandatory determinant floor before metric-source preparation.

    This is not an angular AIM run: the original requested angle is attempted
    later by the guarded epoch. Passes consumed here come out of the same
    improvement allowance, and all native work remains in the owning scope.
    ``refinement`` is the preceding native hard-size phase, if any.
    """
    improvement = _run_native_improvement(
        carrier.state,
        minimum_dihedral_degrees=0.0,
        minimum_relative_determinant=validity_policy.relative_determinant_floor,
        maximum_passes=schedule.improvement_passes,
        work_limit=_native_volume_remaining_work(specification.limits, spent, scope=scope),
        record_phase=record_phase,
    )
    if improvement.status == MeshcoreStatus.CAPACITY_EXCEEDED:
        _exhausted(
            MeshingStageKind.OPTIMIZATION,
            "Mandatory tetrahedral validity repair exceeded its declared budget.",
            improvement,
        )
    return PlcRefinement(
        refinement, improvement, None, None, spent + improvement.counters.work_units
    )


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
    improvement = _run_native_improvement(
        state,
        minimum_dihedral_degrees=schedule.minimum_dihedral_degrees,
        minimum_relative_determinant=validity_policy.relative_determinant_floor,
        maximum_passes=schedule.improvement_passes,
        work_limit=_native_volume_remaining_work(limits, spent, scope=scope),
        record_phase=record_phase,
    )
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
    source_points = (
        complex_.vertices
        if complex_.geometry is None
        else np.asarray(complex_.geometry.coordinates, dtype=np.float64)
    )
    source_direction = np.zeros_like(along)
    source_direction[on_curve] = (
        source_points[plc_edges[edge_sources[on_curve], 1]]
        - source_points[plc_edges[edge_sources[on_curve], 0]]
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
    source_points = (
        complex_.vertices
        if complex_.geometry is None
        else np.asarray(complex_.geometry.coordinates, dtype=np.float64)
    )
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
        # Native input rows retain their declared identity through the exact
        # coefficient lift. Compare the published physical representative with
        # that authored source row, never with the former execution view.
        if original >= 0 and np.array_equal(points[vertex], source_points[original]):
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

    execution = current_native_execution_scope()
    if execution is not None:
        # Index conversion, polygon/facet gathers, region-pair gather and mask.
        execution.charge(work=5 * input_polygons.size)
    polygons = input_polygons.astype(np.int64)
    incidence = complex_.facet_regions[complex_.polygon_facets[polygons]]
    kept = incidence[:, 0] != incidence[:, 1]
    if execution is not None:
        execution.charge(work=kept.size + len(complex_.region_ids))
        kept_count = int(np.count_nonzero(kept))
        # Actual retained facets own output copies, range/finite reductions,
        # triangle corner/projection banks around native exact predicates, and
        # streaming array fingerprints. Predicate queries are native counters,
        # not evaluations of a coordinate source.
        execution.charge(
            work=120 * kept_count + 3 * complex_.vertices.size
            + 2 * len(complex_.region_ids)
            + sum(map(len, complex_.region_ids)) + len(source_id) + 512
        )
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
    source_domain: mapped_domain_owner.MappedReferenceDomain | None = None,
    scope: NativeExecutionScope,
) -> VolumeConstruction:
    """Recover and finish under one source allowance, including initialization.

    ``validity_policy`` is the cell validity policy of the publishing audit; the
    native improvement repairs or reports every cell below its determinant floor.
    """
    if complex_.geometry is not None and source_domain is None:
        raise ValueError(
            "generate_plc_volume.source_domain must declare the physical mapped source of exact PLC geometry; got no mapped reference domain"
        )
    if source_domain is not None and (
        source_domain.source_id != source_id
        or source_domain.source_revision != source_revision
        or source_domain.region_ids != complex_.region_ids
    ):
        raise ValueError(
            "generate_plc_volume.source_domain must retain the source, revision, and material identities; got a different mapped source"
        )

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
    if source_domain is not None:
        construction = replace(construction, domain=source_domain)
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


def _extend_plc_unclassified_vertices(
    source: TetMeshSourceComplex,
    count: int,
    /,
    *,
    scope: NativeExecutionScope,
) -> TetMeshSourceComplex:
    """Append unclassified rows; the native constructor must certify new points."""
    count = nonnegative_integer(count, "additional_vertices")
    if not count:
        return source
    storage = current_native_host_workspace()
    if storage is None:
        raise RuntimeError("PLC vertex extension requires the active native host-storage owner.")
    previous = source.witness_strata.size
    total = previous + count
    work, scratch = 4 * total, 21 * total
    with scope.batch(work, bytes_upper=scratch) as ticket:
        ticket.consume(work, managed_bytes=scratch)
        strata = np.zeros(total, dtype=np.int8)
        entities = np.full(total, -1, dtype=np.int32)
        parameters = np.zeros((total, 2), dtype=np.float64)
        strata[:previous] = source.witness_strata
        entities[:previous] = source.witness_entities
        parameters[:previous] = source.witness_parameters
        extended = replace(
            source,
            witness_strata=strata,
            witness_entities=entities,
            witness_parameters=parameters,
        )
    storage.retain_owner(extended)
    return extended


def _fixed_boundary_interior_stars(
    tables: PlcCarrierTables,
    source: TetMeshSourceComplex,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[PlcCarrierTables, TetMeshSourceComplex, int, int]:
    """Give edge-locked fixed cells interior edges without touching source facets.

    A represented interior point replaces one parent by its four positive
    vertex-replacement tetrahedra. All old constrained triangles, segments,
    labels and source banks remain identical. This optional construction leaves
    the original valid source intact when its extra vertices/cells cannot fit.
    The native constructor validates any adopted partition and source witnesses.
    """
    if tables.boundary_policy != "fixed":
        return tables, source, 0, 0
    if (
        tables.points.shape[0] >= limits.maximum_vertices
        or tables.tetrahedra.shape[0] + 3 > limits.maximum_cells
        or limits.maximum_cavity_cells < 5
    ):
        return tables, source, 0, 0
    before = scope.remaining().remaining_work_units
    cells, faces, segments = tables.tetrahedra, tables.faces, tables.segments
    count, face_count, segment_count = cells.shape[0], faces.shape[0], segments.shape[0]
    work = 18 * count + 9 * face_count + 2 * segment_count
    scratch = 128 * (count + face_count + segment_count)
    with scope.batch(work, bytes_upper=scratch) as ticket:
        ticket.consume(work, managed_bytes=scratch)
        pairs = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
        cell_edges = np.sort(cells[:, pairs], axis=2)
        face_pairs = np.asarray(((0, 1), (0, 2), (1, 2)))
        protected = np.concatenate((
            np.sort(faces[:, face_pairs], axis=2).reshape((-1, 2)),
            np.sort(segments, axis=1),
        ))
        stride = tables.points.shape[0]
        protected_codes = np.unique(protected[:, 0] * stride + protected[:, 1])
        edge_codes = cell_edges[:, :, 0] * stride + cell_edges[:, :, 1]
        locked = np.flatnonzero(np.all(np.isin(edge_codes, protected_codes), axis=1))
        del cell_edges, protected, protected_codes, edge_codes
    selected = locked.size
    if not selected:
        return tables, source, 0, before - scope.remaining().remaining_work_units
    vertices, target_cells = tables.points.shape[0] + selected, count + 3 * selected
    if vertices > limits.maximum_vertices or target_cells > limits.maximum_cells:
        return tables, source, 0, before - scope.remaining().remaining_work_units
    work = 6 * tables.points.shape[0] + 8 * count + 64 * selected
    scratch = 64 * tables.points.shape[0] + 128 * count + 1024 * selected
    with scope.batch(work, bytes_upper=scratch, cavity_upper=5) as ticket:
        ticket.consume(work, managed_bytes=scratch, cavity_cells=5)
        corners = tables.points[cells[locked]]
        # Scaling before summation avoids overflow of a finite convex average.
        centers = np.sum(corners * 0.25, axis=1)
        points = np.concatenate((tables.points, centers))
        children = np.repeat(cells[locked], 4, axis=0)
        children[np.arange(4 * selected), np.tile(np.arange(4), selected)] = np.repeat(
            tables.points.shape[0] + np.arange(selected), 4
        )
        child_points = points[children]
        parent_signs = exact_orient3d(
            corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
        )
        child_signs = exact_orient3d(
            child_points[:, 0], child_points[:, 1], child_points[:, 2], child_points[:, 3]
        )
        if np.any(parent_signs == 0) or np.any(child_signs != np.repeat(parent_signs, 4)):
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Fixed-boundary construction could not certify a strictly interior represented centroid.",
                stage="fixed-boundary-interior-stars",
            )
        keep = np.ones(count, dtype=np.bool_)
        keep[locked] = False
        updated = PlcCarrierTables(
            points,
            np.concatenate((cells[keep], children)),
            np.concatenate((
                tables.tetrahedron_regions[keep],
                np.repeat(tables.tetrahedron_regions[locked], 4),
            )),
            tables.faces, tables.face_sources, tables.segments, tables.segment_sources,
            tables.boundary_policy,
            np.concatenate((tables.protection_radii, np.zeros(selected))),
        )
        declared = _extend_plc_unclassified_vertices(source, selected, scope=scope)
    storage = current_native_host_workspace()
    if storage is None:
        raise RuntimeError("PLC interior construction requires the active native host-storage owner.")
    storage.retain_owner(updated)
    return updated, declared, int(selected), before - scope.remaining().remaining_work_units


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
    tables = PlcCarrierTables(
        recovery.points,
        recovery.tetrahedra,
        recovery.tetrahedron_regions,
        recovery.faces,
        recovery.face_sources,
        segments,
        segment_sources,
        complex_.boundary,
        recovery.protection_radii,
    )
    interior_stars, interior_work = 0, 0
    if complex_.boundary == "fixed" and schedule.prepare_fixed_boundary_adaptation:
        with measure_phase(record_phase, "construction"):
            tables, declared, interior_stars, interior_work = _fixed_boundary_interior_stars(
                tables, declared, limits, scope=scope
            )
    if interior_work:
        construction_counters += (
            ("fixed_boundary_interior_stars", interior_stars),
            ("fixed_boundary_interior_star_work_units", interior_work),
        )
    state = _plc_carrier_state(tables, limits, declared)
    construction_id = canonical_fingerprint(
        {"kind": "plc-recovery", "input": input_id, "counters": construction_counters}
    )
    stage = _dependency_phydrax_meshing__trace.MeshingStageReport(
        MeshingStageKind.VOLUME_FILL,
        _dependency_phydrax_meshing__trace.MeshingStageStatus.PASSED,
        input_ids=(input_id,),
        output_ids=(construction_id,),
        created_count=tables.tetrahedra.shape[0],
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
        source_work_units + recovery.counters.work_units + interior_work,
        declared,
        tables=tables,
        source_constraints=PlcSourceConstraints(
            points=np.asarray(complex_.vertices, dtype=np.float64),
            plc_edges=recovery.plc_edges,
            input_triangles=recovery.input_triangles,
            input_polygons=recovery.input_polygons,
            counters=recovery.counters,
            memory_counters=recovery.memory_counters,
        ),
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
    goal = _uniform_background_goal(specification, validity_policy)
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
        source_constraints=constraints,
    )


@final
class PlcCarrierTables(StrictModule):
    """Native constructor tables of an unpublished PLC carrier."""

    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    boundary_policy: TetMeshBoundaryPolicy
    protection_radii: np.ndarray


def _plc_carrier_state(
    tables: PlcCarrierTables,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    source: TetMeshSourceComplex | None,
    /,
) -> TetMesh3D:
    """Construct the native carrier of recovered tables under the request caps."""
    return TetMesh3D(
        tables.points,
        tables.tetrahedra,
        tables.tetrahedron_regions,
        tables.faces,
        tables.face_sources,
        tables.segments,
        tables.segment_sources,
        boundary_policy=tables.boundary_policy,
        protection_radii=tables.protection_radii,
        max_vertices=limits.maximum_vertices,
        max_tetrahedra=limits.maximum_cells,
        max_scratch_bytes=limits.maximum_scratch_bytes,
        source=source,
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
    tables: PlcCarrierTables | None = None
    source_constraints: PlcSourceConstraints | None = None


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
    aim_evidence: (
        _dependency_phydrax_meshing__tetra_metric.NativeTetraAimEvidence | None
    ) = None


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
    size_goal = _uniform_background_goal(specification, validity_policy)
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
    hard_goal_first: bool,
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
    # Ordinary improvement passes already spent on mandatory validity repair
    # come out of the requested AIM improvement passes.
    aim_improvement_passes = (
        max(
            0,
            schedule.improvement_passes
            - (0 if improvement is None else improvement.counters.passes),
        )
        if hard_goal_first
        else None
    )
    # A zero-deviation native source has no bounded-coordinate lift to build:
    # the native constructor and each commit enforce its exact source rows.
    # Do not re-run the full source witness audit merely to discover that its
    # bound is zero. The final published state still issues that full proof.
    source_evidence = (
        state.source_evidence()
        if size_goal is not None
        and source is not None
        and (
            np.any(source.face_tolerances > 0.0)
            or np.any(source.segment_tolerances > 0.0)
        )
        else None
    )
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
            source_evidence=source_evidence,
            source_id=source_id,
            source_revision=source_revision,
            initial_geometry=carrier.geometry,
            aim_improvement_passes=aim_improvement_passes,
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
                    ("metric:minimum_metric_quality", metric_optimization.minimum_metric_quality),
                    ("metric:out_of_range_edges", metric_optimization.out_of_range_edges),
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
    # No committed native transaction means the same source witness bank still
    # proves this exact carrier. Rejected proposals are not successor sources.
    # A changed hard or AIM state must issue fresh native source evidence.
    if source_evidence is None or (
        metric_optimization is not None
        and (
            metric_optimization.splits
            or metric_optimization.collapses
            or metric_optimization.flips
            or metric_optimization.relocations
            or metric_optimization.compound_splits
            or metric_optimization.expanded_insertions
            or (
                metric_outcome is not None
                and metric_outcome.aim_evidence is not None
                and metric_outcome.aim_evidence.accepted_operations
            )
        )
    ):
        source_evidence = state.source_evidence()
    quality = state.quality(sliver_degrees=schedule.minimum_dihedral_degrees)
    aim_evidence = metric_outcome.aim_evidence if metric_outcome is not None else None
    if (
        hard_goal_first
        and metric_optimization is not None
        and metric_optimization.converged
        and aim_evidence is None
    ):
        raise RuntimeError(
            "The hard-goal-first epoch must return the requested guarded AIM outcome after earning the hard goal."
        )
    if aim_evidence is not None:
        if not aim_evidence.measurement_complete:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "The guarded AIM phase could not finish measuring its accepted state within the original allowance.",
                stage=MeshingStageKind.OPTIMIZATION.value,
                requested=(("maximum_work_units", specification.limits.maximum_work_units),),
                achieved=(
                    ("work_units", spent),
                    ("aim:work_units", aim_evidence.work_units),
                    ("aim:attempts", aim_evidence.attempts),
                    ("aim:measurement_complete", 0.0),
                ),
            )
        if aim_evidence.unmet is None:
            raise RuntimeError("A complete guarded AIM measurement must include its current unmet bank.")
        # The guarded epoch owns a complete bank for the current accepted
        # topology. Never publish stale native unmet rows from an earlier run.
        unmet_cells = aim_evidence.unmet
    return PlcOptimizedCarrier(
        arrays, source_evidence, quality, unmet_cells, metric_optimization, spent,
        aim_evidence,
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


def _authored_plc_geometry(
    complex_: PiecewiseLinearComplex,
    declared: TetMeshSourceComplex,
    evidence: TetMeshSourceEvidence,
    arrays: TetMeshArrays,
    used: np.ndarray,
    mesh: CellMesh,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _owner_phydrax_discretization__cell_geometry.CellGeometrySpec:
    """Lift native row witnesses to the independently authored exact source.

    Input vertices retain their declared coefficient rows. A constructed
    constrained vertex uses its exact barycentric coordinates in the native
    source entity, not rounded reported parameters. Applying those weights to
    the authored rows supplies the physical vertex. Body vertices keep their
    literal physical coordinates. Publication independently proves the resulting
    cell maps, their embedding, and source-domain coverage.
    """
    source_geometry = complex_.geometry
    if source_geometry is None or not isinstance(
        source_geometry.exact_source,
        rational_geometry_owner.ExactRationalCellGeometrySource,
    ):
        raise TypeError(
            "authored_plc_geometry.source must retain canonical exact rational coefficient actions; got another geometry source"
        )
    source = source_geometry.exact_source
    if source.parent.exact_source is not None:
        raise ValueError(
            "authored_plc_geometry.parent must retain the declared affine coefficient bank; got nested coordinate authority"
        )
    original_actions = source.action_rows()
    frame = np.asarray(source.parent.coordinates, dtype=np.float64)
    maximum_bits = 16 * (source.maximum_bits + 1078)
    bank_bytes, scratch_bytes = (
        _owner_phydrax_discretization__exact_plc_geometry.plc_storage_bounds(
            used.size + frame.shape[0], maximum_bits
        )
    )
    budget.reserve(0, bank_bytes + scratch_bytes)
    parent_points = np.concatenate((frame, arrays.points[used]), axis=0)
    parent = _owner_phydrax_discretization__cell_geometry.CellGeometrySpec.affine(
        CellMesh.from_tetrahedra(
            parent_points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
        ),
        budget=budget,
    )
    actions = []
    for row, vertex in enumerate(used.tolist()):
        budget.reserve(1)
        if vertex < complex_.vertices.shape[0]:
            actions.append(original_actions[vertex])
            continue
        stratum, entity = (
            int(evidence.witness_strata[vertex]),
            int(evidence.witness_entities[vertex]),
        )
        if stratum == 0:
            if arrays.vertex_dimension[vertex] != 3:
                raise ValueError(
                    "authored_plc_geometry.boundary must have an issued native source row; got an unconstrained boundary vertex"
                )
            actions.append(((frame.shape[0] + row, 1, 1),))
            continue
        if stratum not in (1, 2):
            raise ValueError(
                f"authored_plc_geometry.stratum must name a segment or facet; got {stratum}"
            )
        vertices = declared.segments[entity] if stratum == 1 else declared.faces[entity]
        # A segment/triangle Gram solve has dimension at most two. The
        # preadmitted fixed bound includes conversion, membership and solve;
        # subsequent sparse products are charged for each actual entry.
        budget.reserve(256)
        carrier = tuple(Fraction(float(value)) for value in arrays.points[vertex])
        corners = tuple(
            tuple(Fraction(float(value)) for value in declared.points[index])
            for index in vertices.tolist()
        )
        if not _owner_phydrax_discretization__exact_plc_geometry.in_entity(
            carrier, corners
        ):
            raise ValueError(
                "authored_plc_geometry.witness must lie exactly in its native source row; got an off-entity carrier"
            )
        parameters = _owner_phydrax_discretization__exact_plc_geometry._row_parameters(
            carrier, corners
        )
        weights = (1 - sum(parameters, Fraction(0)), *parameters)
        combined: dict[int, Fraction] = {}
        for index, weight in zip(vertices.tolist(), weights, strict=True):
            for position, numerator, denominator in original_actions[index]:
                budget.reserve(2)
                combined[position] = combined.get(position, Fraction(0)) + weight * Fraction(
                    numerator, denominator
                )
        actions.append(
            tuple(
                (position, value.numerator, value.denominator)
                for position, value in sorted(combined.items())
                if value
            )
        )
    elements = {
        block.name: _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
            block.cell_kind, 1
        )
        for block in mesh.blocks
    }
    routes = {block.name: block.vertices for block in mesh.blocks}
    exact = rational_geometry_owner.ExactRationalCellGeometrySource(
        parent,
        tuple(actions),
        rational_geometry_owner.rational_source_layouts(elements, routes),
        maximum_work=budget.maximum_work_units,
        maximum_bits=maximum_bits,
        budget=budget,
    )
    return _owner_phydrax_discretization__cell_geometry.CellGeometrySpec(
        elements,
        routes,
        exact.representatives(budget=budget),
        exact_source=exact,
        budget=budget,
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
    aim = optimized.aim_evidence
    aim_requested: tuple[tuple[str, float], ...] = ()
    aim_achieved: tuple[tuple[str, float], ...] = ()
    if aim is not None:
        radius, angle = aim.maximum_radius_edge, aim.minimum_dihedral_degrees
        radius_count, angle_count = aim.radius_unmet_cells, aim.dihedral_unmet_cells
        if radius is None or angle is None or radius_count is None or angle_count is None:
            raise RuntimeError("A complete guarded AIM measurement must include every achieved shape value.")
        aim_requested = (
            ("aim:radius_edge_bound", aim.requested_radius_edge_bound),
            ("aim:minimum_dihedral_degrees", aim.requested_minimum_dihedral_degrees),
        )
        aim_achieved = (
            ("aim:maximum_radius_edge", radius),
            ("aim:minimum_dihedral_degrees", angle),
            ("aim:radius_unmet_cells", float(radius_count)),
            ("aim:dihedral_unmet_cells", float(angle_count)),
            ("aim:refinement_rounds", float(aim.refinement_rounds)),
            ("aim:improvement_passes", float(aim.improvement_passes)),
            ("aim:attempts", float(aim.attempts)),
            ("aim:accepted_operations", float(aim.accepted_operations)),
            ("aim:refused_proposals", float(aim.refused_proposals)),
            ("aim:work_units", float(aim.work_units)),
            ("aim:measurement_complete", float(aim.measurement_complete)),
            (f"aim:termination:{aim.terminal_reason}", 1.0),
        )
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
    elif complex_.geometry is not None:
        if source is None:
            raise ValueError(
                "PlcCarrier.authored_geometry must retain native source row identities; got no declared source"
            )
        with native_volume_coordinate_budget(specification.limits, scope=scope) as budget:
            geometry = _authored_plc_geometry(
                complex_, source, source_evidence, arrays, used, mesh, budget=budget
            )
            mesh = _dependency_phydrax_meshing__canonical.canonicalize_cell_mesh(
                CellMesh.from_tetrahedra(
                    geometry.coordinates, tetrahedra, numeric_version=source_revision
                )
            )
            geometry.resolve(mesh, budget=budget)
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
            and (aim is None or aim.terminal_reason == "satisfied")
            else _dependency_phydrax_meshing__trace.MeshingStageStatus.WARNING,
            input_ids=tuple(dict.fromkeys((input_id, *construction_stage.output_ids))),
            output_ids=tuple(
                dict.fromkeys(
                    (
                        mesh.mesh_id,
                        *(() if metric_optimization is None else (metric_optimization.evidence_id,)),
                        *(() if aim is None else (aim.state_binding,)),
                    )
                )
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
        aim_requested=aim_requested,
        aim_achieved=aim_achieved,
        source_constraints=carrier.source_constraints,
        source_input_points=None if carrier.tables is None else carrier.tables.points[:complex_.vertices.shape[0]],
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
        # Keep represented-coordinate carriers' existing immutable admission,
        # and execute an explicitly requested weighted exudation stage on its
        # existing route rather than silently omitting it from this epoch.
        hard_goal_first = (
            protected.size_goal is not None
            and carrier.geometry is None
            and schedule.exudation_passes == 0
        )
        protected = replace(
            protected,
            construction_counters=(
                *protected.construction_counters,
                ("initial_goal_shape_certified", int(admitted)),
                ("initial_goal_certification_work_units", admission_work),
                ("hard_goal_first", int(hard_goal_first)),
            ),
            construction_work_units=protected.construction_work_units + admission_work,
        )
        if admitted:
            runs = PlcRefinement(
                None, None, None, None, protected.construction_work_units
            )
        elif hard_goal_first:
            goal = protected.size_goal
            if goal is None:
                raise RuntimeError("The hard-goal-first route requires its uniform size goal.")
            # Conforming Delaunay size refinement resolves every source gap
            # narrower than the hard lower size bound by encroachment, so it
            # may over-resolve the lower statistic there. Such sources keep
            # the metric route unless an independent native replica measures
            # complete hard size compliance after the same refinement.
            separated, separation_work = _source_features_separated(
                carrier, _uniform_edge_bounds(goal)[0], scope=scope
            )
            protected = replace(
                protected,
                construction_counters=(
                    *protected.construction_counters,
                    ("source_separation_work_units", separation_work),
                ),
                construction_work_units=protected.construction_work_units
                + separation_work,
            )
            refinement, spent = None, protected.construction_work_units
            if separated:
                partition, protected = _orthogonal_plc_size_grid(
                    carrier,
                    protected,
                    specification,
                    goal,
                    scope=scope,
                    record_phase=record_phase,
                )
                if partition is not None:
                    carrier = partition
                    spent = protected.construction_work_units
                else:
                    refinement, spent = _refine_hard_size(
                        carrier, protected, specification, goal,
                        scope=scope, record_phase=record_phase,
                    )
            else:
                trial, refinement, protected = _trial_hard_size(
                    carrier,
                    protected,
                    specification,
                    goal,
                    scope=scope,
                    record_phase=record_phase,
                )
                if trial is not None:
                    carrier.state.close()
                    carrier = trial
                spent = protected.construction_work_units
            # The phase is reported only when its native refinement is
            # retained by the published carrier.
            protected = replace(
                protected,
                construction_counters=(
                    *protected.construction_counters,
                    ("hard_size_native_phase", int(refinement is not None)),
                ),
            )
            runs = _repair_plc_validity(
                carrier,
                specification,
                schedule,
                refinement,
                spent,
                validity_policy=validity_policy,
                scope=scope,
                record_phase=record_phase,
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
        if hard_goal_first and runs.improvement is not None:
            protected = replace(
                protected,
                construction_counters=(
                    *protected.construction_counters,
                    ("hard_validity_repair_minimum_dihedral_degrees", 0),
                    ("hard_validity_repair_passes", runs.improvement.counters.passes),
                    ("hard_validity_repair_work_units", runs.improvement.counters.work_units),
                ),
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
            hard_goal_first=hard_goal_first,
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
