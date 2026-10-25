# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native planar motion, physical field preparation, Poisson update, and restart."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from fractions import Fraction
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax import (
    array_tree_fingerprint,
    ArrayArchiveLimits,
    BranchDifferentiationPolicy,
    canonical_fingerprint,
    StrictModule,
)
from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._frozendict import frozendict
from phydrax._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    certify_cell_geometry_validity,
    ExactPlcCellGeometrySource,
    FieldTracePolicy,
    FiniteElementDirichletConstraint,
    FiniteElementDiscretization,
    FiniteElementPlan,
    PolygonalConnectivity,
)
from phydrax.discretization._cell_geometry_transfer import (
    transition_displaced_cell_geometry,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
from phydrax.discretization._exact_plc_geometry import translated_plc_layout_matches
from phydrax.equations import CompiledFiniteElementProblem
from phydrax.geometry import PiecewiseLinearDomain, PlanarMeshRegion, SegmentMesh
from phydrax.geometry._domain_coverage import DomainCoverageCertificate
from phydrax.geometry._source_fidelity import fidelity_norm_bound
from phydrax.lifecycle import CompositionEntry
from phydrax.lifecycle._meshing_field_records import (
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    prepare_meshing_field_index_binding,
    prepare_meshing_field_owner,
)
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.linalg import LinearSolvePolicy
from phydrax.meshing import (
    CellMeshAuditDisposition,
    CellMeshAuditPolicy,
    CellMeshingResult,
    CellMeshTransition,
    evaluate_cell_quality,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
    MeshCertificationPreparedEvidence,
    MeshCertificationSchedule,
    MeshingComplianceReport,
    MeshingLimits,
    MeshingScope,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshLabel,
    MeshLineage,
    MeshPart,
    MeshPatch,
    MeshTransitionKind,
    MeshZone,
    NativeMeshingPhaseRecorder,
    NativeMeshingPlan,
    OversetConnectivity,
    OversetRegistration,
    ProtectedFeature,
    SurfaceMeshingSpec,
    UniformSizeControl,
)
from phydrax.meshing._association import PlcAssociationTransfer, retain_association_scopes
from phydrax.meshing._lineage import identity_lineage, inherit_mesh_organization
from phydrax.meshing._native_preparation import (
    native_preparation_ledger,
    NativeCompletedSourceIssue,
    stage_native_completed_source_issue,
)
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativePlanarSource,
)
from phydrax.meshing.providers._native_planar_domain import (
    measure_edge_lengths,
    planar_compliance,
    PreparedPlanarDomain,
    source_arrays,
    source_carrier_evidence,
)
from phydrax.meshing.providers._native_publication import (
    bind_native_execution_result,
    coordinate_budget,
    coordinate_phase,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import publication_policy
from phydrax.meshing.providers._native_sources import source_entity_id
from phydrax.typing import checked


type _HostFloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type _HostIndexArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]


def _cell_result(part: MeshPart) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            f"native_moving_overset.part.carrier must be CellMeshingResult; got {type(carrier).__name__}"
        )
    return carrier


def _archive_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"native_moving_overset.archive.{name} must be a named record mapping; got {type(value).__name__}"
        )
    records: dict[str, object] = {}
    for key, record in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"native_moving_overset.archive.{name}.key must be str; got {type(key).__name__}"
            )
        records[key] = record
    return records


def _planar_carrier_deviations(
    authority: ExactPlcCellGeometrySource,
    points: _HostFloatArray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_HostFloatArray, _HostFloatArray]:
    """Certify numerical carriers against retained PLC source coordinates."""
    prepared = authority.prepare(points, budget=budget)
    rows = np.asarray(authority.vertex_rows, dtype=np.int64)
    strata = np.asarray(authority.vertex_strata, dtype=np.int64)
    source_bounds = np.asarray(authority.source_segment_bounds, dtype=np.float64)
    source_deviations = np.zeros(source_bounds.shape, dtype=np.float64)
    carrier_deviations = np.zeros(points.shape[0], dtype=np.float64)
    for vertex in np.flatnonzero(strata == 1).tolist():
        difference = tuple(
            value - Fraction(float(carrier))
            for value, carrier in zip(
                prepared.vertices[vertex], points[vertex], strict=True
            )
        )
        deviation = fidelity_norm_bound(
            tuple({(0,): value} for value in difference), 1, budget=budget
        )
        edge = int(rows[vertex])
        if deviation > source_bounds[edge]:
            raise ValueError(
                f"native_moving_overset.source_edge.carrier_deviation must meet the declared bound for edge {edge}; got {deviation} above {source_bounds[edge]}."
            )
        carrier_deviations[vertex] = deviation
        source_deviations[edge] = max(source_deviations[edge], deviation)
    return source_deviations, carrier_deviations


@final
class _RigidMotionInputs(StrictModule):
    before: CellMeshingResult
    accepted_domain: PiecewiseLinearDomain
    connectivity: PolygonalConnectivity
    shift: _HostFloatArray


@final
class _TranslatedPlan(StrictModule):
    source: NativePlanarSource
    specification: SurfaceMeshingSpec
    prepared: PreparedPlanarDomain
    plan: NativeMeshingPlan
    revision: str = eqx.field(static=True)


@final
class _CarrierEvidence(StrictModule):
    geometry: CellGeometrySpec
    authority: ExactPlcCellGeometrySource
    source_deviations: _HostFloatArray
    carrier_deviations: _HostFloatArray
    certified_deviation: float
    source_bound: float


@final
class _MotionCertificate(StrictModule):
    geometry: CellGeometrySpec
    request: NativeCertificationRequest
    prepared: MeshCertificationPreparedEvidence
    audit_policy: CellMeshAuditPolicy
    patches: tuple[MeshPatch, ...]
    zones: tuple[MeshZone, ...]
    labels: tuple[MeshLabel, ...]
    associations: tuple[GeometryAssociation, ...]


@final
class _InteriorMotionInputs(StrictModule):
    domain: PiecewiseLinearDomain
    points: _HostFloatArray
    connectivity: PolygonalConnectivity
    mesh: CellMesh


@final
class _InteriorSupport(StrictModule):
    geometry: CellGeometrySpec
    request: NativeCertificationRequest
    prepared: MeshCertificationPreparedEvidence
    audit_policy: CellMeshAuditPolicy
    lineage: MeshLineage
    proved: tuple[GeometryAssociation, ...]


def _rigid_motion_inputs(
    part: MeshPart, offset: ArrayLike, motion_id: str, specification: SurfaceMeshingSpec
) -> _RigidMotionInputs:
    before = _cell_result(part)
    source_certificate = before.certification
    if source_certificate is None or source_certificate.request.domain is None:
        raise ValueError(
            f"native_moving_overset.before.certification must retain a represented-domain request; got {source_certificate!r}"
        )
    accepted_domain = source_certificate.request.domain
    if not isinstance(accepted_domain, PiecewiseLinearDomain):
        raise TypeError(
            f"native_moving_overset.before.certification.request.domain must be PiecewiseLinearDomain; got {type(accepted_domain).__name__}"
        )
    connectivity = before.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError(
            f"native_moving_overset.before.mesh.connectivity must be PolygonalConnectivity; got {type(connectivity).__name__}"
        )
    if before.compliance.specification_id != specification.specification_id:
        raise ValueError(
            f"native_moving_overset.before.compliance.specification_id must equal {specification.specification_id!r}; got {before.compliance.specification_id!r}"
        )
    shift = np.asarray(offset, dtype=np.float64)
    if shift.shape != (2,) or not np.all(np.isfinite(shift)):
        raise ValueError(
            f"native_moving_overset.offset must be a finite two-dimensional translation; got {shift!r}"
        )
    if not motion_id:
        raise ValueError(
            f"native_moving_overset.motion_id must be nonempty; got {motion_id!r}"
        )
    return _RigidMotionInputs(before, accepted_domain, connectivity, shift)


def _translate_planar_plan(
    plan: NativeMeshingPlan,
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    shift: _HostFloatArray,
    motion_id: str,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> _TranslatedPlan:
    revision = canonical_fingerprint(
        {
            "kind": "native-planar-rigid-pose",
            "source": source.binding_id,
            "motion": str(motion_id),
            "offset": array_tree_fingerprint(shift),
        }
    )
    edges, offsets = (
        np.asarray(source.region.edges),
        np.asarray(source.region.loop_offsets),
    )
    loops = tuple(
        tuple(int(row) for row in edges[first:last, 0])
        for first, last in zip(offsets[:-1], offsets[1:], strict=True)
    )
    region = PlanarMeshRegion(
        np.asarray(source.region.vertices) + shift,
        loops,
        feature_id=source.source_id,
    )
    embedded = (
        None
        if source.embedded is None
        else SegmentMesh(
            source.embedded.vertices + shift,
            source.embedded.edges,
            source_id=source.embedded.source_id,
        )
    )
    moved_source = NativePlanarSource(region, revision, embedded=embedded)

    def rebound(scope: MeshingScope) -> MeshingScope:
        if (scope.source_id, scope.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            raise ValueError(
                f"native_moving_overset.control.scope must bind source {(source.source_id, source.source_revision)!r}; got {(scope.source_id, scope.source_revision)!r}"
            )
        return MeshingScope(
            scope.source_id,
            revision,
            scope.entity_kind,
            scope.entity_dimension,
            scope.entity_set_id,
            scope.entity_ids,
        )

    controls = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            raise TypeError(
                f"native_moving_overset.specification.size_controls must contain UniformSizeControl; got {type(control).__name__}"
            )
        controls.append(
            UniformSizeControl(
                rebound(control.scope),
                control.target_size,
                minimum_size=control.minimum_size,
                maximum_size=control.maximum_size,
                maximum_growth_rate=control.maximum_growth_rate,
                strength=control.strength,
                priority=control.priority,
            )
        )
    features = tuple(
        ProtectedFeature(
            rebound(feature.scope),
            feature.feature_kind,
            maximum_deviation=feature.maximum_deviation,
            hard=feature.hard,
        )
        for feature in specification.protected_features
    )
    moved_specification = SurfaceMeshingSpec(
        specification.target,
        rebound(specification.scope),
        planar_embedding=specification.planar_embedding,
        size_controls=tuple(controls),
        protected_features=features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        periodic_constraints=specification.periodic_constraints,
        layer_controls=specification.layer_controls,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        quality_target=specification.quality_target,
        background_metric=specification.background_metric,
        deterministic=specification.deterministic,
    )
    moved_plan = NativeMeshingProvider(plan.options).plan(
        moved_source,
        moved_specification,
        coordinate_contract=plan.coordinate_contract,
        record_phase=record_phase,
    )
    prepared = moved_plan.prepared
    if not isinstance(prepared, PreparedPlanarDomain):
        raise TypeError(
            f"native_moving_overset.moved_plan.prepared must be PreparedPlanarDomain; got {type(prepared).__name__}"
        )
    return _TranslatedPlan(
        moved_source, moved_specification, prepared, moved_plan, revision
    )


def _rigid_constraint_rows(
    before: CellMeshingResult,
    source: NativePlanarSource,
    connectivity: PolygonalConnectivity,
) -> tuple[_HostIndexArray, _HostIndexArray]:
    edge_set = before.mesh.entity_set(1)
    ids = np.asarray(edge_set.entity_ids)
    order = np.argsort(ids, kind="stable")
    _, source_edges, _ = source_arrays(source)
    edge_names = {
        source_entity_id(source.source_revision, "edge", index): index
        for index in range(len(source_edges))
    }
    pairs, sources = [], []
    for association in before.associations:
        if (
            association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
            and association.source_id == source.source_id
            and association.source_revision == source.source_revision
            and association.target_entity_set_id == edge_set.entity_set_id
        ):
            roles = association.source_entities.roles()
            if roles is None:
                raise ValueError(
                    "native_moving_overset.association.source_entity_roles must be explicit; got None"
                )
            for identifier, name, dimension, role in zip(
                np.asarray(association.target_global_ids),
                association.source_entities.entity_ids(),
                np.asarray(association.source_dimensions),
                roles,
                strict=True,
            ):
                if dimension == 2 and role is GeometrySourceEntityRole.REGION:
                    continue
                if dimension != 1 or role is not GeometrySourceEntityRole.EDGE:
                    raise ValueError(
                        f"native_moving_overset.association.source_stratum must be an edge of dimension 1; got dimension={dimension!r}, role={role!r}"
                    )
                if name not in edge_names:
                    raise ValueError(
                        f"native_moving_overset.association.source_entity_id must name a source edge; got {name!r}"
                    )
                row = order[np.searchsorted(ids[order], identifier)]
                pairs.append(np.asarray(connectivity.edges)[row])
                sources.append(edge_names[name])
    if not pairs or set(sources) != set(range(len(source_edges))):
        raise ValueError(
            f"native_moving_overset.constraint_sources must cover every source edge; got {sources!r}"
        )
    constrained, constraint_sources = (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(sources, dtype=np.int64),
    )
    return constrained, constraint_sources


def _rigid_carrier_evidence(
    before: CellMeshingResult,
    shift: _HostFloatArray,
    points: _HostFloatArray,
    prepared: PreparedPlanarDomain,
    moved_source: NativePlanarSource,
    budget: CoordinateEnclosureBudget,
) -> _CarrierEvidence:
    match before.geometry.exact_source:
        case ExactPlcCellGeometrySource() as previous_authority:
            moved_authority = ExactPlcCellGeometrySource(
                prepared.source_points,
                previous_authority.source_triangles,
                prepared.source_edges,
                previous_authority.vertex_strata,
                previous_authority.vertex_rows,
                previous_authority.vertex_parameters,
                domain_source_id=moved_source.source_id,
                domain_source_revision=moved_source.source_revision,
                source_triangle_ids=previous_authority.source_triangle_ids,
                source_triangle_bounds=previous_authority.source_triangle_bounds,
                source_segment_ids=previous_authority.source_segment_ids,
                source_segment_bounds=prepared.source_bounds,
                vertex_parameter_ratios=previous_authority.vertex_parameter_ratios,
                maximum_work=previous_authority.maximum_work,
                maximum_bits=previous_authority.maximum_bits,
                budget=budget,
            )
        case _:
            raise ValueError(
                f"native_moving_overset.geometry.exact_source must be ExactPlcCellGeometrySource; got {type(before.geometry.exact_source).__name__}"
            )
    moved_geometry = phx.discretization.CellGeometrySpec(
        dict(
            zip(
                before.geometry.block_names,
                before.geometry.elements,
                strict=True,
            )
        ),
        dict(
            zip(
                before.geometry.block_names,
                before.geometry.geometry_dofs,
                strict=True,
            )
        ),
        before.geometry.coordinates + shift,
        restriction_source=before.geometry.restriction_source,
        exact_source=moved_authority,
        budget=budget,
    )
    source_deviations, carrier_deviations = _planar_carrier_deviations(
        moved_authority, points, budget=budget
    )
    certified_deviation = float(np.max(source_deviations, initial=0.0))
    source_bound = float(np.max(prepared.source_bounds, initial=0.0))
    return _CarrierEvidence(
        moved_geometry,
        moved_authority,
        source_deviations,
        carrier_deviations,
        certified_deviation,
        source_bound,
    )


def _rigid_motion_certificate(
    before: CellMeshingResult,
    mesh: CellMesh,
    plan: NativeMeshingPlan,
    original_source: NativePlanarSource,
    translated: _TranslatedPlan,
    domain: PiecewiseLinearDomain,
    accepted_domain: PiecewiseLinearDomain,
    shift: _HostFloatArray,
    evidence: _CarrierEvidence,
    budget: CoordinateEnclosureBudget,
) -> _MotionCertificate:
    source = original_source
    moved_source = translated.source
    moved_specification = translated.specification
    moved_geometry = evidence.geometry

    def authority_transfer(
        authority: NativePlanarSource,
        support_domain: PiecewiseLinearDomain,
    ) -> PlcAssociationTransfer:
        authority_points, authority_edges, _ = source_arrays(authority)
        return PlcAssociationTransfer(
            support_domain,
            plan.coordinate_contract,
            authority.source_revision,
            edge_vertices=authority_edges,
            source_vertices=authority_points,
            triangle_vertices=np.empty((0, 3), dtype=np.int64),
            triangle_facets=np.empty((0,), dtype=np.int64),
            facet_regions=np.empty((0, 2), dtype=np.int64),
        )

    predecessor_transfer = authority_transfer(source, accepted_domain)
    successor_transfer = authority_transfer(moved_source, domain)
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    region_labels = np.zeros(mesh.blocks[0].cell_count, dtype=np.int64)
    request = NativeCertificationRequest(
        MeshCertificationSchedule("volume_plc"),
        moved_source.source_id,
        moved_source.source_revision,
        moved_specification.limits,
        domain=domain,
        cell_regions=region_labels,
    )
    validity = certify_cell_geometry_validity(
        moved_geometry,
        mesh=mesh,
        policy=audit_policy.validity_policy,
        budget=budget,
    )
    premises = MeshCertificationPreparedEvidence(
        mesh,
        moved_geometry,
        validity,
        schedule=request.schedule,
        domain=domain,
        cell_regions=region_labels,
        limits=request.certificate_limits,
        budget=budget,
    )
    if not isinstance(premises.coverage, DomainCoverageCertificate):
        raise TypeError("Rigid moving PLC support requires exact domain coverage")
    patches, zones, labels, associations = (
        successor_transfer.transition_source_associations(
            before,
            predecessor_transfer,
            mesh,
            geometry=moved_geometry,
            translation=shift,
            embedding=premises.embedding,
            coverage=premises.coverage,
            budget=budget,
        )
    )
    return _MotionCertificate(
        moved_geometry,
        request,
        premises,
        audit_policy,
        patches,
        zones,
        labels,
        associations,
    )


def _rigid_motion_compliance(
    before: CellMeshingResult,
    mesh: CellMesh,
    points: _HostFloatArray,
    connectivity: PolygonalConnectivity,
    constrained: _HostIndexArray,
    constraint_sources: _HostIndexArray,
    original_prepared: PreparedPlanarDomain,
    translated: _TranslatedPlan,
    evidence: _CarrierEvidence,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> MeshingComplianceReport:
    prepared = translated.prepared
    moved_specification = translated.specification
    moved_authority = evidence.authority
    source_deviations = evidence.source_deviations
    carrier_deviations = evidence.carrier_deviations
    certified_deviation = evidence.certified_deviation
    source_bound = evidence.source_bound
    measured = evaluate_cell_quality(mesh, budget=budget)
    minimum_angle = float(np.min(np.asarray(measured.minimum_angle)))
    retained_steiner = mesh.coordinates.shape[0] - original_prepared.points.shape[0]
    if retained_steiner < 0:
        raise ValueError(
            f"native_moving_overset.mesh.vertex_count must retain source constraint vertices; got {mesh.coordinates.shape[0]} below {original_prepared.points.shape[0]}"
        )
    # This status is the observed unchanged-topology predicate; no CDT was run.
    status = "ok" if mesh.topology_id == before.mesh.topology_id else "topology_changed"
    edges = np.sort(np.asarray(connectivity.edges), axis=1)
    compliance = planar_compliance(
        moved_specification,
        prepared,
        points,
        np.asarray(mesh.blocks[0].vertices),
        edges,
        measure_edge_lengths(points, edges, scope=scope),
        (np.sort(constrained, axis=1), constraint_sources),
        (np.degrees(minimum_angle), retained_steiner, status),
        (certified_deviation, source_bound),
        scope=scope,
    )
    source_requested, source_achieved = source_carrier_evidence(
        prepared,
        moved_authority,
        np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        carrier_deviations,
        source_deviations,
    )
    compliance = phx.meshing.MeshingComplianceReport(
        moved_specification.specification_id,
        issues=compliance.issues,
        requested=(*compliance.requested, *source_requested),
        achieved=(*compliance.achieved, *source_achieved),
    )
    return compliance


def _interior_motion_inputs(
    before: CellMeshingResult,
    coordinates: ArrayLike,
    motion_id: str,
    specification: SurfaceMeshingSpec,
) -> _InteriorMotionInputs:
    if before.compliance.specification_id != specification.specification_id:
        raise ValueError(
            f"native_moving_overset.before.compliance.specification_id must equal {specification.specification_id!r}; got {before.compliance.specification_id!r}"
        )
    certificate = before.certification
    if certificate is None or not isinstance(
        certificate.request.domain, PiecewiseLinearDomain
    ):
        raise ValueError(
            f"native_moving_overset.before.certification must retain a PiecewiseLinearDomain request; got {certificate!r}"
        )
    domain = certificate.request.domain
    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.shape != before.mesh.coordinates.shape
        or not np.all(np.isfinite(points))
        or not motion_id
    ):
        raise ValueError(
            f"native_moving_overset.motion must have finite same-layout coordinates and a nonempty identity; got coordinates={points!r}, motion_id={motion_id!r}"
        )
    connectivity = before.mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError(
            f"native_moving_overset.before.mesh.connectivity must be PolygonalConnectivity; got {type(connectivity).__name__}"
        )
    boundary = np.unique(
        np.asarray(connectivity.edges)[np.asarray(connectivity.boundary_edges)]
    )
    if not np.array_equal(
        points[boundary], np.asarray(before.mesh.coordinates)[boundary]
    ):
        raise ValueError(
            f"native_moving_overset.coordinates.boundary must equal the accepted boundary coordinates; got {points[boundary]!r}"
        )
    if not np.array_equal(before.geometry.coordinates, before.mesh.coordinates):
        raise ValueError(
            f"native_moving_overset.before.geometry.coordinates must equal mesh coordinates; got {before.geometry.coordinates!r}"
        )
    revision = canonical_fingerprint(
        {
            "kind": "native-planar-interior-motion",
            "source": before.result_id,
            "motion": motion_id,
        }
    )
    mesh = before.mesh.with_coordinates(points, numeric_version=revision)
    return _InteriorMotionInputs(domain, points, connectivity, mesh)


def _interior_motion_support(
    before: CellMeshingResult,
    mesh: CellMesh,
    points: _HostFloatArray,
    domain: PiecewiseLinearDomain,
    plan: NativeMeshingPlan,
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    budget: CoordinateEnclosureBudget,
) -> _InteriorSupport:
    geometry = before.geometry.with_coordinates(points, budget=budget)
    source_points, source_edges, _ = source_arrays(source)
    transfer = PlcAssociationTransfer(
        domain,
        plan.coordinate_contract,
        source.source_revision,
        edge_vertices=source_edges,
        source_vertices=source_points,
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=np.empty((0, 2), dtype=np.int64),
        maximum_support_queries=specification.limits.maximum_geometry_queries,
    )
    audit_policy = CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
    )
    region_labels = np.zeros(mesh.blocks[0].cell_count, dtype=np.int64)
    request = NativeCertificationRequest(
        MeshCertificationSchedule("volume_plc"),
        source.source_id,
        source.source_revision,
        specification.limits,
        domain=domain,
        cell_regions=region_labels,
    )
    validity = certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=audit_policy.validity_policy,
        budget=budget,
    )
    prepared = MeshCertificationPreparedEvidence(
        mesh,
        geometry,
        validity,
        schedule=request.schedule,
        domain=domain,
        cell_regions=region_labels,
        limits=request.certificate_limits,
        budget=budget,
    )
    if not isinstance(prepared.coverage, DomainCoverageCertificate):
        raise TypeError("Interior moving PLC support requires exact domain coverage")
    lineage = identity_lineage(before.mesh, mesh)
    proved = transfer.propagate(
        before,
        lineage,
        mesh,
        geometry=geometry,
        embedding=prepared.embedding,
        coverage=prepared.coverage,
        budget=budget,
    )
    return _InteriorSupport(geometry, request, prepared, audit_policy, lineage, proved)


def _interior_motion_compliance(
    mesh: CellMesh,
    points: _HostFloatArray,
    connectivity: PolygonalConnectivity,
    original_prepared: PreparedPlanarDomain,
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    support: _InteriorSupport,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> MeshingComplianceReport:
    geometry = support.geometry
    proved = support.proved
    edge_set = mesh.entity_set(1)
    edge_support = next(
        value for value in proved if value.target_entity_set_id == edge_set.entity_set_id
    )
    edge_rows = edge_support.target_rows(np.asarray(edge_set.entity_ids))
    selected = np.asarray(edge_support.source_dimensions)[edge_rows] == 1
    constrained = np.sort(np.asarray(connectivity.edges)[selected], axis=1)
    constraint_sources = np.asarray(edge_support.source_indices)[edge_rows][selected]
    original_prepared.domain(source)
    match geometry.exact_source:
        case ExactPlcCellGeometrySource() as authority:
            source_deviations, carrier_deviations = _planar_carrier_deviations(
                authority, points, budget=budget
            )
        case _:
            raise ValueError(
                f"native_moving_overset.geometry.exact_source must be ExactPlcCellGeometrySource; got {type(geometry.exact_source).__name__}"
            )
    certified_deviation = float(np.max(source_deviations, initial=0.0))
    source_bound = float(np.max(original_prepared.source_bounds, initial=0.0))
    measured = evaluate_cell_quality(mesh, budget=budget)
    edges = np.sort(np.asarray(connectivity.edges), axis=1)
    compliance = planar_compliance(
        specification,
        original_prepared,
        points,
        np.asarray(mesh.blocks[0].vertices),
        edges,
        measure_edge_lengths(points, edges, scope=scope),
        (constrained, constraint_sources),
        (
            np.degrees(float(np.min(np.asarray(measured.minimum_angle)))),
            mesh.coordinates.shape[0] - original_prepared.points.shape[0],
            "ok",
        ),
        (certified_deviation, source_bound),
        scope=scope,
    )
    source_requested, source_achieved = source_carrier_evidence(
        original_prepared,
        authority,
        np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        carrier_deviations,
        source_deviations,
    )
    compliance = phx.meshing.MeshingComplianceReport(
        specification.specification_id,
        issues=compliance.issues,
        requested=(*compliance.requested, *source_requested),
        achieved=(*compliance.achieved, *source_achieved),
    )
    return compliance


def _publish_planar_motion(
    part: MeshPart,
    before: CellMeshingResult,
    mesh: CellMesh,
    plan: NativeMeshingPlan,
    certificate: _MotionCertificate,
    compliance: MeshingComplianceReport,
    identity: dict[str, str],
    input_ids: tuple[str, ...],
    output_ids: tuple[str, ...],
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    budget: CoordinateEnclosureBudget,
) -> MeshPart:
    geometry = certificate.geometry
    request = certificate.request
    prepared = certificate.prepared
    audit_policy = certificate.audit_policy
    patches, zones, labels, associations = (
        certificate.patches,
        certificate.zones,
        certificate.labels,
        certificate.associations,
    )
    result = publish_native_result(
        mesh,
        plan.coordinate_contract,
        compliance,
        (
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_ASSOCIATION,
                MeshingStageStatus.PASSED,
                input_ids=input_ids,
                output_ids=output_ids,
            ),
        ),
        before.provider,
        identity,
        replace(request, prepared=prepared),
        policy=replace(publication_policy("planar"), audit_policy=audit_policy),
        organization=NativePublicationOrganization(
            patches=patches, zones=zones, labels=labels, associations=associations
        ),
        execution=NativePublicationExecution(scope, workspace, record_phase, budget),
        geometry=geometry,
    )
    return MeshPart(part.name, result)


def _prepare_planar_motion_issue(
    successor: MeshPart,
    before: CellMeshingResult,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> NativeCompletedSourceIssue:
    """Prepay the actual target's handover while retaining the source as history."""
    previous = before.execution_evidence
    if previous is None or previous.owner_id is None:
        raise RuntimeError("native_moving_overset.motion must retain owner-bound source execution history")
    with budget.bind_ticket(scope.ticket_view()):
        native_preparation_ledger(scope).record_history(
            scope, previous, completed_source=before,
        )
        return stage_native_completed_source_issue(
            scope, _cell_result(successor), budget=budget,
        )


def _ended_planar_motion(
    successor: MeshPart,
    before: CellMeshingResult,
    limits: MeshingLimits,
    scope: NativeExecutionScope,
    issue: NativeCompletedSourceIssue,
) -> MeshPart:
    """Bind the actual ended motion request; the predecessor is history only."""
    try:
        if scope.evidence is None:
            raise RuntimeError("native_moving_overset.motion must retain its actual ended execution scope")
        previous = before.execution_evidence
        if previous is None or previous.owner_id is None:
            raise RuntimeError("native_moving_overset.motion must retain owner-bound source execution history")
        bound = bind_native_execution_result(
            _cell_result(successor), scope.evidence, limits, scope.operation_started,
            history_evidence=(previous,),
        )
        issue.complete(bound)
    except BaseException:
        issue.close()
        raise
    return MeshPart(successor.name, bound)


@final
class _RestartInputs(StrictModule):
    records: dict[str, object]
    roles: dict[str, str]


def _prepare_native_restart(
    connectivity: OversetConnectivity,
    coefficients: Mapping[str, Array],
    plans: Mapping[str, NativeMeshingPlan],
    declarations: Mapping[str, MeshingFiniteElementFieldDeclaration],
) -> _RestartInputs:
    connectivity.require_complete()
    names = {part.name for part in connectivity.assembly.parts}
    if set(coefficients) != names or set(plans) != names or set(declarations) != names:
        raise ValueError(
            f"native_moving_overset.restart.inputs must name every part {names!r}; got coefficients={set(coefficients)!r}, plans={set(plans)!r}, declarations={set(declarations)!r}"
        )
    primary_part = next(
        (
            part
            for part in connectivity.assembly.parts
            if _cell_result(part).certification is not None
        ),
        None,
    )
    if primary_part is None:
        raise ValueError(
            "native_moving_overset.restart.primary_part must retain source certification; got None"
        )
    primary = _cell_result(primary_part)
    certification = primary.certification
    if certification is None:
        raise ValueError(
            "native_moving_overset.restart.certification must retain a source request; got None"
        )
    accepted = {}
    roles = {}
    for name, declaration in declarations.items():
        coefficient_roles = tuple(
            role for role in declaration.state_roles if role.role == "coefficients"
        )
        if (
            declaration.history_policy != "none"
            or len(coefficient_roles) != 1
            or len(declaration.state_roles) != 1
        ):
            raise ValueError(
                f"native_moving_overset.restart.declaration must have one history-free coefficient role; got history_policy={declaration.history_policy!r}, state_roles={declaration.state_roles!r}"
            )
        role = coefficient_roles[0]
        roles[name] = role.entry_name
        accepted[role.entry_name] = coefficients[name]
    records: dict[str, object] = {
        "certification_inputs": certification.request,
        "report": certification,
        "associations": primary.associations,
        "registration": connectivity.registration(),
        "generation_sources": {name: plan.source for name, plan in plans.items()},
        "generation_specifications": {
            name: plan.specification for name, plan in plans.items()
        },
        "generation_options": {name: plan.options for name, plan in plans.items()},
        "primary_generation_part": primary_part.name,
        "field_declarations": declarations,
        "accepted_data": {"fields": accepted},
    }
    return _RestartInputs(records, roles)


def _restore_native_at(
    target: Path,
    inputs: _RestartInputs,
    limits: ArrayArchiveLimits,
    connectivity: OversetConnectivity,
    coefficients: Mapping[str, Array],
    plans: Mapping[str, NativeMeshingPlan],
) -> tuple[
    OversetConnectivity,
    dict[str, Array],
    dict[str, NativeMeshingPlan],
    dict[str, FiniteElementDiscretization],
]:
    records, roles = inputs.records, inputs.roles
    names = {part.name for part in connectivity.assembly.parts}
    receipt = write_meshing_source_closure(target, records, limits=limits)
    restored = _archive_mapping(
        read_meshing_source_closure(
            receipt.path,
            expected_content_id=receipt.content_id,
            limits=limits,
        ),
        "source closure",
    )
    registration = restored["registration"]
    if not isinstance(registration, OversetRegistration):
        raise TypeError(
            f"native_moving_overset.restart.registration must be OversetRegistration; got {type(registration).__name__}"
        )
    result = registration.prepare()
    result.require_complete()
    if result.connectivity_id != connectivity.connectivity_id:
        raise RuntimeError(
            f"native_moving_overset.restart.connectivity_id must equal {connectivity.connectivity_id!r}; got {result.connectivity_id!r}"
        )
    accepted_data = _archive_mapping(restored["accepted_data"], "accepted data")
    archived_values = _archive_mapping(accepted_data["fields"], "accepted fields")
    sources = _archive_mapping(restored["generation_sources"], "generation sources")
    specifications = _archive_mapping(
        restored["generation_specifications"], "generation specifications"
    )
    options = _archive_mapping(restored["generation_options"], "generation options")
    field_records = _archive_mapping(restored["field_declarations"], "field declarations")
    restored_values: dict[str, Array] = {}
    restored_plans: dict[str, NativeMeshingPlan] = {}
    restored_fields: dict[str, FiniteElementDiscretization] = {}
    for part in result.assembly.parts:
        name = part.name
        value = archived_values[roles[name]]
        source, specification, route_options = (
            sources[name],
            specifications[name],
            options[name],
        )
        declaration = field_records[name]
        if not isinstance(value, Array):
            raise TypeError(
                f"native_moving_overset.restart.coefficients must be a JAX array; got {type(value).__name__}"
            )
        if not isinstance(source, NativePlanarSource) or not isinstance(
            specification, SurfaceMeshingSpec
        ):
            raise TypeError(
                f"native_moving_overset.restart.source_request must be NativePlanarSource and SurfaceMeshingSpec; got {type(source).__name__}, {type(specification).__name__}"
            )
        if not isinstance(route_options, NativeMeshingOptions):
            raise TypeError(
                f"native_moving_overset.restart.options must be NativeMeshingOptions; got {type(route_options).__name__}"
            )
        if not isinstance(declaration, MeshingFiniteElementFieldDeclaration):
            raise TypeError(
                f"native_moving_overset.restart.declaration must be MeshingFiniteElementFieldDeclaration; got {type(declaration).__name__}"
            )
        restored_values[name] = value
        restored_plans[name] = NativeMeshingProvider(route_options).plan(
            source,
            specification,
            coordinate_contract=part.coordinate_contract,
        )
        prepared, reconstruction = prepare_meshing_field_owner(
            declaration, _cell_result(part)
        )
        if (
            not isinstance(prepared, FiniteElementDiscretization)
            or reconstruction is not None
        ):
            raise TypeError(
                f"native_moving_overset.restart.field_owner must be FiniteElementDiscretization without reconstruction; got {type(prepared).__name__}, reconstruction={reconstruction!r}"
            )
        restored_fields[name] = prepared
    for name in names:
        np.testing.assert_array_equal(restored_values[name], coefficients[name])
        if restored_plans[name].plan_id != plans[name].plan_id:
            raise RuntimeError(
                f"native_moving_overset.restart.plan_id must equal {plans[name].plan_id!r}; got {restored_plans[name].plan_id!r}"
            )
    return result, restored_values, restored_plans, restored_fields


@checked
def rigid_planar_successor(
    part: MeshPart,
    offset: ArrayLike,
    *,
    motion_id: str,
    plan: NativeMeshingPlan,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> tuple[MeshPart, NativeMeshingPlan]:
    """Translate the native planar source and recertify unchanged mesh topology.

    The admitted plan retains source constraints and the hard physical request.
    Moved points receive recomputed source-bound evidence.
    """
    if not isinstance(plan.source, NativePlanarSource):
        raise TypeError(
            f"native_moving_overset.plan.source must be NativePlanarSource; got {type(plan.source).__name__}"
        )
    if not isinstance(plan.specification, SurfaceMeshingSpec) or not isinstance(
        plan.prepared, PreparedPlanarDomain
    ):
        raise TypeError(
            f"native_moving_overset.plan.route must contain SurfaceMeshingSpec and PreparedPlanarDomain; got {type(plan.specification).__name__}, {type(plan.prepared).__name__}"
        )
    limits = plan.specification.limits
    with NativeExecutionScope(limits.native_limits()) as scope:
        scope.observe_managed_admission()
        with (
            scope.ticket_view() as ticket,
            ticket.native_target(),
            scope.host_workspace() as workspace,
        ):
            budget = coordinate_budget(limits)
            with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
                inputs = _rigid_motion_inputs(part, offset, motion_id, plan.specification)
                before, accepted_domain, connectivity, shift = (
                    inputs.before,
                    inputs.accepted_domain,
                    inputs.connectivity,
                    inputs.shift,
                )
                source, specification = plan.source, plan.specification
                translated = _translate_planar_plan(
                    plan, source, specification, shift, motion_id, record_phase
                )
                moved_source, moved_plan, prepared = (
                    translated.source,
                    translated.plan,
                    translated.prepared,
                )
                mesh = before.mesh.with_coordinates(
                    before.mesh.coordinates + shift, numeric_version=translated.revision
                )
                if mesh.topology_id != before.mesh.topology_id:
                    raise RuntimeError(
                        f"native_moving_overset.mesh.topology_id must equal {before.mesh.topology_id!r}; got {mesh.topology_id!r}"
                    )
                constrained, constraint_sources = _rigid_constraint_rows(
                    before, source, connectivity
                )
                points = np.asarray(mesh.coordinates)
                domain = prepared.domain(moved_source)
                with coordinate_phase(scope, limits, budget):
                    evidence = _rigid_carrier_evidence(
                        before, shift, points, prepared, moved_source, budget
                    )
                    certificate = _rigid_motion_certificate(
                        before,
                        mesh,
                        plan,
                        source,
                        translated,
                        domain,
                        accepted_domain,
                        shift,
                        evidence,
                        budget,
                    )
                    compliance = _rigid_motion_compliance(
                        before,
                        mesh,
                        points,
                        connectivity,
                        constrained,
                        constraint_sources,
                        plan.prepared,
                        translated,
                        evidence,
                        budget,
                        scope,
                    )
                successor = _publish_planar_motion(
                    part,
                    before,
                    mesh,
                    plan,
                    certificate,
                    compliance,
                    {
                        "kind": "native-planar-rigid-motion",
                        "source": moved_source.binding_id,
                        "source_result": before.result_id,
                        "plan": moved_plan.plan_id,
                        "motion": str(motion_id),
                    },
                    (before.result_id, source.binding_id),
                    (mesh.mesh_id, moved_source.binding_id),
                    record_phase,
                    scope,
                    workspace,
                    budget,
                )
                successor_geometry = _cell_result(successor).geometry
                with coordinate_phase(scope, limits, budget):
                    if not translated_plc_layout_matches(
                        before.geometry,
                        successor_geometry,
                        translation=shift,
                        budget=budget,
                    ):
                        raise RuntimeError(
                            f"native_moving_overset.successor.geometry must retain the translated coordinate layout; got {successor_geometry!r}"
                        )
                issue = _prepare_planar_motion_issue(successor, before, scope, budget)
    return _ended_planar_motion(successor, before, limits, scope, issue), moved_plan


@checked
def deform_planar_interior(
    part: MeshPart,
    coordinates: ArrayLike,
    *,
    motion_id: str,
    plan: NativeMeshingPlan,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    record_motion: Callable[[CellMeshTransition], None] | None = None,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> MeshPart:
    """Recertify motion and optionally retain its actual same-topology transition.

    The transition binds this source mesh and the published target with the
    exact lineage used for native source-association propagation. It is not a
    refinement/remesh or a second generation verdict.
    An explicit operand collector retains only numerical kernel operands earned
    by this request, not certification verdicts. Its caller aborts failed captures.
    """
    before = _cell_result(part)
    if (
        not isinstance(plan.source, NativePlanarSource)
        or not isinstance(plan.prepared, PreparedPlanarDomain)
        or not isinstance(plan.specification, SurfaceMeshingSpec)
    ):
        raise TypeError(
            f"native_moving_overset.plan.route must contain NativePlanarSource, PreparedPlanarDomain and SurfaceMeshingSpec; got {type(plan.source).__name__}, {type(plan.prepared).__name__}, {type(plan.specification).__name__}"
        )
    limits = plan.specification.limits
    with NativeExecutionScope(limits.native_limits()) as scope:
        scope.observe_managed_admission()
        with (
            scope.ticket_view() as ticket,
            ticket.native_target(),
            scope.host_workspace() as workspace,
        ):
            if coordinate_operands is not None:
                coordinate_operands.begin(
                    scope, None, native_preparation_ledger(scope).overlap(scope, None)
                )
            budget = coordinate_budget(limits)
            budget.capture_coordinate_operands = coordinate_operands is not None
            budget.capture_polynomial_operands = (
                coordinate_operands is not None and coordinate_operands.capture_polynomials
            )
            budget.capture_measure_operands = (
                coordinate_operands is not None and coordinate_operands.capture_measures
            )
            budget.capture_signed_determinant_operands = (
                coordinate_operands is not None and coordinate_operands.capture_signed_determinants
            )
            with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
                inputs = _interior_motion_inputs(
                    before, coordinates, motion_id, plan.specification
                )
                domain, points, connectivity, mesh = (
                    inputs.domain,
                    inputs.points,
                    inputs.connectivity,
                    inputs.mesh,
                )
                geometry_transition = None
                with coordinate_phase(scope, limits, budget):
                    support = _interior_motion_support(
                        before,
                        mesh,
                        points,
                        domain,
                        plan,
                        plan.source,
                        plan.specification,
                        budget,
                    )
                    compliance = _interior_motion_compliance(
                        mesh,
                        points,
                        connectivity,
                        plan.prepared,
                        plan.source,
                        plan.specification,
                        support,
                        budget,
                        scope,
                    )
                    if record_motion is not None:
                        geometry_transition = transition_displaced_cell_geometry(
                            before.mesh, before.geometry, mesh, budget=budget
                        )
                associations = retain_association_scopes(
                    before.associations, support.proved
                )
                patches, zones, labels = inherit_mesh_organization(
                    before, mesh, support.lineage
                )
                certificate = _MotionCertificate(
                    support.geometry,
                    support.request,
                    support.prepared,
                    support.audit_policy,
                    patches,
                    zones,
                    labels,
                    associations,
                )
                successor = _publish_planar_motion(
                    part,
                    before,
                    mesh,
                    plan,
                    certificate,
                    compliance,
                    {
                        "kind": "native-planar-interior-motion",
                        "source_result": before.result_id,
                        "plan": plan.plan_id,
                        "motion": motion_id,
                    },
                    (before.result_id,),
                    (mesh.mesh_id,),
                    record_phase,
                    scope,
                    workspace,
                    budget,
                )
                if coordinate_operands is not None:
                    published = _cell_result(successor)
                    with coordinate_phase(scope, limits, budget):
                        coordinate_operands.capture(
                            published.mesh, published.geometry, budget, scope,
                            input_owners=(before,),
                            coordinate_contract=published.coordinate_contract,
                        )
                issue = _prepare_planar_motion_issue(successor, before, scope, budget)
    successor = _ended_planar_motion(successor, before, limits, scope, issue)
    if coordinate_operands is not None:
        receipt = _cell_result(successor).execution_evidence
        if receipt is None:
            raise RuntimeError("Native planar motion must bind its operand earning receipt")
        coordinate_operands.complete(receipt)
    if record_motion is not None:
        transition = CellMeshTransition(
            before.mesh.mesh_id,
            before.mesh.topology_id,
            _cell_result(successor),
            support.lineage,
            MeshTransitionKind.MOTION,
            budget=budget,
            geometry_transition=geometry_transition,
        )
        record_motion(transition)
    return successor


def native_planar_part(
    name: str,
    half: float,
    *,
    cavity: float | None = None,
    size: float = 0.4,
) -> tuple[MeshPart, NativeMeshingPlan]:
    """Author and execute the protected native planar physical request."""
    vertices = np.asarray(
        ((-half, -half), (half, -half), (half, half), (-half, half)), dtype=np.float64
    )
    loops = ((0, 1, 2, 3),)
    if cavity is not None:
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(
                    (
                        (-cavity, -cavity),
                        (-cavity, cavity),
                        (cavity, cavity),
                        (cavity, -cavity),
                    ),
                    dtype=np.float64,
                ),
            )
        )
        loops = (*loops, (4, 5, 6, 7))
    revision = canonical_fingerprint(
        {
            "kind": "native-moving-planar-authority",
            "name": name,
            "vertices": vertices.tolist(),
            "loops": loops,
        }
    )
    source = phx.meshing.NativePlanarSource(
        phx.geometry.PlanarMeshRegion(vertices, loops, feature_id=name),
        revision,
    )
    scope = phx.meshing.MeshingScope(
        name,
        revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "fluid-region",
        np.asarray((0,), dtype=np.int64),
    )
    protected = tuple(
        phx.meshing.ProtectedFeature(
            phx.meshing.MeshingScope(
                name,
                revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                1,
                f"source-loop:{index}",
                np.arange(4 * index, 4 * index + 4, dtype=np.int64),
            ),
            phx.meshing.FeatureKind.CURVE,
            maximum_deviation=1e-10,
            hard=True,
        )
        for index in range(len(loops))
    )
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5,
            target_statistics=("p50",),
        ),
        protected_features=protected,
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=np.radians(25.0), hard=True
        ),
    )
    plan = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("planar_constrained_delaunay"),
    ).plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
    return phx.meshing.MeshPart(name, plan.execute()), plan


def overset_field_templates() -> tuple[
    tuple[str, phx.discretization.FiniteElementFieldSpec], ...
]:
    """Return the normalized-temperature declaration shared by every overset part."""
    return (
        (
            "intensive",
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("triangle", 1)
            ),
        ),
    )


@checked
def field_plans(connectivity: OversetConnectivity) -> dict[str, FiniteElementPlan]:
    """Return physical field plans for source-preserving restart."""
    plans = {}
    field = overset_field_templates()[0][1]
    for part in connectivity.assembly.parts:
        carrier = _cell_result(part)
        plans[part.name] = FiniteElementPlan(
            carrier.mesh,
            field,
            coordinate_spec=carrier.geometry,
        )
    return plans


@checked
def fields(connectivity: OversetConnectivity) -> dict[str, FiniteElementDiscretization]:
    return {name: plan.prepare() for name, plan in field_plans(connectivity).items()}


@checked
def field_declarations(
    connectivity: OversetConnectivity,
    plans: Mapping[str, FiniteElementPlan],
    discretizations: Mapping[str, FiniteElementDiscretization],
    coefficients: Mapping[str, Array],
) -> dict[str, MeshingFiniteElementFieldDeclaration]:
    """Declare normalized-temperature state in its accepted FE layout.

    ``u`` is dimensionless; coordinates are SI lengths, so the unit Poisson
    source denotes one inverse-square-meter in this normalized model. Geometry
    decisions are frozen within an accepted epoch. This steady solve has no
    time/material history. The historical ``temperature:<part>`` entry names
    remain explicit state-role identifiers, not a Kelvin-unit inference.
    """
    names = {part.name for part in connectivity.assembly.parts}
    if set(plans) != names or set(discretizations) != names or set(coefficients) != names:
        raise ValueError(
            f"native_moving_overset.field.inputs must name every accepted part {names!r}; got plans={set(plans)!r}, discretizations={set(discretizations)!r}, coefficients={set(coefficients)!r}"
        )
    declarations = {}
    specs = {spec.part_name: spec for spec in connectivity.specs}
    for part in connectivity.assembly.parts:
        name, plan, prepared = part.name, plans[part.name], discretizations[part.name]
        carrier = _cell_result(part)
        if prepared.plan_id != plan.plan_id:
            raise ValueError(
                f"native_moving_overset.field.plan_id must equal {plan.plan_id!r}; got {prepared.plan_id!r}"
            )
        if tuple(field.name for field in plan.fields) != ("u",):
            raise ValueError(
                f"native_moving_overset.field.names must equal ('u',); got {tuple(field.name for field in plan.fields)!r}"
            )
        value = coefficients[name]
        index_binding = prepare_meshing_field_index_binding(
            carrier,
            "mesh/vertex_ids",
            field_space=prepared.field_spaces[0],
        )
        declarations[name] = MeshingFiniteElementFieldDeclaration(
            part_name=name,
            topology_id=carrier.mesh.topology_id,
            geometry_layout_id=carrier.geometry.geometry_layout_id,
            field_space_ids=frozendict(
                {space.name: space.field_space_id for space in prepared.field_spaces}
            ),
            value_units=frozendict({"u": (phx.units.ONE,)}),
            maximum_derivative_orders=frozendict({"u": 1}),
            branch_policy=BranchDifferentiationPolicy.FROZEN_DECISION,
            trace_policy=FieldTracePolicy("cell-sided"),
            state_roles=(
                MeshingFieldStateRole(
                    f"temperature:{name}",
                    "u",
                    "coefficients",
                    tuple(value.shape),
                    np.dtype(value.dtype),
                    connectivity.epoch,
                    index_binding=index_binding,
                ),
            ),
            history_policy="none",
            numeric_version=prepared.numeric_version,
            finite_element_fields=plan.fields,
            precision_policy=plan.precision_policy,
            source_wall_policy=specs[name],
        )
    return declarations


@checked
def field_entries(
    connectivity: OversetConnectivity,
    discretizations: Mapping[str, FiniteElementDiscretization],
    coefficients: Mapping[str, Array],
    revision: str,
) -> tuple[tuple[CompositionEntry, ...], tuple[CompositionEntry, ...]]:
    part_entries = {entry.entry_id: entry for entry in connectivity.composition_entries()}
    prepared, states = [], []
    for name, field in sorted(discretizations.items()):
        part = part_entries[f"overset:part:{name}"]
        artifact = phx.lifecycle.CompositionEntry(
            field,
            entry_id=f"fe:{name}",
            role="discretization",
            owner_id="native-overset-example",
            structure_id=field.dof_maps[0].dof_map_id,
            revision_id=field.prepared_id,
            semantics_id=f"temperature-space:{name}",
            dependencies=(part.binding("revision"),),
        )
        prepared.append(artifact)
        states.append(
            phx.lifecycle.CompositionEntry(
                coefficients[name],
                entry_id=f"temperature:{name}",
                role="physical-state",
                owner_id="native-overset-example",
                structure_id=artifact.structure_id,
                revision_id=f"temperature:{name}:{revision}",
                semantics_id="temperature",
                dependencies=(artifact.binding("revision"),),
            )
        )
    return tuple(prepared), tuple(states)


@checked
def prepare_poisson_update(
    field: FiniteElementDiscretization,
    carried: Array,
    *,
    constraint: FiniteElementDirichletConstraint | None = None,
) -> CompiledFiniteElementProblem:
    """Compile ``-Δu = 1`` with strong data ``carried`` on the declared rows.

    The default constrains every exterior DOF. Composite overset consumers constrain
    physical and blanked rows, retaining receptor rows for coupling equations.
    """
    form = phx.equations.FiniteElementForm(
        "moving-annulus-poisson",
        "u",
        (phx.equations.DiffusionAction("u", 1.0), phx.equations.SourceAction("u", 1.0)),
    )
    selected = (
        phx.discretization.dirichlet_constraint(field, "u")
        if constraint is None
        else constraint
    )
    if selected.field_name != "u":
        raise ValueError(
            f"native_moving_overset.poisson.constraint.field_name must equal 'u'; got {selected.field_name!r}"
        )
    return phx.equations.compile_finite_element_problem(
        form,
        field,
        constraint=selected,
        dirichlet_values=carried,
    )


@checked
def poisson_solve_policy(field: FiniteElementDiscretization) -> LinearSolvePolicy:
    """Return absolute solve accuracy below the independent system criterion."""
    return phx.linalg.LinearSolvePolicy(
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0,
            absolute=1e-11,
            max_steps=max(
                32, 4 * field.dof_maps[field.field_index("u")].global_dof_count
            ),
        )
    )


@checked
def poisson_update(
    field: FiniteElementDiscretization,
    carried: Array,
) -> tuple[Array, float]:
    problem = prepare_poisson_update(field, carried)
    system, rhs = problem.linear_system()
    if not isinstance(rhs, Array):
        raise TypeError(
            f"native_moving_overset.poisson.rhs must be a JAX array; got {type(rhs).__name__}"
        )
    result = phx.linalg.solve(system, rhs, policy=poisson_solve_policy(field))
    if not bool(jnp.all(result.successful)):
        raise RuntimeError(
            f"native_moving_overset.poisson.successful must hold for every solve; got {result.successful!r}"
        )
    if not isinstance(result.value, Array):
        raise TypeError(
            f"native_moving_overset.poisson.solution must be a JAX array; got {type(result.value).__name__}"
        )
    action = system.operator.mv(result.value)
    if not isinstance(action, Array):
        raise TypeError(
            f"native_moving_overset.poisson.operator_action must be a JAX array; got {type(action).__name__}"
        )
    residual = float(jnp.max(jnp.abs(action - rhs)))
    if residual > 1e-9:
        raise RuntimeError(
            f"native_moving_overset.poisson.residual must be at most 1e-9; got {residual!r}"
        )
    expanded = problem.expand(result.value)
    if not isinstance(expanded, Array):
        raise TypeError(
            f"native_moving_overset.poisson.coefficients must be a JAX array; got {type(expanded).__name__}"
        )
    return expanded, residual


@checked
def restart_native(
    connectivity: OversetConnectivity,
    coefficients: Mapping[str, Array],
    *,
    plans: Mapping[str, NativeMeshingPlan],
    declarations: Mapping[str, MeshingFiniteElementFieldDeclaration],
    path: str | PathLike[str] | None = None,
    archive_limits: ArrayArchiveLimits | None = None,
) -> tuple[
    OversetConnectivity,
    dict[str, Array],
    dict[str, NativeMeshingPlan],
    dict[str, FiniteElementDiscretization],
]:
    """Restore sources, hard requests, field declarations and state from an archive.

    Source-closure codecs serialize records. Prepared spatial connectivity,
    solver plans and field query caches are rebuilt after restore. This steady
    example retains one history-free coefficient field per part; stateful
    consumers must pass their named state to lifecycle.
    """

    limits = DEFAULT_ARRAY_ARCHIVE_LIMITS if archive_limits is None else archive_limits

    inputs = _prepare_native_restart(connectivity, coefficients, plans, declarations)
    if path is not None:
        return _restore_native_at(
            Path(path), inputs, limits, connectivity, coefficients, plans
        )
    with TemporaryDirectory(prefix="phydrax-native-source-overset-") as directory:
        return _restore_native_at(
            Path(directory) / "native-overset.source-closure.zip",
            inputs,
            limits,
            connectivity,
            coefficients,
            plans,
        )
