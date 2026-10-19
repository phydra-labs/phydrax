#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native planar route: graded size fields and refined constrained Delaunay.

Uniform size controls scoped to the planar region (``region:0``) or to source
edges (boundary/embedded patches) are resolved through `resolve_size_controls`
into one size field ``h(x) = min(h_region, min_e h_e + (g - 1) d_e(x))`` where
``d_e`` is the distance to the controlled source edge ``e`` and ``g`` the
smallest declared growth rate, so ``h`` is ``(g - 1)``-Lipschitz by
construction. Every source edge (region loops and embedded segments) is bisected
once until each piece is no longer than ``h`` at its ends and midpoint; the
split chains are shared by construction rather than welded afterwards. The
meshcore constrained Delaunay triangulation then removes holes and
exterior, Ruppert/Chew refinement enforces the area and minimum-angle
requests, and free edges longer than the local target are split until the
field is met within the work and scratch budgets. Publication certifies global
embedding and coverage of the authored region loops and region measure.
Source-edge identities and rational split parameters compose through native
Steiner ancestry. Each binary64 carrier has a certified deviation bound from
its rational ancestry, published alongside the requested source-fidelity bound.
Balanced interior sizing is a corrective construction only: execution retains
the established CDT when its applicable hard size statistics and quality pass.
Both constructions share one cumulative budget and deadline. The selected
preparation identity is emitted by execution and supplies all source ancestry;
the plan's initial preparation and a repaired execution need not share an ID.
"""

from __future__ import annotations

import numpy as np

import phydrax.discretization as cell_geometry_owner
import phydrax.discretization as cell_mesh_owner
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import MeshingProviderInfo, SurfaceMeshingSpec
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing.providers._native_route_policy import publication_policy

from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ...typing import checked
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._canonical import canonicalize_cell_mesh
from .._controls import FeatureKind
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from . import _native_sources as sources_owner
from ._native_planar_domain import (
    constrained_edges,
    constraint_coordinate_source,
    organization_and_associations,
    planar_compliance,
    PreparedPlanarDomain,
    select_triangulation,
    size_control_issues,
    source_arrays,
    source_carrier_evidence,
)
from ._native_publication import (
    coordinate_budget,
    coordinate_phase,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import check_deadline, simplex_entity_limits


@checked
def planar_support_issues(
    source: sources_owner.NativePlanarSource, specification: SurfaceMeshingSpec, /
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
        unsupported.append("the planar geometry region scope")
    vertices, edges, _ = source_arrays(source)
    unsupported.extend(size_control_issues(specification, edges.shape[0]))
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
            unsupported.append("protected features on planar geometry entities")
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


@checked
def execute_planar_route(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Triangulate, refine, associate, certify, and publish one planar domain."""
    budget = coordinate_budget(specification.limits)
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        started = scope.operation_started
        limits = specification.limits
        with measure_phase(record_phase, "refinement"):
            prepared, triangulation, inserted = select_triangulation(
                source, specification, prepared, started, scope, workspace
            )
        with measure_phase(record_phase, "construction"):
            evidence = triangulation.evidence
            triangles = np.asarray(triangulation.triangles, dtype=np.int64)
            used, compact = np.unique(triangles, return_inverse=True)
            vertices = np.asarray(triangulation.points, dtype=np.float64)[used]
            triangles = compact.reshape(triangles.shape).astype(np.int32)
            segment_ids = np.asarray(triangulation.segment_ids, dtype=np.int64)
            constrained, constrained_sources = constrained_edges(
                triangles, segment_ids, prepared.segment_sources
            )
            missing = np.setdiff1d(
                np.arange(prepared.source_directions.shape[0]), constrained_sources
            )
            if missing.size:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "source.edges must lie within the meshed planar region; got uncovered edges",
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
            with coordinate_phase(scope, specification.limits, budget):
                mesh = canonicalize_cell_mesh(
                    cell_mesh_owner.CellMesh.from_triangles(
                        vertices, triangles, numeric_version=source.source_revision
                    ),
                    budget=budget,
                )
            with coordinate_phase(scope, specification.limits, budget):
                authority, source_deviations, carrier_deviations = (
                    constraint_coordinate_source(
                        source, prepared, triangulation, used, budget=budget
                    )
                )
            with coordinate_phase(scope, specification.limits, budget):
                geometry = cell_geometry_owner.CellGeometrySpec.plc(
                    mesh, authority, budget=budget
                )
        with measure_phase(record_phase, "geometry_association"):
            with coordinate_phase(scope, specification.limits, budget):
                patches, labels, associations = organization_and_associations(
                    mesh,
                    source,
                    prepared,
                    constrained,
                    constrained_sources,
                    used,
                    authority,
                    budget=budget,
                )
            domain = prepared.domain(source)
            admissible = float(np.max(prepared.source_bounds, initial=0.0))
        with measure_phase(record_phase, "compliance"):
            compliance = planar_compliance(
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
                (float(np.max(source_deviations, initial=0.0)), admissible),
            )
            source_requested, source_achieved = source_carrier_evidence(
                prepared,
                authority,
                np.asarray(mesh.vertex_global_ids, dtype=np.int64),
                carrier_deviations,
                source_deviations,
            )
            compliance = MeshingComplianceReport(
                specification.specification_id,
                issues=compliance.issues,
                requested=(*compliance.requested, *source_requested),
                achieved=(*compliance.achieved, *source_achieved),
            )
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
            policy=publication_policy(
                "planar",
                audit_policy=CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=CellMeshAuditDisposition.REJECT,
                ),
            ),
            organization=NativePublicationOrganization(
                patches=patches, labels=labels, associations=associations
            ),
            execution=NativePublicationExecution(scope, workspace, record_phase, budget),
            geometry=geometry,
        )


__all__ = [
    "execute_planar_route",
    "planar_support_issues",
]
