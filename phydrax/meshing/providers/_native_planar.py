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
split chains are shared by construction rather than welded afterwards. An
admitted uniform median (one hard region median whose interval contains
``[h/2, h]``) is selected at preparation instead: source edges are cut into
equal pieces at the canonical lattice spacing and the meshcore seeds the carved
region with that equilateral lattice. When feature-forced quality refinement
pulls that lattice mesh's median below its interval (and nothing else fails),
one corrective construction (boundary pieces at ``h / sqrt(2)``, charged square
interior sites at ``0.55 h``) is prepared and built under the same request
budget and deadline; its preparation identity is emitted by execution and
supplies all source ancestry. The meshcore constrained Delaunay
triangulation removes holes and exterior, Ruppert/Chew refinement enforces the
area, minimum-angle and (uniform) edge-length requests in one call, and graded
free edges longer than the local target are split until the field is met
within the work and scratch budgets. Independent compliance measures every
published edge, statistic and angle under charged linear passes. Publication
certifies global embedding and coverage of the authored region loops and region
measure. Source-edge identities and rational split parameters compose through
native Steiner ancestry. Each binary64 carrier has a certified deviation bound
from its rational ancestry, published alongside the requested source-fidelity
bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import phydrax.discretization as cell_geometry_owner
import phydrax.discretization as cell_mesh_owner
import phydrax.geometry as triangulation_owner
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import MeshingProviderInfo, SurfaceMeshingSpec
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing.providers._native_route_policy import publication_policy

from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._coordinate_operands import CoordinateOperandCollector
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
    corrective_domain,
    measure_edge_lengths,
    median_repair_needed,
    organization_and_associations,
    planar_compliance,
    PreparedPlanarDomain,
    size_control_issues,
    source_arrays,
    source_carrier_evidence,
    triangulate,
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


@dataclass(frozen=True, slots=True)
class _PlanarCandidate:
    """Published topology of one triangulation and its once-measured edges."""

    used: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    constrained: np.ndarray
    constrained_sources: np.ndarray
    mesh: cell_mesh_owner.CellMesh
    edges: np.ndarray
    lengths: np.ndarray


def _planar_candidate(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedPlanarDomain,
    triangulation: triangulation_owner.ConstrainedDelaunayTriangulation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> _PlanarCandidate:
    """Compact one triangulation into its canonical mesh and measure its edges.

    The edges are the mesh topology's own unique edges; their lengths are
    measured once, charged, and shared by the median repair decision and
    compliance.
    """
    triangle_count = triangulation.triangles.shape[0]
    point_count = triangulation.points.shape[0]
    corners = 3 * triangle_count
    with scope.host_workspace() as storage:
        # Triangle, compact-index and segment-id banks with the vertex unique
        # sort (160 per triangle), compacted coordinates (32 per point), the
        # constrained-edge pairs, their unique sort and the source coverage
        # difference (192 per triangle), and the per-source-edge rows.
        storage.set_bound(
            352 * triangle_count
            + 32 * point_count
            + 32 * prepared.source_directions.shape[0]
            + 4096
        )
        # Two sorts over the corners (vertex unique, constrained-pair unique),
        # the source coverage difference, gathers and the entity-limit scan.
        scope.charge(
            work=2 * corners * ((max(corners, 2) - 1).bit_length() + 2)
            + 4 * point_count
            + 2 * prepared.source_directions.shape[0]
        )
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
            specification.limits,
            MeshingStageKind.SURFACE_MESHING,
            cell_kind="triangle",
        )
        del compact, segment_ids, missing
    with coordinate_phase(scope, specification.limits, budget):
        mesh = canonicalize_cell_mesh(
            cell_mesh_owner.CellMesh.from_triangles(
                vertices, triangles, numeric_version=source.source_revision
            ),
            budget=budget,
        )
    mesh_edges = mesh.connectivity.edges  # ty: ignore[unresolved-attribute]
    edge_count = mesh_edges.shape[0]
    with scope.host_workspace() as storage:
        # Host edge copy and its row sort (16 bytes per edge each).
        storage.set_bound(32 * edge_count + 4096)
        scope.charge(work=4 * edge_count)
        edges = np.sort(np.asarray(mesh_edges, dtype=np.int64), axis=1)
    return _PlanarCandidate(
        used,
        vertices,
        triangles,
        constrained,
        constrained_sources,
        mesh,
        edges,
        measure_edge_lengths(vertices, edges, scope=scope),
    )


def _candidate_banks(candidate: _PlanarCandidate, /) -> tuple[np.ndarray, ...]:
    """Host banks a candidate keeps after its construction phases close."""
    return (
        candidate.used,
        candidate.vertices,
        candidate.triangles,
        candidate.constrained,
        candidate.constrained_sources,
        candidate.edges,
        candidate.lengths,
    )


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
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> CellMeshingResult:
    """Triangulate, refine, associate, certify, and publish one planar domain.

    The admitted lattice construction is retained whenever it is compliant;
    an eligible undersized median (read from the candidate's once-measured
    edges) alone selects the corrective construction, built and measured under
    the same request budget, retaining the selected ancestry.
    """
    budget = coordinate_budget(specification.limits)
    budget.capture_coordinate_operands = coordinate_operands is not None
    budget.capture_polynomial_operands = (
        coordinate_operands is not None and coordinate_operands.capture_polynomials
    )
    budget.capture_signed_determinant_operands = (
        coordinate_operands is not None
        and coordinate_operands.capture_signed_determinants
    )
    budget.capture_affine_areas = (
        coordinate_operands is not None and coordinate_operands.capture_affine_areas
    )
    budget.capture_measure_operands = (
        coordinate_operands is not None and coordinate_operands.capture_measures
    )
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        started = scope.operation_started
        limits = specification.limits
        with scope.host_workspace() as initial_storage:
            with measure_phase(record_phase, "refinement"):
                triangulation, inserted = triangulate(
                    prepared, started, specification, scope=scope
                )
                initial_storage.retain_owner(triangulation)
            with measure_phase(record_phase, "construction"):
                candidate = _planar_candidate(
                    source,
                    specification,
                    prepared,
                    triangulation,
                    scope=scope,
                    budget=budget,
                )
                initial_storage.retain_owner(_candidate_banks(candidate))
            repair = median_repair_needed(
                prepared,
                specification,
                triangulation.evidence,
                candidate.lengths,
                scope=scope,
            )
        if repair:
            # Closing its workspace and dropping the last local references
            # releases the obsolete buffers before preparing another complex.
            del candidate, triangulation
            with measure_phase(record_phase, "refinement"):
                prepared = corrective_domain(
                    source, specification, prepared, started, scope
                )
                workspace.retain_owner(prepared)
                triangulation, inserted = triangulate(
                    prepared, started, specification, scope=scope
                )
                workspace.retain_owner(triangulation)
            with measure_phase(record_phase, "construction"):
                candidate = _planar_candidate(
                    source,
                    specification,
                    prepared,
                    triangulation,
                    scope=scope,
                    budget=budget,
                )
        else:
            workspace.retain_owner(triangulation)
        workspace.retain_owner(_candidate_banks(candidate))
        evidence = triangulation.evidence
        mesh, used = candidate.mesh, candidate.used
        constrained = candidate.constrained
        constrained_sources = candidate.constrained_sources
        with measure_phase(record_phase, "construction"):
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
            source_rows = prepared.source_points.shape[0] + prepared.source_edges.shape[0]
            with scope.host_workspace() as storage:
                # Oriented facet, mask and region-row banks of the domain.
                storage.set_bound(256 * source_rows + 4096)
                scope.charge(work=8 * source_rows)
                domain = prepared.domain(source)
                admissible = float(np.max(prepared.source_bounds, initial=0.0))
            workspace.retain_owner(domain)
        with measure_phase(record_phase, "compliance"):
            compliance = planar_compliance(
                specification,
                prepared,
                candidate.vertices,
                candidate.triangles,
                candidate.edges,
                candidate.lengths,
                (constrained, constrained_sources),
                (
                    evidence.minimum_angle_degrees,
                    evidence.steiner_count + inserted,
                    evidence.status,
                ),
                (float(np.max(source_deviations, initial=0.0)), admissible),
                scope=scope,
            )
            vertex_count = mesh.vertex_global_ids.shape[0]
            carrier_rows = prepared.source_edges.shape[0] + vertex_count
            with scope.host_workspace() as storage:
                # Host vertex ids and one requested/achieved key-value record
                # pair per source edge and carrier vertex.
                storage.set_bound(8 * vertex_count + 512 * carrier_rows + 4096)
                scope.charge(work=vertex_count + 4 * carrier_rows)
                source_requested, source_achieved = source_carrier_evidence(
                    prepared,
                    authority,
                    np.asarray(mesh.vertex_global_ids, dtype=np.int64),
                    carrier_deviations,
                    source_deviations,
                )
            workspace.retain_owner((source_requested, source_achieved))
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
                created_count=candidate.triangles.shape[0],
            ),
        )
        result = publish_native_result(
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
        if coordinate_operands is not None:
            # Capture only whitelisted reconstruction/incidence operands, not
            # any publication verdict earned by the completed scientific checks.
            with coordinate_phase(scope, limits, budget):
                coordinate_operands.capture(
                    result.mesh,
                    result.geometry,
                    budget,
                    scope,
                    coordinate_contract=result.coordinate_contract,
                )
        return result


__all__ = [
    "execute_planar_route",
    "planar_support_issues",
]
