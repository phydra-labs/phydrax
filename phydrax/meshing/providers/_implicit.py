#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native implicit-surface route: discovery, realization, certified publication.

``phydrax.geometry.implicit`` owns discovery and realization. The route runs
either the fixed-lattice discovery (`ImplicitSurfacePolicy`), whose realization
differentiates with respect to design parameters along the fixed route, or
adaptive error-controlled discovery (`AdaptiveImplicitSurfacePolicy`), which is
not differentiable. Both publish only after the surface acceptance schedule:
global embedding and two-sided source fidelity bounded through
`ImplicitBoundarySource`, certified where the field's distance bounds are
established and recorded as sampled otherwise.
"""

from __future__ import annotations

import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax.discretization as cell_mesh_owner
import phydrax.geometry as contracts_owner
import phydrax.geometry.design as schema_owner
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingDerivativeMode,
    MeshingLimits,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...geometry import (
    ImplicitBoundarySource,
    ImplicitProjectionBoundarySource,
)
from ...geometry.analytic._extended import TorusKernel
from ...geometry.analytic._primitives import BallKernel
from ...geometry.implicit import (
    AdaptiveImplicitSurface,
    AdaptiveImplicitSurfaceEvidence,
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
    discover_implicit_surface,
    ImplicitSurfacePlan,
    ImplicitSurfacePolicy,
    ImplicitSurfaceStatus,
)
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from ...typing import checked
from .._controls import FeatureKind
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._result import CellMeshingResult, MeshingComplianceReport
from .._sizing import UniformSizeControl
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from . import _native_sources as sources_owner
from ._native_publication import (
    coordinate_budget,
    coordinate_phase,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import (
    check_deadline,
    edge_size_evidence,
    simplex_entity_limits,
    uniform_size_compliance,
    unique_edges,
)
from ._native_route_policy import publication_policy


_ZERO_SET = "implicit-zero-set"


@checked
def implicit_support_issues(
    source: sources_owner.NativeImplicitSource, specification: SurfaceMeshingSpec, /
) -> list[str]:
    """Physical requests of one surface specification this route cannot enforce."""

    unsupported: list[str] = []
    target = specification.target
    families = target.cell_families
    if set((*families.required, *families.preferred)) != {"triangle"}:
        unsupported.append("a triangular surface target")
    if target.ambient_dimension != 3 or target.geometry_order != 1:
        unsupported.append("affine surfaces in ambient dimension three")
    if families.allowed_transitions or families.allow_mixed:
        unsupported.append("mixed-cell transition policies")
    if specification.planar_embedding is not None:
        unsupported.append("planar embeddings")
    match specification.size_controls:
        case (UniformSizeControl() as control,):
            if control.scope.scope_id != specification.scope.scope_id:
                unsupported.append("local size-control scopes")
        case _:
            unsupported.append("exactly one whole-surface uniform size control")
    if len(specification.protected_features) > 1 or any(
        feature.feature_kind is not FeatureKind.SURFACE
        or feature.scope.scope_id != specification.scope.scope_id
        for feature in specification.protected_features
    ):
        unsupported.append("one surface-fidelity feature over the whole surface")
    if specification.region_controls:
        unsupported.append("region controls")
    if specification.patch_controls:
        unsupported.append("patch/interface controls")
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")
    if specification.quality_target is not None:
        unsupported.append("mesh quality targets")
    if source.geometry.ambient_dimension != 3 or len(source.grid.axes) != 3:
        unsupported.append("a compiled three-dimensional geometry and lattice")
    return unsupported


def _bounded_policy(
    policy: ImplicitSurfacePolicy, limits: MeshingLimits, /
) -> ImplicitSurfacePolicy:
    bytes_per_vertex = 3 * np.dtype(np.float64).itemsize
    bytes_per_face = 3 * np.dtype(np.int32).itemsize
    data_entity_capacity = limits.maximum_data_bytes // (
        bytes_per_vertex + bytes_per_face
    )
    face_capacity = min(
        policy.maximum_faces,
        limits.maximum_faces,
        limits.maximum_cells,
        limits.maximum_connectivity_entries // 3,
        data_entity_capacity,
    )
    vertex_capacity = min(
        policy.maximum_vertices,
        limits.maximum_vertices,
        data_entity_capacity,
    )
    if face_capacity < 4 or vertex_capacity < 4 or limits.maximum_edges < 6:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            (
                "implicit.bounded_policy must admit a minimal closed implicit surface within declared limits; got "
                + "Meshing limits cannot admit a minimal closed implicit surface."
            ),
            provider_code="preflight",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
            requested=(
                ("maximum_faces", float(face_capacity)),
                ("maximum_vertices", float(vertex_capacity)),
                ("maximum_edges", float(limits.maximum_edges)),
            ),
            achieved=(
                ("minimal_closed_faces", 4.0),
                ("minimal_closed_vertices", 4.0),
                ("minimal_closed_edges", 6.0),
            ),
        )
    return replace(
        policy,
        maximum_crossings=min(policy.maximum_crossings, vertex_capacity),
        maximum_vertices=vertex_capacity,
        maximum_faces=face_capacity,
    )


@checked
def prepare_implicit_route(
    source: sources_owner.NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    policy: ImplicitSurfacePolicy,
    /,
) -> ImplicitSurfacePlan:
    """Discover and freeze the implicit surface topology within the budgets."""

    limits = specification.limits
    bounded = _bounded_policy(policy, limits)
    lattice_point_count = 1
    for axis in source.grid.structured_axes:
        lattice_point_count *= axis.point_coordinates.shape[0]
    # One field value and one coordinate triple per lattice point is the
    # minimal discovery scratch; each point is one geometry query.
    scratch_bytes = lattice_point_count * 4 * np.dtype(np.float64).itemsize
    if (
        lattice_point_count > bounded.maximum_lattice_points
        or lattice_point_count > limits.maximum_geometry_queries
        or scratch_bytes > limits.maximum_scratch_bytes
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            (
                "implicit.prepare_implicit_route must fit point, query, and scratch limits; got "
                + "Implicit lattice preparation exceeds its point, query, or scratch budget."
            ),
            provider_code="preflight",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
            requested=(
                ("maximum_lattice_points", float(bounded.maximum_lattice_points)),
                ("maximum_geometry_queries", float(limits.maximum_geometry_queries)),
                ("maximum_scratch_bytes", float(limits.maximum_scratch_bytes)),
            ),
            achieved=(
                ("lattice_points", float(lattice_point_count)),
                ("scratch_bytes", float(scratch_bytes)),
            ),
        )
    try:
        return discover_implicit_surface(
            source.geometry,
            source.grid,
            policy=bounded,
            source_id=source.source_id,
        )
    except ValueError as error:
        # Discovery reports exhausted capacities as ValueError; only those are
        # resource refusals, every other invalid input propagates unchanged.
        message = str(error)
        if "maximum_" not in message and "exceeds policy" not in message:
            raise
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            message,
            provider_code="preflight",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        ) from error


def _implicit_compliance(
    specification: SurfaceMeshingSpec,
    vertices: np.ndarray,
    faces: np.ndarray,
    /,
    *,
    minimum_face_area: float,
    maximum_implicit_residual: float,
) -> MeshingComplianceReport:
    control = specification.size_controls[0]
    if not isinstance(control, UniformSizeControl):
        raise TypeError(
            (
                "implicit.implicit_compliance must bind an admitted UniformSizeControl; got "
                + "Admitted implicit size control must be UniformSizeControl."
            )
        )
    lengths, growth = edge_size_evidence(vertices, unique_edges(faces, "triangle"))
    requested, achieved, issues = uniform_size_compliance(
        control, specification.size_compliance, lengths, growth
    )
    policy = specification.size_compliance
    requested.extend(
        (
            ("size_compliance_absolute_tolerance", policy.absolute_tolerance),
            ("size_compliance_relative_tolerance", policy.relative_tolerance),
        )
    )
    achieved.extend(
        (
            ("minimum_face_area", minimum_face_area),
            ("maximum_implicit_residual", maximum_implicit_residual),
        )
    )
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


@checked
def prepare_adaptive_implicit_route(
    source: sources_owner.NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    policy: AdaptiveImplicitSurfacePolicy,
    /,
    *,
    scope: NativeExecutionScope,
) -> AdaptiveImplicitSurface:
    """Discover the zero set by adaptive octree refinement within the budgets.

    The discovery box is the bounding box of the source lattice; face and
    field-evaluation capacities are clipped to the declared limits, whose
    exhaustion discovery reports as unresolved boxes, never as success.
    """
    budget = coordinate_budget(specification.limits)
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        limits = specification.limits
        domain = np.asarray(
            [
                (np.min(axis.point_coordinates), np.max(axis.point_coordinates))
                for axis in source.grid.structured_axes
            ],
            dtype=np.float64,
        ).T
        bounded = replace(
            policy,
            maximum_faces=min(
                policy.maximum_faces, limits.maximum_faces, limits.maximum_cells
            ),
            maximum_evaluations=min(
                policy.maximum_evaluations, limits.maximum_geometry_queries
            ),
        )
        with coordinate_phase(scope, specification.limits, budget):
            return discover_adaptive_implicit_surface(
                source.geometry,
                domain=domain,
                policy=bounded,
                source_id=source.source_id,
                budget=budget,
            )


def _certification_request(
    source: sources_owner.NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    geometry: contracts_owner.CompiledGeometry,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> NativeCertificationRequest:
    """Require certified two-sided fidelity for every accepted implicit surface.

    A fidelity feature declares the tolerance; otherwise the target size
    supplies the existing resolution-consistent bound. Sampled or unresolved
    evidence is not a successful surface certificate.
    """

    control = specification.size_controls[0]
    if not isinstance(control, UniformSizeControl):
        raise TypeError(
            (
                "implicit.implicit_compliance must bind an admitted UniformSizeControl; got "
                + "Admitted implicit size control must be UniformSizeControl."
            )
        )
    features = specification.protected_features
    tolerance = features[0].maximum_deviation if features else control.target_size
    # A boundary sample grid of half-diagonal tolerance / 4 leaves room for the
    # mesh deviation inside the covering radius of the source samples.
    spacing = (tolerance if tolerance > 0.0 else control.target_size) / (
        2.0 * math.sqrt(geometry.ambient_dimension)
    )

    if isinstance(geometry.kernel, (BallKernel, TorusKernel)):
        fidelity_source = ImplicitProjectionBoundarySource(
            AnalyticImplicitProfile(
                geometry,
                coordinate_contract,
                source_id=source.source_id,
                source_revision=source.source_revision,
                budget=budget,
            )
        )
    else:
        fidelity_source = ImplicitBoundarySource(
            geometry, source_id=source.source_id, spacing=spacing
        )
    request = NativeCertificationRequest(
        MeshCertificationSchedule("surface"),
        source.source_id,
        source.source_revision,
        specification.limits,
        fidelity_source=fidelity_source,
        fidelity_tolerance=tolerance,
    )
    return request


def _publish_surface(
    source: sources_owner.NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    geometry: contracts_owner.CompiledGeometry,
    vertices: np.ndarray,
    faces: np.ndarray,
    construction: tuple[MeshingStageReport, ...],
    provenance: dict[str, str],
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    /,
    *,
    started: float,
    derivative_mode: MeshingDerivativeMode,
    minimum_face_area: float,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Associate, certify, and publish one closed implicit surface."""
    budget = coordinate_budget(specification.limits)
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        limits = specification.limits
        simplex_entity_limits(
            vertices,
            faces,
            limits,
            MeshingStageKind.SURFACE_MESHING,
            cell_kind="triangle",
        )
        with measure_phase(record_phase, "construction"):
            mesh = cell_mesh_owner.CellMesh.from_triangles(
                vertices, faces, numeric_version=source.source_revision
            )
        with measure_phase(record_phase, "compliance"):
            # Only field evaluation crosses to the dynamic geometry kernel.
            centroids = np.mean(vertices[faces], axis=1)
            residuals = np.abs(
                np.asarray(
                    jax.device_get(geometry.boundary_field(jnp.asarray(centroids)))
                )
            )
            compliance = _implicit_compliance(
                specification,
                vertices,
                faces,
                minimum_face_area=minimum_face_area,
                maximum_implicit_residual=float(np.max(residuals)),
            )
        with measure_phase(record_phase, "construction"):
            metadata = SurfaceMetadata(
                source_id=source.source_id,
                source_revision=source.source_revision,
                coordinate_contract=coordinate_contract,
                provenance=(provenance["kind"], provenance["surface_plan"]),
                cell_tags=tuple(_ZERO_SET for _ in range(mesh.blocks[0].cell_count)),
            )
            boundary = SurfaceModel.from_triangles(
                mesh.coordinates,
                mesh.blocks[0].vertices,
                metadata,
                vertex_global_ids=mesh.vertex_global_ids,
                cell_global_ids=mesh.blocks[0].global_ids,
                numeric_version=source.source_revision,
                repair_orientation=False,
            )
        with measure_phase(record_phase, "geometry_association"):
            face_set = mesh.entity_set(2)
            association = GeometryAssociation(
                GeometryAssociationKind.IMPLICIT,
                source.source_id,
                source.source_revision,
                face_set.entity_set_id,
                face_set.entity_ids,
                tuple(_ZERO_SET for _ in range(face_set.count)),
                residuals,
                resolved=np.ones((face_set.count,), dtype=np.bool_),
                exact=False,
            )
        check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
        with coordinate_phase(scope, specification.limits, budget):
            request = _certification_request(
                source, specification, geometry, coordinate_contract, budget=budget
            )
        result = publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            construction,
            provider,
            {**provenance, "specification": specification.specification_id},
            request,
            # The realized dual surface is a closed manifold by construction.
            policy=publication_policy("implicit", derivative_mode=derivative_mode),
            organization=NativePublicationOrganization(
                boundary=boundary,
                associations=(association,),
            ),
            execution=NativePublicationExecution(scope, workspace, record_phase, budget),
        )
        check_deadline(started, limits, MeshingStageKind.SPECIFICATION_COMPLIANCE)
        return result


@checked
def execute_implicit_route(
    source: sources_owner.NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    surface_plan: ImplicitSurfacePlan,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    state: schema_owner.DesignState | None,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Realize, certify, and publish the fixed-topology surface for one state.

    Certification consumes the gradient-stopped primal realization, while
    ``result.geometry.coordinates`` remains the JAX realization of ``state``.
    """

    started = scope.operation_started
    limits = specification.limits
    selected = source.geometry.state if state is None else state
    with measure_phase(record_phase, "construction"):
        realization = surface_plan.realize(selected)
        primal = jax.lax.stop_gradient(
            (
                realization.vertices,
                realization.evidence.status,
                realization.evidence.minimum_face_area,
            )
        )
        primal_host, faces = jax.device_get((primal, realization.faces))
        vertices_host = np.asarray(primal_host[0], dtype=np.float64)
        realization_status = int(np.asarray(primal_host[1]))
        faces_host = np.asarray(faces, dtype=np.int32)
    check_deadline(started, limits, MeshingStageKind.SURFACE_MESHING)
    if realization_status != int(ImplicitSurfaceStatus.SUCCESS):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            (
                "implicit.execute_implicit_route must report successful implicit realization evidence; got "
                + "Implicit surface realization was rejected by its runtime evidence."
            ),
            provider_code=f"implicit_status:{realization_status}",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )
    construction = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.source_revision,),
            output_ids=(surface_plan.plan_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING,
            MeshingStageStatus.PASSED,
            input_ids=(surface_plan.plan_id,),
            output_ids=(surface_plan.plan_id,),
            created_count=faces_host.shape[0],
        ),
    )
    certified = _publish_surface(
        source,
        specification,
        source.geometry.with_state(jax.lax.stop_gradient(selected)),
        vertices_host,
        faces_host,
        construction,
        {
            "kind": "native-implicit-dual-surface",
            "route": "implicit_surface",
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "surface_plan": surface_plan.plan_id,
            "plan": plan_id,
            "specification": specification.specification_id,
        },
        coordinate_contract,
        provider,
        started=started,
        derivative_mode=MeshingDerivativeMode.FIXED_ROUTE_PIECEWISE,
        minimum_face_area=float(np.asarray(primal_host[2])),
        record_phase=record_phase,
        scope=scope,
        workspace=workspace,
    )
    # The certified host coordinates are the primal of the realization; the
    # published leaf carries the realization itself so design derivatives flow.
    return eqx.tree_at(
        lambda result: result.geometry.coordinates,
        certified,
        realization.vertices,
    )


def _discovery_quantities(
    evidence: AdaptiveImplicitSurfaceEvidence, /
) -> tuple[tuple[str, float], ...]:
    return (
        ("accuracy_certified", float(evidence.certified)),
        ("status_flags", float(evidence.status)),
        ("unresolved_boxes", float(evidence.unresolved_count)),
        ("leaf_count", float(evidence.leaf_count)),
        ("box_evaluations", float(evidence.box_evaluations)),
        ("point_evaluations", float(evidence.point_evaluations)),
        ("maximum_vertex_residual", evidence.maximum_vertex_residual),
    )


@checked
def execute_adaptive_implicit_route(
    source: sources_owner.NativeImplicitSource,
    specification: SurfaceMeshingSpec,
    surface: AdaptiveImplicitSurface,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Certify and publish the adaptively discovered surface.

    Refuses with the discovery evidence when no closed surface was extracted,
    or when certified fidelity was demanded and discovery did not certify its
    decomposition. The adaptive product is not differentiable.
    """

    started = scope.operation_started
    evidence = surface.evidence
    mesh = surface.mesh
    if mesh is None:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            (
                "implicit.execute_adaptive_implicit_route must publish a closed surface with certified discovery evidence; got "
                + f"Adaptive implicit discovery published no surface ({evidence.status_flags!r})."
            ),
            provider_code=f"adaptive_status:{evidence.status}",
            stage=MeshingStageKind.SURFACE_MESHING.value,
            achieved=_discovery_quantities(evidence),
        )
    geometry = source.geometry
    if not evidence.certified:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            (
                "implicit.execute_adaptive_implicit_route must publish a closed surface with certified discovery evidence; got "
                + f"Certified fidelity was demanded but adaptive discovery is {evidence.accuracy}."
            ),
            stage=MeshingStageKind.CERTIFICATION.value,
            requested=(("accuracy_certified", 1.0),),
            achieved=_discovery_quantities(evidence),
        )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    corners = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    construction = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.source_revision,),
            output_ids=(evidence.evidence_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING,
            MeshingStageStatus.PASSED,
            input_ids=(evidence.evidence_id,),
            output_ids=(evidence.evidence_id,),
            created_count=faces.shape[0],
        ),
    )
    return _publish_surface(
        source,
        specification,
        geometry,
        vertices,
        faces,
        construction,
        {
            "kind": "native-adaptive-implicit-surface",
            "route": "implicit_surface",
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "surface_plan": evidence.evidence_id,
            "discovery_accuracy": evidence.accuracy,
            "plan": plan_id,
            "specification": specification.specification_id,
        },
        coordinate_contract,
        provider,
        started=started,
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        minimum_face_area=float(np.min(areas)),
        record_phase=record_phase,
        scope=scope,
        workspace=workspace,
    )


__all__ = [
    "execute_adaptive_implicit_route",
    "execute_implicit_route",
    "implicit_support_issues",
    "prepare_adaptive_implicit_route",
    "prepare_implicit_route",
]
