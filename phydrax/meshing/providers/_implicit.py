#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import replace
from time import monotonic

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._tensor_support import PreparedTensorGrid
from ...geometry import CompiledGeometry, DesignState
from ...geometry.implicit import (
    discover_implicit_surface,
    ImplicitSurfacePlan,
    ImplicitSurfacePolicy,
    ImplicitSurfaceStatus,
)
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from ...typing import checked
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import audit_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
    SurfaceMeshingSpec,
)
from .._quality import evaluate_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._sizing import SizeControlStrength, UniformSizeControl
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


def _admit_implicit_specification(
    specification: SurfaceMeshingSpec, /
) -> UniformSizeControl:
    unsupported: list[str] = []
    target = specification.target
    if target.ambient_dimension != 3 or target.geometry_order != 1:
        unsupported.append("affine surfaces in ambient dimension three")
    if target.cell_families.allowed_transitions or target.cell_families.allow_mixed:
        unsupported.append("mixed-cell transition policies")
    if specification.planar_embedding is not None:
        unsupported.append("planar embeddings")
    if len(specification.size_controls) != 1 or not isinstance(
        specification.size_controls[0], UniformSizeControl
    ):
        unsupported.append("exactly one whole-surface uniform size control")
    else:
        control = specification.size_controls[0]
        if control.scope.scope_id != specification.scope.scope_id:
            unsupported.append("local size-control scopes")
    if specification.protected_features:
        unsupported.append("protected features")
    if specification.region_controls:
        unsupported.append("region controls")
    if specification.patch_controls:
        unsupported.append("patch/interface controls")
    if specification.periodic_constraints:
        unsupported.append("periodic constraints")
    if specification.layer_controls:
        unsupported.append("boundary-layer controls")
    if unsupported:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Native implicit meshing does not enforce: " + ", ".join(unsupported) + ".",
            provider_code="preflight",
        )
    # ty: ignore[invalid-return-type]
    return specification.size_controls[0]


def _bounded_implicit_policy(
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
            "Meshing limits cannot admit a minimal closed implicit surface.",
            provider_code="preflight",
        )
    return replace(
        policy,
        maximum_crossings=min(policy.maximum_crossings, vertex_capacity),
        maximum_vertices=vertex_capacity,
        maximum_faces=face_capacity,
    )


def _check_implicit_result_limits(
    vertices: np.ndarray,
    faces: np.ndarray,
    limits: MeshingLimits,
    /,
) -> None:
    face_count = faces.shape[0]
    if (
        vertices.shape[0] > limits.maximum_vertices
        or face_count > limits.maximum_faces
        or face_count > limits.maximum_cells
        or faces.size > limits.maximum_connectivity_entries
        or vertices.nbytes + faces.nbytes > limits.maximum_data_bytes
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Implicit meshing exceeded its declared entity or data limits.",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )
    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0)
    if np.unique(np.sort(edges, axis=1), axis=0).shape[0] > limits.maximum_edges:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Implicit meshing exceeded its declared edge limit.",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )


def _implicit_size_compliance(
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
        raise TypeError("Admitted implicit size control must be UniformSizeControl.")
    edges = np.concatenate(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    if not lengths.size or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Implicit meshing produced no finite positive edge-size evidence.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    vertex_minimum = np.full((vertices.shape[0],), np.inf)
    vertex_maximum = np.zeros((vertices.shape[0],), dtype=np.float64)
    np.minimum.at(vertex_minimum, edges[:, 0], lengths)
    np.minimum.at(vertex_minimum, edges[:, 1], lengths)
    np.maximum.at(vertex_maximum, edges[:, 0], lengths)
    np.maximum.at(vertex_maximum, edges[:, 1], lengths)
    active = np.isfinite(vertex_minimum) & (vertex_minimum > 0.0)
    growth = float(np.max(vertex_maximum[active] / vertex_minimum[active], initial=1.0))
    key = f"size:{control.control_id}"
    requested = [
        (f"{key}:target_size", control.target_size),
        (
            "size_compliance_absolute_tolerance",
            specification.size_compliance.absolute_tolerance,
        ),
        (
            "size_compliance_relative_tolerance",
            specification.size_compliance.relative_tolerance,
        ),
    ]
    optional = (
        ("minimum_size", control.minimum_size),
        ("maximum_size", control.maximum_size),
        ("maximum_growth_rate", control.maximum_growth_rate),
    )
    requested.extend(
        (f"{key}:{name}", value) for name, value in optional if value is not None
    )
    achieved = [
        ("minimum_face_area", minimum_face_area),
        ("maximum_implicit_residual", maximum_implicit_residual),
        (f"{key}:minimum_edge", float(np.min(lengths))),
        (f"{key}:maximum_edge", float(np.max(lengths))),
        (f"{key}:maximum_local_edge_ratio", growth),
    ]
    statistics: dict[str, float] = {}
    for statistic in specification.size_compliance.target_statistics:
        quantile = {"p50": 0.5, "p95": 0.95}[statistic]
        value = float(np.quantile(lengths, quantile))
        statistics[statistic] = value
        achieved.append((f"{key}:{statistic}_edge", value))

    issues: list[str] = []
    if control.strength is SizeControlStrength.HARD:
        policy = specification.size_compliance
        target_tolerance = policy.absolute_tolerance + (
            policy.relative_tolerance * abs(control.target_size)
        )
        for statistic, value in statistics.items():
            if abs(value - control.target_size) > target_tolerance:
                issues.append(f"target_size_{statistic}:{control.control_id}")
        if control.minimum_size is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.minimum_size)
            )
            if float(np.min(lengths)) < control.minimum_size - tolerance:
                issues.append(f"minimum_size:{control.control_id}")
        if control.maximum_size is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.maximum_size)
            )
            if float(np.max(lengths)) > control.maximum_size + tolerance:
                issues.append(f"maximum_size:{control.control_id}")
        if control.maximum_growth_rate is not None:
            tolerance = policy.absolute_tolerance + (
                policy.relative_tolerance * abs(control.maximum_growth_rate)
            )
            if growth > control.maximum_growth_rate + tolerance:
                issues.append(f"maximum_growth_rate:{control.control_id}")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _check_deadline(started: float, limits: MeshingLimits, /) -> None:
    if monotonic() - started > limits.maximum_wall_seconds:
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            "Implicit meshing exceeded its enforced wall-time limit.",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )


class ImplicitMeshingPlan(StrictModule, NonTrainableState):
    geometry: CompiledGeometry
    grid: PreparedTensorGrid
    specification: SurfaceMeshingSpec
    surface_plan: ImplicitSurfacePlan
    coordinate_contract: SpatialCoordinateContract
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: CompiledGeometry,
        grid: PreparedTensorGrid,
        specification: SurfaceMeshingSpec,
        surface_plan: ImplicitSurfacePlan,
        coordinate_contract: SpatialCoordinateContract,
        source_id: str,
        source_revision: str,
        /,
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        if not source or not revision:
            raise ValueError("Implicit meshing source identities must be non-empty.")
        kernel_matches = eqx.tree_equal(
            geometry.kernel,
            surface_plan.kernel,
            typematch=True,
        )
        if (
            geometry.schema != surface_plan.schema
            or not bool(np.asarray(kernel_matches))
            or source != surface_plan.source_id
        ):
            raise ValueError(
                "Implicit surface evidence must belong to the exact geometry and source."
            )
        self.geometry = geometry
        self.grid = grid
        self.specification = specification
        self.surface_plan = surface_plan
        self.coordinate_contract = coordinate_contract
        self.source_id = source
        self.source_revision = revision
        self.plan_id = canonical_fingerprint(
            {
                "kind": "implicit-meshing-plan",
                "surface_plan": surface_plan.plan_id,
                "specification": specification.specification_id,
                "coordinate_contract": coordinate_contract.spatial_id,
                "source_id": source,
                "source_revision": revision,
            }
        )

    def execute(self, state: DesignState | None = None, /) -> CellMeshingResult:
        """Realize, certify, and publish the fixed-topology surface for one state.

        Execution is an eager in-process host boundary: certification consumes the
        gradient-stopped primal realization, while ``result.geometry.coordinates``
        remains the JAX realization of ``state``. Differentiating an observable of
        those coordinates with respect to the design state therefore follows this
        plan's fixed topology and route. The wall-time limit is enforced at every
        stage boundary; no result is published after the deadline.
        """

        started = monotonic()
        limits = self.specification.limits
        selected = self.geometry.state if state is None else state
        realization = self.surface_plan.realize(selected)
        primal = jax.lax.stop_gradient(
            (
                realization.vertices,
                realization.evidence.status,
                realization.evidence.minimum_face_area,
            )
        )
        vertices_host = np.asarray(primal[0], dtype=np.float64)
        realization_status = int(np.asarray(primal[1]))
        minimum_face_area = float(np.asarray(primal[2]))
        faces_host = np.asarray(realization.faces, dtype=np.int32)
        _check_deadline(started, limits)
        if realization_status != int(ImplicitSurfaceStatus.SUCCESS):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Implicit surface realization was rejected by its runtime evidence.",
                provider_code=f"implicit_status:{realization_status}",
                stage=MeshingStageKind.SURFACE_MESHING.value,
            )
        _check_implicit_result_limits(vertices_host, faces_host, limits)
        centroids = jnp.mean(jnp.asarray(vertices_host)[faces_host], axis=1)
        residuals = jnp.abs(
            self.geometry.with_state(jax.lax.stop_gradient(selected)).boundary_field(
                centroids
            )
        )
        compliance = _implicit_size_compliance(
            self.specification,
            vertices_host,
            faces_host,
            minimum_face_area=minimum_face_area,
            maximum_implicit_residual=float(np.max(np.asarray(residuals))),
        )
        if not compliance.passed:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "; ".join(compliance.issues),
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        mesh = CellMesh.from_triangles(
            vertices_host,
            faces_host,
            numeric_version=self.source_revision,
        )
        metadata = SurfaceMetadata(
            source_id=self.source_id,
            source_revision=self.source_revision,
            coordinate_contract=self.coordinate_contract,
            provenance=("native-implicit-dual-surface", self.surface_plan.plan_id),
            cell_tags=tuple(
                "implicit-zero-set" for _ in range(mesh.blocks[0].cell_count)
            ),
        )
        boundary = SurfaceModel.from_triangles(
            mesh.coordinates,
            mesh.blocks[0].vertices,
            metadata,
            vertex_global_ids=mesh.vertex_global_ids,
            cell_global_ids=mesh.blocks[0].global_ids,
            numeric_version=self.source_revision,
            repair_orientation=False,
        )
        face_set = mesh.entity_set(2)
        association = GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            self.source_id,
            self.source_revision,
            face_set.entity_set_id,
            face_set.entity_ids,
            tuple("implicit-zero-set" for _ in range(face_set.count)),
            residuals,
            resolved=np.ones((face_set.count,), dtype=np.bool_),
            exact=False,
        )
        host_geometry = CellGeometrySpec.affine(mesh)
        quality_evaluation = evaluate_cell_quality(mesh, host_geometry.coordinates)
        audit = audit_cell_mesh(
            mesh,
            host_geometry,
            quality_evaluation,
            associations=(association,),
        )
        if not audit.passed:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "; ".join(audit.issues),
                stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            )
        _check_deadline(started, limits)
        stages = (
            MeshingStageReport(
                MeshingStageKind.SOURCE_INSPECTION,
                MeshingStageStatus.PASSED,
                input_ids=(self.source_revision,),
                output_ids=(self.surface_plan.plan_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.SURFACE_MESHING,
                MeshingStageStatus.PASSED,
                input_ids=(self.surface_plan.plan_id,),
                output_ids=(mesh.mesh_id,),
                created_count=mesh.blocks[0].cell_count,
            ),
            MeshingStageReport(
                MeshingStageKind.GEOMETRY_ASSOCIATION,
                MeshingStageStatus.PASSED,
                input_ids=(mesh.mesh_id,),
                output_ids=(association.association_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.TOPOLOGY_AUDIT,
                MeshingStageStatus.PASSED,
                input_ids=(mesh.topology_id,),
                output_ids=(audit.report_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.SPECIFICATION_COMPLIANCE,
                MeshingStageStatus.PASSED,
                input_ids=(self.specification.specification_id,),
                output_ids=(compliance.report_id,),
            ),
        )
        trace = MeshingTrace(stages)
        provider = NativeImplicitProvider.info()
        runtime = MeshingRuntimeInfo(
            provider.provider_id,
            provider.version,
            MeshingExecutionMode.IN_PROCESS,
            deterministic=True,
            enforced_limits=(
                "vertices",
                "edges",
                "faces",
                "cells",
                "connectivity_entries",
                "data_bytes",
                "wall_seconds",
                "grid_capacity",
                "surface_capacity",
                "projection",
            ),
        )
        provenance = SemanticProvenance(
            {
                "kind": "native-implicit-cell-mesh",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "surface_plan": self.surface_plan.plan_id,
                "plan": self.plan_id,
                "specification": self.specification.specification_id,
                "mesh": mesh.mesh_id,
            }
        )
        certified = CellMeshingResult(
            mesh,
            host_geometry,
            self.coordinate_contract,
            audit,
            audit.quality,
            compliance,
            trace,
            provider,
            runtime,
            MeshingDerivativeMode.FIXED_ROUTE_PIECEWISE,
            provenance,
            boundary=boundary,
            associations=(association,),
        )
        # The certified host coordinates are the primal of the realization; the
        # published leaf carries the realization itself so design derivatives flow.
        return eqx.tree_at(
            lambda result: result.geometry.coordinates,
            certified,
            realization.vertices,
        )


class NativeImplicitProvider:
    @staticmethod
    def info() -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "phydrax-implicit",
            "current",
            "Proprietary",
            operations=(MeshingOperation.MESH_SURFACE,),
            source_kinds=(MeshingSourceKind.IMPLICIT,),
            capabilities=(
                MeshingCapability.DETERMINISTIC,
                MeshingCapability.IMPLICIT_CONFORMING,
            ),
            cell_kinds=("triangle",),
            dimensions=(2,),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    @checked
    def plan(
        self,
        geometry: CompiledGeometry,
        grid: PreparedTensorGrid,
        specification: SurfaceMeshingSpec,
        /,
        *,
        source_id: str,
        source_revision: str,
        coordinate_contract: SpatialCoordinateContract,
        policy: ImplicitSurfacePolicy | None = None,
    ) -> ImplicitMeshingPlan:
        if specification.target.topological_dimension != 2 or set(
            (
                *specification.target.cell_families.required,
                *specification.target.cell_families.preferred,
            )
        ) != {"triangle"}:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Native implicit meshing requires a triangular surface target.",
            )
        _admit_implicit_specification(specification)
        if (
            not isinstance(geometry, CompiledGeometry)
            or not isinstance(grid, PreparedTensorGrid)
            or geometry.ambient_dimension != 3
            or len(grid.axes) != 3
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Native implicit meshing requires a compiled three-dimensional geometry and grid.",
            )
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        if (
            specification.scope.source_id != source
            or specification.scope.source_revision != revision
        ):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Implicit meshing scope does not bind the supplied source revision.",
            )
        selected_policy = ImplicitSurfacePolicy() if policy is None else policy
        if not isinstance(selected_policy, ImplicitSurfacePolicy):
            raise TypeError("policy must be ImplicitSurfacePolicy or None.")
        selected_policy = _bounded_implicit_policy(selected_policy, specification.limits)
        lattice_point_count = 1
        for axis in grid.structured_axes:
            lattice_point_count *= axis.point_coordinates.shape[0]
        if (
            lattice_point_count > selected_policy.maximum_lattice_points
            or lattice_point_count > specification.limits.maximum_data_bytes // (8 * 8)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Implicit lattice preparation exceeds its point or data budget.",
                provider_code="preflight",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        try:
            surface_plan = discover_implicit_surface(
                geometry,
                grid,
                policy=selected_policy,
                source_id=source,
            )
        except ValueError as error:
            message = str(error)
            if "maximum_" not in message and "exceeds policy" not in message:
                raise
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                message,
                provider_code="preflight",
                stage=MeshingStageKind.SURFACE_MESHING.value,
            ) from error
        return ImplicitMeshingPlan(
            geometry,
            grid,
            specification,
            surface_plan,
            coordinate_contract,
            source,
            revision,
        )


__all__ = ["ImplicitMeshingPlan", "NativeImplicitProvider"]
