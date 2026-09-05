#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
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
)
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import audit_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
    SurfaceMeshingSpec,
)
from .._quality import evaluate_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
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
    ):
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("geometry must be CompiledGeometry.")
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be PreparedTensorGrid.")
        if not isinstance(specification, SurfaceMeshingSpec):
            raise TypeError("specification must be SurfaceMeshingSpec.")
        if not isinstance(surface_plan, ImplicitSurfacePlan):
            raise TypeError("surface_plan must be ImplicitSurfacePlan.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        if not source or not revision:
            raise ValueError("Implicit meshing source identities must be non-empty.")
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
        selected = self.geometry.state if state is None else state
        realization = self.surface_plan.realize(selected)
        if not bool(np.asarray(realization.evidence.accepted)):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Implicit surface realization was rejected by its runtime evidence.",
                stage=MeshingStageKind.SURFACE_MESHING.value,
            )
        mesh = CellMesh.from_triangles(
            realization.vertices,
            realization.faces,
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
        centroids = jnp.mean(
            mesh.coordinates[jnp.asarray(mesh.blocks[0].vertices, dtype=jnp.int32)],
            axis=1,
        )
        residuals = jnp.abs(self.geometry.with_state(selected).boundary_field(centroids))
        association = GeometryAssociation(
            GeometryAssociationKind.IMPLICIT,
            self.source_id,
            self.source_revision,
            face_set.entity_set_id,
            face_set.entity_ids,
            tuple("implicit-zero-set" for _ in range(face_set.count)),
            residuals,
            resolved=np.ones((face_set.count,), dtype=bool),
            exact=False,
        )
        geometry = CellGeometrySpec.affine(mesh)
        quality_evaluation = evaluate_cell_quality(mesh, geometry.coordinates)
        audit = audit_cell_mesh(
            mesh,
            geometry,
            quality_evaluation,
            associations=(association,),
        )
        if not audit.passed:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "; ".join(audit.issues),
                stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            )
        compliance = MeshingComplianceReport(
            self.specification.specification_id,
            achieved=(
                ("minimum_face_area", float(realization.evidence.minimum_face_area)),
                (
                    "maximum_implicit_residual",
                    float(np.max(np.asarray(residuals))),
                ),
            ),
        )
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
            enforced_limits=("grid_capacity", "surface_capacity", "projection"),
        )
        provenance = SemanticProvenance(
            {
                "kind": "native-implicit-cell-mesh",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "surface_plan": self.surface_plan.plan_id,
                "mesh": mesh.mesh_id,
            }
        )
        return CellMeshingResult(
            mesh,
            geometry,
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
        surface_plan = discover_implicit_surface(
            geometry,
            grid,
            policy=selected_policy,
            source_id=source,
        )
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
