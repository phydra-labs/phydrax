#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from enum import StrEnum
from importlib.metadata import version

import numpy as np

from ..._identity import SemanticProvenance
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._canonical import certify_cell_mesh
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._result import CellMeshingResult, MeshingRuntimeInfo
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


class SurfaceBooleanOperation(StrEnum):
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


class ManifoldProvider:
    """Closed oriented triangle-surface booleans using double-precision Manifold.

    Empty intersections are rejected: CellMeshingResult represents a nonempty
    carrier. Input semantic selections are not silently inherited across cuts.
    """

    @staticmethod
    def info() -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "manifold",
            version("manifold3d"),
            "Apache-2.0",
            operations=(MeshingOperation.BOOLEAN_SURFACE,),
            source_kinds=(MeshingSourceKind.SURFACE,),
            capabilities=(),
            cell_kinds=("triangle",),
            dimensions=(2,),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    def execute(
        self,
        left: SurfaceModel,
        right: SurfaceModel,
        operation: SurfaceBooleanOperation,
        /,
    ) -> CellMeshingResult:
        import manifold3d

        if not isinstance(left, SurfaceModel) or not isinstance(right, SurfaceModel):
            raise TypeError("Boolean operands must be SurfaceModel values.")
        if not isinstance(operation, SurfaceBooleanOperation):
            raise TypeError("operation must be SurfaceBooleanOperation.")
        contract = left.metadata.coordinate_contract
        if contract.spatial_id != right.metadata.coordinate_contract.spatial_id:
            raise ValueError(
                "Boolean operands require identical spatial coordinate contracts."
            )
        if left.selections or right.selections or left.interfaces or right.interfaces:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Boolean selection/interface transfer requires explicit lineage lowering.",
            )
        operands = []
        original_id = manifold3d.Manifold.reserve_ids(2)
        input_faces = []
        for index, surface in enumerate((left, right)):
            faces = np.concatenate(
                [np.asarray(block.vertices) for block in surface.mesh.blocks]
            )
            input_faces.append(faces)
            operand = manifold3d.Manifold(
                manifold3d.Mesh64(
                    np.array(
                        surface.mesh.coordinates, dtype=np.float64, order="C", copy=True
                    ),
                    np.ascontiguousarray(faces, dtype=np.uint64),
                    run_index=np.asarray((0, faces.size), dtype=np.uint64),
                    run_original_id=np.asarray((original_id + index,), dtype=np.uint32),
                    face_id=np.arange(len(faces), dtype=np.uint64),
                )
            )
            if operand.status() != manifold3d.Error.NoError:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    f"Manifold rejected the input surface: {operand.status().name}.",
                )
            operands.append(operand)
        if operation is SurfaceBooleanOperation.UNION:
            solid = operands[0] + operands[1]
        elif operation is SurfaceBooleanOperation.DIFFERENCE:
            solid = operands[0] - operands[1]
        else:
            solid = operands[0] ^ operands[1]
        if solid.status() != manifold3d.Error.NoError:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Manifold boolean failed: {solid.status().name}.",
            )
        arrays = solid.to_mesh64()
        if len(arrays.tri_verts) == 0:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Boolean result is empty; no nonempty cell carrier can be produced.",
            )
        provenance = SemanticProvenance(
            {
                "kind": "surface-boolean",
                "operation": operation.value,
                "left": left.mesh.mesh_id,
                "right": right.mesh.mesh_id,
            }
        )
        run_offsets = np.asarray(arrays.run_index, dtype=np.int64) // 3
        source_indices = np.repeat(
            np.asarray(arrays.run_original_id, dtype=np.int64) - original_id,
            np.diff(run_offsets),
        )
        source_faces = np.asarray(arrays.face_id, dtype=np.int64)
        if source_indices.shape != source_faces.shape or np.any(
            (source_indices < 0) | (source_indices > 1)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.LINEAGE_FAILED,
                "Invalid Manifold source-face runs.",
            )
        surfaces = (left, right)
        tags = []
        for source_index, face_index in zip(source_indices, source_faces, strict=True):
            source = surfaces[source_index]
            if face_index < 0 or face_index >= len(input_faces[source_index]):
                raise MeshingFailure(
                    MeshingFailureCategory.LINEAGE_FAILED,
                    "Invalid Manifold source-face index.",
                )
            tags.append(
                source.metadata.cell_tags[face_index]
                if source.metadata.cell_tags
                else source.metadata.source_id
            )
        boundary = SurfaceModel.from_triangles(
            np.asarray(arrays.vert_properties)[:, :3],
            np.asarray(arrays.tri_verts),
            SurfaceMetadata(
                source_id=provenance.semantic_id,
                source_revision="0",
                coordinate_contract=contract,
                provenance=("manifold", operation.value),
                cell_tags=tuple(tags),
            ),
        )
        associations = []
        target_entities = boundary.mesh.entity_set(2)
        target_points = np.asarray(boundary.mesh.coordinates)[
            np.asarray(arrays.tri_verts)
        ]
        for source_index, source in enumerate(surfaces):
            selected = np.flatnonzero(source_indices == source_index)
            if not selected.size:
                continue
            face_indices = source_faces[selected]
            source_points = np.asarray(source.mesh.coordinates)[
                input_faces[source_index][face_indices]
            ]
            normals = np.cross(
                source_points[:, 1] - source_points[:, 0],
                source_points[:, 2] - source_points[:, 0],
            )
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            residuals = np.max(
                np.abs(
                    np.sum(
                        (target_points[selected] - source_points[:, :1])
                        * normals[:, None],
                        axis=2,
                    )
                ),
                axis=1,
            )
            ids = np.concatenate(
                [np.asarray(block.global_ids) for block in source.mesh.blocks]
            )
            associations.append(
                GeometryAssociation(
                    GeometryAssociationKind.SURFACE,
                    source.metadata.source_id,
                    source.metadata.source_revision,
                    target_entities.entity_set_id,
                    np.asarray(target_entities.entity_ids)[selected],
                    tuple(str(value) for value in ids[face_indices]),
                    residuals,
                )
            )
        certified = certify_cell_mesh(
            boundary.mesh, contract, associations=tuple(associations)
        )
        provider = self.info()
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING,
                    MeshingStageStatus.PASSED,
                    input_ids=(left.mesh.mesh_id, right.mesh.mesh_id),
                    output_ids=(certified.mesh.mesh_id,),
                ),
                *certified.trace.stages,
            )
        )
        return CellMeshingResult(
            certified.mesh,
            certified.geometry,
            contract,
            certified.audit,
            certified.quality,
            certified.compliance,
            trace,
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                provider.version,
                MeshingExecutionMode.IN_PROCESS,
                deterministic=False,
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            provenance,
            boundary=boundary,
            associations=certified.associations,
        )


__all__ = ["ManifoldProvider", "SurfaceBooleanOperation"]
