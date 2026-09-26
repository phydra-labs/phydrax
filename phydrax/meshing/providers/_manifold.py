#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from importlib.metadata import version
from importlib.util import find_spec

import numpy as np
from numpy.typing import ArrayLike

from ..._identity import SemanticProvenance
from ...discretization import CellMesh
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from .._association import GeometryAssociation, GeometryAssociationKind
from .._canonical import canonicalize_cell_mesh, certify_cell_mesh
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._organization import MeshAttribute, MeshAttributeRole
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


_BARYCENTRIC_PLAN = SmallLinearSolvePlan(2)
# Relative bound on backend-transferred values against independently recomputed
# barycentric interpolation on the reported source face.
_PROPERTY_TRANSFER_TOLERANCE = 1.0e-8


class SurfaceBooleanOperation(StrEnum):
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"


def _surface_faces(surface: SurfaceModel, /) -> np.ndarray:
    return np.concatenate([np.asarray(block.vertices) for block in surface.mesh.blocks])


def _validate_operands(
    operands: tuple[SurfaceModel, ...],
    operation: SurfaceBooleanOperation,
    /,
) -> None:
    if not isinstance(operands, tuple) or len(operands) < 2:
        raise TypeError("Boolean operands must be a tuple of at least two surfaces.")
    if not all(isinstance(surface, SurfaceModel) for surface in operands):
        raise TypeError("Boolean operands must be SurfaceModel values.")
    if not isinstance(operation, SurfaceBooleanOperation):
        raise TypeError("operation must be SurfaceBooleanOperation.")
    spatial = operands[0].metadata.coordinate_contract.spatial_id
    if any(
        surface.metadata.coordinate_contract.spatial_id != spatial for surface in operands
    ):
        raise ValueError(
            "Boolean operands require identical spatial coordinate contracts."
        )
    if any(surface.selections or surface.interfaces for surface in operands):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Boolean selection/interface transfer requires explicit lineage lowering.",
        )


def _vertex_property_columns(
    operands: tuple[SurfaceModel, ...],
    vertex_properties: Mapping[str, tuple[ArrayLike, ...]],
    /,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], tuple[np.ndarray, ...]]:
    """Validate named per-vertex operand properties into backend column blocks."""

    if not isinstance(vertex_properties, Mapping):
        raise TypeError("vertex_properties must map names to per-operand arrays.")
    names = tuple(sorted(str(name).strip() for name in vertex_properties))
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Vertex property names must be unique and non-empty.")
    blocks: list[list[np.ndarray]] = [[] for _ in operands]
    shapes: list[tuple[int, ...]] = []
    for name, values in sorted(
        (str(name).strip(), values) for name, values in vertex_properties.items()
    ):
        if not isinstance(values, tuple) or len(values) != len(operands):
            raise ValueError(f"Vertex property {name!r} needs one array per operand.")
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
        component_shape = arrays[0].shape[1:]
        for surface, array, block in zip(operands, arrays, blocks, strict=True):
            if (
                array.shape[:1] != surface.mesh.coordinates.shape[:1]
                or array.shape[1:] != component_shape
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    f"Vertex property {name!r} must be finite with one row per "
                    "operand vertex and one component shape."
                )
            block.append(array.reshape((array.shape[0], -1)))
        shapes.append(component_shape)
    columns = tuple(
        np.concatenate(block, axis=1)
        if block
        else np.zeros((surface.mesh.coordinates.shape[0], 0), dtype=np.float64)
        for surface, block in zip(operands, blocks, strict=True)
    )
    return names, tuple(shapes), columns


def _barycentric_interpolation(
    points: np.ndarray,
    triangles: np.ndarray,
    corner_values: np.ndarray,
    /,
) -> np.ndarray:
    """Interpolate corner values at points projected onto their triangles' planes."""

    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    offset = points - triangles[:, 0]
    gram = np.stack(
        (
            np.stack((np.sum(first * first, 1), np.sum(first * second, 1)), axis=-1),
            np.stack((np.sum(first * second, 1), np.sum(second * second, 1)), axis=-1),
        ),
        axis=-2,
    )
    right = np.stack((np.sum(first * offset, 1), np.sum(second * offset, 1)), axis=-1)
    solved = solve_small_linear(_BARYCENTRIC_PLAN, gram, right)
    if not np.all(np.asarray(solved.successful)):
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "Property transfer evidence requires nondegenerate source faces.",
        )
    weights = np.asarray(solved.value)
    return (
        (1.0 - weights[:, :1] - weights[:, 1:]) * corner_values[:, 0]
        + weights[:, :1] * corner_values[:, 1]
        + weights[:, 1:] * corner_values[:, 2]
    )


class ManifoldProvider:
    """Closed oriented n-ary triangle-surface booleans using double-precision Manifold.

    DIFFERENCE subtracts every later operand from the first. Empty results are
    rejected: CellMeshingResult represents a nonempty carrier. Input semantic
    selections are not silently inherited across cuts.

    Named vertex properties are carried by the backend as extra vertex channels.
    Along cut curves each output triangle keeps the values of its own source
    face, so the transferred data are published per face corner. Every
    transferred corner value is independently checked against barycentric
    interpolation of the source vertex values on the backend-reported source
    face; the maximum relative residual is recorded in the provenance.
    """

    @staticmethod
    def info() -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "manifold",
            version("manifold3d")
            if find_spec("manifold3d") is not None
            else "unavailable",
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
        operands: tuple[SurfaceModel, ...],
        operation: SurfaceBooleanOperation,
        /,
        *,
        vertex_properties: Mapping[str, tuple[ArrayLike, ...]] | None = None,
    ) -> CellMeshingResult:
        if find_spec("manifold3d") is None:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "Manifold booleans require the optional 'meshing-manifold' "
                "dependency group.",
            )
        import manifold3d

        _validate_operands(operands, operation)
        names, shapes, property_columns = _vertex_property_columns(
            operands, {} if vertex_properties is None else vertex_properties
        )
        contract = operands[0].metadata.coordinate_contract
        original_id = manifold3d.Manifold.reserve_ids(len(operands))
        input_faces = tuple(_surface_faces(surface) for surface in operands)
        solids = []
        for index, (surface, faces, columns) in enumerate(
            zip(operands, input_faces, property_columns, strict=True)
        ):
            solid = manifold3d.Manifold(
                manifold3d.Mesh64(
                    np.ascontiguousarray(
                        np.concatenate(
                            (np.asarray(surface.mesh.coordinates, np.float64), columns),
                            axis=1,
                        )
                    ),
                    np.ascontiguousarray(faces, dtype=np.uint64),
                    run_index=np.asarray((0, faces.size), dtype=np.uint64),
                    run_original_id=np.asarray((original_id + index,), dtype=np.uint32),
                    face_id=np.arange(len(faces), dtype=np.uint64),
                )
            )
            if solid.status() != manifold3d.Error.NoError:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    f"Manifold rejected the input surface: {solid.status().name}.",
                )
            solids.append(solid)
        match operation:
            case SurfaceBooleanOperation.UNION:
                kind = manifold3d.OpType.Add
            case SurfaceBooleanOperation.DIFFERENCE:
                kind = manifold3d.OpType.Subtract
            case SurfaceBooleanOperation.INTERSECTION:
                kind = manifold3d.OpType.Intersect
            case _:
                raise ValueError(f"Unsupported boolean operation {operation!r}.")
        solid = manifold3d.Manifold.batch_boolean(solids, kind)
        if solid.status() != manifold3d.Error.NoError:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Manifold boolean failed: {solid.status().name}.",
            )
        arrays = solid.to_mesh64()
        property_vertices = np.asarray(arrays.vert_properties, dtype=np.float64)
        property_faces = np.asarray(arrays.tri_verts, dtype=np.int64)
        if not len(property_faces):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Boolean result is empty; no nonempty cell carrier can be produced.",
            )
        run_offsets = np.asarray(arrays.run_index, dtype=np.int64) // 3
        source_indices = np.repeat(
            np.asarray(arrays.run_original_id, dtype=np.int64) - original_id,
            np.diff(run_offsets),
        )
        source_faces = np.asarray(arrays.face_id, dtype=np.int64)
        face_counts = np.asarray([len(faces) for faces in input_faces], dtype=np.int64)
        if (
            source_indices.shape != source_faces.shape
            or np.any((source_indices < 0) | (source_indices >= len(operands)))
            or np.any(
                (source_faces < 0)
                | (
                    source_faces
                    >= face_counts[np.clip(source_indices, 0, len(operands) - 1)]
                )
            )
        ):
            raise MeshingFailure(
                MeshingFailureCategory.LINEAGE_FAILED,
                "Invalid Manifold source-face runs.",
            )
        # Property channels split vertices along cuts; merge them back into one
        # geometric vertex per position for the manifold carrier.
        canonical = np.arange(property_vertices.shape[0], dtype=np.int64)
        canonical[np.asarray(arrays.merge_from_vert, dtype=np.int64)] = np.asarray(
            arrays.merge_to_vert, dtype=np.int64
        )
        used, geometric_faces = np.unique(canonical[property_faces], return_inverse=True)
        geometric_faces = geometric_faces.reshape(property_faces.shape)
        provenance_payload: dict[str, object] = {
            "kind": "surface-boolean",
            "operation": operation.value,
            "operands": [surface.mesh.mesh_id for surface in operands],
        }
        tags = tuple(
            operands[source].metadata.cell_tags[face]
            if operands[source].metadata.cell_tags
            else operands[source].metadata.source_id
            for source, face in zip(
                source_indices.tolist(), source_faces.tolist(), strict=True
            )
        )
        transfer = self._property_transfer(
            operands,
            input_faces,
            property_columns,
            names,
            shapes,
            property_vertices,
            property_faces,
            source_indices,
            source_faces,
        )
        if names:
            provenance_payload["vertex_property_transfer"] = {
                name: {
                    "route": "manifold-vertex-channel-barycentric-on-source-face",
                    "published_on": "face-corners",
                    "maximum_relative_interpolation_residual": residual,
                }
                for name, (_, residual) in zip(names, transfer, strict=True)
            }
        provenance = SemanticProvenance(provenance_payload)
        boundary = SurfaceModel.from_triangles(
            property_vertices[used, :3],
            geometric_faces,
            SurfaceMetadata(
                source_id=provenance.semantic_id,
                source_revision="0",
                coordinate_contract=contract,
                provenance=("manifold", operation.value),
                cell_tags=tags,
            ),
        )
        mesh = canonicalize_cell_mesh(boundary.mesh)
        associations = self._associations(
            operands,
            input_faces,
            mesh,
            boundary,
            geometric_faces,
            source_indices,
            source_faces,
        )
        attributes = self._property_attributes(mesh, boundary, names, transfer)
        certified = certify_cell_mesh(
            mesh,
            contract,
            associations=associations,
            attributes=attributes,
        )
        compliance = MeshingComplianceReport(
            provenance.semantic_id,
            achieved=tuple(
                (f"vertex_property:{name}:maximum_relative_interpolation_residual", r)
                for name, (_, r) in zip(names, transfer, strict=True)
            ),
        )
        provider = self.info()
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING,
                    MeshingStageStatus.PASSED,
                    input_ids=tuple(surface.mesh.mesh_id for surface in operands),
                    output_ids=(certified.mesh.mesh_id,),
                ),
                *certified.trace.stages[:-1],
                MeshingStageReport(
                    MeshingStageKind.SPECIFICATION_COMPLIANCE,
                    MeshingStageStatus.PASSED,
                    input_ids=(provenance.semantic_id,),
                    output_ids=(compliance.report_id,),
                ),
            )
        )
        return CellMeshingResult(
            certified.mesh,
            certified.geometry,
            contract,
            certified.audit,
            certified.quality,
            compliance,
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
            attributes=certified.attributes,
            associations=certified.associations,
        )

    @staticmethod
    def _property_transfer(
        operands: tuple[SurfaceModel, ...],
        input_faces: tuple[np.ndarray, ...],
        property_columns: tuple[np.ndarray, ...],
        names: tuple[str, ...],
        shapes: tuple[tuple[int, ...], ...],
        property_vertices: np.ndarray,
        property_faces: np.ndarray,
        source_indices: np.ndarray,
        source_faces: np.ndarray,
    ) -> tuple[tuple[np.ndarray, float], ...]:
        """Per-face-corner transferred values with independent interpolation residuals."""

        if not names:
            return ()
        width = property_columns[0].shape[1]
        corner_points = property_vertices[property_faces, :3]
        transferred = property_vertices[property_faces, 3 : 3 + width]
        source_triangles = np.empty(corner_points.shape, dtype=np.float64)
        source_values = np.empty(corner_points.shape[:1] + (3, width), dtype=np.float64)
        for index, surface in enumerate(operands):
            selected = source_indices == index
            vertices = input_faces[index][source_faces[selected]]
            source_triangles[selected] = np.asarray(surface.mesh.coordinates)[vertices]
            source_values[selected] = property_columns[index][vertices]
        expected = _barycentric_interpolation(
            corner_points.reshape((-1, 3)),
            np.repeat(source_triangles, 3, axis=0),
            np.repeat(source_values, 3, axis=0),
        ).reshape(transferred.shape)
        results = []
        start = 0
        for name, shape in zip(names, shapes, strict=True):
            size = int(np.prod(shape, dtype=np.int64))
            columns = slice(start, start + size)
            start += size
            scale = max(
                1.0,
                max(
                    float(np.max(np.abs(values[:, columns]), initial=0.0))
                    for values in property_columns
                ),
            )
            residual = float(
                np.max(np.abs(transferred[..., columns] - expected[..., columns])) / scale
            )
            if not np.isfinite(residual) or residual > _PROPERTY_TRANSFER_TOLERANCE:
                raise MeshingFailure(
                    MeshingFailureCategory.LINEAGE_FAILED,
                    f"Manifold vertex property {name!r} is not barycentric on its "
                    f"reported source face (relative residual {residual}).",
                )
            results.append(
                (
                    transferred[..., columns].reshape(transferred.shape[:2] + shape),
                    residual,
                )
            )
        return tuple(results)

    @staticmethod
    def _property_attributes(
        mesh: CellMesh,
        boundary: SurfaceModel,
        names: tuple[str, ...],
        transfer: tuple[tuple[np.ndarray, float], ...],
    ) -> tuple[MeshAttribute, ...]:
        if not names:
            return ()
        face_ids = np.asarray(boundary.mesh.blocks[0].global_ids, dtype=np.int64)
        order = np.argsort(face_ids, kind="stable")
        faces = mesh.entity_set(2)
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            faces.entity_set_id,
            face_ids,
        )
        return tuple(
            MeshAttribute(
                f"face_corner_{name}",
                MeshAttributeRole.USER,
                scope,
                values[order],
            )
            for name, (values, _) in zip(names, transfer, strict=True)
        )

    @staticmethod
    def _associations(
        operands: tuple[SurfaceModel, ...],
        input_faces: tuple[np.ndarray, ...],
        mesh: CellMesh,
        boundary: SurfaceModel,
        geometric_faces: np.ndarray,
        source_indices: np.ndarray,
        source_faces: np.ndarray,
    ) -> tuple[GeometryAssociation, ...]:
        associations = []
        target_set_id = mesh.entity_set(2).entity_set_id
        target_ids = np.asarray(boundary.mesh.blocks[0].global_ids, dtype=np.int64)
        target_points = np.asarray(boundary.mesh.coordinates)[geometric_faces]
        for source_index, source in enumerate(operands):
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
                    target_set_id,
                    target_ids[selected],
                    tuple(str(value) for value in ids[face_indices]),
                    residuals,
                )
            )
        return tuple(associations)


__all__ = ["ManifoldProvider", "SurfaceBooleanOperation"]
