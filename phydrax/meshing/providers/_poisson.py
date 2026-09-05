#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._audit import CellMeshAuditPolicy
from .._canonical import certify_cell_mesh
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
    SurfaceMeshingSpec,
)
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


def _open3d():
    try:
        import open3d
    except (ImportError, OSError) as exc:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "Poisson reconstruction requires the optional Open3D Python bindings "
            "(open3d>=0.19,<0.20).",
        ) from exc
    return open3d


class OrientedPointCloud(StrictModule, NonTrainableState):
    """Physical Cartesian samples with caller-supplied, consistently oriented normals.

    Normals are normalized, not estimated or reoriented. They must describe the
    intended surface orientation; reconstruction does not preserve sample IDs,
    sharp features, boundaries, labels, or an exact source surface.
    """

    coordinates: Array
    normals: Array
    coordinate_contract: SpatialCoordinateContract
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    cloud_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        normals: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        source_id: str,
        source_revision: str,
    ):
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if (
            coordinate_contract.length_coordinate_kind != "physical"
            or coordinate_contract.coordinate_system != "cartesian"
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Poisson reconstruction requires physical Cartesian coordinates.",
            )
        if np.iscomplexobj(coordinates) or np.iscomplexobj(normals):
            raise ValueError("Point-cloud coordinates and normals must be real.")
        points = np.asarray(coordinates, dtype=np.float64)
        vectors = np.asarray(normals, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
            raise ValueError("coordinates must have shape (count, 3), with count >= 4.")
        if vectors.shape != points.shape:
            raise ValueError("normals must have the same shape as coordinates.")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(vectors)):
            raise ValueError("Point-cloud coordinates and normals must be finite.")
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths == 0.0) or not np.all(np.isfinite(lengths)):
            raise ValueError("Point-cloud normals must have finite, nonzero lengths.")
        extent = float(np.max(np.max(points, axis=0) - np.min(points, axis=0)))
        if not np.isfinite(extent) or extent <= 0.0:
            raise ValueError("Point-cloud extent must be positive and finite.")
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        if not source or not revision:
            raise ValueError("Point-cloud source identities must be non-empty.")
        self.coordinates = jnp.asarray(points)
        self.normals = jnp.asarray(vectors / lengths[:, None])
        self.coordinate_contract = coordinate_contract
        self.source_id = source
        self.source_revision = revision
        self.cloud_id = canonical_fingerprint(
            {
                "kind": "oriented-point-cloud",
                "coordinates": np.asarray(self.coordinates).tolist(),
                "normals": np.asarray(self.normals).tolist(),
                "coordinate_contract": coordinate_contract.spatial_id,
                "source_id": source,
                "source_revision": revision,
            }
        )


class PoissonReconstructionSpec(StrictModule, NonTrainableState):
    """Screened Poisson octree controls, not edge-length or feature guarantees.

    Depth is an upper bound on adaptive octree depth. Scale is the reconstruction
    cube/sample bounding-cube diameter ratio. No geometric interpolation or
    maximum approximation-error guarantee is implied by either control.
    """

    depth: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    linear_fit: bool = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    def __init__(self, *, depth: int = 8, scale: float = 1.1, linear_fit: bool = False):
        depth_ = operator.index(depth)
        scale_ = float(scale)
        if isinstance(depth, bool) or not 2 <= depth_ <= 30:
            raise ValueError("Poisson octree depth must be an integer in [2, 30].")
        if not np.isfinite(scale_) or scale_ <= 1.0 or scale_ > np.finfo(np.float32).max:
            raise ValueError("Poisson scale must be representable in float32 and > 1.")
        # Open3D's native scale argument is float, not double.
        scale_ = float(np.float32(scale_))
        if scale_ <= 1.0:
            raise ValueError("Poisson scale must remain > 1 after float32 conversion.")
        self.depth = depth_
        self.scale = scale_
        self.linear_fit = bool(linear_fit)
        self.specification_id = canonical_fingerprint(
            {
                "kind": "poisson-reconstruction-spec",
                "depth": depth_,
                "scale": scale_,
                "linear_fit": self.linear_fit,
            }
        )


class PoissonProvider:
    """Real Open3D/Kazhdan screened Poisson surface reconstruction on the host."""

    @staticmethod
    def info() -> MeshingProviderInfo:
        backend = _open3d()
        return MeshingProviderInfo(
            "open3d-poisson",
            backend.__version__,
            "MIT",
            operations=(MeshingOperation.MESH_SURFACE,),
            source_kinds=(MeshingSourceKind.POINT_CLOUD,),
            capabilities=(),
            cell_kinds=("triangle",),
            dimensions=(2,),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    def execute(
        self,
        source: OrientedPointCloud,
        specification: PoissonReconstructionSpec,
        /,
        *,
        audit_policy: CellMeshAuditPolicy | None = None,
    ) -> CellMeshingResult:
        if not isinstance(source, OrientedPointCloud):
            raise TypeError("source must be OrientedPointCloud.")
        if isinstance(specification, SurfaceMeshingSpec):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Poisson does not enforce surface size, protected-feature, periodic, "
                "or deterministic meshing contracts; use PoissonReconstructionSpec.",
            )
        if not isinstance(specification, PoissonReconstructionSpec):
            raise TypeError("specification must be PoissonReconstructionSpec.")
        backend = _open3d()
        provider = self.info()
        points = np.asarray(source.coordinates, dtype=np.float64)
        # Reconstruct near the origin in dimensionless coordinates. The explicit
        # inverse mapping keeps native single-precision solver offsets out of the
        # physical reference frame and preserves the caller's length unit.
        lower = np.min(points, axis=0)
        extent = float(np.max(np.max(points, axis=0) - lower))
        cloud = backend.geometry.PointCloud()
        cloud.points = backend.utility.Vector3dVector((points - lower) / extent)
        cloud.normals = backend.utility.Vector3dVector(np.array(source.normals))
        try:
            reconstructed, _ = (
                backend.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    cloud,
                    depth=specification.depth,
                    scale=specification.scale,
                    linear_fit=specification.linear_fit,
                    n_threads=1,
                )
            )
        except RuntimeError as exc:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"Open3D screened Poisson reconstruction failed: {exc}",
                stage=MeshingStageKind.SURFACE_MESHING.value,
            ) from exc
        coordinates = np.asarray(reconstructed.vertices) * extent + lower
        faces = np.asarray(reconstructed.triangles)
        if not len(faces):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Poisson reconstruction produced no surface cells.",
            )
        provenance = SemanticProvenance(
            {
                "kind": "screened-poisson-reconstruction",
                "provider": provider.provider_id,
                "cloud": source.cloud_id,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "specification": specification.specification_id,
                "coordinate_contract": source.coordinate_contract.spatial_id,
                "sample_identity_preserved": False,
            }
        )
        boundary = SurfaceModel.from_triangles(
            coordinates,
            faces,
            SurfaceMetadata(
                source_id=provenance.semantic_id,
                source_revision="0",
                coordinate_contract=source.coordinate_contract,
                provenance=("open3d-screened-poisson", source.cloud_id),
            ),
        )
        certified = certify_cell_mesh(
            boundary.mesh,
            source.coordinate_contract,
            audit_policy=audit_policy,
        )
        compliance = MeshingComplianceReport(
            specification.specification_id,
            requested=(
                ("maximum_octree_depth", specification.depth),
                ("bounding_cube_scale", specification.scale),
            ),
            achieved=(
                ("vertex_count", certified.audit.vertex_count),
                ("triangle_count", len(faces)),
            ),
        )
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.SOURCE_INSPECTION,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.cloud_id,),
                    output_ids=(specification.specification_id,),
                ),
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.cloud_id, specification.specification_id),
                    output_ids=(boundary.mesh.mesh_id,),
                    created_count=len(faces),
                ),
                *certified.trace.stages[:-1],
                MeshingStageReport(
                    MeshingStageKind.SPECIFICATION_COMPLIANCE,
                    MeshingStageStatus.PASSED,
                    input_ids=(specification.specification_id,),
                    output_ids=(compliance.report_id,),
                ),
            )
        )
        return CellMeshingResult(
            certified.mesh,
            certified.geometry,
            source.coordinate_contract,
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
                enforced_limits=("octree_depth", "single_thread"),
                unenforced_limits=("wall_time", "memory"),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            provenance,
            boundary=boundary,
        )


__all__ = ["OrientedPointCloud", "PoissonReconstructionSpec", "PoissonProvider"]
