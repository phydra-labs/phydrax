#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module
from typing import Protocol, runtime_checkable

import equinox as eqx
import numpy as np
from numpy.typing import NDArray

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spatial import morton_decode_integer, SparseVoxelField
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


@runtime_checkable
class _FloatGridAccessor(Protocol):
    def setValueOn(self, ijk: tuple[int, int, int], value: float, /) -> None: ...


@runtime_checkable
class _FloatGrid(Protocol):
    def getAccessor(self) -> _FloatGridAccessor: ...

    def convertToPolygons(
        self, *, isovalue: float, adaptivity: float
    ) -> tuple[NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]: ...


@runtime_checkable
class _OpenVDB(Protocol):
    @property
    def LIBRARY_VERSION(self) -> tuple[int, int, int]: ...

    def FloatGrid(self, background: float, /) -> _FloatGrid: ...


def _openvdb() -> _OpenVDB:
    try:
        backend = import_module("openvdb")
    except (ImportError, OSError) as exc:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "OpenVDB extraction requires OpenVDB 13 Python bindings with NumPy "
            "support (conda-forge: openvdb=13). The obsolete PyPI pyopenvdb "
            "package is not this binding.",
        ) from exc
    if not isinstance(backend, _OpenVDB) or backend.LIBRARY_VERSION[0] != 13:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "The supported OpenVDB 13 binding must expose LIBRARY_VERSION and FloatGrid.",
        )
    return backend


class OpenVDBMeshingSpec(StrictModule, NonTrainableState):
    """Scalar isovalue and native index-space adaptivity, not physical edge sizing.

    OpenVDB's float32 grid and polygonizer approximate the supplied scalar field.
    Adaptivity zero disables adaptive polygon merging; it does not make the
    extracted surface an exact interpolation of the original source geometry.
    """

    isovalue: float = eqx.field(static=True)
    adaptivity: float = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)

    def __init__(self, *, isovalue: float = 0.0, adaptivity: float = 0.0):
        level = float(isovalue)
        adaptive = float(adaptivity)
        if not np.isfinite(level) or abs(level) > np.finfo(np.float32).max:
            raise ValueError("isovalue must be finite and representable in float32.")
        if not np.isfinite(adaptive) or not 0.0 <= adaptive <= 1.0:
            raise ValueError("adaptivity must lie in [0, 1].")
        self.isovalue = level
        self.adaptivity = adaptive
        self.specification_id = canonical_fingerprint(
            {
                "kind": "openvdb-meshing-spec",
                "isovalue": level,
                "adaptivity": adaptive,
            }
        )


class OpenVDBProvider:
    """Extract a real sparse scalar isosurface with OpenVDB 13.

    Active voxel values are lowered to FloatGrid without densifying the domain.
    Inactive voxels and the exterior of the Morton box have the declared constant
    background value. Unknown background, periodic grids, vector fields, and
    nonphysical/non-Cartesian coordinates are rejected. Every integer VDB sample
    maps to the corresponding physical voxel *center*, including anisotropic
    spacing; no world-space output is transformed twice. Native quads are split
    into triangles without claiming source face or selection preservation.
    """

    @staticmethod
    def info() -> MeshingProviderInfo:
        backend = _openvdb()
        return MeshingProviderInfo(
            "openvdb",
            ".".join(map(str, backend.LIBRARY_VERSION)),
            "Apache-2.0",
            operations=(MeshingOperation.MESH_SURFACE,),
            source_kinds=(MeshingSourceKind.TENSOR_GRID,),
            capabilities=(),
            cell_kinds=("triangle",),
            dimensions=(2,),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    def execute(
        self,
        field: SparseVoxelField,
        coordinate_contract: SpatialCoordinateContract,
        specification: OpenVDBMeshingSpec,
        /,
        *,
        source_id: str,
        source_revision: str,
        audit_policy: CellMeshAuditPolicy | None = None,
    ) -> CellMeshingResult:
        if not isinstance(field, SparseVoxelField):
            raise TypeError("field must be SparseVoxelField.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if isinstance(specification, SurfaceMeshingSpec):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "OpenVDB does not enforce physical edge size, protected-feature, "
                "periodic, or deterministic contracts; use OpenVDBMeshingSpec.",
            )
        if not isinstance(specification, OpenVDBMeshingSpec):
            raise TypeError("specification must be OpenVDBMeshingSpec.")
        grid = field.grid
        address = grid.address_plan
        if (
            coordinate_contract.length_coordinate_kind != "physical"
            or coordinate_contract.coordinate_system != "cartesian"
            or grid.dimension != 3
            or field.values.ndim != 2
            or field.background_mode != "constant"
            or any(address.periodic_axes)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "OpenVDB requires a nonperiodic 3D scalar field with explicit "
                "constant background in physical Cartesian coordinates.",
            )
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        if not source or not revision:
            raise ValueError("Sparse-volume source identities must be non-empty.")
        active = np.asarray(grid.voxel_active) & np.asarray(grid.brick_active)[:, None]
        brick_slots, local_slots = np.nonzero(active)
        values = np.asarray(field.values)[active]
        if np.iscomplexobj(values) or np.iscomplexobj(field.background_value):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "OpenVDB scalar samples and background must be real.",
            )
        background = float(np.asarray(field.background_value))
        if (
            not len(values)
            or not np.all(np.isfinite(values))
            or not np.isfinite(background)
            or np.any(np.abs(values) > np.finfo(np.float32).max)
            or abs(background) > np.finfo(np.float32).max
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "OpenVDB requires nonempty finite scalar samples representable in float32.",
            )
        if float(np.float32(background)) == specification.isovalue:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The constant background cannot equal the extracted isovalue.",
            )
        if not bool(np.asarray(grid.evidence.successful)):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Sparse voxel preparation evidence was not accepted.",
            )
        if grid.brick_depth:
            brick_coordinates = np.asarray(
                morton_decode_integer(
                    grid.brick_codes[brick_slots],
                    3,
                    grid.brick_depth,
                )
            )
        else:
            brick_coordinates = np.zeros((len(brick_slots), 3), dtype=np.int64)
        local_coordinates = np.stack(
            np.unravel_index(
                local_slots,
                (grid.brick_size,) * 3,
            ),
            axis=1,
        )
        indices = brick_coordinates * grid.brick_size + local_coordinates
        source_fingerprint = canonical_fingerprint(
            {
                "kind": "openvdb-sparse-source",
                "grid": grid.grid_id,
                "values": values.tolist(),
                "background": background,
                "coordinate_contract": coordinate_contract.spatial_id,
                "source_id": source,
                "source_revision": revision,
            }
        )
        backend = _openvdb()
        provider = self.info()
        native = backend.FloatGrid(background)
        if not isinstance(native, _FloatGrid):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "OpenVDB FloatGrid requires getAccessor and NumPy polygon extraction.",
            )
        accessor = native.getAccessor()
        if not isinstance(accessor, _FloatGridAccessor):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "OpenVDB FloatGrid accessor requires setValueOn.",
            )
        try:
            for index, value in zip(indices, values, strict=True):
                accessor.setValueOn(
                    (int(index[0]), int(index[1]), int(index[2])), float(value)
                )
            points, triangles, quads = native.convertToPolygons(
                isovalue=specification.isovalue,
                adaptivity=specification.adaptivity,
            )
        except (RuntimeError, ValueError) as exc:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"OpenVDB sparse isosurface extraction failed: {exc}",
                stage=MeshingStageKind.SURFACE_MESHING.value,
            ) from exc
        if not len(triangles) and not len(quads):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "OpenVDB extraction produced no surface cells at the requested isovalue.",
            )
        faces = np.concatenate((triangles, quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
        # The native transform remains identity: returned world positions are
        # therefore index coordinates, mapped once to the substrate's centers.
        spacing = (
            np.asarray(address.upper) - np.asarray(address.lower)
        ) / address.resolution
        coordinates = (
            np.asarray(address.lower)
            + (np.asarray(points, dtype=np.float64) + 0.5) * spacing
        )
        provenance = SemanticProvenance(
            {
                "kind": "openvdb-isosurface",
                "provider": provider.provider_id,
                "source": source_fingerprint,
                "source_id": source,
                "source_revision": revision,
                "specification": specification.specification_id,
                "coordinate_contract": coordinate_contract.spatial_id,
                "native_scalar_dtype": "float32",
                "quad_triangulation": "diagonal-0-2",
                "exterior": "constant-background",
                "source_identity_preserved": False,
            }
        )
        boundary = SurfaceModel.from_triangles(
            coordinates,
            faces,
            SurfaceMetadata(
                source_id=provenance.semantic_id,
                source_revision="0",
                coordinate_contract=coordinate_contract,
                provenance=("openvdb-isosurface", source_fingerprint),
            ),
        )
        certified = certify_cell_mesh(
            boundary.mesh,
            coordinate_contract,
            audit_policy=audit_policy,
        )
        compliance = MeshingComplianceReport(
            specification.specification_id,
            requested=(
                ("isovalue", specification.isovalue),
                ("index_space_adaptivity", specification.adaptivity),
            ),
            achieved=(
                ("active_voxels", len(values)),
                ("vertex_count", certified.audit.vertex_count),
                ("triangle_count", len(faces)),
            ),
        )
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.SOURCE_INSPECTION,
                    MeshingStageStatus.PASSED,
                    input_ids=(source_fingerprint,),
                    output_ids=(specification.specification_id,),
                ),
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING,
                    MeshingStageStatus.PASSED,
                    input_ids=(source_fingerprint, specification.specification_id),
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
            coordinate_contract,
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
                unenforced_limits=("wall_time", "memory"),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            provenance,
            boundary=boundary,
        )


__all__ = ["OpenVDBMeshingSpec", "OpenVDBProvider"]
