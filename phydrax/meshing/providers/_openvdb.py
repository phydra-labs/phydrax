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
from ...typing import checked
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
class _FloatGrid(Protocol):
    def copyFromArray(
        self,
        array: NDArray[np.float32],
        *,
        ijk: tuple[int, int, int],
        tolerance: float,
    ) -> None: ...

    def convertToPolygons(
        self, *, isovalue: float, adaptivity: float
    ) -> tuple[NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]: ...

    def createLevelSetFromPolygons(
        self,
        points: NDArray[np.float32],
        *,
        triangles: NDArray[np.uint32],
        quads: NDArray[np.uint32],
        halfWidth: float,
    ) -> _FloatGrid: ...


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


class OpenVDBLevelSetRebuild(StrictModule, NonTrainableState):
    """Rebuild the isosurface as a narrow-band signed-distance level set.

    The isovalue surface is polygonized without adaptivity and converted back to
    a closed narrow-band level set of ``half_width`` voxels on each side (this is
    OpenVDB's LevelSetRebuild route: volume to mesh to level set). The rebuilt
    grid is negative inside the surface, its constant background is
    ``half_width`` voxel units outside the band, and extraction then runs at
    isovalue zero with the requested adaptivity.
    """

    half_width: float = eqx.field(static=True)
    operation_id: str = eqx.field(static=True)

    def __init__(self, *, half_width: float = 3.0) -> None:
        width = float(half_width)
        if not np.isfinite(width) or not 1.0 <= width <= np.finfo(np.float32).max:
            raise ValueError("half_width must be a float32-representable value >= 1.")
        self.half_width = width
        self.operation_id = canonical_fingerprint(
            {"kind": "openvdb-level-set-rebuild", "half_width": width}
        )


class OpenVDBMeshingSpec(StrictModule, NonTrainableState):
    """Scalar isovalue and native index-space adaptivity, not physical edge sizing.

    OpenVDB's float32 grid and polygonizer approximate the supplied scalar field.
    Adaptivity zero disables adaptive polygon merging; it does not make the
    extracted surface an exact interpolation of the original source geometry.
    An optional level-set rebuild re-distances the extracted surface first.
    """

    isovalue: float = eqx.field(static=True)
    adaptivity: float = eqx.field(static=True)
    rebuild: OpenVDBLevelSetRebuild | None
    specification_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        isovalue: float = 0.0,
        adaptivity: float = 0.0,
        rebuild: OpenVDBLevelSetRebuild | None = None,
    ) -> None:
        level = float(isovalue)
        adaptive = float(adaptivity)
        if not np.isfinite(level) or abs(level) > np.finfo(np.float32).max:
            raise ValueError("isovalue must be finite and representable in float32.")
        if not np.isfinite(adaptive) or not 0.0 <= adaptive <= 1.0:
            raise ValueError("adaptivity must lie in [0, 1].")
        if rebuild is not None and not isinstance(rebuild, OpenVDBLevelSetRebuild):
            raise TypeError("rebuild must be OpenVDBLevelSetRebuild or None.")
        self.isovalue = level
        self.adaptivity = adaptive
        self.rebuild = rebuild
        self.specification_id = canonical_fingerprint(
            {
                "kind": "openvdb-meshing-spec",
                "isovalue": level,
                "adaptivity": adaptive,
                "rebuild": None if rebuild is None else rebuild.operation_id,
            }
        )


def _extract_polygons(
    native: _FloatGrid,
    specification: OpenVDBMeshingSpec,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polygonize, optionally after a narrow-band level-set rebuild."""

    rebuild = specification.rebuild
    if rebuild is None:
        return native.convertToPolygons(
            isovalue=specification.isovalue,
            adaptivity=specification.adaptivity,
        )
    points, triangles, quads = native.convertToPolygons(
        isovalue=specification.isovalue,
        adaptivity=0.0,
    )
    if not len(triangles) and not len(quads):
        return points, triangles, quads
    rebuilt = native.createLevelSetFromPolygons(
        np.ascontiguousarray(points, dtype=np.float32),
        triangles=np.ascontiguousarray(triangles, dtype=np.uint32),
        quads=np.ascontiguousarray(quads, dtype=np.uint32),
        halfWidth=rebuild.half_width,
    )
    return rebuilt.convertToPolygons(
        isovalue=0.0,
        adaptivity=specification.adaptivity,
    )


class OpenVDBProvider:
    """Extract a real sparse scalar isosurface with OpenVDB 13.

    Active voxel values are lowered to FloatGrid brick by brick through the
    binding's dense-block ingestion (copyFromArray), without densifying the
    domain or issuing per-voxel calls. Inactive voxels and the exterior of the
    Morton box have the declared constant background value; an active sample
    exactly equal to the float32 background is stored as background. The
    OpenVDB 13 binding exposes no CSG or level-set filter tools (only per-value
    Python callbacks), so neither is offered here. Unknown background, periodic grids, vector fields, and
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

    @checked
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
        if isinstance(specification, SurfaceMeshingSpec):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "OpenVDB does not enforce physical edge size, protected-feature, "
                "periodic, or deterministic contracts; use OpenVDBMeshingSpec.",
            )
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
        active = (
            np.asarray(grid.voxel_active)
            & np.asarray(grid.brick_groups.group_active)[:, None]
        )
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
                    grid.brick_groups.group_keys[brick_slots],
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
                "indices": indices,
                "values": values,
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
                "OpenVDB FloatGrid requires dense-array ingestion, level-set "
                "construction, and NumPy polygon extraction.",
            )
        occupied, first, brick_rank = np.unique(
            brick_slots, return_index=True, return_inverse=True
        )
        blocks = np.full(
            (occupied.size, grid.brick_size, grid.brick_size, grid.brick_size),
            np.float32(background),
            dtype=np.float32,
        )
        blocks[(brick_rank, *local_coordinates.T)] = values.astype(np.float32)
        block_origins = brick_coordinates[first] * grid.brick_size
        try:
            # The binding ingests dense blocks only; one call per occupied brick.
            for origin, block in zip(block_origins.tolist(), blocks, strict=True):
                native.copyFromArray(block, ijk=tuple(origin), tolerance=0.0)
            points, triangles, quads = _extract_polygons(native, specification)
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
                "level_set_rebuild": None
                if specification.rebuild is None
                else {
                    "route": "volume-to-mesh-to-level-set",
                    "half_width_voxels": specification.rebuild.half_width,
                    "background_voxels": specification.rebuild.half_width,
                    "interior_sign": "negative",
                },
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


__all__ = ["OpenVDBLevelSetRebuild", "OpenVDBMeshingSpec", "OpenVDBProvider"]
