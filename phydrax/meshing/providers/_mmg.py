#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

import meshio
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._physical import SpatialCoordinateContract
from ...discretization import CellMesh, PolygonalConnectivity, TetrahedralConnectivity
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from .._audit import CellMeshAuditPolicy
from .._canonical import certify_cell_mesh
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
)
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import MeshMetricField
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)


@dataclass(frozen=True, slots=True)
class MmgOptions:
    """Mmg geometric tolerance in the mesh's coordinate units.

    An explicit executable must be the appropriate mmg2d, mmgs, or mmg3d
    program. Otherwise PATH and the optional pymmg binary wheel are searched.
    Metric sizes are adaptation targets, not hard output edge-length bounds.
    """

    hausdorff_distance: float = 0.01
    executable: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.hausdorff_distance) or self.hausdorff_distance <= 0:
            raise ValueError("hausdorff_distance must be positive and finite.")
        if self.executable is not None and not str(self.executable).strip():
            raise ValueError("executable must be non-empty or None.")


@dataclass(frozen=True, slots=True)
class MmgAdaptationPlan:
    mesh: CellMesh
    metric: MeshMetricField
    coordinate_contract: SpatialCoordinateContract
    options: MmgOptions
    limits: MeshingLimits
    audit_policy: CellMeshAuditPolicy
    plan_id: str = field(init=False)

    def __post_init__(self):
        _metric_rows(self.mesh, self.metric)
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if not isinstance(self.options, MmgOptions):
            raise TypeError("options must be MmgOptions.")
        if not isinstance(self.limits, MeshingLimits):
            raise TypeError("limits must be MeshingLimits.")
        if not isinstance(self.audit_policy, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy.")
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "mmg-adaptation-plan",
                    "mesh": self.mesh.mesh_id,
                    "metric": self.metric.metric_id,
                    "coordinates": self.coordinate_contract.spatial_id,
                    "hausdorff_distance": self.options.hausdorff_distance,
                    "executable": self.options.executable,
                    "limits": self.limits.limits_id,
                    "audit_policy": self.audit_policy.policy_id,
                }
            ),
        )

    def execute(self) -> CellMeshingResult:
        return MmgProvider(self.options).execute(self)


def _backend(mesh: CellMesh) -> str:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if len(mesh.blocks) != 1:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Mmg adaptation accepts one unlabelled simplex block; block/material transfer is unavailable.",
        )
    kinds = {block.cell_kind for block in mesh.blocks}
    if kinds == {"triangle"} and mesh.ambient_dimension in (2, 3):
        return "mmg2d" if mesh.ambient_dimension == 2 else "mmgs"
    if kinds == {"tetrahedron"} and mesh.ambient_dimension == 3:
        return "mmg3d"
    raise MeshingFailure(
        MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
        "Mmg requires affine triangles in 2D/3D or affine tetrahedra in 3D.",
    )


def _metric_rows(mesh: CellMesh, metric: MeshMetricField) -> np.ndarray:
    _backend(mesh)
    if not isinstance(metric, MeshMetricField):
        raise TypeError("metric must be MeshMetricField.")
    scope = metric.scope
    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    scope_ids = np.asarray(scope.entity_ids, dtype=np.int64)
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != 0
        or scope.entity_set_id != vertex_set.entity_set_id
        or not np.array_equal(scope_ids, np.sort(vertex_ids))
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Mmg metric must bind every vertex of this exact mesh and numeric revision.",
        )
    values = np.asarray(metric.values, dtype=np.float64)
    if values.shape[1:] != (mesh.ambient_dimension, mesh.ambient_dimension):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "Mmg requires an ambient-coordinate SPD metric at every vertex.",
        )
    eigenvalues = np.linalg.eigvalsh(values)
    if (
        np.min(eigenvalues) < (1.0 / metric.maximum_size**2) * (1.0 - 1.0e-10)
        or np.max(eigenvalues) > (1.0 / metric.minimum_size**2) * (1.0 + 1.0e-10)
        or np.any(
            np.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0])
            > metric.maximum_anisotropy * (1.0 + 1.0e-10)
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "Normalize the metric before adaptation: eigenvalues exceed its declared size/anisotropy bounds.",
        )
    # MeshingScope sorts IDs. Its values follow that order, not CellMesh row order.
    return values[np.searchsorted(scope_ids, vertex_ids)]


def _executable(name: str, override: str | None) -> str:
    if override is not None:
        resolved = shutil.which(override)
        if resolved is not None:
            return resolved
    else:
        for candidate in (name, f"{name}_O3"):
            resolved = shutil.which(candidate)
            if resolved is not None:
                return resolved
        module = find_spec(name)
        if module is not None and module.origin is not None:
            directory = Path(module.origin).parent
            for candidate in (f"{name}_O3", name, f"{name}_O3.exe", f"{name}.exe"):
                binary = directory / candidate
                if binary.is_file():
                    return str(binary)
    raise MeshingFailure(
        MeshingFailureCategory.PROVIDER_UNAVAILABLE,
        f"Cannot locate {name}; install pymmg>=1.0.0 or supply its native executable.",
    )


def _write_metric(path: Path, values: np.ndarray) -> None:
    dimension = values.shape[1]
    # Medit stores the LOWER triangular matrix by rows:
    # 2D: xx xy yy; 3D: xx xy yy xz yz zz (not Mmg's in-memory order).
    rows, columns = np.tril_indices(dimension)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            f"MeshVersionFormatted 2\nDimension {dimension}\nSolAtVertices\n{len(values)}\n1 3\n"
        )
        np.savetxt(stream, values[:, rows, columns], fmt="%.17g")
        stream.write("End\n")


def _fresh_ids(source_ids: np.ndarray, count: int) -> np.ndarray:
    start = int(np.max(source_ids, initial=-1)) + 1
    if start + count > np.iinfo(np.int64).max:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Generated mesh identities overflow int64.",
        )
    return np.arange(start, start + count, dtype=np.int64)


def _check_arrays(points: np.ndarray, cells: np.ndarray, limits: MeshingLimits) -> None:
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or not np.all(np.isfinite(points))
        or cells.ndim != 2
        or cells.shape[0] == 0
        or not np.issubdtype(cells.dtype, np.integer)
        or np.any(cells < 0)
        or np.any(cells >= len(points))
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Backend returned invalid vertices or connectivity.",
        )
    if (
        len(points) > limits.maximum_vertices
        or len(cells) > limits.maximum_cells
        or cells.size > limits.maximum_connectivity_entries
        or points.nbytes + cells.nbytes > limits.maximum_data_bytes
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Backend arrays exceed the requested meshing limits.",
        )


def _check_result_limits(result: CellMeshingResult, limits: MeshingLimits) -> None:
    counts = result.audit.entity_counts
    if (
        counts[1] > limits.maximum_edges
        or (len(counts) > 2 and counts[2] > limits.maximum_faces)
        or result.audit.connectivity_entries > limits.maximum_connectivity_entries
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Canonical mesh incidence exceeds the requested meshing limits.",
        )


def _identity_report(source: CellMesh, mesh: CellMesh, provider: str) -> AdapterReport:
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        provider,
        "phydrax-cell-mesh",
        source_id=source.mesh_id,
        target_id=mesh.mesh_id,
        coordinate_mapping=("identity",),
        assumptions=(
            "No entity correspondence or field-transfer map is supplied by this backend.",
            "Only unlabelled simplex geometry is converted; provider-generated feature flags are not imported.",
        ),
        losses=(
            AdapterLoss(
                "entity_global_ids",
                "import",
                "synthesized",
                "Generated output IDs are not source row/region identities; lineage is unknown.",
                changes_interpretation=False,
            ),
        ),
    )


class MmgProvider:
    """Native Mmg2D/MMGS/MMG3D metric adaptation without invented lineage."""

    def __init__(self, options: MmgOptions | None = None, /):
        self.options = MmgOptions() if options is None else options
        if not isinstance(self.options, MmgOptions):
            raise TypeError("options must be MmgOptions or None.")

    @property
    def info(self) -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "mmg",
            "runtime",
            "LGPL-3.0-or-later",
            operations=(MeshingOperation.REMESH_SURFACE, MeshingOperation.ADAPT_VOLUME),
            source_kinds=(MeshingSourceKind.CELL_MESH,),
            capabilities=(MeshingCapability.ANISOTROPIC_METRIC,),
            cell_kinds=("triangle", "tetrahedron"),
            dimensions=(2, 3),
            execution_modes=(MeshingExecutionMode.SUBPROCESS,),
        )

    @staticmethod
    def vertex_scope(mesh: CellMesh, /) -> MeshingScope:
        _backend(mesh)
        vertices = mesh.entity_set(0)
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            vertices.entity_ids,
        )

    def plan(
        self,
        mesh: CellMesh,
        metric: MeshMetricField,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        limits: MeshingLimits | None = None,
        audit_policy: CellMeshAuditPolicy | None = None,
    ) -> MmgAdaptationPlan:
        return MmgAdaptationPlan(
            mesh,
            metric,
            coordinate_contract,
            self.options,
            MeshingLimits() if limits is None else limits,
            CellMeshAuditPolicy() if audit_policy is None else audit_policy,
        )

    def adapt(
        self,
        mesh: CellMesh,
        metric: MeshMetricField,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        limits: MeshingLimits | None = None,
        audit_policy: CellMeshAuditPolicy | None = None,
    ) -> CellMeshingResult:
        return self.plan(
            mesh, metric, coordinate_contract, limits=limits, audit_policy=audit_policy
        ).execute()

    def execute(self, plan: MmgAdaptationPlan, /) -> CellMeshingResult:
        if not isinstance(plan, MmgAdaptationPlan):
            raise TypeError("plan must be MmgAdaptationPlan.")
        metric = _metric_rows(plan.mesh, plan.metric)
        name = _backend(plan.mesh)
        executable = _executable(name, plan.options.executable)
        source = plan.mesh
        limits = plan.limits
        points = np.asarray(source.coordinates, dtype=np.float64)
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in source.blocks]
        )
        _check_arrays(points, cells, limits)
        certify_cell_mesh(source, plan.coordinate_contract)
        meshio_kind = "tetra" if name == "mmg3d" else "triangle"
        with tempfile.TemporaryDirectory(prefix="phydrax-mmg-") as directory:
            root = Path(directory)
            input_path, output_path, metric_path = (
                root / "input.mesh",
                root / "output.mesh",
                root / "input.sol",
            )
            # Medit references describe regions, not global identities. One
            # unlabelled material is sent; arbitrary IDs must never become refs.
            meshio.write(
                input_path,
                meshio.Mesh(points, [(meshio_kind, cells)]),
                file_format="medit",
            )
            _write_metric(metric_path, metric)
            # Mmg requires hmin < hmax even for a constant input metric.
            # Round the clamping interval outward by one representable value.
            minimum = float(np.nextafter(plan.metric.minimum_size, 0.0))
            maximum = float(np.nextafter(plan.metric.maximum_size, np.inf))
            command = [
                executable,
                "-in",
                str(input_path),
                "-out",
                str(output_path),
                "-sol",
                str(metric_path),
                "-hmin",
                str(minimum),
                "-hmax",
                str(maximum),
                "-hgrad",
                str(plan.metric.maximum_gradation),
                "-hausd",
                str(plan.options.hausdorff_distance),
                "-v",
                "0",
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=limits.maximum_wall_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise MeshingFailure(
                    MeshingFailureCategory.TIMED_OUT, "Mmg exceeded maximum_wall_seconds."
                ) from error
            except OSError as error:
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_UNAVAILABLE, str(error)
                ) from error
            log = completed.stdout + completed.stderr
            if completed.returncode != 0 or not output_path.is_file():
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    f"{name} failed: {log[-12000:]}",
                    provider_code=str(completed.returncode),
                )
            if output_path.stat().st_size > limits.maximum_data_bytes:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Mmg output exceeds maximum_data_bytes.",
                )
            converted = meshio.read(output_path, file_format="medit")
        version_match = re.search(r"Release\s+([^\s(]+)", log)
        if version_match is None:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Mmg supplied no release identity in its execution banner.",
            )
        points = np.asarray(
            converted.points[:, : source.ambient_dimension], dtype=np.float64
        )
        blocks = [block.data for block in converted.cells if block.type == meshio_kind]
        if not blocks:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Mmg returned no requested simplex cells.",
            )
        cells = np.concatenate(blocks).astype(np.int64, copy=False)
        _check_arrays(points, cells, limits)
        # Native outputs may include vertices used only by discarded lower-dimensional cells.
        used, remapped = np.unique(cells, return_inverse=True)
        cells = remapped.reshape(cells.shape)
        points = points[used]
        if name == "mmg3d":
            determinants = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])
            negative = determinants < 0
            cells[negative, :2] = cells[negative, 1::-1]
        vertex_ids = _fresh_ids(np.asarray(source.vertex_global_ids), len(points))
        cell_ids = _fresh_ids(
            np.concatenate([np.asarray(block.global_ids) for block in source.blocks]),
            len(cells),
        )
        constructor = (
            CellMesh.from_tetrahedra if name == "mmg3d" else CellMesh.from_triangles
        )
        mesh = constructor(
            points,
            cells,
            vertex_global_ids=vertex_ids,
            cell_global_ids=cell_ids,
            numeric_version=plan.plan_id,
        )
        certified = certify_cell_mesh(
            mesh, plan.coordinate_contract, audit_policy=plan.audit_policy
        )
        _check_result_limits(certified, limits)
        connectivity = mesh.connectivity
        assert isinstance(connectivity, (PolygonalConnectivity, TetrahedralConnectivity))
        edges = np.asarray(connectivity.edges)
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        compliance = MeshingComplianceReport(
            plan.plan_id,
            requested=(
                ("metric_minimum_size", plan.metric.minimum_size),
                ("metric_maximum_size", plan.metric.maximum_size),
                ("metric_gradation", plan.metric.maximum_gradation),
            ),
            achieved=(
                ("minimum_edge", float(lengths.min())),
                ("maximum_edge", float(lengths.max())),
            ),
        )
        provider = self.info
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.CONTROL_RESOLUTION,
                    MeshingStageStatus.PASSED,
                    input_ids=(plan.metric.metric_id,),
                    output_ids=(plan.plan_id,),
                ),
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING
                    if name != "mmg3d"
                    else MeshingStageKind.VOLUME_FILL,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.mesh_id,),
                    output_ids=(mesh.mesh_id,),
                    created_count=len(cells),
                ),
                MeshingStageReport(
                    MeshingStageKind.TOPOLOGY_AUDIT,
                    MeshingStageStatus.PASSED,
                    output_ids=(certified.audit.report_id,),
                ),
                MeshingStageReport(
                    MeshingStageKind.SPECIFICATION_COMPLIANCE,
                    MeshingStageStatus.PASSED,
                    output_ids=(compliance.report_id,),
                ),
            )
        )
        return CellMeshingResult(
            mesh,
            certified.geometry,
            plan.coordinate_contract,
            certified.audit,
            certified.quality,
            compliance,
            trace,
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                f"{name} {version_match.group(1)}",
                MeshingExecutionMode.SUBPROCESS,
                deterministic=False,
                enforced_limits=(
                    "wall_time",
                    "output_vertices",
                    "output_cells",
                    "output_incidence",
                    "converted_arrays",
                ),
                unenforced_limits=("provider_workspace",),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            SemanticProvenance(
                {
                    "kind": "mmg-adaptation",
                    "plan": plan.plan_id,
                    "source": source.mesh_id,
                    "metric": plan.metric.metric_id,
                    "mesh": mesh.mesh_id,
                    "backend": name,
                    "version": version_match.group(1),
                    "lineage": "unknown",
                    "output_ids": "generated",
                }
            ),
            adapter_reports=(_identity_report(source, mesh, name),),
        )


__all__ = ["MmgAdaptationPlan", "MmgOptions", "MmgProvider"]
