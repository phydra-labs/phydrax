#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from importlib import import_module, metadata

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ...discretization import CellMesh, PolygonalConnectivity, TetrahedralConnectivity
from ...geometry.simplicial import TriangleMesh, TriangleMeshQueryIndex
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import CellMeshAuditPolicy
from .._canonical import certify_cell_mesh
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceDescriptor,
    MeshingSourceKind,
    ProviderSupportReport,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import SizeControlStrength, UniformSizeControl
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from ._mmg import _check_arrays, _check_result_limits, _fresh_ids, _identity_report


_FTETWILD_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class FTetWildOptions:
    """fTetWild parameters; envelope_distance uses the source coordinate units.

    The envelope is independently checked at output boundary vertices and face
    centroids. This sampled check is not a continuous Hausdorff certificate.
    stop_quality is the backend's AMIPS stopping target, not an acceptance test;
    audit_policy provides independently enforced cell-quality requirements.
    """

    envelope_distance: float = 0.001
    maximum_iterations: int = 80
    stop_quality: float = 10.0
    maximum_threads: int = 1
    skip_simplify: bool = False
    coarsen: bool = True

    def __post_init__(self):
        if not np.isfinite(self.envelope_distance) or self.envelope_distance <= 0:
            raise ValueError("envelope_distance must be positive and finite.")
        if not np.isfinite(self.stop_quality) or self.stop_quality <= 0:
            raise ValueError("stop_quality must be positive and finite.")
        if self.maximum_iterations <= 0 or self.maximum_threads <= 0:
            raise ValueError("maximum_iterations and maximum_threads must be positive.")


@dataclass(frozen=True, slots=True)
class FTetWildMeshingPlan:
    source: SurfaceModel
    specification: VolumeMeshingSpec
    options: FTetWildOptions
    audit_policy: CellMeshAuditPolicy
    support: ProviderSupportReport
    plan_id: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.options, FTetWildOptions):
            raise TypeError("options must be FTetWildOptions.")
        if not isinstance(self.audit_policy, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy.")
        actual_support = FTetWildProvider(self.options).validate(
            self.source, self.specification
        )
        actual_support.require_supported()
        if actual_support.report_id != self.support.report_id:
            raise ValueError(
                "support report is not bound to this source and specification."
            )
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "ftetwild-meshing-plan",
                    "source_mesh": self.source.mesh.mesh_id,
                    "source_model": self.source.model_id,
                    "specification": self.specification.specification_id,
                    "envelope_distance": self.options.envelope_distance,
                    "maximum_iterations": self.options.maximum_iterations,
                    "stop_quality": self.options.stop_quality,
                    "maximum_threads": self.options.maximum_threads,
                    "skip_simplify": self.options.skip_simplify,
                    "coarsen": self.options.coarsen,
                    "audit_policy": self.audit_policy.policy_id,
                }
            ),
        )

    def execute(self) -> CellMeshingResult:
        return FTetWildProvider(self.options).execute(self)


def _boundary(mesh: CellMesh, plan: FTetWildMeshingPlan) -> SurfaceModel:
    connectivity = mesh.connectivity
    assert isinstance(connectivity, TetrahedralConnectivity)
    boundary_mask = np.asarray(connectivity.boundary_faces)
    faces = np.asarray(connectivity.faces)[boundary_mask].copy()
    signs = np.zeros(len(connectivity.faces))
    np.add.at(
        signs,
        np.asarray(connectivity.cell_faces).ravel(),
        np.asarray(connectivity.cell_face_signs).ravel(),
    )
    negative = signs[boundary_mask] < 0
    faces[negative, :2] = faces[negative, 1::-1]
    used, remapped = np.unique(faces, return_inverse=True)
    return SurfaceModel.from_triangles(
        np.asarray(mesh.coordinates)[used],
        remapped.reshape(faces.shape),
        SurfaceMetadata(
            source_id=f"ftetwild:{mesh.mesh_id}",
            source_revision=plan.plan_id,
            coordinate_contract=plan.source.metadata.coordinate_contract,
            provenance=("ftetwild-generated-boundary", plan.source.mesh.mesh_id),
        ),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids)[used],
        cell_global_ids=np.asarray(mesh.entity_set(2).entity_ids)[boundary_mask],
        numeric_version=plan.plan_id,
    )


class FTetWildProvider:
    """Robust native tetrahedralization through wildmeshing's fTetWild binding.

    Only whole-surface, affine, unlabelled, soft uniform-size volume requests
    are supported. Source-face correspondence is nearest-surface evidence, not
    an exact constraint or lineage map. Exterior tetrahedra are always removed
    using the backend's input-surface winding-number filter.
    """

    def __init__(self, options: FTetWildOptions | None = None, /):
        self.options = FTetWildOptions() if options is None else options
        if not isinstance(self.options, FTetWildOptions):
            raise TypeError("options must be FTetWildOptions or None.")

    @property
    def info(self) -> MeshingProviderInfo:
        return MeshingProviderInfo(
            "ftetwild",
            "runtime",
            "MPL-2.0",
            operations=(MeshingOperation.MESH_VOLUME,),
            source_kinds=(MeshingSourceKind.SURFACE,),
            capabilities=(),
            cell_kinds=("tetrahedron",),
            dimensions=(3,),
            execution_modes=(MeshingExecutionMode.IN_PROCESS,),
        )

    @staticmethod
    def whole_scope(source: SurfaceModel, /) -> MeshingScope:
        if not isinstance(source, SurfaceModel):
            raise TypeError("source must be SurfaceModel.")
        faces = source.mesh.entity_set(2)
        return MeshingScope(
            source.metadata.source_id,
            source.metadata.source_revision,
            MeshingEntityKind.MESH,
            2,
            faces.entity_set_id,
            faces.entity_ids,
        )

    @staticmethod
    def inspect_source(source: SurfaceModel, /) -> MeshingSourceDescriptor:
        if not isinstance(source, SurfaceModel):
            raise TypeError("source must be SurfaceModel.")
        connectivity = source.mesh.connectivity
        assert isinstance(connectivity, PolygonalConnectivity)
        return MeshingSourceDescriptor(
            source.metadata.source_id,
            source.metadata.source_revision,
            MeshingSourceKind.SURFACE,
            2,
            3,
            closed=not bool(np.any(np.asarray(connectivity.boundary_edges))),
        )

    def validate(
        self, source: SurfaceModel, specification: VolumeMeshingSpec, /
    ) -> ProviderSupportReport:
        descriptor = self.inspect_source(source)
        if not isinstance(specification, VolumeMeshingSpec):
            raise TypeError("specification must be VolumeMeshingSpec.")
        unsupported = []
        scope = self.whole_scope(source)
        if specification.boundary_scope.scope_id != scope.scope_id:
            unsupported.append("fTetWild requires the entire exact source surface scope")
        target = specification.target
        kinds = {*target.cell_families.required, *target.cell_families.preferred}
        if (
            kinds != {"tetrahedron"}
            or target.ambient_dimension != 3
            or target.geometry_order != 1
        ):
            unsupported.append(
                "fTetWild requires affine tetrahedra in ambient dimension three"
            )
        if specification.fill_strategy is not VolumeFillStrategy.SIMPLEX:
            unsupported.append("fTetWild supports only simplex volume fill")
        if specification.deterministic:
            unsupported.append(
                "the native binding does not promise deterministic execution"
            )
        if (
            specification.protected_features
            or specification.region_controls
            or specification.region_seeds
            or specification.hole_seeds
            or specification.layer_controls
            or specification.periodic_constraints
        ):
            unsupported.append(
                "protected features, region/hole seeds, layers, and periodic controls are unsupported"
            )
        if source.selections or source.interfaces or source.metadata.cell_tags:
            unsupported.append(
                "fTetWild cannot preserve source selections, interfaces, or cell tags"
            )
        if len(specification.size_controls) != 1:
            unsupported.append("fTetWild requires one whole-surface uniform size target")
        for control in specification.size_controls:
            if not isinstance(control, UniformSizeControl):
                unsupported.append("fTetWild supports only uniform sizing")
            elif control.scope.scope_id != scope.scope_id:
                unsupported.append("fTetWild cannot apply local size scopes")
            elif control.strength is not SizeControlStrength.SOFT:
                unsupported.append(
                    "fTetWild uniform sizing is a soft target, not hard edge bounds"
                )
        return ProviderSupportReport(
            self.info,
            descriptor,
            specification,
            unsupported=tuple(unsupported),
            weakened_guarantees=(
                "sampled boundary envelope audit only",
                "source entity lineage unavailable",
                "native workspace and wall time are not bounded",
            ),
        )

    def plan(
        self,
        source: SurfaceModel,
        specification: VolumeMeshingSpec,
        /,
        *,
        audit_policy: CellMeshAuditPolicy | None = None,
    ) -> FTetWildMeshingPlan:
        return FTetWildMeshingPlan(
            source,
            specification,
            self.options,
            CellMeshAuditPolicy() if audit_policy is None else audit_policy,
            self.validate(source, specification),
        )

    def execute(self, plan: FTetWildMeshingPlan, /) -> CellMeshingResult:
        if not isinstance(plan, FTetWildMeshingPlan):
            raise TypeError("plan must be FTetWildMeshingPlan.")
        self.validate(plan.source, plan.specification).require_supported()
        try:
            native = import_module("wildmeshing")
            version = metadata.version("wildmeshing")
        except (ImportError, metadata.PackageNotFoundError) as error:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "Install wildmeshing>=0.4.1 for native fTetWild tetrahedralization.",
            ) from error
        source = plan.source.mesh
        points = np.asarray(source.coordinates, dtype=np.float64)
        faces = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32) for block in source.blocks]
        )
        limits = plan.specification.limits
        _check_arrays(points, faces, limits)
        diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
        if diagonal <= 0 or not np.isfinite(diagonal):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Source bounding-box diagonal is invalid.",
            )
        size = plan.specification.size_controls[0]
        assert isinstance(size, UniformSizeControl)
        # Native parameters are relative to the input bounding-box diagonal,
        # while the native contract carries physical coordinate-unit lengths.
        with _FTETWILD_LOCK:
            try:
                tetrahedralizer = native.Tetrahedralizer(
                    stop_quality=plan.options.stop_quality,
                    max_its=plan.options.maximum_iterations,
                    max_threads=plan.options.maximum_threads,
                    epsilon=plan.options.envelope_distance / diagonal,
                    edge_length_r=size.target_size / diagonal,
                    skip_simplify=plan.options.skip_simplify,
                    coarsen=plan.options.coarsen,
                )
                tetrahedralizer.set_log_level(6)
                tetrahedralizer.set_mesh(points, faces)
                tetrahedralizer.tetrahedralize()
                points_out, cells, _native_flags = tetrahedralizer.get_tet_mesh(
                    all_mesh=False,
                    use_input_for_wn=True,
                    manifold_surface=True,
                    correct_surface_orientation=True,
                )
            except (RuntimeError, ValueError, TypeError) as error:
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    f"fTetWild {version}: {error}",
                ) from error
        points_out = np.asarray(points_out, dtype=np.float64)
        cells = np.asarray(cells)
        _check_arrays(points_out, cells, limits)
        if points_out.shape[1] != 3 or cells.shape[1] != 4:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "fTetWild returned non-tetrahedral output.",
            )
        used, remapped = np.unique(cells, return_inverse=True)
        points_out = points_out[used]
        cells = remapped.reshape(cells.shape)
        determinants = np.linalg.det(points_out[cells[:, 1:]] - points_out[cells[:, :1]])
        negative = determinants < 0
        cells[negative, :2] = cells[negative, 1::-1]
        mesh = CellMesh.from_tetrahedra(
            points_out,
            cells,
            vertex_global_ids=_fresh_ids(
                np.asarray(source.vertex_global_ids), len(points_out)
            ),
            cell_global_ids=_fresh_ids(
                np.concatenate([np.asarray(block.global_ids) for block in source.blocks]),
                len(cells),
            ),
            numeric_version=plan.plan_id,
        )
        boundary = _boundary(mesh, plan)
        boundary_faces = np.asarray(boundary.mesh.blocks[0].vertices)
        boundary_points = np.asarray(boundary.mesh.coordinates)
        centroids = np.mean(boundary_points[boundary_faces], axis=1)
        query = TriangleMeshQueryIndex(
            TriangleMesh(points, faces, source_id=source.mesh_id)
        ).query(np.concatenate((boundary_points, centroids)))
        distances = np.asarray(query.distance)
        sampled_deviation = float(distances.max())
        tolerance = (
            128 * np.finfo(float).eps * max(diagonal, float(np.max(np.abs(points))))
        )
        if (
            not np.all(np.isfinite(distances))
            or sampled_deviation > plan.options.envelope_distance + tolerance
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Boundary deviation {sampled_deviation} exceeds envelope {plan.options.envelope_distance}.",
            )
        source_face_ids = np.concatenate(
            [np.asarray(block.global_ids) for block in source.blocks]
        )
        nearest_faces = source_face_ids[
            np.asarray(query.face_index)[len(boundary_points) :]
        ]
        # Nearest centroid candidates do not establish unique full-face
        # provenance: retain their residuals without claiming resolved labels.
        association = GeometryAssociation(
            GeometryAssociationKind.SURFACE,
            plan.source.metadata.source_id,
            plan.source.metadata.source_revision,
            mesh.entity_set(2).entity_set_id,
            boundary.mesh.blocks[0].global_ids,
            tuple(str(int(identifier)) for identifier in nearest_faces),
            distances[len(boundary_points) :],
            resolved=np.zeros(len(centroids), dtype=bool),
            exact=False,
        )
        certified = certify_cell_mesh(
            mesh,
            plan.source.metadata.coordinate_contract,
            audit_policy=plan.audit_policy,
            associations=(association,),
        )
        _check_result_limits(certified, limits)
        connectivity = mesh.connectivity
        assert isinstance(connectivity, TetrahedralConnectivity)
        edges = np.asarray(connectivity.edges)
        lengths = np.linalg.norm(
            points_out[edges[:, 1]] - points_out[edges[:, 0]], axis=1
        )
        compliance = MeshingComplianceReport(
            plan.specification.specification_id,
            requested=(
                ("soft_target_edge_length", size.target_size),
                ("boundary_envelope", plan.options.envelope_distance),
            ),
            achieved=(
                ("minimum_edge", float(lengths.min())),
                ("maximum_edge", float(lengths.max())),
                ("maximum_sampled_boundary_deviation", sampled_deviation),
            ),
        )
        provider = self.info
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.SOURCE_INSPECTION,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.mesh_id,),
                    output_ids=(plan.support.source_descriptor_id,),
                ),
                MeshingStageReport(
                    MeshingStageKind.VOLUME_FILL,
                    MeshingStageStatus.PASSED,
                    input_ids=(plan.plan_id,),
                    output_ids=(mesh.mesh_id,),
                    created_count=len(cells),
                ),
                MeshingStageReport(
                    MeshingStageKind.GEOMETRY_ASSOCIATION,
                    MeshingStageStatus.WARNING,
                    output_ids=(association.association_id,),
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
            plan.source.metadata.coordinate_contract,
            certified.audit,
            certified.quality,
            compliance,
            trace,
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                f"wildmeshing {version}",
                MeshingExecutionMode.IN_PROCESS,
                deterministic=False,
                enforced_limits=(
                    "output_vertices",
                    "output_cells",
                    "output_incidence",
                    "converted_arrays",
                ),
                unenforced_limits=("provider_workspace", "wall_time"),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            SemanticProvenance(
                {
                    "kind": "ftetwild-tetrahedralization",
                    "plan": plan.plan_id,
                    "source": source.mesh_id,
                    "mesh": mesh.mesh_id,
                    "binding_version": version,
                    "lineage": "unknown",
                    "output_ids": "generated",
                    "envelope_audit": "boundary vertices and face centroids",
                }
            ),
            boundary=boundary,
            associations=(association,),
            adapter_reports=(_identity_report(source, mesh, "ftetwild"),),
        )


__all__ = ["FTetWildMeshingPlan", "FTetWildOptions", "FTetWildProvider"]
