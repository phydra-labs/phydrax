#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"Publish native quadrilateral and hexahedral meshes with source metadata and certificates."

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import replace
from itertools import accumulate
from typing import final, Literal, Protocol

import equinox as eqx
import jax
import numpy as np

from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    CellValidityPolicy,
    EntitySet,
    ExactPlcCellGeometrySource,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import ConstrainedDelaunayTriangulation
from phydrax.geometry._source_fidelity import MappedDomainBoundarySource
from phydrax.linalg import SmallLinearSolvePlan, solve_small_linear
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._host_banks import entities, family_host_array
from phydrax.typing import checked, parse

from ..._fingerprint import canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_geometry import (
    require_scalar_coordinate_element,
)
from ...discretization._cell_geometry_validity import (
    _ConstructedTensorGrid,
    cell_geometry_id,
)
from ...discretization._cell_mesh import (
    cell_block_construction_bounds,
    cell_block_construction_work,
    cell_mesh_construction_bounds,
    cell_mesh_construction_work,
)
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from .._association import MappedReferenceAssociationTransfer
from .._association_routes import AssociationPropagationError
from .._dual_extraction import (
    DualExtraction,
    SurfaceCrossField,
)
from .._family_execution import family_route_refusals
from .._hex_dominant import extract_hex_dominant
from .._hex_frame_map import (
    _extract_source_block_tensor_grid,
    _extract_source_tensor_frame_grid,
    prepare_source_block_connection,
    prepare_source_hex_frame,
)
from .._hex_frame_records import (
    _SourceBlockTensorGrid,
    PreparedSourceHexFrame,
    SourceBlockConnection,
)
from .._hex_generation import (
    _HOST_ENTRY_BYTES,
    exact_mapped_volume_quality,
    extract_volume_hexes,
    generate_integer_grid_hexes,
    GridHexConstruction,
    mapped_source_lipschitz_bound,
    MappedGridHexConstruction,
    NativeHexGridSchedule,
    realize_mapped_grid_hexes,
)
from .._mapped_reference_association import _tensor_reference_associations
from .._measurements import (
    measure_phase,
    NativeMeshingPhaseRecorder,
)
from .._organization_records import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from .._plc_mapped_support import certify_mapped_plc_associations
from .._quad_generation import (
    extract_surface_quads,
    prepare_surface_cross_field,
    remap_dual_metadata,
)
from .._quality import evaluate_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import UniformSizeControl
from .._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus
from .._volume_generation import (
    declared_plc_domain,
    generate_plc_volume,
    NativeVolumeSchedule,
    prepare_plc_source,
    VolumeConstruction,
)
from . import _native_sources as sources_owner, _native_volume as volume_owner
from ._native_planar import planar_support_issues
from ._native_planar_domain import (
    _FIELD_WORKING_ENTRIES,
    constrained_edges,
    constraint_coordinate_source,
    control_edges,
    graded_size_compliance,
    organization_and_associations,
    PreparedPlanarDomain,
    source_carrier_evidence,
    triangulate,
)
from ._native_plc_geometry import exact_plc_geometry
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
    uniform_size_compliance,
)
from ._native_route_metadata import mesh_entity_scope, NativeRouteMetadata
from ._native_route_policy import publication_policy


VolumeExtractionRoute = Literal["plc_dual_hex", "plc_hex_dominant"]


def _triangle_scaffold(specification: SurfaceMeshingSpec, /) -> SurfaceMeshingSpec:
    """Internal decomposition request; never published as the user's result."""
    return SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            specification.target.ambient_dimension,
            CellFamilyPolicy(required=("triangle",)),
            geometry_order=specification.target.geometry_order,
        ),
        specification.scope,
        planar_embedding=specification.planar_embedding,
        size_controls=specification.size_controls,
        protected_features=specification.protected_features,
        background_metric=specification.background_metric,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        periodic_constraints=specification.periodic_constraints,
        layer_controls=specification.layer_controls,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        quality_target=specification.quality_target,
        deterministic=specification.deterministic,
    )


def _tetrahedron_scaffold(specification: VolumeMeshingSpec, /) -> VolumeMeshingSpec:
    "Keep the declared sizing as a conservative decomposition preference.\n\n    A dual hex edge is at most half its parent tetrahedron's longest edge.\n    This sufficient upper bound does not prescribe a statistical target,\n    guarantee a child minimum, or justify doubling every construction size.\n    Final acceptance evaluates the unchanged request on the output.\n"
    return VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            specification.target.ambient_dimension,
            CellFamilyPolicy(required=("tetrahedron",)),
            geometry_order=specification.target.geometry_order,
        ),
        specification.boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=specification.size_controls,
        protected_features=specification.protected_features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=specification.layer_controls,
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        deterministic=specification.deterministic,
    )


def _grouped_rows(values: np.ndarray, /) -> dict[int, np.ndarray]:
    """Ascending rows of each present value, in increasing value order.

    Each group equals ``np.flatnonzero(values == value)``; one stable sort
    replaces a full comparison scan per value.
    """
    if values.size == 0:
        return {}
    order = np.argsort(values, kind="stable")
    present, starts = np.unique(values[order], return_index=True)
    return dict(zip(present.tolist(), np.split(order, starts[1:]), strict=True))


def _size_evidence_work(edges: int, vertices: int, statistics: int, /) -> int:
    """Host work of ``edge_size_evidence`` and ``uniform_size_compliance``.

    Per edge: its row copy, endpoint gathers and difference, its length, the
    finite and positive checks, four incident extrema updates, the minimum,
    maximum and hard-bound reductions, and one partition per target
    statistic; per vertex: the coordinate copy and the two extrema banks with
    their ratio.
    """
    return edges * (16 + statistics) + 6 * vertices


@checked
def dual_quad_support_issues(
    source: sources_owner.NativePlanarSource, specification: SurfaceMeshingSpec, /
) -> list[str]:
    """Admit pure quads and quad-dominant policies served by the same pure route.

    An allowed triangle remainder is optional, not an obligation to introduce
    triangles. A required triangle family is different and cannot be fulfilled
    by this extraction. Required all-quad requests never publish the scaffold.
    """
    issues = planar_support_issues(source, _triangle_scaffold(specification))
    policy = specification.target.cell_families
    requested = set((*policy.required, *policy.preferred))
    if (
        "quadrilateral" not in requested
        or set(policy.required) - {"quadrilateral"}
        or requested - {"quadrilateral", "triangle"}
        or set(policy.allowed_transitions) - {"triangle", "quadrilateral"}
    ):
        issues.append(
            "a required or preferred quadrilateral target with only optional triangle transitions"
        )
    return issues


@checked
def dual_hex_support_issues(
    source: sources_owner.NativePlcSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    issues = volume_owner.volume_support_issues(
        source, _tetrahedron_scaffold(specification)
    )
    policy = specification.target.cell_families
    if (
        set((*policy.required, *policy.preferred)) != {"hexahedron"}
        or policy.allow_mixed
        or policy.allowed_transitions
    ):
        issues.append("a pure hexahedral target")
    if specification.fill_strategy is not VolumeFillStrategy.MULTIZONE:
        issues.append("the multizone fill strategy for tetrahedral-dual block closure")
    if source.complex.boundary == "fixed":
        issues.append(
            "subdividable PLC boundary facets; immutable triangles cannot be hex faces"
        )
    return issues


@final
class PreparedDualQuad(StrictModule, NonTrainableState):
    scaffold: PreparedPlanarDomain
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlanarSource,
        specification: SurfaceMeshingSpec,
        /,
        *,
        scope: NativeExecutionScope,
    ) -> None:
        issues = dual_quad_support_issues(source, specification)
        if issues:
            raise ValueError(
                "PreparedDualQuad.specification must satisfy supported quad capabilities; got "
                + "; ".join(issues)
            )
        self.scaffold = PreparedPlanarDomain(
            source, _triangle_scaffold(specification), scope=scope
        )
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dual-quad",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "scaffold": self.scaffold.prepared_id,
            }
        )


@final
class PreparedDualHex(StrictModule, NonTrainableState):
    schedule: NativeVolumeSchedule
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlcSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        issues = dual_hex_support_issues(source, specification)
        if issues:
            raise ValueError(
                "PreparedDualHex.specification must satisfy supported hexahedral capabilities; got "
                + "; ".join(issues)
            )
        self.schedule = schedule
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dual-hex",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
            }
        )


@checked
def execute_quad_route(
    source: sources_owner.NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedDualQuad,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with (
        family_route_refusals(specification.limits, "dual_quad"),
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        if (
            source.binding_id != prepared.source_binding_id
            or specification.specification_id != prepared.specification_id
        ):
            raise ValueError(
                "PreparedDualQuad.binding must match the source and specification IDs; got mismatched IDs"
            )
        started, limits = (scope.operation_started, specification.limits)
        with measure_phase(record_phase, "topology_construction"):
            triangulation, inserted = triangulate(
                prepared.scaffold,
                started,
                _triangle_scaffold(specification),
                scope=scope,
            )
        triangle_count = triangulation.triangles.shape[0]
        point_count = triangulation.points.shape[0]
        source_edge_count = prepared.scaffold.source_directions.shape[0]
        with scope.host_workspace() as phase_storage:
            # Sort copies, permutations, inverse and carried-pair temporaries
            # stay within 640 bytes per triangle; gathered points and the
            # source-edge difference within 48 per point and 96 per source edge.
            phase_bytes = (
                640 * triangle_count + 48 * point_count + 96 * source_edge_count + 4096
            )
            phase_storage.set_bound(phase_bytes)
            # Triangle/segment copies and their compaction (four passes over
            # 3T entries), the used-vertex and constrained-edge sorts, the
            # point gather, and the source-edge difference sort.
            sorted_rows = source_edge_count + 3 * triangle_count
            scope.charge(
                work=12 * triangle_count
                + point_count
                + (3 * triangle_count + sorted_rows)
                * (sorted_rows.bit_length() + 1)
            )
            triangles = np.asarray(triangulation.triangles, dtype=np.int64)
            used, compact = np.unique(triangles, return_inverse=True)
            points = np.asarray(triangulation.points, dtype=np.float64)[used]
            triangles = compact.reshape(triangles.shape)
            segment_ids = np.asarray(triangulation.segment_ids, dtype=np.int64)
            constrained, constrained_sources = constrained_edges(
                triangles, segment_ids, prepared.scaffold.segment_sources
            )
            missing = np.setdiff1d(
                np.arange(source_edge_count), constrained_sources
            )
            if missing.size:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "quad.source_edges must intersect the meshed region; got an omitted source edge",
                    entity_ids=tuple(missing.tolist()),
                )
            vertex_count, triangle_rows = max(1, points.shape[0]), max(1, triangles.shape[0])
            # from_triangles builds its block, then the mesh, beside the phase.
            phase_storage.set_bound(
                phase_bytes
                + max(
                    cell_block_construction_bounds(triangle_rows, 3),
                    cell_mesh_construction_bounds(
                        vertex_count,
                        triangulation.points.shape[1],
                        (("triangle", triangle_rows, 3),),
                    ),
                )
            )
            scope.charge(
                work=cell_block_construction_work(triangle_rows, 3)
                + cell_mesh_construction_work(
                    vertex_count,
                    triangulation.points.shape[1],
                    (("triangle", triangle_rows, 3),),
                )
            )
            scaffold = CellMesh.from_triangles(
                points, triangles, numeric_version=source.source_revision
            )
        with coordinate_phase(scope, specification.limits, budget):
            authority, source_deviations, carrier_deviations = (
                constraint_coordinate_source(
                    source, prepared.scaffold, triangulation, used, budget=budget
                )
            )
        with coordinate_phase(scope, specification.limits, budget):
            scaffold_geometry = CellGeometrySpec.plc(scaffold, authority, budget=budget)
        with coordinate_phase(scope, specification.limits, budget):
            patches, labels, associations = organization_and_associations(
                scaffold,
                source,
                prepared.scaffold,
                constrained,
                constrained_sources,
                used,
                authority,
                budget=budget,
            )
        with scope.host_workspace() as phase_storage:
            phase_storage.set_bound(128 * source_edge_count + 4096)
            # The two maxima over the source-edge deviation and bound banks.
            scope.charge(work=2 * source_edge_count)
            domain = prepared.scaffold.domain(source)
            rounding = float(np.max(source_deviations, initial=0.0))
            admissible = float(np.max(prepared.scaffold.source_bounds, initial=0.0))
        scaffold_edge_count = scaffold.topology.entities(1).count
        constrained_count = constrained.shape[0]
        with scope.host_workspace() as phase_storage:
            # Edge IDs, their Python rows and keyed dictionary (384 bytes per
            # edge); constrained rows and the feature list (192 per edge).
            phase_storage.set_bound(
                384 * scaffold_edge_count + 192 * constrained_count + 4096
            )
            # One keyed row per scaffold edge and one lookup per constrained
            # edge, each with its sorted pair.
            scope.charge(work=2 * (scaffold_edge_count + constrained_count))
            edge_rows = {
                tuple(sorted(edge)): row
                for row, edge in enumerate(entities(scaffold, 1).tolist())
            }
            features = np.asarray(
                [edge_rows[tuple(sorted(edge))] for edge in constrained.tolist()],
                dtype=np.int64,
            )
            del edge_rows
        field = prepare_surface_cross_field(
            scaffold,
            feature_edges=features,
            record_phase=record_phase,
            source_geometry=scaffold_geometry,
            scope=scope,
            budget=budget,
        )
        with measure_phase(record_phase, "topology_construction"):
            extraction = extract_surface_quads(
                scaffold,
                limits,
                cross_field=field,
                source_geometry=scaffold_geometry,
                scope=scope,
                budget=budget,
            )
        mesh = extraction.mesh
        with measure_phase(record_phase, "geometry_association"):
            zones, patches, labels, associations = remap_dual_metadata(
                extraction,
                patches=patches,
                labels=labels,
                associations=associations,
                scope=scope,
            )
        phase = _quad_compliance(
            _QuadComplianceInputs(
                mesh,
                prepared,
                extraction,
                scaffold,
                features,
                constrained_sources,
                field,
                rounding,
                admissible,
                triangulation,
                inserted,
                authority,
                carrier_deviations,
                source_deviations,
            ),
            specification,
            scope,
            budget,
        )
        compliance = phase.compliance
        check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
        face_count = mesh.topology.entities(2).count
        cell_regions = family_host_array((face_count,), np.int64, scope=scope)
        cell_regions.fill(0)
        return publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            (
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.binding_id, prepared.prepared_id),
                    output_ids=(mesh.mesh_id,),
                    created_count=mesh.topology.entities(2).count,
                ),
            ),
            provider,
            {
                "kind": "native-dual-quad",
                "source": source.binding_id,
                "plan": plan_id,
                "specification": specification.specification_id,
            },
            NativeCertificationRequest(
                MeshCertificationSchedule("volume_plc"),
                source.source_id,
                source.source_revision,
                limits,
                domain=domain,
                cell_regions=cell_regions,
            ),
            policy=publication_policy("dual"),
            organization=NativePublicationOrganization(
                patches=patches, zones=zones, labels=labels, associations=associations
            ),
            execution=NativePublicationExecution(
                scope=scope, workspace=workspace, record_phase=record_phase, budget=budget
            ),
            geometry=extraction.geometry,
        )


@checked
def execute_hex_route(
    source: sources_owner.NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedDualHex,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with (
        family_route_refusals(specification.limits, "dual_hex"),
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        if (
            source.binding_id != prepared.source_binding_id
            or specification.specification_id != prepared.specification_id
        ):
            raise ValueError(
                "PreparedDualHex.binding must match the source and specification IDs; got mismatched IDs"
            )
        started, limits = (scope.operation_started, specification.limits)
        # Hex extraction certifies its tetrahedral scaffold under the default
        # validity policy; the native scaffold improvement enforces that floor.
        construction = generate_plc_volume(
            source.complex,
            _tetrahedron_scaffold(specification),
            prepared.schedule,
            validity_policy=CellValidityPolicy(),
            source_id=source.source_id,
            source_revision=source.source_revision,
            input_id=prepared.prepared_id,
            record_phase=record_phase,
            scope=scope,
        )
        with measure_phase(record_phase, "geometry_association"):
            with coordinate_phase(scope, specification.limits, budget):
                source_geometry = exact_plc_geometry(construction, source, budget=budget)
            if source_geometry is None:
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    "dual_hex.construction.geometry must retain the PLC coordinate source; got no geometry",
                    stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                )
        with measure_phase(record_phase, "topology_construction"):
            extraction = extract_volume_hexes(
                construction.mesh,
                limits,
                source_geometry=source_geometry,
                scope=scope,
                budget=budget,
            )
        return _publish_volume_extraction(
            source,
            specification,
            prepared,
            extraction,
            construction,
            coordinate_contract,
            provider,
            plan_id,
            started,
            "plc_dual_hex",
            record_phase=record_phase,
            scope=scope,
            workspace=workspace,
            budget=budget,
        )


def _publish_volume_extraction(
    source: sources_owner.NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedDualHex | PreparedHexDominant,
    extraction: DualExtraction,
    construction: VolumeConstruction,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    started: float,
    route: VolumeExtractionRoute,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    mapped_quality: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    budget: CoordinateEnclosureBudget,
) -> CellMeshingResult:
    route = parse(route, VolumeExtractionRoute, "hex.route")
    with family_route_refusals(specification.limits, "volume_family_publication"):
        mesh, limits = extraction.mesh, specification.limits
        if extraction.geometry is None:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                "hex.geometry must retain source coordinate maps including transition apex maps; got no geometry",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            )
        geometry = extraction.geometry
        region_count = extraction.parent_cells.size
        cell_regions = family_host_array(
            (region_count,), construction.cell_regions.dtype, scope=scope
        )
        # One region gather per published cell.
        scope.charge(work=region_count)
        np.take(construction.cell_regions, extraction.parent_cells, out=cell_regions)
        certification_request = NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            source.source_id,
            source.source_revision,
            limits,
            domain=construction.domain,
            cell_regions=cell_regions,
        )
        with measure_phase(record_phase, "geometry_association"):
            zones, patches, labels, associations = remap_dual_metadata(
                extraction,
                zones=construction.zones,
                patches=construction.patches,
                labels=construction.labels,
                associations=construction.associations,
                scope=scope,
            )
            # Constraint ordering follows native source preparation.
            # Support uses Q1 hexahedral and rational transition coordinate maps.
            support_source = prepare_plc_source(
                source.complex,
                source.source_id,
                source.source_revision,
                coordinate_contract,
                limits=limits,
                record_phase=record_phase,
                scope=scope,
            )
            if (
                support_source.association_transfer.domain.domain_id
                != construction.domain.domain_id
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    "hex.support.domain_id must equal publication.domain_id; got different PLC domains",
                    stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                )
            try:
                with coordinate_phase(scope, specification.limits, budget):
                    association_proof = certify_mapped_plc_associations(
                        support_source.association_transfer,
                        mesh,
                        geometry,
                        associations,
                        validity=extraction.validity,
                        certificate_limits=certification_request.certificate_limits,
                        budget=budget,
                    )
                associations = association_proof.associations
                certification_request = replace(
                    certification_request, prepared=association_proof.support.prepared
                )
            except AssociationPropagationError as error:
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    str(error),
                    stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                    entity_ids=tuple(
                        int(identifier) for identifier in error.target_ids.tolist()
                    ),
                ) from error
            except ValueError as error:
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    str(error),
                    stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                ) from error
        phase = _volume_extraction_compliance(
            mesh,
            specification,
            extraction,
            construction,
            geometry,
            associations,
            mapped_quality,
            record_phase,
            scope,
            budget,
        )
        compliance = phase.compliance
        check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
        return publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            (
                *construction.stages,
                MeshingStageReport(
                    MeshingStageKind.VOLUME_FILL,
                    MeshingStageStatus.PASSED,
                    input_ids=(construction.mesh.mesh_id,),
                    output_ids=(mesh.mesh_id,),
                    created_count=mesh.topology.entities(3).count,
                ),
            ),
            provider,
            {
                "kind": "native-volume-family-extraction",
                "route": route,
                "source": source.binding_id,
                "plan": plan_id,
                "specification": specification.specification_id,
                "schedule": prepared.schedule.schedule_id,
            },
            certification_request,
            policy=publication_policy("dual"),
            organization=NativePublicationOrganization(
                patches=patches, zones=zones, labels=labels, associations=associations
            ),
            execution=NativePublicationExecution(
                scope=scope, workspace=workspace, record_phase=record_phase, budget=budget
            ),
            geometry=geometry,
        )


@checked
def hex_dominant_support_issues(
    source: sources_owner.NativePlcSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    issues = volume_owner.volume_support_issues(
        source, _tetrahedron_scaffold(specification)
    )
    policy = specification.target.cell_families
    families = set((*policy.required, *policy.preferred, *policy.allowed_transitions))
    if (
        not policy.allow_mixed
        or "hexahedron" not in set((*policy.required, *policy.preferred))
        or not families <= {"hexahedron", "pyramid", "tetrahedron"}
    ):
        issues.append(
            "a mixed hex/pyramid/tetrahedron family policy with hexahedral core"
        )
    if (
        specification.fill_strategy
        is not VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION
    ):
        issues.append("the hex-dominant simplex-transition fill strategy")
    if source.complex.boundary == "fixed":
        issues.append("subdividable PLC boundary facets for hybrid closure")
    return issues


@final
class PreparedHexDominant(StrictModule, NonTrainableState):
    schedule: NativeVolumeSchedule
    core_fraction: float = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlcSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
        *,
        core_fraction: float = 0.8,
    ) -> None:
        issues = hex_dominant_support_issues(source, specification)
        if issues:
            raise ValueError(
                "PreparedHexDominant.specification must satisfy supported mixed-family capabilities; got "
                + "; ".join(issues)
            )
        if not np.isfinite(core_fraction) or not 0.0 < core_fraction <= 1.0:
            raise ValueError(
                "PreparedHexDominant.core_fraction must lie in (0, 1]; got a nonfinite or out-of-range value"
            )
        self.schedule = schedule
        self.core_fraction = core_fraction
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hex-dominant",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
                "core_fraction": core_fraction,
            }
        )


@checked
def execute_hex_dominant_route(
    source: sources_owner.NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedHexDominant,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with (
        family_route_refusals(specification.limits, "hex_dominant"),
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        if (
            source.binding_id != prepared.source_binding_id
            or specification.specification_id != prepared.specification_id
        ):
            raise ValueError(
                "PreparedHexDominant.binding must match the source and specification IDs; got mismatched IDs"
            )
        started = scope.operation_started
        construction = generate_plc_volume(
            source.complex,
            _tetrahedron_scaffold(specification),
            prepared.schedule,
            validity_policy=CellValidityPolicy(),
            source_id=source.source_id,
            source_revision=source.source_revision,
            input_id=prepared.prepared_id,
            record_phase=record_phase,
            scope=scope,
        )
        with measure_phase(record_phase, "geometry_association"):
            with coordinate_phase(scope, specification.limits, budget):
                source_geometry = exact_plc_geometry(construction, source, budget=budget)
            if source_geometry is None:
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    "hex_dominant.construction.geometry must retain the PLC coordinate source; got no geometry",
                    stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                )
        cell_count = construction.mesh.topology.entities(3).count
        with scope.host_workspace() as phase_storage:
            # The entries above as float64/int64 plus the sort permutation
            # and key copies: at most 512 bytes per tetrahedron.
            phase_storage.set_bound(512 * cell_count + 4096)
            # Per tetrahedron: its four-corner copy and gather, three edge
            # differences, one determinant with its scaling and negated key,
            # its identifier copy, and the two-key lexicographic sort with the
            # selected-prefix gather.
            scope.charge(
                work=cell_count * (16 + 2 * (cell_count.bit_length() + 1))
            )
            cells = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
            points = np.asarray(construction.mesh.coordinates, dtype=np.float64)[cells]
            # Source volumes rank a deterministic robustness-oriented core selection.
            # This does not advertise a frame-aligned or quality-optimal placement.
            volume = (
                np.linalg.det(
                    np.stack(
                        (
                            points[:, 1] - points[:, 0],
                            points[:, 2] - points[:, 0],
                            points[:, 3] - points[:, 0],
                        ),
                        axis=-1,
                    )
                )
                / 6.0
            )
            identifiers = np.asarray(
                construction.mesh.topology.entities(3).entity_ids, dtype=np.int64
            )
            order = np.lexsort((identifiers, -volume))
            selected_count = max(
                1, int(np.ceil(prepared.core_fraction * cells.shape[0]))
            )
            hex_core_cells = identifiers[order[:selected_count]]
            del cells, points, volume, identifiers, order
        with measure_phase(record_phase, "topology_construction"):
            extraction = extract_hex_dominant(
                construction.mesh,
                specification.limits,
                specification.target.cell_families,
                hex_core_cells=hex_core_cells,
                source_geometry=source_geometry,
                scope=scope,
                budget=budget,
            )
        return _publish_volume_extraction(
            source,
            specification,
            prepared,
            extraction,
            construction,
            coordinate_contract,
            provider,
            plan_id,
            started,
            "plc_hex_dominant",
            record_phase=record_phase,
            scope=scope,
            workspace=workspace,
            budget=budget,
        )


@final
class PreparedGridHex(StrictModule, NonTrainableState):
    volume_schedule: NativeVolumeSchedule
    grid_schedule: NativeHexGridSchedule
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlcSource,
        specification: VolumeMeshingSpec,
        volume_schedule: NativeVolumeSchedule,
        grid_schedule: NativeHexGridSchedule,
        /,
    ) -> None:
        issues = dual_hex_support_issues(source, specification)
        if issues:
            raise ValueError(
                "PreparedGridHex.specification must satisfy supported integer-grid capabilities; got "
                + "; ".join(issues)
            )
        self.volume_schedule = volume_schedule
        self.grid_schedule = grid_schedule
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-grid-hex",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "volume_schedule": volume_schedule.schedule_id,
                "grid_schedule": grid_schedule.schedule_id,
            }
        )


def _grid_metadata(
    source: sources_owner.NativePlcSource,
    old: VolumeConstruction,
    grid: GridHexConstruction,
    /,
    *,
    scope: NativeExecutionScope,
) -> NativeRouteMetadata:
    "Bind grid entities to PLC strata by plane/chain facts."
    mesh, complex_ = grid.mesh, source.complex
    connectivity = mesh.connectivity
    match connectivity:
        case HexahedralConnectivity():
            pass
        case _:
            raise RuntimeError(
                "grid.mesh.connectivity must be hexahedral; got a different connectivity family"
            )
    sets = tuple(mesh.topology.entities(degree) for degree in range(4))
    vertex_count, edge_count, face_count, cell_count = (
        local_entities.count for local_entities in sets
    )
    facet_count = complex_.facet_regions.shape[0]
    old_edge_set = old.mesh.topology.entities(1)
    old_vertex_count = old.mesh.coordinates.shape[0]
    source_edge_count = sum(
        association.target_global_ids.shape[0]
        for association in old.associations
        if association.target_entity_set_id == old_edge_set.entity_set_id
    )
    source_vertex_count = complex_.vertices.shape[0]
    with scope.host_workspace() as workspace:
        # Conservative CPython container bounds per cell, face, edge, and
        # vertex (identity banks, region strings, incidence sets, ridge plans
        # and resolution maps), per source polygon, facet, and source vertex,
        # per old edge and source edge row (pullback rows, their line index
        # and covering intervals), plus the old-point, old-edge, chart,
        # edge-row and grouping-sort banks.
        workspace.set_bound(
            _HOST_ENTRY_BYTES
            * (
                2 * cell_count
                + 2 * face_count
                + 6 * edge_count
                + 5 * vertex_count
                + complex_.polygon_facets.size
                + facet_count
                + 2 * source_vertex_count
                + old_edge_set.count
                + 5 * source_edge_count
            )
            + 32 * old_vertex_count
            + 16 * old_edge_set.count
            + 9 * source_edge_count
            + 40 * vertex_count
            + 16 * edge_count
            + 24 * (cell_count + face_count)
            + 4096
        )
        sort_depth = max(cell_count, face_count).bit_length() + 1
        # Identity and connectivity copies; the region and facet groupings,
        # their scope sorts, the unique-polygon sort and label scopes; face
        # incidence, label masks, normals, orientations and association
        # columns; every source polygon loop entry; the coordinate copy and
        # rounding scan; and the per-face edge/vertex facet incidence.
        scope.charge(
            work=8 * vertex_count
            + 4 * edge_count
            + 42 * face_count
            + 12 * cell_count
            + 4 * complex_.polygon_vertices.size
            + (2 * cell_count + 4 * face_count) * sort_depth
        )
        ids = tuple(
            np.asarray(local_entities.entity_ids, dtype=np.int64)
            for local_entities in sets
        )
        facets = grid.face_source_facets
        face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
        face_edges = np.asarray(connectivity.face_edges, dtype=np.int64)
        edge_vertices = entities(mesh, 1)
        cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
        boundary_faces = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
        if np.any(boundary_faces & (facets < 0)):
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "grid.face_source_facets must identify every exposed face; got an unassigned boundary face",
            )
        del boundary_faces
        cell_regions = np.asarray(grid.cell_regions)
        region_rows = _grouped_rows(cell_regions)
        zones = tuple(
            MeshZone(
                name,
                MeshZoneRole.REGION,
                mesh_entity_scope(mesh, 3, ids[3][region_rows[region]]),
            )
            for region, name in enumerate(complex_.region_ids)
            if region in region_rows
        )
        del region_rows
        facet_rows = np.flatnonzero(facets >= 0)
        facet_groups = _grouped_rows(facets[facet_rows])
        present_facets = list(facet_groups)
        patches = tuple(
            MeshPatch(
                f"facet:{facet}",
                mesh_entity_scope(mesh, 2, ids[2][facet_rows[rows]]),
            )
            for facet, rows in facet_groups.items()
        )
        del facet_groups
        incidence = complex_.facet_regions[facets[facet_rows]]
        interface = (
            (incidence[:, 0] >= 0)
            & (incidence[:, 1] >= 0)
            & (incidence[:, 0] != incidence[:, 1])
        )
        sheet = incidence[:, 0] == incidence[:, 1]
        labels = tuple(
            MeshLabel(name, mesh_entity_scope(mesh, 2, selected))
            for name, selected in (
                ("interface", ids[2][facet_rows][interface]),
                ("internal_sheet", ids[2][facet_rows][sheet]),
            )
            if selected.size
        )
        del incidence, interface, sheet
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        rounding = 128.0 * np.finfo(np.float64).eps * np.max(np.abs(points), initial=1.0)
        facet_names = {
            facet: sources_owner.source_entity_id(source.source_revision, "facet", facet)
            for facet in present_facets
        }
        source_normals = {}
        for polygon in np.unique(grid.face_source_polygons[facet_rows]).tolist():
            loop = complex_.polygon_vertices[
                complex_.polygon_offsets[polygon] : complex_.polygon_offsets[polygon + 1]
            ]
            vertices = complex_.vertices[loop]
            source_normals[polygon] = np.sum(
                np.cross(vertices, np.roll(vertices, -1, axis=0)), axis=0
            )
        normals = np.cross(
            points[face_vertices[facet_rows, 1]] - points[face_vertices[facet_rows, 0]],
            points[face_vertices[facet_rows, 3]] - points[face_vertices[facet_rows, 0]],
        )
        orientations = np.asarray(
            [
                np.sign(np.sum(normal * source_normals[int(polygon)]))
                for normal, polygon in zip(
                    normals, grid.face_source_polygons[facet_rows], strict=True
                )
            ],
            dtype=np.int8,
        )
        del normals, source_normals
        associations = [
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                sets[3].entity_set_id,
                ids[3],
                tuple(
                    sources_owner.source_entity_id(
                        source.source_revision, "region", int(region)
                    )
                    for region in cell_regions
                ),
                np.zeros((cells.shape[0],), dtype=np.float64),
                exact=True,
            ),
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                sets[2].entity_set_id,
                ids[2][facet_rows],
                tuple(facet_names[int(facet)] for facet in facets[facet_rows]),
                np.full((facet_rows.size,), rounding, dtype=np.float64),
                orientations=orientations,
            ),
        ]
        edge_facets: list[set[int]] = [set() for _ in range(edge_vertices.shape[0])]
        vertex_facets: list[set[int]] = [set() for _ in range(points.shape[0])]
        for face in facet_rows.tolist():
            facet = int(facets[face])
            for edge in face_edges[face].tolist():
                edge_facets[edge].add(facet)
            for vertex in face_vertices[face].tolist():
                vertex_facets[vertex].add(facet)
        # Source-associated segment unions certify a ridge chain continuously.
        # Identity comes from their old associations, not nearby coordinates.
        phase = _grid_ridge_associations(
            source,
            old,
            grid,
            points,
            edge_vertices,
            edge_facets,
            facet_names,
            ids,
            sets,
            rounding,
            associations,
            scope=scope,
        )
        associations = phase.associations
        ridge_names = phase.ridge_names
        phase = _grid_vertex_associations(
            source,
            grid,
            points,
            edge_vertices,
            ridge_names,
            vertex_facets,
            facet_names,
            cells,
            ids,
            sets,
            rounding,
            associations,
            scope=scope,
        )
        associations = phase.associations
    return NativeRouteMetadata(zones, patches, labels, tuple(associations))


@checked
def execute_grid_hex_route(
    source: sources_owner.NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedGridHex,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with (
        family_route_refusals(specification.limits, "grid_hex"),
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        if (
            source.binding_id != prepared.source_binding_id
            or specification.specification_id != prepared.specification_id
        ):
            raise ValueError(
                "PreparedGridHex.binding must match the source and specification IDs; got mismatched IDs"
            )
        started = scope.operation_started
        construction = generate_plc_volume(
            source.complex,
            _tetrahedron_scaffold(specification),
            prepared.volume_schedule,
            validity_policy=CellValidityPolicy(),
            source_id=source.source_id,
            source_revision=source.source_revision,
            input_id=prepared.prepared_id,
            record_phase=record_phase,
            scope=scope,
        )
        control = specification.size_controls[0]
        match control:
            case UniformSizeControl():
                pass
            case _:
                raise TypeError(
                    "grid_hex.size_controls must contain a uniform control; got a different control kind"
                )
        # The maximum is an admissibility bound, not the requested resolution.
        size = control.target_size
        grid = generate_integer_grid_hexes(
            source.complex,
            construction,
            prepared.grid_schedule,
            specification.limits,
            size,
            maximum_size=min(
                control.target_size + specification.size_compliance.tolerance(control.target_size),
                float("inf") if control.maximum_size is None else control.maximum_size + specification.size_compliance.tolerance(control.maximum_size),
            ),
            record_phase=record_phase,
            scope=scope,
            budget=budget,
        )
        with measure_phase(record_phase, "geometry_association"):
            metadata = _grid_metadata(source, construction, grid, scope=scope)
            zones = metadata.zones
            patches = metadata.patches
            labels = metadata.labels
            associations = metadata.associations
        mesh = grid.mesh
        cell_count = mesh.blocks[0].cell_count
        with (
            coordinate_phase(scope, specification.limits, budget),
            budget.temporary_scope(),
        ):
            # evaluate_cell_quality charges its own corner queries; the three
            # host quality copies are temporary.
            budget.reserve(0, 24 * cell_count + 4096)
            quality = evaluate_cell_quality(mesh, budget=budget)
            minimum = float(np.min(np.asarray(quality.scaled_jacobian)))
            mean_ratio = float(np.min(np.asarray(quality.mean_ratios)))
            aspect_ratio = float(np.max(np.asarray(quality.aspect_ratios)))
        edge_count = mesh.topology.entities(1).count
        vertex_count = mesh.coordinates.shape[0]
        with scope.host_workspace() as size_workspace:
            # Host point and edge copies, endpoint gathers and differences,
            # lengths, a partition copy, and per-vertex extrema banks.
            size_workspace.set_bound(128 * edge_count + 72 * vertex_count + 4096)
            # Size evidence plus the coordinate-magnitude rounding scan.
            scope.charge(
                work=_size_evidence_work(
                    edge_count,
                    vertex_count,
                    len(specification.size_compliance.target_statistics),
                )
                + 3 * vertex_count
            )
            points = np.asarray(mesh.coordinates, dtype=np.float64)
            lengths, growth = edge_size_evidence(points, entities(mesh, 1))
            requested, achieved, issues = uniform_size_compliance(
                control, specification.size_compliance, lengths, growth
            )
            rounding = (
                128.0 * np.finfo(np.float64).eps * np.max(np.abs(points), initial=1.0)
            )
            del points, lengths
        requested.append(
            ("minimum_scaled_jacobian", prepared.grid_schedule.minimum_scaled_jacobian)
        )
        achieved.extend(
            (
                ("minimum_scaled_jacobian", minimum),
                ("hexahedron_count", mesh.blocks[0].cell_count),
                ("octree_closure_rounds", grid.closure_rounds),
                (
                    "frame_field_objective",
                    float(np.asarray(grid.frame.optimization.objective)),
                ),
                (
                    "frame_alignment_residual",
                    float(np.max(grid.frame.alignment_residuals)),
                ),
                ("singular_source_vertices", grid.frame.singular_vertices.size),
            )
        )
        if minimum < prepared.grid_schedule.minimum_scaled_jacobian:
            issues.append("minimum_scaled_jacobian")
        requested.extend(
            (
                ("minimum_mean_ratio", prepared.grid_schedule.minimum_mean_ratio),
                ("maximum_aspect_ratio", prepared.grid_schedule.maximum_aspect_ratio),
            )
        )
        achieved.extend(
            (("minimum_mean_ratio", mean_ratio), ("maximum_aspect_ratio", aspect_ratio))
        )
        if mean_ratio < prepared.grid_schedule.minimum_mean_ratio:
            issues.append("minimum_mean_ratio")
        if aspect_ratio > prepared.grid_schedule.maximum_aspect_ratio:
            issues.append("maximum_aspect_ratio")
        for feature in specification.protected_features:
            requested.append(
                (
                    f"protected:{feature.feature_id}:maximum_deviation",
                    feature.maximum_deviation,
                )
            )
            achieved.append(
                (f"protected:{feature.feature_id}:maximum_deviation", rounding)
            )
            if rounding > feature.maximum_deviation:
                issues.append(f"protected:{feature.feature_id}:maximum_deviation")
        compliance = MeshingComplianceReport(
            specification.specification_id,
            requested=tuple(requested),
            achieved=tuple(achieved),
            issues=tuple(issues),
        )
        check_deadline(started, specification.limits, MeshingStageKind.GEOMETRY_AUDIT)
        return publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            (
                *construction.stages,
                MeshingStageReport(
                    MeshingStageKind.VOLUME_FILL,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.binding_id, prepared.prepared_id),
                    output_ids=(grid.construction_id, mesh.mesh_id),
                    created_count=mesh.blocks[0].cell_count,
                ),
            ),
            provider,
            {
                "kind": "native-integer-grid-hex",
                "route": prepared.grid_schedule.route,
                "source": source.binding_id,
                "plan": plan_id,
                "specification": specification.specification_id,
                "grid_schedule": prepared.grid_schedule.schedule_id,
            },
            NativeCertificationRequest(
                MeshCertificationSchedule("volume_plc"),
                source.source_id,
                source.source_revision,
                specification.limits,
                domain=construction.domain,
                cell_regions=grid.cell_regions,
            ),
            policy=publication_policy("grid"),
            organization=NativePublicationOrganization(
                patches=patches, zones=zones, labels=labels, associations=associations
            ),
            execution=NativePublicationExecution(
                scope=scope, workspace=workspace, record_phase=record_phase, budget=budget
            ),
        )


class _MappedHexSourceBinding(Protocol):
    reference: sources_owner.NativePlcSource
    domain: MappedReferenceDomain
    source_id: str
    source_revision: str
    binding_id: str


def _mapped_face_regions(domain: MappedReferenceDomain, /) -> tuple[tuple[str, ...], ...]:
    "Read material adjacency from the declared root complex."
    mesh = domain.reference_mesh
    faces, cells = entities(mesh, 2), entities(mesh, 3)
    lookup = {tuple(sorted(vertices.tolist())): row for row, vertices in enumerate(faces)}
    adjacent: list[set[str]] = [set() for _ in faces]
    for cell, region in zip(cells, domain.cell_regions, strict=True):
        for local in reference_cell_topology("hexahedron").entities[2]:
            adjacent[lookup[tuple(sorted(cell[list(local)].tolist()))]].add(
                domain.region_ids[region]
            )
    return tuple(tuple(sorted(regions)) for regions in adjacent)


def _mapped_control_issues(
    source: _MappedHexSourceBinding, specification: VolumeMeshingSpec, /
) -> list[str]:
    """Admit physical identities on independently named root strata."""
    domain, mesh = source.domain, source.domain.reference_mesh
    issues: list[str] = []
    cells = np.asarray(mesh.topology.entities(3).entity_ids, dtype=np.int64)
    requests = {}
    for control in specification.region_controls:
        scope = control.scope
        if control.region_name not in domain.region_ids:
            issues.append("region controls naming an authoritative mapped material")
            continue
        region = domain.region_ids.index(control.region_name)
        expected = cells[domain.cell_regions == region]
        if (
            scope.source_id != source.source_id
            or scope.source_revision != source.source_revision
            or scope.entity_dimension != 3
            or scope.entity_set_id != domain.entity_set_id(3)
            or not np.array_equal(
                np.sort(np.asarray(scope.entity_ids)), np.sort(expected)
            )
            or not control.meshing_enabled
        ):
            issues.append(
                "enabled mapped region controls owning their complete exact root cells"
            )
        previous = requests.get(control.region_name)
        if previous is not None and (
            previous.material_id != control.material_id or previous.role != control.role
        ):
            issues.append("noncontradictory mapped region material and role identities")
        requests[control.region_name] = control
    face_ids = np.asarray(mesh.topology.entities(2).entity_ids, dtype=np.int64)
    face_rows = {int(identifier): row for row, identifier in enumerate(face_ids)}
    adjacency = _mapped_face_regions(domain)
    for control in specification.patch_controls:
        scope, identifiers = control.scope, np.asarray(control.scope.entity_ids)
        if (
            scope.source_id != source.source_id
            or scope.source_revision != source.source_revision
            or scope.entity_dimension != 2
            or scope.entity_set_id != domain.entity_set_id(2)
            or np.any(~np.isin(identifiers, face_ids))
        ):
            issues.append("patch controls naming exact mapped root faces")
            continue
        if any(
            adjacency[face_rows[int(identifier)]] != control.adjacent_region_names
            for identifier in identifiers
        ):
            issues.append(
                "mapped patch adjacency equal to authoritative root material incidence"
            )
    return issues


@checked
def mapped_grid_support_issues(
    source: _MappedHexSourceBinding, specification: VolumeMeshingSpec, /
) -> list[str]:
    issues: list[str] = []
    domain = source.domain
    if (
        source.source_id != domain.source_id
        or source.source_revision != domain.source_revision
    ):
        issues.append("the mapped domain's authoritative source identity/revision")
    if (
        declared_plc_domain(
            source.reference.complex, source.reference.source_id
        ).domain_id
        != domain.reference_domain.domain_id
    ):
        issues.append("the exact independently declared reference PLC domain")
    target, policy = specification.target, specification.target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or set((*policy.required, *policy.preferred)) != {"hexahedron"}
        or policy.allow_mixed
        or policy.allowed_transitions
    ):
        issues.append("a pure hexahedral mapped volume target")
    if specification.fill_strategy is not VolumeFillStrategy.MULTIZONE:
        issues.append("the multizone mapped-grid fill strategy")
    source_degrees = [
        require_scalar_coordinate_element(element, "mapped source").degree
        for element in domain.source_geometry.elements
    ]
    if set(source_degrees) != {target.geometry_order}:
        issues.append("a geometry order equal to the independently declared source maps")
    boundary = specification.boundary_scope
    source_faces = domain.reference_mesh.topology.entities(2)
    if (
        boundary.source_id != source.source_id
        or boundary.source_revision != source.source_revision
        or boundary.entity_set_id != source.domain.entity_set_id(2)
        or np.any(
            ~np.isin(np.asarray(boundary.entity_ids), np.asarray(source_faces.entity_ids))
        )
    ):
        issues.append(
            "mapped-root face scientific IDs in the authoritative boundary scope"
        )
    match specification.size_controls:
        case [UniformSizeControl() as control]:
            if control.scope.scope_id != boundary.scope_id:
                issues.append("a size control on the requested mapped boundary scope")
        case _:
            issues.append("one whole-domain uniform physical size control")
    for feature in specification.protected_features:
        degree = feature.scope.entity_dimension
        if (
            degree < 0
            or degree > 2
            or feature.scope.entity_set_id != source.domain.entity_set_id(degree)
            or np.any(
                ~np.isin(
                    np.asarray(feature.scope.entity_ids),
                    np.asarray(
                        domain.reference_mesh.topology.entities(degree).entity_ids
                    ),
                )
            )
        ):
            issues.append("protected features naming exact mapped-root strata")
    if specification.region_seeds or specification.hole_seeds:
        issues.append("physical region/hole seeds require an owning mapped inverse query")
    issues.extend(_mapped_control_issues(source, specification))
    if (
        specification.periodic_constraints
        or domain.reference_mesh.periodic_topology is not None
    ):
        issues.append("mapped quotient/periodic source subdivision")
    if specification.layer_controls:
        issues.append("mapped boundary layers")
    return issues


@final
class PreparedMappedGridHex(StrictModule, NonTrainableState):
    grid_schedule: NativeHexGridSchedule
    source_frame: PreparedSourceHexFrame | None
    source_connection: SourceBlockConnection | None
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: _MappedHexSourceBinding,
        specification: VolumeMeshingSpec,
        grid_schedule: NativeHexGridSchedule,
        /,
        *,
        scope: NativeExecutionScope,
        record_phase: NativeMeshingPhaseRecorder | None = None,
    ) -> None:
        budget = coordinate_budget(specification.limits)
        with (
            family_route_refusals(specification.limits, "mapped_grid_preparation"),
            budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
        ):
            issues = mapped_grid_support_issues(source, specification)
            if issues:
                raise ValueError(
                    "PreparedMappedGridHex.specification must satisfy supported mapped-grid capabilities; got "
                    + "; ".join(issues)
                )
            self.grid_schedule = grid_schedule
            self.source_frame = (
                prepare_source_hex_frame(
                    source.domain.reference_mesh,
                    source.domain.source_geometry,
                    record_phase=record_phase,
                    scope=scope,
                    budget=budget,
                )
                if grid_schedule.route == "frame_grid"
                else None
            )
            self.source_connection = (
                None
                if self.source_frame is None
                else prepare_source_block_connection(
                    self.source_frame,
                    record_phase=record_phase,
                    scope=scope,
                )
            )
            self.source_binding_id = source.binding_id
            self.specification_id = specification.specification_id
            self.prepared_id = canonical_fingerprint(
                {
                    "kind": "prepared-mapped-grid-hex",
                    "source": source.binding_id,
                    "specification": specification.specification_id,
                    "grid_schedule": grid_schedule.schedule_id,
                    "source_frame": None
                    if self.source_frame is None
                    else self.source_frame.frame_id,
                    "source_connection": None
                    if self.source_connection is None
                    else self.source_connection.holonomy.connection_id,
                }
            )


def _mapped_grid_metadata(
    source: _MappedHexSourceBinding,
    mapped: MappedGridHexConstruction,
    specification: VolumeMeshingSpec,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    association_inputs: list[_ConstructedTensorGrid] | None = None,
) -> NativeRouteMetadata:
    "Map source-root strata by chart containment and scientific incidence."

    mesh = mapped.mesh
    transfer = MappedReferenceAssociationTransfer(
        source.domain,
        maximum_support_queries=specification.limits.maximum_geometry_queries,
    )
    with coordinate_phase(scope, specification.limits, budget):
        if association_inputs:
            construction = association_inputs.pop()
            try:
                associations = _tensor_reference_associations(
                    source.domain, mesh, mapped.geometry, construction,
                    maximum_support_queries=specification.limits.maximum_geometry_queries,
                    budget=budget,
                )
            finally:
                construction.active = False
                construction.owner_fields = ()
                construction.parents = ()
        else:
            associations = transfer.associations(mesh, mapped.geometry, budget=budget)
    vertex_count = mesh.coordinates.shape[0]
    face_count = mesh.topology.entities(2).count
    cell_count = mesh.topology.entities(3).count
    control_entries = sum(
        len(control.scope.entity_ids) for control in specification.patch_controls
    )
    with scope.host_workspace() as workspace:
        # Conservative CPython bounds for the face-key map, owner lists, and
        # interface rows, plus identity, face, and padded cell banks with
        # their selection masks, grouping sorts and patch-control scope copies.
        workspace.set_bound(
            _HOST_ENTRY_BYTES * (3 * face_count + 2 * cell_count)
            + 8 * vertex_count
            + 88 * face_count
            + 184 * cell_count
            + 8 * control_entries
            + 4096
        )
        sort_depth = max(cell_count, face_count, control_entries).bit_length() + 1
        patch_control_count = len(specification.patch_controls)
        # Face, cell and vertex-ID copies with one sorted key per face; the
        # region and parent-face groupings with their scope sorts and region
        # control lookups; per patch control its parent membership sort,
        # selection and scope; six sorted, keyed owner lookups per cell; and
        # the interface scan with its scope.
        scope.charge(
            work=vertex_count
            + 24 * face_count
            + 68 * cell_count
            + (2 * cell_count + 4 * face_count) * sort_depth
            + len(source.domain.region_ids) * len(specification.region_controls)
            + patch_control_count * face_count * (3 + 2 * sort_depth)
            + control_entries * sort_depth
        )
        face_rows = entities(mesh, 2)
        target_cells = entities(mesh, 3)
        target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        face_lookup = {
            tuple(sorted(target_vertex_ids[vertices].tolist())): row
            for row, vertices in enumerate(face_rows)
        }
        topology = reference_cell_topology("hexahedron")

        def entity_scope(degree: int, identifiers: np.ndarray, /) -> MeshingScope:
            return MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                degree,
                mesh.topology.entities(degree).entity_set_id,
                identifiers,
            )

        cell_ids = np.asarray(mesh.topology.entities(3).entity_ids, dtype=np.int64)
        zones = []
        region_rows = _grouped_rows(np.asarray(mapped.cell_regions))
        for region, region_id in enumerate(source.domain.region_ids):
            if region not in region_rows:
                continue
            control = next(
                (
                    item
                    for item in specification.region_controls
                    if item.region_name == region_id
                ),
                None,
            )
            zones.append(
                MeshZone(
                    region_id,
                    MeshZoneRole.REGION,
                    entity_scope(3, cell_ids[region_rows[region]]),
                    material_id=None if control is None else control.material_id,
                    region_role=None if control is None else control.role,
                )
            )
        del region_rows
        dimensions = np.asarray(associations[2].parent_dimensions, dtype=np.int64)
        parent_ids = np.asarray(associations[2].parent_ids, dtype=np.int64)
        faces = np.asarray(mesh.topology.entities(2).entity_ids, dtype=np.int64)
        face_parents = np.flatnonzero(dimensions == 2)
        patches = [
            MeshPatch(
                source.domain.image_entity_id(2, parent),
                entity_scope(2, faces[face_parents[rows]]),
            )
            for parent, rows in _grouped_rows(parent_ids[face_parents]).items()
        ]
        del face_parents
        for control in specification.patch_controls:
            selected = (dimensions == 2) & np.isin(
                parent_ids, np.asarray(control.scope.entity_ids)
            )
            if control.required and not np.any(selected):
                raise MeshingFailure(
                    MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "mapped_grid.patch_controls must publish child faces for every required patch; got an empty patch",
                )
            patches.append(MeshPatch(control.name, entity_scope(2, faces[selected])))
        interface_faces = []
        face_owners: list[list[int]] = [[] for _ in face_rows]
        for cell_row, vertices in enumerate(target_cells):
            for local in topology.entities[2]:
                key = tuple(sorted(target_vertex_ids[vertices[list(local)]].tolist()))
                face_owners[face_lookup[key]].append(cell_row)
        del face_lookup
        for row, owners in enumerate(face_owners):
            if (
                len(owners) == 2
                and mapped.cell_regions[owners[0]] != mapped.cell_regions[owners[1]]
            ):
                if dimensions[row] != 2:
                    raise MeshingFailure(
                        MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                        "mapped_grid.interface.source_face must identify a declared root face; got an unassigned interface",
                    )
                interface_faces.append(row)
        del face_owners
        labels = (
            (
                MeshLabel(
                    "interface",
                    entity_scope(2, faces[np.asarray(interface_faces, dtype=np.int64)]),
                ),
            )
            if interface_faces
            else ()
        )
    return NativeRouteMetadata(tuple(zones), tuple(patches), labels, associations)


@checked
def execute_mapped_grid_hex_route(
    source: _MappedHexSourceBinding,
    specification: VolumeMeshingSpec,
    prepared: PreparedMappedGridHex,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with (
        family_route_refusals(specification.limits, "mapped_grid_hex"),
        budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)),
    ):
        if (
            source.binding_id != prepared.source_binding_id
            or specification.specification_id != prepared.specification_id
        ):
            raise ValueError(
                "PreparedMappedGridHex.binding must match the source and specification IDs; got mismatched IDs"
            )
        started = scope.operation_started
        control = specification.size_controls[0]
        match control:
            case UniformSizeControl():
                pass
            case _:
                raise TypeError(
                    "mapped_grid.size_controls must contain a uniform control; got a different control kind"
                )
        # Aim at the authored size; compliance still checks every original bound.
        size = control.target_size
        if (prepared.grid_schedule.route == "frame_grid") != (
            prepared.source_frame is not None
        ) or (prepared.grid_schedule.route == "frame_grid") != (
            prepared.source_connection is not None
        ):
            raise ValueError(
                "PreparedMappedGridHex.source_frame must agree with grid_schedule.route; got inconsistent frame preparation"
            )
        if prepared.source_frame is not None and (
            prepared.source_frame.source_id
            != cell_geometry_id(source.domain.source_geometry)
            or prepared.source_frame.reference_mesh.mesh_id
            != source.domain.reference_mesh.mesh_id
        ):
            raise ValueError(
                "PreparedMappedGridHex.source_frame must bind the declared coordinate source and reference mesh; got mismatched IDs"
            )
        association_inputs: list[_ConstructedTensorGrid] = []
        association_domain = source.domain
        association_bindings = (
            source.source_id, source.source_revision, source.binding_id,
            specification.specification_id, prepared.prepared_id,
            prepared.grid_schedule.schedule_id,
        )
        phase = _mapped_grid_construction(
            source, specification, prepared, size, record_phase, scope, budget,
            association_inputs=association_inputs,
        )
        mapped = phase.mapped
        grid = phase.grid
        grid_identity = phase.grid_identity
        bound = phase.bound
        if (
            source.binding_id != prepared.source_binding_id
            or specification.specification_id != prepared.specification_id
            or source.domain is not association_domain
            or association_bindings != (
                source.source_id, source.source_revision, source.binding_id,
                specification.specification_id, prepared.prepared_id,
                prepared.grid_schedule.schedule_id,
            )
        ):
            raise ValueError("mapped_grid.association must retain current source, specification and plan")
        with measure_phase(record_phase, "geometry_association"):
            metadata = _mapped_grid_metadata(
                source, mapped, specification, scope=scope, budget=budget,
                association_inputs=association_inputs,
            )
            zones = metadata.zones
            patches = metadata.patches
            labels = metadata.labels
            associations = metadata.associations
        mesh = mapped.mesh
        phase = _mapped_grid_compliance(
            mesh,
            mapped,
            grid,
            bound,
            control,
            specification,
            prepared,
            record_phase,
            scope,
        )
        compliance = phase.compliance
        check_deadline(started, specification.limits, MeshingStageKind.GEOMETRY_AUDIT)
        fidelity_source = MappedDomainBoundarySource(source.domain, mapped.cell_regions)
        return publish_native_result(
            mesh,
            coordinate_contract,
            compliance,
            (
                MeshingStageReport(
                    MeshingStageKind.VOLUME_FILL,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.binding_id, prepared.prepared_id),
                    output_ids=(grid_identity, mapped.mapped_id, mesh.mesh_id),
                    created_count=mesh.topology.entities(3).count,
                ),
            ),
            provider,
            {
                "kind": "native-mapped-integer-grid-hex",
                "source": source.binding_id,
                "plan": plan_id,
                "specification": specification.specification_id,
                "grid_schedule": prepared.grid_schedule.schedule_id,
                "source_frame": None
                if prepared.source_frame is None
                else prepared.source_frame.frame_id,
                "source_connection": None
                if prepared.source_connection is None
                else prepared.source_connection.holonomy.connection_id,
                "integer_construction": grid_identity,
            },
            NativeCertificationRequest(
                MeshCertificationSchedule("mapped_volume"),
                source.source_id,
                source.source_revision,
                specification.limits,
                domain=source.domain,
                cell_regions=mapped.cell_regions,
                fidelity_source=fidelity_source,
                fidelity_tolerance=0.0,
            ),
            policy=publication_policy("grid"),
            organization=NativePublicationOrganization(
                patches=patches, zones=zones, labels=labels, associations=associations
            ),
            execution=NativePublicationExecution(
                scope=scope, workspace=workspace, record_phase=record_phase, budget=budget
            ),
            geometry=mapped.geometry,
        )


@final
class _QuadCompliancePhase(StrictModule):
    compliance: MeshingComplianceReport


def _quad_compliance(
    inputs: _QuadComplianceInputs,
    specification: SurfaceMeshingSpec,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    /,
) -> _QuadCompliancePhase:
    mesh = inputs.mesh
    prepared = inputs.prepared
    extraction = inputs.extraction
    scaffold = inputs.scaffold
    features = inputs.features
    constrained_sources = inputs.constrained_sources
    field = inputs.field
    rounding = inputs.rounding
    admissible = inputs.admissible
    triangulation = inputs.triangulation
    inserted = inputs.inserted
    authority = inputs.authority
    carrier_deviations = inputs.carrier_deviations
    source_deviations = inputs.source_deviations
    """Quad compliance."""
    edge_count = mesh.topology.entities(1).count
    vertex_count = mesh.coordinates.shape[0]
    dimension = mesh.ambient_dimension
    patch_count = prepared.scaffold.patch_sizes.size
    scaffold_edge_count = scaffold.topology.entities(1).count
    control_ids = [
        int(np.size(control.scope.entity_ids))
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    ]
    with scope.host_workspace() as phase_storage:
        # Edge and point banks (at most 64 bytes per edge and coordinate plus
        # 384 per edge for targets, gradation, ancestry and one control's
        # selections), vertex banks, membership keys, and one size-field chunk:
        # at most 128 bytes for each of max(chunk entries, patch count) entries.
        phase_storage.set_bound(
            (64 * dimension + 384) * edge_count
            + (8 * dimension + 32) * vertex_count
            + 8 * scaffold_edge_count
            + 64 * max(control_ids, default=0)
            + (128 * max(_FIELD_WORKING_ENTRIES, patch_count) if patch_count else 0)
            + 4096
        )
        statistics = len(specification.size_compliance.target_statistics)
        depth = (2 * edge_count + max(control_ids, default=0)).bit_length() + 1
        # Coordinate copy and the later feature-rounding scan; per edge its
        # row copy, endpoint gathers and difference, length, midpoint, sample
        # minimum, gradation, ancestry gather and sorted constraint selection;
        # three size-field samples per edge and one per vertex against every
        # controlled patch edge; the scaffold-edge source bank; and per control
        # its sorted edge copy, membership sorts, selection, ratio, reductions
        # and one partition per target statistic.
        scope.charge(
            work=2 * dimension * vertex_count
            + 17 * edge_count
            + (3 * edge_count + vertex_count) * max(1, patch_count)
            + 2 * scaffold_edge_count
            + sum(
                edge_count * (10 + statistics + 2 * depth) + ids * depth
                for ids in control_ids
            )
        )
        vertices = np.asarray(mesh.coordinates, dtype=np.float64)
        edges = entities(mesh, 1)
        a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
        lengths = np.linalg.norm(b - a, axis=1)
        local = np.min(
            prepared.scaffold.sizes(np.concatenate((a, b, 0.5 * (a + b)))).reshape(
                3, -1
            ),
            axis=0,
        )
        endpoint = prepared.scaffold.sizes(vertices)
        gradation = 1.0 + float(
            np.max(np.abs(endpoint[edges[:, 1]] - endpoint[edges[:, 0]]) / lengths)
        )
        ancestor_edges = extraction.entity_parent_rows[1]
        ancestor_dimensions = extraction.entity_parent_dimensions[1]
        old_edge_sources = np.full((scaffold_edge_count,), -1, dtype=np.int64)
        old_edge_sources[features] = constrained_sources
        child_sources = np.where(
            ancestor_dimensions == 1,
            old_edge_sources[np.minimum(ancestor_edges, old_edge_sources.size - 1)],
            -1,
        )
        child_constraints = np.sort(edges[child_sources >= 0], axis=1)
        child_constraint_sources = child_sources[child_sources >= 0]
        requested, achieved, issues = [], [], []
        for control in specification.size_controls:
            match control:
                case UniformSizeControl():
                    pass
                case _:
                    raise TypeError(
                        "dual_quad.size_controls must contain uniform controls; got a different control kind"
                    )
            rows = control_edges(
                control,
                prepared.scaffold,
                np.sort(edges, axis=1),
                local,
                child_constraints,
                child_constraint_sources,
            )
            req, act, failed = graded_size_compliance(
                control,
                specification.size_compliance,
                lengths[rows],
                local[rows],
                gradation,
            )
            requested.extend(req)
            achieved.extend(act)
            issues.extend(failed)
    with coordinate_phase(scope, specification.limits, budget):
        quality = evaluate_cell_quality(mesh, budget=budget)
    minimum_angle = float(np.min(np.asarray(quality.minimum_angle)))
    if specification.quality_target is not None:
        requested.append(("minimum_angle", specification.quality_target.minimum_angle))
        if (
            specification.quality_target.hard
            and minimum_angle < specification.quality_target.minimum_angle
        ):
            issues.append("minimum_angle")
    if triangulation.evidence.status != "ok":
        issues.append(f"refinement_limit:{triangulation.evidence.status}")
    if rounding > admissible:
        issues.append("constraint_rounding")
    feature_rounding = rounding + 64.0 * np.finfo(np.float64).eps * np.max(
        np.abs(vertices), initial=1.0
    )
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        achieved.append(
            (f"protected:{feature.feature_id}:maximum_deviation", feature_rounding)
        )
        if feature_rounding > feature.maximum_deviation:
            issues.append(f"protected:{feature.feature_id}:maximum_deviation")
    achieved.extend(
        (
            ("quadrilateral_count", mesh.topology.entities(2).count),
            ("minimum_angle", minimum_angle),
            (
                "cross_field_objective",
                float(np.asarray(field.optimization.objective)),
            ),
            ("cross_field_status", int(np.asarray(field.optimization.status))),
            (
                "maximum_feature_angle_residual",
                float(np.max(field.feature_residuals, initial=0.0)),
            ),
            ("singular_vertex_count", extraction.singular_vertices.size),
            ("steiner_points", triangulation.evidence.steiner_count + inserted),
        )
    )
    diagnostics = field.optimization.diagnostics
    achieved.extend(
        (
            (
                "cross_field_objective_evaluations",
                int(np.asarray(diagnostics.objective_evaluations)),
            ),
            (
                "cross_field_gradient_evaluations",
                int(np.asarray(diagnostics.gradient_evaluations)),
            ),
            ("cross_field_counts_complete", int(diagnostics.counts_complete)),
        )
    )
    vertex_rows = int(np.size(authority.vertex_rows))
    carrier_rows = prepared.scaffold.source_bounds.size + vertex_rows
    with scope.host_workspace() as phase_storage:
        # Keyed report tuples and strings, plus host vertex IDs.
        phase_storage.set_bound(512 * carrier_rows + 8 * vertex_rows + 4096)
        # One requested and one achieved row per source edge and carrier.
        scope.charge(work=2 * carrier_rows)
        source_requested, source_achieved = source_carrier_evidence(
            prepared.scaffold,
            authority,
            np.asarray(scaffold.vertex_global_ids, dtype=np.int64),
            carrier_deviations,
            source_deviations,
        )
    requested.extend(source_requested)
    achieved.extend(source_achieved)
    compliance = MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(requested),
        achieved=tuple(achieved),
        issues=tuple(issues),
    )
    return _QuadCompliancePhase(compliance)


@final
class _VolumeCompliancePhase(StrictModule):
    compliance: MeshingComplianceReport


def _volume_extraction_compliance(
    mesh: CellMesh,
    specification: VolumeMeshingSpec,
    extraction: DualExtraction,
    construction: VolumeConstruction,
    geometry: CellGeometrySpec,
    associations: tuple[GeometryAssociation, ...],
    mapped_quality: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> _VolumeCompliancePhase:
    """Volume extraction compliance."""
    limits = specification.limits
    control = specification.size_controls[0]
    match control:
        case UniformSizeControl():
            pass
        case _:
            raise TypeError(
                "dual_hex.size_controls must contain a uniform control; got a different control kind"
            )
    # Decomposition and closure work are known before compliance evaluation.
    if construction.work_units + extraction.work_units > limits.maximum_work_units:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "hex.work_units must fit decomposition plus closure within maximum_work_units; got an exhausted budget",
            stage=MeshingStageKind.VOLUME_FILL.value,
        )
    edge_count = mesh.topology.entities(1).count
    vertex_count = mesh.coordinates.shape[0]
    feature_rows = []
    for feature in specification.protected_features:
        entity_set = mesh.entity_set(feature.scope.entity_dimension).entity_set_id
        requested_count = int(np.size(feature.scope.entity_ids))
        feature_rows.append(
            (
                requested_count,
                tuple(
                    int(np.size(association.source_indices))
                    for association in associations
                    if association.target_entity_set_id == entity_set
                    and association.exact
                    and association.source_dimensions is not None
                    and association.source_indices is not None
                ),
            )
        )
    with scope.host_workspace() as phase_storage:
        # Edge banks and one statistic copy (128 bytes per edge), vertex
        # extrema (64 per vertex), and per feature/association membership
        # temporaries, Python ID rows and the present-ID set (128 per row).
        phase_storage.set_bound(
            128 * edge_count
            + 64 * vertex_count
            + sum(
                64 * requested
                + sum(128 * (rows + requested) for rows in association_rows)
                for requested, association_rows in feature_rows
            )
            + 4096
        )
        # Size evidence; per protected feature its requested-ID copy and
        # missing scan, and per exact association row its stratum copies,
        # membership sort, selection, present-set update and residual maximum.
        scope.charge(
            work=_size_evidence_work(
                edge_count,
                vertex_count,
                len(specification.size_compliance.target_statistics),
            )
            + sum(
                2 * requested
                + sum(
                    5 * rows
                    + (rows + requested) * ((rows + requested).bit_length() + 1)
                    for rows in association_rows
                )
                for requested, association_rows in feature_rows
            )
        )
        vertices = np.asarray(mesh.coordinates, dtype=np.float64)
        edges = entities(mesh, 1)
        lengths, growth = edge_size_evidence(vertices, edges)
        requested, achieved, issues = uniform_size_compliance(
            control, specification.size_compliance, lengths, growth
        )
        for feature in specification.protected_features:
            key = f"protected:{feature.feature_id}:maximum_deviation"
            requested_indices = np.asarray(feature.scope.entity_ids, dtype=np.int64)
            entity_set = mesh.entity_set(feature.scope.entity_dimension).entity_set_id
            present: set[int] = set()
            deviation = 0.0
            for association in associations:
                if (
                    association.target_entity_set_id != entity_set
                    or not association.exact
                    or association.source_dimensions is None
                    or association.source_indices is None
                ):
                    continue
                dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
                indices = np.asarray(association.source_indices, dtype=np.int64)
                matched = (dimensions == feature.scope.entity_dimension) & np.isin(
                    indices, requested_indices
                )
                present.update(indices[matched].tolist())
                deviation = max(
                    deviation,
                    float(
                        np.max(np.asarray(association.residuals)[matched], initial=0.0)
                    ),
                )
            missing = tuple(
                int(index) for index in requested_indices if int(index) not in present
            )
            if missing:
                raise MeshingFailure(
                    MeshingFailureCategory.ASSOCIATION_FAILED,
                    "hex.protected_features must retain source-map associations for every requested entity; got missing associations",
                    stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                    entity_ids=missing,
                )
            requested.append((key, feature.maximum_deviation))
            achieved.append((key, deviation))
            if feature.hard and deviation > feature.maximum_deviation:
                issues.append(key)
    counts: dict[str, int] = {}
    for block in mesh.blocks:
        counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
    with coordinate_phase(scope, specification.limits, budget):
        quality = evaluate_cell_quality(mesh, budget=budget)
    if mapped_quality is None:
        with measure_phase(record_phase, "compliance"):
            mapped_quality = exact_mapped_volume_quality(
                mesh, geometry, scope=scope, budget=budget
            )
    scaled_lower, mean_lower, aspect_upper = mapped_quality
    achieved.extend(
        ((f"family_count:{family}", count) for family, count in sorted(counts.items()))
    )
    achieved.extend(
        (
            (
                "hex_cell_fraction",
                counts.get("hexahedron", 0) / mesh.topology.entities(3).count,
            ),
            (
                "minimum_scaled_jacobian",
                float(np.min(np.asarray(quality.scaled_jacobian))),
            ),
            ("minimum_mapped_scaled_jacobian", float(np.min(scaled_lower))),
            ("minimum_mapped_mean_ratio", float(np.min(mean_lower))),
            ("maximum_mapped_aspect_ratio", float(np.max(aspect_upper))),
            ("singular_vertex_count", extraction.singular_vertices.size),
            ("decomposition_work_units", construction.work_units),
            ("closure_work_units", extraction.work_units),
            (
                "decomposition_maximum_radius_edge",
                construction.quality.maximum_radius_edge,
            ),
            (
                "decomposition_minimum_dihedral_degrees",
                construction.quality.minimum_dihedral,
            ),
            *(
                (f"decomposition_unmet:{criterion}:{reason}", count)
                for criterion, reason, count in construction.unmet
            ),
        )
    )
    compliance = MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(requested),
        achieved=tuple(achieved),
        issues=tuple(issues),
    )
    return _VolumeCompliancePhase(compliance)


@final
class _MappedGridConstructionPhase(StrictModule):
    mapped: MappedGridHexConstruction
    grid: GridHexConstruction | _SourceBlockTensorGrid
    grid_identity: str = eqx.field(static=True)
    bound: float = eqx.field(static=True)


def _mapped_grid_construction(
    source: _MappedHexSourceBinding,
    specification: VolumeMeshingSpec,
    prepared: PreparedMappedGridHex,
    size: float,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    *,
    association_inputs: list[_ConstructedTensorGrid] | None = None,
) -> _MappedGridConstructionPhase:
    """Mapped grid construction."""
    control = specification.size_controls[0]
    assert isinstance(control, UniformSizeControl)
    maximum_size = min(
        size + specification.size_compliance.tolerance(size),
        float("inf") if control.maximum_size is None else control.maximum_size + specification.size_compliance.tolerance(control.maximum_size),
    )
    with measure_phase(record_phase, "construction"), scope.host_workspace() as sizing_workspace:
        roots = sum(block.cell_count for block in source.domain.reference_mesh.blocks)
        sizing_workspace.set_bound((48 * roots if prepared.source_frame is not None else 0) + 24)
        root_axis_bounds = None if prepared.source_frame is None else family_host_array((roots, 3), np.float64, scope=scope)
        root_reference_axis_bounds = None if prepared.source_frame is None else family_host_array((roots, 3), np.float64, scope=scope)
        reference_axis_bounds = family_host_array((3,), np.float64, scope=scope)
        bound = mapped_source_lipschitz_bound(
            source.domain.reference_mesh,
            source.domain.source_geometry,
            root_axis_bounds=root_axis_bounds,
            reference_axis_bounds=reference_axis_bounds,
            root_reference_axis_bounds=root_reference_axis_bounds,
            scope=scope,
            budget=budget,
        )
        if prepared.source_frame is None:
            grid = generate_integer_grid_hexes(
                source.reference.complex,
                source.domain,
                prepared.grid_schedule,
                specification.limits,
                size,
                maximum_size=maximum_size,
                desired_axis_scale=bound,
                reference_axis_bounds=tuple(float(value) for value in reference_axis_bounds),
                record_phase=record_phase,
                scope=scope,
                budget=budget,
            )
            grid_identity = grid.construction_id
        else:
            grid = _extract_source_block_tensor_grid(
                prepared.source_frame,
                None,
                source.domain.cell_regions,
                specification.limits,
                prepared.grid_schedule.maximum_depth,
                source_connection=prepared.source_connection,
                record_phase=record_phase,
                scope=scope,
                budget=budget,
                axis_bounds=root_axis_bounds,
                desired_axis_bounds=root_reference_axis_bounds,
                physical_lipschitz_bound=bound,
                target_size=size,
                maximum_size=maximum_size,
            )
            grid_identity = grid.grid_id
        del root_axis_bounds, root_reference_axis_bounds, reference_axis_bounds
    with measure_phase(record_phase, "curving"):
        match grid:
            case GridHexConstruction():
                mapped = realize_mapped_grid_hexes(
                    grid,
                    source.domain.reference_mesh,
                    source.domain.source_geometry,
                    source.domain.cell_regions,
                    specification.limits,
                    scope=scope,
                    budget=budget,
                )
            case _SourceBlockTensorGrid():
                match prepared.source_frame:
                    case PreparedSourceHexFrame() as frame:
                        mapped = _extract_source_tensor_frame_grid(
                            frame,
                            grid,
                            source.domain.cell_regions,
                            specification.limits,
                            scope=scope,
                            budget=budget,
                            quality_policy=prepared.grid_schedule,
                            association_inputs=association_inputs,
                        )
                    case _:
                        raise ValueError(
                            "mapped_grid.source_frame must accompany source-block integer topology; got no frame"
                        )
            case _:
                raise TypeError(
                    "mapped_grid.topology must be grid or source-block integer topology; got an unsupported topology"
                )
    return _MappedGridConstructionPhase(mapped, grid, grid_identity, bound)


@final
class _MappedGridCompliancePhase(StrictModule):
    compliance: MeshingComplianceReport


def _mapped_grid_compliance(
    mesh: CellMesh,
    mapped: MappedGridHexConstruction,
    grid: GridHexConstruction | _SourceBlockTensorGrid,
    bound: float,
    control: UniformSizeControl,
    specification: VolumeMeshingSpec,
    prepared: PreparedMappedGridHex,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    /,
) -> _MappedGridCompliancePhase:
    """Mapped grid compliance."""
    edge_count = mesh.topology.entities(1).count
    vertex_count = mesh.coordinates.shape[0]
    with (
        measure_phase(record_phase, "compliance"),
        scope.host_workspace() as workspace,
    ):
        # Host point and edge copies, endpoint gathers and differences,
        # lengths, a partition copy, and per-vertex extrema banks.
        workspace.set_bound(128 * edge_count + 72 * vertex_count + 4096)
        # Size evidence plus the three per-cell mapped quality reductions.
        scope.charge(
            work=_size_evidence_work(
                edge_count,
                vertex_count,
                len(specification.size_compliance.target_statistics),
            )
            + 3 * mesh.topology.entities(3).count
        )
        lengths, growth = edge_size_evidence(
            np.asarray(mesh.coordinates, dtype=np.float64), entities(mesh, 1)
        )
        requested, achieved, issues = uniform_size_compliance(
            control, specification.size_compliance, lengths, growth
        )
        del lengths
        minimum_scaled = float(np.min(mapped.scaled_jacobian_lower))
        minimum_mean = float(np.min(mapped.mean_ratio_lower))
        aspect = float(np.max(mapped.aspect_ratio_upper))
        requested.extend(
            (
                (
                    "minimum_mapped_scaled_jacobian",
                    prepared.grid_schedule.minimum_scaled_jacobian,
                ),
                (
                    "minimum_mapped_mean_ratio",
                    prepared.grid_schedule.minimum_mean_ratio,
                ),
                (
                    "maximum_mapped_aspect_ratio",
                    prepared.grid_schedule.maximum_aspect_ratio,
                ),
            )
        )
        achieved.extend(
            (
                ("minimum_mapped_scaled_jacobian", minimum_scaled),
                ("minimum_mapped_mean_ratio", minimum_mean),
                ("maximum_mapped_aspect_ratio", aspect),
                ("hexahedron_count", mesh.topology.entities(3).count),
                ("source_placement_lipschitz_bound", bound),
            )
        )
        if minimum_scaled < prepared.grid_schedule.minimum_scaled_jacobian:
            issues.append("minimum_mapped_scaled_jacobian")
        if minimum_mean < prepared.grid_schedule.minimum_mean_ratio:
            issues.append("minimum_mapped_mean_ratio")
        if aspect > prepared.grid_schedule.maximum_aspect_ratio:
            issues.append("maximum_mapped_aspect_ratio")
        if prepared.source_frame is not None:
            achieved.extend(
                (
                    (
                        "varying_source_root_count",
                        int(np.count_nonzero(prepared.source_frame.varying_roots)),
                    ),
                    (
                        "source_frame_connection_count",
                        prepared.source_frame.face_connections.shape[0],
                    ),
                )
            )
        if prepared.source_connection is not None:
            achieved.extend(
                (
                    (
                        "source_chart_cycle_count",
                        prepared.source_connection.holonomy.cycle_edges.size,
                    ),
                    (
                        "source_chart_singular_cycle_count",
                        prepared.source_connection.holonomy.singular_cycles.size,
                    ),
                )
            )
        if isinstance(grid, _SourceBlockTensorGrid):
            achieved.extend(
                (
                    ("integer_source_root_count", grid.root_intervals.shape[0]),
                    ("integer_maximum_axis_count", int(np.max(grid.root_intervals))),
                )
            )
        for feature in specification.protected_features:
            requested.append(
                (
                    f"protected:{feature.feature_id}:maximum_deviation",
                    feature.maximum_deviation,
                )
            )
            achieved.append((f"protected:{feature.feature_id}:maximum_deviation", 0.0))
        compliance = MeshingComplianceReport(
            specification.specification_id,
            requested=tuple(requested),
            achieved=tuple(achieved),
            issues=tuple(issues),
        )
    return _MappedGridCompliancePhase(compliance)


@final
class _GridRidgePhase(StrictModule):
    associations: list[GeometryAssociation]
    ridge_names: dict[int, str] = eqx.field(static=True)


def _grid_ridge_associations(
    source: sources_owner.NativePlcSource,
    old: VolumeConstruction,
    grid: GridHexConstruction,
    points: np.ndarray,
    edge_vertices: np.ndarray,
    edge_facets: list[set[int]],
    facet_names: dict[int, str],
    ids: tuple[np.ndarray, ...],
    sets: tuple[EntitySet, ...],
    rounding: float,
    associations: list[GeometryAssociation],
    /,
    *,
    scope: NativeExecutionScope,
) -> _GridRidgePhase:
    """Grid ridge associations.

    Source segments that vary along one chart axis are indexed by that axis
    and their fixed chart coordinates, sorted by lower endpoint with running
    upper maxima. A ridge visits only the indexed segments its interval can
    stab and inherits the overlapping ones in source association order.
    """
    old_edges = entities(old.mesh, 1)
    old_vertex_count = old.mesh.coordinates.shape[0]
    old_edge_set = old.mesh.topology.entities(1)
    source_edge_count = sum(
        association.target_global_ids.shape[0]
        for association in old.associations
        if association.target_entity_set_id == old_edge_set.entity_set_id
    )
    edge_count = edge_vertices.shape[0]
    depth = max(source_edge_count, edge_count).bit_length() + 1
    # Per old vertex its 3x3 frame system, solve and host copy; per old edge
    # its row, identifier and keyed map entry; per source edge row its
    # endpoint gather, identity, orientation, varying-axis test, line key,
    # extrema and the per-line lower-endpoint sort with running maxima; the
    # chart transform; per grid edge its facet test, endpoints, ridge axis,
    # line lookup, two bisections, the sorted association row and columns.
    scope.charge(
        work=13 * old_vertex_count
        + 3 * old_edges.shape[0]
        + 16 * source_edge_count
        + 2 * source_edge_count * depth
        + 9 * points.shape[0]
        + edge_count * (12 + 3 * depth)
    )
    solve = solve_small_linear(
        SmallLinearSolvePlan(3),
        np.broadcast_to(grid.frame.basis.T, (old_vertex_count, 3, 3)),
        old.mesh.coordinates,
    )
    host_solution, host_success = jax.device_get((solve.value, solve.successful))
    old_points = np.asarray(host_solution, dtype=np.float64)
    if not np.all(np.asarray(host_success, dtype=np.bool_)):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "grid.source_edge_pullback must solve every source chart coordinate; got an unsuccessful linear solve",
        )
    old_ids = np.asarray(old_edge_set.entity_ids, dtype=np.int64)
    old_rows = {identifier: row for row, identifier in enumerate(old_ids.tolist())}
    source_edges: list[tuple[np.ndarray, str, int]] = []
    for association in old.associations:
        if association.target_entity_set_id != old_edge_set.entity_set_id:
            continue
        target_ids, orientations = jax.device_get(
            (association.target_global_ids, association.orientations)
        )
        host_target_ids = np.asarray(target_ids, dtype=np.int64)
        host_orientations = np.asarray(orientations, dtype=np.int8)
        for row, identifier in enumerate(host_target_ids.tolist()):
            source_edges.append(
                (
                    old_points[old_edges[old_rows[identifier]]],
                    association.source_entities.entity_id(row),
                    int(host_orientations[row]),
                )
            )
    # A segment can cover a ridge only when it varies along exactly the ridge
    # axis and both endpoints share the ridge's fixed chart coordinates. NaN
    # extrema never satisfy the overlap comparisons, so they are not indexed.
    line_rows: dict[
        tuple[int, tuple[float, ...]], list[tuple[np.float64, np.float64, int]]
    ] = {}
    for index, (segment, _, _) in enumerate(source_edges):
        varying = np.flatnonzero(segment[0] != segment[1])
        if varying.size != 1:
            continue
        axis = int(varying[0])
        first, last = np.min(segment[:, axis]), np.max(segment[:, axis])
        if np.isnan(first) or np.isnan(last):
            continue
        fixed = [coordinate for coordinate in range(3) if coordinate != axis]
        line_rows.setdefault((axis, tuple(segment[0, fixed].tolist())), []).append(
            (first, last, index)
        )
    lines: dict[
        tuple[int, tuple[float, ...]],
        tuple[list[tuple[np.float64, np.float64, int]], list[float], list[float]],
    ] = {}
    for key, rows in line_rows.items():
        rows.sort(key=lambda item: item[0])
        lines[key] = (
            rows,
            [float(item[0]) for item in rows],
            list(accumulate((float(item[1]) for item in rows), max)),
        )
    del line_rows
    chart = points @ grid.frame.basis.T
    # Plan every ridge before visiting segments so the exact visit count is
    # admitted first. Planning stops at the first multiaxis ridge, which the
    # traversal below refuses in its original order.
    plans: list[
        tuple[
            np.ndarray,
            int,
            np.float64,
            list[tuple[np.float64, np.float64, int]],
            int,
            int,
        ]
    ] = []
    multiaxis_row = -1
    visits = 0
    for row, adjacent in enumerate(edge_facets):
        if len(adjacent) < 2:
            continue
        endpoints = chart[edge_vertices[row]]
        variable = np.flatnonzero(endpoints[0] != endpoints[1])
        if variable.size != 1:
            multiaxis_row = row
            break
        axis = int(variable[0])
        fixed = [coordinate for coordinate in range(3) if coordinate != axis]
        lower, upper = np.min(endpoints[:, axis]), np.max(endpoints[:, axis])
        line = lines.get((axis, tuple(endpoints[0, fixed].tolist())))
        if line is None:
            plans.append((endpoints, axis, lower, [], 0, 0))
            continue
        rows, firsts, reach = line
        # Rows before ``start`` end at or below ``lower``; rows from ``stop``
        # begin at or above ``upper``.
        start, stop = bisect_right(reach, lower), bisect_left(firsts, upper)
        plans.append((endpoints, axis, lower, rows, start, stop))
        visits += max(0, stop - start)
    # Per visited segment its overlap test, relative orientation, interval
    # row, and its share of the source-order and covering-interval sorts.
    scope.charge(work=visits * (8 + depth))
    edge_names: dict[int, str] = {}
    edge_orientations: dict[int, int] = {}
    ridge_names: dict[int, str] = {}
    planned = iter(plans)
    for row, adjacent in enumerate(edge_facets):
        if not adjacent:
            continue
        if len(adjacent) == 1:
            edge_names[row] = facet_names[next(iter(adjacent))]
            edge_orientations[row] = 0
            continue
        if row == multiaxis_row:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "grid.ridge.logical_coordinates must vary along one axis; got a multiaxis ridge",
            )
        endpoints, axis, lower, rows, start, stop = next(planned)
        hits = sorted(
            (index, first, last)
            for first, last, index in rows[start:stop]
            if last > lower
        )
        intervals: dict[str, list[tuple[float, float, int]]] = {}
        for index, first, last in hits:
            segment, entity, orientation = source_edges[index]
            relative = orientation * (
                1
                if (segment[1, axis] - segment[0, axis])
                * (endpoints[1, axis] - endpoints[0, axis])
                > 0.0
                else -1
            )
            intervals.setdefault(entity, []).append((float(first), float(last), relative))
        del hits
        upper = np.max(endpoints[:, axis])
        complete = []
        for entity, pieces in sorted(intervals.items()):
            end = float(lower)
            orientation = pieces[0][2]
            for first, last, _ in sorted(pieces):
                if first > end:
                    break
                end = max(end, last)
            if end >= upper:
                complete.append((entity, orientation))
        del intervals
        if len(complete) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "grid.ridge.source_chain must identify one covering source chain; got missing or ambiguous coverage",
                entity_ids=(int(ids[1][row]),),
            )
        edge_names[row], edge_orientations[row] = complete[0]
        ridge_names[row] = edge_names[row]
    edge_rows = np.asarray(sorted(edge_names), dtype=np.int64)
    associations.append(
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            source.source_revision,
            sets[1].entity_set_id,
            ids[1][edge_rows],
            tuple(edge_names[row] for row in edge_rows.tolist()),
            np.full((edge_rows.size,), rounding, dtype=np.float64),
            orientations=np.asarray(
                [edge_orientations[row] for row in edge_rows.tolist()],
                dtype=np.int8,
            ),
        )
    )
    return _GridRidgePhase(associations, ridge_names)


@final
class _GridVertexPhase(StrictModule):
    associations: list[GeometryAssociation]


def _grid_vertex_associations(
    source: sources_owner.NativePlcSource,
    grid: GridHexConstruction,
    points: np.ndarray,
    edge_vertices: np.ndarray,
    ridge_names: dict[int, str],
    vertex_facets: list[set[int]],
    facet_names: dict[int, str],
    cells: np.ndarray,
    ids: tuple[np.ndarray, ...],
    sets: tuple[EntitySet, ...],
    rounding: float,
    associations: list[GeometryAssociation],
    /,
    *,
    scope: NativeExecutionScope,
) -> _GridVertexPhase:
    """Grid vertex associations."""
    source_vertex_count = grid.frame.solve.value.shape[0]
    knot_depth = max(len(knots) for knots in grid.axis_knots).bit_length() + 1
    vertex_count = points.shape[0]
    # Per source vertex its logical copy, three knot searches and key row and
    # the corner-name entry; per ridge edge two vertex incidences; per cell
    # eight region incidences; per grid vertex its incidence sets, key lookup,
    # resolution and association columns.
    scope.charge(
        work=source_vertex_count * (8 + 3 * knot_depth)
        + 2 * len(ridge_names)
        + 9 * cells.shape[0]
        + 8 * vertex_count
    )
    raw_logical = np.asarray(jax.device_get(grid.frame.solve.value), dtype=np.float64)
    raw_keys = np.stack(
        [
            np.searchsorted(grid.axis_knots[axis], raw_logical[:, axis])
            * grid.subdivisions
            for axis in range(3)
        ],
        axis=1,
    )
    corner_names = {
        tuple(key): sources_owner.source_entity_id(source.source_revision, "vertex", row)
        for row, key in enumerate(raw_keys.tolist())
    }
    vertex_ridges: list[set[str]] = [set() for _ in range(points.shape[0])]
    for edge, entity in ridge_names.items():
        for vertex in edge_vertices[edge].tolist():
            vertex_ridges[vertex].add(entity)
    vertex_regions: list[set[int]] = [set() for _ in range(points.shape[0])]
    for cell, region in zip(cells.tolist(), grid.cell_regions.tolist(), strict=True):
        for vertex in cell:
            vertex_regions[vertex].add(region)
    vertex_names = []
    for row, key in enumerate(grid.logical_vertices.tolist()):
        if tuple(key) in corner_names:
            entity = corner_names[tuple(key)]
        elif len(vertex_ridges[row]) == 1:
            entity = next(iter(vertex_ridges[row]))
        elif len(vertex_facets[row]) == 1:
            entity = facet_names[next(iter(vertex_facets[row]))]
        elif not vertex_facets[row] and len(vertex_regions[row]) == 1:
            entity = sources_owner.source_entity_id(
                source.source_revision, "region", next(iter(vertex_regions[row]))
            )
        else:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "grid.vertex.source_ancestry must identify a corner, ridge, facet, or region; got an unresolved junction",
                entity_ids=(int(ids[0][row]),),
            )
        vertex_names.append(entity)
    associations.append(
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            source.source_revision,
            sets[0].entity_set_id,
            ids[0],
            tuple(vertex_names),
            np.full((points.shape[0],), rounding, dtype=np.float64),
        )
    )
    return _GridVertexPhase(associations)


@final
class _QuadComplianceInputs(StrictModule):
    mesh: CellMesh
    prepared: PreparedDualQuad
    extraction: DualExtraction
    scaffold: CellMesh
    features: np.ndarray
    constrained_sources: np.ndarray
    field: SurfaceCrossField
    rounding: float = eqx.field(static=True)
    admissible: float = eqx.field(static=True)
    triangulation: ConstrainedDelaunayTriangulation
    inserted: int = eqx.field(static=True)
    authority: ExactPlcCellGeometrySource
    carrier_deviations: np.ndarray
    source_deviations: np.ndarray
