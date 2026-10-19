#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native parametric-surface route: shared curves, chart CDT, physical refinement.

Admission compiles the request against the meshing domain's strata
(`compile_surface_domain`); execution runs `generate_surface`, associates
every face with its source surface (oriented against the source normal) and
every feature-curve edge with its curve (oriented against the curve tangent),
checks the physical size, quality and fidelity requests independently of the
refinement criteria, audits the mesh (watertight when the selected patches
close up) and publishes it.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

import phydrax.discretization as cell_complex_owner
import phydrax.discretization as cell_geometry_owner
import phydrax.discretization as cell_mesh_owner
from phydrax.meshing._association_state import (
    entity_rows,
    GeometryAssociation,
    target_dimension,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._host_banks import entities
from phydrax.meshing.providers._native_route_policy import publication_policy

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import NativeExecutionScope, NativeHostStorageWorkspace
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._mapped_geometry_measure import prepare_mapped_edge_arc_length
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._domain_boundary import MeshingDomainBoundarySource
from ...geometry._meshing_domain import MeshingDomain, PatchCurveUse, PatchPoleUse
from ...geometry._surface_source_support import (
    prepare_surface_source_root_atlas,
    PreparedSurfaceSourceSupport,
    restrict_surface_source_atlas_geometry,
    SurfaceNativeRestrictionBoundarySource,
    SurfaceRestrictionRows,
    SurfaceSourceCharts,
    SurfaceSourceRootAtlas,
)
from ...geometry.brep._patches import BSplineSurfacePatch, LineCurve
from ...typing import checked
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._canonical import canonicalize_cell_mesh
from .._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
)
from .._controls import FeatureKind
from .._distributed_generation import (
    execute_distributed_surface_initial,
    PreparedDistributedSurfaceGeneration,
)
from .._domain import compile_surface_domain, CompiledSurfaceDomain, surface_domain_issues
from .._dual_extraction import (
    DualExtraction,
)
from .._initial_certification import publish_distributed_surface_initial
from .._measurements import measure_phase, NativeMeshingPhaseRecorder
from .._metric import metric_edge_lengths, metric_simplex_quality
from .._organization_records import (
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionBoundaryEvidence,
)
from .._quad_generation import (
    extract_surface_quads,
    prepare_surface_cross_field,
    remap_dual_metadata,
)
from .._quality import evaluate_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    SizeControlStrength,
    UniformSizeControl,
)
from .._surface_construction import SurfaceConstruction
from .._surface_generation import generate_surface
from .._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from . import _native_sources as sources_owner
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
    edge_growth_evidence,
    edge_size_evidence,
    simplex_entity_limits,
    uniform_size_compliance,
    unique_edges,
)
from ._native_route_metadata import mesh_entity_scope
from ._native_schedules import NativeSurfaceSchedule
from ._native_surface_association import associations, simplex_rows


@checked
def parametric_surface_support_issues(
    source: sources_owner.NativeSurfaceSource, specification: SurfaceMeshingSpec, /
) -> list[str]:
    """Physical requests of one surface specification this route cannot enforce."""

    unsupported: list[str] = []
    target = specification.target
    families = target.cell_families
    kinds = set((*families.required, *families.preferred))
    if kinds == {"quadrilateral"}:
        selected = source.domain.resolve_indices(
            2, np.asarray(specification.scope.entity_ids)
        )
        for patch in selected.tolist():
            surface = source.domain.patches[patch]
            if not isinstance(surface.surface, BSplineSurfacePatch):
                unsupported.append("rational spline patch maps for source quads")
            if surface.reversed:
                unsupported.append(
                    "positive source-UV orientation for source quad extraction"
                )
            if any(
                not isinstance(use, PatchCurveUse)
                or not isinstance(use.pcurve, LineCurve)
                for loop in surface.loops
                for use in loop
            ):
                unsupported.append("affine UV trim coedges for source quad coverage")
        if target.ambient_dimension != 3:
            unsupported.append("surface quads in ambient dimension three")
    elif kinds == {"triangle"}:
        if target.ambient_dimension != 3 or target.geometry_order != 1:
            unsupported.append("affine triangles in ambient dimension three")
    else:
        unsupported.append("a pure triangular or quadrilateral surface target")
    if families.allowed_transitions or families.allow_mixed:
        unsupported.append("mixed-cell transition policies")
    background = specification.background_metric
    if background is not None and (
        background.mesh.topological_dimension not in (2, 3)
        or background.mesh.ambient_dimension != 3
    ):
        unsupported.append("an ambient-3 affine simplex background metric source cover")
    unsupported.extend(surface_domain_issues(source.domain, specification))
    return unsupported


@final
class PreparedParametricSurface(StrictModule, NonTrainableState):
    """Compiled constraints, worksets and refinement aim of one admitted request."""

    compiled: CompiledSurfaceDomain
    schedule: NativeSurfaceSchedule
    generation: PreparedDistributedSurfaceGeneration | None
    minimum_angle: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativeSurfaceSource,
        specification: SurfaceMeshingSpec,
        schedule: NativeSurfaceSchedule,
        /,
        *,
        initial_partition: PreparedDistributedSurfaceGeneration | None = None,
        scope: NativeExecutionScope,
    ) -> None:
        budget = coordinate_budget(specification.limits)
        with coordinate_phase(scope, specification.limits, budget):
            compiled = compile_surface_domain(source.domain, specification, budget=budget)
        aim = math.radians(schedule.quality_angle_degrees)
        quality = specification.quality_target
        if quality is not None:
            aim = max(aim, quality.minimum_angle)
        generation = initial_partition
        if generation is not None:
            generation.__post_init__()
            if (
                generation.compiled.compiled_id != compiled.compiled_id
                or generation.specification.specification_id
                != specification.specification_id
                or generation.schedule.schedule_id != schedule.schedule_id
            ):
                raise ValueError(
                    "initial_partition.binding must match source, specification, and schedule; got a different binding"
                )
        self.compiled = compiled
        self.schedule = schedule
        self.minimum_angle = aim
        self.generation = generation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-parametric-surface",
                "source": source.binding_id,
                "compiled": compiled.compiled_id,
                "schedule": schedule.schedule_id,
                "minimum_angle": aim,
                "generation": None
                if generation is None
                else (
                    array_tree_fingerprint(generation.patch_owners),
                    generation.layout.vertex_capacity,
                    generation.layout.cell_capacity,
                    generation.layout.candidate_capacity,
                    generation.layout.maximum_work_units,
                    generation.maximum_metadata_bytes,
                    generation.partition_count,
                ),
            }
        )


@checked
def closed(compiled: CompiledSurfaceDomain, /) -> bool:
    """Whether every curve of the selected patches is used exactly twice."""

    domain = compiled.domain
    counts = {curve: 0 for curve in compiled.curves.tolist()}
    for patch in compiled.patches.tolist():
        for loop in domain.patches[patch].loops:
            for use in loop:
                match use:
                    case PatchCurveUse():
                        counts[use.curve] += 1
                    case PatchPoleUse():
                        pass
                    case _:
                        raise TypeError(
                            "compiled.domain.loops must contain curve or pole uses; got an unsupported use"
                        )
    return all(count == 2 for count in counts.values())


def _patch_edges(construction: SurfaceConstruction, patches: np.ndarray, /) -> np.ndarray:
    return unique_edges(
        construction.triangles[np.isin(construction.triangle_patches, patches)],
        "triangle",
    )


@final
class _SurfaceControlSource(StrictModule):
    entities: np.ndarray
    dimension: int


def _resolve_surface_control(
    control: UniformSizeControl | CurvatureSizeControl | ProximitySizeControl,
    domain: MeshingDomain,
    /,
) -> _SurfaceControlSource:
    match control:
        case ProximitySizeControl():
            source_entities = np.unique(
                np.concatenate(
                    (
                        domain.resolve_indices(
                            2,
                            np.asarray(
                                control.source_scope.global_entity_ids, dtype=np.int64
                            ),
                        ),
                        domain.resolve_indices(
                            2,
                            np.asarray(
                                control.target_scope.global_entity_ids, dtype=np.int64
                            ),
                        ),
                    )
                )
            )
            dimension = 2
        case UniformSizeControl() | CurvatureSizeControl():
            source_entities = domain.resolve_indices(
                control.scope.entity_dimension,
                np.asarray(control.scope.global_entity_ids, dtype=np.int64),
            )
            if control.scope.entity_dimension == 3:
                source_entities = np.unique(
                    np.asarray(
                        [
                            patch
                            for region in source_entities.tolist()
                            for patch, _ in domain.regions[region].boundary
                        ],
                        dtype=np.int64,
                    )
                )
            dimension = control.scope.entity_dimension
        case _:
            raise TypeError(
                f"specification.size_controls must contain supported controls; got {type(control).__name__}"
            )
    return _SurfaceControlSource(source_entities, dimension)


def _triangle_size_compliance(
    specification: SurfaceMeshingSpec,
    construction: SurfaceConstruction,
    domain: MeshingDomain,
    compiled: CompiledSurfaceDomain,
    budget: CoordinateEnclosureBudget,
    /,
) -> _SurfaceControlCompliance:
    vertices = construction.vertices
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    for control in specification.size_controls:
        source_entities = _resolve_surface_control(control, domain).entities
        match control:
            case UniformSizeControl():
                edges = (
                    construction.curve_edges[
                        np.isin(construction.curve_edge_curves, source_entities)
                    ]
                    if control.scope.entity_dimension == 1
                    else _patch_edges(construction, source_entities)
                )
                lengths, growth = edge_size_evidence(vertices, edges)
                control_requested, control_achieved, control_issues = (
                    uniform_size_compliance(
                        control, specification.size_compliance, lengths, growth
                    )
                )
                requested.extend(control_requested)
                achieved.extend(control_achieved)
                issues.extend(control_issues)
                continue
            case CurvatureSizeControl() | ProximitySizeControl():
                pass
            case _:
                raise TypeError(
                    f"specification.size_controls must contain supported controls; got {type(control).__name__}"
                )
        key = f"source_size:{control.control_id}"
        ratios = []
        for patch in source_entities.tolist():
            selected = construction.triangle_patches == patch
            charts = np.mean(construction.triangle_parameters[selected], axis=1)
            points = domain.evaluate(
                np.full((charts.shape[0],), patch), charts, budget=budget
            )
            sizes = compiled.source_sizing.evaluate(
                domain, patch, charts, points, budget=budget
            )
            corners = vertices[construction.triangles[selected]]
            lengths = np.linalg.norm(
                np.roll(corners, -1, axis=1) - np.roll(corners, -2, axis=1), axis=2
            )
            ratios.append(np.max(lengths, axis=1) / sizes)
        maximum_ratio = float(np.max(np.concatenate(ratios), initial=0.0))
        requested.append((f"{key}:maximum_local_edge_ratio", 1.0))
        achieved.append((f"{key}:maximum_local_edge_ratio", maximum_ratio))
        if (
            control.strength is SizeControlStrength.HARD
            and maximum_ratio > 1 + specification.size_compliance.relative_tolerance
        ):
            issues.append(key)
        match control:
            case CurvatureSizeControl():
                values = construction.triangle_normal_bounds[
                    np.isin(construction.triangle_patches, source_entities)
                ]
                normal_bound = float(np.max(values, initial=0.0))
                requested.append((f"{key}:normal_angle", control.normal_angle))
                achieved.append((f"{key}:normal_angle", normal_bound))
                if (
                    control.strength is SizeControlStrength.HARD
                    and normal_bound > control.normal_angle
                ):
                    issues.append(f"{key}:normal_angle")
            case ProximitySizeControl():
                pass
            case _:
                raise TypeError(
                    f"specification.size_controls must contain curvature or proximity controls; got {type(control).__name__}"
                )
    return _SurfaceControlCompliance(tuple(requested), tuple(achieved), tuple(issues))


def _compliance(
    specification: SurfaceMeshingSpec,
    construction: SurfaceConstruction,
    domain: MeshingDomain,
    compiled: CompiledSurfaceDomain,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> MeshingComplianceReport:
    vertices = construction.vertices
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    controls = _triangle_size_compliance(
        specification, construction, domain, compiled, budget
    )
    requested.extend(controls.requested)
    achieved.extend(controls.achieved)
    issues.extend(controls.issues)
    quality = specification.quality_target
    achieved.append(("minimum_angle", construction.minimum_angle))
    if quality is not None:
        requested.append(("minimum_angle", quality.minimum_angle))
        if quality.hard and construction.minimum_angle < quality.minimum_angle:
            issues.append(f"minimum_angle:{quality.target_id}")
    for feature in specification.protected_features:
        source_entities = domain.resolve_indices(
            feature.scope.entity_dimension,
            np.asarray(feature.scope.global_entity_ids, dtype=np.int64),
        )
        match feature.feature_kind:
            case FeatureKind.SURFACE:
                values = construction.triangle_deviation_bounds[
                    np.isin(construction.triangle_patches, source_entities)
                ]
            case FeatureKind.CURVE:
                values = construction.curve_edge_deviation_bounds[
                    np.isin(construction.curve_edge_curves, source_entities)
                ]
            case FeatureKind.CORNER:
                # Corners are source vertices of every construction.
                values = np.zeros((1,), dtype=np.float64)
            case _:
                raise TypeError(
                    "specification.protected_features must contain corners, curves, or surfaces; got an unsupported feature"
                )
        key = f"protected:{feature.feature_id}:maximum_deviation"
        deviation = float(np.max(values, initial=0.0))
        requested.append((key, feature.maximum_deviation))
        achieved.append((f"{key}:continuous_interpolation_bound", deviation))
        achieved.append((key, deviation))
        if feature.hard and deviation > feature.maximum_deviation:
            issues.append(f"protected:{feature.feature_id}")
    achieved.extend(
        (
            ("refinement_rounds", construction.rounds),
            ("inserted_points", construction.inserted),
            ("refused_points", construction.refused),
            ("physical_flips", construction.flips),
            ("geometry_queries", construction.geometry_queries),
            ("work_units", construction.work_units),
            ("oversized_triangles", int(np.sum(construction.unresolved))),
            ("incomplete_refinement", float(construction.incomplete_refinement)),
        )
    )
    metric = compiled.source_sizing.metric
    if metric is not None:
        values, work = metric.sample(
            vertices,
            specification.limits.maximum_work_units - construction.work_units,
            budget=budget,
        )
        edges = unique_edges(construction.triangles, "triangle")
        lengths = np.asarray(
            metric_edge_lengths(values, vertices, edges), dtype=np.float64
        )
        bound = metric.control.maximum_metric_edge_length or 1.0
        maximum = float(np.max(lengths, initial=0.0))
        quality_values = np.asarray(
            metric_simplex_quality(values, vertices, construction.triangles)
        )
        requested.append(("maximum_metric_edge_length", bound))
        achieved.extend(
            (
                ("maximum_metric_edge_length", maximum),
                ("minimum_metric_mean_ratio", float(np.min(quality_values))),
                ("metric_compliance_location_pairs", work),
            )
        )
        if maximum > bound * (1 + specification.size_compliance.relative_tolerance):
            issues.append("maximum_metric_edge_length")
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


def _certification_request(
    source: sources_owner.NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    compiled: CompiledSurfaceDomain,
    construction: SurfaceConstruction,
    /,
) -> NativeCertificationRequest:
    """Independent global embedding and continuous two-sided source acceptance.

    The source owner independently verifies the full oriented chart chain,
    including collapsed pole triangles and trim ribbons, and recomputes
    Taylor bounds. Nodal residuals alone never certify this request.
    """

    bounded = compiled.patch_deviations[np.isfinite(compiled.patch_deviations)]
    tolerance = float(np.min(bounded) if bounded.size else np.min(compiled.patch_sizes))
    return NativeCertificationRequest(
        MeshCertificationSchedule("surface"),
        source.source_id,
        source.source_revision,
        specification.limits,
        fidelity_source=MeshingDomainBoundarySource(
            source.domain,
            tuple(compiled.patches.tolist()),
            chart_triangulations=construction.chart_triangulations,
        ),
        fidelity_tolerance=tolerance,
    )


@final
class _SurfaceFaceSemantics(StrictModule):
    patches: tuple[MeshPatch, ...]
    zones: tuple[MeshZone, ...]
    face_ids: np.ndarray
    source_patches: np.ndarray


@final
class _SurfaceCurveSemantics(StrictModule):
    patches: tuple[MeshPatch, ...]


def _surface_face_semantics(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    mesh: cell_mesh_owner.CellMesh,
    construction: SurfaceConstruction,
    /,
) -> _SurfaceFaceSemantics:
    face_set = mesh.entity_set(2)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    rows = simplex_rows(
        construction.triangles,
        np.concatenate(
            tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
        ),
    )
    source_patches = construction.triangle_patches[rows]

    patches: list[MeshPatch] = []
    for patch in np.unique(source_patches).tolist():
        controls = tuple(
            control
            for control in specification.patch_controls
            if control.scope.entity_dimension == 2
            and patch
            in domain.resolve_indices(
                2, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
            )
        )
        pair = domain.patch_regions[patch]
        source_pair = (
            None if pair[0] < 0 else domain.entity_id(3, int(pair[0])),
            None if pair[1] < 0 else domain.entity_id(3, int(pair[1])),
        )
        names = tuple(control.name for control in controls) or (f"surface:{patch}",)
        for name in names:
            patches.append(
                MeshPatch(
                    name,
                    mesh_entity_scope(mesh, 2, face_ids[source_patches == patch]),
                    source_adjacent_region_ids=None
                    if source_pair == (None, None)
                    else source_pair,
                )
            )
    zones = tuple(
        MeshZone(
            control.region_name,
            MeshZoneRole.REGION,
            mesh_entity_scope(
                mesh,
                2,
                face_ids[
                    np.isin(
                        source_patches,
                        domain.resolve_indices(
                            2, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
                        ),
                    )
                ],
            ),
            material_id=control.material_id,
            region_role=control.role,
        )
        for control in specification.region_controls
        if control.scope.entity_dimension == 2
    )

    return _SurfaceFaceSemantics(tuple(patches), zones, face_ids, source_patches)


def _surface_curve_semantics(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    mesh: cell_mesh_owner.CellMesh,
    construction: SurfaceConstruction,
    faces: _SurfaceFaceSemantics,
    /,
) -> _SurfaceCurveSemantics:
    patches = list(faces.patches)
    zones = faces.zones
    curve_controls = tuple(
        control
        for control in specification.patch_controls
        if control.scope.entity_dimension == 1
    )
    if curve_controls:
        match mesh.connectivity:
            case cell_complex_owner.PolygonalConnectivity() as connectivity:
                pass
            case _:
                raise TypeError(
                    "mesh.connectivity must be PolygonalConnectivity for curve patches; got another incidence type"
                )
        edge_set = mesh.entity_set(1)
        edge_rows = simplex_rows(
            np.asarray(connectivity.edges, dtype=np.int64), construction.curve_edges
        )
        edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[edge_rows]
        for control in curve_controls:
            curves = domain.resolve_indices(
                1, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
            )
            scope = MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                1,
                edge_set.entity_set_id,
                edge_ids[np.isin(construction.curve_edge_curves, curves)],
            )
            adjacent_patches = {
                patch
                for curve in curves.tolist()
                for patch, _ in domain.curve_uses(curve)
            }
            region_controls = tuple(
                region
                for region in specification.region_controls
                if region.scope.entity_dimension == 2
            )
            adjacent_zones = tuple(
                zone.zone_id
                for region, zone in zip(region_controls, zones, strict=True)
                if adjacent_patches.intersection(
                    domain.resolve_indices(
                        2, np.asarray(region.scope.global_entity_ids, dtype=np.int64)
                    ).tolist()
                )
            )
            patches.append(
                MeshPatch(
                    control.name,
                    scope,
                    connected=curves.size == 1,
                    adjacent_zone_ids=adjacent_zones,
                )
            )

    return _SurfaceCurveSemantics(tuple(patches))


def _surface_region_semantics(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    mesh: cell_mesh_owner.CellMesh,
    labels: tuple[MeshLabel, ...],
    faces: _SurfaceFaceSemantics,
    curves: _SurfaceCurveSemantics,
    /,
) -> NativePublicationOrganization:
    face_ids, source_patches = faces.face_ids, faces.source_patches
    patches, zones = curves.patches, faces.zones
    boundary_labels = list(labels)
    evidence: list[RegionBoundaryEvidence] = []
    for region, source_region in enumerate(domain.regions):
        identifier = domain.entity_id(3, region)
        sides = tuple(
            (
                patch.patch_id,
                1 if patch.source_adjacent_region_ids[0] == identifier else -1,
            )
            for patch in patches
            if patch.source_adjacent_region_ids is not None
            and identifier in patch.source_adjacent_region_ids
        )
        if not sides:
            continue
        controls = tuple(
            control
            for control in specification.region_controls
            if (
                control.scope.entity_dimension == 3
                and region
                in domain.resolve_indices(
                    3, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
                )
            )
        )
        control = controls[0] if controls else None
        boundary = np.asarray(
            [patch for patch, _ in source_region.boundary], dtype=np.int64
        )
        name = (
            f"region:{identifier}"
            if control is None
            else control.region_name
            if control.scope.global_entity_ids.shape[0] == 1
            else f"{control.region_name}:{identifier}"
        )
        label = MeshLabel(
            name, mesh_entity_scope(mesh, 2, face_ids[np.isin(source_patches, boundary)])
        )
        boundary_labels.append(label)
        scope = MeshingScope(
            domain.source_id,
            domain.source_revision,
            MeshingEntityKind.GEOMETRY,
            3,
            domain.entity_set_id(3),
            np.asarray((domain.scope_indices(3)[region],), dtype=np.int64),
        )
        evidence.append(
            RegionBoundaryEvidence(
                mesh,
                scope,
                identifier,
                domain.domain_id if control is None else control.control_id,
                None if control is None else control.material_id,
                None if control is None else control.role,
                label,
                sides,
            )
        )

    return NativePublicationOrganization(
        patches=tuple(patches),
        labels=tuple(boundary_labels),
        zones=zones,
        region_boundary_evidence=tuple(evidence),
    )


def _surface_semantics(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    mesh: cell_mesh_owner.CellMesh,
    construction: SurfaceConstruction,
    labels: tuple[MeshLabel, ...],
    /,
) -> NativePublicationOrganization:
    """Lower source incidence into overlapping patches and region boundaries."""
    faces = _surface_face_semantics(domain, specification, mesh, construction)
    curves = _surface_curve_semantics(domain, specification, mesh, construction, faces)
    return _surface_region_semantics(domain, specification, mesh, labels, faces, curves)


def _quad_decomposition_specification(
    specification: SurfaceMeshingSpec, /
) -> SurfaceMeshingSpec:
    """Internal chart topology request; final source-map policies remain unchanged."""
    return SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        specification.scope,
        size_controls=specification.size_controls,
        background_metric=specification.background_metric,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        periodic_constraints=specification.periodic_constraints,
        layer_controls=specification.layer_controls,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=specification.limits,
        deterministic=specification.deterministic,
    )


def _original_triangle_maps(
    construction: SurfaceConstruction,
    atlas: SurfaceSourceRootAtlas,
    /,
    *,
    maximum_support_queries: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[cell_mesh_owner.CellMesh, cell_geometry_owner.CellGeometrySpec]:
    """Pass native topology to the geometry-owned source restriction."""

    target = cell_mesh_owner.CellMesh.from_triangles(
        construction.vertices,
        construction.triangles,
        numeric_version=atlas.domain.source_revision,
    )
    return restrict_surface_source_atlas_geometry(
        atlas,
        target,
        construction.triangle_patches,
        construction.triangle_parameters,
        maximum_support_queries=maximum_support_queries,
        budget=budget,
    )


def _quad_associations(
    extraction: DualExtraction,
    support: PreparedSurfaceSourceSupport,
    associations: tuple[GeometryAssociation, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[GeometryAssociation, ...]:
    """Keep topology strata and rebuild source UV/curve parameters."""

    geometry = extraction.geometry
    if geometry is None:
        raise ValueError(
            "extraction.geometry must contain source coordinate maps; got None"
        )
    proof = support.prove_native_restrictions(extraction.mesh, geometry, budget=budget)
    uv: dict[tuple[int, int], tuple[Fraction, Fraction]] = {}
    for patch, ids, charts in zip(
        proof.patches.tolist(), proof.vertex_ids, proof.source_corners, strict=True
    ):
        for identifier, point in zip(ids, charts, strict=True):
            key = (patch, identifier)
            if key in uv and uv[key] != point:
                raise ValueError(
                    "source_charts.vertex_parameters must agree across chart uses; got conflicting UV coordinates"
                )
            uv[key] = point
    _, _, _, inherited = remap_dual_metadata(
        extraction, associations=associations, surface_source_cover=True
    )
    outputs = []
    for association in inherited:
        dimension = target_dimension(extraction.mesh, association)
        rows = entity_rows(
            extraction.mesh, dimension, np.asarray(association.target_global_ids)
        )
        vertices = entities(extraction.mesh, dimension)[rows]
        parameters = np.zeros((rows.size, 2), dtype=np.float64)
        for row, (source_dimension, index, path, values) in enumerate(
            zip(
                np.asarray(association.source_dimensions).tolist(),
                np.asarray(association.source_indices).tolist(),
                association.source_entities.occurrence_paths(),
                vertices.tolist(),
                strict=True,
            )
        ):
            source_row = support.domain_row(support.domain, source_dimension, index, path)
            if source_dimension == 0:
                continue
            if source_dimension == 2:
                patch = source_row
            else:
                patch, loop, position = map(int, support.domain.curve_owners[source_row])
            points = [
                uv[(patch, int(np.asarray(extraction.mesh.vertex_global_ids)[vertex]))]
                for vertex in values
                if vertex >= 0
            ]
            middle = tuple(
                sum((point[axis] for point in points), Fraction(0)) / len(points)
                for axis in range(2)
            )
            if source_dimension == 2:
                parameters[row] = tuple(float(value) for value in middle)
            else:
                use = support.domain.patches[patch].loops[loop][position]
                if not isinstance(use, PatchCurveUse) or not isinstance(
                    use.pcurve, LineCurve
                ):
                    raise ValueError(
                        "source_charts.curve_parameters must bind an affine UV coedge; got another trim representation"
                    )
                origin = tuple(
                    Fraction(float(value)) for value in np.asarray(use.pcurve.origin)
                )
                direction = tuple(
                    Fraction(float(value)) for value in np.asarray(use.pcurve.direction)
                )
                parameter = sum(
                    (
                        (value - first) * axis
                        for value, first, axis in zip(
                            middle, origin, direction, strict=True
                        )
                    ),
                    Fraction(0),
                ) / sum((value * value for value in direction), Fraction(0))
                parameters[row, 0] = float(parameter)
        rounding = float(np.max(proof.coordinate_corner_errors, initial=0.0))
        outputs.append(
            GeometryAssociation(
                association.association_kind,
                association.source_id,
                association.source_revision,
                association.target_entity_set_id,
                association.target_global_ids,
                association.source_entities.entity_ids(),
                np.full((rows.size,), rounding, dtype=np.float64),
                parameters=parameters,
                resolved=association.resolved,
                ambiguous=association.ambiguous,
                exact=rounding == 0.0,
                orientations=association.orientations,
                source_dimensions=association.source_dimensions,
                source_indices=association.source_indices,
                source_entity_roles=association.source_entities.roles(),
                source_occurrence_paths=association.source_entities.occurrence_paths(),
            )
        )
    return tuple(outputs)


@checked
def quad_edge_evidence(
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    specification: SurfaceMeshingSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Certified physical arc lengths of coordinate-map edges."""

    edges = entities(mesh, 1)
    lookup = {tuple(sorted(pair)): row for row, pair in enumerate(edges.tolist())}
    value = np.full(edges.shape[0], np.nan, dtype=np.float64)
    lower, upper = value.copy(), value.copy()
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    coefficients = geometry.source_coordinates(budget=budget)
    topology = reference_cell_topology("quadrilateral")
    points = tuple(tuple(Fraction(float(x)) for x in row) for row in topology.vertices)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for vertices, dofs in zip(
            np.asarray(block.vertices).tolist(), np.asarray(route).tolist(), strict=True
        ):
            local = tuple(coefficients[index] for index in dofs)
            for first, last in topology.entities[1]:
                row = lookup[tuple(sorted((vertices[first], vertices[last])))]
                if np.isfinite(value[row]):
                    continue
                enclosure = prepare_mapped_edge_arc_length(
                    element, local, points[first], points[last], budget=budget
                ).integrate(
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-12,
                    maximum_work=specification.limits.maximum_work_units,
                    maximum_subcells=min(specification.limits.maximum_cells, 10000),
                    budget=budget,
                )
                value[row], lower[row], upper[row] = (
                    enclosure.value,
                    enclosure.lower,
                    enclosure.upper,
                )
    return edges, value, lower, upper


@final
class _QuadComplianceEvidence(StrictModule):
    edges: np.ndarray
    lengths: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    proof: SurfaceRestrictionRows
    edge_patches: dict[tuple[int, int], set[int]]
    curve_rows: dict[int, int]


@final
class _QuadControlScope(StrictModule):
    entities: np.ndarray
    dimension: int
    rows: np.ndarray


@final
class _SurfaceControlCompliance(StrictModule):
    requested: tuple[tuple[str, float], ...]
    achieved: tuple[tuple[str, float], ...]
    issues: tuple[str, ...]


def _prepare_quad_compliance(
    specification: SurfaceMeshingSpec,
    extraction: DualExtraction,
    support: PreparedSurfaceSourceSupport,
    associations: tuple[GeometryAssociation, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> _QuadComplianceEvidence:
    geometry = extraction.geometry
    if geometry is None:
        raise ValueError(
            "extraction.geometry must contain mapped coordinate geometry; got None"
        )
    mesh, domain = extraction.mesh, support.domain
    edges, lengths, lower, upper = quad_edge_evidence(
        mesh, geometry, specification, budget=budget
    )
    proof = support.prove_native_restrictions(mesh, geometry, budget=budget)
    edge_patches: dict[tuple[int, int], set[int]] = {
        tuple(sorted(edge)): set() for edge in edges.tolist()
    }
    for vertices, patch in zip(
        (row for block in mesh.blocks for row in np.asarray(block.vertices).tolist()),
        proof.patches.tolist(),
        strict=True,
    ):
        for first, last in ((0, 1), (1, 2), (2, 3), (3, 0)):
            edge_patches[tuple(sorted((vertices[first], vertices[last])))].add(patch)
    curve_rows: dict[int, int] = {}
    for association in associations:
        if target_dimension(mesh, association) != 1:
            continue
        rows = entity_rows(mesh, 1, np.asarray(association.target_global_ids))
        for row, dimension, index, path in zip(
            rows.tolist(),
            np.asarray(association.source_dimensions).tolist(),
            np.asarray(association.source_indices).tolist(),
            association.source_entities.occurrence_paths(),
            strict=True,
        ):
            if dimension == 1:
                curve_rows[row] = support.domain_row(domain, dimension, index, path)

    return _QuadComplianceEvidence(
        edges, lengths, lower, upper, proof, edge_patches, curve_rows
    )


def _quad_control_scope(
    control: UniformSizeControl | CurvatureSizeControl | ProximitySizeControl,
    domain: MeshingDomain,
    evidence: _QuadComplianceEvidence,
    /,
) -> _QuadControlScope:
    edges, edge_patches, curve_rows = (
        evidence.edges,
        evidence.edge_patches,
        evidence.curve_rows,
    )
    source = _resolve_surface_control(control, domain)
    source_entities, entity_dimension = source.entities, source.dimension
    rows = np.asarray(
        [
            row
            for row, edge in enumerate(edges.tolist())
            if (
                curve_rows.get(row) in source_entities
                if entity_dimension == 1
                else bool(
                    edge_patches[tuple(sorted(edge))] & set(source_entities.tolist())
                )
            )
        ],
        dtype=np.int64,
    )

    return _QuadControlScope(source_entities, entity_dimension, rows)


def _quad_source_control_compliance(
    control: CurvatureSizeControl | ProximitySizeControl,
    specification: SurfaceMeshingSpec,
    construction: SurfaceConstruction,
    domain: MeshingDomain,
    compiled: CompiledSurfaceDomain,
    evidence: _QuadComplianceEvidence,
    selected: _QuadControlScope,
    budget: CoordinateEnclosureBudget,
    /,
) -> _SurfaceControlCompliance:
    source_entities, entity_dimension = selected.entities, selected.dimension
    edges, upper, proof, edge_patches = (
        evidence.edges,
        evidence.upper,
        evidence.proof,
        evidence.edge_patches,
    )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    if entity_dimension == 1:
        source_entities = np.asarray(
            [
                patch
                for patch in range(len(domain.patches))
                if set(domain.patch_curves(patch).tolist())
                & set(source_entities.tolist())
            ],
            dtype=np.int64,
        )
    maximum = 0.0
    for patch in source_entities.tolist():
        selected = proof.patches == patch
        uv = np.mean(np.asarray(proof.source_corners, dtype=np.float64)[selected], axis=1)
        sizes = compiled.source_sizing.evaluate(
            domain,
            patch,
            uv,
            domain.evaluate(np.full(uv.shape[0], patch), uv, budget=budget),
            budget=budget,
        )
        patch_edges = [
            row
            for row, edge in enumerate(edges.tolist())
            if patch in edge_patches[tuple(sorted(edge))]
        ]
        maximum = max(
            maximum,
            float(np.max(upper[patch_edges], initial=0.0)) / float(np.min(sizes)),
        )
    key = f"source_size:{control.control_id}"
    requested.append((f"{key}:maximum_local_edge_ratio", 1.0))
    achieved.append((f"{key}:maximum_local_edge_ratio", maximum))
    if (
        control.strength is SizeControlStrength.HARD
        and maximum > 1.0 + specification.size_compliance.relative_tolerance
    ):
        issues.append(key)
    match control:
        case CurvatureSizeControl():
            normal = float(
                np.max(
                    construction.triangle_normal_bounds[
                        np.isin(construction.triangle_patches, source_entities)
                    ],
                    initial=0.0,
                )
            )
            requested.append((f"{key}:normal_angle", control.normal_angle))
            achieved.append((f"{key}:normal_angle", normal))
            if (
                control.strength is SizeControlStrength.HARD
                and normal > control.normal_angle
            ):
                issues.append(f"{key}:normal_angle")
        case ProximitySizeControl():
            pass
        case _:
            raise TypeError(
                f"specification.size_controls must contain curvature or proximity controls; got {type(control).__name__}"
            )

    return _SurfaceControlCompliance(tuple(requested), tuple(achieved), tuple(issues))


def _quad_compliance(
    specification: SurfaceMeshingSpec,
    construction: SurfaceConstruction,
    extraction: DualExtraction,
    support: PreparedSurfaceSourceSupport,
    compiled: CompiledSurfaceDomain,
    associations: tuple[GeometryAssociation, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> MeshingComplianceReport:
    evidence = _prepare_quad_compliance(
        specification, extraction, support, associations, budget
    )
    mesh, domain = extraction.mesh, support.domain
    edges, lengths, lower, upper = (
        evidence.edges,
        evidence.lengths,
        evidence.lower,
        evidence.upper,
    )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    for control in specification.size_controls:
        selected = _quad_control_scope(control, domain, evidence)
        rows = selected.rows
        match control:
            case UniformSizeControl():
                growth = edge_growth_evidence(
                    np.concatenate((lower[rows], upper[rows])),
                    np.concatenate((edges[rows], edges[rows])),
                    mesh.coordinates.shape[0],
                )
                req, act, failed = uniform_size_compliance(
                    control, specification.size_compliance, lengths[rows], growth
                )
                for endpoint in (lower[rows], upper[rows]):
                    _, _, endpoint_failed = uniform_size_compliance(
                        control, specification.size_compliance, endpoint, growth
                    )
                    failed.extend(endpoint_failed)
                requested.extend(req)
                achieved.extend(act)
                issues.extend(failed)

            case CurvatureSizeControl() | ProximitySizeControl():
                compliance = _quad_source_control_compliance(
                    control,
                    specification,
                    construction,
                    domain,
                    compiled,
                    evidence,
                    selected,
                    budget,
                )
                requested.extend(compliance.requested)
                achieved.extend(compliance.achieved)
                issues.extend(compliance.issues)
            case _:
                raise TypeError(
                    f"specification.size_controls must contain supported controls; got {type(control).__name__}"
                )
    angle = float(
        np.min(np.asarray(evaluate_cell_quality(mesh, budget=budget).minimum_angle))
    )
    achieved.extend(
        (
            ("minimum_angle", angle),
            ("quadrilateral_count", mesh.entity_set(2).count),
            ("maximum_edge_length_enclosure_width", float(np.max(upper - lower))),
        )
    )
    if specification.quality_target is not None:
        target = specification.quality_target
        requested.append(("minimum_angle", target.minimum_angle))
        if target.hard and angle < target.minimum_angle:
            issues.append(f"minimum_angle:{target.target_id}")
    for feature in specification.protected_features:
        key = f"protected:{feature.feature_id}:maximum_deviation"
        requested.append((key, feature.maximum_deviation))
        achieved.append((key, 0.0))
    return MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(requested),
        achieved=tuple(achieved),
        issues=tuple(sorted(set(issues))),
    )


def _execute_exact_source_quads(
    source: sources_owner.NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedParametricSurface,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    budget = coordinate_budget(specification.limits)
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        started = scope.operation_started
        limits = specification.limits
        with coordinate_phase(scope, specification.limits, budget):
            atlas = prepare_surface_source_root_atlas(
                source.domain,
                tuple(prepared.compiled.patches.tolist()),
                coordinate_contract,
                maximum_support_queries=limits.maximum_geometry_queries,
                budget=budget,
            )
        original = MeshingDomainBoundarySource(
            source.domain, tuple(prepared.compiled.patches.tolist())
        )
        support = PreparedSurfaceSourceSupport(
            source.domain,
            atlas,
            atlas.root_parameters,
            original,
            maximum_support_queries=limits.maximum_geometry_queries,
        )
        with coordinate_phase(scope, specification.limits, budget):
            decomp = compile_surface_domain(
                source.domain,
                _quad_decomposition_specification(specification),
                budget=budget,
            )
        construction = generate_surface(
            decomp,
            prepared.schedule,
            limits,
            math.radians(prepared.schedule.quality_angle_degrees),
            required_minimum_angle=0.0,
            size_compliance=specification.size_compliance,
            record_phase=record_phase,
            scope=scope,
            budget=budget,
        )
        with coordinate_phase(scope, specification.limits, budget):
            scaffold, source_geometry = _original_triangle_maps(
                construction,
                atlas,
                maximum_support_queries=limits.maximum_geometry_queries,
                budget=budget,
            )
        with coordinate_phase(scope, specification.limits, budget):
            source_associations, _, labels = associations(
                source, scaffold, construction, budget=budget
            )
        semantics = _surface_semantics(
            source.domain, specification, scaffold, construction, labels
        )
        patches, labels, zones, region_boundaries = (
            semantics.patches,
            semantics.labels,
            semantics.zones,
            semantics.region_boundary_evidence,
        )
        edges = entities(scaffold, 1)
        lookup = {tuple(sorted(pair)): row for row, pair in enumerate(edges.tolist())}
        features = np.asarray(
            [lookup[tuple(sorted(pair))] for pair in construction.curve_edges.tolist()],
            dtype=np.int64,
        )
        field = prepare_surface_cross_field(
            scaffold,
            source_geometry=source_geometry,
            feature_edges=features,
            record_phase=record_phase,
            scope=scope,
            budget=budget,
        )
        extraction = extract_surface_quads(
            scaffold,
            limits,
            source_geometry=source_geometry,
            cross_field=field,
            scope=scope,
            budget=budget,
        )
        geometry = extraction.geometry
        if geometry is None:
            raise ValueError(
                "extraction.geometry must contain source coordinate maps; got None"
            )
        source_labels = labels
        zones, patches, labels, _ = remap_dual_metadata(
            extraction, zones=zones, patches=patches, labels=labels
        )
        label_map = {
            old.label_id: new for old, new in zip(source_labels, labels, strict=True)
        }
        region_boundaries = tuple(
            RegionBoundaryEvidence(
                extraction.mesh,
                evidence.source_scope,
                evidence.source_region_id,
                evidence.control_id,
                evidence.material_id,
                evidence.role,
                label_map[evidence.boundary_label.label_id],
                evidence.patch_sides,
            )
            for evidence in region_boundaries
        )
        with coordinate_phase(scope, specification.limits, budget):
            local_associations = _quad_associations(
                extraction, support, source_associations, budget=budget
            )
        with coordinate_phase(scope, specification.limits, budget):
            compliance = _quad_compliance(
                specification,
                construction,
                extraction,
                support,
                prepared.compiled,
                local_associations,
                budget=budget,
            )
        boundary = SurfaceNativeRestrictionBoundarySource(
            original, support, extraction.mesh, geometry
        )
        finite = prepared.compiled.patch_deviations[
            np.isfinite(prepared.compiled.patch_deviations)
        ]
        tolerance = (
            float(np.min(finite))
            if finite.size
            else float(np.min(prepared.compiled.patch_sizes))
        )
        request = NativeCertificationRequest(
            MeshCertificationSchedule("surface"),
            source.source_id,
            source.source_revision,
            limits,
            fidelity_source=boundary,
            fidelity_tolerance=tolerance,
        )
        check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
        result = publish_native_result(
            extraction.mesh,
            coordinate_contract,
            compliance,
            (
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.binding_id, atlas.atlas_id),
                    output_ids=(extraction.mesh.mesh_id,),
                    created_count=extraction.mesh.entity_set(2).count,
                ),
            ),
            provider,
            {
                "kind": "native-parametric-source-quads",
                "source": source.binding_id,
                "plan": plan_id,
                "specification": specification.specification_id,
            },
            request,
            policy=publication_policy(
                "source_quads",
                audit_policy=CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=CellMeshAuditDisposition.REJECT
                    if closed(prepared.compiled)
                    else CellMeshAuditDisposition.SKIP,
                ),
            ),
            organization=NativePublicationOrganization(
                patches=patches,
                zones=zones,
                labels=labels,
                associations=local_associations,
                region_boundary_evidence=region_boundaries,
            ),
            execution=NativePublicationExecution(scope, workspace, record_phase, budget),
            geometry=geometry,
        )
        return result


@final
class _SurfaceTopology(StrictModule):
    mesh: cell_mesh_owner.CellMesh
    geometry: cell_geometry_owner.CellGeometrySpec
    atlas: SurfaceSourceRootAtlas | None


@final
class _SurfaceSourcePublication(StrictModule):
    charts: SurfaceSourceCharts
    certification: NativeCertificationRequest


@final
class _SurfaceStages(StrictModule):
    stages: tuple[MeshingStageReport, ...]


def _prepare_surface_topology(
    source: sources_owner.NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedParametricSurface,
    construction: SurfaceConstruction,
    coordinate_contract: SpatialCoordinateContract,
    execution: NativePublicationExecution,
    budget: CoordinateEnclosureBudget,
    /,
) -> _SurfaceTopology:
    compiled = prepared.compiled
    limits = specification.limits
    scope = execution.scope
    record_phase = execution.record_phase
    triangles = construction.triangles.astype(np.int32)
    simplex_entity_limits(
        construction.vertices,
        triangles,
        limits,
        MeshingStageKind.SURFACE_MESHING,
        cell_kind="triangle",
    )
    with measure_phase(record_phase, "topology_construction"):
        exact_source = all(
            isinstance(source.domain.patches[int(patch)].surface, BSplineSurfacePatch)
            for patch in compiled.patches
        )
        atlas: SurfaceSourceRootAtlas | None = None
        if exact_source:
            with coordinate_phase(scope, specification.limits, budget):
                atlas = prepare_surface_source_root_atlas(
                    source.domain,
                    tuple(compiled.patches.tolist()),
                    coordinate_contract,
                    maximum_support_queries=limits.maximum_geometry_queries,
                    budget=budget,
                )
            with coordinate_phase(scope, specification.limits, budget):
                mesh, geometry = _original_triangle_maps(
                    construction,
                    atlas,
                    maximum_support_queries=limits.maximum_geometry_queries,
                    budget=budget,
                )
        else:
            with coordinate_phase(scope, specification.limits, budget):
                mesh = canonicalize_cell_mesh(
                    cell_mesh_owner.CellMesh.from_triangles(
                        construction.vertices,
                        triangles,
                        numeric_version=source.source_revision,
                    ),
                    budget=budget,
                )
            geometry = cell_geometry_owner.CellGeometrySpec.affine(mesh)
    return _SurfaceTopology(mesh, geometry, atlas)


def _prepare_surface_publication(
    source: sources_owner.NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedParametricSurface,
    construction: SurfaceConstruction,
    topology: _SurfaceTopology,
    coordinate_contract: SpatialCoordinateContract,
    local_associations: tuple[GeometryAssociation, ...],
    execution: NativePublicationExecution,
    /,
) -> _SurfaceSourcePublication:
    compiled = prepared.compiled
    limits = specification.limits
    mesh, geometry, atlas = topology.mesh, topology.geometry, topology.atlas
    scope, workspace = execution.scope, execution.workspace
    faces = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    )
    rows = simplex_rows(construction.triangles, faces)
    order = np.argsort(
        entity_rows(
            mesh,
            2,
            np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks)),
        ),
        kind="stable",
    )
    parameters = scope.allocate_host_array((faces.shape[0], 3, 2), np.float64)
    root_patches = scope.allocate_host_array((faces.shape[0],), np.int64)
    for output, index in enumerate(order.tolist()):
        row = rows[index]
        root_patches[output] = construction.triangle_patches[row]
        for slot, vertex in enumerate(faces[index].tolist()):
            local = int(np.flatnonzero(construction.triangles[row] == vertex)[0])
            parameters[output, slot] = construction.triangle_parameters[row, local]
    if atlas is not None:
        root_boundary = MeshingDomainBoundarySource(
            source.domain,
            tuple(compiled.patches.tolist()),
            chart_triangulations=construction.chart_triangulations,
        )
        exact_support = PreparedSurfaceSourceSupport(
            source.domain,
            atlas,
            atlas.root_parameters,
            root_boundary,
            maximum_support_queries=limits.maximum_geometry_queries,
        )
        fidelity_source = SurfaceNativeRestrictionBoundarySource(
            root_boundary, exact_support, mesh, geometry
        )
        finite = compiled.patch_deviations[np.isfinite(compiled.patch_deviations)]
        tolerance = (
            float(np.min(finite)) if finite.size else float(np.min(compiled.patch_sizes))
        )
        certification_request = NativeCertificationRequest(
            MeshCertificationSchedule("surface"),
            source.source_id,
            source.source_revision,
            limits,
            fidelity_source=fidelity_source,
            fidelity_tolerance=tolerance,
        )
    else:
        root_boundary = MeshingDomainBoundarySource(
            source.domain,
            tuple(compiled.patches.tolist()),
            chart_triangulations=construction.chart_triangulations,
        )
        certification_request = _certification_request(
            source, specification, compiled, construction
        )
    source_charts = SurfaceSourceCharts(
        source.domain,
        mesh,
        geometry,
        coordinate_contract,
        parameters,
        root_patches,
        local_associations,
        root_boundary,
        maximum_support_queries=limits.maximum_geometry_queries,
    )
    if workspace is not None:
        workspace.retain_owner(source_charts)
    return _SurfaceSourcePublication(source_charts, certification_request)


def _surface_construction_stages(
    source: sources_owner.NativeSurfaceSource,
    prepared: PreparedParametricSurface,
    construction: SurfaceConstruction,
    mesh: cell_mesh_owner.CellMesh,
    /,
) -> _SurfaceStages:
    triangles = construction.triangles
    construction_stages = (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(source.binding_id,),
            output_ids=(prepared.prepared_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING,
            MeshingStageStatus.WARNING
            if construction.incomplete_refinement
            else MeshingStageStatus.PASSED,
            input_ids=(prepared.prepared_id,),
            output_ids=(mesh.mesh_id,),
            created_count=triangles.shape[0],
            diagnostics=(
                MeshingDiagnostic(
                    MeshingDiagnosticSeverity.WARNING,
                    "Physical bounds were met, but bounded refinement did not meet its soft shape aim.",
                    provider_code="surface_refinement_incomplete",
                    quantities=(
                        (
                            "refinement_rounds",
                            float(prepared.schedule.maximum_rounds),
                            float(construction.rounds),
                        ),
                    ),
                ),
            )
            if construction.incomplete_refinement
            else (),
        ),
    )
    return _SurfaceStages(construction_stages)


@final
class _SurfacePublished(StrictModule):
    result: CellMeshingResult


def _publish_parametric_surface(
    source: sources_owner.NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedParametricSurface,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    execution: NativePublicationExecution,
    construction: SurfaceConstruction,
    topology: _SurfaceTopology,
    compliance: MeshingComplianceReport,
    organization: NativePublicationOrganization,
    /,
) -> _SurfacePublished:
    scope, workspace, record_phase, budget = (
        execution.scope,
        execution.workspace,
        execution.record_phase,
        execution.budget,
    )
    started, limits, compiled = (
        scope.operation_started,
        specification.limits,
        prepared.compiled,
    )
    mesh, geometry = topology.mesh, topology.geometry
    patches, labels, zones = organization.patches, organization.labels, organization.zones
    region_boundaries, local_associations = (
        organization.region_boundary_evidence,
        organization.associations,
    )
    check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
    construction_stages = _surface_construction_stages(
        source, prepared, construction, mesh
    ).stages
    watertight = (
        CellMeshAuditDisposition.REJECT
        if closed(compiled)
        else CellMeshAuditDisposition.SKIP
    )

    publication = _prepare_surface_publication(
        source,
        specification,
        prepared,
        construction,
        topology,
        coordinate_contract,
        local_associations,
        execution,
    )
    result = publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        construction_stages,
        provider,
        {
            "kind": "native-parametric-surface-cell-mesh",
            "route": "parametric_surface",
            "source": source.binding_id,
            "domain": source.domain.domain_id,
            "plan": plan_id,
            "specification": specification.specification_id,
        },
        publication.certification,
        policy=publication_policy(
            "parametric_surface",
            audit_policy=CellMeshAuditPolicy(
                require_complete_association=True, watertight_boundary=watertight
            ),
        ),
        organization=NativePublicationOrganization(
            patches=patches,
            zones=zones,
            labels=labels,
            associations=local_associations,
            region_boundary_evidence=region_boundaries,
            surface_source=publication.charts,
        ),
        execution=NativePublicationExecution(scope, workspace, record_phase, budget),
        geometry=geometry,
    )
    return _SurfacePublished(result)


@checked
def execute_parametric_surface_route(
    source: sources_owner.NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedParametricSurface,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
) -> CellMeshingResult:
    """Generate, associate, check, audit and publish one parametric surface."""
    budget = coordinate_budget(specification.limits)
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        limits = specification.limits
        execution = NativePublicationExecution(scope, workspace, record_phase, budget)
        compiled = prepared.compiled
        if compiled.domain.domain_id != source.domain.domain_id:
            raise ValueError(
                "prepared.compiled.domain must match source.domain; got a different domain revision"
            )
        if compiled.specification_id != specification.specification_id:
            raise ValueError(
                "prepared.compiled.specification_id must match specification.specification_id; got another request"
            )
        if set(
            (
                *specification.target.cell_families.required,
                *specification.target.cell_families.preferred,
            )
        ) == {"quadrilateral"}:
            if prepared.generation is not None:
                raise ValueError(
                    "prepared.generation must be None for source quad extraction; got distributed generation"
                )
            return _execute_exact_source_quads(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                record_phase=record_phase,
                scope=scope,
                workspace=workspace,
            )
        if prepared.generation is not None:
            with measure_phase(record_phase, "construction"):
                generated = execute_distributed_surface_initial(
                    prepared.generation, minimum_angle=prepared.minimum_angle
                )
            quality = specification.quality_target
            required_angle = (
                0.0 if quality is None or not quality.hard else quality.minimum_angle
            )
            with measure_phase(record_phase, "certification"):
                result = publish_distributed_surface_initial(
                    prepared.generation,
                    generated,
                    source,
                    coordinate_contract,
                    provider,
                    minimum_angle=required_angle,
                    plan_id=plan_id,
                )
            return result
        quality = specification.quality_target
        required_angle = (
            0.0 if quality is None or not quality.hard else quality.minimum_angle
        )
        with measure_phase(record_phase, "construction"):
            construction = generate_surface(
                compiled,
                prepared.schedule,
                limits,
                prepared.minimum_angle,
                record_phase=record_phase,
                required_minimum_angle=required_angle,
                size_compliance=specification.size_compliance,
                scope=scope,
                budget=budget,
            )
        topology = _prepare_surface_topology(
            source,
            specification,
            prepared,
            construction,
            coordinate_contract,
            execution,
            budget,
        )
        mesh = topology.mesh
        with measure_phase(record_phase, "geometry_association"):
            with coordinate_phase(scope, specification.limits, budget):
                local_associations, patches, labels = associations(
                    source, mesh, construction, budget=budget
                )
            semantics = _surface_semantics(
                source.domain, specification, mesh, construction, labels
            )
            patches, labels, zones, region_boundaries = (
                semantics.patches,
                semantics.labels,
                semantics.zones,
                semantics.region_boundary_evidence,
            )
        with measure_phase(record_phase, "compliance"):
            with coordinate_phase(scope, specification.limits, budget):
                compliance = _compliance(
                    specification, construction, source.domain, compiled, budget=budget
                )
        organization = NativePublicationOrganization(
            patches=patches,
            labels=labels,
            zones=zones,
            region_boundary_evidence=region_boundaries,
            associations=local_associations,
        )
        return _publish_parametric_surface(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            execution,
            construction,
            topology,
            compliance,
            organization,
        ).result


__all__ = [
    "PreparedParametricSurface",
    "execute_parametric_surface_route",
    "parametric_surface_support_issues",
]
