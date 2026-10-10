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
from time import monotonic
from typing import final, TypedDict

import equinox as eqx
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellGeometrySpec, CellMesh, PolygonalConnectivity
from ...geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainBoundarySource,
    PatchCurveUse,
)
from ...geometry._surface_source_support import (
    prepare_surface_source_root_atlas,
    PreparedSurfaceSourceSupport,
    SurfaceNativeRestrictionBoundarySource,
    SurfaceSourceCharts,
    SurfaceSourceRootAtlas,
)
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._canonical import canonicalize_cell_mesh
from .._certification import MeshCertificationSchedule
from .._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingDerivativeMode,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
)
from .._controls import FeatureKind
from .._distributed_generation import PreparedDistributedSurfaceGeneration
from .._domain import compile_surface_domain, CompiledSurfaceDomain, surface_domain_issues
from .._measurements import NativeMeshingPhaseRecorder, phase_started, record_elapsed
from .._metric import metric_edge_lengths, metric_simplex_quality
from .._organization import (
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    RegionBoundaryEvidence,
)
from .._quad_generation import (
    DualExtraction,
    extract_surface_quads,
    prepare_surface_cross_field,
    remap_dual_metadata,
)
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind, MeshingScope
from .._sizing import (
    CurvatureSizeControl,
    ProximitySizeControl,
    SizeControlStrength,
    UniformSizeControl,
)
from .._surface_generation import generate_surface, SurfaceConstruction
from .._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from ._native_options import NativeSurfaceSchedule
from ._native_publication import (
    check_deadline,
    edge_size_evidence,
    NativeCertificationRequest,
    publish_native_result,
    simplex_entity_limits,
    uniform_size_compliance,
    unique_edges,
)
from ._native_sources import NativeSurfaceSource


def parametric_surface_support_issues(
    source: NativeSurfaceSource, specification: SurfaceMeshingSpec, /
) -> list[str]:
    """Physical requests of one surface specification this route cannot enforce."""

    unsupported: list[str] = []
    target = specification.target
    families = target.cell_families
    kinds = set((*families.required, *families.preferred))
    if kinds == {"quadrilateral"}:
        from ...geometry.brep._patches import BSplineSurfacePatch, LineCurve

        selected = source.domain.resolve_indices(
            2, np.asarray(specification.scope.entity_ids)
        )
        for patch in selected.tolist():
            surface = source.domain.patches[patch]
            if not isinstance(surface.surface, BSplineSurfacePatch):
                unsupported.append(
                    "original rational spline patch maps for exact source quads"
                )
            if surface.reversed:
                unsupported.append(
                    "positive source-UV orientation for exact source quad extraction"
                )
            if any(
                not isinstance(use, PatchCurveUse)
                or not isinstance(use.pcurve, LineCurve)
                for loop in surface.loops
                for use in loop
            ):
                unsupported.append(
                    "original affine UV trim coedges for exact source quad coverage"
                )
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

    def __init__(
        self,
        source: NativeSurfaceSource,
        specification: SurfaceMeshingSpec,
        schedule: NativeSurfaceSchedule,
        /,
        *,
        initial_partition: PreparedDistributedSurfaceGeneration | None = None,
    ) -> None:
        if not isinstance(schedule, NativeSurfaceSchedule):
            raise TypeError("schedule must be NativeSurfaceSchedule.")
        compiled = compile_surface_domain(source.domain, specification)
        aim = math.radians(schedule.quality_angle_degrees)
        quality = specification.quality_target
        if quality is not None:
            aim = max(aim, quality.minimum_angle)
        generation = initial_partition
        if generation is not None:
            if not isinstance(generation, PreparedDistributedSurfaceGeneration):
                raise TypeError(
                    "initial_partition must be actual prepared distributed surface generation."
                )
            generation.__post_init__()
            if (
                generation.compiled.compiled_id != compiled.compiled_id
                or generation.specification.specification_id
                != specification.specification_id
                or generation.schedule.schedule_id != schedule.schedule_id
            ):
                raise ValueError(
                    "Initial partition does not bind this exact native source, specification, and schedule."
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


def _closed(compiled: CompiledSurfaceDomain, /) -> bool:
    """Whether every curve of the selected patches is used exactly twice."""

    domain = compiled.domain
    counts = {curve: 0 for curve in compiled.curves.tolist()}
    for patch in compiled.patches.tolist():
        for loop in domain.patches[patch].loops:
            for use in loop:
                if isinstance(use, PatchCurveUse):
                    counts[use.curve] += 1
    return all(count == 2 for count in counts.values())


def _patch_edges(construction: SurfaceConstruction, patches: np.ndarray, /) -> np.ndarray:
    return unique_edges(
        construction.triangles[np.isin(construction.triangle_patches, patches)],
        "triangle",
    )


def _compliance(
    specification: SurfaceMeshingSpec,
    construction: SurfaceConstruction,
    domain: MeshingDomain,
    compiled: CompiledSurfaceDomain,
    /,
) -> MeshingComplianceReport:
    vertices = construction.vertices
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    for control in specification.size_controls:
        if isinstance(control, ProximitySizeControl):
            entities = np.unique(
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
        else:
            entities = domain.resolve_indices(
                control.scope.entity_dimension,
                np.asarray(control.scope.global_entity_ids, dtype=np.int64),
            )
            if control.scope.entity_dimension == 3:
                entities = np.unique(
                    np.asarray(
                        [
                            patch
                            for region in entities.tolist()
                            for patch, _ in domain.regions[region].boundary
                        ],
                        dtype=np.int64,
                    )
                )
        if isinstance(control, UniformSizeControl):
            edges = (
                construction.curve_edges[
                    np.isin(construction.curve_edge_curves, entities)
                ]
                if control.scope.entity_dimension == 1
                else _patch_edges(construction, entities)
            )
            lengths, growth = edge_size_evidence(vertices, edges)
            control_requested, control_achieved, control_issues = uniform_size_compliance(
                control, specification.size_compliance, lengths, growth
            )
            requested.extend(control_requested)
            achieved.extend(control_achieved)
            issues.extend(control_issues)
            continue
        key = f"source_size:{control.control_id}"
        ratios = []
        for patch in entities.tolist():
            selected = construction.triangle_patches == patch
            charts = np.mean(construction.triangle_parameters[selected], axis=1)
            points = domain.evaluate(np.full((charts.shape[0],), patch), charts)
            sizes = compiled.source_sizing.evaluate(domain, patch, charts, points)
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
        if isinstance(control, CurvatureSizeControl):
            values = construction.triangle_normal_bounds[
                np.isin(construction.triangle_patches, entities)
            ]
            normal_bound = float(np.max(values, initial=0.0))
            requested.append((f"{key}:normal_angle", control.normal_angle))
            achieved.append((f"{key}:normal_angle", normal_bound))
            if (
                control.strength is SizeControlStrength.HARD
                and normal_bound > control.normal_angle
            ):
                issues.append(f"{key}:normal_angle")
    quality = specification.quality_target
    achieved.append(("minimum_angle", construction.minimum_angle))
    if quality is not None:
        requested.append(("minimum_angle", quality.minimum_angle))
        if quality.hard and construction.minimum_angle < quality.minimum_angle:
            issues.append(f"minimum_angle:{quality.target_id}")
    for feature in specification.protected_features:
        entities = domain.resolve_indices(
            feature.scope.entity_dimension,
            np.asarray(feature.scope.global_entity_ids, dtype=np.int64),
        )
        match feature.feature_kind:
            case FeatureKind.SURFACE:
                values = construction.triangle_deviation_bounds[
                    np.isin(construction.triangle_patches, entities)
                ]
            case FeatureKind.CURVE:
                values = construction.curve_edge_deviation_bounds[
                    np.isin(construction.curve_edge_curves, entities)
                ]
            case FeatureKind.CORNER:
                # Corners are exact vertices of every construction.
                values = np.zeros((1,), dtype=np.float64)
            case _:
                raise TypeError(
                    "Admitted protected features are corners/curves/surfaces."
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
            vertices, specification.limits.maximum_work_units - construction.work_units
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


def _rows(canonical: np.ndarray, pairs: np.ndarray, /) -> np.ndarray:
    """Rows of ``canonical`` simplices (any vertex order) matching ``pairs``."""

    keys = {key: row for row, key in enumerate(map(tuple, np.sort(canonical, axis=1)))}
    return np.asarray(
        [keys[key] for key in map(tuple, np.sort(pairs, axis=1).tolist())],
        dtype=np.int64,
    )


class _SourceMetadata(TypedDict):
    source_dimensions: np.ndarray
    source_indices: np.ndarray
    source_occurrence_paths: tuple[tuple[str, ...], ...]


def _source_metadata(
    domain: MeshingDomain, dimensions: np.ndarray, indices: np.ndarray, /
) -> _SourceMetadata:
    """Retain authored strata for both analytic and native B-Rep publications."""
    return {
        "source_dimensions": dimensions,
        "source_indices": np.asarray(
            [
                domain.source_indices[int(dimension)][int(index)]
                for dimension, index in zip(dimensions, indices, strict=True)
            ],
            dtype=np.int64,
        ),
        "source_occurrence_paths": tuple(
            domain.source_occurrences[int(dimension)][int(index)]
            for dimension, index in zip(dimensions, indices, strict=True)
        ),
    }


def _associations(
    source: NativeSurfaceSource,
    mesh: CellMesh,
    construction: SurfaceConstruction,
    /,
) -> tuple[tuple[GeometryAssociation, ...], tuple[MeshPatch, ...], tuple[MeshLabel, ...]]:
    domain = source.domain
    revision = source.source_revision
    kind = (
        GeometryAssociationKind.BREP
        if domain.source_kinds == ("vertex", "edge", "face")
        else GeometryAssociationKind.SURFACE
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    face_set = mesh.entity_set(2)
    faces = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    )
    rows = _rows(construction.triangles, faces)
    patches = construction.triangle_patches[rows]
    charts = construction.triangle_charts[rows]
    normals, _ = domain.oriented_normals(patches, charts)
    flat = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    face_association = GeometryAssociation(
        kind,
        source.source_id,
        revision,
        face_set.entity_set_id,
        face_ids,
        tuple(domain.entity_id(2, patch) for patch in patches.tolist()),
        construction.triangle_deviations[rows],
        exact=False,
        parameters=charts,
        orientations=np.sign(np.sum(flat * normals, axis=1)).astype(np.int8),
        **_source_metadata(domain, np.full(patches.shape, 2, dtype=np.int8), patches),
    )
    edge_set = mesh.entity_set(1)
    # ty: ignore[unresolved-attribute]
    canonical_edges = np.asarray(mesh.connectivity.edges, dtype=np.int64)
    edge_rows = _rows(canonical_edges, construction.curve_edges)
    edge_ids = np.asarray(edge_set.entity_ids, dtype=np.int64)[edge_rows]
    order = np.argsort(edge_ids, kind="stable")
    edge_ids = edge_ids[order]
    curves = construction.curve_edge_curves[order]
    along = construction.curve_edges[order]
    canonical = canonical_edges[edge_rows[order]]
    tangent = points[along[:, 1]] - points[along[:, 0]]
    sense = np.ones(curves.shape, dtype=np.int8)
    if kind is GeometryAssociationKind.BREP:
        for row, curve in enumerate(curves):
            patch, loop, position = domain.curve_owners[curve]
            use = domain.patches[patch].loops[loop][position]
            sense[row] = np.sign(use.last - use.first)
    edge_association = GeometryAssociation(
        kind,
        source.source_id,
        revision,
        edge_set.entity_set_id,
        edge_ids,
        tuple(domain.entity_id(1, curve) for curve in curves.tolist()),
        construction.curve_edge_deviations[order],
        exact=False,
        orientations=np.sign(
            np.sum((points[canonical[:, 1]] - points[canonical[:, 0]]) * tangent, axis=1)
        ).astype(np.int8)
        * sense,
        **_source_metadata(domain, np.full(curves.shape, 1, dtype=np.int8), curves),
    )

    def scope(
        dimension: int, entity_set_id: str, selected: np.ndarray, /
    ) -> MeshingScope:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            dimension,
            entity_set_id,
            np.sort(selected),
        )

    surface_patches = tuple(
        MeshPatch(
            f"surface:{patch}",
            scope(2, face_set.entity_set_id, face_ids[patches == patch]),
        )
        for patch in np.unique(patches).tolist()
    )
    curve_labels = tuple(
        MeshLabel(
            f"curve:{curve}",
            scope(1, edge_set.entity_set_id, edge_ids[curves == curve]),
        )
        for curve in np.unique(curves).tolist()
    )
    vertex_set = mesh.entity_set(0)
    vertex_ids = np.asarray(vertex_set.entity_ids, dtype=np.int64)
    vertex_association = GeometryAssociation(
        kind,
        source.source_id,
        revision,
        vertex_set.entity_set_id,
        vertex_ids,
        tuple(
            domain.entity_id(int(dimension), int(index))
            for dimension, index in zip(
                construction.vertex_source_dimensions,
                construction.vertex_source_indices,
                strict=True,
            )
        ),
        np.zeros((vertex_ids.size,)),
        parameters=construction.vertex_parameters,
        exact=kind is not GeometryAssociationKind.BREP,
        **_source_metadata(
            domain,
            construction.vertex_source_dimensions,
            construction.vertex_source_indices,
        ),
    )
    return (
        (face_association, edge_association, vertex_association),
        surface_patches,
        curve_labels,
    )


def _certification_request(
    source: NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    compiled: CompiledSurfaceDomain,
    construction: SurfaceConstruction,
    /,
) -> NativeCertificationRequest:
    """Independent global embedding and continuous two-sided source acceptance.

    The source owner independently verifies the full oriented chart chain,
    including collapsed pole triangles and exact trim ribbons, and recomputes
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


def _surface_semantics(
    domain: MeshingDomain,
    specification: SurfaceMeshingSpec,
    mesh: CellMesh,
    construction: SurfaceConstruction,
    labels: tuple[MeshLabel, ...],
    /,
) -> tuple[
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[MeshZone, ...],
    tuple[RegionBoundaryEvidence, ...],
]:
    """Lower authored source incidence; region boundaries may overlap."""
    face_set = mesh.entity_set(2)
    face_ids = np.asarray(face_set.entity_ids, dtype=np.int64)
    rows = _rows(
        construction.triangles,
        np.concatenate(
            tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
        ),
    )
    source_patches = construction.triangle_patches[rows]

    def target_scope(selected: np.ndarray, /) -> MeshingScope:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            face_set.entity_set_id,
            selected,
        )

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
                    target_scope(face_ids[source_patches == patch]),
                    source_adjacent_region_ids=None
                    if source_pair == (None, None)
                    else source_pair,
                )
            )
    zones = tuple(
        MeshZone(
            control.region_name,
            MeshZoneRole.REGION,
            target_scope(
                face_ids[
                    np.isin(
                        source_patches,
                        domain.resolve_indices(
                            2, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
                        ),
                    )
                ]
            ),
            material_id=control.material_id,
            region_role=control.role,
        )
        for control in specification.region_controls
        if control.scope.entity_dimension == 2
    )
    curve_controls = tuple(
        control
        for control in specification.patch_controls
        if control.scope.entity_dimension == 1
    )
    if curve_controls:
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError(
                "Surface curve patches require canonical surface edge incidence."
            )
        edge_set = mesh.entity_set(1)
        edge_rows = _rows(
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
        label = MeshLabel(name, target_scope(face_ids[np.isin(source_patches, boundary)]))
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
    return tuple(patches), tuple(boundary_labels), zones, tuple(evidence)


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
) -> tuple[CellMesh, CellGeometrySpec]:
    """Pass native topology to the geometry-owned exact source restriction."""
    from ...geometry._surface_source_support import restrict_surface_source_atlas_geometry

    target = CellMesh.from_triangles(
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
    )


def _quad_associations(
    extraction: DualExtraction,
    support: PreparedSurfaceSourceSupport,
    associations: tuple[GeometryAssociation, ...],
    /,
) -> tuple[GeometryAssociation, ...]:
    """Keep topology strata and rebuild actual original UV/curve parameters."""
    from ...geometry.brep._patches import LineCurve
    from .._association import _entity_rows, _target_dimension
    from .._quad_generation import _entities

    geometry = extraction.geometry
    if geometry is None:
        raise ValueError("Original source quads require their exact coordinate maps.")
    proof = support.prove_native_restrictions(extraction.mesh, geometry)
    uv: dict[tuple[int, int], tuple[Fraction, Fraction]] = {}
    for patch, ids, charts in zip(
        proof.patches.tolist(), proof.vertex_ids, proof.source_corners, strict=True
    ):
        for identifier, point in zip(ids, charts, strict=True):
            key = (patch, identifier)
            if key in uv and uv[key] != point:
                raise ValueError(
                    "Current source charts disagree on original UV vertex identity."
                )
            uv[key] = point
    _, _, _, inherited = remap_dual_metadata(
        extraction, associations=associations, surface_source_cover=True
    )
    outputs = []
    for association in inherited:
        dimension = _target_dimension(extraction.mesh, association)
        rows = _entity_rows(
            extraction.mesh, dimension, np.asarray(association.target_global_ids)
        )
        vertices = _entities(extraction.mesh, dimension)[rows]
        parameters = np.zeros((rows.size, 2), dtype=np.float64)
        for row, (source_dimension, index, path, values) in enumerate(
            zip(
                np.asarray(association.source_dimensions).tolist(),
                np.asarray(association.source_indices).tolist(),
                association.source_occurrence_paths,
                vertices.tolist(),
                strict=True,
            )
        ):
            source_row = support._domain_row(
                support.domain, source_dimension, index, path
            )
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
                        "Exact source curve parameters require their original affine UV coedge."
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
                association.source_entity_ids,
                np.full((rows.size,), rounding, dtype=np.float64),
                parameters=parameters,
                resolved=association.resolved,
                ambiguous=association.ambiguous,
                exact=rounding == 0.0,
                orientations=association.orientations,
                source_dimensions=association.source_dimensions,
                source_indices=association.source_indices,
                source_entity_roles=association.source_entity_roles,
                source_occurrence_paths=association.source_occurrence_paths,
            )
        )
    return tuple(outputs)


def _quad_edge_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: SurfaceMeshingSpec,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Certified physical arc lengths of actual coordinate-map edges."""
    from ...discretization._cell_geometry_transfer import _prepare_mapped_edge_arc_length
    from ...discretization._reference_cell import reference_cell_topology
    from .._quad_generation import _entities

    edges = _entities(mesh, 1)
    lookup = {tuple(sorted(pair)): row for row, pair in enumerate(edges.tolist())}
    value = np.full(edges.shape[0], np.nan, dtype=np.float64)
    lower, upper = value.copy(), value.copy()
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
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
                enclosure = _prepare_mapped_edge_arc_length(
                    element, local, points[first], points[last]
                ).integrate(
                    absolute_tolerance=1e-12,
                    relative_tolerance=1e-12,
                    maximum_work=specification.limits.maximum_work_units,
                    maximum_subcells=min(specification.limits.maximum_cells, 10000),
                )
                value[row], lower[row], upper[row] = (
                    enclosure.value,
                    enclosure.lower,
                    enclosure.upper,
                )
    return edges, value, lower, upper


def _quad_compliance(
    specification: SurfaceMeshingSpec,
    construction: SurfaceConstruction,
    extraction: DualExtraction,
    support: PreparedSurfaceSourceSupport,
    compiled: CompiledSurfaceDomain,
    associations: tuple[GeometryAssociation, ...],
    /,
) -> MeshingComplianceReport:
    from .._association import _entity_rows, _target_dimension
    from .._quality import evaluate_cell_quality
    from ._native_publication import edge_growth_evidence

    geometry = extraction.geometry
    if geometry is None:
        raise ValueError("Exact source quad compliance requires actual mapped geometry.")
    mesh, domain = extraction.mesh, support.domain
    edges, lengths, lower, upper = _quad_edge_evidence(mesh, geometry, specification)
    proof = support.prove_native_restrictions(mesh, geometry)
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
        if _target_dimension(mesh, association) != 1:
            continue
        rows = _entity_rows(mesh, 1, np.asarray(association.target_global_ids))
        for row, dimension, index, path in zip(
            rows.tolist(),
            np.asarray(association.source_dimensions).tolist(),
            np.asarray(association.source_indices).tolist(),
            association.source_occurrence_paths,
            strict=True,
        ):
            if dimension == 1:
                curve_rows[row] = support._domain_row(domain, dimension, index, path)
    requested, achieved, issues = [], [], []
    for control in specification.size_controls:
        if isinstance(control, ProximitySizeControl):
            entities = np.unique(
                np.concatenate(
                    (
                        domain.resolve_indices(
                            2, np.asarray(control.source_scope.global_entity_ids)
                        ),
                        domain.resolve_indices(
                            2, np.asarray(control.target_scope.global_entity_ids)
                        ),
                    )
                )
            )
        else:
            entities = domain.resolve_indices(
                control.scope.entity_dimension,
                np.asarray(control.scope.global_entity_ids),
            )
            if control.scope.entity_dimension == 3:
                entities = np.unique(
                    np.asarray(
                        [
                            patch
                            for region in entities.tolist()
                            for patch, _ in domain.regions[region].boundary
                        ],
                        dtype=np.int64,
                    )
                )
        entity_dimension = (
            2
            if isinstance(control, ProximitySizeControl)
            else control.scope.entity_dimension
        )
        rows = np.asarray(
            [
                row
                for row, edge in enumerate(edges.tolist())
                if (
                    curve_rows.get(row) in entities
                    if entity_dimension == 1
                    else bool(edge_patches[tuple(sorted(edge))] & set(entities.tolist()))
                )
            ],
            dtype=np.int64,
        )
        if isinstance(control, UniformSizeControl):
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
        else:
            if entity_dimension == 1:
                entities = np.asarray(
                    [
                        patch
                        for patch in range(len(domain.patches))
                        if set(domain.patch_curves(patch).tolist())
                        & set(entities.tolist())
                    ],
                    dtype=np.int64,
                )
            maximum = 0.0
            for patch in entities.tolist():
                selected = proof.patches == patch
                uv = np.mean(
                    np.asarray(proof.source_corners, dtype=np.float64)[selected], axis=1
                )
                sizes = compiled.source_sizing.evaluate(
                    domain, patch, uv, domain.evaluate(np.full(uv.shape[0], patch), uv)
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
            if isinstance(control, CurvatureSizeControl):
                normal = float(
                    np.max(
                        construction.triangle_normal_bounds[
                            np.isin(construction.triangle_patches, entities)
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
    angle = float(np.min(np.asarray(evaluate_cell_quality(mesh).minimum_angle)))
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
    source: NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedParametricSurface,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> CellMeshingResult:
    from .._quad_generation import _entities

    started = monotonic()
    limits = specification.limits
    atlas = prepare_surface_source_root_atlas(
        source.domain,
        tuple(prepared.compiled.patches.tolist()),
        coordinate_contract,
        maximum_support_queries=limits.maximum_geometry_queries,
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
    decomp = compile_surface_domain(
        source.domain, _quad_decomposition_specification(specification)
    )
    construction = generate_surface(
        decomp,
        prepared.schedule,
        limits,
        math.radians(prepared.schedule.quality_angle_degrees),
        started,
        required_minimum_angle=0.0,
        size_compliance=specification.size_compliance,
        record_phase=record_phase,
    )
    scaffold, source_geometry = _original_triangle_maps(
        construction,
        atlas,
        maximum_support_queries=limits.maximum_geometry_queries,
    )
    source_associations, _, labels = _associations(source, scaffold, construction)
    patches, labels, zones, region_boundaries = _surface_semantics(
        source.domain, specification, scaffold, construction, labels
    )
    edges = _entities(scaffold, 1)
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
    )
    extraction = extract_surface_quads(
        scaffold, limits, source_geometry=source_geometry, cross_field=field
    )
    geometry = extraction.geometry
    if geometry is None:
        raise ValueError(
            "Exact original source quad extraction lost its coordinate maps."
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
    associations = _quad_associations(extraction, support, source_associations)
    compliance = _quad_compliance(
        specification, construction, extraction, support, prepared.compiled, associations
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
        geometry=geometry,
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=CellMeshAuditDisposition.REJECT
            if _closed(prepared.compiled)
            else CellMeshAuditDisposition.SKIP,
        ),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "wall_seconds",
        ),
        unenforced_limits=(
            "work_units",
            "geometry_queries",
            "scratch_bytes",
            "cavity_cells",
        ),
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
        region_boundary_evidence=region_boundaries,
        record_phase=record_phase,
    )
    return result


def execute_parametric_surface_route(
    source: NativeSurfaceSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedParametricSurface,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Generate, associate, check, audit and publish one parametric surface."""

    started = monotonic()
    limits = specification.limits
    compiled = prepared.compiled
    if compiled.domain.domain_id != source.domain.domain_id:
        raise ValueError("The prepared constraints are bound to another domain revision.")
    if compiled.specification_id != specification.specification_id:
        raise ValueError("The prepared source controls bind another physical request.")
    if set(
        (
            *specification.target.cell_families.required,
            *specification.target.cell_families.preferred,
        )
    ) == {"quadrilateral"}:
        if prepared.generation is not None:
            raise ValueError(
                "Exact original source quad publication requires its serial source chart extraction."
            )
        return _execute_exact_source_quads(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            record_phase=record_phase,
        )
    if prepared.generation is not None:
        from .._distributed_generation import execute_distributed_surface_initial
        from .._initial_certification import publish_distributed_surface_initial

        phase_start = phase_started(record_phase)
        generated = execute_distributed_surface_initial(
            prepared.generation, minimum_angle=prepared.minimum_angle
        )
        record_elapsed(record_phase, "construction", phase_start)
        quality = specification.quality_target
        required_angle = (
            0.0 if quality is None or not quality.hard else quality.minimum_angle
        )
        phase_start = phase_started(record_phase)
        result = publish_distributed_surface_initial(
            prepared.generation,
            generated,
            source,
            coordinate_contract,
            provider,
            minimum_angle=required_angle,
            plan_id=plan_id,
        )
        record_elapsed(record_phase, "certification", phase_start)
        return result
    quality = specification.quality_target
    required_angle = 0.0 if quality is None or not quality.hard else quality.minimum_angle
    phase_start = phase_started(record_phase)
    construction = generate_surface(
        compiled,
        prepared.schedule,
        limits,
        prepared.minimum_angle,
        started,
        record_phase=record_phase,
        required_minimum_angle=required_angle,
        size_compliance=specification.size_compliance,
    )
    record_elapsed(record_phase, "construction", phase_start)
    triangles = construction.triangles.astype(np.int32)
    simplex_entity_limits(
        construction.vertices,
        triangles,
        limits,
        MeshingStageKind.SURFACE_MESHING,
        cell_kind="triangle",
    )
    phase_start = phase_started(record_phase)
    from ...geometry.brep._patches import BSplineSurfacePatch

    exact_source = all(
        isinstance(source.domain.patches[int(patch)].surface, BSplineSurfacePatch)
        for patch in compiled.patches
    )
    atlas: SurfaceSourceRootAtlas | None = None
    if exact_source:
        atlas = prepare_surface_source_root_atlas(
            source.domain,
            tuple(compiled.patches.tolist()),
            coordinate_contract,
            maximum_support_queries=limits.maximum_geometry_queries,
        )
        mesh, geometry = _original_triangle_maps(
            construction,
            atlas,
            maximum_support_queries=limits.maximum_geometry_queries,
        )
    else:
        mesh = canonicalize_cell_mesh(
            CellMesh.from_triangles(
                construction.vertices,
                triangles,
                numeric_version=source.source_revision,
            )
        )
        geometry = CellGeometrySpec.affine(mesh)
    record_elapsed(record_phase, "topology_construction", phase_start)
    phase_start = phase_started(record_phase)
    associations, patches, labels = _associations(source, mesh, construction)
    patches, labels, zones, region_boundaries = _surface_semantics(
        source.domain, specification, mesh, construction, labels
    )
    record_elapsed(record_phase, "geometry_association", phase_start)
    phase_start = phase_started(record_phase)
    compliance = _compliance(specification, construction, source.domain, compiled)
    record_elapsed(record_phase, "compliance", phase_start)
    check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
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
    watertight = (
        CellMeshAuditDisposition.REJECT
        if _closed(compiled)
        else CellMeshAuditDisposition.SKIP
    )
    from ..._meshcore import (
        current_native_execution_budget,
        current_native_host_workspace,
    )
    from .._association import _entity_rows

    faces = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    )
    rows = _rows(construction.triangles, faces)
    order = np.argsort(
        _entity_rows(
            mesh,
            2,
            np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks)),
        ),
        kind="stable",
    )
    budget = current_native_execution_budget()
    if budget is None:
        raise RuntimeError(
            "Native surface source retention requires its original active execution allowance."
        )
    parameters = budget.allocate_host_array((faces.shape[0], 3, 2), np.float64)
    root_patches = budget.allocate_host_array((faces.shape[0],), np.int64)
    for output, index in enumerate(order.tolist()):
        row = rows[index]
        root_patches[output] = construction.triangle_patches[row]
        for slot, vertex in enumerate(faces[index].tolist()):
            local = int(np.flatnonzero(construction.triangles[row] == vertex)[0])
            parameters[output, slot] = construction.triangle_parameters[row, local]
    if exact_source:
        if atlas is None:
            raise RuntimeError("Exact spline surface publication lost its root atlas.")
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
        associations,
        root_boundary,
        maximum_support_queries=limits.maximum_geometry_queries,
    )
    workspace = current_native_host_workspace()
    if workspace is not None:
        workspace.retain_owner(source_charts)
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
        certification_request,
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True, watertight_boundary=watertight
        ),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=(
            "vertices",
            "edges",
            "faces",
            "cells",
            "connectivity_entries",
            "data_bytes",
            "work_units",
            "geometry_queries",
            "wall_seconds",
            "cavity_cells",
            "scratch_bytes",
        ),
        unenforced_limits=(),
        patches=patches,
        zones=zones,
        region_boundary_evidence=region_boundaries,
        labels=labels,
        associations=associations,
        record_phase=record_phase,
        geometry=geometry,
        surface_source=source_charts,
    )
    return result


__all__ = [
    "PreparedParametricSurface",
    "execute_parametric_surface_route",
    "parametric_surface_support_issues",
]
