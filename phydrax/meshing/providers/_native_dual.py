#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry-to-canonical-result adapters for exact PL quad/all-hex dual routes."""

from __future__ import annotations

from dataclasses import replace
from time import monotonic
from typing import final, Protocol

import equinox as eqx
import jax
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh, CellValidityPolicy
from ...discretization._cell_geometry import (
    _require_scalar_coordinate_element,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._hexahedral import HexahedralConnectivity
from ...discretization._reference_cell import reference_cell_topology
from ...geometry._mapped_reference_domain import MappedReferenceDomain
from ...geometry._mesh_certificates import MappedDomainBoundarySource
from ...linalg._small_batched import SmallLinearSolvePlan, solve_small_linear
from .._association import (
    AssociationPropagationError,
    GeometryAssociation,
    GeometryAssociationKind,
)
from .._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from .._certification import MeshCertificationSchedule
from .._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingDerivativeMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from .._hex_dominant import extract_hex_dominant
from .._hex_frame_map import (
    extract_source_block_integer_grid,
    extract_source_frame_grid,
    prepare_source_block_connection,
    prepare_source_hex_frame,
    PreparedSourceHexFrame,
    SourceBlockConnection,
    SourceBlockIntegerGrid,
)
from .._hex_generation import (
    exact_mapped_volume_quality,
    extract_volume_hexes,
    generate_integer_grid_hexes,
    GridHexConstruction,
    mapped_source_lipschitz_bound,
    MappedGridHexConstruction,
    NativeHexGridSchedule,
    realize_mapped_grid_hexes,
    source_block_interval_counts,
)
from .._measurements import (
    measure_phase,
    NativeMeshingPhaseRecorder,
)
from .._organization import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from .._plc_mapped_support import certify_mapped_plc_associations
from .._quad_generation import (
    _entities,
    DualExtraction,
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
from ._native_planar import (
    _constrained_edges,
    _control_edges,
    _graded_size_compliance,
    _organization_and_associations,
    _triangulate,
    planar_support_issues,
    PreparedPlanarDomain,
)
from ._native_publication import (
    check_deadline,
    edge_size_evidence,
    NativeCertificationRequest,
    publish_native_result,
    uniform_size_compliance,
)
from ._native_sources import NativePlanarSource, NativePlcSource, source_entity_id
from ._native_volume import _exact_plc_geometry, volume_support_issues


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
    """Keep the declared sizing as a conservative decomposition preference.

    A dual hex edge is at most half its parent tetrahedron's longest edge.
    This sufficient upper bound does not prescribe a statistical target,
    guarantee a child minimum, or justify doubling every construction size.
    Final acceptance evaluates the unchanged request on the actual output.
    """
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


def dual_quad_support_issues(
    source: NativePlanarSource, specification: SurfaceMeshingSpec, /
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


def dual_hex_support_issues(
    source: NativePlcSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    issues = volume_support_issues(source, _tetrahedron_scaffold(specification))
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

    def __init__(
        self, source: NativePlanarSource, specification: SurfaceMeshingSpec, /
    ) -> None:
        issues = dual_quad_support_issues(source, specification)
        if issues:
            raise ValueError("Unsupported dual quad request: " + "; ".join(issues))
        self.scaffold = PreparedPlanarDomain(source, _triangle_scaffold(specification))
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

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        issues = dual_hex_support_issues(source, specification)
        if issues:
            raise ValueError("Unsupported dual hex request: " + "; ".join(issues))
        if not isinstance(schedule, NativeVolumeSchedule):
            raise TypeError("schedule must be NativeVolumeSchedule.")
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


def execute_quad_route(
    source: NativePlanarSource,
    specification: SurfaceMeshingSpec,
    prepared: PreparedDualQuad,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    if (
        source.binding_id != prepared.source_binding_id
        or specification.specification_id != prepared.specification_id
    ):
        raise ValueError(
            "Prepared dual quad route binds another source or specification."
        )
    started, limits = monotonic(), specification.limits
    with measure_phase(record_phase, "topology_construction"):
        triangulation, inserted = _triangulate(
            prepared.scaffold, started, _triangle_scaffold(specification)
        )
    triangles = np.asarray(triangulation.triangles, dtype=np.int64)
    used, compact = np.unique(triangles, return_inverse=True)
    points = np.asarray(triangulation.points, dtype=np.float64)[used]
    triangles = compact.reshape(triangles.shape)
    constrained, constrained_sources = _constrained_edges(
        triangles,
        np.asarray(triangulation.segment_ids, dtype=np.int64),
        prepared.scaffold.segment_sources,
    )
    missing = np.setdiff1d(
        np.arange(prepared.scaffold.source_directions.shape[0]), constrained_sources
    )
    if missing.size:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Source edge lies outside the meshed region.",
            entity_ids=tuple(missing.tolist()),
        )
    scaffold = CellMesh.from_triangles(
        points, triangles, numeric_version=source.source_revision
    )
    patches, labels, associations = _organization_and_associations(
        scaffold, source, prepared.scaffold, constrained, constrained_sources, used
    )
    domain, rounding, admissible = prepared.scaffold.domain(
        source, points, constrained, constrained_sources
    )
    edge_rows = {
        tuple(sorted(edge)): row
        for row, edge in enumerate(_entities(scaffold, 1).tolist())
    }
    features = np.asarray(
        [edge_rows[tuple(sorted(edge))] for edge in constrained.tolist()], dtype=np.int64
    )
    field = prepare_surface_cross_field(
        scaffold, feature_edges=features, record_phase=record_phase
    )
    with measure_phase(record_phase, "topology_construction"):
        extraction = extract_surface_quads(scaffold, limits, cross_field=field)
    mesh = extraction.mesh
    with measure_phase(record_phase, "geometry_association"):
        zones, patches, labels, associations = remap_dual_metadata(
            extraction, patches=patches, labels=labels, associations=associations
        )
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    edges = _entities(mesh, 1)
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    lengths = np.linalg.norm(b - a, axis=1)
    local = np.min(
        prepared.scaffold.sizes(np.concatenate((a, b, 0.5 * (a + b)))).reshape(3, -1),
        axis=0,
    )
    endpoint = prepared.scaffold.sizes(vertices)
    gradation = 1.0 + float(
        np.max(np.abs(endpoint[edges[:, 1]] - endpoint[edges[:, 0]]) / lengths)
    )
    ancestor_edges = extraction.entity_parent_rows[1]
    ancestor_dimensions = extraction.entity_parent_dimensions[1]
    old_edge_sources = np.full((scaffold.topology.entities(1).count,), -1, dtype=np.int64)
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
        if not isinstance(control, UniformSizeControl):
            raise TypeError("Admitted dual quad size control must be uniform.")
        rows = _control_edges(
            control,
            prepared.scaffold,
            np.sort(edges, axis=1),
            local,
            child_constraints,
            child_constraint_sources,
        )
        req, act, failed = _graded_size_compliance(
            control, specification.size_compliance, lengths[rows], local[rows], gradation
        )
        requested.extend(req)
        achieved.extend(act)
        issues.extend(failed)
    quality = evaluate_cell_quality(mesh)
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
            ("cross_field_objective", float(np.asarray(field.optimization.objective))),
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
    compliance = MeshingComplianceReport(
        specification.specification_id,
        requested=tuple(requested),
        achieved=tuple(achieved),
        issues=tuple(issues),
    )
    check_deadline(started, limits, MeshingStageKind.GEOMETRY_AUDIT)
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
            cell_regions=np.zeros((mesh.topology.entities(2).count,), dtype=np.int64),
        ),
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=CellMeshAuditDisposition.REJECT,
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
            "cavity_cells",
            "geometry_queries",
            "scratch_bytes",
            "data_bytes",
        ),
        zones=zones,
        patches=patches,
        labels=labels,
        associations=associations,
        record_phase=record_phase,
    )


def execute_hex_route(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedDualHex,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    if (
        source.binding_id != prepared.source_binding_id
        or specification.specification_id != prepared.specification_id
    ):
        raise ValueError("Prepared dual hex route binds another source or specification.")
    started, limits = monotonic(), specification.limits
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
    )
    with measure_phase(record_phase, "geometry_association"):
        source_geometry = _exact_plc_geometry(construction, source)
        if source_geometry is None:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                "Dual hex construction requires the actual original PLC coordinate source.",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            )
    with measure_phase(record_phase, "topology_construction"):
        extraction = extract_volume_hexes(
            construction.mesh, limits, source_geometry=source_geometry
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
    )


def _publish_volume_extraction(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedDualHex | PreparedHexDominant,
    extraction: DualExtraction,
    construction: VolumeConstruction,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    started: float,
    route: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    mapped_quality: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> CellMeshingResult:
    mesh, limits = extraction.mesh, specification.limits
    if extraction.geometry is None:
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "Hex publication requires the actual owning source coordinate maps, including transition apex maps.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        )
    geometry = extraction.geometry
    certification_request = NativeCertificationRequest(
        MeshCertificationSchedule("volume_plc"),
        source.source_id,
        source.source_revision,
        limits,
        domain=construction.domain,
        cell_regions=construction.cell_regions[extraction.parent_cells],
    )
    with measure_phase(record_phase, "geometry_association"):
        zones, patches, labels, associations = remap_dual_metadata(
            extraction,
            zones=construction.zones,
            patches=construction.patches,
            labels=construction.labels,
            associations=construction.associations,
        )
        if route in ("plc_dual_hex", "plc_hex_dominant", "plc_balanced_cut_hex"):
            # Source constraints retain the original native authority ordering.
            # The proof evaluates actual Q1 hex and rational pyramid/transition
            # maps, not rounded membership in the tetrahedral scaffold.
            support_source = prepare_plc_source(
                source.complex,
                source.source_id,
                source.source_revision,
                coordinate_contract,
                limits=limits,
                record_phase=record_phase,
            )
            try:
                if (
                    support_source.association_transfer.domain.domain_id
                    != construction.domain.domain_id
                ):
                    raise ValueError(
                        "Mapped source support and publication name different authoritative PLC domains."
                    )
                association_proof = certify_mapped_plc_associations(
                    support_source.association_transfer,
                    mesh,
                    geometry,
                    associations,
                    validity=extraction.validity,
                    certificate_limits=certification_request.certificate_limits,
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
    control = specification.size_controls[0]
    if not isinstance(control, UniformSizeControl):
        raise TypeError("Admitted dual hex size control must be uniform.")
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    edges = _entities(mesh, 1)
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
                float(np.max(np.asarray(association.residuals)[matched], initial=0.0)),
            )
        missing = tuple(
            int(index) for index in requested_indices if int(index) not in present
        )
        if missing:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                "Protected source features lack complete exact published source-map association evidence.",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
                entity_ids=missing,
            )
        requested.append((key, feature.maximum_deviation))
        achieved.append((key, deviation))
        if feature.hard and deviation > feature.maximum_deviation:
            issues.append(key)
    if construction.work_units + extraction.work_units > limits.maximum_work_units:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Constrained decomposition plus hex closure exhausts the work budget.",
            stage=MeshingStageKind.VOLUME_FILL.value,
        )
    counts: dict[str, int] = {}
    for block in mesh.blocks:
        counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
    quality = evaluate_cell_quality(mesh)
    if mapped_quality is None:
        with measure_phase(record_phase, "compliance"):
            mapped_quality = exact_mapped_volume_quality(mesh, geometry)
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
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=CellMeshAuditDisposition.REJECT,
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
            "cavity_cells",
            "geometry_queries",
            "scratch_bytes",
            "data_bytes",
        ),
        zones=zones,
        patches=patches,
        labels=labels,
        associations=associations,
        geometry=geometry,
        record_phase=record_phase,
    )


def hex_dominant_support_issues(
    source: NativePlcSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    issues = volume_support_issues(source, _tetrahedron_scaffold(specification))
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

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
        *,
        core_fraction: float = 0.8,
    ) -> None:
        issues = hex_dominant_support_issues(source, specification)
        if issues:
            raise ValueError("Unsupported hex-dominant request: " + "; ".join(issues))
        if not isinstance(schedule, NativeVolumeSchedule):
            raise TypeError("schedule must be NativeVolumeSchedule.")
        if not np.isfinite(core_fraction) or not 0.0 < core_fraction <= 1.0:
            raise ValueError("core_fraction must lie in (0, 1].")
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


def execute_hex_dominant_route(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedHexDominant,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    if (
        source.binding_id != prepared.source_binding_id
        or specification.specification_id != prepared.specification_id
    ):
        raise ValueError(
            "Prepared hex-dominant route binds another source or specification."
        )
    started = monotonic()
    construction = generate_plc_volume(
        source.complex,
        _tetrahedron_scaffold(specification),
        prepared.schedule,
        validity_policy=CellValidityPolicy(),
        source_id=source.source_id,
        source_revision=source.source_revision,
        input_id=prepared.prepared_id,
        record_phase=record_phase,
    )
    with measure_phase(record_phase, "geometry_association"):
        source_geometry = _exact_plc_geometry(construction, source)
        if source_geometry is None:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                "Hex-dominant construction requires the actual original PLC coordinate source.",
                stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
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
    selected_count = max(1, int(np.ceil(prepared.core_fraction * cells.shape[0])))
    with measure_phase(record_phase, "topology_construction"):
        extraction = extract_hex_dominant(
            construction.mesh,
            specification.limits,
            specification.target.cell_families,
            hex_core_cells=identifiers[order[:selected_count]],
            source_geometry=source_geometry,
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
    )


class PreparedGridHex(StrictModule, NonTrainableState):
    volume_schedule: NativeVolumeSchedule
    grid_schedule: NativeHexGridSchedule
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        volume_schedule: NativeVolumeSchedule,
        grid_schedule: NativeHexGridSchedule,
        /,
    ) -> None:
        issues = dual_hex_support_issues(source, specification)
        if issues:
            raise ValueError("Unsupported grid hex request: " + "; ".join(issues))
        if not isinstance(volume_schedule, NativeVolumeSchedule) or not isinstance(
            grid_schedule, NativeHexGridSchedule
        ):
            raise TypeError(
                "Grid hex preparation requires native volume and grid schedules."
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
    source: NativePlcSource,
    old: VolumeConstruction,
    grid: GridHexConstruction,
    /,
) -> tuple[
    tuple[MeshZone, ...],
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[GeometryAssociation, ...],
]:
    """Bind grid entities to authoritative PLC strata by exact plane/chain facts."""
    mesh, complex_ = grid.mesh, source.complex
    connectivity = mesh.connectivity
    if not isinstance(connectivity, HexahedralConnectivity):
        raise RuntimeError(
            "Integer-grid metadata requires the actual canonical hexahedral connectivity."
        )
    sets = tuple(mesh.topology.entities(degree) for degree in range(4))
    ids = tuple(np.asarray(entities.entity_ids, dtype=np.int64) for entities in sets)
    facets = grid.face_source_facets
    face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
    face_edges = np.asarray(connectivity.face_edges, dtype=np.int64)
    edge_vertices = _entities(mesh, 1)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    boundary_faces = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    if np.any(boundary_faces & (facets < 0)):
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            "An exposed grid face has no authoritative source facet.",
        )

    def scope(dimension: int, selected: np.ndarray, /) -> MeshingScope:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            dimension,
            sets[dimension].entity_set_id,
            selected,
        )

    zones = tuple(
        MeshZone(name, MeshZoneRole.REGION, scope(3, ids[3][grid.cell_regions == region]))
        for region, name in enumerate(complex_.region_ids)
        if np.any(grid.cell_regions == region)
    )
    facet_rows = np.flatnonzero(facets >= 0)
    patches = tuple(
        MeshPatch(f"facet:{facet}", scope(2, ids[2][facets == facet]))
        for facet in np.unique(facets[facet_rows]).tolist()
    )
    incidence = complex_.facet_regions[facets[facet_rows]]
    interface = (
        (incidence[:, 0] >= 0)
        & (incidence[:, 1] >= 0)
        & (incidence[:, 0] != incidence[:, 1])
    )
    sheet = incidence[:, 0] == incidence[:, 1]
    labels = tuple(
        MeshLabel(name, scope(2, selected))
        for name, selected in (
            ("interface", ids[2][facet_rows][interface]),
            ("internal_sheet", ids[2][facet_rows][sheet]),
        )
        if selected.size
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    rounding = 128.0 * np.finfo(np.float64).eps * np.max(np.abs(points), initial=1.0)
    facet_names = {
        facet: source_entity_id(source.source_revision, "facet", facet)
        for facet in np.unique(facets[facet_rows]).tolist()
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
    associations = [
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            source.source_revision,
            sets[3].entity_set_id,
            ids[3],
            tuple(
                source_entity_id(source.source_revision, "region", int(region))
                for region in grid.cell_regions
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
    old_edges = _entities(old.mesh, 1)
    solve = solve_small_linear(
        SmallLinearSolvePlan(3),
        np.broadcast_to(grid.frame.basis.T, (old.mesh.coordinates.shape[0], 3, 3)),
        old.mesh.coordinates,
    )
    old_points = np.asarray(jax.device_get(solve.value), dtype=np.float64)
    if not np.all(np.asarray(jax.device_get(solve.successful), dtype=np.bool_)):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Source edge chart pullback failed.",
        )
    old_ids = np.asarray(old.mesh.topology.entities(1).entity_ids, dtype=np.int64)
    old_rows = {identifier: row for row, identifier in enumerate(old_ids.tolist())}
    source_edges: list[tuple[np.ndarray, str, int]] = []
    for association in old.associations:
        if (
            association.target_entity_set_id
            != old.mesh.topology.entities(1).entity_set_id
        ):
            continue
        for row, identifier in enumerate(
            np.asarray(association.target_global_ids).tolist()
        ):
            source_edges.append(
                (
                    old_points[old_edges[old_rows[identifier]]],
                    association.source_entity_ids[row],
                    int(np.asarray(association.orientations)[row]),
                )
            )
    chart = points @ grid.frame.basis.T
    edge_names: dict[int, str] = {}
    edge_orientations: dict[int, int] = {}
    ridge_names: dict[int, str] = {}
    for row, adjacent in enumerate(edge_facets):
        if not adjacent:
            continue
        if len(adjacent) == 1:
            edge_names[row] = facet_names[next(iter(adjacent))]
            edge_orientations[row] = 0
            continue
        endpoints = chart[edge_vertices[row]]
        variable = np.flatnonzero(endpoints[0] != endpoints[1])
        if variable.size != 1:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "A grid ridge has no exact one-axis source chain.",
            )
        axis = int(variable[0])
        fixed = [coordinate for coordinate in range(3) if coordinate != axis]
        lower, upper = np.min(endpoints[:, axis]), np.max(endpoints[:, axis])
        intervals: dict[str, list[tuple[float, float, int]]] = {}
        for segment, entity, orientation in source_edges:
            if (
                np.any(segment[:, fixed] != endpoints[0, fixed])
                or np.count_nonzero(segment[0] != segment[1]) != 1
            ):
                continue
            first, last = np.min(segment[:, axis]), np.max(segment[:, axis])
            if first < upper and last > lower:
                relative = orientation * (
                    1
                    if (segment[1, axis] - segment[0, axis])
                    * (endpoints[1, axis] - endpoints[0, axis])
                    > 0.0
                    else -1
                )
                intervals.setdefault(entity, []).append(
                    (float(first), float(last), relative)
                )
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
        if len(complete) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "Grid ridge has ambiguous or incomplete authoritative source-chain coverage.",
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
                [edge_orientations[row] for row in edge_rows.tolist()], dtype=np.int8
            ),
        )
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
        tuple(key): source_entity_id(source.source_revision, "vertex", row)
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
            entity = source_entity_id(
                source.source_revision, "region", next(iter(vertex_regions[row]))
            )
        else:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "A grid coordinate vertex has unresolved source-junction ancestry.",
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
    return zones, patches, labels, tuple(associations)


def execute_grid_hex_route(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedGridHex,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    if (
        source.binding_id != prepared.source_binding_id
        or specification.specification_id != prepared.specification_id
    ):
        raise ValueError("Prepared grid hex route binds another source or specification.")
    started = monotonic()
    construction = generate_plc_volume(
        source.complex,
        _tetrahedron_scaffold(specification),
        prepared.volume_schedule,
        validity_policy=CellValidityPolicy(),
        source_id=source.source_id,
        source_revision=source.source_revision,
        input_id=prepared.prepared_id,
        record_phase=record_phase,
    )
    control = specification.size_controls[0]
    if not isinstance(control, UniformSizeControl):
        raise TypeError("Admitted grid hex size control must be uniform.")
    size = (
        control.maximum_size if control.maximum_size is not None else control.target_size
    )
    grid = generate_integer_grid_hexes(
        source.complex,
        construction,
        prepared.grid_schedule,
        specification.limits,
        size,
        record_phase=record_phase,
    )
    with measure_phase(record_phase, "geometry_association"):
        zones, patches, labels, associations = _grid_metadata(source, construction, grid)
    mesh = grid.mesh
    quality = evaluate_cell_quality(mesh)
    minimum = float(np.min(np.asarray(quality.scaled_jacobian)))
    lengths, growth = edge_size_evidence(
        np.asarray(mesh.coordinates, dtype=np.float64), _entities(mesh, 1)
    )
    requested, achieved, issues = uniform_size_compliance(
        control, specification.size_compliance, lengths, growth
    )
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
            ("frame_alignment_residual", float(np.max(grid.frame.alignment_residuals))),
            ("singular_source_vertices", grid.frame.singular_vertices.size),
        )
    )
    if minimum < prepared.grid_schedule.minimum_scaled_jacobian:
        issues.append("minimum_scaled_jacobian")
    mean_ratio = float(np.min(np.asarray(quality.mean_ratios)))
    aspect_ratio = float(np.max(np.asarray(quality.aspect_ratios)))
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
    rounding = (
        128.0
        * np.finfo(np.float64).eps
        * np.max(np.abs(np.asarray(mesh.coordinates)), initial=1.0)
    )
    for feature in specification.protected_features:
        requested.append(
            (
                f"protected:{feature.feature_id}:maximum_deviation",
                feature.maximum_deviation,
            )
        )
        achieved.append((f"protected:{feature.feature_id}:maximum_deviation", rounding))
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
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=CellMeshAuditDisposition.REJECT,
        ),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=("vertices", "cells", "connectivity_entries", "wall_seconds"),
        unenforced_limits=(
            "edges",
            "faces",
            "work_units",
            "cavity_cells",
            "geometry_queries",
            "scratch_bytes",
            "data_bytes",
        ),
        zones=zones,
        patches=patches,
        labels=labels,
        associations=associations,
        record_phase=record_phase,
    )


class _MappedHexSourceBinding(Protocol):
    reference: NativePlcSource
    domain: MappedReferenceDomain
    source_id: str
    source_revision: str
    binding_id: str


def _mapped_face_regions(domain: MappedReferenceDomain, /) -> tuple[tuple[str, ...], ...]:
    """Read authoritative material adjacency from the declared root complex."""
    mesh = domain.reference_mesh
    faces, cells = _entities(mesh, 2), _entities(mesh, 3)
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
    """Admit physical identities only on complete, independently named root strata."""
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
        _require_scalar_coordinate_element(element, "mapped source").degree
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
    if len(specification.size_controls) != 1 or not isinstance(
        specification.size_controls[0], UniformSizeControl
    ):
        issues.append("one whole-domain uniform physical size control")
    elif specification.size_controls[0].scope.scope_id != boundary.scope_id:
        issues.append("a size control on the requested mapped boundary scope")
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


class PreparedMappedGridHex(StrictModule, NonTrainableState):
    grid_schedule: NativeHexGridSchedule
    source_frame: PreparedSourceHexFrame | None
    source_connection: SourceBlockConnection | None
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: _MappedHexSourceBinding,
        specification: VolumeMeshingSpec,
        grid_schedule: NativeHexGridSchedule,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
    ) -> None:
        issues = mapped_grid_support_issues(source, specification)
        if issues:
            raise ValueError("Unsupported mapped grid hex request: " + "; ".join(issues))
        if not isinstance(grid_schedule, NativeHexGridSchedule):
            raise TypeError("grid_schedule must be NativeHexGridSchedule.")
        self.grid_schedule = grid_schedule
        self.source_frame = (
            prepare_source_hex_frame(
                source.domain.reference_mesh,
                source.domain.source_geometry,
                record_phase=record_phase,
            )
            if grid_schedule.route == "frame_grid"
            else None
        )
        self.source_connection = (
            None
            if self.source_frame is None
            else prepare_source_block_connection(
                self.source_frame, record_phase=record_phase
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
) -> tuple[
    tuple[MeshZone, ...],
    tuple[MeshPatch, ...],
    tuple[MeshLabel, ...],
    tuple[GeometryAssociation, ...],
]:
    """Map source-root strata by exact chart containment and scientific incidence."""
    from .._association import MappedReferenceAssociationTransfer

    mesh = mapped.mesh
    transfer = MappedReferenceAssociationTransfer(
        source.domain,
        maximum_support_queries=specification.limits.maximum_geometry_queries,
    )
    associations = transfer.associations(mesh, mapped.geometry)
    target_entities = tuple(_entities(mesh, degree) for degree in range(4))
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    target_lookup = tuple(
        {
            tuple(sorted(target_vertex_ids[vertices].tolist())): row
            for row, vertices in enumerate(rows)
        }
        for rows in target_entities
    )
    topology = reference_cell_topology("hexahedron")
    target_cells = target_entities[3]

    def scope(degree: int, identifiers: np.ndarray, /) -> MeshingScope:
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
    for region, region_id in enumerate(source.domain.region_ids):
        if not np.any(mapped.cell_regions == region):
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
                scope(3, cell_ids[mapped.cell_regions == region]),
                material_id=None if control is None else control.material_id,
                region_role=None if control is None else control.role,
            )
        )
    dimensions = np.asarray(associations[2].parent_dimensions, dtype=np.int64)
    parent_ids = np.asarray(associations[2].parent_ids, dtype=np.int64)
    faces = np.asarray(mesh.topology.entities(2).entity_ids, dtype=np.int64)
    patches = [
        MeshPatch(
            source.domain.image_entity_id(2, parent),
            scope(2, faces[(dimensions == 2) & (parent_ids == parent)]),
        )
        for parent in np.unique(parent_ids[dimensions == 2]).tolist()
    ]
    for control in specification.patch_controls:
        selected = (dimensions == 2) & np.isin(
            parent_ids, np.asarray(control.scope.entity_ids)
        )
        if control.required and not np.any(selected):
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                "A required mapped patch has no published child faces.",
            )
        patches.append(MeshPatch(control.name, scope(2, faces[selected])))
    interface_faces = []
    face_owners: list[list[int]] = [[] for _ in target_entities[2]]
    for cell_row, vertices in enumerate(target_cells):
        for local in topology.entities[2]:
            key = tuple(sorted(target_vertex_ids[vertices[list(local)]].tolist()))
            face_owners[target_lookup[2][key]].append(cell_row)
    for row, owners in enumerate(face_owners):
        if (
            len(owners) == 2
            and mapped.cell_regions[owners[0]] != mapped.cell_regions[owners[1]]
        ):
            if dimensions[row] != 2:
                raise MeshingFailure(
                    MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                    "Mapped material interface lacks a declared root-face identity.",
                )
            interface_faces.append(row)
    labels = (
        (
            MeshLabel(
                "interface", scope(2, faces[np.asarray(interface_faces, dtype=np.int64)])
            ),
        )
        if interface_faces
        else ()
    )
    return tuple(zones), tuple(patches), labels, associations


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
) -> CellMeshingResult:
    if (
        source.binding_id != prepared.source_binding_id
        or specification.specification_id != prepared.specification_id
    ):
        raise ValueError(
            "Prepared mapped grid route binds another source or specification."
        )
    started = monotonic()
    control = specification.size_controls[0]
    if not isinstance(control, UniformSizeControl):
        raise TypeError("Mapped grid size control must be uniform.")
    size = (
        control.maximum_size if control.maximum_size is not None else control.target_size
    )
    if (prepared.grid_schedule.route == "frame_grid") != (
        prepared.source_frame is not None
    ) or (prepared.grid_schedule.route == "frame_grid") != (
        prepared.source_connection is not None
    ):
        raise ValueError(
            "Prepared source-frame extraction differs from its declared portfolio route."
        )
    if prepared.source_frame is not None and (
        prepared.source_frame.source_id != cell_geometry_id(source.domain.source_geometry)
        or prepared.source_frame.reference_mesh.mesh_id
        != source.domain.reference_mesh.mesh_id
    ):
        raise ValueError(
            "Prepared block frames name a different original coordinate source or reference mesh."
        )
    with measure_phase(record_phase, "construction"):
        bound = mapped_source_lipschitz_bound(
            source.domain.reference_mesh, source.domain.source_geometry
        )
        if prepared.source_frame is None:
            grid = generate_integer_grid_hexes(
                source.reference.complex,
                source.domain,
                prepared.grid_schedule,
                specification.limits,
                size / bound,
                record_phase=record_phase,
            )
            grid_identity = grid.construction_id
        else:
            intervals = source_block_interval_counts(
                source.domain.reference_mesh,
                bound,
                size,
                prepared.grid_schedule.maximum_depth,
            )
            grid = extract_source_block_integer_grid(
                prepared.source_frame,
                intervals,
                source.domain.cell_regions,
                specification.limits,
                prepared.grid_schedule.maximum_depth,
                source_connection=prepared.source_connection,
                record_phase=record_phase,
            )
            grid_identity = grid.grid_id
    with measure_phase(record_phase, "curving"):
        if prepared.source_frame is None:
            mapped = realize_mapped_grid_hexes(
                grid,
                source.domain.reference_mesh,
                source.domain.source_geometry,
                source.domain.cell_regions,
                specification.limits,
            )
        else:
            mapped = extract_source_frame_grid(
                prepared.source_frame,
                grid,
                source.domain.cell_regions,
                specification.limits,
            )
    with measure_phase(record_phase, "geometry_association"):
        zones, patches, labels, associations = _mapped_grid_metadata(
            source, mapped, specification
        )
    mesh = mapped.mesh
    with measure_phase(record_phase, "compliance"):
        lengths, growth = edge_size_evidence(
            np.asarray(mesh.coordinates, dtype=np.float64), _entities(mesh, 1)
        )
        requested, achieved, issues = uniform_size_compliance(
            control, specification.size_compliance, lengths, growth
        )
        minimum_scaled = float(np.min(mapped.scaled_jacobian_lower))
        minimum_mean = float(np.min(mapped.mean_ratio_lower))
        aspect = float(np.max(mapped.aspect_ratio_upper))
        requested.extend(
            (
                (
                    "minimum_mapped_scaled_jacobian",
                    prepared.grid_schedule.minimum_scaled_jacobian,
                ),
                ("minimum_mapped_mean_ratio", prepared.grid_schedule.minimum_mean_ratio),
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
        if isinstance(grid, SourceBlockIntegerGrid):
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
        geometry=mapped.geometry,
        record_phase=record_phase,
        audit_policy=CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=CellMeshAuditDisposition.REJECT,
        ),
        derivative_mode=MeshingDerivativeMode.NONDIFFERENTIABLE,
        enforced_limits=("vertices", "cells", "connectivity_entries", "wall_seconds"),
        unenforced_limits=(
            "edges",
            "faces",
            "work_units",
            "cavity_cells",
            "geometry_queries",
            "scratch_bytes",
            "data_bytes",
        ),
        zones=zones,
        patches=patches,
        labels=labels,
        associations=associations,
    )
