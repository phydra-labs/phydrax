#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh BRep execution lifecycle: prepare, generate, extract, organize, certify."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._identity import SemanticProvenance
from ...discretization import CellBlock, CellMesh
from ...discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ...discretization._hexahedral import HexahedralConnectivity
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._audit import audit_cell_mesh
from .._canonical import canonicalize_cell_mesh
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
    VolumeMeshingSpec,
)
from .._controls import BackgroundMetricControl, BackgroundMetricMode
from .._organization import MeshAttribute, MeshPatch, MeshZone
from .._quality import evaluate_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._sizing import ProximitySizeControl, SizeControlStrength, UniformSizeControl
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from ._gmsh_constraints import (
    _apply_protected_features,
    _audit_periodic,
    _audit_protected_features,
    _set_periodic,
)
from ._gmsh_elements import (
    _cell_geometry,
    _CurvedGeometry,
    _element_rows,
    _local_connectivity,
    _native_quality,
    _NativeQuality,
)
from ._gmsh_evidence import (
    _boundary_association,
    _canonical_cell_solid_ids,
    _EvidenceSection,
    _planar_surface_evidence,
    _region_evidence,
    _semantic_surface_evidence,
)
from ._gmsh_import import (
    _CadImportCache,
    _resolve_cad_entity_map,
    _resolve_planar_cad_entity_map,
    _source_scale,
)
from ._gmsh_layers import (
    _apply_boundary_layer_field,
    _apply_planar_band_constraints,
    _audit_boundary_layer_field,
    _audit_layers,
    _audit_planar_band_fronts,
    _audit_swept_interfaces,
    _install_swept_cells,
    _layer_attributes,
    _prepare_swept_geometry,
)
from ._gmsh_options import GmshHighOrderOptimization
from ._gmsh_sizing import (
    _apply_size_fields,
    _background_metric_evidence,
    _edge_size_evidence,
    _proximity_evidence,
    _semantic_size_compliance,
    _size_values,
    _SizeFields,
)


@dataclass(frozen=True, slots=True)
class _Generation:
    """Gmsh model state installed before mesh generation."""

    shape: object
    dimension: int
    geometry_order: int
    semantic_volume: bool
    semantic_surface: bool
    requested_kinds: set[str]
    target: float | None
    cad_entities: object
    size_fields: _SizeFields
    sweep: object
    band_generation: object
    layer_field: object
    periodic_records: tuple
    protected: tuple


def _configure_options(
    gmsh: Any,
    plan: Any,
    background: BackgroundMetricControl | None,
    requested_kinds: set[str],
    /,
) -> None:
    specification = plan.specification
    options = plan.options
    minimum, target, maximum, curvature_points = _size_values(specification)
    anisotropic = (
        background is not None and background.mode is BackgroundMetricMode.ANISOTROPIC
    )
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1 if options.terminal_output else 0)
    gmsh.option.setNumber("General.NumThreads", options.num_threads)
    gmsh.option.setNumber("Mesh.Algorithm", options.algorithm_2d.gmsh_code)
    gmsh.option.setNumber("Mesh.Algorithm3D", options.algorithm_3d.gmsh_code)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0 if minimum is None else minimum)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0e22 if maximum is None else maximum)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_points)
    gmsh.option.setNumber(
        "Mesh.MeshSizeFromPoints", 1 if target is not None and not anisotropic else 0
    )
    # A tensor background field alone sizes anisotropic boundaries and interiors.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0 if anisotropic else 1)
    # Options outlive gmsh.clear(); every execution sets each option it relies on.
    gmsh.option.setNumber(
        "Mesh.AnisoMax", background.metric.maximum_anisotropy if anisotropic else 1.0e33
    )
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 0)
    gmsh.option.setNumber("Mesh.Renumber", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", specification.target.geometry_order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber(
        "Mesh.HighOrderOptimize", options.high_order_optimization.gmsh_code
    )
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if options.optimize_netgen else 0)
    pure_recombined = requested_kinds in ({"quadrilateral"}, {"hexahedron"})
    # Full-quad algorithm 3 halves every edge and cannot represent a single
    # exact normal layer. Blossom recombination preserves those transfinite
    # one-cell strips; later family compliance rejects unrecombined remainder.
    recombination_algorithm = (
        1 if pure_recombined and plan.planar_bands is not None else 3
    )
    gmsh.option.setNumber(
        "Mesh.RecombinationAlgorithm",
        recombination_algorithm if pure_recombined else 0,
    )
    gmsh.model.add(f"phydrax-{plan.plan_id[:12]}")


def _prepare_generation(
    gmsh: Any,
    plan: Any,
    cache: _CadImportCache,
    background: BackgroundMetricControl | None,
    /,
) -> _Generation:
    source = plan.source
    specification = plan.specification
    report = source.report
    entry = cache.acquire(source)
    shape = entry.shape
    limits = specification.limits
    if report.num_faces + report.num_edges + report.num_vertices > limits.maximum_faces:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "BRep entity count exceeds the meshing limit.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    _, target, _, _ = _size_values(specification)
    dimension = specification.target.topological_dimension
    semantic_volume = isinstance(specification, VolumeMeshingSpec) and bool(
        specification.region_controls
    )
    semantic_surface = isinstance(specification, SurfaceMeshingSpec) and bool(
        specification.target.ambient_dimension == 2
        or specification.region_controls
        or specification.patch_controls
        or plan.planar_bands is not None
    )
    family_policy = specification.target.cell_families
    requested_kinds = {
        *family_policy.required,
        *family_policy.preferred,
        *family_policy.allowed_transitions,
    }
    _configure_options(gmsh, plan, background, requested_kinds)
    cache.import_shapes(gmsh, entry)
    if semantic_volume:
        cad_entities = cache.entity_map(
            entry, "volume", lambda: _resolve_cad_entity_map(gmsh, source, shape)
        )
    elif semantic_surface:
        cad_entities = cache.entity_map(
            entry, "planar", lambda: _resolve_planar_cad_entity_map(gmsh, source, shape)
        )
    else:
        cad_entities = None
    protected = _apply_protected_features(
        gmsh,
        source,
        shape,
        specification,
        not (
            semantic_volume
            or semantic_surface
            or specification.periodic_constraints
            or (
                isinstance(specification, VolumeMeshingSpec)
                and specification.layer_controls
            )
        ),
    )
    size_fields = _apply_size_fields(
        gmsh,
        source,
        shape,
        specification,
        cad_entities,
        1.0e22 if target is None else target,
        background,
    )
    sweep = (
        _prepare_swept_geometry(gmsh, plan, shape, cad_entities)
        if isinstance(specification, VolumeMeshingSpec)
        else None
    )
    if dimension == 2 and "quadrilateral" in requested_kinds:
        for entity_dimension, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setRecombine(entity_dimension, tag)
    band_generation = _apply_planar_band_constraints(
        gmsh,
        plan.planar_bands,
        cad_entities,
        requested_kinds,
        target,
    )
    layer_field = _apply_boundary_layer_field(
        gmsh, specification, cad_entities, requested_kinds
    )
    periodic_records = _set_periodic(gmsh, plan, shape)
    return _Generation(
        shape,
        dimension,
        specification.target.geometry_order,
        semantic_volume,
        semantic_surface,
        requested_kinds,
        target,
        cad_entities,
        size_fields,
        sweep,
        band_generation,
        layer_field,
        periodic_records,
        protected,
    )


def _generate(gmsh: Any, plan: Any, generation: _Generation, /) -> None:
    limits = plan.specification.limits
    top_entities = sorted(gmsh.model.getEntities(generation.dimension))
    if not top_entities:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "BRep source has no requested top-dimensional entities.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    if len(gmsh.model.getEntities()) > limits.maximum_faces:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Imported Gmsh entities exceed the declared limit.",
        )
    point_entities = gmsh.model.getEntities(0)
    if point_entities and generation.target is not None:
        gmsh.model.mesh.setSize(point_entities, generation.target)
    for entity_dimension, tag in top_entities:
        gmsh.model.addPhysicalGroup(entity_dimension, [tag], tag=tag)
    if generation.sweep is None:
        gmsh.model.mesh.generate(generation.dimension)
    else:
        gmsh.model.mesh.generate(2)
        # ty: ignore[invalid-argument-type]
        _install_swept_cells(gmsh, generation.sweep)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 0)


@dataclass(frozen=True, slots=True)
class _Extraction:
    node_tags: np.ndarray
    points: np.ndarray
    top: tuple
    cell_count: int
    native: _NativeQuality
    periodic: _EvidenceSection
    layer_audit: object
    bands: _EvidenceSection
    layer_field: _EvidenceSection


def _extract(gmsh: Any, plan: Any, generation: _Generation, /) -> _Extraction:
    limits = plan.specification.limits
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    order = np.argsort(node_tags, kind="stable")
    node_tags = node_tags[order]
    points = np.asarray(node_coordinates, dtype=np.float64).reshape((-1, 3))[order]
    if points.shape[0] > limits.maximum_vertices:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Generated Gmsh mesh exceeds maximum_vertices.",
        )
    top = _element_rows(gmsh, generation.dimension, generation.geometry_order)
    cell_count = sum(rows.tags.size for rows in top)
    if cell_count > limits.maximum_cells:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Generated Gmsh mesh exceeds maximum_cells.",
        )
    native = _native_quality(gmsh, top)
    periodic = _audit_periodic(gmsh, generation.periodic_records, node_tags, points)
    layer_audit = _audit_layers(generation.sweep, top, node_tags, points)
    band_requested, band_achieved = _audit_planar_band_fronts(
        gmsh,
        # ty: ignore[invalid-argument-type]
        generation.band_generation,
    )
    field_requested, field_achieved, field_issues = _audit_boundary_layer_field(
        gmsh,
        # ty: ignore[invalid-argument-type]
        generation.layer_field,
        top,
        node_tags,
        points,
    )
    return _Extraction(
        node_tags,
        points,
        top,
        cell_count,
        native,
        periodic,
        layer_audit,
        _EvidenceSection(band_requested, band_achieved),
        _EvidenceSection(field_requested, field_achieved, field_issues),
    )


@dataclass(frozen=True, slots=True)
class _CanonicalMesh:
    mesh: CellMesh
    mesh_points: np.ndarray
    vertex_ids: np.ndarray
    output_points: np.ndarray
    top_vertices: dict[str, np.ndarray]
    corner_nodes: np.ndarray
    source_to_corner: np.ndarray
    row_orders: dict[str, np.ndarray]
    boundary_rows: tuple
    mixed_boundary: bool


def _output_points(plan: Any, points: np.ndarray, /) -> np.ndarray:
    specification = plan.specification
    if not (
        isinstance(specification, SurfaceMeshingSpec)
        and specification.target.ambient_dimension == 2
    ):
        return points
    embedding = specification.planar_embedding
    if embedding is None:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "Ambient-dimension-two execution lost its PlanarEmbedding.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    residuals = np.abs(embedding.plane_residual(points))
    plane_scale = max(
        1.0,
        float(np.max(np.abs(points), initial=0.0)),
        float(np.max(np.abs(np.asarray(embedding.origin)), initial=0.0)),
    )
    plane_tolerance = 8192.0 * np.finfo(np.float64).eps * plane_scale
    if np.max(residuals, initial=0.0) > plane_tolerance:
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "CAD-associated Gmsh nodes do not lie in the declared planar embedding.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        )
    return embedding.to_planar(points)


def _canonical_mesh(
    gmsh: Any, plan: Any, generation: _Generation, extraction: _Extraction, /
) -> _CanonicalMesh:
    top = extraction.top
    top_vertices = {
        rows.block_name: _local_connectivity(extraction.node_tags, rows.vertices)
        for rows in top
    }
    corner_nodes = np.unique(
        np.concatenate(
            [
                top_vertices[rows.block_name][:, : rows.corner_count].reshape(-1)
                for rows in top
            ]
        )
    )
    output_points = _output_points(plan, extraction.points)
    corner_points = output_points[corner_nodes]
    corner_order = np.lexsort(
        tuple(
            corner_points[:, column]
            for column in range(corner_points.shape[1] - 1, -1, -1)
        )
    )
    corner_nodes = corner_nodes[corner_order]
    source_to_corner = np.full((extraction.points.shape[0],), -1, dtype=np.int32)
    source_to_corner[corner_nodes] = np.arange(corner_nodes.size, dtype=np.int32)
    row_orders: dict[str, np.ndarray] = {}
    blocks = []
    next_cell_id = 0
    for rows in top:
        corners = source_to_corner[top_vertices[rows.block_name][:, : rows.corner_count]]
        keys = np.sort(corners, axis=1)
        row_order = np.lexsort(
            tuple(keys[:, column] for column in range(keys.shape[1] - 1, -1, -1))
        )
        row_orders[rows.block_name] = row_order
        blocks.append(
            CellBlock(
                rows.block_name,
                rows.cell_kind,
                corners[row_order],
                global_ids=np.arange(
                    next_cell_id,
                    next_cell_id + rows.tags.size,
                    dtype=np.int64,
                ),
            )
        )
        next_cell_id += rows.tags.size
    mesh_points = output_points[corner_nodes]
    vertex_ids = np.arange(corner_nodes.size, dtype=np.int64)
    mesh = CellMesh(
        mesh_points,
        tuple(blocks),
        vertex_global_ids=vertex_ids,
        numeric_version=plan.source.report.source_revision,
    )
    boundary_rows = (
        _element_rows(gmsh, 2, generation.geometry_order)
        if generation.dimension == 3
        else top
    )
    return _CanonicalMesh(
        mesh,
        mesh_points,
        vertex_ids,
        output_points,
        top_vertices,
        corner_nodes,
        source_to_corner,
        row_orders,
        boundary_rows,
        any(rows.cell_kind == "quadrilateral" for rows in boundary_rows),
    )


def _boundary_surface(
    plan: Any,
    generation: _Generation,
    extraction: _Extraction,
    canonical: _CanonicalMesh,
    /,
) -> SurfaceModel:
    report = plan.source.report
    boundary_triangles = []
    for rows in canonical.boundary_rows:
        corners = canonical.source_to_corner[
            _local_connectivity(
                extraction.node_tags, rows.vertices[:, : rows.corner_count]
            )
        ]
        if np.any(corners < 0):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Boundary corner is absent from the volume mesh.",
            )
        splits = ((0, 1, 2),) if rows.cell_kind == "triangle" else ((0, 1, 2), (0, 2, 3))
        for split in splits:
            boundary_triangles.append(corners[:, split])
    boundary_triangles = np.concatenate(boundary_triangles)
    boundary_keys = np.sort(boundary_triangles, axis=1)
    boundary_order = np.lexsort(
        tuple(
            boundary_keys[:, column]
            for column in range(boundary_keys.shape[1] - 1, -1, -1)
        )
    )
    boundary_metadata = SurfaceMetadata(
        source_id=report.source_id,
        source_revision=report.source_revision,
        coordinate_contract=plan.source.coordinate_contract,
        provenance=("gmsh-occ", plan.plan_id),
        cell_tags=("gmsh-occ-surface",) * boundary_triangles.shape[0],
    )
    return SurfaceModel.from_triangles(
        canonical.mesh_points,
        boundary_triangles[boundary_order],
        boundary_metadata,
        vertex_global_ids=canonical.vertex_ids,
        cell_global_ids=np.arange(boundary_triangles.shape[0], dtype=np.int64),
        numeric_version=report.source_revision,
        repair_orientation=True,
        orient_closed_outward=generation.dimension == 3,
    )


@dataclass(frozen=True, slots=True)
class _Organization:
    mesh: CellMesh
    boundary: SurfaceModel | None
    zones: tuple[MeshZone, ...]
    patches: tuple[MeshPatch, ...]
    associations: tuple[GeometryAssociation, ...]
    attributes: tuple[MeshAttribute, ...]
    region_count: int
    cell_solid_ids: np.ndarray | None
    layer_interfaces: _EvidenceSection


def _organize(
    gmsh: Any,
    plan: Any,
    generation: _Generation,
    extraction: _Extraction,
    canonical: _CanonicalMesh,
    /,
) -> _Organization:
    source = plan.source
    specification = plan.specification
    report = source.report
    dimension = generation.dimension
    mesh = canonical.mesh
    boundary = None
    if not generation.semantic_volume and specification.target.ambient_dimension == 3:
        boundary = _boundary_surface(plan, generation, extraction, canonical)
        if (
            dimension == 2
            and not canonical.mixed_boundary
            and not generation.semantic_surface
        ):
            mesh = boundary.mesh
    mesh = canonicalize_cell_mesh(mesh)
    layer_attributes = _layer_attributes(
        mesh,
        extraction.top,
        canonical.row_orders,
        # ty: ignore[invalid-argument-type]
        extraction.layer_audit,
    )
    if generation.semantic_volume:
        cad_entities = generation.cad_entities
        if cad_entities is None:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Semantic volume execution lost its resolved CAD entity map.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        cell_solid_ids = _canonical_cell_solid_ids(
            mesh,
            extraction.top,
            canonical.row_orders,
            # ty: ignore[invalid-argument-type]
            cad_entities,
        )
        surface_evidence = _semantic_surface_evidence(
            gmsh,
            source,
            mesh,
            canonical.boundary_rows,
            extraction.node_tags,
            canonical.source_to_corner,
            cell_solid_ids,
            # ty: ignore[invalid-argument-type]
            cad_entities,
            plan.plan_id,
        )
        layer_interface_achieved = _audit_swept_interfaces(
            mesh,
            source,
            cell_solid_ids,
            surface_evidence.mesh_face_source,
            # ty: ignore[invalid-argument-type]
            generation.sweep,
        )
        region_zones, patches = _region_evidence(
            source,
            mesh,
            specification,
            cell_solid_ids,
            surface_evidence.mesh_face_source,
        )
        cell_entity_set = mesh.entity_set(3)
        cell_association = GeometryAssociation(
            GeometryAssociationKind.BREP,
            report.source_id,
            report.source_revision,
            cell_entity_set.entity_set_id,
            cell_entity_set.entity_ids,
            tuple(
                f"{report.source_revision}:solid:{int(owner)}" for owner in cell_solid_ids
            ),
            np.zeros((cell_solid_ids.size,), dtype=np.float64),
            exact=True,
            source_dimensions=np.full((cell_solid_ids.size,), 3, dtype=np.int8),
            source_indices=np.asarray(cell_solid_ids, dtype=np.int64),
        )
        return _Organization(
            mesh,
            surface_evidence.boundary,
            (*surface_evidence.zones, *region_zones),
            patches,
            (surface_evidence.association, cell_association),
            (surface_evidence.attribute, *layer_attributes),
            len(region_zones),
            cell_solid_ids,
            _EvidenceSection((), layer_interface_achieved),
        )
    if generation.semantic_surface:
        cad_entities = generation.cad_entities
        if cad_entities is None:
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Semantic planar execution lost its resolved CAD entity map.",
                stage=MeshingStageKind.CANONICALIZATION.value,
            )
        planar_evidence = _planar_surface_evidence(
            gmsh,
            source,
            mesh,
            specification,
            extraction.top,
            canonical.row_orders,
            extraction.node_tags,
            canonical.source_to_corner,
            # ty: ignore[invalid-argument-type]
            cad_entities,
            generation.geometry_order,
        )
        return _Organization(
            mesh,
            boundary,
            planar_evidence.zones,
            planar_evidence.patches,
            planar_evidence.associations,
            (*planar_evidence.attributes, *layer_attributes),
            0,
            None,
            _EvidenceSection(),
        )
    if dimension == 3 and boundary is None:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Volume execution lost its generated boundary mesh.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    association, boundary_zones, provider_attribute = _boundary_association(
        source,
        # ty: ignore[unresolved-attribute]
        mesh if dimension == 2 else boundary.mesh,
        plan.options.association_tolerance_factor,
    )
    return _Organization(
        mesh,
        boundary,
        boundary_zones if dimension == 2 else (),
        (),
        (association,),
        (provider_attribute, *layer_attributes),
        0,
        None,
        _EvidenceSection(),
    )


def _mesh_edges(mesh: CellMesh, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if not isinstance(
        connectivity,
        (
            PolygonalConnectivity,
            TetrahedralConnectivity,
            HexahedralConnectivity,
            PolyhedralConnectivity,
        ),
    ):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh surface/volume conversion requires two- or three-dimensional connectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    return np.asarray(connectivity.edges, dtype=np.int32)


def _size_compliance(
    specification: Any,
    mesh: CellMesh,
    organization: _Organization,
    connectivity_edges: np.ndarray,
    minimum_edge: float,
    maximum_edge: float,
    semantic_surface: bool,
    semantic_volume: bool,
    size_field_count: int,
    /,
) -> _EvidenceSection:
    issues = []
    requested = [
        (
            "size_compliance_absolute_tolerance",
            specification.size_compliance.absolute_tolerance,
        ),
        (
            "size_compliance_relative_tolerance",
            specification.size_compliance.relative_tolerance,
        ),
    ]
    achieved = []
    required_patches = float(
        sum(control.required for control in specification.patch_controls)
    )
    if semantic_surface:
        requested.extend(
            (
                ("region_count", float(len(specification.region_controls))),
                ("required_patch_count", required_patches),
            )
        )
        achieved.extend(
            (
                ("region_count", float(len(organization.zones))),
                ("patch_count", float(len(organization.patches))),
            )
        )
    if semantic_volume:
        size_issues, local_requested, local_achieved = _semantic_size_compliance(
            mesh,
            specification,
            # ty: ignore[invalid-argument-type]
            organization.cell_solid_ids,
        )
        issues.extend(size_issues)
        requested.extend(local_requested)
        requested.extend(
            (
                ("region_count", float(len(specification.region_controls))),
                ("required_patch_count", required_patches),
            )
        )
        achieved.extend(local_achieved)
        achieved.extend(
            (
                ("region_count", float(organization.region_count)),
                ("patch_count", float(len(organization.patches))),
                ("size_field_count", float(size_field_count)),
            )
        )
        return _EvidenceSection(tuple(requested), tuple(achieved), tuple(issues))
    top_scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
    for control in specification.size_controls:
        if (
            isinstance(control, ProximitySizeControl)
            or control.scope.scope_id != top_scope.scope_id
        ):
            continue
        if isinstance(control, UniformSizeControl):
            size_issues, local_requested, local_achieved = _edge_size_evidence(
                control,
                connectivity_edges,
                np.asarray(mesh.coordinates, dtype=np.float64),
                specification,
            )
            issues.extend(size_issues)
            requested.extend(local_requested)
            achieved.extend(local_achieved)
            continue
        key = f"size:{control.control_id}"
        requested.append((f"{key}:normal_angle", control.normal_angle))
        optional_bounds = (
            ("minimum_size", control.minimum_size),
            ("maximum_size", control.maximum_size),
        )
        requested.extend(
            (f"{key}:{name}", value)
            for name, value in optional_bounds
            if value is not None
        )
        achieved.extend(
            (
                (f"{key}:minimum_edge", minimum_edge),
                (f"{key}:maximum_edge", maximum_edge),
            )
        )
        if control.strength is SizeControlStrength.HARD:
            policy = specification.size_compliance
            if control.minimum_size is not None:
                tolerance = policy.absolute_tolerance + (
                    policy.relative_tolerance * abs(control.minimum_size)
                )
                if minimum_edge < control.minimum_size - tolerance:
                    issues.append(f"minimum_size:{control.control_id}")
            if control.maximum_size is not None:
                tolerance = policy.absolute_tolerance + (
                    policy.relative_tolerance * abs(control.maximum_size)
                )
                if maximum_edge > control.maximum_size + tolerance:
                    issues.append(f"maximum_size:{control.control_id}")
    achieved.append(("size_field_count", float(size_field_count)))
    return _EvidenceSection(tuple(requested), tuple(achieved), tuple(issues))


def _audit_gmsh_mesh(
    specification: Any,
    specification_id: str,
    geometry: Any,
    organization: _Organization,
    requested_kinds: set[str],
    minimum_jacobian: float,
    semantic_surface: bool,
    semantic_volume: bool,
    size_field_count: int,
    sections: tuple[_EvidenceSection, ...],
    /,
) -> Any:
    mesh = organization.mesh
    quality_evaluation = evaluate_cell_quality(mesh, mesh.coordinates)
    audit = audit_cell_mesh(
        mesh,
        geometry,
        quality_evaluation,
        patches=organization.patches,
        associations=organization.associations,
        attributes=organization.attributes,
        zones=organization.zones,
        boundary=organization.boundary,
    )
    if not audit.passed:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=audit.quality.worst_cell_global_ids,
        )
    family_policy = specification.target.cell_families
    achieved_kinds = {block.cell_kind for block in mesh.blocks}
    connectivity_edges = _mesh_edges(mesh)
    edge_lengths = np.linalg.norm(
        np.asarray(mesh.coordinates)[connectivity_edges[:, 1]]
        - np.asarray(mesh.coordinates)[connectivity_edges[:, 0]],
        axis=1,
    )
    minimum_edge = float(np.min(edge_lengths))
    maximum_edge = float(np.max(edge_lengths))
    vertex_minimum = np.full((mesh.coordinates.shape[0],), np.inf)
    vertex_maximum = np.zeros((mesh.coordinates.shape[0],), dtype=np.float64)
    np.minimum.at(vertex_minimum, connectivity_edges[:, 0], edge_lengths)
    np.minimum.at(vertex_minimum, connectivity_edges[:, 1], edge_lengths)
    np.maximum.at(vertex_maximum, connectivity_edges[:, 0], edge_lengths)
    np.maximum.at(vertex_maximum, connectivity_edges[:, 1], edge_lengths)
    active = np.isfinite(vertex_minimum) & (vertex_minimum > 0.0)
    maximum_local_edge_ratio = float(
        np.max(vertex_maximum[active] / vertex_minimum[active], initial=1.0)
    )
    compliance_issues = []
    if (
        not set(family_policy.required) <= achieved_kinds
        or not achieved_kinds <= requested_kinds
        or (len(achieved_kinds) > 1 and not family_policy.allow_mixed)
    ):
        compliance_issues.append("cell_family")
    size = _size_compliance(
        specification,
        mesh,
        organization,
        connectivity_edges,
        minimum_edge,
        maximum_edge,
        semantic_surface,
        semantic_volume,
        size_field_count,
    )
    compliance_issues.extend(size.issues)
    compliance = MeshingComplianceReport(
        specification_id,
        issues=(
            *compliance_issues,
            *(issue for section in sections for issue in section.issues),
        ),
        requested=(
            *size.requested,
            *(entry for section in sections for entry in section.requested),
        ),
        achieved=(
            ("minimum_edge", minimum_edge),
            ("maximum_edge", maximum_edge),
            ("minimum_curved_jacobian_determinant", minimum_jacobian),
            ("maximum_local_edge_ratio", maximum_local_edge_ratio),
            *size.achieved,
            *(entry for section in sections for entry in section.achieved),
        ),
    )
    return audit, compliance


def _stages(
    plan: Any,
    generation: _Generation,
    extraction: _Extraction,
    organization: _Organization,
    background: BackgroundMetricControl | None,
    geometry_layout_id: str,
    audit: Any,
    compliance: Any,
    /,
) -> tuple[MeshingStageReport, ...]:
    specification = plan.specification
    report = plan.source.report
    mesh = organization.mesh
    options = plan.options
    semantic_control_ids = (
        *(control.control_id for control in specification.region_controls),
        *(control.control_id for control in specification.patch_controls),
    )
    band_control_ids = (
        ()
        if plan.planar_bands is None
        else tuple(control.control_id for control in plan.planar_bands.controls)
    )
    feature_ids = tuple(value.feature_id for value in specification.protected_features)
    size_field_ids = (
        *(value.control.control_id for value in generation.size_fields.proximity),
        *(() if background is None else (background.control_id,)),
    )
    optimized = options.optimize_netgen or (
        options.high_order_optimization is not GmshHighOrderOptimization.NONE
    )
    return (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(report.source_revision,),
            output_ids=(plan.support.source_descriptor_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SCOPE_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=(specification.specification_id,),
            output_ids=(
                (
                    specification.scope.scope_id
                    if isinstance(specification, SurfaceMeshingSpec)
                    else specification.boundary_scope.scope_id
                ),
            ),
        ),
        MeshingStageReport(
            MeshingStageKind.CONTROL_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=(
                *(control.control_id for control in specification.size_controls),
                *semantic_control_ids,
                *band_control_ids,
                *feature_ids,
            ),
            output_ids=(plan.plan_id,),
        ),
        *(
            (
                MeshingStageReport(
                    MeshingStageKind.SIZE_FIELD_RESOLUTION,
                    MeshingStageStatus.PASSED,
                    input_ids=size_field_ids,
                    output_ids=(plan.plan_id,),
                    created_count=len(generation.size_fields.field_ids),
                ),
            )
            if size_field_ids
            else ()
        ),
        *(
            (
                MeshingStageReport(
                    MeshingStageKind.LAYER_GENERATION,
                    MeshingStageStatus.PASSED,
                    input_ids=(
                        tuple(value.control_id for value in specification.layer_controls)
                        if generation.sweep is not None
                        or generation.layer_field is not None
                        else band_control_ids
                    ),
                    output_ids=(mesh.mesh_id,),
                    created_count=extraction.cell_count,
                ),
            )
            if generation.sweep is not None
            or generation.band_generation is not None
            or generation.layer_field is not None
            else ()
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING
            if generation.dimension == 2
            else MeshingStageKind.VOLUME_FILL,
            MeshingStageStatus.PASSED,
            input_ids=(plan.plan_id,),
            output_ids=(mesh.mesh_id,),
            created_count=extraction.cell_count,
        ),
        *(
            (
                MeshingStageReport(
                    MeshingStageKind.OPTIMIZATION,
                    MeshingStageStatus.PASSED,
                    input_ids=(options.options_id,),
                    output_ids=(mesh.mesh_id,),
                ),
            )
            if optimized
            else ()
        ),
        MeshingStageReport(
            MeshingStageKind.CANONICALIZATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(mesh.topology_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.GEOMETRY_ASSOCIATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=tuple(value.association_id for value in organization.associations),
        ),
        MeshingStageReport(
            MeshingStageKind.QUALITY_EVALUATION,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.mesh_id,),
            output_ids=(audit.quality.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.GEOMETRY_AUDIT,
            MeshingStageStatus.PASSED,
            input_ids=(geometry_layout_id,),
            output_ids=(audit.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.TOPOLOGY_AUDIT,
            MeshingStageStatus.PASSED,
            input_ids=(mesh.topology_id,),
            output_ids=(audit.report_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SPECIFICATION_COMPLIANCE,
            MeshingStageStatus.PASSED,
            input_ids=(specification.specification_id,),
            output_ids=(compliance.report_id,),
        ),
    )


def _conformity_section(curved: _CurvedGeometry, /) -> _EvidenceSection:
    if curved.conformity_residual is None:
        return _EvidenceSection()
    return _EvidenceSection(
        (), (("high_order_node_conformity_residual", curved.conformity_residual),)
    )


def _execute_brep(
    gmsh: Any,
    plan: Any,
    version: str,
    info: MeshingProviderInfo,
    cache: _CadImportCache,
    background: BackgroundMetricControl | None,
    /,
) -> CellMeshingResult:
    source = plan.source
    specification = plan.specification
    report = source.report
    generation = _prepare_generation(gmsh, plan, cache, background)
    _generate(gmsh, plan, generation)
    extraction = _extract(gmsh, plan, generation)
    canonical = _canonical_mesh(gmsh, plan, generation, extraction)
    organization = _organize(gmsh, plan, generation, extraction, canonical)
    mesh = organization.mesh
    embedding = (
        specification.planar_embedding
        if isinstance(specification, SurfaceMeshingSpec)
        and specification.target.ambient_dimension == 2
        else None
    )
    curved = _cell_geometry(
        gmsh,
        mesh,
        extraction.top,
        canonical.row_orders,
        canonical.top_vertices,
        canonical.corner_nodes,
        extraction.points,
        canonical.output_points,
        (lambda values: values) if embedding is None else embedding.to_planar,
        generation.geometry_order,
        generation.dimension == 2 and not canonical.mixed_boundary,
    )
    geometry = curved.geometry
    corner_points = extraction.points[canonical.corner_nodes]
    size_fields = generation.size_fields
    corner_edges = (
        _mesh_edges(mesh)
        if size_fields.proximity or size_fields.background is not None
        else np.empty((0, 2), dtype=np.int32)
    )
    metric_tolerance = (
        report.linear_deflection * plan.options.association_tolerance_factor
    )
    sections = (
        extraction.periodic,
        _EvidenceSection(
            # ty: ignore[unresolved-attribute]
            extraction.layer_audit.requested,
            # ty: ignore[unresolved-attribute]
            extraction.layer_audit.achieved,
        ),
        extraction.bands,
        extraction.layer_field,
        organization.layer_interfaces,
        _EvidenceSection((), extraction.native.achieved, extraction.native.issues),
        _conformity_section(curved),
        _audit_protected_features(
            gmsh,
            generation.protected,
            mesh,
            extraction.node_tags,
            extraction.points,
            canonical.source_to_corner,
            1.0e-9 * _source_scale(source),
        ),
        _proximity_evidence(
            gmsh,
            generation.size_fields.proximity,
            corner_edges,
            corner_points,
            specification.size_compliance,
        ),
        _background_metric_evidence(
            gmsh,
            generation.size_fields.background,
            corner_edges,
            corner_points,
            specification.size_compliance,
            metric_tolerance,
        ),
    )
    audit, compliance = _audit_gmsh_mesh(
        specification,
        specification.specification_id,
        geometry,
        organization,
        generation.requested_kinds,
        extraction.native.minimum_jacobian,
        generation.semantic_surface,
        generation.semantic_volume,
        len(generation.size_fields.field_ids),
        sections,
    )
    if not compliance.passed:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "; ".join(compliance.issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    trace = MeshingTrace(
        _stages(
            plan,
            generation,
            extraction,
            organization,
            background,
            geometry.geometry_layout_id,
            audit,
            compliance,
        )
    )
    runtime = MeshingRuntimeInfo(
        plan.support.provider_id,
        version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=plan.options.num_threads == 1,
        enforced_limits=("entities", "vertices", "cells"),
        unenforced_limits=("provider_workspace", "converted_arrays", "wall_time"),
    )
    provenance = SemanticProvenance(
        {
            "kind": "gmsh-cell-meshing-result",
            "source_revision": report.source_revision,
            "plan": plan.plan_id,
            "mesh": mesh.mesh_id,
            "associations": tuple(
                value.association_id for value in organization.associations
            ),
            "zones": tuple(value.zone_id for value in organization.zones),
            "patches": tuple(value.patch_id for value in organization.patches),
        },
        resource_ids={"source": report.source_id},
    )
    return CellMeshingResult(
        mesh,
        geometry,
        source.coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        info,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        boundary=organization.boundary,
        patches=organization.patches,
        zones=organization.zones,
        attributes=organization.attributes,
        associations=organization.associations,
    )
