#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh discrete-surface classification, reparametrization, and remeshing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..._identity import SemanticProvenance
from ..._physical import SpatialCoordinateContract
from ...discretization import CellMesh
from ...geometry.simplicial import TriangleMesh
from ...geometry.surface import SurfaceAuditPolicy, SurfaceMetadata, SurfaceModel
from .._association import GeometryAssociation, GeometryAssociationKind
from .._canonical import canonicalize_cell_mesh
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
)
from .._controls import BackgroundMetricControl
from .._organization import MeshAttribute, MeshAttributeRole
from .._result import CellMeshingResult, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from ._gmsh_elements import (
    _cell_geometry,
    _element_rows,
    _local_connectivity,
    _native_quality,
)
from ._gmsh_evidence import _EvidenceSection
from ._gmsh_execute import (
    _audit_gmsh_mesh,
    _conformity_section,
    _mesh_edges,
    _Organization,
)
from ._gmsh_options import GmshHighOrderOptimization
from ._gmsh_sizing import _background_metric_evidence, _background_view, _size_values


def _surface_source(
    source: SurfaceModel | CellMesh,
    coordinate_contract: SpatialCoordinateContract | None,
    /,
) -> SurfaceModel:
    """Bind a discrete remeshing source to its explicit coordinate contract."""
    if isinstance(source, SurfaceModel):
        if coordinate_contract is not None:
            raise ValueError(
                "SurfaceModel sources already own their coordinate contract."
            )
        return source
    if not isinstance(source, CellMesh):
        raise TypeError("Remeshing sources must be SurfaceModel or CellMesh values.")
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("CellMesh remeshing sources require a SpatialCoordinateContract.")
    return SurfaceModel(
        source,
        SurfaceMetadata(
            source_id=source.mesh_id,
            source_revision=source.numeric_version,
            coordinate_contract=coordinate_contract,
            provenance=("cell-mesh", source.mesh_id),
        ),
    )


def _surface_triangles(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    )


def _euler_characteristic(mesh: CellMesh, /) -> int:
    connectivity = mesh.connectivity
    return (
        mesh.coordinates.shape[0]
        # ty: ignore[unresolved-attribute]
        - np.asarray(connectivity.edges).shape[0]
        + _surface_triangles(mesh).shape[0]
    )


def _configure_remeshing(gmsh: Any, plan: Any, /) -> float | None:
    surface = plan.specification.surface
    options = plan.options
    minimum, target, maximum, curvature_points = _size_values(surface)
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1 if options.terminal_output else 0)
    gmsh.option.setNumber("General.NumThreads", options.num_threads)
    gmsh.option.setNumber("Mesh.Algorithm", options.algorithm_2d.gmsh_code)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0 if minimum is None else minimum)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0e22 if maximum is None else maximum)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_points)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1 if target is not None else 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", surface.target.geometry_order)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber(
        "Mesh.HighOrderOptimize", options.high_order_optimization.gmsh_code
    )
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    gmsh.option.setNumber("Mesh.AnisoMax", 1.0e33)
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 0)
    gmsh.model.add(f"phydrax-{plan.plan_id[:12]}")
    return target


def _reconstruct(gmsh: Any, plan: Any, /) -> int:
    """Classify sharp features, then build reparametrizable discrete geometry."""
    source = plan.source
    reconstruction = plan.reconstruction
    points = np.asarray(source.mesh.coordinates, dtype=np.float64)
    triangles = _surface_triangles(source.mesh)
    entity = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(
        2,
        entity,
        np.arange(1, points.shape[0] + 1, dtype=np.int64),
        points.reshape(-1),
    )
    gmsh.model.mesh.addElementsByType(
        entity,
        gmsh.model.mesh.getElementType("Triangle", 1),
        [],
        (triangles + 1).reshape(-1),
    )
    gmsh.model.mesh.classifySurfaces(
        reconstruction.feature_angle,
        True,
        reconstruction.force_parametrizable_patches,
        reconstruction.curve_angle,
    )
    gmsh.model.mesh.createGeometry()
    patches = gmsh.model.getEntities(2)
    if not patches:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Gmsh surface classification produced no reparametrized patches.",
            stage=MeshingStageKind.FEATURE_DISCOVERY.value,
        )
    return len(patches)


def _apply_remeshing_sizes(
    gmsh: Any,
    target: float | None,
    background: BackgroundMetricControl | None,
    /,
) -> Any:
    fields = []
    if target is not None:
        point_entities = gmsh.model.getEntities(0)
        if point_entities:
            gmsh.model.mesh.setSize(point_entities, target)
        constant = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(constant, "F", f"{target:.17g}")
        fields.append(constant)
    view = None if background is None else _background_view(gmsh, background)
    if view is not None:
        fields.append(view.field)
    if fields:
        active = fields[0]
        if len(fields) > 1:
            active = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(active, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(active)
    return tuple(fields), view


def _nearest_source_cells(
    source: SurfaceModel, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    query_mesh = TriangleMesh(
        np.asarray(source.mesh.coordinates, dtype=np.float64),
        _surface_triangles(source.mesh),
        source_id=f"{source.mesh.mesh_id}:remesh-query",
    )
    query = query_mesh.query_index().query(jnp.asarray(points))
    return (
        np.asarray(query.face_index, dtype=np.int64),
        np.asarray(query.distance, dtype=np.float64),
    )


def _oriented_surface(
    plan: Any,
    mesh_points: np.ndarray,
    triangles: np.ndarray,
    closed: bool,
    /,
) -> tuple[SurfaceModel, float]:
    """Orient the remeshed surface like its source; report the agreement fraction."""
    source = plan.source
    metadata = SurfaceMetadata(
        source_id=source.metadata.source_id,
        source_revision=source.metadata.source_revision,
        coordinate_contract=source.metadata.coordinate_contract,
        provenance=("gmsh-discrete-remesh", plan.plan_id),
    )

    def build(values: np.ndarray, /) -> SurfaceModel:
        return SurfaceModel.from_triangles(
            mesh_points,
            values,
            metadata,
            vertex_global_ids=np.arange(mesh_points.shape[0], dtype=np.int64),
            cell_global_ids=np.arange(values.shape[0], dtype=np.int64),
            numeric_version=source.mesh.numeric_version,
            repair_orientation=True,
            orient_closed_outward=closed,
        )

    def agreement(model: SurfaceModel, /) -> float:
        faces = _surface_triangles(model.mesh)
        corners = mesh_points[faces]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        nearest, _ = _nearest_source_cells(source, np.mean(corners, axis=1))
        source_points = np.asarray(source.mesh.coordinates, dtype=np.float64)
        source_faces = _surface_triangles(source.mesh)[nearest]
        source_corners = source_points[source_faces]
        source_normals = np.cross(
            source_corners[:, 1] - source_corners[:, 0],
            source_corners[:, 2] - source_corners[:, 0],
        )
        return float(np.mean(np.sum(normals * source_normals, axis=1) > 0.0))

    surface = build(triangles)
    fraction = agreement(surface)
    if not closed and fraction < 0.5:
        # Swap the last two corners: an orientation reversal the curved-map router
        # recognizes as a reference reflection.
        surface = build(triangles[:, (0, 2, 1)])
        fraction = agreement(surface)
    return surface, fraction


@dataclass(frozen=True, slots=True)
class _RemeshedSurface:
    surface: SurfaceModel
    mesh: CellMesh
    orientation_agreement: float
    rows: tuple
    row_orders: dict[str, np.ndarray]
    top_vertices: dict[str, np.ndarray]
    corner_nodes: np.ndarray
    source_to_corner: np.ndarray


def _canonical_surface(
    plan: Any, node_tags: Any, points: Any, top: Any, closed: bool, /
) -> _RemeshedSurface:
    top_vertices = {
        rows.block_name: _local_connectivity(node_tags, rows.vertices) for rows in top
    }
    corner_nodes = np.unique(
        np.concatenate(
            [
                top_vertices[rows.block_name][:, : rows.corner_count].reshape(-1)
                for rows in top
            ]
        )
    )
    corner_points = points[corner_nodes]
    corner_order = np.lexsort(
        tuple(corner_points[:, column] for column in range(2, -1, -1))
    )
    corner_nodes = corner_nodes[corner_order]
    source_to_corner = np.full((points.shape[0],), -1, dtype=np.int32)
    source_to_corner[corner_nodes] = np.arange(corner_nodes.size, dtype=np.int32)
    (rows,) = top
    corners = source_to_corner[top_vertices[rows.block_name][:, :3]]
    keys = np.sort(corners, axis=1)
    row_order = np.lexsort(tuple(keys[:, column] for column in range(2, -1, -1)))
    surface, agreement = _oriented_surface(
        plan, points[corner_nodes], corners[row_order], closed
    )
    return _RemeshedSurface(
        surface,
        canonicalize_cell_mesh(surface.mesh),
        agreement,
        top,
        {rows.block_name: row_order},
        top_vertices,
        corner_nodes,
        source_to_corner,
    )


def _fidelity_section(
    plan: Any, points: np.ndarray, remeshed: _RemeshedSurface, /
) -> tuple[_EvidenceSection, np.ndarray, np.ndarray]:
    """Two-sided vertex deviation and topological invariants against the source."""
    source = plan.source
    reconstruction = plan.reconstruction
    _, forward = _nearest_source_cells(source, points)
    remesh_query = TriangleMesh(
        np.asarray(remeshed.mesh.coordinates, dtype=np.float64),
        _surface_triangles(remeshed.mesh),
        source_id=f"{remeshed.mesh.mesh_id}:remesh-fidelity",
    ).query_index()
    backward = np.asarray(
        remesh_query.query(jnp.asarray(source.mesh.coordinates)).distance,
        dtype=np.float64,
    )
    cell_points = np.asarray(remeshed.mesh.coordinates, dtype=np.float64)[
        _surface_triangles(remeshed.mesh)
    ]
    nearest, residuals = _nearest_source_cells(source, np.mean(cell_points, axis=1))
    source_audit = source.audit()
    target_audit = remeshed.surface.audit()
    source_euler = _euler_characteristic(source.mesh)
    target_euler = _euler_characteristic(remeshed.mesh)
    deviation = max(float(np.max(forward)), float(np.max(backward)))
    issues = []
    if deviation > reconstruction.maximum_deviation:
        issues.append("surface_deviation")
    if (
        source_euler != target_euler
        or source_audit.component_count != target_audit.component_count
        or source_audit.boundary_loop_count != target_audit.boundary_loop_count
    ):
        issues.append("surface_topology")
    section = _EvidenceSection(
        (
            ("surface_maximum_deviation", reconstruction.maximum_deviation),
            ("surface_euler_characteristic", float(source_euler)),
            ("surface_component_count", float(source_audit.component_count)),
            ("surface_boundary_loop_count", float(source_audit.boundary_loop_count)),
        ),
        (
            ("surface_maximum_remesh_to_source_distance", float(np.max(forward))),
            ("surface_maximum_source_to_remesh_distance", float(np.max(backward))),
            ("surface_euler_characteristic", float(target_euler)),
            ("surface_component_count", float(target_audit.component_count)),
            ("surface_boundary_loop_count", float(target_audit.boundary_loop_count)),
            ("surface_orientation_agreement", remeshed.orientation_agreement),
        ),
        tuple(issues),
    )
    return section, nearest, residuals


def _organization(
    plan: Any, remeshed: _RemeshedSurface, nearest: np.ndarray, residuals: np.ndarray, /
) -> _Organization:
    source = plan.source
    mesh = remeshed.mesh
    cell_set = mesh.entity_set(2)
    source_cells = np.asarray(source.mesh.entity_set(2).entity_ids, dtype=np.int64)[
        nearest
    ]
    association = GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        source.metadata.source_id,
        source.metadata.source_revision,
        cell_set.entity_set_id,
        cell_set.entity_ids,
        tuple(f"{source.mesh.mesh_id}:cell:{int(value)}" for value in source_cells),
        residuals,
        resolved=residuals <= plan.reconstruction.maximum_deviation,
        exact=False,
    )
    if not association.complete:
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "Remeshed cells could not be matched to the source surface within tolerance.",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            entity_ids=tuple(
                np.asarray(cell_set.entity_ids)[
                    residuals > plan.reconstruction.maximum_deviation
                ]
            ),
        )
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        2,
        cell_set.entity_set_id,
        cell_set.entity_ids,
    )
    attribute = MeshAttribute(
        "source_surface_cell",
        MeshAttributeRole.GEOMETRY_CLASSIFICATION,
        scope,
        source_cells,
    )
    return _Organization(
        mesh,
        remeshed.surface,
        (),
        (),
        (association,),
        (attribute,),
        0,
        None,
        _EvidenceSection(),
    )


def _remesh_stages(
    plan: Any,
    patch_count: int,
    cell_count: int,
    organization: _Organization,
    background: BackgroundMetricControl | None,
    geometry_layout_id: str,
    audit: Any,
    compliance: Any,
    /,
) -> tuple[MeshingStageReport, ...]:
    specification = plan.specification
    surface = specification.surface
    mesh = organization.mesh
    options = plan.options
    optimized = options.high_order_optimization is not GmshHighOrderOptimization.NONE
    return (
        MeshingStageReport(
            MeshingStageKind.SOURCE_INSPECTION,
            MeshingStageStatus.PASSED,
            input_ids=(plan.source.mesh.mesh_id,),
            output_ids=(plan.support.source_descriptor_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.SCOPE_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=(specification.specification_id,),
            output_ids=(surface.scope.scope_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.CONTROL_RESOLUTION,
            MeshingStageStatus.PASSED,
            input_ids=(
                *(control.control_id for control in surface.size_controls),
                plan.reconstruction.control_id,
            ),
            output_ids=(plan.plan_id,),
        ),
        MeshingStageReport(
            MeshingStageKind.FEATURE_DISCOVERY,
            MeshingStageStatus.PASSED,
            input_ids=(plan.source.mesh.mesh_id, plan.reconstruction.control_id),
            output_ids=(plan.plan_id,),
            created_count=patch_count,
        ),
        *(
            (
                MeshingStageReport(
                    MeshingStageKind.SIZE_FIELD_RESOLUTION,
                    MeshingStageStatus.PASSED,
                    input_ids=(background.control_id,),
                    output_ids=(plan.plan_id,),
                ),
            )
            if background is not None
            else ()
        ),
        MeshingStageReport(
            MeshingStageKind.SURFACE_MESHING,
            MeshingStageStatus.PASSED,
            input_ids=(plan.plan_id,),
            output_ids=(mesh.mesh_id,),
            created_count=cell_count,
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


def _execute_remesh(
    gmsh: Any,
    plan: Any,
    version: str,
    info: MeshingProviderInfo,
    /,
) -> CellMeshingResult:
    source = plan.source
    surface = plan.specification.surface
    limits = surface.limits
    background = plan.background_metric
    geometry_order = surface.target.geometry_order
    closed = bool(source.audit().closed)
    target = _configure_remeshing(gmsh, plan)
    patch_count = _reconstruct(gmsh, plan)
    size_field_ids, view = _apply_remeshing_sizes(gmsh, target, background)
    gmsh.model.mesh.generate(2)
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
    top = _element_rows(gmsh, 2, geometry_order)
    cell_count = sum(rows.tags.size for rows in top)
    if cell_count > limits.maximum_cells:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Generated Gmsh mesh exceeds maximum_cells.",
        )
    if len(top) != 1 or top[0].cell_kind != "triangle":
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Gmsh discrete remeshing returned elements other than one triangle family.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    native = _native_quality(gmsh, top)
    remeshed = _canonical_surface(plan, node_tags, points, top, closed)
    surface_audit = remeshed.surface.audit(
        SurfaceAuditPolicy(require_closed=closed, require_outward_orientation=closed)
    )
    if not surface_audit.valid:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(surface_audit.issues) or "Remeshed surface audit failed.",
            stage=MeshingStageKind.TOPOLOGY_AUDIT.value,
        )
    curved = _cell_geometry(
        gmsh,
        remeshed.mesh,
        top,
        remeshed.row_orders,
        remeshed.top_vertices,
        remeshed.corner_nodes,
        points,
        points,
        lambda values: values,
        geometry_order,
        True,
    )
    fidelity, nearest, residuals = _fidelity_section(plan, points, remeshed)
    organization = _organization(plan, remeshed, nearest, residuals)
    corner_points = points[remeshed.corner_nodes]
    sections = (
        _EvidenceSection((), native.achieved, native.issues),
        _conformity_section(curved),
        fidelity,
        _background_metric_evidence(
            gmsh,
            view,
            _mesh_edges(remeshed.mesh) if view is not None else np.empty((0, 2)),
            corner_points,
            surface.size_compliance,
            plan.reconstruction.maximum_deviation,
        ),
    )
    audit, compliance = _audit_gmsh_mesh(
        surface,
        plan.specification.specification_id,
        curved.geometry,
        organization,
        {"triangle"},
        native.minimum_jacobian,
        False,
        False,
        len(size_field_ids),
        sections,
    )
    if not compliance.passed:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "; ".join(compliance.issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    mesh = organization.mesh
    trace = MeshingTrace(
        _remesh_stages(
            plan,
            patch_count,
            cell_count,
            organization,
            background,
            curved.geometry.geometry_layout_id,
            audit,
            compliance,
        )
    )
    runtime = MeshingRuntimeInfo(
        plan.support.provider_id,
        version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=plan.options.num_threads == 1,
        enforced_limits=("vertices", "cells"),
        unenforced_limits=("provider_workspace", "converted_arrays", "wall_time"),
    )
    provenance = SemanticProvenance(
        {
            "kind": "gmsh-surface-remeshing-result",
            "source_mesh": source.mesh.mesh_id,
            "plan": plan.plan_id,
            "mesh": mesh.mesh_id,
            "associations": tuple(
                value.association_id for value in organization.associations
            ),
        },
        resource_ids={"source": source.metadata.source_id},
    )
    return CellMeshingResult(
        mesh,
        curved.geometry,
        source.metadata.coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        info,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        boundary=organization.boundary,
        attributes=organization.attributes,
        associations=organization.associations,
    )
