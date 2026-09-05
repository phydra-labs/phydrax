#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import CellVertexGeometryElement
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._reference import FiniteElementSpec
from ..geometry.surface import SurfaceModel
from ._association import GeometryAssociation
from ._organization import (
    MeshAttribute,
    MeshLabel,
    MeshPatch,
    MeshZone,
    validate_mesh_labels,
    validate_mesh_zones,
)
from ._quality import (
    CellQualityEvaluation,
    CellQualityReport,
    evaluate_cell_quality,
    summarize_cell_quality,
)
from ._scope import MeshingEntityKind


class CellMeshAuditPolicy(StrictModule, NonTrainableState):
    require_all_vertices_used: bool = eqx.field(static=True)
    require_complete_association: bool = eqx.field(static=True)
    minimum_measure: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    maximum_connectivity_entries: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        require_all_vertices_used: bool = True,
        require_complete_association: bool = False,
        minimum_measure: float = 0.0,
        minimum_mean_ratio: float = 0.0,
        maximum_aspect_ratio: float = 1.0e300,
        maximum_connectivity_entries: int = 500_000_000,
    ):
        measure = float(minimum_measure)
        ratio = float(minimum_mean_ratio)
        aspect = float(maximum_aspect_ratio)
        entries = int(maximum_connectivity_entries)
        if not np.isfinite(measure) or measure < 0.0:
            raise ValueError("minimum_measure must be finite and non-negative.")
        if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError("minimum_mean_ratio must lie in [0, 1].")
        if np.isnan(aspect) or aspect < 1.0:
            raise ValueError("maximum_aspect_ratio must be at least one.")
        if entries <= 0:
            raise ValueError("maximum_connectivity_entries must be positive.")
        self.require_all_vertices_used = bool(require_all_vertices_used)
        self.require_complete_association = bool(require_complete_association)
        self.minimum_measure = measure
        self.minimum_mean_ratio = ratio
        self.maximum_aspect_ratio = aspect
        self.maximum_connectivity_entries = entries
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-audit-policy",
                "require_all_vertices_used": bool(require_all_vertices_used),
                "require_complete_association": bool(require_complete_association),
                "minimum_measure": measure,
                "minimum_mean_ratio": ratio,
                "maximum_aspect_ratio": aspect,
                "maximum_connectivity_entries": entries,
            }
        )


class CellMeshAuditReport(StrictModule, NonTrainableState):
    mesh_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    quality_scope: str = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    entity_counts: tuple[int, ...] = eqx.field(static=True)
    boundary_counts: tuple[int, ...] = eqx.field(static=True)
    connectivity_entries: int = eqx.field(static=True)
    unused_vertex_count: int = eqx.field(static=True)
    quality: CellQualityReport
    policy_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def require_passed(self, /) -> None:
        if not self.passed:
            raise ValueError("Cell mesh audit failed: " + "; ".join(self.issues))


def _connectivity_entries(mesh: CellMesh, /) -> int:
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolyhedralConnectivity):
        return int(
            connectivity.edges.size
            + connectivity.face_vertex_values.size
            + connectivity.face_edge_values.size
            + connectivity.cell_face_values.size
            + connectivity.cell_vertex_values.size
        )
    if isinstance(connectivity, IntervalConnectivity):
        return int(connectivity.cell_vertices.size)
    if isinstance(connectivity, PolygonalConnectivity):
        return int(
            connectivity.edges.size
            + connectivity.cell_vertices.size
            + connectivity.cell_edges.size
        )
    if isinstance(connectivity, TetrahedralConnectivity):
        return int(
            connectivity.edges.size
            + connectivity.faces.size
            + connectivity.face_edges.size
            + connectivity.cell_faces.size
        )
    if isinstance(connectivity, HexahedralConnectivity):
        return int(
            connectivity.edges.size
            + connectivity.faces.size
            + connectivity.face_edges.size
            + connectivity.cell_faces.size
        )
    raise TypeError("Unsupported CellMesh connectivity.")


def _geometry_id(geometry: CellGeometrySpec, /) -> str:
    return canonical_fingerprint(
        {
            "layout": geometry.geometry_layout_id,
            "coordinates": array_tree_fingerprint(np.asarray(geometry.coordinates)),
        }
    )


def _evidence_id(patches, zones, labels, attributes, associations, /) -> str:
    return canonical_fingerprint(
        {
            "patches": [value.patch_id for value in patches],
            "zones": [value.zone_id for value in zones],
            "labels": [value.label_id for value in labels],
            "attributes": [value.attribute_id for value in attributes],
            "associations": [value.association_id for value in associations],
        }
    )


def _geometry_binding(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[str, tuple[str, ...]]:
    elements, routes, coordinates = geometry.resolve(mesh)
    points = np.asarray(coordinates)
    if points.shape[1] != mesh.ambient_dimension:
        return "unsupported", ("geometry_ambient_dimension",)
    scope = "vertex_geometry"
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if isinstance(element, FiniteElementSpec):
            if element.degree != 1:
                scope = "corner_cells"
            basis, _ = element.tabulate(
                jnp.asarray(reference_cell_topology(block.cell_kind).vertices)
            )
            corners = np.einsum(
                "vi,cia->cva", np.asarray(basis), points[np.asarray(route)]
            )
        elif isinstance(element, CellVertexGeometryElement):
            corners = points[np.asarray(route)]
        else:
            return "unsupported", ("unsupported_geometry_element",)
        expected = np.asarray(mesh.coordinates)[np.asarray(block.vertices)]
        scale = max(float(np.max(np.abs(expected), initial=0.0)), np.finfo(float).tiny)
        if corners.shape != expected.shape or not np.allclose(
            corners, expected, rtol=0.0, atol=64.0 * np.finfo(float).eps * scale
        ):
            return scope, ("geometry_corner_binding",)
    return scope, ()


def _boundary_issues(mesh: CellMesh, boundary: SurfaceModel | None, /) -> tuple[str, ...]:
    if boundary is None:
        return ()
    if not isinstance(boundary, SurfaceModel):
        raise TypeError("boundary must be SurfaceModel or None.")
    if mesh.ambient_dimension != 3 or mesh.topological_dimension not in (2, 3):
        return ("boundary_dimension",)
    surface = boundary.mesh
    mesh_ids = np.asarray(mesh.vertex_global_ids)
    surface_ids = np.asarray(surface.vertex_global_ids)
    positions = {int(identifier): index for index, identifier in enumerate(mesh_ids)}
    if any(int(identifier) not in positions for identifier in surface_ids):
        return ("boundary_vertex_ids",)
    indices = np.asarray(
        [positions[int(identifier)] for identifier in surface_ids], dtype=np.int64
    )
    if not np.array_equal(
        np.asarray(surface.coordinates), np.asarray(mesh.coordinates)[indices]
    ):
        return ("boundary_coordinates",)
    if mesh.topological_dimension == 2:
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("Surface boundary audit requires polygonal connectivity.")
        loops = tuple(
            tuple(int(value) for value in mesh_ids[row])
            for block in mesh.blocks
            for row in np.asarray(block.vertices)
        )
    else:
        connectivity = mesh.connectivity
        if not isinstance(
            connectivity,
            (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
        ):
            raise TypeError("Volume boundary audit requires volume connectivity.")
        boundary_mask = np.asarray(connectivity.boundary_faces, dtype=bool)
        if isinstance(connectivity, PolyhedralConnectivity):
            offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
            rows = tuple(
                values[offsets[index] : offsets[index + 1]]
                for index, is_boundary in enumerate(boundary_mask)
                if is_boundary
            )
        else:
            rows = np.asarray(connectivity.faces)[boundary_mask]
        loops = tuple(tuple(int(value) for value in mesh_ids[row]) for row in rows)
    vertex_faces: dict[int, set[int]] = {}
    for index, loop in enumerate(loops):
        for vertex in loop:
            vertex_faces.setdefault(vertex, set()).add(index)
    triangles = [[] for _ in loops]
    for block in surface.blocks:
        for row in np.asarray(block.vertices):
            triangle = tuple(int(value) for value in surface_ids[row])
            owners = set.intersection(
                *(vertex_faces.get(vertex, set()) for vertex in triangle)
            )
            if len(owners) != 1:
                return ("boundary_topology",)
            triangles[owners.pop()].append(triangle)
    for loop, faces in zip(loops, triangles, strict=True):
        if len(faces) != len(loop) - 2:
            return ("boundary_coverage",)
        edges = Counter(
            tuple(sorted((face[index], face[(index + 1) % 3])))
            for face in faces
            for index in range(3)
        )
        perimeter = {
            tuple(sorted((loop[index], loop[(index + 1) % len(loop)])))
            for index in range(len(loop))
        }
        if any(edges[edge] != 1 for edge in perimeter) or any(
            count != (1 if edge in perimeter else 2) for edge, count in edges.items()
        ):
            return ("boundary_topology",)
    return ()


def _mesh_evidence_issues(
    mesh, boundary, patches, zones, labels, attributes, associations, /
):
    issues = list(_boundary_issues(mesh, boundary))
    meshes = (mesh,) if boundary is None else (mesh, boundary.mesh)
    bindings = {
        entities.entity_set_id: (owner, entities)
        for owner in meshes
        for entities in owner.topology.entity_sets
    }
    for value in (*patches, *zones, *labels, *attributes):
        scope = value.scope
        binding = bindings.get(scope.entity_set_id)
        if binding is None:
            issues.append("organization_entity_set")
            continue
        _, entities = binding
        owners = tuple(
            owner
            for owner in meshes
            if any(
                value.entity_set_id == scope.entity_set_id
                for value in owner.topology.entity_sets
            )
        )
        if (
            scope.entity_kind != MeshingEntityKind.MESH
            or scope.entity_dimension != entities.intrinsic_dimension
            or not any(
                scope.source_id == owner.mesh_id
                and scope.source_revision == owner.numeric_version
                for owner in owners
            )
        ):
            issues.append("organization_binding")
        if not np.all(
            np.isin(np.asarray(scope.entity_ids), np.asarray(entities.entity_ids))
        ):
            issues.append("organization_entity_ids")
    resolved_sources = {}
    for association in associations:
        binding = bindings.get(association.target_entity_set_id)
        if binding is None:
            issues.append("association_entity_set")
            continue
        try:
            association.validate_target(binding[1])
        except ValueError:
            issues.append("association_target_ids")
        for identifier, source, resolved in zip(
            np.asarray(association.target_global_ids),
            association.source_entity_ids,
            np.asarray(association.resolved),
            strict=True,
        ):
            if not resolved:
                continue
            key = (
                association.association_kind,
                association.source_id,
                association.source_revision,
                association.target_entity_set_id,
                int(identifier),
            )
            previous = resolved_sources.setdefault(key, source)
            if previous != source:
                issues.append("conflicting_geometry_association")
    return tuple(dict.fromkeys(issues))


def _complete_association_coverage(mesh, boundary, associations, /) -> bool:
    if not associations or any(not value.complete for value in associations):
        return False
    meshes = (mesh,) if boundary is None else (mesh, boundary.mesh)
    bindings = {
        entities.entity_set_id: (owner, entities)
        for owner in meshes
        for entities in owner.topology.entity_sets
    }
    coverage: dict[tuple[str, str, str, str], set[int]] = {}
    for association in associations:
        key = (
            association.association_kind.value,
            association.source_id,
            association.source_revision,
            association.target_entity_set_id,
        )
        coverage.setdefault(key, set()).update(
            int(value) for value in np.asarray(association.target_global_ids)
        )
    for key, identifiers in coverage.items():
        binding = bindings.get(key[-1])
        if binding is None:
            return False
        owner, entities = binding
        required = np.asarray(entities.entity_ids)
        if owner.topological_dimension == 3 and entities.intrinsic_dimension < 3:
            required = required[np.asarray(entities.subset("boundary").mask)]
        if not set(int(value) for value in required) <= identifiers:
            return False
    return True


def audit_cell_mesh(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    quality: CellQualityEvaluation,
    /,
    *,
    policy: CellMeshAuditPolicy | None = None,
    boundary: SurfaceModel | None = None,
    patches: tuple[MeshPatch, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
) -> CellMeshAuditReport:
    """Audit exact bindings and vertex-cell quality, not high-order Jacobian extrema.

    Association residuals are structurally validated by GeometryAssociation; source-
    specific residual tolerances remain the generating provider's responsibility.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    if not isinstance(quality, CellQualityEvaluation):
        raise TypeError("quality must be CellQualityEvaluation.")
    audit_policy = CellMeshAuditPolicy() if policy is None else policy
    if not isinstance(audit_policy, CellMeshAuditPolicy):
        raise TypeError("policy must be CellMeshAuditPolicy or None.")
    quality_scope, geometry_issues = _geometry_binding(mesh, geometry)
    validate_mesh_zones(tuple(zones))
    validate_mesh_labels(tuple(labels))
    if not all(isinstance(value, MeshPatch) for value in patches):
        raise TypeError("patches must contain MeshPatch values.")
    if not all(isinstance(value, GeometryAssociation) for value in associations):
        raise TypeError("associations must contain GeometryAssociation values.")
    if not all(isinstance(value, MeshAttribute) for value in attributes):
        raise TypeError("attributes must contain MeshAttribute values.")

    issues = list(geometry_issues)
    issues.extend(
        _mesh_evidence_issues(
            mesh,
            boundary,
            patches,
            zones,
            labels,
            attributes,
            associations,
        )
    )
    coordinates = np.asarray(geometry.coordinates, dtype=float)
    if not np.all(np.isfinite(coordinates)):
        issues.append("nonfinite_geometry")
    used = np.zeros((mesh.coordinates.shape[0],), dtype=bool)
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int32)
        valid = np.asarray(block.vertex_valid, dtype=bool)
        used[np.unique(vertices[valid])] = True
    unused = int(np.count_nonzero(~used))
    if audit_policy.require_all_vertices_used and unused:
        issues.append("unused_vertices")
    entries = _connectivity_entries(mesh)
    if entries > audit_policy.maximum_connectivity_entries:
        issues.append("connectivity_capacity_exceeded")
    expected_quality = evaluate_cell_quality(mesh)
    if (
        quality.topology_id != mesh.topology_id
        or quality.block_names != expected_quality.block_names
        or quality.block_offsets != expected_quality.block_offsets
        or not all(
            np.array_equal(supplied, expected)
            for supplied, expected in (
                (
                    np.asarray(quality.cell_global_ids),
                    np.asarray(expected_quality.cell_global_ids),
                ),
                (np.asarray(quality.measures), np.asarray(expected_quality.measures)),
                (
                    np.asarray(quality.mean_ratios),
                    np.asarray(expected_quality.mean_ratios),
                ),
                (
                    np.asarray(quality.aspect_ratios),
                    np.asarray(expected_quality.aspect_ratios),
                ),
                (np.asarray(quality.valid), np.asarray(expected_quality.valid)),
            )
        )
    ):
        issues.append("quality_binding")
    quality_report = summarize_cell_quality(quality)
    if quality_report.invalid_count:
        issues.append("invalid_cells")
    if quality_report.minimum_measure <= audit_policy.minimum_measure:
        issues.append("minimum_measure")
    if quality_report.minimum_mean_ratio < audit_policy.minimum_mean_ratio:
        issues.append("minimum_mean_ratio")
    if quality_report.maximum_aspect_ratio > audit_policy.maximum_aspect_ratio:
        issues.append("maximum_aspect_ratio")
    if audit_policy.require_complete_association and not _complete_association_coverage(
        mesh, boundary, associations
    ):
        issues.append("incomplete_geometry_association")

    entity_counts = tuple(entities.count for entities in mesh.topology.entity_sets)
    boundary_counts = tuple(
        int(np.count_nonzero(np.asarray(entities.subset("boundary").mask)))
        if "boundary" in {subset.name for subset in entities.subsets}
        else 0
        for entities in mesh.topology.entity_sets
    )
    normalized_issues = tuple(dict.fromkeys(issues))
    geometry_id = _geometry_id(geometry)
    evidence_id = _evidence_id(patches, zones, labels, attributes, associations)
    report = CellMeshAuditReport(
        mesh_id=mesh.mesh_id,
        geometry_layout_id=geometry.geometry_layout_id,
        geometry_id=geometry_id,
        evidence_id=evidence_id,
        quality_scope=quality_scope,
        passed=not normalized_issues,
        issues=normalized_issues,
        vertex_count=int(mesh.coordinates.shape[0]),
        entity_counts=entity_counts,
        boundary_counts=boundary_counts,
        connectivity_entries=entries,
        unused_vertex_count=unused,
        quality=quality_report,
        policy_id=audit_policy.policy_id,
        report_id=canonical_fingerprint(
            {
                "kind": "cell-mesh-audit-report",
                "mesh": mesh.mesh_id,
                "geometry_layout": geometry.geometry_layout_id,
                "geometry": geometry_id,
                "evidence": evidence_id,
                "quality_scope": quality_scope,
                "quality": quality_report.report_id,
                "issues": normalized_issues,
                "entity_counts": entity_counts,
                "boundary_counts": boundary_counts,
                "connectivity_entries": entries,
                "policy": audit_policy.policy_id,
            }
        ),
    )
    return report


__all__ = ["CellMeshAuditPolicy", "CellMeshAuditReport", "audit_cell_mesh"]
