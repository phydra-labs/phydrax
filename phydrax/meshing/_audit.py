#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellVertexGeometryElement,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._reference import FiniteElementSpec
from ..geometry.surface import SurfaceModel
from ._association import GeometryAssociation
from ._audit_topology import audit_welded_topology
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


class CellMeshAuditDisposition(StrEnum):
    """How one audit check affects the verdict.

    REJECT turns findings into failing issues, RECORD reports them without
    failing, and SKIP does not evaluate the check.
    """

    REJECT = "reject"
    RECORD = "record"
    SKIP = "skip"


def _disposition(
    value: CellMeshAuditDisposition, name: str, /
) -> CellMeshAuditDisposition:
    if not isinstance(value, CellMeshAuditDisposition):
        raise TypeError(f"{name} must be CellMeshAuditDisposition.")
    return value


class CellMeshAuditPolicy(StrictModule, NonTrainableState):
    """Dispositions and limits of the cell-mesh audit.

    Topology checks run on the welded complex in which vertices closer than
    ``coincident_vertex_tolerance`` times the bounding-box diagonal are merged.
    ``maximum_coincidence_candidates`` bounds welding and
    ``maximum_intersection_candidates`` bounds the streamed polygon-edge and
    triangle-pair broad phases. Exhaustion is unresolved; ``unresolved`` decides
    whether such capacities or predicates are rejected or recorded (it cannot
    be SKIP).
    """

    unused_entities: CellMeshAuditDisposition = eqx.field(static=True)
    require_complete_association: bool = eqx.field(static=True)
    minimum_measure: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    maximum_connectivity_entries: int = eqx.field(static=True)
    coincident_vertices: CellMeshAuditDisposition = eqx.field(static=True)
    coincident_vertex_tolerance: float = eqx.field(static=True)
    maximum_coincidence_candidates: int = eqx.field(static=True)
    maximum_intersection_candidates: int = eqx.field(static=True)
    duplicate_cells: CellMeshAuditDisposition = eqx.field(static=True)
    nonmanifold: CellMeshAuditDisposition = eqx.field(static=True)
    inconsistent_orientation: CellMeshAuditDisposition = eqx.field(static=True)
    watertight_boundary: CellMeshAuditDisposition = eqx.field(static=True)
    self_intersection: CellMeshAuditDisposition = eqx.field(static=True)
    invalid_geometry: CellMeshAuditDisposition = eqx.field(static=True)
    unresolved: CellMeshAuditDisposition = eqx.field(static=True)
    validity_policy: CellValidityPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        unused_entities: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        require_complete_association: bool = False,
        minimum_measure: float = 0.0,
        minimum_mean_ratio: float = 0.0,
        maximum_aspect_ratio: float = 1.0e300,
        maximum_connectivity_entries: int = 500_000_000,
        coincident_vertices: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        coincident_vertex_tolerance: float = 1.0e-12,
        maximum_coincidence_candidates: int = 50_000_000,
        maximum_intersection_candidates: int = 5_000_000,
        duplicate_cells: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        nonmanifold: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        inconsistent_orientation: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        watertight_boundary: CellMeshAuditDisposition = CellMeshAuditDisposition.SKIP,
        self_intersection: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        invalid_geometry: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        unresolved: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        validity_policy: CellValidityPolicy | None = None,
    ) -> None:
        measure = float(minimum_measure)
        ratio = float(minimum_mean_ratio)
        aspect = float(maximum_aspect_ratio)
        entries = int(maximum_connectivity_entries)
        tolerance = float(coincident_vertex_tolerance)
        coincidence_candidates = int(maximum_coincidence_candidates)
        intersection_candidates = int(maximum_intersection_candidates)
        dispositions = {
            name: _disposition(value, name)
            for name, value in (
                ("unused_entities", unused_entities),
                ("coincident_vertices", coincident_vertices),
                ("duplicate_cells", duplicate_cells),
                ("nonmanifold", nonmanifold),
                ("inconsistent_orientation", inconsistent_orientation),
                ("watertight_boundary", watertight_boundary),
                ("self_intersection", self_intersection),
                ("invalid_geometry", invalid_geometry),
                ("unresolved", unresolved),
            )
        }
        validity = CellValidityPolicy() if validity_policy is None else validity_policy
        if not np.isfinite(measure) or measure < 0.0:
            raise ValueError("minimum_measure must be finite and non-negative.")
        if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError("minimum_mean_ratio must lie in [0, 1].")
        if np.isnan(aspect) or aspect < 1.0:
            raise ValueError("maximum_aspect_ratio must be at least one.")
        if entries <= 0:
            raise ValueError("maximum_connectivity_entries must be positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
            raise ValueError("coincident_vertex_tolerance must lie in [0, 1).")
        if coincidence_candidates <= 0:
            raise ValueError("maximum_coincidence_candidates must be positive.")
        if intersection_candidates <= 0:
            raise ValueError("maximum_intersection_candidates must be positive.")
        if unresolved == CellMeshAuditDisposition.SKIP:
            raise ValueError("Unresolved checks must be rejected or recorded.")
        if not isinstance(validity, CellValidityPolicy):
            raise TypeError("validity_policy must be CellValidityPolicy or None.")
        self.unused_entities = unused_entities
        self.require_complete_association = bool(require_complete_association)
        self.minimum_measure = measure
        self.minimum_mean_ratio = ratio
        self.maximum_aspect_ratio = aspect
        self.maximum_connectivity_entries = entries
        self.coincident_vertices = coincident_vertices
        self.coincident_vertex_tolerance = tolerance
        self.maximum_coincidence_candidates = coincidence_candidates
        self.maximum_intersection_candidates = intersection_candidates
        self.duplicate_cells = duplicate_cells
        self.nonmanifold = nonmanifold
        self.inconsistent_orientation = inconsistent_orientation
        self.watertight_boundary = watertight_boundary
        self.self_intersection = self_intersection
        self.invalid_geometry = invalid_geometry
        self.unresolved = unresolved
        self.validity_policy = validity
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-audit-policy",
                "dispositions": {
                    name: value.value for name, value in dispositions.items()
                },
                "require_complete_association": bool(require_complete_association),
                "minimum_measure": measure,
                "minimum_mean_ratio": ratio,
                "maximum_aspect_ratio": aspect,
                "maximum_connectivity_entries": entries,
                "coincident_vertex_tolerance": tolerance,
                "maximum_coincidence_candidates": coincidence_candidates,
                "maximum_intersection_candidates": intersection_candidates,
                "validity_policy": validity.policy_id,
            }
        )


class CellMeshAuditScope(StrEnum):
    """Extent actually consumed by an audit, separate from a global verdict."""

    DENSE_SERIAL = "dense-serial"
    OWNER_LOCAL_CLOSURE = "owner-local-closure"


class CellMeshAuditReport(StrictModule, NonTrainableState):
    """Audit verdict with rejected issues, recorded findings, and evidence.

    ``evaluated_checks`` and ``skipped_checks`` partition the dispositioned
    checks by whether the policy evaluated them; ``passed`` certifies nothing
    about a skipped check. ``check_counts`` lists every evaluated check with its
    finding count in deterministic order; ``unresolved`` names checks that could
    not be decided and ``mandatory_unresolved`` those governed by a REJECT
    disposition (a successful result cannot carry them).

    Evidence scopes are separate: ``corner_checks`` cover welded topology and
    corner self-intersection, while ``mapped_checks`` cover Bernstein validity
    and sampled quality of scalar coordinate maps. Vertex-defined polygon and
    polyhedron quality retains its owning corner/star convention.
    Neither is a global-embedding or source-fidelity certificate; those are
    owned by ``phydrax.geometry`` mesh certificates. ``failing_cells`` lists
    cell global ids per mapped-cell check.
    """

    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    storage_id: str | None = eqx.field(static=True)
    audit_scope: CellMeshAuditScope = eqx.field(static=True)
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    quality_scope: str = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    recorded: tuple[str, ...] = eqx.field(static=True)
    unresolved: tuple[str, ...] = eqx.field(static=True)
    mandatory_unresolved: tuple[str, ...] = eqx.field(static=True)
    evaluated_checks: tuple[str, ...] = eqx.field(static=True)
    skipped_checks: tuple[str, ...] = eqx.field(static=True)
    corner_checks: tuple[str, ...] = eqx.field(static=True)
    mapped_checks: tuple[str, ...] = eqx.field(static=True)
    check_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    failing_cells: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    entity_counts: tuple[int, ...] = eqx.field(static=True)
    boundary_counts: tuple[int, ...] = eqx.field(static=True)
    connectivity_entries: int = eqx.field(static=True)
    unused_vertex_count: int = eqx.field(static=True)
    quality: CellQualityReport
    validity: CellValidityCertificate
    policy_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def require_passed(self, /) -> None:
        if not self.passed:
            raise ValueError("Cell mesh audit failed: " + "; ".join(self.issues))

    def require_decided(self, /) -> None:
        """Refuse a verdict that recorded an undecided REJECT-governed check."""

        if self.mandatory_unresolved:
            raise ValueError(
                "Cell mesh audit left mandatory checks unresolved: "
                + "; ".join(self.mandatory_unresolved)
            )


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
        return connectivity.cell_vertices.size
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


def _evidence_id(
    patches: Any, zones: Any, labels: Any, attributes: Any, associations: Any, /
) -> str:
    return canonical_fingerprint(
        {
            "patches": [value.patch_id for value in patches],
            "zones": [value.zone_id for value in zones],
            "labels": [value.label_id for value in labels],
            "attributes": [value.attribute_id for value in attributes],
            "associations": [value.association_id for value in associations],
        }
    )


def _required_audit_policy(
    policy: CellMeshAuditPolicy,
    required_checks: tuple[str, ...],
    /,
) -> CellMeshAuditPolicy:
    """Retain authored limits while executing a source's required closure check."""
    if (
        "open_boundary" not in required_checks
        or policy.watertight_boundary != CellMeshAuditDisposition.SKIP
    ):
        return policy
    return CellMeshAuditPolicy(
        unused_entities=policy.unused_entities,
        require_complete_association=policy.require_complete_association,
        minimum_measure=policy.minimum_measure,
        minimum_mean_ratio=policy.minimum_mean_ratio,
        maximum_aspect_ratio=policy.maximum_aspect_ratio,
        maximum_connectivity_entries=policy.maximum_connectivity_entries,
        coincident_vertices=policy.coincident_vertices,
        coincident_vertex_tolerance=policy.coincident_vertex_tolerance,
        maximum_coincidence_candidates=policy.maximum_coincidence_candidates,
        maximum_intersection_candidates=policy.maximum_intersection_candidates,
        duplicate_cells=policy.duplicate_cells,
        nonmanifold=policy.nonmanifold,
        inconsistent_orientation=policy.inconsistent_orientation,
        watertight_boundary=CellMeshAuditDisposition.REJECT,
        self_intersection=policy.self_intersection,
        invalid_geometry=policy.invalid_geometry,
        unresolved=policy.unresolved,
        validity_policy=policy.validity_policy,
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
        if isinstance(
            element,
            (
                FiniteElementSpec,
                BarycentricCellGeometryElement,
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
                SplineCellGeometryElement,
                LayerColumnCellGeometryElement,
            ),
        ):
            scope = "mapped_coordinate_cells"
            basis, _ = element.tabulate(
                jnp.asarray(reference_cell_topology(block.cell_kind).vertices)
            )
            corners = ein.contract(
                "vi,cia->cva", np.asarray(basis), points[np.asarray(route)]
            )
        elif isinstance(element, CellVertexGeometryElement):
            corners = points[np.asarray(route)]
        else:
            return "unsupported", ("unsupported_geometry_element",)
        expected = np.asarray(mesh.coordinates)[np.asarray(block.vertices)]
        scale = max(
            float(np.max(np.abs(expected), initial=0.0)), np.finfo(np.float64).tiny
        )
        if corners.shape != expected.shape or not np.allclose(
            corners, expected, rtol=0.0, atol=64.0 * np.finfo(np.float64).eps * scale
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
            tuple(mesh_ids[row])
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
        boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
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
        loops = tuple(tuple(mesh_ids[row]) for row in rows)
    vertex_faces: dict[int, set[int]] = {}
    for index, loop in enumerate(loops):
        for vertex in loop:
            vertex_faces.setdefault(vertex, set()).add(index)
    triangles = [[] for _ in loops]
    for block in surface.blocks:
        for row in np.asarray(block.vertices):
            triangle = tuple(surface_ids[row])
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


def _patch_zone_adjacency_issues(
    mesh: CellMesh,
    patches: tuple[MeshPatch, ...],
    zones: tuple[MeshZone, ...],
    /,
) -> tuple[str, ...]:
    adjacent_patches = tuple(patch for patch in patches if patch.adjacent_zone_ids)
    if not adjacent_patches:
        return ()
    connectivity = mesh.connectivity
    if mesh.topological_dimension == 2 and isinstance(
        connectivity, PolygonalConnectivity
    ):
        patch_dimension = 1
        zone_dimension = 2
    elif mesh.topological_dimension == 3 and isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
    ):
        patch_dimension = 2
        zone_dimension = 3
    else:
        return ("patch_zone_adjacency",)

    patch_entities = mesh.entity_set(patch_dimension)
    cell_entities = mesh.entity_set(zone_dimension)
    patch_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(patch_entities.entity_ids))
    }
    incidence = mesh.topology.incidences[-1]
    valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
    source_rows = np.asarray(incidence.relation.source_indices)[valid]
    target_rows = np.asarray(incidence.relation.target_indices)[valid]
    patch_cells = [set() for _ in range(patch_entities.count)]
    cell_ids = np.asarray(cell_entities.entity_ids)
    for patch_row, cell_row in zip(source_rows, target_rows, strict=True):
        patch_cells[int(patch_row)].add(int(cell_ids[int(cell_row)]))

    eligible_zone_ids: set[str] = set()
    cell_zones: dict[int, str] = {}
    for zone in zones:
        scope = zone.scope
        if (
            scope.source_id != mesh.mesh_id
            or scope.source_revision != mesh.numeric_version
            or scope.entity_dimension != zone_dimension
            or scope.entity_set_id != cell_entities.entity_set_id
        ):
            continue
        eligible_zone_ids.add(zone.zone_id)
        for identifier in np.asarray(scope.entity_ids):
            cell_zones[int(identifier)] = zone.zone_id

    for patch in adjacent_patches:
        scope = patch.scope
        expected = set(patch.adjacent_zone_ids)
        if (
            scope.source_id != mesh.mesh_id
            or scope.source_revision != mesh.numeric_version
            or scope.entity_dimension != patch_dimension
            or scope.entity_set_id != patch_entities.entity_set_id
            or not expected <= eligible_zone_ids
        ):
            return ("patch_zone_adjacency",)
        for identifier in np.asarray(scope.entity_ids):
            patch_row = patch_rows.get(int(identifier))
            if patch_row is None:
                return ("patch_zone_adjacency",)
            incident = patch_cells[patch_row]
            observed = {cell_zones[cell] for cell in incident if cell in cell_zones}
            if len(incident) != len(expected) or observed != expected:
                return ("patch_zone_adjacency",)
    return ()


def _mesh_evidence_issues(
    mesh: Any,
    boundary: Any,
    patches: Any,
    zones: Any,
    labels: Any,
    attributes: Any,
    associations: Any,
    /,
) -> Any:
    issues = list(_boundary_issues(mesh, boundary))
    issues.extend(_patch_zone_adjacency_issues(mesh, patches, zones))
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


def _complete_association_coverage(
    mesh: Any, boundary: Any, patches: Any, associations: Any, /
) -> bool:
    if (
        isinstance(mesh, CellMesh)
        and mesh.storage is not None
        and mesh.storage.global_entity_counts[mesh.topological_dimension] > 0
        and all(entities.count == 0 for entities in mesh.topology.entity_sets)
        and boundary is None
        and not associations
    ):
        # No resident target entity requires a local association. Global source
        # coverage remains the independent collective acceptance theorem's job.
        return True
    if not associations or any(not value.complete for value in associations):
        return False
    meshes = (mesh,) if boundary is None else (mesh, boundary.mesh)
    bindings = {
        entities.entity_set_id: (owner, entities)
        for owner in meshes
        for entities in owner.topology.entity_sets
    }
    coverage: dict[str, set[int]] = {}
    for association in associations:
        coverage.setdefault(association.target_entity_set_id, set()).update(
            int(value) for value in np.asarray(association.target_global_ids)
        )
    patch_ids: dict[str, set[int]] = {}
    for patch in patches:
        patch_ids.setdefault(patch.scope.entity_set_id, set()).update(
            int(value) for value in np.asarray(patch.scope.entity_ids)
        )
    for entity_set_id, identifiers in coverage.items():
        binding = bindings.get(entity_set_id)
        if binding is None:
            return False
        owner, entities = binding
        required = np.asarray(entities.entity_ids)
        if owner.topological_dimension == 3 and entities.intrinsic_dimension < 3:
            required = required[np.asarray(entities.subset("boundary").mask)]
        elif owner.topological_dimension == 2 and entities.intrinsic_dimension == 1:
            boundary_mask = (
                entities.subset("boundary").mask
                if owner.storage is None
                else owner.storage.local_physical_boundary_facets
            )
            if boundary_mask is None:
                raise ValueError(
                    "Owner-local association coverage requires physical boundary witnesses."
                )
            boundary_ids = required[np.asarray(boundary_mask)]
            required = np.asarray(
                sorted(
                    {
                        *(int(value) for value in boundary_ids),
                        *patch_ids.get(entity_set_id, set()),
                    }
                ),
                dtype=np.int64,
            )
        if not {int(value) for value in required} <= identifiers:
            return False
    return True


def _quality_binding(
    mesh: CellMesh, geometry: CellGeometrySpec, quality: CellQualityEvaluation | None, /
) -> tuple[CellQualityEvaluation, bool]:
    """Evaluate once, or verify supplied quality against the actual source map."""

    if quality is not None and not isinstance(quality, CellQualityEvaluation):
        raise TypeError("quality must be CellQualityEvaluation or None.")
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    token = _COORDINATE_BUDGET.set(None)
    try:
        expected = evaluate_cell_quality(
            mesh,
            geometry=geometry,
            metric=None if quality is None else quality.metric,
        )
    finally:
        _COORDINATE_BUDGET.reset(token)
    if quality is None:
        return expected, True
    bound = (
        quality.topology_id == mesh.topology_id
        and quality.geometry_layout_id == geometry.geometry_layout_id
        and quality.block_names == expected.block_names
        and quality.block_offsets == expected.block_offsets
        and all(
            np.array_equal(
                np.asarray(supplied),
                np.asarray(reference),
                equal_nan=np.issubdtype(np.asarray(reference).dtype, np.floating),
            )
            for supplied, reference in zip(
                jax.tree_util.tree_leaves(quality),
                jax.tree_util.tree_leaves(expected),
                strict=True,
            )
        )
    )
    return quality, bound


def _unused_counts(mesh: CellMesh, geometry: CellGeometrySpec, /) -> tuple[int, int]:
    used = np.zeros((mesh.coordinates.shape[0],), dtype=np.bool_)
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int32)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        used[np.unique(vertices[valid])] = True
    nodes = np.zeros((geometry.coordinates.shape[0],), dtype=np.bool_)
    for route in geometry.geometry_dofs:
        nodes[np.asarray(route).reshape(-1)] = True
    return int(np.count_nonzero(~used)), int(np.count_nonzero(~nodes))


def _dispose(
    findings: tuple[tuple[str, int, CellMeshAuditDisposition], ...], /
) -> tuple[list[str], list[str], tuple[tuple[str, int], ...], tuple[str, ...]]:
    issues = []
    recorded = []
    counts = []
    skipped = []
    for name, count, disposition in findings:
        match disposition:
            case CellMeshAuditDisposition.SKIP:
                skipped.append(name)
                continue
            case CellMeshAuditDisposition.REJECT:
                target = issues
            case CellMeshAuditDisposition.RECORD:
                target = recorded
            case _:
                raise ValueError(f"Unknown audit disposition {disposition!r}.")
        counts.append((name, count))
        if count:
            target.append(name)
    return issues, recorded, tuple(counts), tuple(skipped)


def _topology_findings(
    mesh: CellMesh, policy: CellMeshAuditPolicy, /
) -> tuple[tuple[tuple[str, int, CellMeshAuditDisposition], ...], tuple[str, ...]]:
    skip = CellMeshAuditDisposition.SKIP
    evidence = audit_welded_topology(
        mesh,
        np.asarray(mesh.coordinates, dtype=np.float64),
        coincident_tolerance=(
            None
            if policy.coincident_vertices == skip
            else policy.coincident_vertex_tolerance
        ),
        candidate_capacity=policy.maximum_coincidence_candidates,
        intersection_candidate_capacity=policy.maximum_intersection_candidates,
        check_manifold=policy.nonmanifold != skip,
        check_watertight=policy.watertight_boundary != skip,
        check_self_intersection=policy.self_intersection != skip,
    )
    findings = (
        ("coincident_vertices", evidence.coincident_vertices, policy.coincident_vertices),
        ("collapsed_cells", evidence.collapsed_cells, policy.coincident_vertices),
        ("duplicate_cells", evidence.duplicate_cells, policy.duplicate_cells),
        ("nonmanifold_facets", evidence.nonmanifold_facets, policy.nonmanifold),
        ("nonmanifold_edges", evidence.nonmanifold_edges, policy.nonmanifold),
        ("nonmanifold_vertices", evidence.nonmanifold_vertices, policy.nonmanifold),
        (
            "inconsistent_orientation",
            evidence.inconsistent_facets,
            policy.inconsistent_orientation,
        ),
        ("open_boundary", evidence.open_facets, policy.watertight_boundary),
        ("self_intersection", evidence.self_intersections, policy.self_intersection),
    )
    return findings, evidence.unresolved


_MAPPED_CHECKS = ("invalid_geometry", "unresolved_geometry_validity")


def _mandatory_unresolved(
    unresolved: tuple[str, ...], policy: CellMeshAuditPolicy, /
) -> tuple[str, ...]:
    """Unresolved checks whose governing disposition is REJECT."""

    governing = {
        "coincident_vertex_capacity": policy.coincident_vertices,
        "self_intersection_capacity": policy.self_intersection,
        "self_intersection_predicates": policy.self_intersection,
        "geometry_validity": policy.invalid_geometry,
    }
    return tuple(
        name for name in unresolved if governing[name] == CellMeshAuditDisposition.REJECT
    )


def _failing_cells(
    mesh: CellMesh, validity: CellValidityCertificate, /
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    cell_ids = (
        np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        if mesh.blocks
        else np.empty((0,), dtype=np.int64)
    )
    status = np.asarray(validity.status)
    if status.shape != cell_ids.shape:
        raise ValueError(
            "Geometry validity findings must match the actual resident cell identity bank."
        )
    failing = []
    for name, code in (
        ("invalid_geometry", CellValidityStatus.INVALID),
        ("unresolved_geometry_validity", CellValidityStatus.UNRESOLVED),
    ):
        selected = status == code
        if np.any(selected):
            failing.append((name, tuple(int(value) for value in cell_ids[selected])))
    return tuple(failing)


def audit_cell_mesh(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    quality: CellQualityEvaluation | None = None,
    /,
    *,
    policy: CellMeshAuditPolicy | None = None,
    prepared_validity: CellValidityCertificate | None = None,
    boundary: SurfaceModel | None = None,
    patches: tuple[MeshPatch, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
) -> CellMeshAuditReport:
    """Audit bindings, welded topology, sampled quality, and certified validity.

    Geometric validity of every mapped cell (including high-order geometry) is
    decided by the Bernstein validity certificate; sampled source-Jacobian
    quality feeds the unchanged quality thresholds. When ``quality`` is omitted
    it is evaluated once here; a supplied evaluation is verified against the
    actual coordinate map and its source layout. Association
    residuals are structurally validated by GeometryAssociation; source-specific
    residual tolerances remain the generating provider's responsibility.
    ``prepared_validity`` reuses only a wholly positive certificate bound to the
    exact geometry, topology and validity policy; foreign evidence is refused.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    audit_policy = CellMeshAuditPolicy() if policy is None else policy
    if not isinstance(audit_policy, CellMeshAuditPolicy):
        raise TypeError("policy must be CellMeshAuditPolicy or None.")
    quality_evaluation, quality_bound = _quality_binding(mesh, geometry, quality)
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    token = _COORDINATE_BUDGET.set(None)
    try:
        quality_scope, geometry_issues = _geometry_binding(mesh, geometry)
    finally:
        _COORDINATE_BUDGET.reset(token)
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
    if not quality_bound:
        issues.append("quality_binding")
    entries = _connectivity_entries(mesh)
    if entries > audit_policy.maximum_connectivity_entries:
        issues.append("connectivity_capacity_exceeded")
    unused, unused_nodes = _unused_counts(mesh, geometry)
    if prepared_validity is None:
        validity = certify_cell_geometry_validity(
            geometry, mesh=mesh, policy=audit_policy.validity_policy
        )
    else:
        if not isinstance(prepared_validity, CellValidityCertificate):
            raise TypeError("prepared_validity must be CellValidityCertificate or None.")
        prepared_validity.require_bound(geometry, mesh=mesh)
        if (
            not prepared_validity.all_certified
            or not np.all(
                np.asarray(prepared_validity.status) == CellValidityStatus.CERTIFIED_VALID
            )
            or prepared_validity.status.shape[0]
            != sum(block.vertices.shape[0] for block in mesh.blocks)
            or prepared_validity.unsupported_block_names
            or prepared_validity.unresolved_reasons
            or prepared_validity.policy_id != audit_policy.validity_policy.policy_id
        ):
            raise ValueError(
                "Prepared validity must be positive under the exact audit policy."
            )
        validity = prepared_validity
    topology_findings, unresolved_checks = _topology_findings(mesh, audit_policy)
    unresolved = list(unresolved_checks)
    if validity.unresolved_count:
        unresolved.append("geometry_validity")
    checks = (
        ("unused_vertices", unused, audit_policy.unused_entities),
        ("unused_geometry_nodes", unused_nodes, audit_policy.unused_entities),
        *topology_findings,
        ("invalid_geometry", validity.invalid_count, audit_policy.invalid_geometry),
    )
    dispositioned = (
        *checks,
        *((f"unresolved_{name}", 1, audit_policy.unresolved) for name in unresolved),
    )
    rejected, recorded, check_counts, skipped_checks = _dispose(dispositioned)
    evaluated_checks = tuple(
        name
        for name, _, disposition in dispositioned
        if disposition != CellMeshAuditDisposition.SKIP
    )
    issues.extend(rejected)
    quality_report = summarize_cell_quality(quality_evaluation, mesh=mesh)
    if quality_report.minimum_measure <= audit_policy.minimum_measure:
        issues.append("minimum_measure")
    if quality_report.minimum_mean_ratio < audit_policy.minimum_mean_ratio:
        issues.append("minimum_mean_ratio")
    if quality_report.maximum_aspect_ratio > audit_policy.maximum_aspect_ratio:
        issues.append("maximum_aspect_ratio")
    if audit_policy.require_complete_association and not _complete_association_coverage(
        mesh, boundary, patches, associations
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
    recorded_ = tuple(recorded)
    unresolved_ = tuple(unresolved)
    mandatory = _mandatory_unresolved(unresolved_, audit_policy)
    source_quality = quality_scope == "mapped_coordinate_cells"
    mapped_checks = (
        *(name for name in evaluated_checks if name in _MAPPED_CHECKS),
        *(("sampled_quality",) if source_quality else ()),
    )
    corner_checks = (
        *(name for name in evaluated_checks if name not in _MAPPED_CHECKS),
        *(("sampled_quality",) if not source_quality else ()),
    )
    failing_cells = _failing_cells(mesh, validity)
    geometry_id = cell_geometry_id(geometry)
    evidence_id = _evidence_id(patches, zones, labels, attributes, associations)
    storage_id = None if mesh.storage is None else mesh.storage.storage_id
    audit_scope = (
        CellMeshAuditScope.DENSE_SERIAL
        if mesh.storage is None
        else CellMeshAuditScope.OWNER_LOCAL_CLOSURE
    )
    global_entity_counts = (
        entity_counts if mesh.storage is None else mesh.storage.global_entity_counts
    )
    return CellMeshAuditReport(
        mesh_id=mesh.mesh_id,
        topology_id=mesh.topology_id,
        geometry_layout_id=geometry.geometry_layout_id,
        geometry_id=geometry_id,
        evidence_id=evidence_id,
        storage_id=storage_id,
        audit_scope=audit_scope,
        global_entity_counts=global_entity_counts,
        quality_scope=quality_scope,
        passed=not normalized_issues,
        issues=normalized_issues,
        recorded=recorded_,
        unresolved=unresolved_,
        mandatory_unresolved=mandatory,
        evaluated_checks=evaluated_checks,
        skipped_checks=skipped_checks,
        corner_checks=corner_checks,
        mapped_checks=mapped_checks,
        check_counts=check_counts,
        failing_cells=failing_cells,
        vertex_count=mesh.coordinates.shape[0],
        entity_counts=entity_counts,
        boundary_counts=boundary_counts,
        connectivity_entries=entries,
        unused_vertex_count=unused,
        quality=quality_report,
        validity=validity,
        policy_id=audit_policy.policy_id,
        report_id=canonical_fingerprint(
            {
                "kind": "cell-mesh-audit-report",
                "mesh": mesh.mesh_id,
                "topology": mesh.topology_id,
                "geometry_layout": geometry.geometry_layout_id,
                "geometry": geometry_id,
                "evidence": evidence_id,
                "storage": storage_id,
                "audit_scope": audit_scope.value,
                "global_entity_counts": global_entity_counts,
                "quality_scope": quality_scope,
                "quality": quality_report.report_id,
                "validity": validity.certificate_id,
                "issues": normalized_issues,
                "recorded": recorded_,
                "unresolved": unresolved_,
                "mandatory_unresolved": mandatory,
                "evaluated_checks": evaluated_checks,
                "skipped_checks": skipped_checks,
                "check_counts": check_counts,
                "failing_cells": failing_cells,
                "entity_counts": entity_counts,
                "boundary_counts": boundary_counts,
                "connectivity_entries": entries,
                "policy": audit_policy.policy_id,
            }
        ),
    )


__all__ = [
    "CellMeshAuditDisposition",
    "CellMeshAuditPolicy",
    "CellMeshAuditReport",
    "audit_cell_mesh",
]
