#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any, TYPE_CHECKING

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
    CellVertexGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import SimplicialConnectivity
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
    _with_certified_mapped_measures,
    CellQualityEvaluation,
    CellQualityReport,
    evaluate_cell_quality,
    summarize_cell_quality,
)
from ._scope import MeshingEntityKind


if TYPE_CHECKING:
    from ..geometry._global_embedding import GlobalEmbeddingCertificate


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
    """Dispositions and mathematical tolerances of the cell-mesh audit.

    Topology diagnostics use the welded complex at the declared geometric
    tolerance without modifying source identities. Candidate processing is
    complete and streamed; unresolved predicates are rejected or recorded,
    never silently skipped.
    """

    unused_entities: CellMeshAuditDisposition = eqx.field(static=True)
    require_complete_association: bool = eqx.field(static=True)
    minimum_measure: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    coincident_vertices: CellMeshAuditDisposition = eqx.field(static=True)
    coincident_vertex_tolerance: float = eqx.field(static=True)
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
        coincident_vertices: CellMeshAuditDisposition = CellMeshAuditDisposition.REJECT,
        coincident_vertex_tolerance: float = 1.0e-12,
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
        tolerance = float(coincident_vertex_tolerance)
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
        if not np.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 1.0:
            raise ValueError("coincident_vertex_tolerance must lie in [0, 1).")
        if unresolved == CellMeshAuditDisposition.SKIP:
            raise ValueError("Unresolved checks must be rejected or recorded.")
        if not isinstance(validity, CellValidityPolicy):
            raise TypeError("validity_policy must be CellValidityPolicy or None.")
        self.unused_entities = unused_entities
        self.require_complete_association = bool(require_complete_association)
        self.minimum_measure = measure
        self.minimum_mean_ratio = ratio
        self.maximum_aspect_ratio = aspect
        self.coincident_vertices = coincident_vertices
        self.coincident_vertex_tolerance = tolerance
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
                "coincident_vertex_tolerance": tolerance,
                "validity_policy": validity.policy_id,
            }
        )


class CellMeshManifoldScope(StrictModule, NonTrainableState):
    """Declared vertex scope of the manifold (star-connectivity) checks.

    A resident patch of a distributed mesh is legitimately pinched where its
    resident cells end. The scope names the vertices whose complete star this
    audit certifies (``vertex_ids``, all resident in the audited mesh) and the
    non-resident star cells that complete those stars: dense blocks of cell
    IDs and global vertex IDs per cell kind. Vertices outside the scope are
    certified by the scopes of the other patches of the same partition, which
    ``kind`` names (``"owned-star"``: every vertex is certified by its owner).
    """

    kind: str = eqx.field(static=True)
    vertex_ids: jax.Array
    star_cell_kinds: tuple[str, ...] = eqx.field(static=True)
    star_cell_ids: tuple[jax.Array, ...]
    star_cell_vertex_ids: tuple[jax.Array, ...]
    scope_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        vertex_ids: Any,
        /,
        *,
        star_cells: tuple[tuple[str, Any, Any], ...] = (),
    ) -> None:
        if not isinstance(kind, str) or not kind:
            raise ValueError("Manifold scope kind must be a non-empty string.")
        vertices = np.asarray(vertex_ids)
        if (
            vertices.ndim != 1
            or vertices.dtype.kind not in "iu"
            or np.any(vertices < 0)
            or np.unique(vertices).size != vertices.size
        ):
            raise ValueError("Manifold scope vertex IDs must be unique non-negative IDs.")
        kinds: list[str] = []
        cell_ids: list[np.ndarray] = []
        cell_vertices: list[np.ndarray] = []
        for cell_kind, identifiers, rows in sorted(
            star_cells, key=lambda value: value[0]
        ):
            ids = np.asarray(identifiers)
            values = np.asarray(rows)
            if (
                not isinstance(cell_kind, str)
                or cell_kind in kinds
                or cell_kind == "polyhedron"
                or ids.ndim != 1
                or ids.dtype.kind not in "iu"
                or values.ndim != 2
                or values.dtype.kind not in "iu"
                or values.shape[0] != ids.size
                or values.shape[1] < 2
                or np.any(values < 0)
            ):
                raise ValueError(
                    "Manifold scope star cells must be dense blocks of IDs and vertex IDs."
                )
            order = np.argsort(ids, kind="stable")
            kinds.append(cell_kind)
            cell_ids.append(ids[order].astype(np.int64))
            cell_vertices.append(values[order].astype(np.int64))
        every_cell = np.concatenate(cell_ids) if cell_ids else np.empty((0,), np.int64)
        if np.unique(every_cell).size != every_cell.size:
            raise ValueError("Manifold scope star cell IDs must be unique.")
        sorted_vertices = np.sort(vertices.astype(np.int64))
        self.kind = kind
        self.vertex_ids = jnp.asarray(sorted_vertices)
        self.star_cell_kinds = tuple(kinds)
        self.star_cell_ids = tuple(jnp.asarray(value) for value in cell_ids)
        self.star_cell_vertex_ids = tuple(jnp.asarray(value) for value in cell_vertices)
        self.scope_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-manifold-scope",
                "scope": kind,
                "vertices": sorted_vertices,
                "star_cells": [
                    {"cell_kind": name, "cell_ids": ids, "vertex_ids": rows}
                    for name, ids, rows in zip(
                        kinds, cell_ids, cell_vertices, strict=True
                    )
                ],
            }
        )


class CellMeshAuditReport(StrictModule, NonTrainableState):
    """Audit verdict with rejected issues, recorded findings, and evidence.

    ``evaluated_checks`` and ``skipped_checks`` partition the dispositioned
    checks by whether the policy evaluated them; ``passed`` certifies nothing
    about a skipped check. ``check_counts`` lists every evaluated check with its
    finding count in deterministic order; ``unresolved`` names checks that could
    not be decided. ``manifold_scope`` is the declared vertex scope of the
    manifold checks (``None``: every vertex of the mesh was checked).
    """

    mesh_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    quality_scope: str = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    recorded: tuple[str, ...] = eqx.field(static=True)
    unresolved: tuple[str, ...] = eqx.field(static=True)
    evaluated_checks: tuple[str, ...] = eqx.field(static=True)
    skipped_checks: tuple[str, ...] = eqx.field(static=True)
    check_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    entity_counts: tuple[int, ...] = eqx.field(static=True)
    boundary_counts: tuple[int, ...] = eqx.field(static=True)
    connectivity_entries: int = eqx.field(static=True)
    unused_vertex_count: int = eqx.field(static=True)
    manifold_scope: CellMeshManifoldScope | None
    quality: CellQualityReport
    validity: CellValidityCertificate
    policy_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def require_passed(self, /) -> None:
        if not self.passed:
            raise ValueError("Cell mesh audit failed: " + "; ".join(self.issues))


def _connectivity_entries(mesh: CellMesh, /) -> int:
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        return sum(rows.size for rows in connectivity.entities[1:]) + sum(
            rows.size for rows in connectivity.cell_entities[1:-1]
        )
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
            corners = ein.contract(
                "vi,cia->cva", np.asarray(basis), points[np.asarray(route)]
            )
        elif isinstance(element, CellVertexGeometryElement):
            corners = points[np.asarray(route)]
        elif isinstance(element, ScalarCoordinateElement):
            scope = "corner_cells"
            reference = np.asarray(
                reference_cell_topology(block.cell_kind).vertices, dtype=np.float64
            )
            corners = geometry.evaluate(
                mesh,
                block.global_ids,
                np.broadcast_to(reference, (block.global_ids.size, *reference.shape)),
            )
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
        if not isinstance(
            mesh.connectivity, (PolygonalConnectivity, SimplicialConnectivity)
        ):
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
            (
                TetrahedralConnectivity,
                HexahedralConnectivity,
                PolyhedralConnectivity,
                SimplicialConnectivity,
            ),
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
        elif isinstance(connectivity, SimplicialConnectivity):
            rows = np.asarray(connectivity.entities[2])[boundary_mask]
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
        connectivity, (PolygonalConnectivity, SimplicialConnectivity)
    ):
        patch_dimension = 1
        zone_dimension = 2
    elif mesh.topological_dimension == 3 and isinstance(
        connectivity,
        (
            TetrahedralConnectivity,
            HexahedralConnectivity,
            PolyhedralConnectivity,
            SimplicialConnectivity,
        ),
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
            boundary_ids = required[np.asarray(entities.subset("boundary").mask)]
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
    mesh: CellMesh, quality: CellQualityEvaluation | None, /
) -> tuple[CellQualityEvaluation, bool]:
    """Evaluate quality once, or verify a supplied evaluation against the mesh."""

    expected = evaluate_cell_quality(mesh)
    if quality is None:
        return expected, True
    if not isinstance(quality, CellQualityEvaluation):
        raise TypeError("quality must be CellQualityEvaluation or None.")
    bound = (
        quality.topology_id == mesh.topology_id
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


def _periodic_self_intersection_evidence(
    embedding: GlobalEmbeddingCertificate, /
) -> tuple[int, bool]:
    contact_checks = ("periodic_cell_contact", "periodic_cell_lift_collision")
    periodic_findings = tuple(
        finding for finding in embedding.findings if finding.check.startswith("periodic_")
    )
    violations = tuple(
        finding
        for finding in periodic_findings
        if finding.check in contact_checks and finding.status == "violated"
    )
    complete = "periodic_exact_cell_contacts" in embedding.evaluated_checks and all(
        finding.check in contact_checks and finding.status == "violated"
        for finding in periodic_findings
    )
    return len({finding.finding_id for finding in violations}), not complete


def _manifold_scope_rows(
    mesh: CellMesh, scope: CellMeshManifoldScope, /
) -> tuple[np.ndarray, tuple[tuple[str, np.ndarray], ...]]:
    """Scope vertex rows and star-cell vertex keys of one audited mesh.

    Resident vertices key by mesh row; non-resident vertices take keys past
    the vertex count in ascending global-ID order.
    """

    mesh_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(mesh_ids, kind="stable")
    sorted_ids = mesh_ids[order]

    def resident_rows(identifiers: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        position = np.minimum(
            np.searchsorted(sorted_ids, identifiers), max(sorted_ids.size - 1, 0)
        )
        found = (
            sorted_ids[position] == identifiers
            if sorted_ids.size
            else np.zeros(identifiers.shape, dtype=np.bool_)
        )
        return np.where(found, order[position] if sorted_ids.size else -1, -1), found

    scope_ids = np.asarray(scope.vertex_ids, dtype=np.int64)
    scope_rows, scope_found = resident_rows(scope_ids)
    if not np.all(scope_found):
        raise ValueError("A manifold scope names vertices absent from the audited mesh.")
    cell_ids = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64).reshape(-1)
            for block in mesh.blocks
        ]
    )
    star_ids = tuple(np.asarray(value, dtype=np.int64) for value in scope.star_cell_ids)
    if any(np.any(np.isin(value, cell_ids)) for value in star_ids):
        raise ValueError("Manifold scope star cells must be absent from the mesh.")
    star_rows = tuple(
        np.asarray(value, dtype=np.int64) for value in scope.star_cell_vertex_ids
    )
    every_vertex = (
        np.concatenate([value.reshape(-1) for value in star_rows])
        if star_rows
        else np.empty((0,), dtype=np.int64)
    )
    _, every_found = resident_rows(every_vertex)
    foreign = np.unique(every_vertex[~every_found])
    supplement = []
    for cell_kind, rows in zip(scope.star_cell_kinds, star_rows, strict=True):
        resident, found = resident_rows(rows.reshape(-1))
        keys = np.where(
            found,
            resident,
            mesh_ids.size + np.searchsorted(foreign, rows.reshape(-1)),
        )
        supplement.append((cell_kind, keys.reshape(rows.shape)))
    return scope_rows, tuple(supplement)


def _topology_findings(
    mesh: CellMesh,
    policy: CellMeshAuditPolicy,
    /,
    *,
    global_embedding: GlobalEmbeddingCertificate | None = None,
    manifold_scope: CellMeshManifoldScope | None = None,
) -> tuple[tuple[tuple[str, int, CellMeshAuditDisposition], ...], tuple[str, ...]]:
    skip = CellMeshAuditDisposition.SKIP
    periodic_mesh = mesh.periodic_topology is not None
    periodic_embedding = periodic_mesh and global_embedding is not None
    scope_rows, star_supplement = (
        (None, ())
        if manifold_scope is None
        else _manifold_scope_rows(mesh, manifold_scope)
    )
    evidence = audit_welded_topology(
        mesh,
        np.asarray(mesh.coordinates, dtype=np.float64),
        coincident_tolerance=(
            None
            if policy.coincident_vertices == skip
            else policy.coincident_vertex_tolerance
        ),
        check_manifold=policy.nonmanifold != skip,
        check_watertight=policy.watertight_boundary != skip,
        check_self_intersection=(
            policy.self_intersection != skip and not periodic_embedding
        ),
        manifold_vertices=scope_rows,
        star_supplement=star_supplement,
    )
    self_intersections = evidence.self_intersections
    unresolved = list(evidence.unresolved)
    if policy.self_intersection != skip and periodic_mesh:
        if periodic_embedding:
            if global_embedding is None:
                raise RuntimeError("Periodic audit lost its quotient embedding proof.")
            self_intersections, unresolved_contact = _periodic_self_intersection_evidence(
                global_embedding
            )
            if unresolved_contact:
                unresolved.append("self_intersection")
        else:
            unresolved.append("self_intersection")
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
        ("self_intersection", self_intersections, policy.self_intersection),
    )
    return findings, tuple(dict.fromkeys(unresolved))


def audit_cell_mesh(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    quality: CellQualityEvaluation | None = None,
    /,
    *,
    policy: CellMeshAuditPolicy | None = None,
    boundary: SurfaceModel | None = None,
    patches: tuple[MeshPatch, ...] = (),
    associations: tuple[GeometryAssociation, ...] = (),
    attributes: tuple[MeshAttribute, ...] = (),
    zones: tuple[MeshZone, ...] = (),
    labels: tuple[MeshLabel, ...] = (),
    validity: CellValidityCertificate | None = None,
    global_embedding: GlobalEmbeddingCertificate | None = None,
    manifold_scope: CellMeshManifoldScope | None = None,
) -> CellMeshAuditReport:
    """Audit bindings, welded topology, sampled quality, and certified validity.

    Geometric validity of every mapped cell (including high-order geometry) is
    decided by the Bernstein validity certificate; sampled corner quality only
    feeds the quality thresholds. When ``quality`` is omitted it is evaluated
    once here; a supplied evaluation is verified against the mesh. Association
    residuals are structurally validated by GeometryAssociation; source-specific
    residual tolerances remain the generating provider's responsibility. A supplied
    global embedding proof is reused for periodic quotient self-intersection and
    must bind the exact same geometry and periodic topology. A ``manifold_scope``
    restricts the vertex and edge star checks of a resident patch to the declared
    vertex stars; the report records it.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    audit_policy = CellMeshAuditPolicy() if policy is None else policy
    if not isinstance(audit_policy, CellMeshAuditPolicy):
        raise TypeError("policy must be CellMeshAuditPolicy or None.")
    if manifold_scope is not None and (
        not isinstance(manifold_scope, CellMeshManifoldScope)
        or audit_policy.nonmanifold == CellMeshAuditDisposition.SKIP
    ):
        raise ValueError(
            "manifold_scope must be CellMeshManifoldScope of an evaluated manifold check."
        )
    quality_evaluation, quality_bound = _quality_binding(mesh, quality)
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
    if not quality_bound:
        issues.append("quality_binding")
    entries = _connectivity_entries(mesh)
    unused, unused_nodes = _unused_counts(mesh, geometry)
    validity_ = (
        certify_cell_geometry_validity(
            geometry, mesh=mesh, policy=audit_policy.validity_policy
        )
        if validity is None
        else validity
    )
    if not isinstance(validity_, CellValidityCertificate):
        raise TypeError("validity must be CellValidityCertificate or None.")
    validity_.require_bound(geometry, mesh=mesh)
    if quality_scope == "corner_cells" and validity_.all_certified:
        quality_evaluation = _with_certified_mapped_measures(
            quality_evaluation, mesh, geometry, validity_
        )
    if global_embedding is not None:
        from ..geometry._global_embedding import GlobalEmbeddingCertificate

        if not isinstance(global_embedding, GlobalEmbeddingCertificate):
            raise TypeError(
                "global_embedding must be GlobalEmbeddingCertificate or None."
            )
        global_embedding.require_current(mesh, geometry, validity=validity_)
    elif (
        mesh.periodic_topology is not None
        and audit_policy.self_intersection != CellMeshAuditDisposition.SKIP
    ):
        # Lifted-image intersection tests cannot decide quotient overlap; the
        # exact quotient embedding proof is the only periodic witness.
        from ..geometry._global_embedding import certify_global_embedding

        global_embedding = certify_global_embedding(mesh, geometry, validity_)
    topology_findings, unresolved_checks = _topology_findings(
        mesh,
        audit_policy,
        global_embedding=global_embedding,
        manifold_scope=manifold_scope,
    )
    unresolved = list(unresolved_checks)
    if validity_.unresolved_count:
        unresolved.append("geometry_validity")
    checks = (
        ("unused_vertices", unused, audit_policy.unused_entities),
        ("unused_geometry_nodes", unused_nodes, audit_policy.unused_entities),
        *topology_findings,
        ("invalid_geometry", validity_.invalid_count, audit_policy.invalid_geometry),
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
    quality_report = summarize_cell_quality(quality_evaluation)
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
    geometry_id = cell_geometry_id(geometry)
    evidence_id = _evidence_id(patches, zones, labels, attributes, associations)
    return CellMeshAuditReport(
        mesh_id=mesh.mesh_id,
        geometry_layout_id=geometry.geometry_layout_id,
        geometry_id=geometry_id,
        evidence_id=evidence_id,
        quality_scope=quality_scope,
        passed=not normalized_issues,
        issues=normalized_issues,
        recorded=recorded_,
        unresolved=unresolved_,
        evaluated_checks=evaluated_checks,
        skipped_checks=skipped_checks,
        check_counts=check_counts,
        vertex_count=mesh.coordinates.shape[0],
        entity_counts=entity_counts,
        boundary_counts=boundary_counts,
        connectivity_entries=entries,
        unused_vertex_count=unused,
        manifold_scope=manifold_scope,
        quality=quality_report,
        validity=validity_,
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
                "validity": validity_.certificate_id,
                "issues": normalized_issues,
                "recorded": recorded_,
                "unresolved": unresolved_,
                "evaluated_checks": evaluated_checks,
                "skipped_checks": skipped_checks,
                "check_counts": check_counts,
                "entity_counts": entity_counts,
                "boundary_counts": boundary_counts,
                "connectivity_entries": entries,
                "policy": audit_policy.policy_id,
                **(
                    {}
                    if manifold_scope is None
                    else {"manifold_scope": manifold_scope.scope_id}
                ),
            }
        ),
    )


__all__ = [
    "CellMeshAuditDisposition",
    "CellMeshAuditPolicy",
    "CellMeshAuditReport",
    "CellMeshManifoldScope",
    "audit_cell_mesh",
]
