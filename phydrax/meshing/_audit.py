#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from fractions import Fraction
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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
    coordinate_source_basis,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._coordinate_enclosure import evaluate_expression
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._reference import (
    FiniteElementSpec,
    lagrange_reference_node_data,
)
from ..geometry._interval_enclosure import interval_subtract
from ..geometry._rational import fraction_down, fraction_up
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ..geometry.surface import SurfaceModel
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
    source_boundary_association_piece_rows,
    source_boundary_image_association_piece_rows,
)
from ._audit_topology import audit_welded_topology
from ._curve_association_replay import (
    curve_vertex_closure_issues,
    is_curve_vertex_closure,
)
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
    from ._association import _LayerImageSupportPreparation
    from ._certification_inputs import MeshCertificationInputs
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
    from ._layer_source_association_replay import _ComputedLayerAssociationReplay


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
    ``policy`` is the audit policy that decided the report; ``report_id`` binds
    its identity, so the report is recomputed from itself.
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
    policy: CellMeshAuditPolicy
    report_id: str = eqx.field(static=True)

    @property
    def policy_id(self, /) -> str:
        return self.policy.policy_id

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


# Corner binding certifies |c - x| <= 64 eps * scale for every corner coordinate:
# c is the bound geometry's exact image of a reference cell corner, x the mesh
# vertex coordinate, and scale the block's largest vertex coordinate magnitude
# (at least the smallest normal binary64). The tolerance is exact: 2^-46 * scale.
_CORNER_BINDING_TOLERANCE = Fraction(64) * Fraction(float(np.finfo(np.float64).eps))
_CORNER_DOFS: dict[str, tuple[int, ...] | None] = {}


def _vertex_dofs(element: FiniteElementSpec, arity: int, /) -> tuple[int, ...] | None:
    """Local DOFs equal to the geometry map at each reference corner, if any.

    A nodal basis is dual to its point functionals, so the map takes exactly
    the coefficient of a vertex-owned node at that vertex. Other elements use
    their exact source basis when it selects one coefficient per corner.
    """
    key = element.element_id
    if key not in _CORNER_DOFS:
        dofs: tuple[int, ...] | None = None
        if element.form_basis is None and element.representation == "point_value":
            try:
                nodes = lagrange_reference_node_data(element)
            except ValueError:
                nodes = ()
            owners = {
                node.entity_vertices[0]: node.local_dof
                for node in nodes
                if node.entity_dimension == 0 and len(node.entity_vertices) == 1
            }
            if set(owners) == set(range(arity)):
                dofs = tuple(owners[vertex] for vertex in range(arity))
        if dofs is None:
            corners = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(element.cell_kind).vertices
            )
            try:
                basis = coordinate_source_basis(element)
                values = tuple(
                    tuple(evaluate_expression(function, corner) for function in basis)
                    for corner in corners
                )
            except (TypeError, ValueError, ZeroDivisionError):
                values = ()
            selected = tuple(
                tuple(index for index, value in enumerate(row) if value != 0)
                for row in values
            )
            if len(selected) == arity and all(
                len(indices) == 1 and row[indices[0]] == 1
                for indices, row in zip(selected, values, strict=True)
            ):
                dofs = tuple(indices[0] for indices in selected)
        _CORNER_DOFS[key] = dofs
    return _CORNER_DOFS[key]


def _within_tolerance(
    first: np.ndarray, second: np.ndarray, tolerance: Fraction, /
) -> bool:
    """Exact ``|first - second| <= tolerance`` for binary64 arrays."""
    lower, upper = interval_subtract((first, first), (second, second))
    lower, upper = np.asarray(lower), np.asarray(upper)
    inner, outer = fraction_down(tolerance), fraction_up(tolerance)
    if np.any((lower > outer) | (upper < -outer)):
        return False
    undecided = ~((lower >= -inner) & (upper <= inner))
    return all(
        abs(Fraction(float(first[index])) - Fraction(float(second[index]))) <= tolerance
        for index in zip(*np.nonzero(undecided), strict=True)
    )


def _corner_binding_holds(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    block: Any,
    element: Any,
    route: np.ndarray,
    points: np.ndarray,
    /,
) -> bool | None:
    """Exact corner binding of one block; ``None`` without exact corner images."""
    vertices = np.asarray(block.vertices, dtype=np.int64)
    expected = np.asarray(mesh.coordinates, dtype=np.float64)[vertices]
    scale = max(float(np.max(np.abs(expected), initial=0.0)), np.finfo(np.float64).tiny)
    tolerance = _CORNER_BINDING_TOLERANCE * Fraction(scale)
    exact_bank = geometry.exact_coordinate_bank is not None
    dofs: tuple[int, ...] | None = None
    if isinstance(element, CellVertexGeometryElement):
        dofs = tuple(range(route.shape[-1])) if route.ndim == 2 else None
    elif isinstance(element, FiniteElementSpec) and (
        exact_bank or mesh.periodic_topology is None
    ):
        dofs = _vertex_dofs(element, vertices.shape[1])
    if dofs is not None:
        if route.ndim != 2 or max(dofs) >= route.shape[1]:
            return False
        rows = route[:, np.asarray(dofs, dtype=np.int64)]
        if rows.shape != vertices.shape:
            return False
        pairs = np.unique(
            np.stack((rows.reshape(-1), vertices.reshape(-1)), axis=1), axis=0
        )
        targets = np.asarray(mesh.coordinates, dtype=np.float64)[pairs[:, 1]]
        if not exact_bank:
            return _within_tolerance(
                np.asarray(points, dtype=np.float64)[pairs[:, 0]], targets, tolerance
            )
        bank = geometry.source_coordinates(pairs[:, 0])
        return all(
            abs(value - Fraction(float(target))) <= tolerance
            for exact_row, target_row in zip(bank, targets, strict=True)
            for value, target in zip(exact_row, target_row, strict=True)
        )
    if not isinstance(element, ScalarCoordinateElement):
        return None
    reference = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(
            reference_cell_topology(block.cell_kind).vertices, dtype=np.float64
        )
    )
    identifiers = np.asarray(block.global_ids, dtype=np.int64)
    try:
        images = geometry.evaluate_exact(
            mesh, identifiers, tuple(reference for _ in range(identifiers.size))
        )
    except (TypeError, ValueError):
        return None
    if len(images) != expected.shape[0] or any(
        len(cell) != expected.shape[1]
        or any(len(corner) != expected.shape[2] for corner in cell)
        for cell in images
    ):
        return False
    return all(
        abs(value - Fraction(float(target))) <= tolerance
        for cell, cell_targets in zip(images, expected, strict=True)
        for corner, corner_targets in zip(cell, cell_targets, strict=True)
        for value, target in zip(corner, corner_targets, strict=True)
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
        elif not isinstance(element, CellVertexGeometryElement):
            if not isinstance(element, ScalarCoordinateElement):
                return "unsupported", ("unsupported_geometry_element",)
            scope = "corner_cells"
        bound = _corner_binding_holds(
            mesh, geometry, block, element, np.asarray(route), points
        )
        if bound is None:
            return "unsupported", ("unsupported_geometry_element",)
        if not bound:
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
        for row, (identifier, source, resolved) in enumerate(
            zip(
                np.asarray(association.target_global_ids),
                association.source_entity_ids,
                np.asarray(association.resolved),
                strict=True,
            )
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
            identity = (source, association.source_occurrence_paths[row])
            piece_witness = (
                association.source_boundary_witness_id
                if association.target_reference_pieces[row] is not None
                else None
            )
            interval = association.target_parameter_intervals[row]
            source_interval = association.source_parameter_intervals[row]
            facet_edge = (
                mesh.topological_dimension >= 1
                and association.target_entity_set_id == mesh.entity_set(1).entity_set_id
                and association.source_roles[row] is GeometrySourceEntityRole.FACET
            )
            edge_face = (
                mesh.topological_dimension >= 2
                and association.target_entity_set_id == mesh.entity_set(2).entity_set_id
                and association.source_roles[row] is GeometrySourceEntityRole.EDGE
                and piece_witness is not None
            )
            previous_rows = resolved_sources.setdefault(key, [])
            role = association.source_roles[row]
            curve_vertex = is_curve_vertex_closure(mesh, association, row)
            for (
                previous_identity,
                previous_witness,
                previous_interval,
                previous_facet_edge,
                previous_role,
                previous_curve_vertex,
                previous_edge_face,
                previous_source_interval,
            ) in previous_rows:
                if edge_face or previous_edge_face:
                    if (
                        (
                            edge_face and previous_edge_face and previous_identity == identity
                            and (
                                source_interval is None or previous_source_interval is None
                                or max(source_interval[0], previous_source_interval[0])
                                < min(source_interval[1], previous_source_interval[1])
                            )
                        )
                        or (not edge_face and (role is not GeometrySourceEntityRole.FACET or piece_witness is None))
                        or (not previous_edge_face and (previous_role is not GeometrySourceEntityRole.FACET or previous_witness is None))
                    ):
                        issues.append("conflicting_geometry_association")
                    continue
                if facet_edge or previous_facet_edge:
                    if (
                        (
                            facet_edge
                            and previous_facet_edge
                            and previous_identity == identity
                        )
                        or role
                        not in (
                            GeometrySourceEntityRole.EDGE,
                            GeometrySourceEntityRole.FACET,
                        )
                        or previous_role
                        not in (
                            GeometrySourceEntityRole.EDGE,
                            GeometrySourceEntityRole.FACET,
                        )
                        or (role is GeometrySourceEntityRole.FACET and not facet_edge)
                        or (
                            previous_role is GeometrySourceEntityRole.FACET
                            and not previous_facet_edge
                        )
                    ):
                        issues.append("overlapping_geometry_association")
                    continue
                if interval is not None and previous_interval is not None:
                    if max(interval[0], previous_interval[0]) < min(
                        interval[1], previous_interval[1]
                    ):
                        issues.append("overlapping_geometry_association")
                elif piece_witness is not None and previous_witness == piece_witness:
                    if previous_identity == identity:
                        issues.append("overlapping_geometry_association")
                elif (
                    interval is not None
                    or previous_interval is not None
                    or piece_witness is not None
                    or previous_witness is not None
                ):
                    issues.append("overlapping_geometry_association")
                elif previous_identity != identity and not (
                    curve_vertex and previous_curve_vertex
                ):
                    issues.append("conflicting_geometry_association")
            previous_rows.append(
                (identity, piece_witness, interval, facet_edge, role, curve_vertex,
                 edge_face, source_interval)
            )
    return tuple(dict.fromkeys(issues))


def _facet_edge_proof_tables(
    mesh: CellMesh,
    associations: tuple[GeometryAssociation, ...],
    association_source: Any,
    /,
) -> set[str]:
    """Find facet-edge closure claims needing independent incidence replay."""
    if mesh.topological_dimension < 1:
        return set()
    from .providers._native_sources import NativeLayerCoreSource

    authored = None
    if type(association_source) is NativeLayerCoreSource:
        authored = (
            association_source
            if association_source.reference_source is None
            else association_source.reference_source
        )
    edge_set = mesh.entity_set(1).entity_set_id
    rows: dict[tuple[Any, ...], list[tuple[GeometryAssociation, int]]] = {}
    for association in associations:
        if association.target_entity_set_id != edge_set:
            continue
        for row, identifier in enumerate(association.target_global_ids):
            key = (
                association.association_kind,
                association.source_id,
                association.source_revision,
                int(identifier),
            )
            rows.setdefault(key, []).append((association, row))
    result: set[str] = set()
    for group in rows.values():
        for association, row in group:
            if association.source_roles[row] is not GeometrySourceEntityRole.FACET:
                continue
            if len(group) > 1 or (
                authored is not None
                and (association.source_id, association.source_revision)
                == (authored.source_id, authored.source_revision)
            ):
                result.add(association.association_id)
    return result


def _record_source_witness_issue(
    issues: list[str],
    stage: str,
    group: tuple[GeometryAssociation, ...] | list[GeometryAssociation],
    witness_id: str | None,
    reason: str,
    /,
) -> None:
    """Retain the precise source-witness failure and affected target scopes."""
    scopes: dict[str, set[int]] = {}
    for association in group:
        scopes.setdefault(association.target_entity_set_id, set()).update(
            np.asarray(association.target_global_ids, dtype=np.int64).tolist()
        )
    scope = tuple((key, tuple(sorted(values))) for key, values in sorted(scopes.items()))
    issues.append("association_source_witness")
    issues.append(
        "association_source_witness_detail:"
        + json.dumps(
            {"stage": stage, "witness_id": witness_id, "target_scopes": scope, "reason": reason},
            sort_keys=True, separators=(",", ":"),
        )
    )


def _validated_layer_wall_association_ids(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    associations: tuple[GeometryAssociation, ...],
    source: Any,
    witness: SourceBoundaryChartWitness,
    inputs: MeshCertificationInputs | None,
    root: LayerCoreRootGeometry | None,
    computed_root: ComputedLayerCoreRoot | None,
    computed_replay: _ComputedLayerAssociationReplay | None,
    /,
) -> set[str]:
    """Reprove original wall classifications against actual layer/root ancestry."""
    from ._layer_core_association_transfer import _semantic_rows
    from ._layer_source_association_replay import (
        _ComputedLayerAssociationReplay,
        replay_layer_source_associations,
    )
    from ._layer_wall_source_replay import replay_layer_wall_associations
    from .providers._native_sources import NativeLayerCoreSource

    if (
        type(source) is not NativeLayerCoreSource
        or inputs is None
        or inputs.mesh is not mesh
        or inputs.geometry is not geometry
        or inputs.feature_source is not source
        or inputs.layer_core_root_geometry is not root
    ):
        raise ValueError("Wall association replay requires its actual source and certification inputs.")
    if computed_replay is None:
        layer = replay_layer_source_associations(
            source, mesh, geometry, root, computed_root=computed_root
        )
    elif type(computed_replay) is _ComputedLayerAssociationReplay:
        layer = computed_replay.require(source, mesh, geometry, root)
    else:
        raise TypeError("Wall association replay requires its bound operation-local proof.")
    expected = replay_layer_wall_associations(
        source, mesh, geometry, witness, layer, inputs.fidelity_tolerance
    )
    expected_rows = _semantic_rows(mesh, expected)
    actual_rows = _semantic_rows(mesh, associations)
    if not expected_rows or any(actual_rows.get(key) != value for key, value in expected_rows.items()):
        raise ValueError("Wall source identity, occurrence, side, orientation or support changed.")
    covered = set()
    for association in associations:
        if all(
            (association.source_id, association.source_revision, association.association_kind,
             dimension, int(identifier)) in expected_rows
            for dimension in range(mesh.topological_dimension + 1)
            if association.target_entity_set_id == mesh.entity_set(dimension).entity_set_id
            for identifier in np.asarray(association.target_global_ids, dtype=np.int64).tolist()
        ) and any(
            association.target_entity_set_id == mesh.entity_set(dimension).entity_set_id
            for dimension in range(mesh.topological_dimension + 1)
        ):
            covered.add(association.association_id)
    return covered


def _validated_partial_face_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    associations: tuple[GeometryAssociation, ...],
    validated: tuple[ValidatedSourceFidelityWitness, ...],
    association_source: Any,
    layer_core_root_geometry: LayerCoreRootGeometry | None,
    certification_inputs: MeshCertificationInputs | None,
    computed_root: ComputedLayerCoreRoot | None,
    computed_replay: _ComputedLayerAssociationReplay | None,
    /,
) -> tuple[dict[str, set[int]], tuple[str, ...], set[str]]:
    """Match each face restriction to actual, independently validated chart rows."""
    tables: dict[str, list[GeometryAssociation]] = {}
    for association in associations:
        if (
            association.source_boundary_witness_id is not None
            and mesh.topological_dimension >= 2
            and association.target_entity_set_id == mesh.entity_set(2).entity_set_id
        ):
            tables.setdefault(association.source_boundary_witness_id, []).append(
                association
            )
    from .providers._native_sources import NativeLayerCoreSource

    if type(association_source) is NativeLayerCoreSource:
        for token in validated:
            if (
                type(token) is ValidatedSourceFidelityWitness
                and token.binds(mesh, geometry, token.source)
                and isinstance(token.witness, SourceBoundaryFaceImageWitness)
                and (token.witness.source_id, token.witness.source_revision)
                == (association_source.source_id, association_source.source_revision)
            ):
                tables.setdefault(token.witness.witness_id, [])
    covered: dict[str, set[int]] = {}
    wall_association_ids: set[str] = set()
    issues: list[str] = []
    selected_wall_query = None
    if type(association_source) is NativeLayerCoreSource and any(
        item.association_kind is GeometryAssociationKind.BREP
        for group in tables.values()
        for item in group
    ):
        from ._layer_curving import layer_wall_boundary_query

        wall_authority = (
            association_source
            if association_source.reference_source is None
            else association_source.reference_source
        )
        if wall_authority.source_wall_charts is not None:
            selected_wall_query = layer_wall_boundary_query(
                wall_authority.layers, wall_authority.source_wall_charts
            )
    for witness_id, group in tables.items():
        matches = [
            token
            for token in validated
            if type(token) is ValidatedSourceFidelityWitness
            and token.binds(mesh, geometry, token.source)
            and isinstance(
                token.witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
            )
            and token.witness.witness_id == witness_id
        ]
        matches = [
            token for index, token in enumerate(matches)
            if not any(
                prior.source is token.source and prior.witness is token.witness
                for prior in matches[:index]
            )
        ]
        if len(matches) != 1:
            _record_source_witness_issue(
                issues, "face_token", group, witness_id,
                f"ValueError: expected one bound token, found {len(matches)}",
            )
            continue
        token = matches[0]
        from .providers._native_sources import (
            NativeLayerCoreSource,
            NativeStructuredSource,
            NativeSweepSource,
        )

        if type(association_source) is NativeLayerCoreSource:
            authored = (
                association_source
                if association_source.reference_source is None
                else association_source.reference_source
            )
        elif type(association_source) in (NativeStructuredSource, NativeSweepSource):
            authored = association_source
            if (
                not isinstance(token.witness, SourceBoundaryChartWitness)
                or token.source is not association_source.fidelity_source
                or certification_inputs is None
                or certification_inputs.mesh is not mesh
                or certification_inputs.geometry is not geometry
                or certification_inputs.feature_source is not association_source
                or certification_inputs.domain is not association_source.domain
            ):
                _record_source_witness_issue(
                    issues, "face_binding", group, witness_id,
                    "ValueError: source, query, domain or current operands do not match",
                )
                continue
        else:
            _record_source_witness_issue(
                issues, "face_source", group, witness_id,
                f"TypeError: unsupported source type {type(association_source).__name__}",
            )
            continue
        from ..geometry._domain_boundary import MeshingDomainBoundarySource
        from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
        from ..geometry._mesh_certificates import _same_tree_content

        is_wall = (
            type(association_source) is NativeLayerCoreSource
            and isinstance(selected_wall_query, MeshingDomainBoundarySource)
            and isinstance(token.source, MeshingDomainBoundarySource)
            and isinstance(token.witness, SourceBoundaryChartWitness)
            and token.witness.source_query_id == selected_wall_query.source_scope_id
            and token.source.source_scope_id == selected_wall_query.source_scope_id
            and _same_tree_content(token.source, selected_wall_query)
            and bool(group)
            and all(item.association_kind is GeometryAssociationKind.BREP for item in group)
        )

        expected_kind = (
            GeometryAssociationKind.BREP if is_wall
            else GeometryAssociationKind.MAPPED_REFERENCE
            if type(association_source) is NativeSweepSource
            and isinstance(token.source, MappedReferenceBoundarySource)
            else GeometryAssociationKind.PIECEWISE_LINEAR
        )
        if any(
            association.association_kind is not expected_kind
            or association.target_entity_set_id != mesh.entity_set(2).entity_set_id
            or (association.source_id, association.source_revision)
            != (authored.source_id, authored.source_revision)
            for association in group
        ):
            _record_source_witness_issue(
                issues, "face_kind", group, witness_id,
                f"ValueError: expected {expected_kind.value} association bound to the source and target",
            )
            continue
        witness = token.witness
        if isinstance(witness, SourceBoundaryChartWitness):
            expected = source_boundary_association_piece_rows(witness)
        elif isinstance(witness, SourceBoundaryFaceImageWitness):
            expected = source_boundary_image_association_piece_rows(witness)
        else:
            raise ValueError("A face association requires a validated source witness.")
        expected_by_face: dict[int, set[tuple[int, str, tuple[str, ...]]]] = {}
        for key in expected:
            expected_by_face.setdefault(key[0], set()).add(key)
        observed: dict[int, set[tuple[int, str, tuple[str, ...]]]] = {}
        duplicate = False
        try:
            if is_wall:
                if not isinstance(witness, SourceBoundaryChartWitness):
                    raise TypeError("Authored wall replay requires its exact chart witness.")
                checked_wall = _validated_layer_wall_association_ids(
                    mesh, geometry, associations, association_source, witness,
                    certification_inputs, layer_core_root_geometry, computed_root, computed_replay,
                )
                if any(item.association_id not in checked_wall for item in group):
                    raise ValueError("Wall source face table differs from the independently replayed wall.")
                wall_association_ids.update(checked_wall)
            elif isinstance(witness, SourceBoundaryFaceImageWitness):
                from ._certification_inputs import MeshCertificationInputs
                from ._layer_core_boundary_replay import (
                    _validate_partial_image_association_rows,
                )

                if (
                    type(certification_inputs) is not MeshCertificationInputs
                    or certification_inputs.mesh is not mesh
                    or certification_inputs.geometry is not geometry
                    or certification_inputs.feature_source is not association_source
                    or certification_inputs.layer_core_root_geometry is not layer_core_root_geometry
                    or certification_inputs.domain is None
                    or certification_inputs.cell_regions is None
                ):
                    raise ValueError("Image support requires its current material assignments.")
                _validate_partial_image_association_rows(
                    mesh, geometry, token, associations, association_source,
                    layer_core_root_geometry, certification_inputs,
                    computed_root=computed_root,
                    computed_replay=computed_replay,
                )
            elif any(
                role is GeometrySourceEntityRole.FACET
                for association in group for role in association.source_roles
            ):
                if type(association_source) in (NativeStructuredSource, NativeSweepSource):
                    from ._layer_core_association_transfer import _semantic_rows
                    from .providers._native_structured import _source_face_associations

                    expected_associations = _source_face_associations(
                        association_source, mesh, witness
                    )
                    if _semantic_rows(mesh, tuple(group)) != _semantic_rows(mesh, expected_associations):
                        raise ValueError("Sweep source face support, sides or orientation changed.")
                else:
                    from ._layer_core_boundary_replay import (
                        validate_partial_association_rows,
                    )

                    validate_partial_association_rows(
                        mesh, geometry, token, associations, association_source
                    )
            for association in group:
                for identifier, source, path, pieces, role in zip(
                    association.target_global_ids,
                    association.source_entity_ids,
                    association.source_occurrence_paths,
                    association.target_reference_pieces,
                    association.source_roles,
                    strict=True,
                ):
                    if role is not GeometrySourceEntityRole.FACET:
                        continue
                    face = int(identifier)
                    key = (face, source, tuple(path))
                    if expected.get(key) != pieces:
                        raise ValueError(
                            "Association changes its validated chart pieces."
                        )
                    keys = observed.setdefault(face, set())
                    if key in keys:
                        duplicate = True
                    keys.add(key)
        except (TypeError, ValueError) as error:
            _record_source_witness_issue(
                issues, "face_replay", group, witness_id,
                f"{type(error).__name__}: {error}",
            )
            continue
        if duplicate:
            issues.append("overlapping_geometry_association")
            continue
        for face, keys in observed.items():
            if keys == expected_by_face.get(face):
                identifiers = covered.setdefault(mesh.entity_set(2).entity_set_id, set())
                if face in identifiers:
                    issues.append("overlapping_geometry_association")
                identifiers.add(face)
            else:
                issues.append("incomplete_geometry_association")
    return covered, tuple(dict.fromkeys(issues)), wall_association_ids


def _validated_partial_edge_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    associations: tuple[GeometryAssociation, ...],
    validated: tuple[ValidatedSourceFidelityWitness, ...],
    association_source: Any,
    facet_edge_proofs: set[str],
    layer_core_root_geometry: LayerCoreRootGeometry | None,
    computed_root: ComputedLayerCoreRoot | None,
    computed_replay: _ComputedLayerAssociationReplay | None,
    image_preparations: list[_LayerImageSupportPreparation],
    /,
) -> tuple[dict[str, set[int]], tuple[str, ...]]:
    """Replay partial source edges, then prove an exact target interval partition."""
    groups: dict[tuple[str, str, str | None], list[GeometryAssociation]] = {}
    for association in associations:
        if (
            association.association_id in facet_edge_proofs
            or any(value is not None for value in association.target_parameter_intervals)
            or (
                association.source_boundary_witness_id is not None
                and association.target_entity_set_id == mesh.entity_set(1).entity_set_id
            )
        ):
            groups.setdefault(
                (
                    association.source_id,
                    association.source_revision,
                    association.source_boundary_witness_id,
                ),
                [],
            ).append(association)
    covered: dict[str, set[int]] = {}
    issues: list[str] = []
    intervals: dict[tuple[str, str, str, int], list[Any]] = {}
    for (source_id, source_revision, witness_id), group in groups.items():
        source_key = (source_id, source_revision)
        tokens = [
            token
            for token in validated
            if type(token) is ValidatedSourceFidelityWitness
            and token.binds(mesh, geometry, token.source)
            and isinstance(
                token.witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
            )
            and (token.witness.source_id, token.witness.source_revision) == source_key
            and (
                token.witness.witness_id == witness_id
                if witness_id is not None
                else token.source is getattr(association_source, "fidelity_source", None)
            )
        ]
        tokens = [
            token for index, token in enumerate(tokens)
            if not any(
                prior.source is token.source and prior.witness is token.witness
                for prior in tokens[:index]
            )
        ]
        if len(tokens) != 1:
            _record_source_witness_issue(
                issues, "edge_token", group, witness_id,
                f"ValueError: expected one bound token, found {len(tokens)}",
            )
            continue
        from .providers._native_sources import NativeLayerCoreSource

        if type(association_source) is not NativeLayerCoreSource:
            _record_source_witness_issue(
                issues, "edge_source", group, witness_id,
                f"TypeError: unsupported source type {type(association_source).__name__}",
            )
            continue
        authored = (
            association_source
            if association_source.reference_source is None
            else association_source.reference_source
        )
        if source_key != (authored.source_id, authored.source_revision) or any(
            association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or association.target_entity_set_id != mesh.entity_set(1).entity_set_id
            or (
                association.source_boundary_witness_id is not None
                and association.source_boundary_witness_id != tokens[0].witness.witness_id
            )
            for association in group
        ):
            _record_source_witness_issue(
                issues, "edge_binding", group, witness_id,
                "ValueError: edge kind, source identity or witness binding differs",
            )
            continue
        try:
            from ._layer_core_boundary_replay import (
                _validate_partial_edge_association_rows,
            )

            _validate_partial_edge_association_rows(
                mesh,
                geometry,
                tokens[0],
                associations,
                association_source,
                validated,
                layer_core_root_geometry,
                computed_root=computed_root,
                computed_replay=computed_replay,
                image_preparations=image_preparations,
            )
        except (TypeError, ValueError) as error:
            _record_source_witness_issue(
                issues, "edge_replay", group, witness_id,
                f"{type(error).__name__}: {error}",
            )
            continue
        for association in group:
            for identifier, interval in zip(
                association.target_global_ids,
                association.target_parameter_intervals,
                strict=True,
            ):
                if interval is not None:
                    intervals.setdefault(
                        (
                            source_id,
                            source_revision,
                            association.target_entity_set_id,
                            int(identifier),
                        ),
                        [],
                    ).append(interval)
    from ._layer_core_strata_replay import require_interval_partition

    for (_, _, entity_set_id, identifier), pieces in intervals.items():
        # Distinct proven facet closures may share an edge with its curve.
        # Their supports are a union, not additional whole-edge locks.
        union = []
        for first, last in sorted(pieces):
            if union and first <= union[-1][1]:
                union[-1] = (union[-1][0], max(union[-1][1], last))
            else:
                union.append((first, last))
        try:
            require_interval_partition(union)
        except ValueError:
            # A legitimate unprotected source stratum may occupy only part
            # of a mesh edge; it cannot count as whole-edge coverage.
            continue
        else:
            covered.setdefault(entity_set_id, set()).add(identifier)
    return covered, tuple(dict.fromkeys(issues))


def _complete_association_coverage(
    mesh: Any,
    boundary: Any,
    patches: Any,
    associations: Any,
    validated_support_coverage: dict[str, set[int]],
    /,
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
    for entity_set_id, identifiers in validated_support_coverage.items():
        coverage.setdefault(entity_set_id, set()).update(identifiers)
    for association in associations:
        entity_set_id = association.target_entity_set_id
        identifiers = coverage.setdefault(entity_set_id, set())
        for row, value in enumerate(np.asarray(association.target_global_ids)):
            if (
                association.target_parameter_intervals[row] is None
                and association.target_reference_pieces[row] is None
            ):
                identifiers.add(int(value))
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
    validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...] = (),
    association_source: Any = None,
    layer_core_root_geometry: LayerCoreRootGeometry | None = None,
    certification_inputs: MeshCertificationInputs | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
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
    Partial layer/core source associations additionally require the current
    validated fidelity witness and original ``association_source``; exact
    occurrence, side, orientation and target support are independently replayed.
    Physical face-image support additionally requires the actual
    ``certification_inputs`` and root owner so incident cell materials and
    exact source-side orientation are checked independently of the claims.
    Partial intervals contribute whole-edge coverage only when they form an exact
    partition. Other partial-source authorities are not inferred from identifiers.
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
        curve_vertex_closure_issues(mesh, geometry, associations, validated_fidelity)
    )
    supported_piece_sets = {
        mesh.entity_set(dimension).entity_set_id
        for dimension in (1, 2)
        if dimension <= mesh.topological_dimension
    }
    if any(
        value.source_boundary_witness_id is not None
        and value.target_entity_set_id not in supported_piece_sets
        for value in associations
    ):
        _record_source_witness_issue(
            issues, "target_scope",
            tuple(value for value in associations if value.source_boundary_witness_id is not None
                  and value.target_entity_set_id not in supported_piece_sets),
            None, "ValueError: witness-bound target entity set is unsupported",
        )
    facet_edge_proofs = _facet_edge_proof_tables(mesh, associations, association_source)
    from .providers._native_sources import NativeLayerCoreSource
    computed_replay = None
    image_preparations: list[_LayerImageSupportPreparation] = []
    if computed_root is not None:
        from ._layer_core_root_geometry import ComputedLayerCoreRoot

        if (
            type(computed_root) is not ComputedLayerCoreRoot
            or type(association_source) is not NativeLayerCoreSource
            or layer_core_root_geometry is None
        ):
            raise ValueError("Computed root requires its exact layer/core source and root.")
        computed_root.require(
            association_source, layer_core_root_geometry.specification,
            layer_core_root_geometry.mesh, layer_core_root_geometry.geometry,
            layer_core_root_geometry,
        )
        from ._layer_source_association_replay import _ComputedLayerAssociationReplay

        try:
            computed_replay = _ComputedLayerAssociationReplay(
                association_source, mesh, geometry, layer_core_root_geometry,
                computed_root=computed_root,
            )
        except (TypeError, ValueError) as error:
            _record_source_witness_issue(
                issues, "layer_replay", associations, None, f"{type(error).__name__}: {error}"
            )

    if (
        type(association_source) is NativeLayerCoreSource
        and association_source.conforming_execution
    ):
        from ._layer_core_root_geometry import LayerCoreRootGeometry

        if (
            type(layer_core_root_geometry) is not LayerCoreRootGeometry
            or layer_core_root_geometry.source_binding_id != association_source.binding_id
            or layer_core_root_geometry.specification_id
            != layer_core_root_geometry.specification.specification_id
        ):
            issues.append("association_layer_root_geometry")

        if association_source.cap_refinement == "conforming":
            from ._layer_core_boundary_replay import require_scoped_cap_coverage

            try:
                require_scoped_cap_coverage(
                    mesh, geometry, validated_fidelity, association_source
                )
            except (TypeError, ValueError):
                issues.append("association_cap_source_coverage")
    if type(association_source) is NativeLayerCoreSource:
        from ._layer_core_boundary_replay import _validate_source_edge_face_overlays

        chart_tokens = [
            token for token in validated_fidelity
            if type(token) is ValidatedSourceFidelityWitness
            and token.binds(mesh, geometry, token.source)
            and (
                isinstance(token.witness, SourceBoundaryFaceImageWitness)
                or (
                    isinstance(token.witness, SourceBoundaryChartWitness)
                    and token.witness.chart_dimension == 2
                )
            )
            and (token.witness.source_id, token.witness.source_revision)
            == (association_source.source_id, association_source.source_revision)
        ]
        if chart_tokens:
            token = next(
                (value for value in chart_tokens
                 if value.source is association_source.fidelity_source),
                chart_tokens[0],
            )
            try:
                _validate_source_edge_face_overlays(
                    mesh, geometry, token, associations, association_source,
                    validated_fidelity, layer_core_root_geometry,
                    computed_root=computed_root,
                    computed_replay=computed_replay,
                    image_preparations=image_preparations,
                )
            except (TypeError, ValueError) as error:
                _record_source_witness_issue(
                    issues, "edge_face_overlay", associations,
                    token.witness.witness_id, f"{type(error).__name__}: {error}",
                )
    support_coverage, face_issues, wall_association_ids = _validated_partial_face_coverage(
        mesh, geometry, associations, validated_fidelity, association_source,
        layer_core_root_geometry, certification_inputs, computed_root, computed_replay,
    )
    issues.extend(face_issues)
    facet_edge_proofs.difference_update(wall_association_ids)
    edge_coverage, edge_issues = _validated_partial_edge_coverage(
        mesh,
        geometry,
        associations,
        validated_fidelity,
        association_source,
        facet_edge_proofs,
        layer_core_root_geometry,
        computed_root,
        computed_replay,
        image_preparations,
    )
    issues.extend(edge_issues)
    for entity_set_id, identifiers in edge_coverage.items():
        support_coverage.setdefault(entity_set_id, set()).update(identifiers)
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
    quality_report = summarize_cell_quality(
        mesh,
        quality_evaluation,
        minimum_measure=audit_policy.minimum_measure,
        minimum_mean_ratio=audit_policy.minimum_mean_ratio,
        maximum_aspect_ratio=audit_policy.maximum_aspect_ratio,
    )
    issues.extend(quality_report.failed_gates)
    if audit_policy.require_complete_association and not _complete_association_coverage(
        mesh, boundary, patches, associations, support_coverage
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
        policy=audit_policy,
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
