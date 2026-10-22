#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Audited mesh-to-geometry associations and their B-Rep classification rules.

A B-Rep association classifies mesh entities on B-Rep entities of one declared
source revision: every row names the entity's dimension and index, the closest
point parameters, the projection residual, ambiguity, orientation relation, and
parent provenance. Classification of mesh edges, faces, and cells from vertex
classes follows one rule set shared by adaptation, curving, and re-derivation:

* a mesh cell lies in the unique region (solid, or face for two-dimensional
  meshes) whose closure contains every vertex class;
* a mesh entity interior to the mesh with one adjacent region lies in that region;
* a mesh entity on the mesh boundary or on a region interface lies on the
  lowest-dimensional B-Rep entity containing every vertex class in its closure
  and lying in the closure of every adjacent region; ties are ambiguous.

Through topology changes (:func:`propagate_association`) preserved vertices
(including B-Rep corner vertices) keep their class, relocated vertices keep it
and are re-projected, children of a split inherit the class of the source mesh
entity they were created on (edge children inherit its edge, face-interior
children its face, and cell-interior children its region) and are projected.
Interior children with only cell refinement lineage inherit the uniquely
owning source cell; geometrically classified volume-interior orphans locate
their solid when no lower-dimensional source stratum is present. A collapse is
legal only when the kept vertex's class lies in the closure of the removed
vertex's class; illegal collapses raise :class:`AssociationPropagationError`.
Swaps change no vertex
class. Unknown lineage (provider remeshing) is re-derived by classification
transfer and projection (:func:`rederive_association`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, final, Literal, NamedTuple, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax.discretization._cell_mesh as cell_mesh_owner
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    MaterialChartDeformation,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    exact_orient2d,
    MeshcoreStatus,
    point_triangle_locations,
)
from .._physical import SpatialCoordinateContract
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, positive_integer
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_predecessor import (
    CellGeometryPredecessor,
    prepare_cell_geometry_predecessor,
)
from ..discretization._cell_geometry_transfer import nested_edge_midpoints_admitted
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
)
from ..discretization._exact_plc_geometry import (
    ExactPlcCellGeometrySource,
    translated_plc_layout_matches,
)
from ..geometry._domain_coverage import (
    DomainCoverageCertificate,
    DomainCoveragePredecessor,
    PiecewiseLinearDomain,
)
from ..geometry._global_embedding import GlobalEmbeddingCertificate, ray_crossings_3d
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._periodic_restriction import PeriodicRestrictionPremise
from ..geometry._surface_source_support import (
    PreparedSurfaceSourceSupport,
    SurfaceSourceReceipts,
)
from ..geometry.brep._projection_contracts import (
    AbstractBRepProjection,
    brep_entity_id,
    BRepProjectionResult,
    BRepProjectionStatus,
)
from ..typing import (
    checked,
    Dim,
    HostFloat64,
    HostInt64,
    parse,
    Scope,
)

# Module references defer symbol resolution across mutually dependent execution owners.
from . import (
    _association_state as association_state_owner,
    _certification as certification_owner,
    _certification_inputs as certification_inputs_owner,
    _lineage as lineage_owner,
    _lineage_records as lineage_records_owner,
    _mapped_reference_association as mapped_reference_association_owner,
    _organization_records as organization_records_owner,
    _plc_mapped_support as plc_mapped_support_owner,
    _plc_planar_support as plc_planar_support_owner,
    _result as result_owner,
    _surface_association_transfer as surface_association_transfer_owner,
)
from ._association_routes import (
    association_for_dimension,
    association_lookup,
    AssociationPropagationError,
    boundary_mask,
    child_sources,
    compose_incidence_pairs,
    incidence_pairs,
    ordered_vertices,
    PlcEntityClasses,
    require_transfer_lineage,
    surface_source_association_kind,
)
from ._association_state import (
    entity_rows,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
    target_dimension,
)
from ._contracts import MeshingLimits
from ._host_banks import entity_key
from ._scope import resolve_mesh_scope


if TYPE_CHECKING:
    from ..geometry._mapped_source_support import MappedSourceFacetAuthority


def retain_association_scopes(
    originals: tuple[GeometryAssociation, ...],
    proved: tuple[GeometryAssociation, ...],
    /,
) -> tuple[GeometryAssociation, ...]:
    """Retain fixed-topology scientific scopes after fresh support proof for every row."""
    result = []
    for original in originals:
        matches = tuple(
            value
            for value in proved
            if (
                value.association_kind,
                value.source_id,
                value.source_revision,
                value.target_entity_set_id,
            )
            == (
                original.association_kind,
                original.source_id,
                original.source_revision,
                original.target_entity_set_id,
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"GeometryAssociation.motion_support must provide one fresh table per accepted scope; got matching tables {len(matches)} for association {original.association_id!r}"
            )
        fresh = matches[0]
        if (
            np.array_equal(fresh.target_global_ids, original.target_global_ids)
            and fresh.source_entities.occurrence_paths()
            == original.source_entities.occurrence_paths()
            and fresh.parent_association_id == original.association_id
        ):
            result.append(fresh)
            continue
        rows = fresh.target_rows(original.target_global_ids)
        indexed = fresh.indexed_source_strata
        result.append(
            GeometryAssociation(
                fresh.association_kind,
                fresh.source_id,
                fresh.source_revision,
                fresh.target_entity_set_id,
                original.target_global_ids,
                tuple(fresh.source_entities.entity_id(row) for row in rows),
                fresh.residuals[rows],
                resolved=fresh.resolved[rows],
                ambiguous=fresh.ambiguous[rows],
                exact=fresh.exact,
                source_dimensions=fresh.source_dimensions[rows] if indexed else None,
                source_indices=fresh.source_indices[rows] if indexed else None,
                source_occurrence_paths=original.source_entities.occurrence_paths(),
                source_entity_roles=(
                    None
                    if not fresh.source_entities.indexed_roles
                    else tuple(fresh.source_entities.role(row) for row in rows)
                ),
                parameters=fresh.parameters[rows],
                orientations=fresh.orientations[rows],
                parent_dimensions=fresh.parent_dimensions[rows],
                parent_ids=fresh.parent_ids[rows],
                parent_association_id=original.association_id,
                provenance=fresh.provenance,
            )
        )
    return tuple(result)


# -- policy and errors ---------------------------------------------------------


class AssociationPropagationPolicy(StrictModule, NonTrainableState):
    """Tolerances and standalone query allowance of B-Rep association work.

    ``classification_tolerance`` is the absolute distance within which geometric
    classification places a point on a B-Rep entity. ``maximum_residual`` marks
    rows whose projection residual exceeds it unresolved; ``None`` keeps residuals
    as evidence only. ``maximum_projection_queries`` bounds the source-facing
    projection, classification and solid-location query points of one call made
    without a caller budget (default: the ``MeshingLimits`` geometry-query
    allowance); a supplied request budget charges its own ticket and caps
    instead.
    """

    classification_tolerance: float = eqx.field(static=True)
    maximum_residual: float | None = eqx.field(static=True)
    maximum_projection_queries: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        classification_tolerance: float = 1.0e-6,
        maximum_residual: float | None = None,
        maximum_projection_queries: int = MeshingLimits().maximum_geometry_queries,
    ) -> None:
        tolerance = float(classification_tolerance)
        maximum = None if maximum_residual is None else float(maximum_residual)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                f"AssociationPropagationPolicy.classification_tolerance must be finite and positive; got {tolerance!r}"
            )
        if maximum is not None and (not np.isfinite(maximum) or maximum < 0.0):
            raise ValueError(
                f"AssociationPropagationPolicy.maximum_residual must be finite and nonnegative or None; got {maximum!r}"
            )
        queries = positive_integer(
            maximum_projection_queries,
            "AssociationPropagationPolicy.maximum_projection_queries",
        )
        self.classification_tolerance = tolerance
        self.maximum_residual = maximum
        self.maximum_projection_queries = queries
        self.policy_id = canonical_fingerprint(
            {
                "kind": "association-propagation-policy",
                "classification_tolerance": tolerance,
                "maximum_residual": maximum,
                "maximum_projection_queries": queries,
            }
        )


def _vertex_rows(
    mesh: cell_mesh_owner.CellMesh, identifiers: np.ndarray, /
) -> np.ndarray:
    """Mesh vertex row of each global ID (-1 when absent)."""
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    position = np.minimum(np.searchsorted(vertex_ids[order], identifiers), order.size - 1)
    return np.where(vertex_ids[order[position]] == identifiers, order[position], -1)


# -- classification --------------------------------------------------------------


RESOLVED_STATUS = (int(BRepProjectionStatus.UNIQUE), int(BRepProjectionStatus.SEAM))


class _Classes(NamedTuple):
    """B-Rep class of every mesh entity row of one dimension (-1 unclassified)."""

    dimensions: np.ndarray
    indices: np.ndarray
    status: np.ndarray
    occurrence_paths: tuple[tuple[str, ...], ...] = ()

    @property
    def paths(self) -> tuple[tuple[str, ...], ...]:
        return self.occurrence_paths or ((),) * self.dimensions.size

    @property
    def resolved(self) -> np.ndarray:
        return np.isin(self.status, RESOLVED_STATUS)


def _vertex_classes(
    mesh: cell_mesh_owner.CellMesh, association: GeometryAssociation, /
) -> _Classes:
    """Vertex classes in mesh row order; the association must cover every vertex."""
    rows = entity_rows(mesh, 0, np.asarray(association.target_global_ids))
    count = mesh.coordinates.shape[0]
    if np.any(rows < 0) or np.unique(rows).size != count or rows.size != count:
        raise ValueError(
            f"GeometryAssociation.target_global_ids must cover every mesh vertex once; got row routes {rows!r}, vertex count {count}"
        )
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    dims[rows] = np.asarray(association.source_dimensions, dtype=np.int64)
    indices[rows] = np.asarray(association.source_indices, dtype=np.int64)
    status[rows] = np.where(
        np.asarray(association.resolved),
        BRepProjectionStatus.UNIQUE,
        np.where(
            np.asarray(association.ambiguous),
            BRepProjectionStatus.AMBIGUOUS,
            BRepProjectionStatus.FAILED,
        ),
    )
    paths: list[tuple[str, ...]] = [()] * count
    for row, path in zip(
        rows.tolist(), association.source_entities.occurrence_paths(), strict=True
    ):
        paths[row] = path
    return _Classes(dims, indices, status, tuple(paths))


def centroids(
    mesh: cell_mesh_owner.CellMesh, members: np.ndarray, count: int, /
) -> np.ndarray:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    totals = np.zeros((count, points.shape[1]))
    np.add.at(totals, members[:, 0], points[members[:, 1]])
    return totals / np.maximum(np.bincount(members[:, 0], minlength=count), 1)[:, None]


def _candidate_residuals(
    projection: AbstractBRepProjection,
    points: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    occurrence_paths: tuple[tuple[str, ...], ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> np.ndarray:
    """Distance of each point to its candidate entity (solids: 0 inside, else inf)."""
    residuals = np.full((points.shape[0],), np.inf)
    lower = np.flatnonzero(dimensions <= 2)
    if lower.size:
        budget.reserve_geometry_queries(lower.size)
        result = projection.project(
            points[lower],
            dimensions[lower],
            indices[lower],
            occurrence_paths=tuple(occurrence_paths[row] for row in lower),
            budget=budget,
        )
        residuals[lower] = np.where(
            np.asarray(result.status) == BRepProjectionStatus.FAILED,
            np.inf,
            np.asarray(result.residuals),
        )
    solid = np.flatnonzero(dimensions == 3)
    if solid.size:
        budget.reserve_geometry_queries(solid.size)
        located = projection.locate_solids(
            points[solid], occurrence_paths=tuple(occurrence_paths[row] for row in solid)
        )
        inside = (np.asarray(located.dimensions) == 3) & (
            np.asarray(located.indices) == indices[solid]
        )
        inside &= np.asarray(
            [
                path == occurrence_paths[row]
                for row, path in zip(solid, located.source_occurrence_paths, strict=True)
            ]
        )
        residuals[solid] = np.where(inside, 0.0, np.inf)
    return residuals


def _lowest_containers(
    projection: AbstractBRepProjection,
    members: np.ndarray,
    member_classes: _Classes,
    count: int,
    low: int,
    high: np.ndarray,
    within: np.ndarray | None,
    centroids: np.ndarray,
    budget: CoordinateEnclosureBudget,
    /,
) -> _Classes:
    """Lowest-dimensional B-Rep entity containing every member class of each owner.

    ``members`` are ``(owner, member_row)`` pairs. Candidates have dimension in
    ``[low, high[owner]]`` and lie in the closure of every ``within`` entity code
    listed for their owner. Equal-dimension ties are broken by the owner
    centroid's distance to each candidate; indistinguishable ties are AMBIGUOUS.
    """
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    paths: list[tuple[str, ...]] = [()] * count
    member_dims = member_classes.dimensions[members[:, 1]]
    classified = member_dims >= 0
    failed = np.bincount(members[~classified, 0], minlength=count) > 0
    failed |= np.bincount(members[:, 0], minlength=count) == 0
    codes = projection.entity_codes(
        member_dims[classified],
        member_classes.indices[members[classified, 1]],
        occurrence_paths=tuple(
            member_classes.paths[row] for row in members[classified, 1]
        ),
    )
    owner_class = np.unique(
        np.stack((members[classified, 0], codes), axis=1), axis=0
    ).reshape(-1, 2)
    if owner_class.shape[0] == 0:
        return _Classes(dims, indices, status)
    class_count = np.bincount(owner_class[:, 0], minlength=count)
    distinct, inverse = np.unique(owner_class[:, 1], return_inverse=True)
    distinct_paths = projection.code_occurrence_paths(distinct)
    row, container_dims, container_indices, container_paths = projection.containers(
        *projection.code_entities(distinct), occurrence_paths=distinct_paths
    )
    keep = container_dims >= low
    reach = compose_incidence_pairs(
        np.stack((owner_class[:, 0], inverse.reshape(-1)), axis=1),
        np.stack(
            (
                row[keep],
                projection.entity_codes(
                    container_dims[keep],
                    container_indices[keep],
                    occurrence_paths=tuple(
                        container_paths[row_] for row_ in np.flatnonzero(keep)
                    ),
                ),
            ),
            axis=1,
        ),
        unique=False,
    )
    candidates, hits = np.unique(reach, axis=0, return_counts=True)
    candidates = candidates.reshape(-1, 2)
    candidate_dims, candidate_indices = projection.code_entities(candidates[:, 1])
    admissible = (hits == class_count[candidates[:, 0]]) & (
        candidate_dims <= high[candidates[:, 0]]
    )
    if within is not None and candidates.shape[0]:
        joined = compose_incidence_pairs(
            np.stack((np.arange(candidates.shape[0]), candidates[:, 0]), axis=1),
            within,
            unique=False,
        )
        container_dims_, container_indices_ = projection.code_entities(joined[:, 1])
        inside = projection.contains(
            container_dims_,
            container_indices_,
            candidate_dims[joined[:, 0]],
            candidate_indices[joined[:, 0]],
            occurrence_paths=projection.code_occurrence_paths(joined[:, 1]),
            member_occurrence_paths=projection.code_occurrence_paths(
                candidates[joined[:, 0], 1]
            ),
        )
        admissible[joined[~inside, 0]] = False
    candidates = candidates[admissible]
    candidate_dims = candidate_dims[admissible]
    candidate_indices = candidate_indices[admissible]
    if candidates.shape[0]:
        lowest = np.full((count,), 4, dtype=np.int64)
        np.minimum.at(lowest, candidates[:, 0], candidate_dims)
        at = candidate_dims == lowest[candidates[:, 0]]
        owners = candidates[at, 0]
        at_dims, at_indices = candidate_dims[at], candidate_indices[at]
        at_codes = candidates[at, 1]
        ties = np.bincount(owners, minlength=count)
        residual = np.zeros(owners.shape)
        tied = ties[owners] > 1
        if np.any(tied):
            residual[tied] = _candidate_residuals(
                projection,
                centroids[owners[tied]],
                at_dims[tied],
                at_indices[tied],
                projection.code_occurrence_paths(at_codes[tied]),
                budget,
            )
        order = np.lexsort((at_indices, residual, owners))
        owners, at_dims, at_indices, residual = (
            owners[order],
            at_dims[order],
            at_indices[order],
            residual[order],
        )
        at_codes = at_codes[order]
        first = np.unique(owners, return_index=True)[1]
        second = np.minimum(first + 1, owners.size - 1)
        distinguishable = (ties[owners[first]] == 1) | (
            (owners[second] == owners[first])
            & (residual[second] > residual[first] + projection.policy.ambiguity_tolerance)
        )
        rows = owners[first]
        dims[rows] = at_dims[first]
        indices[rows] = at_indices[first]
        for row_, path in zip(
            rows, projection.code_occurrence_paths(at_codes[first]), strict=True
        ):
            paths[row_] = path
        status[rows] = np.where(
            distinguishable, BRepProjectionStatus.UNIQUE, BRepProjectionStatus.AMBIGUOUS
        )
    dims[failed] = -1
    indices[failed] = -1
    status[failed] = BRepProjectionStatus.FAILED
    for row_ in np.flatnonzero(failed):
        paths[row_] = ()
    return _Classes(dims, indices, status, tuple(paths))


def _classify_mesh(
    mesh: cell_mesh_owner.CellMesh,
    vertex_classes: _Classes,
    projection: AbstractBRepProjection,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[_Classes, ...]:
    """B-Rep classes of every mesh entity dimension ``0..D`` (module rules)."""
    top = mesh.topological_dimension
    classes: list[_Classes] = [vertex_classes] * (top + 1)
    members = incidence_pairs(mesh, 0, top)[:, ::-1]
    count = mesh.entity_set(top).count
    classes[top] = _lowest_containers(
        projection,
        members,
        vertex_classes,
        count,
        top,
        np.full((count,), top),
        None,
        centroids(mesh, members, count),
        budget,
    )
    for dimension in range(top - 1, 0, -1):
        classes[dimension] = _classify_level(
            mesh, projection, vertex_classes, classes[dimension + 1], dimension, budget
        )
    return tuple(classes)


def _classify_level(
    mesh: cell_mesh_owner.CellMesh,
    projection: AbstractBRepProjection,
    vertex_classes: _Classes,
    upper: _Classes,
    dimension: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> _Classes:
    """Classify entities of ``dimension`` from their classified cofaces.

    The cofaces on the lowest-dimensional adjacent classes decide: one such class
    that the entity does not bound (a coface class of higher dimension, or exactly
    two cofaces on it) is inherited; otherwise the entity lies on the lowest
    B-Rep entity containing its vertex classes within the closure of every
    adjacent lowest class. A lone coface on one embedded (lower-than-ambient)
    class also admits that class itself, so a partial-patch boundary entity on
    no lower entity lies in the class interior under its occurrence-qualified
    code; full-dimensional regions never admit it.
    """
    count = mesh.entity_set(dimension).count
    pairs = incidence_pairs(mesh, dimension, dimension + 1)
    good = np.isin(upper.status[pairs[:, 1]], RESOLVED_STATUS)
    unresolved = np.bincount(pairs[~good, 0], minlength=count) > 0
    pairs = pairs[good]
    upper_dims = upper.dimensions[pairs[:, 1]]
    lowest = np.full((count,), 4, dtype=np.int64)
    np.minimum.at(lowest, pairs[:, 0], upper_dims)
    at = upper_dims == lowest[pairs[:, 0]]
    adjacent = np.unique(
        np.stack(
            (
                pairs[at, 0],
                projection.entity_codes(
                    upper_dims[at],
                    upper.indices[pairs[at, 1]],
                    occurrence_paths=tuple(upper.paths[row] for row in pairs[at, 1]),
                ),
            ),
            axis=1,
        ),
        axis=0,
    ).reshape(-1, 2)
    distinct = np.bincount(adjacent[:, 0], minlength=count)
    cofaces = np.bincount(pairs[at, 0], minlength=count)
    inherited = (
        (distinct == 1) & ((lowest > dimension + 1) | (cofaces == 2)) & ~unresolved
    )
    # Any lower entity carrying the mesh entity contains its vertex classes and
    # still wins by the lowest-class rule.
    interior = (
        (distinct == 1)
        & (cofaces == 1)
        & (lowest == dimension + 1)
        & (lowest < mesh.ambient_dimension)
        & ~unresolved
    )
    members = incidence_pairs(mesh, 0, dimension)[:, ::-1]
    # Only entities bounding their coface classes search for a lower class.
    bounding = members[~inherited[members[:, 0]]]
    bounded = _lowest_containers(
        projection,
        bounding,
        vertex_classes,
        count,
        dimension,
        np.where(interior, lowest, np.minimum(lowest, 4) - 1),
        adjacent,
        centroids(mesh, members, count),
        budget,
    )
    code = np.full((count,), -1, dtype=np.int64)
    code[adjacent[:, 0]] = adjacent[:, 1]
    inherited_dims, inherited_indices = projection.code_entities(np.maximum(code, 0))
    dims = np.where(inherited, inherited_dims, bounded.dimensions)
    indices = np.where(inherited, inherited_indices, bounded.indices)
    status = np.where(inherited, BRepProjectionStatus.UNIQUE, bounded.status)
    status = np.where(
        unresolved & ~inherited & (status != BRepProjectionStatus.FAILED),
        BRepProjectionStatus.AMBIGUOUS,
        status,
    ).astype(np.int8)
    inherited_paths = projection.code_occurrence_paths(np.maximum(code, 0))
    paths = tuple(
        inherited_paths[row] if inherited[row] else bounded.paths[row]
        for row in range(count)
    )
    return _Classes(dims.astype(np.int64), indices.astype(np.int64), status, paths)


def _orientations(
    mesh: cell_mesh_owner.CellMesh,
    dimension: int,
    classes: _Classes,
    projected: BRepProjectionResult,
    /,
) -> np.ndarray:
    """Relation of each entity's incidence-defined vertex order to its B-Rep orientation."""
    result = np.zeros((classes.dimensions.size,), dtype=np.int8)
    ordered = ordered_vertices(mesh, dimension)
    if ordered is None:
        return result
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if dimension == 1:
        rows = np.flatnonzero(classes.dimensions == 1)
        tangent = np.asarray(projected.tangents)[rows, 0]
        direction = points[ordered[rows, 1]] - points[ordered[rows, 0]]
        sign = np.sign(np.sum(direction * tangent, axis=1))
    elif dimension == 2 and mesh.ambient_dimension == 3:
        rows = np.flatnonzero(classes.dimensions == 2)
        normal = np.asarray(projected.normals)[rows]
        corners = points[ordered[rows]]
        facet = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        sign = np.sign(np.sum(facet * normal, axis=1))
    else:
        return result
    result[rows] = np.where(np.isfinite(sign), sign, 0).astype(np.int8)
    return result


def _brep_association(
    projection: AbstractBRepProjection,
    mesh: cell_mesh_owner.CellMesh,
    dimension: int,
    classes: _Classes,
    points: np.ndarray,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    provenance: GeometryAssociationProvenance,
    parent_association_id: str | None,
    parent_dimensions: np.ndarray | None = None,
    parent_ids: np.ndarray | None = None,
) -> GeometryAssociation:
    """Project classified entity points and assemble the audited association."""
    count = classes.dimensions.size
    entity_set = mesh.entity_set(dimension)
    # Solid rows inherit region membership; only vertices, edges, and faces project.
    lower = np.flatnonzero((classes.dimensions >= 0) & (classes.dimensions <= 2))
    budget.reserve_geometry_queries(lower.size)
    projected = projection.project(
        points[lower],
        classes.dimensions[lower],
        classes.indices[lower],
        occurrence_paths=tuple(classes.paths[row] for row in lower),
        budget=budget,
    )
    if projected.source_occurrence_paths != tuple(classes.paths[row] for row in lower):
        raise ValueError(
            f"GeometryAssociation.projected_occurrence_paths must retain classified paths; got {projected.source_occurrence_paths!r}, expected {tuple(classes.paths[row] for row in lower)!r}"
        )
    residuals = np.where(classes.dimensions == 3, 0.0, np.inf)
    parameters = np.full((count, 2), np.nan)
    projection_status = np.where(
        classes.dimensions == 3, BRepProjectionStatus.UNIQUE, BRepProjectionStatus.FAILED
    ).astype(np.int8)
    residuals[lower] = np.asarray(projected.residuals)
    parameters[lower] = np.asarray(projected.parameters)
    projection_status[lower] = np.asarray(projected.status)
    path_table = tuple(dict.fromkeys(path for path in classes.paths if path))
    path_rows = {path: row for row, path in enumerate(path_table)}
    full = BRepProjectionResult(
        projection.source_revision,
        classes.dimensions,
        classes.indices,
        points,
        np.full(points.shape, np.nan),
        parameters,
        residuals,
        projection_status,
        _scatter(lower, np.asarray(projected.normals), (count, points.shape[1])),
        _scatter(lower, np.asarray(projected.tangents), (count, 2, points.shape[1])),
        occurrence_indices=np.asarray(
            [path_rows[path] if path else -1 for path in classes.paths], dtype=np.int32
        ),
        occurrence_paths=path_table,
    )
    ambiguous_statuses = (
        BRepProjectionStatus.AMBIGUOUS,
        BRepProjectionStatus.NON_ISOLATED,
    )
    ambiguous = np.isin(classes.status, ambiguous_statuses) | np.isin(
        projection_status, ambiguous_statuses
    )
    resolved = (
        np.isin(classes.status, RESOLVED_STATUS)
        & np.isin(projection_status, RESOLVED_STATUS)
        & (
            residuals <= policy.maximum_residual
            if policy.maximum_residual is not None
            else np.isfinite(residuals)
        )
    )
    return GeometryAssociation(
        GeometryAssociationKind.BREP,
        projection.source_id,
        projection.source_revision,
        entity_set.entity_set_id,
        np.asarray(entity_set.entity_ids, dtype=np.int64),
        tuple(
            brep_entity_id(
                projection.source_revision, entity, index, occurrence_path=path
            )
            if entity >= 0
            else association_state_owner.unclassified_entity_id(
                projection.source_revision
            )
            for entity, index, path in zip(
                classes.dimensions.tolist(),
                classes.indices.tolist(),
                classes.paths,
                strict=True,
            )
        ),
        residuals,
        resolved=resolved,
        ambiguous=ambiguous & ~resolved,
        source_dimensions=classes.dimensions,
        source_indices=classes.indices,
        source_occurrence_paths=classes.paths,
        parameters=parameters,
        orientations=_orientations(mesh, dimension, classes, full),
        parent_dimensions=parent_dimensions,
        parent_ids=parent_ids,
        parent_association_id=parent_association_id,
        provenance=provenance,
    )


def _scatter(rows: np.ndarray, values: np.ndarray, shape: tuple[int, ...], /) -> Any:
    output = np.full(shape, np.nan)
    output[rows] = values
    return output


def _require_brep(
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    mesh: cell_mesh_owner.CellMesh,
    dimension: int,
    /,
) -> None:
    if association.association_kind is not GeometryAssociationKind.BREP:
        raise ValueError(
            f"GeometryAssociation.association_kind must be BREP for B-Rep classification; got {association.association_kind!r}"
        )
    if (
        association.source_revision != projection.source_revision
        or association.source_id != projection.source_id
    ):
        raise ValueError(
            f"GeometryAssociation.source_binding must match the projection source and revision; got {(association.source_id, association.source_revision)!r}, projection {(projection.source_id, projection.source_revision)!r}"
        )
    if association.target_entity_set_id != mesh.entity_set(dimension).entity_set_id:
        raise ValueError(
            f"GeometryAssociation.target_entity_set_id must bind the mesh stratum; got {association.target_entity_set_id!r}, expected {mesh.entity_set(dimension).entity_set_id!r}"
        )
    if mesh.ambient_dimension != projection.ambient_dimension:
        raise ValueError(
            f"GeometryAssociation.ambient_dimension must agree between mesh and projection; got mesh {mesh.ambient_dimension}, projection {projection.ambient_dimension}"
        )
    resolved = np.flatnonzero(np.asarray(association.resolved))
    projection.entity_codes(
        np.asarray(association.source_dimensions)[resolved],
        np.asarray(association.source_indices)[resolved],
        occurrence_paths=tuple(
            association.source_entities.occurrence_path(row) for row in resolved
        ),
    )


# -- public classification ----------------------------------------------------------


@checked
def mesh_entity_classes(
    mesh: cell_mesh_owner.CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[_Classes, ...]:
    """Host B-Rep classes ``(dimensions, indices, status)`` of mesh dimensions ``0..D``.

    ``association`` is the vertex association covering every vertex of ``mesh``. Consumers
    (adaptation constraints, curving) use the classes to protect, constrain, and
    project entities by the module rules. Source queries that break class ties
    are charged as in :func:`propagate_association`.
    """
    with support_budget(
        MeshCertificateLimits().maximum_work_units,
        policy.maximum_projection_queries,
        budget,
        maximum_memory_bytes=(
            MeshCertificateLimits().maximum_scratch_bytes
            if budget is None
            else budget.maximum_memory_bytes
        ),
    ) as budget:
        return _mesh_classes(mesh, association, projection, budget)


def _mesh_classes(
    mesh: cell_mesh_owner.CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[_Classes, ...]:
    _require_brep(association, projection, mesh, 0)
    return _classify_mesh(mesh, _vertex_classes(mesh, association), projection, budget)


@checked
def associate_mesh_entities(
    mesh: cell_mesh_owner.CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    dimension: int,
    /,
    *,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget | None = None,
) -> GeometryAssociation:
    """Derive the B-Rep association of mesh entities of ``dimension`` (1..D).

    Classes follow the module rules from the vertex ``association`` covering every vertex;
    residuals and parameters come from projecting each entity centroid (region
    rows carry zero residual), orientations from the incidence-defined vertex order.
    Source queries are charged as in :func:`propagate_association`.
    """
    with support_budget(
        MeshCertificateLimits().maximum_work_units,
        policy.maximum_projection_queries,
        budget,
        maximum_memory_bytes=(
            MeshCertificateLimits().maximum_scratch_bytes
            if budget is None
            else budget.maximum_memory_bytes
        ),
    ) as budget:
        return _entity_association(
            mesh, association, projection, dimension, policy, budget
        )


def _entity_association(
    mesh: cell_mesh_owner.CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    dimension: int,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget,
    /,
) -> GeometryAssociation:
    target = int(dimension)
    if not 1 <= target <= mesh.topological_dimension:
        raise ValueError(
            f"GeometryAssociation.dimension must lie in [1, D] for derived entities; got {target}, D {mesh.topological_dimension}"
        )
    classes = _mesh_classes(mesh, association, projection, budget)[target]
    members = incidence_pairs(mesh, 0, target)[:, ::-1]
    count = mesh.entity_set(target).count
    return _brep_association(
        projection,
        mesh,
        target,
        classes,
        centroids(mesh, members, count),
        policy,
        budget,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=association.association_id,
    )


def _geometric_vertex_classes(
    mesh: cell_mesh_owner.CellMesh,
    projection: AbstractBRepProjection,
    tolerance: float,
    groups: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]
    ]
    | None,
    budget: CoordinateEnclosureBudget,
    /,
) -> _Classes:
    """Lowest-dimensional B-Rep entity within ``tolerance`` of every mesh vertex.

    Mesh-boundary vertices classify on entities of dimension below the mesh
    dimension (optionally restricted by transfer ``groups``); interior vertices
    of volume meshes lie on an interface face within tolerance or in the solid
    containing them; every vertex of a two-dimensional mesh classifies on its
    lowest entity within tolerance.
    """
    top = mesh.topological_dimension
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    count = points.shape[0]
    boundary = boundary_mask(mesh, 0)
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    paths: list[tuple[str, ...]] = [()] * count

    def assign(rows: np.ndarray, result: BRepProjectionResult, /) -> None:
        dims[rows] = np.asarray(result.dimensions, dtype=np.int64)
        indices[rows] = np.asarray(result.indices, dtype=np.int64)
        status[rows] = np.asarray(result.status)
        for row, path in zip(rows, result.source_occurrence_paths, strict=True):
            paths[row] = path

    rows = np.flatnonzero(boundary)
    if rows.size:
        vertex_groups: np.ndarray | None = None
        allowed_groups: np.ndarray | None = None
        allowed_dims: np.ndarray | None = None
        allowed_indices: np.ndarray | None = None
        allowed_paths: tuple[tuple[str, ...], ...] | None = None
        if groups is not None:
            (
                source_groups,
                allowed_groups,
                allowed_dims,
                allowed_indices,
                allowed_paths,
            ) = groups
            vertex_groups = source_groups[rows]
        budget.reserve_geometry_queries(rows.size)
        assign(
            rows,
            projection.classify(
                points[rows],
                tolerance=tolerance,
                maximum_dimension=min(top - 1, 2),
                groups=vertex_groups,
                allowed_groups=allowed_groups,
                allowed_dimensions=allowed_dims,
                allowed_indices=allowed_indices,
                allowed_occurrence_paths=allowed_paths,
                budget=budget,
            ),
        )
    rows = np.flatnonzero(~boundary)
    if rows.size:
        budget.reserve_geometry_queries(rows.size)
        assign(
            rows,
            projection.classify(points[rows], tolerance=tolerance, budget=budget),
        )
        if top == 3:
            inside = rows[dims[rows] < 0]
            if inside.size:
                budget.reserve_geometry_queries(inside.size)
                assign(inside, projection.locate_solids(points[inside]))
    return _Classes(dims, indices, status, tuple(paths))


@checked
def associate_mesh_vertices(
    mesh: cell_mesh_owner.CellMesh,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget | None = None,
) -> GeometryAssociation:
    """Classify every mesh vertex geometrically and project it onto its class.

    Source queries are charged as in :func:`propagate_association`.
    """
    if mesh.ambient_dimension != projection.ambient_dimension:
        raise ValueError(
            f"GeometryAssociation.ambient_dimension must agree between mesh and projection; got mesh {mesh.ambient_dimension}, projection {projection.ambient_dimension}"
        )
    with support_budget(
        MeshCertificateLimits().maximum_work_units,
        policy.maximum_projection_queries,
        budget,
        maximum_memory_bytes=(
            MeshCertificateLimits().maximum_scratch_bytes
            if budget is None
            else budget.maximum_memory_bytes
        ),
    ) as budget:
        classes = _geometric_vertex_classes(
            mesh, projection, policy.classification_tolerance, None, budget
        )
        return _brep_association(
            projection,
            mesh,
            0,
            classes,
            np.asarray(mesh.coordinates, dtype=np.float64),
            policy,
            budget,
            provenance=GeometryAssociationProvenance.CLASSIFICATION,
            parent_association_id=None,
        )


# -- propagation through lineage --------------------------------------------------


@checked
def propagate_association(
    association: GeometryAssociation,
    lineage: lineage_records_owner.MeshLineage,
    source_mesh: cell_mesh_owner.CellMesh,
    target_mesh: cell_mesh_owner.CellMesh,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget | None = None,
) -> GeometryAssociation:
    """Carry B-Rep vertex associations through a bound topology lineage.

    Preserved vertices keep their class and projection; relocated vertices keep
    their class and are re-projected; created vertices inherit the class of the
    source entity they lie in and are projected; vertices without any parent are
    classified geometrically. Raises :class:`AssociationPropagationError` for a
    collapse whose kept vertex class is not in the closure of the removed class.

    Every point submitted to ``projection.project``, ``classify`` or
    ``locate_solids`` is admitted as one geometry query on ``budget`` before
    that call; topology lookups are not queries. The counter does not see
    inside one call: candidate-entity projections of a classification and
    per-occurrence containment tests of a solid location are not counted
    separately. ``project`` and ``classify`` receive ``budget`` for their own
    work; ``locate_solids`` has no budget input. A request ``budget`` is
    activated with its caps unchanged; without one the call settles against
    ``policy.maximum_projection_queries``.
    """
    with support_budget(
        MeshCertificateLimits().maximum_work_units,
        policy.maximum_projection_queries,
        budget,
        maximum_memory_bytes=(
            MeshCertificateLimits().maximum_scratch_bytes
            if budget is None
            else budget.maximum_memory_bytes
        ),
    ) as budget:
        return _propagated_association(
            association, lineage, source_mesh, target_mesh, projection, policy, budget
        )


def _propagated_association(
    association: GeometryAssociation,
    lineage: lineage_records_owner.MeshLineage,
    source_mesh: cell_mesh_owner.CellMesh,
    target_mesh: cell_mesh_owner.CellMesh,
    projection: AbstractBRepProjection,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget,
    /,
) -> GeometryAssociation:
    _require_brep(association, projection, source_mesh, 0)
    require_transfer_lineage(
        source_mesh,
        lineage,
        target_mesh,
        message=f"GeometryAssociation.lineage.topology_ids must bind source and target meshes; got lineage {(lineage.source_topology_id, lineage.target_topology_id)!r}, meshes {(source_mesh.topology_id, target_mesh.topology_id)!r}",
    )
    source_classes = _vertex_classes(source_mesh, association)
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    record = lineage.entity_lineage(0)
    kinds = np.asarray(record.relation_kinds)
    sources = np.asarray(record.source_global_ids, dtype=np.int64)
    targets = np.asarray(record.target_global_ids, dtype=np.int64)
    _check_collapses(
        projection,
        source_classes,
        entity_rows(
            source_mesh,
            0,
            sources[kinds == lineage_records_owner.EntityLineageKind.COLLAPSED_INTO],
        ),
        entity_rows(
            source_mesh,
            0,
            targets[kinds == lineage_records_owner.EntityLineageKind.COLLAPSED_INTO],
        ),
        targets[kinds == lineage_records_owner.EntityLineageKind.COLLAPSED_INTO],
    )
    count = target_ids.size
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    paths: list[tuple[str, ...]] = [()] * count
    parent_dims = np.full((count,), -1, dtype=np.int8)
    parent_ids = np.full((count,), -1, dtype=np.int64)
    kept = np.flatnonzero(entity_rows(source_mesh, 0, target_ids) >= 0)
    kept_source = entity_rows(source_mesh, 0, target_ids[kept])
    dims[kept] = source_classes.dimensions[kept_source]
    indices[kept] = source_classes.indices[kept_source]
    status[kept] = source_classes.status[kept_source]
    for target_row, source_row in zip(kept, kept_source, strict=True):
        paths[target_row] = source_classes.paths[source_row]
    parent_dims[kept] = 0
    parent_ids[kept] = target_ids[kept]
    created_rows = np.flatnonzero(entity_rows(source_mesh, 0, target_ids) < 0)
    created = target_ids[created_rows]
    order = np.argsort(created, kind="stable")
    created_rows, created = created_rows[order], created[order]
    if created.size:
        entity_dims, local_entity_rows = child_sources(
            lineage, source_mesh, target_mesh, created
        )
        found = entity_dims >= 0
        if np.any(found):
            source_levels = _classify_mesh(
                source_mesh, source_classes, projection, budget
            )
            for level in np.unique(entity_dims[found]).tolist():
                selected = np.flatnonzero(entity_dims == level)
                level_classes = source_levels[level]
                rows = local_entity_rows[selected]
                target_rows = created_rows[selected]
                dims[target_rows] = level_classes.dimensions[rows]
                indices[target_rows] = level_classes.indices[rows]
                status[target_rows] = level_classes.status[rows]
                for target_row, source_row in zip(target_rows, rows, strict=True):
                    paths[target_row] = level_classes.paths[source_row]
                parent_dims[target_rows] = level
                parent_ids[target_rows] = np.asarray(
                    source_mesh.entity_set(level).entity_ids, dtype=np.int64
                )[rows]
        orphans = created_rows[~found]
        if orphans.size:
            budget.reserve_geometry_queries(orphans.size)
            geometric = projection.classify(
                np.asarray(target_mesh.coordinates, dtype=np.float64)[orphans],
                tolerance=policy.classification_tolerance,
                budget=budget,
            )
            dims[orphans] = np.asarray(geometric.dimensions, dtype=np.int64)
            indices[orphans] = np.asarray(geometric.indices, dtype=np.int64)
            status[orphans] = np.asarray(geometric.status)
            for target_row, path in zip(
                orphans, geometric.source_occurrence_paths, strict=True
            ):
                paths[target_row] = path
            if target_mesh.topological_dimension == 3:
                inside = orphans[
                    (dims[orphans] < 0) & ~boundary_mask(target_mesh, 0)[orphans]
                ]
                if inside.size:
                    budget.reserve_geometry_queries(inside.size)
                    solid = projection.locate_solids(
                        np.asarray(target_mesh.coordinates, dtype=np.float64)[inside]
                    )
                    dims[inside] = np.asarray(solid.dimensions, dtype=np.int64)
                    indices[inside] = np.asarray(solid.indices, dtype=np.int64)
                    status[inside] = np.asarray(solid.status)
                    for target_row, path in zip(
                        inside, solid.source_occurrence_paths, strict=True
                    ):
                        paths[target_row] = path
    return _brep_association(
        projection,
        target_mesh,
        0,
        _Classes(dims, indices, status, tuple(paths)),
        np.asarray(target_mesh.coordinates, dtype=np.float64),
        policy,
        budget,
        provenance=GeometryAssociationProvenance.LINEAGE,
        parent_association_id=association.association_id,
        parent_dimensions=parent_dims,
        parent_ids=parent_ids,
    )


def _check_collapses(
    projection: AbstractBRepProjection,
    classes: _Classes,
    removed: np.ndarray,
    kept: np.ndarray,
    kept_ids: np.ndarray,
    /,
) -> None:
    """A collapse keeps a vertex whose class lies in the removed vertex's closure."""
    if removed.size == 0:
        return
    if np.any(removed < 0) or np.any(kept < 0):
        raise ValueError(
            f"GeometryAssociation.collapse_lineage must connect declared source vertices; got removed rows {removed!r}, kept rows {kept!r}"
        )
    removed_dims = classes.dimensions[removed]
    kept_dims = classes.dimensions[kept]
    known = (removed_dims >= 0) & (kept_dims >= 0)
    legal = np.zeros(removed.shape, dtype=np.bool_)
    legal[known] = projection.contains(
        removed_dims[known],
        classes.indices[removed][known],
        kept_dims[known],
        classes.indices[kept][known],
        occurrence_paths=tuple(classes.paths[row] for row in removed[known]),
        member_occurrence_paths=tuple(classes.paths[row] for row in kept[known]),
    )
    if not np.all(legal):
        raise AssociationPropagationError(
            f"GeometryAssociation.collapse_classes must place the kept class in the removed class closure; got violating kept IDs {kept_ids[~legal]!r}",
            kept_ids[~legal],
        )


# -- re-derivation after unknown lineage --------------------------------------------


def _facet_signatures(
    result: result_owner.CellMeshingResult, rows: int, /
) -> tuple[np.ndarray, tuple]:
    """Facet organization signature (names of facet patches, zones, and labels)."""
    mesh = result.mesh
    dimension = mesh.topological_dimension - 1
    items = sorted(
        (value.name, value.scope)
        for value in (*result.patches, *result.zones, *result.labels)
        if value.scope.entity_dimension == dimension
    )
    columns = [
        np.asarray(resolve_mesh_scope(mesh, scope).mask, dtype=np.bool_)
        for _, scope in items
    ]
    matrix = np.stack(columns, axis=1) if columns else np.zeros((rows, 0), dtype=np.bool_)
    return matrix, tuple(name for name, _ in items)


def _transfer_groups(
    source: result_owner.CellMeshingResult,
    target: result_owner.CellMeshingResult,
    source_classes: tuple[_Classes, ...],
    projection: AbstractBRepProjection,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
    """Admissible B-Rep entities per target boundary vertex from facet classes.

    Source boundary facets carrying one organization signature admit the B-Rep
    classes of those facets and their closures; target boundary vertices admit
    the union over their incident target facets with the same signatures.
    """
    top = source.mesh.topological_dimension
    facet_count = source.mesh.entity_set(top - 1).count
    source_matrix, source_names = _facet_signatures(source, facet_count)
    target_matrix, target_names = _facet_signatures(
        target, target.mesh.entity_set(top - 1).count
    )
    names = tuple(sorted(set(source_names) | set(target_names)))

    def aligned(matrix: np.ndarray, own: tuple[str, ...], /) -> np.ndarray:
        output = np.zeros((matrix.shape[0], len(names)), dtype=np.bool_)
        output[:, [names.index(name) for name in own]] = matrix
        return output

    boundary = boundary_mask(source.mesh, top - 1)
    _, signature = np.unique(
        np.concatenate(
            (
                aligned(source_matrix, source_names),
                aligned(target_matrix, target_names),
            )
        ),
        axis=0,
        return_inverse=True,
    )
    signature = signature.reshape(-1)
    source_signature = signature[:facet_count]
    target_signature = signature[facet_count:]
    facet_classes = source_classes[top - 1]
    rows = np.flatnonzero(boundary & (facet_classes.dimensions >= 0))
    allowed = _members_closure(
        projection,
        source_signature[rows],
        facet_classes.dimensions[rows],
        facet_classes.indices[rows],
        tuple(facet_classes.paths[row] for row in rows),
    )
    target_boundary = boundary_mask(target.mesh, top - 1)
    vertex_facets = incidence_pairs(target.mesh, 0, top - 1)
    vertex_facets = vertex_facets[target_boundary[vertex_facets[:, 1]]]
    vertex_signatures = np.unique(
        np.stack((vertex_facets[:, 0], target_signature[vertex_facets[:, 1]]), axis=1),
        axis=0,
    ).reshape(-1, 2)
    vertex_count = target.mesh.coordinates.shape[0]
    width = max(1, int(np.max(np.bincount(vertex_signatures[:, 0], minlength=1))))
    # One group per distinct set of incident facet signatures (padded rows).
    padded = np.full((vertex_count, width), -1, dtype=np.int64)
    slot = np.arange(vertex_signatures.shape[0]) - np.searchsorted(
        vertex_signatures[:, 0], vertex_signatures[:, 0]
    )
    padded[vertex_signatures[:, 0], slot] = vertex_signatures[:, 1]
    combinations, groups = np.unique(padded, axis=0, return_inverse=True)
    group_signature = np.argwhere(combinations >= 0)
    group_signature[:, 1] = combinations[group_signature[:, 0], group_signature[:, 1]]
    group_codes = compose_incidence_pairs(group_signature, allowed)
    dims, indices = projection.code_entities(group_codes[:, 1])
    return (
        groups.reshape(-1),
        group_codes[:, 0],
        dims,
        indices,
        projection.code_occurrence_paths(group_codes[:, 1]),
    )


def _members_closure(
    projection: AbstractBRepProjection,
    signatures: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    occurrence_paths: tuple[tuple[str, ...], ...],
    /,
) -> np.ndarray:
    """Unique ``(signature, entity_code)`` of every classified facet and its closure."""
    owner, member_dims, member_indices, member_paths = projection.members(
        dimensions, indices, occurrence_paths=occurrence_paths
    )
    return np.unique(
        np.stack(
            (
                signatures[owner],
                projection.entity_codes(
                    member_dims,
                    member_indices,
                    occurrence_paths=member_paths,
                ),
            ),
            axis=1,
        ),
        axis=0,
    ).reshape(-1, 2)


@checked
def rederive_association(
    association: GeometryAssociation,
    source: result_owner.CellMeshingResult,
    target: result_owner.CellMeshingResult,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget | None = None,
) -> GeometryAssociation:
    """Re-derive a vertex association on a remeshed target with unknown lineage.

    Classification transfer: source boundary facets are classified from the
    source vertex ``association``; target boundary vertices may classify only on
    the B-Rep classes (with closures) of source facets sharing the organization
    signature (facet patches, zones, labels, which remeshing providers carry as
    references/class IDs) of their incident target facets. Each vertex then takes
    the lowest-dimensional admissible entity within the classification tolerance
    and is projected; residuals are the evidence. Source queries are charged as
    in :func:`propagate_association`.
    """
    with support_budget(
        MeshCertificateLimits().maximum_work_units,
        policy.maximum_projection_queries,
        budget,
        maximum_memory_bytes=(
            MeshCertificateLimits().maximum_scratch_bytes
            if budget is None
            else budget.maximum_memory_bytes
        ),
    ) as budget:
        return _rederived_association(
            association, source, target, projection, policy, budget
        )


def _rederived_association(
    association: GeometryAssociation,
    source: result_owner.CellMeshingResult,
    target: result_owner.CellMeshingResult,
    projection: AbstractBRepProjection,
    policy: AssociationPropagationPolicy,
    budget: CoordinateEnclosureBudget,
    /,
) -> GeometryAssociation:
    _require_brep(association, projection, source.mesh, 0)
    if target.mesh.topological_dimension != source.mesh.topological_dimension:
        raise ValueError(
            f"GeometryAssociation.rederivation_dimension must agree between source and target; got source {source.mesh.topological_dimension}, target {target.mesh.topological_dimension}"
        )
    source_levels = _classify_mesh(
        source.mesh, _vertex_classes(source.mesh, association), projection, budget
    )
    groups = _transfer_groups(source, target, source_levels, projection)
    classes = _geometric_vertex_classes(
        target.mesh, projection, policy.classification_tolerance, groups, budget
    )
    return _brep_association(
        projection,
        target.mesh,
        0,
        classes,
        np.asarray(target.mesh.coordinates, dtype=np.float64),
        policy,
        budget,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=association.association_id,
    )


# -- result-level transfer ---------------------------------------------------------


@final
class BRepAssociationTransfer(StrictModule, NonTrainableState):
    """Carries a result's B-Rep associations through topology changes.

    Binds one prepared projection and propagation policy. A source result must
    carry exactly one vertex association of the projection's revision (covering
    every vertex) and may carry B-Rep associations of higher mesh dimensions; the
    target receives the propagated or re-derived vertex association and the
    associations of the same higher dimensions derived from it.
    """

    projection: AbstractBRepProjection
    policy: AssociationPropagationPolicy
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        projection: AbstractBRepProjection,
        /,
        *,
        policy: AssociationPropagationPolicy | None = None,
    ) -> None:
        policy_ = AssociationPropagationPolicy() if policy is None else policy
        self.projection = projection
        self.policy = policy_
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "brep-association-transfer",
                "projection": projection.projection_id,
                "policy": policy_.policy_id,
            }
        )

    @checked
    def source_associations(
        self, source: result_owner.CellMeshingResult, /
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        """The source vertex association and the higher dimensions to rebuild."""

        dimensions = []
        vertex = None
        for association in source.associations:
            if (
                association.association_kind is not GeometryAssociationKind.BREP
                or association.source_revision != self.projection.source_revision
            ):
                raise ValueError(
                    f"BRepAssociationTransfer.source_associations must be BREP tables of the projection revision; got kind {association.association_kind!r}, revision {association.source_revision!r}, expected {self.projection.source_revision!r}"
                )
            dimension = target_dimension(source.mesh, association)
            if dimension == 0:
                if vertex is not None:
                    raise ValueError(
                        f"BRepAssociationTransfer.vertex_association must be unique; got duplicate vertex table {association.association_id!r}"
                    )
                vertex = association
            else:
                dimensions.append(dimension)
        if vertex is None:
            raise ValueError(
                f"BRepAssociationTransfer.vertex_association must be present; got {vertex!r}"
            )
        if len(set(dimensions)) != len(dimensions):
            raise ValueError(
                f"BRepAssociationTransfer.source_associations must have distinct target dimensions; got {dimensions!r}"
            )
        _vertex_classes(source.mesh, vertex)
        return vertex, tuple(sorted(dimensions))

    def _budget(
        self, budget: CoordinateEnclosureBudget | None, /
    ) -> AbstractContextManager[CoordinateEnclosureBudget]:
        return support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.policy.maximum_projection_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        )

    @checked
    def classes(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[_Classes, ...]:
        """B-Rep classes of every source mesh dimension (adaptation constraints)."""
        vertex, _ = self.source_associations(source)
        with self._budget(budget) as budget:
            return _mesh_classes(source.mesh, vertex, self.projection, budget)

    def _derived(
        self,
        vertex: GeometryAssociation,
        dimensions: tuple[int, ...],
        mesh: cell_mesh_owner.CellMesh,
        budget: CoordinateEnclosureBudget,
        /,
    ) -> tuple[GeometryAssociation, ...]:
        return (vertex,) + tuple(
            _entity_association(
                mesh, vertex, self.projection, dimension, self.policy, budget
            )
            for dimension in dimensions
        )

    @checked
    def propagate(
        self,
        source: result_owner.CellMeshingResult,
        lineage: lineage_records_owner.MeshLineage,
        target: cell_mesh_owner.CellMesh,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        """Return target associations through one declared topology lineage.

        Source queries are charged as in :func:`propagate_association`; a
        request ``budget`` keeps its own caps and ticket.
        """
        vertex, dimensions = self.source_associations(source)
        with self._budget(budget) as budget:
            propagated = _propagated_association(
                vertex, lineage, source.mesh, target, self.projection, self.policy, budget
            )
            return self._derived(propagated, dimensions, target, budget)

    @checked
    def rederive(
        self,
        source: result_owner.CellMeshingResult,
        target: result_owner.CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        """Target associations of a remeshed result with unknown lineage."""
        vertex, dimensions = self.source_associations(source)
        with self._budget(budget) as budget:
            derived = _rederived_association(
                vertex, source, target, self.projection, self.policy, budget
            )
            return self._derived(derived, dimensions, target.mesh, budget)


@contextmanager
def support_budget(
    maximum_work_units: int,
    maximum_support_queries: int,
    budget: CoordinateEnclosureBudget | None,
    /,
    *,
    maximum_memory_bytes: int,
) -> Iterator[CoordinateEnclosureBudget]:
    """Share one caller-owned cap and ledger, or settle a standalone host batch."""
    maximum_memory_bytes = positive_integer(
        maximum_memory_bytes, "support_budget.maximum_memory_bytes"
    )
    if budget is not None:
        if budget.maximum_memory_bytes != maximum_memory_bytes:
            raise ValueError("Association support budget must match the caller memory limit")
        with budget.activate():
            yield budget
        return
    with HostBatchTicket(
        maximum_work_units,
        queries_upper=maximum_support_queries,
        bytes_upper=maximum_memory_bytes,
    ) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work_units,
            maximum_memory_bytes=maximum_memory_bytes,
        )
        with ledger.activate():
            yield ledger


class _PlcAmbientDim(Dim, minimum=2):
    """Declared Cartesian dimension of source PLC vertices."""


class _PlcEdgeDim(Dim, minimum=1):
    """Declared source PLC edges."""


class _PlcTriangleDim(Dim):
    """Declared source facet triangles."""


class _PlcFacetDim(Dim):
    """Declared source facet incidence groups."""


class _PlcRegionDim(Dim, minimum=1):
    """Explicit scientific region identifiers aligned to domain definitions."""


class _PlcVertexDim(Dim, minimum=1):
    """Explicit source vertex identifiers."""


@dataclass
class PlcBankContext:
    """One explicit namespace using shared, independently certified coverage."""

    mapped: plc_mapped_support_owner.MappedPlcSupport
    facet_indices: np.ndarray
    region_map: np.ndarray
    work: list[int]
    target_mapped: plc_mapped_support_owner.MappedPlcSupport | None = None
    mapped_facets: MappedSourceFacetAuthority | None = None

    def indices(self, association: GeometryAssociation, /) -> np.ndarray:
        result = np.array(association.source_indices, dtype=np.int64, copy=True)
        dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
        positions = {int(label): row for row, label in enumerate(self.facet_indices)}
        for row in np.flatnonzero(dimensions == 2):
            label = int(result[row])
            if label not in positions:
                raise ValueError(
                    f"PlcBankContext.facet_indices must declare each associated source facet; got label {label}, declared {self.facet_indices!r}"
                )
            result[row] = positions[label]
        return result

    def source_index(self, dimension: int, index: int, /) -> int:
        return int(self.facet_indices[index]) if dimension == 2 else index

    def support_index(
        self, transfer: PlcAssociationTransfer, dimension: int, index: int, /
    ) -> int:
        if dimension != 3:
            return index
        matches = np.flatnonzero(transfer.region_indices == index)
        if matches.size != 1:
            raise ValueError(
                f"PlcBankContext.region_map must declare one associated source region; got region {index}, matching rows {matches!r}"
            )
        return int(self.region_map[matches[0]])

    def bank_regions(
        self,
        transfer: PlcAssociationTransfer,
        mapped: plc_mapped_support_owner.MappedPlcSupport | None = None,
        /,
    ) -> np.ndarray:
        proof = self.mapped if mapped is None else mapped
        result = np.full(proof.cell_regions.shape, -1, dtype=np.int64)
        for source_region, material_region in zip(
            transfer.region_indices, self.region_map, strict=True
        ):
            result[proof.cell_regions == material_region] = source_region
        return result


class _MappedSourceBindings(StrictModule):
    tables: dict[int, GeometryAssociation]


def _validate_mapped_source_bindings(
    source: result_owner.CellMeshingResult,
    domain: MappedReferenceDomain,
    expected: tuple[GeometryAssociation, ...],
) -> _MappedSourceBindings:
    """Validate all retained source tables against their freshly certified root images."""
    by_dimension = association_lookup(source)
    if len(by_dimension) != len(source.associations) or set(by_dimension) != set(
        range(4)
    ):
        raise ValueError(
            f"MappedReferenceAssociationTransfer.source_associations must contain one table for each dimension in [0, 3]; got dimensions {tuple(by_dimension)!r}, table count {len(source.associations)}"
        )
    for degree, proof in enumerate(expected):
        item = by_dimension[degree]
        item.validate_target(source.mesh.entity_set(degree))
        rows = item.target_rows(np.asarray(proof.target_global_ids))
        if (
            item.association_kind is not GeometryAssociationKind.MAPPED_REFERENCE
            or item.source_id != domain.source_id
            or item.source_revision != domain.source_revision
            or not item.exact
            or not np.all(np.asarray(item.resolved))
            or np.any(np.asarray(item.ambiguous))
            or tuple(item.source_entities.entity_id(row) for row in rows)
            != proof.source_entities.entity_ids()
            or not np.array_equal(
                np.asarray(item.parent_dimensions)[rows],
                np.asarray(proof.parent_dimensions),
            )
            or not np.array_equal(
                np.asarray(item.parent_ids)[rows], np.asarray(proof.parent_ids)
            )
            or not np.array_equal(
                np.asarray(item.orientations)[rows],
                np.asarray(proof.orientations),
            )
        ):
            raise ValueError(
                f"MappedReferenceAssociationTransfer.source_associations must retain certified root strata, parents, and orientations; got dimension {degree}, association {item.association_id!r}, expected proof {proof.association_id!r}"
            )
    return _MappedSourceBindings(by_dimension)


class _PlcGeometryTables(StrictModule):
    authority_vertices: np.ndarray
    edges: np.ndarray
    triangles: np.ndarray
    facets: np.ndarray
    regions: np.ndarray


class _PlcIdentityTables(StrictModule):
    labels: np.ndarray
    vertex_labels: np.ndarray
    edge_labels: np.ndarray


def _prepare_plc_geometry(
    domain: PiecewiseLinearDomain,
    edge_vertices: ArrayLike,
    triangle_vertices: ArrayLike,
    triangle_facets: ArrayLike,
    facet_regions: ArrayLike,
    source_vertices: ArrayLike | None,
    scope: Scope,
) -> _PlcGeometryTables:
    """Prepare host source geometry and incidence tables in one shared dimension scope."""
    raw = tuple(
        np.asarray(value)
        for value in (
            edge_vertices,
            triangle_vertices,
            triangle_facets,
            facet_regions,
        )
    )
    if any(not np.issubdtype(value.dtype, np.integer) for value in raw):
        raise TypeError(
            f"PlcAssociationTransfer.identity_tables must contain integer indices; got dtypes {tuple(str(value.dtype) for value in raw)!r}"
        )
    authority_vertices = parse(
        np.array(
            domain.vertices if source_vertices is None else source_vertices,
            dtype=np.float64,
            copy=True,
        ),
        HostFloat64[_PlcVertexDim, _PlcAmbientDim],
        "source_vertices",
        scope=scope,
    )
    if authority_vertices.shape[1] != domain.ambient_dimension or not np.all(
        np.isfinite(authority_vertices)
    ):
        raise ValueError(
            f"PlcAssociationTransfer.source_vertices must be finite Cartesian rows in the domain frame; got shape {authority_vertices.shape!r}, ambient dimension {domain.ambient_dimension}, finite {bool(np.all(np.isfinite(authority_vertices)))}"
        )
    if domain.ambient_dimension == 3 and not np.array_equal(
        authority_vertices, domain.vertices
    ):
        raise ValueError(
            f"PlcAssociationTransfer.source_vertices must match volume domain vertices in row order; got source shape {authority_vertices.shape!r}, domain shape {domain.vertices.shape!r}, equal {bool(np.array_equal(authority_vertices, domain.vertices))}"
        )
    edges = parse(
        np.array(raw[0], dtype=np.int64, copy=True),
        HostInt64[_PlcEdgeDim, Literal[2]],
        "edge_vertices",
        scope=scope,
    )
    triangles = parse(
        np.array(raw[1], dtype=np.int64, copy=True),
        HostInt64[_PlcTriangleDim, Literal[3]],
        "triangle_vertices",
        scope=scope,
    )
    facets = parse(
        np.array(raw[2], dtype=np.int64, copy=True),
        HostInt64[_PlcTriangleDim],
        "triangle_facets",
        scope=scope,
    )
    regions = parse(
        np.array(raw[3], dtype=np.int64, copy=True),
        HostInt64[_PlcFacetDim, Literal[2]],
        "facet_regions",
        scope=scope,
    )
    return _PlcGeometryTables(authority_vertices, edges, triangles, facets, regions)


def _prepare_plc_identity(
    domain: PiecewiseLinearDomain,
    geometry: _PlcGeometryTables,
    region_indices: ArrayLike | None,
    vertex_indices: ArrayLike | None,
    edge_indices: ArrayLike | None,
    scope: Scope,
) -> _PlcIdentityTables:
    """Validate source row identifiers against their admitted geometry table extents."""
    authority_vertices, edges = geometry.authority_vertices, geometry.edges
    labels_raw = (
        np.arange(len(domain.region_ids), dtype=np.int64)
        if region_indices is None
        else np.asarray(region_indices)
    )
    if not np.issubdtype(labels_raw.dtype, np.integer):
        raise TypeError(
            f"PlcAssociationTransfer.region_indices must contain integer identifiers; got dtype {labels_raw.dtype}"
        )
    labels = parse(
        np.array(labels_raw, dtype=np.int64, copy=True),
        HostInt64[_PlcRegionDim],
        "region_indices",
        scope=scope,
    )
    if (
        labels.shape != (len(domain.region_ids),)
        or np.any(labels < 0)
        or np.unique(labels).size != labels.size
    ):
        raise ValueError(
            f"PlcAssociationTransfer.region_indices must uniquely name each domain region with nonnegative indices; got {labels!r}, region count {len(domain.region_ids)}"
        )
    vertex_labels_raw = (
        np.arange(authority_vertices.shape[0], dtype=np.int64)
        if vertex_indices is None
        else np.asarray(vertex_indices)
    )
    edge_labels_raw = (
        np.arange(edges.shape[0], dtype=np.int64)
        if edge_indices is None
        else np.asarray(edge_indices)
    )
    if not np.issubdtype(vertex_labels_raw.dtype, np.integer) or not np.issubdtype(
        edge_labels_raw.dtype, np.integer
    ):
        raise TypeError(
            f"PlcAssociationTransfer.primitive_indices must contain integer vectors; got vertex dtype {vertex_labels_raw.dtype}, edge dtype {edge_labels_raw.dtype}"
        )
    vertex_labels = parse(
        np.array(vertex_labels_raw, dtype=np.int64, copy=True),
        HostInt64[_PlcVertexDim],
        "vertex_indices",
        scope=scope,
    )
    edge_labels = parse(
        np.array(edge_labels_raw, dtype=np.int64, copy=True),
        HostInt64[_PlcEdgeDim],
        "edge_indices",
        scope=scope,
    )
    if (
        vertex_labels.shape != (authority_vertices.shape[0],)
        or edge_labels.shape != (edges.shape[0],)
        or np.any(vertex_labels < 0)
        or np.any(edge_labels < 0)
        or np.unique(vertex_labels).size != vertex_labels.size
        or np.unique(edge_labels).size != edge_labels.size
    ):
        raise ValueError(
            f"PlcAssociationTransfer.primitive_indices must uniquely name each nonnegative vertex and edge row; got vertex labels {vertex_labels!r}, edge labels {edge_labels!r}, counts {(authority_vertices.shape[0], edges.shape[0])!r}"
        )
    return _PlcIdentityTables(labels, vertex_labels, edge_labels)


def _validate_plc_incidence(
    domain: PiecewiseLinearDomain, geometry: _PlcGeometryTables
) -> _PlcGeometryTables:
    """Validate planar or volume incidence without changing source namespace indices."""
    authority_vertices, edges, triangles, facets, regions = (
        geometry.authority_vertices,
        geometry.edges,
        geometry.triangles,
        geometry.facets,
        geometry.regions,
    )
    if domain.ambient_dimension == 3 and (
        triangles.shape[0] == 0 or regions.shape[0] == 0
    ):
        raise ValueError(
            f"PlcAssociationTransfer.volume_facets must contain triangles and region incidence; got triangle rows {triangles.shape[0]}, facet rows {regions.shape[0]}"
        )
    if domain.ambient_dimension == 2 and triangles.shape[0] != 0:
        raise ValueError(
            f"PlcAssociationTransfer.planar_facets must contain no facet triangles; got triangle rows {triangles.shape[0]}"
        )
    if (
        np.any(edges < 0)
        or np.any(edges >= authority_vertices.shape[0])
        or np.any(triangles < 0)
        or np.any(triangles >= authority_vertices.shape[0])
    ):
        raise ValueError(
            f"PlcAssociationTransfer.vertex_incidence must index declared source vertices; got edges {edges!r}, triangles {triangles!r}, vertex count {authority_vertices.shape[0]}"
        )
    if (
        np.any(facets < 0)
        or np.any(facets >= regions.shape[0])
        or np.any(regions < -1)
        or np.any(regions >= len(domain.region_ids))
    ):
        raise ValueError(
            f"PlcAssociationTransfer.facet_incidence must index declared facets and regions or void -1; got facets {facets!r}, regions {regions!r}, domain region count {len(domain.region_ids)}"
        )
    if (
        np.any(edges[:, 0] == edges[:, 1])
        or np.unique(np.sort(edges, axis=1), axis=0).shape[0] != edges.shape[0]
    ):
        raise ValueError(
            f"PlcAssociationTransfer.edge_vertices must be distinct noncollapsed edges; got {edges!r}"
        )
    return geometry


def _volume_stratum_consensus(
    mapped: plc_mapped_support_owner.MappedPlcSupport,
    dimension: int,
    cell_regions: np.ndarray,
    /,
) -> np.ndarray:
    """Reduce authenticated cell incidence, excluding certified boundary strata."""
    if dimension == 3:
        return cell_regions.copy()
    offsets, parents = mapped.entity_parents[dimension]
    populated = offsets[1:] > offsets[:-1]
    regions = np.full(populated.shape, -1, dtype=np.int64)
    if parents.size:
        starts = offsets[:-1][populated]
        labels = cell_regions[parents]
        low = np.minimum.reduceat(labels, starts)
        high = np.maximum.reduceat(labels, starts)
        regions[populated] = np.where(low == high, low, -1)
    regions[mapped.boundary_entities[dimension]] = -1
    return regions


class _PlcAssociationBinding(StrictModule):
    dimension: int
    dimensions: np.ndarray


def _periodic_restriction_premise(
    source: result_owner.CellMeshingResult,
    lineage: lineage_records_owner.MeshLineage,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    predecessor: CellGeometryPredecessor | None = None,
    geometry_transition: CellGeometryTransition | None = None,
    target_request: certification_inputs_owner.MeshCertificationInputs | None = None,
) -> PeriodicRestrictionPremise | None:
    """Declare each periodic child's lineage parent for the exact partition theorem.

    Declarations are hypotheses only; the embedding owner proves containment,
    sign cells and exact volumes, and otherwise runs its bounded image proof.
    """
    certificate = source.certification
    if (
        source.mesh.periodic_topology is None
        or target.periodic_topology is None
        or certificate is None
        or certificate.embedding is None
        or certificate.embedding.status != "certified"
        or not source.audit.validity.all_certified
    ):
        return None
    cells = next(
        (item for item in lineage.entities if item.dimension == target.topological_dimension),
        None,
    )
    if cells is None:
        return None
    children = np.asarray(cells.target_global_ids, dtype=np.int64)
    parents = np.asarray(cells.source_global_ids, dtype=np.int64)
    inverse = children.size != np.unique(children).size
    if inverse:
        if parents.size != np.unique(parents).size:
            return None
        children, parents = parents, children
    source_request = certificate.request
    same_regions = (
        source_request.domain is not None
        and target_request is not None
        and target_request.domain is not None
        and source_request.domain.domain_id == target_request.domain.domain_id
        and source_request.cell_regions is not None
        and target_request.cell_regions is not None
    )
    return PeriodicRestrictionPremise(
        source.mesh,
        source.geometry,
        source.audit.validity,
        certificate.embedding,
        tuple(
            (int(child), int(parent))
            for child, parent in zip(children.tolist(), parents.tolist(), strict=True)
        ),
        inverse=inverse,
        mapped_predecessor=predecessor,
        mapped_transition=geometry_transition,
        source_regions=tuple(int(value) for value in np.asarray(source_request.cell_regions))
        if same_regions else None,
        target_regions=tuple(int(value) for value in np.asarray(target_request.cell_regions))
        if same_regions and target_request is not None else None,
        region_namespace_id=source_request.domain.domain_id
        if same_regions and source_request.domain is not None else None,
        source_domain=source_request.domain if same_regions else None,
    )


class _PlcLineageTables(StrictModule):
    table: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    obligations: list[np.ndarray]


class _PlcSourceClassification(StrictModule):
    associations: tuple[GeometryAssociation, ...]
    active_dimensions: frozenset[int]
    proof: plc_mapped_support_owner.MappedPlcSupport
    cell_regions: np.ndarray
    work: list[int]


class _PlcSourceTableProof(StrictModule):
    association: GeometryAssociation
    dimension: int
    rows: np.ndarray
    indices: np.ndarray


class _PlcSourceClasses(StrictModule):
    classes: tuple[PlcEntityClasses, ...]


class _PlcTargetSupport(StrictModule):
    mapped: plc_mapped_support_owner.MappedPlcSupport


class _PlcCertifiedRows(StrictModule):
    orientations: np.ndarray


class _PlcTranslationSource(StrictModule):
    vertex: GeometryAssociation
    dimensions: tuple[int, ...]


class _PlcAssociationBatch(StrictModule):
    associations: tuple[GeometryAssociation, ...]


@final
class PlcAssociationTransfer(StrictModule, NonTrainableState):
    """Transfer represented PLC source strata through certified coordinate restrictions.

    Source vertex, edge, and facet tables define explicit identity banks. Planar
    source vertices may differ from the refined support-domain carrier; Steiner
    rows retain carrier identities rather than becoming source vertices.
    ``transition_source_associations`` retains planar strata under a declared
    translation after checking incidence, source labels, and coordinate support.
    """

    __strict_contract__ = True
    domain: PiecewiseLinearDomain
    coordinate_contract: SpatialCoordinateContract
    source_vertices: HostFloat64[_PlcVertexDim, _PlcAmbientDim]
    edge_vertices: HostInt64[_PlcEdgeDim, Literal[2]]
    vertex_indices: HostInt64[_PlcVertexDim]
    edge_indices: HostInt64[_PlcEdgeDim]
    triangle_vertices: HostInt64[_PlcTriangleDim, Literal[3]]
    triangle_facets: HostInt64[_PlcTriangleDim]
    facet_regions: HostInt64[_PlcFacetDim, Literal[2]]
    region_indices: HostInt64[_PlcRegionDim]
    source_revision: str = eqx.field(static=True)
    maximum_support_queries: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: PiecewiseLinearDomain,
        coordinate_contract: SpatialCoordinateContract,
        source_revision: str,
        /,
        *,
        edge_vertices: ArrayLike,
        triangle_vertices: ArrayLike,
        triangle_facets: ArrayLike,
        facet_regions: ArrayLike,
        region_indices: ArrayLike | None = None,
        source_vertices: ArrayLike | None = None,
        vertex_indices: ArrayLike | None = None,
        edge_indices: ArrayLike | None = None,
        maximum_support_queries: int = 1 << 26,
    ) -> None:

        if domain.ambient_dimension not in (2, 3):
            raise TypeError(
                f"PlcAssociationTransfer.domain must have ambient dimension two or three; got {domain.ambient_dimension}"
            )
        if not coordinate_contract.is_orthonormal_cartesian:
            raise ValueError(
                f"PlcAssociationTransfer.coordinate_contract must declare orthonormal Cartesian coordinates; got spatial ID {coordinate_contract.spatial_id!r}, Cartesian {coordinate_contract.is_orthonormal_cartesian}"
            )
        revision = canonical_identifier(source_revision, "source_revision")
        limit = positive_integer(maximum_support_queries, "maximum_support_queries")
        scope = Scope()
        geometry_tables = _prepare_plc_geometry(
            domain,
            edge_vertices,
            triangle_vertices,
            triangle_facets,
            facet_regions,
            source_vertices,
            scope,
        )
        identity_tables = _prepare_plc_identity(
            domain,
            geometry_tables,
            region_indices,
            vertex_indices,
            edge_indices,
            scope,
        )
        _validate_plc_incidence(domain, geometry_tables)
        authority_vertices, edges, triangles, facets, regions = (
            geometry_tables.authority_vertices,
            geometry_tables.edges,
            geometry_tables.triangles,
            geometry_tables.facets,
            geometry_tables.regions,
        )
        labels, vertex_labels, edge_labels = (
            identity_tables.labels,
            identity_tables.vertex_labels,
            identity_tables.edge_labels,
        )
        snapshot = PiecewiseLinearDomain(
            np.array(domain.vertices, dtype=np.float64, copy=True),
            np.array(domain.facets, dtype=np.int64, copy=True),
            np.array(domain.facet_regions, dtype=np.int64, copy=True),
            domain.region_ids,
            source_id=domain.source_id,
        )
        for value in (
            snapshot.vertices,
            snapshot.facets,
            snapshot.facet_regions,
            authority_vertices,
            edges,
            triangles,
            facets,
            regions,
            labels,
            vertex_labels,
            edge_labels,
        ):
            value.setflags(write=False)
        self.domain = snapshot
        self.source_vertices = authority_vertices
        self.coordinate_contract = coordinate_contract
        self.edge_vertices = edges
        self.triangle_vertices = triangles
        self.triangle_facets = facets
        self.facet_regions = regions
        self.region_indices = labels
        self.vertex_indices = vertex_labels
        self.edge_indices = edge_labels
        self.source_revision = revision
        self.maximum_support_queries = limit
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "plc-association-transfer",
                "domain": snapshot.domain_id,
                "source_revision": revision,
                "coordinate_contract": coordinate_contract.spatial_id,
                "tables": array_tree_fingerprint(
                    (
                        edges,
                        triangles,
                        facets,
                        regions,
                        labels,
                        vertex_labels,
                        edge_labels,
                    )
                ),
                "maximum_support_queries": limit,
                **(
                    {}
                    if np.array_equal(authority_vertices, snapshot.vertices)
                    else {"source_vertices": array_tree_fingerprint(authority_vertices)}
                ),
            }
        )

    def _admit_planar_translation(
        self,
        source: result_owner.CellMeshingResult,
        predecessor: PlcAssociationTransfer,
        target: cell_mesh_owner.CellMesh,
        geometry: CellGeometrySpec,
        translation: ArrayLike,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> _PlcTranslationSource:
        """Validate rigid source correspondence and retain the admitted predecessor tables."""
        if (
            self.domain.ambient_dimension != 2
            or predecessor.domain.ambient_dimension != 2
        ):
            raise ValueError(
                f"PlcAssociationTransfer.translation_domains must both be planar; got predecessor dimension {predecessor.domain.ambient_dimension}, successor dimension {self.domain.ambient_dimension}"
            )
        if self.domain.source_id != predecessor.domain.source_id or (
            self.coordinate_contract.spatial_id
            != predecessor.coordinate_contract.spatial_id
        ):
            raise ValueError(
                f"PlcAssociationTransfer.translation_binding must retain source identity and physical frame; got predecessor {(predecessor.domain.source_id, predecessor.coordinate_contract.spatial_id)!r}, successor {(self.domain.source_id, self.coordinate_contract.spatial_id)!r}"
            )
        if self.source_revision == predecessor.source_revision:
            raise ValueError(
                f"PlcAssociationTransfer.source_revision must change for a source translation; got predecessor {predecessor.source_revision!r}, successor {self.source_revision!r}"
            )
        tables = (
            (self.edge_vertices, predecessor.edge_vertices),
            (self.vertex_indices, predecessor.vertex_indices),
            (self.edge_indices, predecessor.edge_indices),
            (self.triangle_vertices, predecessor.triangle_vertices),
            (self.triangle_facets, predecessor.triangle_facets),
            (self.facet_regions, predecessor.facet_regions),
            (self.region_indices, predecessor.region_indices),
        )
        if any(not np.array_equal(actual, original) for actual, original in tables):
            raise ValueError(
                f"PlcAssociationTransfer.translation_tables must retain stratum incidence and indices; got table equality {tuple(bool(np.array_equal(actual, previous)) for actual, previous in tables)!r}"
            )
        if not np.array_equal(self.domain.facets, predecessor.domain.facets) or (
            not np.array_equal(
                self.domain.facet_regions, predecessor.domain.facet_regions
            )
        ):
            raise ValueError(
                f"PlcAssociationTransfer.translation_domain must retain facet and region incidence; got facet equality {bool(np.array_equal(self.domain.facets, predecessor.domain.facets))}, region equality {bool(np.array_equal(self.domain.facet_regions, predecessor.domain.facet_regions))}"
            )
        for transfer in (predecessor, self):
            expected_regions = tuple(
                entity_key(transfer.source_revision, "region", int(index))
                for index in transfer.region_indices
            )
            if transfer.domain.region_ids != expected_regions:
                raise ValueError(
                    f"PlcAssociationTransfer.domain.region_ids must encode declared source region indices; got {transfer.domain.region_ids!r}, expected {expected_regions!r}"
                )
        shift = np.asarray(translation, dtype=np.float64)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError(
                f"PlcAssociationTransfer.translation must be a finite planar displacement; got {shift!r}"
            )
        layout_retained = (
            geometry.geometry_layout_id == source.geometry.geometry_layout_id
        )
        match source.geometry.exact_source, geometry.exact_source:
            case (
                ExactPlcCellGeometrySource() as old,
                ExactPlcCellGeometrySource() as new,
            ):
                layout_retained = (
                    translated_plc_layout_matches(
                        source.geometry,
                        geometry,
                        translation=shift,
                        budget=budget,
                    )
                    and new.domain_source_id == self.domain.source_id
                    and old.domain_source_id == predecessor.domain.source_id
                    and new.domain_source_revision == self.source_revision
                    and old.domain_source_revision == predecessor.source_revision
                    and np.array_equal(new.source_points, self.source_vertices)
                    and np.array_equal(old.source_points, predecessor.source_vertices)
                )
            case _:
                pass
        if target.topology_id != source.mesh.topology_id or not layout_retained:
            raise ValueError(
                f"PlcAssociationTransfer.translation_geometry must retain mesh topology and coordinate layout; got source topology {source.mesh.topology_id!r}, target topology {target.topology_id!r}, layout retained {layout_retained}"
            )
        for dimension in range(3):
            if not np.array_equal(
                target.entity_set(dimension).entity_ids,
                source.mesh.entity_set(dimension).entity_ids,
            ):
                raise ValueError(
                    f"PlcAssociationTransfer.translation_entities must retain each dimension row correspondence; got dimension {dimension}, source IDs {source.mesh.entity_set(dimension).entity_ids!r}, target IDs {target.entity_set(dimension).entity_ids!r}"
                )
        for actual, original in (
            (self.source_vertices, predecessor.source_vertices),
            (self.domain.vertices, predecessor.domain.vertices),
            (np.asarray(target.coordinates), np.asarray(source.mesh.coordinates)),
            (
                np.asarray(geometry.coordinates),
                np.asarray(source.geometry.coordinates),
            ),
        ):
            if actual.shape != original.shape or not np.array_equal(
                actual, original + shift
            ):
                raise ValueError(
                    f"PlcAssociationTransfer.translation_coordinates must equal predecessor coordinates plus displacement; got shapes {actual.shape!r}, {original.shape!r}, displacement {shift!r}, translated equality {bool(np.array_equal(actual, original + shift)) if actual.shape == original.shape else False}"
                )
        old_vertex, dimensions = predecessor.source_associations(source, budget=budget)
        if old_vertex is None:
            raise ValueError(
                f"PlcAssociationTransfer.translation_vertex_association must be present; got {old_vertex!r}"
            )

        return _PlcTranslationSource(old_vertex, dimensions)

    def _publish_planar_translation(
        self,
        source: result_owner.CellMeshingResult,
        target: cell_mesh_owner.CellMesh,
        admitted: _PlcTranslationSource,
        levels: tuple[PlcEntityClasses, ...],
        orientations: tuple[np.ndarray, ...],
    ) -> _PlcAssociationBatch:
        """Publish translated tables with the predecessor scopes, orientation, and lineage IDs."""
        old_vertex, dimensions = admitted.vertex, admitted.dimensions
        output: list[GeometryAssociation] = []
        for dimension in (0, *dimensions):
            previous = (
                old_vertex
                if dimension == 0
                else association_for_dimension(source, dimension)
            )
            # Fixed topology transports the caller's declared association scope,
            # not every entity whose source stratum was proved internally.
            identifiers = np.asarray(previous.target_global_ids, dtype=np.int64)
            rows = entity_rows(target, dimension, identifiers)
            level = levels[dimension]
            local_source_entity_roles = plc_planar_support_owner.source_entity_roles(
                level.dimensions[rows]
            )
            names = tuple(
                entity_key(self.source_revision, role.value, int(index))
                for role, index in zip(
                    local_source_entity_roles, level.indices[rows], strict=True
                )
            )
            output.append(
                GeometryAssociation(
                    GeometryAssociationKind.PIECEWISE_LINEAR,
                    self.domain.source_id,
                    self.source_revision,
                    target.entity_set(dimension).entity_set_id,
                    identifiers,
                    names,
                    np.zeros(identifiers.shape, dtype=np.float64),
                    exact=True,
                    source_dimensions=level.dimensions[rows],
                    source_indices=level.indices[rows],
                    source_occurrence_paths=previous.source_entities.occurrence_paths(),
                    source_entity_roles=local_source_entity_roles,
                    parameters=plc_planar_support_owner.preserved_parameters(
                        source.mesh, target, dimension, previous, level
                    )[rows],
                    orientations=orientations[dimension][rows],
                    parent_dimensions=np.full(
                        identifiers.shape, dimension, dtype=np.int64
                    ),
                    parent_ids=identifiers,
                    parent_association_id=previous.association_id,
                    provenance=GeometryAssociationProvenance.LINEAGE,
                )
            )
        return _PlcAssociationBatch(tuple(output))

    @checked
    def transition_source_associations(
        self,
        source: result_owner.CellMeshingResult,
        predecessor: PlcAssociationTransfer,
        target: cell_mesh_owner.CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        translation: ArrayLike,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[
        tuple[organization_records_owner.MeshPatch, ...],
        tuple[organization_records_owner.MeshZone, ...],
        tuple[organization_records_owner.MeshLabel, ...],
        tuple[GeometryAssociation, ...],
    ]:
        """Revalidate source strata under a declared translation.

        Source vertices and edges are independent of the represented support
        carrier; Steiner rows retain carrier identities. Incidence, scientific
        entity correspondence, and the translated coordinate map are checked
        before successor associations are published. Supplied planar evidence
        must bind the target map and source domain alongside the stratum checks.
        """
        with support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            if source.mesh.storage is not None or target.storage is not None:
                raise ValueError(
                    f"PlcAssociationTransfer.translation_storage must use host-only carriers; got source storage present {source.mesh.storage is not None}, target storage present {target.storage is not None}"
                )
            if (
                self.domain.ambient_dimension == 3
                and predecessor.domain.ambient_dimension == 3
            ):
                if embedding is not None or coverage is not None:
                    raise ValueError(
                        f"PlcAssociationTransfer.translation_evidence must be internally prepared for volume carriers; got supplied embedding {embedding is not None}, coverage {coverage is not None}"
                    )

                return plc_mapped_support_owner.transition_mapped_source_associations(
                    self,
                    source,
                    predecessor,
                    target,
                    geometry=geometry,
                    translation=translation,
                    budget=budget,
                )
            admitted = self._admit_planar_translation(
                source,
                predecessor,
                target,
                geometry,
                translation,
                budget=budget,
            )
            levels = predecessor.classes(source, budget=budget)
            orientations = plc_planar_support_owner.prove_levels(
                self,
                target,
                geometry,
                levels,
                [self.maximum_support_queries],
                embedding=embedding,
                coverage=coverage,
                budget=budget,
            )
            output = self._publish_planar_translation(
                source,
                target,
                admitted,
                levels,
                orientations,
            ).associations
            patches, zones, labels = lineage_owner.inherit_mesh_organization(
                source, target, lineage_owner.identity_lineage(source.mesh, target)
            )
            return patches, zones, labels, tuple(output)

    def _validate_source_binding(
        self,
        source: result_owner.CellMeshingResult,
        association: GeometryAssociation,
        dimensions: list[int],
        counts: tuple[int, ...],
        source_roles: tuple[GeometrySourceEntityRole, ...],
        *,
        _bank: PlcBankContext | None,
    ) -> _PlcAssociationBinding:
        """Validate one PLC table before recording its dimension and source namespace."""
        if (
            association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
            or association.source_id != self.domain.source_id
            or association.source_revision != self.source_revision
        ):
            raise ValueError(
                f"PlcAssociationTransfer.source_association must identify the declared PLC source and revision; got kind {association.association_kind!r}, source {(association.source_id, association.source_revision)!r}, expected {(self.domain.source_id, self.source_revision)!r}"
            )
        dimension = target_dimension(source.mesh, association)
        if dimension in dimensions:
            raise ValueError(
                f"PlcAssociationTransfer.source_associations must have one table per dimension; got repeated dimension {dimension}, preceding dimensions {dimensions!r}"
            )
        association.validate_target(source.mesh.entity_set(dimension))
        if (
            not np.all(np.asarray(association.resolved))
            or np.any(np.asarray(association.ambiguous))
            or not association.exact
        ):
            raise ValueError(
                f"PlcAssociationTransfer.source_association must resolve all rows unambiguously with zero-residual support; got GeometryAssociation.exact={association.exact}, resolved={np.asarray(association.resolved)!r}, ambiguous={np.asarray(association.ambiguous)!r}"
            )
        dims = np.asarray(association.source_dimensions, dtype=np.int64)
        indices = np.asarray(association.source_indices, dtype=np.int64)
        if np.any(dims < 0) or np.any(dims > 3) or np.any(indices < 0):
            raise ValueError(
                f"PlcAssociationTransfer.source_strata must use dimensions in [0, 3] and nonnegative indices; got dimensions {dims!r}, indices {indices!r}"
            )
        facet_authority = (
            np.arange(counts[2], dtype=np.int64) if _bank is None else _bank.facet_indices
        )
        if (
            np.any(~np.isin(indices[dims == 0], self.vertex_indices))
            or np.any(~np.isin(indices[dims == 1], self.edge_indices))
            or np.any(~np.isin(indices[dims == 2], facet_authority))
            or np.any(~np.isin(indices[dims == 3], self.region_indices))
        ):
            raise ValueError(
                f"PlcAssociationTransfer.source_indices must belong to their declared source stratum; got dimensions {dims!r}, indices {indices!r}"
            )
        roles = tuple(source_roles[kind] for kind in dims.tolist())
        if association.source_entities.roles() != roles:
            raise ValueError(
                f"PlcAssociationTransfer.source_entity_roles must agree with volume strata; got {association.source_entities.roles()!r}, expected {roles!r}"
            )
        expected = tuple(
            entity_key(self.source_revision, role.value, index)
            for role, index in zip(roles, indices.tolist(), strict=True)
        )
        if association.source_entities.entity_ids() != expected:
            raise ValueError(
                f"PlcAssociationTransfer.source_entity_ids must encode declared source roles and indices; got {association.source_entities.entity_ids()!r}, expected {expected!r}"
            )
        return _PlcAssociationBinding(dimension, dims)

    @checked
    def source_associations(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        _bank: PlcBankContext | None = None,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation | None, tuple[int, ...]]:
        with support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            if self.domain.ambient_dimension == 2:
                return plc_planar_support_owner.planar_source_associations(
                    self, source, budget=budget
                )

            if (
                source.coordinate_contract.spatial_id
                != self.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    f"PlcAssociationTransfer.source.coordinate_contract must match the transfer frame; got {source.coordinate_contract.spatial_id!r}, expected {self.coordinate_contract.spatial_id!r}"
                )
            certificate = source.certification
            coverage_domain = self.domain if _bank is None else _bank.mapped.domain
            if (
                certificate is None
                or not certificate.passed
                or not isinstance(certificate.coverage, DomainCoverageCertificate)
                or certificate.coverage.domain_id != coverage_domain.domain_id
            ):
                raise ValueError(
                    f"PlcAssociationTransfer.source.certification must pass and exactly cover the declared domain; got passed {None if certificate is None else certificate.passed}, expected {coverage_domain.domain_id!r}"
                )
            if certificate.mesh_id != source.mesh.mesh_id:
                raise ValueError(
                    f"PlcAssociationTransfer.source.certification.mesh_id must bind the source mesh; got {certificate.mesh_id!r}, expected {source.mesh.mesh_id!r}"
                )
            certificate.coverage.binding.require(source.mesh, source.geometry)
            if _bank is not None:
                if not plc_mapped_support_owner._support_proof_is_issued(
                    _bank.mapped.proof
                ):
                    raise ValueError(
                        "PlcAssociationTransfer.prepared_source must retain an issued support proof; got a stale or foreign proof"
                    )
                # This bank preparation already checked every rounded carrier
                # corner under its persistent support ledger.
                _bank.mapped.prepared.require(
                    source.mesh,
                    source.geometry,
                    certificate.request,
                    budget=budget,
                )
            dimensions: list[int] = []
            vertex: GeometryAssociation | None = None
            counts = (
                self.domain.vertices.shape[0],
                self.edge_vertices.shape[0],
                self.facet_regions.shape[0],
                len(self.domain.region_ids),
            )
            source_roles = (
                GeometrySourceEntityRole.VERTEX,
                GeometrySourceEntityRole.EDGE,
                GeometrySourceEntityRole.FACET,
                GeometrySourceEntityRole.REGION,
            )
            for association in source.associations:
                if _bank is not None and (
                    association.source_id,
                    association.source_revision,
                ) != (self.domain.source_id, self.source_revision):
                    continue
                binding = self._validate_source_binding(
                    source,
                    association,
                    dimensions,
                    counts,
                    source_roles,
                    _bank=_bank,
                )
                dimension, dims = binding.dimension, binding.dimensions
                dimensions.append(dimension)
                if dimension == 0:
                    if _bank is None:
                        association.target_rows(
                            np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
                        )
                    vertex = association
                if dimension == 3:
                    if _bank is None:
                        association.target_rows(
                            np.asarray(
                                source.mesh.entity_set(3).entity_ids, dtype=np.int64
                            )
                        )
                    if np.any(dims != 3):
                        raise ValueError(
                            f"PlcAssociationTransfer.cell_dimensions must identify source regions with dimension three; got {dims!r}"
                        )
            if _bank is None and (vertex is None or 3 not in dimensions):
                raise ValueError(
                    f"PlcAssociationTransfer.source_associations must include vertex and cell tables; got vertex present {vertex is not None}, dimensions {dimensions!r}"
                )
            if _bank is None:
                prepared = [] if _prepared_source is None else _prepared_source
                if len(prepared) > 1:
                    raise ValueError(
                        f"PlcAssociationTransfer.prepared_source must contain at most one support proof; got count {len(prepared)}"
                    )
                cell_regions = self._source_cell_regions(source)
                if not prepared:
                    prepared.append(
                        plc_mapped_support_owner.prepare_mapped_plc_support(
                            source.mesh,
                            source.geometry,
                            self.domain,
                            cell_regions,
                            self.region_indices,
                            maximum_support_queries=self.maximum_support_queries,
                            embedding=certificate.embedding,
                            validity=source.audit.validity,
                            certificate_limits=certificate.request.limits,
                            certification_request=certificate.request,
                            budget=budget,
                        )
                    )
                proof = prepared[0]
                if (
                    proof.ledger is not budget
                    or proof.domain.domain_id != self.domain.domain_id
                    or proof.limits.maximum_candidate_pairs
                    != self.maximum_support_queries
                    or not np.array_equal(proof.cell_regions, cell_regions)
                    or not plc_mapped_support_owner._support_proof_is_issued(proof.proof)
                ):
                    raise ValueError(
                        f"PlcAssociationTransfer.prepared_source must be bound to this request ledger and match domain, support allowance and source cell regions; got same ledger {proof.ledger is budget}, domain {proof.domain.domain_id!r}, allowance {proof.limits.maximum_candidate_pairs}, expected {(self.domain.domain_id, self.maximum_support_queries)!r}"
                    )
                proof.prepared.require(
                    source.mesh, source.geometry, certificate.request, budget=budget
                )
            return vertex, tuple(
                sorted(dimension for dimension in dimensions if dimension != 0)
            )

    @staticmethod
    def _source_cell_regions(source: result_owner.CellMeshingResult, /) -> np.ndarray:
        """Declared source-region labels of the source cells in entity order."""
        cell = association_for_dimension(source, 3)
        ids = np.asarray(source.mesh.entity_set(3).entity_ids, dtype=np.int64)
        return np.asarray(cell.source_indices, dtype=np.int64)[cell.target_rows(ids)]

    @checked
    def bind_source_proof(
        self,
        source: result_owner.CellMeshingResult,
        proof: plc_mapped_support_owner.MappedPlcSupportProof,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        import_charges: bool = True,
    ) -> plc_mapped_support_owner.MappedPlcSupport:
        """Bind an ended source-support proof of this transfer to a request ledger.

        The source, domain, allowance, cell regions and certified request are
        authenticated before the proof's retained graph is admitted; a stale or
        foreign proof refuses. ``import_charges=False`` is for the request whose
        own charged preparation receipt already contains the proof's work.
        """
        certificate = source.certification
        if certificate is None or self.domain.ambient_dimension != 3:
            raise ValueError(
                f"PlcAssociationTransfer.source_proof must bind a certified volume source; got certification present {certificate is not None}, dimension {self.domain.ambient_dimension}"
            )
        return plc_mapped_support_owner.bind_mapped_plc_support(
            proof,
            source.mesh,
            source.geometry,
            self.domain,
            self._source_cell_regions(source),
            maximum_support_queries=self.maximum_support_queries,
            certification_request=certificate.request,
            budget=budget,
            import_charges=import_charges,
        )

    def _prepare_source_classification(
        self,
        source: result_owner.CellMeshingResult,
        prepared: list[plc_mapped_support_owner.MappedPlcSupport],
        *,
        _bank: PlcBankContext | None,
    ) -> _PlcSourceClassification:
        """Bind one namespace's tables, material regions, and shared support allowance."""
        associations = tuple(
            value
            for value in source.associations
            if _bank is None
            or (value.source_id, value.source_revision)
            == (self.domain.source_id, self.source_revision)
        )
        active_dimensions = frozenset(
            target_dimension(source.mesh, value) for value in associations
        )
        if _bank is None:
            proof = prepared[0]
            cell_regions = proof.cell_regions
        else:
            cell_regions = _bank.bank_regions(self)
            proof = _bank.mapped
        work = [self.maximum_support_queries] if _bank is None else _bank.work
        return _PlcSourceClassification(
            associations, active_dimensions, proof, cell_regions, work
        )

    def _certify_source_classification(
        self,
        source: result_owner.CellMeshingResult,
        association: GeometryAssociation,
        prepared: _PlcSourceClassification,
        *,
        _bank: PlcBankContext | None,
        budget: CoordinateEnclosureBudget,
    ) -> _PlcSourceTableProof:
        """Certify a table's continuous source support and requested orientations in row order."""
        proof, work = prepared.proof, prepared.work
        dimension = target_dimension(source.mesh, association)
        target_ids = np.asarray(association.target_global_ids, dtype=np.int64)
        source_dimensions = np.asarray(association.source_dimensions)
        source_indices = np.asarray(association.source_indices)
        orientations = np.asarray(association.orientations)
        vertex_indices = np.asarray(self.vertex_indices)
        edge_indices = np.asarray(self.edge_indices)
        rows = entity_rows(
            source.mesh,
            dimension,
            target_ids,
        )
        bank_indices = source_indices if _bank is None else _bank.indices(association)
        association_row = 0
        while association_row < rows.size:
            row = int(rows[association_row])
            if source_dimensions[association_row] == 3:
                end = association_row + 1
                while end < rows.size and source_dimensions[end] == 3:
                    end += 1
                with budget.temporary_scope():
                    budget.reserve(0, 256 + 96 * (end - association_row))
                    regions = bank_indices[association_row:end]
                    support_regions = (
                        regions
                        if _bank is None
                        else np.asarray(
                            [
                                _bank.support_index(self, 3, int(region))
                                for region in regions
                            ],
                            dtype=np.int64,
                        )
                    )
                    outcomes = proof.region_support(
                        dimension, rows[association_row:end], support_regions, work
                    )
                    failed = np.flatnonzero(
                        ~outcomes | (orientations[association_row:end] != 0)
                    )
                    if failed.size:
                        position = association_row + int(failed[0])
                        supported = bool(outcomes[failed[0]])
                        expected = int(orientations[position])
                        raise AssociationPropagationError(
                            f"PlcAssociationTransfer.source_support must certify source strata and requested orientation; got bank {self.domain.source_id!r}, dimension {dimension}, target ID {int(target_ids[position])}, stratum {(3, int(source_indices[position]))!r}, support {supported}, orientation 0, requested orientation {expected}",
                            target_ids[position : position + 1],
                        )
                association_row = end
                continue
            kind = int(source_dimensions[association_row])
            index = int(bank_indices[association_row])
            local_index = index
            if kind in (0, 1):
                matches = np.flatnonzero(
                    (vertex_indices if kind == 0 else edge_indices) == index
                )
                if matches.size != 1:
                    raise ValueError(
                        f"PlcAssociationTransfer.source_index must identify one source primitive; got dimension {kind}, index {index}, matching rows {matches!r}"
                    )
                local_index = int(matches[0])
            if _bank is not None:
                local_index = _bank.support_index(self, kind, local_index)
            if kind == 2 and _bank is not None and _bank.mapped_facets is not None:
                supported, orientation, work[0] = _bank.mapped_facets.support(
                    proof,
                    dimension,
                    row,
                    local_index,
                    work[0],
                    budget=budget,
                )
            elif dimension == 0:
                supported = proof.vertex_support(
                    row,
                    kind,
                    local_index,
                    edge_vertices=self.edge_vertices,
                    triangle_vertices=self.triangle_vertices,
                    triangle_facets=self.triangle_facets,
                    work=work,
                    source_vertices=self.source_vertices,
                )
                orientation = 0
            else:
                supported, orientation = proof.entity_support(
                    dimension,
                    row,
                    kind,
                    local_index,
                    edge_vertices=self.edge_vertices,
                    triangle_vertices=self.triangle_vertices,
                    triangle_facets=self.triangle_facets,
                    work=work,
                    source_vertices=self.source_vertices,
                )
            expected = int(orientations[association_row])
            if not supported or (expected != 0 and expected != orientation):
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.source_support must certify source strata and requested orientation; got bank {self.domain.source_id!r}, dimension {dimension}, target ID {int(target_ids[association_row])}, stratum {(kind, int(source_indices[association_row]))!r}, support {supported}, orientation {orientation}, requested orientation {expected}",
                    target_ids[association_row : association_row + 1],
                )
            association_row += 1
        return _PlcSourceTableProof(association, dimension, rows, bank_indices)

    def _publish_source_classification(
        self,
        source: result_owner.CellMeshingResult,
        prepared: _PlcSourceClassification,
        bindings: tuple[_PlcSourceTableProof, ...],
        *,
        _bank: PlcBankContext | None,
    ) -> _PlcSourceClasses:
        """Publish immutable consensus strata and collision-free namespace codes."""
        cell_regions = prepared.cell_regions
        active_dimensions = prepared.active_dimensions
        offsets = np.cumsum(
            np.asarray(
                (
                    0,
                    self.vertex_indices.shape[0],
                    self.edge_indices.shape[0],
                    self.facet_regions.shape[0],
                ),
                dtype=np.int64,
            )
        )
        result: list[PlcEntityClasses] = []
        for dimension in range(4):
            regions = _volume_stratum_consensus(prepared.proof, dimension, cell_regions)
            dims = np.where(regions >= 0, 3, -1).astype(np.int64)
            indices = regions.copy()
            for binding in bindings:
                association = binding.association
                if binding.dimension != dimension:
                    continue
                rows = binding.rows
                dims[rows] = np.asarray(association.source_dimensions, dtype=np.int64)
                indices[rows] = binding.indices
            if (
                _bank is not None
                and dimension not in active_dimensions
                and dimension != 3
            ):
                dims.fill(-1)
                indices.fill(-1)
            resolved = (dims >= 0) & (indices >= 0)
            if _bank is None and not np.all(resolved):
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.interface_strata must resolve every target entity; got dimension {dimension}, unresolved IDs {np.asarray(source.mesh.entity_set(dimension).entity_ids)[~resolved]!r}",
                    np.asarray(source.mesh.entity_set(dimension).entity_ids)[~resolved],
                )
            codes = np.full_like(indices, -1)
            authorities = (
                self.vertex_indices,
                self.edge_indices,
                np.arange(self.facet_regions.shape[0], dtype=np.int64),
                self.region_indices,
            )
            for kind, authority in enumerate(authorities):
                selected = dims == kind
                positions = {label: row for row, label in enumerate(authority.tolist())}
                codes[selected] = offsets[kind] + np.asarray(
                    [positions[label] for label in indices[selected].tolist()],
                    dtype=np.int64,
                )
            for value in (dims, indices, resolved, codes):
                value.setflags(write=False)
            result.append(PlcEntityClasses(dims, indices, resolved, codes))
        return _PlcSourceClasses(tuple(result))

    @checked
    def classes(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        _bank: PlcBankContext | None = None,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[PlcEntityClasses, ...]:
        with support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            if self.domain.ambient_dimension == 2:
                return plc_planar_support_owner.planar_classes(
                    self, source, budget=budget
                )
            prepared = [] if _prepared_source is None else _prepared_source
            self.source_associations(
                source, _bank=_bank, _prepared_source=prepared, budget=budget
            )
            return self._classified_source(
                source, prepared, _bank=_bank, budget=budget
            )

    def _classified_source(
        self,
        source: result_owner.CellMeshingResult,
        prepared: list[plc_mapped_support_owner.MappedPlcSupport],
        *,
        _bank: PlcBankContext | None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[PlcEntityClasses, ...]:
        """Classify volume tables after this call has authenticated their source binding."""
        if _bank is None:
            earned = plc_mapped_support_owner.validated_mapped_source_classes(
                prepared[0], source, self, budget=budget
            )
            if earned is not None:
                return earned
        work_start = budget.work_units
        ticket_start = budget.ticket.work_units, budget.ticket.geometry_queries
        classification = self._prepare_source_classification(
            source, prepared, _bank=_bank
        )
        bindings = tuple(
            self._certify_source_classification(
                source,
                association,
                classification,
                _bank=_bank,
                budget=budget,
            )
            for association in classification.associations
        )
        classes = self._publish_source_classification(
            source, classification, bindings, _bank=_bank
        ).classes
        if _bank is not None:
            return classes
        finalized = plc_mapped_support_owner.finalize_mapped_source_classification(
            classification.proof, source, self, classes,
            work_start=work_start, ticket_start=ticket_start, budget=budget,
        )
        prepared[0] = finalized
        earned = finalized.proof.source_classification
        if earned is None:
            raise RuntimeError("Validated PLC source classification was not retained.")
        return earned.classes

    def _region_triangles(self, region: int, /) -> np.ndarray:
        matches = np.flatnonzero(self.region_indices == region)
        if matches.size != 1:
            raise ValueError(
                f"PlcAssociationTransfer.region_index must name one declared region; got {region}, matching rows {matches!r}"
            )
        region = int(matches[0])
        pairs = self.domain.facet_regions
        selected = (pairs[:, 0] == region) | (pairs[:, 1] == region)
        rows = np.array(self.domain.facets[selected], dtype=np.int64, copy=True)
        reversed_ = pairs[selected, 1] == region
        rows[reversed_] = rows[reversed_][:, (0, 2, 1)]
        return self.domain.vertices[rows]

    @checked
    def point_support(
        self,
        dimension: int,
        index: int,
        point: np.ndarray,
        work: list[int],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> bool:

        if self.domain.ambient_dimension == 2:
            return plc_planar_support_owner.planar_point_support(
                self, dimension, index, point, work, budget=budget
            )
        if dimension in (0, 1):
            authority = self.vertex_indices if dimension == 0 else self.edge_indices
            matches = np.flatnonzero(authority == index)
            if matches.size != 1:
                raise ValueError(
                    f"PlcAssociationTransfer.source_index must name one declared primitive; got dimension {dimension}, index {index}, matching rows {matches!r}"
                )
            index = int(matches[0])
        if dimension == 0:
            self._spend(work, 1)
            budget.reserve_geometry_queries(1)
            return np.array_equal(point, self.domain.vertices[index])
        if dimension == 1:
            self._spend(work, 1)
            budget.reserve_geometry_queries(1)
            edge = self.domain.vertices[self.edge_vertices[index]]
            for axes in ((0, 1), (1, 2), (2, 0)):
                if (
                    exact_orient2d(edge[0, axes], edge[1, axes], point[list(axes)]).item()
                    != 0
                ):
                    return False
            axis = int(np.argmax(np.abs(edge[1] - edge[0])))
            return bool(min(edge[:, axis]) <= point[axis] <= max(edge[:, axis]))
        if dimension == 2:
            triangles = self.domain.vertices[
                self.triangle_vertices[self.triangle_facets == index]
            ]
        elif dimension == 3:
            triangles = self._region_triangles(index)
        else:
            raise ValueError(
                f"PlcAssociationTransfer.source_dimension must identify a declared PLC stratum; got {dimension}"
            )
        self._spend(work, triangles.shape[0])
        budget.reserve_geometry_queries(triangles.shape[0])
        _, features, status = point_triangle_locations(
            np.broadcast_to(point, (triangles.shape[0], 3)), triangles
        )
        if np.any(status != MeshcoreStatus.OK):
            raise ValueError(
                f"PlcAssociationTransfer.support_status must be OK for every triangle predicate; got {status!r}"
            )
        on_boundary = bool(np.any(features >= 0))
        if dimension == 2:
            return on_boundary
        if on_boundary:
            return False
        self._spend(work, triangles.shape[0])
        budget.reserve_geometry_queries(triangles.shape[0])
        winding, decided = ray_crossings_3d(point, triangles, budget=budget)
        return decided and winding == 1

    def _spend(self, work: list[int], amount: int, /) -> None:
        work[0] -= amount
        if work[0] < 0:
            raise ValueError(
                f"PlcAssociationTransfer.support_queries must fit the query budget; got remaining allowance {work[0]}"
            )

    @checked
    def protected_edges(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        midpoint_required: bool = True,
        exact_chart_midpoints: bool = False,
        _bank: PlcBankContext | None = None,
        _classes: tuple[PlcEntityClasses, ...] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        with support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            if self.domain.ambient_dimension == 2:
                return plc_planar_support_owner.planar_protected_edges(
                    self,
                    source,
                    midpoint_required=midpoint_required,
                    exact_chart_midpoints=exact_chart_midpoints,
                    _classes=_classes,
                    budget=budget,
                )
            levels = (
                self.classes(source, _bank=_bank, budget=budget)
                if _classes is None
                else _classes
            )
            vertices = plc_mapped_support_owner.plc_mapped_vertices(source.mesh, 1)
            coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
            endpoints = coordinates[vertices]
            midpoints = np.sum(endpoints * np.float64(0.5), axis=1)
            protected = np.all(midpoints == endpoints[:, 0], axis=1) | np.all(
                midpoints == endpoints[:, 1], axis=1
            )
            if midpoint_required:
                work = [self.maximum_support_queries] if _bank is None else _bank.work
                candidates = (levels[1].dimensions >= 0) & (levels[1].dimensions < 3)
                unsupported = np.zeros_like(protected)
                for row in np.flatnonzero(candidates).tolist():
                    unsupported[row] = not self.point_support(
                        int(levels[1].dimensions[row]),
                        int(levels[1].indices[row]),
                        midpoints[row],
                        work,
                        budget=budget,
                    )
                if exact_chart_midpoints:
                    # A resolved class proves the whole exact edge chart in its
                    # stratum. Release a rounded-midpoint refusal only where the
                    # nested producer publishes that exact chart midpoint within
                    # the source's original row bounds.
                    released = np.flatnonzero(
                        unsupported & levels[1].resolved & ~protected
                    )
                    if released.size:
                        unsupported[released] = ~nested_edge_midpoints_admitted(
                            source.mesh,
                            source.geometry,
                            vertices[released],
                            budget=budget,
                        )
                protected |= unsupported
            return protected

    def _inherit_target_strata(
        self,
        source: result_owner.CellMeshingResult,
        lineage: lineage_records_owner.MeshLineage,
        target: cell_mesh_owner.CellMesh,
        old: tuple[PlcEntityClasses, ...],
    ) -> _PlcLineageTables:
        """Prepare retained and inherited target rows while preserving unknown-lineage refusal order."""
        table: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        obligations: list[np.ndarray] = []
        for dimension in range(4):
            ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
            dims = np.full(ids.shape, -1, dtype=np.int64)
            indices = np.full(ids.shape, -1, dtype=np.int64)
            parent_dims = np.full(ids.shape, -1, dtype=np.int8)
            parent_ids = np.full(ids.shape, -1, dtype=np.int64)
            kept = entity_rows(source.mesh, dimension, ids)
            present = kept >= 0
            dims[present], indices[present] = (
                old[dimension].dimensions[kept[present]],
                old[dimension].indices[kept[present]],
            )
            parent_dims[present], parent_ids[present] = dimension, ids[present]
            required = np.zeros(ids.shape, dtype=np.bool_)
            required[present] = old[dimension].resolved[kept[present]]
            record = lineage.entity_lineage(dimension)
            kinds = np.asarray(record.relation_kinds, dtype=np.int32)
            if np.any(kinds == lineage_records_owner.EntityLineageKind.UNKNOWN):
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.lineage.relation_kinds must identify source strata; got unknown relations for dimension {dimension}, target IDs {ids!r}",
                    ids,
                )
            source_ids = np.asarray(record.source_global_ids, dtype=np.int64)
            source_rows = entity_rows(
                source.mesh,
                dimension,
                source_ids,
            )
            target_rows = entity_rows(
                target,
                dimension,
                np.asarray(record.target_global_ids, dtype=np.int64),
            )
            if np.any(source_rows < 0) or np.any(target_rows < 0):
                raise ValueError(
                    f"PlcAssociationTransfer.lineage.entity_ids must belong to source and target entity sets; got source routes {source_rows!r}, target routes {target_rows!r}"
                )
            if target_rows.size:
                order = np.argsort(target_rows, kind="stable")
                ordered_rows = target_rows[order]
                starts = np.r_[
                    0, np.flatnonzero(ordered_rows[1:] != ordered_rows[:-1]) + 1
                ]
                rows = ordered_rows[starts]
                parents = source_rows[order]
                parent_dimensions = old[dimension].dimensions[parents]
                parent_indices = old[dimension].indices[parents]
                required[rows] |= np.logical_or.reduceat(
                    old[dimension].resolved[parents], starts
                )
                same = (
                    np.minimum.reduceat(parent_dimensions, starts)
                    == np.maximum.reduceat(parent_dimensions, starts)
                ) & (
                    np.minimum.reduceat(parent_indices, starts)
                    == np.maximum.reduceat(parent_indices, starts)
                )
                agreed = rows[same]
                dims[agreed] = parent_dimensions[starts[same]]
                indices[agreed] = parent_indices[starts[same]]
                parent_dims[agreed] = dimension
                parent_ids[agreed] = np.minimum.reduceat(source_ids[order], starts)[
                    same
                ]
            table.append((dims, indices, parent_dims, parent_ids))
            obligations.append(required)
        return _PlcLineageTables(table, obligations)

    def _prepare_target_support(
        self,
        source: result_owner.CellMeshingResult,
        target: cell_mesh_owner.CellMesh,
        geometry: CellGeometrySpec,
        inherited: _PlcLineageTables,
        lineage: lineage_records_owner.MeshLineage,
        work: list[int],
        *,
        _bank: PlcBankContext | None,
        _certification_request: certification_inputs_owner.MeshCertificationInputs | None,
        _geometry_transition: CellGeometryTransition | None,
        budget: CoordinateEnclosureBudget,
    ) -> _PlcTargetSupport:
        """Certify all target cell regions before lower-stratum closure classification."""
        table = inherited.table
        cell_dims, cell_regions, _, _ = table[3]
        cell_ids = np.asarray(target.entity_set(3).entity_ids, dtype=np.int64)
        if _bank is None or _bank.target_mapped is None:
            if np.any(cell_dims != 3) or np.any(cell_regions < 0):
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.target_cell_strata must resolve inherited region labels; got dimensions {cell_dims!r}, regions {cell_regions!r}",
                    cell_ids[cell_dims != 3],
                )
            if (
                _bank is not None
                and self.domain.domain_id != _bank.mapped.domain.domain_id
            ):
                raise ValueError(
                    f"PlcAssociationTransfer.material_domain must establish composed target coverage first; got bank domain {self.domain.domain_id!r}, material domain {_bank.mapped.domain.domain_id!r}"
                )
            predecessor = None
            source_certificate = source.certification
            if (
                (
                    geometry.restriction_source is not None
                    or (
                        source.mesh.periodic_topology is not None
                        and _geometry_transition is not None
                    )
                )
                and source_certificate is not None
                and source_certificate.embedding is not None
                and source.audit.validity.all_certified
            ):
                predecessor = prepare_cell_geometry_predecessor(
                    source.mesh,
                    source.geometry,
                    source.audit.validity,
                    source_certificate.embedding,
                    target,
                    geometry,
                    source_limits=source_certificate.request.limits,
                    budget=budget,
                )
            coverage_predecessor = None
            if (
                predecessor is not None and source_certificate is not None
                and isinstance(source_certificate.coverage, DomainCoverageCertificate)
                and source_certificate.request.cell_regions is not None
            ):
                source_regions = np.asarray(
                    source_certificate.request.cell_regions, dtype=np.int64
                ).reshape(-1)
                budget.reserve(source_regions.size, 256 + 16 * source_regions.size)
                coverage_predecessor = DomainCoveragePredecessor(
                    predecessor, source_certificate.coverage,
                    tuple(int(value) for value in source_regions.tolist()),
                    source_certificate.request.limits,
                )
            mapped = plc_mapped_support_owner.prepare_mapped_plc_support(
                target,
                geometry,
                self.domain,
                cell_regions,
                self.region_indices,
                maximum_support_queries=self.maximum_support_queries,
                certification_request=_certification_request,
                predecessor=predecessor,
                coverage_predecessor=coverage_predecessor,
                periodic_restriction=_periodic_restriction_premise(
                    source, lineage, target, predecessor=predecessor,
                    geometry_transition=_geometry_transition,
                    target_request=_certification_request,
                ),
                budget=budget,
            )
        else:
            mapped = _bank.target_mapped
            if not plc_mapped_support_owner._support_proof_is_issued(mapped.proof):
                raise ValueError(
                    "PlcAssociationTransfer.prepared_target must retain an issued support proof; got a stale or foreign proof"
                )
            expected_regions = _bank.bank_regions(self, mapped)
            if not np.array_equal(cell_regions, expected_regions) or np.any(
                cell_dims[expected_regions >= 0] != 3
            ):
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.target_cell_strata must agree with material coverage; got regions {cell_regions!r}, expected {expected_regions!r}, dimensions {cell_dims!r}",
                    cell_ids[cell_regions != expected_regions],
                )
        with budget.temporary_scope():
            budget.reserve(0, 256 + 96 * cell_regions.size)
            rows = np.flatnonzero(cell_regions >= 0)
            regions = cell_regions[rows]
            support_regions = (
                regions
                if _bank is None
                else np.asarray(
                    [_bank.support_index(self, 3, int(region)) for region in regions],
                    dtype=np.int64,
                )
            )
            supported = mapped.region_support(3, rows, support_regions, work)
            failed = np.flatnonzero(~supported)
            if failed.size:
                row = int(rows[failed[0]])
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.target_cell_support must be contained in its inherited region; got cell ID {cell_ids[row]}, region {int(cell_regions[row])}, support False",
                    cell_ids[row : row + 1],
                )
        return _PlcTargetSupport(mapped)

    def _close_target_strata(
        self,
        target: cell_mesh_owner.CellMesh,
        inherited: _PlcLineageTables,
        mapped: plc_mapped_support_owner.MappedPlcSupport,
    ) -> _PlcLineageTables:
        """Fill lower strata from constrained faces and incident-cell region consensus."""
        table = inherited.table
        cell_regions = table[3][1]
        for dimension in (2, 1):
            dims, indices, _, _ = table[dimension]
            if dimension == 1:
                links = incidence_pairs(target, 1, 2)
                face_dims, face_indices, _, _ = table[2]
                constrained = links[face_dims[links[:, 1]] == 2]
                low = np.full(dims.shape, np.iinfo(np.int64).max, dtype=np.int64)
                high = np.full(dims.shape, -1, dtype=np.int64)
                np.minimum.at(low, constrained[:, 0], face_indices[constrained[:, 1]])
                np.maximum.at(high, constrained[:, 0], face_indices[constrained[:, 1]])
                agreed = (dims < 0) & (low == high)
                dims[agreed], indices[agreed] = 2, low[agreed]
            missing = dims < 0
            regions = _volume_stratum_consensus(mapped, dimension, cell_regions)
            dims[missing & (regions >= 0)] = 3
            indices[missing & (regions >= 0)] = regions[missing & (regions >= 0)]
        return inherited

    def _inherit_target_vertices(
        self,
        source: result_owner.CellMeshingResult,
        lineage: lineage_records_owner.MeshLineage,
        target: cell_mesh_owner.CellMesh,
        old: tuple[PlcEntityClasses, ...],
        inherited: _PlcLineageTables,
        mapped: plc_mapped_support_owner.MappedPlcSupport,
        work: list[int],
        *,
        _bank: PlcBankContext | None,
    ) -> _PlcLineageTables:
        """Resolve created vertices from lower strata before admitting interior cell ancestry."""
        table, obligations = inherited.table, inherited.obligations
        cell_regions = table[3][1]
        target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
        dims, indices, parent_dims, parent_ids = table[0]
        missing = np.flatnonzero(dims < 0)
        if missing.size:
            created = target_ids[missing]
            order = np.argsort(created, kind="stable")
            entity_dims, local_entity_rows = child_sources(
                lineage, source.mesh, target, created[order]
            )
            for dimension in range(4):
                selected = (entity_dims == dimension) & (local_entity_rows >= 0)
                positions = missing[order[selected]]
                rows = local_entity_rows[selected]
                dims[positions] = old[dimension].dimensions[rows]
                indices[positions] = old[dimension].indices[rows]
                obligations[0][positions] |= old[dimension].resolved[rows]
                parent_dims[positions] = dimension
                parent_ids[positions] = np.asarray(
                    source.mesh.entity_set(dimension).entity_ids
                )[rows]
        missing = np.flatnonzero(dims < 0)
        if missing.size:
            # A mixed cell-center vertex has cell ancestry, not invented
            # dimension-zero parent rows. Lower-stratum witnesses take priority.
            offsets, incident_cells = mapped.entity_parents[0]
            vertex_rows = np.repeat(np.arange(dims.size), np.diff(offsets))
            cell_record = lineage.entity_lineage(3)
            source_cells = entity_rows(
                source.mesh,
                3,
                np.asarray(cell_record.source_global_ids, dtype=np.int64),
            )
            target_cells = entity_rows(
                target, 3, np.asarray(cell_record.target_global_ids, dtype=np.int64)
            )
            source_cell_ids = np.asarray(
                source.mesh.entity_set(3).entity_ids, dtype=np.int64
            )
            constrained = mapped.boundary_entities[0].copy()
            for dimension in (2, 1):
                vertices = mapped.entity_vertices[dimension]
                rows, slots = np.nonzero(vertices >= 0)
                kinds = table[dimension][0][rows]
                np.logical_or.at(
                    constrained, vertices[rows, slots], (kinds >= 0) & (kinds < 3)
                )
            # Reduce ancestry once per incidence, rather than rescanning every
            # target-cell lineage row for each newly created interior vertex.
            absent = np.iinfo(np.int64).max
            cell_low = np.full(cell_regions.shape, absent, dtype=np.int64)
            cell_high = np.full(cell_regions.shape, -1, dtype=np.int64)
            np.minimum.at(cell_low, target_cells, source_cells)
            np.maximum.at(cell_high, target_cells, source_cells)
            owner_low = np.full(dims.shape, absent, dtype=np.int64)
            owner_high = np.full(dims.shape, -1, dtype=np.int64)
            np.minimum.at(owner_low, vertex_rows, cell_low[incident_cells])
            np.maximum.at(owner_high, vertex_rows, cell_high[incident_cells])
            regions = _volume_stratum_consensus(mapped, 0, cell_regions)
            candidates = missing[
                ~constrained[missing]
                & (owner_low[missing] >= 0)
                & (owner_low[missing] == owner_high[missing])
            ]
            for position in candidates.tolist():
                owner = int(owner_low[position])
                region = int(old[3].indices[owner])
                if (
                    old[3].dimensions[owner] != 3
                    or region < 0
                    or regions[position] != region
                ):
                    continue
                if not mapped.vertex_support(
                    position,
                    3,
                    region if _bank is None else _bank.support_index(self, 3, region),
                    edge_vertices=self.edge_vertices,
                    triangle_vertices=self.triangle_vertices,
                    triangle_facets=self.triangle_facets,
                    work=work,
                    source_vertices=self.source_vertices,
                ):
                    continue
                dims[position], indices[position] = 3, region
                parent_dims[position], parent_ids[position] = (
                    3,
                    source_cell_ids[owner],
                )
        return inherited

    def _certify_target_rows(
        self,
        dimension: int,
        ids: np.ndarray,
        dims: np.ndarray,
        indices: np.ndarray,
        selected: np.ndarray,
        mapped: plc_mapped_support_owner.MappedPlcSupport,
        work: list[int],
        *,
        _bank: PlcBankContext | None,
        budget: CoordinateEnclosureBudget,
    ) -> _PlcCertifiedRows:
        """Certify each inherited source image and retain its orientation in row order."""
        orientations = np.zeros(ids.shape, dtype=np.int8)
        rows = np.flatnonzero(selected)
        position = 0
        while position < rows.size:
            row = int(rows[position])
            if dims[row] == 3:
                end = position + 1
                while end < rows.size and dims[rows[end]] == 3:
                    end += 1
                with budget.temporary_scope():
                    budget.reserve(0, 256 + 96 * (end - position))
                    region_rows = rows[position:end]
                    regions = indices[region_rows]
                    support_regions = (
                        regions
                        if _bank is None
                        else np.asarray(
                            [
                                _bank.support_index(self, 3, int(region))
                                for region in regions
                            ],
                            dtype=np.int64,
                        )
                    )
                    outcomes = mapped.region_support(
                        dimension, region_rows, support_regions, work
                    )
                    failed = np.flatnonzero(~outcomes)
                    if failed.size:
                        row = int(region_rows[failed[0]])
                        index = int(indices[row])
                        source_index = (
                            index if _bank is None else _bank.source_index(3, index)
                        )
                        raise AssociationPropagationError(
                            f"PlcAssociationTransfer.target_support must certify the inherited source stratum; got bank {self.domain.source_id!r}, dimension {dimension}, target ID {int(ids[row])}, stratum {(3, source_index)!r}, support False",
                            ids[row : row + 1],
                        )
                position = end
                continue
            position += 1
            kind, index = int(dims[row]), int(indices[row])
            local_index = index
            if kind in (0, 1):
                matches = np.flatnonzero(
                    (self.vertex_indices if kind == 0 else self.edge_indices) == index
                )
                if matches.size != 1:
                    raise AssociationPropagationError(
                        f"PlcAssociationTransfer.target_source_index must name one source primitive; got dimension {kind}, index {index}, matching rows {matches!r}",
                        ids[row : row + 1],
                    )
                local_index = int(matches[0])
            if _bank is not None:
                local_index = _bank.support_index(self, kind, local_index)
            if kind == 2 and _bank is not None and _bank.mapped_facets is not None:
                supported, orientation, work[0] = _bank.mapped_facets.support(
                    mapped,
                    dimension,
                    int(row),
                    local_index,
                    work[0],
                    budget=budget,
                )
                orientations[row] = orientation
            elif dimension == 0:
                # The coordinate-map corner before its binary64 rounding.
                supported = mapped.vertex_support(
                    int(row),
                    kind,
                    local_index,
                    edge_vertices=self.edge_vertices,
                    triangle_vertices=self.triangle_vertices,
                    triangle_facets=self.triangle_facets,
                    work=work,
                    source_vertices=self.source_vertices,
                )
            else:
                supported, orientation = mapped.entity_support(
                    dimension,
                    int(row),
                    kind,
                    local_index,
                    edge_vertices=self.edge_vertices,
                    triangle_vertices=self.triangle_vertices,
                    triangle_facets=self.triangle_facets,
                    work=work,
                    source_vertices=self.source_vertices,
                )
                orientations[row] = orientation
            if not supported:
                source_index = index if _bank is None else _bank.source_index(kind, index)
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.target_support must certify the inherited source stratum; got bank {self.domain.source_id!r}, dimension {dimension}, target ID {int(ids[row])}, stratum {(kind, source_index)!r}, support {supported}",
                    ids[row : row + 1],
                )
        return _PlcCertifiedRows(orientations)

    def _publish_target_associations(
        self,
        source: result_owner.CellMeshingResult,
        target: cell_mesh_owner.CellMesh,
        vertex: GeometryAssociation | None,
        dimensions: tuple[int, ...],
        inherited: _PlcLineageTables,
        mapped: plc_mapped_support_owner.MappedPlcSupport,
        work: list[int],
        *,
        _bank: PlcBankContext | None,
        budget: CoordinateEnclosureBudget,
    ) -> _PlcAssociationBatch:
        """Publish supported target rows while retaining source namespace and predecessor IDs."""
        table, obligations = inherited.table, inherited.obligations
        output: list[GeometryAssociation] = []
        output_dimensions = (
            (0, *dimensions)
            if _bank is None
            else (() if vertex is None else (0,)) + dimensions
        )
        for dimension in output_dimensions:
            dims, indices, parent_dims, parent_ids = table[dimension]
            ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
            unresolved = (dims < 0) | (indices < 0)
            required = (
                np.ones(ids.shape, dtype=np.bool_)
                if _bank is None
                else obligations[dimension]
            )
            if np.any(unresolved & required):
                raise AssociationPropagationError(
                    f"PlcAssociationTransfer.target_strata must identify required source parents; got unresolved target IDs {ids[unresolved & required]!r}",
                    ids[unresolved & required],
                )
            selected = ~unresolved
            orientations = self._certify_target_rows(
                dimension,
                ids,
                dims,
                indices,
                selected,
                mapped,
                work,
                _bank=_bank,
                budget=budget,
            ).orientations
            previous = (
                vertex
                if dimension == 0
                else association_for_dimension(
                    source,
                    dimension,
                    namespace=None
                    if _bank is None
                    else (self.domain.source_id, self.source_revision),
                )
            )
            dims, parent_dims, parent_ids = (
                dims[selected],
                parent_dims[selected],
                parent_ids[selected],
            )
            indices = (
                indices[selected]
                if _bank is None
                else np.asarray(
                    [
                        _bank.source_index(int(kind), int(index))
                        for kind, index in zip(dims, indices[selected], strict=True)
                    ],
                    dtype=np.int64,
                )
            )
            ids, orientations = ids[selected], orientations[selected]
            if previous is None:
                raise ValueError(
                    f"PlcAssociationTransfer.predecessor_association must bind the propagated namespace; got dimension {dimension}, predecessor {previous!r}"
                )
            # Volume classification uses vertex/edge/facet/region strata; planar
            # region roles are constructed by the separate planar owner.
            source_roles = (
                GeometrySourceEntityRole.VERTEX,
                GeometrySourceEntityRole.EDGE,
                GeometrySourceEntityRole.FACET,
                GeometrySourceEntityRole.REGION,
            )
            roles = tuple(source_roles[kind] for kind in dims.tolist())
            output.append(
                GeometryAssociation(
                    GeometryAssociationKind.PIECEWISE_LINEAR,
                    self.domain.source_id,
                    self.source_revision,
                    target.entity_set(dimension).entity_set_id,
                    ids,
                    tuple(
                        entity_key(self.source_revision, role.value, index)
                        for role, index in zip(roles, indices.tolist(), strict=True)
                    ),
                    np.zeros(ids.shape, dtype=np.float64),
                    exact=True,
                    source_dimensions=dims,
                    source_indices=indices,
                    source_entity_roles=roles,
                    orientations=orientations,
                    parent_dimensions=parent_dims,
                    parent_ids=parent_ids,
                    parent_association_id=previous.association_id,
                    provenance=GeometryAssociationProvenance.LINEAGE,
                )
            )
        return _PlcAssociationBatch(tuple(output))

    @checked
    def propagate(
        self,
        source: result_owner.CellMeshingResult,
        lineage: lineage_records_owner.MeshLineage,
        target: cell_mesh_owner.CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
        _certification_request: certification_inputs_owner.MeshCertificationInputs
        | None = None,
        _geometry_transition: CellGeometryTransition | None = None,
        _prepared_target: list[certification_owner.MeshCertificationPreparedEvidence]
        | None = None,
        _bank: PlcBankContext | None = None,
        _prepared_source: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
        _mapped_target: list[plc_mapped_support_owner.MappedPlcSupport] | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        """Prove lineage support, consuming current collective premises for local planar carriers."""
        with support_budget(
            MeshCertificateLimits().maximum_work_units,
            self.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            if (_certification_request is None) != (_prepared_target is None):
                raise ValueError(
                    f"PlcAssociationTransfer.target_evidence must pair a certification request with its collector; got request present {_certification_request is not None}, collector present {_prepared_target is not None}"
                )
            if self.domain.ambient_dimension != 3 and _prepared_target is not None:
                raise ValueError(
                    f"PlcAssociationTransfer.target_evidence must collect volume evidence only for three-dimensional domains; got dimension {self.domain.ambient_dimension}"
                )
            if self.domain.ambient_dimension == 2:
                return plc_planar_support_owner.planar_propagate(
                    self,
                    source,
                    lineage,
                    target,
                    geometry=geometry,
                    embedding=embedding,
                    coverage=coverage,
                    budget=budget,
                )

            if embedding is not None or coverage is not None:
                raise ValueError(
                    f"PlcAssociationTransfer.collective_evidence must be supplied only for planar carriers; got dimension {self.domain.ambient_dimension}, embedding present {embedding is not None}, coverage present {coverage is not None}"
                )
            # A supplied source view was bound to this ledger from an ended
            # preparation; otherwise the source support is prepared here.
            prepared_source = [] if _prepared_source is None else _prepared_source
            vertex, dimensions = self.source_associations(
                source,
                _bank=_bank,
                _prepared_source=prepared_source,
                budget=budget,
            )
            if _bank is None and vertex is None:
                raise ValueError(
                    f"PlcAssociationTransfer.vertex_association must be present for volume propagation; got {vertex!r}"
                )
            require_transfer_lineage(
                source.mesh,
                lineage,
                target,
                message=f"PlcAssociationTransfer.lineage.topology_ids must bind source and target meshes; got lineage {(lineage.source_topology_id, lineage.target_topology_id)!r}, meshes {(source.mesh.topology_id, target.topology_id)!r}",
            )
            if _bank is not None and _bank.target_mapped is not None:
                # Reuse only this prepared target/request pair; no unmetered
                # coordinate-map reconstruction is allowed for another bank.
                _bank.target_mapped.prepared.require(
                    target,
                    geometry,
                    _bank.target_mapped.prepared.request
                    if _certification_request is None
                    else _certification_request,
                    budget=budget,
                )
            old = self._classified_source(
                source, prepared_source, _bank=_bank, budget=budget
            )
            work = [self.maximum_support_queries] if _bank is None else _bank.work
            inherited = self._inherit_target_strata(source, lineage, target, old)
            # All cell labels precede lower-stratum closure classification.
            mapped = self._prepare_target_support(
                source,
                target,
                geometry,
                inherited,
                lineage,
                work,
                _bank=_bank,
                _certification_request=_certification_request,
                _geometry_transition=_geometry_transition,
                budget=budget,
            ).mapped
            inherited = self._close_target_strata(target, inherited, mapped)
            inherited = self._inherit_target_vertices(
                source,
                lineage,
                target,
                old,
                inherited,
                mapped,
                work,
                _bank=_bank,
            )
            output = self._publish_target_associations(
                source,
                target,
                vertex,
                dimensions,
                inherited,
                mapped,
                work,
                _bank=_bank,
                budget=budget,
            ).associations
            if _prepared_target is not None:
                if (
                    _certification_request is None
                    or _certification_request.cell_regions is None
                ):
                    raise RuntimeError(
                        f"PlcAssociationTransfer.target_evidence.request must provide cell-region labels; got request present {_certification_request is not None}, labels present {_certification_request is not None and _certification_request.cell_regions is not None}"
                    )
                # Target support certified coverage for this exact request and
                # required it against the target; publish that one proof rather
                # than certifying identical coverage twice.
                _prepared_target.append(mapped.prepared)
            if _mapped_target is not None:
                _mapped_target.append(mapped)
            return tuple(output)


@final
class SurfaceAssociationTransfer(StrictModule, NonTrainableState):
    """Preserve parametric source identities through native coordinate restrictions.

    Support records are prepared collectively and retained immutably. Owner-local
    methods consume those records without gathering global source banks or
    initiating rank-local source collectives.
    """

    support: PreparedSurfaceSourceSupport
    policy: AssociationPropagationPolicy
    receipts: SurfaceSourceReceipts | None
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        support: PreparedSurfaceSourceSupport,
        /,
        *,
        policy: AssociationPropagationPolicy | None = None,
        receipts: SurfaceSourceReceipts | None = None,
    ) -> None:
        surface_source_association_kind(support.domain.source_kinds)
        if receipts is not None and receipts.support is not support:
            raise ValueError(
                f"SurfaceAssociationTransfer.receipts.support must be the supplied support object; got support ID {receipts.support.support_id!r}, supplied support ID {support.support_id!r}, same object {receipts.support is support}"
            )
        self.support = support
        self.policy = AssociationPropagationPolicy() if policy is None else policy
        self.receipts = receipts
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "surface-association-transfer",
                "support": support.support_id,
                "policy": self.policy.policy_id,
                "receipts": None if receipts is None else receipts.receipt_id,
            }
        )

    @checked
    def with_receipts(
        self, receipts: SurfaceSourceReceipts, /
    ) -> SurfaceAssociationTransfer:
        """Bind the current projection receipts without mutating epochs."""
        return SurfaceAssociationTransfer(
            self.support, policy=self.policy, receipts=receipts
        )

    @checked
    def source_associations(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        with support_budget(
            self.support.maximum_support_queries,
            self.support.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            return surface_association_transfer_owner.source_associations(
                self.support, source, budget=budget
            )

    @checked
    def classes(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[surface_association_transfer_owner.SurfaceEntityClasses, ...]:
        with support_budget(
            self.support.maximum_support_queries,
            self.support.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            return surface_association_transfer_owner.classes(
                self.support, source, budget=budget
            )

    @checked
    def protected_edges(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        midpoint_required: bool = True,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        with support_budget(
            self.support.maximum_support_queries,
            self.support.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            return surface_association_transfer_owner.protected_edges(
                self.support, source, midpoint_required=midpoint_required, budget=budget
            )

    @checked
    def propagate(
        self,
        source: result_owner.CellMeshingResult,
        lineage: lineage_records_owner.MeshLineage,
        target: cell_mesh_owner.CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
        deformation: MaterialChartDeformation | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        with support_budget(
            self.support.maximum_support_queries,
            self.support.maximum_support_queries,
            budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            return surface_association_transfer_owner.propagate(
                self.support,
                source,
                lineage,
                target,
                geometry=geometry,
                receipts=self.receipts,
                embedding=embedding,
                coverage=coverage,
                deformation=deformation,
                budget=budget,
            )


@final
class MappedReferenceAssociationTransfer(StrictModule, NonTrainableState):
    """Carry source-root image identities through nested mixed-cell epochs."""

    domain: MappedReferenceDomain
    maximum_support_queries: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, domain: MappedReferenceDomain, /, *, maximum_support_queries: int = 1 << 26
    ) -> None:

        if any(
            block.cell_kind not in ("tetrahedron", "prism", "hexahedron")
            for block in domain.reference_mesh.blocks
        ):
            raise ValueError(
                f"MappedReferenceAssociationTransfer.cell_kinds must be tetrahedron, prism, or hexahedron; got {tuple(block.cell_kind for block in domain.reference_mesh.blocks)!r}"
            )
        self.domain = domain
        self.maximum_support_queries = positive_integer(
            maximum_support_queries, "maximum_support_queries"
        )
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "mapped-reference-association-transfer",
                "domain": domain.domain_id,
                "maximum_support_queries": self.maximum_support_queries,
            }
        )

    @checked
    def associations(
        self,
        mesh: cell_mesh_owner.CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        with support_budget(
            self.maximum_support_queries, self.maximum_support_queries, budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            return mapped_reference_association_owner.mapped_reference_associations(
                self.domain,
                mesh,
                geometry,
                maximum_support_queries=self.maximum_support_queries,
                budget=budget,
            )

    @checked
    def source_associations(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        with support_budget(
            self.maximum_support_queries, self.maximum_support_queries, budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            certificate = source.certification
            if (
                certificate is None
                or not certificate.passed
                or not isinstance(certificate.coverage, DomainCoverageCertificate)
                or certificate.coverage.domain_id != self.domain.domain_id
            ):
                raise ValueError(
                    f"MappedReferenceAssociationTransfer.source.certification must pass and exactly cover the declared domain; got passed {None if certificate is None else certificate.passed}, expected {self.domain.domain_id!r}"
                )
            certificate.request.validate_source_integrity(budget=budget)
            certificate.coverage.binding.require(source.mesh, source.geometry)
            expected = self.associations(source.mesh, source.geometry, budget=budget)
            by_dimension = _validate_mapped_source_bindings(
                source, self.domain, expected
            ).tables
            return by_dimension[0], (1, 2, 3)

    @checked
    def classes(
        self,
        source: result_owner.CellMeshingResult,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[PlcEntityClasses, ...]:
        with support_budget(
            self.maximum_support_queries, self.maximum_support_queries, budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            self.source_associations(source, budget=budget)
            authority = self.domain.reference_mesh
            offsets = np.cumsum(
                np.asarray(
                    (0, *(authority.entity_set(degree).count for degree in range(3))),
                    dtype=np.int64,
                )
            )
            result = []
            for degree in range(4):
                item = association_for_dimension(source, degree)
                rows = item.target_rows(
                    np.asarray(source.mesh.entity_set(degree).entity_ids)
                )
                dimensions = np.asarray(item.parent_dimensions, dtype=np.int64)[rows]
                identifiers = np.asarray(item.parent_ids, dtype=np.int64)[rows]
                codes = np.empty(identifiers.shape, dtype=np.int64)
                for kind in range(4):
                    selected = dimensions == kind
                    authority_rows = entity_rows(authority, kind, identifiers[selected])
                    if np.any(authority_rows < 0):
                        missing = identifiers[selected][authority_rows < 0]
                        raise KeyError(int(missing[0]))
                    codes[selected] = offsets[kind] + authority_rows
                result.append(
                    PlcEntityClasses(
                        dimensions,
                        identifiers,
                        np.ones(dimensions.shape, dtype=np.bool_),
                        codes,
                    )
                )
            return tuple(result)

    @checked
    def propagate(
        self,
        source: result_owner.CellMeshingResult,
        lineage: lineage_records_owner.MeshLineage,
        target: cell_mesh_owner.CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        with support_budget(
            self.maximum_support_queries, self.maximum_support_queries, budget,
            maximum_memory_bytes=(
                MeshCertificateLimits().maximum_scratch_bytes
                if budget is None
                else budget.maximum_memory_bytes
            ),
        ) as budget:
            self.source_associations(source, budget=budget)
            require_transfer_lineage(
                source.mesh,
                lineage,
                target,
                message=f"MappedReferenceAssociationTransfer.lineage.topology_ids must bind source and target meshes; got lineage {(lineage.source_topology_id, lineage.target_topology_id)!r}, meshes {(source.mesh.topology_id, target.topology_id)!r}",
            )
            for degree in range(4):
                if np.any(
                    np.asarray(lineage.entity_lineage(degree).relation_kinds)
                    == lineage_records_owner.EntityLineageKind.UNKNOWN
                ):
                    raise ValueError(
                        f"MappedReferenceAssociationTransfer.lineage.relation_kinds must identify root strata; got dimension {degree}, relations {np.asarray(lineage.entity_lineage(degree).relation_kinds)!r}"
                    )
            return self.associations(target, geometry, budget=budget)


__all__ = [
    "AssociationPropagationPolicy",
    "BRepAssociationTransfer",
    "PlcAssociationTransfer",
    "MappedReferenceAssociationTransfer",
    "SurfaceAssociationTransfer",
    "associate_mesh_entities",
    "associate_mesh_vertices",
    "propagate_association",
    "rederive_association",
]
