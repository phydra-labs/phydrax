#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Audited mesh-to-geometry associations and their B-Rep classification rules.

A B-Rep association classifies mesh entities on B-Rep entities of one exact
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
children its face) and are projected, and a collapse is legal only when the kept
vertex's class lies in the closure of the removed vertex's class; illegal
collapses raise :class:`AssociationPropagationError`. Swaps change no vertex
class. Unknown lineage (provider remeshing) is re-derived by classification
transfer and projection (:func:`rederive_association`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, EntitySet
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..geometry.brep._projection import (
    brep_entity_id,
    BRepProjectionResult,
    BRepProjectionStatus,
    PreparedBRepProjection,
)
from ._scope import resolve_mesh_scope


if TYPE_CHECKING:
    from ._lineage import MeshLineage
    from ._result import CellMeshingResult


class GeometryAssociationKind(StrEnum):
    BREP = "brep"
    SURFACE = "surface"
    IMPLICIT = "implicit"


class GeometryAssociationProvenance(StrEnum):
    """How the rows were established.

    ``PROVIDER``: emitted by the generating mesher. ``LINEAGE``: propagated
    through an exact topology lineage from ``parent_association_id``.
    ``CLASSIFICATION``: derived by geometric classification and projection
    (optionally restricted by classification transfer from the parent).
    """

    PROVIDER = "provider"
    LINEAGE = "lineage"
    CLASSIFICATION = "classification"


def _unclassified_id(source_revision: str, /) -> str:
    return f"{source_revision}:unclassified"


def _rows_or(value: ArrayLike | None, fill, shape, dtype, /) -> np.ndarray:
    return (
        np.full(shape, fill, dtype=dtype) if value is None else np.asarray(value, dtype)
    )


class GeometryAssociation(StrictModule, NonTrainableState):
    """Audited map from exact mesh entities to authoritative geometry entities.

    B-Rep rows carry ``source_dimensions``/``source_indices`` (the ``BRepEntityId``
    of ``source_revision``); ``-1`` marks an unresolved row without a candidate
    (``<revision>:unclassified``). ``parameters`` hold face ``(u, v)`` or edge
    ``(t, nan)`` closest-point parameters (NaN otherwise). ``orientations`` relate
    the mesh entity's canonical vertex order to the B-Rep edge tangent or oriented
    face normal (``+1``/``-1``; ``0`` when undefined). ``parent_dimensions`` and
    ``parent_ids`` name the parent mesh entity whose class a row inherited
    (``-1`` for geometric classification). Infinite residuals are allowed only on
    unresolved rows (no closest point).
    """

    association_kind: GeometryAssociationKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    target_global_ids: Array
    source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    residuals: Array
    resolved: Array
    ambiguous: Array
    source_dimensions: Array
    source_indices: Array
    parameters: Array
    orientations: Array
    parent_dimensions: Array
    parent_ids: Array
    exact: bool = eqx.field(static=True)
    provenance: GeometryAssociationProvenance = eqx.field(static=True)
    parent_association_id: str | None = eqx.field(static=True)
    association_id: str = eqx.field(static=True)

    def __init__(
        self,
        association_kind: GeometryAssociationKind,
        source_id: str,
        source_revision: str,
        target_entity_set_id: str,
        target_global_ids: ArrayLike,
        source_entity_ids: tuple[str, ...],
        residuals: ArrayLike,
        /,
        *,
        resolved: ArrayLike | None = None,
        ambiguous: ArrayLike | None = None,
        exact: bool = False,
        source_dimensions: ArrayLike | None = None,
        source_indices: ArrayLike | None = None,
        parameters: ArrayLike | None = None,
        orientations: ArrayLike | None = None,
        parent_dimensions: ArrayLike | None = None,
        parent_ids: ArrayLike | None = None,
        parent_association_id: str | None = None,
        provenance: GeometryAssociationProvenance = GeometryAssociationProvenance.PROVIDER,
    ):
        if not isinstance(association_kind, GeometryAssociationKind):
            raise TypeError("association_kind must be GeometryAssociationKind.")
        if not isinstance(provenance, GeometryAssociationProvenance):
            raise TypeError("provenance must be GeometryAssociationProvenance.")
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        entity_set = str(target_entity_set_id).strip()
        if not source or not revision or not entity_set:
            raise ValueError("Geometry association identities must be non-empty.")
        targets = np.asarray(target_global_ids)
        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("target_global_ids must be one integer vector.")
        targets = targets.astype(np.int64, copy=False)
        if (
            targets.size == 0
            or np.any(targets < 0)
            or np.unique(targets).size != targets.size
        ):
            raise ValueError(
                "Target geometry association IDs must be unique and non-negative."
            )
        count = targets.size
        sources = tuple(str(value).strip() for value in source_entity_ids)
        distances = np.asarray(residuals, dtype=np.float64)
        if len(sources) != count or any(not value for value in sources):
            raise ValueError(
                "Source geometry IDs must match target entities and be non-empty."
            )
        resolved_ = _rows_or(resolved, True, (count,), np.bool_)
        ambiguous_ = _rows_or(ambiguous, False, (count,), np.bool_)
        if resolved_.shape != targets.shape or ambiguous_.shape != targets.shape:
            raise ValueError("Association status masks must match target entities.")
        if (
            distances.shape != targets.shape
            or np.any(np.isnan(distances))
            or np.any(distances < 0)
            or np.any(np.isinf(distances) & resolved_)
        ):
            raise ValueError(
                "Association residuals must be non-negative and aligned; infinite "
                "residuals are allowed only on unresolved rows."
            )
        if np.any(resolved_ & ambiguous_):
            raise ValueError("Resolved geometry associations cannot be ambiguous.")
        if exact and (
            not np.all(resolved_) or np.any(ambiguous_) or np.any(distances != 0.0)
        ):
            raise ValueError(
                "Exact geometry associations require zero-residual unique coverage."
            )
        dimensions, indices = _source_entities(
            association_kind, revision, sources, source_dimensions, source_indices
        )
        if association_kind is GeometryAssociationKind.BREP and np.any(
            (dimensions < 0) & resolved_
        ):
            raise ValueError("Resolved B-Rep association rows must name an entity.")
        parameters_, orientations_, parent_dimensions_, parent_ids_ = _row_metadata(
            count, parameters, orientations, parent_dimensions, parent_ids
        )
        parent = None if parent_association_id is None else str(parent_association_id)
        if parent is not None and not parent.strip():
            raise ValueError("parent_association_id must be non-empty when provided.")
        if provenance is GeometryAssociationProvenance.LINEAGE and parent is None:
            raise ValueError("Lineage associations require their parent association.")
        self.association_kind = association_kind
        self.source_id = source
        self.source_revision = revision
        self.target_entity_set_id = entity_set
        self.target_global_ids = jnp.asarray(targets)
        self.source_entity_ids = sources
        self.residuals = jnp.asarray(distances)
        self.resolved = jnp.asarray(resolved_)
        self.ambiguous = jnp.asarray(ambiguous_)
        self.source_dimensions = jnp.asarray(dimensions)
        self.source_indices = jnp.asarray(indices)
        self.parameters = jnp.asarray(parameters_)
        self.orientations = jnp.asarray(orientations_)
        self.parent_dimensions = jnp.asarray(parent_dimensions_)
        self.parent_ids = jnp.asarray(parent_ids_)
        self.exact = bool(exact)
        self.provenance = provenance
        self.parent_association_id = parent
        self.association_id = canonical_fingerprint(
            {
                "kind": "geometry-association",
                "association_kind": association_kind.value,
                "source_id": source,
                "source_revision": revision,
                "target_entity_set_id": entity_set,
                "target_global_ids": array_tree_fingerprint(targets),
                "source_entity_ids": sources,
                "residuals": array_tree_fingerprint(distances),
                "resolved": array_tree_fingerprint(resolved_),
                "ambiguous": array_tree_fingerprint(ambiguous_),
                "parameters": array_tree_fingerprint(parameters_),
                "orientations": array_tree_fingerprint(orientations_),
                "parent_dimensions": array_tree_fingerprint(parent_dimensions_),
                "parent_ids": array_tree_fingerprint(parent_ids_),
                "exact": bool(exact),
                "provenance": provenance.value,
                "parent_association_id": parent,
            }
        )

    @property
    def complete(self) -> bool:
        return bool(np.all(np.asarray(self.resolved))) and not bool(
            np.any(np.asarray(self.ambiguous))
        )

    @property
    def maximum_residual(self) -> float:
        """Largest residual over resolved rows (0 when none is resolved)."""
        resolved = np.asarray(self.resolved)
        residuals = np.asarray(self.residuals)[resolved]
        return float(np.max(residuals)) if residuals.size else 0.0

    def validate_target(self, entity_set: EntitySet, /) -> None:
        """Require an exact target binding, not merely resolved row statuses."""
        if not isinstance(entity_set, EntitySet):
            raise TypeError("entity_set must be EntitySet.")
        if self.target_entity_set_id != entity_set.entity_set_id:
            raise ValueError("Geometry association targets a different entity set.")
        if not np.all(
            np.isin(np.asarray(self.target_global_ids), np.asarray(entity_set.entity_ids))
        ):
            raise ValueError("Geometry association contains undeclared target IDs.")


def _row_metadata(
    count: int,
    parameters: ArrayLike | None,
    orientations: ArrayLike | None,
    parent_dimensions: ArrayLike | None,
    parent_ids: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validated closest-point parameters, orientations, and parents of every row."""
    parameters_ = _rows_or(parameters, np.nan, (count, 2), np.float64)
    orientations_ = _rows_or(orientations, 0, (count,), np.int8)
    parent_dimensions_ = _rows_or(parent_dimensions, -1, (count,), np.int8)
    parent_ids_ = _rows_or(parent_ids, -1, (count,), np.int64)
    if (
        parameters_.shape != (count, 2)
        or np.any(np.isinf(parameters_))
        or orientations_.shape != (count,)
        or parent_dimensions_.shape != (count,)
        or parent_ids_.shape != (count,)
    ):
        raise ValueError(
            "Association parameters, orientations, and parents must be aligned."
        )
    if not np.all(np.isin(orientations_, (-1, 0, 1))):
        raise ValueError("Association orientations must be -1, 0, or 1.")
    if (
        np.any((parent_dimensions_ < -1) | (parent_dimensions_ > 3))
        or np.any(parent_ids_ < -1)
        or np.any((parent_dimensions_ < 0) != (parent_ids_ < 0))
    ):
        raise ValueError("Association parents must pair a dimension with an ID.")
    return parameters_, orientations_, parent_dimensions_, parent_ids_


def _source_entities(
    kind: GeometryAssociationKind,
    revision: str,
    sources: tuple[str, ...],
    dimensions: ArrayLike | None,
    indices: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(sources)
    if kind is not GeometryAssociationKind.BREP:
        if dimensions is not None or indices is not None:
            raise ValueError("Only B-Rep associations carry B-Rep entity dimensions.")
        return np.full((count,), -1, np.int8), np.full((count,), -1, np.int32)
    if dimensions is None or indices is None:
        raise ValueError("B-Rep associations require source dimensions and indices.")
    dimensions_ = np.asarray(dimensions)
    indices_ = np.asarray(indices)
    if not np.issubdtype(dimensions_.dtype, np.integer) or not np.issubdtype(
        indices_.dtype, np.integer
    ):
        raise TypeError("B-Rep source dimensions and indices must be integers.")
    if dimensions_.shape != (count,) or indices_.shape != (count,):
        raise ValueError("B-Rep source dimensions and indices must align with rows.")
    if np.any((dimensions_ < -1) | (dimensions_ > 3)) or np.any(
        (dimensions_ >= 0) != (indices_ >= 0)
    ):
        raise ValueError("B-Rep source entities must pair a dimension with an index.")
    expected = tuple(
        brep_entity_id(revision, dimension, index)
        if dimension >= 0
        else _unclassified_id(revision)
        for dimension, index in zip(dimensions_.tolist(), indices_.tolist(), strict=True)
    )
    if expected != sources:
        raise ValueError(
            "B-Rep source entity IDs must be <revision>:<kind>:<index> of their rows."
        )
    return dimensions_.astype(np.int8), indices_.astype(np.int32)


# -- policy and errors ---------------------------------------------------------


class AssociationPropagationPolicy(StrictModule, NonTrainableState):
    """Tolerances of association propagation and re-derivation.

    ``classification_tolerance`` is the absolute distance within which geometric
    classification places a point on a B-Rep entity. ``maximum_residual`` marks
    rows whose projection residual exceeds it unresolved; ``None`` keeps residuals
    as evidence only.
    """

    classification_tolerance: float = eqx.field(static=True)
    maximum_residual: float | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        classification_tolerance: float = 1.0e-6,
        maximum_residual: float | None = None,
    ):
        tolerance = float(classification_tolerance)
        maximum = None if maximum_residual is None else float(maximum_residual)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("classification_tolerance must be finite and positive.")
        if maximum is not None and (not np.isfinite(maximum) or maximum < 0.0):
            raise ValueError("maximum_residual must be finite and non-negative or None.")
        self.classification_tolerance = tolerance
        self.maximum_residual = maximum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "association-propagation-policy",
                "classification_tolerance": tolerance,
                "maximum_residual": maximum,
            }
        )


class AssociationPropagationError(ValueError):
    """A topology change violates the B-Rep classification rules."""

    def __init__(self, message: str, target_ids: ArrayLike, /):
        super().__init__(message)
        self.target_ids = np.asarray(target_ids, dtype=np.int64)


# -- mesh topology helpers -------------------------------------------------------


def _compose(
    first: np.ndarray, second: np.ndarray, /, *, unique: bool = True
) -> np.ndarray:
    """Compose ``(a, b)`` pairs with ``(b, c)`` pairs into ``(a, c)`` pairs.

    With ``unique=False`` every path is kept, so multiplicities count the
    intermediate ``b`` connecting each ``(a, c)``.
    """
    if first.shape[0] == 0 or second.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    order = np.argsort(second[:, 0], kind="stable")
    keys = second[order, 0]
    start = np.searchsorted(keys, first[:, 1], side="left")
    counts = np.searchsorted(keys, first[:, 1], side="right") - start
    owner = np.repeat(np.arange(first.shape[0]), counts)
    offsets = np.cumsum(counts) - counts
    position = np.repeat(start - offsets, counts) + np.arange(owner.size)
    pairs = np.stack((first[owner, 0], second[order[position], 1]), axis=1)
    return np.unique(pairs, axis=0).reshape(-1, 2) if unique else pairs


def _incidence_pairs(mesh: CellMesh, low: int, high: int, /) -> np.ndarray:
    """``(low_row, high_row)`` pairs with the low entity in the high entity's closure."""
    if low == high:
        rows = np.arange(mesh.entity_set(low).count, dtype=np.int64)
        return np.stack((rows, rows), axis=1)
    pairs = None
    for degree in range(low, high):
        relation = mesh.topology.incidences[degree].relation
        valid = np.asarray(relation.valid, dtype=np.bool_).reshape(-1)
        step = np.stack(
            (
                np.asarray(relation.source_indices, dtype=np.int64).reshape(-1)[valid],
                np.asarray(relation.target_indices, dtype=np.int64).reshape(-1)[valid],
            ),
            axis=1,
        )
        pairs = step if pairs is None else _compose(pairs, step)
    return np.unique(pairs, axis=0).reshape(-1, 2)


def _boundary_mask(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Entities of ``dimension`` in the closure of mesh boundary facets."""
    top = mesh.topological_dimension
    facets = _incidence_pairs(mesh, top - 1, top)
    boundary_facets = np.flatnonzero(
        np.bincount(facets[:, 0], minlength=mesh.entity_set(top - 1).count) == 1
    )
    mask = np.zeros((mesh.entity_set(dimension).count,), dtype=np.bool_)
    if dimension == top:
        return mask
    pairs = _incidence_pairs(mesh, dimension, top - 1)
    mask[pairs[np.isin(pairs[:, 1], boundary_facets), 0]] = True
    return mask


def _vertex_rows(mesh: CellMesh, identifiers: np.ndarray, /) -> np.ndarray:
    """Mesh vertex row of each global ID (-1 when absent)."""
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    position = np.minimum(np.searchsorted(vertex_ids[order], identifiers), order.size - 1)
    return np.where(vertex_ids[order[position]] == identifiers, order[position], -1)


def _entity_rows(
    mesh: CellMesh, dimension: int, identifiers: np.ndarray, /
) -> np.ndarray:
    entity_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    order = np.argsort(entity_ids, kind="stable")
    position = np.minimum(np.searchsorted(entity_ids[order], identifiers), order.size - 1)
    return np.where(entity_ids[order[position]] == identifiers, order[position], -1)


def _ordered_vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray | None:
    """Canonical vertex rows of every entity of ``dimension`` (simplex routes)."""
    connectivity = mesh.connectivity
    if dimension == 1 and isinstance(
        connectivity, (PolygonalConnectivity, TetrahedralConnectivity)
    ):
        return np.asarray(connectivity.edges, dtype=np.int64)
    if dimension == 2 and isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.faces, dtype=np.int64)
    if dimension == mesh.topological_dimension == 2 and all(
        block.cell_kind == "triangle" for block in mesh.blocks
    ):
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        rows = _entity_rows(mesh, dimension, identifiers)
        ordered = np.empty_like(cells)
        ordered[rows] = cells
        return ordered
    return None


# -- classification --------------------------------------------------------------


_RESOLVED_STATUS = (int(BRepProjectionStatus.UNIQUE), int(BRepProjectionStatus.SEAM))


class _Classes(NamedTuple):
    """B-Rep class of every mesh entity row of one dimension (-1 unclassified)."""

    dimensions: np.ndarray
    indices: np.ndarray
    status: np.ndarray

    @property
    def resolved(self) -> np.ndarray:
        return np.isin(self.status, _RESOLVED_STATUS)


def _vertex_classes(mesh: CellMesh, association: GeometryAssociation, /) -> _Classes:
    """Vertex classes in mesh row order; the association must cover every vertex."""
    rows = _entity_rows(mesh, 0, np.asarray(association.target_global_ids))
    count = mesh.coordinates.shape[0]
    if np.any(rows < 0) or np.unique(rows).size != count or rows.size != count:
        raise ValueError("The vertex association must cover exactly the mesh vertices.")
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
    return _Classes(dims, indices, status)


def _centroids(mesh: CellMesh, members: np.ndarray, count: int, /) -> np.ndarray:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    totals = np.zeros((count, points.shape[1]))
    np.add.at(totals, members[:, 0], points[members[:, 1]])
    return totals / np.maximum(np.bincount(members[:, 0], minlength=count), 1)[:, None]


def _candidate_residuals(
    projection: PreparedBRepProjection,
    points: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    /,
) -> np.ndarray:
    """Distance of each point to its candidate entity (solids: 0 inside, else inf)."""
    residuals = np.full((points.shape[0],), np.inf)
    lower = np.flatnonzero(dimensions <= 2)
    if lower.size:
        result = projection.project(points[lower], dimensions[lower], indices[lower])
        residuals[lower] = np.where(
            np.asarray(result.status) == BRepProjectionStatus.FAILED,
            np.inf,
            np.asarray(result.residuals),
        )
    solid = np.flatnonzero(dimensions == 3)
    if solid.size:
        located = projection.locate_solids(points[solid])
        inside = (np.asarray(located.dimensions) == 3) & (
            np.asarray(located.indices) == indices[solid]
        )
        residuals[solid] = np.where(inside, 0.0, np.inf)
    return residuals


def _lowest_containers(
    projection: PreparedBRepProjection,
    members: np.ndarray,
    member_classes: _Classes,
    count: int,
    low: int,
    high: np.ndarray,
    within: np.ndarray | None,
    centroids: np.ndarray,
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
    member_dims = member_classes.dimensions[members[:, 1]]
    classified = member_dims >= 0
    failed = np.bincount(members[~classified, 0], minlength=count) > 0
    failed |= np.bincount(members[:, 0], minlength=count) == 0
    codes = projection.entity_codes(
        member_dims[classified], member_classes.indices[members[classified, 1]]
    )
    owner_class = np.unique(
        np.stack((members[classified, 0], codes), axis=1), axis=0
    ).reshape(-1, 2)
    if owner_class.shape[0] == 0:
        return _Classes(dims, indices, status)
    class_count = np.bincount(owner_class[:, 0], minlength=count)
    distinct, inverse = np.unique(owner_class[:, 1], return_inverse=True)
    row, container_dims, container_indices = projection.containers(
        *projection.code_entities(distinct)
    )
    keep = container_dims >= low
    reach = _compose(
        np.stack((owner_class[:, 0], inverse.reshape(-1)), axis=1),
        np.stack(
            (
                row[keep],
                projection.entity_codes(container_dims[keep], container_indices[keep]),
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
        joined = _compose(
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
        ties = np.bincount(owners, minlength=count)
        residual = np.zeros(owners.shape)
        tied = ties[owners] > 1
        if np.any(tied):
            residual[tied] = _candidate_residuals(
                projection, centroids[owners[tied]], at_dims[tied], at_indices[tied]
            )
        order = np.lexsort((at_indices, residual, owners))
        owners, at_dims, at_indices, residual = (
            owners[order],
            at_dims[order],
            at_indices[order],
            residual[order],
        )
        first = np.unique(owners, return_index=True)[1]
        second = np.minimum(first + 1, owners.size - 1)
        distinguishable = (ties[owners[first]] == 1) | (
            (owners[second] == owners[first])
            & (residual[second] > residual[first] + projection.policy.ambiguity_tolerance)
        )
        rows = owners[first]
        dims[rows] = at_dims[first]
        indices[rows] = at_indices[first]
        status[rows] = np.where(
            distinguishable, BRepProjectionStatus.UNIQUE, BRepProjectionStatus.AMBIGUOUS
        )
    dims[failed] = -1
    indices[failed] = -1
    status[failed] = BRepProjectionStatus.FAILED
    return _Classes(dims, indices, status)


def _classify_mesh(
    mesh: CellMesh,
    vertex_classes: _Classes,
    projection: PreparedBRepProjection,
    /,
) -> tuple[_Classes, ...]:
    """B-Rep classes of every mesh entity dimension ``0..D`` (module rules)."""
    top = mesh.topological_dimension
    classes: list[_Classes] = [vertex_classes] * (top + 1)
    members = _incidence_pairs(mesh, 0, top)[:, ::-1]
    count = mesh.entity_set(top).count
    classes[top] = _lowest_containers(
        projection,
        members,
        vertex_classes,
        count,
        top,
        np.full((count,), top),
        None,
        _centroids(mesh, members, count),
    )
    for dimension in range(top - 1, 0, -1):
        classes[dimension] = _classify_level(
            mesh, projection, vertex_classes, classes[dimension + 1], dimension
        )
    return tuple(classes)


def _classify_level(
    mesh: CellMesh,
    projection: PreparedBRepProjection,
    vertex_classes: _Classes,
    upper: _Classes,
    dimension: int,
    /,
) -> _Classes:
    """Classify entities of ``dimension`` from their classified cofaces.

    The cofaces on the lowest-dimensional adjacent classes decide: one such class
    that the entity does not bound (a coface class of higher dimension, or exactly
    two cofaces on it) is inherited; otherwise the entity lies on the lowest
    B-Rep entity containing its vertex classes within the closure of every
    adjacent lowest class.
    """
    count = mesh.entity_set(dimension).count
    pairs = _incidence_pairs(mesh, dimension, dimension + 1)
    good = np.isin(upper.status[pairs[:, 1]], _RESOLVED_STATUS)
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
                projection.entity_codes(upper_dims[at], upper.indices[pairs[at, 1]]),
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
    members = _incidence_pairs(mesh, 0, dimension)[:, ::-1]
    # Only entities bounding their coface classes search for a lower class.
    bounding = members[~inherited[members[:, 0]]]
    bounded = _lowest_containers(
        projection,
        bounding,
        vertex_classes,
        count,
        dimension,
        np.minimum(lowest, 4) - 1,
        adjacent,
        _centroids(mesh, members, count),
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
    return _Classes(dims.astype(np.int64), indices.astype(np.int64), status)


def _orientations(
    mesh: CellMesh,
    dimension: int,
    classes: _Classes,
    projected: BRepProjectionResult,
    /,
) -> np.ndarray:
    """Relation of each entity's canonical vertex order to its B-Rep orientation."""
    result = np.zeros((classes.dimensions.size,), dtype=np.int8)
    ordered = _ordered_vertices(mesh, dimension)
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
    projection: PreparedBRepProjection,
    mesh: CellMesh,
    dimension: int,
    classes: _Classes,
    points: np.ndarray,
    policy: AssociationPropagationPolicy,
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
    projected = projection.project(
        points[lower], classes.dimensions[lower], classes.indices[lower]
    )
    residuals = np.where(classes.dimensions == 3, 0.0, np.inf)
    parameters = np.full((count, 2), np.nan)
    projection_status = np.where(
        classes.dimensions == 3, BRepProjectionStatus.UNIQUE, BRepProjectionStatus.FAILED
    ).astype(np.int8)
    residuals[lower] = np.asarray(projected.residuals)
    parameters[lower] = np.asarray(projected.parameters)
    projection_status[lower] = np.asarray(projected.status)
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
    )
    ambiguous = (classes.status == BRepProjectionStatus.AMBIGUOUS) | (
        projection_status == BRepProjectionStatus.AMBIGUOUS
    )
    resolved = (
        np.isin(classes.status, _RESOLVED_STATUS)
        & np.isin(projection_status, _RESOLVED_STATUS)
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
            brep_entity_id(projection.source_revision, entity, index)
            if entity >= 0
            else _unclassified_id(projection.source_revision)
            for entity, index in zip(
                classes.dimensions.tolist(), classes.indices.tolist(), strict=True
            )
        ),
        residuals,
        resolved=resolved,
        ambiguous=ambiguous & ~resolved,
        source_dimensions=classes.dimensions,
        source_indices=classes.indices,
        parameters=parameters,
        orientations=_orientations(mesh, dimension, classes, full),
        parent_dimensions=parent_dimensions,
        parent_ids=parent_ids,
        parent_association_id=parent_association_id,
        provenance=provenance,
    )


def _scatter(rows: np.ndarray, values: np.ndarray, shape: tuple[int, ...], /):
    output = np.full(shape, np.nan)
    output[rows] = values
    return output


def _require_brep(
    association: GeometryAssociation,
    projection: PreparedBRepProjection,
    mesh: CellMesh,
    dimension: int,
    /,
) -> None:
    if not isinstance(association, GeometryAssociation):
        raise TypeError("association must be GeometryAssociation.")
    if not isinstance(projection, PreparedBRepProjection):
        raise TypeError("projection must be PreparedBRepProjection.")
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if association.association_kind is not GeometryAssociationKind.BREP:
        raise ValueError("B-Rep classification requires a B-Rep association.")
    if association.source_revision != projection.source_revision:
        raise ValueError("The association and projection bind different revisions.")
    if association.target_entity_set_id != mesh.entity_set(dimension).entity_set_id:
        raise ValueError("The association does not target this mesh's entities.")
    if mesh.ambient_dimension != projection.ambient_dimension:
        raise ValueError("Mesh and projection ambient dimensions differ.")


def _policy(policy: AssociationPropagationPolicy, /) -> AssociationPropagationPolicy:
    if not isinstance(policy, AssociationPropagationPolicy):
        raise TypeError("policy must be AssociationPropagationPolicy.")
    return policy


# -- public classification ----------------------------------------------------------


def _mesh_entity_classes(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: PreparedBRepProjection,
    /,
) -> tuple[_Classes, ...]:
    """Host B-Rep classes ``(dimensions, indices, status)`` of mesh dimensions ``0..D``.

    ``association`` is the complete vertex association of ``mesh``. Consumers
    (adaptation constraints, curving) use the classes to protect, constrain, and
    project entities by the module rules.
    """
    _require_brep(association, projection, mesh, 0)
    return _classify_mesh(mesh, _vertex_classes(mesh, association), projection)


def associate_mesh_entities(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: PreparedBRepProjection,
    dimension: int,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Derive the B-Rep association of mesh entities of ``dimension`` (1..D).

    Classes follow the module rules from the complete vertex ``association``;
    residuals and parameters come from projecting each entity centroid (region
    rows carry zero residual), orientations from the canonical vertex order.
    """
    policy_ = _policy(policy)
    target = int(dimension)
    if not 1 <= target <= mesh.topological_dimension:
        raise ValueError("dimension must lie in 1..D; vertex associations are direct.")
    classes = _mesh_entity_classes(mesh, association, projection)[target]
    members = _incidence_pairs(mesh, 0, target)[:, ::-1]
    count = mesh.entity_set(target).count
    return _brep_association(
        projection,
        mesh,
        target,
        classes,
        _centroids(mesh, members, count),
        policy_,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=association.association_id,
    )


def _geometric_vertex_classes(
    mesh: CellMesh,
    projection: PreparedBRepProjection,
    tolerance: float,
    groups: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
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
    boundary = _boundary_mask(mesh, 0)
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)

    def assign(rows: np.ndarray, result: BRepProjectionResult, /) -> None:
        dims[rows] = np.asarray(result.dimensions, dtype=np.int64)
        indices[rows] = np.asarray(result.indices, dtype=np.int64)
        status[rows] = np.asarray(result.status)

    rows = np.flatnonzero(boundary)
    if rows.size:
        if groups is None:
            options = {}
        else:
            vertex_groups, allowed_groups, allowed_dims, allowed_indices = groups
            options = {
                "groups": vertex_groups[rows],
                "allowed_groups": allowed_groups,
                "allowed_dimensions": allowed_dims,
                "allowed_indices": allowed_indices,
            }
        assign(
            rows,
            projection.classify(
                points[rows],
                tolerance=tolerance,
                maximum_dimension=min(top - 1, 2),
                **options,
            ),
        )
    rows = np.flatnonzero(~boundary)
    if rows.size:
        assign(rows, projection.classify(points[rows], tolerance=tolerance))
        if top == 3:
            inside = rows[dims[rows] < 0]
            if inside.size:
                assign(inside, projection.locate_solids(points[inside]))
    return _Classes(dims, indices, status)


def associate_mesh_vertices(
    mesh: CellMesh,
    projection: PreparedBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Classify every mesh vertex geometrically and project it onto its class."""
    policy_ = _policy(policy)
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(projection, PreparedBRepProjection):
        raise TypeError("projection must be PreparedBRepProjection.")
    if mesh.ambient_dimension != projection.ambient_dimension:
        raise ValueError("Mesh and projection ambient dimensions differ.")
    classes = _geometric_vertex_classes(
        mesh, projection, policy_.classification_tolerance, None
    )
    return _brep_association(
        projection,
        mesh,
        0,
        classes,
        np.asarray(mesh.coordinates, dtype=np.float64),
        policy_,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=None,
    )


# -- propagation through lineage --------------------------------------------------


def _child_sources(
    lineage: MeshLineage,
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    created: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest-dimensional source entity ``(dimension, row)`` each created vertex lies in.

    A vertex on a target edge (face) refined from a source edge (face) lies in
    that source entity; otherwise its split parents span the source simplex.
    """
    from ._lineage import EntityLineageKind
    from ._topology_edit import entity_keys, key_rows

    top = source_mesh.topological_dimension
    dims = np.full((created.size,), -1, dtype=np.int64)
    rows = np.full((created.size,), -1, dtype=np.int64)
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    refined = (int(EntityLineageKind.REFINED_FROM), int(EntityLineageKind.SPLIT_FROM))
    for dimension in range(1, top):
        record = lineage.entity_lineage(dimension)
        kinds = np.asarray(record.relation_kinds)
        mask = np.isin(kinds, refined)
        source_rows = _entity_rows(
            source_mesh, dimension, np.asarray(record.source_global_ids)[mask]
        )
        target_rows = _entity_rows(
            target_mesh, dimension, np.asarray(record.target_global_ids)[mask]
        )
        vertices = _incidence_pairs(target_mesh, 0, dimension)
        entity_vertices = _compose(
            np.stack((np.arange(target_rows.size), target_rows), axis=1),
            vertices[:, ::-1],
        )
        vertex_ids = target_ids[entity_vertices[:, 1]]
        position = np.searchsorted(created, vertex_ids)
        position = np.minimum(position, created.size - 1)
        hit = (created.size > 0) & (created[position] == vertex_ids)
        open_ = dims[position[hit]] < 0
        chosen = position[hit][open_]
        dims[chosen] = dimension
        rows[chosen] = source_rows[entity_vertices[hit, 0][open_]]
    record = lineage.entity_lineage(0)
    split = np.asarray(record.relation_kinds) == EntityLineageKind.SPLIT_FROM
    parents = np.stack(
        (
            np.asarray(record.target_global_ids)[split],
            np.asarray(record.source_global_ids)[split],
        ),
        axis=1,
    )
    open_rows = np.flatnonzero(dims < 0)
    if open_rows.size and parents.size:
        parents = parents[np.isin(parents[:, 0], created[open_rows])]
        parents = parents[np.lexsort((parents[:, 1], parents[:, 0]))]
        owners, starts, sizes = np.unique(
            parents[:, 0], return_index=True, return_counts=True
        )
        for size in np.unique(sizes).tolist():
            if size - 1 > top or size < 2:
                continue
            selected = sizes == size
            keys = np.stack(
                [parents[starts[selected] + offset, 1] for offset in range(size)], axis=1
            )
            if size - 1 == top:
                cells = np.concatenate(
                    [
                        np.asarray(block.vertices, dtype=np.int64)
                        for block in source_mesh.blocks
                    ]
                )
                identifiers = np.concatenate(
                    [
                        np.asarray(block.global_ids, dtype=np.int64)
                        for block in source_mesh.blocks
                    ]
                )
                table = np.sort(
                    np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)[cells],
                    axis=1,
                )
                found = key_rows(table, keys)
                entity = np.where(
                    found >= 0,
                    _entity_rows(source_mesh, top, identifiers[np.maximum(found, 0)]),
                    -1,
                )
            else:
                entity = key_rows(entity_keys(source_mesh, size - 1), keys)
            position = np.searchsorted(created, owners[selected])
            dims[position] = np.where(entity >= 0, size - 1, -1)
            rows[position] = entity
    return dims, rows


def propagate_association(
    association: GeometryAssociation,
    lineage: MeshLineage,
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    projection: PreparedBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Carry a complete B-Rep vertex association through one exact topology lineage.

    Preserved vertices keep their class and projection; relocated vertices keep
    their class and are re-projected; created vertices inherit the class of the
    source entity they lie in and are projected; vertices without any parent are
    classified geometrically. Raises :class:`AssociationPropagationError` for a
    collapse whose kept vertex class is not in the closure of the removed class.
    """
    from ._lineage import EntityLineageKind, MeshLineage

    policy_ = _policy(policy)
    _require_brep(association, projection, source_mesh, 0)
    if not isinstance(lineage, MeshLineage):
        raise TypeError("lineage must be MeshLineage.")
    if not isinstance(target_mesh, CellMesh):
        raise TypeError("target_mesh must be CellMesh.")
    if (
        lineage.source_topology_id != source_mesh.topology_id
        or lineage.target_topology_id != target_mesh.topology_id
    ):
        raise ValueError("The lineage does not relate these source and target meshes.")
    source_classes = _vertex_classes(source_mesh, association)
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    record = lineage.entity_lineage(0)
    kinds = np.asarray(record.relation_kinds)
    sources = np.asarray(record.source_global_ids, dtype=np.int64)
    targets = np.asarray(record.target_global_ids, dtype=np.int64)
    _check_collapses(
        projection,
        source_classes,
        _entity_rows(source_mesh, 0, sources[kinds == EntityLineageKind.COLLAPSED_INTO]),
        _entity_rows(source_mesh, 0, targets[kinds == EntityLineageKind.COLLAPSED_INTO]),
        targets[kinds == EntityLineageKind.COLLAPSED_INTO],
    )
    count = target_ids.size
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    parent_dims = np.full((count,), -1, dtype=np.int8)
    parent_ids = np.full((count,), -1, dtype=np.int64)
    kept = np.flatnonzero(_entity_rows(source_mesh, 0, target_ids) >= 0)
    kept_source = _entity_rows(source_mesh, 0, target_ids[kept])
    dims[kept] = source_classes.dimensions[kept_source]
    indices[kept] = source_classes.indices[kept_source]
    status[kept] = source_classes.status[kept_source]
    parent_dims[kept] = 0
    parent_ids[kept] = target_ids[kept]
    created_rows = np.flatnonzero(_entity_rows(source_mesh, 0, target_ids) < 0)
    created = target_ids[created_rows]
    order = np.argsort(created, kind="stable")
    created_rows, created = created_rows[order], created[order]
    if created.size:
        entity_dims, entity_rows = _child_sources(
            lineage, source_mesh, target_mesh, created
        )
        found = entity_dims >= 0
        if np.any(found):
            source_levels = _classify_mesh(source_mesh, source_classes, projection)
            for level in np.unique(entity_dims[found]).tolist():
                selected = np.flatnonzero(entity_dims == level)
                level_classes = source_levels[level]
                rows = entity_rows[selected]
                target_rows = created_rows[selected]
                dims[target_rows] = level_classes.dimensions[rows]
                indices[target_rows] = level_classes.indices[rows]
                status[target_rows] = level_classes.status[rows]
                parent_dims[target_rows] = level
                parent_ids[target_rows] = np.asarray(
                    source_mesh.entity_set(level).entity_ids, dtype=np.int64
                )[rows]
        orphans = created_rows[~found]
        if orphans.size:
            geometric = projection.classify(
                np.asarray(target_mesh.coordinates, dtype=np.float64)[orphans],
                tolerance=policy_.classification_tolerance,
            )
            dims[orphans] = np.asarray(geometric.dimensions, dtype=np.int64)
            indices[orphans] = np.asarray(geometric.indices, dtype=np.int64)
            status[orphans] = np.asarray(geometric.status)
    return _brep_association(
        projection,
        target_mesh,
        0,
        _Classes(dims, indices, status),
        np.asarray(target_mesh.coordinates, dtype=np.float64),
        policy_,
        provenance=GeometryAssociationProvenance.LINEAGE,
        parent_association_id=association.association_id,
        parent_dimensions=parent_dims,
        parent_ids=parent_ids,
    )


def _check_collapses(
    projection: PreparedBRepProjection,
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
        raise ValueError("Collapse relations must connect source vertices.")
    removed_dims = classes.dimensions[removed]
    kept_dims = classes.dimensions[kept]
    known = (removed_dims >= 0) & (kept_dims >= 0)
    legal = np.zeros(removed.shape, dtype=np.bool_)
    legal[known] = projection.contains(
        removed_dims[known],
        classes.indices[removed][known],
        kept_dims[known],
        classes.indices[kept][known],
    )
    if not np.all(legal):
        raise AssociationPropagationError(
            "A collapse removed a vertex whose B-Rep class does not bound the kept "
            "vertex's class.",
            kept_ids[~legal],
        )


# -- re-derivation after unknown lineage --------------------------------------------


def _facet_signatures(
    result: CellMeshingResult, rows: int, /
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
    source: CellMeshingResult,
    target: CellMeshingResult,
    source_classes: tuple[_Classes, ...],
    projection: PreparedBRepProjection,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    boundary = _boundary_mask(source.mesh, top - 1)
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
    )
    target_boundary = _boundary_mask(target.mesh, top - 1)
    vertex_facets = _incidence_pairs(target.mesh, 0, top - 1)
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
    group_codes = _compose(group_signature, allowed)
    dims, indices = projection.code_entities(group_codes[:, 1])
    return groups.reshape(-1), group_codes[:, 0], dims, indices


def _members_closure(
    projection: PreparedBRepProjection,
    signatures: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    /,
) -> np.ndarray:
    """Unique ``(signature, entity_code)`` of every classified facet and its closure."""
    owner, member_dims, member_indices = projection.members(dimensions, indices)
    return np.unique(
        np.stack(
            (signatures[owner], projection.entity_codes(member_dims, member_indices)),
            axis=1,
        ),
        axis=0,
    ).reshape(-1, 2)


def rederive_association(
    association: GeometryAssociation,
    source: CellMeshingResult,
    target: CellMeshingResult,
    projection: PreparedBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Re-derive a vertex association on a remeshed target with unknown lineage.

    Classification transfer: source boundary facets are classified from the
    source vertex ``association``; target boundary vertices may classify only on
    the B-Rep classes (with closures) of source facets sharing the organization
    signature (facet patches, zones, labels, which remeshing providers carry as
    references/class IDs) of their incident target facets. Each vertex then takes
    the lowest-dimensional admissible entity within the classification tolerance
    and is projected; residuals are the evidence.
    """
    from ._result import CellMeshingResult

    policy_ = _policy(policy)
    if not isinstance(source, CellMeshingResult) or not isinstance(
        target, CellMeshingResult
    ):
        raise TypeError("source and target must be CellMeshingResult values.")
    _require_brep(association, projection, source.mesh, 0)
    if target.mesh.topological_dimension != source.mesh.topological_dimension:
        raise ValueError("Re-derivation requires one topological dimension.")
    source_levels = _classify_mesh(
        source.mesh, _vertex_classes(source.mesh, association), projection
    )
    groups = _transfer_groups(source, target, source_levels, projection)
    classes = _geometric_vertex_classes(
        target.mesh, projection, policy_.classification_tolerance, groups
    )
    return _brep_association(
        projection,
        target.mesh,
        0,
        classes,
        np.asarray(target.mesh.coordinates, dtype=np.float64),
        policy_,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=association.association_id,
    )


# -- result-level transfer ---------------------------------------------------------


def _target_dimension(mesh: CellMesh, association: GeometryAssociation, /) -> int:
    for dimension in range(mesh.topological_dimension + 1):
        if mesh.entity_set(dimension).entity_set_id == association.target_entity_set_id:
            return dimension
    raise ValueError("A B-Rep association does not target the source mesh entities.")


@final
class BRepAssociationTransfer(StrictModule, NonTrainableState):
    """Carries a result's B-Rep associations through topology changes.

    Binds one prepared projection and propagation policy. A source result must
    carry exactly one vertex association of the projection's revision (covering
    every vertex) and may carry B-Rep associations of higher mesh dimensions; the
    target receives the propagated or re-derived vertex association and the
    associations of the same higher dimensions derived from it.
    """

    projection: PreparedBRepProjection
    policy: AssociationPropagationPolicy
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        projection: PreparedBRepProjection,
        /,
        *,
        policy: AssociationPropagationPolicy | None = None,
    ):
        if not isinstance(projection, PreparedBRepProjection):
            raise TypeError("projection must be PreparedBRepProjection.")
        policy_ = AssociationPropagationPolicy() if policy is None else _policy(policy)
        self.projection = projection
        self.policy = policy_
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "brep-association-transfer",
                "projection": projection.projection_id,
                "policy": policy_.policy_id,
            }
        )

    def source_associations(
        self, source: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        """The source vertex association and the higher dimensions to rebuild."""
        from ._result import CellMeshingResult

        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        dimensions = []
        vertex = None
        for association in source.associations:
            if (
                association.association_kind is not GeometryAssociationKind.BREP
                or association.source_revision != self.projection.source_revision
            ):
                raise ValueError(
                    "B-Rep association transfer carries associations of its revision only."
                )
            dimension = _target_dimension(source.mesh, association)
            if dimension == 0:
                if vertex is not None:
                    raise ValueError("A source carries one vertex association.")
                vertex = association
            else:
                dimensions.append(dimension)
        if vertex is None:
            raise ValueError("B-Rep association transfer requires a vertex association.")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("A source carries one association per mesh dimension.")
        _vertex_classes(source.mesh, vertex)
        return vertex, tuple(sorted(dimensions))

    def classes(self, source: CellMeshingResult, /) -> tuple[_Classes, ...]:
        """B-Rep classes of every source mesh dimension (adaptation constraints)."""
        vertex, _ = self.source_associations(source)
        return _mesh_entity_classes(source.mesh, vertex, self.projection)

    def _derived(
        self,
        vertex: GeometryAssociation,
        dimensions: tuple[int, ...],
        mesh: CellMesh,
        /,
    ) -> tuple[GeometryAssociation, ...]:
        return (vertex,) + tuple(
            associate_mesh_entities(
                mesh, vertex, self.projection, dimension, policy=self.policy
            )
            for dimension in dimensions
        )

    def propagate(
        self, source: CellMeshingResult, lineage: MeshLineage, target: CellMesh, /
    ) -> tuple[GeometryAssociation, ...]:
        """Target associations through one exact topology lineage."""
        vertex, dimensions = self.source_associations(source)
        propagated = propagate_association(
            vertex, lineage, source.mesh, target, self.projection, policy=self.policy
        )
        return self._derived(propagated, dimensions, target)

    def rederive(
        self, source: CellMeshingResult, target: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, ...]:
        """Target associations of a remeshed result with unknown lineage."""
        vertex, dimensions = self.source_associations(source)
        derived = rederive_association(
            vertex, source, target, self.projection, policy=self.policy
        )
        return self._derived(derived, dimensions, target.mesh)


__all__ = [
    "AssociationPropagationError",
    "AssociationPropagationPolicy",
    "BRepAssociationTransfer",
    "GeometryAssociation",
    "GeometryAssociationKind",
    "GeometryAssociationProvenance",
    "associate_mesh_entities",
    "associate_mesh_vertices",
    "propagate_association",
    "rederive_association",
]
