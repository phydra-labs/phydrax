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

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, final, Literal, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    exact_orient2d,
    MeshcoreStatus,
    point_triangle_locations,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, EntitySet
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..geometry._mesh_certificates import (
    _ray_crossings_3d,
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    PiecewiseLinearDomain,
)
from ..geometry.brep._projection_contracts import (
    AbstractBRepProjection,
    brep_entity_id,
    BRepProjectionResult,
    BRepProjectionStatus,
)
from ..typing import (
    as_host_array,
    checked,
    Dim,
    HostFloat64,
    HostInt32,
    HostInt64,
    HostInteger,
    parse,
    Scope,
)
from ._plc_mapped_support import (
    MappedPlcSupport,
    plc_mapped_vertices,
    prepare_mapped_plc_support,
)
from ._scope import resolve_mesh_scope


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..geometry._mapped_source_support import MappedSourceFacetAuthority
    from ._certification import MeshCertificationPreparedEvidence
    from ._certification_inputs import MeshCertificationInputs
    from ._lineage import MeshLineage
    from ._organization import MeshLabel, MeshPatch, MeshZone
    from ._result import CellMeshingResult
    from ._surface_association_transfer import SurfaceEntityClasses


class GeometryAssociationKind(StrEnum):
    """Representation owning the associated source entities.

    ``PIECEWISE_LINEAR`` names represented vertices, segments, facets, and
    regions; indexed rows retain their explicit source stratum and identifier.
    ``CURVE`` names parametric one-dimensional chart sources. ``IMPLICIT``
    names the scalar source's zero-set boundary in its source/revision namespace.
    ``MAPPED_REFERENCE`` names exact image entities of a declared reference mesh.
    """

    BREP = "brep"
    SURFACE = "surface"
    IMPLICIT = "implicit"
    PIECEWISE_LINEAR = "piecewise_linear"
    CURVE = "curve"
    MAPPED_REFERENCE = "mapped_reference"


class GeometrySourceEntityRole(StrEnum):
    """Declared source stratum role, independent of ambient or equal dimensions."""

    VERTEX = "vertex"
    EDGE = "edge"
    FACET = "facet"
    REGION = "region"


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


def _rows_or(value: ArrayLike | None, fill: Any, shape: Any, dtype: Any, /) -> np.ndarray:
    return (
        np.full(shape, fill, dtype=dtype) if value is None else np.asarray(value, dtype)
    )


class GeometryAssociation(StrictModule, NonTrainableState):
    """Audited map from exact mesh entities to authoritative geometry entities.

    B-Rep rows carry ``source_dimensions``/``source_indices`` as execution
    definition metadata and explicit ``source_occurrence_paths`` as semantic
    instance identity. ``-1`` marks an unresolved row without a candidate
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
    source_entity_roles: tuple[GeometrySourceEntityRole | None, ...] | None = eqx.field(
        static=True
    )
    residuals: Array
    resolved: Array
    ambiguous: Array
    source_dimensions: Array
    source_indices: Array
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
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
        source_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        source_entity_roles: tuple[GeometrySourceEntityRole | None, ...] | None = None,
        parameters: ArrayLike | None = None,
        orientations: ArrayLike | None = None,
        parent_dimensions: ArrayLike | None = None,
        parent_ids: ArrayLike | None = None,
        parent_association_id: str | None = None,
        provenance: GeometryAssociationProvenance = GeometryAssociationProvenance.PROVIDER,
    ) -> None:
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
        paths = (
            ((),) * count if source_occurrence_paths is None else source_occurrence_paths
        )
        if (
            not isinstance(paths, tuple)
            or len(paths) != count
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(name, str) or not name for name in path)
                for path in paths
            )
        ):
            raise ValueError(
                "Source occurrence paths must be authoritative tuple paths aligned with rows."
            )
        if association_kind not in (
            GeometryAssociationKind.BREP,
            GeometryAssociationKind.SURFACE,
        ) and any(paths):
            raise ValueError(
                "Only B-Rep and surface associations carry source occurrence paths."
            )
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
        roles = _source_roles(
            association_kind,
            count,
            source_dimensions,
            source_indices,
            source_entity_roles,
        )
        dimensions, indices = _source_entities(
            association_kind,
            revision,
            sources,
            source_dimensions,
            source_indices,
            paths,
            roles,
        )
        if (
            source_dimensions is not None
            and association_kind
            in (
                GeometryAssociationKind.BREP,
                GeometryAssociationKind.PIECEWISE_LINEAR,
                GeometryAssociationKind.SURFACE,
            )
            and np.any((dimensions < 0) & resolved_)
        ):
            family = {
                GeometryAssociationKind.PIECEWISE_LINEAR: "PLC",
                GeometryAssociationKind.BREP: "B-Rep",
                GeometryAssociationKind.SURFACE: "Surface",
            }[association_kind]
            raise ValueError(f"Resolved {family} association rows must name an entity.")
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
        self.source_entity_roles = roles
        self.residuals = jnp.asarray(distances)
        self.resolved = jnp.asarray(resolved_)
        self.ambiguous = jnp.asarray(ambiguous_)
        self.source_dimensions = jnp.asarray(dimensions)
        self.source_indices = jnp.asarray(indices)
        self.source_occurrence_paths = paths
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
                "source_occurrence_paths": paths,
                **(
                    {
                        "source_entity_roles": tuple(
                            None if role is None else role.value for role in roles
                        ),
                    }
                    if roles is not None
                    else {}
                ),
                **(
                    {
                        "source_dimensions": array_tree_fingerprint(dimensions),
                        "source_indices": array_tree_fingerprint(indices),
                    }
                    if roles is not None
                    or (
                        association_kind is GeometryAssociationKind.SURFACE
                        and source_dimensions is not None
                    )
                    else {}
                ),
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

    @checked
    def validate_target(self, entity_set: EntitySet, /) -> None:
        """Require an exact target binding, not merely resolved row statuses."""
        if self.target_entity_set_id != entity_set.entity_set_id:
            raise ValueError("Geometry association targets a different entity set.")
        if not np.all(
            np.isin(np.asarray(self.target_global_ids), np.asarray(entity_set.entity_ids))
        ):
            raise ValueError("Geometry association contains undeclared target IDs.")

    def target_rows(self, target_global_ids: ArrayLike, /) -> np.ndarray:
        """Host rows of exact target global IDs; every requested ID must be a row."""
        requested = np.asarray(target_global_ids)
        if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
            raise TypeError("target_global_ids must be one integer vector.")
        targets = np.asarray(self.target_global_ids, dtype=np.int64)
        order = np.argsort(targets, kind="stable")
        position = np.minimum(np.searchsorted(targets[order], requested), order.size - 1)
        rows = order[position]
        if not np.array_equal(targets[rows], requested):
            raise ValueError("Geometry association does not classify every requested ID.")
        return rows


def _retain_association_scopes(
    originals: tuple[GeometryAssociation, ...],
    proved: tuple[GeometryAssociation, ...],
    /,
) -> tuple[GeometryAssociation, ...]:
    """Retain fixed-topology scientific scopes after complete fresh support proof."""
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
                "Fixed-topology motion requires one freshly proved owning association per accepted scope."
            )
        fresh = matches[0]
        if (
            np.array_equal(fresh.target_global_ids, original.target_global_ids)
            and fresh.source_occurrence_paths == original.source_occurrence_paths
            and fresh.parent_association_id == original.association_id
        ):
            result.append(fresh)
            continue
        rows = fresh.target_rows(original.target_global_ids)
        result.append(
            GeometryAssociation(
                fresh.association_kind,
                fresh.source_id,
                fresh.source_revision,
                fresh.target_entity_set_id,
                original.target_global_ids,
                tuple(fresh.source_entity_ids[row] for row in rows),
                fresh.residuals[rows],
                resolved=fresh.resolved[rows],
                ambiguous=fresh.ambiguous[rows],
                exact=fresh.exact,
                source_dimensions=fresh.source_dimensions[rows],
                source_indices=fresh.source_indices[rows],
                source_occurrence_paths=original.source_occurrence_paths,
                source_entity_roles=(
                    None
                    if fresh.source_entity_roles is None
                    else tuple(fresh.source_entity_roles[row] for row in rows)
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


class _SourceEntityRowDim(Dim):
    """Rows sharing one authoritative source-entity metadata binding."""


def _source_entity_rows(
    dimensions: ArrayLike | None,
    indices: ArrayLike | None,
    count: int,
    family: str,
    index_dtype: type[np.int32] | type[np.int64],
    /,
) -> tuple[
    HostInteger[_SourceEntityRowDim],
    HostInt32[_SourceEntityRowDim] | HostInt64[_SourceEntityRowDim],
]:
    if dimensions is None or indices is None:
        raise ValueError(f"{family} associations require source dimensions and indices.")
    dimensions_ = np.asarray(dimensions)
    indices_ = np.asarray(indices)
    if not np.issubdtype(dimensions_.dtype, np.integer) or not np.issubdtype(
        indices_.dtype, np.integer
    ):
        raise TypeError(f"{family} source dimensions and indices must be integers.")
    if dimensions_.shape != (count,) or indices_.shape != (count,):
        raise ValueError(f"{family} source dimensions and indices must align with rows.")
    if (
        np.any((dimensions_ < -1) | (dimensions_ > 3))
        or np.any(indices_ < -1)
        or np.any((dimensions_ >= 0) != (indices_ >= 0))
    ):
        raise ValueError(f"{family} source entities must pair a dimension with an index.")
    if np.any(indices_ > np.iinfo(index_dtype).max):
        raise ValueError(
            f"{family} definition indices must fit the native local index contract."
        )
    scope = Scope()
    dimensions_ = parse(
        dimensions_.astype(np.int8),
        HostInteger[_SourceEntityRowDim],
        "source_dimensions",
        scope=scope,
    )
    if index_dtype is np.int64:
        indices_ = as_host_array(
            indices_,
            HostInt64[_SourceEntityRowDim],
            "source_indices",
            scope=scope,
        )
    else:
        indices_ = as_host_array(
            indices_,
            HostInt32[_SourceEntityRowDim],
            "source_indices",
            scope=scope,
        )
    return dimensions_, indices_


def _source_roles(
    kind: GeometryAssociationKind,
    count: int,
    dimensions: ArrayLike | None,
    indices: ArrayLike | None,
    roles: tuple[GeometrySourceEntityRole | None, ...] | None,
    /,
) -> tuple[GeometrySourceEntityRole | None, ...] | None:
    indexed_plc = kind is GeometryAssociationKind.PIECEWISE_LINEAR and (
        dimensions is not None or indices is not None
    )
    if not indexed_plc:
        if roles is not None:
            raise ValueError("Source roles require indexed piecewise-linear strata.")
        return None
    if not isinstance(roles, tuple):
        raise TypeError("Indexed PLC associations require explicit source_entity_roles.")
    if len(roles) != count:
        raise ValueError("Source entity roles must align with association rows.")
    return tuple(
        None
        if role is None
        else parse(role, GeometrySourceEntityRole, "source_entity_role")
        for role in roles
    )


def _plc_source_entity_keys(
    revision: str,
    dimensions: np.ndarray,
    indices: np.ndarray,
    roles: tuple[GeometrySourceEntityRole | None, ...],
    /,
) -> tuple[str, ...]:
    from ._volume_generation import _entity

    keys: list[str] = []
    for dimension, index, role in zip(
        dimensions.tolist(),
        indices.tolist(),
        roles,
        strict=True,
    ):
        match role:
            case None:
                if dimension != -1 or index != -1:
                    raise ValueError(
                        "Unclassified PLC roles require absent source dimension and index."
                    )
                keys.append(_unclassified_id(revision))
                continue
            case GeometrySourceEntityRole.VERTEX:
                valid = dimension == 0
            case GeometrySourceEntityRole.EDGE:
                valid = dimension == 1
            case GeometrySourceEntityRole.FACET:
                valid = dimension == 2
            case GeometrySourceEntityRole.REGION:
                valid = 1 <= dimension <= 3
            case _:
                raise TypeError("Source roles must be GeometrySourceEntityRole values.")
        if not valid or index < 0:
            raise ValueError(
                "PLC source role disagrees with its declared geometric dimension."
            )
        keys.append(_entity(revision, role.value, index))
    return tuple(keys)


def _source_entities(
    kind: GeometryAssociationKind,
    revision: str,
    sources: tuple[str, ...],
    dimensions: ArrayLike | None,
    indices: ArrayLike | None,
    occurrence_paths: tuple[tuple[str, ...], ...],
    roles: tuple[GeometrySourceEntityRole | None, ...] | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(sources)
    if (
        kind is not GeometryAssociationKind.BREP
        and dimensions is None
        and indices is None
    ):
        return np.full((count,), -1, np.int8), np.full((count,), -1, np.int32)
    match kind:
        case GeometryAssociationKind.BREP:
            dimensions_, indices_ = _source_entity_rows(
                dimensions,
                indices,
                count,
                "B-Rep",
                np.int32,
            )
            expected = tuple(
                brep_entity_id(revision, dimension, index, occurrence_path=path)
                if dimension >= 0
                else _unclassified_id(revision)
                for dimension, index, path in zip(
                    dimensions_.tolist(),
                    indices_.tolist(),
                    occurrence_paths,
                    strict=True,
                )
            )
            if expected != sources:
                raise ValueError(
                    "B-Rep source IDs must name the canonical revision, definition entity, and occurrence path."
                )
        case GeometryAssociationKind.PIECEWISE_LINEAR:
            if roles is None:
                raise ValueError(
                    "Indexed PLC associations require explicit source roles."
                )
            dimensions_, indices_ = _source_entity_rows(
                dimensions,
                indices,
                count,
                "PLC",
                np.int64,
            )
            expected = _plc_source_entity_keys(revision, dimensions_, indices_, roles)
            if expected != sources:
                raise ValueError(
                    "PLC source IDs must name the canonical revision and source stratum."
                )
        case GeometryAssociationKind.IMPLICIT:
            dimensions_, indices_ = _source_entity_rows(
                dimensions,
                indices,
                count,
                "Implicit",
                np.int32,
            )
            if np.any((dimensions_ < 0) | (dimensions_ > 2)) or np.any(indices_ != 0):
                raise ValueError(
                    "Implicit associations must name the scalar source's zero-set boundary."
                )
            if sources != ("implicit-zero-set",) * count:
                raise ValueError(
                    "Implicit source IDs must name the authoritative zero-set entity."
                )
        case GeometryAssociationKind.SURFACE:
            dimensions_, indices_ = _source_entity_rows(
                dimensions,
                indices,
                count,
                "Surface",
                np.int64,
            )
            if np.any(dimensions_ > 2):
                raise ValueError(
                    "Surface source strata must be corners, curves, or patches."
                )
        case GeometryAssociationKind.CURVE | GeometryAssociationKind.MAPPED_REFERENCE:
            raise ValueError(
                "This source representation does not declare indexed source strata."
            )
        case _:
            raise TypeError("association_kind must be GeometryAssociationKind.")
    return dimensions_, indices_


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
    ) -> None:
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

    def __init__(self, message: str, target_ids: ArrayLike, /) -> None:
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
    # ty: ignore[no-matching-overload]
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
    occurrence_paths: tuple[tuple[str, ...], ...] = ()

    @property
    def paths(self) -> tuple[tuple[str, ...], ...]:
        return self.occurrence_paths or ((),) * self.dimensions.size

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
    paths: list[tuple[str, ...]] = [()] * count
    for row, path in zip(rows.tolist(), association.source_occurrence_paths, strict=True):
        paths[row] = path
    return _Classes(dims, indices, status, tuple(paths))


def _centroids(mesh: CellMesh, members: np.ndarray, count: int, /) -> np.ndarray:
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
    /,
) -> np.ndarray:
    """Distance of each point to its candidate entity (solids: 0 inside, else inf)."""
    residuals = np.full((points.shape[0],), np.inf)
    lower = np.flatnonzero(dimensions <= 2)
    if lower.size:
        result = projection.project(
            points[lower],
            dimensions[lower],
            indices[lower],
            occurrence_paths=tuple(occurrence_paths[row] for row in lower),
        )
        residuals[lower] = np.where(
            np.asarray(result.status) == BRepProjectionStatus.FAILED,
            np.inf,
            np.asarray(result.residuals),
        )
    solid = np.flatnonzero(dimensions == 3)
    if solid.size:
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
    reach = _compose(
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
    mesh: CellMesh,
    vertex_classes: _Classes,
    projection: AbstractBRepProjection,
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
    projection: AbstractBRepProjection,
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
    adjacent lowest class. A lone coface on one embedded (lower-than-ambient)
    class also admits that class itself, so a partial-patch boundary entity on
    no lower entity lies in the class interior under its occurrence-qualified
    code; full-dimensional regions never admit it.
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
    members = _incidence_pairs(mesh, 0, dimension)[:, ::-1]
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
    inherited_paths = projection.code_occurrence_paths(np.maximum(code, 0))
    paths = tuple(
        inherited_paths[row] if inherited[row] else bounded.paths[row]
        for row in range(count)
    )
    return _Classes(dims.astype(np.int64), indices.astype(np.int64), status, paths)


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
    projection: AbstractBRepProjection,
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
        points[lower],
        classes.dimensions[lower],
        classes.indices[lower],
        occurrence_paths=tuple(classes.paths[row] for row in lower),
    )
    if projected.source_occurrence_paths != tuple(classes.paths[row] for row in lower):
        raise ValueError(
            "Projection changed the authoritative occurrence path of a classified entity."
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
            brep_entity_id(
                projection.source_revision, entity, index, occurrence_path=path
            )
            if entity >= 0
            else _unclassified_id(projection.source_revision)
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
    mesh: CellMesh,
    dimension: int,
    /,
) -> None:
    if not isinstance(association, GeometryAssociation):
        raise TypeError("association must be GeometryAssociation.")
    if not isinstance(projection, AbstractBRepProjection):
        raise TypeError("projection must be AbstractBRepProjection.")
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if association.association_kind is not GeometryAssociationKind.BREP:
        raise ValueError("B-Rep classification requires a B-Rep association.")
    if (
        association.source_revision != projection.source_revision
        or association.source_id != projection.source_id
    ):
        raise ValueError(
            "The association and projection bind different authoritative sources."
        )
    if association.target_entity_set_id != mesh.entity_set(dimension).entity_set_id:
        raise ValueError("The association does not target this mesh's entities.")
    if mesh.ambient_dimension != projection.ambient_dimension:
        raise ValueError("Mesh and projection ambient dimensions differ.")
    resolved = np.flatnonzero(np.asarray(association.resolved))
    projection.entity_codes(
        np.asarray(association.source_dimensions)[resolved],
        np.asarray(association.source_indices)[resolved],
        occurrence_paths=tuple(
            association.source_occurrence_paths[row] for row in resolved
        ),
    )


def _policy(policy: AssociationPropagationPolicy, /) -> AssociationPropagationPolicy:
    if not isinstance(policy, AssociationPropagationPolicy):
        raise TypeError("policy must be AssociationPropagationPolicy.")
    return policy


# -- public classification ----------------------------------------------------------


def _mesh_entity_classes(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
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
    projection: AbstractBRepProjection,
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
    projection: AbstractBRepProjection,
    tolerance: float,
    groups: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]
    ]
    | None,
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
            ),
        )
    rows = np.flatnonzero(~boundary)
    if rows.size:
        assign(rows, projection.classify(points[rows], tolerance=tolerance))
        if top == 3:
            inside = rows[dims[rows] < 0]
            if inside.size:
                assign(inside, projection.locate_solids(points[inside]))
    return _Classes(dims, indices, status, tuple(paths))


def associate_mesh_vertices(
    mesh: CellMesh,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Classify every mesh vertex geometrically and project it onto its class."""
    policy_ = _policy(policy)
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(projection, AbstractBRepProjection):
        raise TypeError("projection must be AbstractBRepProjection.")
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
    projection: AbstractBRepProjection,
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
    paths: list[tuple[str, ...]] = [()] * count
    parent_dims = np.full((count,), -1, dtype=np.int8)
    parent_ids = np.full((count,), -1, dtype=np.int64)
    kept = np.flatnonzero(_entity_rows(source_mesh, 0, target_ids) >= 0)
    kept_source = _entity_rows(source_mesh, 0, target_ids[kept])
    dims[kept] = source_classes.dimensions[kept_source]
    indices[kept] = source_classes.indices[kept_source]
    status[kept] = source_classes.status[kept_source]
    for target_row, source_row in zip(kept, kept_source, strict=True):
        paths[target_row] = source_classes.paths[source_row]
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
                for target_row, source_row in zip(target_rows, rows, strict=True):
                    paths[target_row] = level_classes.paths[source_row]
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
            for target_row, path in zip(
                orphans, geometric.source_occurrence_paths, strict=True
            ):
                paths[target_row] = path
    return _brep_association(
        projection,
        target_mesh,
        0,
        _Classes(dims, indices, status, tuple(paths)),
        np.asarray(target_mesh.coordinates, dtype=np.float64),
        policy_,
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
        occurrence_paths=tuple(classes.paths[row] for row in removed[known]),
        member_occurrence_paths=tuple(classes.paths[row] for row in kept[known]),
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
        tuple(facet_classes.paths[row] for row in rows),
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


def rederive_association(
    association: GeometryAssociation,
    source: CellMeshingResult,
    target: CellMeshingResult,
    projection: AbstractBRepProjection,
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


class _PlcAmbientDim(Dim, minimum=2):
    """Declared Cartesian dimension of authoritative PLC vertices."""


class _PlcEdgeDim(Dim, minimum=1):
    """Authoritative source PLC edges."""


class _PlcTriangleDim(Dim):
    """Authoritative source facet triangles."""


class _PlcFacetDim(Dim):
    """Authoritative source facet incidence groups."""


class _PlcRegionDim(Dim, minimum=1):
    """Explicit scientific region identifiers aligned to domain definitions."""


class _PlcVertexDim(Dim, minimum=1):
    """Explicit source vertex identifiers."""


class PlcEntityClasses(NamedTuple):
    """Source strata and explicit namespace codes of one mesh dimension."""

    dimensions: np.ndarray
    indices: np.ndarray
    resolved: np.ndarray
    codes: np.ndarray


def _plc_consensus(
    mesh: CellMesh,
    dimension: int,
    cell_regions: np.ndarray,
    /,
) -> np.ndarray:
    """Region of each stratum interior; interfaces require explicit source rows."""
    incidence = _incidence_pairs(mesh, dimension, 3)
    pairs = np.unique(
        np.stack((incidence[:, 0], cell_regions[incidence[:, 1]]), axis=1), axis=0
    )
    rows, counts = np.unique(pairs[:, 0], return_counts=True)
    result = np.full((mesh.entity_set(dimension).count,), -1, dtype=np.int64)
    single = rows[counts == 1]
    result[single] = pairs[np.searchsorted(pairs[:, 0], single), 1]
    return result


@dataclass
class _PlcBankContext:
    """One explicit namespace using shared, independently certified coverage."""

    mapped: MappedPlcSupport
    facet_indices: np.ndarray
    region_map: np.ndarray
    work: list[int]
    target_mapped: MappedPlcSupport | None = None
    mapped_facets: MappedSourceFacetAuthority | None = None

    def indices(self, association: GeometryAssociation, /) -> np.ndarray:
        result = np.array(association.source_indices, dtype=np.int64, copy=True)
        dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
        positions = {int(label): row for row, label in enumerate(self.facet_indices)}
        for row in np.flatnonzero(dimensions == 2):
            label = int(result[row])
            if label not in positions:
                raise ValueError("Composed PLC facet lacks its original source identity.")
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
            raise ValueError("Composed PLC region lacks its original source identity.")
        return int(self.region_map[matches[0]])

    def bank_regions(
        self, transfer: PlcAssociationTransfer, mapped: MappedPlcSupport | None = None, /
    ) -> np.ndarray:
        proof = self.mapped if mapped is None else mapped
        result = np.full(proof.cell_regions.shape, -1, dtype=np.int64)
        for source_region, material_region in zip(
            transfer.region_indices, self.region_map, strict=True
        ):
            result[proof.cell_regions == material_region] = source_region
        return result


@final
class PlcAssociationTransfer(StrictModule, NonTrainableState):
    """Exact represented-PLC source transfer, never a synthetic B-Rep projection.

    Source vertex, edge and facet tables are authoritative identity banks emitted
    by the PLC owner, not reconstructed from coordinate proximity. Planar source
    vertices may be distinct from a refined support-domain carrier; its Steiner
    rows never become source authorities. Exact translated planar sources retain
    their strata through ``transition_source_associations``.
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
        from .._validation import canonical_identifier, positive_integer

        if not isinstance(
            domain, PiecewiseLinearDomain
        ) or domain.ambient_dimension not in (2, 3):
            raise TypeError(
                "PLC transfer requires an authoritative planar or volume domain."
            )
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if not coordinate_contract.is_orthonormal_cartesian:
            raise ValueError(
                "PLC support predicates require declared orthonormal Cartesian coordinates."
            )
        revision = canonical_identifier(source_revision, "source_revision")
        limit = positive_integer(maximum_support_queries, "maximum_support_queries")
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
            raise TypeError("PLC source identity tables must contain integer indices.")
        scope = Scope()
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
                "Source vertices must be finite Cartesian rows in the domain frame."
            )
        if domain.ambient_dimension == 3 and not np.array_equal(
            authority_vertices, domain.vertices
        ):
            raise ValueError(
                "Volume PLC support requires its authoritative vertices in domain-row order."
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
        labels_raw = (
            np.arange(len(domain.region_ids), dtype=np.int64)
            if region_indices is None
            else np.asarray(region_indices)
        )
        if not np.issubdtype(labels_raw.dtype, np.integer):
            raise TypeError("region_indices must contain explicit integer identifiers.")
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
                "region_indices must uniquely bind every declared domain region."
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
            raise TypeError("Source vertex and edge authorities must be integer vectors.")
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
                "Vertex and edge authorities must uniquely name every declared source row."
            )
        if domain.ambient_dimension == 3 and (
            triangles.shape[0] == 0 or regions.shape[0] == 0
        ):
            raise ValueError(
                "A volume PLC requires authoritative facet triangles and incidences."
            )
        if domain.ambient_dimension == 2 and triangles.shape[0] != 0:
            raise ValueError(
                "A planar PLC uses source segments rather than facet triangles."
            )
        if (
            np.any(edges < 0)
            or np.any(edges >= authority_vertices.shape[0])
            or np.any(triangles < 0)
            or np.any(triangles >= authority_vertices.shape[0])
        ):
            raise ValueError("PLC identity tables reference undeclared vertices.")
        if (
            np.any(facets < 0)
            or np.any(facets >= regions.shape[0])
            or np.any(regions < -1)
            or np.any(regions >= len(domain.region_ids))
        ):
            raise ValueError("PLC facet identities or region incidences are invalid.")
        if (
            np.any(edges[:, 0] == edges[:, 1])
            or np.unique(np.sort(edges, axis=1), axis=0).shape[0] != edges.shape[0]
        ):
            raise ValueError(
                "PLC source edge identities must be distinct noncollapsed edges."
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

    def transition_source_associations(
        self,
        source: CellMeshingResult,
        predecessor: PlcAssociationTransfer,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        translation: ArrayLike,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
    ) -> tuple[
        tuple[MeshPatch, ...],
        tuple[MeshZone, ...],
        tuple[MeshLabel, ...],
        tuple[GeometryAssociation, ...],
    ]:
        """Revalidate unchanged source strata under a declared translation.

        Source authority vertices and edges are independent of the represented
        support-domain carrier. No Steiner row acquires a source vertex identity.
        Incidence, scientific entity correspondence, and the actual translated
        coordinate map must all survive before new source associations are issued.
        Optional planar premises must bind this exact current target map and
        source domain; they replace no source-stratum or correspondence check.
        """
        from ..discretization._cell_geometry import CellGeometrySpec
        from ._lineage import identity_lineage, inherit_mesh_organization
        from ._plc_planar_support import _preserved_parameters, _prove_levels, _roles
        from ._result import CellMeshingResult
        from ._volume_generation import _entity

        if not isinstance(source, CellMeshingResult) or not isinstance(
            predecessor, PlcAssociationTransfer
        ):
            raise TypeError(
                "Source transition requires an accepted result and its predecessor transfer."
            )
        if not isinstance(target, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Source transition requires the actual target mesh and coordinate map."
            )
        if source.mesh.storage is not None or target.storage is not None:
            raise ValueError(
                "Translated source associations require host-only accepted mesh carriers."
            )
        if (
            self.domain.ambient_dimension == 3
            and predecessor.domain.ambient_dimension == 3
        ):
            if embedding is not None or coverage is not None:
                raise ValueError(
                    "Prepared translated-support premises currently require a planar PLC."
                )
            from ._plc_mapped_support import transition_mapped_source_associations

            return transition_mapped_source_associations(
                self,
                source,
                predecessor,
                target,
                geometry=geometry,
                translation=translation,
            )
        if (
            self.domain.ambient_dimension != 2
            or predecessor.domain.ambient_dimension != 2
        ):
            raise ValueError(
                "Translated source association transitions require planar PLC authority."
            )
        if self.domain.source_id != predecessor.domain.source_id or (
            self.coordinate_contract.spatial_id
            != predecessor.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Source transition cannot change source identity, frame, or units."
            )
        if self.source_revision == predecessor.source_revision:
            raise ValueError(
                "Source translation requires an explicit successor revision."
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
                "Source translation cannot change declared stratum incidence or authority indices."
            )
        if not np.array_equal(self.domain.facets, predecessor.domain.facets) or (
            not np.array_equal(
                self.domain.facet_regions, predecessor.domain.facet_regions
            )
        ):
            raise ValueError(
                "Source translation cannot change represented boundary or region incidence."
            )
        for transfer in (predecessor, self):
            expected_regions = tuple(
                _entity(transfer.source_revision, "region", int(index))
                for index in transfer.region_indices
            )
            if transfer.domain.region_ids != expected_regions:
                raise ValueError(
                    "Translated planar domains must bind their explicit source region indices."
                )
        shift = np.asarray(translation, dtype=np.float64)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("Source translation must be a finite planar displacement.")
        if (
            target.topology_id != source.mesh.topology_id
            or geometry.geometry_layout_id != source.geometry.geometry_layout_id
        ):
            raise ValueError(
                "Source translation must retain mesh topology and coordinate-map layout."
            )
        for dimension in range(3):
            if not np.array_equal(
                target.entity_set(dimension).entity_ids,
                source.mesh.entity_set(dimension).entity_ids,
            ):
                raise ValueError(
                    "Source translation must retain every scientific mesh entity correspondence."
                )
        for actual, original in (
            (self.source_vertices, predecessor.source_vertices),
            (self.domain.vertices, predecessor.domain.vertices),
            (np.asarray(target.coordinates), np.asarray(source.mesh.coordinates)),
            (np.asarray(geometry.coordinates), np.asarray(source.geometry.coordinates)),
        ):
            if actual.shape != original.shape or not np.array_equal(
                actual, original + shift
            ):
                raise ValueError(
                    "Source transition differs from its declared rigid translation."
                )
        old_vertex, dimensions = predecessor.source_associations(source)
        if old_vertex is None:
            raise ValueError(
                "Planar source transition requires an accepted vertex association."
            )
        levels = predecessor.classes(source)
        orientations = _prove_levels(
            self,
            target,
            geometry,
            levels,
            [self.maximum_support_queries],
            embedding=embedding,
            coverage=coverage,
        )
        output: list[GeometryAssociation] = []
        for dimension in (0, *dimensions):
            previous = (
                old_vertex
                if dimension == 0
                else next(
                    value
                    for value in source.associations
                    if _target_dimension(source.mesh, value) == dimension
                )
            )
            # Fixed topology transports the caller's exact association scope,
            # not every entity whose source stratum was proved internally.
            identifiers = np.asarray(previous.target_global_ids, dtype=np.int64)
            rows = _entity_rows(target, dimension, identifiers)
            level = levels[dimension]
            roles = _roles(level.dimensions[rows])
            names = tuple(
                _entity(self.source_revision, role.value, int(index))
                for role, index in zip(roles, level.indices[rows], strict=True)
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
                    source_occurrence_paths=previous.source_occurrence_paths,
                    source_entity_roles=roles,
                    parameters=_preserved_parameters(
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
        patches, zones, labels = inherit_mesh_organization(
            source, target, identity_lineage(source.mesh, target)
        )
        return patches, zones, labels, tuple(output)

    def source_associations(
        self,
        source: CellMeshingResult,
        /,
        *,
        _bank: _PlcBankContext | None = None,
        _prepared_source: list[MappedPlcSupport] | None = None,
    ) -> tuple[GeometryAssociation | None, tuple[int, ...]]:
        from ._result import CellMeshingResult
        from ._volume_generation import _entity

        if self.domain.ambient_dimension == 2:
            from ._plc_planar_support import planar_source_associations

            return planar_source_associations(self, source)

        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        if source.coordinate_contract.spatial_id != self.coordinate_contract.spatial_id:
            raise ValueError("PLC transfer binds another coordinate frame or units.")
        certificate = source.certification
        coverage_domain = self.domain if _bank is None else _bank.mapped.domain
        if (
            certificate is None
            or not certificate.passed
            or certificate.coverage is None
            or certificate.coverage.domain_id != coverage_domain.domain_id
        ):
            raise ValueError(
                "PLC transfer requires current independent coverage of its domain."
            )
        if certificate.mesh_id != source.mesh.mesh_id:
            raise ValueError("PLC transfer requires the source's current mesh binding.")
        certificate.coverage.binding.require(source.mesh, source.geometry)
        if _bank is not None:
            # This exact bank preparation already checked every rounded carrier
            # corner under its persistent support ledger.
            _bank.mapped.prepared.require(
                source.mesh, source.geometry, certificate.request
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
            if (
                association.association_kind
                is not GeometryAssociationKind.PIECEWISE_LINEAR
                or association.source_id != self.domain.source_id
                or association.source_revision != self.source_revision
            ):
                raise ValueError(
                    "PLC transfer carries its represented source identity only."
                )
            dimension = _target_dimension(source.mesh, association)
            if dimension in dimensions:
                raise ValueError(
                    "A PLC source carries one association per mesh dimension."
                )
            association.validate_target(source.mesh.entity_set(dimension))
            if (
                not np.all(np.asarray(association.resolved))
                or np.any(np.asarray(association.ambiguous))
                or not association.exact
            ):
                raise ValueError(
                    "PLC transfer requires resolved exact source associations."
                )
            dims = np.asarray(association.source_dimensions, dtype=np.int64)
            indices = np.asarray(association.source_indices, dtype=np.int64)
            if np.any(dims < 0) or np.any(dims > 3) or np.any(indices < 0):
                raise ValueError("PLC associations name undeclared source strata.")
            facet_authority = (
                np.arange(counts[2], dtype=np.int64)
                if _bank is None
                else _bank.facet_indices
            )
            if (
                np.any(~np.isin(indices[dims == 0], self.vertex_indices))
                or np.any(~np.isin(indices[dims == 1], self.edge_indices))
                or np.any(~np.isin(indices[dims == 2], facet_authority))
                or np.any(~np.isin(indices[dims == 3], self.region_indices))
            ):
                raise ValueError("PLC associations name undeclared source authorities.")
            roles = tuple(source_roles[kind] for kind in dims.tolist())
            if association.source_entity_roles != roles:
                raise ValueError(
                    "PLC association roles disagree with the declared volume strata."
                )
            expected = tuple(
                _entity(self.source_revision, role.value, index)
                for role, index in zip(roles, indices.tolist(), strict=True)
            )
            if association.source_entity_ids != expected:
                raise ValueError(
                    "PLC association IDs disagree with their authoritative source indices."
                )
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
                        np.asarray(source.mesh.entity_set(3).entity_ids, dtype=np.int64)
                    )
                if np.any(dims != 3):
                    raise ValueError("PLC cells must bind explicit source regions.")
        if _bank is None and (vertex is None or 3 not in dimensions):
            raise ValueError(
                "PLC transfer requires complete vertex and cell association tables."
            )
        if _bank is None:
            prepared = [] if _prepared_source is None else _prepared_source
            if len(prepared) > 1:
                raise ValueError("PLC source support requires one exact prepared source.")
            if not prepared:
                cell = next(
                    value
                    for value in source.associations
                    if _target_dimension(source.mesh, value) == 3
                )
                ids = np.asarray(source.mesh.entity_set(3).entity_ids, dtype=np.int64)
                cell_regions = np.asarray(cell.source_indices, dtype=np.int64)[
                    cell.target_rows(ids)
                ]
                prepared.append(
                    prepare_mapped_plc_support(
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
                    )
                )
            proof = prepared[0]
            if (
                proof.domain.domain_id != self.domain.domain_id
                or proof.limits.maximum_candidate_pairs != self.maximum_support_queries
            ):
                raise ValueError(
                    "PLC source support belongs to another domain or support policy."
                )
            proof.prepared.require(source.mesh, source.geometry, certificate.request)
        return vertex, tuple(
            sorted(dimension for dimension in dimensions if dimension != 0)
        )

    def classes(
        self,
        source: CellMeshingResult,
        /,
        *,
        _bank: _PlcBankContext | None = None,
        _prepared_source: list[MappedPlcSupport] | None = None,
    ) -> tuple[PlcEntityClasses, ...]:
        if self.domain.ambient_dimension == 2:
            from ._plc_planar_support import planar_classes

            return planar_classes(self, source)
        prepared = [] if _prepared_source is None else _prepared_source
        self.source_associations(source, _bank=_bank, _prepared_source=prepared)
        associations = tuple(
            value
            for value in source.associations
            if _bank is None
            or (value.source_id, value.source_revision)
            == (self.domain.source_id, self.source_revision)
        )
        active_dimensions = {
            _target_dimension(source.mesh, value) for value in associations
        }
        if _bank is None:
            proof = prepared[0]
            cell_regions = proof.cell_regions
        else:
            cell_regions = _bank.bank_regions(self)
            proof = _bank.mapped
        work = [self.maximum_support_queries] if _bank is None else _bank.work
        for association in associations:
            dimension = _target_dimension(source.mesh, association)
            rows = _entity_rows(
                source.mesh,
                dimension,
                np.asarray(association.target_global_ids, dtype=np.int64),
            )
            bank_indices = (
                np.asarray(association.source_indices)
                if _bank is None
                else _bank.indices(association)
            )
            for association_row, row in enumerate(rows.tolist()):
                kind = int(np.asarray(association.source_dimensions)[association_row])
                index = int(bank_indices[association_row])
                local_index = index
                if kind in (0, 1):
                    matches = np.flatnonzero(
                        (self.vertex_indices if kind == 0 else self.edge_indices) == index
                    )
                    if matches.size != 1:
                        raise ValueError(
                            "Source association lacks its explicit primitive authority."
                        )
                    local_index = int(matches[0])
                if _bank is not None:
                    local_index = _bank.support_index(self, kind, local_index)
                if kind == 2 and _bank is not None and _bank.mapped_facets is not None:
                    supported, orientation = _bank.mapped_facets.support(
                        proof, dimension, row, local_index, work
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
                expected = int(np.asarray(association.orientations)[association_row])
                if not supported or (expected != 0 and expected != orientation):
                    raise AssociationPropagationError(
                        f"Source PLC bank {self.domain.source_id!r} fails dimension-{dimension} target "
                        f"{int(np.asarray(association.target_global_ids)[association_row])}: "
                        f"source stratum ({kind}, {int(np.asarray(association.source_indices)[association_row])}), "
                        f"support={supported}, orientation={orientation}, requested_orientation={expected}.",
                        np.asarray(association.target_global_ids)[
                            association_row : association_row + 1
                        ],
                    )
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
            regions = _plc_consensus(source.mesh, dimension, cell_regions)
            dims = np.where(regions >= 0, 3, -1).astype(np.int64)
            indices = regions.copy()
            for association in associations:
                if _target_dimension(source.mesh, association) != dimension:
                    continue
                rows = _entity_rows(
                    source.mesh,
                    dimension,
                    np.asarray(association.target_global_ids, dtype=np.int64),
                )
                dims[rows] = np.asarray(association.source_dimensions, dtype=np.int64)
                indices[rows] = (
                    np.asarray(association.source_indices, dtype=np.int64)
                    if _bank is None
                    else _bank.indices(association)
                )
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
                    "PLC interface strata lack authoritative source association.",
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
        return tuple(result)

    def _region_triangles(self, region: int, /) -> np.ndarray:
        matches = np.flatnonzero(self.region_indices == region)
        if matches.size != 1:
            raise ValueError("PLC support query names an undeclared region identifier.")
        region = int(matches[0])
        pairs = self.domain.facet_regions
        selected = (pairs[:, 0] == region) | (pairs[:, 1] == region)
        rows = np.array(self.domain.facets[selected], dtype=np.int64, copy=True)
        reversed_ = pairs[selected, 1] == region
        rows[reversed_] = rows[reversed_][:, (0, 2, 1)]
        return self.domain.vertices[rows]

    def _point_support(
        self, dimension: int, index: int, point: np.ndarray, work: list[int], /
    ) -> bool:
        from .._meshcore import charge_native_geometry_queries

        if self.domain.ambient_dimension == 2:
            from ._plc_planar_support import planar_point_support

            return planar_point_support(self, dimension, index, point, work)
        if dimension in (0, 1):
            authority = self.vertex_indices if dimension == 0 else self.edge_indices
            matches = np.flatnonzero(authority == index)
            if matches.size != 1:
                raise ValueError(
                    "PLC support query names an undeclared source authority."
                )
            index = int(matches[0])
        if dimension == 0:
            self._spend(work, 1)
            charge_native_geometry_queries(1)
            return np.array_equal(point, self.domain.vertices[index])
        if dimension == 1:
            self._spend(work, 1)
            charge_native_geometry_queries(1)
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
            raise ValueError("Unknown PLC source stratum dimension.")
        self._spend(work, triangles.shape[0])
        charge_native_geometry_queries(triangles.shape[0])
        _, features, status = point_triangle_locations(
            np.broadcast_to(point, (triangles.shape[0], 3)), triangles
        )
        if np.any(status != MeshcoreStatus.OK):
            raise ValueError("PLC exact source-support predicate was unresolved.")
        on_boundary = bool(np.any(features >= 0))
        if dimension == 2:
            return on_boundary
        if on_boundary:
            return False
        self._spend(work, triangles.shape[0])
        charge_native_geometry_queries(triangles.shape[0])
        winding, decided = _ray_crossings_3d(point, triangles)
        return decided and winding == 1

    def _spend(self, work: list[int], amount: int, /) -> None:
        work[0] -= amount
        if work[0] < 0:
            raise ValueError("PLC exact source-support query budget exhausted.")

    def protected_edges(
        self,
        source: CellMeshingResult,
        /,
        *,
        midpoint_required: bool = True,
        _bank: _PlcBankContext | None = None,
        _classes: tuple[PlcEntityClasses, ...] | None = None,
    ) -> np.ndarray:
        if self.domain.ambient_dimension == 2:
            from ._plc_planar_support import planar_protected_edges

            return planar_protected_edges(
                self, source, midpoint_required=midpoint_required
            )
        levels = self.classes(source, _bank=_bank) if _classes is None else _classes
        vertices = plc_mapped_vertices(source.mesh, 1)
        coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
        endpoints = coordinates[vertices]
        midpoints = np.sum(endpoints * np.float64(0.5), axis=1)
        protected = np.all(midpoints == endpoints[:, 0], axis=1) | np.all(
            midpoints == endpoints[:, 1], axis=1
        )
        if midpoint_required:
            work = [self.maximum_support_queries] if _bank is None else _bank.work
            for row in np.flatnonzero(
                (levels[1].dimensions >= 0) & (levels[1].dimensions < 3)
            ).tolist():
                protected[row] |= not self._point_support(
                    int(levels[1].dimensions[row]),
                    int(levels[1].indices[row]),
                    midpoints[row],
                    work,
                )
        return protected

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
        _certification_request: MeshCertificationInputs | None = None,
        _prepared_target: list[MeshCertificationPreparedEvidence] | None = None,
        _bank: _PlcBankContext | None = None,
        _mapped_target: list[MappedPlcSupport] | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        """Prove lineage support, consuming current collective premises for local planar carriers."""
        from ._lineage import EntityLineageKind, MeshLineage
        from ._volume_generation import _entity

        if (_certification_request is None) != (_prepared_target is None):
            raise ValueError(
                "Coherent target evidence needs its exact owning request and collector."
            )
        if self.domain.ambient_dimension != 3 and _prepared_target is not None:
            raise ValueError(
                "Volume target evidence collection requires a three-dimensional PLC."
            )
        if self.domain.ambient_dimension == 2:
            from ._plc_planar_support import planar_propagate

            return planar_propagate(
                self,
                source,
                lineage,
                target,
                geometry=geometry,
                embedding=embedding,
                coverage=coverage,
            )

        if embedding is not None or coverage is not None:
            raise ValueError(
                "Externally supplied collective PLC premises require a planar carrier."
            )
        prepared_source: list[MappedPlcSupport] = []
        vertex, dimensions = self.source_associations(
            source, _bank=_bank, _prepared_source=prepared_source
        )
        if _bank is None and vertex is None:
            raise ValueError("Volume PLC transfer requires a vertex association.")
        if (
            not isinstance(lineage, MeshLineage)
            or lineage.source_topology_id != source.mesh.topology_id
            or lineage.target_topology_id != target.topology_id
        ):
            raise ValueError(
                "PLC transfer requires the actual source-to-target topology lineage."
            )
        if _bank is not None and _bank.target_mapped is not None:
            # Reuse only this exact prepared target/request; no unmetered
            # coordinate-map reconstruction is allowed for another bank.
            _bank.target_mapped.prepared.require(
                target,
                geometry,
                _bank.target_mapped.prepared.request
                if _certification_request is None
                else _certification_request,
            )
        old = self.classes(source, _bank=_bank, _prepared_source=prepared_source)
        work = [self.maximum_support_queries] if _bank is None else _bank.work
        table: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        obligations: list[np.ndarray] = []
        for dimension in range(4):
            ids = np.asarray(target.entity_set(dimension).entity_ids, dtype=np.int64)
            dims = np.full(ids.shape, -1, dtype=np.int64)
            indices = np.full(ids.shape, -1, dtype=np.int64)
            parent_dims = np.full(ids.shape, -1, dtype=np.int8)
            parent_ids = np.full(ids.shape, -1, dtype=np.int64)
            kept = _entity_rows(source.mesh, dimension, ids)
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
            if np.any(kinds == EntityLineageKind.UNKNOWN):
                raise AssociationPropagationError(
                    "Unknown lineage cannot identify represented PLC strata.", ids
                )
            source_rows = _entity_rows(
                source.mesh,
                dimension,
                np.asarray(record.source_global_ids, dtype=np.int64),
            )
            target_rows = _entity_rows(
                target, dimension, np.asarray(record.target_global_ids, dtype=np.int64)
            )
            if np.any(source_rows < 0) or np.any(target_rows < 0):
                raise ValueError("PLC lineage references undeclared entities.")
            for row in np.unique(target_rows).tolist():
                parents = source_rows[target_rows == row]
                required[row] |= bool(np.any(old[dimension].resolved[parents]))
                candidates = np.unique(
                    np.stack(
                        (
                            old[dimension].dimensions[parents],
                            old[dimension].indices[parents],
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
                if candidates.shape[0] == 1:
                    dims[row], indices[row] = candidates[0]
                    parent_dims[row] = dimension
                    parent_ids[row] = int(
                        np.min(np.asarray(record.source_global_ids)[target_rows == row])
                    )
            table.append((dims, indices, parent_dims, parent_ids))
            obligations.append(required)
        # Complete cell labels precede lower-stratum closure classification.
        cell_dims, cell_regions, _, _ = table[3]
        cell_ids = np.asarray(target.entity_set(3).entity_ids, dtype=np.int64)
        if _bank is None or _bank.target_mapped is None:
            if np.any(cell_dims != 3) or np.any(cell_regions < 0):
                raise AssociationPropagationError(
                    "Target PLC cells lack unambiguous region lineage.",
                    cell_ids[cell_dims != 3],
                )
            if (
                _bank is not None
                and self.domain.domain_id != _bank.mapped.domain.domain_id
            ):
                raise ValueError(
                    "The complete material owner must establish composed target coverage first."
                )
            mapped = prepare_mapped_plc_support(
                target,
                geometry,
                self.domain,
                cell_regions,
                self.region_indices,
                maximum_support_queries=self.maximum_support_queries,
                certification_request=_certification_request,
            )
        else:
            mapped = _bank.target_mapped
            expected_regions = _bank.bank_regions(self, mapped)
            if not np.array_equal(cell_regions, expected_regions) or np.any(
                cell_dims[expected_regions >= 0] != 3
            ):
                raise AssociationPropagationError(
                    "Composed bank cell lineage disagrees with independent material coverage.",
                    cell_ids[cell_regions != expected_regions],
                )
        for row in np.flatnonzero(cell_regions >= 0):
            region = int(cell_regions[row])
            support_region = (
                region if _bank is None else _bank.support_index(self, 3, region)
            )
            supported, _ = mapped.entity_support(
                3,
                int(row),
                3,
                support_region,
                edge_vertices=self.edge_vertices,
                triangle_vertices=self.triangle_vertices,
                triangle_facets=self.triangle_facets,
                work=work,
                source_vertices=self.source_vertices,
            )
            if not supported:
                raise AssociationPropagationError(
                    "A target coordinate image is not contained in its inherited PLC region.",
                    cell_ids[row : row + 1],
                )
        for dimension in (2, 1):
            dims, indices, parent_dims, parent_ids = table[dimension]
            if dimension == 1:
                links = _incidence_pairs(target, 1, 2)
                face_dims, face_indices, _, _ = table[2]
                for row in np.flatnonzero(dims < 0).tolist():
                    faces = links[links[:, 0] == row, 1]
                    constrained = faces[face_dims[faces] == 2]
                    candidates = np.unique(face_indices[constrained])
                    if candidates.size == 1:
                        dims[row], indices[row] = 2, candidates[0]
            missing = dims < 0
            regions = _plc_consensus(target, dimension, cell_regions)
            dims[missing & (regions >= 0)] = 3
            indices[missing & (regions >= 0)] = regions[missing & (regions >= 0)]
        target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
        dims, indices, parent_dims, parent_ids = table[0]
        missing = np.flatnonzero(dims < 0)
        if missing.size:
            created = target_ids[missing]
            order = np.argsort(created, kind="stable")
            entity_dims, entity_rows = _child_sources(
                lineage, source.mesh, target, created[order]
            )
            for position, source_dimension, source_row in zip(
                missing[order].tolist(),
                entity_dims.tolist(),
                entity_rows.tolist(),
                strict=True,
            ):
                if source_dimension < 0 or source_row < 0:
                    continue
                dims[position] = old[source_dimension].dimensions[source_row]
                indices[position] = old[source_dimension].indices[source_row]
                obligations[0][position] |= bool(
                    old[source_dimension].resolved[source_row]
                )
                parent_dims[position] = source_dimension
                parent_ids[position] = int(
                    np.asarray(source.mesh.entity_set(source_dimension).entity_ids)[
                        source_row
                    ]
                )
        missing = np.flatnonzero(dims < 0)
        if missing.size:
            # A mixed cell-center vertex has cell ancestry, not invented
            # dimension-zero parent rows. Lower-stratum witnesses take priority.
            vertex_cells = _incidence_pairs(target, 0, 3)
            vertex_faces = _incidence_pairs(target, 0, 2)
            vertex_edges = _incidence_pairs(target, 0, 1)
            cell_record = lineage.entity_lineage(3)
            source_cells = _entity_rows(
                source.mesh, 3, np.asarray(cell_record.source_global_ids, dtype=np.int64)
            )
            target_cells = _entity_rows(
                target, 3, np.asarray(cell_record.target_global_ids, dtype=np.int64)
            )
            source_cell_ids = np.asarray(
                source.mesh.entity_set(3).entity_ids, dtype=np.int64
            )
            for position in missing.tolist():
                faces = vertex_faces[vertex_faces[:, 0] == position, 1]
                edges = vertex_edges[vertex_edges[:, 0] == position, 1]
                if (
                    mapped.boundary_entities[0][position]
                    or np.any((table[2][0][faces] >= 0) & (table[2][0][faces] < 3))
                    or np.any((table[1][0][edges] >= 0) & (table[1][0][edges] < 3))
                ):
                    continue
                cells = vertex_cells[vertex_cells[:, 0] == position, 1]
                owners = np.unique(source_cells[np.isin(target_cells, cells)])
                if owners.size != 1 or owners[0] < 0:
                    continue
                owner = int(owners[0])
                region = int(old[3].indices[owner])
                if (
                    old[3].dimensions[owner] != 3
                    or region < 0
                    or not np.all(cell_regions[cells] == region)
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
                parent_dims[position], parent_ids[position] = 3, source_cell_ids[owner]
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
                    "PLC target strata lack authoritative source parents.",
                    ids[unresolved & required],
                )
            selected = ~unresolved
            orientations = np.zeros(ids.shape, dtype=np.int8)
            for row in np.flatnonzero(selected):
                kind, index = int(dims[row]), int(indices[row])
                local_index = index
                if kind in (0, 1):
                    matches = np.flatnonzero(
                        (self.vertex_indices if kind == 0 else self.edge_indices) == index
                    )
                    if matches.size != 1:
                        raise AssociationPropagationError(
                            "Target entity names an undeclared source authority.",
                            ids[row : row + 1],
                        )
                    local_index = int(matches[0])
                if _bank is not None:
                    local_index = _bank.support_index(self, kind, local_index)
                if kind == 2 and _bank is not None and _bank.mapped_facets is not None:
                    supported, orientation = _bank.mapped_facets.support(
                        mapped, dimension, int(row), local_index, work
                    )
                    orientations[row] = orientation
                elif dimension == 0:
                    # The exact coordinate-map corner, not its binary64 rounding.
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
                    source_index = (
                        index if _bank is None else _bank.source_index(kind, index)
                    )
                    raise AssociationPropagationError(
                        f"Target PLC bank {self.domain.source_id!r} fails dimension-{dimension} target "
                        f"{int(ids[row])}: inherited source stratum ({kind}, {source_index}), "
                        f"original root support={supported}.",
                        ids[row : row + 1],
                    )
            previous = (
                vertex
                if dimension == 0
                else next(
                    value
                    for value in source.associations
                    if _target_dimension(source.mesh, value) == dimension
                    and (
                        _bank is None
                        or (value.source_id, value.source_revision)
                        == (self.domain.source_id, self.source_revision)
                    )
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
                    "PLC propagated namespace lacks its actual predecessor association."
                )
            # These are the volume owner's exact classification strata; planar
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
                        _entity(self.source_revision, role.value, index)
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
        if _prepared_target is not None:
            if (
                _certification_request is None
                or _certification_request.cell_regions is None
            ):
                raise RuntimeError(
                    "Target PLC evidence lost its complete certification request."
                )
            from ..discretization._coordinate_enclosure import (
                CoordinateEnclosureResourceError,
            )
            from ._certification import (
                _MeshCertificationPremiseFailure,
                MeshCertificationPreparedEvidence,
            )

            try:
                with mapped.ledger.activate():
                    prepared_target = MeshCertificationPreparedEvidence(
                        target,
                        geometry,
                        mapped.prepared.validity,
                        schedule=_certification_request.schedule,
                        domain=mapped.domain,
                        cell_regions=_certification_request.cell_regions,
                        embedding=mapped.embedding,
                        limits=_certification_request.limits,
                        source=_certification_request.source,
                        fidelity_tolerance=_certification_request.fidelity_tolerance,
                        fidelity_sample_order=_certification_request.fidelity_sample_order,
                        scoped_fidelity=_certification_request.scoped_fidelity,
                    )
            except _MeshCertificationPremiseFailure as error:
                refusals = [
                    finding
                    for finding in error.coverage.findings
                    if finding.resource == "coefficient_work"
                ]
                if len(refusals) != 1:
                    raise
                requested = dict(refusals[0].requested)
                achieved = dict(refusals[0].achieved)
                raise CoordinateEnclosureResourceError(
                    "coefficient_work",
                    requested["limit"],
                    requested["requested"],
                    achieved["completed"],
                ) from error
            prepared_target.require(target, geometry, _certification_request)
            _prepared_target.append(prepared_target)
        if _mapped_target is not None:
            _mapped_target.append(mapped)
        return tuple(output)


def _surface_source_association_kind(
    source_kinds: tuple[str, ...], /
) -> GeometryAssociationKind:
    """Resolve the declared original stratum namespace, never relabel an authority."""
    match source_kinds:
        case ("corner", "curve", "surface"):
            return GeometryAssociationKind.SURFACE
        case ("vertex", "edge", "face"):
            return GeometryAssociationKind.BREP
        case _:
            raise ValueError(
                "Parametric source transfer requires native surface or B-Rep stratum identities."
            )


from ..discretization._sphere_chart_deformation import PreparedSphereChartDeformation
from ..discretization._surface_chart_deformation import PreparedSurfaceChartDeformation
from ..geometry._surface_source_support import (
    PreparedSurfaceSourceSupport,
    SurfaceSourceReceipts,
)


@final
class SurfaceAssociationTransfer(StrictModule, NonTrainableState):
    """Preserve original parametric-source authority through native restrictions.

    Receipt preparation is collective and immutable. Owner-local methods never
    gather global source banks or initiate rank-local source collectives.
    """

    support: PreparedSurfaceSourceSupport
    policy: AssociationPropagationPolicy
    receipts: SurfaceSourceReceipts | None
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: PreparedSurfaceSourceSupport,
        /,
        *,
        policy: AssociationPropagationPolicy | None = None,
        receipts: SurfaceSourceReceipts | None = None,
    ) -> None:
        if not isinstance(support, PreparedSurfaceSourceSupport):
            raise TypeError(
                "Surface transfer requires prepared original surface source support."
            )
        _surface_source_association_kind(support.domain.source_kinds)
        if receipts is not None and (
            not isinstance(receipts, SurfaceSourceReceipts)
            or receipts.support is not support
        ):
            raise ValueError(
                "Current surface receipts must consume this exact original source support owner."
            )
        self.support = support
        self.policy = (
            AssociationPropagationPolicy() if policy is None else _policy(policy)
        )
        self.receipts = receipts
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "surface-association-transfer",
                "support": support.support_id,
                "policy": self.policy.policy_id,
                "receipts": None if receipts is None else receipts.receipt_id,
            }
        )

    def with_receipts(
        self, receipts: SurfaceSourceReceipts, /
    ) -> SurfaceAssociationTransfer:
        """Bind a genuinely prepared current projection, without mutating epochs."""
        return SurfaceAssociationTransfer(
            self.support, policy=self.policy, receipts=receipts
        )

    def source_associations(
        self, source: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        from ._surface_association_transfer import source_associations

        return source_associations(self.support, source)

    def classes(self, source: CellMeshingResult, /) -> tuple[SurfaceEntityClasses, ...]:
        from ._surface_association_transfer import classes

        return classes(self.support, source)

    def protected_edges(
        self, source: CellMeshingResult, /, *, midpoint_required: bool = True
    ) -> np.ndarray:
        from ._surface_association_transfer import protected_edges

        return protected_edges(self.support, source, midpoint_required=midpoint_required)

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
        embedding: GlobalEmbeddingCertificate | None = None,
        coverage: DomainCoverageCertificate | None = None,
        deformation: PreparedSurfaceChartDeformation
        | PreparedSphereChartDeformation
        | None = None,
    ) -> tuple[GeometryAssociation, ...]:
        from ._surface_association_transfer import propagate

        return propagate(
            self.support,
            source,
            lineage,
            target,
            geometry=geometry,
            receipts=self.receipts,
            embedding=embedding,
            coverage=coverage,
            deformation=deformation,
        )


from ..geometry._mapped_reference_domain import MappedReferenceDomain


@final
class MappedReferenceAssociationTransfer(StrictModule, NonTrainableState):
    """Carry exact original-root image identities through nested mixed-cell epochs."""

    domain: MappedReferenceDomain
    maximum_support_queries: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self, domain: MappedReferenceDomain, /, *, maximum_support_queries: int = 1 << 26
    ) -> None:
        from .._validation import positive_integer

        if not isinstance(domain, MappedReferenceDomain):
            raise TypeError(
                "Mapped transfer requires an independently declared mapped source."
            )
        if any(
            block.cell_kind not in ("tetrahedron", "prism", "hexahedron")
            for block in domain.reference_mesh.blocks
        ):
            raise ValueError(
                "Mapped transfer requires canonical tetrahedral, prismatic or hexahedral source-root charts."
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

    def associations(
        self, mesh: CellMesh, geometry: CellGeometrySpec, /
    ) -> tuple[GeometryAssociation, ...]:
        from ._mapped_reference_association import mapped_reference_associations

        return mapped_reference_associations(
            self.domain,
            mesh,
            geometry,
            maximum_support_queries=self.maximum_support_queries,
        )

    def source_associations(
        self, source: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        from ._result import CellMeshingResult

        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be an accepted CellMeshingResult.")
        certificate = source.certification
        if (
            certificate is None
            or not certificate.passed
            or certificate.coverage is None
            or certificate.coverage.domain_id != self.domain.domain_id
        ):
            raise ValueError("Mapped transfer requires current original-source coverage.")
        certificate.request.validate_source_integrity()
        certificate.coverage.binding.require(source.mesh, source.geometry)
        expected = self.associations(source.mesh, source.geometry)
        by_dimension = {
            _target_dimension(source.mesh, item): item for item in source.associations
        }
        if len(by_dimension) != len(source.associations) or set(by_dimension) != set(
            range(4)
        ):
            raise ValueError(
                "Mapped source requires one complete association table per dimension."
            )
        for degree, proof in enumerate(expected):
            item = by_dimension[degree]
            item.validate_target(source.mesh.entity_set(degree))
            rows = item.target_rows(np.asarray(proof.target_global_ids))
            if (
                item.association_kind is not GeometryAssociationKind.MAPPED_REFERENCE
                or item.source_id != self.domain.source_id
                or item.source_revision != self.domain.source_revision
                or not item.exact
                or not np.all(np.asarray(item.resolved))
                or np.any(np.asarray(item.ambiguous))
                or tuple(item.source_entity_ids[row] for row in rows)
                != proof.source_entity_ids
                or not np.array_equal(
                    np.asarray(item.parent_dimensions)[rows],
                    np.asarray(proof.parent_dimensions),
                )
                or not np.array_equal(
                    np.asarray(item.parent_ids)[rows], np.asarray(proof.parent_ids)
                )
                or not np.array_equal(
                    np.asarray(item.orientations)[rows], np.asarray(proof.orientations)
                )
            ):
                raise ValueError(
                    "Mapped source associations disagree with their exact original-root images."
                )
        return by_dimension[0], (1, 2, 3)

    def classes(self, source: CellMeshingResult, /) -> tuple[PlcEntityClasses, ...]:
        self.source_associations(source)
        authority = self.domain.reference_mesh
        offsets = np.cumsum(
            np.asarray(
                (0, *(authority.entity_set(degree).count for degree in range(3))),
                dtype=np.int64,
            )
        )
        result = []
        for degree in range(4):
            item = next(
                item
                for item in source.associations
                if _target_dimension(source.mesh, item) == degree
            )
            rows = item.target_rows(np.asarray(source.mesh.entity_set(degree).entity_ids))
            dimensions = np.asarray(item.parent_dimensions, dtype=np.int64)[rows]
            identifiers = np.asarray(item.parent_ids, dtype=np.int64)[rows]
            codes = np.empty(identifiers.shape, dtype=np.int64)
            for kind in range(4):
                lookup = {
                    int(identifier): row
                    for row, identifier in enumerate(
                        np.asarray(authority.entity_set(kind).entity_ids)
                    )
                }
                selected = dimensions == kind
                codes[selected] = offsets[kind] + np.asarray(
                    [lookup[int(identifier)] for identifier in identifiers[selected]],
                    dtype=np.int64,
                )
            result.append(
                PlcEntityClasses(
                    dimensions,
                    identifiers,
                    np.ones(dimensions.shape, dtype=np.bool_),
                    codes,
                )
            )
        return tuple(result)

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        geometry: CellGeometrySpec,
    ) -> tuple[GeometryAssociation, ...]:
        from ._lineage import EntityLineageKind, MeshLineage

        self.source_associations(source)
        if (
            not isinstance(lineage, MeshLineage)
            or lineage.source_topology_id != source.mesh.topology_id
            or lineage.target_topology_id != target.topology_id
        ):
            raise ValueError(
                "Mapped transfer requires the actual source-to-target topology lineage."
            )
        for degree in range(4):
            if np.any(
                np.asarray(lineage.entity_lineage(degree).relation_kinds)
                == EntityLineageKind.UNKNOWN
            ):
                raise ValueError(
                    "Unknown topology lineage cannot preserve original mapped source strata."
                )
        return self.associations(target, geometry)


__all__ = [
    "AssociationPropagationError",
    "AssociationPropagationPolicy",
    "BRepAssociationTransfer",
    "PlcAssociationTransfer",
    "MappedReferenceAssociationTransfer",
    "SurfaceAssociationTransfer",
    "PlcEntityClasses",
    "GeometryAssociation",
    "GeometryAssociationKind",
    "GeometrySourceEntityRole",
    "GeometryAssociationProvenance",
    "associate_mesh_entities",
    "associate_mesh_vertices",
    "propagate_association",
    "rederive_association",
]
