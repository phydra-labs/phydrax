#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry association records, source strata, and host entity row lookup."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any, assert_never, final, SupportsIndex

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._topology as topology_owner

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
)
from ..geometry.brep._projection_contracts import brep_entity_id
from ..typing import (
    as_host_array,
    checked,
    Dim,
    HostInt32,
    HostInt64,
    HostInteger,
    parse,
    Scope,
)
from ._host_banks import entity_key


@jax.jit
def _target_id_membership(target_ids: Array, entity_ids: Array) -> Array:
    return jnp.all(jnp.isin(target_ids, entity_ids))


class GeometryAssociationKind(StrEnum):
    """Representation of the associated source entities.

    ``PIECEWISE_LINEAR`` names represented vertices, segments, facets, and
    regions; indexed rows retain their explicit source stratum and identifier.
    ``CURVE`` names parametric one-dimensional chart sources. ``IMPLICIT``
    names the scalar source's zero-set boundary in its source/revision namespace.
    ``MAPPED_REFERENCE`` names source-defined image entities of a declared reference mesh.
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


type AssociationSourceIdentity = tuple[
    str, GeometrySourceEntityRole | None, tuple[str, ...]
]


def _source_identity_key(
    value: AssociationSourceIdentity,
) -> tuple[str, str, tuple[str, ...]]:
    return value[0], "" if value[1] is None else value[1].value, value[2]


def _require_source_identity(
    entry: AssociationSourceIdentity, indexed_roles: bool, /
) -> None:
    if type(entry) is not tuple or len(entry) != 3:
        raise TypeError(
            "GeometryAssociationSourceBank.entries must contain identity triples; got another row"
        )
    name, role, path = entry
    if type(name) is not str or not name or name != name.strip():
        raise ValueError(
            "GeometryAssociationSourceBank.entries must retain nonempty trimmed source IDs; got an invalid name"
        )
    match role:
        case None | GeometrySourceEntityRole():
            pass
        case _:
            raise TypeError(
                "GeometryAssociationSourceBank.entries must retain declared source roles; got another role type"
            )
    if not indexed_roles and role is not None:
        raise ValueError(
            "GeometryAssociationSourceBank.indexed_roles must declare retained roles; got a role on an absent role bank"
        )
    if type(path) is not tuple or any(
        type(value) is not str or not value or value != value.strip() for value in path
    ):
        raise ValueError(
            "GeometryAssociationSourceBank.entries must retain occurrence-name tuples; got an invalid path"
        )


def _require_source_bank_shape(
    entries: object, row_indices: object, indexed_roles: object, /
) -> None:
    if type(entries) is not tuple or type(row_indices) is not tuple:
        raise TypeError(
            "GeometryAssociationSourceBank.metadata must use immutable tuples; got another bank or route representation"
        )
    if type(indexed_roles) is not bool:
        raise TypeError(
            "GeometryAssociationSourceBank.indexed_roles must be a bool; got another role declaration"
        )


def _require_source_bank(
    entries: tuple[AssociationSourceIdentity, ...],
    row_indices: tuple[int, ...],
    indexed_roles: bool,
    /,
) -> None:
    _require_source_bank_shape(entries, row_indices, indexed_roles)
    for entry in entries:
        _require_source_identity(entry, indexed_roles)
    if tuple(sorted(set(entries), key=_source_identity_key)) != entries:
        raise ValueError(
            "GeometryAssociationSourceBank.entries must be sorted and unique; got duplicate or unordered identities"
        )
    if any(
        type(index) is not int or not 0 <= index < len(entries) for index in row_indices
    ) or set(row_indices) != set(range(len(entries))):
        raise ValueError(
            "GeometryAssociationSourceBank.row_indices must cover every declared identity in range; got an unused or undeclared identity"
        )


@final
@dataclass(frozen=True, slots=True, init=False)
class GeometryAssociationSourceBank:
    """Store named source identities once with their ordered semantic row routes."""

    entries: tuple[AssociationSourceIdentity, ...]
    row_indices: tuple[int, ...]
    indexed_roles: bool

    def __init__(
        self,
        entries: tuple[AssociationSourceIdentity, ...],
        row_indices: tuple[int, ...],
        indexed_roles: bool,
    ) -> None:
        _require_source_bank(entries, row_indices, indexed_roles)
        # This array-free frozen dataclass is not a StrictModule. Assign only
        # after every metadata invariant has been checked.
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "row_indices", row_indices)
        object.__setattr__(self, "indexed_roles", indexed_roles)

    @classmethod
    def from_rows(
        cls: type[GeometryAssociationSourceBank],
        entity_ids: tuple[str, ...],
        roles: tuple[GeometrySourceEntityRole | None, ...] | None,
        paths: tuple[tuple[str, ...], ...],
    ) -> GeometryAssociationSourceBank:
        rows = tuple(
            zip(
                entity_ids,
                (None,) * len(entity_ids) if roles is None else roles,
                paths,
                strict=True,
            )
        )
        entries = tuple(sorted(set(rows), key=_source_identity_key))
        positions = {value: index for index, value in enumerate(entries)}
        return cls(entries, tuple(positions[value] for value in rows), roles is not None)

    def entity_id(self, row: SupportsIndex, /) -> str:
        return self.entries[self.row_indices[row]][0]

    def role(self, row: SupportsIndex, /) -> GeometrySourceEntityRole | None:
        return self.entries[self.row_indices[row]][1]

    def occurrence_path(self, row: SupportsIndex, /) -> tuple[str, ...]:
        return self.entries[self.row_indices[row]][2]

    def entity_ids(self) -> tuple[str, ...]:
        return tuple(self.entity_id(row) for row in range(len(self.row_indices)))

    def roles(self) -> tuple[GeometrySourceEntityRole | None, ...] | None:
        return (
            tuple(self.role(row) for row in range(len(self.row_indices)))
            if self.indexed_roles
            else None
        )

    def occurrence_paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self.occurrence_path(row) for row in range(len(self.row_indices)))

    def row_columns(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[GeometrySourceEntityRole | None, ...] | None,
        tuple[tuple[str, ...], ...],
    ]:
        """Expand aligned pointer columns, borrowing immutable entry values."""
        entries, routes = self.entries, self.row_indices
        return (
            tuple(entries[index][0] for index in routes),
            (
                tuple(entries[index][1] for index in routes)
                if self.indexed_roles
                else None
            ),
            tuple(entries[index][2] for index in routes),
        )

    def to_record(self) -> Mapping[str, Any]:
        """Return explicit identity columns and ordered row routes."""
        _require_source_bank(self.entries, self.row_indices, self.indexed_roles)
        return {
            "entity_ids": tuple(entry[0] for entry in self.entries),
            "entity_roles": tuple(
                None if entry[1] is None else entry[1].value for entry in self.entries
            ),
            "occurrence_paths": tuple(entry[2] for entry in self.entries),
            "row_indices": self.row_indices,
            "indexed_roles": self.indexed_roles,
        }

    @classmethod
    def from_record(
        cls: type[GeometryAssociationSourceBank], payload: Mapping[str, Any], /
    ) -> GeometryAssociationSourceBank:
        """Restore identity columns through full bank validation."""
        if set(payload) != {
            "entity_ids",
            "entity_roles",
            "occurrence_paths",
            "row_indices",
            "indexed_roles",
        } or any(
            type(payload[name]) is not tuple
            for name in (
                "entity_ids",
                "entity_roles",
                "occurrence_paths",
                "row_indices",
            )
        ):
            raise ValueError(
                "GeometryAssociationSourceBank.payload must contain immutable identity columns and row routes; got invalid fields"
            )
        identifiers = payload["entity_ids"]
        roles = payload["entity_roles"]
        paths = payload["occurrence_paths"]
        if len(identifiers) != len(roles) or len(identifiers) != len(paths):
            raise ValueError(
                "GeometryAssociationSourceBank.payload must align every identity column; got differing lengths"
            )
        if any(role is not None and type(role) is not str for role in roles):
            raise TypeError(
                "GeometryAssociationSourceBank.entity_roles must contain role values or None; got another wire type"
            )
        declared_roles = {role.value: role for role in GeometrySourceEntityRole}
        if any(role is not None and role not in declared_roles for role in roles):
            raise ValueError(
                "GeometryAssociationSourceBank.entity_roles must name declared source roles; got an unknown role value"
            )
        return cls(
            tuple(
                (
                    identifier,
                    None if role is None else declared_roles[role],
                    path,
                )
                for identifier, role, path in zip(identifiers, roles, paths, strict=True)
            ),
            payload["row_indices"],
            payload["indexed_roles"],
        )


class GeometryAssociationProvenance(StrEnum):
    """How the rows were established.

    ``PROVIDER``: emitted by the generating mesher. ``LINEAGE``: propagated
    through a declared topology lineage from ``parent_association_id``.
    ``CLASSIFICATION``: derived by geometric classification and projection
    (optionally restricted by classification transfer from the parent).
    """

    PROVIDER = "provider"
    LINEAGE = "lineage"
    CLASSIFICATION = "classification"


def unclassified_entity_id(source_revision: str, /) -> str:
    return f"{source_revision}:unclassified"


def association_rows_or(
    value: ArrayLike | None, fill: Any, shape: Any, dtype: Any, /
) -> np.ndarray:
    return (
        np.full(shape, fill, dtype=dtype) if value is None else np.asarray(value, dtype)
    )


class _AssociationTargets(StrictModule):
    source: str
    revision: str
    entity_set: str
    targets: np.ndarray
    paths: tuple[tuple[str, ...], ...]
    sources: tuple[str, ...]
    distances: np.ndarray


class _AssociationStatus(StrictModule):
    resolved: np.ndarray
    ambiguous: np.ndarray


def _prepare_association_targets(
    association_kind: GeometryAssociationKind,
    source_id: str,
    source_revision: str,
    target_entity_set_id: str,
    target_global_ids: ArrayLike,
    source_entity_ids: tuple[str, ...],
    residuals: ArrayLike,
    source_occurrence_paths: tuple[tuple[str, ...], ...] | None,
) -> _AssociationTargets:
    """Prepare row-aligned source identities, occurrence paths, and residuals."""
    source = str(source_id).strip()
    revision = str(source_revision).strip()
    entity_set = str(target_entity_set_id).strip()
    if not source or not revision or not entity_set:
        raise ValueError(
            f"GeometryAssociation.identities must be nonempty; got source {source!r}, revision {revision!r}, entity set {entity_set!r}"
        )
    targets = np.asarray(target_global_ids)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise TypeError(
            f"GeometryAssociation.target_global_ids must be one integer vector; got shape {targets.shape!r}, dtype {targets.dtype}"
        )
    targets = targets.astype(np.int64, copy=False)
    if (
        targets.size == 0
        or np.any(targets < 0)
        or np.unique(targets).size != targets.size
    ):
        raise ValueError(
            f"GeometryAssociation.target_global_ids must be nonempty, unique, and nonnegative; got {targets!r}"
        )
    count = targets.size
    paths = ((),) * count if source_occurrence_paths is None else source_occurrence_paths
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
            f"GeometryAssociation.source_occurrence_paths must be tuples of nonempty names aligned with target rows; got {paths!r}, row count {count}"
        )
    if association_kind not in (
        GeometryAssociationKind.BREP,
        GeometryAssociationKind.SURFACE,
    ) and any(paths):
        raise ValueError(
            f"GeometryAssociation.source_occurrence_paths must be empty outside B-Rep and surface representations; got kind {association_kind!r}, paths {paths!r}"
        )
    sources = tuple(str(value).strip() for value in source_entity_ids)
    distances = np.asarray(residuals, dtype=np.float64)
    if len(sources) != count or any(not value for value in sources):
        raise ValueError(
            f"GeometryAssociation.source_entity_ids must be nonempty names aligned with target rows; got {sources!r}, row count {count}"
        )
    return _AssociationTargets(
        source, revision, entity_set, targets, paths, sources, distances
    )


def _validate_association_status(
    target: _AssociationTargets,
    resolved: ArrayLike | None,
    ambiguous: ArrayLike | None,
    exact: bool,
) -> _AssociationStatus:
    """Validate support and ambiguity masks against the row residual and exactness contract."""
    targets, distances = target.targets, target.distances
    count = targets.size
    resolved_ = association_rows_or(resolved, True, (count,), np.bool_)
    ambiguous_ = association_rows_or(ambiguous, False, (count,), np.bool_)
    if resolved_.shape != targets.shape or ambiguous_.shape != targets.shape:
        raise ValueError(
            f"GeometryAssociation.status_masks must align with target rows; got resolved shape {resolved_.shape!r}, ambiguous shape {ambiguous_.shape!r}, target shape {targets.shape!r}"
        )
    if (
        distances.shape != targets.shape
        or np.any(np.isnan(distances))
        or np.any(distances < 0)
        or np.any(np.isinf(distances) & resolved_)
    ):
        raise ValueError(
            f"GeometryAssociation.residuals must align with target rows and be nonnegative with infinity only on unresolved rows; got residuals {distances!r}, resolved {resolved_!r}, target shape {targets.shape!r}"
        )
    if np.any(resolved_ & ambiguous_):
        raise ValueError(
            f"GeometryAssociation.status_masks must not resolve ambiguous rows; got overlap rows {np.flatnonzero(resolved_ & ambiguous_)!r}"
        )
    if exact and (
        not np.all(resolved_) or np.any(ambiguous_) or np.any(distances != 0.0)
    ):
        raise ValueError(
            f"GeometryAssociation.exact must have resolved, unambiguous, zero-residual rows; got resolved {resolved_!r}, ambiguous {ambiguous_!r}, residuals {distances!r}"
        )
    return _AssociationStatus(resolved_, ambiguous_)


class GeometryAssociation(StrictModule, NonTrainableState):
    """Audited map from declared mesh entities to identified geometry entities.

    B-Rep rows carry ``source_dimensions``/``source_indices`` as execution
    definition metadata and explicit ``source_occurrence_paths`` as semantic
    instance identity. ``-1`` marks an unresolved row without a candidate
    (``<revision>:unclassified``). ``parameters`` hold face ``(u, v)`` or edge
    ``(t, nan)`` closest-point parameters (NaN otherwise). ``orientations`` relate
    the mesh entity's incidence-defined vertex order to the B-Rep edge tangent or oriented
    face normal (``+1``/``-1``; ``0`` when undefined). ``parent_dimensions`` and
    ``parent_ids`` name the parent mesh entity whose class a row inherited
    (``-1`` for geometric classification). Infinite residuals are allowed only on
    unresolved rows (no closest point).

    Implicit rows identify ``implicit-zero-set`` independently of optional
    definition indices; paired ``-1`` values omit those indices without changing
    resolved status. A revision-scoped unclassified source ID instead requires
    unresolved status and paired absent indices.

    Curve and mapped-reference rows omit indexed constructor inputs and store
    paired ``-1`` banks. Stored-field replay selects those absent inputs from
    ``association_kind`` and authenticates every retained field, including the
    sentinel banks, against the constructor result.
    """

    association_kind: GeometryAssociationKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    target_global_ids: Array
    source_entities: GeometryAssociationSourceBank = eqx.field(static=True)
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
            raise TypeError(
                f"GeometryAssociation.association_kind must be GeometryAssociationKind; got {association_kind!r}"
            )
        if not isinstance(provenance, GeometryAssociationProvenance):
            raise TypeError(
                f"GeometryAssociation.provenance must be GeometryAssociationProvenance; got {provenance!r}"
            )
        target = _prepare_association_targets(
            association_kind,
            source_id,
            source_revision,
            target_entity_set_id,
            target_global_ids,
            source_entity_ids,
            residuals,
            source_occurrence_paths,
        )
        source, revision, entity_set = target.source, target.revision, target.entity_set
        targets, paths, sources, distances = (
            target.targets,
            target.paths,
            target.sources,
            target.distances,
        )
        count = targets.size
        status = _validate_association_status(target, resolved, ambiguous, exact)
        resolved_, ambiguous_ = status.resolved, status.ambiguous
        roles = source_roles(
            association_kind,
            count,
            source_dimensions,
            source_indices,
            source_entity_roles,
        )
        dimensions, indices = association_source_entities(
            association_kind,
            revision,
            sources,
            source_dimensions,
            source_indices,
            paths,
            roles,
            resolved_,
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
            raise ValueError(
                f"GeometryAssociation.source_dimensions must identify entities on resolved rows; got family {family!r}, unresolved dimensions {dimensions!r}, resolved mask {resolved_!r}"
            )
        parameters_, orientations_, parent_dimensions_, parent_ids_ = (
            association_row_metadata(
                count, parameters, orientations, parent_dimensions, parent_ids
            )
        )
        parent = None if parent_association_id is None else str(parent_association_id)
        if parent is not None and not parent.strip():
            raise ValueError(
                f"GeometryAssociation.parent_association_id must be nonempty when provided; got {parent!r}"
            )
        if provenance is GeometryAssociationProvenance.LINEAGE and parent is None:
            raise ValueError(
                f"GeometryAssociation.parent_association_id must identify the predecessor for lineage provenance; got {parent!r}"
            )
        source_entities = GeometryAssociationSourceBank.from_rows(sources, roles, paths)
        self.association_kind = association_kind
        self.source_id = source
        self.source_revision = revision
        self.target_entity_set_id = entity_set
        self.target_global_ids = jax.device_put(np.asarray(targets))
        self.source_entities = source_entities
        self.residuals = jax.device_put(np.asarray(distances))
        self.resolved = jax.device_put(np.asarray(resolved_))
        self.ambiguous = jax.device_put(np.asarray(ambiguous_))
        self.source_dimensions = jax.device_put(np.asarray(dimensions))
        self.source_indices = jax.device_put(np.asarray(indices))
        self.parameters = jax.device_put(np.asarray(parameters_))
        self.orientations = jax.device_put(np.asarray(orientations_))
        self.parent_dimensions = jax.device_put(np.asarray(parent_dimensions_))
        self.parent_ids = jax.device_put(np.asarray(parent_ids_))
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[GeometryAssociation],
        value: GeometryAssociation,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> GeometryAssociation:
        """Rebuild stored rows through their kind-specific validation.

        An active ``budget`` admits replay work and temporary storage, then owns
        the returned graph for its retained lifetime. Immutable source entries
        are borrowed; expanded columns and hash buffers are not.
        """
        if cls is not GeometryAssociation or type(value) is not GeometryAssociation:
            raise TypeError(
                "stored_fields.GeometryAssociation.type must be the registered GeometryAssociation; "
                f"got {type(value).__qualname__}"
            )
        if type(value.source_entities) is not GeometryAssociationSourceBank:
            raise TypeError(
                "stored_fields.GeometryAssociation.source_entities must retain its typed source bank; got another owner"
            )
        if budget is not None:
            with _association_reconstruction_admission(value, budget=budget) as storage:
                rebuilt = cls._replay_stored_fields(value)
                budget.retain_owner(rebuilt, temporary_storage_credit=storage)
                return rebuilt
        _require_source_bank(
            value.source_entities.entries,
            value.source_entities.row_indices,
            value.source_entities.indexed_roles,
        )
        return cls._replay_stored_fields(value)

    @classmethod
    def _replay_stored_fields(
        cls: type[GeometryAssociation], value: GeometryAssociation, /
    ) -> GeometryAssociation:
        kind = value.association_kind
        if not isinstance(kind, GeometryAssociationKind):
            raise TypeError(
                f"stored_fields.GeometryAssociation.association_kind must be GeometryAssociationKind; got {kind!r}"
            )
        dimensions: Array | None
        indices: Array | None
        match kind:
            case GeometryAssociationKind.CURVE | GeometryAssociationKind.MAPPED_REFERENCE:
                dimensions = None
                indices = None
            case (
                GeometryAssociationKind.BREP
                | GeometryAssociationKind.SURFACE
                | GeometryAssociationKind.PIECEWISE_LINEAR
                | GeometryAssociationKind.IMPLICIT
            ):
                dimensions = value.source_dimensions
                indices = value.source_indices
            case _:
                assert_never(kind)
        sources, roles, paths = value.source_entities.row_columns()
        rebuilt = cls(
            kind,
            value.source_id,
            value.source_revision,
            value.target_entity_set_id,
            value.target_global_ids,
            sources,
            value.residuals,
            resolved=value.resolved,
            ambiguous=value.ambiguous,
            exact=value.exact,
            source_dimensions=dimensions,
            source_indices=indices,
            source_occurrence_paths=paths,
            source_entity_roles=roles,
            parameters=value.parameters,
            orientations=value.orientations,
            parent_dimensions=value.parent_dimensions,
            parent_ids=value.parent_ids,
            parent_association_id=value.parent_association_id,
            provenance=value.provenance,
        )
        for member in fields(value):
            if not tree_content_equal(
                getattr(rebuilt, member.name), getattr(value, member.name)
            ):
                raise ValueError(
                    f"stored_fields.GeometryAssociation.{member.name} must match reconstructed retained content; got changed {member.name}"
                )
        return rebuilt

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
    def target_matches(self, entity_set: topology_owner.EntitySet, /) -> bool:
        """Return whether the table binds declared IDs of this entity set."""
        if self.target_entity_set_id != entity_set.entity_set_id:
            return False
        if all(
            identifiers.is_fully_addressable
            and isinstance(identifiers.sharding, jax.sharding.SingleDeviceSharding)
            for identifiers in (self.target_global_ids, entity_set.entity_ids)
        ):
            return bool(
                np.all(
                    np.isin(
                        np.asarray(self.target_global_ids),
                        np.asarray(entity_set.entity_ids),
                    )
                )
            )
        return bool(_target_id_membership(self.target_global_ids, entity_set.entity_ids))

    @checked
    def validate_target(self, entity_set: topology_owner.EntitySet, /) -> None:
        """Require target entity-set identity and membership of every association row."""
        if self.target_entity_set_id != entity_set.entity_set_id:
            raise ValueError(
                f"GeometryAssociation.target_entity_set_id must match the target entity set; got {self.target_entity_set_id!r}, expected {entity_set.entity_set_id!r}"
            )
        if not self.target_matches(entity_set):
            raise ValueError(
                f"GeometryAssociation.target_global_ids must belong to the target entity set; got membership failure for association {self.association_id!r}"
            )

    def target_rows(self, target_global_ids: ArrayLike, /) -> np.ndarray:
        """Return host rows for target global IDs; every requested ID must be present."""
        requested = np.asarray(target_global_ids)
        if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
            raise TypeError(
                f"GeometryAssociation.target_global_ids must be one integer vector; got shape {requested.shape!r}, dtype {requested.dtype}"
            )
        targets = np.asarray(self.target_global_ids, dtype=np.int64)
        order = np.argsort(targets, kind="stable")
        position = np.minimum(np.searchsorted(targets[order], requested), order.size - 1)
        rows = order[position]
        if not np.array_equal(targets[rows], requested):
            raise ValueError(
                f"GeometryAssociation.target_global_ids must classify every requested ID; got requested {requested!r}, declared {targets!r}"
            )
        return rows


@contextmanager
def _association_reconstruction_admission(
    value: GeometryAssociation, /, *, budget: CoordinateEnclosureBudget
) -> Iterator[CoordinateLiveStorage]:
    """Admit replay's shared entry graph, pointer columns and buffers."""
    bank = value.source_entities
    _require_source_bank_shape(bank.entries, bank.row_indices, bank.indexed_roles)
    with budget.temporary_scope(), budget.live_storage() as storage:
        storage.set_bound(512, work=1)
        metadata = budget.live_object_storage_upper(value)
        arrays = tuple(
            item
            for member in fields(value)
            if isinstance(item := getattr(value, member.name), (Array, np.ndarray))
        )
        rows, keys = len(bank.row_indices), len(bank.entries)
        array_bytes = sum(max(array.nbytes, 8 * array.size) for array in arrays)
        # Valid retained numeric owners have ten banks. Malformed alternate
        # representations still need bounded normalization before owner refusal.
        alternate_storage = 0 if len(arrays) == 10 else 8 * metadata
        # Work follows the retained-owner convention: one unit per visited row
        # route, entry and numeric bank. Storage bounds the replay copies.
        storage.set_bound(
            storage.bound
            + 4 * array_bytes
            + alternate_storage
            + 1024 * (rows + keys + 1),
            work=rows + keys + len(arrays),
        )
        # This is the same validation as the unbudgeted entry point, performed
        # once before indexing. Row columns borrow names/roles/path tuples.
        _require_source_bank(bank.entries, bank.row_indices, bank.indexed_roles)
        text_lengths: list[int] = []
        path_lengths: list[int] = []
        for name, role, path in bank.entries:
            budget.reserve(1 + len(path))
            text_lengths.append(
                len(name)
                + (0 if role is None else len(role.value))
                + sum(len(part) for part in path)
                + 2 * len(path)
                + 16
            )
            path_lengths.append(len(path))
        expanded_text = sum(text_lengths[index] for index in bank.row_indices)
        path_visits = sum(path_lengths[index] for index in bank.row_indices)
        # Unicode/UTF-8 serialization and generated identity text are buffers,
        # not new deep Python entry graphs for each route.
        revision_width = (
            len(value.source_revision)
            if isinstance(value.source_revision, str)
            else metadata
        )
        key_text = rows * (revision_width + 32)
        storage.set_bound(
            storage.bound + 24 * (expanded_text + key_text), work=path_visits
        )
        budget.ticket.checkpoint()
        yield storage
        budget.ticket.checkpoint()


def association_row_metadata(
    count: int,
    parameters: ArrayLike | None,
    orientations: ArrayLike | None,
    parent_dimensions: ArrayLike | None,
    parent_ids: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validated closest-point parameters, orientations, and parents of every row."""
    parameters_ = association_rows_or(parameters, np.nan, (count, 2), np.float64)
    orientations_ = association_rows_or(orientations, 0, (count,), np.int8)
    parent_dimensions_ = association_rows_or(parent_dimensions, -1, (count,), np.int8)
    parent_ids_ = association_rows_or(parent_ids, -1, (count,), np.int64)
    if (
        parameters_.shape != (count, 2)
        or np.any(np.isinf(parameters_))
        or orientations_.shape != (count,)
        or parent_dimensions_.shape != (count,)
        or parent_ids_.shape != (count,)
    ):
        raise ValueError(
            f"GeometryAssociation.row_metadata must align with target rows and omit infinite parameters; got shapes {(parameters_.shape, orientations_.shape, parent_dimensions_.shape, parent_ids_.shape)!r}, row count {count}, infinite parameters {bool(np.any(np.isinf(parameters_)))}"
        )
    if not np.all(np.isin(orientations_, (-1, 0, 1))):
        raise ValueError(
            f"GeometryAssociation.orientations must be -1, 0, or 1; got {orientations_!r}"
        )
    if (
        np.any((parent_dimensions_ < -1) | (parent_dimensions_ > 3))
        or np.any(parent_ids_ < -1)
        or np.any((parent_dimensions_ < 0) != (parent_ids_ < 0))
    ):
        raise ValueError(
            f"GeometryAssociation.parents must pair dimensions in [-1, 3] with nonnegative IDs or paired -1 values; got dimensions {parent_dimensions_!r}, IDs {parent_ids_!r}"
        )
    return parameters_, orientations_, parent_dimensions_, parent_ids_


class SourceEntityRowDim(Dim):
    """Rows sharing one declared source-entity metadata binding."""


def source_entity_rows(
    dimensions: ArrayLike | None,
    indices: ArrayLike | None,
    count: int,
    family: str,
    index_dtype: type[np.int32] | type[np.int64],
    /,
) -> tuple[
    HostInteger[SourceEntityRowDim],
    HostInt32[SourceEntityRowDim] | HostInt64[SourceEntityRowDim],
]:
    if dimensions is None or indices is None:
        raise ValueError(
            f"GeometryAssociation.source_strata must provide dimensions and indices; got family {family!r}, dimensions present {dimensions is not None}, indices present {indices is not None}"
        )
    dimensions_ = np.asarray(dimensions)
    indices_ = np.asarray(indices)
    if not np.issubdtype(dimensions_.dtype, np.integer) or not np.issubdtype(
        indices_.dtype, np.integer
    ):
        raise TypeError(
            f"GeometryAssociation.source_strata must use integer dimensions and indices; got family {family!r}, dtypes {(str(dimensions_.dtype), str(indices_.dtype))!r}"
        )
    if dimensions_.shape != (count,) or indices_.shape != (count,):
        raise ValueError(
            f"GeometryAssociation.source_strata must align with source rows; got family {family!r}, shapes {(dimensions_.shape, indices_.shape)!r}, row count {count}"
        )
    if (
        np.any((dimensions_ < -1) | (dimensions_ > 3))
        or np.any(indices_ < -1)
        or np.any((dimensions_ >= 0) != (indices_ >= 0))
    ):
        raise ValueError(
            f"GeometryAssociation.source_strata must pair dimensions in [-1, 3] with nonnegative indices or paired -1 values; got family {family!r}, dimensions {dimensions_!r}, indices {indices_!r}"
        )
    if np.any(indices_ > np.iinfo(index_dtype).max):
        raise ValueError(
            f"GeometryAssociation.source_indices must fit the native index dtype; got family {family!r}, indices {indices_!r}, limit {np.iinfo(index_dtype).max}"
        )
    scope = Scope()
    dimensions_ = parse(
        dimensions_.astype(np.int8),
        HostInteger[SourceEntityRowDim],
        "source_dimensions",
        scope=scope,
    )
    if index_dtype is np.int64:
        indices_ = as_host_array(
            indices_,
            HostInt64[SourceEntityRowDim],
            "source_indices",
            scope=scope,
        )
    else:
        indices_ = as_host_array(
            indices_,
            HostInt32[SourceEntityRowDim],
            "source_indices",
            scope=scope,
        )
    return dimensions_, indices_


def source_roles(
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
            raise ValueError(
                f"GeometryAssociation.source_entity_roles must be absent outside indexed PLC representations; got kind {kind!r}, roles {roles!r}"
            )
        return None
    if not isinstance(roles, tuple):
        raise TypeError(
            f"GeometryAssociation.source_entity_roles must be a tuple for indexed PLC rows; got {roles!r}"
        )
    if len(roles) != count:
        raise ValueError(
            f"GeometryAssociation.source_entity_roles must align with source rows; got count {len(roles)}, expected {count}"
        )
    return tuple(
        None
        if role is None
        else parse(role, GeometrySourceEntityRole, "source_entity_role")
        for role in roles
    )


def plc_source_entity_keys(
    revision: str,
    dimensions: np.ndarray,
    indices: np.ndarray,
    roles: tuple[GeometrySourceEntityRole | None, ...],
    /,
) -> tuple[str, ...]:

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
                        f"GeometryAssociation.unclassified_source must pair absent role with dimension and index -1; got {(dimension, index)!r}"
                    )
                keys.append(unclassified_entity_id(revision))
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
                raise TypeError(
                    f"GeometryAssociation.source_entity_roles must contain GeometrySourceEntityRole values; got {role!r}"
                )
        if not valid or index < 0:
            raise ValueError(
                f"GeometryAssociation.source_entity_roles must match source dimension and nonnegative index; got role {role!r}, dimension {dimension}, index {index}"
            )
        keys.append(entity_key(revision, role.value, index))
    return tuple(keys)


def association_source_entities(
    kind: GeometryAssociationKind,
    revision: str,
    sources: tuple[str, ...],
    dimensions: ArrayLike | None,
    indices: ArrayLike | None,
    occurrence_paths: tuple[tuple[str, ...], ...],
    roles: tuple[GeometrySourceEntityRole | None, ...] | None,
    resolved: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(sources)
    if (
        kind not in (GeometryAssociationKind.BREP, GeometryAssociationKind.IMPLICIT)
        and dimensions is None
        and indices is None
    ):
        return np.full((count,), -1, np.int8), np.full((count,), -1, np.int32)
    match kind:
        case GeometryAssociationKind.BREP:
            dimensions_, indices_ = source_entity_rows(
                dimensions,
                indices,
                count,
                "B-Rep",
                np.int32,
            )
            expected = tuple(
                brep_entity_id(revision, dimension, index, occurrence_path=path)
                if dimension >= 0
                else unclassified_entity_id(revision)
                for dimension, index, path in zip(
                    dimensions_.tolist(),
                    indices_.tolist(),
                    occurrence_paths,
                    strict=True,
                )
            )
            if expected != sources:
                raise ValueError(
                    f"GeometryAssociation.source_entity_ids must encode the B-Rep revision, entity, and occurrence path; got {sources!r}, expected {expected!r}"
                )
        case GeometryAssociationKind.PIECEWISE_LINEAR:
            if roles is None:
                raise ValueError(
                    f"GeometryAssociation.source_entity_roles must identify indexed PLC strata; got {roles!r}"
                )
            dimensions_, indices_ = source_entity_rows(
                dimensions,
                indices,
                count,
                "PLC",
                np.int64,
            )
            expected = plc_source_entity_keys(revision, dimensions_, indices_, roles)
            if expected != sources:
                raise ValueError(
                    f"GeometryAssociation.source_entity_ids must encode the PLC revision and stratum; got {sources!r}, expected {expected!r}"
                )
        case GeometryAssociationKind.IMPLICIT:
            if dimensions is None and indices is None:
                dimensions = np.full((count,), -1, dtype=np.int8)
                indices = np.full((count,), -1, dtype=np.int32)
            dimensions_, indices_ = source_entity_rows(
                dimensions,
                indices,
                count,
                "Implicit",
                np.int32,
            )
            indexed = dimensions_ >= 0
            if np.any(dimensions_[indexed] > 2) or np.any(indices_[indexed] != 0):
                raise ValueError(
                    f"GeometryAssociation.implicit_strata must use dimensions in [0, 2] and zero indices on indexed rows; got dimensions {dimensions_!r}, indices {indices_!r}"
                )
            unclassified = tuple(
                source == unclassified_entity_id(revision) for source in sources
            )
            permitted_unclassified = ~indexed & ~resolved
            if any(
                source != "implicit-zero-set"
                and not (missing and permitted_unclassified[row])
                for row, (source, missing) in enumerate(
                    zip(sources, unclassified, strict=True)
                )
            ):
                raise ValueError(
                    f"GeometryAssociation.source_entity_ids must name implicit-zero-set or an unresolved unclassified row; got {sources!r}, resolved {resolved!r}, dimensions {dimensions_!r}"
                )
        case GeometryAssociationKind.SURFACE:
            dimensions_, indices_ = source_entity_rows(
                dimensions,
                indices,
                count,
                "Surface",
                np.int64,
            )
            if np.any(dimensions_ > 2):
                raise ValueError(
                    f"GeometryAssociation.surface_dimensions must identify corners, curves, or patches in [0, 2]; got {dimensions_!r}"
                )
        case GeometryAssociationKind.CURVE | GeometryAssociationKind.MAPPED_REFERENCE:
            raise ValueError(
                f"GeometryAssociation.indexed_strata must be absent for this representation; got kind {kind!r}, dimensions {dimensions!r}, indices {indices!r}"
            )
        case _:
            raise TypeError(
                f"GeometryAssociation.association_kind must be GeometryAssociationKind; got {kind!r}"
            )
    return dimensions_, indices_


@checked
def entity_rows(
    mesh: cell_mesh_owner.CellMesh, dimension: int, identifiers: np.ndarray, /
) -> np.ndarray:
    entity_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    order = np.argsort(entity_ids, kind="stable")
    position = np.minimum(np.searchsorted(entity_ids[order], identifiers), order.size - 1)
    return np.where(entity_ids[order[position]] == identifiers, order[position], -1)


@checked
def target_dimension(
    mesh: cell_mesh_owner.CellMesh, association: GeometryAssociation, /
) -> int:
    for dimension in range(mesh.topological_dimension + 1):
        if mesh.entity_set(dimension).entity_set_id == association.target_entity_set_id:
            return dimension
    raise ValueError(
        f"GeometryAssociation.target_entity_set_id must belong to the source mesh; got {association.target_entity_set_id!r}, mesh topology {mesh.topology_id!r}"
    )
