#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer
from ..discretization import CellMesh, EntitySelection, EntitySet
from ..geometry._source_identity import source_entity_id
from ..typing import checked, ConvertibleToArray


if TYPE_CHECKING:
    from ..geometry.brep._differentiable import FixedTopologyBRepSource
    from ..geometry.brep._model import BRepEntityId, BRepModel
    from ..geometry.brep._source import BRepSource
    from ..geometry.multiregion_surface._label_extraction import (
        LabelFieldSurfaceExtractionResult,
    )
    from ..geometry.surface import SurfaceModel


@runtime_checkable
class _SourceEntityScopes(Protocol):
    """An authoritative geometry source's prepared entity-ID projection."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray: ...

    def scope_entity_set_id(self, dimension: int, /) -> str: ...

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]: ...


def _geometry_model(
    source: BRepModel | BRepSource | FixedTopologyBRepSource,
    /,
) -> BRepModel:
    from ..geometry.brep._differentiable import FixedTopologyBRepSource
    from ..geometry.brep._model import BRepModel
    from ..geometry.brep._source import BRepSource

    if isinstance(source, BRepModel):
        return source
    if isinstance(source, (BRepSource, FixedTopologyBRepSource)):
        return source.model
    raise TypeError("Geometry scopes require a BRep model or source.")


def _source_scope_owner(
    source: BRepModel
    | BRepSource
    | FixedTopologyBRepSource
    | LabelFieldSurfaceExtractionResult
    | _SourceEntityScopes,
    /,
) -> _SourceEntityScopes:
    """Resolve source-owned entity banks without importing an external engine."""
    if isinstance(source, _SourceEntityScopes):
        return source
    from ..geometry.brep._differentiable import FixedTopologyBRepSource
    from ..geometry.brep._model import BRepModel
    from ..geometry.brep._source import BRepSource

    if isinstance(source, (BRepModel, BRepSource, FixedTopologyBRepSource)):
        from ..geometry._meshing_domain import MeshingDomain

        return MeshingDomain.from_brep(_geometry_model(source))
    from ..geometry.multiregion_surface._label_extraction import (
        LabelFieldSurfaceExtractionResult,
    )

    if isinstance(source, LabelFieldSurfaceExtractionResult):
        from .providers._native_compartment import _normalize_label_field_source

        return _normalize_label_field_source(source).source
    raise TypeError("Geometry scopes require a declared source entity owner.")


def _brep_scope_dimension(kind: str, /) -> int:
    match kind:
        case "vertex":
            return 0
        case "edge":
            return 1
        case "face":
            return 2
        case "solid":
            return 3
        case _:
            raise ValueError("Unsupported BRep entity kind.")


def _brep_scope_ids(
    owner: _SourceEntityScopes,
    entities: tuple[BRepEntityId, ...],
    dimension: int | None,
    /,
) -> tuple[int, np.ndarray]:
    """Resolve definition selectors only through unique declared occurrence uses."""
    target = _brep_scope_dimension(entities[0].kind)
    if dimension is not None and nonnegative_integer(dimension, "dimension") != target:
        raise ValueError("The requested dimension disagrees with the BRep entity kind.")
    if any(
        entity.source_id != owner.source_id
        or entity.source_revision != owner.source_revision
        or entity.kind != entities[0].kind
        for entity in entities
    ):
        raise ValueError(
            "Geometry scopes require one entity kind from the actual source owner and revision."
        )
    bank = {
        owner.source_entity_key_for_scope(target, int(identifier)): int(identifier)
        for identifier in owner.scope_entity_ids(target)
    }
    paths = tuple(sorted({path for _, path in bank}))
    selected: list[int] = []
    for entity in entities:
        candidate_paths = (entity.occurrence_path,) if entity.occurrence_path else paths
        matches = []
        for path in candidate_paths:
            key = source_entity_id(
                entity.source_id,
                entity.source_revision,
                entity.kind,
                entity.index,
                occurrence_path=path,
            )
            if (key, path) in bank:
                matches.append(bank[(key, path)])
        if len(matches) != 1:
            raise ValueError(
                "A BRep selector must resolve to exactly one declared occurrence; specify its occurrence path."
            )
        selected.append(matches[0])
    return target, np.asarray(selected, dtype=np.int64)


def _geometry_scope_ids(
    owner: _SourceEntityScopes,
    entity_ids: Sequence[BRepEntityId] | BRepEntityId | ArrayLike,
    dimension: int | None,
    /,
) -> tuple[int, np.ndarray]:
    from ..geometry.brep._model import BRepEntityId

    if isinstance(entity_ids, BRepEntityId):
        return _brep_scope_ids(owner, (entity_ids,), dimension)
    if isinstance(entity_ids, Sequence) and any(
        isinstance(value, BRepEntityId) for value in entity_ids
    ):
        selectors: list[BRepEntityId] = []
        for value in entity_ids:
            if not isinstance(value, BRepEntityId):
                raise TypeError("BRep selectors cannot be mixed with numeric scope IDs.")
            selectors.append(value)
        return _brep_scope_ids(owner, tuple(selectors), dimension)
    if dimension is None:
        raise ValueError("A source entity bank requires an explicit dimension.")
    target = nonnegative_integer(dimension, "dimension")
    identifiers = np.atleast_1d(np.asarray(entity_ids))
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError("Source entity IDs must form one integer vector.")
    if not np.all(np.isin(identifiers, owner.scope_entity_ids(target))):
        raise ValueError("Scope IDs must belong to the declared source entity bank.")
    return target, identifiers


class MeshingEntityKind(StrEnum):
    GEOMETRY = "geometry"
    MESH = "mesh"
    PART = "part"
    LABEL = "label"
    ZONE = "zone"


class MeshingScope(StrictModule, NonTrainableState):
    """Exact entity scope bound to one immutable source revision."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    entity_kind: MeshingEntityKind = eqx.field(static=True)
    entity_dimension: int = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    entity_ids: Array
    scope_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        entity_kind: MeshingEntityKind,
        entity_dimension: int,
        entity_set_id: str,
        entity_ids: ConvertibleToArray,
        /,
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        entity_set = str(entity_set_id).strip()
        if not source or not revision or not entity_set:
            raise ValueError("Meshing scope identities must be non-empty.")
        if not isinstance(entity_kind, MeshingEntityKind):
            raise TypeError("entity_kind must be MeshingEntityKind.")
        dimension = int(entity_dimension)
        if dimension < 0:
            raise ValueError("entity_dimension must be non-negative.")
        identifiers = np.asarray(entity_ids)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("Meshing scope entity_ids must be one integer vector.")
        identifiers = identifiers.astype(np.int64, copy=False)
        if identifiers.size == 0:
            raise ValueError("Meshing scopes must contain at least one entity.")
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("Meshing scope entity_ids must be unique and non-negative.")
        identifiers = np.sort(identifiers, kind="stable")
        self.source_id = source
        self.source_revision = revision
        self.entity_kind = entity_kind
        self.entity_dimension = dimension
        self.entity_set_id = entity_set
        self.entity_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.scope_id = canonical_fingerprint(
            {
                "kind": "meshing-scope",
                "source_id": source,
                "source_revision": revision,
                "entity_kind": entity_kind.value,
                "entity_dimension": dimension,
                "entity_set_id": entity_set,
                "entity_ids": array_tree_fingerprint(identifiers),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MeshingScope], value: MeshingScope, /
    ) -> MeshingScope:
        if type(value) is not cls:
            raise TypeError("value must be MeshingScope.")
        if type(value.entity_dimension) is not int:
            raise TypeError("Stored meshing scope dimension must be an int.")
        stored_ids = np.asarray(value.entity_ids)
        if stored_ids.ndim != 1 or stored_ids.dtype != np.dtype(np.int64):
            raise TypeError("Stored meshing scope entity_ids must be int64.")
        rebuilt = cls(
            value.source_id,
            value.source_revision,
            value.entity_kind,
            value.entity_dimension,
            value.entity_set_id,
            value.entity_ids,
        )
        if (
            rebuilt.scope_id != value.scope_id
            or rebuilt.source_id != value.source_id
            or rebuilt.source_revision != value.source_revision
            or rebuilt.entity_kind is not value.entity_kind
            or rebuilt.entity_dimension != value.entity_dimension
            or rebuilt.entity_set_id != value.entity_set_id
            or not np.array_equal(
                np.asarray(rebuilt.entity_ids), np.asarray(value.entity_ids)
            )
        ):
            raise ValueError("Stored meshing scope differs from its exact entity bank.")
        return rebuilt


    @classmethod
    def from_geometry(
        cls,
        source: BRepModel
        | BRepSource
        | FixedTopologyBRepSource
        | LabelFieldSurfaceExtractionResult
        | _SourceEntityScopes,
        entity_ids: Sequence[BRepEntityId] | BRepEntityId | ArrayLike,
        /,
        *,
        dimension: int | None = None,
    ) -> MeshingScope:
        """Bind exact source entities; ambiguous definition uses need an occurrence."""
        owner = _source_scope_owner(source)
        target, identifiers = _geometry_scope_ids(owner, entity_ids, dimension)
        return cls(
            owner.source_id,
            owner.source_revision,
            MeshingEntityKind.GEOMETRY,
            target,
            owner.scope_entity_set_id(target),
            identifiers,
        )

    @classmethod
    def whole(
        cls,
        source: (
            BRepModel
            | BRepSource
            | FixedTopologyBRepSource
            | SurfaceModel
            | CellMesh
            | _SourceEntityScopes
            | LabelFieldSurfaceExtractionResult
        ),
        dimension: int,
        /,
    ) -> MeshingScope:
        """Select all active entities of an explicitly declared dimension."""
        from ..geometry.surface import SurfaceModel

        target = nonnegative_integer(dimension, "dimension")
        if isinstance(source, (SurfaceModel, CellMesh)):
            mesh = source.mesh if isinstance(source, SurfaceModel) else source
            if target > mesh.topological_dimension:
                raise ValueError(
                    "The requested entity dimension is absent from the mesh."
                )
            entities = mesh.entity_set(target)
            return cls(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                target,
                entities.entity_set_id,
                np.asarray(entities.entity_ids)[np.asarray(entities.active_mask)],
            )
        owner = _source_scope_owner(source)
        return cls.from_geometry(owner, owner.scope_entity_ids(target), dimension=target)

    @classmethod
    @checked
    def from_selection(
        cls,
        source_id: str,
        source_revision: str,
        entities: EntitySet,
        selection: EntitySelection,
        /,
    ) -> MeshingScope:
        """Convert a positional selection using its exact owning entity set."""
        if selection.entity_set_id != entities.entity_set_id:
            raise ValueError("Selection must belong to the supplied entity set.")
        mask = np.asarray(selection.mask, dtype=np.bool_)
        if mask.shape != (entities.count,) or not np.array_equal(
            selection.active_mask, entities.active_mask
        ):
            raise ValueError(
                "Selection must match the entity set's capacity and active mask."
            )
        return cls(
            source_id,
            source_revision,
            MeshingEntityKind.MESH,
            entities.intrinsic_dimension,
            entities.entity_set_id,
            np.asarray(entities.entity_ids)[mask],
        )

    @checked
    def _check_compatible(self, other: MeshingScope, /) -> None:
        binding = (
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
        )
        other_binding = (
            other.source_id,
            other.source_revision,
            other.entity_kind,
            other.entity_dimension,
            other.entity_set_id,
        )
        if binding != other_binding:
            raise ValueError("Meshing scope set operations require one exact binding.")

    def union(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        return MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            np.union1d(np.asarray(self.entity_ids), np.asarray(other.entity_ids)),
        )

    def intersection(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        values = np.intersect1d(np.asarray(self.entity_ids), np.asarray(other.entity_ids))
        if values.size == 0:
            raise ValueError("Meshing scope intersection is empty.")
        return MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            values,
        )

    def difference(self, other: MeshingScope, /) -> MeshingScope:
        self._check_compatible(other)
        values = np.setdiff1d(np.asarray(self.entity_ids), np.asarray(other.entity_ids))
        if values.size == 0:
            raise ValueError("Meshing scope difference is empty.")
        return MeshingScope(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            values,
        )

    def __or__(self, other: MeshingScope) -> MeshingScope:
        return self.union(other)

    def __and__(self, other: MeshingScope) -> MeshingScope:
        return self.intersection(other)

    def __sub__(self, other: MeshingScope) -> MeshingScope:
        return self.difference(other)


def resolve_mesh_scope(mesh: CellMesh, scope: MeshingScope, /) -> EntitySelection:
    """Resolve persistent mesh-entity IDs onto one exact mesh topology."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    if scope.source_id != mesh.mesh_id:
        raise ValueError("Meshing scope belongs to another mesh source.")
    if scope.source_revision != mesh.numeric_version:
        raise ValueError("Meshing scope belongs to another mesh revision.")
    if scope.entity_kind is not MeshingEntityKind.MESH:
        raise ValueError("Meshing scope must select mesh entities.")
    if scope.entity_dimension < 0 or scope.entity_dimension > mesh.topological_dimension:
        raise ValueError("Meshing scope entity dimension is not present on the mesh.")
    entities = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entities.entity_set_id:
        raise ValueError("Meshing scope does not match the mesh entity set.")
    entity_ids = np.asarray(entities.entity_ids, dtype=np.int64)
    active = np.asarray(entities.active_mask, dtype=np.bool_)
    requested = np.asarray(scope.entity_ids, dtype=np.int64)
    mask = active & np.isin(entity_ids, requested)
    if np.count_nonzero(mask) != requested.size:
        raise ValueError("Meshing scope contains undeclared mesh entity IDs.")
    return EntitySelection(entities, mask)


class ScopeResolutionReport(StrictModule, NonTrainableState):
    query: str = eqx.field(static=True)
    matched_names: tuple[str, ...] = eqx.field(static=True)
    unmatched_names: tuple[str, ...] = eqx.field(static=True)
    scope: MeshingScope
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        query: str,
        scope: MeshingScope,
        /,
        *,
        matched_names: tuple[str, ...] = (),
        unmatched_names: tuple[str, ...] = (),
    ) -> None:
        expression = str(query).strip()
        if not expression:
            raise ValueError("Scope resolution query must be non-empty.")
        matched = tuple(str(value).strip() for value in matched_names)
        unmatched = tuple(str(value).strip() for value in unmatched_names)
        if any(not value for value in (*matched, *unmatched)):
            raise ValueError("Scope resolution names must be non-empty.")
        self.query = expression
        self.matched_names = matched
        self.unmatched_names = unmatched
        self.scope = scope
        self.report_id = canonical_fingerprint(
            {
                "kind": "scope-resolution-report",
                "query": expression,
                "matched_names": matched,
                "unmatched_names": unmatched,
                "scope": scope.scope_id,
            }
        )


__all__ = [
    "MeshingEntityKind",
    "MeshingScope",
    "ScopeResolutionReport",
    "resolve_mesh_scope",
]
