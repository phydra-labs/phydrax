#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import EntitySelection, EntitySet


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
        entity_ids: ArrayLike,
        /,
    ):
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
    def from_selection(
        cls,
        source_id: str,
        source_revision: str,
        entities: EntitySet,
        selection: EntitySelection,
        /,
    ) -> MeshingScope:
        """Convert a positional selection using its exact owning entity set."""
        if not isinstance(entities, EntitySet):
            raise TypeError("entities must be EntitySet.")
        if not isinstance(selection, EntitySelection):
            raise TypeError("selection must be EntitySelection.")
        if selection.entity_set_id != entities.entity_set_id:
            raise ValueError("Selection must belong to the supplied entity set.")
        mask = np.asarray(selection.mask, dtype=bool)
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

    def _check_compatible(self, other: MeshingScope, /) -> None:
        if not isinstance(other, MeshingScope):
            raise TypeError("Scope set operations require MeshingScope values.")
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


class ScopeResolutionReport(StrictModule, NonTrainableState):
    query: str = eqx.field(static=True)
    matched_names: tuple[str, ...] = eqx.field(static=True)
    unmatched_names: tuple[str, ...] = eqx.field(static=True)
    scope: MeshingScope
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        query: str,
        scope: MeshingScope,
        /,
        *,
        matched_names: tuple[str, ...] = (),
        unmatched_names: tuple[str, ...] = (),
    ):
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
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
]
