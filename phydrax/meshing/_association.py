#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import EntitySet


class GeometryAssociationKind(StrEnum):
    BREP = "brep"
    SURFACE = "surface"
    IMPLICIT = "implicit"


class GeometryAssociation(StrictModule, NonTrainableState):
    """Audited map from exact mesh entities to authoritative geometry entities."""

    association_kind: GeometryAssociationKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    target_global_ids: Array
    source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    residuals: Array
    resolved: Array
    ambiguous: Array
    exact: bool = eqx.field(static=True)
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
    ):
        if not isinstance(association_kind, GeometryAssociationKind):
            raise TypeError("association_kind must be GeometryAssociationKind.")
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
        sources = tuple(str(value).strip() for value in source_entity_ids)
        distances = np.asarray(residuals, dtype=float)
        if len(sources) != targets.size or any(not value for value in sources):
            raise ValueError(
                "Source geometry IDs must match target entities and be non-empty."
            )
        if (
            distances.shape != targets.shape
            or np.any(~np.isfinite(distances))
            or np.any(distances < 0)
        ):
            raise ValueError(
                "Association residuals must be finite, non-negative, and aligned."
            )
        resolved_ = (
            np.ones(targets.shape, dtype=bool)
            if resolved is None
            else np.asarray(resolved, dtype=bool)
        )
        ambiguous_ = (
            np.zeros(targets.shape, dtype=bool)
            if ambiguous is None
            else np.asarray(ambiguous, dtype=bool)
        )
        if resolved_.shape != targets.shape or ambiguous_.shape != targets.shape:
            raise ValueError("Association status masks must match target entities.")
        if np.any(resolved_ & ambiguous_):
            raise ValueError("Resolved geometry associations cannot be ambiguous.")
        if exact and (
            not np.all(resolved_) or np.any(ambiguous_) or np.any(distances != 0.0)
        ):
            raise ValueError(
                "Exact geometry associations require zero-residual unique coverage."
            )
        self.association_kind = association_kind
        self.source_id = source
        self.source_revision = revision
        self.target_entity_set_id = entity_set
        self.target_global_ids = jnp.asarray(targets)
        self.source_entity_ids = sources
        self.residuals = jnp.asarray(distances)
        self.resolved = jnp.asarray(resolved_)
        self.ambiguous = jnp.asarray(ambiguous_)
        self.exact = bool(exact)
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
                "exact": bool(exact),
            }
        )

    @property
    def complete(self) -> bool:
        return bool(np.all(np.asarray(self.resolved))) and not bool(
            np.any(np.asarray(self.ambiguous))
        )

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


__all__ = ["GeometryAssociation", "GeometryAssociationKind"]
