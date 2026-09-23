#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class AssociationTable(StrictModule, NonTrainableState):
    """Bounded weighted relation between two explicitly named event collections."""

    source_indices: Array
    target_indices: Array
    weights: Array
    active: Array
    valid: Array
    source_collection: str = eqx.field(static=True)
    target_collection: str = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    association_capacity: int = eqx.field(static=True)
    association_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        source_collection: str,
        target_collection: str,
        source_capacity: int,
        target_capacity: int,
    ):
        source = jnp.asarray(source_indices, dtype=jnp.int32)
        target = jnp.asarray(target_indices, dtype=jnp.int32)
        weights_ = jnp.asarray(weights)
        if (
            source.ndim != 2
            or source.shape != target.shape
            or source.shape != weights_.shape
        ):
            raise ValueError(
                "Association arrays must share shape (event_capacity, association_capacity)."
            )
        active_ = (
            jnp.ones(source.shape, dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        if active_.shape != source.shape:
            raise ValueError("active must align with association arrays.")
        if (
            isinstance(source_capacity, bool)
            or not isinstance(source_capacity, Integral)
            or isinstance(target_capacity, bool)
            or not isinstance(target_capacity, Integral)
        ):
            raise TypeError("Collection capacities must be integers.")
        source_capacity_ = int(source_capacity)
        target_capacity_ = int(target_capacity)
        if source_capacity_ < 1 or target_capacity_ < 1:
            raise ValueError("Collection capacities must be positive.")
        source_name = str(source_collection).strip()
        target_name = str(target_collection).strip()
        if not source_name or not target_name:
            raise ValueError("Collection names must be non-empty.")
        valid = (
            jnp.isfinite(weights_)
            & (weights_ >= 0.0)
            & (source >= 0)
            & (source < source_capacity_)
            & (target >= 0)
            & (target < target_capacity_)
        )
        self.source_indices = source
        self.target_indices = target
        self.weights = jnp.where(active_, weights_, 0.0)
        self.active = active_
        self.valid = jnp.where(active_, valid, True)
        self.source_collection = source_name
        self.target_collection = target_name
        self.source_capacity = source_capacity_
        self.target_capacity = target_capacity_
        self.association_capacity = source.shape[1]
        self.association_id = canonical_fingerprint(
            {
                "kind": "hep-association-table",
                "source": source_name,
                "target": target_name,
                "source_capacity": source_capacity_,
                "target_capacity": target_capacity_,
                "association_capacity": self.association_capacity,
                "content": array_tree_fingerprint(
                    {
                        "source_indices": source,
                        "target_indices": target,
                        "weights": self.weights,
                        "active": active_,
                        "valid": self.valid,
                    }
                ),
            }
        )

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid)


__all__ = ["AssociationTable"]
