#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._resources import TopologyResourceError


class PersistenceDiagram(StrictModule, NonTrainableState):
    """Natural-sized persistence intervals with explicit essential-bar semantics."""

    degrees: Array
    birth_values: Array
    death_values: Array
    birth_entity_ids: Array
    death_entity_ids: Array
    has_finite_death: Array
    pairing_indices: Array
    source_id: str = eqx.field(static=True)
    diagram_id: str = eqx.field(static=True)

    def __init__(
        self,
        degrees: ArrayLike,
        birth_values: ArrayLike,
        death_values: ArrayLike,
        birth_entity_ids: ArrayLike,
        death_entity_ids: ArrayLike,
        has_finite_death: ArrayLike,
        pairing_indices: ArrayLike,
        /,
        *,
        source_id: str,
    ):
        degree_array = np.asarray(degrees, dtype=np.int32)
        births = np.asarray(birth_values)
        deaths = np.asarray(death_values)
        birth_ids = np.asarray(birth_entity_ids, dtype=np.int64)
        death_ids = np.asarray(death_entity_ids, dtype=np.int64)
        finite = np.asarray(has_finite_death, dtype=bool)
        indices = np.asarray(pairing_indices, dtype=np.int32)
        arrays = (degree_array, births, deaths, birth_ids, death_ids, finite, indices)
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("Persistence diagram arrays must be rank-1.")
        if len({value.shape for value in arrays}) != 1:
            raise ValueError("Persistence diagram arrays must have identical shapes.")
        if np.any(degree_array < 0):
            raise ValueError("Persistence homological degrees must be non-negative.")
        if np.any(~np.isfinite(births)) or np.any(~np.isfinite(deaths[finite])):
            raise ValueError("Active persistence endpoint values must be finite.")
        if np.any(birth_ids < 0) or np.any(death_ids[finite] < 0):
            raise ValueError("Active persistence entity IDs must be non-negative.")
        source = str(source_id)
        if not source:
            raise ValueError("Persistence diagram source_id must be non-empty.")
        self.degrees = jnp.asarray(degree_array)
        self.birth_values = jnp.asarray(births)
        self.death_values = jnp.asarray(deaths)
        self.birth_entity_ids = jnp.asarray(birth_ids)
        self.death_entity_ids = jnp.asarray(death_ids)
        self.has_finite_death = jnp.asarray(finite)
        self.pairing_indices = jnp.asarray(indices)
        self.source_id = source
        self.diagram_id = canonical_fingerprint(
            {
                "kind": "persistence-diagram",
                "source": source,
                "degrees": array_tree_fingerprint(degree_array),
                "births": array_tree_fingerprint(births),
                "deaths": array_tree_fingerprint(deaths),
                "birth_entity_ids": array_tree_fingerprint(birth_ids),
                "death_entity_ids": array_tree_fingerprint(death_ids),
                "finite": array_tree_fingerprint(finite),
                "pairing_indices": array_tree_fingerprint(indices),
            }
        )

    @property
    def interval_count(self) -> int:
        return int(self.degrees.shape[0])

    @property
    def essential_count(self) -> int:
        return int(np.count_nonzero(~np.asarray(self.has_finite_death)))


class PackedPersistenceDiagram(StrictModule, NonTrainableState):
    """Fixed-capacity JAX persistence intervals with inert padded slots."""

    active_mask: Array
    degrees: Array
    birth_values: Array
    death_values: Array
    birth_entity_ids: Array
    death_entity_ids: Array
    has_finite_death: Array
    pairing_indices: Array
    capacity: int = eqx.field(static=True)
    source_diagram_id: str = eqx.field(static=True)
    packed_id: str = eqx.field(static=True)

    def __init__(
        self,
        diagram: PersistenceDiagram,
        capacity: int,
        /,
    ):
        if not isinstance(diagram, PersistenceDiagram):
            raise TypeError("Packing requires a PersistenceDiagram.")
        size = int(capacity)
        if size <= 0:
            raise ValueError("Packed persistence capacity must be positive.")
        if diagram.interval_count > size:
            raise TopologyResourceError(
                "Persistence diagram exceeds the requested packed capacity."
            )
        count = diagram.interval_count
        active = np.arange(size) < count

        def pad(value: Array, *, dtype=None):
            source = np.asarray(value, dtype=dtype)
            output = np.zeros((size,), dtype=source.dtype)
            output[:count] = source
            return jnp.asarray(output)

        self.active_mask = jnp.asarray(active)
        self.degrees = pad(diagram.degrees, dtype=np.int32)
        self.birth_values = pad(diagram.birth_values)
        self.death_values = pad(diagram.death_values)
        self.birth_entity_ids = pad(diagram.birth_entity_ids, dtype=np.int64)
        self.death_entity_ids = pad(diagram.death_entity_ids, dtype=np.int64)
        self.has_finite_death = pad(diagram.has_finite_death, dtype=bool)
        self.pairing_indices = pad(diagram.pairing_indices, dtype=np.int32)
        self.capacity = size
        self.source_diagram_id = diagram.diagram_id
        self.packed_id = canonical_fingerprint(
            {
                "kind": "packed-persistence-diagram",
                "diagram": diagram.diagram_id,
                "capacity": size,
            }
        )

    @property
    def interval_count(self) -> Array:
        return jnp.sum(self.active_mask.astype(jnp.int32))


__all__ = ["PackedPersistenceDiagram", "PersistenceDiagram"]
