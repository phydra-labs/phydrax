# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Capacity storage is not a Hilbert space: native spaces contain active rows only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import AbstractVectorSpace
from ...typing import Dim, Int32
from .._gram import diagonal_gram_space


class ActivePointDim(Dim):
    """Compact active meshfree coordinates."""


class CapacityPointDim(Dim):
    """Storage slots, including inactive slots."""


@final
class MeshfreeCapacityMap(StrictModule):
    __strict_contract__ = True
    active_indices: Int32[ActivePointDim]
    inverse_indices: Int32[CapacityPointDim]
    capacity: int = eqx.field(static=True)
    active_count: int = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(self, capacity: int, active_indices: ArrayLike, /) -> None:
        indices = np.asarray(active_indices)
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, (int, np.integer))
            or capacity < 1
        ):
            raise ValueError("Capacity must be positive.")
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Active indices must be an integer vector.")
        if (
            indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= capacity)
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(
                "Active indices must be nonempty, increasing, unique and in capacity."
            )
        inverse = np.full(capacity, -1, dtype=np.int32)
        inverse[indices] = np.arange(indices.size, dtype=np.int32)
        self.active_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.inverse_indices = jnp.asarray(inverse)
        self.capacity, self.active_count = int(capacity), int(indices.size)
        self.mapping_id = canonical_fingerprint(
            {"kind": "meshfree-capacity", "capacity": capacity, "active": indices}
        )

    def compact(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[0] != self.capacity:
            raise ValueError("Storage values must have one row per capacity slot.")
        return array[self.active_indices]

    def expand(self, values: ArrayLike, /, *, fill: float = 0.0) -> Array:
        array = jnp.asarray(values)
        if array.shape[0] != self.active_count:
            raise ValueError("Compact values must have one row per active point.")
        return (
            jnp.full((self.capacity, *array.shape[1:]), fill, dtype=array.dtype)
            .at[self.active_indices]
            .set(array)
        )

    def positive_space(self, compact_measures: ArrayLike, /) -> AbstractVectorSpace:
        weights = np.asarray(compact_measures, dtype=np.float64)
        if weights.shape != (self.active_count,) or not np.all(
            np.isfinite(weights) & (weights > 0)
        ):
            raise ValueError(
                "Native active spaces require strictly positive compact measures."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "meshfree-active-space",
                "mapping": self.mapping_id,
                "measures": weights,
            }
        )
        space, _mass = diagonal_gram_space(weights, dtype=np.float64, space_id=identifier)
        return space


@final
class MeshfreeCapacityPolicy(StrictModule):
    buckets: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, buckets: Sequence[int], /) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in buckets
        ):
            raise TypeError("Capacity buckets must be integers.")
        values = tuple(int(value) for value in buckets)
        if (
            not values
            or values[0] < 1
            or any(right <= left for left, right in zip(values, values[1:]))
        ):
            raise ValueError("Capacity buckets must be positive and strictly increasing.")
        self.buckets = values

    def allocate(self, active_count: int, /) -> MeshfreeCapacityMap:
        if (
            isinstance(active_count, bool)
            or not isinstance(active_count, (int, np.integer))
            or active_count < 1
        ):
            raise ValueError("An active surface cannot be empty.")
        capacity = next((size for size in self.buckets if size >= active_count), None)
        if capacity is None:
            raise ValueError("Meshfree capacity exhausted; no declared bucket fits.")
        return MeshfreeCapacityMap(capacity, np.arange(active_count, dtype=np.int32))


__all__ = ["MeshfreeCapacityMap", "MeshfreeCapacityPolicy"]
