#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class StatePartition(StrictModule, NonTrainableState):
    """Named, disjoint, complete static masks over one array-valued state."""

    masks: tuple[Array, ...]
    names: tuple[str, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        masks: Mapping[str, ArrayLike],
        /,
        *,
        partition_id: str | None = None,
    ):
        if not isinstance(masks, Mapping) or len(masks) < 2:
            raise TypeError("masks must map at least two names to masks.")
        names = tuple(str(name) for name in masks)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Partition names must be unique and non-empty.")
        arrays = tuple(np.asarray(value, dtype=bool) for value in masks.values())
        shape = tuple(int(size) for size in arrays[0].shape)
        if any(array.shape != shape for array in arrays):
            raise ValueError("Every state-partition mask must share one shape.")
        if any(not np.any(array) for array in arrays):
            raise ValueError("Every state partition must contain at least one scalar.")
        occupancy = np.sum(np.stack(arrays, axis=0), axis=0)
        if np.any(occupancy != 1):
            raise ValueError("State partitions must be disjoint and complete.")
        payload = {
            "kind": "state-partition",
            "names": list(names),
            "masks": array_tree_fingerprint(arrays),
        }
        identifier = (
            f"state-partition:{canonical_fingerprint(payload)}"
            if partition_id is None
            else str(partition_id)
        )
        if not identifier:
            raise ValueError("partition_id must be non-empty or None.")
        self.masks = tuple(jnp.asarray(array) for array in arrays)
        self.names = names
        self.state_shape = shape
        self.partition_id = identifier

    def mask(self, name: str, /) -> Array:
        for candidate, mask in zip(self.names, self.masks, strict=True):
            if candidate == name:
                return mask
        raise KeyError(f"Unknown state partition {name!r}.")

    def project(self, name: str, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != self.state_shape:
            raise ValueError("Partition projection value has the wrong state shape.")
        return jnp.where(self.mask(name), array, 0)


__all__ = ["StatePartition"]
