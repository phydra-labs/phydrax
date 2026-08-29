#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CausalTimeSlabSchedule(StrictModule, NonTrainableState):
    """Ordered time slabs with explicit overlap and causal loss weights."""

    boundaries: Array
    overlap_fraction: float
    causal_strength: float
    schedule_id: str

    def __init__(
        self,
        boundaries: Sequence[float] | ArrayLike,
        /,
        *,
        overlap_fraction: float = 0.0,
        causal_strength: float = 1.0,
        schedule_id: str = "causal-time-slabs",
    ):
        values = np.asarray(boundaries, dtype=float)
        if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
            raise ValueError("boundaries must contain at least two finite times.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("Time-slab boundaries must be strictly increasing.")
        overlap = float(overlap_fraction)
        strength = float(causal_strength)
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap_fraction must lie in [0, 1).")
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("causal_strength must be finite and nonnegative.")
        identifier = str(schedule_id)
        if not identifier:
            raise ValueError("schedule_id must be non-empty.")
        self.boundaries = jnp.asarray(values)
        self.overlap_fraction = overlap
        self.causal_strength = strength
        self.schedule_id = identifier

    @property
    def slab_count(self) -> int:
        return int(self.boundaries.size) - 1

    def bounds(self, index: int, /) -> tuple[Array, Array]:
        position = int(index)
        if position < 0 or position >= self.slab_count:
            raise IndexError("Time-slab index is out of range.")
        lower = self.boundaries[position]
        upper = self.boundaries[position + 1]
        if position > 0 and self.overlap_fraction > 0.0:
            lower = lower - self.overlap_fraction * (upper - lower)
        return lower, upper

    def active(self, times: ArrayLike, index: int, /) -> Array:
        """Return the closed-interval membership mask for one slab."""

        values = jnp.asarray(times)
        lower, upper = self.bounds(index)
        return (values >= lower) & (values <= upper)

    def local_coordinate(self, times: ArrayLike, index: int, /) -> Array:
        """Map one slab to [0, 1], clipping overlap and out-of-slab values."""

        values = jnp.asarray(times)
        lower, upper = self.bounds(index)
        return jnp.clip((values - lower) / (upper - lower), 0.0, 1.0)

    def causal_weights(self, slab_losses: ArrayLike, /) -> Array:
        """Exponentially downweight a slab by all preceding detached losses."""

        losses = jnp.asarray(slab_losses, dtype=float)
        if losses.ndim < 1 or int(losses.shape[-1]) != self.slab_count:
            raise ValueError(f"slab_losses must end in {self.slab_count} slab values.")
        losses = jnp.maximum(losses, 0.0)
        preceding = jnp.concatenate(
            (jnp.zeros_like(losses[..., :1]), jnp.cumsum(losses[..., :-1], axis=-1)),
            axis=-1,
        )
        return jax.lax.stop_gradient(jnp.exp(-self.causal_strength * preceding))


__all__ = ["CausalTimeSlabSchedule"]
