#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class _BoundedVectorDomain(StrictModule):
    """Validated affine map between a physical vector box and the unit cube."""

    initial: Array
    lower: Array
    upper: Array
    affine_scale: Array
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        initial_vector: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        /,
    ):
        initial = np.asarray(initial_vector)
        lower = np.asarray(lower_bounds)
        upper = np.asarray(upper_bounds)
        if initial.ndim != 1:
            raise ValueError("initial_vector must be one-dimensional.")
        if initial.size == 0:
            raise ValueError("Bounded search requires at least one dimension.")
        if lower.shape != initial.shape or upper.shape != initial.shape:
            raise ValueError(
                "initial_vector, lower_bounds, and upper_bounds must have identical shapes."
            )
        dtype = np.result_type(initial.dtype, lower.dtype, upper.dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Bounded-search vectors and bounds must be real-valued.")
        initial = initial.astype(dtype, copy=False)
        lower = lower.astype(dtype, copy=False)
        upper = upper.astype(dtype, copy=False)
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_vector must be finite.")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Bounded-search bounds must be finite.")
        if np.any(lower >= upper):
            raise ValueError("Every lower bound must be smaller than its upper bound.")
        if np.any((initial < lower) | (initial > upper)):
            raise ValueError("initial_vector lies outside the search bounds.")
        scale = np.maximum(
            np.maximum(np.abs(lower), np.abs(upper)),
            1.0,
        )
        self.initial = jnp.asarray(initial)
        self.lower = jnp.asarray(lower)
        self.upper = jnp.asarray(upper)
        self.affine_scale = jnp.asarray(scale)
        self.dimension = int(initial.size)

    def to_unit(self, physical: ArrayLike, /) -> Array:
        value = jnp.asarray(physical, dtype=self.initial.dtype)
        if value.shape[-1:] != (self.dimension,):
            raise ValueError(
                "Physical search vectors must have trailing shape "
                f"({self.dimension},), got {value.shape}."
            )
        scale = self.affine_scale
        lower = self.lower / scale
        upper = self.upper / scale
        return (value / scale - lower) / (upper - lower)

    def from_unit(self, unit: ArrayLike, /) -> Array:
        value = jnp.asarray(unit, dtype=self.initial.dtype)
        if value.shape[-1:] != (self.dimension,):
            raise ValueError(
                "Unit search vectors must have trailing shape "
                f"({self.dimension},), got {value.shape}."
            )
        scale = self.affine_scale
        lower = self.lower / scale
        upper = self.upper / scale
        return scale * ((1.0 - value) * lower + value * upper)


__all__: list[str] = []
