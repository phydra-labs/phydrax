#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState


class SphereLaplacianLevels(StrictModule, NonTrainableState):
    """Complete unit-sphere Laplacian levels with exact multiplicities."""

    levels: Array
    eigenvalues: Array
    multiplicities: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    max_level: int = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, max_level: int, /):
        resolved_dimension = int(dimension)
        resolved_level = int(max_level)
        if resolved_dimension < 1:
            raise ValueError("Sphere dimension must be at least one.")
        if resolved_level < 0:
            raise ValueError("max_level must be nonnegative.")
        levels = tuple(range(resolved_level + 1))
        eigenvalues = tuple(level * (level + resolved_dimension - 1) for level in levels)
        multiplicities = tuple(
            math.comb(level + resolved_dimension, resolved_dimension)
            - (
                math.comb(level + resolved_dimension - 2, resolved_dimension)
                if level >= 2
                else 0
            )
            for level in levels
        )
        self.levels = jnp.asarray(levels, dtype=jnp.int32)
        self.eigenvalues = jnp.asarray(eigenvalues, dtype=float)
        self.multiplicities = multiplicities
        self.dimension = resolved_dimension
        self.max_level = resolved_level
        self.spectrum_id = f"sphere-S{resolved_dimension}-levels-0:{resolved_level}"

    @property
    def mode_count(self) -> int:
        return sum(self.multiplicities)


__all__ = ["SphereLaplacianLevels"]
