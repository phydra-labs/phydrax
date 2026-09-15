#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FixedGridGeometry(StrictModule, NonTrainableState):
    """Uniform nodal Cartesian topology for one fixed-grid Z4c evolution."""

    shape: tuple[int, int, int] = eqx.field(static=True)
    lower: tuple[float, float, float] = eqx.field(static=True)
    spacing: tuple[float, float, float] = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, int, int],
        lower: tuple[float, float, float],
        spacing: tuple[float, float, float],
        /,
        *,
        periodic: bool,
    ):
        shape_ = tuple(int(value) for value in shape)
        lower_ = tuple(float(value) for value in lower)
        spacing_ = tuple(float(value) for value in spacing)
        if len(shape_) != 3 or any(value < 5 for value in shape_):
            raise ValueError("shape must contain three extents of at least five points.")
        if len(lower_) != 3 or any(not isfinite(value) for value in lower_):
            raise ValueError("lower must contain three finite coordinates.")
        if len(spacing_) != 3 or any(
            not isfinite(value) or value <= 0.0 for value in spacing_
        ):
            raise ValueError("spacing must contain three finite positive values.")
        if not isinstance(periodic, bool):
            raise TypeError("periodic must be Boolean.")
        self.shape = shape_
        self.lower = lower_
        self.spacing = spacing_
        self.periodic = periodic
        self.grid_id = canonical_fingerprint(
            {
                "kind": "fixed-cartesian-z4c-grid",
                "shape": list(shape_),
                "lower": list(lower_),
                "spacing": list(spacing_),
                "periodic": periodic,
            }
        )

    @property
    def upper(self) -> tuple[float, float, float]:
        return tuple(
            self.lower[i] + self.spacing[i] * (self.shape[i] - (1 if self.periodic else 1))
            for i in range(3)
        )

    @property
    def coordinates(self) -> Array:
        axes = tuple(
            jnp.asarray(self.lower[i])
            + jnp.asarray(self.spacing[i]) * jnp.arange(self.shape[i])
            for i in range(3)
        )
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=0)

    @property
    def cell_volume(self) -> float:
        return self.spacing[0] * self.spacing[1] * self.spacing[2]

    def validate_field(self, field: ArrayLike, /) -> Array:
        value = jnp.asarray(field)
        if value.shape[-3:] != self.shape:
            raise ValueError(
                f"field trailing shape must be {self.shape}; got {value.shape[-3:]}."
            )
        return value

    def boundary_mask(self, width: int = 1, /) -> Array:
        """Return the static outer shell mask, or all-false for periodic topology."""

        width_ = int(width)
        if width_ < 1 or 2 * width_ >= min(self.shape):
            raise ValueError("boundary width leaves no interior points.")
        if self.periodic:
            return jnp.zeros(self.shape, dtype=bool)
        indices = jnp.meshgrid(
            *(jnp.arange(extent) for extent in self.shape), indexing="ij"
        )
        mask = jnp.zeros(self.shape, dtype=bool)
        for axis, extent in enumerate(self.shape):
            mask = mask | (indices[axis] < width_) | (
                indices[axis] >= extent - width_
            )
        return mask

    def interior_mask(self, width: int = 1, /) -> Array:
        return ~self.boundary_mask(width)

    def radial_geometry(
        self, center: ArrayLike = (0.0, 0.0, 0.0), /
    ) -> tuple[Array, Array]:
        """Return radius and outward radial unit vectors on the full grid."""

        center_ = jnp.asarray(center)
        if center_.shape != (3,):
            raise ValueError("center must have shape (3,).")
        displacement = self.coordinates - center_[:, None, None, None]
        radius = jnp.sqrt(jnp.sum(displacement * displacement, axis=0))
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        return radius, displacement / safe_radius[None, ...]


__all__ = ["FixedGridGeometry"]
