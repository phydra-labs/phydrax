#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class ShallowWaterCoriolisSource(StrictModule, NonTrainableState):
    """Identified f- or beta-plane Coriolis source for two-dimensional flow."""

    f0: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    meridional_axis: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        f0: float,
        /,
        *,
        beta: float = 0.0,
        meridional_axis: int = 1,
    ):
        f0_ = float(f0)
        beta_ = float(beta)
        axis = int(meridional_axis)
        if not np.isfinite(f0_) or not np.isfinite(beta_):
            raise ValueError("Coriolis parameters must be finite.")
        if axis not in (0, 1):
            raise ValueError("Coriolis meridional_axis must be zero or one.")
        self.f0 = f0_
        self.beta = beta_
        self.meridional_axis = axis
        self.source_id = canonical_fingerprint(
            {
                "kind": "shallow-water-coriolis",
                "f0": f0_,
                "beta": beta_,
                "meridional_axis": axis,
                "orientation": "du=fv,dv=-fu",
            }
        )

    def parameter(self, coordinates: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        if points.ndim == 0 or points.shape[-1] != 2:
            raise ValueError("Coriolis coordinates must have two spatial components.")
        return self.f0 + self.beta * points[..., self.meridional_axis]

    def __call__(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        args: Any = None,
    ) -> Array:
        del time, args
        value = jnp.asarray(state)
        if value.ndim == 0 or value.shape[-1] != 3:
            raise ValueError("Coriolis shallow water requires state (h, hu, hv).")
        coriolis = self.parameter(coordinates).astype(value.dtype)
        source = jnp.zeros_like(value)
        source = source.at[..., 1].set(coriolis * value[..., 2])
        return source.at[..., 2].set(-coriolis * value[..., 1])

    def stable_step(
        self,
        coordinates: ArrayLike,
        /,
        *,
        safety: float = 1.0,
    ) -> Array:
        safety_ = float(safety)
        if not np.isfinite(safety_) or not 0.0 < safety_ <= 1.0:
            raise ValueError("Coriolis stability safety must lie in (0, 1].")
        maximum = jnp.max(jnp.abs(self.parameter(coordinates)))
        return jnp.where(
            maximum > 0.0,
            safety_ * math.sqrt(3.0) / maximum,
            jnp.inf,
        )


__all__ = ["ShallowWaterCoriolisSource"]
