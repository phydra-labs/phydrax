#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FLRWBackground(StrictModule, NonTrainableState):
    hubble_constant: float = eqx.field(static=True)
    matter_density: float = eqx.field(static=True)
    radiation_density: float = eqx.field(static=True)
    dark_energy_density: float = eqx.field(static=True)
    background_id: str = eqx.field(static=True)

    def __init__(
        self,
        hubble_constant: float,
        matter_density: float,
        /,
        *,
        radiation_density: float = 0.0,
        dark_energy_density: float | None = None,
    ):
        h0 = float(hubble_constant)
        matter = float(matter_density)
        radiation = float(radiation_density)
        dark = (
            1.0 - matter - radiation
            if dark_energy_density is None
            else float(dark_energy_density)
        )
        if (
            not np.isfinite(h0)
            or h0 <= 0.0
            or any(
                not np.isfinite(value) or value < 0.0
                for value in (matter, radiation, dark)
            )
            or abs(matter + radiation + dark - 1.0) > 1e-10
        ):
            raise ValueError("FLRW background requires finite flat density parameters.")
        self.hubble_constant = h0
        self.matter_density = matter
        self.radiation_density = radiation
        self.dark_energy_density = dark
        self.background_id = canonical_fingerprint(
            {
                "kind": "flat-flrw-background",
                "hubble_constant": h0,
                "matter_density": matter,
                "radiation_density": radiation,
                "dark_energy_density": dark,
            }
        )

    def hubble(self, scale_factor: ArrayLike, /) -> Array:
        scale = jnp.asarray(scale_factor)
        scale = eqx.error_if(
            scale,
            jnp.any(~jnp.isfinite(scale)) | jnp.any(scale <= 0.0),
            "Scale factor must be finite and positive.",
        )
        return self.hubble_constant * jnp.sqrt(
            self.radiation_density / scale**4
            + self.matter_density / scale**3
            + self.dark_energy_density
        )

    def drift_factor(self, start: ArrayLike, end: ArrayLike, /) -> Array:
        midpoint = 0.5 * (jnp.asarray(start) + jnp.asarray(end))
        return (jnp.asarray(end) - jnp.asarray(start)) / (
            midpoint**3 * self.hubble(midpoint)
        )

    def kick_factor(self, start: ArrayLike, end: ArrayLike, /) -> Array:
        midpoint = 0.5 * (jnp.asarray(start) + jnp.asarray(end))
        return (jnp.asarray(end) - jnp.asarray(start)) / (
            midpoint**2 * self.hubble(midpoint)
        )


__all__ = ["FLRWBackground"]
