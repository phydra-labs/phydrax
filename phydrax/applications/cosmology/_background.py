#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract


class FLRWBackground(StrictModule):
    """Differentiable flat radiation, matter, and cosmological-constant background."""

    hubble_constant: Array
    matter_density: Array
    radiation_density: Array
    dark_energy_density: Array
    scale: CosmologyScaleContract
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        hubble_constant: ArrayLike,
        matter_density: ArrayLike,
        /,
        *,
        radiation_density: ArrayLike = 0.0,
        dark_energy_density: ArrayLike | None = None,
        scale: CosmologyScaleContract = CODE_COSMOLOGY_SCALE,
    ):
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be a CosmologyScaleContract.")
        dtype = jnp.result_type(
            hubble_constant,
            matter_density,
            radiation_density,
            0.0 if dark_energy_density is None else dark_energy_density,
        )
        h0 = jnp.asarray(hubble_constant, dtype=dtype)
        matter = jnp.asarray(matter_density, dtype=dtype)
        radiation = jnp.asarray(radiation_density, dtype=dtype)
        dark = (
            1.0 - matter - radiation
            if dark_energy_density is None
            else jnp.asarray(dark_energy_density, dtype=dtype)
        )
        if any(value.shape != () for value in (h0, matter, radiation, dark)):
            raise ValueError("FLRW background parameters must be scalar.")
        h0 = eqx.error_if(
            h0,
            ~jnp.isfinite(h0)
            | (h0 <= 0.0)
            | ~jnp.isfinite(matter)
            | ~jnp.isfinite(radiation)
            | ~jnp.isfinite(dark)
            | (matter < 0.0)
            | (radiation < 0.0)
            | (dark < 0.0)
            | (jnp.abs(matter + radiation + dark - 1.0) > 1.0e-10),
            "FLRW background requires finite flat density parameters.",
        )
        self.hubble_constant = h0
        self.matter_density = matter
        self.radiation_density = radiation
        self.dark_energy_density = dark
        self.scale = scale
        self.model_id = canonical_fingerprint(
            {
                "kind": "flat-flrw-background-model",
                "scale": scale.scale_id,
            }
        )

    def expansion_squared(self, scale_factor: ArrayLike, /) -> Array:
        """Return E(a)^2 = H(a)^2 / H0^2."""
        scale = jnp.asarray(scale_factor, dtype=self.hubble_constant.dtype)
        scale = eqx.error_if(
            scale,
            jnp.any(~jnp.isfinite(scale)) | jnp.any(scale <= 0.0),
            "Scale factor must be finite and positive.",
        )
        return (
            self.radiation_density / scale**4
            + self.matter_density / scale**3
            + self.dark_energy_density
        )

    def hubble(self, scale_factor: ArrayLike, /) -> Array:
        return self.hubble_constant * jnp.sqrt(self.expansion_squared(scale_factor))

    def matter_fraction(self, scale_factor: ArrayLike, /) -> Array:
        scale = jnp.asarray(scale_factor, dtype=self.hubble_constant.dtype)
        return self.matter_density / scale**3 / self.expansion_squared(scale)

    def radiation_fraction(self, scale_factor: ArrayLike, /) -> Array:
        scale = jnp.asarray(scale_factor, dtype=self.hubble_constant.dtype)
        return self.radiation_density / scale**4 / self.expansion_squared(scale)

    def dark_energy_fraction(self, scale_factor: ArrayLike, /) -> Array:
        return self.dark_energy_density / self.expansion_squared(scale_factor)

    def dlog_hubble_dlog_scale(self, scale_factor: ArrayLike, /) -> Array:
        scale = jnp.asarray(scale_factor, dtype=self.hubble_constant.dtype)
        expansion_squared = self.expansion_squared(scale)
        derivative = (
            -4.0 * self.radiation_density / scale**4
            - 3.0 * self.matter_density / scale**3
        )
        return 0.5 * derivative / expansion_squared

    def drift_factor(self, start: ArrayLike, end: ArrayLike, /) -> Array:
        """Midpoint integral for dx/da = p / (m a^3 H(a))."""
        start_ = jnp.asarray(start, dtype=self.hubble_constant.dtype)
        end_ = jnp.asarray(end, dtype=self.hubble_constant.dtype)
        midpoint = 0.5 * (start_ + end_)
        return (end_ - start_) / (midpoint**3 * self.hubble(midpoint))

    def kick_factor(self, start: ArrayLike, end: ArrayLike, /) -> Array:
        """Midpoint integral for dp/da = m g_psi / (a^2 H(a))."""
        start_ = jnp.asarray(start, dtype=self.hubble_constant.dtype)
        end_ = jnp.asarray(end, dtype=self.hubble_constant.dtype)
        midpoint = 0.5 * (start_ + end_)
        return (end_ - start_) / (midpoint**2 * self.hubble(midpoint))


__all__ = ["FLRWBackground"]
