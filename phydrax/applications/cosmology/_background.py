#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._closure import CosmologyPhysicalState, PhysicalDependencyProjection
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract


class FLRWBackground(StrictModule):
    """Differentiable radiation, matter, curvature, and CPL dark-energy FLRW model."""

    hubble_constant: Array
    matter_density: Array
    radiation_density: Array
    curvature_density: Array
    dark_energy_density: Array
    dark_energy_w0: Array
    dark_energy_wa: Array
    scale: CosmologyScaleContract
    model_form_id: str = eqx.field(static=True)

    def __init__(
        self,
        hubble_constant: ArrayLike,
        matter_density: ArrayLike,
        /,
        *,
        radiation_density: ArrayLike = 0.0,
        curvature_density: ArrayLike = 0.0,
        dark_energy_density: ArrayLike | None = None,
        dark_energy_w0: ArrayLike = -1.0,
        dark_energy_wa: ArrayLike = 0.0,
        scale: CosmologyScaleContract = CODE_COSMOLOGY_SCALE,
    ):
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be a CosmologyScaleContract.")
        dtype = jnp.result_type(
            hubble_constant,
            matter_density,
            radiation_density,
            curvature_density,
            0.0 if dark_energy_density is None else dark_energy_density,
            dark_energy_w0,
            dark_energy_wa,
        )
        h0 = jnp.asarray(hubble_constant, dtype=dtype)
        matter = jnp.asarray(matter_density, dtype=dtype)
        radiation = jnp.asarray(radiation_density, dtype=dtype)
        curvature = jnp.asarray(curvature_density, dtype=dtype)
        w0 = jnp.asarray(dark_energy_w0, dtype=dtype)
        wa = jnp.asarray(dark_energy_wa, dtype=dtype)
        dark = (
            1.0 - matter - radiation - curvature
            if dark_energy_density is None
            else jnp.asarray(dark_energy_density, dtype=dtype)
        )
        parameters = (h0, matter, radiation, curvature, dark, w0, wa)
        if any(value.shape != () for value in parameters):
            raise ValueError("FLRW background parameters must be scalar.")
        h0 = eqx.error_if(
            h0,
            jnp.any(~jnp.isfinite(jnp.stack(parameters)))
            | (h0 <= 0.0)
            | (matter < 0.0)
            | (radiation < 0.0)
            | (dark < 0.0)
            | (jnp.abs(matter + radiation + curvature + dark - 1.0) > 1.0e-10),
            "FLRW background requires finite densities satisfying closure.",
        )
        self.hubble_constant = h0
        self.matter_density = matter
        self.radiation_density = radiation
        self.curvature_density = curvature
        self.dark_energy_density = dark
        self.dark_energy_w0 = w0
        self.dark_energy_wa = wa
        self.scale = scale
        self.model_form_id = canonical_fingerprint(
            {
                "kind": "flrw-radiation-matter-curvature-cpl",
                "scale": scale.scale_id,
            }
        )

    @property
    def physical_state(self) -> CosmologyPhysicalState:
        return CosmologyPhysicalState(
            jnp.stack(
                (
                    self.hubble_constant,
                    self.matter_density,
                    self.radiation_density,
                    self.curvature_density,
                    self.dark_energy_density,
                    self.dark_energy_w0,
                    self.dark_energy_wa,
                )
            ),
            (
                "hubble_constant",
                "matter_density",
                "radiation_density",
                "curvature_density",
                "dark_energy_density",
                "dark_energy_w0",
                "dark_energy_wa",
            ),
            self.scale.scale_id,
        )

    @property
    def realization(self):
        return PhysicalDependencyProjection(self.physical_state.names).project(
            self.physical_state
        )

    def equation_of_state(self, scale_factor: ArrayLike, /) -> Array:
        scale = self._validated_scale(scale_factor)
        return self.dark_energy_w0 + self.dark_energy_wa * (1.0 - scale)

    def dark_energy_scaling(self, scale_factor: ArrayLike, /) -> Array:
        scale = self._validated_scale(scale_factor)
        exponent = -3.0 * (1.0 + self.dark_energy_w0 + self.dark_energy_wa)
        return scale**exponent * jnp.exp(-3.0 * self.dark_energy_wa * (1.0 - scale))

    def _validated_scale(self, scale_factor: ArrayLike, /) -> Array:
        scale = jnp.asarray(scale_factor, dtype=self.hubble_constant.dtype)
        return eqx.error_if(
            scale,
            jnp.any(~jnp.isfinite(scale)) | jnp.any(scale <= 0.0),
            "Scale factor must be finite and positive.",
        )

    def expansion_squared(self, scale_factor: ArrayLike, /) -> Array:
        """Return E(a)^2 = H(a)^2 / H0^2."""
        scale = self._validated_scale(scale_factor)
        expansion = (
            self.radiation_density / scale**4
            + self.matter_density / scale**3
            + self.curvature_density / scale**2
            + self.dark_energy_density * self.dark_energy_scaling(scale)
        )
        return eqx.error_if(
            expansion,
            jnp.any(~jnp.isfinite(expansion)) | jnp.any(expansion <= 0.0),
            "FLRW expansion is non-finite or non-positive at the requested scale.",
        )

    def hubble(self, scale_factor: ArrayLike, /) -> Array:
        return self.hubble_constant * jnp.sqrt(self.expansion_squared(scale_factor))

    def matter_fraction(self, scale_factor: ArrayLike, /) -> Array:
        scale = self._validated_scale(scale_factor)
        return self.matter_density / scale**3 / self.expansion_squared(scale)

    def radiation_fraction(self, scale_factor: ArrayLike, /) -> Array:
        scale = self._validated_scale(scale_factor)
        return self.radiation_density / scale**4 / self.expansion_squared(scale)

    def curvature_fraction(self, scale_factor: ArrayLike, /) -> Array:
        scale = self._validated_scale(scale_factor)
        return self.curvature_density / scale**2 / self.expansion_squared(scale)

    def dark_energy_fraction(self, scale_factor: ArrayLike, /) -> Array:
        scale = self._validated_scale(scale_factor)
        return (
            self.dark_energy_density
            * self.dark_energy_scaling(scale)
            / self.expansion_squared(scale)
        )

    def dlog_hubble_dlog_scale(self, scale_factor: ArrayLike, /) -> Array:
        scale = self._validated_scale(scale_factor)
        expansion_squared = self.expansion_squared(scale)
        dark = self.dark_energy_density * self.dark_energy_scaling(scale)
        derivative = (
            -4.0 * self.radiation_density / scale**4
            - 3.0 * self.matter_density / scale**3
            - 2.0 * self.curvature_density / scale**2
            - 3.0 * (1.0 + self.equation_of_state(scale)) * dark
        )
        return 0.5 * derivative / expansion_squared

    def require_flat(self, token: ArrayLike, /) -> Array:
        return eqx.error_if(
            jnp.asarray(token),
            self.curvature_density != 0.0,
            "Periodic Cartesian cosmology requires zero spatial curvature.",
        )

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
