#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._background import FLRWBackground
from ._products import CosmologyProductProvenance, CosmologyRealizationSignature


class FLRWDistanceResult(StrictModule):
    redshift: Array
    radial_comoving_distance: Array
    transverse_comoving_distance: Array
    angular_diameter_distance: Array
    luminosity_distance: Array
    differential_comoving_volume: Array
    lookback_time: Array
    realization: CosmologyRealizationSignature
    provenance: CosmologyProductProvenance


class FLRWDistancePlan(StrictModule, NonTrainableState):
    """Fixed-quadrature radial and transverse FLRW geometry."""

    nodes: Array
    weights: Array
    light_speed: float = eqx.field(static=True)
    near_flat_threshold: float = eqx.field(static=True)
    order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        light_speed: float = 1.0,
        order: int = 64,
        near_flat_threshold: float = 1.0e-6,
    ):
        speed = float(light_speed)
        order_ = int(order)
        threshold = float(near_flat_threshold)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("light_speed must be finite and positive in scale units.")
        if order_ < 2:
            raise ValueError("FLRW distance quadrature order must be at least two.")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("near_flat_threshold must be finite and positive.")
        rule = gauss_legendre_data(order_)
        self.nodes = jnp.asarray(rule.nodes)
        self.weights = jnp.asarray(rule.weights)
        self.light_speed = speed
        self.near_flat_threshold = threshold
        self.order = order_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flrw-distance-plan",
                "light_speed": speed,
                "order": order_,
                "near_flat_threshold": threshold,
            }
        )

    def _integrate_redshift(self, redshift: Array, integrand) -> Array:
        mapped = 0.5 * redshift[..., None] * (self.nodes + 1.0)
        values = integrand(mapped)
        return 0.5 * redshift * jnp.sum(self.weights * values, axis=-1)

    def radial_comoving_distance(
        self, background: FLRWBackground, redshift: ArrayLike, /
    ) -> Array:
        z = jnp.asarray(redshift, dtype=background.hubble_constant.dtype)
        z = eqx.error_if(
            z,
            jnp.any(~jnp.isfinite(z)) | jnp.any(z < 0.0),
            "FLRW distance redshift must be finite and non-negative.",
        )
        return self.light_speed * self._integrate_redshift(
            z,
            lambda node: 1.0 / background.hubble(1.0 / (1.0 + node)),
        )

    def transverse_comoving_distance(
        self, background: FLRWBackground, redshift: ArrayLike, /
    ) -> Array:
        radial = self.radial_comoving_distance(background, redshift)
        dimensionless = background.hubble_constant * radial / self.light_speed
        curvature = background.curvature_density
        q = curvature * dimensionless**2
        series = dimensionless * (1.0 + q / 6.0 + q**2 / 120.0 + q**3 / 5040.0)
        absolute = jnp.abs(curvature)
        safe_root = jnp.sqrt(jnp.maximum(absolute, jnp.finfo(radial.dtype).tiny))
        argument = safe_root * dimensionless
        direct = jnp.where(
            curvature > 0.0,
            jnp.sinh(argument) / safe_root,
            jnp.sin(argument) / safe_root,
        )
        dimensionless_transverse = jnp.where(
            jnp.abs(q) <= self.near_flat_threshold,
            series,
            direct,
        )
        return self.light_speed * dimensionless_transverse / background.hubble_constant

    def lookback_time(self, background: FLRWBackground, redshift: ArrayLike, /) -> Array:
        z = jnp.asarray(redshift, dtype=background.hubble_constant.dtype)
        z = eqx.error_if(
            z,
            jnp.any(~jnp.isfinite(z)) | jnp.any(z < 0.0),
            "FLRW lookback redshift must be finite and non-negative.",
        )
        return self._integrate_redshift(
            z,
            lambda node: 1.0 / ((1.0 + node) * background.hubble(1.0 / (1.0 + node))),
        )

    def cosmic_time_between(
        self,
        background: FLRWBackground,
        start_scale_factor: ArrayLike,
        end_scale_factor: ArrayLike,
        /,
    ) -> Array:
        start = jnp.asarray(start_scale_factor, dtype=background.hubble_constant.dtype)
        end = jnp.asarray(end_scale_factor, dtype=background.hubble_constant.dtype)
        if start.shape != end.shape:
            raise ValueError("Cosmic-time scale-factor arrays must have equal shape.")
        start = eqx.error_if(
            start,
            jnp.any(~jnp.isfinite(start))
            | jnp.any(~jnp.isfinite(end))
            | jnp.any(start <= 0.0)
            | jnp.any(end <= start),
            "Cosmic-time bounds must be finite, positive, and increasing.",
        )
        mapped = (
            0.5 * (end - start)[..., None] * self.nodes + 0.5 * (end + start)[..., None]
        )
        values = 1.0 / (mapped * background.hubble(mapped))
        return 0.5 * (end - start) * jnp.sum(self.weights * values, axis=-1)

    def conformal_time_between(
        self,
        background: FLRWBackground,
        start_scale_factor: ArrayLike,
        end_scale_factor: ArrayLike,
        /,
    ) -> Array:
        start = jnp.asarray(start_scale_factor, dtype=background.hubble_constant.dtype)
        end = jnp.asarray(end_scale_factor, dtype=background.hubble_constant.dtype)
        if start.shape != end.shape:
            raise ValueError("Conformal-time scale-factor arrays must have equal shape.")
        start = eqx.error_if(
            start,
            jnp.any(~jnp.isfinite(start))
            | jnp.any(~jnp.isfinite(end))
            | jnp.any(start <= 0.0)
            | jnp.any(end <= start),
            "Conformal-time bounds must be finite, positive, and increasing.",
        )
        mapped = (
            0.5 * (end - start)[..., None] * self.nodes + 0.5 * (end + start)[..., None]
        )
        values = 1.0 / (mapped**2 * background.hubble(mapped))
        return 0.5 * (end - start) * jnp.sum(self.weights * values, axis=-1)

    def evaluate(
        self, background: FLRWBackground, redshift: ArrayLike, /
    ) -> FLRWDistanceResult:
        z = jnp.asarray(redshift, dtype=background.hubble_constant.dtype)
        radial = self.radial_comoving_distance(background, z)
        transverse = self.transverse_comoving_distance(background, z)
        one_plus = 1.0 + z
        hubble = background.hubble(1.0 / one_plus)
        provenance = CosmologyProductProvenance(
            producer="phydrax.applications.cosmology.FLRWDistancePlan",
            producer_version="native",
            model_form_id=background.model_form_id,
            request_id="native-dynamic-background",
            numerical_policy_id=self.plan_id,
            physics_policy_id="flrw-background-geometry",
            scale_id=background.scale.scale_id,
            source_kind="native",
            differentiation="native-parameter",
        )
        return FLRWDistanceResult(
            redshift=z,
            radial_comoving_distance=radial,
            transverse_comoving_distance=transverse,
            angular_diameter_distance=transverse / one_plus,
            luminosity_distance=transverse * one_plus,
            differential_comoving_volume=self.light_speed * transverse**2 / hubble,
            lookback_time=self.lookback_time(background, z),
            realization=background.realization,
            provenance=provenance,
        )


__all__ = ["FLRWDistancePlan", "FLRWDistanceResult"]
