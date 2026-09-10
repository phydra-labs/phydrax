#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....interchange import GeomagneticHarmonicModel, ICGEMGravityModel
from ....special._spherical_harmonic import _real_spherical_harmonic_table


Normalization = Literal["unnormalized", "schmidt", "fully_normalized"]


def _normalization_name(value: str) -> Normalization:
    normalized = value.strip().lower().replace("-", "_")
    mapping = {
        "unnormalized": "unnormalized",
        "schmidt": "schmidt",
        "schmidt_semi_normalized": "schmidt",
        "fully_normalized": "fully_normalized",
        "fully-normalized": "fully_normalized",
    }
    if normalized not in mapping:
        raise ValueError("Unsupported spherical-harmonic normalization.")
    return cast(Normalization, mapping[normalized])


class SphericalGravityResult(StrictModule):
    potential_m2_s2: Array
    acceleration_m_s2: Array
    finite: Array


class SphericalHarmonicGravityPlan(StrictModule, NonTrainableState):
    model: ICGEMGravityModel
    maximum_degree: int = eqx.field(static=True)
    normalization: Normalization = eqx.field(static=True)

    def __init__(self, model: ICGEMGravityModel, /, *, maximum_degree: int | None = None):
        if not isinstance(model, ICGEMGravityModel):
            raise TypeError("Spherical gravity requires ICGEMGravityModel.")
        maximum = model.maximum_degree if maximum_degree is None else int(maximum_degree)
        if not 0 <= maximum <= model.maximum_degree:
            raise ValueError("Gravity synthesis degree lies outside model support.")
        self.model, self.maximum_degree = model, maximum
        self.normalization = _normalization_name(model.normalization)

    def potential(self, position_m: ArrayLike, /) -> Array:
        position = jnp.asarray(position_m)
        if position.shape != (3,):
            raise ValueError("Spherical gravity position must be one Cartesian vector.")
        radius = jnp.sqrt(jnp.sum(position**2))
        radius = eqx.error_if(
            radius,
            ~jnp.isfinite(radius) | (radius < self.model.reference_radius_m),
            "Exterior spherical gravity position must be finite and outside reference radius.",
        )
        cosine_basis, sine_basis = _real_spherical_harmonic_table(
            self.maximum_degree,
            position / radius,
            normalization=self.normalization,
            condon_shortley=False,
        )
        radial = self.model.reference_radius_m / radius
        degrees = jnp.arange(self.maximum_degree + 1)
        orders = jnp.arange(self.maximum_degree + 1)
        order_grid = orders[None, :]
        degree_grid = degrees[:, None]
        size = self.maximum_degree + 1
        angular = (
            self.model.cosine[:size, :size] * cosine_basis
            + self.model.sine[:size, :size] * sine_basis
        )
        inner = jnp.sum(
            jnp.where(order_grid <= degree_grid, angular, 0.0),
            axis=-1,
        )
        total = jnp.sum(radial**degrees * inner)
        return self.model.gravitational_constant_m3_s2 * total / radius

    def evaluate(self, positions_m: ArrayLike, /) -> SphericalGravityResult:
        positions = jnp.asarray(positions_m)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Spherical gravity positions must have shape (n,3).")
        potential, acceleration = jax.vmap(jax.value_and_grad(self.potential))(positions)
        finite = jnp.all(jnp.isfinite(potential)) & jnp.all(jnp.isfinite(acceleration))
        return SphericalGravityResult(potential, acceleration, finite)


class SphericalMagneticResult(StrictModule):
    field_nT: Array
    total_field_anomaly_nT: Array
    finite: Array


class SphericalHarmonicMagneticPlan(StrictModule, NonTrainableState):
    model: GeomagneticHarmonicModel
    maximum_degree: int = eqx.field(static=True)

    def __init__(
        self, model: GeomagneticHarmonicModel, /, *, maximum_degree: int | None = None
    ):
        if not isinstance(model, GeomagneticHarmonicModel):
            raise TypeError("Spherical magnetics requires GeomagneticHarmonicModel.")
        maximum = model.maximum_degree if maximum_degree is None else int(maximum_degree)
        if not 1 <= maximum <= model.maximum_degree:
            raise ValueError("Magnetic synthesis degree lies outside model support.")
        self.model, self.maximum_degree = model, maximum

    def scalar_potential(
        self, position_m: ArrayLike, decimal_year: ArrayLike, /
    ) -> Array:
        position, year = jnp.asarray(position_m), jnp.asarray(decimal_year)
        if position.shape != (3,) or year.shape != ():
            raise ValueError("Magnetic position/year shapes are invalid.")
        radius = jnp.sqrt(jnp.sum(position**2))
        radius = eqx.error_if(
            radius,
            ~jnp.isfinite(radius)
            | (radius < self.model.reference_radius_m)
            | ~jnp.isfinite(year),
            "Exterior magnetic position and epoch must be finite and supported.",
        )
        cosine_basis, sine_basis = _real_spherical_harmonic_table(
            self.maximum_degree,
            position / radius,
            normalization="schmidt",
            condon_shortley=False,
        )
        elapsed = year - self.model.epoch_decimal_year
        total = jnp.asarray(0.0, dtype=position.dtype)
        radial = self.model.reference_radius_m / radius
        for degree in range(1, self.maximum_degree + 1):
            inner = jnp.asarray(0.0, dtype=position.dtype)
            for order in range(degree + 1):
                g = (
                    self.model.g_nT[degree, order]
                    + elapsed * self.model.secular_g_nT_year[degree, order]
                )
                h = (
                    self.model.h_nT[degree, order]
                    + elapsed * self.model.secular_h_nT_year[degree, order]
                )
                inner = (
                    inner
                    + cosine_basis[degree, order] * g
                    + sine_basis[degree, order] * h
                )
            total = total + radial ** (degree + 1) * inner
        return self.model.reference_radius_m * total

    def evaluate(
        self,
        positions_m: ArrayLike,
        decimal_year: ArrayLike,
        /,
        *,
        reference_direction: ArrayLike,
    ) -> SphericalMagneticResult:
        positions = jnp.asarray(positions_m)
        years = jnp.broadcast_to(jnp.asarray(decimal_year), (positions.shape[0],))
        direction = jnp.asarray(reference_direction)
        if positions.ndim != 2 or positions.shape[1] != 3 or direction.shape != (3,):
            raise ValueError(
                "Magnetic positions or reference direction have wrong shape."
            )
        direction = eqx.error_if(
            direction,
            jnp.any(~jnp.isfinite(direction))
            | (jnp.abs(jnp.sum(direction**2) - 1) > 1e-10),
            "Magnetic reference direction must be finite unit vector.",
        )
        field = -jax.vmap(jax.grad(self.scalar_potential, argnums=0))(positions, years)
        total = field @ direction
        finite = jnp.all(jnp.isfinite(field)) & jnp.all(jnp.isfinite(total))
        return SphericalMagneticResult(field, total, finite)


__all__ = [
    "SphericalGravityResult",
    "SphericalHarmonicGravityPlan",
    "SphericalHarmonicMagneticPlan",
    "SphericalMagneticResult",
]
