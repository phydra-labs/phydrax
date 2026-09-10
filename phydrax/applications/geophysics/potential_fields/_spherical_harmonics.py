#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import exp, lgamma, sqrt
from typing import cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....interchange import GeomagneticHarmonicModel, ICGEMGravityModel


Normalization = Literal["unnormalized", "schmidt", "fully_normalized"]


def _normalization(degree: int, order: int, kind: Normalization) -> float:
    if kind == "unnormalized":
        return 1.0
    factor = 2.0 if order > 0 else 1.0
    log_ratio = lgamma(degree - order + 1.0) - lgamma(degree + order + 1.0)
    if kind == "fully_normalized":
        factor *= 2 * degree + 1
    return sqrt(factor * exp(log_ratio))


def _associated_legendre(
    maximum_degree: int, argument: Array, normalization: Normalization
) -> Array:
    dtype = argument.dtype
    values = jnp.zeros((maximum_degree + 1, maximum_degree + 1), dtype=dtype)
    values = values.at[0, 0].set(1.0)
    root = jnp.sqrt(jnp.maximum(1.0 - argument**2, 0.0))

    def diagonal(order, current):
        return current.at[order, order].set(
            (2 * order - 1) * root * current[order - 1, order - 1]
        )

    values = jax.lax.fori_loop(1, maximum_degree + 1, diagonal, values)

    def adjacent(order, current):
        return current.at[order + 1, order].set(
            (2 * order + 1) * argument * current[order, order]
        )

    values = jax.lax.fori_loop(0, maximum_degree, adjacent, values)

    def column(order, current):
        def recurrence(degree, table):
            active = degree >= order + 2
            denominator = jnp.where(active, degree - order, 1)
            candidate = (
                (2 * degree - 1) * argument * table[degree - 1, order]
                - (degree + order - 1) * table[degree - 2, order]
            ) / denominator
            return table.at[degree, order].set(
                jnp.where(active, candidate, table[degree, order])
            )

        return jax.lax.fori_loop(0, maximum_degree + 1, recurrence, current)

    values = jax.lax.fori_loop(0, maximum_degree + 1, column, values)
    factors = jnp.asarray(
        [
            [
                _normalization(degree, order, normalization) if order <= degree else 0.0
                for order in range(maximum_degree + 1)
            ]
            for degree in range(maximum_degree + 1)
        ],
        dtype=dtype,
    )
    return values * factors


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
        longitude = jnp.arctan2(position[1], position[0])
        latitude_sine = position[2] / radius
        legendre = _associated_legendre(
            self.maximum_degree, latitude_sine, self.normalization
        )
        radial = self.model.reference_radius_m / radius
        degrees = jnp.arange(self.maximum_degree + 1)
        orders = jnp.arange(self.maximum_degree + 1)
        order_grid = orders[None, :]
        degree_grid = degrees[:, None]
        angular = self.model.cosine * jnp.cos(
            order_grid * longitude
        ) + self.model.sine * jnp.sin(order_grid * longitude)
        inner = jnp.sum(
            jnp.where(order_grid <= degree_grid, legendre * angular, 0.0),
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
        longitude = jnp.arctan2(position[1], position[0])
        legendre = _associated_legendre(
            self.maximum_degree, position[2] / radius, "schmidt"
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
                angle = order * longitude
                inner = inner + legendre[degree, order] * (
                    g * jnp.cos(angle) + h * jnp.sin(angle)
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
