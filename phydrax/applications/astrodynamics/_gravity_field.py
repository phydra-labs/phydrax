#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._forces import AbstractAstrodynamicsForce, AstrodynamicsForceEvaluation
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


def _associated_legendre(maximum_degree: int, argument: Array, /) -> Array:
    values = jnp.zeros((maximum_degree + 1, maximum_degree + 1), dtype=argument.dtype)
    values = values.at[0, 0].set(1.0)
    root = jnp.sqrt(jnp.maximum(1.0 - argument * argument, 0.0))
    for order in range(1, maximum_degree + 1):
        values = values.at[order, order].set(
            -(2 * order - 1) * root * values[order - 1, order - 1]
        )
    for order in range(maximum_degree):
        values = values.at[order + 1, order].set(
            (2 * order + 1) * argument * values[order, order]
        )
    for order in range(maximum_degree + 1):
        for degree in range(order + 2, maximum_degree + 1):
            values = values.at[degree, order].set(
                (
                    (2 * degree - 1) * argument * values[degree - 1, order]
                    - (degree + order - 1) * values[degree - 2, order]
                )
                / (degree - order)
            )
    return values


class SphericalHarmonicGravityField(eqx.Module):
    cosine: Array
    sine: Array
    mu: Array
    reference_radius: Array
    context: AstrodynamicsContext
    provenance: AstrodynamicsDataProvenance
    maximum_degree: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    tide_system: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        cosine: ArrayLike,
        sine: ArrayLike,
        mu: ArrayLike,
        reference_radius: ArrayLike,
        context: AstrodynamicsContext,
        provenance: AstrodynamicsDataProvenance,
        /,
        *,
        maximum_degree: int | None = None,
        maximum_order: int | None = None,
        tide_system: str = "tide-free",
    ):
        cosine_host = np.asarray(cosine, dtype=float)
        sine_host = np.asarray(sine, dtype=float)
        if cosine_host.ndim != 2 or cosine_host.shape != sine_host.shape:
            raise ValueError("Gravity coefficients must be matching square arrays.")
        if (
            cosine_host.shape[0] != cosine_host.shape[1]
            or np.any(~np.isfinite(cosine_host))
            or np.any(~np.isfinite(sine_host))
        ):
            raise ValueError("Gravity coefficients must be finite square arrays.")
        available = cosine_host.shape[0] - 1
        degree = available if maximum_degree is None else int(maximum_degree)
        order = degree if maximum_order is None else int(maximum_order)
        if not 0 <= order <= degree <= available:
            raise ValueError("Gravity degree/order exceeds coefficient capacity.")
        self.cosine = jnp.asarray(cosine_host)
        self.sine = jnp.asarray(sine_host)
        self.mu = jnp.asarray(mu).reshape(())
        self.reference_radius = jnp.asarray(reference_radius).reshape(())
        self.context = context
        self.provenance = provenance
        self.maximum_degree = degree
        self.maximum_order = order
        self.tide_system = str(tide_system)
        self.field_id = canonical_fingerprint(
            {
                "kind": "spherical-harmonic-gravity-field",
                "degree": degree,
                "order": order,
                "tide_system": self.tide_system,
                "provenance": provenance.provenance_id,
            }
        )

    def potential(self, position: Array, /) -> Array:
        radius = _norm(position)
        longitude = jnp.arctan2(position[1], position[0])
        sine_latitude = position[2] / jnp.where(radius > 0.0, radius, 1.0)
        legendre = _associated_legendre(self.maximum_degree, sine_latitude)
        series = jnp.asarray(0.0, dtype=position.dtype)
        for degree in range(self.maximum_degree + 1):
            inner = jnp.asarray(0.0, dtype=position.dtype)
            for order in range(min(degree, self.maximum_order) + 1):
                inner = inner + legendre[degree, order] * (
                    self.cosine[degree, order] * jnp.cos(order * longitude)
                    + self.sine[degree, order] * jnp.sin(order * longitude)
                )
            series = series + (self.reference_radius / radius) ** degree * inner
        return -self.mu / jnp.where(radius > 0.0, radius, 1.0) * series


class SphericalHarmonicGravity(AbstractAstrodynamicsForce):
    field: SphericalHarmonicGravityField
    context: AstrodynamicsContext
    force_id: str = eqx.field(static=True)

    def __init__(self, field: SphericalHarmonicGravityField, /):
        self.field = field
        self.context = field.context
        self.force_id = field.field_id

    def evaluate(self, time, state, args: Any = None, /) -> AstrodynamicsForceEvaluation:
        del time, args
        packed = jnp.asarray(state)
        position = packed[:3]
        radius = _norm(position)
        valid = packed.shape == (6,) and jnp.all(jnp.isfinite(packed)) & (
            radius > 0.0
        ) & (self.field.mu > 0.0) & (self.field.reference_radius > 0.0)
        safe_position = jnp.where(valid, position, jnp.asarray((1.0, 0.0, 0.0)))
        potential = self.field.potential(safe_position)
        acceleration = -jax.grad(self.field.potential)(safe_position)
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.INVALID_DOMAIN),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            jnp.where(valid, acceleration, 0.0),
            jnp.where(valid, potential, jnp.nan),
            status[None],
            jnp.asarray(valid),
            status,
            self.force_id,
        )

    def jacobian(self, position: ArrayLike, /) -> Array:
        value = jnp.asarray(position)
        return -jax.hessian(self.field.potential)(value)


class GravityCoefficientCorrection(eqx.Module):
    delta_cosine: Array
    delta_sine: Array
    correction_id: str = eqx.field(static=True)

    def apply(
        self, field: SphericalHarmonicGravityField, /
    ) -> SphericalHarmonicGravityField:
        if (
            self.delta_cosine.shape != field.cosine.shape
            or self.delta_sine.shape != field.sine.shape
        ):
            raise ValueError("Gravity correction capacity does not match field.")
        return SphericalHarmonicGravityField(
            field.cosine + self.delta_cosine,
            field.sine + self.delta_sine,
            field.mu,
            field.reference_radius,
            field.context,
            field.provenance,
            maximum_degree=field.maximum_degree,
            maximum_order=field.maximum_order,
            tide_system=field.tide_system,
        )


__all__ = [
    "GravityCoefficientCorrection",
    "SphericalHarmonicGravity",
    "SphericalHarmonicGravityField",
]
