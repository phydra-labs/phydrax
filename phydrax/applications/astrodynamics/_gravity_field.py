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
from ...special._spherical_harmonic import _real_spherical_harmonic_table
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._forces import AbstractAstrodynamicsForce, AstrodynamicsForceEvaluation
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


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
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        cosine_basis, sine_basis = _real_spherical_harmonic_table(
            self.maximum_degree,
            position / safe_radius,
            normalization="unnormalized",
            condon_shortley=True,
        )
        degrees = jnp.arange(self.maximum_degree + 1)
        orders = jnp.arange(self.maximum_degree + 1)
        degree_grid = degrees[:, None]
        order_grid = orders[None, :]
        mask = (order_grid <= degree_grid) & (order_grid <= self.maximum_order)
        size = self.maximum_degree + 1
        cosine = self.cosine[:size, :size]
        sine = self.sine[:size, :size]
        angular = cosine * cosine_basis + sine * sine_basis
        inner = jnp.sum(jnp.where(mask, angular, 0.0), axis=-1)
        series = jnp.sum((self.reference_radius / safe_radius) ** degrees * inner)
        return -self.mu * series / safe_radius


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
        potential, potential_gradient = jax.value_and_grad(self.field.potential)(
            safe_position
        )
        acceleration = -potential_gradient
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
