#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ._context import AstrodynamicsContext
from ._forces import AbstractAstrodynamicsForce, AstrodynamicsForceEvaluation
from ._status import AstrodynamicsStatus


_SPEED_OF_LIGHT = 299792458.0
_GRAVITATIONAL_CONSTANT = 6.67430e-11


class SchwarzschildRelativity(AbstractAstrodynamicsForce):
    mu: jnp.ndarray
    context: AstrodynamicsContext
    speed_of_light: jnp.ndarray
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        mu: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        speed_of_light: ArrayLike = _SPEED_OF_LIGHT,
    ):
        self.mu = jnp.asarray(mu).reshape(())
        self.context = context
        self.speed_of_light = jnp.asarray(speed_of_light).reshape(())
        self.force_id = canonical_fingerprint(
            {"kind": "schwarzschild-1pn", "context": context.context_id}
        )

    def evaluate(self, time, state, args: Any = None, /):
        del time, args
        packed = jnp.asarray(state)
        position, velocity = packed[:3], packed[3:]
        radius = jnp.sqrt(jnp.sum(position * position))
        speed_squared = jnp.sum(velocity * velocity)
        radial_dot = jnp.sum(position * velocity)
        acceleration = (
            self.mu
            / (self.speed_of_light**2 * radius**3)
            * (
                (4.0 * self.mu / radius - speed_squared) * position
                + 4.0 * radial_dot * velocity
            )
        )
        valid = (
            jnp.all(jnp.isfinite(packed))
            & (radius > 0.0)
            & (self.mu > 0.0)
            & (self.speed_of_light > 0.0)
        )
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.INVALID_DOMAIN),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            jnp.where(valid, acceleration, 0.0),
            jnp.asarray(jnp.nan),
            status[None],
            valid,
            status,
            self.force_id,
        )


class LenseThirringRelativity(AbstractAstrodynamicsForce):
    spin_angular_momentum: jnp.ndarray
    context: AstrodynamicsContext
    gravitational_constant: jnp.ndarray
    speed_of_light: jnp.ndarray
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        spin_angular_momentum,
        context,
        /,
        *,
        gravitational_constant=_GRAVITATIONAL_CONSTANT,
        speed_of_light=_SPEED_OF_LIGHT,
    ):
        spin = jnp.asarray(spin_angular_momentum)
        if spin.shape != (3,):
            raise ValueError("Spin angular momentum must have shape (3,).")
        self.spin_angular_momentum = spin
        self.context = context
        self.gravitational_constant = jnp.asarray(gravitational_constant).reshape(())
        self.speed_of_light = jnp.asarray(speed_of_light).reshape(())
        self.force_id = canonical_fingerprint(
            {"kind": "lense-thirring", "context": context.context_id}
        )

    def evaluate(self, time, state, args=None, /):
        del time, args
        packed = jnp.asarray(state)
        position, velocity = packed[:3], packed[3:]
        radius = jnp.sqrt(jnp.sum(position * position))
        projection = jnp.sum(position * self.spin_angular_momentum) / radius**2
        gravitomagnetic = self.spin_angular_momentum - 3.0 * projection * position
        acceleration = (
            2.0
            * self.gravitational_constant
            / (self.speed_of_light**2 * radius**3)
            * jnp.cross(velocity, gravitomagnetic)
        )
        valid = jnp.all(jnp.isfinite(acceleration)) & (radius > 0.0)
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.INVALID_DOMAIN),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            jnp.where(valid, acceleration, 0.0),
            jnp.asarray(jnp.nan),
            status[None],
            valid,
            status,
            self.force_id,
        )


__all__ = ["LenseThirringRelativity", "SchwarzschildRelativity"]
