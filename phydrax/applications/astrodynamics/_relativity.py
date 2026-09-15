#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ._context import AstrodynamicsContext
from ._forces import AbstractAstrodynamicsForce, AstrodynamicsForceEvaluation
from ._status import AstrodynamicsStatus


_SPEED_OF_LIGHT = 299792458.0
_GRAVITATIONAL_CONSTANT = 6.67430e-11


class Schwarzschild1PNForce(AbstractAstrodynamicsForce):
    """First post-Newtonian Schwarzschild acceleration correction."""

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
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        mu_host = np.asarray(mu)
        light_host = np.asarray(speed_of_light)
        if mu_host.shape != () or light_host.shape != ():
            raise ValueError("mu and speed_of_light must be scalars.")
        if (
            not np.isfinite(mu_host)
            or not np.isfinite(light_host)
            or mu_host <= 0.0
            or light_host <= 0.0
        ):
            raise ValueError("mu and speed_of_light must be finite and positive.")
        self.mu = jnp.asarray(mu_host)
        self.context = context
        self.speed_of_light = jnp.asarray(light_host)
        self.force_id = canonical_fingerprint(
            {
                "kind": "schwarzschild-test-particle-1pn-force",
                "convention": "harmonic-coordinate-acceleration-correction",
                "context": context.context_id,
                "mu": mu_host,
                "speed_of_light": light_host,
            }
        )

    def evaluate(self, time, state, args: Any = None, /):
        del time, args
        packed = jnp.asarray(state)
        if packed.shape != (6,):
            raise ValueError("Astrodynamics force state must have shape (6,).")
        position, velocity = packed[:3], packed[3:]
        radius = jnp.sqrt(jnp.sum(position * position))
        finite = jnp.all(jnp.isfinite(packed))
        safe_radius = jnp.where(finite & (radius > 0.0), radius, 1.0)
        speed_squared = jnp.sum(velocity * velocity)
        radial_dot = jnp.sum(position * velocity)
        acceleration = (
            self.mu
            / (self.speed_of_light**2 * safe_radius**3)
            * (
                (4.0 * self.mu / safe_radius - speed_squared) * position
                + 4.0 * radial_dot * velocity
            )
        )
        valid = finite & (radius > 0.0)
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                radius > 0.0,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.COLLISION),
            ),
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


__all__ = ["LenseThirringRelativity", "Schwarzschild1PNForce"]
