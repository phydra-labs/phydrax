#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ._context import AstrodynamicsContext
from ._ephemeris import TabulatedEphemeris
from ._forces import AbstractAstrodynamicsForce, AstrodynamicsForceEvaluation
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


class ThirdBodyGravity(AbstractAstrodynamicsForce):
    ephemeris: TabulatedEphemeris
    body_index: int = eqx.field(static=True)
    context: AstrodynamicsContext
    force_id: str = eqx.field(static=True)

    def __init__(self, ephemeris: TabulatedEphemeris, body_index: int, /):
        if not isinstance(ephemeris, TabulatedEphemeris):
            raise TypeError("ephemeris must be a TabulatedEphemeris.")
        index = int(body_index)
        if not 0 <= index < ephemeris.catalog.capacity:
            raise ValueError("body_index is outside ephemeris capacity.")
        self.ephemeris = ephemeris
        self.body_index = index
        self.context = ephemeris.catalog.context
        self.force_id = canonical_fingerprint(
            {
                "kind": "third-body-gravity",
                "ephemeris": ephemeris.ephemeris_id,
                "body_index": index,
            }
        )

    def evaluate(self, time, state, args=None, /) -> AstrodynamicsForceEvaluation:
        del args
        packed = jnp.asarray(state)
        if packed.shape != (6,):
            raise ValueError("Third-body force state must have shape (6,).")
        body = self.ephemeris.evaluate(time, self.body_index)
        body_position = body.state.position
        relative = body_position - packed[:3]
        body_radius = _norm(body_position)
        relative_radius = _norm(relative)
        mu = self.ephemeris.catalog.gravitational_parameters[self.body_index]
        domain = body.valid & (body_radius > 0.0) & (relative_radius > 0.0)
        acceleration = mu * (
            relative / jnp.where(relative_radius > 0.0, relative_radius**3, 1.0)
            - body_position / jnp.where(body_radius > 0.0, body_radius**3, 1.0)
        )
        acceleration = jnp.where(domain, acceleration, jnp.zeros_like(acceleration))
        status = jnp.where(
            domain,
            int(AstrodynamicsStatus.SUCCESS),
            jnp.where(
                body.valid,
                int(AstrodynamicsStatus.COLLISION),
                body.status,
            ),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            acceleration,
            jnp.asarray(jnp.nan, dtype=packed.dtype),
            status[None],
            domain,
            status,
            self.force_id,
        )


def _zonal_potential(
    position: Array,
    mu: Array,
    radius: Array,
    j2: Array,
    j3: Array,
    j4: Array,
    /,
) -> Array:
    distance = _norm(position)
    cosine = position[2] / jnp.where(distance > 0.0, distance, 1.0)
    p2 = 0.5 * (3.0 * cosine**2 - 1.0)
    p3 = 0.5 * (5.0 * cosine**3 - 3.0 * cosine)
    p4 = (35.0 * cosine**4 - 30.0 * cosine**2 + 3.0) / 8.0
    ratio = radius / jnp.where(distance > 0.0, distance, 1.0)
    correction = 1.0 - j2 * ratio**2 * p2 - j3 * ratio**3 * p3 - j4 * ratio**4 * p4
    return -mu / jnp.where(distance > 0.0, distance, 1.0) * correction


class ZonalHarmonicGravity(AbstractAstrodynamicsForce):
    mu: Array
    reference_radius: Array
    j2: Array
    j3: Array
    j4: Array
    context: AstrodynamicsContext
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        mu: ArrayLike,
        reference_radius: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        j2: ArrayLike = 0.0,
        j3: ArrayLike = 0.0,
        j4: ArrayLike = 0.0,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        self.mu = jnp.asarray(mu).reshape(())
        self.reference_radius = jnp.asarray(reference_radius).reshape(())
        self.j2 = jnp.asarray(j2).reshape(())
        self.j3 = jnp.asarray(j3).reshape(())
        self.j4 = jnp.asarray(j4).reshape(())
        self.context = context
        self.force_id = canonical_fingerprint(
            {
                "kind": "zonal-harmonic-gravity",
                "context": context.context_id,
                "maximum_degree": 4,
            }
        )

    def evaluate(self, time, state, args: Any = None, /) -> AstrodynamicsForceEvaluation:
        del time, args
        packed = jnp.asarray(state)
        if packed.shape != (6,):
            raise ValueError("Zonal gravity state must have shape (6,).")
        position = packed[:3]
        distance = _norm(position)
        finite = (
            jnp.all(jnp.isfinite(packed))
            & jnp.isfinite(self.mu)
            & jnp.isfinite(self.reference_radius)
            & jnp.isfinite(self.j2)
            & jnp.isfinite(self.j3)
            & jnp.isfinite(self.j4)
        )
        domain = (
            finite & (self.mu > 0.0) & (self.reference_radius > 0.0) & (distance > 0.0)
        )
        safe_position = jnp.where(domain, position, jnp.asarray((1.0, 0.0, 0.0)))
        potential_fn = lambda value: _zonal_potential(
            value,
            jnp.where(domain, self.mu, 1.0),
            jnp.where(domain, self.reference_radius, 1.0),
            self.j2,
            self.j3,
            self.j4,
        )
        potential = potential_fn(safe_position)
        acceleration = -jax.grad(potential_fn)(safe_position)
        acceleration = jnp.where(domain, acceleration, jnp.zeros_like(acceleration))
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                distance <= 0.0,
                int(AstrodynamicsStatus.COLLISION),
                jnp.where(
                    domain,
                    int(AstrodynamicsStatus.SUCCESS),
                    int(AstrodynamicsStatus.INVALID_DOMAIN),
                ),
            ),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            acceleration,
            jnp.where(domain, potential, jnp.asarray(jnp.nan, dtype=packed.dtype)),
            status[None],
            domain,
            status,
            self.force_id,
        )


__all__ = ["ThirdBodyGravity", "ZonalHarmonicGravity"]
