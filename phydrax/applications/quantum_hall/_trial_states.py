#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact real-space trial amplitudes on the monopole sphere."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...operators.quantum import LogAmplitude


class LaughlinSphereAmplitude(StrictModule):
    """Spin-polarized Laughlin amplitude in one explicit north-patch gauge."""

    particle_count: int = eqx.field(static=True)
    exponent: int = eqx.field(static=True)
    twice_monopole_flux: int = eqx.field(static=True)
    amplitude_id: str = eqx.field(static=True)

    def __init__(self, particle_count: int, exponent: int, /):
        particles = int(particle_count)
        power = int(exponent)
        if particles < 2 or power < 1:
            raise ValueError("Laughlin particle count and exponent must be positive.")
        self.particle_count = particles
        self.exponent = power
        self.twice_monopole_flux = power * (particles - 1)
        self.amplitude_id = canonical_fingerprint(
            {
                "kind": "laughlin-sphere-amplitude",
                "particle_count": particles,
                "exponent": power,
                "gauge": "north-patch-spinor",
            }
        )

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        coordinates = jnp.asarray(configuration)
        if coordinates.shape != (self.particle_count, 2):
            raise ValueError(
                f"Laughlin sphere coordinates must have shape ({self.particle_count}, 2)."
            )
        theta = coordinates[:, 0]
        phi = coordinates[:, 1]
        valid = (
            jnp.all(jnp.isfinite(coordinates))
            & jnp.all(theta >= 0.0)
            & jnp.all(theta <= jnp.pi)
        )
        u = jnp.cos(0.5 * theta) * jnp.exp(0.5j * phi)
        v = jnp.sin(0.5 * theta) * jnp.exp(-0.5j * phi)
        log_abs = jnp.asarray(0.0, dtype=theta.dtype)
        phase = jnp.asarray(1.0 + 0.0j)
        for first in range(self.particle_count):
            for second in range(first + 1, self.particle_count):
                pair = u[first] * v[second] - u[second] * v[first]
                magnitude = jnp.abs(pair)
                log_abs = log_abs + self.exponent * jnp.log(magnitude)
                safe_magnitude = jnp.where(magnitude > 0.0, magnitude, 1.0)
                unit = jnp.where(magnitude > 0.0, pair / safe_magnitude, 1.0 + 0.0j)
                phase = phase * unit**self.exponent
        return LogAmplitude(log_abs, phase, valid=valid)


__all__ = ["LaughlinSphereAmplitude"]
