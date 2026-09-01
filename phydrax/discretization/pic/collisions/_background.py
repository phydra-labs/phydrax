#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._coulomb import _isotropic_directions
from ._types import PICCollisionResult


class BackgroundMCCPlan(StrictModule, NonTrainableState):
    """Null-collision elastic scattering against a prescribed background."""

    collision_frequency: float = eqx.field(static=True)
    maximum_probability: float = eqx.field(static=True)
    background_velocity: tuple[float, float, float] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        collision_frequency: float,
        /,
        *,
        maximum_probability: float = 0.25,
        background_velocity=(0.0, 0.0, 0.0),
    ):
        frequency = float(collision_frequency)
        maximum = float(maximum_probability)
        background = tuple(float(value) for value in background_velocity)
        if not np.isfinite(frequency) or frequency < 0.0:
            raise ValueError("collision_frequency must be finite and nonnegative.")
        if not np.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            raise ValueError("maximum_probability must lie in (0,1].")
        if len(background) != 3 or any(not np.isfinite(value) for value in background):
            raise ValueError("background_velocity must be a finite three-vector.")
        self.collision_frequency = frequency
        self.maximum_probability = maximum
        self.background_velocity = background
        self.plan_id = canonical_fingerprint(
            {
                "kind": "background-mcc",
                "frequency": frequency,
                "maximum_probability": maximum,
                "background_velocity": background,
            }
        )

    def collide(
        self,
        velocity: ArrayLike,
        mass: ArrayLike,
        active_mask: ArrayLike,
        key,
        step_size: ArrayLike,
        /,
    ) -> PICCollisionResult:
        values = jnp.asarray(velocity)
        masses = jnp.asarray(mass, dtype=values.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("velocity must have shape (capacity,3).")
        if masses.shape != active.shape or masses.shape != (values.shape[0],):
            raise ValueError("MCC particle arrays must preserve capacity.")
        dt = jnp.asarray(step_size, dtype=values.dtype).reshape(())
        probability = 1.0 - jnp.exp(-self.collision_frequency * dt)
        stable = jnp.isfinite(probability) & (probability <= self.maximum_probability)
        event_key, direction_key = jr.split(key)
        collided = active & (
            jr.uniform(event_key, active.shape, dtype=values.dtype) < probability
        )
        background = jnp.asarray(self.background_velocity, dtype=values.dtype)
        relative = values - background
        speed = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
        direction = _isotropic_directions(direction_key, values.shape[0], values.dtype)
        scattered = background + speed[:, None] * direction
        candidate = jnp.where(collided[:, None], scattered, values)
        candidate = jnp.where(active[:, None], candidate, 0.0)
        particle_momentum_before = jnp.sum(masses[:, None] * values, axis=0)
        particle_momentum_after = jnp.sum(masses[:, None] * candidate, axis=0)
        particle_energy_before = 0.5 * jnp.sum(masses * jnp.sum(values * values, axis=-1))
        particle_energy_after = 0.5 * jnp.sum(
            masses * jnp.sum(candidate * candidate, axis=-1)
        )
        momentum_source = particle_momentum_before - particle_momentum_after
        energy_source = particle_energy_before - particle_energy_after
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(momentum_source))
            & jnp.isfinite(energy_source)
        )
        successful = stable & finite
        return PICCollisionResult(
            candidate,
            jnp.where(successful, candidate, values),
            collided,
            jnp.sum(collided, dtype=jnp.int32),
            jnp.sqrt(jnp.sum(momentum_source**2)),
            -energy_source,
            momentum_source,
            energy_source,
            probability,
            finite,
            stable,
            successful,
            self.plan_id,
        )


__all__ = ["BackgroundMCCPlan"]
