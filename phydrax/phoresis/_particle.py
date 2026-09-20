#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class PhoreticParticleStep:
    position: Array
    velocity: Array
    finite: Array


def advance_phoretic_particle(
    position: ArrayLike,
    fluid_velocity: ArrayLike,
    phoretic_velocity: ArrayLike,
    step_size_s: ArrayLike,
    /,
):
    velocity = jnp.asarray(fluid_velocity) + jnp.asarray(phoretic_velocity)
    candidate = jnp.asarray(position) + jnp.asarray(step_size_s) * velocity
    finite = jnp.all(jnp.isfinite(candidate)) & (jnp.asarray(step_size_s) > 0)
    return PhoreticParticleStep(
        jnp.where(finite, candidate, jnp.asarray(position)), velocity, finite
    )


__all__ = ["PhoreticParticleStep", "advance_phoretic_particle"]
