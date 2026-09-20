#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def elrod_adams_flux(
    pressure_gradient: ArrayLike,
    film_thickness: ArrayLike,
    saturation: ArrayLike,
    viscosity: float,
    sliding_velocity: float,
    /,
):
    h = jnp.asarray(film_thickness)
    theta = jnp.clip(jnp.asarray(saturation), 0, 1)
    return (
        -(h**3) * theta * jnp.asarray(pressure_gradient) / (12 * float(viscosity))
        + 0.5 * float(sliding_velocity) * h * theta
    )


__all__ = ["elrod_adams_flux"]
