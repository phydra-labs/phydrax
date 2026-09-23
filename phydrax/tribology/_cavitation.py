#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
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
    if not isfinite(viscosity) or viscosity <= 0 or not isfinite(sliding_velocity):
        raise ValueError(
            "Cavitation viscosity must be finite/positive and velocity finite."
        )
    gradient = jnp.asarray(pressure_gradient)
    h = jnp.asarray(film_thickness)
    theta = jnp.asarray(saturation)
    if gradient.shape != h.shape or theta.shape != h.shape:
        raise ValueError("Cavitation fields must have matching shapes.")
    h = eqx.error_if(
        h,
        jnp.any(
            ~jnp.isfinite(gradient)
            | ~jnp.isfinite(h)
            | ~jnp.isfinite(theta)
            | (h <= 0)
            | (theta < 0)
            | (theta > 1)
        ),
        "Cavitation fields must be finite with positive film and saturation in [0, 1].",
    )
    return (
        -(h**3) * theta * gradient / (12 * float(viscosity))
        + 0.5 * float(sliding_velocity) * h * theta
    )


__all__ = ["elrod_adams_flux"]
