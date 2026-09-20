#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def upper_convected_rate(
    conformation: ArrayLike,
    velocity_gradient: ArrayLike,
    material_derivative: ArrayLike,
    relaxation: ArrayLike,
    /,
):
    a = jnp.asarray(conformation)
    g = jnp.asarray(velocity_gradient)
    return (
        jnp.asarray(material_derivative)
        - g @ a
        - a @ jnp.swapaxes(g, -1, -2)
        - jnp.asarray(relaxation)
    )


def gordon_schowalter_rate(
    tensor: ArrayLike, velocity_gradient: ArrayLike, slip_parameter: float, /
):
    t = jnp.asarray(tensor)
    g = jnp.asarray(velocity_gradient)
    d = 0.5 * (g + jnp.swapaxes(g, -1, -2))
    w = 0.5 * (g - jnp.swapaxes(g, -1, -2))
    return w @ t - t @ w + float(slip_parameter) * (d @ t + t @ d)


__all__ = ["gordon_schowalter_rate", "upper_convected_rate"]
