#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def optical_path_difference(
    displacement_m: ArrayLike,
    surface_normal: ArrayLike,
    refractive_index: ArrayLike = 1.0,
    /,
):
    return (
        2
        * jnp.asarray(refractive_index)
        * jnp.sum(jnp.asarray(displacement_m) * jnp.asarray(surface_normal), axis=-1)
    )


def rms_wavefront_error(optical_path_difference_m: ArrayLike, weights: ArrayLike, /):
    opd = jnp.asarray(optical_path_difference_m)
    w = jnp.asarray(weights) / jnp.sum(weights)
    mean = jnp.sum(w * opd)
    return jnp.sqrt(jnp.sum(w * (opd - mean) ** 2))


__all__ = ["optical_path_difference", "rms_wavefront_error"]
