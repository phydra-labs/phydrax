#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def exponential_maximum_entropy_density(
    coordinate: ArrayLike, moment0: ArrayLike, moment1: ArrayLike, /
):
    mean = jnp.asarray(moment1) / jnp.asarray(moment0)
    x = jnp.asarray(coordinate)
    return jnp.asarray(moment0) / mean * jnp.exp(-x / mean)


__all__ = ["exponential_maximum_entropy_density"]
