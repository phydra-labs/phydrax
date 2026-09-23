#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def exponential_maximum_entropy_density(
    coordinate: ArrayLike, moment0: ArrayLike, moment1: ArrayLike, /
):
    m0 = jnp.asarray(moment0)
    m1 = jnp.asarray(moment1)
    x = jnp.asarray(coordinate)
    m0 = eqx.error_if(
        m0,
        jnp.any(
            ~jnp.isfinite(m0)
            | ~jnp.isfinite(m1)
            | ~jnp.isfinite(x)
            | (m0 <= 0)
            | (m1 <= 0)
            | (x < 0)
        ),
        "Maximum-entropy moments must be finite/positive and coordinates finite/nonnegative.",
    )
    mean = m1 / m0
    return m0 / mean * jnp.exp(-x / mean)


__all__ = ["exponential_maximum_entropy_density"]
