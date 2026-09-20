#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def jmak_fraction(time_s: ArrayLike, rate_s_inv: ArrayLike, exponent: float, /) -> Array:
    if exponent <= 0:
        raise ValueError("JMAK exponent must be positive.")
    return 1 - jnp.exp(
        -(
            jnp.maximum(jnp.asarray(rate_s_inv) * jnp.asarray(time_s), 0)
            ** float(exponent)
        )
    )


def koistinen_marburger_fraction(
    temperature_k: ArrayLike, start_temperature_k: float, coefficient_k_inv: float, /
) -> Array:
    if coefficient_k_inv <= 0:
        raise ValueError("Coefficient must be positive.")
    return 1 - jnp.exp(
        -float(coefficient_k_inv)
        * jnp.maximum(float(start_temperature_k) - jnp.asarray(temperature_k), 0)
    )


__all__ = ["jmak_fraction", "koistinen_marburger_fraction"]
