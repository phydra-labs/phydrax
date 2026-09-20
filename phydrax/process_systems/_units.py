#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def counterflow_heat_exchanger_effectiveness(
    ntu: ArrayLike, capacity_ratio: ArrayLike, /
):
    n = jnp.asarray(ntu)
    c = jnp.asarray(capacity_ratio)
    return jnp.where(
        jnp.abs(1 - c) < 1e-10,
        n / (1 + n),
        (1 - jnp.exp(-n * (1 - c))) / (1 - c * jnp.exp(-n * (1 - c))),
    )


def cstr_concentration_rate(
    concentration: ArrayLike,
    inlet_concentration: ArrayLike,
    residence_time_s: float,
    reaction_rate: ArrayLike,
    /,
):
    return (jnp.asarray(inlet_concentration) - jnp.asarray(concentration)) / float(
        residence_time_s
    ) + jnp.asarray(reaction_rate)


__all__ = ["counterflow_heat_exchanger_effectiveness", "cstr_concentration_rate"]
