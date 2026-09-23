#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def counterflow_heat_exchanger_effectiveness(
    ntu: ArrayLike, capacity_ratio: ArrayLike, /
):
    n = jnp.asarray(ntu)
    c = jnp.asarray(capacity_ratio)
    if n.shape != c.shape and n.shape != () and c.shape != ():
        raise ValueError("NTU and capacity ratio must be scalar or broadcast-aligned.")
    n = eqx.error_if(
        n,
        jnp.any(~jnp.isfinite(n) | ~jnp.isfinite(c) | (n < 0) | (c < 0) | (c > 1)),
        "Heat-exchanger NTU must be nonnegative and capacity ratio in [0, 1].",
    )
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
    if not isfinite(residence_time_s) or residence_time_s <= 0:
        raise ValueError("CSTR residence time must be finite and positive.")
    concentration_ = jnp.asarray(concentration)
    inlet = jnp.asarray(inlet_concentration)
    rate = jnp.asarray(reaction_rate)
    if concentration_.shape != inlet.shape or rate.shape not in (
        (),
        concentration_.shape,
    ):
        raise ValueError("CSTR concentration and reaction-rate fields are incompatible.")
    concentration_ = eqx.error_if(
        concentration_,
        jnp.any(
            ~jnp.isfinite(concentration_)
            | ~jnp.isfinite(inlet)
            | ~jnp.isfinite(rate)
            | (concentration_ < 0)
            | (inlet < 0)
        ),
        "CSTR concentrations/rates must be finite with nonnegative concentrations.",
    )
    return (inlet - concentration_) / float(residence_time_s) + rate


__all__ = ["counterflow_heat_exchanger_effectiveness", "cstr_concentration_rate"]
