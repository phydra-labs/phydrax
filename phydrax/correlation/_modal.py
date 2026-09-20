#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def stabilization_mask(
    frequencies: ArrayLike,
    damping: ArrayLike,
    previous_frequencies: ArrayLike,
    previous_damping: ArrayLike,
    frequency_tolerance: float = 0.01,
    damping_tolerance: float = 0.05,
    /,
):
    f = jnp.asarray(frequencies)
    d = jnp.asarray(damping)
    pf = jnp.asarray(previous_frequencies)
    pd = jnp.asarray(previous_damping)
    return (jnp.abs(f - pf) / jnp.maximum(jnp.abs(f), 1e-30) <= frequency_tolerance) & (
        jnp.abs(d - pd) <= damping_tolerance
    )


def modal_pairing_cost(
    mac: ArrayLike,
    frequency_reference: ArrayLike,
    frequency_candidate: ArrayLike,
    frequency_weight: float = 1.0,
    /,
):
    return (
        1
        - jnp.asarray(mac)
        + float(frequency_weight)
        * jnp.abs(
            jnp.asarray(frequency_reference)[:, None]
            - jnp.asarray(frequency_candidate)[None, :]
        )
        / jnp.maximum(jnp.asarray(frequency_reference)[:, None], 1e-30)
    )


__all__ = ["modal_pairing_cost", "stabilization_mask"]
