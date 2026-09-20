#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def kwn_population_rate(
    number_density: ArrayLike,
    growth_velocity_m_s: ArrayLike,
    nucleation_rate_m3_s: ArrayLike,
    bin_widths_m: ArrayLike,
    /,
) -> Array:
    density = jnp.asarray(number_density)
    velocity = jnp.asarray(growth_velocity_m_s)
    widths = jnp.asarray(bin_widths_m)
    if density.shape != velocity.shape or density.shape != widths.shape:
        raise ValueError("KWN arrays must align.")
    faces = jnp.concatenate(
        (
            velocity[:1] * density[:1],
            0.5 * (velocity[:-1] * density[:-1] + velocity[1:] * density[1:]),
            velocity[-1:] * density[-1:],
        )
    )
    rate = -jnp.diff(faces) / widths
    return rate.at[0].add(jnp.asarray(nucleation_rate_m3_s) / widths[0])


__all__ = ["kwn_population_rate"]
