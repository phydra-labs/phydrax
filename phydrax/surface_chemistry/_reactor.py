#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def packed_bed_pressure_gradient(
    velocity_m_s: ArrayLike,
    viscosity_pa_s: float,
    density_kg_m3: float,
    particle_diameter_m: float,
    porosity: float,
    /,
):
    v = jnp.asarray(velocity_m_s)
    return -(
        150
        * (1 - porosity) ** 2
        * float(viscosity_pa_s)
        * v
        / (porosity**3 * particle_diameter_m**2)
        + 1.75
        * (1 - porosity)
        * float(density_kg_m3)
        * v
        * jnp.abs(v)
        / (porosity**3 * particle_diameter_m)
    )


def catalyst_pellet_rate(
    bulk_concentration: ArrayLike,
    intrinsic_rate_s_inv: float,
    effectiveness_factor: ArrayLike,
    /,
):
    return (
        jnp.asarray(effectiveness_factor)
        * float(intrinsic_rate_s_inv)
        * jnp.asarray(bulk_concentration)
    )


__all__ = ["catalyst_pellet_rate", "packed_bed_pressure_gradient"]
