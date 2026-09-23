#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
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
    if (
        not isfinite(viscosity_pa_s)
        or viscosity_pa_s <= 0
        or not isfinite(density_kg_m3)
        or density_kg_m3 <= 0
        or not isfinite(particle_diameter_m)
        or particle_diameter_m <= 0
        or not isfinite(porosity)
        or not 0 < porosity < 1
    ):
        raise ValueError("Packed-bed properties are outside physical bounds.")
    v = jnp.asarray(velocity_m_s)
    v = eqx.error_if(
        v,
        jnp.any(~jnp.isfinite(v)),
        "Packed-bed velocity must be finite.",
    )
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
    if not isfinite(intrinsic_rate_s_inv) or intrinsic_rate_s_inv < 0:
        raise ValueError("Intrinsic catalyst rate must be finite and nonnegative.")
    concentration = jnp.asarray(bulk_concentration)
    effectiveness = jnp.asarray(effectiveness_factor)
    if concentration.shape != effectiveness.shape and effectiveness.shape != ():
        raise ValueError(
            "Catalyst effectiveness must be scalar or concentration-aligned."
        )
    concentration = eqx.error_if(
        concentration,
        jnp.any(
            ~jnp.isfinite(concentration)
            | ~jnp.isfinite(effectiveness)
            | (concentration < 0)
            | (effectiveness < 0)
            | (effectiveness > 1)
        ),
        "Catalyst concentration/effectiveness must be finite and physical.",
    )
    return effectiveness * float(intrinsic_rate_s_inv) * concentration


__all__ = ["catalyst_pellet_rate", "packed_bed_pressure_gradient"]
