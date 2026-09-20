#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def rod_longitudinal_wavenumber(
    angular_frequency_rad_s: ArrayLike, young_modulus_pa: float, density_kg_m3: float, /
):
    return jnp.asarray(angular_frequency_rad_s) / jnp.sqrt(
        float(young_modulus_pa) / float(density_kg_m3)
    )


__all__ = ["rod_longitudinal_wavenumber"]
