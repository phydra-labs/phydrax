#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def rod_longitudinal_wavenumber(
    angular_frequency_rad_s: ArrayLike, young_modulus_pa: float, density_kg_m3: float, /
):
    if (
        not isfinite(young_modulus_pa)
        or young_modulus_pa <= 0
        or not isfinite(density_kg_m3)
        or density_kg_m3 <= 0
    ):
        raise ValueError("Rod modulus and density must be finite and positive.")
    frequency = jnp.asarray(angular_frequency_rad_s)
    frequency = eqx.error_if(
        frequency,
        jnp.any(~jnp.isfinite(frequency) | (frequency < 0)),
        "Angular frequency must be finite and nonnegative.",
    )
    return frequency / jnp.sqrt(float(young_modulus_pa) / float(density_kg_m3))


__all__ = ["rod_longitudinal_wavenumber"]
