#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def rohsenow_boiling_heat_flux(
    superheat_k: ArrayLike,
    latent_heat_j_kg: float,
    density_liquid: float,
    density_vapor: float,
    surface_tension_n_m: float,
    viscosity_pa_s: float,
    heat_capacity_j_kg_k: float,
    prandtl: float,
    surface_coefficient: float = 0.013,
    /,
):
    gravity = 9.80665
    factor = (
        float(viscosity_pa_s)
        * float(latent_heat_j_kg)
        * (
            gravity
            * (float(density_liquid) - float(density_vapor))
            / float(surface_tension_n_m)
        )
        ** 0.5
    )
    return (
        factor
        * (
            float(heat_capacity_j_kg_k)
            * jnp.asarray(superheat_k)
            / (float(surface_coefficient) * float(latent_heat_j_kg) * float(prandtl))
        )
        ** 3
    )


def ablation_recession_rate(
    surface_heat_flux_w_m2: ArrayLike,
    effective_heat_of_ablation_j_kg: float,
    density_kg_m3: float,
    /,
):
    return jnp.asarray(surface_heat_flux_w_m2) / (
        float(effective_heat_of_ablation_j_kg) * float(density_kg_m3)
    )


__all__ = ["ablation_recession_rate", "rohsenow_boiling_heat_flux"]
