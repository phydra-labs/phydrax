#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def stress_coupled_diffusive_flux(
    concentration_gradient: ArrayLike,
    hydrostatic_stress_gradient: ArrayLike,
    mobility: float,
    partial_molar_volume: float,
    /,
):
    return -float(mobility) * (
        jnp.asarray(concentration_gradient)
        - float(partial_molar_volume) * jnp.asarray(hydrostatic_stress_gradient)
    )


def phase_field_fracture_driving_energy(
    elastic_energy: ArrayLike, toughness: ArrayLike, length_scale: float, /
):
    return 2 * float(length_scale) * jnp.asarray(elastic_energy) / jnp.asarray(toughness)


__all__ = ["phase_field_fracture_driving_energy", "stress_coupled_diffusive_flux"]
