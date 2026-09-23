#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def stress_coupled_diffusive_flux(
    concentration_gradient: ArrayLike,
    hydrostatic_stress_gradient: ArrayLike,
    mobility: float,
    partial_molar_volume: float,
    /,
):
    if not isfinite(mobility) or mobility < 0 or not isfinite(partial_molar_volume):
        raise ValueError(
            "Diffusive mobility must be nonnegative and coefficients finite."
        )
    concentration = jnp.asarray(concentration_gradient)
    stress = jnp.asarray(hydrostatic_stress_gradient)
    if concentration.shape != stress.shape:
        raise ValueError("Concentration and stress gradients must have matching shapes.")
    concentration = eqx.error_if(
        concentration,
        jnp.any(~jnp.isfinite(concentration) | ~jnp.isfinite(stress)),
        "Diffusive gradients must be finite.",
    )
    return -float(mobility) * (concentration - float(partial_molar_volume) * stress)


def phase_field_fracture_driving_energy(
    elastic_energy: ArrayLike, toughness: ArrayLike, length_scale: float, /
):
    if not isfinite(length_scale) or length_scale <= 0:
        raise ValueError("Fracture length scale must be finite and positive.")
    toughness_ = jnp.asarray(toughness)
    energy = jnp.asarray(elastic_energy)
    toughness_ = eqx.error_if(
        toughness_,
        jnp.any(~jnp.isfinite(toughness_) | (toughness_ <= 0)),
        "Fracture toughness must be finite and positive.",
    )
    energy = eqx.error_if(
        energy,
        jnp.any(~jnp.isfinite(energy) | (energy < 0)),
        "Elastic energy must be finite and nonnegative.",
    )
    return 2 * float(length_scale) * energy / toughness_


__all__ = ["phase_field_fracture_driving_energy", "stress_coupled_diffusive_flux"]
