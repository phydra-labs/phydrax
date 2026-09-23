#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pressure-, chemical-, and electric-potential driven membrane transport."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


def solution_diffusion_flux(
    permeance_mol_m2_s_pa: ArrayLike,
    feed_fugacity_pa: ArrayLike,
    permeate_fugacity_pa: ArrayLike,
    /,
) -> Array:
    permeance = jnp.asarray(permeance_mol_m2_s_pa)
    feed = jnp.asarray(feed_fugacity_pa)
    permeate = jnp.asarray(permeate_fugacity_pa)
    permeance, feed, permeate = jnp.broadcast_arrays(permeance, feed, permeate)
    permeance = eqx.error_if(
        permeance,
        jnp.any(
            ~jnp.isfinite(permeance)
            | ~jnp.isfinite(feed)
            | ~jnp.isfinite(permeate)
            | (permeance < 0)
            | (feed < 0)
            | (permeate < 0)
        ),
        "Membrane permeance/fugacities must be finite and nonnegative.",
    )
    return permeance * (feed - permeate)


def reverse_osmosis_flux(
    hydraulic_permeability_m_pa_s: float,
    pressure_difference_pa: ArrayLike,
    osmotic_pressure_difference_pa: ArrayLike,
    reflection_coefficient: float = 1.0,
    /,
) -> Array:
    if (
        not isfinite(hydraulic_permeability_m_pa_s)
        or hydraulic_permeability_m_pa_s < 0.0
        or not isfinite(reflection_coefficient)
        or not 0.0 <= reflection_coefficient <= 1.0
    ):
        raise ValueError("RO permeability/reflection coefficient are outside bounds.")
    pressure = jnp.asarray(pressure_difference_pa)
    osmotic = jnp.asarray(osmotic_pressure_difference_pa)
    pressure, osmotic = jnp.broadcast_arrays(pressure, osmotic)
    pressure = eqx.error_if(
        pressure,
        jnp.any(~jnp.isfinite(pressure) | ~jnp.isfinite(osmotic)),
        "RO pressure differences must be finite.",
    )
    return float(hydraulic_permeability_m_pa_s) * (
        pressure - float(reflection_coefficient) * osmotic
    )


def nernst_planck_membrane_flux(
    concentration_mol_m3: ArrayLike,
    concentration_gradient_mol_m4: ArrayLike,
    electric_field_v_m: ArrayLike,
    diffusivity_m2_s: float,
    mobility_m2_v_s: float,
    charge_number: int,
    velocity_m_s: ArrayLike = 0.0,
    /,
) -> Array:
    if (
        not isfinite(diffusivity_m2_s)
        or diffusivity_m2_s < 0
        or not isfinite(mobility_m2_v_s)
        or mobility_m2_v_s < 0
        or isinstance(charge_number, bool)
        or not isinstance(charge_number, int)
    ):
        raise ValueError("Nernst-Planck transport parameters are invalid.")
    concentration = jnp.asarray(concentration_mol_m3)
    gradient = jnp.asarray(concentration_gradient_mol_m4)
    field = jnp.asarray(electric_field_v_m)
    velocity = jnp.asarray(velocity_m_s)
    if gradient.ndim == 0:
        raise ValueError("Nernst-Planck gradients require a component axis.")
    expected = concentration.shape + (gradient.shape[-1],)
    if (
        gradient.shape != expected
        or field.shape != expected
        or velocity.shape not in ((), expected)
    ):
        raise ValueError(
            "Nernst-Planck concentration and vector fields are incompatible."
        )
    concentration = eqx.error_if(
        concentration,
        jnp.any(
            ~jnp.isfinite(concentration)
            | ~jnp.isfinite(gradient)
            | ~jnp.isfinite(field)
            | ~jnp.isfinite(velocity)
            | (concentration < 0)
        ),
        "Nernst-Planck fields must be finite with nonnegative concentration.",
    )
    return (
        -float(diffusivity_m2_s) * gradient
        + int(charge_number) * float(mobility_m2_v_s) * concentration[..., None] * field
        + concentration[..., None] * velocity
    )


def membranes_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("membranes.solution-diffusion", "fugacity-permeance"),
        ("membranes.reverse-osmosis", "kedem-katchalsky-water"),
        ("membranes.electrodialysis", "nernst-planck-membrane"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("mass-conservation", "analytic-control", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "membranes_candidate_profiles",
    "nernst_planck_membrane_flux",
    "reverse_osmosis_flux",
    "solution_diffusion_flux",
]
