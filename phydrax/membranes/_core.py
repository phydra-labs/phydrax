#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pressure-, chemical-, and electric-potential driven membrane transport."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


def solution_diffusion_flux(
    permeance_mol_m2_s_pa: ArrayLike,
    feed_fugacity_pa: ArrayLike,
    permeate_fugacity_pa: ArrayLike,
    /,
) -> Array:
    return jnp.asarray(permeance_mol_m2_s_pa) * (
        jnp.asarray(feed_fugacity_pa) - jnp.asarray(permeate_fugacity_pa)
    )


def reverse_osmosis_flux(
    hydraulic_permeability_m_pa_s: float,
    pressure_difference_pa: ArrayLike,
    osmotic_pressure_difference_pa: ArrayLike,
    reflection_coefficient: float = 1.0,
    /,
) -> Array:
    if hydraulic_permeability_m_pa_s < 0.0 or not 0.0 <= reflection_coefficient <= 1.0:
        raise ValueError("RO permeability/reflection coefficient are outside bounds.")
    return float(hydraulic_permeability_m_pa_s) * (
        jnp.asarray(pressure_difference_pa)
        - float(reflection_coefficient) * jnp.asarray(osmotic_pressure_difference_pa)
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
    concentration = jnp.asarray(concentration_mol_m3)
    return (
        -float(diffusivity_m2_s) * jnp.asarray(concentration_gradient_mol_m4)
        + int(charge_number)
        * float(mobility_m2_v_s)
        * concentration[..., None]
        * jnp.asarray(electric_field_v_m)
        + concentration[..., None] * jnp.asarray(velocity_m_s)
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
