#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Electrochemical species transport, kinetics, and current distributions."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


FARADAY_C_MOL = 96485.33212
GAS_CONSTANT_J_MOL_K = 8.314462618


def nernst_equilibrium_potential(
    standard_potential_v: float,
    temperature_k: ArrayLike,
    reaction_quotient: ArrayLike,
    electron_count: int,
    /,
) -> Array:
    if electron_count == 0:
        raise ValueError("Electron count cannot be zero.")
    return float(standard_potential_v) - GAS_CONSTANT_J_MOL_K * jnp.asarray(
        temperature_k
    ) / (electron_count * FARADAY_C_MOL) * jnp.log(jnp.asarray(reaction_quotient))


def butler_volmer_current_density(
    exchange_current_a_m2: ArrayLike,
    overpotential_v: ArrayLike,
    temperature_k: ArrayLike,
    electron_count: int = 1,
    transfer_coefficient: float = 0.5,
    /,
) -> Array:
    thermal = (
        electron_count
        * FARADAY_C_MOL
        / (GAS_CONSTANT_J_MOL_K * jnp.asarray(temperature_k))
    )
    eta = jnp.asarray(overpotential_v)
    return jnp.asarray(exchange_current_a_m2) * (
        jnp.exp(float(transfer_coefficient) * thermal * eta)
        - jnp.exp(-(1.0 - float(transfer_coefficient)) * thermal * eta)
    )


def nernst_planck_flux(
    concentration_mol_m3: ArrayLike,
    concentration_gradient_mol_m4: ArrayLike,
    electric_field_v_m: ArrayLike,
    velocity_m_s: ArrayLike,
    diffusivity_m2_s: float,
    mobility_m2_v_s: float,
    charge_number: int,
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


def porous_effective_property(
    bulk_property: ArrayLike, porosity: ArrayLike, bruggeman_exponent: float = 1.5, /
) -> Array:
    return jnp.asarray(bulk_property) * jnp.asarray(porosity) ** float(bruggeman_exponent)


def electrochemistry_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("electrochemistry.nernst", "ideal-activities"),
        ("electrochemistry.butler-volmer", "single-reaction"),
        ("electrochemistry.nernst-planck", "dilute-transport"),
        ("electrochemistry.porous-electrode", "bruggeman-property"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("charge-balance", "species-balance", "analytic-control"),
        )
        for name, formulation in specs
    )


__all__ = [
    "FARADAY_C_MOL",
    "GAS_CONSTANT_J_MOL_K",
    "butler_volmer_current_density",
    "electrochemistry_candidate_profiles",
    "nernst_equilibrium_potential",
    "nernst_planck_flux",
    "porous_effective_property",
]
