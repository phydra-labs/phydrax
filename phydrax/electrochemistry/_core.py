#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Electrochemical species transport, kinetics, and current distributions."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
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
    if (
        isinstance(electron_count, bool)
        or not isinstance(electron_count, int)
        or electron_count == 0
        or not isfinite(standard_potential_v)
    ):
        raise ValueError("Nernst electron count/potential are invalid.")
    temperature = jnp.asarray(temperature_k)
    quotient = jnp.asarray(reaction_quotient)
    temperature, quotient = jnp.broadcast_arrays(temperature, quotient)
    temperature = eqx.error_if(
        temperature,
        jnp.any(
            ~jnp.isfinite(temperature)
            | ~jnp.isfinite(quotient)
            | (temperature <= 0)
            | (quotient <= 0)
        ),
        "Nernst temperature and reaction quotient must be finite and positive.",
    )
    return float(standard_potential_v) - GAS_CONSTANT_J_MOL_K * temperature / (
        electron_count * FARADAY_C_MOL
    ) * jnp.log(quotient)


def butler_volmer_current_density(
    exchange_current_a_m2: ArrayLike,
    overpotential_v: ArrayLike,
    temperature_k: ArrayLike,
    electron_count: int = 1,
    transfer_coefficient: float = 0.5,
    /,
) -> Array:
    if (
        isinstance(electron_count, bool)
        or not isinstance(electron_count, int)
        or electron_count == 0
        or not isfinite(transfer_coefficient)
        or not 0 < transfer_coefficient < 1
    ):
        raise ValueError("Butler-Volmer electron count/transfer coefficient are invalid.")
    exchange = jnp.asarray(exchange_current_a_m2)
    eta = jnp.asarray(overpotential_v)
    temperature = jnp.asarray(temperature_k)
    exchange, eta, temperature = jnp.broadcast_arrays(exchange, eta, temperature)
    exchange = eqx.error_if(
        exchange,
        jnp.any(
            ~jnp.isfinite(exchange)
            | ~jnp.isfinite(eta)
            | ~jnp.isfinite(temperature)
            | (exchange < 0)
            | (temperature <= 0)
        ),
        "Butler-Volmer inputs must be finite with nonnegative exchange current and positive temperature.",
    )
    thermal = electron_count * FARADAY_C_MOL / (GAS_CONSTANT_J_MOL_K * temperature)
    return exchange * (
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
    if (
        not isfinite(diffusivity_m2_s)
        or diffusivity_m2_s < 0
        or not isfinite(mobility_m2_v_s)
        or mobility_m2_v_s < 0
        or isinstance(charge_number, bool)
        or not isinstance(charge_number, int)
    ):
        raise ValueError("Nernst-Planck transport coefficients are invalid.")
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
        or velocity.shape != expected
    ):
        raise ValueError("Nernst-Planck concentration/vector fields are incompatible.")
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


def porous_effective_property(
    bulk_property: ArrayLike,
    porosity: ArrayLike,
    bruggeman_exponent: float = 1.5,
    /,
) -> Array:
    if not isfinite(bruggeman_exponent) or bruggeman_exponent < 0:
        raise ValueError("Bruggeman exponent must be finite and nonnegative.")
    bulk = jnp.asarray(bulk_property)
    porosity_ = jnp.asarray(porosity)
    bulk, porosity_ = jnp.broadcast_arrays(bulk, porosity_)
    bulk = eqx.error_if(
        bulk,
        jnp.any(
            ~jnp.isfinite(bulk)
            | ~jnp.isfinite(porosity_)
            | (bulk < 0)
            | (porosity_ < 0)
            | (porosity_ > 1)
        ),
        "Porous property/porosity must be finite and nonnegative with porosity in [0, 1].",
    )
    return bulk * porosity_ ** float(bruggeman_exponent)


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
