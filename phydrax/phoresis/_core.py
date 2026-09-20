#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reduced phoretic particle mobility and force laws."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


def smoluchowski_electrophoretic_velocity(
    electric_field_v_m: ArrayLike,
    permittivity_f_m: float,
    zeta_potential_v: float,
    viscosity_pa_s: float,
    /,
) -> Array:
    if viscosity_pa_s <= 0.0 or permittivity_f_m <= 0.0:
        raise ValueError("Electrophoretic permittivity and viscosity must be positive.")
    mobility = float(permittivity_f_m) * float(zeta_potential_v) / float(viscosity_pa_s)
    return mobility * jnp.asarray(electric_field_v_m)


def dielectrophoretic_force(
    electric_intensity_gradient_v2_m3: ArrayLike,
    radius_m: float,
    medium_permittivity_f_m: float,
    clausius_mossotti_real: ArrayLike,
    /,
) -> Array:
    if radius_m <= 0.0 or medium_permittivity_f_m <= 0.0:
        raise ValueError("DEP radius and medium permittivity must be positive.")
    coefficient = 2.0 * jnp.pi * float(medium_permittivity_f_m) * float(radius_m) ** 3
    return (
        coefficient
        * jnp.asarray(clausius_mossotti_real)
        * jnp.asarray(electric_intensity_gradient_v2_m3)
    )


def thermophoretic_velocity(
    temperature_gradient_k_m: ArrayLike, thermophoretic_mobility_m2_s_k: float, /
) -> Array:
    return -float(thermophoretic_mobility_m2_s_k) * jnp.asarray(temperature_gradient_k_m)


def diffusiophoretic_velocity(
    log_concentration_gradient_m_inv: ArrayLike, diffusiophoretic_mobility_m2_s: float, /
) -> Array:
    return float(diffusiophoretic_mobility_m2_s) * jnp.asarray(
        log_concentration_gradient_m_inv
    )


def acoustic_radiation_force(
    potential_gradient_j_m: ArrayLike, particle_volume_m3: float, /
) -> Array:
    if particle_volume_m3 <= 0.0:
        raise ValueError("Particle volume must be positive.")
    return -float(particle_volume_m3) * jnp.asarray(potential_gradient_j_m)


def magnetophoretic_force(
    magnetic_energy_gradient_j_m: ArrayLike,
    particle_volume_m3: float,
    susceptibility_contrast: float,
    /,
) -> Array:
    if particle_volume_m3 <= 0.0:
        raise ValueError("Particle volume must be positive.")
    return (
        float(particle_volume_m3)
        * float(susceptibility_contrast)
        * jnp.asarray(magnetic_energy_gradient_j_m)
    )


def phoresis_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    names = (
        "phoresis.electrophoresis",
        "phoresis.dielectrophoresis",
        "phoresis.thermophoresis",
        "phoresis.diffusiophoresis",
        "phoresis.acoustophoresis",
        "phoresis.magnetophoresis",
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"regime": "reduced-dilute-sphere"}),),
            required_gates=("analytic-mobility", "force-balance", "support-refusal"),
        )
        for name in names
    )


__all__ = [
    "acoustic_radiation_force",
    "dielectrophoretic_force",
    "diffusiophoretic_velocity",
    "magnetophoretic_force",
    "phoresis_candidate_profiles",
    "smoluchowski_electrophoretic_velocity",
    "thermophoretic_velocity",
]
