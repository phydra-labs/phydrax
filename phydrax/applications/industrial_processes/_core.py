#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded welding, casting, forming, cure, machining, and process-TCAD models."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...qualification import CapabilityProfile, SupportTuple


def goldak_double_ellipsoid_source(
    coordinates_m: ArrayLike,
    center_m: ArrayLike,
    power_w: float,
    efficiency: float,
    front_length_m: float,
    rear_length_m: float,
    half_width_m: float,
    depth_m: float,
    /,
) -> Array:
    point = jnp.asarray(coordinates_m) - jnp.asarray(center_m)
    x, y, z = point[..., 0], point[..., 1], point[..., 2]
    length = jnp.where(x >= 0.0, float(front_length_m), float(rear_length_m))
    fraction = jnp.where(
        x >= 0.0,
        2.0 * front_length_m / (front_length_m + rear_length_m),
        2.0 * rear_length_m / (front_length_m + rear_length_m),
    )
    normalization = (
        6.0
        * jnp.sqrt(3.0)
        * fraction
        * float(efficiency)
        * float(power_w)
        / (jnp.pi * jnp.sqrt(jnp.pi) * length * float(half_width_m) * float(depth_m))
    )
    return normalization * jnp.exp(
        -3.0
        * (
            x**2 / length**2
            + y**2 / float(half_width_m) ** 2
            + z**2 / float(depth_m) ** 2
        )
    )


def casting_solid_fraction(
    temperature_k: ArrayLike, solidus_k: float, liquidus_k: float, /
) -> Array:
    if not solidus_k < liquidus_k:
        raise ValueError("Casting solidus must be below liquidus.")
    return jnp.clip(
        (float(liquidus_k) - jnp.asarray(temperature_k))
        / (float(liquidus_k) - float(solidus_k)),
        0.0,
        1.0,
    )


def von_mises_equivalent(stress_pa: ArrayLike, /) -> Array:
    stress = jnp.asarray(stress_pa)
    dimension = stress.shape[-1]
    deviator = (
        stress
        - jnp.trace(stress, axis1=-2, axis2=-1)[..., None, None]
        * jnp.eye(dimension)
        / dimension
    )
    return jnp.sqrt(1.5 * jnp.sum(deviator * deviator, axis=(-2, -1)))


def kamal_sourour_cure_rate(
    degree: ArrayLike,
    temperature_k: ArrayLike,
    first_prefactor_s_inv: float,
    second_prefactor_s_inv: float,
    first_activation_j_mol: float,
    second_activation_j_mol: float,
    autocatalytic_exponent: float,
    remaining_exponent: float,
    /,
) -> Array:
    gas_constant = 8.314462618
    alpha = jnp.clip(jnp.asarray(degree), 0.0, 1.0)
    temperature = jnp.asarray(temperature_k)
    first = float(first_prefactor_s_inv) * jnp.exp(
        -float(first_activation_j_mol) / (gas_constant * temperature)
    )
    second = float(second_prefactor_s_inv) * jnp.exp(
        -float(second_activation_j_mol) / (gas_constant * temperature)
    )
    return (first + second * alpha ** float(autocatalytic_exponent)) * (
        1.0 - alpha
    ) ** float(remaining_exponent)


def machining_cutting_power(
    cutting_force_n: ArrayLike, cutting_velocity_m_s: ArrayLike, /
) -> Array:
    return jnp.asarray(cutting_force_n) * jnp.asarray(cutting_velocity_m_s)


def deal_grove_oxide_thickness(
    time_s: ArrayLike,
    linear_rate_m_s: float,
    parabolic_rate_m2_s: float,
    initial_thickness_m: float = 0.0,
    /,
) -> Array:
    time = jnp.asarray(time_s)
    linear = float(parabolic_rate_m2_s) / float(linear_rate_m_s)
    shifted = float(initial_thickness_m)
    return 0.5 * (
        -linear
        + jnp.sqrt(
            (linear + 2.0 * shifted) ** 2 + 4.0 * float(parabolic_rate_m2_s) * time
        )
    )


def industrial_process_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("welding.goldak-source", "double-ellipsoid"),
        ("casting.solidification", "linear-mushy-fraction"),
        ("forming.von-mises", "finite-stress-evidence"),
        ("composite-manufacturing.cure", "kamal-sourour"),
        ("machining.cutting-power", "force-times-speed"),
        ("semiconductor-process.oxidation", "deal-grove"),
        ("heat-treatment.phase-transformation", "materials-owned"),
        ("polymer-processing.rheology", "rheology-owned"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("analytic-control", "process-ledger", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "casting_solid_fraction",
    "deal_grove_oxide_thickness",
    "goldak_double_ellipsoid_source",
    "industrial_process_candidate_profiles",
    "kamal_sourour_cure_rate",
    "machining_cutting_power",
    "von_mises_equivalent",
]
