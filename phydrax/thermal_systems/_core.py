#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surface radiation, phase change, and compact thermal-system models."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ..qualification import CapabilityProfile, SupportTuple


STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


def enclosure_radiosity(
    temperature_k: ArrayLike, emissivity: ArrayLike, view_factors: ArrayLike, /
) -> tuple[Array, Array]:
    temperature = jnp.asarray(temperature_k)
    emissivity_ = jnp.asarray(emissivity)
    factors = jnp.asarray(view_factors)
    count = temperature.size
    if (
        temperature.shape != (count,)
        or emissivity_.shape != (count,)
        or factors.shape != (count, count)
    ):
        raise ValueError(
            "Enclosure temperatures, emissivities, and view factors do not align."
        )
    matrix = jnp.eye(count) - (1.0 - emissivity_)[:, None] * factors
    emitted = emissivity_ * STEFAN_BOLTZMANN_W_M2_K4 * temperature**4
    space = ArraySpace((count,), dtype=temperature.dtype)
    radiosity = solve(
        LinearSystem(DenseLinearOperator(matrix, source=space, target=space)),
        emitted,
        policy=LinearSolvePolicy(DenseLU()),
    ).value
    irradiation = factors @ radiosity
    return radiosity, radiosity - irradiation


def stefan_front_position(
    time_s: ArrayLike, diffusivity_m2_s: float, similarity_parameter: float, /
) -> Array:
    if diffusivity_m2_s <= 0.0 or similarity_parameter < 0.0:
        raise ValueError(
            "Stefan diffusivity must be positive and similarity parameter non-negative."
        )
    return (
        2.0
        * float(similarity_parameter)
        * jnp.sqrt(float(diffusivity_m2_s) * jnp.asarray(time_s))
    )


def heat_pipe_capillary_margin(
    capillary_pressure_pa: ArrayLike,
    liquid_pressure_drop_pa: ArrayLike,
    vapor_pressure_drop_pa: ArrayLike,
    gravity_pressure_drop_pa: ArrayLike,
    /,
) -> Array:
    return (
        jnp.asarray(capillary_pressure_pa)
        - jnp.asarray(liquid_pressure_drop_pa)
        - jnp.asarray(vapor_pressure_drop_pa)
        - jnp.asarray(gravity_pressure_drop_pa)
    )


def thermal_system_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("thermal.surface-radiation", "gray-diffuse-enclosure"),
        ("thermal.phase-change", "one-phase-stefan-control"),
        ("thermal.heat-pipe", "capillary-limit-balance"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("analytic-control", "energy-conservation", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "STEFAN_BOLTZMANN_W_M2_K4",
    "enclosure_radiosity",
    "heat_pipe_capillary_margin",
    "stefan_front_position",
    "thermal_system_candidate_profiles",
]
