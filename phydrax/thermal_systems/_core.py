#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surface radiation, phase change, and compact thermal-system models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
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


@dataclass(frozen=True, slots=True)
class EnclosureRadiosityResult:
    radiosity_w_m2: Array
    outward_heat_flux_w_m2: Array
    residual_norm: Array
    successful: Array


def enclosure_radiosity(
    temperature_k: ArrayLike, emissivity: ArrayLike, view_factors: ArrayLike, /
) -> EnclosureRadiosityResult:
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
    temperature = eqx.error_if(
        temperature,
        jnp.any(
            ~jnp.isfinite(temperature)
            | ~jnp.isfinite(emissivity_)
            | ~jnp.isfinite(factors)
            | (temperature <= 0)
            | (emissivity_ <= 0)
            | (emissivity_ > 1)
            | (factors < 0)
        )
        | jnp.any(jnp.abs(jnp.sum(factors, axis=1) - 1) > 1e-10),
        "Enclosure data must be finite, physical, and have normalized view-factor rows.",
    )
    matrix = jnp.eye(count) - (1.0 - emissivity_)[:, None] * factors
    emitted = emissivity_ * STEFAN_BOLTZMANN_W_M2_K4 * temperature**4
    space = ArraySpace((count,), dtype=temperature.dtype)
    solved = solve(
        LinearSystem(DenseLinearOperator(matrix, source=space, target=space)),
        emitted,
        policy=LinearSolvePolicy(DenseLU()),
    )
    irradiation = factors @ solved.value
    residual_norm = jnp.linalg.norm(matrix @ solved.value - emitted)
    successful = (
        solved.successful
        & jnp.all(jnp.isfinite(solved.value))
        & jnp.isfinite(residual_norm)
    )
    return EnclosureRadiosityResult(
        solved.value,
        solved.value - irradiation,
        residual_norm,
        successful,
    )


def stefan_front_position(
    time_s: ArrayLike, diffusivity_m2_s: float, similarity_parameter: float, /
) -> Array:
    if (
        not isfinite(diffusivity_m2_s)
        or diffusivity_m2_s <= 0.0
        or not isfinite(similarity_parameter)
        or similarity_parameter < 0.0
    ):
        raise ValueError(
            "Stefan diffusivity must be finite/positive and similarity parameter finite/non-negative."
        )
    time = jnp.asarray(time_s)
    time = eqx.error_if(
        time,
        jnp.any(~jnp.isfinite(time) | (time < 0)),
        "Stefan time must be finite and nonnegative.",
    )
    return 2.0 * float(similarity_parameter) * jnp.sqrt(float(diffusivity_m2_s) * time)


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
    "EnclosureRadiosityResult",
    "enclosure_radiosity",
    "heat_pipe_capillary_margin",
    "stefan_front_position",
    "thermal_system_candidate_profiles",
]
