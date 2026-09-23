#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hydrodynamic lubrication, point-contact elasticity, and wear."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
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


@dataclass(frozen=True, slots=True)
class ReynoldsPressureResult:
    pressure_pa: Array
    residual_norm: Array
    cavitation_respected: Array
    successful: Array


def reynolds_1d_pressure(
    film_thickness_m: ArrayLike,
    spacing_m: float,
    viscosity_pa_s: float,
    sliding_velocity_m_s: float,
    /,
    *,
    inlet_pressure_pa: float = 0.0,
    outlet_pressure_pa: float = 0.0,
) -> ReynoldsPressureResult:
    height = np.asarray(film_thickness_m, dtype=np.float64)
    if (
        height.ndim != 1
        or height.size < 3
        or not np.all(np.isfinite(height))
        or np.any(height <= 0.0)
    ):
        raise ValueError(
            "Film thickness must be a finite positive vector with interior points."
        )
    scalar_parameters = (
        spacing_m,
        viscosity_pa_s,
        sliding_velocity_m_s,
        inlet_pressure_pa,
        outlet_pressure_pa,
    )
    if (
        not all(isfinite(value) for value in scalar_parameters)
        or spacing_m <= 0.0
        or viscosity_pa_s <= 0.0
        or inlet_pressure_pa < 0.0
        or outlet_pressure_pa < 0.0
    ):
        raise ValueError(
            "Lubrication inputs must be finite with positive spacing/viscosity and nonnegative boundary pressure."
        )
    count = height.size
    matrix = np.zeros((count, count), dtype=np.float64)
    right = np.zeros((count,), dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[-1, -1] = 1.0
    right[0] = inlet_pressure_pa
    right[-1] = outlet_pressure_pa
    coefficient = height**3
    for index in range(1, count - 1):
        left = 0.5 * (coefficient[index - 1] + coefficient[index]) / spacing_m**2
        right_face = 0.5 * (coefficient[index] + coefficient[index + 1]) / spacing_m**2
        matrix[index, index - 1] = left
        matrix[index, index] = -(left + right_face)
        matrix[index, index + 1] = right_face
        right[index] = (
            6.0
            * viscosity_pa_s
            * sliding_velocity_m_s
            * (height[index + 1] - height[index - 1])
            / (2.0 * spacing_m)
        )
    space = ArraySpace((count,), dtype=jnp.float64)
    operator = DenseLinearOperator(jnp.asarray(matrix), source=space, target=space)
    result = solve(
        LinearSystem(operator), jnp.asarray(right), policy=LinearSolvePolicy(DenseLU())
    )
    pressure = result.value
    residual = operator.mv(pressure) - jnp.asarray(right)
    residual_norm = jnp.linalg.norm(residual)
    cavitation_respected = jnp.all(pressure >= 0)
    successful = result.successful & cavitation_respected & jnp.isfinite(residual_norm)
    return ReynoldsPressureResult(
        pressure,
        residual_norm,
        cavitation_respected,
        successful,
    )


def archard_wear_depth(
    normal_pressure_pa: ArrayLike,
    sliding_distance_m: ArrayLike,
    wear_coefficient: float,
    hardness_pa: float,
    /,
) -> Array:
    if (
        not isfinite(wear_coefficient)
        or wear_coefficient < 0.0
        or not isfinite(hardness_pa)
        or hardness_pa <= 0.0
    ):
        raise ValueError(
            "Wear coefficient must be finite/non-negative and hardness finite/positive."
        )
    pressure = jnp.asarray(normal_pressure_pa)
    distance = jnp.asarray(sliding_distance_m)
    pressure = eqx.error_if(
        pressure,
        jnp.any(
            ~jnp.isfinite(pressure)
            | ~jnp.isfinite(distance)
            | (pressure < 0)
            | (distance < 0)
        ),
        "Wear pressure and sliding distance must be finite and nonnegative.",
    )
    return float(wear_coefficient) * pressure * distance / float(hardness_pa)


def hertz_point_contact_radius(
    load_n: ArrayLike, sphere_radius_m: float, effective_modulus_pa: float, /
) -> Array:
    if (
        not isfinite(sphere_radius_m)
        or sphere_radius_m <= 0.0
        or not isfinite(effective_modulus_pa)
        or effective_modulus_pa <= 0.0
    ):
        raise ValueError("Hertz radius and modulus must be finite and positive.")
    load = jnp.asarray(load_n)
    load = eqx.error_if(
        load,
        jnp.any(~jnp.isfinite(load) | (load < 0)),
        "Hertz load must be finite and nonnegative.",
    )
    return (
        3.0 * load * float(sphere_radius_m) / (4.0 * float(effective_modulus_pa))
    ) ** (1.0 / 3.0)


def tribology_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("tribology.reynolds-lubrication", "one-dimensional-steady"),
        ("tribology.archard-wear", "pressure-distance"),
        ("tribology.hertz-point-contact", "elastic-sphere-halfspace"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("analytic-control", "load-balance", "refinement"),
        )
        for name, formulation in specs
    )


__all__ = [
    "archard_wear_depth",
    "hertz_point_contact_radius",
    "ReynoldsPressureResult",
    "reynolds_1d_pressure",
    "tribology_candidate_profiles",
]
