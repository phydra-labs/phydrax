#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hydrodynamic lubrication, point-contact elasticity, and wear."""

from __future__ import annotations

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


def reynolds_1d_pressure(
    film_thickness_m: ArrayLike,
    spacing_m: float,
    viscosity_pa_s: float,
    sliding_velocity_m_s: float,
    /,
    *,
    inlet_pressure_pa: float = 0.0,
    outlet_pressure_pa: float = 0.0,
) -> Array:
    height = np.asarray(film_thickness_m, dtype=float)
    if height.ndim != 1 or height.size < 3 or np.any(height <= 0.0):
        raise ValueError("Film thickness must be a positive vector with interior points.")
    if spacing_m <= 0.0 or viscosity_pa_s <= 0.0:
        raise ValueError("Lubrication spacing and viscosity must be positive.")
    count = height.size
    matrix = np.zeros((count, count), dtype=float)
    right = np.zeros((count,), dtype=float)
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
    return jnp.maximum(result.value, 0.0)


def archard_wear_depth(
    normal_pressure_pa: ArrayLike,
    sliding_distance_m: ArrayLike,
    wear_coefficient: float,
    hardness_pa: float,
    /,
) -> Array:
    if wear_coefficient < 0.0 or hardness_pa <= 0.0:
        raise ValueError("Wear coefficient must be non-negative and hardness positive.")
    return (
        float(wear_coefficient)
        * jnp.asarray(normal_pressure_pa)
        * jnp.asarray(sliding_distance_m)
        / float(hardness_pa)
    )


def hertz_point_contact_radius(
    load_n: ArrayLike, sphere_radius_m: float, effective_modulus_pa: float, /
) -> Array:
    if sphere_radius_m <= 0.0 or effective_modulus_pa <= 0.0:
        raise ValueError("Hertz radius and modulus must be positive.")
    return (
        3.0
        * jnp.asarray(load_n)
        * float(sphere_radius_m)
        / (4.0 * float(effective_modulus_pa))
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
    "reynolds_1d_pressure",
    "tribology_candidate_profiles",
]
