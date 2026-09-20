#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Chemo-mechanical swelling, stress-coupled potential, and degradation."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


def isotropic_chemical_strain(
    concentration: ArrayLike,
    reference_concentration: float,
    partial_molar_volume_m3_mol: float,
    dimension: int,
    /,
) -> Array:
    if dimension <= 0:
        raise ValueError("Chemical-strain dimension must be positive.")
    increment = (
        float(partial_molar_volume_m3_mol)
        * (jnp.asarray(concentration) - float(reference_concentration))
        / dimension
    )
    return increment[..., None, None] * jnp.eye(dimension)


def stress_coupled_chemical_potential(
    chemical_potential_j_mol: ArrayLike,
    stress_pa: ArrayLike,
    partial_molar_volume_m3_mol: float,
    /,
) -> Array:
    stress = jnp.asarray(stress_pa)
    hydrostatic = jnp.trace(stress, axis1=-2, axis2=-1) / stress.shape[-1]
    return (
        jnp.asarray(chemical_potential_j_mol)
        - float(partial_molar_volume_m3_mol) * hydrostatic
    )


def hydrogen_degraded_toughness(
    base_toughness_j_m2: float, occupancy: ArrayLike, degradation: float, /
) -> Array:
    occupied = jnp.clip(jnp.asarray(occupancy), 0.0, 1.0)
    return float(base_toughness_j_m2) * jnp.maximum(
        1.0 - float(degradation) * occupied, 0.0
    )


def isotropic_growth_tensor(growth_ratio: ArrayLike, dimension: int, /) -> Array:
    ratio = jnp.asarray(growth_ratio)
    if dimension <= 0:
        raise ValueError("Growth dimension must be positive.")
    return ratio[..., None, None] ** (1.0 / dimension) * jnp.eye(dimension)


def chemo_mechanics_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("chemo-mechanics.diffusion-stress", "isotropic-eigenstrain"),
        ("chemo-mechanics.stress-chemical-potential", "larche-cahn-hydrostatic"),
        ("chemo-mechanics.hydrogen-fracture", "occupancy-toughness"),
        ("chemo-mechanics.growth", "multiplicative-isotropic"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=(
                "thermodynamic-consistency",
                "analytic-control",
                "public-workflow",
            ),
        )
        for name, formulation in specs
    )


__all__ = [
    "chemo_mechanics_candidate_profiles",
    "hydrogen_degraded_toughness",
    "isotropic_chemical_strain",
    "isotropic_growth_tensor",
    "stress_coupled_chemical_potential",
]
