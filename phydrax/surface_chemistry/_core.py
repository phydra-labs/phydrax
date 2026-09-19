#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surface coverage kinetics and porous-catalyst effectiveness."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


def langmuir_adsorption_rate(
    gas_concentration_mol_m3: ArrayLike,
    coverage: ArrayLike,
    adsorption_rate_m_s: float,
    desorption_rate_s_inv: float,
    maximum_site_density_mol_m2: float,
    /,
) -> Array:
    occupied = jnp.asarray(coverage)
    return float(adsorption_rate_m_s) * jnp.asarray(gas_concentration_mol_m3) * (
        1.0 - occupied
    ) - float(desorption_rate_s_inv) * occupied * float(maximum_site_density_mol_m2)


def langmuir_hinshelwood_rate(
    coverage_a: ArrayLike, coverage_b: ArrayLike, rate_constant_mol_m2_s: float, /
) -> Array:
    return (
        float(rate_constant_mol_m2_s) * jnp.asarray(coverage_a) * jnp.asarray(coverage_b)
    )


def spherical_pellet_effectiveness_factor(thiele_modulus: ArrayLike, /) -> Array:
    phi = jnp.asarray(thiele_modulus)
    safe = jnp.maximum(phi, 1.0e-12)
    finite = 3.0 / safe * (1.0 / jnp.tanh(safe) - 1.0 / safe)
    return jnp.where(phi < 1.0e-6, 1.0 - phi**2 / 15.0, finite)


def surface_chemistry_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("surface-chemistry.langmuir", "adsorption-desorption"),
        ("surface-chemistry.langmuir-hinshelwood", "bimolecular-surface-reaction"),
        ("surface-chemistry.pellet-effectiveness", "spherical-first-order"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("site-balance", "analytic-control", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "langmuir_adsorption_rate",
    "langmuir_hinshelwood_rate",
    "spherical_pellet_effectiveness_factor",
    "surface_chemistry_candidate_profiles",
]
