#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surface coverage kinetics and porous-catalyst effectiveness."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
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
    if (
        not isfinite(adsorption_rate_m_s)
        or adsorption_rate_m_s < 0
        or not isfinite(desorption_rate_s_inv)
        or desorption_rate_s_inv < 0
        or not isfinite(maximum_site_density_mol_m2)
        or maximum_site_density_mol_m2 <= 0
    ):
        raise ValueError("Langmuir kinetic parameters are outside physical bounds.")
    occupied = jnp.asarray(coverage)
    concentration = jnp.asarray(gas_concentration_mol_m3)
    occupied = eqx.error_if(
        occupied,
        jnp.any(
            ~jnp.isfinite(occupied)
            | ~jnp.isfinite(concentration)
            | (occupied < 0)
            | (occupied > 1)
            | (concentration < 0)
        ),
        "Langmuir concentration/coverage must be finite and physical.",
    )
    return float(adsorption_rate_m_s) * concentration * (1.0 - occupied) - float(
        desorption_rate_s_inv
    ) * occupied * float(maximum_site_density_mol_m2)


def langmuir_hinshelwood_rate(
    coverage_a: ArrayLike, coverage_b: ArrayLike, rate_constant_mol_m2_s: float, /
) -> Array:
    if not isfinite(rate_constant_mol_m2_s) or rate_constant_mol_m2_s < 0:
        raise ValueError("Surface reaction rate constant must be finite and nonnegative.")
    a = jnp.asarray(coverage_a)
    b = jnp.asarray(coverage_b)
    if a.shape != b.shape:
        raise ValueError("Surface coverages must have matching shapes.")
    a = eqx.error_if(
        a,
        jnp.any(~jnp.isfinite(a) | ~jnp.isfinite(b) | (a < 0) | (b < 0) | (a + b > 1)),
        "Surface coverages must be finite, nonnegative, and share available sites.",
    )
    return float(rate_constant_mol_m2_s) * a * b


def spherical_pellet_effectiveness_factor(thiele_modulus: ArrayLike, /) -> Array:
    phi = jnp.asarray(thiele_modulus)
    phi = eqx.error_if(
        phi,
        jnp.any(~jnp.isfinite(phi) | (phi < 0)),
        "Thiele modulus must be finite and nonnegative.",
    )
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
