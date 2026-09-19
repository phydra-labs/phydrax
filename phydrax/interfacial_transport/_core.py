#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surfactant transport, dynamic wetting, and thin-film closures."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class LangmuirSurfactantLaw:
    clean_surface_tension_n_m: float
    gas_constant_j_mol_k: float
    temperature_k: float
    maximum_surface_concentration_mol_m2: float

    def surface_tension(self, surface_concentration_mol_m2: ArrayLike, /) -> Array:
        coverage = jnp.clip(
            jnp.asarray(surface_concentration_mol_m2)
            / self.maximum_surface_concentration_mol_m2,
            0.0,
            1.0 - 1.0e-12,
        )
        return (
            self.clean_surface_tension_n_m
            + self.gas_constant_j_mol_k
            * self.temperature_k
            * self.maximum_surface_concentration_mol_m2
            * jnp.log(1.0 - coverage)
        )


@dataclass(frozen=True, slots=True)
class AdsorptionKinetics:
    adsorption_rate_m_s: float
    desorption_rate_s_inv: float
    maximum_surface_concentration_mol_m2: float

    def rate(
        self,
        bulk_concentration_mol_m3: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        /,
    ) -> Array:
        bulk = jnp.asarray(bulk_concentration_mol_m3)
        surface = jnp.asarray(surface_concentration_mol_m2)
        return (
            self.adsorption_rate_m_s
            * bulk
            * (1.0 - surface / self.maximum_surface_concentration_mol_m2)
            - self.desorption_rate_s_inv * surface
        )


@dataclass(frozen=True, slots=True)
class CoxVoinovWettingLaw:
    equilibrium_angle_rad: float
    microscopic_length_m: float
    macroscopic_length_m: float

    def dynamic_angle(self, capillary_number: ArrayLike, /) -> Array:
        logarithm = jnp.log(self.macroscopic_length_m / self.microscopic_length_m)
        cube = (
            self.equilibrium_angle_rad**3
            + 9.0 * jnp.asarray(capillary_number) * logarithm
        )
        return jnp.cbrt(jnp.maximum(cube, 0.0))


def thin_film_pressure(
    height_m: ArrayLike,
    surface_tension_n_m: float,
    spacing_m: float,
    hamaker_j: float = 0.0,
    /,
) -> Array:
    height = jnp.asarray(height_m)
    curvature = (jnp.roll(height, -1) - 2.0 * height + jnp.roll(height, 1)) / spacing_m**2
    disjoining = -float(hamaker_j) / (6.0 * jnp.pi * jnp.maximum(height, 1.0e-12) ** 3)
    return -float(surface_tension_n_m) * curvature + disjoining


def interfacial_transport_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("interfacial-transport.langmuir-surfactant", {"state": "surface-concentration"}),
        ("interfacial-transport.adsorption", {"kinetics": "langmuir"}),
        ("interfacial-transport.dynamic-wetting", {"law": "cox-voinov"}),
        ("interfacial-transport.thin-film", {"pressure": "capillary-disjoining"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=(
                "analytic-control",
                "surface-conservation",
                "public-workflow",
            ),
        )
        for name, attrs in specs
    )


__all__ = [
    "AdsorptionKinetics",
    "CoxVoinovWettingLaw",
    "LangmuirSurfactantLaw",
    "interfacial_transport_candidate_profiles",
    "thin_film_pressure",
]
