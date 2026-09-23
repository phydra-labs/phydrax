#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surfactant transport, dynamic wetting, and thin-film closures."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class LangmuirSurfactantLaw:
    clean_surface_tension_n_m: float
    gas_constant_j_mol_k: float
    temperature_k: float
    maximum_surface_concentration_mol_m2: float

    def __post_init__(self):
        if (
            not isfinite(self.clean_surface_tension_n_m)
            or self.clean_surface_tension_n_m <= 0
            or not isfinite(self.gas_constant_j_mol_k)
            or self.gas_constant_j_mol_k <= 0
            or not isfinite(self.temperature_k)
            or self.temperature_k <= 0
            or not isfinite(self.maximum_surface_concentration_mol_m2)
            or self.maximum_surface_concentration_mol_m2 <= 0
        ):
            raise ValueError(
                "Langmuir surfactant parameters are outside physical bounds."
            )

    def surface_tension(self, surface_concentration_mol_m2: ArrayLike, /) -> Array:
        concentration = jnp.asarray(surface_concentration_mol_m2)
        concentration = eqx.error_if(
            concentration,
            jnp.any(
                ~jnp.isfinite(concentration)
                | (concentration < 0)
                | (concentration >= self.maximum_surface_concentration_mol_m2)
            ),
            "Surface concentration must be finite and below Langmuir capacity.",
        )
        coverage = concentration / self.maximum_surface_concentration_mol_m2
        tension = (
            self.clean_surface_tension_n_m
            + self.gas_constant_j_mol_k
            * self.temperature_k
            * self.maximum_surface_concentration_mol_m2
            * jnp.log(1.0 - coverage)
        )
        return eqx.error_if(
            tension,
            jnp.any(~jnp.isfinite(tension) | (tension <= 0)),
            "Langmuir state produces nonpositive surface tension.",
        )


@dataclass(frozen=True, slots=True)
class AdsorptionKinetics:
    adsorption_rate_m_s: float
    desorption_rate_s_inv: float
    maximum_surface_concentration_mol_m2: float

    def __post_init__(self):
        if (
            not isfinite(self.adsorption_rate_m_s)
            or self.adsorption_rate_m_s < 0
            or not isfinite(self.desorption_rate_s_inv)
            or self.desorption_rate_s_inv < 0
            or not isfinite(self.maximum_surface_concentration_mol_m2)
            or self.maximum_surface_concentration_mol_m2 <= 0
        ):
            raise ValueError("Adsorption kinetics are outside physical bounds.")

    def rate(
        self,
        bulk_concentration_mol_m3: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        /,
    ) -> Array:
        bulk = jnp.asarray(bulk_concentration_mol_m3)
        surface = jnp.asarray(surface_concentration_mol_m2)
        if bulk.shape != surface.shape:
            raise ValueError("Bulk and surface concentrations must be aligned.")
        surface = eqx.error_if(
            surface,
            jnp.any(
                ~jnp.isfinite(bulk)
                | ~jnp.isfinite(surface)
                | (bulk < 0)
                | (surface < 0)
                | (surface > self.maximum_surface_concentration_mol_m2)
            ),
            "Adsorption concentrations must be finite and within capacity.",
        )
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

    def __post_init__(self):
        if (
            not isfinite(self.equilibrium_angle_rad)
            or not 0 < self.equilibrium_angle_rad < pi
            or not isfinite(self.microscopic_length_m)
            or self.microscopic_length_m <= 0
            or not isfinite(self.macroscopic_length_m)
            or self.macroscopic_length_m <= self.microscopic_length_m
        ):
            raise ValueError("Cox-Voinov wetting parameters are outside physical bounds.")

    def dynamic_angle(self, capillary_number: ArrayLike, /) -> Array:
        capillary = jnp.asarray(capillary_number)
        logarithm = jnp.log(self.macroscopic_length_m / self.microscopic_length_m)
        cube = self.equilibrium_angle_rad**3 + 9.0 * capillary * logarithm
        cube = eqx.error_if(
            cube,
            jnp.any(
                ~jnp.isfinite(capillary)
                | ~jnp.isfinite(cube)
                | (cube < 0)
                | (cube >= pi**3)
            ),
            "Capillary number produces an invalid Cox-Voinov angle.",
        )
        return jnp.cbrt(cube)


def thin_film_pressure(
    height_m: ArrayLike,
    surface_tension_n_m: float,
    spacing_m: float,
    hamaker_j: float = 0.0,
    /,
) -> Array:
    if (
        not isfinite(surface_tension_n_m)
        or surface_tension_n_m <= 0
        or not isfinite(spacing_m)
        or spacing_m <= 0
        or not isfinite(hamaker_j)
    ):
        raise ValueError("Thin-film material and spacing parameters are invalid.")
    height = jnp.asarray(height_m)
    height = eqx.error_if(
        height,
        jnp.any(~jnp.isfinite(height) | (height <= 0)),
        "Thin-film height must be finite and positive.",
    )
    curvature = (jnp.roll(height, -1) - 2.0 * height + jnp.roll(height, 1)) / spacing_m**2
    disjoining = -float(hamaker_j) / (6.0 * jnp.pi * height**3)
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
