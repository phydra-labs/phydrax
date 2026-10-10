#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surfactant equation of state, adsorption kinetics and dynamic wetting laws.

Physical coefficients are ``parameter_field`` leaves, so they can be inferred
or differentiated; constructors validate concrete values on the host. Every
surfactant law describes one interface. Symmetric soap films apply the factor
two for their two interfaces explicitly in the film routes, never here.
"""

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import xlogy
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import parameter_field, ParameterOwner
from ..qualification import CapabilityProfile, SupportTuple


# Exact SI molar gas constant N_A k_B (2019 SI).
_GAS_CONSTANT_J_MOL_K = 6.02214076e23 * 1.380649e-23


def _host_scalar(value: ArrayLike, name: str, /, *, positive: bool) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not isfinite(float(host)):
        raise ValueError(f"{name} must be a finite scalar.")
    if (float(host) <= 0.0) if positive else (float(host) < 0.0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {bound}.")
    return jnp.asarray(host)


class SurfactantStateEvaluation(StrictModule):
    """Unchecked Langmuir evaluation with its admissibility mask.

    Solvers use this form so an inadmissible trial state becomes a reported
    status instead of a runtime error; ``admissible`` is ``0 <= Gamma <
    Gamma_inf`` with finite positive tension.
    """

    surface_tension_n_m: Array
    tension_derivative: Array
    gibbs_elasticity_n_m: Array
    admissible: Array

    def __init__(
        self,
        surface_tension_n_m: Array,
        tension_derivative: Array,
        gibbs_elasticity_n_m: Array,
        admissible: Array,
        /,
    ) -> None:
        self.surface_tension_n_m = jnp.asarray(surface_tension_n_m)
        self.tension_derivative = jnp.asarray(tension_derivative)
        self.gibbs_elasticity_n_m = jnp.asarray(gibbs_elasticity_n_m)
        self.admissible = jnp.asarray(admissible, dtype=jnp.bool_)


class LangmuirSurfactantLaw(StrictModule, ParameterOwner):
    """Szyszkowski--Langmuir equation of state for one interface.

    ``sigma(Gamma) = sigma_0 + R T Gamma_inf ln(1 - Gamma / Gamma_inf)`` and the
    single-interface Gibbs elasticity ``E_s = -Gamma dsigma/dGamma =
    R T Gamma_inf Gamma / (Gamma_inf - Gamma)``.
    """

    clean_surface_tension_n_m: Array = parameter_field()
    temperature_k: Array = parameter_field()
    maximum_surface_concentration_mol_m2: Array = parameter_field()

    def __init__(
        self,
        clean_surface_tension_n_m: ArrayLike,
        temperature_k: ArrayLike,
        maximum_surface_concentration_mol_m2: ArrayLike,
        /,
    ) -> None:
        tension = _host_scalar(
            clean_surface_tension_n_m, "clean_surface_tension_n_m", positive=True
        )
        temperature = _host_scalar(temperature_k, "temperature_k", positive=True)
        capacity = _host_scalar(
            maximum_surface_concentration_mol_m2,
            "maximum_surface_concentration_mol_m2",
            positive=True,
        )
        self.clean_surface_tension_n_m = tension
        self.temperature_k = temperature
        self.maximum_surface_concentration_mol_m2 = capacity

    @property
    def surface_pressure_scale_n_m(self) -> Array:
        """Return ``R T Gamma_inf``."""
        return (
            _GAS_CONSTANT_J_MOL_K
            * self.temperature_k
            * self.maximum_surface_concentration_mol_m2
        )

    def evaluate(
        self, surface_concentration_mol_m2: ArrayLike, /
    ) -> SurfactantStateEvaluation:
        """Return tension, ``dsigma/dGamma`` and ``E_s`` without raising."""
        concentration = jnp.asarray(surface_concentration_mol_m2)
        capacity = self.maximum_surface_concentration_mol_m2
        inside = (concentration >= 0) & (concentration < capacity)
        remaining = jnp.where(inside, capacity - concentration, capacity)
        scale = self.surface_pressure_scale_n_m
        tension = self.clean_surface_tension_n_m + scale * jnp.log(remaining / capacity)
        derivative = -scale / remaining
        elasticity = -concentration * derivative
        admissible = (
            inside & jnp.isfinite(concentration) & jnp.isfinite(tension) & (tension > 0)
        )
        return SurfactantStateEvaluation(tension, derivative, elasticity, admissible)

    def _checked(
        self, surface_concentration_mol_m2: ArrayLike, /
    ) -> SurfactantStateEvaluation:
        evaluation = self.evaluate(surface_concentration_mol_m2)
        admissible = eqx.error_if(
            evaluation.admissible,
            ~jnp.all(evaluation.admissible),
            "Surface concentration must be finite, nonnegative, below Langmuir "
            "capacity, and give positive tension.",
        )
        return SurfactantStateEvaluation(
            evaluation.surface_tension_n_m,
            evaluation.tension_derivative,
            evaluation.gibbs_elasticity_n_m,
            admissible,
        )

    def surface_tension(self, surface_concentration_mol_m2: ArrayLike, /) -> Array:
        """Return ``sigma(Gamma)``; refuses states at or beyond capacity."""
        return self._checked(surface_concentration_mol_m2).surface_tension_n_m

    def tension_derivative(self, surface_concentration_mol_m2: ArrayLike, /) -> Array:
        """Return ``dsigma/dGamma``; refuses states at or beyond capacity."""
        return self._checked(surface_concentration_mol_m2).tension_derivative

    def gibbs_elasticity(self, surface_concentration_mol_m2: ArrayLike, /) -> Array:
        """Return single-interface ``E_s = -Gamma dsigma/dGamma`` in N/m."""
        return self._checked(surface_concentration_mol_m2).gibbs_elasticity_n_m

    def free_energy_density(self, surface_concentration_mol_m2: ArrayLike, /) -> Array:
        """Return the unchecked interfacial free energy per area ``f(Gamma)``.

        ``f = sigma_0 + R T Gamma_inf [theta ln theta + (1 - theta) ln(1 - theta)]``
        with ``theta = Gamma / Gamma_inf`` satisfies ``f - Gamma f' = sigma`` and
        is convex on the admissible range.
        """
        coverage = (
            jnp.asarray(surface_concentration_mol_m2)
            / self.maximum_surface_concentration_mol_m2
        )
        mixing = xlogy(coverage, coverage) + xlogy(1.0 - coverage, 1.0 - coverage)
        return self.clean_surface_tension_n_m + self.surface_pressure_scale_n_m * mixing


class AdsorptionKinetics(StrictModule, ParameterOwner):
    """Langmuir adsorption flux ``j = k_a c (1 - Gamma/Gamma_inf) - k_d Gamma``.

    ``j`` (mol m^-2 s^-1) is positive from the adjacent bulk onto one
    interface. Its equilibrium is the Langmuir isotherm with
    ``K = k_a / (k_d Gamma_inf)``.
    """

    adsorption_rate_m_s: Array = parameter_field()
    desorption_rate_s_inv: Array = parameter_field()
    maximum_surface_concentration_mol_m2: Array = parameter_field()

    def __init__(
        self,
        adsorption_rate_m_s: ArrayLike,
        desorption_rate_s_inv: ArrayLike,
        maximum_surface_concentration_mol_m2: ArrayLike,
        /,
    ) -> None:
        adsorption = _host_scalar(
            adsorption_rate_m_s, "adsorption_rate_m_s", positive=False
        )
        desorption = _host_scalar(
            desorption_rate_s_inv, "desorption_rate_s_inv", positive=False
        )
        capacity = _host_scalar(
            maximum_surface_concentration_mol_m2,
            "maximum_surface_concentration_mol_m2",
            positive=True,
        )
        self.adsorption_rate_m_s = adsorption
        self.desorption_rate_s_inv = desorption
        self.maximum_surface_concentration_mol_m2 = capacity

    def flux(
        self,
        bulk_concentration_mol_m3: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        /,
    ) -> Array:
        """Return the unchecked adsorption flux for solver residuals."""
        bulk = jnp.asarray(bulk_concentration_mol_m3)
        surface = jnp.asarray(surface_concentration_mol_m2)
        return (
            self.adsorption_rate_m_s
            * bulk
            * (1.0 - surface / self.maximum_surface_concentration_mol_m2)
            - self.desorption_rate_s_inv * surface
        )

    def rate(
        self,
        bulk_concentration_mol_m3: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        /,
    ) -> Array:
        """Return the flux; refuses nonfinite, negative or over-capacity states."""
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
        return self.flux(bulk, surface)


class CoxVoinovWettingLaw(StrictModule, ParameterOwner):
    """Cox--Voinov dynamic angle ``theta^3 = theta_e^3 + 9 Ca ln(L / l)``."""

    equilibrium_angle_rad: Array = parameter_field()
    microscopic_length_m: Array = parameter_field()
    macroscopic_length_m: Array = parameter_field()

    def __init__(
        self,
        equilibrium_angle_rad: ArrayLike,
        microscopic_length_m: ArrayLike,
        macroscopic_length_m: ArrayLike,
        /,
    ) -> None:
        angle = _host_scalar(
            equilibrium_angle_rad, "equilibrium_angle_rad", positive=True
        )
        microscopic = _host_scalar(
            microscopic_length_m, "microscopic_length_m", positive=True
        )
        macroscopic = _host_scalar(
            macroscopic_length_m, "macroscopic_length_m", positive=True
        )
        if float(angle) >= pi or float(macroscopic) <= float(microscopic):
            raise ValueError("Cox-Voinov wetting parameters are outside physical bounds.")
        self.equilibrium_angle_rad = angle
        self.microscopic_length_m = microscopic
        self.macroscopic_length_m = macroscopic

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


def interfacial_transport_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("interfacial-transport.langmuir-surfactant", {"state": "surface-concentration"}),
        ("interfacial-transport.adsorption", {"kinetics": "langmuir"}),
        ("interfacial-transport.dynamic-wetting", {"law": "cox-voinov"}),
        (
            "interfacial-transport.surface-lubrication",
            {"geometry": "fixed-manifold", "state": "liquid-volume"},
        ),
        (
            "interfacial-transport.symmetric-film-surfactant",
            {"geometry": "fixed-manifold", "leaflets": "symmetric"},
        ),
        (
            "interfacial-transport.surface-plug-flow",
            {"regime": "plug-flow", "leaflets": "symmetric"},
        ),
        (
            "interfacial-transport.moving-surface",
            {"motion": "fixed-topology", "transport": "relative-velocity"},
        ),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
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
    "SurfactantStateEvaluation",
    "interfacial_transport_candidate_profiles",
]
