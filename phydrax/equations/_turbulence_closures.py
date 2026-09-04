#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._transport_closures import (
    AbstractTransportClosure,
    TransportProperties,
)


class RANSTurbulenceArguments(StrictModule):
    """Velocity-gradient, wall-distance, and model-state inputs for RANS models."""

    velocity_gradient: Array
    wall_distance: Array
    model_state: Array


class AbstractRANSEddyViscosityPlan(StrictModule, NonTrainableState):
    """Base interface for transport-equation RANS eddy-viscosity plans."""

    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def kinematic_viscosity(
        self, state: Array, arguments: RANSTurbulenceArguments, /
    ) -> Array:
        raise NotImplementedError


class SpalartAllmarasPlan(AbstractRANSEddyViscosityPlan):
    cb1: float = eqx.field(static=True)
    cw1: float = eqx.field(static=True)

    def __init__(self, cb1: float = 0.1355, cw1: float = 3.239, /):
        self.cb1 = float(cb1)
        self.cw1 = float(cw1)
        if self.cb1 <= 0.0 or self.cw1 <= 0.0:
            raise ValueError("Spalart-Allmaras constants must be positive.")
        self.plan_id = canonical_fingerprint(
            {"kind": "spalart-allmaras-plan", "cb1": self.cb1, "cw1": self.cw1}
        )

    def kinematic_viscosity(
        self, state: Array, arguments: RANSTurbulenceArguments, /
    ) -> Array:
        del state
        nu_tilde = jnp.maximum(arguments.model_state[..., 0], 0.0)
        chi = nu_tilde / jnp.maximum(arguments.model_state[..., 1], 1.0e-30)
        fv1 = chi**3 / (chi**3 + 7.1**3)
        return nu_tilde * fv1

    def source(self, arguments: RANSTurbulenceArguments, /) -> Array:
        gradient = arguments.velocity_gradient
        rotation = 0.5 * (gradient - jnp.swapaxes(gradient, -1, -2))
        vorticity = jnp.sqrt(
            2.0 * ein.contract("...ij,...ij->...", rotation, rotation, backend="jax")
        )
        nu_tilde = jnp.maximum(arguments.model_state[..., 0], 0.0)
        production = self.cb1 * vorticity * nu_tilde
        destruction = (
            self.cw1 * (nu_tilde / jnp.maximum(arguments.wall_distance, 1.0e-12)) ** 2
        )
        return production - destruction


class KOmegaSSTPlan(AbstractRANSEddyViscosityPlan):
    a1: float = eqx.field(static=True)
    beta_star: float = eqx.field(static=True)

    def __init__(self, a1: float = 0.31, beta_star: float = 0.09, /):
        self.a1 = float(a1)
        self.beta_star = float(beta_star)
        if self.a1 <= 0.0 or self.beta_star <= 0.0:
            raise ValueError("k-omega SST constants must be positive.")
        self.plan_id = canonical_fingerprint(
            {"kind": "k-omega-sst-plan", "a1": self.a1, "beta_star": self.beta_star}
        )

    def kinematic_viscosity(
        self, state: Array, arguments: RANSTurbulenceArguments, /
    ) -> Array:
        del state
        kinetic = jnp.maximum(arguments.model_state[..., 0], 0.0)
        omega = jnp.maximum(arguments.model_state[..., 1], 1.0e-12)
        strain = 0.5 * (
            arguments.velocity_gradient
            + jnp.swapaxes(arguments.velocity_gradient, -1, -2)
        )
        strain_norm = jnp.sqrt(
            2.0 * ein.contract("...ij,...ij->...", strain, strain, backend="jax")
        )
        return self.a1 * kinetic / jnp.maximum(self.a1 * omega, strain_norm)

    def source(self, arguments: RANSTurbulenceArguments, /) -> Array:
        kinetic = jnp.maximum(arguments.model_state[..., 0], 0.0)
        omega = jnp.maximum(arguments.model_state[..., 1], 1.0e-12)
        production = self.kinematic_viscosity(jnp.empty((0,)), arguments) * omega**2
        return jnp.stack(
            (
                production - self.beta_star * kinetic * omega,
                production / jnp.maximum(self.a1 * kinetic, 1.0e-12) - 0.075 * omega**2,
            ),
            axis=-1,
        )


class TurbulentTransportClosure(AbstractTransportClosure):
    molecular: AbstractTransportClosure
    turbulence: AbstractRANSEddyViscosityPlan
    specific_heat_cp: float = eqx.field(static=True)
    turbulent_prandtl: float = eqx.field(static=True)

    def __init__(
        self,
        molecular: AbstractTransportClosure,
        turbulence: AbstractRANSEddyViscosityPlan,
        /,
        *,
        specific_heat_cp: float,
        turbulent_prandtl: float = 0.9,
    ):
        if not isinstance(molecular, AbstractTransportClosure) or not isinstance(
            turbulence, AbstractRANSEddyViscosityPlan
        ):
            raise TypeError(
                "Turbulent transport requires molecular and RANS turbulence plans."
            )
        cp = float(specific_heat_cp)
        prandtl = float(turbulent_prandtl)
        if cp <= 0.0 or prandtl <= 0.0:
            raise ValueError("Turbulent heat-transport constants must be positive.")
        self.molecular = molecular
        self.turbulence = turbulence
        self.specific_heat_cp = cp
        self.turbulent_prandtl = prandtl
        self.closure_id = canonical_fingerprint(
            {
                "kind": "turbulent-transport-closure",
                "molecular": molecular.closure_id,
                "turbulence": turbulence.plan_id,
                "specific_heat_cp": cp,
                "turbulent_prandtl": prandtl,
            }
        )

    def properties(
        self,
        temperature: Array,
        state: Array,
        args: Any = None,
        /,
    ) -> TransportProperties:
        if not isinstance(args, RANSTurbulenceArguments):
            raise TypeError("Turbulent transport requires RANSTurbulenceArguments.")
        molecular = self.molecular.properties(temperature, state, args)
        density = state[..., 0]
        turbulent_dynamic = density * self.turbulence.kinematic_viscosity(state, args)
        return TransportProperties(
            molecular.dynamic_viscosity + turbulent_dynamic,
            molecular.bulk_viscosity,
            molecular.thermal_conductivity
            + turbulent_dynamic * self.specific_heat_cp / self.turbulent_prandtl,
        )


__all__ = [
    "AbstractRANSEddyViscosityPlan",
    "KOmegaSSTPlan",
    "RANSTurbulenceArguments",
    "SpalartAllmarasPlan",
    "TurbulentTransportClosure",
]
