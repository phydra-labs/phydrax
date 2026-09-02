#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._transport_closures import (
    AbstractTransportClosure,
    TransportProperties,
)


class TurbulenceArguments(StrictModule):
    velocity_gradient: Array
    wall_distance: Array
    filter_width: Array
    model_state: Array


class AbstractEddyViscosityPlan(StrictModule, NonTrainableState):
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def kinematic_viscosity(
        self, state: Array, arguments: TurbulenceArguments, /
    ) -> Array:
        raise NotImplementedError


class WALEPlan(AbstractEddyViscosityPlan):
    coefficient: float = eqx.field(static=True)

    def __init__(self, coefficient: float = 0.325, /):
        coefficient_ = float(coefficient)
        if not math.isfinite(coefficient_) or coefficient_ < 0.0:
            raise ValueError("WALE coefficient must be finite and nonnegative.")
        self.coefficient = coefficient_
        self.plan_id = canonical_fingerprint(
            {"kind": "wale-les-plan", "coefficient": coefficient_}
        )

    def kinematic_viscosity(
        self, state: Array, arguments: TurbulenceArguments, /
    ) -> Array:
        del state
        gradient = jnp.asarray(arguments.velocity_gradient)
        symmetric = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        squared = oe.contract("...ik,...kj->...ij", gradient, gradient, backend="jax")
        squared_symmetric = 0.5 * (squared + jnp.swapaxes(squared, -1, -2))
        dimension = gradient.shape[-1]
        trace = jnp.trace(squared_symmetric, axis1=-2, axis2=-1)
        deviatoric = (
            squared_symmetric
            - trace[..., None, None]
            * jnp.eye(dimension, dtype=gradient.dtype)
            / dimension
        )
        strain_norm = oe.contract("...ij,...ij->...", symmetric, symmetric, backend="jax")
        deviatoric_norm = oe.contract(
            "...ij,...ij->...", deviatoric, deviatoric, backend="jax"
        )
        numerator = deviatoric_norm**1.5
        denominator = strain_norm**2.5 + deviatoric_norm**1.25
        return (
            (self.coefficient * arguments.filter_width) ** 2
            * numerator
            / jnp.maximum(denominator, 1.0e-30)
        )


class VremanPlan(AbstractEddyViscosityPlan):
    coefficient: float = eqx.field(static=True)

    def __init__(self, coefficient: float = 0.07, /):
        coefficient_ = float(coefficient)
        if not math.isfinite(coefficient_) or coefficient_ < 0.0:
            raise ValueError("Vreman coefficient must be finite and nonnegative.")
        self.coefficient = coefficient_
        self.plan_id = canonical_fingerprint(
            {"kind": "vreman-les-plan", "coefficient": coefficient_}
        )

    def kinematic_viscosity(
        self, state: Array, arguments: TurbulenceArguments, /
    ) -> Array:
        del state
        gradient = jnp.asarray(arguments.velocity_gradient)
        beta = oe.contract("...ki,...kj->...ij", gradient, gradient, backend="jax")
        alpha = oe.contract("...ij,...ij->...", gradient, gradient, backend="jax")
        dimension = gradient.shape[-1]
        invariant = jnp.zeros_like(alpha)
        for first in range(dimension):
            for second in range(first + 1, dimension):
                invariant = invariant + (
                    beta[..., first, first] * beta[..., second, second]
                    - beta[..., first, second] ** 2
                )
        return (
            self.coefficient
            * arguments.filter_width**2
            * jnp.sqrt(jnp.maximum(invariant, 0.0) / jnp.maximum(alpha, 1.0e-30))
        )


class SpalartAllmarasPlan(AbstractEddyViscosityPlan):
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
        self, state: Array, arguments: TurbulenceArguments, /
    ) -> Array:
        del state
        nu_tilde = jnp.maximum(arguments.model_state[..., 0], 0.0)
        chi = nu_tilde / jnp.maximum(arguments.model_state[..., 1], 1.0e-30)
        fv1 = chi**3 / (chi**3 + 7.1**3)
        return nu_tilde * fv1

    def source(self, arguments: TurbulenceArguments, /) -> Array:
        gradient = arguments.velocity_gradient
        rotation = 0.5 * (gradient - jnp.swapaxes(gradient, -1, -2))
        vorticity = jnp.sqrt(
            2.0 * oe.contract("...ij,...ij->...", rotation, rotation, backend="jax")
        )
        nu_tilde = jnp.maximum(arguments.model_state[..., 0], 0.0)
        production = self.cb1 * vorticity * nu_tilde
        destruction = (
            self.cw1 * (nu_tilde / jnp.maximum(arguments.wall_distance, 1.0e-12)) ** 2
        )
        return production - destruction


class KOmegaSSTPlan(AbstractEddyViscosityPlan):
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
        self, state: Array, arguments: TurbulenceArguments, /
    ) -> Array:
        del state
        kinetic = jnp.maximum(arguments.model_state[..., 0], 0.0)
        omega = jnp.maximum(arguments.model_state[..., 1], 1.0e-12)
        strain = 0.5 * (
            arguments.velocity_gradient
            + jnp.swapaxes(arguments.velocity_gradient, -1, -2)
        )
        strain_norm = jnp.sqrt(
            2.0 * oe.contract("...ij,...ij->...", strain, strain, backend="jax")
        )
        return self.a1 * kinetic / jnp.maximum(self.a1 * omega, strain_norm)

    def source(self, arguments: TurbulenceArguments, /) -> Array:
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
    turbulence: AbstractEddyViscosityPlan
    specific_heat_cp: float = eqx.field(static=True)
    turbulent_prandtl: float = eqx.field(static=True)

    def __init__(
        self,
        molecular: AbstractTransportClosure,
        turbulence: AbstractEddyViscosityPlan,
        /,
        *,
        specific_heat_cp: float,
        turbulent_prandtl: float = 0.9,
    ):
        if not isinstance(molecular, AbstractTransportClosure) or not isinstance(
            turbulence, AbstractEddyViscosityPlan
        ):
            raise TypeError(
                "Turbulent transport requires molecular and turbulence plans."
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
        if not isinstance(args, TurbulenceArguments):
            raise TypeError("Turbulent transport requires TurbulenceArguments.")
        molecular = self.molecular.properties(temperature, state, args)
        density = state[..., 0]
        turbulent_dynamic = density * self.turbulence.kinematic_viscosity(state, args)
        return TransportProperties(
            molecular.dynamic_viscosity + turbulent_dynamic,
            molecular.bulk_viscosity,
            molecular.thermal_conductivity
            + turbulent_dynamic * self.specific_heat_cp / self.turbulent_prandtl,
        )


class EquilibriumWallModelEvidence(StrictModule, NonTrainableState):
    friction_velocity: Array
    shear_stress: Array
    residual: Array
    converged: Array
    evidence_id: str = eqx.field(static=True)


class EquilibriumWallModel(StrictModule, NonTrainableState):
    kappa: float = eqx.field(static=True)
    additive_constant: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        kappa: float = 0.41,
        additive_constant: float = 5.2,
        iterations: int = 12,
    ):
        self.kappa = float(kappa)
        self.additive_constant = float(additive_constant)
        self.iterations = int(iterations)
        if self.kappa <= 0.0 or self.iterations <= 0:
            raise ValueError("Equilibrium wall model controls are invalid.")
        self.model_id = canonical_fingerprint(
            {
                "kind": "equilibrium-wall-model",
                "kappa": self.kappa,
                "additive_constant": self.additive_constant,
                "iterations": self.iterations,
            }
        )

    def evaluate(
        self,
        tangential_speed: ArrayLike,
        wall_distance: ArrayLike,
        density: ArrayLike,
        kinematic_viscosity: ArrayLike,
        /,
    ) -> EquilibriumWallModelEvidence:
        speed = jnp.abs(jnp.asarray(tangential_speed))
        distance = jnp.asarray(wall_distance)
        density_ = jnp.asarray(density)
        viscosity = jnp.asarray(kinematic_viscosity)
        friction = jnp.sqrt(
            jnp.maximum(speed * viscosity / jnp.maximum(distance, 1.0e-12), 1.0e-20)
        )
        for _iteration in range(self.iterations):
            argument = jnp.maximum(distance * friction / viscosity, 1.0 + 1.0e-12)
            residual = (
                speed / friction - jnp.log(argument) / self.kappa - self.additive_constant
            )
            derivative = -speed / friction**2 - 1.0 / (self.kappa * friction)
            friction = jnp.maximum(friction - residual / derivative, 1.0e-12)
        argument = jnp.maximum(distance * friction / viscosity, 1.0 + 1.0e-12)
        residual = (
            speed / friction - jnp.log(argument) / self.kappa - self.additive_constant
        )
        shear = density_ * friction**2
        return EquilibriumWallModelEvidence(
            friction,
            shear,
            residual,
            jnp.abs(residual) <= 1.0e-8,
            self.model_id,
        )


class SyntheticTurbulenceInflowPlan(StrictModule, NonTrainableState):
    wavevectors: Array
    amplitudes: Array
    phases: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavevectors: ArrayLike,
        amplitudes: ArrayLike,
        phases: ArrayLike,
        /,
    ):
        waves = np.asarray(wavevectors, dtype=float)
        amplitudes_ = np.asarray(amplitudes, dtype=float)
        phases_ = np.asarray(phases, dtype=float)
        if (
            waves.ndim != 2
            or amplitudes_.shape != waves.shape
            or phases_.shape != (waves.shape[0],)
            or np.any(~np.isfinite(waves))
        ):
            raise ValueError("Synthetic turbulence modes are incompatible.")
        projection = (
            amplitudes_
            - (
                np.sum(amplitudes_ * waves, axis=-1, keepdims=True)
                / np.maximum(np.sum(waves * waves, axis=-1, keepdims=True), 1.0e-30)
            )
            * waves
        )
        self.wavevectors = jnp.asarray(waves)
        self.amplitudes = jnp.asarray(projection)
        self.phases = jnp.asarray(phases_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "synthetic-turbulence-inflow",
                "wavevectors": array_tree_fingerprint(waves),
                "amplitudes": array_tree_fingerprint(projection),
                "phases": array_tree_fingerprint(phases_),
            }
        )

    def velocity(self, coordinates: ArrayLike, time: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        phase = oe.contract("...d,md->...m", points, self.wavevectors, backend="jax")
        phase = phase + self.phases + jnp.asarray(time)
        return oe.contract(
            "...m,md->...d", jnp.cos(phase), self.amplitudes, backend="jax"
        )


__all__ = [
    "AbstractEddyViscosityPlan",
    "EquilibriumWallModel",
    "EquilibriumWallModelEvidence",
    "KOmegaSSTPlan",
    "SpalartAllmarasPlan",
    "SyntheticTurbulenceInflowPlan",
    "TurbulenceArguments",
    "TurbulentTransportClosure",
    "VremanPlan",
    "WALEPlan",
]
