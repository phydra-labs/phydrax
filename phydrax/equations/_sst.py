#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class SSTEvaluation(StrictModule):
    eddy_viscosity: Array
    blending_f1: Array
    blending_f2: Array
    turbulent_kinetic_energy_source: Array
    specific_dissipation_source: Array
    turbulent_kinetic_energy_diffusivity: Array
    specific_dissipation_diffusivity: Array
    production: Array
    cross_diffusion: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SSTTurbulencePlan(StrictModule, NonTrainableState):
    """Named NASA-TMR SST-m/v variants with complete k-omega source closure."""

    variant: Literal["sst-1994-m", "sst-2003-m", "sst-2003-v"] = eqx.field(static=True)
    beta_star: float = eqx.field(static=True)
    a1: float = eqx.field(static=True)
    turbulent_prandtl: float = eqx.field(static=True)
    turbulent_schmidt: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        variant: Literal["sst-1994-m", "sst-2003-m", "sst-2003-v"],
        /,
        *,
        turbulent_prandtl: float = 0.9,
        turbulent_schmidt: float = 0.9,
    ):
        prandtl = float(turbulent_prandtl)
        schmidt = float(turbulent_schmidt)
        if (
            variant not in ("sst-1994-m", "sst-2003-m", "sst-2003-v")
            or not np.isfinite(prandtl)
            or prandtl <= 0.0
            or not np.isfinite(schmidt)
            or schmidt <= 0.0
        ):
            raise ValueError("SST variant or turbulent transport numbers are invalid.")
        self.variant = variant
        self.beta_star = 0.09
        self.a1 = 0.31
        self.turbulent_prandtl = prandtl
        self.turbulent_schmidt = schmidt
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sst-turbulence",
                "variant": variant,
                "beta_star": self.beta_star,
                "a1": self.a1,
                "turbulent_prandtl": prandtl,
                "turbulent_schmidt": schmidt,
            }
        )

    def evaluate(
        self,
        density: ArrayLike,
        molecular_viscosity: ArrayLike,
        turbulent_kinetic_energy: ArrayLike,
        specific_dissipation_rate: ArrayLike,
        velocity_gradient: ArrayLike,
        kinetic_energy_gradient: ArrayLike,
        dissipation_gradient: ArrayLike,
        wall_distance: ArrayLike,
        /,
    ) -> SSTEvaluation:
        density_ = jnp.asarray(density)
        viscosity = jnp.asarray(molecular_viscosity, dtype=density_.dtype)
        kinetic = jnp.asarray(turbulent_kinetic_energy, dtype=density_.dtype)
        omega = jnp.asarray(specific_dissipation_rate, dtype=density_.dtype)
        gradient = jnp.asarray(velocity_gradient, dtype=density_.dtype)
        kinetic_gradient = jnp.asarray(kinetic_energy_gradient, dtype=density_.dtype)
        omega_gradient = jnp.asarray(dissipation_gradient, dtype=density_.dtype)
        distance = jnp.asarray(wall_distance, dtype=density_.dtype)
        dimension = gradient.shape[-1]
        shape = density_.shape
        if (
            viscosity.shape != shape
            or kinetic.shape != shape
            or omega.shape != shape
            or distance.shape not in ((), shape)
            or gradient.shape != shape + (dimension, dimension)
            or kinetic_gradient.shape != shape + (dimension,)
            or omega_gradient.shape != shape + (dimension,)
        ):
            raise ValueError("SST state, gradient, or wall-distance shapes are invalid.")
        distance = jnp.broadcast_to(distance, shape)
        tiny = jnp.finfo(density_.dtype).tiny
        nu = viscosity / jnp.maximum(density_, tiny)
        cross_raw = (
            2.0
            * density_
            * 0.856
            / jnp.maximum(omega, tiny)
            * contract("...i,...i->...", kinetic_gradient, omega_gradient, backend="jax")
        )
        cross_denominator = jnp.maximum(cross_raw, 1.0e-20)
        first_argument = jnp.minimum(
            jnp.maximum(
                jnp.sqrt(jnp.maximum(kinetic, 0.0))
                / jnp.maximum(self.beta_star * omega * distance, tiny),
                500.0 * nu / jnp.maximum(distance * distance * omega, tiny),
            ),
            4.0
            * density_
            * 0.856
            * kinetic
            / jnp.maximum(cross_denominator * distance * distance, tiny),
        )
        f1 = jnp.tanh(first_argument**4)
        second_argument = jnp.maximum(
            2.0
            * jnp.sqrt(jnp.maximum(kinetic, 0.0))
            / jnp.maximum(self.beta_star * omega * distance, tiny),
            500.0 * nu / jnp.maximum(distance * distance * omega, tiny),
        )
        f2 = jnp.tanh(second_argument**2)
        strain_tensor = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        strain_magnitude = jnp.sqrt(
            jnp.maximum(2.0 * jnp.sum(strain_tensor * strain_tensor, axis=(-2, -1)), 0.0)
        )
        vorticity_tensor = 0.5 * (gradient - jnp.swapaxes(gradient, -1, -2))
        vorticity_magnitude = jnp.sqrt(
            jnp.maximum(
                2.0 * jnp.sum(vorticity_tensor * vorticity_tensor, axis=(-2, -1)), 0.0
            )
        )
        limiter_measure = (
            vorticity_magnitude if self.variant == "sst-2003-v" else strain_magnitude
        )
        eddy = (
            density_
            * self.a1
            * kinetic
            / jnp.maximum(self.a1 * omega, limiter_measure * f2)
        )
        raw_production = eddy * strain_magnitude**2
        production_limit = (
            (10.0 if self.variant == "sst-1994-m" else 20.0)
            * self.beta_star
            * density_
            * kinetic
            * omega
        )
        production = jnp.minimum(raw_production, production_limit)
        beta_1, beta_2 = 0.075, 0.0828
        gamma_1, gamma_2 = 5.0 / 9.0, 0.44
        sigma_k_1, sigma_k_2 = 0.85, 1.0
        sigma_w_1, sigma_w_2 = 0.5, 0.856
        beta = f1 * beta_1 + (1.0 - f1) * beta_2
        gamma = f1 * gamma_1 + (1.0 - f1) * gamma_2
        sigma_k = f1 * sigma_k_1 + (1.0 - f1) * sigma_k_2
        sigma_w = f1 * sigma_w_1 + (1.0 - f1) * sigma_w_2
        cross_diffusion = (1.0 - f1) * cross_raw
        kinetic_source = production - self.beta_star * density_ * kinetic * omega
        omega_source = (
            gamma * density_ * strain_magnitude**2
            - beta * density_ * omega**2
            + cross_diffusion
        )
        kinetic_diffusivity = viscosity + sigma_k * eddy
        omega_diffusivity = viscosity + sigma_w * eddy
        finite = (
            jnp.isfinite(eddy)
            & jnp.isfinite(kinetic_source)
            & jnp.isfinite(omega_source)
            & jnp.isfinite(kinetic_diffusivity)
            & jnp.isfinite(omega_diffusivity)
        )
        successful = (
            finite
            & (density_ > 0.0)
            & (viscosity > 0.0)
            & (kinetic >= 0.0)
            & (omega > 0.0)
            & (distance > 0.0)
            & (eddy >= 0.0)
            & (kinetic_diffusivity > 0.0)
            & (omega_diffusivity > 0.0)
        )
        return SSTEvaluation(
            eddy,
            f1,
            f2,
            kinetic_source,
            omega_source,
            kinetic_diffusivity,
            omega_diffusivity,
            production,
            cross_diffusion,
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["SSTEvaluation", "SSTTurbulencePlan"]
