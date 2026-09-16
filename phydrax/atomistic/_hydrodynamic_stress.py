#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


HydrodynamicStressConvention = Literal["tension-positive-cauchy"]


class HydrodynamicStressPlan(StrictModule):
    dynamic_viscosity: float = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    require_stresslets: bool = eqx.field(static=True)
    require_brownian_extra: bool = eqx.field(static=True)
    stress_symmetry_tolerance: float = eqx.field(static=True)
    incompressibility_tolerance: float = eqx.field(static=True)
    convention: HydrodynamicStressConvention = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamic_viscosity: float,
        volume: float,
        temperature: float,
        /,
        *,
        boltzmann_constant: float = 1.0,
        require_stresslets: bool = False,
        require_brownian_extra: bool = False,
        stress_symmetry_tolerance: float = 1.0e-10,
        incompressibility_tolerance: float = 1.0e-10,
    ):
        values = (dynamic_viscosity, volume, boltzmann_constant)
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in values
        ):
            raise ValueError(
                "Viscosity, volume, and Boltzmann constant must be positive."
            )
        if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
            raise ValueError("temperature must be finite and nonnegative.")
        symmetry_tolerance = float(stress_symmetry_tolerance)
        incompressibility_tolerance_ = float(incompressibility_tolerance)
        if (
            not math.isfinite(symmetry_tolerance)
            or symmetry_tolerance < 0.0
            or not math.isfinite(incompressibility_tolerance_)
            or incompressibility_tolerance_ < 0.0
        ):
            raise ValueError(
                "Hydrodynamic stress tolerances must be finite and nonnegative."
            )
        self.dynamic_viscosity = float(dynamic_viscosity)
        self.volume = float(volume)
        self.temperature = float(temperature)
        self.boltzmann_constant = float(boltzmann_constant)
        self.require_stresslets = bool(require_stresslets)
        self.require_brownian_extra = bool(require_brownian_extra)
        self.stress_symmetry_tolerance = symmetry_tolerance
        self.incompressibility_tolerance = incompressibility_tolerance_
        self.convention = "tension-positive-cauchy"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydrodynamic-stress",
                "viscosity": self.dynamic_viscosity,
                "volume": self.volume,
                "temperature": self.temperature,
                "boltzmann": self.boltzmann_constant,
                "require_stresslets": self.require_stresslets,
                "require_brownian_extra": self.require_brownian_extra,
                "stress_symmetry_tolerance": symmetry_tolerance,
                "incompressibility_tolerance": incompressibility_tolerance_,
                "convention": self.convention,
            }
        )


class HydrodynamicStressResult(StrictModule):
    solvent_cauchy_stress: Array
    stresslet_cauchy_stress: Array
    ideal_brownian_cauchy_stress: Array
    extra_brownian_cauchy_stress: Array
    total_cauchy_stress: Array
    pressure: Array
    flow_power: Array
    stresslet_available: Array
    brownian_extra_available: Array
    symmetry_residual: Array
    incompressibility_residual: Array
    finite: Array
    successful: Array
    convention: HydrodynamicStressConvention = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def hydrodynamic_stress(
    plan: HydrodynamicStressPlan,
    rate_of_strain: ArrayLike,
    active_particle_count: int,
    /,
    *,
    particle_stresslets: ArrayLike | None = None,
    brownian_extra_cauchy_stress: ArrayLike | None = None,
    stresslet_convention: HydrodynamicStressConvention = "tension-positive-cauchy",
) -> HydrodynamicStressResult:
    if not isinstance(plan, HydrodynamicStressPlan):
        raise TypeError("plan must be HydrodynamicStressPlan.")
    if stresslet_convention != plan.convention:
        raise ValueError("Hydrodynamic stresslets must use tension-positive Cauchy sign.")
    rate = jnp.asarray(rate_of_strain)
    if rate.shape != (3, 3):
        raise ValueError("rate_of_strain must have shape (3, 3).")
    count = int(active_particle_count)
    if count < 0:
        raise ValueError("active_particle_count must be nonnegative.")
    solvent = 2.0 * plan.dynamic_viscosity * rate
    stresslet_available = particle_stresslets is not None
    if particle_stresslets is None:
        stresslet = jnp.zeros((3, 3), dtype=rate.dtype)
    else:
        supplied = jnp.asarray(particle_stresslets, dtype=rate.dtype)
        if supplied.ndim != 3 or supplied.shape[1:] != (3, 3):
            raise ValueError("particle_stresslets must have shape (particle, 3, 3).")
        if supplied.shape[0] != count:
            raise ValueError("particle_stresslets must match active_particle_count.")
        stresslet = jnp.sum(supplied, axis=0) / plan.volume
    ideal = -(count * plan.boltzmann_constant * plan.temperature / plan.volume) * jnp.eye(
        3, dtype=rate.dtype
    )
    brownian_available = brownian_extra_cauchy_stress is not None
    if brownian_extra_cauchy_stress is None:
        extra = jnp.zeros((3, 3), dtype=rate.dtype)
    else:
        extra = jnp.asarray(brownian_extra_cauchy_stress, dtype=rate.dtype)
        if extra.shape != (3, 3):
            raise ValueError("brownian_extra_cauchy_stress must have shape (3, 3).")
    total = solvent + stresslet + ideal + extra
    pressure = -jnp.trace(total) / 3.0
    power = plan.volume * jnp.sum(total * rate)
    symmetry = jnp.max(jnp.abs(total - total.T))
    incompressibility = jnp.abs(jnp.trace(rate))
    finite = jnp.all(jnp.isfinite(total)) & jnp.isfinite(power) & jnp.isfinite(pressure)
    required = (not plan.require_stresslets or stresslet_available) and (
        not plan.require_brownian_extra or brownian_available
    )
    successful = (
        finite
        & required
        & (symmetry <= plan.stress_symmetry_tolerance)
        & (incompressibility <= plan.incompressibility_tolerance)
    )
    return HydrodynamicStressResult(
        solvent,
        stresslet,
        ideal,
        extra,
        total,
        pressure,
        power,
        jnp.asarray(stresslet_available),
        jnp.asarray(brownian_available),
        symmetry,
        incompressibility,
        finite,
        successful,
        plan.convention,
        plan.plan_id,
    )


__all__ = [
    "HydrodynamicStressConvention",
    "HydrodynamicStressPlan",
    "HydrodynamicStressResult",
    "hydrodynamic_stress",
]
